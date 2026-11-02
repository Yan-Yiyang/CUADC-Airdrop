"""MPEG-TS 段写入与按随机访问范围解码。"""

from __future__ import annotations

import json
import logging
import math
import queue
import sys
import threading
import time
from collections.abc import Callable, Generator, Iterable, Iterator
from contextlib import suppress
from fractions import Fraction
from pathlib import Path
from types import GeneratorType
from typing import Any

import numpy as np

from .cache import DecodedFrameCache
from .segments import SegmentLifecycle, SegmentState
from .storage import TsStorageLease

LOGGER = logging.getLogger(__name__)


class TsDataNotReady(EOFError):
    """活动 TS 段的索引或尾部尚未写到目标帧。"""


def _request_stream(
    first: tuple[int, int, int], requests: queue.Queue[tuple[int, int, int]]
) -> Iterator[tuple[int, int, int]]:
    """把逐次读取请求送入一个保持打开的解码生成器。"""
    yield first
    while True:
        yield requests.get()


class TsSegmentWriter:
    """复制输入压缩包到每个连接周期独立的 MPEG-TS 文件。"""

    def __init__(self, directory: Path, *, segment: int, input_stream: Any) -> None:
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.segment = segment
        self.path = directory / f"segment-{segment:06d}.ts"
        self.index_path = directory / f"segment-{segment:06d}.rap.jsonl"
        self.complete_path = directory / f"segment-{segment:06d}.complete"
        existing = [
            path for path in (self.path, self.index_path, self.complete_path) if path.exists()
        ]
        if existing:
            raise FileExistsError(f"拒绝覆盖现有 TS 分段文件: {existing[0]}")
        import av

        self._file = self.path.open("xb")
        self._container: Any = None
        self._index_file: Any = None
        try:
            self._container = av.open(
                self._file,
                mode="w",
                format="mpegts",
                options={
                    "mpegts_copyts": "1",
                    "avoid_negative_ts": "disabled",
                    "flush_packets": "1",
                },
            )
            self._stream = self._container.add_stream_from_template(input_stream)
            self._index_file = self.index_path.open("x", encoding="utf-8", newline="\n")
        except Exception:
            created_paths = [self.path]
            if self._index_file is not None:
                created_paths.append(self.index_path)
            try:
                if self._container is not None:
                    self._container.close()
            finally:
                try:
                    if self._index_file is not None:
                        self._index_file.close()
                finally:
                    self._file.close()
                for path in created_paths:
                    with suppress(FileNotFoundError):
                        path.unlink()
            raise
        assert self._container is not None and self._index_file is not None
        self._closed = False

    def mux(self, packet: Any) -> None:
        """将输入包交给 TS muxer，保留包的时间戳与关键帧标记。"""
        if packet.size <= 0 or (packet.pts is None and packet.dts is None):
            return
        import av

        copied = av.Packet(bytes(packet))
        copied.pts = packet.pts
        copied.dts = packet.dts
        copied.duration = packet.duration
        copied.time_base = packet.time_base
        copied.is_keyframe = packet.is_keyframe
        copied.stream = self._stream
        self._container.mux(copied)
        if packet.is_keyframe and packet.pts is not None and packet.dts is not None:
            time_base = packet.time_base
            if time_base is None:
                raise ValueError("随机访问包缺少 time_base")
            self._index_file.write(
                json.dumps(
                    {
                        "pts": int(packet.pts),
                        "dts": int(packet.dts),
                        "time_base_num": int(time_base.numerator),
                        "time_base_den": int(time_base.denominator),
                    },
                    separators=(",", ":"),
                )
                + "\n"
            )
            self._index_file.flush()
        self._file.flush()

    def close(self, *, complete: bool = True) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._container.close()
        finally:
            try:
                try:
                    if self._index_file is not None:
                        self._index_file.flush()
                finally:
                    if self._index_file is not None:
                        self._index_file.close()
            finally:
                try:
                    self._file.flush()
                finally:
                    self._file.close()
        if complete:
            self.complete_path.touch(exist_ok=False)


class TsRangeDecoder:
    """按目标媒体时间从关键帧开始解码，并缓存一个 GOP 范围。"""

    def __init__(
        self,
        directory: str | Path,
        *,
        max_cache_bytes: int = 64 * 1024 * 1024,
        cursor_idle_timeout_s: float = 30.0,
        segment_lifecycle: SegmentLifecycle | None = None,
        decoded_cache: DecodedFrameCache | None = None,
        cache_enabled: bool = True,
    ) -> None:
        if max_cache_bytes < 1:
            raise ValueError("历史解码缓存上限必须为正")
        if not math.isfinite(cursor_idle_timeout_s) or cursor_idle_timeout_s <= 0:
            raise ValueError("解码游标空闲超时必须为正有限秒数")
        if decoded_cache is not None and not cache_enabled:
            raise ValueError("禁用缓存的解码器不能绑定共享缓存")
        self.directory = Path(directory)
        self.max_cache_bytes = (
            decoded_cache.max_bytes if decoded_cache is not None else max_cache_bytes
        )
        self._decoded_cache = (
            decoded_cache
            if decoded_cache is not None
            else DecodedFrameCache(max_cache_bytes)
            if cache_enabled
            else None
        )
        self._cache_lease_active = self._decoded_cache is not None
        self._cache_lease_lock = threading.RLock()
        self._storage_lease: TsStorageLease | None = None
        self._storage_managed = False
        self._gop_coordinator = None
        if self._decoded_cache is not None:
            self._decoded_cache.acquire()
        self.cursor_idle_timeout_s = cursor_idle_timeout_s
        self.segment_lifecycle = segment_lifecycle
        self._keyframe_cache: dict[int, list[tuple[Fraction, Fraction]]] = {}
        self._keyframe_signatures: dict[int, int] = {}
        self._keyframe_offsets: dict[int, int] = {}
        self._lock = threading.RLock()
        self._lifecycle_lock = threading.RLock()
        # close 在等待生命周期锁之前先发出取消信号，避免等待中的读取阻止退出。
        self._cancel_lock = threading.Lock()
        self._read_cancelled = threading.Event()
        self._closing_count = 0
        self._closed = False
        self._close_incomplete = False
        self._cursor_cleanup_error: BaseException | None = None
        self._iteration_condition = threading.Condition()
        self._active_iterations = 0
        self._iteration_threads: dict[int, int] = {}
        self._cursor: Generator[tuple[tuple[int, int, int], np.ndarray]] | None = None
        self._cursor_requests: queue.Queue[tuple[int, int, int]] | None = None
        self._cursor_segment: int | None = None
        self._cursor_time: Fraction | None = None
        self._cursor_output_size: tuple[int, int] | None = None
        self._cursor_last_used = 0.0
        # 容器关闭失败时保留其对象和目录租约，后续 close 可针对同一对象重试。
        self._open_containers: dict[int, Any] = {}
        self._reaper_stop = threading.Event()
        self._reaper = threading.Thread(
            target=self._reap_idle_cursor,
            args=(self._reaper_stop,),
            name="ts-decoder-cursor-reaper",
            daemon=True,
        )
        try:
            self._reaper.start()
        except BaseException as start_error:
            try:
                self._release_cache_lease()
            except BaseException as cleanup_error:  # noqa: BLE001 - 必须保留 start 原异常
                start_error.add_note(
                    f"释放解码缓存租约时也发生异常：{type(cleanup_error).__name__}: {cleanup_error}"
                )
            raise

    def read(
        self,
        segment: int,
        pts: int,
        num: int,
        den: int,
        *,
        output_size: tuple[int, int] | None = None,
    ) -> np.ndarray | None:
        """返回指定 PTS 的 BGR 图像；同一关键帧范围连续读取复用解码结果。"""
        with self._lifecycle_lock:
            with self._cancel_lock:
                if self._closing_count:
                    raise RuntimeError("历史解码器正在关闭")
                if self._close_incomplete:
                    raise RuntimeError(
                        "历史解码器关闭未完成，必须重试关闭"
                    ) from self._cursor_cleanup_error
                if self._closed:
                    if self._storage_managed and self._storage_lease is None:
                        raise RuntimeError("临时 TS 存储租约已释放，无法重新读取")
                    self._read_cancelled = threading.Event()
                    self._closed = False
            coordinator = self._gop_coordinator
            if coordinator is None:
                return self._read(segment, pts, num, den, output_size=output_size)
            cancelled = self._read_cancelled
        return coordinator.read(self, segment, pts, num, den, output_size, cancelled)

    def enable_shared_gop(self, coordinator=None):
        """使该读取者与其他冻结视图共享按随机访问点组织的解码作业。"""
        from .gop import GopCoordinator

        with self._lifecycle_lock:
            if self._closed or self._close_incomplete:
                raise RuntimeError("历史解码器已关闭或关闭未完成")
            if self._decoded_cache is None:
                raise RuntimeError("未启用缓存的校验解码器不能共享 GOP 作业")
            if self._gop_coordinator is not None:
                if coordinator is not None and coordinator is not self._gop_coordinator:
                    raise RuntimeError("历史解码器已经绑定其他 GOP 协调器")
                return self._gop_coordinator
            with self._lock:
                with self._cancel_lock:
                    if self._close_incomplete:
                        raise RuntimeError(
                            "历史解码器关闭未完成，必须重试关闭"
                        ) from self._cursor_cleanup_error
                self._reset_cursor_locked()
            selected = coordinator or GopCoordinator(self._decoded_cache)
            selected.attach(self)
            self._gop_coordinator = selected
            return selected

    def _read(
        self,
        segment: int,
        pts: int,
        num: int,
        den: int,
        *,
        output_size: tuple[int, int] | None = None,
    ) -> np.ndarray | None:
        target_time = Fraction(pts * num, den)
        width, height = output_size or (0, 0)
        cache = self._decoded_cache
        if cache is not None:
            self._acquire_cache_lease()
        key = self._make_cache_key(segment, target_time, width, height)
        with self._lock:
            with self._cancel_lock:
                if self._close_incomplete:
                    raise RuntimeError(
                        "历史解码器关闭未完成，必须重试关闭"
                    ) from self._cursor_cleanup_error
            self._ensure_reaper_locked()
            if (
                self._cursor is not None
                and time.monotonic() - self._cursor_last_used >= self.cursor_idle_timeout_s
            ):
                self._reset_cursor_locked()
            cached = self._get_cache(key)
            if cached is not None:
                return cached
            if (
                self._cursor is None
                or self._cursor_segment != segment
                or self._cursor_output_size != output_size
                or (self._cursor_time is not None and target_time <= self._cursor_time)
            ):
                self._reset_cursor_locked()
                requests: queue.Queue[tuple[int, int, int]] = queue.Queue()
                iterator = self.iter_range(
                    segment,
                    _request_stream((pts, num, den), requests),
                    output_size=output_size,
                )
                self._cursor = iterator
                self._cursor_requests = requests
                self._cursor_segment = segment
                self._cursor_output_size = output_size
            else:
                assert self._cursor_requests is not None
                self._cursor_requests.put((pts, num, den))
            assert self._cursor is not None
            try:
                returned, image = next(self._cursor)
                if returned != (pts, num, den):
                    raise RuntimeError("解码游标返回了不匹配的帧请求")
            except BaseException as read_error:
                try:
                    self._reset_cursor_locked()
                except BaseException as cleanup_error:  # noqa: BLE001 - 保留原读取异常
                    read_error.add_note(f"解码游标清理失败: {cleanup_error}")
                if self._open_containers:
                    with self._cancel_lock:
                        self._close_incomplete = True
                raise
            self._cursor_time = target_time
            self._cursor_last_used = time.monotonic()
            return image

    def iter_range(
        self,
        segment: int,
        requests: Iterable[tuple[int, int, int]],
        *,
        output_size: tuple[int, int] | None = None,
        on_decoded: Callable[[Fraction, np.ndarray], None] | None = None,
    ) -> Generator[tuple[tuple[int, int, int], np.ndarray]]:
        """跟踪每次迭代，防止另一线程在原生解码运行期间关闭容器。"""
        cancelled = self._read_cancelled
        with self._cancel_lock:
            if self._close_incomplete:
                raise RuntimeError(
                    "历史解码器关闭未完成，必须重试关闭"
                ) from self._cursor_cleanup_error
        inner = self._iter_range_impl(
            segment, requests, output_size=output_size, on_decoded=on_decoded
        )
        try:
            while True:
                with self._iteration_condition:
                    if cancelled.is_set():
                        raise InterruptedError("历史解码读取已取消")
                    self._begin_iteration_locked()
                try:
                    item = next(inner)
                except StopIteration:
                    return
                finally:
                    with self._iteration_condition:
                        self._end_iteration_locked()
                yield item
        finally:
            with self._iteration_condition:
                self._begin_iteration_locked()
            try:
                inner.close()
            finally:
                with self._iteration_condition:
                    self._end_iteration_locked()

    def _begin_iteration_locked(self) -> None:
        thread_id = threading.get_ident()
        self._active_iterations += 1
        self._iteration_threads[thread_id] = self._iteration_threads.get(thread_id, 0) + 1

    def _end_iteration_locked(self) -> None:
        thread_id = threading.get_ident()
        self._active_iterations -= 1
        count = self._iteration_threads[thread_id] - 1
        if count:
            self._iteration_threads[thread_id] = count
        else:
            del self._iteration_threads[thread_id]
        self._iteration_condition.notify_all()

    def _iter_range_impl(  # noqa: PLR0912, PLR0915 - 活动段尾部需要分别保留重试和封口失败语义
        self,
        segment: int,
        requests: Iterable[tuple[int, int, int]],
        *,
        output_size: tuple[int, int] | None = None,
        on_decoded: Callable[[Fraction, np.ndarray], None] | None = None,
    ) -> Generator[tuple[tuple[int, int, int], np.ndarray]]:
        """从一个关键帧连续解码请求序列，缓存受限时保留生成器解码游标。

        请求须按媒体时间递增。生成器暂停时会保留解码器游标；调用方应继续迭代
        或显式关闭生成器，避免占住该解码器实例的串行读取锁。
        ``on_decoded`` 在逐帧解码后同步调用，供未来的共享作业分发；传入的
        数组视图只在回调期间有效且不可写，需要持有图像时由回调方自行复制。
        """
        pending = iter(requests)
        cancelled = self._read_cancelled
        if cancelled.is_set():
            raise InterruptedError("历史解码读取已取消")
        try:
            request = next(pending)
        except StopIteration:
            return
        target_time = Fraction(request[0] * request[1], request[2])
        path = self.directory / f"segment-{segment:06d}.ts"
        if not path.is_file():
            raise FileNotFoundError(f"缺少 MPEG-TS 分段: {path}")
        complete_path = path.with_suffix(".complete")
        lifecycle = self.segment_lifecycle
        status = lifecycle.snapshot(segment) if lifecycle is not None else None
        complete_at_open = complete_path.is_file() or (
            status is not None and status.state is SegmentState.CLOSED_COMPLETE
        )
        active_at_open = not complete_at_open and (
            status is not None and status.state is SegmentState.ACTIVE
        )
        with self._lock:
            keyframes = self._get_keyframes(segment, path)
        random_access = [item for item in keyframes if item[0] <= target_time]
        if not random_access:
            if active_at_open:
                raise TsDataNotReady(f"活动 MPEG-TS 段尚未写出随机访问点: {path}")
            raise ValueError(f"目标时间 {target_time} 之前没有可用关键帧: {path}")
        _, dts_time = random_access[-1]
        import av

        container = av.open(str(path), mode="r")
        with self._lock:
            self._open_containers[id(container)] = container
        try:
            if not container.streams.video:
                raise ValueError(f"MPEG-TS 分段没有视频流: {path}")
            stream = container.streams.video[0]
            offset = int(
                dts_time / Fraction(stream.time_base.numerator, stream.time_base.denominator)
            )
            container.seek(offset, stream=stream, backward=True, any_frame=False)
            current_request = request
            current_time = target_time
            for packet in container.demux(stream):
                if cancelled.is_set():
                    raise InterruptedError("历史解码读取已取消")
                if packet.size > 0 and packet.is_corrupt:
                    if active_at_open:
                        raise TsDataNotReady(
                            f"活动 MPEG-TS 段包暂不可验证: PTS={packet.pts}: {path}"
                        )
                    raise ValueError(f"MPEG-TS 包标记为损坏: PTS={packet.pts}: {path}")
                for frame in packet.decode():
                    if cancelled.is_set():
                        raise InterruptedError("历史解码读取已取消")
                    if frame.is_corrupt:
                        if active_at_open:
                            raise TsDataNotReady(
                                f"活动 MPEG-TS 段帧暂不可验证: PTS={frame.pts}: {path}"
                            )
                        raise ValueError(f"MPEG-TS 解码帧标记为损坏: PTS={frame.pts}: {path}")
                    if frame.pts is None or frame.time_base is None:
                        continue
                    frame_time = Fraction(frame.pts) * Fraction(
                        frame.time_base.numerator, frame.time_base.denominator
                    )
                    if current_time < frame_time:
                        if active_at_open:
                            raise TsDataNotReady(
                                f"活动 MPEG-TS 解码游标尚未读到目标 PTS {current_time}: {path}"
                            )
                        raise ValueError(f"MPEG-TS 中找不到精确 PTS {current_time}: {path}")
                    image = frame.to_ndarray(format="bgr24")
                    if output_size is not None and (image.shape[1], image.shape[0]) != output_size:
                        import cv2

                        image = cv2.resize(image, output_size)
                    width, height = output_size or (0, 0)
                    cache_key = self._make_cache_key(segment, frame_time, width, height)
                    self._put_cache(cache_key, image)
                    if on_decoded is not None:
                        decoded_view = image.view()
                        decoded_view.flags.writeable = False
                        on_decoded(frame_time, decoded_view)
                    if current_time != frame_time:
                        continue
                    yield current_request, image.copy()
                    try:
                        current_request = next(pending)
                    except StopIteration:
                        return
                    next_time = Fraction(
                        current_request[0] * current_request[1], current_request[2]
                    )
                    if next_time < current_time:
                        raise ValueError("iter_range 的请求必须按媒体时间递增")
                    current_time = next_time
            if active_at_open:
                raise TsDataNotReady(f"活动 MPEG-TS 段尾部尚未写出目标 PTS {current_time}: {path}")
            raise ValueError(f"MPEG-TS 中找不到精确 PTS {current_time}: {path}")
        finally:
            previous_error = sys.exception()
            with self._lock:
                if id(container) in self._open_containers:
                    try:
                        container.close()
                    except BaseException as close_error:
                        if previous_error is None or isinstance(previous_error, GeneratorExit):
                            raise
                        previous_error.add_note(f"关闭 TS 容器失败: {close_error}")
                    else:
                        self._open_containers.pop(id(container), None)

    def close(self) -> None:
        """关闭容器并释放本解码器持有的缓存租约。"""
        with self._cancel_lock:
            self._closing_count += 1
            self._read_cancelled.set()
        try:
            coordinator = self._gop_coordinator
            if coordinator is not None:
                coordinator.detach(self)
                self._gop_coordinator = None
            self._close_impl()
        except BaseException:
            with self._cancel_lock:
                self._close_incomplete = True
            raise
        else:
            with self._cancel_lock:
                self._close_incomplete = False
                self._cursor_cleanup_error = None
        finally:
            with self._cancel_lock:
                self._closed = True
                self._closing_count -= 1

    def _close_impl(self) -> None:
        """在生命周期锁下关闭解码游标和缓存租约。"""
        deadline = time.monotonic() + 5.0
        with self._iteration_condition:
            if self._iteration_threads.get(threading.get_ident()):
                raise RuntimeError("不能在解码回调内关闭当前解码器")
            while self._active_iterations:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("历史解码仍在执行，容器和存储租约已保留")
                self._iteration_condition.wait(remaining)
        with self._lifecycle_lock:
            reaper = self._reaper
            reaper_stop = self._reaper_stop
            reaper_stop.set()
            if reaper is not threading.current_thread():
                reaper.join(timeout=max(1.0, self.cursor_idle_timeout_s + 0.1))
                if reaper.is_alive():
                    raise TimeoutError("历史解码器的空闲回收线程未能退出")
            with self._lock:
                cursor = self._cursor
                try:
                    try:
                        self._reset_cursor_locked()
                        self._close_open_containers_locked()
                    finally:
                        try:
                            self._release_cache_lease()
                        finally:
                            self._keyframe_cache.clear()
                            self._keyframe_signatures.clear()
                            self._keyframe_offsets.clear()
                except BaseException as close_error:
                    if cursor is not None and (
                        not isinstance(cursor, GeneratorType) or cursor.gi_frame is not None
                    ):
                        # 未结束的生成器仍可能持有视频文件，须保留目录租约供重试。
                        self._cursor = cursor
                    if not self._open_containers and self._cursor is None:
                        try:
                            self._release_storage_lease()
                        except BaseException as cleanup_error:  # noqa: BLE001 - 保留原关闭异常
                            close_error.add_note(f"释放 TS 存储租约失败: {cleanup_error}")
                    raise
                else:
                    self._release_storage_lease()

    def _reap_idle_cursor(self, stop_event: threading.Event) -> None:
        """空闲超时后主动关闭解码器，即使后续没有读取请求。"""
        while not stop_event.wait(self.cursor_idle_timeout_s):
            with self._lock:
                if (
                    self._cursor is not None
                    and time.monotonic() - self._cursor_last_used >= self.cursor_idle_timeout_s
                ):
                    try:
                        self._reset_cursor_locked()
                    except BaseException:  # 保留句柄，等待显式关闭重试
                        LOGGER.exception("空闲历史解码游标关闭失败，必须重试关闭")
                        stop_event.set()
                        return

    def _ensure_reaper_locked(self) -> None:
        """close() 后再次读取时恢复空闲回收器。"""
        if not self._reaper_stop.is_set():
            return
        stop_event = threading.Event()
        self._reaper_stop = stop_event
        self._reaper = threading.Thread(
            target=self._reap_idle_cursor,
            args=(stop_event,),
            name="ts-decoder-cursor-reaper",
            daemon=True,
        )
        self._reaper.start()

    def _get_keyframes(self, segment: int, path: Path) -> list[tuple[Fraction, Fraction]]:
        """扫描包头元数据建立随机访问索引，不解码图像。"""
        index_path = path.with_suffix(".rap.jsonl")
        if index_path.is_file():
            size = index_path.stat().st_size
            with self._lock:
                cached = self._keyframe_cache.get(segment)
                offset = self._keyframe_offsets.get(segment, 0)
                if cached is not None and offset == size:
                    return cached
            result = list(cached or [])
            if offset > size:
                result.clear()
                offset = 0
            with index_path.open("rb") as handle:
                handle.seek(offset)
                raw = handle.read()
            lines = raw.splitlines(keepends=True)
            consumed = 0
            for line in lines:
                if not line.endswith((b"\n", b"\r")):
                    break
                try:
                    item = json.loads(line)
                    num, den = int(item["time_base_num"]), int(item["time_base_den"])
                    if den <= 0:
                        raise ValueError("time_base_den 必须为正")
                    time_base = Fraction(num, den)
                    result.append((int(item["pts"]) * time_base, int(item["dts"]) * time_base))
                    consumed += len(line)
                except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                    raise ValueError(
                        f"随机访问索引损坏: {index_path}:{offset + consumed + 1}: {exc}"
                    ) from exc
            if result:
                with self._lock:
                    self._keyframe_cache[segment] = result
                    self._keyframe_signatures[segment] = offset + consumed
                    self._keyframe_offsets[segment] = offset + consumed
                return result
        with self._lock:
            cached = self._keyframe_cache.get(segment)
            if cached is not None:
                return cached
        import av

        container = av.open(str(path), mode="r")
        with self._lock:
            self._open_containers[id(container)] = container
        try:
            if not container.streams.video:
                raise ValueError(f"MPEG-TS 分段没有视频流: {path}")
            stream = container.streams.video[0]
            result: list[tuple[Fraction, Fraction]] = []
            for packet in container.demux(stream):
                if not packet.is_keyframe or packet.pts is None or packet.dts is None:
                    continue
                time_base = Fraction(packet.time_base.numerator, packet.time_base.denominator)
                result.append((packet.pts * time_base, packet.dts * time_base))
            if not result:
                raise ValueError(f"MPEG-TS 分段缺少随机访问包: {path}")
            with self._lock:
                self._keyframe_cache[segment] = result
            return result
        finally:
            container.close()
            with self._lock:
                self._open_containers.pop(id(container), None)

    def _make_cache_key(
        self, segment: int, media_time: Fraction, width: int, height: int
    ) -> tuple[object, ...]:
        cache = self._decoded_cache
        if cache is None:
            return ()
        return (
            cache.storage_identity,
            str(self.directory.resolve()),
            segment,
            media_time.numerator,
            media_time.denominator,
            width,
            height,
        )

    @property
    def _cache_bytes(self) -> int:
        return self._decoded_cache.byte_size if self._decoded_cache is not None else 0

    @property
    def decoded_cache(self) -> DecodedFrameCache | None:
        """返回可供同一存储视图共享的缓存对象。"""
        return self._decoded_cache

    def bind_storage_lease(self, lease: TsStorageLease) -> None:
        """为解码器取得独立目录租约，调用方仍保留原租约。"""
        with self._lifecycle_lock:
            if self._closed or self._cursor is not None:
                raise RuntimeError("解码器开始读取或关闭后不能绑定存储租约")
            if self._storage_lease is not None:
                raise RuntimeError("解码器已经绑定存储租约")
            if self.directory.resolve() != lease.directory.resolve():
                raise ValueError("解码器目录与 TS 存储租约不一致")
            self._storage_lease = lease.fork()
            self._storage_managed = True

    def fork_storage_lease(self) -> TsStorageLease | None:
        """为独立的解码作业派生租约；永久航次无需租约。"""
        with self._lifecycle_lock:
            with self._cancel_lock:
                if self._closed or self._closing_count or self._close_incomplete:
                    raise RuntimeError("历史解码器已关闭或关闭未完成，不能派生作业租约")
            if self._storage_managed and self._storage_lease is None:
                raise RuntimeError("临时 TS 存储租约已释放")
            return None if self._storage_lease is None else self._storage_lease.fork()

    def unbind_storage_lease(self) -> None:
        """在采集启动前回滚目录租约绑定。"""
        with self._lifecycle_lock:
            if self._closed or self._cursor is not None:
                raise RuntimeError("解码器开始读取或关闭后不能撤销存储租约")
            self._release_storage_lease()
            self._storage_managed = False

    def _release_storage_lease(self) -> None:
        lease = self._storage_lease
        if lease is not None:
            # 租约 close 在清理目录失败时也已失效，不能作为可重新读取的凭据。
            self._storage_lease = None
            lease.close()

    def _put_cache(self, key: tuple[object, ...], image: np.ndarray) -> None:
        cache = self._decoded_cache
        if cache is not None:
            with self._cache_lease_lock:
                if self._cache_lease_active:
                    cache.put(key, image)

    def _get_cache(self, key: tuple[object, ...]) -> np.ndarray | None:
        cache = self._decoded_cache
        if cache is None:
            return None
        with self._cache_lease_lock:
            if not self._cache_lease_active:
                return None
            return cache.get(key)

    def _acquire_cache_lease(self) -> None:
        cache = self._decoded_cache
        if cache is None:
            return
        with self._cache_lease_lock:
            if not self._cache_lease_active:
                cache.acquire()
                self._cache_lease_active = True

    def _release_cache_lease(self) -> None:
        cache = self._decoded_cache
        if cache is None:
            return
        with self._cache_lease_lock:
            if self._cache_lease_active:
                cache.release()
                self._cache_lease_active = False

    def _reset_cursor_locked(self) -> None:
        cursor = self._cursor
        self._cursor = None
        self._cursor_requests = None
        self._cursor_segment = None
        self._cursor_time = None
        self._cursor_output_size = None
        self._cursor_last_used = 0.0
        if cursor is not None:
            try:
                cursor.close()
            except BaseException as exc:
                if not isinstance(cursor, GeneratorType) or cursor.gi_frame is not None:
                    self._cursor = cursor
                with self._cancel_lock:
                    self._cursor_cleanup_error = exc
                    self._close_incomplete = True
                    self._read_cancelled.set()
                raise

    def _close_open_containers_locked(self) -> None:
        """只在 native 容器确认关闭后才释放目录租约。"""
        for key, container in list(self._open_containers.items()):
            container.close()
            del self._open_containers[key]
