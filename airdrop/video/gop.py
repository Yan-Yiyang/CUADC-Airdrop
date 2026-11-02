"""同一随机访问范围的并发请求协调。"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from fractions import Fraction
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .cache import DecodedFrameCache
    from .cold import TsRangeDecoder


@dataclass(eq=False, slots=True)
class _Waiter:
    target: Fraction
    request: tuple[int, int, int]
    owner: object
    ready: threading.Event = field(default_factory=threading.Event)
    image: np.ndarray | None = None
    error: BaseException | None = None


class _GopJob:
    """独占一个 native 解码器，向等待中的媒体时刻逐帧分发。"""

    def __init__(
        self,
        decoder: TsRangeDecoder,
        cache: DecodedFrameCache,
        segment: int,
        size: tuple[int, int] | None,
        idle_timeout: float,
    ):
        self.decoder = decoder
        self.cache = cache
        self.segment = segment
        self.size = size
        self.idle_timeout = idle_timeout
        self.condition = threading.Condition()
        self.waiters: list[_Waiter] = []
        self.outstanding = 0
        self.stopping = False
        self.busy = False
        self.last_used = time.monotonic()
        self.failed: BaseException | None = None
        self.cleanup_error: BaseException | None = None
        self.cleaned = False
        self._cleanup_lock = threading.Lock()
        self.thread = threading.Thread(target=self._run, name="ts-gop-job", daemon=True)
        cache.acquire()
        try:
            self.thread.start()
        except BaseException as exc:
            cache.release()
            try:
                decoder.close()
            except BaseException as cleanup_error:  # noqa: BLE001 - 保留线程启动异常
                exc.add_note(f"关闭 GOP 解码器失败: {cleanup_error}")
            raise

    def add(self, waiter: _Waiter) -> None:
        with self.condition:
            if self.stopping:
                raise RuntimeError("GOP 解码作业正在关闭")
            if self.failed is not None:
                raise RuntimeError("GOP 解码作业已失败") from self.failed
            if self.outstanding >= 8:
                raise RuntimeError("GOP 解码等待请求已达到上限")
            self.waiters.append(waiter)
            self.outstanding += 1
            self.condition.notify_all()

    def cancel(self, waiter: _Waiter) -> None:
        with self.condition:
            if waiter in self.waiters:
                self.waiters.remove(waiter)
                if self.busy and not self.waiters:
                    self.stopping = True
            self.condition.notify_all()

    def finish(self, waiter: _Waiter) -> None:
        """读取者取走或放弃结果后释放在途额度。"""
        with self.condition:
            if waiter in self.waiters:
                self.waiters.remove(waiter)
            self.outstanding -= 1
            if self.outstanding < 0:
                raise RuntimeError("GOP 在途请求计数下溢")
            self.condition.notify_all()

    def cancel_owner(self, owner: object) -> None:
        with self.condition:
            removed = False
            for waiter in tuple(self.waiters):
                if waiter.owner is owner:
                    self.waiters.remove(waiter)
                    waiter.error = InterruptedError("历史解码读取已取消")
                    waiter.ready.set()
                    removed = True
            if removed and self.busy and not self.waiters:
                self.stopping = True
            self.condition.notify_all()

    def is_idle(self) -> bool:
        with self.condition:
            return not self.busy and self.outstanding == 0

    def has_outstanding(self) -> bool:
        with self.condition:
            return self.outstanding != 0

    def stop(self) -> None:
        with self.condition:
            self.stopping = True
            for waiter in self.waiters:
                waiter.error = InterruptedError("GOP 解码作业已关闭")
                waiter.ready.set()
            self.waiters.clear()
            self.condition.notify_all()
        self.thread.join(timeout=5.0)
        if self.thread.is_alive():
            raise TimeoutError("GOP 解码仍在执行，已保留容器和存储租约")
        with self._cleanup_lock:
            if not self.cleaned:
                # 工作线程已退出，重试同一个 decoder 的原生容器关闭。
                self.decoder.close()
                self.cache.release()
                self.cleaned = True
                self.cleanup_error = None

    def _dispatch(self, frame_time: Fraction, image: np.ndarray) -> None:
        with self.condition:
            if self.stopping:
                raise InterruptedError("GOP 解码作业已关闭")
        width, height = self.size or (0, 0)
        self.cache.put(
            (
                self.cache.storage_identity,
                str(self.decoder.directory.resolve()),
                self.segment,
                frame_time.numerator,
                frame_time.denominator,
                width,
                height,
            ),
            image,
        )
        with self.condition:
            matching = [w for w in self.waiters if w.target == frame_time]
            if not matching:
                return
            # 所有并发等待者共享一份在途副本，调用方取结果时再各自复制。
            result = image.copy()
            for waiter in matching:
                self.waiters.remove(waiter)
                waiter.image = result
                waiter.ready.set()

    def _run(self) -> None:
        cursor: Any = None
        requests: queue.Queue[tuple[int, int, int]] = queue.Queue()
        last: Fraction | None = None
        try:
            while True:
                with self.condition:
                    while not self.waiters and not self.stopping:
                        if self.outstanding:
                            self.condition.wait()
                        elif not self.condition.wait(timeout=self.idle_timeout):
                            self.stopping = True
                    if self.stopping:
                        break
                    # 先向最高目标推进，以便一次遍历顺带交付更早的请求。
                    waiter = max(self.waiters, key=lambda item: item.target)
                    self.busy = True
                if cursor is None or (last is not None and waiter.target <= last):
                    if cursor is not None:
                        cursor.close()
                    requests = queue.Queue()

                    def ordered(first: tuple[int, int, int], pending=requests):
                        yield first
                        while True:
                            yield pending.get()

                    cursor = self.decoder.iter_range(
                        self.segment,
                        ordered(waiter.request),
                        output_size=self.size,
                        on_decoded=self._dispatch,
                    )
                else:
                    requests.put(waiter.request)
                next(cursor)
                last = waiter.target
                with self.condition:
                    self.busy = False
                    self.last_used = time.monotonic()
        except BaseException as exc:  # noqa: BLE001 - 通知所有等待者并保留关闭异常
            with self.condition:
                self.failed = exc
                self.busy = False
                for waiter in self.waiters:
                    waiter.error = exc
                    waiter.ready.set()
                self.waiters.clear()
                self.condition.notify_all()
        finally:
            cleanup_error = None
            try:
                if cursor is not None:
                    cursor.close()
            except BaseException as exc:  # noqa: BLE001 - 关闭失败必须向 owner 报告
                cleanup_error = exc
            try:
                self.decoder.close()
            except BaseException as exc:  # noqa: BLE001 - 保留租约以便再次关闭
                cleanup_error = exc
            if cleanup_error is None:
                self.cache.release()
                self.cleaned = True
            self.cleanup_error = cleanup_error


class GopCoordinator:
    """为同一缓冲及其冻结视图合并历史读取。"""

    def __init__(self, cache: DecodedFrameCache):
        self.cache = cache
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._views: set[TsRangeDecoder] = set()
        self._jobs: dict[tuple[object, ...], _GopJob] = {}
        self._closing = False

    def attach(self, decoder: TsRangeDecoder) -> None:
        with self._lock:
            if self._closing:
                raise RuntimeError("GOP 协调器已关闭，不能绑定新读取者")
            self._views.add(decoder)

    def detach(self, decoder: TsRangeDecoder) -> None:
        with self._lock:
            self._views.discard(decoder)
            jobs = list(self._jobs.items())
            last = not self._views
            if last:
                self._closing = True
        for _, job in jobs:
            job.cancel_owner(decoder)
        if last:
            errors = []
            for key, job in jobs:
                try:
                    job.stop()
                except BaseException as exc:  # noqa: BLE001 - 所有作业都要尝试关闭
                    errors.append(exc)
                else:
                    with self._lock:
                        if self._jobs.get(key) is job:
                            del self._jobs[key]
            if errors:
                raise BaseExceptionGroup("GOP 作业未能关闭", errors)

    def read(  # noqa: PLR0912, PLR0915, PLR0917 - 租约回滚及取消须显式处理
        self,
        decoder: TsRangeDecoder,
        segment: int,
        pts: int,
        num: int,
        den: int,
        size: tuple[int, int] | None,
        cancelled: threading.Event,
    ) -> np.ndarray:
        from .cold import TsDataNotReady, TsRangeDecoder
        from .segments import SegmentState

        if cancelled.is_set():
            raise InterruptedError("历史解码读取已取消")
        target = Fraction(pts * num, den)
        width, height = size or (0, 0)
        cache_key = decoder._make_cache_key(segment, target, width, height)  # noqa: SLF001
        cached = self.cache.get(cache_key)
        if cached is not None:
            if cancelled.is_set():
                raise InterruptedError("历史解码读取已取消")
            return cached
        path = decoder.directory / f"segment-{segment:06d}.ts"
        with decoder._lock:  # noqa: SLF001 - 协调器共享该 decoder 的随机访问索引锁
            points = decoder._get_keyframes(segment, path)  # noqa: SLF001
        before = [item for item in points if item[0] <= target]
        if not before:
            lifecycle = decoder.segment_lifecycle
            if lifecycle is not None and lifecycle.snapshot(segment).state is SegmentState.ACTIVE:
                raise TsDataNotReady(f"活动 MPEG-TS 段尚未写出随机访问点: {path}")
            raise ValueError(f"目标时间 {target} 之前没有可用关键帧: {path}")
        key = (str(decoder.directory.resolve()), segment, before[-1][1], size)
        waiter = _Waiter(target, (pts, num, den), decoder)
        with self._condition:
            if decoder not in self._views:
                raise InterruptedError("历史解码读取已取消")
            while True:
                for stale_key, stale in list(self._jobs.items()):
                    if (
                        not stale.thread.is_alive()
                        and stale.cleaned
                        and not stale.has_outstanding()
                    ):
                        del self._jobs[stale_key]
                job = self._jobs.get(key)
                if job is not None and (not job.thread.is_alive() or job.stopping):
                    if job.has_outstanding():
                        if cancelled.is_set():
                            raise InterruptedError("历史解码读取已取消")
                        self._condition.wait(timeout=0.05)
                        continue
                    job.stop()
                    del self._jobs[key]
                    job = None
                if job is not None:
                    try:
                        job.add(waiter)
                    except RuntimeError:
                        if not job.stopping:
                            raise
                        job.stop()
                        del self._jobs[key]
                        continue
                    break
                if len(self._jobs) >= 8:
                    idle = [
                        (item_key, item) for item_key, item in self._jobs.items() if item.is_idle()
                    ]
                    if not idle:
                        if cancelled.is_set():
                            raise InterruptedError("历史解码读取已取消")
                        self._condition.wait(timeout=0.05)
                        continue
                    stale_key, stale = min(idle, key=lambda item: item[1].last_used)
                    stale.stop()
                    del self._jobs[stale_key]
                lease = decoder.fork_storage_lease()
                worker = None
                try:
                    worker = TsRangeDecoder(
                        decoder.directory,
                        max_cache_bytes=decoder.max_cache_bytes,
                        cursor_idle_timeout_s=decoder.cursor_idle_timeout_s,
                        segment_lifecycle=decoder.segment_lifecycle,
                        cache_enabled=False,
                    )
                    if lease is not None:
                        worker.bind_storage_lease(lease)
                        lease.close()
                        lease = None
                    job = _GopJob(worker, self.cache, segment, size, decoder.cursor_idle_timeout_s)
                except BaseException as exc:
                    if worker is not None:
                        try:
                            worker.close()
                        except BaseException as cleanup_error:  # noqa: BLE001 - 保留构造异常
                            exc.add_note(f"关闭 GOP 解码器失败: {cleanup_error}")
                    if lease is not None:
                        try:
                            lease.close()
                        except BaseException as cleanup_error:  # noqa: BLE001 - 保留构造异常
                            exc.add_note(f"释放 GOP 存储租约失败: {cleanup_error}")
                    raise
                self._jobs[key] = job
                try:
                    job.add(waiter)
                except RuntimeError:
                    if not job.stopping:
                        raise
                    job.stop()
                    del self._jobs[key]
                    continue
                break
        try:
            while not waiter.ready.wait(timeout=0.05):
                if cancelled.is_set():
                    job.cancel(waiter)
                    raise InterruptedError("历史解码读取已取消")
            if waiter.error is not None:
                raise waiter.error
            if waiter.image is None:
                raise RuntimeError("GOP 请求结束但没有图像")
            return waiter.image.copy()
        finally:
            job.finish(waiter)
