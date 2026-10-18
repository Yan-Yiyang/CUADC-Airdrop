"""有界实时画面合成与 FFmpeg UDP 发布器。"""

from __future__ import annotations

import logging
import queue
import subprocess
import threading
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from airdrop.qgcvideo.models import DisplaySnapshot
from airdrop.qgcvideo.render import FrameComposer

if TYPE_CHECKING:
    from airdrop.config import QgcVideoConfig


class Composer(Protocol):
    """渲染器只需把快照转换成 BGR 画布。"""

    def compose(self, snapshot: DisplaySnapshot) -> np.ndarray: ...


LOGGER = logging.getLogger(__name__)
_WIDTH = 1760
_HEIGHT = 990


def _url_host(host: str) -> str:
    """为 IPv6 URL 补齐方括号，同时保留配置中的 IPv4/主机名写法。"""
    if ":" in host and not host.startswith("["):
        return f"[{host}]"
    return host


@dataclass(frozen=True, slots=True)
class QgcVideoStats:
    """发布器的线程安全统计快照。"""

    submitted: int = 0
    encoded: int = 0
    dropped: int = 0
    restarts: int = 0
    failures: int = 0
    last_error: str | None = None
    last_latency_s: float | None = None


class QgcVideoPublisher:
    """从只读快照合成画面，并通过 FFmpeg 将 H.264 发布到 UDP。"""

    def __init__(
        self,
        config: QgcVideoConfig,
        frame_supplier: Callable[[], DisplaySnapshot | None],
        composer: Composer | None = None,
        on_event: Callable[[str, dict[str, object]], None] | None = None,
        *,
        subprocess_factory: Callable[..., Any] = subprocess.Popen,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config
        self.frame_supplier = frame_supplier
        self.composer = composer or FrameComposer()
        self.on_event = on_event
        self._process_factory = subprocess_factory
        self._clock = clock
        self._frames: queue.Queue[tuple[np.ndarray, float, float | None] | None] = queue.Queue(
            maxsize=1
        )
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._process_lock = threading.Lock()
        self._process: Any | None = None
        self._producer: threading.Thread | None = None
        self._encoder: threading.Thread | None = None
        self._running = False
        self._submitted = 0
        self._encoded = 0
        self._dropped = 0
        self._restarts = 0
        self._failures = 0
        self._last_error: str | None = None
        self._last_latency_s: float | None = None
        self._write_started: float | None = None
        self._ffmpeg: str | None = None

    @property
    def running(self) -> bool:
        with self._lock:
            return self._running

    @property
    def stats(self) -> QgcVideoStats:
        with self._lock:
            return QgcVideoStats(
                self._submitted,
                self._encoded,
                self._dropped,
                self._restarts,
                self._failures,
                self._last_error,
                self._last_latency_s,
            )

    def start(self) -> QgcVideoPublisher:
        """启动渲染与编码线程；重复调用不会创建第二组线程。"""
        with self._lock:
            if self._running:
                return self
            if any(
                thread is not None and thread.is_alive()
                for thread in (self._producer, self._encoder)
            ):
                raise RuntimeError("上一次 QGC 视频线程尚未退出")
        try:
            ffmpeg = self._resolve_ffmpeg()
        except Exception as exc:  # noqa: BLE001 — 显示编码不可用时不得中断任务。
            self._record_error(exc)
            return self
        with self._lock:
            if self._running:
                return self
            if any(
                thread is not None and thread.is_alive()
                for thread in (self._producer, self._encoder)
            ):
                raise RuntimeError("上一次 QGC 视频线程尚未退出")
            # 每轮启动使用独立邮箱，避免上次 stop 的哨兵或旧帧关闭新编码线程。
            self._frames = queue.Queue(maxsize=1)
            self._stop.clear()
            self._ffmpeg = ffmpeg
            self._running = True
            self._producer = threading.Thread(
                target=self._produce, name="qgc-video-render", daemon=True
            )
            self._encoder = threading.Thread(
                target=self._encode, name="qgc-video-encode", daemon=True
            )
            self._producer.start()
            self._encoder.start()
        return self

    def stop(self, timeout: float | None = None) -> None:
        """先终止 FFmpeg 解开阻塞写，再有限等待两个工作线程。"""
        with self._lock:
            producer = self._producer
            encoder = self._encoder
            if producer is None and encoder is None:
                self._running = False
                return
            self._running = False
            self._stop.set()
        self._terminate_process()
        deadline = self._clock() + (
            self.config.shutdown_timeout_s if timeout is None else max(0.0, timeout)
        )
        for thread in (producer, encoder):
            if thread is not None:
                thread.join(max(0.0, deadline - self._clock()))
        alive = [thread.name for thread in (producer, encoder) if thread and thread.is_alive()]
        if alive:
            raise TimeoutError(f"等待 QGC 视频线程退出超时：{', '.join(alive)}")

    def _resolve_ffmpeg(self) -> str:
        executable = self.config.ffmpeg_executable
        if executable:
            return executable
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()

    def _command(self) -> list[str]:
        ffmpeg = self._ffmpeg or self._resolve_ffmpeg()
        fps = float(self.config.fps)
        gop = max(1, round(fps))
        args = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pixel_format",
            "bgr24",
            "-video_size",
            f"{_WIDTH}x{_HEIGHT}",
            "-framerate",
            str(fps),
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-tune",
            "zerolatency",
            "-pix_fmt",
            "yuv420p",
            "-bf",
            "0",
            "-g",
            str(gop),
            "-keyint_min",
            str(gop),
            "-sc_threshold",
            "0",
            "-x264-params",
            "repeat-headers=1",
            "-b:v",
            f"{int(self.config.bitrate_kbps)}k",
        ]
        host = _url_host(self.config.host)
        if self.config.transport == "rtp":
            url = f"rtp://{host}:{self.config.port}?pkt_size=1200&local_rtpport=0&local_rtcpport=0"
            args.extend(["-f", "rtp", "-payload_type", "96", url])
        else:
            url = f"udp://{host}:{self.config.port}?pkt_size=1316"
            args.extend(["-f", "mpegts", url])
        return args

    def _produce(self) -> None:
        period = 1.0 / float(self.config.fps)
        next_frame = self._clock()
        while not self._stop.is_set():
            try:
                # 即使本轮 supplier 或合成失败，也持续检查编码器写入看门狗。
                self._check_write_timeout()
                # supplier 与 compose 运行在本线程，绝不占用采集或感知回调。
                snapshot = self.frame_supplier()
                if snapshot is not None:
                    set_stats = getattr(self.composer, "set_publisher_stats", None)
                    if set_stats is not None:
                        set_stats(self.stats)
                    frame = self.composer.compose(snapshot)
                    if frame.shape != (_HEIGHT, _WIDTH, 3) or frame.dtype != np.uint8:
                        raise ValueError("合成器必须返回 uint8 BGR 1760×990 画面")
                    current_frame = getattr(snapshot, "frame", None)
                    capture_time = (
                        None
                        if getattr(snapshot, "replay_mode", False)
                        else getattr(current_frame, "capture_timestamp", None)
                    )
                    self._offer(frame, self._clock(), capture_time)
            except Exception as exc:  # noqa: BLE001 — 隔离画面供应及合成错误。
                self._record_error(exc)
            finally:
                self._check_write_timeout()
            next_frame += period
            self._stop.wait(max(0.0, next_frame - self._clock()))
            if next_frame < self._clock() - period:
                next_frame = self._clock()

    def _offer(self, frame: np.ndarray, submitted_at: float, capture_time: float | None) -> None:
        if self._stop.is_set():
            return
        # 帧进入邮箱前复制，渲染缓存可由 compose 后续复用而不会污染编码数据。
        valid_capture_time = (
            float(capture_time) if capture_time is not None and np.isfinite(capture_time) else None
        )
        item = (np.ascontiguousarray(frame).copy(), submitted_at, valid_capture_time)
        with self._lock:
            self._submitted += 1
        try:
            self._frames.put_nowait(item)
        except queue.Full:
            with suppress(queue.Empty):
                self._frames.get_nowait()
            with self._lock:
                self._dropped += 1
            try:
                self._frames.put_nowait(item)
            except queue.Full:
                with self._lock:
                    self._dropped += 1

    def _encode(self) -> None:
        restarts_left = int(self.config.max_restarts)
        process = self._start_process()
        while not self._stop.is_set():
            try:
                item = self._frames.get(timeout=0.1)
            except queue.Empty:
                if process is None or process.poll() is not None:
                    if restarts_left <= 0:
                        self._record_error(RuntimeError("FFmpeg 重启次数已耗尽"))
                        break
                    restarts_left -= 1
                    self._restart_delay()
                    process = self._start_process(restart=True)
                continue
            if item is None:
                break
            if process is None or process.poll() is not None:
                if restarts_left <= 0:
                    with self._lock:
                        self._dropped += 1
                    self._record_error(RuntimeError("FFmpeg 重启次数已耗尽"))
                    break
                restarts_left -= 1
                self._restart_delay()
                process = self._start_process(restart=True)
            if process is None:
                with self._lock:
                    self._dropped += 1
                continue
            frame, submitted_at, capture_time = item
            try:
                data = memoryview(frame).cast("B")
                offset = 0
                with self._lock:
                    self._write_started = self._clock()
                while offset < len(data):
                    written = process.stdin.write(data[offset:])
                    if written is None or written <= 0:
                        raise BrokenPipeError("FFmpeg stdin 未能继续接收画面")
                    offset += written
                process.stdin.flush()
                with self._lock:
                    self._encoded += 1
                    capture_latency = (
                        time.time() - capture_time
                        if capture_time is not None and abs(time.time() - capture_time) < 86_400
                        else self._clock() - submitted_at
                    )
                    self._last_latency_s = max(0.0, capture_latency)
                    self._write_started = None
            except (BrokenPipeError, OSError, ValueError) as exc:
                with self._lock:
                    self._write_started = None
                    self._dropped += 1
                self._record_error(exc)
                self._terminate_process(process)
                process = None
        self._terminate_process(process)
        self._stop.set()
        with self._lock:
            self._running = False

    def _start_process(self, *, restart: bool = False) -> Any | None:
        if self._stop.is_set():
            return None
        try:
            process = self._process_factory(
                self._command(),
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                bufsize=0,
            )
            with self._process_lock:
                self._process = process
            if self._stop.is_set():
                self._terminate_process(process)
                return None
            if restart:
                with self._lock:
                    self._restarts += 1
            return process
        except (OSError, ValueError) as exc:
            self._record_error(exc)
            return None

    def _restart_delay(self) -> None:
        self._stop.wait(float(self.config.restart_delay_s))

    def _check_write_timeout(self) -> None:
        with self._lock:
            started = self._write_started
        if started is not None and self._clock() - started > float(self.config.write_timeout_s):
            self._record_error(TimeoutError("FFmpeg 写入画面超时，已终止编码进程"))
            self._terminate_process()

    def _terminate_process(self, process: Any | None = None) -> None:
        with self._process_lock:
            selected = process if process is not None else self._process
            if selected is None:
                return
            if selected is self._process:
                self._process = None
        try:
            if selected.poll() is None:
                selected.terminate()
                try:
                    selected.wait(timeout=0.5)
                except subprocess.TimeoutExpired:
                    selected.kill()
                    selected.wait(timeout=0.5)
        except OSError, subprocess.SubprocessError:
            LOGGER.debug("终止 FFmpeg 进程时发生错误", exc_info=True)

    def _replace_pending(self, item: tuple[np.ndarray, float, float | None] | None) -> None:
        with suppress(queue.Empty):
            self._frames.get_nowait()
        with suppress(queue.Full):
            self._frames.put_nowait(item)

    def _record_error(self, error: Exception) -> None:
        message = f"{type(error).__name__}: {error}"
        with self._lock:
            self._failures += 1
            if isinstance(error, TimeoutError) or self._last_error is None:
                self._last_error = message
        LOGGER.warning("QGC 视频发布失败：%s", message)
        if self.on_event is not None:
            try:
                self.on_event("qgc_video_error", {"error": message})
            except Exception:
                LOGGER.debug("QGC 视频错误事件回调失败", exc_info=True)
