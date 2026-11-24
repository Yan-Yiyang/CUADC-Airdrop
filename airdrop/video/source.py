"""思翼 HM30 图传拉流（ffmpeg 子进程，唯一后端）。

接线与地址
----------
HM30 的地面端不"推"视频给上位机，而是把机载以太网（``192.168.144.0/24``）
**透明桥接**到 LAN 口 / 内置 WiFi / USB WiFi 上：地面端 LAN 口插到电脑、电脑
配一个同网段地址，就能直接访问机载 IP 相机。官方 FAQ 的说法是：只要 IP 相机
输出 RTSP、能接以太网，就能配合 HM30 工作。

所以"拉流"就是打开一条 RTSP，默认 ``rtsp://192.168.144.25:8554/main.264``
（SIYI 云台/相机系列的约定地址）。**地址写错时本模块不探测、不猜测**：直接
失败并在拉流源上给出明确错误（``stats.last_error`` + 错误日志），由使用方自己
核对地址。想手动确认，用 ffmpeg 命令行试一次即可::

    ffmpeg -rtsp_transport udp -i rtsp://192.168.144.25:8554/main.264 \
        -frames:v 1 -f null -

同理，输出尺寸也不自动探测：默认按 720p（1280x720）切帧，要换分辨率就在
:class:`VideoConfig` 里显式改，而不是先猜一个再跑出错误的画面比例。

接收实现
--------
使用 ``subprocess`` 启动 ffmpeg，从管道读取 ``rawvideo/bgr24`` 裸帧。子进程可由父进程
停止，避免拉流线程永久阻塞在不可中断的媒体读取调用中，也避免在 Python 侧重复解码。

UDP 边界
--------
消费者处理速度低于输入速率时，数据报可能在 UDP 套接字层丢失；这类丢失不计入
``stats.dropped``，该统计只覆盖管道下游已观察到的帧。探测不可 seek 的 UDP 流时，
ffmpeg 也可能在建立解码输出前读取并丢弃输入数据。因此 ``stats.dropped == 0`` 不足以
证明网络未丢包或启动阶段没有丢帧。

本模块只依赖 numpy 与标准库（不再 import cv2）。

每一帧都不丢：sink
------------------
采集线程读出的**每一帧都会先交给注册的 sink**（:meth:`Hm30VideoSource.add_sink`），
一帧都不会被跳过。这是本项目的硬要求：目标出现的时间可能极短，漏一帧就可能漏掉
目标；而"先侦查后空投"的模式允许一定的处理延时，不允许丢帧。把
:class:`~airdrop.video.buffer.AlignmentWriter` 挂上去，每帧就会连同它拍摄时刻的遥测
一起写进环形缓冲，之后随便什么时候回看都在。

sink 在采集线程里**同步**执行，所以它必须快（缓冲写入只做一次 JPEG 编码，几毫秒）。
sink 抛出的异常只记日志，不会打断拉流。

:meth:`Hm30VideoSource.read` / :meth:`latest` 面向**实时**消费者（预览、即时引导）：
它们只关心"现在这一帧"，慢消费者会丢掉中间帧并计入 :attr:`VideoStats.dropped`。
那是实时路径自己的取舍，与 sink/缓冲那条"每帧都留存"的路径互不影响——历史在
缓冲里，不会因为某个实时消费者掉队而缺失。

注意 ``VideoFrame.image`` 是**复用管道缓冲区的视图**，下一帧读入会覆盖它；
跨帧持有画面必须先 :meth:`VideoFrame.copy`。

性能提示
--------
管道输出为裸 BGR，分辨率与帧率会直接影响内存带宽和 CPU 拷贝开销。
``ffmpeg_decoder`` 可配置为平台支持的硬件解码器；具体可用性取决于 FFmpeg 构建和设备。
"""

from __future__ import annotations

import contextlib
import logging
import multiprocessing
import os
import queue
import shutil
import subprocess
import tempfile
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace
from multiprocessing.connection import wait as wait_connections
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from typing import Any

import numpy as np

from ..telemetry.models import TelemetrySnapshot
from .segments import SegmentLifecycle, SegmentState

LOGGER = logging.getLogger(__name__)


class VideoStorageError(RuntimeError):
    """视频持久化或帧映射失败，禁止自动按网络断流重连。"""


def _pyav_input_options(config: VideoConfig) -> dict[str, str]:
    """校验实验 PyAV 路径支持的输入参数并转换为 av.open 选项。"""
    if config.extra_output_args or config.ffmpeg_decoder:
        raise VideoStorageError("PyAV 子进程暂不支持 FFmpeg 输出参数或 decoder 配置")
    args = config.extra_input_args
    if not args:
        return {}
    if len(args) != 2 or args[0] != "-protocol_whitelist" or not args[1].strip():
        raise VideoStorageError(
            "PyAV 子进程仅支持 -protocol_whitelist <protocols> 输入参数；请在启动前修正 VideoConfig"
        )
    return {"protocol_whitelist": args[1]}


# HM30 常见地址（地面端与机载相机同处 192.168.144.0/24）
HM30_CAMERA_IP = "192.168.144.25"
HM30_GROUND_IP = "192.168.144.12"
# SIYI 相机/云台默认 RTSP 主码流
HM30_DEFAULT_RTSP = f"rtsp://{HM30_CAMERA_IP}:8554/main.264"

SUPPORTED_TRANSPORTS = frozenset({"udp", "tcp"})

# 图传链路固定延时的默认估计（秒）：从"机载相机曝光"到"本进程从管道里切出
# 一帧"之间的处理时间，含 H.264 编码、无线传输、解码与管道缓冲；应按实际链路标定。
# 拿到一帧后用它反推该画面真正的拍摄时刻（VideoFrame.capture_timestamp），
# 再按那一刻取遥测，见 :mod:`airdrop.video.align`。
DEFAULT_TELEMETRY_LAG = 0.15


# ----------------------------------------------------------------------
# 配置与数据结构
# ----------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class VideoConfig:
    """拉流配置。

    ``width``/``height`` 是输出尺寸（默认 720p）。本模块**不探测**流分辨率：
    ffmpeg 后端要按固定尺寸从管道切帧，所以要换分辨率必须在这里显式改。
    """

    url: str = HM30_DEFAULT_RTSP
    # RTSP 传输层：udp 延迟更低，tcp 更耐丢包
    transport: str = "udp"
    width: int = 1280
    height: int = 720
    # 断流判定超时（秒），对应 ffmpeg 的 -timeout
    read_timeout: float = 3.0
    # 图传链路的固定延时（秒），会被写进每一帧（VideoFrame.lag）
    telemetry_lag: float = DEFAULT_TELEMETRY_LAG
    reconnect_delay: float = 1.0
    max_reconnect_delay: float = 10.0
    # ffmpeg 可执行文件；None 表示用 imageio-ffmpeg 自带的那份
    ffmpeg_executable: str | None = None
    # 指定解码器，例如 "h264_cuvid" / "hevc_cuvid"（NVDEC 硬解）
    ffmpeg_decoder: str | None = None
    # 追加到 ffmpeg 输入参数（-i 之前）的原始参数
    extra_input_args: tuple[str, ...] = ()
    # 追加到 ffmpeg 输出参数（-i 之后）的原始参数
    extra_output_args: tuple[str, ...] = ()

    def validated(self) -> "VideoConfig":
        if self.transport not in SUPPORTED_TRANSPORTS:
            raise ValueError(
                f"不支持的传输层: {self.transport}，可选 {sorted(SUPPORTED_TRANSPORTS)}"
            )
        if self.width <= 0 or self.height <= 0:
            raise ValueError(
                f"输出尺寸必须为正: width={self.width} height={self.height}；"
                f"本模块不探测流分辨率（默认 720p 即 1280x720）"
            )
        if self.telemetry_lag < 0:
            raise ValueError(f"telemetry_lag 不能为负: {self.telemetry_lag}")
        return self


@dataclass(frozen=True, slots=True)
class VideoFrame:
    """一帧图传画面。

    ``image`` 是 BGR 顺序的 ndarray。**它可能在下一帧到达时被覆盖**：ffmpeg
    后端是从复用的管道缓冲区上切出视图（零拷贝换带宽），所以拿到的数组只在
    "下一次读取之前"有效。要跨帧持有画面（入缓冲区、异步推理、落盘）必须先
    :meth:`copy`——否则你保存下来的每一帧最后都会显示成同一张最新的画。同样，
    不要就地修改 ``image``（对视图而言就是在改管道缓冲区）。

    ``timestamp`` 是**收到**这一帧的本地墙钟时间；``lag`` 是链路固定延时
    （取自 :attr:`VideoConfig.telemetry_lag`），二者之差
    :attr:`capture_timestamp` 才是画面真正被拍下来的时刻。按时间取遥测时
    必须用后者，否则会带上整段链路延时的系统性偏差——无人机以 10 m/s 平飞
    时，150ms 就是 1.5 米的位置差。
    """

    index: int
    image: np.ndarray
    timestamp: float
    lag: float = 0.0
    media_pts: int | None = None
    time_base_num: int | None = None
    time_base_den: int | None = None
    segment: int = 0
    random_access_dts: int | None = None
    random_access_time_base_num: int | None = None
    random_access_time_base_den: int | None = None
    aligned_snapshot: TelemetrySnapshot | None = None
    aligned_extrapolated: bool = False
    aligned_offset: float = 0.0
    alignment_disabled: bool = False

    @property
    def width(self) -> int:
        return int(self.image.shape[1])

    @property
    def height(self) -> int:
        return int(self.image.shape[0])

    @property
    def age(self) -> float:
        """距收到该帧已经过去多少秒。"""
        return time.time() - self.timestamp

    @property
    def capture_timestamp(self) -> float:
        """该画面真正的拍摄时刻（本地时钟）= ``timestamp - lag``。

        把它交给 :meth:`airdrop.broker.TelemetryBroker.get_snapshot_at`
        才能拿到与画面同一时刻的飞机状态。``lag`` 为 0 时与 ``timestamp``
        相同，即不做延时补偿。
        """
        return self.timestamp - self.lag

    @property
    def capture_age(self) -> float:
        """距该画面被拍下已经过去多少秒（= ``age + lag``）。"""
        return self.age + self.lag

    def copy(self) -> "VideoFrame":
        """返回一份图像数据独立的副本。

        跨帧持有画面时必须用它：ffmpeg 后端的 ``image`` 是复用管道缓冲区的
        视图，下一帧读入时就会被覆盖（见类文档）。
        """
        return replace(self, image=self.image.copy())


@dataclass(slots=True)
class VideoStats:
    """拉流统计，供日志与链路质量判断使用。"""

    state: str = "idle"
    frames: int = 0
    dropped: int = 0
    reconnects: int = 0
    fps: float = 0.0
    last_error: str | None = None
    sink_errors: int = 0
    pyav_queue_age_s: float = 0.0
    pyav_queue_age_max_s: float = 0.0


# ----------------------------------------------------------------------
# 工具
# ----------------------------------------------------------------------
def _find_ffmpeg(explicit: str | None = None) -> str | None:
    """定位 ffmpeg：显式路径 → imageio-ffmpeg 自带 → PATH。"""
    if explicit:
        return explicit
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001 - 取不到就记日志，任何失败都退回 PATH 查找
        return shutil.which("ffmpeg")


def _is_rtsp(url: str) -> bool:
    return url.lower().startswith("rtsp")


def _watch_parent_exit() -> None:
    """父进程被强制终止时由看门线程硬退出采集子进程。"""
    parent = multiprocessing.parent_process()
    if parent is not None:
        wait_connections([parent.sentinel])
        os._exit(91)


def _is_normal_input_termination(error: Exception, exit_error_type: type[BaseException]) -> bool:
    """识别网络读取超时或连接结束；解码错误不属于可正常封口的断流。"""
    return isinstance(
        error,
        (
            TimeoutError,
            EOFError,
            ConnectionError,
            BrokenPipeError,
            BlockingIOError,
            InterruptedError,
            exit_error_type,
        ),
    )


def _pyav_capture_worker(  # noqa: PLR0912, PLR0915, PLR0917 - 子进程收尾需显式分支
    config: VideoConfig,
    recording_directory: str | None,
    segment: int,
    messages: Any,
    free_slots: Any,
    slot_names: tuple[str, ...],
    frame_bytes: int,
    stop_event: Any,
) -> None:
    """在可强制终止的独立进程中解码并写 TS，IPC 最多排队两帧。"""
    import threading
    from fractions import Fraction

    threading.Thread(target=_watch_parent_exit, name="pyav-parent-watch", daemon=True).start()

    def send(message: dict[str, Any]) -> bool:
        while True:
            try:
                messages.put(message, timeout=0.1)
                return True
            except queue.Full:
                continue

    container = None
    writer = None
    shared_slots: list[SharedMemory] = []
    error_kind = "network"
    error_message: str | None = None
    segment_sealable = False
    finished = False
    startup_packets_skipped = 0
    try:
        import av
        from av.error import ExitError

        from .cold import TsSegmentWriter

        if not send({"kind": "opening"}):
            return
        options = {"probesize": "500000", "analyzeduration": "0"}
        options.update(_pyav_input_options(config))
        if _is_rtsp(config.url):
            options["rtsp_transport"] = config.transport
        timeout = max(float(config.read_timeout), 0.1)
        container = av.open(
            config.url,
            mode="r",
            options=options,
            timeout=(timeout, timeout),
        )
        if not container.streams.video:
            raise RuntimeError(f"输入没有视频流: {config.url}")
        stream = container.streams.video[0]
        stream.thread_type = "NONE"
        if recording_directory is not None:
            error_kind = "storage"
            writer = TsSegmentWriter(
                Path(recording_directory), segment=segment, input_stream=stream
            )
            if not send({"kind": "segment_opened", "segment": segment}):
                return
        shared_slots = [SharedMemory(name=name) for name in slot_names]
        error_kind = "network"
        random_access: list[tuple[Fraction, int, int, int]] = []
        started = False
        startup_deadline = time.monotonic() + max(30.0, timeout)
        packets = iter(container.demux(stream))
        while True:
            if stop_event.is_set():
                segment_sealable = started
                finished = True
                break
            try:
                packet = next(packets)
            except StopIteration:
                segment_sealable = started
                finished = True
                break
            except Exception as exc:
                if started and _is_normal_input_termination(exc, ExitError):
                    # 读取下一包时链路正常超时/断开；当前已 mux 的分段可正常封口，
                    # 同时把断流交给父线程触发重连。
                    segment_sealable = True
                    if stop_event.is_set():
                        finished = True
                    else:
                        error_message = f"{type(exc).__name__}: {exc}"
                    break
                raise
            packet_tb = packet.time_base
            if not started:
                if time.monotonic() >= startup_deadline:
                    raise RuntimeError("等待视频随机访问关键帧超过 30 秒")
                if getattr(packet, "is_corrupt", False):
                    if packet.is_keyframe:
                        raise RuntimeError("首个随机访问关键包标记为损坏")
                    startup_packets_skipped += 1
                    continue
                if packet.size <= 0:
                    continue
                if not packet.is_keyframe:
                    startup_packets_skipped += 1
                    continue
                if packet.pts is None or packet.dts is None or packet_tb is None:
                    raise VideoStorageError("首个随机访问包缺少 PTS、DTS 或 time_base")
                if not send({"kind": "ready", "startup_packets_skipped": startup_packets_skipped}):
                    return
                started = True
            elif packet.size > 0 and getattr(packet, "is_corrupt", False):
                raise RuntimeError("输入视频包标记为损坏")
            if (
                packet.size > 0
                and packet.is_keyframe
                and packet.pts is not None
                and packet.dts is not None
                and packet_tb is not None
            ):
                tb = Fraction(packet_tb.numerator, packet_tb.denominator)
                random_access.append(
                    (Fraction(packet.pts) * tb, packet.dts, tb.numerator, tb.denominator)
                )
            decoded = packet.decode()
            ready_times = [(time.time(), time.monotonic_ns()) for _ in decoded]
            committed_time: Fraction | None = None
            if writer is not None and packet.size > 0:
                error_kind = "storage"
                committed_time = writer.mux(packet)
                error_kind = "network"
            elif writer is not None:
                committed_time = writer.committed_time
            for frame, (frame_timestamp, frame_monotonic_ns) in zip(
                decoded, ready_times, strict=True
            ):
                if getattr(frame, "is_corrupt", False):
                    raise RuntimeError("解码帧标记为损坏")
                if frame.pts is None or frame.time_base is None:
                    raise VideoStorageError("解码帧缺少 PTS 或 time_base")
                frame_tb = Fraction(frame.time_base.numerator, frame.time_base.denominator)
                frame_time = Fraction(frame.pts) * frame_tb
                access = next(
                    (item for item in reversed(random_access) if item[0] <= frame_time),
                    None,
                )
                if access is None:
                    raise VideoStorageError(f"解码帧 PTS={frame.pts} 找不到对应随机访问点")
                image = frame.to_ndarray(format="bgr24")
                if image.shape[1] != config.width or image.shape[0] != config.height:
                    import cv2

                    image = cv2.resize(image, (config.width, config.height))
                image = np.ascontiguousarray(image)
                while True:
                    try:
                        slot_id = free_slots.get(timeout=0.1)
                        break
                    except queue.Empty:
                        continue
                slot = shared_slots[slot_id]
                image_view = memoryview(image).cast("B")
                if image_view.nbytes != frame_bytes:
                    raise VideoStorageError(f"IPC 帧大小不符：{image_view.nbytes} != {frame_bytes}")
                slot.buf[:frame_bytes] = image_view
                frame_msg: dict[str, Any] = {
                    "kind": "frame",
                    "timestamp": frame_timestamp,
                    "producer_monotonic_ns": frame_monotonic_ns,
                    "slot": slot_id,
                    "pts": int(frame.pts),
                    "time_base_num": frame_tb.numerator,
                    "time_base_den": frame_tb.denominator,
                    "segment": segment,
                    "random_access_dts": access[1],
                    "random_access_time_base_num": access[2],
                    "random_access_time_base_den": access[3],
                }
                if committed_time is not None:
                    frame_msg["committed_time_num"] = committed_time.numerator
                    frame_msg["committed_time_den"] = committed_time.denominator
                if not send(frame_msg):
                    return
    except Exception as exc:  # noqa: BLE001 - 子进程需把所有失败传回父进程
        if isinstance(exc, VideoStorageError):
            error_kind = "fatal"
        error_message = f"{type(exc).__name__}: {exc}"
    finally:
        segment_state: str | None = None
        if writer is not None:
            try:
                complete = segment_sealable and error_kind not in {"storage", "fatal"}
                writer.close(complete=complete)
                segment_state = (
                    SegmentState.CLOSED_COMPLETE.value
                    if complete
                    else SegmentState.CLOSED_INCOMPLETE.value
                )
            except Exception as exc:  # noqa: BLE001 - 清理错误必须经 IPC 报告
                error_kind = "storage"
                error_message = f"TS close: {type(exc).__name__}: {exc}"
                segment_state = SegmentState.FAILED.value
        if container is not None:
            try:
                container.close()
            except Exception as exc:  # noqa: BLE001 - 输入容器关闭错误需报告
                if error_message is None:
                    error_kind = "network"
                    error_message = f"input close: {type(exc).__name__}: {exc}"
        for slot in shared_slots:
            slot.close()
        if error_message is not None:
            message = {
                "kind": (
                    "storage_error"
                    if error_kind == "storage"
                    else "fatal_error"
                    if error_kind == "fatal"
                    else "stream_error"
                ),
                "error": error_message,
            }
            if segment_state is not None:
                message["segment_state"] = segment_state
            send(message)
        elif finished:
            message = {"kind": "eof"}
            if segment_state is not None:
                message["segment_state"] = segment_state
            send(message)


# ----------------------------------------------------------------------
# 拉流源
# ----------------------------------------------------------------------
class Hm30VideoSource:
    """HM30 图传拉流源：后台线程收流，每帧交给 sink，实时读取看最新一帧。

    典型用法（每帧都留存，供侦查与事后分析）::

        config = VideoConfig(width=1280, height=720)
        buffer = AlignmentBuffer(capacity_for(30, 180))
        with Hm30VideoSource(config) as source:
            source.add_sink(AlignmentWriter(buffer, FrameTelemetryAligner(broker)))
            ...

    只想实时看画面时，用 :meth:`read` / :meth:`latest` 即可::

        frame = source.read(timeout=5.0)     # 阻塞到第一帧
        while frame is not None:
            results = model(frame.image)     # YOLO / OCR
            frame = source.read(timeout=1.0)

    地址或链路有问题时不会静默重试到底：第一帧都还没收到就失败时，错误会以
    错误日志 + ``stats.last_error`` 的形式明确抛出（含 ffmpeg 的输出），
    方便直接核对地址与网段。

    两条消费路径的取舍：``add_sink`` 注册的回调在采集线程里**逐帧**调用，
    一帧不落（这是"目标可能只出现一瞬"的要求）；而 :meth:`read` / :meth:`latest`
    是实时的，消费速度慢于帧率时中间帧会被覆盖并计入 :attr:`VideoStats.dropped`。
    """

    def __init__(
        self,
        config: VideoConfig | None = None,
        *,
        segment_lifecycle: SegmentLifecycle | None = None,
    ) -> None:
        self._config = (config or VideoConfig()).validated()
        self._condition = threading.Condition(threading.RLock())
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._sinks: list[Callable[[VideoFrame], None]] = []

        self._frame: VideoFrame | None = None
        self._delivered = 0
        self._counter = 0
        self._stats = VideoStats()
        self._fps_timestamps: deque[float] = deque(maxlen=8192)

        self._process: subprocess.Popen | None = None
        self._ffmpeg_executable: str | None = None
        self._recording_directory: Path | None = None
        self._segment_counter = 0
        self._ts_writer = None
        self._container: Any = None
        self._container_lock = threading.Lock()
        self._pyav_process: Any = None
        self._pyav_messages: Any = None
        self._pyav_stop_event: Any = None
        self._pyav_shared_slots: list[SharedMemory] = []
        self._segment_lifecycle = segment_lifecycle or SegmentLifecycle()
        self._segment_lifecycle_bound = segment_lifecycle is not None
        self._active_pyav_segment: int | None = None

    # ------------------------------------------------------------------
    # 每帧回调（sink）
    # ------------------------------------------------------------------
    def add_sink(self, sink: Callable[[VideoFrame], None]) -> None:
        """注册一个**每帧**回调，用于"一帧都不能丢"的消费者。

        回调在采集线程里同步执行，所以必须快——它每慢一毫秒，就直接吃掉一毫秒的
        拉流预算（缓冲写入只做一次 JPEG 编码，几毫秒，可以接受；写磁盘之类请自己
        再开线程或队列）。回调抛异常只记日志，不会打断拉流。

        典型用法是挂 :class:`airdrop.video.buffer.AlignmentWriter`，把每帧连同拍摄时刻的
        遥测写进环形缓冲。
        """
        sink_buffer = getattr(sink, "_buffer", None)
        lifecycle = getattr(sink_buffer, "segment_lifecycle", None)
        if isinstance(lifecycle, SegmentLifecycle):
            self.bind_segment_lifecycle(lifecycle)
        with self._condition:
            if sink not in self._sinks:
                self._sinks.append(sink)

    def remove_sink(self, sink: Callable[[VideoFrame], None]) -> None:
        with self._condition:
            if sink in self._sinks:
                self._sinks.remove(sink)

    @property
    def sinks(self) -> tuple[Callable[[VideoFrame], None], ...]:
        with self._condition:
            return tuple(self._sinks)

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    @property
    def config(self) -> VideoConfig:
        return self._config

    @property
    def segment_lifecycle(self) -> SegmentLifecycle:
        """实验采集端共享的进程内分段状态。"""
        return self._segment_lifecycle

    def bind_segment_lifecycle(self, lifecycle: SegmentLifecycle) -> None:
        """在采集启动前与冷缓冲绑定同一分段状态。"""
        if lifecycle is self._segment_lifecycle:
            self._segment_lifecycle_bound = True
            return
        if self._segment_lifecycle_bound or self.running or self._segment_counter:
            raise RuntimeError("多个 sink 必须共享同一视频分段生命周期")
        self._segment_lifecycle = lifecycle
        self._segment_lifecycle_bound = True

    def validate_pyav_config(self) -> None:
        """在启动外部资源前检查 PyAV 采集支持的输入选项。"""
        _pyav_input_options(self._config)

    def set_recording_directory(self, directory: str | Path) -> None:
        """设置 remux TS 的目录，必须在拉流开始前调用。"""
        if self.running:
            raise RuntimeError("视频 TS 目录必须在采集启动前设置")
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        self._recording_directory = path

    @property
    def recording_directory(self) -> Path | None:
        """已绑定的 TS 分段写入目录；未启用 TS 写入时为 None。"""
        return self._recording_directory

    @property
    def stats(self) -> VideoStats:
        """返回统计快照。"""
        with self._condition:
            return replace(self._stats)

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def raise_if_failed(self) -> None:
        """在任务心跳中报告致命采集或逐帧 sink 故障。"""
        with self._condition:
            if self._stats.sink_errors:
                raise VideoStorageError(
                    self._stats.last_error or "图传帧 sink 发生错误，逐帧数据可能不完整"
                )
            thread = self._thread
            stopped_unexpectedly = (
                thread is not None and not thread.is_alive() and not self._stop_event.is_set()
            )
            error = self._stats.last_error
        if stopped_unexpectedly:
            raise VideoStorageError(error or "视频采集线程意外退出")

    def start(self) -> "Hm30VideoSource":
        """启动后台拉流线程（幂等）。"""
        with self._condition:
            if self.running:
                return self
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._run,
                name="hm30-video",
                daemon=True,
            )
            self._set_state("connecting")
            self._thread.start()
        return self

    def stop(self, timeout: float = 5.0) -> None:
        """停止拉流并等待线程退出（幂等）。"""
        deadline = time.monotonic() + max(timeout, 0.0)
        self._stop_event.set()
        # 兜底：ffmpeg 若正卡在管道读上，杀掉子进程即可让 readinto 立刻收到 EOF。
        process = self._process
        if process is not None and process.poll() is None:
            process.terminate()
        pyav_stop_event = self._pyav_stop_event
        if pyav_stop_event is not None:
            pyav_stop_event.set()
        with self._condition:
            self._condition.notify_all()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(max(0.0, deadline - time.monotonic()))
            if thread.is_alive():
                self._terminate_pyav_worker(max(0.0, deadline - time.monotonic()))
                thread.join(max(0.0, deadline - time.monotonic()))
            if thread.is_alive():
                # 线程没停就别清引用：清了 running 会变 False、start() 会再起
                # 一个线程，两个拉流线程抢同一个源。保留引用，稍后可再调 stop()。
                LOGGER.warning("拉流线程未在 %.1fs 内退出，保留引用（可再调 stop()）", timeout)
                return
        if self._pyav_process is not None:
            self._terminate_pyav_worker(max(0.0, deadline - time.monotonic()))
        self._thread = None
        self._set_state("stopped")

    def _terminate_pyav_worker(self, timeout: float) -> None:
        """强制结束可能卡在网络或 FFmpeg 调用中的 PyAV 工作进程。"""
        worker = self._pyav_process
        if worker is None:
            return
        stop_event = self._pyav_stop_event
        if stop_event is not None:
            stop_event.set()
        if worker.is_alive():
            worker.terminate()
            worker.join(max(0.0, timeout))
        if worker.is_alive():
            LOGGER.error("PyAV 工作进程未能在期限内退出，保留进程与队列引用")
            segment = self._active_pyav_segment
            if (
                segment is not None
                and self._segment_lifecycle.snapshot(segment).state is SegmentState.ACTIVE
            ):
                self._segment_lifecycle.failed(segment)
            raise VideoStorageError("PyAV 工作进程未能确认退出；禁止继续启动新的采集周期")
        segment = self._active_pyav_segment
        if segment is not None:
            self._settle_segment_after_worker_exit(segment, alive=False)
        self._pyav_process = None
        messages = self._pyav_messages
        self._pyav_messages = None
        self._pyav_stop_event = None
        self._active_pyav_segment = None
        if messages is not None:
            messages.cancel_join_thread()
            messages.close()
        self._release_pyav_shared_slots()

    def _settle_segment_after_worker_exit(self, segment: int, *, alive: bool) -> None:
        """子进程无终止消息时以完成标记兜底，不猜测写入器是否仍活动。"""
        if self._segment_lifecycle.snapshot(segment).state is not SegmentState.ACTIVE:
            return
        if alive:
            self._segment_lifecycle.failed(segment)
            return
        complete = (
            self._recording_directory is not None
            and (self._recording_directory / f"segment-{segment:06d}.complete").is_file()
        )
        self._segment_lifecycle.closed(segment, complete=complete)

    def _release_pyav_shared_slots(self) -> None:
        """在工作进程退出后释放父端拥有的固定帧槽。"""
        slots, self._pyav_shared_slots = self._pyav_shared_slots, []
        for slot in slots:
            try:
                slot.close()
            except Exception:
                LOGGER.exception("关闭 PyAV 共享内存帧槽失败")
            try:
                slot.unlink()
            except FileNotFoundError:
                pass
            except Exception:
                LOGGER.exception("删除 PyAV 共享内存帧槽失败")

    def __enter__(self) -> "Hm30VideoSource":
        return self.start()

    def __exit__(self, *exc_info: object) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # 读取接口
    # ------------------------------------------------------------------
    def latest(self) -> VideoFrame | None:
        """立刻返回最新一帧，没有则不等待（实时路径，同样会跳过中间帧）。"""
        with self._condition:
            return self._frame

    def read(self, timeout: float | None = None) -> VideoFrame | None:
        """阻塞等待**下一帧新画面**；超时或已停止返回 None。

        这是**实时**读取：只关心当前画面，跟进不上就会丢掉中间帧（计入
        :attr:`VideoStats.dropped`）。要保证一帧不丢，就别在这条路径上做重活——
        用 :meth:`add_sink` 挂一个 :class:`airdrop.video.buffer.AlignmentWriter`，
        历史都在缓冲里。

        面向单消费者：内部记录投递进度。
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            while True:
                frame = self._frame
                if frame is not None and frame.index > self._delivered:
                    # 中间被超过的帧就是"消费者没看上"的帧，如实累计
                    self._stats.dropped += frame.index - self._delivered - 1
                    self._delivered = frame.index
                    return frame
                if self._stop_event.is_set():
                    return None
                remaining = None
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return None
                self._condition.wait(remaining)

    def wait_ready(self, timeout: float = 10.0) -> bool:
        """阻塞直到拿到第一帧；超时返回 False。"""
        return self.read(timeout=timeout) is not None

    def iter_frames(self, timeout: float | None = 1.0) -> Iterator[VideoFrame]:
        """迭代新帧；连续 ``timeout`` 秒没有新画面就结束迭代。

        ``timeout=None`` 表示一直等到 :meth:`stop`。
        """
        while not self._stop_event.is_set():
            frame = self.read(timeout=timeout)
            if frame is None:
                return
            yield frame

    # ------------------------------------------------------------------
    # 后台线程
    # ------------------------------------------------------------------
    def _run(self) -> None:
        delay = self._config.reconnect_delay
        while not self._stop_event.is_set():
            try:
                self._stream_pyav_process()
                delay = self._config.reconnect_delay
            except VideoStorageError as exc:
                self._fail(str(exc))
                break
            except Exception as exc:
                LOGGER.exception("图传拉流异常")
                self._fail(str(exc))
            if self._stop_event.is_set():
                break
            with self._condition:
                self._stats.reconnects += 1
            self._set_state("reconnecting")
            if self._stop_event.wait(delay):
                break
            delay = min(delay * 2, self._config.max_reconnect_delay)
        self._set_state("stopped")

    def _stream_pyav_process(self) -> None:  # noqa: PLR0912, PLR0915
        """显式实验入口：用独立 PyAV 进程输出有界的帧消息。"""
        existing_worker = self._pyav_process
        if existing_worker is not None:
            if existing_worker.is_alive():
                raise VideoStorageError("上一 PyAV 工作进程仍存活；禁止覆盖资源或启动新分段")
            self._terminate_pyav_worker(0.0)
        _pyav_input_options(self._config)
        context: Any = None
        messages: Any = None
        free_slots: Any = None
        stop_event: Any = None
        slots: list[SharedMemory] = []
        worker: Any = None
        normal_exit = False
        segment: int | None = None
        segment_registered = False
        try:
            context = multiprocessing.get_context("spawn")
            messages = context.Queue(maxsize=2)
            free_slots = context.Queue(maxsize=2)
            stop_event = context.Event()
            self._segment_counter += 1
            segment = self._segment_counter
            self._active_pyav_segment = segment
            directory = (
                None if self._recording_directory is None else str(self._recording_directory)
            )
            frame_bytes = self._config.width * self._config.height * 3
            for _ in range(2):
                slots.append(SharedMemory(create=True, size=frame_bytes))
                self._pyav_shared_slots = slots
            for slot_id in range(len(slots)):
                free_slots.put(slot_id)
            worker = context.Process(
                target=_pyav_capture_worker,
                args=(
                    self._config,
                    directory,
                    segment,
                    messages,
                    free_slots,
                    tuple(slot.name for slot in slots),
                    frame_bytes,
                    stop_event,
                ),
                name=f"pyav-capture-{segment}",
                daemon=True,
            )
            self._pyav_process = worker
            self._pyav_messages = messages
            self._pyav_stop_event = stop_event
            self._set_state("connecting")
            worker.start()
            while True:
                try:
                    message = messages.get(timeout=0.1)
                except queue.Empty:
                    if not worker.is_alive():
                        raise RuntimeError(
                            f"PyAV 工作进程意外退出，exitcode={worker.exitcode}"
                        ) from None
                    continue
                kind = message.get("kind")
                if kind == "segment_opened":
                    if message.get("segment") != segment:
                        raise VideoStorageError("PyAV IPC 分段编号不匹配")
                    self._segment_lifecycle.opened(segment)
                    segment_registered = True
                elif kind == "ready":
                    self._set_state("streaming")
                    skipped = int(message.get("startup_packets_skipped", 0))
                    if skipped:
                        LOGGER.warning("PyAV 启动时跳过了关键帧前的 %d 个视频包", skipped)
                elif kind == "frame":
                    if message.get("segment") != segment:
                        raise VideoStorageError("PyAV IPC 帧分段编号不匹配")
                    if segment_registered:
                        from fractions import Fraction

                        c_num = message.get("committed_time_num")
                        c_den = message.get("committed_time_den")
                        committed = (
                            Fraction(int(c_num), int(c_den))
                            if c_num is not None and c_den is not None and int(c_den) > 0
                            else None
                        )
                        self._segment_lifecycle.progress(segment, committed_time=committed)
                    slot_id = int(message["slot"])
                    if not 0 <= slot_id < len(slots):
                        raise VideoStorageError(f"PyAV IPC 返回非法共享槽位: {slot_id}")
                    image = np.ndarray(
                        (self._config.height, self._config.width, 3),
                        dtype=np.uint8,
                        buffer=slots[slot_id].buf,
                    ).copy()
                    received_monotonic_ns = time.monotonic_ns()
                    decoded_timestamp = float(message["timestamp"])
                    queue_age = max(
                        (received_monotonic_ns - int(message["producer_monotonic_ns"]))
                        / 1_000_000_000,
                        0.0,
                    )
                    with self._condition:
                        self._stats.pyav_queue_age_s = queue_age
                        self._stats.pyav_queue_age_max_s = max(
                            self._stats.pyav_queue_age_max_s, queue_age
                        )
                    try:
                        self._publish(
                            image,
                            received_timestamp=decoded_timestamp,
                            media_pts=message["pts"],
                            time_base_num=message["time_base_num"],
                            time_base_den=message["time_base_den"],
                            segment=message["segment"],
                            random_access_dts=message["random_access_dts"],
                            random_access_time_base_num=message["random_access_time_base_num"],
                            random_access_time_base_den=message["random_access_time_base_den"],
                            fail_on_sink_error=True,
                        )
                    finally:
                        free_slots.put(slot_id)
                elif kind in {"storage_error", "fatal_error", "stream_error", "eof"}:
                    if (
                        segment_registered
                        and self._segment_lifecycle.snapshot(segment).state is SegmentState.ACTIVE
                    ):
                        reported_state = message.get("segment_state")
                        if reported_state == SegmentState.CLOSED_COMPLETE.value:
                            self._segment_lifecycle.closed(segment, complete=True)
                        elif reported_state == SegmentState.CLOSED_INCOMPLETE.value:
                            self._segment_lifecycle.closed(segment, complete=False)
                        else:
                            self._segment_lifecycle.failed(
                                segment, error=str(message.get("error", "")) or None
                            )
                    if kind == "storage_error":
                        raise VideoStorageError(str(message.get("error", "未知 TS 写入错误")))
                    if kind == "fatal_error":
                        raise VideoStorageError(str(message.get("error", "视频帧映射失败")))
                    if kind == "stream_error":
                        raise RuntimeError(str(message.get("error", "未知输入流错误")))
                    worker.join(timeout=1.0)
                    normal_exit = not worker.is_alive()
                    if self._stop_event.is_set():
                        return
                    raise RuntimeError(f"输入流结束或中断（{self._config.url}）")
        finally:
            alive = False
            if stop_event is not None:
                stop_event.set()
            if worker is not None and worker.pid is not None:
                worker.join(timeout=1.0)
                if worker.is_alive():
                    worker.terminate()
                    worker.join(timeout=1.0)
                alive = worker.is_alive()
            if alive:
                LOGGER.error("PyAV 子进程未退出，保留进程与共享内存引用")
                if segment_registered and segment is not None:
                    self._settle_segment_after_worker_exit(segment, alive=True)
                raise VideoStorageError("PyAV 子进程未能确认退出；保留资源并禁止自动重连")
            else:
                if segment_registered and segment is not None:
                    self._settle_segment_after_worker_exit(segment, alive=False)
                self._pyav_process = None
                self._pyav_messages = None
                self._pyav_stop_event = None
                for q in (messages, free_slots):
                    if q is not None:
                        q.cancel_join_thread()
                        q.close()
                self._release_pyav_shared_slots()
                self._active_pyav_segment = None
            if normal_exit and self._stop_event.is_set():
                self._set_state("stopped")

    def _stream_pyav(self) -> None:
        """单路 demux 同时解码图像并复制压缩包到本连接周期的 TS。"""
        from fractions import Fraction

        import av

        from .cold import TsSegmentWriter

        config = self._config
        options = {
            "fflags": "nobuffer",
            "flags": "low_delay",
            "probesize": "500000",
            "analyzeduration": "0",
        }
        if _is_rtsp(config.url):
            options["rtsp_transport"] = config.transport
        timeout = max(float(config.read_timeout), 0.1)
        self._set_state("connecting")
        container = av.open(
            config.url,
            mode="r",
            options=options,
            timeout=(timeout, timeout),
        )
        writer = None
        with self._container_lock:
            self._container = container
        try:
            if not container.streams.video:
                raise RuntimeError(f"输入没有视频流: {config.url}")
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            self._segment_counter += 1
            segment = self._segment_counter
            if self._recording_directory is not None:
                try:
                    writer = TsSegmentWriter(
                        self._recording_directory, segment=segment, input_stream=stream
                    )
                except Exception as exc:
                    raise VideoStorageError(f"无法创建 TS 分段: {exc}") from exc
                self._ts_writer = writer
            random_access: list[tuple[Fraction, int, int, int, int]] = []
            self._set_state("streaming")
            for packet in container.demux(stream):
                if self._stop_event.is_set():
                    break
                packet_tb = packet.time_base
                if (
                    packet.size > 0
                    and packet.is_keyframe
                    and packet.pts is not None
                    and packet.dts is not None
                    and packet_tb is not None
                ):
                    tb = Fraction(packet_tb.numerator, packet_tb.denominator)
                    random_access.append(
                        (
                            Fraction(packet.pts) * tb,
                            packet.dts,
                            tb.numerator,
                            tb.denominator,
                            packet.pts,
                        )
                    )
                decoded = packet.decode()
                if writer is not None and packet.size > 0:
                    # 先保证压缩包已写入冷存储，再把对应解码帧交给对齐消费者。
                    # remux 失败时不得继续发布看似可用、实际无法回读的帧。
                    self._mux_pyav_packet(writer, packet)
                for frame in decoded:
                    self._publish_pyav_frame(frame, config, segment, random_access)
            if not self._stop_event.is_set():
                self._fail(f"输入流结束或中断（{config.url}）")
        finally:
            try:
                if writer is not None:
                    try:
                        writer.close()
                    except Exception as exc:
                        raise VideoStorageError(f"关闭 TS 分段失败: {exc}") from exc
            finally:
                self._ts_writer = None
                self._close_pyav_container(container)

    @staticmethod
    def _mux_pyav_packet(writer: Any, packet: Any) -> None:
        """写入压缩包失败时中止当前持久化任务。"""
        try:
            writer.mux(packet)
        except Exception as exc:
            raise VideoStorageError(f"TS remux 写入失败: {exc}") from exc

    def _publish_pyav_frame(
        self,
        frame: Any,
        config: VideoConfig,
        segment: int,
        random_access: list[tuple[Any, int, int, int, int]],
    ) -> None:
        """验证帧的媒体时间映射后再呈现给下游。"""
        from fractions import Fraction

        import cv2

        if frame.pts is None or frame.time_base is None:
            raise VideoStorageError("解码帧缺少 PTS 或 time_base，无法映射业务帧与 TS")
        frame_tb = Fraction(frame.time_base.numerator, frame.time_base.denominator)
        frame_time = Fraction(frame.pts) * frame_tb
        access = next(
            (item for item in reversed(random_access) if item[0] <= frame_time),
            None,
        )
        if access is None:
            raise VideoStorageError(f"解码帧 PTS={frame.pts} 在本连接周期内找不到对应随机访问点")
        image = frame.to_ndarray(format="bgr24")
        if image.shape[1] != config.width or image.shape[0] != config.height:
            image = cv2.resize(image, (config.width, config.height))
        self._publish(
            image,
            media_pts=frame.pts,
            time_base_num=frame_tb.numerator,
            time_base_den=frame_tb.denominator,
            segment=segment,
            random_access_dts=access[1],
            random_access_time_base_num=access[2],
            random_access_time_base_den=access[3],
            fail_on_sink_error=True,
        )

    def _interrupt_pyav_read(self) -> None:
        """停止时关闭活动输入容器，尽可能中断阻塞中的 PyAV 读取。"""
        with self._container_lock:
            container = self._container
            self._container = None
        if container is not None:
            try:
                container.close()
            except Exception:
                LOGGER.exception("关闭 PyAV 输入容器以中断读取失败")

    def _close_pyav_container(self, container: Any) -> None:
        """关闭输入容器；与 stop() 并发时避免重复释放。"""
        with self._container_lock:
            if self._container is not container:
                return
            self._container = None
        container.close()

    def _stream_ffmpeg(self) -> None:
        executable = self._ffmpeg_executable
        if executable is None:  # pragma: no cover - _run 已保证
            self._fail("找不到 ffmpeg 可执行文件")
            return
        width, height = self._config.width, self._config.height

        command = self._build_ffmpeg_command(executable, width, height)
        LOGGER.debug("ffmpeg 命令: %s", " ".join(command))
        self._set_state("connecting")
        # stderr 落到临时文件而不是管道：管道写满会让 ffmpeg 阻塞在日志上、
        # 连带卡死解码；临时文件既能保留错误信息，又不会反压。
        # SIM115：句柄交给 _terminate(process, stderr_file) 关闭，不能在这里 with 掉。
        stderr_file = tempfile.TemporaryFile()  # noqa: SIM115
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=stderr_file,
            stdin=subprocess.DEVNULL,
        )
        if process.stdout is None:  # pragma: no cover - PIPE 已显式指定
            _terminate(process, stderr_file)
            self._fail("无法创建 ffmpeg 输出管道")
            return
        frame_bytes = width * height * 3
        self._process = process
        try:
            buffer = bytearray(frame_bytes)
            self._set_state("streaming")
            while not self._stop_event.is_set():
                if not _read_exact(process.stdout, buffer):
                    self._fail(
                        f"拉流失败或中断（{self._config.url}）："
                        f"{_read_stderr(process, stderr_file)}"
                    )
                    return
                image = np.frombuffer(buffer, dtype=np.uint8).reshape(height, width, 3)
                # 注意：image 是 buffer 的视图（零拷贝），下一轮 readinto 会覆盖
                # 它。要保留画面必须 copy —— 见 VideoFrame.copy()。
                self._publish(image)
        finally:
            _terminate(process, stderr_file)
            self._process = None

    def _build_ffmpeg_command(self, executable: str, width: int, height: int) -> list[str]:
        config = self._config
        # -hide_banner -nostdin：别让 ffmpeg 抢标准输入，日志也压到最少
        command = [executable, "-hide_banner", "-nostdin", "-loglevel", "error"]
        # 输入端：把探测与缓冲全部压掉，这是低延迟的关键
        command += ["-fflags", "nobuffer", "-flags", "low_delay"]
        command += ["-probesize", "500000", "-analyzeduration", "0"]
        if config.ffmpeg_decoder:
            command += ["-c:v", config.ffmpeg_decoder]
        if _is_rtsp(config.url):
            command += ["-rtsp_transport", config.transport]
        # 断流判定的关键 flag。实测（ffmpeg 7.1，UDP 输入）：
        #   -rw_timeout 2s  → 12 秒内不退出，拉流线程永久卡死
        #   -timeout   2s   → 2.7 秒退出 ✓
        # -timeout 是 udp 协议与 rtsp 解复用器共有的 socket I/O 超时（微秒），
        # 两种流都能覆盖。**但它只属于网络输入**：本地文件类输入（SITL 演练用的
        # SDP 文件）没有这个选项，加在前面会让 ffmpeg 直接
        # "Option timeout not found" 打不开输入——所以按 URL 形态决定加不加。
        if "://" in config.url:
            command += ["-timeout", str(int(config.read_timeout * 1_000_000))]
        command += list(config.extra_input_args)
        command += ["-i", config.url]
        # 输出端：直接吐 bgr24 裸帧，numpy 零解析接管
        command += ["-an", "-sn", "-dn"]
        command += ["-vf", f"scale={width}:{height}"]
        command += list(config.extra_output_args)
        command += ["-pix_fmt", "bgr24", "-f", "rawvideo", "-"]
        return command

    # ------------------------------------------------------------------
    # 状态与发布
    # ------------------------------------------------------------------
    def _publish(
        self,
        image: np.ndarray,
        *,
        media_pts: int | None = None,
        time_base_num: int | None = None,
        time_base_den: int | None = None,
        segment: int = 0,
        random_access_dts: int | None = None,
        random_access_time_base_num: int | None = None,
        random_access_time_base_den: int | None = None,
        fail_on_sink_error: bool = False,
        received_timestamp: float | None = None,
    ) -> None:
        now = time.time() if received_timestamp is None else received_timestamp
        with self._condition:
            self._counter += 1
            frame = VideoFrame(
                index=self._counter,
                image=image,
                timestamp=now,
                lag=self._config.telemetry_lag,
                media_pts=media_pts,
                time_base_num=time_base_num,
                time_base_den=time_base_den,
                segment=segment,
                random_access_dts=random_access_dts,
                random_access_time_base_num=random_access_time_base_num,
                random_access_time_base_den=random_access_time_base_den,
            )
            self._frame = frame
            # 用最近一秒的实际发布时刻，避免启动突发持续污染固定窗口。
            fps_now = time.monotonic()
            self._fps_timestamps.append(fps_now)
            while self._fps_timestamps and self._fps_timestamps[0] < fps_now - 1.0:
                self._fps_timestamps.popleft()
            elapsed = fps_now - self._fps_timestamps[0]
            self._stats.fps = (len(self._fps_timestamps) - 1) / elapsed if elapsed >= 0.5 else 0.0
            self._stats.frames += 1
            self._stats.state = "streaming"
            sinks = tuple(self._sinks)
            self._condition.notify_all()

        # 逐帧投递给 sink，且放在锁外——别让编码/落盘拖住 read()/latest()。
        # "每一帧都被留存"的保证就在这里：frame 是复用视图，sink 必须在这
        # 一帧还有效时把它接管（缓冲写入会拷贝，见 AlignmentWriter）。
        for sink in sinks:
            try:
                sink(frame)
            except Exception as exc:
                with self._condition:
                    self._stats.sink_errors += 1
                    self._stats.last_error = f"图传帧 sink 执行失败: {type(exc).__name__}: {exc}"
                LOGGER.exception("图传帧 sink 执行失败")
                if fail_on_sink_error:
                    raise VideoStorageError(self._stats.last_error) from exc

    def _set_state(self, state: str) -> None:
        with self._condition:
            self._stats.state = state
            self._condition.notify_all()

    def _fail(self, message: str) -> None:
        """记录失败原因并提示。

        一帧都没收到就失败（地址写错、链路不通、配置不对）属于"用不了"，
        用错误级别并原样给出 ffmpeg 的输出；已经跑起来之后的断流则是
        "暂时中断"，警告级别即可——重连逻辑会继续尝试。
        """
        with self._condition:
            never_streamed = self._stats.frames == 0
            self._stats.last_error = message
            self._condition.notify_all()
        if never_streamed:
            LOGGER.error("图传不可用：%s", message)
        else:
            LOGGER.warning("图传中断：%s", message)


# ----------------------------------------------------------------------
# 管道工具
# ----------------------------------------------------------------------
def _read_exact(stream, buffer: bytearray) -> bool:
    """把管道读满 ``len(buffer)`` 字节；EOF 返回 False。"""
    view = memoryview(buffer)
    filled = 0
    total = len(buffer)
    while filled < total:
        chunk = stream.readinto(view[filled:])
        if not chunk:
            return False
        filled += chunk
    return True


def _read_stderr(process: subprocess.Popen, stderr_file) -> str:
    """取回 ffmpeg 的错误输出（stderr 落临时文件，不会有管道反压）。"""
    try:
        stderr_file.seek(0)
        text = stderr_file.read().decode("utf-8", "replace").strip()
    except Exception:  # noqa: BLE001 - 日志读不出来只影响报错信息
        return "无法读取 ffmpeg 日志"
    return text or f"ffmpeg 异常退出，returncode={process.poll()}"


def _terminate(process: subprocess.Popen, stderr_file=None) -> None:
    """确保 ffmpeg 子进程被回收（Windows 上 terminate 后仍要 wait）。"""
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2.0)
    if process.stdout is not None:
        # 关闭失败不能影响重连/停止流程，所以刻意吞掉异常
        with contextlib.suppress(Exception):
            process.stdout.close()
    if stderr_file is not None:
        with contextlib.suppress(Exception):
            stderr_file.close()


def open_hm30_video(
    config: VideoConfig | None = None, *, autostart: bool = True
) -> Hm30VideoSource:
    """创建拉流源；``autostart=False`` 时需自行调用 :meth:`Hm30VideoSource.start`。"""
    source = Hm30VideoSource(config)
    if autostart:
        source.start()
    return source
