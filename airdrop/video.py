'思翼 HM30 图传拉流。\n\nHM30 的地面端并不"推"视频给上位机，而是把机载以太网（``192.168.144.0/24``）\n**透明桥接**到 LAN 口 / 内置 WiFi / USB WiFi 上：地面端 LAN 口插到电脑，电脑\n配一个同网段地址，就能直接访问机载 IP 相机。官方 FAQ 明确写着\n"As long as the IP camera outputs RTSP video stream and can connect the\nEthernet port, it will work with HM30"。\n\n因此这里的"拉流"就是一条 RTSP：默认 ``rtsp://192.168.144.25:8554/main.264``\n（SIYI 云台/相机系列的约定地址）。若用的是 HDMI-Ethernet 输入转换模块或第三方\nIP 相机，地址不同，请用 :func:`probe_stream` 或用本模块的候选探测先确认。\n\n两条后端\n--------\n``ffmpeg``（**默认**，走 imageio-ffmpeg 自带的二进制）\n    ``subprocess`` 拉起 ffmpeg，从管道读 ``rawvideo/bgr24`` 裸帧。能压掉\n    FFmpeg 的探测与缓冲（``-fflags nobuffer -flags low_delay``），断流能被\n    ``-rw_timeout`` 及时判定，子进程还能被父进程直接杀掉，因此拉流线程永远\n    停得下来。\n``opencv``\n    ``cv2.VideoCapture`` + OpenCV 自带的 FFmpeg。零额外进程、代码最短，\n    但**延迟不可控**：OpenCV 5.0 已移除 ``OPENCV_FFMPEG_CAPTURE_OPTIONS``\n    （实测 cv2.pyd 里连这个字符串都不存在，设置它毫无效果），也没有其它受支持\n    的途径传入 ``nobuffer``/``low_delay``。\n\n语义与 :class:`~airdrop.broker.TelemetryBroker` 一致：后台线程**只保留最新一帧**。\n图传引导必须基于"当前画面"，慢消费者应该丢掉旧帧而不是积压——积压等于给无人机\n看历史照片。因此 :meth:`Hm30VideoSource.read` 返回的是"下一帧新画面"，\n:meth:`Hm30VideoSource.latest` 返回"此刻最新画面"，二者都不排队。'

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass, replace
from typing import Iterator

# OPENCV_FFMPEG_CAPTURE_OPTIONS 必须尽量早地写进环境：OpenCV 的 FFmpeg 后端在
# 首次打开串流时读取它。放在 import cv2 之前，是唯一 100% 生效的位置。
_DEFAULT_OPENCV_FFMPEG_OPTIONS = (
    "rtsp_transport;udp"
    "|fflags;nobuffer"
    "|flags;low_delay"
    "|probesize;500000"
    "|analyzeduration;0"
    "|max_delay;500000"
    "|reorder_queue_size;0"
    "|rw_timeout;3000000"
)
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", _DEFAULT_OPENCV_FFMPEG_OPTIONS)

import cv2  # noqa: E402 - 必须在设置捕获选项之后导入
import numpy as np  # noqa: E402

LOGGER = logging.getLogger(__name__)

# HM30 常见地址（地面端与机载相机同处 192.168.144.0/24）
HM30_CAMERA_IP = "192.168.144.25"
HM30_GROUND_IP = "192.168.144.12"
# SIYI 相机/云台默认 RTSP 主码流
HM30_DEFAULT_RTSP = f"rtsp://{HM30_CAMERA_IP}:8554/main.264"

SUPPORTED_BACKENDS = frozenset({"auto", "opencv", "ffmpeg"})
SUPPORTED_TRANSPORTS = frozenset({"udp", "tcp"})

# 图传链路固定延时的默认估计（秒）：从"机载相机曝光"到"本进程从管道里切出
# 一帧"之间的处理时间，含 H.264 编码、无线传输、解码与管道缓冲。本机在
# ffmpeg 后端实测 ≈0.15s（见模块开头的实测数据）。
# 拿到一帧后用它反推该画面真正的拍摄时刻（VideoFrame.capture_timestamp），
# 再按那一刻取遥测，见 :mod:`airdrop.alignment`。
# 注意：改用 OpenCV 后端做对照时链路延时约 0.57s，这个值要同步调大，否则
# 对齐结果会系统地偏向"未来"。
DEFAULT_TELEMETRY_LAG = 0.15


# ----------------------------------------------------------------------
# 配置与数据结构
# ----------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class VideoConfig:
    """拉流配置。

    ``backend="auto"`` 表示优先 ffmpeg（延迟可控），找不到 ffmpeg 才退回 opencv。
    ``width``/``height`` 为 0 时使用源分辨率；ffmpeg 后端必须先知道输出尺寸才能
    从管道切帧，因此为 0 时会在启动前用 OpenCV 探测一次。
    """

    url: str = HM30_DEFAULT_RTSP
    backend: str = "auto"
    transport: str = "udp"
    width: int = 0
    height: int = 0
    open_timeout: float = 5.0
    read_timeout: float = 3.0
    # 图传链路的固定延时（秒），会被写进每一帧（VideoFrame.lag）。默认
    # DEFAULT_TELEMETRY_LAG=0.15；用 OpenCV 后端做对照时链路延时约 0.57s，
    # 需同步调大。0 表示不做延时补偿（帧的拍摄时刻 = 收到时刻）。
    telemetry_lag: float = DEFAULT_TELEMETRY_LAG
    reconnect_delay: float = 1.0
    max_reconnect_delay: float = 10.0
    # 覆盖 OPENCV_FFMPEG_CAPTURE_OPTIONS。注意：OpenCV 5.0 已移除该环境变量，
    # 本机实测设了等于没设；保留只为兼容还在 OpenCV 4.x 的环境。
    opencv_ffmpeg_options: str | None = None
    # ffmpeg 可执行文件；None 表示用 imageio-ffmpeg 自带的那份
    ffmpeg_executable: str | None = None
    # 指定解码器，例如 "h264_cuvid" / "hevc_cuvid"（NVDEC 硬解）
    ffmpeg_decoder: str | None = None
    # 追加到 ffmpeg 输入参数（-i 之前）的原始参数
    extra_input_args: tuple[str, ...] = ()
    # 追加到 ffmpeg 输出参数（-i 之后）的原始参数
    extra_output_args: tuple[str, ...] = ()

    def validated(self) -> "VideoConfig":
        if self.backend not in SUPPORTED_BACKENDS:
            raise ValueError(
                f"不支持的后端: {self.backend}，可选 {sorted(SUPPORTED_BACKENDS)}"
            )
        if self.transport not in SUPPORTED_TRANSPORTS:
            raise ValueError(
                f"不支持的传输层: {self.transport}，可选 "
                f"{sorted(SUPPORTED_TRANSPORTS)}"
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


@dataclass(frozen=True, slots=True)
class StreamInfo:
    """一次探测的结果。"""

    url: str
    width: int
    height: int
    fps: float
    open_seconds: float
    first_frame_seconds: float


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
    except Exception:  # noqa: BLE001 - 任何失败都退回 PATH 查找
        return shutil.which("ffmpeg")


def _opencv_ffmpeg_options(config: VideoConfig) -> str:
    'OpenCV 4.x 的 FFmpeg 捕获选项串（只负责低延迟 flag）。'
    if config.opencv_ffmpeg_options is not None:
        return config.opencv_ffmpeg_options
    return (
        f"rtsp_transport;{config.transport}"
        "|fflags;nobuffer"
        "|flags;low_delay"
        "|probesize;500000"
        "|analyzeduration;0"
        "|max_delay;500000"
        "|reorder_queue_size;0"
    )


def _is_rtsp(url: str) -> bool:
    return url.lower().startswith("rtsp")


def probe_stream(
    url: str = HM30_DEFAULT_RTSP,
    *,
    transport: str = "udp",
    timeout: float = 5.0,
) -> StreamInfo | None:
    """探测一条流是否能打开，并返回分辨率等基本信息。

    用 OpenCV 后端实现，因此只依赖 opencv-python，不需要额外进程。
    注意：本函数会临时改写进程级的 ``OPENCV_FFMPEG_CAPTURE_OPTIONS``
    （对 OpenCV 5.0 无效，见 :func:`_opencv_ffmpeg_options`），
    因此不要在拉流线程运行期间调用。
    """
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = _opencv_ffmpeg_options(
        VideoConfig(transport=transport, read_timeout=timeout)
    )
    capture = cv2.VideoCapture()
    started = time.monotonic()
    try:
        # 与 _stream_opencv 一致：超时只能走 open() 的 params
        if not capture.open(
            url,
            cv2.CAP_FFMPEG,
            [
                cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
                int(timeout * 1000),
                cv2.CAP_PROP_READ_TIMEOUT_MSEC,
                int(timeout * 1000),
            ],
        ):
            return None
        opened = time.monotonic()
        ok, image = capture.read()
        if not ok or image is None:
            return None
        first = time.monotonic()
        return StreamInfo(
            url=url,
            width=int(image.shape[1]),
            height=int(image.shape[0]),
            fps=float(capture.get(cv2.CAP_PROP_FPS) or 0.0),
            open_seconds=opened - started,
            first_frame_seconds=first - started,
        )
    finally:
        capture.release()


def probe_candidates(
    urls: list[str] | tuple[str, ...] | None = None,
    *,
    transports: tuple[str, ...] = ("udp",),
    timeout: float = 5.0,
) -> list[StreamInfo]:
    """按顺序探测候选地址，返回所有成功的流。

    HM30 现场最常见的坑就是"地址记错了"——相机、HDMI 输入转换模块、地面端
    自身的地址各不相同。先用它把正确的地址筛出来，再谈调优。
    """
    if urls is None:
        urls = [HM30_DEFAULT_RTSP]
    found: list[StreamInfo] = []
    for url in urls:
        for transport in transports:
            if not _is_rtsp(url) and transport != transports[0]:
                continue  # 非 RTSP 流没有传输层可选，只试一次
            info = probe_stream(url, transport=transport, timeout=timeout)
            if info is not None:
                found.append(info)
    return found


# ----------------------------------------------------------------------
# 拉流源
# ----------------------------------------------------------------------
class Hm30VideoSource:
    """HM30 图传拉流源：后台线程收流，对外只暴露"最新一帧"。

    典型用法::

        with Hm30VideoSource(VideoConfig(url=HM30_DEFAULT_RTSP)) as source:
            frame = source.read(timeout=5.0)   # 阻塞到第一帧
            while frame is not None:
                results = model(frame.image)   # YOLO / OCR
                frame = source.read(timeout=1.0)

    消费速度慢于帧率时，中间帧会被直接覆盖并计入
    :attr:`VideoStats.dropped`——这是刻意的，不是 bug。
    """

    def __init__(self, config: VideoConfig | None = None) -> None:
        self._config = (config or VideoConfig()).validated()
        self._condition = threading.Condition(threading.RLock())
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        self._frame: VideoFrame | None = None
        self._delivered = 0
        self._counter = 0
        self._stats = VideoStats()
        self._fps_window_start = 0.0
        self._fps_window_frames = 0

        # 仅 ffmpeg 后端使用
        self._native_size: tuple[int, int] | None = None
        self._output_size: tuple[int, int] | None = None
        self._process: subprocess.Popen | None = None
        self._ffmpeg_executable: str | None = None

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    @property
    def config(self) -> VideoConfig:
        return self._config

    @property
    def stats(self) -> VideoStats:
        """返回统计快照。"""
        with self._condition:
            return replace(self._stats)

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

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
        self._stop_event.set()
        # 兜底：ffmpeg 若正卡在管道读上，杀掉子进程即可让 readinto 立刻收到 EOF。
        process = self._process
        if process is not None and process.poll() is None:
            process.terminate()
        with self._condition:
            self._condition.notify_all()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout)
            if thread.is_alive():
                LOGGER.warning("拉流线程未在 %.1fs 内退出", timeout)
        self._thread = None
        self._set_state("stopped")

    def __enter__(self) -> "Hm30VideoSource":
        return self.start()

    def __exit__(self, *exc_info: object) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # 读取接口
    # ------------------------------------------------------------------
    def latest(self) -> VideoFrame | None:
        """立刻返回最新一帧，没有则不等待。"""
        with self._condition:
            return self._frame

    def read(self, timeout: float | None = None) -> VideoFrame | None:
        """阻塞等待**下一帧新画面**；超时或已停止返回 None。

        面向单消费者：内部记录投递进度。多消费者场景请用 :meth:`latest` 各自
        取帧，或每个消费者各自持有读游标。
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
                if self._effective_backend() == "ffmpeg":
                    self._ensure_output_size()
                    if self._output_size is not None:
                        self._stream_ffmpeg()
                else:
                    self._stream_opencv()
                delay = self._config.reconnect_delay
            except Exception as exc:  # noqa: BLE001 - 线程不能因单次失败而死
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

    def _effective_backend(self) -> str:
        """把 ``auto`` 解析成实际后端：优先 ffmpeg（延迟可控），找不到才退回 OpenCV。"""
        if self._config.backend != "auto":
            return self._config.backend
        if self._ffmpeg_executable is None:
            self._ffmpeg_executable = _find_ffmpeg(self._config.ffmpeg_executable)
            if self._ffmpeg_executable is None:
                LOGGER.warning(
                    "找不到 ffmpeg，auto 退回 OpenCV 后端（画面滞后约 0.5s）；"
                    "此时应把 telemetry_lag 调到 ≈0.57s，否则帧-遥测时间对齐会偏"
                )
        return "opencv" if self._ffmpeg_executable is None else "ffmpeg"

    def _stream_opencv(self) -> None:
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = _opencv_ffmpeg_options(
            self._config
        )
        capture = cv2.VideoCapture()
        self._set_state("connecting")
        try:
            # 超时必须通过 open() 的 params 传入。实测（OpenCV 5.0）
            # cap.set(CAP_PROP_READ_TIMEOUT_MSEC) 在 open 之前调用无效，
            # OPENCV_FFMPEG_CAPTURE_OPTIONS 里的 rw_timeout 也无效——两者都会
            # 退回到 30 秒默认值，链路一断拉流线程就卡死半分钟。
            if not capture.open(
                self._config.url,
                cv2.CAP_FFMPEG,
                [
                    cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
                    int(self._config.open_timeout * 1000),
                    cv2.CAP_PROP_READ_TIMEOUT_MSEC,
                    int(self._config.read_timeout * 1000),
                ],
            ):
                self._fail(f"OpenCV 打开失败: {self._config.url}")
                return
            self._set_state("streaming")
            while not self._stop_event.is_set():
                ok, image = capture.read()
                if not ok or image is None:
                    self._fail("图传读取中断")
                    return
                self._publish(image)
        finally:
            capture.release()

    def _stream_ffmpeg(self) -> None:
        executable = self._ffmpeg_executable or _find_ffmpeg(
            self._config.ffmpeg_executable
        )
        if executable is None:
            self._fail("找不到 ffmpeg 可执行文件")
            return
        size = self._output_size
        if size is None:
            self._fail("ffmpeg 后端需要先确定输出尺寸（config.width/height）")
            return
        width, height = size

        command = self._build_ffmpeg_command(executable, width, height)
        LOGGER.debug("ffmpeg 命令: %s", " ".join(command))
        self._set_state("connecting")
        # stderr 落到临时文件而不是管道：管道写满会让 ffmpeg 阻塞在日志上、
        # 连带卡死解码；临时文件既能保留错误信息，又不会反压。
        stderr_file = tempfile.TemporaryFile()
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
                        f"ffmpeg 管道结束: {_read_stderr(process, stderr_file)}"
                    )
                    return
                image = np.frombuffer(buffer, dtype=np.uint8).reshape(
                    height, width, 3
                )
                # 注意：image 是 buffer 的视图（零拷贝），下一轮 readinto 会覆盖
                # 它。要保留画面必须 copy —— 见 VideoFrame.copy()。
                self._publish(image)
        finally:
            _terminate(process, stderr_file)
            self._process = None

    def _build_ffmpeg_command(
        self, executable: str, width: int, height: int
    ) -> list[str]:
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
        # 两种流都能覆盖。
        command += ["-timeout", str(int(config.read_timeout * 1_000_000))]
        command += list(config.extra_input_args)
        command += ["-i", config.url]
        # 输出端：直接吐 bgr24 裸帧，numpy 零解析接管
        command += ["-an", "-sn", "-dn"]
        if self._native_size != (width, height):
            # 只在需要时插滤镜：等比缩放到目标尺寸后再切帧
            command += ["-vf", f"scale={width}:{height}"]
        command += list(config.extra_output_args)
        command += ["-pix_fmt", "bgr24", "-f", "rawvideo", "-"]
        return command

    def _ensure_output_size(self) -> None:
        """ffmpeg 后端必须提前知道输出尺寸：优先用配置，其次探测源。"""
        if self._output_size is not None:
            return
        config = self._config
        if config.width > 0 and config.height > 0:
            self._output_size = (config.width, config.height)
            return
        info = probe_stream(
            config.url,
            transport=config.transport,
            timeout=config.open_timeout,
        )
        if info is None:
            self._fail(f"无法探测流分辨率，请显式指定 width/height: {config.url}")
            return
        self._native_size = (info.width, info.height)
        self._output_size = (config.width or info.width, config.height or info.height)

    # ------------------------------------------------------------------
    # 状态与发布
    # ------------------------------------------------------------------
    def _publish(self, image: np.ndarray) -> None:
        now = time.time()
        with self._condition:
            self._counter += 1
            self._frame = VideoFrame(
                index=self._counter,
                image=image,
                timestamp=now,
                lag=self._config.telemetry_lag,
            )
            # 帧率按 1 秒滑窗统计：链路刚接上时往往会先吐一批缓冲帧，
            # 用相邻帧间隔做 EMA 会被这波突发带飞到几百 fps，失去参考价值。
            if self._fps_window_start == 0.0:
                self._fps_window_start = now
            self._fps_window_frames += 1
            elapsed = now - self._fps_window_start
            if elapsed >= 1.0:
                self._stats.fps = self._fps_window_frames / elapsed
                self._fps_window_start = now
                self._fps_window_frames = 0
            self._stats.frames += 1
            self._stats.state = "streaming"
            self._condition.notify_all()

    def _set_state(self, state: str) -> None:
        with self._condition:
            self._stats.state = state
            self._condition.notify_all()

    def _fail(self, message: str) -> None:
        LOGGER.warning("图传中断: %s", message)
        with self._condition:
            self._stats.last_error = message
            self._condition.notify_all()


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
    except Exception:  # noqa: BLE001
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
        try:
            process.stdout.close()
        except Exception:  # noqa: BLE001
            pass
    if stderr_file is not None:
        try:
            stderr_file.close()
        except Exception:  # noqa: BLE001
            pass


def open_hm30_video(
    config: VideoConfig | None = None, *, autostart: bool = True
) -> Hm30VideoSource:
    """创建拉流源；``autostart=False`` 时需自行调用 :meth:`Hm30VideoSource.start`。"""
    source = Hm30VideoSource(config)
    if autostart:
        source.start()
    return source
