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

为什么只有 ffmpeg 一个后端
--------------------------
``cv2.VideoCapture`` 的对照实现已删除，理由是**延迟与可控性**（2026-09 用
UDP/H.264 测试流同口径复测过）：

* **流水线深度**：cv2 路线掐掉发送端后还能读出 **17 帧（≈567ms 画面滞后）**，
  而本模块的 ffmpeg 子进程路线是 **0 帧**——管道反压让接收侧始终保持浅流水。
  实测这个 17 帧**不是**解复用/探测/套接字/解码线程的缓冲：``OPENCV_FFMPEG_CAPTURE_OPTIONS``
  里设 ``fflags;nobuffer|flags;low_delay``、``analyzeduration;0|probesize;500000``、
  ``buffer_size;8192``、``threads;1``、``max_delay;0``，以及 ``CAP_PROP_BUFFERSIZE=1``，
  积压都纹丝不动（环境变量本身是生效的，DEBUG 日志能打出
  ``using capture options from environment``）；它是 OpenCV FFmpeg 后端的内部队列深度。
* **断流不可控**：``read()`` 是 C 层不可中断调用，链路一断默认要**阻塞 30 秒**
  （实测 30.008s）才返回。想改只能走 params 形式
  （``VideoCapture(url, api, [CAP_PROP_READ_TIMEOUT_MSEC, 3000])``，实测压到 3.06s）——
  卡住的 ``read()`` 仍然只能靠**杀 ffmpeg 进程**解决。
* **本模块的做法**：``subprocess`` 拉起 ffmpeg，从管道读 ``rawvideo/bgr24`` 裸帧；
  ``-timeout``（微秒）让断流能被及时判定；子进程可被父进程直接杀掉，所以拉流线程
  永远停得下来；也省掉 Python 侧再解一遍码。

两条边界要知道（2026-09 用带帧号的合成流实测）：

* **消费者跟得上时不丢帧**（432/450 帧逐号连续），"浅流水"不是拿丢帧换来的；
* **但消费者一旦跟不上（反压），丢的是 UDP 套接字层的数据报**——实测 sink 睡 150ms 时
  450 帧只交付 76 帧、后段整段消失，**这类丢帧不计入 ``stats.dropped``**（那是管道
  下游的计数器）。无缓冲 + UDP 下这是物理必然，别把 ``stats.dropped == 0`` 当成
  "网络没丢帧"；
* 连接建立后还有一段**启动盲区**：ffmpeg 探测格式时读走的字节（``-probesize 500000``）
  对不可 seek 的 UDP 流是丢弃的，实测恰好丢 ``500KB / 每帧字节数`` 帧（低码率 165 帧、
  高码率 15 帧）——真机约 0.5~2s。

本模块只依赖 numpy 与标准库（不再 import cv2）。

每一帧都不丢：sink
------------------
采集线程读出的**每一帧都会先交给注册的 sink**（:meth:`Hm30VideoSource.add_sink`），
一帧都不会被跳过。这是本项目的硬要求：目标出现的时间可能极短，漏一帧就可能漏掉
目标；而"先侦查后空投"的模式允许一定的处理延时，不允许丢帧。把
:class:`~airdrop.buffer.AlignmentWriter` 挂上去，每帧就会连同它拍摄时刻的遥测
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
管道里流的是裸 BGR：1080p30 约 186 MB/s，swscale 转换与管道拷贝各占一部分
CPU。做 YOLO/OCR 时把 ``width``/``height`` 设到接近推理分辨率（如 960x540），
带宽立刻降到 1/4。要硬解就把 ``ffmpeg_decoder`` 设成 ``h264_cuvid`` /
``hevc_cuvid``（需要 NVIDIA GPU，开发机实测可用）。
"""

from __future__ import annotations

import contextlib
import logging
import shutil
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace

import numpy as np

LOGGER = logging.getLogger(__name__)

# HM30 常见地址（地面端与机载相机同处 192.168.144.0/24）
HM30_CAMERA_IP = "192.168.144.25"
HM30_GROUND_IP = "192.168.144.12"
# SIYI 相机/云台默认 RTSP 主码流
HM30_DEFAULT_RTSP = f"rtsp://{HM30_CAMERA_IP}:8554/main.264"

SUPPORTED_TRANSPORTS = frozenset({"udp", "tcp"})

# 图传链路固定延时的默认估计（秒）：从"机载相机曝光"到"本进程从管道里切出
# 一帧"之间的处理时间，含 H.264 编码、无线传输、解码与管道缓冲。在
# ffmpeg 后端实测 ≈0.15s。
# 拿到一帧后用它反推该画面真正的拍摄时刻（VideoFrame.capture_timestamp），
# 再按那一刻取遥测，见 :mod:`airdrop.alignment`。
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

    def __init__(self, config: VideoConfig | None = None) -> None:
        self._config = (config or VideoConfig()).validated()
        self._condition = threading.Condition(threading.RLock())
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._sinks: list[Callable[[VideoFrame], None]] = []

        self._frame: VideoFrame | None = None
        self._delivered = 0
        self._counter = 0
        self._stats = VideoStats()
        self._fps_window_start = 0.0
        self._fps_window_frames = 0

        self._process: subprocess.Popen | None = None
        self._ffmpeg_executable: str | None = None

    # ------------------------------------------------------------------
    # 每帧回调（sink）
    # ------------------------------------------------------------------
    def add_sink(self, sink: Callable[[VideoFrame], None]) -> None:
        """注册一个**每帧**回调，用于"一帧都不能丢"的消费者。

        回调在采集线程里同步执行，所以必须快——它每慢一毫秒，就直接吃掉一毫秒的
        拉流预算（缓冲写入只做一次 JPEG 编码，几毫秒，可以接受；写磁盘之类请自己
        再开线程或队列）。回调抛异常只记日志，不会打断拉流。

        典型用法是挂 :class:`airdrop.buffer.AlignmentWriter`，把每帧连同拍摄时刻的
        遥测写进环形缓冲。
        """
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
                # 线程没停就别清引用：清了 running 会变 False、start() 会再起
                # 一个线程，两个拉流线程抢同一个源。保留引用，稍后可再调 stop()。
                LOGGER.warning("拉流线程未在 %.1fs 内退出，保留引用（可再调 stop()）", timeout)
                return
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
        """立刻返回最新一帧，没有则不等待（实时路径，同样会跳过中间帧）。"""
        with self._condition:
            return self._frame

    def read(self, timeout: float | None = None) -> VideoFrame | None:
        """阻塞等待**下一帧新画面**；超时或已停止返回 None。

        这是**实时**读取：只关心当前画面，跟进不上就会丢掉中间帧（计入
        :attr:`VideoStats.dropped`）。要保证一帧不丢，就别在这条路径上做重活——
        用 :meth:`add_sink` 挂一个 :class:`airdrop.buffer.AlignmentWriter`，
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
        if self._ffmpeg_executable is None:
            self._ffmpeg_executable = _find_ffmpeg(self._config.ffmpeg_executable)
        if self._ffmpeg_executable is None:
            # 环境问题，重连多少次都没用：直接说明白并停下
            self._fail("找不到 ffmpeg 可执行文件（imageio-ffmpeg 与 PATH 里都没有），无法拉流")
            self._set_state("stopped")
            return

        delay = self._config.reconnect_delay
        while not self._stop_event.is_set():
            try:
                self._stream_ffmpeg()
                delay = self._config.reconnect_delay
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
    def _publish(self, image: np.ndarray) -> None:
        now = time.time()
        with self._condition:
            self._counter += 1
            frame = VideoFrame(
                index=self._counter,
                image=image,
                timestamp=now,
                lag=self._config.telemetry_lag,
            )
            self._frame = frame
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
            sinks = tuple(self._sinks)
            self._condition.notify_all()

        # 逐帧投递给 sink，且放在锁外——别让编码/落盘拖住 read()/latest()。
        # "每一帧都被留存"的保证就在这里：frame 是复用视图，sink 必须在这
        # 一帧还有效时把它接管（缓冲写入会拷贝，见 AlignmentWriter）。
        for sink in sinks:
            try:
                sink(frame)
            except Exception:
                LOGGER.exception("图传帧 sink 执行失败")

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
