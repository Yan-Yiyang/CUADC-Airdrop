"""图传帧与遥测的时间对齐。

为什么需要对齐
--------------
图传链路从"机载相机曝光"到"本进程从管道里切出一帧"之间有一段固定的处理
延时：H.264 编码（相机侧）、无线链路传输、解码、以及 ffmpeg 输出管道的缓冲。
实测这段约为 150ms（见 :mod:`airdrop.video` 开头的实测数据）。

这个延时不改变数据的含义，但如果不处理，做视觉引导时就会拿"现在的遥测"
去解释"150ms 前的画面"。无人机以 10 m/s 平飞时，150ms 对应 1.5 米的位移——
对精准空投来说这是不能忽略的系统性偏差，而且它会随着速度变化伪装成"算法
不准"。

因此每拿到一帧后，先把它收到的时间减掉这段固定延时，得到画面真正的
拍摄时刻，再用这个时刻去 :class:`~airdrop.broker.TelemetryBroker` 取那一刻
的遥测（默认在历史之间线性内插）::

    aligner = FrameTelemetryAligner(broker)          # 默认扣 150ms
    sample = aligner.align(source.read(timeout=1.0))
    if sample is not None and not sample.extrapolated:
        aim_at(sample.image, sample.snapshot.latitude_deg, sample.snapshot.yaw_deg)

等价的手写形式（不需要本模块也能做）::

    timestamp = frame.capture_timestamp             # = frame.timestamp - 0.15
    snapshot = broker.get_snapshot_at(timestamp, mode="interpolate")

延时放在哪里
------------
延时是链路属性而不是消费者的选择，因此它定义在
:attr:`airdrop.video.VideoConfig.telemetry_lag` 上，由拉流源写进每一帧
（:attr:`airdrop.video.VideoFrame.capture_timestamp`）。对齐器默认直接采用
帧上携带的值，也可以用 ``lag=`` 显式覆盖（换后端做对照、或现场标定出新值
后临时试探）。

标定方式：见 ``docs/calibration_opencv.md`` 的步骤二（棋盘格 PnP / 光流求画面角速度，
与飞控角速度互相关得到画面-遥测时间差），比秒表法可靠。本模块只处理"常数
延时"，不做抖动自适应——真实链路还有毫秒级抖动的部分，那要靠相机自己的
RTCP 时间戳解决，不是这里能修好的。

关于等待（``max_wait``）
------------------------
正常情况下拍摄时刻（= 收到时间 − 150ms）已经落在历史范围内，直接内插即可。
但遥测偶尔会比视频慢半拍：那一刻的拍摄时刻比历史最新时间还新，只能外推。
外推几十毫秒虽不算致命，却是系统性的（每次慢半拍都外推同一小段）。

所以给对齐器设 ``max_wait``（计划定稿 1.0s）：先等遥测追上拍摄时刻，
追上了就内插；等满上限还没追上，说明遥测流本身出了问题——那时按
``on_timeout="drop"`` 丢弃该帧（不拿外推凑数），并记 warning 日志与
:attr:`FrameTelemetryAligner.stats`，让链路故障显式暴露出来。

代价要说清楚：等待发生在调用 :meth:`align` 的线程里。挂成 sink 时就是
阻塞采集线程——正常 10~30Hz 遥测下只有几十毫秒，但遥测真停顿时每帧
都会等满上限（拉流会随之积压）。这是刻意的取舍：遥测停顿本身就是事故，
宁可丢帧并报警，也不要"看起来还在工作"。

关于外推
--------
查询时刻落在历史范围之外时，代理会做线性外推并返回结果。外推在短距离内
近似可用，但拉长就失去意义（例如遥测断了 3 秒还在"外推"飞机的姿态）。
:class:`AlignedSample` 因此如实标记 ``extrapolated`` 与超出边界的秒数
``offset``；需要硬保证时给对齐器设 ``max_extrapolation``，超限的直接
返回 None。
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

from ..telemetry.broker import SUPPORTED_QUERY_MODES, TelemetryBroker
from ..telemetry.models import TelemetrySnapshot
from .source import DEFAULT_TELEMETRY_LAG, VideoFrame

LOGGER = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_TELEMETRY_LAG",
    "AlignStats",
    "AlignedSample",
    "FrameTelemetryAligner",
]

# 等待超时的日志节流：第一次必打，之后每 N 次打一条（持续停顿时不至于刷屏）
_TIMEOUT_LOG_EVERY = 30

# 等待超时后的可选动作（"drop" = 丢弃该帧）
SUPPORTED_TIMEOUT_ACTIONS = frozenset({"drop"})


@dataclass(frozen=True, slots=True)
class AlignStats:
    """对齐统计的只读快照（线程安全地取）。"""

    aligned: int = 0  # 成功产出的样本数
    no_telemetry: int = 0  # 历史为空（一条遥测都还没有）
    waits: int = 0  # 发生过等待的帧数
    wait_timeouts: int = 0  # 等到超时、按 on_timeout 丢弃的帧数
    extrapolation_drops: int = 0  # 超出 max_extrapolation 被丢弃的帧数
    max_wait_s: float = 0.0  # 单次等待的最长耗时（秒）


@dataclass(frozen=True, slots=True)
class AlignedSample:
    """一帧画面，以及按"该画面的拍摄时刻"取到的遥测快照。

    ``timestamp`` 是对齐后的采样时刻，即画面拍摄时刻
    （``frame.timestamp - lag``），也是实际传给遥测代理的查询时间；它与
    ``frame.timestamp``（收到帧的时间）差一个 ``lag``，别混用。
    """

    frame: VideoFrame
    snapshot: TelemetrySnapshot
    timestamp: float
    lag: float
    mode: str
    # 查询时刻落在历史范围之外（快照是外推得到的），offset 为超出边界的秒数
    extrapolated: bool = False
    offset: float = 0.0

    @property
    def image(self):
        """转发 :attr:`VideoFrame.image`（BGR ndarray）。"""
        return self.frame.image

    @property
    def age(self) -> float:
        """该画面拍摄至今过去了多少秒（= 收到帧至今 + 链路延时 ``lag``）。"""
        return time.time() - self.timestamp


class FrameTelemetryAligner:
    """把视频帧对齐到"拍摄时刻"的遥测。

    参数
    ----
    broker:
        同进程的 :class:`~airdrop.broker.TelemetryBroker`。
    lag:
        链路固定延时（秒）。``None``（默认）表示采用帧自带的
        :attr:`VideoFrame.lag`（通常来自 ``VideoConfig.telemetry_lag``，
        默认 0.15）；给了数值就覆盖帧上的值。
    mode:
        查询模式，``"interpolate"``（默认，历史之间线性内插）或
        ``"nearest"``（时间上最近的一条历史）。
    max_extrapolation:
        允许的最大外推秒数；``None`` 表示不限制。设置后，查询时刻超出
        历史边界超过该秒数时 :meth:`align` 返回 None（宁可让调用方跳过这一帧，
        也不要拿不可信的遥测去引导）。
    max_wait:
        等待上限（秒）。拍摄时刻比历史最新时间还新时，先等一小会儿让遥测
        追上，从而能内插而不是外推；超过该上限仍没等到就按 ``on_timeout``
        处理。默认 0.0 = 不等待（直接外推）。
    on_timeout:
        等待超时后的动作。目前支持 ``"drop"``：丢弃该帧（返回 None）。
        正常遥测（10~30Hz）下等待只有几十毫秒；真的等到超时说明遥测流本身
        有问题，那时这些帧本来也无法可信对齐——丢弃并把问题喊出来
        （warning 日志 + :attr:`stats` 的 ``wait_timeouts``），而不是外推凑数。

    注意 max_wait 会让 :meth:`align` 阻塞（在 sink 里就是阻塞采集线程），
    所以它是"短暂等待"，不是"等到天荒地老"的兜底。

    本类的计数用锁保护，可以在多个消费者线程里共享。
    """

    def __init__(
        self,
        broker: TelemetryBroker,
        lag: float | None = None,
        mode: str = "interpolate",
        max_extrapolation: float | None = None,
        max_wait: float = 0.0,
        on_timeout: str = "drop",
    ) -> None:
        if mode not in SUPPORTED_QUERY_MODES:
            raise ValueError(f"不支持的查询模式: {mode}，可选 {sorted(SUPPORTED_QUERY_MODES)}")
        if lag is not None and lag < 0:
            raise ValueError(f"lag 不能为负: {lag}")
        if max_extrapolation is not None and max_extrapolation < 0:
            raise ValueError(f"max_extrapolation 不能为负: {max_extrapolation}")
        if max_wait < 0:
            raise ValueError(f"max_wait 不能为负: {max_wait}")
        if on_timeout not in SUPPORTED_TIMEOUT_ACTIONS:
            raise ValueError(
                f"不支持的 on_timeout: {on_timeout}，可选 {sorted(SUPPORTED_TIMEOUT_ACTIONS)}"
            )
        self._broker = broker
        self._lag = lag
        self._mode = mode
        self._max_extrapolation = max_extrapolation
        self._max_wait = max_wait
        self._on_timeout = on_timeout
        self._stats_lock = threading.Lock()
        self._aligned = 0
        self._no_telemetry = 0
        self._waits = 0
        self._wait_timeouts = 0
        self._extrapolation_drops = 0
        self._max_wait_seen = 0.0

    # ------------------------------------------------------------------
    # 属性
    # ------------------------------------------------------------------
    @property
    def lag(self) -> float | None:
        """覆盖用的固定延时；None 表示采用每一帧自带的值。"""
        return self._lag

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def max_extrapolation(self) -> float | None:
        return self._max_extrapolation

    @property
    def max_wait(self) -> float:
        """等待上限（秒）；0 表示不等待。"""
        return self._max_wait

    @property
    def on_timeout(self) -> str:
        return self._on_timeout

    @property
    def stats(self) -> AlignStats:
        """对齐计数快照（线程安全）。"""
        with self._stats_lock:
            return AlignStats(
                aligned=self._aligned,
                no_telemetry=self._no_telemetry,
                waits=self._waits,
                wait_timeouts=self._wait_timeouts,
                extrapolation_drops=self._extrapolation_drops,
                max_wait_s=self._max_wait_seen,
            )

    # ------------------------------------------------------------------
    # 对齐
    # ------------------------------------------------------------------
    def align(self, frame: VideoFrame | None) -> AlignedSample | None:
        """把一帧对齐到它拍摄时刻的遥测。

        返回 ``None`` 的四种情况：

        1. ``frame`` 为 None；
        2. 遥测历史为空（一条遥测都还没收到）；
        3. 查询时刻比历史还新、且等满 ``max_wait`` 仍未追上（``on_timeout="drop"``）；
        4. 查询时刻超出历史范围且超过 ``max_extrapolation``。

        传入 :meth:`Hm30VideoSource.latest` 拿到的"当前画面"同样适用。
        """
        if frame is None:
            return None

        lag = self._lag if self._lag is not None else frame.lag
        timestamp = frame.timestamp - lag

        # 历史为空：一条遥测都没收到；等也没用，直接跳过（细分到 no_telemetry）
        if self._broker.history_span() is None:
            self._bump("no_telemetry")
            return None

        if self._max_wait > 0 and not self._covered(timestamp):
            self._wait_for_coverage(timestamp)
            if not self._covered(timestamp):
                # 等到超时：遥测没追上拍摄时刻，这些帧本来也对不齐，
                # 按约定丢弃（不拿外推凑数），并让问题可见。
                self._bump("wait_timeouts")
                self._warn_timeout(timestamp)
                if self._on_timeout == "drop":
                    return None

        snapshot = self._broker.get_snapshot_at(timestamp, mode=self._mode)
        if snapshot is None:  # 防御：上面已排除空历史，理论上到不了这里
            self._bump("no_telemetry")
            return None

        extrapolated, offset = self._classify(timestamp)
        if self._max_extrapolation is not None and offset > self._max_extrapolation:
            LOGGER.debug(
                "对齐时刻 %.3f 超出历史边界 %.3fs（阈值 %.3fs），跳过该帧",
                timestamp,
                offset,
                self._max_extrapolation,
            )
            self._bump("extrapolation_drops")
            return None

        self._bump("aligned")
        return AlignedSample(
            frame=frame,
            snapshot=snapshot,
            timestamp=timestamp,
            lag=lag,
            mode=self._mode,
            extrapolated=extrapolated,
            offset=offset,
        )

    # ------------------------------------------------------------------
    # 内部实现
    # ------------------------------------------------------------------
    def _covered(self, timestamp: float) -> bool:
        """历史是否覆盖该时刻（能内插，而不是只能外推）。"""
        span = self._broker.history_span()
        return span is not None and span[0] <= timestamp <= span[1]

    def _wait_for_coverage(self, timestamp: float) -> None:
        """等待遥测历史追上拍摄时刻，最长 ``max_wait`` 秒。"""
        started = time.monotonic()
        self._broker.wait_history_until(timestamp, timeout=self._max_wait)
        elapsed = time.monotonic() - started
        self._bump("waits", max_wait=elapsed)

    def _bump(
        self,
        field: str,
        max_wait: float | None = None,
    ) -> None:
        with self._stats_lock:
            if field == "aligned":
                self._aligned += 1
            elif field == "no_telemetry":
                self._no_telemetry += 1
            elif field == "waits":
                self._waits += 1
            elif field == "wait_timeouts":
                self._wait_timeouts += 1
            elif field == "extrapolation_drops":
                self._extrapolation_drops += 1
            if max_wait is not None and max_wait > self._max_wait_seen:
                self._max_wait_seen = max_wait

    def _warn_timeout(self, timestamp: float) -> None:
        """超时告警（第一次必打，之后每 ``_TIMEOUT_LOG_EVERY`` 次打一条）。

        遥测停满 ``max_wait`` 属于链路故障级别的异常，必须显眼；
        但同时每帧都会触发，所以要做节流，避免把日志刷爆。
        """
        with self._stats_lock:
            count = self._wait_timeouts
        if count == 1 or count % _TIMEOUT_LOG_EVERY == 0:
            LOGGER.warning(
                "遥测未在 %.2fs 内追上拍摄时刻（已累计 %d 帧丢弃）：遥测流可能已停顿，请检查链路",
                self._max_wait,
                count,
            )

    def _classify(self, timestamp: float) -> tuple[bool, float]:
        """判断查询时刻是否落在历史范围之外，返回 ``(是否外推, 超出秒数)``。"""
        span = self._broker.history_span()
        if span is None:
            return False, 0.0
        earliest, latest = span
        if timestamp < earliest:
            return True, earliest - timestamp
        if timestamp > latest:
            return True, timestamp - latest
        return False, 0.0
