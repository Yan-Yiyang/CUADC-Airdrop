'图传帧与遥测的时间对齐。\n\n这个延时不改变数据的含义，但如果不处理，做视觉引导时就会拿"现在的遥测"\n去解释"150ms 前的画面"。无人机以 10 m/s 平飞时，150ms 对应 1.5 米的位移——\n对精准空投来说这是不能忽略的系统性偏差，而且它会随着速度变化伪装成"算法\n不准"。\n\n因此每拿到一帧后，先把它**收到的时间减掉这段固定延时**，得到画面真正的\n拍摄时刻，再用这个时刻去 :class:`~airdrop.broker.TelemetryBroker` 取那一刻\n的遥测（默认在历史之间线性内插）::\n\n    aligner = FrameTelemetryAligner(broker)          # 默认扣 150ms\n    sample = aligner.align(source.read(timeout=1.0))\n    if sample is not None and not sample.extrapolated:\n        aim_at(sample.image, sample.snapshot.latitude_deg, sample.snapshot.yaw_deg)\n\n等价的手写形式（不需要本模块也能做）::\n\n    timestamp = frame.capture_timestamp             # = frame.timestamp - 0.15\n    snapshot = broker.get_snapshot_at(timestamp, mode="interpolate")\n\n延时放在哪里\n------------\n延时是**链路属性**而不是消费者的选择，因此它定义在\n:attr:`airdrop.video.VideoConfig.telemetry_lag` 上，由拉流源写进每一帧\n（:attr:`airdrop.video.VideoFrame.capture_timestamp`）。对齐器默认直接采用\n帧上携带的值，也可以用 ``lag=`` 显式覆盖（换后端做对照、或现场标定出新值\n后临时试探）。\n\n标定方式：对着秒表拍摄，比较画面里的读数与本地时钟。本模块只处理"常数\n延时"，不做抖动自适应——真实链路还有毫秒级抖动的部分，那要靠相机自己的\nRTCP 时间戳解决，不是这里能修好的。\n\n关于外推\n--------\n查询时刻落在历史范围之外时，代理会做线性外推并返回结果。外推在短距离内\n近似可用，但拉长就失去意义（例如遥测断了 3 秒还在"外推"飞机的姿态）。\n:class:`AlignedSample` 因此如实标记 ``extrapolated`` 与超出边界的秒数\n``offset``；需要硬保证时给对齐器设 ``max_extrapolation``，超限的直接\n返回 None。\n'

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from .broker import SUPPORTED_QUERY_MODES, TelemetryBroker
from .models import TelemetrySnapshot
from .video import DEFAULT_TELEMETRY_LAG, VideoFrame

LOGGER = logging.getLogger(__name__)

__all__ = [
    "AlignedSample",
    "DEFAULT_TELEMETRY_LAG",
    "FrameTelemetryAligner",
]


@dataclass(frozen=True, slots=True)
class AlignedSample:
    """一帧画面，以及按"该画面的拍摄时刻"取到的遥测快照。

    ``timestamp`` 是**对齐后的采样时刻**，即画面拍摄时刻
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

    本类无状态、线程安全（动作全部委托给代理自己的锁），可以在多个消费者
    线程里共享。
    """

    def __init__(
        self,
        broker: TelemetryBroker,
        lag: float | None = None,
        mode: str = "interpolate",
        max_extrapolation: float | None = None,
    ) -> None:
        if mode not in SUPPORTED_QUERY_MODES:
            raise ValueError(
                f"不支持的查询模式: {mode}，可选 {sorted(SUPPORTED_QUERY_MODES)}"
            )
        if lag is not None and lag < 0:
            raise ValueError(f"lag 不能为负: {lag}")
        if max_extrapolation is not None and max_extrapolation < 0:
            raise ValueError(f"max_extrapolation 不能为负: {max_extrapolation}")
        self._broker = broker
        self._lag = lag
        self._mode = mode
        self._max_extrapolation = max_extrapolation

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

    # ------------------------------------------------------------------
    # 对齐
    # ------------------------------------------------------------------
    def align(self, frame: VideoFrame | None) -> AlignedSample | None:
        """把一帧对齐到它拍摄时刻的遥测。

        返回 ``None`` 的三种情况：``frame`` 为 None；遥测历史为空；查询时刻
        超出历史范围且超过 ``max_extrapolation``。

        传入 :meth:`Hm30VideoSource.latest` 拿到的"当前画面"同样适用。
        """
        if frame is None:
            return None

        lag = self._lag if self._lag is not None else frame.lag
        timestamp = frame.timestamp - lag
        snapshot = self._broker.get_snapshot_at(timestamp, mode=self._mode)
        if snapshot is None:  # 历史为空：还没有任何遥测
            return None

        extrapolated, offset = self._classify(timestamp)
        if (
            self._max_extrapolation is not None
            and offset > self._max_extrapolation
        ):
            LOGGER.debug(
                "对齐时刻 %.3f 超出历史边界 %.3fs（阈值 %.3fs），跳过该帧",
                timestamp,
                offset,
                self._max_extrapolation,
            )
            return None

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
