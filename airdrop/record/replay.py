"""飞行目录回放：把录下来的帧与遥测按**原时间轴**重新播一遍。

为什么要"原时间轴"
------------------
回放的用途是**离线迭代识别与坐标转换**：拿同一份素材反复调 perception /
georef / targeting 的参数，然后和当时的 ``detections.jsonl`` 对比。要做到
"改了算法、别的都没变"，就必须让回放时的每一帧仍然带着它**拍摄那一刻**
的原始 ``capture_timestamp``，遥测也仍然按原始时间戳进入 broker——
于是帧-遥测对齐、坐标解算、聚类统计**一行都不用改**，实飞与回放走的是
同一条代码路径。缺了这一点，离线结果就没有可比性。

两个组件
--------
* :class:`ReplayVideoSource`：与 :class:`~airdrop.video.source.Hm30VideoSource`
  **同一套公开接口**（``add_sink`` / ``read`` / ``latest`` / ``stats`` /
  ``stop`` / ``iter_frames``），把它换进现有装配即可。帧从
  ``frames/%06d.jpg`` 读，时间戳取自 ``frames_index.jsonl``：写入
  ``VideoFrame.timestamp = capture_timestamp + lag``，从而
  ``frame.capture_timestamp`` 与录制时**逐位相同**。
* :func:`load_broker_from_log` + :class:`TelemetryPacer`：把
  ``telemetry.jsonl`` 灌进 broker。**必须在帧前推进**——某一帧要能在
  "拍摄时刻"查询内插，那一刻的遥测就必须已经进了历史。所以由
  :class:`ReplayVideoSource` 在**投递每一帧之前**调用
  :meth:`TelemetryPacer.publish_until`，把该帧时刻之前的快照全部补上。

线程模型
--------
与实飞一致：一个后台线程按 ``speed`` 定速播放，逐帧调用 sink（一帧不落），
同时更新 ``latest()`` 供实时查看。``speed=0`` 全速（算力能跑多快跑多快，
用于批量回归），``speed=1.0`` 原速，``speed=2.0`` 两倍速。
"""

from __future__ import annotations

import bisect
import json
import logging
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..telemetry.broker import TelemetryBroker
from ..telemetry.models import TelemetrySnapshot
from ..video.buffer import AlignmentBuffer
from ..video.source import VideoConfig, VideoFrame

LOGGER = logging.getLogger(__name__)

__all__ = [
    "FlightLog",
    "FlightLogError",
    "FrameIndexError",
    "FrameRecord",
    "ReplayStats",
    "ReplayVideoSource",
    "TelemetryPacer",
    "load_broker_from_log",
]

# 与 FlightRecorder 落盘格式一致的文件名（见 airdrop.record.recorder）
FRAMES_DIR = "frames"
FRAMES_INDEX_NAME = "frames_index.jsonl"
TELEMETRY_NAME = "telemetry.jsonl"
EVENTS_NAME = "events.jsonl"
DETECTIONS_NAME = "detections.jsonl"

# 回放期间每次等待的切片长度（秒）。等待被切成小片，``stop()`` 才能及时生效。
_WAIT_SLICE = 0.05

# 全速回放（speed=0）时的最小帧间隔。完全不限速会让生产线程在毫秒内把整段素材
# 播完，``read()`` 这条实时路径只能看到最后一帧、中间帧被"跳过"——那正是实飞源
# 里要计入 dropped 的行为。给一个 1ms 的节奏下限，read() 就能正常跟帧。
_MIN_FRAME_INTERVAL_S = 0.001


class FlightLogError(RuntimeError):
    """飞行目录不可用（缺文件、格式不对）。"""


class FrameIndexError(FlightLogError):
    """``frames_index.jsonl`` 损坏（时间戳不递增、字段缺失）。"""


# ----------------------------------------------------------------------
# 索引记录
# ----------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class FrameRecord:
    """``frames_index.jsonl`` 的一行：一帧的落盘位置与时间信息。"""

    index: int
    filename: str
    capture_timestamp: float
    received_timestamp: float
    lag: float
    extrapolated: bool = False
    offset: float = 0.0
    bytes: int = 0

    @property
    def timestamp(self) -> float:
        """还原"收到时刻"：``capture_timestamp + lag``。

        有了它，``VideoFrame.capture_timestamp``（= ``timestamp - lag``）
        回到与录制时**完全一致**的值——回放的对齐结果因此可与实飞逐帧比对。
        """
        return self.capture_timestamp + self.lag

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, line_no: int) -> "FrameRecord":
        try:
            return cls(
                index=int(data["index"]),
                filename=str(data["filename"]),
                capture_timestamp=float(data["capture_timestamp"]),
                received_timestamp=float(data.get("received_timestamp", 0.0)),
                lag=float(data.get("lag", 0.0)),
                extrapolated=bool(data.get("extrapolated", False)),
                offset=float(data.get("offset", 0.0)),
                bytes=int(data.get("bytes", 0)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise FrameIndexError(
                f"frames_index.jsonl 第 {line_no} 行字段缺失或类型不对: {exc}"
            ) from exc


# ----------------------------------------------------------------------
# 飞行目录
# ----------------------------------------------------------------------
@dataclass(slots=True)
class FlightLog:
    """一个飞行目录的只读视图（帧索引 + 遥测）。

    ``read_frames()`` 之外的解析都**不碰磁盘**：帧索引一次性读进来（10800 帧
    ≈1.5 MB，可忽略），遥测流式读，避免把整段飞行驻留内存。
    """

    flight_dir: Path
    frames: list[FrameRecord] = field(default_factory=list)

    @classmethod
    def open(cls, flight_dir: str | Path) -> "FlightLog":
        """打开目录并解析帧索引（遥测按需流式读）。"""
        path = Path(flight_dir)
        if not path.is_dir():
            raise FlightLogError(f"飞行目录不存在: {path}")
        frames = _read_frame_index(path / FRAMES_INDEX_NAME)
        log = cls(flight_dir=path, frames=frames)
        LOGGER.info("回放素材就绪：%s（%d 帧）", path, len(frames))
        return log

    @property
    def frames_dir(self) -> Path:
        return self.flight_dir / FRAMES_DIR

    @property
    def telemetry_path(self) -> Path:
        return self.flight_dir / TELEMETRY_NAME

    @property
    def has_telemetry(self) -> bool:
        return self.telemetry_path.is_file()

    def time_span(self) -> tuple[float, float] | None:
        """帧覆盖的时间范围 ``(拍摄最早, 拍摄最晚)``；无帧返回 None。"""
        if not self.frames:
            return None
        return self.frames[0].capture_timestamp, self.frames[-1].capture_timestamp

    def iter_telemetry(self) -> Iterator[TelemetrySnapshot]:
        """流式读出 ``telemetry.jsonl`` 的每条快照。

        坏行只记 warning 跳过：一段飞行的素材很宝贵，不该因为最后一行被截断
        就整份作废。
        """
        path = self.telemetry_path
        if not path.is_file():
            raise FlightLogError(f"缺少遥测日志: {path}")
        expected = set(TelemetrySnapshot.__dataclass_fields__)
        with open(path, encoding="utf-8") as handle:
            for line_no, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    LOGGER.warning("telemetry.jsonl 第 %d 行不是合法 JSON，跳过", line_no)
                    continue
                if not isinstance(data, dict):
                    LOGGER.warning("telemetry.jsonl 第 %d 行不是对象，跳过", line_no)
                    continue
                known = {k: v for k, v in data.items() if k in expected}
                try:
                    snapshot = TelemetrySnapshot(**known)
                except TypeError as exc:
                    LOGGER.warning("telemetry.jsonl 第 %d 行字段异常，跳过: %s", line_no, exc)
                    continue
                yield snapshot


def _read_frame_index(path: Path) -> list[FrameRecord]:
    """解析帧索引并校验时间戳单调递增。"""
    if not path.is_file():
        raise FlightLogError(f"缺少帧索引: {path}")
    records: list[FrameRecord] = []
    last_capture: float | None = None
    with open(path, encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                raise FrameIndexError(
                    f"frames_index.jsonl 第 {line_no} 行不是合法 JSON: {exc}"
                ) from exc
            record = FrameRecord.from_dict(data, line_no=line_no)
            if last_capture is not None and record.capture_timestamp < last_capture:
                raise FrameIndexError(
                    f"frames_index.jsonl 第 {line_no} 行时间戳倒流："
                    f"{record.capture_timestamp} < {last_capture}"
                )
            last_capture = record.capture_timestamp
            records.append(record)
    if not records:
        raise FrameIndexError(f"帧索引为空: {path}")
    return records


# ----------------------------------------------------------------------
# 遥测回填
# ----------------------------------------------------------------------
class TelemetryPacer:
    """把遥测日志按原始时间戳**逐步**灌进 broker（回放的时间轴推进器）。

    用法上它不是一个独立线程，而是由 :class:`ReplayVideoSource` 在投递每一帧
    之前驱动：``publish_until(frame.capture_timestamp)``。这样"查询某帧时刻的
    遥测"必然已经落在历史区间内，对齐器走内插而**不是**外推，也不需要等待
    ——1.0s 的等待上限在回放里永远不会被触发。

    另外它会多推进 ``ahead_s``（默认 0.5s）：内插需要**右端点**，
    只推进到帧时刻本身的话，帧时刻正好是历史区间的右边界，数值上虽能内插
    （比率 1.0），但一旦遥测有细微抖动就会退化成外推。

    由 :class:`ReplayVideoSource` 在每帧投递前按需调用 :meth:`finish`：只把遥测
    覆盖补到当前帧的拍摄时刻。**必须补这一步**——录制器按固定频率写遥测，日志的
    最后一条通常比末帧早几十毫秒，不补的话末帧会去等对齐器的 1.0s 上限、然后被
    记成"链路停顿"丢弃（回放里那是假警报）。
    """

    def __init__(
        self,
        broker: TelemetryBroker,
        snapshots: list[TelemetrySnapshot] | Any,
        *,
        ahead_s: float = 0.5,
        tail_warn_s: float = 5.0,
    ) -> None:
        if ahead_s < 0:
            raise ValueError(f"ahead_s 不能为负: {ahead_s}")
        self._broker = broker
        self._snapshots: list[TelemetrySnapshot] = sorted(snapshots, key=lambda s: s.timestamp)
        self._timestamps = [s.timestamp for s in self._snapshots]
        self._ahead_s = ahead_s
        self._tail_warn_s = tail_warn_s
        self._cursor = 0
        self._finished = 0
        self._warned_no_coverage = False
        self._warned_tail = False
        self._coverage_timestamp = float("-inf")

    @classmethod
    def from_log(
        cls,
        broker: TelemetryBroker,
        flight_dir: str | Path,
        **kwargs: Any,
    ) -> "TelemetryPacer":
        """从飞行目录的 ``telemetry.jsonl`` 建 pacer。"""
        log = FlightLog.open(flight_dir)
        return cls(broker, list(log.iter_telemetry()), **kwargs)

    # ------------------------------------------------------------------
    # 状态
    # ------------------------------------------------------------------
    @property
    def published(self) -> int:
        """已经灌进 broker 的快照条数。"""
        return self._cursor

    @property
    def total(self) -> int:
        return len(self._snapshots)

    @property
    def exhausted(self) -> bool:
        """日志里的快照是否已经全部灌完。"""
        return self._cursor >= len(self._snapshots)

    @property
    def finished_to(self) -> float:
        """日志末端快照的时间戳；没有快照时返回 ``-inf``。"""
        if not self._snapshots:
            return float("-inf")
        return self._snapshots[-1].timestamp

    @property
    def tail_extended(self) -> int:
        """为补足素材覆盖而额外延伸的快照条数。

        ⚠ 逐帧调用 :meth:`finish` 时它会按帧数增长（30 fps 下补 30 秒就是
        ~30 条，每帧一条），不是"正常素材 0 或 1"；要判断"尾部缺了多少"
        看 ``cover_to()`` 与最后一条真实快照的差，或看有没有那条 WARNING。
        """
        return self._finished

    def cover_to(self) -> float:
        """当前已推进到的时间；没有快照时返回 ``-inf``。"""
        if self._cursor == 0:
            return float("-inf")
        return max(self._timestamps[self._cursor - 1], self._coverage_timestamp)

    # ------------------------------------------------------------------
    # 推进
    # ------------------------------------------------------------------
    def publish_until(self, timestamp: float) -> int:
        """把时间戳 ≤ ``timestamp + ahead_s`` 的快照全部发布；返回本次条数。"""
        limit = timestamp + self._ahead_s
        stop = bisect.bisect_right(self._timestamps, limit)
        count = 0
        while self._cursor < stop:
            self._broker.publish(self._snapshots[self._cursor])
            self._cursor += 1
            count += 1
        if not self._snapshots and not self._warned_no_coverage:
            self._warned_no_coverage = True
            LOGGER.warning("遥测日志为空，回放中没有可用的遥测")
        return count

    def finish(self, capture_timestamp: float) -> int:
        """把遥测覆盖补到 ``capture_timestamp``（回放源在每帧投递前按需调用）。

        必要的原因：素材往往**遥测先于视频结束**——录制器按固定频率写遥测，
        而视频流停在最后一帧，日志的最后一条通常比末帧早几十毫秒。不补的话，
        末帧的对齐会去等那 1.0s 上限（回放里等多久都不会有新遥测），最后被
        记成"链路停顿"丢弃——那是**假警报**，而且白白浪费 1 秒。

        补法是"把最后一条快照沿时间轴向前延伸"：状态取最后一个已知值，
        时间戳按原时间轴单调前进。**尾部缺失量按最后一条真实快照算**
        （不是按上一次的延伸点）——逐帧调用时后者每步只差几十毫秒，一整段
        缺失会被摊平，``tail_warn_s`` 的告警就永远不会触发。超过
        ``tail_warn_s`` 秒只告警**一次**（同一次回放里素材尾部就这一处问题）。

        到这一刻为止**尚未覆盖的历史区间**（日志中途就断掉的场合）也一并用最后
        一条快照填上：否则那一段里的帧会一帧一帧各等满 1.0s 再被丢弃——回放里
        空耗时间，还会把"素材日志不完整"伪装成"链路停顿"。⚠ 中段断档走不到这里
        （``_timestamps[-1]`` 还在后面，本方法直接返回），那些帧仍会如实触发
        对齐等待与告警。

        返回本次**延伸**出来的条数（0 表示不需要补）；先把待灌的真实快照灌完
        的那部分不计入其中（那不算"延伸"）。
        """
        if not self._snapshots:
            return 0
        # 只推进当前帧所需的真实遥测；不能把帧之后的整段日志提前灌入 broker。
        self.publish_until(capture_timestamp)
        if self._timestamps[-1] >= capture_timestamp:
            return 0
        anchor = self.cover_to()
        if self._cursor == 0:
            return 0
        if capture_timestamp + 1e-6 <= anchor:
            return 0
        missing = capture_timestamp - self._timestamps[-1]
        if missing > self._tail_warn_s and not self._warned_tail:
            self._warned_tail = True
            LOGGER.warning(
                "遥测日志在 %.3f 就结束了，而素材帧一直到 %.3f（差 %.2fs）——"
                "末尾这段没有真实遥测，已按最后一条快照延伸，结果仅供参考",
                self._timestamps[-1],
                capture_timestamp,
                missing,
            )
        last = self._snapshots[self._cursor - 1]
        # 逐步推进（每步不超过 ahead_s），保证区间连续、内插不会落到空档
        step = max(self._ahead_s, 0.05)
        point = anchor
        added = 0
        while point < capture_timestamp:
            point = min(point + step, capture_timestamp)
            extended = replace(last, timestamp=point + 1e-6)
            self._broker.publish(extended)
            added += 1
        self._coverage_timestamp = capture_timestamp
        self._finished += added
        LOGGER.debug(
            "补齐素材末端：遥测从 %.3f 延伸到 %.3f（%d 条）",
            anchor,
            capture_timestamp,
            added,
        )
        return added


def load_broker_from_log(
    flight_dir: str | Path,
    *,
    broker: TelemetryBroker | None = None,
    history_maxlen: int = 0,
    ahead_s: float = 0.5,
) -> tuple[TelemetryBroker, TelemetryPacer]:
    """建一个专用 broker，并返回``(broker, pacer)``——**遥测尚未灌入**。

    灌入由 :class:`ReplayVideoSource` 在播放过程中按帧时刻驱动（见
    :class:`TelemetryPacer`），这样时间轴与帧严格同步；不要在这里一次性
    全灌——那样虽然对齐也能查到，但"等待遥测追上"这类行为就永远测不到了。

    ``history_maxlen=0`` 表示不限长（回放整段飞行的历史都要在），
    ``history_interval`` 固定为 0：回填的时间轴不能被采样节流改动。
    """
    if history_maxlen < 0:
        raise ValueError(f"history_maxlen 不能为负: {history_maxlen}")
    target = (
        broker
        if broker is not None
        else TelemetryBroker(
            history_maxlen=history_maxlen or 100_000,
            history_interval=0.0,
        )
    )
    log = FlightLog.open(flight_dir)
    if not log.has_telemetry:
        raise FlightLogError(f"缺少遥测日志: {log.telemetry_path}")
    snapshots = list(log.iter_telemetry())
    pacer = TelemetryPacer(target, snapshots, ahead_s=ahead_s)
    span = (snapshots[0].timestamp, snapshots[-1].timestamp) if snapshots else None
    LOGGER.info(
        "遥测回填就绪：%d 条%s", pacer.total, f"（{span[0]:.3f} → {span[1]:.3f}）" if span else ""
    )
    return target, pacer


# ----------------------------------------------------------------------
# 统计
# ----------------------------------------------------------------------
@dataclass(slots=True)
class ReplayStats:
    """回放统计。字段与 :class:`~airdrop.video.source.VideoStats` 对齐，
    便于直接把回放源当作实飞源使用；另外多了几个回放专有的量。"""

    state: str = "idle"
    frames: int = 0  # 已投递帧数（sink 逐帧，一帧不落）
    dropped: int = 0  # read() 实时路径跳过的帧（sink 路径不受影响）
    reconnects: int = 0  # 回放无链路：恒为 0
    fps: float = 0.0  # 按回放墙钟统计的实际播放帧率
    last_error: str | None = None
    total: int = 0  # 索引里的总帧数
    skipped: int = 0  # 素材缺文件/解码失败而跳过的帧
    telemetry_snapshots: int = 0  # 已灌进 broker 的遥测条数
    tail_extended: int = 0  # 为覆盖素材末尾而延伸的遥测条数
    duration_s: float = 0.0  # 素材自身的时间跨度（拍摄时刻之差）


# ----------------------------------------------------------------------
# 回放源
# ----------------------------------------------------------------------
class ReplayVideoSource:
    """回放源：与 :class:`Hm30VideoSource` 同一套公开接口。

    典型用法（离线迭代视觉与坐标解算，全程无需硬件）::

        broker, pacer = load_broker_from_log(flight_dir)
        buffer = AlignmentBuffer()
        source = ReplayVideoSource(flight_dir, speed=0.0, telemetry=pacer)
        source.add_sink(AlignmentWriter(buffer, FrameTelemetryAligner(broker)))
        source.start()
        ...
        source.stop()

    参数
    ----
    flight_dir:
        飞行目录（``FlightRecorder`` 的产出）。
    speed:
        ``0``（默认）全速——只受算法吞吐限制，用于批量回归；``1.0`` 原速；
        也可以 ``2.0`` 这样加速。全速时"实时读取"路径没有意义，但 sink
        路径一帧不落，正是离线迭代要的。
    telemetry:
        :class:`TelemetryPacer`；给了就在**投递每帧之前**推进遥测，使该帧的
        拍摄时刻一定有遥测覆盖。不给则纯放帧（只测视频管线时用）。
    buffer:
        给了就改成"跟随缓冲"模式：忽略帧索引里的 ``capture_timestamp``，
        用 ``buffer.wait_new`` 逐条消费缓冲并取时间戳（缓冲里的记录本身就是
        按帧时刻对齐过的）。适合一边采一边处理的联机调试。
    repeat:
        播完一遍后是否循环（做长时间稳定性验证时有用）。
    strict:
        ``True``（默认）：素材损坏（帧文件缺失/解码失败）或 **sink 抛异常**都
        直接让回放以 ``state="error"`` 结束——做正式评估时宁可失败，也不要
        "看起来跑通了、其实少了一半帧"。``False`` 则从宽：缺帧记账跳过，
        sink 异常只记日志、继续播后面的帧（适合"先看看效果"的探索阶段）。
    """

    def __init__(
        self,
        flight_dir: str | Path,
        *,
        speed: float = 0.0,
        telemetry: "TelemetryPacer | None" = None,
        buffer: AlignmentBuffer | None = None,
        repeat: bool = False,
        strict: bool = True,
        tail_s: float = 0.0,
    ) -> None:
        if speed < 0:
            raise ValueError(f"speed 不能为负: {speed}")
        if tail_s < 0:
            raise ValueError(f"tail_s 不能为负: {tail_s}")

        self._log = FlightLog.open(flight_dir)
        self._speed = float(speed)
        self._telemetry = telemetry
        self._buffer = buffer
        self._repeat = repeat
        self._strict = strict
        self._tail_s = tail_s

        self._condition = threading.Condition(threading.RLock())
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._sinks: list[Callable[[VideoFrame], None]] = []

        self._frame: VideoFrame | None = None
        self._delivered = 0  # read() 的投递游标
        self._produced = 0  # 已投递帧数（sink 逐帧）
        self._skipped = 0
        self._telemetry_published = 0
        self._stats = ReplayStats(
            total=len(self._log.frames),
            duration_s=self._material_duration(),
        )
        self._config: VideoConfig | None = None
        # fps 滑窗（1 秒）
        self._fps_start: float | None = None
        self._fps_frames = 0

    # ------------------------------------------------------------------
    # 只读属性
    # ------------------------------------------------------------------
    @property
    def flight_dir(self) -> Path:
        return self._log.flight_dir

    @property
    def frames(self) -> tuple[FrameRecord, ...]:
        """帧索引（只读元组）。"""
        return tuple(self._log.frames)

    @property
    def config(self) -> VideoConfig:
        """回放的"视频配置"：尺寸取自素材，``telemetry_lag`` 取首帧记录值。

        让回放源和实飞源在装配代码里可以互换（例如
        :class:`~airdrop.record.FlightRecorder` 只关心 sink）。
        """
        if self._config is None:
            first = self._log.frames[0]
            image = self._read_image(first)
            height, width = (image.shape[0], image.shape[1]) if image is not None else (0, 0)
            self._config = VideoConfig(
                url=str(self._log.flight_dir),
                width=width or 1,
                height=height or 1,
                telemetry_lag=first.lag,
            )
        return self._config

    @property
    def stats(self) -> ReplayStats:
        with self._condition:
            return replace(self._stats)

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------------
    # sink 接口（与 Hm30VideoSource 一致）
    # ------------------------------------------------------------------
    def add_sink(self, sink: Callable[[VideoFrame], None]) -> None:
        """注册每帧回调；回放**不会丢帧**，sink 每帧都被调用一次。"""
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
    def start(self) -> "ReplayVideoSource":
        """启动回放线程（幂等）。"""
        with self._condition:
            if self.running:
                return self
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._run,
                name="replay-video",
                daemon=True,
            )
            self._set_state("playing")
            self._thread.start()
        return self

    def stop(self, timeout: float = 5.0) -> None:
        """停止回放并等线程退出（幂等）。"""
        self._stop_event.set()
        with self._condition:
            self._condition.notify_all()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout)
            if thread.is_alive():
                # 线程没停就别清引用（否则 running 会撒谎、start() 会起第二个）
                LOGGER.warning("回放线程未在 %.1fs 内退出，保留引用（可再调 stop()）", timeout)
                return
        self._thread = None
        self._set_state("stopped")

    def __enter__(self) -> "ReplayVideoSource":
        return self.start()

    def __exit__(self, *exc_info: object) -> None:
        self.stop()

    def wait_ready(self, timeout: float = 10.0) -> bool:
        """等到第一帧被投递（与实飞源同义）。"""
        return self.read(timeout=timeout) is not None

    # ------------------------------------------------------------------
    # 读取接口
    # ------------------------------------------------------------------
    def latest(self) -> VideoFrame | None:
        """立刻返回最新一帧（实时路径）。"""
        with self._condition:
            return self._frame

    def read(self, timeout: float | None = None) -> VideoFrame | None:
        """阻塞等待下一帧；播完/超时/已停止返回 None。

        **这是实时路径**：与实飞源一样只保证"当前这一帧"，消费速度跟不上时
        中间帧会被跳过并计入 ``stats.dropped``。而 ``add_sink`` 注册的回调
        在播放线程里逐帧调用，**一帧不落**——离线迭代请走 sink/缓冲那条路，
        别用 ``read()`` 循环去喂算法。

        播完后本方法返回 None（``state="finished"``）；正在播放但消费太慢时
        生产线程会先等一小会儿（见 ``_wait_for_consumer``），不会让最后一帧
        凭空消失。
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            while True:
                frame = self._frame
                if frame is not None and frame.index > self._delivered:
                    # 中间被超过的帧就是"实时路径没看上"的帧（与实飞源同义）。
                    # 注意：sink 路径一帧不落，这里的计数只反映 read() 这条路。
                    self._stats.dropped += frame.index - self._delivered - 1
                    self._delivered = frame.index
                    return frame
                # 播放线程已收工且没有更新的帧 → 结束（不是"停"，是"播完了"）
                if self._stats.state in {"finished", "error"}:
                    return None
                if self._stop_event.is_set():
                    return None
                remaining = None
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return None
                self._condition.wait(remaining)

    def iter_frames(self, timeout: float | None = 1.0) -> Iterator[VideoFrame]:
        """迭代新帧；连续 ``timeout`` 秒没有新画面就结束。"""
        while not self._stop_event.is_set():
            frame = self.read(timeout=timeout)
            if frame is None:
                return
            yield frame

    # ------------------------------------------------------------------
    # 播放线程
    # ------------------------------------------------------------------
    def _run(self) -> None:
        try:
            self._play_all()
            # 播完但还有帧没被 read() 取走时，先别急着置 finished：read() 一旦
            # 看到 finished 就返回 None，最后一帧会被静默丢掉。生产速度快于
            # 消费速度（全速回放很常见）时，这里就是唯一的补救点。
            self._wait_for_consumer()
        except Exception as exc:
            LOGGER.exception("回放异常")
            with self._condition:
                self._stats.last_error = str(exc)
            self._set_state("error")
            return
        if self._stop_event.is_set():
            self._set_state("stopped")
        else:
            self._set_state("finished")

    def _wait_for_consumer(self, grace_s: float = 0.5) -> None:
        """等 ``read()`` 把已投递的帧消费完（或等超时/被 stop）。

        默认只等 ``grace_s + 0.01s/帧``：没有 read() 消费者的场合（纯 sink
        离线跑）不该为此白等，而慢消费者只要还在读就能一直把它等下去——
        每取走一帧都会 ``notify``，循环会重新计时到同一个 deadline。
        """
        outstanding = max(self._produced - self._delivered, 0)
        deadline = time.monotonic() + grace_s + 0.01 * outstanding
        with self._condition:
            while self._delivered < self._produced and not self._stop_event.is_set():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    LOGGER.debug(
                        "回放已播完，仍有 %d 帧未被读取",
                        self._produced - self._delivered,
                    )
                    return
                self._condition.wait(remaining)

    def _play_all(self) -> None:
        while not self._stop_event.is_set():
            if self._buffer is not None:
                self._play_from_buffer()
            else:
                self._play_files()
            if not self._repeat or self._stop_event.is_set():
                return
            # 循环：重置投递游标，让 read() 能继续拿到"新"帧
            with self._condition:
                self._delivered = 0
                self._frame = None
                self._condition.notify_all()
            LOGGER.info("回放循环：重新播放 %s", self._log.flight_dir)

    def _play_files(self) -> None:
        """按文件播放：时间戳以帧索引为准，逐帧投递。"""
        self._set_state("playing")
        records = self._log.frames
        if not records:
            return
        base_capture = records[0].capture_timestamp
        base_wall = time.monotonic()
        last_wall = base_wall
        for record in records:
            if self._stop_event.is_set():
                return
            if self._speed > 0:
                target = base_wall + (record.capture_timestamp - base_capture) / self._speed
                if not self._sleep_until(target):
                    return
            else:
                # 全速：只保证一个最小节奏，避免生产线程把整段素材"一口气"播完
                # 而让 read() 看不到中间帧（见 _MIN_FRAME_INTERVAL_S 说明）。
                if not self._sleep_until(last_wall + _MIN_FRAME_INTERVAL_S):
                    return
                last_wall = time.monotonic()
            if self._telemetry is not None:
                if record.capture_timestamp > self._telemetry.cover_to():
                    self._telemetry_published += self._telemetry.finish(record.capture_timestamp)
                self._telemetry_published += self._telemetry.publish_until(record.capture_timestamp)
            image = self._read_image(record)
            if image is None:
                self._skip(record, "帧文件缺失或无法解码")
                continue
            frame = VideoFrame(
                index=record.index,
                image=image,
                timestamp=record.timestamp,
                lag=record.lag,
            )
            self._publish(frame)
        if self._tail_s > 0:
            self._sleep_until(time.monotonic() + self._tail_s)

    def _play_from_buffer(self) -> None:
        """跟随缓冲播放（联机调试用）。

        时间戳取自缓冲记录（``wait_new`` 逐条消费，不跳帧），帧索引只用于
        判断"素材该有的帧是否都播完了"。注意缓冲里的记录**已经是对齐结果**
        （自带遥测），所以这条路径一般不接 :class:`AlignmentWriter`，
        而是直接消费画面。
        """
        assert self._buffer is not None
        self._set_state("playing")
        expected = {record.index for record in self._log.frames}
        cursor = 0
        last_capture: float | None = None
        while not self._stop_event.is_set():
            record = self._buffer.wait_new(after_index=cursor, timeout=0.5)
            if record is None:
                if self._produced >= len(expected):
                    break
                continue
            cursor = record.index
            if record.index not in expected:
                continue
            if self._telemetry is not None:
                self._telemetry_published += self._telemetry.publish_until(record.capture_timestamp)
            frame = VideoFrame(
                index=record.index,
                image=record.image,
                timestamp=record.received_timestamp,
                lag=record.lag,
            )
            self._publish(frame)
            last_capture = record.capture_timestamp
        if self._telemetry is not None and last_capture is not None:
            self._telemetry_published += self._telemetry.finish(last_capture)

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------
    def _read_image(self, record: FrameRecord) -> np.ndarray | None:
        path = self._log.frames_dir / record.filename
        if not path.is_file():
            if self._strict:
                raise FlightLogError(f"帧文件不存在: {path}")
            return None
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None and self._strict:
            raise FlightLogError(f"帧文件无法解码: {path}")
        return image

    def _skip(self, record: FrameRecord, reason: str) -> None:
        with self._condition:
            self._skipped += 1
            self._stats.skipped = self._skipped
        LOGGER.error("回放跳过帧 #%d：%s", record.index, reason)

    def _publish(self, frame: VideoFrame) -> None:
        """把一帧交给 sink（锁外）并更新 latest()，与实飞源同一顺序。"""
        with self._condition:
            self._frame = frame
            self._produced += 1
            self._stats.frames = self._produced
            self._stats.telemetry_snapshots = self._telemetry_published
            self._stats.state = "playing"
            self._update_fps_locked()
            sinks = tuple(self._sinks)
            self._condition.notify_all()

        for sink in sinks:
            try:
                sink(frame)
            except Exception:
                if self._strict:
                    # strict 模式下不吞：sink 抛异常通常意味着配置/接线错了，
                    # 静默跳过只会让人以为"回放跑通了"。
                    raise
                LOGGER.exception("回放帧 sink 执行失败")

    def _update_fps_locked(self) -> None:
        """按回放墙钟统计实际帧率（``speed=0`` 时就是处理吞吐）。"""
        now = time.monotonic()
        if self._fps_start is None:
            self._fps_start = now
            self._fps_frames = 1
            return
        self._fps_frames += 1
        elapsed = now - self._fps_start
        if elapsed >= 1.0:
            self._stats.fps = self._fps_frames / elapsed
            self._fps_start = now
            self._fps_frames = 0

    def _sleep_until(self, target: float) -> bool:
        """睡到 ``target``（monotonic）；期间被 stop 唤醒则返回 False。"""
        while True:
            remaining = target - time.monotonic()
            if remaining <= 0:
                return True
            with self._condition:
                if self._stop_event.is_set():
                    return False
                self._condition.wait(min(remaining, _WAIT_SLICE))

    def _material_duration(self) -> float:
        span = self._log.time_span()
        return 0.0 if span is None else span[1] - span[0]

    def _set_state(self, state: str) -> None:
        with self._condition:
            self._stats.state = state
            self._stats.frames = self._produced
            self._stats.telemetry_snapshots = self._telemetry_published
            self._stats.skipped = self._skipped
            self._stats.tail_extended = (
                self._telemetry.tail_extended if self._telemetry is not None else 0
            )
            self._condition.notify_all()
