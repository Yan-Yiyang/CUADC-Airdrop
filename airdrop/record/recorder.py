"""飞行记录：把对齐缓冲、遥测、事件、检测同步落盘为标准飞行目录。

为什么要这套目录结构
--------------------
飞后回看、定位漏检、迭代算法——这些事全要靠"飞过那一刻的原数据"：

* 视频帧（``frames/%06d.jpg`` + 索引）：唯一能复现视觉处理现场的东西。
  零重编码：缓冲里已有 jpeg 字节（``AlignmentBuffer.read_jpeg_bytes``），
  直接落盘，避免再走一遍编码损失画质 + 多花 CPU。
* 遥测（``telemetry.jsonl``）：每条历史快照单独一行，回放时按原时间轴
  注入 broker，与视频帧在同一条时间线上对得上。
* 检测（``detections.jsonl``）：每条 Detection 与当时的 TargetPoint；P3
  只留接口，P5 PerceptionWorker 接入。
* 投放（``drops.jsonl``）：**每一次实际投放**记一条
  :class:`~airdrop.ballistics.drops.DropRecord`——投放瞬间的位置/速度/姿态/风、
  目标点、判据的前推位置与预测落点。这份数据事后**无法重建**（飞机当时怎么飞的、
  弹落在哪，都不在别的文件里），而反演弹道参数只能靠它，所以默认就写
  （:attr:`FlightRecorder.drops`，现场再手填 ``impacts.jsonl`` 提供实测落点）。
* 事件（``events.jsonl``）：状态机转移、任务上传、对齐统计、释放评估、
  投放动作、错误——结构化 JSON，方便事后脚本过滤。
* 文本日志（``flight.log``）：所有模块的 ``logging`` 输出汇总，**通过
  ``logging.FileHandler`` 挂到 root logger**，业务模块不用关心"我在被
  录制"，自然就进去了。
* 配置快照（``config_snapshot.json``）：本次飞行**全部**配置（含视频地址、
  标定文件路径等），事后能立刻看清"当时是怎么配的"。

并发与职责
----------
``FlightRecorder`` 启停期间会跑两个常驻线程：

* **frame writer**：``buffer.wait_new`` 跟随最新帧 → 取 jpeg 字节 → 写盘。
  唯一会解码的场合是 ``storage="raw"`` 缓冲（此时 ``read_jpeg_bytes`` 返回
  None，writer 用 ``cv2.imencode`` 重新编码）。写盘落后到超过缓冲容量时会
  **跳号**：跳掉的帧计入 :attr:`RecorderStats.frames_skipped` 并发 WARNING
  ——绝不静默地少录；
* **telemetry writer**：``broker.wait_next_snapshot`` 等到下一条快照 →
  按 ``telemetry_hz`` 节流（默认 10Hz）→ 写盘。30Hz 原始流不必每条都留。

事件和检测是同步 API：业务线程随时调 :attr:`FlightRecorder.events` /
:attr:`FlightRecorder.detections` 的 ``emit/append``，内部用 ``RLock`` 串行
化写入。日志通过 root handler，业务模块完全无感。

零重编码的代价
--------------
你必须接受"落盘的 jpeg 就是缓冲里的那份"——同一个 720p30 任务，默认画质
下磁盘写入约 4.5 MB/s。换更高画质（``storage="raw"``）开销 18 倍，1.5
分钟就是 13 GiB。先按 jpeg 跑，回看时发现像素级问题再临时切 raw。
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..ballistics.drops import DROPS_NAME, DropRecord
from ..config import Config, RecordConfig
from ..telemetry.broker import TelemetryBroker
from ..video.buffer import AlignmentBuffer, BufferedFrame

LOGGER = logging.getLogger(__name__)

__all__ = [
    "DROPS_NAME",
    "DetectionWriter",
    "DropWriter",
    "EventLog",
    "FlightRecorder",
    "RecorderStats",
]


# ----------------------------------------------------------------------
# 文件名 / 子目录常量
# ----------------------------------------------------------------------
FLIGHT_LOG_NAME = "flight.log"
TELEMETRY_NAME = "telemetry.jsonl"
DETECTIONS_NAME = "detections.jsonl"
EVENTS_NAME = "events.jsonl"
FRAMES_DIR = "frames"
FRAMES_INDEX_NAME = "frames_index.jsonl"
CONFIG_SNAPSHOT_NAME = "config_snapshot.json"

# frame writer 在 raw 缓冲下重新编码时使用的 JPEG 画质
_REENCODE_JPEG_QUALITY = 90
_INDEX_FLUSH_EVERY = 16
_TELEMETRY_FLUSH_EVERY = 16

# 飞行目录名的时间戳格式（与排序友好，便于 ls 浏览）
_FLIGHT_DIR_FMT = "%Y%m%d-%H%M%S"


# ----------------------------------------------------------------------
# 结构化 JSONL 写入器（事件流、检测流共用同一线程安全骨架）
# ----------------------------------------------------------------------
class _JsonlWriter:
    """线程安全的 JSONL 写入器（每条 JSON 占一行，写完即 flush）。

    设计要点：

    * 单一 ``RLock`` 串行化写入——任意线程可调 ``emit/append``；
    * 显式 ``flush``：落盘后立即可见，断电不留半行；
    * 异常只记日志、不抛出——日志写入失败不能让业务逻辑挂掉；
    * ``close()`` 幂等，``stop()`` 时调用一次即可。
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.RLock()
        # SIM115：这个句柄要活到 close()，是有意为之的长生命周期（不是漏了 with）
        self._file = open(path, "a", encoding="utf-8")  # noqa: SIM115
        self._closed = False
        self._count = 0

    @property
    def path(self) -> Path:
        return self._path

    @property
    def count(self) -> int:
        with self._lock:
            return self._count

    def _write_locked(self, record: Mapping[str, Any]) -> bool:
        """写入一条；返回是否成功。失败只记日志、不抛出。"""
        try:
            line = json.dumps(record, ensure_ascii=False, default=_json_default)
        except TypeError, ValueError:
            LOGGER.exception("无法序列化 JSONL 记录: %s", dict(record))
            return False
        try:
            self._file.write(line + "\n")
            self._file.flush()
        except OSError:
            LOGGER.exception("JSONL 写入失败: %s", self._path)
            return False
        self._count += 1
        return True

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            try:
                self._file.flush()
                self._file.close()
            except OSError:
                LOGGER.exception("JSONL 关闭失败: %s", self._path)
            self._closed = True


class EventLog(_JsonlWriter):
    """事件流写入器（``events.jsonl``）。

    任意模块、任意线程::

        recorder.events.emit("state", from_state="RECON", to_state="HOLD")
        recorder.events.emit("release", target=(37.42, -122.08), radius_m=2.0)
        recorder.events.emit("error", module="perception", message="...")

    每条记录的字段：

    * ``timestamp``（float）：本地墙钟（``time.time()``），便于与
      telemetry/frames 在同一时间轴上对位；
    * ``kind``（str）：事件类型，建议使用简短的名词短语（如 ``"state"``、
      ``"release"``、``"upload_mission"``、``"align_warning"``）；
    * 其它键值对由调用方提供（``ensure_ascii=False`` 保中文可读）。
    """

    def emit(self, kind: str, **data: Any) -> bool:
        """记录一条事件；返回是否真的写盘（False 通常只发生在磁盘满 / IO 错）。

        已关闭时调用 :meth:`close` 之后再 :meth:`emit` 会抛 :class:`RuntimeError`——
        业务模块如果延迟回调到 recorder 关闭之后，应自行 try/except。
        """
        with self._lock:
            if self._closed:
                raise RuntimeError("EventLog 已关闭，不能再 emit")
            record = {"timestamp": time.time(), "kind": kind, **data}
            return self._write_locked(record)


class DetectionWriter(_JsonlWriter):
    """检测结果写入器（``detections.jsonl``）。

    P3 阶段只把接口落好；``record`` 是任意 Mapping（含 Detection dataclass
    的 ``as_dict()`` 即可）。P5 PerceptionWorker 接入时建议传 dataclass
    实例，本写入器会自动通过 ``as_dict()`` 落盘。

    写入字段完全由调用方决定——本类不臆造额外字段。
    """

    def append(self, record: Mapping[str, Any] | Any) -> bool:
        """记录一条 Detection；dataclass / SimpleNamespace / Mapping 都接受。"""
        payload = _payload_of(record)
        if payload is None:
            LOGGER.error("DetectionWriter.append 收到不支持的类型: %r", type(record))
            return False
        with self._lock:
            if self._closed:
                LOGGER.warning("DetectionWriter 已关闭，丢弃检测")
                return False
            return self._write_locked(payload)


class DropWriter(_JsonlWriter):
    """投放记录写入器（``drops.jsonl``）——接 ``MissionRunner`` 的 ``on_drop``。

    一行一次投放，写的是 :meth:`~airdrop.ballistics.drops.DropRecord.as_dict` 的
    全部字段（**不四舍五入**：反演要用原始数值）。现场量到落点后，在同一个飞行目录里
    另写 ``impacts.jsonl`` 即可，不必改动这个文件。
    """

    def append(self, record: DropRecord | Mapping[str, Any] | Any) -> bool:
        """记录一次投放（``DropRecord`` / Mapping / 有 ``as_dict()`` 的对象）。"""
        payload = _payload_of(record)
        if payload is None:
            LOGGER.error("DropWriter.append 收到不支持的类型: %r", type(record))
            return False
        with self._lock:
            if self._closed:
                # 收尾阶段才投出去的那一次：记日志、返回 False，但不抛——调用方
                # （MissionRunner）已经投完了，异常改变不了任何事。
                LOGGER.error("DropWriter 已关闭，这次投放记录没有落盘")
                return False
            return self._write_locked(payload)


def _payload_of(record: Any) -> dict[str, Any] | None:
    """把任意"记录对象"摊成字典：``as_dict()`` > Mapping > ``__dict__`` > None。"""
    as_dict = getattr(record, "as_dict", None)
    if callable(as_dict):
        payload = as_dict()
        # 只认映射：as_dict() 返回别的东西就当没这个方法，继续往下退（不抛异常）
        if isinstance(payload, Mapping):
            return dict(payload)
    if isinstance(record, Mapping):
        return dict(record)
    if hasattr(record, "__dict__"):
        return dict(vars(record))
    return None


# ----------------------------------------------------------------------
# JSON 序列化兜底（处理 numpy 标量 / dataclass 等）
# ----------------------------------------------------------------------
def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "as_dict") and callable(value.as_dict):
        return value.as_dict()
    raise TypeError(f"无法序列化 {type(value).__name__}")


# ----------------------------------------------------------------------
# Recorder 状态汇总
# ----------------------------------------------------------------------
class RecorderStats:
    """Recorder 运行统计（线程安全地取）。"""

    __slots__ = (
        "detections",
        "drops",
        "events",
        "flight_dir",
        "frames_skipped",
        "frames_written",
        "raw_reencoded",
        "telemetry_written",
    )

    def __init__(  # noqa: PLR0917 - 统计项本身就是这个数量，位置参数与字段一一对应
        self,
        flight_dir: Path,
        frames_written: int,
        telemetry_written: int,
        events: int,
        detections: int,
        drops: int,
        raw_reencoded: int,
        frames_skipped: int = 0,
    ) -> None:
        self.flight_dir = flight_dir
        self.frames_written = frames_written
        self.telemetry_written = telemetry_written
        self.events = events
        self.detections = detections
        self.drops = drops
        self.raw_reencoded = raw_reencoded
        #: 因缓冲驱逐/写盘落后而**没写进磁盘**的帧数（>0 说明录制不完整，日志里有 WARNING）
        self.frames_skipped = frames_skipped


# ----------------------------------------------------------------------
# 主类
# ----------------------------------------------------------------------
class FlightRecorder:
    """飞行目录录制器：把对齐缓冲、遥测、事件、检测同步落盘。

    用法::

        config = Config().validated()
        recorder = FlightRecorder(config)
        recorder.start(broker=broker, buffer=buffer)
        try:
            ... # 任务主循环；随时 recorder.events.emit("...", ...)
        finally:
            recorder.stop()                  # 幂等；关掉所有 writer

    启动时的行为
    ------------
    1. 在 ``Config.record.dir`` 下创建 ``flights/<YYYYMMDD-HHMMSS>/``（重名
       自动追加 ``-1`` / ``-2`` …）；同时建 ``frames/`` 子目录；
    2. 立刻把 ``Config`` 全字段写到 ``config_snapshot.json``；
    3. 打开 ``flight.log``、``telemetry.jsonl``、``detections.jsonl``、
       ``events.jsonl``、``frames_index.jsonl`` 并启动 frame/telemetry 两个
       后台线程；
    4. 给 root logger 挂 ``FileHandler``——任何模块的 ``logger.info(...)``
       都会顺道进 ``flight.log``。

    停止时的行为
    ------------
    1. 设 ``_stop_event``，两个后台线程在 ``wait_new`` / ``wait_next_snapshot``
       的超时点退出；
    2. ``join`` 线程；
    3. 关闭所有文件句柄，摘掉 root handler；
    4. ``stats()`` 仍可读，作为"这次飞了多长"的总结。
    """

    def __init__(self, config: Config, *, base_dir: str | Path | None = None) -> None:
        self._config = config
        record = config.record
        self._base_dir = Path(base_dir) if base_dir is not None else Path(record.dir)
        self._telemetry_hz = max(float(record.telemetry_hz), 0.1)

        self._flight_dir: Path | None = None
        self._frames_dir: Path | None = None

        # 文件句柄（启动后才有）
        self._log_handler: logging.FileHandler | None = None
        self._telemetry_file = None
        self._frames_index_file = None
        self._events: EventLog | None = None
        self._detections: DetectionWriter | None = None
        self._drops: DropWriter | None = None

        # 线程
        self._frame_thread: threading.Thread | None = None
        self._telemetry_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._started_lock = threading.Lock()

        # 运行统计
        self._frames_written = 0
        self._frames_skipped = 0
        self._telemetry_written = 0
        self._raw_reencoded = 0
        self._events_written = 0
        self._detections_written = 0
        self._drops_written = 0

        # 依赖（start 时绑定）
        self._broker: TelemetryBroker | None = None
        self._buffer: AlignmentBuffer | None = None

    # ------------------------------------------------------------------
    # 状态
    # ------------------------------------------------------------------
    @property
    def is_running(self) -> bool:
        return self._flight_dir is not None and not self._stop_event.is_set()

    @property
    def flight_dir(self) -> Path | None:
        """本次飞行的根目录；启动前为 None。"""
        return self._flight_dir

    @property
    def events(self) -> EventLog:
        """事件流写入器；启动后才有（未启动时访问会抛 RuntimeError）。"""
        if self._events is None:
            raise RuntimeError("FlightRecorder 尚未启动")
        return self._events

    @property
    def detections(self) -> DetectionWriter:
        """检测流写入器；启动后才有。"""
        if self._detections is None:
            raise RuntimeError("FlightRecorder 尚未启动")
        return self._detections

    @property
    def drops(self) -> DropWriter:
        """投放记录写入器（``drops.jsonl``）；启动后才有。

        接法是 ``MissionRunner(..., on_drop=recorder.drops.append)``。
        """
        if self._drops is None:
            raise RuntimeError("FlightRecorder 尚未启动")
        return self._drops

    @property
    def record_config(self) -> RecordConfig:
        return self._config.record

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    def start(
        self,
        *,
        broker: TelemetryBroker,
        buffer: AlignmentBuffer,
    ) -> "FlightRecorder":
        """创建飞行目录、打开五件套 + 投放记录、启动后台线程。

        幂等：重复调用直接返回 self。``broker`` / ``buffer`` 在第一次启动时
        必须给出，后续启动会用相同的依赖（重新 start 会复用已有目录，但
        不会改依赖绑定——这是有意为之，避免跑一半换 broker 把数据写错位置）。
        """
        with self._started_lock:
            if self.is_running:
                LOGGER.debug("FlightRecorder 已在运行，跳过 start: %s", self._flight_dir)
                return self
            if self._flight_dir is not None:
                # 已 stop 过的实例不复用旧目录
                raise RuntimeError(
                    f"FlightRecorder 已停止于 {self._flight_dir}，"
                    "请新建一个 FlightRecorder 实例再录新一次"
                )

            self._broker = broker
            self._buffer = buffer
            self._flight_dir = self._make_flight_dir()
            self._frames_dir = self._flight_dir / FRAMES_DIR
            self._frames_dir.mkdir(parents=False, exist_ok=False)

            self._dump_config_snapshot()
            self._open_files()
            self._install_log_handler()

            self._stop_event.clear()
            self._frame_thread = threading.Thread(
                target=self._frame_writer_loop,
                name="recorder-frames",
                daemon=True,
            )
            self._telemetry_thread = threading.Thread(
                target=self._telemetry_writer_loop,
                name="recorder-telemetry",
                daemon=True,
            )
            self._frame_thread.start()
            self._telemetry_thread.start()

            self.events.emit(
                "recorder_started",
                flight_dir=str(self._flight_dir),
                telemetry_hz=self._telemetry_hz,
                buffer_storage=buffer.storage,
                buffer_capacity=buffer.capacity,
            )
            LOGGER.info(
                "FlightRecorder 已启动：%s（帧 %s，遥测 %.1fHz）",
                self._flight_dir,
                buffer.storage,
                self._telemetry_hz,
            )
            # 显式 flush：让 flight.log 立刻可见，便于调试
            if self._log_handler is not None:
                self._log_handler.flush()
        return self

    def stop(self, timeout: float = 5.0) -> None:
        """停掉后台线程、关闭所有文件（幂等）。"""
        if self._flight_dir is None and not self._stop_event.is_set():
            return  # 从未启动过
        self._stop_event.set()
        alive = False
        for thread in (self._frame_thread, self._telemetry_thread):
            if thread is not None and thread.is_alive():
                thread.join(timeout)
                if thread.is_alive():
                    alive = True
                    LOGGER.warning("recorder 线程未在 %.1fs 内退出: %s", timeout, thread.name)
        if alive:
            LOGGER.warning("后台写入线程仍在运行，保留文件句柄，稍后可再次调用 stop() 收尾")
            return
        self._frame_thread = None
        self._telemetry_thread = None

        # 顺序：先收尾事件、摘 handler、再关文件（避免 flush 时 handler 已被摘）
        if self._events is not None:
            try:
                self._events.emit(
                    "recorder_stopped",
                    frames_written=self._frames_written,
                    telemetry_written=self._telemetry_written,
                    events=self._events.count,
                    detections=self._detections.count if self._detections else 0,
                    drops=self._drops.count if self._drops else 0,
                )
            except Exception:
                LOGGER.exception("recorder_stopped 事件写入失败")

        self._uninstall_log_handler()
        self._close_files()

    def stats(self) -> RecorderStats | None:
        """运行统计快照；未启动返回 None。"""
        if self._flight_dir is None:
            return None
        # 优先取实时计数（运行中）；stop 后 _events/_detections 会被置 None，
        # 此时用缓存字段（_events_written / _detections_written）。
        events = self._events.count if self._events else self._events_written
        detections = self._detections.count if self._detections else self._detections_written
        drops = self._drops.count if self._drops else self._drops_written
        return RecorderStats(
            flight_dir=self._flight_dir,
            frames_written=self._frames_written,
            telemetry_written=self._telemetry_written,
            events=events,
            detections=detections,
            drops=drops,
            raw_reencoded=self._raw_reencoded,
            frames_skipped=self._frames_skipped,
        )

    # ------------------------------------------------------------------
    # 后台线程
    # ------------------------------------------------------------------
    def _frame_writer_loop(self) -> None:
        """跟随缓冲逐帧落盘（零重编码优先；raw 缓冲下重新编码）。

        ``wait_new`` 只能返回"当前保留的最早一条"：一旦写盘落后到超过缓冲
        容量（默认 5400 帧 ≈3 分钟），被驱逐的帧就补不回来了。所以这里显式
        检查序号跳变——跳号只记账 + 告警，不假装"录全了"。
        """
        assert self._buffer is not None and self._frames_dir is not None
        buffer = self._buffer
        frames_dir = self._frames_dir
        index_file = self._frames_index_file
        assert index_file is not None
        cursor = 0
        while not self._stop_event.is_set():
            frame = buffer.wait_new(after_index=cursor, timeout=1.0)
            if frame is None:
                continue
            if frame.index > cursor + 1:
                skipped = frame.index - cursor - 1
                self._frames_skipped += skipped
                LOGGER.warning(
                    "录制跳号：序号 %d→%d 之间的 %d 帧未落盘"
                    "（写盘落后于缓冲容量，或录制在帧产生之后才启动）；"
                    "飞行目录里这几帧缺失，统计见 frames_skipped",
                    cursor + 1,
                    frame.index,
                    skipped,
                )
            cursor = frame.index
            try:
                self._write_one_frame(frame, frames_dir, index_file)
            except Exception:
                LOGGER.exception("录制帧 #%d 失败", frame.index)

    def _write_one_frame(
        self,
        frame: BufferedFrame,
        frames_dir: Path,
        index_file,
    ) -> None:
        assert self._buffer is not None
        jpeg = self._buffer.read_jpeg_bytes(frame.index)
        if jpeg is None:
            # raw 缓冲：内部存的是 ndarray，自己编码一次
            ok, encoded = cv2.imencode(
                ".jpg",
                frame.image,
                [int(cv2.IMWRITE_JPEG_QUALITY), _REENCODE_JPEG_QUALITY],
            )
            if not ok:
                LOGGER.warning("raw 缓冲帧 #%d 编码失败，跳过", frame.index)
                return
            jpeg = encoded.tobytes()
            self._raw_reencoded += 1

        filename = f"{frame.index:06d}.jpg"
        (frames_dir / filename).write_bytes(jpeg)

        index_line = json.dumps(
            {
                "index": frame.index,
                "filename": filename,
                "capture_timestamp": frame.capture_timestamp,
                "received_timestamp": frame.received_timestamp,
                "lag": frame.lag,
                "extrapolated": frame.extrapolated,
                "offset": frame.offset,
                "bytes": len(jpeg),
            },
            ensure_ascii=False,
        )
        index_file.write(index_line + "\n")
        self._frames_written += 1
        if self._frames_written % _INDEX_FLUSH_EVERY == 0:
            index_file.flush()

    def _telemetry_writer_loop(self) -> None:
        """按 ``telemetry_hz`` 节流写盘。"""
        assert self._broker is not None and self._telemetry_file is not None
        broker = self._broker
        out = self._telemetry_file
        interval = 1.0 / self._telemetry_hz
        last_write_ts = 0.0
        while not self._stop_event.is_set():
            snapshot = broker.wait_next_snapshot(
                predicate=lambda s, last=last_write_ts: (
                    # 一条遥测都没收到时的初始空快照不能入库：回放时它会成为
                    # 历史区间最左端的一个"全 None"端点，让最先几帧内插到
                    # None。正式飞行里它出现在第一帧遥测之前，会被这里挡掉。
                    s.is_valid() and s.timestamp >= last + interval
                ),
                timeout=1.0,
            )
            if snapshot is None:
                continue
            last_write_ts = snapshot.timestamp
            try:
                out.write(
                    json.dumps(snapshot.as_dict(), ensure_ascii=False, default=_json_default) + "\n"
                )
                self._telemetry_written += 1
                if self._telemetry_written % _TELEMETRY_FLUSH_EVERY == 0:
                    out.flush()
            except OSError:
                LOGGER.exception("telemetry 写入失败")
                continue

    # ------------------------------------------------------------------
    # 目录 / 文件
    # ------------------------------------------------------------------
    def _make_flight_dir(self) -> Path:
        base = self._base_dir
        base.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime(_FLIGHT_DIR_FMT)
        candidate = base / stamp
        suffix = 0
        while candidate.exists():
            suffix += 1
            candidate = base / f"{stamp}-{suffix}"
        candidate.mkdir(parents=False)
        return candidate

    def _dump_config_snapshot(self) -> None:
        assert self._flight_dir is not None
        path = self._flight_dir / CONFIG_SNAPSHOT_NAME
        path.write_text(
            json.dumps(
                asdict(self._config),
                ensure_ascii=False,
                indent=2,
                default=_json_default,
            ),
            encoding="utf-8",
        )

    def _open_files(self) -> None:
        assert self._flight_dir is not None
        self._events = EventLog(self._flight_dir / EVENTS_NAME)
        self._detections = DetectionWriter(self._flight_dir / DETECTIONS_NAME)
        self._drops = DropWriter(self._flight_dir / DROPS_NAME)
        # SIM115：两个句柄都活到 _close_files()，长生命周期是有意为之
        self._telemetry_file = open(  # noqa: SIM115
            self._flight_dir / TELEMETRY_NAME, "a", encoding="utf-8"
        )
        self._frames_index_file = open(  # noqa: SIM115
            self._flight_dir / FRAMES_INDEX_NAME, "a", encoding="utf-8"
        )

    def _close_files(self) -> None:
        if self._events is not None:
            self._events_written = self._events.count
            self._events.close()
            self._events = None
        if self._detections is not None:
            self._detections_written = self._detections.count
            self._detections.close()
            self._detections = None
        if self._drops is not None:
            self._drops_written = self._drops.count
            self._drops.close()
            self._drops = None
        for attr in ("_telemetry_file", "_frames_index_file"):
            handle = getattr(self, attr)
            if handle is not None and not handle.closed:
                try:
                    handle.flush()
                    handle.close()
                except OSError:
                    LOGGER.exception("文件关闭失败: %s", attr)
            setattr(self, attr, None)

    # ------------------------------------------------------------------
    # 日志 handler（flight.log）
    # ------------------------------------------------------------------
    def _install_log_handler(self) -> None:
        assert self._flight_dir is not None
        handler = logging.FileHandler(self._flight_dir / FLIGHT_LOG_NAME, encoding="utf-8")
        handler.setLevel(logging.INFO)
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s %(levelname)-7s %(name)s %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        root = logging.getLogger()
        # 默认 root level 是 WARNING：INFO 消息根本到不了任何 handler。
        # 这里把 level 临时降到 INFO（start 时记录原值，stop 时还原），
        # 让所有业务模块的 logger.info(...) 自动汇总到 flight.log——
        # 既不要求业务模块知道自己在被录制，又能完整留痕。
        self._original_root_level = root.level
        root.setLevel(logging.INFO)
        root.addHandler(handler)
        self._log_handler = handler

    def _uninstall_log_handler(self) -> None:
        handler = self._log_handler
        if handler is None:
            return
        try:
            logging.getLogger().removeHandler(handler)
        except Exception:
            LOGGER.exception("移除 log handler 失败")
        try:
            handler.flush()
            handler.close()
        except OSError:
            LOGGER.exception("log handler 关闭失败")
        logging.getLogger().setLevel(self._original_root_level)
        self._log_handler = None
