"""``FlightRecorder`` 的单元测试。

所有用例都跑在 ``pytest`` 的 ``tmp_path`` 下，离线——不需要飞控、不需要
ffmpeg、不需要硬件。手动往 ``AlignmentBuffer`` 传入帧、往 ``TelemetryBroker``
灌数据，验证五个记录文件 + 投放记录写入磁盘 + 帧零重编码 + 线程安全 + 幂等行为。
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from airdrop import (
    AlignedSample,
    AlignmentBuffer,
    AlignmentWriter,
    Config,
    DetectionWriter,
    DropRecord,
    EventLog,
    FlightRecorder,
    FrameTelemetryAligner,
    Hm30VideoSource,
    TelemetryBroker,
    TelemetrySnapshot,
    VideoFrame,
)
from airdrop.config import RecordConfig

pytestmark = pytest.mark.integration


# ----------------------------------------------------------------------
# 构造工具
# ----------------------------------------------------------------------
def _make_image(seed: int) -> np.ndarray:
    """生成一张 16x16、颜色由 seed 决定的图像——容易目视判断零重编码。"""
    np.random.seed(seed)
    return np.random.randint(0, 256, (16, 16, 3), dtype=np.uint8)


def _make_sample(index: int, capture_ts: float | None = None) -> AlignedSample:
    """构造一条 ``AlignedSample``（与 AlignedSample 真实字段对齐）。"""
    if capture_ts is None:
        capture_ts = 1_000_000.0 + index * 0.1
    image = _make_image(index)
    frame = VideoFrame(
        index=index,
        image=image,
        timestamp=capture_ts + 0.15,
        lag=0.15,
    )
    snapshot = SimpleNamespace(timestamp=capture_ts)
    return AlignedSample(
        frame=frame,
        snapshot=snapshot,
        timestamp=capture_ts,
        lag=0.15,
        mode="interpolate",
    )


def _bump_broker(broker: TelemetryBroker, ts: float) -> None:
    """往 broker 灌一条极简遥测（只需更新 latest）。"""
    pos = SimpleNamespace(north_m=1.0, east_m=2.0, down_m=3.0)
    vel = SimpleNamespace(north_m_s=0.1, east_m_s=0.2, down_m_s=0.0)
    broker.update_local_position_velocity(pos, vel)
    # broker 会用 time.time() 作 timestamp；为节流测试的可控性改一下
    # （走内部最近一次 _update，但 timestamp 是 time.time()，控制不了）。
    # 因此下面的节流测试用绝对数量断言而非时间窗口。


@dataclass
class _FakeDetection:
    """用于测试 DetectionWriter 接受 dataclass 的最小样本。"""

    code: int
    confidence: float
    frame_index: int = field(default=0)

    def as_dict(self) -> dict[str, Any]:
        return {"code": self.code, "confidence": self.confidence, "frame_index": self.frame_index}


def _wait_until(predicate, timeout: float = 4.0, interval: float = 0.05) -> bool:
    """轮询直到 ``predicate()`` 为真或超时（避免硬编码 sleep）。"""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


# ----------------------------------------------------------------------
# 核心场景
# ----------------------------------------------------------------------
def test_recorder_rolls_back_partial_start(tmp_path: Path, broker, monkeypatch) -> None:
    """打开部分文件后失败，不能伪装为运行中或泄漏句柄。"""
    recorder = FlightRecorder(Config().validated(), base_dir=tmp_path)
    handles = []

    def fail():
        recorder._events = EventLog(tmp_path / "partial.jsonl")
        handles.append(recorder._events._file)
        raise OSError("模拟打开失败")

    monkeypatch.setattr(recorder, "_open_files", fail)
    with pytest.raises(OSError, match="模拟打开失败"):
        recorder.start(broker=broker, buffer=AlignmentBuffer(capacity=4))
    try:
        assert not recorder.is_running
        assert handles[0].closed
    finally:
        recorder.stop()


def test_recorder_stop_drains_tail_and_retains_resources_on_timeout(
    tmp_path: Path, broker, monkeypatch
) -> None:
    """磁盘写入被阻塞时保留资源，解除阻塞后再 stop 必须录完截止尾帧。"""
    buffer = AlignmentBuffer(capacity=8)
    for index in range(1, 5):
        buffer.put(_make_sample(index))
    recorder = FlightRecorder(Config().validated(), base_dir=tmp_path)
    entered, release = threading.Event(), threading.Event()
    original = recorder._write_one_frame

    def write(frame, frames_dir, index_file):
        entered.set()
        assert release.wait(5)
        original(frame, frames_dir, index_file)

    monkeypatch.setattr(recorder, "_write_one_frame", write)
    recorder.start(broker=broker, buffer=buffer)
    try:
        assert entered.wait(2)
        recorder.stop(timeout=0.01)
        assert recorder._frame_thread is not None
        assert not recorder._frames_index_file.closed
        assert recorder.is_running
    finally:
        release.set()
        recorder.stop()
    assert recorder.stats().frames_written == 4
    assert not recorder.is_running


def test_jsonl_encoding_failure_does_not_escape_or_poison_next_record(tmp_path: Path) -> None:
    """非法 Unicode 记录失败返回 False，下一条有效记录仍能正常写盘。"""
    writer = DetectionWriter(tmp_path / "encoding.jsonl")
    try:
        assert writer.append({"text": "\ud800"}) is False
        assert writer.append({"text": "物资"}) is True
        assert writer.count == 1
    finally:
        writer.close()
    assert json.loads(writer.path.read_text(encoding="utf-8")) == {"text": "物资"}


def test_recorder_rolls_back_second_thread_start_failure(
    tmp_path: Path, broker, monkeypatch
) -> None:
    """第一条线程已启动、第二条启动失败，线程和 root handler 都须回收。"""
    recorder = FlightRecorder(Config().validated(), base_dir=tmp_path)
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    original = threading.Thread.start

    def start(thread):
        if thread.name == "recorder-telemetry":
            raise RuntimeError("模拟第二线程失败")
        original(thread)

    monkeypatch.setattr(threading.Thread, "start", start)
    with pytest.raises(RuntimeError, match="第二线程失败"):
        recorder.start(broker=broker, buffer=AlignmentBuffer(capacity=4))
    assert not recorder.is_running
    assert recorder._frame_thread is None
    assert recorder._events is None
    assert root.handlers == handlers and root.level == level


def _full_update_recorder(tmp_path: Path) -> FlightRecorder:
    config = Config()
    config = replace(config, record=replace(config.record, telemetry_hz=0.0))
    return FlightRecorder(config.validated(), base_dir=tmp_path)


def _publish_full_update(broker: TelemetryBroker, index: int) -> None:
    broker.publish(
        TelemetrySnapshot(
            timestamp=100.0 + index / 100.0,
            quaternion_timestamp_us=1_000_000 + index * 10_000,
            quaternion_received_at=100.0 + index / 100.0,
            quaternion_w=1.0,
            quaternion_x=0.0,
            quaternion_y=0.0,
            quaternion_z=0.0,
        )
    )


def test_full_update_recorder_saves_every_broker_publication(tmp_path: Path) -> None:
    """全量模式经 broker 订阅逐条落盘，不按 writer 醒来时的最新值抽样。"""
    broker = TelemetryBroker()
    recorder = _full_update_recorder(tmp_path)
    recorder.start(broker=broker, buffer=AlignmentBuffer(capacity=4))
    for index in range(1_000):
        _publish_full_update(broker, index)
    recorder.stop()
    stats = recorder.stats()
    assert stats is not None
    assert stats.telemetry_received == stats.telemetry_enqueued == stats.telemetry_written == 1_000
    assert stats.telemetry_dropped == stats.telemetry_write_errors == stats.telemetry_pending == 0
    rows = [
        json.loads(line)
        for line in (recorder.flight_dir / "telemetry.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(rows) == 1_000
    assert [row["quaternion_timestamp_us"] for row in rows] == [
        1_000_000 + index * 10_000 for index in range(1_000)
    ]


def test_full_update_stop_timeout_retry_drains_accepted_tail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """慢 writer 超时后保留资源，重试 stop 会写完所有已接纳样本。"""
    broker = TelemetryBroker()
    recorder = _full_update_recorder(tmp_path)
    entered, release = threading.Event(), threading.Event()
    original = recorder._full_telemetry_writer_loop

    def slow_writer() -> None:
        entered.set()
        assert release.wait(5)
        original()

    monkeypatch.setattr(recorder, "_full_telemetry_writer_loop", slow_writer)
    recorder.start(broker=broker, buffer=AlignmentBuffer(capacity=4))
    for index in range(20):
        _publish_full_update(broker, index)
    assert entered.wait(2)
    recorder.stop(timeout=0.01)
    assert recorder._telemetry_thread is not None
    assert recorder._telemetry_file is not None and not recorder._telemetry_file.closed
    release.set()
    recorder.stop()
    stats = recorder.stats()
    assert stats is not None and stats.telemetry_written == 20
    assert stats.telemetry_pending == stats.telemetry_dropped == 0


def test_full_update_start_failure_unsubscribes_and_ignores_late_callback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """启动事件失败会回滚订阅，broker 后续发布不会再进已回滚记录器。"""
    broker = TelemetryBroker()
    recorder = _full_update_recorder(tmp_path)
    emit = EventLog.emit

    def fail_started(event_log, kind, **data):
        if kind == "recorder_started":
            raise OSError("模拟启动事件失败")
        return emit(event_log, kind, **data)

    monkeypatch.setattr(EventLog, "emit", fail_started)
    with pytest.raises(OSError, match="启动事件失败"):
        recorder.start(broker=broker, buffer=AlignmentBuffer(capacity=4))
    assert recorder._telemetry_callback not in broker._subscribers
    _publish_full_update(broker, 1)
    stats = recorder.stats()
    assert stats is not None and stats.telemetry_received == 0
    assert not recorder.is_running


def test_full_update_stop_rejects_callback_already_in_broker_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """已被 broker 复制的迟到回调在关闭接纳后不能进入队列。"""
    broker = TelemetryBroker()
    recorder = _full_update_recorder(tmp_path)
    entered, release = threading.Event(), threading.Event()
    original = recorder._telemetry_callback

    def delayed_callback(snapshot: TelemetrySnapshot) -> None:
        entered.set()
        assert release.wait(5)
        original(snapshot)

    monkeypatch.setattr(recorder, "_telemetry_callback", delayed_callback)
    recorder.start(broker=broker, buffer=AlignmentBuffer(capacity=4))
    publisher = threading.Thread(target=_publish_full_update, args=(broker, 1))
    publisher.start()
    assert entered.wait(2)
    stop_done = threading.Event()

    def stop() -> None:
        recorder.stop()
        stop_done.set()

    stopper = threading.Thread(target=stop)
    stopper.start()
    assert _wait_until(recorder._stop_event.is_set)
    release.set()
    publisher.join(2)
    stopper.join(2)
    assert stop_done.is_set()
    stats = recorder.stats()
    assert stats is not None and stats.telemetry_received == stats.telemetry_written == 0


def test_full_update_queue_overflow_is_reported_as_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """满队列不静默丢样本，stop 明确报告标定采集不完整。"""
    broker = TelemetryBroker()
    recorder = _full_update_recorder(tmp_path)
    recorder._telemetry_queue = queue.Queue(maxsize=1)
    entered, release = threading.Event(), threading.Event()
    original = recorder._full_telemetry_writer_loop

    def blocked_writer() -> None:
        entered.set()
        assert release.wait(5)
        original()

    monkeypatch.setattr(recorder, "_full_telemetry_writer_loop", blocked_writer)
    recorder.start(broker=broker, buffer=AlignmentBuffer(capacity=4))
    assert entered.wait(2)
    _publish_full_update(broker, 1)
    _publish_full_update(broker, 2)
    release.set()
    with pytest.raises(RuntimeError, match="dropped=1"):
        recorder.stop()
    stats = recorder.stats()
    assert stats is not None and stats.telemetry_received == 2
    assert stats.telemetry_enqueued == stats.telemetry_written == 1
    assert stats.telemetry_dropped == 1


@pytest.mark.parametrize("failure", ["write", "encode", "flush"])
def test_full_update_writer_failures_are_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    """全量记录的序列化、写盘和最终 flush 错误都让停止返回失败。"""
    broker = TelemetryBroker()
    recorder = _full_update_recorder(tmp_path)
    if failure in {"write", "flush"}:
        original_open = recorder._open_files

        class FailingFile:
            def __init__(self, handle) -> None:
                self.handle = handle
                self.closed = False

            def write(self, value: str) -> int:
                if failure == "write":
                    raise OSError("模拟遥测写入失败")
                return self.handle.write(value)

            def flush(self) -> None:
                if failure == "flush":
                    raise OSError("模拟遥测 flush 失败")
                self.handle.flush()

            def close(self) -> None:
                self.handle.close()
                self.closed = True

        def open_files() -> None:
            original_open()
            recorder._telemetry_file = FailingFile(recorder._telemetry_file)

        monkeypatch.setattr(recorder, "_open_files", open_files)
    else:
        monkeypatch.setattr(
            TelemetrySnapshot,
            "as_dict",
            lambda self: (_ for _ in ()).throw(TypeError("模拟遥测编码失败")),
        )
    recorder.start(broker=broker, buffer=AlignmentBuffer(capacity=4))
    _publish_full_update(broker, 1)
    with pytest.raises(RuntimeError, match="write_errors=[1-9]"):
        recorder.stop()
    stats = recorder.stats()
    assert stats is not None and stats.telemetry_write_errors > 0
    event_rows = [
        json.loads(line)
        for line in (recorder.flight_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert event_rows[-1]["kind"] == "recorder_stopped"
    assert event_rows[-1]["telemetry_write_errors"] > 0


def test_full_update_frame_failure_is_reported_as_incomplete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """全量遥测模式也把视频帧失败作为标定采集不完整处理。"""
    recorder = _full_update_recorder(tmp_path)
    buffer = AlignmentBuffer(capacity=4)
    buffer.put(_make_sample(1))

    def fail_frame(*args, **kwargs):
        raise OSError("模拟帧写入失败")

    monkeypatch.setattr(recorder, "_write_one_frame", fail_frame)
    recorder.start(broker=TelemetryBroker(), buffer=buffer)
    assert _wait_until(lambda: recorder._frames_skipped == 1)
    with pytest.raises(RuntimeError, match="frames_skipped=1"):
        recorder.stop()


def test_full_update_detects_frame_failure_before_buffer_index(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """源吞掉中间帧入库异常后，连续缓冲序号不能掩盖缺帧。"""
    broker = TelemetryBroker()
    buffer = AlignmentBuffer(capacity=8)
    source = Hm30VideoSource()
    source.add_sink(AlignmentWriter(buffer, FrameTelemetryAligner(broker), capture_only=True))
    encode = buffer._encode
    calls = 0

    def fail_middle(image):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("模拟入库编码失败")
        return encode(image)

    monkeypatch.setattr(buffer, "_encode", fail_middle)
    recorder = _full_update_recorder(tmp_path)
    recorder.start(broker=broker, buffer=buffer)
    for index in range(3):
        source._publish(_make_image(index))
    with pytest.raises(RuntimeError, match="frame_encode_errors=1"):
        recorder.stop()
    stats = recorder.stats()
    assert stats is not None
    assert source.stats.frames == 3
    assert stats.frames_written == buffer.stats.last_index == 2
    assert stats.frames_skipped == 0
    assert stats.frame_encode_errors == 1
    assert recorder.flight_dir is not None
    events = [
        json.loads(line)
        for line in (recorder.flight_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert events[-1]["frame_encode_errors"] == 1


@pytest.mark.parametrize("failure", ["flush", "close"])
def test_full_update_index_finalization_failure_is_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    """帧索引最终 flush 或 close 失败必须写入收尾事件并拒绝完整记录。"""
    recorder = _full_update_recorder(tmp_path)
    open_files = recorder._open_files

    class FailingIndex:
        def __init__(self, handle) -> None:
            self.handle = handle

        @property
        def closed(self) -> bool:
            return self.handle.closed

        def flush(self) -> None:
            if failure == "flush":
                raise OSError("模拟帧索引 flush 失败")
            self.handle.flush()

        def close(self) -> None:
            self.handle.close()
            if failure == "close":
                raise OSError("模拟帧索引 close 失败")

    def open_failing_index() -> None:
        open_files()
        recorder._frames_index_file = FailingIndex(recorder._frames_index_file)

    monkeypatch.setattr(recorder, "_open_files", open_failing_index)
    recorder.start(broker=TelemetryBroker(), buffer=AlignmentBuffer(capacity=4))
    with pytest.raises(RuntimeError, match="finalize_errors=1"):
        recorder.stop()
    assert recorder.flight_dir is not None
    events = [
        json.loads(line)
        for line in (recorder.flight_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert events[-1]["finalize_errors"] == 1


def test_recorder_counts_failed_frame_writes(tmp_path: Path, broker, monkeypatch) -> None:
    """编码失败与写盘失败都记入缺帧统计，不影响后续帧。"""
    buffer = AlignmentBuffer(capacity=4, storage="raw")
    for index in range(1, 4):
        buffer.put(_make_sample(index))
    recorder = FlightRecorder(Config().validated(), base_dir=tmp_path)
    original = recorder._write_one_frame

    def write(frame, frames_dir, index_file):
        if frame.index == 1:
            raise OSError("模拟磁盘失败")
        original(frame, frames_dir, index_file)

    monkeypatch.setattr(recorder, "_write_one_frame", write)
    recorder.start(broker=broker, buffer=buffer)
    recorder.stop()
    stats = recorder.stats()
    assert stats is not None
    assert stats.frames_written == 2 and stats.frames_skipped == 1


def test_recorder_writes_full_flight_directory(tmp_path: Path, broker: TelemetryBroker) -> None:
    """完整生命周期：start → 传入帧/遥测/事件/检测 → stop → 五个记录文件齐全。"""
    buffer = AlignmentBuffer(capacity=32, storage="jpeg")
    config = Config().validated()
    recorder = FlightRecorder(config, base_dir=tmp_path)

    # 启动前访问 events/detections 应该报错
    with pytest.raises(RuntimeError):
        _ = recorder.events
    with pytest.raises(RuntimeError):
        _ = recorder.detections

    assert recorder.flight_dir is None
    assert not recorder.is_running

    recorder.start(broker=broker, buffer=buffer)
    try:
        assert recorder.is_running
        flight_dir = recorder.flight_dir
        assert flight_dir is not None
        assert flight_dir.exists()
        assert (flight_dir / "frames").is_dir()
        assert (flight_dir / "config_snapshot.json").exists()  # 立刻写入磁盘

        # 5 帧
        for i in range(1, 6):
            buffer.put(_make_sample(i))

        # 灌一批遥测（30Hz × 1s 模拟，recorder 10Hz 节流）
        for _ in range(30):
            _bump_broker(broker, time.time())
            time.sleep(0.005)

        # 事件 + 检测 + 投放（中间夹杂）
        recorder.events.emit("state", from_state="RECON", to_state="HOLD")
        recorder.detections.append({"frame_index": 1, "code": 42, "confidence": 0.9})
        recorder.detections.append(_FakeDetection(code=7, confidence=0.85, frame_index=2))
        recorder.drops.append(
            DropRecord(
                index=1,
                timestamp=time.time(),
                position_ned=(120.0, -35.0, -20.0),
                velocity_ned=(0.0, 18.0, 0.0),
                ground_z=0.0,
                roll_deg=4.0,
                pitch_deg=-3.0,
                yaw_deg=90.0,
                wind_ned=(2.0, -1.0, 0.0),
                reason="predict",
                delay_s=0.08,
                predicted_impact_ned=(0.4, 0.2, 0.0),
                predicted_flight_time_s=2.1,
                predicted_error_m=0.45,
            )
        )

        # 等 recorder 把帧全部写入磁盘
        assert _wait_until(
            lambda: recorder._frames_written == 5,
            timeout=4.0,
        ), f"frame writer 未追上：{recorder._frames_written}/5"

    finally:
        recorder.stop()

    # ---- stop 后校验 ----
    assert not recorder.is_running
    flight_dir = recorder.flight_dir
    assert flight_dir is not None

    expected_files = {
        flight_dir / "flight.log",
        flight_dir / "telemetry.jsonl",
        flight_dir / "detections.jsonl",
        flight_dir / "drops.jsonl",
        flight_dir / "events.jsonl",
        flight_dir / "frames_index.jsonl",
        flight_dir / "config_snapshot.json",
    }
    missing = {p for p in expected_files if not p.exists()}
    assert not missing, f"五个记录文件 + 投放记录缺失: {missing}"

    # 帧文件齐
    frame_files = sorted((flight_dir / "frames").glob("*.jpg"))
    assert len(frame_files) == 5
    assert frame_files[0].name == "000001.jpg"
    assert frame_files[-1].name == "000005.jpg"
    for f in frame_files:
        data = f.read_bytes()
        assert data[:2] == b"\xff\xd8", f"{f} 不是 JPEG"

    # frames_index.jsonl 行数 == 5，字段齐全
    # cast 只为让类型检查器知道这是 str（Path.read_text 的桩返回 str，但组合
    # 表达式里 pyright 推成了 Never）；运行时就是普通的 str。
    index_lines = (
        cast(str, (flight_dir / "frames_index.jsonl").read_text(encoding="utf-8"))
        .strip()
        .split("\n")
    )
    assert len(index_lines) == 5
    for line in index_lines:
        rec = json.loads(line)
        assert rec["index"] >= 1
        assert rec["filename"].endswith(".jpg")
        assert "capture_timestamp" in rec
        assert "received_timestamp" in rec
        assert rec["lag"] == pytest.approx(0.15)
        assert rec["extrapolated"] is False
        assert rec["bytes"] > 0

    # events.jsonl 含 recorder_started/state/recorder_stopped
    ev_lines = (
        cast(str, (flight_dir / "events.jsonl").read_text(encoding="utf-8")).strip().split("\n")
    )
    kinds = [json.loads(item)["kind"] for item in ev_lines]
    assert kinds[0] == "recorder_started"
    assert "state" in kinds
    assert kinds[-1] == "recorder_stopped"
    state_record = next(
        json.loads(item) for item in ev_lines if json.loads(item)["kind"] == "state"
    )
    assert state_record["from_state"] == "RECON"
    assert state_record["to_state"] == "HOLD"

    # detections.jsonl 含 1 条 dict + 1 条 dataclass
    det_lines = (flight_dir / "detections.jsonl").read_text(encoding="utf-8").strip().split("\n")
    assert len(det_lines) == 2
    first = json.loads(det_lines[0])
    second = json.loads(det_lines[1])
    assert first["code"] == 42 and first["confidence"] == pytest.approx(0.9)
    assert second["code"] == 7 and second["confidence"] == pytest.approx(0.85)
    assert "as_dict" not in second  # dataclass 已展开为字段

    # drops.jsonl：投放瞬间的状态（位置/速度/姿态/风）与判据的预测都要在
    drop_lines = (flight_dir / "drops.jsonl").read_text(encoding="utf-8").strip().split("\n")
    assert len(drop_lines) == 1
    drop = json.loads(drop_lines[0])
    assert drop["index"] == 1
    assert drop["position_ned"] == pytest.approx([120.0, -35.0, -20.0])
    assert drop["velocity_ned"] == pytest.approx([0.0, 18.0, 0.0])
    assert drop["attitude_deg"] == pytest.approx({"roll": 4.0, "pitch": -3.0, "yaw": 90.0})
    assert drop["wind_ned"] == pytest.approx([2.0, -1.0, 0.0])
    assert drop["delay_s"] == pytest.approx(0.08)
    assert drop["predicted_error_m"] == pytest.approx(0.45)

    # flight.log 含 recorder 启动信息（root logger 被挂上 FileHandler）
    log_text = (flight_dir / "flight.log").read_text(encoding="utf-8")
    assert "FlightRecorder 已启动" in log_text

    # config_snapshot.json 含 video/record/perception 全套
    cfg = json.loads((flight_dir / "config_snapshot.json").read_text(encoding="utf-8"))
    assert "video" in cfg and "record" in cfg and "perception" in cfg
    assert cfg["record"]["telemetry_hz"] == pytest.approx(10.0)

    # telemetry 节流：30Hz 灌入约 150ms，期望 1~10 行（10Hz 兜底；远小于 30）
    tlm_lines = (flight_dir / "telemetry.jsonl").read_text(encoding="utf-8").strip().split("\n")
    assert 1 <= len(tlm_lines) <= 15, f"telemetry 节流异常：{len(tlm_lines)} 行"

    # stats 仍可读
    stats = recorder.stats()
    assert stats is not None
    assert stats.frames_written == 5
    assert stats.raw_reencoded == 0
    assert stats.events >= 3  # started + state + stopped
    assert stats.drops == 1
    assert stats.detections == 2


# ----------------------------------------------------------------------
# 零重编码（核心承诺）
# ----------------------------------------------------------------------
def test_jpeg_storage_writes_zero_reencode(tmp_path: Path, broker: TelemetryBroker) -> None:
    """jpeg 模式下，写入磁盘的 jpeg 字节 == 缓冲里的 jpeg 字节（hash 完全相等）。"""
    buffer = AlignmentBuffer(capacity=4, storage="jpeg", jpeg_quality=80)
    config = Config().validated()
    recorder = FlightRecorder(config, base_dir=tmp_path)
    recorder.start(broker=broker, buffer=buffer)
    try:
        buffer.put(_make_sample(1))
        assert _wait_until(lambda: recorder._frames_written == 1, timeout=3.0)
    finally:
        recorder.stop()

    flight_dir = recorder.flight_dir
    assert flight_dir is not None
    on_disk = (flight_dir / "frames" / "000001.jpg").read_bytes()
    in_buffer = buffer.read_jpeg_bytes(1)
    assert in_buffer is not None
    assert on_disk == in_buffer, "jpeg 写入磁盘字节 != 缓冲字节（发生重编码）"

    stats = recorder.stats()
    assert stats is not None
    assert stats.raw_reencoded == 0


# ----------------------------------------------------------------------
# raw 缓冲：自动重新编码
# ----------------------------------------------------------------------
def test_raw_storage_reencodes_with_cv2(tmp_path: Path, broker: TelemetryBroker) -> None:
    """raw 缓冲下 ``read_jpeg_bytes`` 返回 None → recorder 自行 cv2.imencode 写入磁盘。"""
    buffer = AlignmentBuffer(capacity=4, storage="raw")
    config = Config().validated()
    recorder = FlightRecorder(config, base_dir=tmp_path)
    recorder.start(broker=broker, buffer=buffer)
    try:
        buffer.put(_make_sample(1))
        assert _wait_until(lambda: recorder._frames_written == 1, timeout=3.0)
    finally:
        recorder.stop()

    flight_dir = recorder.flight_dir
    assert flight_dir is not None
    on_disk = (flight_dir / "frames" / "000001.jpg").read_bytes()
    assert on_disk[:2] == b"\xff\xd8"

    stats = recorder.stats()
    assert stats is not None
    assert stats.raw_reencoded == 1


# ----------------------------------------------------------------------
# 跳号可见性：落后到帧被驱逐时必须记账 + 告警
# ----------------------------------------------------------------------
def test_recorder_counts_evicted_frames(
    tmp_path: Path, broker: TelemetryBroker, caplog: pytest.LogCaptureFixture
) -> None:
    """写盘跟不上、帧已被缓冲驱逐：不能假装录全了，要计数并 WARNING。"""
    buffer = AlignmentBuffer(capacity=2, storage="jpeg")
    for index in range(1, 6):  # 容量 2：帧 1~3 会被驱逐
        buffer.put(_make_sample(index))

    recorder = FlightRecorder(Config().validated(), base_dir=tmp_path)
    with caplog.at_level(logging.WARNING, logger="airdrop.record.recorder"):
        recorder.start(broker=broker, buffer=buffer)
        try:
            assert _wait_until(lambda: recorder._frames_written == 2, timeout=3.0)
        finally:
            recorder.stop()

    assert [record for record in caplog.records if "跳号" in record.message], caplog.text
    stats = recorder.stats()
    assert stats is not None
    assert stats.frames_written == 2
    assert stats.frames_skipped == 3, "序号 1~3 被驱逐，必须如实计数"
    flight_dir = recorder.flight_dir
    assert flight_dir is not None
    assert (flight_dir / "frames" / "000004.jpg").is_file()
    assert not (flight_dir / "frames" / "000001.jpg").exists()


# ----------------------------------------------------------------------
# start 幂等 / stop 幂等
# ----------------------------------------------------------------------
def test_start_is_idempotent_when_running(tmp_path: Path, broker: TelemetryBroker) -> None:
    buffer = AlignmentBuffer(capacity=2)
    recorder = FlightRecorder(Config().validated(), base_dir=tmp_path)
    recorder.start(broker=broker, buffer=buffer)
    try:
        first_dir = recorder.flight_dir
        recorder.start(broker=broker, buffer=buffer)  # 第二次不应新建目录
        assert recorder.flight_dir == first_dir
    finally:
        recorder.stop()


def test_stop_is_idempotent(tmp_path: Path, broker: TelemetryBroker) -> None:
    buffer = AlignmentBuffer(capacity=2)
    recorder = FlightRecorder(Config().validated(), base_dir=tmp_path)
    # 未启动就 stop
    recorder.stop()
    recorder.stop()
    # 启动后 stop 多次
    recorder.start(broker=broker, buffer=buffer)
    recorder.stop()
    recorder.stop()
    assert not recorder.is_running


def test_cannot_restart_after_stop(tmp_path: Path, broker: TelemetryBroker) -> None:
    """stop 后再 start 应当明确报错（防止目录错乱）。"""
    buffer = AlignmentBuffer(capacity=2)
    recorder = FlightRecorder(Config().validated(), base_dir=tmp_path)
    recorder.start(broker=broker, buffer=buffer)
    recorder.stop()
    with pytest.raises(RuntimeError, match="已停止于"):
        recorder.start(broker=broker, buffer=buffer)


# ----------------------------------------------------------------------
# 同名时间戳目录不冲突
# ----------------------------------------------------------------------
def test_repeated_timestamp_increments_suffix(tmp_path: Path, broker: TelemetryBroker) -> None:
    """两次 start 用同样的"时间戳"时，第二次要落到 ``-1`` 子目录。"""
    import time as _time

    buffer = AlignmentBuffer(capacity=2)
    config = Config().validated()

    real_strftime = _time.strftime
    fixed_stamp = "某架次"
    counter = {"n": 0}

    def fake_strftime(_fmt):
        counter["n"] += 1
        return fixed_stamp

    _time.strftime = fake_strftime  # type: ignore[assignment]
    try:
        r1 = FlightRecorder(config, base_dir=tmp_path)
        r1.start(broker=broker, buffer=buffer)
        r1.stop()
        d1 = r1.flight_dir

        r2 = FlightRecorder(config, base_dir=tmp_path)
        r2.start(broker=broker, buffer=buffer)
        r2.stop()
        d2 = r2.flight_dir

        assert d1 is not None and d2 is not None
        assert d1.name == "某架次"
        assert d2.name == "某架次-1"
        assert d1.exists() and d2.exists()
    finally:
        _time.strftime = real_strftime  # type: ignore[assignment]


# ----------------------------------------------------------------------
# EventLog / DetectionWriter 单元
# ----------------------------------------------------------------------
def test_event_log_thread_safe(tmp_path: Path) -> None:
    """多线程并发 emit：全部成功 + 顺序无关。"""
    log = EventLog(tmp_path / "events.jsonl")

    def worker(prefix: str, count: int) -> None:
        for i in range(count):
            log.emit(prefix, i=i)

    threads = [threading.Thread(target=worker, args=(f"w{i}", 50)) for i in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    log.close()

    lines = (tmp_path / "events.jsonl").read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 250
    parsed = [json.loads(l) for l in lines]
    kinds = {p["kind"] for p in parsed}
    assert kinds == {"w0", "w1", "w2", "w3", "w4"}
    # 索引值各自连续
    by_kind: dict[str, list[int]] = {}
    for p in parsed:
        by_kind.setdefault(p["kind"], []).append(p["i"])
    for kind, values in by_kind.items():
        assert sorted(values) == list(range(50))


def test_event_log_close_is_idempotent(tmp_path: Path) -> None:
    log = EventLog(tmp_path / "e.jsonl")
    log.close()
    log.close()  # 不抛
    with pytest.raises(RuntimeError):
        log.emit("x")


def test_detection_writer_accepts_dict_mapping_and_dataclass(tmp_path: Path) -> None:
    writer = DetectionWriter(tmp_path / "d.jsonl")
    writer.append({"code": 1, "conf": 0.9})
    writer.append(SimpleNamespace(code=2, conf=0.8))  # Mapping-like
    writer.append(_FakeDetection(code=3, confidence=0.7))
    writer.append(object())  # 不可序列化
    writer.close()

    lines = (tmp_path / "d.jsonl").read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 3  # object() 被拒绝
    parsed = [json.loads(l) for l in lines]
    assert [p["code"] for p in parsed] == [1, 2, 3]


def test_recorder_logger_handler_removed_after_stop(
    tmp_path: Path, broker: TelemetryBroker
) -> None:
    """stop 后移除 handler，并恢复 root logger 原始级别。"""
    buffer = AlignmentBuffer(capacity=2)
    recorder = FlightRecorder(Config().validated(), base_dir=tmp_path)
    root = logging.getLogger()
    original_level = root.level
    root.setLevel(logging.ERROR)

    try:
        before = list(root.handlers)
        recorder.start(broker=broker, buffer=buffer)
        during = list(root.handlers)
        added = [h for h in during if h not in before]
        assert len(added) == 1  # 多挂了 1 个 FileHandler

        recorder.stop()
        after = list(root.handlers)
        assert added[0] not in after, "stop 后未摘除 root logger handler"
        assert root.level == logging.ERROR
    finally:
        root.setLevel(original_level)


# ----------------------------------------------------------------------
# 遥测节流（10Hz vs 30Hz 输入）
# ----------------------------------------------------------------------
@pytest.mark.integration
def test_telemetry_throttle_respects_snapshot_timestamp_boundaries(tmp_path: Path) -> None:
    """真实 broker 与 recorder 按快照时间戳处理阈值前、等于和之后的输入。"""
    broker = TelemetryBroker(history_maxlen=200, history_interval=0.0)
    buffer = AlignmentBuffer(capacity=2)
    record_cfg = RecordConfig(dir=str(tmp_path), telemetry_hz=10.0)
    # 用 replace 派生（frozen dataclass）
    from dataclasses import replace

    config = replace(Config().validated(), record=record_cfg)

    recorder = FlightRecorder(config, base_dir=tmp_path)
    observations: queue.Queue[tuple[float | None, bool]] = queue.Queue()
    wait_next = broker.wait_next_snapshot

    def tracked_wait_next_snapshot(*, predicate=None, timeout=None):
        def track(snapshot):
            accepted = predicate(snapshot) if predicate is not None else True
            observations.put((snapshot.timestamp, accepted))
            return accepted

        return wait_next(predicate=track, timeout=timeout)

    broker.wait_next_snapshot = tracked_wait_next_snapshot
    recorder.start(broker=broker, buffer=buffer)
    try:
        for timestamp in (1000.0, 1000.099, 1000.1, 1000.199, 1000.2, 1000.301):
            broker.publish(TelemetrySnapshot(timestamp=timestamp, north_m=1, east_m=2, down_m=3))
            while True:
                observed, accepted = observations.get(timeout=1.0)
                if observed == timestamp:
                    assert accepted is (timestamp in {1000.0, 1000.1, 1000.2, 1000.301})
                    break
    finally:
        recorder.stop()

    flight_dir = recorder.flight_dir
    assert flight_dir is not None
    lines = (flight_dir / "telemetry.jsonl").read_text(encoding="utf-8").strip().split("\n")
    timestamps = [json.loads(line)["timestamp"] for line in lines]
    assert timestamps == pytest.approx([1000.0, 1000.1, 1000.2, 1000.301])
