"""``FlightRecorder`` 的单元测试。

所有用例都跑在 ``pytest`` 的 ``tmp_path`` 下，离线——不需要飞控、不需要
ffmpeg、不需要硬件。手动往 ``AlignmentBuffer`` 传入帧、往 ``TelemetryBroker``
灌数据，验证五个记录文件 + 投放记录写入磁盘 + 帧零重编码 + 线程安全 + 幂等行为。
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from airdrop import (
    AlignedSample,
    AlignmentBuffer,
    Config,
    DetectionWriter,
    DropRecord,
    EventLog,
    FlightRecorder,
    TelemetryBroker,
    VideoFrame,
)
from airdrop.config import RecordConfig


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
    fixed_stamp = "20260101-120000"
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
        assert d1.name == "20260101-120000"
        assert d2.name == "20260101-120000-1"
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
def test_telemetry_throttle_reduces_high_rate_input(tmp_path: Path) -> None:
    """10Hz 配置下，~200Hz 注入 1 秒应得 5~15 条（远小于 200）。"""
    broker = TelemetryBroker(history_maxlen=200, history_interval=0.0)
    buffer = AlignmentBuffer(capacity=2)
    record_cfg = RecordConfig(dir=str(tmp_path), telemetry_hz=10.0)
    # 用 replace 派生（frozen dataclass）
    from dataclasses import replace

    config = replace(Config().validated(), record=record_cfg)

    recorder = FlightRecorder(config, base_dir=tmp_path)
    recorder.start(broker=broker, buffer=buffer)
    try:
        # ~200Hz 注入 1 秒
        end = time.monotonic() + 1.0
        while time.monotonic() < end:
            _bump_broker(broker, time.time())
        # 给 telemetry writer 一点时间追
        time.sleep(0.3)
    finally:
        recorder.stop()

    flight_dir = recorder.flight_dir
    assert flight_dir is not None
    lines = (flight_dir / "telemetry.jsonl").read_text(encoding="utf-8").strip().split("\n")
    # 10Hz × 1.3s ≈ 13 行；留 5~25 的余量；绝对不能 == 200
    assert 3 <= len(lines) <= 25, f"节流异常: {len(lines)} 行"
