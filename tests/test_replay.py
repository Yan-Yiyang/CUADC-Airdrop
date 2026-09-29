"""回放（P4）离线测试：读飞行目录 → 按原时间轴重放帧与遥测。

这个套件不起 ffmpeg、不连飞控、不用 tmp_path：素材由用例自己按
``FlightRecorder`` 写入磁盘的格式手写（帧索引 + jpeg + telemetry.jsonl），
写到工作区下的临时目录里，测完删除。

为什么不用 ``tmp_path``：部分受限执行环境里 pytest 的 basetemp 目录不可枚举
（``PermissionError: [WinError 5]``），会让用例在 setup 阶段就挂掉；素材格式
本来就是公开约定，手写更可控。
"""

from __future__ import annotations

import json
import logging
import shutil
import time
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from airdrop import (
    AlignmentBuffer,
    AlignmentWriter,
    FlightLog,
    FlightLogError,
    FrameIndexError,
    FrameRecord,
    FrameTelemetryAligner,
    ReplayStats,
    ReplayVideoSource,
    TelemetryBroker,
    TelemetryPacer,
    TelemetrySnapshot,
    load_broker_from_log,
)

# 手写素材的落点：工作区内，避免依赖系统临时目录的权限
WORK_ROOT = Path(__file__).resolve().parents[1] / ".replay-test-tmp"


# ----------------------------------------------------------------------
# 素材构造
# ----------------------------------------------------------------------
def _image(tag: int, width: int = 96, height: int = 64) -> np.ndarray:
    """造一张可区分的画面：左上角亮度随 tag 变化。"""
    image = np.full((height, width, 3), 40, np.uint8)
    image[:16, :16] = (tag * 7) % 256
    return image


def _jpeg(image: np.ndarray) -> bytes:
    ok, buffer = cv2.imencode(".jpg", image)
    assert ok
    return buffer.tobytes()


def _snapshot(timestamp: float, *, index: int = 0, valid: bool = True) -> TelemetrySnapshot:
    if not valid:
        return TelemetrySnapshot(timestamp=timestamp)
    return TelemetrySnapshot(
        timestamp=timestamp,
        north_m=float(index),
        east_m=float(index) * 0.5,
        down_m=-20.0,
        yaw_deg=float(index) * 3.0,
        latitude_deg=30.0 + index * 1e-5,
    )


def _write_flight(
    root: Path,
    *,
    frames: int = 6,
    frame_dt: float = 0.1,
    tlm_dt: float = 0.1,
    tlm_lead: float = 0.2,
    lag: float = 0.2,
    t0: float | None = None,
    telemetry: bool = True,
    blank_first_snapshot: bool = False,
) -> tuple[Path, float, float, list[float]]:
    """按 FlightRecorder 的格式手写一个飞行目录。

    返回 ``(目录, 首帧拍摄时刻, 遥测起点, 遥测时间戳列表)``。
    """
    flight_dir = root / uuid.uuid4().hex[:12]
    (flight_dir / "frames").mkdir(parents=True)
    t0 = time.time() - 10.0 if t0 is None else t0

    captures: list[float] = []
    index_lines = []
    for i in range(1, frames + 1):
        capture = t0 + (i - 1) * frame_dt
        captures.append(capture)
        filename = f"{i:06d}.jpg"
        payload = _jpeg(_image(i))
        (flight_dir / "frames" / filename).write_bytes(payload)
        index_lines.append(
            json.dumps(
                {
                    "index": i,
                    "filename": filename,
                    "capture_timestamp": capture,
                    # received = capture + lag，与 VideoFrame.timestamp 同义
                    "received_timestamp": capture + lag,
                    "lag": lag,
                    "extrapolated": False,
                    "offset": 0.0,
                    "bytes": len(payload),
                }
            )
        )
    (flight_dir / "frames_index.jsonl").write_text("\n".join(index_lines) + "\n", encoding="utf-8")

    if telemetry:
        tlm_start = t0 - tlm_lead
        span = (frames - 1) * frame_dt + tlm_lead + 0.5
        count = int(span / tlm_dt) + 1
        stamps = [tlm_start + k * tlm_dt for k in range(count)]
        lines = []
        if blank_first_snapshot:
            lines.append(json.dumps(_snapshot(tlm_start - tlm_dt, valid=False).as_dict()))
        lines += [
            json.dumps(_snapshot(ts, index=k, valid=True).as_dict()) for k, ts in enumerate(stamps)
        ]
        (flight_dir / "telemetry.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    else:
        tlm_start, stamps = t0, []
    return flight_dir, captures[0], tlm_start, stamps


@pytest.fixture
def workdir() -> Iterator[Path]:
    """工作区内的临时目录；用例结束整棵删掉。"""
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    path = WORK_ROOT / uuid.uuid4().hex[:8]
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _play_to_completion(source: ReplayVideoSource, timeout: float = 20.0) -> None:
    """启动并等播放线程自然结束（不调 stop()，否则 state 会变成 stopped）。"""
    source.start()
    thread = source._thread
    assert thread is not None
    thread.join(timeout)
    assert not thread.is_alive(), "回放线程未在超时内结束"


# ----------------------------------------------------------------------
# FlightLog：素材解析
# ----------------------------------------------------------------------
def test_flight_log_reads_frame_index(workdir: Path) -> None:
    flight_dir, _, _, _ = _write_flight(workdir, frames=5)
    log = FlightLog.open(flight_dir)

    assert len(log.frames) == 5
    assert log.frames_dir == flight_dir / "frames"
    assert log.has_telemetry
    first = log.frames[0]
    assert isinstance(first, FrameRecord)
    assert first.index == 1
    assert first.filename == "000001.jpg"
    assert first.lag == pytest.approx(0.2)
    # 还原出来的"收到时刻" = 拍摄时刻 + lag，于是 VideoFrame.capture_timestamp 回到原值
    assert first.timestamp == pytest.approx(first.capture_timestamp + first.lag)
    span = log.time_span()
    assert span is not None
    assert span[1] - span[0] == pytest.approx(4 * 0.1)


def test_flight_log_rejects_missing_dir(workdir: Path) -> None:
    with pytest.raises(FlightLogError, match="不存在"):
        FlightLog.open(workdir / "nope")


def test_flight_log_rejects_missing_index(workdir: Path) -> None:
    empty = workdir / "empty"
    empty.mkdir()
    with pytest.raises(FlightLogError, match="缺少帧索引"):
        FlightLog.open(empty)


def test_flight_log_rejects_empty_index(workdir: Path) -> None:
    empty = workdir / "empty"
    empty.mkdir()
    (empty / "frames_index.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(FrameIndexError, match="为空"):
        FlightLog.open(empty)


def test_flight_log_rejects_bad_json_line(workdir: Path) -> None:
    flight_dir, *_ = _write_flight(workdir, frames=3)
    (flight_dir / "frames_index.jsonl").write_text("{oops\n", encoding="utf-8")
    with pytest.raises(FrameIndexError, match="不是合法 JSON"):
        FlightLog.open(flight_dir)


def test_flight_log_rejects_missing_field(workdir: Path) -> None:
    flight_dir, *_ = _write_flight(workdir, frames=3)
    (flight_dir / "frames_index.jsonl").write_text(
        json.dumps({"index": 1, "filename": "000001.jpg"}) + "\n", encoding="utf-8"
    )
    with pytest.raises(FrameIndexError, match="字段缺失"):
        FlightLog.open(flight_dir)


def test_flight_log_rejects_time_going_backwards(workdir: Path) -> None:
    """时间戳倒流必须报错，而不是让对齐在内插时拿到乱序历史。"""
    flight_dir, *_ = _write_flight(workdir, frames=3)
    lines = (flight_dir / "frames_index.jsonl").read_text(encoding="utf-8").strip().split("\n")
    records = [json.loads(line) for line in lines]
    records[1]["capture_timestamp"] = records[0]["capture_timestamp"] - 1.0
    (flight_dir / "frames_index.jsonl").write_text(
        "\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8"
    )
    with pytest.raises(FrameIndexError, match="时间戳倒流"):
        FlightLog.open(flight_dir)


def test_iter_telemetry_streams_snapshots(workdir: Path) -> None:
    flight_dir, _, tlm_start, stamps = _write_flight(workdir, frames=4)
    log = FlightLog.open(flight_dir)
    snapshots = list(log.iter_telemetry())

    assert len(snapshots) == len(stamps)
    assert snapshots[0].timestamp == pytest.approx(tlm_start)
    assert snapshots[0].is_valid()
    assert snapshots[1].north_m == pytest.approx(1.0)


def test_iter_telemetry_skips_broken_lines(workdir: Path) -> None:
    """坏行只记 warning 跳过：一段素材不该因为一行损坏而整份作废。"""
    flight_dir, *_ = _write_flight(workdir, frames=3, telemetry=True)
    path = flight_dir / "telemetry.jsonl"
    lines = path.read_text(encoding="utf-8").strip().split("\n")
    lines.insert(1, "{坏行")
    lines.insert(2, "[1, 2, 3]")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    snapshots = list(FlightLog.open(flight_dir).iter_telemetry())
    # 12 行里有两行被拒：一行不是合法 JSON，一行不是对象
    assert len(lines) == 12
    assert len(snapshots) == len(lines) - 2
    assert all(s.timestamp is not None for s in snapshots)


def test_iter_telemetry_requires_file(workdir: Path) -> None:
    flight_dir, *_ = _write_flight(workdir, frames=2, telemetry=False)
    with pytest.raises(FlightLogError, match="缺少遥测日志"):
        list(FlightLog.open(flight_dir).iter_telemetry())


# ----------------------------------------------------------------------
# TelemetryPacer：按帧推进
# ----------------------------------------------------------------------
def test_pacer_publishes_only_up_to_limit(workdir: Path) -> None:
    flight_dir, *_ = _write_flight(workdir, frames=6, tlm_dt=0.1)
    log = FlightLog.open(flight_dir)
    snapshots = list(log.iter_telemetry())

    broker = TelemetryBroker(history_maxlen=1000, history_interval=0.0)
    pacer = TelemetryPacer(broker, snapshots, ahead_s=0.0)
    assert pacer.cover_to() == float("-inf")

    target = snapshots[2].timestamp
    published = pacer.publish_until(target)
    assert published == 3
    assert pacer.cover_to() == pytest.approx(target)
    span = broker.history_span()
    assert span is not None
    assert span[1] == pytest.approx(target)
    # 没到点的快照不能提前进 broker（否则时间轴就"剧透"了）
    assert span[0] == pytest.approx(snapshots[0].timestamp)


def test_pacer_ahead_keeps_right_endpoint(workdir: Path) -> None:
    """多推进 ahead_s：内插需要右端点，不能只推到帧时刻本身。"""
    flight_dir, *_ = _write_flight(workdir, frames=6, tlm_dt=0.1)
    snapshots = list(FlightLog.open(flight_dir).iter_telemetry())
    broker = TelemetryBroker(history_maxlen=1000, history_interval=0.0)
    pacer = TelemetryPacer(broker, snapshots, ahead_s=0.5)

    frame_ts = snapshots[1].timestamp
    pacer.publish_until(frame_ts)
    span = broker.history_span()
    assert span is not None and span[1] > frame_ts


def test_pacer_is_monotonic_and_idempotent(workdir: Path) -> None:
    flight_dir, *_ = _write_flight(workdir, frames=6)
    snapshots = list(FlightLog.open(flight_dir).iter_telemetry())
    broker = TelemetryBroker(history_maxlen=1000, history_interval=0.0)
    pacer = TelemetryPacer(broker, snapshots, ahead_s=0.0)

    first = pacer.publish_until(snapshots[3].timestamp)
    again = pacer.publish_until(snapshots[3].timestamp)
    assert first == 4 and again == 0  # 同一时刻重复调用不重复灌
    assert pacer.published == 4


def test_pacer_sorts_out_of_order_snapshots() -> None:
    broker = TelemetryBroker(history_maxlen=100, history_interval=0.0)
    snapshots = [_snapshot(100.0 + i) for i in (2, 0, 1)]
    pacer = TelemetryPacer(broker, snapshots, ahead_s=0.0)
    pacer.publish_until(101.0)
    span = broker.history_span()
    assert span is not None
    assert span[0] == pytest.approx(100.0)
    assert span[1] == pytest.approx(101.0)


def test_pacer_finish_covers_material_tail() -> None:
    """素材末帧晚于最后一条遥测时，finish() 把覆盖补到末帧（否则末帧会空等 1s）。"""
    broker = TelemetryBroker(history_maxlen=100, history_interval=0.0)
    snapshots = [_snapshot(100.0, index=0), _snapshot(100.5, index=1)]
    pacer = TelemetryPacer(broker, snapshots, ahead_s=0.0)

    pacer.publish_until(100.5)
    assert pacer.finish(100.55) == 1
    assert pacer.tail_extended == 1
    span = broker.history_span()
    assert span is not None and span[1] >= 100.55
    # 延伸得到的是最后一条的状态（不是凭空外推出的别的值）
    query = broker.get_snapshot_at(100.55)
    assert query is not None and query.north_m == pytest.approx(1.0)
    # 末帧落在覆盖区间内 → 对齐器不需要等待
    assert span[0] <= 100.55 <= span[1]


def test_pacer_finish_is_noop_when_log_already_covers() -> None:
    broker = TelemetryBroker(history_maxlen=100, history_interval=0.0)
    pacer = TelemetryPacer(broker, [_snapshot(100.0), _snapshot(101.0)], ahead_s=0.0)
    assert pacer.finish(100.5) == 0
    assert pacer.tail_extended == 0


def test_pacer_finish_flushes_pending_before_extending() -> None:
    """finish() 先把待灌的真实快照灌完再延伸——否则历史会被写乱序。

    历史必须按时间递增（内插靠二分），一旦延伸点插到真实点前面，
    ``history_span()`` 会把最后那条延伸点当成"最早"，整个查询就崩了。
    """
    broker = TelemetryBroker(history_maxlen=100, history_interval=0.0)
    snapshots = [_snapshot(100.0 + i * 0.1, index=i) for i in range(5)]
    pacer = TelemetryPacer(broker, snapshots, ahead_s=0.0)

    added = pacer.finish(105.0)  # 还没 publish_until 过就直接 finish
    assert added > 0
    assert pacer.tail_extended == added, "返回值与 tail_extended 应当一致"
    history = [s.timestamp for s in broker._history]
    assert history == sorted(history), f"历史被写乱序: {history}"
    span = broker.history_span()
    assert span is not None
    assert span[0] == pytest.approx(100.0), "最早一条被延伸点挤掉了"
    assert span[1] >= 105.0


def test_pacer_finish_warns_on_suspicious_gap(caplog) -> None:
    """末尾遥测缺得太多（日志被截断/链路掉了）要喊出来，不能默默延伸。

    告警量按**最后一条真实快照**算：逐帧调用 ``finish`` 时如果按上一次的
    延伸点算，一整段缺失会被摊成每帧几十毫秒，warning 永远不触发。
    """
    broker = TelemetryBroker(history_maxlen=100, history_interval=0.0)
    pacer = TelemetryPacer(
        broker, [_snapshot(100.0), _snapshot(100.5)], ahead_s=0.0, tail_warn_s=1.0
    )
    with caplog.at_level(logging.WARNING, logger="airdrop.record.replay"):
        added = pacer.finish(102.0)  # 最后一条真实遥测在 100.5 → 缺 1.5s > tail_warn_s
    assert added >= 1
    assert pacer.tail_extended == added
    span = broker.history_span()
    assert span is not None and span[1] >= 102.0
    assert [record for record in caplog.records if "末尾" in record.message], caplog.text


def test_pacer_finish_warns_once_for_incrementally_extended_tail(caplog) -> None:
    """逐帧补尾部（_play_files 的真实调用方式）：缺 10s 也只告警一次。"""
    broker = TelemetryBroker(history_maxlen=500, history_interval=0.0)
    pacer = TelemetryPacer(
        broker, [_snapshot(100.0), _snapshot(100.5)], ahead_s=0.0, tail_warn_s=5.0
    )
    with caplog.at_level(logging.WARNING, logger="airdrop.record.replay"):
        for step in range(1, 41):  # 0.25s 一帧，补到 110.5（缺 10s）
            target = 100.5 + step * 0.25
            if target > pacer.cover_to():
                pacer.finish(target)
            pacer.publish_until(target)
    warnings = [record for record in caplog.records if "末尾" in record.message]
    assert len(warnings) == 1, caplog.text
    assert pacer.tail_extended > 1, "逐帧延伸会积累多条，不是 0/1 那点量"
    span = broker.history_span()
    assert span is not None and span[1] >= 110.5


def test_pacer_finish_without_snapshots_is_noop() -> None:
    broker = TelemetryBroker(history_maxlen=100, history_interval=0.0)
    pacer = TelemetryPacer(broker, [], ahead_s=0.0)
    assert pacer.finish(100.0) == 0
    assert pacer.cover_to() == float("-inf")
    assert pacer.finished_to == float("-inf")


def test_pacer_warns_on_empty_log() -> None:
    broker = TelemetryBroker(history_maxlen=100, history_interval=0.0)
    pacer = TelemetryPacer(broker, [], ahead_s=0.0)
    assert pacer.publish_until(1.0) == 0
    assert broker.history_span() is None


def test_pacer_rejects_negative_ahead() -> None:
    broker = TelemetryBroker()
    with pytest.raises(ValueError, match="ahead_s"):
        TelemetryPacer(broker, [], ahead_s=-1.0)


def test_load_broker_from_log(workdir: Path) -> None:
    flight_dir, *_ = _write_flight(workdir, frames=4)
    broker, pacer = load_broker_from_log(flight_dir)

    # 只是"就绪"，还没灌进去（灌入由回放源按帧驱动）
    assert broker.history_span() is None
    assert pacer.total > 0
    assert pacer.published == 0


def test_load_broker_from_log_requires_telemetry(workdir: Path) -> None:
    flight_dir, *_ = _write_flight(workdir, frames=2, telemetry=False)
    with pytest.raises(FlightLogError, match="缺少遥测日志"):
        load_broker_from_log(flight_dir)


# ----------------------------------------------------------------------
# ReplayVideoSource：与实飞源同一套接口
# ----------------------------------------------------------------------
def test_replay_delivers_every_frame_with_original_timestamps(workdir: Path) -> None:
    flight_dir, first_capture, _, _ = _write_flight(workdir, frames=6, frame_dt=0.1, lag=0.2)
    source = ReplayVideoSource(flight_dir, speed=0.0)

    seen: list[tuple[int, float, float]] = []
    images: list[np.ndarray] = []
    source.add_sink(
        lambda f: (
            seen.append((f.index, f.capture_timestamp, f.lag)),
            images.append(f.image.copy()),
        )
    )
    _play_to_completion(source)

    stats = source.stats
    assert stats.frames == 6
    assert stats.total == 6
    assert stats.skipped == 0
    assert stats.dropped == 0  # 回放不丢帧
    assert stats.state == "finished"
    assert [idx for idx, _, _ in seen] == [1, 2, 3, 4, 5, 6]
    # 拍摄时刻逐位等于录制时的值（回放的全部意义所在）
    for i, (_, capture, lag) in enumerate(seen):
        assert capture == pytest.approx(first_capture + i * 0.1, abs=1e-12)
        assert lag == pytest.approx(0.2)
    # 画面确实是"逐帧不同"的（没有复用同一块缓冲）
    assert not np.array_equal(images[0], images[-1])


def test_replay_missing_frames_are_counted_but_not_log_spammed(
    workdir: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """裁剪过的素材（索引仍列着已删帧号）：非严格模式跳过并计数，但**不逐条刷日志**。

    实测架次 某架次 的裁剪目录曾刷 3568 行 ERROR；现在只逐条打前
    ``SKIP_LOG_LIMIT`` 条，收尾给一条"共跳过 N 帧"的汇总。
    """
    from airdrop.record.replay import SKIP_LOG_LIMIT

    flight_dir, *_ = _write_flight(workdir, frames=8)
    for name in ("000003.jpg", "000004.jpg"):
        (flight_dir / "frames" / name).unlink()

    source = ReplayVideoSource(flight_dir, speed=0.0, strict=False)
    seen: list[int] = []
    source.add_sink(lambda frame: seen.append(frame.index))
    with caplog.at_level(logging.WARNING, logger="airdrop.record.replay"):
        _play_to_completion(source)

    stats = source.stats
    assert stats.frames == 6 and stats.skipped == 2
    assert stats.state == "finished"
    assert seen == [1, 2, 5, 6, 7, 8]
    per_frame = [r for r in caplog.records if r.getMessage().startswith("回放跳过帧")]
    assert len(per_frame) <= SKIP_LOG_LIMIT, "逐条日志要有上限，不能一帧一行"
    assert any("回放跳过 2 帧" in r.getMessage() for r in caplog.records), "收尾要有汇总"


def test_replay_frames_have_consistent_capture_timestamp(workdir: Path) -> None:
    """``timestamp - lag`` 必须等于索引里的拍摄时刻：对齐器就吃这个值。"""
    flight_dir, *_ = _write_flight(workdir, frames=3, lag=0.2)
    log = FlightLog.open(flight_dir)
    source = ReplayVideoSource(flight_dir, speed=0.0)
    _play_to_completion(source)

    assert log.frames[0].timestamp - log.frames[0].lag == pytest.approx(
        log.frames[0].capture_timestamp
    )


def test_replay_read_and_latest(workdir: Path) -> None:
    flight_dir, *_ = _write_flight(workdir, frames=4)
    source = ReplayVideoSource(flight_dir, speed=0.0)
    source.start()
    try:
        first = source.read(timeout=10.0)
        assert first is not None and first.index == 1
        second = source.read(timeout=10.0)
        assert second is not None and second.index == 2
        assert source.latest() is not None
    finally:
        source.stop()
    assert source.stats.state == "stopped"


def test_replay_read_drains_every_frame_then_returns_none(workdir: Path) -> None:
    """read() 逐帧读完（含最后一帧）后才返回 None，且不丢帧。"""
    flight_dir, *_ = _write_flight(workdir, frames=4)
    source = ReplayVideoSource(flight_dir, speed=0.0)
    source.start()
    try:
        indices = []
        while True:
            frame = source.read(timeout=15.0)
            if frame is None:
                break
            indices.append(frame.index)
        assert indices == [1, 2, 3, 4], "read() 路径丢了帧"
        assert source.stats.state == "finished"
        assert source.stats.dropped == 0
        assert source.read(timeout=0.5) is None
    finally:
        source.stop()


def test_replay_iter_frames(workdir: Path) -> None:
    flight_dir, *_ = _write_flight(workdir, frames=5)
    source = ReplayVideoSource(flight_dir, speed=0.0)
    source.start()
    try:
        indices = [frame.index for frame in source.iter_frames(timeout=2.0)]
    finally:
        source.stop()
    assert indices == [1, 2, 3, 4, 5]


def test_replay_wait_ready_and_context_manager(workdir: Path) -> None:
    flight_dir, *_ = _write_flight(workdir, frames=3)
    with ReplayVideoSource(flight_dir, speed=0.0) as source:
        assert source.wait_ready(timeout=10.0)
        assert source.running
    assert not source.running


def test_replay_config_mirrors_material(workdir: Path) -> None:
    flight_dir, *_ = _write_flight(workdir, frames=3, lag=0.2)
    source = ReplayVideoSource(flight_dir, speed=0.0)
    config = source.config
    assert config.width == 96 and config.height == 64
    assert config.telemetry_lag == pytest.approx(0.2)


def test_replay_speed_one_paces_in_real_time(workdir: Path) -> None:
    """原速回放的墙钟耗时 ≈ 素材自身的时间跨度。"""
    flight_dir, *_ = _write_flight(workdir, frames=8, frame_dt=0.1)
    source = ReplayVideoSource(flight_dir, speed=1.0)
    started = time.monotonic()
    _play_to_completion(source, timeout=10.0)
    elapsed = time.monotonic() - started
    assert elapsed >= 0.6, f"原速回放播得太快: {elapsed:.2f}s"
    assert elapsed < 5.0, f"原速回放播得太慢: {elapsed:.2f}s"


def test_replay_speed_zero_is_immediate(workdir: Path) -> None:
    """全速回放不受素材跨度约束（离线迭代要的就是"能多快就多快"）。"""
    flight_dir, *_ = _write_flight(workdir, frames=8, frame_dt=0.5)
    source = ReplayVideoSource(flight_dir, speed=0.0)
    started = time.monotonic()
    _play_to_completion(source)
    assert time.monotonic() - started < 2.0


def test_replay_read_skips_like_live_source_but_sink_does_not(workdir: Path) -> None:
    """read() 是实时路径（允许跳帧并计入 dropped）；sink 路径一帧不落。

    这是与实飞源对齐的语义，也是"要留存就走 sink"这条铁律在回放里的体现。
    """
    flight_dir, *_ = _write_flight(workdir, frames=5)
    source = ReplayVideoSource(flight_dir, speed=0.0)
    sunk: list[int] = []
    source.add_sink(lambda f: sunk.append(f.index))
    source.start()
    try:
        first = source.read(timeout=15.0)
        assert first is not None and first.index == 1
        time.sleep(0.2)  # 慢消费：生产早已播完
        later = source.read(timeout=15.0)
        assert later is not None and later.index == 5
        # 中间 2/3/4 是"实时路径没看上"的帧，如实计数（sink 侧一帧不少）
        assert source.stats.dropped == 3
    finally:
        source.stop()
    assert sunk == [1, 2, 3, 4, 5], "sink 路径丢了帧"


def test_replay_stop_is_idempotent_and_interrupts(workdir: Path) -> None:
    """stop() 幂等，且能把正在原速播放的回放立刻打断。"""
    flight_dir, *_ = _write_flight(workdir, frames=200, frame_dt=0.05)
    source = ReplayVideoSource(flight_dir, speed=1.0)
    source.start()
    assert source.wait_ready(timeout=10.0)
    time.sleep(0.2)
    source.stop()
    source.stop()  # 幂等
    assert not source.running
    assert source.stats.frames < 200, "stop 没有中断播放"
    assert source.stats.state == "stopped"


def test_replay_repeat_loops_material(workdir: Path) -> None:
    flight_dir, *_ = _write_flight(workdir, frames=3)
    source = ReplayVideoSource(flight_dir, speed=0.0, repeat=True)
    source.start()
    try:
        deadline = time.monotonic() + 10.0
        while source.stats.frames < 7 and time.monotonic() < deadline:
            time.sleep(0.02)
        assert source.stats.frames >= 7, "repeat 没有循环播放"
    finally:
        source.stop()


def test_replay_skips_missing_frame_file(workdir: Path) -> None:
    """非 strict：缺文件记账跳过（一次飞行里坏一帧不该让整份素材不可用）。"""
    flight_dir, *_ = _write_flight(workdir, frames=4)
    (flight_dir / "frames" / "000002.jpg").unlink()

    source = ReplayVideoSource(flight_dir, speed=0.0, strict=False)
    seen: list[int] = []
    source.add_sink(lambda f: seen.append(f.index))
    _play_to_completion(source)

    assert seen == [1, 3, 4]
    assert source.stats.skipped == 1
    assert source.stats.frames == 3
    assert source.stats.last_error is None
    assert source.stats.state == "finished"


def test_replay_strict_raises_on_missing_frame(workdir: Path) -> None:
    """strict 模式：素材损坏直接抛错（做正式评估时宁可失败也不静默少帧）。"""
    flight_dir, *_ = _write_flight(workdir, frames=4)
    (flight_dir / "frames" / "000002.jpg").unlink()

    source = ReplayVideoSource(flight_dir, speed=0.0, strict=True)
    source.start()
    thread = source._thread
    assert thread is not None
    thread.join(10.0)
    try:
        assert not thread.is_alive()
        assert source.stats.state == "error"
        assert "000002.jpg" in (source.stats.last_error or "")
    finally:
        source.stop()


def test_replay_rejects_bad_params(workdir: Path) -> None:
    flight_dir, *_ = _write_flight(workdir, frames=2)
    with pytest.raises(ValueError, match="speed"):
        ReplayVideoSource(flight_dir, speed=-1.0)
    with pytest.raises(ValueError, match="tail_s"):
        ReplayVideoSource(flight_dir, tail_s=-1.0)


def test_replay_tail_keeps_source_alive(workdir: Path) -> None:
    """tail_s：播完后多留一会儿，给慢消费者追上的时间。"""
    flight_dir, *_ = _write_flight(workdir, frames=2)
    source = ReplayVideoSource(flight_dir, speed=0.0, tail_s=0.3)
    started = time.monotonic()
    _play_to_completion(source)
    assert time.monotonic() - started >= 0.25


def test_replay_sink_exception_does_not_stop_playback(workdir: Path) -> None:
    """非 strict：sink 抛异常只记日志，其余 sink 与后续帧不受影响。"""
    flight_dir, *_ = _write_flight(workdir, frames=4)
    source = ReplayVideoSource(flight_dir, speed=0.0, strict=False)
    good: list[int] = []

    def bad_sink(frame) -> None:
        raise RuntimeError("sink 抛出异常了")

    source.add_sink(bad_sink)
    source.add_sink(lambda f: good.append(f.index))
    _play_to_completion(source)
    assert good == [1, 2, 3, 4]
    assert source.stats.frames == 4
    assert source.stats.state == "finished"


def test_replay_strict_sink_exception_fails_fast(workdir: Path) -> None:
    """strict：sink 抛异常直接让回放进入 error，不静默吞掉。"""
    flight_dir, *_ = _write_flight(workdir, frames=4)
    source = ReplayVideoSource(flight_dir, speed=0.0, strict=True)
    source.add_sink(lambda frame: (_ for _ in ()).throw(RuntimeError("sink 抛出异常了")))
    _play_to_completion(source)
    assert source.stats.state == "error"
    assert "sink" in (source.stats.last_error or "")


def test_replay_remove_sink(workdir: Path) -> None:
    flight_dir, *_ = _write_flight(workdir, frames=3)
    source = ReplayVideoSource(flight_dir, speed=0.0)
    got: list[int] = []

    def sink(frame: Any) -> None:
        got.append(frame.index)

    source.add_sink(sink)
    assert sink in source.sinks
    source.remove_sink(sink)
    assert source.sinks == ()
    _play_to_completion(source)
    assert got == []


def test_replay_stats_snapshot_is_a_copy(workdir: Path) -> None:
    flight_dir, *_ = _write_flight(workdir, frames=2)
    source = ReplayVideoSource(flight_dir, speed=0.0)
    stats = source.stats
    assert isinstance(stats, ReplayStats)
    stats.frames = 999
    assert source.stats.frames == 0


# ----------------------------------------------------------------------
# 端到端：回放驱动"对齐 → 缓冲"，结果与日志逐位一致
# ----------------------------------------------------------------------
def test_replay_feeds_alignment_pipeline(workdir: Path) -> None:
    """回放 + 遥测回填 → FrameTelemetryAligner → AlignmentBuffer。

    这是 P4 的验收口径：回放时每一帧都拿到"拍摄时刻"的遥测，且与直接
    从日志查询的结果完全一致（说明回放没有改变时间轴），一帧不落。
    """
    flight_dir, first_capture, _, _ = _write_flight(
        workdir, frames=6, frame_dt=0.1, tlm_dt=0.05, tlm_lead=0.2, lag=0.2
    )
    broker, pacer = load_broker_from_log(flight_dir)

    # 参考 broker：同一份日志一次性灌入，作为"正确时间轴"的对照
    reference = TelemetryBroker(history_maxlen=1000, history_interval=0.0)
    for snapshot in FlightLog.open(flight_dir).iter_telemetry():
        reference.publish(snapshot)

    buffer = AlignmentBuffer(capacity=16, storage="jpeg")
    aligner = FrameTelemetryAligner(broker, max_wait=1.0)
    source = ReplayVideoSource(flight_dir, speed=0.0, telemetry=pacer)
    source.add_sink(AlignmentWriter(buffer, aligner))

    _play_to_completion(source)

    assert source.stats.frames == 6
    assert source.stats.skipped == 0
    assert buffer.stats.put == 6, "回放的帧没有全部进缓冲"
    assert aligner.stats.aligned == 6
    # 回放时遥测一定已经追上了（pacer 按帧推进）→ 不该有任何等待/丢弃/外推
    assert aligner.stats.wait_timeouts == 0
    assert aligner.stats.no_telemetry == 0
    assert aligner.stats.extrapolation_drops == 0

    for record in buffer.iter_between():
        assert record.index in range(1, 7)
        assert record.lag == pytest.approx(0.2)
        assert record.capture_timestamp == pytest.approx(
            first_capture + (record.index - 1) * 0.1, abs=1e-12
        )
        expect = reference.get_snapshot_at(record.capture_timestamp)
        assert record.snapshot.north_m == pytest.approx(expect.north_m)
        assert record.snapshot.yaw_deg == pytest.approx(expect.yaw_deg)
        assert record.extrapolated is False

    # 遥测回填是"逐帧推进"的：全部推完
    assert pacer.published == pacer.total + pacer.tail_extended


def test_replay_covers_material_tail_without_aligner_timeout(workdir: Path) -> None:
    """素材"遥测先结束、视频后结束"时，末帧也必须内插命中（不触发 1s 等待）。

    录制器按固定频率写遥测，日志最后一条通常早于末帧几十毫秒。没有 pacer.finish
    的补齐，末帧会去等 1.0s 上限然后被记成"链路停顿"丢弃——回放里那是假警报。
    """
    flight_dir, *_ = _write_flight(
        workdir, frames=6, frame_dt=0.1, tlm_dt=0.1, tlm_lead=0.2, lag=0.2
    )
    # 砍掉日志尾部，让末帧明显晚于最后一条遥测（真实素材里通常是几十毫秒，
    # 这里放大到几百毫秒，便于确定性断言）
    path = flight_dir / "telemetry.jsonl"
    lines = path.read_text(encoding="utf-8").strip().split("\n")
    path.write_text("\n".join(lines[:-7]) + "\n", encoding="utf-8")
    assert max(s.timestamp for s in FlightLog.open(flight_dir).iter_telemetry()) < (
        FlightLog.open(flight_dir).frames[-1].capture_timestamp
    )

    broker, pacer = load_broker_from_log(flight_dir)
    buffer = AlignmentBuffer(capacity=16, storage="jpeg")
    aligner = FrameTelemetryAligner(broker, max_wait=1.0)
    source = ReplayVideoSource(flight_dir, speed=0.0, telemetry=pacer)
    source.add_sink(AlignmentWriter(buffer, aligner))

    _play_to_completion(source)

    assert source.stats.frames == 6
    assert buffer.stats.put == 6, "末帧因等遥测被丢掉了"
    assert aligner.stats.wait_timeouts == 0
    assert pacer.tail_extended > 0
    assert source.stats.tail_extended == pacer.tail_extended


def test_replay_without_pacer_reports_no_telemetry(workdir: Path) -> None:
    """不给 pacer 就只是放帧：对齐器如实报"没有遥测"，而不是静默造假。"""
    flight_dir, *_ = _write_flight(workdir, frames=3)
    broker = TelemetryBroker(history_maxlen=100, history_interval=0.0)
    buffer = AlignmentBuffer(capacity=8, storage="jpeg")
    aligner = FrameTelemetryAligner(broker)

    source = ReplayVideoSource(flight_dir, speed=0.0)
    source.add_sink(AlignmentWriter(buffer, aligner))
    _play_to_completion(source)

    assert source.stats.frames == 3
    assert aligner.stats.no_telemetry == 3
    assert buffer.stats.put == 0


def test_replay_from_buffer_mode(workdir: Path) -> None:
    """跟随缓冲模式：时间戳取自缓冲记录，逐条消费不跳帧。"""
    from airdrop import AlignedSample, VideoFrame

    flight_dir, *_ = _write_flight(workdir, frames=4, lag=0.2)
    log = FlightLog.open(flight_dir)

    buffer = AlignmentBuffer(capacity=8, storage="jpeg")
    for record in log.frames:
        buffer.put(
            AlignedSample(
                frame=VideoFrame(
                    index=record.index,
                    image=_image(record.index),
                    timestamp=record.timestamp,
                    lag=record.lag,
                ),
                snapshot=_snapshot(record.capture_timestamp, index=record.index),
                timestamp=record.capture_timestamp,
                lag=record.lag,
                mode="interpolate",
            )
        )

    source = ReplayVideoSource(flight_dir, speed=0.0, buffer=buffer)
    seen: list[tuple[int, float]] = []
    source.add_sink(lambda f: seen.append((f.index, f.capture_timestamp)))
    _play_to_completion(source)

    assert [idx for idx, _ in seen] == [1, 2, 3, 4]
    for idx, capture in seen:
        assert capture == pytest.approx(log.frames[idx - 1].capture_timestamp)


def test_replay_finished_state_distinguishable_from_stopped(workdir: Path) -> None:
    """自然播完是 "finished"，主动 stop 是 "stopped"——上层据此区分两种结束。"""
    flight_dir, *_ = _write_flight(workdir, frames=2)
    source = ReplayVideoSource(flight_dir, speed=0.0)
    _play_to_completion(source)
    assert source.stats.state == "finished"
