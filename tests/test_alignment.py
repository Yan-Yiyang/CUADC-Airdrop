"""帧-遥测时间对齐的单元测试。

不依赖飞控与视频硬件：直接在 broker 的历史里注入确定性时间戳，再用手工构造
的 :class:`~airdrop.video.VideoFrame` 验证整条链路——

    收到帧的时间 - 链路固定延时（150ms） → 按该时刻取遥测快照

覆盖内插、nearest、外推标记、``max_extrapolation`` 保护、延时优先级
（帧自带 / 显式覆盖 / 0 不补偿）、参数校验，以及 P2 引入的等待语义：
``max_wait`` 让遥测追上拍摄时刻再内插，超时则按 ``on_timeout="drop"`` 丢弃。
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from airdrop import (
    DEFAULT_TELEMETRY_LAG,
    FrameTelemetryAligner,
    TelemetryBroker,
    TelemetrySnapshot,
    VideoConfig,
    VideoFrame,
)


class Fake:
    """最小 MAVSDK 对象替身。

    字段是随手挂上去的（``p.latitude_deg = ...``），所以显式声明成"任意属性"，
    类型检查器才不会把未声明的赋值判成错误。
    """

    if TYPE_CHECKING:

        def __getattr__(self, name: str) -> Any: ...
        def __setattr__(self, name: str, value: Any) -> None: ...


def make_position(lat, lon=8.0, abs_alt=500.0, rel_alt=10.0):
    p = Fake()
    p.latitude_deg = lat
    p.longitude_deg = lon
    p.absolute_altitude_m = abs_alt
    p.relative_altitude_m = rel_alt
    return p


def make_frame(timestamp: float, lag: float = DEFAULT_TELEMETRY_LAG) -> VideoFrame:
    """构造一帧"刚收到"的画面（图像内容无关紧要）。"""
    return VideoFrame(
        index=1,
        image=np.zeros((4, 4, 3), dtype=np.uint8),
        timestamp=timestamp,
        lag=lag,
    )


def make_broker() -> TelemetryBroker:
    """history_interval=0：每次更新都入库，方便确定性地构造历史。"""
    return TelemetryBroker(history_maxlen=200, history_interval=0.0)


def fill_history(
    broker: TelemetryBroker,
    base: float,
    count: int,
    step: float,
    lat0: float = 47.0,
    dlat: float = 0.001,
) -> float:
    """注入确定性的历史快照，返回最新一条的时间戳。

    直接写 ``_history`` 而不走 ``_update``：``get_snapshot_at`` 只读历史，
    这样时间轴完全可控，不需要靠 sleep 去凑时间。
    """
    for i in range(count):
        broker._history.append(
            TelemetrySnapshot(
                timestamp=base + i * step,
                latitude_deg=lat0 + i * dlat,
                longitude_deg=8.0,
                relative_altitude_m=10.0 + i,
                yaw_deg=float(i),
            )
        )
    return base + (count - 1) * step


# ----------------------------------------------------------------------
# 帧上的拍摄时刻
# ----------------------------------------------------------------------
def test_frame_capture_timestamp() -> None:
    frame = make_frame(1000.0)
    assert frame.lag == pytest.approx(DEFAULT_TELEMETRY_LAG)
    assert frame.capture_timestamp == pytest.approx(999.85)
    # 拍摄至今 = 收到至今 + 链路延时（time.time() 量级下减法有 ~1e-7 的浮点误差）
    assert frame.capture_age > frame.age
    assert frame.capture_age - frame.age == pytest.approx(DEFAULT_TELEMETRY_LAG, abs=1e-5)

    # lag=0：拍摄时刻就是收到时刻（不做补偿）
    raw = make_frame(1000.0, lag=0.0)
    assert raw.capture_timestamp == 1000.0
    assert raw.capture_age == pytest.approx(raw.age)

    assert pytest.approx(0.15) == DEFAULT_TELEMETRY_LAG


def test_config_lag_reaches_frame(make_source) -> None:
    source = make_source(VideoConfig(url="udp://127.0.0.1:1"))
    assert source.config.telemetry_lag == DEFAULT_TELEMETRY_LAG

    # 未启动线程也可以直接发布一帧，验证配置确实写进了帧
    source._publish(np.zeros((4, 4, 3), dtype=np.uint8))
    frame = source.latest()
    assert frame is not None
    assert frame.lag == pytest.approx(DEFAULT_TELEMETRY_LAG)
    assert frame.capture_timestamp == pytest.approx(frame.timestamp - DEFAULT_TELEMETRY_LAG)

    custom = make_source(VideoConfig(url="udp://127.0.0.1:1", telemetry_lag=0.57))
    custom._publish(np.zeros((4, 4, 3), dtype=np.uint8))
    assert custom.latest().lag == pytest.approx(0.57)


def test_negative_lag_is_rejected() -> None:
    # 负延时等于把对齐结果推向未来，没有物理意义
    with pytest.raises(ValueError, match="telemetry_lag"):
        VideoConfig(telemetry_lag=-0.1).validated()


# ----------------------------------------------------------------------
# 对齐：内插 / nearest / 外推
# ----------------------------------------------------------------------
def test_align_interpolates() -> None:
    broker = make_broker()
    base = time.time() - 1.0
    latest = fill_history(broker, base, 10, 0.1)  # 纬度 47.000..47.009

    aligner = FrameTelemetryAligner(broker)  # 默认扣 150ms
    sample = aligner.align(make_frame(latest))
    assert sample is not None

    expected = latest - DEFAULT_TELEMETRY_LAG  # = base + 0.75
    assert sample.timestamp == pytest.approx(expected)
    assert not sample.extrapolated
    assert sample.offset == 0.0
    assert sample.mode == "interpolate"
    assert sample.lag == pytest.approx(DEFAULT_TELEMETRY_LAG)
    # 内插结果的 timestamp 就是查询时刻，便于调用方核对
    assert sample.snapshot.timestamp == pytest.approx(expected)
    # 落在第 8/9 条（47.007, 47.008）正中间
    assert sample.snapshot.latitude_deg == pytest.approx(47.0075, abs=1e-6)
    assert sample.snapshot.yaw_deg == pytest.approx(7.5, abs=1e-6)
    assert sample.image is sample.frame.image
    # 最新历史在 now-0.1，扣掉 150ms 后拍摄时刻约在 now-0.25
    assert 0.2 < sample.age < 0.4


def test_align_lag_priority() -> None:
    broker = make_broker()
    base = time.time() - 1.0
    latest = fill_history(broker, base, 10, 0.1)

    # 帧自带 600ms（例如换了链路后拉流源配置了新值）
    frame = make_frame(latest, lag=0.6)

    # 1) 对齐器不指定 lag → 用帧自带的值
    sample = FrameTelemetryAligner(broker).align(frame)
    assert sample.timestamp == pytest.approx(latest - 0.6)
    assert sample.lag == pytest.approx(0.6)

    # 2) 显式覆盖帧上的值
    sample = FrameTelemetryAligner(broker, lag=0.1).align(frame)
    assert sample.timestamp == pytest.approx(latest - 0.1)
    assert sample.lag == pytest.approx(0.1)
    assert not sample.extrapolated

    # 3) lag=0 → 不补偿，直接用收到时刻
    assert FrameTelemetryAligner(broker, lag=0.0).align(frame).timestamp == latest


def test_align_marks_extrapolation() -> None:
    broker = make_broker()
    base = time.time() - 1.0
    latest = fill_history(broker, base, 10, 0.1)
    aligner = FrameTelemetryAligner(broker)

    # 帧收到时间早于最早历史：扣掉 150ms 后落在历史之前 → 向后外推
    early = aligner.align(make_frame(base + 0.1))
    assert early is not None and early.extrapolated
    assert early.offset == pytest.approx(0.05, abs=1e-6)

    # 帧收到时间晚于最新历史（遥测断了，画面还在来）→ 向前外推
    late = aligner.align(make_frame(latest + 1.0))
    assert late is not None and late.extrapolated
    assert late.offset == pytest.approx(0.85, abs=1e-6)


def test_max_extrapolation_guard() -> None:
    broker = make_broker()
    base = time.time() - 1.0
    latest = fill_history(broker, base, 10, 0.1)
    guarded = FrameTelemetryAligner(broker, max_extrapolation=0.5)

    # 超限：宁可跳过这一帧，也不给一份"猜出来的"遥测
    assert guarded.align(make_frame(latest + 1.0)) is None
    # 未超限：正常返回（offset = 0.25）
    assert guarded.align(make_frame(latest + 0.4)) is not None


def test_align_without_history() -> None:
    broker = make_broker()
    aligner = FrameTelemetryAligner(broker)
    assert broker.history_span() is None
    assert aligner.align(make_frame(time.time())) is None
    assert aligner.align(None) is None

    # 只有一条历史：仍能给出结果（恒等于该条），且如实标记为外推
    broker.update_global_position(make_position(47.0))
    span = broker.history_span()
    assert span is not None and span[0] == span[1]
    sample = aligner.align(make_frame(time.time()))
    assert sample is not None
    assert sample.snapshot.latitude_deg == 47.0
    assert sample.extrapolated and sample.offset > 0.0


def test_align_nearest_mode() -> None:
    broker = make_broker()
    base = time.time() - 1.0
    fill_history(broker, base, 10, 0.1)

    aligner = FrameTelemetryAligner(broker, mode="nearest")
    # 收到时间 base+0.95 → 查询时刻 base+0.8，正好命中第 9 条（47.008）
    sample = aligner.align(make_frame(base + 0.95))
    assert sample is not None and sample.mode == "nearest"
    assert sample.snapshot.timestamp == pytest.approx(base + 0.8)
    assert sample.snapshot.latitude_deg == pytest.approx(47.008)


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"mode": "bilinear"}, id="非法 mode"),
        pytest.param({"lag": -1.0}, id="负 lag"),
        pytest.param({"max_extrapolation": -1.0}, id="负外推阈值"),
    ],
)
def test_aligner_rejects_bad_params(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        FrameTelemetryAligner(make_broker(), **kwargs)


def test_align_follows_live_updates() -> None:
    """走真实 update 路径（而非注入历史）：连续写入 0.4s 遥测后对齐一帧。"""
    broker = make_broker()
    lat = 47.0
    deadline = time.monotonic() + 0.4
    while time.monotonic() < deadline:
        broker.update_global_position(make_position(lat))
        lat += 0.001
        time.sleep(0.02)

    sample = FrameTelemetryAligner(broker).align(make_frame(time.time()))
    assert sample is not None
    assert not sample.extrapolated, (sample.extrapolated, sample.offset)

    lo, hi = sorted((broker._history[0].latitude_deg, broker._history[-1].latitude_deg))
    assert lo - 1e-9 <= sample.snapshot.latitude_deg <= hi + 1e-9
    assert sample.age < 0.5


# ----------------------------------------------------------------------
# P2：等待语义（max_wait + on_timeout="drop"）
# ----------------------------------------------------------------------
def test_align_no_wait_when_history_covers_capture_ts(broker: TelemetryBroker) -> None:
    """历史已覆盖拍摄时刻：不等待、纯内插，waits 计数 0。"""
    now = time.time()
    base = now - 5.0
    fill_history(broker, base, count=11, step=0.1)  # 最新 = base+1
    capture_ts = base + 0.5  # 在历史中段
    frame = make_frame(capture_ts + DEFAULT_TELEMETRY_LAG)

    aligner = FrameTelemetryAligner(broker, max_wait=1.0, on_timeout="drop")
    t0 = time.perf_counter()
    sample = aligner.align(frame)
    elapsed = time.perf_counter() - t0

    assert sample is not None
    assert not sample.extrapolated
    assert elapsed < 0.05, f"历史已覆盖却仍然等了 {elapsed * 1000:.0f}ms"
    s = aligner.stats
    assert s.waits == 0
    assert s.wait_timeouts == 0
    assert s.aligned == 1


def test_align_max_wait_zero_keeps_old_extrapolation(
    broker: TelemetryBroker,
) -> None:
    """max_wait=0（默认）= 未覆盖时直接外推，不等待。"""
    now = time.time()
    base = now - 5.0
    fill_history(broker, base, count=11, step=0.1)
    capture_ts = (base + 1) + 0.5  # 在历史之后
    frame = make_frame(capture_ts + DEFAULT_TELEMETRY_LAG)

    aligner = FrameTelemetryAligner(broker)  # 默认 max_wait=0
    sample = aligner.align(frame)

    assert sample is not None
    assert sample.extrapolated is True
    assert sample.offset == pytest.approx(0.5, abs=1e-6)
    s = aligner.stats
    assert s.waits == 0
    assert s.aligned == 1


def test_align_waits_then_interpolates_when_telemetry_catches_up(
    broker: TelemetryBroker,
) -> None:
    """max_wait 内追上了：内插出样本、waits 计数 +1、wait_timeouts=0。"""
    now = time.time()
    base = now - 5.0
    fill_history(broker, base, count=11, step=0.1)  # 最新 = base+1（5s 前）
    capture_ts = (base + 1) + 0.05  # 仅 0.05s 超出历史最新
    frame = make_frame(capture_ts + DEFAULT_TELEMETRY_LAG)

    aligner = FrameTelemetryAligner(broker, max_wait=1.0, on_timeout="drop")

    # 60ms 后注入新遥测（其时间戳自动为 time.time()，必 > capture_ts）
    def push() -> None:
        time.sleep(0.06)
        broker.update_global_position(make_position(lat=99.0))

    t = threading.Thread(target=push, daemon=True)
    t0 = time.perf_counter()
    t.start()
    sample = aligner.align(frame)
    elapsed = time.perf_counter() - t0
    t.join(timeout=1.0)

    assert sample is not None, "应当等到新遥测并产出样本"
    assert not sample.extrapolated, f"应为内插（offset={sample.offset}）"
    assert 0.04 < elapsed < 0.5, f"等待时长异常: {elapsed * 1000:.0f}ms"
    s = aligner.stats
    assert s.waits == 1
    assert s.wait_timeouts == 0
    assert s.aligned == 1
    assert s.max_wait_s >= 0.05


def test_align_times_out_and_drops_when_telemetry_stalls(
    broker: TelemetryBroker,
) -> None:
    """max_wait 内没追上：丢弃样本、wait_timeouts +1、返回 None。"""
    now = time.time()
    base = now - 5.0
    fill_history(broker, base, count=11, step=0.1)
    capture_ts = (base + 1) + 0.5
    frame = make_frame(capture_ts + DEFAULT_TELEMETRY_LAG)

    aligner = FrameTelemetryAligner(broker, max_wait=0.2, on_timeout="drop")
    t0 = time.perf_counter()
    sample = aligner.align(frame)
    elapsed = time.perf_counter() - t0

    assert sample is None
    assert 0.15 < elapsed < 0.5, f"等待时长异常: {elapsed * 1000:.0f}ms"
    s = aligner.stats
    assert s.waits == 1
    assert s.wait_timeouts == 1
    assert s.aligned == 0
    assert s.max_wait_s >= 0.15


def test_align_empty_history_skips_wait_and_reports_no_telemetry(
    broker: TelemetryBroker,
) -> None:
    """历史为空（一条遥测都没收到）：不等、计入 no_telemetry。"""
    frame = make_frame(time.time())
    aligner = FrameTelemetryAligner(broker, max_wait=1.0, on_timeout="drop")
    t0 = time.perf_counter()
    sample = aligner.align(frame)
    elapsed = time.perf_counter() - t0

    assert sample is None
    assert elapsed < 0.05
    s = aligner.stats
    assert s.no_telemetry == 1
    assert s.waits == 0
    assert s.wait_timeouts == 0


def test_align_constructor_validates_max_wait_and_on_timeout(
    broker: TelemetryBroker,
) -> None:
    with pytest.raises(ValueError, match="max_wait"):
        FrameTelemetryAligner(broker, max_wait=-0.1)
    with pytest.raises(ValueError, match="on_timeout"):
        FrameTelemetryAligner(broker, max_wait=0.5, on_timeout="ignore")


def test_writer_unsynced_via_aligner_stats(broker: TelemetryBroker) -> None:
    """``AlignmentWriter.skipped`` 是"丢了几帧"，细看对齐器 ``stats.wait_timeouts``。"""
    from airdrop import AlignmentBuffer, AlignmentWriter

    now = time.time()
    base = now - 5.0
    fill_history(broker, base, count=11, step=0.1)

    aligner = FrameTelemetryAligner(broker, max_wait=0.15, on_timeout="drop")
    buf = AlignmentBuffer(capacity=8, storage="raw")
    writer = AlignmentWriter(buf, aligner)

    for _ in range(3):
        capture_ts = (base + 1) + 1.0
        writer(make_frame(capture_ts + DEFAULT_TELEMETRY_LAG))

    assert writer.skipped == 3
    assert writer.written == 0
    assert aligner.stats.wait_timeouts == 3
