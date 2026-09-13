"""图传帧 ↔ 遥测时间对齐的离线测试。

不需要飞控与图传硬件：直接在 broker 的历史里注入**确定性**的时间戳，再用
手工构造的 :class:`~airdrop.video.VideoFrame` 验证整条链路——

    收到帧的时间 - 链路固定延时(150ms) → 按该时刻取遥测快照

覆盖内插、nearest、外推标记、``max_extrapolation`` 保护，以及延时优先级
（帧自带 / 显式覆盖）与参数校验。

运行：

    ./.venv/Scripts/python.exe tests/test_alignment.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# 让 `python tests/test_alignment.py` 与 `python -m tests.test_alignment` 都能跑
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from tests._harness import use_utf8_console  # noqa: E402

from airdrop import (  # noqa: E402
    DEFAULT_TELEMETRY_LAG,
    FrameTelemetryAligner,
    Hm30VideoSource,
    TelemetryBroker,
    TelemetrySnapshot,
    VideoConfig,
    VideoFrame,
)


class Fake:
    """最小 MAVSDK 对象替身（与 test_telemetry.py 同款）。"""


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


def test_frame_capture_timestamp() -> None:
    frame = make_frame(1000.0)
    assert abs(frame.lag - DEFAULT_TELEMETRY_LAG) < 1e-12, frame.lag
    assert abs(frame.capture_timestamp - 999.85) < 1e-9, frame.capture_timestamp
    # 拍摄至今 = 收到至今 + 链路延时
    assert frame.capture_age > frame.age
    assert abs((frame.capture_age - frame.age) - DEFAULT_TELEMETRY_LAG) < 1e-6

    # lag=0：拍摄时刻就是收到时刻（不做补偿）
    raw = make_frame(1000.0, lag=0.0)
    assert raw.capture_timestamp == 1000.0
    assert abs(raw.capture_age - raw.age) < 1e-6

    # 默认常量就是 150ms
    assert abs(DEFAULT_TELEMETRY_LAG - 0.15) < 1e-12
    print("[ok] VideoFrame 拍摄时刻 = 收到时间 - 链路延时")


def test_config_lag_reaches_frame() -> None:
    source = Hm30VideoSource(VideoConfig(url="udp://127.0.0.1:1"))
    custom = Hm30VideoSource(
        VideoConfig(url="udp://127.0.0.1:1", telemetry_lag=0.57)
    )
    try:
        assert source.config.telemetry_lag == DEFAULT_TELEMETRY_LAG
        # 未启动线程也可以直接发布一帧，验证配置确实写进了帧
        source._publish(np.zeros((4, 4, 3), dtype=np.uint8))
        frame = source.latest()
        assert frame is not None
        assert abs(frame.lag - DEFAULT_TELEMETRY_LAG) < 1e-12, frame.lag
        assert abs(
            frame.capture_timestamp - (frame.timestamp - DEFAULT_TELEMETRY_LAG)
        ) < 1e-9

        custom._publish(np.zeros((4, 4, 3), dtype=np.uint8))
        assert abs(custom.latest().lag - 0.57) < 1e-12
    finally:
        source.stop()
        custom.stop()

    # 负数延时直接拒绝（负延时等于把对齐结果推向未来，没有物理意义）
    try:
        VideoConfig(telemetry_lag=-0.1).validated()
    except ValueError as exc:
        assert "telemetry_lag" in str(exc), exc
    else:
        raise AssertionError("负数 telemetry_lag 应当报错")
    print("[ok] VideoConfig.telemetry_lag 写入每一帧（含非法值校验）")


def test_align_interpolates() -> None:
    broker = TelemetryBroker(history_maxlen=100, history_interval=0.0)
    base = time.time() - 1.0
    latest = fill_history(broker, base, 10, 0.1)  # base .. base+0.9，纬度 47.000..47.009

    aligner = FrameTelemetryAligner(broker)  # 默认扣 150ms
    frame = make_frame(latest)
    sample = aligner.align(frame)
    assert sample is not None, "有历史时不应返回 None"

    expected = latest - DEFAULT_TELEMETRY_LAG  # = base + 0.75
    assert abs(sample.timestamp - expected) < 1e-9, sample.timestamp
    assert not sample.extrapolated and sample.offset == 0.0, sample.offset
    assert sample.mode == "interpolate"
    assert sample.lag == DEFAULT_TELEMETRY_LAG
    # 内插结果的 timestamp 就是查询时刻，便于调用方核对
    assert abs(sample.snapshot.timestamp - expected) < 1e-9
    # 落在第 8/9 条(47.007, 47.008)正中间
    assert abs(sample.snapshot.latitude_deg - 47.0075) < 1e-6, (
        sample.snapshot.latitude_deg
    )
    # 只有一端有值的字段不会被内插破坏：yaw 两端都有，取中点
    assert abs(sample.snapshot.yaw_deg - 7.5) < 1e-6, sample.snapshot.yaw_deg
    assert sample.image is frame.image
    # 最新历史在 now-0.1，扣掉 150ms 后拍摄时刻约在 now-0.25
    assert 0.2 < sample.age < 0.4, sample.age
    print("[ok] 内插对齐（收到时间 - 150ms 落在两条历史之间）")


def test_align_lag_priority() -> None:
    broker = TelemetryBroker(history_maxlen=100, history_interval=0.0)
    base = time.time() - 1.0
    latest = fill_history(broker, base, 10, 0.1)

    # 帧自带 600ms（例如 OpenCV 后端拉来的帧）
    frame = make_frame(latest, lag=0.6)

    default = FrameTelemetryAligner(broker)
    sample = default.align(frame)
    assert abs(sample.timestamp - (latest - 0.6)) < 1e-9, sample.timestamp
    assert sample.lag == 0.6

    # 显式覆盖帧上的值（现场标定出新延时后临时试探）
    override = FrameTelemetryAligner(broker, lag=0.1)
    sample = override.align(frame)
    assert abs(sample.timestamp - (latest - 0.1)) < 1e-9
    assert sample.lag == 0.1
    assert not sample.extrapolated

    # lag=0：不补偿，直接用收到时刻
    zero = FrameTelemetryAligner(broker, lag=0.0)
    assert zero.align(frame).timestamp == latest
    print("[ok] 延时优先级（帧自带 / 显式覆盖 / 0 不补偿）")


def test_align_extrapolation() -> None:
    broker = TelemetryBroker(history_maxlen=100, history_interval=0.0)
    base = time.time() - 1.0
    latest = fill_history(broker, base, 10, 0.1)
    aligner = FrameTelemetryAligner(broker)

    # 帧收到时间早于最早历史：扣掉 150ms 后落在历史之前 → 外推
    early = aligner.align(make_frame(base + 0.1))
    assert early is not None and early.extrapolated
    assert abs(early.offset - 0.05) < 1e-6, early.offset

    # 帧收到时间晚于最新历史（遥测断了，画面还在来）→ 向前外推
    late = aligner.align(make_frame(latest + 1.0))
    assert late is not None and late.extrapolated
    assert abs(late.offset - 0.85) < 1e-6, late.offset

    # 外推保护：超限直接跳过该帧，而不是给一份"猜出来的"遥测
    guarded = FrameTelemetryAligner(broker, max_extrapolation=0.5)
    assert guarded.align(make_frame(latest + 1.0)) is None
    assert guarded.align(make_frame(latest + 0.4)) is not None  # offset=0.25
    print("[ok] 外推标记与 max_extrapolation 保护")


def test_align_without_history() -> None:
    broker = TelemetryBroker(history_maxlen=100, history_interval=0.0)
    aligner = FrameTelemetryAligner(broker)
    assert broker.history_span() is None
    assert aligner.align(make_frame(time.time())) is None, "历史为空时应返回 None"
    assert aligner.align(None) is None, "frame 为 None 时应返回 None"

    # 只有一条历史：仍能给出结果（恒等于该条），且如实标记为外推
    broker.update_global_position(make_position(47.0))
    span = broker.history_span()
    assert span is not None and span[0] == span[1]
    sample = aligner.align(make_frame(time.time()))
    assert sample is not None
    assert sample.snapshot.latitude_deg == 47.0
    assert sample.extrapolated and sample.offset > 0.0
    print("[ok] 无历史/单条历史下的对齐行为")


def test_align_nearest_and_validation() -> None:
    broker = TelemetryBroker(history_maxlen=100, history_interval=0.0)
    base = time.time() - 1.0
    fill_history(broker, base, 10, 0.1)

    aligner = FrameTelemetryAligner(broker, mode="nearest")
    # 收到时间 base+0.95 → 查询时刻 base+0.8，正好命中第 9 条(47.008)
    sample = aligner.align(make_frame(base + 0.95))
    assert sample is not None and sample.mode == "nearest"
    assert abs(sample.snapshot.timestamp - (base + 0.8)) < 1e-9
    assert abs(sample.snapshot.latitude_deg - 47.008) < 1e-9, (
        sample.snapshot.latitude_deg
    )

    for kwargs in (
        {"mode": "bilinear"},
        {"lag": -1.0},
        {"max_extrapolation": -1.0},
    ):
        try:
            FrameTelemetryAligner(broker, **kwargs)
        except ValueError:
            continue
        raise AssertionError(f"{kwargs} 应当抛出 ValueError")
    print("[ok] nearest 模式与非法参数校验")


def test_align_live_broker() -> None:
    """走真实 update 路径（而非注入历史）：连续写入 0.4s 遥测后对齐一帧。"""
    broker = TelemetryBroker(history_maxlen=200, history_interval=0.0)
    lat = 47.0
    deadline = time.monotonic() + 0.4
    while time.monotonic() < deadline:
        broker.update_global_position(make_position(lat))
        lat += 0.001
        time.sleep(0.02)

    frame = make_frame(time.time())
    sample = FrameTelemetryAligner(broker).align(frame)
    assert sample is not None
    assert not sample.extrapolated, (sample.extrapolated, sample.offset)

    first = broker._history[0].latitude_deg
    last = broker._history[-1].latitude_deg
    lo, hi = sorted((first, last))
    assert lo - 1e-9 <= sample.snapshot.latitude_deg <= hi + 1e-9, (
        sample.snapshot.latitude_deg
    )
    assert sample.age < 0.5, sample.age
    print("[ok] 真实时间路径（update 写历史 → 对齐落在历史范围内）")


def main() -> int:
    use_utf8_console()
    tests = [
        test_frame_capture_timestamp,
        test_config_lag_reaches_frame,
        test_align_interpolates,
        test_align_lag_priority,
        test_align_extrapolation,
        test_align_without_history,
        test_align_nearest_and_validation,
        test_align_live_broker,
    ]
    for test in tests:
        test()
    print(f"\ntest_alignment: 全部通过（{len(tests)} 组）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
