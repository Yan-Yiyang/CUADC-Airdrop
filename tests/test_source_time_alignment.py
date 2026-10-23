"""源时标字段对齐：异步传输延迟、变速仿真与失效边界。"""

from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast

import pytest

from airdrop.telemetry.broker import TelemetryBroker
from airdrop.telemetry.models import TelemetrySnapshot


def sample(source: float, received: float, *, position_delay: float = 0.0):
    """匀速直线位置，姿态保持单位四元数；可独立改变位置接收时刻。"""
    return TelemetrySnapshot(
        timestamp=received + position_delay,
        north_m=source * 10,
        east_m=0.0,
        down_m=-30.0,
        vx_m_s=10.0,
        vy_m_s=0.0,
        vz_m_s=0.0,
        quaternion_w=1.0,
        quaternion_x=0.0,
        quaternion_y=0.0,
        quaternion_z=0.0,
        position_timestamp_us=round(source * 1e6),
        quaternion_timestamp_us=round(source * 1e6),
        quaternion_received_at=received,
        position_velocity_received_at=received + position_delay,
    )


@pytest.mark.parametrize("delay", [0.0, 0.18, 0.35])
def test_position_transport_delay_does_not_change_source_interpolation(delay):
    """位置晚到且异步时，仍应按源时刻而非位置接收时间解算。"""
    broker = TelemetryBroker(source_time_alignment=True, history_interval=0)
    for i in range(8):
        broker._update_from_snapshot(sample(1 + i * 0.1, 100 + i * 0.1, position_delay=delay))
    result = broker.get_snapshot_at(100.45)
    assert result is not None
    assert result.north_m == pytest.approx(14.5)
    assert result.alignment_source_timestamp_us == 1_450_000


def test_variable_simulation_rate_uses_local_mapping():
    broker = TelemetryBroker(source_time_alignment=True, history_interval=0)
    for source, wall in [(1, 100), (1.1, 100.2), (1.2, 100.3), (1.3, 100.4)]:
        broker._update_from_snapshot(sample(source, wall))
    first, second = broker.get_snapshot_at(100.1), broker.get_snapshot_at(100.25)
    assert first is not None and second is not None
    assert first.north_m == pytest.approx(10.5)
    assert second.north_m == pytest.approx(11.5)


def test_missing_or_stalled_source_fields_are_rejected_and_can_be_disabled():
    broker = TelemetryBroker(source_time_alignment=True, history_interval=0)
    broker._update_from_snapshot(TelemetrySnapshot(timestamp=100, north_m=1))
    broker._update_from_snapshot(TelemetrySnapshot(timestamp=101, north_m=2))
    assert broker.get_snapshot_at(100.5) is None
    broker.source_time_alignment = False
    result = broker.get_snapshot_at(100.5)
    assert result is not None and result.north_m == 1.5
    broker.reset()
    broker.source_time_alignment = True
    for source, wall in [(1, 100), (2, 101)]:
        broker._update_from_snapshot(sample(source, wall))
    assert broker.get_snapshot_at(100.5) is None


def test_reset_clears_both_source_histories():
    broker = TelemetryBroker(source_time_alignment=True, history_interval=0)
    for i in range(4):
        broker._update_from_snapshot(sample(10 + 0.1 * i, 100 + 0.1 * i))
    assert broker.get_snapshot_at(100.15) is not None
    # 飞控重启后保留 Windows 墙钟递增；不能与旧源钟插值。
    broker._update_from_snapshot(sample(0.1, 101))
    assert broker.get_snapshot_at(100.15) is None
    broker.reset()
    assert broker.get_snapshot_at(101) is None


def test_source_clock_restart_does_not_reuse_held_old_attitude():
    """位置源钟回退后，合并快照携带的旧四元数不能重入新钟域。"""
    broker = TelemetryBroker(source_time_alignment=True, history_interval=0)
    broker._update_from_snapshot(
        TelemetrySnapshot(
            timestamp=100.20,
            north_m=2.0,
            east_m=0.0,
            down_m=-10.0,
            vx_m_s=1.0,
            vy_m_s=0.0,
            vz_m_s=0.0,
            position_timestamp_us=2_000_000,
            position_velocity_received_at=100.20,
            quaternion_w=1.0,
            quaternion_x=0.0,
            quaternion_y=0.0,
            quaternion_z=0.0,
            quaternion_timestamp_us=50_000,
            quaternion_received_at=100.00,
        )
    )
    # 新位置时标回退；这些快照仍携带旧的 50ms 四元数状态。
    for source, received in ((100_000, 100.21), (200_000, 100.22)):
        broker._update_from_snapshot(
            TelemetrySnapshot(
                timestamp=received,
                north_m=source / 1e6,
                east_m=0.0,
                down_m=-10.0,
                vx_m_s=1.0,
                vy_m_s=0.0,
                vz_m_s=0.0,
                position_timestamp_us=source,
                position_velocity_received_at=received,
                quaternion_w=1.0,
                quaternion_x=0.0,
                quaternion_y=0.0,
                quaternion_z=0.0,
                quaternion_timestamp_us=50_000,
                quaternion_received_at=100.00,
            )
        )
    broker._update_from_snapshot(
        TelemetrySnapshot(
            timestamp=100.23,
            position_timestamp_us=200_000,
            position_velocity_received_at=100.22,
            quaternion_w=1.0,
            quaternion_x=0.0,
            quaternion_y=0.0,
            quaternion_z=0.0,
            quaternion_timestamp_us=150_000,
            quaternion_received_at=100.23,
        )
    )

    assert broker.get_snapshot_at(100.20) is None


def test_quaternion_timestamp_is_its_own_not_euler_timestamp():
    snapshot = TelemetrySnapshot(attitude_timestamp_us=123)
    snapshot.update_attitude_quaternion(SimpleNamespace(w=1, x=0, y=0, z=0, timestamp_us=456))
    assert snapshot.quaternion_timestamp_us == 456
    assert snapshot.attitude_timestamp_us == 123


def test_quaternion_slerp_and_euler_are_consistent():
    import math

    broker = TelemetryBroker(source_time_alignment=True, history_interval=0)
    a = sample(1, 100)
    b = replace(
        sample(1.1, 100.1), quaternion_w=math.cos(math.pi / 4), quaternion_z=math.sin(math.pi / 4)
    )
    broker._update_from_snapshot(a)
    broker._update_from_snapshot(b)
    result = broker.get_snapshot_at(100.05)
    assert result is not None
    assert result.yaw_deg == pytest.approx(45)
    assert result.quaternion_w == pytest.approx(math.cos(math.pi / 8))


def test_raw_position_stream_retains_boot_timestamp():
    import asyncio
    import json

    from airdrop.telemetry.mavsdk_thread import MavsdkThread

    async def messages(name):
        assert name == "LOCAL_POSITION_NED"
        yield SimpleNamespace(
            fields_json=json.dumps(dict(time_boot_ms=1234, x=1, y=2, z=-3, vx=4, vy=5, vz=6))
        )

    broker = TelemetryBroker(source_time_alignment=True)
    thread = MavsdkThread(broker, source_time_alignment=True)
    drone = SimpleNamespace(mavlink_direct=SimpleNamespace(message=messages))
    asyncio.run(thread._stream_source_position(cast(Any, drone)))
    result = broker.get_snapshot()
    assert result.position_timestamp_us == 1_234_000
    assert result.north_m == 1
    assert result.position_velocity_received_at == result.timestamp


def test_small_wall_clock_rollback_clears_stale_source_fields(monkeypatch):
    import airdrop.telemetry.broker as broker_module

    times = iter((100.0, 100.04, 100.08, 100.05))
    monkeypatch.setattr(broker_module.time, "time", lambda: next(times))
    broker = TelemetryBroker(history_interval=0.1, source_time_alignment=True)

    for stamp in (1_000_000, 1_040_000, 1_080_000):

        def update_attitude(snapshot: TelemetrySnapshot, stamp: int = stamp) -> None:
            snapshot.quaternion_w = 1.0
            snapshot.quaternion_x = 0.0
            snapshot.quaternion_y = 0.0
            snapshot.quaternion_z = 0.0
            snapshot.quaternion_timestamp_us = stamp

        broker._update(update_attitude, quaternion_received=True)

    def update_position(snapshot: TelemetrySnapshot) -> None:
        snapshot.north_m = 1.0
        snapshot.east_m = 2.0
        snapshot.down_m = -3.0
        snapshot.vx_m_s = 0.0
        snapshot.vy_m_s = 0.0
        snapshot.vz_m_s = 0.0
        snapshot.position_timestamp_us = 1_050_000

    broker._update(update_position, position_velocity_received=True)

    assert not broker._source_history.attitude
    assert len(broker._source_history.position) == 1
    assert broker.get_snapshot().quaternion_received_at is None


def test_mavsdk_thread_rejects_conflicting_alignment_configuration():
    from airdrop.telemetry.mavsdk_thread import MavsdkThread

    broker = TelemetryBroker(source_time_alignment=True)
    with pytest.raises(ValueError, match="配置不一致"):
        MavsdkThread(broker, source_time_alignment=False)
