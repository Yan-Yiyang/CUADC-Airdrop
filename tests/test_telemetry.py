"""MAVSDK 基础设施测试。

不需要真实飞控：遥测部分用最小替身对象构造数据，MAVSDK 部分只验证启停、错误路径
与"会话结束必须释放 mavsdk_server"这类关键约定。
"""

from __future__ import annotations

import math
import threading
import time
from typing import TYPE_CHECKING, Any

import pytest

from airdrop import Command, MavsdkThread, TelemetryBroker
from airdrop.config import TelemetryConfig


class Fake:
    """最小 MAVSDK 对象替身。

    字段是随手挂上去的（``p.latitude_deg = ...``），所以显式声明成"任意属性"，
    类型检查器才不会把未声明的赋值判成错误。
    """

    if TYPE_CHECKING:

        def __getattr__(self, name: str) -> Any: ...
        def __setattr__(self, name: str, value: Any) -> None: ...


def make_position(lat, lon, abs_alt, rel_alt):
    p = Fake()
    p.latitude_deg = lat
    p.longitude_deg = lon
    p.absolute_altitude_m = abs_alt
    p.relative_altitude_m = rel_alt
    return p


def make_euler(roll, pitch, yaw, ts_us=0):
    e = Fake()
    e.roll_deg = roll
    e.pitch_deg = pitch
    e.yaw_deg = yaw
    e.timestamp_us = ts_us
    return e


def make_quat(w, x, y, z):
    q = Fake()
    q.w = w
    q.x = x
    q.y = y
    q.z = z
    return q


def make_wind(x, y, z, alt_msl=100.0):
    """MAVSDK ``Wind`` 的字段名是 ``wind_x/y/z_ned_m_s``（x=北、y=东、z=地）。"""
    w = Fake()
    w.wind_x_ned_m_s = x
    w.wind_y_ned_m_s = y
    w.wind_z_ned_m_s = z
    w.wind_altitude_msl_m = alt_msl
    return w


def test_wind_fields_are_published_and_interpolated(broker: TelemetryBroker) -> None:
    """风估计是可选流，但既然进了快照，就该和其它数值字段一样能查能内插。"""
    assert broker.get_snapshot().wind_north_m_s is None, "没有风估计时应当是 None"

    s1 = broker.update_wind(make_wind(2.0, -1.0, 0.5))
    assert broker.get_snapshot().wind_north_m_s == pytest.approx(2.0)
    assert broker.get_snapshot().wind_east_m_s == pytest.approx(-1.0)
    assert broker.get_snapshot().wind_down_m_s == pytest.approx(0.5)

    time.sleep(0.01)
    s2 = broker.update_wind(make_wind(6.0, -3.0, 1.5))
    mid = (s1.timestamp + s2.timestamp) / 2
    interp = broker.get_snapshot_at(mid, mode="interpolate")
    assert interp is not None
    assert interp.wind_north_m_s == pytest.approx(4.0, abs=0.5), "风也该线性内插"
    assert interp.wind_east_m_s == pytest.approx(-2.0, abs=0.5)


def test_wind_fields_survive_json_round_trip() -> None:
    """回放要靠 as_dict/字典重建；风字段必须跟着走，否则回放里的弹道会静默变零风。"""
    import json

    from airdrop.telemetry.models import TelemetrySnapshot

    snapshot = TelemetrySnapshot(timestamp=1.0)
    snapshot.update_wind(make_wind(3.0, 4.0, 0.0))
    payload = json.loads(json.dumps(snapshot.as_dict()))
    assert payload["wind_north_m_s"] == pytest.approx(3.0)
    assert payload["wind_east_m_s"] == pytest.approx(4.0)
    restored = TelemetrySnapshot(
        **{k: v for k, v in payload.items() if k in TelemetrySnapshot.__dataclass_fields__}
    )
    assert restored.wind_north_m_s == pytest.approx(3.0)


def test_in_air_stream_is_published_to_the_snapshot(broker: TelemetryBroker) -> None:
    """``in_air`` 是可选流，但既然进了快照，就要能查、能推、能按时刻取。

    状态机的 ``WAIT_AIRBORNE`` 那道门读的就是它：``None`` 表示"这条流还没数据"
    （退化到 ``relative_altitude_m`` 兜底），**不是**"在地面上"——两者不能混。
    """
    assert broker.get_snapshot().in_air is None, "没有这条流时必须是 None，别默认 False"

    pushed: list[bool | None] = []
    broker.subscribe(lambda snapshot: pushed.append(snapshot.in_air))

    broker.update_in_air(True)
    assert broker.get_snapshot().in_air is True
    assert broker._history[-1].in_air is True, "也要进历史，回放/按时刻查询才看得到"

    broker.update_in_air(0)  # 飞控侧偶尔给 0/1，统一按 bool 归一
    assert broker.get_snapshot().in_air is False
    nearest = broker.get_snapshot_at(time.time(), mode="nearest")
    assert nearest is not None and nearest.in_air is False
    assert pushed[-2:] == [True, False], "订阅推送也要带上这个字段"


def test_in_air_survives_json_round_trip() -> None:
    """回放靠 ``as_dict`` / 字典重建快照：``in_air`` 丢了，回放就会卡在 WAIT_AIRBORNE。"""
    import json

    from airdrop.telemetry.models import TelemetrySnapshot

    snapshot = TelemetrySnapshot(timestamp=1.0, in_air=True)
    payload = json.loads(json.dumps(snapshot.as_dict()))
    assert payload["in_air"] is True
    restored = TelemetrySnapshot(
        **{k: v for k, v in payload.items() if k in TelemetrySnapshot.__dataclass_fields__}
    )
    assert restored.in_air is True


def test_broker_snapshot_interpolation_subscribe(broker: TelemetryBroker, received: list) -> None:
    s1 = broker.update_global_position(make_position(47.0, 8.0, 500.0, 10.0))

    snap = broker.get_snapshot()
    assert snap.latitude_deg == 47.0
    snap.latitude_deg = 0.0  # 修改副本不影响仓库
    assert broker.get_snapshot().latitude_deg == 47.0

    time.sleep(0.01)
    s2 = broker.update_global_position(make_position(49.0, 9.0, 600.0, 20.0))

    # 线性内插
    mid = (s1.timestamp + s2.timestamp) / 2
    interp = broker.get_snapshot_at(mid, mode="interpolate")
    assert interp is not None
    assert 47.4 < interp.latitude_deg < 48.6

    # nearest
    nearest = broker.get_snapshot_at(s1.timestamp, mode="nearest")
    assert nearest is not None and nearest.latitude_deg == 47.0

    # 订阅推送（前两次 yaw 为 None）
    assert received == [None, None]
    broker.update_attitude_euler(make_euler(1.0, 2.0, 90.0, 123))
    assert received[-1] == 90.0

    broker.reset()
    assert broker.get_snapshot_at(time.time()) is None
    assert not broker.get_snapshot().is_valid()


def test_wait_next_snapshot(broker: TelemetryBroker) -> None:
    got: list = []

    def waiter() -> None:
        got.append(
            broker.wait_next_snapshot(
                timeout=2.0,
                predicate=lambda s: s.roll_deg is not None and s.roll_deg > 5,
            )
        )

    thread = threading.Thread(target=waiter)
    thread.start()
    time.sleep(0.1)
    broker.update_attitude_euler(make_euler(10.0, 0.0, 45.0, 456))
    thread.join(timeout=3.0)

    assert got and got[0] is not None
    assert got[0].roll_deg == 10.0


def test_query_mode_validation(broker: TelemetryBroker) -> None:
    broker.update_global_position(make_position(47.0, 8.0, 500.0, 10.0))
    with pytest.raises(ValueError, match="不支持的查询模式"):
        broker.get_snapshot_at(time.time(), mode="bilinear")
    # 历史为空时也一样抛：错误语义不该随"有没有收到遥测"而变
    empty = TelemetryBroker()
    with pytest.raises(ValueError, match="不支持的查询模式"):
        empty.get_snapshot_at(time.time(), mode="bilinear")
    assert empty.get_snapshot_at(time.time(), mode="nearest") is None


def test_history_throttle() -> None:
    broker = TelemetryBroker(history_maxlen=100, history_interval=0.05)
    for i in range(5):
        broker.update_global_position(make_position(47.0 + i, 8.0, 500.0, 10.0))
    # 5 次更新间隔远小于 50ms，只有第一条进入历史
    assert len(broker._history) == 1

    time.sleep(0.06)
    broker.update_global_position(make_position(52.0, 8.0, 500.0, 10.0))
    assert len(broker._history) == 2
    # 最新快照不受节流影响，总是实时的
    assert broker.get_snapshot().latitude_deg == 52.0

    broker.reset()
    broker.update_global_position(make_position(47.0, 8.0, 500.0, 10.0))
    assert len(broker._history) == 1


def test_angle_wrap_interpolation() -> None:
    broker = TelemetryBroker(history_maxlen=100, history_interval=0.0)
    s1 = broker.update_attitude_euler(make_euler(0.0, 0.0, 179.0, 1))
    time.sleep(0.01)
    s2 = broker.update_attitude_euler(make_euler(0.0, 0.0, -179.0, 2))

    # 179° 与 -179° 的中点应接近 ±180°，而不是朴素线性插值得到的 0°
    snap = broker.get_snapshot_at((s1.timestamp + s2.timestamp) / 2)
    assert snap.yaw_deg is not None
    assert abs(abs(snap.yaw_deg) - 180.0) < 1.0


def test_quaternion_slerp() -> None:
    broker = TelemetryBroker(history_maxlen=100, history_interval=0.0)
    s1 = broker.update_attitude_quaternion(make_quat(1.0, 0.0, 0.0, 0.0))
    time.sleep(0.01)
    s2 = broker.update_attitude_quaternion(make_quat(0.0, 0.0, 0.0, 1.0))

    # 绕 z 转 180° 的 slerp 中点：w = z = √2/2，且模长保持 1
    snap = broker.get_snapshot_at((s1.timestamp + s2.timestamp) / 2)
    assert snap.quaternion_w == pytest.approx(math.sqrt(0.5), abs=0.01)
    assert snap.quaternion_z == pytest.approx(math.sqrt(0.5), abs=0.01)
    norm = math.sqrt(
        snap.quaternion_w**2 + snap.quaternion_x**2 + snap.quaternion_y**2 + snap.quaternion_z**2
    )
    assert norm == pytest.approx(1.0, abs=1e-6)


def test_quaternion_sign_flip() -> None:
    """q 与 -q 表示同一姿态：slerp 走最短路径，不能坍缩成零向量。"""
    broker = TelemetryBroker(history_maxlen=100, history_interval=0.0)
    s1 = broker.update_attitude_quaternion(make_quat(1.0, 0.0, 0.0, 0.0))
    time.sleep(0.01)
    s2 = broker.update_attitude_quaternion(make_quat(-1.0, 0.0, 0.0, 0.0))

    snap = broker.get_snapshot_at((s1.timestamp + s2.timestamp) / 2)
    assert abs(abs(snap.quaternion_w) - 1.0) < 0.01


def test_history_bisect_at_scale() -> None:
    """写满 1200 条历史（继续写触发左端驱逐）后，查询仍应精确。

    二分直接作用在 deque 上（``bisect`` 的 key 参数），不重建整张时间列表。
    """
    broker = TelemetryBroker(history_maxlen=1200, history_interval=0.0)
    for i in range(1500):
        broker.update_global_position(make_position(40.0 + i * 0.001, 8.0, 500.0 + i, 10.0))
        time.sleep(0.0002)
    assert len(broker._history) == 1200

    # 范围内：正好命中某条历史时，内插与 nearest 都应返回该条的值
    target = broker._history[500]
    snap = broker.get_snapshot_at(target.timestamp)
    assert snap.latitude_deg == pytest.approx(target.latitude_deg, abs=1e-9)
    nearest = broker.get_snapshot_at(target.timestamp, mode="nearest")
    assert nearest.latitude_deg == target.latitude_deg

    # 两条历史正中间：内插值应落在两端之间
    left, right = broker._history[700], broker._history[701]
    snap = broker.get_snapshot_at((left.timestamp + right.timestamp) / 2)
    lo, hi = sorted((left.latitude_deg, right.latitude_deg))
    assert lo - 1e-9 <= snap.latitude_deg <= hi + 1e-9

    # 两端外推：早出/晚出历史范围都应给出结果而不是报错
    assert broker.get_snapshot_at(broker._history[0].timestamp - 10.0) is not None
    assert broker.get_snapshot_at(broker._history[-1].timestamp + 10.0) is not None
    near_first = broker.get_snapshot_at(broker._history[0].timestamp - 10.0, mode="nearest")
    assert near_first.latitude_deg == broker._history[0].latitude_deg


def test_mavsdk_thread_lifecycle_and_errors() -> None:
    broker = TelemetryBroker()
    mav = MavsdkThread(broker=broker, reconnect=False)
    mav.start()
    try:
        r1 = mav.send_command(Command("no_such_command"), timeout=5.0)
        assert not r1.success and "不支持" in r1.error

        r2 = mav.send_command(Command("arm"), timeout=5.0)
        assert not r2.success and "尚未连接" in r2.error
        assert not mav.connected
    finally:
        mav.stop()

    # 重复 start/stop 也应正常工作
    mav.start()
    mav.stop()


def test_drone_released_on_session_end(monkeypatch) -> None:
    """会话结束时必须显式终止 mavsdk_server，而不是等 GC 收 ``__del__``。"""
    import airdrop.telemetry.mavsdk_thread as mt

    released: list[bool] = []

    class FakeSystem:
        def __init__(self, *args, **kwargs):
            pass

        async def connect(self, system_address=None):
            raise RuntimeError("connect failed (fake)")

        def _stop_mavsdk_server(self):
            released.append(True)

    monkeypatch.setattr(mt, "System", FakeSystem)

    mav = mt.MavsdkThread(reconnect=False)
    supervisor = mav.submit(mav._serve("udpin://127.0.0.1:1"))
    supervisor.result(timeout=10.0)
    mav.stop()

    assert released, "会话结束后未显式终止 mavsdk_server 子进程"


def test_release_never_blocks_the_caller(monkeypatch) -> None:
    """回归：释放 mavsdk_server 卡住时，会话收尾与 ``stop()`` 都必须有超时兜底。

    实测（2026-09）：没有飞控/SITL、``connect()`` 超时之后，这条释放路径会**永久阻塞**
    （gRPC poller 线程报 ``Event loop is closed``）。挂在事件循环线程上会卡死会话收尾，
    挂在调用方线程上会把 ``stop()`` 一起拖住——``tests/test_sitl_recon.py`` 的探活因此
    skip 不了，全量测试永远跑不完。``_release_drone`` 现在把释放放进守护线程，
    只等 ``RELEASE_TIMEOUT_S`` 秒。
    """
    import airdrop.telemetry.mavsdk_thread as mt

    entered = threading.Event()
    forever = threading.Event()

    class HangingSystem:
        def __init__(self, *args, **kwargs):
            pass

        async def connect(self, system_address=None):
            raise RuntimeError("connect failed (fake)")

        def _stop_mavsdk_server(self):
            entered.set()
            forever.wait()  # 模拟实测到的"永不返回"

    monkeypatch.setattr(mt, "System", HangingSystem)

    mav = mt.MavsdkThread(reconnect=False)
    started = time.monotonic()
    supervisor = mav.submit(mav._serve("udpin://127.0.0.1:1"))
    supervisor.result(timeout=30.0)  # 会话收尾不能被释放路径卡死
    mav.stop()  # 调用方也不能被拖死
    elapsed = time.monotonic() - started

    assert entered.is_set(), "没走到释放路径，用例没覆盖到目标场景"
    assert elapsed < mt.RELEASE_TIMEOUT_S * 2 + 5.0, f"收尾耗时 {elapsed:.1f}s，超时兜底没生效"


# ----------------------------------------------------------------------
# P2：连接就绪后下发遥测速率（set_rate_*）
# ----------------------------------------------------------------------
def test_apply_telemetry_rates_calls_set_rate_with_configured_hz() -> None:
    """用 MAVSDK 的 FakeSystem：_apply_telemetry_rates 应按配置下发各字段的速率。"""
    import asyncio

    import airdrop.telemetry.mavsdk_thread as mt

    calls: dict[str, float] = {}

    def make_fake_telemetry() -> "object":
        class _T:
            async def set_rate_position(self, rate):
                calls["position"] = rate

            async def set_rate_position_velocity_ned(self, rate):
                calls["pvn"] = rate

            async def set_rate_attitude_euler(self, rate):
                calls["euler"] = rate

            async def set_rate_attitude_quaternion(self, rate):
                calls["quat"] = rate

        return _T()

    class FakeDrone:
        telemetry = make_fake_telemetry()

    config = TelemetryConfig(
        position_rate_hz=10.0,
        position_velocity_ned_rate_hz=12.0,
        attitude_rate_hz=33.0,
    )
    mav = mt.MavsdkThread.from_config(config, broker=TelemetryBroker())
    asyncio.run(mav._apply_telemetry_rates(FakeDrone()))  # pyright: ignore[reportArgumentType] - 替身只实现被调到的那几个 setter

    assert calls["position"] == 10.0
    assert calls["pvn"] == 12.0
    assert calls["euler"] == 33.0
    assert calls["quat"] == 33.0  # 姿态用同一速率覆盖四元数与欧拉


def test_apply_telemetry_rates_skips_when_rate_is_none() -> None:
    """默认 None 时不下发（飞控沿用默认速率，不主动覆盖）。"""
    import asyncio

    import airdrop.telemetry.mavsdk_thread as mt

    calls: list[str] = []

    class _T:
        async def set_rate_position(self, rate):
            calls.append("position")

        async def set_rate_position_velocity_ned(self, rate):
            calls.append("pvn")

        async def set_rate_attitude_euler(self, rate):
            calls.append("euler")

        async def set_rate_attitude_quaternion(self, rate):
            calls.append("quat")

    class FakeDrone:
        telemetry = _T()

    mav = mt.MavsdkThread()  # 全部 rate=None
    asyncio.run(mav._apply_telemetry_rates(FakeDrone()))  # pyright: ignore[reportArgumentType] - 替身只实现被调到的那几个 setter
    assert calls == []


def test_apply_telemetry_rates_swallows_individual_failures() -> None:
    """单条 set_rate 失败不应影响其它流（飞控/固件可能不支持个别流）。"""
    import asyncio

    import airdrop.telemetry.mavsdk_thread as mt

    calls: list[str] = []

    class _T:
        async def set_rate_position(self, rate):
            calls.append("position")

        async def set_rate_position_velocity_ned(self, rate):
            raise RuntimeError("firmware not supported")

        async def set_rate_attitude_euler(self, rate):
            calls.append("euler")

        async def set_rate_attitude_quaternion(self, rate):
            calls.append("quat")

    class FakeDrone:
        telemetry = _T()

    config = TelemetryConfig(
        position_rate_hz=10.0,
        position_velocity_ned_rate_hz=10.0,
        attitude_rate_hz=30.0,
    )
    mav = mt.MavsdkThread.from_config(config, broker=TelemetryBroker())
    asyncio.run(mav._apply_telemetry_rates(FakeDrone()))  # pyright: ignore[reportArgumentType] - 替身只实现被调到的那几个 setter  # 不应抛
    assert "position" in calls and "euler" in calls
