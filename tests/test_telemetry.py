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

from airdrop import Command, MavsdkThread, TelemetryBroker, TelemetrySnapshot
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


def test_wall_clock_regression_does_not_freeze_or_corrupt_history() -> None:
    """墙钟回退不能让历史停写，更不能留下乱序数据。

    节流判据曾用 ``snapshot.timestamp - _last >= interval``，而 ``time.time()``
    在 Windows 上受 NTP / 手动改表 / 虚拟机时间同步影响**可以往回跳**（本项目的
    SITL 就在 WSL 虚拟机里跑）。回退 t 秒 ⇒ 接下来 t 秒一条都不入库，而
    ``connected`` 仍是 True、``align()`` 也不报错——遥测静默停摆。

    更糟的是回退后历史不再是升序，而内插的正确性完全建立在升序上：
    ``_evaluate_snapshot_locked`` 用 ``history[0]`` / ``history[-1]`` 当最小/最大，
    乱序后这两个边界判断本身就错 ⇒ 本该内插的查询被送进外推分支，且
    ``dt`` 可能为负，产出"看起来完全正常"的错误快照。
    """
    import airdrop.telemetry.broker as broker_mod

    base = 1_700_000_000.0
    wall = {"t": base}
    mono = {"t": 1000.0}
    real_time, real_mono = time.time, time.monotonic
    broker_mod.time.time = lambda: wall["t"]  # type: ignore[assignment]
    broker_mod.time.monotonic = lambda: mono["t"]  # type: ignore[assignment]
    try:
        broker = TelemetryBroker(history_maxlen=200, history_interval=0.1)
        for i in range(20):
            wall["t"] = base + i * 0.1
            mono["t"] = 1000.0 + i * 0.1
            broker.update_global_position(make_position(30.0 + i * 0.001, 8.0, 500.0, 10.0))
        before = len(broker._history)
        assert before > 5, "正常情况下历史应当增长"

        # 墙钟回退 1 小时，monotonic 照常前进
        for i in range(20):
            wall["t"] = base - 3600.0 + i * 0.1
            mono["t"] = 1002.0 + i * 0.1
            broker.update_global_position(make_position(31.0 + i * 0.001, 8.0, 500.0, 10.0))

        stamps = [s.timestamp for s in broker._history]
        assert stamps == sorted(stamps), f"历史乱序：{stamps}"
        # 关键：即使墙钟仍处于回退状态，节流也必须照常推进（用单调钟）
        assert len(broker._history) > 5, "墙钟回退后历史停写了（节流判据用错了时钟）"
    finally:
        broker_mod.time.time = real_time  # type: ignore[assignment]
        broker_mod.time.monotonic = real_mono  # type: ignore[assignment]


def test_stop_takes_the_supervisor_lock_that_connect_holds() -> None:
    """``stop()`` 必须与 ``connect()`` 拿同一把 ``_supervisor_lock``。

    原来 stop() 全程不持锁，可复现的时序是：
      A(connect) 判完 ``_supervisor.done()`` 为 True，正准备装入新的 _serve；
      B(stop)   置位 _stop_requested、取走并 cancel 掉**旧** supervisor、停循环、
                join、清引用——B 认为已经干净了；
      A 接着把 _stop_requested 清掉（抹掉 B 的停止意图）并装入新的 _serve
        → System() → mavsdk_server 子进程占死固定的 gRPC 50051，
        而 _thread/_loop 引用已被 B 清成 None ⇒ 之后没人能停它。

    竞态本身难确定性复现，这里钉住它的必要条件：stop() 也要进这把锁。
    """
    mav = MavsdkThread(reconnect=False)
    entered = threading.Event()
    release = threading.Event()
    done = threading.Event()

    def blocker() -> None:
        with mav._supervisor_lock:
            entered.set()
            release.wait(timeout=5.0)

    holder = threading.Thread(target=blocker, daemon=True)
    holder.start()
    assert entered.wait(timeout=5.0), "前置条件：先占住 _supervisor_lock"

    def stopper() -> None:
        mav.stop(timeout=0.1)
        done.set()

    worker = threading.Thread(target=stopper, daemon=True)
    worker.start()
    time.sleep(0.2)
    assert not done.is_set(), "stop() 没有等 _supervisor_lock —— 与 connect() 存在竞态"

    release.set()
    assert done.wait(timeout=5.0), "释放锁后 stop() 应当完成"
    holder.join(timeout=2.0)
    worker.join(timeout=2.0)


def test_broker_history_is_discarded_when_a_session_ends() -> None:
    """会话一结束就作废代理历史，重连窗口里不得继续拿上一条会话的数据外推。

    ``broker.reset()`` 原来在 ``connect()`` **成功之后**才调，于是重连窗口
    （reconnect_delay 5s + connect_timeout 30s，最长 35s）里 broker 还留着上一条
    会话的历史。``connected`` 已被清成 False，但 ``align()`` 只看
    ``history_span()``、完全不看 connected ⇒ 它会从旧历史向后外推最多 35 秒，
    产出 extrapolated=True / offset 很大的样本，而默认 max_extrapolation=None
    不过滤。任务路径有 telemetry_stale_s 看门狗兜着（会 ABORT），纯感知链路没有。
    """
    import asyncio

    from airdrop.telemetry import mavsdk_thread as mt

    broker = TelemetryBroker(history_maxlen=50, history_interval=0.0)
    mav = mt.MavsdkThread(broker=broker, reconnect=False)
    broker.publish(TelemetrySnapshot(timestamp=100.0, north_m=1.0))
    broker.publish(TelemetrySnapshot(timestamp=100.1, north_m=2.0))
    assert broker.history_span() is not None, "前置条件：会话期间历史是有的"

    def boom(*args: object, **kwargs: object) -> object:
        raise RuntimeError("模拟 mavsdk_server 起不来（无飞控）")

    monkey = pytest.MonkeyPatch()
    # 只让 System() 失败，_run_session 本身照真实控制流跑完（含 finally）
    monkey.setattr(mt, "System", boom, raising=True)
    try:
        asyncio.run(mav._serve("udpin://0.0.0.0:14540"))
    finally:
        monkey.undo()

    assert broker.history_span() is None, "会话结束后历史必须已作废（不能跨会话外推）"
    assert mav.connected is False


def test_discrete_fields_are_never_interpolated() -> None:
    """``in_air`` / ``mission_current`` / ``mission_total`` 不是连续量，不能内插。

    ``isinstance(a, (int, float))`` 拦不住它们（bool 是 int 的子类），于是曾经：
    ``in_air: True → False`` 内插出 ``0.5``（真值）⇒ "等起飞"判据把整个过渡
    窗口判成"在空中"；``mission_current: 2 → 4`` 内插出 ``3.0`` ⇒
    ``int(3.0)`` 让任务完成判定提前一格。
    """
    broker = TelemetryBroker(history_maxlen=100, history_interval=0.0)
    broker._history.append(
        TelemetrySnapshot(
            timestamp=1.0, in_air=True, mission_current=2, mission_total=4, roll_deg=10.0
        )
    )
    broker._history.append(
        TelemetrySnapshot(
            timestamp=2.0, in_air=False, mission_current=4, mission_total=4, roll_deg=20.0
        )
    )

    mid = broker.get_snapshot_at(1.5)
    assert mid.in_air is True, f"in_air 内插成了 {mid.in_air!r}（必须是 bool）"
    assert isinstance(mid.mission_current, int)
    assert mid.mission_current == 2, f"mission_current 内插成了 {mid.mission_current!r}"
    assert mid.mission_total == 4
    # 连续量照常内插（别把修复做过头）
    assert mid.roll_deg == pytest.approx(15.0)

    # 外推取右端原值
    after = broker.get_snapshot_at(3.0)
    assert after.in_air is False


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
