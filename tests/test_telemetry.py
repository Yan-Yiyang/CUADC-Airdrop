"""MAVSDK 基础设施测试。

不需要真实飞控：遥测部分用最小替身对象构造数据，MAVSDK 部分只验证启停、错误路径
与"会话结束必须释放 mavsdk_server"这类关键约定。
"""

from __future__ import annotations

import math
import queue
import threading
import time

import pytest

from airdrop import Command, ExternalBridge, MavsdkThread, TelemetryBroker


class Fake:
    """最小 MAVSDK 对象替身。"""


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


def test_broker_snapshot_interpolation_subscribe(
    broker: TelemetryBroker, received: list
) -> None:
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
        snap.quaternion_w**2 + snap.quaternion_x**2
        + snap.quaternion_y**2 + snap.quaternion_z**2
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
        broker.update_global_position(
            make_position(40.0 + i * 0.001, 8.0, 500.0 + i, 10.0)
        )
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
    near_first = broker.get_snapshot_at(
        broker._history[0].timestamp - 10.0, mode="nearest"
    )
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
    import airdrop.mavsdk_thread as mt

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


def test_external_bridge_snapshot_and_commands() -> None:
    """跨进程桥接：共享 dict 可读，指令经队列往返（未连接 → 干净失败）。"""
    broker = TelemetryBroker()
    mav = MavsdkThread(broker=broker, reconnect=False)
    bridge = ExternalBridge(mav, broker=broker, publish_hz=20.0)
    try:
        bridge.start()
    except PermissionError:
        # 受限沙箱里创建不了 Manager 的命名管道，这不是代码问题
        mav.stop()
        pytest.skip("当前环境无法创建 multiprocessing.Manager（受限沙箱）")

    try:
        assert "latitude_deg" in dict(bridge.snapshot)
        cmd = Command("arm")
        bridge.command_queue.put(cmd.as_dict())

        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            try:
                raw = bridge.result_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if raw.get("id") == cmd.id:
                assert raw.get("success") is False
                assert "尚未连接" in (raw.get("error") or "")
                break
        else:
            pytest.fail("10s 内未收到指令结果")
    finally:
        bridge.stop()
        mav.stop()
