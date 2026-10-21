"""SITL 侦查精度测试。

分两部分：

* :func:`test_sitl_recon_accuracy`——**端到端**，标 ``sitl``、要 WSL 里的 SITL 已经在跑，
  默认跳过::

      ./.venv/Scripts/python.exe -m pytest tests/test_sitl_recon.py -m sitl -v

  前置（本测试只**驱动**，不负责启动飞控/世界/图传）::

      bash vendor/airdrop-sim/airdrop_sim/assets/run_sitl.sh r2          # WSL 里起 PX4 SITL + Gazebo + 图传

  断言（报告写入用例级临时目录的 ``report.json``）：

  * 状态机走到 ``DONE``——标准流程：起飞 → 侦查监视 → 空中出目标 → 飞掠投放 → 降落；
  * 三个编号 94/12/56 全部识别，任务结果按"中位数"选中 **56**；
  * 最终误差 < 2 m（原始口径与"自标定延时"口径各查一遍，留一验证的结果也要 < 2 m）。

  SITL 没在跑就 ``skip``（不报失败）——避免"忘了起仿真"被当成代码坏了。
  全流程分钟级，所以这条用例把超时放宽到 30 min，**而不是关掉兜底**（``timeout(0)``）：
  探活那一步曾经因为 ``MavsdkThread.stop()`` 永久阻塞而永远回不来，把整轮全量测试挂死。

* 报告阶段的原点解析与 lag 扫描——**纯离线**，默认跑，盯的是"任务跑完了但报告崩在
  最后一步"那条缺陷（见 :func:`test_report_origin_survives_broker_reset`）。
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import json
import math
import socket
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("airdrop_sim", reason="SITL 集成需要仿真子模块")

from airdrop import Config, TelemetryBroker, TelemetrySnapshot, load_broker_from_log

REPO_ROOT = Path(__file__).resolve().parents[1]


def _sitl_is_up(timeout: float = 8.0) -> bool:
    """SITL 在跑吗：直接听 14540 上有没有数据，**不建 MAVSDK 会话**。

    刻意不用 ``MavsdkThread.connect()`` 探活：那要拉起 mavsdk_server 与 gRPC 通道，而
    "连不上再收尾"这条路径实测会卡死整个进程（gRPC poller 线程报
    ``Event loop is closed``）——探活只需要知道
    "14540 上有没有人在发心跳"。
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.bind(("0.0.0.0", 14540))
    except OSError:
        # 端口被占：已经有别的会话/地面站在收，或者本来就没有 SITL——都不足以判定"可以跑"
        return False
    try:
        sock.settimeout(timeout)
        try:
            sock.recv(2048)
        except TimeoutError:
            return False
        return True
    finally:
        sock.close()


@pytest.mark.sitl
@pytest.mark.timeout(1800)  # 分钟级全流程：放宽到 30 min，但**不**像 timeout(0) 那样彻底关掉兜底
def test_sitl_recon_accuracy(tmp_path: Path) -> None:
    if not _sitl_is_up():
        pytest.skip("SITL 没在跑（先 bash vendor/airdrop-sim/airdrop_sim/assets/run_sitl.sh r2）")

    from airdrop.runtime import sitl

    exit_code = sitl.main(work_dir=str(tmp_path))
    # 先断言退出码：否则退出码非 0 时下面读 report 会先抛 FileNotFoundError /
    # JSONDecodeError，把真正的失败原因盖掉
    assert exit_code in (0, 7), f"退出码 {exit_code}"
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))

    # ⚠ 键名是 "mission"（运行时分析模块 写的就是它）；曾经这里读的是
    # "runner_stats"——那个名字只是 _RunOutcome 的字段名，report.json 里根本没有，
    # 于是 runner 恒为 {}，这条唯一的端到端验收用例**永远失败**（它带 sitl 标记
    # 默认跳过，所以从没被发现）。
    runner = report.get("mission") or {}
    assert str(runner.get("state", "")).endswith("DONE"), f"任务没走完：{runner}"
    codes = {int(cluster["code"]) for cluster in report["clusters"] if cluster["code"] is not None}
    assert codes >= {94, 12, 56}, f"三个靶标没全识别：{codes}"
    selected = report["selected"]
    assert selected is not None and selected["code"] == 56, f"任务结果选错：{selected}"
    assert selected["error_m"] < 2.0, f"误差超 2 m：{selected}"
    calibration = report.get("lag_calibration") or {}
    holdout = calibration.get("holdout_error_m")
    assert holdout is None or holdout < 2.0, f"留一验证超 2 m：{calibration}"


def test_zero_test_params_keeps_original_value_when_write_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """置零失败时，账本也必须保留已经读到的参数原值。"""
    from airdrop.runtime import sitl_operations

    ledger: dict[str, int] = {}
    monkeypatch.setattr(sitl_operations, "read_param", lambda _thread, _name: 7)

    def fail_write(_thread: Any, _name: str, _value: int) -> None:
        raise TimeoutError("写入结果未知")

    monkeypatch.setattr(sitl_operations, "_write_param", fail_write)
    with pytest.raises(TimeoutError, match="未知"):
        sitl_operations.zero_test_params(object(), ("NAV_DLL_ACT",), ledger=ledger)

    assert ledger == {"NAV_DLL_ACT": 7}


def test_zero_test_params_returns_original_values_for_legacy_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """不传账本时保留原有的 dict 返回接口；传入时返回完整账本。"""
    from airdrop.runtime import sitl_operations

    monkeypatch.setattr(
        sitl_operations, "read_param", lambda _thread, name: 2 if name == "A" else 5
    )
    monkeypatch.setattr(sitl_operations, "_write_param", lambda *_args, **_kwargs: None)

    assert sitl_operations.zero_test_params(object(), ("A", "B")) == {"A": 2, "B": 5}
    ledger = {"A": 8}
    assert sitl_operations.zero_test_params(object(), ("A", "B"), ledger=ledger) is ledger
    assert ledger == {"A": 8, "B": 5}


def test_zero_test_params_read_failure_does_not_record_or_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """读取失败时没有可恢复的原值，因此不能写参数或污染账本。"""
    from airdrop.runtime import sitl_operations

    ledger: dict[str, int] = {}
    writes: list[tuple[str, int]] = []

    def fail_read(_thread: Any, name: str) -> int:
        if name == "NAV_RCL_ACT":
            raise RuntimeError("参数读取失败")
        return 4

    monkeypatch.setattr(sitl_operations, "read_param", fail_read)
    monkeypatch.setattr(
        sitl_operations, "_write_param", lambda _thread, name, value: writes.append((name, value))
    )
    with pytest.raises(RuntimeError, match="读取失败"):
        sitl_operations.zero_test_params(object(), ("NAV_DLL_ACT", "NAV_RCL_ACT"), ledger=ledger)

    assert ledger == {"NAV_DLL_ACT": 4}
    assert writes == [("NAV_DLL_ACT", 0)]


def test_restore_params_uses_exact_values_and_continues_after_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """恢复写失败应继续其它参数，NAV_DLL_ACT 也按账本原值恢复。"""
    from airdrop.runtime import sitl_operations

    calls: list[tuple[str, int, bool]] = []

    def write(_thread: Any, name: str, value: int, *, recovery: bool = False) -> None:
        calls.append((name, value, recovery))
        if name == "BROKEN":
            raise RuntimeError("模拟恢复失败")

    monkeypatch.setattr(sitl_operations, "_write_param", write)
    with pytest.raises(RuntimeError, match="BROKEN"):
        sitl_operations.restore_params(object(), {"NAV_DLL_ACT": 5, "BROKEN": 9, "NAV_RCL_ACT": 3})

    assert calls == [
        ("NAV_DLL_ACT", 5, True),
        ("BROKEN", 9, True),
        ("NAV_RCL_ACT", 3, True),
    ]


def test_timed_out_parameter_write_finishes_before_recovery_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """写超时后恢复等待协程收尾，不能让迟到的置零覆盖恢复值。"""
    from airdrop.runtime import sitl_operations
    from airdrop.telemetry._operations import OperationGate

    class LoopThread:
        def __init__(self) -> None:
            self.loop = asyncio.new_event_loop()
            self.errors: list[dict[str, Any]] = []
            self.loop.set_exception_handler(lambda _loop, context: self.errors.append(context))
            self.thread = threading.Thread(target=self.loop.run_forever, daemon=True)
            self.thread.start()

        def submit(self, coro: Any) -> concurrent.futures.Future[Any]:
            future = asyncio.run_coroutine_threadsafe(coro, self.loop)
            future.add_done_callback(self._consume)
            return future

        @staticmethod
        def _consume(future: concurrent.futures.Future[Any]) -> None:
            with contextlib.suppress(Exception):
                future.result()

        def close(self) -> None:
            async def cancel_pending() -> None:
                current = asyncio.current_task()
                tasks = [
                    task
                    for task in asyncio.all_tasks(self.loop)
                    if task is not current and not task.done()
                ]
                for task in tasks:
                    task.cancel()
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)

            asyncio.run_coroutine_threadsafe(cancel_pending(), self.loop).result(timeout=1)
            self.loop.call_soon_threadsafe(self.loop.stop)
            self.thread.join(timeout=1)
            assert not self.thread.is_alive(), "测试事件循环未能退出"
            self.loop.close()
            assert not self.errors, f"事件循环出现未处理异常：{self.errors!r}"

    class Param:
        def __init__(self) -> None:
            self.value = 6
            self.entered = threading.Event()
            self.finish_cancelled_write = asyncio.Event()
            self.cancelled_write_finished = threading.Event()
            self.writes: list[int] = []

        async def get_param_int(self, _name: str) -> int:
            return self.value

        async def set_param_int(self, _name: str, value: int) -> None:
            self.writes.append(value)
            if value == 0:
                self.entered.set()
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    await self.finish_cancelled_write.wait()
                    self.value = value
                    self.cancelled_write_finished.set()
                    raise
            self.value = value

    class Thread(LoopThread):
        def __init__(self) -> None:
            super().__init__()
            self._gate = OperationGate()
            self.param = Param()
            self.drone = type("Drone", (), {"param": self.param})()

        def require_drone(self) -> Any:
            return self.drone

    thread = Thread()
    monkeypatch.setattr(sitl_operations, "PARAM_RPC_TIMEOUT_S", 0.05)
    outcome: concurrent.futures.Future[BaseException | None] = concurrent.futures.Future()

    def zero() -> None:
        try:
            sitl_operations._write_param(thread, "NAV_DLL_ACT", 0)
        except BaseException as exc:  # noqa: BLE001 - 记录预期的超时结果。
            outcome.set_result(exc)
        else:
            outcome.set_result(None)

    caller = threading.Thread(target=zero)
    caller.start()
    try:
        assert thread.param.entered.wait(1)
        error = outcome.result(timeout=1)
        assert isinstance(error, TimeoutError)

        with pytest.raises(RuntimeError, match="NAV_DLL_ACT"):
            sitl_operations.restore_params(thread, {"NAV_DLL_ACT": 6})
        assert thread.param.writes == [0], "前序写入未收尾时不能提交恢复写"

        thread.loop.call_soon_threadsafe(thread.param.finish_cancelled_write.set)
        assert thread.param.cancelled_write_finished.wait(1)
        sitl_operations.restore_params(thread, {"NAV_DLL_ACT": 6})
        assert thread.param.writes == [0, 6]
        assert thread.param.value == 6
    finally:
        thread.loop.call_soon_threadsafe(thread.param.finish_cancelled_write.set)
        caller.join(timeout=1)
        assert not caller.is_alive(), "参数写入调用线程未退出"
        thread.close()


# ----------------------------------------------------------------------
# 报告阶段的原点解析（**纯离线**，默认跑）
# ----------------------------------------------------------------------
@pytest.fixture
def workdir(tmp_path: Path) -> Path:
    """为本用例提供独立素材目录。"""
    return tmp_path


def _flight_with_origin(flight_dir: Path, *, count: int = 5) -> None:
    """手写一个最小飞行目录，快照带 EKF 原点（记录器就是这么写的）。

    ⚠ ``FlightLog.open`` 会校验**整份**记录，缺 ``frames_index.jsonl`` 直接抛
    ``FlightLogError``——所以这里连帧索引一起写出来（内容对原点解析无影响，只是让
    记录合法）。
    """
    flight_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        json.dumps(
            TelemetrySnapshot(
                timestamp=1000.0 + k * 0.1,
                north_m=float(k),
                east_m=float(k),
                down_m=-20.0,
                origin_longitude_deg=8.5461637,
                origin_latitude_deg=47.3979710,
                origin_altitude_m=0.177,
            ).as_dict()
        )
        for k in range(count)
    ]
    (flight_dir / "telemetry.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (flight_dir / "frames_index.jsonl").write_text(
        json.dumps(
            {
                "index": 1,
                "filename": "000001.jpg",
                "capture_timestamp": 1000.0,
                "received_timestamp": 1000.49,
                "lag": 0.49,
                "extrapolated": False,
                "offset": 0.0,
                "bytes": 1,
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _outcome_stub(**kwargs: Any) -> Any:
    """只塞 ``_analysis_broker`` / ``_origin_lla`` 真会碰到的三个字段。

    这两个函数只读 ``flight_dir`` / ``broker`` / ``origin``，而 ``_RunOutcome`` 其余
    字段（``perception`` 之类）要真对象才建得出来——为一个纯函数测试去伪造整套感知
    统计件既没意义也容易和真结构走偏。
    """
    fields: dict[str, Any] = {"flight_dir": None, "broker": None, "origin": None}
    fields.update(kwargs)
    return SimpleNamespace(**fields)


def test_report_origin_survives_broker_reset(workdir: Path) -> None:
    """会话结束（``broker.reset()``）之后，报告仍要拿得到 EKF 原点。

    回归的是一条**只在跑完才暴露**的缺陷：``_run_session`` 的 finally 会
    :meth:`TelemetryBroker.reset`，把快照换成全新的空快照、原点字段全成 ``None``；
    报告把它当 ``ref`` 传进 ``wgs84_to_ned`` 就是
    ``TypeError: float() argument must be ... not 'NoneType'``——任务好好走完 DONE，
    ``report.json`` 却从未写出来、退出码 1（2026-09-30 实测，白等 8 分钟）。

    所以原点必须从**飞行记录重建**的 broker 里取（与 ``lag_sweep`` 同一个来源），
    实在没有才退回 ``MissionRunner.origin``。
    """
    from airdrop.runtime import sitl_analysis

    flight = workdir / "flight"
    _flight_with_origin(flight)

    # 会话结束后的活 broker：快照已被 reset 清空
    live = TelemetryBroker(history_interval=0.0)
    live.publish(
        TelemetrySnapshot(
            timestamp=1.0,
            origin_longitude_deg=1.0,
            origin_latitude_deg=2.0,
            origin_altitude_m=3.0,
        )
    )
    live.reset()
    assert live.get_snapshot().origin_longitude_deg is None, "前提不成立：reset 没清空原点"

    outcome = _outcome_stub(flight_dir=flight, broker=live)
    analysis = sitl_analysis._analysis_broker(outcome)
    origin_lla = sitl_analysis._origin_lla(outcome, analysis)

    assert origin_lla == pytest.approx((8.5461637, 47.3979710, 0.177))
    # 报告真正用到它的地方：把天井真值换算到本地 NED（原来就是在这里炸的）
    north, east = sitl_analysis._well_truth_ned(56, origin_lla)
    assert abs(north) < 1e4 and abs(east) < 1e4


def test_report_origin_falls_back_to_runner_then_fails_loudly(workdir: Path) -> None:
    """没有飞行记录时退回 runner 记的原点；两处都没有就**显式报错**。

    退化到 runner 那份（``--no-record`` 或记录读不出来），再退化就该说人话，而不是
    让 ``None`` 一路传到 georef 深处变成看不懂的 ``TypeError``。
    """
    from airdrop.runtime import sitl_analysis

    live = TelemetryBroker(history_interval=0.0)
    live.reset()
    runner_origin = (8.5, 47.4, 0.2)

    outcome = _outcome_stub(flight_dir=None, broker=live, origin=runner_origin)
    assert sitl_analysis._origin_lla(outcome, sitl_analysis._analysis_broker(outcome)) == (
        pytest.approx(runner_origin)
    )

    outcome = _outcome_stub(flight_dir=None, broker=live, origin=None)
    with pytest.raises(RuntimeError, match="原点"):
        sitl_analysis._origin_lla(outcome, sitl_analysis._analysis_broker(outcome))


# ----------------------------------------------------------------------
# 自动生成的侦查航线：**固定翼几何**（纯离线，默认跑）
# ----------------------------------------------------------------------
def _heading(a: tuple[float, float], b: tuple[float, float]) -> float:
    """ENU 平面上 a→b 的航向（度，0=北、90=东）。"""
    return math.degrees(math.atan2(b[0] - a[0], b[1] - a[1])) % 360.0


def _turn_deg(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float:
    """在 b 处从 a→b 转到 b→c 的转角（度，取最小角）。"""
    delta = abs((_heading(a, b) - _heading(b, c) + 180.0) % 360.0 - 180.0)
    return delta


def _dense_route() -> tuple[tuple[float, float], ...]:
    from airdrop.runtime import sitl_geometry

    flat: list[tuple[float, float]] = []
    for start, end in sitl_geometry.RECON_LEGS_ENU:
        for point in (start, end):
            if not flat or flat[-1] != point:
                flat.append(point)
    return sitl_geometry.round_corners(flat, sitl_geometry.TURN_RADIUS_M)


def test_turn_radius_follows_speed_and_bank() -> None:
    """``R = v²/(g·tanφ)``：速度翻倍半径变四倍，倾角越大半径越小。

    这是航线设计的输入，所以公式本身要被钉住——改错了整条线的可行性判断就全错。
    """
    from airdrop.runtime import sitl_geometry

    base = sitl_geometry.turn_radius_m(15.0, 45.0)
    assert base == pytest.approx(15.0**2 / 9.80665, rel=1e-3)
    assert sitl_geometry.turn_radius_m(30.0, 45.0) == pytest.approx(4 * base, rel=1e-6)
    assert sitl_geometry.turn_radius_m(15.0, 30.0) > base, "倾角变小 ⇒ 半径变大"
    assert sitl_geometry.turn_radius_m(15.0, 60.0) < base, "倾角变大 ⇒ 半径变小"
    with pytest.raises(ValueError, match="倾角"):
        sitl_geometry.turn_radius_m(15.0, 0.0)


def test_recon_route_has_no_in_place_reversal() -> None:
    """侦查航线不许出现"原地 180° 掉头"。

    原来的航线是 ``(265,-12) → (150,-12)`` 紧接 ``(150,-12) → (265,-12)``：到达时朝东、
    下一点正西，等于让飞机在原地翻个面。固定翼做不到——它得先拐个弯，而两条平行线
    只隔 19 m、掉头环半径只有 9.5 m，20 m/s 下要 77° 倾角。PX4 只能"尽量绕"，
    实际航迹与航线对不上，而坐标解算完全依赖飞机此刻在哪。

    回归口径：圆弧化之后**相邻航点的航向差**不超过一个步进角（45°）。这条同时挡住
    原地掉头和"某一段忽然要转 90°+"——两者对固定翼都是飞不出来/航迹对不上的。
    """
    dense = _dense_route()
    worst = max(_turn_deg(dense[i - 1], dense[i], dense[i + 1]) for i in range(1, len(dense) - 1))
    assert worst <= 45.0 + 1e-6, f"最大转角 {worst:.1f}°，超过一个步进角"
    assert worst < 90.0, "绝不允许接近 180° 的掉头"


def test_recon_route_turns_lie_on_the_declared_radius() -> None:
    """拐角切出的圆弧必须真的落在声明的半径上，而不是"大概转一下"。

    做法：取一段至少 3 个弧点的连续弯道，拟合外接圆，断言半径 ≈ 声明值。
    直线段与弧的衔接处不参与拟合（两侧长度不等，那个公式会失效）。
    """
    from airdrop.runtime import sitl_geometry

    dense = _dense_route()
    radius = sitl_geometry.TURN_RADIUS_M

    def circumradius(a, b, c) -> float | None:
        side_a, side_b, side_c = math.dist(b, c), math.dist(a, c), math.dist(a, b)
        area = abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])) / 2.0
        if area < 1e-6:
            return None
        return side_a * side_b * side_c / (4.0 * area)

    fitted = []
    for i in range(1, len(dense) - 2):
        turn = _turn_deg(dense[i - 1], dense[i], dense[i + 1])
        next_turn = _turn_deg(dense[i], dense[i + 1], dense[i + 2])
        if turn < 5.0 or next_turn < 5.0:
            continue  # 直线或直线/弧衔接，跳过
        value = circumradius(dense[i - 1], dense[i], dense[i + 1])
        if value is not None and value < 3 * radius:
            fitted.append(value)
    assert fitted, "没找到任何圆弧段——拐角没被切圆弧"
    assert sum(fitted) / len(fitted) == pytest.approx(radius, rel=0.35)


def test_reverse_diagnostic_refuses_mismatched_bearings() -> None:
    """两遍**不同轴**时不给延时估计——只报航迹角与原因。

    回归的是一条会误导人的输出：``Δeast/(2v)`` 只在两遍沿**同一条轴反向**通过时成立。
    侦查航线的反向观测来自西南向对角线（差 40~50°），只按东向速度符号分组会把它算进
    "-east" 组，于是 Δeast 主要反映**切向差**。实测（架次 某架次）就这样算出
    ``implied_lag = -0.349 s``——比真实延时大一个数量级且**符号相反**。

    口径：这种输入下**不给数**，只给"两遍不同轴"和两个航迹角。
    """
    from airdrop.runtime import sitl_analysis

    # 西南向对角线（227°）与东向扫掠线（90°）差 137°，不是反向 → 拦住
    diagonal = sitl_analysis._axis_check_note({"+east": 90.0, "-east": 227.0})
    assert diagonal is not None and "同一条轴" in diagonal

    # 真正的同轴反向：正东 90° / 正西 270°（相差正好 180°）要放行
    assert sitl_analysis._axis_check_note({"+east": 90.0, "-east": 270.0}) is None
    # 转弯中的短暂偏航在容差内
    assert sitl_analysis._axis_check_note({"+east": 100.0, "-east": 285.0}) is None
    # 缺一组要显式说缺
    assert "缺少" in (sitl_analysis._axis_check_note({"+east": 90.0}) or "")


def test_circular_mean_deg_does_not_flip_at_180() -> None:
    """角度的圆周均值：179°/181° 的均值是 180°，不是 0°。"""
    from airdrop.runtime import sitl_analysis

    assert sitl_analysis._circular_mean_deg([179.0, 181.0]) == pytest.approx(180.0, abs=0.1)
    assert sitl_analysis._circular_mean_deg([89.0, 91.0]) == pytest.approx(90.0, abs=0.1)
    assert sitl_analysis._circular_mean_deg([350.0, 10.0]) == pytest.approx(0.0, abs=0.1)


@pytest.mark.realdata
def test_reverse_diagnostic_on_a_recorded_flight(tmp_path: Path) -> None:
    """真跑一遍：拿架次 某架次 的记录验同轴校验真的会拦住那条误导输出。"""
    from airdrop.runtime import sitl_analysis, sitl_geometry

    flight = REPO_ROOT / "flights/<架次>"
    if not (flight / "telemetry.jsonl").is_file():
        pytest.skip(f"没有架次记录：{flight}")
    detail = sitl_analysis._load_detail(flight / "detections.jsonl")
    if not detail:
        pytest.skip("该架次没有检测明细")
    broker, pacer = load_broker_from_log(flight, history_maxlen=0)
    pacer.publish_until(float("inf"))

    codes = {row.get("code") for row in detail if row.get("code") is not None}
    assert codes, "该架次没有读出任何编号，换一个架次再验"
    result = sitl_analysis._reverse_diagnostic(
        detail,
        broker,
        calib_file=str(tmp_path / "camera_calib_sim.json"),
        code=min(code for code in codes if code is not None),
        lag_s=sitl_geometry.TELEMETRY_LAG_S,
    )
    if "implied_lag_s" in result:
        # 给了数就必须是通过同轴校验的：两个航迹角都得贴正东/正西
        assert set(result["bearings_deg"]) == {"+east", "-east"}
    else:
        # 没给数就必须说清为什么，而且不能是静默的 {}
        assert result and "note" in result


def _det(telemetry: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    """造一条检测记录（只填诊断要用的字段）。"""
    row = {
        "frame_index": kwargs.pop("frame_index", 1),
        "capture_timestamp": kwargs.pop("capture_timestamp", 1000.0),
        "pixel": kwargs.pop("pixel", [640.0, 360.0]),
        "confidence": 0.9,
        "code": kwargs.pop("code", 56),
        "ocr_confidence": 1.0,
        "telemetry": telemetry,
    }
    row.update(kwargs)
    return row


def _snap(
    *,
    yaw: float,
    course: float | None = None,
    speed: float = 15.0,
    pose: tuple[float, float, float] = (0.0, 0.0, -20.0),
    roll: float = 0.0,
    pitch: float = 0.0,
) -> dict[str, Any]:
    """造一份遥测：姿态（``roll``/``pitch``/``yaw``）、速度（航迹角 = ``course``）、位置 ``pose``。"""
    course = yaw if course is None else course
    return {
        "north_m": pose[0],
        "east_m": pose[1],
        "down_m": pose[2],
        "roll_deg": roll,
        "pitch_deg": pitch,
        "yaw_deg": yaw,
        "vx_m_s": speed * math.cos(math.radians(course)),
        "vy_m_s": speed * math.sin(math.radians(course)),
    }


def test_course_deg_needs_speed_and_both_components() -> None:
    """速度太小或缺字段时**不给**航迹角——停机坪上方向是纯噪声。"""
    from airdrop.runtime import sitl_analysis

    assert sitl_analysis._course_deg(_snap(yaw=90.0)) == pytest.approx(90.0)
    assert sitl_analysis._course_deg(_snap(yaw=0.0, speed=0.4)) is None, "速度 0.4 m/s 不该给方向"
    assert sitl_analysis._course_deg({"vx_m_s": 5.0}) is None, "缺 vy 就是缺一半方向"
    assert sitl_analysis._course_deg({}) is None
    assert sitl_analysis._course_deg(None) is None


def test_angle_diff_deg_folds_across_180() -> None:
    """角度差必须折叠到 0~180：正东 90° 与正西 270° 的差是 180° 而不是 −0°。"""
    from airdrop.runtime import sitl_analysis

    assert sitl_analysis._angle_diff_deg(90.0, 270.0) == pytest.approx(180.0)
    assert sitl_analysis._angle_diff_deg(270.0, 90.0) == pytest.approx(180.0)
    assert sitl_analysis._angle_diff_deg(350.0, 10.0) == pytest.approx(20.0)
    assert sitl_analysis._angle_diff_deg(-90.0, 90.0) == pytest.approx(180.0)
    assert sitl_analysis._angle_diff_deg(5.0, 5.0) == pytest.approx(0.0)


def test_heading_consistency_flags_the_post_takeoff_transient() -> None:
    """yaw 偏 25~35° 的那批要被抓出来，且**不剔点**（n 仍等于总观测数）。"""
    from airdrop.runtime import sitl_analysis

    dets = [_det(_snap(yaw=112.0, course=90.0)) for _ in range(4)]  # 起飞后瞬态：偏 22°
    dets += [_det(_snap(yaw=95.0, course=90.0)) for _ in range(6)]  # 正常段：偏 5°
    dets += [_det(_snap(yaw=90.0, course=90.0, speed=0.5))]  # 太慢，不参与
    dets += [{"telemetry": {}}]  # 缺遥测，不能炸也不能计入

    stats = sitl_analysis._heading_consistency(dets)
    assert stats["n"] == 10, "只统计有速度且速度够的观测"
    assert stats["over_tolerance"] == 4
    assert stats["over_fraction"] == pytest.approx(0.4)
    assert stats["max_deg"] == pytest.approx(22.0, abs=0.1)
    assert stats["tolerance_deg"] == 10.0
    # 超阈的**照旧留在统计里**——剔了就看不到精度是被姿态吃掉的
    assert stats["n"] == 10


def test_heading_consistency_is_empty_not_wrong_when_nothing_is_usable() -> None:
    """一条都算不出来时只报 n=0，不能拿 0 冒充"偏差 0°"。"""
    from airdrop.runtime import sitl_analysis

    stats = sitl_analysis._heading_consistency([{"telemetry": {}}, {"telemetry": None}])
    assert stats["n"] == 0
    assert "median_deg" not in stats, "没有样本就不该有中位数"


def test_error_stats_reports_median_rms_p90_and_under_2m() -> None:
    from airdrop.runtime import sitl_analysis

    stats = sitl_analysis._error_stats([0.5, 1.0, 1.5, 2.0, 13.0])
    assert stats["n"] == 5
    assert stats["median_m"] == pytest.approx(1.5)
    assert stats["max_m"] == pytest.approx(13.0)
    assert stats["p90_m"] == pytest.approx(13.0)
    assert stats["under_2m_fraction"] == pytest.approx(0.6)
    assert stats["rms_m"] == pytest.approx(math.sqrt((0.25 + 1 + 2.25 + 4 + 169) / 5), abs=1e-3)
    assert sitl_analysis._error_stats([]) == {"n": 0}


def test_observation_errors_split_by_heading_tolerance(workdir: Path) -> None:
    """逐观测误差分两个口径：全量 vs 航向一致子集。

    口径来自实测（6 个架次 114 个带编号观测，只把 yaw 换成航迹角重算）：
    地面误差中位**1.31 → 0.82 m**、均方根**3.72 → 1.09 m**、最大**13.16 → 3.15 m**。
    这里构造两条观测把机制钉住：航向一致的误差应当≈0，航向偏 30° 的那条把误差撑到米级。
    """
    import numpy as np

    from airdrop import load_camera_model
    from airdrop.georef.project import _attitude_matrix
    from airdrop.runtime import sitl_analysis, sitl_geometry

    calib = sitl_geometry.write_sim_calib(workdir / "calib.json", lag_s=0.49)
    camera = load_camera_model(str(calib), fallback_size=(1280, 720))
    origin = sitl_geometry.WORLD_ORIGIN_LLA
    truth = sitl_analysis._well_truth_ned(56, origin)
    # 相机放在**目标西侧 15 m、高 20 m**：视线离天顶约 37°，求交条件良好。
    # ⚠ 别把相机放在世界原点——真值在 E≈200 m 外，那样视线近乎水平、求交病态，
    # 连"零误差"那一条都会算出 4 m 的假误差（实测踩过）。
    pose = (truth[0], truth[1] - 15.0, -20.0)
    euler = (0.0, 0.0, 0.0)

    # 由真值反投影出"应该被检到"的像素；再用它构造检测 ⇒ 解算闭环、误差≈0
    rnb = _attitude_matrix(quaternion=None, euler_deg=euler)
    cam = np.array(pose) + rnb @ camera.t_bc
    direction = camera.r_bc.T @ (rnb.T @ (np.array([truth[0], truth[1], 0.0]) - cam))
    assert direction[2] > 0.5, "真值必须落在相机视锥里，否则这条用例没意义"
    cx, cy = camera.principal_point()
    pixel = [
        cx + camera.focal_px * direction[0] / direction[2],
        cy + camera.focal_px * direction[1] / direction[2],
    ]

    good = _det(_snap(yaw=0.0, course=2.0, pose=pose), pixel=pixel, code=56)
    bad = _det(_snap(yaw=30.0, course=0.0, pose=pose), pixel=[pixel[0] + 300.0, pixel[1]], code=56)
    stats = sitl_analysis._observation_errors([good, bad], origin_lla=origin, calib_file=str(calib))

    assert stats["all"]["n"] == 2
    assert stats["heading_consistent"]["n"] == 1, "航向偏 30° 的那条不进子集"
    assert stats["heading_consistent"]["median_m"] < 0.05, "航向一致的那条误差应当≈0"
    assert stats["all"]["max_m"] > 5.0, "偏 30° yaw 加偏 300px 必须是大误差"
    assert stats["worst"][0]["frame_index"] == bad["frame_index"]
    assert stats["worst"][0]["heading_dev_deg"] == pytest.approx(30.0, abs=0.1)


def test_heading_diagnostic_reads_velocity_from_the_flight_record() -> None:
    """诊断必须只依赖 ``detections.jsonl``（不碰活的 broker），所以速度要落盘。

    回归点：``Detection.as_dict()`` 的遥测子字典里必须有 ``vx_m_s``/``vy_m_s``。
    少了它们，飞行记录重建出来的报告里这条诊断永远是 ``n=0``——离线复算不了。
    """
    from airdrop.perception.models import Detection, PixelBox
    from airdrop.runtime import sitl_analysis

    def detection(vx: float, vy: float, yaw: float) -> dict[str, Any]:
        snapshot = TelemetrySnapshot(
            timestamp=1.0,
            north_m=1.0,
            east_m=2.0,
            down_m=-20.0,
            roll_deg=0.0,
            pitch_deg=-2.0,
            yaw_deg=yaw,
            vx_m_s=vx,
            vy_m_s=vy,
        )
        return Detection(
            frame_index=7,
            capture_timestamp=1.0,
            pixel=(640.0, 360.0),
            box=PixelBox(x1=600.0, y1=320.0, x2=680.0, y2=400.0),
            confidence=0.9,
            code=56,
            telemetry=snapshot,
        ).as_dict()["telemetry"]

    # 速度以东为主 ⇒ 航迹角≈90，与 yaw 一致；反之（以北为主）⇒ 航迹角≈0，与 yaw 差 90°
    eastbound = detection(0.5, 14.0, 90.0)
    assert eastbound["vx_m_s"] == pytest.approx(0.5)
    assert eastbound["vy_m_s"] == pytest.approx(14.0)
    east_course = sitl_analysis._course_deg(eastbound)
    north_course = sitl_analysis._course_deg(detection(14.0, 0.5, 90.0))
    assert east_course is not None and north_course is not None
    assert east_course == pytest.approx(87.95, abs=0.01)
    assert sitl_analysis._angle_diff_deg(east_course, 90.0) == pytest.approx(2.05, abs=0.01)
    assert sitl_analysis._angle_diff_deg(north_course, 90.0) == pytest.approx(87.95, abs=0.01)


def test_eps_retry_widens_only_when_labeled_classes_are_few() -> None:
    """带编号的类不足靶标数时，放宽半径重聚一次；够用就不放宽。

    口径来自实测（同一批点只改 ``eps``）：

    * ``eps=0.75`` → 56 的 3 个点散在 ~20 m 上，聚不起来，只剩 2 个带编号的类；
    * ``eps=1.5`` → **3 个类全齐**，94 → 0.32 m、56 → 1.22 m、12 → 0.75 m，全 < 2 m。

    所以默认 1.5 已经够用（20 m 靶间距的 13 倍余量），重试只是兜底。

    回归要点：① 够用时**不许**放宽（否则报告里的精度是放宽门限得来的，却看不出来）；
    ② 放宽后要多聚出类才采纳，否则保持原结果。
    """
    from dataclasses import replace

    from airdrop import TargetPoint, analyze
    from airdrop.runtime import sitl_analysis

    base = replace(Config().targeting, selection_rule="median")
    assert base.eps_m == pytest.approx(1.5), "默认 eps 应当是 1.5"

    def well(north: float, east: float, code: int) -> TargetPoint:
        return TargetPoint(
            north_m=north,
            east_m=east,
            down_m=0.0,
            capture_timestamp=0.0,
            frame_index=0,
            code=code,
            confidence=0.9,
        )

    # 三个靶标各 3 个点、组内散布 0.3 m（1.5 门限绰绰有余）
    good = tuple(
        well(n, e, code)
        for code, (n, e) in ((94, (6.9, 213.1)), (56, (-10.8, 200.3)), (12, (-14.0, 178.0)))
        for n, e in ((n, e), (n + 0.2, e + 0.2), (n - 0.2, e - 0.2))
    )
    result = analyze(good, base)
    assert sum(1 for c in result.clusters if c.code is not None) == 3

    # 56 的点散开 2.5 m（1.5 装不下、3.0 装得下）→ 应当放宽并多聚出一个类
    loose = tuple(well(n, e, 56) for n, e in ((-13.0, 200.0), (-10.5, 200.5), (-8.0, 201.0)))
    narrow = analyze(loose, base)
    wide = analyze(loose, replace(base, eps_m=sitl_analysis.RETRY_EPS_M))
    assert sum(1 for c in wide.clusters if c.code is not None) > sum(
        1 for c in narrow.clusters if c.code is not None
    ), "3.0 应当比 1.5 多聚出类，否则重试没意义"


def test_eps_retry_reports_the_radius_actually_used() -> None:
    """报告里必须记**实际用的**聚类半径。

    否则"三个靶标误差都很小"会让人以为 1.5 m 就够了，而其实是放宽到 3.0 聚出来的——
    精度是被门限换来的，报告却看不出来。
    """
    from airdrop.runtime import sitl_analysis, sitl_geometry

    assert sitl_analysis.RETRY_EPS_M > 1.5, "重试半径应当比默认大"
    # 靶标间距 20 m：3.0 仍有 6.7 倍余量，不会把相邻靶标并成一个
    min_gap = min(
        abs(a[0] - b[0]) + abs(a[1] - b[1])
        for i, a in enumerate(sitl_geometry.WELLS_A_ENU.values())
        for b in list(sitl_geometry.WELLS_A_ENU.values())[i + 1 :]
    )
    assert min_gap > sitl_analysis.RETRY_EPS_M * 2, (
        f"重试半径 {sitl_analysis.RETRY_EPS_M} 会把靶标并起来"
    )


def test_default_eps_covers_target_spacing() -> None:
    """默认 ``eps`` 相对靶标间距要有足够余量（同一靶标的观测散布远小于靶间距）。"""
    from airdrop.runtime import sitl_geometry

    base = Config().targeting
    min_gap = min(
        abs(a[0] - b[0]) + abs(a[1] - b[1])
        for i, a in enumerate(sitl_geometry.WELLS_A_ENU.values())
        for b in list(sitl_geometry.WELLS_A_ENU.values())[i + 1 :]
    )
    assert min_gap > base.eps_m * 5, f"默认 eps={base.eps_m} 相对靶间距 {min_gap:.1f} m 余量不足"


def test_round_corners_leaves_straight_legs_alone() -> None:
    """共线段不该被塞进弧点（否则直线被采样成一堆冗余航点）。"""
    from airdrop.runtime import sitl_geometry

    line = ((0.0, 0.0), (50.0, 0.0), (100.0, 0.0), (150.0, 0.0))
    assert sitl_geometry.round_corners(line, 20.0) == line


def test_round_corners_rejects_nonsense() -> None:
    from airdrop.runtime import sitl_geometry

    with pytest.raises(ValueError, match="半径"):
        sitl_geometry.round_corners(((0.0, 0.0), (1.0, 1.0)), 0.0)
    with pytest.raises(ValueError, match="步进角"):
        sitl_geometry.round_corners(((0.0, 0.0), (1.0, 1.0)), 10.0, step_deg=120.0)


def test_recon_route_waypoints_are_built() -> None:
    """航线要真的能变成航点（首点带起飞高度，其余侦查高度）。"""
    from airdrop.runtime import sitl_geometry

    route = sitl_geometry.recon_route()
    assert len(route) >= 4
    assert route[0].alt_m == pytest.approx(sitl_geometry.TAKEOFF_ALT_M)
    assert all(wp.alt_m == pytest.approx(sitl_geometry.RECON_ALT_M) for wp in route[1:])


def test_lag_sweep_runs_off_a_recorded_flight(workdir: Path) -> None:
    """lag 扫描要能从记录里查出历史——它本来就是离线分析，不依赖会话还活着。

    顺带钉住"原点不在某个运行时目录也能算"：天井真值是按本地 NED 比的，
    原点一旦拿不到，整节精度分析就作废。
    """
    from airdrop.runtime import sitl_analysis

    flight = workdir / "flight"
    _flight_with_origin(flight, count=50)
    outcome = _outcome_stub(flight_dir=flight, broker=TelemetryBroker(history_interval=0.0))
    analysis = sitl_analysis._analysis_broker(outcome)
    pacer_broker, pacer = load_broker_from_log(flight, history_maxlen=0)
    pacer.publish_until(float("inf"))

    assert analysis is not pacer_broker  # 两个独立来源，只是恰好都从记录来
    assert pacer_broker.get_snapshot_at(1002.0, "interpolate") is not None, "历史没灌进去"
    origin_lla = sitl_analysis._origin_lla(outcome, analysis)
    assert sitl_analysis._well_truth_ned(56, origin_lla) == pytest.approx((-10.83, 200.29), abs=0.5)
