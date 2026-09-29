"""P9 离线单测：状态机、航线规划、任务主循环（假控制器驱动）。

全部离线：控制器是假对象（按 :class:`~airdrop.telemetry.MissionController` 协议
实现），遥测走真实 :class:`~airdrop.telemetry.TelemetryBroker`（传入本地 NED 位置），
时钟与节拍都是注入的假时钟——于是分钟级的任务可以确定性地压成毫秒级跑完，
不存在"睡一会儿看看状态对不对"这种用例。

验收口径来自计划 P9：fake controller 状态机单测（含无目标 → 备用点分支）。
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import math
import time
from collections.abc import Callable, Sequence
from types import SimpleNamespace
from typing import Any

import pytest

from airdrop import (
    MAV_CMD_NAV_LAND,
    MAV_CMD_NAV_TAKEOFF,
    MAV_CMD_NAV_WAYPOINT,
    MAV_FRAME_GLOBAL_RELATIVE_ALT,
    BallisticsConfig,
    BallisticsModel,
    Config,
    ControllerError,
    DroneController,
    DropConfig,
    DropMissionPlan,
    DropRecord,
    DryRunController,
    GripperConfig,
    GroundConfig,
    InvalidTransition,
    MissionConfig,
    MissionItem,
    MissionMonitor,
    MissionRunner,
    MissionState,
    MissionStateMachine,
    NedOrigin,
    OverflyConfig,
    PlanningError,
    Preflight,
    PreflightCheck,
    PreflightConfig,
    PreflightError,
    PreflightLike,
    ReleaseDecision,
    ReleaseJudge,
    RoutesConfig,
    TargetingConfig,
    TargetPoint,
    TelemetryBroker,
    TelemetrySnapshot,
    Waypoint,
    analyze,
    build_drop_mission,
    build_recon_mission,
    llaref_of,
    overfly_positions,
    overfly_waypoints,
    to_raw_item,
    waypoint_to_ned,
)
from airdrop.georef import ned_to_wgs84, wgs84_to_ned
from airdrop.mission import emit_event

# 测试用原点：随便取一个中纬度点，高度 500m
ORIGIN = NedOrigin(lat_deg=47.0, lon_deg=8.0, alt_m=500.0)


# ----------------------------------------------------------------------
# 假对象：时钟 / 控制器 / 判据 / 遥测代理
# ----------------------------------------------------------------------
class FakeClock:
    """可手动推进的假时钟（``sleep`` 即推进，于是 ``run()`` 立刻跑完假时间）。"""

    def __init__(self, start: float | None = None) -> None:
        self.now = time.time() if start is None else float(start)

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> float:
        self.now += float(seconds)
        return self.now

    def sleep(self, seconds: float) -> None:
        self.advance(seconds)


class FakeController:
    """按 :class:`~airdrop.telemetry.MissionController` 协议实现的假控制器。

    记录调用轨迹（``calls``）与上传的任务（``uploads``）；``fail_on`` 里的命令一律
    抛 :class:`ControllerError`（验证失败路径）。``start_mission`` 把 ``finished``
    复位——与真飞控一致：任务一旦启动，``is_mission_finished()`` 先变成 False。
    """

    def __init__(
        self,
        *,
        origin: NedOrigin | None = ORIGIN,
        release_ok: bool = True,
        sticky_finished: bool = False,
    ) -> None:
        self.calls: list[str] = []
        self.uploads: list[tuple[MissionItem, ...]] = []
        self.origin = origin
        self.release_ok = release_ok
        self.finished = False
        #: 飞控是否在任务模式（启动确认用；``False`` 模拟"启动没生效"）
        self.mission_mode = True
        self.fail_on: set[str] = set()
        #: True 时 ``start_mission`` 不复位 ``finished``——模拟"新任务上传后
        #: 飞控仍回上一次的 True"这种陈旧读数
        self.sticky_finished = sticky_finished

    # -- 测试辅助 ------------------------------------------------------
    def finish_mission(self) -> None:
        """模拟"飞控报任务飞完"。"""
        self.finished = True

    def count(self, name: str) -> int:
        return self.calls.count(name)

    # -- MissionController --------------------------------------------
    def _record(self, name: str) -> None:
        self.calls.append(name)
        if name in self.fail_on:
            raise ControllerError(f"{name} 失败（假控制器）")

    def upload_mission(self, items: Sequence[MissionItem], /) -> int:
        self._record("upload_mission")
        self.uploads.append(tuple(items))
        return len(items)

    def start_mission(self) -> None:
        self._record("start_mission")
        if not self.sticky_finished:
            self.finished = False

    def in_mission_mode(self) -> bool:
        self._record("in_mission_mode")
        return self.mission_mode

    def hold(self) -> None:
        self._record("hold")

    def rtl(self) -> None:
        self._record("rtl")

    def gripper_release(self) -> bool:
        self._record("gripper_release")
        return self.release_ok

    def request_origin(self) -> NedOrigin | None:
        self._record("request_origin")
        return self.origin

    def mission_finished(self) -> bool:
        self._record("mission_finished")
        return self.finished


class FakeJudge:
    """假投放判据：按脚本给结果，并记录每拍收到了什么。"""

    def __init__(self, decisions: Sequence[ReleaseDecision] = ()) -> None:
        self._decisions = list(decisions)
        self.targets: list[tuple[float, float, float]] = []
        self.updates = 0
        self.resets = 0
        self.ground_z: float | None = None
        self.ground_altitude_m: float | None = None
        self.now: float | None = None

    def update(
        self,
        snapshot: TelemetrySnapshot,
        target_ned: Sequence[float],
        *,
        ground_z: float | None = None,
        ground_altitude_m: float | None = None,
        wind_ned: Sequence[float] | None = None,
        now: float | None = None,
    ) -> ReleaseDecision:
        self.updates += 1
        self.targets.append((float(target_ned[0]), float(target_ned[1]), float(target_ned[2])))
        self.ground_z = ground_z
        self.ground_altitude_m = ground_altitude_m
        self.now = now
        if self._decisions:
            return self._decisions.pop(0)
        return ReleaseDecision(False, "waiting", float(now or 0.0))

    def reset(self) -> None:
        self.resets += 1


class FakeBroker:
    """只实现 runner 用到的 ``get_snapshot()``：返回一份可以很旧/无效的快照。"""

    def __init__(
        self,
        clock: FakeClock,
        *,
        age_s: float = 0.0,
        valid: bool = True,
        in_air: bool | None = True,
        altitude_m: float | None = None,
    ) -> None:
        self._clock = clock
        self._age_s = age_s
        self._valid = valid
        self.in_air = in_air
        self.altitude_m = altitude_m

    def get_snapshot(self) -> TelemetrySnapshot:
        snapshot = TelemetrySnapshot(timestamp=self._clock() - self._age_s)
        # 离线测试默认"已经起飞"：WAIT_AIRBORNE 那道门由 test_airborne_* 专门覆盖
        snapshot.in_air = self.in_air
        if self.altitude_m is not None:
            snapshot.relative_altitude_m = float(self.altitude_m)
        if self._valid:
            snapshot.north_m = 0.0
            snapshot.east_m = 0.0
            snapshot.down_m = -100.0
        return snapshot


def _live_broker() -> TelemetryBroker:
    """真实 broker + 一份有效 NED 位置（时间戳是真实墙钟）。"""
    broker = TelemetryBroker()
    broker.update_local_position_velocity(
        SimpleNamespace(north_m=0.0, east_m=0.0, down_m=-100.0),
        SimpleNamespace(north_m_s=15.0, east_m_s=0.0, down_m_s=0.0),
    )
    broker.update_in_air(True)  # 离线用例默认"已经起飞"（等待起飞另有专门用例）
    return broker


def _config(
    *,
    recon: int = 3,
    landing: int = 2,
    backup: bool = True,
    overfly: OverflyConfig | None = None,
    mission: MissionConfig | None = None,
    preflight: PreflightConfig | None = None,
) -> Config:
    recon_route = tuple(
        Waypoint(lat=47.0 + index * 1e-3, lon=8.0, alt_m=60.0) for index in range(recon)
    )
    # 降落段按固定翼的几何要求造：进场点 40m 高、离落点约 330m（下滑斜率 tan≈0.12，
    # 约 6.8°，小于上限 tan(8.1°)），否则 build_drop_mission 的本地预检会（正确地）把它拒掉。
    landing_route: tuple[Waypoint, ...] = ()
    if landing >= 2:
        landing_route = (
            Waypoint(lat=47.0 - 3e-3, lon=8.01, alt_m=40.0),
            Waypoint(lat=47.0 - 6e-3, lon=8.01, alt_m=0.0),
        )
    elif landing == 1:
        landing_route = (Waypoint(lat=47.0 - 3e-3, lon=8.01, alt_m=0.0),)
    return Config(
        routes=RoutesConfig(
            recon_route=recon_route,
            backup_point=Waypoint(lat=47.05, lon=8.05, alt_m=0.0) if backup else None,
            landing_route=landing_route,
        ),
        overfly=overfly or OverflyConfig(heading_deg=90.0, altitude_m=20.0, leg_length_m=200.0),
        mission=mission or MissionConfig(tick_hz=20.0, recon_upload="auto"),
        # 起飞前自检默认全关（离线测试没有 GPU/相机）；预检本身由 test_preflight_* 覆盖
        preflight=preflight
        or PreflightConfig(
            load_detector=False,
            load_ocr=False,
            load_camera=False,
            check_video=False,
        ),
        ground=GroundConfig(ground_point_alt=500.0),
    ).validated()


def _runner(
    clock: FakeClock,
    controller: FakeController,
    *,
    config: Config | None = None,
    broker: Any = None,
    judge: FakeJudge | None = None,
    target_result: Any = None,
    target_busy: Any = None,
    events: list[tuple[str, dict[str, Any]]] | None = None,
    on_drop: Any = None,
    preflight: PreflightLike | None = None,
    sleep: Callable[[float], None] | None = None,
) -> MissionRunner:
    # 默认用跟随假时钟的 broker：任务里假时间会推进几分钟，真实墙钟快照会被
    # 看门狗判成"陈旧"（那是另一组用例专门验证的行为）。
    return MissionRunner(
        config or _config(),
        controller,
        FakeBroker(clock) if broker is None else broker,
        target_result=target_result,
        target_busy=target_busy,
        release_judge=judge,
        on_event=None if events is None else (lambda kind, data: events.append((kind, data))),
        on_drop=on_drop,
        preflight=preflight,
        clock=clock,
        sleep=clock.sleep if sleep is None else sleep,
    )


def _events_of(events: list[tuple[str, dict[str, Any]]], kind: str) -> list[dict[str, Any]]:
    return [data for name, data in events if name == kind]


def _drive(runner: MissionRunner, clock: FakeClock, *, ticks: int, step: float = 0.05) -> None:
    """手动推进若干拍（每拍之后推进假时钟）。"""
    for _ in range(ticks):
        runner.update()
        clock.advance(step)


def _drive_until(
    runner: MissionRunner,
    clock: FakeClock,
    state: MissionState,
    *,
    step: float = 0.05,
    limit: int = 5000,
) -> bool:
    """推进到进入目标状态（或终态）为止；返回是否命中目标状态。

    ⚠ 状态是在某一拍结束时进入的，所以命中之后那一拍还没跑过——
    需要"新状态里跑一拍"的断言要自己再调一次 :func:`_drive`。
    """
    for _ in range(limit):
        if runner.state is state:
            return True
        if runner.state in (MissionState.DONE, MissionState.ABORT):
            return runner.state is state
        runner.update()
        clock.advance(step)
    return runner.state is state


def _to_hold(
    runner: MissionRunner,
    clock: FakeClock,
    controller: FakeController,
    *,
    step: float = 0.05,
    limit: int = 5000,
) -> None:
    """推进到 HOLD_PROCESS：先在侦查段跑两拍（让看门狗看到"任务在跑"）再报飞完。

    这两拍是必需的：新任务刚上传时 ``is_mission_finished()`` 的读数不作数
    （见 :class:`MissionMonitor`），得先观测到一次 ``False``。
    """
    _drive(runner, clock, ticks=2, step=step)
    assert runner.state is MissionState.RECON
    controller.finish_mission()
    _drive_until(runner, clock, MissionState.HOLD_PROCESS, step=step, limit=limit)
    assert runner.state is MissionState.HOLD_PROCESS, runner.history[-1].reason


# ----------------------------------------------------------------------
# 状态机
# ----------------------------------------------------------------------
def test_state_machine_walks_the_planned_route_and_emits_events() -> None:
    events: list[tuple[str, dict[str, Any]]] = []
    machine = MissionStateMachine(on_event=lambda kind, data: events.append((kind, data)))

    # 顺序照正式流程：INIT（遥测/原点）→ PREFLIGHT（自检）→ WAIT_AIRBORNE（等在空中）→ 侦查…
    machine.transition(MissionState.PREFLIGHT, reason="telemetry_ready", now=0.5)
    machine.transition(MissionState.WAIT_AIRBORNE, reason="preflight_ok", now=0.7)
    machine.transition(MissionState.RECON, reason="recon_started", now=1.0)
    machine.transition(MissionState.HOLD_PROCESS, reason="recon_finished", now=2.0)
    machine.transition(MissionState.OVERFLY, reason="drop_mission_target", now=3.0)
    machine.transition(MissionState.LAND, reason="released_predict", now=4.0)
    machine.transition(MissionState.DONE, reason="mission_finished", now=5.0)

    assert [record.to_state for record in machine.history] == [
        MissionState.PREFLIGHT,
        MissionState.WAIT_AIRBORNE,
        MissionState.RECON,
        MissionState.HOLD_PROCESS,
        MissionState.OVERFLY,
        MissionState.LAND,
        MissionState.DONE,
    ]
    assert machine.history[0].from_state is MissionState.INIT
    assert machine.history[5].reason == "released_predict"
    assert machine.history[2].timestamp == 1.0
    assert machine.is_terminal
    assert [data["to_state"] for data in _events_of(events, "state")] == [
        "PREFLIGHT",
        "WAIT_AIRBORNE",
        "RECON",
        "HOLD_PROCESS",
        "OVERFLY",
        "LAND",
        "DONE",
    ]


def test_state_machine_rejects_illegal_transition_without_moving() -> None:
    machine = MissionStateMachine()
    assert not machine.can_transition(MissionState.LAND)
    with pytest.raises(InvalidTransition, match="INIT → LAND"):
        machine.transition(MissionState.LAND, reason="nope")
    assert machine.state is MissionState.INIT
    assert machine.history == []


def test_terminal_states_have_no_outgoing_edges() -> None:
    for state in (MissionState.DONE, MissionState.ABORT):
        machine = MissionStateMachine(state=state)
        assert machine.is_terminal
        with pytest.raises(InvalidTransition):
            machine.transition(MissionState.RECON)


def test_emit_event_swallows_callback_failures() -> None:
    """事件写盘失败不能把状态机带崩（recorder 关闭后 emit 会抛）。"""

    def boom(kind: str, data: dict[str, Any]) -> None:
        raise RuntimeError("EventLog 已关闭")

    emit_event(boom, "state", {"to_state": "RECON"})  # 不该抛
    emit_event(None, "state", {})  # 没有回调也无所谓
    machine = MissionStateMachine(on_event=boom)
    # 同样不该抛：连跳三拍（自检 → 等起飞 → 侦查），每一跳的事件回调都在抛
    machine.transition(MissionState.PREFLIGHT, reason="ok")
    machine.transition(MissionState.WAIT_AIRBORNE, reason="ok")
    machine.transition(MissionState.RECON, reason="ok")
    assert machine.state is MissionState.RECON


def test_runner_confirms_the_mission_actually_started() -> None:
    """正常路径：RECON 第一拍就确认进了 MISSION，并记一条 ``mission_confirmed`` 事件。"""
    clock = FakeClock()
    controller = FakeController()
    events: list[tuple[str, dict[str, Any]]] = []
    runner = _runner(clock, controller, judge=FakeJudge(), events=events)

    _drive(runner, clock, ticks=1)  # INIT → RECON（上传+启动）
    _drive(runner, clock, ticks=1)  # RECON 第一拍：确认

    assert runner.state is MissionState.RECON
    assert _events_of(events, "mission_confirmed") == [{"mode": "MISSION"}]


def test_runner_aborts_when_the_mission_never_enters_mission_mode() -> None:
    """``start_mission`` 回成功 ≠ 真的在飞：确认窗口内没进 MISSION 就显式失败。

    这是 2026-09 SITL 演练暴露的易错点：飞控拒绝了模式切换却仍然回 ACK，飞机继续盘旋，
    状态机等到 recon_max_s（600s）才失败。现在有一个短的确认窗口。
    """
    clock = FakeClock()
    controller = FakeController()
    controller.mission_mode = False
    events: list[tuple[str, dict[str, Any]]] = []
    runner = _runner(clock, controller, judge=FakeJudge(), events=events)

    _drive(runner, clock, ticks=1)
    assert runner.state is MissionState.RECON

    for _ in range(80):
        runner.update()
        clock.advance(1.0)
        if runner.state is MissionState.ABORT:
            break

    assert runner.state is MissionState.ABORT
    assert runner.history[-1].reason == "mission_not_started"
    assert runner.stats.aborts == 1
    assert _events_of(events, "mission_confirmed") == []


def test_runner_without_a_flight_mode_stream_skips_the_confirmation() -> None:
    """模式流不可用时退化为不确认（只告警），不能因为取不到模式就把任务判失败。"""
    clock = FakeClock()
    controller = FakeController()
    controller.fail_on = {"in_mission_mode"}
    events: list[tuple[str, dict[str, Any]]] = []
    runner = _runner(clock, controller, judge=FakeJudge(), events=events)

    _drive(runner, clock, ticks=1)
    assert runner.state is MissionState.RECON
    for _ in range(10):
        runner.update()
        clock.advance(1.0)
    assert runner.state is MissionState.RECON, "取不到模式就继续等（由状态超时兜底）"

    controller.finish_mission()
    _drive_until(runner, clock, MissionState.HOLD_PROCESS)
    assert runner.state is MissionState.HOLD_PROCESS


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# 起飞前自检（PREFLIGHT）与"等起飞"（WAIT_AIRBORNE）
# ----------------------------------------------------------------------
class FakePreflight:
    """按 :class:`~airdrop.preflight.PreflightLike` 协议实现的假预检。"""

    def __init__(self, *, fail: str | None = None, checks=()) -> None:
        self.calls = 0
        self.fail = fail
        self._checks = tuple(checks)

    def run(self):
        self.calls += 1
        if self.fail is not None:
            raise PreflightError(f"{self.fail}: 假预检失败")
        if self._checks:
            return self._checks
        return (PreflightCheck("detector", True, "fake"),)


class FakeVideoSource:
    """假图传源：只提供预检会读的两样东西（``stats.frames`` 与 ``latest()``）。"""

    def __init__(self, *, frames: int = 12, size: tuple[int, int] | None = (1280, 720)) -> None:
        self.stats = SimpleNamespace(frames=frames)
        self._size = size

    def latest(self):
        if self._size is None:
            return None
        width, height = self._size
        return SimpleNamespace(width=width, height=height)


def _preflight(
    config: PreflightConfig,
    *,
    loaders: dict[str, Any] | None = None,
    source: FakeVideoSource | None = None,
    camera_model: Any = None,
    events: list[tuple[str, dict[str, Any]]] | None = None,
    clock: FakeClock | None = None,
    sleep: Callable[[float], None] | None = None,
) -> Preflight:
    """按正式入口的接法装配一个真 :class:`~airdrop.Preflight`（时钟/节拍可注入）。"""
    clock = clock or FakeClock()
    return Preflight(
        config,
        model_loaders=loaders,
        video_source=source,
        camera_model=camera_model,
        on_event=None if events is None else (lambda kind, data: events.append((kind, data))),
        clock=clock,
        sleep=clock.sleep if sleep is None else sleep,
    )


def test_real_preflight_checks_models_and_video_and_marks_disabled_items() -> None:
    """真预检：四项顺序（detector → ocr → camera → video），关掉的项记 ``ok=None``。

    关掉 ≠ 通过：事件里那个 ``null`` 就是"这一项没把关"，正式任务起飞前要核对。
    """
    events: list[tuple[str, dict[str, Any]]] = []
    config = PreflightConfig(
        load_detector=True,
        load_ocr=False,  # 关掉
        load_camera=False,  # 关掉
        check_video=True,
        video_probe_s=0.5,
        video_min_frames=5,
    )
    preflight = _preflight(
        config,
        loaders={"detector": lambda: "best2.pt"},
        source=FakeVideoSource(frames=10),
        events=events,
    )

    checks = preflight.run()

    assert [check.name for check in checks] == ["detector", "ocr", "camera", "video"]
    assert [check.ok for check in checks] == [True, None, None, True]
    assert [check.skipped for check in checks] == [False, True, True, False]
    payloads = _events_of(events, "preflight")
    assert [item["check"] for item in payloads] == ["detector", "ocr", "camera", "video"]
    assert [item["ok"] for item in payloads] == [True, None, None, True]
    assert preflight.run() is checks, "run() 幂等：第二次数模型不会重来一遍"


def test_real_preflight_fails_loudly_when_a_switch_is_on_without_a_loader() -> None:
    """开着却没给载入回调 = 装配错误：直接判失败，绝不静默当成通过。"""
    config = PreflightConfig(
        load_detector=False,
        load_ocr=True,  # 开着
        load_camera=False,
        check_video=False,
    )
    preflight = _preflight(config, loaders={})

    with pytest.raises(PreflightError, match="ocr"):
        preflight.run()


def test_real_preflight_reports_a_failing_loader_with_the_check_name() -> None:
    config = PreflightConfig(
        load_detector=False, load_ocr=False, load_camera=True, check_video=False
    )

    def boom() -> Any:
        raise FileNotFoundError("camera_calib.json 读不了")

    with pytest.raises(PreflightError, match="camera: "):
        _preflight(config, loaders={"camera": boom}).run()


def test_real_preflight_video_check_needs_enough_frames() -> None:
    """视频自检在 ``video_probe_s`` 窗口内等够 ``video_min_frames`` 帧，等不到就失败。"""
    config = PreflightConfig(
        load_detector=False,
        load_ocr=False,
        load_camera=False,
        check_video=True,
        video_probe_s=0.5,
        video_min_frames=10,
    )
    source = FakeVideoSource(frames=3)

    with pytest.raises(PreflightError, match="只收到 3 帧"):
        _preflight(config, source=source).run()

    # 帧数是"等一等就够"的：窗口内涨到 10 帧就通过
    growing = FakeVideoSource(frames=3)
    clock = FakeClock()

    def sleep(seconds: float) -> None:
        clock.sleep(seconds)
        growing.stats.frames += 10  # 这一觉睡来了 10 帧

    checks = _preflight(config, source=growing, clock=clock, sleep=sleep).run()
    assert checks[-1].name == "video" and checks[-1].ok is True


def test_real_preflight_video_check_rejects_a_resolution_mismatch() -> None:
    """有标定时核对最新帧尺寸：与标定不一致 → 失败（内参是按标定分辨率算的）。"""
    config = PreflightConfig(
        load_detector=False,
        load_ocr=False,
        load_camera=False,
        check_video=True,
        video_probe_s=0.5,
        video_min_frames=1,
    )
    preflight = _preflight(
        config,
        source=FakeVideoSource(frames=5, size=(640, 360)),
        camera_model=SimpleNamespace(width=1280, height=720),
    )

    with pytest.raises(PreflightError, match="不一致"):
        preflight.run()


def test_preflight_passes_then_waits_for_airborne_and_enters_recon() -> None:
    """顺序：INIT（遥测/原点，不上传任务）→ PREFLIGHT → WAIT_AIRBORNE → RECON。"""
    clock = FakeClock()
    controller = FakeController()
    events: list[tuple[str, dict[str, Any]]] = []
    preflight = FakePreflight()
    runner = _runner(clock, controller, judge=FakeJudge(), events=events, preflight=preflight)

    _drive(runner, clock, ticks=1)

    assert runner.state is MissionState.RECON
    assert [str(record.to_state) for record in runner.history] == [
        "PREFLIGHT",
        "WAIT_AIRBORNE",
        "RECON",
    ]
    assert runner.history[0].reason == "telemetry_ready"
    assert preflight.calls == 1
    assert _events_of(events, "airborne"), "进场要记 airborne 事件"


def test_preflight_failure_aborts_with_the_check_name() -> None:
    """预检失败 → ``ABORT("preflight_failed:<check>")``（原因带检查名，便于定位）。"""
    clock = FakeClock()
    events: list[tuple[str, dict[str, Any]]] = []
    runner = _runner(
        clock,
        FakeController(),
        judge=FakeJudge(),
        events=events,
        preflight=FakePreflight(fail="ocr"),
    )

    _drive(runner, clock, ticks=1)

    assert runner.state is MissionState.ABORT
    assert runner.history[-1].reason == "preflight_failed:ocr"
    assert runner.stats.aborts == 1


def test_preflight_result_with_ok_false_also_aborts() -> None:
    """实现方也可以返回 ``ok=False`` 的项（不抛异常）——调用方两种都认。"""
    clock = FakeClock()
    preflight = FakePreflight(checks=(PreflightCheck("video", False, "只收到 2 帧"),))
    runner = _runner(clock, FakeController(), judge=FakeJudge(), preflight=preflight)

    _drive(runner, clock, ticks=1)

    assert runner.state is MissionState.ABORT
    assert runner.history[-1].reason == "preflight_failed:video"


def test_missing_preflight_is_skipped_and_recorded() -> None:
    """没注入预检（离线测试）→ 记一条 ``preflight_skipped`` 后放过，不假装通过。"""
    clock = FakeClock()
    events: list[tuple[str, dict[str, Any]]] = []
    runner = _runner(clock, FakeController(), judge=FakeJudge(), events=events)

    _drive(runner, clock, ticks=1)

    assert runner.state is MissionState.RECON
    assert [record.reason for record in runner.history] == [
        "telemetry_ready",
        "preflight_skipped",
        "airborne",
    ]
    assert _events_of(events, "preflight_skipped")


def test_operator_mode_never_uploads_the_recon_mission() -> None:
    """正式任务（``recon_upload="operator"``）：不上传，只等操作手在 QGC 启动。"""
    clock = FakeClock()
    controller = FakeController()
    controller.mission_mode = False  # 操作手还没启动
    events: list[tuple[str, dict[str, Any]]] = []
    config = _config(
        mission=MissionConfig(tick_hz=20.0, recon_upload="operator"),
    )
    runner = _runner(clock, controller, config=config, judge=FakeJudge(), events=events)

    _drive(runner, clock, ticks=1)
    assert runner.state is MissionState.RECON
    assert controller.uploads == [], "operator 模式绝不能自己上传侦查航线"
    assert _events_of(events, "recon_waiting_operator")

    _drive(runner, clock, ticks=2)
    assert runner.state is MissionState.RECON, "任务没启动就继续等"

    controller.mission_mode = True  # 操作手启动了任务
    _drive(runner, clock, ticks=1)
    assert runner.state is MissionState.RECON

    controller.finish_mission()
    _drive_until(runner, clock, MissionState.HOLD_PROCESS)
    assert runner.state is MissionState.HOLD_PROCESS
    assert controller.uploads == [], "HOLD_PROCESS 里本包仍然什么都没上传"

    # 上传发生在 HOLD_PROCESS 的下一拍（进 OVERFLY 时），所以要驱动到 OVERFLY 再断言
    _drive_until(runner, clock, MissionState.OVERFLY)
    assert runner.state is MissionState.OVERFLY
    assert len(controller.uploads) == 1, "进入飞掠段才轮到本包上传（飞掠+降落），且只这一次"


def test_auto_mode_uploads_the_recon_mission() -> None:
    """自动测试档（``recon_upload="auto"``）：本包上传并启动侦查航线。"""
    clock = FakeClock()
    controller = FakeController()
    runner = _runner(clock, controller, judge=FakeJudge())

    _drive(runner, clock, ticks=1)

    assert runner.state is MissionState.RECON
    assert len(controller.uploads) == 1 and controller.count("start_mission") == 1


def test_airborne_gate_waits_until_in_air() -> None:
    """``in_air=False`` 就一直等（在停机坪上不进侦查），转 True 才走。"""
    clock = FakeClock()
    controller = FakeController()
    broker = FakeBroker(clock)
    broker.in_air = False
    runner = _runner(clock, controller, broker=broker, judge=FakeJudge())

    _drive(runner, clock, ticks=1)
    assert runner.state is MissionState.WAIT_AIRBORNE
    assert controller.uploads == []

    _drive(runner, clock, ticks=3)
    assert runner.state is MissionState.WAIT_AIRBORNE

    broker.in_air = True
    _drive(runner, clock, ticks=1)
    assert runner.state is MissionState.RECON
    assert runner.history[-1].reason == "airborne"


def test_airborne_falls_back_to_altitude_when_in_air_is_missing() -> None:
    """``in_air`` 取不到时用 ``relative_altitude_m >= airborne_alt_m`` 兜底。"""
    clock = FakeClock()
    events: list[tuple[str, dict[str, Any]]] = []
    broker = FakeBroker(clock)
    broker.in_air = None
    broker.altitude_m = 1.0  # 低于 airborne_alt_m（默认 5m）
    runner = _runner(clock, FakeController(), broker=broker, judge=FakeJudge(), events=events)

    _drive(runner, clock, ticks=2)
    assert runner.state is MissionState.WAIT_AIRBORNE
    assert _events_of(events, "airborne") == [], "高度没到就不该进侦查"

    broker.altitude_m = 30.0
    _drive(runner, clock, ticks=1)

    assert runner.state is MissionState.RECON
    assert _events_of(events, "airborne") == [{"source": "altitude"}], "记下用的是哪条判据"


def test_require_airborne_false_passes_the_gate_without_waiting() -> None:
    """``require_airborne=False``（地面演练）：INIT 之后不等起飞，一拍进 RECON。

    放行必须留痕：记一条带 ``reason`` 的 ``airborne_skipped`` 事件，而不是静默通过
    （否则"正式任务忘了打开该检查"就再也查不出来了）。
    """
    clock = FakeClock()
    controller = FakeController()
    events: list[tuple[str, dict[str, Any]]] = []
    broker = FakeBroker(clock)
    broker.in_air = False  # 飞机停在停机坪上
    config = _config(
        mission=MissionConfig(tick_hz=20.0, recon_upload="auto", require_airborne=False)
    )
    runner = _runner(
        clock, controller, config=config, broker=broker, judge=FakeJudge(), events=events
    )

    _drive(runner, clock, ticks=1)  # 一拍走完 INIT → PREFLIGHT → WAIT_AIRBORNE → RECON

    assert runner.state is MissionState.RECON, "关闭等待起飞后应当立即进入侦查"
    assert [str(record.to_state) for record in runner.history] == [
        "PREFLIGHT",
        "WAIT_AIRBORNE",
        "RECON",
    ]
    assert runner.history[-1].reason == "airborne_skipped"
    skipped = _events_of(events, "airborne_skipped")
    assert len(skipped) == 1, "放行必须记事件，不得无提示通过"
    assert skipped[0].get("reason"), "事件里要说清为什么放行"
    assert _events_of(events, "airborne") == [], "没检测到起飞就不该有 airborne 事件"


def test_require_airborne_true_still_waits_on_the_ground() -> None:
    """默认（``require_airborne=True``）行为不变：没起飞就等，且不记 ``airborne_skipped``。"""
    clock = FakeClock()
    controller = FakeController()
    events: list[tuple[str, dict[str, Any]]] = []
    broker = FakeBroker(clock)
    broker.in_air = False
    config = _config()  # 默认档：require_airborne 保持 True
    runner = _runner(
        clock, controller, config=config, broker=broker, judge=FakeJudge(), events=events
    )

    assert config.mission.require_airborne is True, "默认必须等飞机真的在空中"
    _drive(runner, clock, ticks=3)

    assert runner.state is MissionState.WAIT_AIRBORNE
    assert controller.uploads == [], "等起飞期间什么都不上传"
    assert _events_of(events, "airborne_skipped") == [], "默认档不走临时放行开关"

    broker.in_air = True  # 飞机真的起飞了
    _drive(runner, clock, ticks=1)

    assert runner.state is MissionState.RECON
    assert runner.history[-1].reason == "airborne"
    assert _events_of(events, "airborne") == [{"source": "in_air"}]


def test_airborne_timeout_aborts_explicitly() -> None:
    """一直不起飞 → 超 ``airborne_timeout_s`` 显式失败，不无限等。"""
    clock = FakeClock()
    broker = FakeBroker(clock)
    broker.in_air = False
    config = _config(
        mission=MissionConfig(tick_hz=20.0, recon_upload="auto", airborne_timeout_s=5.0)
    )
    runner = _runner(
        clock, config=config, controller=FakeController(), broker=broker, judge=FakeJudge()
    )

    for _ in range(20):
        runner.update()
        clock.advance(1.0)
        if runner.state is MissionState.ABORT:
            break

    assert runner.state is MissionState.ABORT
    assert runner.history[-1].reason == "airborne_timeout"


def test_mission_monitor_ignores_stale_finished() -> None:
    """刚上传的新任务可能仍报"上一次已飞完" —— 必须先见到 False 才认 True。"""
    monitor = MissionMonitor()
    assert monitor.update(True) is False
    assert monitor.armed is False
    assert monitor.update(False) is False
    assert monitor.update(True) is True
    assert monitor.true_seen == 2


# ----------------------------------------------------------------------
# 任务项与 raw 转换（控制器的离线部分）
# ----------------------------------------------------------------------
def test_raw_item_uses_mavlink_fields_and_sets_current_on_first_only() -> None:
    """命令/帧/参数原样过去；``current`` 只有第 0 项为 1。"""
    items = build_recon_mission(_config(recon=2))

    first = to_raw_item(items[0], 0)
    second = to_raw_item(items[1], 1)

    assert first.command == MAV_CMD_NAV_TAKEOFF
    assert second.command == MAV_CMD_NAV_WAYPOINT
    assert first.seq == 0 and first.current == 1
    assert second.seq == 1 and second.current == 0, (
        "全 0 会被 MAVSDK 判 CURRENT_INVALID（实测），全 1 含义错乱"
    )
    assert first.x == int(round(47.0 * 1e7)) and first.z == 60.0
    assert first.frame == MAV_FRAME_GLOBAL_RELATIVE_ALT
    assert first.mission_type == 0


def test_takeoff_and_land_are_single_items_not_expanded() -> None:
    """``NAV_TAKEOFF`` / ``NAV_LAND`` 各是一项，不被翻译层拆开。

    这是 2026-09 SITL 演练的根因：MAVSDK 会把 ``vehicle_action=LAND`` 拆成
    "同坐标航点 + ``NAV_LAND``"，PX4 固定翼的降落判据（紧前一项必须高于落点）
    随即把整条任务判为不可行，飞机原地盘旋而 ``start_mission()`` 仍回成功。
    """
    takeoff = MissionItem.takeoff(47.0, 8.0, 60.0)
    land = MissionItem.land(47.0, 8.0)

    assert takeoff.command == MAV_CMD_NAV_TAKEOFF and takeoff.param1 == 15.0
    assert land.command == MAV_CMD_NAV_LAND and land.alt_m == 0.0
    assert land.is_positional
    assert not MissionItem(command=189).is_positional, "DO_LAND_START 这类指令项不带位置"


def test_waypoint_acceptance_radius_defaults_to_three_metres() -> None:
    """默认接受半径 3m（与旧实现一致）；0 在 PX4 上含义不同（用 ``NAV_ACC_RAD``）。"""
    default = MissionItem.from_waypoint(Waypoint(lat=1.0, lon=2.0, alt_m=30.0))
    custom = MissionItem.from_waypoint(
        Waypoint(lat=1.0, lon=2.0, alt_m=30.0, acceptance_radius_m=5.0)
    )

    assert default.param2 == 3.0 and custom.param2 == 5.0
    assert math.isnan(MissionItem.waypoint(1.0, 2.0, 30.0).param4), "偏航角不指定 = NaN"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"command": -1}, "MAV_CMD"),
        ({"lat": 91.0}, "纬度"),
        ({"lon": -181.0}, "经度"),
        ({"alt_m": float("nan")}, "高度"),
    ],
)
def test_mission_item_rejects_invalid_values(kwargs: dict, message: str) -> None:
    params = {
        "command": MAV_CMD_NAV_WAYPOINT,
        "lat": 47.0,
        "lon": 8.0,
        "alt_m": 20.0,
        **kwargs,
    }
    with pytest.raises(ValueError, match=message):
        MissionItem(**params)


# ----------------------------------------------------------------------
# 航线规划
# ----------------------------------------------------------------------
def test_llaref_of_keeps_field_order() -> None:
    """NedOrigin 是 lat 在前、LLARef 是 lon 在前 —— 按名字搬，别按位置。"""
    ref = llaref_of(NedOrigin(lat_deg=47.5, lon_deg=8.25, alt_m=500.0))
    assert (ref.lat_deg, ref.lon_deg, ref.alt_m) == (47.5, 8.25, 500.0)


def test_overfly_positions_are_symmetric_about_the_target() -> None:
    target = (100.0, 50.0, 0.0)
    entry, exit_ = overfly_positions(target, heading_deg=90.0, leg_length_m=200.0)

    assert entry[0] == pytest.approx(100.0) and exit_[0] == pytest.approx(100.0)
    assert entry[1] == pytest.approx(-50.0)  # 航向 90°（正东）：entry 在西侧
    assert exit_[1] == pytest.approx(150.0)
    assert math.dist(entry[:2], exit_[:2]) == pytest.approx(200.0)
    midpoint = ((entry[0] + exit_[0]) / 2, (entry[1] + exit_[1]) / 2)
    assert midpoint == pytest.approx((100.0, 50.0))
    # 沿航向飞：entry 在目标之前（内积为负）、exit 在之后（内积为正）
    assert entry[1] - target[1] < 0 < exit_[1] - target[1]


def test_overfly_waypoints_round_trip_through_wgs84() -> None:
    target = (300.0, -120.0, 0.0)
    origin = llaref_of(ORIGIN)
    entry, exit_ = overfly_waypoints(
        target, heading_deg=45.0, leg_length_m=400.0, altitude_m=20.0, origin=origin
    )

    assert entry.alt_m == 20.0 and exit_.alt_m == 20.0, "高度是相对起飞点的高度"
    expected_entry, expected_exit = overfly_positions(target, heading_deg=45.0, leg_length_m=400.0)
    # 换算回 NED 时海拔要按"原点海拔 + 相对高度"给，否则切平面对不准（米级以下的小偏）
    entry_ned = wgs84_to_ned(entry.lon, entry.lat, origin.alt_m + entry.alt_m, origin)
    exit_ned = wgs84_to_ned(exit_.lon, exit_.lat, origin.alt_m + exit_.alt_m, origin)
    assert entry_ned[:2] == pytest.approx(expected_entry[:2], abs=0.05)
    assert exit_ned[:2] == pytest.approx(expected_exit[:2], abs=0.05)
    # 顺带验证 ned_to_wgs84 ↔ wgs84_to_ned 不经航点包装时也自洽
    lon, lat, alt = ned_to_wgs84(target, origin)
    back = wgs84_to_ned(lon, lat, alt, origin)
    assert back[:2] == pytest.approx(target[:2], abs=1e-6)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"heading_deg": float("nan")}, "航向"),
        ({"leg_length_m": 0.0}, "段长"),
        ({"leg_length_m": float("inf")}, "段长"),
    ],
)
def test_overfly_positions_rejects_bad_geometry(kwargs: dict, message: str) -> None:
    params = {"heading_deg": 0.0, "leg_length_m": 200.0, **kwargs}
    with pytest.raises(PlanningError, match=message):
        overfly_positions((0.0, 0.0, 0.0), **params)


def test_overfly_positions_rejects_non_finite_target() -> None:
    with pytest.raises(PlanningError, match="有限值"):
        overfly_positions((float("inf"), 0.0, 0.0), heading_deg=0.0, leg_length_m=100.0)
    with pytest.raises(PlanningError, match="3 维"):
        overfly_positions((1.0, 2.0), heading_deg=0.0, leg_length_m=100.0)


def test_overfly_waypoints_rejects_non_finite_altitude() -> None:
    with pytest.raises(PlanningError, match="高度"):
        overfly_waypoints(
            (0.0, 0.0, 0.0),
            heading_deg=0.0,
            leg_length_m=100.0,
            altitude_m=float("nan"),
            origin=llaref_of(ORIGIN),
        )


def test_build_recon_mission_marks_takeoff_on_first_item() -> None:
    items = build_recon_mission(_config(recon=3))
    assert len(items) == 3
    assert items[0].command == MAV_CMD_NAV_TAKEOFF
    assert all(item.command == MAV_CMD_NAV_WAYPOINT for item in items[1:])
    assert [item.alt_m for item in items] == [60.0, 60.0, 60.0]

    plain = build_recon_mission(_config(recon=3, mission=MissionConfig(takeoff_first=False)))
    assert all(item.command == MAV_CMD_NAV_WAYPOINT for item in plain)


def test_build_recon_mission_rejects_empty_route() -> None:
    with pytest.raises(PlanningError, match="侦查航线为空"):
        build_recon_mission(_config(recon=0))


def test_build_drop_mission_merges_overfly_and_landing() -> None:
    config = _config(landing=2)
    plan = build_drop_mission(config, origin=llaref_of(ORIGIN), target_ned=(120.0, 60.0, 0.0))

    assert isinstance(plan, DropMissionPlan)
    assert plan.source == "target"
    assert plan.target_ned == (120.0, 60.0, 0.0)
    assert plan.heading_deg == 90.0
    assert len(plan.items) == 4 and plan.overfly_count == 2 and plan.landing_count == 2
    # 顺序：飞掠 entry → 飞掠 exit → 降落航线……（Q13：拼成一条任务）
    assert plan.items[0].lat == pytest.approx(plan.entry.lat)
    assert plan.items[1].lat == pytest.approx(plan.exit.lat)
    assert plan.items[2].lon == pytest.approx(config.routes.landing_route[0].lon)
    assert plan.items[-1].command == MAV_CMD_NAV_LAND
    assert all(item.command == MAV_CMD_NAV_WAYPOINT for item in plan.items[:-1])
    assert plan.as_dict()["source"] == "target"


def test_build_drop_mission_normalises_heading_and_can_skip_land_action() -> None:
    config = _config(
        overfly=OverflyConfig(heading_deg=450.0, altitude_m=25.0, leg_length_m=100.0),
        mission=MissionConfig(land_last=False),
    )
    plan = build_drop_mission(config, origin=llaref_of(ORIGIN), target_ned=(0.0, 0.0, 0.0))
    assert plan.heading_deg == 90.0, "450° 应当归一化到 90°"
    assert plan.items[-1].command == MAV_CMD_NAV_WAYPOINT
    assert all(item.alt_m == 25.0 for item in plan.items[:2])


def test_build_drop_mission_falls_back_to_backup_point() -> None:
    """无目标分支：以备用点为目标生成同样航点（Q11）。"""
    config = _config()
    plan = build_drop_mission(config, origin=llaref_of(ORIGIN), target_ned=None)

    assert plan.source == "backup"
    expected = waypoint_to_ned(config.routes.backup_point, llaref_of(ORIGIN))
    assert plan.target_ned == pytest.approx(expected)
    assert plan.target_ned[2] == 0.0, "目标是地面点"
    assert len(plan.items) == 4


def test_build_drop_mission_requires_target_or_backup() -> None:
    with pytest.raises(PlanningError, match="备用点"):
        build_drop_mission(_config(backup=False), origin=llaref_of(ORIGIN), target_ned=None)


def test_build_drop_mission_requires_origin_and_landing_route() -> None:
    with pytest.raises(PlanningError, match="NED 原点"):
        build_drop_mission(_config(), origin=None, target_ned=(0.0, 0.0, 0.0))
    with pytest.raises(PlanningError, match="降落段为空"):
        build_drop_mission(_config(landing=0), origin=llaref_of(ORIGIN), target_ned=(0.0, 0.0, 0.0))


def test_build_drop_mission_rejects_non_finite_target() -> None:
    with pytest.raises(PlanningError, match="有限值"):
        build_drop_mission(_config(), origin=llaref_of(ORIGIN), target_ned=(float("nan"), 0.0, 0.0))


# ----------------------------------------------------------------------
# MissionRunner：全链路
# ----------------------------------------------------------------------
def test_runner_completes_full_mission() -> None:
    clock = FakeClock()
    controller = FakeController()
    judge = FakeJudge([ReleaseDecision(True, "predict", horizontal_error_m=0.4)])
    events: list[tuple[str, dict[str, Any]]] = []
    runner = _runner(
        clock,
        controller,
        judge=judge,
        target_result=_labeled_points,
        target_busy=lambda: False,
        events=events,
    )

    assert runner.state is MissionState.INIT
    _drive(runner, clock, ticks=1)
    assert runner.state is MissionState.RECON, "遥测与原点就绪后立刻上传并启动侦查航线"

    _drive(runner, clock, ticks=1)  # 侦查段跑一拍：让看门狗看到"任务在跑"
    controller.finish_mission()
    _drive_until(runner, clock, MissionState.HOLD_PROCESS)
    assert runner.state is MissionState.HOLD_PROCESS
    assert controller.count("hold") == 1

    _drive_until(runner, clock, MissionState.OVERFLY)
    assert runner.state is MissionState.OVERFLY
    assert judge.resets == 1, "进入飞掠段要清掉判据的锁存"
    _drive_until(runner, clock, MissionState.LAND)
    assert runner.state is MissionState.LAND
    assert controller.count("gripper_release") == 1

    _drive(runner, clock, ticks=1)  # 让看门狗在 LAND 里先看到一次"任务在跑"
    controller.finish_mission()
    _drive_until(runner, clock, MissionState.DONE)

    assert runner.state is MissionState.DONE
    assert runner.stats.uploads == 2 and runner.stats.releases == 1
    assert runner.stats.aborts == 0 and runner.stats.errors == 0
    assert [str(record.to_state) for record in runner.history] == [
        "PREFLIGHT",
        "WAIT_AIRBORNE",
        "RECON",
        "HOLD_PROCESS",
        "OVERFLY",
        "LAND",
        "DONE",
    ]
    assert [data["to_state"] for data in _events_of(events, "state")][-1] == "DONE"
    assert _events_of(events, "origin")[0]["lat_deg"] == 47.0
    assert _events_of(events, "drop_plan")[0]["source"] == "target"
    assert _events_of(events, "drop")[0]["reason"] == "predict"
    assert runner.plan is not None and runner.plan.source == "target"


def test_runner_uploads_recon_then_merged_drop_mission() -> None:
    clock = FakeClock()
    controller = FakeController()
    judge = FakeJudge([ReleaseDecision(True, "predict")])
    runner = _runner(clock, controller, judge=judge)

    _drive(runner, clock, ticks=2)  # 第 1 拍 INIT→RECON，第 2 拍让看门狗看到"任务在跑"
    controller.finish_mission()
    _drive_until(runner, clock, MissionState.OVERFLY)
    _drive(runner, clock, ticks=1)  # OVERFLY 的判据评估发生在进入状态后的下一拍

    # 判据每一拍都拿到 ground_z 与密度基准（地面海拔）：默认地面点未配置时用原点海拔 500m
    assert judge.ground_z == pytest.approx(0.0)
    assert judge.ground_altitude_m == pytest.approx(ORIGIN.alt_m)

    assert len(controller.uploads) == 2
    recon, drop = controller.uploads
    assert [item.command for item in recon] == [
        MAV_CMD_NAV_TAKEOFF,
        MAV_CMD_NAV_WAYPOINT,
        MAV_CMD_NAV_WAYPOINT,
    ]
    assert [item.command for item in drop] == [
        MAV_CMD_NAV_WAYPOINT,
        MAV_CMD_NAV_WAYPOINT,
        MAV_CMD_NAV_WAYPOINT,
        MAV_CMD_NAV_LAND,
    ]
    # 上传 → 启动成对出现，且侦查在前、盘旋在后（顺序不能反）
    assert controller.calls.index("upload_mission") < controller.calls.index("start_mission")
    assert controller.calls.index("start_mission") < controller.calls.index("hold")


def _labeled_points() -> Any:
    """一批带编号的观测 + 一个孤立点（P7 的输入）。"""
    points = [
        TargetPoint(
            north_m=100.0,
            east_m=50.0,
            capture_timestamp=1.0 + index * 0.1,
            frame_index=index,
            code=56,
            confidence=0.9,
        )
        for index in range(3)
    ]
    points.append(
        TargetPoint(
            north_m=300.0,
            east_m=300.0,
            capture_timestamp=1.5,
            frame_index=3,
            code=12,
            confidence=0.9,
        )
    )
    return analyze(points, TargetingConfig(eps_m=0.75, min_samples=2))


def test_runner_feeds_targeting_result_to_the_judge() -> None:
    """P7 → P9 的接缝：选中类的坐标原样进投放判据。"""
    clock = FakeClock()
    controller = FakeController()
    judge = FakeJudge([ReleaseDecision(True, "predict")])
    result = _labeled_points()
    assert result.ok and result.code == 56

    runner = _runner(
        clock,
        controller,
        judge=judge,
        target_result=lambda: result,
        target_busy=lambda: False,
    )
    _drive(runner, clock, ticks=2)  # 第 1 拍 INIT→RECON，第 2 拍让看门狗看到"任务在跑"
    controller.finish_mission()
    _drive_until(runner, clock, MissionState.OVERFLY)
    _drive(runner, clock, ticks=1)  # 飞掠段的第一拍才会问判据

    assert judge.targets == [(100.0, 50.0, 0.0)]
    assert judge.ground_z == 0.0, "原点海拔 500 = 地面点海拔 500 → ground_z=0"
    assert runner.plan is not None and runner.plan.source == "target"


def test_runner_treats_future_snapshot_timestamp_as_fresh() -> None:
    """快照时间戳比本地时钟还新（时钟偏差）时，不能算"陈旧"。"""
    clock = FakeClock(start=time.time() - 30.0)
    controller = FakeController()
    runner = _runner(clock, controller, broker=_live_broker())
    _drive(runner, clock, ticks=1)
    assert runner.state is MissionState.RECON


class MovingBroker:
    """让飞机沿航向匀速前进的假 broker——给真投放判据传入动态快照。

    :class:`FakeBroker` 是静止的，弹道判据在静止状态下的预测落点永远在原地，
    测不出"飞到某个位置才该投"。这里让位置随时间推进，判据才会在正确的时刻触发。
    """

    def __init__(
        self,
        clock: FakeClock,
        *,
        east0: float = -200.0,
        speed_m_s: float = 18.0,
        down_m: float = -20.0,
        north_m: float = 0.0,
        attitude_deg: tuple[float, float, float] | None = None,
        quaternion: tuple[float, float, float, float] | None = None,
        wind_ned: tuple[float, float, float] | None = None,
    ) -> None:
        self._clock = clock
        self._east0 = east0
        self._speed = speed_m_s
        self._down = down_m
        self._north = north_m
        self._attitude = attitude_deg
        self._quaternion = quaternion
        self._wind = wind_ned
        self._started_at = clock()

    def get_snapshot(self) -> TelemetrySnapshot:
        now = self._clock()
        snapshot = TelemetrySnapshot(timestamp=now)
        snapshot.in_air = True  # 离线用例默认"已经起飞"（等待起飞另有专门用例）
        snapshot.north_m = self._north
        snapshot.east_m = self._east0 + self._speed * (now - self._started_at)
        snapshot.down_m = self._down
        snapshot.vx_m_s = 0.0
        snapshot.vy_m_s = self._speed
        snapshot.vz_m_s = 0.0
        if self._attitude is not None:
            snapshot.roll_deg, snapshot.pitch_deg, snapshot.yaw_deg = self._attitude
        if self._quaternion is not None:
            (
                snapshot.quaternion_w,
                snapshot.quaternion_x,
                snapshot.quaternion_y,
                snapshot.quaternion_z,
            ) = self._quaternion
        if self._wind is not None:
            (
                snapshot.wind_north_m_s,
                snapshot.wind_east_m_s,
                snapshot.wind_down_m_s,
            ) = self._wind
        return snapshot


def test_runner_drops_with_the_real_ballistics_judge() -> None:
    """真判据接进状态机（P8 × P9 的接缝）：飞过目标前触发，并把完整预测落进事件。

    这一条刻意不用假判据：假判据只模仿接口，签名/参数一旦对不上它照样通过。
    """
    clock = FakeClock()
    controller = FakeController()
    judge = ReleaseJudge(
        DropConfig(radius_m=2.0),
        BallisticsModel(BallisticsConfig()),
        overfly_heading_deg=90.0,  # 正东，与假 broker 的飞行方向一致
        ground_z=0.0,
    )
    # 目标在正东 0m（飞机从 -200m 处往东飞）、北向 0m
    target_points = [
        TargetPoint(north_m=0.0, east_m=0.0, capture_timestamp=float(i), frame_index=i, code=56)
        for i in range(2)
    ]
    result = analyze(target_points, TargetingConfig(eps_m=0.75, min_samples=2))
    events: list[tuple[str, dict[str, Any]]] = []
    runner = _runner(
        clock,
        controller,
        broker=MovingBroker(clock),
        judge=judge,
        target_result=lambda: result,
        target_busy=lambda: False,
        events=events,
    )

    _drive(runner, clock, ticks=2)
    controller.finish_mission()
    _drive_until(runner, clock, MissionState.HOLD_PROCESS)
    _drive_until(runner, clock, MissionState.OVERFLY)
    _drive_until(runner, clock, MissionState.LAND, step=0.1, limit=2000)

    assert runner.state is MissionState.LAND
    assert controller.count("gripper_release") == 1
    drop = _events_of(events, "drop")[0]
    assert drop["reason"] == "predict"
    assert drop["should_release"] is True
    assert drop["horizontal_error_m"] <= 2.0, "预测落点必须在投放半径内"
    assert drop["target_source"] == "target"
    assert drop["target_ned"] == [0.0, 0.0, 0.0]
    # 投放点应当落在"落点前推量"附近：20m 高、18m/s、二次阻力 → 约 35~40m 提前量
    release_east = drop["release_position"][1]
    assert -45.0 <= release_east <= -25.0, release_east


def test_runner_records_the_drop_state_for_the_ballistics_fit() -> None:
    """投放瞬间的飞机状态（位置/速度/姿态/风）必须当场记下来。

    这份数据事后无法重建，而 :func:`~airdrop.ballistics.fit.fit_ballistics` 反演弹道
    参数只能靠它——所以这里断言的是"记录内容与判据看到的快照一致"，而不是"回调被调过"。
    """
    clock = FakeClock()
    controller = FakeController()
    judge = ReleaseJudge(
        DropConfig(radius_m=2.0, delay_s=0.08),
        BallisticsModel(BallisticsConfig()),
        overfly_heading_deg=90.0,
        ground_z=0.0,
    )
    broker = MovingBroker(
        clock,
        attitude_deg=(4.0, -3.0, 90.0),
        quaternion=(0.0, 0.0, 0.7071067811865476, 0.7071067811865476),
        wind_ned=(2.0, -1.0, 0.0),
    )
    target_points = [
        TargetPoint(north_m=0.0, east_m=0.0, capture_timestamp=float(i), frame_index=i, code=56)
        for i in range(2)
    ]
    result = analyze(target_points, TargetingConfig(eps_m=0.75, min_samples=2))
    recorded: list[DropRecord] = []
    runner = _runner(
        clock,
        controller,
        broker=broker,
        judge=judge,
        target_result=lambda: result,
        target_busy=lambda: False,
        on_drop=recorded.append,
    )

    _drive(runner, clock, ticks=2)
    controller.finish_mission()
    _drive_until(runner, clock, MissionState.HOLD_PROCESS)
    _drive_until(runner, clock, MissionState.OVERFLY)
    _drive_until(runner, clock, MissionState.LAND, step=0.1, limit=2000)

    assert runner.state is MissionState.LAND
    assert len(recorded) == 1
    record = recorded[0]
    assert runner.drops == (record,)

    # 位置/速度：与判据看到的动态快照一致（投放点在目标西侧几十米处）
    assert record.index == 1
    assert record.position_ned[0] == pytest.approx(0.0)
    assert -45.0 <= record.position_ned[1] <= -25.0
    assert record.position_ned[2] == pytest.approx(-20.0)
    assert record.velocity_ned == pytest.approx((0.0, 18.0, 0.0))
    assert record.horizontal_speed_m_s == pytest.approx(18.0)
    assert record.height_agl_m == pytest.approx(20.0)
    assert record.heading_deg == pytest.approx(90.0)
    # 姿态与风（姿态各留一份；反演用它把挂点偏移转到 NED）
    assert record.euler_deg == (4.0, -3.0, 90.0)
    assert record.quaternion_wxyz == pytest.approx(
        (0.0, 0.0, 0.7071067811865476, 0.7071067811865476)
    )
    assert record.wind_ned == pytest.approx((2.0, -1.0, 0.0))
    # 判据的账：目标、延迟、前推位置与预测落点
    assert record.target_ned == pytest.approx((0.0, 0.0, 0.0))
    assert record.delay_s == pytest.approx(0.08)
    assert record.reason == "predict"
    assert record.ground_z == pytest.approx(0.0)
    assert record.origin is not None and record.origin.alt_m == pytest.approx(ORIGIN.alt_m)
    assert record.predicted_impact_ned is not None
    assert record.predicted_error_m is not None and record.predicted_error_m <= 2.0
    assert record.release_position_ned is not None
    # 前推位置 = 记录位置 + 速度 × delay（一阶，与判据同款）
    assert record.release_position_ned[1] == pytest.approx(record.position_ned[1] + 18.0 * 0.08)
    # 写入磁盘与回读一致（drops.jsonl 走的就是 as_dict/from_dict）
    assert DropRecord.from_dict(record.as_dict()) == record


def test_runner_records_a_drop_even_when_the_callback_fails() -> None:
    """回调抛异常只记日志：弹已经出去了，不能因为写不进记录就改变任务状态。"""
    clock = FakeClock()
    controller = FakeController()
    judge = ReleaseJudge(
        DropConfig(radius_m=2.0),
        BallisticsModel(BallisticsConfig()),
        overfly_heading_deg=90.0,
        ground_z=0.0,
    )
    target_points = [
        TargetPoint(north_m=0.0, east_m=0.0, capture_timestamp=float(i), frame_index=i, code=56)
        for i in range(2)
    ]
    result = analyze(target_points, TargetingConfig(eps_m=0.75, min_samples=2))
    events: list[tuple[str, dict[str, Any]]] = []

    def boom(record: DropRecord) -> None:
        raise RuntimeError("磁盘满了（假）")

    runner = _runner(
        clock,
        controller,
        broker=MovingBroker(clock),
        judge=judge,
        target_result=lambda: result,
        target_busy=lambda: False,
        events=events,
        on_drop=boom,
    )

    _drive(runner, clock, ticks=2)
    controller.finish_mission()
    _drive_until(runner, clock, MissionState.HOLD_PROCESS)
    _drive_until(runner, clock, MissionState.OVERFLY)
    _drive_until(runner, clock, MissionState.LAND, step=0.1, limit=2000)

    assert runner.state is MissionState.LAND  # 没有因为回调抛错而 ABORT
    assert controller.count("gripper_release") == 1
    assert len(runner.drops) == 1  # 内存里那份仍然记下了
    assert _events_of(events, "drop_record")[0]["index"] == 1


def test_record_drop_reports_a_broken_snapshot_without_touching_the_state() -> None:
    """快照缺位置/速度时：记一条 error 事件、不写记录、不改变任务状态。"""
    clock = FakeClock()
    controller = FakeController()
    events: list[tuple[str, dict[str, Any]]] = []
    runner = _runner(clock, controller, events=events)
    decision = ReleaseDecision(
        should_release=True,
        reason="predict",
        timestamp=clock(),
        predicted=None,
        horizontal_error_m=0.1,
        passed_target=True,
        release_position=(1.0, 2.0, -3.0),
    )
    snapshot = TelemetrySnapshot(timestamp=clock())  # 什么都没有的空快照
    runner._record_drop(decision, snapshot)  # 白盒：直接传入坏快照

    assert runner.drops == ()
    errors = _events_of(events, "error")
    assert errors and errors[0]["where"] == "drop_record"


def test_runner_works_against_a_real_broker_snapshot() -> None:
    clock = FakeClock()
    controller = FakeController()
    broker = _live_broker()
    runner = _runner(clock, controller, broker=broker, judge=FakeJudge())

    _drive(runner, clock, ticks=1)

    assert runner.state is MissionState.RECON
    assert runner.origin is not None and runner.origin.alt_m == ORIGIN.alt_m
    assert len(controller.uploads) == 1


# ----------------------------------------------------------------------
# MissionRunner：HOLD_PROCESS 与无目标 → 备用点
# ----------------------------------------------------------------------
def test_runner_without_target_uses_backup_point() -> None:
    """验收要求的分支：没有结果 → 备用点，走同一套投放判据（Q11）。"""
    clock = FakeClock()
    controller = FakeController()
    judge = FakeJudge([ReleaseDecision(True, "fallback")])
    empty = analyze([], TargetingConfig(eps_m=0.75, min_samples=2))
    assert not empty.ok
    events: list[tuple[str, dict[str, Any]]] = []
    runner = _runner(
        clock,
        controller,
        judge=judge,
        target_result=lambda: empty,
        target_busy=lambda: False,
        events=events,
    )

    _drive(runner, clock, ticks=2)  # 第 1 拍 INIT→RECON，第 2 拍让看门狗看到"任务在跑"
    controller.finish_mission()
    _drive_until(runner, clock, MissionState.OVERFLY)
    _drive(runner, clock, ticks=1)

    assert runner.plan is not None
    assert runner.plan.source == "backup"
    expected = waypoint_to_ned(_config().routes.backup_point, llaref_of(ORIGIN))
    assert judge.targets[0] == pytest.approx(expected)
    payload = _events_of(events, "targeting")[0]
    assert payload["reason"] == "no_target", "处理完且无结果 → 提前结束等待，不硬等满 10s"
    assert payload["elapsed_s"] < 10.0
    assert payload["targeting"]["selected"] is None
    assert _events_of(events, "drop_plan")[0]["source"] == "backup"


def test_runner_waits_full_timeout_while_processing_is_busy() -> None:
    clock = FakeClock()
    controller = FakeController()
    empty = analyze([], TargetingConfig(eps_m=0.75, min_samples=2))
    events: list[tuple[str, dict[str, Any]]] = []
    runner = _runner(
        clock,
        controller,
        judge=FakeJudge([ReleaseDecision(True, "fallback")]),
        target_result=lambda: empty,
        target_busy=lambda: True,  # 永远"还在处理"
        events=events,
    )

    _drive(runner, clock, ticks=2)  # 第 1 拍 INIT→RECON，第 2 拍让看门狗看到"任务在跑"
    controller.finish_mission()
    _drive_until(runner, clock, MissionState.OVERFLY, step=0.25, limit=200)

    payload = _events_of(events, "targeting")[0]
    assert payload["reason"] == "timeout"
    assert payload["elapsed_s"] == pytest.approx(10.0, abs=0.5)
    assert runner.plan is not None and runner.plan.source == "backup"


def test_runner_keeps_waiting_for_a_result_up_to_the_limit() -> None:
    """先没结果、后面才出结果：应当在出结果那一拍就走，而不是等到超时。"""
    clock = FakeClock()
    controller = FakeController()
    good = _labeled_points()
    empty = analyze([], TargetingConfig(eps_m=0.75, min_samples=2))
    ready = {"value": False}
    events: list[tuple[str, dict[str, Any]]] = []
    runner = _runner(
        clock,
        controller,
        judge=FakeJudge([ReleaseDecision(True, "predict")]),
        target_result=lambda: good if ready["value"] else empty,
        target_busy=lambda: True,  # 不给"处理完了"的信号 → 只能等结果或超时
        events=events,
    )

    _drive(runner, clock, ticks=2)  # 第 1 拍 INIT→RECON，第 2 拍让看门狗看到"任务在跑"
    controller.finish_mission()
    _drive_until(runner, clock, MissionState.HOLD_PROCESS)
    _drive(runner, clock, ticks=20, step=0.25)  # 5s：仍然没有结果
    assert runner.state is MissionState.HOLD_PROCESS
    assert _events_of(events, "targeting") == []

    ready["value"] = True
    _drive_until(runner, clock, MissionState.OVERFLY, step=0.25, limit=40)
    assert runner.state is MissionState.OVERFLY
    payload = _events_of(events, "targeting")[0]
    assert payload["reason"] == "result"
    assert payload["targeting"]["selected"]["code"] == 56
    assert runner.plan is not None and runner.plan.source == "target"


def test_runner_survives_a_failing_target_reader() -> None:
    """统计回调抛异常：记一次事件 + 按“暂时没有结果”继续，到点走备用点。"""
    clock = FakeClock()
    controller = FakeController()
    events: list[tuple[str, dict[str, Any]]] = []

    def boom() -> Any:
        raise RuntimeError("聚类抛出异常了")

    runner = _runner(
        clock,
        controller,
        judge=FakeJudge([ReleaseDecision(True, "fallback")]),
        target_result=boom,
        target_busy=lambda: True,
        events=events,
    )
    _drive(runner, clock, ticks=2)  # 第 1 拍 INIT→RECON，第 2 拍让看门狗看到"任务在跑"
    controller.finish_mission()
    _drive_until(runner, clock, MissionState.OVERFLY, step=0.25, limit=200)

    errors = _events_of(events, "error")
    assert len(errors) == 1, "同一种错误只记一次，别每拍刷屏"
    assert errors[0]["where"] == "target_result"
    assert runner.plan is not None and runner.plan.source == "backup"


# ----------------------------------------------------------------------
# MissionRunner：失败路径
# ----------------------------------------------------------------------
def test_runner_aborts_when_upload_fails() -> None:
    clock = FakeClock()
    controller = FakeController()
    controller.fail_on = {"upload_mission"}
    events: list[tuple[str, dict[str, Any]]] = []
    runner = _runner(clock, controller, events=events)

    _drive(runner, clock, ticks=2)

    assert runner.state is MissionState.ABORT
    assert runner.history[-1].reason == "recon_upload_failed"
    assert "start_mission" not in controller.calls
    assert controller.count("hold") == 1, "默认 ABORT 动作是 hold"
    assert runner.stats.aborts == 1 and runner.stats.errors == 1
    assert _events_of(events, "error")[0]["where"] == "recon_upload"


def test_runner_aborts_when_start_fails() -> None:
    clock = FakeClock()
    controller = FakeController()
    controller.fail_on = {"start_mission"}
    runner = _runner(clock, controller)
    _drive(runner, clock, ticks=2)
    assert runner.state is MissionState.ABORT
    assert runner.history[-1].reason == "recon_upload_failed"
    assert len(controller.uploads) == 1


@pytest.mark.parametrize(("action", "expected"), [("rtl", "rtl"), ("none", None)])
def test_runner_abort_action_is_configurable(action: str, expected: str | None) -> None:
    clock = FakeClock()
    controller = FakeController()
    controller.fail_on = {"upload_mission"}
    runner = _runner(
        clock,
        controller,
        config=_config(mission=MissionConfig(abort_action=action, recon_upload="auto")),
    )
    _drive(runner, clock, ticks=2)

    assert runner.state is MissionState.ABORT
    if expected is None:
        assert "rtl" not in controller.calls and "hold" not in controller.calls
    else:
        assert controller.count(expected) == 1
        assert "hold" not in controller.calls


def test_runner_aborts_when_telemetry_goes_stale() -> None:
    clock = FakeClock()
    controller = FakeController()
    events: list[tuple[str, dict[str, Any]]] = []
    runner = _runner(clock, controller, broker=FakeBroker(clock, age_s=30.0), events=events)
    _drive(runner, clock, ticks=2)  # INIT 只看有没有位置，RECON 才看看门狗

    assert runner.state is MissionState.ABORT
    assert runner.history[-1].reason == "telemetry_lost"
    assert "遥测已" in _events_of(events, "error")[0]["message"]
    assert controller.count("hold") == 1, "ABORT 动作仍然要下（链路能不能收到是另一回事）"


def test_runner_aborts_when_telemetry_never_arrives() -> None:
    clock = FakeClock()
    controller = FakeController()
    runner = _runner(
        clock,
        controller,
        config=_config(mission=MissionConfig(init_max_s=1.0)),
        broker=FakeBroker(clock, valid=False),
    )
    _drive(runner, clock, ticks=6, step=0.25)

    assert runner.state is MissionState.ABORT
    assert runner.history[-1].reason == "init_no_telemetry"
    assert "upload_mission" not in controller.calls


def test_runner_waits_for_origin_then_aborts_on_timeout() -> None:
    clock = FakeClock()
    controller = FakeController(origin=None)
    runner = _runner(clock, controller, config=_config(mission=MissionConfig(init_max_s=1.0)))

    _drive(runner, clock, ticks=3, step=0.25)
    assert runner.state is MissionState.INIT, "原点没就绪时应当停在 INIT 反复重试"
    assert controller.count("request_origin") >= 1

    _drive(runner, clock, ticks=4, step=0.25)
    assert runner.state is MissionState.ABORT
    assert runner.history[-1].reason == "init_no_origin"


def test_runner_retries_origin_query_failures_without_crashing() -> None:
    clock = FakeClock()
    controller = FakeController()
    controller.fail_on = {"request_origin"}
    events: list[tuple[str, dict[str, Any]]] = []
    runner = _runner(
        clock,
        controller,
        config=_config(mission=MissionConfig(init_max_s=1.0)),
        events=events,
    )
    _drive(runner, clock, ticks=2, step=0.25)
    assert runner.state is MissionState.INIT, "查询失败只是这一拍没拿到，不能就此失败"

    controller.fail_on.clear()
    _drive(runner, clock, ticks=2, step=0.25)
    assert runner.state is MissionState.RECON


def test_runner_reports_a_repeating_query_failure_only_once() -> None:
    """查询是每拍都调的：持续失败只报一次，否则 20Hz 的日志会把真原因埋掉。"""
    clock = FakeClock()
    controller = FakeController()
    controller.fail_on = {"mission_finished"}
    events: list[tuple[str, dict[str, Any]]] = []
    runner = _runner(
        clock,
        controller,
        config=_config(mission=MissionConfig(recon_max_s=600.0)),
        events=events,
    )
    _drive(runner, clock, ticks=2)  # 进 RECON
    _drive(runner, clock, ticks=20)  # 20 拍全都查失败

    errors = _events_of(events, "error")
    assert len(errors) == 1 and errors[0]["where"] == "mission_finished"
    assert runner.state is MissionState.RECON, "查不到就继续等（状态超时兜底）"


def test_runner_aborts_when_recon_times_out() -> None:
    clock = FakeClock()
    controller = FakeController()  # 永远不报"飞完"
    runner = _runner(clock, controller, config=_config(mission=MissionConfig(recon_max_s=2.0)))
    _drive(runner, clock, ticks=20, step=0.25)

    assert runner.state is MissionState.ABORT
    assert runner.history[-1].reason == "recon_timeout"


def test_runner_does_not_trust_a_stale_finished_flag() -> None:
    """回归：新任务刚上传时飞控可能仍报上一次的 True —— 不能就此判定侦查结束。"""
    clock = FakeClock()
    controller = FakeController(sticky_finished=True)
    controller.finished = True  # 一直回 True，从未出现过 False
    runner = _runner(clock, controller, config=_config(mission=MissionConfig(recon_max_s=2.0)))
    _drive(runner, clock, ticks=20, step=0.25)

    assert runner.state is MissionState.ABORT
    assert runner.history[-1].reason == "recon_timeout"
    assert runner.monitor.armed is False and runner.monitor.false_seen == 0


def test_runner_aborts_when_the_mission_ends_without_a_release() -> None:
    clock = FakeClock()
    controller = FakeController()
    judge = FakeJudge()  # 永远 waiting
    events: list[tuple[str, dict[str, Any]]] = []
    runner = _runner(clock, controller, judge=judge, events=events)

    _drive(runner, clock, ticks=2)  # 第 1 拍 INIT→RECON，第 2 拍让看门狗看到"任务在跑"
    controller.finish_mission()
    _drive_until(runner, clock, MissionState.OVERFLY)
    _drive(runner, clock, ticks=2)  # 飞掠段里跑两拍：判据评估过、且看到任务在跑
    assert judge.updates >= 1

    controller.finish_mission()  # 整条任务飞完却没投出去
    _drive(runner, clock, ticks=3)

    assert runner.state is MissionState.ABORT
    assert runner.history[-1].reason == "overfly_finished_without_release"
    assert controller.count("gripper_release") == 0
    assert runner.stats.releases == 0


def test_runner_aborts_when_gripper_release_is_refused() -> None:
    clock = FakeClock()
    controller = FakeController(release_ok=False)
    runner = _runner(clock, controller, judge=FakeJudge([ReleaseDecision(True, "predict")]))
    _drive(runner, clock, ticks=2)  # 第 1 拍 INIT→RECON，第 2 拍让看门狗看到"任务在跑"
    controller.finish_mission()
    _drive_until(runner, clock, MissionState.OVERFLY)
    _drive(runner, clock, ticks=2)

    assert runner.state is MissionState.ABORT
    assert runner.history[-1].reason == "gripper_release_failed"
    assert runner.stats.releases == 0


def test_runner_aborts_when_planning_fails() -> None:
    """没有备用点、也没出结果：不猜、不飞，直接失败。"""
    clock = FakeClock()
    controller = FakeController()
    events: list[tuple[str, dict[str, Any]]] = []
    runner = _runner(
        clock,
        controller,
        config=_config(backup=False),
        judge=FakeJudge(),
        target_busy=lambda: False,
        events=events,
    )
    _drive(runner, clock, ticks=2)  # 第 1 拍 INIT→RECON，第 2 拍让看门狗看到"任务在跑"
    controller.finish_mission()
    _drive(runner, clock, ticks=20, step=0.25)

    assert runner.state is MissionState.ABORT
    assert runner.history[-1].reason == "plan_drop_failed"
    assert "备用点" in _events_of(events, "error")[0]["message"]
    assert len(controller.uploads) == 1, "只上传过侦查航线"


def test_runner_without_judge_skips_the_drop_and_lands() -> None:
    """演练模式（没有判据）：不投放，但状态机照常收尾到 DONE。"""
    clock = FakeClock()
    controller = FakeController()
    events: list[tuple[str, dict[str, Any]]] = []
    runner = _runner(clock, controller, judge=None, events=events)

    _drive(runner, clock, ticks=2)  # 第 1 拍 INIT→RECON，第 2 拍让看门狗看到"任务在跑"
    controller.finish_mission()
    _drive_until(runner, clock, MissionState.LAND)
    assert runner.state is MissionState.LAND
    assert _events_of(events, "drop_skipped")[0]["reason"] == "no_judge"

    _drive(runner, clock, ticks=1)  # LAND 里先看到一次"任务在跑"
    controller.finish_mission()
    _drive_until(runner, clock, MissionState.DONE)
    assert runner.state is MissionState.DONE
    assert controller.count("gripper_release") == 0


def test_abort_is_idempotent_and_ignored_after_done() -> None:
    clock = FakeClock()
    controller = FakeController()
    runner = _runner(clock, controller)
    runner.abort("operator")
    assert runner.state is MissionState.ABORT
    assert controller.count("hold") == 1
    runner.abort("again")
    assert controller.count("hold") == 1, "已经是终态，不能重复下动作"
    assert runner.stats.aborts == 1

    done_clock = FakeClock()
    done_controller = FakeController()
    done = _runner(done_clock, done_controller, judge=None)  # 演练模式 → 一路到 DONE
    _drive(done, done_clock, ticks=2)
    done_controller.finish_mission()
    _drive_until(done, done_clock, MissionState.LAND)
    _drive(done, done_clock, ticks=1)
    done_controller.finish_mission()
    _drive_until(done, done_clock, MissionState.DONE)

    assert done.state is MissionState.DONE
    done.abort("too late")
    assert done.state is MissionState.DONE, "DONE 之后不该被改成 ABORT"
    assert done.stats.aborts == 0


# ----------------------------------------------------------------------
# DroneController：MAVSDK 插件调用（用假线程 + 假 drone 离线测）
# ----------------------------------------------------------------------
class FakeThread:
    """假 MavsdkThread：``submit`` 直接把协程跑完，``get_snapshot`` 提供任务进度。

    ``mission_progress`` 就是快照里的 ``(mission_current, mission_total)``——
    ``None`` 表示进度流还没数据（真机上这时 ``mission_finished()`` 会显式失败）。
    """

    def __init__(
        self,
        drone: Any,
        *,
        mission_progress: tuple[int, int] | None = None,
        flight_mode: str | None = "MISSION",
    ) -> None:
        self._drone = drone
        self.mission_progress = mission_progress
        self.flight_mode = flight_mode

    def require_drone(self) -> Any:
        return self._drone

    def get_snapshot(self) -> TelemetrySnapshot:
        snapshot = TelemetrySnapshot(timestamp=0.0)
        if self.mission_progress is not None:
            snapshot.mission_current, snapshot.mission_total = self.mission_progress
        snapshot.flight_mode = self.flight_mode
        return snapshot

    def submit(self, coro: Any) -> concurrent.futures.Future:
        future: concurrent.futures.Future = concurrent.futures.Future()
        try:
            future.set_result(asyncio.run(coro))
        except BaseException as exc:  # noqa: BLE001 - 与 run_coroutine_threadsafe 一致：BaseException 一起转交
            future.set_exception(exc)
        return future


class FakeDrone:
    """只包含 DroneController 会用到的插件（mission_raw / mission / action / gripper / telemetry）。

    上传走 ``mission_raw``（原样项），``download_mission`` 用来复现"上传后回读校验"：
    默认回读到的就是刚存下的那份；要模拟"飞控存的不一样"，改 ``drone.stored``。
    """

    def __init__(self) -> None:
        self.uploaded: list[Any] = []
        self.stored: list[Any] = []
        self.calls: list[str] = []
        self.started = 0
        self.holds = 0
        self.rtls = 0
        self.releases: list[int] = []
        self.origin_calls = 0
        self.fail_upload: BaseException | None = None
        self.fail_origin = False
        drone = self

        class _MissionRaw:
            async def upload_mission(self, items: Any) -> None:
                if drone.fail_upload is not None:
                    raise drone.fail_upload
                drone.uploaded.append(list(items))
                drone.stored = list(items)

            async def download_mission(self) -> Any:
                return list(drone.stored)

            async def set_current_mission_item(self, index: int) -> None:
                drone.calls.append("set_current_mission_item:%d" % index)

        class _Mission:
            async def start_mission(self) -> None:
                drone.calls.append("start_mission")
                drone.started += 1

        class _Action:
            async def hold(self) -> None:
                drone.holds += 1

            async def return_to_launch(self) -> None:
                drone.rtls += 1

        class _Gripper:
            async def release(self, instance: int) -> None:
                drone.releases.append(instance)

        class _Telemetry:
            async def get_gps_global_origin(self) -> Any:
                drone.origin_calls += 1
                if drone.fail_origin:
                    raise RuntimeError("原点还没就绪")
                return SimpleNamespace(
                    latitude_deg=ORIGIN.lat_deg,
                    longitude_deg=ORIGIN.lon_deg,
                    altitude_m=ORIGIN.alt_m,
                )

        self.mission_raw = _MissionRaw()
        self.mission = _Mission()
        self.action = _Action()
        self.gripper = _Gripper()
        self.telemetry = _Telemetry()


def _controller(
    drone: FakeDrone | None = None,
    *,
    events: list[tuple[str, dict[str, Any]]] | None = None,
    **kwargs: Any,
) -> tuple[DroneController, FakeDrone]:
    drone = drone or FakeDrone()
    controller = DroneController(
        FakeThread(drone),
        on_event=None if events is None else (lambda kind, data: events.append((kind, data))),
        **kwargs,
    )
    return controller, drone


def test_drone_controller_uploads_the_planned_mission() -> None:
    events: list[tuple[str, dict[str, Any]]] = []
    controller, drone = _controller(events=events)
    items = build_recon_mission(_config(recon=2))

    count = controller.upload_mission(items)
    controller.start_mission()

    assert count == 2 and drone.started == 1
    sent = drone.uploaded[0]
    assert [item.command for item in sent] == [MAV_CMD_NAV_TAKEOFF, MAV_CMD_NAV_WAYPOINT]
    assert sent[0].z == 60.0 and sent[0].x == int(round(47.0 * 1e7))
    assert sent[0].current == 1 and sent[1].current == 0
    payload = _events_of(events, "mission_upload")[0]
    assert payload["count"] == 2
    assert payload["items"][0]["command"] == MAV_CMD_NAV_TAKEOFF
    assert _events_of(events, "mission_start") != []


def test_drone_controller_rejects_empty_mission() -> None:
    controller, drone = _controller()
    with pytest.raises(ControllerError, match="任务项为空"):
        controller.upload_mission([])
    assert drone.uploaded == [], "空任务等于清空飞控上的任务，绝不能静默上传"


def test_drone_controller_translates_mavsdk_failures() -> None:
    drone = FakeDrone()
    drone.fail_upload = RuntimeError("飞控拒绝上传")
    controller, _ = _controller(drone)
    with pytest.raises(ControllerError, match="上传任务"):
        controller.upload_mission(build_recon_mission(_config(recon=1)))


def test_drone_controller_times_out_waiting_for_the_future() -> None:
    class DeadThread:
        def require_drone(self) -> Any:  # pragma: no cover - 不该被调到
            raise AssertionError

        def submit(self, coro: Any) -> concurrent.futures.Future:
            coro.close()
            return concurrent.futures.Future()  # 永远不会有结果

    controller = DroneController(DeadThread(), timeout=0.01)
    with pytest.raises(ControllerError, match="没有结果"):
        controller.hold()


def test_drone_controller_hold_rtl_and_release() -> None:
    sleeps: list[float] = []
    controller, drone = _controller(sleep=sleeps.append, release_settle_s=0.25)

    controller.hold()
    controller.rtl()
    assert controller.gripper_release() is True

    assert (drone.holds, drone.rtls) == (1, 1)
    assert drone.releases == [0], "默认 gripper 实例号是 0"
    assert sleeps == [0.25], "投放后要留出伺服动作时间"


def test_drone_controller_respects_disabled_gripper() -> None:
    events: list[tuple[str, dict[str, Any]]] = []
    controller, drone = _controller(events=events, gripper_enabled=False)

    assert controller.gripper_release() is False
    assert drone.releases == [], "没启用就绝不发指令，也不能假装投了"
    assert _events_of(events, "gripper_skipped")[0]["reason"] == "disabled"


def test_drone_controller_uses_configured_gripper_instance() -> None:
    controller, drone = _controller(release_settle_s=0.0, gripper_instance=2)
    assert controller.gripper_release() is True
    assert drone.releases == [2]


def test_drone_controller_rewinds_the_mission_before_starting() -> None:
    """启动前先 ``set_current_mission_item(0)``——清掉飞控"任务已飞完"的锁存。

    实测（2026-09 SITL，同一飞控同一会话）：内容相同的任务再启动时，不复位会让
    ``start_mission()`` 回成功而模式停在 ``HOLD``；先复位则 100% 进 ``MISSION``。
    """
    controller, drone = _controller()
    controller.upload_mission(build_recon_mission(_config(recon=1)))

    controller.start_mission()

    assert drone.calls == ["set_current_mission_item:0", "start_mission"]


def test_drone_controller_reports_whether_the_vehicle_is_in_mission_mode() -> None:
    """启动确认读的是快照里的飞行模式，不是"命令有没有抛异常"。"""
    drone = FakeDrone()
    thread = FakeThread(drone, flight_mode="HOLD")
    controller = DroneController(thread)
    assert controller.in_mission_mode() is False

    thread.flight_mode = "MISSION"
    assert controller.in_mission_mode() is True

    thread.flight_mode = None
    with pytest.raises(ControllerError, match="飞行模式"):
        controller.in_mission_mode()


def test_drone_controller_origin_and_finished_queries() -> None:
    """完成判定读快照里的任务进度（``current == total`` ⇒ 飞完）。"""
    drone = FakeDrone()
    thread = FakeThread(drone)
    controller = DroneController(thread)

    origin = controller.request_origin()
    assert origin == ORIGIN
    assert drone.origin_calls == 1

    with pytest.raises(ControllerError, match="任务进度"):
        # 进度流还没数据就显式失败，不猜（原来的写法把说明写成了表达式语句，等于没有断言）
        controller.mission_finished()

    thread.mission_progress = (1, 3)
    assert controller.mission_finished() is False
    thread.mission_progress = (3, 3)
    assert controller.mission_finished() is True

    drone.fail_origin = True
    assert controller.request_origin() is None, "原点没就绪只是“还没有”，不是异常"


def test_drone_controller_from_config_reads_gripper_section() -> None:
    drone = FakeDrone()
    sleeps: list[float] = []
    config = _config().replace(gripper=GripperConfig(enabled=False, instance=3))
    controller = DroneController.from_config(config, FakeThread(drone), sleep=sleeps.append)
    assert controller.gripper_release() is False
    assert drone.releases == []


def test_drone_controller_events_do_not_break_control() -> None:
    def boom(kind: str, data: dict[str, Any]) -> None:
        raise RuntimeError("事件日志已关闭")

    controller, drone = _controller(events=None)
    controller = DroneController(FakeThread(drone), on_event=boom)
    assert controller.upload_mission(build_recon_mission(_config(recon=1))) == 1
    controller.start_mission()


# ----------------------------------------------------------------------
# 演练控制器（P10：SITL / 不装弹演练）
# ----------------------------------------------------------------------
def test_dry_run_controller_delegates_everything_but_the_release() -> None:
    inner = FakeController()
    events: list[tuple[str, dict[str, Any]]] = []
    dry = DryRunController(inner, on_event=lambda kind, data: events.append((kind, data)))

    assert dry.upload_mission(build_recon_mission(_config(recon=1))) == 1
    dry.start_mission()
    dry.hold()
    dry.rtl()
    assert dry.request_origin() == ORIGIN
    assert dry.mission_finished() is False

    assert dry.gripper_release() is True, "演练里算“投出去了”，状态机才能继续"
    assert dry.releases == 1

    assert inner.calls == [
        "upload_mission",
        "start_mission",
        "hold",
        "rtl",
        "request_origin",
        "mission_finished",
    ], "除投放外全部委托给真控制器"
    assert _events_of(events, "drop_dry_run")[0]["count"] == 1


def test_dry_run_controller_swallows_event_failures() -> None:
    def boom(kind: str, data: dict[str, Any]) -> None:
        raise RuntimeError("EventLog 已关闭")

    dry = DryRunController(FakeController(), on_event=boom)
    assert dry.gripper_release() is True, "事件写盘失败不能让演练中断"


def test_runner_reaches_done_with_a_dry_run_controller() -> None:
    """演练控制器接进状态机：整条链路走完，真控制器一次投放都没收到。"""
    clock = FakeClock()
    inner = FakeController()
    dry = DryRunController(inner)
    runner = _runner(clock, dry, judge=FakeJudge([ReleaseDecision(True, "predict")]))

    _drive(runner, clock, ticks=2)
    inner.finish_mission()
    _drive_until(runner, clock, MissionState.OVERFLY)
    _drive_until(runner, clock, MissionState.LAND)
    assert runner.state is MissionState.LAND
    assert dry.releases == 1

    _drive(runner, clock, ticks=1)
    inner.finish_mission()
    _drive_until(runner, clock, MissionState.DONE)

    assert runner.state is MissionState.DONE
    assert "gripper_release" not in inner.calls, "演练模式绝不能把投放指令发到飞控"


# ----------------------------------------------------------------------
# MissionRunner：主循环
# ----------------------------------------------------------------------
class TimedController(FakeController):
    """按假时钟自动推进"任务进度"的假控制器（把分钟级任务压成毫秒级）。"""

    def __init__(self, clock: FakeClock, *, recon_s: float = 60.0, drop_s: float = 120.0) -> None:
        super().__init__()
        self._clock = clock
        self._recon_s = recon_s
        self._drop_s = drop_s
        self._started_at: float | None = None

    def start_mission(self) -> None:
        super().start_mission()
        self._started_at = self._clock()

    def mission_finished(self) -> bool:
        self._record("mission_finished")
        if self._started_at is None:
            return False
        limit = self._recon_s if len(self.uploads) == 1 else self._drop_s
        return (self._clock() - self._started_at) >= limit


def test_run_loop_finishes_the_whole_mission_on_a_fake_clock() -> None:
    clock = FakeClock()
    controller = TimedController(clock, recon_s=60.0, drop_s=120.0)
    runner = _runner(
        clock,
        controller,
        config=_config(mission=MissionConfig(tick_hz=5.0, recon_upload="auto")),
        judge=FakeJudge([ReleaseDecision(True, "predict")]),
        target_result=_labeled_points,
        target_busy=lambda: False,
    )
    started = clock.now

    state = runner.run(max_ticks=5000)

    assert state is MissionState.DONE
    assert runner.stats.uploads == 2 and runner.stats.releases == 1
    assert runner.stats.aborts == 0
    assert runner.plan is not None and runner.plan.source == "target"
    # 假时间推进 ≈ 侦查 60s + 处理 + 飞掠/降落 120s（节拍 5Hz → 每拍 0.2s）
    elapsed = clock.now - started
    assert 180.0 <= elapsed <= 200.0


def test_stop_aborts_the_mission_and_leaves_the_loop() -> None:
    clock = FakeClock()
    controller = TimedController(clock, recon_s=60.0)
    stopped = {"value": False}

    def sleep(seconds: float) -> None:
        clock.advance(seconds)
        if not stopped["value"]:
            stopped["value"] = True
            runner.stop("operator_stop")

    runner = _runner(
        clock,
        controller,
        config=_config(mission=MissionConfig(tick_hz=5.0)),
        sleep=sleep,
    )
    state = runner.run(max_ticks=100)

    assert state is MissionState.ABORT
    assert runner.stopped
    assert runner.history[-1].reason == "operator_stop"
    assert controller.count("hold") == 1
