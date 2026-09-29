"""任务主循环：驱动控制器、监视遥测、推进状态机（计划 4.7）。

一条完整任务（Q/计划 1 章）::

    INIT      等遥测 + NED 原点 → 上传并启动侦察航线
    RECON     侦察航线在飞（约 1 分钟），mission 报飞完为止
    HOLD_PROCESS  hold()（固定翼盘旋）→ 等 targeting 出结果，上限 10s
                  ├─ 有结果 → 目标 NED
                  └─ 无结果 → 备用点（Q11：走同一套弹道判据）
    OVERFLY   上传"飞掠段 + 降落航线"（一条任务，Q13）并启动 → 激活投放判据
                  ├─ 判据触发 → gripper 投放 → LAND
                  └─ 任务飞完仍未投放 → ABORT（显式失败，不静默收工）
    LAND      沿降落航线返航；mission 报飞完 → DONE
    ABORT     下 MissionConfig.abort_action（默认 hold）

设计取舍
--------
* 不自己起线程：:meth:`MissionRunner.run` 是普通循环，调用方决定跑在哪个线程
  （示例里是主线程 + Ctrl-C）；:meth:`update` 可以单拍驱动，测试才可能确定性地跑。
* 面向协议编程：控制器只要求满足 :class:`~airdrop.telemetry.MissionController`
  （离线测试用假控制器），投放判据只要求 :class:`ReleaseJudgeLike`。
* 完成判定读**飞控侧任务进度**（``mission_raw`` 的 ``mission_current == mission_total``，
  经 :meth:`~airdrop.telemetry.DroneController.mission_finished`），不是自己解析
  MAVLink 进度流；这是一条"飞控状态读数"，新任务刚上传、还没启动时可能仍是上一次
  任务的 ``True`` → 由 :class:`MissionMonitor` 要求"先见到 False 再见到 True"，
  避免刚起飞就以为任务飞完了。
* 失败必须显式：命令失败、规划失败、链路陈旧、超时 —— 一律进 ``ABORT`` 并写
  事件日志；不会"假装成功继续跑"。
* 只有两个时钟来源：注入的 ``clock``（时间）与 ``sleep``（节拍），测试里换成
  假时钟即可把分钟级的任务压成毫秒级跑完。
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from ..ballistics.drops import DropRecord
from ..ballistics.model import wind_from_snapshot
from ..ballistics.release import ReleaseDecision
from ..config import Config
from ..georef import LLARef
from ..preflight import PreflightError, PreflightLike
from ..targeting.models import TargetingResult
from ..telemetry.broker import TelemetryBroker
from ..telemetry.controller import (
    ControllerError,
    MissionController,
    NedOrigin,
)
from ..telemetry.models import TelemetrySnapshot
from .items import MissionItem
from .planner import (
    DropMissionPlan,
    PlanningError,
    build_drop_mission,
    build_recon_mission,
    llaref_of,
)
from .states import MissionState, MissionStateMachine, MissionTransition, emit_event

LOGGER = logging.getLogger(__name__)

__all__ = [
    "MissionMonitor",
    "MissionRunner",
    "MissionStats",
    "PreflightLike",
    "ReleaseJudgeLike",
    "TargetResultReader",
]


@runtime_checkable
class ReleaseJudgeLike(Protocol):
    """投放判据（实飞用 :class:`~airdrop.ballistics.ReleaseJudge`，测试用假判据）。"""

    def update(
        self,
        snapshot: TelemetrySnapshot,
        target_ned: Sequence[float],
        *,
        ground_z: float | None = None,
        ground_altitude_m: float | None = None,
        wind_ned: Sequence[float] | None = None,
        now: float | None = None,
    ) -> ReleaseDecision: ...

    def reset(self) -> None: ...


#: HOLD_PROCESS 里读"当前统计结果"的回调（analyze(points, config.targeting) 之类）
TargetResultReader = Callable[[], TargetingResult]


#: 一拍内允许连跳的"瞬时"状态：条件当场满足就没有必要占满一拍
_INSTANT_STATES = frozenset({MissionState.PREFLIGHT, MissionState.WAIT_AIRBORNE})


@dataclass
class MissionMonitor:
    """跟踪一次上传任务的完成情况，对"已飞完"的陈旧读数免疫。

    背景：完成判定读的是飞控侧任务进度（``mission_raw`` 的 ``mission_current ==
    mission_total``，见 :meth:`~airdrop.telemetry.DroneController.mission_finished`），
    它是**飞控状态读数**——新任务刚上传、还没跑起来的那一拍，可能仍是上一次任务留下的
    ``True``。判据：先看到 ``False``（任务确实在跑），之后的 ``True`` 才算数。
    每上传一次任务换一个新实例（正常任务都是分钟级，绝不会整段时间都观测不到 ``False``）。
    """

    armed: bool = False
    false_seen: int = 0
    true_seen: int = 0

    def update(self, finished: bool) -> bool:
        """传入一拍任务完成读数，返回"这次任务是否已完成"。"""
        if not finished:
            self.armed = True
            self.false_seen += 1
            return False
        self.true_seen += 1
        return self.armed

    def as_dict(self) -> dict[str, Any]:
        return {
            "armed": self.armed,
            "false_seen": self.false_seen,
            "true_seen": self.true_seen,
        }


@dataclass
class MissionStats:
    """任务计数（复盘与测试断言用）。"""

    ticks: int = 0
    uploads: int = 0
    releases: int = 0
    aborts: int = 0
    errors: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "ticks": self.ticks,
            "uploads": self.uploads,
            "releases": self.releases,
            "aborts": self.aborts,
            "errors": self.errors,
        }


class MissionRunner:
    """状态机主循环（单拍 :meth:`update` / 连续 :meth:`run`）。

    参数
    ----
    config:
        全局配置（用到 ``mission`` / ``routes`` / ``overfly`` / ``ground`` / ``drop``）。
    controller:
        :class:`~airdrop.telemetry.MissionController`——实飞是
        :class:`~airdrop.telemetry.DroneController`，测试是假控制器。
    broker:
        遥测代理；状态机从它取快照（位置/时间戳）。
    target_result:
        HOLD_PROCESS 里读"当前统计结果"的回调，返回
        :class:`~airdrop.targeting.TargetingResult`；``None`` 表示这架次没有统计来源
        （到点直接走备用点）。
    target_busy:
        "感知/坐标解算还有没处理的帧"的回调；给了它，处理完且无结果时可以提前
        结束等待，而不是硬等满 10s。
    release_judge:
        投放判据；``None`` 表示不自动投放（演练/手动投），OVERFLY 会直接进 LAND。
    on_drop:
        投放记录回调——指令被飞控接受之后调一次，拿到的是
        :class:`~airdrop.ballistics.drops.DropRecord`（投放瞬间的位置/速度/姿态/风、
        目标点、判据的前推位置与预测落点）。这条数据事后无法重建，所以要写入磁盘
        （实飞接 ``recorder.drops.append``），也是投放试验反演弹道参数的唯一输入。
        回调抛异常只记日志——投放已经发生了，不能因为写不进记录就改变任务状态。
    on_event:
        ``(kind, data)`` 事件回调（状态转移、投放、规划、错误……）。
    clock / sleep:
        注入的时间源与节拍（测试用假时钟）。
    """

    def __init__(  # noqa: PLR0913 - 依赖全部按协议注入，关键字参数是刻意的可读性
        self,
        config: Config,
        controller: MissionController,
        broker: TelemetryBroker,
        *,
        target_result: TargetResultReader | None = None,
        target_busy: Callable[[], bool] | None = None,
        release_judge: ReleaseJudgeLike | None = None,
        on_event: Callable[[str, dict[str, Any]], None] | None = None,
        on_drop: Callable[[DropRecord], None] | None = None,
        preflight: PreflightLike | None = None,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._config = config
        self._controller = controller
        self._broker = broker
        self._target_result = target_result
        self._target_busy = target_busy
        self._judge = release_judge
        self._on_event = on_event
        self._on_drop = on_drop
        self._preflight = preflight
        self._clock = clock
        self._sleep = sleep

        self._machine = MissionStateMachine(on_event=on_event)
        self._stats = MissionStats()
        self._stop = threading.Event()
        self._monitor = MissionMonitor()

        self._origin: LLARef | None = None
        self._ned_origin: NedOrigin | None = None
        self._plan: DropMissionPlan | None = None
        self._target: TargetingResult | None = None
        self._drops: list[DropRecord] = []

        self._entered_at = float(self._clock())
        self._deadline: float | None = None
        self._ground_warned = False
        self._target_error_logged = False
        self._query_reported: set[str] = set()
        #: 启动确认：``True`` = 已经确认"任务真的在跑"（或模式流不可用、退化为不确认）
        self._start_confirmed = True
        self._start_deadline: float | None = None
        #: 起飞前自检（载入模型 + 视频自检，见 airdrop.preflight）
        self._preflight = preflight

    # ------------------------------------------------------------------
    # 只读视图
    # ------------------------------------------------------------------
    @property
    def state(self) -> MissionState:
        return self._machine.state

    @property
    def machine(self) -> MissionStateMachine:
        return self._machine

    @property
    def history(self) -> tuple[MissionTransition, ...]:
        """状态转移历史（按发生顺序）。"""
        return tuple(self._machine.history)

    @property
    def stats(self) -> MissionStats:
        return self._stats

    @property
    def origin(self) -> LLARef | None:
        return self._origin

    @property
    def plan(self) -> DropMissionPlan | None:
        return self._plan

    @property
    def targeting_result(self) -> TargetingResult | None:
        return self._target

    @property
    def monitor(self) -> MissionMonitor:
        return self._monitor

    @property
    def drops(self) -> tuple[DropRecord, ...]:
        """本架次已记下的投放记录（顺序即投放顺序）。"""
        return tuple(self._drops)

    @property
    def stopped(self) -> bool:
        return self._stop.is_set()

    # ------------------------------------------------------------------
    # 主循环
    # ------------------------------------------------------------------
    def run(self, *, max_ticks: int | None = None) -> MissionState:
        """按 ``MissionConfig.tick_hz`` 推进，直到终态 / :meth:`stop` / 拍数上限。

        ``max_ticks`` 只是兜底（避免调用方写错时死循环），正常任务不会用到。
        """
        period = 1.0 / float(self._config.mission.tick_hz)
        ticks = 0
        while not self._machine.is_terminal and not self._stop.is_set():
            self.update()
            ticks += 1
            if self._machine.is_terminal or self._stop.is_set():
                break
            if max_ticks is not None and ticks >= max_ticks:
                break
            self._sleep(period)
        return self._machine.state

    def stop(self, reason: str = "stopped") -> None:
        """请求退出主循环；任务未结束就按 ``ABORT`` 处理（含 ``abort_action``）。"""
        self._stop.set()
        self.abort(reason)

    def update(self, now: float | None = None) -> MissionState:
        """推进一步（主循环里每拍调一次；测试里手动调）。

        ``PREFLIGHT`` / ``WAIT_AIRBORNE`` 是瞬时状态：条件当场就满足时（预检跳过、
        飞机已经在空中）允许在同一拍里连续转移，于是"已经起飞"的离线用例仍是一拍进
        ``RECON``；但每一次转移都照常记进历史与事件（不会把两跳藏起来）。
        """
        stamp = float(self._clock()) if now is None else float(now)
        self._stats.ticks += 1
        for _ in range(3):
            state = self._machine.state
            if state is MissionState.INIT:
                self._tick_init(stamp)
            elif state is MissionState.PREFLIGHT:
                self._tick_preflight(stamp)
            elif state is MissionState.WAIT_AIRBORNE:
                self._tick_airborne(stamp)
            elif state is MissionState.RECON:
                self._tick_recon(stamp)
            elif state is MissionState.HOLD_PROCESS:
                self._tick_hold(stamp)
            elif state is MissionState.OVERFLY:
                self._tick_overfly(stamp)
            elif state is MissionState.LAND:
                self._tick_land(stamp)
            if self._machine.is_terminal or self._machine.state is state:
                break
            if self._machine.state not in _INSTANT_STATES:
                break
        return self._machine.state

    def abort(self, reason: str) -> None:
        """进入 ``ABORT``（终态时是空操作）；操作手/外部故障也可以直接调。"""
        if self._machine.is_terminal:
            return
        stamp = float(self._clock())
        self._transition(MissionState.ABORT, reason, stamp)

    # ------------------------------------------------------------------
    # 各状态的一拍
    # ------------------------------------------------------------------
    def _tick_init(self, now: float) -> None:
        """``INIT``：等遥测与 NED 原点（不上传任何任务）→ ``PREFLIGHT``。"""
        snapshot = self._broker.get_snapshot()
        if not self._has_position(snapshot):
            if self._timed_out(now, self._config.mission.init_max_s):
                self.abort("init_no_telemetry")
            return
        if self._origin is None:
            origin = self._query(self._controller.request_origin, "request_origin", default=None)
            if origin is None:
                if self._timed_out(now, self._config.mission.init_max_s):
                    self.abort("init_no_origin")
                return
            self._ned_origin = origin
            self._origin = llaref_of(origin)
            self._emit(
                "origin",
                {
                    "lat_deg": origin.lat_deg,
                    "lon_deg": origin.lon_deg,
                    "alt_m": origin.alt_m,
                },
            )
        self._transition(MissionState.PREFLIGHT, "telemetry_ready", now)

    def _tick_preflight(self, now: float) -> None:
        """``PREFLIGHT``：载入模型 + 视频自检（一次性，见 :mod:`airdrop.preflight`）。

        * 没有注入预检（离线测试）→ 记一条 ``preflight_skipped`` 事件后放过；
        * 任一项失败（异常或 ``ok=False``）→ ``ABORT("preflight_failed:<check>")``；
        * 整体超过 ``PreflightConfig.max_s`` → ``ABORT("preflight_timeout")``。
        """
        if self._preflight is None:
            self._emit(
                "preflight_skipped",
                {"reason": "没有注入预检（离线测试/演练）"},
            )
            LOGGER.warning("没有注入起飞前自检：跳过（正式入口必须传 preflight）")
            self._transition(MissionState.WAIT_AIRBORNE, "preflight_skipped", now)
            return
        started = float(self._clock())
        try:
            checks = self._preflight.run()
        except PreflightError as exc:
            self._preflight_failed(str(exc).split(":")[0].strip() or "unknown", exc)
            return
        except Exception as exc:  # noqa: BLE001 - 预检自己不抛别的，防御性兜底
            self._preflight_failed("unknown", exc)
            return
        failed = [check for check in checks if check.ok is False]
        if failed:
            self._preflight_failed(
                failed[0].name,
                PreflightError(f"{failed[0].name}: {failed[0].detail}"),
            )
            return
        elapsed = float(self._clock()) - started
        if elapsed > float(self._config.preflight.max_s):
            self._stats.errors += 1
            LOGGER.error("起飞前自检超时：%.1fs", elapsed)
            self._emit(
                "error",
                {
                    "module": "mission",
                    "where": "preflight:timeout",
                    "message": f"预检用了 {elapsed:.1f}s，超过上限 {self._config.preflight.max_s:.1f}s",
                },
            )
            self.abort("preflight_timeout")
            return
        self._transition(MissionState.WAIT_AIRBORNE, "preflight_ok", now)

    def _preflight_failed(self, check: str, exc: BaseException) -> None:
        """预检失败：记错误事件与计数，然后 ``ABORT("preflight_failed:<check>")``。"""
        self._stats.errors += 1
        LOGGER.error("起飞前自检失败（%s）：%s", check, exc)
        self._emit(
            "error",
            {"module": "mission", "where": f"preflight:{check}", "message": str(exc)},
        )
        self.abort(f"preflight_failed:{check}")

    def _tick_airborne(self, now: float) -> None:
        """``WAIT_AIRBORNE``：什么都不下发，等飞机真的在空中再进侦察。

        在停机坪上就进侦察等于让飞机在地面执行任务：PX4 会在起飞前就
        开始"追"第一个航点，或者干脆因为任务不可行而盘旋。

        ``MissionConfig.require_airborne=False`` 是地面演练/测试的临时放行开关：
        该检查立即放行，但绝不静默——记一条 ``airborne_skipped`` 事件加一条
        WARNING 日志（正式任务必须保持 ``True``，否则飞机在停机坪上就会开始侦察）。
        """
        if not self._config.mission.require_airborne:
            reason = "require_airborne=False（仅地面演练与离线测试使用，正式任务必须为 True）"
            self._emit("airborne_skipped", {"reason": reason})
            LOGGER.warning(
                "跳过等待起飞（require_airborne=False）：%s；本架次飞机未起飞也会进入侦察",
                reason,
            )
            self._begin_recon(now, "airborne_skipped")
            return
        if not self._link_ok(now):
            return
        snapshot = self._broker.get_snapshot()
        source = self._airborne(snapshot)
        if source is not None:
            self._emit("airborne", {"source": source})
            LOGGER.info("已检测到飞机在空中（判据：%s），进入侦察", source)
            self._begin_recon(now, "airborne")
            return
        if self._timed_out(now, self._config.mission.airborne_timeout_s):
            self.abort("airborne_timeout")

    def _airborne(self, snapshot: TelemetrySnapshot) -> str | None:
        """在空中吗？返回判据名（``in_air`` / ``altitude``）或 ``None``（还没起飞）。

        以遥测 ``in_air`` 为主；它取不到（可选流未就绪）时用
        ``relative_altitude_m >= MissionConfig.airborne_alt_m`` 兜底。
        """
        if snapshot.in_air is not None:
            return "in_air" if snapshot.in_air else None
        altitude = snapshot.relative_altitude_m
        if altitude is not None and float(altitude) >= float(self._config.mission.airborne_alt_m):
            return "altitude"
        return None

    def _begin_recon(self, now: float, reason: str) -> None:
        """进入侦察：``recon_upload="auto"`` 自己上传并启动，``"operator"`` 等操作手启动。

        正式任务走 ``operator``：侦察航线由操作手在 QGC 里上传并启动，本包只等它开始、
        然后监视进度（进度是飞控侧状态，谁上传的都一样能读到）。``auto`` 只用于自动测试。
        """
        flags = self._config.mission
        if flags.recon_upload == "auto":
            try:
                items = build_recon_mission(self._config)
            except PlanningError as exc:
                self._fail("plan_recon", exc)
                return
            if not self._upload_and_start(items, "recon"):
                return
        else:
            self._emit(
                "recon_waiting_operator",
                {"note": "等操作手在 QGC 上传并启动侦察航线"},
            )
            LOGGER.info("侦察航线由操作手上传（mission.recon_upload=operator）：等任务启动后再监视")
            self._start_confirmed = False
            self._start_deadline = now + float(self._config.mission.recon_max_s)
        self._monitor = MissionMonitor()
        self._transition(MissionState.RECON, reason, now)

    def _tick_recon(self, now: float) -> None:
        if not self._link_ok(now):
            return
        if not self._confirm_started(now):
            return
        finished = self._query(self._controller.mission_finished, "mission_finished", default=False)
        if self._monitor.update(bool(finished)):
            self._transition(MissionState.HOLD_PROCESS, "recon_finished", now)
            return
        if self._timed_out(now, self._config.mission.recon_max_s):
            self.abort("recon_timeout")

    def _tick_hold(self, now: float) -> None:
        if not self._link_ok(now):
            return
        result = self._read_result()
        elapsed = now - self._entered_at
        if result is not None and result.ok:
            self._finish_hold(now, result, "result")
            return
        if elapsed >= self._config.mission.hold_process_max_s:
            self._finish_hold(now, result, "timeout")
            return
        if elapsed >= self._config.mission.hold_process_min_s and not self._busy():
            self._finish_hold(now, result, "no_target")

    def _tick_overfly(self, now: float) -> None:
        if not self._link_ok(now):
            return
        if not self._confirm_started(now):
            return
        if self._judge is not None and self._plan is not None:
            snapshot = self._broker.get_snapshot()
            try:
                decision = self._judge.update(
                    snapshot,
                    self._plan.target_ned,
                    ground_z=self._ground_z(),
                    ground_altitude_m=self._ground_altitude(),
                    now=now,
                )
            except Exception as exc:  # noqa: BLE001 - 判据异常按任务失败处理，不让主循环带崩
                self._fail("release_judge", exc)
                return
            if decision.should_release:
                # 把判据看到的那份快照一起带过去：投放记录记的是判据用的同一份状态，
                # 免得记录与决策之间又插进一拍遥测（那会让残差归因错位）。
                self._release(now, decision, snapshot)
                return
        finished = self._query(self._controller.mission_finished, "mission_finished", default=False)
        if self._monitor.update(bool(finished)):
            # 飞掠+降落整条任务都飞完了却没投出去 —— 这是失败，不是"顺利完成"
            self.abort("overfly_finished_without_release")
            return
        if self._deadline is not None and now >= self._deadline:
            self.abort("overfly_timeout")

    def _tick_land(self, now: float) -> None:
        if not self._link_ok(now):
            return
        finished = self._query(self._controller.mission_finished, "mission_finished", default=False)
        if self._monitor.update(bool(finished)):
            self._transition(MissionState.DONE, "mission_finished", now)
            return
        if self._deadline is not None and now >= self._deadline:
            self.abort("land_timeout")

    # ------------------------------------------------------------------
    # HOLD_PROCESS → OVERFLY
    # ------------------------------------------------------------------
    def _finish_hold(self, now: float, result: TargetingResult | None, reason: str) -> None:
        """结束等待：有结果用结果，没有就用备用点（Q11）。"""
        self._target = result
        payload: dict[str, Any] = {
            "reason": reason,
            "elapsed_s": round(now - self._entered_at, 3),
            "targeting": None if result is None else result.as_dict(),
        }
        self._emit("targeting", payload)
        target_ned = result.ned if (result is not None and result.ok) else None
        try:
            plan = build_drop_mission(self._config, origin=self._origin, target_ned=target_ned)
        except PlanningError as exc:
            self._fail("plan_drop", exc)
            return
        if not self._upload_and_start(plan.items, "drop"):
            return
        self._plan = plan
        self._monitor = MissionMonitor()
        self._emit("drop_plan", plan.as_dict())
        self._transition(MissionState.OVERFLY, f"drop_mission_{plan.source}", now)

    def _release(
        self,
        now: float,
        decision: ReleaseDecision,
        snapshot: TelemetrySnapshot | None = None,
    ) -> None:
        """判据触发了：发投放指令，成功则进 LAND（并记下投放瞬间的飞机状态）。"""
        payload = dict(decision.as_dict())
        payload["target_ned"] = (
            None if self._plan is None else [round(v, 3) for v in self._plan.target_ned]
        )
        payload["target_source"] = None if self._plan is None else self._plan.source
        self._emit("drop", payload)
        try:
            released = self._controller.gripper_release()
        except ControllerError as exc:
            self._fail("gripper_release", exc)
            return
        if not released:
            self._fail(
                "gripper_release",
                ControllerError("投放指令没有发出（gripper 未启用或飞控拒绝）"),
            )
            return
        self._stats.releases += 1
        # 投放已经发生了，从这里往后任何失败都只记日志：记录写不进去不能改变
        # 任务状态（真实世界里的弹已经出去了）。
        self._record_drop(
            decision,
            snapshot if snapshot is not None else self._broker.get_snapshot(),
        )
        self._transition(MissionState.LAND, f"released_{decision.reason}", now)

    def _record_drop(
        self,
        decision: ReleaseDecision,
        snapshot: TelemetrySnapshot,
    ) -> None:
        """记下投放瞬间的飞机状态（位置/速度/姿态/风）+ 判据的预测。

        动作失败/快照缺字段都只记日志与事件——见 :meth:`_release` 的注释。
        """
        try:
            record = DropRecord.from_snapshot(
                snapshot,
                index=len(self._drops) + 1,
                ground_z=self._ground_z(),
                wind_ned=self._resolve_wind(snapshot),
                origin=self._origin,
                target_ned=None if self._plan is None else self._plan.target_ned,
                reason=decision.reason,
                delay_s=float(decision.delay_s),
                release_position_ned=decision.release_position,
                predicted_impact_ned=(
                    None if decision.predicted is None else decision.predicted.ned
                ),
                predicted_flight_time_s=(
                    None if decision.predicted is None else decision.predicted.flight_time_s
                ),
                predicted_error_m=decision.horizontal_error_m,
                ballistics=self._config.ballistics,
            )
        except Exception as exc:
            LOGGER.exception("投放记录失败（任务继续，投放已发生）")
            self._emit(
                "error",
                {
                    "module": "mission",
                    "where": "drop_record",
                    "message": f"{type(exc).__name__}: {exc}",
                },
            )
            return
        self._drops.append(record)
        self._emit("drop_record", record.as_dict())
        if self._on_drop is not None:
            try:
                self._on_drop(record)
            except Exception:
                LOGGER.exception("投放记录回调失败（任务继续）")

    def _resolve_wind(self, snapshot: TelemetrySnapshot) -> tuple[float, float, float] | None:
        """记录里存的风：取不到就存 None（不写 0，见 ballistics.model 的说明）。"""
        return wind_from_snapshot(snapshot, self._config.ballistics)

    # ------------------------------------------------------------------
    # 状态进入时的动作
    # ------------------------------------------------------------------
    def _on_enter(self, state: MissionState, now: float) -> None:
        if state is MissionState.HOLD_PROCESS:
            # 等待上限按"进入本状态的时刻"算（见 _tick_hold 的 elapsed）
            self._command(self._controller.hold, "hold")
        elif state is MissionState.OVERFLY:
            # 飞掠与降落同属一条上传的任务，共用这个兜底上限
            self._deadline = now + float(self._config.mission.land_max_s)
            if self._judge is not None:
                self._judge.reset()
            else:
                LOGGER.warning("没有投放判据（release_judge=None）：本架次只飞航线、不自动投放")
                self._emit("drop_skipped", {"reason": "no_judge"})
                self._transition(MissionState.LAND, "no_release_judge", now)
        elif state is MissionState.ABORT:
            self._stats.aborts += 1
            self._do_abort_action()

    def _do_abort_action(self) -> None:
        """ABORT 时下的安全动作（``MissionConfig.abort_action``）。"""
        action = self._config.mission.abort_action
        if action == "none":
            self._emit("abort_action", {"action": "none"})
            return
        call = self._controller.rtl if action == "rtl" else self._controller.hold
        try:
            call()
        except ControllerError as exc:
            # 链路断了时这条指令本来就发不出去，交给飞控自己的 failsafe
            LOGGER.error("ABORT 动作（%s）失败：%s", action, exc)
            self._emit(
                "error",
                {"module": "mission", "where": f"abort_{action}", "message": str(exc)},
            )
            return
        self._emit("abort_action", {"action": action})

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------
    def _transition(self, to_state: MissionState, reason: str, now: float) -> None:
        self._machine.transition(to_state, reason=reason, now=now)
        self._entered_at = now
        self._on_enter(to_state, now)

    def _upload_and_start(self, items: Sequence[MissionItem], leg: str) -> bool:
        """上传并启动一条任务；失败返回 False（已进 ABORT）。"""
        try:
            count = self._controller.upload_mission(items)
            self._controller.start_mission()
        except ControllerError as exc:
            self._fail(f"{leg}_upload", exc)
            return False
        self._stats.uploads += 1
        # 启动回成功 ≠ 飞控真的进了任务模式 —— 交给 _confirm_started 在有限时间内确认
        self._start_confirmed = False
        self._start_deadline = float(self._clock()) + float(
            self._config.mission.mission_start_timeout_s
        )
        LOGGER.info("%s 任务已上传并启动：%d 个任务项", leg, count)
        return True

    def _confirm_started(self, now: float) -> bool:
        """确认"任务真的在跑"：飞控进了任务模式。

        为什么必须有这一步：``mission.start_mission()`` 回成功不等于进了任务模式
        ——飞控可能拒绝模式切换却仍然回 ACK，飞机继续盘旋。这里给它一个短的确认窗口
        （``MissionConfig.mission_start_timeout_s``），超时按显式失败处理。

        模式流不可用时（``flight_mode`` 为 ``None``）退化为跳过确认并只记一次日志：
        宁可跳过确认，也不要因为取不到模式就把任务判失败。
        """
        if self._start_confirmed:
            return True
        try:
            in_mission = bool(self._controller.in_mission_mode())
        except ControllerError as exc:
            if "flight_mode" not in self._query_reported:
                self._query_reported.add("flight_mode")
                LOGGER.warning("读不到飞行模式（%s），本次不确认任务是否真的启动", exc)
            self._start_confirmed = True
            self._start_deadline = None
            return True
        if in_mission:
            self._start_confirmed = True
            self._start_deadline = None
            self._emit("mission_confirmed", {"mode": "MISSION"})
            return True
        if self._start_deadline is not None and now >= self._start_deadline:
            self.abort("mission_not_started")
            return False
        return False

    def _command(self, call: Callable[[], Any], where: str) -> Any | None:
        """执行一条控制命令；失败进 ABORT 并返回 None。"""
        try:
            return call()
        except ControllerError as exc:
            self._fail(where, exc)
            return None

    def _query(self, call: Callable[[], Any], where: str, *, default: Any) -> Any:
        """查询类调用：失败只记日志并返回 ``default``。

        链路真断了会被 :meth:`_link_ok` 的看门狗抓出来（遥测陈旧 → ABORT），
        所以这里不必再叠一层计数——状态自身的超时上限是最后的兜底。

        ⚠ 失败按 ``where`` 只报一次：查询是每拍都调的（20Hz），
        一个持续失败（比如飞控一直拒绝回 ``is_mission_finished``）会把日志和
        事件刷爆，反而盖住真正的原因。
        """
        try:
            return call()
        except ControllerError as exc:
            if where not in self._query_reported:
                self._query_reported.add(where)
                LOGGER.warning("%s 查询失败：%s", where, exc)
                self._emit("error", {"module": "mission", "where": where, "message": str(exc)})
            else:
                LOGGER.debug("%s 查询仍然失败：%s", where, exc)
            return default

    def _fail(self, where: str, exc: BaseException) -> None:
        """命令/规划失败：记事件 + 计数 + 进 ABORT（不吞）。"""
        self._stats.errors += 1
        LOGGER.error("任务失败（%s）：%s", where, exc)
        self._emit("error", {"module": "mission", "where": where, "message": str(exc)})
        self.abort(f"{where}_failed")

    def _link_ok(self, now: float) -> bool:
        """链路看门狗：遥测无效或陈旧到超过 ``telemetry_stale_s`` → ABORT。"""
        snapshot = self._broker.get_snapshot()
        if snapshot.is_valid():
            age = now - float(snapshot.timestamp)
            if age <= float(self._config.mission.telemetry_stale_s):
                return True
            message = f"遥测已 {age:.1f}s 未更新"
        else:
            message = "遥测无效（从未收到有效快照）"
        LOGGER.error("链路异常：%s", message)
        self._emit("error", {"module": "mission", "where": "telemetry", "message": message})
        self.abort("telemetry_lost")
        return False

    def _read_result(self) -> TargetingResult | None:
        """读当前统计结果；没有来源或读取失败都按"暂时没有结果"处理。"""
        if self._target_result is None:
            return None
        try:
            return self._target_result()
        except Exception as exc:
            if not self._target_error_logged:
                self._target_error_logged = True
                self._stats.errors += 1
                LOGGER.exception("读取目标统计结果失败（按“暂时没有结果”处理）")
                self._emit(
                    "error",
                    {
                        "module": "mission",
                        "where": "target_result",
                        "message": f"{type(exc).__name__}: {exc}",
                    },
                )
            return None

    def _busy(self) -> bool:
        """感知/坐标解算是否还有没消费完的帧（没有这个信号时按"处理完了"看待）。"""
        if self._target_busy is None:
            return False
        try:
            return bool(self._target_busy())
        except Exception:  # noqa: BLE001 - 目标回调异常一律视为「还没好」
            return False

    def _ground_z(self) -> float:
        alt = None if self._ned_origin is None else self._ned_origin.alt_m
        ground = self._config.ground.ground_z(alt)
        if ground is None:
            if not self._ground_warned:
                self._ground_warned = True
                LOGGER.warning(
                    "地面点参数不全（GroundConfig），投放判据按 ground_z=0（原点高度面）预测"
                )
            return 0.0
        return float(ground)

    def _ground_altitude(self) -> float:
        """地面平面的**海拔**（AMSL）——弹道密度的基准（见 ``predict_impact``）。

        有地面点参数时就是地面点海拔（``origin.alt − ground_z``）；没配地面点时退到
        NED 原点海拔（原点高度面，通常离目标不远，误差可接受）；连原点都还没有（不该
        出现在 OVERFLY）时返回 0，即旧的"地面当海平面"行为。
        """
        if self._ned_origin is None:
            return 0.0
        ground = self._config.ground.ground_z(self._ned_origin.alt_m)
        if ground is None:
            return float(self._ned_origin.alt_m)
        return float(self._ned_origin.alt_m) - float(ground)

    @staticmethod
    def _has_position(snapshot: TelemetrySnapshot) -> bool:
        return all(
            value is not None for value in (snapshot.north_m, snapshot.east_m, snapshot.down_m)
        )

    def _timed_out(self, now: float, limit: float) -> bool:
        return (now - self._entered_at) >= float(limit)

    def _emit(self, kind: str, data: dict[str, Any]) -> None:
        emit_event(self._on_event, kind, data)
