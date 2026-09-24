"""投放判据：实时预测落点 → 触发投放；飞过目标后强制投放（计划 4.6）。

两条判据（都在同一个入口里，实飞与回放走同一条路）：

1. 预测判据：把飞机状态前推 ``delay_s``（补偿"指令→离机"延迟），用
   :class:`~airdrop.ballistics.model.BallisticsModel` 预测落点，落点与目标的水平距离
   ≤ ``radius_m`` 就投；
2. 强制投放判据（越过目标后）：``dot(飞机位置 − 目标, 飞掠航向) > 0``——已经沿飞掠方向越过目标却还没投，
   立刻投。目标点与备用点用同一套判据。

一次投放即锁存（一次性防抖）：之后任何调用都只回 ``already_released``，绝不重复投。

日志与事件
----------
判据按调用方的节拍评估（``DropConfig.evaluation_hz``，默认 20Hz），但摘要按 5Hz 落
事件日志（``summary_hz``）——20Hz 全量写盘只会把事件日志淹没，而 5Hz 足够复盘。
触发那一瞬间写完整预测（落点/飞行时间/水平误差/前推后的状态）。
``on_event`` 的签名是 ``(kind, data)``，接到 recorder 上是
``lambda kind, data: recorder.events.emit(kind, **data)``。
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..config import DropConfig
from ..telemetry.models import TelemetrySnapshot
from .model import ZERO_WIND, BallisticsModel, Impact, as_float_triple, wind_from_snapshot

LOGGER = logging.getLogger(__name__)

__all__ = ["ReleaseDecision", "ReleaseJudge", "heading_unit_vector"]

#: 摘要进事件日志的最高频率（Hz）——判据本身按调用方节拍跑
SUMMARY_HZ = 5.0


def heading_unit_vector(heading_deg: float) -> tuple[float, float]:
    """罗盘航向（自北顺时针，度）→ NED 水平单位矢量 ``(north, east)``。"""
    radians = math.radians(float(heading_deg))
    return (math.cos(radians), math.sin(radians))


@dataclass(frozen=True, slots=True)
class ReleaseDecision:
    """一次判据评估的结果。``reason`` 说明为什么投/不投。

    ``delay_s`` 是这一拍用的状态前推量（``DropConfig.delay_s``）。它必须跟着决策
    一起走：投放记录（:class:`~airdrop.ballistics.drops.DropRecord`）要用判据当时
    真正用的值，否则事后反演会把"延迟填错"算到阻力系数头上。
    """

    should_release: bool
    reason: str  # predict / fallback / waiting / already_released / ...
    timestamp: float = 0.0
    predicted: Impact | None = None
    horizontal_error_m: float | None = None
    passed_target: bool = False
    #: 前推 ``delay_s`` 之后用于预测的位置（NED，米）
    release_position: tuple[float, float, float] | None = None
    #: 这一拍用的前推量（秒）
    delay_s: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "should_release": self.should_release,
            "reason": self.reason,
            "passed_target": self.passed_target,
            "delay_s": float(self.delay_s),
        }
        if self.horizontal_error_m is not None:
            data["horizontal_error_m"] = round(self.horizontal_error_m, 3)
        if self.release_position is not None:
            data["release_position"] = [round(v, 3) for v in self.release_position]
        if self.predicted is not None and self.predicted.ok:
            data["impact_ned"] = [round(v, 3) for v in self.predicted.ned or ()]
            data["flight_time_s"] = round(self.predicted.flight_time_s, 3)
            data["impact_speed_m_s"] = round(self.predicted.speed_m_s, 3)
        return data


@dataclass
class ReleaseJudge:
    """投放判据（每拍调一次 :meth:`update`）。"""

    config: DropConfig
    ballistics: BallisticsModel
    overfly_heading_deg: float
    ground_z: float = 0.0
    #: 地面平面的海拔（AMSL，GPS 原点/地面点）——密度的基准；0 = 未接（地面当海平面）
    ground_altitude_m: float = 0.0
    #: 显式风矢量（NED，m/s）；None 表示**每个节拍按 ``BallisticsConfig.wind_source``
    #: 从快照取**（取不到则零风降级并记一次日志）。
    wind_ned: tuple[float, float, float] | None = None
    on_event: Callable[[str, dict[str, Any]], None] | None = None
    summary_hz: float = SUMMARY_HZ

    _released: bool = field(default=False, init=False)
    _last_summary: float = field(default=0.0, init=False)
    _evaluations: int = field(default=0, init=False)
    _wind_warned: bool = field(default=False, init=False)

    # ------------------------------------------------------------------
    @property
    def released(self) -> bool:
        return self._released

    @property
    def evaluations(self) -> int:
        """已评估次数（回放/演练复盘用）。"""
        return self._evaluations

    def reset(self) -> None:
        """清除锁存（换目标/换架次时用）。"""
        self._released = False
        self._last_summary = 0.0
        self._evaluations = 0

    # ------------------------------------------------------------------
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
        """评估一次；``target_ned`` 是目标（或备用点）的 NED 坐标。

        ``ground_altitude_m`` 是地面海拔（密度基准，见
        :meth:`BallisticsModel.predict_impact`）；不给就用构造时的值。
        """
        stamp = time.time() if now is None else float(now)
        self._evaluations += 1

        if self._released:
            return ReleaseDecision(False, "already_released", stamp)

        position = self._position(snapshot)
        velocity = self._velocity(snapshot)
        if position is None or velocity is None:
            return self._decide(ReleaseDecision(False, "unknown_position", stamp), stamp)
        if len(target_ned) != 3:
            raise ValueError("target_ned 必须是 3 维 NED 矢量")

        # 1) 状态前推 delay_s：一阶近似（位置 += 速度×delay）。
        #    这里刻意不做二次项——延迟是几十毫秒量级，加速度项贡献在厘米级以下。
        delay = max(float(self.config.delay_s), 0.0)
        release_position: tuple[float, float, float] = (
            position[0] + velocity[0] * delay,
            position[1] + velocity[1] * delay,
            position[2] + velocity[2] * delay,
        )

        wind = self._resolve_wind(snapshot, wind_ned)
        impact = self.ballistics.predict_impact(
            release_position,
            velocity,
            ground_z=self.ground_z if ground_z is None else ground_z,
            ground_altitude_m=(
                self.ground_altitude_m if ground_altitude_m is None else ground_altitude_m
            ),
            wind=wind,
        )

        target = (float(target_ned[0]), float(target_ned[1]))
        error = None
        if impact.ok and impact.ned is not None:
            error = math.hypot(impact.ned[0] - target[0], impact.ned[1] - target[1])

        # 2) 强制投放判据：沿飞掠航向已经越过目标
        north, east = heading_unit_vector(self.overfly_heading_deg)
        passed = (
            (release_position[0] - target[0]) * north + (release_position[1] - target[1]) * east
        ) > 0.0

        if error is not None and error <= self.config.radius_m:
            return self._release("predict", stamp, impact, error, passed, release_position, delay)
        if passed and self.config.force_after_pass:
            return self._release("fallback", stamp, impact, error, passed, release_position, delay)

        reason = "waiting" if impact.ok else f"no_prediction:{impact.reason}"
        return self._decide(
            ReleaseDecision(
                should_release=False,
                reason=reason,
                timestamp=stamp,
                predicted=impact,
                horizontal_error_m=error,
                passed_target=passed,
                release_position=release_position,
                delay_s=delay,
            ),
            stamp,
        )

    # ------------------------------------------------------------------
    def _resolve_wind(
        self,
        snapshot: TelemetrySnapshot,
        override: Sequence[float] | None,
    ) -> tuple[float, float, float]:
        """这一拍用哪个风：显式入参 > 快照（按 ``wind_source``）> 零风降级。

        "没有风估计"只记一次日志（20Hz 每拍都记会把日志淹没）。
        """
        if override is not None:
            return (float(override[0]), float(override[1]), float(override[2]))
        if self.wind_ned is not None:
            return self.wind_ned
        wind = wind_from_snapshot(snapshot, self.ballistics.config)
        if wind is not None:
            return wind
        if not self._wind_warned:
            self._wind_warned = True
            LOGGER.warning(
                "遥测里没有风估计（wind_source=%s），本次投放按零风预测——"
                "风会平移落点，需要风补偿时先确认飞控的风估计流可用",
                self.ballistics.config.wind_source,
            )
        return ZERO_WIND

    def _position(self, snapshot: TelemetrySnapshot) -> tuple[float, float, float] | None:
        return as_float_triple((snapshot.north_m, snapshot.east_m, snapshot.down_m))

    def _velocity(self, snapshot: TelemetrySnapshot) -> tuple[float, float, float] | None:
        return as_float_triple((snapshot.vx_m_s, snapshot.vy_m_s, snapshot.vz_m_s))

    def _release(  # noqa: PLR0917 - 内部方法，参数顺序与 ReleaseDecision 字段一致
        self,
        reason: str,
        stamp: float,
        impact: Impact,
        error: float | None,
        passed: bool,
        release_position: tuple[float, float, float],
        delay: float = 0.0,
    ) -> ReleaseDecision:
        """锁存并记录完整预测。"""
        self._released = True
        decision = ReleaseDecision(
            should_release=True,
            reason=reason,
            timestamp=stamp,
            predicted=impact,
            horizontal_error_m=error,
            passed_target=passed,
            release_position=release_position,
            delay_s=delay,
        )
        LOGGER.warning(
            "投放触发（%s）：落点误差 %s m，飞行时间 %.2fs，前推后位置 (%.2f, %.2f, %.2f)",
            reason,
            "未知" if error is None else f"{error:.2f}",
            impact.flight_time_s,
            *release_position,
        )
        self._emit("release", decision, stamp)
        return decision

    def _decide(self, decision: ReleaseDecision, stamp: float) -> ReleaseDecision:
        self._maybe_summary(decision, stamp)
        return decision

    def _maybe_summary(self, decision: ReleaseDecision, stamp: float) -> None:
        """按 ``summary_hz`` 节流写摘要（判据本身仍在每次调用都算）。"""
        interval = 1.0 / self.summary_hz if self.summary_hz > 0 else 0.0
        if interval > 0.0 and (stamp - self._last_summary) < interval:
            return
        self._last_summary = stamp
        self._emit("drop_check", decision, stamp)

    def _emit(self, kind: str, decision: ReleaseDecision, stamp: float) -> None:
        if self.on_event is None:
            return
        payload = decision.as_dict()
        payload["overfly_heading_deg"] = self.overfly_heading_deg
        payload["radius_m"] = self.config.radius_m
        payload["evaluated_at"] = stamp
        try:
            self.on_event(kind, payload)
        except Exception:
            LOGGER.exception("写入投放事件失败（判据继续）")
