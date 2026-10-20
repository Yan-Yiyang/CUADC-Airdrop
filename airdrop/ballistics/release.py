"""投放判据：实时预测航向过零投放；过零后距离兜底（计划 4.6）。

三层判据（都在同一个入口里，实飞与回放走同一条路）：

1. 预测判据：预测落点相对目标的沿航向误差 ``s`` 连续从负值穿过零，且横向误差
   ``abs(c) <= radius_m`` 时投放；
2. 近靶兜底：本轮已观测到预测沿向过零后，当前有效预测落点进入 ``post_cross_radius_m``
   水平半径内时投放；
3. 距离兜底：本轮已观测到预测沿向过零后，当前有效预测 ``s > fallback_after_m`` 时投放。
距离兜底由 ``force_after_pass`` 控制；近靶兜底始终启用。所有投放判断均不依赖飞机瞬时位置。

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
    #: 兼容保留的飞机位置诊断：本次飞掠里是否出现过飞机投影位于目标前方；不参与投放判据。
    approached: bool = False
    #: 前推 ``delay_s`` 之后用于预测的位置（NED，米）
    release_position: tuple[float, float, float] | None = None
    #: 这一拍用的前推量（秒）
    delay_s: float = 0.0
    #: 预测落点相对目标的沿航向误差（米），正值表示已越过目标断面。
    along_track_error_m: float | None = None
    #: 预测落点相对目标的横向误差（米）。
    cross_track_error_m: float | None = None
    #: 本轮是否已观测预测沿向误差从负值过零到非负；首样为零不锁存。
    prediction_crossed_target: bool = False

    def as_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "should_release": self.should_release,
            "reason": self.reason,
            "passed_target": self.passed_target,
            "approached": self.approached,
            "prediction_crossed_target": self.prediction_crossed_target,
            "delay_s": float(self.delay_s),
        }
        if self.horizontal_error_m is not None:
            data["horizontal_error_m"] = round(self.horizontal_error_m, 3)
        if self.along_track_error_m is not None:
            data["along_track_error_m"] = round(self.along_track_error_m, 3)
        if self.cross_track_error_m is not None:
            data["cross_track_error_m"] = round(self.cross_track_error_m, 3)
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
    _prediction_context: tuple[object, ...] | None = field(default=None, init=False)
    _prediction_prev_s: float | None = field(default=None, init=False)
    _prediction_prev_time: float | None = field(default=None, init=False)
    _prediction_crossed: bool = field(default=False, init=False)
    #: 飞机位置诊断字段，保留既有语义；不参与新的预测兜底。
    _approached: bool = field(default=False, init=False)

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
        self._clear_prediction_track()
        self._approached = False

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
            self._clear_prediction_track()
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

        effective_ground_z = self.ground_z if ground_z is None else ground_z
        effective_ground_altitude = (
            self.ground_altitude_m if ground_altitude_m is None else ground_altitude_m
        )
        wind = self._resolve_wind(snapshot, wind_ned)
        inputs = (
            *release_position,
            *velocity,
            *target_ned,
            self.overfly_heading_deg,
            snapshot.position_velocity_received_at,
            snapshot.timestamp,
            snapshot.origin_latitude_deg,
            snapshot.origin_longitude_deg,
            snapshot.origin_altitude_m,
        )
        if not all(value is None or math.isfinite(float(value)) for value in inputs) or not all(
            math.isfinite(float(value))
            for value in (*wind, effective_ground_z, effective_ground_altitude)
        ):
            self._clear_prediction_track()
            return self._decide(
                ReleaseDecision(
                    False,
                    "no_prediction:non_finite_input",
                    stamp,
                    passed_target=False,
                    approached=self._approached,
                    release_position=release_position,
                    delay_s=delay,
                ),
                stamp,
            )
        impact = self.ballistics.predict_impact(
            release_position,
            velocity,
            ground_z=effective_ground_z,
            ground_altitude_m=effective_ground_altitude,
            wind=wind,
        )

        target = (float(target_ned[0]), float(target_ned[1]))
        error = None
        along_error = None
        cross_error = None
        crossed_now = False
        sample_fresh = False
        if impact.ok and impact.ned is not None:
            delta_n = impact.ned[0] - target[0]
            delta_e = impact.ned[1] - target[1]
            error = math.hypot(delta_n, delta_e)
            north, east = heading_unit_vector(self.overfly_heading_deg)
            along_error = delta_n * north + delta_e * east
            cross_error = delta_n * -east + delta_e * north
            sample_fresh, crossed_now = self._observe_prediction(
                snapshot,
                target_ned=target_ned,
                ground_z=effective_ground_z,
                ground_altitude_m=effective_ground_altitude,
                along_error=along_error,
                velocity=velocity,
                now=stamp,
            )
        else:
            self._clear_prediction_track()

        # 既有飞机位置诊断维持兼容，仅供 passed_target / approached 解释。
        north, east = heading_unit_vector(self.overfly_heading_deg)
        passed = (
            (release_position[0] - target[0]) * north + (release_position[1] - target[1]) * east
        ) > 0.0
        if not passed:
            self._approached = True
        # 正常判据优先：只在首次连续过零的有效新样本上检查横向容限。
        if (
            sample_fresh
            and crossed_now
            and cross_error is not None
            and abs(cross_error) <= self.config.radius_m
        ):
            return self._release(
                "predict",
                stamp,
                impact,
                error,
                passed,
                release_position,
                delay,
                along_error,
                cross_error,
            )
        if (
            sample_fresh
            and self._prediction_crossed
            and error is not None
            and error <= self.config.post_cross_radius_m
        ):
            return self._release(
                "near_target",
                stamp,
                impact,
                error,
                passed,
                release_position,
                delay,
                along_error,
                cross_error,
            )
        if (
            sample_fresh
            and self.config.force_after_pass
            and self._prediction_crossed
            and along_error is not None
            and along_error > self.config.fallback_after_m
        ):
            return self._release(
                "fallback",
                stamp,
                impact,
                error,
                passed,
                release_position,
                delay,
                along_error,
                cross_error,
            )

        reason = "waiting" if impact.ok else f"no_prediction:{impact.reason}"
        return self._decide(
            ReleaseDecision(
                should_release=False,
                reason=reason,
                timestamp=stamp,
                predicted=impact,
                horizontal_error_m=error,
                passed_target=passed,
                approached=self._approached,
                release_position=release_position,
                delay_s=delay,
                along_track_error_m=along_error,
                cross_track_error_m=cross_error,
                prediction_crossed_target=self._prediction_crossed,
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

    def _clear_prediction_track(self) -> None:
        """清除沿向过零历史，避免跨任务、断档或无效预测复用旧样本。"""
        self._prediction_context = None
        self._prediction_prev_s = None
        self._prediction_prev_time = None
        self._prediction_crossed = False

    def _observe_prediction(
        self,
        snapshot: TelemetrySnapshot,
        *,
        target_ned: Sequence[float],
        ground_z: float,
        ground_altitude_m: float,
        along_error: float,
        velocity: tuple[float, float, float],
        now: float,
    ) -> tuple[bool, bool]:
        """记录连续新鲜预测，并返回（新样本，当前拍首次过零）。"""
        north, east = heading_unit_vector(self.overfly_heading_deg)
        speed_along = velocity[0] * north + velocity[1] * east
        received = snapshot.position_velocity_received_at
        timestamp = snapshot.timestamp if received is None else received
        values = (
            timestamp,
            now,
            along_error,
            speed_along,
            ground_z,
            ground_altitude_m,
            *target_ned,
            snapshot.origin_latitude_deg,
            snapshot.origin_longitude_deg,
            snapshot.origin_altitude_m,
            self.overfly_heading_deg,
        )
        if not all(value is None or math.isfinite(float(value)) for value in values):
            self._clear_prediction_track()
            return False, False
        timestamp = float(timestamp)
        now = float(now)
        if not 0.0 <= now - timestamp <= 0.5 or speed_along <= 0.0:
            self._clear_prediction_track()
            return False, False

        context = (
            *(float(v) for v in target_ned),
            snapshot.origin_latitude_deg,
            snapshot.origin_longitude_deg,
            snapshot.origin_altitude_m,
            self.overfly_heading_deg,
            float(ground_z),
            float(ground_altitude_m),
        )
        if context != self._prediction_context:
            self._clear_prediction_track()
            self._prediction_context = context

        previous_s = self._prediction_prev_s
        previous_time = self._prediction_prev_time
        if previous_time is not None:
            interval = timestamp - previous_time
            if interval == 0.0:
                # 重复的新鲜位置速度样本不消费负向前样本，也不触发判据。
                return False, False
            if interval < 0.0 or interval > 0.5:
                self._clear_prediction_track()
                self._prediction_context = context
                previous_s = None

        true_crossing = previous_s is not None and previous_s < 0.0 and along_error >= -1e-9
        crossed_now = true_crossing or (previous_s is None and abs(along_error) <= 1e-9)
        if true_crossing:
            self._prediction_crossed = True
        self._prediction_prev_s = along_error
        self._prediction_prev_time = timestamp
        return True, crossed_now

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
        along_error: float | None = None,
        cross_error: float | None = None,
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
            approached=self._approached,
            release_position=release_position,
            delay_s=delay,
            along_track_error_m=along_error,
            cross_track_error_m=cross_error,
            prediction_crossed_target=self._prediction_crossed,
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
        payload["fallback_after_m"] = self.config.fallback_after_m
        payload["post_cross_radius_m"] = self.config.post_cross_radius_m
        payload["evaluated_at"] = stamp
        try:
            self.on_event(kind, payload)
        except Exception:
            LOGGER.exception("写入投放事件失败（判据继续）")
