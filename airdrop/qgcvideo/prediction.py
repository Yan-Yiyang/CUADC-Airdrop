"""QGC 专用的只读实时弹道预测邮箱。"""

from __future__ import annotations

import copy
import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace

from ..ballistics.model import BallisticsModel, Impact, as_float_triple, wind_from_snapshot
from ..config import Config
from ..telemetry.models import TelemetrySnapshot

MAX_INPUT_AGE_S = 0.5
MAX_RESULT_AGE_S = 0.5
MAX_COMPUTE_S = 0.25


@dataclass(frozen=True, slots=True)
class LiveImpactPrediction:
    """一次按当前整份遥测状态计算的显示预测。"""

    valid: bool
    reason: str
    impact_ned: tuple[float, float, float] | None
    error_m: float | None
    flight_time_s: float | None
    position_velocity_received_at: float | None
    computed_mono: float
    generation: int
    wind_available: bool
    command_delay_s: float
    input_position_ned: tuple[float, float, float] | None
    input_velocity_ned: tuple[float, float, float] | None
    serial: int = 0
    lifecycle_token: int = 0


@dataclass(frozen=True, slots=True)
class _Request:
    snapshot: TelemetrySnapshot
    target_ned: tuple[float, float, float] | None
    stage: str | None
    generation: int
    serial: int = 0
    lifecycle_token: int = 0
    cancelled: threading.Event = field(default_factory=threading.Event)


class LiveImpactPredictor:
    """以 10Hz 单槽计算预测；新请求覆盖旧请求，不积累队列。"""

    def __init__(self, config: Config, *, clock: Callable[[], float] = time.time) -> None:
        self._config = config
        self._model = BallisticsModel(config.ballistics)
        self._clock = clock
        self._condition = threading.Condition()
        self._request: _Request | None = None
        self._result: LiveImpactPrediction | None = None
        self._signature: tuple[object, ...] | None = None
        self._generation = 0
        self._serial = 0
        self._active_cancel: threading.Event | None = None
        self._computing_cancel: threading.Event | None = None
        self._running = False
        self._stopping = False
        self._cancel = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        with self._condition:
            if self._running:
                if self._stopping:
                    raise RuntimeError("QGC 实时弹道线程仍在收尾，不能重复启动")
                return
            self._stopping = False
            self._cancel.clear()
            self._generation += 1
            self._signature = None
            thread = threading.Thread(target=self._run, name="qgc-live-impact", daemon=False)
            self._thread = thread
            try:
                thread.start()
            except BaseException:
                self._thread = None
                raise
            self._running = True

    def stop(self, timeout: float | None = 2.0) -> None:
        with self._condition:
            self._stopping = True
            self._cancel.set()
            if self._active_cancel is not None:
                self._active_cancel.set()
            if self._computing_cancel is not None:
                self._computing_cancel.set()
            self._request = None
            self._generation += 1
            self._signature = None
            thread = self._thread
            self._condition.notify_all()
        if thread is not None:
            thread.join(timeout=max(float(timeout or 0.0), 0.0))
        with self._condition:
            if thread is not None and thread.is_alive():
                raise TimeoutError("QGC 实时弹道计算线程未能在限定时间内结束")
            self._thread = None
            self._running = False
            self._result = None

    def submit(
        self,
        snapshot: TelemetrySnapshot,
        target_ned: tuple[float, float, float] | None,
        stage: str | None,
        lifecycle_token: int = 0,
    ) -> int:
        """复制输入并覆盖单槽；停止后拒绝接收。"""
        with self._condition:
            if not self._running or self._stopping:
                return self._generation
            origin = snapshot.origin_altitude_m
            signature = (
                target_ned,
                snapshot.origin_latitude_deg,
                snapshot.origin_longitude_deg,
                origin,
                stage,
                lifecycle_token,
            )
            changed = signature != self._signature
            if changed:
                self._generation += 1
                self._signature = signature
                self._result = None
            self._serial += 1
            copied = copy.copy(snapshot)
            quick_reason = self._quick_invalid(copied, target_ned, stage)
            if changed or quick_reason:
                if self._active_cancel is not None:
                    self._active_cancel.set()
                if self._request is not None:
                    self._request.cancelled.set()
                if self._computing_cancel is not None:
                    self._computing_cancel.set()
                cancellation = threading.Event()
            else:
                cancellation = threading.Event()
            self._active_cancel = cancellation
            request = _Request(
                copied,
                target_ned,
                stage,
                self._generation,
                self._serial,
                lifecycle_token,
                cancellation,
            )
            if quick_reason:
                self._generation += 1
                request = replace(request, generation=self._generation)
                self._result = _invalid_for_request(request, quick_reason)
                self._request = None
                self._condition.notify()
                return self._generation
            if changed:
                self._result = None
            self._request = request
            self._condition.notify()
            return self._generation

    def current(self) -> LiveImpactPrediction | None:
        """只返回当前代、输入与计算均未过期的结果。"""
        with self._condition:
            result = self._result
            generation = self._generation
        if result is None or result.generation != generation:
            return None
        age = self._clock() - (result.position_velocity_received_at or math.inf)
        compute_age = time.monotonic() - result.computed_mono
        if not math.isfinite(age) or age < 0 or age > MAX_INPUT_AGE_S:
            return replace(result, valid=False, reason="位置速度遥测已过期", impact_ned=None)
        if not math.isfinite(compute_age) or compute_age < 0 or compute_age > MAX_RESULT_AGE_S:
            return replace(result, valid=False, reason="预测结果已过期", impact_ned=None)
        return result

    def _quick_invalid(
        self,
        snapshot: TelemetrySnapshot,
        target_ned: tuple[float, float, float] | None,
        stage: str | None,
    ) -> str:
        reason = _quick_invalid_reason(snapshot, self._clock())
        if reason:
            return reason
        if stage not in {"OVERFLY", "LAND"}:
            return "尚未进入投放航线"
        if (
            target_ned is None
            or len(target_ned) != 3
            or not all(math.isfinite(value) for value in target_ned)
        ):
            return "目标坐标未知"
        origin_altitude = snapshot.origin_altitude_m
        if origin_altitude is None:
            return "NED 原点海拔未知"
        ground_z = self._config.ground.ground_z(origin_altitude)
        if ground_z is None or not math.isfinite(ground_z):
            return "地面高程未知"
        wind = wind_from_snapshot(snapshot, self._config.ballistics)
        if wind is not None and not all(math.isfinite(value) for value in wind):
            return "风矢量包含非有限值"
        return ""

    def _run(self) -> None:
        deadline = time.monotonic()
        while True:
            with self._condition:
                while not self._stopping:
                    if self._request is None:
                        self._condition.wait(timeout=0.1)
                    else:
                        remaining = deadline - time.monotonic()
                        if remaining > 0:
                            self._condition.wait(timeout=remaining)
                        else:
                            break
                if self._stopping:
                    return
                request = self._request
                self._request = None
                self._computing_cancel = None if request is None else request.cancelled
            if request is None:
                continue
            result = self._compute(request)
            with self._condition:
                self._computing_cancel = None
                if not self._stopping and request.generation == self._generation:
                    self._result = result
            deadline = time.monotonic() + 0.1

    def _compute(self, request: _Request) -> LiveImpactPrediction:
        result = self._compute_impl(request)
        return LiveImpactPrediction(
            result.valid,
            result.reason,
            result.impact_ned,
            result.error_m,
            result.flight_time_s,
            result.position_velocity_received_at,
            result.computed_mono,
            result.generation,
            result.wind_available,
            result.command_delay_s,
            result.input_position_ned,
            result.input_velocity_ned,
            request.serial,
            request.lifecycle_token,
        )

    def _compute_impl(self, request: _Request) -> LiveImpactPrediction:  # noqa: PLR0911
        snapshot = request.snapshot
        received = snapshot.position_velocity_received_at
        now = self._clock()
        if request.stage not in {"OVERFLY", "LAND"}:
            return _invalid("尚未进入投放航线", request.generation, received)
        if snapshot.in_air is not True:
            return _invalid("飞行状态未知或未离地", request.generation, received)
        if (
            received is None
            or not math.isfinite(received)
            or now < received
            or now - received > MAX_INPUT_AGE_S
        ):
            return _invalid("位置速度遥测已过期或时间无效", request.generation, received)
        position = as_float_triple((snapshot.north_m, snapshot.east_m, snapshot.down_m))
        velocity = as_float_triple((snapshot.vx_m_s, snapshot.vy_m_s, snapshot.vz_m_s))
        origin_altitude = snapshot.origin_altitude_m
        if position is None or velocity is None or origin_altitude is None:
            return _invalid("位置、速度或 NED 原点海拔未知", request.generation, received)
        values = (*position, *velocity, origin_altitude)
        if not all(math.isfinite(value) for value in values):
            return _invalid("位置、速度或原点包含非有限值", request.generation, received)
        ground_z = self._config.ground.ground_z(origin_altitude)
        if ground_z is None or not math.isfinite(ground_z):
            return _invalid("地面高程未知", request.generation, received)
        target = request.target_ned
        if target is None or len(target) != 3 or not all(math.isfinite(value) for value in target):
            return _invalid("目标坐标未知", request.generation, received)
        wind = wind_from_snapshot(snapshot, self._config.ballistics)
        wind_available = wind is not None
        if wind is None:
            wind = (0.0, 0.0, 0.0)
        if not all(math.isfinite(value) for value in wind):
            return _invalid("风矢量包含非有限值", request.generation, received)
        delay = max(float(self._config.drop.delay_s), 0.0)
        start = tuple(position[i] + velocity[i] * delay for i in range(3))
        started = time.monotonic()
        deadline = started + MAX_COMPUTE_S

        def should_cancel() -> bool:
            return (
                self._cancel.is_set() or request.cancelled.is_set() or time.monotonic() >= deadline
            )

        try:
            impact: Impact = self._model.predict_impact(
                start,
                velocity,
                ground_z=ground_z,
                ground_altitude_m=origin_altitude - ground_z,
                wind=wind,
                should_cancel=should_cancel,
            )
        except ArithmeticError, OverflowError, ValueError:
            return _invalid("弹道计算输入无效", request.generation, received)
        computed = time.monotonic()
        if not impact.ok or impact.ned is None:
            if impact.reason == "cancelled":
                return _invalid("弹道计算超出显示时限或已取消", request.generation, received)
            return _invalid("弹道模型无有效落点", request.generation, received)
        if not all(math.isfinite(value) for value in impact.ned) or not math.isfinite(
            impact.flight_time_s
        ):
            return _invalid("弹道模型返回非有限结果", request.generation, received)
        error = math.hypot(impact.ned[0] - target[0], impact.ned[1] - target[1])
        return LiveImpactPrediction(
            True,
            "",
            impact.ned,
            error,
            impact.flight_time_s,
            received,
            computed,
            request.generation,
            wind_available,
            delay,
            position,
            velocity,
            request.serial,
            request.lifecycle_token,
        )


def _invalid(
    reason: str,
    generation: int,
    received: float | None,
) -> LiveImpactPrediction:
    return LiveImpactPrediction(
        False,
        reason,
        None,
        None,
        None,
        received,
        time.monotonic(),
        generation,
        False,
        0.0,
        None,
        None,
    )


def _invalid_for_request(request: _Request, reason: str) -> LiveImpactPrediction:
    result = _invalid(reason, request.generation, request.snapshot.position_velocity_received_at)
    return LiveImpactPrediction(
        result.valid,
        result.reason,
        result.impact_ned,
        result.error_m,
        result.flight_time_s,
        result.position_velocity_received_at,
        result.computed_mono,
        result.generation,
        result.wind_available,
        result.command_delay_s,
        result.input_position_ned,
        result.input_velocity_ned,
        request.serial,
        request.lifecycle_token,
    )


def _quick_invalid_reason(snapshot: TelemetrySnapshot, now: float) -> str:
    received = snapshot.position_velocity_received_at
    if snapshot.in_air is not True:
        return "飞行状态未知或未离地"
    if (
        received is None
        or not math.isfinite(received)
        or now < received
        or now - received > MAX_INPUT_AGE_S
    ):
        return "位置速度遥测已过期或时间无效"
    position = (snapshot.north_m, snapshot.east_m, snapshot.down_m)
    velocity = (snapshot.vx_m_s, snapshot.vy_m_s, snapshot.vz_m_s)
    if any(value is None for value in (*position, *velocity, snapshot.origin_altitude_m)):
        return "位置、速度或 NED 原点海拔未知"
    if not all(
        math.isfinite(value)
        for value in (*position, *velocity, snapshot.origin_altitude_m)
        if value is not None
    ):
        return "位置、速度或原点包含非有限值"
    return ""
