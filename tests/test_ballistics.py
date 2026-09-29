"""P8 离线单测：弹道积分（解析解对照 / 终端速度 / 风平移）与投放判据。

全部离线：纯数值与假遥测快照，不碰飞控。
验收口径来自计划 P8：零阻力抛物线对照、终端速度渐近、风平移、强制投放触发。
"""

from __future__ import annotations

import math

import pytest

from airdrop.ballistics import (
    BallisticsModel,
    ReleaseJudge,
    heading_unit_vector,
    isa_air_density,
)
from airdrop.config import BallisticsConfig, DropConfig
from airdrop.telemetry.models import TelemetrySnapshot

GRAVITY = 9.80665


def _model(**overrides) -> BallisticsModel:
    return BallisticsModel(BallisticsConfig(**overrides))


def _snapshot(
    position: tuple[float, float, float] = (0.0, 0.0, -50.0),
    velocity: tuple[float, float, float] = (10.0, 0.0, 0.0),
    *,
    drop_velocity: bool = False,
) -> TelemetrySnapshot:
    return TelemetrySnapshot(
        timestamp=0.0,
        north_m=position[0],
        east_m=position[1],
        down_m=position[2],
        vx_m_s=None if drop_velocity else velocity[0],
        vy_m_s=None if drop_velocity else velocity[1],
        vz_m_s=None if drop_velocity else velocity[2],
    )


# ----------------------------------------------------------------------
# 弹道：解析解对照
# ----------------------------------------------------------------------
def test_zero_drag_matches_analytic_parabola() -> None:
    """无阻力时应当精确回到斜抛解析解：t=√(2h/g)、水平位移=v·t。"""
    model = _model(drag_coefficient=0.0)
    height, speed = 50.0, 15.0
    impact = model.predict_impact((0.0, 0.0, -height), (speed, 0.0, 0.0))

    assert impact.ok and impact.ned is not None
    analytic_t = math.sqrt(2.0 * height / GRAVITY)
    assert impact.flight_time_s == pytest.approx(analytic_t, rel=1e-4)
    assert impact.ned[0] == pytest.approx(speed * analytic_t, rel=1e-4)
    assert impact.ned[1] == pytest.approx(0.0, abs=1e-9)
    assert impact.ned[2] == pytest.approx(0.0, abs=1e-9), "落点必须在 ground_z 上"


def test_zero_drag_with_initial_climb() -> None:
    """带向上的初速时，解析解仍然对得上（先去最高点再落回来）。"""
    model = _model(drag_coefficient=0.0)
    vz_up, height = -8.0, 30.0  # NED 里"向上"是负 z
    impact = model.predict_impact((0.0, 0.0, -height), (0.0, 5.0, vz_up))

    assert impact.ok
    # z(t) = -h + vz·t + ½g t²，落到 0 的正根
    a, b, c = 0.5 * GRAVITY, vz_up, -height
    analytic_t = (-b + math.sqrt(b * b - 4 * a * c)) / (2 * a)
    assert impact.flight_time_s == pytest.approx(analytic_t, rel=1e-4)
    assert impact.ned is not None
    assert impact.ned[1] == pytest.approx(5.0 * analytic_t, rel=1e-4)


def _analytic_vertical_drop(drop_height_m: float, terminal: float) -> tuple[float, float]:
    """竖直下落 + 二次阻力的解析解：由高度反解时间，再算速度。

    ``h(t) = (v_t²/g)·ln(cosh(g t / v_t))``、``v(t) = v_t·tanh(g t / v_t)``。
    """

    def fallen(t: float) -> float:
        return (terminal**2 / GRAVITY) * math.log(math.cosh(GRAVITY * t / terminal))

    low, high = 0.0, 10.0 * terminal / GRAVITY + 10.0
    for _ in range(200):
        mid = 0.5 * (low + high)
        if fallen(mid) < drop_height_m:
            low = mid
        else:
            high = mid
    time = 0.5 * (low + high)
    return time, terminal * math.tanh(GRAVITY * time / terminal)


def test_drag_fall_matches_analytic_vertical_solution() -> None:
    """有阻力时竖直下落也必须与解析解一致（这是对阻力项与 RK4 的强校验）。"""
    model = _model()
    for height in (20.0, 60.0, 150.0):
        # 关闭 ISA 的常密度取"投放海拔"处的 ISA 值 → 解析解要用同一高度处的 v_t
        terminal = model.terminal_velocity(height)
        impact = model.predict_impact((0.0, 0.0, -height), (0.0, 0.0, 0.0))
        assert impact.ok
        expect_t, expect_v = _analytic_vertical_drop(height, terminal)
        assert impact.flight_time_s == pytest.approx(expect_t, rel=5e-3)
        assert impact.speed_m_s == pytest.approx(expect_v, rel=5e-3)


def test_off_mode_density_is_isa_at_release_altitude() -> None:
    """关闭 ISA 也不假设海平面：常密度取"地面海拔 + 投放离地高度"的 ISA 值。

    1500m 高原 + 100m 投放 ⇒ ρ = isa_air_density(1600)；空气稀薄、阻力更小，
    所以比"地面=海平面"落得更快（时间更短、落点速度更大）。
    """
    model = _model()
    height, baseline = 100.0, 1500.0
    impact = model.predict_impact((0.0, 0.0, -height), (0.0, 0.0, 0.0), ground_altitude_m=baseline)
    assert impact.ok
    terminal = model.terminal_velocity(height, ground_altitude_m=baseline)
    expect_t, expect_v = _analytic_vertical_drop(height, terminal)
    assert impact.flight_time_s == pytest.approx(expect_t, rel=5e-3)
    assert impact.speed_m_s == pytest.approx(expect_v, rel=5e-3)

    sea_level = model.predict_impact((0.0, 0.0, -height), (0.0, 0.0, 0.0))
    assert impact.flight_time_s < sea_level.flight_time_s, "高原空气稀薄 → 落得更快"
    assert impact.speed_m_s > sea_level.speed_m_s


def test_isa_mode_uses_altitude_baseline() -> None:
    """开启 ISA：逐级按真实海拔算密度，高海拔站点飞得更远、落得更快。"""
    model = _model(air_density_isa=True)
    position, velocity = (0.0, 0.0, -300.0), (20.0, 0.0, 0.0)
    sea_level = model.predict_impact(position, velocity)
    plateau = model.predict_impact(position, velocity, ground_altitude_m=2500.0)
    assert sea_level.ok and plateau.ok
    assert plateau.ned[0] > sea_level.ned[0] + 1.0, "稀薄空气 → 射程明显更远"
    assert plateau.flight_time_s < sea_level.flight_time_s


def test_drag_factor_and_terminal_velocity_use_ground_altitude() -> None:
    """两个点查询都按"地面海拔 + 离地高度"取 ISA 密度（与开关无关）。"""
    model = _model(drag_coefficient=0.0)  # 无阻力时先只验 terminal 的无穷
    assert model.terminal_velocity(0.0, ground_altitude_m=3000.0) == float("inf")

    model = _model()
    assert model.terminal_velocity(50.0, ground_altitude_m=1500.0) == pytest.approx(
        model.terminal_velocity(1550.0), rel=1e-12
    )
    assert model.drag_factor((10.0, 0.0, 0.0), 50.0, ground_altitude_m=1500.0) == pytest.approx(
        model.drag_factor((10.0, 0.0, 0.0), 1550.0), rel=1e-12
    )


def test_terminal_velocity_asymptote() -> None:
    """终端速度：落下速度永远不超过**它自己那条弹道的** v_t，且高度翻倍后只是逼近它。"""
    model = _model()
    assert model.terminal_velocity(0.0) == pytest.approx(
        math.sqrt(
            2
            * model.config.mass_kg
            * GRAVITY
            / (1.225 * model.config.drag_coefficient * model.config.cross_area_m2)
        ),
        rel=1e-9,
    )

    near = model.predict_impact((0.0, 0.0, -50.0), (0.0, 0.0, 0.0))
    far = model.predict_impact((0.0, 0.0, -500.0), (0.0, 0.0, 0.0))
    # 常密度取"投放海拔"处的 ISA 值 → 参照各自投放高度上的 v_t
    near_terminal = model.terminal_velocity(50.0)
    far_terminal = model.terminal_velocity(500.0)
    assert near.speed_m_s < near_terminal and far.speed_m_s < far_terminal
    assert far.speed_m_s > near.speed_m_s, "落得更高应当更快（但不超过 v_t）"
    gap_near = near_terminal - near.speed_m_s
    gap_far = far_terminal - far.speed_m_s
    assert gap_far < gap_near * 0.5, "从 500m 落下应当明显更接近终端速度"


def test_no_drag_has_infinite_terminal_velocity() -> None:
    assert _model(drag_coefficient=0.0).terminal_velocity() == float("inf")


def test_isa_density_monotone_and_nominal() -> None:
    assert isa_air_density(0.0) == pytest.approx(1.225, rel=1e-9)
    assert isa_air_density(-5.0) == pytest.approx(1.225, rel=1e-9), "负高度按 0 处理"
    heights = [0.0, 100.0, 500.0, 2000.0]
    densities = [isa_air_density(h) for h in heights]
    assert densities == sorted(densities, reverse=True)
    assert isa_air_density(2000.0) == pytest.approx(1.0066, rel=2e-3)


# ----------------------------------------------------------------------
# 弹道：风
# ----------------------------------------------------------------------
def test_wind_shifts_impact_downwind() -> None:
    """顺风落点向下风偏移，且偏移量不可能超过风把它完全吹走的上限 W·t。"""
    model = _model()
    position, velocity = (0.0, 0.0, -60.0), (0.0, 0.0, 0.0)  # 悬停投放
    still = model.predict_impact(position, velocity)
    windy = model.predict_impact(position, velocity, wind=(5.0, 0.0, 0.0))

    assert still.ok and windy.ok and still.ned and windy.ned
    shift = windy.ned[0] - still.ned[0]
    assert shift > 1.0, "5 m/s 的风应当把落点吹出米级偏移"
    assert shift < 5.0 * windy.flight_time_s, "偏移不能超过空气完全带着它走的上限"
    assert windy.ned[1] == pytest.approx(still.ned[1], abs=1e-9), "侧风分量应为 0"


def test_headwind_shortens_range() -> None:
    """逆风让水平射程变短（同一次投放，只改风向）。"""
    model = _model()
    position, velocity = (0.0, 0.0, -60.0), (15.0, 0.0, 0.0)  # 向北平飞投放
    still = model.predict_impact(position, velocity)
    head = model.predict_impact(position, velocity, wind=(-8.0, 0.0, 0.0))
    tail = model.predict_impact(position, velocity, wind=(8.0, 0.0, 0.0))

    assert still.ned and head.ned and tail.ned
    assert head.ned[0] < still.ned[0] < tail.ned[0]


# ----------------------------------------------------------------------
# 弹道：病态输入
# ----------------------------------------------------------------------
def test_below_ground_is_rejected_not_guessed() -> None:
    impact = _model().predict_impact((0.0, 0.0, 0.5), (0.0, 0.0, 0.0))
    assert not impact.ok and impact.ned is None
    assert impact.reason == "below_ground"


def test_ground_z_offset_is_respected() -> None:
    """落点平面由 ground_z 给（不是硬编码 0）。"""
    impact = _model(drag_coefficient=0.0).predict_impact(
        (0.0, 0.0, -20.0), (0.0, 0.0, 0.0), ground_z=-10.0
    )
    assert impact.ok and impact.ned is not None
    assert impact.ned[2] == pytest.approx(-10.0)


def test_wrong_dimension_raises() -> None:
    with pytest.raises(ValueError, match="3 维"):
        _model().predict_impact((0.0, 0.0), (0.0, 0.0, 0.0))


# ----------------------------------------------------------------------
# 投放判据
# ----------------------------------------------------------------------
def _judge(**drop_overrides):
    model = _model()
    config = DropConfig(**drop_overrides)
    events: list[tuple[str, dict]] = []
    judge = ReleaseJudge(
        config=config,
        ballistics=model,
        overfly_heading_deg=0.0,  # 向北飞掠
        on_event=lambda kind, data: events.append((kind, data)),
    )
    return judge, model, events


def test_heading_unit_vector() -> None:
    assert heading_unit_vector(0.0) == pytest.approx((1.0, 0.0))
    north, east = heading_unit_vector(90.0)
    assert north == pytest.approx(0.0, abs=1e-9) and east == pytest.approx(1.0)


def test_release_when_predicted_impact_hits_target() -> None:
    judge, model, events = _judge()
    snapshot = _snapshot(position=(0.0, 0.0, -50.0), velocity=(12.0, 0.0, 0.0))
    predicted = model.predict_impact((0.0, 0.0, -50.0), (12.0, 0.0, 0.0))
    assert predicted.ok and predicted.ned

    decision = judge.update(snapshot, predicted.ned, now=100.0)
    assert decision.should_release and decision.reason == "predict"
    assert decision.horizontal_error_m is not None
    assert decision.horizontal_error_m <= judge.config.radius_m
    assert judge.released


def test_no_release_while_far_and_not_passed() -> None:
    judge, _, _ = _judge()
    snapshot = _snapshot(position=(0.0, 0.0, -50.0), velocity=(12.0, 0.0, 0.0))
    decision = judge.update(snapshot, (500.0, 500.0, 0.0), now=100.0)

    assert not decision.should_release
    assert decision.reason == "waiting"
    assert not decision.passed_target
    assert decision.horizontal_error_m is not None and decision.horizontal_error_m > 2.0


def test_judge_passes_ground_altitude_into_the_prediction() -> None:
    """``ground_altitude_m`` 必须真的进到预测：高原空气稀薄 → 落点更远。"""
    model = _model(air_density_isa=True)
    snapshot = _snapshot(position=(0.0, 0.0, -300.0), velocity=(20.0, 0.0, 0.0))
    target = (9999.0, 9999.0, 0.0)  # 够不着，只取预测落点

    sea_level = ReleaseJudge(
        DropConfig(), model, overfly_heading_deg=0.0, ground_altitude_m=0.0
    ).update(snapshot, target, now=100.0)
    plateau = ReleaseJudge(
        DropConfig(), model, overfly_heading_deg=0.0, ground_altitude_m=2500.0
    ).update(snapshot, target, now=100.0)

    assert sea_level.predicted is not None and sea_level.predicted.ok
    assert plateau.predicted is not None and plateau.predicted.ok
    assert plateau.predicted.ned[0] > sea_level.predicted.ned[0] + 1.0
    # per-call 覆盖也要生效
    override = ReleaseJudge(
        DropConfig(), model, overfly_heading_deg=0.0, ground_altitude_m=0.0
    ).update(snapshot, target, ground_altitude_m=2500.0, now=100.0)
    assert override.predicted is not None
    assert override.predicted.ned[0] == pytest.approx(plateau.predicted.ned[0])


def test_fallback_after_passing_target() -> None:
    """从目标前方飞过来、沿航向越过却还没投 → 强制投放（落点判据够不着时）。"""
    judge, _, events = _judge(force_after_pass=True)
    # 先在目标前方（还没越过）：判据只等待
    before = judge.update(
        _snapshot(position=(-100.0, 0.0, -50.0), velocity=(12.0, 0.0, 0.0)),
        (0.0, 300.0, 0.0),
        now=99.0,
    )
    assert not before.should_release and not before.passed_target

    snapshot = _snapshot(position=(100.0, 0.0, -50.0), velocity=(12.0, 0.0, 0.0))
    target = (0.0, 300.0, 0.0)  # 已经越过（北向 100 > 0），但横向很远

    decision = judge.update(snapshot, target, now=100.0)
    assert decision.should_release and decision.reason == "fallback"
    assert decision.passed_target and decision.approached
    kinds = [kind for kind, _ in events]
    assert "release" in kinds


def test_fallback_waits_until_the_target_is_really_passed() -> None:
    """进入飞掠时已在目标后方（**没从前面接近过**）不算"越过"——不许假触发。

    实测（r2 世界架次 某架次）：OVERFLY 开始时飞机刚结束盘旋、还在飞往
    入场点，投影已在目标后方 ⇒ 旧口径在第一拍就强制投放，误差 113 m。
    """
    judge, _, _ = _judge(force_after_pass=True)
    snapshot = _snapshot(position=(100.0, 0.0, -50.0), velocity=(12.0, 0.0, 0.0))
    decision = judge.update(snapshot, (0.0, 300.0, 0.0), now=100.0)

    assert not decision.should_release, "没接近过就投 = 实测 113 m 级误差"
    assert decision.passed_target and not decision.approached
    assert not judge.released


def test_fallback_can_be_disabled() -> None:
    judge, _, _ = _judge(force_after_pass=False)
    snapshot = _snapshot(position=(100.0, 0.0, -50.0), velocity=(12.0, 0.0, 0.0))
    decision = judge.update(snapshot, (0.0, 300.0, 0.0), now=100.0)

    assert not decision.should_release
    assert decision.passed_target, "越过了但配置禁止强制投放"


def test_release_is_latched_once() -> None:
    """一次投放即锁存：后续评估只回 already_released，不再触发。"""
    judge, model, events = _judge()
    snapshot = _snapshot(position=(0.0, 0.0, -50.0), velocity=(12.0, 0.0, 0.0))
    predicted = model.predict_impact((0.0, 0.0, -50.0), (12.0, 0.0, 0.0))
    assert predicted.ned is not None

    first = judge.update(snapshot, predicted.ned, now=100.0)
    second = judge.update(snapshot, predicted.ned, now=100.1)
    third = judge.update(snapshot, predicted.ned, now=100.2)

    assert first.should_release
    assert not second.should_release and second.reason == "already_released"
    assert not third.should_release
    assert [kind for kind, _ in events].count("release") == 1


def test_reset_clears_latch() -> None:
    judge, model, _ = _judge()
    snapshot = _snapshot()
    predicted = model.predict_impact((0.0, 0.0, -50.0), (10.0, 0.0, 0.0))
    assert predicted.ned is not None
    judge.update(snapshot, predicted.ned, now=100.0)
    assert judge.released

    judge.reset()
    assert not judge.released and judge.evaluations == 0


def test_unknown_position_waits() -> None:
    judge, _, _ = _judge()
    decision = judge.update(_snapshot(drop_velocity=True), (0.0, 0.0, 0.0), now=100.0)
    assert not decision.should_release and decision.reason == "unknown_position"


def test_delay_advances_the_release_state() -> None:
    """delay_s 把状态前推：3 秒延迟、20m/s 平飞 → 投放点前移 60m。"""
    judge, _, _ = _judge(delay_s=3.0)
    snapshot = _snapshot(position=(0.0, 0.0, -50.0), velocity=(20.0, 0.0, 0.0))
    decision = judge.update(snapshot, (9999.0, 9999.0, 0.0), now=100.0)

    assert decision.release_position is not None
    assert decision.release_position[0] == pytest.approx(60.0)
    assert decision.release_position[2] == pytest.approx(-50.0)


def test_ground_z_and_wind_can_be_overridden_per_call() -> None:
    judge, _, _ = _judge()
    snapshot = _snapshot(position=(0.0, 0.0, 0.0))  # 已经在地面以下（z=0）
    decision = judge.update(
        snapshot, (0.0, 0.0, 0.0), ground_z=-10.0, wind_ned=(3.0, 0.0, 0.0), now=100.0
    )
    assert decision.reason.startswith("no_prediction")
    assert decision.predicted is not None and not decision.predicted.ok


def test_summary_events_are_throttled_to_5hz() -> None:
    """判据每拍都算，但摘要按 5Hz 落事件日志（0.2s 一条）。"""
    judge, _, events = _judge()
    snapshot = _snapshot()
    for step in range(10):  # 0.00 ~ 0.45s，每 0.05s 一次
        judge.update(snapshot, (9999.0, 9999.0, 0.0), now=100.0 + step * 0.05)

    summaries = [kind for kind, _ in events if kind == "drop_check"]
    assert judge.evaluations == 10
    assert len(summaries) == 3, f"0/0.2/0.4 三条，实际 {len(summaries)}"


def test_release_event_carries_full_prediction() -> None:
    judge, model, events = _judge()
    # ⚠ 预测与判据必须用同一套状态：这里速度要和 _snapshot 一致，否则落点对不上、
    # 判据不会触发（第一版就是栽在这儿）。
    state = ((0.0, 0.0, -50.0), (12.0, 0.0, 0.0))
    predicted = model.predict_impact(state[0], state[1])
    assert predicted.ned is not None
    judge.update(_snapshot(position=state[0], velocity=state[1]), predicted.ned, now=100.0)

    payload = next(data for kind, data in events if kind == "release")
    for key in (
        "impact_ned",
        "flight_time_s",
        "horizontal_error_m",
        "radius_m",
        "release_position",
        "overfly_heading_deg",
        "reason",
    ):
        assert key in payload, f"触发事件里应当有 {key}"
    assert payload["reason"] == "predict"


def test_event_callback_failure_does_not_break_judging() -> None:
    """事件写盘失败不能让投放判据失效（记日志继续）。"""
    judge = ReleaseJudge(
        config=DropConfig(),
        ballistics=_model(),
        overfly_heading_deg=0.0,
        on_event=lambda kind, data: (_ for _ in ()).throw(RuntimeError("磁盘满")),
    )
    predicted = judge.ballistics.predict_impact((0.0, 0.0, -50.0), (10.0, 0.0, 0.0))
    assert predicted.ned is not None
    decision = judge.update(_snapshot(), predicted.ned, now=100.0)
    assert decision.should_release and judge.released


# ----------------------------------------------------------------------
# 风来源（快照 → 弹道）
# ----------------------------------------------------------------------
def test_wind_from_snapshot_distinguishes_missing_from_zero() -> None:
    """ "没有风估计"必须是 None，不能和"风确实是 0"混为一谈。"""
    from airdrop.ballistics import wind_from_snapshot

    config = BallisticsConfig(wind_source="telemetry")
    assert wind_from_snapshot(_snapshot(), config) is None, "快照里没有风字段"

    snapshot = _snapshot()
    snapshot.wind_north_m_s, snapshot.wind_east_m_s, snapshot.wind_down_m_s = 3.0, -2.0, 0.5
    assert wind_from_snapshot(snapshot, config) == pytest.approx((3.0, -2.0, 0.5))
    assert wind_from_snapshot(snapshot, BallisticsConfig(wind_source="zero")) == (0.0, 0.0, 0.0)


def test_wind_from_snapshot_rejects_unknown_source() -> None:
    from airdrop.ballistics import wind_from_snapshot

    with pytest.raises(ValueError, match="wind_source"):
        wind_from_snapshot(_snapshot(), BallisticsConfig(wind_source="guess"))


def test_judge_uses_snapshot_wind_when_available() -> None:
    """快照里有风就用它：落点会被吹偏，因此触发点与零风时不同。"""
    judge, model, _ = _judge()  # wind_ned 默认 None → 走快照
    state = ((0.0, 0.0, -50.0), (12.0, 0.0, 0.0))
    still = model.predict_impact(state[0], state[1])

    snapshot = _snapshot(position=state[0], velocity=state[1])
    snapshot.wind_north_m_s, snapshot.wind_east_m_s, snapshot.wind_down_m_s = -6.0, 0.0, 0.0
    decision = judge.update(snapshot, still.ned, now=100.0)

    # 目标点按零风落点摆的，实际用了逆风 → 落点变近 → 误差变大、不该触发
    assert decision.predicted is not None and decision.predicted.ok
    assert decision.predicted.north_m < still.north_m
    assert not decision.should_release


def test_judge_degrades_to_zero_wind_and_warns_once(caplog) -> None:
    """没有风估计时按零风算，并且只记一次日志（20Hz 每拍都记会淹没日志）。"""
    import logging

    judge, model, _ = _judge()
    predicted = model.predict_impact((0.0, 0.0, -50.0), (12.0, 0.0, 0.0))
    assert predicted.ned is not None
    caplog.set_level(logging.WARNING, logger="airdrop.ballistics.release")

    for step in range(5):
        judge.update(_snapshot(velocity=(12.0, 0.0, 0.0)), (9999.0, 9999.0, 0.0), now=100.0 + step)
    warnings = [r for r in caplog.records if "没有风估计" in r.getMessage()]
    assert len(warnings) == 1, f"应当只记一次，实际 {len(warnings)} 次"


def test_explicit_wind_overrides_snapshot() -> None:
    """显式入参优先于快照（回放/离线想固定风时用）。"""
    judge, model, _ = _judge()
    state = ((0.0, 0.0, -50.0), (12.0, 0.0, 0.0))
    still = model.predict_impact(state[0], state[1])
    snapshot = _snapshot(position=state[0], velocity=state[1])
    snapshot.wind_north_m_s = -6.0
    snapshot.wind_east_m_s = 0.0
    snapshot.wind_down_m_s = 0.0

    decision = judge.update(snapshot, still.ned, wind_ned=(0.0, 0.0, 0.0), now=100.0)
    assert decision.should_release, "显式零风 ⇒ 落点与目标一致 ⇒ 应当触发"
