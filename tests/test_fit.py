"""P12：投放记录（位置/速度/姿态）+ 弹道参数反演。

口径
----
* **合成本**：用**已知真值**参数的弹道模型算出一批"实测落点"，再拿基准（占位）参数
  去反演——验的是"能不能收回真值"，而不是"能不能把残差压小"（后者随便一组退化参数
  都能做到，正是本模块要拦住的东西）。
* 另外三条同样重要：**参数分不开时要拒绝**（同速同高投放下的 Cd vs 释放延迟、
  Cd/m/A 的退化）、**测量文件的各种写法**（经纬度 / NED / CSV / 模板）、
  以及 ``tools/fit_ballistics.py`` 这条现场流程真能跑通。

全部离线、不用硬件；临时目录放在工作区内（不用 ``tmp_path``，见 AGENTS.md）。
"""

from __future__ import annotations

import csv
import json
import math
import shutil
import uuid
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import pytest

from airdrop import (
    DROPS_NAME,
    BallisticsConfig,
    BallisticsModel,
    Config,
    DropRecord,
    DropSample,
    DropWriter,
    FitConfig,
    FlightRecorder,
    LLARef,
    append_drop,
    fit_ballistics,
    load_drops,
    load_impacts,
    match_impacts,
    ned_to_wgs84,
    predict_record_impact,
    release_conditions,
    resolve_wind,
    write_impact_template,
)
from airdrop.ballistics.drops import IMPACTS_JSONL_NAME, ImpactMeasurement
from airdrop.ballistics.fit import FIT_PARAMETERS

WORK_ROOT = Path(__file__).resolve().parents[1] / ".fit-test-tmp"

#: NED 原点（合成用）：中纬度、海拔 500m
ORIGIN = LLARef(lon_deg=8.0, lat_deg=47.0, alt_m=500.0)
#: 合成"真值"：阻力系数 0.95（基准配置里是 0.6 的占位值）
TRUE_CD = 0.95
#: 合成真值里的释放延迟（记录里的 delay_s 刻意写成 0，看反演能不能找回来）
TRUE_DELAY = 0.25
#: 合成风（记录里存的就是它）
WIND = (2.0, -1.5, 0.0)


@pytest.fixture
def workdir() -> Iterator[Path]:
    """工作区内的临时目录；用例结束整棵删掉（不用 ``tmp_path``：见 AGENTS）。"""
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    path = WORK_ROOT / uuid.uuid4().hex[:8]
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


# ----------------------------------------------------------------------
# 合成素材
# ----------------------------------------------------------------------
def _record(
    index: int,
    *,
    height_m: float = 40.0,
    speed_m_s: float = 15.0,
    heading_deg: float = 0.0,
    delay_s: float = TRUE_DELAY,
    wind: tuple[float, float, float] | None = WIND,
    ground_z: float = 0.0,
    position_ned: tuple[float, float, float] | None = None,
    quaternion: tuple[float, float, float, float] | None = (1.0, 0.0, 0.0, 0.0),
    roll_deg: float | None = 3.0,
    pitch_deg: float | None = -2.0,
    origin: LLARef | None = ORIGIN,
) -> DropRecord:
    """造一条投放记录（默认水平前飞、机头朝北、离地 40m、15m/s）。"""
    radians = math.radians(heading_deg)
    position = position_ned or (100.0 + index * 5.0, 200.0 - index * 3.0, -height_m)
    return DropRecord(
        index=index,
        timestamp=1000.0 + index,
        position_ned=position,
        velocity_ned=(
            speed_m_s * math.cos(radians),
            speed_m_s * math.sin(radians),
            -1.0,
        ),
        ground_z=ground_z,
        roll_deg=roll_deg,
        pitch_deg=pitch_deg,
        yaw_deg=heading_deg,
        quaternion_wxyz=quaternion,
        wind_ned=wind,
        origin=origin,
        target_ned=(position[0], position[1], ground_z),
        reason="predict",
        delay_s=delay_s,
        ballistics=BallisticsConfig(),
    )


def _measure(
    records: list[DropRecord],
    *,
    ballistics: BallisticsConfig | None = None,
    delay_s: float = TRUE_DELAY,
    wind_scale: float = 1.0,
    offsets: dict[int, tuple[float, float]] | None = None,
) -> tuple[DropSample, ...]:
    """用**真值**参数正演出"实测落点"（合成实验的全部依据）。

    ⚠ 默认让"真值延迟"与记录里的 ``delay_s`` **一致**（= 一次 ``DropConfig.delay_s``
    配对的飞行）。要让反演**去估**延迟，就把记录的 ``delay_s`` 改成 0 而让这里的
    真值保持 0.25——不拟合延迟时正演用记录里的值，两者不一致时 Cd 会替它背锅
    （这正是 :mod:`airdrop.ballistics.fit` docstring 里说的第 2 条陷阱）。
    ``wind_scale`` 用来造"真风是记录值若干倍"的场景。
    """
    model = BallisticsModel(ballistics or BallisticsConfig(drag_coefficient=TRUE_CD))
    samples: list[DropSample] = []
    for record in records:
        impact = predict_record_impact(record, model, delay_s=delay_s, wind_scale=wind_scale)
        assert impact.ok, impact.reason
        north, east, down = impact.ned
        if offsets and record.index in offsets:
            north += offsets[record.index][0]
            east += offsets[record.index][1]
        samples.append(
            DropSample(
                record=record,
                impact_ned=(north, east, down),
                impact_source="synthetic",
            )
        )
    return tuple(samples)


def _spread_campaign() -> list[DropRecord]:
    """高差/速差都拉开的投放（Cd 与释放延迟在这里才分得开）。"""
    return [
        _record(1, height_m=25.0, speed_m_s=12.0, heading_deg=0.0),
        _record(2, height_m=40.0, speed_m_s=16.0, heading_deg=90.0),
        _record(3, height_m=55.0, speed_m_s=20.0, heading_deg=200.0),
        _record(4, height_m=70.0, speed_m_s=24.0, heading_deg=300.0),
    ]


def _flat_campaign() -> list[DropRecord]:
    """同高同速、只换航向的投放（Cd 与释放延迟在这里**分不开**）。

    ⚠ 必须是**零风**：一旦有风，同样的地速在不同航向下对应不同的**空速**，
    Cd 与延迟就又能部分分开了（实战里有风反而是好事）。
    """
    return [
        _record(index, height_m=30.0, speed_m_s=15.0, heading_deg=heading, wind=(0.0, 0.0, 0.0))
        for index, heading in enumerate((0.0, 90.0, 180.0, 270.0), 1)
    ]


# ----------------------------------------------------------------------
# 记录本身的读写与语义
# ----------------------------------------------------------------------
def test_drop_record_round_trips_through_jsonl(workdir: Path) -> None:
    record = _record(1, delay_s=0.05)
    record = replace(
        record,
        predicted_impact_ned=(1.0, 2.0, 0.0),
        predicted_flight_time_s=2.5,
        predicted_error_m=0.4,
    )
    path = workdir / DROPS_NAME
    append_drop(path, record)
    append_drop(path, _record(2, wind=None, quaternion=None))

    loaded = load_drops(workdir)  # 给目录也能找
    assert [item.index for item in loaded] == [1, 2]

    first = loaded[0]
    assert first.position_ned == record.position_ned
    assert first.velocity_ned == record.velocity_ned
    assert first.euler_deg == (3.0, -2.0, 0.0)
    assert first.quaternion_wxyz == (1.0, 0.0, 0.0, 0.0)
    assert first.wind_ned == WIND
    assert first.origin == ORIGIN
    assert first.delay_s == pytest.approx(0.05)
    assert first.predicted_flight_time_s == pytest.approx(2.5)
    assert first.ballistics.drag_coefficient == pytest.approx(0.6)

    second = loaded[1]
    assert second.wind_ned is None and second.quaternion_wxyz is None
    assert second.euler_deg == (3.0, -2.0, 0.0)  # 欧拉角还在
    assert second.height_agl_m == pytest.approx(40.0)


def test_drop_record_derived_fields() -> None:
    record = _record(1, height_m=30.0, speed_m_s=15.0, heading_deg=90.0, ground_z=-5.0)
    assert record.horizontal_speed_m_s == pytest.approx(15.0)
    assert record.heading_deg == pytest.approx(90.0)
    assert record.height_agl_m == pytest.approx(25.0)
    # 悬停（水平速度为零）没有航迹方位，不能编一个出来
    hovering = replace(record, velocity_ned=(0.0, 0.0, 0.0))
    assert hovering.heading_deg is None


def test_release_conditions_apply_delay_and_body_offset() -> None:
    record = _record(1, height_m=40.0, speed_m_s=20.0, heading_deg=0.0, delay_s=0.0)
    # 1) 一阶前推：位置 += 速度 × 延迟（与 ReleaseJudge 同款）
    position, velocity = release_conditions(record, delay_s=0.5)
    assert position == pytest.approx(
        (record.position_ned[0] + 10.0, record.position_ned[1], record.position_ned[2] - 0.5)
    )
    assert velocity == record.velocity_ned

    # 2) 默认沿用记录里的 delay_s
    default_position, _ = release_conditions(record, delay_s=None)
    assert default_position == pytest.approx(record.position_ned)

    # 3) 挂点偏移要经**姿态**转到 NED：右滚 90° 时机体"下方"指向西（east = -1）
    rolled = replace(
        record, quaternion_wxyz=(math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0)
    )
    position, _ = release_conditions(rolled, delay_s=0.0, offset_body_m=(0.0, 0.0, 1.0))
    assert position == pytest.approx(
        (record.position_ned[0], record.position_ned[1] - 1.0, record.position_ned[2])
    )
    # 没有姿态就不许假装水平
    without_attitude = replace(
        rolled, quaternion_wxyz=None, roll_deg=None, pitch_deg=None, yaw_deg=None
    )
    with pytest.raises(ValueError, match="没有姿态"):
        release_conditions(without_attitude, offset_body_m=(0.0, 0.0, 1.0))


def test_wind_resolution_never_invents_a_wind() -> None:
    record = _record(1)
    assert resolve_wind(record) == WIND
    assert resolve_wind(record, scale=2.0) == pytest.approx((4.0, -3.0, 0.0))
    assert resolve_wind(replace(record, wind_ned=None)) == (0.0, 0.0, 0.0)


def test_predict_record_impact_matches_a_direct_model_call() -> None:
    record = _record(1, height_m=40.0, speed_m_s=16.0, heading_deg=45.0)
    model = BallisticsModel(BallisticsConfig(drag_coefficient=TRUE_CD))
    wrapped = predict_record_impact(record, model, delay_s=TRUE_DELAY)
    position, velocity = release_conditions(record, delay_s=TRUE_DELAY)
    baseline = record.ground_altitude_m
    assert baseline is not None
    direct = model.predict_impact(
        position,
        velocity,
        ground_z=record.ground_z,
        ground_altitude_m=baseline,
        wind=WIND,
    )
    assert wrapped.ok and direct.ok
    assert wrapped.ned == pytest.approx(direct.ned)
    assert wrapped.flight_time_s == pytest.approx(direct.flight_time_s)


def test_ground_altitude_baseline_comes_from_the_record_origin() -> None:
    """密度基准来自记录里的 GPS 原点：origin 海拔 − ground_z；缺原点退回 None/0。"""
    record = replace(
        _record(1, ground_z=-20.0), origin=LLARef(lon_deg=8.0, lat_deg=47.0, alt_m=1500.0)
    )
    assert record.ground_altitude_m == pytest.approx(1520.0)
    assert replace(record, origin=None).ground_altitude_m is None

    model = BallisticsModel(BallisticsConfig(drag_coefficient=TRUE_CD, air_density_isa=True))
    with_origin = predict_record_impact(record, model, delay_s=TRUE_DELAY)
    without_origin = predict_record_impact(replace(record, origin=None), model, delay_s=TRUE_DELAY)
    assert with_origin.ok and without_origin.ok
    assert with_origin.flight_time_s < without_origin.flight_time_s, (
        "1500m 高原空气稀薄 → 下落更快（时间更短）；缺原点会退回海平面基准"
    )


def test_fit_uses_the_record_ground_altitude_for_isa_density() -> None:
    """反演的正演也要吃记录里的海拔基准：高原数据 + ISA 开，仍能把真值 Cd 收回。"""
    plateau = LLARef(lon_deg=8.0, lat_deg=47.0, alt_m=1500.0)
    base = BallisticsConfig(drag_coefficient=TRUE_CD, air_density_isa=True)
    records = [replace(record, origin=plateau) for record in _spread_campaign()]
    samples = _measure(records, ballistics=base)
    result = fit_ballistics(samples, FitConfig(base=BallisticsConfig(air_density_isa=True)))

    assert result.ok and result.reliable, result.reason
    assert result.parameters["drag_coefficient"] == pytest.approx(TRUE_CD, rel=1e-6)


def test_fit_warns_when_records_have_no_origin() -> None:
    """缺原点的记录会让密度基准退回海平面：不能静默，报告里要告警。"""
    records = [replace(record, origin=None) for record in _spread_campaign()]
    result = fit_ballistics(_measure(records), FitConfig())
    assert any("没有 NED 原点" in warning for warning in result.warnings)


# ----------------------------------------------------------------------
# 反演：能收回真值
# ----------------------------------------------------------------------
def test_fit_recovers_the_true_drag_coefficient() -> None:
    samples = _measure(_spread_campaign())
    result = fit_ballistics(samples, FitConfig())

    assert result.ok and result.reliable, result.reason
    assert result.fitted == ("drag_coefficient",)
    assert result.parameters["drag_coefficient"] == pytest.approx(TRUE_CD, rel=1e-6)
    assert result.ballistics.drag_coefficient == pytest.approx(TRUE_CD, rel=1e-6)
    # 只动被拟合的那个，其余原样（**质量是实测输入，绝不能被动过**）
    assert result.ballistics.mass_kg == pytest.approx(0.365)
    assert result.ballistics.cross_area_m2 == pytest.approx(0.004)
    # 识别量：κ = Cd·A/m（数据真正识别到的数），以及按实测质量换算出的 Cd·A
    assert result.drag_k_per_m == pytest.approx(TRUE_CD * 0.004 / 0.365, rel=1e-6)
    assert result.drag_area_m2 == pytest.approx(TRUE_CD * 0.004, rel=1e-6)
    assert result.drag_area_sigma_m2 == pytest.approx(
        0.004 * result.sigma["drag_coefficient"], rel=1e-6
    )
    assert result.dof == 2 * len(samples) - 1
    assert result.rms_error_m == pytest.approx(0.0, abs=1e-6)
    assert result.sigma["drag_coefficient"] < 1e-6
    assert result.condition_number == pytest.approx(1.0)
    assert not result.at_bound and not result.warnings
    # 留一验证：无噪声数据上重拟合也应回到真值
    assert result.leave_one_out_rms_m is not None
    assert result.leave_one_out_rms_m < 1e-4
    # 每条残差都要有可用的对照行
    assert len(result.per_drop) == len(samples)
    assert all(item.reason == "" for item in result.per_drop)
    assert {item.label for item in result.per_drop} == {"#1", "#2", "#3", "#4"}


def test_mass_is_a_measured_input_not_a_fitted_parameter() -> None:
    """质量靠称重（用户明确要求），所以它**不在**可拟合集合里——想拟合也没有入口。"""
    assert "mass_kg" not in FIT_PARAMETERS
    assert not hasattr(FitConfig(), "fit_mass_kg")
    assert not hasattr(FitConfig(), "mass_bounds")
    with pytest.raises(TypeError):
        FitConfig(**{"fit_mass_kg": True})  # type: ignore[call-arg]


def test_identified_quantity_is_kappa_given_the_measured_mass() -> None:
    """质量是**输入**：数据只识别 κ = Cd·A/m；Cd 与 Cd·A 都随质量假设同比例变。

    这就是"质量必须实测、不必反演"的数值依据——反过来说，质量称得准，κ 就直接是
    结论；换配重（形状不变 ⇒ Cd·A 不变）时按 ``κ' = Cd·A/m'`` 重算，不必重做试验。
    """
    records = _spread_campaign()
    samples = _measure(records)

    light = fit_ballistics(samples, FitConfig(base=BallisticsConfig(mass_kg=0.365)))
    heavy = fit_ballistics(samples, FitConfig(base=BallisticsConfig(mass_kg=0.500)))
    assert light.ok and heavy.ok
    ratio = 0.500 / 0.365

    # 识别量 κ 与质量假设无关（弹道只认它）
    assert heavy.drag_k_per_m == pytest.approx(light.drag_k_per_m, rel=1e-9)
    # Cd 与 Cd·A 都按同一比例随质量假设走：质量填错多少，它们就错多少
    assert heavy.parameters["drag_coefficient"] == pytest.approx(
        light.parameters["drag_coefficient"] * ratio, rel=1e-6
    )
    assert heavy.drag_area_m2 == pytest.approx(light.drag_area_m2 * ratio, rel=1e-6)
    # 两个配置预测同一个落点（弹道只认 κ）
    first = predict_record_impact(
        samples[0].record, BallisticsModel(light.ballistics), delay_s=TRUE_DELAY
    )
    second = predict_record_impact(
        samples[0].record, BallisticsModel(heavy.ballistics), delay_s=TRUE_DELAY
    )
    assert first.ned == pytest.approx(second.ned, abs=1e-9)


def test_fit_is_reproducible() -> None:
    samples = _measure(_spread_campaign())
    first = fit_ballistics(samples, FitConfig())
    second = fit_ballistics(samples, FitConfig())
    assert first.parameters == second.parameters
    assert first.rms_error_m == second.rms_error_m


def test_fit_recovers_release_delay_when_speeds_spread() -> None:
    """记录里以为延迟是 0，真实延迟 0.25s——速度/高度拉开时这两个量分得开。"""
    records = [replace(record, delay_s=0.0) for record in _spread_campaign()]
    samples = _measure(records, delay_s=TRUE_DELAY)
    result = fit_ballistics(samples, FitConfig(fit_release_delay_s=True))

    assert result.ok and result.reliable, result.reason
    assert result.parameters["drag_coefficient"] == pytest.approx(TRUE_CD, rel=1e-3)
    assert result.parameters["release_delay_s"] == pytest.approx(TRUE_DELAY, rel=1e-2)
    assert result.release_delay_s == pytest.approx(TRUE_DELAY, rel=1e-2)
    correlation = result.correlation["drag_coefficient"]["release_delay_s"]
    assert 0.0 < correlation < 0.99  # 确实相关，但分得开
    assert result.condition_number < 1e3


def test_fit_recovers_a_wind_scale_with_measured_wind() -> None:
    records = [
        _record(
            index,
            height_m=25.0 + 10.0 * index,
            speed_m_s=12.0 + 3.0 * index,
            heading_deg=heading,
            delay_s=0.0,
        )
        for index, heading in enumerate((0.0, 90.0, 200.0), 1)
    ]
    # 真风是记录值的两倍：按 2 倍正演出"实测落点"，反演应当把它找回来
    truth = _measure(records, delay_s=0.0, wind_scale=2.0)

    result = fit_ballistics(truth, FitConfig(fit_wind_scale=True))
    assert result.ok and result.reliable, result.reason
    assert result.parameters["wind_scale"] == pytest.approx(2.0, rel=1e-3)
    assert result.wind_scale == pytest.approx(2.0, rel=1e-3)


def test_fit_keeps_a_constant_offset_in_the_bias_not_in_the_parameters() -> None:
    """整体偏 2m：这不该被"调参调掉"，而要作为常数偏差报出来。"""
    samples = _measure(_spread_campaign(), offsets=dict.fromkeys((1, 2, 3, 4), (2.0, 0.0)))
    result = fit_ballistics(samples, FitConfig())

    assert result.ok
    # 常数偏移只能落在偏差里：北向 -2m，东向应当基本没有
    assert result.bias_north_m == pytest.approx(-2.0, abs=0.1)
    assert result.bias_east_m == pytest.approx(0.0, abs=0.2)
    assert result.rms_error_m == pytest.approx(2.0, abs=0.25)
    # 参数仍然只在真值附近（常数偏移解释不成阻力）
    assert result.parameters["drag_coefficient"] == pytest.approx(TRUE_CD, abs=0.15)


# ----------------------------------------------------------------------
# 反演：该拒绝的必须拒绝
# ----------------------------------------------------------------------
def test_fit_refuses_degenerate_drag_parameters() -> None:
    """Cd 与迎风面积只以乘积出现，同时拟合必然有无穷多组解（质量压根不可拟合）。"""
    samples = _measure(_spread_campaign())
    result = fit_ballistics(samples, FitConfig(fit_cross_area_m2=True))
    assert not result.ok and not result.reliable
    assert "退化" in result.reason
    assert "Cd·A" in result.reason
    assert result.parameters == {}


def test_fit_refuses_parameters_that_cannot_be_separated() -> None:
    """同高同速的投放里，Cd 与释放延迟几乎完全相关——必须拒绝，而不是给一组假数。"""
    samples = _measure(_flat_campaign())
    result = fit_ballistics(samples, FitConfig(fit_release_delay_s=True))
    assert not result.ok
    assert "不可辨识" in result.reason
    assert result.condition_number > 1e6
    # 诊断照给：相关性 ≈ 1，协方差退化
    assert abs(result.correlation["drag_coefficient"]["release_delay_s"]) == pytest.approx(
        1.0, abs=1e-3
    )

    forced = fit_ballistics(
        samples, FitConfig(fit_release_delay_s=True, allow_ill_conditioned=True)
    )
    assert forced.ok and not forced.reliable
    assert any("条件数" in warning for warning in forced.warnings)


def test_fit_refuses_when_the_wind_is_missing() -> None:
    records = [replace(record, wind_ned=None) for record in _spread_campaign()]
    samples = _measure(records)
    result = fit_ballistics(samples, FitConfig(fit_wind_scale=True))
    assert not result.ok
    assert "风估计" in result.reason


def test_fit_refuses_a_wind_scale_with_zero_wind() -> None:
    """全是零风矢量时风比例不可辨识（乘它等于没乘）——靠条件数挡住。"""
    records = [replace(record, wind_ned=(0.0, 0.0, 0.0)) for record in _spread_campaign()]
    samples = _measure(records)
    result = fit_ballistics(samples, FitConfig(fit_wind_scale=True))
    assert not result.ok
    assert "不可辨识" in result.reason


def test_fit_refuses_underdetermined_problems() -> None:
    samples = _measure(_spread_campaign())[:1]  # 2 个残差
    config = FitConfig(fit_release_delay_s=True, fit_wind_scale=True)  # 3 个参数
    result = fit_ballistics(samples, config)
    assert not result.ok and "欠定" in result.reason

    forced = fit_ballistics(
        samples,
        replace(config, allow_underdetermined=True, allow_ill_conditioned=True),
    )
    assert forced.ok and not forced.reliable
    assert forced.dof < 0


def test_fit_reports_zero_degrees_of_freedom_as_unreliable() -> None:
    """样本刚好够"凑"出零残差时，ok 但不 reliable（没有多余信息可校验）。"""
    samples = _measure(_spread_campaign())[:1]
    result = fit_ballistics(samples, FitConfig(fit_release_delay_s=True))
    assert result.ok and not result.reliable
    assert result.dof == 0
    assert any("自由度" in warning for warning in result.warnings)


def test_fit_rejects_empty_inputs() -> None:
    empty = fit_ballistics([], FitConfig())
    assert not empty.ok and "没有样本" in empty.reason
    no_parameters = fit_ballistics(
        _measure(_spread_campaign()),
        FitConfig(fit_drag_coefficient=False),
    )
    assert not no_parameters.ok and "没有选中" in no_parameters.reason


def test_fit_rejects_a_starting_point_outside_the_bounds() -> None:
    samples = _measure(_spread_campaign())
    result = fit_ballistics(samples, FitConfig(base=BallisticsConfig(drag_coefficient=5.0)))
    assert not result.ok and "初值越界" in result.reason


def test_fit_marks_a_parameter_stuck_to_its_bound() -> None:
    """真值在边界之外：拟合会顶到边界，此时必须说"别信"而不是照抄边界值。"""
    samples = _measure(_spread_campaign(), ballistics=BallisticsConfig(drag_coefficient=4.0))
    result = fit_ballistics(samples, FitConfig())
    assert result.ok and not result.reliable
    assert result.at_bound == ("drag_coefficient",)
    assert any("边界" in warning for warning in result.warnings)


# ----------------------------------------------------------------------
# 测量文件：现场会怎么填
# ----------------------------------------------------------------------
def test_match_impacts_accepts_wgs84_and_reports_unknown_indices() -> None:
    records = _spread_campaign()
    samples = _measure(records)
    rows = []
    for sample in samples:
        lon, lat, alt = ned_to_wgs84(sample.impact_ned, ORIGIN)
        rows.append({"index": sample.index, "lat_deg": lat, "lon_deg": lon})
    matched = match_impacts(
        records,
        tuple(
            ImpactMeasurement(index=row["index"], lat_deg=row["lat_deg"], lon_deg=row["lon_deg"])
            for row in rows
        ),
    )
    assert len(matched) == len(records)
    for got, want in zip(matched, samples, strict=True):
        assert got.impact_ned[0] == pytest.approx(want.impact_ned[0], abs=1e-3)
        assert got.impact_ned[1] == pytest.approx(want.impact_ned[1], abs=1e-3)
        assert got.impact_ned[2] == pytest.approx(want.impact_ned[2])
        assert got.impact_source == "gps"

    with pytest.raises(ValueError, match="不存在"):
        match_impacts(records, (ImpactMeasurement(index=99, north_m=0.0, east_m=0.0),))
    with pytest.raises(ValueError, match="不止一次"):
        match_impacts(
            records,
            (
                ImpactMeasurement(index=1, north_m=0.0, east_m=0.0),
                ImpactMeasurement(index=1, north_m=1.0, east_m=1.0),
            ),
        )
    with pytest.raises(ValueError, match="填了一半"):
        match_impacts(records, (ImpactMeasurement(index=1, lat_deg=47.0),))


def test_match_impacts_skips_undecided_rows_and_needs_an_origin() -> None:
    records = _spread_campaign()
    # 全空 = 还没量：跳过而不是报错
    assert match_impacts(records, (ImpactMeasurement(index=1),)) == ()
    # 经纬度但记录里没有 NED 原点 → 显式报错（换算需要原点）
    no_origin = [replace(record, origin=None) for record in records]
    with pytest.raises(ValueError, match="NED 原点"):
        match_impacts(
            no_origin,
            (ImpactMeasurement(index=1, lat_deg=47.0, lon_deg=8.0),),
        )
    # NED 形式不需要原点
    matched = match_impacts(no_origin, (ImpactMeasurement(index=1, north_m=3.0, east_m=4.0),))
    assert matched[0].impact_ned == (3.0, 4.0, 0.0)


def test_impact_files_support_jsonl_and_csv(workdir: Path) -> None:
    records = _spread_campaign()
    samples = _measure(records)

    jsonl = workdir / IMPACTS_JSONL_NAME
    lines = [
        {
            "index": record.index,
            "north_m": sample.impact_ned[0],
            "east_m": sample.impact_ned[1],
            "down_m": sample.impact_ned[2],
        }
        for record, sample in zip(records, samples, strict=True)
    ]
    # 第一条留空（还没量）+ 一条注释用不到的字段
    jsonl.write_text(
        json.dumps({"index": 1, "lat_deg": None, "lon_deg": None, "note": "待测"})
        + "\n"
        + "\n".join(json.dumps(row, ensure_ascii=False) for row in lines[1:])
        + "\n",
        encoding="utf-8",
    )
    matched = match_impacts(records, load_impacts(jsonl))
    assert [sample.index for sample in matched] == [2, 3, 4]
    assert matched[0].impact_ned[0] == pytest.approx(samples[1].impact_ned[0], abs=1e-9)

    csv_path = workdir / "impacts.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["index", "lat_deg", "lon_deg"])
        for sample in samples[:2]:
            lon, lat, _ = ned_to_wgs84(sample.impact_ned, ORIGIN)
            writer.writerow([sample.index, lat, lon])
    matched = match_impacts(records, load_impacts(csv_path))
    assert [sample.index for sample in matched] == [1, 2]
    assert matched[0].impact_ned[0] == pytest.approx(samples[0].impact_ned[0], abs=1e-3)

    bad = workdir / "bad.jsonl"
    bad.write_text("{not json}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="不是合法 JSON"):
        load_impacts(bad)


def test_impact_template_round_trips_into_the_loader(workdir: Path) -> None:
    records = _spread_campaign()
    path = write_impact_template(workdir / IMPACTS_JSONL_NAME, records)
    measurements = load_impacts(path)
    assert [item.index for item in measurements] == [1, 2, 3, 4]
    assert not any(item.measured for item in measurements)
    assert match_impacts(records, measurements) == ()

    payload = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    assert payload["lat_deg"] is None and payload["north_m"] is None
    with pytest.raises(FileNotFoundError):
        load_impacts(workdir / "nope.jsonl")


# ----------------------------------------------------------------------
# 记录落盘：DropWriter 与 FlightRecorder 的接线
# ----------------------------------------------------------------------
def test_drop_writer_appends_records(workdir: Path) -> None:
    writer = DropWriter(workdir / DROPS_NAME)
    try:
        assert writer.append(_record(1))
        assert writer.append(
            {"index": 2, "position_ned": [0.0, 0.0, -10.0], "velocity_ned": [5.0, 0.0, 0.0]}
        )
        assert not writer.append(object())  # 不支持的类型：记 ERROR、返回 False
        assert writer.count == 2
    finally:
        writer.close()
    records = load_drops(workdir / DROPS_NAME)
    assert [record.index for record in records] == [1, 2]

    writer.append(_record(3))  # 关闭后：不抛，直接拒绝
    assert writer.count == 2


def test_recorder_opens_the_drop_log_in_the_flight_dir(workdir: Path) -> None:
    """只验文件接线（不起后台线程）——完整的录制流程见下一个用例。"""
    recorder = FlightRecorder(Config(), base_dir=workdir)
    with pytest.raises(RuntimeError, match="尚未启动"):
        _ = recorder.drops
    recorder._flight_dir = workdir  # 白盒：直接给目录，跳过 start()
    recorder._open_files()
    try:
        assert recorder.drops.append(_record(1))
        assert (workdir / DROPS_NAME).exists()
        assert recorder.stats().drops == 1
    finally:
        recorder._close_files()
    assert recorder.stats().drops == 1
    assert load_drops(workdir / DROPS_NAME)[0].index == 1


def test_flight_recorder_writes_the_drop_log(workdir: Path) -> None:
    """真起一次 ``FlightRecorder``（工作区内，不用 tmp_path）：投放记录与检测/事件并列落盘。

    ⚠ 与 ``tests/test_recorder.py`` 里那条完整飞行目录用例是同一个契约；这里再写一遍
    是因为受限沙箱里 ``tmp_path`` 不可用（那 20 个 error），而**投放记录是反演的唯一
    输入**，这条链路必须在本地就能跑绿。
    """
    from airdrop import AlignmentBuffer, TelemetryBroker

    recorder = FlightRecorder(Config(), base_dir=workdir)
    recorder.start(broker=TelemetryBroker(), buffer=AlignmentBuffer(capacity=4))
    try:
        assert recorder.drops.append(_record(1, delay_s=0.08))
        recorder.detections.append({"frame_index": 1, "code": 56})
        recorder.events.emit("drop", index=1)
        stats = recorder.stats()
        assert stats is not None and stats.drops == 1
    finally:
        recorder.stop()

    flight_dir = recorder.flight_dir
    assert flight_dir is not None
    assert (flight_dir / DROPS_NAME).exists()
    assert recorder.stats().drops == 1
    record = load_drops(flight_dir)[0]
    assert record.position_ned == pytest.approx(_record(1).position_ned)
    assert record.euler_deg == (3.0, -2.0, 0.0)
    assert record.delay_s == pytest.approx(0.08)


# ----------------------------------------------------------------------
# 现场流程：tools/fit_ballistics.py
# ----------------------------------------------------------------------
def _flight_dir(workdir: Path, *, with_impacts: bool, rows: int = 4) -> Path:
    """造一个（够真的）飞行目录：只有 drops.jsonl 与 impacts.jsonl。"""
    records = _spread_campaign()[:rows]
    samples = _measure(records)
    flight = workdir / "某架次"
    flight.mkdir(parents=True, exist_ok=True)
    for record in records:
        append_drop(flight / DROPS_NAME, record)
    if with_impacts:
        lines = []
        for sample in samples:
            lon, lat, _ = ned_to_wgs84(sample.impact_ned, ORIGIN)
            lines.append(
                json.dumps(
                    {"index": sample.index, "lat_deg": lat, "lon_deg": lon}, ensure_ascii=False
                )
            )
        (flight / IMPACTS_JSONL_NAME).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return flight


def test_fit_tool_runs_the_whole_field_workflow(workdir: Path, capsys) -> None:
    from tools import fit_ballistics as tool

    flight = _flight_dir(workdir, with_impacts=True)
    output = workdir / "report.json"
    code = tool.main(
        (tool.FlightInput(path=flight),),
        output_path=output,
        fit_config=FitConfig(),
    )
    printed = capsys.readouterr().out
    assert code == 0
    assert "某架次#1" in printed
    assert "结论：ok=True reliable=True" in printed
    assert "BallisticsConfig(" in printed

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["fit"]["reliable"] is True
    assert report["fit"]["parameters"]["drag_coefficient"] == pytest.approx(TRUE_CD, rel=1e-3)
    assert report["flights"][0]["measured"] == 4
    assert len(report["samples"]) == 4
    assert report["samples"][0]["impact_source"] == "gps"
    assert "position_ned" in report["samples"][0]["record"]


def test_fit_tool_writes_a_template_when_nothing_is_measured(workdir: Path, capsys) -> None:
    from tools import fit_ballistics as tool

    flight = _flight_dir(workdir, with_impacts=False)
    code = tool.main((tool.FlightInput(path=flight),), output_path=workdir / "report.json")
    printed = capsys.readouterr().out
    assert code == 3
    template = flight / IMPACTS_JSONL_NAME
    assert template.exists()
    assert "已生成待填落点" in printed
    assert not (workdir / "report.json").exists()


def test_fit_tool_can_point_at_a_drops_file_and_an_impacts_file(workdir: Path, capsys) -> None:
    from tools import fit_ballistics as tool

    flight = _flight_dir(workdir, with_impacts=True)
    elsewhere = workdir / "measured.jsonl"
    shutil.copy(flight / IMPACTS_JSONL_NAME, elsewhere)
    (flight / IMPACTS_JSONL_NAME).unlink()
    code = tool.main(
        (tool.FlightInput(path=flight / DROPS_NAME, impacts=elsewhere),),
        output_path=workdir / "report.json",
        fit_config=FitConfig(),
    )
    assert code == 0
    assert "measured.jsonl" in capsys.readouterr().out


def test_fit_tool_reports_an_unreliable_fit(workdir: Path, capsys) -> None:
    """同一个架次里 Cd 与延迟分不开时，工具要返回非零并说明原因。"""
    from tools import fit_ballistics as tool

    records = _flat_campaign()
    samples = _measure(records)
    flight = workdir / "某架次"
    flight.mkdir()
    for record in records:
        append_drop(flight / DROPS_NAME, record)
    (flight / IMPACTS_JSONL_NAME).write_text(
        "\n".join(
            json.dumps(
                {"index": s.index, "north_m": s.impact_ned[0], "east_m": s.impact_ned[1]},
                ensure_ascii=False,
            )
            for s in samples
        )
        + "\n",
        encoding="utf-8",
    )
    code = tool.main(
        (tool.FlightInput(path=flight),),
        output_path=workdir / "report.json",
        fit_config=FitConfig(fit_release_delay_s=True),
    )
    printed = capsys.readouterr().out
    assert code == 2
    assert "不可辨识" in printed
    assert "reliable=False" in printed
    assert json.loads((workdir / "report.json").read_text(encoding="utf-8"))["fit"]["ok"] is False


def test_describe_shows_the_judged_and_the_fitted_error_side_by_side() -> None:
    from tools.fit_ballistics import describe

    records = _spread_campaign()
    judged = _measure(records, delay_s=TRUE_DELAY)
    # 判据当时用的是占位参数：记录里带上它的预测落点（含误差）
    records = [
        replace(
            record,
            delay_s=TRUE_DELAY,
            predicted_error_m=1.5,
            predicted_impact_ned=sample.impact_ned,
        )
        for record, sample in zip(records, judged, strict=True)
    ]
    samples = _measure(records, delay_s=TRUE_DELAY)
    result = fit_ballistics(samples, FitConfig())
    text = describe(result, samples)
    assert "判据误差m" in text and "反演后误差m" in text
    assert "1.500" in text  # 判据当时那一列
    assert "常数偏差" in text
