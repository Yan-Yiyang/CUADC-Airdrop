"""QGC ``.plan`` 解析与固定翼降落预检的离线用例。

判据不是"照文档抄的"，是照 PX4 1.17
``src/modules/navigator/MissionFeasibility/FeasibilityChecker.cpp`` 的实际规则写的：
``NAV_LAND`` 的紧前一项必须严格高于落点、下滑角不超过 ``tan(FW_LND_ANG+0.1°)``、
进场项只能是 ``NAV_WAYPOINT`` / ``NAV_LOITER_TO_ALT``、落点必须在盘旋圈外。

降落剖面不满足这套判据时，飞控会把整条任务判为不可行
（``No valid mission available, loitering``）——所以这套判据值得逐条钉住。
"""

from __future__ import annotations

import json
import logging
import math
import shutil
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from airdrop import (
    MAV_CMD_DO_LAND_START,
    MAV_CMD_DO_SET_CAM_TRIGG_DIST,
    MAV_CMD_IMAGE_STOP_CAPTURE,
    MAV_CMD_NAV_LAND,
    MAV_CMD_NAV_LOITER_TO_ALT,
    MAV_CMD_NAV_TAKEOFF,
    MAV_CMD_NAV_WAYPOINT,
    MAV_CMD_VIDEO_STOP_CAPTURE,
    MAV_FRAME_GLOBAL,
    MAV_FRAME_GLOBAL_RELATIVE_ALT,
    MAV_FRAME_MISSION,
    Config,
    GroundConfig,
    MissionItem,
    NedOrigin,
    OverflyConfig,
    PlanError,
    PlanningError,
    RoutesConfig,
    Waypoint,
    build_drop_mission,
    build_recon_mission,
    check_fixed_wing_landing,
    llaref_of,
    load_plan,
)

WORK_ROOT = Path(__file__).resolve().parents[1] / ".plan-test-tmp"

#: 合成原点（中纬度、海拔 500m）
ORIGIN = NedOrigin(lat_deg=47.0, lon_deg=8.0, alt_m=500.0)
#: 合规的进场/落点：40m 高、约 334m 远 ⇒ 下滑斜率 tan≈0.12（约 6.8°）< tan(8.1°)=0.142
APPROACH = (47.000, 8.000, 40.0)
TOUCHDOWN = (46.997, 8.000, 0.0)
#: 不合规的那条（2026-09 示例里用的就是它）：40m 高、183m 远 ⇒ 斜率 tan≈0.219（约 12.4°）
STEEP_TOUCHDOWN = (46.99835, 8.000, 0.0)


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
# 造 .plan（字段与 QGC 实际写入磁盘一致：params[4..6] = 纬度/经度/高度）
# ----------------------------------------------------------------------
def _item(
    command: int,
    lat: float,
    lon: float,
    alt: float,
    *,
    frame: int = 3,
    param1: float | None = 0.0,
    param2: float | None = 0.0,
    do_jump_id: int = 1,
    autocontinue: bool = True,
) -> dict[str, Any]:
    return {
        "AMSLAltAboveTerrain": None,
        "Altitude": alt,
        "AltitudeMode": 1,
        "autoContinue": autocontinue,
        "command": command,
        "doJumpId": do_jump_id,
        "frame": frame,
        "params": [param1, param2, 0, None, lat, lon, alt],
        "type": "SimpleItem",
    }


def _plan(
    items: list[dict[str, Any]], *, vehicle_type: int = 1, firmware_type: int = 12
) -> dict[str, Any]:
    return {
        "fileType": "Plan",
        "groundStation": "QGroundControl",
        "mission": {
            "cruiseSpeed": 15,
            "firmwareType": firmware_type,
            "globalPlanAltitudeMode": 1,
            "hoverSpeed": 5,
            "items": items,
            "plannedHomePosition": [47.0, 8.0, 500],
            "vehicleType": vehicle_type,
            "version": 2,
        },
        "version": 1,
    }


def _write(path: Path, data: Any) -> Path:
    file_path = path / "route.plan"
    file_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return file_path


def _landing_items(*, touchdown: tuple[float, float, float] = TOUCHDOWN) -> list[dict[str, Any]]:
    return [
        _item(MAV_CMD_NAV_WAYPOINT, *APPROACH, do_jump_id=1),
        _item(MAV_CMD_NAV_LAND, *touchdown, do_jump_id=2),
    ]


def _config(**routes: Any) -> Config:
    base = {
        "recon_route": (
            Waypoint(lat=47.0, lon=8.0, alt_m=60.0),
            Waypoint(lat=47.001, lon=8.0, alt_m=60.0),
        ),
        "backup_point": Waypoint(lat=47.05, lon=8.05, alt_m=0.0),
        "landing_route": (
            Waypoint(lat=APPROACH[0], lon=APPROACH[1], alt_m=APPROACH[2]),
            Waypoint(lat=TOUCHDOWN[0], lon=TOUCHDOWN[1], alt_m=TOUCHDOWN[2]),
        ),
    }
    base.update(routes)
    return Config(
        routes=RoutesConfig(**base),
        overfly=OverflyConfig(heading_deg=0.0, altitude_m=20.0, leg_length_m=200.0),
        ground=GroundConfig(ground_point_alt=500.0),
    ).validated()


# ----------------------------------------------------------------------
# 解析
# ----------------------------------------------------------------------
def test_load_plan_reads_items_and_metadata(workdir: Path) -> None:
    path = _write(workdir, _plan(_landing_items()))

    plan = load_plan(path)

    assert plan.path == str(path)
    assert [item.command for item in plan.items] == [MAV_CMD_NAV_WAYPOINT, MAV_CMD_NAV_LAND]
    assert plan.items[0].alt_m == 40.0
    assert plan.items[1].lat == pytest.approx(TOUCHDOWN[0])
    assert plan.vehicle_type == 1 and plan.firmware_type == 12
    assert plan.cruise_speed_m_s == 15.0
    assert plan.home_position == (47.0, 8.0, 500.0)
    assert plan.as_dict()["items"] == 2
    assert plan.as_dict()["commands"] == ["NAV_WAYPOINT", "NAV_LAND"]


def test_load_plan_keeps_frames_and_land_start_without_position(workdir: Path) -> None:
    """``DO_LAND_START`` 这类指令项不带位置——不能拿"纬度为 0"去判它非法。"""
    items = [
        _item(MAV_CMD_DO_LAND_START, 0.0, 0.0, 0.0, frame=MAV_FRAME_GLOBAL),
        *_landing_items(),
    ]
    plan = load_plan(_write(workdir, _plan(items)))

    assert plan.items[0].command == MAV_CMD_DO_LAND_START
    assert plan.items[0].frame == MAV_FRAME_GLOBAL, "frame 原样保留"
    assert not plan.items[0].is_positional
    assert plan.items[1].alt_m == 40.0


def test_load_plan_maps_null_params_to_unset(workdir: Path) -> None:
    items = [
        _item(MAV_CMD_NAV_WAYPOINT, *APPROACH, param1=None, param2=None),
        _item(MAV_CMD_NAV_LAND, *TOUCHDOWN),
    ]
    plan = load_plan(_write(workdir, _plan(items)))

    assert math.isnan(plan.items[0].param1), "null → NaN（不指定）"
    assert math.isnan(plan.items[0].param2)
    assert plan.items[0].lat == pytest.approx(APPROACH[0]), "位置字段照常读到"


def test_load_plan_reports_missing_file(workdir: Path) -> None:
    with pytest.raises(PlanError, match="读不了"):
        load_plan(workdir / "nope.plan")


def test_load_plan_rejects_broken_json(workdir: Path) -> None:
    broken = workdir / "broken.plan"
    broken.write_text("{not json", encoding="utf-8")
    with pytest.raises(PlanError, match="不是合法 JSON"):
        load_plan(broken)


def test_load_plan_rejects_non_plan_json(workdir: Path) -> None:
    wrong = workdir / "other.json"
    wrong.write_text(json.dumps({"fileType": "Something"}), encoding="utf-8")
    with pytest.raises(PlanError, match="不是 QGC"):
        load_plan(wrong)


def test_load_plan_rejects_empty_mission(workdir: Path) -> None:
    with pytest.raises(PlanError, match="为空"):
        load_plan(_write(workdir, _plan([])))


def _landing_pattern(
    *,
    land: tuple[float, float, float] = TOUCHDOWN,
    approach: tuple[float, float, float] = APPROACH,
    radius: float = 75.0,
    clockwise: bool = True,
    use_loiter_to_alt: bool = True,
    relative: bool = True,
    complex_type: str = "fwLandingPattern",
) -> dict[str, Any]:
    """QGC 的"固定翼降落航线"复杂项（字段名与 QGC 实际写入磁盘一致）。"""
    return {
        "type": "ComplexItem",
        "complexItemType": complex_type,
        "version": 2,
        "valueSetIsDistance": False,
        "altitudesAreRelative": relative,
        "landCoordinate": list(land),
        "landingApproachCoordinate": list(approach),
        "loiterRadius": radius,
        "loiterClockwise": clockwise,
        "useLoiterToAlt": use_loiter_to_alt,
        "useDoChangeSpeed": False,
        "finalApproachSpeed": 9,
        "stopTakingPhotos": True,
        "stopVideoPhotos": True,
    }


def test_load_plan_expands_the_fixed_wing_landing_pattern(workdir: Path) -> None:
    """复杂项按 QGC ``LandingComplexItem::appendMissionItems`` 的顺序与参数展开。"""
    path = _write(workdir, _plan([_landing_pattern()]))

    plan = load_plan(path)

    assert [item.command for item in plan.items] == [
        MAV_CMD_DO_LAND_START,
        MAV_CMD_DO_SET_CAM_TRIGG_DIST,
        MAV_CMD_IMAGE_STOP_CAPTURE,
        MAV_CMD_VIDEO_STOP_CAPTURE,
        MAV_CMD_NAV_LOITER_TO_ALT,
        MAV_CMD_NAV_LAND,
    ]
    land_start = plan.items[0]
    assert land_start.frame == MAV_FRAME_MISSION, "PX4 的 DO_LAND_START 不带坐标"
    assert not land_start.is_positional and land_start.param1 == 0.0
    loiter = plan.items[4]
    assert loiter.lat == pytest.approx(APPROACH[0]) and loiter.alt_m == APPROACH[2]
    assert loiter.param1 == 1.0 and loiter.param2 == 75.0 and loiter.param4 == 1.0, (
        "param2 = 半径（顺时针为正）、param4 = 1 表示切出"
    )
    land = plan.items[5]
    assert land.lat == pytest.approx(TOUCHDOWN[0]) and land.alt_m == 0.0
    assert land.frame == MAV_FRAME_GLOBAL_RELATIVE_ALT
    assert check_fixed_wing_landing(plan.items) == ()


def test_expanded_landing_pattern_uses_a_negative_radius_when_counter_clockwise(
    workdir: Path,
) -> None:
    path = _write(workdir, _plan([_landing_pattern(clockwise=False)]))
    loiter = [i for i in load_plan(path).items if i.command == MAV_CMD_NAV_LOITER_TO_ALT][0]
    assert loiter.param2 == -75.0


def test_expanded_landing_pattern_can_use_a_plain_waypoint_entrance(workdir: Path) -> None:
    """``useLoiterToAlt=false`` 时进场项是普通航点（QGC 的另一条分支）。"""
    path = _write(workdir, _plan([_landing_pattern(use_loiter_to_alt=False)]))

    items = load_plan(path).items

    assert items[-2].command == MAV_CMD_NAV_WAYPOINT
    assert items[-2].param2 == 0.0, "QGC 这里传 0 = 用飞控默认接受半径"
    assert check_fixed_wing_landing(items) == ()


def test_load_plan_rejects_unsupported_complex_item(workdir: Path) -> None:
    """其它复杂项（VTOL 降落、测绘/结构）不猜着展开。"""
    path = _write(workdir, _plan([_landing_pattern(complex_type="vtolLandingPattern")]))
    with pytest.raises(PlanError, match="只支持固定翼降落航线"):
        load_plan(path)


def test_load_plan_rejects_short_params(workdir: Path) -> None:
    items = [{"type": "SimpleItem", "command": 16, "frame": 3, "params": [0, 0, 0]}]
    with pytest.raises(PlanError, match="7 个元素"):
        load_plan(_write(workdir, _plan(items)))


# ----------------------------------------------------------------------
# 固定翼降落预检（照 PX4 判据）
# ----------------------------------------------------------------------
def test_precheck_passes_a_compliant_landing(workdir: Path) -> None:
    plan = load_plan(_write(workdir, _plan(_landing_items())))
    assert check_fixed_wing_landing(plan.items) == ()


def test_precheck_rejects_approach_at_landing_altitude() -> None:
    """SITL 演练里被拒的就是这条：紧前一项与落点同高。"""
    items = [
        MissionItem.waypoint(*APPROACH),
        MissionItem.land(TOUCHDOWN[0], TOUCHDOWN[1], alt_m=APPROACH[2]),
    ]
    problems = check_fixed_wing_landing(items)

    assert problems and "不高于落点" in problems[0]
    assert "above the landing point" in problems[0], "把飞控的原话带上，便于对照日志"


def test_precheck_rejects_too_steep_glide_slope() -> None:
    """40m 高、183m 远 = 0.219 > tan(8.1°)：示例里原来的降落航线就是这条。"""
    items = [MissionItem.waypoint(*APPROACH), MissionItem.land(*STEEP_TOUCHDOWN[:2])]
    problems = check_fixed_wing_landing(items)

    assert problems and "下滑角" in problems[0] and "FW_LND_ANG" in problems[0]
    assert "至少要" in problems[0], "报错要给出可行解（进场距离下限）"


def test_precheck_accepts_loiter_to_alt_entrance_outside_the_orbit() -> None:
    """QGC 固定翼"降落航线"生成的是 LOITER_TO_ALT 进场；落点在圈外就合格。"""
    items = [
        MissionItem(command=MAV_CMD_NAV_LOITER_TO_ALT, lat=47.0, lon=8.0, alt_m=40.0, param2=80.0),
        MissionItem.land(*TOUCHDOWN[:2]),
    ]
    assert check_fixed_wing_landing(items) == ()


def test_precheck_rejects_landing_inside_the_orbit() -> None:
    items = [
        MissionItem(command=MAV_CMD_NAV_LOITER_TO_ALT, lat=47.0, lon=8.0, alt_m=40.0, param2=800.0),
        MissionItem.land(*TOUCHDOWN[:2]),
    ]
    problems = check_fixed_wing_landing(items)

    assert problems and "盘旋圈内" in problems[0]


def test_precheck_rejects_unsupported_entrance_type() -> None:
    items = [
        MissionItem(command=MAV_CMD_NAV_TAKEOFF, lat=47.0, lon=8.0, alt_m=40.0),
        MissionItem.land(*TOUCHDOWN[:2]),
    ]
    problems = check_fixed_wing_landing(items)

    assert problems and "只允许 NAV_WAYPOINT" in problems[0]


def test_precheck_uses_the_preceding_item_for_a_landing_tail() -> None:
    """飞掠段插在降落段前面时，降落段的紧前一项是飞掠段的 exit。"""
    tail = [MissionItem.land(*TOUCHDOWN[:2])]

    assert check_fixed_wing_landing(tail, preceding=MissionItem.waypoint(*APPROACH)) == ()
    problems = check_fixed_wing_landing(tail)
    assert problems and "前面没有任何任务项" in problems[0]


def test_precheck_reports_mixed_altitude_frames() -> None:
    """相对高度 vs 海拔：本地比不了就说比不了，不猜。"""
    items = [
        MissionItem.waypoint(47.0, 8.0, 540.0, frame=MAV_FRAME_GLOBAL),
        MissionItem.land(*TOUCHDOWN[:2]),
    ]
    problems = check_fixed_wing_landing(items)

    assert problems and "高度基准不同" in problems[0]


# ----------------------------------------------------------------------
# 与规划器/配置的装配
# ----------------------------------------------------------------------
def test_build_drop_mission_puts_overfly_before_the_plan(workdir: Path) -> None:
    """运营方式：操作手画好降落段，飞掠段插在它前面，合成一条任务上传。"""
    path = _write(workdir, _plan(_landing_items()))
    config = _config(landing_route=(), land_plan=str(path))

    plan = build_drop_mission(config, origin=llaref_of(ORIGIN), target_ned=(300.0, 0.0, 0.0))

    assert plan.tail_source == f"plan:{path}"
    assert len(plan.items) == 4 and plan.overfly_count == 2 and plan.landing_count == 2
    assert [item.command for item in plan.items] == [
        MAV_CMD_NAV_WAYPOINT,
        MAV_CMD_NAV_WAYPOINT,
        MAV_CMD_NAV_WAYPOINT,
        MAV_CMD_NAV_LAND,
    ]
    assert plan.items[2].alt_m == 40.0, "降落段原样使用（plan 里的项没有被改写）"
    assert plan.as_dict()["tail_source"].startswith("plan:")


def test_build_drop_mission_refuses_a_plan_px4_would_reject(workdir: Path) -> None:
    """不合规的 plan 在规划阶段就带着原因失败，而不是上传后干等。"""
    path = _write(workdir, _plan(_landing_items(touchdown=STEEP_TOUCHDOWN)))
    config = _config(landing_route=(), land_plan=str(path))

    with pytest.raises(PlanningError, match="下滑角"):
        build_drop_mission(config, origin=llaref_of(ORIGIN), target_ned=(300.0, 0.0, 0.0))


def test_build_drop_mission_refuses_a_plan_without_landing_but_keeps_other_routes(
    workdir: Path,
) -> None:
    """plan 里没有降落项时不报错（操作手自己落地），但几何问题仍会被报出来。"""
    path = _write(workdir, _plan([_item(MAV_CMD_NAV_WAYPOINT, *APPROACH)]))
    config = _config(landing_route=(), land_plan=str(path))

    plan = build_drop_mission(config, origin=llaref_of(ORIGIN), target_ned=(300.0, 0.0, 0.0))

    assert len(plan.items) == 3
    assert all(item.command == MAV_CMD_NAV_WAYPOINT for item in plan.items)


def test_build_drop_mission_rejects_unreadable_plan(workdir: Path) -> None:
    config = _config(landing_route=(), land_plan=str(workdir / "missing.plan"))
    with pytest.raises(PlanningError, match="降落段.*读不了"):
        build_drop_mission(config, origin=llaref_of(ORIGIN), target_ned=(300.0, 0.0, 0.0))


def test_build_recon_mission_uses_plan_verbatim(
    workdir: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """侦查段来自 plan 时原样使用；缺起飞项只告警（不悄悄补一项进去）。"""
    items = [
        _item(MAV_CMD_NAV_WAYPOINT, 47.0, 8.0, 50.0, do_jump_id=1),
        _item(MAV_CMD_NAV_WAYPOINT, 47.002, 8.0, 50.0, do_jump_id=2),
    ]
    path = _write(workdir, _plan(items))
    config = _config(recon_route=(), recon_plan=str(path))

    with caplog.at_level(logging.WARNING):
        built = build_recon_mission(config)

    assert [item.command for item in built] == [MAV_CMD_NAV_WAYPOINT, MAV_CMD_NAV_WAYPOINT]
    assert "没有起飞项" in caplog.text
    assert not any(item.command == MAV_CMD_NAV_TAKEOFF for item in built)


def test_config_rejects_two_sources_for_one_leg() -> None:
    """一条腿两个来源 → 报错。静默让其中一个优先，比报错危险得多。"""
    with pytest.raises(ValueError, match="降落段只能二选一"):
        _config(land_plan="routes/land.plan")
    with pytest.raises(ValueError, match="侦查段只能二选一"):
        _config(recon_plan="routes/recon.plan")
    with pytest.raises(ValueError, match="fw_land_angle_deg"):
        Config(routes=RoutesConfig(fw_land_angle_deg=90.0)).validated()


def test_empty_plan_strings_keep_the_waypoint_sources() -> None:
    """默认（空串）走配置航点，行为与引入 plan 之前一致。"""
    config = _config()
    items = build_recon_mission(config)
    plan = build_drop_mission(config, origin=llaref_of(ORIGIN), target_ned=(300.0, 0.0, 0.0))

    assert items[0].command == MAV_CMD_NAV_TAKEOFF
    assert plan.tail_source == "route"
    assert plan.items[-1].command == MAV_CMD_NAV_LAND
