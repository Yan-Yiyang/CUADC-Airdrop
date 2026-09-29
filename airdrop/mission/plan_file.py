"""QGC ``.plan`` 航线文件的解析与固定翼降落预检（纯 JSON，离线可测）。

运营方式
--------
航线由操作手在 QGroundControl 里画好、另存为 ``.plan``（放 ``routes/``），项目只负责
读进来，必要时把飞掠段插在它前面（见 :func:`airdrop.mission.planner.build_drop_mission`）。
这样"降落剖面"这类必须让飞控认可的几何，由画航线的人（QGC）负责，而不是由本包拼出来。

QGC 会把航线里的复杂项（如"固定翼降落航线"）存成一个 ``ComplexItem`` 对象，
上传时才展开成 MAVLink 项。本包照 QGC 的展开逻辑（``LandingComplexItem::
appendMissionItems``）在本地展开——所以 ``load_plan`` 是纯文件解析，不需要连飞控。
目前支持 ``fwLandingPattern``（固定翼降落航线）；VTOL 降落、测绘/结构航线显式报错，
不猜着展开。

为什么需要预检
--------------
PX4 固定翼上传任务时会跑可行性检查（``MissionFeasibility/FeasibilityChecker.cpp``），
不合规就整条任务被拒；而被拒之后导航器只会"没有可用任务，盘旋"
（``No valid mission available, loitering``），``start_mission()`` 依然返回成功——
2026-09 的 SITL 演练里，飞机就这么原地盘旋了 15 分钟才被状态机超时抓出来。

:func:`check_fixed_wing_landing` 把同一套判据在本地做一遍，让"必然被拒"的任务
在加载/规划阶段就带着原因失败：

1. ``NAV_LAND`` 的紧前一项必须严格高于落点（PX4 用的是 ``< FLT_EPSILON`` 判据）；
2. 下滑角 ``(前项高 − 落点高) / 水平距离`` 不得超过 ``tan(FW_LND_ANG + 0.1°)``；
3. 进场点类型只能是 ``NAV_WAYPOINT`` 或 ``NAV_LOITER_TO_ALT``（绕圈下降到高度）；
4. ``NAV_LOITER_TO_ALT`` 进场时，落点必须在盘旋圈之外。

⚠ 这是"照抄判据"而不是"调用飞控"，所以它只覆盖上面四条几何规则；飞控那边还有
别的检查（空域、第一条航点距离 ``MIS_DIST_1WP`` 等）。真机首飞仍要有人看着。
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .items import (
    MAV_CMD_DO_CHANGE_SPEED,
    MAV_CMD_DO_LAND_START,
    MAV_CMD_DO_SET_CAM_TRIGG_DIST,
    MAV_CMD_IMAGE_STOP_CAPTURE,
    MAV_CMD_NAV_LAND,
    MAV_CMD_NAV_LOITER_TO_ALT,
    MAV_CMD_NAV_WAYPOINT,
    MAV_CMD_VIDEO_STOP_CAPTURE,
    MAV_FRAME_GLOBAL,
    MAV_FRAME_GLOBAL_RELATIVE_ALT,
    MAV_FRAME_GLOBAL_TERRAIN_ALT,
    MAV_FRAME_MISSION,
    UNSET,
    MissionItem,
    command_name,
)

LOGGER = logging.getLogger(__name__)

__all__ = [
    "FW_DEFAULT_LAND_ANGLE_DEG",
    "PLAN_FIRMWARE_PX4",
    "PLAN_VEHICLE_FIXED_WING",
    "PlanError",
    "QgcPlan",
    "check_fixed_wing_landing",
    "load_plan",
]

#: QGC ``mission.firmwareType``：12 = PX4
PLAN_FIRMWARE_PX4 = 12
#: QGC ``mission.vehicleType``：1 = 固定翼
PLAN_VEHICLE_FIXED_WING = 1
#: PX4 ``FW_LND_ANG`` 的出厂默认值（度）。飞机改过这个参数时，把新值传进来。
FW_DEFAULT_LAND_ANGLE_DEG = 8.0
#: PX4 判据里的 0.1° 浮点余量（``FeasibilityChecker.cpp`` 原文）。
_GLIDE_SLOPE_MARGIN_DEG = 0.1

#: 球面半径。**刻意**与 PX4 的 ``CONSTANTS_RADIUS_OF_EARTH`` 取同一个值
#: （``PX4-Autopilot-1.17.0/src/lib/geo/geo.h``），让本地降落预检与飞控的
#: ``get_distance_to_next_waypoint``（``geo.cpp``）用同一把尺子——
#: **不要**换成 WGS84 椭球/``pyproj.Geod``，那会在门限边界附近与飞控的接受/拒绝不一致。
_EARTH_RADIUS_M = 6371000.0
_RELATIVE_FRAMES = (MAV_FRAME_GLOBAL_RELATIVE_ALT,)
_ABSOLUTE_FRAMES = (MAV_FRAME_GLOBAL, MAV_FRAME_GLOBAL_TERRAIN_ALT)


class PlanError(ValueError):
    """``.plan`` 读不了 / 结构不认识 / 内容不合法（消息里带文件路径与原因）。"""


@dataclass(frozen=True, slots=True)
class QgcPlan:
    """一份解析好的 QGC 航线。字段保留原始元信息，便于日志与复盘。"""

    path: str
    items: tuple[MissionItem, ...]
    vehicle_type: int | None = None
    firmware_type: int | None = None
    cruise_speed_m_s: float | None = None
    home_position: tuple[float, float, float] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "items": len(self.items),
            "vehicle_type": self.vehicle_type,
            "firmware_type": self.firmware_type,
            "cruise_speed_m_s": self.cruise_speed_m_s,
            "home_position": None if self.home_position is None else list(self.home_position),
            "commands": [command_name(item.command) for item in self.items],
        }


def _number(value: Any, default: float) -> float:
    """QGC 的 ``null`` = 不指定 → ``default``（位置字段给 0，参数给 :data:`UNSET`）。"""
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PlanError(f"任务项里出现了非数值：{value!r}")
    return float(value)


def _simple_item_from_json(entry: dict[str, Any], index: int, path: Path) -> MissionItem:
    command = entry.get("command")
    if not isinstance(command, int):
        raise PlanError(f"{path}：第 {index} 项缺少合法的 command：{command!r}")
    frame = entry.get("frame")
    if not isinstance(frame, int):
        mode = entry.get("AltitudeMode")
        frame = {
            0: MAV_FRAME_GLOBAL,
            1: MAV_FRAME_GLOBAL_RELATIVE_ALT,
            2: MAV_FRAME_GLOBAL_TERRAIN_ALT,
        }.get(
            mode if isinstance(mode, int) else 1,
            MAV_FRAME_GLOBAL_RELATIVE_ALT,
        )
    params = entry.get("params")
    if not isinstance(params, list) or len(params) < 7:
        raise PlanError(f"{path}：第 {index} 项的 params 必须是有 7 个元素的数组")

    item = MissionItem(
        command=int(command),
        lat=_number(params[4], 0.0),
        lon=_number(params[5], 0.0),
        alt_m=_number(params[6], 0.0),
        frame=int(frame),
        param1=_number(params[0], UNSET),
        param2=_number(params[1], UNSET),
        param3=_number(params[2], UNSET),
        param4=_number(params[3], UNSET),
        autocontinue=bool(entry.get("autoContinue", True)),
        do_jump_id=int(entry.get("doJumpId") or 0),
    )
    altitude = entry.get("Altitude")
    if (
        isinstance(altitude, (int, float))
        and not isinstance(altitude, bool)
        and abs(float(altitude) - item.alt_m) > 1e-6
    ):
        LOGGER.warning(
            "%s：第 %d 项的 Altitude=%.3f 与 params[6]=%.3f 不一致，按 params[6] 取",
            path,
            index,
            float(altitude),
            item.alt_m,
        )
    return item


#: 本包支持的复杂项：QGC 的"固定翼降落航线"。
_COMPLEX_LANDING_TYPES = ("fwLandingPattern",)


def _coordinate(value: Any, what: str, index: int, path: Path) -> tuple[float, float, float]:
    if not isinstance(value, list) or len(value) < 3:
        raise PlanError(f"{path}：第 {index} 项的 {what} 必须是 [纬度, 经度, 高度]")
    return (
        _number(value[0], 0.0),
        _number(value[1], 0.0),
        _number(value[2], 0.0),
    )


def _expand_landing_pattern(
    entry: dict[str, Any], index: int, path: Path
) -> tuple[MissionItem, ...]:
    """把 QGC 的"固定翼降落航线"复杂项展开成 MAVLink 项。

    顺序与参数照抄 QGC ``LandingComplexItem::appendMissionItems``
    （``src/MissionManager/LandingComplexItem.cc``）：

    1. ``DO_LAND_START``——PX4 的 ``specifiesCoordinate=false``，所以不带坐标、
       ``frame=MAV_FRAME_MISSION``、参数全 0；
    2. 可选 ``DO_CHANGE_SPEED``（``useDoChangeSpeed``；param1=1 空速、param3=-1）；
    3. 可选停止拍照（``DO_SET_CAM_TRIGG_DIST`` + ``IMAGE_STOP_CAPTURE``）与停止录像
       （``VIDEO_STOP_CAPTURE``）；
    4. 进场项——``useLoiterToAlt`` 时 ``NAV_LOITER_TO_ALT``（param1=1 要求航向、
       param2=盘旋半径（顺时针为正）、param4=1 切出），否则 ``NAV_WAYPOINT``；
    5. ``NAV_LAND``——落点坐标与高度，高度基准跟 ``altitudesAreRelative`` 走。
    """
    kind = entry.get("complexItemType")
    if kind not in _COMPLEX_LANDING_TYPES:
        raise PlanError(
            f"{path}：第 {index} 项是复杂项 {kind!r}，本包只支持固定翼降落航线"
            "（fwLandingPattern）——VTOL 降落、测绘/结构航线请在 QGC 里改存成普通航点"
        )

    relative_value = entry.get("altitudesAreRelative")
    if relative_value is None:  # 旧版 plan（version 1）的写法
        loiter_relative = bool(entry.get("loiterAltitudeRelative", True))
        landing_relative = bool(entry.get("landingAltitudeRelative", True))
        if loiter_relative != landing_relative:
            raise PlanError(
                f"{path}：第 {index} 项的进场高度与落点高度基准不一致（旧版 plan），展开会有歧义"
            )
        relative = loiter_relative
    else:
        relative = bool(relative_value)
    frame = MAV_FRAME_GLOBAL_RELATIVE_ALT if relative else MAV_FRAME_GLOBAL

    land = _coordinate(entry.get("landCoordinate"), "landCoordinate", index, path)
    approach_value = entry.get("landingApproachCoordinate")
    if approach_value is None:  # QGC 里 loiterCoordinate 是它的旧名字
        approach_value = entry.get("loiterCoordinate")
    approach = _coordinate(approach_value, "landingApproachCoordinate", index, path)
    radius = _number(entry.get("loiterRadius"), 0.0)
    clockwise = bool(entry.get("loiterClockwise", False))

    items: list[MissionItem] = [
        MissionItem(
            command=MAV_CMD_DO_LAND_START,
            frame=MAV_FRAME_MISSION,
            param1=0.0,
            param2=0.0,
            param3=0.0,
            param4=0.0,
        )
    ]
    if bool(entry.get("useDoChangeSpeed", False)):
        items.append(
            MissionItem(
                command=MAV_CMD_DO_CHANGE_SPEED,
                frame=MAV_FRAME_MISSION,
                param1=1.0,  # 1 = 空速
                param2=_number(entry.get("finalApproachSpeed"), UNSET),
                param3=-1.0,  # 油门不变（QGC 传 -1）
            )
        )
    if bool(entry.get("stopTakingPhotos", False)):
        items.append(
            MissionItem(
                command=MAV_CMD_DO_SET_CAM_TRIGG_DIST,
                frame=MAV_FRAME_MISSION,
                param1=0.0,  # 0 = 停止按距离触发
                param2=0.0,
                param3=0.0,
                param4=0.0,
            )
        )
        items.append(
            MissionItem(
                command=MAV_CMD_IMAGE_STOP_CAPTURE,
                frame=MAV_FRAME_MISSION,
                param1=0.0,
            )
        )
    if bool(entry.get("stopVideoPhotos", False)):
        items.append(
            MissionItem(
                command=MAV_CMD_VIDEO_STOP_CAPTURE,
                frame=MAV_FRAME_MISSION,
                param1=0.0,
            )
        )

    if bool(entry.get("useLoiterToAlt", True)):
        items.append(
            MissionItem(
                command=MAV_CMD_NAV_LOITER_TO_ALT,
                lat=approach[0],
                lon=approach[1],
                alt_m=approach[2],
                frame=frame,
                param1=1.0,  # 要求航向后再切出
                param2=radius * (1.0 if clockwise else -1.0),
                param3=0.0,
                param4=1.0,  # 切出时与落点相切
            )
        )
    else:
        items.append(
            MissionItem(
                command=MAV_CMD_NAV_WAYPOINT,
                lat=approach[0],
                lon=approach[1],
                alt_m=approach[2],
                frame=frame,
                param1=0.0,
                param2=0.0,
                param3=0.0,
                param4=UNSET,
            )
        )
    items.append(
        MissionItem(
            command=MAV_CMD_NAV_LAND,
            lat=land[0],
            lon=land[1],
            alt_m=land[2],
            frame=frame,
        )
    )
    LOGGER.info(
        "展开降落航线复杂项（第 %d 项，%s高度）：%s",
        index,
        "相对" if relative else "绝对",
        "、".join(command_name(item.command) for item in items),
    )
    return tuple(items)


def _items_from_json(entry: Any, index: int, path: Path) -> tuple[MissionItem, ...]:
    """一条 plan 项 → 一个或多个 MAVLink 项（复杂项要展开）。"""
    if not isinstance(entry, dict):
        raise PlanError(f"{path}：第 {index} 项不是对象：{entry!r}")
    kind = entry.get("type")
    if kind == "ComplexItem":
        return _expand_landing_pattern(entry, index, path)
    if kind != "SimpleItem":
        raise PlanError(f"{path}：第 {index} 项的类型是 {kind!r}，本包只支持 SimpleItem")
    return (_simple_item_from_json(entry, index, path),)


def load_plan(path: str | Path) -> QgcPlan:
    """读一份 QGC ``.plan``（纯文件解析，不连飞控、不需要 MAVSDK）。"""
    file_path = Path(path)
    try:
        text = file_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise PlanError(f"航线文件读不了：{file_path}（{exc}）") from exc
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise PlanError(f"航线文件不是合法 JSON：{file_path}（{exc}）") from exc
    if not isinstance(raw, dict) or raw.get("fileType") != "Plan":
        raise PlanError(f'不是 QGC .plan 文件（fileType 应为 "Plan"）：{file_path}')
    mission = raw.get("mission")
    if not isinstance(mission, dict):
        raise PlanError(f"{file_path}：缺少 mission 段")
    raw_items = mission.get("items")
    if not isinstance(raw_items, list) or not raw_items:
        raise PlanError(f"{file_path}：mission.items 为空，没有可飞的航线")

    expanded: list[MissionItem] = []
    for index, entry in enumerate(raw_items):
        expanded.extend(_items_from_json(entry, index, file_path))
    items = tuple(expanded)
    positional = [item for item in items if item.is_positional]
    if not positional:
        raise PlanError(f"{file_path}：没有任何带位置的任务项，飞控无法执行")

    home = mission.get("plannedHomePosition")
    home_position = None
    if isinstance(home, list) and len(home) >= 3:
        home_position = (_number(home[0], 0.0), _number(home[1], 0.0), _number(home[2], 0.0))

    vehicle_type = mission.get("vehicleType")
    firmware_type = mission.get("firmwareType")
    cruise_speed = mission.get("cruiseSpeed")
    plan = QgcPlan(
        path=str(file_path),
        items=items,
        vehicle_type=int(vehicle_type) if isinstance(vehicle_type, int) else None,
        firmware_type=int(firmware_type) if isinstance(firmware_type, int) else None,
        cruise_speed_m_s=float(cruise_speed) if isinstance(cruise_speed, (int, float)) else None,
        home_position=home_position,
    )
    if plan.firmware_type is not None and plan.firmware_type != PLAN_FIRMWARE_PX4:
        LOGGER.warning(
            "%s：firmwareType=%s 不是 PX4（%d），按 PX4 的判据预检可能不准",
            file_path,
            plan.firmware_type,
            PLAN_FIRMWARE_PX4,
        )
    if plan.vehicle_type is not None and plan.vehicle_type != PLAN_VEHICLE_FIXED_WING:
        LOGGER.warning(
            "%s：vehicleType=%s 不是固定翼（%d）——降落判据完全不同，务必核对",
            file_path,
            plan.vehicle_type,
            PLAN_VEHICLE_FIXED_WING,
        )
    LOGGER.info(
        "航线文件已加载：%s（%d 项：%s）",
        file_path,
        len(items),
        "、".join(command_name(item.command) for item in items),
    )
    return plan


def _distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """两点水平距离（米）：球面 haversine，半径 :data:`_EARTH_RADIUS_M`。

    ⚠ **刻意不换 ``pyproj.Geod``**：PX4 的降落可行性判据用的就是同款球面公式
    （``lib/geo/geo.cpp`` 的 ``get_distance_to_next_waypoint``），本地预检必须与
    飞控判断一致；椭球大地线更接近真实地表距离，但会在这个判据上引入与飞控不同的
    结果。两者差 ≤0.3%（400m 量级约 1m、10km 约 2m），远小于预检门限的余量
    （``FW_LND_ANG=8°`` 时约 2.4%）。
    """
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = phi2 - phi1
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    return 2.0 * _EARTH_RADIUS_M * math.asin(min(1.0, math.sqrt(a)))


def check_fixed_wing_landing(
    items: tuple[MissionItem, ...] | list[MissionItem],
    *,
    preceding: MissionItem | None = None,
    land_angle_deg: float = FW_DEFAULT_LAND_ANGLE_DEG,
) -> tuple[str, ...]:
    """按 PX4 固定翼的判据预检降落剖面；返回问题列表（空 = 通过）。

    只检查 ``NAV_LAND`` 及其紧前一项——这正是被拒的那条规则。``preceding`` 用来告诉
    本函数"``items`` 之前还有一项"（比如飞掠段插在降落段前面时，降落段的紧前一项是
    飞掠段的 exit）；不给就按"``items`` 就是整条任务"判。
    """
    problems: list[str] = []
    max_slope = math.tan(math.radians(float(land_angle_deg) + _GLIDE_SLOPE_MARGIN_DEG))
    for index, item in enumerate(items):
        if item.command != MAV_CMD_NAV_LAND:
            continue
        previous = items[index - 1] if index > 0 else preceding
        if previous is None:
            problems.append(
                "降落项前面没有任何任务项（PX4：Mission rejected: starts with land waypoint）"
            )
            continue
        if not previous.is_positional:
            problems.append(
                f"{command_name(item.command)} 的紧前一项是 {command_name(previous.command)}，不带位置"
            )
            continue
        relative_previous = previous.frame in _RELATIVE_FRAMES
        relative_land = item.frame in _RELATIVE_FRAMES
        if relative_previous != relative_land:
            problems.append(
                f"{command_name(item.command)} 与紧前一项的高度基准不同"
                f"（frame {previous.frame} vs {item.frame}）：本地无法比较，请在 QGC 里统一高度模式"
            )
            continue
        altitude_gain = float(previous.alt_m) - float(item.alt_m)
        if altitude_gain < 1e-7:
            problems.append(
                f"{command_name(item.command)}：紧前一项高度 {previous.alt_m:.2f}m 不高于落点 "
                f"{item.alt_m:.2f}m"
                "（PX4：the approach waypoint must be above the landing point）"
            )
            continue
        if previous.command == MAV_CMD_NAV_WAYPOINT:
            distance = _distance_m(previous.lat, previous.lon, item.lat, item.lon)
        elif previous.command == MAV_CMD_NAV_LOITER_TO_ALT:
            radius = abs(float(previous.param2)) if math.isfinite(previous.param2) else 0.0
            distance = _distance_m(previous.lat, previous.lon, item.lat, item.lon)
            if radius > 0 and distance <= radius:
                problems.append(
                    f"第 {index} 项 NAV_LAND：落点在盘旋圈内（距离 {distance:.1f}m ≤ 半径 {radius:.1f}m）"
                    "（PX4：the landing point must be outside the orbit radius）"
                )
                continue
            distance = math.sqrt(max(0.0, distance * distance - radius * radius))
        else:
            problems.append(
                f"{command_name(item.command)} 的进场项是 {command_name(previous.command)}，"
                "只允许 NAV_WAYPOINT 或 NAV_LOITER_TO_ALT"
            )
            continue
        if distance <= 0.0:
            problems.append(f"{command_name(item.command)}：进场水平距离为 0，下滑角无法定义")
            continue
        slope = altitude_gain / distance
        if slope > max_slope:
            problems.append(
                f"{command_name(item.command)}：下滑角 {math.degrees(math.atan(slope)):.1f}° 超过 "
                f"FW_LND_ANG={land_angle_deg:.1f}°（进场 {previous.alt_m:.1f}m / 距离 {distance:.0f}m，"
                f"至少要 {altitude_gain / max_slope:.0f}m）"
            )
    return tuple(problems)
