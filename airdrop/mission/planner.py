"""航线规划：侦查航线、飞掠段（entry/exit）与降落航线的拼接（计划 4.7 / Q13 / Q14）。

三件事，全是**纯函数**（不碰 MAVSDK、不碰网络，离线可测）：

1. :func:`build_recon_mission`：侦查航线 → 任务项（首项带起飞项）；
2. :func:`overfly_positions` / :func:`overfly_waypoints`：以目标为中心、沿配置航向
   前后各半段长生成 `[entry, exit]`——飞机从 entry 进、**飞过目标**、从 exit 出，
   方向与判据里的"越过目标"（``dot(位置−目标, 航向) > 0``）严格一致，符号不能反；
3. :func:`build_drop_mission`：飞掠段 + 降落航线**拼成一条任务上传**（Q13），
   没有目标时用备用点（Q11：备用点走同一套弹道判据）。

坐标口径
--------
- 目标/飞掠段用 **NED（米）**，与 :mod:`airdrop.georef` 的输出、投放判据的输入一致；
- 航点用 **WGS84**，由 :func:`airdrop.georef.ned_to_wgs84` 换算，原点就是
  ``GPS_GLOBAL_ORIGIN``（INIT 阶段取到的那一个）；
- ``Waypoint.alt_m`` 与 ``MissionItemSpec.alt_m`` 都是**相对起飞点的高度**
  （MAVSDK ``relative_altitude_m``），不是海拔——所以飞掠高度直接取
  ``OverflyConfig.altitude_m``，不做任何高程换算。备用点换算成 NED 时按
  "原点海拔 + 相对高度"给海拔（见 :func:`waypoint_to_ned`），但那只是把切平面
  对准，返回的 down 一律按地面处理。
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, replace
from typing import Any, Sequence

from ..ballistics.release import heading_unit_vector
from ..config import Config, Waypoint
from ..georef import LLARef, ned_to_wgs84, wgs84_to_ned
from ..telemetry.controller import (
    ACTION_LAND,
    ACTION_TAKEOFF,
    MissionItemSpec,
    NedOrigin,
)

LOGGER = logging.getLogger(__name__)

__all__ = [
    "DropMissionPlan",
    "PlanningError",
    "build_drop_mission",
    "build_recon_mission",
    "llaref_of",
    "overfly_positions",
    "overfly_waypoints",
    "waypoint_to_ned",
]

#: 目标坐标的 down 分量（判据只用水平距离；目标是地面点）
GROUND_DOWN_M = 0.0


class PlanningError(ValueError):
    """航线无法生成（缺目标、缺备用点、缺原点、几何参数非法……）。

    一律**显式失败**：宁可任务起不来，也不要上传一条"看着能飞"的错航线。
    """


@dataclass(frozen=True, slots=True)
class DropMissionPlan:
    """飞掠 + 降落合并任务（Q13）。

    ``source`` 是 ``"target"``（用了统计结果）或 ``"backup"``（用了备用点）——
    落进事件日志，复盘时一眼能看出这架次投的是哪一个。
    """

    items: tuple[MissionItemSpec, ...]
    target_ned: tuple[float, float, float]
    source: str
    heading_deg: float
    entry: Waypoint
    exit: Waypoint

    @property
    def overfly_count(self) -> int:
        return 2

    @property
    def landing_count(self) -> int:
        return len(self.items) - self.overfly_count

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target_ned": [round(value, 3) for value in self.target_ned],
            "heading_deg": round(self.heading_deg, 3),
            "entry": _waypoint_dict(self.entry),
            "exit": _waypoint_dict(self.exit),
            "items": len(self.items),
            "landing_items": self.landing_count,
        }


def _waypoint_dict(waypoint: Waypoint) -> dict[str, float]:
    return {
        "lat": round(waypoint.lat, 8),
        "lon": round(waypoint.lon, 8),
        "alt_m": round(waypoint.alt_m, 3),
    }


def llaref_of(origin: NedOrigin) -> LLARef:
    """控制器给的 :class:`NedOrigin` → georef 的 :class:`LLARef`。

    ⚠ 两者字段顺序**相反**（``NedOrigin`` 是 lat 在前、``LLARef`` 是 lon 在前），
    这里按字段名逐个搬，绝不按位置传。
    """
    return LLARef(
        lon_deg=float(origin.lon_deg),
        lat_deg=float(origin.lat_deg),
        alt_m=float(origin.alt_m),
    )


def waypoint_to_ned(waypoint: Waypoint, origin: LLARef) -> tuple[float, float, float]:
    """航点 → 地面上一点的 NED（米）。

    ``down`` 恒为 :data:`GROUND_DOWN_M`（目标是地面点，投放判据也只用水平距离）。
    水平位置用 ``原点海拔 + 航点相对高度`` 做换算——**不能拿 0 当海拔**：
    NED 是原点处的切平面，海拔填错会把水平分量也带偏（实测 500m 高度差 ⇒ 2cm）。
    """
    north, east, _down = wgs84_to_ned(
        float(waypoint.lon),
        float(waypoint.lat),
        float(origin.alt_m) + float(waypoint.alt_m),
        origin,
    )
    return (north, east, GROUND_DOWN_M)


def _check_target(target_ned: Sequence[float], *, what: str = "目标坐标") -> tuple[float, float, float]:
    if len(target_ned) != 3:
        raise PlanningError(f"{what}必须是 3 维 NED 矢量，收到 {len(target_ned)} 维")
    north, east, down = (float(target_ned[0]), float(target_ned[1]), float(target_ned[2]))
    if not (math.isfinite(north) and math.isfinite(east)):
        raise PlanningError(f"{what}不是有限值：({north}, {east})")
    return (north, east, down)


def overfly_positions(
    target_ned: Sequence[float],
    *,
    heading_deg: float,
    leg_length_m: float,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """飞掠段的 ``(entry, exit)``（NED）。

    以目标为中心、沿航向前后各 ``leg_length_m / 2``：飞机从 entry 进、飞过目标、
    从 exit 出。目标在段中点，前后各有半段加速/修正的余量（Q14：段长需实验验证）。
    """
    target = _check_target(target_ned)
    if not math.isfinite(float(heading_deg)):
        raise PlanningError(f"飞掠航向不是有限值：{heading_deg}")
    if not math.isfinite(float(leg_length_m)) or float(leg_length_m) <= 0:
        raise PlanningError(f"飞掠段长必须为正：{leg_length_m}")
    north, east = heading_unit_vector(heading_deg)
    half = float(leg_length_m) / 2.0
    entry = (target[0] - north * half, target[1] - east * half, GROUND_DOWN_M)
    exit_ = (target[0] + north * half, target[1] + east * half, GROUND_DOWN_M)
    return entry, exit_


def overfly_waypoints(
    target_ned: Sequence[float],
    *,
    heading_deg: float,
    leg_length_m: float,
    altitude_m: float,
    origin: LLARef,
) -> tuple[Waypoint, Waypoint]:
    """飞掠段两个航点（WGS84，高度 = ``altitude_m`` 相对高度）。"""
    if not math.isfinite(float(altitude_m)):
        raise PlanningError(f"飞掠高度不是有限值：{altitude_m}")
    entry_ned, exit_ned = overfly_positions(
        target_ned, heading_deg=heading_deg, leg_length_m=leg_length_m
    )
    return (
        _ned_to_waypoint(entry_ned, altitude_m, origin),
        _ned_to_waypoint(exit_ned, altitude_m, origin),
    )


def _ned_to_waypoint(
    ned: tuple[float, float, float], altitude_m: float, origin: LLARef
) -> Waypoint:
    # 第三维传 0（原点所处的高程面）：返回的 alt 没用（我们要的是相对高度），
    # 传什么都一样 —— 但别改成一个"看起来更对"的值，那只会让人误以为参与了解算。
    lon, lat, _alt = ned_to_wgs84((ned[0], ned[1], GROUND_DOWN_M), origin)
    return Waypoint(lat=float(lat), lon=float(lon), alt_m=float(altitude_m))


def build_recon_mission(config: Config) -> tuple[MissionItemSpec, ...]:
    """侦查航线 → 任务项；``MissionConfig.takeoff_first`` 时首项带起飞项。

    固定翼的起飞由首航点的起飞项承担（Q/计划 4.1）；没有起飞项的任务在
    PX4 上不会被自动起飞，这一点写在示例与文档里。
    """
    route = tuple(config.routes.recon_route)
    if not route:
        raise PlanningError("侦查航线为空（RoutesConfig.recon_route），无任务可上传")
    items = [MissionItemSpec.from_waypoint(waypoint) for waypoint in route]
    if config.mission.takeoff_first:
        items[0] = replace(items[0], action=ACTION_TAKEOFF)
    LOGGER.info(
        "侦查航线：%d 个航点%s",
        len(items),
        "（首项带起飞）" if config.mission.takeoff_first else "",
    )
    return tuple(items)


def build_drop_mission(
    config: Config,
    *,
    origin: LLARef | None,
    target_ned: Sequence[float] | None = None,
) -> DropMissionPlan:
    """飞掠段 + 降落航线合并成一条任务（Q13）。

    ``target_ned=None`` 时用 ``RoutesConfig.backup_point``（无目标分支，Q11）；
    两者都没有、或没有 NED 原点、或降落航线为空 —— 全部抛 :class:`PlanningError`。
    """
    if origin is None:
        raise PlanningError("还没有 NED 原点，无法把目标坐标换算成航点")
    overfly = config.overfly
    routes = config.routes

    landing = tuple(routes.landing_route)
    if not landing:
        raise PlanningError(
            "降落航线为空（RoutesConfig.landing_route）：拒绝上传只有飞掠段的任务"
        )

    if target_ned is not None:
        target = _check_target(target_ned)
        source = "target"
    elif routes.backup_point is not None:
        target = waypoint_to_ned(routes.backup_point, origin)
        source = "backup"
    else:
        raise PlanningError(
            "既没有目标坐标、也没有备用点（RoutesConfig.backup_point）："
            "无法生成飞掠航线"
        )

    entry, exit_ = overfly_waypoints(
        target,
        heading_deg=overfly.heading_deg,
        leg_length_m=overfly.leg_length_m,
        altitude_m=overfly.altitude_m,
        origin=origin,
    )
    items = [MissionItemSpec.from_waypoint(entry), MissionItemSpec.from_waypoint(exit_)]
    items.extend(MissionItemSpec.from_waypoint(waypoint) for waypoint in landing)
    if config.mission.land_last:
        items[-1] = replace(items[-1], action=ACTION_LAND)

    plan = DropMissionPlan(
        items=tuple(items),
        target_ned=target,
        source=source,
        heading_deg=float(overfly.heading_deg) % 360.0,
        entry=entry,
        exit=exit_,
    )
    LOGGER.info(
        "飞掠+降落任务：来源 %s，目标 NED (%.2f, %.2f)，航向 %.1f°，段长 %.0fm，"
        "共 %d 个任务项（飞掠 2 + 降落 %d）",
        source,
        target[0],
        target[1],
        plan.heading_deg,
        overfly.leg_length_m,
        len(items),
        plan.landing_count,
    )
    return plan
