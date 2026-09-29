"""航线规划：侦察航线、飞掠段（entry/exit）与降落段的拼接（计划 4.7 / Q13 / Q14）。

四件事，全是纯函数（不碰 MAVSDK、不碰网络，离线可测）：

1. :func:`build_recon_mission`：侦察航线 → 任务项（首项带起飞项）；
2. :func:`overfly_positions` / :func:`overfly_waypoints`：以目标为中心、沿配置航向
   前后各半段长生成 [entry, exit]——飞机从 entry 进、飞过目标、从 exit 出，
   方向与判据里的"越过目标"（``dot(位置−目标, 航向) > 0``）严格一致，符号不能反；
3. :func:`build_drop_mission`：飞掠段 + 降落段拼成一条任务上传（Q13），
   没有目标时用备用点（Q11：备用点走同一套弹道判据）；
4. :func:`airdrop.mission.plan_file.load_plan`：降落段可以直接用操作手在 QGC 里
   画好、另存为 ``.plan`` 的航线（``RoutesConfig.land_plan``）——飞掠段插在它前面。
   这样"必须让飞控认可的降落剖面"由画航线的人负责；本包只按 PX4 的判据预检
   （:func:`airdrop.mission.plan_file.check_fixed_wing_landing`），不合格就带着原因拒绝。

坐标口径
--------
- 目标/飞掠段用 NED（米），与 :mod:`airdrop.georef` 的输出、投放判据的输入一致；
- 航点用 WGS84，由 :func:`airdrop.georef.ned_to_wgs84` 换算，原点就是
  ``GPS_GLOBAL_ORIGIN``（INIT 阶段取到的那一个）；
- ``Waypoint.alt_m`` 与 :attr:`airdrop.mission.items.MissionItem.alt_m` 都是相对
  起飞点的高度（``MAV_FRAME_GLOBAL_RELATIVE_ALT``），不是海拔——所以飞掠高度直接取
  ``OverflyConfig.altitude_m``，不做任何高程换算。备用点换算成 NED 时按
  "原点海拔 + 相对高度"给海拔（见 :func:`waypoint_to_ned`），但那只是把切平面
  对准，返回的 down 一律按地面处理。
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..ballistics.release import heading_unit_vector
from ..config import Config, Waypoint
from ..georef import LLARef, ned_to_wgs84, wgs84_to_ned
from .items import MAV_CMD_NAV_TAKEOFF, MissionItem
from .plan_file import PlanError, check_fixed_wing_landing, load_plan

if TYPE_CHECKING:  # 仅类型标注：运行时导入 telemetry 会让装配顺序变脆
    from ..telemetry.controller import NedOrigin

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

    一律显式失败：宁可任务起不来，也不要上传一条"看着能飞"的错航线。
    """


@dataclass(frozen=True, slots=True)
class DropMissionPlan:
    """飞掠 + 降落段合并任务（Q13）。

    ``source`` 是 ``"target"``（用了统计结果）或 ``"backup"``（用了备用点）——
    落进事件日志，复盘时一眼能看出这架次投的是哪一个。

    ``tail_source`` 说明飞掠段后面那段是哪来的：``"plan:<文件>"``（操作手的
    QGC 航线）或 ``"route"``（配置里的 ``RoutesConfig.landing_route`` 生成）。
    """

    items: tuple[MissionItem, ...]
    target_ned: tuple[float, float, float]
    source: str
    heading_deg: float
    entry: Waypoint
    exit: Waypoint
    tail_source: str = "route"

    @property
    def overfly_count(self) -> int:
        return 2

    @property
    def landing_count(self) -> int:
        return len(self.items) - self.overfly_count

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "tail_source": self.tail_source,
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

    ⚠ 两者字段顺序相反（``NedOrigin`` 是 lat 在前、``LLARef`` 是 lon 在前），
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
    水平位置用 ``原点海拔 + 航点相对高度`` 做换算——不能拿 0 当海拔：
    NED 是原点处的切平面，海拔填错会把水平分量也带偏（实测 500m 高度差 ⇒ 2cm）。
    """
    north, east, _down = wgs84_to_ned(
        float(waypoint.lon),
        float(waypoint.lat),
        float(origin.alt_m) + float(waypoint.alt_m),
        origin,
    )
    return (north, east, GROUND_DOWN_M)


def _check_target(
    target_ned: Sequence[float], *, what: str = "目标坐标"
) -> tuple[float, float, float]:
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


def _load_plan_items(path: str, *, what: str) -> tuple[MissionItem, ...]:
    """读 QGC ``.plan`` 并把 :class:`~airdrop.mission.plan_file.PlanError`
    统一翻成 :class:`PlanningError`（状态机只需要处理一种异常）。"""
    try:
        return load_plan(path).items
    except PlanError as exc:
        raise PlanningError(f"{what}读不了：{exc}") from exc


def build_recon_mission(config: Config) -> tuple[MissionItem, ...]:
    """侦察航线 → 任务项。

    两个来源二选一（``Config.validated()`` 保证不会同时配置）：

    * ``RoutesConfig.recon_plan``：操作手在 QGC 里画的 ``.plan``，原样使用；
    * ``RoutesConfig.recon_route``：配置航点，``MissionConfig.takeoff_first``
      时首项做成起飞项（固定翼的起飞由首航点的起飞项承担）。

    刻意不自动补 plan 里缺的起飞项：补出来的项是"看着能飞"的隐患，宁可告警让
    操作手自己决定（先手飞起飞再启动任务）。
    """
    plan_path = str(config.routes.recon_plan or "")
    if plan_path:
        items = _load_plan_items(plan_path, what="侦察航线（RoutesConfig.recon_plan）")
        if config.mission.takeoff_first and not any(
            item.command == MAV_CMD_NAV_TAKEOFF for item in items
        ):
            LOGGER.warning(
                "侦察航线来自 %s，但里面没有起飞项：飞控不会自动起飞——"
                "请操作手先起飞（手飞/RC）再启动任务，或改配置 mission.takeoff_first",
                plan_path,
            )
        LOGGER.info("侦察航线：来自 %s，共 %d 项", plan_path, len(items))
        return items

    route = tuple(config.routes.recon_route)
    if not route:
        raise PlanningError(
            "侦察航线为空（RoutesConfig.recon_route 与 recon_plan 都没配），无任务可上传"
        )
    if config.mission.takeoff_first:
        first = route[0]
        items = [
            MissionItem.takeoff(float(first.lat), float(first.lon), float(first.alt_m)),
            *(MissionItem.from_waypoint(waypoint) for waypoint in route[1:]),
        ]
    else:
        items = [MissionItem.from_waypoint(waypoint) for waypoint in route]
    LOGGER.info(
        "侦察航线：%d 个航点%s",
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
    """飞掠段 + 降落段合并成一条任务（Q13）：``[entry, exit] + 降落段``。

    降落段两个来源二选一（``Config.validated()`` 保证）：

    * ``RoutesConfig.land_plan``：操作手的 QGC ``.plan``（飞掠段插在它前面）；
    * ``RoutesConfig.landing_route``：配置航点生成，``MissionConfig.land_last``
      时末项做成降落项。

    ``target_ned=None`` 时用 ``RoutesConfig.backup_point``（无目标分支，Q11）；
    两者都没有、或没有 NED 原点、或降落段为空 —— 全部抛 :class:`PlanningError`。

    ⚠ 降落段会先过一遍 :func:`~airdrop.mission.plan_file.check_fixed_wing_landing`
    的本地预检：PX4 固定翼对降落剖面有硬性几何要求，不满足就整条任务被拒，
    而飞控只会"没有可用任务，盘旋"（2026-09 演练里因此空等 15 分钟）。宁可在规划
    阶段带着原因失败。
    """
    if origin is None:
        raise PlanningError("还没有 NED 原点，无法把目标坐标换算成航点")
    overfly = config.overfly
    routes = config.routes

    plan_path = str(routes.land_plan or "")
    if plan_path:
        tail = _load_plan_items(plan_path, what="降落段（RoutesConfig.land_plan）")
        tail_source = f"plan:{plan_path}"
    else:
        landing = tuple(routes.landing_route)
        if not landing:
            raise PlanningError(
                "降落段为空（RoutesConfig.landing_route 与 land_plan 都没配）："
                "拒绝上传只有飞掠段的任务"
            )
        tail = tuple(MissionItem.from_waypoint(waypoint) for waypoint in landing)
        if config.mission.land_last:
            last = landing[-1]
            tail = (*tail[:-1], MissionItem.land(float(last.lat), float(last.lon)))
        tail_source = "route"

    if target_ned is not None:
        target = _check_target(target_ned)
        source = "target"
    elif routes.backup_point is not None:
        target = waypoint_to_ned(routes.backup_point, origin)
        source = "backup"
    else:
        raise PlanningError(
            "既没有目标坐标、也没有备用点（RoutesConfig.backup_point）：无法生成飞掠航线"
        )

    entry, exit_ = overfly_waypoints(
        target,
        heading_deg=overfly.heading_deg,
        leg_length_m=overfly.leg_length_m,
        altitude_m=overfly.altitude_m,
        origin=origin,
    )
    head = (
        MissionItem.from_waypoint(entry),
        MissionItem.from_waypoint(exit_),
    )
    # 降落段的紧前一项是飞掠段的 exit —— 判据要连这一项一起看（只检查降落段会
    # 把"降落项是段内第一项"误判成"任务以降落开头"）。
    problems = check_fixed_wing_landing(
        tail, preceding=head[-1], land_angle_deg=float(routes.fw_land_angle_deg)
    )
    if problems:
        raise PlanningError(
            f"降落段不合格（{tail_source}）——PX4 固定翼会拒整条任务：" + "；".join(problems)
        )
    items = (*head, *tail)

    plan = DropMissionPlan(
        items=items,
        target_ned=target,
        source=source,
        heading_deg=float(overfly.heading_deg) % 360.0,
        entry=entry,
        exit=exit_,
        tail_source=tail_source,
    )
    LOGGER.info(
        "飞掠+降落任务：来源 %s，目标 NED (%.2f, %.2f)，航向 %.1f°，段长 %.0fm，"
        "共 %d 个任务项（飞掠 2 + 降落 %d，降落段来自 %s）",
        source,
        target[0],
        target[1],
        plan.heading_deg,
        overfly.leg_length_m,
        len(items),
        plan.landing_count,
        tail_source,
    )
    return plan
