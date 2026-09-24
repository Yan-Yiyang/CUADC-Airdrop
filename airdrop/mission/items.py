"""任务项的**唯一表示**：MAVLink 级（command / frame / params / 位置）。

为什么不用 MAVSDK 的 ``MissionItem``
-----------------------------------
MAVSDK 上传时有一层**不透明的翻译**：``vehicle_action=LAND`` 的一项会被拆成
"一个同坐标的 ``NAV_WAYPOINT`` + 一个 ``NAV_LAND``"两项。PX4 固定翼的可行性检查
（``MissionFeasibility/FeasibilityChecker.cpp``）要求 ``NAV_LAND`` 的**紧前一项严格
高于落点**且下滑角不超过 ``FW_LND_ANG``——被拆出来的那一项与落点同坐标、同高度，
于是**整条任务被拒**：

    [mission_feasibility_checker] Mission rejected: the approach waypoint must be
    above the landing point.
    [navigator] No valid mission available, loitering

更糟的是 MAVSDK 的 ``start_mission()`` 仍然返回成功（命令确实被 ACK 了），飞机却
在原地盘旋——2026-09 的 SITL 演练里为此白等了 15 分钟。实测（同一飞控、同一会话）：

| 上传内容 | 飞控实际存下 |
| --- | --- |
| ``WP 50m`` + ``LAND alt=0`` | ``WP 50m``、``WP 0m``（复制品）、``NAV_LAND 0m`` |
| ``WP 50m`` + ``LAND alt=15`` | ``WP 50m``、``WP 15m``（复制品跟着抬）、``NAV_LAND 0m`` |

所以本包**不再经过那层翻译**：任务项在规划阶段就是明文 MAVLink 项，上传一律走
``mission_raw``（:meth:`airdrop.telemetry.DroneController.upload_mission`），
QGC ``.plan`` 里的项也能原样混进同一条任务。

坐标口径
--------
``alt_m`` 的含义由 ``frame`` 决定：:data:`MAV_FRAME_GLOBAL_RELATIVE_ALT`（默认）
是**相对起飞点**的高度，:data:`MAV_FRAME_GLOBAL` 是海拔。别把两者混着填。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

__all__ = [
    "DEFAULT_TAKEOFF_PITCH_DEG",
    "DEFAULT_WAYPOINT_ACCEPTANCE_M",
    "MAV_CMD_DO_CHANGE_SPEED",
    "MAV_CMD_DO_LAND_START",
    "MAV_CMD_DO_SET_CAM_TRIGG_DIST",
    "MAV_CMD_IMAGE_STOP_CAPTURE",
    "MAV_CMD_NAV_LAND",
    "MAV_CMD_NAV_LOITER_TIME",
    "MAV_CMD_NAV_LOITER_TO_ALT",
    "MAV_CMD_NAV_LOITER_UNLIM",
    "MAV_CMD_NAV_TAKEOFF",
    "MAV_CMD_NAV_WAYPOINT",
    "MAV_CMD_VIDEO_STOP_CAPTURE",
    "MAV_FRAME_GLOBAL",
    "MAV_FRAME_GLOBAL_RELATIVE_ALT",
    "MAV_FRAME_GLOBAL_TERRAIN_ALT",
    "MAV_FRAME_MISSION",
    "UNSET",
    "MissionItem",
    "command_name",
]

#: "不指定"的占位值。**不要用 0 代替**：0 是合法取值（盘旋 0 秒、接受半径 0 =
#: 用飞控默认 ``NAV_ACC_RAD``），含义完全不同。
UNSET = float("nan")

MAV_CMD_NAV_WAYPOINT = 16
MAV_CMD_NAV_LOITER_UNLIM = 17
MAV_CMD_NAV_LOITER_TIME = 19
MAV_CMD_NAV_LAND = 21
MAV_CMD_NAV_TAKEOFF = 22
MAV_CMD_NAV_LOITER_TO_ALT = 31
MAV_CMD_DO_CHANGE_SPEED = 178
MAV_CMD_DO_LAND_START = 189
MAV_CMD_DO_SET_CAM_TRIGG_DIST = 206
MAV_CMD_IMAGE_STOP_CAPTURE = 2001
MAV_CMD_VIDEO_STOP_CAPTURE = 2501

#: 带位置的命令（``lat``/``lon``/``alt_m`` 有意义）。``DO_LAND_START`` 这类纯指令项
#: 不在内——QGC 会把它们的位置字段写成 0，不能拿"纬度为 0"去判它非法。
_POSITIONAL_COMMANDS = frozenset(
    {
        MAV_CMD_NAV_WAYPOINT,
        MAV_CMD_NAV_LOITER_UNLIM,
        MAV_CMD_NAV_LOITER_TIME,
        MAV_CMD_NAV_LAND,
        MAV_CMD_NAV_TAKEOFF,
        MAV_CMD_NAV_LOITER_TO_ALT,
    }
)

MAV_FRAME_GLOBAL = 0
MAV_FRAME_MISSION = 2
MAV_FRAME_GLOBAL_RELATIVE_ALT = 3
MAV_FRAME_GLOBAL_TERRAIN_ALT = 10

#: 普通航点的接受半径（米）。刻意与旧实现（MAVSDK ``MissionItem`` 里 ``UNSET`` 的
#: 实际落值 3.0）保持一致——换实现不能让"到点判据"悄悄变宽或变窄。填 0 在 PX4 上
#: 表示"用飞控参数 ``NAV_ACC_RAD``"，与"接受半径 3 米"不是一回事。
DEFAULT_WAYPOINT_ACCEPTANCE_M = 3.0

#: 固定翼起飞项的默认抬头角（度），对应 ``NAV_TAKEOFF`` 的 ``param1``。
DEFAULT_TAKEOFF_PITCH_DEG = 15.0

_COMMAND_NAMES = {
    MAV_CMD_NAV_WAYPOINT: "NAV_WAYPOINT",
    MAV_CMD_NAV_LOITER_UNLIM: "NAV_LOITER_UNLIM",
    MAV_CMD_NAV_LOITER_TIME: "NAV_LOITER_TIME",
    MAV_CMD_NAV_LAND: "NAV_LAND",
    MAV_CMD_NAV_TAKEOFF: "NAV_TAKEOFF",
    MAV_CMD_NAV_LOITER_TO_ALT: "NAV_LOITER_TO_ALT",
    MAV_CMD_DO_CHANGE_SPEED: "DO_CHANGE_SPEED",
    MAV_CMD_DO_LAND_START: "DO_LAND_START",
    MAV_CMD_DO_SET_CAM_TRIGG_DIST: "DO_SET_CAM_TRIGG_DIST",
    MAV_CMD_IMAGE_STOP_CAPTURE: "IMAGE_STOP_CAPTURE",
    MAV_CMD_VIDEO_STOP_CAPTURE: "VIDEO_STOP_CAPTURE",
}


def command_name(command: int) -> str:
    """常见 ``MAV_CMD`` 的可读名（未知的命令回十进制字符串，不猜）。"""
    return _COMMAND_NAMES.get(int(command), str(int(command)))


def _finite(value: float, what: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{what}必须是有限值：{value!r}")
    return number


@dataclass(frozen=True, slots=True)
class MissionItem:
    """一个 MAVLink 任务项（不可变）。

    ``param1..param4`` 是 ``MAV_CMD`` 自己的参数（如 ``NAV_WAYPOINT`` 的
    ``param1``=到点停留秒数、``param2``=接受半径、``param4``=偏航角），
    用 :data:`UNSET` 表示"不指定"。
    """

    command: int
    lat: float = 0.0
    lon: float = 0.0
    alt_m: float = 0.0
    frame: int = MAV_FRAME_GLOBAL_RELATIVE_ALT
    param1: float = UNSET
    param2: float = UNSET
    param3: float = UNSET
    param4: float = UNSET
    autocontinue: bool = True
    do_jump_id: int = 0

    def __post_init__(self) -> None:
        if int(self.command) < 0:
            raise ValueError(f"MAV_CMD 不能为负：{self.command}")
        _finite(self.alt_m, "任务项高度")
        if self.is_positional:
            if not -90.0 <= float(self.lat) <= 90.0:
                raise ValueError(f"纬度超出 ±90°: {self.lat}")
            if not -180.0 <= float(self.lon) <= 180.0:
                raise ValueError(f"经度超出 ±180°: {self.lon}")

    # ------------------------------------------------------------------
    # 构造
    # ------------------------------------------------------------------
    @property
    def is_positional(self) -> bool:
        """该项是否带位置（``DO_LAND_START`` 这类纯指令项不带）。"""
        return int(self.command) in _POSITIONAL_COMMANDS

    @classmethod
    def waypoint(
        cls,
        lat: float,
        lon: float,
        alt_m: float,
        *,
        acceptance_radius_m: float | None = None,
        frame: int = MAV_FRAME_GLOBAL_RELATIVE_ALT,
        yaw_deg: float | None = None,
    ) -> "MissionItem":
        """普通航点（``NAV_WAYPOINT``）：到点不停留，接受半径默认 3 米。"""
        return cls(
            command=MAV_CMD_NAV_WAYPOINT,
            lat=float(lat),
            lon=float(lon),
            alt_m=float(alt_m),
            frame=int(frame),
            param1=0.0,
            param2=(
                DEFAULT_WAYPOINT_ACCEPTANCE_M
                if acceptance_radius_m is None
                else _finite(acceptance_radius_m, "接受半径")
            ),
            param3=0.0,
            param4=UNSET if yaw_deg is None else _finite(yaw_deg, "偏航角"),
        )

    @classmethod
    def takeoff(
        cls,
        lat: float,
        lon: float,
        alt_m: float,
        *,
        pitch_deg: float = DEFAULT_TAKEOFF_PITCH_DEG,
        yaw_deg: float | None = None,
        frame: int = MAV_FRAME_GLOBAL_RELATIVE_ALT,
    ) -> "MissionItem":
        """起飞项（``NAV_TAKEOFF``）：固定翼按 ``pitch_deg`` 爬升到 ``alt_m``。"""
        return cls(
            command=MAV_CMD_NAV_TAKEOFF,
            lat=float(lat),
            lon=float(lon),
            alt_m=float(alt_m),
            frame=int(frame),
            param1=_finite(pitch_deg, "起飞俯仰角"),
            param4=UNSET if yaw_deg is None else _finite(yaw_deg, "偏航角"),
        )

    @classmethod
    def land(
        cls,
        lat: float,
        lon: float,
        *,
        alt_m: float = 0.0,
        frame: int = MAV_FRAME_GLOBAL_RELATIVE_ALT,
    ) -> "MissionItem":
        """降落项（``NAV_LAND``）。

        ⚠ 固定翼上它不是"想放就放"：**紧前一项必须严格高于落点**，
        且下滑角 ``(前项高-落点高)/水平距离`` 不超过 ``tan(FW_LND_ANG+0.1°)``。
        自己拼降落项时务必按这条检查（见 :mod:`airdrop.mission.plan_file` 的说明）。
        """
        return cls(
            command=MAV_CMD_NAV_LAND,
            lat=float(lat),
            lon=float(lon),
            alt_m=float(alt_m),
            frame=int(frame),
        )

    @classmethod
    def from_waypoint(cls, waypoint: Any, *, land: bool = False) -> "MissionItem":
        """由配置里的航点构造（``waypoint`` 只要带 ``lat``/``lon``/``alt_m``）。

        ``land=True`` 时做成降落项；接受半径沿用 ``waypoint.acceptance_radius_m``
        （未配置则用默认值）。⚠ 降落项按降落语义构造（相对高度 0、到地面），
        **不沿用**航点的 ``alt_m`` 与接受半径——``NAV_LAND`` 本身就落在地面。
        """
        radius = float(getattr(waypoint, "acceptance_radius_m", 0.0) or 0.0)
        if land:
            return cls.land(float(waypoint.lat), float(waypoint.lon))
        return cls.waypoint(
            float(waypoint.lat),
            float(waypoint.lon),
            float(waypoint.alt_m),
            acceptance_radius_m=radius if radius > 0 else None,
        )

    # ------------------------------------------------------------------
    # 视图
    # ------------------------------------------------------------------
    def as_dict(self) -> dict[str, Any]:
        return {
            "command": int(self.command),
            "command_name": command_name(self.command),
            "lat": round(float(self.lat), 8),
            "lon": round(float(self.lon), 8),
            "alt_m": round(float(self.alt_m), 3),
            "frame": int(self.frame),
            "param1": None if math.isnan(self.param1) else round(float(self.param1), 4),
            "param2": None if math.isnan(self.param2) else round(float(self.param2), 4),
            "param3": None if math.isnan(self.param3) else round(float(self.param3), 4),
            "param4": None if math.isnan(self.param4) else round(float(self.param4), 4),
            "autocontinue": bool(self.autocontinue),
            "do_jump_id": int(self.do_jump_id),
        }
