"""MAVSDK 工作线程、遥测代理和外部桥接共享的数据模型。"""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class TelemetrySnapshot:
    """最新的同步飞机状态快照。

    ``timestamp`` 是快照组装完成时的本地墙钟时间。
    MAVSDK 提供的时间戳（如有）会单独保留。
    所有字段都是可选的；``is_valid()`` 用于判断是否已经收到任何遥测数据。
    """

    timestamp: float = field(default_factory=time.time)

    # 世界/全局坐标
    latitude_deg: float | None = None
    longitude_deg: float | None = None
    absolute_altitude_m: float | None = None
    relative_altitude_m: float | None = None

    # 本地坐标（NED，单位：米）
    north_m: float | None = None
    east_m: float | None = None
    down_m: float | None = None

    # 速度（NED，单位：米/秒）
    vx_m_s: float | None = None
    vy_m_s: float | None = None
    vz_m_s: float | None = None

    # 姿态（欧拉角，单位：度）
    roll_deg: float | None = None
    pitch_deg: float | None = None
    yaw_deg: float | None = None

    # 姿态（四元数）
    quaternion_w: float | None = None
    quaternion_x: float | None = None
    quaternion_y: float | None = None
    quaternion_z: float | None = None

    # HOME 点 GPS 坐标（不一定等于 NED 原点）
    home_latitude_deg: float | None = None
    home_longitude_deg: float | None = None
    home_absolute_altitude_m: float | None = None
    home_relative_altitude_m: float | None = None

    # 本地 NED 坐标系原点（GPS_GLOBAL_ORIGIN / ref_lat, ref_lon, ref_alt）
    origin_latitude_deg: float | None = None
    origin_longitude_deg: float | None = None
    origin_altitude_m: float | None = None

    # 风估计（NED，单位：米/秒）——来自 MAVSDK ``telemetry.wind()`` 的可选流，
    # 飞控没有风估计时一直是 None（投放判据会降级成零风并记日志，见 ballistics）。
    wind_north_m_s: float | None = None
    wind_east_m_s: float | None = None
    wind_down_m_s: float | None = None

    # MAVSDK 时间戳（系统启动后的微秒数），仅在可用时填充
    attitude_timestamp_us: int | None = None
    #: 任务进度（PX4 ``MISSION_CURRENT``）：``mission_current == mission_total`` 即飞完
    mission_current: int | None = None
    mission_total: int | None = None
    #: 飞行模式（MAVSDK ``FlightMode`` 的名字，如 ``"MISSION"`` / ``"HOLD"``）
    flight_mode: str | None = None
    #: 是否在空中（MAVSDK ``in_air``）：状态机的"等起飞"门用它
    in_air: bool | None = None

    def update_global_position(self, position: Any) -> None:
        """从 MAVSDK ``Telemetry.Position`` 对象复制字段。"""
        self.latitude_deg = float(position.latitude_deg)
        self.longitude_deg = float(position.longitude_deg)
        self.absolute_altitude_m = float(position.absolute_altitude_m)
        self.relative_altitude_m = float(position.relative_altitude_m)

    def update_home_position(self, position: Any) -> None:
        """从 MAVSDK ``Telemetry.Position`` 对象复制 HOME 点字段。"""
        self.home_latitude_deg = float(position.latitude_deg)
        self.home_longitude_deg = float(position.longitude_deg)
        self.home_absolute_altitude_m = float(position.absolute_altitude_m)
        self.home_relative_altitude_m = float(position.relative_altitude_m)

    def update_gps_global_origin(self, origin: Any) -> None:
        """从 MAVSDK ``GpsGlobalOrigin`` 对象复制 NED 原点字段。"""
        self.origin_latitude_deg = float(origin.latitude_deg)
        self.origin_longitude_deg = float(origin.longitude_deg)
        self.origin_altitude_m = float(origin.altitude_m)

    def update_local_position_velocity(self, position_ned: Any, velocity_ned: Any) -> None:
        """从 MAVSDK ``PositionNed`` 和 ``VelocityNed`` 对象复制字段。"""
        self.north_m = float(position_ned.north_m)
        self.east_m = float(position_ned.east_m)
        self.down_m = float(position_ned.down_m)
        self.vx_m_s = float(velocity_ned.north_m_s)
        self.vy_m_s = float(velocity_ned.east_m_s)
        self.vz_m_s = float(velocity_ned.down_m_s)

    def update_attitude_euler(self, euler: Any) -> None:
        """从 MAVSDK ``EulerAngle`` 对象复制字段。"""
        self.roll_deg = float(euler.roll_deg)
        self.pitch_deg = float(euler.pitch_deg)
        self.yaw_deg = float(euler.yaw_deg)
        self.attitude_timestamp_us = int(euler.timestamp_us)

    def update_wind(self, wind: Any) -> None:
        """从 MAVSDK ``Wind`` 对象复制风估计字段。

        ⚠ MAVSDK 的字段名是 ``wind_x/y/z_ned_m_s``（x=北、y=东、z=地），
        与快照里的 ``vx/vy/vz_m_s`` 命名习惯不同——这里显式对齐到 north/east/down。
        """
        self.wind_north_m_s = float(wind.wind_x_ned_m_s)
        self.wind_east_m_s = float(wind.wind_y_ned_m_s)
        self.wind_down_m_s = float(wind.wind_z_ned_m_s)

    def update_mission_progress(self, current: int, total: int) -> None:
        """任务进度（PX4 ``MISSION_CURRENT``）。

        ⚠ 语义来自 MAVSDK：``current`` 是0 基的任务项下标，``current == total``
        表示任务飞完。任务项一律走 ``mission_raw`` 上传，所以进度也只能从这里来——
        MAVSDK ``mission`` 插件的 ``is_mission_finished()`` 依赖它自己上传过的任务，
        走 raw 之后它对任务一无所知（实测：飞控已 "Mission finished, loitering"，
        它仍然回 ``False``）。
        """
        self.mission_current = int(current)
        self.mission_total = int(total)

    def update_flight_mode(self, mode: Any) -> None:
        """飞行模式（MAVSDK ``FlightMode`` 枚举）→ 字符串名字。

        ⚠ ``start_mission()`` 回成功不等于进了任务模式：飞控可能拒绝模式切换而
        只发 ACK（实测：``mission.start_mission()`` 不抛异常，模式却停在 ``HOLD``，
        导航器随后打印 ``No valid mission available, loitering``）。状态机靠这个字段
        做"真的进了 MISSION"的正向确认（见 :meth:`MissionRunner._tick_recon`）。
        """
        self.flight_mode = str(mode).split(".")[-1]

    def update_in_air(self, in_air: Any) -> None:
        """是否在空中（可选流）。

        正式任务的"等起飞"就靠它：只有真的离地才进侦查（在停机坪上就上传并启动
        侦查航线，等于让飞机在地面就想执行任务）。
        """
        self.in_air = bool(in_air)

    def update_attitude_quaternion(self, quaternion: Any) -> None:
        """从 MAVSDK ``Quaternion`` 对象复制字段。"""
        self.quaternion_w = float(quaternion.w)
        self.quaternion_x = float(quaternion.x)
        self.quaternion_y = float(quaternion.y)
        self.quaternion_z = float(quaternion.z)

    def is_valid(self) -> bool:
        """当至少存在一个位置/速度/姿态值时返回 True。"""
        return any(
            value is not None
            for value in (
                self.latitude_deg,
                self.north_m,
                self.vx_m_s,
                self.roll_deg,
                self.quaternion_w,
            )
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Command:
    """发送到 MAVSDK 工作线程的指令。

    ``name`` 是受支持指令之一（参见 ``MavsdkThread``）。
    ``params`` 包含该指令特有的关键字参数。
    """

    name: str
    params: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: uuid.uuid4().hex)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Command:
        return cls(
            name=str(data["name"]),
            params=dict(data.get("params") or {}),
            id=str(data.get("id") or uuid.uuid4().hex),
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CommandResult:
    """在 MAVSDK 工作线程中执行 :class:`Command` 的结果。"""

    id: str
    name: str
    success: bool
    error: str | None = None
    data: dict[str, Any] | None = None
    timestamp: float = field(default_factory=time.time)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CommandResult:
        return cls(
            id=str(data["id"]),
            name=str(data["name"]),
            success=bool(data["success"]),
            error=data.get("error"),
            data=data.get("data"),
            timestamp=float(data.get("timestamp", time.time())),
        )
