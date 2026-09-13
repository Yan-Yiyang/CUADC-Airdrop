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

    # MAVSDK 时间戳（系统启动后的微秒数），仅在可用时填充
    attitude_timestamp_us: int | None = None

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

    def update_local_position_velocity(
        self, position_ned: Any, velocity_ned: Any
    ) -> None:
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
