"""遥测与控制模块。

- :class:`MavsdkThread`：MAVSDK 专用工作线程（独立 asyncio 循环 + supervisor 重连）；
- :class:`TelemetryBroker`：线程安全遥测代理（最新快照合并 + 定采样率历史 + 内插/外推）；
- :class:`DroneController`：任务级控制（mission 上传/启动、hold/rtl、gripper 投放、
  查 NED 原点）——状态机面向 :class:`MissionController` 协议编程，离线测试用假控制器；
- :mod:`airdrop.telemetry.models`：遥测快照与指令数据模型。
"""

from .broker import SUPPORTED_QUERY_MODES, TelemetryBroker
from .controller import (
    ACTION_LAND,
    ACTION_NONE,
    ACTION_TAKEOFF,
    MISSION_ACTIONS,
    UNSET,
    ControllerError,
    DroneController,
    DryRunController,
    MissionController,
    MissionItemSpec,
    NedOrigin,
    to_mission_item,
)
from .mavsdk_thread import MavsdkThread
from .models import Command, CommandResult, TelemetrySnapshot

__all__ = [
    "ACTION_LAND",
    "ACTION_NONE",
    "ACTION_TAKEOFF",
    "MISSION_ACTIONS",
    "SUPPORTED_QUERY_MODES",
    "UNSET",
    "Command",
    "CommandResult",
    "ControllerError",
    "DroneController",
    "DryRunController",
    "MavsdkThread",
    "MissionController",
    "MissionItemSpec",
    "NedOrigin",
    "TelemetryBroker",
    "TelemetrySnapshot",
    "to_mission_item",
]
