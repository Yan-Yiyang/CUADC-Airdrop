"""AirDrop 无人机控制脚手架。

当前重点：
1. MAVSDK 运行在专用线程，并拥有自己的 asyncio 事件循环。
2. 遥测通过线程安全的进程内代理和可选的跨进程桥接发布。
3. 任意线程/进程都可以向 MAVSDK 工作线程发送指令。
"""

from .broker import TelemetryBroker
from .external_bridge import ExternalBridge
from .mavsdk_thread import MavsdkThread
from .models import Command, CommandResult, TelemetrySnapshot

__all__ = [
    "Command",
    "CommandResult",
    "ExternalBridge",
    "MavsdkThread",
    "TelemetryBroker",
    "TelemetrySnapshot",
]
