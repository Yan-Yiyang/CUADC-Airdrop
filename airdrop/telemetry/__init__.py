"""遥测与控制模块。

- :class:`MavsdkThread`：MAVSDK 专用工作线程（独立 asyncio 循环 + supervisor 重连）；
- :class:`TelemetryBroker`：线程安全遥测代理（最新快照合并 + 定采样率历史 + 内插/外推）；
- :mod:`airdrop.telemetry.models`：遥测快照与指令数据模型。

后续阶段在此加入 ``controller.py``（任务级控制：mission 上传/启动、hold、gripper）。
"""

from .broker import SUPPORTED_QUERY_MODES, TelemetryBroker
from .mavsdk_thread import MavsdkThread
from .models import Command, CommandResult, TelemetrySnapshot

__all__ = [
    "SUPPORTED_QUERY_MODES",
    "Command",
    "CommandResult",
    "MavsdkThread",
    "TelemetryBroker",
    "TelemetrySnapshot",
]
