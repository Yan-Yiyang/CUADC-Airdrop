"""遥测与控制模块。

- :class:`MavsdkThread`：MAVSDK 专用工作线程（独立 asyncio 循环 + supervisor 重连）；
- :class:`TelemetryBroker`：线程安全遥测代理（最新快照合并 + 定采样率历史 + 内插/外推）；
- :class:`DroneController`：任务级控制（mission 上传/启动、hold/rtl、gripper 投放、
  查 NED 原点）——状态机面向 :class:`MissionController` 协议编程，离线测试用假控制器；
- :mod:`airdrop.telemetry.models`：遥测快照与指令数据模型。

⚠ 惰性导出（见 :mod:`airdrop._lazy`）：import airdrop.telemetry 只执行本文件。
``mavsdk_thread`` / ``controller`` 会拉 mavsdk，所以排在最后——取快照数据模型不该顺带
加载飞控栈（CLI 的 ``--help`` 依赖这一点）。
"""

from .._lazy import lazy_dir, lazy_exports

__getattr__ = lazy_exports(__name__, ("models", "broker", "controller", "mavsdk_thread"))
__dir__ = lazy_dir(__name__)

__all__ = [
    "MISSION_TYPE_MISSION",
    "SUPPORTED_QUERY_MODES",
    "Command",
    "CommandResult",
    "ControllerError",
    "DroneController",
    "DryRunController",
    "MavsdkThread",
    "MissionController",
    "NedOrigin",
    "TelemetryBroker",
    "TelemetrySnapshot",
    "to_raw_item",
]
