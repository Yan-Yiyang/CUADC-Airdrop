"""飞行记录与回放。

- :class:`FlightRecorder`：启动后把同进程内的对齐缓冲、遥测、事件、检测落盘，
  构成本次飞行的"五件套"目录（详见 :mod:`airdrop.record.recorder`）；
- :mod:`airdrop.record.replay`：P4 阶段实现，把录制的目录重放为可与原架构
  完全对接的"假实时"视频流与遥测代理。
"""

from .recorder import DetectionWriter, EventLog, FlightRecorder, RecorderStats

__all__ = [
    "DetectionWriter",
    "EventLog",
    "FlightRecorder",
    "RecorderStats",
]
