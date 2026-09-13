"""飞行记录与回放。

- :class:`FlightRecorder`：启动后把同进程内的对齐缓冲、遥测、事件、检测落盘，
  构成本次飞行的"五件套"目录（详见 :mod:`airdrop.record.recorder`）；
- :class:`ReplayVideoSource` / :func:`load_broker_from_log`：把录制的目录按
  **原时间轴**重放为可与原架构完全对接的"假实时"视频流与遥测代理
  （详见 :mod:`airdrop.record.replay`）——离线迭代 perception / georef /
  targeting 时，实飞与回放走同一条代码路径。
"""

from .recorder import DetectionWriter, EventLog, FlightRecorder, RecorderStats
from .replay import (
    FlightLog,
    FlightLogError,
    FrameIndexError,
    FrameRecord,
    ReplayStats,
    ReplayVideoSource,
    TelemetryPacer,
    load_broker_from_log,
)

__all__ = [
    "DetectionWriter",
    "EventLog",
    "FlightLog",
    "FlightLogError",
    "FlightRecorder",
    "FrameIndexError",
    "FrameRecord",
    "RecorderStats",
    "ReplayStats",
    "ReplayVideoSource",
    "TelemetryPacer",
    "load_broker_from_log",
]
