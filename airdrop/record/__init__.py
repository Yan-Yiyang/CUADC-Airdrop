"""飞行记录与回放。

- :class:`FlightRecorder`：启动后把同进程内的对齐缓冲、遥测、事件、检测、投放记录
  写入磁盘，构成本次飞行的"五个记录文件 + 投放记录"目录（详见 :mod:`airdrop.record.recorder`）；
- :class:`ReplayVideoSource` / :func:`load_broker_from_log`：把录制的目录按
  原时间轴重放为可与原架构完全对接的"假实时"视频流与遥测代理
  （详见 :mod:`airdrop.record.replay`）——离线迭代 perception / georef /
  targeting 时，实飞与回放走同一条代码路径。

⚠ 惰性导出（见 :mod:`airdrop._lazy`）：import airdrop.record 只执行本文件。
``recorder``（帧写入磁盘用 cv2）与 ``replay``（解码用 cv2）都在第一次取名字时才加载。
"""

from .._lazy import lazy_dir, lazy_exports

__getattr__ = lazy_exports(__name__, ("recorder", "replay"))
__dir__ = lazy_dir(__name__)

__all__ = [
    "DROPS_NAME",
    "DetectionWriter",
    "DropWriter",
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
