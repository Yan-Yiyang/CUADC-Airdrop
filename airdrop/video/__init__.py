"""视频接收模块。

- :mod:`airdrop.video.source`：HM30 图传拉流（ffmpeg 子进程唯一后端，sink 逐帧不丢）；
- :mod:`airdrop.video.align`：帧-遥测时间对齐（按拍摄时刻取遥测）；
- :mod:`airdrop.video.buffer`：对齐结果的 FIFO 环形缓冲（画面 + 拍摄时刻遥测）。
"""

from .align import AlignedSample, FrameTelemetryAligner
from .buffer import (
    DEFAULT_BUFFER_CAPACITY,
    AlignmentBuffer,
    AlignmentWriter,
    BufferStats,
    BufferedFrame,
    StoredFrame,
    capacity_for,
    raw_frame_bytes,
)
from .source import (
    DEFAULT_TELEMETRY_LAG,
    HM30_CAMERA_IP,
    HM30_DEFAULT_RTSP,
    HM30_GROUND_IP,
    Hm30VideoSource,
    VideoConfig,
    VideoFrame,
    VideoStats,
    open_hm30_video,
)

__all__ = [
    "AlignedSample",
    "AlignmentBuffer",
    "AlignmentWriter",
    "BufferStats",
    "BufferedFrame",
    "DEFAULT_BUFFER_CAPACITY",
    "DEFAULT_TELEMETRY_LAG",
    "FrameTelemetryAligner",
    "HM30_CAMERA_IP",
    "HM30_DEFAULT_RTSP",
    "HM30_GROUND_IP",
    "Hm30VideoSource",
    "VideoConfig",
    "VideoFrame",
    "VideoStats",
    "capacity_for",
    "open_hm30_video",
    "raw_frame_bytes",
]
