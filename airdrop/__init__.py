"""AirDrop 无人机控制脚手架。

当前重点：
1. MAVSDK 运行在专用线程，并拥有自己的 asyncio 事件循环。
2. 遥测通过线程安全的进程内代理和可选的跨进程桥接发布。
3. 任意线程/进程都可以向 MAVSDK 工作线程发送指令。
4. 思翼 HM30 图传按"最新一帧"语义拉取，供后续视觉引导使用。
5. 图传帧按"收到时间 - 固定链路延时（默认 150ms）"对齐到拍摄时刻的遥测。
6. 对齐结果（画面 + 遥测）可存入环形缓冲，供其他模块回看/异步消费。

注意：本包在导入时就加载 :mod:`airdrop.video`，因此 ``cv2`` 也会被一并导入。
这带来约 0.4 秒的导入开销；如果只想用遥测部分，可以改成延迟导入。
（顺带设置 ``OPENCV_FFMPEG_CAPTURE_OPTIONS`` 只对 OpenCV 4.x 有意义，
OpenCV 5.0 已移除该变量，拉流默认走 ffmpeg 子进程后端。）
"""

from .alignment import AlignedSample, FrameTelemetryAligner
from .broker import TelemetryBroker
from .buffer import (
    DEFAULT_BUFFER_CAPACITY,
    AlignmentBuffer,
    BufferStats,
    BufferedFrame,
    capacity_for,
    raw_frame_bytes,
)
from .external_bridge import ExternalBridge
from .mavsdk_thread import MavsdkThread
from .models import Command, CommandResult, TelemetrySnapshot
from .video import (
    DEFAULT_TELEMETRY_LAG,
    HM30_CAMERA_IP,
    HM30_DEFAULT_RTSP,
    HM30_GROUND_IP,
    Hm30VideoSource,
    StreamInfo,
    VideoConfig,
    VideoFrame,
    VideoStats,
    open_hm30_video,
    probe_candidates,
    probe_stream,
)

__all__ = [
    "AlignedSample",
    "AlignmentBuffer",
    "BufferStats",
    "BufferedFrame",
    "Command",
    "CommandResult",
    "DEFAULT_BUFFER_CAPACITY",
    "DEFAULT_TELEMETRY_LAG",
    "ExternalBridge",
    "FrameTelemetryAligner",
    "HM30_CAMERA_IP",
    "HM30_DEFAULT_RTSP",
    "HM30_GROUND_IP",
    "Hm30VideoSource",
    "MavsdkThread",
    "StreamInfo",
    "TelemetryBroker",
    "TelemetrySnapshot",
    "VideoConfig",
    "VideoFrame",
    "VideoStats",
    "capacity_for",
    "open_hm30_video",
    "probe_candidates",
    "probe_stream",
    "raw_frame_bytes",
]
