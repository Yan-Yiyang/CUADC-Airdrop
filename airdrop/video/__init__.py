"""视频接收模块。

- :mod:`airdrop.video.source`：RTSP 视频拉流（ffmpeg 子进程唯一后端，sink 逐帧不丢）；
- :mod:`airdrop.video.align`：帧-遥测时间对齐（按拍摄时刻取遥测）；
- :mod:`airdrop.video.buffer`：对齐结果的 FIFO 环形缓冲（画面 + 拍摄时刻遥测）。

⚠ 惰性导出（见 :mod:`airdrop._lazy`）：import airdrop.video 只执行本文件。
``buffer`` 会拉 cv2，所以它排在最后——"取个 VideoConfig"不该顺带加载 OpenCV。
"""

from .._lazy import lazy_dir, lazy_exports

__getattr__ = lazy_exports(__name__, ("source", "align", "buffer"))
__dir__ = lazy_dir(__name__)

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
