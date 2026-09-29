"""视频处理：YOLO 检测 + （OCR 读编号 | 12 类直出编号）。

模块划分
--------
* :mod:`~airdrop.perception.detector`：YOLO 封装（显式 device、预计算去畸变）；
* :mod:`~airdrop.perception.cropproc`：裁剪图后处理（五边形 → 转正 → RapidOCR），
  移植自旧版实战实现；
* :mod:`~airdrop.perception.number`：编号纠错（收敛到 00–99）；
* :mod:`~airdrop.perception.ocr_worker`：OCR 独立工作进程；
* :mod:`~airdrop.perception.pipeline`：主循环（缓冲 → 检测 → 送检/直出 → Detection）。

重依赖（``ultralytics`` / ``rapidocr`` / ``torch``）都在使用时才导入，
所以 ``import airdrop.perception`` 在没有 GPU 栈的机器上也能成功——离线单测
依赖这一点。导出同样是惰性的（见 :mod:`airdrop._lazy`）：``cropproc`` / ``detector``
会拉 cv2，第一次取名字时才加载。
"""

from .._lazy import lazy_dir, lazy_exports

__getattr__ = lazy_exports(
    __name__,
    ("models", "number", "cropproc", "detector", "ocr_worker", "pipeline"),
)
__dir__ = lazy_dir(__name__)

__all__ = [
    "CropResult",
    "Detection",
    "DetectionBatch",
    "Detector",
    "DetectorConfig",
    "OCR_CHAR_MAP",
    "OcrEngine",
    "OcrEngineConfig",
    "OcrOrientation",
    "OcrRequest",
    "OcrResult",
    "OcrText",
    "OcrWorkerPool",
    "OpenCvPostProcess",
    "PerceptionConfigLike",
    "PerceptionStats",
    "PerceptionWorker",
    "PixelBox",
    "correct_ocr_number",
    "ocr_worker_main",
]
