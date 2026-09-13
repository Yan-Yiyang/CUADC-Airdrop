"""P5 真实素材端到端验证（转成用例）：视频帧 → YOLO → 裁剪 → OCR。

素材：旧版 实战视频 `flight.mp4` 的目标段。
"""
from __future__ import annotations

import os
from pathlib import Path

import cv2
import pytest

VIDEO = Path(r"input/flight.mp4")
WEIGHTS = Path("models/best2.pt")
MODELS_DIR = Path("models/ppocr")

pytestmark = pytest.mark.realdata


def _read_frame(capture: cv2.VideoCapture, index: int):
    capture.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = capture.read()
    return frame if ok else None


@pytest.mark.skipif(not VIDEO.is_file(), reason="缺少 旧版 实战视频素材")
@pytest.mark.skipif(not WEIGHTS.is_file(), reason="缺少 YOLO 权重 models/best2.pt")
@pytest.mark.skipif(not (MODELS_DIR / "PP-OCRv6_det_medium.pth").is_file(),
                    reason="缺少 PP-OCR 权重")
def test_real_video_reads_target_codes(tmp_path: Path) -> None:
    """实战视频的目标段：检出的真目标必须读出正确编号（56/95）。"""
    pytest.importorskip("ultralytics")
    pytest.importorskip("rapidocr")
    if os.environ.get("AIRDROP_SKIP_GPU"):
        pytest.skip("设了 AIRDROP_SKIP_GPU，跳过需要 GPU 的真实素材用例")

    from airdrop.perception.cropproc import OcrEngineConfig, OpenCvPostProcess
    from airdrop.perception.detector import Detector, DetectorConfig

    detector = Detector(DetectorConfig(model_path=str(WEIGHTS), device="0"))
    detector.load()
    post = OpenCvPostProcess(
        color="blue", ocr_conf_threshold=0.6, engine_config=OcrEngineConfig()
    )

    capture = cv2.VideoCapture(str(VIDEO))
    assert capture.isOpened()
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    results: dict[int, list[int | None]] = {}
    try:
        for index in (924, 1535, 1840):
            frame = _read_frame(capture, index)
            assert frame is not None, f"读不到第 {index} 帧"
            batch = detector.detect(frame, frame_index=index, mode="ocr")
            assert batch.detections, f"第 {index} 帧没有检出目标"
            codes = []
            for detection in batch.detections:
                crop = detector.crop(batch, detection)
                codes.append(post.recognize(crop).number)
            results[index] = codes
    finally:
        capture.release()

    # 每帧最高置信度的那个框应当读出编号；三帧分别是 56 / 56 / 95
    expected = {924: 56, 1535: 56, 1840: 95}
    for index, want in expected.items():
        codes = [c for c in results[index] if c is not None]
        assert want in codes, f"第 {index} 帧没读出 {want}（实际 {results[index]}）"


def test_weights_reject_junk_crop(tmp_path: Path) -> None:
    """纯色/无目标裁剪图必须读不出编号（不能瞎猜一个）。"""
    pytest.importorskip("rapidocr")
    if os.environ.get("AIRDROP_SKIP_GPU"):
        pytest.skip("设了 AIRDROP_SKIP_GPU")
    import numpy as np

    from airdrop.perception.cropproc import OcrEngineConfig, OpenCvPostProcess

    post = OpenCvPostProcess(color="blue", engine_config=OcrEngineConfig())
    blank = np.full((64, 64, 3), 120, np.uint8)
    result = post.recognize(blank)
    assert result.number is None
    assert result.stage in ("no_pentagon", "none", "empty")
