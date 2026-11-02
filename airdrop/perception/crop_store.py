"""飞行记录中的检测裁切图保存。"""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path

import cv2
import numpy as np

from .models import Detection

LOGGER = logging.getLogger(__name__)


class CropStore:
    """按检测保存 JPG，并追加可恢复的 JSONL 元数据索引。"""

    def __init__(self, directory: str | Path, *, quality: int = 95) -> None:
        if not 1 <= quality <= 100:
            raise ValueError("JPG 质量必须在 1 到 100 之间")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._quality = quality
        self._lock = threading.Lock()
        self._index = (self.directory / "index.jsonl").open("a", encoding="utf-8", newline="\n")

    def save(
        self,
        detection: Detection,
        ordinal: int,
        image: np.ndarray,
        bounds: tuple[int, int, int, int],
        *,
        undistorted: bool,
    ) -> Path:
        """原子写一张 JPG，再写索引；写盘失败向调用方传播。"""
        if image.size == 0:
            raise ValueError("检测裁切为空")
        name = f"frame-{detection.frame_index:09d}-d{ordinal:03d}_box.jpg"
        path = self.directory / name
        temporary = path.with_suffix(".jpg.tmp")
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, self._quality])
        if not ok:
            raise OSError(f"JPG 编码失败: {path}")
        try:
            temporary.write_bytes(encoded.tobytes())
            os.replace(temporary, path)
            record = {
                "crop": name,
                "frame_index": detection.frame_index,
                "detection_index": ordinal,
                "capture_timestamp": detection.capture_timestamp,
                "box": detection.box.as_dict(),
                "crop_bounds": list(bounds),
                "undistorted": undistorted,
                "mode": detection.mode,
                "confidence": detection.confidence,
            }
            with self._lock:
                self._index.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                self._index.flush()
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        return path

    def close(self) -> None:
        with self._lock:
            if not self._index.closed:
                self._index.flush()
                self._index.close()

    def finalize_ocr_result(
        self,
        crop_name: str,
        *,
        number: int | None,
        text: str,
        error: str | None,
        archive_error: str | None,
        rectified_written: bool,
        confidence: float,
        side_px: float,
        stage: str,
    ) -> Path:
        """按实际识别阶段选最终图名，并清理该检测的临时备选图。"""
        original = self.directory / crop_name
        rectified = original.with_name(original.stem.removesuffix("_box") + "_rectified.jpg")
        if number is not None:
            source = rectified if rectified_written and rectified.is_file() else original
            suffix = "ocr"
        elif rectified_written and rectified.is_file():
            source = rectified
            suffix = "rectified"
        else:
            source = original
            suffix = "box"
        final_path = original.with_name(original.stem.removesuffix("_box") + f"_{suffix}.jpg")
        if source != final_path and source.is_file():
            os.replace(source, final_path)
        if not final_path.is_file():
            raise FileNotFoundError(f"检测裁切图缺失，无法完成归档: {final_path}")
        record = {
            "crop": final_path.name,
            "previous_crop": crop_name,
            "event": "ocr_result",
            "number": number,
            "raw_text": text,
            "error": error,
            "image_type": suffix,
            "archive_error": archive_error,
            "ocr_confidence": confidence,
            "side_px": side_px,
            "stage": stage,
        }
        with self._lock:
            self._index.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            self._index.flush()
        if original != final_path:
            original.unlink(missing_ok=True)
        if rectified != final_path:
            rectified.unlink(missing_ok=True)
        return final_path
