from __future__ import annotations

import json
from typing import Any, cast

import cv2
import numpy as np

from airdrop.perception.crop_store import CropStore
from airdrop.perception.models import Detection, PixelBox
from airdrop.telemetry.models import TelemetrySnapshot


def test_crop_store_writes_jpg95_and_detection_index(tmp_path) -> None:
    detection = Detection(
        frame_index=12,
        capture_timestamp=34.5,
        pixel=(10.0, 11.0),
        box=PixelBox(2.0, 3.0, 18.0, 19.0),
        confidence=0.91,
        telemetry=TelemetrySnapshot(),
        mode="cls12",
    )
    image = np.full((16, 16, 3), (20, 100, 220), dtype=np.uint8)
    store = CropStore(tmp_path)
    path = store.save(detection, 0, image, (2, 3, 18, 19), undistorted=True)
    final = store.finalize_ocr_result(
        path.name,
        number=None,
        text="",
        error="failed",
        archive_error=None,
        rectified_written=False,
        confidence=0.0,
        side_px=-1.0,
        stage="failed",
    )
    store.close()

    assert path.name == "frame-000000012-d000_box.jpg"
    assert final == path and path.is_file()
    decoded = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None
    assert decoded.shape == image.shape
    records = [
        json.loads(line)
        for line in (tmp_path / "index.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert records[0]["crop_bounds"] == [2, 3, 18, 19]
    assert records[0]["undistorted"] is True
    assert records[0]["mode"] == "cls12"
    assert records[-1]["crop"] == final.name


def test_crop_store_uses_rectified_candidate_and_keeps_one_final_jpg(tmp_path) -> None:
    detection = Detection(
        frame_index=3,
        capture_timestamp=4.0,
        pixel=(1.0, 1.0),
        box=PixelBox(0.0, 0.0, 2.0, 2.0),
        confidence=0.8,
        telemetry=TelemetrySnapshot(),
        mode="ocr",
    )
    store = CropStore(tmp_path)
    original = store.save(
        detection, 0, np.zeros((8, 8, 3), np.uint8), (0, 0, 8, 8), undistorted=False
    )
    rectified = original.with_name(original.stem.removesuffix("_box") + "_rectified.jpg")
    assert cv2.imwrite(str(rectified), np.full((8, 8, 3), 180, np.uint8))

    final = store.finalize_ocr_result(
        original.name,
        number=42,
        text="42",
        error=None,
        archive_error=None,
        rectified_written=True,
        confidence=0.95,
        side_px=12.0,
        stage="parallel",
    )
    store.close()

    assert final.name.endswith("_ocr.jpg")
    assert sorted(path.suffix for path in tmp_path.glob("*.jpg")) == [".jpg"]
    decoded = cv2.imdecode(np.fromfile(final, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded is not None and float(decoded.mean()) > 150


def test_crop_store_finalization_can_retry_after_index_write_failure(tmp_path) -> None:
    detection = Detection(
        frame_index=5,
        capture_timestamp=6.0,
        pixel=(1.0, 1.0),
        box=PixelBox(0.0, 0.0, 2.0, 2.0),
        confidence=0.8,
        telemetry=TelemetrySnapshot(),
        mode="ocr",
    )
    store = CropStore(tmp_path)
    original = store.save(
        detection, 0, np.zeros((4, 4, 3), np.uint8), (0, 0, 4, 4), undistorted=False
    )

    class FailOnce:
        failed = False

        def write(self, value: str) -> int:
            if not self.failed:
                self.failed = True
                raise OSError("injected index write failure")
            return store._index.write(value)

        def flush(self) -> None:
            store._index.flush()

    index = store._index
    store._index = cast(Any, FailOnce())
    try:
        try:
            store.finalize_ocr_result(
                original.name,
                number=42,
                text="42",
                error=None,
                archive_error=None,
                rectified_written=False,
                confidence=0.9,
                side_px=20.0,
                stage="parallel",
            )
        except OSError:
            pass
        else:
            raise AssertionError("应传播注入的索引写入错误")
        store._index = index
        final = store.finalize_ocr_result(
            original.name,
            number=42,
            text="42",
            error=None,
            archive_error=None,
            rectified_written=False,
            confidence=0.9,
            side_px=20.0,
            stage="parallel",
        )
    finally:
        store._index = index
        store.close()

    assert final.is_file()
    assert not original.exists()
    assert list(tmp_path.glob("*.jpg")) == [final]
