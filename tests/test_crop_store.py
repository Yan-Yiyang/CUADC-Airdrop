from __future__ import annotations

import json
import threading
import time
from typing import Any, cast

import cv2
import numpy as np
import pytest

from airdrop.perception.crop_store import CropStore
from airdrop.perception.detector import DetectionBatch, Detector
from airdrop.perception.models import Detection, PixelBox
from airdrop.perception.pipeline import PerceptionConfigLike, PerceptionWorker
from airdrop.telemetry.models import TelemetrySnapshot
from airdrop.video.buffer import AlignedSample, AlignmentBuffer
from airdrop.video.source import VideoFrame


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


def test_crop_store_async_write_end_to_end_preserves_files_and_index_order(tmp_path) -> None:
    """异步写盘端到端验证：惰性启动线程、保存副本、归档重命名与索引顺序一致。"""
    det0 = Detection(
        frame_index=10,
        capture_timestamp=11.5,
        pixel=(4.0, 5.0),
        box=PixelBox(1.0, 2.0, 9.0, 10.0),
        confidence=0.92,
        telemetry=TelemetrySnapshot(),
        mode="ocr",
    )
    det1 = Detection(
        frame_index=10,
        capture_timestamp=11.5,
        pixel=(20.0, 21.0),
        box=PixelBox(16.0, 17.0, 24.0, 25.0),
        confidence=0.87,
        telemetry=TelemetrySnapshot(),
        mode="ocr",
    )
    store = CropStore(tmp_path, async_write=True)
    assert store._thread is None

    crop0 = np.full((12, 12, 3), (30, 140, 210), dtype=np.uint8)
    crop1 = np.full((12, 12, 3), (10, 20, 30), dtype=np.uint8)
    path0 = store.save(det0, 0, crop0, (1, 2, 9, 10), undistorted=True)
    # 立即覆写调用方缓冲区，验证异步入队前已独立拷贝图像。
    crop0[:] = 0
    path1 = store.save(det1, 1, crop1, (16, 17, 24, 25), undistorted=False)
    assert store._thread is not None
    assert store._thread.name == "crop-store-writer"
    assert store._thread.daemon

    rectified1 = path1.with_name(path1.stem.removesuffix("_box") + "_rectified.jpg")
    assert cv2.imwrite(str(rectified1), np.full((12, 12, 3), 200, dtype=np.uint8))

    final0 = store.finalize_ocr_result(
        path0.name,
        number=None,
        text="",
        error="failed",
        archive_error=None,
        rectified_written=False,
        confidence=0.0,
        side_px=-1.0,
        stage="failed",
    )
    final1 = store.finalize_ocr_result(
        path1.name,
        number=42,
        text="42",
        error=None,
        archive_error=None,
        rectified_written=True,
        confidence=0.96,
        side_px=15.0,
        stage="parallel",
    )
    store.close()

    assert store._thread is None
    assert store._index.closed
    assert final0 == path0 and final0.is_file()
    assert final1.name == "frame-000000010-d001_ocr.jpg" and final1.is_file()
    assert not path1.exists()
    assert not rectified1.exists()
    assert not list(tmp_path.glob("*.tmp"))

    decoded0 = cv2.imdecode(np.fromfile(final0, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded0 is not None and float(decoded0[:, :, 2].mean()) > 180
    decoded1 = cv2.imdecode(np.fromfile(final1, dtype=np.uint8), cv2.IMREAD_COLOR)
    assert decoded1 is not None and float(decoded1.mean()) > 170

    records = [
        json.loads(line)
        for line in (tmp_path / "index.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(records) == 4
    assert [(r["crop"], r.get("event")) for r in records] == [
        ("frame-000000010-d000_box.jpg", None),
        ("frame-000000010-d001_box.jpg", None),
        ("frame-000000010-d000_box.jpg", "ocr_result"),
        ("frame-000000010-d001_ocr.jpg", "ocr_result"),
    ]


def test_crop_store_async_write_propagates_writer_error(tmp_path, monkeypatch) -> None:
    """后台写盘异常必须锁存并通过 flush / save / finalize / close 传播且不死锁。"""
    detection = Detection(
        frame_index=7,
        capture_timestamp=8.0,
        pixel=(4.0, 4.0),
        box=PixelBox(0.0, 0.0, 8.0, 8.0),
        confidence=0.85,
        telemetry=TelemetrySnapshot(),
        mode="cls12",
    )
    image = np.full((8, 8, 3), 90, dtype=np.uint8)
    store = CropStore(tmp_path, async_write=True)

    def fail_write(*_args: object, **_kwargs: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(store, "_write_crop", fail_write)
    path = store.save(detection, 0, image, (0, 0, 8, 8), undistorted=False)

    with pytest.raises(OSError, match="disk full"):
        store.flush()
    with pytest.raises(OSError, match="disk full"):
        store.save(detection, 1, image, (0, 0, 8, 8), undistorted=False)
    with pytest.raises(OSError, match="disk full"):
        store.finalize_ocr_result(
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
    with pytest.raises(OSError, match="disk full"):
        store.close()

    assert store._thread is None
    assert store._index.closed


def test_perception_worker_drain_fails_on_async_crop_store_error(tmp_path, monkeypatch) -> None:
    """感知排空时须在锁外 flush 裁切存储，并将异步落盘失败记为 drain 失败与错误状态。"""

    class _SingleDetectionDetector(Detector):
        def load(self) -> Detector:
            return self

        def detect(self, image: np.ndarray, **kwargs: Any) -> DetectionBatch:
            box = PixelBox(2.0, 2.0, 10.0, 10.0)
            return DetectionBatch(
                image=image,
                detections=(
                    Detection(
                        frame_index=int(kwargs["frame_index"]),
                        capture_timestamp=float(kwargs["capture_timestamp"]),
                        pixel=box.center,
                        box=box,
                        confidence=0.9,
                        telemetry=kwargs["telemetry"],
                        code=5,
                        mode="cls12",
                        extra={"image_size": (image.shape[1], image.shape[0])},
                    ),
                ),
            )

    buffer = AlignmentBuffer(capacity=8, storage="raw")
    worker = PerceptionWorker(
        PerceptionConfigLike(mode="cls12"),
        buffer=buffer,
        detector=_SingleDetectionDetector(),
    )
    worker.configure_crop_storage(tmp_path / "crops")
    store = worker._crop_store
    assert store is not None
    assert store._async_write is True
    assert store._thread is None

    entered = threading.Event()
    release = threading.Event()

    def slow_failing_write(*_args: object, **_kwargs: object) -> None:
        entered.set()
        release.wait(5.0)
        raise OSError("disk full")

    monkeypatch.setattr(store, "_write_crop", slow_failing_write)

    stamp = time.time()
    buffer.put(
        AlignedSample(
            frame=VideoFrame(
                index=1,
                image=np.zeros((16, 16, 3), dtype=np.uint8),
                timestamp=stamp,
                lag=0.0,
            ),
            snapshot=TelemetrySnapshot(),
            timestamp=stamp,
            lag=0.0,
            mode="interpolate",
        )
    )
    worker.start()
    try:
        assert entered.wait(5.0)
        # 在后台释放失败信号，使 drain 阻塞于 crop_store.flush() 后捕获异常。
        releaser = threading.Timer(0.05, release.set)
        releaser.start()
        try:
            assert not worker.drain(timeout=5.0)
        finally:
            releaser.cancel()
        assert "disk full" in (worker.stats.last_error or "")
        assert "disk full" in (worker._fatal_error or "")
    finally:
        release.set()
        with pytest.raises(OSError, match="disk full"):
            worker.stop(timeout=5.0)
    assert not worker.running
    assert worker._crop_store is None
