"""飞行记录中的检测裁切图保存。"""

from __future__ import annotations

import json
import logging
import os
import queue
import threading
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .models import Detection

LOGGER = logging.getLogger(__name__)


class CropStore:
    """按检测保存 JPG，并追加可恢复的 JSONL 元数据索引。"""

    def __init__(
        self,
        directory: str | Path,
        *,
        quality: int = 95,
        async_write: bool = False,
        max_pending: int = 128,
    ) -> None:
        if not 1 <= quality <= 100:
            raise ValueError("JPG 质量必须在 1 到 100 之间")
        if max_pending < 1:
            raise ValueError("异步写盘队列上限必须为正整数")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._quality = quality
        self._async_write = bool(async_write)
        self._lock = threading.Lock()
        self._index = (self.directory / "index.jsonl").open("a", encoding="utf-8", newline="\n")
        self._queue: queue.Queue[tuple[Path, np.ndarray, dict[str, Any]] | None] | None = (
            queue.Queue(maxsize=max_pending) if self._async_write else None
        )
        self._thread: threading.Thread | None = None
        self._error: Exception | None = None
        self._closed = False

    def raise_if_failed(self) -> None:
        """若后台写盘线程已发生错误，立即向调用方重新抛出首个异常。"""
        with self._lock:
            error = self._error
        if error is not None:
            raise error

    def flush(self) -> None:
        """等待所有已入队的异步裁切落盘完成，并检查是否已锁存错误。"""
        if self._queue is not None:
            self._queue.join()
        self.raise_if_failed()

    def _ensure_writer(self) -> None:
        """惰性启动单一后台写盘线程，避免仅配置未写入时遗留线程。"""
        with self._lock:
            if self._error is not None:
                raise self._error
            if self._closed:
                raise RuntimeError("CropStore 已关闭，不能继续写入")
            if self._thread is None:
                thread = threading.Thread(
                    target=self._writer_loop,
                    name="crop-store-writer",
                    daemon=True,
                )
                thread.start()
                self._thread = thread

    def _writer_loop(self) -> None:
        assert self._queue is not None
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    return
                with self._lock:
                    failed = self._error is not None
                if failed:
                    continue
                path, image, record = item
                self._write_crop(path, image, record)
            except Exception as exc:
                with self._lock:
                    if self._error is None:
                        self._error = exc
                LOGGER.exception("异步保存检测裁切失败")
            finally:
                self._queue.task_done()

    def _write_crop(self, path: Path, image: np.ndarray, record: dict[str, Any]) -> None:
        """原子写一张 JPG，再写索引；失败时清理临时文件并抛出异常。"""
        temporary = path.with_suffix(".jpg.tmp")
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, self._quality])
        if not ok:
            raise OSError(f"JPG 编码失败: {path}")
        try:
            with temporary.open("wb") as stream:
                stream.write(memoryview(encoded))
            os.replace(temporary, path)
            line = json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
            with self._lock:
                self._index.write(line)
                self._index.flush()
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise

    def save(
        self,
        detection: Detection,
        ordinal: int,
        image: np.ndarray,
        bounds: tuple[int, int, int, int],
        *,
        undistorted: bool,
    ) -> Path:
        """原子写一张 JPG，再写索引；异步模式下入队由后台线程落盘。"""
        if image.size == 0:
            raise ValueError("检测裁切为空")
        self.raise_if_failed()
        name = f"frame-{detection.frame_index:09d}-d{ordinal:03d}_box.jpg"
        path = self.directory / name
        record: dict[str, Any] = {
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
        if self._queue is not None:
            self._ensure_writer()
            self._queue.put((path, image.copy(), record))
            return path
        self._write_crop(path, image, record)
        return path

    def close(self) -> None:
        """排空待写裁切、停止后台线程并关闭索引；若有锁存错误则抛出。"""
        flush_error: Exception | None = None
        try:
            self.flush()
        except Exception as exc:  # noqa: BLE001 - 先保留刷盘异常，再确保停止线程并关闭索引
            flush_error = exc
        finally:
            with self._lock:
                self._closed = True
                thread = self._thread
                self._thread = None
                pending_queue = self._queue
            if thread is not None and pending_queue is not None:
                pending_queue.put(None)
                thread.join()
            with self._lock:
                if not self._index.closed:
                    try:
                        self._index.flush()
                    except Exception as exc:  # noqa: BLE001 - 保留刷盘错误并继续关闭句柄
                        if self._error is None:
                            self._error = exc
                    finally:
                        try:
                            self._index.close()
                        except Exception as exc:  # noqa: BLE001 - 保留句柄关闭异常并统一上报
                            if self._error is None:
                                self._error = exc
                latched = self._error
        error = flush_error or latched
        if error is not None:
            raise error

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
        self.flush()
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
