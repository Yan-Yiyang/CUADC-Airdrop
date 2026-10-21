"""对任意 JPEG 目录抽样，测量生产 YOLO 检测器的分段耗时。"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

IMAGES_DIR = None
OUTPUT_PATH = None
CALIB_FILE = None
MODEL_PATH = None
DEVICE = "0"
IMGSZ = 1280
THREADS = 4
SAMPLE_COUNT = 200


def _sample(paths: list[Path], count: int) -> list[Path]:
    if count < 1:
        raise ValueError("sample_count 必须至少为 1")
    if count >= len(paths):
        return paths
    if count == 1:
        return [paths[0]]
    indices = [round(i * (len(paths) - 1) / (count - 1)) for i in range(count)]
    return [paths[index] for index in indices]


def main(
    *,
    images_dir: Path | None,
    output_path: Path | None,
    calib_file: Path | None = None,
    model_path: Path | None = None,
    device: str = "0",
    imgsz: int = 1280,
    threads: int = 4,
    sample_count: int = 200,
) -> int:
    """加载 YOLO 后设置 OpenCV 线程，分别记录冷帧与抽样帧分段时间。"""
    if images_dir is None or output_path is None:
        raise ValueError("必须指定 --images-dir 和 --out")
    if output_path.exists():
        raise FileExistsError(f"拒绝覆盖已有诊断结果：{output_path}")
    if not images_dir.is_dir():
        raise FileNotFoundError(f"图像目录不存在：{images_dir}")
    if imgsz < 1 or threads < 1 or sample_count < 1:
        raise ValueError("imgsz、threads、sample_count 必须为正整数")
    paths = sorted(
        path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg"}
    )
    if not paths:
        raise ValueError(f"目录没有 JPEG 图像：{images_dir}")
    selected = _sample(paths, sample_count)

    import cv2
    import numpy as np

    from airdrop.config import Config
    from airdrop.georef import load_camera_model
    from airdrop.perception.detector import Detector, DetectorConfig
    from tools.performance import percentile_summary, write_json_new

    effective_model = (
        Path(model_path) if model_path is not None else Path(Config().perception.model_path)
    )
    camera_kwargs: dict[str, Any] = {"undistort": False}
    if calib_file is not None:
        camera = load_camera_model(str(calib_file))
        camera_kwargs = {
            "camera_matrix": camera.camera_matrix,
            "dist_coeffs": camera.dist_coeffs,
            "calib_size": camera.image_size,
            "undistort": True,
        }
    detector = Detector(
        DetectorConfig(model_path=str(effective_model), device=device, imgsz=imgsz, **camera_kwargs)
    )
    load_started = time.perf_counter_ns()
    detector.load()
    model_load_ms = (time.perf_counter_ns() - load_started) / 1_000_000
    cv2.setNumThreads(threads)
    actual_threads = cv2.getNumThreads()

    def measure(path: Path) -> dict[str, Any]:
        overall = time.perf_counter_ns()
        read_start = time.perf_counter_ns()
        encoded = path.read_bytes()
        read_ms = (time.perf_counter_ns() - read_start) / 1_000_000
        decode_start = time.perf_counter_ns()
        image = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
        decode_ms = (time.perf_counter_ns() - decode_start) / 1_000_000
        if image is None:
            raise ValueError(f"JPEG 解码失败：{path}")
        remap_ms: float | None = None
        predict_ms: float | None = None
        result_speed: dict[str, float] | None = None
        original_undistort = detector.undistort
        model = detector._model  # noqa: SLF001 - 仅临时计时包裹已加载的模型
        original_predict = model.predict

        def timed_undistort(image: np.ndarray):
            nonlocal remap_ms
            start = time.perf_counter_ns()
            result = original_undistort(image)
            remap_ms = (time.perf_counter_ns() - start) / 1_000_000
            return result

        def timed_predict(*args: Any, **kwargs: Any):
            nonlocal predict_ms, result_speed
            start = time.perf_counter_ns()
            result = original_predict(*args, **kwargs)
            predict_ms = (time.perf_counter_ns() - start) / 1_000_000
            if result and isinstance(getattr(result[0], "speed", None), dict):
                result_speed = {str(k): float(v) for k, v in result[0].speed.items()}
            return result

        detector.undistort = timed_undistort
        model.predict = timed_predict
        try:
            detect_start = time.perf_counter_ns()
            batch = detector.detect(image)
            detect_ms = (time.perf_counter_ns() - detect_start) / 1_000_000
        finally:
            detector.undistort = original_undistort
            model.predict = original_predict
        return {
            "image": path.name,
            "jpeg_bytes": len(encoded),
            "shape": list(image.shape),
            "read_ms": read_ms,
            "decode_ms": decode_ms,
            "remap_ms": remap_ms,
            "predict_ms": predict_ms,
            "ultralytics_speed_ms": result_speed,
            "detect_total_ms": detect_ms,
            "read_decode_detect_ms": (time.perf_counter_ns() - overall) / 1_000_000,
            "detections": len(batch.detections),
        }

    cold = measure(selected[0])
    warmup_started = time.perf_counter_ns()
    warmup_ok = detector.warmup(int(cold["shape"][1]), int(cold["shape"][0]))
    warmup_ms = (time.perf_counter_ns() - warmup_started) / 1_000_000
    if not warmup_ok:
        raise RuntimeError("YOLO 预热失败")
    rows = [measure(path) for path in selected[1:]]
    result = {
        "format_version": 1,
        "parameters": {
            "images_dir": str(images_dir),
            "calib_file": str(calib_file) if calib_file else None,
            "model_path": str(effective_model),
            "device": device,
            "imgsz": imgsz,
            "requested_threads": threads,
            "actual_threads": actual_threads,
            "sample_count": len(selected),
        },
        "model_load_ms": model_load_ms,
        "warmup_ms": warmup_ms,
        "cold_frame": cold,
        "steady_samples": rows,
        "steady_summary": {
            key: percentile_summary(float(row[key]) for row in rows if row[key] is not None)
            for key in (
                "read_ms",
                "decode_ms",
                "remap_ms",
                "predict_ms",
                "detect_total_ms",
                "read_decode_detect_ms",
            )
        },
    }
    write_json_new(output_path, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0
