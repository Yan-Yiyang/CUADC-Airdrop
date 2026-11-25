"""TensorRT FP16 OCR 后端与致命异常传播离线单元测试。"""

from __future__ import annotations

import queue
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
import pytest

from airdrop.perception import timing
from airdrop.perception.cropproc import OcrEngine, OcrEngineConfig, OpenCvPostProcess
from airdrop.perception.detector import DetectionBatch, Detector, FrameSizeMismatchError
from airdrop.perception.models import Detection, PixelBox
from airdrop.perception.ocr_worker import OcrRequest, OcrResult, OcrWorkerPool, ocr_worker_main
from airdrop.perception.pipeline import PerceptionConfigLike, PerceptionWorker
from airdrop.perception.trt_ocr import TrtOcrBackend, TrtTextDetector, TrtTextRecognizer
from airdrop.perception.trt_runtime import (
    TensorSpec,
    TrtEngineError,
    TrtExecutionRunner,
    TrtManifest,
    TrtShapeOutOfRangeError,
    sha256_file,
)
from airdrop.telemetry.models import TelemetrySnapshot
from airdrop.video.buffer import AlignedSample, AlignmentBuffer
from airdrop.video.source import VideoFrame


class _MockTrtRunner:
    """离线测试用模拟 TensorRT 执行器（实现 TrtExecutionRunner 核心契约）。"""

    def __init__(
        self,
        manifest: TrtManifest,
        fn: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> None:
        self._manifest = manifest
        self._fn = fn
        self.calls: list[tuple[tuple[int, ...], str]] = []

    @property
    def manifest(self) -> TrtManifest:
        return self._manifest

    @property
    def input_spec(self) -> TensorSpec:
        return self._manifest.inputs[0]

    @property
    def output_spec(self) -> TensorSpec:
        return self._manifest.outputs[0]

    def run_numpy(self, array: np.ndarray) -> np.ndarray:
        self.calls.append((tuple(int(d) for d in array.shape), str(array.dtype)))
        if self._fn is not None:
            return self._fn(array)
        h, w = array.shape[2], array.shape[3]
        return np.zeros((array.shape[0], 1, h, w), dtype=np.float32)

    def clone_context(self) -> _MockTrtRunner:
        return _MockTrtRunner(self._manifest, fn=self._fn)


def _make_det_manifest(
    *,
    max_h: int = 960,
    max_w: int = 960,
    ocr_det: dict[str, Any] | None = None,
    omit_ocr_det: bool = False,
) -> TrtManifest:
    model: dict[str, Any] = {
        "inputs": [],
        "outputs": [],
    }
    if not omit_ocr_det:
        model["ocr_det"] = (
            dict(ocr_det) if ocr_det is not None else {"limit_type": "min", "limit_side_len": 736}
        )
    return TrtManifest(
        schema_version=1,
        kind="ocr_det",
        engine_file="det.engine",
        engine_sha256="0" * 64,
        source={},
        model=model,
        environment={
            "platform": sys.platform,
            "tensorrt_version": "10.0",
            "compute_capability": "sm89",
        },
        verification={},
        inputs=(
            TensorSpec(
                name="x",
                dtype="float32",
                shape=(1, 3, -1, -1),
                min_shape=(1, 3, 32, 32),
                opt_shape=(1, 3, 736, 736),
                max_shape=(1, 3, max_h, max_w),
            ),
        ),
        outputs=(TensorSpec(name="y", dtype="float32", shape=(1, 1, -1, -1)),),
    )


def _make_rec_manifest(
    *,
    dict_sha256: str | None,
    max_batch: int = 6,
    max_w: int = 2048,
    rec_image_shape: list[int] | None = None,
    omit_ocr_rec: bool = False,
) -> TrtManifest:
    source: dict[str, Any] = {}
    if dict_sha256 is not None:
        source["dict_sha256"] = dict_sha256
    model: dict[str, Any] = {
        "inputs": [],
        "outputs": [],
    }
    if not omit_ocr_rec:
        model["ocr_rec"] = {
            "rec_image_shape": [3, 48, 320] if rec_image_shape is None else rec_image_shape
        }
    return TrtManifest(
        schema_version=1,
        kind="ocr_rec",
        engine_file="rec.engine",
        engine_sha256="1" * 64,
        source=source,
        model=model,
        environment={
            "platform": sys.platform,
            "tensorrt_version": "10.0",
            "compute_capability": "sm89",
        },
        verification={},
        inputs=(
            TensorSpec(
                name="x",
                dtype="float32",
                shape=(-1, 3, 48, -1),
                min_shape=(1, 3, 48, 32),
                opt_shape=(6, 3, 48, 320),
                max_shape=(max_batch, 3, 48, max_w),
            ),
        ),
        outputs=(TensorSpec(name="y", dtype="float32", shape=(-1, -1, 12)),),
    )


def _write_trt_dummy_files(tmp_path: Path) -> tuple[OcrEngineConfig, str]:
    det_file = tmp_path / "det.engine"
    rec_file = tmp_path / "rec.engine"
    keys_file = tmp_path / "digits_dict.txt"
    det_file.write_bytes(b"det")
    rec_file.write_bytes(b"rec")
    keys_file.write_text("\n".join(str(i) for i in range(10)) + "\n", encoding="utf-8")
    config = OcrEngineConfig(
        model_dir=str(tmp_path),
        det_model="det.engine",
        rec_model="rec.engine",
        rec_keys="digits_dict.txt",
        use_cuda=True,
        cls_engine="off",
        cls_autorotate=False,
        det_max_side_len=960,
        text_score=0.5,
        rec_batch_num=2,
    )
    return config, sha256_file(keys_file)


def _encode_ctc_logits(
    chars: list[str],
    sequences: list[tuple[str, float]],
    *,
    steps: int = 8,
) -> np.ndarray:
    """构造 CTC 输出概率张量，供 CTCLabelDecode 确定性解码。"""
    char_to_idx = {ch: idx for idx, ch in enumerate(chars)}
    vocab = len(chars)
    batch = len(sequences)
    logits = np.full((batch, steps, vocab), 0.01, dtype=np.float32)
    logits[:, :, 0] = 0.99  # 默认 blank
    for b_idx, (text, conf) in enumerate(sequences):
        pos = 0
        for ch in text:
            logits[b_idx, pos, :] = 0.01
            logits[b_idx, pos, char_to_idx[ch]] = float(conf)
            pos += 2  # 中间保留 blank 以便区分相同连续字符
    return logits


def _blue_pentagon_image() -> np.ndarray:
    """构造可稳定通过蓝色五边形几何转正的合成目标图。"""
    canvas = np.full((240, 240, 3), 240, dtype=np.uint8)
    side = 80.0
    tri = side * 0.8660254037844386
    x0, y0 = 80.0, 100.0
    pts = np.array(
        [
            [x0 + side / 2.0, y0 - tri],
            [x0, y0],
            [x0, y0 + side],
            [x0 + side, y0 + side],
            [x0 + side, y0],
        ],
        dtype=np.int32,
    )
    # BGR 鲜蓝色（H≈120, S=255, V=220）
    cv2.fillPoly(canvas, [pts], (220, 30, 20))
    return canvas


def _buffer_with(count: int) -> AlignmentBuffer:
    buffer = AlignmentBuffer(capacity=max(count + 2, 8))
    for idx in range(1, count + 1):
        buffer.put(
            AlignedSample(
                frame=VideoFrame(
                    index=idx,
                    image=np.zeros((64, 64, 3), dtype=np.uint8),
                    timestamp=1000.0 + idx,
                ),
                snapshot=TelemetrySnapshot(timestamp=1000.0 + idx),
                timestamp=1000.0 + idx,
                lag=0.0,
                mode="interpolate",
            )
        )
    return buffer


class _FakeDetector:
    def __init__(self) -> None:
        self.calls: list[int] = []

    def load(self) -> "_FakeDetector":
        return self

    def detect(
        self,
        image: np.ndarray,
        *,
        frame_index: int = 0,
        capture_timestamp: float = 0.0,
        telemetry: TelemetrySnapshot | None = None,
        mode: str = "",
        undistort: bool | None = None,
    ) -> DetectionBatch:
        del undistort
        self.calls.append(frame_index)
        snapshot = telemetry if telemetry is not None else TelemetrySnapshot()
        box = PixelBox(10.0, 10.0, 50.0, 50.0)
        detection = Detection(
            frame_index=frame_index,
            capture_timestamp=capture_timestamp,
            pixel=box.center,
            box=box,
            confidence=0.9,
            telemetry=snapshot,
            code=1 if mode == "cls12" else None,
            mode=mode,
        )
        return DetectionBatch(image=image, detections=(detection,))

    @staticmethod
    def crop_with_bounds(
        batch: DetectionBatch, detection: Detection
    ) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        del detection
        return batch.image[10:50, 10:50].copy(), (10, 10, 50, 50)


# ----------------------------------------------------------------------
# OcrEngineConfig 路径解析与校验
# ----------------------------------------------------------------------
def test_ocr_engine_config_path_resolution_and_backend_kind(tmp_path: Path) -> None:
    pure = OcrEngineConfig(
        model_dir="models/ppocr",
        det_model="det.engine",
        rec_model="rec.engine",
        rec_keys="dict.txt",
    )
    assert pure.backend_kind() == "tensorrt"
    assert pure.det_path() == Path("models/ppocr/det.engine")
    assert pure.rec_path() == Path("models/ppocr/rec.engine")
    assert pure.keys_path() == Path("models/ppocr/dict.txt")

    rel = OcrEngineConfig(
        model_dir="models/ppocr",
        det_model="trt_bundle/det.engine",
        rec_model="trt_bundle/rec.engine",
        rec_keys="dicts/v6.txt",
    )
    assert rel.det_path() == Path("trt_bundle/det.engine")
    assert rel.rec_path() == Path("trt_bundle/rec.engine")
    assert rel.keys_path() == Path("dicts/v6.txt")

    abs_det = tmp_path / "abs_det.pth"
    abs_rec = tmp_path / "abs_rec.pth"
    abs_keys = tmp_path / "abs_dict.txt"
    absolute = OcrEngineConfig(
        model_dir="ignored_dir",
        det_model=str(abs_det),
        rec_model=str(abs_rec),
        rec_keys=str(abs_keys),
    )
    assert absolute.backend_kind() == "torch"
    assert absolute.det_path() == abs_det
    assert absolute.rec_path() == abs_rec
    assert absolute.keys_path() == abs_keys


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        pytest.param(
            {"det_model": "det.pth", "rec_model": "rec.engine"},
            "混用",
            id="mix-pth-engine",
        ),
        pytest.param(
            {"det_model": "det.engine", "rec_model": "rec.pth"},
            "混用",
            id="mix-engine-pth",
        ),
        pytest.param(
            {"det_model": "det.onnx", "rec_model": "rec.onnx"},
            "仅支持",
            id="unsupported-onnx",
        ),
        pytest.param(
            {"det_model": "det.engine", "rec_model": "rec.engine", "use_cuda": False},
            "use_cuda",
            id="trt-cpu",
        ),
        pytest.param(
            {"det_model": "det.engine", "rec_model": "rec.engine", "cls_engine": "onnx"},
            "cls_engine",
            id="trt-cls-onnx",
        ),
        pytest.param(
            {"det_model": "det.engine", "rec_model": "rec.engine", "cls_engine": "torch"},
            "cls_engine",
            id="trt-cls-torch",
        ),
        pytest.param(
            {"det_model": "det.engine", "rec_model": "rec.engine", "cls_autorotate": True},
            "cls_autorotate",
            id="trt-cls-autorotate",
        ),
        pytest.param(
            {"det_model": "det.engine", "rec_model": "rec.engine", "det_max_side_len": 0},
            "det_max_side_len",
            id="trt-det-max-side-zero",
        ),
    ],
)
def test_ocr_engine_config_validation_rejects_invalid_trt_combinations(
    kwargs: dict[str, Any], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        OcrEngineConfig(**kwargs)


# ----------------------------------------------------------------------
# OcrEngine TRT 加载、隔离、清单校验与错误传播
# ----------------------------------------------------------------------
def test_ocr_engine_trt_missing_files_raises_and_marks_failed(tmp_path: Path) -> None:
    default_cfg = OcrEngineConfig()
    assert default_cfg.backend_kind() == "tensorrt"
    assert default_cfg.det_model == "models/trt/PP-OCRv6_det_medium_fp16.engine"
    assert default_cfg.rec_model == "models/trt/PP-OCRv6_rec_medium_fp16.engine"

    config = OcrEngineConfig(
        model_dir=str(tmp_path),
        det_model="missing_det.engine",
        rec_model="missing_rec.engine",
        rec_keys="missing_dict.txt",
    )
    engine = OcrEngine(config)
    with pytest.raises(FileNotFoundError, match="OCR 模型文件缺失") as exc_info:
        engine.ensure()
    assert "build-trt --target all" in str(exc_info.value)
    assert "--backend pytorch" in str(exc_info.value)
    assert engine.failed is True
    # TensorRT 缺失引擎严禁静默降级返回空列表，read_text 与二次 ensure 均继续抛出。
    with pytest.raises(FileNotFoundError, match="build-trt --target all"):
        engine.read_text(np.zeros((32, 32, 3), dtype=np.uint8))


def test_ocr_engine_trt_loads_without_onnxruntime_or_rapidocr(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, dict_sha = _write_trt_dummy_files(tmp_path)
    det_runner = _MockTrtRunner(_make_det_manifest())
    rec_runner = _MockTrtRunner(_make_rec_manifest(dict_sha256=dict_sha))

    # 阻断 onnxruntime 与 RapidOCR 主模块导入，确保 TensorRT 路径完全不触发它们
    monkeypatch.setitem(sys.modules, "onnxruntime", cast(Any, None))
    monkeypatch.setitem(sys.modules, "rapidocr.main", cast(Any, None))

    loaded_kinds: list[tuple[Path, str, int]] = []

    def fake_load_trt_runner(
        engine_path: str | Path, *, expected_kind: str, device_id: int = 0
    ) -> _MockTrtRunner:
        loaded_kinds.append((Path(engine_path), expected_kind, device_id))
        if expected_kind == "ocr_det":
            return det_runner
        if expected_kind == "ocr_rec":
            return rec_runner
        raise AssertionError(f"意外 kind: {expected_kind}")

    monkeypatch.setattr("airdrop.perception.trt_runtime.load_trt_runner", fake_load_trt_runner)

    engine = OcrEngine(config)
    backend = engine.ensure()
    assert isinstance(backend, TrtOcrBackend)
    assert engine.available is True
    assert loaded_kinds == [
        (config.det_path(), "ocr_det", config.device_id),
        (config.rec_path(), "ocr_rec", config.device_id),
    ]
    assert sys.modules["onnxruntime"] is None
    assert sys.modules["rapidocr.main"] is None


@pytest.mark.parametrize(
    ("manifest_kwargs", "match"),
    [
        pytest.param({"omit_ocr_det": True}, "model.ocr_det", id="missing-ocr-det"),
        pytest.param(
            {"ocr_det": {"limit_type": "max", "limit_side_len": 736}},
            "limit_type",
            id="wrong-limit-type",
        ),
        pytest.param(
            {"ocr_det": {"limit_type": "min", "limit_side_len": 960}},
            "limit_side_len",
            id="wrong-limit-side-len",
        ),
        pytest.param({"max_h": 736, "max_w": 960}, "det_max_side_len", id="max-h-too-small"),
        pytest.param({"max_h": 960, "max_w": 736}, "det_max_side_len", id="max-w-too-small"),
    ],
)
def test_trt_text_detector_validates_manifest_and_profile(
    tmp_path: Path, manifest_kwargs: dict[str, Any], match: str
) -> None:
    config, _ = _write_trt_dummy_files(tmp_path)
    runner = _MockTrtRunner(_make_det_manifest(**manifest_kwargs))
    with pytest.raises(TrtEngineError, match=match):
        TrtTextDetector(config, runner=cast(TrtExecutionRunner, runner))


@pytest.mark.parametrize(
    ("manifest_overrides", "match"),
    [
        pytest.param({"dict_sha256": None}, "dict_sha256", id="missing-dict-sha"),
        pytest.param({"dict_sha256": "f" * 64}, "SHA256", id="wrong-dict-sha"),
        pytest.param({"omit_ocr_rec": True}, "model.ocr_rec", id="missing-ocr-rec"),
        pytest.param({"rec_image_shape": [3, 32, 320]}, "rec_image_shape", id="wrong-rec-shape"),
        pytest.param({"max_batch": 1}, "rec_batch_num", id="batch-exceeds-max"),
        pytest.param({"max_w": 256}, "rec_image_shape", id="max-w-below-320"),
    ],
)
def test_trt_text_recognizer_validates_dict_sha_and_profile(
    tmp_path: Path, manifest_overrides: dict[str, Any], match: str
) -> None:
    config, dict_sha = _write_trt_dummy_files(tmp_path)
    params: dict[str, Any] = {"dict_sha256": dict_sha}
    params.update(manifest_overrides)
    runner = _MockTrtRunner(_make_rec_manifest(**params))
    with pytest.raises(TrtEngineError, match=match):
        TrtTextRecognizer(config, runner=cast(TrtExecutionRunner, runner))


def test_trt_engine_error_propagates_without_swallowing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, dict_sha = _write_trt_dummy_files(tmp_path)

    # 场景 1：ensure() 初始化阶段抛 TrtEngineError，read_text() 首次与再次调用均必须重抛
    def fail_load(engine_path: str | Path, *, expected_kind: str, device_id: int = 0) -> Any:
        del engine_path, expected_kind, device_id
        raise TrtEngineError("manifest sha mismatch")

    monkeypatch.setattr("airdrop.perception.trt_runtime.load_trt_runner", fail_load)
    init_engine = OcrEngine(config)
    with pytest.raises(TrtEngineError, match="manifest sha mismatch"):
        init_engine.read_text(np.zeros((100, 100, 3), dtype=np.uint8))
    assert init_engine.failed is True
    with pytest.raises(TrtEngineError, match="manifest sha mismatch"):
        init_engine.read_text(np.zeros((100, 100, 3), dtype=np.uint8))

    # 场景 2：推理阶段抛 TrtShapeOutOfRangeError，穿透 read_text() 与 OpenCvPostProcess.recognize()
    def fail_det(array: np.ndarray) -> np.ndarray:
        del array
        raise TrtShapeOutOfRangeError("det shape exceeded")

    det_runner = _MockTrtRunner(_make_det_manifest(), fn=fail_det)
    rec_runner = _MockTrtRunner(_make_rec_manifest(dict_sha256=dict_sha))
    backend = TrtOcrBackend(
        config,
        text_det=TrtTextDetector(config, runner=cast(TrtExecutionRunner, det_runner)),
        text_rec=TrtTextRecognizer(config, runner=cast(TrtExecutionRunner, rec_runner)),
    )
    ocr_engine = OcrEngine(config)
    ocr_engine._engine = backend

    with pytest.raises(TrtShapeOutOfRangeError, match="det shape exceeded"):
        ocr_engine.read_text(np.full((120, 120, 3), 255, dtype=np.uint8))

    post = OpenCvPostProcess(color="blue", engine=ocr_engine)
    with pytest.raises(TrtShapeOutOfRangeError, match="det shape exceeded"):
        post.recognize(_blue_pentagon_image())


# ----------------------------------------------------------------------
# TrtOcrBackend 端到端推理、批处理、过滤、预热与计时插桩
# ----------------------------------------------------------------------
def test_trt_ocr_backend_end_to_end_and_text_score_filtering(tmp_path: Path) -> None:
    config, dict_sha = _write_trt_dummy_files(tmp_path)
    chars = ["blank", *[str(i) for i in range(10)], " "]

    # 检测热图在两个互不重叠区域置高响应（上方宽框 + 下方窄框）
    def fake_det_fn(pre: np.ndarray) -> np.ndarray:
        _, _, h, w = pre.shape
        prob = np.zeros((1, 1, h, w), dtype=np.float32)
        # 区域 1（上方，宽高比更大）：y in [100, 160], x in [80, 380]
        prob[0, 0, 100:160, 80:380] = 0.95
        # 区域 2（下方，宽高比较小）：y in [420, 480], x in [120, 260]
        prob[0, 0, 420:480, 120:260] = 0.95
        return prob

    # 识别按区域背景色均值区分两框：按顺序（上方宽框红通道高 -> "42"@0.92，下方窄框蓝通道高 -> "07"@0.25）
    def fake_rec_fn(batch: np.ndarray) -> np.ndarray:
        # 归一化后值在 [-1, 1]，比较通道 2（R）与通道 0（B）区分两块裁切图
        specs: list[tuple[str, float]] = []
        for idx in range(batch.shape[0]):
            r_mean = float(batch[idx, 2, :, :48].mean())
            b_mean = float(batch[idx, 0, :, :48].mean())
            if r_mean > b_mean:
                specs.append(("42", 0.92))
            else:
                specs.append(("07", 0.25))
        return _encode_ctc_logits(chars, specs)

    det_runner = _MockTrtRunner(_make_det_manifest(), fn=fake_det_fn)
    rec_runner = _MockTrtRunner(_make_rec_manifest(dict_sha256=dict_sha), fn=fake_rec_fn)

    # 构造 736x736 图像：上方区域涂红，下方区域涂蓝，以便识别 mock 区分二者
    image = np.zeros((736, 736, 3), dtype=np.uint8)
    image[80:180, 60:400] = (10, 10, 240)
    image[400:500, 100:280] = (240, 10, 10)

    # text_score=0.5：保留 "42" (0.92)，过滤 "07" (0.25)
    backend_strict = TrtOcrBackend(
        config,
        text_det=TrtTextDetector(config, runner=cast(TrtExecutionRunner, det_runner)),
        text_rec=TrtTextRecognizer(config, runner=cast(TrtExecutionRunner, rec_runner)),
    )
    strict_res = backend_strict(image)
    assert strict_res.txts == ("42",)
    assert strict_res.scores is not None and len(strict_res.scores) == 1
    assert strict_res.scores[0] == pytest.approx(0.92, abs=1e-4)
    assert strict_res.boxes is not None and strict_res.boxes.shape == (1, 4, 2)

    # text_score=0.1：两条结果全部保留，且按检测框从上到下排序恢复顺序
    loose_cfg = OcrEngineConfig(
        model_dir=str(tmp_path),
        det_model="det.engine",
        rec_model="rec.engine",
        rec_keys="digits_dict.txt",
        text_score=0.1,
        rec_batch_num=2,
    )
    backend_loose = TrtOcrBackend(
        loose_cfg,
        text_det=TrtTextDetector(loose_cfg, runner=cast(TrtExecutionRunner, det_runner)),
        text_rec=TrtTextRecognizer(loose_cfg, runner=cast(TrtExecutionRunner, rec_runner)),
    )
    loose_res = backend_loose(image)
    assert loose_res.txts == ("42", "07")
    assert loose_res.scores is not None
    assert loose_res.scores[0] == pytest.approx(0.92, abs=1e-4)
    assert loose_res.scores[1] == pytest.approx(0.25, abs=1e-4)
    assert loose_res.boxes is not None and loose_res.boxes.shape == (2, 4, 2)
    assert float(loose_res.boxes[0, 0, 1]) < float(loose_res.boxes[1, 0, 1])


def test_trt_recognizer_sorts_batches_normalizes_and_restores_order(tmp_path: Path) -> None:
    config, dict_sha = _write_trt_dummy_files(tmp_path)
    chars = ["blank", *[str(i) for i in range(10)], " "]
    seen_batches: list[tuple[int, ...]] = []

    def fake_rec_fn(batch: np.ndarray) -> np.ndarray:
        seen_batches.append(tuple(int(d) for d in batch.shape))
        # 每张图在通道 0 写入不同常量值（25 / 128 / 230）用以核对顺序还原
        specs: list[tuple[str, float]] = []
        for idx in range(batch.shape[0]):
            val = float(batch[idx, 0, 0, 0])
            if val < -0.5:
                specs.append(("11", 0.91))
            elif val < 0.5:
                specs.append(("22", 0.92))
            else:
                specs.append(("33", 0.93))
        return _encode_ctc_logits(chars, specs)

    rec_runner = _MockTrtRunner(_make_rec_manifest(dict_sha256=dict_sha), fn=fake_rec_fn)
    recognizer = TrtTextRecognizer(config, runner=cast(TrtExecutionRunner, rec_runner))

    # 输入顺序：中宽（"22"）、最窄（"11"）、超宽（"33"）；rec_batch_num=2 应拆成 [2, 1] 两批
    img_mid = np.full((48, 240, 3), 128, dtype=np.uint8)
    img_narrow = np.full((48, 96, 3), 25, dtype=np.uint8)
    img_wide = np.full((48, 480, 3), 230, dtype=np.uint8)

    out = recognizer([img_mid, img_narrow, img_wide])
    assert seen_batches == [(2, 3, 48, 320), (1, 3, 48, 480)]
    assert out.txts == ("22", "11", "33")
    assert out.scores is not None
    assert out.scores == pytest.approx((0.92, 0.91, 0.93), abs=1e-4)

    # 超出 max_w=2048 的极宽图像必须抛出 TrtShapeOutOfRangeError
    ultra_wide = np.full((48, 2400, 3), 200, dtype=np.uint8)
    with pytest.raises(TrtShapeOutOfRangeError):
        recognizer([ultra_wide])


def test_trt_ocr_warmup_and_timing_instrumentation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, dict_sha = _write_trt_dummy_files(tmp_path)
    chars = ["blank", *[str(i) for i in range(10)], " "]

    det_runner = _MockTrtRunner(_make_det_manifest())
    rec_runner = _MockTrtRunner(
        _make_rec_manifest(dict_sha256=dict_sha),
        fn=lambda batch: _encode_ctc_logits(chars, [("00", 0.5)] * batch.shape[0]),
    )
    backend = TrtOcrBackend(
        config,
        text_det=TrtTextDetector(config, runner=cast(TrtExecutionRunner, det_runner)),
        text_rec=TrtTextRecognizer(config, runner=cast(TrtExecutionRunner, rec_runner)),
    )
    ocr_engine = OcrEngine(config)
    ocr_engine._engine = backend

    assert ocr_engine.warmup() is True
    assert det_runner.calls == [((1, 3, 736, 736), "float32")]
    assert rec_runner.calls == [((1, 3, 48, 320), "float32")]

    # 验证 ocr_worker.py 的 timing.instrument_call_class 可无缝包装 TrtTextDetector / TrtTextRecognizer
    emitted: list[str] = []
    monkeypatch.setattr(timing, "emit", lambda event, **kw: emitted.append(event))
    orig_det_call = TrtTextDetector.__call__
    orig_rec_call = TrtTextRecognizer.__call__
    try:
        timing.instrument_call_class(backend.text_det, "ocr_text_det")
        timing.instrument_call_class(backend.text_rec, "ocr_text_rec")
        backend.text_det(np.full((320, 320, 3), 255, dtype=np.uint8))
        backend.recognize_txt([np.full((48, 320, 3), 255, dtype=np.uint8)])
    finally:
        TrtTextDetector.__call__ = orig_det_call  # type: ignore[method-assign]
        TrtTextRecognizer.__call__ = orig_rec_call  # type: ignore[method-assign]
    assert emitted == ["ocr_text_det", "ocr_text_rec"]


# ----------------------------------------------------------------------
# ocr_worker_main 与 PerceptionWorker 的 TrtEngineError 致命传播
# ----------------------------------------------------------------------
def test_ocr_worker_main_emits_fatal_result_on_trt_engine_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    crop_path = tmp_path / "crop.png"
    assert cv2.imwrite(str(crop_path), np.full((64, 64, 3), 120, dtype=np.uint8))

    class FailingPostProcess:
        def __init__(self, **kwargs: Any) -> None:
            del kwargs
            self.color = "blue"

        def warmup(self) -> bool:
            return True

        def recognize(self, crop: np.ndarray) -> Any:
            del crop
            raise TrtShapeOutOfRangeError("rec width 2400 out of profile")

    monkeypatch.setattr("airdrop.perception.ocr_worker.OpenCvPostProcess", FailingPostProcess)

    req_q: queue.Queue[OcrRequest | None] = queue.Queue()
    res_q: queue.Queue[OcrResult] = queue.Queue()
    status_q: queue.Queue[tuple[int, str | None]] = queue.Queue()
    req_q.put(
        OcrRequest(
            request_id=7,
            frame_index=42,
            capture_timestamp=1234.5,
            crop_path=str(crop_path),
        )
    )
    req_q.put(None)

    ocr_worker_main(req_q, res_q, {"color": "blue", "engine": {}}, worker_id=0, status_q=status_q)

    assert status_q.get_nowait() == (0, None)
    result = res_q.get_nowait()
    assert result.fatal is True
    assert result.stage == "trt_error"
    assert result.request_id == 7
    assert result.frame_index == 42
    assert result.number is None
    assert result.side_px == -1.0
    assert result.raw_text == ""
    assert result.confidence == 0.0
    assert result.error is not None
    assert "TrtShapeOutOfRangeError" in result.error
    assert "rec width 2400 out of profile" in result.error


def test_perception_worker_propagates_yolo_trt_engine_error() -> None:
    class BrokenYoloDetector(_FakeDetector):
        def detect(self, *args: Any, **kwargs: Any) -> DetectionBatch:
            self.calls.append(int(kwargs["frame_index"]))
            raise TrtEngineError("YOLO TensorRT execute_async_v3 失败")

    detector = BrokenYoloDetector()
    worker = PerceptionWorker(
        PerceptionConfigLike(mode="cls12"),
        buffer=_buffer_with(3),
        detector=cast(Detector, detector),
    )
    worker.start()
    try:
        assert not worker.drain(timeout=5)
        with pytest.raises(TrtEngineError, match="execute_async_v3"):
            worker.raise_if_failed()
        assert detector.calls == [1]
        assert worker.stats.frames == 0
    finally:
        worker.stop()
    # 停机后再次启动或查询仍保留原始 TrtEngineError（不会变成 FrameSizeMismatchError）
    with pytest.raises(TrtEngineError, match="execute_async_v3"):
        worker.start()


def test_perception_worker_propagates_ocr_fatal_result_without_fake_detection() -> None:
    class FatalOcrPool:
        def __init__(self) -> None:
            self._started = False
            self._next = 0
            self._pending: list[OcrResult] = []
            self.acked: list[int] = []

        def start(self) -> "FatalOcrPool":
            self._started = True
            return self

        def stop(self, timeout: float = 5.0) -> None:
            del timeout
            self._started = False

        @property
        def alive(self) -> int:
            return 1 if self._started else 0

        def check_alive(self) -> int:
            return self.alive

        @property
        def stats(self) -> dict[str, int]:
            return {
                "workers": 1,
                "alive": self.alive,
                "submitted": self._next,
                "dropped": 0,
                "results": len(self.acked),
                "pending": self._next - len(self.acked),
                "dispatch_failures": 0,
            }

        def submit(
            self,
            crop: np.ndarray,
            *,
            frame_index: int,
            capture_timestamp: float,
            payload: Any = None,
            rectified_path: Path | None = None,
        ) -> int:
            del crop, rectified_path
            self._next += 1
            self._pending.append(
                OcrResult(
                    request_id=self._next,
                    frame_index=frame_index,
                    capture_timestamp=capture_timestamp,
                    number=None,
                    side_px=-1.0,
                    raw_text="",
                    confidence=0.0,
                    stage="trt_error",
                    payload=payload,
                    error="TrtEngineError: OCR engine context execution failed",
                    fatal=True,
                )
            )
            return self._next

        def poll(self, timeout: float = 0.0) -> OcrResult | None:
            del timeout
            return self._pending.pop(0) if self._pending else None

        def ack(self, result: OcrResult) -> None:
            self.acked.append(result.request_id)

    pool = FatalOcrPool()
    detector = _FakeDetector()
    worker = PerceptionWorker(
        PerceptionConfigLike(mode="ocr"),
        buffer=_buffer_with(3),
        detector=cast(Detector, detector),
        pool=cast(OcrWorkerPool, pool),
    )
    worker.start()
    try:
        assert not worker.drain(timeout=5)
        with pytest.raises(TrtEngineError, match="OCR engine context execution failed"):
            worker.raise_if_failed()
        assert worker.drain_results() == []
        assert pool.acked == []
        assert detector.calls == [1]
    finally:
        worker.stop()
    with pytest.raises(TrtEngineError, match="OCR engine context execution failed"):
        worker.raise_if_failed()


def test_perception_worker_preserves_frame_size_mismatch_error() -> None:
    """确认引入 _fatal_exception 不改变既有 FrameSizeMismatchError 语义。"""

    class BadSizeDetector(_FakeDetector):
        def detect(self, *args: Any, **kwargs: Any) -> DetectionBatch:
            del args, kwargs
            raise FrameSizeMismatchError("尺寸不符")

    worker = PerceptionWorker(
        PerceptionConfigLike(mode="cls12"),
        buffer=_buffer_with(1),
        detector=cast(Detector, BadSizeDetector()),
    )
    worker.start()
    try:
        assert not worker.drain(timeout=5)
        with pytest.raises(FrameSizeMismatchError, match="尺寸不符"):
            worker.raise_if_failed()
    finally:
        worker.stop()


def test_ocr_worker_pool_uses_inprocess_shared_engine_threads_for_trt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """TensorRT OCR 池应在进程内只加载一次引擎并克隆独立上下文供多线程并发推理。"""
    from dataclasses import asdict

    config, dict_sha = _write_trt_dummy_files(tmp_path)
    chars = ["blank", *[str(i) for i in range(10)], " "]
    load_calls: list[str] = []

    def fake_det_fn(pre: np.ndarray) -> np.ndarray:
        _, _, h, w = pre.shape
        prob = np.zeros((1, 1, h, w), dtype=np.float32)
        prob[0, 0, 120:200, 120:300] = 0.95
        return prob

    def fake_rec_fn(batch: np.ndarray) -> np.ndarray:
        return _encode_ctc_logits(chars, [("56", 0.94)] * batch.shape[0])

    def fake_load_trt_runner(
        engine_path: str | Path, *, expected_kind: str, device_id: int = 0
    ) -> _MockTrtRunner:
        del engine_path, device_id
        load_calls.append(expected_kind)
        if expected_kind == "ocr_det":
            return _MockTrtRunner(_make_det_manifest(), fn=fake_det_fn)
        if expected_kind == "ocr_rec":
            return _MockTrtRunner(_make_rec_manifest(dict_sha256=dict_sha), fn=fake_rec_fn)
        raise AssertionError(f"意外 kind: {expected_kind}")

    monkeypatch.setattr("airdrop.perception.trt_runtime.load_trt_runner", fake_load_trt_runner)

    pool = OcrWorkerPool(
        workers=2,
        queue_size=16,
        config={"color": "blue", "ocr_conf_threshold": 0.6, "engine": asdict(config)},
    )
    pool.set_spool_directory(tmp_path / "spool")
    assert pool.inprocess_trt is True

    pool.start(timeout=10.0)
    try:
        assert pool.started is True
        assert pool.alive == 2
        assert load_calls == ["ocr_det", "ocr_rec"]
        assert pool._processes == []
        assert len(pool._worker_threads) == 2

        crop = _blue_pentagon_image()
        req_ids = [
            pool.submit(crop, frame_index=idx, capture_timestamp=100.0 + idx, payload={"f": idx})
            for idx in range(1, 5)
        ]
        assert req_ids == [1, 2, 3, 4]

        got: list[OcrResult] = []
        for _ in range(4):
            res = pool.poll(timeout=5.0)
            assert res is not None
            pool.ack(res)
            got.append(res)
        assert sorted(r.request_id for r in got) == [1, 2, 3, 4]
        assert all(r.number == 56 for r in got)
        assert all(r.fatal is False for r in got)
        assert pool.stats["pending"] == 0
    finally:
        pool.stop(timeout=5.0)
    assert pool.started is False
    assert pool.alive == 0


def test_ocr_worker_pool_inprocess_trt_rolls_back_on_warmup_failure_and_propagates_fatal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """进程内 TRT 工作池在预热失败时应干净回滚，在推理越界时应回报 fatal 结果。"""
    from dataclasses import asdict

    config, dict_sha = _write_trt_dummy_files(tmp_path)

    # 场景 1：预热抛异常 -> start() 失败并回滚全部工作线程
    def fail_warmup_det(pre: np.ndarray) -> np.ndarray:
        del pre
        raise RuntimeError("warmup boom")

    def fake_load_fail(
        engine_path: str | Path, *, expected_kind: str, device_id: int = 0
    ) -> _MockTrtRunner:
        del engine_path, device_id
        if expected_kind == "ocr_det":
            return _MockTrtRunner(_make_det_manifest(), fn=fail_warmup_det)
        return _MockTrtRunner(_make_rec_manifest(dict_sha256=dict_sha))

    monkeypatch.setattr("airdrop.perception.trt_runtime.load_trt_runner", fake_load_fail)
    bad_pool = OcrWorkerPool(
        workers=2,
        config={"color": "blue", "ocr_conf_threshold": 0.6, "engine": asdict(config)},
    )
    bad_pool.set_spool_directory(tmp_path / "spool_fail")
    with pytest.raises(RuntimeError, match="初始化失败"):
        bad_pool.start(timeout=5.0)
    assert bad_pool.started is False
    assert bad_pool.alive == 0
    bad_pool.stop()

    # 场景 2：预热成功但推理阶段触发 TrtShapeOutOfRangeError -> 回传 fatal=True
    calls = {"n": 0}

    def fail_after_warmup_det(pre: np.ndarray) -> np.ndarray:
        calls["n"] += 1
        if calls["n"] <= 2:
            _, _, h, w = pre.shape
            return np.zeros((1, 1, h, w), dtype=np.float32)
        raise TrtShapeOutOfRangeError("det shape exceeded")

    chars = ["blank", *[str(i) for i in range(10)], " "]

    def fake_load_runtime_err(
        engine_path: str | Path, *, expected_kind: str, device_id: int = 0
    ) -> _MockTrtRunner:
        del engine_path, device_id
        if expected_kind == "ocr_det":
            return _MockTrtRunner(_make_det_manifest(), fn=fail_after_warmup_det)
        return _MockTrtRunner(
            _make_rec_manifest(dict_sha256=dict_sha),
            fn=lambda batch: _encode_ctc_logits(chars, [("00", 0.5)] * batch.shape[0]),
        )

    monkeypatch.setattr("airdrop.perception.trt_runtime.load_trt_runner", fake_load_runtime_err)
    pool = OcrWorkerPool(
        workers=2,
        config={"color": "blue", "ocr_conf_threshold": 0.6, "engine": asdict(config)},
    )
    pool.set_spool_directory(tmp_path / "spool_fatal")
    pool.start(timeout=10.0)
    try:
        pool.submit(_blue_pentagon_image(), frame_index=10, capture_timestamp=1.0)
        res = pool.poll(timeout=5.0)
        assert res is not None
        assert res.fatal is True
        assert res.stage == "trt_error"
        assert res.error is not None and "TrtShapeOutOfRangeError" in res.error
        pool.ack(res)
    finally:
        pool.stop(timeout=5.0)
