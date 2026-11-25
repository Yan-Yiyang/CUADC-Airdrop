"""TensorRT 离线引擎构建配置、原子发布、后端校验与感知覆盖回归测试。"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from airdrop.config import Config, PerceptionConfig
from airdrop.perception.trt_runtime import manifest_path_for, sha256_bytes
from airdrop.record.replay import load_config_snapshot
from airdrop.runtime import mission as mission_runtime
from airdrop.runtime import replay as replay_runtime
from airdrop.runtime.session import LiveComponents
from tools import build_trt_engines as builder


def _sample_manifest(engine_name: str, engine_bytes: bytes) -> dict[str, Any]:
    """构造仅包含文件名的合法最小伴随清单。"""
    return {
        "schema_version": 1,
        "kind": "yolo",
        "engine_file": engine_name,
        "engine_sha256": sha256_bytes(engine_bytes),
        "source": {
            "weight_file": "best2.pt",
            "weight_sha256": "0" * 64,
        },
        "model": {
            "precision": "fp16",
            "strongly_typed": True,
            "yolo": {
                "imgsz": 1280,
                "input_hw": [736, 1280],
                "class_count": 1,
                "names": {"0": "target"},
                "end2end": False,
            },
            "inputs": [{"name": "images", "dtype": "float16", "shape": [1, 3, 736, 1280]}],
            "outputs": [{"name": "output0", "dtype": "float16", "shape": [1, 5, 19320]}],
        },
        "environment": {
            "platform": "win32",
            "compute_capability": "sm89",
            "tensorrt_version": "11.4.0",
        },
        "verification": {"verified": True, "cases": []},
    }


# ----------------------------------------------------------------------
# build_config 与输出路径规范
# ----------------------------------------------------------------------
def test_build_config_defaults_and_output_paths() -> None:
    cfg = builder.build_config()
    assert cfg.target == "all"
    assert cfg.out_dir == Path("models/trt")
    assert cfg.device == "0"
    assert cfg.yolo_model == Path("models/best2.pt")
    assert cfg.imgsz == 1280
    assert cfg.yolo_input_hw == (736, 1280)
    assert cfg.ocr_models_dir == Path("models/ppocr")
    assert cfg.ocr_det_model == "PP-OCRv6_det_medium.pth"
    assert cfg.ocr_rec_model == "PP-OCRv6_rec_medium.pth"
    assert cfg.ocr_rec_keys == "ppocrv6_dict.txt"
    assert cfg.det_max_side_len == 960
    assert cfg.rec_max_batch == 6
    assert cfg.rec_max_width == 1024
    assert cfg.workspace_mb == 1024
    assert cfg.precision == "fp16"
    assert cfg.verify is True
    assert cfg.force is False

    assert builder.yolo_engine_path(cfg) == Path("models/trt/best2_736x1280_fp16.engine")
    assert builder.ocr_det_engine_path(cfg) == Path("models/trt/PP-OCRv6_det_medium_fp16.engine")
    assert builder.ocr_rec_engine_path(cfg) == Path("models/trt/PP-OCRv6_rec_medium_fp16.engine")


def test_build_config_normalizes_paths_and_target_scoped_validation() -> None:
    yolo_only = builder.build_config(
        target="yolo",
        out_dir="out/engines",
        yolo_model="weights/custom.pt",
        yolo_input_hw=[640, 640],
        precision="fp32",
        ocr_det_model="ignored_for_yolo_only",
    )
    assert yolo_only.out_dir == Path("out/engines")
    assert yolo_only.yolo_model == Path("weights/custom.pt")
    assert yolo_only.yolo_input_hw == (640, 640)
    assert builder.yolo_engine_path(yolo_only) == Path("out/engines/custom_640x640_fp32.engine")

    ocr_only = builder.build_config(
        target="ocr",
        ocr_models_dir="weights/ppocr",
        yolo_model="ignored_for_ocr_only.bin",
    )
    assert ocr_only.ocr_models_dir == Path("weights/ppocr")


@pytest.mark.parametrize(
    ("overrides", "pattern"),
    [
        ({"target": "cls"}, "target"),
        ({"precision": "int8"}, "precision"),
        ({"device": "cpu"}, "CUDA"),
        ({"device": "0,1"}, "CUDA"),
        ({"device": "mps"}, "CUDA"),
        ({"out_dir": "   "}, "out_dir"),
        ({"imgsz": 16}, "imgsz"),
        ({"imgsz": True}, "imgsz"),
        ({"yolo_input_hw": (720, 1280)}, "32 倍数"),
        ({"yolo_input_hw": (16, 64)}, "yolo_input_hw"),
        ({"yolo_input_hw": (True, 640)}, "yolo_input_hw"),
        ({"det_max_side_len": 640}, "det_max_side_len"),
        ({"det_max_side_len": 750}, "32 倍数"),
        ({"rec_max_batch": 0}, "rec_max_batch"),
        ({"rec_max_width": 16}, "rec_max_width"),
        ({"workspace_mb": 0}, "workspace_mb"),
        ({"yolo_model": "models/best2.onnx"}, r"\.pt"),
        ({"ocr_det_model": "det.engine"}, r"\.pth"),
        ({"ocr_rec_model": "sub/rec.pth"}, r"\.pth"),
        ({"ocr_rec_keys": "sub/dict.txt"}, "ocr_rec_keys"),
    ],
)
def test_build_config_rejects_invalid_values(overrides: dict[str, Any], pattern: str) -> None:
    with pytest.raises(ValueError, match=pattern):
        builder.build_config(**overrides)


def test_strongly_typed_trt_detection() -> None:
    trt11 = SimpleNamespace(
        BuilderFlag=SimpleNamespace(TF32=1),
        NetworkDefinitionCreationFlag=SimpleNamespace(STRONGLY_TYPED=0),
    )
    trt10 = SimpleNamespace(
        BuilderFlag=SimpleNamespace(FP16=1, TF32=2),
        NetworkDefinitionCreationFlag=SimpleNamespace(STRONGLY_TYPED=0),
    )
    assert builder._is_strongly_typed_trt(trt11) is True
    assert builder._is_strongly_typed_trt(trt10) is False


# ----------------------------------------------------------------------
# atomic_publish_engine 原子发布与异常清理
# ----------------------------------------------------------------------
def test_atomic_publish_engine_order_fsync_and_no_temp_left(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine_bytes = b"serialized-trt-engine-v1"
    final_engine = tmp_path / "best2_736x1280_fp16.engine"
    final_manifest = manifest_path_for(final_engine)
    manifest = _sample_manifest(final_engine.name, engine_bytes)

    replace_calls: list[tuple[Path, Path]] = []
    fsync_count = 0
    real_replace = os.replace
    real_fsync = os.fsync

    def tracked_replace(src: Any, dst: Any) -> None:
        replace_calls.append((Path(src), Path(dst)))
        real_replace(src, dst)

    def tracked_fsync(fd: int) -> None:
        nonlocal fsync_count
        fsync_count += 1
        real_fsync(fd)

    monkeypatch.setattr(os, "replace", tracked_replace)
    monkeypatch.setattr(os, "fsync", tracked_fsync)

    published_engine, published_manifest = builder.atomic_publish_engine(
        engine_bytes, manifest, final_engine, force=False
    )

    assert published_engine == final_engine
    assert published_manifest == final_manifest
    assert fsync_count == 2
    assert len(replace_calls) == 2
    assert replace_calls[0][1] == final_engine
    assert replace_calls[1][1] == final_manifest
    assert replace_calls[0][0].name == f".{final_engine.name}.tmp.{os.getpid()}"
    assert replace_calls[1][0].name == f".{final_manifest.name}.tmp.{os.getpid()}"
    assert final_engine.read_bytes() == engine_bytes
    saved_manifest = json.loads(final_manifest.read_text(encoding="utf-8"))
    assert saved_manifest["engine_file"] == final_engine.name
    assert saved_manifest["source"]["weight_file"] == "best2.pt"
    assert not [p for p in tmp_path.iterdir() if ".tmp." in p.name]


def test_atomic_publish_engine_respects_force_flag(tmp_path: Path) -> None:
    old_bytes = b"old-engine"
    new_bytes = b"new-engine-v2"
    final_engine = tmp_path / "model_fp16.engine"
    final_manifest = manifest_path_for(final_engine)

    builder.atomic_publish_engine(
        old_bytes, _sample_manifest(final_engine.name, old_bytes), final_engine
    )

    with pytest.raises(FileExistsError):
        builder.atomic_publish_engine(
            new_bytes, _sample_manifest(final_engine.name, new_bytes), final_engine, force=False
        )
    assert final_engine.read_bytes() == old_bytes

    # 仅存在残留清单时，force=False 也必须拒绝覆盖。
    final_engine.unlink()
    with pytest.raises(FileExistsError):
        builder.atomic_publish_engine(
            new_bytes, _sample_manifest(final_engine.name, new_bytes), final_engine, force=False
        )

    # force=True 原子替换为新产物。
    builder.atomic_publish_engine(
        new_bytes, _sample_manifest(final_engine.name, new_bytes), final_engine, force=True
    )
    assert final_engine.read_bytes() == new_bytes
    saved = json.loads(final_manifest.read_text(encoding="utf-8"))
    assert saved["engine_sha256"] == sha256_bytes(new_bytes)


def test_atomic_publish_engine_cleans_temp_files_on_replace_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    engine_bytes = b"engine-payload"
    final_engine = tmp_path / "fail_fp16.engine"
    manifest = _sample_manifest(final_engine.name, engine_bytes)

    def failing_replace(_src: Any, _dst: Any) -> None:
        raise OSError("simulated replace failure")

    monkeypatch.setattr(os, "replace", failing_replace)

    with pytest.raises(OSError, match="simulated replace failure"):
        builder.atomic_publish_engine(engine_bytes, manifest, final_engine)

    assert not final_engine.exists()
    assert not manifest_path_for(final_engine).exists()
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "bad_value",
    [
        "/abs/path/best2.pt",
        r"Z:\placeholder\best2.pt",
        r"\\server\share\best2.pt",
    ],
)
def test_atomic_publish_engine_rejects_absolute_paths_in_manifest(
    tmp_path: Path, bad_value: str
) -> None:
    engine_bytes = b"engine-payload"
    final_engine = tmp_path / "best2_736x1280_fp16.engine"
    manifest = _sample_manifest(final_engine.name, engine_bytes)
    manifest["source"]["weight_file"] = bad_value

    with pytest.raises(ValueError, match="绝对路径"):
        builder.atomic_publish_engine(engine_bytes, manifest, final_engine)

    assert not final_engine.exists()
    assert not manifest_path_for(final_engine).exists()


def test_atomic_publish_engine_rejects_non_basename_engine_file(tmp_path: Path) -> None:
    engine_bytes = b"engine-payload"
    final_engine = tmp_path / "best2_736x1280_fp16.engine"
    manifest = _sample_manifest("subdir/best2_736x1280_fp16.engine", engine_bytes)

    with pytest.raises(ValueError, match="engine_file"):
        builder.atomic_publish_engine(engine_bytes, manifest, final_engine)


# ----------------------------------------------------------------------
# Config.validated() 感知后端后缀与 CUDA 约束
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "perception",
    [
        PerceptionConfig(model_path="models/best2.pt", device="cpu"),
        PerceptionConfig(model_path="engines/best2_736x1280_fp16.engine", device="0"),
        PerceptionConfig(model_path="engines/best2_736x1280_fp16.engine", device="cuda"),
        PerceptionConfig(model_path="engines/best2_736x1280_fp16.engine", device="cuda:1"),
        PerceptionConfig(
            det_model="PP-OCRv6_det_medium_fp16.engine",
            rec_model="PP-OCRv6_rec_medium_fp16.engine",
            ocr_use_cuda=True,
        ),
    ],
)
def test_config_validated_accepts_valid_backend_combinations(perception: PerceptionConfig) -> None:
    validated = Config(perception=perception).validated()
    assert validated.perception == perception


@pytest.mark.parametrize(
    ("perception", "pattern"),
    [
        (PerceptionConfig(model_path="models/best2.onnx"), r"\.pt 或 \.engine"),
        (PerceptionConfig(model_path=""), "model_path"),
        (PerceptionConfig(model_path="engines/best2.engine", device="cpu"), "CUDA"),
        (PerceptionConfig(model_path="engines/best2.engine", device="mps"), "CUDA"),
        (PerceptionConfig(model_path="engines/best2.engine", device="0,1"), "CUDA"),
        (
            PerceptionConfig(
                det_model="PP-OCRv6_det_medium_fp16.engine",
                rec_model="PP-OCRv6_rec_medium.pth",
            ),
            r"同为 \.pth 或同为 \.engine",
        ),
        (
            PerceptionConfig(
                det_model="PP-OCRv6_det_medium.pth",
                rec_model="PP-OCRv6_rec_medium_fp16.engine",
            ),
            r"同为 \.pth 或同为 \.engine",
        ),
        (
            PerceptionConfig(
                det_model="det.onnx",
                rec_model="rec.onnx",
            ),
            r"\.pth 或 \.engine",
        ),
        (
            PerceptionConfig(
                det_model="PP-OCRv6_det_medium_fp16.engine",
                rec_model="PP-OCRv6_rec_medium_fp16.engine",
                ocr_use_cuda=False,
            ),
            "ocr_use_cuda",
        ),
    ],
)
def test_config_validated_rejects_invalid_backend_combinations(
    perception: PerceptionConfig, pattern: str
) -> None:
    with pytest.raises(ValueError, match=pattern):
        Config(perception=perception).validated()


# ----------------------------------------------------------------------
# mission / sitl / replay 感知覆盖与快照严格模式兼容
# ----------------------------------------------------------------------
def test_mission_build_config_applies_explicit_perception_overrides() -> None:
    cfg = mission_runtime.build_config(
        model_path="engines/best2_736x1280_fp16.engine",
        ocr_models_dir="engines/ppocr",
        ocr_det_model="PP-OCRv6_det_medium_fp16.engine",
        ocr_rec_model="PP-OCRv6_rec_medium_fp16.engine",
        ocr_workers=1,
    )
    assert cfg.perception.model_path == "engines/best2_736x1280_fp16.engine"
    assert cfg.perception.models_dir == "engines/ppocr"
    assert cfg.perception.det_model == "PP-OCRv6_det_medium_fp16.engine"
    assert cfg.perception.rec_model == "PP-OCRv6_rec_medium_fp16.engine"
    assert cfg.perception.ocr_workers == 1

    with pytest.raises(ValueError, match=r"同为 \.pth 或同为 \.engine"):
        mission_runtime.build_config(ocr_det_model="PP-OCRv6_det_medium.pth")


def test_backend_preset_switches_between_trt_and_pytorch_with_override_precedence() -> None:
    # 默认配置使用 TensorRT 引擎。
    default_cfg = mission_runtime.build_config()
    assert default_cfg.perception.model_path == "models/trt/best2_736x1280_fp16.engine"
    assert default_cfg.perception.det_model == "models/trt/PP-OCRv6_det_medium_fp16.engine"
    assert default_cfg.perception.rec_model == "models/trt/PP-OCRv6_rec_medium_fp16.engine"

    # --backend pytorch 复位为 PyTorch 备用权重，且显式路径参数优先覆盖预设。
    pt_cfg = mission_runtime.build_config(backend="pytorch")
    assert pt_cfg.perception.model_path == "models/best2.pt"
    assert pt_cfg.perception.det_model == "PP-OCRv6_det_medium.pth"
    assert pt_cfg.perception.rec_model == "PP-OCRv6_rec_medium.pth"

    pt_override = mission_runtime.build_config(backend="pytorch", model_path="models/custom.pt")
    assert pt_override.perception.model_path == "models/custom.pt"
    assert pt_override.perception.det_model == "PP-OCRv6_det_medium.pth"
    assert pt_override.perception.rec_model == "PP-OCRv6_rec_medium.pth"

    # 未知后端直接拒绝。
    with pytest.raises(ValueError, match="推理后端只支持"):
        mission_runtime.build_config(backend="onnx")


def test_mission_main_propagates_perception_overrides_to_factory() -> None:
    seen: list[Config] = []

    class DummyThread:
        def connect(self) -> None:
            pass

        def stop(self, timeout: float = 5.0) -> None:
            del timeout

    def factory(_options, config: Config) -> LiveComponents:
        seen.append(config)
        return LiveComponents(thread=DummyThread())

    rc = mission_runtime.main(
        mode="real",
        run_mission=False,
        capture=True,
        recognize=False,
        record=False,
        qgc_video=False,
        max_seconds=0.000001,
        backend="pytorch",
        model_path="engines/best2_736x1280_fp16.engine",
        ocr_models_dir="engines/ppocr",
        ocr_det_model="PP-OCRv6_det_medium_fp16.engine",
        ocr_rec_model="PP-OCRv6_rec_medium_fp16.engine",
        ocr_workers=1,
        components_factory=factory,
    )
    assert rc == 0
    assert len(seen) == 1
    assert seen[0].perception.model_path == "engines/best2_736x1280_fp16.engine"
    assert seen[0].perception.models_dir == "engines/ppocr"
    assert seen[0].perception.det_model == "PP-OCRv6_det_medium_fp16.engine"
    assert seen[0].perception.rec_model == "PP-OCRv6_rec_medium_fp16.engine"
    assert seen[0].perception.ocr_workers == 1


def test_mission_main_forwards_perception_overrides_to_sitl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys
    import types

    captured: dict[str, Any] = {}

    def fake_sitl_main(**kwargs: Any) -> int:
        captured.update(kwargs)
        return 0

    fake_module = types.ModuleType("airdrop.runtime.sitl")
    fake_module.main = fake_sitl_main  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "airdrop.runtime.sitl", fake_module)

    rc = mission_runtime.main(
        mode="sitl",
        run_mission=False,
        capture=True,
        recognize=False,
        record=False,
        qgc_video=False,
        backend="pytorch",
        model_path="engines/best2_736x1280_fp16.engine",
        ocr_models_dir="engines/ppocr",
        ocr_det_model="PP-OCRv6_det_medium_fp16.engine",
        ocr_rec_model="PP-OCRv6_rec_medium_fp16.engine",
        ocr_workers=1,
    )
    assert rc == 0
    assert captured["backend"] == "pytorch"
    assert captured["model_path"] == "engines/best2_736x1280_fp16.engine"
    assert captured["ocr_models_dir"] == "engines/ppocr"
    assert captured["ocr_det_model"] == "PP-OCRv6_det_medium_fp16.engine"
    assert captured["ocr_rec_model"] == "PP-OCRv6_rec_medium_fp16.engine"
    assert captured["ocr_workers"] == 1


def test_replay_build_config_preserves_snapshot_schema_and_applies_perception_overrides(
    tmp_path: Path,
) -> None:
    flight_dir = tmp_path / "flight_snapshot"
    flight_dir.mkdir()
    recorded = Config().replace(
        perception=replace(Config().perception, ocr_workers=3, device="0"),
    )
    (flight_dir / "config_snapshot.json").write_text(
        json.dumps(asdict(recorded), ensure_ascii=False), encoding="utf-8"
    )

    # 严格恢复快照本身不发生字段漂移。
    restored = load_config_snapshot(flight_dir)
    assert restored == recorded

    # 未传入覆盖时保留快照原始感知参数。
    unchanged = replay_runtime.build_config(flight_dir=flight_dir)
    assert unchanged.perception == recorded.perception

    # 显式 --backend pytorch 切换为 PyTorch 备用权重，同时保留快照其它字段。
    as_pytorch = replay_runtime.build_config(flight_dir=flight_dir, backend="pytorch")
    assert as_pytorch.perception.model_path == "models/best2.pt"
    assert as_pytorch.perception.det_model == "PP-OCRv6_det_medium.pth"
    assert as_pytorch.perception.rec_model == "PP-OCRv6_rec_medium.pth"
    assert as_pytorch.perception.ocr_workers == 3
    assert as_pytorch.video == recorded.video

    # 历史 PyTorch 录制快照在不传 --backend 时保持原样，传 --backend trt 时切换为默认引擎。
    pt_flight = tmp_path / "flight_pytorch_snapshot"
    pt_flight.mkdir()
    pt_recorded = Config().replace(
        perception=replace(
            Config().perception,
            model_path="models/best2.pt",
            det_model="PP-OCRv6_det_medium.pth",
            rec_model="PP-OCRv6_rec_medium.pth",
        ),
    )
    (pt_flight / "config_snapshot.json").write_text(
        json.dumps(asdict(pt_recorded), ensure_ascii=False), encoding="utf-8"
    )
    assert replay_runtime.build_config(flight_dir=pt_flight).perception == pt_recorded.perception
    upgraded = replay_runtime.build_config(flight_dir=pt_flight, backend="trt")
    assert upgraded.perception.model_path == "models/trt/best2_736x1280_fp16.engine"
    assert upgraded.perception.det_model == "models/trt/PP-OCRv6_det_medium_fp16.engine"
    assert upgraded.perception.rec_model == "models/trt/PP-OCRv6_rec_medium_fp16.engine"

    # 显式覆盖仅修改指定字段，其余快照配置保持不变。
    overridden = replay_runtime.build_config(
        flight_dir=flight_dir,
        model_path="engines/best2_736x1280_fp16.engine",
        ocr_models_dir="engines/ppocr",
        ocr_det_model="PP-OCRv6_det_medium_fp16.engine",
        ocr_rec_model="PP-OCRv6_rec_medium_fp16.engine",
        ocr_workers=1,
    )
    assert overridden.perception.model_path == "engines/best2_736x1280_fp16.engine"
    assert overridden.perception.models_dir == "engines/ppocr"
    assert overridden.perception.det_model == "PP-OCRv6_det_medium_fp16.engine"
    assert overridden.perception.rec_model == "PP-OCRv6_rec_medium_fp16.engine"
    assert overridden.perception.ocr_workers == 1
    assert overridden.perception.device == "0"
    assert overridden.video == recorded.video

    # 覆盖后仍执行 Config.validated()，拒绝半套 OCR 引擎覆盖。
    with pytest.raises(ValueError, match=r"同为 \.pth 或同为 \.engine"):
        replay_runtime.build_config(
            flight_dir=flight_dir,
            ocr_det_model="PP-OCRv6_det_medium.pth",
        )

    # 历史 PyTorch CPU 快照在覆盖为 TensorRT 时必须被拦截。
    cpu_flight = tmp_path / "flight_cpu"
    cpu_flight.mkdir()
    cpu_recorded = Config().replace(
        perception=replace(
            Config().perception,
            model_path="models/best2.pt",
            det_model="PP-OCRv6_det_medium.pth",
            rec_model="PP-OCRv6_rec_medium.pth",
            device="cpu",
            ocr_use_cuda=False,
        ),
    )
    (cpu_flight / "config_snapshot.json").write_text(
        json.dumps(asdict(cpu_recorded), ensure_ascii=False), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="CUDA"):
        replay_runtime.build_config(
            flight_dir=cpu_flight,
            model_path="engines/best2_736x1280_fp16.engine",
        )
    with pytest.raises(ValueError, match="CUDA"):
        replay_runtime.build_config(flight_dir=cpu_flight, backend="trt")
