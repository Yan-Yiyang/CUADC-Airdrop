"""TensorRT 运行时契约与 Detector TensorRT 后端离线单测（不依赖 GPU 或实装 TensorRT）。"""

from __future__ import annotations

import contextlib
import copy
import json
import sys
import threading
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from airdrop.perception import trt_runtime
from airdrop.perception.detector import (
    Detector,
    DetectorConfig,
    _letterbox_bgr,
    _TrtYoloModel,
)
from airdrop.perception.trt_runtime import (
    MANIFEST_SCHEMA_VERSION,
    TensorSpec,
    TrtEngineError,
    TrtExecutionRunner,
    TrtManifest,
    TrtShapeOutOfRangeError,
    load_trt_runner,
    manifest_path_for,
    parse_cuda_device_id,
    sha256_bytes,
    sha256_file,
    validate_manifest_dict,
    validate_runtime_environment,
)
from airdrop.telemetry.models import TelemetrySnapshot

_VALID_SHA = "a" * 64


# ----------------------------------------------------------------------
# 测试辅助构造器
# ----------------------------------------------------------------------
def _make_manifest_dict(
    *,
    kind: str = "yolo",
    engine_file: str = "best2.engine",
    engine_sha256: str = _VALID_SHA,
    inputs: list[dict[str, Any]] | None = None,
    outputs: list[dict[str, Any]] | None = None,
    yolo_meta: dict[str, Any] | None = None,
    environment: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if inputs is None:
        inputs = [{"name": "images", "dtype": "float16", "shape": [1, 3, 736, 1280]}]
    if outputs is None:
        outputs = [{"name": "output0", "dtype": "float16", "shape": [1, 5, 19320]}]
    model_section: dict[str, Any] = {
        "inputs": inputs,
        "outputs": outputs,
    }
    if kind == "yolo":
        model_section["yolo"] = (
            {
                "input_hw": [736, 1280],
                "names": {"0": "target"},
                "end2end": False,
            }
            if yolo_meta is None
            else yolo_meta
        )
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "kind": kind,
        "engine_file": engine_file,
        "engine_sha256": engine_sha256,
        "source": {"weights": "best2.pt", "sha256": "b" * 64},
        "model": model_section,
        "environment": (
            {
                "platform": "win32",
                "gpu_name": "Reference GPU",
                "compute_capability": "sm89",
                "cuda_version": "13.0",
                "tensorrt_version": "10.9.0",
                "torch_version": "2.13.0",
                "onnx_version": "1.18.0",
            }
            if environment is None
            else environment
        ),
        "verification": {"status": "ok"},
    }


def _make_manifest(
    *,
    kind: str = "yolo",
    input_spec: TensorSpec | None = None,
    output_spec: TensorSpec | None = None,
    yolo_meta: dict[str, Any] | None = None,
    inputs: tuple[TensorSpec, ...] | None = None,
    outputs: tuple[TensorSpec, ...] | None = None,
) -> TrtManifest:
    in_specs = (
        inputs
        if inputs is not None
        else (
            input_spec
            if input_spec is not None
            else TensorSpec(name="images", dtype="float16", shape=(1, 3, 736, 1280)),
        )
    )
    out_specs = (
        outputs
        if outputs is not None
        else (
            output_spec
            if output_spec is not None
            else TensorSpec(name="output0", dtype="float16", shape=(1, 5, 16)),
        )
    )
    model_section: dict[str, Any] = {}
    if kind == "yolo" and yolo_meta is not None:
        model_section["yolo"] = yolo_meta
    elif kind == "yolo":
        model_section["yolo"] = {
            "input_hw": [736, 1280],
            "names": {"0": "target"},
            "end2end": False,
        }
    return TrtManifest(
        schema_version=MANIFEST_SCHEMA_VERSION,
        kind=kind,
        engine_file="best2.engine",
        engine_sha256=_VALID_SHA,
        source={"weights": "best2.pt"},
        model=model_section,
        environment={
            "platform": "win32",
            "tensorrt_version": "10.9.0",
            "compute_capability": "sm89",
        },
        verification={"status": "ok"},
        inputs=in_specs,
        outputs=out_specs,
    )


class _FakeCudaStream:
    def __init__(self, handle: int = 0xBEEF) -> None:
        self.cuda_stream = handle
        self.waited_streams: list[Any] = []
        self.sync_calls = 0

    def wait_stream(self, other: Any) -> None:
        self.waited_streams.append(other)

    def synchronize(self) -> None:
        self.sync_calls += 1


class _FakeTorchCudaModule:
    """用真实 CPU PyTorch 张量替代 CUDA 运行时，以离线验证 TrtExecutionRunner 控制流。"""

    float16 = torch.float16
    float32 = torch.float32
    int32 = torch.int32
    int64 = torch.int64

    def __init__(self) -> None:
        self.created_streams: list[_FakeCudaStream] = []
        self.current = _FakeCudaStream(handle=0x1111)
        self.cuda = self._CudaNamespace(self)

    class _CudaNamespace:
        def __init__(self, owner: _FakeTorchCudaModule) -> None:
            self._owner = owner

        def Stream(self, *, device: Any) -> _FakeCudaStream:
            del device
            stream = _FakeCudaStream(handle=0x2000 + len(self._owner.created_streams))
            self._owner.created_streams.append(stream)
            return stream

        def current_stream(self, device: Any) -> _FakeCudaStream:
            del device
            return self._owner.current

        def stream(self, stream_obj: Any):
            del stream_obj
            return contextlib.nullcontext()

    @staticmethod
    def device(spec: str) -> torch.device:
        del spec
        return torch.device("cpu")

    @staticmethod
    def no_grad():
        return torch.no_grad()

    @staticmethod
    def empty(shape: tuple[int, ...], *, device: Any, dtype: Any) -> torch.Tensor:
        return torch.zeros(shape, device=device, dtype=dtype)

    @staticmethod
    def from_numpy(array: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(array)


class _FakeTrtContext:
    """模拟 TensorRT ExecutionContext 的形状解析与异步执行。"""

    def __init__(
        self,
        *,
        output_shape: tuple[int, ...] = (1, 5, 16),
        set_input_shape_ok: bool = True,
        execute_ok: bool = True,
        dynamic_out_shape_fn: Any = None,
    ) -> None:
        self.output_shape = output_shape
        self.set_input_shape_ok = set_input_shape_ok
        self.execute_ok = execute_ok
        self.dynamic_out_shape_fn = dynamic_out_shape_fn
        self.last_input_shape: tuple[int, ...] | None = None
        self.addresses: dict[str, int] = {}
        self.executed_streams: list[int] = []

    def set_input_shape(self, name: str, shape: tuple[int, ...]) -> bool:
        del name
        self.last_input_shape = shape
        if self.dynamic_out_shape_fn is not None and self.set_input_shape_ok:
            self.output_shape = self.dynamic_out_shape_fn(shape)
        return self.set_input_shape_ok

    def get_tensor_shape(self, name: str) -> tuple[int, ...]:
        del name
        return self.output_shape

    def set_tensor_address(self, name: str, ptr: int) -> None:
        self.addresses[name] = ptr

    def execute_async_v3(self, stream_handle: int) -> bool:
        self.executed_streams.append(stream_handle)
        return self.execute_ok


class _FakeYoloRunner:
    """离线 CPU 端假 TensorRT 执行器，与 _TrtYoloModel 依赖的公开契约完全一致。"""

    def __init__(
        self,
        *,
        manifest: TrtManifest,
        output_tensor: torch.Tensor | None = None,
        run_error: Exception | None = None,
    ) -> None:
        self.manifest = manifest
        self.input_spec = manifest.inputs[0]
        self.output_spec = manifest.outputs[0]
        self.device = torch.device("cpu")
        self.input_torch_dtype = (
            torch.float16 if self.input_spec.dtype == "float16" else torch.float32
        )
        self._output_tensor = output_tensor
        self._run_error = run_error
        self.seen_inputs: list[torch.Tensor] = []

    def run_tensor(self, input_tensor: torch.Tensor) -> torch.Tensor:
        self.seen_inputs.append(input_tensor.clone())
        if self._run_error is not None:
            raise self._run_error
        if self._output_tensor is not None:
            return self._output_tensor
        return torch.zeros(self.output_spec.shape, dtype=self.input_torch_dtype)


def _block_ultralytics(monkeypatch: pytest.MonkeyPatch) -> None:
    """将 ultralytics 在 sys.modules 中置为 None，确保任何导入立即抛出 ModuleNotFoundError。"""
    monkeypatch.setitem(sys.modules, "ultralytics", None)


def _build_raw_yolo_output(
    nc: int,
    rows: list[tuple[float, float, float, float, dict[int, float]]],
    *,
    num_anchors: int | None = None,
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """按 ``(1, 4 + nc, N)`` 布局构造 YOLO 原始输出张量。"""
    n = max(len(rows), num_anchors or len(rows), 1)
    out = torch.zeros((1, 4 + nc, n), dtype=torch.float32)
    for idx, (cx, cy, w, h, cls_map) in enumerate(rows):
        out[0, 0, idx] = cx
        out[0, 1, idx] = cy
        out[0, 2, idx] = w
        out[0, 3, idx] = h
        for cls_id, score in cls_map.items():
            out[0, 4 + cls_id, idx] = score
    return out.to(dtype=dtype)


# ======================================================================
# 1. trt_runtime 离线单测
# ======================================================================
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("0", 0),
        ("1", 1),
        (" 2 ", 2),
        ("cuda", 0),
        ("CUDA", 0),
        ("cuda:0", 0),
        ("cuda:3", 3),
        (" CUDA:1 ", 1),
    ],
)
def test_parse_cuda_device_id_valid(raw: str, expected: int) -> None:
    assert parse_cuda_device_id(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "   ",
        "cpu",
        "mps",
        "0,1",
        "cuda:0,1",
        "cuda:-1",
        "cuda:abc",
        "cuda:",
        "-1",
        "gpu:0",
    ],
)
def test_parse_cuda_device_id_rejects_invalid_inputs(raw: str) -> None:
    with pytest.raises(ValueError, match="CUDA"):
        parse_cuda_device_id(raw)


def test_manifest_path_and_sha256_helpers(tmp_path: Path) -> None:
    engine = tmp_path / "best2.engine"
    payload = b"trt-binary-payload-bytes"
    engine.write_bytes(payload)
    assert manifest_path_for(engine) == tmp_path / "best2.engine.manifest.json"
    assert sha256_file(engine) == sha256_bytes(payload)


def test_validate_manifest_dict_accepts_static_and_dynamic() -> None:
    static_raw = _make_manifest_dict(engine_sha256=_VALID_SHA.upper())
    static_manifest = validate_manifest_dict(
        static_raw,
        expected_kind="yolo",
        expected_engine_name="best2.engine",
        expected_engine_sha256=_VALID_SHA,
    )
    assert static_manifest.kind == "yolo"
    assert static_manifest.engine_sha256 == _VALID_SHA
    assert not static_manifest.inputs[0].is_dynamic
    assert static_manifest.inputs[0].shape == (1, 3, 736, 1280)

    dynamic_raw = _make_manifest_dict(
        kind="ocr_det",
        engine_file="det.engine",
        inputs=[
            {
                "name": "x",
                "dtype": "float32",
                "shape": [1, 3, -1, -1],
                "min_shape": [1, 3, 64, 64],
                "opt_shape": [1, 3, 320, 320],
                "max_shape": [1, 3, 960, 960],
            }
        ],
        outputs=[{"name": "out", "dtype": "float32", "shape": [1, 1, -1, -1]}],
    )
    dyn_manifest = validate_manifest_dict(
        dynamic_raw,
        expected_kind="ocr_det",
        expected_engine_name="det.engine",
        expected_engine_sha256=_VALID_SHA,
    )
    assert dyn_manifest.inputs[0].is_dynamic
    assert dyn_manifest.inputs[0].min_shape == (1, 3, 64, 64)
    assert dyn_manifest.inputs[0].max_shape == (1, 3, 960, 960)


def test_validate_manifest_dict_rejects_unknown_expected_kind() -> None:
    with pytest.raises(ValueError, match="未知的 TensorRT 清单类别"):
        validate_manifest_dict(
            _make_manifest_dict(),
            expected_kind="unknown",
            expected_engine_name="best2.engine",
            expected_engine_sha256=_VALID_SHA,
        )


@pytest.mark.parametrize(
    ("mutator", "pattern"),
    [
        pytest.param(lambda _d: ["not-a-dict"], "顶层必须是 JSON 对象", id="non-mapping"),
        pytest.param(lambda d: {**d, "schema_version": 2}, "版本不匹配", id="wrong-schema-int"),
        pytest.param(lambda d: {**d, "schema_version": True}, "版本不匹配", id="bool-schema"),
        pytest.param(lambda d: {**d, "kind": "ocr_rec"}, "类别不匹配", id="wrong-kind"),
        pytest.param(
            lambda d: {**d, "engine_file": "other.engine"}, "引擎文件名不匹配", id="wrong-file"
        ),
        pytest.param(lambda d: {**d, "engine_sha256": "0" * 64}, "SHA256", id="wrong-sha256"),
        pytest.param(lambda d: {**d, "source": None}, "缺少有效字段 source", id="missing-source"),
        pytest.param(lambda d: {**d, "model": "bad"}, "缺少有效字段 model", id="bad-model"),
        pytest.param(
            lambda d: {**d, "model": {**d["model"], "inputs": []}},
            "model.inputs",
            id="empty-inputs",
        ),
        pytest.param(
            lambda d: {**d, "model": {**d["model"], "outputs": []}},
            "model.outputs",
            id="empty-outputs",
        ),
        pytest.param(
            lambda d: {
                **d,
                "model": {
                    **d["model"],
                    "inputs": [{"name": "x", "dtype": "bfloat16", "shape": [1, 3, 64, 64]}],
                },
            },
            "dtype 不受支持",
            id="bad-dtype",
        ),
        pytest.param(
            lambda d: {
                **d,
                "model": {
                    **d["model"],
                    "inputs": [{"name": "x", "dtype": "float16", "shape": [1, 3, -1, -1]}],
                },
            },
            "min_shape",
            id="dynamic-missing-profile",
        ),
        pytest.param(
            lambda d: {
                **d,
                "model": {
                    **d["model"],
                    "inputs": [
                        {
                            "name": "x",
                            "dtype": "float16",
                            "shape": [1, 3, -1, -1],
                            "min_shape": [1, 3, 64],
                            "opt_shape": [1, 3, 320, 320],
                            "max_shape": [1, 3, 960, 960],
                        }
                    ],
                },
            },
            "秩与 shape 不一致",
            id="dynamic-rank-mismatch",
        ),
        pytest.param(
            lambda d: {
                **d,
                "model": {
                    **d["model"],
                    "inputs": [
                        {
                            "name": "x",
                            "dtype": "float16",
                            "shape": [1, 3, -1, -1],
                            "min_shape": [1, 3, 512, 512],
                            "opt_shape": [1, 3, 320, 320],
                            "max_shape": [1, 3, 960, 960],
                        }
                    ],
                },
            },
            "动态 profile 边界无效",
            id="dynamic-min-gt-opt",
        ),
        pytest.param(
            lambda d: {
                **d,
                "model": {
                    **d["model"],
                    "inputs": [
                        {
                            "name": "x",
                            "dtype": "float16",
                            "shape": [1, 3, -1, -1],
                            "min_shape": [1, 3, 64, 64],
                            "opt_shape": [1, 3, 960, 960],
                            "max_shape": [1, 3, 640, 640],
                        }
                    ],
                },
            },
            "动态 profile 边界无效",
            id="dynamic-opt-gt-max",
        ),
        pytest.param(
            lambda d: {
                **d,
                "model": {
                    **d["model"],
                    "inputs": [
                        {
                            "name": "x",
                            "dtype": "float16",
                            "shape": [1, 3, -1, -1],
                            "min_shape": [1, 3, 0, 64],
                            "opt_shape": [1, 3, 320, 320],
                            "max_shape": [1, 3, 960, 960],
                        }
                    ],
                },
            },
            "无效维度值",
            id="dynamic-nonpositive-bound",
        ),
    ],
)
def test_validate_manifest_dict_rejects_invalid_fields(mutator: Any, pattern: str) -> None:
    base = _make_manifest_dict()
    mutated = mutator(copy.deepcopy(base))
    with pytest.raises(TrtEngineError, match=pattern):
        validate_manifest_dict(
            mutated,
            expected_kind="yolo",
            expected_engine_name="best2.engine",
            expected_engine_sha256=_VALID_SHA,
        )


def test_load_trt_runner_file_manifest_json_and_sha_checks(tmp_path: Path) -> None:
    missing = tmp_path / "missing.engine"
    with pytest.raises(FileNotFoundError, match="引擎文件不存在"):
        load_trt_runner(missing, expected_kind="yolo")

    engine = tmp_path / "best2.engine"
    engine.write_bytes(b"fake-engine-bytes")
    with pytest.raises(TrtEngineError, match="伴随清单缺失"):
        load_trt_runner(engine, expected_kind="yolo")

    mpath = manifest_path_for(engine)
    mpath.write_text("{not valid json", encoding="utf-8")
    with pytest.raises(TrtEngineError, match="无法读取或解析"):
        load_trt_runner(engine, expected_kind="yolo")

    bad_sha_manifest = _make_manifest_dict(engine_file=engine.name, engine_sha256="f" * 64)
    mpath.write_text(json.dumps(bad_sha_manifest), encoding="utf-8")
    with pytest.raises(TrtEngineError, match="SHA256"):
        load_trt_runner(engine, expected_kind="yolo")


@pytest.mark.parametrize(
    ("override_key", "override_val", "pattern"),
    [
        pytest.param("platform", "linux", "操作系统平台", id="platform-mismatch"),
        pytest.param("tensorrt_version", "10.8.0", "TensorRT 版本", id="trt-mismatch"),
        pytest.param("compute_capability", "sm86", "GPU 计算能力", id="sm-mismatch"),
        pytest.param("compute_capability", "", "缺少环境字段", id="missing-env-field"),
    ],
)
def test_validate_runtime_environment_rejects_mismatch(
    override_key: str, override_val: str, pattern: str
) -> None:
    manifest = _make_manifest()
    env = dict(manifest.environment)
    env[override_key] = override_val
    bad_manifest = TrtManifest(
        schema_version=manifest.schema_version,
        kind=manifest.kind,
        engine_file=manifest.engine_file,
        engine_sha256=manifest.engine_sha256,
        source=manifest.source,
        model=manifest.model,
        environment=env,
        verification=manifest.verification,
        inputs=manifest.inputs,
        outputs=manifest.outputs,
    )
    current_env = {
        "platform": "win32",
        "tensorrt_version": "10.9.0",
        "compute_capability": "sm89",
    }
    validate_runtime_environment(manifest, current_env)
    with pytest.raises(TrtEngineError, match=pattern):
        validate_runtime_environment(bad_manifest, current_env)


def test_trt_runner_rejects_multi_io_manifest() -> None:
    manifest = _make_manifest(
        inputs=(
            TensorSpec("a", "float16", (1, 3, 64, 64)),
            TensorSpec("b", "float16", (1, 3, 64, 64)),
        )
    )
    with pytest.raises(TrtEngineError, match="单输入单输出"):
        TrtExecutionRunner(
            manifest=manifest,
            runtime=object(),
            engine=object(),
            context=_FakeTrtContext(),
            device_id=0,
            torch_module=_FakeTorchCudaModule(),
        )


def test_trt_runner_clone_context_shares_engine_with_independent_context_and_stream() -> None:
    manifest = _make_manifest(
        kind="ocr_det",
        input_spec=TensorSpec(
            name="x",
            dtype="float32",
            shape=(1, 3, -1, -1),
            min_shape=(1, 3, 32, 32),
            opt_shape=(1, 3, 736, 736),
            max_shape=(1, 3, 960, 960),
        ),
        output_spec=TensorSpec(name="y", dtype="float32", shape=(1, 1, -1, -1)),
    )
    fake_torch = _FakeTorchCudaModule()
    runtime = object()
    created_contexts: list[_FakeTrtContext] = []

    class _FakeEngine:
        def __init__(self, *, fail_clone: bool = False) -> None:
            self.fail_clone = fail_clone

        def create_execution_context(self) -> _FakeTrtContext | None:
            if self.fail_clone:
                return None
            ctx = _FakeTrtContext(output_shape=(1, 1, 64, 64))
            created_contexts.append(ctx)
            return ctx

    engine = _FakeEngine()
    base = TrtExecutionRunner(
        manifest=manifest,
        runtime=runtime,
        engine=engine,
        context=_FakeTrtContext(output_shape=(1, 1, 64, 64)),
        device_id=0,
        torch_module=fake_torch,
    )
    cloned = base.clone_context()

    assert cloned.manifest is base.manifest
    assert cloned._runtime is base._runtime
    assert cloned._engine is base._engine
    assert cloned._context is created_contexts[0]
    assert cloned._context is not base._context
    assert cloned._stream is not base._stream
    assert cloned.device_id == base.device_id == 0
    assert len(fake_torch.created_streams) == 2

    engine.fail_clone = True
    with pytest.raises(TrtEngineError, match="独立 ExecutionContext"):
        base.clone_context()


def test_trt_runner_static_shape_out_of_range() -> None:
    manifest = _make_manifest(
        input_spec=TensorSpec("images", "float16", (1, 3, 736, 1280)),
        output_spec=TensorSpec("output0", "float16", (1, 5, 16)),
    )
    runner = TrtExecutionRunner(
        manifest=manifest,
        runtime=object(),
        engine=object(),
        context=_FakeTrtContext(output_shape=(1, 5, 16)),
        device_id=0,
        torch_module=_FakeTorchCudaModule(),
    )
    with pytest.raises(TrtShapeOutOfRangeError, match="维度数不匹配"):
        runner.run_tensor(torch.zeros((3, 736, 1280), dtype=torch.float16))
    with pytest.raises(TrtShapeOutOfRangeError, match="静态输入"):
        runner.run_tensor(torch.zeros((1, 3, 640, 640), dtype=torch.float16))


def test_trt_runner_dynamic_shape_bounds_and_buffer_caching() -> None:
    manifest = _make_manifest(
        kind="ocr_det",
        input_spec=TensorSpec(
            name="x",
            dtype="float32",
            shape=(1, 3, -1, -1),
            min_shape=(1, 3, 64, 64),
            opt_shape=(1, 3, 320, 320),
            max_shape=(1, 3, 960, 960),
        ),
        output_spec=TensorSpec(name="out", dtype="float32", shape=(1, 1, -1, -1)),
    )
    fake_torch = _FakeTorchCudaModule()
    ctx = _FakeTrtContext(dynamic_out_shape_fn=lambda shape: (1, 1, shape[2], shape[3]))
    runner = TrtExecutionRunner(
        manifest=manifest,
        runtime=object(),
        engine=object(),
        context=ctx,
        device_id=0,
        torch_module=fake_torch,
    )

    out1 = runner.run_tensor(torch.zeros((1, 3, 128, 256), dtype=torch.float32))
    out2 = runner.run_tensor(torch.zeros((1, 3, 128, 256), dtype=torch.float32))
    assert out1.shape == (1, 1, 128, 256)
    assert out2 is out1, "相同输出形状应复用已分配的输出缓冲"
    assert ctx.executed_streams == [fake_torch.created_streams[0].cuda_stream] * 2

    out3 = runner.run_tensor(torch.zeros((1, 3, 256, 256), dtype=torch.float32))
    assert out3.shape == (1, 1, 256, 256)
    assert out3 is not out1

    with pytest.raises(TrtShapeOutOfRangeError, match="超出引擎 profile 范围"):
        runner.run_tensor(torch.zeros((1, 3, 32, 64), dtype=torch.float32))
    with pytest.raises(TrtShapeOutOfRangeError, match="超出引擎 profile 范围"):
        runner.run_tensor(torch.zeros((1, 3, 1024, 64), dtype=torch.float32))
    with pytest.raises(TrtShapeOutOfRangeError, match="固定为 3"):
        runner.run_tensor(torch.zeros((1, 1, 128, 128), dtype=torch.float32))

    ctx.set_input_shape_ok = False
    with pytest.raises(TrtShapeOutOfRangeError, match="拒绝设置输入"):
        runner.run_tensor(torch.zeros((1, 3, 128, 128), dtype=torch.float32))

    ctx.set_input_shape_ok = True
    ctx.dynamic_out_shape_fn = lambda _shape: (1, 1, -1, 128)
    with pytest.raises(TrtEngineError, match="未能解析输出"):
        runner.run_tensor(torch.zeros((1, 3, 128, 128), dtype=torch.float32))


def test_trt_runner_execute_async_failure_and_run_numpy() -> None:
    manifest = _make_manifest(
        input_spec=TensorSpec("images", "float16", (1, 3, 32, 32)),
        output_spec=TensorSpec("output0", "float16", (1, 5, 4)),
    )
    fake_torch = _FakeTorchCudaModule()
    ctx = _FakeTrtContext(output_shape=(1, 5, 4), execute_ok=True)
    runner = TrtExecutionRunner(
        manifest=manifest,
        runtime=object(),
        engine=object(),
        context=ctx,
        device_id=0,
        torch_module=fake_torch,
    )

    result_np = runner.run_numpy(np.zeros((1, 3, 32, 32), dtype=np.float32))
    assert result_np.dtype == np.float32
    assert result_np.shape == (1, 5, 4)
    assert fake_torch.created_streams[0].sync_calls == 1

    ctx.execute_ok = False
    with pytest.raises(TrtEngineError, match="TensorRT 执行失败"):
        runner.run_tensor(torch.zeros((1, 3, 32, 32), dtype=torch.float16))
    with pytest.raises(TrtEngineError, match="TensorRT 执行失败"):
        runner.run_numpy(np.zeros((1, 3, 32, 32), dtype=np.float32))


# ======================================================================
# 2. Detector TensorRT 加载、校验与错误边界单测
# ======================================================================
def test_detector_engine_load_does_not_import_ultralytics(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _block_ultralytics(monkeypatch)
    engine_path = tmp_path / "best2.engine"
    engine_path.write_bytes(b"engine")

    manifest = _make_manifest()
    fake_runner = _FakeYoloRunner(manifest=manifest)
    load_calls: list[tuple[Path, str, int]] = []

    def fake_load_trt_runner(
        path: str | Path, *, expected_kind: str, device_id: int = 0
    ) -> _FakeYoloRunner:
        load_calls.append((Path(path), expected_kind, device_id))
        return fake_runner

    monkeypatch.setattr(trt_runtime, "load_trt_runner", fake_load_trt_runner)

    detector = Detector(DetectorConfig(model_path=str(engine_path), device="cuda:1", imgsz=1280))
    assert detector.load() is detector
    assert detector.loaded
    assert detector.names == {0: "target"}
    assert detector.class_count == 1
    assert isinstance(detector._model, _TrtYoloModel)
    assert load_calls == [(engine_path, "yolo", 1)]

    # 幂等：二次加载不重复调用底层加载器
    assert detector.load() is detector
    assert len(load_calls) == 1

    # 预热也不触发 ultralytics 导入
    assert detector.warmup(1280, 720)
    assert len(fake_runner.seen_inputs) == 1
    assert fake_runner.seen_inputs[0].shape == (1, 3, 736, 1280)


@pytest.mark.parametrize(
    ("manifest", "imgsz", "pattern"),
    [
        pytest.param(
            _make_manifest(
                input_spec=TensorSpec("images", "float16", (1, 3, 640, 640)),
                yolo_meta={"input_hw": [640, 640], "names": {"0": "target"}, "end2end": False},
            ),
            1280,
            "imgsz=1280",
            id="imgsz-mismatch",
        ),
        pytest.param(
            _make_manifest(
                input_spec=TensorSpec("images", "float16", (1, 3, 640, 1280)),
                yolo_meta={"input_hw": [736, 1280], "names": {"0": "target"}, "end2end": False},
            ),
            1280,
            "输入张量形状",
            id="input-spec-shape-mismatch",
        ),
        pytest.param(
            _make_manifest(
                output_spec=TensorSpec("output0", "float16", (1, 16, 16)),
                yolo_meta={"input_hw": [736, 1280], "names": {"0": "target"}, "end2end": False},
            ),
            1280,
            "输出类别数",
            id="class-count-mismatch",
        ),
        pytest.param(
            _make_manifest(
                yolo_meta={"input_hw": [736, 1280], "names": {"0": "target"}, "end2end": True},
            ),
            1280,
            "end2end=True",
            id="end2end-rejected",
        ),
        pytest.param(
            _make_manifest(
                yolo_meta={"input_hw": [736, 1280], "names": {"0": "target"}, "end2end": "no"},
            ),
            1280,
            "end2end 必须为布尔值",
            id="end2end-non-bool",
        ),
        pytest.param(
            _make_manifest(yolo_meta={"input_hw": [736, 1280], "names": {}, "end2end": False}),
            1280,
            "非空类别映射",
            id="empty-names",
        ),
        pytest.param(
            _make_manifest(
                yolo_meta={"input_hw": [736, 1280], "names": {"bad": "target"}, "end2end": False}
            ),
            1280,
            "类别索引无效",
            id="bad-class-index",
        ),
        pytest.param(
            _make_manifest(
                yolo_meta={"input_hw": [736, 1280], "names": {0: 123}, "end2end": False}
            ),
            1280,
            "必须为字符串",
            id="non-str-class-name",
        ),
        pytest.param(
            _make_manifest(
                yolo_meta={"input_hw": [True, 1280], "names": {"0": "target"}, "end2end": False}
            ),
            1280,
            "input_hw 必须为正整数",
            id="bool-input-hw",
        ),
        pytest.param(
            _make_manifest(
                output_spec=TensorSpec("output0", "float16", (1, 5)),
                yolo_meta={"input_hw": [736, 1280], "names": {"0": "target"}, "end2end": False},
            ),
            1280,
            "输出张量形状无效",
            id="bad-output-rank",
        ),
    ],
)
def test_detector_engine_rejects_invalid_yolo_manifest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    manifest: TrtManifest,
    imgsz: int,
    pattern: str,
) -> None:
    _block_ultralytics(monkeypatch)
    engine_path = tmp_path / "bad.engine"
    engine_path.write_bytes(b"engine")
    monkeypatch.setattr(
        trt_runtime,
        "load_trt_runner",
        lambda *_a, **_kw: _FakeYoloRunner(manifest=manifest),
    )
    detector = Detector(DetectorConfig(model_path=str(engine_path), device="0", imgsz=imgsz))
    with pytest.raises(TrtEngineError, match=pattern):
        detector.load()
    assert not detector.loaded
    assert detector.names == {}
    assert detector.class_count == 0


def test_detector_engine_load_failure_is_retryable_and_thread_safe(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _block_ultralytics(monkeypatch)
    engine_path = tmp_path / "retry.engine"
    engine_path.write_bytes(b"engine")

    bad_manifest = _make_manifest(
        output_spec=TensorSpec("output0", "float16", (1, 6, 16)),
        yolo_meta={"input_hw": [736, 1280], "names": {"0": "target"}, "end2end": False},
    )
    good_manifest = _make_manifest()
    calls = 0

    def flaky_loader(*_args: Any, **_kwargs: Any) -> _FakeYoloRunner:
        nonlocal calls
        calls += 1
        return _FakeYoloRunner(manifest=bad_manifest if calls == 1 else good_manifest)

    monkeypatch.setattr(trt_runtime, "load_trt_runner", flaky_loader)
    detector = Detector(DetectorConfig(model_path=str(engine_path), device="0", imgsz=1280))

    with pytest.raises(TrtEngineError, match="输出类别数"):
        detector.load()
    assert not detector.loaded
    assert detector.names == {}

    # 第二次重试成功并串行发布
    results: list[Detector] = []
    threads = [threading.Thread(target=lambda: results.append(detector.load())) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=2)
    assert results == [detector, detector, detector]
    assert calls == 2
    assert detector.loaded
    assert detector.names == {0: "target"}


@pytest.mark.parametrize("filename", ["model.onnx", "weights.bin", "plain_no_suffix"])
def test_detector_rejects_unknown_suffix(tmp_path: Path, filename: str) -> None:
    path = tmp_path / filename
    path.write_bytes(b"data")
    detector = Detector(DetectorConfig(model_path=str(path), device="0"))
    with pytest.raises(ValueError, match="不支持的 YOLO 模型后缀"):
        detector.load()


def test_detector_engine_missing_file_and_invalid_device(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _block_ultralytics(monkeypatch)
    assert DetectorConfig().model_path == "models/trt/best2_736x1280_fp16.engine"

    missing = tmp_path / "absent.engine"
    with pytest.raises(FileNotFoundError, match=r"build-trt --target all") as exc_info:
        Detector(DetectorConfig(model_path=str(missing), device="0")).load()
    assert "权重不存在" in str(exc_info.value)
    assert "--backend pytorch" in str(exc_info.value)

    existing = tmp_path / "present.engine"
    existing.write_bytes(b"engine")
    with pytest.raises(ValueError, match="CUDA"):
        Detector(DetectorConfig(model_path=str(existing), device="cpu")).load()


def test_trt_runtime_failure_messages_guide_build_trt_and_pytorch_fallback(tmp_path: Path) -> None:
    missing_engine = tmp_path / "missing.engine"
    with pytest.raises(FileNotFoundError, match=r"build-trt --target all") as err_missing:
        trt_runtime.load_trt_runner(missing_engine, expected_kind="yolo")
    assert "--backend pytorch" in str(err_missing.value)

    no_manifest_engine = tmp_path / "no_manifest.engine"
    no_manifest_engine.write_bytes(b"engine-data")
    with pytest.raises(TrtEngineError, match=r"build-trt --target all") as err_manifest:
        trt_runtime.load_trt_runner(no_manifest_engine, expected_kind="yolo")
    assert "--backend pytorch" in str(err_manifest.value)

    bad_sha_manifest = _make_manifest_dict(engine_file="no_manifest.engine", engine_sha256="0" * 64)
    with pytest.raises(TrtEngineError, match=r"build-trt --target all") as err_sha:
        trt_runtime.validate_manifest_dict(
            bad_sha_manifest,
            expected_kind="yolo",
            expected_engine_name="no_manifest.engine",
            expected_engine_sha256="1" * 64,
        )
    assert "--backend pytorch" in str(err_sha.value)

    manifest = _make_manifest()
    with pytest.raises(TrtEngineError, match=r"build-trt --target all") as err_env:
        trt_runtime.validate_runtime_environment(
            manifest,
            {
                "platform": manifest.environment["platform"],
                "tensorrt_version": "0.0.0",
                "compute_capability": manifest.environment["compute_capability"],
            },
        )
    assert "--backend pytorch" in str(err_env.value)


def test_detector_trt_error_propagates_from_detect_but_warmup_returns_false(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _block_ultralytics(monkeypatch)
    engine_path = tmp_path / "fail.engine"
    engine_path.write_bytes(b"engine")

    # 加载阶段 TrtEngineError：detect 透传，warmup 返回 False
    monkeypatch.setattr(
        trt_runtime,
        "load_trt_runner",
        lambda *_a, **_kw: (_ for _ in ()).throw(TrtEngineError("manifest corrupted")),
    )
    det_load_fail = Detector(DetectorConfig(model_path=str(engine_path), device="0"))
    assert det_load_fail.warmup(1280, 720) is False
    with pytest.raises(TrtEngineError, match="manifest corrupted"):
        det_load_fail.detect(np.zeros((720, 1280, 3), dtype=np.uint8))

    # 推理阶段 TrtEngineError：detect 透传，warmup 返回 False
    runner_exec_fail = _FakeYoloRunner(
        manifest=_make_manifest(),
        run_error=TrtEngineError("execute_async_v3 failed"),
    )
    monkeypatch.setattr(trt_runtime, "load_trt_runner", lambda *_a, **_kw: runner_exec_fail)
    det_exec_fail = Detector(DetectorConfig(model_path=str(engine_path), device="0"))
    assert det_exec_fail.warmup(1280, 720) is False
    with pytest.raises(TrtEngineError, match="execute_async_v3 failed"):
        det_exec_fail.detect(np.zeros((720, 1280, 3), dtype=np.uint8))


# ======================================================================
# 3. Letterbox、预处理与坐标反算单测
# ======================================================================
def test_letterbox_fast_path_and_arbitrary_aspect_ratios() -> None:
    # 720x1280 -> 736x1280 快路径
    src_hd = np.zeros((720, 1280, 3), dtype=np.uint8)
    src_hd[:, :, 0] = 10
    src_hd[:, :, 1] = 20
    src_hd[:, :, 2] = 30
    lb_hd, gain, pad_x, pad_y = _letterbox_bgr(src_hd, (736, 1280))
    assert lb_hd.shape == (736, 1280, 3)
    assert (gain, pad_x, pad_y) == (1.0, 0.0, 8.0)
    assert np.all(lb_hd[:8] == 114)
    assert np.all(lb_hd[728:] == 114)
    assert np.array_equal(lb_hd[8:728], src_hd)

    # 竖屏 1000x500 -> 736x1280：高受限，左右补灰边
    portrait = np.full((1000, 500, 3), 200, dtype=np.uint8)
    lb_p, gain_p, pad_xp, pad_yp = _letterbox_bgr(portrait, (736, 1280))
    assert lb_p.shape == (736, 1280, 3)
    assert gain_p == pytest.approx(736.0 / 1000.0)
    assert pad_xp == pytest.approx((1280 - 368) / 2.0)
    assert pad_yp == 0.0
    assert np.all(lb_p[:, :456] == 114)
    assert np.all(lb_p[:, 1280 - 456 :] == 114)
    assert np.all(lb_p[:, 456 : 1280 - 456] == 200)

    # 超宽 400x1600 -> 736x1280：宽受限，上下补灰边
    wide = np.full((400, 1600, 3), 50, dtype=np.uint8)
    lb_w, gain_w, pad_xw, pad_yw = _letterbox_bgr(wide, (736, 1280))
    assert lb_w.shape == (736, 1280, 3)
    assert gain_w == pytest.approx(0.8)
    assert pad_xw == 0.0
    assert pad_yw == pytest.approx((736 - 320) / 2.0)
    assert np.all(lb_w[:208] == 114)
    assert np.all(lb_w[528:] == 114)

    # 奇数差值 601x1280 -> 736x1280：top=67, bottom=68, pad_y=67.5
    odd = np.full((601, 1280, 3), 88, dtype=np.uint8)
    lb_odd, gain_odd, pad_xodd, pad_yodd = _letterbox_bgr(odd, (736, 1280))
    assert lb_odd.shape == (736, 1280, 3)
    assert (gain_odd, pad_xodd, pad_yodd) == (1.0, 0.0, 67.5)
    assert np.all(lb_odd[:67] == 114)
    assert np.all(lb_odd[67:668] == 88)
    assert np.all(lb_odd[668:] == 114)

    # 尺寸精确匹配：零补边
    exact = np.full((736, 1280, 3), 42, dtype=np.uint8)
    lb_exact, gain_e, pad_xe, pad_ye = _letterbox_bgr(exact, (736, 1280))
    assert (gain_e, pad_xe, pad_ye) == (1.0, 0.0, 0.0)
    assert np.array_equal(lb_exact, exact)

    with pytest.raises(ValueError, match="3 通道"):
        _letterbox_bgr(np.zeros((720, 1280), dtype=np.uint8), (736, 1280))


def test_trt_yolo_preprocess_swaps_bgr_to_rgb_and_normalizes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _block_ultralytics(monkeypatch)
    engine_path = tmp_path / "norm.engine"
    engine_path.write_bytes(b"engine")

    manifest = _make_manifest(
        input_spec=TensorSpec("images", "float16", (1, 3, 736, 1280)),
        output_spec=TensorSpec("output0", "float16", (1, 5, 4)),
    )
    runner = _FakeYoloRunner(manifest=manifest)
    monkeypatch.setattr(trt_runtime, "load_trt_runner", lambda *_a, **_kw: runner)

    # 构造只读输入，验证非可写数组同样安全
    img = np.zeros((720, 1280, 3), dtype=np.uint8)
    img[:, :, 0] = 51  # B
    img[:, :, 1] = 102  # G
    img[:, :, 2] = 204  # R
    img.setflags(write=False)

    detector = Detector(DetectorConfig(model_path=str(engine_path), device="0", imgsz=1280))
    detector.detect(img, undistort=False)

    inp = runner.seen_inputs[0]
    assert inp.shape == (1, 3, 736, 1280)
    assert inp.dtype == torch.float16
    assert inp.is_contiguous()
    # 中心有效图像区：RGB 顺序应为 (204, 102, 51) / 255
    assert float(inp[0, 0, 100, 100]) == pytest.approx(204.0 / 255.0, abs=1e-3)
    assert float(inp[0, 1, 100, 100]) == pytest.approx(102.0 / 255.0, abs=1e-3)
    assert float(inp[0, 2, 100, 100]) == pytest.approx(51.0 / 255.0, abs=1e-3)
    # 顶部灰边区：114 / 255
    assert float(inp[0, 0, 0, 0]) == pytest.approx(114.0 / 255.0, abs=1e-3)


def test_detector_trt_coordinate_unscaling_fast_and_general(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _block_ultralytics(monkeypatch)
    engine_path = tmp_path / "unscale.engine"
    engine_path.write_bytes(b"engine")

    # 1) 720x1280 -> 736x1280 快路径（pad_y = 8.0, gain = 1.0）
    raw_fast = _build_raw_yolo_output(
        nc=1,
        rows=[(160.0, 118.0, 120.0, 120.0, {0: 0.9})],
        num_anchors=8,
        dtype=torch.float32,
    )
    manifest = _make_manifest(
        input_spec=TensorSpec("images", "float32", (1, 3, 736, 1280)),
        output_spec=TensorSpec("output0", "float32", (1, 5, 8)),
    )
    monkeypatch.setattr(
        trt_runtime,
        "load_trt_runner",
        lambda *_a, **_kw: _FakeYoloRunner(manifest=manifest, output_tensor=raw_fast),
    )
    detector = Detector(DetectorConfig(model_path=str(engine_path), device="0", imgsz=1280))
    batch = detector.detect(np.zeros((720, 1280, 3), dtype=np.uint8), mode="ocr", undistort=False)
    assert len(batch.detections) == 1
    box = batch.detections[0].box
    assert (box.x1, box.y1, box.x2, box.y2) == pytest.approx((100.0, 50.0, 220.0, 170.0), abs=1e-3)

    # 2) 任意宽高比 480x640 -> 736x1280（gain = 736/480, nw = 981, pad_x = 149.5, pad_y = 0）
    orig_h, orig_w = 480, 640
    gain = 736.0 / 480.0
    nw = int(round(orig_w * gain))
    pad_x = (1280 - nw) / 2.0
    want_x1, want_y1, want_x2, want_y2 = 120.0, 80.0, 260.0, 200.0
    lb_cx = ((want_x1 + want_x2) / 2.0) * gain + pad_x
    lb_cy = ((want_y1 + want_y2) / 2.0) * gain
    lb_w = (want_x2 - want_x1) * gain
    lb_h = (want_y2 - want_y1) * gain

    raw_gen = _build_raw_yolo_output(
        nc=1,
        rows=[
            (lb_cx, lb_cy, lb_w, lb_h, {0: 0.88}),
            # 越界框应被裁剪到原图范围内
            (pad_x, 0.0, 100.0, 100.0, {0: 0.70}),
        ],
        num_anchors=8,
        dtype=torch.float32,
    )
    monkeypatch.setattr(
        trt_runtime,
        "load_trt_runner",
        lambda *_a, **_kw: _FakeYoloRunner(manifest=manifest, output_tensor=raw_gen),
    )
    detector_gen = Detector(DetectorConfig(model_path=str(engine_path), device="0", imgsz=1280))
    batch_gen = detector_gen.detect(
        np.zeros((orig_h, orig_w, 3), dtype=np.uint8), mode="ocr", undistort=False
    )
    assert len(batch_gen.detections) == 2
    b0 = batch_gen.detections[0].box
    assert (b0.x1, b0.y1, b0.x2, b0.y2) == pytest.approx(
        (want_x1, want_y1, want_x2, want_y2), abs=1e-2
    )
    b1 = batch_gen.detections[1].box
    assert b1.x1 == 0.0 and b1.y1 == 0.0
    assert b1.x2 <= orig_w - 1.0 and b1.y2 <= orig_h - 1.0


# ======================================================================
# 4. 单类（ocr）与多类（cls12）后处理、NMS 与基准工具兼容性单测
# ======================================================================
def test_detector_trt_single_class_ocr_mode_and_empty_frame(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _block_ultralytics(monkeypatch)
    engine_path = tmp_path / "ocr.engine"
    engine_path.write_bytes(b"engine")

    manifest = _make_manifest(
        input_spec=TensorSpec("images", "float16", (1, 3, 736, 1280)),
        output_spec=TensorSpec("output0", "float16", (1, 5, 10)),
    )

    # 空检出帧（全部低于 conf_threshold）
    empty_raw = _build_raw_yolo_output(
        nc=1,
        rows=[(200.0, 200.0, 60.0, 60.0, {0: 0.25})],  # 严格大于 conf_threshold=0.25
        num_anchors=10,
    )
    monkeypatch.setattr(
        trt_runtime,
        "load_trt_runner",
        lambda *_a, **_kw: _FakeYoloRunner(manifest=manifest, output_tensor=empty_raw),
    )
    detector = Detector(
        DetectorConfig(
            model_path=str(engine_path),
            device="0",
            imgsz=1280,
            conf_threshold=0.25,
            iou_threshold=0.45,
        )
    )
    snap = TelemetrySnapshot(north_m=1.0, east_m=2.0, down_m=-30.0)
    empty_batch = detector.detect(
        np.zeros((720, 1280, 3), dtype=np.uint8),
        frame_index=7,
        capture_timestamp=12.5,
        telemetry=snap,
        mode="ocr",
        undistort=False,
    )
    assert empty_batch.detections == ()

    # 多框 + 低于阈值过滤 + 同类 NMS 抑制 + 置信度降序
    raw = _build_raw_yolo_output(
        nc=1,
        rows=[
            (300.0, 308.0, 80.0, 80.0, {0: 0.65}),  # 目标 B（较低置信度）
            (150.0, 158.0, 100.0, 100.0, {0: 0.92}),  # 目标 A（最高置信度）
            (152.0, 160.0, 98.0, 98.0, {0: 0.80}),  # 与目标 A 高重叠，应被 NMS 抑制
            (500.0, 508.0, 40.0, 40.0, {0: 0.10}),  # 低于阈值
        ],
        num_anchors=10,
    )
    monkeypatch.setattr(
        trt_runtime,
        "load_trt_runner",
        lambda *_a, **_kw: _FakeYoloRunner(manifest=manifest, output_tensor=raw),
    )
    detector2 = Detector(
        DetectorConfig(
            model_path=str(engine_path),
            device="0",
            imgsz=1280,
            conf_threshold=0.25,
            iou_threshold=0.45,
        )
    )
    batch = detector2.detect(
        np.zeros((720, 1280, 3), dtype=np.uint8),
        frame_index=9,
        capture_timestamp=20.0,
        telemetry=snap,
        mode="ocr",
        undistort=False,
    )
    assert len(batch.detections) == 2
    assert batch.detections[0].confidence == pytest.approx(0.92, abs=1e-2)
    assert batch.detections[1].confidence == pytest.approx(0.65, abs=1e-2)
    for det in batch.detections:
        assert det.code is None
        assert det.mode == "ocr"
        assert det.frame_index == 9
        assert det.capture_timestamp == 20.0
        assert det.telemetry is snap
        assert det.extra == {
            "class_name": "target",
            "undistorted": False,
            "image_size": (1280, 720),
        }


def test_detector_trt_multiclass_cls12_class_offset_nms_and_max_detections(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _block_ultralytics(monkeypatch)
    engine_path = tmp_path / "cls12.engine"
    engine_path.write_bytes(b"engine")

    names_12 = {str(i): f"num_{i + 1}" for i in range(12)}
    manifest = _make_manifest(
        input_spec=TensorSpec("images", "float16", (1, 3, 736, 1280)),
        output_spec=TensorSpec("output0", "float16", (1, 16, 12)),
        yolo_meta={"input_hw": [736, 1280], "names": names_12, "end2end": False},
    )
    raw = _build_raw_yolo_output(
        nc=12,
        rows=[
            # 类别 0（code=1）与类别 5（code=6）完全同框：类别偏移 NMS 必须双双保留
            (150.0, 158.0, 100.0, 100.0, {0: 0.90, 2: 0.10}),
            (150.0, 158.0, 100.0, 100.0, {5: 0.85}),
            # 类别 0 高度重叠的低分框：必须被同类别高分框抑制
            (152.0, 160.0, 98.0, 98.0, {0: 0.72}),
            # 类别 11（code=12）最高置信度
            (450.0, 358.0, 100.0, 100.0, {11: 0.96}),
        ],
        num_anchors=12,
    )
    monkeypatch.setattr(
        trt_runtime,
        "load_trt_runner",
        lambda *_a, **_kw: _FakeYoloRunner(manifest=manifest, output_tensor=raw),
    )

    detector = Detector(
        DetectorConfig(
            model_path=str(engine_path),
            device="0",
            imgsz=1280,
            conf_threshold=0.25,
            iou_threshold=0.45,
            max_detections=300,
        )
    )
    batch = detector.detect(np.zeros((720, 1280, 3), dtype=np.uint8), mode="cls12", undistort=False)
    assert [(d.code, d.extra["class_name"]) for d in batch.detections] == [
        (12, "num_12"),
        (1, "num_1"),
        (6, "num_6"),
    ]
    assert [round(d.confidence, 2) for d in batch.detections] == [0.96, 0.90, 0.85]

    # max_detections 截断到最高置信度的前 2 个
    detector_capped = Detector(
        DetectorConfig(
            model_path=str(engine_path),
            device="0",
            imgsz=1280,
            conf_threshold=0.25,
            iou_threshold=0.45,
            max_detections=2,
        )
    )
    capped_batch = detector_capped.detect(
        np.zeros((720, 1280, 3), dtype=np.uint8), mode="cls12", undistort=False
    )
    assert [d.code for d in capped_batch.detections] == [12, 1]


def test_detector_trt_supports_benchmark_predict_wrapping(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """验证 benchmark_perception 的 model.predict 实例级包裹模式兼容 _TrtYoloModel。"""
    _block_ultralytics(monkeypatch)
    engine_path = tmp_path / "bench.engine"
    engine_path.write_bytes(b"engine")

    manifest = _make_manifest()
    raw = _build_raw_yolo_output(nc=1, rows=[(200.0, 208.0, 80.0, 80.0, {0: 0.9})], num_anchors=16)
    monkeypatch.setattr(
        trt_runtime,
        "load_trt_runner",
        lambda *_a, **_kw: _FakeYoloRunner(manifest=manifest, output_tensor=raw),
    )
    detector = Detector(DetectorConfig(model_path=str(engine_path), device="0", imgsz=1280))
    detector.load()

    model = detector._model
    original_predict = model.predict
    wrapped_calls = 0

    def timed_predict(*args: Any, **kwargs: Any):
        nonlocal wrapped_calls
        wrapped_calls += 1
        return original_predict(*args, **kwargs)

    model.predict = timed_predict
    try:
        batch = detector.detect(np.zeros((720, 1280, 3), dtype=np.uint8), undistort=False)
    finally:
        model.predict = original_predict

    assert wrapped_calls == 1
    assert len(batch.detections) == 1
