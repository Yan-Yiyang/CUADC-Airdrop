"""TensorRT 离线引擎清单校验与执行封装（延迟导入 tensorrt 与 torch）。

设计契约
--------
* 所有 TensorRT 引擎（``.engine``）必须附带同目录伴随清单 ``<name>.engine.manifest.json``，
  记录产物 SHA256、源权重/字典哈希、输入输出规格及目标运行环境；
* 加载时先整块读取引擎字节并在内存中校验 SHA256 与运行环境（操作系统、TensorRT 版本、
  GPU 计算能力），任一不匹配直接抛出 :class:`TrtEngineError`，不在任务中重建或静默回退；
* 使用独立非默认 CUDA stream 执行 ``execute_async_v3``，避免默认流额外隐式同步，
  且直接复用 PyTorch CUDA 张量显存地址，不依赖 ``pycuda`` 或 ``cuda-bindings``。
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

__all__ = [
    "MANIFEST_SCHEMA_VERSION",
    "TensorSpec",
    "TrtEngineError",
    "TrtExecutionRunner",
    "TrtManifest",
    "TrtShapeOutOfRangeError",
    "current_cuda_environment",
    "load_trt_runner",
    "manifest_path_for",
    "parse_cuda_device_id",
    "sha256_bytes",
    "sha256_file",
    "validate_manifest_dict",
    "validate_runtime_environment",
]

MANIFEST_SCHEMA_VERSION = 1
_VALID_KINDS = frozenset({"yolo", "ocr_det", "ocr_rec"})
_TRT_LOGGER: Any = None


def _get_trt_logger(trt: Any) -> Any:
    """复用进程级 TensorRT Logger，避免多引擎加载时触发重复注册告警。"""
    global _TRT_LOGGER  # noqa: PLW0603 - TensorRT 运行时要求进程内复用同一 Logger 实例
    if _TRT_LOGGER is None:
        _TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    return _TRT_LOGGER


class TrtEngineError(RuntimeError):
    """TensorRT 引擎清单无效、环境不匹配、反序列化或执行失败。"""


class TrtShapeOutOfRangeError(TrtEngineError):
    """推理输入张量形状超出 TensorRT 引擎优化配置边界。"""


@dataclass(frozen=True, slots=True)
class TensorSpec:
    """单个输入/输出张量的契约描述。"""

    name: str
    dtype: str
    shape: tuple[int, ...]
    min_shape: tuple[int, ...] | None = None
    opt_shape: tuple[int, ...] | None = None
    max_shape: tuple[int, ...] | None = None

    @property
    def is_dynamic(self) -> bool:
        return any(dim < 0 for dim in self.shape)


@dataclass(frozen=True, slots=True)
class TrtManifest:
    """经过结构校验的 TensorRT 引擎伴随清单。"""

    schema_version: int
    kind: str
    engine_file: str
    engine_sha256: str
    source: dict[str, Any]
    model: dict[str, Any]
    environment: dict[str, Any]
    verification: dict[str, Any]
    inputs: tuple[TensorSpec, ...]
    outputs: tuple[TensorSpec, ...]


def manifest_path_for(engine_path: str | Path) -> Path:
    """返回引擎文件对应的伴随清单路径。"""
    path = Path(engine_path)
    return path.with_name(f"{path.name}.manifest.json")


def sha256_bytes(data: bytes) -> str:
    """计算内存字节串的 SHA256 十六进制摘要。"""
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    """流式计算文件的 SHA256 十六进制摘要。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_cuda_device_id(device: str) -> int:
    """解析单一 CUDA 设备编号；拒绝 CPU、MPS、多卡或无效格式。"""
    normalized = str(device).strip().lower()
    if normalized == "cuda":
        return 0
    if normalized.startswith("cuda:"):
        suffix = normalized.removeprefix("cuda:")
        if suffix.isdigit():
            return int(suffix)
        raise ValueError(f"TensorRT 仅支持单一 CUDA 设备编号: {device!r}")
    if normalized.isdigit():
        return int(normalized)
    raise ValueError(f"TensorRT 仅支持单一 CUDA 设备编号（不支持 {device!r}）")


def _parse_int_tuple(
    value: Any, field_name: str, *, allow_negative: bool = False
) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise TrtEngineError(f"清单字段 {field_name} 必须是非空整数数组")
    out: list[int] = []
    for item in value:
        if type(item) is not int or (not allow_negative and item <= 0):
            raise TrtEngineError(f"清单字段 {field_name} 包含无效维度值: {item!r}")
        out.append(item)
    return tuple(out)


def _parse_tensor_spec(item: Any, field_name: str, *, is_input: bool) -> TensorSpec:
    if not isinstance(item, Mapping):
        raise TrtEngineError(f"清单字段 {field_name} 必须是对象")
    name = item.get("name")
    dtype = item.get("dtype")
    if not isinstance(name, str) or not name.strip():
        raise TrtEngineError(f"清单字段 {field_name}.name 无效")
    if not isinstance(dtype, str) or dtype not in ("float16", "float32", "int32", "int64"):
        raise TrtEngineError(f"清单字段 {field_name}.dtype 不受支持: {dtype!r}")
    shape = _parse_int_tuple(item.get("shape"), f"{field_name}.shape", allow_negative=True)
    min_shape = None
    opt_shape = None
    max_shape = None
    if is_input and any(dim < 0 for dim in shape):
        min_shape = _parse_int_tuple(item.get("min_shape"), f"{field_name}.min_shape")
        opt_shape = _parse_int_tuple(item.get("opt_shape"), f"{field_name}.opt_shape")
        max_shape = _parse_int_tuple(item.get("max_shape"), f"{field_name}.max_shape")
        if not (len(min_shape) == len(opt_shape) == len(max_shape) == len(shape)):
            raise TrtEngineError(f"清单字段 {field_name} 的动态 profile 秩与 shape 不一致")
        for lo, opt, hi in zip(min_shape, opt_shape, max_shape, strict=True):
            if not (lo <= opt <= hi):
                raise TrtEngineError(
                    f"清单字段 {field_name} 的动态 profile 边界无效: {min_shape} / {opt_shape} / {max_shape}"
                )
    return TensorSpec(
        name=name,
        dtype=dtype,
        shape=shape,
        min_shape=min_shape,
        opt_shape=opt_shape,
        max_shape=max_shape,
    )


def _require_mapping(data: Mapping[str, Any], field_name: str) -> dict[str, Any]:
    section = data.get(field_name)
    if not isinstance(section, Mapping):
        raise TrtEngineError(f"TensorRT 清单缺少有效字段 {field_name}")
    return {str(key): value for key, value in section.items()}


def validate_manifest_dict(
    data: Any,
    *,
    expected_kind: str,
    expected_engine_name: str,
    expected_engine_sha256: str,
) -> TrtManifest:
    """校验清单结构、产物哈希与声明类别；失败抛出 :class:`TrtEngineError`。"""
    if expected_kind not in _VALID_KINDS:
        raise ValueError(f"未知的 TensorRT 清单类别: {expected_kind!r}")
    if not isinstance(data, Mapping):
        raise TrtEngineError("TensorRT 清单顶层必须是 JSON 对象")
    schema_version = data.get("schema_version")
    if type(schema_version) is not int or schema_version != MANIFEST_SCHEMA_VERSION:
        raise TrtEngineError(
            f"TensorRT 清单版本不匹配: 实际 {schema_version!r}，期望 {MANIFEST_SCHEMA_VERSION}"
        )
    kind = data.get("kind")
    if kind != expected_kind:
        raise TrtEngineError(f"TensorRT 清单类别不匹配: 实际 {kind!r}，期望 {expected_kind!r}")
    engine_file = data.get("engine_file")
    if engine_file != expected_engine_name:
        raise TrtEngineError(
            f"TensorRT 清单记录的引擎文件名不匹配: 实际 {engine_file!r}，期望 {expected_engine_name!r}"
        )
    engine_sha256 = data.get("engine_sha256")
    if (
        not isinstance(engine_sha256, str)
        or engine_sha256.lower() != expected_engine_sha256.lower()
    ):
        raise TrtEngineError("TensorRT 引擎文件 SHA256 与清单不一致，请重新构建并原子发布")

    source = _require_mapping(data, "source")
    model = _require_mapping(data, "model")
    environment = _require_mapping(data, "environment")
    verification = _require_mapping(data, "verification")

    raw_inputs = model.get("inputs")
    raw_outputs = model.get("outputs")
    if not isinstance(raw_inputs, list) or not raw_inputs:
        raise TrtEngineError("TensorRT 清单 model.inputs 必须是非空数组")
    if not isinstance(raw_outputs, list) or not raw_outputs:
        raise TrtEngineError("TensorRT 清单 model.outputs 必须是非空数组")

    inputs = tuple(
        _parse_tensor_spec(item, f"model.inputs[{idx}]", is_input=True)
        for idx, item in enumerate(raw_inputs)
    )
    outputs = tuple(
        _parse_tensor_spec(item, f"model.outputs[{idx}]", is_input=False)
        for idx, item in enumerate(raw_outputs)
    )
    return TrtManifest(
        schema_version=schema_version,
        kind=str(kind),
        engine_file=str(engine_file),
        engine_sha256=engine_sha256.lower(),
        source=source,
        model=model,
        environment=environment,
        verification=verification,
        inputs=inputs,
        outputs=outputs,
    )


def validate_runtime_environment(manifest: TrtManifest, current_env: Mapping[str, str]) -> None:
    """核对当前运行环境与引擎构建环境（平台、TensorRT 版本、GPU 计算能力）。"""
    env = manifest.environment
    for key, label in (
        ("platform", "操作系统平台"),
        ("tensorrt_version", "TensorRT 版本"),
        ("compute_capability", "GPU 计算能力"),
    ):
        expected = env.get(key)
        actual = current_env.get(key)
        if not isinstance(expected, str) or not expected:
            raise TrtEngineError(f"TensorRT 清单缺少环境字段 environment.{key}")
        if expected != actual:
            raise TrtEngineError(
                f"TensorRT 引擎{label}不匹配: 清单为 {expected!r}，当前环境为 {actual!r}；"
                "请在当前目标机器重新离线构建引擎"
            )


def current_cuda_environment(device_id: int = 0) -> dict[str, str]:
    """采集当前 CUDA 与 TensorRT 运行环境信息（惰性导入重依赖）。"""
    try:
        import torch
    except ImportError as exc:
        raise TrtEngineError("加载 TensorRT 引擎需要可用的 PyTorch CUDA 运行时") from exc
    if not torch.cuda.is_available():
        raise TrtEngineError("当前环境未检测到可用 CUDA 设备，无法运行 TensorRT 引擎")
    device_count = torch.cuda.device_count()
    if device_id < 0 or device_id >= device_count:
        raise TrtEngineError(
            f"请求的 CUDA 设备编号 {device_id} 超出范围（可用设备数 {device_count}）"
        )
    try:
        import tensorrt as trt
    except ImportError as exc:
        raise TrtEngineError("当前环境未安装 tensorrt 运行库，无法加载 .engine 模型") from exc

    major, minor = torch.cuda.get_device_capability(device_id)
    gpu_name = str(torch.cuda.get_device_name(device_id))
    cuda_ver = str(torch.version.cuda or "")
    onnx_ver = ""
    try:
        import onnx

        onnx_ver = str(getattr(onnx, "__version__", ""))
    except ImportError:
        onnx_ver = ""
    return {
        "platform": sys.platform,
        "gpu_name": gpu_name,
        "compute_capability": f"sm{major}{minor}",
        "cuda_version": cuda_ver,
        "tensorrt_version": str(getattr(trt, "__version__", "")),
        "torch_version": str(getattr(torch, "__version__", "")),
        "onnx_version": onnx_ver,
    }


def _dtype_name_from_trt(trt_dtype: Any) -> str:
    import tensorrt as trt_mod

    trt: Any = cast(Any, trt_mod)
    np_dtype = np.dtype(trt.nptype(trt_dtype))
    return str(np_dtype.name)


def _torch_dtype_for_name(dtype_name: str, torch_module: Any) -> Any:
    mapping = {
        "float16": torch_module.float16,
        "float32": torch_module.float32,
        "int32": torch_module.int32,
        "int64": torch_module.int64,
    }
    if dtype_name not in mapping:
        raise TrtEngineError(f"不支持的张量数据类型: {dtype_name!r}")
    return mapping[dtype_name]


def _check_input_shape(spec: TensorSpec, shape: tuple[int, ...]) -> None:
    if len(shape) != len(spec.shape):
        raise TrtShapeOutOfRangeError(
            f"输入 {spec.name!r} 维度数不匹配: 实际 {shape}，期望秩 {len(spec.shape)}"
        )
    if not spec.is_dynamic:
        if shape != spec.shape:
            raise TrtShapeOutOfRangeError(
                f"静态输入 {spec.name!r} 形状不匹配: 实际 {shape}，期望 {spec.shape}"
            )
        return
    assert spec.min_shape is not None and spec.max_shape is not None
    for axis, (actual, expected, lo, hi) in enumerate(
        zip(shape, spec.shape, spec.min_shape, spec.max_shape, strict=True)
    ):
        if expected > 0 and actual != expected:
            raise TrtShapeOutOfRangeError(
                f"输入 {spec.name!r} 第 {axis} 维固定为 {expected}，实际收到 {actual}"
            )
        if not (lo <= actual <= hi):
            raise TrtShapeOutOfRangeError(
                f"输入 {spec.name!r} 形状 {shape} 超出引擎 profile 范围 [{spec.min_shape}, {spec.max_shape}]"
            )


class TrtExecutionRunner:
    """单输入单输出 TensorRT 引擎执行器（基于独立 CUDA stream 与 PyTorch 显存）。"""

    def __init__(
        self,
        *,
        manifest: TrtManifest,
        runtime: Any,
        engine: Any,
        context: Any,
        device_id: int,
        torch_module: Any,
    ) -> None:
        self._manifest = manifest
        self._runtime = runtime
        self._engine = engine
        self._context = context
        self._device_id = int(device_id)
        self._torch = torch_module
        self._device = torch_module.device(f"cuda:{device_id}")
        self._stream = torch_module.cuda.Stream(device=self._device)
        if len(manifest.inputs) != 1 or len(manifest.outputs) != 1:
            raise TrtEngineError("当前执行器仅支持单输入单输出 TensorRT 引擎")
        self._input_spec = manifest.inputs[0]
        self._output_spec = manifest.outputs[0]
        self._input_dtype = _torch_dtype_for_name(self._input_spec.dtype, torch_module)
        self._output_dtype = _torch_dtype_for_name(self._output_spec.dtype, torch_module)
        self._cached_out_shape: tuple[int, ...] | None = None
        self._cached_out_buf: Any = None

    @property
    def manifest(self) -> TrtManifest:
        return self._manifest

    @property
    def input_spec(self) -> TensorSpec:
        return self._input_spec

    @property
    def output_spec(self) -> TensorSpec:
        return self._output_spec

    @property
    def device(self) -> Any:
        return self._device

    @property
    def device_id(self) -> int:
        return self._device_id

    @property
    def input_torch_dtype(self) -> Any:
        return self._input_dtype

    def clone_context(self) -> TrtExecutionRunner:
        """基于当前已反序列化引擎创建独立执行上下文与非默认 CUDA stream。

        共享不可变的 ``runtime``、``engine`` 与 ``manifest``，供同进程多线程工作池
        并发执行，避免重复反序列化引擎和创建额外 CUDA 主上下文。
        """
        set_device = getattr(self._torch.cuda, "set_device", None)
        if callable(set_device):
            set_device(self._device_id)
        context = self._engine.create_execution_context()
        if context is None:
            raise TrtEngineError(
                f"无法为 TensorRT 引擎创建独立 ExecutionContext（kind={self._manifest.kind}）"
            )
        return TrtExecutionRunner(
            manifest=self._manifest,
            runtime=self._runtime,
            engine=self._engine,
            context=context,
            device_id=self._device_id,
            torch_module=self._torch,
        )

    def run_tensor(self, input_tensor: Any) -> Any:
        """执行一次前向推理并返回同设备上的输出 PyTorch 张量。"""
        torch = self._torch
        shape = tuple(int(dim) for dim in input_tensor.shape)
        _check_input_shape(self._input_spec, shape)

        with torch.no_grad():
            inp = input_tensor
            if inp.device != self._device or inp.dtype != self._input_dtype:
                inp = inp.to(device=self._device, dtype=self._input_dtype)
            if not inp.is_contiguous():
                inp = inp.contiguous()
            if self._input_spec.is_dynamic and not self._context.set_input_shape(
                self._input_spec.name, shape
            ):
                raise TrtShapeOutOfRangeError(
                    f"TensorRT 拒绝设置输入 {self._input_spec.name!r} 形状 {shape}"
                )
            out_shape = tuple(
                int(dim) for dim in self._context.get_tensor_shape(self._output_spec.name)
            )
            if any(dim < 0 for dim in out_shape):
                raise TrtEngineError(
                    f"TensorRT 未能解析输出 {self._output_spec.name!r} 的有效形状: {out_shape}"
                )
            if self._cached_out_shape != out_shape or self._cached_out_buf is None:
                self._cached_out_buf = torch.empty(
                    out_shape, device=self._device, dtype=self._output_dtype
                )
                self._cached_out_shape = out_shape
            out_buf = self._cached_out_buf

            current_stream = torch.cuda.current_stream(self._device)
            self._stream.wait_stream(current_stream)
            self._context.set_tensor_address(self._input_spec.name, int(inp.data_ptr()))
            self._context.set_tensor_address(self._output_spec.name, int(out_buf.data_ptr()))
            ok = self._context.execute_async_v3(self._stream.cuda_stream)
            if not ok:
                raise TrtEngineError(f"TensorRT 执行失败（kind={self._manifest.kind}）")
            current_stream.wait_stream(self._stream)
            return out_buf

    def run_numpy(self, array: np.ndarray) -> np.ndarray:
        """接收 NumPy 输入，在独立 CUDA stream 上执行推理并返回 float32 NumPy 数组。"""
        torch = self._torch
        contiguous = np.ascontiguousarray(array)
        shape = tuple(int(dim) for dim in contiguous.shape)
        _check_input_shape(self._input_spec, shape)

        with torch.no_grad():
            with torch.cuda.stream(self._stream):
                inp = torch.from_numpy(contiguous).to(device=self._device, non_blocking=False)
                if inp.dtype != self._input_dtype:
                    inp = inp.to(dtype=self._input_dtype)
                if self._input_spec.is_dynamic and not self._context.set_input_shape(
                    self._input_spec.name, shape
                ):
                    raise TrtShapeOutOfRangeError(
                        f"TensorRT 拒绝设置输入 {self._input_spec.name!r} 形状 {shape}"
                    )
                out_shape = tuple(
                    int(dim) for dim in self._context.get_tensor_shape(self._output_spec.name)
                )
                if any(dim < 0 for dim in out_shape):
                    raise TrtEngineError(
                        f"TensorRT 未能解析输出 {self._output_spec.name!r} 的有效形状: {out_shape}"
                    )
                if self._cached_out_shape != out_shape or self._cached_out_buf is None:
                    self._cached_out_buf = torch.empty(
                        out_shape, device=self._device, dtype=self._output_dtype
                    )
                    self._cached_out_shape = out_shape
                out_buf = self._cached_out_buf
                self._context.set_tensor_address(self._input_spec.name, int(inp.data_ptr()))
                self._context.set_tensor_address(self._output_spec.name, int(out_buf.data_ptr()))
                ok = self._context.execute_async_v3(self._stream.cuda_stream)
                if not ok:
                    raise TrtEngineError(f"TensorRT 执行失败（kind={self._manifest.kind}）")
                out_fp32 = out_buf.float() if out_buf.dtype != torch.float32 else out_buf
            self._stream.synchronize()
            return out_fp32.cpu().numpy()


def _verify_engine_io_against_manifest(engine: Any, manifest: TrtManifest) -> None:
    """核对反序列化后的引擎张量名称、方向、类型与 profile 边界与清单一致。"""
    import tensorrt as trt_mod

    trt: Any = cast(Any, trt_mod)

    num_io = int(engine.num_io_tensors)
    inputs: dict[str, tuple[str, tuple[int, ...]]] = {}
    outputs: dict[str, tuple[str, tuple[int, ...]]] = {}
    for index in range(num_io):
        name = str(engine.get_tensor_name(index))
        mode = engine.get_tensor_mode(name)
        dtype = _dtype_name_from_trt(engine.get_tensor_dtype(name))
        shape = tuple(int(dim) for dim in engine.get_tensor_shape(name))
        if mode == trt.TensorIOMode.INPUT:
            inputs[name] = (dtype, shape)
        elif mode == trt.TensorIOMode.OUTPUT:
            outputs[name] = (dtype, shape)

    if len(inputs) != len(manifest.inputs) or len(outputs) != len(manifest.outputs):
        raise TrtEngineError(
            f"TensorRT 引擎 I/O 数量与清单不符: 引擎 ({len(inputs)}, {len(outputs)})，"
            f"清单 ({len(manifest.inputs)}, {len(manifest.outputs)})"
        )
    for spec in manifest.inputs:
        if spec.name not in inputs:
            raise TrtEngineError(f"TensorRT 引擎缺少清单声明的输入张量 {spec.name!r}")
        actual_dtype, actual_shape = inputs[spec.name]
        if actual_dtype != spec.dtype or actual_shape != spec.shape:
            raise TrtEngineError(
                f"TensorRT 输入张量 {spec.name!r} 与清单不符: "
                f"实际 ({actual_dtype}, {actual_shape})，清单 ({spec.dtype}, {spec.shape})"
            )
        if spec.is_dynamic:
            if int(engine.num_optimization_profiles) < 1:
                raise TrtEngineError("动态 TensorRT 引擎缺少优化配置（optimization profile）")
            lo, opt, hi = engine.get_tensor_profile_shape(spec.name, 0)
            actual_bounds = (
                tuple(int(d) for d in lo),
                tuple(int(d) for d in opt),
                tuple(int(d) for d in hi),
            )
            if actual_bounds != (spec.min_shape, spec.opt_shape, spec.max_shape):
                raise TrtEngineError(
                    f"TensorRT 输入 {spec.name!r} profile 与清单不符: "
                    f"实际 {actual_bounds}，清单 {(spec.min_shape, spec.opt_shape, spec.max_shape)}"
                )
    for spec in manifest.outputs:
        if spec.name not in outputs:
            raise TrtEngineError(f"TensorRT 引擎缺少清单声明的输出张量 {spec.name!r}")
        actual_dtype, actual_shape = outputs[spec.name]
        if actual_dtype != spec.dtype or actual_shape != spec.shape:
            raise TrtEngineError(
                f"TensorRT 输出张量 {spec.name!r} 与清单不符: "
                f"实际 ({actual_dtype}, {actual_shape})，清单 ({spec.dtype}, {spec.shape})"
            )


def load_trt_runner(
    engine_path: str | Path,
    *,
    expected_kind: str,
    device_id: int = 0,
) -> TrtExecutionRunner:
    """读取、校验清单与环境并反序列化 TensorRT 引擎。"""
    path = Path(engine_path)
    if not path.is_file():
        raise FileNotFoundError(f"TensorRT 引擎文件不存在: {path}")
    mpath = manifest_path_for(path)
    if not mpath.is_file():
        raise TrtEngineError(f"TensorRT 伴随清单缺失: {mpath}（请通过离线构建工具原子发布）")

    engine_bytes = path.read_bytes()
    actual_sha = sha256_bytes(engine_bytes)
    try:
        raw_manifest = json.loads(mpath.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TrtEngineError(f"无法读取或解析 TensorRT 伴随清单 {mpath}: {exc}") from exc

    manifest = validate_manifest_dict(
        raw_manifest,
        expected_kind=expected_kind,
        expected_engine_name=path.name,
        expected_engine_sha256=actual_sha,
    )
    current_env = current_cuda_environment(device_id)
    validate_runtime_environment(manifest, current_env)

    import tensorrt as trt_mod
    import torch

    trt: Any = cast(Any, trt_mod)
    torch.cuda.set_device(device_id)
    # TensorRT 计算全部在 CUDA 执行；限制 PyTorch CPU OpenMP 线程数为 1，
    # 避免多线程并发推理时触发 OpenMP 线程池争用与自旋空转。
    set_num_threads = getattr(torch, "set_num_threads", None)
    if callable(set_num_threads):
        set_num_threads(1)
    logger = _get_trt_logger(trt)
    runtime = trt.Runtime(logger)
    if runtime is None:
        raise TrtEngineError("无法创建 TensorRT Runtime 实例")
    engine = runtime.deserialize_cuda_engine(engine_bytes)
    if engine is None:
        raise TrtEngineError(f"TensorRT 引擎反序列化失败: {path}")
    _verify_engine_io_against_manifest(engine, manifest)
    context = engine.create_execution_context()
    if context is None:
        raise TrtEngineError(f"无法创建 TensorRT ExecutionContext: {path}")
    return TrtExecutionRunner(
        manifest=manifest,
        runtime=runtime,
        engine=engine,
        context=context,
        device_id=device_id,
        torch_module=torch,
    )
