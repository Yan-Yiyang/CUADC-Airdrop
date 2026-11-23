"""TensorRT 离线引擎构建与校验工具（纯库模块，重依赖全部惰性导入）。

职责与不变量
------------
* 本模块不实现命令行解析（CLI 由 ``airdrop/run.py`` 的 ``build-trt`` 子命令统一解析）；
* 所有重依赖（``torch`` / ``ultralytics`` / ``rapidocr`` / ``onnx`` / ``tensorrt``）
  仅在构建与校验函数体内惰性导入，保证 ``--help`` 与离线测试零重库加载；
* 兼容 TensorRT 11+ 强类型模式（无 ``BuilderFlag.FP16``、启用 ``STRONGLY_TYPED``，
  直接导出 FP16 ONNX）以及 TensorRT 10 ``BuilderFlag.FP16`` 构建路径；
* 默认执行离线数值校验（``verify=True``）：在临时目录加载 ``TrtExecutionRunner``，
  对照 PyTorch 参考模型核验形状、有限性与误差上界，未通过立即中止并清理临时文件；
* 发布采用 :func:`atomic_publish_engine` 原子落盘（先写临时文件并 ``fsync``，
  再先替换 ``.engine`` 后替换 ``.engine.manifest.json``），伴随清单只记录文件名与摘要，
  禁止写入任何本机绝对路径。
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import tempfile
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

__all__ = [
    "DET_MAX_SIDE_LEN",
    "DEVICE",
    "FORCE",
    "IMGSZ",
    "OCR_DET_MODEL",
    "OCR_MODELS_DIR",
    "OCR_REC_KEYS",
    "OCR_REC_MODEL",
    "OUT_DIR",
    "PRECISION",
    "REC_MAX_BATCH",
    "REC_MAX_WIDTH",
    "TARGET",
    "VALID_PRECISIONS",
    "VALID_TARGETS",
    "VERIFY",
    "WORKSPACE_MB",
    "YOLO_INPUT_HW",
    "YOLO_MODEL",
    "BuildTrtConfig",
    "atomic_publish_engine",
    "build_config",
    "build_ocr_engines",
    "build_yolo_engine",
    "main",
    "ocr_det_engine_path",
    "ocr_rec_engine_path",
    "yolo_engine_path",
]

LOGGER = logging.getLogger("build_trt_engines")

# ----------------------------------------------------------------------
# 模块级默认常量（由 airdrop/run.py 读取作为 CLI 默认值）
# ----------------------------------------------------------------------
TARGET = "all"
OUT_DIR = ".local/trt"
DEVICE = "0"
YOLO_MODEL = "models/best2.pt"
IMGSZ = 1280
YOLO_INPUT_HW = (736, 1280)
OCR_MODELS_DIR = "models/ppocr"
OCR_DET_MODEL = "PP-OCRv6_det_medium.pth"
OCR_REC_MODEL = "PP-OCRv6_rec_medium.pth"
OCR_REC_KEYS = "ppocrv6_dict.txt"
DET_MAX_SIDE_LEN = 960
REC_MAX_BATCH = 6
REC_MAX_WIDTH = 1024
WORKSPACE_MB = 1024
PRECISION = "fp16"
VERIFY = True
FORCE = False

VALID_TARGETS = frozenset({"all", "yolo", "ocr"})
VALID_PRECISIONS = frozenset({"fp16", "fp32"})

_DET_MIN_SIDE_LEN = 736
_REC_IMAGE_HEIGHT = 48
_ABS_PATH_RE = re.compile(r"^(?:[A-Za-z]:[\\/]|\\\\|/)")


@dataclass(frozen=True, slots=True)
class BuildTrtConfig:
    """TensorRT 离线引擎构建参数。"""

    target: str = TARGET
    out_dir: Path = Path(OUT_DIR)
    device: str = DEVICE
    yolo_model: Path = Path(YOLO_MODEL)
    imgsz: int = IMGSZ
    yolo_input_hw: tuple[int, int] = YOLO_INPUT_HW
    ocr_models_dir: Path = Path(OCR_MODELS_DIR)
    ocr_det_model: str = OCR_DET_MODEL
    ocr_rec_model: str = OCR_REC_MODEL
    ocr_rec_keys: str = OCR_REC_KEYS
    det_max_side_len: int = DET_MAX_SIDE_LEN
    rec_max_batch: int = REC_MAX_BATCH
    rec_max_width: int = REC_MAX_WIDTH
    workspace_mb: int = WORKSPACE_MB
    precision: str = PRECISION
    verify: bool = VERIFY
    force: bool = FORCE

    def validated(self) -> BuildTrtConfig:  # noqa: PLR0912 - 参数逐项校验便于定位失败字段
        """校验目标范围、精度、单一 CUDA 设备、输入尺寸与源权重文件名约束。"""
        from airdrop.perception.trt_runtime import parse_cuda_device_id

        if self.target not in VALID_TARGETS:
            raise ValueError(f"target 只支持 {sorted(VALID_TARGETS)}: {self.target!r}")
        if self.precision not in VALID_PRECISIONS:
            raise ValueError(f"precision 只支持 {sorted(VALID_PRECISIONS)}: {self.precision!r}")
        parse_cuda_device_id(self.device)
        if not str(self.out_dir).strip():
            raise ValueError("out_dir 不能为空")
        if type(self.imgsz) is not int or self.imgsz < 32:
            raise ValueError(f"imgsz 必须是 >= 32 的整数: {self.imgsz!r}")
        if (
            not isinstance(self.yolo_input_hw, tuple)
            or len(self.yolo_input_hw) != 2
            or any(type(dim) is not int or dim < 32 or dim % 32 != 0 for dim in self.yolo_input_hw)
        ):
            raise ValueError(
                f"yolo_input_hw 必须是两个 >= 32 且为 32 倍数的整数 (H, W): {self.yolo_input_hw!r}"
            )
        if (
            type(self.det_max_side_len) is not int
            or self.det_max_side_len < _DET_MIN_SIDE_LEN
            or self.det_max_side_len % 32 != 0
        ):
            raise ValueError(
                f"det_max_side_len 必须是 >= {_DET_MIN_SIDE_LEN} 且为 32 倍数的整数: "
                f"{self.det_max_side_len!r}"
            )
        if type(self.rec_max_batch) is not int or self.rec_max_batch < 1:
            raise ValueError(f"rec_max_batch 必须是至少为 1 的整数: {self.rec_max_batch!r}")
        if type(self.rec_max_width) is not int or self.rec_max_width < 32:
            raise ValueError(f"rec_max_width 必须是 >= 32 的整数: {self.rec_max_width!r}")
        if type(self.workspace_mb) is not int or self.workspace_mb < 1:
            raise ValueError(f"workspace_mb 必须是正整数: {self.workspace_mb!r}")
        if type(self.verify) is not bool:
            raise ValueError(f"verify 必须是 bool: {self.verify!r}")
        if type(self.force) is not bool:
            raise ValueError(f"force 必须是 bool: {self.force!r}")

        if self.target in ("all", "yolo"):
            if not str(self.yolo_model).strip():
                raise ValueError("yolo_model 不能为空")
            if self.yolo_model.suffix.lower() != ".pt":
                raise ValueError(f"YOLO 源权重必须为 .pt 文件: {self.yolo_model!r}")

        if self.target in ("all", "ocr"):
            if not str(self.ocr_models_dir).strip():
                raise ValueError("ocr_models_dir 不能为空")
            for field_name, filename in (
                ("ocr_det_model", self.ocr_det_model),
                ("ocr_rec_model", self.ocr_rec_model),
            ):
                if not isinstance(filename, str) or not filename.strip():
                    raise ValueError(f"{field_name} 不能为空")
                if Path(filename).name != filename or Path(filename).suffix.lower() != ".pth":
                    raise ValueError(f"{field_name} 必须是纯 .pth 文件名: {filename!r}")
            if (
                not isinstance(self.ocr_rec_keys, str)
                or not self.ocr_rec_keys.strip()
                or Path(self.ocr_rec_keys).name != self.ocr_rec_keys
            ):
                raise ValueError(f"ocr_rec_keys 必须是纯文件名: {self.ocr_rec_keys!r}")
        return self


def build_config(**overrides: Any) -> BuildTrtConfig:
    """按关键字覆盖派生并校验 :class:`BuildTrtConfig`。"""
    normalized: dict[str, Any] = dict(overrides)
    for key in ("out_dir", "yolo_model", "ocr_models_dir"):
        if key in normalized:
            value = normalized[key]
            if isinstance(value, str):
                if not value.strip():
                    raise ValueError(f"{key} 不能为空")
                normalized[key] = Path(value)
            elif not isinstance(value, Path):
                raise ValueError(f"{key} 必须是路径或字符串: {value!r}")
    if "yolo_input_hw" in normalized:
        hw = normalized["yolo_input_hw"]
        if isinstance(hw, list) and len(hw) == 2:
            normalized["yolo_input_hw"] = (hw[0], hw[1])
    return replace(BuildTrtConfig(), **normalized).validated()


def yolo_engine_path(cfg: BuildTrtConfig) -> Path:
    """返回 YOLO TensorRT 引擎的规范输出路径。"""
    in_h, in_w = cfg.yolo_input_hw
    return cfg.out_dir / f"{Path(cfg.yolo_model).stem}_{in_h}x{in_w}_{cfg.precision}.engine"


def ocr_det_engine_path(cfg: BuildTrtConfig) -> Path:
    """返回 OCR 文本检测 TensorRT 引擎的规范输出路径。"""
    return cfg.out_dir / f"{Path(cfg.ocr_det_model).stem}_{cfg.precision}.engine"


def ocr_rec_engine_path(cfg: BuildTrtConfig) -> Path:
    """返回 OCR 文本识别 TensorRT 引擎的规范输出路径。"""
    return cfg.out_dir / f"{Path(cfg.ocr_rec_model).stem}_{cfg.precision}.engine"


def _validate_manifest_no_absolute_paths(value: Any, path: str = "manifest") -> None:
    """递归检查伴随清单内容，禁止出现任何本机绝对路径。"""
    if isinstance(value, str):
        if _ABS_PATH_RE.match(value):
            raise ValueError(f"伴随清单字段 {path} 禁止包含绝对路径: {value!r}")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _validate_manifest_no_absolute_paths(item, f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _validate_manifest_no_absolute_paths(item, f"{path}[{index}]")


def atomic_publish_engine(
    engine_bytes: bytes,
    manifest_data: dict[str, Any],
    final_engine_path: Path,
    *,
    force: bool = False,
) -> tuple[Path, Path]:
    """原子发布 TensorRT 引擎文件及伴随清单。

    参数
    ----
    engine_bytes:
        已序列化并通过校验的 TensorRT 引擎字节串。
    manifest_data:
        伴随清单字典（仅允许文件名与摘要，禁止包含本机绝对路径）。
    final_engine_path:
        最终引擎输出路径；伴随清单路径由 ``manifest_path_for`` 推导。
    force:
        为 ``False`` 时拒绝覆盖已存在的引擎或伴随清单。

    返回
    ----
    tuple[Path, Path]
        最终发布后的 ``(engine_path, manifest_path)``。
    """
    from airdrop.perception.trt_runtime import manifest_path_for

    if not isinstance(engine_bytes, (bytes, bytearray)) or not engine_bytes:
        raise ValueError("待发布的 TensorRT 引擎字节不能为空")
    final_engine = Path(final_engine_path)
    final_manifest = manifest_path_for(final_engine)
    if not force and (final_engine.exists() or final_manifest.exists()):
        raise FileExistsError(
            f"目标引擎或伴随清单已存在（如需覆盖请传入 --force / force=True）: {final_engine}"
        )
    _validate_manifest_no_absolute_paths(manifest_data)
    declared_engine = manifest_data.get("engine_file")
    if isinstance(declared_engine, str) and (
        Path(declared_engine).name != declared_engine or declared_engine != final_engine.name
    ):
        raise ValueError(
            f"伴随清单 engine_file 必须为目标文件名 {final_engine.name!r}，实际 {declared_engine!r}"
        )

    final_engine.parent.mkdir(parents=True, exist_ok=True)
    pid = os.getpid()
    tmp_engine = final_engine.with_name(f".{final_engine.name}.tmp.{pid}")
    tmp_manifest = final_manifest.with_name(f".{final_manifest.name}.tmp.{pid}")
    manifest_bytes = (json.dumps(manifest_data, ensure_ascii=False, indent=2) + "\n").encode(
        "utf-8"
    )
    try:
        with tmp_engine.open("wb") as stream:
            stream.write(engine_bytes)
            stream.flush()
            os.fsync(stream.fileno())
        with tmp_manifest.open("wb") as stream:
            stream.write(manifest_bytes)
            stream.flush()
            os.fsync(stream.fileno())
        # 先替换引擎再替换清单：中断时加载端会因清单缺失或 SHA256 不匹配明确拒绝。
        os.replace(tmp_engine, final_engine)
        os.replace(tmp_manifest, final_manifest)
    except BaseException:
        with contextlib.suppress(OSError):
            tmp_engine.unlink(missing_ok=True)
        with contextlib.suppress(OSError):
            tmp_manifest.unlink(missing_ok=True)
        raise
    return final_engine, final_manifest


def _is_strongly_typed_trt(trt_module: Any) -> bool:
    """判断当前 TensorRT 是否采用强类型构建（TRT 11+ 移除 BuilderFlag.FP16）。"""
    return not hasattr(trt_module.BuilderFlag, "FP16") and hasattr(
        trt_module.NetworkDefinitionCreationFlag, "STRONGLY_TYPED"
    )


def _make_synthetic_input(
    shape: tuple[int, ...],
    *,
    low: float,
    high: float,
    seed: int,
) -> Any:
    """生成确定性的结构化浮点张量，用于导出追踪与 PyTorch/TRT 输出比对。"""
    import numpy as np

    rng = np.random.default_rng(seed)
    batch, channels, height, width = shape
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)[None, None, :, None]
    x = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, None, None, :]
    pattern = 0.5 * (np.sin(2.0 * np.pi * x * 2.0) * np.cos(2.0 * np.pi * y * 2.0) + 1.0)
    pattern = np.broadcast_to(pattern, (batch, channels, height, width)).copy()
    noise = rng.uniform(0.0, 1.0, size=shape).astype(np.float32)
    mixed = 0.7 * pattern + 0.3 * noise
    return np.ascontiguousarray((low + (high - low) * mixed).astype(np.float32))


def _reset_yolo_head_cache(model: Any) -> None:
    """清空 Detect 头的缓存形状，避免 FP32 anchor 缓存残留到 FP16 ONNX 导出。"""
    for module in model.modules():
        if hasattr(module, "anchors") and hasattr(module, "strides") and hasattr(module, "shape"):
            module.shape = None


def _wrap_single_output(inner_model: Any) -> Any:
    """包装 PyTorch 模型使其固定返回单一张量输出。"""
    import torch

    class _SingleOutputWrapper(torch.nn.Module):
        def __init__(self, inner: Any) -> None:
            super().__init__()
            self.inner = inner

        def forward(self, x: Any) -> Any:
            y = self.inner(x)
            return y[0] if isinstance(y, (tuple, list)) else y

    return _SingleOutputWrapper(inner_model)


def _compute_reference_cases(
    model: Any,
    *,
    shapes: list[tuple[int, ...]],
    device_id: int,
    value_range: tuple[float, float],
) -> list[tuple[tuple[int, ...], Any, Any]]:
    """在切换到半精度导出前用 FP32 PyTorch 模型计算代表性形状的参考输出。"""
    import numpy as np
    import torch

    from airdrop.perception.trt_runtime import TrtEngineError

    device = torch.device(f"cuda:{device_id}")
    _reset_yolo_head_cache(model)
    model = model.float().eval().to(device)
    cases: list[tuple[tuple[int, ...], Any, Any]] = []
    with torch.no_grad():
        for index, shape in enumerate(shapes):
            inp_np = _make_synthetic_input(
                shape, low=value_range[0], high=value_range[1], seed=2026 + index
            )
            inp_t = torch.from_numpy(inp_np).to(device=device, dtype=torch.float32)
            out_t = model(inp_t)
            if isinstance(out_t, (tuple, list)):
                out_t = out_t[0]
            ref_np = out_t.float().cpu().numpy()
            if not np.all(np.isfinite(ref_np)):
                raise TrtEngineError(f"PyTorch 参考模型在形状 {shape} 下产生了非有限值")
            cases.append((shape, inp_np, ref_np))
    return cases


def _export_onnx(
    model: Any,
    *,
    dummy_shape: tuple[int, ...],
    input_name: str,
    output_name: str,
    dynamic_axes: dict[str, dict[int, str]] | None,
    onnx_path: Path,
    device_id: int,
    export_fp16: bool,
    value_range: tuple[float, float],
) -> None:
    """将 PyTorch 模型导出为 opset 17 ONNX，并执行结构校验。"""
    import onnx
    import torch

    device = torch.device(f"cuda:{device_id}")
    _reset_yolo_head_cache(model)
    model = model.eval().to(device)
    dummy_np = _make_synthetic_input(
        dummy_shape, low=value_range[0], high=value_range[1], seed=1001
    )
    dummy = torch.from_numpy(dummy_np).to(device)
    if export_fp16:
        model = model.half()
        dummy = dummy.half()
    else:
        model = model.float()
        dummy = dummy.float()

    with torch.no_grad():
        torch.onnx.export(
            model,
            (dummy,),
            str(onnx_path),
            opset_version=17,
            dynamo=False,
            input_names=[input_name],
            output_names=[output_name],
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
        )
    onnx.checker.check_model(str(onnx_path))


def _build_serialized_engine(
    onnx_path: Path,
    *,
    precision: str,
    workspace_mb: int,
    device_id: int,
    profiles: dict[str, tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]] | None,
) -> tuple[bytes, bool]:
    """解析 ONNX 并使用 TensorRT Builder 构建序列化引擎字节。"""
    import tensorrt as _trt
    import torch

    from airdrop.perception.trt_runtime import TrtEngineError

    trt: Any = _trt
    torch.cuda.set_device(device_id)
    strongly_typed = _is_strongly_typed_trt(trt)
    use_strongly_typed_fp16 = precision == "fp16" and strongly_typed

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    if builder is None:
        raise TrtEngineError("无法创建 TensorRT Builder 实例")

    if use_strongly_typed_fp16:
        network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    elif hasattr(trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH"):
        network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    else:
        network_flags = 0

    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_path)):
        errors = [str(parser.get_error(i)) for i in range(int(parser.num_errors))]
        raise TrtEngineError(
            f"TensorRT 解析 ONNX 失败 ({onnx_path.name}): {'; '.join(errors) or '未知错误'}"
        )

    builder_config = builder.create_builder_config()
    builder_config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, int(workspace_mb) * (1 << 20)
    )
    if precision == "fp16" and not use_strongly_typed_fp16:
        if hasattr(trt.BuilderFlag, "FP16"):
            builder_config.set_flag(trt.BuilderFlag.FP16)
        else:
            raise TrtEngineError("当前 TensorRT 既无 BuilderFlag.FP16 也无 STRONGLY_TYPED 支持")

    if profiles:
        profile = builder.create_optimization_profile()
        for tensor_name, (min_shape, opt_shape, max_shape) in profiles.items():
            profile.set_shape(tensor_name, min_shape, opt_shape, max_shape)
        builder_config.add_optimization_profile(profile)

    serialized = builder.build_serialized_network(network, builder_config)
    if serialized is None:
        raise TrtEngineError(f"TensorRT 引擎构建返回空结果: {onnx_path.name}")
    return bytes(serialized), use_strongly_typed_fp16


def _introspect_engine_io(
    engine_bytes: bytes,
    *,
    device_id: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """反序列化引擎字节并提取 I/O 张量规格，确保与运行时清单校验一致。"""
    import numpy as np
    import tensorrt as _trt
    import torch

    from airdrop.perception.trt_runtime import TrtEngineError

    trt: Any = _trt
    torch.cuda.set_device(device_id)
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    if runtime is None:
        raise TrtEngineError("无法创建 TensorRT Runtime 实例")
    engine = runtime.deserialize_cuda_engine(engine_bytes)
    if engine is None:
        raise TrtEngineError("无法反序列化已构建的 TensorRT 引擎")

    inputs: list[dict[str, Any]] = []
    outputs: list[dict[str, Any]] = []
    for index in range(int(engine.num_io_tensors)):
        name = str(engine.get_tensor_name(index))
        mode = engine.get_tensor_mode(name)
        dtype = str(np.dtype(trt.nptype(engine.get_tensor_dtype(name))).name)
        shape = [int(dim) for dim in engine.get_tensor_shape(name)]
        spec: dict[str, Any] = {"name": name, "dtype": dtype, "shape": shape}
        if mode == trt.TensorIOMode.INPUT:
            if any(dim < 0 for dim in shape):
                if int(engine.num_optimization_profiles) < 1:
                    raise TrtEngineError(f"动态输入 {name!r} 缺少 optimization profile")
                lo, opt, hi = engine.get_tensor_profile_shape(name, 0)
                spec["min_shape"] = [int(dim) for dim in lo]
                spec["opt_shape"] = [int(dim) for dim in opt]
                spec["max_shape"] = [int(dim) for dim in hi]
            inputs.append(spec)
        elif mode == trt.TensorIOMode.OUTPUT:
            outputs.append(spec)
    if not inputs or not outputs:
        raise TrtEngineError("TensorRT 引擎缺少有效的输入或输出张量")
    return inputs, outputs


def _verify_engine_outputs(
    *,
    engine_bytes: bytes,
    manifest_data: dict[str, Any],
    kind: str,
    precision: str,
    device_id: int,
    reference_cases: list[tuple[tuple[int, ...], Any, Any]],
    temp_dir: Path,
) -> list[dict[str, Any]]:
    """在临时目录加载 TrtExecutionRunner 并对照 PyTorch 参考输出校验形状与数值误差。"""
    import numpy as np

    from airdrop.perception.trt_runtime import TrtEngineError, load_trt_runner

    tmp_engine = temp_dir / str(manifest_data["engine_file"])
    atomic_publish_engine(engine_bytes, manifest_data, tmp_engine, force=True)
    runner = load_trt_runner(tmp_engine, expected_kind=kind, device_id=device_id)
    results: list[dict[str, Any]] = []
    for shape, inp_np, ref_np in reference_cases:
        trt_np = runner.run_numpy(inp_np)
        if trt_np.shape != ref_np.shape:
            raise TrtEngineError(
                f"离线校验失败（{kind}，shape={shape}）：输出形状不匹配，"
                f"TRT={trt_np.shape}，PyTorch={ref_np.shape}"
            )
        if not np.all(np.isfinite(trt_np)):
            raise TrtEngineError(f"离线校验失败（{kind}，shape={shape}）：TRT 输出包含 NaN/Inf")
        abs_diff = np.abs(trt_np.astype(np.float64) - ref_np.astype(np.float64))
        max_abs = float(abs_diff.max()) if abs_diff.size else 0.0
        mean_abs = float(abs_diff.mean()) if abs_diff.size else 0.0
        scale = max(1.0, float(np.max(np.abs(ref_np)))) if ref_np.size else 1.0
        if precision == "fp16":
            max_tol = max(0.1, 0.05 * scale)
            mean_tol = max(0.02, 0.005 * scale)
        else:
            max_tol = max(1e-2, 5e-3 * scale)
            mean_tol = max(1e-3, 1e-3 * scale)
        if max_abs > max_tol or mean_abs > mean_tol:
            raise TrtEngineError(
                f"离线校验失败（{kind}，shape={shape}）："
                f"max_abs={max_abs:.6g}（上限 {max_tol:.6g}），"
                f"mean_abs={mean_abs:.6g}（上限 {mean_tol:.6g}）"
            )
        results.append(
            {
                "shape": list(shape),
                "output_shape": [int(dim) for dim in trt_np.shape],
                "max_abs_diff": max_abs,
                "mean_abs_diff": mean_abs,
                "max_abs_tol": max_tol,
                "mean_abs_tol": mean_tol,
            }
        )
    return results


def _check_not_exists(final_engine_path: Path, *, force: bool) -> None:
    """在耗时的导出与构建之前提前拒绝覆盖已有产物。"""
    from airdrop.perception.trt_runtime import manifest_path_for

    if not force and (final_engine_path.exists() or manifest_path_for(final_engine_path).exists()):
        raise FileExistsError(
            f"目标引擎或伴随清单已存在（如需覆盖请传入 --force / force=True）: {final_engine_path}"
        )


def _export_build_verify_publish(
    *,
    model: Any,
    kind: str,
    final_engine_path: Path,
    source_meta: dict[str, Any],
    model_extra: dict[str, Any],
    input_name: str,
    output_name: str,
    dummy_shape: tuple[int, ...],
    dynamic_axes: dict[str, dict[int, str]] | None,
    profiles: dict[str, tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]] | None,
    verify_shapes: list[tuple[int, ...]],
    value_range: tuple[float, float],
    cfg: BuildTrtConfig,
    device_id: int,
) -> tuple[Path, Path]:
    """执行单个模型的参考计算、ONNX 导出、TensorRT 构建、离线校验与原子发布。"""
    import tensorrt as trt

    from airdrop.perception.trt_runtime import (
        MANIFEST_SCHEMA_VERSION,
        current_cuda_environment,
        sha256_bytes,
        validate_manifest_dict,
    )

    _check_not_exists(final_engine_path, force=cfg.force)
    env_info = current_cuda_environment(device_id)
    strongly_typed = _is_strongly_typed_trt(trt)
    export_fp16 = cfg.precision == "fp16" and strongly_typed

    reference_cases: list[tuple[tuple[int, ...], Any, Any]] = []
    if cfg.verify:
        reference_cases = _compute_reference_cases(
            model,
            shapes=verify_shapes,
            device_id=device_id,
            value_range=value_range,
        )

    with tempfile.TemporaryDirectory(prefix=f"trt_build_{kind}_") as tmp_root:
        temp_dir = Path(tmp_root)
        onnx_path = temp_dir / f"{final_engine_path.stem}.onnx"
        _export_onnx(
            model,
            dummy_shape=dummy_shape,
            input_name=input_name,
            output_name=output_name,
            dynamic_axes=dynamic_axes,
            onnx_path=onnx_path,
            device_id=device_id,
            export_fp16=export_fp16,
            value_range=value_range,
        )
        engine_bytes, used_strongly_typed = _build_serialized_engine(
            onnx_path,
            precision=cfg.precision,
            workspace_mb=cfg.workspace_mb,
            device_id=device_id,
            profiles=profiles,
        )
        inputs, outputs = _introspect_engine_io(engine_bytes, device_id=device_id)
        engine_sha = sha256_bytes(engine_bytes)
        manifest_data: dict[str, Any] = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "kind": kind,
            "engine_file": final_engine_path.name,
            "engine_sha256": engine_sha,
            "built_at_utc": datetime.now(UTC).isoformat(),
            "source": dict(source_meta),
            "model": {
                **model_extra,
                "precision": cfg.precision,
                "strongly_typed": used_strongly_typed,
                "inputs": inputs,
                "outputs": outputs,
            },
            "environment": env_info,
            "verification": {"verified": False, "cases": []},
        }
        if cfg.verify:
            verify_dir = temp_dir / "verify"
            cases = _verify_engine_outputs(
                engine_bytes=engine_bytes,
                manifest_data=manifest_data,
                kind=kind,
                precision=cfg.precision,
                device_id=device_id,
                reference_cases=reference_cases,
                temp_dir=verify_dir,
            )
            manifest_data["verification"] = {"verified": True, "cases": cases}

        validate_manifest_dict(
            manifest_data,
            expected_kind=kind,
            expected_engine_name=final_engine_path.name,
            expected_engine_sha256=engine_sha,
        )
        return atomic_publish_engine(
            engine_bytes,
            manifest_data,
            final_engine_path,
            force=cfg.force,
        )


def build_yolo_engine(cfg: BuildTrtConfig) -> tuple[Path, Path]:
    """将 YOLO ``.pt`` 权重离线构建并原子发布为 TensorRT 静态输入引擎。"""
    from airdrop.perception.trt_runtime import parse_cuda_device_id, sha256_file

    cfg = cfg.validated()
    final_path = yolo_engine_path(cfg)
    _check_not_exists(final_path, force=cfg.force)
    weight_path = Path(cfg.yolo_model)
    if not weight_path.is_file():
        raise FileNotFoundError(f"YOLO 源权重不存在: {weight_path}")

    device_id = parse_cuda_device_id(cfg.device)
    from ultralytics import YOLO

    yolo: Any = YOLO(str(weight_path))
    raw_model: Any = yolo.model
    if hasattr(raw_model, "fuse"):
        raw_model.fuse()
    raw_names = getattr(yolo, "names", None) or getattr(raw_model, "names", None) or {}
    names = {str(int(key)): str(value) for key, value in dict(raw_names).items()}
    end2end = bool(
        getattr(raw_model, "end2end", False)
        or any(bool(getattr(module, "end2end", False)) for module in raw_model.modules())
    )
    in_h, in_w = cfg.yolo_input_hw
    wrapped = _wrap_single_output(raw_model)

    return _export_build_verify_publish(
        model=wrapped,
        kind="yolo",
        final_engine_path=final_path,
        source_meta={
            "weight_file": weight_path.name,
            "weight_sha256": sha256_file(weight_path),
        },
        model_extra={
            "yolo": {
                "imgsz": cfg.imgsz,
                "input_hw": [in_h, in_w],
                "class_count": len(names),
                "names": names,
                "end2end": end2end,
            },
        },
        input_name="images",
        output_name="output0",
        dummy_shape=(1, 3, in_h, in_w),
        dynamic_axes=None,
        profiles=None,
        verify_shapes=[(1, 3, in_h, in_w)],
        value_range=(0.0, 1.0),
        cfg=cfg,
        device_id=device_id,
    )


def _load_ocr_torch_models(cfg: BuildTrtConfig, device_id: int) -> tuple[Any, Any]:
    """加载 PP-OCR det 与 rec 的 PyTorch 模型；优先直接经 RapidOCR ModelLoader 离线加载。"""
    import torch

    det_path = cfg.ocr_models_dir / cfg.ocr_det_model
    rec_path = cfg.ocr_models_dir / cfg.ocr_rec_model
    keys_path = cfg.ocr_models_dir / cfg.ocr_rec_keys
    missing = [str(path) for path in (det_path, rec_path, keys_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"OCR 源权重或字典缺失: {', '.join(missing)}")

    device = torch.device(f"cuda:{device_id}")
    try:
        from rapidocr.inference_engine.pytorch.networks.main import ModelLoader

        det_model = ModelLoader({"model_path": str(det_path)}, device).predictor
        rec_model = ModelLoader({"model_path": str(rec_path)}, device).predictor
        return det_model, rec_model
    except ImportError:
        from airdrop.perception.cropproc import OcrEngine, OcrEngineConfig

        engine = OcrEngine(
            OcrEngineConfig(
                model_dir=str(cfg.ocr_models_dir),
                det_model=cfg.ocr_det_model,
                rec_model=cfg.ocr_rec_model,
                rec_keys=cfg.ocr_rec_keys,
                cls_engine="off",
                use_cuda=True,
                device_id=device_id,
                det_max_side_len=cfg.det_max_side_len,
                rec_batch_num=cfg.rec_max_batch,
            )
        ).ensure()
        return engine.text_det.session.predictor, engine.text_rec.session.predictor


def _dedupe_shapes(shapes: list[tuple[int, ...]]) -> list[tuple[int, ...]]:
    seen: set[tuple[int, ...]] = set()
    ordered: list[tuple[int, ...]] = []
    for shape in shapes:
        if shape not in seen:
            seen.add(shape)
            ordered.append(shape)
    return ordered


def build_ocr_engines(cfg: BuildTrtConfig) -> tuple[tuple[Path, Path], tuple[Path, Path]]:
    """将 PP-OCR det 与 rec ``.pth`` 权重离线构建并原子发布为动态形状 TensorRT 引擎。"""
    from airdrop.perception.trt_runtime import parse_cuda_device_id, sha256_file

    cfg = cfg.validated()
    det_final = ocr_det_engine_path(cfg)
    rec_final = ocr_rec_engine_path(cfg)
    _check_not_exists(det_final, force=cfg.force)
    _check_not_exists(rec_final, force=cfg.force)

    det_src = cfg.ocr_models_dir / cfg.ocr_det_model
    rec_src = cfg.ocr_models_dir / cfg.ocr_rec_model
    keys_src = cfg.ocr_models_dir / cfg.ocr_rec_keys
    device_id = parse_cuda_device_id(cfg.device)
    det_model, rec_model = _load_ocr_torch_models(cfg, device_id)

    det_min = (1, 3, 32, 32)
    det_opt = (1, 3, _DET_MIN_SIDE_LEN, min(800, cfg.det_max_side_len))
    det_max = (1, 3, cfg.det_max_side_len, cfg.det_max_side_len)
    det_verify_shapes = _dedupe_shapes([(1, 3, 64, 64), det_opt, det_max])

    det_published = _export_build_verify_publish(
        model=_wrap_single_output(det_model),
        kind="ocr_det",
        final_engine_path=det_final,
        source_meta={
            "weight_file": det_src.name,
            "weight_sha256": sha256_file(det_src),
        },
        model_extra={
            "ocr_det": {
                "limit_type": "min",
                "limit_side_len": _DET_MIN_SIDE_LEN,
                "det_max_side_len": cfg.det_max_side_len,
                "mean": [0.5, 0.5, 0.5],
                "std": [0.5, 0.5, 0.5],
                "thresh": 0.3,
                "box_thresh": 0.5,
                "max_candidates": 1000,
                "unclip_ratio": 1.6,
                "use_dilation": True,
                "score_mode": "fast",
            },
        },
        input_name="x",
        output_name="y",
        dummy_shape=det_opt,
        dynamic_axes={"x": {2: "h", 3: "w"}, "y": {2: "h", 3: "w"}},
        profiles={"x": (det_min, det_opt, det_max)},
        verify_shapes=det_verify_shapes,
        value_range=(-1.0, 1.0),
        cfg=cfg,
        device_id=device_id,
    )

    rec_min = (1, 3, _REC_IMAGE_HEIGHT, 32)
    rec_opt = (min(2, cfg.rec_max_batch), 3, _REC_IMAGE_HEIGHT, min(320, cfg.rec_max_width))
    rec_max = (cfg.rec_max_batch, 3, _REC_IMAGE_HEIGHT, cfg.rec_max_width)
    rec_small = (1, 3, _REC_IMAGE_HEIGHT, min(64, cfg.rec_max_width))
    rec_verify_shapes = _dedupe_shapes([rec_small, rec_opt, rec_max])

    rec_published = _export_build_verify_publish(
        model=_wrap_single_output(rec_model),
        kind="ocr_rec",
        final_engine_path=rec_final,
        source_meta={
            "weight_file": rec_src.name,
            "weight_sha256": sha256_file(rec_src),
            "dict_file": keys_src.name,
            "dict_sha256": sha256_file(keys_src),
        },
        model_extra={
            "ocr_rec": {
                "rec_image_shape": [3, _REC_IMAGE_HEIGHT, 320],
                "rec_max_batch": cfg.rec_max_batch,
                "rec_max_width": cfg.rec_max_width,
            },
        },
        input_name="x",
        output_name="y",
        dummy_shape=rec_opt,
        dynamic_axes={"x": {0: "b", 3: "w"}, "y": {0: "b", 1: "t"}},
        profiles={"x": (rec_min, rec_opt, rec_max)},
        verify_shapes=rec_verify_shapes,
        value_range=(-1.0, 1.0),
        cfg=cfg,
        device_id=device_id,
    )
    return det_published, rec_published


def main(**overrides: Any) -> int:
    """按参数构建所选 TensorRT 引擎并原子发布；返回进程退出码。"""
    cfg = build_config(**overrides)
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )

    planned: list[Path] = []
    if cfg.target in ("all", "yolo"):
        planned.append(yolo_engine_path(cfg))
    if cfg.target in ("all", "ocr"):
        planned.append(ocr_det_engine_path(cfg))
        planned.append(ocr_rec_engine_path(cfg))
    for path in planned:
        _check_not_exists(path, force=cfg.force)

    published: list[tuple[Path, Path]] = []
    if cfg.target in ("all", "yolo"):
        published.append(build_yolo_engine(cfg))
    if cfg.target in ("all", "ocr"):
        det_pair, rec_pair = build_ocr_engines(cfg)
        published.extend([det_pair, rec_pair])

    for engine_path, manifest_path in published:
        LOGGER.info("已发布 TensorRT 引擎：%s（清单：%s）", engine_path, manifest_path.name)
    return 0
