"""YOLO 检测：把一帧画面变成若干 :class:`Detection`。

设计要点
--------
* **双后端按模型后缀分发**：``.pt`` 走 Ultralytics YOLO，``.engine`` 走
  TensorRT 离线引擎（基于 :mod:`~airdrop.perception.trt_runtime` 与纯 PyTorch /
  ``torchvision.ops.nms`` 后处理，完全不导入 ``ultralytics``）。
* **device 通过配置显式指定**，避免部署环境的自动设备选择变化造成不可预期的性能差异。
* **预计算 remap 去畸变**：标定文件给出内参与畸变系数后，在
  :meth:`Detector.prepare_undistort` 里计算映射表，之后每帧复用映射表校正图像。
* **帧尺寸与标定不一致就拒绝检测**：仅跳过去畸变仍会让下游使用错误内参。
  检查在模型加载/推理前完成；尺寸错误抛异常，由入口终止本次运行。
* 重依赖（``ultralytics`` / ``torch`` / ``torchvision`` / ``tensorrt``）**惰性导入**：
  本模块要被离线测试导入，不能因为环境里没有 GPU 栈就连 ``import airdrop.perception`` 都失败。
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np

import airdrop.perception.timing as timing  # noqa: PLR0402 - 避免惰性包循环导入

from ..telemetry.models import TelemetrySnapshot
from . import trt_runtime
from .models import Detection, PixelBox
from .trt_runtime import TRT_BUILD_HINT, TrtEngineError, TrtExecutionRunner, parse_cuda_device_id

LOGGER = logging.getLogger(__name__)

__all__ = ["DetectionBatch", "Detector", "DetectorConfig"]

_LETTERBOX_BORDER_BGR = (114, 114, 114)
_MULTICLASS_NMS_OFFSET = 7680.0


class FrameSizeMismatchError(ValueError):
    """输入尺寸与相机模型不匹配，必须修正配置后重新启动。"""


@dataclass(frozen=True, slots=True)
class _TrtPredictResult:
    """TensorRT YOLO 单帧后处理结果（已一次性拷回主机端 NumPy 数组）。"""

    #: 形状 ``(K, 6)`` 的 ``float32`` 数组，列依次为 ``[x1, y1, x2, y2, conf, cls_id]``。
    detections_np: np.ndarray
    speed: dict[str, float] | None = None


def _letterbox_bgr(
    image: np.ndarray,
    input_hw: tuple[int, int],
) -> tuple[np.ndarray, float, float, float]:
    """将 BGR 图像按固定目标尺寸做等比缩放与灰边填充（对齐 Ultralytics ``LetterBox(auto=False)``）。

    返回 ``(letterbox_image, gain, pad_x, pad_y)``，坐标逆变换为
    ``(x - pad_x) / gain``、``(y - pad_y) / gain``。
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"YOLO 输入必须为 3 通道 BGR 图像，实际形状: {image.shape}")
    h, w = int(image.shape[0]), int(image.shape[1])
    if h <= 0 or w <= 0:
        raise ValueError(f"YOLO 输入图像尺寸无效: {image.shape}")
    in_h, in_w = input_hw
    if h == 720 and w == 1280 and in_h == 736 and in_w == 1280:
        lb = cv2.copyMakeBorder(
            image,
            8,
            8,
            0,
            0,
            cv2.BORDER_CONSTANT,
            value=_LETTERBOX_BORDER_BGR,
        )
        return lb, 1.0, 0.0, 8.0

    gain = min(in_h / h, in_w / w)
    nh = min(in_h, max(1, int(round(h * gain))))
    nw = min(in_w, max(1, int(round(w * gain))))
    resized = (
        cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR) if (w, h) != (nw, nh) else image
    )
    pad_x = (in_w - nw) / 2.0
    pad_y = (in_h - nh) / 2.0
    top = int(round(pad_y - 0.1))
    bottom = int(round(pad_y + 0.1))
    left = int(round(pad_x - 0.1))
    right = int(round(pad_x + 0.1))
    if top or bottom or left or right:
        lb = cv2.copyMakeBorder(
            resized,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=_LETTERBOX_BORDER_BGR,
        )
    else:
        lb = resized
    return lb, gain, pad_x, pad_y


def _parse_yolo_input_hw(raw_hw: Any) -> tuple[int, int]:
    if not isinstance(raw_hw, (list, tuple)) or len(raw_hw) != 2:
        raise TrtEngineError(
            f"TensorRT YOLO 清单 model.yolo.input_hw 必须为 2 元正整数: {raw_hw!r}"
        )
    in_h, in_w = raw_hw
    if type(in_h) is not int or type(in_w) is not int or in_h <= 0 or in_w <= 0:
        raise TrtEngineError(f"TensorRT YOLO 清单 model.yolo.input_hw 必须为正整数: {raw_hw!r}")
    return in_h, in_w


def _parse_yolo_names(raw_names: Any) -> dict[int, str]:
    if not isinstance(raw_names, Mapping) or not raw_names:
        raise TrtEngineError("TensorRT YOLO 清单 model.yolo.names 必须为非空类别映射")
    names: dict[int, str] = {}
    for raw_key, raw_val in raw_names.items():
        if type(raw_key) is int:
            idx = raw_key
        elif isinstance(raw_key, str) and raw_key.isascii() and raw_key.isdigit():
            idx = int(raw_key)
        else:
            raise TrtEngineError(f"TensorRT YOLO 清单 model.yolo.names 类别索引无效: {raw_key!r}")
        if idx < 0:
            raise TrtEngineError(f"TensorRT YOLO 清单 model.yolo.names 类别索引不能为负: {idx}")
        if idx in names:
            raise TrtEngineError(f"TensorRT YOLO 清单 model.yolo.names 存在重复类别索引: {idx}")
        if not isinstance(raw_val, str):
            raise TrtEngineError(
                f"TensorRT YOLO 清单 model.yolo.names[{idx}] 必须为字符串: {raw_val!r}"
            )
        names[idx] = raw_val
    return names


def _validate_yolo_trt_manifest(
    runner: TrtExecutionRunner,
    *,
    expected_imgsz: int,
) -> tuple[tuple[int, int], dict[int, str]]:
    """校验 YOLO TensorRT 清单规格与引擎 I/O 张量契约。"""
    yolo_meta = runner.manifest.model.get("yolo")
    if not isinstance(yolo_meta, Mapping):
        raise TrtEngineError("TensorRT YOLO 清单缺少有效的 model.yolo 字段")

    end2end = yolo_meta.get("end2end", False)
    if not isinstance(end2end, bool):
        raise TrtEngineError(f"TensorRT YOLO 清单 model.yolo.end2end 必须为布尔值: {end2end!r}")
    if end2end:
        raise TrtEngineError("当前 TensorRT YOLO 后端暂不支持 end2end=True 引擎")

    in_h, in_w = _parse_yolo_input_hw(yolo_meta.get("input_hw"))
    if max(in_h, in_w) != expected_imgsz:
        raise TrtEngineError(
            f"TensorRT YOLO 引擎输入尺寸 {in_h}×{in_w}（长边 {max(in_h, in_w)}）"
            f"与配置 imgsz={expected_imgsz} 不一致"
        )
    if runner.input_spec.shape != (1, 3, in_h, in_w):
        raise TrtEngineError(
            f"TensorRT YOLO 引擎输入张量形状 {runner.input_spec.shape} "
            f"与 model.yolo.input_hw={(in_h, in_w)} 不一致"
        )

    names = _parse_yolo_names(yolo_meta.get("names"))
    out_shape = runner.output_spec.shape
    if len(out_shape) != 3 or out_shape[0] != 1 or out_shape[1] <= 4 or out_shape[2] <= 0:
        raise TrtEngineError(
            f"TensorRT YOLO 引擎输出张量形状无效: {out_shape}，期望 (1, 4 + nc, N)"
        )
    if out_shape[1] - 4 != len(names):
        raise TrtEngineError(
            f"TensorRT YOLO 引擎输出类别数 ({out_shape[1] - 4}) "
            f"与清单 model.yolo.names 数量 ({len(names)}) 不一致"
        )
    return (in_h, in_w), names


class _TrtYoloModel:
    """TensorRT YOLO 推理封装（纯 PyTorch + torchvision NMS，不依赖 ultralytics）。

    有意保留普通实例字典，以便离线基准工具临时包裹 ``model.predict`` 计时。
    """

    def __init__(
        self,
        *,
        runner: TrtExecutionRunner,
        input_hw: tuple[int, int],
        names: dict[int, str],
    ) -> None:
        import torch
        from torchvision.ops import nms as tv_nms

        self.runner = runner
        self.input_hw = input_hw
        self.names = dict(names)
        self._torch = torch
        self._nms = tv_nms

    def predict(
        self,
        image: np.ndarray,
        *,
        conf: float = 0.25,
        iou: float = 0.45,
        max_det: int = 300,
        **unused: Any,
    ) -> list[_TrtPredictResult]:
        """执行预处理、TensorRT 前向推理、设备端 NMS 与单次 D2H 坐标反算。"""
        del unused
        orig_h, orig_w = int(image.shape[0]), int(image.shape[1])
        lb, gain, pad_x, pad_y = _letterbox_bgr(image, self.input_hw)
        if not lb.flags.c_contiguous or not lb.flags.writeable:
            lb = np.array(lb, copy=True, order="C")

        torch = self._torch
        with torch.no_grad():
            tensor = torch.from_numpy(lb).to(device=self.runner.device)
            # BGR -> RGB，HWC -> NCHW，转换至引擎输入精度并归一化到 [0, 1]
            inp = (
                tensor.flip(2)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .to(dtype=self.runner.input_torch_dtype, memory_format=torch.contiguous_format)
            )
            inp.mul_(1.0 / 255.0)

            raw = self.runner.run_tensor(inp)
            if max_det <= 0:
                return [_TrtPredictResult(detections_np=np.empty((0, 6), dtype=np.float32))]

            pred = raw[0].float().transpose(0, 1)
            boxes_cxcywh = pred[:, :4]
            cls_scores = pred[:, 4:]
            scores, cls_ids = cls_scores.max(dim=1)

            cand = torch.nonzero(scores > conf, as_tuple=False).squeeze(1)
            if cand.numel() == 0:
                return [_TrtPredictResult(detections_np=np.empty((0, 6), dtype=np.float32))]

            boxes_cxcywh = boxes_cxcywh[cand]
            scores = scores[cand]
            cls_ids = cls_ids[cand]

            half_w = boxes_cxcywh[:, 2] * 0.5
            half_h = boxes_cxcywh[:, 3] * 0.5
            boxes_xyxy = torch.stack(
                (
                    boxes_cxcywh[:, 0] - half_w,
                    boxes_cxcywh[:, 1] - half_h,
                    boxes_cxcywh[:, 0] + half_w,
                    boxes_cxcywh[:, 1] + half_h,
                ),
                dim=1,
            )
            offsets = cls_ids.unsqueeze(1).float() * _MULTICLASS_NMS_OFFSET
            keep = self._nms(boxes_xyxy + offsets, scores, float(iou))[:max_det]
            if keep.numel() == 0:
                return [_TrtPredictResult(detections_np=np.empty((0, 6), dtype=np.float32))]

            packed = torch.cat(
                (
                    boxes_xyxy[keep],
                    scores[keep].unsqueeze(1),
                    cls_ids[keep].unsqueeze(1).float(),
                ),
                dim=1,
            )
            det_np = packed.cpu().numpy()

        det_np[:, 0] = np.clip((det_np[:, 0] - pad_x) / gain, 0.0, float(orig_w))
        det_np[:, 1] = np.clip((det_np[:, 1] - pad_y) / gain, 0.0, float(orig_h))
        det_np[:, 2] = np.clip((det_np[:, 2] - pad_x) / gain, 0.0, float(orig_w))
        det_np[:, 3] = np.clip((det_np[:, 3] - pad_y) / gain, 0.0, float(orig_h))
        return [_TrtPredictResult(detections_np=det_np)]


@dataclass(frozen=True, slots=True)
class DetectorConfig:
    """YOLO 检测参数。"""

    # YOLO 检测权重（默认 TensorRT 静态 FP16 引擎，可覆盖为 .pt 备用权重）
    model_path: str = ".local/trt/best2_736x1280_fp16.engine"
    device: str = "0"  # 显式指定：见模块 docstring
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    imgsz: int = 1280  # 权重按 imgsz=640 训练；实测 640/1280 检出一致
    max_detections: int = 300
    #: CUDA 上是否启用 FP16 推理（对应 Ultralytics ``quantize=16``；CPU/MPS 自动保持 FP32）。
    half: bool = True
    # 目标框外扩比例（裁剪时保留完整轮廓）
    crop_expand_ratio: float = 0.2
    # 去畸变（标定产出；为 None 表示不做）
    camera_matrix: np.ndarray | None = None
    dist_coeffs: np.ndarray | None = None
    undistort: bool = True
    #: 标定时的图像尺寸 (width, height)。给了它，:meth:`prepare_undistort` 就只在
    #: 这个分辨率上建映射表：内参只对标定分辨率有效，换了尺寸硬套会把画面扭坏。
    #: 给了它而尺寸对不上时**拒绝建表**并告警，而不是"按首帧尺寸默默建一张"。
    calib_size: tuple[int, int] | None = None


@dataclass(frozen=True, slots=True)
class DetectionBatch:
    """一帧的检测输出：画面尺寸 + 各目标。

    ``image`` 是**已去畸变**的画面（若启用了去畸变）——下游裁剪必须用同一张，
    否则像素坐标与解算所用的内参对不上。``owns_image`` 为 ``True`` 表示
    ``image`` 由本次检测新分配且独占，无外部可变别名。
    """

    image: np.ndarray
    detections: tuple[Detection, ...]
    undistorted: bool = False
    owns_image: bool = False


class Detector:
    """YOLO 封装：惰性加载权重、可选去畸变、输出 :class:`Detection`。

    ``Detector`` 自身**不碰遥测**：调用方把该帧拍摄时刻的
    :class:`TelemetrySnapshot` 传进来，检测结果就带上它。这样 detector 既能在
    实时链路上用，也能在离线回放里用（回放时遥测来自日志）。
    """

    def __init__(self, config: DetectorConfig | None = None) -> None:
        self._config = config or DetectorConfig()
        self._load_lock = threading.Lock()
        self._model: Any = None
        self._names: dict[int, str] = {}
        self._remap: tuple[np.ndarray, np.ndarray] | None = None
        self._remap_shape: tuple[int, int] | None = None
        self._undistort_skipped_logged = False
        self._calib_size_rejected_logged = False

    # ------------------------------------------------------------------
    # 状态
    # ------------------------------------------------------------------
    @property
    def config(self) -> DetectorConfig:
        return self._config

    @property
    def loaded(self) -> bool:
        """权重是否已加载（未加载时 :meth:`detect` 会自动加载）。"""
        with self._load_lock:
            return self._model is not None

    @property
    def names(self) -> dict[int, str]:
        """类别索引 → 名称（加载后可用）。"""
        with self._load_lock:
            return dict(self._names)

    @property
    def class_count(self) -> int:
        with self._load_lock:
            return len(self._names)

    # ------------------------------------------------------------------
    # 加载与预热
    # ------------------------------------------------------------------
    def load(self) -> "Detector":
        """加载模型（幂等）。按后缀分发 ``.pt``（Ultralytics）或 ``.engine``（TensorRT）。"""
        with self._load_lock:
            if self._model is not None:
                return self
            path = Path(self._config.model_path)
            suffix = path.suffix.lower()
            if suffix not in (".pt", ".engine"):
                raise ValueError(
                    f"不支持的 YOLO 模型后缀 {suffix or '<无后缀>'!r}（仅支持 .pt 或 .engine）: {path}"
                )
            if not path.is_file():
                if suffix == ".engine":
                    raise FileNotFoundError(
                        f"YOLO TensorRT 引擎权重不存在: {path.resolve()}（PerceptionConfig.model_path）；"
                        f"{TRT_BUILD_HINT}"
                    )
                raise FileNotFoundError(
                    f"YOLO 权重不存在: {path.resolve()}（PerceptionConfig.model_path）"
                )
            if suffix == ".engine":
                model, names = self._load_trt_engine(path)
            else:
                model, names = self._load_ultralytics_pt(path)
            self._names = names
            self._model = model
            LOGGER.info("权重类别 %d 个: %s", len(names), names)
            return self

    def _load_ultralytics_pt(self, path: Path) -> tuple[Any, dict[int, str]]:
        from ultralytics import YOLO  # 惰性：重型依赖

        LOGGER.info("加载 YOLO 权重 %s（device=%s）", path, self._config.device)
        model = YOLO(str(path))
        model_names = getattr(model, "names", None) or {}
        names = {int(key): str(value) for key, value in dict(model_names).items()}
        return model, names

    def _load_trt_engine(self, path: Path) -> tuple[_TrtYoloModel, dict[int, str]]:
        device_id = parse_cuda_device_id(self._config.device)
        LOGGER.info("加载 TensorRT YOLO 引擎 %s（cuda:%d）", path, device_id)
        runner = trt_runtime.load_trt_runner(
            path,
            expected_kind="yolo",
            device_id=device_id,
        )
        input_hw, names = _validate_yolo_trt_manifest(runner, expected_imgsz=self._config.imgsz)
        model = _TrtYoloModel(runner=runner, input_hw=input_hw, names=names)
        return model, names

    @staticmethod
    def _is_cuda_device(device: str) -> bool:
        normalized = str(device).strip().lower()
        if not normalized or normalized in ("cpu", "mps"):
            return False
        if normalized.startswith("cuda"):
            return True
        return all(part.strip().isdigit() for part in normalized.split(","))

    def _predict_kwargs(self) -> dict[str, Any]:
        config = self._config
        kwargs: dict[str, Any] = {
            "device": config.device,
            "conf": config.conf_threshold,
            "iou": config.iou_threshold,
            "imgsz": config.imgsz,
            "max_det": config.max_detections,
            "verbose": False,
        }
        if config.half and self._is_cuda_device(config.device):
            kwargs["quantize"] = 16
        return kwargs

    def warmup(self, width: int, height: int) -> bool:
        """按目标分辨率跑一次空推理，把 CUDA 上下文/内核编译开销挪到起飞前。

        起飞前预热可把首次推理初始化开销移出任务航线。底层失败只记录日志并返回 ``False``，
        任务会话必须据此中止启动，不能把权重加载成功当作预热成功。
        """
        try:
            self.load()
            blank = np.zeros((height, width, 3), np.uint8)
            self._model.predict(blank, **self._predict_kwargs())
            return True
        except Exception:
            LOGGER.exception("YOLO 预热失败")
            return False

    # ------------------------------------------------------------------
    # 去畸变
    # ------------------------------------------------------------------
    def prepare_undistort(self, width: int, height: int, *, force: bool = False) -> bool:
        """为给定分辨率预计算去畸变映射表（按尺寸缓存）。

        标定内参只在**标定时的那个分辨率**下有效，所以映射表一旦算好就锁在那个
        尺寸上：换尺寸的帧会被 :meth:`undistort` 跳过（并告警），而不是悄悄用
        错误的内参把画面扭坏。真要改分辨率就显式 ``force=True`` 重算。

        返回是否算出了可用的映射表。
        """
        config = self._config
        if not config.undistort or config.camera_matrix is None:
            return False
        size = (int(width), int(height))
        if config.calib_size is not None and tuple(config.calib_size) != size:
            # 每帧都会走到这里（undistort 每次都调 prepare_undistort），所以要节流
            if not self._calib_size_rejected_logged:
                self._calib_size_rejected_logged = True
                LOGGER.warning(
                    "帧尺寸 %dx%d 与标定分辨率 %s 不符，拒绝建立去畸变映射表"
                    "（内参只对标定分辨率有效，硬套会把画面扭坏）；本帧不去畸变。"
                    "请配置与新分辨率匹配的内参和 calib_size",
                    size[0],
                    size[1],
                    tuple(config.calib_size),
                )
            return False
        if self._remap is not None and self._remap_shape == size:
            return True
        if self._remap is not None and not force:
            LOGGER.warning(
                "已有 %s 的去畸变映射表，拒绝为 %s 重算"
                "（标定内参只对单一分辨率有效；确需改分辨率请 force=True）",
                self._remap_shape,
                size,
            )
            return False
        matrix = np.asarray(config.camera_matrix, dtype=np.float64).reshape(3, 3)
        dist = (
            np.zeros(5, dtype=np.float64)
            if config.dist_coeffs is None
            else np.asarray(config.dist_coeffs, dtype=np.float64).reshape(-1)
        )
        # 保留标定 K：下游 georef 用同一 K 反投影，不能只在这里改变焦距/主点。
        # 映射仍只构建一次，逐帧复用定点查表；即使畸变系数为零也保持相同重采样路径以贴合真实相机。
        map_x, map_y = cv2.initUndistortRectifyMap(matrix, dist, None, matrix, size, cv2.CV_16SC2)
        self._remap = (map_x, map_y)
        self._remap_shape = size
        self._undistort_skipped_logged = False
        LOGGER.info("已预计算去畸变映射表 %dx%d", width, height)
        return True

    def undistort(self, image: np.ndarray) -> tuple[np.ndarray, bool]:
        """按需去畸变；返回 ``(画面, 是否真的做了)``。

        帧尺寸与映射表不符时**跳过**并记一次 warning：标定内参只对单一分辨率
        有效，硬套只会把画面弄坏。
        """
        config = self._config
        if not config.undistort or config.camera_matrix is None:
            return image, False
        height, width = image.shape[:2]
        self.prepare_undistort(width, height)
        if self._remap is None or self._remap_shape != (width, height):
            if not self._undistort_skipped_logged:
                self._undistort_skipped_logged = True
                LOGGER.warning(
                    "帧尺寸 %dx%d 与去畸变映射表 %s 不符，本帧跳过去畸变",
                    width,
                    height,
                    self._remap_shape,
                )
            return image, False
        map_x, map_y = self._remap
        return cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR), True

    @staticmethod
    def _iter_result_boxes(
        result: Any,
    ) -> list[tuple[float, float, float, float, float, int]]:
        """将 TensorRT 预传 NumPy 结果或 Ultralytics Results 统一解包为检测行。"""
        if isinstance(result, _TrtPredictResult):
            return [
                (float(x1), float(y1), float(x2), float(y2), float(conf), int(cls))
                for x1, y1, x2, y2, conf, cls in result.detections_np
            ]
        boxes = getattr(result, "boxes", None)
        if boxes is None or not len(boxes):
            return []
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)
        return [
            (float(x1), float(y1), float(x2), float(y2), float(conf), int(cls))
            for (x1, y1, x2, y2), conf, cls in zip(xyxy, confs, classes, strict=True)
        ]

    # ------------------------------------------------------------------
    # 检测
    # ------------------------------------------------------------------
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
        """对一帧做检测，返回 :class:`DetectionBatch`。

        ``mode`` 只作为标签写进结果（``"ocr"`` / ``"cls12"``），detector 不关心
        编号怎么来的——``ocr`` 模式下编号由裁剪图 OCR 补上，``cls12`` 模式下
        直接取类别。已知相机尺寸与输入不符时抛 FrameSizeMismatchError，不加载或运行模型。
        """
        config = self._config
        height, width = image.shape[:2]
        expected_size = config.calib_size or self._remap_shape
        if expected_size is not None and (width, height) != tuple(expected_size):
            raise FrameSizeMismatchError(
                f"帧尺寸 {width}x{height} 与相机模型尺寸 {tuple(expected_size)} 不符，"
                "停止运行；请配置匹配的相机标定后重新启动"
            )
        self.load()
        assert self._model is not None  # load() 已保证

        trace_start = timing.now_ns() if timing.ENABLED else 0
        do_undistort = config.undistort if undistort is None else undistort
        if do_undistort and config.camera_matrix is not None:
            picture, rectified = self.undistort(image)
        else:
            picture, rectified = image, False

        predict_start = timing.now_ns() if timing.ENABLED else 0
        results = self._model.predict(picture, **self._predict_kwargs())
        predict_end = timing.now_ns() if timing.ENABLED else 0
        snapshot = telemetry if telemetry is not None else TelemetrySnapshot()
        height, width = picture.shape[:2]
        detections: list[Detection] = []
        if results:
            for x1, y1, x2, y2, conf, cls in self._iter_result_boxes(results[0]):
                box = PixelBox(x1, y1, x2, y2).clipped(width, height)
                if box.area <= 0:
                    continue
                detections.append(
                    Detection(
                        frame_index=frame_index,
                        capture_timestamp=capture_timestamp,
                        pixel=box.center,
                        box=box,
                        confidence=conf,
                        telemetry=snapshot,
                        # cls12：类别直出编号（1 起）。ocr 模式留空待补。
                        code=cls + 1 if mode == "cls12" else None,
                        mode=mode,
                        # ``undistorted`` 是给下游坐标解算看的：像素在**去过畸变**的
                        # 图上时，georef 不能再纠正一遍（否则纠正两次）。
                        extra={
                            "class_name": self._names.get(cls, ""),
                            "undistorted": bool(rectified),
                            "image_size": (width, height),
                        },
                    )
                )
        detections.sort(key=lambda d: d.confidence, reverse=True)
        if timing.ENABLED:
            timing.emit(
                "detector",
                frame_index=frame_index,
                start_ns=trace_start,
                remap_end_ns=predict_start,
                predict_end_ns=predict_end,
                end_ns=timing.now_ns(),
                ultralytics_speed_ms=getattr(results[0], "speed", None) if results else None,
                detections=len(detections),
            )
        return DetectionBatch(
            image=picture,
            detections=tuple(detections),
            undistorted=rectified,
            owns_image=bool(rectified),
        )

    def crop(self, batch: DetectionBatch, detection: Detection) -> np.ndarray:
        """按检测框裁出目标图（含外扩，保留完整五边形轮廓）。"""
        image, _ = self.crop_with_bounds(batch, detection)
        return image

    def crop_with_bounds(
        self, batch: DetectionBatch, detection: Detection
    ) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        """返回外扩裁切图及其在输入图像中的半开像素范围。"""
        box = detection.box.expanded(self._config.crop_expand_ratio).clipped(
            batch.image.shape[1], batch.image.shape[0]
        )
        x1, y1 = int(box.x1), int(box.y1)
        x2, y2 = int(np.ceil(box.x2)), int(np.ceil(box.y2))
        if x2 <= x1 or y2 <= y1:
            return np.zeros((0, 0, 3), np.uint8), (x1, y1, x2, y2)
        return batch.image[y1:y2, x1:x2].copy(), (x1, y1, x2, y2)

    def replace_config(self, **changes: Any) -> "Detector":
        """派生一个改了参数的 detector（权重重新加载）。"""
        with self._load_lock:
            self._config = replace(self._config, **changes)
            self._model = None
            self._names = {}
            self._remap = None
            self._remap_shape = None
            self._undistort_skipped_logged = False
            self._calib_size_rejected_logged = False
        return self
