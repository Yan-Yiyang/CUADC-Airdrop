"""TensorRT FP16 OCR 后端：复刻 RapidOCR 文本检测与识别前后处理。

设计契约
--------
* 检测（``ocr_det``）与识别（``ocr_rec``）均通过
  :func:`~airdrop.perception.trt_runtime.load_trt_runner` 加载离线构建的 ``.engine``
  及其伴随清单，加载期核验字典 SHA256、预处理契约与动态 profile 边界；
* 复用 ``rapidocr`` 的无框架前后处理工具（``DetPreProcess``、``DBPostProcess``、
  ``CTCLabelDecode`` 及图像变换函数），不导入 ``RapidOCR`` 主入口或 ``onnxruntime``；
* 暴露 ``text_det``、``text_rec``、``recognize_txt`` 与 ``__call__`` 契约，兼容
  :meth:`~airdrop.perception.cropproc.OcrEngine.warmup` 与
  :func:`~airdrop.perception.timing.instrument_call_class`；
* 引擎校验或推理形状越界一律抛出 :class:`~airdrop.perception.trt_runtime.TrtEngineError`
  （或其子类 :class:`~airdrop.perception.trt_runtime.TrtShapeOutOfRangeError`），
  不在本模块吞咽降级。
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

if TYPE_CHECKING:
    from .cropproc import OcrEngineConfig
    from .trt_runtime import TensorSpec, TrtExecutionRunner

__all__ = [
    "TrtDetOutput",
    "TrtOcrBackend",
    "TrtRecOutput",
    "TrtTextDetector",
    "TrtTextRecognizer",
]

#: 与 RapidOCR 默认 Det 配置保持一致的检测短边下限。
_DET_LIMIT_SIDE_LEN = 736
#: 与 RapidOCR TextDetector.sorted_boxes 一致的同行判定纵向阈值（像素）。
_BOX_SORT_Y_THRESHOLD = 10
#: 与 RapidOCR 默认 Rec 配置保持一致的输入形状 (C, H, W)。
_REC_IMAGE_SHAPE: tuple[int, int, int] = (3, 48, 320)
#: 与 RapidOCR Global 默认预处理参数保持一致。
_GLOBAL_MIN_SIDE_LEN = 30
_GLOBAL_MAX_SIDE_LEN = 2000
_GLOBAL_MIN_HEIGHT = 30
_GLOBAL_WIDTH_HEIGHT_RATIO = 8


@dataclass(slots=True)
class TrtDetOutput:
    """TensorRT 文本检测输出（对齐 RapidOCR ``TextDetOutput`` 契约）。"""

    img: np.ndarray | None = None
    boxes: np.ndarray | None = None
    scores: list[float] | None = None
    elapse: float = 0.0

    def __len__(self) -> int:
        return 0 if self.boxes is None else len(self.boxes)


@dataclass(slots=True)
class TrtRecOutput:
    """TensorRT 文本识别输出（对齐 RapidOCR ``TextRecOutput`` 契约）。"""

    imgs: list[np.ndarray] | None = None
    txts: tuple[str, ...] | None = None
    scores: tuple[float, ...] | None = None
    elapse: float = 0.0

    def __len__(self) -> int:
        return 0 if self.txts is None else len(self.txts)


def _ensure_bgr_image(image: np.ndarray) -> np.ndarray:
    """将输入 NumPy 图像规范为 3 通道 BGR（对齐 RapidOCR ``LoadImage`` 契约）。"""
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3:
        channels = image.shape[2]
        if channels == 3:
            return image
        if channels == 1:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if channels == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        raise ValueError(f"OCR 输入图像通道数不受支持: {channels}（仅支持 1/3/4）")
    raise ValueError(f"OCR 输入图像维度数不受支持: {image.ndim}（仅支持 2/3）")


def _ensure_within_profile(spec: TensorSpec, shape: tuple[int, ...]) -> None:
    """校验本次推理输入形状处于 TensorRT 引擎优化配置边界内。"""
    from .trt_runtime import TrtEngineError, TrtShapeOutOfRangeError

    if len(shape) != len(spec.shape):
        raise TrtShapeOutOfRangeError(
            f"TensorRT OCR 输入 {spec.name!r} 维度数不匹配: 实际 {shape}，期望秩 {len(spec.shape)}"
        )
    if not spec.is_dynamic:
        if shape != spec.shape:
            raise TrtShapeOutOfRangeError(
                f"TensorRT OCR 静态输入 {spec.name!r} 形状不匹配: 实际 {shape}，期望 {spec.shape}"
            )
        return
    if spec.min_shape is None or spec.max_shape is None:
        raise TrtEngineError(f"TensorRT OCR 动态输入 {spec.name!r} 缺少 profile 边界")
    for axis, (actual, expected, lo, hi) in enumerate(
        zip(shape, spec.shape, spec.min_shape, spec.max_shape, strict=True)
    ):
        if expected > 0 and actual != expected:
            raise TrtShapeOutOfRangeError(
                f"TensorRT OCR 输入 {spec.name!r} 第 {axis} 维固定为 {expected}，实际收到 {actual}"
            )
        if not (lo <= actual <= hi):
            raise TrtShapeOutOfRangeError(
                f"TensorRT OCR 输入 {spec.name!r} 形状 {shape} 超出引擎 profile 范围 "
                f"[{spec.min_shape}, {spec.max_shape}]"
            )


class TrtTextDetector:
    """TensorRT 文本检测器（``DetPreProcess`` → TRT → ``DBPostProcess`` → ``sorted_boxes``）。"""

    def __init__(
        self,
        config: OcrEngineConfig,
        *,
        runner: TrtExecutionRunner | None = None,
    ) -> None:
        from rapidocr.ch_ppocr_det.utils import DBPostProcess, DetPreProcess

        from .trt_runtime import TrtEngineError, load_trt_runner

        self._config = config
        self._runner = (
            runner
            if runner is not None
            else load_trt_runner(
                config.det_path(),
                expected_kind="ocr_det",
                device_id=config.device_id,
            )
        )
        manifest = self._runner.manifest
        det_meta = manifest.model.get("ocr_det")
        if not isinstance(det_meta, Mapping):
            raise TrtEngineError("TensorRT OCR 检测清单缺少 model.ocr_det 字段")

        limit_type = det_meta.get("limit_type")
        limit_side_len = det_meta.get("limit_side_len")
        if (
            limit_type != "min"
            or type(limit_side_len) is not int
            or limit_side_len != _DET_LIMIT_SIDE_LEN
        ):
            raise TrtEngineError(
                "TensorRT OCR 检测清单预处理参数不匹配: "
                f"limit_type={limit_type!r}, limit_side_len={limit_side_len!r} "
                f"（期望 limit_type='min', limit_side_len={_DET_LIMIT_SIDE_LEN}）"
            )

        input_spec = self._runner.input_spec
        max_shape = input_spec.max_shape
        if max_shape is None or len(max_shape) != 4:
            raise TrtEngineError(
                f"TensorRT OCR 检测引擎必须具有 4 维动态输入 profile: "
                f"shape={input_spec.shape}, max_shape={max_shape}"
            )
        if max_shape[1] != 3:
            raise TrtEngineError(
                f"TensorRT OCR 检测引擎输入通道数必须为 3，实际 max_shape={max_shape}"
            )
        if max_shape[2] < config.det_max_side_len or max_shape[3] < config.det_max_side_len:
            raise TrtEngineError(
                f"TensorRT OCR 检测引擎 profile 最大尺寸 ({max_shape[2]}x{max_shape[3]}) "
                f"小于配置的 det_max_side_len={config.det_max_side_len}"
            )

        mean = det_meta.get("mean", [0.5, 0.5, 0.5])
        std = det_meta.get("std", [0.5, 0.5, 0.5])
        self._preprocess_op = DetPreProcess(
            limit_side_len=_DET_LIMIT_SIDE_LEN,
            limit_type="min",
            mean=mean,
            std=std,
        )
        self._postprocess_op = DBPostProcess(
            thresh=float(det_meta.get("thresh", 0.3)),
            box_thresh=float(det_meta.get("box_thresh", 0.5)),
            max_candidates=int(det_meta.get("max_candidates", 1000)),
            unclip_ratio=float(det_meta.get("unclip_ratio", 1.6)),
            use_dilation=bool(det_meta.get("use_dilation", True)),
            score_mode=str(det_meta.get("score_mode", "fast")),
        )

    @property
    def runner(self) -> TrtExecutionRunner:
        return self._runner

    @staticmethod
    def sorted_boxes(dt_boxes: np.ndarray) -> np.ndarray:
        """按纵坐标分行、同行内按横坐标升序排列检测框（与 RapidOCR 一致）。"""
        if len(dt_boxes) == 0:
            return dt_boxes

        y_coords = dt_boxes[:, 0, 1]
        y_order = np.argsort(y_coords, kind="stable")
        boxes_y_sorted = dt_boxes[y_order]
        y_sorted = y_coords[y_order]

        dy = np.diff(y_sorted)
        line_increments = (dy >= _BOX_SORT_Y_THRESHOLD).astype(np.int32)
        line_ids = np.concatenate([[0], np.cumsum(line_increments)])

        x_coords = boxes_y_sorted[:, 0, 0]
        final_order = np.lexsort((x_coords, line_ids))
        return boxes_y_sorted[final_order]

    def __call__(self, image: np.ndarray) -> TrtDetOutput:
        """执行单帧文本检测并返回有序四点框。"""
        start_time = time.perf_counter()
        if image is None:
            raise ValueError("image 不能为 None")

        ori_shape = (int(image.shape[0]), int(image.shape[1]))
        pre = self._preprocess_op(image)
        if pre is None:
            return TrtDetOutput()

        _ensure_within_profile(self._runner.input_spec, tuple(int(d) for d in pre.shape))
        preds = self._runner.run_numpy(pre)
        boxes, scores = self._postprocess_op(preds, ori_shape)
        if len(boxes) < 1:
            return TrtDetOutput()

        ordered = self.sorted_boxes(boxes)
        elapse = time.perf_counter() - start_time
        return TrtDetOutput(img=image, boxes=ordered, scores=list(scores), elapse=elapse)


class TrtTextRecognizer:
    """TensorRT 文本识别器（按宽高比排序分批 → ``resize_norm_img`` → TRT → ``CTCLabelDecode``）。"""

    def __init__(
        self,
        config: OcrEngineConfig,
        *,
        runner: TrtExecutionRunner | None = None,
    ) -> None:
        from rapidocr.ch_ppocr_rec.utils import CTCLabelDecode

        from .trt_runtime import TrtEngineError, load_trt_runner, sha256_file

        self._config = config
        self._runner = (
            runner
            if runner is not None
            else load_trt_runner(
                config.rec_path(),
                expected_kind="ocr_rec",
                device_id=config.device_id,
            )
        )
        manifest = self._runner.manifest

        expected_dict_sha = manifest.source.get("dict_sha256")
        if not isinstance(expected_dict_sha, str) or not expected_dict_sha.strip():
            raise TrtEngineError("TensorRT OCR 识别清单缺少 source.dict_sha256 字段")
        actual_dict_sha = sha256_file(config.keys_path())
        if expected_dict_sha.lower() != actual_dict_sha.lower():
            raise TrtEngineError(
                f"OCR 字典文件 SHA256 与 TensorRT 清单不一致: "
                f"实际 {actual_dict_sha}，清单 {expected_dict_sha}"
            )

        rec_meta = manifest.model.get("ocr_rec")
        if not isinstance(rec_meta, Mapping):
            raise TrtEngineError("TensorRT OCR 识别清单缺少 model.ocr_rec 字段")
        raw_rec_shape = rec_meta.get("rec_image_shape")
        if (
            not isinstance(raw_rec_shape, (list, tuple))
            or len(raw_rec_shape) != 3
            or any(type(dim) is not int for dim in raw_rec_shape)
            or tuple(raw_rec_shape) != _REC_IMAGE_SHAPE
        ):
            raise TrtEngineError(
                f"TensorRT OCR 识别清单 model.ocr_rec.rec_image_shape 不匹配: "
                f"实际 {raw_rec_shape!r}，期望 {list(_REC_IMAGE_SHAPE)}"
            )

        input_spec = self._runner.input_spec
        max_shape = input_spec.max_shape
        if max_shape is None or len(max_shape) != 4:
            raise TrtEngineError(
                f"TensorRT OCR 识别引擎必须具有 4 维动态输入 profile: "
                f"shape={input_spec.shape}, max_shape={max_shape}"
            )
        if config.rec_batch_num < 1 or config.rec_batch_num > max_shape[0]:
            raise TrtEngineError(
                f"OCR rec_batch_num={config.rec_batch_num} 超出 TensorRT 引擎最大批大小 {max_shape[0]}"
            )
        if (
            max_shape[1] != _REC_IMAGE_SHAPE[0]
            or max_shape[2] != _REC_IMAGE_SHAPE[1]
            or max_shape[3] < _REC_IMAGE_SHAPE[2]
        ):
            raise TrtEngineError(
                f"TensorRT OCR 识别引擎 profile 维度与 rec_image_shape={list(_REC_IMAGE_SHAPE)} 不兼容: "
                f"max_shape={max_shape}"
            )

        self._rec_batch_num = int(config.rec_batch_num)
        self._rec_image_shape = _REC_IMAGE_SHAPE
        self._postprocess_op = CTCLabelDecode(character_path=str(config.keys_path()))

    @property
    def runner(self) -> TrtExecutionRunner:
        return self._runner

    def resize_norm_img(self, img: np.ndarray, max_wh_ratio: float) -> np.ndarray:
        """按批次最大宽高比缩放、归一化并右补零（与 RapidOCR ``TextRecognizer`` 一致）。"""
        img_channel, img_height, _ = self._rec_image_shape
        if img.ndim != 3 or img.shape[2] != img_channel:
            raise ValueError(f"文本行图像必须为 {img_channel} 通道，实际 shape={img.shape}")

        h, w = img.shape[:2]
        if h <= 0 or w <= 0:
            raise ValueError(f"文本行图像尺寸无效: {img.shape}")

        img_width = int(img_height * max_wh_ratio)
        ratio = w / float(h)
        if math.ceil(img_height * ratio) > img_width:
            resized_w = img_width
        else:
            resized_w = int(math.ceil(img_height * ratio))

        resized_image = cv2.resize(img, (resized_w, img_height))
        resized_image = resized_image.astype("float32")
        resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5

        padding_im = np.zeros((img_channel, img_height, img_width), dtype=np.float32)
        padding_im[:, :, 0:resized_w] = resized_image
        return padding_im

    def __call__(self, img_list: Sequence[np.ndarray] | np.ndarray) -> TrtRecOutput:
        """批量识别文本行并按原始输入顺序返回文本与置信度。"""
        start_time = time.perf_counter()
        images = [img_list] if isinstance(img_list, np.ndarray) else list(img_list)
        if not images:
            return TrtRecOutput(imgs=[], txts=(), scores=(), elapse=0.0)

        width_list = [img.shape[1] / float(img.shape[0]) for img in images]
        indices = np.argsort(np.array(width_list))

        img_num = len(images)
        rec_res: list[tuple[str, float]] = [("", 0.0)] * img_num
        _, img_h, img_w = self._rec_image_shape
        base_wh_ratio = img_w / img_h

        for beg_img_no in range(0, img_num, self._rec_batch_num):
            end_img_no = min(img_num, beg_img_no + self._rec_batch_num)
            max_wh_ratio = base_wh_ratio
            wh_ratio_list: list[float] = []
            for ino in range(beg_img_no, end_img_no):
                h, w = images[int(indices[ino])].shape[:2]
                wh_ratio = w * 1.0 / h
                max_wh_ratio = max(max_wh_ratio, wh_ratio)
                wh_ratio_list.append(wh_ratio)

            norm_batch = np.concatenate(
                [
                    self.resize_norm_img(images[int(indices[ino])], max_wh_ratio)[np.newaxis, :]
                    for ino in range(beg_img_no, end_img_no)
                ]
            ).astype(np.float32)

            _ensure_within_profile(self._runner.input_spec, tuple(int(d) for d in norm_batch.shape))
            preds = self._runner.run_numpy(norm_batch)
            line_results, _ = self._postprocess_op(
                preds,
                False,
                wh_ratio_list=wh_ratio_list,
                max_wh_ratio=max_wh_ratio,
            )
            for rno, one_res in enumerate(line_results):
                rec_res[int(indices[beg_img_no + rno])] = (str(one_res[0]), float(one_res[1]))

        txts = tuple(item[0] for item in rec_res)
        scores = tuple(item[1] for item in rec_res)
        elapse = time.perf_counter() - start_time
        return TrtRecOutput(imgs=images, txts=txts, scores=scores, elapse=elapse)


class TrtOcrBackend:
    """TensorRT 整图 OCR 后端（复刻 ``RapidOCR`` 流水线契约，不依赖 ``onnxruntime``）。"""

    def __init__(
        self,
        config: OcrEngineConfig,
        *,
        text_det: TrtTextDetector | None = None,
        text_rec: TrtTextRecognizer | None = None,
    ) -> None:
        from rapidocr.utils.process_img import (
            apply_vertical_padding,
            get_rotate_crop_image,
            map_boxes_to_original,
            resize_image_within_bounds,
        )

        self._config = config
        self._text_score = float(config.text_score)
        self._resize_within_bounds = resize_image_within_bounds
        self._apply_vertical_padding = apply_vertical_padding
        self._get_rotate_crop_image = get_rotate_crop_image
        self._map_boxes_to_original = map_boxes_to_original
        self.text_det = text_det if text_det is not None else TrtTextDetector(config)
        self.text_rec = text_rec if text_rec is not None else TrtTextRecognizer(config)

    def clone_context(self) -> TrtOcrBackend:
        """复用已反序列化引擎，创建带独立执行上下文、CUDA stream 与前后处理状态的副本。"""
        return TrtOcrBackend(
            self._config,
            text_det=TrtTextDetector(
                self._config,
                runner=self.text_det.runner.clone_context(),
            ),
            text_rec=TrtTextRecognizer(
                self._config,
                runner=self.text_rec.runner.clone_context(),
            ),
        )

    def recognize_txt(self, images: Sequence[np.ndarray] | np.ndarray) -> TrtRecOutput:
        """执行纯识别阶段（供 :meth:`OcrEngine.warmup` 与流水线内部复用）。"""
        return self.text_rec(images)

    def _preprocess_img(self, ori_img: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        img, ratio_h, ratio_w = self._resize_within_bounds(
            ori_img, _GLOBAL_MIN_SIDE_LEN, _GLOBAL_MAX_SIDE_LEN
        )
        op_record: dict[str, Any] = {"preprocess": {"ratio_h": ratio_h, "ratio_w": ratio_w}}
        return img, op_record

    def __call__(self, image: np.ndarray) -> SimpleNamespace:
        """执行整图检测、透视裁切、识别、坐标映射与置信度过滤。"""
        if image is None or image.size == 0:
            return SimpleNamespace(boxes=None, txts=None, scores=None)

        ori_img = _ensure_bgr_image(image)
        ori_h, ori_w = ori_img.shape[:2]
        img, op_record = self._preprocess_img(ori_img)
        img, op_record = self._apply_vertical_padding(
            img, op_record, _GLOBAL_WIDTH_HEIGHT_RATIO, _GLOBAL_MIN_HEIGHT
        )

        det_res = self.text_det(img)
        if det_res.boxes is None or len(det_res.boxes) == 0:
            return SimpleNamespace(boxes=None, txts=None, scores=None)

        crops = [
            self._get_rotate_crop_image(img, np.asarray(box, dtype=np.float32).copy())
            for box in det_res.boxes
        ]
        rec_res = self.recognize_txt(crops)
        if rec_res.txts is None or rec_res.scores is None:
            return SimpleNamespace(boxes=None, txts=None, scores=None)

        mapped_boxes = self._map_boxes_to_original(det_res.boxes.copy(), op_record, ori_h, ori_w)

        filtered_boxes: list[np.ndarray] = []
        filtered_txts: list[str] = []
        filtered_scores: list[float] = []
        for box, txt, score in zip(mapped_boxes, rec_res.txts, rec_res.scores, strict=True):
            if not txt.strip():
                continue
            if score < self._text_score:
                continue
            filtered_boxes.append(box)
            filtered_txts.append(txt)
            filtered_scores.append(float(score))

        if not filtered_txts:
            return SimpleNamespace(boxes=None, txts=None, scores=None)
        return SimpleNamespace(
            boxes=np.array(filtered_boxes),
            txts=tuple(filtered_txts),
            scores=tuple(filtered_scores),
        )
