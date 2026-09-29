"""YOLO 检测：把一帧画面变成若干 :class:`Detection`。

设计要点（都来自实测，不是随手选的）
------------------------------------
* **device 必须显式指定**。旧环境实测 ultralytics 的自动检测会误选 CPU
  （GPU 9ms/帧 vs CPU 39ms/帧）。P5 复测结论：开发机
  torch 2.13.0+cu130 + ultralytics 8.4.123 下自动检测**已不再误选**
  （自动落在 ``cuda:0``，实测 720p GPU 28.6ms / CPU 114.8ms），但显式指定
  依旧保留——它把"哪天自动检测又变了"这类问题挡在配置里，代价为零。
* **预计算 remap 去畸变**：逐帧 ``cv2.undistort`` 比预计算 ``cv2.remap`` 慢
  3~5 倍。标定文件给出 K/畸变系数后就在 :meth:`Detector.prepare_undistort`
  里算好映射表，之后每帧只是两次查表。
* **帧尺寸与标定不一致就去畸变跳过**（不是报错）：标定是在某个分辨率下做的，
  换成别的分辨率硬套内参会把画面搞坏；跳过并记一次 warning 更安全。
* 重依赖（``ultralytics`` / ``torch``）**惰性导入**：本模块要被离线测试导入，
  不能因为环境里没有 GPU 栈就连 ``import airdrop.perception`` 都失败。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..telemetry.models import TelemetrySnapshot
from .models import Detection, PixelBox

LOGGER = logging.getLogger(__name__)

__all__ = ["DetectionBatch", "Detector", "DetectorConfig"]


@dataclass(frozen=True, slots=True)
class DetectorConfig:
    """YOLO 检测参数。"""

    # YOLO 检测权重（单类 target）
    model_path: str = "models/best2.pt"
    device: str = "0"  # 显式指定：见模块 docstring
    conf_threshold: float = 0.25
    iou_threshold: float = 0.45
    imgsz: int = 1280  # best2.pt 训练 imgsz=640；实测 640/1280 检出一致
    max_detections: int = 300
    # 目标框外扩比例（裁剪时保留完整轮廓）
    crop_expand_ratio: float = 0.2
    # 去畸变（标定产出；为 None 表示不做）
    camera_matrix: np.ndarray | None = None
    dist_coeffs: np.ndarray | None = None
    undistort: bool = True


@dataclass(frozen=True, slots=True)
class DetectionBatch:
    """一帧的检测输出：画面尺寸 + 各目标。

    ``image`` 是**已去畸变**的画面（若启用了去畸变）——下游裁剪必须用同一张，
    否则像素坐标与解算所用的内参对不上。
    """

    image: np.ndarray
    detections: tuple[Detection, ...]
    undistorted: bool = False


class Detector:
    """YOLO 封装：惰性加载权重、可选去畸变、输出 :class:`Detection`。

    ``Detector`` 自身**不碰遥测**：调用方把该帧拍摄时刻的
    :class:`TelemetrySnapshot` 传进来，检测结果就带上它。这样 detector 既能在
    实时链路上用，也能在离线回放里用（回放时遥测来自日志）。
    """

    def __init__(self, config: DetectorConfig | None = None) -> None:
        self._config = config or DetectorConfig()
        self._model: Any = None
        self._names: dict[int, str] = {}
        self._remap: tuple[np.ndarray, np.ndarray] | None = None
        self._remap_shape: tuple[int, int] | None = None
        self._undistort_skipped_logged = False

    # ------------------------------------------------------------------
    # 状态
    # ------------------------------------------------------------------
    @property
    def config(self) -> DetectorConfig:
        return self._config

    @property
    def loaded(self) -> bool:
        """权重是否已加载（未加载时 :meth:`detect` 会自动加载）。"""
        return self._model is not None

    @property
    def names(self) -> dict[int, str]:
        """类别索引 → 名称（加载后可用）。"""
        return dict(self._names)

    @property
    def class_count(self) -> int:
        return len(self._names)

    # ------------------------------------------------------------------
    # 加载与预热
    # ------------------------------------------------------------------
    def load(self) -> "Detector":
        """加载权重（幂等）。找不到文件时抛出带路径的错误。"""
        if self._model is not None:
            return self
        path = Path(self._config.model_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"YOLO 权重不存在: {path.resolve()}（PerceptionConfig.model_path）"
            )
        from ultralytics import YOLO  # 惰性：重型依赖

        LOGGER.info("加载 YOLO 权重 %s（device=%s）", path, self._config.device)
        self._model = YOLO(str(path))
        names = getattr(self._model, "names", None) or {}
        self._names = {int(k): str(v) for k, v in dict(names).items()}
        LOGGER.info("权重类别 %d 个: %s", len(self._names), self._names)
        return self

    def warmup(self, width: int, height: int) -> bool:
        """按目标分辨率跑一次空推理，把 CUDA 上下文/内核编译开销挪到起飞前。

        ``ultralytics`` 首次推理要 80ms+（含上载与 cuDNN 选核），后续 20~30ms；
        起飞前预热能避免第一帧的卡顿。返回是否成功（失败只记日志不抛出——
        预热失败不该拦住任务，真正的错误会在 :meth:`detect` 里暴露）。
        """
        try:
            self.load()
            blank = np.zeros((height, width, 3), np.uint8)
            self._model.predict(
                blank,
                device=self._config.device,
                conf=self._config.conf_threshold,
                iou=self._config.iou_threshold,
                imgsz=self._config.imgsz,
                max_det=self._config.max_detections,
                verbose=False,
            )
            return True
        except Exception:
            LOGGER.exception("YOLO 预热失败（不影响后续真实检测）")
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
        new_matrix, _ = cv2.getOptimalNewCameraMatrix(
            matrix, dist, size, alpha=0.0, newImgSize=size
        )
        map_x, map_y = cv2.initUndistortRectifyMap(
            matrix, dist, None, new_matrix, size, cv2.CV_16SC2
        )
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
        直接取类别。
        """
        self.load()
        assert self._model is not None  # load() 已保证

        do_undistort = self._config.undistort if undistort is None else undistort
        config = self._config
        if do_undistort and config.camera_matrix is not None:
            picture, rectified = self.undistort(image)
        else:
            picture, rectified = image, False

        results = self._model.predict(
            picture,
            device=config.device,
            conf=config.conf_threshold,
            iou=config.iou_threshold,
            imgsz=config.imgsz,
            max_det=config.max_detections,
            verbose=False,
        )
        snapshot = telemetry if telemetry is not None else TelemetrySnapshot()
        height, width = picture.shape[:2]
        detections: list[Detection] = []
        if results:
            boxes = getattr(results[0], "boxes", None)
            if boxes is not None and len(boxes):
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                classes = boxes.cls.cpu().numpy().astype(int)
                for (x1, y1, x2, y2), conf, cls in zip(xyxy, confs, classes, strict=True):
                    box = PixelBox(float(x1), float(y1), float(x2), float(y2)).clipped(
                        width, height
                    )
                    if box.area <= 0:
                        continue
                    detections.append(
                        Detection(
                            frame_index=frame_index,
                            capture_timestamp=capture_timestamp,
                            pixel=box.center,
                            box=box,
                            confidence=float(conf),
                            telemetry=snapshot,
                            # cls12：类别直出编号（1 起）。ocr 模式留空待补。
                            code=int(cls) + 1 if mode == "cls12" else None,
                            mode=mode,
                            # ``undistorted`` 是给下游坐标解算看的：像素在**去过畸变**的
                            # 图上时，georef 不能再纠正一遍（否则纠正两次）。
                            extra={
                                "class_name": self._names.get(int(cls), ""),
                                "undistorted": bool(rectified),
                            },
                        )
                    )
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return DetectionBatch(
            image=picture,
            detections=tuple(detections),
            undistorted=rectified,
        )

    def crop(self, batch: DetectionBatch, detection: Detection) -> np.ndarray:
        """按检测框裁出目标图（含外扩，保留完整五边形轮廓）。"""
        box = detection.box.expanded(self._config.crop_expand_ratio).clipped(
            batch.image.shape[1], batch.image.shape[0]
        )
        x1, y1 = int(box.x1), int(box.y1)
        x2, y2 = int(np.ceil(box.x2)), int(np.ceil(box.y2))
        if x2 <= x1 or y2 <= y1:
            return np.zeros((0, 0, 3), np.uint8)
        return batch.image[y1:y2, x1:x2].copy()

    def replace_config(self, **changes: Any) -> "Detector":
        """派生一个改了参数的 detector（权重重新加载）。"""
        self._config = replace(self._config, **changes)
        self._model = None
        self._remap = None
        self._remap_shape = None
        return self
