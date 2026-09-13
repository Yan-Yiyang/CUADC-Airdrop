"""裁剪图后处理：五边形定位 → 几何转正 → 整图 OCR 读编号。

移植自 旧版 ``drone/cvproc.py``（久经实测的目标形态与参数）。目标形态：
**五边形 = 正方形 + 等边三角形**（边长 1m），编号印在正方形中心的白底方块上
（0.6m，黑字）。

流水线（顺序不可随意改动，每一步都在 旧版 有实测理由）
--------------------------------------------------------
1. **JPEG 往返去噪**：DVR 图传的细颗粒噪点会干扰轮廓提取，编解码一次即可抹平；
2. **等比放大到短边 300px**（上限 10 倍）：小目标识别率显著提升。``side_px``
   最后要除回去，还原成原始像素边长（坐标解算用原始尺度）；
3. **颜色掩码饱和度逐级回退**：无人机接近时目标褪色（蓝实测 S≈20~30），
   固定阈值会突然检不到；blue 走 100→60→40→20，red 走 100→80→60→40→20；
4. **形态学 + 凸包 + approxPolyDP 扫 epsilon=3..40** 找出五边形。扫描 epsilon
   而不是 ±1 步进，避免旧的振荡问题；
5. **转正**：平行边法为主（正方形左右两边是轮廓里唯一互相平行的边对，三角形
   顶角是唯一不与任一平行边共享端点的顶点），失败回退最小内角法（40~130° 门限，
   斜视角会把 60° 投影到这个区间）；
6. **转正图整图 OCR**。注意用**彩图**——PP-OCRv6 的 det 对灰度图检不出文本框
   （旧版 实测坑，P5 未复测，保守沿用）；
7. **候选合并**：单数字框按行聚类拼成两位数；两位结果置信度加权（×1.15）、
   单位 ×0.6，防止把 ``"7"`` 误报成 ``07``；
8. 编号交给 :func:`~airdrop.perception.number.correct_ocr_number` 纠错。

与 旧版 的差异（都是被 rapidocr 3.9.2 的 API 逼的，不是主动改的）
------------------------------------------------------------------
* 旧版 的 ``_rapid_readtext`` 往 ``__call__`` 上传 ``allowlist`` / ``low_text``
  / ``text_threshold`` / ``batch_size`` 等参数；3.9.2 的 ``__call__`` **只接受
  图像**，这些参数已不存在。替代方案：
  - 置信度过滤交给 ``Global.text_score``（构造时设定，等价于低阈值放行 +
    自己按 ``ocr_conf_threshold`` 复筛）；
  - ``allowlist`` 没有对应配置项，改由 :func:`correct_ocr_number` 过滤非数字
    （它本来就在做这件事，且更严格）；
  - ``rec_batch_num`` 走 ``Rec.rec_batch_num`` 配置。
* 方向分类（Cls）**关闭**：``rapidocr`` 的 ``arch_config.yaml`` 里没有 Cls 的
  torch 架构条目（只有 det/rec），TORCH 引擎加载不了 Cls 权重；而几何转正已经
  保证了编号方向，Cls 本来就是冗余的。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .number import correct_ocr_number

LOGGER = logging.getLogger(__name__)

__all__ = ["CropResult", "OpenCvPostProcess", "OcrEngine", "OcrEngineConfig"]


# ----------------------------------------------------------------------
# RapidOCR 引擎封装
# ----------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class OcrEngineConfig:
    'RapidOCR（TORCH 引擎）参数。'

    model_dir: str = "models/ppocr"
    det_model: str = "PP-OCRv6_det_medium.pth"
    rec_model: str = "PP-OCRv6_rec_medium.pth"
    rec_keys: str = "ppocrv6_dict.txt"
    use_cuda: bool = True
    device_id: int = 0
    text_score: float = 0.1
    rec_batch_num: int = 6
    log_level: str = "error"

    def det_path(self) -> Path:
        return Path(self.model_dir) / self.det_model

    def rec_path(self) -> Path:
        return Path(self.model_dir) / self.rec_model

    def keys_path(self) -> Path:
        return Path(self.model_dir) / self.rec_keys


@dataclass(frozen=True, slots=True)
class OcrText:
    """一条 OCR 文本结果（框 + 内容 + 置信度）。"""

    box: np.ndarray      # (4, 2) float
    text: str
    score: float


class OcrEngine:
    """``rapidocr.RapidOCR`` 的薄封装：显式模型路径 + 统一结果形状。

    为什么不用库的自动下载：起飞前必须**固定模型版本**（否则某次飞行前悄悄换了
    模型，识别率变化无法归因），而 ``default_models.yaml`` 里没有 ``torch`` 段，
    TORCH 引擎只能靠显式 ``model_path`` 指定本地权重（P5 复测结论）。
    """

    def __init__(self, config: OcrEngineConfig | None = None) -> None:
        self._config = config or OcrEngineConfig()
        self._engine: Any = None
        self._failed = False

    @property
    def config(self) -> OcrEngineConfig:
        return self._config

    @property
    def available(self) -> bool:
        """引擎是否已就绪（不触发加载）。"""
        return self._engine is not None

    @property
    def failed(self) -> bool:
        """是否已经尝试加载并失败（不再反复重试）。"""
        return self._failed

    def ensure(self) -> Any:
        """惰性初始化引擎；失败抛出（调用方决定是否降级）。

        首次初始化约 2s（加载 .pth 到 GPU），应该由 :meth:`warmup` 在起飞前完成。
        """
        if self._engine is not None:
            return self._engine
        if self._failed:
            raise RuntimeError("OCR 引擎此前初始化失败，请检查模型文件与 CUDA")
        config = self._config
        missing = [
            str(path)
            for path in (config.det_path(), config.rec_path(), config.keys_path())
            if not path.is_file()
        ]
        if missing:
            self._failed = True
            raise FileNotFoundError(
                "OCR 模型文件缺失: " + ", ".join(missing)
                + "（见 OcrEngineConfig.model_dir）"
            )
        try:
            from rapidocr import EngineType, RapidOCR  # 惰性：重型依赖

            self._engine = RapidOCR(
                params={
                    "Det.engine_type": EngineType.TORCH,
                    "Det.model_path": str(config.det_path()),
                    "Rec.engine_type": EngineType.TORCH,
                    "Rec.model_path": str(config.rec_path()),
                    "Rec.rec_keys_path": str(config.keys_path()),
                    "Rec.rec_batch_num": config.rec_batch_num,
                    "Cls.engine_type": EngineType.TORCH,
                    "Global.use_cls": False,      # 见模块 docstring
                    "Global.text_score": config.text_score,
                    "EngineConfig.torch.use_cuda": config.use_cuda,
                    "EngineConfig.torch.cuda_ep_cfg.device_id": config.device_id,
                    "Global.log_level": config.log_level,
                }
            )
        except Exception:
            self._failed = True
            raise
        LOGGER.info(
            "OCR 引擎就绪（TORCH，det=%s rec=%s cuda=%s）",
            config.det_model,
            config.rec_model,
            config.use_cuda,
        )
        return self._engine

    def warmup(self, size: int = 320) -> bool:
        """跑一次空图，把模型加载/CUDA 上下文开销挪到起飞前。"""
        try:
            engine = self.ensure()
            blank = np.full((size, size, 3), 255, np.uint8)
            engine(blank)
            return True
        except Exception:  # noqa: BLE001 - 预热失败只记录
            LOGGER.exception("OCR 预热失败（不影响后续识别，但首帧会变慢）")
            return False

    def read_text(self, image: np.ndarray) -> list[OcrText]:
        """整图识别，返回文本列表；引擎不可用或识别失败时返回空列表。

        **绝不抛出**：单帧 OCR 失败不该打断整条拉流/处理链路，上层按"没读出
        编号"处理即可（``Detection.code is None``）。
        """
        try:
            engine = self.ensure()
        except Exception:  # noqa: BLE001
            LOGGER.exception("OCR 引擎不可用")
            return []
        try:
            result = engine(image)
        except Exception:  # noqa: BLE001
            LOGGER.exception("OCR 单帧识别失败")
            return []
        if not result:
            return []
        texts = getattr(result, "txts", None)
        scores = getattr(result, "scores", None)
        boxes = getattr(result, "boxes", None)
        if texts is None or boxes is None:
            return []
        out: list[OcrText] = []
        for index, text in enumerate(texts):
            if not text:
                continue
            box = np.asarray(boxes[index], dtype=np.float64).reshape(-1, 2)
            if len(box) < 4:
                continue
            score = 1.0 if scores is None else float(scores[index])
            out.append(OcrText(box=box[:4], text=str(text), score=score))
        return out


# ----------------------------------------------------------------------
# 结果
# ----------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class CropResult:
    """一次裁剪图识别的结果；``number`` 为 None 表示没读出来。"""

    number: int | None
    side_px: float
    rectified: np.ndarray | None
    raw_text: str
    confidence: float
    saturation_level: int = 0
    stage: str = ""       # "parallel" / "min_angle" / "none"，排故用

    @property
    def ok(self) -> bool:
        return self.number is not None


# ----------------------------------------------------------------------
# 后处理主类
# ----------------------------------------------------------------------
class OpenCvPostProcess:
    """目标裁剪图后处理：颜色掩码 → 凸包五边形 → 转正 → 整图 OCR。

    参数命名与 旧版 对齐（``color`` / ``ocr_conf_threshold``），便于对照实测值；
    逻辑按新的 RapidOCR 接口重写，几何部分逐行保留。
    """

    # 蓝色掩码饱和度下界回退序列（无人机接近时目标蓝色显著褪色，实测 S≈20-30）
    BLUE_S_MIN_LEVELS = (100, 60, 40, 20)
    # 红色掩码饱和度下界回退序列（DVR 画质/逆光红靶标同样会褪色）
    RED_S_MIN_LEVELS = (100, 80, 60, 40, 20)

    def __init__(
        self,
        color: str = "blue",
        ocr_conf_threshold: float = 0.6,
        *,
        engine: OcrEngine | None = None,
        engine_config: OcrEngineConfig | None = None,
        target_side_px: float = 300.0,
        max_upscale: float = 10.0,
    ) -> None:
        if color not in ("blue", "red"):
            raise ValueError(f"目标颜色只支持 blue/red: {color!r}")
        self.color = color
        self.ocr_conf_threshold = float(ocr_conf_threshold)
        self.target_side_px = float(target_side_px)
        self.max_upscale = float(max_upscale)
        self._engine = engine if engine is not None else OcrEngine(engine_config)
        self.image: np.ndarray | None = None
        self.s_min = self.BLUE_S_MIN_LEVELS[0]
        self.scale = 1.0

    # ------------------------------------------------------------------
    # 引擎
    # ------------------------------------------------------------------
    @property
    def engine(self) -> OcrEngine:
        return self._engine

    def warmup(self) -> bool:
        return self._engine.warmup()

    # ------------------------------------------------------------------
    # 颜色提取
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_blue(image: np.ndarray, s_min: int) -> np.ndarray:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(
            hsv, np.array([90, s_min, 50]), np.array([130, 255, 255])
        )
        blue = cv2.bitwise_and(image, image, mask=mask)
        return cv2.cvtColor(blue, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def _extract_red_level(image: np.ndarray, s_min: int) -> np.ndarray:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        m1 = cv2.inRange(hsv, np.array([0, s_min, 100]), np.array([10, 255, 255]))
        m2 = cv2.inRange(hsv, np.array([160, s_min, 100]), np.array([179, 255, 255]))
        red = cv2.bitwise_and(image, image, mask=m1 + m2)
        return cv2.cvtColor(red, cv2.COLOR_BGR2GRAY)

    def _sat_levels(self) -> tuple[int, ...]:
        return self.BLUE_S_MIN_LEVELS if self.color == "blue" else self.RED_S_MIN_LEVELS

    # ------------------------------------------------------------------
    # 几何工具
    # ------------------------------------------------------------------
    @staticmethod
    def _angle(pt1, pt0, pt2) -> float:
        """``pt0`` 处的夹角（度）。"""
        d1 = np.array(pt1, dtype=np.float64) - np.array(pt0, dtype=np.float64)
        d2 = np.array(pt2, dtype=np.float64) - np.array(pt0, dtype=np.float64)
        n1, n2 = np.linalg.norm(d1), np.linalg.norm(d2)
        if n1 < 1e-9 or n2 < 1e-9:
            return 0.0
        cos_a = float(np.clip(np.dot(d1, d2) / (n1 * n2), -1.0, 1.0))
        return float(np.degrees(np.arccos(cos_a)))

    @classmethod
    def _find_min_angle(cls, approx) -> tuple[list[float], float]:
        """最小内角顶点（三角形顶角 60°，斜视下投影到 40~130°）。"""
        min_angle, min_point = 360.0, [0.0, 0.0]
        length = len(approx)
        for i in range(length):
            t = cls._angle(
                approx[i][0],
                approx[(i + 1) % length][0],
                approx[(i + 2) % length][0],
            )
            if t < min_angle:
                min_angle = t
                min_point = [float(approx[(i + 1) % length][0][0]),
                             float(approx[(i + 1) % length][0][1])]
        return min_point, min_angle

    @staticmethod
    def _edge_angle_deg(a, b) -> float:
        """边方向角（度，模 180：平行边同角）。"""
        d = np.asarray(b, dtype=np.float64) - np.asarray(a, dtype=np.float64)
        if np.linalg.norm(d) < 1e-9:
            return 0.0
        return float(np.degrees(np.arctan2(d[1], d[0]))) % 180.0

    @staticmethod
    def _angle_diff_deg(x: float, y: float) -> float:
        d = abs(x - y) % 180.0
        return min(d, 180.0 - d)

    @staticmethod
    def _angle_norm_deg(x: float) -> float:
        """角度归一到 [-180, 180)。"""
        return (x + 180.0) % 360.0 - 180.0

    @classmethod
    def _parallel_rectify_angle(cls, approx, pair, apex, center) -> float:
        """平行边法转正角。

        两条平行边的平均方向转到竖直；旋转后若顶点落在边下方（说明整体倒了
        180°）再翻一次。``getRotationMatrix2D(θ)`` 的实际变换是
        ``α_after = α - θ``（旧版 实验确认），所以"转竖直"用 ``row = ea - 90``。
        """
        pts = approx.reshape(-1, 2).astype(np.float64)
        n = len(pts)
        e1 = cls._edge_angle_deg(pts[pair[0]], pts[(pair[0] + 1) % n])
        e2 = cls._edge_angle_deg(pts[pair[1]], pts[(pair[1] + 1) % n])
        d = e2 - e1
        if d > 90.0:
            e2 -= 180.0
        elif d < -90.0:
            e2 += 180.0
        ea = (e1 + e2) / 2.0
        row = ea - 90.0
        theta = float(
            np.degrees(np.arctan2(apex[1] - center[1], apex[0] - center[0]))
        )
        after = cls._angle_norm_deg(theta - row)
        if abs(after - 90.0) < abs(after + 90.0):  # 更接近正下方（+90°）
            row += 180.0
        return row

    @classmethod
    def _find_apex_by_parallel(cls, approx, tol: float = 15.0):
        """平行边法定顶点。

        五边形 = 正方形 + 等边三角形：正方形的左右两边是轮廓中**唯一互相平行**
        的边对；三角形顶角是**唯一不与任一平行边共享端点**的顶点。
        返回 ``(顶点, 平行边对索引, 平行角差)``，找不到返回 ``(None, None, None)``。
        """
        pts = approx.reshape(-1, 2).astype(np.float64)
        n = len(pts)
        if n != 5:
            return None, None, None
        ea = [cls._edge_angle_deg(pts[i], pts[(i + 1) % n]) for i in range(n)]
        best_pair, best_diff = None, 180.0
        for i in range(n):
            for j in range(i + 1, n):
                d = cls._angle_diff_deg(ea[i], ea[j])
                if d < best_diff:
                    best_diff, best_pair = d, (i, j)
        if best_pair is None or best_diff >= tol:
            return None, None, None
        i, j = best_pair
        ends = {i, (i + 1) % n, j, (j + 1) % n}
        apex_idx = [k for k in range(n) if k not in ends]
        if len(apex_idx) != 1:
            return None, None, None
        return pts[apex_idx[0]], best_pair, best_diff

    @staticmethod
    def _center_of_polygon(approx) -> list[float]:
        """多边形中心。

        用顶点均值而不是 ``cv2.moments``：OpenCV 5.0 的 ``moments`` 对
        ``float64`` 轮廓直接抛 "Invalid image type (must be single-channel)"
        （``approxPolyDP`` 的输出类型取决于输入，不能保证是 int32），而这里
        只需要一个"用于判断旋转方向"的参考点，凸多边形顶点均值足够且没有
        类型坑。
        """
        points = np.asarray(approx, dtype=np.float64).reshape(-1, 2)
        if points.size == 0:
            return [0.0, 0.0]
        return [float(points[:, 0].mean()), float(points[:, 1].mean())]

    # ------------------------------------------------------------------
    # 五边形检测
    # ------------------------------------------------------------------
    def edge_detection(self) -> tuple[np.ndarray, bool]:
        """按当前 ``s_min`` 提取颜色并找五边形；返回 ``(approx, ok)``。

        确定性扫描 ``epsilon``（3..40）：旧版按 ±1 步进调整 epsilon 会在边界
        附近来回振荡，扫描一遍取第一个成功值既确定又简单。
        """
        image = self.image
        if image is None:
            return np.empty((0, 1, 2), np.int32), False
        blurred = cv2.GaussianBlur(image, (3, 3), 0)
        if self.color == "blue":
            gray = self._extract_blue(blurred, self.s_min)
        elif self.color == "red":
            gray = self._extract_red_level(blurred, self.s_min)
        else:
            raise ValueError(f"未知目标颜色: {self.color}")

        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        erosion = cv2.erode(opening, kernel, iterations=1)
        contours, _ = cv2.findContours(
            erosion, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return np.empty((0, 1, 2), np.int32), False
        contour = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(contour)
        for eps in range(3, 41):
            approx = cv2.approxPolyDP(hull, float(eps), True)
            if len(approx) == 5:
                return approx, True
        return np.empty((0, 1, 2), np.int32), False

    def _detect_pentagon(self, image: np.ndarray) -> tuple[np.ndarray, bool]:
        """饱和度逐级回退找五边形。"""
        self.image = image
        for s_min in self._sat_levels():
            self.s_min = s_min
            approx, ok = self.edge_detection()
            if ok:
                return approx, True
        return np.empty((0, 1, 2), np.int32), False

    # ------------------------------------------------------------------
    # 预处理 / 转正
    # ------------------------------------------------------------------
    @staticmethod
    def _denoise(image: np.ndarray) -> np.ndarray:
        """JPEG 往返去噪：抹平 DVR 帧的细颗粒噪点，稳定轮廓与 OCR。"""
        ok, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ok:
            return image
        decoded = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        return decoded if decoded is not None else image

    @staticmethod
    def _rotate_img(image: np.ndarray, angle_deg: float) -> np.ndarray:
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
        cos, sin = abs(matrix[0, 0]), abs(matrix[0, 1])
        new_w = int(height * sin + width * cos)
        new_h = int(height * cos + width * sin)
        matrix[0, 2] += (new_w / 2) - center[0]
        matrix[1, 2] += (new_h / 2) - center[1]
        return cv2.warpAffine(
            image, matrix, (new_w, new_h), borderValue=(0, 0, 0)
        )

    # ------------------------------------------------------------------
    # OCR 候选
    # ------------------------------------------------------------------
    @staticmethod
    def _collect_candidates(texts: list[OcrText]) -> list[tuple]:
        """把 OCR 文本整理成候选编号。

        单数字框按行聚类拼成两位数（印刷体数字被 det 拆成两个框是常态）；
        其它长度的文本直接作为候选。
        """
        candidates: list[tuple] = []
        singles: list[list] = []
        for item in texts:
            xs = item.box[:, 0]
            ys = item.box[:, 1]
            cx = float(xs.mean())
            cy = float(ys.mean())
            w = float(xs.max() - xs.min())
            h = float(ys.max() - ys.min())
            if len(item.text) == 1 and item.text.isdigit():
                singles.append([cx, cy, w, h, item.text, item.score])
            else:
                candidates.append((cx, cy, item.text, item.score))
        for i in range(len(singles)):
            for j in range(i + 1, len(singles)):
                a, b = singles[i], singles[j]
                # 同一行（纵向中心差 < 较小高度的 40% + 8px）
                if abs(a[1] - b[1]) > 0.4 * min(a[3], b[3]) + 8:
                    continue
                # 横向邻近（中心距 < 2.5 倍平均宽度）
                if abs(a[0] - b[0]) > 2.5 * (a[2] + b[2]) / 2:
                    continue
                left, right = (a, b) if a[0] < b[0] else (b, a)
                candidates.append(
                    (
                        (left[0] + right[0]) / 2,
                        (left[1] + right[1]) / 2,
                        left[4] + right[4],
                        min(left[5], right[5]),
                    )
                )
        return candidates

    def _ocr_candidates(self, rectified: np.ndarray) -> list[tuple]:
        """转正图整图 OCR（彩图，不放大）：PP-OCRv6 det 对灰度图检不出文本框。"""
        texts = self._engine.read_text(rectified)
        return self._collect_candidates(texts)

    def _best_candidate(
        self,
        candidates: list[tuple],
        side_px: float,
        probe: np.ndarray,
        best: CropResult,
        *,
        saturation_level: int,
        stage: str,
    ) -> CropResult:
        """挑最优候选，返回新的 :class:`CropResult`（更差时原样返回 ``best``）。

        两位结果置信度加权 ×1.15、单位 ×0.6：单数字框被读成 ``"7"`` 时我们会
        补成 ``07``，但那种"猜出来"的结果不该压过真正的两位读数。
        """
        for _, _, raw, conf in candidates:
            if not raw or conf < self.ocr_conf_threshold:
                continue
            number = correct_ocr_number(raw)
            if number is None:
                continue
            score = conf * (1.15 if number >= 10 else 0.6)
            if score > best.confidence:
                best = CropResult(
                    number=number,
                    side_px=side_px,
                    rectified=probe,
                    raw_text=raw,
                    confidence=score,
                    saturation_level=saturation_level,
                    stage=stage,
                )
        return best

    # ------------------------------------------------------------------
    # 识别入口
    # ------------------------------------------------------------------
    def recognize(self, image: np.ndarray) -> CropResult:
        """识别一张裁剪图，返回 :class:`CropResult`。

        流程：去噪 → 等比放大到短边 300px → 掩码回退找五边形 → 几何转正 →
        转正图整图 OCR。``side_px`` 会还原成**原始**像素边长供坐标解算使用。
        """
        if image is None or image.size == 0:
            return CropResult(None, -1.0, None, "", 0.0, stage="empty")
        try:
            picture = self._denoise(image)
            height, width = picture.shape[:2]
            scale = min(self.target_side_px / max(min(height, width), 1), self.max_upscale)
            scale = scale if scale > 1.0 else 1.0
            if scale > 1.0:
                picture = cv2.resize(
                    picture, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
                )
            self.scale = scale
            approx, ok = self._detect_pentagon(picture)
            if not ok:
                return CropResult(None, -1.0, None, "", 0.0, stage="no_pentagon")
            result = self._geometry_ocr(picture)
            if result.side_px > 0:
                result = CropResult(
                    number=result.number,
                    side_px=result.side_px / scale,
                    rectified=result.rectified,
                    raw_text=result.raw_text,
                    confidence=result.confidence,
                    saturation_level=result.saturation_level,
                    stage=result.stage,
                )
            return result
        except Exception:  # noqa: BLE001 - 单个目标失败不影响整体
            LOGGER.exception("裁剪图识别异常")
            return CropResult(None, -1.0, None, "", 0.0, stage="error")

    def _geometry_ocr(self, image: np.ndarray) -> CropResult:
        """几何路径：掩码回退 → 五边形 → 转正 → 整图 OCR。"""
        best = CropResult(None, -1.0, None, "", 0.0, stage="none")
        for s_min in self._sat_levels():
            self.s_min = s_min
            approx, ok = self.edge_detection()
            if not ok:
                continue
            side_px = float(cv2.arcLength(approx, True)) / 5.0
            center = self._center_of_polygon(approx)
            apex, pair, _ = self._find_apex_by_parallel(approx)
            if apex is not None:
                row_deg = self._parallel_rectify_angle(approx, pair, apex, center)
                stage = "parallel"
            else:
                point, min_deg = self._find_min_angle(approx)
                if not 40.0 < min_deg < 130.0:
                    continue
                vx = point[0] - center[0]
                vy = point[1] - center[1]
                # 把顶点旋到正上方（-90°，图像 y 向下）
                row_deg = 90.0 + float(np.degrees(np.arctan2(vy, vx)))
                stage = "min_angle"
            rectified = self._rotate_img(image, row_deg)
            candidates = self._ocr_candidates(rectified)
            best = self._best_candidate(
                candidates,
                side_px,
                rectified,
                best,
                saturation_level=s_min,
                stage=stage,
            )
            if best.number is not None:
                return best
        return best

    # ------------------------------------------------------------------
    # 调试入口
    # ------------------------------------------------------------------
    def visualize(self, image: np.ndarray) -> tuple[np.ndarray, CropResult]:
        """排故用：把五边形/转正结果画出来，返回 ``(可视化图, 识别结果)``。"""
        result = self.recognize(image)
        canvas = image.copy()
        if self.image is not None:
            approx, ok = self._detect_pentagon(self.image)
            if ok:
                cv2.drawContours(canvas, [approx], -1, (0, 255, 0), 2)
        if result.ok:
            cv2.putText(
                canvas,
                f"{result.number:02d}",
                (5, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )
        return canvas, result
