"""裁剪图后处理：五边形定位 → 几何转正 → 整图 OCR 读编号。

移植自旧版实现 ``drone/cvproc.py``（久经实测的目标形态与参数）。目标形态：
五边形 = 正方形 + 等边三角形（边长 1m），编号印在正方形中心的白底方块上
（0.6m，黑字）。

流水线（顺序不可随意改动，每一步都在旧版实现里有实测理由）
--------------------------------------------------------
1. JPEG 往返去噪：DVR 图传的细颗粒噪点会干扰轮廓提取，编解码一次即可抹平；
2. 等比放大到短边 300px（上限 10 倍）：小目标识别率显著提升。``side_px``
   最后要除回去，还原成原始像素边长（坐标解算用原始尺度）；
3. 颜色掩码饱和度逐级回退：无人机接近时目标褪色（蓝实测 S≈20~30），
   固定阈值会突然检不到；blue 走 100→60→40→20，red 走 100→80→60→40→20；
4. 形态学 + 凸包 + approxPolyDP 扫 epsilon=3..40 找出五边形。扫描 epsilon
   而不是 ±1 步进，避免旧的振荡问题；
5. 转正：平行边法为主（正方形左右两边是轮廓里唯一互相平行的边对，三角形
   顶角是唯一不与任一平行边共享端点的顶点），失败回退最小内角法（40~130° 门限，
   斜视角会把 60° 投影到这个区间）；
6. 转正图整图 OCR。注意用彩图——PP-OCRv6 的 det 对灰度图检不出文本框
   （旧版实测易错点，P5 未复测，保守沿用）；
7. 候选合并：单数字框按行聚类拼成两位数；两位结果置信度加权（×1.15）、
   单位 ×0.6，防止把 ``"7"`` 误报成 ``07``；
8. 编号交给 :func:`~airdrop.perception.number.correct_ocr_number` 纠错。

与旧版实现的差异（都是被 rapidocr 3.9.2 的 API 逼的，不是主动改的）
------------------------------------------------------------------
* 旧版实现的 ``_rapid_readtext`` 往 ``__call__`` 上传 ``allowlist`` / ``low_text``
  / ``text_threshold`` / ``batch_size`` 等参数；3.9.2 的 ``__call__`` 只接受
  图像，这些参数已不存在。替代方案：
  - 置信度过滤交给 ``Global.text_score``（构造时设定，等价于低阈值放行 +
    自己按 ``ocr_conf_threshold`` 复筛）；
  - ``allowlist`` 没有对应配置项，改由 :func:`correct_ocr_number` 过滤非数字
    （它本来就在做这件事，且更严格）；
  - ``rec_batch_num`` 走 ``Rec.rec_batch_num`` 配置。
* 方向分类（Cls）的权重文件名保持 RapidOCR 原样（TORCH 版是
  ``ch_ptocr_mobile_v2.0_cls_mobile.pth``，引擎按文件名 stem 查架构）；
  用哪个引擎由 :class:`OcrEngineConfig` 的 ``cls_engine`` 决定（默认 ONNX）。
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

__all__ = [
    "CropResult",
    "OcrEngine",
    "OcrEngineConfig",
    "OcrOrientation",
    "OpenCvPostProcess",
]


# ----------------------------------------------------------------------
# RapidOCR 引擎封装
# ----------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class OcrEngineConfig:
    """RapidOCR（TORCH 引擎）参数。

    默认值来自 P5 复测：PP-OCRv6 medium det+rec 在 RTX 3060 Laptop 上
    单帧约 97ms（GPU）/ 9.2s（CPU，即 GPU 约 95 倍），初始化约 2.0s；
    small 组合约 90ms 但会把五边形边缘多认出一个字（``'一'``），更吵。
    """

    model_dir: str = "models/ppocr"
    det_model: str = "PP-OCRv6_det_medium.pth"
    rec_model: str = "PP-OCRv6_rec_medium.pth"
    rec_keys: str = "ppocrv6_dict.txt"
    #: 方向分类（Cls）用哪个引擎：``"onnx"`` / ``"torch"`` / ``"off"``。
    #: 默认 ONNX：实测（同权重、6 张一批）ONNX-CUDA 0.96ms/张、ONNX-CPU 1.28、
    #: TORCH-CUDA 2.28、TORCH-CPU 4.14，而三者判定逐位相同（同分数 0.9801/0.9799）。
    cls_engine: str = "onnx"
    cls_onnx_model: str = "ch_ppocr_mobile_v2.0_cls_mobile.onnx"
    #: 文件名保持 RapidOCR 原拼写（``ptocr``），不要规范化
    cls_torch_model: str = "ch_ptocr_mobile_v2.0_cls_mobile.pth"
    #: Cls 是否用 CUDA。⚠ ONNX 的 CUDA EP 依赖 torch 先被 import
    #: （torch/lib 里的 cuDNN/cuBLAS 才在进程搜索路径上），否则 onnxruntime 会静默
    #: 退回 CPU；本项目 det/rec 走 TORCH 引擎，天然满足，但仍会显式核对并告警。
    cls_use_cuda: bool = True
    #: 判定为 180° 的置信度门限（低置信度不动图，只记录判定）
    cls_thresh: float = 0.9
    #: 是否让 OCR 流水线按判定自动旋转文本行（``Global.use_cls``）。
    #: 关掉时方向判别仍可用（:meth:`OcrEngine.classify_orientation`），只是不参与识别，
    #: 适合"先只观测、不改结果"的交叉验证。
    cls_autorotate: bool = True
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

    def cls_onnx_path(self) -> Path:
        return Path(self.model_dir) / self.cls_onnx_model

    def cls_torch_path(self) -> Path:
        return Path(self.model_dir) / self.cls_torch_model

    def cls_path(self) -> Path:
        """按 :attr:`cls_engine` 选对应的权重路径。"""
        if self.cls_engine == "torch":
            return self.cls_torch_path()
        return self.cls_onnx_path()

    def cls_enabled(self) -> bool:
        return self.cls_engine != "off"


@dataclass(frozen=True, slots=True)
class OcrOrientation:
    """方向判别结果：``label`` 是 ``"0"``（正立）或 ``"180"``（倒置）。"""

    label: str
    score: float

    @property
    def flipped(self) -> bool:
        return self.label == "180"


@dataclass(frozen=True, slots=True)
class OcrText:
    """一条 OCR 文本结果（框 + 内容 + 置信度）。"""

    box: np.ndarray  # (4, 2) float
    text: str
    score: float


class OcrEngine:
    """``rapidocr.RapidOCR`` 的薄封装：显式模型路径 + 统一结果形状。

    为什么不用库的自动下载：起飞前必须固定模型版本（否则某次飞行前悄悄换了
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
            raise RuntimeError("OCR 引擎初始化失败，请检查模型文件与 CUDA")
        config = self._config
        missing = [
            str(path)
            for path in (config.det_path(), config.rec_path(), config.keys_path())
            if not path.is_file()
        ]
        if config.cls_enabled() and not config.cls_path().is_file():
            missing.append(str(config.cls_path()))
        if missing:
            self._failed = True
            raise FileNotFoundError(
                "OCR 模型文件缺失: " + ", ".join(missing) + "（见 OcrEngineConfig.model_dir）"
            )
        try:
            from rapidocr import EngineType, RapidOCR  # 惰性：重型依赖

            params: dict[str, Any] = {
                "Det.engine_type": EngineType.TORCH,
                "Det.model_path": str(config.det_path()),
                "Rec.engine_type": EngineType.TORCH,
                "Rec.model_path": str(config.rec_path()),
                "Rec.rec_keys_path": str(config.keys_path()),
                "Rec.rec_batch_num": config.rec_batch_num,
                "Global.text_score": config.text_score,
                "EngineConfig.torch.use_cuda": config.use_cuda,
                "EngineConfig.torch.cuda_ep_cfg.device_id": config.device_id,
                "Global.log_level": config.log_level,
            }
            params.update(self._cls_params(EngineType))
            self._engine = RapidOCR(params=params)
        except Exception:
            self._failed = True
            raise
        LOGGER.info(
            "OCR 引擎就绪（TORCH，det=%s rec=%s cuda=%s）",
            config.det_model,
            config.rec_model,
            config.use_cuda,
        )
        self._report_cls_device()
        return self._engine

    def _cls_params(self, engine_type: Any) -> dict[str, Any]:
        """方向分类（Cls）的构造参数；``cls_engine="off"`` 只关自动旋转、不加载权重。"""
        config = self._config
        if not config.cls_enabled():
            # rapidocr 会无条件构造 det/cls/rec 三个模块（Global.use_cls 只管流水线），
            # 所以"关掉"也得给它一个本地路径：Cls.model_path 为空时它会去 modelscope
            # 下载默认 Cls 模型——离线机器上就是启动失败。
            params: dict[str, Any] = {"Global.use_cls": False}
            local = next(
                (
                    path
                    for path in (config.cls_onnx_path(), config.cls_torch_path())
                    if path.is_file()
                ),
                None,
            )
            if local is not None:
                params["Cls.model_path"] = str(local)
            return params
        if config.cls_engine == "onnx":
            device = {
                "Cls.engine_type": engine_type.ONNXRUNTIME,
                "EngineConfig.onnxruntime.use_cuda": config.cls_use_cuda,
            }
        elif config.cls_engine == "torch":
            device = {
                "Cls.engine_type": engine_type.TORCH,
                "EngineConfig.torch.use_cuda": config.cls_use_cuda,
            }
        else:
            raise ValueError(f"未知的 cls_engine: {config.cls_engine!r}（应为 onnx / torch / off）")
        return {
            "Cls.model_path": str(config.cls_path()),
            "Cls.cls_thresh": config.cls_thresh,
            "Global.use_cls": config.cls_autorotate,
            **device,
        }

    def cls_providers(self) -> tuple[str, ...]:
        """方向分类会话实际拿到的执行提供者（空元组 = 未启用或取不到）。

        ⚠ 这只是读取，不会触发加载：引擎还没 :meth:`ensure` 时必然返回空元组。
        要核对设备，先 ``ensure()``（起飞前 :meth:`warmup` 也是这个时机）。
        """
        classifier = getattr(self._engine, "text_cls", None)
        session = getattr(getattr(classifier, "session", None), "session", None)
        if session is None:
            return ()
        try:
            return tuple(session.get_providers())
        except Exception:  # noqa: BLE001 - 取不到 provider 列表就当空，只用于日志
            return ()

    def _report_cls_device(self) -> None:
        """核对方向分类真正跑在哪——onnxruntime 的 CUDA EP 会静默退回 CPU。"""
        config = self._config
        if not config.cls_enabled():
            return
        providers = self.cls_providers()
        if not providers:
            # TORCH 引擎没有 provider 概念（会话对象不同），只有 onnx 取不到才值得告警
            if config.cls_engine == "onnx":
                LOGGER.warning("方向分类已启用（onnx），但取不到会话 provider，无法核对设备")
            else:
                LOGGER.info(
                    "OCR 方向分类就绪（engine=%s model=%s autorotate=%s）",
                    config.cls_engine,
                    config.cls_path().name,
                    config.cls_autorotate,
                )
            return
        LOGGER.info(
            "OCR 方向分类就绪（engine=%s device=%s model=%s autorotate=%s）",
            config.cls_engine,
            providers[0],
            config.cls_path().name,
            config.cls_autorotate,
        )
        if (
            config.cls_engine == "onnx"
            and config.cls_use_cuda
            and providers[0] != "CUDAExecutionProvider"
        ):
            LOGGER.warning(
                "方向分类想要 CUDA，实际落在 %s：onnxruntime 的 CUDA EP 依赖进程里"
                "先 import torch（torch/lib 自带 cuDNN/cuBLAS，ORT 才找得到），"
                "否则它会无提示地退回 CPU。本项目 det/rec 走 TORCH 引擎，正常顺序下已满足；"
                "看到这条说明加载顺序被改了（功能仍可用，只是慢一些）。",
                providers[0],
            )

    def classify_orientation(self, image: np.ndarray) -> OcrOrientation:
        """判断一张文本行是否倒置——独立于几何转正的第二判据。

        ⚠ 输入应当是文本行（编号框那种扁长条），不是整幅转正图：
        ``ch_ppocr_mobile_v2.0_cls`` 是按文本行训练的（内部缩放到 48×192）。

        用途：给几何转正做交叉验证。两条路用的信息完全不同——几何走五边形形状，
        判别走学出来的笔画方向——一致才说明转正可信；不一致的地方就是转正的疑点。
        """
        if not self._config.cls_enabled():
            raise RuntimeError("未启用方向分类（OcrEngineConfig.cls_engine='off'）")
        engine = self.ensure()
        classifier = getattr(engine, "text_cls", None)
        if classifier is None:
            raise RuntimeError("OCR 引擎里没有方向分类模块")
        output = classifier(image)
        label, score = output.cls_res[0]
        return OcrOrientation(label=str(label), score=float(score))

    def warmup(self, size: int = 320) -> bool:
        """跑一次空图，把模型加载/CUDA 上下文开销挪到起飞前。"""
        try:
            engine = self.ensure()
            blank = np.full((size, size, 3), 255, np.uint8)
            engine(blank)
            return True
        except Exception:
            LOGGER.exception("OCR 预热失败（不影响后续识别，但首帧会变慢）")
            return False

    def read_text(self, image: np.ndarray) -> list[OcrText]:
        """整图识别，返回文本列表；引擎不可用或识别失败时返回空列表。

        绝不抛出：单帧 OCR 失败不该打断整条拉流/处理链路，上层按"没读出
        编号"处理即可（``Detection.code is None``）。
        """
        try:
            engine = self.ensure()
        except Exception:
            LOGGER.exception("OCR 引擎不可用")
            return []
        try:
            result = engine(image)
        except Exception:
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
    stage: str = ""  # "parallel" / "min_angle" / "none"，排故用
    #: 五边形是否通过"正方形 + 等边三角形"的形态判据（顶角 ≈60°）。
    #: False 只表示这次"转正"的几何不成立、结果不可尽信，不代表一定读错
    house_ok: bool = False

    @property
    def ok(self) -> bool:
        return self.number is not None


# ----------------------------------------------------------------------
# 后处理主类
# ----------------------------------------------------------------------
class OpenCvPostProcess:
    """目标裁剪图后处理：颜色掩码 → 凸包五边形 → 转正 → 整图 OCR。

    参数命名与旧版实现对位（``color`` / ``ocr_conf_threshold``），便于对照实测值；
    逻辑按新的 RapidOCR 接口重写，几何部分逐行保留。
    """

    #: "房子"形态判据：等边三角形的内角恒为 60°，所以顶角就是 60°
    HOUSE_APEX_ANGLE_DEG = 60.0
    #: 顶角允许的偏差（透视、掩码噪声、approxPolyDP 都会让量出来的角偏离 60°）
    HOUSE_APEX_ANGLE_TOL_DEG = 15.0

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
        mask = cv2.inRange(hsv, np.array([90, s_min, 50]), np.array([130, 255, 255]))
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
        """最小内角顶点（三角形顶角；目标形态下该角为 60°）。

        必须用"顶点自身 + 它的前后邻居"来量角度。原实现写成
        ``_angle(pts[i], pts[i+1], pts[i+2])`` 并把 ``pts[i+1]`` 当作该角所在顶点，
        这看着像"顶点+邻居"，其实不是：``pts[i]`` 与 ``pts[i+2]`` 只是"相隔两个"的
        点，当它们恰好不是顶点的邻居时会量出一个根本不属于该顶点的角。
        五边形上这一错会把 60° 的顶角量成 15°（用 ``bl``—``apex``—``tl`` 代替
        真正的 ``tl``—``apex``—``tr``），进而被 40~130° 门限误拒。
        """
        points = np.asarray(approx, dtype=np.float64).reshape(-1, 2)
        length = len(points)
        min_angle, min_point = 360.0, [0.0, 0.0]
        for i in range(length):
            current = points[i]
            t = cls._angle(
                points[(i - 1) % length],
                current,
                points[(i + 1) % length],
            )
            if t < min_angle:
                min_angle = t
                min_point = [float(current[0]), float(current[1])]
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
        ``α_after = α - θ``（旧版实验确认），所以"转竖直"用 ``row = ea - 90``。
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
        theta = float(np.degrees(np.arctan2(apex[1] - center[1], apex[0] - center[0])))
        after = cls._angle_norm_deg(theta - row)
        if abs(after - 90.0) < abs(after + 90.0):  # 更接近正下方（+90°）
            row += 180.0
        return row

    @classmethod
    def _find_apex_by_parallel(cls, approx, tol: float = 15.0):
        """平行边法定顶点。

        五边形 = 正方形 + 等边三角形：正方形的左右两边是轮廓中唯一互相平行
        的边对；三角形顶角是唯一不与任一平行边共享端点的顶点。
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

    @classmethod
    def _interior_angles_deg(cls, approx) -> list[float]:
        """每个顶点的真实内角（顶点 + 它的真实前后邻居）。"""
        points = np.asarray(approx, dtype=np.float64).reshape(-1, 2)
        length = len(points)
        return [
            cls._angle(points[(i - 1) % length], points[i], points[(i + 1) % length])
            for i in range(length)
        ]

    @classmethod
    def _is_house_pentagon(cls, approx, apex) -> bool:
        """这个五边形是不是"正方形 + 架在上边的等边三角形"。

        判据只用一条几何事实：顶角恒为 60°（等边三角形内角）。为什么需要它——
        平行边法只验证"存在一对平行边"，而掩码偏严时圈到的可能只是编号底板
        （近方形），approxPolyDP 照样能凑出 5 个顶点、也照样有一对平行边，
        于是算出一个任意角度的"转正"，OCR 还可能读出一个自信的错编号：
        实测帧 1840 的 s_min=100 就是这样把 56 转成了 95（同一帧其余三级都读出
        正确的 56）。顶角是房子特有的、且与旋转无关的量，正好拿来当门限。
        """
        points = np.asarray(approx, dtype=np.float64).reshape(-1, 2)
        target = np.asarray(apex, dtype=np.float64).reshape(2)
        index = int(np.argmin(np.linalg.norm(points - target, axis=1)))
        angle = cls._interior_angles_deg(approx)[index]
        return abs(angle - cls.HOUSE_APEX_ANGLE_DEG) <= cls.HOUSE_APEX_ANGLE_TOL_DEG

    @staticmethod
    def _center_of_polygon(approx) -> list[float]:
        """多边形中心。

        用顶点均值而不是 ``cv2.moments``：OpenCV 5.0 的 ``moments`` 对
        ``float64`` 轮廓直接抛 "Invalid image type (must be single-channel)"
        （``approxPolyDP`` 的输出类型取决于输入，不能保证是 int32），而这里
        只需要一个"用于判断旋转方向"的参考点，凸多边形顶点均值足够且没有
        类型易错点。
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
        contours, _ = cv2.findContours(erosion, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
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
        return cv2.warpAffine(image, matrix, (new_w, new_h), borderValue=(0, 0, 0))

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
        house_ok: bool = False,
    ) -> CropResult:
        """挑最优候选，返回新的 :class:`CropResult`（更差时原样返回 ``best``）。

        两位结果置信度加权 ×1.15、单位 ×0.6：单数字框被读成 ``"7"`` 时我们会
        补成 ``07``，但那种"猜出来"的结果不该压过真正的两位读数。

        形态成立的候选优先，其次才比置信度：几何不成立的"转正"再自信也不能
        压过真正的房子——实测帧 1840 里错转正读出的 ``95`` 加权分（1.15）确实
        高于正确转正读出的 ``56``，只看置信度就会选错。
        """
        for _, _, raw, conf in candidates:
            if not raw or conf < self.ocr_conf_threshold:
                continue
            number = correct_ocr_number(raw)
            if number is None:
                continue
            score = conf * (1.15 if number >= 10 else 0.6)
            if house_ok and not best.house_ok:
                better = True
            elif house_ok == best.house_ok:
                better = score > best.confidence
            else:
                better = False
            if better:
                best = CropResult(
                    number=number,
                    side_px=side_px,
                    rectified=probe,
                    raw_text=raw,
                    confidence=score,
                    saturation_level=saturation_level,
                    stage=stage,
                    house_ok=house_ok,
                )
        return best

    # ------------------------------------------------------------------
    # 识别入口
    # ------------------------------------------------------------------
    def recognize(self, image: np.ndarray) -> CropResult:
        """识别一张裁剪图，返回 :class:`CropResult`。

        流程：去噪 → 等比放大到短边 300px → 掩码回退找五边形 → 几何转正 →
        转正图整图 OCR。``side_px`` 会还原成原始像素边长供坐标解算使用。
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
            _, ok = self._detect_pentagon(picture)
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
                    house_ok=result.house_ok,
                )
            return result
        except Exception:
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
            house_ok = False
            if apex is not None:
                row_deg = self._parallel_rectify_angle(approx, pair, apex, center)
                stage = "parallel"
                # 平行边法只保证"存在一对平行边"，不保证这是房子——掩码偏严时
                # 圈到的可能只是编号底板（近方形），照样凑得出 5 个顶点和一对
                # 平行边，算出的转正角却是任意的。用顶角（恒 60°）把这类挡掉。
                house_ok = self._is_house_pentagon(approx, apex)
            else:
                point, min_deg = self._find_min_angle(approx)
                if not 40.0 < min_deg < 130.0:
                    continue
                vx = point[0] - center[0]
                vy = point[1] - center[1]
                # 把顶点旋到正上方（-90°，图像 y 向下）
                row_deg = 90.0 + float(np.degrees(np.arctan2(vy, vx)))
                stage = "min_angle"
                # 兜底路径量到的也是顶角，只是门限宽（40~130°）；严格成立仍要求 ≈60°
                house_ok = abs(min_deg - self.HOUSE_APEX_ANGLE_DEG) <= self.HOUSE_APEX_ANGLE_TOL_DEG
            rectified = self._rotate_img(image, row_deg)
            candidates = self._ocr_candidates(rectified)
            best = self._best_candidate(
                candidates,
                side_px,
                rectified,
                best,
                saturation_level=s_min,
                stage=stage,
                house_ok=house_ok,
            )
            # ⚠ 只有形态成立的结果才提前返回。形态不成立时继续往下试更宽松的
            # 饱和度：真正的房子常常在那些级别才露出来（帧 1840 就是这样——严格
            # 级别给出假的"95"并一度抢先返回，真房子在 s_min=60 给出正确的 56）。
            if best.number is not None and best.house_ok:
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
