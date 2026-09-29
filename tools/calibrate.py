"""三步标定流水线：内参 → 画面/遥测时间差 → 手眼外参。

用法（输入 = ``FlightRecorder`` 录出的飞行目录，见计划第 6 章）::

    ./.venv/Scripts/python.exe -m airdrop.run calibrate --flight flights/20260913-185512
    ./.venv/Scripts/python.exe -m airdrop.run calibrate --help

本文件是**纯库模块**：顶部常量是默认值，:class:`CalibrateConfig` / :func:`build_config`
/ :func:`main` 都能按关键字覆盖；**命令行解析集中在 `airdrop/run.py`**（本文件不 import
argparse）。重依赖 ``cv2`` 只在各个真正做标定的函数体内导入，所以 `--help` 不加载 OpenCV。

采集方式（计划第 6 章）
----------------------
棋盘格**固定**放置，手持飞机在棋盘格上方缓慢平移 + 旋转，让棋盘格全程可见且有
充分的**旋转激励**。录制用 ``examples/calibration_capture.py``（复用
FlightRecorder 的标准飞行目录）。

⚠ 这里的关键是"**板子不动、飞机在动**"：于是棋盘格相对机体的姿态**逐帧都在变**，
``A`` 与 ``B`` 的旋转块才有区别、``AX = XB`` 才有解。反过来若把机体位姿写成
"棋盘格位姿左乘一个常量"（不管那个常量是不是真值），相对旋转里的常量会被消掉，
``R_a ≡ R_b``，方程**退化成只剩单位矩阵解**——详见 :func:`solve_hand_eye`。

三步为什么是这个顺序
--------------------
1. **内参**：棋盘格多视图 Zhang 标定，得到 ``K``、畸变；
2. **时间差 Δt**：用**刚体角速度模长与安装角无关**这条性质（``|ω_cam| = |ω_body|``），
   互相关画面角速度与飞控角速度得到画面-遥测的时间差。**必须先于外参**——它对外参零依赖；
3. **外参**：把棋盘格位姿与**同一时刻**的遥测位姿配成对（:func:`align_body_poses`），
   再解 ``AX = XB`` 求相机→机体旋转。

使用说明
--------
* ``t_bc``（杆臂）在手持采集下估不准：``(R_a − I)t_x = R_x t_b − t_a`` 里的
  ``t_a`` 只能来自遥测位置，而室内无 GPS 时它不可信。所以按计划"旋转以标定为准、
  平移以尺量为准"：标定文件里的 ``t_bc`` 写尺量值 :data:`MEASURED_T_BC`，
  估计值（:func:`solve_hand_eye` 的平移分支）照算，但只写进 ``meta`` 供**对比校验**
  ——两者应当同量级、逐轴接近；差得离谱正是"遥测位置不可用"的直接证据
  （:func:`run_extrinsics` 同时给出残差判据与 ``estimated_reliable``）。
* 互相关只能分辨到采样间隔量级；要更高精度需要 Kalibr 式样条联合优化（计划列为
  可选增强，未进首版）。
* **``cv2.calibrateHandEye`` 在 Python 里不可用，但原因不是"被移除"**：
  OpenCV 5.0 把 ``calib3d`` 拆成 ``geometry`` / ``calib`` / ``stereo`` /
  ``ptcloud``，函数搬进 ``modules/calib``，**C++ 声明与签名一字未改**，
  但绑定标记从 ``CV_EXPORTS_W`` 变成了 ``CV_EXPORTS``（实测 5.0.0 的
  ``calib.hpp``，同文件里 ``calibrateCamera`` 等仍带 ``_W``），于是 Python 侧没生成。
  旁证：``cv2/__init__.pyi`` 里 ``CALIB_HAND_EYE_*`` 与 ``HandEyeCalibrationMethod``
  都在，唯独没有函数；``cv2.calibrateHandEye`` 与 ``cv2.calibrateRobotWorldHandEye``
  运行时都不存在。
  ``opencv-contrib-python`` **解决不了**这个问题——contrib 用的是同一份主仓源码，
  一样没有 ``_W``。所以第三步是自己实现（:func:`solve_hand_eye`，Horaud 向量化），
  合成数据上验到机器精度。
"""

from __future__ import annotations

import json
import logging
import math
from bisect import bisect_left
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from airdrop.georef import rotation_z
from airdrop.telemetry.models import TelemetrySnapshot

LOGGER = logging.getLogger("calibrate")

# ----------------------------------------------------------------------
# 配置（改这里，或走命令行）
# ----------------------------------------------------------------------
FLIGHT_DIR = Path("flights")  # 标定采集目录（FlightRecorder 产出）
OUTPUT_PATH = Path("camera_calib.json")
#: 棋盘格内角点数（列, 行）——注意是**内角点**，不是方格数
PATTERN_SIZE = (9, 6)
#: 方格边长（米），用于 PnP 的度量和 t_bc 估计
SQUARE_SIZE_M = 0.025
#: 参与内参标定的最少/最多视图数
MIN_VIEWS = 8
MAX_VIEWS = 60
#: 一次标定最多读取多少帧
MAX_FRAMES = 200
#: 严格模式：外参/时间差没标出来时返回非 0（默认只警告，照样写文件）
STRICT = False
#: 时间差搜索范围（秒）与步长
LAG_SEARCH_S = 0.5
LAG_STEP_S = 0.005
#: 运动对的最小旋转激励（度）：低于此值的帧对对手眼标定无信息
MIN_PAIR_ROTATION_DEG = 5.0
#: 相对旋转**轴**的最小张角（度）。OpenCV 文档：至少要 2 个不平行的旋转轴才有唯一解；
#: 所有相对旋转都绕同一根轴时，解会沿该轴自由（经典退化），必须显式报错而不是给个解
MIN_AXIS_SPREAD_DEG = 15.0
#: 参与解算的运动对上限（全对是 O(n²)；按固定步长抽样，保证确定性）
MAX_MOTION_PAIRS = 2000
#: 尺量杆臂（米），标定文件里的 t_bc 就用它（手持标定估不准平移）
MEASURED_T_BC = (0.0, 0.0, 0.0)
#: 杆臂**估计值**可采信的判据：平移方程残差（米）≤ max(下限, 该比例 × 画面平移跨度)。
#: ``t_a`` 是真是假都解得出一个 ``t_x``，区别在残差——所以判据必须落在残差上。
LEVER_ARM_RESIDUAL_RATIO = 0.05
LEVER_ARM_RESIDUAL_FLOOR_M = 0.02


# ----------------------------------------------------------------------
# 结果数据类
# ----------------------------------------------------------------------
@dataclass
class IntrinsicsResult:
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    image_size: tuple[int, int]
    rms: float
    views: int
    per_view_errors: list[float] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.views >= 3 and math.isfinite(self.rms)


@dataclass
class LagResult:
    lag_s: float
    peak: float
    samples: int
    #: 互相关曲线（偏移秒 → 归一化相关系数），供报告与人工核对
    curve: list[tuple[float, float]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.samples > 0 and math.isfinite(self.lag_s)


@dataclass
class ExtrinsicsResult:
    r_bc: np.ndarray
    #: 相机光心在机体系的位置（杆臂）**估计值**。手持采集时遥测位置不可信，这个值
    #: **只用于与尺量值对比校验**，能否采信看 :attr:`t_bc_reliable`；
    #: 写进标定文件的 ``t_bc`` 始终是尺量值
    t_bc: np.ndarray | None
    pairs: int
    residual_rotation_deg: float
    #: 相对旋转轴的最大张角（度），用来判断激励是否只绕一根轴
    axis_spread_deg: float = 0.0
    #: 杆臂估计是否可采信（平移方程自洽），判据见 :func:`run_extrinsics`
    t_bc_reliable: bool = False
    #: 平移方程 ``(R_a−I)t_x + t_a − R_x t_b`` 的最大残差（米）
    residual_translation_m: float = float("nan")
    #: 遥测位置跨度（米，包围盒对角线）——与实际运动量对比即可看出遥测位置是否可用
    body_span_m: float = 0.0
    #: 画面（PnP）位置跨度（米）
    camera_span_m: float = 0.0
    methods: dict[str, float] = field(default_factory=dict)  # 方法名 → 残差（度）

    @property
    def ok(self) -> bool:
        return self.pairs >= 3 and math.isfinite(self.residual_rotation_deg)


# ----------------------------------------------------------------------
# 棋盘格
# ----------------------------------------------------------------------
def board_object_points(
    pattern_size: tuple[int, int] = PATTERN_SIZE,
    square_size: float = SQUARE_SIZE_M,
) -> np.ndarray:
    """棋盘格内角点的棋盘坐标系坐标（z=0 平面），形状 (N,1,3)。"""
    cols, rows = pattern_size
    points = np.zeros((cols * rows, 3), np.float64)
    points[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    points *= float(square_size)
    return points.reshape(-1, 1, 3)


def detect_corners(
    image: np.ndarray,
    pattern_size: tuple[int, int] = PATTERN_SIZE,
    *,
    refine: bool = True,
) -> np.ndarray | None:
    """找棋盘格内角点；找不到返回 None。

    先用 ``findChessboardCornersSB``（更稳、自带亚像素），失败再退回经典
    ``findChessboardCorners`` + ``cornerSubPix``。
    """
    import cv2  # 惰性导入：重依赖，只在真正做标定时加载（见模块 docstring）

    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    found = None
    if hasattr(cv2, "findChessboardCornersSB"):
        flags = cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
        ok, corners = cv2.findChessboardCornersSB(gray, pattern_size, flags=flags)
        if ok:
            found = corners.astype(np.float64).reshape(-1, 1, 2)
    if found is None:
        ok, corners = cv2.findChessboardCorners(
            gray,
            pattern_size,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE,
        )
        if not ok:
            return None
        found = corners
        if refine:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-4)
            found = cv2.cornerSubPix(gray, found, (11, 11), (-1, -1), criteria)
    return found.reshape(-1, 1, 2)


def pattern_phase_ok(
    image: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    *,
    pattern_size: tuple[int, int] = PATTERN_SIZE,
    square_size: float = SQUARE_SIZE_M,
) -> float:
    """用**方格明暗相位**校验渲染/位姿的几何一致性，返回相符比例（0~1）。

    用途只有一个：**发现"棋盘格整体错格"这类几何错误**。例如 :func:`render_board`
    的 ``to_board`` 少减一个 ``tile`` 时，棋盘格相对对象点整体错开一整格，
    此时按给定位姿把方格中心投影回图里，落到的是**反相**的格子，相符率会掉到 0.5 以下
    （实测正确渲染是 1.000）。重投影误差对这类错误**完全不敏感**（仍有 0.1 px），
    只有相位能发现。

    ⚠ **它不能区分"正序/反序标签"**：棋盘格（格数无论奇偶）绕板面法向转 180° 后图像
    一模一样，两种标签分配的相位相符率都是 1.000（实测）。标签二义性要靠
    :func:`solve_board_poses` 的**相邻帧连续性**解决，别指望这个函数。

    阈值取"所有采样点的中值"，所以不依赖绝对灰度（真实照片的曝光差异不影响）。
    """
    import cv2  # 惰性导入：重依赖，只在真正做标定时加载（见模块 docstring）

    cols, rows = pattern_size
    centres: list[tuple[float, float, float]] = []
    want_light: list[bool] = []
    for i in range(-1, cols):
        for j in range(-1, rows):
            centres.append(((i + 0.5) * square_size, (j + 0.5) * square_size, 0.0))
            want_light.append((i + j) % 2 == 0)
    points = np.asarray(centres, dtype=np.float64).reshape(-1, 1, 3)
    projected, _ = cv2.projectPoints(points, rvec, tvec, camera_matrix, dist_coeffs)
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    samples: list[tuple[float, bool]] = []
    for (u, v), expect_light in zip(projected.reshape(-1, 2), want_light, strict=True):
        x, y = int(round(float(u))), int(round(float(v)))
        if x < 3 or y < 3 or x >= gray.shape[1] - 4 or y >= gray.shape[0] - 4:
            continue
        patch = gray[y - 3 : y + 4, x - 3 : x + 4]
        samples.append((float(patch.mean()), expect_light))
    if not samples:
        return 0.0
    values = [value for value, _ in samples]
    threshold = 0.5 * (min(values) + max(values))
    matched = sum(1 for value, expect in samples if (value > threshold) == expect)
    return matched / len(samples)


def solve_board_poses(
    images: Sequence[np.ndarray],
    object_points: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    *,
    pattern_size: tuple[int, int] = PATTERN_SIZE,
    square_size: float = SQUARE_SIZE_M,
) -> list[tuple[np.ndarray, np.ndarray] | None]:
    """逐帧检测 + 解位姿；用**相邻帧连续性**把"正序/反序标签"统一到同一支。

    为什么必须统一
    --------------
    棋盘格绕板面法向转 180° 后图像**完全一样**（方格数无论奇偶都如此），所以单帧图像
    根本无法判断"哪个角点是第 0 号"：两种标签分配都拟合到亚像素（实测 PnP 残差都是
    0.1 px），**方格相位也完全一致**（实测两者相符率都是 1.000）。

    但两种分配的位姿恰好相差"绕板面法向 180°"，那是一个**常量右乘** ``R_b``::

        P'_i = P_i·R_b   ⇒   B'_ij = P'_j P'_i⁻¹ = P_j P_i⁻¹ = B_ij

    也就是说：**只要全局用同一支，``AX=XB`` 的解一模一样，选哪支都无所谓；
    混用才会让解算崩掉**（残差会飙到几十度）。这里用"与上一帧位姿夹角最小"来保证
    一致——板子固定、飞机连续运动，真姿态不可能帧间翻 180°。

    返回与 ``images`` 等长的列表，检不出角点的位置为 ``None``。
    """
    import cv2  # 惰性导入：重依赖，只在真正做标定时加载（见模块 docstring）

    poses: list[tuple[np.ndarray, np.ndarray] | None] = []
    previous: np.ndarray | None = None
    for image in images:
        corners = detect_corners(image, pattern_size)
        if corners is None:
            poses.append(None)
            continue
        candidates: list[tuple[np.ndarray, np.ndarray]] = []
        for candidate_corners in (corners, corners[::-1]):
            ok, rvec, tvec = cv2.solvePnP(
                object_points,
                candidate_corners,
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_IPPE,
            )
            if ok:
                candidates.append((rvec, tvec))
        if not candidates:
            poses.append(None)
            continue
        if previous is None:
            chosen = candidates[0]
        else:
            chosen = min(
                candidates,
                key=lambda item: rotation_angle_deg(
                    _pose_matrix(item[0], item[1])[:3, :3].T @ previous
                ),
            )
        previous = _pose_matrix(chosen[0], chosen[1])[:3, :3].copy()
        poses.append(chosen)
    return poses


# ----------------------------------------------------------------------
# 合成数据（自洽验证用；也可用于离线演练）
# ----------------------------------------------------------------------
def render_board(
    rvec: np.ndarray,
    tvec: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    image_size: tuple[int, int],
    *,
    pattern_size: tuple[int, int] = PATTERN_SIZE,
    square_size: float = SQUARE_SIZE_M,
    pixels_per_square: int = 70,
    margin: int = 40,
    background: int = 20,
    tile_value: int = 235,
) -> np.ndarray:
    """把棋盘格按给定位姿渲染成一张图（用于标定流水线的合成自洽验证）。

    实现方式：在棋盘坐标系里先做一次正投影得到"正视模板"，再用
    ``cv2.getPerspectiveTransform`` 把模板的四角映射到真实位姿下的四角——
    对 z=0 平面来说这就是精确的透视变换（不是近似）。

    **配色必须是"深底 + 亮格"（左上角为深格）**：``findChessboardCorners`` 依赖
    标准棋盘格的明暗极性，反了会直接检不出（实测：极性反了 SB 与经典版
    双双返回 False，而 ``checkChessboard`` 仍为 True——说明几何没问题、只是极性）。
    """
    import cv2  # 惰性导入：重依赖，只在真正做标定时加载（见模块 docstring）

    cols, rows = pattern_size
    tile = int(pixels_per_square)
    # 模板**不带外框边距**：模板边界就是棋盘的物理边界（(cols+1)×(rows+1) 个格子）。
    # 早期版本在四周留了 margin，结果背景与边格同色，"棋盘边界"在视觉上不可辨
    # （图上边一排亮格像是凸出来的标签），findChessboardCorners 直接检不出——
    # 标准棋盘必须是干净的矩形边界。
    template_w = (cols + 1) * tile
    template_h = (rows + 1) * tile
    template = np.full((template_h, template_w), background, np.uint8)
    for row in range(rows + 1):
        for col in range(cols + 1):
            if (row + col) % 2:
                continue
            cv2.rectangle(
                template,
                (col * tile, row * tile),
                (
                    min((col + 1) * tile - 1, template_w - 1),
                    min((row + 1) * tile - 1, template_h - 1),
                ),
                tile_value,
                -1,
            )

    # 模板四角在棋盘坐标系里的位置。
    # ⚠ **(0,0) 对应的是棋盘的第一个"内角点"，不是模板左上角像素**：一格 = 一个方格，
    # 而内角点在格子交点上，所以模板像素 (tile, tile) 才是棋盘坐标 (0,0)。
    # 少了这个 −tile 偏移，渲染出来的棋盘格会整体错开**整整一格**（方格黑白还整体反相），
    # 检测出的角点落在"板子真实点 +1 格"处，PnP 只能靠平面二义性去凑一个 180° 的解——
    # 实测正面视图误差 248 px、姿态差精确 180°，而重投影误差仍只有 0.19 px（极难察觉）。
    def to_board(px: float, py: float) -> tuple[float, float, float]:
        return (
            (px - tile) / tile * square_size,
            (py - tile) / tile * square_size,
            0.0,
        )

    template_corners = np.array(
        [
            to_board(0, 0),
            to_board(template.shape[1] - 1, 0),
            to_board(template.shape[1] - 1, template.shape[0] - 1),
            to_board(0, template.shape[0] - 1),
        ],
        dtype=np.float64,
    )
    projected, _ = cv2.projectPoints(template_corners, rvec, tvec, camera_matrix, dist_coeffs)
    src = np.array(
        [
            [0.0, 0.0],
            [template.shape[1] - 1.0, 0.0],
            [template.shape[1] - 1.0, template.shape[0] - 1.0],
            [0.0, template.shape[0] - 1.0],
        ],
        dtype=np.float32,
    )
    dst = projected.reshape(-1, 2).astype(np.float32)
    # 目标图要足够大以容纳投影后的板子
    canvas_w = max(int(image_size[0]), int(np.abs(dst[:, 0]).max()) + 10)
    canvas_h = max(int(image_size[1]), int(np.abs(dst[:, 1]).max()) + 10)
    matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(
        template,
        matrix,
        (canvas_w, canvas_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=background,
    )
    # 裁到目标画幅
    canvas = np.full((image_size[1], image_size[0]), background, np.uint8)
    h = min(image_size[1], warped.shape[0])
    w = min(image_size[0], warped.shape[1])
    canvas[:h, :w] = warped[:h, :w]
    return cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)


def look_at_board(
    distance: float,
    camera_matrix: np.ndarray,
    *,
    tilt_deg: float = 0.0,
    yaw_deg: float = 0.0,
    rim_roll_deg: float = 0.0,
    target_px: tuple[float, float] | None = None,
    pattern_size: tuple[int, int] = PATTERN_SIZE,
    square_size: float = SQUARE_SIZE_M,
) -> tuple[np.ndarray, np.ndarray]:
    """让相机正对棋盘格中心（返回相机在棋盘系下的 rvec/tvec）。

    棋盘在 z=0 平面上、**内角点**范围是 ``[0,(cols-1)·s] × [0,(rows-1)·s]``，
    板子几何中心是 ``((cols-1)·s/2, (rows-1)·s/2, 0)``——不是原点。

    位姿构造方式：取"相机在板子正上方、相机 z 轴指向板子"这个基准位姿，
    再用绕**相机自身轴**的旋转叠加 tilt/yaw/rim_roll 扰动（标定需要的姿态
    多样性）。**基准位姿矩阵不手推**，而是构造一个已知能投影到目标点的相机模型
    来取得——手推旋转矩阵经历过两次约定错误（一次把相机轴当行/列搞反，
    一次 180° 绕错轴），不值得再来第三次。

    ⚠ 扰动是**旋转矩阵右乘**，不是把角度加在旋转向量上：旋转向量只在零附近才近似
    线性，基准位姿的 ``|rvec|`` 一旦接近 π（例如``v`` 轴反向的写法就会得到 π），
    "向量相加"会给出与预期完全不同的姿态，而且症状是"时好时坏"，极难定位。

    返回的 ``(rvec, tvec)`` 由 ``cv2.solvePnP`` 在"棋盘格角点 + 这些相机参数"
    上解出，因此**保证与 OpenCV 的投影约定一致**。
    """
    import cv2  # 惰性导入：重依赖，只在真正做标定时加载（见模块 docstring）

    cols, rows = pattern_size
    center = np.array([(cols - 1) * square_size / 2.0, (rows - 1) * square_size / 2.0, 0.0])
    camera_matrix = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)
    if target_px is None:
        target_px = (float(camera_matrix[0, 2]), float(camera_matrix[1, 2]))
    base = _base_look_down_pose(
        distance,
        camera_matrix,
        target_px,
        center,
        pattern_size=pattern_size,
        square_size=square_size,
    )
    disturbance = np.radians(np.array([rim_roll_deg, tilt_deg, yaw_deg], dtype=np.float64))
    rotation = cv2.Rodrigues(base[0].reshape(3, 1))[0] @ cv2.Rodrigues(disturbance.reshape(3, 1))[0]
    return cv2.Rodrigues(rotation)[0], base[1]


def _base_look_down_pose(
    distance: float,
    camera_matrix: np.ndarray,
    target_px: tuple[float, float],
    board_center: np.ndarray,
    *,
    pattern_size: tuple[int, int] = PATTERN_SIZE,
    square_size: float = SQUARE_SIZE_M,
) -> tuple[np.ndarray, np.ndarray]:
    """构造"相机在板子正上方、光轴竖直向下、**看板子正面**"的位姿，用 solvePnP 反解 rvec/tvec。

    直接手推旋转矩阵容易在约定上出错；这里改用"造一个临时相机模型 → solvePnP"
    的办法，让 OpenCV 自己给出与它投影约定一致的位姿。

    ⚠ 图像 v 轴与板子 +y **同向**（``v = cy + dy·scale``）是经过实测的：反过来会让
    合成图上的棋盘格成为**镜像**，检测出的角点标签整体反序，PnP 于是稳定地落到
    另一个分支上（实测 4 个视图全部差 180°）。别"顺手改成更直观的"写法。
    """
    import cv2  # 惰性导入：重依赖，只在真正做标定时加载（见模块 docstring）

    cols, rows = pattern_size
    # 临时相机：主点设在希望板子中心落在的位置，光轴竖直向下（俯视）
    temp = camera_matrix.copy()
    temp[0, 2] = float(target_px[0])
    temp[1, 2] = float(target_px[1])
    board = board_object_points(pattern_size, square_size)
    # 临时视图：板子四角在临时相机下的投影（俯视 → 正视缩放）
    focal = (float(temp[0, 0]) + float(temp[1, 1])) / 2.0
    scale = focal / float(distance)  # 米 → 像素
    image_points = np.zeros((cols * rows, 1, 2), np.float64)
    for index in range(cols * rows):
        bx, by, _ = board[index, 0]
        dx = bx - float(board_center[0])
        dy = by - float(board_center[1])
        # 图像 u 随板子 +x 增大、v 随板子 +y 增大（与模板像素轴一致，保证不镜像）
        image_points[index, 0, 0] = temp[0, 2] + dx * scale
        image_points[index, 0, 1] = temp[1, 2] + dy * scale
    ok, rvec, tvec = cv2.solvePnP(
        board, image_points, temp, np.zeros(5), flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not ok:
        raise RuntimeError("无法构造棋盘格基准位姿（solvePnP 失败）")
    return rvec.reshape(3), tvec.reshape(3, 1)


# ----------------------------------------------------------------------
# 步骤一：内参
# ----------------------------------------------------------------------
def run_intrinsics(
    frames: Sequence[np.ndarray],
    *,
    pattern_size: tuple[int, int] = PATTERN_SIZE,
    square_size: float = SQUARE_SIZE_M,
    min_views: int = MIN_VIEWS,
    max_views: int = MAX_VIEWS,
) -> IntrinsicsResult:
    """棋盘格多视图内参标定。

    ⚠ **OpenCV 5.0 的 ``calibrateCamera`` 只收 float32 的 objectPoints/imagePoints**
    （实测：任一为 float64 都抛 ``objectPoints should contain vector of vectors of
    points of type Point3f``）。而 :func:`detect_corners` 的 SB 路径给的是 float64，
    所以这里必须显式转 float32——不转的话内参这步会直接失败。
    """
    import cv2  # 惰性导入：重依赖，只在真正做标定时加载（见模块 docstring）

    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    size: tuple[int, int] | None = None
    board = board_object_points(pattern_size, square_size).astype(np.float32)

    for frame in frames:
        if size is None:
            size = (int(frame.shape[1]), int(frame.shape[0]))
        corners = detect_corners(frame, pattern_size)
        if corners is None:
            continue
        object_points.append(board.copy())
        image_points.append(corners.astype(np.float32))
        if len(object_points) >= max_views:
            break

    if size is None:
        raise ValueError("没有任何图像帧")
    if len(object_points) < max(3, min_views):
        raise ValueError(
            f"有效视图不足：找到 {len(object_points)} 个（至少需要 {max(3, min_views)} 个）。"
            "检查棋盘格内角点数 PATTERN_SIZE 是否与实际板子一致，以及棋盘格是否全程可见"
        )

    rms, matrix, dist, rvecs, tvecs = cv2.calibrateCamera(
        object_points, image_points, size, None, None
    )
    per_view: list[float] = []
    for index, (obj, img) in enumerate(zip(object_points, image_points, strict=True)):
        projected, _ = cv2.projectPoints(obj, rvecs[index], tvecs[index], matrix, dist)
        error = float(np.sqrt(np.mean(np.sum((projected - img) ** 2, axis=2))))
        per_view.append(error)
    return IntrinsicsResult(
        camera_matrix=np.asarray(matrix, dtype=np.float64),
        dist_coeffs=np.asarray(dist, dtype=np.float64).reshape(-1),
        image_size=size,
        rms=float(rms),
        views=len(object_points),
        per_view_errors=per_view,
    )


# ----------------------------------------------------------------------
# 步骤二：画面-遥测时间差
# ----------------------------------------------------------------------
def rotation_angle_deg(rotation: np.ndarray) -> float:
    """旋转矩阵对应的旋转角（度）。"""
    trace = float(np.trace(rotation))
    cos_theta = max(-1.0, min(1.0, (trace - 1.0) / 2.0))
    return math.degrees(math.acos(cos_theta))


def quaternion_to_matrix_local(
    quaternion: tuple[float, float, float, float],
) -> np.ndarray:
    """四元数 (w,x,y,z) → 旋转矩阵（与 georef.quaternion_to_matrix 同一约定）。"""
    from airdrop.georef import quaternion_to_matrix

    return quaternion_to_matrix(*quaternion)


def angular_speed_from_poses(
    timestamps: Sequence[float], rotations: Sequence[np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """由逐帧旋转序列算角速度模长（度/秒）。

    返回 ``(中间时刻, 角速度)``。刚体角速度的**模长与安装角无关**，
    所以这一步可以完全在外参未知的情况下做——这正是它能先于外参的原因。
    """
    times: list[float] = []
    speeds: list[float] = []
    for index in range(1, len(rotations)):
        dt = float(timestamps[index] - timestamps[index - 1])
        if dt <= 0:
            continue
        delta = rotations[index - 1].T @ rotations[index]
        angle = rotation_angle_deg(delta)
        times.append(0.5 * (timestamps[index] + timestamps[index - 1]))
        speeds.append(angle / dt)
    return np.asarray(times), np.asarray(speeds, dtype=np.float64)


def estimate_lag_by_correlation(
    camera_times: np.ndarray,
    camera_speeds: np.ndarray,
    body_times: np.ndarray,
    body_speeds: np.ndarray,
    *,
    search_s: float = LAG_SEARCH_S,
    step_s: float = LAG_STEP_S,
) -> LagResult:
    """互相关求"画面比遥测晚多少秒"。

    把画面角速度序列按 ``+lag`` 平移后再与机体系列比对，取相关系数峰值；
    峰值附近做抛物线插值取亚样本精度。``lag > 0`` 表示画面比姿态**滞后**。
    """
    if camera_speeds.size < 3 or body_speeds.size < 3:
        return LagResult(lag_s=float("nan"), peak=float("nan"), samples=0)

    base = min(float(camera_times[0]), float(body_times[0]))
    cam_t = camera_times - base
    body_t = body_times - base

    # 重采样到统一时间基（线性插值，超出范围用端点值而不是外推）
    def resample(times: np.ndarray, values: np.ndarray, grid: np.ndarray) -> np.ndarray:
        return np.interp(grid, times, values)

    n = max(16, min(2000, int(max(cam_t[-1], body_t[-1]) / max(step_s, 1e-3)) + 1))
    grid = np.linspace(0.0, max(cam_t[-1], body_t[-1]), n)
    cam = resample(cam_t, camera_speeds, grid)
    body = resample(body_t, body_speeds, grid)

    cam_centered = cam - cam.mean()
    body_centered = body - body.mean()
    cam_norm = float(np.linalg.norm(cam_centered))
    body_norm = float(np.linalg.norm(body_centered))
    if cam_norm < 1e-9 or body_norm < 1e-9:
        return LagResult(lag_s=float("nan"), peak=float("nan"), samples=0)

    shifts = np.arange(-search_s, search_s + 1e-12, step_s)
    # 网格点之间的时间间隔
    dt_grid = float(grid[1] - grid[0]) if n > 1 else step_s
    curve: list[tuple[float, float]] = []
    correlations: list[float] = []
    for lag in shifts:
        # 画面滞后 lag → 把画面序列前移 lag/grid 个样本与机体对齐
        offset = int(round(lag / dt_grid))
        if offset == 0:
            shifted = cam_centered
        elif offset > 0:
            shifted = np.concatenate([cam_centered[offset:], np.zeros(offset)])
        else:
            shifted = np.concatenate([np.zeros(-offset), cam_centered[:offset]])
        denominator = cam_norm * body_norm
        value = float(np.dot(shifted, body_centered) / denominator) if denominator else 0.0
        correlations.append(value)
        curve.append((float(lag), value))

    values = np.asarray(correlations)
    best = int(np.argmax(values))
    lag = float(shifts[best])
    # 抛物线亚样本插值
    if 0 < best < len(values) - 1:
        left, middle, right = values[best - 1], values[best], values[best + 1]
        denom = left - 2.0 * middle + right
        if abs(denom) > 1e-12:
            lag += 0.5 * (left - right) / denom * step_s
    return LagResult(
        lag_s=lag,
        peak=float(values[best]),
        samples=int(camera_speeds.size),
        curve=curve,
    )


# ----------------------------------------------------------------------
# 步骤三：手眼外参（自实现 AX=XB；Python 侧没有 cv2.calibrateHandEye）
# ----------------------------------------------------------------------
def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = float(vector[0]), float(vector[1]), float(vector[2])
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def rotation_log(matrix: np.ndarray) -> np.ndarray:
    """旋转矩阵 → 轴角向量。"""
    trace = float(np.trace(matrix))
    cos_theta = max(-1.0, min(1.0, (trace - 1.0) / 2.0))
    theta = math.acos(cos_theta)
    if theta < 1e-12:
        return np.zeros(3)
    if abs(math.pi - theta) < 1e-6:
        diag = np.clip((np.diag(matrix) + 1.0) / 2.0, 0.0, None)
        axis = np.sqrt(diag)
        index = int(np.argmax(axis))
        if axis[index] > 1e-12:
            others = [(index + 1) % 3, (index + 2) % 3]
            for other in others:
                axis[other] = matrix[index, other] / (2.0 * axis[index])
        norm = float(np.linalg.norm(axis))
        return axis / norm * theta if norm > 1e-12 else np.zeros(3)
    axis = np.array(
        [
            matrix[2, 1] - matrix[1, 2],
            matrix[0, 2] - matrix[2, 0],
            matrix[1, 0] - matrix[0, 1],
        ]
    ) / (2.0 * math.sin(theta))
    return axis * theta


def solve_hand_eye(
    a_list: Sequence[np.ndarray],
    b_list: Sequence[np.ndarray],
    *,
    use_translation: bool = True,
) -> tuple[np.ndarray, np.ndarray | None, float, float]:
    """求解 ``A X = X B``（与 ``cv::calibrateHandEye`` 同一约定）。

    A、B 的来源是唯一正确的配对，别换
    ----------------------------------
    * ``A``：机体在**世界系**里的相对运动，来自**遥测位姿** ``G = T_world_body``::

          A_ij = G_j⁻¹ · G_i

    * ``B``：棋盘格在**相机系**里的相对运动，来自 **PnP** ``P = T_cam_board``::

          B_ij = P_j · P_i⁻¹

    待求 ``X = T_body_cam = [R_bc | t_bc]``，与 :mod:`airdrop.georef` 的
    ``R_bc``（相机系→机体系）、``t_bc``（光心在机体系）**同一语义**。

    推导：棋盘格固定在世界里（``C = T_world_board`` 为常量），于是
    ``P_i = X⁻¹ G_i⁻¹ C``；代入两帧得 ``B_ij = X⁻¹ A_ij X``，即 ``A X = X B``。

    ⚠ 退化陷阱（症状是"稳定解出单位矩阵"）
    ------------------------------------------------------
    若把"机体位姿"写成**由棋盘格位姿左乘一个常量**导出的量（``T_body = X · P``，
    哪怕左乘的就是真值），相对运动里那个常量会被整体消掉::

        A = (X P_i)⁻¹ (X P_j) = P_i⁻¹ P_j ≡ B

    于是 ``R_a ≡ R_b``，方程退化成"求与所有 ``R_a`` 可交换的矩阵"，唯一解是单位矩阵，
    **真值根本不在解空间里**。这跟"板子有没有转"无关——板子固定、飞机在动时棋盘格
    相对机体的姿态当然逐帧在变；根因是**拿未知量当已知量用**。
    正确做法只有一条：``A`` 必须来自**独立的遥测位姿**。
    回归用例 ``test_hand_eye_degenerate_when_body_derived_from_camera`` 把它钉死。

    旋转（Horaud 向量化）
    --------------------
    ``R_a R_x = R_x R_b``。按**列优先** vec 的两个恒等式
    ``vec(R_a X) = (I ⊗ R_a) vec(X)``、``vec(X R_b) = (R_bᵀ ⊗ I) vec(X)`` 得::

        [ (I ⊗ R_a) − (R_bᵀ ⊗ I) ] · vec(R_x) = 0

    堆叠全部运动对后取最小奇异值对应的**右奇异向量**（``vt[-1]``：``np.linalg.svd``
    返回的 ``vt`` 每一**行**才是右奇异向量），再正交化。

    ⚠ **必须按 F 序拆回 3×3**：``reshape(3, 3)`` 默认是 C 序，会给出另一个"看起来也像
    合法旋转矩阵"的错解，肉眼分辨不出来。

    平移
    ----
    ``R_a t_x + t_a = R_x t_b + t_x`` ⇒ ``(R_a − I) t_x = R_x t_b − t_a``。
    注意它**必须用到 ``t_a``**，也就是遥测位置。室内手持采集没有 GPS，``t_a`` 不可信，
    所以 ``t_x`` 在本项目里**只当校验用**：标定文件的杆臂是尺量值，这里算出来的估计值
    与它对比（差得离谱 ⇒ 遥测位置不可用）。能否采信由 :func:`run_extrinsics` 的残差
    判据决定——``t_a`` 是真是假都解得出一个 ``t_x``，光看数字大小分辨不出来。

    返回 ``(R_x, t_x | None, 相对奇异值, 旋转轴张角(度))``。
    相对奇异值 ``s[-1]/s[0]`` 越接近 0 说明运动激励越充分（绝对值随量纲变化，只有相对量可比）。
    """
    if len(a_list) != len(b_list) or len(a_list) < 2:
        raise ValueError(f"运动对数量不足: {len(a_list)}")
    rows = []
    axes: list[np.ndarray] = []
    for a_matrix, b_matrix in zip(a_list, b_list, strict=True):
        rotation_a = a_matrix[:3, :3]
        rotation_b = b_matrix[:3, :3]
        rows.append(np.kron(np.eye(3), rotation_a) - np.kron(rotation_b.T, np.eye(3)))
        axes.append(_rotation_axis(rotation_a))
    stacked = np.vstack(rows)
    _, singular, vt = np.linalg.svd(stacked)
    relative = float(singular[-1] / singular[0]) if singular[0] > 0 else float("inf")
    candidate = vt[-1].reshape(3, 3, order="F")
    # ⚠ 零空间向量只定到"差一个符号"，而 ``vec(R)`` 与 ``vec(−R)`` 表示**同一个旋转**。
    # 但 ``−R`` 的行列式是 −1、不是旋转矩阵：若直接把它喂给"翻转最小奇异值"式的正交化，
    # 会得到 ``R·diag(−1,−1,1)``——**恰好是绕 z 的 180° 旋转**。症状极其阴险：
    # 零空间确实是一维、真值确实在里面，解出来却精确偏 180°，而且**时对时错**
    # （取决于 SVD 给出的符号）。正确做法是先按行列式**整体取反**，再做正交化。
    if float(np.linalg.det(candidate)) < 0.0:
        candidate = -candidate
    u, _, vt2 = np.linalg.svd(candidate)
    rotation = u @ np.diag([1.0, 1.0, float(np.linalg.det(u @ vt2))]) @ vt2

    translation: np.ndarray | None = None
    if use_translation:
        d_matrix = np.zeros((3, 3))
        d_vector = np.zeros(3)
        for a_matrix, b_matrix in zip(a_list, b_list, strict=True):
            d_matrix += a_matrix[:3, :3] - np.eye(3)
            d_vector += rotation @ b_matrix[:3, 3] - a_matrix[:3, 3]
        translation, *_ = np.linalg.lstsq(d_matrix, d_vector, rcond=None)
    return rotation, translation, relative, _axis_spread_deg(axes)


def _rotation_axis(rotation: np.ndarray) -> np.ndarray:
    """旋转矩阵的转轴（单位向量）；无旋转返回零向量。"""
    vector = rotation_log(rotation)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-12 else np.zeros(3)


def _axis_spread_deg(axes: Sequence[np.ndarray]) -> float:
    """一组转轴之间的最大夹角（度）。转轴是双向的，所以正负号不影响结果。"""
    spread = 0.0
    for index, first in enumerate(axes):
        for second in axes[index + 1 :]:
            norm = float(np.linalg.norm(first) * np.linalg.norm(second))
            if norm <= 1e-12:
                continue
            cosine = abs(float(np.dot(first, second))) / norm
            spread = max(spread, math.degrees(math.acos(max(-1.0, min(1.0, cosine)))))
    return spread


def _translation_span(poses: Sequence[np.ndarray]) -> float:
    """一组位姿的位置跨度（米）：三轴极差组成的包围盒对角线。

    用来把"遥测说机体动了多少"和"PnP 说相机动了多少"并排看——两者本该同量级
    （只差一个杆臂随姿态转动带来的小量），差一个数量级就是遥测位置不可用的铁证。
    """
    if not poses:
        return 0.0
    points = np.asarray([pose[:3, 3] for pose in poses], dtype=np.float64)
    return float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))


def _translation_residual(
    a_list: Sequence[np.ndarray],
    b_list: Sequence[np.ndarray],
    rotation: np.ndarray,
    translation: np.ndarray | None,
) -> float:
    """平移方程 ``(R_a − I)t_x + t_a − R_x t_b`` 的最大残差（米）。

    这是判"遥测位置能不能用"的关键量。``t_a`` 是真是假都解得出一个 ``t_x``
    （最小二乘照样给你一个数），但假 ``t_a`` 会让方程**本身不自洽**：残差涨到与真实
    位移同量级。所以别只看 ``t_x`` 的大小，要看这个残差。
    """
    if translation is None:
        return float("nan")
    identity = np.eye(3)
    worst = 0.0
    for a_matrix, b_matrix in zip(a_list, b_list, strict=True):
        offset = (
            (a_matrix[:3, :3] - identity) @ translation
            + a_matrix[:3, 3]
            - rotation @ b_matrix[:3, 3]
        )
        worst = max(worst, float(np.linalg.norm(offset)))
    return worst


def relative_transform(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """相对位姿 ``T_a^{-1} T_b``（输入为 4×4）。"""
    return np.linalg.inv(a) @ b


# ----------------------------------------------------------------------
# 遥测位姿：与画面**同时刻**的 T_world_body
# ----------------------------------------------------------------------
def _quaternion_of(snapshot: TelemetrySnapshot) -> tuple[float, float, float, float] | None:
    """快照里的姿态四元数 ``(w, x, y, z)``；任一分量缺失就返回 None（**不拿 0 顶上**）。"""
    w, x, y, z = (
        snapshot.quaternion_w,
        snapshot.quaternion_x,
        snapshot.quaternion_y,
        snapshot.quaternion_z,
    )
    if w is None or x is None or y is None or z is None:
        return None
    return (float(w), float(x), float(y), float(z))


def body_pose_matrix(snapshot: TelemetrySnapshot) -> np.ndarray | None:
    """遥测快照 → ``T_world_body``（4×4）。位置或姿态缺一不可，否则返回 None。"""
    if (
        snapshot.north_m is None
        or snapshot.east_m is None
        or snapshot.down_m is None
        or snapshot.quaternion_w is None
        or snapshot.quaternion_x is None
        or snapshot.quaternion_y is None
        or snapshot.quaternion_z is None
    ):
        return None
    matrix = np.eye(4)
    matrix[:3, :3] = quaternion_to_matrix_local(
        (
            float(snapshot.quaternion_w),
            float(snapshot.quaternion_x),
            float(snapshot.quaternion_y),
            float(snapshot.quaternion_z),
        )
    )
    matrix[:3, 3] = (snapshot.north_m, snapshot.east_m, snapshot.down_m)
    return matrix


def _slerp(first: Sequence[float], second: Sequence[float], weight: float) -> np.ndarray:
    """四元数球面插值（w 在前，含 q/−q 翻转）。"""
    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    dot = float(np.dot(a, b))
    if dot < 0.0:
        b = -b
        dot = -dot
    if dot > 0.9995:  # 夹角太小：线性插值再归一化更稳
        return (1.0 - weight) * a + weight * b
    theta = math.acos(min(1.0, max(-1.0, dot)))
    return (math.sin((1.0 - weight) * theta) * a + math.sin(weight * theta) * b) / math.sin(theta)


def telemetry_pose_at(
    snapshots: Sequence[TelemetrySnapshot], timestamp: float
) -> np.ndarray | None:
    """取 ``timestamp`` 时刻的 ``T_world_body``：位置线性内插、姿态 slerp。

    只做**范围内**内插；时刻落在遥测覆盖之外返回 None（不外推——标定不能被外推出来的
    姿态污染）。``snapshots`` 需按 ``timestamp`` 升序（``FlightLog.iter_telemetry`` 即升序）。
    """
    usable = [item for item in snapshots if body_pose_matrix(item) is not None]
    if len(usable) < 2:
        return None
    times = [item.timestamp for item in usable]
    index = bisect_left(times, timestamp)
    if index == 0 or index >= len(usable):
        return None
    lower, upper = usable[index - 1], usable[index]
    span = upper.timestamp - lower.timestamp
    weight = 0.0 if span <= 0.0 else (timestamp - lower.timestamp) / span
    weight = min(1.0, max(0.0, weight))
    lower_pose = body_pose_matrix(lower)
    upper_pose = body_pose_matrix(upper)
    assert lower_pose is not None and upper_pose is not None
    # body_pose_matrix 已保证四个四元数分量非空，这里逐项显式判一次，
    # 好让类型检查器也按"非空"收窄（属性访问不具备那种传递性）。
    lower_q = _quaternion_of(lower)
    upper_q = _quaternion_of(upper)
    if lower_q is None or upper_q is None:
        return None
    quaternion = _slerp(lower_q, upper_q, weight)
    matrix = np.eye(4)
    matrix[:3, :3] = quaternion_to_matrix_local(tuple(quaternion))
    matrix[:3, 3] = (1.0 - weight) * lower_pose[:3, 3] + weight * upper_pose[:3, 3]
    return matrix


def align_body_poses(
    board_poses: Sequence[tuple[float, np.ndarray, np.ndarray]],
    snapshots: Sequence[TelemetrySnapshot],
    *,
    lag_s: float = 0.0,
) -> tuple[list[tuple[float, np.ndarray, np.ndarray]], list[np.ndarray]]:
    """把棋盘格位姿与**同一时刻**的遥测位姿配成对。

    配不上的（遥测覆盖不到的时刻）直接丢弃，返回两个等长列表
    ``(棋盘格位姿, 机体位姿)``——它们随即喂给 :func:`run_extrinsics`。

    ⚠ ``lag_s`` 的符号（这里错一次，外参就会带一个恒定的姿态偏差）：
    :func:`estimate_lag_by_correlation` 里 ``lag > 0`` 表示"**画面**比姿态滞后"，
    即该帧**内容**对应的真实时刻比它的 ``capture_timestamp`` 早 ``lag`` 秒，
    所以查遥测用 ``capture_timestamp − lag_s``。
    """
    paired_board: list[tuple[float, np.ndarray, np.ndarray]] = []
    paired_body: list[np.ndarray] = []
    for stamp, rvec, tvec in board_poses:
        pose = telemetry_pose_at(snapshots, stamp - lag_s)
        if pose is None:
            continue
        paired_board.append((stamp, rvec, tvec))
        paired_body.append(pose)
    return paired_board, paired_body


def run_extrinsics(
    *,
    board_poses: Sequence[tuple[float, np.ndarray, np.ndarray]],
    body_poses: Sequence[np.ndarray],
    min_pair_rotation_deg: float = MIN_PAIR_ROTATION_DEG,
    min_axis_spread_deg: float = MIN_AXIS_SPREAD_DEG,
    estimate_body_translation: bool = True,
    max_pairs: int = MAX_MOTION_PAIRS,
) -> ExtrinsicsResult:
    """手眼标定：``board_poses`` 与 ``body_poses`` 必须**逐项同时刻**。

    * ``board_poses``：``(时刻, rvec, tvec)``，PnP 给出的 ``T_cam_board``；
    * ``body_poses``：``T_world_body``（4×4），用 :func:`align_body_poses` 按帧时刻重采样。

    ⚠ 下面两件事都要做到，否则结果没有意义：
    1. **一一对应**：遥测 10 Hz、画面 30 Hz，"第 i 条遥测配第 i 帧"是错的；
    2. **时刻一致**：扣掉画面-遥测时间差 Δt，见 :func:`align_body_poses`。

    运动对取所有 ``i < j``（超过 ``max_pairs`` 时按固定步长抽样，保证结果确定），
    ``A`` 用遥测相对运动、``B`` 用 PnP 相对运动；方程与解法见 :func:`solve_hand_eye`。

    平移（杆臂）**照算，但只当校验用**
    ----------------------------------
    ``estimate_body_translation=True``（默认）会连 ``t_bc`` 一起解出来——它需要遥测位置
    ``t_a``，室内手持无 GPS 时不可信，所以这个估计值**不进标定文件**，只用于和尺量值
    对表。可采信与否交给残差判据：``t_bc_reliable`` 要求平移方程残差
    ≤ ``max(LEVER_ARM_RESIDUAL_FLOOR_M, LEVER_ARM_RESIDUAL_RATIO × 画面平移跨度)``。
    理由是``t_a`` 是真是假**都**解得出一个 ``t_x``（最小二乘照样给数），假 ``t_a``
    表现在方程不自洽、残差涨到与真实位移同量级，所以判据必须落在残差上。
    """
    if len(board_poses) != len(body_poses):
        raise ValueError(
            f"棋盘格位姿（{len(board_poses)}）与机体位姿（{len(body_poses)}）数量不一致："
            "必须逐项同时刻，用 align_body_poses 按帧时刻重采样遥测"
        )
    if len(board_poses) < 3:
        raise ValueError(f"位姿太少（{len(board_poses)} < 3），至少需要 3 个不同位姿")

    candidates = [
        (index, other)
        for index in range(len(board_poses))
        for other in range(index + 1, len(board_poses))
    ]
    if len(candidates) > max_pairs:
        stride = len(candidates) // max_pairs + 1
        candidates = candidates[::stride]

    body_inverse = [np.linalg.inv(pose) for pose in body_poses]
    camera_pose = [_pose_matrix(rvec, tvec) for _, rvec, tvec in board_poses]
    camera_inverse = [np.linalg.inv(item) for item in camera_pose]

    a_list: list[np.ndarray] = []
    b_list: list[np.ndarray] = []
    for index, other in candidates:
        a_matrix = body_inverse[other] @ body_poses[index]
        b_matrix = camera_pose[other] @ camera_inverse[index]
        if rotation_angle_deg(a_matrix[:3, :3]) < min_pair_rotation_deg:
            continue
        a_list.append(a_matrix)
        b_list.append(b_matrix)

    if len(a_list) < 3:
        raise ValueError(
            f"有效运动对不足（{len(a_list)} < 3）：旋转激励不够。采集时要让飞机绕**多根轴**"
            "都转起来（俯仰/横滚/偏航），全程只平移是解不出来的"
        )
    rotation, translation, relative_sv, axis_spread = solve_hand_eye(
        a_list, b_list, use_translation=estimate_body_translation
    )
    if axis_spread < min_axis_spread_deg:
        raise ValueError(
            f"相对旋转轴过于单一（最大张角 {axis_spread:.1f}° < {min_axis_spread_deg}°）："
            "所有运动对都绕同一根轴时解不唯一（OpenCV 文档同样要求至少 2 个不平行的"
            "旋转轴），请重新采集"
        )

    residuals = [
        rotation_angle_deg((a_matrix[:3, :3] @ rotation).T @ (rotation @ b_matrix[:3, :3]))
        for a_matrix, b_matrix in zip(a_list[:50], b_list[:50], strict=True)
    ]
    mean_residual = float(np.mean(residuals)) if residuals else float("nan")
    body_span = _translation_span(body_poses)
    camera_span = _translation_span(camera_pose)
    translation_residual = _translation_residual(a_list, b_list, rotation, translation)
    tolerance = lever_arm_tolerance(camera_span)
    return ExtrinsicsResult(
        r_bc=rotation,
        t_bc=translation,
        pairs=len(a_list),
        residual_rotation_deg=mean_residual,
        axis_spread_deg=axis_spread,
        t_bc_reliable=translation is not None and translation_residual <= tolerance,
        residual_translation_m=translation_residual,
        body_span_m=body_span,
        camera_span_m=camera_span,
        methods={"horaud": mean_residual, "relative_singular": relative_sv},
    )


def lever_arm_tolerance(camera_span_m: float) -> float:
    """杆臂**估计值**的采信门限（米）：平移残差不超过它才算可信。

    门限随实际运动量放大（``LEVER_ARM_RESIDUAL_RATIO × 画面平移跨度``），因为残差本身
    是"米"，与动得多少直接相关；动得很小时用 ``LEVER_ARM_RESIDUAL_FLOOR_M`` 兜底。
    """
    return max(LEVER_ARM_RESIDUAL_FLOOR_M, LEVER_ARM_RESIDUAL_RATIO * camera_span_m)


def lever_arm_report(
    result: ExtrinsicsResult | None,
    measured: Sequence[float] | None = None,
) -> dict[str, Any]:
    """杆臂记录：**以尺量为准**，同时给出标定估计值供对比校验。

    估计值来自 :func:`solve_hand_eye` 的平移分支，需要遥测位置 ``t_a``；室内手持采集
    没有 GPS 时它不可信，所以估计值**不进**标定结果的 ``t_bc`` 字段（georef 读的是尺量
    值），只写进 ``meta`` 供人核对：两者应当同量级、逐轴接近，``delta_m`` 是欧氏距离。
    差得离谱正是"遥测位置不可用"的证据——此时 ``estimated_reliable`` 也会是 False。

    ``measured=None`` 时取 :data:`MEASURED_T_BC`——**在调用时取**，不是定义时绑定：
    那个常量是给人改的，写进默认参数会让"改了常量、报告里还是旧值"。
    """
    measured_list = [float(value) for value in (MEASURED_T_BC if measured is None else measured)]
    report: dict[str, Any] = {
        "authoritative": measured_list,
        "source": "measured",
        "estimated": None,
        "delta_xyz_m": None,
        "delta_m": None,
        "estimated_reliable": False,
        "residual_translation_m": None,
        "residual_tolerance_m": None,
        "body_span_m": None,
        "camera_span_m": None,
        "note": (
            "t_bc 以尺量为准（手持采集时遥测位置不可信）；"
            "estimated 仅供对比校验，估计值与尺量值差得离谱即说明遥测位置不可用"
        ),
    }
    if result is None or result.t_bc is None:
        return report
    estimated = np.asarray(result.t_bc, dtype=np.float64).reshape(3)
    delta = estimated - np.asarray(measured_list, dtype=np.float64)
    report.update(
        estimated=estimated.tolist(),
        delta_xyz_m=[float(value) for value in delta],
        delta_m=float(np.linalg.norm(delta)),
        estimated_reliable=bool(result.t_bc_reliable),
        residual_translation_m=float(result.residual_translation_m),
        residual_tolerance_m=lever_arm_tolerance(result.camera_span_m),
        body_span_m=float(result.body_span_m),
        camera_span_m=float(result.camera_span_m),
    )
    return report


def _pose_matrix(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    import cv2  # 惰性导入：重依赖，只在真正做标定时加载（见模块 docstring）

    rotation, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
    return matrix


def _pose_matrix_from_rotation(rotation: np.ndarray, translation: Sequence[float]) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return matrix


# ----------------------------------------------------------------------
# 飞行目录读取
# ----------------------------------------------------------------------
def load_flight_frames(flight_dir: Path, limit: int = 200) -> list[np.ndarray]:
    """读飞行目录里的帧（``frames_index.jsonl`` + ``frames/``）。"""
    import cv2  # 惰性导入：重依赖，只在真正做标定时加载（见模块 docstring）

    index_path = flight_dir / "frames_index.jsonl"
    if not index_path.is_file():
        raise FileNotFoundError(f"缺少帧索引: {index_path}")
    frames: list[np.ndarray] = []
    for line in index_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        path = flight_dir / "frames" / record["filename"]
        if not path.is_file():
            continue
        image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is not None:
            frames.append(image)
        if len(frames) >= limit:
            break
    return frames


def load_flight_telemetry(flight_dir: Path) -> list[TelemetrySnapshot]:
    """读飞行目录的 ``telemetry.jsonl``。"""
    from airdrop.record import FlightLog

    return list(FlightLog.open(flight_dir).iter_telemetry())


def load_flight_timestamps(flight_dir: Path, limit: int = 200) -> list[float]:
    """读帧索引里的拍摄时刻。"""
    index_path = flight_dir / "frames_index.jsonl"
    stamps: list[float] = []
    for line in index_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        stamps.append(float(json.loads(line)["capture_timestamp"]))
        if len(stamps) >= limit:
            break
    return stamps


# ----------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------
def calibrate(
    flight_dir: Path,
    *,
    pattern_size: tuple[int, int] = PATTERN_SIZE,
    square_size: float = SQUARE_SIZE_M,
    lag_search_s: float = LAG_SEARCH_S,
    lag_step_s: float = LAG_STEP_S,
    max_frames: int = 200,
) -> dict[str, Any]:
    """跑完三步，返回可写入 ``camera_calib.json`` 的字典。"""
    import cv2  # 惰性导入：重依赖，只在真正做标定时加载（见模块 docstring）

    LOGGER.info("读取标定素材: %s", flight_dir)
    frames = load_flight_frames(flight_dir, max_frames)
    if not frames:
        raise ValueError(f"{flight_dir} 里没有可用帧")
    LOGGER.info("共 %d 帧", len(frames))

    # ---- 步骤一 ----
    LOGGER.info("步骤一：内参标定")
    intrinsics = run_intrinsics(frames, pattern_size=pattern_size, square_size=square_size)
    LOGGER.info(
        "  视图 %d，重投影 RMS %.4f px，fx=%.1f fy=%.1f",
        intrinsics.views,
        intrinsics.rms,
        intrinsics.camera_matrix[0, 0],
        intrinsics.camera_matrix[1, 1],
    )

    # ---- 步骤二 ----
    LOGGER.info("步骤二：画面-遥测时间差")
    timestamps = load_flight_timestamps(flight_dir, max_frames)
    board_points = board_object_points(pattern_size, square_size)
    board_rotations: list[np.ndarray] = []
    pose_times: list[float] = []
    board_poses: list[tuple[float, np.ndarray, np.ndarray]] = []
    rejected = 0
    # 用 solve_board_poses（含相邻帧连续性定支）而不是裸 solvePnP：
    # 检测器的角点顺序可能与我们枚举的对象点相反，单帧无法分辨（见其 docstring），
    # 而**混用两种标签**会让第三步残差飙到几十度。
    for stamp, pose in zip(
        timestamps,
        solve_board_poses(
            frames,
            board_points,
            intrinsics.camera_matrix,
            intrinsics.dist_coeffs,
            pattern_size=pattern_size,
            square_size=square_size,
        ),
        strict=True,
    ):
        if pose is None:
            rejected += 1
            continue
        rvec, tvec = pose
        rotation, _ = cv2.Rodrigues(rvec)
        board_rotations.append(rotation)
        pose_times.append(stamp)
        board_poses.append((stamp, rvec, tvec))
    if rejected:
        LOGGER.warning("  丢弃 %d 帧（角点检不出）", rejected)

    snapshots = [
        item for item in load_flight_telemetry(flight_dir) if _quaternion_of(item) is not None
    ]
    lag = LagResult(lag_s=0.0, peak=float("nan"), samples=0)
    if len(board_rotations) >= 3 and len(snapshots) >= 3:
        cam_t, cam_w = angular_speed_from_poses(pose_times, board_rotations)
        body_rotations = [
            quaternion_to_matrix_local(quaternion)
            for item in snapshots
            if (quaternion := _quaternion_of(item)) is not None
        ]
        body_t, body_w = angular_speed_from_poses(
            [item.timestamp for item in snapshots], body_rotations
        )
        lag = estimate_lag_by_correlation(
            cam_t,
            cam_w,
            body_t,
            body_w,
            search_s=lag_search_s,
            step_s=lag_step_s,
        )
        LOGGER.info("  Δt = %.4f s（相关系数 %.3f，样本 %d）", lag.lag_s, lag.peak, lag.samples)
    else:
        LOGGER.warning("  棋盘格位姿或四元数不足，跳过时间差估计")

    # ---- 步骤三 ----
    LOGGER.info("步骤三：手眼外参")
    extrinsics: ExtrinsicsResult | None = None
    if board_poses and len(snapshots) >= 2:
        # 与画面**同时刻**的机体位姿：遥测 10 Hz、画面 30 Hz，
        # 直接把"第 i 条遥测"配给"第 i 帧"是错的，必须按帧时刻重采样
        paired_board, paired_body = align_body_poses(
            board_poses, snapshots, lag_s=lag.lag_s if lag.ok else 0.0
        )
        dropped = len(board_poses) - len(paired_board)
        if dropped:
            LOGGER.info("  丢弃 %d 个时刻不在遥测覆盖内的棋盘格位姿", dropped)
        try:
            extrinsics = run_extrinsics(board_poses=paired_board, body_poses=paired_body)
            LOGGER.info(
                "  运动对 %d，旋转残差 %.3f°，旋转轴张角 %.1f°",
                extrinsics.pairs,
                extrinsics.residual_rotation_deg,
                extrinsics.axis_spread_deg,
            )
            # 杆臂：**尺量为准**（写进标定文件的是它），标定估计值只用来对表
            arm = lever_arm_report(extrinsics)
            if arm["estimated"] is None:
                LOGGER.info("  杆臂按尺量: t_bc=%s（未估）", list(MEASURED_T_BC))
            else:
                LOGGER.info(
                    "  杆臂：尺量 %s，标定估计 %s，差 %.3f m",
                    [round(value, 4) for value in arm["authoritative"]],
                    [round(value, 4) for value in arm["estimated"]],
                    arm["delta_m"],
                )
                if arm["estimated_reliable"]:
                    LOGGER.info(
                        "  估计值可采信（平移残差 %.4f m 在 %.4f m 以内；"
                        "遥测位置跨度 %.2f m vs 画面 %.2f m）",
                        arm["residual_translation_m"],
                        arm["residual_tolerance_m"],
                        arm["body_span_m"],
                        arm["camera_span_m"],
                    )
                else:
                    LOGGER.warning(
                        "  估计值**不可信**（平移残差 %.4f m 超出 %.4f m；遥测位置跨度 "
                        "%.2f m vs 画面 %.2f m）——室内手持无 GPS 时遥测位置不可用，"
                        "以尺量值 %s 为准；两者差得离谱正是这个原因",
                        arm["residual_translation_m"],
                        arm["residual_tolerance_m"],
                        arm["body_span_m"],
                        arm["camera_span_m"],
                        list(MEASURED_T_BC),
                    )
        except ValueError as exc:
            LOGGER.warning("  手眼标定失败: %s", exc)
    else:
        LOGGER.warning("  缺少棋盘格位姿或遥测位姿，跳过外参")

    payload: dict[str, Any] = {
        "camera_matrix": intrinsics.camera_matrix.tolist(),
        "dist_coeffs": intrinsics.dist_coeffs.tolist(),
        "image_size": list(intrinsics.image_size),
        "R_bc": (extrinsics.r_bc.tolist() if extrinsics is not None else _DEFAULT_R_BC_LIST),
        "t_bc": list(MEASURED_T_BC),
        "telemetry_lag": float(lag.lag_s) if lag.ok else None,
        "meta": {
            "intrinsics": {
                "rms_px": intrinsics.rms,
                "views": intrinsics.views,
                "per_view_errors": intrinsics.per_view_errors,
                "pattern_size": list(pattern_size),
                "square_size_m": square_size,
            },
            "time_offset": {
                "lag_s": lag.lag_s if lag.ok else None,
                "peak_correlation": lag.peak if lag.ok else None,
                "samples": lag.samples,
                "search_s": lag_search_s,
                "step_s": lag_step_s,
                "curve": lag.curve if lag.ok else [],
            },
            "extrinsics": (
                {
                    "pairs": extrinsics.pairs,
                    "residual_rotation_deg": extrinsics.residual_rotation_deg,
                    "axis_spread_deg": extrinsics.axis_spread_deg,
                    "t_bc_source": "measured",
                    "lever_arm": lever_arm_report(extrinsics),
                    "methods": extrinsics.methods,
                    "note": (
                        "R_bc 以手眼标定为准；t_bc 以尺量为准——手持采集时遥测位置不可信，"
                        "标定给出的杆臂估计值只用于与尺量值对比校验"
                    ),
                }
                if extrinsics is not None
                else None
            ),
            "source_dir": str(flight_dir),
        },
    }
    return payload


#: 外参标定失败时写入文件用的默认值（绕 z +90°，与 georef.DEFAULT_R_BC 一致）
_DEFAULT_R_BC_LIST = rotation_z(90.0).tolist()


# ----------------------------------------------------------------------
# 配置与入口（命令行解析在 airdrop/run.py，本文件只暴露库接口）
# ----------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class CalibrateConfig:
    """一次标定用到的全部参数（默认值 = 本文件顶部的常量）。"""

    #: 标定采集目录（FlightRecorder 产出）
    flight_dir: Path = FLIGHT_DIR
    output_path: Path = OUTPUT_PATH
    #: 棋盘格内角点数（列, 行）与方格边长（米）
    pattern_size: tuple[int, int] = PATTERN_SIZE
    square_size: float = SQUARE_SIZE_M
    #: 时间差搜索范围与步长（秒）
    lag_search_s: float = LAG_SEARCH_S
    lag_step_s: float = LAG_STEP_S
    #: 最多读取多少帧
    max_frames: int = MAX_FRAMES
    #: 严格模式：外参/时间差没标出来时返回非 0（默认只警告，照样写文件）
    strict: bool = STRICT


def build_config(**overrides) -> CalibrateConfig:
    """按关键字覆盖派生一份标定配置（``dataclasses.replace``；未知字段直接报错）。"""
    return replace(CalibrateConfig(), **overrides)


def main(**overrides) -> int:
    """跑一次三步标定并写出 JSON；关键字与 :func:`build_config` 一致。

    退出码：0 = 正常写出；2 = ``strict=True`` 且外参或时间差没标出来；1 = 出错（异常抛出）。
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    settings = build_config(**overrides)
    flight_dir = Path(settings.flight_dir)
    if not flight_dir.is_dir():
        # 失败要显式：目录不存在直接抛，不静默换个目录
        raise FileNotFoundError(f"飞行目录不存在: {flight_dir}")

    payload = calibrate(
        flight_dir,
        pattern_size=settings.pattern_size,
        square_size=settings.square_size,
        lag_search_s=settings.lag_search_s,
        lag_step_s=settings.lag_step_s,
        max_frames=settings.max_frames,
    )
    out_path = Path(settings.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("标定结果已写入 %s", out_path.resolve())
    if not settings.strict:
        return 0
    # 严格模式：把"写进文件的那两处降级"当失败报出来（默认只警告）
    problems: list[str] = []
    if payload.get("telemetry_lag") is None:
        problems.append("画面-遥测时间差没有标出来（telemetry_lag=null）")
    if payload["meta"].get("extrinsics") is None:
        problems.append("手眼外参没有标出来（R_bc 用的是默认值）")
    if problems:
        for item in problems:
            LOGGER.error("严格模式判定标定不合格：%s", item)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
