"""相机模型：内参、畸变、以及**标定产出**的相机→机体外参。

为什么外参不能硬编码
--------------------
旧版实现把安装角写成常量 ``_CAM_YAW_OFFSET_DEG = 90``（见其 ``coordinates.py``）。
那个 90° 是**当时那台相机在那个支架上**的安装偏航；换相机/换支架就全错，而且
错误会以"目标偏 90°"这种极难归因的方式表现出来。P6 改成从标定文件读 ``R_bc``，
并把 90° 只作为**没有标定文件时的默认值**（它正好等价于 legacy 的固定安装角，
所以未标定时行为与旧版实现一致）。

约定（务必一致，错了会让目标镜像/旋转）
----------------------------------------
* 相机系：OpenCV 惯例，``x`` 向右、``y`` 向下、``z`` 沿光轴向前；
* 机体系：NED 惯例，``x`` 向前（机头）、``y`` 向右、``z`` 向下；
* ``R_bc``：**相机系向量 → 机体系向量**的旋转，即 ``v_body = R_bc @ v_cam``；
* ``t_bc``：相机光心在**机体系**中的位置（杆臂），默认零。

``Config.camera.calib_file`` 指向 ``tools/calibrate.py`` 产出的 ``camera_calib.json``；
文件不存在时用默认值并 warning（**不静默**），这样"忘了标定"会立刻在日志里看到。
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

LOGGER = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_R_BC",
    "CameraModel",
    "euler_to_matrix",
    "load_camera_model",
    "rotation_x",
    "rotation_y",
    "rotation_z",
]


def rotation_x(angle_deg: float) -> np.ndarray:
    a = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rotation_y(angle_deg: float) -> np.ndarray:
    a = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rotation_z(angle_deg: float) -> np.ndarray:
    a = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def euler_to_matrix(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    """机体 3-2-1（Z-Y-X）姿态 → 机体→NED 旋转矩阵 ``R_nb``。

    与旧版实现 ``coordinates.euler_to_rotation_matrix`` 完全一致：``Rz @ Ry @ Rx``。
    """
    return rotation_z(yaw_deg) @ rotation_y(pitch_deg) @ rotation_x(roll_deg)


#: 没有标定文件时的默认"相机→机体"外参 = 绕机体 Z（向下）转 +90°。
#:
#: 这不是随手取的：旧版实现的 ``pixel_to_ned`` 用 ``yaw + 90°`` 构造旋转矩阵，
#: 数学上等价于在**机体→NED** 之前先做一次绕机体 z 的 +90°（相机系→机体系）。
#: 实测确认了这一点：该模型下"图像右移 → 机体右、图像下移 → 机头方向"。
DEFAULT_R_BC = rotation_z(90.0)


@dataclass(frozen=True, slots=True)
class CameraModel:
    """相机内参 + 畸变 + 相机→机体外参。

    ``camera_matrix`` 是 3×3（像素单位），``dist_coeffs`` 是长度 4/5/8/12/14 的
    畸变系数（OpenCV 顺序 ``k1 k2 p1 p2 [k3 ...]``）。
    """

    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray = field(default_factory=lambda: np.zeros(5, dtype=np.float64))
    #: 相机系 → 机体系
    # default_factory 要的就是"可调用对象"：写 DEFAULT_R_BC.copy 与 lambda 等价，
    # 但 lambda 更能看出"每个实例各拿一份拷贝"，故保留。
    r_bc: np.ndarray = field(default_factory=lambda: DEFAULT_R_BC.copy())  # noqa: PLW0108
    #: 相机光心在机体系中的位置（杆臂），米
    t_bc: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    #: 标定时的图像尺寸 (width, height)；用于"帧尺寸不符就跳过去畸变"
    image_size: tuple[int, int] | None = None
    #: 标定链路延时（秒），供 VideoConfig.telemetry_lag 覆盖
    telemetry_lag: float | None = None
    #: 标定文件路径（排故用；默认模型时为 None）
    source: str | None = None

    def __post_init__(self) -> None:
        matrix = np.asarray(self.camera_matrix, dtype=np.float64).reshape(3, 3)
        if not np.isfinite(matrix).all() or matrix[0, 0] <= 0 or matrix[1, 1] <= 0:
            raise ValueError(f"内参矩阵不合法: {matrix.tolist()}")
        object.__setattr__(self, "camera_matrix", matrix)

        dist = np.asarray(self.dist_coeffs, dtype=np.float64).reshape(-1)
        object.__setattr__(self, "dist_coeffs", dist)

        rotation = np.asarray(self.r_bc, dtype=np.float64).reshape(3, 3)
        if not _is_rotation(rotation):
            raise ValueError(f"R_bc 不是合法旋转矩阵（需正交且 det=+1）: {rotation.tolist()}")
        object.__setattr__(self, "r_bc", rotation)

        offset = np.asarray(self.t_bc, dtype=np.float64).reshape(3)
        object.__setattr__(self, "t_bc", offset)

    # ------------------------------------------------------------------
    # 常用量
    # ------------------------------------------------------------------
    @property
    def fx(self) -> float:
        return float(self.camera_matrix[0, 0])

    @property
    def fy(self) -> float:
        return float(self.camera_matrix[1, 1])

    @property
    def cx(self) -> float:
        return float(self.camera_matrix[0, 2])

    @property
    def cy(self) -> float:
        return float(self.camera_matrix[1, 2])

    @property
    def focal_px(self) -> float:
        """等效焦距（fx/fy 均值），边长法估深度用。"""
        return (self.fx + self.fy) / 2.0

    def principal_point(self) -> tuple[float, float]:
        return self.cx, self.cy

    def is_calibrated(self) -> bool:
        """是否来自标定文件（否则是默认外参）。"""
        return self.source is not None

    def scaled(self, scale: float) -> "CameraModel":
        """按比例缩放内参（用于"标定在某分辨率、实际跑另一分辨率"的换算）。"""
        if scale <= 0:
            raise ValueError(f"scale 必须为正: {scale}")
        matrix = self.camera_matrix.copy()
        matrix[0, 0] *= scale
        matrix[1, 1] *= scale
        matrix[0, 2] *= scale
        matrix[1, 2] *= scale
        size = (
            None
            if self.image_size is None
            else (int(round(self.image_size[0] * scale)), int(round(self.image_size[1] * scale)))
        )
        return CameraModel(
            camera_matrix=matrix,
            dist_coeffs=self.dist_coeffs.copy(),
            r_bc=self.r_bc.copy(),
            t_bc=self.t_bc.copy(),
            image_size=size,
            telemetry_lag=self.telemetry_lag,
            source=self.source,
        )


def _is_rotation(matrix: np.ndarray, tol: float = 1e-6) -> bool:
    if not np.isfinite(matrix).all():
        return False
    if not np.allclose(matrix @ matrix.T, np.eye(3), atol=tol):
        return False
    return abs(float(np.linalg.det(matrix)) - 1.0) <= tol


def default_camera_model(width: int = 1280, height: int = 720) -> CameraModel:
    """没有标定时可用的占位模型（**仅用于让链路跑起来**，精度不可信）。

    内参按"水平视场角约 60°"粗估，外参用 :data:`DEFAULT_R_BC`。真要用坐标解算
    必须先标定——所以 :func:`load_camera_model` 找不到文件时会 warning。
    """
    fx = width / (2.0 * math.tan(math.radians(60.0) / 2.0))
    matrix = np.array([[fx, 0.0, width / 2.0], [0.0, fx, height / 2.0], [0.0, 0.0, 1.0]])
    return CameraModel(
        camera_matrix=matrix,
        image_size=(int(width), int(height)),
        source=None,
    )


def load_camera_model(
    path: str | Path,
    *,
    fallback_size: tuple[int, int] = (1280, 720),
) -> CameraModel:
    """读 ``camera_calib.json``；缺失/损坏时退化为默认模型并 warning。

    文件格式（``tools/calibrate.py`` 产出）::

        {
          "camera_matrix": [[fx,0,cx],[0,fy,cy],[0,0,1]],
          "dist_coeffs": [k1,k2,p1,p2,k3],
          "R_bc": [[...],[...],[...]],     # 相机系 → 机体系
          "t_bc": [x, y, z],               # 相机光心在机体系的位置（米）
          "image_size": [width, height],
          "telemetry_lag": 0.15,
          "meta": {...}                    # 标定报告（RMS/残差等），本函数忽略
        }
    """
    file_path = Path(path)
    if not file_path.is_file():
        LOGGER.warning(
            "标定文件不存在: %s —— 使用默认相机模型（外参=绕 z +90°，内参按 60° 视场角"
            "粗估）。坐标解算结果不可信，正式任务前请先跑 tools/calibrate.py",
            file_path,
        )
        return default_camera_model(*fallback_size)
    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        LOGGER.error("标定文件无法解析: %s (%s)，使用默认相机模型", file_path, exc)
        return default_camera_model(*fallback_size)

    try:
        size = payload.get("image_size")
        model = CameraModel(
            camera_matrix=np.asarray(payload["camera_matrix"], dtype=np.float64),
            dist_coeffs=np.asarray(payload.get("dist_coeffs", [0.0] * 5), dtype=np.float64),
            r_bc=(
                np.asarray(payload["R_bc"], dtype=np.float64)
                if payload.get("R_bc") is not None
                else DEFAULT_R_BC.copy()
            ),
            t_bc=np.asarray(payload.get("t_bc", [0.0, 0.0, 0.0]), dtype=np.float64),
            image_size=(int(size[0]), int(size[1])) if size else None,
            telemetry_lag=(
                float(payload["telemetry_lag"])
                if payload.get("telemetry_lag") is not None
                else None
            ),
            source=str(file_path),
        )
    except (KeyError, TypeError, ValueError) as exc:
        LOGGER.error("标定文件字段不合法: %s (%s)，使用默认相机模型", file_path, exc)
        return default_camera_model(*fallback_size)

    LOGGER.info(
        "已加载标定 %s：fx=%.1f fy=%.1f size=%s lag=%s",
        file_path,
        model.fx,
        model.fy,
        model.image_size,
        model.telemetry_lag,
    )
    return model
