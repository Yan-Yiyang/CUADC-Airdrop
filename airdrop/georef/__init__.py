"""坐标处理：像素 → NED → WGS84。

模块划分
--------
* :mod:`~airdrop.georef.camera`：相机模型（内参 / 畸变 / **标定产出的相机→机体外参**）；
* :mod:`~airdrop.georef.project`：像素 + 遥测 → 地面交点（视线与 ``z=ground_z`` 求交），
  以及边长法深度的交叉验证；
* :mod:`~airdrop.georef.geo`：NED ↔ WGS84（pyproj，移植自 旧版，已过 Geod 真值验证）。

坐标约定见 :mod:`~airdrop.georef.camera` 的模块 docstring——相机系与机体系的轴
定义、``R_bc`` 的方向（相机→机体）都在那里写死了，改之前先读它。
"""

from .camera import (
    DEFAULT_R_BC,
    CameraModel,
    default_camera_model,
    euler_to_matrix,
    load_camera_model,
    rotation_x,
    rotation_y,
    rotation_z,
)
from .geo import LLARef, ned_distance, ned_to_wgs84, wgs84_to_ned
from .project import (
    GroundIntersection,
    SideLengthCheck,
    MIN_DOWN_COS,
    cross_check_by_side,
    estimate_depth_by_side,
    pixel_to_ned,
    pixel_to_ray_ned,
    quaternion_to_matrix,
    undistort_pixel,
)

__all__ = [
    "CameraModel",
    "DEFAULT_R_BC",
    "GroundIntersection",
    "LLARef",
    "MIN_DOWN_COS",
    "SideLengthCheck",
    "cross_check_by_side",
    "default_camera_model",
    "estimate_depth_by_side",
    "euler_to_matrix",
    "load_camera_model",
    "ned_distance",
    "ned_to_wgs84",
    "pixel_to_ned",
    "pixel_to_ray_ned",
    "quaternion_to_matrix",
    "rotation_x",
    "rotation_y",
    "rotation_z",
    "undistort_pixel",
    "wgs84_to_ned",
]
