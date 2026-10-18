"""像素 + 遥测 → NED：视线与地平面求交。

与旧版实现的关系（有意改变的一处）
----------------------------------
旧版实现的 ``pixel_to_ned`` 先用目标已知边长估一个深度
（``depth = f·S/(side_px·cos_tilt)``），再用这个深度把像素反投影出去。它把
"深度估计"和"反投影"混在一次计算里，而两者其实互相矛盾：用估出的深度反投影得到
地面点之后，那个点的真实深度并不等于最初估的值（除非像素正好在光心）。小目标、
小视场角下这个不一致很小。

本模块改成精确做法：像素确定一条视线，视线与地面平面求交。目标边长改为
独立的交叉验证量（见 :func:`estimate_depth_by_side`），不再参与主解算——
这样坐标系一改（换安装角、换标定）就能立刻在验证里发现，而不会像旧代码那样
让误差藏在"深度近似"里。

流程
----
1. 去畸变（``cv2.undistortPoints``，用标定畸变系数）；
2. 像素 → 相机系方向向量 ``d_cam = K⁻¹·[u,v,1]``（相机系：x 右、y 下、z 前）；
3. 相机系 → 机体系：``R_bc``（标定外参，未标定时为绕 z +90°）；
4. 机体系 → NED：``R_nb``（由姿态四元数/欧拉角构造）；
5. 相机光心在 NED 中的位置 = 机体位置 + ``R_nb @ t_bc``（杆臂）；
6. 与平面 ``z = ground_z`` 求交：``s = (ground_z - cam_down) / d_ned[2]``。
"""

from __future__ import annotations

import logging
import math
from typing import NamedTuple, cast

import cv2
import numpy as np

from .camera import CameraModel, euler_to_matrix

LOGGER = logging.getLogger(__name__)

__all__ = [
    "GroundIntersection",
    "ProjectedPoint",
    "SideLengthCheck",
    "cross_check_by_side",
    "estimate_depth_by_side",
    "ned_to_pixel",
    "ned_to_pixels",
    "normalize",
    "pixel_to_ned",
    "pixel_to_ray_ned",
    "quaternion_to_matrix",
]


#: 视线与地面夹角过小时（视线几乎与地面平行）求交病态，低于此余弦值判失败
MIN_DOWN_COS = 0.05


class GroundIntersection(NamedTuple):
    """一次像素→地面解算的结果。

    ``ned`` 是地面交点（NED，米）；``ok=False`` 时其余字段仍可参考（排故用）。
    """

    ok: bool
    ned: tuple[float, float, float] | None
    reason: str = ""
    depth_m: float = 0.0  # 相机光心沿光轴到交点的距离
    distance_h_m: float = 0.0  # 飞机到交点的水平距离
    ray_ned: tuple[float, float, float] = (0.0, 0.0, 0.0)
    cam_origin_ned: tuple[float, float, float] = (0.0, 0.0, 0.0)


class ProjectedPoint(NamedTuple):
    """NED 点到图像的投影；越界点保留有限像素供边缘指示使用。"""

    ned: tuple[float, float, float] | None
    pixel: tuple[float, float] | None
    valid: bool
    in_bounds: bool
    reason: str = ""


class SideLengthCheck(NamedTuple):
    """边长法深度与地面求交深度的交叉验证结果。"""

    ok: bool
    depth_by_side_m: float
    depth_by_intersection_m: float
    relative_error: float
    reason: str = ""


def normalize(vector: np.ndarray) -> np.ndarray:
    """单位化；零向量返回原样（调用方需自行判断）。"""
    array = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(array))
    if norm < 1e-12:
        return array
    return array / norm


def quaternion_to_matrix(
    w: float, x: float, y: float, z: float, *, normalize_input: bool = True
) -> np.ndarray:
    """四元数 → 机体→NED 旋转矩阵。

    与 :func:`~airdrop.georef.camera.euler_to_matrix` 同一约定（3-2-1，Rz·Ry·Rx）。
    用四元数而不是欧拉角，是因为 tilt-rotor/固定翼大机动下欧拉角有万向节问题
    （计划 4.4 明确要求）。
    """
    q = np.array([w, x, y, z], dtype=np.float64)
    if normalize_input:
        norm = float(np.linalg.norm(q))
        if norm < 1e-12:
            raise ValueError("四元数模长为零")
        q = q / norm
    w, x, y, z = (float(q[0]), float(q[1]), float(q[2]), float(q[3]))
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _attitude_matrix(
    *,
    quaternion: tuple[float, float, float, float] | None,
    euler_deg: tuple[float, float, float] | None,
) -> np.ndarray:
    if quaternion is not None:
        return quaternion_to_matrix(*quaternion)
    if euler_deg is not None:
        return euler_to_matrix(*euler_deg)
    raise ValueError("必须给出四元数或欧拉角之一")


def undistort_pixel(pixel: tuple[float, float], camera: CameraModel) -> tuple[float, float]:
    """把畸变像素坐标纠正成理想针孔坐标（归一化前的像素平面坐标）。"""
    points = np.array([[pixel]], dtype=np.float64)
    if camera.dist_coeffs.size == 0 or not np.any(camera.dist_coeffs):
        return float(pixel[0]), float(pixel[1])
    corrected = cv2.undistortPoints(
        points, camera.camera_matrix, camera.dist_coeffs, P=camera.camera_matrix
    )
    return float(corrected[0, 0, 0]), float(corrected[0, 0, 1])


def pixel_to_ray_ned(
    pixel: tuple[float, float],
    *,
    camera: CameraModel,
    quaternion: tuple[float, float, float, float] | None = None,
    euler_deg: tuple[float, float, float] | None = None,
    undistort: bool = True,
) -> np.ndarray:
    """像素 → 单位视线方向（NED 系）。

    这是坐标解算的核心一步，单独暴露出来便于单测与可视化（画视线）。
    """
    u, v = pixel
    if undistort:
        u, v = undistort_pixel((u, v), camera)
    ray_cam = np.linalg.inv(camera.camera_matrix) @ np.array([u, v, 1.0])
    ray_body = camera.r_bc @ ray_cam
    rotation_nb = _attitude_matrix(quaternion=quaternion, euler_deg=euler_deg)
    return normalize(rotation_nb @ ray_body)


def pixel_to_ned(
    pixel: tuple[float, float],
    *,
    camera: CameraModel,
    ground_z: float,
    position_ned: tuple[float, float, float],
    quaternion: tuple[float, float, float, float] | None = None,
    euler_deg: tuple[float, float, float] | None = None,
    undistort: bool = True,
) -> GroundIntersection:
    """像素 + 姿态 + 位置 → 地面交点（NED）。

    ``ground_z`` 是地面在 NED 下的 z（向下为正），由
    ``Config.ground`` 的"地面点与 NED 原点高差"换算而来。
    """
    ray_ned = pixel_to_ray_ned(
        pixel,
        camera=camera,
        quaternion=quaternion,
        euler_deg=euler_deg,
        undistort=undistort,
    )
    rotation_nb = _attitude_matrix(quaternion=quaternion, euler_deg=euler_deg)
    body = np.asarray(position_ned, dtype=np.float64).reshape(3)
    cam_origin = body + rotation_nb @ camera.t_bc

    # 视线向量与相机光心（NED）都以三元组形式放进结果里，便于排故对照
    ray_triple = (float(ray_ned[0]), float(ray_ned[1]), float(ray_ned[2]))
    origin_triple = (
        float(cam_origin[0]),
        float(cam_origin[1]),
        float(cam_origin[2]),
    )

    down_component = float(ray_ned[2])
    if down_component <= MIN_DOWN_COS:
        # 视线水平或朝上：地面无限远/在背后，无法求交
        reason = "视线朝上或接近水平" if down_component <= 0 else "视线与地面夹角过小"
        return GroundIntersection(
            ok=False,
            ned=None,
            reason=reason,
            ray_ned=ray_triple,
            cam_origin_ned=origin_triple,
        )

    scale = (float(ground_z) - float(cam_origin[2])) / down_component
    if not math.isfinite(scale) or scale <= 0:
        # 地面在相机上方（高度算错或地形反了），无解
        return GroundIntersection(
            ok=False,
            ned=None,
            reason=f"解出的距离非正（{scale:.3f}），检查 ground_z 与高度符号",
            ray_ned=ray_triple,
            cam_origin_ned=origin_triple,
        )

    point = cam_origin + scale * ray_ned
    # 深度取"沿光轴"的分量，与边长法一致，便于两者对照
    ray_body = rotation_nb.T @ ray_ned
    depth = float(scale * float(ray_body[2]))
    horizontal = float(math.hypot(point[0] - body[0], point[1] - body[1]))
    return GroundIntersection(
        ok=True,
        ned=(float(point[0]), float(point[1]), float(point[2])),
        reason="",
        depth_m=depth,
        distance_h_m=horizontal,
        ray_ned=ray_triple,
        cam_origin_ned=origin_triple,
    )


def ned_to_pixel(
    point_ned: tuple[float, float, float],
    *,
    camera: CameraModel,
    position_ned: tuple[float, float, float],
    image_size: tuple[int, int],
    quaternion: tuple[float, float, float, float] | None = None,
    euler_deg: tuple[float, float, float] | None = None,
    distort: bool = True,
    camera_matrix: np.ndarray | tuple[tuple[float, float, float], ...] | None = None,
) -> ProjectedPoint:
    """按 ``pixel_to_ned`` 的姿态和杆臂约定投影 NED 点。

    ``distort=False`` 用于已经由检测器 remap 的帧；当前检测器保持 ``K'=K``。
    原始畸变图使用 ``distort=True``。有效但在画面外的点仍返回像素坐标。
    """
    point = np.asarray(point_ned, dtype=np.float64).reshape(3)
    body = np.asarray(position_ned, dtype=np.float64).reshape(3)
    if not np.isfinite(point).all() or not np.isfinite(body).all():
        return ProjectedPoint(None, None, False, False, "NED 坐标含非有限值")
    try:
        rotation_nb = _attitude_matrix(quaternion=quaternion, euler_deg=euler_deg)
    except (ValueError, TypeError) as exc:
        return ProjectedPoint(_as_ned(point), None, False, False, str(exc))
    if not np.isfinite(rotation_nb).all():
        return ProjectedPoint(_as_ned(point), None, False, False, "姿态无效")
    cam_origin = body + rotation_nb @ camera.t_bc
    point_cam = camera.r_bc.T @ rotation_nb.T @ (point - cam_origin)
    if not np.isfinite(point_cam).all():
        return ProjectedPoint(_as_ned(point), None, False, False, "相机坐标含非有限值")
    if point_cam[2] <= 1e-9:
        return ProjectedPoint(_as_ned(point), None, False, False, "目标位于相机后方")

    matrix = (
        camera.camera_matrix
        if camera_matrix is None
        else np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)
    )
    if not np.isfinite(matrix).all():
        return ProjectedPoint(_as_ned(point), None, False, False, "相机内参无效")
    distortion = camera.dist_coeffs if distort else np.zeros_like(camera.dist_coeffs)
    projected, _ = cv2.projectPoints(
        point_cam.reshape(1, 1, 3),
        np.zeros(3, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        matrix,
        distortion,
    )
    u, v = (float(item) for item in projected.reshape(2))
    if not math.isfinite(u) or not math.isfinite(v):
        return ProjectedPoint(_as_ned(point), None, False, False, "投影结果含非有限值")
    width, height = image_size
    in_bounds = 0.0 <= u < width and 0.0 <= v < height
    return ProjectedPoint(
        _as_ned(point),
        (u, v),
        True,
        in_bounds,
        "" if in_bounds else "目标投影在画面外",
    )


def ned_to_pixels(  # noqa: PLR0912
    points_ned: tuple[tuple[float, float, float] | None, ...],
    *,
    camera: CameraModel,
    position_ned: tuple[float, float, float],
    image_size: tuple[int, int],
    quaternion: tuple[float, float, float, float] | None = None,
    euler_deg: tuple[float, float, float] | None = None,
    distort: bool = True,
    camera_matrix: np.ndarray | tuple[tuple[float, float, float], ...] | None = None,
) -> tuple[ProjectedPoint, ...]:
    """批量投影观测云，复用姿态变换并一次调用 OpenCV。"""
    normalized: list[np.ndarray | None] = []
    invalid: dict[int, ProjectedPoint] = {}
    for index, point in enumerate(points_ned):
        if point is None:
            normalized.append(None)
            invalid[index] = ProjectedPoint(None, None, False, False, "观测无有效坐标")
            continue
        try:
            item = np.asarray(point, dtype=np.float64).reshape(3)
        except ValueError, TypeError:
            normalized.append(None)
            invalid[index] = ProjectedPoint(None, None, False, False, "NED 坐标维度无效")
            continue
        if not np.isfinite(item).all():
            normalized.append(None)
            invalid[index] = ProjectedPoint(None, None, False, False, "NED 坐标含非有限值")
            continue
        normalized.append(item)

    try:
        body = np.asarray(position_ned, dtype=np.float64).reshape(3)
        rotation_nb = _attitude_matrix(quaternion=quaternion, euler_deg=euler_deg)
    except (ValueError, TypeError) as exc:
        return _fallback_projection(normalized, invalid, str(exc))
    if not np.isfinite(body).all() or not np.isfinite(rotation_nb).all():
        return _fallback_projection(normalized, invalid, "位置或姿态无效")
    width, height = image_size
    if width <= 0 or height <= 0:
        return _fallback_projection(normalized, invalid, "图像尺寸无效")
    matrix = (
        camera.camera_matrix
        if camera_matrix is None
        else np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)
    )
    if not np.isfinite(matrix).all():
        return _fallback_projection(normalized, invalid, "相机内参无效")

    cam_origin = body + rotation_nb @ camera.t_bc
    camera_points: list[np.ndarray] = []
    indices: list[int] = []
    for index, point in enumerate(normalized):
        if point is None:
            continue
        point_cam = (point - cam_origin) @ rotation_nb @ camera.r_bc
        if not np.isfinite(point_cam).all():
            invalid[index] = ProjectedPoint(
                _as_ned(point), None, False, False, "相机坐标含非有限值"
            )
        elif point_cam[2] <= 1e-9:
            invalid[index] = ProjectedPoint(_as_ned(point), None, False, False, "目标位于相机后方")
        else:
            indices.append(index)
            camera_points.append(point_cam)

    results = dict(invalid)
    if camera_points:
        distortion = camera.dist_coeffs if distort else np.zeros_like(camera.dist_coeffs)
        projected, _ = cv2.projectPoints(
            np.asarray(camera_points, dtype=np.float64).reshape(-1, 1, 3),
            np.zeros(3, dtype=np.float64),
            np.zeros(3, dtype=np.float64),
            matrix,
            distortion,
        )
        for index, pixel in zip(indices, projected.reshape(-1, 2), strict=True):
            u, v = float(pixel[0]), float(pixel[1])
            point = normalized[index]
            assert point is not None
            if not math.isfinite(u) or not math.isfinite(v):
                results[index] = ProjectedPoint(
                    _as_ned(point), None, False, False, "投影结果含非有限值"
                )
                continue
            in_bounds = 0 <= u < width and 0 <= v < height
            results[index] = ProjectedPoint(
                _as_ned(point),
                (u, v),
                True,
                in_bounds,
                "" if in_bounds else "目标投影在画面外",
            )
    return tuple(results[i] for i in range(len(points_ned)))


def _fallback_projection(
    points: list[np.ndarray | None],
    invalid: dict[int, ProjectedPoint],
    reason: str,
) -> tuple[ProjectedPoint, ...]:
    """为尚未投影的有效点填写统一错误，并保留输入错误原因。"""
    result: list[ProjectedPoint] = []
    for index, point in enumerate(points):
        if index in invalid:
            result.append(invalid[index])
        else:
            assert point is not None
            result.append(ProjectedPoint(_as_ned(point), None, False, False, reason))
    return tuple(result)


def _as_ned(point: np.ndarray) -> tuple[float, float, float]:
    """将已校验的三维数组转为静态可见的三元组。"""
    return cast(tuple[float, float, float], tuple(float(value) for value in point))


def estimate_depth_by_side(
    side_px: float,
    *,
    camera: CameraModel,
    side_m: float,
    quaternion: tuple[float, float, float, float] | None = None,
    euler_deg: tuple[float, float, float] | None = None,
) -> float | None:
    """边长法独立估深度：``depth ≈ f·S / (side_px·cos_tilt)``。

    这是旧版实现的主算法；这里只作为交叉验证（计划 4.4 要求互校）。
    ``cos_tilt`` 是光轴与竖直方向夹角的余弦——光轴越倾斜，同样的像素边长对应
    越远的实际距离。

    返回 None 表示无法估计（边长非正，或光轴接近水平导致发散）。
    """
    if side_px <= 0 or side_m <= 0:
        return None
    rotation_nb = _attitude_matrix(quaternion=quaternion, euler_deg=euler_deg)
    # 光轴在机体系是 +z；在 NED 下是 R_nb @ R_bc @ [0,0,1]
    axis_ned = rotation_nb @ (camera.r_bc @ np.array([0.0, 0.0, 1.0]))
    cos_tilt = abs(float(axis_ned[2]))
    if cos_tilt < MIN_DOWN_COS:
        return None
    return camera.focal_px * side_m / (side_px * cos_tilt)


def cross_check_by_side(
    *,
    side_px: float,
    side_m: float,
    depth_by_intersection_m: float,
    camera: CameraModel,
    tolerance: float = 0.25,
    quaternion: tuple[float, float, float, float] | None = None,
    euler_deg: tuple[float, float, float] | None = None,
) -> SideLengthCheck:
    """用已知边长独立估深度，与地面求交的深度互校。

    ``tolerance`` 是允许的相对误差；超出时不参与结果（调用方应记事件日志）。
    两者差异大的常见原因：目标不在平地假设上、检测框把编号方块当成了整个目标、
    或者外参/内参标定错了。
    """
    by_side = estimate_depth_by_side(
        side_px,
        camera=camera,
        side_m=side_m,
        quaternion=quaternion,
        euler_deg=euler_deg,
    )
    if by_side is None:
        return SideLengthCheck(
            ok=False,
            depth_by_side_m=float("nan"),
            depth_by_intersection_m=depth_by_intersection_m,
            relative_error=float("nan"),
            reason="边长法无法估深度（边长非正或光轴接近水平）",
        )
    if depth_by_intersection_m <= 0:
        return SideLengthCheck(
            ok=False,
            depth_by_side_m=by_side,
            depth_by_intersection_m=depth_by_intersection_m,
            relative_error=float("nan"),
            reason="地面求交深度非正",
        )
    relative = abs(by_side - depth_by_intersection_m) / depth_by_intersection_m
    return SideLengthCheck(
        ok=relative <= tolerance,
        depth_by_side_m=by_side,
        depth_by_intersection_m=depth_by_intersection_m,
        relative_error=relative,
        reason=""
        if relative <= tolerance
        else f"两种深度差 {relative * 100:.1f}%，超出 {tolerance * 100:.0f}%",
    )
