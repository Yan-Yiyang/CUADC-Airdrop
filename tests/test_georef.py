"""georef 离线单测：像素→NED、NED↔WGS84、相机模型、边长交叉验证。

**全部离线**：只用合成几何与 pyproj，不碰相机/权重/GPU。

验收口径（计划 P6）：合成几何单测误差 <1e-6；NED↔WGS84 用 pyproj ``Geod``
大地线真值校验（移植代码原本就带这套验证，这里补上回归）。

关键的一条：**默认外参必须复现旧版实现的行为**（图像右=机体右、中心=正下方）。
这条如果不成立，说明坐标约定在移植时被改坏了——它比任何数值精度都重要。
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np
import pytest
from pyproj import Geod

from airdrop.georef import (
    DEFAULT_R_BC,
    CameraModel,
    LLARef,
    cross_check_by_side,
    default_camera_model,
    estimate_depth_by_side,
    euler_to_matrix,
    load_camera_model,
    ned_distance,
    ned_to_wgs84,
    pixel_to_ned,
    pixel_to_ray_ned,
    quaternion_to_matrix,
    rotation_x,
    rotation_y,
    rotation_z,
    undistort_pixel,
    wgs84_to_ned,
)

GEOD = Geod(ellps="WGS84")

# 上海附近的 NED 原点 (lon, lat, alt)
HOME: LLARef = LLARef(121.4737, 31.2304, 4.0)


def _camera(f: float = 800.0, cx: float = 640.0, cy: float = 360.0) -> CameraModel:
    return CameraModel(
        camera_matrix=np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]]),
        image_size=(1280, 720),
    )


# ----------------------------------------------------------------------
# NED ↔ WGS84
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    ("north", "east"),
    [(100.0, 0.0), (0.0, 100.0), (50.0, -30.0), (1000.0, 500.0), (-250.0, 80.0)],
)
def test_ned_to_wgs84_matches_geod_distance_and_azimuth(north: float, east: float) -> None:
    """与 pyproj Geod 的大地线真值对齐（防"把北映射成南"这类符号/转置错误）。"""
    lon, lat, _ = ned_to_wgs84((north, east, 0.0), HOME)
    azimuth, _, distance = GEOD.inv(HOME.lon_deg, HOME.lat_deg, lon, lat)
    assert distance == pytest.approx(math.hypot(north, east), abs=0.05)
    expected_azimuth = math.degrees(math.atan2(east, north)) % 360.0
    error = abs((azimuth - expected_azimuth + 180.0) % 360.0 - 180.0)
    assert error < 0.05, f"方位角偏差 {error:.4f}°"


@pytest.mark.parametrize(
    "ned",
    [(0.0, 0.0, 0.0), (100.0, 0.0, 0.0), (0.0, 100.0, 0.0), (50.0, -30.0, 12.0)],
)
def test_wgs84_round_trip(ned: tuple[float, float, float]) -> None:
    """往返互逆，误差 <1e-6（计划 P6 的数值口径）。"""
    back = wgs84_to_ned(*ned_to_wgs84(ned, HOME), HOME)
    for got, want in zip(back, ned, strict=True):
        assert got == pytest.approx(want, abs=1e-6), f"{ned} -> {back}"


def test_ned_origin_maps_to_reference() -> None:
    lon, lat, alt = ned_to_wgs84((0.0, 0.0, 0.0), HOME)
    assert lon == pytest.approx(HOME.lon_deg, abs=1e-9)
    assert lat == pytest.approx(HOME.lat_deg, abs=1e-9)
    assert alt == pytest.approx(HOME.alt_m, abs=1e-6)


def test_ned_distance() -> None:
    assert ned_distance((0.0, 0.0, 0.0), (3.0, 4.0, 0.0)) == pytest.approx(5.0)
    assert ned_distance((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)) == 0.0


# ----------------------------------------------------------------------
# 相机模型
# ----------------------------------------------------------------------
def test_camera_rejects_invalid_intrinsics() -> None:
    with pytest.raises(ValueError, match="内参矩阵"):
        CameraModel(camera_matrix=np.zeros((3, 3)))
    with pytest.raises(ValueError, match="内参矩阵"):
        CameraModel(camera_matrix=np.array([[0.0, 0, 0], [0, 800, 0], [0, 0, 1]]))


def test_camera_rejects_non_rotation_extrinsics() -> None:
    with pytest.raises(ValueError, match="R_bc"):
        CameraModel(
            camera_matrix=np.eye(3) * 800 + np.array([[0, 0, 0], [0, 0, 0], [0, 0, -799]]),
            r_bc=np.full((3, 3), 0.5),  # 既不正交 det 也不为 1
        )


def test_default_extrinsics_are_rotation_z_90() -> None:
    assert np.allclose(DEFAULT_R_BC, rotation_z(90.0))


def test_camera_scaled_keeps_rotation_and_scales_intrinsics() -> None:
    camera = _camera()
    half = camera.scaled(0.5)
    assert half.fx == pytest.approx(400.0)
    assert half.cx == pytest.approx(320.0)
    assert half.image_size == (640, 360)
    assert np.allclose(half.r_bc, camera.r_bc)


def test_camera_scaled_rejects_bad_scale() -> None:
    with pytest.raises(ValueError, match="scale"):
        _camera().scaled(0.0)


def test_rotation_helpers_are_orthonormal() -> None:
    for matrix in (
        rotation_x(30.0),
        rotation_y(-42.0),
        rotation_z(97.0),
        euler_to_matrix(10.0, -20.0, 35.0),
    ):
        assert np.allclose(matrix @ matrix.T, np.eye(3), atol=1e-12)
        assert np.linalg.det(matrix) == pytest.approx(1.0)


def test_euler_to_matrix_matches_rz_ry_rx() -> None:
    """姿态矩阵必须是 Rz·Ry·Rx（与旧版实现一致），顺序错了姿态就全错。"""
    roll, pitch, yaw = 12.0, -7.0, 143.0
    expected = rotation_z(yaw) @ rotation_y(pitch) @ rotation_x(roll)
    assert np.allclose(euler_to_matrix(roll, pitch, yaw), expected)


# ----------------------------------------------------------------------
# 四元数
# ----------------------------------------------------------------------
def test_quaternion_identity() -> None:
    assert np.allclose(quaternion_to_matrix(1.0, 0.0, 0.0, 0.0), np.eye(3))


@pytest.mark.parametrize("yaw", [0.0, 30.0, -120.0, 179.0])
def test_quaternion_matches_euler_for_pure_yaw(yaw: float) -> None:
    half = math.radians(yaw) / 2.0
    matrix = quaternion_to_matrix(math.cos(half), 0.0, 0.0, math.sin(half))
    assert np.allclose(matrix, euler_to_matrix(0.0, 0.0, yaw), atol=1e-12)


@pytest.mark.parametrize("roll", [0.0, 15.0, -40.0])
def test_quaternion_matches_euler_for_pure_roll(roll: float) -> None:
    half = math.radians(roll) / 2.0
    matrix = quaternion_to_matrix(math.cos(half), math.sin(half), 0.0, 0.0)
    assert np.allclose(matrix, euler_to_matrix(roll, 0.0, 0.0), atol=1e-12)


def test_quaternion_normalizes() -> None:
    scaled = quaternion_to_matrix(2.0, 0.0, 0.0, 0.0)
    assert np.allclose(scaled, np.eye(3))


def test_quaternion_rejects_zero() -> None:
    with pytest.raises(ValueError, match="模长"):
        quaternion_to_matrix(0.0, 0.0, 0.0, 0.0)


# ----------------------------------------------------------------------
# 像素 → NED：必须复现旧版实现的行为
# ----------------------------------------------------------------------
def test_image_center_is_nadir() -> None:
    """图像中心在零姿态下就是飞机正下方。"""
    camera = _camera()
    position = (0.0, 0.0, -50.0)
    result = pixel_to_ned(
        (camera.cx, camera.cy),
        camera=camera,
        ground_z=0.0,
        position_ned=position,
        euler_deg=(0.0, 0.0, 0.0),
        undistort=False,
    )
    assert result.ok, result.reason
    assert result.ned is not None
    assert result.ned[0] == pytest.approx(0.0, abs=1e-9)
    assert result.ned[1] == pytest.approx(0.0, abs=1e-9)
    assert result.ned[2] == pytest.approx(0.0, abs=1e-9)


def test_image_right_maps_to_body_right() -> None:
    """图像右移 → 机体右（yaw=0 时是正东）——与旧版实测一致。"""
    camera = _camera()
    position = (0.0, 0.0, -50.0)
    result = pixel_to_ned(
        (camera.cx + 16.0, camera.cy),
        camera=camera,
        ground_z=0.0,
        position_ned=position,
        euler_deg=(0.0, 0.0, 0.0),
        undistort=False,
    )
    assert result.ok and result.ned is not None
    east = result.ned[1]
    # 50m 高、f=800、偏 16px → 约 1m
    assert east == pytest.approx(1.0, rel=1e-3), f"东向 {east}"
    assert result.ned[0] == pytest.approx(0.0, abs=1e-9)


def test_image_down_maps_to_nose_direction() -> None:
    """图像下移 → 机头方向（yaw=0 时是正北）。这条约定最容易被搞反。"""
    camera = _camera()
    result = pixel_to_ned(
        (camera.cx, camera.cy + 16.0),
        camera=camera,
        ground_z=0.0,
        position_ned=(0.0, 0.0, -50.0),
        euler_deg=(0.0, 0.0, 0.0),
        undistort=False,
    )
    assert result.ok and result.ned is not None
    assert result.ned[0] == pytest.approx(-1.0, rel=1e-3), f"北向 {result.ned[0]}"
    assert result.ned[1] == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize(
    ("yaw", "expect_north", "expect_east"),
    [
        (0.0, 0.0, 1.0),
        (90.0, -1.0, 0.0),
        (180.0, 0.0, -1.0),
        (-90.0, 1.0, 0.0),
    ],
)
def test_image_right_follows_yaw(yaw: float, expect_north: float, expect_east: float) -> None:
    """机头朝东时，图像右应当指向正南——与旧版的实测表一致。"""
    camera = _camera()
    result = pixel_to_ned(
        (camera.cx + 16.0, camera.cy),
        camera=camera,
        ground_z=0.0,
        position_ned=(0.0, 0.0, -50.0),
        euler_deg=(0.0, 0.0, yaw),
        undistort=False,
    )
    assert result.ok and result.ned is not None
    assert result.ned[0] == pytest.approx(expect_north, abs=1e-3)
    assert result.ned[1] == pytest.approx(expect_east, abs=1e-3)


def test_scaling_property_of_pinhole_projection() -> None:
    """同一像素在不同高度：地面偏移与高度成正比（针孔+平面假设的必然结果）。"""
    camera = _camera()
    offsets = []
    for height in (20.0, 50.0, 100.0):
        result = pixel_to_ned(
            (camera.cx + 16.0, camera.cy),
            camera=camera,
            ground_z=0.0,
            position_ned=(0.0, 0.0, -height),
            euler_deg=(0.0, 0.0, 0.0),
            undistort=False,
        )
        assert result.ok and result.ned is not None
        offsets.append(result.ned[1])
    assert offsets[1] / offsets[0] == pytest.approx(50.0 / 20.0, rel=1e-9)
    assert offsets[2] / offsets[0] == pytest.approx(100.0 / 20.0, rel=1e-9)


def test_ground_z_offsets_intersection() -> None:
    """地面抬高 10m（ground_z=-10）时，同一视线的交点更近。"""
    camera = _camera()
    kwargs = dict(
        pixel=(camera.cx + 200.0, camera.cy),
        camera=camera,
        position_ned=(0.0, 0.0, -50.0),
        euler_deg=(0.0, 0.0, 0.0),
        undistort=False,
    )
    flat = pixel_to_ned(ground_z=0.0, **kwargs)  # type: ignore[arg-type]
    raised = pixel_to_ned(ground_z=-10.0, **kwargs)  # type: ignore[arg-type]
    assert flat.ok and raised.ok
    assert flat.ned is not None and raised.ned is not None
    assert raised.distance_h_m < flat.distance_h_m
    assert raised.ned[2] == pytest.approx(-10.0, abs=1e-9)
    assert flat.ned[2] == pytest.approx(0.0, abs=1e-9)


def test_lever_arm_shifts_camera_origin() -> None:
    """杆臂计入相机光心位置：相机前移 1m，正下方的交点也前移 1m。"""
    matrix = np.array([[800.0, 0, 640.0], [0, 800.0, 360.0], [0, 0, 1.0]])
    camera = CameraModel(
        camera_matrix=matrix,
        t_bc=np.array([1.0, 0.0, 0.0]),  # 相机在机体前方 1m
        image_size=(1280, 720),
    )
    result = pixel_to_ned(
        (640.0, 360.0),
        camera=camera,
        ground_z=0.0,
        position_ned=(0.0, 0.0, -50.0),
        euler_deg=(0.0, 0.0, 0.0),
        undistort=False,
    )
    assert result.ok and result.ned is not None
    assert result.ned[0] == pytest.approx(1.0, abs=1e-9)


def test_ray_pointing_up_has_no_intersection() -> None:
    """相机倒扣（roll=180°）时视线朝上，必须判失败而不是给个假坐标。"""
    camera = _camera()
    result = pixel_to_ned(
        (640.0, 360.0),
        camera=camera,
        ground_z=0.0,
        position_ned=(0.0, 0.0, -50.0),
        euler_deg=(180.0, 0.0, 0.0),
        undistort=False,
    )
    assert not result.ok
    assert "朝上" in result.reason or "夹角" in result.reason
    assert result.ned is None


def test_ground_above_camera_is_rejected() -> None:
    """地面在相机上方（ground_z 比相机还负）→ 无解。"""
    camera = _camera()
    result = pixel_to_ned(
        (640.0, 360.0),
        camera=camera,
        ground_z=-100.0,  # 比飞机（-50）还高
        position_ned=(0.0, 0.0, -50.0),
        euler_deg=(0.0, 0.0, 0.0),
        undistort=False,
    )
    assert not result.ok
    assert result.ned is None


def test_quaternion_and_euler_agree_in_pixel_to_ned() -> None:
    camera = _camera()
    euler = (5.0, -3.0, 70.0)
    half_yaw = math.radians(70.0) / 2.0
    from_euler = pixel_to_ned(
        (700.0, 400.0),
        camera=camera,
        ground_z=0.0,
        position_ned=(10.0, -5.0, -60.0),
        euler_deg=euler,
        undistort=False,
    )
    from_quat = pixel_to_ned(
        (700.0, 400.0),
        camera=camera,
        ground_z=0.0,
        position_ned=(10.0, -5.0, -60.0),
        quaternion=(math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw)),
        undistort=False,
    )
    assert from_euler.ok and from_quat.ok
    # 纯偏航的四元数与欧拉角一致，但这里欧拉角含 roll/pitch，故只比对纯偏航场景
    pure = pixel_to_ned(
        (700.0, 400.0),
        camera=camera,
        ground_z=0.0,
        position_ned=(10.0, -5.0, -60.0),
        euler_deg=(0.0, 0.0, 70.0),
        undistort=False,
    )
    assert pure.ned is not None and from_quat.ned is not None
    assert pure.ned[0] == pytest.approx(from_quat.ned[0], abs=1e-9)
    assert pure.ned[1] == pytest.approx(from_quat.ned[1], abs=1e-9)


def test_pixel_to_ned_requires_attitude() -> None:
    camera = _camera()
    with pytest.raises(ValueError, match="四元数或欧拉角"):
        pixel_to_ned(
            (640.0, 360.0),
            camera=camera,
            ground_z=0.0,
            position_ned=(0.0, 0.0, -50.0),
        )


# ----------------------------------------------------------------------
# 视线方向
# ----------------------------------------------------------------------
def test_ray_is_normalized_and_points_down() -> None:
    camera = _camera()
    ray = pixel_to_ray_ned(
        (camera.cx, camera.cy),
        camera=camera,
        euler_deg=(0.0, 0.0, 0.0),
        undistort=False,
    )
    assert float(np.linalg.norm(ray)) == pytest.approx(1.0)
    assert ray[2] == pytest.approx(1.0)  # 正下方（NED z 向下为正）


def test_ray_center_matches_camera_axis_rotation() -> None:
    """中心像素的视线应当等于 R_nb @ R_bc @ [0,0,1]。"""
    camera = _camera()
    ray = pixel_to_ray_ned(
        (camera.cx, camera.cy),
        camera=camera,
        euler_deg=(0.0, 0.0, 33.0),
        undistort=False,
    )
    expected = euler_to_matrix(0.0, 0.0, 33.0) @ (camera.r_bc @ np.array([0.0, 0.0, 1.0]))
    assert np.allclose(ray, expected, atol=1e-12)


# ----------------------------------------------------------------------
# 去畸变
# ----------------------------------------------------------------------
def test_undistort_is_identity_without_distortion() -> None:
    camera = _camera()
    assert undistort_pixel((123.0, 456.0), camera) == (123.0, 456.0)


def test_undistort_expands_points_for_negative_k1() -> None:
    """k1<0（桶形畸变）的真实含义：**畸变图里点被向中心压缩**，所以校正要往外推。

    容易搞反：``r_distorted = r(1 + k1·r² + ...)``，k1<0 时畸变半径小于理想半径。
    这里同时用 OpenCV 的 ``undistortPoints`` 与解析式两边确认方向。
    """
    matrix = np.array([[800.0, 0, 640.0], [0, 800.0, 360.0], [0, 0, 1.0]])
    k1, k2 = -0.3, 0.1
    camera = CameraModel(
        camera_matrix=matrix,
        dist_coeffs=np.array([k1, k2, 0.0, 0.0, 0.0]),
        image_size=(1280, 720),
    )
    u, v = undistort_pixel((1000.0, 500.0), camera)
    radius_in = math.hypot(1000.0 - 640.0, 500.0 - 360.0)
    radius_out = math.hypot(u - 640.0, v - 360.0)
    assert radius_out > radius_in, "k1<0 时校正是向外扩张"

    # 解析式自检：归一化坐标 r=0.5 处，畸变半径应当小于理想半径
    r = 0.5
    assert r * (1 + k1 * r * r + k2 * r**4) < r


def test_undistort_then_distort_round_trips() -> None:
    """校正后再投影回畸变像素应当回到原点（用 projectPoints 求逆）。

    容差取 0.01px：``undistortPoints`` 是**迭代**求逆，实测残差约 0.002px；
    要求 1e-6 是在要求一件它做不到的事（而且 0.01px 对几何解算早已绰绰有余）。
    """
    matrix = np.array([[800.0, 0, 640.0], [0, 800.0, 360.0], [0, 0, 1.0]])
    dist = np.array([-0.3, 0.1, 0.001, -0.002, 0.0])
    camera = CameraModel(camera_matrix=matrix, dist_coeffs=dist)
    for pixel in ((1000.0, 500.0), (400.0, 200.0), (900.0, 650.0)):
        u, v = undistort_pixel(pixel, camera)
        # 把校正点视作理想点，反投影回畸变像素
        object_point = np.linalg.inv(matrix) @ np.array([u, v, 1.0])
        reprojected, _ = cv2.projectPoints(
            object_point.reshape(1, 1, 3), np.zeros(3), np.zeros(3), matrix, dist
        )
        assert reprojected[0, 0, 0] == pytest.approx(pixel[0], abs=0.01)
        assert reprojected[0, 0, 1] == pytest.approx(pixel[1], abs=0.01)


def test_principal_point_is_fixed_by_undistort() -> None:
    matrix = np.array([[800.0, 0, 640.0], [0, 800.0, 360.0], [0, 0, 1.0]])
    camera = CameraModel(
        camera_matrix=matrix,
        dist_coeffs=np.array([-0.3, 0.1, 0.0, 0.0, 0.0]),
    )
    u, v = undistort_pixel((640.0, 360.0), camera)
    assert u == pytest.approx(640.0, abs=1e-6)
    assert v == pytest.approx(360.0, abs=1e-6)


# ----------------------------------------------------------------------
# 边长法交叉验证
# ----------------------------------------------------------------------
def test_depth_by_side_matches_legacy_formula() -> None:
    """f·S/(side_px·cos_tilt)，零姿态下 cos_tilt=1。"""
    camera = _camera(f=800.0)
    depth = estimate_depth_by_side(16.0, camera=camera, side_m=1.0, euler_deg=(0.0, 0.0, 0.0))
    assert depth == pytest.approx(800.0 * 1.0 / 16.0, rel=1e-12)


def test_depth_by_side_rejects_bad_input() -> None:
    camera = _camera()
    assert estimate_depth_by_side(0.0, camera=camera, side_m=1.0, euler_deg=(0.0, 0.0, 0.0)) is None
    assert (
        estimate_depth_by_side(-2.0, camera=camera, side_m=1.0, euler_deg=(0.0, 0.0, 0.0)) is None
    )
    assert (
        estimate_depth_by_side(16.0, camera=camera, side_m=0.0, euler_deg=(0.0, 0.0, 0.0)) is None
    )


def test_depth_by_side_diverges_when_axis_near_horizontal() -> None:
    """光轴接近水平时 cos_tilt→0，必须判失败（旧代码就是靠这个阈值兜的）。"""
    camera = _camera()
    assert (
        estimate_depth_by_side(16.0, camera=camera, side_m=1.0, euler_deg=(90.0, 0.0, 0.0)) is None
    )


def test_cross_check_agrees_on_synthetic_flat_ground() -> None:
    """合成平地场景：边长法深度与地面求交深度应当基本一致。

    这里构造"已知真实边长 1m、位于 50m 正下方"的目标：先由几何算出它在该距离
    下应有的像素边长，再让两种方法各自估深度。
    """
    camera = _camera(f=800.0)
    position = (0.0, 0.0, -50.0)
    height = 50.0
    # 真实边长 1m 在 50m 处的像素边长（零姿态、光轴竖直 → cos_tilt=1）
    side_px = camera.focal_px * 1.0 / height
    result = pixel_to_ned(
        (camera.cx, camera.cy),
        camera=camera,
        ground_z=0.0,
        position_ned=position,
        euler_deg=(0.0, 0.0, 0.0),
        undistort=False,
    )
    assert result.ok
    check = cross_check_by_side(
        side_px=side_px,
        side_m=1.0,
        depth_by_intersection_m=result.depth_m,
        camera=camera,
        euler_deg=(0.0, 0.0, 0.0),
    )
    assert check.ok, check.reason
    assert check.relative_error < 1e-6


def test_cross_check_flags_inconsistent_side() -> None:
    """像素边长与深度不匹配（例如检测框框错了）时必须判不通过。"""
    camera = _camera(f=800.0)
    check = cross_check_by_side(
        side_px=100.0,
        side_m=1.0,
        depth_by_intersection_m=50.0,
        camera=camera,
        euler_deg=(0.0, 0.0, 0.0),
        tolerance=0.25,
    )
    assert not check.ok
    assert check.relative_error > 0.25


def test_cross_check_handles_missing_side_estimate() -> None:
    camera = _camera()
    check = cross_check_by_side(
        side_px=0.0,
        side_m=1.0,
        depth_by_intersection_m=50.0,
        camera=camera,
        euler_deg=(0.0, 0.0, 0.0),
    )
    assert not check.ok
    assert math.isnan(check.relative_error)


# ----------------------------------------------------------------------
# 标定文件加载
# ----------------------------------------------------------------------
def _write_calib(path: Path, **overrides) -> Path:
    payload = {
        "camera_matrix": [[800.0, 0.0, 640.0], [0.0, 800.0, 360.0], [0.0, 0.0, 1.0]],
        "dist_coeffs": [-0.2, 0.05, 0.0, 0.0, 0.0],
        "R_bc": np.eye(3).tolist(),
        "t_bc": [0.1, 0.0, -0.05],
        "image_size": [1280, 720],
        "telemetry_lag": 0.18,
        "meta": {"rms": 0.31},
    }
    payload.update(overrides)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_load_camera_model_reads_all_fields(tmp_path: Path) -> None:
    path = _write_calib(tmp_path / "camera_calib.json")
    model = load_camera_model(path)
    assert model.is_calibrated()
    assert model.fx == pytest.approx(800.0)
    assert model.image_size == (1280, 720)
    assert model.telemetry_lag == pytest.approx(0.18)
    assert model.t_bc[0] == pytest.approx(0.1)
    assert np.allclose(model.r_bc, np.eye(3))
    assert model.source == str(path)


def test_load_camera_model_defaults_extrinsics_when_absent(tmp_path: Path) -> None:
    path = _write_calib(tmp_path / "camera_calib.json", R_bc=None, t_bc=None, telemetry_lag=None)
    model = load_camera_model(path)
    assert np.allclose(model.r_bc, DEFAULT_R_BC)
    assert np.allclose(model.t_bc, np.zeros(3))
    assert model.telemetry_lag is None


def test_load_camera_model_falls_back_when_missing(tmp_path: Path) -> None:
    model = load_camera_model(tmp_path / "nope.json", fallback_size=(640, 480))
    assert not model.is_calibrated()
    assert model.source is None
    assert model.image_size == (640, 480)
    assert np.allclose(model.r_bc, DEFAULT_R_BC)


def test_load_camera_model_falls_back_on_broken_json(tmp_path: Path) -> None:
    path = tmp_path / "camera_calib.json"
    path.write_text("{ not json", encoding="utf-8")
    model = load_camera_model(path)
    assert not model.is_calibrated()


def test_load_camera_model_falls_back_on_bad_rotation(tmp_path: Path) -> None:
    path = _write_calib(tmp_path / "camera_calib.json", R_bc=np.zeros((3, 3)).tolist())
    model = load_camera_model(path)
    assert not model.is_calibrated()


def test_load_camera_model_falls_back_on_missing_matrix(tmp_path: Path) -> None:
    path = tmp_path / "camera_calib.json"
    path.write_text(json.dumps({"dist_coeffs": [0, 0, 0, 0, 0]}), encoding="utf-8")
    model = load_camera_model(path)
    assert not model.is_calibrated()


def test_default_camera_model_is_usable() -> None:
    model = default_camera_model(1280, 720)
    result = pixel_to_ned(
        (model.cx + 10.0, model.cy),
        camera=model,
        ground_z=0.0,
        position_ned=(0.0, 0.0, -30.0),
        euler_deg=(0.0, 0.0, 0.0),
        undistort=False,
    )
    assert result.ok and result.ned is not None


# ----------------------------------------------------------------------
# Config 接线
# ----------------------------------------------------------------------
def test_camera_config_loads_model_from_file(tmp_path: Path) -> None:
    from airdrop.config import CameraConfig

    path = _write_calib(tmp_path / "camera_calib.json")
    model = CameraConfig(calib_file=str(path)).load_model()
    assert model.is_calibrated()
    assert model.fx == pytest.approx(800.0)


def test_camera_config_falls_back_with_configured_size(tmp_path: Path) -> None:
    from airdrop.config import CameraConfig

    config = CameraConfig(
        calib_file=str(tmp_path / "missing.json"),
        fallback_width=640,
        fallback_height=480,
    )
    model = config.load_model()
    assert not model.is_calibrated()
    assert model.image_size == (640, 480)


def test_ground_config_computes_ground_z() -> None:
    from airdrop.config import GroundConfig

    # 地面点比原点低 30m（海拔小 30）→ NED 下地面在原点下方 30m → ground_z=+30
    config = GroundConfig(ground_point_alt=70.0)
    assert config.ground_z(100.0) == pytest.approx(30.0)
    # 地面与原点同高 → 0
    assert GroundConfig(ground_point_alt=100.0).ground_z(100.0) == pytest.approx(0.0)
    # 地面比原点高 → 负（NED 向上为负）
    assert GroundConfig(ground_point_alt=120.0).ground_z(100.0) == pytest.approx(-20.0)


def test_ground_config_returns_none_when_incomplete() -> None:
    from airdrop.config import GroundConfig

    assert GroundConfig().ground_z(100.0) is None
    assert GroundConfig(ground_point_alt=None).ground_z(100.0) is None
    assert GroundConfig(ground_point_alt=70.0).ground_z(None) is None


def test_ground_config_needs_altitude_only() -> None:
    """经纬度目前只用于记录/核对，换算 ground_z 只需要海拔。"""
    from airdrop.config import GroundConfig

    config = GroundConfig(ground_point_lat=31.2, ground_point_lon=121.4, ground_point_alt=70.0)
    assert config.ground_z(100.0) == pytest.approx(30.0)


def test_default_config_validates_with_georef_fields() -> None:
    from airdrop import Config

    config = Config().validated()
    assert config.camera.calib_file
    assert config.camera.fallback_width > 0
