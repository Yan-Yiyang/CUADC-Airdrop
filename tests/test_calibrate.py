"""tools/calibrate.py 标定流水线的合成自洽验证（P6 验收口径之一）。

做法：用已知内参 / 已知 Δt / 已知 R_bc 生成合成棋盘格视图与机体姿态，
再让流水线去把它们解回来，逐项比对。

链路顺序（与真实采集一致）
--------------------------
相机（棋盘格在相机系下的位姿）由"机体在世界里的位姿 + 真值 X=R_bc/t_bc"导出，
因此流水线拿到的像素数据里天然含有真值外参——解不回来说明算法或约定错了。
"""

from __future__ import annotations

import json
import math
import shutil
import uuid
from collections.abc import Iterator, Sequence
from pathlib import Path

import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from airdrop.georef import rotation_x, rotation_y, rotation_z
from airdrop.telemetry.models import TelemetrySnapshot
from tools import calibrate as calibrate_module
from tools.calibrate import (
    MEASURED_T_BC,
    ExtrinsicsResult,
    align_body_poses,
    angular_speed_from_poses,
    board_object_points,
    calibrate,
    detect_corners,
    estimate_lag_by_correlation,
    lever_arm_report,
    lever_arm_tolerance,
    look_at_board,
    pattern_phase_ok,
    quaternion_to_matrix_local,
    render_board,
    rotation_angle_deg,
    rotation_log,
    run_extrinsics,
    run_intrinsics,
    solve_board_poses,
    solve_hand_eye,
    telemetry_pose_at,
)

# 真值
TRUE_FX, TRUE_FY = 900.0, 902.0
TRUE_CX, TRUE_CY = 640.0, 360.0
TRUE_SIZE = (1280, 720)
TRUE_DIST = np.array([-0.12, 0.03, 0.0005, -0.0007, 0.0])
PATTERN = (7, 5)
SQUARE = 0.03


def true_camera_matrix() -> np.ndarray:
    return np.array([[TRUE_FX, 0.0, TRUE_CX], [0.0, TRUE_FY, TRUE_CY], [0.0, 0.0, 1.0]])


def true_r_bc() -> np.ndarray:
    """相机→机体：绕机体 z +90°，再叠一点真实的安装倾斜。"""
    return rotation_z(90.0) @ rotation_y(-3.0) @ rotation_x(2.0)


TRUE_T_BC = np.array([0.06, -0.03, 0.09])


def _board_pose_in_camera(
    body_rotation: np.ndarray,
    body_position: np.ndarray,
    r_bc: np.ndarray,
    t_bc: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """给定机体位姿与真值外参，返回棋盘格在相机系下的 (rvec, tvec)。

    棋盘格固定在世界原点（板面 z=0），机体位姿在世界系里给。
    ``T_cam_board = X^{-1} · T_body_world``，其中 ``X = [R_bc | t_bc]``。
    """
    body_world = np.eye(4)
    body_world[:3, :3] = body_rotation
    body_world[:3, 3] = body_position
    x = np.eye(4)
    x[:3, :3] = r_bc
    x[:3, 3] = t_bc
    camera = np.linalg.inv(x) @ body_world
    rvec, _ = cv2.Rodrigues(camera[:3, :3])
    return rvec, camera[:3, 3]


def _render_views(count: int, distance: float = 0.55) -> tuple[list[np.ndarray], list]:
    """渲染一批姿态多样的棋盘格视图（用真值内参/畸变），返回 (帧, 相机位姿)。"""
    frames: list[np.ndarray] = []
    poses: list[tuple[np.ndarray, np.ndarray]] = []
    for index in range(count):
        rvec, tvec = look_at_board(
            distance,
            true_camera_matrix(),
            tilt_deg=14.0 * math.sin(index * 0.9),
            yaw_deg=11.0 * math.cos(index * 0.6),
            rim_roll_deg=9.0 * math.sin(index * 1.7),
            pattern_size=PATTERN,
            square_size=SQUARE,
        )
        frames.append(
            render_board(
                rvec,
                tvec,
                true_camera_matrix(),
                TRUE_DIST,
                TRUE_SIZE,
                pattern_size=PATTERN,
                square_size=SQUARE,
            )
        )
        poses.append((rvec, tvec))
    return frames, poses


def _sample_body_poses(count: int, seed: int = 3) -> list[tuple[np.ndarray, np.ndarray]]:
    """生成有足够姿态多样性的机体位姿（棋盘格固定在世界原点）。

    机体位置要让棋盘格中心落在像面中心附近并占据足够像素：``X`` 的杆臂很小，
    所以机体的 ``(north, east)`` 基本就是"相机相对棋盘格的横向偏移"，
    把它设成 0 才让板子居中（板子中心=棋盘原点）。
    """
    rng = np.random.default_rng(seed)
    distance = 0.28
    poses: list[tuple[np.ndarray, np.ndarray]] = []
    for index in range(count):
        roll = 12.0 * math.sin(index * 0.7)
        pitch = 10.0 * math.cos(index * 0.5)
        yaw = (index * 37.0) % 360.0
        rotation = rotation_z(yaw) @ rotation_y(pitch) @ rotation_x(roll)
        # 横向偏移很小（保持板子在画面内），高度在标称距离附近抖动
        position = np.array(
            [
                rng.normal(scale=0.008),
                rng.normal(scale=0.008),
                -distance + rng.normal(scale=0.01),
            ]
        )
        poses.append((rotation, position))
    return poses


# ----------------------------------------------------------------------
# 步骤一：内参
# ----------------------------------------------------------------------
def test_intrinsics_recovered_from_synthetic_views() -> None:
    """合成多视图 → 内参应当能解回真值（相对误差 < 2%）。"""
    frames, _ = _render_views(20)
    result = run_intrinsics(frames, pattern_size=PATTERN, square_size=SQUARE, min_views=5)
    assert result.views >= 5, f"只检出 {result.views} 个视图"
    assert result.rms < 1.0, f"重投影 RMS 过大: {result.rms}"
    assert result.camera_matrix[0, 0] == pytest.approx(TRUE_FX, rel=0.02)
    assert result.camera_matrix[1, 1] == pytest.approx(TRUE_FY, rel=0.02)
    assert result.camera_matrix[0, 2] == pytest.approx(TRUE_CX, abs=12.0)
    assert result.camera_matrix[1, 2] == pytest.approx(TRUE_CY, abs=12.0)


def test_intrinsics_needs_enough_views() -> None:
    blank = [np.full((720, 1280, 3), 128, np.uint8) for _ in range(4)]
    with pytest.raises(ValueError, match="有效视图不足"):
        run_intrinsics(blank, pattern_size=PATTERN, square_size=SQUARE, min_views=5)


def test_detect_corners_finds_rendered_board() -> None:
    frames, _ = _render_views(1)
    corners = detect_corners(frames[0], PATTERN)
    assert corners is not None, "渲染的棋盘格应当能被检出"
    assert corners.shape == (PATTERN[0] * PATTERN[1], 1, 2)


def test_rendered_board_projects_inside_frame() -> None:
    """板子四角必须投影在画幅内——被裁掉一半的板子检不出角点。"""
    frames, poses = _render_views(8)
    board = board_object_points(PATTERN, SQUARE)
    for rvec, tvec in poses:
        projected, _ = cv2.projectPoints(board, rvec, tvec, true_camera_matrix(), TRUE_DIST)
        points = projected.reshape(-1, 2)
        assert points[:, 0].min() > 0 and points[:, 0].max() < TRUE_SIZE[0]
        assert points[:, 1].min() > 0 and points[:, 1].max() < TRUE_SIZE[1]


def test_look_at_board_centers_the_board() -> None:
    """板子几何中心应当投影到主点附近（对准了才不会被裁）。"""
    cols, rows = PATTERN
    center = np.array([[[(cols - 1) * SQUARE / 2, (rows - 1) * SQUARE / 2, 0.0]]], dtype=np.float64)
    rvec, tvec = look_at_board(0.55, true_camera_matrix(), pattern_size=PATTERN, square_size=SQUARE)
    projected, _ = cv2.projectPoints(center, rvec, tvec, true_camera_matrix(), np.zeros(5))
    assert float(projected[0, 0, 0]) == pytest.approx(TRUE_CX, abs=1.0)
    assert float(projected[0, 0, 1]) == pytest.approx(TRUE_CY, abs=1.0)


def test_detect_corners_returns_none_on_blank() -> None:
    assert detect_corners(np.full((480, 640, 3), 127, np.uint8), PATTERN) is None


# ----------------------------------------------------------------------
# 步骤二：时间差
# ----------------------------------------------------------------------
def test_lag_recovered_from_shifted_series() -> None:
    """把机体角速度序列整体平移已知量，互相关应当把它解回来。"""
    step = 0.01
    times = np.arange(0.0, 3.0, step)
    # 一段有明显起伏的角速度（避免常数序列相关为 0）
    body_speed = (
        20.0 + 15.0 * np.sin(2 * math.pi * 1.5 * times) + 5.0 * np.sin(2 * math.pi * 5.0 * times)
    )
    true_lag = 0.12
    camera_times = times + true_lag  # 画面比姿态晚 true_lag
    camera_speed = body_speed.copy()
    result = estimate_lag_by_correlation(
        camera_times,
        camera_speed,
        times,
        body_speed,
        search_s=0.5,
        step_s=0.005,
    )
    assert result.ok
    assert result.lag_s == pytest.approx(true_lag, abs=0.02), f"解出 {result.lag_s}"
    assert result.peak > 0.9


@pytest.mark.parametrize("true_lag", [0.0, -0.15, 0.25])
def test_lag_recovers_various_offsets(true_lag: float) -> None:
    step = 0.01
    times = np.arange(0.0, 3.0, step)
    body_speed = 25.0 + 18.0 * np.sin(2 * math.pi * 1.2 * times)
    camera_times = times + true_lag
    result = estimate_lag_by_correlation(
        camera_times,
        body_speed.copy(),
        times,
        body_speed,
        search_s=0.5,
        step_s=0.005,
    )
    assert result.lag_s == pytest.approx(true_lag, abs=0.02)


def test_lag_reports_failure_on_flat_series() -> None:
    times = np.arange(0.0, 2.0, 0.01)
    flat = np.full_like(times, 30.0)
    result = estimate_lag_by_correlation(times, flat, times, flat)
    assert not result.ok


def test_lag_reports_failure_on_too_few_samples() -> None:
    result = estimate_lag_by_correlation(
        np.array([0.0, 0.1]),
        np.array([1.0, 2.0]),
        np.array([0.0, 0.1]),
        np.array([1.0, 2.0]),
    )
    assert not result.ok


def test_angular_speed_from_poses_matches_constant_rate() -> None:
    """匀速旋转 30°/s：算出的角速度应当处处是 30°/s。"""
    rate = 30.0
    step = 0.02
    times = np.arange(0.0, 1.0, step)
    rotations = [rotation_z(rate * t) for t in times]
    out_times, speeds = angular_speed_from_poses(times, rotations)  # pyright: ignore[reportArgumentType]
    assert out_times.size == times.size - 1
    assert np.allclose(speeds, rate, atol=1e-6)


def test_rotation_angle_and_log_round_trip() -> None:
    for axis in (
        np.array([1.0, 0, 0]),
        np.array([0, 1.0, 0]),
        np.array([0, 0, 1.0]),
        np.array([1.0, 2.0, -0.5]) / np.linalg.norm([1.0, 2.0, -0.5]),
    ):
        for angle_deg in (0.1, 5.0, 45.0, 179.0):
            vector = axis * math.radians(angle_deg)
            matrix = _axis_angle_to_matrix(vector)
            assert rotation_angle_deg(matrix) == pytest.approx(angle_deg, abs=1e-6)
            assert np.allclose(rotation_log(matrix), vector, atol=1e-6)


def _axis_angle_to_matrix(vector: np.ndarray) -> np.ndarray:
    theta = float(np.linalg.norm(vector))
    if theta < 1e-12:
        return np.eye(3)
    axis = vector / theta
    k = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    return np.eye(3) + math.sin(theta) * k + (1 - math.cos(theta)) * (k @ k)


# ----------------------------------------------------------------------
# 步骤三：手眼（纯算法）
# ----------------------------------------------------------------------
def true_x_matrix() -> np.ndarray:
    """真值 ``X = T_body_cam = [R_bc | t_bc]``（相机→机体，与 georef 同语义）。"""
    matrix = np.eye(4)
    matrix[:3, :3] = true_r_bc()
    matrix[:3, 3] = TRUE_T_BC
    return matrix


def pose_of(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:3, :3] = cv2.Rodrigues(np.asarray(rvec, float).reshape(3, 1))[0]
    matrix[:3, 3] = np.asarray(tvec, float).reshape(3)
    return matrix


def body_pose_from_board(board_pose: np.ndarray, x_true: np.ndarray) -> np.ndarray:
    """由"棋盘格在相机系"的位姿导出"机体在世界系"的位姿。

    棋盘格固定在世界原点（``C = T_world_board = I``），于是
    ``G X P = C`` ⇒ ``G = P⁻¹ X⁻¹``。

    ⚠ 这里只有一个正确方向。写成 ``G = X·P``（等价于把"棋盘格在机体系"当成机体位姿）
    就等于是拿未知量当已知量用：相对运动里那个常量会被整体消掉，
    方程退化成"求与所有 R 可交换的矩阵"，真值根本不在解空间里。
    回归用例 ``test_body_pose_must_come_from_telemetry`` 把它钉死。
    """
    return np.linalg.inv(board_pose) @ np.linalg.inv(x_true)


def _motion_pairs(
    bodies: Sequence[np.ndarray], cameras: Sequence[np.ndarray]
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """``A_ij = G_j⁻¹G_i``（遥测）、``B_ij = P_jP_i⁻¹``（PnP）——与 run_extrinsics 一致。"""
    a_list: list[np.ndarray] = []
    b_list: list[np.ndarray] = []
    for i in range(len(bodies)):
        for j in range(i + 1, len(bodies)):
            a_list.append(np.linalg.inv(bodies[j]) @ bodies[i])
            b_list.append(cameras[j] @ np.linalg.inv(cameras[i]))
    return a_list, b_list


def _board_views(count: int = 24, distance: float = 0.55):
    """渲染一批"相机对着板子"的视图，返回 (帧, 棋盘格→相机 位姿)。"""
    frames: list[np.ndarray] = []
    poses: list[np.ndarray] = []
    for index in range(count):
        rvec, tvec = look_at_board(
            distance,
            true_camera_matrix(),
            tilt_deg=16.0 * math.sin(index * 0.9),
            yaw_deg=13.0 * math.cos(index * 0.6),
            rim_roll_deg=10.0 * math.sin(index * 1.7),
            pattern_size=PATTERN,
            square_size=SQUARE,
        )
        frames.append(
            render_board(
                rvec,
                tvec,
                true_camera_matrix(),
                TRUE_DIST,
                TRUE_SIZE,
                pattern_size=PATTERN,
                square_size=SQUARE,
            )
        )
        poses.append(pose_of(rvec, tvec))
    return frames, poses


def _random_rotations(count: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    rotations = []
    for _ in range(count):
        vector = rng.normal(size=3)
        vector = vector / np.linalg.norm(vector) * math.radians(rng.uniform(-60, 60))
        rotations.append(_axis_angle_to_matrix(vector))
    return rotations


# 多组种子：解法的"零空间符号"曾出过 bug（真值在零空间里，却解出精确 180° 的那个版本），
# 单组数据有 50% 概率蒙对，所以这里跑多组。
@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_solver_recovers_known_transform(seed: int) -> None:
    """``A`` 来遥测、``B`` 来 PnP 时，``AX=XB`` 应当解到机器精度。"""
    x_true = true_x_matrix()
    rng = np.random.default_rng(seed)
    bodies = []
    for rotation in _random_rotations(12, seed):
        matrix = np.eye(4)
        matrix[:3, :3] = rotation
        matrix[:3, 3] = rng.normal(scale=0.3, size=3)  # 机体位置是独立量
        bodies.append(matrix)
    cameras = [np.linalg.inv(x_true) @ np.linalg.inv(g) for g in bodies]
    a_list, b_list = _motion_pairs(bodies, cameras)

    rotation, translation, smallest, spread = solve_hand_eye(a_list, b_list)
    assert smallest < 1e-9, f"激励充足时最小奇异值应接近 0，实际 {smallest}"
    assert spread > 15.0
    assert translation is not None
    assert np.allclose(rotation, true_r_bc(), atol=1e-9), np.abs(rotation - true_r_bc()).max()
    assert np.allclose(translation, TRUE_T_BC, atol=1e-9), np.abs(translation - TRUE_T_BC).max()


def test_body_pose_must_come_from_telemetry() -> None:
    """反例固化：机体位姿若由棋盘格位姿左乘常量导出，就解不出真值。

    这不是"板子没转"（板子固定、飞机在动时姿态当然在变），而是把未知量当已知量用。
    """
    x_true = true_x_matrix()
    rng = np.random.default_rng(7)
    cameras = []
    for index, rotation in enumerate(_random_rotations(10, 7)):
        matrix = np.eye(4)
        matrix[:3, :3] = rotation
        matrix[:3, 3] = rng.normal(scale=0.3, size=3)
        cameras.append(matrix)
    bodies = [x_true @ p for p in cameras]  # 错误方向：G = X·P
    a_list, b_list = _motion_pairs(bodies, cameras)
    rotation, *_ = solve_hand_eye(a_list, b_list)
    error = rotation_angle_deg(rotation.T @ true_r_bc())
    assert error > 10.0, f"错误构造居然解出了真值（误差 {error:.3f}°），用例失效"


def test_solver_requires_pairs() -> None:
    with pytest.raises(ValueError, match="运动对数量不足"):
        solve_hand_eye([np.eye(4)], [np.eye(4)])


def test_extrinsics_rejects_single_rotation_axis() -> None:
    """所有运动对绕同一根轴时解不唯一（OpenCV 文档同样要求 ≥2 个不平行轴）。"""
    x_true = true_x_matrix()
    bodies = []
    for index in range(12):
        matrix = np.eye(4)
        matrix[:3, :3] = rotation_z(7.0 * index)  # 只绕 z 转
        matrix[:3, 3] = np.array([0.01 * index, 0.0, -0.45])
        bodies.append(matrix)
    cameras = [np.linalg.inv(x_true) @ np.linalg.inv(g) for g in bodies]
    board_poses = [
        (float(index), cv2.Rodrigues(p[:3, :3])[0], p[:3, 3]) for index, p in enumerate(cameras)
    ]
    with pytest.raises(ValueError, match="旋转轴过于单一"):
        run_extrinsics(board_poses=board_poses, body_poses=bodies)


def _bodies_with_real_translation(seed: int, count: int = 14) -> list[np.ndarray]:
    """一批位置真的在动的遥测位姿（室内无 GPS 时取不到的就是这个）。"""
    rng = np.random.default_rng(seed)
    bodies = []
    for rotation in _random_rotations(count, seed):
        matrix = np.eye(4)
        matrix[:3, :3] = rotation
        matrix[:3, 3] = rng.normal(scale=0.3, size=3)
        bodies.append(matrix)
    return bodies


def _board_poses_from_cameras(
    cameras: Sequence[np.ndarray],
) -> list[tuple[float, np.ndarray, np.ndarray]]:
    """PnP 位姿 → ``run_extrinsics`` 要的 ``(时刻, rvec, tvec)``。"""
    return [
        (float(index), cv2.Rodrigues(p[:3, :3])[0], p[:3, 3]) for index, p in enumerate(cameras)
    ]


def test_extrinsics_estimates_lever_arm_from_real_telemetry_positions() -> None:
    """遥测位置可信时杆臂估计值应当回到真值——这正是它拿来和尺量值比对的意义。"""
    x_true = true_x_matrix()
    bodies = _bodies_with_real_translation(5)
    cameras = [np.linalg.inv(x_true) @ np.linalg.inv(g) for g in bodies]
    result = run_extrinsics(
        board_poses=_board_poses_from_cameras(cameras),
        body_poses=bodies,
    )
    assert result.t_bc is not None
    assert np.allclose(result.t_bc, TRUE_T_BC, atol=1e-9), np.abs(result.t_bc - TRUE_T_BC).max()
    assert result.t_bc_reliable
    assert result.residual_translation_m < 1e-9


def test_extrinsics_flags_lever_arm_estimate_from_fake_telemetry_positions() -> None:
    """无 GPS 时遥测位置恒零：旋转照样解得对，杆臂却估不准，必须被标为不可信。

    照抄真实采集的失效模式：PnP 看到的相机位移是真的、遥测报的机体位置是假的，
    于是平移方程 ``(R_a−I)t_x = R_x t_b − t_a`` 本身不自洽。判据必须落在残差上——
    ``t_a`` 是真是假都解得出一个 ``t_x``，光看数字大小分辨不出来。
    """
    x_true = true_x_matrix()
    bodies = _bodies_with_real_translation(3)
    cameras = [np.linalg.inv(x_true) @ np.linalg.inv(g) for g in bodies]
    lying = []
    for pose in bodies:
        copy = pose.copy()
        copy[:3, 3] = 0.0  # 位置恒零（室内无 GPS 的典型表现）
        lying.append(copy)

    result = run_extrinsics(
        board_poses=_board_poses_from_cameras(cameras),
        body_poses=lying,
    )
    # 1) 旋转不受影响：A 的旋转块与位置无关，解照样精确（1e-3 度是纯数值噪声量级）
    assert rotation_angle_deg(result.r_bc.T @ true_r_bc()) < 1e-3
    # 2) 平移估计偏了好几厘米，残差远超门限
    assert result.t_bc is not None
    assert np.linalg.norm(result.t_bc - TRUE_T_BC) > 0.05, result.t_bc
    assert result.residual_translation_m > 0.5
    assert not result.t_bc_reliable
    # 3) 诊断量直接说明原因：遥测说机体没动，画面说相机动了米级
    assert result.body_span_m == 0.0
    assert result.camera_span_m > 1.0


def test_extrinsics_can_skip_lever_arm_estimate() -> None:
    """``estimate_body_translation=False`` 时不算平移；此时不能把 None 当成"可信"。"""
    x_true = true_x_matrix()
    bodies = _bodies_with_real_translation(5)
    cameras = [np.linalg.inv(x_true) @ np.linalg.inv(g) for g in bodies]
    result = run_extrinsics(
        board_poses=_board_poses_from_cameras(cameras),
        body_poses=bodies,
        estimate_body_translation=False,
    )
    assert result.t_bc is None
    assert not result.t_bc_reliable
    assert result.r_bc is not None


def test_lever_arm_report_pairs_estimate_with_measured_value() -> None:
    """报告里尺量值是 authoritative，估计值只用于比对（逐轴差 + 欧氏距离）。"""
    result = ExtrinsicsResult(
        r_bc=true_r_bc(),
        t_bc=np.array([0.07, -0.02, 0.08]),
        pairs=12,
        residual_rotation_deg=0.1,
        t_bc_reliable=True,
        residual_translation_m=0.003,
        body_span_m=1.2,
        camera_span_m=1.3,
    )
    report = lever_arm_report(result, (0.06, -0.03, 0.09))
    assert report["authoritative"] == [0.06, -0.03, 0.09]
    assert report["source"] == "measured"
    assert report["estimated"] == pytest.approx([0.07, -0.02, 0.08])
    assert report["delta_xyz_m"] == pytest.approx([0.01, 0.01, -0.01])
    assert report["delta_m"] == pytest.approx(math.sqrt(3e-4))
    assert report["estimated_reliable"] is True
    assert report["residual_tolerance_m"] == pytest.approx(lever_arm_tolerance(1.3))


def test_lever_arm_report_keeps_measured_value_without_estimate() -> None:
    """解不出外参或没估平移时，报告仍要给尺量值，估计值一律 None（不是 0）。"""
    empty = lever_arm_report(None)
    assert empty["authoritative"] == list(MEASURED_T_BC)
    assert empty["estimated"] is None
    assert empty["delta_m"] is None
    assert empty["estimated_reliable"] is False

    skipped = ExtrinsicsResult(r_bc=true_r_bc(), t_bc=None, pairs=9, residual_rotation_deg=0.2)
    assert lever_arm_report(skipped)["estimated"] is None


def test_extrinsics_requires_one_body_pose_per_board_pose() -> None:
    """遥测 10 Hz、画面 30 Hz，不能"第 i 条遥测配第 i 帧"——数量对不上必须报错。"""
    x_true = true_x_matrix()
    frames, poses = _board_views(6)
    bodies = [body_pose_from_board(p, x_true) for p in poses]
    board_poses = [
        (float(index), cv2.Rodrigues(p[:3, :3])[0], p[:3, 3]) for index, p in enumerate(poses)
    ]
    with pytest.raises(ValueError, match="数量不一致"):
        run_extrinsics(board_poses=board_poses, body_poses=bodies[:3])


# ----------------------------------------------------------------------
# 步骤三：遥测位姿的重采样与配对
# ----------------------------------------------------------------------
def _yaw_snapshot(stamp: float, yaw_deg: float, down: float = -0.45) -> TelemetrySnapshot:
    half = math.radians(yaw_deg) / 2.0
    return TelemetrySnapshot(
        timestamp=stamp,
        north_m=0.0,
        east_m=0.0,
        down_m=down,
        quaternion_w=math.cos(half),
        quaternion_x=0.0,
        quaternion_y=0.0,
        quaternion_z=math.sin(half),
    )


def test_telemetry_pose_at_interpolates_and_refuses_extrapolation() -> None:
    snapshots = [_yaw_snapshot(index * 0.1, 10.0 * index) for index in range(5)]
    pose = telemetry_pose_at(snapshots, 0.05)
    assert pose is not None
    assert rotation_angle_deg(pose[:3, :3].T @ rotation_z(5.0)) < 1e-9
    assert telemetry_pose_at(snapshots, -0.1) is None  # 不外推
    assert telemetry_pose_at(snapshots, 10.0) is None


def test_align_body_poses_applies_lag_with_the_right_sign() -> None:
    """``lag > 0`` 表示画面滞后 ⇒ 该帧查的是更早的遥测。

    造一段匀速偏航的遥测，让棋盘格位姿对应 ``capture_timestamp − lag`` 时刻的机体姿态；
    带上 lag 配对应当精确复原，不带就有 ``lag·ω`` 的固定偏差。
    """
    rate, step, lag = 40.0, 0.02, 0.10  # °/s, s, s
    snapshots = [_yaw_snapshot(index * step, rate * index * step) for index in range(60)]
    x_true = true_x_matrix()
    board_poses = []
    for index in range(5, 50):
        stamp = index * step
        body = np.eye(4)
        body[:3, :3] = rotation_z(rate * (stamp - lag))  # 真实拍摄时刻的机体姿态
        body[:3, 3] = np.array([0.0, 0.0, -0.45])
        camera = np.linalg.inv(x_true) @ np.linalg.inv(body)
        board_poses.append((stamp, cv2.Rodrigues(camera[:3, :3])[0], camera[:3, 3]))

    paired_board, paired_body = align_body_poses(board_poses, snapshots, lag_s=lag)
    assert len(paired_board) == len(paired_body) > 20
    expected = rotation_z(rate * (paired_board[0][0] - lag))
    error = rotation_angle_deg(paired_body[0][:3, :3].T @ expected)
    assert error < 0.05, f"扣掉 lag 后应当复原，实际差 {error:.3f}°"

    _, wrong_body = align_body_poses(board_poses, snapshots, lag_s=0.0)
    drift = rotation_angle_deg(wrong_body[0][:3, :3].T @ expected)
    assert drift > 3.0, f"不扣 lag 应当有 {rate * lag:.1f}° 的偏差，实际 {drift:.3f}°"


# ----------------------------------------------------------------------
# 渲染链路的自洽性（棋盘格相位）
# ----------------------------------------------------------------------
def test_rendered_board_matches_requested_pose() -> None:
    """渲染出来的棋盘格必须与给定位姿同相位。

    回归用例：to_board 少减一个 tile 时，棋盘格相对对象点整体错开一整格，
    检测出的角点落在"板子真实点 +1 格"处——重投影误差仍然只有 0.1 px，
    但位姿差 180°，手眼标定直接报废，靠重投影根本发现不了。只有方格相位能发现
    （实测错格时相符率 < 0.5，正确时 1.000）。
    """
    board = board_object_points(PATTERN, SQUARE)
    for index in (0, 3, 7, 11):
        rvec, tvec = look_at_board(
            0.55,
            true_camera_matrix(),
            tilt_deg=16.0 * math.sin(index * 0.9),
            yaw_deg=13.0 * math.cos(index * 0.6),
            rim_roll_deg=10.0 * math.sin(index * 1.7),
            pattern_size=PATTERN,
            square_size=SQUARE,
        )
        frame = render_board(
            rvec,
            tvec,
            true_camera_matrix(),
            TRUE_DIST,
            TRUE_SIZE,
            pattern_size=PATTERN,
            square_size=SQUARE,
        )
        ratio = pattern_phase_ok(
            frame,
            rvec,
            tvec,
            true_camera_matrix(),
            TRUE_DIST,
            pattern_size=PATTERN,
            square_size=SQUARE,
        )
        assert ratio > 0.9, f"第 {index} 帧相位相符率只有 {ratio:.2f}：棋盘格与对象点错格了"

        pose = solve_board_poses(
            [frame],
            board,
            true_camera_matrix(),
            TRUE_DIST,
            pattern_size=PATTERN,
            square_size=SQUARE,
        )[0]
        assert pose is not None, f"第 {index} 帧应当能解出位姿"
        corners = detect_corners(frame, PATTERN)
        assert corners is not None
        projected, _ = cv2.projectPoints(board, pose[0], pose[1], true_camera_matrix(), TRUE_DIST)
        # cv2 的类型桩把 projectPoints 的结果标成可能为 None，运行时它总是 ndarray；
        # 这里显式收窄，顺带让后面那行减法/平方有明确类型。
        assert projected is not None
        residual = (projected - corners) ** 2
        rms = float(np.sqrt(np.mean(np.sum(residual, axis=2))))
        assert rms < 0.5, f"第 {index} 帧解出的位姿重投影残差 {rms:.3f} px"


def test_pattern_phase_is_blind_to_label_ambiguity() -> None:
    """记录一条边界：相位分不出正序/反序标签——两支的相位相符率都是 1.000。

    棋盘格绕板面法向转 180° 后图像一模一样，单帧图像里没有信息能定下"哪个角点是 0 号"。
    这正是 :func:`solve_board_poses` 必须靠相邻帧连续性定支的原因。
    """
    frames, camera_poses = _board_views(8)
    frame = frames[3]
    rvec = cv2.Rodrigues(camera_poses[3][:3, :3])[0]
    tvec = camera_poses[3][:3, 3]
    board = board_object_points(PATTERN, SQUARE)
    corners = detect_corners(frame, PATTERN)
    assert corners is not None
    good = pattern_phase_ok(
        frame,
        rvec,
        tvec,
        true_camera_matrix(),
        TRUE_DIST,
        pattern_size=PATTERN,
        square_size=SQUARE,
    )
    # 另一个标签支：把检测到的角点顺序整体反过来的那个解
    ok, other_rvec, other_tvec = cv2.solvePnP(
        board,
        corners[::-1],
        true_camera_matrix(),
        TRUE_DIST,
        flags=cv2.SOLVEPNP_IPPE,
    )
    assert ok
    assert other_rvec is not None and other_tvec is not None
    assert (
        rotation_angle_deg(pose_of(other_rvec, other_tvec)[:3, :3].T @ pose_of(rvec, tvec)[:3, :3])
        > 170.0
    ), "反序解应当与正序解差 180°"
    flipped = pattern_phase_ok(
        frame,
        other_rvec,
        other_tvec,
        true_camera_matrix(),
        TRUE_DIST,
        pattern_size=PATTERN,
        square_size=SQUARE,
    )
    assert good > 0.9, f"正确位姿的相位相符率只有 {good:.2f}"
    assert flipped > 0.9, f"另一支的相位本该也是 1.0，实际 {flipped:.2f}"


# ----------------------------------------------------------------------
# 三步串起来：渲染 → 检测（含相位消解）→ 配对 → 解外参
# ----------------------------------------------------------------------
def test_full_pipeline_recovers_extrinsics_from_synthetic_views() -> None:
    """端到端：像素数据里含有真值外参，整条链路要能把它解回来。

    链路：渲染棋盘格 → 检测 + 相位消解标签 → PnP 得 ``P`` → 由 ``G = P⁻¹X⁻¹`` 得
    机体位姿（模拟"板子固定、飞机在动"的采集）→ ``AX=XB`` 解 ``X``。
    """
    x_true = true_x_matrix()
    board = board_object_points(PATTERN, SQUARE)
    frames, camera_poses = _board_views(24)

    intrinsics = run_intrinsics(frames, pattern_size=PATTERN, square_size=SQUARE, min_views=8)
    assert intrinsics.rms < 1.0
    assert intrinsics.camera_matrix[0, 0] == pytest.approx(TRUE_FX, rel=0.02)

    solved_raw = solve_board_poses(
        frames,
        board,
        intrinsics.camera_matrix,
        intrinsics.dist_coeffs,
        pattern_size=PATTERN,
        square_size=SQUARE,
    )
    solved: list[tuple[float, np.ndarray, np.ndarray]] = []
    branches: set[bool] = set()
    for index, pose in enumerate(solved_raw):
        assert pose is not None, f"第 {index} 帧应当能解出位姿"
        rotation = cv2.Rodrigues(pose[0])[0]
        truth = camera_poses[index][:3, :3]
        forward = rotation_angle_deg(rotation.T @ truth)
        mirrored = rotation_angle_deg(rotation.T @ (truth @ rotation_z(180.0)))
        assert min(forward, mirrored) < 1.0, (
            f"第 {index} 帧既不是正序也不是反序解: {forward:.3f}° / {mirrored:.3f}°"
        )
        branches.add(forward < mirrored)
        solved.append((float(index), pose[0], pose[1]))
    assert len(branches) == 1, "标签支必须全程一致——混用会让 AX=XB 崩掉"

    bodies = [body_pose_from_board(p, x_true) for p in camera_poses]
    extrinsics = run_extrinsics(board_poses=solved, body_poses=bodies)
    assert extrinsics.ok
    error = rotation_angle_deg(extrinsics.r_bc.T @ true_r_bc())
    assert error < 3.0, f"外参旋转误差 {error:.3f}° 过大"
    assert extrinsics.axis_spread_deg > 15.0
    assert extrinsics.pairs >= 10
    # 杆臂估计值现在也解出来（进标定文件的仍是尺量值，见 lever_arm_report）：
    # 合成链路里 A 由 P 导出、遥测位置天然"可信"，所以估计值应当接近真值（实测差 ~7mm）
    assert extrinsics.t_bc is not None
    assert np.allclose(extrinsics.t_bc, TRUE_T_BC, atol=0.03), extrinsics.t_bc
    assert extrinsics.t_bc_reliable


def test_full_pipeline_rejects_insufficient_rotation_excitation() -> None:
    """旋转激励不足时必须明确报错，而不是给一个看起来"成功"的解。"""
    x_true = true_x_matrix()
    frames, camera_poses = _board_views(8)
    bodies = []
    for index in range(8):
        # 姿态几乎不变（< MIN_PAIR_ROTATION_DEG），只有位置在动
        matrix = np.eye(4)
        matrix[:3, :3] = rotation_z(0.4 * index)
        matrix[:3, 3] = np.array([0.01 * index, 0.0, -0.45])
        bodies.append(matrix)
    camera_poses = [np.linalg.inv(x_true) @ np.linalg.inv(g) for g in bodies]
    board_poses = [
        (float(index), cv2.Rodrigues(p[:3, :3])[0], p[:3, 3])
        for index, p in enumerate(camera_poses)
    ]
    with pytest.raises(ValueError, match="旋转激励不够"):
        run_extrinsics(board_poses=board_poses, body_poses=bodies)


# ----------------------------------------------------------------------
# 标定 JSON 的输出契约（杆臂：尺量为准 + 估计值比对）
# ----------------------------------------------------------------------
CALIB_TMP_ROOT = Path(__file__).resolve().parents[1] / ".calibrate-test-tmp"


@pytest.fixture
def workdir() -> Iterator[Path]:
    """工作区内的临时目录；用例结束整棵删掉。

    临时目录位于工作区内，避免依赖 pytest 的 basetemp 目录。
    """
    CALIB_TMP_ROOT.mkdir(parents=True, exist_ok=True)
    path = CALIB_TMP_ROOT / uuid.uuid4().hex[:8]
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _write_calibration_flight(flight_dir: Path, count: int = 30) -> None:
    """手写一个可标定的飞行目录（渲染棋盘格帧 + 与之自洽的遥测）。

    相机位姿由 :func:`look_at_board` 给出（保证板子全程在视野内），机体位姿取
    ``G = P⁻¹X⁻¹``（棋盘格固定在世界原点，``C = I``）——与合成用例同一套约定，
    遥测里因此天然含真值外参，解不回来就是流水线错了。

    遥测在首末各多写一条把帧区间包住：``telemetry_pose_at`` 只做范围内内插、
    拒绝外推，不包住的话首末帧会整帧被丢。
    """
    (flight_dir / "frames").mkdir(parents=True)
    t0 = 1_000_000.0
    frame_dt = 1.0 / 30.0
    lag = 0.15  # 与 VideoConfig.telemetry_lag 默认值一致
    x_true = true_x_matrix()
    index_lines: list[str] = []
    entries: list[tuple[float, np.ndarray]] = []

    for index in range(count):
        rvec, tvec = look_at_board(
            0.55,
            true_camera_matrix(),
            tilt_deg=15.0 * math.sin(index * 0.9),
            yaw_deg=12.0 * math.cos(index * 0.6),
            rim_roll_deg=10.0 * math.sin(index * 1.7),
            pattern_size=PATTERN,
            square_size=SQUARE,
        )
        image = render_board(
            rvec,
            tvec,
            true_camera_matrix(),
            TRUE_DIST,
            TRUE_SIZE,
            pattern_size=PATTERN,
            square_size=SQUARE,
        )
        name = f"{index:06d}.jpg"
        ok, buffer = cv2.imencode(".jpg", image)
        assert ok
        payload = buffer.tobytes()
        (flight_dir / "frames" / name).write_bytes(payload)
        capture = t0 + index * frame_dt
        index_lines.append(
            json.dumps(
                {
                    "index": index,
                    "filename": name,
                    "capture_timestamp": capture,
                    "received_timestamp": capture + lag,  # 与 VideoFrame.timestamp 同义
                    "lag": lag,
                    "extrapolated": False,
                    "offset": 0.0,
                    "bytes": len(payload),
                }
            )
        )
        entries.append((capture, body_pose_from_board(pose_of(rvec, tvec), x_true)))

    (flight_dir / "frames_index.jsonl").write_text("\n".join(index_lines) + "\n", encoding="utf-8")

    timeline = [(entries[0][0] - 0.3, entries[0][1])]
    timeline += entries
    timeline.append((entries[-1][0] + 0.3, entries[-1][1]))

    lines = []
    for stamp, pose in timeline:
        quaternion = Rotation.from_matrix(pose[:3, :3]).as_quat()  # scipy 是 (x,y,z,w)
        w, x, y, z = (
            float(quaternion[3]),
            float(quaternion[0]),
            float(quaternion[1]),
            float(quaternion[2]),
        )
        # 钉住约定：写进遥测的 (w,x,y,z) 反解回来必须就是这个旋转矩阵
        assert np.allclose(quaternion_to_matrix_local((w, x, y, z)), pose[:3, :3], atol=1e-9)
        lines.append(
            json.dumps(
                TelemetrySnapshot(
                    timestamp=stamp,
                    north_m=float(pose[0, 3]),
                    east_m=float(pose[1, 3]),
                    down_m=float(pose[2, 3]),
                    quaternion_w=w,
                    quaternion_x=x,
                    quaternion_y=y,
                    quaternion_z=z,
                ).as_dict()
            )
        )
    (flight_dir / "telemetry.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_calibrate_payload_records_lever_arm_against_measured_value(
    workdir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """输出契约：``t_bc`` 恒为尺量值，标定估计值进 ``meta`` 供对比校验。

    这条同时盯住"改了文件顶部的 ``MEASURED_T_BC``，输出必须跟着变"——该常量曾是
    ``lever_arm_report`` 的默认参数（定义时绑定），那样改常量不会生效。
    """
    flight_dir = workdir / "flight"
    flight_dir.mkdir()
    _write_calibration_flight(flight_dir)
    measured = (0.05, -0.04, 0.08)
    monkeypatch.setattr(calibrate_module, "MEASURED_T_BC", measured)

    # 合成板子是 (7,5)/30mm，而 calibrate() 的默认值是实拍用的 (9,6)/25mm——显式传入
    payload = calibrate(flight_dir, pattern_size=PATTERN, square_size=SQUARE)

    # 1) 标定文件里的 t_bc 恒为尺量值（georef 读的就是这个字段）
    assert payload["t_bc"] == list(measured)
    assert payload["meta"]["extrinsics"]["t_bc_source"] == "measured"
    arm = payload["meta"]["extrinsics"]["lever_arm"]
    assert arm["authoritative"] == list(measured)
    assert arm["source"] == "measured"
    assert arm["estimated"] is not None
    # 2) 估计值与尺量值的差：逐轴差值 + 欧氏距离，两者必须自洽
    delta = np.asarray(arm["delta_xyz_m"])
    assert np.allclose(delta, np.asarray(arm["estimated"]) - np.asarray(measured))
    assert arm["delta_m"] == pytest.approx(float(np.linalg.norm(delta)))
    # 3) 合成链路里遥测位置是真的：估计值接近真值（也就接近尺量值）且被判为可信
    assert np.allclose(arm["estimated"], TRUE_T_BC, atol=0.03), arm["estimated"]
    assert arm["estimated_reliable"] is True
    assert arm["residual_translation_m"] < arm["residual_tolerance_m"]
    # 4) 旋转仍以手眼标定为准
    assert rotation_angle_deg(np.asarray(payload["R_bc"]).T @ true_r_bc()) < 5.0
