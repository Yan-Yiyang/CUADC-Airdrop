from __future__ import annotations

import numpy as np
import pytest

from airdrop.georef.camera import CameraModel
from airdrop.georef.project import ned_to_pixel, ned_to_pixels, pixel_to_ned


def _camera() -> CameraModel:
    return CameraModel(
        camera_matrix=np.array([[820.0, 0.0, 640.0], [0.0, 815.0, 360.0], [0.0, 0.0, 1.0]]),
        dist_coeffs=np.array([-0.08, 0.01, 0.001, -0.002, 0.0]),
        t_bc=np.array([0.25, -0.1, 0.35]),
        image_size=(1280, 720),
    )


@pytest.mark.parametrize("undistort", [True, False])
def test_pixel_ned_projection_round_trip_with_pose_and_lever_arm(undistort: bool) -> None:
    camera = _camera()
    pixel = (590.0, 410.0)
    position = (12.0, -8.0, -120.0)
    euler = (4.0, -6.0, 18.0)
    world = pixel_to_ned(
        pixel,
        camera=camera,
        ground_z=0.0,
        position_ned=position,
        euler_deg=euler,
        undistort=undistort,
    )
    assert world.ok and world.ned is not None

    projected = ned_to_pixel(
        world.ned,
        camera=camera,
        position_ned=position,
        image_size=(1280, 720),
        euler_deg=euler,
        distort=undistort,
    )
    assert projected.valid and projected.in_bounds
    assert projected.pixel == pytest.approx(pixel, abs=1e-6)


def test_out_of_frame_projection_retains_direction_and_rear_point_is_invalid() -> None:
    camera = _camera()
    position = (0.0, 0.0, -100.0)
    outside_ned = pixel_to_ned(
        (2000.0, 360.0),
        camera=camera,
        ground_z=0.0,
        position_ned=position,
        euler_deg=(0.0, 0.0, 0.0),
    )
    assert outside_ned.ok and outside_ned.ned is not None
    outside = ned_to_pixel(
        outside_ned.ned,
        camera=camera,
        position_ned=position,
        image_size=(1280, 720),
        euler_deg=(0.0, 0.0, 0.0),
    )
    assert outside.valid
    assert not outside.in_bounds
    assert outside.pixel is not None and outside.pixel[0] > 1280

    rear = ned_to_pixel(
        (0.25, -0.1, -119.65),
        camera=camera,
        position_ned=(0.0, 0.0, -100.0),
        image_size=(1280, 720),
        euler_deg=(0.0, 0.0, 0.0),
    )
    assert not rear.valid
    assert rear.pixel is None


def test_non_finite_world_coordinate_is_explicitly_invalid() -> None:
    result = ned_to_pixel(
        (float("nan"), 0.0, 0.0),
        camera=_camera(),
        position_ned=(0.0, 0.0, -100.0),
        image_size=(1280, 720),
        euler_deg=(0.0, 0.0, 0.0),
    )
    assert not result.valid
    assert result.pixel is None


def test_batch_projection_matches_single_projection_and_preserves_invalid_slots() -> None:
    camera = _camera()
    position = (12.0, -8.0, -120.0)
    euler = (4.0, -6.0, 18.0)
    source = pixel_to_ned(
        (590.0, 410.0),
        camera=camera,
        ground_z=0.0,
        position_ned=position,
        euler_deg=euler,
    )
    assert source.ok and source.ned is not None

    batch = ned_to_pixels(
        (source.ned, None, (float("nan"), 0.0, 0.0)),
        camera=camera,
        position_ned=position,
        image_size=(1280, 720),
        euler_deg=euler,
    )
    single = ned_to_pixel(
        source.ned,
        camera=camera,
        position_ned=position,
        image_size=(1280, 720),
        euler_deg=euler,
    )

    assert batch[0] == single
    assert len(batch) == 3
    assert not batch[1].valid and batch[1].reason == "观测无有效坐标"
    assert not batch[2].valid and batch[2].reason == "NED 坐标含非有限值"
