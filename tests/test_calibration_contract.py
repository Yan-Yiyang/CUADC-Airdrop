"""离线标定包的独立文件格式、导入边界与几何契约。"""

from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np
import pytest

pytestmark = pytest.mark.integration

pytest.importorskip("airdrop_calibration", reason="跨仓契约需要标定子模块")


def test_handwritten_flightrecorder_files_and_camera_model_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """独立手写 JSONL 可读取，来源 schema 可由生产相机模型加载。"""
    from airdrop_calibration import calibrate
    from airdrop_calibration.flight_log import (
        load_flight_frames_paired,
        load_flight_stamps,
        load_flight_telemetry,
    )
    from airdrop_calibration.models import (
        ExtrinsicsResult,
        IntrinsicsResult,
        LagMeasurement,
        LagResult,
    )
    from airdrop_calibration.report import build_calibration_report

    from airdrop.georef import load_camera_model
    from airdrop.georef.project import pixel_to_ned

    flight = tmp_path / "flight"
    frames = flight / "frames"
    frames.mkdir(parents=True)
    ok, encoded = cv2.imencode(".jpg", np.zeros((20, 32, 3), dtype=np.uint8))
    assert ok
    (frames / "000001.jpg").write_bytes(encoded.tobytes())
    index_rows = [
        {
            "index": 1,
            "filename": "000001.jpg",
            "capture_timestamp": 10.0,
            "received_timestamp": 10.2,
            "lag": 0.2,
        },
        {
            "index": 2,
            "filename": "000002.jpg",
            "capture_timestamp": 10.1,
            "received_timestamp": 10.3,
        },
    ]
    (flight / "frames_index.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in index_rows),
        encoding="utf-8",
    )
    telemetry_rows = [
        {"timestamp": 10.1, "north_m": 1.0, "east_m": 2.0, "down_m": 3.0},
        {"north_m": 4.0},
        {"timestamp": float("nan"), "north_m": 5.0},
        {"timestamp": 10.0, "north_m": 0.0, "east_m": 0.0, "down_m": 0.0},
    ]
    (flight / "telemetry.jsonl").write_text(
        "".join(json.dumps(row, allow_nan=True) + "\n" for row in telemetry_rows),
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path.parent)
    paired = load_flight_frames_paired(flight.resolve())
    stamps = load_flight_stamps(flight.resolve())
    telemetry = load_flight_telemetry(flight.resolve())
    assert len(paired) == 1 and paired[0][0].received_timestamp == pytest.approx(10.2)
    assert stamps[1].lag_s == 0.0
    assert [item.timestamp for item in telemetry] == [10.0, 10.1]
    assert telemetry[1].north_m == pytest.approx(1.0)

    lag = LagResult(lag_s=0.15, peak=0.99, samples=20)
    payload = build_calibration_report(
        intrinsics=IntrinsicsResult(
            camera_matrix=np.array([[800.0, 0.0, 16.0], [0.0, 800.0, 10.0], [0.0, 0.0, 1.0]]),
            dist_coeffs=np.zeros(5),
            image_size=(32, 20),
            rms=0.1,
            views=12,
        ),
        lag=lag,
        measurement=LagMeasurement(lag, None, 0.2, True),
        extrinsics=ExtrinsicsResult(
            r_bc=calibrate.rotation_z(90.0),
            t_bc=np.array([1.0, 2.0, 0.5]),
            pairs=8,
            residual_rotation_deg=0.01,
        ),
        flight_dir=flight.resolve(),
        pattern_size=(7, 5),
        square_size=0.03,
        lag_search_s=0.5,
        lag_step_s=0.005,
        measured_t_bc=(1.0, 2.0, 0.5),
        source="simulated_calibration",
    )
    assert calibrate.CalibrateConfig().source == "measured_calibration"
    assert payload["meta"]["schema_version"] == 1
    assert payload["meta"]["source"] == "simulated_calibration"
    assert payload["meta"]["coordinates"]["time_base"] == "received"
    assert payload["meta"]["coordinates"]["R_bc"] == "camera optical frame to body FRD frame"
    calibration_path = tmp_path / "camera_calib.json"
    calibration_path.write_text(json.dumps(payload), encoding="utf-8")
    camera = load_camera_model(calibration_path)
    assert camera.image_size == (32, 20)
    assert camera.telemetry_lag == pytest.approx(0.15)

    pose = (1.0, 2.0, -10.0)
    center = pixel_to_ned(
        (16.0, 10.0),
        camera=camera,
        ground_z=0.0,
        position_ned=pose,
        euler_deg=(0.0, 0.0, 0.0),
        undistort=False,
    )
    assert center.ok and center.ned == pytest.approx((2.0, 4.0, 0.0), abs=1e-9)

    off_axis = pixel_to_ned(
        (24.0, 10.0),
        camera=camera,
        ground_z=0.0,
        position_ned=pose,
        euler_deg=(0.0, 0.0, 0.0),
        undistort=False,
    )
    down = 1.0 / math.sqrt(1.0001)
    assert off_axis.ok and off_axis.ned == pytest.approx((2.0, 4.095, 0.0), abs=1e-9)
    assert off_axis.ray_ned == pytest.approx((0.0, 0.01 * down, down), abs=1e-9)
