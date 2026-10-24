"""离线标定包的独立文件格式、导入边界与几何契约。"""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

pytestmark = pytest.mark.integration

pytest.importorskip("airdrop_calibration", reason="跨仓契约需要标定子模块")


@pytest.mark.parametrize("position_delay", [0.0, 0.18])
def test_calibration_pose_uses_the_same_source_time_as_runtime(position_delay: float) -> None:
    """异步位置和变速源钟下，标定配对须与在线查询保持同一位姿。"""
    from airdrop_calibration.models import Telemetry
    from airdrop_calibration.source_timeline import SourceTimeline

    from airdrop import Config, TelemetryBroker
    from airdrop.georef import quaternion_to_matrix
    from airdrop.telemetry.models import TelemetrySnapshot

    assert Config().telemetry.source_time_alignment
    events = []
    for index in range(9):
        source = 10 + index * 0.1
        received = 100 + min(index, 4) * 0.1 + max(index - 4, 0) * 0.2
        angle = source * 0.3
        events.extend(
            [
                (
                    received,
                    {
                        "quaternion_timestamp_us": round(source * 1e6),
                        "quaternion_received_at": received,
                        "quaternion_w": math.cos(angle / 2),
                        "quaternion_x": 0.0,
                        "quaternion_y": 0.0,
                        "quaternion_z": math.sin(angle / 2),
                    },
                ),
                (
                    received + position_delay,
                    {
                        "position_timestamp_us": round(source * 1e6),
                        "position_velocity_received_at": received + position_delay,
                        "north_m": source**2,
                        "east_m": source * 2,
                        "down_m": -20.0,
                        "vx_m_s": source * 2,
                        "vy_m_s": 2.0,
                        "vz_m_s": 0.0,
                    },
                ),
            ]
        )
    broker = TelemetryBroker(history_interval=0, source_time_alignment=True)
    state = {}
    rows = []
    fields = Telemetry.__dataclass_fields__
    for received, update in sorted(events, key=lambda item: item[0]):
        state.update(update)
        snapshot = TelemetrySnapshot(timestamp=received, **state)
        broker._update_from_snapshot(snapshot)
        rows.append(
            Telemetry(**{key: value for key, value in snapshot.as_dict().items() if key in fields})
        )
    timeline = SourceTimeline(rows)
    for stamp in (100.05, 100.25, 100.45, 100.7, 101.05):
        online = broker.get_snapshot_at(stamp)
        offline = timeline.pose_at(stamp)
        assert online is not None and offline is not None
        assert offline[:3, 3] == pytest.approx((online.north_m, online.east_m, online.down_m))
        quaternion = (
            online.quaternion_w,
            online.quaternion_x,
            online.quaternion_y,
            online.quaternion_z,
        )
        assert all(value is not None for value in quaternion)
        rotation = quaternion_to_matrix(
            *(float(value) for value in quaternion if value is not None)
        )
        assert offline[:3, :3] == pytest.approx(rotation, abs=1e-10)


def test_full_update_recorder_contract_reaches_calibration_validator(tmp_path: Path) -> None:
    """真实记录器的全量队列统计、JSONL 与标定校验器保持同一契约。"""
    from airdrop_calibration.calibrate import _validate_full_update_capture
    from airdrop_calibration.flight_log import load_flight_telemetry
    from airdrop_calibration.source_timeline import SourceTimeline

    from airdrop import AlignmentBuffer, Config, FlightRecorder, TelemetryBroker
    from airdrop.telemetry.models import TelemetrySnapshot
    from airdrop.video.align import AlignedSample
    from airdrop.video.source import VideoFrame

    config = Config()
    config = replace(config, record=replace(config.record, dir=str(tmp_path), telemetry_hz=0.0))
    broker = TelemetryBroker()
    buffer = AlignmentBuffer(capacity=4)
    recorder = FlightRecorder(config.validated(), base_dir=tmp_path / "records")
    recorder.start(broker=broker, buffer=buffer)
    for index in range(6):
        received = 100.0 + index * 0.01
        position_index = index // 2
        snapshot = TelemetrySnapshot(
            timestamp=received,
            north_m=position_index / 2,
            east_m=0.0,
            down_m=-10.0,
            quaternion_w=1.0,
            quaternion_x=0.0,
            quaternion_y=0.0,
            quaternion_z=0.0,
            position_timestamp_us=20_000 + position_index * 20_000,
            position_velocity_received_at=100.0 + position_index * 0.02,
            quaternion_timestamp_us=100_000 + index * 10_000,
            quaternion_received_at=received,
        )
        broker.publish(snapshot)
    for index in (1, 2):
        received = 100.0 + index * 0.1
        frame = VideoFrame(
            index=index,
            image=np.zeros((16, 16, 3), dtype=np.uint8),
            timestamp=received,
        )
        buffer.put(
            AlignedSample(
                frame=frame,
                snapshot=TelemetrySnapshot(timestamp=received),
                timestamp=received,
                lag=0.0,
                mode="interpolate",
            )
        )
    recorder.stop()
    flight = recorder.flight_dir
    assert flight is not None
    telemetry_rows = load_flight_telemetry(flight)
    quality = _validate_full_update_capture(flight, telemetry_rows)
    assert quality is not None
    assert quality["telemetry"]["telemetry_written"] == 6
    timeline = SourceTimeline(load_flight_telemetry(flight))
    assert timeline.quality_meta["quaternion"]["native_samples"] == 6
    assert timeline.quality_meta["position"]["native_samples"] == 3
    assert timeline.quality_meta["quaternion"]["received_hz"] == pytest.approx(100.0)
    assert timeline.quality_meta["position"]["received_hz"] == pytest.approx(50.0)
    first_frame = next((flight / "frames").glob("*.jpg"))
    first_frame.unlink()
    with pytest.raises(ValueError, match="帧文件缺失"):
        _validate_full_update_capture(flight, telemetry_rows)


@pytest.mark.parametrize(
    "event_update",
    [
        {"kind": "wrong_end"},
        {"telemetry_dropped": 1},
        {"telemetry_write_errors": 1},
        {"telemetry_pending": 1},
        {"frame_encode_errors": 1},
        {"telemetry_written": 99},
    ],
)
def test_full_update_validator_rejects_incomplete_capture(
    tmp_path: Path, event_update: dict[str, object]
) -> None:
    """全量记录的结束事件和写入计数缺失或不一致时拒绝标定。"""
    import json

    from airdrop_calibration.calibrate import _validate_full_update_capture

    flight = tmp_path / "flight"
    flight.mkdir()
    (flight / "config_snapshot.json").write_text(
        json.dumps({"record": {"telemetry_hz": 0}}), encoding="utf-8"
    )
    ended = {
        "kind": "recorder_stopped",
        "telemetry_received": 0,
        "telemetry_enqueued": 0,
        "telemetry_written": 0,
        "telemetry_dropped": 0,
        "telemetry_write_errors": 0,
        "telemetry_pending": 0,
        "frames_written": 2,
        "frames_skipped": 0,
        "frame_encode_errors": 0,
        "finalize_errors": 0,
        "events": 1,
    }
    ended.update(event_update)
    (flight / "events.jsonl").write_text(
        json.dumps({"kind": "recorder_started"}) + "\n" + json.dumps(ended) + "\n",
        encoding="utf-8",
    )
    (flight / "telemetry.jsonl").write_text("", encoding="utf-8")
    (flight / "frames_index.jsonl").write_text(
        json.dumps({"received_timestamp": 10.0})
        + "\n"
        + json.dumps({"received_timestamp": 11.0})
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        _validate_full_update_capture(flight, [])


def test_handwritten_flightrecorder_files_and_camera_model_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """独立手写 JSONL 可读取，来源 schema 可由生产相机模型加载。"""
    from airdrop_calibration import calibrate
    from airdrop_calibration.algorithms import rotation_z
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
            r_bc=rotation_z(90.0),
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
        translation_diagnostic={
            "status": "disabled",
            "reason": "disabled_by_user",
            "measured": [1.0, 2.0, 0.5],
            "estimated": None,
            "delta_xyz_m": None,
            "delta_m": None,
            "residual_m": None,
            "pairs": 0,
        },
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
