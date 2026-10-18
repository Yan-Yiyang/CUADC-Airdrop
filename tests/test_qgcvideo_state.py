from __future__ import annotations

from dataclasses import replace

import numpy as np

from airdrop.config import Config, GroundConfig, QgcVideoConfig
from airdrop.georef.camera import CameraModel
from airdrop.perception.models import (
    Detection,
    PixelBox,
    PoseSnapshot,
    ProcessedFrame,
    readonly_image,
    snapshot_detection,
)
from airdrop.qgcvideo.state import DisplayState
from airdrop.telemetry.models import TelemetrySnapshot


def test_display_timers_use_state_event_and_telemetry_timestamps() -> None:
    state = DisplayState(Config(), None, clock=lambda: 10.0)
    state.on_event("state", {"to_state": "RECON", "timestamp": 7.0})
    state.update_progress(
        lag_frames=0,
        processing=False,
        ocr_pending=0,
        queued_results=0,
        telemetry_timestamp=8.0,
    )
    snapshot = state.snapshot()
    assert snapshot.observed_at == 10.0
    assert snapshot.progress.stage_elapsed_s == 3.0
    assert snapshot.progress.telemetry_age_s == 2.0


def test_impact_details_are_read_from_existing_decision_event() -> None:
    state = DisplayState(Config(), None)
    state.on_event(
        "drop_check",
        {
            "impact_ned": [1.0, 2.0, 0.0],
            "horizontal_error_m": 1.4,
            "flight_time_s": 3.2,
        },
    )
    snapshot = state.snapshot()
    assert snapshot.predicted_impact_ned == (1.0, 2.0, 0.0)
    assert snapshot.impact_error_m == 1.4
    assert snapshot.impact_flight_time_s == 3.2


def test_inflight_detection_cannot_report_zero_remaining_time() -> None:
    state = DisplayState(Config(), None)
    state.update_progress(lag_frames=0, processing=True, ocr_pending=0, queued_results=0)
    assert state.snapshot().progress.eta_s is None


def test_dry_run_is_marked_before_software_release_record() -> None:
    state = DisplayState(Config(), None, simulated_release=True)
    state.on_event("drop_command_pending", {"simulated": True})
    assert state.snapshot().release_simulated
    state.on_event("drop_dry_run", {"count": 1})
    snapshot = state.snapshot()
    assert snapshot.release_command_status == "sent"
    assert snapshot.release_simulated
    assert not snapshot.released


def test_completed_stage_history_does_not_invent_skipped_drop_or_landing() -> None:
    state = DisplayState(Config(), None)
    for stage in ("RECON", "HOLD_PROCESS", "DONE"):
        state.on_event("state", {"to_state": stage})
    snapshot = state.snapshot()
    assert snapshot.completed_stages == ("HOLD_PROCESS", "RECON")
    assert "OVERFLY" not in snapshot.completed_stages
    assert "LAND" not in snapshot.completed_stages


def _frame(
    index: int,
    *,
    timestamp: float | None = None,
    image: np.ndarray | None = None,
    pose: PoseSnapshot | None = None,
) -> ProcessedFrame:
    pixels = np.zeros((720, 1280, 3), np.uint8) if image is None else image
    return ProcessedFrame(
        frame_index=index,
        received_timestamp=float(index if timestamp is None else timestamp),
        capture_timestamp=float(index if timestamp is None else timestamp),
        processed_timestamp=float(index),
        image=readonly_image(pixels),
        image_size=(pixels.shape[1], pixels.shape[0]),
        camera_matrix=None,
        undistorted=False,
        pose=pose,
        detections=(),
    )


def _detection(index: int, code: int | None = None, *, timestamp: float | None = None) -> Detection:
    box = PixelBox(639, 359, 641, 361)
    telemetry = TelemetrySnapshot(
        north_m=0,
        east_m=0,
        down_m=-100,
        roll_deg=0,
        pitch_deg=0,
        yaw_deg=0,
        origin_latitude_deg=31,
        origin_longitude_deg=121,
        origin_altitude_m=10,
    )
    return Detection(
        frame_index=index,
        capture_timestamp=float(index if timestamp is None else timestamp),
        pixel=box.center,
        box=box,
        confidence=0.9,
        telemetry=telemetry,
        code=code,
        mode="ocr",
    )


def _state(*, history_frames: int = 8, cloud_max: int = 16) -> DisplayState:
    config = replace(
        Config(),
        qgc_video=QgcVideoConfig(history_frames=history_frames, cloud_max=cloud_max),
        ground=GroundConfig(ground_point_alt=10),
    )
    camera = CameraModel(
        camera_matrix=np.array([[800, 0, 640], [0, 800, 360], [0, 0, 1]], dtype=float),
        image_size=(1280, 720),
    )
    return DisplayState(config, camera)


def test_zero_detection_frame_updates_and_history_is_bounded() -> None:
    state = _state(history_frames=2)
    state.on_frame(_frame(1))
    state.on_frame(_frame(2))
    state.on_frame(_frame(3))

    snapshot = state.snapshot()
    assert snapshot.frame is not None and snapshot.frame.frame_index == 3
    assert tuple(frame.frame_index for frame in snapshot.history_frames) == (2, 3)
    assert snapshot.detections == ()


def test_result_observer_matches_historical_frame_and_rejects_wrong_timestamp() -> None:
    state = _state()
    state.on_frame(
        _frame(
            4,
            pose=PoseSnapshot(
                position_ned=(0, 0, -100),
                quaternion_wxyz=(1, 0, 0, 0),
                euler_deg=(0, 0, 0),
                origin_lla=(121, 31, 10),
            ),
        )
    )
    state.on_detection(_detection(4, 12))
    state.on_detection(_detection(4, 13, timestamp=40))

    snapshot = state.snapshot()
    assert snapshot.observations[0].has_thumbnail
    assert snapshot.observations[0].position_ned == (0, 0, 0)
    assert not snapshot.observations[1].has_thumbnail
    assert snapshot.observations[1].position_ned == (0, 0, 0)
    assert snapshot.geometry_error == "结果没有匹配的在存历史帧"
    assert [entry.code for entry in snapshot.panel] == [12, 13]


def test_late_result_after_frame_eviction_keeps_counts_without_wrong_thumbnail() -> None:
    state = _state(history_frames=1)
    state.on_frame(_frame(1))
    state.on_frame(_frame(2))
    state.on_detection(_detection(1, 7))

    snapshot = state.snapshot()
    assert snapshot.observations[0].position_ned == (0, 0, 0)
    assert not snapshot.observations[0].has_thumbnail
    assert snapshot.panel[0].thumbnail_frame_index is None
    assert snapshot.recent_codes == (7,)


def test_hold_mode_uses_published_business_progress() -> None:
    state = _state()
    state.on_event("state", {"to_state": "HOLD_PROCESS"})
    state.update_progress(lag_frames=0, processing=False, ocr_pending=1, queued_results=0)
    assert state.snapshot().mode == "hold_follow"
    state.update_progress(lag_frames=0, processing=False, ocr_pending=0, queued_results=0)
    assert state.snapshot().mode == "hold_live"


def test_only_confirmed_release_event_marks_success_and_event_data_is_copied() -> None:
    state = _state()
    payload: list[int] = [1, 2]
    state.on_event("drop", {"reason": "candidate", "payload": payload})
    assert not state.snapshot().released
    payload.append(3)
    state.on_event("drop_command_pending", {"timestamp": 11.5})
    assert state.snapshot().release_command_status == "pending"
    state.on_event("drop_command_sent", {"timestamp": 12.0})
    state.on_event("drop_released", {"timestamp": 12.5, "count": 1, "simulated": True})

    snapshot = state.snapshot()
    assert snapshot.released
    assert snapshot.release_timestamp == 12.5
    assert snapshot.release_simulated
    assert snapshot.release_command_status == "sent"
    assert snapshot.release_command_timestamp == 12.0
    assert dict(snapshot.events[0].data)["payload"] == (1, 2)


def test_controller_release_event_marks_command_sent_before_runner_confirmation() -> None:
    state = DisplayState(Config(), None, clock=lambda: 100.0)
    state.on_event("drop_command_pending", {"timestamp": 90.0})
    state.on_event("gripper_release", {"instance": 1})
    state.on_event("drop_command_sent", {"timestamp": 105.0})

    snapshot = state.snapshot()
    assert snapshot.release_command_status == "sent"
    assert snapshot.release_command_timestamp == 100.0
    assert not snapshot.released


def test_hold_eta_and_timeout_use_published_progress_window() -> None:
    now = [0.0]
    config = replace(
        Config(),
        qgc_video=QgcVideoConfig(eta_window_s=2.0),
        ground=GroundConfig(ground_point_alt=10),
    )
    state = DisplayState(config, None, clock=lambda: now[0])
    state.on_event("state", {"to_state": "HOLD_PROCESS", "timestamp": 0.0})
    state.update_progress(
        lag_frames=6,
        processing=False,
        ocr_pending=2,
        queued_results=4,
        processed_frames=10,
        ocr_completed=8,
    )
    now[0] = 2.0
    state.update_progress(
        lag_frames=6,
        processing=False,
        ocr_pending=2,
        queued_results=2,
        processed_frames=20,
        ocr_completed=12,
    )

    progress = state.snapshot().progress
    assert progress.hold_elapsed_s == 2.0
    assert progress.hold_remaining_s == Config().mission.hold_process_max_s - 2.0
    assert progress.processing_fps == 5.0
    assert progress.ocr_rate_per_s == 2.0
    assert progress.result_drain_rate_per_s == 1.0
    assert progress.eta_s == 2.0


def test_live_frame_is_used_only_after_hold_drain_and_old_boxes_are_cleared() -> None:
    state = _state()
    old = _frame(1)
    state.on_frame(old)
    live_detection = _detection(2)
    live = replace(_frame(900), detections=(snapshot_detection(live_detection),))
    state.on_event("state", {"to_state": "RECON", "timestamp": 1.0})
    assert state.snapshot(live).frame is old

    state.on_event("state", {"to_state": "HOLD_PROCESS", "timestamp": 2.0})
    state.update_progress(lag_frames=1, processing=False, ocr_pending=0, queued_results=0)
    assert state.snapshot(live).frame is old
    state.update_progress(lag_frames=0, processing=False, ocr_pending=0, queued_results=0)
    state_live = state.snapshot(live)
    assert state_live.frame is not None and state_live.frame.frame_index == 900
    assert state_live.detections == ()
    state.on_event("state", {"to_state": "OVERFLY", "timestamp": 3.0})
    assert state.snapshot(live).mode == "drop"
