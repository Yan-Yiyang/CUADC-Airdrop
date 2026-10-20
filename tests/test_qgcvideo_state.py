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
from airdrop.qgcvideo.prediction import LiveImpactPrediction
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


def test_recon_timer_starts_at_recon_not_during_preflight() -> None:
    now = [100.0]
    state = DisplayState(Config(), None, clock=lambda: now[0])
    state.on_event("state", {"to_state": "WAIT_AIRBORNE", "timestamp": 1.0})
    waiting = state.snapshot()
    assert waiting.mode == "standby"
    assert waiting.progress.stage_elapsed_s == 99.0

    now[0] = 203.0
    state.on_event("state", {"to_state": "RECON", "timestamp": 200.0})
    recon = state.snapshot()
    assert recon.mode == "recon"
    assert recon.progress.stage_elapsed_s == 3.0

    state.on_event("state", {"to_state": "UNRECOGNIZED_STAGE", "timestamp": 202.0})
    assert state.snapshot().mode == "standby"


def test_impact_details_are_read_from_existing_decision_event() -> None:
    state = DisplayState(Config(), None, clock=lambda: 10.0)
    state.on_event(
        "drop_check",
        {
            "impact_ned": [1.0, 2.0, 0.0],
            "horizontal_error_m": 1.4,
            "flight_time_s": 3.2,
            "evaluated_at": 8.5,
        },
    )
    snapshot = state.snapshot()
    assert snapshot.predicted_impact_ned is None
    assert snapshot.impact_error_m == 1.4
    assert snapshot.impact_flight_time_s == 3.2
    assert snapshot.impact_prediction_timestamp == 8.5


def test_impact_prediction_timestamp_falls_back_to_event_clock() -> None:
    state = DisplayState(Config(), None, clock=lambda: 17.0)
    state.on_event("drop_check", {"impact_ned": [1.0, 2.0, 0.0]})
    assert state.snapshot().impact_prediction_timestamp == 17.0


def test_released_prediction_freezes_only_matching_successful_release_event() -> None:
    state = DisplayState(Config(), None, clock=lambda: 100.0)
    state.on_event(
        "release",
        {
            "impact_ned": [10.0, 20.0, 0.0],
            "horizontal_error_m": 2.0,
            "flight_time_s": 3.0,
            "evaluated_at": 90.0,
        },
    )
    state.on_event("drop_check", {"impact_ned": [99.0, 99.0, 0.0], "horizontal_error_m": 90.0})
    assert state.snapshot().release_impact_ned is None
    state.on_event("drop_command_pending", {"count": 1})
    state.on_event("drop_released", {"timestamp": 91.0, "count": 1})
    state.on_event("drop_check", {"impact_ned": [80.0, 80.0, 0.0], "horizontal_error_m": 70.0})
    snapshot = state.snapshot()
    assert snapshot.release_impact_ned == (10.0, 20.0, 0.0)
    assert snapshot.release_error_m == 2.0
    assert snapshot.release_flight_time_s == 3.0
    assert snapshot.release_prediction_timestamp == 90.0


def test_live_prediction_is_projected_only_as_current_drop_stage_state() -> None:
    state = DisplayState(Config(), None)
    prediction = LiveImpactPrediction(
        True,
        "",
        (10.0, 20.0, 0.0),
        3.0,
        2.0,
        10.0,
        1.0,
        1,
        False,
        0.0,
        (0.0, 0.0, -10.0),
        (10.0, 0.0, 0.0),
    )
    state.on_event("state", {"to_state": "OVERFLY"})
    token, _, _ = state.prediction_context()
    state.set_live_prediction(prediction, token)
    assert state.snapshot().predicted_impact_ned == (10.0, 20.0, 0.0)
    state.on_event("state", {"to_state": "DONE"})
    assert state.snapshot().predicted_impact_ned is None


def test_stopped_prediction_context_rejects_late_snapshot_writeback() -> None:
    state = DisplayState(Config(), None)
    state.on_event("state", {"to_state": "OVERFLY"})
    old_token, _, _ = state.prediction_context()
    stopped_token = state.invalidate_live_prediction()
    late = LiveImpactPrediction(
        True,
        "",
        (1.0, 2.0, 0.0),
        3.0,
        2.0,
        10.0,
        1.0,
        1,
        True,
        0.0,
        (0.0, 0.0, -10.0),
        (10.0, 0.0, 0.0),
    )
    assert stopped_token > old_token
    assert not state.set_live_prediction(late, old_token)
    assert state.snapshot().live_prediction is None


def test_release_candidate_requires_matching_pending_and_released_count() -> None:
    state = DisplayState(Config(), None)
    payload = {
        "impact_ned": [1.0, 2.0, 0.0],
        "horizontal_error_m": 1.0,
        "flight_time_s": 2.0,
        "evaluated_at": 3.0,
    }
    state.on_event("release", payload)
    state.on_event("drop_released", {"count": 1})
    assert state.snapshot().release_impact_ned is None

    state.on_event("release", payload)
    state.on_event("drop_command_pending", {"count": 2})
    state.on_event("drop_released", {"count": 3})
    assert state.snapshot().release_impact_ned is None

    state.on_event("release", payload)
    state.on_event("drop_command_pending", {"count": 4})
    state.on_event("drop_released", {"count": 4})
    assert state.snapshot().release_impact_ned == (1.0, 2.0, 0.0)

    state.on_event("drop_released", {"count": 4})
    assert state.snapshot().release_impact_ned == (1.0, 2.0, 0.0)


def _set_mission_progress(state: DisplayState, current: int | None, total: int | None = 16) -> None:
    state.update_progress(
        lag_frames=0,
        processing=False,
        ocr_pending=0,
        queued_results=0,
        mission_current=current,
        mission_total=total,
    )


def _drop_plan_event() -> dict[str, object]:
    return {
        "target_ned": [10.0, 20.0, 0.0],
        "items": 16,
        "overfly_exit_seq": 1,
        "overfly_item_count": 2,
    }


def test_flypast_requires_new_plan_start_confirmed_matching_progress_then_latches() -> None:
    state = DisplayState(Config(), None)
    _set_mission_progress(state, 16)
    assert not state.snapshot().flypast_completed  # 侦查任务的旧 16/16 不继承

    state.on_event("drop_plan", _drop_plan_event())
    _set_mission_progress(state, 16)
    state.on_event("mission_start", {})
    state.on_event("mission_confirmed", {"mode": "MISSION"})
    _set_mission_progress(state, 16)
    assert not state.snapshot().flypast_completed  # 新任务同项数已完成进度也不算通过出口

    _set_mission_progress(state, 0)
    _set_mission_progress(state, 1)
    assert not state.snapshot().flypast_completed
    _set_mission_progress(state, 2)
    assert state.snapshot().flypast_completed

    state.on_event("state", {"to_state": "LAND"})
    _set_mission_progress(state, None, None)
    assert state.snapshot().flypast_completed  # 后续返航进度变化不撤销锁存


def test_flypast_lost_or_invalid_progress_clears_precompletion_armed_state() -> None:
    state = DisplayState(Config(), None)
    state.on_event("drop_plan", _drop_plan_event())
    state.on_event("mission_start", {})
    state.on_event("mission_confirmed", {"mode": "MISSION"})
    _set_mission_progress(state, 0)
    _set_mission_progress(state, None, None)
    _set_mission_progress(state, 16, 16)
    _set_mission_progress(state, 2, 16)
    assert not state.snapshot().flypast_completed

    _set_mission_progress(state, 1, 16)
    _set_mission_progress(state, 16, 15)
    _set_mission_progress(state, 2, 16)
    assert not state.snapshot().flypast_completed

    _set_mission_progress(state, 1, 16)
    _set_mission_progress(state, 2, 16)
    assert state.snapshot().flypast_completed


def test_flypast_is_not_inferred_from_release_land_or_missing_plan_metadata() -> None:
    state = DisplayState(Config(), None)
    state.on_event("drop_plan", {"items": 16, "target_ned": [1.0, 2.0, 0.0]})
    state.on_event("mission_start", {})
    state.on_event("mission_confirmed", {"mode": "MISSION"})
    _set_mission_progress(state, 2)
    assert not state.snapshot().flypast_completed

    state.on_event("drop_released", {"simulated": True})
    state.on_event("state", {"to_state": "LAND"})
    assert not state.snapshot().flypast_completed

    state.on_event("drop_plan", _drop_plan_event())
    assert not state.snapshot().flypast_completed
    _set_mission_progress(state, 2)
    assert not state.snapshot().flypast_completed


def test_new_lifecycle_clears_old_target_and_flypast_state() -> None:
    state = DisplayState(Config(), None)
    state.on_event("drop_plan", _drop_plan_event())
    state.on_event("mission_start", {})
    state.on_event("mission_confirmed", {"mode": "MISSION"})
    _set_mission_progress(state, 1)
    _set_mission_progress(state, 2)
    assert state.snapshot().flypast_completed

    state.on_event("state", {"to_state": "INIT"})
    snapshot = state.snapshot()
    assert snapshot.mode == "standby"
    assert snapshot.target_ned is None
    assert not snapshot.flypast_completed


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
    original = snapshot_detection(_detection(4, None))
    state.on_frame(
        replace(
            _frame(
                4,
                pose=PoseSnapshot(
                    position_ned=(0, 0, -100),
                    quaternion_wxyz=(1, 0, 0, 0),
                    euler_deg=(0, 0, 0),
                    origin_lla=(121, 31, 10),
                ),
            ),
            detections=(original,),
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


def test_panel_keeps_latest_thumbnail_after_frame_history_and_observation_cloud_expire() -> None:
    state = _state(history_frames=2, cloud_max=2)
    source = np.zeros((720, 1280, 3), dtype=np.uint8)
    source[358:362, 638:642] = (10, 80, 220)
    first = _detection(1, None)
    state.on_frame(replace(_frame(1, image=source), detections=(snapshot_detection(first),)))
    state.on_detection(replace(snapshot_detection(first), code=56))
    for index in range(2, 12):
        state.on_frame(_frame(index))
    state.on_detection(replace(snapshot_detection(_detection(10, 12)), code=12))
    state.on_detection(replace(snapshot_detection(_detection(11, 94)), code=94))

    snapshot = state.snapshot()
    entry = next(item for item in snapshot.panel if item.code == 56)
    assert all(item.code != 56 for item in snapshot.observations)
    assert entry.count == 0
    assert entry.thumbnail_frame_index == 1
    assert entry.thumbnail is not None
    assert entry.thumbnail.shape[0] <= 128 and entry.thumbnail.shape[1] <= 128
    assert not np.shares_memory(entry.thumbnail, source)
    assert not entry.thumbnail.flags.writeable
    assert 56 in snapshot.recent_codes

    from airdrop.qgcvideo.render import FrameComposer

    assert FrameComposer._thumbnail(snapshot, entry) is entry.thumbnail
    assert snapshot.observed_at is not None
    assert FrameComposer._thumbnail_age(snapshot, entry, snapshot.observed_at) == (
        f"{max(0.0, snapshot.observed_at - 1.0):.1f}秒前"
    )
    composed = FrameComposer().compose(snapshot)
    assert np.any(np.all(composed[226:306, 1306:1744] == (10, 80, 220), axis=2))


def test_newest_code_thumbnail_wins_and_delayed_old_result_cannot_replace_it() -> None:
    state = _state(history_frames=1)
    first = _detection(1, None)
    second = _detection(2, None)
    state.on_frame(replace(_frame(1), detections=(snapshot_detection(first),)))
    state.on_detection(replace(snapshot_detection(first), code=12))
    state.on_frame(replace(_frame(2), detections=(snapshot_detection(second),)))
    state.on_detection(replace(snapshot_detection(second), code=12))
    state.on_detection(replace(snapshot_detection(first), code=12))

    entry = next(item for item in state.snapshot().panel if item.code == 12)
    assert entry.thumbnail_frame_index == 2


def test_late_ocr_uses_exact_bounded_pending_thumbnail_and_new_state_is_empty() -> None:
    state = _state(history_frames=1)
    detections = []
    for index in range(1, 131):
        detection = _detection(index, None)
        detections.append(detection)
        image = np.zeros((720, 1280, 3), dtype=np.uint8)
        color = (7, 180, 33) if index == 129 else (200, 20, 90)
        image[358:362, 638:642] = color
        state.on_frame(
            replace(
                _frame(index, image=image),
                detections=(snapshot_detection(detection),),
            )
        )
    assert len(state._pending_thumbnails) == 128
    assert 129 not in state._history

    delayed = snapshot_detection(detections[-2])
    state.on_detection(replace(delayed, code=94))
    entry = next(item for item in state.snapshot().panel if item.code == 94)
    assert entry.thumbnail_frame_index == 129
    assert entry.count == 1
    assert entry.thumbnail is not None
    assert tuple(entry.thumbnail[0, 0]) == (7, 180, 33)
    assert _state().snapshot().panel == ()


def test_invalid_or_unidentified_thumbnail_does_not_replace_valid_code_image() -> None:
    state = _state()
    first = _detection(1, None)
    state.on_frame(replace(_frame(1), detections=(snapshot_detection(first),)))
    state.on_detection(replace(snapshot_detection(first), code=17))
    wrong_time = snapshot_detection(_detection(1, None, timestamp=10.0))
    state.on_detection(replace(wrong_time, code=17))
    invalid = replace(snapshot_detection(_detection(2, None)), box=(float("nan"), 0, 4, 4))
    state.on_frame(replace(_frame(2), detections=(invalid,)))
    state.on_detection(replace(invalid, code=17))
    state.on_detection(replace(snapshot_detection(_detection(3, None)), code=None))

    entry = next(item for item in state.snapshot().panel if item.code == 17)
    assert entry.thumbnail_frame_index == 1


def test_same_frame_wrong_detection_identity_cannot_replace_thumbnail() -> None:
    state = _state()
    source = np.zeros((720, 1280, 3), dtype=np.uint8)
    source[358:362, 638:642] = (10, 80, 220)
    original = snapshot_detection(_detection(1, None))
    state.on_frame(replace(_frame(1, image=source), detections=(original,)))
    state.on_detection(replace(original, code=17))

    wrong_identity = replace(original, pixel=(700.0, 400.0), box=(698.0, 398.0, 702.0, 402.0))
    state.on_detection(replace(wrong_identity, code=17))

    entry = next(item for item in state.snapshot().panel if item.code == 17)
    assert entry.thumbnail_frame_index == 1
    assert entry.thumbnail is not None
    assert tuple(entry.thumbnail[0, 0]) == (10, 80, 220)


def test_repeated_wrong_identity_cannot_self_certify_a_new_thumbnail() -> None:
    state = _state()
    original = snapshot_detection(_detection(1, None))
    state.on_frame(replace(_frame(1), detections=(original,)))
    wrong_identity = replace(original, pixel=(700.0, 400.0), box=(698.0, 398.0, 702.0, 402.0))
    state.on_detection(replace(wrong_identity, code=18))
    state.on_detection(replace(wrong_identity, code=18))

    entry = next(item for item in state.snapshot().panel if item.code == 18)
    assert entry.count == 2
    assert entry.thumbnail is None
    from airdrop.qgcvideo.render import FrameComposer

    assert FrameComposer._thumbnail(state.snapshot(), entry) is None


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
