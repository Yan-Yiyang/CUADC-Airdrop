from __future__ import annotations

from dataclasses import replace

import numpy as np

from airdrop.georef.project import ProjectedPoint
from airdrop.perception.models import DetectionSnapshot, PoseSnapshot, ProcessedFrame
from airdrop.qgcvideo.models import (
    DisplayObservation,
    DisplaySnapshot,
    PanelEntry,
    ProgressSnapshot,
)
from airdrop.qgcvideo.render import FrameComposer


def _snapshot(mode: str = "recon") -> tuple[DisplaySnapshot, np.ndarray]:
    image = np.full((360, 640, 3), (20, 40, 60), dtype=np.uint8)
    detection = DetectionSnapshot(
        frame_index=4,
        capture_timestamp=1.0,
        pixel=(30.0, 30.0),
        box=(20.0, 20.0, 60.0, 60.0),
        confidence=0.9,
        code=17,
        side_px=25.0,
        raw_text="17",
        ocr_confidence=0.8,
        mode="recon",
    )
    frame = ProcessedFrame(
        frame_index=4,
        received_timestamp=1.0,
        capture_timestamp=1.0,
        processed_timestamp=1.1,
        image=image,
        image_size=(640, 360),
        camera_matrix=None,
        undistorted=False,
        pose=PoseSnapshot(None, None, None, None),
        detections=(detection,),
    )
    invalid = ProjectedPoint((1.0, 2.0, 3.0), None, False, False, "behind")
    return (
        DisplaySnapshot(
            stage="RECON",
            mode=mode,
            frame=frame,
            history_frames=(frame,),
            detections=(detection,),
            observations=(),
            projected_observations=(),
            panel=(PanelEntry(17, 3, None, None, 4, (20.0, 20.0, 60.0, 60.0)),),
            recent_codes=(17,),
            progress=ProgressSnapshot(),
            target_ned=None,
            predicted_impact_ned=None,
            target_projected=invalid,
            impact_projected=invalid,
            released=False,
            release_timestamp=None,
            release_simulated=False,
            release_command_status="not_sent",
            release_command_timestamp=None,
            events=(),
            geometry_error=None,
            capture_delay_s=0.1,
        ),
        image,
    )


def test_compose_letterboxes_without_mutating_frame_and_draws_same_frame_box() -> None:
    snapshot, source = _snapshot()
    original = source.copy()

    result = FrameComposer().compose(snapshot)

    assert result.shape == (990, 1760, 3)
    assert result.dtype == np.uint8
    assert np.array_equal(source, original)
    # 640×360 按比例放到 1280×720，不留黑边。
    assert tuple(result[400, 500]) == (20, 40, 60)
    # 框映射到 x=40, y=170。
    assert tuple(result[170, 40]) != (20, 40, 60)
    # 侧栏含目标编号面板，整块区域与底色不同。
    assert not np.array_equal(result[150, 1300], np.array((18, 22, 27), dtype=np.uint8))


def test_drop_mode_never_reuses_detection_boxes() -> None:
    snapshot, _ = _snapshot("drop")

    result = FrameComposer().compose(snapshot)

    # drop 模式的原帧区域没有 recon 检测框的绿色边线。
    assert tuple(result[240, 40]) == (20, 40, 60)


def test_bad_projected_point_does_not_draw_a_stale_or_invalid_marker() -> None:
    snapshot, _ = _snapshot("drop")

    result = FrameComposer().compose(snapshot)

    assert tuple(result[240, 40]) == (20, 40, 60)


def test_out_of_bounds_observation_cannot_draw_into_status_or_panel() -> None:
    snapshot, _ = _snapshot()
    outside = ProjectedPoint((1.0, 2.0, 3.0), (700.0, 10.0), True, False, "out of bounds")
    snapshot = replace(snapshot, projected_observations=(outside,))

    expected = FrameComposer().compose(replace(snapshot, projected_observations=()))
    result = FrameComposer().compose(snapshot)

    # 错误观测映射后会落在右侧面板；它必须被视频视口检查拦截。
    assert np.array_equal(result, expected)


def test_release_text_distinguishes_command_from_recorded_release() -> None:
    snapshot, _ = _snapshot()

    assert FrameComposer._release_text(replace(snapshot, release_command_status="pending")) == (
        "正在发送指令"
    )
    assert FrameComposer._release_text(replace(snapshot, release_command_status="sent")) == (
        "投放指令已发出"
    )
    simulated = replace(snapshot, release_simulated=True)
    for status, expected in (
        ("pending", "正在发送指令（模拟）"),
        ("sent", "投放指令已发出（模拟）"),
        ("failed", "投放指令失败（模拟）"),
        ("unknown", "投放结果未知（模拟）"),
    ):
        assert (
            FrameComposer._release_text(replace(simulated, release_command_status=status))
            == expected
        )
    assert (
        FrameComposer._release_text(replace(snapshot, released=True, release_timestamp=2.0))
        == "投放已记录"
    )


def test_observation_circle_is_clipped_to_video_viewport() -> None:
    snapshot, _ = _snapshot()
    edge = ProjectedPoint((1.0, 2.0, 3.0), (639.0, 10.0), True, True, "")
    snapshot = replace(snapshot, projected_observations=(edge,))

    result = FrameComposer().compose(snapshot)

    assert tuple(result[140, 1280]) == (60, 49, 38)


def test_hold_live_keeps_colored_observation_cloud_without_old_detection_box() -> None:
    snapshot, _ = _snapshot("hold_live")
    point = ProjectedPoint((1.0, 2.0, 3.0), (100.0, 100.0), True, True, "")
    snapshot = replace(
        snapshot,
        observations=(
            DisplayObservation(
                4, 1.0, (1.0, 2.0, 3.0), 17, 0.9, (100.0, 100.0), (90, 90, 110, 110), True
            ),
        ),
        projected_observations=(point,),
    )

    result = FrameComposer().compose(snapshot)

    assert tuple(result[330, 200]) == (130, 59, 148)
    assert tuple(result[220, 40]) == (20, 40, 60)


def test_hold_stage_card_shows_timeout_and_processing_eta(monkeypatch) -> None:
    snapshot, _ = _snapshot()
    snapshot = replace(
        snapshot,
        stage="HOLD_PROCESS",
        mode="hold_follow",
        progress=replace(
            snapshot.progress,
            stage_elapsed_s=34.6,
            hold_elapsed_s=34.6,
            hold_remaining_s=7.4,
            eta_s=1.8,
        ),
    )
    rendered: list[str] = []
    original = FrameComposer._draw_text_commands

    def capture(self: FrameComposer, canvas: np.ndarray) -> None:
        rendered.extend(command[0] for command in self._text_commands)
        original(self, canvas)

    monkeypatch.setattr(FrameComposer, "_draw_text_commands", capture)
    FrameComposer().compose(snapshot)

    assert any("距超时 7.4 秒" in item for item in rendered)
    assert any("预计处理 1.8 秒" in item for item in rendered)


def test_drop_markers_use_distinct_target_and_impact_shapes() -> None:
    snapshot, _ = _snapshot("drop")
    snapshot = replace(
        snapshot,
        target_projected=ProjectedPoint((1.0, 2.0, 3.0), (100.0, 100.0), True, True),
        impact_projected=ProjectedPoint((4.0, 5.0, 6.0), (200.0, 100.0), True, True),
    )

    result = FrameComposer().compose(snapshot)

    assert tuple(result[330, 200]) == (20, 40, 60)  # 中空准星不遮住实体靶标
    assert tuple(result[330, 226]) == (55, 220, 100)
    assert tuple(result[330, 400]) == (35, 80, 245)


def test_abort_does_not_mark_any_mission_phase_complete(monkeypatch) -> None:
    snapshot, _ = _snapshot()
    snapshot = replace(snapshot, stage="ABORT", mode="drop")
    rendered: list[str] = []
    original = FrameComposer._draw_text_commands

    def capture(self: FrameComposer, canvas: np.ndarray) -> None:
        rendered.extend(command[0] for command in self._text_commands)
        original(self, canvas)

    monkeypatch.setattr(FrameComposer, "_draw_text_commands", capture)
    result = FrameComposer().compose(snapshot)

    assert any("任务已中止" in item for item in rendered)
    assert tuple(result[45, 45]) == (58, 45, 34)


def test_non_recon_stage_labels_mission_counter_as_current_route(monkeypatch) -> None:
    snapshot, _ = _snapshot()
    snapshot = replace(
        snapshot,
        stage="OVERFLY",
        progress=replace(snapshot.progress, mission_current=2, mission_total=5),
    )
    rendered: list[str] = []
    original = FrameComposer._draw_text_commands

    def capture(self: FrameComposer, canvas: np.ndarray) -> None:
        rendered.extend(command[0] for command in self._text_commands)
        original(self, canvas)

    monkeypatch.setattr(FrameComposer, "_draw_text_commands", capture)
    FrameComposer().compose(snapshot)

    assert any("当前航线 2/5" in item for item in rendered)
    assert not any("侦查航线 2/5" in item for item in rendered)


def test_replay_uses_source_timeline_without_live_delay_claims(monkeypatch) -> None:
    snapshot, _ = _snapshot()
    assert snapshot.frame is not None
    snapshot = replace(
        snapshot,
        stage=None,
        replay_mode=True,
        observed_at=snapshot.frame.capture_timestamp,
        capture_delay_s=None,
        progress=replace(snapshot.progress, telemetry_age_s=None),
    )
    rendered: list[str] = []
    original = FrameComposer._draw_text_commands

    def capture(self: FrameComposer, canvas: np.ndarray) -> None:
        rendered.extend(command[0] for command in self._text_commands)
        original(self, canvas)

    monkeypatch.setattr(FrameComposer, "_draw_text_commands", capture)
    FrameComposer().compose(snapshot)

    assert any("离线回放（原时间轴）" in item for item in rendered)
    assert "遥测时延 回放不适用" in rendered
    assert "回放不包含投放状态" in rendered


def test_impact_error_is_shown_without_substituting_aircraft_distance(monkeypatch) -> None:
    snapshot, _ = _snapshot("drop")
    snapshot = replace(snapshot, impact_error_m=1.4, impact_flight_time_s=3.2)
    rendered: list[str] = []
    original = FrameComposer._draw_text_commands

    def capture(self: FrameComposer, canvas: np.ndarray) -> None:
        rendered.extend(command[0] for command in self._text_commands)
        original(self, canvas)

    monkeypatch.setattr(FrameComposer, "_draw_text_commands", capture)
    FrameComposer().compose(snapshot)

    assert any("落点误差 1.4 米" in item for item in rendered)
    assert any("飞行 3.2 秒" in item for item in rendered)


def test_legal_projected_marker_at_view_edge_cannot_touch_panel_or_status() -> None:
    snapshot, _ = _snapshot("drop")
    baseline = FrameComposer().compose(
        replace(snapshot, target_projected=None, impact_projected=None)
    )
    edge = ProjectedPoint((1.0, 2.0, 3.0), (639.0, 180.0), True, True, "")
    result = FrameComposer().compose(
        replace(snapshot, target_projected=edge, impact_projected=None)
    )

    assert np.array_equal(result[:130], baseline[:130])
    assert np.array_equal(result[:, 1280:], baseline[:, 1280:])


def test_done_without_landing_does_not_mark_skipped_phases_complete() -> None:
    snapshot, _ = _snapshot()
    snapshot = replace(snapshot, stage="DONE", completed_stages=("RECON", "HOLD_PROCESS"))
    result = FrameComposer().compose(snapshot)
    assert tuple(result[45, 750]) == (58, 45, 34)  # 飞掠投放未经历
    assert tuple(result[45, 930]) == (58, 45, 34)  # 返航降落未经历


def test_competition_demo_uses_physical_scale_and_median_target() -> None:
    from tools import qgc_demo
    from tools.qgc_video_test import _snapshot as demo_snapshot

    snapshot = demo_snapshot(12, 7.0, 10.0)
    assert snapshot.frame is not None and snapshot.frame.pose is not None
    assert snapshot.frame.image_size == (1280, 720)
    assert snapshot.frame.pose.position_ned == (-16.0, 190.0, -15.0)
    assert {entry.code for entry in snapshot.panel} == {12, 56, 94}
    assert snapshot.target_ned == qgc_demo.SELECTED.ned
    assert qgc_demo.SELECTED.number == sorted(entry.code for entry in snapshot.panel)[1]
    assert snapshot.target_projected is not None
    selected = next(item for item in snapshot.frame.detections if item.code == 56)
    assert snapshot.target_projected.pixel == selected.pixel
    assert len(qgc_demo.WELLS) == 4 and sum(well.code is not None for well in qgc_demo.WELLS) == 3
    for first_frame, second_frame in zip(
        snapshot.history_frames[:-1], snapshot.history_frames[1:], strict=True
    ):
        assert first_frame.pose is not None and second_frame.pose is not None
        first_position, second_position = (
            first_frame.pose.position_ned,
            second_frame.pose.position_ned,
        )
        assert first_position is not None and second_position is not None
        distance = np.linalg.norm(np.array(second_position) - np.array(first_position))
        interval = second_frame.capture_timestamp - first_frame.capture_timestamp
        assert distance / interval < 20
        assert second_frame.frame_index - first_frame.frame_index == round(interval * 12)
    for i, first in enumerate(qgc_demo.WELLS):
        for second in qgc_demo.WELLS[i + 1 :]:
            assert np.hypot(first.north_m - second.north_m, first.east_m - second.east_m) > 20


def test_competition_demo_does_not_publish_target_before_selection() -> None:
    from tools.qgc_video_test import _snapshot as demo_snapshot

    snapshot = demo_snapshot(12, 1.0, 10.0)
    assert snapshot.target_ned is None and snapshot.predicted_impact_ned is None
    assert snapshot.frame is not None
    assert {item.code for item in snapshot.frame.detections} == {12, 56}
    assert not snapshot.frame.image.flags.writeable
