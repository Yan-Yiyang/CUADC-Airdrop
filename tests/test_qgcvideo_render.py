from __future__ import annotations

from dataclasses import replace

import cv2
import numpy as np
from PIL import Image, ImageDraw

from airdrop.georef.project import ProjectedPoint
from airdrop.perception.models import DetectionSnapshot, PoseSnapshot, ProcessedFrame
from airdrop.qgcvideo.models import (
    DisplayObservation,
    DisplaySnapshot,
    PanelEntry,
    ProgressSnapshot,
)
from airdrop.qgcvideo.render import FrameComposer, _fit_text_anchor, _font


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


def test_impact_error_is_labeled_as_decision_prediction_error(monkeypatch) -> None:
    snapshot, _ = _snapshot("drop")
    snapshot = replace(snapshot, impact_error_m=1.4, impact_flight_time_s=3.2)
    rendered: list[str] = []
    original = FrameComposer._draw_text_commands

    def capture(self: FrameComposer, canvas: np.ndarray) -> None:
        rendered.extend(command[0] for command in self._text_commands)
        original(self, canvas)

    monkeypatch.setattr(FrameComposer, "_draw_text_commands", capture)
    FrameComposer().compose(snapshot)

    assert any("最近判据预测" in item and "误差 1.4 米" in item for item in rendered)
    assert any("弹体预计飞行 3.2 秒" in item for item in rendered)


def test_projected_labels_use_live_horizontal_range_without_changing_decision_error(
    monkeypatch,
) -> None:
    snapshot, _ = _snapshot("drop")
    frame = snapshot.frame
    assert frame is not None
    pose = PoseSnapshot((100.0, 200.0, -50.0), None, None, None)
    frame = replace(frame, pose=pose)
    snapshot = replace(
        snapshot,
        frame=frame,
        target_ned=(103.0, 204.0, -10.0),
        impact_error_m=1.4,
        impact_flight_time_s=3.2,
        impact_prediction_timestamp=8.0,
        observed_at=11.0,
        target_projected=ProjectedPoint((103.0, 204.0, -10.0), (100.0, 100.0), True, True),
        impact_projected=ProjectedPoint((110.0, 212.0, 0.0), (200.0, 100.0), True, True),
    )
    rendered: list[str] = []
    original = FrameComposer._draw_text_commands

    def capture(self: FrameComposer, canvas: np.ndarray) -> None:
        rendered.extend(command[0] for command in self._text_commands)
        original(self, canvas)

    monkeypatch.setattr(FrameComposer, "_draw_text_commands", capture)
    FrameComposer().compose(snapshot)
    assert any("目标 · 水平距离 5 米" in item for item in rendered)
    assert any("最近预测落点 · 距该落点 16 米" in item for item in rendered)
    assert any("最近判据预测 · 事件年龄 3.0 秒 · 误差 1.4 米" in item for item in rendered)

    moved_frame = replace(frame, pose=replace(pose, position_ned=(102.0, 202.0, -50.0)))
    moved = replace(snapshot, frame=moved_frame)
    moved_rendered: list[str] = []

    def capture_moved(self: FrameComposer, canvas: np.ndarray) -> None:
        moved_rendered.extend(command[0] for command in self._text_commands)
        original(self, canvas)

    monkeypatch.setattr(FrameComposer, "_draw_text_commands", capture_moved)
    FrameComposer().compose(moved)
    assert any("目标 · 水平距离 2 米" in item for item in moved_rendered)
    assert any("误差 1.4 米" in item for item in moved_rendered)


def test_edge_arrow_length_is_fixed_and_clipped_labels_fit_inside_letterboxed_viewport(
    monkeypatch,
) -> None:
    snapshot, _ = _snapshot("drop")
    assert snapshot.frame is not None
    letterboxed_image = np.full((640, 1280, 3), (20, 40, 60), dtype=np.uint8)
    frame = replace(snapshot.frame, image=letterboxed_image, image_size=(1280, 640))
    snapshot = replace(snapshot, frame=frame)
    baseline = FrameComposer().compose(
        replace(snapshot, target_projected=None, impact_projected=None)
    )
    arrows: list[tuple[tuple[int, int], tuple[int, int]]] = []
    original_arrow = cv2.arrowedLine
    original_draw = FrameComposer._draw_text_commands
    clips: list[tuple[str, tuple[int, int, int, int]]] = []

    def capture_arrow(image, start, end, *args, **kwargs):
        arrows.append((start, end))
        return original_arrow(image, start, end, *args, **kwargs)

    def capture_text(self: FrameComposer, canvas: np.ndarray) -> None:
        clips.extend(
            (text, clip) for text, _, _, _, clip in self._text_commands if clip is not None
        )
        original_draw(self, canvas)

    monkeypatch.setattr("airdrop.qgcvideo.render.cv2.arrowedLine", capture_arrow)
    monkeypatch.setattr(FrameComposer, "_draw_text_commands", capture_text)
    composer = FrameComposer()
    edge_pixels = (
        (-1e300, -1e300),
        (1e300, -1e300),
        (-1e300, 1e300),
        (1e300, 1e300),
        (0.0, -1e300),
        (1e300, 0.0),
        (0.0, 1e300),
        (-1e300, 0.0),
    )
    result = baseline
    for pixel in edge_pixels:
        projected = ProjectedPoint((1.0, 2.0, 3.0), pixel, True, False)
        result = composer.compose(replace(snapshot, target_projected=projected))
    near_edge = ProjectedPoint((1.0, 2.0, 3.0), (0.0, 0.0), True, True)
    result = composer.compose(replace(snapshot, target_projected=near_edge))

    assert len(arrows) == len(edge_pixels)
    for start, end in arrows:
        assert abs(np.hypot(end[0] - start[0], end[1] - start[1]) - 48) <= 1
    assert np.array_equal(result[:130], baseline[:130])
    assert np.array_equal(result[:, 1280:], baseline[:, 1280:])
    assert np.array_equal(result[850:], baseline[850:])
    assert clips
    draw = ImageDraw.Draw(Image.new("RGB", (1760, 990)))
    for text, clip in clips:
        fitted = _fit_text_anchor(draw, text, _font(14), (clip[2] + 1000, clip[1] - 1000), clip)
        assert fitted is not None
        _, _, bounds = fitted
        assert bounds[0] - 7 >= clip[0]
        assert bounds[1] - 4 >= clip[1]
        assert bounds[2] + 7 < clip[2]
        assert bounds[3] + 4 < clip[3]


def test_overlapping_target_and_impact_labels_use_distinct_rows_at_viewport_corners(
    monkeypatch,
) -> None:
    snapshot, _ = _snapshot("drop")
    assert snapshot.frame is not None
    pose = PoseSnapshot((10.0, 20.0, -50.0), None, None, None)
    frame = replace(snapshot.frame, pose=pose)
    original = FrameComposer._draw_text_commands

    for pixel in ((0.0, 0.0), (639.0, 0.0), (0.0, 359.0), (639.0, 359.0)):
        labels: list[tuple[str, tuple[int, int], int, tuple[int, int, int, int]]] = []

        def capture(
            self: FrameComposer,
            canvas: np.ndarray,
            captured: list[tuple[str, tuple[int, int], int, tuple[int, int, int, int]]] = labels,
        ) -> None:
            captured.extend(
                (text, origin, size, clip)
                for text, origin, size, _, clip in self._text_commands
                if clip is not None
                and (text.startswith("目标 ·") or text.startswith("最近预测落点 ·"))
            )
            original(self, canvas)

        monkeypatch.setattr(FrameComposer, "_draw_text_commands", capture)
        projected = ProjectedPoint((12.0, 24.0, 0.0), pixel, True, True)
        FrameComposer().compose(
            replace(
                snapshot,
                frame=frame,
                target_projected=projected,
                impact_projected=projected,
            )
        )
        assert len(labels) == 2
        draw = ImageDraw.Draw(Image.new("RGB", (1760, 990)))
        rects = []
        for text, origin, size, clip in labels:
            fitted = _fit_text_anchor(draw, text, _font(size), origin, clip)
            assert fitted is not None
            _, _, bounds = fitted
            rects.append((bounds[0] - 7, bounds[1] - 4, bounds[2] + 7, bounds[3] + 4))
        first, second = rects
        assert first[0] >= 0 and first[1] >= 130 and first[2] < 1280 and first[3] < 850
        assert second[0] >= 0 and second[1] >= 130 and second[2] < 1280 and second[3] < 850
        assert (
            first[0] >= second[2]
            or second[0] >= first[2]
            or first[1] >= second[3]
            or second[1] >= first[3]
        ), (pixel, labels, rects)


def test_behind_geometry_is_neutral_and_never_claims_flypast(monkeypatch) -> None:
    snapshot, _ = _snapshot("drop")
    rendered: list[str] = []
    original = FrameComposer._draw_text_commands

    def capture(self: FrameComposer, canvas: np.ndarray) -> None:
        rendered.extend(command[0] for command in self._text_commands)
        original(self, canvas)

    monkeypatch.setattr(FrameComposer, "_draw_text_commands", capture)
    composer = FrameComposer()
    for reason in ("behind", "point behind camera"):
        composer.compose(
            replace(
                snapshot,
                target_projected=ProjectedPoint((0.0, 0.0, 0.0), None, False, False, reason),
            )
        )
    assert any("目标当前不在相机前方" in item for item in rendered)
    assert not any("已掠过" in item or "已飞掠" in item for item in rendered)


def test_flypast_latch_replaces_projection_arrows_and_uses_released_error_label(
    monkeypatch,
) -> None:
    snapshot, _ = _snapshot("drop")
    snapshot = replace(
        snapshot,
        released=True,
        impact_error_m=1.4,
        impact_flight_time_s=3.2,
        target_projected=ProjectedPoint((1.0, 2.0, 3.0), (100.0, 100.0), True, True),
        impact_projected=ProjectedPoint((4.0, 5.0, 6.0), (200.0, 100.0), True, True),
    )
    baseline = FrameComposer().compose(snapshot)
    rendered: list[str] = []
    arrow_calls = 0
    original = FrameComposer._draw_text_commands
    original_arrow = cv2.arrowedLine

    def capture(self: FrameComposer, canvas: np.ndarray) -> None:
        rendered.extend(command[0] for command in self._text_commands)
        original(self, canvas)

    def count_arrow(image, start, end, *args, **kwargs):
        nonlocal arrow_calls
        arrow_calls += 1
        return original_arrow(image, start, end, *args, **kwargs)

    monkeypatch.setattr(FrameComposer, "_draw_text_commands", capture)
    monkeypatch.setattr("airdrop.qgcvideo.render.cv2.arrowedLine", count_arrow)
    result = FrameComposer().compose(replace(snapshot, flypast_completed=True))

    assert arrow_calls == 0
    assert any("已飞掠目标航段" in item for item in rendered)
    assert any("投放时预测误差 1.4 米" in item for item in rendered)
    assert any("弹体预计飞行 3.2 秒" in item for item in rendered)
    assert tuple(result[330, 200]) == (20, 40, 60)
    assert not np.array_equal(result[130:850, :1280], baseline[130:850, :1280])


def test_preflight_stage_card_does_not_claim_recon_or_show_recon_timer(monkeypatch) -> None:
    snapshot, _ = _snapshot("standby")
    snapshot = replace(
        snapshot,
        stage="WAIT_AIRBORNE",
        progress=replace(
            snapshot.progress, stage_elapsed_s=120.0, mission_current=2, mission_total=16
        ),
    )
    rendered: list[str] = []
    original = FrameComposer._draw_text_commands

    def capture(self: FrameComposer, canvas: np.ndarray) -> None:
        rendered.extend(command[0] for command in self._text_commands)
        original(self, canvas)

    monkeypatch.setattr(FrameComposer, "_draw_text_commands", capture)
    FrameComposer().compose(snapshot)
    assert any("起飞前：仅实时预览 / 自检" in item for item in rendered)
    assert any("侦查尚未开始" in item for item in rendered)
    assert not any("侦查航线 2/16" in item or "阶段用时" in item for item in rendered)


def test_done_summary_does_not_fall_through_to_preflight(monkeypatch) -> None:
    snapshot, _ = _snapshot("done")
    snapshot = replace(snapshot, stage="DONE")
    rendered: list[str] = []
    original = FrameComposer._draw_text_commands

    def capture(self: FrameComposer, canvas: np.ndarray) -> None:
        rendered.extend(command[0] for command in self._text_commands)
        original(self, canvas)

    monkeypatch.setattr(FrameComposer, "_draw_text_commands", capture)
    FrameComposer().compose(snapshot)
    assert any("任务已完成" in item for item in rendered)
    assert not any("起飞前" in item or "侦查尚未开始" in item for item in rendered)


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
