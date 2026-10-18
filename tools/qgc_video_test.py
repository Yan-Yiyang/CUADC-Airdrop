"""生成合成任务画面并通过 QGC 视频发布器发送。"""

from __future__ import annotations

import math
import time
from pathlib import Path

HOST = "127.0.0.1"
PORT = 5602
TRANSPORT = "rtp"
FPS = 12.0
SECONDS = 10.0
_STAGE_ORDER = ("PREFLIGHT", "WAIT_AIRBORNE", "RECON", "HOLD_PROCESS", "OVERFLY", "LAND")

_PHASES = (
    ("RECON", "recon"),
    ("HOLD_PROCESS", "hold_follow"),
    ("HOLD_PROCESS", "hold_live"),
    ("OVERFLY", "drop"),
    ("LAND", "drop"),
)


def _snapshot(index: int, elapsed: float, seconds: float):
    from airdrop.georef.geo import ned_to_wgs84
    from airdrop.georef.project import ned_to_pixel
    from airdrop.perception.models import DetectionSnapshot, PoseSnapshot, ProcessedFrame
    from airdrop.qgcvideo.models import (
        DisplayObservation,
        DisplaySnapshot,
        PanelEntry,
        ProgressSnapshot,
    )

    from . import qgc_demo as demo

    phase_index = min(len(_PHASES) - 1, int(elapsed / seconds * len(_PHASES)))
    stage, mode = _PHASES[phase_index]
    now = time.time()
    # 三个历史时刻的相机都遵循 15m 下视几何；主画面只覆盖当时视场。
    views = (
        (demo.WELLS[0].north_m, demo.WELLS[0].east_m),
        (demo.WELLS[1].north_m, demo.WELLS[1].east_m),
        (-16.0, 190.0),
    )
    # 历史采样间隔留出正常飞行时间，帧号按演示的 12fps 对齐。
    history_ages = (4.0, 1.0, 0.0)
    frames = []
    original = {}
    for offset, (well, (north, east)) in enumerate(zip(demo.WELLS[:3], views, strict=True)):
        frame_index = 4820 + index - round(history_ages[offset] * FPS)
        capture = now - 0.25 - history_ages[offset]
        pose = PoseSnapshot(
            (north, east, -demo.ALTITUDE_M),
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            demo.ORIGIN_LLA,
        )
        visible = []
        for candidate in demo.WELLS[:3]:
            pixel = demo.project(candidate.ned, north, east)
            box = demo.well_box(candidate, north, east)
            if not (0 <= pixel[0] < demo.WIDTH and 0 <= pixel[1] < demo.HEIGHT):
                continue
            detection = DetectionSnapshot(
                frame_index=frame_index,
                capture_timestamp=capture,
                pixel=pixel,
                box=box,
                confidence=0.97,
                code=candidate.code,
                side_px=box[2] - box[0],
                raw_text=f"{candidate.code:02d}",
                ocr_confidence=0.96,
                mode="demo-cuadc",
            )
            visible.append(detection)
            if candidate.code == well.code:
                original[well.code] = detection
        frames.append(
            ProcessedFrame(
                frame_index=frame_index,
                received_timestamp=capture + 0.15,
                capture_timestamp=capture,
                processed_timestamp=now - 0.03,
                image=demo.render_scene(north, east),
                image_size=(demo.WIDTH, demo.HEIGHT),
                camera_matrix=tuple(
                    (float(row[0]), float(row[1]), float(row[2]))
                    for row in demo.camera().camera_matrix
                ),
                undistorted=True,
                pose=pose,
                detections=tuple(visible),
            )
        )
    frame = frames[-1]
    north, east = views[-1]
    # 频次表示多次观测，不参与中位选靶；第二轮 12/56/94 的中位数为 56。
    counts = {94: 19, 12: 37, 56: 29}
    observations = tuple(
        DisplayObservation(
            frame_index=original[well.code].frame_index,
            capture_timestamp=original[well.code].capture_timestamp,
            position_ned=(
                well.north_m + 0.18 * math.sin(sample * 2.3),
                well.east_m + 0.18 * math.cos(sample * 1.7),
                0.0,
            ),
            code=well.code,
            confidence=0.97,
            pixel=original[well.code].pixel,
            box=original[well.code].box,
            has_thumbnail=True,
        )
        for well in demo.WELLS[:3]
        for sample in range(counts[well.number])
    )
    camera_pose = (north, east, -demo.ALTITUDE_M)

    def project(point):
        return ned_to_pixel(
            point,
            camera=demo.camera(),
            position_ned=camera_pose,
            image_size=(demo.WIDTH, demo.HEIGHT),
            euler_deg=(0.0, 0.0, 0.0),
            distort=False,
        )

    dropping = mode == "drop"
    impact = (demo.SELECTED.north_m - 0.8, demo.SELECTED.east_m + 1.1, 0.0)
    return DisplaySnapshot(
        stage=stage,
        mode=mode,
        frame=frame,
        history_frames=tuple(frames),
        detections=frame.detections if mode in {"recon", "hold_follow"} else (),
        observations=observations,
        projected_observations=tuple(project(item.position_ned) for item in observations),
        panel=tuple(
            PanelEntry(
                well.number,
                counts[well.number],
                well.ned,
                ned_to_wgs84(well.ned, demo.ORIGIN_LLA),
                original[well.code].frame_index,
                original[well.code].box,
            )
            for well in sorted(demo.WELLS[:3], key=lambda item: -counts[item.number])
        ),
        recent_codes=(56, 12, 94),
        progress=ProgressSnapshot(
            lag_frames=3 if mode == "hold_follow" else 0,
            processing=mode == "hold_follow",
            queued_results=3 if mode == "hold_follow" else 0,
            ocr_pending=2 if mode == "hold_follow" else 0,
            processed_frames=4821 + index,
            ocr_completed=4702 + index,
            mission_current=14,
            mission_total=18,
            hold_elapsed_s=34.6 if stage == "HOLD_PROCESS" else None,
            hold_remaining_s=7.4 if stage == "HOLD_PROCESS" else None,
            eta_s=1.8 if mode == "hold_follow" else 0.0,
            processing_fps=12.4,
            ocr_rate_per_s=10.2,
            result_drain_rate_per_s=11.6,
            stage_elapsed_s=42.0 if stage == "RECON" else 18.3,
            telemetry_age_s=0.1,
        ),
        target_ned=demo.SELECTED.ned if dropping else None,
        predicted_impact_ned=impact if dropping else None,
        target_projected=project(demo.SELECTED.ned) if dropping else None,
        impact_projected=project(impact) if dropping else None,
        released=False,
        release_timestamp=None,
        release_simulated=True,
        release_command_status="pending"
        if phase_index == 3
        else "sent"
        if phase_index == 4
        else "not_sent",
        release_command_timestamp=now - 1.0 if dropping else None,
        events=(),
        geometry_error=None,
        capture_delay_s=0.25,
        observed_at=now,
        impact_error_m=math.hypot(0.8, 1.1) if dropping else None,
        impact_flight_time_s=1.75 if dropping else None,
        completed_stages=_STAGE_ORDER[: _STAGE_ORDER.index(stage)],
    )


def main(
    *,
    host: str = HOST,
    port: int = PORT,
    transport: str = TRANSPORT,
    fps: float = FPS,
    seconds: float = SECONDS,
    preview_path: Path | None = None,
    ffmpeg_executable: str | None = None,
) -> int:
    """发布合成画面；可选保存首帧预览，不连接飞控或加载模型。"""
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("seconds 必须是正的有限秒数")
    import cv2

    from airdrop.config import QgcVideoConfig
    from airdrop.qgcvideo.publisher import QgcVideoPublisher
    from airdrop.qgcvideo.render import FrameComposer

    config = QgcVideoConfig(
        enabled=True,
        host=host,
        port=port,
        transport=transport,
        fps=fps,
        ffmpeg_executable=ffmpeg_executable,
    ).validated()
    start = time.monotonic()
    index = 3

    def supplier():
        nonlocal index
        elapsed = min(seconds, max(0.0, time.monotonic() - start))
        result = _snapshot(index, elapsed, seconds)
        index += 1
        return result

    composer = FrameComposer()
    if preview_path is not None:
        # 预览固定取投放阶段，方便检查实时投影、编号缩略图和指令提示。
        image = composer.compose(_snapshot(index, seconds * 0.8, seconds))
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(preview_path), image):
            return 1

    publisher = QgcVideoPublisher(config, supplier, composer)
    publisher.start()
    if not publisher.running:
        return 1
    try:
        deadline = start + seconds
        while time.monotonic() < deadline:
            time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))
    finally:
        publisher.stop()
    return 0 if publisher.stats.encoded else 1
