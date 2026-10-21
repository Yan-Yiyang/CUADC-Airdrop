"""real、SITL 与 replay 共用的帧对齐和感知装配。"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from airdrop.config import Config
    from airdrop.georef import CameraModel
    from airdrop.mission.targets import PerceptionTargetSource, TargetTracker
    from airdrop.perception.models import Detection
    from airdrop.perception.pipeline import PerceptionWorker
    from airdrop.record.recorder import FlightRecorder
    from airdrop.telemetry.broker import TelemetryBroker
    from airdrop.video.buffer import AlignmentBuffer, AlignmentWriter

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class PerceptionComponents:
    """由阶段开关决定创建的相机、worker 和目标跟踪组件。"""

    writer: AlignmentWriter | None = None
    camera: CameraModel | None = None
    worker: PerceptionWorker | None = None
    tracker: TargetTracker | None = None
    targets: PerceptionTargetSource | None = None


def configure_opencv_threads(worker: PerceptionWorker, threads: int) -> None:
    """加载 YOLO 后设置主进程 OpenCV 线程数。"""
    worker.detector.load()
    import cv2

    cv2.setNumThreads(threads)
    LOGGER.info(
        "OpenCV 主进程线程数：请求 %d，实际 %d",
        threads,
        cv2.getNumThreads(),
    )


def build_perception_components(
    config: Config,
    broker: TelemetryBroker,
    buffer: AlignmentBuffer,
    *,
    capture: bool,
    recognize: bool,
    recorder: FlightRecorder | None = None,
    on_event: Callable[[str, dict[str, Any]], None] | None = None,
    side_check_tolerance: float | None = None,
    defer_until_airborne: bool = False,
) -> PerceptionComponents:
    """按需构造共享感知链路；关闭识别时完全不加载相机或模型。"""
    if recognize and not capture:
        raise ValueError("启用识别必须接入采集")
    if not capture:
        return PerceptionComponents()

    from airdrop import AlignmentWriter, FrameTelemetryAligner, PerceptionTargetSource
    from airdrop.mission.targets import DEFAULT_SIDE_TOLERANCE, TargetTracker
    from airdrop.perception.pipeline import PerceptionWorker

    writer = AlignmentWriter(
        buffer,
        FrameTelemetryAligner(
            broker,
            max_wait=config.align.max_wait,
            mode=config.align.mode,
            on_timeout=config.align.on_timeout,
            max_extrapolation=config.align.max_extrapolation,
        ),
        defer_until_airborne=defer_until_airborne,
    )
    if not recognize:
        return PerceptionComponents(writer=writer)

    camera = config.camera.load_model()
    worker = PerceptionWorker(config.perception.to_pipeline_config(camera), buffer=buffer)
    tracker = TargetTracker(
        config,
        camera=camera,
        on_event=on_event,
        side_check_tolerance=(
            DEFAULT_SIDE_TOLERANCE if side_check_tolerance is None else float(side_check_tolerance)
        ),
    )

    def record_detection(detection: Detection) -> None:
        if recorder is not None:
            recorder.detections.append(detection)

    targets = PerceptionTargetSource(
        worker=worker,
        tracker=tracker,
        on_detection=record_detection if recorder is not None else None,
    )
    return PerceptionComponents(writer, camera, worker, tracker, targets)
