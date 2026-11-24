"""real、SITL 与 replay 共用的帧对齐和感知装配。"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
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
    from airdrop.video.segments import SegmentLifecycle

LOGGER = logging.getLogger(__name__)


def build_alignment_buffer(
    *,
    capture: bool,
    ts_directory: Path = Path("."),
    segment_lifecycle: SegmentLifecycle | None = None,
    cold_storage: bool = True,
) -> AlignmentBuffer | None:
    """建立在线与回放共用的原始帧缓冲；关闭冷存储时按消费者游标即时回收内存。"""
    if not capture:
        return None
    from airdrop import AlignmentBuffer, capacity_for

    if not cold_storage:
        return AlignmentBuffer(
            capacity_for(30.0, 180.0),
            storage="raw",
            max_bytes=128 * 1024 * 1024,
            evict_consumed=True,
        )

    from airdrop.video.cold import TsRangeDecoder

    decoder = TsRangeDecoder(
        ts_directory,
        max_cache_bytes=64 * 1024 * 1024,
        segment_lifecycle=segment_lifecycle,
    )
    try:
        return AlignmentBuffer(
            capacity_for(30.0, 180.0),
            storage="raw",
            cold_decoder=decoder,
            hot_seconds=1.0,
            max_hot_bytes=128 * 1024 * 1024,
            segment_lifecycle=segment_lifecycle,
        )
    except BaseException as exc:
        try:
            decoder.close()
        except BaseException as cleanup_error:  # noqa: BLE001 - 回滚必须保留构造原始异常
            exc.add_note(f"历史解码器清理失败: {cleanup_error}")
        raise


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


def prepare_perception_components(
    components: PerceptionComponents,
    *,
    output_size: tuple[int, int],
    opencv_threads: int,
) -> None:
    """校验相机尺寸并预热感知模型，供在线与回放共用。"""
    worker = components.worker
    if worker is None:
        return
    camera = components.camera
    if camera is not None and camera.image_size is not None:
        calib_size = tuple(int(value) for value in camera.image_size)
        if calib_size != output_size:
            raise RuntimeError(f"相机标定尺寸 {calib_size} 与视频输出尺寸 {output_size} 不一致")
    configure_opencv_threads(worker, opencv_threads)
    if not worker.detector.warmup(*output_size):
        raise RuntimeError(f"YOLO 按视频输出尺寸 {output_size} 预热失败")


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
    calibration_capture: bool = False,
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
        capture_only=calibration_capture,
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
