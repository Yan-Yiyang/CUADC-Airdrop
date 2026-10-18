"""QGC 画面发布器与只读遥测、感知快照之间的适配。"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from airdrop.config import Config
    from airdrop.georef import CameraModel
    from airdrop.perception.models import ProcessedFrame
    from airdrop.perception.pipeline import PerceptionWorker
    from airdrop.qgcvideo.models import DisplaySnapshot
    from airdrop.qgcvideo.publisher import QgcVideoPublisher
    from airdrop.telemetry.broker import TelemetryBroker
    from airdrop.video.source import VideoFrame

LOGGER = logging.getLogger(__name__)


class _VideoSource(Protocol):
    """QGC 快照读取所需的最小视频源接口。"""

    def latest(self) -> VideoFrame | None: ...


class QgcVideoRuntime:
    """仅通过只读快照更新叠加层，不参与目标或任务判据计算。"""

    def __init__(
        self,
        config: Config,
        source: _VideoSource,
        broker: TelemetryBroker,
        camera: CameraModel | None,
        worker: PerceptionWorker | None,
        *,
        on_event: Callable[[str, dict[str, Any]], None] | None = None,
        simulated_release: bool = False,
        replay_mode: bool = False,
    ) -> None:
        from airdrop.qgcvideo.publisher import QgcVideoPublisher
        from airdrop.qgcvideo.state import DisplayState
        from airdrop.video.align import FrameTelemetryAligner

        self.config = config
        self.source = source
        self.broker = broker
        self.camera = camera
        self.worker = worker
        self.replay_mode = bool(replay_mode)
        self.publisher: QgcVideoPublisher
        self.aligner = FrameTelemetryAligner(
            broker,
            max_extrapolation=(
                0.5 if config.align.max_extrapolation is None else config.align.max_extrapolation
            ),
            max_wait=0.0,
            on_timeout="drop",
        )
        self.state = DisplayState(config, camera, simulated_release=simulated_release)
        self.last_snapshot: DisplaySnapshot | None = None
        self.publisher = QgcVideoPublisher(
            config.qgc_video,
            self._snapshot,
            on_event=on_event,
        )
        self._frame_observer = self.state.on_frame
        self._result_observer = self.state.on_detection
        if worker is not None:
            worker.add_frame_observer(self._frame_observer)
            worker.add_result_observer(self._result_observer)

    @property
    def running(self) -> bool:
        return self.publisher.running

    def start(self) -> object:
        return self.publisher.start()

    def stop(self, timeout: float | None = None) -> None:
        """结束发布并断开旁路观察者，保留最后画面快照。"""
        errors: list[BaseException] = []
        if self.worker is not None:
            try:
                self.worker.remove_frame_observer(self._frame_observer)
            except BaseException as exc:  # noqa: BLE001 - 即使中断也要尝试停止发布器
                errors.append(exc)
            try:
                self.worker.remove_result_observer(self._result_observer)
            except BaseException as exc:  # noqa: BLE001 - 即使中断也要尝试停止发布器
                errors.append(exc)
        try:
            self.last_snapshot = self._snapshot()
        except BaseException as exc:
            errors.append(exc)
            LOGGER.exception("保存 QGC 最终显示快照失败")
        try:
            self.publisher.stop(timeout=timeout)
        except BaseException as exc:
            errors.append(exc)
            LOGGER.exception("QGC 视频发布停止失败")
        if errors:
            raise BaseExceptionGroup(f"QGC 显示收尾有 {len(errors)} 项失败", errors)

    def on_event(self, kind: str, data: dict[str, Any]) -> None:
        try:
            self.state.on_event(kind, data)
        except Exception:
            LOGGER.exception("QGC 显示事件更新失败：%s", kind)

    def _snapshot(self) -> DisplaySnapshot:
        frame = self.source.latest()
        if frame is not None:
            try:
                live_frame = self._make_live_snapshot(frame.copy())
            except Exception:
                LOGGER.exception("QGC 实时帧快照创建失败")
                live_frame = None
        else:
            live_frame = None
        worker = self.worker
        try:
            telemetry = self.broker.get_snapshot()
            if worker is None:
                self.state.update_progress(
                    lag_frames=0,
                    processing=False,
                    ocr_pending=0,
                    queued_results=0,
                    mission_current=telemetry.mission_current,
                    mission_total=telemetry.mission_total,
                    telemetry_timestamp=telemetry.timestamp if telemetry.is_valid() else None,
                )
            else:
                stats = replace(worker.stats)
                self.state.update_progress(
                    lag_frames=int(stats.lag_frames),
                    processing=bool(worker.processing),
                    ocr_pending=max(
                        0,
                        int(stats.submitted) - int(stats.results) - int(stats.ocr_dropped),
                    ),
                    queued_results=int(worker.queued_results),
                    processed_frames=int(stats.frames),
                    ocr_completed=int(stats.results),
                    mission_current=telemetry.mission_current,
                    mission_total=telemetry.mission_total,
                    telemetry_timestamp=telemetry.timestamp if telemetry.is_valid() else None,
                )
        except Exception:
            LOGGER.exception("QGC 只读处理进度更新失败")
        result = self.state.snapshot(live_frame)
        if self.replay_mode:
            result = replace(
                result,
                capture_delay_s=None,
                observed_at=None if frame is None else frame.capture_timestamp,
                replay_mode=True,
                progress=replace(result.progress, telemetry_age_s=None),
            )
        self.last_snapshot = result
        return result

    def _make_live_snapshot(self, frame: VideoFrame) -> ProcessedFrame:
        from airdrop.perception.models import ProcessedFrame, readonly_image, snapshot_pose

        image = readonly_image(frame.image)
        aligned = self.aligner.align(frame)
        pose = snapshot_pose(None if aligned is None else aligned.snapshot)
        camera_matrix: tuple[tuple[float, float, float], ...] | None = None
        if self.camera is not None:
            matrix = self.camera.camera_matrix
            camera_matrix = (
                (float(matrix[0, 0]), float(matrix[0, 1]), float(matrix[0, 2])),
                (float(matrix[1, 0]), float(matrix[1, 1]), float(matrix[1, 2])),
                (float(matrix[2, 0]), float(matrix[2, 1]), float(matrix[2, 2])),
            )
        return ProcessedFrame(
            frame_index=frame.index,
            received_timestamp=frame.timestamp,
            capture_timestamp=frame.capture_timestamp,
            processed_timestamp=time.time(),
            image=image,
            image_size=(frame.width, frame.height),
            camera_matrix=camera_matrix,
            undistorted=False,
            pose=pose,
            detections=(),
        )
