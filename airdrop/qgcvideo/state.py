"""QGC 发布器只读的有界业务显示状态。"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import OrderedDict, deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import cv2
import numpy as np

from ..georef.camera import CameraModel
from ..georef.geo import ned_to_wgs84
from ..georef.project import ProjectedPoint, ned_to_pixel, ned_to_pixels, pixel_to_ned
from ..perception.models import (
    Detection,
    DetectionSnapshot,
    ProcessedFrame,
    snapshot_detection,
)
from .models import (
    DisplayEvent,
    DisplayObservation,
    DisplaySnapshot,
    PanelEntry,
    ProgressSnapshot,
)

if TYPE_CHECKING:
    from ..config import Config

LOGGER = logging.getLogger(__name__)

__all__ = ["DisplayState"]


@dataclass(frozen=True, slots=True)
class _ObservationRecord:
    serial: int
    detection: DetectionSnapshot
    has_thumbnail: bool


class DisplayState:
    """用短锁维护显示自有的有限历史，不读取或推进任务统计器。"""

    def __init__(
        self,
        config: Config,
        camera: CameraModel | None,
        *,
        clock: Callable[[], float] = time.time,
        simulated_release: bool = False,
    ) -> None:
        self._config = config
        self._camera = camera
        self._clock = clock
        self._simulated_release = bool(simulated_release)
        video = config.qgc_video
        self._history_max = int(video.history_frames)
        self._cloud_max = int(video.cloud_max)
        self._history: OrderedDict[int, ProcessedFrame] = OrderedDict()
        self._observations: deque[_ObservationRecord] = deque(maxlen=self._cloud_max)
        self._geometry_cache: dict[int, tuple[float, float, float] | None] = {}
        self._next_observation_serial = 0
        self._events: deque[DisplayEvent] = deque(maxlen=100)
        self._progress_samples: deque[tuple[float, int, int, int]] = deque(maxlen=256)
        self._lock = threading.Lock()
        self._stage: str | None = None
        self._completed_stages: set[str] = set()
        self._stage_started_at: float | None = None
        self._hold_started_at: float | None = None
        self._progress = ProgressSnapshot()
        self._target_ned: tuple[float, float, float] | None = None
        self._predicted_impact_ned: tuple[float, float, float] | None = None
        self._impact_error_m: float | None = None
        self._impact_flight_time_s: float | None = None
        self._release_command_status = "not_sent"
        self._release_command_timestamp: float | None = None
        self._released = False
        self._release_timestamp: float | None = None
        self._release_simulated = bool(simulated_release)
        self._geometry_error: str | None = None

    def on_frame(self, frame: ProcessedFrame) -> None:
        """收下处理帧，包括零检测帧；只保存有界历史。"""
        with self._lock:
            existing = self._history.get(frame.frame_index)
            if existing is not None and existing.capture_timestamp != frame.capture_timestamp:
                self._geometry_error = "同一帧号对应不同拍摄时刻"
                return
            self._history[frame.frame_index] = frame
            self._history.move_to_end(frame.frame_index)
            while len(self._history) > self._history_max:
                self._history.popitem(last=False)

    def on_detection(self, detection: Detection | DetectionSnapshot) -> None:
        """旁路接收一条结果；只用对应帧快照解算，不触碰业务 tracker。"""
        item = snapshot_detection(detection) if isinstance(detection, Detection) else detection
        with self._lock:
            frame = self._history.get(item.frame_index)
            matched = frame is not None and math.isclose(
                frame.capture_timestamp, item.capture_timestamp, rel_tol=0.0, abs_tol=1e-6
            )
            if matched and frame is not None:
                current = list(frame.detections)
                key = (item.pixel, item.box)
                current = [value for value in current if (value.pixel, value.box) != key]
                current.append(item)
                self._history[item.frame_index] = replace(frame, detections=tuple(current))
            else:
                self._geometry_error = "结果没有匹配的在存历史帧"

            serial = self._next_observation_serial
            self._next_observation_serial += 1
            self._observations.append(_ObservationRecord(serial, item, bool(matched)))

    def on_event(self, kind: str, data: Mapping[str, object]) -> None:
        """旁路接收事件；只复制必要标量，不保留事件字典引用。"""
        now = float(self._clock())
        frozen_data = tuple(sorted((str(key), _freeze(value)) for key, value in data.items()))
        event = DisplayEvent(str(kind), now, frozen_data)
        with self._lock:
            self._events.append(event)
            values = dict(frozen_data)
            if kind == "state":
                raw = values.get("to_state", values.get("to", values.get("state")))
                next_stage = _as_stage(raw)
                previous = self._stage or _as_stage(values.get("from_state"))
                if previous is not None and next_stage not in {None, "ABORT", previous}:
                    self._completed_stages.add(previous)
                self._stage = next_stage
                stamp = values.get("timestamp")
                event_time = float(stamp) if isinstance(stamp, (int, float)) else now
                self._stage_started_at = event_time
                self._hold_started_at = event_time if self._stage == "HOLD_PROCESS" else None
            elif kind == "drop_plan":
                self._target_ned = _triple(values.get("target_ned"))
            elif kind in {"drop_check", "release"}:
                prediction = values.get(
                    "predicted_impact_ned",
                    values.get("impact_ned", values.get("predicted_ned")),
                )
                self._predicted_impact_ned = _triple(prediction)
                self._impact_error_m = _finite_nonnegative(values.get("horizontal_error_m"))
                self._impact_flight_time_s = _finite_nonnegative(values.get("flight_time_s"))
            elif kind in {"gripper_release", "drop_dry_run"}:
                self._release_command_status = "sent"
                self._release_command_timestamp = now
                self._release_simulated = kind == "drop_dry_run"
            elif kind == "drop_released":
                self._released = True
                timestamp = values.get("timestamp")
                self._release_timestamp = (
                    float(timestamp) if isinstance(timestamp, (float, int)) else now
                )
                self._release_simulated = bool(values.get("simulated", self._simulated_release))
            elif kind in {
                "drop_command_pending",
                "drop_command_sent",
                "drop_command_failed",
                "drop_command_unknown",
            }:
                self._release_simulated = bool(values.get("simulated", self._simulated_release))
                status = {
                    "drop_command_pending": "pending",
                    "drop_command_sent": "sent",
                    "drop_command_failed": "failed",
                    "drop_command_unknown": "unknown",
                }[kind]
                # 控制器成功事件发生在同步等待返回前，保留其较早的确认时刻。
                if not (
                    kind == "drop_command_sent"
                    and self._release_command_status == "sent"
                    and self._release_command_timestamp is not None
                ):
                    self._release_command_status = status
                    timestamp = values.get("timestamp")
                    self._release_command_timestamp = (
                        float(timestamp) if isinstance(timestamp, (float, int)) else now
                    )

    def update_progress(
        self,
        *,
        lag_frames: int,
        processing: bool,
        ocr_pending: int,
        queued_results: int,
        processed_frames: int | None = None,
        ocr_completed: int | None = None,
        mission_current: int | None = None,
        mission_total: int | None = None,
        telemetry_timestamp: float | None = None,
    ) -> None:
        """更新任务线程发布的完成进度；此处不读取 worker 或 tracker。"""
        progress = ProgressSnapshot(
            lag_frames=max(int(lag_frames), 0),
            processing=bool(processing),
            ocr_pending=max(int(ocr_pending), 0),
            queued_results=max(int(queued_results), 0),
            processed_frames=_nonnegative_optional(processed_frames),
            ocr_completed=_nonnegative_optional(ocr_completed),
            mission_current=_nonnegative_optional(mission_current),
            mission_total=_nonnegative_optional(mission_total),
        )
        now = float(self._clock())
        if telemetry_timestamp is not None and math.isfinite(telemetry_timestamp):
            progress = replace(progress, telemetry_age_s=max(now - telemetry_timestamp, 0.0))
        with self._lock:
            if progress.processed_frames is not None and progress.ocr_completed is not None:
                self._progress_samples.append(
                    (
                        now,
                        progress.processed_frames,
                        progress.ocr_completed,
                        progress.queued_results,
                    )
                )
                cutoff = now - float(self._config.qgc_video.eta_window_s)
                while len(self._progress_samples) > 1 and self._progress_samples[0][0] < cutoff:
                    self._progress_samples.popleft()
            progress = self._estimate_progress(progress, now)
            self._progress = progress

    def _estimate_progress(self, progress: ProgressSnapshot, now: float) -> ProgressSnapshot:
        hold_elapsed = None
        hold_remaining = None
        if self._stage == "HOLD_PROCESS" and self._hold_started_at is not None:
            hold_elapsed = max(now - self._hold_started_at, 0.0)
            hold_remaining = max(float(self._config.mission.hold_process_max_s) - hold_elapsed, 0.0)
        first = self._progress_samples[0] if self._progress_samples else None
        last = self._progress_samples[-1] if self._progress_samples else None
        processing_fps = None
        ocr_rate = None
        result_rate = None
        if first is not None and last is not None and last[0] > first[0]:
            duration = last[0] - first[0]
            if last[1] >= first[1]:
                processing_fps = _positive_rate(last[1] - first[1], duration)
            if last[2] >= first[2]:
                ocr_rate = _positive_rate(last[2] - first[2], duration)
            if first[3] >= last[3]:
                result_rate = _positive_rate(first[3] - last[3], duration)
        eta = _estimate_eta(progress, processing_fps, ocr_rate, result_rate)
        return replace(
            progress,
            hold_elapsed_s=hold_elapsed,
            hold_remaining_s=hold_remaining,
            processing_fps=processing_fps,
            ocr_rate_per_s=ocr_rate,
            result_drain_rate_per_s=result_rate,
            eta_s=eta,
            stage_elapsed_s=(
                None if self._stage_started_at is None else max(now - self._stage_started_at, 0.0)
            ),
        )

    def snapshot(self, frame: ProcessedFrame | None = None) -> DisplaySnapshot:
        """复制有界状态并按当前帧姿态投影；锁内不做几何计算。"""
        with self._lock:
            history = tuple(self._history.values())
            records = tuple(self._observations)
            geometry_cache = dict(self._geometry_cache)
            progress = self._progress
            stage = self._stage
            completed_stages = tuple(sorted(self._completed_stages))
            events = tuple(self._events)
            target = self._target_ned
            impact = self._predicted_impact_ned
            impact_error = self._impact_error_m
            impact_flight_time = self._impact_flight_time_s
            released = self._released
            release_timestamp = self._release_timestamp
            release_simulated = self._release_simulated
            release_command_status = self._release_command_status
            release_command_timestamp = self._release_command_timestamp
            geometry_error = self._geometry_error

        mode = _display_mode(stage, progress)
        snapshot_time = float(self._clock())
        if mode in {"hold_live", "drop"} and frame is not None:
            selected = replace(frame, detections=())
        elif history:
            selected = history[-1]
        else:
            selected = None if frame is None else replace(frame, detections=())
        detections = () if selected is None else selected.detections

        frame_by_index = {item.frame_index: item for item in history}
        observations_list: list[DisplayObservation] = []
        new_geometry: dict[int, tuple[float, float, float] | None] = {}
        for record in records:
            detection = record.detection
            source = frame_by_index.get(detection.frame_index)
            if source is not None and not math.isclose(
                source.capture_timestamp, detection.capture_timestamp, rel_tol=0.0, abs_tol=1e-6
            ):
                source = None
            point = geometry_cache.get(record.serial)
            if record.serial not in geometry_cache:
                point = self._solve_detection(detection, source)
                new_geometry[record.serial] = point
            observations_list.append(
                DisplayObservation(
                    frame_index=detection.frame_index,
                    capture_timestamp=detection.capture_timestamp,
                    position_ned=point,
                    code=detection.code,
                    confidence=detection.confidence,
                    pixel=detection.pixel,
                    box=detection.box,
                    has_thumbnail=bool(record.has_thumbnail and source is not None),
                )
            )
        observations = tuple(observations_list)
        if new_geometry:
            with self._lock:
                active = {record.serial for record in self._observations}
                self._geometry_cache.update(
                    {key: value for key, value in new_geometry.items() if key in active}
                )
                self._geometry_cache = {
                    key: value for key, value in self._geometry_cache.items() if key in active
                }
        counts, sums, latest = _summarize(observations)
        projected: tuple[ProjectedPoint, ...] = ()
        target_projection = None
        impact_projection = None
        panel = self._panel(counts, sums, latest, observations, selected, history)
        if selected is not None:
            projected = self._project_observations(observations, selected)
            target_projection = self._project_ned(target, selected)
            impact_projection = self._project_ned(impact, selected)

        return DisplaySnapshot(
            stage=stage,
            mode=mode,
            frame=selected,
            history_frames=history,
            detections=detections,
            observations=observations,
            projected_observations=projected,
            panel=panel,
            recent_codes=tuple(
                code
                for code, _ in sorted(latest.items(), key=lambda pair: pair[1], reverse=True)[:3]
            ),
            progress=progress,
            target_ned=target,
            predicted_impact_ned=impact,
            target_projected=target_projection,
            impact_projected=impact_projection,
            release_command_status=release_command_status,
            release_command_timestamp=release_command_timestamp,
            released=released,
            release_timestamp=release_timestamp,
            release_simulated=release_simulated,
            events=events,
            geometry_error=geometry_error,
            capture_delay_s=(
                None
                if selected is None or not math.isfinite(snapshot_time)
                else max(snapshot_time - selected.capture_timestamp, 0.0)
            ),
            observed_at=snapshot_time,
            impact_error_m=impact_error,
            impact_flight_time_s=impact_flight_time,
            completed_stages=completed_stages,
        )

    def _solve_detection(
        self, detection: DetectionSnapshot, frame: ProcessedFrame | None
    ) -> tuple[float, float, float] | None:
        if self._camera is None:
            return None
        pose = frame.pose if frame is not None else detection.pose
        if pose is None:
            return None
        if pose.position_ned is None or (pose.quaternion_wxyz is None and pose.euler_deg is None):
            return None
        origin = pose.origin_lla
        ground_z = self._config.ground.ground_z(None if origin is None else origin[2])
        if ground_z is None:
            return None
        try:
            camera = self._camera if frame is None else self._frame_camera(frame)
            result = pixel_to_ned(
                detection.pixel,
                camera=camera,
                ground_z=ground_z,
                position_ned=pose.position_ned,
                quaternion=pose.quaternion_wxyz,
                euler_deg=pose.euler_deg,
                undistort=not (frame.undistorted if frame is not None else detection.undistorted),
            )
        except (ValueError, TypeError, np.linalg.LinAlgError) as exc:
            LOGGER.debug("显示观测坐标解算失败：%s", exc)
            return None
        return result.ned if result.ok else None

    def _project_ned(
        self, point: tuple[float, float, float] | None, frame: ProcessedFrame
    ) -> ProjectedPoint:
        pose = frame.pose
        if point is None or self._camera is None or pose is None or pose.position_ned is None:
            return ProjectedPoint(point, None, False, False, "缺少相机或位姿")
        if pose.quaternion_wxyz is None and pose.euler_deg is None:
            return ProjectedPoint(point, None, False, False, "缺少姿态")
        try:
            return ned_to_pixel(
                point,
                camera=self._frame_camera(frame),
                position_ned=pose.position_ned,
                image_size=frame.image_size,
                quaternion=pose.quaternion_wxyz,
                euler_deg=pose.euler_deg,
                distort=not frame.undistorted,
                camera_matrix=frame.camera_matrix,
            )
        except (ValueError, TypeError, np.linalg.LinAlgError, cv2.error) as exc:
            return ProjectedPoint(point, None, False, False, str(exc))

    def _project_observations(
        self, observations: tuple[DisplayObservation, ...], frame: ProcessedFrame
    ) -> tuple[ProjectedPoint, ...]:
        """一次准备当前帧相机与姿态，批量投影整片显示观测云。"""
        pose = frame.pose
        if self._camera is None or pose is None or pose.position_ned is None:
            return tuple(
                ProjectedPoint(item.position_ned, None, False, False, "缺少相机或位姿")
                for item in observations
            )
        if pose.quaternion_wxyz is None and pose.euler_deg is None:
            return tuple(
                ProjectedPoint(item.position_ned, None, False, False, "缺少姿态")
                for item in observations
            )
        try:
            camera = self._frame_camera(frame)
            return ned_to_pixels(
                tuple(item.position_ned for item in observations),
                camera=camera,
                position_ned=pose.position_ned,
                image_size=frame.image_size,
                quaternion=pose.quaternion_wxyz,
                euler_deg=pose.euler_deg,
                distort=not frame.undistorted,
                camera_matrix=frame.camera_matrix,
            )
        except (ValueError, TypeError, np.linalg.LinAlgError, cv2.error) as exc:
            return tuple(
                ProjectedPoint(item.position_ned, None, False, False, str(exc))
                for item in observations
            )

    def _frame_camera(self, frame: ProcessedFrame) -> CameraModel:
        camera = self._camera
        if camera is None:
            raise ValueError("未配置相机模型")
        if camera.image_size is not None and camera.image_size != frame.image_size:
            raise ValueError(f"帧尺寸 {frame.image_size} 与相机标定尺寸 {camera.image_size} 不符")
        if frame.camera_matrix is None:
            return camera
        matrix = np.asarray(frame.camera_matrix, dtype=np.float64)
        if np.allclose(matrix, camera.camera_matrix, rtol=0.0, atol=1e-9):
            return camera
        return replace(camera, camera_matrix=matrix)

    def _panel(
        self,
        counts: dict[int, int],
        sums: dict[int, tuple[float, float, float, int]],
        latest: dict[int, tuple[float, int]],
        observations: tuple[DisplayObservation, ...],
        frame: ProcessedFrame | None,
        history: tuple[ProcessedFrame, ...],
    ) -> tuple[PanelEntry, ...]:
        if frame is None:
            return ()
        reference = None if frame.pose is None else frame.pose.origin_lla
        frame_indices = {item.frame_index for item in history}
        entries: list[PanelEntry] = []
        for code, count in sorted(counts.items(), key=lambda pair: (-pair[1], pair[0])):
            total = sums.get(code)
            position = None
            lla = None
            if total is not None and total[3] > 0:
                n, e, d, samples = total
                position = (n / samples, e / samples, d / samples)
                if reference is not None:
                    try:
                        lla = ned_to_wgs84(position, reference)
                    except ValueError, OverflowError:
                        lla = None
            matching = [item for item in observations if item.code == code]
            latest_item = (
                max(matching, key=lambda item: (item.capture_timestamp, item.frame_index))
                if matching
                else None
            )
            thumbnail_index = None
            thumbnail_box = None
            if (
                latest_item is not None
                and latest_item.has_thumbnail
                and latest_item.frame_index in frame_indices
            ):
                thumbnail_index = latest_item.frame_index
                thumbnail_box = latest_item.box
            entries.append(
                PanelEntry(
                    code,
                    count,
                    position,
                    lla,
                    thumbnail_index,
                    thumbnail_box,
                )
            )
        return tuple(entries)


def _display_mode(stage: str | None, progress: ProgressSnapshot) -> str:
    if stage == "HOLD_PROCESS":
        busy = (
            progress.lag_frames > 0
            or progress.processing
            or progress.ocr_pending > 0
            or progress.queued_results > 0
        )
        return "hold_follow" if busy else "hold_live"
    if stage in {"OVERFLY", "LAND"}:
        return "drop"
    return "recon"


def _as_stage(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    if "." in text:
        text = text.rsplit(".", maxsplit=1)[-1]
    return text.upper()


def _triple(value: object) -> tuple[float, float, float] | None:
    if not isinstance(value, (tuple, list)) or len(value) != 3:
        return None
    try:
        result = tuple(float(item) for item in value)
    except TypeError, ValueError:
        return None
    if not all(math.isfinite(item) for item in result):
        return None
    return result  # type: ignore[return-value]


def _freeze(value: object) -> object:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return tuple(sorted((str(key), _freeze(item)) for key, item in value.items()))
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    return str(value)


def _nonnegative_optional(value: int | None) -> int | None:
    return None if value is None else max(int(value), 0)


def _finite_nonnegative(value: object) -> float | None:
    if isinstance(value, (int, float)) and math.isfinite(value) and value >= 0:
        return float(value)
    return None


def _positive_rate(delta: int, duration: float) -> float | None:
    if delta <= 0 or duration <= 0 or not math.isfinite(duration):
        return None
    return delta / duration


def _estimate_eta(
    progress: ProgressSnapshot,
    processing_fps: float | None,
    ocr_rate: float | None,
    result_rate: float | None,
) -> float | None:
    pending_rates = (
        (progress.lag_frames + int(progress.processing), processing_fps),
        (progress.ocr_pending, ocr_rate),
        (progress.queued_results, result_rate),
    )
    estimates: list[float] = []
    for count, rate in pending_rates:
        if count <= 0:
            continue
        if rate is None or rate <= 0:
            return None
        estimates.append(count / rate)
    return max(estimates, default=0.0)


def _summarize(
    observations: tuple[DisplayObservation, ...],
) -> tuple[
    dict[int, int],
    dict[int, tuple[float, float, float, int]],
    dict[int, tuple[float, int]],
]:
    counts: dict[int, int] = {}
    sums: dict[int, tuple[float, float, float, int]] = {}
    latest: dict[int, tuple[float, int]] = {}
    for item in observations:
        if item.code is None:
            continue
        counts[item.code] = counts.get(item.code, 0) + 1
        latest[item.code] = max(
            latest.get(item.code, (item.capture_timestamp, item.frame_index)),
            (item.capture_timestamp, item.frame_index),
        )
        if item.position_ned is not None:
            n, e, d, number = sums.get(item.code, (0.0, 0.0, 0.0, 0))
            sums[item.code] = (
                n + item.position_ned[0],
                e + item.position_ned[1],
                d + item.position_ned[2],
                number + 1,
            )
    return counts, sums, latest
