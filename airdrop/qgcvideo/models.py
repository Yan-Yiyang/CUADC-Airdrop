"""QGC 显示状态使用的不可变数据传输对象。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..georef.project import ProjectedPoint
from ..perception.models import DetectionSnapshot, ProcessedFrame
from .prediction import LiveImpactPrediction

__all__ = [
    "DisplayEvent",
    "DisplayObservation",
    "DisplaySnapshot",
    "PanelEntry",
    "ProgressSnapshot",
]


@dataclass(frozen=True, slots=True)
class ProgressSnapshot:
    """业务线程发布的排空进度。"""

    lag_frames: int = 0
    processing: bool = False
    ocr_pending: int = 0
    queued_results: int = 0
    processed_frames: int | None = None
    ocr_completed: int | None = None
    mission_current: int | None = None
    mission_total: int | None = None
    hold_elapsed_s: float | None = None
    hold_remaining_s: float | None = None
    processing_fps: float | None = None
    ocr_rate_per_s: float | None = None
    result_drain_rate_per_s: float | None = None
    eta_s: float | None = None
    stage_elapsed_s: float | None = None
    telemetry_age_s: float | None = None


@dataclass(frozen=True, slots=True)
class DisplayObservation:
    """显示自持的坐标观测，不引用业务 tracker 的点或缓存。"""

    frame_index: int
    capture_timestamp: float
    position_ned: tuple[float, float, float] | None
    code: int | None
    confidence: float
    pixel: tuple[float, float]
    box: tuple[float, float, float, float]
    has_thumbnail: bool


@dataclass(frozen=True, slots=True)
class PanelEntry:
    """显示云窗口内一种编号的观测均值，不等同任务聚类统计。"""

    code: int
    count: int
    position_ned: tuple[float, float, float] | None
    position_lla: tuple[float, float, float] | None
    thumbnail_frame_index: int | None
    thumbnail_box: tuple[float, float, float, float] | None
    thumbnail: np.ndarray | None = None
    thumbnail_capture_timestamp: float | None = None


@dataclass(frozen=True, slots=True)
class DisplayEvent:
    """已复制并递归收窄为不可变值的业务事件。"""

    kind: str
    timestamp: float
    data: tuple[tuple[str, object], ...]


@dataclass(frozen=True, slots=True)
class DisplaySnapshot:
    """一次渲染所需的只读一致性快照。"""

    stage: str | None
    mode: str
    frame: ProcessedFrame | None
    history_frames: tuple[ProcessedFrame, ...]
    detections: tuple[DetectionSnapshot, ...]
    observations: tuple[DisplayObservation, ...]
    projected_observations: tuple[ProjectedPoint, ...]
    panel: tuple[PanelEntry, ...]
    recent_codes: tuple[int, ...]
    progress: ProgressSnapshot
    target_ned: tuple[float, float, float] | None
    predicted_impact_ned: tuple[float, float, float] | None
    target_projected: ProjectedPoint | None
    impact_projected: ProjectedPoint | None
    release_command_status: str
    release_command_timestamp: float | None
    released: bool
    release_timestamp: float | None
    release_simulated: bool
    events: tuple[DisplayEvent, ...]
    geometry_error: str | None
    capture_delay_s: float | None
    observed_at: float | None = None
    impact_error_m: float | None = None
    impact_flight_time_s: float | None = None
    replay_mode: bool = False
    completed_stages: tuple[str, ...] = ()
    impact_prediction_timestamp: float | None = None
    flypast_completed: bool = False
    live_prediction: LiveImpactPrediction | None = None
    release_impact_ned: tuple[float, float, float] | None = None
    release_error_m: float | None = None
    release_flight_time_s: float | None = None
    release_prediction_timestamp: float | None = None
