"""感知结果的数据模型。

``Detection`` 是**感知链路的唯一产物**：一条"某一帧、某个像素位置上有个编号
为 X 的目标"的记录。它同时带上：

* 帧身份（``frame_index``）与画面拍摄时刻（``capture_timestamp``）——坐标解算
  必须用拍摄时刻的遥测，不能用"处理时刻"；
* 像素位置（``pixel``）——georef 的输入；
* 编号（``code``）与置信度——targeting 聚类的依据；
* 该时刻的遥测快照（``telemetry``）——坐标解算与记录都要。

字段设计对齐 OCR 链路的输出约定：编号来自 OCR 或类别直出，置信度由检测器给出。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np

from ..telemetry.models import TelemetrySnapshot

__all__ = [
    "Detection",
    "DetectionSnapshot",
    "PixelBox",
    "PoseSnapshot",
    "ProcessedFrame",
    "readonly_image",
    "snapshot_detection",
    "snapshot_pose",
]


@dataclass(frozen=True, slots=True)
class PixelBox:
    """像素坐标的检测框（左上/右下），``x1<=x2``、``y1<=y2``。"""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def center(self) -> tuple[float, float]:
        return (self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0

    @property
    def area(self) -> float:
        return max(self.width, 0.0) * max(self.height, 0.0)

    def expanded(self, ratio: float) -> "PixelBox":
        """按比例向外扩（裁剪时保留目标完整轮廓，见 4.3 的"外扩 0.2"）。"""
        dx = self.width * ratio / 2.0
        dy = self.height * ratio / 2.0
        return PixelBox(self.x1 - dx, self.y1 - dy, self.x2 + dx, self.y2 + dy)

    def clipped(self, width: int, height: int) -> "PixelBox":
        """裁剪到画面范围内。"""
        x1 = min(max(self.x1, 0.0), max(width - 1.0, 0.0))
        y1 = min(max(self.y1, 0.0), max(height - 1.0, 0.0))
        x2 = min(max(self.x2, x1), max(width - 1.0, 0.0))
        y2 = min(max(self.y2, y1), max(height - 1.0, 0.0))
        return PixelBox(x1, y1, x2, y2)

    def as_dict(self) -> dict[str, float]:
        return {"x1": self.x1, "y1": self.y1, "x2": self.x2, "y2": self.y2}


@dataclass(frozen=True, slots=True)
class Detection:
    """一条感知结果。

    ``code`` 为 None 表示**检测到了目标但没读出编号**（OCR 失败）——这种记录
    对坐标解算仍有价值（画面里确实有东西），但不能参与按编号筛选的统计
    （见 ``TargetingConfig.require_label``）。
    """

    frame_index: int
    capture_timestamp: float
    pixel: tuple[float, float]
    box: PixelBox
    confidence: float
    telemetry: TelemetrySnapshot
    code: int | None = None
    side_px: float = 0.0
    raw_text: str = ""
    ocr_confidence: float = 0.0
    mode: str = ""
    #: 检测器提供 undistorted 和 image_size（宽、高），供下游核对像素模型。
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def has_code(self) -> bool:
        return self.code is not None

    def as_dict(self) -> dict[str, Any]:
        """落盘用（``detections.jsonl``）：遥测只留关键字段，避免每行几百字节。"""
        tlm = self.telemetry
        return {
            "frame_index": self.frame_index,
            "capture_timestamp": self.capture_timestamp,
            "pixel": list(self.pixel),
            "box": self.box.as_dict(),
            "confidence": self.confidence,
            "code": self.code,
            "side_px": self.side_px,
            "raw_text": self.raw_text,
            "ocr_confidence": self.ocr_confidence,
            "mode": self.mode,
            "telemetry": {
                "timestamp": tlm.timestamp,
                "latitude_deg": tlm.latitude_deg,
                "longitude_deg": tlm.longitude_deg,
                "absolute_altitude_m": tlm.absolute_altitude_m,
                "north_m": tlm.north_m,
                "east_m": tlm.east_m,
                "down_m": tlm.down_m,
                "roll_deg": tlm.roll_deg,
                "pitch_deg": tlm.pitch_deg,
                "yaw_deg": tlm.yaw_deg,
                # 速度：**姿态 yaw 的独立参照**。离线诊断
                # ``runtime/sitl_analysis.py`` 的 ``_heading_consistency`` 用
                # ``atan2(vy, vx)`` 的航迹角和 yaw 比——PX4 的 yaw 在起飞后有一段
                # 瞬态能偏 25~35°（实测 6 个架次），落点误差会被它放大，而那不是
                # 坐标解算的锅。少这两个字段这条诊断就只能靠活的 broker，离线复算不了。
                "vx_m_s": tlm.vx_m_s,
                "vy_m_s": tlm.vy_m_s,
            },
            **({"extra": dict(self.extra)} if self.extra else {}),
        }


@dataclass(frozen=True, slots=True)
class PoseSnapshot:
    """显示线程使用的纯标量位姿；origin_lla 顺序为经度、纬度、海拔。"""

    position_ned: tuple[float, float, float] | None
    quaternion_wxyz: tuple[float, float, float, float] | None
    euler_deg: tuple[float, float, float] | None
    origin_lla: tuple[float, float, float] | None

    @classmethod
    def from_telemetry(cls, telemetry: TelemetrySnapshot) -> PoseSnapshot | None:
        """从可变遥测对象提取纯标量位姿。"""
        position = cast(
            tuple[float, float, float] | None,
            _finite_tuple(telemetry.north_m, telemetry.east_m, telemetry.down_m),
        )
        quaternion = cast(
            tuple[float, float, float, float] | None,
            _finite_tuple(
                telemetry.quaternion_w,
                telemetry.quaternion_x,
                telemetry.quaternion_y,
                telemetry.quaternion_z,
            ),
        )
        euler = cast(
            tuple[float, float, float] | None,
            _finite_tuple(telemetry.roll_deg, telemetry.pitch_deg, telemetry.yaw_deg),
        )
        origin = cast(
            tuple[float, float, float] | None,
            _finite_tuple(
                telemetry.origin_longitude_deg,
                telemetry.origin_latitude_deg,
                telemetry.origin_altitude_m,
            ),
        )
        if all(value is None for value in (position, quaternion, euler, origin)):
            return None
        return cls(position, quaternion, euler, origin)


@dataclass(frozen=True, slots=True)
class DetectionSnapshot:
    """检测的显示副本；不含可变遥测与 extra 字典。"""

    frame_index: int
    capture_timestamp: float
    pixel: tuple[float, float]
    box: tuple[float, float, float, float]
    confidence: float
    code: int | None
    side_px: float
    raw_text: str
    ocr_confidence: float
    mode: str
    pose: PoseSnapshot | None = None
    image_size: tuple[int, int] | None = None
    undistorted: bool = False


@dataclass(frozen=True, slots=True)
class ProcessedFrame:
    """跨线程传递的单帧只读快照。图像由不可变 bytes 支撑，无法恢复写权限。"""

    frame_index: int
    received_timestamp: float
    capture_timestamp: float
    processed_timestamp: float
    image: np.ndarray
    image_size: tuple[int, int]
    camera_matrix: tuple[tuple[float, float, float], ...] | None
    undistorted: bool
    pose: PoseSnapshot | None
    detections: tuple[DetectionSnapshot, ...]


def snapshot_detection(detection: Detection) -> DetectionSnapshot:
    """把业务检测收窄为不含可变引用的显示数据。"""
    box = detection.box
    return DetectionSnapshot(
        frame_index=detection.frame_index,
        capture_timestamp=detection.capture_timestamp,
        pixel=(float(detection.pixel[0]), float(detection.pixel[1])),
        box=(float(box.x1), float(box.y1), float(box.x2), float(box.y2)),
        confidence=float(detection.confidence),
        code=detection.code,
        side_px=float(detection.side_px),
        raw_text=str(detection.raw_text),
        ocr_confidence=float(detection.ocr_confidence),
        mode=str(detection.mode),
        pose=PoseSnapshot.from_telemetry(detection.telemetry),
        image_size=_image_size(detection.extra.get("image_size")),
        undistorted=bool(detection.extra.get("undistorted", False)),
    )


def snapshot_pose(telemetry: TelemetrySnapshot | None) -> PoseSnapshot | None:
    """提取只读位姿；空遥测或完全无几何字段时返回 None。"""
    return None if telemetry is None else PoseSnapshot.from_telemetry(telemetry)


def readonly_image(image: np.ndarray) -> np.ndarray:
    """复制图像到不可变 bytes 底层，调用方不能重新打开其写权限。"""
    array = np.asarray(image)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _finite_tuple(*values: float | None) -> tuple[float, ...] | None:
    if any(value is None for value in values):
        return None
    result = tuple(float(value) for value in values if value is not None)
    if len(result) != len(values) or not all(np.isfinite(value) for value in result):
        return None
    return result


def _image_size(value: object) -> tuple[int, int] | None:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        return None
    width, height = value
    if not isinstance(width, (int, float)) or not isinstance(height, (int, float)):
        return None
    if not np.isfinite(width) or not np.isfinite(height) or width <= 0 or height <= 0:
        return None
    return int(width), int(height)
