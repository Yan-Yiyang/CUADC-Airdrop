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
from typing import Any

from ..telemetry.models import TelemetrySnapshot

__all__ = ["Detection", "PixelBox"]


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
            },
            **({"extra": dict(self.extra)} if self.extra else {}),
        }
