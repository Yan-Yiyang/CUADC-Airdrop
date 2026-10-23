"""字段源时标历史；相机仍使用接收时间减标定延时，不需要曝光时间。"""

from __future__ import annotations

import bisect
import copy
import math
from collections import deque
from collections.abc import Callable

from .models import TelemetrySnapshot


class SourceHistory:
    """位置与四元数独立保留，以四元数接收时轴估计共同源时刻。

    映射由相邻源时标/接收时刻局部内插，允许 SITL 时钟速率变化。它不是
    精确时钟同步：公共传输延迟和相机延迟仍需联合标定。调用者持有 broker 锁。
    """

    def __init__(self, maxlen: int) -> None:
        self.position: deque[TelemetrySnapshot] = deque(maxlen=maxlen)
        self.attitude: deque[TelemetrySnapshot] = deque(maxlen=maxlen)
        self._epoch_started_at: float | None = None

    def clear(self) -> None:
        self.position.clear()
        self.attitude.clear()
        self._epoch_started_at = None

    def add(self, snapshot: TelemetrySnapshot) -> None:
        """重复快照不重复入库；源钟回退一秒以上时清空整个钟域。"""
        streams = (
            (self.position, "position_timestamp_us", "position_velocity_received_at"),
            (self.attitude, "quaternion_timestamp_us", "quaternion_received_at"),
        )
        rollbacks = []
        for history, stamp_name, received_name in streams:
            stamp = getattr(snapshot, stamp_name)
            received = getattr(snapshot, received_name)
            if stamp is None or received is None or stamp <= 0 or not math.isfinite(received):
                continue
            if history:
                previous = getattr(history[-1], stamp_name)
                if stamp < previous - 1_000_000:
                    rollbacks.append(received)
        if rollbacks:
            self.clear()
            self._epoch_started_at = max(rollbacks)

        for history, stamp_name, received_name in streams:
            stamp = getattr(snapshot, stamp_name)
            received = getattr(snapshot, received_name)
            if stamp is None or received is None or stamp <= 0 or not math.isfinite(received):
                continue
            if self._epoch_started_at is not None and received < self._epoch_started_at:
                continue
            if history and stamp <= getattr(history[-1], stamp_name):
                continue
            history.append(copy.copy(snapshot))

    @staticmethod
    def _pair(history, timestamp, key):
        if len(history) < 2 or timestamp < key(history[0]) or timestamp > key(history[-1]):
            return None
        index = bisect.bisect_right(history, timestamp, key=key)
        index = min(max(index, 1), len(history) - 1)
        return history[index - 1], history[index]

    def evaluate(
        self,
        timestamp: float,
        base: TelemetrySnapshot,
        blend: Callable[..., TelemetrySnapshot],
        *,
        mode: str = "interpolate",
    ) -> TelemetrySnapshot | None:
        """严格内插；缺字段、停顿、跨重启和过大采样间隔均拒绝，不伪装成功。"""
        anchors = self._pair(self.attitude, timestamp, lambda s: s.quaternion_received_at)
        if anchors is None:
            return None
        left, right = anchors
        dt = right.quaternion_received_at - left.quaternion_received_at
        if not 0 < dt <= 0.5:
            return None
        ratio = (timestamp - left.quaternion_received_at) / dt
        source = left.quaternion_timestamp_us + ratio * (
            right.quaternion_timestamp_us - left.quaternion_timestamp_us
        )
        position = self._pair(self.position, source, lambda s: s.position_timestamp_us)
        if position is None:
            return None
        result = copy.copy(base)
        for pair, stamp_name, names in (
            (
                position,
                "position_timestamp_us",
                ("north_m", "east_m", "down_m", "vx_m_s", "vy_m_s", "vz_m_s"),
            ),
            (
                anchors,
                "quaternion_timestamp_us",
                ("quaternion_w", "quaternion_x", "quaternion_y", "quaternion_z"),
            ),
        ):
            a, b = pair
            start, end = getattr(a, stamp_name), getattr(b, stamp_name)
            if not 0 < end - start <= 300_000:
                return None
            if any(
                getattr(s, n) is None or not math.isfinite(getattr(s, n))
                for s in pair
                for n in names
            ):
                return None
            a, b = copy.copy(a), copy.copy(b)
            a.timestamp, b.timestamp = start / 1e6, end / 1e6
            if mode == "nearest":
                evaluated = a if source - start <= end - source else b
            else:
                evaluated = blend(a, b, source / 1e6, copy_from=a)
            if (
                stamp_name == "quaternion_timestamp_us"
                and sum(getattr(evaluated, n) ** 2 for n in names) < 1e-12
            ):
                return None
            for name in names:
                setattr(result, name, getattr(evaluated, name))
            setattr(
                result, stamp_name, getattr(evaluated, stamp_name) if mode == "nearest" else start
            )
        # 欧拉角从同一四元数推导，避免另外一条 Euler 流再次混入。
        w, x, y, z = (getattr(result, "quaternion_" + k) for k in "wxyz")
        result.roll_deg = math.degrees(math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)))
        result.pitch_deg = math.degrees(math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x)))))
        result.yaw_deg = math.degrees(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
        result.alignment_source_timestamp_us = round(source)
        result.timestamp = timestamp
        return result
