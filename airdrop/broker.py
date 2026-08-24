"""线程安全的内部遥测代理（“内推”模块）。

该代理是同进程内飞机最新状态的唯一数据源。MAVSDK 工作线程会把部分遥测
更新发布到这里，同进程的其他线程可以：

* 通过 :meth:`TelemetryBroker.get_snapshot` 读取最新的同步快照，或
* 通过 :meth:`TelemetryBroker.subscribe` 订阅推送回调，
* 通过 :meth:`TelemetryBroker.get_snapshot_at` 按时间戳查询历史/内插外推结果。
"""

from __future__ import annotations

import bisect
import copy
import logging
import threading
import time
from collections import deque
from dataclasses import fields
from typing import Any, Callable, Deque, List, Optional

from .models import TelemetrySnapshot

LOGGER = logging.getLogger(__name__)

SnapshotCallback = Callable[[TelemetrySnapshot], None]

# 插值时不需要线性内插的时间类字段
_SKIP_INTERP_FIELDS = {"timestamp", "attitude_timestamp_us"}


class TelemetryBroker:
    """线程安全的最新遥测快照仓库。

    所有公开方法都可在任意线程中调用。快照更新通过可重入锁串行化，
    订阅者会在共享快照更新完成后被调用。
    内部会保存最近 ``history_maxlen`` 条快照，用于按时间戳查询。
    """

    def __init__(self, history_maxlen: int = 1200) -> None:
        self._lock = threading.RLock()
        self._latest = TelemetrySnapshot()
        self._history: Deque[TelemetrySnapshot] = deque(maxlen=history_maxlen)
        self._subscribers: List[SnapshotCallback] = []
        self._condition = threading.Condition(self._lock)

    # ------------------------------------------------------------------
    # 读取接口
    # ------------------------------------------------------------------
    def get_snapshot(self) -> TelemetrySnapshot:
        """返回最新快照的深拷贝（可安全修改）。"""
        with self._lock:
            return copy.deepcopy(self._latest)

    def get_snapshot_at(
        self,
        timestamp: float,
        mode: str = "interpolate",
    ) -> Optional[TelemetrySnapshot]:
        """按时间戳获取飞机状态。

        参数
        ----
        timestamp:
            查询时间戳，单位秒，与 ``TelemetrySnapshot.timestamp`` 使用同一时间轴
            （默认是 ``time.time()`` 的本地墙钟时间）。
        mode:
            ``"interpolate"``（默认）：在历史数据之间线性内插，超出历史范围时线性外推；
            ``"nearest"``：直接返回时间上最接近的一条历史快照。

        返回
        ----
        对应时刻的快照；历史为空时返回 None。
        """
        with self._lock:
            if not self._history:
                return None

            if mode == "nearest":
                return copy.deepcopy(self._nearest_snapshot_locked(timestamp))

            if mode != "interpolate":
                raise ValueError(f"不支持的查询模式: {mode}")

            return copy.deepcopy(self._evaluate_snapshot_locked(timestamp))

    def wait_next_snapshot(
        self,
        timeout: Optional[float] = None,
        predicate: Optional[Callable[[TelemetrySnapshot], bool]] = None,
    ) -> Optional[TelemetrySnapshot]:
        """阻塞直到新快照满足 ``predicate``。

        如果 ``predicate`` 为 None，则任何新快照都会被接受。
        返回快照；超时返回 None。
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            while True:
                snapshot = copy.deepcopy(self._latest)
                if predicate is None or predicate(snapshot):
                    return snapshot
                remaining = None
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return None
                self._condition.wait(remaining)

    # ------------------------------------------------------------------
    # 订阅接口
    # ------------------------------------------------------------------
    def subscribe(self, callback: SnapshotCallback) -> None:
        """注册一个回调，每次有新快照时都会被调用。

        回调运行在发布（MAVSDK 工作）线程中，因此不能长时间阻塞，也不能抛异常。
        """
        with self._lock:
            if callback not in self._subscribers:
                self._subscribers.append(callback)

    def unsubscribe(self, callback: SnapshotCallback) -> None:
        with self._lock:
            if callback in self._subscribers:
                self._subscribers.remove(callback)

    # ------------------------------------------------------------------
    # 写入接口（由 MAVSDK 工作线程调用）
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """清空已保存的快照和历史。在建立新连接前调用。"""
        with self._condition:
            self._latest = TelemetrySnapshot()
            self._history.clear()
            self._condition.notify_all()

    def update_global_position(self, position: Any) -> TelemetrySnapshot:
        """发布一个 MAVSDK 全局 ``Position`` 更新。"""
        return self._update(lambda s: s.update_global_position(position))

    def update_home_position(self, position: Any) -> TelemetrySnapshot:
        """发布一个 MAVSDK HOME ``Position`` 更新。"""
        return self._update(lambda s: s.update_home_position(position))

    def update_gps_global_origin(self, origin: Any) -> TelemetrySnapshot:
        """发布一个 MAVSDK ``GpsGlobalOrigin``（NED 原点）更新。"""
        return self._update(lambda s: s.update_gps_global_origin(origin))

    def update_local_position_velocity(
        self, position_ned: Any, velocity_ned: Any
    ) -> TelemetrySnapshot:
        """发布一个 MAVSDK 本地位置 + 速度更新。"""
        return self._update(
            lambda s: s.update_local_position_velocity(position_ned, velocity_ned)
        )

    def update_attitude_euler(self, euler: Any) -> TelemetrySnapshot:
        """发布一个 MAVSDK 欧拉角姿态更新。"""
        return self._update(lambda s: s.update_attitude_euler(euler))

    def update_attitude_quaternion(self, quaternion: Any) -> TelemetrySnapshot:
        """发布一个 MAVSDK 四元数姿态更新。"""
        return self._update(lambda s: s.update_attitude_quaternion(quaternion))

    # ------------------------------------------------------------------
    # 内部实现
    # ------------------------------------------------------------------
    def _update(
        self,
        mutator: Callable[[TelemetrySnapshot], None],
    ) -> TelemetrySnapshot:
        with self._condition:
            snapshot = copy.deepcopy(self._latest)
            mutator(snapshot)
            snapshot.timestamp = time.time()

            # 保存独立副本，避免订阅者/历史被外部修改影响最新状态。
            self._latest = copy.deepcopy(snapshot)
            self._history.append(copy.deepcopy(snapshot))
            self._condition.notify_all()

        for callback in self._subscribers:
            try:
                callback(snapshot)
            except Exception:  # noqa: BLE001 - 订阅者异常不能影响代理继续工作
                LOGGER.exception("遥测订阅回调执行失败")

        return snapshot

    # ------------------------------------------------------------------
    # 内部：按时间戳查询
    # ------------------------------------------------------------------
    def _history_times_locked(self) -> List[float]:
        return [item.timestamp for item in self._history]

    def _nearest_snapshot_locked(self, timestamp: float) -> TelemetrySnapshot:
        times = self._history_times_locked()
        idx = bisect.bisect_left(times, timestamp)
        if idx == 0:
            return self._history[0]
        if idx >= len(self._history):
            return self._history[-1]
        left = self._history[idx - 1]
        right = self._history[idx]
        if abs(timestamp - left.timestamp) <= abs(right.timestamp - timestamp):
            return left
        return right

    def _evaluate_snapshot_locked(self, timestamp: float) -> TelemetrySnapshot:
        times = self._history_times_locked()
        first = self._history[0]
        last = self._history[-1]

        # 早于最早历史：向后外推
        if timestamp <= first.timestamp:
            if len(self._history) < 2:
                return copy.deepcopy(first)
            return self._extrapolate_locked(
                self._history[0], self._history[1], timestamp
            )

        # 晚于最新历史：向前外推
        if timestamp >= last.timestamp:
            if len(self._history) < 2:
                return copy.deepcopy(last)
            return self._extrapolate_locked(
                self._history[-2], self._history[-1], timestamp
            )

        # 在历史范围内：线性内插
        idx = bisect.bisect_right(times, timestamp)
        left = self._history[idx - 1]
        right = self._history[idx]
        if left.timestamp == right.timestamp:
            return copy.deepcopy(left)
        return self._interpolate_locked(left, right, timestamp)

    @staticmethod
    def _interpolate_locked(
        left: TelemetrySnapshot,
        right: TelemetrySnapshot,
        timestamp: float,
    ) -> TelemetrySnapshot:
        result = copy.deepcopy(left)
        result.timestamp = timestamp
        ratio = (timestamp - left.timestamp) / (right.timestamp - left.timestamp)
        for field in fields(TelemetrySnapshot):
            name = field.name
            if name in _SKIP_INTERP_FIELDS:
                continue
            a = getattr(left, name)
            b = getattr(right, name)
            if a is None or b is None:
                continue
            if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                setattr(result, name, a + (b - a) * ratio)
        return result

    @staticmethod
    def _extrapolate_locked(
        left: TelemetrySnapshot,
        right: TelemetrySnapshot,
        timestamp: float,
    ) -> TelemetrySnapshot:
        result = copy.deepcopy(right)
        result.timestamp = timestamp
        dt = right.timestamp - left.timestamp
        if dt == 0:
            return result

        ratio = (timestamp - left.timestamp) / dt
        for field in fields(TelemetrySnapshot):
            name = field.name
            if name in _SKIP_INTERP_FIELDS:
                continue
            a = getattr(left, name)
            b = getattr(right, name)
            if a is None or b is None:
                continue
            if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                setattr(result, name, a + (b - a) * ratio)
        return result
