"""线程安全的进程内遥测代理。

官方示例中每条遥测流都是独立的 ``async for``；本代理把这些分散的更新
合并成一份"最新快照"，作为同进程内飞机状态的唯一数据源。其他线程可以：

* 通过 :meth:`TelemetryBroker.get_snapshot` 读取最新快照的副本；
* 通过 :meth:`TelemetryBroker.subscribe` 订阅推送回调；
* 通过 :meth:`TelemetryBroker.get_snapshot_at` 按时间戳查询历史（内插/外推）。
"""

from __future__ import annotations

import bisect
import copy
import logging
import math
import operator
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import fields
from typing import Any

from .models import TelemetrySnapshot

LOGGER = logging.getLogger(__name__)

SnapshotCallback = Callable[[TelemetrySnapshot], None]

# get_snapshot_at 支持的查询模式
SUPPORTED_QUERY_MODES = frozenset({"interpolate", "nearest"})

# 时间估计时不参与数值内插的时间类字段
_SKIP_INTERP_FIELDS = {"timestamp", "attitude_timestamp_us"}

# 欧拉角字段：混合时必须走"最短弧"，否则 179° → -179° 会错误地经过 0°
_ANGULAR_FIELDS = frozenset({"roll_deg", "pitch_deg", "yaw_deg"})

# 四元数字段：线性混合会破坏单位模长，必须用球面插值（slerp）
_QUATERNION_FIELDS = (
    "quaternion_w",
    "quaternion_x",
    "quaternion_y",
    "quaternion_z",
)
_QUATERNION_FIELD_SET = frozenset(_QUATERNION_FIELDS)

# bisect 的 key：让二分查找直接作用在历史 deque 上，避免每次查询
# 都重建整张时间列表（n=1200 时是每次查询一次 O(n) 分配）。
_snapshot_timestamp = operator.attrgetter("timestamp")


def _lerp_angle_deg(a: float, b: float, ratio: float) -> float:
    """角度按最短弧混合（单位：度），结果归一化到 (-180, 180]。

    解决 ±180° 环绕：179° 与 -179° 的中点是 ±180°，而不是 0°。
    ``ratio`` 超出 [0, 1] 时沿同一方向外推。
    """
    delta = (b - a + 180.0) % 360.0 - 180.0
    value = (a + delta * ratio + 180.0) % 360.0 - 180.0
    return 180.0 if value == -180.0 else value


def _slerp_fields(
    left: TelemetrySnapshot,
    right: TelemetrySnapshot,
    ratio: float,
) -> tuple[float, float, float, float] | None:
    """对两条快照的四元数做球面插值（slerp）；任一端缺失时返回 None。

    自动处理符号翻转（q 与 -q 表示同一姿态）；``ratio`` 超出 [0, 1]
    时沿同一圆弧外推。
    """
    q0 = [getattr(left, name) for name in _QUATERNION_FIELDS]
    q1 = [getattr(right, name) for name in _QUATERNION_FIELDS]
    if any(value is None for value in q0 + q1):
        return None

    dot = sum(a * b for a, b in zip(q0, q1, strict=True))
    if dot < 0.0:  # 取反其中一端，保证走最短路径
        q1 = [-value for value in q1]
        dot = -dot

    if dot > 0.9995:  # 夹角极小：线性混合后重新归一化即可
        blended = [a + (b - a) * ratio for a, b in zip(q0, q1, strict=True)]
        norm = math.sqrt(sum(value * value for value in blended))
        return tuple(value / norm for value in blended)

    theta = math.acos(min(1.0, dot))
    sin_theta = math.sin(theta)
    scale0 = math.sin((1.0 - ratio) * theta) / sin_theta
    scale1 = math.sin(ratio * theta) / sin_theta
    return tuple(scale0 * a + scale1 * b for a, b in zip(q0, q1, strict=True))


class TelemetryBroker:
    """线程安全的最新遥测快照仓库。

    快照更新通过可重入锁串行化，订阅者在快照更新完成后被调用。

    历史按固定采样率写入（``history_interval``，默认 0.1 秒）：多条高频
    遥测流在这里被合并成均匀的快照序列，默认 1200 条历史约覆盖 2 分钟；
    设为 0 表示每次遥测更新都写入历史。最新快照与订阅推送不受采样率
    限制，总是实时的。

    ``TelemetrySnapshot`` 的字段全部是标量，因此传递副本时浅拷贝就足够：
    调用方修改副本不会影响仓库内容，开销也远小于深拷贝。
    """

    def __init__(
        self,
        history_maxlen: int = 1200,
        history_interval: float = 0.1,
    ) -> None:
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._latest = TelemetrySnapshot()
        self._history: deque[TelemetrySnapshot] = deque(maxlen=history_maxlen)
        self._history_interval = max(history_interval, 0.0)
        self._last_history_timestamp = 0.0
        self._subscribers: list[SnapshotCallback] = []

    # ------------------------------------------------------------------
    # 读取接口
    # ------------------------------------------------------------------
    def get_snapshot(self) -> TelemetrySnapshot:
        """返回最新快照的副本（可安全修改）。"""
        with self._lock:
            return copy.copy(self._latest)

    def get_snapshot_at(
        self,
        timestamp: float,
        mode: str = "interpolate",
    ) -> TelemetrySnapshot | None:
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
        对应时刻的快照；历史为空时返回 None。非法 ``mode`` 抛 ``ValueError``
        （与历史是否为空无关）。
        """
        if mode not in SUPPORTED_QUERY_MODES:  # 先校验模式：错误语义不该随历史有无而变
            raise ValueError(f"不支持的查询模式: {mode}，可选 {sorted(SUPPORTED_QUERY_MODES)}")
        with self._lock:
            if not self._history:
                return None

            if mode == "nearest":
                return copy.copy(self._nearest_snapshot_locked(timestamp))

            return copy.copy(self._evaluate_snapshot_locked(timestamp))

    def history_span(self) -> tuple[float, float] | None:
        """返回历史覆盖的时间范围 ``(最早, 最新)``；历史为空时返回 None。

        调用方据此判断 :meth:`get_snapshot_at` 给出的结果是不是外推出来的
        （查询时刻落在范围之外），例如 :mod:`airdrop.alignment` 用它在
        帧-遥测对齐时标记"这一刻没有可信遥测"。
        """
        with self._lock:
            if not self._history:
                return None
            return self._history[0].timestamp, self._history[-1].timestamp

    def wait_history_until(
        self,
        timestamp: float,
        timeout: float | None = None,
    ) -> bool:
        """阻塞直到**历史**覆盖到 ``timestamp``（最新一条的时间戳 ≥ 它）。

        给帧-遥测对齐用：当画面的拍摄时刻比手上最新的遥测还新时，先等一小会儿
        让遥测追上，这样就能**内插**而不是外推（见
        :mod:`airdrop.video.align` 的 ``max_wait``）。

        为什么以历史为准而不是最新快照：内插只吃历史。若只等最新快照，
        节流写入的 0.1s 间隔仍会让查询落在外推区间里。

        返回 ``True`` 表示历史已覆盖该时刻；超时返回 ``False``。
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            while True:
                if self._history and self._history[-1].timestamp >= timestamp:
                    return True
                remaining = None
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return False
                self._condition.wait(remaining)

    def wait_next_snapshot(
        self,
        timeout: float | None = None,
        predicate: Callable[[TelemetrySnapshot], bool] | None = None,
    ) -> TelemetrySnapshot | None:
        """阻塞直到快照满足 ``predicate``。

        如果 ``predicate`` 为 None，则任何快照都会被接受。
        返回快照副本；超时返回 None。
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._condition:
            while True:
                snapshot = copy.copy(self._latest)
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
        每个订阅者拿到的是独立副本，可以随意修改。
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
            self._last_history_timestamp = 0.0
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
        return self._update(lambda s: s.update_local_position_velocity(position_ned, velocity_ned))

    def update_attitude_euler(self, euler: Any) -> TelemetrySnapshot:
        """发布一个 MAVSDK 欧拉角姿态更新。"""
        return self._update(lambda s: s.update_attitude_euler(euler))

    def update_attitude_quaternion(self, quaternion: Any) -> TelemetrySnapshot:
        """发布一个 MAVSDK 四元数姿态更新。"""
        return self._update(lambda s: s.update_attitude_quaternion(quaternion))

    def update_wind(self, wind: Any) -> TelemetrySnapshot:
        """发布一个 MAVSDK 风估计更新（可选流；没有它时投放判据按零风降级）。"""
        return self._update(lambda s: s.update_wind(wind))

    def update_mission_progress(self, current: int, total: int) -> TelemetrySnapshot:
        """发布一次任务进度更新（可选流；状态机靠它判断"任务是否飞完"）。"""
        return self._update(lambda s: s.update_mission_progress(current, total))

    def update_flight_mode(self, mode: Any) -> TelemetrySnapshot:
        """发布一次飞行模式更新（可选流；状态机靠它确认"真的进了任务模式"）。"""
        return self._update(lambda s: s.update_flight_mode(mode))

    def update_in_air(self, in_air: Any) -> TelemetrySnapshot:
        """发布一次"是否在空中"更新（可选流；状态机的等起飞门用它）。"""
        return self._update(lambda s: s.update_in_air(in_air))

    def publish(self, snapshot: TelemetrySnapshot) -> TelemetrySnapshot:
        """直接发布一条**完整快照**（保留它自己的 ``timestamp``、不做节流）。

        给"离线回填"用：:class:`~airdrop.record.replay.TelemetryPacer` 把飞行
        目录里 ``telemetry.jsonl`` 的快照按**原始时间戳**按序灌回来，于是
        ``get_snapshot_at`` 能在与当时完全一致的时间轴上查询——回放与实飞
        走同一条对齐/解算代码，不需要为离线模式开分支。

        与 ``update_*`` 的区别：那些方法从 MAVSDK 消息更新**单个字段**、时间戳
        取当前墙钟；本方法要求调用方自己保证 ``timestamp`` 单调递增，且**每次
        调用都进历史**（不看 ``history_interval``）——回填的时间轴不能像实时流
        那样被采样节流修改。
        """
        return self._update_from_snapshot(snapshot)

    # ------------------------------------------------------------------
    # 内部实现
    # ------------------------------------------------------------------
    def _update_from_snapshot(self, snapshot: TelemetrySnapshot) -> TelemetrySnapshot:
        """把整条快照作为最新值 + 历史写入，时间戳原样保留。"""
        with self._condition:
            record = copy.copy(snapshot)
            self._latest = record
            self._history.append(record)
            self._last_history_timestamp = record.timestamp
            # 锁内复制订阅者列表，避免遍历时与 subscribe/unsubscribe 竞态
            subscribers = tuple(self._subscribers)
            self._condition.notify_all()

        for callback in subscribers:
            try:
                callback(copy.copy(record))
            except Exception:
                LOGGER.exception("遥测订阅回调执行失败")

        return record

    def _update(
        self,
        mutator: Callable[[TelemetrySnapshot], None],
    ) -> TelemetrySnapshot:
        with self._condition:
            snapshot = copy.copy(self._latest)
            mutator(snapshot)
            snapshot.timestamp = time.time()
            self._latest = snapshot
            # 历史按固定采样率写入：各条高频遥测流在此合并成均匀序列
            if snapshot.timestamp - self._last_history_timestamp >= self._history_interval:
                self._history.append(snapshot)
                self._last_history_timestamp = snapshot.timestamp
            # 锁内复制订阅者列表，避免遍历时与 subscribe/unsubscribe 竞态。
            subscribers = tuple(self._subscribers)
            self._condition.notify_all()

        for callback in subscribers:
            try:
                callback(copy.copy(snapshot))
            except Exception:
                LOGGER.exception("遥测订阅回调执行失败")

        return snapshot

    # ------------------------------------------------------------------
    # 内部：按时间戳查询
    # ------------------------------------------------------------------
    def _nearest_snapshot_locked(self, timestamp: float) -> TelemetrySnapshot:
        history = self._history
        idx = bisect.bisect_left(history, timestamp, key=_snapshot_timestamp)
        if idx == 0:
            return history[0]
        if idx >= len(history):
            return history[-1]
        left = history[idx - 1]
        right = history[idx]
        if abs(timestamp - left.timestamp) <= abs(right.timestamp - timestamp):
            return left
        return right

    def _evaluate_snapshot_locked(self, timestamp: float) -> TelemetrySnapshot:
        history = self._history
        first = history[0]
        last = history[-1]

        # 早于最早历史：用最早两条向后外推
        if timestamp <= first.timestamp:
            if len(history) < 2:
                return copy.copy(first)
            return self._blend_locked(
                history[0],
                history[1],
                timestamp,
                copy_from=history[1],
            )

        # 晚于最新历史：用最新两条向前外推
        if timestamp >= last.timestamp:
            if len(history) < 2:
                return copy.copy(last)
            return self._blend_locked(
                history[-2],
                history[-1],
                timestamp,
                copy_from=history[-1],
            )

        # 在历史范围内：内插。bisect 的 key 参数直接探测快照时间戳，
        # 只有这个分支才需要二分，两个外推分支连查找都不用做。
        idx = bisect.bisect_right(history, timestamp, key=_snapshot_timestamp)
        left = history[idx - 1]
        right = history[idx]
        return self._blend_locked(left, right, timestamp, copy_from=left)

    @staticmethod
    def _blend_locked(
        left: TelemetrySnapshot,
        right: TelemetrySnapshot,
        timestamp: float,
        *,
        copy_from: TelemetrySnapshot,
    ) -> TelemetrySnapshot:
        """按时间戳混合两条相邻快照。

        数值字段线性混合；欧拉角走最短弧；四元数做 slerp。
        ``copy_from`` 决定只有一端有值的字段取自哪条快照（内插取 left、
        外推取 right）；混合比例落在 [0, 1] 之外时即为外推。
        """
        result = copy.copy(copy_from)
        result.timestamp = timestamp
        dt = right.timestamp - left.timestamp
        if dt == 0:
            return result
        ratio = (timestamp - left.timestamp) / dt

        for f in fields(TelemetrySnapshot):
            name = f.name
            if name in _SKIP_INTERP_FIELDS or name in _QUATERNION_FIELD_SET:
                continue
            a = getattr(left, name)
            b = getattr(right, name)
            if a is None or b is None:
                continue
            if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
                continue
            if name in _ANGULAR_FIELDS:
                setattr(result, name, _lerp_angle_deg(a, b, ratio))
            else:
                setattr(result, name, a + (b - a) * ratio)

        quaternion = _slerp_fields(left, right, ratio)
        if quaternion is not None:
            (
                result.quaternion_w,
                result.quaternion_x,
                result.quaternion_y,
                result.quaternion_z,
            ) = quaternion
        return result
