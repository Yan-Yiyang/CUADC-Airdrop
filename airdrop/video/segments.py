"""实验视频采集的进程内分段生命周期。"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from enum import StrEnum


class SegmentState(StrEnum):
    """当前运行进程能够证明的 TS 分段状态。"""

    UNKNOWN = "unknown"
    ACTIVE = "active"
    CLOSED_COMPLETE = "closed_complete"
    CLOSED_INCOMPLETE = "closed_incomplete"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class SegmentSnapshot:
    """分段状态及其单调递增修订号。"""

    segment: int
    state: SegmentState
    revision: int


class SegmentLifecycle:
    """让同一采集会话的写入端与冷读端共享分段活动状态。"""

    def __init__(self) -> None:
        self._condition = threading.Condition(threading.RLock())
        self._states: dict[int, SegmentState] = {}
        self._revisions: dict[int, int] = {}
        self._revision = 0

    def snapshot(self, segment: int) -> SegmentSnapshot:
        """读取分段状态；缺少本进程证据时返回 UNKNOWN。"""
        with self._condition:
            return SegmentSnapshot(
                segment,
                self._states.get(segment, SegmentState.UNKNOWN),
                self._revisions.get(segment, 0),
            )

    def opened(self, segment: int) -> SegmentSnapshot:
        """登记新分段的写入器已经创建。"""
        with self._condition:
            state = self._states.get(segment, SegmentState.UNKNOWN)
            if state is not SegmentState.UNKNOWN:
                raise RuntimeError(f"TS 分段 #{segment} 重复 opened: {state.value}")
            return self._update_locked(segment, SegmentState.ACTIVE)

    def progress(self, segment: int) -> SegmentSnapshot:
        """记录活动分段的新帧已写入并可供冷读。"""
        with self._condition:
            state = self._states.get(segment, SegmentState.UNKNOWN)
            if state is not SegmentState.ACTIVE:
                raise RuntimeError(f"非活动 TS 分段 #{segment} 收到 progress: {state.value}")
            return self._update_locked(segment, SegmentState.ACTIVE)

    def closed(self, segment: int, *, complete: bool) -> SegmentSnapshot:
        """确认 writer 已关闭，并记录分段是否干净封口。"""
        target = SegmentState.CLOSED_COMPLETE if complete else SegmentState.CLOSED_INCOMPLETE
        return self._close(segment, target)

    def failed(self, segment: int) -> SegmentSnapshot:
        """确认分段写入器或其工作进程未能稳定关闭。"""
        return self._close(segment, SegmentState.FAILED)

    def _close(self, segment: int, target: SegmentState) -> SegmentSnapshot:
        with self._condition:
            state = self._states.get(segment, SegmentState.UNKNOWN)
            if state in {
                SegmentState.CLOSED_COMPLETE,
                SegmentState.CLOSED_INCOMPLETE,
                SegmentState.FAILED,
            }:
                if state is target:
                    return self.snapshot(segment)
                raise RuntimeError(f"TS 分段 #{segment} 终态冲突: {state.value} -> {target.value}")
            return self._update_locked(segment, target)

    def _update_locked(self, segment: int, state: SegmentState) -> SegmentSnapshot:
        self._revision += 1
        self._states[segment] = state
        self._revisions[segment] = self._revision
        self._condition.notify_all()
        return SegmentSnapshot(segment, state, self._revision)

    def wait_for_change(
        self, segment: int, revision: int, timeout: float | None = None
    ) -> SegmentSnapshot:
        """在锁外由调用方等待分段进展或终态。"""
        with self._condition:
            self._condition.wait_for(
                lambda: (
                    self._revisions.get(segment, 0) > revision
                    or self._states.get(segment, SegmentState.UNKNOWN) is not SegmentState.ACTIVE
                ),
                timeout=timeout,
            )
            return SegmentSnapshot(
                segment,
                self._states.get(segment, SegmentState.UNKNOWN),
                self._revisions.get(segment, 0),
            )
