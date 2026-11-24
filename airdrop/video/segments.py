"""实验视频采集的进程内分段生命周期。"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction


class SegmentState(StrEnum):
    """当前运行进程能够证明的 TS 分段状态。"""

    UNKNOWN = "unknown"
    ACTIVE = "active"
    CLOSED_COMPLETE = "closed_complete"
    CLOSED_INCOMPLETE = "closed_incomplete"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class SegmentSnapshot:
    """分段状态、单调递增修订号、写端提交水位与故障信息。"""

    segment: int
    state: SegmentState
    revision: int
    committed_time: Fraction | None = None
    error: str | None = None


class SegmentLifecycle:
    """让同一采集会话的写入端与冷读端共享分段活动状态与已落盘提交水位。"""

    def __init__(self) -> None:
        self._condition = threading.Condition(threading.RLock())
        self._states: dict[int, SegmentState] = {}
        self._revisions: dict[int, int] = {}
        self._committed_times: dict[int, Fraction] = {}
        self._errors: dict[int, str] = {}
        self._revision = 0

    def snapshot(self, segment: int) -> SegmentSnapshot:
        """读取分段状态与写端提交水位；缺少本进程证据时返回 UNKNOWN。"""
        with self._condition:
            return SegmentSnapshot(
                segment,
                self._states.get(segment, SegmentState.UNKNOWN),
                self._revisions.get(segment, 0),
                self._committed_times.get(segment),
                self._errors.get(segment),
            )

    def has_failed(self) -> tuple[int, str | None] | None:
        """返回首个进入 FAILED 终态的分段编号与错误描述；无故障时返回 None。"""
        with self._condition:
            for segment, state in self._states.items():
                if state is SegmentState.FAILED:
                    return segment, self._errors.get(segment)
            return None

    def opened(self, segment: int) -> SegmentSnapshot:
        """登记新分段的写入器已经创建。"""
        with self._condition:
            state = self._states.get(segment, SegmentState.UNKNOWN)
            if state is not SegmentState.UNKNOWN:
                raise RuntimeError(f"TS 分段 #{segment} 重复 opened: {state.value}")
            return self._update_locked(segment, SegmentState.ACTIVE)

    def progress(self, segment: int, *, committed_time: Fraction | None = None) -> SegmentSnapshot:
        """记录活动分段的新帧已写入并推进写端单调提交水位。"""
        with self._condition:
            state = self._states.get(segment, SegmentState.UNKNOWN)
            if state is not SegmentState.ACTIVE:
                raise RuntimeError(f"非活动 TS 分段 #{segment} 收到 progress: {state.value}")
            if committed_time is not None:
                prev = self._committed_times.get(segment)
                self._committed_times[segment] = (
                    committed_time if prev is None or committed_time > prev else prev
                )
            return self._update_locked(segment, SegmentState.ACTIVE)

    def closed(
        self, segment: int, *, complete: bool, committed_time: Fraction | None = None
    ) -> SegmentSnapshot:
        """确认 writer 已关闭，并记录分段是否干净封口及最终提交水位。"""
        target = SegmentState.CLOSED_COMPLETE if complete else SegmentState.CLOSED_INCOMPLETE
        return self._close(segment, target, committed_time=committed_time)

    def failed(self, segment: int, *, error: str | None = None) -> SegmentSnapshot:
        """确认分段写入器或其工作进程未能稳定关闭。"""
        return self._close(segment, SegmentState.FAILED, error=error)

    def _close(
        self,
        segment: int,
        target: SegmentState,
        *,
        committed_time: Fraction | None = None,
        error: str | None = None,
    ) -> SegmentSnapshot:
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
            if committed_time is not None:
                prev = self._committed_times.get(segment)
                self._committed_times[segment] = (
                    committed_time if prev is None or committed_time > prev else prev
                )
            if error:
                self._errors[segment] = error
            return self._update_locked(segment, target)

    def _update_locked(self, segment: int, state: SegmentState) -> SegmentSnapshot:
        self._revision += 1
        self._states[segment] = state
        self._revisions[segment] = self._revision
        self._condition.notify_all()
        return SegmentSnapshot(
            segment,
            state,
            self._revision,
            self._committed_times.get(segment),
            self._errors.get(segment),
        )

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
                self._committed_times.get(segment),
                self._errors.get(segment),
            )
