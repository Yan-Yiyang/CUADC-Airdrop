"""任务状态机：状态、合法转移与转移历史（计划 4.7）。

::

    INIT ──▶ PREFLIGHT ──▶ WAIT_AIRBORNE ──▶ RECON ──▶ HOLD_PROCESS ──▶ OVERFLY ──▶ LAND ──▶ DONE
      │          │               │             │             │              │          │
      └──────────┴───────────────┴─────────────┴─────────────┴──────────────┴──────────┴──▶ ABORT

各状态的含义与进入条件（判定逻辑在 :mod:`airdrop.mission.runner`）：

* ``INIT``：等遥测与 NED 原点（不上传任何任务）；遥测有效且原点就绪 → ``PREFLIGHT``。
* ``PREFLIGHT``：起飞前自检——载入模型（detector/ocr/camera）→ 视频自检
  （见 :mod:`airdrop.preflight`）；全部通过 → ``WAIT_AIRBORNE``；任一项失败 →
  ``ABORT("preflight_failed:<check>")``；超过 ``preflight.max_s`` → ``ABORT``。
* ``WAIT_AIRBORNE``：什么都不下发，等飞机确实在空中（``in_air``；取不到时用
  ``relative_altitude_m >= airborne_alt_m`` 判定）；``MissionConfig.require_airborne=False``
  （仅地面演练与离线测试）时立即放行并记 ``airborne_skipped``；
  在空中 → ``RECON``；超过 ``airborne_timeout_s`` → ``ABORT("airborne_timeout")``。
* ``RECON``：``recon_upload="operator"``（默认，正式任务）不上传，等操作手在 QGC 启动；
  ``"auto"``（自动测试）由本包上传并启动；mission 报飞完 → ``HOLD_PROCESS``。
* ``HOLD_PROCESS``：``hold()``（固定翼盘旋）；出结果、或处理完且无结果、或 10s 超时
  → 上传"飞掠 + 降落"任务并启动 → ``OVERFLY``。
* ``OVERFLY``：激活投放判据；投放成功 → ``LAND``；任务飞完仍未投放 → ``ABORT``。
* ``LAND``：沿降落航线返航；mission 飞完 → ``DONE``。
* ``ABORT``：下安全动作（``MissionConfig.abort_action``）后终止。

DONE / ABORT 是终态，不再离开——"任务完成后又重新开始"这种事由调用方
新建一个 runner 表达，不在状态机里留后门。
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

LOGGER = logging.getLogger(__name__)

__all__ = [
    "TERMINAL_STATES",
    "TRANSITIONS",
    "InvalidTransition",
    "MissionState",
    "MissionStateMachine",
    "MissionTransition",
    "emit_event",
]


class MissionState(StrEnum):
    """任务状态。继承 ``str`` 是为了直接进 JSONL 事件日志。"""

    INIT = "INIT"
    PREFLIGHT = "PREFLIGHT"
    WAIT_AIRBORNE = "WAIT_AIRBORNE"
    RECON = "RECON"
    HOLD_PROCESS = "HOLD_PROCESS"
    OVERFLY = "OVERFLY"
    LAND = "LAND"
    DONE = "DONE"
    ABORT = "ABORT"


#: 合法转移表（唯一出处；runner 只能走这里列出的边）
TRANSITIONS: dict[MissionState, frozenset[MissionState]] = {
    MissionState.INIT: frozenset({MissionState.PREFLIGHT, MissionState.ABORT}),
    MissionState.PREFLIGHT: frozenset({MissionState.WAIT_AIRBORNE, MissionState.ABORT}),
    MissionState.WAIT_AIRBORNE: frozenset({MissionState.RECON, MissionState.ABORT}),
    MissionState.RECON: frozenset({MissionState.HOLD_PROCESS, MissionState.ABORT}),
    MissionState.HOLD_PROCESS: frozenset({MissionState.OVERFLY, MissionState.ABORT}),
    MissionState.OVERFLY: frozenset({MissionState.LAND, MissionState.ABORT}),
    MissionState.LAND: frozenset({MissionState.DONE, MissionState.ABORT}),
    MissionState.DONE: frozenset(),
    MissionState.ABORT: frozenset(),
}

#: 终态：到这儿就停，不再转移
TERMINAL_STATES = frozenset({MissionState.DONE, MissionState.ABORT})


class InvalidTransition(RuntimeError):
    """非法的状态转移（表里没有这条边）。"""


@dataclass(frozen=True, slots=True)
class MissionTransition:
    """一次状态转移（进事件日志与历史）。"""

    from_state: MissionState
    to_state: MissionState
    timestamp: float
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "from_state": str(self.from_state),
            "to_state": str(self.to_state),
            "reason": self.reason,
            "timestamp": self.timestamp,
        }


def emit_event(
    on_event: Callable[[str, dict[str, Any]], None] | None,
    kind: str,
    data: dict[str, Any],
) -> None:
    """事件回调的统一入口：写盘失败只记日志，绝不影响控制流。

    （recorder 的 ``EventLog.emit`` 在关闭后抛 ``RuntimeError``；任务收尾阶段
    仍可能有事件要落，这种异常不能把状态机带崩。）
    """
    if on_event is None:
        return
    try:
        on_event(kind, data)
    except Exception:
        LOGGER.exception("写入任务事件失败（控制继续）")


@dataclass
class MissionStateMachine:
    """带合法转移校验与历史的状态机（runner 持有一个）。

    ``on_event`` 收到 ``("state", {...})``——即每次成功的转移都记一条，
    非法转移直接抛 :class:`InvalidTransition` 且状态不变。
    """

    state: MissionState = MissionState.INIT
    on_event: Callable[[str, dict[str, Any]], None] | None = None
    history: list[MissionTransition] = field(default_factory=list)

    @property
    def is_terminal(self) -> bool:
        return self.state in TERMINAL_STATES

    def can_transition(self, to_state: MissionState) -> bool:
        return to_state in TRANSITIONS[self.state]

    def transition(
        self,
        to_state: MissionState,
        *,
        reason: str = "",
        now: float | None = None,
    ) -> MissionTransition:
        """执行一次转移；非法边抛 :class:`InvalidTransition`（状态保持原样）。"""
        if not self.can_transition(to_state):
            raise InvalidTransition(
                f"非法状态转移：{self.state} → {to_state}"
                f"（允许 {sorted(str(s) for s in TRANSITIONS[self.state])}）"
            )
        record = MissionTransition(
            from_state=self.state,
            to_state=to_state,
            timestamp=time.time() if now is None else float(now),
            reason=reason,
        )
        self.state = to_state
        self.history.append(record)
        LOGGER.info("任务状态：%s → %s（%s）", record.from_state, record.to_state, reason)
        emit_event(self.on_event, "state", record.as_dict())
        return record
