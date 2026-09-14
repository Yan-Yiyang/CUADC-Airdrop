"""任务编排：状态机 + 航线规划 + 主循环（计划 4.7）。

* :mod:`~airdrop.mission.states`：状态、合法转移表与转移历史；
* :mod:`~airdrop.mission.planner`：侦查航线 / 飞掠段 `[entry, exit]` / 与降落航线
  拼接成一条任务（纯函数，离线可测）；
* :mod:`~airdrop.mission.targets`：:class:`TargetTracker`——检测结果 → georef →
  目标点 → 统计结果（全链路里"坐标解算"那一段，实飞与回放共用）；
* :mod:`~airdrop.mission.runner`：:class:`MissionRunner`——驱动控制器、监视遥测、
  推进状态机（单拍 :meth:`~MissionRunner.update` 或连续 :meth:`~MissionRunner.run`）。

控制器面向 :class:`~airdrop.telemetry.MissionController` **协议**编程，所以本包不需要
飞控、也不需要起线程就能整体离线测试（见 ``tests/test_mission.py`` 的假控制器）。
"""

from .planner import (
    DropMissionPlan,
    PlanningError,
    build_drop_mission,
    build_recon_mission,
    llaref_of,
    overfly_positions,
    overfly_waypoints,
    waypoint_to_ned,
)
from .runner import (
    MissionMonitor,
    MissionRunner,
    MissionStats,
    ReleaseJudgeLike,
    TargetResultReader,
)
from .states import (
    TERMINAL_STATES,
    TRANSITIONS,
    InvalidTransition,
    MissionState,
    MissionStateMachine,
    MissionTransition,
    emit_event,
)
from .targets import DEFAULT_SIDE_TOLERANCE, PerceptionTargetSource, TargetTracker

__all__ = [
    "DEFAULT_SIDE_TOLERANCE",
    "TERMINAL_STATES",
    "TRANSITIONS",
    "DropMissionPlan",
    "InvalidTransition",
    "MissionMonitor",
    "MissionRunner",
    "MissionState",
    "MissionStateMachine",
    "MissionStats",
    "MissionTransition",
    "PerceptionTargetSource",
    "PlanningError",
    "ReleaseJudgeLike",
    "TargetResultReader",
    "TargetTracker",
    "build_drop_mission",
    "build_recon_mission",
    "emit_event",
    "llaref_of",
    "overfly_positions",
    "overfly_waypoints",
    "waypoint_to_ned",
]
