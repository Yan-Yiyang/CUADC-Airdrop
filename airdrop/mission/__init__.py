"""任务编排：状态机 + 航线规划 + 主循环（计划 4.7）。

* :mod:`~airdrop.mission.items`：:class:`MissionItem`——任务的唯一表示（MAVLink
  级 command/frame/params/位置），上传走 ``mission_raw``，不经 MAVSDK 的翻译层；
* :mod:`~airdrop.mission.plan_file`：QGC ``.plan`` 解析 + 固定翼降落预检（本地照
  PX4 的判据先检查一遍，不合格就带着原因失败，而不是上传后被飞控整条拒掉）；
* :mod:`~airdrop.mission.states`：状态、合法转移表与转移历史；
* :mod:`~airdrop.mission.planner`：侦查段 / 飞掠段 [entry, exit] / 与降落段
  拼接成一条任务（纯函数，离线可测）；
* :mod:`~airdrop.mission.targets`：:class:`TargetTracker`——检测结果 → georef →
  目标点 → 统计结果（全链路里"坐标解算"那一段，实飞与回放共用）；
* :mod:`~airdrop.mission.runner`：:class:`MissionRunner`——驱动控制器、监视遥测、
  推进状态机（单拍 :meth:`~MissionRunner.update` 或连续 :meth:`~MissionRunner.run`）。

控制器面向 :class:`~airdrop.telemetry.MissionController` 协议编程，所以本包不需要
飞控、也不需要起线程就能整体离线测试（见 ``tests/test_mission.py`` 的假控制器）。

⚠ 惰性导出（见 :mod:`airdrop._lazy`）：import airdrop.mission 只执行本文件。
``targets``/``planner`` 会拉 georef（cv2），``runner`` 会拉 mavsdk，所以它们排在后面——
"取个 MissionState"不该顺带加载坐标解算与飞控栈。
"""

from .._lazy import lazy_dir, lazy_exports

__getattr__ = lazy_exports(
    __name__, ("states", "items", "plan_file", "targets", "planner", "runner")
)
__dir__ = lazy_dir(__name__)

__all__ = [
    "DEFAULT_SIDE_TOLERANCE",
    "FW_DEFAULT_LAND_ANGLE_DEG",
    "MAV_CMD_DO_CHANGE_SPEED",
    "MAV_CMD_DO_LAND_START",
    "MAV_CMD_DO_SET_CAM_TRIGG_DIST",
    "MAV_CMD_IMAGE_STOP_CAPTURE",
    "MAV_CMD_NAV_LAND",
    "MAV_CMD_NAV_LOITER_TO_ALT",
    "MAV_CMD_NAV_LOITER_UNLIM",
    "MAV_CMD_NAV_TAKEOFF",
    "MAV_CMD_NAV_WAYPOINT",
    "MAV_CMD_VIDEO_STOP_CAPTURE",
    "MAV_FRAME_GLOBAL",
    "MAV_FRAME_GLOBAL_RELATIVE_ALT",
    "MAV_FRAME_GLOBAL_TERRAIN_ALT",
    "MAV_FRAME_MISSION",
    "PLAN_FIRMWARE_PX4",
    "PLAN_VEHICLE_FIXED_WING",
    "TERMINAL_STATES",
    "TRANSITIONS",
    "UNSET",
    "DropMissionPlan",
    "InvalidTransition",
    "MissionItem",
    "MissionMonitor",
    "MissionRunner",
    "MissionState",
    "MissionStateMachine",
    "MissionStats",
    "MissionTransition",
    "PerceptionTargetSource",
    "PlanError",
    "PlanningError",
    "PreflightLike",
    "QgcPlan",
    "ReleaseJudgeLike",
    "TargetResultReader",
    "TargetTracker",
    "build_drop_mission",
    "build_recon_mission",
    "check_fixed_wing_landing",
    "command_name",
    "emit_event",
    "llaref_of",
    "load_plan",
    "overfly_positions",
    "overfly_waypoints",
    "waypoint_to_ned",
]
