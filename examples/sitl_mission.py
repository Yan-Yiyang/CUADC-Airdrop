'SITL 演练：对着 PX4 SITL 把**控制链路**整条跑一遍（不装弹、不用图传）。\n\n用法::\n\n    # 1) 先起 PX4 SITL（WSL / Linux 里；机型按你的空域选，固定翼用 gazebo 的 x500 之类）\n    #    make px4_sitl gz_x500\n    # 2) 在 Windows 侧跑本脚本（MAVSDK 默认连 udpin://0.0.0.0:14540）\n    python -m examples.sitl_mission\n\n它演练什么、不演练什么\n----------------------\n演练（与实飞同一条代码路径）：状态机推进、侦查航线上传与启动、盘旋（hold）、\n结果坐标 → 飞掠航线规划 → 与降落航线合并上传、投放判据（**真弹道、真风降级**）、\n一次性锁存、投放后的降落段、事件日志与记录五件套。\n\n不演练：**感知**（SITL 里没有相机与目标画面）与**开仓**（SITL 没有 gripper 硬件）。\n所以目标坐标由 ``TARGET_OFFSET_NED`` 合成（相对"盘旋点"），投放走\n:class:`~airdrop.DryRunController`——只记日志、不下发指令，其余全真。\n落地后请核对 ``RESULT`` 输出与 ``flights/<本次>/events.jsonl``。\n'

from __future__ import annotations

import logging
import sys
from dataclasses import replace

from airdrop import (
    AlignmentBuffer,
    BallisticsModel,
    Config,
    DroneController,
    DryRunController,
    FlightRecorder,
    MavsdkThread,
    MissionController,
    MissionRunner,
    MissionState,
    ReleaseJudge,
    TargetPoint,
    TargetingConfig,
    TelemetryBroker,
    Waypoint,
    analyze,
    capacity_for,
)

# ----------------------------------------------------------------------
# 场景（改这里）
# ----------------------------------------------------------------------
SYSTEM_ADDRESS = "udpin://0.0.0.0:14540"   # MAVSDK 连 SITL 的地址

# 侦查航线（SITL 里随便一小圈即可；**高度是相对起飞点**）
RECON_ROUTE = (
    Waypoint(lat=47.397742, lon=8.545594, alt_m=50.0),
    Waypoint(lat=47.398742, lon=8.545594, alt_m=50.0),
    Waypoint(lat=47.398742, lon=8.546594, alt_m=50.0),
)
BACKUP_POINT = Waypoint(lat=47.398000, lon=8.546000, alt_m=0.0)
LANDING_ROUTE = (
    Waypoint(lat=47.396500, lon=8.545000, alt_m=40.0),
    Waypoint(lat=47.395000, lon=8.544000, alt_m=0.0),
)
OVERFLY_HEADING_DEG = 0.0
#: 合成目标相对"盘旋点"的 NED 偏移（北, 东, 地）；飞掠航向方向上放 300m
TARGET_OFFSET_NED = (300.0, 0.0, 0.0)
RECORD = True

LOGGER = logging.getLogger("sitl_mission")


def build_config() -> Config:
    base = Config()
    return Config(
        telemetry=replace(base.telemetry, system_address=SYSTEM_ADDRESS),
        routes=replace(
            base.routes,
            recon_route=RECON_ROUTE,
            backup_point=BACKUP_POINT,
            landing_route=LANDING_ROUTE,
        ),
        overfly=replace(base.overfly, heading_deg=OVERFLY_HEADING_DEG),
        # 选唯一结果的规则照实飞配；演练里只有一个类
        targeting=replace(base.targeting, selection_rule="max"),
    ).validated()


class SyntheticTargets:
    """合成目标：以**第一次被问到时的飞机位置**为基准放一个目标点。

    ``MissionRunner`` 只在 HOLD_PROCESS 里问结果，那时飞机已经在盘旋点上——
    于是"目标在盘旋点前方 ``TARGET_OFFSET_NED``"这件事与实飞里"目标在侦查段
    被看到的位置"同构。返回的 :class:`TargetingResult` 是**真的**
    （``analyze()`` 算出来的），不是手搓的假对象。
    """

    def __init__(self, offset_ned: tuple[float, float, float], broker: TelemetryBroker) -> None:
        self._offset = offset_ned
        self._broker = broker
        self._result = None

    def result(self):
        if self._result is None:
            snapshot = self._broker.get_snapshot()
            if snapshot.north_m is None or snapshot.east_m is None:
                return analyze((), TargetingConfig(eps_m=0.75, min_samples=2))
            north = float(snapshot.north_m) + self._offset[0]
            east = float(snapshot.east_m) + self._offset[1]
            # 两次观测才能过 min_samples=2（与实飞"同一目标看到多次"一致）
            points = tuple(
                TargetPoint(
                    north_m=north,
                    east_m=east,
                    capture_timestamp=float(index),
                    frame_index=index,
                    code=56,
                    confidence=0.9,
                )
                for index in (1, 2)
            )
            self._result = analyze(points, TargetingConfig(eps_m=0.75, min_samples=2, selection_rule="max"))
            LOGGER.info(
                "合成目标：NED (%.1f, %.1f)（盘旋点在 (%.1f, %.1f)）",
                north,
                east,
                snapshot.north_m,
                snapshot.east_m,
            )
        return self._result

    def busy(self) -> bool:
        return False


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    config = build_config()

    broker = TelemetryBroker(
        history_interval=config.telemetry.history_interval,
        history_maxlen=config.telemetry.history_maxlen,
    )
    thread = MavsdkThread.from_config(config.telemetry, broker=broker)
    recorder = FlightRecorder(config) if RECORD else None
    on_event = (lambda kind, data: recorder.events.emit(kind, **data)) if recorder else None

    inner = DroneController.from_config(config, thread, on_event=on_event)
    controller: MissionController = DryRunController(inner, on_event=on_event)
    targets = SyntheticTargets(TARGET_OFFSET_NED, broker)
    judge = ReleaseJudge(
        config.drop,
        BallisticsModel(config.ballistics),
        overfly_heading_deg=config.overfly.heading_deg,
        on_event=on_event,
    )
    runner = MissionRunner(
        config,
        controller,
        broker,
        target_result=targets.result,
        target_busy=targets.busy,
        release_judge=judge,          # 判据是真的：会按弹道预测决定何时"投"
        on_event=on_event,
    )
    # 演练没有图传，但记录器需要一个缓冲（它是帧的落盘入口）；这里给它一个空缓冲，
    # 于是 telemetry.jsonl / events.jsonl / flight.log 照常落盘，frames/ 为空。
    buffer = AlignmentBuffer(capacity_for(30.0, 60.0), storage="jpeg")

    exit_code = 0
    try:
        thread.connect()
        LOGGER.info("SITL 已连接：%s", config.telemetry.system_address)
        if recorder is not None:
            recorder.start(broker=broker, buffer=buffer)
            LOGGER.info(
                "演练记录：%s（无图传 → frames/ 为空，遥测/事件/文本日志齐全）",
                recorder.flight_dir,
            )
        state = runner.run()
        LOGGER.info("演练结束：%s", state)
        if state is not MissionState.DONE:
            exit_code = 2
    except KeyboardInterrupt:
        runner.stop("operator_interrupt")
        exit_code = 130
    except Exception:  # noqa: BLE001
        LOGGER.exception("演练异常终止")
        runner.stop("exception")
        exit_code = 1
    finally:
        if recorder is not None:
            recorder.stop()
        thread.stop()

    # ---------------- 演练记录（贴进笔记/issue 用） ----------------
    print("\n===== 演练记录 =====")
    print("地址        :", config.telemetry.system_address)
    print("最终状态    :", runner.state)
    print("状态轨迹    :", " → ".join(str(record.to_state) for record in runner.history))
    print("转移原因    :", [record.reason for record in runner.history])
    print("上传/投放   :", runner.stats.uploads, "/", runner.stats.releases)
    print("判据评估次数:", judge.evaluations, "| 已投放:", judge.released)
    print("ABORT/错误  :", runner.stats.aborts, "/", runner.stats.errors)
    plan = runner.plan
    if plan is not None:
        print("飞掠计划    :", plan.as_dict())
    print("====================\n")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
