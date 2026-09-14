"""完整任务流程示例：起飞 → 侦查 → 处理 → 飞掠投放 → 降落（记录默认开）。

用法::

    python -m examples.full_mission

**改本文件顶部的常量，不带命令行参数**（项目约定）。

起飞前检查清单
--------------
1. ``camera_calib.json`` 已产出（``python -m tools.calibrate``）——没有它相机外参是
   默认值，坐标会系统性偏；标定出的画面/遥测时间差要回填到 ``TELEMETRY_LAG_S``。
2. ``RECON_ROUTE`` / ``BACKUP_POINT`` / ``LANDING_ROUTE`` 按空域填好（WGS84；
   **高度是相对起飞点的高度**，不是海拔）。
3. ``OVERFLY_HEADING_DEG`` 按空域定；高度 20m 与段长 200m 是**待实验验证**的默认值。
4. PX4 侧配置好 gripper 输出（``GripperConfig.instance``）；不装弹演练时把
   ``DRY_RUN`` 置 True——投放那一步只记日志，其余全走真链路。
5. 弹道参数（质量/阻力系数/迎风面积）按投放试验微调过（默认值是 350ml 水瓶的估计）。
   每次投放的**投放瞬间状态**（位置/速度/姿态/风）会自动写进飞行目录的
   ``drops.jsonl``；落地后量出实际落点、填进同目录的 ``impacts.jsonl``，
   再跑 ``python -m tools.fit_ballistics`` 即可反演出这组参数（见
   ``airdrop.ballistics.fit``）。

线程与生命周期
--------------
主线程跑状态机（``MissionRunner.run`` 阻塞），其余各在自己的线程/进程里：MAVSDK
工作线程、图传采集线程、感知线程、OCR 进程池、记录器的两个写盘线程。Ctrl-C 会走
``finally``：先 ``runner.stop()``（状态机按 ABORT 处理并下安全动作），再依次停感知、
图传、记录器与 MAVSDK 线程。
"""

from __future__ import annotations

import logging
import sys
from dataclasses import replace

from airdrop import (
    HM30_DEFAULT_RTSP,
    AlignmentBuffer,
    AlignmentWriter,
    BallisticsModel,
    Config,
    DroneController,
    DryRunController,
    FlightRecorder,
    FrameTelemetryAligner,
    Hm30VideoSource,
    MavsdkThread,
    MissionController,
    MissionRunner,
    MissionState,
    PerceptionTargetSource,
    PerceptionWorker,
    ReleaseJudge,
    TargetTracker,
    TelemetryBroker,
    Waypoint,
    capacity_for,
)

# ----------------------------------------------------------------------
# 配置（改这里）
# ----------------------------------------------------------------------
RTSP_URL = HM30_DEFAULT_RTSP     # HM30 图传地址（SIYI 相机默认值）
TELEMETRY_LAG_S = 0.15           # 画面-遥测链路延时：**标定后改成实测值**

RECON_ROUTE = (
    Waypoint(lat=47.397742, lon=8.545594, alt_m=60.0),
    Waypoint(lat=47.398742, lon=8.545594, alt_m=60.0),
    Waypoint(lat=47.398742, lon=8.546594, alt_m=60.0),
)
BACKUP_POINT = Waypoint(lat=47.398000, lon=8.546000, alt_m=0.0)
LANDING_ROUTE = (
    Waypoint(lat=47.396000, lon=8.545000, alt_m=40.0),
    Waypoint(lat=47.395000, lon=8.544000, alt_m=0.0),
)

OVERFLY_HEADING_DEG = 0.0        # 飞掠航向（按空域定）
PERCEPTION_MODE = "ocr"          # ocr（YOLO + 读数）或 cls12（12 类直出）
SELECTION_RULE = "median"        # 跨候选类取编号中位数还是最大值
DRY_RUN = False                  # True = 投放只记日志（不装弹演练 / SITL）
RECORD = True                    # 落盘五件套

LOGGER = logging.getLogger("full_mission")


def build_config() -> Config:
    """把上面的常量装配成一份完整配置（并做取值域校验）。

    只改需要改的字段，其余沿用默认值——``dataclasses.replace`` 让"改哪几处"一目了然。
    """
    base = Config()
    return Config(
        video=replace(base.video, url=RTSP_URL, telemetry_lag=TELEMETRY_LAG_S),
        routes=replace(
            base.routes,
            recon_route=RECON_ROUTE,
            backup_point=BACKUP_POINT,
            landing_route=LANDING_ROUTE,
        ),
        overfly=replace(base.overfly, heading_deg=OVERFLY_HEADING_DEG),
        perception=replace(base.perception, mode=PERCEPTION_MODE),
        targeting=replace(base.targeting, selection_rule=SELECTION_RULE),
    ).validated()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    config = build_config()

    # 1) 遥测：MAVSDK 工作线程 + 线程安全代理（历史 10Hz×1200 ≈ 2 分钟）
    broker = TelemetryBroker(
        history_interval=config.telemetry.history_interval,
        history_maxlen=config.telemetry.history_maxlen,
    )
    thread = MavsdkThread.from_config(config.telemetry, broker=broker)

    # 2) 记录器先建起来：之后所有模块的事件/检测/投放都在它的飞行目录里（业务模块无感）
    recorder = FlightRecorder(config) if RECORD else None
    on_event = (
        (lambda kind, data: recorder.events.emit(kind, **data)) if recorder else None
    )
    # ⚠ 必须用 lambda 延迟取 recorder.drops：写入器要等 recorder.start() 之后才存在
    on_drop = (lambda record: recorder.drops.append(record)) if recorder else None

    # 3) 控制器：任务级控制；演练模式把"投放"换成日志，其余照真下发
    inner = DroneController.from_config(config, thread, on_event=on_event)
    controller: MissionController = (
        DryRunController(inner, on_event=on_event) if DRY_RUN else inner
    )

    # 4) 图传 → 对齐 → 环形缓冲（一帧不落的那条路：sink 逐帧）
    camera = config.camera.load_model()
    source = Hm30VideoSource(config.video)
    aligner = FrameTelemetryAligner(
        broker, max_wait=config.align.max_wait, mode=config.align.mode
    )
    buffer = AlignmentBuffer(capacity_for(config.video.fps or 30.0, 180.0), storage="jpeg")
    writer = AlignmentWriter(buffer, aligner)

    # 5) 感知 → 坐标解算 → 目标统计（回放与实飞共用这一条）
    worker = PerceptionWorker(config.perception.to_pipeline_config(), buffer=buffer)
    tracker = TargetTracker(config, camera=camera, on_event=on_event)
    targets = PerceptionTargetSource(worker=worker, tracker=tracker)

    # 6) 投放判据：弹道 + 一次性锁存（每拍预测落点，越过目标保底投放）
    judge = ReleaseJudge(
        config.drop,
        BallisticsModel(config.ballistics),
        overfly_heading_deg=config.overfly.heading_deg,
        on_event=on_event,
    )

    # 7) 状态机：INIT → RECON → HOLD_PROCESS → OVERFLY → LAND → DONE
    runner = MissionRunner(
        config,
        controller,
        broker,
        target_result=targets.result,
        target_busy=targets.busy,
        release_judge=judge,
        on_event=on_event,
        on_drop=on_drop,
    )

    exit_code = 0
    try:
        thread.connect()
        LOGGER.info("飞控已连接，NED 原点就绪后开始任务")
        if recorder is not None:
            recorder.start(broker=broker, buffer=buffer)
        worker.start()
        source.add_sink(writer)
        source.start()
        state = runner.run()
        LOGGER.info(
            "任务结束：%s（上传 %s 次、投放 %s 次）",
            state,
            runner.stats.uploads,
            runner.stats.releases,
        )
        if state is not MissionState.DONE:
            exit_code = 2
    except KeyboardInterrupt:
        LOGGER.warning("操作手中断，按 ABORT 处理（下安全动作）")
        runner.stop("operator_interrupt")
        exit_code = 130
    except Exception:  # noqa: BLE001 - 起飞前把错因写进日志再退出
        LOGGER.exception("任务异常终止")
        runner.stop("exception")
        exit_code = 1
    finally:
        source.stop()
        worker.stop()
        if recorder is not None:
            recorder.stop()
        thread.stop()
        LOGGER.info(
            "收尾完成：感知 %s，目标点 %s，记录目录 %s",
            worker.stats,
            tracker.stats,
            recorder.flight_dir if recorder else "（未开记录）",
        )
        if runner.drops:
            LOGGER.info(
                "本次投放 %d 次，投放瞬间状态已记进 %s；"
                "量完实际落点填 impacts.jsonl 后跑 python -m tools.fit_ballistics 反演弹道参数",
                len(runner.drops),
                "drops.jsonl" if recorder else "（未开记录：本次没落盘）",
            )
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
