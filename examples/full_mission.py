"""完整任务流程示例：自检 → 等起飞 → 侦查 → 处理 → 飞掠投放 → 降落（记录默认开）。

用法（集中式入口在 airdrop/run.py，参数不要写在这里）::

    ./.venv/Scripts/python.exe -m airdrop.run full-mission --help
    ./.venv/Scripts/python.exe -m airdrop.run full-mission --no-video --dry-run

本文件是纯库模块：顶部的常量就是默认值，:func:`build_config` /
:func:`main` 的每个关键字都能从外面覆盖（命令行由 airdrop/run.py 解析）。
重依赖（cv2 / mavsdk / ultralytics / torch）都在函数体内导入，所以
import examples.full_mission 与 --help 都不会加载它们。

起飞前检查清单
--------------
1. ``camera_calib.json`` 已产出（``python -m airdrop.run calibrate``）——没有它相机外参是
   默认值，坐标会系统性偏；标定出的画面/遥测时间差要回填到 ``TELEMETRY_LAG_S``。
2. ``RECON_ROUTE`` / ``BACKUP_POINT`` / ``LANDING_ROUTE`` 按空域填好（WGS84；
   高度是相对起飞点的高度，不是海拔）；降落段也可以用操作手的
   ``routes/land.plan``（``LAND_PLAN`` 非空即走 plan，两者二选一）。
3. ``OVERFLY_HEADING_DEG`` 按空域定；高度 20m 与段长 200m 是待实验验证的默认值。
4. 飞机由操作手手动起飞（RC/手抛/手动起飞指令）：本包要等到"飞机真的在空中"
   （``WAIT_AIRBORNE``，见 :class:`~airdrop.config.MissionConfig`）才进侦查，
   停机坪上不会上传任何任务。只有地面演练/测试才把 ``REQUIRE_AIRBORNE`` 置 False
   （那会记一条 ``airborne_skipped`` 事件 + WARNING，正式任务不要关）。
5. 侦查航线由操作手在 QGC 上传并启动（``recon_upload="operator"``，默认）：
   本包只等它开始、然后监视进度；自动测试才改成 ``"auto"``。
6. PX4 侧配置好 gripper 输出（``GripperConfig.instance``）；不装弹演练时把
   ``DRY_RUN`` 置 True（命令行：``--dry-run``）——投放那一步只记日志，其余全走真链路。
7. 弹道参数（质量/阻力系数/迎风面积）按投放试验微调过（默认值是 350ml 水瓶的估计）。
   每次投放的投放瞬间状态（位置/速度/姿态/风）会自动写进飞行目录的
   ``drops.jsonl``；落地后量出实际落点、填进同目录的 ``impacts.jsonl``，
   再跑 ``python -m airdrop.run fit-ballistics`` 即可反演出这组参数。

线程与生命周期
--------------
主线程跑状态机（``MissionRunner.run`` 阻塞），其余各在自己的线程/进程里：MAVSDK
工作线程、图传采集线程、感知线程、OCR 进程池、记录器的两个写盘线程。Ctrl-C 会走
``finally``：先 ``runner.stop()``（状态机按 ABORT 处理并下安全动作），再依次停感知、
图传、记录器与 MAVSDK 线程。

``USE_VIDEO=False``（命令行 ``--no-video``）时不接图传：跳过图传/感知/坐标解算，
状态机照常跑（HOLD_PROCESS 取不到结果就走备用点），预检里的四项也相应关掉。
"""

from __future__ import annotations

import logging
import sys
from dataclasses import replace

from airdrop import HM30_DEFAULT_RTSP, Config, PreflightConfig, Waypoint

# ----------------------------------------------------------------------
# 配置（改这里）
# ----------------------------------------------------------------------
SYSTEM_ADDRESS: str | None = None  # None = 用 TelemetryConfig 的默认地址
RTSP_URL = HM30_DEFAULT_RTSP  # HM30 图传地址（SIYI 相机默认值）
TELEMETRY_LAG_S = 0.15  # 画面-遥测链路延时：标定后改成实测值

RECON_ROUTE = (
    Waypoint(lat=47.397742, lon=8.545594, alt_m=60.0),
    Waypoint(lat=47.398742, lon=8.545594, alt_m=60.0),
    Waypoint(lat=47.398742, lon=8.546594, alt_m=60.0),
)
BACKUP_POINT = Waypoint(lat=47.398000, lon=8.546000, alt_m=0.0)
#: 降落段。固定翼有硬性几何要求（PX4 会照这套判据拒整条任务）：
#: 降落项紧前一项必须严格高于落点，且 **下滑斜率（tan，垂直差/水平距离）**
#: ≤ tan(FW_LND_ANG+0.1°)（默认 8° ⇒ 上限约 0.142）。
#: 下面的 40m/约 394m ⇒ tan≈0.102（约 5.8°）能过；**换空域时按这条算一遍**
#: （`build_drop_mission` 会在规划阶段预检，不合格直接报错，不会等到飞控拒）。
LANDING_ROUTE = (
    Waypoint(lat=47.396500, lon=8.545000, alt_m=40.0),
    Waypoint(lat=47.393600, lon=8.542000, alt_m=0.0),
)
#: 非空 = 用操作手在 QGC 画好的降落航线（与 LANDING_ROUTE 二选一，见 RoutesConfig）
LAND_PLAN = ""

OVERFLY_HEADING_DEG = 0.0  # 飞掠航向（按空域定）
PERCEPTION_MODE = "ocr"  # ocr（YOLO + 读数）或 cls12（12 类直出）
SELECTION_RULE = "median"  # 跨候选类取编号中位数还是最大值
#: 侦查航线谁上传：operator = 操作手在 QGC 上传并启动（正式任务）；auto = 本包上传（自动测试）
RECON_UPLOAD = "operator"
DRY_RUN = False  # True = 投放只记日志（不装弹演练 / SITL）
RECORD = True  # 写入磁盘的五个记录文件
USE_VIDEO = True  # False = 本架次不接图传（无相机/图传时也能跑状态机）
REQUIRE_AIRBORNE = True  # False 只给地面演练/测试（见 MissionConfig，正式任务必须 True）

LOGGER = logging.getLogger("full_mission")

#: ``--no-preflight`` 用的"全关"自检配置（关掉 ≠ 通过：每项都会记 ok=null 的事件）
PREFLIGHT_OFF = PreflightConfig(
    load_detector=False,
    load_ocr=False,
    load_camera=False,
    check_video=False,
)


def build_config(
    *,
    system_address: str | None = SYSTEM_ADDRESS,
    rtsp_url: str = RTSP_URL,
    telemetry_lag_s: float = TELEMETRY_LAG_S,
    recon_route: tuple[Waypoint, ...] = RECON_ROUTE,
    landing_route: tuple[Waypoint, ...] | None = None,
    land_plan: str = LAND_PLAN,
    overfly_heading_deg: float = OVERFLY_HEADING_DEG,
    perception_mode: str = PERCEPTION_MODE,
    selection_rule: str = SELECTION_RULE,
    recon_upload: str = RECON_UPLOAD,
    use_video: bool = USE_VIDEO,
    preflight: PreflightConfig | None = None,
    require_airborne: bool = REQUIRE_AIRBORNE,
) -> Config:
    """把上面的常量装配成一份完整配置（并做取值域校验）。

    参数都能从外面覆盖（airdrop/run.py 与测试按需传值），默认值就是文件顶部的常量。
    只改需要改的字段，其余沿用默认值——``dataclasses.replace`` 让"改哪几处"一目了然。

    ``recon_upload`` 默认 ``"operator"``：正式任务的侦查航线由操作手在 QGC 上传并启动，
    本包只等它开始（自动测试才用 ``"auto"``）。起飞前自检（载入模型 → 视频自检）的开关
    全在 ``config.preflight`` 里：``preflight=None`` 时默认四项全开；``use_video=False``
    （无图传演练）时自动全关——没有相机与图传，那四项本来就无从检查。
    等飞机在空中是 ``WAIT_AIRBORNE`` 那道门（``MissionConfig.airborne_timeout_s`` /
    ``airborne_alt_m``），``require_airborne=False`` 是它的地面演练临时放行开关。
    """
    base = Config()
    if land_plan and landing_route:
        raise ValueError(
            "降落段二选一：landing_route（配置航点）与 land_plan（QGC 航线）不能同时给"
        )
    routes = replace(
        base.routes,
        recon_route=tuple(recon_route),
        backup_point=BACKUP_POINT,
    )
    routes = (
        replace(routes, land_plan=land_plan)
        if land_plan
        else replace(routes, landing_route=tuple(landing_route or LANDING_ROUTE))
    )
    if preflight is None:
        preflight = PREFLIGHT_OFF if not use_video else base.preflight
    telemetry = (
        base.telemetry
        if system_address is None
        else replace(base.telemetry, system_address=system_address)
    )
    return Config(
        telemetry=telemetry,
        video=replace(base.video, url=rtsp_url, telemetry_lag=telemetry_lag_s),
        routes=routes,
        overfly=replace(base.overfly, heading_deg=overfly_heading_deg),
        perception=replace(base.perception, mode=perception_mode),
        targeting=replace(base.targeting, selection_rule=selection_rule),
        mission=replace(base.mission, recon_upload=recon_upload, require_airborne=require_airborne),
        preflight=preflight,
    ).validated()


def main(
    *,
    use_video: bool = USE_VIDEO,
    dry_run: bool = DRY_RUN,
    record: bool = RECORD,
    **config_overrides,
) -> int:
    """跑一次完整任务；``config_overrides`` 原样交给 :func:`build_config`。"""
    # 重依赖在使用时才导入：`python -m airdrop.run full-mission --help` 不加载它们
    from airdrop import (
        AlignmentBuffer,
        AlignmentWriter,
        BallisticsModel,
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
        Preflight,
        ReleaseJudge,
        TargetTracker,
        TelemetryBroker,
        capacity_for,
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    config = build_config(use_video=use_video, **config_overrides)

    # 1) 遥测：MAVSDK 工作线程 + 线程安全代理（历史 10Hz×1200 ≈ 2 分钟）
    broker = TelemetryBroker(
        history_interval=config.telemetry.history_interval,
        history_maxlen=config.telemetry.history_maxlen,
    )
    thread = MavsdkThread.from_config(config.telemetry, broker=broker)

    # 2) 记录器先建起来：之后所有模块的事件/检测/投放都在它的飞行目录里（业务模块无感）
    recorder = FlightRecorder(config) if record else None
    on_event = (lambda kind, data: recorder.events.emit(kind, **data)) if recorder else None
    # ⚠ 必须用 lambda 延迟取 recorder.drops：写入器要等 recorder.start() 之后才存在
    on_drop = (
        (lambda record_: recorder.drops.append(record_))  # noqa: PLW0108 - 必须延迟到 start() 之后取值
        if recorder
        else None
    )

    # 3) 控制器：任务级控制；演练模式把"投放"换成日志，其余照真下发
    inner = DroneController.from_config(config, thread, on_event=on_event)
    controller: MissionController = DryRunController(inner, on_event=on_event) if dry_run else inner

    # 4) 图传 → 对齐 → 环形缓冲（一帧不落的那条路：sink 逐帧）
    #    --no-video（use_video=False）时不接图传/感知：状态机照常跑，
    #    HOLD_PROCESS 取不到结果就走备用点；记录器仍需要一个空缓冲（frames/ 为空）。
    buffer = AlignmentBuffer(capacity_for(30.0, 180.0), storage="jpeg")
    camera = source = writer = worker = tracker = targets = None
    if use_video:
        camera = config.camera.load_model()
        source = Hm30VideoSource(config.video)
        aligner = FrameTelemetryAligner(
            broker, max_wait=config.align.max_wait, mode=config.align.mode
        )
        writer = AlignmentWriter(buffer, aligner)

        # 5) 感知 → 坐标解算 → 目标统计（回放与实飞共用这一条）
        worker = PerceptionWorker(config.perception.to_pipeline_config(), buffer=buffer)
        tracker = TargetTracker(config, camera=camera, on_event=on_event)
        targets = PerceptionTargetSource(
            worker=worker,
            tracker=tracker,
            # 检出写进飞行记录（detections.jsonl）——不接的话这个文件永远是空的
            on_detection=(
                (lambda detection: recorder.detections.append(detection))  # noqa: PLW0108 - 延迟取 recorder
                if recorder is not None
                else None
            ),
        )

    # 6) 投放判据：弹道 + 一次性锁存（每拍预测落点，越过目标后强制投放）
    judge = ReleaseJudge(
        config.drop,
        BallisticsModel(config.ballistics),
        overfly_heading_deg=config.overfly.heading_deg,
        on_event=on_event,
    )

    # 7) 状态机：INIT（起飞前自检）→ WAIT_AIRBORNE → RECON → HOLD_PROCESS → OVERFLY → LAND → DONE
    #
    # 起飞前自检（PREFLIGHT）在这里真装配：每一项都用现成组件，开关全部来自
    # config.preflight（默认四项全开；关掉的项记一条 preflight 事件、ok=null——
    # 关掉 ≠ 通过，正式任务起飞前请核对那几条事件）：
    #   ① detector：载入 YOLO 权重（就是感知线程要用的那个 Detector 实例）；
    #   ② ocr：起感知工作线程 + OCR 进程池（OCR 权重在子进程里载入）；
    #      ⚠ 关掉 load_ocr 等于不启动感知（正式任务别关）；
    #   ③ camera：载入相机标定（camera 那份，与 TargetTracker 用的是同一个文件）；
    #   ④ video：在 video_probe_s 窗口内等够 video_min_frames 帧，并用已载入的标定
    #      核对最新帧分辨率一致。
    # 之后 WAIT_AIRBORNE 等"飞机真的在空中"（in_air，取不到时看相对高度），
    # 最后进 RECON：MissionConfig.recon_upload="operator"（默认）表示**由操作手在
    # QGC 上传并启动侦查航线**，本包不上传、只等它开始再监视进度；自动测试才用 "auto"。
    def _load_camera() -> str:
        """报告相机标定（复用装配时已载入的那份），返回一句可读的细节。

        细节是给 ``preflight`` 事件用的：不要把 ``CameraModel`` 直接返回——它是
        带 numpy 数组的 dataclass，``str()`` 出来是一大坨多行 repr，会把事件日志搞脏。
        ``source`` 为 None 说明用的是默认模型（标定文件缺失/损坏）。
        """
        # 视频链路装配时已经载入过一份（见上面的 camera）：复用同一个实例，
        # 不重复读文件，也避免"预检检查的模型与解算用的不是同一个"的错觉。
        model = camera if camera is not None else config.camera.load_model()
        origin = model.source or "默认模型（标定缺失/损坏，坐标不可信）"
        return f"{config.camera.calib_file} → {origin}，标定尺寸 {model.image_size}"

    loaders = {}
    if worker is not None:
        loaders = {
            "detector": worker.detector.load,
            "ocr": worker.start,
            "camera": _load_camera,
        }
    preflight = Preflight(
        config.preflight,
        model_loaders=loaders,
        video_source=source,  # 真实图传源：读 stats.frames 计数、latest() 取最新帧
        camera_model=camera,  # 已载入的标定：核对画面分辨率是否与标定一致
        on_event=on_event,
    )
    runner = MissionRunner(
        config,
        controller,
        broker,
        target_result=None if targets is None else targets.result,
        target_busy=None if targets is None else targets.busy,
        release_judge=judge,
        on_event=on_event,
        on_drop=on_drop,
        preflight=preflight,
    )

    exit_code = 0
    try:
        thread.connect()
        LOGGER.info("飞控已连接；起飞前自检 → 等在空中 → 侦查（航线由操作手上传）")
        if recorder is not None:
            recorder.start(broker=broker, buffer=buffer)
        if source is not None and writer is not None:
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
    except Exception:
        LOGGER.exception("任务异常终止")
        runner.stop("exception")
        exit_code = 1
    finally:
        if source is not None:
            source.stop()
        if worker is not None:
            worker.stop()
        if recorder is not None:
            recorder.stop()
        thread.stop()
        LOGGER.info(
            "收尾完成：感知 %s，目标点 %s，记录目录 %s",
            worker.stats if worker is not None else "（未接图传）",
            tracker.stats if tracker is not None else "（未接图传）",
            recorder.flight_dir if recorder else "（未开记录）",
        )
        if runner.drops:
            LOGGER.info(
                "本次投放 %d 次，投放瞬间状态已记进 %s；"
                "量完实际落点填 impacts.jsonl 后跑 python -m airdrop.run fit-ballistics 反演弹道参数",
                len(runner.drops),
                "drops.jsonl" if recorder else "（未开启记录：本次未写入磁盘）",
            )
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
