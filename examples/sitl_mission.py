'SITL 演练：对着 PX4 SITL 把**控制链路**整条跑一遍（不装弹、不用图传）。\n\n用法（**集中式入口在 `airdrop/run.py`**）::\n\n    # 1) 先起 PX4 SITL（WSL / Linux 里；机型按你的空域选，固定翼用 gazebo 的 x500 之类）\n    #    make px4_sitl gz_x500\n    # 2) 在 Windows 侧跑（MAVSDK 默认连 udpin://0.0.0.0:14540）\n    ./.venv/Scripts/python.exe -m airdrop.run sitl\n    ./.venv/Scripts/python.exe -m airdrop.run sitl --target-offset 300,0,0\n\n本文件是**纯库模块**：`build_config(**覆盖)` / `main(**kwargs)` 的默认值就是文件顶部常量；\n命令行由 `airdrop/run.py` 解析。重依赖都在函数体内导入（`--help` 不加载它们）。\n\n它演练什么、不演练什么\n----------------------\n演练（与实飞同一条代码路径）：状态机推进、侦查航线上传与启动、盘旋（hold）、\n结果坐标 → 飞掠航线规划 → 与降落航线合并上传、投放判据（**真弹道、真风降级**）、\n一次性锁存、投放后的降落段、事件日志与记录五件套。\n\n不演练：**感知**（SITL 里没有相机与目标画面）与**开仓**（SITL 没有 gripper 硬件）。\n所以目标坐标由 ``TARGET_OFFSET_NED`` 合成（相对"盘旋点"），投放走\n:class:`~airdrop.DryRunController`——只记日志、不下发指令，其余全真。\n\n⚠ **演练默认不等起飞**：``REQUIRE_AIRBORNE = False``（地面演练开关）让 ``WAIT_AIRBORNE``\n那道门立即放行——飞机停在停机坪上也能把状态机整条跑完，不必手动 arm/起飞，也不会白等到\n``airborne_timeout_s``（默认 1800s）。放行不是静默的：会记一条 ``airborne_skipped`` 事件\n加一条 WARNING 日志。**正式任务绝不能关这个开关**（见 ``examples/full_mission.py``）——\n在停机坪上进侦查会让 PX4 在地面"追"第一个航点。\n起飞前自检在演练里不注入（``preflight=None`` → 记一条 ``preflight_skipped``）。\n落地后请核对 ``RESULT`` 输出与 ``flights/<本次>/events.jsonl``。\n'

from __future__ import annotations

import logging
import sys
from dataclasses import replace

from airdrop import (
    BallisticsModel,
    Config,
    MissionState,
    PreflightConfig,
    ReleaseJudge,
    TargetingConfig,
    TelemetryBroker,
    Waypoint,
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
#: 降落段。**固定翼有硬性几何要求**（PX4 会照这套判据拒整条任务）：
#: 降落项的前一项必须严格高于落点，且 ``(前项高 − 落点高)/水平距离 ≤ tan(FW_LND_ANG+0.1°)``
#: （默认 8°，约 0.142）。两种来源**二选一**：
#:
#: * ``LAND_PLAN`` 非空：用操作手在 QGC 里画好、另存为 ``.plan`` 的航线
#:   （``routes/land.plan``；里面的"固定翼降落航线"复杂项会在本地展开成 MAVLink 项）。
#:   飞掠段插在它前面，合成一条任务上传。
#: * ``LAND_PLAN`` 为空：用下面的 ``LANDING_ROUTE`` 生成——进场 40m 高、离落点约 390m
#:   ⇒ 下滑角 0.10 ✓（2026-09 之前的 40m/183m = 0.219 会被飞控判"下滑角过陡"）。
LAND_PLAN = "routes/land.plan"
LANDING_ROUTE = (
    Waypoint(lat=47.396500, lon=8.545000, alt_m=40.0),
    Waypoint(lat=47.393600, lon=8.542000, alt_m=0.0),
)
OVERFLY_HEADING_DEG = 0.0
#: 侦查航线谁上传：演练档用 auto（由本包上传）；正式任务用 operator（操作手在 QGC 启动）
RECON_UPLOAD = "auto"
#: 合成目标相对"盘旋点"的 NED 偏移（北, 东, 地）；飞掠航向方向上放 300m
TARGET_OFFSET_NED = (300.0, 0.0, 0.0)
#: **地面演练开关**：``False`` = 不等飞机起飞就放行 ``WAIT_AIRBORNE`` 那道门。
#: 演练常在地面（或只起了 SITL 还没手动起飞）时跑，若保持默认的 ``True``，
#: 状态机会一直等到 ``airborne_timeout_s``（默认 1800s）才失败——演练根本跑不起来。
#: ⚠ 正式任务**必须**是 ``True``（见 ``examples/full_mission.py``）：在停机坪上进侦查
#: 会让 PX4 在地面"追"第一个航点。放过这道门时会记 ``airborne_skipped`` 事件 + WARNING。
REQUIRE_AIRBORNE = False
RECORD = True

LOGGER = logging.getLogger("sitl_mission")


def build_config(
    *,
    system_address: str = SYSTEM_ADDRESS,
    recon_route: tuple[Waypoint, ...] = RECON_ROUTE,
    land_plan: str = LAND_PLAN,
    landing_route: tuple[Waypoint, ...] = LANDING_ROUTE,
    overfly_heading_deg: float = OVERFLY_HEADING_DEG,
    recon_upload: str = RECON_UPLOAD,
    require_airborne: bool = REQUIRE_AIRBORNE,
) -> Config:
    """装配演练配置（**参数都能从外面覆盖**：入口模块/测试按需传值即可）。

    演练场景下起飞前自检**四项全关**（SITL 里没有相机、不加载感知模型、也不接图传），
    侦查航线由本包上传（``recon_upload="auto"``）——这正是"自动测试"那一档。
    正式任务的顺序（载入模型 → 视频自检 → 等在空中 → 操作手上传侦查航线）见
    ``examples/full_mission.py`` 与 :class:`~airdrop.config.PreflightConfig`。

    ⚠ ``require_airborne`` 默认 ``False``（**地面演练开关**，见文件顶部）：演练基本都在
    地面跑，不做这一步的话 ``WAIT_AIRBORNE`` 会一直等到 ``airborne_timeout_s`` 才失败。
    放过那道门会记 ``airborne_skipped`` 事件 + WARNING，**正式任务必须保持默认的 True**。
    """
    base = Config()
    routes = replace(
        base.routes,
        recon_route=tuple(recon_route),
        backup_point=BACKUP_POINT,
    )
    # 降落段二选一：操作手的 QGC 航线，或本地生成的航线（见文件顶部的说明）
    routes = (
        replace(routes, land_plan=land_plan)
        if land_plan
        else replace(routes, landing_route=tuple(landing_route))
    )
    return Config(
        telemetry=replace(base.telemetry, system_address=system_address),
        routes=routes,
        overfly=replace(base.overfly, heading_deg=overfly_heading_deg),
        # 选唯一结果的规则照实飞配；演练里只有一个类
        targeting=replace(base.targeting, selection_rule="max"),
        # 侦查航线由本包上传（自动测试档）：正式任务默认的 "operator" 是等操作手在 QGC 启动
        # 地面演练不等起飞（require_airborne=False），否则会白等到 airborne_timeout_s
        mission=replace(
            base.mission, recon_upload=recon_upload, require_airborne=require_airborne
        ),
        # 起飞前自检**四项全关**：SITL 里没有相机、没有模型、也不接图传。
        # ⚠ 关掉 ≠ 通过——演练里 `MissionRunner(preflight=None)` 会记一条 `preflight_skipped`
        preflight=PreflightConfig(
            load_detector=False,
            load_ocr=False,
            load_camera=False,
            check_video=False,
        ),
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
        # 惰性导入：`airdrop.targeting` 会拉 sklearn，而命令行只需要读本文件的常量
        from airdrop import TargetPoint, analyze

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


def main(
    *,
    target_offset_ned: tuple[float, float, float] = TARGET_OFFSET_NED,
    record: bool = RECORD,
    **config_overrides,
) -> int:
    """跑一次 SITL 演练；``config_overrides`` 原样交给 :func:`build_config`。"""
    # 重依赖在使用时才导入：`python -m airdrop.run sitl --help` 不加载它们
    from airdrop import (
        AlignmentBuffer,
        DroneController,
        DryRunController,
        FlightRecorder,
        MavsdkThread,
        MissionController,
        MissionRunner,
        capacity_for,
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    config = build_config(**config_overrides)

    broker = TelemetryBroker(
        history_interval=config.telemetry.history_interval,
        history_maxlen=config.telemetry.history_maxlen,
    )
    thread = MavsdkThread.from_config(config.telemetry, broker=broker)
    recorder = FlightRecorder(config) if record else None
    on_event = (lambda kind, data: recorder.events.emit(kind, **data)) if recorder else None

    inner = DroneController.from_config(config, thread, on_event=on_event)
    controller: MissionController = DryRunController(inner, on_event=on_event)
    targets = SyntheticTargets(target_offset_ned, broker)
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
        # 投放记录（P12）：用 lambda 延迟取 `recorder.drops`——写入器要 recorder.start()
        # 之后才存在，而 runner 是提前装配好的。演练投的是"干弹"，但记录是真的。
        on_drop=None if recorder is None else (lambda record: recorder.drops.append(record)),
        # 演练不注入预检（没有相机/模型/图传）：状态机记一条 `preflight_skipped` 后放过。
        # ⚠ 正式入口必须注入 `Preflight`（见 examples/full_mission.py）——关掉 ≠ 通过。
        preflight=None,
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
    print("等起飞门    :", "开" if config.mission.require_airborne else "关（地面演练）")
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
