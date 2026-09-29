"""SITL 演练：对着 PX4 SITL 把控制链路整条跑一遍（不装弹、不用图传）。

用法（集中式入口在 airdrop/run.py）::

    # 1) 先起 PX4 SITL，并让 Gazebo 用本项目的 CUADC 赛区世界（WSL / Linux 里）：
    #    把 sim/worlds/cuadc 挂进 PX4 的 worlds 目录与资源路径，例如：
    #      ln -sf <repo>/sim/worlds/cuadc/cuadc_recon_strike_r2.sdf \
    #             ~/PX4-Autopilot/Tools/simulation/gz/worlds/
    #      export GZ_SIM_RESOURCE_PATH=<repo>/sim/worlds/cuadc:$GZ_SIM_RESOURCE_PATH
    #      PX4_GZ_WORLD=cuadc_recon_strike_r2 make px4_sitl gz_rc_cessna_down_cam
    #    （默认机型是带下视相机的 gz_rc_cessna_down_cam；一条命令的版本：
    #      bash sim/run_sitl.sh r2。世界布局、规则出处与完整启动说明见
    #      docs/simulation_world.md）
    # 2) 在 Windows 侧跑（MAVSDK 默认连 udpin://0.0.0.0:14540）
    ./.venv/Scripts/python.exe -m airdrop.run sitl
    ./.venv/Scripts/python.exe -m airdrop.run sitl --target-offset 0,-20,0

本文件是纯库模块：build_config(**覆盖) / main(**kwargs) 的默认值就是文件顶部常量；
命令行由 airdrop/run.py 解析。重依赖都在函数体内导入（--help 不加载它们）。

⚠ SITL 演练已于 2026-09 在开发机跑通（WSL + PX4 SITL ``gz_rc_cessna`` 固定翼：
``RECON → HOLD_PROCESS → OVERFLY → LAND → DONE``，2 次上传 / 1 次投放 / 0 错误）。
那次记录在"等待起飞"这道检查加入之前，重跑时按新顺序：先手动起飞，
等本包上传侦查航线。

本演练飞的是哪个赛区
--------------------
默认航点对着 ``sim/worlds/cuadc/cuadc_recon_strike_r2.sdf``（ENU 原点 = 起降区中心 =
跑道中点；跑道沿 x 轴 200m x 30m，A / B 两个 60 x 60m 目标区在起飞线两端、
各距起降区约 200m：**A 区（蓝）在 +x、B 区（红）在 -x**）：
侦查段沿 N=25 / N=-27 两排在 A 区上空扫两遍，盘旋点停在 A 区"中位数"天井
（``r2`` 世界的 56 号）以东 20m；合成目标 = 盘旋点 + ``TARGET_OFFSET_NED``
（正西 20m）正好落在那座天井上。
⚠ 天井在区内的位置是**随机的**（规则 3.3），这里写死的是 ``seed=0``（默认）那一批：
换世界或换种子后天井坐标会变，改这一处常量即可（``tests/test_world.py`` 会核验
这批常量对准的是入库世界的 56 号天井）。

它演练什么、不演练什么
----------------------
演练（与实飞同一条代码路径）：状态机推进、侦查航线上传与启动、盘旋（hold）、
结果坐标 → 飞掠航线规划 → 与降落航线合并上传、投放判据（真弹道、真风降级）、
一次性锁存、投放后的降落段、事件日志与记录五个记录文件。

不演练：感知（SITL 里没有相机与目标画面）与开仓（SITL 没有 gripper 硬件）。
所以目标坐标由 ``TARGET_OFFSET_NED`` 合成（相对"盘旋点"），投放走
:class:`~airdrop.DryRunController`——只记日志、不下发指令，其余全真。

⚠ 演练默认不等起飞：``REQUIRE_AIRBORNE = False``（地面演练开关）让 ``WAIT_AIRBORNE``
那道门立即放行——飞机停在停机坪上也能把状态机整条跑完，不必手动 arm/起飞，也不会空等到
``airborne_timeout_s``（默认 1800s）。放行不是静默的：会记一条 ``airborne_skipped`` 事件
加一条 WARNING 日志。正式任务绝不能关这个开关（见 ``examples/full_mission.py``）——
在停机坪上进侦查会让 PX4 在地面"追"第一个航点。
起飞前自检在演练里不注入（``preflight=None`` → 记一条 ``preflight_skipped``）。
落地后请核对 ``RESULT`` 输出与 ``flights/<本次>/events.jsonl``。
"""

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
SYSTEM_ADDRESS = "udpin://0.0.0.0:14540"  # MAVSDK 连 SITL 的地址

# 赛区：sim/worlds/cuadc/cuadc_recon_strike_r2.sdf（ENU 原点 = 起降区中心 = 跑道中点，
# 跑道沿 x 轴；A / B 两个 60x60m 目标区在起飞线两端 (±200, 0)，A 蓝在 +x、B 红在 -x；
# 天井在区内随机摆放，下面是 seed=0（默认）那一批的坐标）。
# 侦查航线：起飞后沿 N=16 / N=-16 两排在 A 区上空扫两遍（50m 相对高度），
# 终点停在 A 区"中位数"天井（r2 世界的 56 号，E=200.30, N=-10.80）以东 20m ——
# 于是合成目标（盘旋点 + TARGET_OFFSET_NED，正西 20m）正好落在那座天井上。
RECON_ROUTE = (
    Waypoint(lat=47.3979710405, lon=8.5481507733, alt_m=50.0),  # 起飞点：跑道东侧 150m
    Waypoint(lat=47.3981149469, lon=8.5484819519, alt_m=50.0),  # A 区西北外侧 (E=175, N=16)
    Waypoint(lat=47.3981149305, lon=8.5491840390, alt_m=50.0),  # 沿 N=16 向东扫过 A 区
    Waypoint(lat=47.3978271054, lon=8.5491840225, alt_m=50.0),  # 东端南转 (E=228, N=-16)
    Waypoint(lat=47.3978738455, lon=8.5490820474, alt_m=50.0),  # 向西收在盘旋点
)
#: 没有目标时用的备用点：A 区中心 (E=200, N=0)
BACKUP_POINT = Waypoint(lat=47.3979710271, lon=8.5488131178, alt_m=0.0)
#: 降落段。固定翼有硬性几何要求（PX4 会照这套判据拒整条任务）：
#: 降落项的前一项必须严格高于落点，且 ``(前项高 − 落点高)/水平距离 ≤ tan(FW_LND_ANG+0.1°)``
#: （默认 8°，约 0.142）。两种来源二选一：
#:
#: * ``LAND_PLAN`` 非空：用操作手在 QGC 里画好、另存为 ``.plan`` 的航线
#:   （``routes/land.plan``；里面的"固定翼降落航线"复杂项会在本地展开成 MAVLink 项）。
#:   飞掠段插在它前面，合成一条任务上传。入库的 ``routes/land.plan`` 已对准本赛区跑道
#:   （进场 30m 高、落点在跑道西段，下滑斜率 tan≈0.103）。
#: * ``LAND_PLAN`` 为空：用下面的 ``LANDING_ROUTE`` 生成——进场 20m 高、离落点 270m
#:   ⇒ 下滑斜率 tan≈0.074 ✓（2026-09 之前的 40m/183m = tan 0.219 ≈ 12.4° 会被飞控判
#:   "下滑角过陡"）。
LAND_PLAN = "routes/land.plan"
LANDING_ROUTE = (
    Waypoint(lat=47.3979709888, lon=8.5421896728, alt_m=20.0),  # 进场：跑道西侧 300m、20m 高
    Waypoint(lat=47.3979710570, lon=8.5457663331, alt_m=0.0),  # 落点：跑道中心线 x=-30m
)
OVERFLY_HEADING_DEG = 0.0
#: 侦查航线谁上传：演练档用 auto（由本包上传）；正式任务用 operator（操作手在 QGC 启动）
RECON_UPLOAD = "auto"
#: 合成目标相对"盘旋点"的 NED 偏移（北, 东, 地）：正西 20m = A 区中位数天井（r2 = 56 号）
TARGET_OFFSET_NED = (0.0, -20.0, 0.0)
#: 地面演练开关：``False`` = 不等飞机起飞就放行 ``WAIT_AIRBORNE`` 那道门。
#: 演练常在地面（或只起了 SITL 还没手动起飞）时跑，若保持默认的 ``True``，
#: 状态机会一直等到 ``airborne_timeout_s``（默认 1800s）才失败——演练根本跑不起来。
#: ⚠ 正式任务必须是 ``True``（见 ``examples/full_mission.py``）：在停机坪上进侦查
#: 会让 PX4 在地面"追"第一个航点。放行时会记 ``airborne_skipped`` 事件 + WARNING。
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
    """装配演练配置（参数都能从外面覆盖：入口模块/测试按需传值即可）。

    演练场景下起飞前自检四项全关（SITL 里没有相机、不加载感知模型、也不接图传），
    侦查航线由本包上传（``recon_upload="auto"``）——这正是"自动测试"那一档。
    正式任务的顺序（载入模型 → 视频自检 → 等在空中 → 操作手上传侦查航线）见
    ``examples/full_mission.py`` 与 :class:`~airdrop.config.PreflightConfig`。

    ⚠ ``require_airborne`` 默认 ``False``（地面演练开关，见文件顶部）：演练基本都在
    地面跑，不做这一步的话 ``WAIT_AIRBORNE`` 会一直等到 ``airborne_timeout_s`` 才失败。
    放过那道门会记 ``airborne_skipped`` 事件 + WARNING，正式任务必须保持默认的 True。
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
        # 地面演练不等起飞（require_airborne=False），否则会空等到 airborne_timeout_s
        mission=replace(base.mission, recon_upload=recon_upload, require_airborne=require_airborne),
        # 起飞前自检四项全关：SITL 里没有相机、没有模型、也不接图传。
        # ⚠ 关掉 ≠ 通过——演练里 MissionRunner(preflight=None) 会记一条 preflight_skipped
        preflight=PreflightConfig(
            load_detector=False,
            load_ocr=False,
            load_camera=False,
            check_video=False,
        ),
    ).validated()


class SyntheticTargets:
    """合成目标：以第一次被问到时的飞机位置为基准放一个目标点。

    ``MissionRunner`` 只在 HOLD_PROCESS 里问结果，那时飞机已经在盘旋点上——
    于是"目标在盘旋点前方 ``TARGET_OFFSET_NED``"这件事与实飞里"目标在侦查段
    被看到的位置"同构。返回的 :class:`TargetingResult` 是真的
    （``analyze()`` 算出来的），不是手搓的假对象。
    """

    def __init__(self, offset_ned: tuple[float, float, float], broker: TelemetryBroker) -> None:
        self._offset = offset_ned
        self._broker = broker
        self._result = None

    def result(self):
        # 惰性导入：airdrop.targeting 会拉 sklearn，而命令行只需要读本文件的常量
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
            self._result = analyze(
                points, TargetingConfig(eps_m=0.75, min_samples=2, selection_rule="max")
            )
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
        release_judge=judge,  # 判据是真的：会按弹道预测决定何时"投"
        on_event=on_event,
        # 投放记录（P12）：用 lambda 延迟取 recorder.drops——写入器要 recorder.start()
        # 之后才存在，而 runner 是提前装配好的。演练投的是"干弹"，但记录是真的。
        on_drop=None if recorder is None else (lambda record: recorder.drops.append(record)),  # noqa: PLW0108 - 同上，延迟取值
        # 演练不注入预检（没有相机/模型/图传）：状态机记一条 preflight_skipped 后放过。
        # ⚠ 正式入口必须注入 Preflight（见 examples/full_mission.py）——关掉 ≠ 通过。
        preflight=None,
    )
    # 演练没有图传，但记录器需要一个缓冲（它是帧写入磁盘的入口）；这里给它一个空缓冲，
    # 于是 telemetry.jsonl / events.jsonl / flight.log 照常写入磁盘，frames/ 为空。
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
    except Exception:
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
    print("等起飞      :", "开" if config.mission.require_airborne else "关（地面演练）")
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
