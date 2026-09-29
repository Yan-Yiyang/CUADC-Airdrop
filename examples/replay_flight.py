"""回放全链路示例：一份素材反复迭代"识别 → 坐标 → 统计 → 航线"，全程不碰硬件。

用法（集中式入口在 airdrop/run.py）::

    ./.venv/Scripts/python.exe -m airdrop.run replay --flight flights/<架次>
    ./.venv/Scripts/python.exe -m airdrop.run replay --speed 2.0

    # SITL 架次（相机是"内参/外参按模型推导"的模拟标定，路径在 .sitl-recon-tmp/）：
    #   --calib     要用录这段素材时同一份标定，否则退回粗估模型、坐标不可信
    #   --land-plan 要规划"飞掠+降落"才需要；不给就只打印飞掠段
    #   --no-strict 帧被裁剪过的目录（索引里留着已删帧号）必须关严格模式
    ./.venv/Scripts/python.exe -m airdrop.run replay --flight flights/<架次> \
        --calib .sitl-recon-tmp/camera_calib_sim.json --land-plan routes/land.plan --no-strict

本文件是纯库模块：顶部常量是默认值，build_config(**覆盖) / main(**kwargs) 可按需传值；
命令行由 airdrop/run.py 解析，重依赖都在函数体内导入（--help 不加载它们）。

把 ``FLIGHT_DIR`` 指向 ``FlightRecorder`` 录出来的飞行目录（或指向 ``flights/``
自动取最新一次）。脚本按原时间轴重放帧与遥测，然后走与实飞完全相同的那条链路::

    回放源 → 帧-遥测对齐 → 环形缓冲 → PerceptionWorker（YOLO + OCR）
           → TargetTracker（像素 → NED）→ targeting（聚类 + 选唯一）
           → 航线规划（飞掠 + 降落，只生成不上传）→ 打印结果坐标（WGS84）

为什么要用回放迭代
------------------
算法改动要在固定输入上比较才有意义。回放时每一帧仍带着录制时的原始
``capture_timestamp``、遥测仍按原始时间戳进 broker，于是对齐、坐标解算、目标统计
一行都不用改。换一批参数再跑一遍，输出可以直接与上一次（以及原
``detections.jsonl``）对比。

速度
----
``SPEED = 0`` 全速（受算力限制，用于批量回归）；``1.0`` 原速；``2.0`` 两倍速。
"""

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path

from airdrop import Config

# ----------------------------------------------------------------------
# 配置（改这里）
# ----------------------------------------------------------------------
FLIGHT_DIR = Path("flights")  # 指向具体某次 flights/<架次>，或 flights/ 取最新
SPEED = 0.0  # 0=全速 / 1.0=原速 / 2.0=两倍速
BUFFER_SECONDS = 180.0
BUFFER_FPS = 30.0
PLAN_DROP_MISSION = True  # 出了结果就生成飞掠+降落航线（不上传，只打印）
STRICT = True  # 回放异常（缺帧/sink 抛异常）就让本次以 error 结束
PERCEPTION_MODE: str | None = None  # None = 用 Config 默认（ocr）
SELECTION_RULE: str | None = None  # None = 用 Config 默认（median）
#: 相机标定文件；None = 用 Config 默认（仓库根的 ``camera_calib.json``，通常不存在，
#: 那时会退回"60° 视场角粗估"的默认模型——坐标解算结果不可信）。
#: ⚠ 回放要**用录这段素材时同一份标定**：SITL 架次是
#: ``.sitl-recon-tmp/camera_calib_sim.json``（见 ``python -m airdrop.run sitl-recon``）。
CALIB_FILE: str | None = None
#: 降落航线 ``.plan``；None = 用 Config 默认（不配降落段 → 只打印飞掠段，不规划降落）。
LAND_PLAN: str | None = None
#: 边长交叉验证门限（``None`` = 用库默认 = **关**；回放优化时才给，比如 0.25）。
#: 它是诊断：用已知目标边长独立估深度、再与地面求交互校——标定不对时会大面积不过
#: （退回粗估模型的回放里实测每帧都不过），所以正式流程默认关掉、只在这里按需打开。
SIDE_CHECK_TOLERANCE: float | None = None

LOGGER = logging.getLogger("replay_flight")


def build_config(
    *,
    perception_mode: str | None = PERCEPTION_MODE,
    selection_rule: str | None = SELECTION_RULE,
    calib_file: str | None = CALIB_FILE,
    land_plan: str | None = LAND_PLAN,
) -> Config:
    """回放用的配置（感知模式 / 选唯一规则 / 相机标定 / 降落航线可覆盖）。

    ``None`` = 不覆盖（沿用 ``airdrop/config.py`` 里的默认值），而不是"设成 None"。
    """
    base = Config()
    if perception_mode is not None:
        base = base.replace(perception=replace(base.perception, mode=perception_mode))
    if selection_rule is not None:
        base = base.replace(targeting=replace(base.targeting, selection_rule=selection_rule))
    if calib_file is not None:
        base = base.replace(camera=replace(base.camera, calib_file=calib_file))
    if land_plan is not None:
        base = base.replace(routes=replace(base.routes, land_plan=land_plan))
    return base.validated()


def resolve_flight_dir(flight_dir: Path = FLIGHT_DIR) -> Path:
    """``flight_dir`` 指向具体目录时直接用；指向 ``flights/`` 时取最新一次。"""
    if (flight_dir / "frames_index.jsonl").is_file():
        return flight_dir
    candidates = sorted(
        (p for p in flight_dir.glob("*") if (p / "frames_index.jsonl").is_file()),
        key=lambda p: p.name,
    )
    if not candidates:
        raise SystemExit(
            f"{flight_dir} 下没有找到飞行目录（需要 frames_index.jsonl + telemetry.jsonl）。\n"
            "先跑一次带录制的飞行，或把 FLIGHT_DIR 指到已有的素材上。"
        )
    LOGGER.info("FLIGHT_DIR 指向 flights/ 根目录，自动选用最新一次：%s", candidates[-1].name)
    return candidates[-1]


def main(
    *,
    flight_dir: Path = FLIGHT_DIR,
    speed: float = SPEED,
    buffer_seconds: float = BUFFER_SECONDS,
    buffer_fps: float = BUFFER_FPS,
    plan_drop_mission: bool = PLAN_DROP_MISSION,
    strict: bool = STRICT,
    side_check_tolerance: float | None = SIDE_CHECK_TOLERANCE,
    **config_overrides,
) -> int:
    """回放一次；``config_overrides`` 原样交给 :func:`build_config`。"""
    # 重依赖在使用时才导入：`python -m airdrop.run replay --help` 不加载它们
    from airdrop import (
        DEFAULT_SIDE_TOLERANCE,
        AlignmentBuffer,
        AlignmentWriter,
        FrameTelemetryAligner,
        LLARef,
        PerceptionTargetSource,
        PerceptionWorker,
        PlanningError,
        ReplayVideoSource,
        TargetTracker,
        build_drop_mission,
        capacity_for,
        load_broker_from_log,
        ned_to_wgs84,
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    config = build_config(**config_overrides)
    resolved = resolve_flight_dir(Path(flight_dir))

    # 1) 遥测回填：专用 broker + 按帧推进的 pacer（帧拍摄时刻必然落在历史区间内）
    broker, pacer = load_broker_from_log(resolved)
    LOGGER.info("素材：%s（%d 条遥测）", resolved, pacer.total)

    # 2) 回放源：与 Hm30VideoSource 同一套接口，直接换掉即可
    source = ReplayVideoSource(resolved, speed=speed, telemetry=pacer, strict=strict)
    LOGGER.info(
        "回放：%d 帧、素材跨度 %.1fs、%s",
        source.stats.total,
        source.stats.duration_s,
        "全速" if speed == 0 else f"{speed}× 速度",
    )

    # 3) 与实飞一致：对齐 → 环形缓冲 → 感知 → 坐标解算 → 目标统计
    aligner = FrameTelemetryAligner(broker, max_wait=config.align.max_wait, mode=config.align.mode)
    buffer = AlignmentBuffer(capacity_for(buffer_fps, buffer_seconds), storage="jpeg")
    writer = AlignmentWriter(buffer, aligner)
    worker = PerceptionWorker(config.perception.to_pipeline_config(), buffer=buffer)
    camera = config.camera.load_model()
    tracker = TargetTracker(
        config,
        camera=camera,
        side_check_tolerance=(
            DEFAULT_SIDE_TOLERANCE if side_check_tolerance is None else float(side_check_tolerance)
        ),
    )
    targets = PerceptionTargetSource(worker=worker, tracker=tracker)

    worker.start()
    try:
        source.add_sink(writer)
        source.start()
        if not source.wait_ready(timeout=30.0):
            LOGGER.error("回放没有出帧：%s", source.stats.last_error or "未知原因")
            return 1
        # 示例里只等播放线程结束；直接碰私有属性是刻意的（没有公开的 join 接口）
        thread = source._thread  # noqa: SLF001
        if thread is not None:
            thread.join()
    finally:
        source.stop()
        worker.stop()

    written, skipped = writer.stats()
    stats = source.stats
    LOGGER.info(
        "回放结束：状态=%s 投递 %d/%d 帧、跳过 %d 帧（对齐入库 %d、因取不到遥测跳过 %d）",
        stats.state,
        stats.frames,
        stats.total,
        stats.skipped,
        written,
        skipped,
    )
    LOGGER.info("感知：%s", worker.stats)

    # 4) 结果坐标 →（可选）航线规划。不上传：这是离线迭代，不是飞任务。
    added = targets.pump()
    result = targets.result()
    LOGGER.info(
        "目标点 %d 个（本次抽干 %d 条），统计：%s", len(tracker.points), added, tracker.stats
    )
    if not result.ok:
        LOGGER.warning("没有可用结果（没看到目标 / 没有带编号的类）——实飞时走备用点分支")
        return 0

    LOGGER.info("结果：编号 %s @ NED (%.2f, %.2f)", result.code, result.north_m, result.east_m)
    LOGGER.info("候选类：%s", [cluster.as_dict() for cluster in result.clusters])

    if plan_drop_mission:
        # NED → WGS84 需要 NED 原点；回放素材里原点在遥测快照里（origin_*）
        snapshot = broker.get_snapshot()
        if None in (
            snapshot.origin_latitude_deg,
            snapshot.origin_longitude_deg,
            snapshot.origin_altitude_m,
        ):
            LOGGER.warning(
                "素材的遥测里没有 NED 原点，跳过航线规划（实飞时原点来自 GPS_GLOBAL_ORIGIN）"
            )
            return 0
        origin = LLARef(
            lon_deg=float(snapshot.origin_longitude_deg),
            lat_deg=float(snapshot.origin_latitude_deg),
            alt_m=float(snapshot.origin_altitude_m),
        )
        try:
            plan = build_drop_mission(config, origin=origin, target_ned=result.ned)
        except PlanningError as exc:
            # 没配降落段（或降落几何过不了预检）时只做分析：回放入口不上传任务，
            # 打印飞掠段所需的坐标就够了，不必把整次回放判成失败。
            LOGGER.warning("跳过航线规划（%s）——目标坐标见上", exc)
            return 0
        lon, lat, _alt = ned_to_wgs84(plan.target_ned, origin)
        LOGGER.info("飞掠航线：来源 %s，目标 WGS84 (%.6f, %.6f)", plan.source, lat, lon)
        LOGGER.info("任务项 %d 个：%s", len(plan.items), [item.as_dict() for item in plan.items])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
