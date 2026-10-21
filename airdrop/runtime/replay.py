"""回放全链路示例：一份素材反复迭代"识别 → 坐标 → 统计 → 航线"，全程不碰硬件。

用法（集中式入口在 airdrop/run.py）::

    ./.venv/Scripts/python.exe -m airdrop.run replay --flight flights/example-flight
    ./.venv/Scripts/python.exe -m airdrop.run replay --speed 2.0

    # 按素材目录重放；需要坐标解算时提供与该素材匹配的相机标定。
    ./.venv/Scripts/python.exe -m airdrop.run replay --flight flights/example-flight --calib camera_calib.json
    # 被裁剪过且索引仍包含缺帧记录的素材，可按需关闭严格模式。
    ./.venv/Scripts/python.exe -m airdrop.run replay --flight flights/example-flight --no-strict

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
import math
import sys
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

from airdrop import Config

# ----------------------------------------------------------------------
# 配置（改这里）
# ----------------------------------------------------------------------
FLIGHT_DIR = Path("flights")  # 指向某次飞行目录，或指向包含多次飞行的目录以选取最新记录
SPEED = 0.0  # 0=全速 / 1.0=原速 / 2.0=两倍速
BUFFER_SECONDS = 180.0
BUFFER_FPS = 30.0
PLAN_DROP_MISSION = True  # 出了结果就生成飞掠+降落航线（不上传，只打印）
STRICT = True  # 回放异常（缺帧/sink 抛异常）就让本次以 error 结束
DRAIN_TIMEOUT_S = 120.0  # 停止生产后等待尾帧与 OCR 的最大秒数
PERCEPTION_MODE: str | None = None  # None = 用 Config 默认（ocr）
SELECTION_RULE: str | None = None  # None = 用 Config 默认（median）
#: 相机标定文件；None = 用 Config 默认（仓库根的 ``camera_calib.json``，通常不存在，
#: 那时会退回"60° 视场角粗估"的默认模型——坐标解算结果不可信）。
#: ⚠ 回放要**用录这段素材时同一份标定**：SITL 架次是
#: ``.sitl-recon-tmp/camera_calib_sim.json``（见 ``python -m airdrop.run mission --mode sitl``）。
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


def main(  # noqa: PLR0912, PLR0913, PLR0915 - CLI 参数和统一回放生命周期
    *,
    flight_dir: Path = FLIGHT_DIR,
    speed: float = SPEED,
    buffer_seconds: float = BUFFER_SECONDS,
    buffer_fps: float = BUFFER_FPS,
    plan_drop_mission: bool = PLAN_DROP_MISSION,
    strict: bool = STRICT,
    drain_timeout_s: float = DRAIN_TIMEOUT_S,
    side_check_tolerance: float | None = SIDE_CHECK_TOLERANCE,
    qgc_video: bool | None = None,
    qgc_host: str | None = None,
    qgc_port: int | None = None,
    qgc_transport: str | None = None,
    qgc_fps: float | None = None,
    qgc_bitrate_kbps: int | None = None,
    **config_overrides,
) -> int:
    """回放一次；``config_overrides`` 原样交给 :func:`build_config`。"""
    if not math.isfinite(drain_timeout_s) or drain_timeout_s < 0:
        raise ValueError("排空超时必须是非负有限秒数")
    # 重依赖在使用时才导入：`python -m airdrop.run replay --help` 不加载它们
    from airdrop import (
        DEFAULT_SIDE_TOLERANCE,
        AlignmentBuffer,
        ReplayVideoSource,
        capacity_for,
        load_broker_from_log,
    )
    from airdrop.perception.detector import FrameSizeMismatchError

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    config = build_config(**config_overrides)
    from .mission import _apply_qgc_video_overrides

    config = _apply_qgc_video_overrides(
        config,
        enabled=qgc_video,
        host=qgc_host,
        port=qgc_port,
        transport=qgc_transport,
        fps=qgc_fps,
        bitrate_kbps=qgc_bitrate_kbps,
    )
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
    buffer = AlignmentBuffer(capacity_for(buffer_fps, buffer_seconds), storage="jpeg")
    from .components import build_perception_components, configure_opencv_threads

    perception = build_perception_components(
        config,
        broker,
        buffer,
        capture=True,
        recognize=True,
        side_check_tolerance=(
            DEFAULT_SIDE_TOLERANCE if side_check_tolerance is None else float(side_check_tolerance)
        ),
    )
    if (
        perception.writer is None
        or perception.worker is None
        or perception.tracker is None
        or perception.targets is None
    ):
        raise RuntimeError("回放感知组件未完整创建")
    writer, worker, tracker, targets = (
        perception.writer,
        perception.worker,
        perception.tracker,
        perception.targets,
    )
    display = None
    display_error = False
    if config.qgc_video.enabled:
        from .qgc import QgcVideoRuntime

        display = QgcVideoRuntime(
            config,
            source,
            broker,
            perception.camera,
            worker,
            replay_mode=True,
        )

    drained = False
    ready = False
    source_failed = False
    try:
        configure_opencv_threads(worker, config.perception.opencv_threads)
        worker.start()
        if display is not None:
            try:
                display.start()
                if not display.running:
                    raise RuntimeError("QGC 视频发布器启动后未运行")
            except Exception:
                LOGGER.exception("QGC 视频发布启动失败；继续回放并关闭显示")
                try:
                    display.stop()
                except Exception:
                    display_error = True
                    LOGGER.exception("QGC 视频发布器启动失败后的清理也失败")
                else:
                    display = None
        source.add_sink(writer)
        source.start()
        ready = source.wait_ready(timeout=30.0)
        if not ready:
            LOGGER.error("回放没有出帧：%s", source.stats.last_error or "未知原因")
        if ready:
            while not source.wait_finished(timeout=0.1):
                worker.raise_if_failed()
            source.join(timeout=1.0)
            worker.raise_if_failed()
            if source.stats.state == "error":
                raise RuntimeError(source.stats.last_error or "回放源异常")
    except (FrameSizeMismatchError, RuntimeError) as exc:
        LOGGER.error("回放终止：%s", exc)
        source_failed = True
    finally:
        active_exception = sys.exception()
        interruptions: list[BaseException] = []
        stats = source.stats  # stop() 会改写状态，保留播放终态用于摘要和返回码。

        def cleanup(
            label: str,
            callback: Callable[[], object],
            *,
            display_step: bool = False,
        ) -> None:
            nonlocal source_failed, display_error
            try:
                callback()
            except BaseException as exc:
                LOGGER.exception("回放清理步骤失败：%s", label)
                if isinstance(exc, Exception):
                    if display_step:
                        display_error = True
                    else:
                        source_failed = True
                else:
                    interruptions.append(exc)

        cleanup("source.stop", source.stop)
        if not source.running:

            def drain_worker() -> None:
                nonlocal drained
                drained = worker.drain(timeout=drain_timeout_s)

            cleanup("worker.drain", drain_worker)
        else:
            source_failed = True
            LOGGER.error("回放源未确认停止，无法排空感知工作器")
        cleanup("worker.stop", worker.stop)
        cleanup("targets.pump", targets.pump)
        if display is not None:
            cleanup("display.stop", display.stop, display_step=True)
        if active_exception is not None:
            raise active_exception
        if interruptions:
            if len(interruptions) == 1:
                raise interruptions[0]
            raise BaseExceptionGroup("回放收尾期间发生多个中断", interruptions)

    return _report_replay(
        config,
        broker=broker,
        stats=stats,
        ready=ready and not source_failed,
        drained=drained,
        worker=worker,
        writer=writer,
        tracker=tracker,
        targets=targets,
        plan_drop_mission=plan_drop_mission,
        display_error=display_error,
    )


def _report_replay(  # noqa: PLR0913 - 汇总所需数据来自彼此独立的回放组件
    config,
    *,
    broker,
    stats,
    ready,
    drained,
    worker,
    writer,
    tracker,
    targets,
    plan_drop_mission,
    display_error=False,
) -> int:
    """汇总回放完整性、识别结果，并按需生成离线航线。"""
    from airdrop import LLARef, PlanningError, build_drop_mission, ned_to_wgs84

    written, skipped = writer.stats()
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
    added = targets.pump()
    result = targets.result()
    LOGGER.info(
        "目标点 %d 个（本次抽干 %d 条），统计：%s", len(tracker.points), added, tracker.stats
    )
    if (
        display_error
        or not ready
        or stats.state != "finished"
        or not drained
        or worker.stats.frames != written
    ):
        LOGGER.error(
            "回放未完整完成：播放=%s 排空=%s 处理=%d/%d，错误=%s",
            stats.state,
            drained,
            worker.stats.frames,
            written,
            stats.last_error or worker.stats.last_error,
        )
        return 1
    if not result.ok:
        LOGGER.warning("没有可用结果（没看到目标 / 没有带编号的类）——实飞时走备用点分支")
        return 0
    LOGGER.info("结果：编号 %s @ NED (%.2f, %.2f)", result.code, result.north_m, result.east_m)
    LOGGER.info("候选类：%s", [cluster.as_dict() for cluster in result.clusters])
    if not plan_drop_mission:
        return 0
    snapshot = broker.get_snapshot()
    if None in (
        snapshot.origin_latitude_deg,
        snapshot.origin_longitude_deg,
        snapshot.origin_altitude_m,
    ):
        LOGGER.warning("素材的遥测里没有 NED 原点，跳过航线规划")
        return 0
    origin = LLARef(
        lon_deg=float(snapshot.origin_longitude_deg),
        lat_deg=float(snapshot.origin_latitude_deg),
        alt_m=float(snapshot.origin_altitude_m),
    )
    try:
        plan = build_drop_mission(config, origin=origin, target_ned=result.ned)
    except PlanningError as exc:
        LOGGER.warning("跳过航线规划（%s）——目标坐标见上", exc)
        return 0
    lon, lat, _alt = ned_to_wgs84(plan.target_ned, origin)
    LOGGER.info("飞掠航线：来源 %s，目标 WGS84 (%.6f, %.6f)", plan.source, lat, lon)
    LOGGER.info("任务项 %d 个：%s", len(plan.items), [item.as_dict() for item in plan.items])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
