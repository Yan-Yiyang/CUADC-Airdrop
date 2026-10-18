"""SITL 场景准备与任务运行适配。"""

from __future__ import annotations

import logging
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal

from airdrop import Config, Waypoint
from airdrop.telemetry._operations import get_operation_gate

from . import sitl_analysis, sitl_geometry, sitl_operations, sitl_report
from .mission import (
    _apply_qgc_video_overrides,
    build_live_components,
    prepare_config,
    run_configured,
)
from .options import RunOptions
from .sitl_monitoring import PerformanceMonitor
from .sitl_scene import load_recon_scene

LOGGER = logging.getLogger(__name__)
WORK_DIR = sitl_geometry.WORK_DIR
RECORD = True
LAG_OVERRIDE_S = None
RECON_TIMEOUT_S = sitl_geometry.RECON_TIMEOUT_S
VIDEO_WAIT_S = sitl_geometry.VIDEO_WAIT_S
READY_TIMEOUT_S = sitl_geometry.READY_TIMEOUT_S
MANIFEST_PATH = None
SYSTEM_ADDRESS = sitl_geometry.SYSTEM_ADDRESS
LAND_PLAN = sitl_geometry.LAND_PLAN
STREAM_SDP_NAME = sitl_geometry.STREAM_SDP_NAME
STREAM_PORT = sitl_geometry.STREAM_PORT
build_config = sitl_geometry.build_config


def main(  # noqa: PLR0913, PLR0915
    *,
    mode: Literal["sitl"] = "sitl",
    run_mission: bool = True,
    capture: bool = True,
    recognize: bool = True,
    record: bool = RECORD,
    drop_stage: bool = True,
    real_release: bool = False,
    auto_recon: bool | None = None,
    auto_land: bool = True,
    max_seconds: float | None = None,
    qgc_video: bool | None = None,
    qgc_host: str | None = None,
    qgc_port: int | None = None,
    qgc_transport: str | None = None,
    qgc_fps: float | None = None,
    qgc_bitrate_kbps: int | None = None,
    work_dir: str = WORK_DIR,
    lag_s: float | None = LAG_OVERRIDE_S,
    recon_timeout_s: float | None = None,
    video_wait_s: float = VIDEO_WAIT_S,
    manifest_path: Path | str | None = MANIFEST_PATH,
    config: Config | None = None,
    system_address: str | None = None,
    land_plan: str | None = None,
    rtsp_url: str | None = None,
    calib_file: str | None = None,
    recon_plan: str = "",
    backup_point: tuple[float, float, float] | None = None,
    components_factory=None,
    **config_overrides,
) -> int:
    """SITL 模式适配：准备场景与参数账本，共用 real 任务组件和生命周期。"""
    options = RunOptions(
        mode=mode,
        run_mission=run_mission,
        capture=capture,
        recognize=recognize,
        record=record,
        drop_stage=drop_stage,
        real_release=real_release,
        auto_recon=auto_recon,
        auto_land=auto_land,
        max_seconds=max_seconds,
        qgc_video=qgc_video,
    ).validated()
    generated_sdp = config is None and options.capture and rtsp_url is None
    write_generated_calib = config is None and calib_file is None and recognize
    scene = load_recon_scene(manifest_path)
    lag = (
        config.video.telemetry_lag
        if config is not None and lag_s is None
        else scene.camera.telemetry_lag_s
        if lag_s is None
        else lag_s
    )
    work = Path(work_dir)
    calib = (
        Path(calib_file)
        if calib_file is not None
        else Path(config.camera.calib_file)
        if config is not None
        else work / "camera_calib_sim.json"
    )
    backup = None
    if backup_point is not None:
        backup = Waypoint(lat=backup_point[0], lon=backup_point[1], alt_m=backup_point[2])
    if config is not None and config_overrides:
        raise ValueError("传入 config 时不能再混用 SITL 配置覆盖参数")
    if config is None:
        base_config = build_config(
            system_address=(SYSTEM_ADDRESS if system_address is None else system_address),
            calib_file=str(calib),
            land_plan=(
                (land_plan if land_plan is not None else LAND_PLAN)
                if options.run_mission and options.drop_stage and options.auto_land
                else ""
            ),
            route=(None if options.run_mission and options.auto_recon is not False else ()),
            recon_plan=recon_plan,
            backup_point=backup,
            recon_timeout_s=(RECON_TIMEOUT_S if recon_timeout_s is None else recon_timeout_s),
            scene=scene,
            **config_overrides,
        )
    else:
        routes = config.routes
        if recon_plan:
            routes = replace(routes, recon_plan=recon_plan, recon_route=())
        if backup is not None:
            routes = replace(routes, backup_point=backup)
        if (
            land_plan is not None
            and options.run_mission
            and options.drop_stage
            and options.auto_land
        ):
            routes = replace(routes, land_plan=land_plan, landing_route=())
        base_config = replace(
            config,
            telemetry=(
                config.telemetry
                if system_address is None
                else replace(config.telemetry, system_address=system_address)
            ),
            camera=(
                config.camera
                if calib_file is None
                else replace(config.camera, calib_file=calib_file)
            ),
            mission=(
                config.mission
                if recon_timeout_s is None
                else replace(config.mission, recon_max_s=recon_timeout_s)
            ),
            routes=routes,
        )
    config = replace(
        base_config,
        video=replace(
            base_config.video,
            url=(
                rtsp_url
                or (base_config.video.url if config is not None else str(work / STREAM_SDP_NAME))
                if options.capture
                else base_config.video.url
            ),
            width=(
                scene.camera.width_px
                if config is None or manifest_path is not None
                else base_config.video.width
            ),
            height=(
                scene.camera.height_px
                if config is None or manifest_path is not None
                else base_config.video.height
            ),
            telemetry_lag=lag,
            extra_input_args=("-protocol_whitelist", "file,udp,rtp"),
        ),
        mission=replace(
            base_config.mission,
            recon_upload=options.recon_upload,
            takeoff_first=(
                False
                if options.run_mission and options.auto_recon is not False
                else base_config.mission.takeoff_first
            ),
        ),
    ).validated()
    config = _apply_qgc_video_overrides(
        config,
        enabled=qgc_video,
        host=qgc_host,
        port=qgc_port,
        transport=qgc_transport,
        fps=qgc_fps,
        bitrate_kbps=qgc_bitrate_kbps,
    )
    config = prepare_config(
        options,
        config,
        qgc_input_rtp_port=STREAM_PORT if generated_sdp else None,
    )
    sdp = sitl_geometry.write_stream_sdp(work / STREAM_SDP_NAME) if options.capture else None
    if write_generated_calib:
        sitl_geometry.write_sim_calib(calib, lag_s=lag, camera=scene.camera)
    factory = build_live_components if components_factory is None else components_factory
    components = factory(options, config)
    previous_on_start = components.on_start
    previous_on_finish = components.on_finish
    original_params: dict[str, int] = {}
    monitor = None
    if (
        options.recognize
        and components.worker is not None
        and components.targets is not None
        and components.source is not None
        and components.writer is not None
    ):
        monitor = PerformanceMonitor(
            components.worker,
            components.targets.tracker,
            components.source,
            components.writer,
        )

    def on_start() -> None:
        if previous_on_start is not None:
            start_result = previous_on_start()
            if start_result:
                raise RuntimeError(f"SITL 前置回调失败，退出码 {start_result}")
        if options.run_mission:
            broker = components.broker
            if broker is None:
                raise RuntimeError("任务控制需要遥测 broker")

            def ready() -> bool:
                snapshot = broker.get_snapshot()
                return (
                    snapshot.is_valid()
                    and snapshot.north_m is not None
                    and snapshot.origin_latitude_deg is not None
                    and snapshot.quaternion_w is not None
                )

            if not sitl_operations.wait_for(ready, READY_TIMEOUT_S, "遥测/原点就绪"):
                raise TimeoutError("等不到 SITL 遥测和原点")
        if components.source is not None and not components.source.wait_ready(video_wait_s):
            raise TimeoutError(f"等不到 SITL 第一帧：{components.source.stats.last_error}")
        if monitor is not None:
            monitor.start()
        if options.run_mission and options.auto_recon is not False:
            sitl_operations.zero_test_params(components.thread, ledger=original_params)
            if components.runner is None or components.controller is None:
                raise RuntimeError("SITL 自动侦查需要任务状态机与控制器")
            result = sitl_operations.operator_takeoff(
                config, components.controller, components.thread, components.broker
            )
            if result:
                raise RuntimeError(f"SITL 自动起飞失败，退出码 {result}")

    def on_finish() -> None:
        errors: list[BaseException] = []
        perf: list[dict[str, Any]] = []
        try:
            if original_params:
                try:
                    sitl_operations.restore_params(components.thread, original_params)
                except BaseException as exc:  # noqa: BLE001 - 中断也不能跳过其它安全收尾
                    errors.append(exc)
            try:
                perf = [] if monitor is None else monitor.stop()
            except BaseException as exc:  # noqa: BLE001 - 中断也不能跳过报告和RPC排空
                errors.append(exc)
            try:
                if (
                    options.recognize
                    and components.runner is not None
                    and components.targets is not None
                ):
                    recorder = components.recorder
                    detections_path = (
                        None
                        if recorder is None or recorder.flight_dir is None
                        else Path(recorder.flight_dir) / "detections.jsonl"
                    )
                    tracker = components.targets.tracker
                    runner = components.runner
                    sitl_report._report(  # noqa: SLF001 - 报告模块只公开内部输出实现
                        sitl_analysis._RunOutcome(  # noqa: SLF001 - 运行结果 DTO 仅供适配层组装
                            work=work,
                            sdp=sdp or work / STREAM_SDP_NAME,
                            calib=calib,
                            lag_s=lag,
                            flown_s=None,
                            result=tracker.result(),
                            stats=tracker.stats,
                            perception=sitl_analysis.PerceptionSummary.from_worker(
                                components.worker
                            ),
                            video_stats=components.source.stats if components.source else None,
                            writer_stats=(
                                components.writer.stats() if components.writer else (0, 0)
                            ),
                            detail=(
                                sitl_analysis._load_detail(detections_path)  # noqa: SLF001
                                if detections_path
                                else []
                            ),
                            broker=components.broker,
                            flight_dir=(
                                None
                                if recorder is None or recorder.flight_dir is None
                                else Path(recorder.flight_dir)
                            ),
                            runner_stats={
                                "state": str(runner.state),
                                "uploads": runner.stats.uploads,
                                "releases": runner.stats.releases,
                                "aborts": runner.stats.aborts,
                                "errors": runner.stats.errors,
                                "history": [str(row.to_state) for row in runner.history],
                                "transitions": [row.as_dict() for row in runner.history],
                                "ended_at": time.time(),
                            },
                            perf=perf,
                            origin=(
                                None
                                if runner.origin is None
                                else (
                                    float(runner.origin.lon_deg),
                                    float(runner.origin.lat_deg),
                                    float(runner.origin.alt_m),
                                )
                            ),
                            tracker=tracker,
                            config=config,
                            scene=scene,
                        )
                    )
            except BaseException as exc:
                errors.append(exc)
                LOGGER.exception("SITL 报告生成失败；继续释放 MAVSDK 与其它资源")
            try:
                if previous_on_finish is not None:
                    previous_on_finish()
            except BaseException as exc:  # noqa: BLE001 - 收尾必须保留工厂回调中断
                errors.append(exc)
        finally:
            gate = get_operation_gate(components.thread)
            while True:
                try:
                    gate.wait_idle()
                    break
                except KeyboardInterrupt as exc:
                    errors.append(exc)
                except BaseException as exc:  # noqa: BLE001 - 记录门控异常后仍由会话停止线程
                    errors.append(exc)
                    break
        if errors:
            raise BaseExceptionGroup(f"SITL 收尾中有 {len(errors)} 项失败", errors)

    components.on_start = on_start
    components.on_finish = on_finish
    return run_configured(options, config, components_factory=lambda _options, _config: components)
