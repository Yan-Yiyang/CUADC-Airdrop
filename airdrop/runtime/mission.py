"""真实与 SITL 共用的任务组件装配和入口。"""

from __future__ import annotations

import logging
import math
import re
import sys
import threading
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Literal
from urllib.parse import urlsplit

from airdrop import HM30_DEFAULT_RTSP, Config, Waypoint

from .options import RunOptions

if TYPE_CHECKING:
    from collections.abc import Callable

    from airdrop.ballistics.drops import DropRecord
    from airdrop.video.buffer import AlignmentWriter

    from .session import LiveComponents

# argparse 从这些默认值读取入口参数。
SYSTEM_ADDRESS: str | None = None
RTSP_URL = HM30_DEFAULT_RTSP
CALIB_FILE: str | None = None
RECON_PLAN = ""
LAND_PLAN = ""
WORK_DIR = ".sitl-recon-tmp"
LAG_OVERRIDE_S: float | None = None
MODE = "real"
RUN_MISSION = True
CAPTURE = True
RECOGNIZE = True
RECORD = True
DROP_STAGE = True
REAL_RELEASE = False
AUTO_LAND = True

LOGGER = logging.getLogger(__name__)


def build_config(
    *,
    system_address: str | None = SYSTEM_ADDRESS,
    rtsp_url: str = RTSP_URL,
    calib_file: str | None = CALIB_FILE,
    recon_plan: str = RECON_PLAN,
    land_plan: str = LAND_PLAN,
    backup_point: Waypoint | None = None,
) -> Config:
    """用真实场地参数生成配置；场地航线不含内置坐标，须由调用方提供。"""
    base = Config()
    telemetry = (
        base.telemetry
        if system_address is None
        else replace(base.telemetry, system_address=system_address)
    )
    camera = base.camera if calib_file is None else replace(base.camera, calib_file=calib_file)
    routes = replace(
        base.routes,
        recon_plan=recon_plan,
        land_plan=land_plan,
        backup_point=backup_point,
    )
    return replace(
        base,
        telemetry=telemetry,
        video=replace(base.video, url=rtsp_url),
        camera=camera,
        routes=routes,
    ).validated()


def build_live_components(options: RunOptions, config: Config) -> LiveComponents:  # noqa: PLR0915 - 阶段资源按依赖顺序装配
    """按阶段开关创建真实任务组件；测试可传入替代组件工厂。"""
    from airdrop import (
        AlignmentBuffer,
        BallisticsModel,
        DroneController,
        DryRunController,
        FlightRecorder,
        Hm30VideoSource,
        MavsdkThread,
        MissionRunner,
        Preflight,
        ReleaseJudge,
        TelemetryBroker,
        capacity_for,
    )

    from .components import build_perception_components, configure_opencv_threads
    from .session import LiveComponents

    broker = TelemetryBroker(
        history_interval=config.telemetry.history_interval,
        history_maxlen=config.telemetry.history_maxlen,
        source_time_alignment=config.telemetry.source_time_alignment,
    )
    thread = MavsdkThread.from_config(config.telemetry, broker=broker)
    recorder = FlightRecorder(config) if options.record else None
    display_runtime = None
    runner: MissionRunner | None = None
    source = Hm30VideoSource(config.video) if options.capture else None
    writer: AlignmentWriter | None = None

    def latest_source_index() -> int | None:
        frame = None if source is None else source.latest()
        return None if frame is None else frame.index

    def on_event(kind: str, data: dict[str, object]) -> None:
        if writer is not None:
            if kind in {"airborne", "airborne_skipped"}:
                if runner is not None and (runner.stopped or runner.machine.is_terminal):
                    writer.seal()
                else:
                    writer.begin_after(latest_source_index)
            elif kind == "state" and str(data.get("to_state")) in {"ABORT", "DONE"}:
                writer.seal()
        if recorder is not None:
            try:
                recorder.events.emit(kind, **data)
            except Exception:
                LOGGER.exception("飞行事件写入失败：%s", kind)
        if display_runtime is not None:
            display_runtime.on_event(kind, data)

    def append_drop(item: DropRecord) -> None:
        if recorder is not None:
            # FlightRecorder 的流式写入器在 start() 后才可访问，闭包延迟取值。
            recorder.drops.append(item)

    on_drop = append_drop if recorder is not None else None

    buffer = AlignmentBuffer(capacity_for(30.0, 180.0), storage="jpeg")
    perception = build_perception_components(
        config,
        broker,
        buffer,
        capture=options.capture,
        recognize=options.recognize,
        recorder=recorder,
        on_event=on_event,
        defer_until_airborne=options.run_mission,
    )
    camera, writer, worker, targets = (
        perception.camera,
        perception.writer,
        perception.worker,
        perception.targets,
    )
    prepare_perception: Callable[[], None] | None = None
    if worker is not None:
        selected_worker = worker
        output_size = (int(config.video.width), int(config.video.height))
        warmup_lock = threading.Lock()
        warmed = False

        def prepare_selected_perception() -> None:
            nonlocal warmed
            with warmup_lock:
                if warmed:
                    return
                if camera is not None and camera.image_size is not None:
                    calib_size = tuple(int(value) for value in camera.image_size)
                    if calib_size != output_size:
                        raise RuntimeError(
                            f"相机标定尺寸 {calib_size} 与视频输出尺寸 {output_size} 不一致"
                        )
                configure_opencv_threads(
                    selected_worker,
                    config.perception.opencv_threads,
                )
                if not selected_worker.detector.warmup(*output_size):
                    raise RuntimeError(f"YOLO 按视频输出尺寸 {output_size} 预热失败")
                warmed = True

        prepare_perception = prepare_selected_perception

    if config.qgc_video.enabled:
        if source is None:
            raise ValueError("QGC 视频发布需要开启视频采集")
        from .qgc import QgcVideoRuntime

        display_runtime = QgcVideoRuntime(
            config,
            source,
            broker,
            camera,
            worker,
            on_event=on_event,
            simulated_release=not options.real_release,
        )

    def preflight_cancelled() -> bool:
        return runner is not None and (runner.stopped or runner.machine.is_terminal)

    if options.run_mission:
        inner = DroneController.from_config(config, thread, on_event=on_event)
        controller = inner if options.real_release else DryRunController(inner, on_event=on_event)
        judge = ReleaseJudge(
            config.drop,
            BallisticsModel(config.ballistics),
            overfly_heading_deg=config.overfly.heading_deg,
            on_event=on_event,
        )

        def load_camera() -> str:
            model = camera if camera is not None else config.camera.load_model()
            origin = model.source or "默认模型（标定缺失/损坏，坐标不可信）"
            return f"{config.camera.calib_file} → {origin}，标定尺寸 {model.image_size}"

        loaders = (
            {"detector": worker.detector.load, "ocr": worker.start, "camera": load_camera}
            if worker is not None
            else {}
        )
        preflight = Preflight(
            config.preflight,
            model_loaders=loaders,
            video_source=source,
            camera_model=camera,
            on_event=on_event,
            cancel_requested=preflight_cancelled,
        )
        runner = MissionRunner(
            config,
            controller,
            broker,
            target_result=None if targets is None else targets.result,
            target_busy=None if targets is None else targets.busy,
            health_check=None if worker is None else worker.raise_if_failed,
            finish_recon=None if worker is None else worker.finish_input,
            release_judge=judge,
            on_event=on_event,
            on_drop=on_drop,
            preflight=preflight,
        )
    return LiveComponents(
        thread=thread,
        source=source,
        writer=writer,
        worker=worker,
        targets=targets,
        recorder=recorder,
        display=display_runtime,
        runner=runner,
        controller=None if runner is None else controller,
        broker=broker,
        buffer=buffer,
        prepare_perception=prepare_perception,
    )


# 参数与统一 CLI 一一对应，保持具名覆盖便于脚本与测试调用。
def main(  # noqa: PLR0913
    *,
    mode: Literal["real", "sitl"] = "real",
    run_mission: bool = True,
    capture: bool = True,
    recognize: bool = True,
    record: bool = True,
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
    config: Config | None = None,
    components_factory: Callable[[RunOptions, Config], LiveComponents] | None = None,
    manifest_path: Path | str | None = None,
    work_dir: str | None = None,
    lag_s: float | None = None,
    **config_overrides,
) -> int:
    """执行统一任务入口；所有参数在构造运行组件与连接飞控前校验。"""
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
    config_overrides = {key: value for key, value in config_overrides.items() if value is not None}
    if mode == "sitl":
        from .sitl import main as sitl_main

        return sitl_main(
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
            qgc_host=qgc_host,
            qgc_port=qgc_port,
            qgc_transport=qgc_transport,
            qgc_fps=qgc_fps,
            qgc_bitrate_kbps=qgc_bitrate_kbps,
            manifest_path=manifest_path,
            work_dir=WORK_DIR if work_dir is None else work_dir,
            lag_s=lag_s,
            config=config,
            components_factory=components_factory,
            **config_overrides,
        )
    if manifest_path is not None or work_dir is not None or lag_s is not None:
        raise ValueError("--manifest、--work-dir 与 --lag 仅适用于 --mode sitl")
    if "backup_point" in config_overrides and isinstance(config_overrides["backup_point"], tuple):
        config_overrides["backup_point"] = Waypoint(
            lat=config_overrides["backup_point"][0],
            lon=config_overrides["backup_point"][1],
            alt_m=config_overrides["backup_point"][2],
        )
    if config is not None and config_overrides:
        raise ValueError("传入 config 时不能再混用配置覆盖参数")
    config = config or build_config(**config_overrides)
    config = _apply_qgc_video_overrides(
        config,
        enabled=qgc_video,
        host=qgc_host,
        port=qgc_port,
        transport=qgc_transport,
        fps=qgc_fps,
        bitrate_kbps=qgc_bitrate_kbps,
    )
    return run_configured(options, config, components_factory=components_factory)


def _apply_qgc_video_overrides(
    config: Config,
    *,
    enabled: bool | None,
    host: str | None,
    port: int | None,
    transport: str | None,
    fps: float | None,
    bitrate_kbps: int | None,
) -> Config:
    """只覆盖命令行明确给出的 QGC 参数，保留注入配置的其它字段。"""
    updates = {
        name: value
        for name, value in {
            "enabled": enabled,
            "host": host,
            "port": port,
            "transport": transport,
            "fps": fps,
            "bitrate_kbps": bitrate_kbps,
        }.items()
        if value is not None
    }
    if not updates:
        return config
    return replace(config, qgc_video=replace(config.qgc_video, **updates)).validated()


def _apply_calibrated_lag(options: RunOptions, config: Config) -> Config:
    """实飞识别使用文件中的同参照标定量；SITL 和录制回放各自保留显式时间轴。"""
    import json
    import math
    from dataclasses import replace

    if options.mode != "real" or not options.capture or not options.recognize:
        return config
    path = Path(config.camera.calib_file)
    if not path.is_file():
        return config
    payload = json.loads(path.read_text(encoding="utf-8"))
    lag = payload.get("telemetry_lag")
    if lag is None:
        return config
    timing = payload.get("meta", {}).get("time_offset", {})
    if (
        timing.get("telemetry_reference") != "quaternion_received_at"
        or timing.get("pose_alignment") != "source_timestamp_us"
    ):
        raise ValueError("相机标定延迟缺少一致的源时标参照，请重新标定")
    if not config.telemetry.source_time_alignment:
        raise ValueError("源时标标定不能用于存在接收时刻混合缺陷的遥测路径")
    if (
        isinstance(lag, bool)
        or not isinstance(lag, (int, float))
        or not math.isfinite(lag)
        or lag < 0
    ):
        raise ValueError("标定 telemetry_lag 必须是有限、非负秒数")
    LOGGER.info("应用源时标标定延迟 %.6f s：%s", lag, path)
    return replace(config, video=replace(config.video, telemetry_lag=float(lag)))


def prepare_config(
    options: RunOptions,
    config: Config,
    *,
    qgc_input_rtp_port: int | None = None,
) -> Config:
    """在构造运行资源前归一化阶段配置并拒绝不完整的航线。"""
    from dataclasses import replace

    if not isinstance(options, RunOptions):
        raise TypeError("options 必须是 RunOptions")
    options.validated()
    config = _apply_calibrated_lag(options, config)
    if options.qgc_video is True and not options.capture:
        raise ValueError("开启 QGC 视频发布必须同时开启视频采集")
    if options.qgc_video is not None:
        config = replace(config, qgc_video=replace(config.qgc_video, enabled=options.qgc_video))
    if options.real_release and not config.gripper.enabled:
        raise ValueError("真实投弹已启用，但 Config.gripper.enabled=False")
    preflight = replace(
        config.preflight,
        load_detector=config.preflight.load_detector and options.recognize,
        load_ocr=config.preflight.load_ocr and options.recognize,
        load_camera=config.preflight.load_camera and options.recognize,
        check_video=config.preflight.check_video and options.capture and options.recognize,
    )
    config = replace(
        config,
        preflight=preflight,
        routes=(
            config.routes
            if options.drop_stage and options.auto_land
            else replace(config.routes, landing_route=(), land_plan="")
        ),
        mission=replace(
            config.mission,
            drop_stage=options.drop_stage,
            auto_land=options.auto_land,
            recon_upload=options.recon_upload,
        ),
    ).validated()
    if config.qgc_video.enabled and not options.capture:
        raise ValueError("配置启用了 QGC 视频发布，但视频采集已关闭")
    _validate_qgc_video_ports(options, config, known_input_rtp_port=qgc_input_rtp_port)
    if options.run_mission:
        if options.recon_upload == "auto" and not (
            config.routes.recon_route or config.routes.recon_plan
        ):
            raise ValueError("自动上传侦查航线需要 --recon-plan 或显式侦查航点")
        if (
            options.drop_stage
            and options.auto_land
            and not (config.routes.landing_route or config.routes.land_plan)
        ):
            raise ValueError("自动降落需要 --land-plan 或显式降落航点")
        if options.drop_stage and not options.recognize and config.routes.backup_point is None:
            raise ValueError("关闭识别且保留投放阶段时，必须提供 --backup-point")
        for label, points in (
            ("侦查航点", config.routes.recon_route),
            (
                "降落航点",
                config.routes.landing_route if options.drop_stage and options.auto_land else (),
            ),
            ("备用点", () if config.routes.backup_point is None else (config.routes.backup_point,)),
        ):
            for point in points:
                if (
                    not math.isfinite(point.lat)
                    or not -90.0 <= point.lat <= 90.0
                    or not math.isfinite(point.lon)
                    or not -180.0 <= point.lon <= 180.0
                    or not math.isfinite(point.alt_m)
                ):
                    raise ValueError(f"{label}必须是有效 WGS84 坐标，alt_m 为相对起飞点米数")
        for label, path, enabled in (
            ("侦查计划", config.routes.recon_plan, options.recon_upload == "auto"),
            ("降落计划", config.routes.land_plan, options.drop_stage and options.auto_land),
        ):
            if enabled and path and not Path(path).is_file():
                raise ValueError(f"{label}文件不存在：{path}")
    return config


def _validate_qgc_video_ports(
    options: RunOptions, config: Config, *, known_input_rtp_port: int | None = None
) -> None:
    """在场景文件或输出目录创建前拒绝 QGC 与输入视频占用同一 UDP 端口。"""
    qgc = config.qgc_video
    if not qgc.enabled:
        return
    input_url = config.video.url
    parsed_url = urlsplit(input_url)
    if parsed_url.scheme.lower() in {"rtsp", "rtsps"}:
        return
    input_port: int | None = known_input_rtp_port
    candidate = Path(input_url)
    if input_port is not None:
        pass
    elif candidate.is_file() or candidate.suffix.lower() == ".sdp":
        if candidate.is_file():
            match = re.search(
                r"(?im)^m=video\s+(\d+)\s+RTP/", candidate.read_text(encoding="utf-8")
            )
            if match:
                input_port = int(match.group(1))
        elif options.mode == "sitl":
            input_port = 5600
    elif parsed_url.scheme.lower() in {"udp", "rtp"}:
        try:
            input_port = parsed_url.port
        except ValueError as exc:
            raise ValueError(f"输入视频 URL 端口无效：{input_url}") from exc
    elif options.mode == "sitl":
        input_port = 5600
    if input_port is None:
        return
    input_ports = {input_port, input_port + 1}
    output_ports = {qgc.port}
    if qgc.transport == "rtp":
        output_ports.add(qgc.port + 1)
    if input_ports & output_ports:
        raise ValueError(
            f"QGC 视频输出端口 {sorted(output_ports)} 与输入视频端口 {sorted(input_ports)} 重叠"
        )


def run_configured(
    options: RunOptions,
    config: Config,
    *,
    components_factory: Callable[[RunOptions, Config], LiveComponents] | None = None,
) -> int:
    """执行已装配配置；SITL 适配只提供配置与操作手回调。"""
    from .session import LiveSession

    config = prepare_config(options, config)
    factory = build_live_components if components_factory is None else components_factory
    components = factory(options, config)
    return LiveSession(options, components).run()


if __name__ == "__main__":
    sys.exit(main())
