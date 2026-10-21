"""在真实组件装配和状态机之间验证阶段参数，仅替换外部飞控输入。"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Literal, cast

import cv2
import numpy as np
import pytest

import airdrop
from airdrop import (
    AlignedSample,
    Config,
    MissionItem,
    MissionRunner,
    MissionState,
    NedOrigin,
    TelemetrySnapshot,
    Waypoint,
)
from airdrop.run import resolve_kwargs
from airdrop.runtime.mission import build_live_components, main
from airdrop.runtime.options import RunOptions
from airdrop.telemetry.broker import TelemetryBroker
from airdrop.video.source import VideoFrame


def _write_qgc_plan(path: Path, *, lat: float = 47.0, lon: float = 8.001) -> Path:
    """写入包含单个位置项的有效 QGC 航线。"""
    path.write_text(
        json.dumps(
            {
                "fileType": "Plan",
                "mission": {
                    "firmwareType": 12,
                    "vehicleType": 1,
                    "plannedHomePosition": [47.0, 8.0, 500.0],
                    "items": [
                        {
                            "type": "SimpleItem",
                            "command": 16,
                            "frame": 3,
                            "doJumpId": 1,
                            "autoContinue": True,
                            "Altitude": 50.0,
                            "params": [0.0, 0.0, 0.0, None, lat, lon, 50.0],
                        }
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def test_live_components_prepare_detector_with_video_dimensions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[object, ...]] = []

    class TelemetryInput:
        @classmethod
        def from_config(cls, *args, **kwargs):
            del args, kwargs
            return SimpleNamespace()

    class Source:
        def __init__(self, config) -> None:
            del config

    class Detector:
        def load(self):
            calls.append(("load",))
            return self

        def warmup(self, width: int, height: int) -> bool:
            calls.append(("warmup", width, height))
            return True

    monkeypatch.setattr(cv2, "setNumThreads", lambda count: calls.append(("threads", count)))

    worker = SimpleNamespace(detector=Detector())
    camera = SimpleNamespace(image_size=(1920, 1080))
    monkeypatch.setattr(airdrop, "MavsdkThread", TelemetryInput)
    monkeypatch.setattr(airdrop, "Hm30VideoSource", Source)
    monkeypatch.setattr(
        "airdrop.runtime.components.build_perception_components",
        lambda *args, **kwargs: SimpleNamespace(
            camera=camera, writer=None, worker=worker, targets=None
        ),
    )
    config = replace(Config(), video=replace(Config().video, width=1920, height=1080))
    components = build_live_components(
        RunOptions(
            mode="real",
            run_mission=False,
            capture=True,
            recognize=True,
            record=False,
        ),
        config,
    )

    assert components.prepare_perception is not None
    components.prepare_perception()
    components.prepare_perception()
    assert calls == [("load",), ("threads", 4), ("warmup", 1920, 1080)]


@pytest.mark.parametrize(
    ("image_size", "warmup_result", "error"),
    [
        ((1280, 720), True, "标定尺寸"),
        ((1920, 1080), False, "YOLO"),
    ],
)
def test_live_components_reject_invalid_detector_warmup(
    monkeypatch: pytest.MonkeyPatch,
    image_size: tuple[int, int],
    warmup_result: bool,
    error: str,
) -> None:
    calls: list[tuple[object, ...]] = []

    class TelemetryInput:
        @classmethod
        def from_config(cls, *args, **kwargs):
            del args, kwargs
            return SimpleNamespace()

    class Source:
        def __init__(self, config) -> None:
            del config

    class Detector:
        def load(self):
            calls.append(("load",))
            return self

        def warmup(self, width: int, height: int) -> bool:
            calls.append(("warmup", width, height))
            return warmup_result

    monkeypatch.setattr(cv2, "setNumThreads", lambda count: calls.append(("threads", count)))

    monkeypatch.setattr(airdrop, "MavsdkThread", TelemetryInput)
    monkeypatch.setattr(airdrop, "Hm30VideoSource", Source)
    monkeypatch.setattr(
        "airdrop.runtime.components.build_perception_components",
        lambda *args, **kwargs: SimpleNamespace(
            camera=SimpleNamespace(image_size=image_size),
            writer=None,
            worker=SimpleNamespace(detector=Detector()),
            targets=None,
        ),
    )
    base = Config()
    config = replace(base, video=replace(base.video, width=1920, height=1080))
    components = build_live_components(
        RunOptions(
            mode="real",
            run_mission=False,
            capture=True,
            recognize=True,
            record=False,
        ),
        config,
    )

    assert components.prepare_perception is not None
    with pytest.raises(RuntimeError, match=error):
        components.prepare_perception()
    assert calls == (
        [] if image_size != (1920, 1080) else [("load",), ("threads", 4), ("warmup", 1920, 1080)]
    )


class _FlightInput:
    def __init__(self, broker: TelemetryBroker, calls: list[str]) -> None:
        self.broker = broker
        self.calls = calls

    def connect(self) -> None:
        self.calls.append("connect")
        self.broker.update_local_position_velocity(
            SimpleNamespace(north_m=0.0, east_m=0.0, down_m=-50.0),
            SimpleNamespace(north_m_s=15.0, east_m_s=0.0, down_m_s=0.0),
        )
        self.broker.update_in_air(True)
        self.broker.update_gps_global_origin(
            SimpleNamespace(latitude_deg=47.0, longitude_deg=8.0, altitude_m=500.0)
        )
        self.broker.update_attitude_quaternion(SimpleNamespace(w=1.0, x=0.0, y=0.0, z=0.0))

    def stop(self, timeout: float = 5.0) -> None:
        del timeout
        self.calls.append("thread.stop")


class _Controller:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls
        self.uploads: list[tuple[MissionItem, ...]] = []
        self.polls = 0
        self.mission_starts = 0
        self.after_mission_poll: Callable[[int], None] | None = None

    def upload_mission(self, items: Sequence[MissionItem], /) -> int:
        self.calls.append("upload")
        self.uploads.append(tuple(items))
        return len(items)

    def start_mission(self) -> None:
        self.calls.append("start")
        self.polls = 0
        self.mission_starts += 1

    def mission_finished(self) -> bool:
        self.polls += 1
        if self.after_mission_poll is not None:
            self.after_mission_poll(self.mission_starts)
        return self.polls > 1

    def in_mission_mode(self) -> bool:
        return True

    def request_origin(self) -> NedOrigin:
        return NedOrigin(lat_deg=47.0, lon_deg=8.0, alt_m=500.0)

    def hold(self) -> None:
        self.calls.append("hold")

    def rtl(self) -> None:
        self.calls.append("rtl")

    def gripper_release(self) -> bool:
        self.calls.append("real.release")
        return True


def test_mission_capture_writer_waits_for_airborne_and_seals_on_abort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """即使关闭识别，任务采集也只从起飞判据之后进入共享缓冲。"""
    aligned: list[int] = []
    alignment_entered = threading.Event()
    release_alignment = threading.Event()
    controller_actions: list[str] = []

    class TelemetryInput:
        @classmethod
        def from_config(cls, config, *, broker):
            del config, broker
            return object()

    class Source:
        def __init__(self, config) -> None:
            del config
            self.frame = VideoFrame(
                index=7,
                image=np.zeros((8, 8, 3), dtype=np.uint8),
                timestamp=10.0,
                lag=0.0,
            )

        def latest(self) -> VideoFrame:
            return self.frame

    class Aligner:
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs

        def align(self, frame: VideoFrame) -> AlignedSample:
            aligned.append(frame.index)
            alignment_entered.set()
            if not release_alignment.wait(3.0):
                raise AssertionError("未释放对齐屏障")
            return AlignedSample(
                frame=frame,
                snapshot=TelemetrySnapshot(timestamp=frame.timestamp),
                timestamp=frame.timestamp,
                lag=frame.lag,
                mode="interpolate",
            )

    class ControllerInput:
        @classmethod
        def from_config(cls, *args, **kwargs):
            del args, kwargs
            return _Controller(controller_actions)

    monkeypatch.setattr(airdrop, "MavsdkThread", TelemetryInput)
    monkeypatch.setattr(airdrop, "Hm30VideoSource", Source)
    monkeypatch.setattr(airdrop, "FrameTelemetryAligner", Aligner)
    monkeypatch.setattr(airdrop, "DroneController", ControllerInput)
    options = RunOptions(
        mode="real",
        run_mission=True,
        capture=True,
        recognize=False,
        record=False,
        drop_stage=False,
    )
    components = build_live_components(options, Config())
    assert components.prepare_perception is None
    assert components.writer is not None
    assert components.runner is not None
    assert isinstance(components.runner, MissionRunner)
    assert isinstance(components.source, Source)
    assert components.buffer is not None
    writer = components.writer
    runner = cast(MissionRunner, components.runner)
    source = components.source
    buffer = components.buffer

    writer(source.latest())
    assert aligned == []
    assert len(buffer) == 0

    runner._emit("airborne", {"source": "in_air"})
    writer(source.latest())
    assert aligned == []
    next_frame = VideoFrame(
        index=8,
        image=source.latest().image,
        timestamp=10.1,
        lag=0.0,
    )
    writer_errors: list[BaseException] = []

    def write_in_flight_frame() -> None:
        try:
            writer(next_frame)
        except BaseException as exc:  # noqa: BLE001 - 把后台写帧异常交回主线程断言
            writer_errors.append(exc)

    writer_thread = threading.Thread(target=write_in_flight_frame)
    writer_thread.start()
    assert alignment_entered.wait(2.0)

    stop_done = threading.Event()
    stop_errors: list[BaseException] = []

    def stop_runner() -> None:
        try:
            runner.stop("test_stop")
        except BaseException as exc:  # noqa: BLE001 - 把停止线程异常交回主线程断言
            stop_errors.append(exc)
        finally:
            stop_done.set()

    stop_thread = threading.Thread(target=stop_runner)
    stop_thread.start()
    assert stop_done.wait(2.0), "runner.stop 被在途帧的对齐操作阻塞"
    assert controller_actions == ["hold"]
    assert len(buffer) == 0

    release_alignment.set()
    writer_thread.join(2.0)
    stop_thread.join(2.0)
    assert not writer_thread.is_alive() and not stop_thread.is_alive()
    assert writer_errors == [] and stop_errors == []
    assert aligned == [8]
    assert buffer.indices() == (1, 1)

    runner._emit("airborne", {"source": "late"})
    release_alignment.set()
    writer(VideoFrame(index=9, image=next_frame.image, timestamp=10.2, lag=0.0))
    assert aligned == [8]


@pytest.mark.parametrize(
    ("mode", "drop_stage", "real_release"),
    [
        ("real", False, False),
        ("real", True, False),
        ("real", True, True),
        ("sitl", False, False),
        ("sitl", True, False),
    ],
)
def test_entry_runs_selected_stages_with_actual_runner_and_preflight(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mode: Literal["real", "sitl"],
    drop_stage: bool,
    real_release: bool,
) -> None:
    if mode == "sitl":
        pytest.importorskip("airdrop_sim", reason="SITL 集成需要仿真子模块")
    calls: list[str] = []
    controller = _Controller(calls)
    created = []
    brokers: list[TelemetryBroker] = []

    class TelemetryInput:
        @classmethod
        def from_config(cls, config, *, broker):
            del config
            brokers.append(broker)
            return _FlightInput(broker, calls)

    class ControllerInput:
        @classmethod
        def from_config(cls, *args, **kwargs):
            del args, kwargs
            drop_start = 2 if mode == "real" and drop_stage else 1
            updated = False

            def progress(mission_start: int) -> None:
                nonlocal updated
                if updated or not drop_stage or mission_start != drop_start:
                    return
                updated = True
                brokers[0].update_local_position_velocity(
                    SimpleNamespace(north_m=20.0, east_m=0.0, down_m=-50.0),
                    SimpleNamespace(north_m_s=15.0, east_m_s=0.0, down_m_s=0.0),
                )

            controller.after_mission_poll = progress
            return controller

    def forbidden(*args, **kwargs):
        del args, kwargs
        pytest.fail("关闭识别/视频的任务仍构造了模型、相机或视频源")

    monkeypatch.setattr(airdrop, "MavsdkThread", TelemetryInput)
    monkeypatch.setattr(airdrop, "DroneController", ControllerInput)
    monkeypatch.setattr(airdrop, "Hm30VideoSource", forbidden)
    monkeypatch.setattr(airdrop.CameraConfig, "load_model", forbidden)
    monkeypatch.setattr(airdrop, "PerceptionWorker", forbidden)
    if mode == "sitl":
        from airdrop.runtime import sitl_operations

        monkeypatch.setattr(sitl_operations, "operator_takeoff", forbidden)
        monkeypatch.setattr(sitl_operations, "zero_test_params", forbidden)

    def factory(options: RunOptions, config: Config):
        components = build_live_components(options, config)
        created.append(components)
        return components

    from airdrop.georef.geo import ned_to_wgs84

    base = Config()
    backup_lon, backup_lat, _ = ned_to_wgs84((60.0, 0.0, 0.0), (8.0, 47.0, 500.0))
    config = replace(
        base,
        routes=replace(
            base.routes,
            recon_route=(Waypoint(lat=47.0, lon=8.0, alt_m=50.0),),
            backup_point=Waypoint(lat=backup_lat, lon=backup_lon, alt_m=0.0),
            land_plan="this-must-not-be-read.plan",
        ),
        mission=replace(base.mission, tick_hz=200.0, hold_process_min_s=0.0),
        ground=replace(base.ground, ground_point_alt=500.0),
        drop=replace(base.drop, radius_m=2.0),
    )

    assert (
        main(
            mode=mode,
            config=config,
            capture=False,
            recognize=False,
            record=False,
            auto_recon=mode == "real",
            auto_land=False,
            drop_stage=drop_stage,
            real_release=real_release,
            components_factory=factory,
            work_dir=str(tmp_path / "sitl") if mode == "sitl" else None,
        )
        == 0
    )

    components = created[0]
    assert isinstance(components.runner, MissionRunner)
    runner = components.runner
    assert runner.state is MissionState.DONE
    assert runner._preflight is not None
    assert runner._preflight._cancel_requested is not None
    assert runner._preflight._cancel_requested(), "真实装配的取消回调应读取已终态 runner"
    assert runner.history[-1].reason == "holding"
    assert runner.stats.uploads == int(mode == "real") + int(drop_stage)
    assert runner.stats.releases == int(drop_stage)
    assert len(runner.drops) == int(drop_stage)
    assert calls.count("real.release") == int(real_release)
    assert calls.count("hold") == (2 if drop_stage else 1)
    assert calls[-1] == "thread.stop"
    assert not (tmp_path / "sitl").exists()
    assert all(
        item.command != airdrop.MAV_CMD_NAV_LAND for leg in controller.uploads for item in leg
    )


@pytest.mark.parametrize("mode", ["real", "sitl"])
def test_cli_preserves_environment_defaults_at_actual_factory_boundary(
    tmp_path: Path, mode: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """真实解析与适配器一起运行，在连接前检查最终地址和航线，避免默认值串用。"""
    if mode == "sitl":
        pytest.importorskip("airdrop_sim", reason="SITL 集成需要仿真子模块")
    arguments = ["--mode", mode, "--no-recognize", "--no-record"]
    if mode == "real":
        land_plan = tmp_path / "land.plan"
        _write_qgc_plan(land_plan)
        arguments += ["--land-plan", str(land_plan), "--backup-point", "47,8,0"]
    else:
        from airdrop.runtime import sitl

        land_plan = _write_qgc_plan(tmp_path / "sitl-land.plan")
        monkeypatch.setattr(sitl, "LAND_PLAN", str(land_plan))
        arguments += ["--work-dir", str(tmp_path)]
    keywords = resolve_kwargs("mission", arguments)
    configs: list[Config] = []

    class CapturedBeforeConnect(Exception):
        pass

    def factory(_options: RunOptions, config: Config):
        configs.append(config)
        raise CapturedBeforeConnect

    with pytest.raises(CapturedBeforeConnect):
        main(**keywords, components_factory=factory)

    config = configs[0]
    if mode == "real":
        assert config.telemetry.system_address == Config().telemetry.system_address
        assert config.video.url == airdrop.HM30_DEFAULT_RTSP
        assert config.routes.land_plan == str(land_plan)
    else:
        from airdrop.runtime import sitl_geometry

        assert config.telemetry.system_address == sitl_geometry.SYSTEM_ADDRESS
        assert config.video.url == str(tmp_path / sitl_geometry.STREAM_SDP_NAME)
        assert config.routes.land_plan == str(land_plan)
        assert Path(config.routes.land_plan).is_file()
        assert config.mission.auto_land
