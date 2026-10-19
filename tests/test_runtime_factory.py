"""在真实组件装配和状态机之间验证阶段参数，仅替换外部飞控输入。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import pytest

import airdrop
from airdrop import Config, MissionItem, MissionRunner, MissionState, NedOrigin, Waypoint
from airdrop.run import resolve_kwargs
from airdrop.runtime.mission import build_live_components, main
from airdrop.runtime.options import RunOptions
from airdrop.telemetry.broker import TelemetryBroker


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

    def upload_mission(self, items: Sequence[MissionItem], /) -> int:
        self.calls.append("upload")
        self.uploads.append(tuple(items))
        return len(items)

    def start_mission(self) -> None:
        self.calls.append("start")
        self.polls = 0

    def mission_finished(self) -> bool:
        self.polls += 1
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

    class TelemetryInput:
        @classmethod
        def from_config(cls, config, *, broker):
            del config
            return _FlightInput(broker, calls)

    class ControllerInput:
        @classmethod
        def from_config(cls, *args, **kwargs):
            del args, kwargs
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

    base = Config()
    config = replace(
        base,
        routes=replace(
            base.routes,
            recon_route=(Waypoint(lat=47.0, lon=8.0, alt_m=50.0),),
            backup_point=Waypoint(lat=47.0, lon=8.0, alt_m=0.0),
            land_plan="this-must-not-be-read.plan",
        ),
        mission=replace(base.mission, tick_hz=200.0, hold_process_min_s=0.0),
        ground=replace(base.ground, ground_point_alt=500.0),
        drop=replace(base.drop, radius_m=10000.0),
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
    tmp_path: Path, mode: str
) -> None:
    """真实解析与适配器一起运行，在连接前检查最终地址和航线，避免默认值串用。"""
    if mode == "sitl":
        pytest.importorskip("airdrop_sim", reason="SITL 集成需要仿真子模块")
    arguments = ["--mode", mode, "--no-recognize", "--no-record"]
    if mode == "real":
        arguments += ["--land-plan", "routes/land.plan", "--backup-point", "47,8,0"]
    else:
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
        assert config.routes.land_plan == "routes/land.plan"
    else:
        from airdrop.runtime import sitl_geometry

        assert config.telemetry.system_address == sitl_geometry.SYSTEM_ADDRESS
        assert config.video.url == str(tmp_path / sitl_geometry.STREAM_SDP_NAME)
        assert config.routes.land_plan == sitl_geometry.LAND_PLAN
        assert Path(config.routes.land_plan).is_file()
