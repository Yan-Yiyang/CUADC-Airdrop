"""统一任务入口的阶段装配回归。"""

from __future__ import annotations

from dataclasses import replace

import pytest

from airdrop import Config
from airdrop.runtime.mission import prepare_config
from airdrop.runtime.options import RunOptions


def test_passive_config_does_not_require_site_routes_or_calibration() -> None:
    options = RunOptions(
        mode="real", run_mission=False, capture=False, recognize=False, record=False
    )

    prepared = prepare_config(options, Config())

    assert prepared.routes.recon_route == ()
    assert prepared.routes.landing_route == ()
    assert prepared.camera.calib_file == Config().camera.calib_file


def test_capture_without_recognition_skips_calibration_video_preflight() -> None:
    options = RunOptions(mode="real", capture=True, recognize=False, drop_stage=False)

    prepared = prepare_config(options, Config())

    assert prepared.preflight.check_video is False
    assert prepared.preflight.load_camera is False


def test_no_drop_stage_ignores_landing_configuration() -> None:
    config = replace(Config(), routes=replace(Config().routes, land_plan="missing.plan"))
    options = RunOptions(mode="real", drop_stage=False, auto_land=True)

    prepared = prepare_config(options, config)

    assert prepared.routes.land_plan == ""
    assert prepared.mission.drop_stage is False


def test_auto_recon_requires_an_explicit_plan_before_factory() -> None:
    options = RunOptions(mode="real", auto_recon=True)

    with pytest.raises(ValueError, match="自动上传侦查航线"):
        prepare_config(options, Config())


def test_auto_land_is_required_only_for_selected_drop_stage() -> None:
    options = RunOptions(mode="real", drop_stage=True, auto_land=True)

    with pytest.raises(ValueError, match="自动降落"):
        prepare_config(options, Config())


def test_prepare_config_calibration_capture_profile() -> None:
    """标定 profile 请求所需速率，不降低已配置的高速请求。"""
    config = Config()
    config = replace(
        config,
        telemetry=replace(
            config.telemetry,
            position_rate_hz=20.0,
            position_velocity_ned_rate_hz=60.0,
            quaternion_rate_hz=120.0,
            attitude_rate_hz=30.0,
            source_time_alignment=False,
        ),
    )
    options = RunOptions(mode="real", run_mission=False, recognize=False, calibration_capture=True)
    prepared = prepare_config(options, config)
    assert prepared.telemetry.position_rate_hz == 20.0
    assert prepared.telemetry.position_velocity_ned_rate_hz == 60.0
    assert prepared.telemetry.quaternion_rate_hz == 120.0
    assert prepared.telemetry.attitude_rate_hz == 30.0
    assert prepared.telemetry.source_time_alignment
    assert prepared.record.telemetry_hz == 0.0
    assert prepared.record.video

    defaults = prepare_config(
        RunOptions(mode="real", run_mission=False, recognize=False, calibration_capture=True),
        Config(),
    )
    assert defaults.telemetry.position_rate_hz == Config().telemetry.position_rate_hz
    assert (
        defaults.telemetry.position_velocity_ned_rate_hz
        == Config().telemetry.position_velocity_ned_rate_hz
    )
    assert defaults.telemetry.quaternion_rate_hz == 100.0
    assert defaults.telemetry.attitude_rate_hz == 30.0


@pytest.mark.parametrize(
    "options",
    [
        RunOptions(mode="sitl", run_mission=False, recognize=False, calibration_capture=True),
        RunOptions(mode="real", run_mission=False, recognize=True, calibration_capture=True),
        RunOptions(mode="real", run_mission=True, recognize=False, calibration_capture=True),
        RunOptions(
            mode="real", run_mission=False, recognize=False, record=False, calibration_capture=True
        ),
    ],
)
def test_calibration_capture_rejects_invalid_modes(options: RunOptions) -> None:
    with pytest.raises(ValueError, match="标定采集"):
        options.validated()


@pytest.mark.parametrize(
    ("options", "message"),
    [
        (RunOptions(mode="real", capture=False, recognize=True), "开启识别"),
        (RunOptions(mode="sitl", real_release=True), "真实投弹"),
        (RunOptions(mode="real", run_mission=True, max_seconds=1), "max_seconds"),
    ],
)
def test_invalid_options_fail_before_factory(options: RunOptions, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        options.validated()
