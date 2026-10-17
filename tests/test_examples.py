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
