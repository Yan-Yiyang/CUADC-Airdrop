from __future__ import annotations

import math
import threading
import time
from dataclasses import replace

import pytest

from airdrop import Config
from airdrop.ballistics.model import BallisticsModel
from airdrop.config import GroundConfig
from airdrop.qgcvideo.prediction import LiveImpactPredictor, _Request
from airdrop.telemetry.models import TelemetrySnapshot


def _snapshot(**changes: object) -> TelemetrySnapshot:
    values: dict[str, object] = {
        "timestamp": time.time(),
        "north_m": 10.0,
        "east_m": 20.0,
        "down_m": -20.0,
        "vx_m_s": 15.0,
        "vy_m_s": 2.0,
        "vz_m_s": 0.0,
        "origin_altitude_m": 100.0,
        "in_air": True,
        "position_velocity_received_at": time.time(),
    }
    values.update(changes)
    return TelemetrySnapshot(**values)  # type: ignore[arg-type]


def _config() -> Config:
    return replace(Config(), ground=GroundConfig(ground_point_alt=0.0))


def test_live_prediction_matches_ballistics_model_without_release_judge() -> None:
    config = replace(_config(), drop=replace(Config().drop, delay_s=0.15))
    predictor = LiveImpactPredictor(config)
    snapshot = _snapshot(wind_north_m_s=2.0, wind_east_m_s=0.0, wind_down_m_s=0.0)
    request = _Request(snapshot, (50.0, 30.0, 0.0), "OVERFLY", 1)

    result = predictor._compute(request)
    assert result.valid
    assert result.input_position_ned == (10.0, 20.0, -20.0)
    assert result.input_velocity_ned == (15.0, 2.0, 0.0)
    expected = BallisticsModel(config.ballistics).predict_impact(
        (12.25, 20.3, -20.0),
        (15.0, 2.0, 0.0),
        ground_z=100.0,
        ground_altitude_m=0.0,
        wind=(2.0, 0.0, 0.0),
    )
    assert expected.ok and expected.ned is not None
    assert result.impact_ned == pytest.approx(expected.ned)
    assert result.error_m == pytest.approx(math.hypot(expected.ned[0] - 50, expected.ned[1] - 30))
    assert result.wind_available


def test_position_speed_and_wind_change_the_live_impact() -> None:
    predictor = LiveImpactPredictor(_config())
    target = (100.0, 0.0, 0.0)
    base = predictor._compute(_Request(_snapshot(), target, "OVERFLY", 1))
    moved = predictor._compute(
        _Request(_snapshot(north_m=110.0, vx_m_s=20.0, wind_east_m_s=4.0), target, "OVERFLY", 1)
    )
    assert base.valid and moved.valid
    assert base.impact_ned != moved.impact_ned


def test_extremely_small_step_is_bounded_by_display_compute_budget() -> None:
    config = replace(_config(), ballistics=replace(Config().ballistics, rk_dt=1e-12))
    predictor = LiveImpactPredictor(config)
    started = time.monotonic()
    result = predictor._compute(_Request(_snapshot(), (50.0, 30.0, 0.0), "OVERFLY", 1))
    assert not result.valid
    assert "超出显示时限" in result.reason
    assert time.monotonic() - started < 1.0


@pytest.mark.parametrize(
    ("changes", "stage", "reason"),
    [
        ({"in_air": False}, "OVERFLY", "未离地"),
        ({"in_air": None}, "OVERFLY", "未离地"),
        ({"north_m": math.nan}, "OVERFLY", "非有限"),
        ({"origin_altitude_m": None}, "OVERFLY", "原点海拔未知"),
        ({"position_velocity_received_at": time.time() - 1.0}, "OVERFLY", "过期"),
        ({}, "RECON", "尚未进入"),
    ],
)
def test_live_prediction_fails_closed(changes: dict[str, object], stage: str, reason: str) -> None:
    predictor = LiveImpactPredictor(_config())
    result = predictor._compute(_Request(_snapshot(**changes), (1.0, 2.0, 0.0), stage, 3))
    assert not result.valid
    assert reason in result.reason
    assert result.impact_ned is None


def test_generation_changes_with_target_stage_and_origin() -> None:
    predictor = LiveImpactPredictor(_config())
    predictor.start()
    try:
        first = predictor.submit(_snapshot(), (1.0, 2.0, 0.0), "OVERFLY")
        second = predictor.submit(_snapshot(), (2.0, 2.0, 0.0), "OVERFLY")
        third = predictor.submit(_snapshot(origin_altitude_m=101.0), (2.0, 2.0, 0.0), "LAND")
        assert first < second < third
    finally:
        predictor.stop()
    assert predictor._thread is None


def test_stop_cancels_an_inflight_model_calculation(monkeypatch: pytest.MonkeyPatch) -> None:
    predictor = LiveImpactPredictor(_config())
    entered = threading.Event()
    original = predictor._model.predict_impact

    def slow(*args: object, **kwargs: object):
        entered.set()
        cancel = kwargs["should_cancel"]
        while not cancel():  # type: ignore[operator]
            time.sleep(0.001)
        return original(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(predictor._model, "predict_impact", slow)
    predictor.start()
    predictor.submit(_snapshot(), (1.0, 2.0, 0.0), "OVERFLY")
    assert entered.wait(1.0)
    predictor.stop(timeout=1.0)
    assert predictor._thread is None


def test_thirty_hz_submission_still_publishes_ten_hz_results() -> None:
    predictor = LiveImpactPredictor(_config())
    completed: set[float] = set()
    predictor.start()
    try:
        until = time.monotonic() + 0.55
        while time.monotonic() < until:
            predictor.submit(_snapshot(), (50.0, 30.0, 0.0), "OVERFLY")
            result = predictor.current()
            if result is not None and result.valid:
                completed.add(result.computed_mono)
            time.sleep(1.0 / 30.0)
        result = predictor.current()
        if result is not None and result.valid:
            completed.add(result.computed_mono)
        assert len(completed) >= 3
    finally:
        predictor.stop()
