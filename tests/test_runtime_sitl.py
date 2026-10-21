"""统一 SITL 适配器的阶段、报告与收尾回归。"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

pytest.importorskip("airdrop_sim", reason="SITL 集成需要仿真子模块")

from airdrop import TelemetryBroker, TelemetrySnapshot, analyze
from airdrop.mission.runner import MissionRunner
from airdrop.mission.states import MissionState
from airdrop.perception.pipeline import PerceptionStats
from airdrop.preflight import Preflight
from airdrop.runtime import sitl, sitl_geometry, sitl_operations
from airdrop.runtime.options import RunOptions
from airdrop.runtime.session import LiveComponents


class _Thread:
    def __init__(self) -> None:
        self.stopped = False

    def connect(self) -> None:
        pass

    def stop(self, timeout: float = 5.0) -> None:
        self.stopped = True


def _write_recon_plan(path: Path) -> Path:
    """写入一条有效的 QGC 固定翼侦查航线。"""
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
                            "command": 178,
                            "frame": 2,
                            "doJumpId": 1,
                            "autoContinue": True,
                            "params": [1.0, 15.0, 0.0, None, 0.0, 0.0, 0.0],
                        },
                        {
                            "type": "SimpleItem",
                            "command": 16,
                            "frame": 3,
                            "doJumpId": 2,
                            "autoContinue": True,
                            "Altitude": 50.0,
                            "params": [0.0, 0.0, 0.0, None, 47.001, 8.002, 50.0],
                        },
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    return path


class _Source:
    running = False
    stats = SimpleNamespace(frames=1, dropped=0, reconnects=0, last_error=None)

    def add_sink(self, _sink: Any) -> None:
        pass

    def start(self) -> None:
        pass

    def stop(self, timeout: float = 5.0) -> None:
        pass

    def wait_ready(self, timeout: float = 10.0) -> bool:
        return True


class _Worker:
    running = False

    def __init__(self) -> None:
        self.stats = PerceptionStats()
        self.queued_results = 0

    def start(self) -> None:
        pass

    def stop(self, timeout: float = 10.0) -> None:
        pass

    def drain(self, timeout: float = 120.0) -> bool:
        return True

    def raise_if_failed(self) -> None:
        pass


class _Writer:
    def __call__(self, _frame: Any) -> None:
        pass

    def stats(self) -> tuple[int, int]:
        return (1, 0)


class _Tracker:
    def __init__(self, config: Any) -> None:
        self.points = None
        self.stats = {"points": 0, "labeled": 0, "no_fix": 0, "side_mismatch": 0}
        self._result = analyze((), config.targeting)

    def result(self) -> Any:
        return self._result


class _Runner:
    state = MissionState.DONE
    stats = SimpleNamespace(uploads=0, releases=0, aborts=0, errors=0)
    history: tuple[Any, ...] = ()
    origin = None

    def run(self) -> MissionState:
        return MissionState.DONE

    def stop(self, reason: str = "stopped") -> None:
        pass

    def close(self, *, timeout: float | None = None) -> None:
        pass


class _Monitor:
    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        self.samples = [
            {"t_s": 0.0, "frames": 0, "submitted": 0, "results": 0, "deduped": 0},
            {"t_s": 5.0, "frames": 10, "submitted": 4, "results": 3, "deduped": 1},
        ]

    def start(self) -> None:
        pass

    def stop(self) -> list[dict[str, Any]]:
        return self.samples


def _components(config: Any, *, passive: bool = False) -> LiveComponents:
    broker = TelemetryBroker()
    broker.publish(
        TelemetrySnapshot(
            north_m=0.0,
            east_m=0.0,
            down_m=0.0,
            quaternion_w=1.0,
            origin_latitude_deg=47.397971,
            origin_longitude_deg=8.545593,
            origin_altitude_m=488.0,
        )
    )
    worker = None if passive else _Worker()
    tracker = None if passive else _Tracker(config)
    targets = None if passive else SimpleNamespace(tracker=tracker, pump=lambda: 0)
    runner = None if passive else _Runner()
    return LiveComponents(
        thread=cast(Any, _Thread()),
        source=cast(Any, None if passive else _Source()),
        writer=cast(Any, None if passive else _Writer()),
        worker=cast(Any, worker),
        targets=cast(Any, targets),
        runner=cast(Any, runner),
        controller=object() if runner is not None else None,
        broker=cast(Any, broker),
        buffer=cast(Any, object()),
    )


def test_sitl_adapter_prepares_config_and_writes_report_with_performance(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(sitl, "PerformanceMonitor", _Monitor)
    captured: dict[str, Any] = {}
    components: LiveComponents | None = None

    def factory(options: RunOptions, config: Any) -> LiveComponents:
        nonlocal components
        captured["config"] = config
        components = _components(config)
        return components

    assert (
        sitl.main(
            mode="sitl",
            auto_recon=False,
            auto_land=False,
            drop_stage=False,
            record=False,
            work_dir=str(tmp_path),
            components_factory=factory,
        )
        == 0
    )
    config = captured["config"]
    assert config.mission.drop_stage is False
    assert config.mission.auto_land is False
    assert config.preflight.load_detector is True
    payload = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert payload["performance"]["samples"] == 2
    assert payload["performance"]["ocr_requests"] == 4
    assert payload["perception"]["frames"] == 0
    assert components is not None and components.thread.stopped


def test_sitl_passive_capture_does_not_require_ned_origin_or_write_report_files(
    tmp_path: Path,
) -> None:
    captured: dict[str, Any] = {}

    def factory(options: RunOptions, config: Any) -> LiveComponents:
        captured["options"] = options
        return _components(config, passive=True)

    assert (
        sitl.main(
            mode="sitl",
            run_mission=False,
            capture=False,
            recognize=False,
            record=False,
            max_seconds=0.01,
            work_dir=str(tmp_path),
            components_factory=factory,
        )
        == 0
    )
    assert not (tmp_path / sitl_geometry.STREAM_SDP_NAME).exists()
    assert not (tmp_path / "camera_calib_sim.json").exists()
    assert not (tmp_path / "report.json").exists()


def test_explicit_environment_overrides_update_supplied_config(tmp_path: Path) -> None:
    base = sitl_geometry.build_config(land_plan="")
    captured: dict[str, Any] = {}

    def factory(_options: RunOptions, config: Any) -> LiveComponents:
        captured["config"] = config
        return _components(config, passive=True)

    calib = tmp_path / "explicit-calibration.json"
    assert (
        sitl.main(
            config=base,
            system_address="udpin://127.0.0.1:14541",
            calib_file=str(calib),
            recon_timeout_s=37.0,
            run_mission=False,
            capture=False,
            recognize=False,
            record=False,
            max_seconds=0.01,
            work_dir=str(tmp_path),
            components_factory=factory,
        )
        == 0
    )
    prepared = captured["config"]
    assert prepared.telemetry.system_address == "udpin://127.0.0.1:14541"
    assert prepared.camera.calib_file == str(calib)
    assert prepared.mission.recon_max_s == 37.0


@pytest.mark.integration
def test_sitl_optional_tool_loads_without_pytest_conftest_before_config_rejection() -> None:
    root = Path(__file__).resolve().parents[1]
    script = """
from airdrop.runtime import sitl

def forbidden_factory(*_args, **_kwargs):
    raise AssertionError('factory must not run before manifest validation')

try:
    sitl.main(
        recon_plan='missing-custom-recon.plan',
        run_mission=True,
        capture=False,
        recognize=False,
        record=False,
        drop_stage=False,
        auto_recon=True,
        auto_land=False,
        components_factory=forbidden_factory,
    )
except ValueError as exc:
    assert '不存在' in str(exc)
else:
    raise AssertionError('missing recon plan should be rejected')
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_takeoff_bootstrap_uses_first_positional_item_from_recon_plan(tmp_path: Path) -> None:
    plan = _write_recon_plan(tmp_path / "recon.plan")
    base = sitl_geometry.build_config(land_plan="")
    config = replace(
        base,
        routes=replace(base.routes, recon_route=(), recon_plan=str(plan)),
        mission=replace(base.mission, takeoff_first=False),
    )
    from airdrop import load_plan

    first = next(item for item in load_plan(plan).items if item.is_positional)
    takeoff = sitl_operations.takeoff_items(config)[0]
    assert (takeoff.lat, takeoff.lon) == (first.lat, first.lon)


def test_sitl_main_accepts_custom_recon_plan_without_inline_route(
    tmp_path: Path, monkeypatch
) -> None:
    plan = _write_recon_plan(tmp_path / "recon.plan")
    monkeypatch.setattr(
        sitl_operations,
        "zero_test_params",
        lambda _thread, ledger: ledger.update({"NAV_DLL_ACT": 2}),
    )
    monkeypatch.setattr(sitl_operations, "operator_takeoff", lambda *_args: 0)
    monkeypatch.setattr(sitl_operations, "restore_params", lambda *_args: None)
    captured: dict[str, Any] = {}

    def factory(_options: RunOptions, config: Any) -> LiveComponents:
        captured["config"] = config
        return _components(config)

    assert (
        sitl.main(
            recon_plan=str(plan),
            capture=False,
            recognize=False,
            record=False,
            drop_stage=False,
            auto_recon=True,
            auto_land=False,
            work_dir=str(tmp_path),
            components_factory=factory,
        )
        == 0
    )
    prepared = captured["config"]
    assert prepared.routes.recon_plan == str(plan)
    assert prepared.routes.recon_route == ()
    assert sitl_operations.takeoff_items(prepared)
    assert not (tmp_path / sitl_geometry.STREAM_SDP_NAME).exists()
    assert not (tmp_path / "camera_calib_sim.json").exists()


def test_report_failure_still_stops_the_sdk_thread(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(sitl, "PerformanceMonitor", _Monitor)
    monkeypatch.setattr(
        "airdrop.runtime.sitl_report._report",
        lambda _outcome: (_ for _ in ()).throw(KeyboardInterrupt("report interrupted")),
    )
    components: LiveComponents | None = None

    def factory(_options: RunOptions, config: Any) -> LiveComponents:
        nonlocal components
        components = _components(config)
        return components

    assert (
        sitl.main(
            mode="sitl",
            auto_recon=False,
            auto_land=False,
            record=False,
            work_dir=str(tmp_path),
            components_factory=factory,
        )
        == 1
    )
    assert components is not None and components.thread.stopped


def test_restore_and_report_errors_still_wait_for_rpc_finally_before_thread_stop(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(sitl, "PerformanceMonitor", _Monitor)
    monkeypatch.setattr(
        "airdrop.runtime.sitl_report._report",
        lambda _outcome: (_ for _ in ()).throw(KeyboardInterrupt("report interrupted")),
    )
    monkeypatch.setattr(
        sitl_operations,
        "zero_test_params",
        lambda _thread, ledger: ledger.update({"NAV_DLL_ACT": 2}),
    )
    monkeypatch.setattr(sitl_operations, "operator_takeoff", lambda *_args: 0)
    monkeypatch.setattr(
        sitl_operations,
        "restore_params",
        lambda *_args: (_ for _ in ()).throw(KeyboardInterrupt("restore interrupted")),
    )
    events: list[str] = []

    class DelayedGate:
        def __init__(self) -> None:
            self.finally_done = threading.Event()

        def wait_idle(self) -> None:
            events.append("gate.wait")
            assert self.finally_done.wait(timeout=2.0)
            events.append("rpc.finally")

    gate = DelayedGate()
    releaser = threading.Thread(
        target=lambda: (threading.Event().wait(0.05), gate.finally_done.set()), daemon=True
    )
    releaser.start()
    components: LiveComponents | None = None

    def factory(_options: RunOptions, config: Any) -> LiveComponents:
        nonlocal components
        components = _components(config)
        thread = components.thread
        thread._operation_gate = gate  # type: ignore[attr-defined]
        thread.stop = lambda timeout=5.0: (  # type: ignore[method-assign]
            events.append("thread.stop"),
            setattr(thread, "stopped", True),
        )

        def previous_finish() -> None:
            events.append("factory.finish")
            raise KeyboardInterrupt("factory finish interrupted")

        components.on_finish = previous_finish
        return components

    assert (
        sitl.main(
            auto_recon=True,
            auto_land=False,
            drop_stage=False,
            record=False,
            work_dir=str(tmp_path),
            components_factory=factory,
        )
        == 1
    )
    assert events[-3:] == ["gate.wait", "rpc.finally", "thread.stop"]
    assert "factory.finish" in events
    assert components is not None and components.thread.stopped


def test_nonzero_factory_start_callback_prevents_sitl_parameters_and_takeoff(
    tmp_path: Path, monkeypatch
) -> None:
    def forbidden(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("前置回调失败后仍执行 SITL 参数修改或起飞")

    monkeypatch.setattr(sitl_operations, "zero_test_params", forbidden)
    monkeypatch.setattr(sitl_operations, "operator_takeoff", forbidden)
    components: LiveComponents | None = None

    def factory(_options: RunOptions, config: Any) -> LiveComponents:
        nonlocal components
        components = _components(config)
        components.on_start = lambda: 7
        return components

    assert (
        sitl.main(
            capture=False,
            recognize=False,
            record=False,
            drop_stage=False,
            auto_recon=True,
            auto_land=False,
            work_dir=str(tmp_path),
            components_factory=factory,
        )
        == 1
    )
    assert components is not None and components.thread.stopped


def test_auto_recon_bootstrap_uses_shared_runner_upload_and_restores_params(
    tmp_path: Path, monkeypatch
) -> None:
    from airdrop import Config, Waypoint

    base = Config()
    config = replace(
        base,
        routes=replace(
            base.routes,
            recon_route=(Waypoint(lat=47.397971, lon=8.545593, alt_m=30.0),),
        ),
        preflight=replace(
            base.preflight,
            load_detector=False,
            load_ocr=False,
            load_camera=False,
            check_video=False,
        ),
        mission=replace(
            base.mission,
            require_airborne=False,
            drop_stage=False,
            auto_land=False,
            tick_hz=100.0,
        ),
    )
    events: list[str] = []
    controller_ref: list[Any] = []
    runner_ref: list[MissionRunner] = []
    restored: list[dict[str, int]] = []

    class Controller:
        def __init__(self) -> None:
            self.finished_queries = 0
            self.uploads: list[tuple[Any, ...]] = []

        def request_origin(self) -> Any:
            return SimpleNamespace(lat_deg=47.397971, lon_deg=8.545593, alt_m=488.0)

        def upload_mission(self, items: Any) -> int:
            events.append("runner.upload")
            self.uploads.append(tuple(items))
            return len(items)

        def start_mission(self) -> None:
            events.append("runner.start")

        def in_mission_mode(self) -> bool:
            return True

        def mission_finished(self) -> bool:
            self.finished_queries += 1
            return self.finished_queries > 1

        def hold(self) -> None:
            events.append("runner.hold")

        def wait_pending(self, *, timeout: float | None = None) -> None:
            pass

    def fake_zero(_thread: Any, ledger: dict[str, int]) -> None:
        events.append("bootstrap.params")
        ledger["NAV_DLL_ACT"] = 2

    def fake_takeoff(_config: Any, controller: Any, _thread: Any, _broker: Any) -> int:
        events.append("bootstrap.takeoff")
        controller_ref.append(controller)
        assert _config.mission.takeoff_first is False
        return 0

    monkeypatch.setattr(sitl_operations, "zero_test_params", fake_zero)
    monkeypatch.setattr(sitl_operations, "operator_takeoff", fake_takeoff)
    monkeypatch.setattr(
        sitl_operations, "restore_params", lambda _thread, ledger: restored.append(dict(ledger))
    )

    def factory(_options: RunOptions, prepared: Any) -> LiveComponents:
        broker = TelemetryBroker()
        broker.publish(
            TelemetrySnapshot(
                north_m=0.0,
                east_m=0.0,
                down_m=0.0,
                quaternion_w=1.0,
                origin_latitude_deg=47.397971,
                origin_longitude_deg=8.545593,
                origin_altitude_m=488.0,
            )
        )
        controller = Controller()
        runner = MissionRunner(
            prepared,
            cast(Any, controller),
            broker,
            preflight=Preflight(prepared.preflight),
        )
        runner_ref.append(runner)
        return LiveComponents(
            thread=cast(Any, _Thread()),
            runner=cast(Any, runner),
            controller=controller,
            broker=broker,
        )

    assert (
        sitl.main(
            config=config,
            capture=False,
            recognize=False,
            record=False,
            drop_stage=False,
            auto_recon=True,
            auto_land=False,
            work_dir=str(tmp_path),
            components_factory=factory,
        )
        == 0
    )
    assert events[:3] == ["bootstrap.params", "bootstrap.takeoff", "runner.upload"]
    assert events[3] == "runner.start"
    assert controller_ref[0] is not None
    assert runner_ref[0].stats.uploads == 1
    assert len(controller_ref[0].uploads) == 1
    assert restored == [{"NAV_DLL_ACT": 2}]
