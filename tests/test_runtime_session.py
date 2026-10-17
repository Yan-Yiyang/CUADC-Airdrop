"""统一运行会话的清理次序与失败语义回归。"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

from airdrop import Config, Waypoint
from airdrop.mission.runner import MissionRunner, MissionStats
from airdrop.mission.states import MissionState
from airdrop.mission.targets import TargetTracker
from airdrop.runtime.options import RunOptions
from airdrop.runtime.session import LiveComponents, LiveSession
from airdrop.targeting.models import TargetingResult
from airdrop.telemetry.broker import TelemetryBroker
from airdrop.telemetry.controller import MissionController, NedOrigin
from airdrop.telemetry.models import TelemetrySnapshot
from airdrop.video.buffer import AlignmentBuffer
from airdrop.video.source import VideoFrame, VideoStats


class _Thread:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    def connect(self) -> None:
        self.calls.append("thread.connect")

    def stop(self, timeout: float = 5.0) -> None:
        del timeout
        self.calls.append("thread.stop")


class _Worker:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls
        self.running = False

    def start(self) -> None:
        self.calls.append("worker.start")
        self.running = True

    def raise_if_failed(self) -> None:
        return

    def drain(self, timeout: float = 120.0) -> bool:
        del timeout
        self.calls.append("worker.drain")
        return True

    def stop(self, timeout: float = 10.0) -> None:
        del timeout
        self.calls.append("worker.stop")
        self.running = False


class _Source:
    def __init__(self, calls: list[str], *, stuck: bool = False) -> None:
        self.calls = calls
        self.running = False
        self.stuck = stuck
        self.stats: VideoStats = VideoStats(frames=1)
        self.stop_on_start = False

    def add_sink(self, sink: Callable[[VideoFrame], None]) -> None:
        del sink

    def start(self) -> _Source:
        self.calls.append("source.start")
        self.running = True
        self.stats.state = "streaming"
        return self

    def wait_ready(self, timeout: float = 10.0) -> bool:
        del timeout
        return True

    def stop(self, timeout: float = 5.0) -> None:
        del timeout
        self.calls.append("source.stop")
        self.running = self.stuck
        self.stats.state = "streaming" if self.stuck else "stopped"


class _Runner:
    def __init__(self, calls: list[str], state: MissionState) -> None:
        self.calls = calls
        self.state = state
        self.stats = MissionStats()
        self.history = ()
        self.origin = None

    def run(self) -> MissionState:
        self.calls.append("runner.run")
        return self.state

    def stop(self, reason: str = "stopped") -> None:
        self.calls.append(f"runner.stop:{reason}")

    def close(self, *, timeout: float | None = None) -> None:
        del timeout
        self.calls.append("runner.close")


class _Targets:
    tracker = cast(TargetTracker, None)

    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    def pump(self) -> int:
        self.calls.append("targets.pump")
        return 0

    def result(self) -> TargetingResult:
        return cast(TargetingResult, None)

    def busy(self) -> bool:
        return False


class _Recorder:
    flight_dir: Path | None = None

    def __init__(self, calls: list[str]) -> None:
        self.calls = calls
        self.is_running = False

    def start(self, *, broker: TelemetryBroker, buffer: AlignmentBuffer) -> None:
        del broker, buffer
        self.calls.append("recorder.start")
        self.is_running = True

    def stop(self, timeout: float = 5.0) -> None:
        del timeout
        self.calls.append("recorder.stop")
        self.is_running = False


def test_session_drains_before_result_pump_and_stops_thread_last() -> None:
    calls: list[str] = []
    worker = _Worker(calls)
    source = _Source(calls)
    components = LiveComponents(
        thread=_Thread(calls),
        source=source,
        worker=worker,
        targets=_Targets(calls),
        recorder=_Recorder(calls),
        broker=TelemetryBroker(),
        buffer=AlignmentBuffer(10),
        on_finish=lambda: calls.append("finish"),
    )
    options = RunOptions(mode="real", run_mission=False, recognize=False, max_seconds=0.001)

    assert LiveSession(options, components).run() == 0
    assert calls.index("source.stop") < calls.index("worker.drain")
    assert calls.index("worker.drain") < calls.index("targets.pump")
    assert calls.index("targets.pump") < calls.index("recorder.stop")
    assert calls.index("recorder.stop") < calls.index("finish") < calls.index("thread.stop")


def test_stuck_video_source_fails_and_still_releases_everything() -> None:
    calls: list[str] = []
    components = LiveComponents(
        thread=_Thread(calls), source=_Source(calls, stuck=True), worker=_Worker(calls)
    )
    options = RunOptions(mode="real", run_mission=False, recognize=False, max_seconds=0.001)

    assert LiveSession(options, components).run() == 1
    assert "worker.drain" not in calls
    assert calls[-1] == "thread.stop"
    assert "worker.stop" in calls


def test_non_done_runner_state_is_failure_and_closes_runner() -> None:
    calls: list[str] = []
    components = LiveComponents(
        thread=_Thread(calls), runner=_Runner(calls, cast(MissionState, "DONE"))
    )

    assert LiveSession(RunOptions(mode="real"), components).run() == 1
    assert "runner.close" in calls
    assert calls[-1] == "thread.stop"


def test_done_enum_is_the_only_success_state() -> None:
    calls: list[str] = []
    components = LiveComponents(thread=_Thread(calls), runner=_Runner(calls, MissionState.DONE))

    assert LiveSession(RunOptions(mode="real"), components).run() == 0


def test_passive_video_stopping_after_first_frame_fails() -> None:
    calls: list[str] = []
    source = _Source(calls)
    options = RunOptions(mode="real", run_mission=False, recognize=False, max_seconds=0.001)

    def stop_during_passive_wait() -> None:
        source.running = False
        source.stats.state = "stopped"

    components = LiveComponents(
        thread=_Thread(calls), source=source, on_start=stop_during_passive_wait
    )
    assert LiveSession(options, components).run() == 1


def test_passive_video_recovery_can_finish_with_streaming_state() -> None:
    calls: list[str] = []
    source = _Source(calls)
    options = RunOptions(mode="real", run_mission=False, recognize=False, max_seconds=0.001)
    components = LiveComponents(thread=_Thread(calls), source=source)
    assert LiveSession(options, components).run() == 0


def test_passive_video_still_reconnecting_at_deadline_fails() -> None:
    calls: list[str] = []
    source = _Source(calls)
    options = RunOptions(mode="real", run_mission=False, recognize=False, max_seconds=0.001)

    def set_reconnecting() -> None:
        source.stats.state = "reconnecting"

    components = LiveComponents(thread=_Thread(calls), source=source, on_start=set_reconnecting)
    assert LiveSession(options, components).run() == 1


def test_mission_without_recognition_waits_for_video_without_camera_preflight() -> None:
    calls: list[str] = []
    source = _Source(calls)
    runner = _Runner(calls, MissionState.DONE)
    components = LiveComponents(thread=_Thread(calls), source=source, runner=runner)

    assert LiveSession(RunOptions(mode="real", recognize=False), components).run() == 0
    assert "source.start" in calls
    assert "runner.run" in calls
    assert calls[-1] == "thread.stop"


def test_transfer_submit_keyboard_interrupt_still_stops_all_resources() -> None:
    calls: list[str] = []

    class Controller:
        def request_origin(self) -> NedOrigin:
            return NedOrigin(lat_deg=31.0, lon_deg=121.0, alt_m=10.0)

        def hold(self) -> None:
            calls.append("controller.hold")

    class Broker:
        def get_snapshot(self) -> TelemetrySnapshot:
            return TelemetrySnapshot(north_m=0.0, east_m=0.0, down_m=0.0)

    def interrupted_submit[T](_operation: Callable[[], T], /) -> Future[T]:
        raise KeyboardInterrupt

    base = Config()
    config = replace(
        base,
        mission=replace(
            base.mission,
            recon_upload="auto",
            require_airborne=False,
            abort_action="hold",
        ),
        routes=replace(base.routes, recon_route=(Waypoint(lat=31.1, lon=121.1, alt_m=30.0),)),
    )
    runner = MissionRunner(
        config,
        cast(MissionController, Controller()),
        cast(TelemetryBroker, Broker()),
        clock=lambda: 1_000.0,
        sleep=lambda _seconds: None,
        transfer_submit=interrupted_submit,
    )
    components = LiveComponents(
        thread=_Thread(calls),
        worker=_Worker(calls),
        recorder=_Recorder(calls),
        runner=runner,
        broker=TelemetryBroker(),
        buffer=AlignmentBuffer(10),
    )

    assert LiveSession(RunOptions(mode="real"), components).run() == 130
    assert "worker.stop" in calls
    assert "recorder.stop" in calls
    assert "thread.stop" in calls
    assert calls[-1] == "thread.stop"


def test_on_finish_keyboard_interrupt_still_stops_thread() -> None:
    calls: list[str] = []

    def interrupt_cleanup() -> None:
        calls.append("finish")
        raise KeyboardInterrupt

    components = LiveComponents(thread=_Thread(calls), on_finish=interrupt_cleanup)
    options = RunOptions(
        mode="real", run_mission=False, capture=False, recognize=False, max_seconds=0.001
    )

    assert LiveSession(options, components).run() == 130
    assert calls == ["thread.connect", "finish", "thread.stop"]


def test_system_exit_during_start_still_runs_cleanup() -> None:
    calls: list[str] = []

    def exit_start() -> None:
        raise SystemExit(7)

    components = LiveComponents(thread=_Thread(calls), on_start=exit_start)

    with pytest.raises(SystemExit, match="7"):
        LiveSession(RunOptions(mode="real"), components).run()
    assert calls[-1] == "thread.stop"
