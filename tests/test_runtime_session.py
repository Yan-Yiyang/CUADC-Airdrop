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
    def __init__(self, calls: list[str], *, stuck: bool = False, ready: bool = True) -> None:
        self.calls = calls
        self.running = False
        self.stuck = stuck
        self.ready = ready
        self.readiness_timeouts: list[float] = []
        self.stats: VideoStats = VideoStats(frames=1)
        self.stop_on_start = False
        self.raise_if_failed: Callable[[], None] = lambda: None

    def add_sink(self, sink: Callable[[VideoFrame], None]) -> None:
        del sink

    def set_recording_directory(self, directory: str | Path) -> None:
        del directory

    def start(self) -> _Source:
        self.calls.append("source.start")
        self.running = True
        self.stats.state = "streaming"
        return self

    def wait_ready(self, timeout: float = 10.0) -> bool:
        self.readiness_timeouts.append(timeout)
        return self.ready

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
    def __init__(self, calls: list[str], flight_dir: Path | None = None) -> None:
        self.calls = calls
        self.flight_dir = flight_dir
        self.is_running = False

    def start(
        self, *, broker: TelemetryBroker, buffer: AlignmentBuffer, ts_video: bool = False
    ) -> None:
        del broker, buffer, ts_video
        self.calls.append("recorder.start")
        self.is_running = True

    def stop(self, timeout: float = 5.0) -> None:
        del timeout
        self.calls.append("recorder.stop")
        self.is_running = False


def test_crop_storage_path_is_injected_after_recorder_start(tmp_path) -> None:
    calls: list[str] = []

    class CropWorker(_Worker):
        def configure_crop_storage(self, path: Path) -> None:
            calls.append(f"worker.crop_storage:{path}")

    flight_dir = tmp_path / "flight"
    worker = CropWorker(calls)
    components = LiveComponents(
        thread=_Thread(calls),
        source=_Source(calls),
        worker=worker,
        recorder=_Recorder(calls, flight_dir),
        broker=TelemetryBroker(),
        buffer=AlignmentBuffer(10),
    )
    options = RunOptions(mode="real", run_mission=False, recognize=False, max_seconds=0.001)

    assert LiveSession(options, components).run() == 0
    assert calls.index("recorder.start") < calls.index(
        f"worker.crop_storage:{flight_dir / 'crops'}"
    )
    assert calls.index(f"worker.crop_storage:{flight_dir / 'crops'}") < calls.index("worker.start")


def test_passive_session_uses_explicit_video_readiness_timeout() -> None:
    calls: list[str] = []
    source = _Source(calls)
    components = LiveComponents(thread=_Thread(calls), source=source, readiness_timeout_s=35.0)
    options = RunOptions(mode="real", run_mission=False, recognize=False, max_seconds=0.001)

    assert LiveSession(options, components).run() == 0
    assert source.readiness_timeouts == [35.0]


def test_passive_session_default_video_readiness_timeout_is_ten_seconds() -> None:
    calls: list[str] = []
    source = _Source(calls)
    components = LiveComponents(thread=_Thread(calls), source=source)
    options = RunOptions(mode="real", run_mission=False, recognize=False, max_seconds=0.001)

    assert LiveSession(options, components).run() == 0
    assert source.readiness_timeouts == [10.0]


def test_ts_session_binds_temporary_directory_and_cleans_after_stop(tmp_path, monkeypatch) -> None:
    from airdrop.video import storage as storage_module
    from airdrop.video.cold import TsRangeDecoder

    calls: list[str] = []

    class Source(_Source):
        directory: Path | None = None

        def validate_pyav_config(self) -> None:
            calls.append("validate")

        def set_recording_directory(self, directory: str | Path) -> None:
            self.directory = Path(directory)
            calls.append("bind")

    real_storage = storage_module.TemporaryTsStorage
    monkeypatch.setattr(storage_module, "TemporaryTsStorage", lambda: real_storage(parent=tmp_path))
    source = Source(calls)
    buffer = AlignmentBuffer(10, storage="raw", cold_decoder=TsRangeDecoder(tmp_path))
    components = LiveComponents(thread=_Thread(calls), source=source, buffer=buffer, ts_video=True)
    options = RunOptions(mode="real", run_mission=False, recognize=False, max_seconds=0.001)

    assert LiveSession(options, components).run() == 0
    assert calls.index("validate") < calls.index("thread.connect") < calls.index("bind")
    assert source.directory is not None and not source.directory.exists()


def test_ts_session_binds_recording_directory_before_source_start(tmp_path) -> None:
    from airdrop.video.cold import TsRangeDecoder

    calls: list[str] = []

    class Source(_Source):
        directory: Path | None = None

        def set_recording_directory(self, directory: str | Path) -> None:
            self.directory = Path(directory)
            calls.append("bind")

    class Recorder(_Recorder):
        def start(self, *, broker, buffer, ts_video=False) -> None:
            assert ts_video
            assert self.flight_dir is not None
            (self.flight_dir / "video").mkdir(parents=True)
            super().start(broker=broker, buffer=buffer)

    source = Source(calls)
    flight_dir = tmp_path / "flight"
    buffer = AlignmentBuffer(10, storage="raw", cold_decoder=TsRangeDecoder(tmp_path))
    components = LiveComponents(
        thread=_Thread(calls),
        source=source,
        buffer=buffer,
        broker=TelemetryBroker(),
        recorder=Recorder(calls, flight_dir),
        ts_video=True,
    )
    options = RunOptions(mode="real", run_mission=False, recognize=False, max_seconds=0.001)

    assert LiveSession(options, components).run() == 0
    assert source.directory == flight_dir / "video"
    assert calls.index("recorder.start") < calls.index("bind") < calls.index("source.start")


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan"), True])
def test_live_components_reject_invalid_video_readiness_timeout(timeout: object) -> None:
    with pytest.raises(ValueError, match="视频首帧等待上限"):
        LiveComponents(thread=_Thread([]), readiness_timeout_s=timeout)  # type: ignore[arg-type]


def test_passive_session_reports_video_readiness_timeout() -> None:
    calls: list[str] = []
    source = _Source(calls, ready=False)
    components = LiveComponents(thread=_Thread(calls), source=source, readiness_timeout_s=35.0)
    options = RunOptions(mode="real", run_mission=False, recognize=False, max_seconds=0.001)

    assert LiveSession(options, components).run() == 1
    assert source.readiness_timeouts == [35.0]
    assert calls[-1] == "thread.stop"


def test_session_drains_before_result_pump_and_stops_thread_last() -> None:
    calls: list[str] = []
    worker = _Worker(calls)
    source = _Source(calls)
    buffer = AlignmentBuffer(10)
    buffer.close_cold_storage = lambda timeout=5.0: calls.append("buffer.close_cold_storage")
    components = LiveComponents(
        thread=_Thread(calls),
        source=source,
        worker=worker,
        targets=_Targets(calls),
        recorder=_Recorder(calls),
        broker=TelemetryBroker(),
        buffer=buffer,
        on_finish=lambda: calls.append("finish"),
    )
    options = RunOptions(mode="real", run_mission=False, recognize=False, max_seconds=0.001)

    assert LiveSession(options, components).run() == 0
    assert calls.index("source.stop") < calls.index("worker.drain")
    assert calls.index("worker.drain") < calls.index("targets.pump")
    assert calls.index("targets.pump") < calls.index("recorder.stop")
    assert calls.index("recorder.stop") < calls.index("buffer.close_cold_storage")
    assert (
        calls.index("buffer.close_cold_storage")
        < calls.index("finish")
        < calls.index("thread.stop")
    )


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


@pytest.mark.parametrize("failed_component", ["source", "buffer"])
def test_passive_session_propagates_video_storage_health_failure(failed_component: str) -> None:
    calls: list[str] = []
    source = _Source(calls)
    buffer = AlignmentBuffer(10)

    def fail() -> None:
        raise OSError(f"模拟 {failed_component} 持久化故障")

    if failed_component == "source":
        source.raise_if_failed = fail
    else:
        buffer.raise_if_failed = fail
    components = LiveComponents(
        thread=_Thread(calls), source=source, buffer=buffer, broker=TelemetryBroker()
    )
    options = RunOptions(mode="real", run_mission=False, recognize=False, max_seconds=0.01)

    assert LiveSession(options, components).run() == 1
    assert calls[-1] == "thread.stop"


def test_prepare_perception_runs_before_worker_and_video() -> None:
    calls: list[str] = []
    components = LiveComponents(
        thread=_Thread(calls),
        source=_Source(calls),
        worker=_Worker(calls),
        prepare_perception=lambda: calls.append("perception.prepare"),
    )
    options = RunOptions(mode="real", run_mission=False, recognize=False, max_seconds=0.001)

    assert LiveSession(options, components).run() == 0
    assert calls.index("thread.connect") < calls.index("perception.prepare")
    assert calls.index("perception.prepare") < calls.index("worker.start")
    assert calls.index("worker.start") < calls.index("source.start")


def test_prepare_perception_failure_rolls_back_without_worker_drain() -> None:
    calls: list[str] = []

    def fail_prepare() -> None:
        calls.append("perception.prepare")
        raise RuntimeError("预热失败")

    components = LiveComponents(
        thread=_Thread(calls),
        source=_Source(calls),
        worker=_Worker(calls),
        prepare_perception=fail_prepare,
    )
    options = RunOptions(mode="real", run_mission=False, recognize=False, max_seconds=0.001)

    assert LiveSession(options, components).run() == 1
    assert "worker.start" not in calls
    assert "worker.drain" not in calls
    assert "source.start" not in calls
    assert "worker.stop" in calls
    assert calls[-1] == "thread.stop"


def test_worker_exit_after_start_still_requires_successful_drain() -> None:
    calls: list[str] = []

    class ExitedWorker(_Worker):
        def start(self) -> None:
            super().start()
            self.running = False

        def drain(self, timeout: float = 120.0) -> bool:
            del timeout
            calls.append("worker.drain")
            return False

    components = LiveComponents(
        thread=_Thread(calls), source=_Source(calls), worker=ExitedWorker(calls)
    )
    options = RunOptions(mode="real", run_mission=False, recognize=False, max_seconds=0.001)

    assert LiveSession(options, components).run() == 1
    assert "worker.drain" in calls
    assert "worker.stop" in calls
    assert calls[-1] == "thread.stop"


def test_prepare_perception_interrupt_rolls_back_before_worker_and_video() -> None:
    calls: list[str] = []

    def interrupt_prepare() -> None:
        calls.append("perception.prepare")
        raise KeyboardInterrupt

    components = LiveComponents(
        thread=_Thread(calls),
        source=_Source(calls),
        worker=_Worker(calls),
        prepare_perception=interrupt_prepare,
    )

    assert (
        LiveSession(
            RunOptions(mode="real", run_mission=False, recognize=False, max_seconds=0.001),
            components,
        ).run()
        == 130
    )
    assert "worker.start" not in calls
    assert "worker.drain" not in calls
    assert "source.start" not in calls
    assert "worker.stop" in calls
    assert calls[-1] == "thread.stop"


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
