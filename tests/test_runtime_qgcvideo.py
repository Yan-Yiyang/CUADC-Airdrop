"""QGC 发布器在统一会话中的独立生命周期。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import ClassVar

import pytest

from airdrop import Config
from airdrop.config import QgcVideoConfig
from airdrop.runtime.options import RunOptions
from airdrop.runtime.qgc import QgcVideoRuntime
from airdrop.runtime.session import LiveComponents, LiveSession
from airdrop.telemetry.broker import TelemetryBroker
from airdrop.video.source import VideoFrame, VideoStats


class _Thread:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    def connect(self) -> None:
        self.calls.append("thread.connect")

    def stop(self, timeout: float = 5.0) -> None:
        del timeout
        self.calls.append("thread.stop")


class _Source:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls
        self.running = False
        self.stats = VideoStats(frames=1)

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
        self.running = False
        self.stats.state = "stopped"


class _Worker:
    def __init__(self, calls: list[str]) -> None:
        self.calls = calls
        self.running = False

    def start(self) -> None:
        self.calls.append("worker.start")
        self.running = True

    def drain(self, timeout: float = 120.0) -> bool:
        del timeout
        self.calls.append("worker.drain")
        return True

    def stop(self, timeout: float = 10.0) -> None:
        del timeout
        self.calls.append("worker.stop")
        self.running = False

    def raise_if_failed(self) -> None:
        return


class _Display:
    def __init__(self, calls: list[str], *, fail_start: bool = False) -> None:
        self.calls = calls
        self.fail_start = fail_start
        self.running = False

    def start(self) -> None:
        self.calls.append("display.start")
        if self.fail_start:
            return
        self.running = True

    def stop(self, timeout: float | None = None) -> None:
        del timeout
        self.calls.append("display.stop")
        self.running = False


def _run(calls: list[str], display: _Display) -> int:
    source = _Source(calls)
    worker = _Worker(calls)
    return LiveSession(
        RunOptions(mode="real", run_mission=False, recognize=False, max_seconds=0.001),
        LiveComponents(thread=_Thread(calls), source=source, worker=worker, display=display),
    ).run()


def test_display_runs_beside_capture_and_stops_after_worker_drain() -> None:
    calls: list[str] = []

    assert _run(calls, _Display(calls)) == 0
    assert calls.index("source.stop") < calls.index("worker.drain")
    assert calls.index("worker.drain") < calls.index("display.stop")
    assert calls.index("display.stop") < calls.index("thread.stop")


def test_display_start_failure_does_not_fail_capture_session() -> None:
    calls: list[str] = []

    assert _run(calls, _Display(calls, fail_start=True)) == 0
    assert "display.start" in calls
    assert "display.stop" in calls
    assert calls[-1] == "thread.stop"


def test_final_snapshot_failure_still_stops_qgc_publisher(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from airdrop.qgcvideo import publisher

    class FakePublisher:
        instances: ClassVar[list[FakePublisher]] = []

        def __init__(
            self,
            config: object,
            frame_supplier: Callable[[], object],
            *,
            on_event: Callable[[str, dict[str, object]], None] | None = None,
        ) -> None:
            del config, frame_supplier, on_event
            self.running = True
            self.stopped = False
            self.instances.append(self)

        def start(self) -> FakePublisher:
            return self

        def stop(self, timeout: float | None = None) -> None:
            del timeout
            self.stopped = True
            self.running = False

    class BrokenSource:
        def latest(self) -> VideoFrame | None:
            raise RuntimeError("video snapshot unavailable")

    monkeypatch.setattr(publisher, "QgcVideoPublisher", FakePublisher)
    config = replace(Config(), qgc_video=replace(QgcVideoConfig(), enabled=True))
    runtime = QgcVideoRuntime(
        config,
        BrokenSource(),
        TelemetryBroker(),
        camera=None,
        worker=None,
    )

    with pytest.raises(ExceptionGroup, match="收尾"):
        runtime.stop()
    assert FakePublisher.instances[0].stopped


def test_runtime_uses_real_display_state_without_consuming_results_or_stale_pose(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import numpy as np

    from airdrop.georef.camera import CameraModel
    from airdrop.perception.models import Detection, PixelBox
    from airdrop.perception.pipeline import PerceptionStats
    from airdrop.qgcvideo import publisher
    from airdrop.qgcvideo.models import DisplaySnapshot
    from airdrop.telemetry.models import TelemetrySnapshot

    class FakePublisher:
        instance: FakePublisher | None = None

        def __init__(
            self,
            config: object,
            frame_supplier: Callable[[], DisplaySnapshot | None],
            *,
            on_event: Callable[[str, dict[str, object]], None] | None = None,
        ) -> None:
            del config, on_event
            self.supplier = frame_supplier
            self.running = False
            FakePublisher.instance = self

        def start(self) -> FakePublisher:
            self.running = True
            return self

        def stop(self, timeout: float | None = None) -> None:
            del timeout
            self.running = False

    class FakeWorker:
        def __init__(self) -> None:
            self.frame_observers: list[Callable[[object], None]] = []
            self.result_observers: list[Callable[[object], None]] = []
            self.stats = PerceptionStats(frames=1, submitted=1)
            self.processing = False
            self.queued_results = 1

        def add_frame_observer(self, observer: Callable[[object], None]) -> None:
            self.frame_observers.append(observer)

        def add_result_observer(self, observer: Callable[[object], None]) -> None:
            self.result_observers.append(observer)

        def remove_frame_observer(self, observer: Callable[[object], None]) -> None:
            self.frame_observers.remove(observer)

        def remove_result_observer(self, observer: Callable[[object], None]) -> None:
            self.result_observers.remove(observer)

    class Source:
        def latest(self) -> VideoFrame:
            return VideoFrame(90, np.zeros((720, 1280, 3), np.uint8), 100.0, 0.0)

    monkeypatch.setattr(publisher, "QgcVideoPublisher", FakePublisher)
    config = replace(Config(), qgc_video=replace(QgcVideoConfig(), enabled=True))
    broker = TelemetryBroker()
    broker.publish(TelemetrySnapshot(timestamp=0.0, north_m=10, down_m=-100))
    worker = FakeWorker()
    camera = CameraModel(
        camera_matrix=np.array([[800, 0, 640], [0, 800, 360], [0, 0, 1]], dtype=float),
        image_size=(1280, 720),
    )
    runtime = QgcVideoRuntime(
        config,
        Source(),
        broker,
        camera,
        worker,  # type: ignore[arg-type]
        replay_mode=True,
    )
    runtime.state.on_event("state", {"to_state": "HOLD_LIVE"})
    telemetry = TelemetrySnapshot(
        north_m=0,
        east_m=0,
        down_m=-100,
        roll_deg=0,
        pitch_deg=0,
        yaw_deg=0,
        origin_latitude_deg=31,
        origin_longitude_deg=121,
        origin_altitude_m=10,
    )
    box = PixelBox(639, 359, 641, 361)
    detection = Detection(1, 1.0, box.center, box, 0.9, telemetry, code=7)
    worker.result_observers[0](detection)
    runtime.state.on_event("drop_released", {"timestamp": 12.5, "simulated": True})

    assert FakePublisher.instance is not None
    snapshot = FakePublisher.instance.supplier()
    assert snapshot is not None

    assert snapshot.frame is not None and snapshot.frame.pose is None
    assert snapshot.projected_observations
    assert snapshot.projected_observations[0].pixel is None
    assert snapshot.released and snapshot.release_timestamp == 12.5
    assert snapshot.release_simulated
    assert snapshot.stage == "HOLD_LIVE"
    assert snapshot.progress.queued_results == 1
    assert snapshot.progress.telemetry_age_s is None
    assert snapshot.capture_delay_s is None
    assert snapshot.observed_at == 100.0
    assert snapshot.replay_mode
    assert worker.queued_results == 1
