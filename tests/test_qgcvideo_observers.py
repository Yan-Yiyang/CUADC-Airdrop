from __future__ import annotations

import threading
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from airdrop.perception.detector import DetectionBatch
from airdrop.perception.models import Detection, PixelBox, ProcessedFrame
from airdrop.perception.pipeline import PerceptionConfigLike, PerceptionWorker
from airdrop.telemetry.models import TelemetrySnapshot


class _Buffer:
    def latest_index(self) -> int:
        return 10


class _Detector:
    config = SimpleNamespace(camera_matrix=None)

    def __init__(self, detections: tuple[Detection, ...]) -> None:
        self._detections = detections

    def detect(self, image: np.ndarray, **_: object) -> DetectionBatch:
        return DetectionBatch(image=image, detections=self._detections)

    def crop(self, *_: object) -> np.ndarray:
        return np.ones((2, 2, 3), np.uint8)


def _detection(index: int = 3) -> Detection:
    box = PixelBox(1, 2, 4, 6)
    return Detection(
        frame_index=index,
        capture_timestamp=3.0,
        pixel=box.center,
        box=box,
        confidence=0.8,
        telemetry=TelemetrySnapshot(north_m=1.0),
        code=2,
        extra={"mutable": [1]},
        mode="cls12",
    )


def _buffered_frame(index: int = 3, image: np.ndarray | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        index=index,
        image=np.zeros((4, 5, 3), np.uint8) if image is None else image,
        capture_timestamp=3.0,
        received_timestamp=2.9,
        snapshot=TelemetrySnapshot(
            north_m=1,
            east_m=2,
            down_m=-10,
            quaternion_w=1,
            quaternion_x=0,
            quaternion_y=0,
            quaternion_z=0,
            origin_latitude_deg=31,
            origin_longitude_deg=121,
            origin_altitude_m=10,
        ),
    )


def _worker(detections: tuple[Detection, ...] = ()) -> PerceptionWorker:
    return PerceptionWorker(
        PerceptionConfigLike(mode="cls12"),
        buffer=_Buffer(),  # type: ignore[arg-type]
        detector=_Detector(detections),  # type: ignore[arg-type]
    )


def test_frame_observer_gets_zero_detection_frame_and_cannot_mutate_image() -> None:
    worker = _worker()
    observed: list[ProcessedFrame] = []
    worker.add_frame_observer(observed.append)
    source = np.zeros((4, 5, 3), np.uint8)
    worker._process_frame(_buffered_frame(image=source))  # type: ignore[arg-type]

    assert len(observed) == 1
    frame = observed[0]
    assert frame.frame_index == 3
    assert frame.image_size == (5, 4)
    assert frame.detections == ()
    assert not frame.image.flags.writeable
    with pytest.raises(ValueError):
        frame.image.flags.writeable = True


def test_result_observers_are_additive_isolated_and_do_not_consume_queue() -> None:
    detection = _detection()
    worker = _worker((detection,))
    legacy: list[Detection] = []
    observed = []
    worker.set_result_callback(legacy.append)
    worker.add_result_observer(lambda _: (_ for _ in ()).throw(RuntimeError("observer")))
    worker.add_result_observer(observed.append)

    worker._process_frame(_buffered_frame())  # type: ignore[arg-type]

    assert legacy == [detection]
    assert len(observed) == 1
    assert observed[0].code == 2
    assert not hasattr(observed[0], "telemetry")
    assert worker.queued_results == 1
    queued = worker.poll_result()
    assert queued is not None and queued.as_dict() == detection.as_dict()


def test_observer_remove_is_idempotent() -> None:
    worker = _worker()
    received = []
    worker.add_frame_observer(received.append)
    worker.remove_frame_observer(received.append)
    worker.remove_frame_observer(received.append)
    worker._process_frame(_buffered_frame())  # type: ignore[arg-type]
    assert received == []


def test_blocked_display_preserves_fixed_perception_cutoff_and_full_recording(
    tmp_path: Path,
) -> None:
    """显示编码阻塞时，识别仍排空固定截止，记录仍保留后续飞行帧。"""
    from airdrop import (
        AlignedSample,
        AlignmentBuffer,
        Config,
        FlightRecorder,
        TelemetryBroker,
        VideoFrame,
    )
    from airdrop.georef.camera import CameraModel
    from airdrop.mission.targets import PerceptionTargetSource, TargetTracker
    from airdrop.qgcvideo.publisher import QgcVideoPublisher
    from airdrop.qgcvideo.state import DisplayState

    class Detector:
        config = SimpleNamespace(camera_matrix=None)

        def load(self):
            return self

        def detect(self, image, **kwargs):
            box = PixelBox(6, 6, 10, 10)
            return DetectionBatch(
                image=image,
                detections=(
                    Detection(
                        frame_index=kwargs["frame_index"],
                        capture_timestamp=kwargs["capture_timestamp"],
                        telemetry=kwargs["telemetry"],
                        pixel=box.center,
                        box=box,
                        confidence=0.9,
                        code=12,
                        mode="cls12",
                    ),
                ),
            )

    class BlockedEncoder:
        def __init__(self):
            self.stdin = self
            self.entered = threading.Event()
            self.closed = threading.Event()

        def write(self, data):
            self.entered.set()
            self.closed.wait(5)
            return len(data)

        def flush(self):
            pass

        def poll(self):
            return 0 if self.closed.is_set() else None

        def terminate(self):
            self.closed.set()

        kill = terminate

        def wait(self, timeout=None):
            self.closed.wait(timeout)
            return 0

    class Composer:
        def compose(self, snapshot):
            return np.zeros((990, 1760, 3), np.uint8)

    config = Config()
    config = replace(config, ground=replace(config.ground, ground_point_alt=10))
    camera = CameraModel(
        np.array([[80.0, 0.0, 8.0], [0.0, 80.0, 8.0], [0.0, 0.0, 1.0]]), image_size=(16, 16)
    )
    buffer = AlignmentBuffer(capacity=32, storage="raw")
    worker = PerceptionWorker(
        PerceptionConfigLike(mode="cls12"),
        buffer=buffer,
        detector=Detector(),  # type: ignore[arg-type]
    )
    state = DisplayState(config, camera)
    worker.add_frame_observer(state.on_frame)
    worker.add_result_observer(state.on_detection)
    recorder = FlightRecorder(config, base_dir=tmp_path)
    encoder = BlockedEncoder()
    publisher = QgcVideoPublisher(
        replace(config.qgc_video, fps=40.0, ffmpeg_executable="fake", write_timeout_s=3.0),
        state.snapshot,
        Composer(),
        subprocess_factory=lambda *args, **kwargs: encoder,
    )

    def append_frame(index: int):
        stamp = time.time()
        snapshot = TelemetrySnapshot(
            north_m=0.0,
            east_m=0.0,
            down_m=-10.0,
            roll_deg=0.0,
            pitch_deg=0.0,
            yaw_deg=0.0,
            origin_altitude_m=10.0,
        )
        buffer.put(
            AlignedSample(
                frame=VideoFrame(
                    index=index, image=np.zeros((16, 16, 3), np.uint8), timestamp=stamp, lag=0.0
                ),
                snapshot=snapshot,
                timestamp=stamp,
                lag=0.0,
                mode="interpolate",
            )
        )

    recorder.start(broker=TelemetryBroker(), buffer=buffer)
    try:
        for index in range(6):
            append_frame(index)
        cutoff = worker.finish_input()
        worker.start()
        publisher.start()
        assert encoder.entered.wait(2)
        state.on_event("state", {"to_state": "OVERFLY"})
        for index in range(6, 12):
            append_frame(index)
        assert worker.drain(timeout=3)
        assert worker.stats.frames == 6
        assert worker.queued_results == 6
        targets = PerceptionTargetSource(
            worker, TargetTracker(config, camera), recorder.detections.append
        )
        assert targets.pump() == 6
        assert worker.queued_results == 0
        deadline = time.monotonic() + 2
        while publisher.stats.dropped == 0 and time.monotonic() < deadline:
            time.sleep(0.01)
        assert publisher.stats.dropped > 0
        assert buffer.latest_index() > cutoff
    finally:
        worker.stop()
        publisher.stop()
        recorder.stop()
    stats = recorder.stats()
    assert stats is not None and stats.frames_written == 12
    assert recorder.flight_dir is not None
    assert (
        len((recorder.flight_dir / "detections.jsonl").read_text(encoding="utf-8").splitlines())
        == 6
    )
