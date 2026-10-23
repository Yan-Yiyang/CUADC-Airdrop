"""被动任务入口的离线生命周期回归：真实协调器、对齐、缓冲与文件写入。"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

import airdrop
from airdrop import Config, FlightRecorder, TelemetryBroker, TelemetrySnapshot
from airdrop.run import resolve_kwargs
from airdrop.runtime.mission import main as run_mission
from airdrop.video.source import VideoConfig, VideoFrame, VideoStats


@pytest.mark.parametrize("ending", ["normal", "interrupt", "not_ready", "empty"])
def test_capture_entry_records_and_closes_resources(
    ending: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """只替换外部输入；写盘完成用事件确认，避免用睡眠猜测线程进度。"""
    frame_written = threading.Event()
    calls: list[str] = []
    recorders: list[FlightRecorder] = []
    brokers: list[TelemetryBroker] = []
    sources = []
    root_handlers = logging.getLogger().handlers.copy()
    write_frame = FlightRecorder._write_one_frame

    def observe_write(self, frame, frames_dir, index_file) -> None:
        write_frame(self, frame, frames_dir, index_file)
        frame_written.set()

    def make_recorder(config: Config) -> FlightRecorder:
        recorder = FlightRecorder(config, base_dir=tmp_path)
        recorders.append(recorder)
        return recorder

    class TelemetryInput:
        @classmethod
        def from_config(cls, config, *, broker: TelemetryBroker):
            brokers.append(broker)
            return cls()

        def connect(self) -> None:
            calls.append("connect")
            base = time.time()
            for index in range(2):
                stamp = base + index * 0.1
                brokers[0]._update_from_snapshot(
                    TelemetrySnapshot(
                        timestamp=stamp,
                        latitude_deg=0.0,
                        longitude_deg=0.0,
                        absolute_altitude_m=10.0,
                        relative_altitude_m=10.0,
                        north_m=0.0,
                        east_m=0.0,
                        down_m=-10.0,
                        vx_m_s=0.0,
                        vy_m_s=0.0,
                        vz_m_s=0.0,
                        quaternion_w=1.0,
                        quaternion_x=0.0,
                        quaternion_y=0.0,
                        quaternion_z=0.0,
                        quaternion_timestamp_us=1_000_000 + index * 100_000,
                        quaternion_received_at=stamp,
                        position_timestamp_us=1_000_000 + index * 100_000,
                        position_velocity_received_at=stamp,
                    )
                )

        def stop(self, timeout: float = 5.0) -> None:
            del timeout
            calls.append("telemetry_stop")

    class VideoInput:
        def __init__(self, config: VideoConfig) -> None:
            self.config = config
            self.stats = VideoStats()
            self.running = False
            self.sinks: list[Callable[[VideoFrame], None]] = []
            sources.append(self)

        def add_sink(self, sink: Callable[[VideoFrame], None]) -> None:
            self.sinks.append(sink)

        def start(self) -> None:
            calls.append("video_start")
            self.running = True
            self.stats.state = "streaming"
            if ending in {"normal", "interrupt"}:
                span = brokers[0].history_span()
                assert span is not None
                frame = VideoFrame(
                    index=1,
                    image=np.zeros((16, 16, 3), dtype=np.uint8),
                    timestamp=span[0] + self.config.telemetry_lag,
                    lag=self.config.telemetry_lag,
                )
                for sink in self.sinks:
                    sink(frame)
                self.stats.frames = 1
            elif ending == "not_ready":
                self.stats.last_error = "synthetic video unavailable"

        def wait_ready(self, timeout: float) -> bool:
            if ending in {"normal", "interrupt"}:
                assert frame_written.wait(5.0), "对齐后的合成帧未写入文件"
            if ending == "interrupt":
                raise KeyboardInterrupt
            return ending != "not_ready"

        def stop(self, timeout: float = 5.0) -> None:
            del timeout
            calls.append("video_stop")
            self.running = False

    monkeypatch.setattr(FlightRecorder, "_write_one_frame", observe_write)
    monkeypatch.setattr(airdrop, "FlightRecorder", make_recorder)
    monkeypatch.setattr(airdrop, "MavsdkThread", TelemetryInput)
    monkeypatch.setattr(airdrop, "Hm30VideoSource", VideoInput)
    caplog.set_level(logging.INFO)

    kwargs = resolve_kwargs(
        "mission",
        [
            "--mode",
            "real",
            "--no-mission-control",
            "--capture",
            "--no-recognize",
            "--record",
            "--max-seconds",
            "0.000001",
        ],
    )
    code = run_mission(**kwargs)

    assert code == (130 if ending == "interrupt" else 0 if ending == "normal" else 1)
    assert calls == ["connect", "video_start", "video_stop", "telemetry_stop"]
    assert not sources[0].running
    recorder = recorders[0]
    assert not recorder.is_running
    assert logging.getLogger().handlers == root_handlers
    flight_dir = recorder.flight_dir
    assert flight_dir is not None and flight_dir.is_dir()
    assert str(flight_dir) in caplog.text
    snapshot = json.loads((flight_dir / "config_snapshot.json").read_text(encoding="utf-8"))
    assert snapshot
    rows = (flight_dir / "frames_index.jsonl").read_text(encoding="utf-8").splitlines()
    if ending in {"normal", "interrupt"}:
        assert len(rows) == 1
        row = json.loads(rows[0])
        assert (flight_dir / "frames" / row["filename"]).is_file()
        assert row["received_timestamp"] - row["capture_timestamp"] == pytest.approx(
            row["lag"], abs=1e-6
        )
    else:
        assert not rows
