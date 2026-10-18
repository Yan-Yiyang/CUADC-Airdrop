from __future__ import annotations

import threading
import time
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from airdrop.config import QgcVideoConfig
from airdrop.qgcvideo.models import DisplaySnapshot
from airdrop.qgcvideo.publisher import QgcVideoPublisher


class _Pipe:
    def __init__(self, *, block: bool = False, short: bool = False, broken: bool = False) -> None:
        self.entered = threading.Event()
        self.released = threading.Event()
        self.block = block
        self.short = short
        self.broken = broken
        self.total = 0

    def write(self, data: memoryview) -> int:
        self.entered.set()
        if self.broken:
            self.broken = False
            raise BrokenPipeError("pipe closed")
        if self.block:
            self.released.wait()
        count = max(1, len(data) // 4) if self.short else len(data)
        self.total += count
        return count

    def flush(self) -> None:
        return


class _Process:
    def __init__(self, pipe: _Pipe) -> None:
        self.stdin = pipe
        self._ended = threading.Event()
        self.terminated = threading.Event()

    def poll(self) -> int | None:
        return 0 if self._ended.is_set() else None

    def terminate(self) -> None:
        self.terminated.set()
        self._ended.set()
        self.stdin.released.set()

    def kill(self) -> None:
        self.terminate()

    def wait(self, timeout: float | None = None) -> int:
        self._ended.wait(timeout)
        return 0


class _ExitOnFlushPipe(_Pipe):
    def __init__(self) -> None:
        super().__init__()
        self.exit_process: _Process | None = None

    def flush(self) -> None:
        if self.exit_process is not None:
            self.exit_process._ended.set()


class _Composer:
    def compose(self, snapshot: DisplaySnapshot) -> np.ndarray:
        del snapshot
        return np.zeros((990, 1760, 3), dtype=np.uint8)


def _snapshot() -> DisplaySnapshot:
    return cast(DisplaySnapshot, object())


def _publisher(factory: Any, **overrides: Any) -> QgcVideoPublisher:
    values: dict[str, Any] = {
        "enabled": True,
        "fps": 20.0,
        "ffmpeg_executable": "test-ffmpeg",
        "restart_delay_s": 0.01,
        "write_timeout_s": 0.3,
        "shutdown_timeout_s": 1.5,
        "max_restarts": 1,
        **overrides,
    }
    settings = QgcVideoConfig(**values)
    return QgcVideoPublisher(settings, _snapshot, _Composer(), subprocess_factory=factory)


def test_partial_pipe_writes_are_completed_before_frame_count_advances() -> None:
    pipe = _Pipe(short=True)
    process = _Process(pipe)
    publisher = _publisher(lambda *_args, **_kwargs: process)

    publisher.start()
    assert pipe.entered.wait(2.0)
    deadline = time.monotonic() + 2.0
    while publisher.stats.encoded == 0 and time.monotonic() < deadline:
        time.sleep(0.01)
    publisher.stop()

    assert publisher.stats.encoded >= 1
    assert pipe.total >= 1760 * 990 * 3


def test_blocked_pipe_does_not_backpressure_supplier_and_stop_unblocks_it() -> None:
    pipe = _Pipe(block=True)
    process = _Process(pipe)
    supplied = 0
    supply_lock = threading.Lock()

    def supplier() -> DisplaySnapshot:
        nonlocal supplied
        with supply_lock:
            supplied += 1
        return _snapshot()

    publisher = QgcVideoPublisher(
        QgcVideoConfig(
            enabled=True,
            fps=20.0,
            ffmpeg_executable="test-ffmpeg",
            shutdown_timeout_s=1.5,
            max_restarts=0,
        ),
        supplier,
        _Composer(),
        subprocess_factory=lambda *_args, **_kwargs: process,
    )
    publisher.start()
    assert pipe.entered.wait(2.0)
    time.sleep(0.2)
    before_stop = supplied
    publisher.stop(timeout=1.5)

    assert before_stop >= 3
    assert publisher.stats.dropped >= 1
    assert process.terminated.is_set()
    assert not publisher.running


def test_write_watchdog_terminates_a_stalled_encoder() -> None:
    pipe = _Pipe(block=True)
    process = _Process(pipe)
    publisher = _publisher(
        lambda *_args, **_kwargs: process,
        write_timeout_s=0.12,
        shutdown_timeout_s=1.0,
        max_restarts=0,
    )

    publisher.start()
    assert pipe.entered.wait(2.0)
    assert process.terminated.wait(1.0)
    publisher.stop(timeout=1.0)

    assert publisher.stats.last_error is not None
    assert "超时" in publisher.stats.last_error


def test_broken_pipe_restarts_within_the_configured_bound() -> None:
    broken = _Process(_Pipe(broken=True))
    healthy_pipe = _Pipe()
    healthy = _Process(healthy_pipe)
    made: list[_Process] = []

    def factory(*_args: object, **_kwargs: object) -> _Process:
        process = broken if not made else healthy
        made.append(process)
        return process

    publisher = _publisher(factory, restart_delay_s=0.01, max_restarts=1)
    publisher.start()
    assert healthy_pipe.entered.wait(3.0)
    deadline = time.monotonic() + 2.0
    while publisher.stats.encoded == 0 and time.monotonic() < deadline:
        time.sleep(0.01)
    publisher.stop()

    assert publisher.stats.failures >= 1
    assert publisher.stats.restarts == 1
    assert publisher.stats.encoded >= 1


def test_exhausted_restarts_clears_running_and_stops_render_thread() -> None:
    pipe = _ExitOnFlushPipe()
    process = _Process(pipe)
    pipe.exit_process = process
    publisher = _publisher(lambda *_args, **_kwargs: process, max_restarts=0)
    publisher.start()

    deadline = time.monotonic() + 2.0
    while publisher.running and time.monotonic() < deadline:
        time.sleep(0.01)
    publisher.stop(timeout=1.0)

    assert not publisher.running
    assert publisher._producer is not None and not publisher._producer.is_alive()
    assert publisher.stats.last_error == "RuntimeError: FFmpeg 重启次数已耗尽"


def test_second_stop_continues_cleanup_after_first_stop_timeout() -> None:
    pipe = _Pipe(block=True)
    process = _Process(pipe)

    class _DelayedCleanupPublisher(QgcVideoPublisher):
        cleanup_calls = 0

        def _terminate_process(self, selected: Any | None = None) -> None:
            self.cleanup_calls += 1
            if self.cleanup_calls == 1:
                return
            super()._terminate_process(selected)

    publisher = _DelayedCleanupPublisher(
        QgcVideoConfig(
            enabled=True,
            fps=20.0,
            ffmpeg_executable="test-ffmpeg",
            write_timeout_s=10.0,
            shutdown_timeout_s=1.0,
        ),
        _snapshot,
        _Composer(),
        subprocess_factory=lambda *_args, **_kwargs: process,
    )
    publisher.start()
    assert pipe.entered.wait(2.0)
    try:
        publisher.stop(timeout=0.0)
    except TimeoutError:
        pass
    else:
        raise AssertionError("首次零时限停止应报告未收尾线程")
    with pytest.raises(RuntimeError, match="尚未退出"):
        publisher.start()

    publisher.stop(timeout=1.0)

    assert not publisher.running
    assert not process.stdin.entered.is_set() or process.terminated.is_set()


def test_process_created_after_stop_is_terminated_before_start_returns() -> None:
    entered = threading.Event()
    release = threading.Event()
    made: list[_Process] = []

    def factory(*_args: object, **_kwargs: object) -> _Process:
        entered.set()
        release.wait()
        process = _Process(_Pipe())
        made.append(process)
        return process

    publisher = _publisher(factory, shutdown_timeout_s=1.0)
    publisher.start()
    assert entered.wait(2.0)
    try:
        publisher.stop(timeout=0.0)
    except TimeoutError:
        pass
    else:
        raise AssertionError("进程工厂阻塞期间 stop 应报告线程仍在启动")
    release.set()
    publisher.stop(timeout=1.5)

    assert len(made) == 1
    assert made[0].terminated.is_set()
    assert not publisher.running


def test_empty_supplier_stays_running_after_stop_and_restart() -> None:
    processes: list[_Process] = []

    def factory(*_args: object, **_kwargs: object) -> _Process:
        process = _Process(_Pipe())
        processes.append(process)
        return process

    publisher = QgcVideoPublisher(
        QgcVideoConfig(
            enabled=True,
            fps=20.0,
            ffmpeg_executable="test-ffmpeg",
            shutdown_timeout_s=1.0,
            max_restarts=0,
        ),
        lambda: None,
        _Composer(),
        subprocess_factory=factory,
    )
    publisher.start()
    publisher.stop(timeout=1.0)
    publisher.stop(timeout=1.0)
    publisher.start()
    time.sleep(0.15)

    assert publisher.running
    publisher.stop(timeout=1.0)


def test_watchdog_runs_even_when_supplier_keeps_raising() -> None:
    pipe = _Pipe(block=True)
    process = _Process(pipe)
    supplied = 0

    def supplier() -> DisplaySnapshot:
        nonlocal supplied
        supplied += 1
        if supplied == 1:
            return _snapshot()
        raise RuntimeError("synthetic supplier error")

    publisher = QgcVideoPublisher(
        QgcVideoConfig(
            enabled=True,
            fps=30.0,
            ffmpeg_executable="test-ffmpeg",
            write_timeout_s=0.12,
            shutdown_timeout_s=1.0,
            max_restarts=0,
        ),
        supplier,
        _Composer(),
        subprocess_factory=lambda *_args, **_kwargs: process,
    )
    publisher.start()
    assert pipe.entered.wait(2.0)
    assert process.terminated.wait(1.0)
    publisher.stop(timeout=1.0)

    assert "超时" in (publisher.stats.last_error or "")


def test_replay_latency_uses_local_submit_time_not_source_timestamp() -> None:
    frame = SimpleNamespace(capture_timestamp=time.time() - 1800.0)
    replay_snapshot = cast(DisplaySnapshot, SimpleNamespace(frame=frame, replay_mode=True))
    pipe = _Pipe()
    process = _Process(pipe)
    publisher = QgcVideoPublisher(
        QgcVideoConfig(
            enabled=True,
            fps=20.0,
            ffmpeg_executable="test-ffmpeg",
            shutdown_timeout_s=1.0,
        ),
        lambda: replay_snapshot,
        _Composer(),
        subprocess_factory=lambda *_args, **_kwargs: process,
    )
    publisher.start()
    assert pipe.entered.wait(2.0)
    deadline = time.monotonic() + 1.0
    while publisher.stats.encoded == 0 and time.monotonic() < deadline:
        time.sleep(0.01)
    publisher.stop(timeout=1.0)

    assert publisher.stats.encoded > 0
    assert publisher.stats.last_latency_s is not None
    assert publisher.stats.last_latency_s < 1.0
