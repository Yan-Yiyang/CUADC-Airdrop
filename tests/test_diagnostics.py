"""通用性能诊断工具的轻量回归。"""

from __future__ import annotations

import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from airdrop import run
from tools import qgc_video_benchmark
from tools.benchmark_perception import _sample
from tools.performance import percentile_summary, write_json_new


def test_percentile_summary_handles_empty_and_interpolates() -> None:
    assert percentile_summary([])["count"] == 0
    summary = percentile_summary([1.0, 2.0, 3.0, 4.0, 5.0])
    assert summary["mean_ms"] == 3.0
    assert summary["p95_ms"] == 4.8


def test_json_writer_refuses_overwrite(tmp_path: Path) -> None:
    target = tmp_path / "result.json"
    write_json_new(target, {"value": 1})
    with pytest.raises(FileExistsError):
        write_json_new(target, {"value": 2})
    assert '"value": 1' in target.read_text(encoding="utf-8")


def test_uniform_sample_covers_edges_without_duplicates(tmp_path: Path) -> None:
    paths = [tmp_path / f"{index}.jpg" for index in range(10)]
    assert _sample(paths, 4) == [paths[0], paths[3], paths[6], paths[9]]
    assert _sample(paths, 1) == [paths[0]]
    with pytest.raises(ValueError):
        _sample(paths, 0)


def test_benchmark_cli_arguments_and_lazy_heavy_imports() -> None:
    values = run.resolve_kwargs("qgc-video-benchmark", ["--out", "rtp.json", "--seconds", "2"])
    assert values["output_path"] == Path("rtp.json")
    assert values["seconds"] == 2.0
    perception = run.resolve_kwargs(
        "benchmark-perception",
        ["--images-dir", "frames", "--out", "bench.json", "--threads", "2"],
    )
    assert perception["images_dir"] == Path("frames")
    assert perception["threads"] == 2
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from airdrop import run; run.resolve_kwargs('benchmark-perception', "
            "['--images-dir', 'frames', '--out', 'out.json']); "
            "assert not {'torch', 'cv2', 'ultralytics'} & set(sys.modules)",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_rtp_receiver_is_closed_when_publisher_stop_times_out(monkeypatch) -> None:
    import airdrop.qgcvideo.publisher as publisher_module

    class FakeSocket:
        closed = False

        def setsockopt(self, *args) -> None:
            del args

        def bind(self, address) -> None:
            del address

        def settimeout(self, timeout) -> None:
            del timeout

        def getsockname(self):
            return ("127.0.0.1", 45678)

        def recv(self, size):
            del size
            time.sleep(0.005)
            raise TimeoutError

        def close(self) -> None:
            self.closed = True

    receiver = FakeSocket()
    monkeypatch.setattr(qgc_video_benchmark.socket, "socket", lambda *args: receiver)
    created_threads: list[threading.Thread] = []
    real_thread = threading.Thread

    def make_thread(*args, **kwargs):
        result = real_thread(*args, **kwargs)
        created_threads.append(result)
        return result

    monkeypatch.setattr(qgc_video_benchmark.threading, "Thread", make_thread)

    class FakePublisher:
        def __init__(self, *args, **kwargs) -> None:
            del args, kwargs
            self.running = True
            self.stats = SimpleNamespace(last_error=None)

        def start(self):
            return self

        def stop(self):
            raise TimeoutError("publisher shutdown timed out")

    monkeypatch.setattr(publisher_module, "QgcVideoPublisher", FakePublisher)
    with pytest.raises(TimeoutError, match="publisher shutdown timed out"):
        qgc_video_benchmark._measure(seconds=0.001, fps=12.0, compose_delay=0.0)
    assert receiver.closed
    assert len(created_threads) == 1
    assert not created_threads[0].is_alive()


def test_rtp_receiver_is_closed_when_socket_setup_fails(monkeypatch) -> None:
    class FakeSocket:
        closed = False

        def setsockopt(self, *args) -> None:
            del args
            raise OSError("socket setup failed")

        def close(self) -> None:
            self.closed = True

    receiver = FakeSocket()
    monkeypatch.setattr(qgc_video_benchmark.socket, "socket", lambda *args: receiver)
    with pytest.raises(OSError, match="socket setup failed"):
        qgc_video_benchmark._measure(seconds=0.001, fps=12.0, compose_delay=0.0)
    assert receiver.closed
