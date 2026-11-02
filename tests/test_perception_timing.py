"""分阶段性能记录不得影响业务，支持多进程独立关联。"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def _run(code: str, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-B", "-c", code],
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=True,
    )


def test_trace_disabled_does_not_create_file(tmp_path: Path) -> None:
    env = dict(os.environ)
    env.pop("AIRDROP_PERF_TRACE_DIR", None)
    _run("from airdrop.perception.timing import emit, flush; emit('disabled'); flush()", env)
    assert list(tmp_path.iterdir()) == []


def test_trace_records_process_context_and_preserves_call_result(tmp_path: Path) -> None:
    env = {**os.environ, "AIRDROP_PERF_TRACE_DIR": str(tmp_path)}
    script = """
from airdrop.perception.timing import TimedCallable, context, emit, flush
context(request_id=12, frame_index=34, worker_id=1)
assert TimedCallable(lambda value: value + 1, 'call')(4) == 5
emit('finish', end_ns=2)
flush()
"""
    _run(script, env)
    _run(script, env)
    files = list(tmp_path.glob("perf-*.jsonl"))
    assert len(files) == 2
    for file in files:
        rows = [json.loads(line) for line in file.read_text(encoding="utf-8").splitlines()]
        assert [row["event"] for row in rows] == ["call", "finish"]
        assert all(row["request_id"] == 12 and row["frame_index"] == 34 for row in rows)
        assert rows[0]["start_ns"] <= rows[0]["end_ns"]


def test_trace_filesystem_failure_does_not_change_result(tmp_path: Path) -> None:
    blocked = tmp_path / "file"
    blocked.write_text("x", encoding="utf-8")
    env = {**os.environ, "AIRDROP_PERF_TRACE_DIR": str(blocked)}
    _run(
        "from airdrop.perception.timing import TimedCallable, flush; "
        "assert TimedCallable(lambda: 42, 'call')() == 42; flush()",
        env,
    )


def test_trace_class_wrapper_preserves_identity_attributes_and_exceptions(tmp_path: Path) -> None:
    env = {**os.environ, "AIRDROP_PERF_TRACE_DIR": str(tmp_path)}
    script = """
from airdrop.perception.timing import context, flush, instrument_call_class
class Fake:
    def __init__(self): self.value = 1
    def __call__(self, x):
        if x == 0: raise ValueError('expected')
        return self.value + x
instance = Fake()
original_id = id(instance)
context(request_id=7, frame_index=9, worker_id=0)
instrument_call_class(instance, 'det')
instrument_call_class(instance, 'det')
instance.value = 3
assert id(instance) == original_id and isinstance(instance, Fake)
assert instance(2) == 5
try: instance(0)
except ValueError as exc: assert str(exc) == 'expected'
else: raise AssertionError('expected ValueError')
flush()
"""
    _run(script, env)
    file = next(tmp_path.glob("perf-*.jsonl"))
    rows = [json.loads(line) for line in file.read_text(encoding="utf-8").splitlines()]
    assert [row["event"] for row in rows] == ["det", "det"]
    assert all(row["request_id"] == 7 and row["frame_index"] == 9 for row in rows)
