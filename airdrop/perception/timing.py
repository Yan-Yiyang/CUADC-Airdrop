"""按需记录性能诊断；关闭时不创建文件，记录失败不影响业务。"""

from __future__ import annotations

import atexit
import contextlib
import functools
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

TRACE_DIR = os.environ.get("AIRDROP_PERF_TRACE_DIR", "").strip()
ENABLED = bool(TRACE_DIR)
_local = threading.local()
_lock = threading.Lock()
_stream: Any = None
_failed = False
_count = 0


def now_ns() -> int:
    """返回跨进程可比较的单调时刻，单位纳秒。"""
    return time.perf_counter_ns()


def context(**values: Any) -> None:
    """设置当前线程的帧和请求关联信息。"""
    if ENABLED:
        _local.values = values


def emit(event: str, **values: Any) -> None:
    """缓冲写入进程独立 JSONL；失败仅禁用本进程的诊断。"""
    global _stream, _failed, _count  # noqa: PLW0603
    if not ENABLED or _failed:
        return
    try:
        row = {"event": event, "pid": os.getpid(), "t_ns": now_ns()}
        row.update(getattr(_local, "values", {}))
        row.update(values)
        line = json.dumps(row, ensure_ascii=False, default=str) + "\n"
        with _lock:
            if _stream is None:
                directory = Path(TRACE_DIR)
                directory.mkdir(parents=True, exist_ok=True)
                _stream = (directory / f"perf-{os.getpid()}.jsonl").open(
                    "a", encoding="utf-8", buffering=65536
                )
            _stream.write(line)
            _count += 1
            if _count % 32 == 0:
                _stream.flush()
    except Exception as exc:  # noqa: BLE001 - 诊断绝不改变任务行为
        _failed = True
        with contextlib.suppress(Exception):
            logging.getLogger(__name__).warning("性能诊断写入失败，已停止本进程记录：%s", exc)


def flush() -> None:
    """写出缓冲；诊断失败不会传播到生产路径。"""
    global _failed  # noqa: PLW0603
    if not ENABLED or _failed:
        return
    try:
        with _lock:
            if _stream is not None:
                _stream.flush()
    except Exception as exc:  # noqa: BLE001
        _failed = True
        with contextlib.suppress(Exception):
            logging.getLogger(__name__).warning("性能诊断刷新失败，已停止本进程记录：%s", exc)


class TimedCallable:
    """为第三方可调用对象旁路计时，透传其属性和异常。"""

    def __init__(self, inner: Any, label: str):
        self.inner = inner
        self.label = label

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        start = now_ns()
        try:
            return self.inner(*args, **kwargs)
        finally:
            emit(self.label, start_ns=start, end_ns=now_ns())


def instrument_call_class(instance: Any, label: str) -> None:
    """装饰对象类型的调用入口，保留实例身份和属性赋值语义。"""
    cls = type(instance)
    original = cls.__call__
    if getattr(original, "_airdrop_timing_wrapped", False):
        return

    @functools.wraps(original)
    def measured(self: Any, *args: Any, **kwargs: Any) -> Any:
        start = now_ns()
        try:
            return original(self, *args, **kwargs)
        finally:
            emit(label, start_ns=start, end_ns=now_ns())

    measured._airdrop_timing_wrapped = True  # type: ignore[attr-defined]  # noqa: SLF001
    cls.__call__ = measured


if ENABLED:
    atexit.register(flush)
