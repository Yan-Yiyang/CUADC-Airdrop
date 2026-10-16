"""RPC 的同步等待、取消收尾与结果未知门控，不负责连接或遥测流。"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import math
import threading
import time
from collections.abc import Callable, Coroutine, Iterator
from typing import Any


class OperationTimeout(TimeoutError):
    """等待预算耗尽或 SDK 自身超时；不能据此判断远端没有执行。"""


def _sdk_timeout(error: BaseException) -> bool:
    """兼容 Python 超时与 MAVSDK 插件的 TIMEOUT 结果，不导入重依赖。"""
    result = getattr(getattr(error, "_result", None), "result", None)
    return isinstance(error, TimeoutError) or getattr(result, "name", None) == "TIMEOUT"


class _Operation:
    """只有包装协程自身能确认已收尾；桥接 Future 的状态不参与确认。"""

    def __init__(self) -> None:
        self.finished = threading.Event()
        self.result: Any = None
        self.error: BaseException | None = None
        self._state_lock = threading.Lock()
        self._cancel_requested = False
        self._task: asyncio.Task[Any] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    async def invoke(self, factory: Callable[[], Coroutine[Any, Any, Any]]) -> None:
        try:
            # 在首次执行时注册 Task，首次执行前直接 cancel 会跳过 finally。
            with self._state_lock:
                self._task = asyncio.current_task()
                self._loop = asyncio.get_running_loop()
                cancelled = self._cancel_requested
            if cancelled:
                raise asyncio.CancelledError
            self.result = await factory()
        except BaseException as exc:  # noqa: BLE001 — 将取消与异常传回同步调用方。
            self.error = exc
        finally:
            self.finished.set()

    def cancel(self) -> None:
        """仅发一次取消请求，避免再次 cancel 打断插件的异步 finally。"""
        with self._state_lock:
            if self._cancel_requested or self.finished.is_set():
                return
            self._cancel_requested = True
            task, loop = self._task, self._loop
        if task is not None and loop is not None:
            # 关闭中的循环可能拒绝请求；仍保留在途记录，不能伪造收尾完成。
            with contextlib.suppress(RuntimeError):
                loop.call_soon_threadsafe(self._cancel_task, task)

    def _cancel_task(self, task: asyncio.Task[Any]) -> None:
        if not self.finished.is_set() and not task.done():
            task.cancel()


class OperationGate:
    """同一线程对象共用门控；结果未知不会随重连或恢复指令自动清除。"""

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self._scope = threading.local()
        self._active: _Operation | None = None
        self._unknown = threading.Event()

    @property
    def state_unknown(self) -> bool:
        return self._unknown.is_set()

    def remaining_timeout(self, default: float) -> float:
        """将命令预算传给同步启动；普通 submit 仍使用原有默认值。"""
        deadline = getattr(self._scope, "deadline", None)
        if deadline is None:
            return default
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise OperationTimeout("等待预算已耗尽，本次调用未提交")
        return min(default, remaining)

    @contextlib.contextmanager
    def transaction(self, timeout: float, label: str) -> Iterator[float]:
        """成组 RPC 共用可重入锁与截止时间；异步收尾绝不获取此锁。"""
        timeout = float(timeout)
        if not math.isfinite(timeout) or not 0 <= timeout <= threading.TIMEOUT_MAX:
            raise ValueError("timeout 必须是有限、非负且可等待的秒数")
        previous = getattr(self._scope, "deadline", None)
        deadline = time.monotonic() + timeout
        if previous is not None:
            deadline = min(deadline, previous)
        if not self.lock.acquire(timeout=max(0.0, deadline - time.monotonic())):
            raise OperationTimeout(f"{label}：等待命令锁超时，本次调用未提交")
        try:
            self._scope.deadline = deadline
            yield deadline
        finally:
            self._scope.deadline = previous
            self.lock.release()

    def run(
        self,
        submit: Callable[[Coroutine[Any, Any, Any]], concurrent.futures.Future[Any]],
        factory: Callable[[], Coroutine[Any, Any, Any]],
        *,
        timeout: float,
        label: str,
        mutating: bool,
        recovery: bool = False,
    ) -> Any:
        with self.transaction(timeout, label) as deadline:
            if self._active is not None:
                if not self._active.finished.wait(max(0.0, deadline - time.monotonic())):
                    raise OperationTimeout(f"{label}：等待前序协程收尾超时，本次调用未提交")
                self._active = None
            if self.state_unknown and mutating and not recovery:
                raise RuntimeError("先前飞控命令结果未知，拒绝继续修改状态；请先核对远端状态")
            if time.monotonic() >= deadline:
                raise OperationTimeout(f"{label}：等待预算已耗尽，本次调用未提交")

            operation = _Operation()
            self._active = operation
            wrapped = operation.invoke(factory)
            try:
                submit(wrapped)
            except BaseException:
                # submit 失败表示尚未接受调度；factory 在 wrapper 内惰性创建。
                wrapped.close()
                self._active = None
                raise
            try:
                finished = operation.finished.wait(max(0.0, deadline - time.monotonic()))
            except BaseException:
                operation.cancel()
                if mutating:
                    self._unknown.set()
                raise
            if not finished:
                operation.cancel()
                if mutating:
                    self._unknown.set()
                meaning = "飞控执行结果未知" if mutating else "查询没有结果"
                raise OperationTimeout(
                    f"{label}：{timeout:g}s 内没有结果（等待超时）；{meaning}；"
                    "已请求本地取消，不代表远端已撤销"
                )
            self._active = None
            error = operation.error
            if error is None:
                return operation.result
            if _sdk_timeout(error):
                if mutating:
                    self._unknown.set()
                meaning = "飞控执行结果未知" if mutating else "查询没有结果"
                raise OperationTimeout(f"{label}：SDK 调用超时；{meaning}") from error
            if isinstance(error, asyncio.CancelledError):
                if mutating:
                    self._unknown.set()
                meaning = "飞控执行结果未知" if mutating else "查询未完成"
                raise RuntimeError(f"{label}：本地协程已取消；{meaning}") from error
            raise error


_CREATION_LOCK = threading.Lock()


def get_operation_gate(thread: Any) -> OperationGate:
    """兼容只实现 submit 的注入对象；新控制器复用同一对象上的门控。"""
    with _CREATION_LOCK:
        gate = getattr(thread, "_operation_gate", None)
        if gate is None:
            gate = OperationGate()
            thread._operation_gate = gate  # noqa: SLF001 — 同模块族共享私有门控。
        return gate
