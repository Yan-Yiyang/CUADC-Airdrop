"""任务上传、回读与启动的协作式传输事务。"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Protocol

from .items import MissionItem


class MissionTransferController(Protocol):
    """任务传输所需的最小控制器接口。"""

    def upload_mission(self, items: Sequence[MissionItem], /) -> int: ...

    def start_mission(self) -> None: ...


class MissionSubmitter(Protocol):
    """接受一个同步任务并返回其实际完成状态。"""

    def __call__[T](self, operation: Callable[[], T], /) -> Future[T]: ...


class MissionTransfer:
    """在单后台worker中串行上传回读、检查取消并启动任务。

    取消只禁止尚未取得启动阶段权的操作继续；不会撤销已送达飞控的上传。运行中的
    控制器RPC必须自行完成其T12收尾，之后worker才会执行恢复动作。
    """

    def __init__(
        self,
        controller: MissionTransferController,
        *,
        submitter: MissionSubmitter | None = None,
    ) -> None:
        self._controller = controller
        if submitter is None:
            self._executor: ThreadPoolExecutor | None = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="mission-transfer"
            )
            self._submitter: MissionSubmitter = self._executor.submit
        else:
            self._executor = None
            self._submitter = submitter
        self._lock = threading.Lock()
        self._cancelled = threading.Event()
        self._future: Future[int] | None = None
        self._recovery_future: Future[object] | None = None
        self._recovery_scheduled = threading.Event()
        self._recovery_scheduled.set()
        self._recovery_errors: list[BaseException] = []
        self._recovery: Callable[[], object] | None = None
        self._recovery_claimed = False
        self._operation_finished = False
        self._closed = False
        self.leg: str | None = None

    @property
    def active(self) -> bool:
        return any(
            future is not None and not future.done()
            for future in (self._future, self._recovery_future)
        )

    @property
    def future(self) -> Future[int] | None:
        return self._future

    def start(self, items: Sequence[MissionItem], leg: str) -> None:
        """提交一条上传事务；已有事务未完成或对象已关闭时拒绝复用。"""
        specs = tuple(items)
        with self._lock:
            if self._closed:
                raise RuntimeError("MissionTransfer 已关闭")
            if self.active:
                raise RuntimeError("上一条任务传输仍在进行")
            self._cancelled.clear()
            self._recovery = None
            self._recovery_claimed = False
            self._operation_finished = False
            self._recovery_future = None
            self.leg = leg
            future: Future[int] = Future()
            # 在调用提交器前登记Future，stop/close并发到来时也能看到在途事务。
            self._future = future

        def operation() -> int:
            try:
                count = self._controller.upload_mission(specs)
                with self._lock:
                    start_claimed = not self._cancelled.is_set()
                if start_claimed:
                    self._controller.start_mission()
            except BaseException as operation_error:
                try:
                    self._finish_operation()
                except BaseException as recovery_error:  # noqa: BLE001 - 保留取消类异常并组合
                    raise BaseExceptionGroup(
                        "任务传输与恢复动作均失败", [operation_error, recovery_error]
                    ) from operation_error
                raise
            else:
                self._finish_operation()
                return count

        try:
            submitted = self._submitter(operation)
        except BaseException as exc:
            # 提交器抛错表示操作未进入worker；记录完成状态，使abort恢复不再等待不存在的RPC。
            with self._lock:
                self._operation_finished = True
                recovery = self._recovery if self._cancelled.is_set() else None
            if recovery is not None:
                self.cancel(recovery)
            future.set_exception(exc)
            raise

        def copy_result(done: Future[int]) -> None:
            try:
                future.set_result(done.result())
            except BaseException as exc:  # noqa: BLE001 - 传播Future取消及后台异常
                future.set_exception(exc)

        submitted.add_done_callback(copy_result)

    def cancel(self, recovery: Callable[[], object]) -> None:
        """请求协作取消，传输收尾后由同一执行器执行一次恢复动作。"""
        operation: Callable[[], object] | None = None
        with self._lock:
            if self._recovery_claimed:
                return
            if self._closed and not (
                self._operation_finished and self._cancelled.is_set() and self._recovery is not None
            ):
                return
            self._cancelled.set()
            if self._recovery is None:
                self._recovery = recovery
            if not self._operation_finished:
                return
            self._recovery_claimed = True
            operation = self._recovery
            self._recovery = None
            self._recovery_scheduled.clear()
        if operation is not None:
            started = threading.Event()
            finished = threading.Event()
            claim_lock = threading.Lock()

            def run_once() -> object:
                with claim_lock:
                    if started.is_set():
                        return None
                    started.set()
                try:
                    try:
                        return operation()
                    except BaseException as recovery_error:  # noqa: BLE001 - close统一报告
                        self._remember_recovery_error(recovery_error)
                        return None
                finally:
                    finished.set()

            try:
                future = self._submitter(run_once)
            except BaseException as submit_error:  # noqa: BLE001 - submitter可能以取消异常拒绝
                self._remember_recovery_error(submit_error)
                run_once()
                # 若提交器启动后才报错，等待已启动的恢复动作真实完成。
                finished.wait()
            else:
                with self._lock:
                    self._recovery_future = future
            finally:
                self._recovery_scheduled.set()

    def _remember_recovery_error(self, error: BaseException) -> None:
        """保存恢复提交或同步fallback错误，供close统一报告。"""
        with self._lock:
            self._recovery_errors.append(error)

    def _finish_operation(self) -> None:
        with self._lock:
            self._operation_finished = True
            if not self._cancelled.is_set() or self._recovery_claimed:
                return
            self._recovery_claimed = True
            recovery = self._recovery
            self._recovery = None
        if recovery is not None:
            recovery()

    def poll(self) -> int | None:
        """非阻塞取回结果；尚未完成返回None，异常按原样抛出。"""
        future = self._future
        if future is None or not future.done():
            return None
        return future.result()

    def close(self, *, timeout: float | None = None) -> None:
        """等待实际worker及其中恢复动作收尾，再关闭内部执行器。"""
        with self._lock:
            self._closed = True
        deadline = None if timeout is None else time.monotonic() + timeout
        errors: list[BaseException] = []
        seen_recovery_errors = 0
        timed_out = False
        waited: set[int] = set()
        while True:
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            if not self._recovery_scheduled.wait(remaining):
                raise TimeoutError("任务恢复动作尚未提交")
            with self._lock:
                futures = tuple(f for f in (self._future, self._recovery_future) if f is not None)
                errors.extend(self._recovery_errors[seen_recovery_errors:])
                seen_recovery_errors = len(self._recovery_errors)
            pending = tuple(future for future in futures if id(future) not in waited)
            if not pending:
                break
            for future in pending:
                waited.add(id(future))
                remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
                try:
                    future.result(timeout=remaining)
                except TimeoutError as exc:
                    if future.done():
                        errors.append(exc)
                    else:
                        timed_out = True
                        break
                except BaseException as exc:  # noqa: BLE001 - Future 可保存取消类异常
                    errors.append(exc)
            if timed_out:
                break
        if self._executor is not None and not timed_out:
            self._executor.shutdown(wait=True)
        if timed_out:
            raise TimeoutError("任务传输或恢复动作尚未实际收尾")
        if len(errors) == 1:
            raise errors[0]
        if errors:
            raise BaseExceptionGroup("任务传输与恢复动作均失败", errors)
