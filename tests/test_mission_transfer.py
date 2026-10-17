from __future__ import annotations

import concurrent.futures
import threading
from collections.abc import Callable, Sequence
from typing import Any

import pytest

from airdrop.mission.items import MissionItem
from airdrop.mission.transfer import MissionTransfer


class Controller:
    def __init__(self) -> None:
        self.uploads = 0
        self.starts = 0
        self.recoveries = 0
        self.upload_error: BaseException | None = None
        self.start_entered: threading.Event | None = None
        self.release_start: threading.Event | None = None

    def upload_mission(self, items: Sequence[MissionItem], /) -> int:
        self.uploads += 1
        if self.upload_error is not None:
            raise self.upload_error
        return len(items)

    def start_mission(self) -> None:
        self.starts += 1
        if self.start_entered is not None:
            self.start_entered.set()
            assert self.release_start is not None
            self.release_start.wait()


def _submitter() -> tuple[
    Callable[[Callable[[], Any]], concurrent.futures.Future[Any]],
    list[tuple[Callable[[], Any], concurrent.futures.Future[Any]]],
]:
    jobs: list[tuple[Callable[[], Any], concurrent.futures.Future[Any]]] = []

    def submit(operation: Callable[[], Any], /) -> concurrent.futures.Future[Any]:
        future: concurrent.futures.Future[Any] = concurrent.futures.Future()
        jobs.append((operation, future))
        return future

    return submit, jobs


def _run_job[T](job: tuple[Callable[[], T], concurrent.futures.Future[T]]) -> None:
    operation, future = job
    try:
        future.set_result(operation())
    except BaseException as exc:  # noqa: BLE001 - 测试执行器转交取消类异常
        future.set_exception(exc)


def _items() -> tuple[MissionItem, ...]:
    return (MissionItem.waypoint(47.0, 8.0, 30.0),)


def test_cancel_before_activation_claim_skips_start_and_recovers() -> None:
    submit, jobs = _submitter()
    controller = Controller()
    transfer = MissionTransfer(controller, submitter=submit)
    transfer.start(_items(), "recon")
    transfer.cancel(lambda: setattr(controller, "recoveries", controller.recoveries + 1))

    _run_job(jobs.pop(0))

    assert controller.uploads == 1
    assert controller.starts == 0
    assert controller.recoveries == 1
    transfer.close()


def test_upload_error_during_cancellation_still_runs_recovery() -> None:
    submit, jobs = _submitter()
    controller = Controller()
    controller.upload_error = RuntimeError("上传失败")
    transfer = MissionTransfer(controller, submitter=submit)
    transfer.start(_items(), "recon")
    transfer.cancel(lambda: setattr(controller, "recoveries", controller.recoveries + 1))
    _run_job(jobs.pop(0))

    with pytest.raises(RuntimeError, match="上传失败"):
        transfer.close()
    assert controller.starts == 0
    assert controller.recoveries == 1


def test_cancel_after_activation_claim_recovers_after_start_finishes() -> None:
    submit, jobs = _submitter()
    controller = Controller()
    controller.start_entered = threading.Event()
    controller.release_start = threading.Event()
    transfer = MissionTransfer(controller, submitter=submit)
    transfer.start(_items(), "recon")
    worker = threading.Thread(target=_run_job, args=(jobs.pop(0),))
    worker.start()
    assert controller.start_entered.wait(1)

    transfer.cancel(lambda: setattr(controller, "recoveries", controller.recoveries + 1))
    assert controller.recoveries == 0
    controller.release_start.set()
    worker.join(timeout=1)

    assert not worker.is_alive()
    assert controller.starts == 1
    assert controller.recoveries == 1
    transfer.close()


def test_late_cancel_after_completed_transfer_queues_recovery_once() -> None:
    submit, jobs = _submitter()
    controller = Controller()
    transfer = MissionTransfer(controller, submitter=submit)
    transfer.start(_items(), "recon")
    _run_job(jobs.pop(0))
    assert transfer.poll() == 1

    transfer.cancel(lambda: setattr(controller, "recoveries", controller.recoveries + 1))
    assert len(jobs) == 1
    _run_job(jobs.pop(0))
    transfer.cancel(lambda: setattr(controller, "recoveries", controller.recoveries + 1))
    transfer.close()

    assert controller.starts == 1
    assert controller.recoveries == 1


def test_close_waits_for_queued_recovery_future() -> None:
    submit, jobs = _submitter()
    controller = Controller()
    transfer = MissionTransfer(controller, submitter=submit)
    transfer.start(_items(), "recon")
    _run_job(jobs.pop(0))
    transfer.cancel(lambda: setattr(controller, "recoveries", controller.recoveries + 1))
    recovery_job = jobs.pop(0)
    close_returned = threading.Event()
    closer = threading.Thread(target=lambda: (transfer.close(), close_returned.set()))
    closer.start()

    assert not close_returned.wait(0.02)
    _run_job(recovery_job)
    assert close_returned.wait(1)
    closer.join(timeout=1)
    assert not closer.is_alive()
    assert controller.recoveries == 1


def test_close_timeout_keeps_pending_transfer_for_a_later_drain() -> None:
    submit, jobs = _submitter()
    controller = Controller()
    transfer = MissionTransfer(controller, submitter=submit)
    transfer.start(_items(), "recon")

    with pytest.raises(TimeoutError, match="尚未实际收尾"):
        transfer.close(timeout=0.01)

    _run_job(jobs.pop(0))
    assert transfer.poll() == 1
    transfer.close(timeout=1)


def test_initial_submit_rejection_is_reported_after_abort_recovery_fallback() -> None:
    controller = Controller()
    submit_calls = 0

    def reject_submit(_operation: Callable[[], Any], /) -> concurrent.futures.Future[Any]:
        nonlocal submit_calls
        submit_calls += 1
        raise RuntimeError(f"提交被拒绝{submit_calls}")

    transfer = MissionTransfer(controller, submitter=reject_submit)
    with pytest.raises(RuntimeError, match="提交被拒绝1"):
        transfer.start(_items(), "recon")
    transfer.cancel(lambda: setattr(controller, "recoveries", controller.recoveries + 1))

    assert controller.uploads == 0
    assert controller.starts == 0
    assert controller.recoveries == 1
    with pytest.raises(BaseExceptionGroup) as caught:
        transfer.close()
    messages = [str(error) for error in caught.value.exceptions]
    assert any("提交被拒绝1" in message for message in messages)
    assert any("提交被拒绝2" in message for message in messages)


def test_recovery_submit_rejection_combines_fallback_failure_once() -> None:
    controller = Controller()
    submit_calls = 0

    def submit(operation: Callable[[], Any], /) -> concurrent.futures.Future[Any]:
        nonlocal submit_calls
        submit_calls += 1
        if submit_calls == 1:
            future: concurrent.futures.Future[Any] = concurrent.futures.Future()
            future.set_result(operation())
            return future
        raise RuntimeError("恢复提交被拒绝")

    transfer = MissionTransfer(controller, submitter=submit)
    transfer.start(_items(), "recon")
    assert transfer.poll() == 1

    def failed_recovery() -> None:
        controller.recoveries += 1
        raise ValueError("同步恢复失败")

    transfer.cancel(failed_recovery)
    with pytest.raises(BaseExceptionGroup) as caught:
        transfer.close()

    messages = [str(error) for error in caught.value.exceptions]
    assert any("恢复提交被拒绝" in message for message in messages)
    assert any("同步恢复失败" in message for message in messages)
    assert controller.recoveries == 1


def test_recovery_that_runs_before_submitter_error_is_not_repeated() -> None:
    controller = Controller()
    submit_calls = 0

    def submit(operation: Callable[[], Any], /) -> concurrent.futures.Future[Any]:
        nonlocal submit_calls
        submit_calls += 1
        future: concurrent.futures.Future[Any] = concurrent.futures.Future()
        if submit_calls == 1:
            future.set_result(operation())
            return future
        operation()
        raise RuntimeError("提交器迟到报错")

    transfer = MissionTransfer(controller, submitter=submit)
    transfer.start(_items(), "recon")
    assert transfer.poll() == 1
    transfer.cancel(lambda: setattr(controller, "recoveries", controller.recoveries + 1))

    with pytest.raises(RuntimeError, match="提交器迟到报错"):
        transfer.close()
    assert controller.recoveries == 1


def test_close_drains_recovery_queued_after_initial_submit_rejection() -> None:
    controller = Controller()
    submit_entered = threading.Event()
    release_submit = threading.Event()
    jobs: list[tuple[Callable[[], Any], concurrent.futures.Future[Any]]] = []
    submit_calls = 0

    def submit(operation: Callable[[], Any], /) -> concurrent.futures.Future[Any]:
        nonlocal submit_calls
        submit_calls += 1
        if submit_calls == 1:
            submit_entered.set()
            assert release_submit.wait(1)
            raise RuntimeError("首次提交拒绝")
        future: concurrent.futures.Future[Any] = concurrent.futures.Future()
        jobs.append((operation, future))
        return future

    transfer = MissionTransfer(controller, submitter=submit)
    start_error: list[BaseException] = []

    def start() -> None:
        try:
            transfer.start(_items(), "recon")
        except BaseException as exc:  # noqa: BLE001 - 收集线程内提交异常
            start_error.append(exc)

    starter = threading.Thread(target=start)
    starter.start()
    assert submit_entered.wait(1)
    transfer.cancel(lambda: setattr(controller, "recoveries", controller.recoveries + 1))

    close_returned = threading.Event()
    close_errors: list[BaseException] = []

    def close() -> None:
        try:
            transfer.close()
        except BaseException as exc:  # noqa: BLE001 - 验证close传播事务错误
            close_errors.append(exc)
        finally:
            close_returned.set()

    closer = threading.Thread(target=close)
    closer.start()
    release_submit.set()
    starter.join(timeout=1)
    assert not starter.is_alive()
    assert start_error and "首次提交拒绝" in str(start_error[0])
    assert len(jobs) == 1
    assert not close_returned.wait(0.02), "close必须等待后来登记的恢复Future"

    _run_job(jobs.pop())
    assert close_returned.wait(1)
    closer.join(timeout=1)
    assert not closer.is_alive()
    assert controller.recoveries == 1
    assert any("首次提交拒绝" in str(error) for error in close_errors)
