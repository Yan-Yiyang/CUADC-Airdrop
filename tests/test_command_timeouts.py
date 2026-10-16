from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import threading
from types import SimpleNamespace
from typing import Any, cast

import pytest
from mavsdk.action import ActionError, ActionResult

from airdrop.telemetry._operations import OperationTimeout, get_operation_gate
from airdrop.telemetry.controller import ControllerError, DroneController
from airdrop.telemetry.mavsdk_thread import MavsdkThread
from airdrop.telemetry.models import Command


class LoopThread:
    """离线真实事件循环线程，模拟 MAVSDK 的 submit 接口。"""

    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self.errors: list[dict[str, Any]] = []
        self.drone: Any = None
        self.loop.set_exception_handler(lambda _loop, context: self.errors.append(context))
        self.thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self.thread.start()

    def submit(self, coro: Any) -> concurrent.futures.Future[Any]:
        future = asyncio.run_coroutine_threadsafe(coro, self.loop)
        # 显式取回后台异常，避免测试通过但事件循环吞报错。
        future.add_done_callback(self._consume)
        return future

    @staticmethod
    def _consume(future: concurrent.futures.Future[Any]) -> None:
        with contextlib.suppress(Exception):
            future.result()

    def require_drone(self) -> Any:
        return self.drone

    def close(self) -> None:
        async def cancel_pending() -> None:
            current = asyncio.current_task()
            tasks = [
                task
                for task in asyncio.all_tasks(self.loop)
                if task is not current and not task.done()
            ]
            for task in tasks:
                task.cancel()
            if tasks:
                await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=0.5)

        asyncio.run_coroutine_threadsafe(cancel_pending(), self.loop).result(timeout=1)
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(timeout=1)
        assert not self.thread.is_alive(), "测试事件循环未能在期限内退出"
        self.loop.close()
        assert not self.errors, f"事件循环出现未处理异常：{self.errors!r}"


class BlockingAction:
    """可控的异步假动作；收尾也等待异步门闩，不阻塞事件循环。"""

    def __init__(self) -> None:
        self.entered = threading.Event()
        self.cleanup_entered = threading.Event()
        self.cleanup_finished = threading.Event()
        self.rtl_entered = threading.Event()
        self.cancel_count = 0
        self.release_cleanup = asyncio.Event()

    async def hold(self) -> None:
        self.entered.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancel_count += 1
            raise
        finally:
            self.cleanup_entered.set()
            await self.release_cleanup.wait()
            self.cleanup_finished.set()

    async def return_to_launch(self) -> None:
        self.rtl_entered.set()


def test_timeout_cancels_task_and_queued_recovery_cannot_run_late() -> None:
    """超时取消真实 task；收尾未结束时恢复动作超时后不能迟到下发。"""
    thread = LoopThread()
    action = BlockingAction()
    releases: list[int] = []

    class Gripper:
        async def release(self, instance: int) -> None:
            releases.append(instance)

    thread.drone = type("Drone", (), {"action": action})()
    thread.drone.gripper = Gripper()
    controller = DroneController(cast(Any, thread), timeout=0.08)
    recover_result: concurrent.futures.Future[Exception | None] = concurrent.futures.Future()

    def try_recover() -> None:
        try:
            controller.rtl()
        except ControllerError as exc:
            recover_result.set_result(exc)
        else:
            recover_result.set_result(None)

    try:
        with pytest.raises(ControllerError, match="结果未知|内没有结果|超时"):
            controller.hold()
        assert action.entered.wait(1), "hold 协程没有启动"
        assert action.cleanup_entered.wait(1), "超时后没有取消并进入 finally"

        caller = threading.Thread(target=try_recover, daemon=True)
        caller.start()
        failure = recover_result.result(timeout=1)
        assert isinstance(failure, ControllerError)
        assert action.cancel_count == 1, "等待收尾超时不能重复取消旧 task"
        assert not action.rtl_entered.is_set(), "旧协程收尾期间不能执行恢复动作"

        # 即使前序 finally 随后结束，已超时的排队恢复也不得被延迟执行。
        thread.loop.call_soon_threadsafe(action.release_cleanup.set)
        assert action.cleanup_finished.wait(1), "异步 finally 未能完成"
        caller.join(timeout=1)
        assert not caller.is_alive()
        assert not action.rtl_entered.is_set(), "超时退出的恢复动作被迟到下发"

        # 未知状态允许新的显式恢复动作；它只在前序 finally 完成后执行。
        controller.rtl()
        assert action.rtl_entered.is_set()
        with pytest.raises(ControllerError, match="未知|确认"):
            controller.gripper_release()
        assert releases == [], "恢复动作不应清除远端状态未知锁存"
    finally:
        thread.loop.call_soon_threadsafe(action.release_cleanup.set)
        thread.close()


def test_cancellation_resistant_operation_blocks_recovery_until_it_finishes() -> None:
    """协程吞掉取消后仍在运行时，gate 继续阻止恢复动作。"""
    thread = LoopThread()
    cancel_seen = threading.Event()
    operation_finished = threading.Event()
    recovery_entered = threading.Event()
    finish_operation = asyncio.Event()

    class ResistantAction:
        async def hold(self) -> None:
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancel_seen.set()
                await finish_operation.wait()
                operation_finished.set()

        async def return_to_launch(self) -> None:
            recovery_entered.set()

    thread.drone = type("Drone", (), {"action": ResistantAction()})()
    first = DroneController(cast(Any, thread), timeout=0.08)
    second = DroneController(cast(Any, thread), timeout=0.08)
    try:
        with pytest.raises(ControllerError, match="结果未知|内没有结果|超时"):
            first.hold()
        assert cancel_seen.wait(1), "超时未向运行中的协程请求取消"

        with pytest.raises(ControllerError, match="未知|超时|没有结果"):
            second.rtl()
        assert not recovery_entered.is_set(), "忽略取消的旧操作仍在运行时下发了 RTL"

        thread.loop.call_soon_threadsafe(finish_operation.set)
        assert operation_finished.wait(1), "忽略取消的协程未按测试门闩结束"
        second.rtl()
        assert recovery_entered.is_set()
    finally:
        thread.loop.call_soon_threadsafe(finish_operation.set)
        thread.close()


def test_query_sdk_timeout_does_not_poison_mutating_actions() -> None:
    """查询本身收到 SDK TimeoutError 后，可继续执行普通动作。"""
    thread = LoopThread()
    hold_entered = threading.Event()

    class Action:
        async def hold(self) -> None:
            hold_entered.set()

        async def return_to_launch(self) -> None:
            pass

    class Telemetry:
        async def get_gps_global_origin(self) -> Any:
            raise TimeoutError("SDK 查询超时")

    thread.drone = type("Drone", (), {"action": Action(), "telemetry": Telemetry()})()
    controller = DroneController(cast(Any, thread), timeout=0.5)
    try:
        assert controller.request_origin() is None
        controller.hold()
        assert hold_entered.is_set()
    finally:
        thread.close()


def test_send_command_and_controller_share_thread_gate() -> None:
    """send_command 排队在 controller 的未收尾操作后，超时后不得迟到下发。"""
    loop_thread = LoopThread()
    altitude_calls: list[float] = []

    async def set_takeoff_altitude(altitude_m: float) -> None:
        altitude_calls.append(altitude_m)

    class CommandAction(BlockingAction):
        async def set_takeoff_altitude(self, altitude_m: float) -> None:
            await set_takeoff_altitude(altitude_m)

    action = CommandAction()
    sdk_thread = MavsdkThread(reconnect=False)
    sdk_thread._loop = loop_thread.loop
    sdk_thread._drone = cast(Any, type("Drone", (), {"action": action})())
    sdk_thread.submit = loop_thread.submit  # type: ignore[method-assign]
    controller = DroneController(sdk_thread, timeout=0.06)
    try:
        with pytest.raises(ControllerError, match="结果未知|超时|没有结果"):
            controller.hold()
        assert action.cleanup_entered.wait(1)

        result = sdk_thread.send_command(
            Command("set_takeoff_altitude", {"altitude_m": 42.0}), timeout=0.06
        )
        assert not result.success
        assert altitude_calls == [], "等待 controller 收尾超时的指令不应被提交"

        loop_thread.loop.call_soon_threadsafe(action.release_cleanup.set)
        assert action.cleanup_finished.wait(1)
        # 让事件循环处理完已经排队的回调，再确认没有被超时指令迟到调用。
        loop_thread.submit(asyncio.sleep(0)).result(timeout=1)
        assert altitude_calls == []
    finally:
        loop_thread.loop.call_soon_threadsafe(action.release_cleanup.set)
        loop_thread.close()


def test_sdk_timeout_on_mutation_latches_unknown_result() -> None:
    """MAVSDK 自身 TIMEOUT 不是调用方 deadline，仍锁存变更结果未知。"""
    thread = LoopThread()
    releases: list[int] = []

    class SdkTimeout(RuntimeError):
        _result = SimpleNamespace(result=SimpleNamespace(name="TIMEOUT"))

    class Action:
        async def hold(self) -> None:
            raise SdkTimeout("插件超时")

        async def return_to_launch(self) -> None:
            pass

    class Gripper:
        async def release(self, instance: int) -> None:
            releases.append(instance)

    thread.drone = type("Drone", (), {"action": Action(), "gripper": Gripper()})()
    controller = DroneController(cast(Any, thread), timeout=0.5)
    try:
        with pytest.raises(ControllerError, match="SDK 调用超时|TIMEOUT"):
            controller.hold()
        with pytest.raises(ControllerError, match="结果未知|确认"):
            controller.gripper_release()
        assert releases == []
    finally:
        thread.close()


def test_submit_failure_closes_wrapper_and_gate_remains_usable() -> None:
    """submit 拒绝任务时关闭惰性 wrapper，随后仍可正常提交。"""
    factory_calls: list[str] = []
    loop_thread = LoopThread()

    class FailedSubmitThread:
        def __init__(self) -> None:
            self.fail_submit = True
            self.wrappers: list[Any] = []

        def submit(self, _coro: Any) -> concurrent.futures.Future[Any]:
            self.wrappers.append(_coro)
            if self.fail_submit:
                raise RuntimeError("loop 已关闭")
            return loop_thread.submit(_coro)

        def require_drone(self) -> Any:
            return SimpleNamespace(action=SimpleNamespace(hold=hold))

    async def hold() -> None:
        factory_calls.append("called")

    thread = FailedSubmitThread()
    controller = DroneController(cast(Any, thread), timeout=0.5)
    try:
        with pytest.raises(ControllerError, match="loop 已关闭"):
            controller.hold()
        assert factory_calls == []
        assert len(thread.wrappers) == 1
        assert thread.wrappers[0].cr_frame is None, "submit 失败后 wrapper 应已关闭"

        thread.fail_submit = False
        controller.hold()
        assert factory_calls == ["called"], "失败提交后 gate 应允许后续正常操作"
    finally:
        loop_thread.close()


def test_nested_transaction_uses_outer_deadline() -> None:
    """复合调用内层 RPC 继承外层剩余预算，不重置整个 timeout。"""
    thread = LoopThread()
    gate = get_operation_gate(thread)
    submits: list[bool] = []

    def submit(coro: Any) -> concurrent.futures.Future[Any]:
        submits.append(True)
        return thread.submit(coro)

    async def operation() -> None:
        await asyncio.Event().wait()

    try:
        with gate.transaction(0.03, "外层事务"):
            threading.Event().wait(0.05)
            with pytest.raises(OperationTimeout, match="预算已耗尽|锁超时"):
                gate.run(
                    submit,
                    operation,
                    timeout=1.0,
                    label="内层 RPC",
                    mutating=True,
                )
        assert submits == [], "共享截止时间已耗尽后不应提交内层 RPC"
    finally:
        thread.close()


def test_mavsdk_submit_passes_remaining_budget_to_start() -> None:
    """submit 把复合调用的剩余预算传给慢 start，耗尽后不调度 RPC。"""
    sdk_thread = MavsdkThread(reconnect=False)
    gate = get_operation_gate(sdk_thread)
    start_budgets: list[float] = []
    factory_calls: list[bool] = []

    def slow_start(timeout: float = 5.0) -> None:
        start_budgets.append(timeout)
        threading.Event().wait(timeout + 0.02)

    sdk_thread.start = slow_start  # type: ignore[method-assign]

    async def operation() -> None:
        factory_calls.append(True)

    with (
        gate.transaction(0.03, "带预算的复合调用"),
        pytest.raises(OperationTimeout, match="预算已耗尽|超时"),
    ):
        gate.run(
            sdk_thread.submit,
            operation,
            timeout=1.0,
            label="提交指令",
            mutating=True,
        )
    assert len(start_budgets) == 1
    assert 0 < start_budgets[0] <= 0.03
    assert factory_calls == [], "start 耗尽预算后不能将 RPC 排入事件循环"


def test_timeout_before_wrapper_starts_does_not_call_factory_when_later_scheduled() -> None:
    """submit 暂存 wrapper 后调用方超时，迟到调度必须跳过 factory。"""
    loop_thread = LoopThread()
    hold_calls: list[str] = []
    rtl_calls: list[str] = []

    class Action:
        async def hold(self) -> None:
            hold_calls.append("hold")

        async def return_to_launch(self) -> None:
            rtl_calls.append("rtl")

    class DeferredSubmitThread:
        def __init__(self) -> None:
            self.defer = True
            self.wrappers: list[Any] = []
            self.pending: concurrent.futures.Future[Any] | None = None
            self.drone = SimpleNamespace(action=Action())

        def submit(self, coro: Any) -> concurrent.futures.Future[Any]:
            if self.defer:
                self.wrappers.append(coro)
                self.pending = concurrent.futures.Future()
                return self.pending
            return loop_thread.submit(coro)

        def require_drone(self) -> Any:
            return self.drone

    thread = DeferredSubmitThread()
    controller = DroneController(cast(Any, thread), timeout=0.06)
    try:
        with pytest.raises(ControllerError, match="结果未知|超时|没有结果"):
            controller.hold()
        assert len(thread.wrappers) == 1

        # 模拟 submit 已暂存但尚未安排到 loop，之后迟到执行 wrapper。
        delayed = loop_thread.submit(thread.wrappers[0])
        delayed.result(timeout=1)
        assert hold_calls == [], "调用方 deadline 后迟到启动的 wrapper 不得运行 factory"

        thread.defer = False
        controller.rtl()
        assert rtl_calls == ["rtl"], "迟到 wrapper 收尾后应允许恢复动作"
    finally:
        if thread.pending is not None:
            thread.pending.cancel()
        loop_thread.close()


def test_action_error_timeout_is_returned_as_unknown_sdk_timeout() -> None:
    """真实 MAVSDK ActionError(TIMEOUT) 要传播到 send_command 的结果字段。"""
    loop_thread = LoopThread()

    class Action:
        async def hold(self) -> None:
            raise ActionError(
                SimpleNamespace(result=ActionResult.Result.TIMEOUT, result_str="超时"), "fake"
            )

    sdk_thread = MavsdkThread(reconnect=False)
    sdk_thread._loop = loop_thread.loop
    sdk_thread._drone = cast(Any, SimpleNamespace(action=Action()))
    sdk_thread.submit = loop_thread.submit  # type: ignore[method-assign]
    try:
        result = sdk_thread.send_command(Command("hold"), timeout=0.5)
        assert not result.success
        assert result.data == {"result_unknown": True, "result": "TIMEOUT"}
    finally:
        loop_thread.close()


def test_start_mission_reset_and_start_are_not_interleaved() -> None:
    """复位 RPC 收尾到启动 RPC 完成期间，另一个控制器不能插入动作。"""
    loop_thread = LoopThread()
    calls: list[str] = []
    rewind_entered = threading.Event()
    release_rewind = asyncio.Event()
    competitor_attempting = threading.Event()

    class MissionRaw:
        async def set_current_mission_item(self, _index: int) -> None:
            calls.append("rewind")
            rewind_entered.set()
            await release_rewind.wait()
            calls.append("rewind_done")

    class Mission:
        async def start_mission(self) -> None:
            calls.append("start")

    class Action:
        async def return_to_launch(self) -> None:
            calls.append("rtl")

    loop_thread.drone = type(
        "Drone",
        (),
        {"mission_raw": MissionRaw(), "mission": Mission(), "action": Action()},
    )()
    start_controller = DroneController(cast(Any, loop_thread), timeout=1.0)
    other_controller = DroneController(cast(Any, loop_thread), timeout=1.0)
    start_result: concurrent.futures.Future[ControllerError | None] = concurrent.futures.Future()
    rtl_result: concurrent.futures.Future[ControllerError | None] = concurrent.futures.Future()

    def start() -> None:
        try:
            start_controller.start_mission()
        except ControllerError as exc:
            start_result.set_result(exc)
        else:
            start_result.set_result(None)

    def rtl() -> None:
        competitor_attempting.set()
        try:
            other_controller.rtl()
        except ControllerError as exc:
            rtl_result.set_result(exc)
        else:
            rtl_result.set_result(None)

    start_caller = threading.Thread(target=start, daemon=True)
    try:
        start_caller.start()
        assert rewind_entered.wait(1), "未开始复位任务项"
        rtl_caller = threading.Thread(target=rtl, daemon=True)
        rtl_caller.start()
        assert competitor_attempting.wait(1)
        # 让另一调用进入门控等待后再放行复位 RPC。
        threading.Event().wait(0.02)
        loop_thread.loop.call_soon_threadsafe(release_rewind.set)

        assert start_result.result(timeout=1) is None
        assert rtl_result.result(timeout=1) is None
        start_caller.join(timeout=1)
        rtl_caller.join(timeout=1)
        assert not start_caller.is_alive() and not rtl_caller.is_alive()
        assert calls == ["rewind", "rewind_done", "start", "rtl"]
    finally:
        loop_thread.loop.call_soon_threadsafe(release_rewind.set)
        loop_thread.close()
