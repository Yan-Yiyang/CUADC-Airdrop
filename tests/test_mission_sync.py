"""mission_raw 任务同步的异步边界回归。"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any, cast

import pytest
from mavsdk.mission_raw import MissionRawError, MissionRawResult

from airdrop.telemetry import TelemetryBroker
from airdrop.telemetry._mission_sync import MissionSync


class FakeMissionRaw:
    """可控的 mission_raw 异步流与下载 RPC。"""

    def __init__(self) -> None:
        self.changed: asyncio.Queue[bool] = asyncio.Queue()
        self.progress: asyncio.Queue[Any] = asyncio.Queue()
        self.downloads: list[list[Any]] = []
        self.download_started: asyncio.Queue[asyncio.Event] = asyncio.Queue()
        self.download_results: asyncio.Queue[Any] = asyncio.Queue()
        self.upload_calls = 0
        self.clear_calls = 0
        self.cancel_calls = 0
        self.cancel_started = asyncio.Event()
        self.finish_cancel = asyncio.Event()
        self.finish_cancel.set()
        self.end_progress = False
        self.end_changed = False

    async def mission_changed(self) -> AsyncIterator[bool]:
        while True:
            value = await self.changed.get()
            if value is None:
                return
            yield value

    async def mission_progress(self) -> AsyncIterator[Any]:
        while True:
            value = await self.progress.get()
            if value is None:
                return
            yield value

    async def download_mission(self) -> list[Any]:
        self.downloads.append([])
        started = asyncio.Event()
        await self.download_started.put(started)
        started.set()
        result = await self.download_results.get()
        if isinstance(result, BaseException):
            raise result
        self.downloads[-1] = list(result)
        return self.downloads[-1]

    async def upload_mission(self, items: list[Any]) -> None:
        self.upload_calls += 1

    async def clear_mission(self) -> None:
        self.clear_calls += 1

    async def cancel_mission_download(self) -> None:
        self.cancel_calls += 1
        self.cancel_started.set()
        if self.finish_cancel.is_set():
            return
        await self.finish_cancel.wait()


class SimpleProgress:
    def __init__(self, *, current: int, total: int) -> None:
        self.current = current
        self.total = total


def _item() -> object:
    return object()


async def _eventually(predicate: Any) -> None:
    for _ in range(100):
        if predicate():
            return
        await asyncio.sleep(0.001)
    pytest.fail("异步状态未在预期时间内更新")


def test_initial_sync_counts_mission_and_ignores_old_terminal_progress() -> None:
    async def scenario() -> None:
        raw = FakeMissionRaw()
        broker = TelemetryBroker()
        raw.download_results.put_nowait([_item(), _item(), _item()])
        task = asyncio.create_task(MissionSync(raw, broker, retry_delay=0.001).run())
        try:
            await _eventually(lambda: broker.get_snapshot().mission_total == 3)
            assert broker.get_snapshot().mission_current is None
            await raw.progress.put(SimpleProgress(current=3, total=3))
            await asyncio.sleep(0)
            assert broker.get_snapshot().mission_current is None
            await raw.progress.put(SimpleProgress(current=1, total=3))
            await _eventually(lambda: broker.get_snapshot().mission_current == 1)
            await raw.progress.put(SimpleProgress(current=3, total=3))
            await _eventually(lambda: broker.get_snapshot().mission_current == 3)
            assert len(raw.downloads) == 1
            assert raw.upload_calls == raw.clear_calls == 0
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    asyncio.run(scenario())


def test_change_during_download_discards_old_generation_and_retries_latest() -> None:
    async def scenario() -> None:
        raw = FakeMissionRaw()
        broker = TelemetryBroker()
        download_started = [asyncio.Event(), asyncio.Event(), asyncio.Event()]

        async def gated_download() -> list[Any]:
            raw.downloads.append([])
            download_started[len(raw.downloads) - 1].set()
            result = await raw.download_results.get()
            raw.downloads[-1] = list(result)
            return raw.downloads[-1]

        raw.download_mission = gated_download  # type: ignore[method-assign]
        task = asyncio.create_task(MissionSync(raw, broker, retry_delay=0.001).run())
        try:
            # 初始下载返回 2 项。
            await asyncio.wait_for(download_started[0].wait(), timeout=1)
            raw.download_results.put_nowait([_item(), _item()])
            await _eventually(lambda: broker.get_snapshot().mission_total == 2)

            # 旧代下载在途时，连续两次变更；中间结果必须丢弃，只保留最后一代。
            await raw.changed.put(True)
            await asyncio.wait_for(download_started[1].wait(), timeout=1)
            await raw.changed.put(True)
            raw.download_results.put_nowait([_item(), _item(), _item()])
            await asyncio.wait_for(download_started[2].wait(), timeout=1)
            assert broker.get_snapshot().mission_total is None
            raw.download_results.put_nowait([_item(), _item(), _item(), _item()])
            await _eventually(lambda: broker.get_snapshot().mission_total == 4)
            assert broker.get_snapshot().mission_current is None
            assert len(raw.downloads) == 3
            assert raw.upload_calls == raw.clear_calls == 0
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    asyncio.run(scenario())


def test_empty_mission_and_failed_download_retry_without_faking_empty() -> None:
    async def scenario() -> None:
        raw = FakeMissionRaw()
        broker = TelemetryBroker()
        task = asyncio.create_task(MissionSync(raw, broker, retry_delay=0.001).run())
        try:
            await asyncio.wait_for(raw.download_started.get(), timeout=1)
            raw.download_results.put_nowait(
                MissionRawError(MissionRawResult(MissionRawResult.Result.BUSY, "busy"), "fake")
            )
            await asyncio.wait_for(raw.download_started.get(), timeout=1)
            assert broker.get_snapshot().mission_total is None
            raw.download_results.put_nowait(
                MissionRawError(
                    MissionRawResult(MissionRawResult.Result.TIMEOUT, "timeout"), "fake"
                )
            )
            await asyncio.wait_for(raw.download_started.get(), timeout=1)
            assert broker.get_snapshot().mission_total is None
            raw.download_results.put_nowait(
                MissionRawError(
                    MissionRawResult(MissionRawResult.Result.NO_MISSION_AVAILABLE, "empty mission"),
                    "fake",
                )
            )
            await _eventually(lambda: broker.get_snapshot().mission_total == 0)
            assert broker.get_snapshot().mission_current is None
            await raw.progress.put(SimpleProgress(current=0, total=0))
            await asyncio.sleep(0)
            assert broker.get_snapshot().mission_current is None
            assert raw.upload_calls == raw.clear_calls == 0
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    asyncio.run(scenario())


def test_cancel_stops_all_mission_workers() -> None:
    async def scenario() -> None:
        raw = FakeMissionRaw()
        broker = TelemetryBroker()
        task = asyncio.create_task(MissionSync(raw, broker, retry_delay=0.001).run())
        await asyncio.wait_for(raw.download_started.get(), timeout=1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert raw.cancel_calls == 1
        assert raw.upload_calls == raw.clear_calls == 0

    asyncio.run(scenario())


def test_new_generation_subscribes_after_count_and_ignores_late_old_progress() -> None:
    async def scenario() -> None:
        raw = FakeMissionRaw()
        broker = TelemetryBroker()
        progress_queues: list[asyncio.Queue[Any]] = []
        raw.download_results.put_nowait([_item(), _item(), _item()])

        async def progress_stream() -> AsyncIterator[Any]:
            queue: asyncio.Queue[Any] = asyncio.Queue()
            progress_queues.append(queue)
            try:
                while True:
                    value = await queue.get()
                    if value is None:
                        return
                    yield value
            except asyncio.CancelledError:
                if len(progress_queues) == 1:
                    # 模拟取消已发出但旧 gRPC 流仍吐出缓存项。
                    yield SimpleProgress(current=1, total=3)
                    yield SimpleProgress(current=3, total=3)
                    return
                raise

        raw.mission_progress = progress_stream  # type: ignore[method-assign]
        task = asyncio.create_task(MissionSync(raw, broker, retry_delay=0.001).run())
        try:
            await _eventually(lambda: broker.get_snapshot().mission_total == 3)
            await _eventually(lambda: len(progress_queues) == 1)
            await raw.changed.put(True)
            await _eventually(lambda: broker.get_snapshot().mission_total is None)
            # 新代下载仍为同样条数；旧流的迟到 nonterminal/terminal 不得发布。
            await _eventually(lambda: len(raw.downloads) >= 2)
            raw.download_results.put_nowait([_item(), _item(), _item()])
            await _eventually(lambda: len(progress_queues) == 2)
            await asyncio.sleep(0)
            assert broker.get_snapshot().mission_current is None
            await progress_queues[1].put(SimpleProgress(current=0, total=3))
            await _eventually(lambda: broker.get_snapshot().mission_current == 0)
            await progress_queues[1].put(SimpleProgress(current=3, total=3))
            await _eventually(lambda: broker.get_snapshot().mission_current == 3)
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    asyncio.run(scenario())


def test_subscription_eof_clears_mission_fields_and_ends_sync_group() -> None:
    async def scenario() -> None:
        raw = FakeMissionRaw()
        broker = TelemetryBroker()
        raw.download_results.put_nowait([_item(), _item()])
        task = asyncio.create_task(MissionSync(raw, broker, retry_delay=0.001).run())
        await _eventually(lambda: broker.get_snapshot().mission_total == 2)
        await _eventually(lambda: not raw.download_started.empty())
        await raw.changed.put(None)  # type: ignore[arg-type]
        with pytest.raises(ExceptionGroup):
            await task
        assert broker.get_snapshot().mission_current is None
        assert broker.get_snapshot().mission_total is None

    asyncio.run(scenario())


def test_out_of_range_progress_cannot_unlock_terminal_completion() -> None:
    async def scenario() -> None:
        raw = FakeMissionRaw()
        broker = TelemetryBroker()
        raw.download_results.put_nowait([_item(), _item()])
        task = asyncio.create_task(MissionSync(raw, broker, retry_delay=0.001).run())
        try:
            await _eventually(lambda: broker.get_snapshot().mission_total == 2)
            await raw.progress.put(SimpleProgress(current=-1, total=2))
            await raw.progress.put(SimpleProgress(current=3, total=2))
            await raw.progress.put(SimpleProgress(current=2, total=2))
            await asyncio.sleep(0.005)
            assert broker.get_snapshot().mission_current is None
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    asyncio.run(scenario())


def test_cancel_keeps_transfer_lock_until_server_cancel_finishes() -> None:
    async def scenario() -> None:
        from airdrop.telemetry._mission_sync import mission_raw_transfer_lock

        raw = FakeMissionRaw()
        drone = SimpleNamespace()
        lock = mission_raw_transfer_lock(drone)
        raw.finish_cancel.clear()
        task = asyncio.create_task(
            MissionSync(
                raw, TelemetryBroker(), drone=drone, download_timeout=60.0, retry_delay=0.001
            ).run()
        )
        await asyncio.wait_for(raw.download_started.get(), timeout=1)
        task.cancel()
        await asyncio.wait_for(raw.cancel_started.wait(), timeout=1)
        assert lock.locked()
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(lock.acquire(), timeout=0.001)
        raw.finish_cancel.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not lock.locked()
        await asyncio.wait_for(lock.acquire(), timeout=0.1)
        lock.release()

    asyncio.run(scenario())


def test_group_cancellation_waits_for_inflight_timeout_cleanup() -> None:
    async def scenario() -> None:
        from airdrop.telemetry._mission_sync import mission_raw_transfer_lock

        raw = FakeMissionRaw()
        raw.finish_cancel.clear()
        drone = SimpleNamespace()
        lock = mission_raw_transfer_lock(drone)
        task = asyncio.create_task(
            MissionSync(
                raw, TelemetryBroker(), drone=drone, download_timeout=0.01, retry_delay=0.001
            ).run()
        )
        await asyncio.wait_for(raw.download_started.get(), timeout=1)
        await asyncio.wait_for(raw.cancel_started.wait(), timeout=1)
        # EOF 取消 worker 时，server 取消 RPC 仍须持锁运行至收尾。
        await raw.changed.put(None)  # type: ignore[arg-type]
        await asyncio.sleep(0)
        assert lock.locked()
        raw.finish_cancel.set()
        with pytest.raises(ExceptionGroup):
            await task
        assert raw.cancel_calls == 1
        assert not lock.locked()

    asyncio.run(scenario())


def test_run_cancellation_during_old_progress_subscription_cleanup() -> None:
    async def scenario() -> None:
        raw = FakeMissionRaw()
        raw.download_results.put_nowait([_item(), _item()])
        cleanup_started = asyncio.Event()
        finish_cleanup = asyncio.Event()

        async def delayed_progress() -> AsyncIterator[Any]:
            try:
                await asyncio.Event().wait()
                yield None
            except asyncio.CancelledError:
                cleanup_started.set()
                await finish_cleanup.wait()
                raise

        raw.mission_progress = delayed_progress  # type: ignore[method-assign]
        broker = TelemetryBroker()
        task = asyncio.create_task(MissionSync(raw, broker, retry_delay=0.001).run())
        try:
            await _eventually(lambda: broker.get_snapshot().mission_total == 2)
            await raw.changed.put(True)
            await asyncio.wait_for(cleanup_started.wait(), timeout=1)
            task.cancel()
            await asyncio.sleep(0)
            finish_cleanup.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=1)
            assert broker.get_snapshot().mission_total is None
        finally:
            finish_cleanup.set()
            if not task.done():
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task

    asyncio.run(scenario())


def test_mavsdk_collector_starts_mission_sync_for_connected_drone() -> None:
    async def scenario() -> None:
        from airdrop.telemetry.mavsdk_thread import MavsdkThread

        raw = FakeMissionRaw()
        raw.download_results.put_nowait([_item(), _item()])
        broker = TelemetryBroker()
        mav = MavsdkThread(broker=broker)
        blocker = asyncio.Event()

        async def idle(drone: Any) -> None:
            await blocker.wait()

        for name in (
            "_stream_position",
            "_stream_home",
            "_stream_position_velocity_ned",
            "_stream_attitude_euler",
            "_stream_attitude_quaternion",
            "_refresh_origin_loop",
        ):
            setattr(mav, name, idle)

        class FakeTelemetry:
            async def wind(self) -> AsyncIterator[Any]:
                await blocker.wait()
                if False:
                    yield None

            async def flight_mode(self) -> AsyncIterator[Any]:
                await blocker.wait()
                if False:
                    yield None

            async def in_air(self) -> AsyncIterator[Any]:
                await blocker.wait()
                if False:
                    yield None

        drone = SimpleNamespace(telemetry=FakeTelemetry(), mission_raw=raw)
        task = asyncio.create_task(mav._collect_telemetry(cast(Any, drone)))
        try:
            await _eventually(lambda: broker.get_snapshot().mission_total == 2)
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        assert raw.upload_calls == raw.clear_calls == 0

    asyncio.run(scenario())
