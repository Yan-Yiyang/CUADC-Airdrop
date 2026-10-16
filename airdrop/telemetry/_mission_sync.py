"""mission_raw 任务只读同步与同连接的传输串行化。"""

from __future__ import annotations

import asyncio
import logging
import threading
import weakref
from collections.abc import AsyncGenerator, AsyncIterable
from contextlib import aclosing
from typing import Any, cast

from .broker import TelemetryBroker

LOGGER = logging.getLogger(__name__)
_TRANSFER_LOCKS: weakref.WeakKeyDictionary[Any, asyncio.Lock] = weakref.WeakKeyDictionary()
_TRANSFER_LOCKS_GUARD = threading.Lock()


def mission_raw_transfer_lock(drone: Any) -> asyncio.Lock:
    """返回此 MAVSDK 连接的 mission_raw RPC 共用锁。"""
    try:
        with _TRANSFER_LOCKS_GUARD:
            lock = _TRANSFER_LOCKS.get(drone)
            if lock is None:
                lock = asyncio.Lock()
                _TRANSFER_LOCKS[drone] = lock
            return lock
    except TypeError:  # 测试替身等不可弱引用对象
        lock = getattr(drone, "_airdrop_mission_raw_lock", None)
        if lock is None:
            lock = asyncio.Lock()
            drone._airdrop_mission_raw_lock = lock  # noqa: SLF001 - 每个假连接保存自己的异步锁
        return lock


class MissionSync:
    """按变更通知回读任务项数，并隔离过期进度。"""

    def __init__(
        self,
        mission_raw: Any,
        broker: TelemetryBroker,
        *,
        drone: Any | None = None,
        retry_delay: float = 0.5,
        download_timeout: float = 15.0,
    ) -> None:
        self._mission_raw = mission_raw
        self._broker = broker
        self._lock = mission_raw_transfer_lock(drone) if drone is not None else asyncio.Lock()
        self._retry_delay = max(0.01, retry_delay)
        self._download_timeout = download_timeout
        self._changed = asyncio.Event()
        self._progress_changed = asyncio.Event()
        self._generation = 0
        self._synced_generation: int | None = None
        self._item_count: int | None = None
        self._saw_nonterminal = False

    async def run(self) -> None:
        """运行进度、变更与串行下载三个协程，取消时一并收尾。"""
        self._broker.clear_mission_progress()
        self._changed.set()
        try:
            async with asyncio.TaskGroup() as group:
                group.create_task(self._watch_changes())
                group.create_task(self._download_worker())
                group.create_task(self._progress_manager(group))
        finally:
            self._broker.clear_mission_progress()

    async def _watch_changes(self) -> None:
        async for changed in cast(AsyncIterable[Any], self._mission_raw.mission_changed()):
            if changed:
                self._generation += 1
                self._synced_generation = None
                self._item_count = None
                self._saw_nonterminal = False
                self._progress_changed.set()
                self._broker.clear_mission_progress()
                self._changed.set()
        raise RuntimeError("mission_changed 订阅已结束")

    async def _progress_manager(self, group: asyncio.TaskGroup) -> None:
        active: asyncio.Task[None] | None = None
        while True:
            await self._progress_changed.wait()
            self._progress_changed.clear()
            if active is not None and not active.done():
                active.cancel()
                try:
                    await active
                except asyncio.CancelledError:
                    current_task = asyncio.current_task()
                    if current_task is not None and current_task.cancelling():
                        raise
            generation = self._synced_generation
            count = self._item_count
            if generation is not None and generation == self._generation and count is not None:
                # 每个已确认代数都新建订阅；旧订阅即使迟到吐值也会检查代数后丢弃。
                active = group.create_task(self._watch_progress(generation, count))
            else:
                active = None

    async def _watch_progress(self, generation: int, count: int) -> None:
        stream = cast(AsyncGenerator[Any], self._mission_raw.mission_progress())
        async with aclosing(stream):
            async for progress in stream:
                if generation != self._generation or self._synced_generation != generation:
                    return
                current = int(progress.current)
                total = int(progress.total)
                if not 0 <= current <= total:
                    continue
                if total != count:
                    self._generation += 1
                    self._synced_generation = None
                    self._item_count = None
                    self._saw_nonterminal = False
                    self._progress_changed.set()
                    self._broker.clear_mission_progress()
                    self._changed.set()
                    return
                if current < total:
                    self._saw_nonterminal = True
                    self._broker.update_mission_progress(current, total)
                elif self._saw_nonterminal:
                    self._broker.update_mission_progress(current, total)
        raise RuntimeError("mission_progress 订阅已结束")

    async def _download_worker(self) -> None:
        while True:
            await self._changed.wait()
            self._changed.clear()
            generation = self._generation
            try:
                async with self._lock:
                    try:
                        items = await asyncio.wait_for(
                            self._mission_raw.download_mission(), timeout=self._download_timeout
                        )
                    except TimeoutError:
                        # 锁内取消只可能影响本 worker 刚发出的下载请求。
                        if await self._cancel_download():
                            raise asyncio.CancelledError from None
                        raise
                    except asyncio.CancelledError:
                        await self._cancel_download()
                        raise
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - MAVSDK RPC 异常按结果码区分空任务和重试
                if self._no_mission_available(exc):
                    items = []
                else:
                    LOGGER.debug("mission_raw 下载任务失败，将重试：%s", exc)
                    await asyncio.sleep(self._retry_delay)
                    if generation == self._generation:
                        self._changed.set()
                    continue

            if generation != self._generation:
                continue
            self._item_count = len(items)
            self._synced_generation = generation
            self._saw_nonterminal = False
            self._broker.update_mission_count(self._item_count)
            self._progress_changed.set()

    @staticmethod
    def _no_mission_available(exc: Exception) -> bool:
        """只把明确的无任务响应折算为空；其它 RPC 错误必须重试。"""
        result = getattr(getattr(exc, "_result", None), "result", None)
        return getattr(result, "name", None) == "NO_MISSION_AVAILABLE"

    async def _cancel_download(self) -> bool:
        """在传输锁内等待有界取消；返回清理期间是否再次收到取消。"""
        cancellation = asyncio.create_task(
            asyncio.wait_for(self._mission_raw.cancel_mission_download(), timeout=2.0)
        )
        interrupted = False
        while not cancellation.done():
            try:
                await asyncio.shield(cancellation)
            except asyncio.CancelledError:
                interrupted = True
            except Exception as exc:  # noqa: BLE001 - 取消失败不掩盖原始超时/取消
                LOGGER.debug("取消 mission_raw 下载未确认：%s", exc)
        return interrupted
