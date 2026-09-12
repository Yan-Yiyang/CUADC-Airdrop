"""跨进程遥测与指令桥接。

外部桥接通过 ``multiprocessing.Manager`` dict 与其他进程共享最新遥测快照，
并通过 manager 队列接受来自其他进程的指令。桥接在所属进程中运行两个小型
守护线程：

* 发布线程：定时把进程内快照推送到共享 dict；
* 指令消费线程：把外部指令转发给 MAVSDK 线程，并把结果发布到结果队列。
"""

from __future__ import annotations

import logging
import multiprocessing
import queue
import threading
from multiprocessing.managers import DictProxy, SyncManager
from typing import Any

from .broker import TelemetryBroker
from .mavsdk_thread import MavsdkThread
from .models import Command

LOGGER = logging.getLogger(__name__)


class ExternalBridge:
    """向其他进程发布遥测，并从其他进程接收指令。"""

    def __init__(
        self,
        mavsdk_thread: MavsdkThread,
        broker: TelemetryBroker | None = None,
        publish_hz: float = 20.0,
    ) -> None:
        self.mavsdk_thread = mavsdk_thread
        self.broker = broker or mavsdk_thread.broker
        self.publish_hz = publish_hz
        self._manager: SyncManager | None = None
        self._snapshot: DictProxy | None = None
        self._command_queue: Any = None
        self._result_queue: Any = None
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    def start(self) -> None:
        """启动 manager、发布线程和指令消费线程。"""
        if self._manager is not None:
            return

        self._manager = multiprocessing.Manager()
        self._snapshot = self._manager.dict(self.broker.get_snapshot().as_dict())
        self._command_queue = self._manager.Queue()
        self._result_queue = self._manager.Queue()

        publisher = threading.Thread(
            target=self._publish_loop,
            name="external-telemetry-publisher",
            daemon=True,
        )
        consumer = threading.Thread(
            target=self._command_loop,
            name="external-command-consumer",
            daemon=True,
        )
        self._threads = [publisher, consumer]
        publisher.start()
        consumer.start()

        LOGGER.info(
            "外部桥接已启动：快照发布频率 %.1f Hz，指令队列已就绪",
            self.publish_hz,
        )

    def stop(self, timeout: float = 5.0) -> None:
        """停止桥接线程并关闭 manager。"""
        self._stop.set()
        if self._command_queue is not None:
            try:
                self._command_queue.put(None, timeout=1.0)
            except Exception:  # noqa: BLE001 - 哨兵发送失败也继续退出
                pass

        for thread in self._threads:
            thread.join(timeout=timeout)
        self._threads.clear()

        if self._manager is not None:
            self._manager.shutdown()
            self._manager = None
        self._snapshot = None
        self._command_queue = None
        self._result_queue = None

    # ------------------------------------------------------------------
    # 供子进程使用的属性
    # ------------------------------------------------------------------
    @property
    def snapshot(self) -> DictProxy:
        """共享 dict 代理；子进程可读取 ``snapshot["north_m"]`` 等字段。"""
        if self._snapshot is None:
            raise RuntimeError("ExternalBridge 尚未启动")
        return self._snapshot

    @property
    def command_queue(self) -> Any:
        """用于向 MAVSDK 线程发送指令 dict 的队列。"""
        if self._command_queue is None:
            raise RuntimeError("ExternalBridge 尚未启动")
        return self._command_queue

    @property
    def result_queue(self) -> Any:
        """用于发布指令结果 dict 的队列。"""
        if self._result_queue is None:
            raise RuntimeError("ExternalBridge 尚未启动")
        return self._result_queue

    # ------------------------------------------------------------------
    # 内部实现
    # ------------------------------------------------------------------
    def _publish_loop(self) -> None:
        interval = 1.0 / max(self.publish_hz, 0.1)
        while not self._stop.is_set():
            try:
                if self._snapshot is not None:
                    self._snapshot.update(self.broker.get_snapshot().as_dict())
            except Exception:  # noqa: BLE001 - 单次发布失败不影响后续发布
                LOGGER.exception("发布外部快照失败")
            # 用事件等待代替 sleep，stop() 时立即退出
            self._stop.wait(interval)

    def _command_loop(self) -> None:
        while not self._stop.is_set():
            command_queue = self._command_queue
            if command_queue is None:
                break
            try:
                raw = command_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            except Exception:  # noqa: BLE001
                if self._stop.is_set():
                    break
                LOGGER.exception("读取外部指令队列失败")
                continue

            if raw is None:
                break

            try:
                command = Command.from_dict(raw)
                result = self.mavsdk_thread.send_command(command)
                if self._result_queue is not None:
                    self._result_queue.put(result.as_dict())
            except Exception:  # noqa: BLE001 - 单条指令失败不影响后续指令
                LOGGER.exception("处理外部指令失败")
