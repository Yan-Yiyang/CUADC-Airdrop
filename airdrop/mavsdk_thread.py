"""专用的 MAVSDK 工作线程。

MAVSDK 基于 asyncio/gRPC。把它运行在专用线程并拥有独立事件循环，
可以避免它的协程污染（或被污染）应用中其他协程/事件循环。
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import threading
from typing import Any, Coroutine, Dict, Optional, Union

from mavsdk import System

from .broker import TelemetryBroker
from .models import Command, CommandResult, TelemetrySnapshot

LOGGER = logging.getLogger(__name__)


class MavsdkThread:
    """在专用后台线程中运行一个 MAVSDK :class:`System`。

    公开方法都是同步且线程安全的：它们会把任务调度到 MAVSDK 事件循环上，
    然后等待结果。任何 MAVSDK 协程都不会在调用方的线程/事件循环中被 await。
    """

    SUPPORTED_COMMANDS = {
        "arm",
        "arm_force",
        "disarm",
        "takeoff",
        "land",
        "hold",
        "kill",
        "rtl",
        "return_to_launch",
        "set_takeoff_altitude",
        "set_return_to_launch_altitude",
        "set_current_speed",
        "get_snapshot_at",
        "get_gps_global_origin",
    }

    def __init__(
        self,
        broker: Optional[TelemetryBroker] = None,
        system_address: str = "udp://:14540",
        thread_name: str = "mavsdk-thread",
    ) -> None:
        self.broker = broker or TelemetryBroker()
        self.system_address = system_address
        self._thread_name = thread_name
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._started = threading.Event()
        self._system: Optional[System] = None
        self._telemetry_tasks: list[asyncio.Task[Any]] = []

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    def start(self, timeout: float = 5.0) -> None:
        """启动专用的事件循环线程。"""
        if self._thread and self._thread.is_alive():
            return

        self._thread = threading.Thread(
            target=self._run_loop,
            name=self._thread_name,
            daemon=True,
        )
        self._thread.start()
        if not self._started.wait(timeout):
            raise RuntimeError("MAVSDK 线程未能启动事件循环")

    def stop(self, timeout: float = 5.0) -> None:
        """停止事件循环并等待工作线程退出。"""
        if self._loop is None:
            return

        try:
            shutdown = asyncio.run_coroutine_threadsafe(
                self._shutdown(), self._loop
            )
            shutdown.result(timeout=timeout)
        except Exception:
            LOGGER.exception("关闭 MAVSDK 线程时出错")

        if self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)

        if self._thread:
            self._thread.join(timeout=timeout)
            self._thread = None

        self._loop = None

    def connect(self, system_address: Optional[str] = None, timeout: float = 15.0) -> System:
        """连接飞机（同步方法，可从任意线程调用）。"""
        self.start()
        address = system_address or self.system_address
        future = self.submit(self._connect(address))
        try:
            return future.result(timeout=timeout)
        except Exception:
            LOGGER.exception("MAVSDK 连接失败")
            raise

    # ------------------------------------------------------------------
    # 遥测 / 指令接口
    # ------------------------------------------------------------------
    def get_snapshot(self):
        """返回最新遥测快照（线程安全的快捷方式）。"""
        return self.broker.get_snapshot()

    def get_snapshot_at(
        self,
        timestamp: float,
        mode: str = "interpolate",
    ) -> Optional[TelemetrySnapshot]:
        """按时间戳查询历史/内插外推快照（线程安全的快捷方式）。"""
        return self.broker.get_snapshot_at(timestamp, mode)

    def get_gps_global_origin(self, timeout: float = 10.0):
        """获取本地 NED 坐标系原点（GPS_GLOBAL_ORIGIN）并写入快照。"""
        future = self.submit(self._get_gps_global_origin())
        return future.result(timeout=timeout)

    def send_command(
        self,
        command: Union[Command, Dict[str, Any]],
        timeout: float = 30.0,
    ) -> CommandResult:
        """向 MAVSDK 线程发送一条指令并等待结果。"""
        if isinstance(command, dict):
            command = Command.from_dict(command)

        if command.name not in self.SUPPORTED_COMMANDS:
            return CommandResult(
                id=command.id,
                name=command.name,
                success=False,
                error=f"不支持的指令: {command.name}",
            )

        future = self.submit(self._execute_command(command))
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            return CommandResult(
                id=command.id,
                name=command.name,
                success=False,
                error=f"指令在 {timeout}s 内超时",
            )
        except Exception as exc:  # pragma: no cover - 防御性处理
            return CommandResult(
                id=command.id,
                name=command.name,
                success=False,
                error=f"{type(exc).__name__}: {exc}",
            )

    def submit(
        self, coro: Coroutine[Any, Any, Any]
    ) -> concurrent.futures.Future[Any]:
        """把协程调度到 MAVSDK 事件循环上执行。

        调用方可以用 ``future.result(timeout=...)`` 同步等待结果。
        """
        self.start()
        if self._loop is None:
            raise RuntimeError("MAVSDK 事件循环不可用")
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    # ------------------------------------------------------------------
    # 内部：事件循环线程
    # ------------------------------------------------------------------
    def _run_loop(self) -> None:
        try:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._started.set()
            LOGGER.info("MAVSDK 事件循环已在线程 %s 中启动", self._thread_name)
            self._loop.run_forever()
        finally:
            self._cleanup_pending_tasks()
            if self._loop is not None:
                self._loop.close()
                self._loop = None

    def _cleanup_pending_tasks(self) -> None:
        if self._loop is None or self._loop.is_closed():
            return
        pending = asyncio.all_tasks(self._loop)
        for task in pending:
            task.cancel()
        if pending:
            self._loop.run_until_complete(
                asyncio.gather(*pending, return_exceptions=True)
            )

    async def _connect(self, address: str) -> System:
        system = System()
        await system.connect(address)
        self._system = system
        self.broker.reset()

        try:
            await self._get_gps_global_origin()
        except Exception:
            LOGGER.warning("获取 GPS 全局原点失败（可能当前飞控不支持）")

        self._telemetry_tasks = [
            asyncio.create_task(self._collect_global_position()),
            asyncio.create_task(self._collect_home_position()),
            asyncio.create_task(self._collect_local_position_velocity()),
            asyncio.create_task(self._collect_attitude_euler()),
            asyncio.create_task(self._collect_attitude_quaternion()),
        ]
        LOGGER.info("已连接到 MAVSDK：%s", address)
        return system

    async def _shutdown(self) -> None:
        for task in self._telemetry_tasks:
            task.cancel()
        if self._telemetry_tasks:
            await asyncio.gather(*self._telemetry_tasks, return_exceptions=True)
        self._telemetry_tasks.clear()

    # ------------------------------------------------------------------
    # 内部：NED 原点查询
    # ------------------------------------------------------------------
    async def _get_gps_global_origin(self):
        if self._system is None:
            raise RuntimeError("MAVSDK 尚未连接")
        origin = await self._system.telemetry.get_gps_global_origin()
        self.broker.update_gps_global_origin(origin)
        return origin

    # ------------------------------------------------------------------
    # 内部：遥测采集
    # ------------------------------------------------------------------
    async def _collect_global_position(self) -> None:
        async for position in self._system.telemetry.position():  # type: ignore[union-attr]
            self.broker.update_global_position(position)

    async def _collect_home_position(self) -> None:
        async for home in self._system.telemetry.home():  # type: ignore[union-attr]
            self.broker.update_home_position(home)

    async def _collect_local_position_velocity(self) -> None:
        async for pv in self._system.telemetry.position_velocity_ned():  # type: ignore[union-attr]
            self.broker.update_local_position_velocity(pv.position, pv.velocity)

    async def _collect_attitude_euler(self) -> None:
        async for euler in self._system.telemetry.attitude_euler():  # type: ignore[union-attr]
            self.broker.update_attitude_euler(euler)

    async def _collect_attitude_quaternion(self) -> None:
        async for quaternion in self._system.telemetry.attitude_quaternion():  # type: ignore[union-attr]
            self.broker.update_attitude_quaternion(quaternion)

    # ------------------------------------------------------------------
    # 内部：指令执行
    # ------------------------------------------------------------------
    async def _execute_command(self, command: Command) -> CommandResult:
        try:
            data = await self._dispatch(command.name, command.params)
            return CommandResult(
                id=command.id,
                name=command.name,
                success=True,
                data=data,
            )
        except Exception as exc:  # noqa: BLE001 - 错误会返回给调用方
            LOGGER.exception("指令 %s 执行失败", command.name)
            return CommandResult(
                id=command.id,
                name=command.name,
                success=False,
                error=f"{type(exc).__name__}: {exc}",
            )

    async def _dispatch(self, name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        if self._system is None:
            raise RuntimeError("MAVSDK 尚未连接")

        if name == "get_snapshot_at":
            timestamp = float(params["timestamp"])
            mode = str(params.get("mode", "interpolate"))
            snapshot = self.broker.get_snapshot_at(timestamp, mode)
            if snapshot is None:
                raise RuntimeError("没有可用于查询的遥测历史")
            return {"timestamp": timestamp, "snapshot": snapshot.as_dict()}

        if name == "get_gps_global_origin":
            origin = await self._get_gps_global_origin()
            return {
                "origin": {
                    "latitude_deg": origin.latitude_deg,
                    "longitude_deg": origin.longitude_deg,
                    "altitude_m": origin.altitude_m,
                }
            }

        action = self._system.action

        if name == "arm":
            await action.arm()
        elif name == "arm_force":
            await action.arm_force()
        elif name == "disarm":
            await action.disarm()
        elif name == "takeoff":
            await action.takeoff()
        elif name == "land":
            await action.land()
        elif name == "hold":
            await action.hold()
        elif name == "kill":
            await action.kill()
        elif name == "rtl" or name == "return_to_launch":
            await action.return_to_launch()
        elif name == "set_takeoff_altitude":
            await action.set_takeoff_altitude(float(params["altitude_m"]))
        elif name == "set_return_to_launch_altitude":
            await action.set_return_to_launch_altitude(
                float(params["relative_altitude_m"])
            )
        elif name == "set_current_speed":
            await action.set_current_speed(float(params["speed_m_s"]))
        else:  # pragma: no cover - 已被 SUPPORTED_COMMANDS 拦截
            raise ValueError(f"不支持的指令: {name}")

        return {"name": name, "params": params}
