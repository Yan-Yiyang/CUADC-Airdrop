"""专用的 MAVSDK 工作线程。

MAVSDK 基于 asyncio/gRPC，官方文档推荐的写法是：``System()`` →
``await drone.connect(system_address=...)`` → 等待 ``core.connection_state()``
上线 → 为每条 ``telemetry`` 数据流创建一个 ``asyncio`` 任务并行采集。

本模块把这套流程原样搬进专用后台线程的事件循环，对外只暴露同步、线程安全的
API，避免 MAVSDK 的协程与应用中其他事件循环互相干扰。与官方示例唯一的差别：
官方示例让采集任务永远运行；这里把任一遥测流的结束视为连接失效，交给监督
循环按配置自动重连（结构化并发由 ``asyncio.TaskGroup`` 负责）。
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import logging
import threading
from collections.abc import AsyncIterable, Coroutine
from typing import TYPE_CHECKING, Any, ClassVar, cast

from mavsdk import System
from mavsdk.action import ActionError
from mavsdk.telemetry import GpsGlobalOrigin, TelemetryError

from .broker import SUPPORTED_QUERY_MODES, TelemetryBroker
from .models import Command, CommandResult, TelemetrySnapshot

if TYPE_CHECKING:  # 仅类型标注用，避免 telemetry → config 的运行时依赖
    from ..config import TelemetryConfig

LOGGER = logging.getLogger(__name__)


def _error_code(exc: BaseException) -> str | None:
    """从 MAVSDK 异常中提取错误码名称（如 ``COMMAND_DENIED``）。"""
    result = getattr(exc, "_result", None)
    code = getattr(result, "result", None)
    if code is None:
        return None
    return getattr(code, "name", None) or str(code)


class MavsdkThread:
    """在专用后台线程中运行一个 MAVSDK :class:`System`。

    公开方法都是同步且线程安全的：它们把协程调度到 MAVSDK 事件循环上执行，
    调用方以普通阻塞方式等待结果，任何 MAVSDK 协程都不会在调用方线程中被
    await。

    参数
    ----
    broker:
        遥测代理，缺省时自动创建。
    system_address:
        MAVSDK 连接地址，推荐 ``udpin://0.0.0.0:14540``。
    connect_timeout:
        ``System.connect()`` 与等待连接状态各自的超时。
    wait_for_health:
        连接后是否等待 ``telemetry.health()`` 就绪。
    health_timeout:
        等待健康检查的超时。
    require_health_for_actions:
        是否在 ``arm`` / ``takeoff`` 前强制等待健康检查（默认关闭）。
    origin_refresh_interval:
        NED 原点后台刷新周期（秒）。
    reconnect:
        遥测流异常/结束时是否自动重连。
    reconnect_delay:
        重连前的等待时间（秒）。
    """

    #: 允许经 `submit_command` 下发的指令名（ClassVar：类级常量，不是每个实例一份）
    SUPPORTED_COMMANDS: ClassVar[set[str]] = {
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

    def __init__(  # noqa: PLR0913 - 连接/速率/看门狗参数多，但都有默认值且按名传
        self,
        broker: TelemetryBroker | None = None,
        system_address: str = "udpin://0.0.0.0:14540",
        *,
        connect_timeout: float = 30.0,
        wait_for_health: bool = False,
        health_timeout: float = 30.0,
        require_health_for_actions: bool = False,
        origin_refresh_interval: float = 5.0,
        reconnect: bool = True,
        reconnect_delay: float = 5.0,
        position_rate_hz: float | None = None,
        position_velocity_ned_rate_hz: float | None = None,
        attitude_rate_hz: float | None = None,
        thread_name: str = "mavsdk-thread",
    ) -> None:
        self.broker = broker or TelemetryBroker()
        self.system_address = system_address
        self.connect_timeout = connect_timeout
        self.wait_for_health = wait_for_health
        self.health_timeout = health_timeout
        self.require_health_for_actions = require_health_for_actions
        self.origin_refresh_interval = origin_refresh_interval
        self.reconnect = reconnect
        self.reconnect_delay = reconnect_delay
        # 遥测速率（None = 不下发，沿用飞控默认）。速率决定帧-遥测内插的精度上限：
        # 位置 1Hz 时两个采样点相隔十几米，内插等于猜。
        self.position_rate_hz = position_rate_hz
        self.position_velocity_ned_rate_hz = position_velocity_ned_rate_hz
        self.attitude_rate_hz = attitude_rate_hz
        self._thread_name = thread_name

        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._drone: System | None = None
        self._supervisor: concurrent.futures.Future[Any] | None = None
        self._supervisor_lock = threading.Lock()
        self._started = threading.Event()
        self._ready = threading.Event()
        self._stop_requested = threading.Event()

    # ------------------------------------------------------------------
    # 配置入口
    # ------------------------------------------------------------------
    @classmethod
    def from_config(
        cls,
        config: "TelemetryConfig",
        *,
        broker: TelemetryBroker | None = None,
        thread_name: str = "mavsdk-thread",
    ) -> "MavsdkThread":
        """按 :class:`airdrop.config.TelemetryConfig` 装配（含遥测速率下发）。

        把"配置 + 构造"合成一步，调用方只需要拿到配置、塞进来即可。
        """
        return cls(
            broker=broker,
            system_address=config.system_address,
            connect_timeout=config.connect_timeout,
            wait_for_health=config.wait_for_health,
            health_timeout=config.health_timeout,
            require_health_for_actions=config.require_health_for_actions,
            origin_refresh_interval=config.origin_refresh_interval,
            reconnect=config.reconnect,
            reconnect_delay=config.reconnect_delay,
            position_rate_hz=config.position_rate_hz,
            position_velocity_ned_rate_hz=config.position_velocity_ned_rate_hz,
            attitude_rate_hz=config.attitude_rate_hz,
            thread_name=thread_name,
        )

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    def start(self, timeout: float = 5.0) -> None:
        """启动专用的事件循环线程（幂等）。"""
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
        """停止连接管理、遥测采集和事件循环，并等待工作线程退出。"""
        self._stop_requested.set()

        supervisor, self._supervisor = self._supervisor, None
        if supervisor is not None:
            supervisor.cancel()
            # 取消/异常都视为已停止（BLE001 是刻意的：这里不该因为收尾失败而抛出）
            with contextlib.suppress(Exception):
                supervisor.result(timeout=timeout)

        if self._loop is not None and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)

        thread = self._thread
        if thread is not None:
            thread.join(timeout=timeout)
            if thread.is_alive():
                # 线程没停就别清引用（清了 start() 会再起一个，且 _loop 正被它用着）。
                # 保留 _loop/_thread，稍后可再调 stop()；其余收尾照常做。
                LOGGER.warning("MAVSDK 线程未在 %.1fs 内退出，保留引用（可再调 stop()）", timeout)
            else:
                self._thread = None
                self._loop = None

        drone, self._drone = self._drone, None
        self._release_drone(drone)  # 幂等；正常路径上会话 finally 已释放过
        self._ready.clear()
        self._started.clear()

    @property
    def connected(self) -> bool:
        """飞控是否已连接并就绪。"""
        return self._ready.is_set()

    def connect(
        self,
        system_address: str | None = None,
        timeout: float | None = None,
    ) -> System:
        """连接飞机并等待就绪（同步方法，可从任意线程调用）。

        流程遵循官方推荐：``System.connect`` → 等待 ``connection_state`` →
        可选等待 ``health`` → 查询 NED 原点 → 并行采集遥测。
        返回已就绪的 :class:`System`。
        """
        self.start()
        with self._supervisor_lock:
            if self._supervisor is None or self._supervisor.done():
                self._stop_requested.clear()
                self._ready.clear()
                self._supervisor = self.submit(self._serve(system_address or self.system_address))

        wait = (
            timeout if timeout is not None else (self.connect_timeout + self.health_timeout + 15.0)
        )
        if not self._ready.wait(wait):
            raise TimeoutError(f"在 {wait:.0f}s 内未能连接飞机")
        return self.require_drone()

    # ------------------------------------------------------------------
    # 遥测 / 指令接口
    # ------------------------------------------------------------------
    def get_snapshot(self) -> TelemetrySnapshot:
        """返回最新遥测快照（线程安全的快捷方式）。"""
        return self.broker.get_snapshot()

    def get_snapshot_at(
        self,
        timestamp: float,
        mode: str = "interpolate",
    ) -> TelemetrySnapshot | None:
        """按时间戳查询历史快照，并进行内插或外推（线程安全的快捷方式）。"""
        return self.broker.get_snapshot_at(timestamp, mode)

    def get_gps_global_origin(self, timeout: float = 10.0) -> GpsGlobalOrigin:
        """实时查询本地 NED 坐标系原点（GPS_GLOBAL_ORIGIN）并写入快照。"""
        return self.submit(self._query_origin()).result(timeout=timeout)

    def send_command(
        self,
        command: Command | dict[str, Any],
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
        except Exception as exc:  # noqa: BLE001 - MAVSDK 异常统一成 CommandResult 失败
            return CommandResult(
                id=command.id,
                name=command.name,
                success=False,
                error=f"{type(exc).__name__}: {exc}",
            )

    def submit(self, coro: Coroutine[Any, Any, Any]) -> concurrent.futures.Future[Any]:
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
            self._cancel_pending_tasks()
            if self._loop is not None:
                self._loop.close()
                self._loop = None

    def _cancel_pending_tasks(self) -> None:
        if self._loop is None or self._loop.is_closed():
            return
        pending = asyncio.all_tasks(self._loop)
        for task in pending:
            task.cancel()
        if pending:
            self._loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))

    # ------------------------------------------------------------------
    # 内部：监督循环（连接 → 采集 → 断开 → 按配置重连）
    # ------------------------------------------------------------------
    async def _serve(self, address: str) -> None:
        while not self._stop_requested.is_set():
            try:
                await self._run_session(address)
            except asyncio.CancelledError:
                raise
            except Exception:
                LOGGER.exception("MAVSDK 连接/运行失败")

            if self._stop_requested.is_set() or not self.reconnect:
                break

            LOGGER.info("%.1fs 后尝试重连 ...", self.reconnect_delay)
            await asyncio.sleep(self.reconnect_delay)

    async def _run_session(self, address: str) -> None:
        """一次完整会话；正常返回或抛出都意味着本次连接结束。

        无论会话以何种方式结束（连接失败、遥测中断、被取消），都在
        finally 里显式终止 mavsdk_server 子进程——不能只把引用置 None
        然后等待 ``System.__del__``，否则旧子进程可能存活到下一次
        连接，撞占固定的 gRPC 端口（默认 50051）。
        """
        drone: System | None = None
        try:
            drone = System()
            await asyncio.wait_for(
                drone.connect(system_address=address),
                timeout=self.connect_timeout,
            )
            self._drone = drone
            self.broker.reset()

            await self._wait_connected(drone)
            LOGGER.info("飞控已连接：%s", address)

            if self.wait_for_health:
                await self._wait_healthy(drone)
                LOGGER.info("飞控健康检查通过")

            # 下发遥测速率。位置/姿态速率直接决定帧-遥测内插的精度上限：
            # 固定翼 15~20 m/s 平飞时，位置 1Hz 的内插误差是米级，10Hz 才到
            # 分米级；姿态 1° 误差在 20m 高度 ≈0.35m 地面误差，所以姿态 30Hz 起步。
            await self._apply_telemetry_rates(drone)

            # 连接就绪后先取一次 NED 原点，之后由采集任务周期刷新。
            await self._publish_origin(drone)

            self._ready.set()
            LOGGER.info("遥测采集已启动")
            await self._collect_telemetry(drone)
        finally:
            self._ready.clear()
            self._drone = None
            self._release_drone(drone)

    @staticmethod
    def _release_drone(drone: System | None) -> None:
        """显式终止 mavsdk_server 子进程，而不是被动等待 ``System.__del__``。

        MAVSDK-Python 没有公开的 ``close()``：``__del__`` 调用的就是这个
        内部方法，且可以重复调用（子进程未启动时是空操作）。显式调用后
        即使别处还残留引用（在途指令协程、调用方持有的旧 System），
        子进程也会立即退出。旧版本 mavsdk 没有该方法时静默跳过，退回
        引用计数/GC 回收路径。
        """
        if drone is None:
            return
        stop_server = getattr(drone, "_stop_mavsdk_server", None)
        if callable(stop_server):
            try:
                stop_server()
            except Exception:
                LOGGER.exception("终止 mavsdk_server 子进程失败")

    async def _apply_telemetry_rates(self, drone: System) -> None:
        """下发遥测速率（连接就绪后调用一次）。

        速率决定 :mod:`airdrop.video.align` 内插的精度上限。
        单个速率失败只记日志、不中断连接（飞控/固件可能不支持某些流）；
        ``None``/非正数表示不下发，沿用飞控默认。
        """
        wanted = (
            ("position", self.position_rate_hz, drone.telemetry.set_rate_position),
            (
                "position_velocity_ned",
                self.position_velocity_ned_rate_hz,
                drone.telemetry.set_rate_position_velocity_ned,
            ),
            ("attitude_euler", self.attitude_rate_hz, drone.telemetry.set_rate_attitude_euler),
            (
                "attitude_quaternion",
                self.attitude_rate_hz,
                drone.telemetry.set_rate_attitude_quaternion,
            ),
        )
        for name, rate, setter in wanted:
            if rate is None or rate <= 0:
                continue
            try:
                await setter(rate)
                LOGGER.info("遥测速率已设置：%s = %.1f Hz", name, rate)
            except Exception as exc:  # noqa: BLE001 - 单条速率设置失败不阻塞其它流
                LOGGER.warning("设置遥测速率失败（%s=%.1fHz）：%s", name, rate, exc)

    async def _wait_connected(self, drone: System) -> None:
        async def wait() -> None:
            async for state in cast(AsyncIterable[Any], drone.core.connection_state()):
                if state.is_connected:
                    return

        try:
            await asyncio.wait_for(wait(), timeout=self.connect_timeout)
        except TimeoutError:
            # from None：这是"等超时"的主动重述，底层 TimeoutError 只是实现细节
            raise TimeoutError("等待飞控连接状态超时") from None

    async def _wait_healthy(self, drone: System) -> None:
        async def wait() -> None:
            async for health in cast(AsyncIterable[Any], drone.telemetry.health()):
                if health.is_global_position_ok and health.is_home_position_ok:
                    return

        try:
            await asyncio.wait_for(wait(), timeout=self.health_timeout)
        except TimeoutError:
            raise TimeoutError(f"飞控健康检查未在 {self.health_timeout:.0f}s 内通过") from None

    # ------------------------------------------------------------------
    # 内部：遥测采集（官方示例写法：每条数据流一个任务）
    # ------------------------------------------------------------------
    async def _collect_telemetry(self, drone: System) -> None:
        """并行订阅各遥测流并写入代理。

        任一数据流结束（关闭或异常）都会让 ``TaskGroup`` 取消其余任务并抛出
        ExceptionGroup，这里记录后正常返回，由监督循环决定重连。

        ⚠ **例外：风估计是"可选流"**（:meth:`_optional_stream`）——它结束或报错只记日志，
        不影响会话。理由：风只影响弹道精度（没有它按零风降级），而飞控未必支持/未必
        开启风估计；让一条增强流把整条遥测链路拖进重连循环是不划算的。
        """
        try:
            async with asyncio.TaskGroup() as tg:
                tg.create_task(self._guarded("position", self._stream_position(drone)))
                tg.create_task(self._guarded("home", self._stream_home(drone)))
                tg.create_task(
                    self._guarded(
                        "position_velocity_ned",
                        self._stream_position_velocity_ned(drone),
                    )
                )
                tg.create_task(self._guarded("attitude_euler", self._stream_attitude_euler(drone)))
                tg.create_task(
                    self._guarded(
                        "attitude_quaternion",
                        self._stream_attitude_quaternion(drone),
                    )
                )
                tg.create_task(self._guarded("gps_global_origin", self._refresh_origin_loop(drone)))
                tg.create_task(self._optional_stream("wind", self._stream_wind(drone)))
                tg.create_task(
                    self._optional_stream("mission_progress", self._stream_mission_progress(drone))
                )
                tg.create_task(
                    self._optional_stream("flight_mode", self._stream_flight_mode(drone))
                )
                tg.create_task(self._optional_stream("in_air", self._stream_in_air(drone)))
        except* Exception as group:  # noqa: BLE001 - 任务组里任一核心流断掉即视为会话结束
            LOGGER.warning("遥测采集中断：%s", group.exceptions[0])

    @staticmethod
    async def _guarded(name: str, stream: Coroutine[Any, Any, None]) -> None:
        """遥测流一旦结束（关闭或异常）就抛出，让 TaskGroup 取消其余任务。"""
        try:
            await stream
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise RuntimeError(f"遥测流 {name} 异常结束: {exc}") from exc
        raise RuntimeError(f"遥测流 {name} 已关闭")

    @staticmethod
    async def _optional_stream(name: str, stream: Coroutine[Any, Any, None]) -> None:
        """**可选**遥测流：结束或异常只记日志，不影响会话（见 :meth:`_run_session`）。"""
        try:
            await stream
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - 可选流失败不是会话失败（见 AGENTS.md 第 3 条）
            LOGGER.info("可选遥测流 %s 不可用（%s），相关功能降级", name, exc)
            return
        LOGGER.info("可选遥测流 %s 已结束，相关功能降级", name)

    async def _stream_wind(self, drone: System) -> None:
        async for wind in cast(AsyncIterable[Any], drone.telemetry.wind()):
            self.broker.update_wind(wind)

    async def _stream_mission_progress(self, drone: System) -> None:
        """任务进度（PX4 ``MISSION_CURRENT``）→ 快照。

        用 ``mission_raw`` 的进度流：任务项走 raw 上传，``mission`` 插件的
        ``is_mission_finished()`` 只知道它自己上传过的任务，走 raw 之后永远回 False
        （实测）。状态机读快照里的 ``mission_current/mission_total`` 判断飞完。
        """
        async for progress in cast(AsyncIterable[Any], drone.mission_raw.mission_progress()):
            self.broker.update_mission_progress(int(progress.current), int(progress.total))

    async def _stream_flight_mode(self, drone: System) -> None:
        """飞行模式 → 快照（状态机用它确认"任务真的启动了"，见 models.py）。"""
        async for mode in cast(AsyncIterable[Any], drone.telemetry.flight_mode()):
            self.broker.update_flight_mode(mode)

    async def _stream_in_air(self, drone: System) -> None:
        """是否在空中 → 快照（状态机的"等起飞"门用它）。"""
        async for in_air in cast(AsyncIterable[Any], drone.telemetry.in_air()):
            self.broker.update_in_air(in_air)

    async def _stream_position(self, drone: System) -> None:
        async for position in cast(AsyncIterable[Any], drone.telemetry.position()):
            self.broker.update_global_position(position)

    async def _stream_home(self, drone: System) -> None:
        async for home in cast(AsyncIterable[Any], drone.telemetry.home()):
            self.broker.update_home_position(home)

    async def _stream_position_velocity_ned(self, drone: System) -> None:
        async for pv in cast(AsyncIterable[Any], drone.telemetry.position_velocity_ned()):
            self.broker.update_local_position_velocity(pv.position, pv.velocity)

    async def _stream_attitude_euler(self, drone: System) -> None:
        async for euler in cast(AsyncIterable[Any], drone.telemetry.attitude_euler()):
            self.broker.update_attitude_euler(euler)

    async def _stream_attitude_quaternion(self, drone: System) -> None:
        async for quaternion in cast(AsyncIterable[Any], drone.telemetry.attitude_quaternion()):
            self.broker.update_attitude_quaternion(quaternion)

    # ------------------------------------------------------------------
    # 内部：NED 原点（GPS_GLOBAL_ORIGIN）
    # ------------------------------------------------------------------
    async def _publish_origin(self, drone: System) -> bool:
        """查询 NED 原点并写入遥测代理；失败时返回 False（保留旧值）。"""
        try:
            origin = await drone.telemetry.get_gps_global_origin()
        except Exception as exc:  # noqa: BLE001 - 原点可能尚未就绪，返回 False 保留旧值
            LOGGER.debug("获取 GPS 全局原点失败：%s", exc)
            return False
        self.broker.update_gps_global_origin(origin)
        return True

    async def _refresh_origin_loop(self, drone: System) -> None:
        interval = max(self.origin_refresh_interval, 0.1)
        while True:
            await asyncio.sleep(interval)
            await self._publish_origin(drone)

    async def _query_origin(self) -> GpsGlobalOrigin:
        drone = self.require_drone()
        try:
            origin = await drone.telemetry.get_gps_global_origin()
        except Exception as exc:
            raise RuntimeError("无法获取 NED 原点（GPS_GLOBAL_ORIGIN）") from exc
        self.broker.update_gps_global_origin(origin)
        return origin

    # ------------------------------------------------------------------
    # 内部：指令执行
    # ------------------------------------------------------------------
    def require_drone(self) -> System:
        """返回已连接的 :class:`System`；未连接时抛出。

        公开给任务级模块用（:class:`~airdrop.telemetry.controller.DroneController`
        要拿它去调 mission/gripper 插件）。**只能在 MAVSDK 线程的事件循环上调用**
        ——跨线程拿到的 System 引用在会话结束后就失效了。
        """
        drone = self._drone
        if drone is None:
            raise RuntimeError("MAVSDK 尚未连接")
        return drone

    async def _execute_command(self, command: Command) -> CommandResult:
        try:
            data = await self._dispatch(command.name, command.params)
            return CommandResult(
                id=command.id,
                name=command.name,
                success=True,
                data=data,
            )
        except (ActionError, TelemetryError) as exc:
            LOGGER.warning("指令 %s 被拒绝：%s", command.name, exc)
            return CommandResult(
                id=command.id,
                name=command.name,
                success=False,
                error=str(exc),
                data={"result": _error_code(exc)},
            )
        except Exception as exc:
            LOGGER.exception("指令 %s 执行失败", command.name)
            return CommandResult(
                id=command.id,
                name=command.name,
                success=False,
                error=f"{type(exc).__name__}: {exc}",
            )

    async def _dispatch(self, name: str, params: dict[str, Any]) -> dict[str, Any]:
        drone = self.require_drone()

        if name == "get_snapshot_at":
            return self._query_snapshot_at(params)
        if name == "get_gps_global_origin":
            return await self._origin_info(drone)

        if name in ("arm", "takeoff") and self.require_health_for_actions:
            await self._wait_healthy(drone)

        await self._run_action(drone, name, params)
        return {"name": name, "params": params}

    def _query_snapshot_at(self, params: dict[str, Any]) -> dict[str, Any]:
        timestamp = float(params["timestamp"])
        mode = str(params.get("mode", "interpolate"))
        if mode not in SUPPORTED_QUERY_MODES:
            raise ValueError(f"不支持的查询模式: {mode}，可选 {sorted(SUPPORTED_QUERY_MODES)}")
        snapshot = self.broker.get_snapshot_at(timestamp, mode)
        if snapshot is None:
            raise RuntimeError("没有可用于查询的遥测历史")
        return {"timestamp": timestamp, "snapshot": snapshot.as_dict()}

    async def _origin_info(self, drone: System) -> dict[str, Any]:
        await self._publish_origin(drone)  # 失败时回退到快照中的旧值
        snapshot = self.broker.get_snapshot()
        if snapshot.origin_latitude_deg is None:
            raise RuntimeError("无法获取 NED 原点（GPS_GLOBAL_ORIGIN）")
        return {
            "origin": {
                "latitude_deg": snapshot.origin_latitude_deg,
                "longitude_deg": snapshot.origin_longitude_deg,
                "altitude_m": snapshot.origin_altitude_m,
            }
        }

    @staticmethod
    async def _run_action(drone: System, name: str, params: dict[str, Any]) -> None:
        action = drone.action
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
        elif name in ("rtl", "return_to_launch"):
            await action.return_to_launch()
        elif name == "set_takeoff_altitude":
            await action.set_takeoff_altitude(float(params["altitude_m"]))
        elif name == "set_return_to_launch_altitude":
            await action.set_return_to_launch_altitude(float(params["relative_altitude_m"]))
        elif name == "set_current_speed":
            await action.set_current_speed(float(params["speed_m_s"]))
        else:  # 已被 SUPPORTED_COMMANDS 拦截
            raise ValueError(f"不支持的指令: {name}")
