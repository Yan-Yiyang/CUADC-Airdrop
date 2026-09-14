"""任务级控制：mission 上传/启动、hold、gripper 投放、请求 NED 原点（计划 4.1）。

为什么单独一层
--------------
:class:`~airdrop.telemetry.mavsdk_thread.MavsdkThread` 管的是"连接 + 遥测流 +
指令串行"，它不知道什么叫"任务"。本模块把 MAVSDK 的 ``mission`` / ``gripper`` /
``action`` 插件包成**任务级动作**，供 :class:`airdrop.mission.MissionRunner` 调用；
同时定义 :class:`MissionItemSpec`——它比 :class:`~airdrop.config.Waypoint` 多一个
动作字段（起飞/降落），因为固定翼起飞由侦查航线首航点的**起飞项**承担（计划 4.1）。

失败语义（刻意统一，状态机只需要处理一种异常）
--------------------------------------------
命令**没做成**一律抛 :class:`ControllerError`；``None`` / ``False`` 只表示
"读到了、但确实还没有"（原点尚未就绪、任务尚未完成）。调用方据此区分
"链路/飞控出问题"（→ ABORT）与"下一拍再查"。

⚠ 本模块**不在导入期读 config**：``airdrop.config`` → ``airdrop.video.source`` →
``airdrop.telemetry`` 这条导入链会让运行时导入 config 形成环。需要按配置装配时走
:meth:`DroneController.from_config`（它在方法体里才导入）。
"""

from __future__ import annotations

import concurrent.futures
import logging
import math
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, NamedTuple, Protocol, Sequence, runtime_checkable

from mavsdk.mission import MissionItem, MissionPlan

if TYPE_CHECKING:  # 仅类型标注：见模块 docstring 的导入顺序说明
    from ..config import Config, GripperConfig
    from .mavsdk_thread import MavsdkThread

LOGGER = logging.getLogger(__name__)

__all__ = [
    "ACTION_LAND",
    "ACTION_NONE",
    "ACTION_TAKEOFF",
    "MISSION_ACTIONS",
    "UNSET",
    "ControllerError",
    "DroneController",
    "DryRunController",
    "MissionController",
    "MissionItemSpec",
    "NedOrigin",
    "to_mission_item",
]

#: 任务项动作（映射到 MAVSDK ``MissionItem.VehicleAction``）
ACTION_NONE = "none"
ACTION_TAKEOFF = "takeoff"
ACTION_LAND = "land"
MISSION_ACTIONS = frozenset({ACTION_NONE, ACTION_TAKEOFF, ACTION_LAND})

#: MAVSDK 约定：**数值字段用 NaN 表示"不指定"**（C++ 结构体默认值即 NAN，
#: 官方 Python 示例也传 ``float('nan')``）。0 是有意义的值（速度 0、盘旋 0 秒），
#: 所以"没用到的字段"必须填 NaN——mavsdk_server 正是按 ``isnan`` 决定要不要生成
#: ``DO_CHANGE_SPEED`` / ``LOITER_TIME`` 这类附加任务项。
UNSET = float("nan")


class ControllerError(RuntimeError):
    """任务级控制失败（未连接、飞控拒绝、上传失败、超时……）。"""


class NedOrigin(NamedTuple):
    """NED 原点（``GPS_GLOBAL_ORIGIN``）。

    ⚠ 字段顺序与 :class:`airdrop.georef.LLARef` **相反**（那边是 lon 在前）——
    刻意用命名构造，别按位置传。
    """

    lat_deg: float
    lon_deg: float
    alt_m: float


@dataclass(frozen=True, slots=True)
class MissionItemSpec:
    """要上传的一个任务项。

    ``alt_m`` 是**相对起飞点的高度**（MAVSDK ``relative_altitude_m``），与
    :attr:`airdrop.config.OverflyConfig.altitude_m` 同一口径——不是海拔。

    ``None`` 表示"不指定、用飞控默认"（落到 MAVSDK 上是 NaN）：速度、盘旋时间、
    接受半径、偏航角都如此，别用 0 代替（0 是合法值，含义完全不同）。
    """

    lat: float
    lon: float
    alt_m: float
    action: str = ACTION_NONE
    acceptance_radius_m: float | None = None
    speed_m_s: float | None = None
    loiter_time_s: float | None = None
    yaw_deg: float | None = None

    def __post_init__(self) -> None:
        if self.action not in MISSION_ACTIONS:
            raise ValueError(
                f"任务项动作只支持 {sorted(MISSION_ACTIONS)}: {self.action!r}"
            )
        if not -90.0 <= float(self.lat) <= 90.0:
            raise ValueError(f"纬度超出 ±90°: {self.lat}")
        if not -180.0 <= float(self.lon) <= 180.0:
            raise ValueError(f"经度超出 ±180°: {self.lon}")
        if not math.isfinite(float(self.alt_m)):
            raise ValueError(f"相对高度必须是有限值: {self.alt_m}")
        if self.acceptance_radius_m is not None and self.acceptance_radius_m < 0:
            raise ValueError(f"接受半径不能为负: {self.acceptance_radius_m}")
        if self.speed_m_s is not None and self.speed_m_s <= 0:
            raise ValueError(f"速度必须为正: {self.speed_m_s}")
        if self.loiter_time_s is not None and self.loiter_time_s < 0:
            raise ValueError(f"盘旋时间不能为负: {self.loiter_time_s}")

    @classmethod
    def from_waypoint(
        cls,
        waypoint: Any,
        *,
        action: str = ACTION_NONE,
        **overrides: Any,
    ) -> "MissionItemSpec":
        """由 :class:`~airdrop.config.Waypoint` 构造（``acceptance_radius_m=0`` → 不指定）。

        ``waypoint`` 只要带 ``lat``/``lon``/``alt_m``/``acceptance_radius_m`` 即可
        （结构化类型，测试里用假航点也行）。
        """
        radius = float(getattr(waypoint, "acceptance_radius_m", 0.0) or 0.0)
        params: dict[str, Any] = {
            "lat": float(waypoint.lat),
            "lon": float(waypoint.lon),
            "alt_m": float(waypoint.alt_m),
            "action": action,
            "acceptance_radius_m": radius if radius > 0 else None,
        }
        params.update(overrides)
        return cls(**params)

    def as_dict(self) -> dict[str, Any]:
        return {
            "lat": round(float(self.lat), 8),
            "lon": round(float(self.lon), 8),
            "alt_m": round(float(self.alt_m), 3),
            "action": self.action,
            "acceptance_radius_m": self.acceptance_radius_m,
            "speed_m_s": self.speed_m_s,
            "loiter_time_s": self.loiter_time_s,
            "yaw_deg": self.yaw_deg,
        }


def to_mission_item(spec: MissionItemSpec) -> MissionItem:
    """任务项 → MAVSDK ``MissionItem``（纯函数，离线可测）。"""
    action = {
        ACTION_NONE: MissionItem.VehicleAction.NONE,
        ACTION_TAKEOFF: MissionItem.VehicleAction.TAKEOFF,
        ACTION_LAND: MissionItem.VehicleAction.LAND,
    }[spec.action]
    return MissionItem(
        latitude_deg=float(spec.lat),
        longitude_deg=float(spec.lon),
        relative_altitude_m=float(spec.alt_m),
        speed_m_s=UNSET if spec.speed_m_s is None else float(spec.speed_m_s),
        # 固定翼到点即过、不停；**降落项例外**——那一点是要落下去的，
        # 让它"飞掠"就等于让飞机想着过点而不是落地。
        is_fly_through=spec.action != ACTION_LAND,
        gimbal_pitch_deg=UNSET,
        gimbal_yaw_deg=UNSET,
        camera_action=MissionItem.CameraAction.NONE,
        loiter_time_s=UNSET if spec.loiter_time_s is None else float(spec.loiter_time_s),
        camera_photo_interval_s=UNSET,
        acceptance_radius_m=UNSET
        if spec.acceptance_radius_m is None
        else float(spec.acceptance_radius_m),
        yaw_deg=UNSET if spec.yaw_deg is None else float(spec.yaw_deg),
        camera_photo_distance_m=UNSET,
        vehicle_action=action,
    )


@runtime_checkable
class MissionController(Protocol):
    """状态机眼里的控制器（假控制器按这个协议实现即可，见 ``tests/test_mission.py``）。

    失败语义同 :class:`DroneController`：命令失败抛 :class:`ControllerError`，
    查询类只回"读到的事实"。
    """

    def upload_mission(self, items: Sequence[MissionItemSpec], /) -> int: ...

    def start_mission(self) -> None: ...

    def hold(self) -> None: ...

    def rtl(self) -> None: ...

    def gripper_release(self) -> bool: ...

    def request_origin(self) -> NedOrigin | None: ...

    def mission_finished(self) -> bool: ...


class DroneController:
    """MAVSDK 任务级控制（同步阻塞 API，可从任意线程调用）。

    参数
    ----
    thread:
        已连接的 :class:`MavsdkThread`（只要有 ``submit`` / ``require_drone`` 即可）。
    gripper_instance / gripper_enabled / release_settle_s:
        来自 :class:`~airdrop.config.GripperConfig`；``enabled=False`` 时
        :meth:`gripper_release` 只记日志并返回 False（**不假装投了**）。
    timeout:
        单条指令的等待上限（秒）。
    on_event:
        ``(kind, data)`` 事件回调，接 recorder 上是
        ``lambda kind, data: recorder.events.emit(kind, **data)``；抛异常不影响控制流。
    sleep:
        注入的睡眠函数（测试里换成假时钟）。
    """

    def __init__(
        self,
        thread: "MavsdkThread",
        *,
        gripper_instance: int = 0,
        gripper_enabled: bool = True,
        release_settle_s: float = 0.5,
        timeout: float = 30.0,
        on_event: Callable[[str, dict[str, Any]], None] | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._thread = thread
        self._gripper_instance = int(gripper_instance)
        self._gripper_enabled = bool(gripper_enabled)
        self._release_settle_s = max(float(release_settle_s), 0.0)
        self._timeout = float(timeout)
        self._on_event = on_event
        self._sleep = sleep
        # 上传/启动/hold 之间不能交错（协程在 await 点上会互相穿插）
        self._lock = threading.RLock()

    @classmethod
    def from_config(
        cls,
        config: "Config",
        thread: "MavsdkThread",
        *,
        on_event: Callable[[str, dict[str, Any]], None] | None = None,
        timeout: float = 30.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> "DroneController":
        """按 :class:`~airdrop.config.Config` 装配（方法体里才导入 config，见模块 docstring）。"""
        gripper: "GripperConfig" = config.gripper
        return cls(
            thread,
            gripper_instance=gripper.instance,
            gripper_enabled=gripper.enabled,
            release_settle_s=gripper.release_settle_s,
            timeout=timeout,
            on_event=on_event,
            sleep=sleep,
        )

    # ------------------------------------------------------------------
    # mission
    # ------------------------------------------------------------------
    def upload_mission(self, items: Sequence[MissionItemSpec], /) -> int:
        """上传任务（覆盖飞控上的旧任务），返回任务项数量。

        空任务**显式报错**：MAVSDK 上传空计划等于清空任务，那多半是调用方的 bug，
        不该静默发生。
        """
        specs = tuple(items)
        if not specs:
            raise ControllerError("任务项为空，拒绝上传")
        with self._lock:
            self._run(lambda: self._upload(specs), "上传任务")
        self._emit(
            "mission_upload",
            {
                "count": len(specs),
                "items": [spec.as_dict() for spec in specs],
            },
        )
        LOGGER.info("任务已上传：%d 个任务项", len(specs))
        return len(specs)

    async def _upload(self, specs: tuple[MissionItemSpec, ...]) -> None:
        drone = self._thread.require_drone()
        await drone.mission.upload_mission(
            MissionPlan([to_mission_item(spec) for spec in specs])
        )

    def start_mission(self) -> None:
        """启动任务。"""
        with self._lock:
            self._run(self._start, "启动任务")
        self._emit("mission_start", {})
        LOGGER.info("任务已启动")

    async def _start(self) -> None:
        drone = self._thread.require_drone()
        await drone.mission.start_mission()

    def mission_finished(self) -> bool:
        """任务是否已飞完（最后一个任务项已到达）。

        用 MAVSDK 的 ``mission.is_mission_finished()`` 而不是自己解析进度流：
        它是**请求/响应**，没有"等下一帧进度"的滞后；判据侧的防陈旧见
        :class:`airdrop.mission.MissionMonitor`。
        """
        return bool(self._run(self._finished, "查询任务完成状态"))

    async def _finished(self) -> bool:
        drone = self._thread.require_drone()
        return bool(await drone.mission.is_mission_finished())

    # ------------------------------------------------------------------
    # 动作
    # ------------------------------------------------------------------
    def hold(self) -> None:
        """原地盘旋（固定翼 = LOITER），用于 HOLD_PROCESS 与 ABORT。"""
        with self._lock:
            self._run(self._hold, "hold")
        self._emit("hold", {})

    async def _hold(self) -> None:
        drone = self._thread.require_drone()
        await drone.action.hold()

    def rtl(self) -> None:
        """返航（``abort_action="rtl"`` 时用）。"""
        with self._lock:
            self._run(self._rtl, "返航")
        self._emit("rtl", {})

    async def _rtl(self) -> None:
        drone = self._thread.require_drone()
        await drone.action.return_to_launch()

    def gripper_release(self) -> bool:
        """投放；返回是否**真的发出了**投放指令。

        ``enabled=False`` 时只记日志并返回 False——调用方（状态机）据此判失败。
        """
        if not self._gripper_enabled:
            LOGGER.warning(
                "gripper 未启用（GripperConfig.enabled=False），本次不投放"
            )
            self._emit("gripper_skipped", {"reason": "disabled"})
            return False
        with self._lock:
            self._run(self._release, "投放（gripper.release）")
        self._emit("gripper_release", {"instance": self._gripper_instance})
        LOGGER.warning("已发出投放指令（gripper instance=%d）", self._gripper_instance)
        if self._release_settle_s > 0:
            # 给伺服/挂架留动作时间：紧接着的下一条指令容易被飞控以 BUSY 拒绝
            self._sleep(self._release_settle_s)
        return True

    async def _release(self) -> None:
        drone = self._thread.require_drone()
        await drone.gripper.release(self._gripper_instance)

    # ------------------------------------------------------------------
    # 查询
    # ------------------------------------------------------------------
    def request_origin(self) -> NedOrigin | None:
        """实时查询 NED 原点（``GPS_GLOBAL_ORIGIN``）；尚未就绪时返回 None。

        注意这**不会**写进遥测快照——快照里的原点由 MavsdkThread 的原点流负责刷新
        （两者同源）。状态机用本方法做就绪判断，坐标换算用同一原点。
        """
        try:
            origin = self._run(self._query_origin, "查询 NED 原点")
        except ControllerError as exc:
            LOGGER.info("NED 原点尚不可用：%s", exc)
            return None
        return NedOrigin(
            lat_deg=float(origin.latitude_deg),
            lon_deg=float(origin.longitude_deg),
            alt_m=float(origin.altitude_m),
        )

    async def _query_origin(self) -> Any:
        drone = self._thread.require_drone()
        return await drone.telemetry.get_gps_global_origin()

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------
    def _run(self, factory: Callable[[], Any], what: str) -> Any:
        """把协程投到 MAVSDK 线程并等结果；任何失败翻译成 :class:`ControllerError`。"""
        try:
            future = self._thread.submit(factory())
        except Exception as exc:  # noqa: BLE001 - 线程/事件循环不可用
            raise ControllerError(f"{what}：无法提交到 MAVSDK 线程（{exc}）") from exc
        try:
            return future.result(timeout=self._timeout)
        except concurrent.futures.TimeoutError as exc:
            raise ControllerError(f"{what}：{self._timeout:.0f}s 内没有结果") from exc
        except Exception as exc:  # noqa: BLE001 - MAVSDK 异常统一成 ControllerError
            raise ControllerError(f"{what}：{exc}") from exc

    def _emit(self, kind: str, data: dict[str, Any]) -> None:
        if self._on_event is None:
            return
        try:
            self._on_event(kind, data)
        except Exception:  # noqa: BLE001 - 事件写盘失败不影响控制
            LOGGER.exception("写入控制事件失败（控制继续）")


class DryRunController:
    """演练用控制器：航线/hold/rtl 照真下发，**只有投放**换成日志（返回 True）。

    给 SITL 与实机"不挂弹"演练用——判据、状态机、事件日志全走真实路径，唯一被替换的
    是"开仓"那一步。这也是 SITL 场景能跑起来的前提：SITL 里没有 gripper 硬件，
    真发投放指令会被飞控拒绝，而状态机把"命令失败"当任务失败（进 ``ABORT``）。

    ⚠ 它**不是**安全措施：``gripper_release()`` 返回 True 表示"演练里算投了"，
    真机上装弹时千万别用它。
    """

    def __init__(
        self,
        inner: MissionController,
        *,
        on_event: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> None:
        self._inner = inner
        self._on_event = on_event
        self.releases = 0

    # ---- 全部委托给真控制器 ------------------------------------------
    def upload_mission(self, items: Sequence[MissionItemSpec], /) -> int:
        return self._inner.upload_mission(items)

    def start_mission(self) -> None:
        self._inner.start_mission()

    def hold(self) -> None:
        self._inner.hold()

    def rtl(self) -> None:
        self._inner.rtl()

    def request_origin(self) -> NedOrigin | None:
        return self._inner.request_origin()

    def mission_finished(self) -> bool:
        return self._inner.mission_finished()

    # ---- 唯一被替换的一步 --------------------------------------------
    def gripper_release(self) -> bool:
        self.releases += 1
        LOGGER.warning(
            "演练模式：拦下投放指令（第 %d 次），只记事件不下发——真机装弹时别用 DryRunController",
            self.releases,
        )
        if self._on_event is not None:
            try:
                self._on_event("drop_dry_run", {"count": self.releases})
            except Exception:  # noqa: BLE001
                LOGGER.exception("写入演练事件失败（控制继续）")
        return True
