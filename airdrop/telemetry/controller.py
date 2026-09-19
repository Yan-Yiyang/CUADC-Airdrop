"""任务级控制：mission 上传/启动、hold、gripper 投放、请求 NED 原点（计划 4.1）。

为什么单独一层
--------------
:class:`~airdrop.telemetry.mavsdk_thread.MavsdkThread` 管的是"连接 + 遥测流 +
指令串行"，它不知道什么叫"任务"。本模块把 MAVSDK 的 ``mission_raw`` / ``gripper`` /
``action`` 插件包成**任务级动作**，供 :class:`airdrop.mission.MissionRunner` 调用。

任务项用 :class:`airdrop.mission.items.MissionItem`（MAVLink 级：command/frame/
params/位置），**上传一律走 ``mission_raw``**——``mission`` 插件那层翻译会把
``vehicle_action=LAND`` 拆成两项、让 PX4 固定翼把整条任务判为不可行（详见
:mod:`airdrop.mission.items` 的说明）。

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
import threading
import time
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, NamedTuple, Protocol, runtime_checkable

from mavsdk.mission_raw import MissionItem as RawMissionItem

if TYPE_CHECKING:  # 仅类型标注：见模块 docstring 的导入顺序说明
    from ..config import Config, GripperConfig
    from ..mission.items import MissionItem
    from .mavsdk_thread import MavsdkThread

LOGGER = logging.getLogger(__name__)

__all__ = [
    "MISSION_TYPE_MISSION",
    "ControllerError",
    "DroneController",
    "DryRunController",
    "MissionController",
    "NedOrigin",
    "to_raw_item",
]

#: MAVLink ``mission_type``：0 = 普通任务（fence / rally 本包不用）。
MISSION_TYPE_MISSION = 0


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


def to_raw_item(item: MissionItem, index: int) -> RawMissionItem:
    """任务项 → MAVSDK **raw** 项（``mission_raw`` 通道用；纯函数，离线可测）。

    为什么走 raw
    ------------
    ``mission`` 插件会做一层不透明翻译：``vehicle_action=LAND`` 的一项被拆成
    "同坐标航点 + ``NAV_LAND``"，PX4 固定翼的降落判据随即把**整条任务**判为不可行，
    飞机原地盘旋而 ``start_mission()`` 仍回成功。raw 通道是原样投递，命令与参数
    怎么写就怎么到飞控（详见 :mod:`airdrop.mission.items`）。

    ⚠ ``current`` 只能给第 0 项置 1：MAVSDK 会校验，全 0 直接报 ``CURRENT_INVALID``
    （实测），全 1 则含义错乱。⚠ ``autocontinue`` 在 protobuf 里是 **int**，
    传 bool 会被 MAVSDK 直接拒（``Expected an int, got a boolean``，实测）。
    """
    return RawMissionItem(
        seq=int(index),
        frame=int(item.frame),
        command=int(item.command),
        current=1 if index == 0 else 0,
        autocontinue=1 if item.autocontinue else 0,
        param1=float(item.param1),
        param2=float(item.param2),
        param3=float(item.param3),
        param4=float(item.param4),
        x=int(round(float(item.lat) * 1e7)),
        y=int(round(float(item.lon) * 1e7)),
        z=float(item.alt_m),
        mission_type=MISSION_TYPE_MISSION,
    )


@runtime_checkable
class MissionController(Protocol):
    """状态机眼里的控制器（假控制器按这个协议实现即可，见 ``tests/test_mission.py``）。

    失败语义同 :class:`DroneController`：命令失败抛 :class:`ControllerError`，
    查询类只回"读到的事实"。
    """

    def upload_mission(self, items: Sequence[MissionItem], /) -> int: ...

    def start_mission(self) -> None: ...

    def in_mission_mode(self) -> bool: ...

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
    def upload_mission(self, items: Sequence[MissionItem], /) -> int:
        """上传任务（覆盖飞控上的旧任务），返回任务项数量。

        两处刻意的显式失败：

        * 空任务**直接报错**——上传空计划等于清空飞控上的任务，那多半是调用方的 bug；
        * 上传后**回读校验**（命令与位置逐项比对）——飞控存下来的不是我们发的那条，
          必须在"以为任务已经在飞"之前暴露出来。
        """
        specs = tuple(items)
        if not specs:
            raise ControllerError("任务项为空，拒绝上传")
        with self._lock:
            self._run(lambda: self._upload(specs), "上传任务")
            self._verify_upload(specs)
        self._emit(
            "mission_upload",
            {
                "count": len(specs),
                "items": [item.as_dict() for item in specs],
            },
        )
        LOGGER.info("任务已上传并回读校验通过：%d 个任务项", len(specs))
        return len(specs)

    async def _upload(self, specs: tuple[MissionItem, ...]) -> None:
        drone = self._thread.require_drone()
        # mission_raw：原样投递。走 mission 插件会被它把 vehicle_action=LAND
        # 拆成两项，PX4 固定翼随即拒掉整条任务（见 to_raw_item 的说明）。
        await drone.mission_raw.upload_mission(
            [to_raw_item(item, index) for index, item in enumerate(specs)]
        )

    async def _download(self) -> list[RawMissionItem]:
        drone = self._thread.require_drone()
        return list(await drone.mission_raw.download_mission())

    def _verify_upload(self, specs: tuple[MissionItem, ...]) -> None:
        """回读飞控上的任务，逐项比对命令与位置；不一致抛 :class:`ControllerError`。"""
        from ..mission.items import command_name  # 惰性导入：避免 telemetry ↔ mission 成环

        stored = self._run(self._download, "回读任务")
        if len(stored) != len(specs):
            raise ControllerError(
                f"上传后回读：飞控上有 {len(stored)} 项，我们发了 {len(specs)} 项"
            )
        problems: list[str] = []
        for index, (sent, got) in enumerate(zip(specs, stored, strict=True)):
            if int(got.command) != int(sent.command):
                problems.append(
                    f"第 {index} 项命令 {sent.command}({command_name(sent.command)}) "
                    f"→ 飞控 {got.command}({command_name(got.command)})"
                )
                continue
            if not sent.is_positional:
                continue
            lat, lon, alt = got.x / 1e7, got.y / 1e7, float(got.z)
            if (
                abs(lat - float(sent.lat)) > 2e-7
                or abs(lon - float(sent.lon)) > 2e-7
                or abs(alt - float(sent.alt_m)) > 0.01
            ):
                problems.append(
                    f"第 {index} 项位置不符：发 ({sent.lat:.7f}, {sent.lon:.7f}, {sent.alt_m:.2f})、"
                    f"回读 ({lat:.7f}, {lon:.7f}, {alt:.2f})"
                )
        if problems:
            raise ControllerError("上传后回读不一致：" + "；".join(problems))

    def start_mission(self) -> None:
        """启动任务：**先把当前任务项复位到 0**，再下发启动。

        为什么要复位
        ------------
        PX4 会把"任务已飞完"的状态锁存；当新上传的任务与上一条**内容相同**
        （``mission_id`` 是同一个 CRC）时飞控不会自动复位它，于是
        ``mission.start_mission()`` 回成功、模式却停在 ``HOLD``，导航器随后打印
        ``No valid mission available, loitering``。2026-09 SITL 对照实验（同一飞控、
        同一会话）：

        ============ ========== ===================================
        上传后模式    直接启动    先 ``set_current_mission_item(0)`` 再启动
        ============ ========== ===================================
        ``HOLD``     ✗ 停在 HOLD ✓ 进入 ``MISSION``
        ``MISSION``  ✓          ✓
        ============ ========== ===================================

        ``MAV_CMD_DO_SET_MISSION_CURRENT``（"把当前项设成第 0 项"）正好清掉那个锁存，
        所以每次都先复位再启动——实测复位那一路 100% 进 ``MISSION``。
        启动后是否**真的**进了任务模式由状态机用 :meth:`in_mission_mode` 确认。
        """
        with self._lock:
            self._run(self._rewind, "复位任务项")
            self._run(self._start, "启动任务")
        self._emit("mission_start", {})
        LOGGER.info("任务已启动（已复位到第 0 项）")

    async def _rewind(self) -> None:
        drone = self._thread.require_drone()
        await drone.mission_raw.set_current_mission_item(0)

    async def _start(self) -> None:
        drone = self._thread.require_drone()
        await drone.mission.start_mission()

    def in_mission_mode(self) -> bool:
        """飞控是否**真的**在任务模式（``FlightMode.MISSION``）。

        与 :meth:`mission_finished` 一样读快照（``flight_mode`` 可选流）。
        ⚠ 不能拿"``start_mission()`` 没抛异常"当证据：飞控可能拒绝模式切换却仍然回
        ACK，飞机继续盘旋（实测症状：状态机等到状态超时才失败）。
        """
        snapshot = self._thread.get_snapshot()
        if snapshot.flight_mode is None:
            raise ControllerError("还没有飞行模式（flight_mode 流未就绪）")
        return snapshot.flight_mode == "MISSION"

    def mission_finished(self) -> bool:
        """任务是否已飞完（最后一项已到达：``mission_current == mission_total``）。

        为什么**不用** MAVSDK 的 ``mission.is_mission_finished()``：它依赖 ``mission``
        插件自己上传过的任务（``last_upload``），而我们的任务项一律走 ``mission_raw``
        ——走 raw 之后插件对任务一无所知：飞控已经 ``Mission finished, loitering`` 了，
        它仍然回 ``False``（2026-09 SITL 实测，状态机因此卡在 RECON 直到超时）。

        进度来自 :class:`~airdrop.telemetry.mavsdk_thread.MavsdkThread` 的
        ``mission_raw.mission_progress()`` 可选流（写进快照的
        ``mission_current/mission_total``），所以这里是**读快照**，不是每拍一次 RPC。
        进度流不可用时抛 :class:`ControllerError`——调用方按"查询失败"处理
        （:meth:`~airdrop.mission.MissionRunner._query` 只记一次日志），最终由状态超时兜底。
        """
        snapshot = self._thread.get_snapshot()
        current, total = snapshot.mission_current, snapshot.mission_total
        if current is None or total is None or int(total) <= 0:
            raise ControllerError("还没有任务进度（mission_raw 进度流未就绪）")
        return int(current) >= int(total)

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
            LOGGER.warning("gripper 未启用（GripperConfig.enabled=False），本次不投放")
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
        except Exception as exc:
            raise ControllerError(f"{what}：无法提交到 MAVSDK 线程（{exc}）") from exc
        try:
            return future.result(timeout=self._timeout)
        except concurrent.futures.TimeoutError as exc:
            raise ControllerError(f"{what}：{self._timeout:.0f}s 内没有结果") from exc
        except Exception as exc:
            raise ControllerError(f"{what}：{exc}") from exc

    def _emit(self, kind: str, data: dict[str, Any]) -> None:
        if self._on_event is None:
            return
        try:
            self._on_event(kind, data)
        except Exception:
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
    def upload_mission(self, items: Sequence[MissionItem], /) -> int:
        return self._inner.upload_mission(items)

    def start_mission(self) -> None:
        self._inner.start_mission()

    def in_mission_mode(self) -> bool:
        return self._inner.in_mission_mode()

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
            except Exception:
                LOGGER.exception("写入演练事件失败（控制继续）")
        return True
