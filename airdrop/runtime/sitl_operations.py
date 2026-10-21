"""SITL 飞控参数账本与操作手启动步骤。"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from airdrop import Config, TelemetryBroker
from airdrop.telemetry._operations import get_operation_gate

from .sitl_geometry import TAKEOFF_ALT_M

LOGGER = logging.getLogger("sitl_recon")
PARAM_RPC_TIMEOUT_S = 15.0


def _write_param(thread: Any, name: str, value: int, *, recovery: bool = False) -> None:
    """在 MAVSDK 线程上写一条飞控参数（同步等待结果）。"""

    async def _write() -> None:
        await thread.require_drone().param.set_param_int(name, value)

    get_operation_gate(thread).run(
        thread.submit,
        _write,
        timeout=PARAM_RPC_TIMEOUT_S,
        label=f"写飞控参数 {name}",
        mutating=True,
        recovery=recovery,
    )


def read_param(thread: Any, name: str) -> int:
    """读一条飞控参数（在 MAVSDK 线程上执行）。"""

    async def _read() -> int:
        return await thread.require_drone().param.get_param_int(name)

    return get_operation_gate(thread).run(
        thread.submit,
        _read,
        timeout=PARAM_RPC_TIMEOUT_S,
        label=f"读飞控参数 {name}",
        mutating=False,
    )


def zero_test_params(
    thread: Any,
    names: tuple[str, ...] = ("NAV_DLL_ACT", "NAV_RCL_ACT"),
    *,
    ledger: dict[str, int] | None = None,
) -> dict[str, int]:
    """把 SITL 测试要临时关掉的飞控参数置 0，返回原值（``main`` 收尾时还原）。

    * ``NAV_DLL_ACT``：SITL 里没有操作手盯数据链，置 0 免得"数据链丢失"的 failsafe
      在飞行中途抢控制；
    * ``NAV_RCL_ACT``：SITL 里**没有遥控**（ulog 的 ``manual_control_signal_lost``
      恒 true），默认动作是 Return；因此仿真测试应明确处理 RC-loss failsafe，
      避免它在任务过程中接管飞行。

    ⚠ "起飞项/降落项是否必需"那条检查（``MIS_TKO_LAND_REQ``）**不在这里改**：
    它已经在仿真机型里永久置 0（``vendor/airdrop-sim/airdrop_sim/assets/airframes/4007_gz_rc_cessna_down_cam``，附了原因）
    ——投弹航线要空中出结果才能生成、降落段没法预先和侦查段拼成一条任务。
    """
    if ledger is None:
        ledger = {}
    for name in names:
        value = read_param(thread, name)
        ledger.setdefault(name, value)
        _write_param(thread, name, 0)
    return ledger


def restore_params(thread: Any, values: dict[str, int]) -> None:
    """还原全部测试参数；单项失败不阻断其它恢复，并在最后报告失败。"""
    failures: list[tuple[str, int, Exception]] = []
    for name, value in values.items():
        try:
            _write_param(thread, name, value, recovery=True)
            LOGGER.info("飞控参数 %s 已还原为 %s", name, value)
        except Exception as exc:  # noqa: BLE001 - 尽量恢复其它参数，再汇总失败
            failures.append((name, value, exc))
            LOGGER.warning("还原飞控参数 %s 失败，原值 %s 仍在账本中", name, value)
    if failures:
        names = ", ".join(name for name, _value, _exc in failures)
        raise RuntimeError(f"恢复 SITL 参数失败：{names}") from failures[0][2]


def start_status_text_logger(thread: Any) -> None:
    """把 PX4 的原生消息（STATUSTEXT）原样打进日志：出问题时第一手证据就在这儿。

    只**监听**，不下任何指令。
    """

    async def _start(_=None) -> None:
        drone = thread.require_drone()

        async def _loop() -> None:
            async for message in drone.telemetry.status_text():
                LOGGER.info("PX4: %s", message.text)

        # 存住引用：任务要一直活到线程退出（RUF006）
        _tasks.append(asyncio.create_task(_loop()))

    thread.submit(_start()).result(timeout=10.0)


#: 后台监听任务（只用它拿住强引用）
_tasks: list[Any] = []


def takeoff_items(config: Config) -> tuple[Any, ...]:
    """只含起飞项的任务：位置取侦查航线首点、高度取 :data:`TAKEOFF_ALT_M`。"""
    from airdrop import MissionItem, build_recon_mission

    first = next((item for item in build_recon_mission(config) if item.is_positional), None)
    if first is None:
        raise ValueError("SITL 起飞需要侦查计划中至少一个有位置的任务项")
    return (MissionItem.takeoff(float(first.lat), float(first.lon), TAKEOFF_ALT_M),)


def operator_takeoff(
    config: Config, controller: Any, thread: Any, broker: TelemetryBroker
) -> int | None:
    """**操作手起跑**（自动化的等价物）：上传"只含起飞项"的任务 → arm → 启动 → 等在空中。

    真实流程里这一步由操作手完成（手飞/RC，或在 QGC 里点起飞）；SITL 里没有遥控，
    用一条只含 ``NAV_TAKEOFF`` 的任务代替——固定翼的起飞由飞控按任务首项执行，
    与"侦查航线首项带起飞"用的是同一套机制（``MissionItem.takeoff``）。起飞完成后
    飞控转 AUTO_LOITER 在起点附近盘旋，等下一步"操作手上传侦查航线"。

    两条硬要求（都是实测踩出来的）：
    * **先解锁、再启动**：``MISSION_START`` 里的 ``arm(mission_start)`` 实测只切模式、
      不解锁（ulog: nav_state=3 / arming_state=1），飞机不动；
    * **任务先过可行性检查**：装了起飞项就必须满足 ``MIS_TKO_LAND_REQ``
      （仿真机型里已永久置 0，见 ``vendor/airdrop-sim/airdrop_sim/assets/airframes/4007_gz_rc_cessna_down_cam``）。

    返回 ``None`` 表示已在空中，否则返回退出码（4 没解锁 / 5 没进任务模式 / 6 没起飞）。
    """

    async def _armed() -> bool:
        async for value in thread.require_drone().telemetry.armed():
            return bool(value)
        return False

    from airdrop import command_name

    disarm = thread.send_command({"name": "disarm"}, timeout=15.0)
    LOGGER.info("起跑前 disarm：%s", "OK" if disarm.success else f"跳过（{disarm.error}）")
    start_status_text_logger(thread)

    items = takeoff_items(config)
    uploaded = controller.upload_mission(items)
    LOGGER.info("起飞任务已上传：%d 项（首项 %s）", uploaded, command_name(items[0].command))

    arm = thread.send_command({"name": "arm"}, timeout=30.0)
    if not arm.success:
        LOGGER.error("解锁指令被拒：%s（预检没过？看 QGC / PX4 的提示）", arm.error)
        return 4
    if not wait_for(lambda: bool(thread.submit(_armed()).result(timeout=5.0)), 20.0, "解锁生效"):
        LOGGER.error("解锁指令被接受但飞机没有真的解锁：预检未过（QGC 会写原因）")
        return 4
    LOGGER.info("已解锁")

    controller.start_mission()
    if not wait_for(controller.in_mission_mode, 20.0, "进入任务模式"):
        LOGGER.error("启动后 20s 没进任务模式（飞控拒绝任务？）")
        return 5
    if not wait_for(
        lambda: (
            bool(broker.get_snapshot().in_air)
            or (broker.get_snapshot().relative_altitude_m or 0.0) >= 5.0
        ),
        60.0,
        "起飞",
    ):
        LOGGER.error("60s 内没有起飞（自动起飞失败？）")
        return 6
    LOGGER.info("已起飞（操作手起跑步骤完成）")
    return None


def wait_for(predicate, timeout_s: float, what: str, *, interval_s: float = 0.2) -> bool:
    """轮询等条件成立；超时返回 False（不抛，交给调用方决定）。"""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            if predicate():
                return True
        except Exception:
            LOGGER.debug("%s：查询失败，继续等", what, exc_info=True)
        time.sleep(interval_s)
    return False
