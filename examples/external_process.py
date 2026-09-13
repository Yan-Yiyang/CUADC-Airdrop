"""示例：子进程读取共享遥测、并发出一条指令。

连接 SITL 或真实飞控（默认 UDP 入站 ``udpin://0.0.0.0:14540``）：

    python -m examples.external_process

参数在下面的"配置"区。
"""

from __future__ import annotations

import logging
import multiprocessing
import queue
import time
from typing import Any

from airdrop import Command, ExternalBridge, MavsdkThread, TelemetryBroker

# ----------------------------------------------------------------------
# 配置（改这里，不用命令行参数）
# ----------------------------------------------------------------------
ADDRESS = "udpin://0.0.0.0:14540"  # MAVSDK 系统地址
CHILD_SPEED_M_S = 3.0  # 子进程下发的 set_current_speed 参数
CHILD_READS = 5  # 子进程读取共享快照的次数
CHILD_READ_INTERVAL = 0.5  # 每次读取的间隔（秒）
CHILD_TIMEOUT = 30.0  # 主进程等待子进程的上限（秒）

LOGGER = logging.getLogger("external_process")


def child_main(
    snapshot_proxy: dict[str, Any],
    command_queue: Any,
    result_queue: Any,
) -> None:
    """运行在独立的子进程中。"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    child_log = logging.getLogger("child")
    child_log.info("子进程已启动")

    # 向父进程中的 MAVSDK 线程发送一条指令。
    command = Command("set_current_speed", {"speed_m_s": CHILD_SPEED_M_S})
    command_queue.put(command.as_dict())
    child_log.info("已发送指令 %s (%s)", command.name, command.id)

    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        try:
            raw = result_queue.get(timeout=0.2)
        except queue.Empty:
            continue
        if raw.get("id") == command.id:
            child_log.info(
                "指令结果：success=%s error=%s data=%s",
                raw.get("success"),
                raw.get("error"),
                raw.get("data"),
            )
            break

    # 多次读取共享快照。
    for _ in range(CHILD_READS):
        snap = dict(snapshot_proxy)
        child_log.info(
            "共享快照：lat=%s lon=%s yaw=%s",
            snap.get("latitude_deg"),
            snap.get("longitude_deg"),
            snap.get("yaw_deg"),
        )
        time.sleep(CHILD_READ_INTERVAL)

    child_log.info("子进程执行完毕")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    broker = TelemetryBroker()
    mavsdk = MavsdkThread(broker=broker, system_address=ADDRESS)
    mavsdk.connect(ADDRESS)

    bridge = ExternalBridge(mavsdk, broker=broker, publish_hz=20.0)
    bridge.start()

    # Manager 代理可被 pickle，可安全传给新创建的子进程。
    process = multiprocessing.Process(
        target=child_main,
        args=(bridge.snapshot, bridge.command_queue, bridge.result_queue),
    )
    process.start()
    process.join(timeout=CHILD_TIMEOUT)

    bridge.stop()
    mavsdk.stop()


if __name__ == "__main__":
    main()
