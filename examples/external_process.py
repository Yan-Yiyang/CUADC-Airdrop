"""示例：子进程读取共享遥测并发出一条指令。

连接 SITL 或真实飞控（UDP）：

    python -m examples.external_process --address udp://:14540
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing
import queue
import time
from typing import Any, Dict

from airdrop import Command, ExternalBridge, MavsdkThread, TelemetryBroker

LOGGER = logging.getLogger("external_process")


def child_main(
    snapshot_proxy: Dict[str, Any],
    command_queue: Any,
    result_queue: Any,
    duration_s: float = 5.0,
) -> None:
    """运行在独立的子进程中。"""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    child_log = logging.getLogger("child")
    child_log.info("子进程已启动")

    # 向父进程中的 MAVSDK 线程发送一条指令。
    command = Command("set_current_speed", {"speed_m_s": 3.0})
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
    for _ in range(5):
        snap = dict(snapshot_proxy)
        child_log.info(
            "共享快照：lat=%s lon=%s yaw=%s",
            snap.get("latitude_deg"),
            snap.get("longitude_deg"),
            snap.get("yaw_deg"),
        )
        time.sleep(0.5)

    child_log.info("子进程执行完毕")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--address", default="udp://:14540", help="MAVSDK 系统地址")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    broker = TelemetryBroker()
    mavsdk = MavsdkThread(broker=broker, system_address=args.address)
    mavsdk.connect(args.address)

    bridge = ExternalBridge(mavsdk, broker=broker, publish_hz=20.0)
    bridge.start()

    # Manager 代理可被 pickle，可安全传给新创建的子进程。
    process = multiprocessing.Process(
        target=child_main,
        args=(bridge.snapshot, bridge.command_queue, bridge.result_queue),
    )
    process.start()
    process.join(timeout=30.0)

    bridge.stop()
    mavsdk.stop()


if __name__ == "__main__":
    main()
