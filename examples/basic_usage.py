"""最小示例：MAVSDK 专用线程 + 遥测代理 + 指令发送。

连接 SITL 或真实飞控（默认 UDP 入站 ``udpin://0.0.0.0:14540``）：

    python -m examples.basic_usage

参数在下面的"配置"区。``DEMO_COMMANDS`` 打开后会依次发 arm -> takeoff ->
land，**只在安全环境（桨已拆除 / SITL）里用**。
"""

from __future__ import annotations

import logging
import threading
import time

from airdrop import (
    Command,
    MavsdkThread,
    TelemetryBroker,
    TelemetrySnapshot,
)

# ----------------------------------------------------------------------
# 配置（改这里，不用命令行参数）
# ----------------------------------------------------------------------
ADDRESS = "udpin://0.0.0.0:14540"  # MAVSDK 系统地址
DEMO_COMMANDS = False  # 是否演示 arm -> takeoff -> land（安全环境才开）

LOGGER = logging.getLogger("basic_usage")


def on_snapshot(snapshot: TelemetrySnapshot) -> None:
    """遥测订阅回调：运行在 MAVSDK 工作线程中。"""
    if snapshot.is_valid() and snapshot.latitude_deg is not None:
        LOGGER.info(
            "推送  纬度=%.7f 经度=%.7f 相对高度=%.2f 偏航=%.1f",
            snapshot.latitude_deg,
            snapshot.longitude_deg,
            snapshot.relative_altitude_m,
            snapshot.yaw_deg,
        )


def reader_loop(broker: TelemetryBroker, stop_event: threading.Event) -> None:
    """另一个线程读取同一份同步快照。"""
    while not stop_event.is_set():
        snapshot = broker.get_snapshot()
        if snapshot.is_valid():
            LOGGER.info(
                "轮询  北=%.2f 东=%.2f 下=%.2f vx=%.2f vy=%.2f vz=%.2f",
                snapshot.north_m,
                snapshot.east_m,
                snapshot.down_m,
                snapshot.vx_m_s,
                snapshot.vy_m_s,
                snapshot.vz_m_s,
            )
        stop_event.wait(0.5)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    broker = TelemetryBroker()
    broker.subscribe(on_snapshot)

    mavsdk = MavsdkThread(broker=broker, system_address=ADDRESS)
    mavsdk.start()
    mavsdk.connect(ADDRESS)

    LOGGER.info("等待第一条遥测快照...")
    first = broker.wait_next_snapshot(timeout=15.0, predicate=lambda s: s.is_valid())
    if first is None:
        LOGGER.error("15 秒内未收到任何遥测数据")
        mavsdk.stop()
        return

    LOGGER.info("第一条快照：%s", first.as_dict())

    # 按时间戳查询：其他模块给出一个时间戳时，可获取那一刻的状态。
    historical = broker.get_snapshot_at(first.timestamp)
    LOGGER.info("按时间戳查询结果：%s", historical.as_dict())

    stop_event = threading.Event()
    reader = threading.Thread(
        target=reader_loop, args=(broker, stop_event), daemon=True
    )
    reader.start()

    if DEMO_COMMANDS:
        for command in (Command("arm"), Command("takeoff"), Command("land")):
            result = mavsdk.send_command(command)
            LOGGER.info(
                "指令 %s -> success=%s error=%s",
                command.name,
                result.success,
                result.error,
            )
            if not result.success:
                break
            time.sleep(2.0)

    # 3.0 起不再有跨进程桥（external_bridge 已删除）：消费者直接读同一个 broker；
    # 唯一需要的多进程组件是感知侧的 OCR 工作进程（见 REWRITE_PLAN.md 4.3）。

    try:
        LOGGER.info("运行中... 按 Ctrl+C 停止")
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        reader.join(timeout=2.0)
        mavsdk.stop()


if __name__ == "__main__":
    main()
