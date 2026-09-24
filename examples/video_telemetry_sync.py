"""图传帧 ↔ 遥测的对齐与留存示例：每一帧都进缓冲，随时回看。

为什么不是"读一帧 → 处理 → 再读一帧"：那样只要有一帧处理慢了，中间帧就被覆盖
丢掉了。而目标出现的时间可能极短，漏一帧就可能漏掉目标。本项目走"先侦查后空投"，
可以接受一定的处理延时，不能接受丢帧。所以这里的结构是：

1. 建一个环形缓冲（默认 30 fps × 3 分钟 = 5400 帧）；
2. 把 :class:`~airdrop.buffer.AlignmentWriter` 挂到拉流源上——它在采集线程里
   逐帧被调用，对齐遥测后写进缓冲，一帧不落；
3. 另起一个线程扮演"其他模块"，从缓冲里按自己的节奏取数据（推理 / 写入磁盘 / 回放），
   快慢都不影响第 2 步。

链路延时（默认 150ms）在写入时就已扣掉：缓冲里的每一帧都带着它拍摄时刻的
遥测，回看的人不用再自己换算。

用法（集中式入口在 airdrop/run.py）::

    ./.venv/Scripts/python.exe -m airdrop.run video-sync
    ./.venv/Scripts/python.exe -m airdrop.run video-sync --rtsp-url rtsp://... --telemetry-lag 0.18

本文件是纯库模块：顶部常量是默认值，build_config(**覆盖) / main(**kwargs) 按需传值；
命令行由 airdrop/run.py 解析，重依赖都在函数体内导入（--help 不加载它们）。
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import replace
from typing import TYPE_CHECKING

from airdrop import (
    DEFAULT_TELEMETRY_LAG,
    HM30_DEFAULT_RTSP,
    Config,
    FrameTelemetryAligner,
    Hm30VideoSource,
    TelemetryBroker,
    VideoConfig,
)

if TYPE_CHECKING:
    from airdrop.video.buffer import AlignmentBuffer

# ----------------------------------------------------------------------
# 配置（改这里）
# ----------------------------------------------------------------------
SYSTEM_ADDRESS: str | None = None  # 保留位：None = 不改 TelemetryConfig 的地址
ADDRESS = "udpin://0.0.0.0:14540"  # MAVSDK 系统地址（与 TelemetryConfig 默认值一致）
VIDEO_URL = HM30_DEFAULT_RTSP  # 图传 RTSP 地址
WIDTH, HEIGHT = 1280, 720  # 图传输出尺寸（不探测，直接按这个尺寸切帧）
LAG = DEFAULT_TELEMETRY_LAG  # 链路固定延时（秒）

BUFFER_SECONDS = 180.0  # 缓冲保留时长（秒）：30 × 180 = 5400 帧
BUFFER_FPS = 30.0  # 容量换算帧率
BUFFER_STORAGE = "jpeg"  # jpeg 省内存（满载约 0.6~1 GiB）/ raw 无损但 ≈13.9 GiB
BUFFER_MAX_MB = 0  # 缓冲占用上限（MB），0=不限

RUN_SECONDS = 0.0  # 运行时长，0=一直跑
REPORT_INTERVAL = 2.0  # 状态汇报间隔（秒）
READ_INTERVAL = 0.05  # 读者线程两次取帧之间的间隔（秒），模拟"慢慢消费"

LOGGER = logging.getLogger("video_telemetry_sync")


def build_config(
    *,
    system_address: str = ADDRESS,
    video_url: str = VIDEO_URL,
    width: int = WIDTH,
    height: int = HEIGHT,
    lag: float = LAG,
) -> Config:
    """图传 + 遥测那两片配置（本示例不用感知/任务）。"""
    base = Config()
    return Config(
        telemetry=replace(base.telemetry, system_address=system_address),
        video=VideoConfig(url=video_url, width=width, height=height, telemetry_lag=lag).validated(),
    ).validated()


def reader_loop(buffer: AlignmentBuffer, stop_event: threading.Event, pace: float) -> None:
    """扮演"其他模块"：从缓冲里按自己的节奏取帧。

    这里刻意放慢节奏（每帧 sleep 一小会儿）——它证明消费端再慢也不会让采集端
    丢帧：慢的只有它自己的游标，缓冲里的历史一帧都不会少。
    真实场景里这一段就是推理、写入磁盘或回放。
    """
    index = 0
    seen = 0
    started = time.monotonic()
    last_report = started
    while not stop_event.is_set():
        record = buffer.wait_new(index, timeout=0.5)
        if record is None:
            continue
        index = record.index
        seen += 1
        # 真正消费的位置：画面 + 那一刻的遥测都在 record 里
        assert record.image.ndim == 3
        time.sleep(pace)

        now = time.monotonic()
        if now - last_report >= REPORT_INTERVAL:
            stats = buffer.stats
            LOGGER.info(
                "读者：已消费 %d 帧（%.1f fps，慢速模拟）| 最新 #%d 拍摄于 %.2fs 前 | "
                "缓冲 %d/%d 帧、%.1f MiB、已驱逐 %d",
                seen,
                seen / max(now - started, 1e-6),
                index,
                record.age,
                stats.frames,
                stats.capacity,
                stats.bytes / 1024 / 1024,
                stats.evicted,
            )
            last_report = now
    LOGGER.info("读者线程退出：共消费 %d 帧", seen)


def review(buffer: AlignmentBuffer) -> None:
    """演示"先侦查、后回看"：把缓冲里留下的数据按时间顺序过一遍。"""
    span = buffer.indices()
    if span is None:
        LOGGER.warning("缓冲里没有数据可回看")
        return
    first_index, last_index = span
    first = next(iter(buffer.iter_between(batch=1)), None)
    if first is None:
        return
    latest = buffer.latest()
    LOGGER.info(
        "回看：缓冲保留 #%d~#%d 共 %d 帧，跨度 %.1f 秒",
        first_index,
        last_index,
        len(buffer),
        (latest.capture_timestamp - first.capture_timestamp) if latest else 0.0,
    )
    LOGGER.info(
        "回看：第 #%d 帧拍摄于 %.3f，yaw=%s；最新 #%d 拍摄于 %.3f，yaw=%s",
        first.index,
        first.capture_timestamp,
        first.snapshot.yaw_deg,
        latest.index if latest else -1,
        latest.capture_timestamp if latest else float("nan"),
        latest.snapshot.yaw_deg if latest else None,
    )


def main(
    *,
    system_address: str = ADDRESS,
    video_url: str = VIDEO_URL,
    width: int = WIDTH,
    height: int = HEIGHT,
    lag: float = LAG,
    buffer_seconds: float = BUFFER_SECONDS,
    buffer_fps: float = BUFFER_FPS,
    buffer_storage: str = BUFFER_STORAGE,
    buffer_max_mb: float = BUFFER_MAX_MB,
    run_seconds: float = RUN_SECONDS,
    report_interval: float = REPORT_INTERVAL,
    read_interval: float = READ_INTERVAL,
) -> int:
    """拉流 + 逐帧留存演示；返回进程退出码。"""
    # 重依赖在使用时才导入：`python -m airdrop.run video-sync --help` 不加载它们
    from airdrop import (
        AlignmentBuffer,
        AlignmentWriter,
        MavsdkThread,
        capacity_for,
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    config = build_config(
        system_address=system_address,
        video_url=video_url,
        width=width,
        height=height,
        lag=lag,
    )

    broker = TelemetryBroker()
    mavsdk = MavsdkThread(broker=broker, system_address=config.telemetry.system_address)
    try:
        mavsdk.start()
        mavsdk.connect(config.telemetry.system_address)
    except BaseException:
        mavsdk.stop()
        raise

    LOGGER.info("等待遥测就绪...")
    if broker.wait_next_snapshot(timeout=15.0, predicate=lambda s: s.is_valid()) is None:
        LOGGER.warning("15 秒内没收到遥测：拉流不受影响，但对齐取不到遥测的帧会被计为 skipped")

    aligner = FrameTelemetryAligner(broker, lag=config.video.telemetry_lag)

    buffer = AlignmentBuffer(
        capacity_for(buffer_fps, buffer_seconds),
        storage=buffer_storage,
        max_bytes=int(buffer_max_mb * 1024 * 1024) or None,
    )
    # 挂在拉流源上的"每帧必写"通道
    writer = AlignmentWriter(buffer, aligner)
    LOGGER.info(
        "对齐缓冲：容量 %d 帧（%.0f fps × %.0f s）、存储 %s",
        buffer.capacity,
        buffer_fps,
        buffer_seconds,
        buffer_storage,
    )

    stop_event = threading.Event()
    reader = threading.Thread(
        target=reader_loop, args=(buffer, stop_event, read_interval), daemon=True
    )
    reader.start()

    LOGGER.info(
        "拉流 %s（%dx%d），链路延时按 %.0fms 扣除",
        config.video.url,
        config.video.width,
        config.video.height,
        config.video.telemetry_lag * 1000.0,
    )

    started = time.monotonic()
    try:
        with Hm30VideoSource(config.video) as source:
            source.add_sink(writer)  # 从这一帧起，读到的每一帧都进缓冲
            if not source.wait_ready(timeout=10.0):
                LOGGER.error("首帧超时：%s", source.stats.last_error or "无错误信息")
                return 1
            LOGGER.info("已开始留存：采集线程每读一帧就写一帧进缓冲")

            # 主循环只做汇报，不在采集路径上做重活（重活在读者线程里）
            while True:
                time.sleep(report_interval)
                stats = source.stats
                written, skipped = writer.stats()
                LOGGER.info(
                    "采集：状态=%s 帧率=%.1f 已收=%d | 缓冲：写入=%d 跳过=%d "
                    "帧数=%d 驱逐=%d | 画面延迟=%.3fs",
                    stats.state,
                    stats.fps,
                    stats.frames,
                    written,
                    skipped,
                    len(buffer),
                    buffer.stats.evicted,
                    (source.latest().age if source.latest() else float("nan")),
                )
                if run_seconds and time.monotonic() - started >= run_seconds:
                    break
                if stats.frames == 0 and stats.state not in ("connecting", "streaming"):
                    LOGGER.error("一帧都没收到，退出：%s", stats.last_error)
                    return 1
    except KeyboardInterrupt:
        LOGGER.info("用户中断")
    finally:
        stop_event.set()
        reader.join(timeout=2.0)
        written, skipped = writer.stats()
        LOGGER.info(
            "收尾：入库 %d 帧、因取不到遥测跳过 %d 帧、缓冲现有 %d 帧（%.1f MiB）",
            written,
            skipped,
            len(buffer),
            buffer.stats.bytes / 1024 / 1024,
        )
        review(buffer)
        mavsdk.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
