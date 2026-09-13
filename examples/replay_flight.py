"""回放飞行目录示例：同一份素材反复迭代识别与坐标解算，全程不需要硬件。

用法::

    python -m examples.replay_flight

把 ``FLIGHT_DIR`` 指向 ``FlightRecorder`` 录出来的飞行目录即可。脚本会：
按原时间轴重放帧与遥测 → 对齐 → 进环形缓冲 → 消费者按自己的节奏取用，
并打印每帧的**拍摄时刻**与那一刻的遥测，最后给出整体统计。

为什么要用回放迭代
------------------
算法改动要在**固定输入**上比较才有意义。回放时每一帧仍带着录制时的原始
``capture_timestamp``、遥测仍按原始时间戳进入 broker，于是帧-遥测对齐、
坐标解算、目标统计**一行都不用改**：实飞怎么跑，离线就怎么跑。换一批参数
再跑一遍，输出可以直接和上一次（以及原 ``detections.jsonl``）对比。

    FLIGHT_DIR = "flights/20260913-185512"   # 走回放
    # 实飞时换回 Hm30VideoSource + 真实 broker，下游代码完全不变

速度
----
``SPEED = 0`` 全速（受算力限制，用于批量回归）；``1.0`` 原速（观察实时行为、
调试需要"时间感"的逻辑）；``2.0`` 两倍速。要"一帧不落"就用 sink + 缓冲，
别用 ``read()`` 循环——与实飞同一条铁律。
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

from airdrop import (
    AlignmentBuffer,
    AlignmentWriter,
    FrameTelemetryAligner,
    ReplayVideoSource,
    capacity_for,
    load_broker_from_log,
)

# ----------------------------------------------------------------------
# 配置（改这里，不用命令行参数）
# ----------------------------------------------------------------------
FLIGHT_DIR = Path("flights")  # 飞行目录；指向具体那一次，例如 flights/20260913-185512
SPEED = 0.0  # 0=全速（批量回归）/ 1.0=原速 / 2.0=两倍速
BUFFER_SECONDS = 180.0  # 回放缓冲时长（秒）
BUFFER_FPS = 30.0  # 容量换算帧率
REPORT_EVERY = 100  # 每处理多少帧汇报一次
READER_PACE = 0.0  # 消费者每帧额外停顿（秒），模拟慢消费者（0=不等待）

LOGGER = logging.getLogger("replay_flight")


def resolve_flight_dir() -> Path:
    """``FLIGHT_DIR`` 指向具体目录时直接用；指向 ``flights/`` 时取最新一次。"""
    if (FLIGHT_DIR / "frames_index.jsonl").is_file():
        return FLIGHT_DIR
    candidates = sorted(
        (p for p in FLIGHT_DIR.glob("*") if (p / "frames_index.jsonl").is_file()),
        key=lambda p: p.name,
    )
    if not candidates:
        raise SystemExit(
            f"{FLIGHT_DIR} 下没有找到飞行目录（需要 frames_index.jsonl + telemetry.jsonl）。\n"
            "先跑一次带录制的飞行，或把 FLIGHT_DIR 指到已有的素材上。"
        )
    LOGGER.info("FLIGHT_DIR 指向 flights/ 根目录，自动选用最新一次：%s", candidates[-1].name)
    return candidates[-1]


def reader_loop(buffer: AlignmentBuffer, stop_event, pace: float) -> None:
    """扮演"其他模块"：从缓冲按自己的节奏取帧（推理 / 落盘 / 画图都放这里）。

    回放时这里的输入与实飞完全一致——缓冲里的每条记录都是"画面 + 拍摄时刻
    的遥测"，所以这一段的代码在实飞与离线之间可以原样复用。
    """
    index = 0
    seen = 0
    while not stop_event.is_set():
        record = buffer.wait_new(index, timeout=0.5)
        if record is None:
            continue
        index = record.index
        seen += 1
        if pace:
            time.sleep(pace)
        if seen % REPORT_EVERY == 0:
            LOGGER.info(
                "消费者：#%d 拍摄于 %.3f（%.2fs 前）| north=%s east=%s yaw=%s",
                record.index,
                record.capture_timestamp,
                record.age,
                record.snapshot.north_m,
                record.snapshot.east_m,
                record.snapshot.yaw_deg,
            )
    LOGGER.info("消费者退出：共处理 %d 帧", seen)


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    flight_dir = resolve_flight_dir()
    flight_dir = Path(flight_dir)

    # 1) 遥测回填：建一个专用 broker + 按帧推进的 pacer
    broker, pacer = load_broker_from_log(flight_dir)
    LOGGER.info("素材：%s（%d 条遥测）", flight_dir, pacer.total)

    # 2) 回放源：与 Hm30VideoSource 同一套接口，直接换掉即可
    replay_kwargs: dict = {"speed": SPEED, "telemetry": pacer}
    source = ReplayVideoSource(flight_dir, **replay_kwargs)
    stats = source.stats
    LOGGER.info(
        "回放：%d 帧、素材跨度 %.1fs、%s",
        stats.total,
        stats.duration_s,
        "全速" if SPEED == 0 else f"{SPEED}× 速度",
    )

    # 3) 下游与实飞一致：对齐 → 缓冲 → 消费者
    aligner = FrameTelemetryAligner(broker, max_wait=1.0)
    buffer = AlignmentBuffer(
        capacity_for(BUFFER_FPS, BUFFER_SECONDS), storage="jpeg"
    )
    writer = AlignmentWriter(buffer, aligner)

    stop_event = threading.Event()
    reader = threading.Thread(
        target=reader_loop, args=(buffer, stop_event, READER_PACE), daemon=True
    )
    reader.start()

    started = time.monotonic()
    try:
        # 一帧不落的那条路：sink 在播放线程里逐帧调用
        source.add_sink(writer)
        source.start()
        if not source.wait_ready(timeout=30.0):
            LOGGER.error("回放没有出帧：%s", source.stats.last_error or "未知原因")
            return 1
        while source.running:
            time.sleep(0.2)
    except KeyboardInterrupt:
        LOGGER.info("用户中断")
    finally:
        source.stop()
        stop_event.set()
        reader.join(timeout=2.0)

    elapsed = time.monotonic() - started
    stats = source.stats
    written, skipped = writer.stats()
    align_stats = aligner.stats
    LOGGER.info(
        "回放结束：状态=%s 投递 %d/%d 帧、跳过 %d 帧、墙钟 %.1fs（素材 %.1fs，%.1f× 实时）",
        stats.state,
        stats.frames,
        stats.total,
        stats.skipped,
        elapsed,
        stats.duration_s,
        stats.duration_s / elapsed if elapsed > 0 else 0.0,
    )
    LOGGER.info(
        "对齐：入库 %d 帧、因取不到遥测跳过 %d 帧 | 对齐 %d、等超时 %d、外推丢弃 %d",
        written,
        skipped,
        align_stats.aligned,
        align_stats.wait_timeouts,
        align_stats.extrapolation_drops,
    )
    if align_stats.wait_timeouts or skipped:
        LOGGER.warning(
            "有帧没能对齐。回放时遥测由 pacer 按帧推进，正常情况下不该等、更不该丢；"
            "出现等待说明素材的遥测中间断过（telemetry.jsonl 被截断或链路掉过），"
            "那几帧的遥测不可信。"
        )
    # 4) 想要"一把拿到全部结果"的自己 list()：缓冲不提供这个接口是有意的
    #    （5400 帧 720p 全解出来是 13.9 GiB），账要由调用方认。
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
