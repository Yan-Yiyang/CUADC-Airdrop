"""图传帧 ↔ 遥测的时间对齐示例：扣掉链路延时取遥测，结果入缓冲供其他模块读。

背景：从"相机曝光"到"本进程读出一帧"之间有约 150ms 的固定链路延时（编码 +
传输 + 解码 + 管道缓冲）。直接用收到帧的时刻去查遥测，等于拿"现在的姿态"
解释"150ms 前的画面"——飞机以 10 m/s 平飞时就是 1.5 米的系统性偏差。
所以每拿到一帧，先把它的接收时间减掉这段延时，得到真正的拍摄时刻，再按
那个时刻向 :class:`~airdrop.broker.TelemetryBroker` 查询遥测（默认线性内插）。

对齐结果同时放进 :class:`~airdrop.buffer.AlignmentBuffer`：本文件的主循环扮演
"写者"，另起一个读者线程演示"其他模块"如何按自己的节奏回看这段数据。

缓冲容量默认按 30 fps × 3 分钟 = 5400 帧。注意未压缩 720p 单帧 2.64 MiB，
5400 帧 ≈ 13.9 GiB，普通机器放不下，所以默认 ``storage="jpeg"``（约 0.6~1 GiB）；
需要像素级无损时用 ``--buffer-storage raw``，并务必配 ``--buffer-max-mb`` 限制内存。

常用命令：

    # 连飞控 + 拉默认 RTSP 地址，缓冲 3 分钟
    python -m examples.video_telemetry_sync --address udpin://0.0.0.0:14540

    # 标定出新延时后覆盖默认的 150ms
    python -m examples.video_telemetry_sync --lag 0.32

    # 只跑 10 秒，缩到 960x540，缓冲只留 30 秒
    python -m examples.video_telemetry_sync --duration 10 --width 960 --height 540 \
        --buffer-seconds 30

    # 无损缓冲（raw），并把内存限制在 4 GiB
    python -m examples.video_telemetry_sync --buffer-storage raw --buffer-max-mb 4096

    # 换成 OpenCV 后端做对照：此时链路延时约 0.57s，必须同步调 --lag
    python -m examples.video_telemetry_sync --backend opencv --lag 0.57

    # 换个图传地址
    python -m examples.video_telemetry_sync --url rtsp://192.168.144.25:8554/main.264

输出里每行会打印：帧号、画面拍摄至今的时长（应略大于 lag）、对齐时刻是内插
还是外推、以及那一刻的姿态与位置；读者线程另行打印缓冲的消费进度与占用。
"""

from __future__ import annotations

import argparse
import logging
import threading
import time

from airdrop import (
    DEFAULT_TELEMETRY_LAG,
    HM30_DEFAULT_RTSP,
    AlignmentBuffer,
    FrameTelemetryAligner,
    Hm30VideoSource,
    MavsdkThread,
    TelemetryBroker,
    VideoConfig,
    capacity_for,
)

LOGGER = logging.getLogger("video_telemetry_sync")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--address", default="udpin://0.0.0.0:14540", help="MAVSDK 系统地址"
    )
    parser.add_argument(
        "--url", default=HM30_DEFAULT_RTSP, help="图传 RTSP 地址"
    )
    parser.add_argument(
        "--lag",
        type=float,
        default=DEFAULT_TELEMETRY_LAG,
        help=(
            "链路固定延时（秒），默认 150ms。它是链路属性：ffmpeg 后端约 0.15，"
            "OpenCV 后端约 0.57"
        ),
    )
    parser.add_argument(
        "--max-extrapolation",
        type=float,
        default=None,
        help="允许的最大外推秒数，超过则跳过该帧（默认不限制，只做标记）",
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "opencv", "ffmpeg"),
        default="auto",
        help="拉流后端，auto=ffmpeg 优先",
    )
    parser.add_argument("--width", type=int, default=960, help="输出宽度，0=源分辨率")
    parser.add_argument("--height", type=int, default=540, help="输出高度，0=源分辨率")
    parser.add_argument("--timeout", type=float, default=8.0, help="探流/首帧超时")
    parser.add_argument("--duration", type=float, default=0.0, help="运行秒数，0=不限")
    parser.add_argument(
        "--interval", type=float, default=1.0, help="主循环打印间隔（秒）"
    )
    parser.add_argument(
        "--buffer-seconds",
        type=float,
        default=180.0,
        help="对齐缓冲保留多少秒的画面，0=关闭缓冲（默认 180 秒）",
    )
    parser.add_argument(
        "--buffer-fps",
        type=float,
        default=30.0,
        help="按多少帧率换算缓冲容量（默认 30，即 30×180=5400 帧）",
    )
    parser.add_argument(
        "--buffer-storage",
        choices=("jpeg", "raw"),
        default="jpeg",
        help="缓冲存储方式：jpeg 省内存（默认），raw 无损但 720p 单帧 2.64 MiB",
    )
    parser.add_argument(
        "--buffer-max-mb",
        type=float,
        default=0.0,
        help="缓冲占用上限（MB），0=不限制；raw 模式强烈建议设置",
    )
    return parser.parse_args()


def fmt(value, pattern: str) -> str:
    return "n/a" if value is None else pattern % value


def report(sample, frame, lag: float) -> None:
    """打印一帧的对齐结果。"""
    if sample is None:
        LOGGER.info(
            "帧#%-6d 拍摄于 %.3fs 前 | 对齐遥测不可用（历史为空或超出外推限制）",
            frame.index,
            frame.capture_age,
        )
        return

    snapshot = sample.snapshot
    mode = "外推 +%.2fs" % sample.offset if sample.extrapolated else "内插"
    LOGGER.info(
        "帧#%-6d 拍摄于 %.3fs 前（lag=%.0fms）| 遥测 %s @%.3fs 前 | "
        "yaw=%s roll=%s 相对高度=%s 北=%s 东=%s",
        frame.index,
        frame.capture_age,
        lag * 1000.0,
        mode,
        sample.age,
        fmt(snapshot.yaw_deg, "%.1f°"),
        fmt(snapshot.roll_deg, "%.1f°"),
        fmt(snapshot.relative_altitude_m, "%.2fm"),
        fmt(snapshot.north_m, "%.2fm"),
        fmt(snapshot.east_m, "%.2fm"),
    )


def reader_loop(
    buffer: AlignmentBuffer, stop_event: threading.Event, interval: float = 2.0
) -> None:
    """模拟"其他模块"从缓冲区回看数据。

    读者自己记游标，用 ``wait_new`` 往前跟；拿到的 ``record`` 里同时有画面和
    那一刻的遥测，可以直接喂给下游（推理、落盘、回放）。
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
        # 真正消费画面/遥测的位置：这里只做一次轻量访问
        assert record.image.ndim == 3

        now = time.monotonic()
        if now - last_report >= interval:
            stats = buffer.stats
            LOGGER.info(
                "读者：已消费 %d 帧（%.1f fps）| 最新 #%d 拍摄于 %.2fs 前 | "
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


def run(
    source: Hm30VideoSource,
    aligner: FrameTelemetryAligner,
    buffer: AlignmentBuffer | None,
    args: argparse.Namespace,
) -> None:
    """主循环：读一帧 → 对齐遥测 → 入缓冲 → 按间隔打印。"""
    started = time.monotonic()
    last_report = 0.0
    while True:
        frame = source.read(timeout=1.0)
        now = time.monotonic()
        if frame is not None:
            # 关键的一步：拿到帧之后立刻对齐，把该帧对应的遥测一并取出来
            sample = aligner.align(frame)
            if sample is not None and buffer is not None:
                buffer.put(sample)  # 缓冲内部会拷贝，不存复用视图
            if now - last_report >= args.interval:
                report(sample, frame, aligner.lag or frame.lag)
                last_report = now
        elif now - last_report >= args.interval:
            stats = source.stats
            LOGGER.warning(
                "等待画面：状态=%s 已收=%d 丢弃=%d 重连=%d 最后错误=%s",
                stats.state,
                stats.frames,
                stats.dropped,
                stats.reconnects,
                stats.last_error,
            )
            last_report = now

        if args.duration and now - started >= args.duration:
            return


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()

    broker = TelemetryBroker()
    mavsdk = MavsdkThread(broker=broker, system_address=args.address)
    mavsdk.start()
    mavsdk.connect(args.address)

    LOGGER.info("等待遥测就绪...")
    first = broker.wait_next_snapshot(timeout=15.0, predicate=lambda s: s.is_valid())
    if first is None:
        LOGGER.warning("15 秒内没收到遥测，仍会拉流，但对齐结果会不可用（全为 n/a）")

    aligner = FrameTelemetryAligner(
        broker,
        lag=args.lag,
        mode="interpolate",
        max_extrapolation=args.max_extrapolation,
    )

    buffer: AlignmentBuffer | None = None
    stop_event = threading.Event()
    reader: threading.Thread | None = None
    if args.buffer_seconds > 0:
        capacity = capacity_for(args.buffer_fps, args.buffer_seconds)
        buffer = AlignmentBuffer(
            capacity,
            storage=args.buffer_storage,
            max_bytes=int(args.buffer_max_mb * 1024 * 1024) or None,
        )
        LOGGER.info(
            "对齐缓冲：容量 %d 帧（%.0f fps × %.0f s）、存储 %s",
            capacity,
            args.buffer_fps,
            args.buffer_seconds,
            args.buffer_storage,
        )
        reader = threading.Thread(
            target=reader_loop, args=(buffer, stop_event), daemon=True
        )
        reader.start()

    config = VideoConfig(
        url=args.url,
        backend=args.backend,
        width=args.width,
        height=args.height,
        open_timeout=args.timeout,
        telemetry_lag=args.lag,
    )
    LOGGER.info(
        "拉流 %s（backend=%s），链路延时按 %.0fms 扣除",
        config.url,
        config.backend,
        config.telemetry_lag * 1000.0,
    )

    try:
        with Hm30VideoSource(config) as source:
            if not source.wait_ready(timeout=args.timeout):
                LOGGER.error("首帧超时：%s", source.stats.last_error or "无错误信息")
                return 1
            run(source, aligner, buffer, args)
    except KeyboardInterrupt:
        LOGGER.info("用户中断")
    finally:
        stop_event.set()
        if reader is not None:
            reader.join(timeout=2.0)
        if buffer is not None:
            stats = buffer.stats
            LOGGER.info(
                "缓冲收尾：入库 %d 帧、驱逐 %d 帧、当前 %d 帧（%.1f MiB）",
                stats.put,
                stats.evicted,
                stats.frames,
                stats.bytes / 1024 / 1024,
            )
        mavsdk.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
