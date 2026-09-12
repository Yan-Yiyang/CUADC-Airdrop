"""思翼 HM30 图传拉流示例：先探流，再拉流，可预览/存盘。

接线上位机（Windows）：
1. 网线插 HM30 **地面端** LAN 口；
2. 电脑网卡配静态地址 ``192.168.144.20`` / 掩码 ``255.255.255.0``（同网段任意空闲地址）；
3. 确认能 ping 通机载相机 ``192.168.144.25``。

常用命令：

    # 只探流，确认地址对不对
    python -m examples.hm30_video --probe

    # 默认地址拉流，统计帧率与延迟
    python -m examples.hm30_video

    # 开窗口预览（按 q 退出）
    python -m examples.hm30_video --preview

    # 缩放到 960x540（管道带宽降到 1/4，YOLO 前处理也更快）
    python -m examples.hm30_video --width 960 --height 540

    # NVDEC 硬解
    python -m examples.hm30_video --decoder h264_cuvid

    # 换成 OpenCV 后端做对照（画面会滞后约 0.5s）
    python -m examples.hm30_video --backend opencv

    # RTSP over TCP（丢包环境更稳但延迟略高）
    python -m examples.hm30_video --transport tcp

    # 地址不确定时一次性试多个候选
    python -m examples.hm30_video --probe --url rtsp://192.168.144.25:8554/main.264 ^
        --url rtsp://192.168.144.25:8554/sub.264
"""

from __future__ import annotations

import argparse
import logging
import time

from airdrop import (
    HM30_DEFAULT_RTSP,
    Hm30VideoSource,
    VideoConfig,
    probe_candidates,
)

LOGGER = logging.getLogger("hm30_video")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HM30 图传拉流")
    parser.add_argument(
        "--url",
        action="append",
        default=None,
        help=f"RTSP 地址，可重复传入以做候选探测（默认 {HM30_DEFAULT_RTSP}）",
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "opencv", "ffmpeg"),
        default="auto",
        help="拉流后端，auto=ffmpeg 优先（延迟可控）；opencv 更简单但画面滞后约 0.5s",
    )
    parser.add_argument(
        "--transport",
        choices=("udp", "tcp"),
        default="udp",
        help="RTSP 传输层，udp 延迟更低、tcp 更耐丢包",
    )
    parser.add_argument("--width", type=int, default=0, help="输出宽度，0=源分辨率")
    parser.add_argument("--height", type=int, default=0, help="输出高度，0=源分辨率")
    parser.add_argument(
        "--decoder",
        default=None,
        help="ffmpeg 解码器，如 h264_cuvid / hevc_cuvid（NVDEC 硬解）",
    )
    parser.add_argument("--probe", action="store_true", help="只探测地址，不拉流")
    parser.add_argument("--preview", action="store_true", help="开窗口预览")
    parser.add_argument("--save", default=None, help="把画面存成 mp4（用于事后核对）")
    parser.add_argument("--duration", type=float, default=0.0, help="运行秒数，0=不限")
    parser.add_argument("--timeout", type=float, default=5.0, help="探流/首帧超时")
    return parser.parse_args()


def do_probe(urls: list[str], transport: str, timeout: float) -> list[str]:
    """探测候选地址，打印可用项，返回拿到的地址列表。"""
    LOGGER.info("开始探测 %d 个候选地址（transport=%s）...", len(urls), transport)
    found = probe_candidates(
        urls,
        transports=(transport, "tcp" if transport == "udp" else "udp"),
        timeout=timeout,
    )
    if not found:
        LOGGER.error("没有探测到可用图传，检查网线/网段/地址")
        return []
    for info in found:
        LOGGER.info(
            "可用   %s  %dx%d @%.1ffps  打开 %.2fs  首帧 %.2fs",
            info.url,
            info.width,
            info.height,
            info.fps,
            info.open_seconds,
            info.first_frame_seconds,
        )
    return [info.url for info in found]


def run(source: Hm30VideoSource, args: argparse.Namespace) -> None:
    """主循环：按"最新帧"消费，并周期性打印链路统计。"""
    if not source.wait_ready(timeout=args.timeout):
        LOGGER.error("首帧超时：%s", source.stats.last_error or "无错误信息")
        return

    writer = None
    started = time.monotonic()
    last_report = started
    last_index = 0

    try:
        while True:
            frame = source.read(timeout=1.0)
            if frame is None:
                LOGGER.warning("读取超时，状态=%s", source.stats.state)
                continue

            if args.save and writer is None:
                import cv2

                writer = cv2.VideoWriter(
                    args.save,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    max(source.stats.fps, 10.0),
                    (frame.width, frame.height),
                )
                LOGGER.info("开始存盘: %s (%dx%d)", args.save, frame.width, frame.height)
            if writer is not None:
                writer.write(frame.image)

            if args.preview:
                import cv2

                cv2.imshow("HM30", frame.image)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            now = time.monotonic()
            if now - last_report >= 2.0:
                stats = source.stats
                # 这两秒里消费了多少帧 / 实际到达了多少帧
                consumed = stats.frames - last_index
                LOGGER.info(
                    "状态=%-11s 帧率=%5.1f 已收=%d 丢弃=%d 重连=%d "
                    "本次读到=%d 画面延迟=%.3fs",
                    stats.state,
                    stats.fps,
                    stats.frames,
                    stats.dropped,
                    stats.reconnects,
                    consumed,
                    frame.age,
                )
                last_index = stats.frames
                last_report = now

            if args.duration and now - started >= args.duration:
                break
    except KeyboardInterrupt:
        LOGGER.info("用户中断")
    finally:
        if writer is not None:
            writer.release()
        if args.preview:
            import cv2

            cv2.destroyAllWindows()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()
    urls = args.url or [HM30_DEFAULT_RTSP]

    if args.probe:
        return 0 if do_probe(urls, args.transport, args.timeout) else 1

    config = VideoConfig(
        url=urls[0],
        backend=args.backend,
        transport=args.transport,
        width=args.width,
        height=args.height,
        open_timeout=args.timeout,
        ffmpeg_decoder=args.decoder,
    )
    LOGGER.info(
        "拉流 %s（backend=%s transport=%s）", config.url, config.backend, config.transport
    )
    with Hm30VideoSource(config) as source:
        run(source, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
