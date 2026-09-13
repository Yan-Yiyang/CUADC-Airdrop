"""思翼 HM30 图传拉流示例：拉流并统计链路质量，可选预览 / 存盘。

接线（Windows 上位机）：
1. 网线插 HM30 **地面端** LAN 口；
2. 电脑网卡配静态地址 ``192.168.144.20`` / 掩码 ``255.255.255.0``（同网段任意空闲地址）；
3. 先确认能 ping 通机载相机 ``192.168.144.25``。

地址不确定时本示例不替你猜：用 ffmpeg 命令行确认一次即可::

    ffmpeg -rtsp_transport udp -i rtsp://192.168.144.25:8554/main.264 -frames:v 1 -f null -

运行（参数都在下面的"配置"区，直接跑）：

    python -m examples.hm30_video

拉流失败会在日志里直接给出 ffmpeg 的输出，不做地址探测、不轮询猜测。

注意本示例消费的是**实时**画面（``read()``）：它保证你看到的是当前一帧，但不保证
每一帧都被处理过。要"一帧都不能丢"（目标可能只出现一瞬），看
``examples/video_telemetry_sync.py``——那里把 ``AlignmentWriter`` 挂在拉流源上，
采集线程每读一帧就写一帧进缓冲。
"""

from __future__ import annotations

import logging
import time

from airdrop import HM30_DEFAULT_RTSP, Hm30VideoSource, VideoConfig

# ----------------------------------------------------------------------
# 配置（改这里，不用命令行参数）
# ----------------------------------------------------------------------
URL = HM30_DEFAULT_RTSP  # 图传 RTSP 地址
WIDTH, HEIGHT = 1280, 720  # 输出尺寸（不探测源分辨率，直接按这个切帧）
TRANSPORT = "udp"  # RTSP 传输层：udp 延迟更低、tcp 更耐丢包
DECODER = None  # 硬解可填 "h264_cuvid" / "hevc_cuvid"（NVDEC）

PREVIEW = False  # 开窗口预览（按 q 退出）
SAVE_PATH = None  # 存成 mp4 的路径，例如 "hm30.mp4"
DURATION = 0.0  # 运行秒数，0=不限

LOGGER = logging.getLogger("hm30_video")


def run(source: Hm30VideoSource) -> None:
    """主循环：按"最新帧"消费，并周期性打印链路统计。"""
    if not source.wait_ready(timeout=10.0):
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

            if SAVE_PATH and writer is None:
                import cv2

                writer = cv2.VideoWriter(
                    SAVE_PATH,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    max(source.stats.fps, 10.0),
                    (frame.width, frame.height),
                )
                LOGGER.info("开始存盘: %s (%dx%d)", SAVE_PATH, frame.width, frame.height)
            if writer is not None:
                writer.write(frame.image)

            if PREVIEW:
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
                    "本次读到=%d 画面延迟=%.3fs 拍摄延迟=%.3fs",
                    stats.state,
                    stats.fps,
                    stats.frames,
                    stats.dropped,
                    stats.reconnects,
                    consumed,
                    frame.age,
                    frame.capture_age,
                )
                last_index = stats.frames
                last_report = now

            if DURATION and now - started >= DURATION:
                break
    except KeyboardInterrupt:
        LOGGER.info("用户中断")
    finally:
        if writer is not None:
            writer.release()
        if PREVIEW:
            import cv2

            cv2.destroyAllWindows()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    config = VideoConfig(
        url=URL,
        transport=TRANSPORT,
        width=WIDTH,
        height=HEIGHT,
        ffmpeg_decoder=DECODER,
    )
    LOGGER.info(
        "拉流 %s（%dx%d transport=%s）",
        config.url,
        config.width,
        config.height,
        config.transport,
    )
    with Hm30VideoSource(config) as source:
        run(source)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
