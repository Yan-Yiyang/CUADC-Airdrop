"""RTSP 视频拉流示例：拉流并统计链路质量，可选预览 / 存盘。

使用前：
1. 将视频源接入网络，确认本地可路由到相机地址；
2. 在 ``VideoConfig``（或命令行 ``--rtsp-url``）中配置 RTSP 地址。

地址不确定时本示例不做探测，可用 ffmpeg 命令行确认::

    ffmpeg -rtsp_transport udp -i <RTSP 地址> -frames:v 1 -f null -

运行（集中式入口在 airdrop/run.py）::

    ./.venv/Scripts/python.exe -m airdrop.run hm30-video
    ./.venv/Scripts/python.exe -m airdrop.run hm30-video --rtsp-url <RTSP 地址>

本文件是纯库模块：顶部常量是默认值，build_config(**覆盖) / main(**kwargs) 按需传值；
命令行由 airdrop/run.py 解析。cv2 只在"预览/存盘"两条分支中导入，--help 不加载它。

拉流失败时日志中会直接给出 ffmpeg 的输出，不做地址探测或轮询。

注意本示例消费实时画面（``read()``）：保证看到当前一帧，但不保证每帧都被处理。
需要逐帧不丢（目标可能仅出现一瞬）时参见 ``examples/video_telemetry_sync.py``——
该示例将 ``AlignmentWriter`` 挂载到拉流源上，采集线程每读一帧即写入缓冲。
"""

from __future__ import annotations

import logging
import time

from airdrop import HM30_DEFAULT_RTSP, Hm30VideoSource, VideoConfig

# ----------------------------------------------------------------------
# 配置（改这里）
# ----------------------------------------------------------------------
URL = HM30_DEFAULT_RTSP  # 视频 RTSP 地址
WIDTH, HEIGHT = 1280, 720  # 输出尺寸（不探测源分辨率，直接按这个切帧）
TRANSPORT = "udp"  # RTSP 传输层：udp 延迟更低、tcp 更耐丢包
DECODER = None  # 硬解可填 "h264_cuvid" / "hevc_cuvid"（NVDEC）

PREVIEW = False  # 开窗口预览（按 q 退出）
SAVE_PATH = None  # 存成 mp4 的路径，例如 "stream.mp4"
DURATION = 0.0  # 运行秒数，0=不限

LOGGER = logging.getLogger("hm30_video")


def build_config(
    *,
    url: str = URL,
    width: int = WIDTH,
    height: int = HEIGHT,
    transport: str = TRANSPORT,
    decoder: str | None = DECODER,
) -> VideoConfig:
    """视频那一片配置（本示例只用得到它）。"""
    return VideoConfig(
        url=url,
        transport=transport,
        width=width,
        height=height,
        ffmpeg_decoder=decoder,
    ).validated()


def run(  # noqa: PLR0912 - 预览、存盘、断流和时限是互斥运行路径
    source: Hm30VideoSource,
    *,
    preview: bool = PREVIEW,
    save_path: str | None = SAVE_PATH,
    duration: float = DURATION,
) -> None:
    """主循环：按"最新帧"消费，并周期性打印链路统计。"""
    if not source.wait_ready(timeout=10.0):
        LOGGER.error("首帧超时：%s", source.stats.last_error or "无错误信息")
        return

    writer = None
    started = time.monotonic()
    deadline = started + duration if duration > 0 else None
    last_report = started
    last_index = 0

    try:
        while True:
            if deadline is not None and time.monotonic() >= deadline:
                break
            timeout = 1.0 if deadline is None else min(1.0, max(deadline - time.monotonic(), 0.0))
            frame = source.read(timeout=timeout)
            if frame is None:
                LOGGER.warning("读取超时，状态=%s", source.stats.state)
                if deadline is not None and time.monotonic() >= deadline:
                    break
                continue

            if save_path and writer is None:
                import cv2

                writer = cv2.VideoWriter(
                    save_path,
                    # OpenCV 4/5 都有这个常量；typeshed 给 cv2 的桩里没列全，
                    # 所以类型检查会报 unknown attribute（运行时正常）
                    cv2.VideoWriter_fourcc(*"mp4v"),  # pyright: ignore[reportAttributeAccessIssue]
                    max(source.stats.fps, 10.0),
                    (frame.width, frame.height),
                )
                if not writer.isOpened():
                    writer.release()
                    writer = None
                    LOGGER.error("无法打开视频输出: %s", save_path)
                    return
                LOGGER.info("开始存盘: %s (%dx%d)", save_path, frame.width, frame.height)
            if writer is not None:
                writer.write(frame.image)

            if preview:
                import cv2

                cv2.imshow("RTSP", frame.image)
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

            if deadline is not None and now >= deadline:
                break
    except KeyboardInterrupt:
        LOGGER.info("用户中断")
    finally:
        if writer is not None:
            writer.release()
        if preview:
            import cv2

            cv2.destroyAllWindows()


def main(
    *,
    url: str = URL,
    width: int = WIDTH,
    height: int = HEIGHT,
    transport: str = TRANSPORT,
    decoder: str | None = DECODER,
    preview: bool = PREVIEW,
    save_path: str | None = SAVE_PATH,
    duration: float = DURATION,
) -> int:
    """拉流并统计链路质量（可选预览/存盘）；返回进程退出码。"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    config = build_config(url=url, width=width, height=height, transport=transport, decoder=decoder)
    LOGGER.info(
        "拉流 %s（%dx%d transport=%s）",
        config.url,
        config.width,
        config.height,
        config.transport,
    )
    with Hm30VideoSource(config) as source:
        run(source, preview=preview, save_path=save_path, duration=duration)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
