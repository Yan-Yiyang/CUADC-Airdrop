'思翼 HM30 图传拉流离线测试。\n\n运行：\n\n    ./.venv/Scripts/python.exe tests/test_video.py'

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

# 让 `python tests/test_video.py` 与 `python -m tests.test_video` 都能跑
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from tests._harness import use_utf8_console  # noqa: E402

from airdrop.video import (  # noqa: E402
    DEFAULT_TELEMETRY_LAG,
    Hm30VideoSource,
    VideoConfig,
    _find_ffmpeg,
    probe_stream,
)

FFMPEG = _find_ffmpeg()
PORT = 51234
URL = f"udp://127.0.0.1:{PORT}"
SRC_W, SRC_H = 640, 360
SRC_FPS = 30


def start_sender() -> subprocess.Popen:
    """起一个实时 H.264/MPEG-TS over UDP 发送端（等价于 HM30 的图传源）。"""
    return subprocess.Popen(
        [
            FFMPEG, "-hide_banner", "-loglevel", "error",
            "-re",
            "-f", "lavfi",
            "-i", f"testsrc=size={SRC_W}x{SRC_H}:rate={SRC_FPS}",
            "-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency",
            "-g", "15", "-pix_fmt", "yuv420p",
            "-f", "mpegts", URL,
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def stop_sender(sender: subprocess.Popen | None) -> None:
    if sender is not None and sender.poll() is None:
        sender.terminate()
        sender.wait(timeout=5)


def read_some(source: Hm30VideoSource, count: int, timeout: float = 3.0) -> list:
    frames = []
    while len(frames) < count:
        frame = source.read(timeout=timeout)
        if frame is None:
            break
        frames.append(frame)
    return frames


def drain_for(source: Hm30VideoSource, seconds: float) -> int:
    """持续消费指定时长，让 1 秒滑窗的帧率统计产生有效读数。"""
    deadline = time.monotonic() + seconds
    seen = 0
    while time.monotonic() < deadline:
        if source.read(timeout=1.0) is not None:
            seen += 1
    return seen


def wait_reconnect(source: Hm30VideoSource, timeout: float = 15.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and source.stats.reconnects == 0:
        time.sleep(0.2)


def test_opencv_backend() -> None:
    source = Hm30VideoSource(
        VideoConfig(url=URL, backend="opencv", open_timeout=8.0, read_timeout=3.0)
    )
    sender = None
    try:
        source.start()
        time.sleep(0.8)  # 先让接收端 bind 住 UDP 端口，再起发送端
        sender = start_sender()

        assert source.wait_ready(timeout=12.0), "首帧未在超时内到达"
        frames = read_some(source, 20)
        assert len(frames) == 20, f"只读到 {len(frames)} 帧"

        image = frames[-1].image
        assert image.shape == (SRC_H, SRC_W, 3), image.shape
        assert image.dtype == np.uint8, image.dtype
        assert image.std() > 10, f"画面不是彩色内容 std={image.std():.1f}"
        assert all(b.index > a.index for a, b in zip(frames, frames[1:])), "帧序号未递增"
        assert all(
            b.timestamp >= a.timestamp for a, b in zip(frames, frames[1:])
        ), "时间戳未递增"

        drain_for(source, 1.6)
        stats = source.stats
        assert 20.0 < stats.fps < 40.0, f"帧率统计异常 fps={stats.fps:.1f}"
        assert stats.state == "streaming", stats.state
        assert 0 <= stats.dropped < stats.frames, stats.dropped
        assert 0.0 <= source.latest().age < 2.0, source.latest().age
    finally:
        stop_sender(sender)
        source.stop()
    print("[ok] OpenCV 后端（cv2.VideoCapture + 内置 FFmpeg）")


def test_ffmpeg_backend() -> None:
    source = Hm30VideoSource(
        VideoConfig(
            url=URL, backend="ffmpeg", width=SRC_W, height=SRC_H, open_timeout=8.0
        )
    )
    sender = None
    try:
        source.start()
        time.sleep(0.8)
        sender = start_sender()

        assert source.wait_ready(timeout=12.0), "首帧未在超时内到达"
        frames = read_some(source, 20)
        assert len(frames) == 20, f"只读到 {len(frames)} 帧"

        image = frames[-1].image
        assert image.shape == (SRC_H, SRC_W, 3), image.shape
        assert image.flags.writeable, "管道帧应当是可直接写入的数组"
        assert image.std() > 10, f"画面不是彩色内容 std={image.std():.1f}"
        assert source.stats.state == "streaming", source.stats.state

        drain_for(source, 1.6)
        assert 20.0 < source.stats.fps < 40.0, source.stats.fps
    finally:
        stop_sender(sender)
        source.stop()
    print("[ok] ffmpeg 后端（子进程 + rawvideo 管道）")


def test_ffmpeg_scaling() -> None:
    source = Hm30VideoSource(
        VideoConfig(
            url=URL, backend="ffmpeg", width=320, height=180, open_timeout=8.0
        )
    )
    sender = None
    try:
        source.start()
        time.sleep(0.8)
        sender = start_sender()
        assert source.wait_ready(timeout=12.0), "首帧未在超时内到达"
        frames = read_some(source, 5)
        assert len(frames) == 5, len(frames)
        assert frames[-1].image.shape == (180, 320, 3), frames[-1].image.shape
    finally:
        stop_sender(sender)
        source.stop()
    print("[ok] ffmpeg 后端缩放（拉到推理分辨率，降低管道带宽）")


def test_probe() -> None:
    sender = start_sender()
    try:
        time.sleep(1.5)
        info = probe_stream(URL, timeout=8.0)
        assert info is not None, "探流失败"
        assert (info.width, info.height) == (SRC_W, SRC_H), (info.width, info.height)
        assert info.open_seconds > 0, info.open_seconds
    finally:
        stop_sender(sender)
    print("[ok] 探流 probe_stream（分辨率/耗时）")


def test_opencv_reconnect() -> None:
    """断流重连：HM30 链路抖动是常态。

    这条用例盯的是 OpenCV 后端的一个坑——``CAP_PROP_READ_TIMEOUT_MSEC``
    若不通过 ``open()`` 的 params 传入，断流后 read 会阻塞满 30 秒默认超时。
    """
    source = Hm30VideoSource(
        VideoConfig(
            url=URL,
            backend="opencv",
            open_timeout=5.0,
            read_timeout=2.0,
            reconnect_delay=1.0,
            max_reconnect_delay=2.0,
        )
    )
    sender = start_sender()
    try:
        source.start()
        assert source.wait_ready(timeout=12.0), "首次连接失败"

        # UDP 没有 EOF，只能靠读超时判定断流
        stop_sender(sender)
        sender = None
        wait_reconnect(source)
        assert source.stats.reconnects >= 1, (
            f"未检测到断流 reconnects={source.stats.reconnects} "
            f"state={source.stats.state}"
        )

        sender = start_sender()
        assert source.wait_ready(timeout=20.0), (
            f"未自动恢复 state={source.stats.state} err={source.stats.last_error}"
        )
    finally:
        stop_sender(sender)
        source.stop()
    print("[ok] OpenCV 后端断流重连（读超时必须走 open params）")


def test_ffmpeg_reconnect() -> None:
    """ffmpeg 后端的断流判定与 stop() 脱身能力。

    ``-timeout`` 是让 ffmpeg 在断流时退出的关键 flag；缺了它，管道读会
    永久阻塞，拉流线程既检测不到断流，stop() 也退不掉。
    """
    source = Hm30VideoSource(
        VideoConfig(
            url=URL,
            backend="ffmpeg",
            width=SRC_W,
            height=SRC_H,
            open_timeout=5.0,
            read_timeout=2.0,
            reconnect_delay=1.0,
            max_reconnect_delay=2.0,
        )
    )
    sender = start_sender()
    elapsed = 0.0
    try:
        source.start()
        assert source.wait_ready(timeout=12.0), "首次连接失败"

        stop_sender(sender)
        sender = None
        wait_reconnect(source)
        assert source.stats.reconnects >= 1, (
            f"未检测到断流 reconnects={source.stats.reconnects} "
            f"state={source.stats.state}"
        )

        sender = start_sender()
        assert source.wait_ready(timeout=20.0), (
            f"未自动恢复 state={source.stats.state} err={source.stats.last_error}"
        )
    finally:
        stop_sender(sender)
        # 管道里可能正卡着一次读，stop() 必须能杀掉子进程把它解开
        started = time.monotonic()
        source.stop()
        elapsed = time.monotonic() - started
    assert elapsed < 5.5, f"stop() 被阻塞的管道读卡住，耗时 {elapsed:.2f}s"
    assert source._process is None, "ffmpeg 子进程未回收"
    print("[ok] ffmpeg 后端断流重连 + stop() 能从阻塞的管道读脱身")


def test_frame_lag_on_live_stream() -> None:
    """真实拉流路径下，帧必须携带链路延时——帧-遥测时间对齐的前提。

    见 :mod:`airdrop.alignment`：拿到帧后要用
    ``frame.capture_timestamp``（= 收到时间 - lag）去查遥测，而不是
    ``frame.timestamp``。
    """
    source = Hm30VideoSource(
        VideoConfig(
            url=URL,
            backend="ffmpeg",
            width=SRC_W,
            height=SRC_H,
            open_timeout=8.0,
            telemetry_lag=0.2,
        )
    )
    sender = None
    try:
        source.start()
        time.sleep(0.8)
        sender = start_sender()
        assert source.wait_ready(timeout=12.0), "首帧未在超时内到达"

        frames = read_some(source, 3)
        assert frames, "未读到帧"
        for frame in frames:
            assert abs(frame.lag - 0.2) < 1e-12, frame.lag
            assert abs(
                frame.capture_timestamp - (frame.timestamp - 0.2)
            ) < 1e-9, frame.capture_timestamp
            assert frame.capture_timestamp < frame.timestamp
            assert frame.capture_age > frame.age
    finally:
        stop_sender(sender)
        source.stop()

    # 默认配置的链路延时就是 150ms
    default = Hm30VideoSource(VideoConfig(url=URL))
    assert abs(default.config.telemetry_lag - DEFAULT_TELEMETRY_LAG) < 1e-12
    print("[ok] 真实流帧携带链路延时（拍摄时刻 = 收到时间 - lag，默认 150ms）")


def test_stop_is_clean() -> None:
    source = Hm30VideoSource(VideoConfig(url=URL, backend="opencv", read_timeout=1.0))
    source.start()
    time.sleep(0.5)
    started = time.monotonic()
    source.stop()
    assert time.monotonic() - started < 6.0, "stop() 耗时过长"
    assert not source.running, "拉流线程未回收"
    assert source.stats.state == "stopped", source.stats.state
    print("[ok] stop() 干净退出（线程回收/状态复位）")


def main() -> int:
    use_utf8_console()
    assert FFMPEG, "找不到 ffmpeg，无法搭建本地测试流"
    print(f"ffmpeg: {FFMPEG}")
    print(f"测试流: {URL}")
    tests = [
        test_opencv_backend,
        test_ffmpeg_backend,
        test_ffmpeg_scaling,
        test_probe,
        test_opencv_reconnect,
        test_ffmpeg_reconnect,
        test_frame_lag_on_live_stream,
        test_stop_is_clean,
    ]
    for test in tests:
        test()
    print(f"\ntest_video: 全部通过（{len(tests)} 组）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
