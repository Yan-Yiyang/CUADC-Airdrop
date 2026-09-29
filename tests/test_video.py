"""RTSP 视频拉流的集成测试。

不需要真实视频硬件：:mod:`tests.conftest` 用 ffmpeg 把 testsrc 编成 H.264/MPEG-TS
推到本地 UDP，再让 :mod:`airdrop.video` 去拉。这条路径与真实 RTSP 一致——同一套
FFmpeg 解复用、同一套低延迟 flag、同一套断流判定与重连逻辑。

标记说明：``@pytest.mark.stream`` 需要 ffmpeg 与 UDP 端口 51234；
``@pytest.mark.network`` 走网络错误路径（会等超时或连接被拒）。
"""

from __future__ import annotations

import time
from dataclasses import replace
from itertools import pairwise

import numpy as np
import pytest

from airdrop import DEFAULT_TELEMETRY_LAG, Hm30VideoSource, VideoConfig

pytestmark = pytest.mark.stream


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


def test_stream_basic(live_source) -> None:
    source = live_source()
    frames = read_some(source, 20)
    assert len(frames) == 20

    image = frames[-1].image
    assert image.shape == (360, 640, 3)
    assert image.dtype == np.uint8
    assert image.flags.writeable, "管道帧应当是可直接写入的数组"
    assert image.std() > 10, f"画面不是彩色内容 std={image.std():.1f}"
    assert all(a.index < b.index for a, b in pairwise(frames)), "帧序号未递增"
    assert all(a.timestamp <= b.timestamp for a, b in pairwise(frames)), "时间戳未递增"

    drain_for(source, 1.6)
    stats = source.stats
    assert 20.0 < stats.fps < 40.0, f"帧率统计异常 fps={stats.fps:.1f}"
    assert stats.state == "streaming"
    assert 0 <= stats.dropped < stats.frames
    assert 0.0 <= source.latest().age < 2.0


def test_stream_scaling(live_source) -> None:
    """缩放到目标尺寸（拉到推理分辨率，降低管道带宽）。"""
    source = live_source(width=320, height=180)
    frames = read_some(source, 5)
    assert len(frames) == 5
    assert frames[-1].image.shape == (180, 320, 3)


def test_frame_carries_telemetry_lag(live_source) -> None:
    """帧必须携带链路延时——帧-遥测时间对齐的前提（见 airdrop.alignment）。"""
    source = live_source(telemetry_lag=0.2)
    frames = read_some(source, 3)
    assert frames

    for frame in frames:
        assert frame.lag == pytest.approx(0.2)
        assert frame.capture_timestamp == pytest.approx(frame.timestamp - 0.2)
        assert frame.capture_timestamp < frame.timestamp
        assert frame.capture_age > frame.age

    # 默认配置的链路延时就是 150ms
    assert VideoConfig().telemetry_lag == pytest.approx(DEFAULT_TELEMETRY_LAG)


def test_sink_receives_every_frame(live_source) -> None:
    """每帧都必须经过 sink——这是"一帧都不能丢"的保证点。

    同时验证：坏 sink 抛异常既不打断拉流、也不影响别的 sink；而 ``read()``
    那条实时路径允许丢帧，两者互不影响。
    """
    seen: list[int] = []

    def counting_sink(frame) -> None:
        seen.append(frame.index)

    def failing_sink(frame) -> None:
        raise RuntimeError("故意失败")

    def configure(source: Hm30VideoSource) -> None:
        # 必须在 start() 之前挂上，否则首帧之后才生效
        source.add_sink(counting_sink)
        source.add_sink(failing_sink)

    source = live_source(configure=configure)
    assert len(source.sinks) == 2

    # 故意慢消费（每帧拖 100ms，远低于 30fps）：实时读取必然丢帧，
    # 而 sink 一帧都不会少
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        source.read(timeout=0.5)
        time.sleep(0.1)

    source.stop()
    # 采集线程已经退出，此刻 stats 与 seen 都是最终值
    # （在流还没停时取快照，会与在途帧差 1~2 帧）
    stats = source.stats
    assert stats.frames > 10
    assert len(seen) == stats.frames, (len(seen), stats.frames)
    assert seen == list(range(1, stats.frames + 1)), "sink 收到的帧号必须连续、无跳号"
    assert stats.dropped > 0, "这种读法本该丢帧，否则本用例失去意义"

    source.remove_sink(counting_sink)
    assert counting_sink not in source.sinks
    assert failing_sink in source.sinks


def test_stream_reconnect(make_source, sender, stream_config) -> None:
    """断流判定与 stop() 脱身能力。

    ``-timeout`` 是让 ffmpeg 在断流时退出的关键 flag；缺了它，管道读会
    永久阻塞，拉流线程既检测不到断流，stop() 也退不掉。
    """
    config = replace(stream_config, read_timeout=2.0, reconnect_delay=1.0, max_reconnect_delay=2.0)
    source = make_source(config)
    source.start()
    time.sleep(0.8)
    process = sender.start()
    assert source.wait_ready(timeout=12.0), "首次连接失败"

    sender.stop(process)  # 掐掉流：UDP 没有 EOF，只能靠 -timeout 判定
    wait_reconnect(source)
    assert source.stats.reconnects >= 1, (
        f"未检测到断流 reconnects={source.stats.reconnects} state={source.stats.state}"
    )

    sender.start()
    assert source.wait_ready(timeout=20.0), (
        f"未自动恢复 state={source.stats.state} err={source.stats.last_error}"
    )

    # 管道里可能正卡着一次读，stop() 必须能杀掉子进程把它解开
    started = time.monotonic()
    source.stop()
    elapsed = time.monotonic() - started
    assert elapsed < 5.5, f"stop() 被阻塞的管道读卡住，耗时 {elapsed:.2f}s"
    assert source._process is None, "ffmpeg 子进程未回收"


@pytest.mark.network
def test_invalid_url_is_reported(make_source) -> None:
    """地址无效要显式报错，而不是静默重试到天荒地老。

    用一个必然连不上的地址：ffmpeg 会很快失败退出，拉流源应把原因（含 URL）
    写进 ``stats.last_error``，并保持"一帧都没收到"的事实。
    """
    bad_url = "rtsp://127.0.0.1:1/nonexistent"
    source = make_source(
        VideoConfig(
            url=bad_url,
            width=640,
            height=360,
            read_timeout=1.0,
            reconnect_delay=0.5,
            max_reconnect_delay=1.0,
        )
    )
    source.start()
    deadline = time.monotonic() + 25.0
    while time.monotonic() < deadline and not source.stats.last_error:
        time.sleep(0.2)

    stats = source.stats
    assert stats.last_error, "地址无效却没有给出任何错误信息"
    assert bad_url in stats.last_error
    assert stats.frames == 0, "地址无效却收到了帧？"

    started = time.monotonic()
    source.stop()
    assert time.monotonic() - started < 6.0, "stop() 被卡住"


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"width": 0}, id="宽度为 0"),
        pytest.param({"height": -1}, id="高度为负"),
        pytest.param({"transport": "sctp"}, id="未知传输层"),
        pytest.param({"telemetry_lag": -0.1}, id="负延时"),
    ],
)
def test_config_validation(kwargs: dict) -> None:
    """尺寸/传输层非法时立刻报错，不去猜、不去探测。"""
    with pytest.raises(ValueError):
        VideoConfig(**kwargs).validated()


def test_source_is_not_started_eagerly() -> None:
    """构造拉流源本身无副作用：线程不启动、不占端口。"""
    source = Hm30VideoSource(VideoConfig(url="udp://127.0.0.1:1"))
    assert not source.running
    assert source.stats.frames == 0
    assert source.latest() is None

    with pytest.raises(ValueError, match="输出尺寸"):
        Hm30VideoSource(VideoConfig(width=0))


def test_stop_is_clean(make_source) -> None:
    source = make_source(VideoConfig(url="udp://127.0.0.1:1", width=640, height=360))
    source.start()
    time.sleep(0.5)
    started = time.monotonic()
    source.stop()
    assert time.monotonic() - started < 6.0, "stop() 耗时过长"
    assert not source.running, "拉流线程未回收"
    assert source.stats.state == "stopped"
