"""对齐结果缓冲的单元测试。

不依赖飞控与视频硬件：手工构造 ``AlignedSample`` / ``VideoFrame`` 后验证写入、驱逐、
时间区间迭代、多读者跟随、字节上限与参数校验，并盯住两条最容易踩的易错点——

* ffmpeg 后端的帧是复用缓冲区的视图，入缓冲必须拷贝，否则历史帧会被后续帧改写；
* 回看历史一律走 ``iter_between()``：分批解码，不会一次把 5400 帧解进内存。
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from airdrop import (
    DEFAULT_BUFFER_CAPACITY,
    DEFAULT_TELEMETRY_LAG,
    AlignedSample,
    AlignmentBuffer,
    AlignmentWriter,
    FrameTelemetryAligner,
    TelemetryBroker,
    TelemetrySnapshot,
    VideoFrame,
    capacity_for,
    raw_frame_bytes,
)


def make_frame(timestamp: float, image: np.ndarray | None = None) -> VideoFrame:
    """构造一帧（``timestamp`` 是收到时间，拍摄时刻 = timestamp - lag）。"""
    if image is None:
        image = np.full((8, 8, 3), 128, dtype=np.uint8)
    return VideoFrame(index=1, image=image, timestamp=timestamp, lag=DEFAULT_TELEMETRY_LAG)


def make_sample(
    capture_timestamp: float,
    *,
    image: np.ndarray | None = None,
    lat: float = 47.0,
) -> AlignedSample:
    """手工构造一条对齐结果（等价于 aligner.align 的返回值）。"""
    return AlignedSample(
        frame=make_frame(capture_timestamp + DEFAULT_TELEMETRY_LAG, image),
        snapshot=TelemetrySnapshot(timestamp=capture_timestamp, latitude_deg=lat),
        timestamp=capture_timestamp,
        lag=DEFAULT_TELEMETRY_LAG,
        mode="interpolate",
    )


# ----------------------------------------------------------------------
# 写入与驱逐
# ----------------------------------------------------------------------
def test_put_and_read() -> None:
    buffer = AlignmentBuffer(capacity=10)
    for i in range(3):
        record = buffer.put(make_sample(1000.0 + i, lat=47.0 + i * 0.1))
        assert record.index == i + 1
        assert record.image.shape == (8, 8, 3)
        assert record.capture_timestamp == 1000.0 + i
        assert record.received_timestamp == pytest.approx(1000.0 + i + DEFAULT_TELEMETRY_LAG)
        assert record.lag == pytest.approx(DEFAULT_TELEMETRY_LAG)
        assert record.snapshot.latitude_deg == pytest.approx(47.0 + i * 0.1)

    assert len(buffer) == 3
    assert buffer.indices() == (1, 3)
    assert buffer.latest_index() == 3
    assert buffer.latest().index == 3
    assert buffer.at(2).snapshot.latitude_deg == pytest.approx(47.1)
    assert buffer.at(99) is None


def test_capacity_evicts_oldest() -> None:
    buffer = AlignmentBuffer(capacity=5)
    for i in range(8):
        buffer.put(make_sample(1000.0 + i))

    assert len(buffer) == 5
    assert buffer.indices() == (4, 8)  # 保留最新 5 条，序号不回退
    assert buffer.at(3) is None
    assert buffer.at(4) is not None

    stats = buffer.stats
    assert stats.evicted == 3
    assert stats.put == 8
    assert stats.last_index == 8


def test_max_bytes_eviction() -> None:
    frame_bytes = 32 * 32 * 3
    buffer = AlignmentBuffer(capacity=100, storage="raw", max_bytes=frame_bytes * 3)
    for i in range(10):
        buffer.put(
            make_sample(
                1000.0 + i,
                image=np.full((32, 32, 3), 100 + i, dtype=np.uint8),
            )
        )

    stats = buffer.stats
    assert stats.frames == 3
    assert stats.bytes <= frame_bytes * 3
    assert stats.evicted == 7
    assert buffer.indices() == (8, 10)


# ----------------------------------------------------------------------
# 图像持有与读者隔离
# ----------------------------------------------------------------------
@pytest.mark.parametrize("storage", ["jpeg", "raw"])
def test_images_are_not_overwritten(storage: str) -> None:
    """ffmpeg 后端的帧复用同一块管道缓冲：入缓冲必须拷贝。

    这里刻意模拟 ``_stream_ffmpeg`` 的写法（同一个 bytearray 反复 reshape 成
    view）；如果缓冲区存的是引用而不是副本，读回来的历史帧会全部变成最后一帧
    的颜色。
    """
    width = height = 32
    shared = bytearray(width * height * 3)  # 模拟 ffmpeg 的管道缓冲
    view = np.frombuffer(shared, dtype=np.uint8).reshape(height, width, 3)

    buffer = AlignmentBuffer(capacity=8, storage=storage)
    for i, value in enumerate((30, 90, 150)):
        view[:] = value  # 每帧复用同一块内存
        sample = make_sample(1000.0 + i, image=view)
        assert sample.frame.image is view, "测试前提：帧持有的是视图"
        buffer.put(sample)

    assert shared[0] == 150, "管道缓冲已被最后一帧改写"
    tolerance = 0 if storage == "raw" else 20  # JPEG 有损
    for index, value in ((1, 30), (2, 90), (3, 150)):
        record = buffer.at(index)
        assert record is not None
        assert float(record.image.mean()) == pytest.approx(value, abs=tolerance)


def test_raw_reader_gets_readonly_view() -> None:
    raw = AlignmentBuffer(capacity=2, storage="raw")
    raw.put(make_sample(1000.0, image=np.full((8, 8, 3), 200, dtype=np.uint8)))
    record = raw.latest()
    assert not record.image.flags.writeable, "raw 模式应返回只读视图"
    with pytest.raises(ValueError):
        record.image[0, 0] = 0

    # 需要就地修改时先 copy()
    editable = record.copy()
    assert editable.image.flags.writeable
    editable.image[:] = 0
    assert raw.latest().image.mean() > 150, "改动副本不应影响缓冲区"


def test_jpeg_reader_gets_independent_array() -> None:
    jpeg = AlignmentBuffer(capacity=2, storage="jpeg")
    jpeg.put(make_sample(1000.0, image=np.full((16, 16, 3), 200, dtype=np.uint8)))
    decoded = jpeg.latest()
    assert decoded.image.flags.writeable
    decoded.image[:] = 0
    assert jpeg.latest().image.mean() > 150


def test_jpeg_vs_raw() -> None:
    gradient = np.tile(np.arange(64, dtype=np.uint8).reshape(1, 64, 1), (48, 1, 3))
    raw = AlignmentBuffer(capacity=2, storage="raw")
    jpeg = AlignmentBuffer(capacity=2, storage="jpeg", jpeg_quality=80)
    raw.put(make_sample(1000.0, image=gradient))
    jpeg.put(make_sample(1000.0, image=gradient))

    assert raw.stats.bytes == 64 * 48 * 3
    assert jpeg.stats.bytes < raw.stats.bytes // 4
    assert np.array_equal(raw.latest().image, gradient), "raw 必须无损"

    decoded = jpeg.latest().image
    assert decoded.shape == gradient.shape
    assert np.abs(decoded.astype(int) - gradient.astype(int)).mean() < 10


# ----------------------------------------------------------------------
# 读取接口
# ----------------------------------------------------------------------
def test_iter_between() -> None:
    buffer = AlignmentBuffer(capacity=10)
    for i in range(10):
        buffer.put(make_sample(1000.0 + i))

    # 闭区间；分批（batch=2/3）与一次取完结果一致
    assert [r.index for r in buffer.iter_between(1002.0, 1005.0, batch=2)] == [3, 4, 5, 6]
    assert [r.index for r in buffer.iter_between(1002.0, 1005.0)] == [3, 4, 5, 6]
    assert [r.index for r in buffer.iter_between(start=1007.0)] == [8, 9, 10]
    assert [r.index for r in buffer.iter_between(end=1001.0)] == [1, 2]
    assert [r.index for r in buffer.iter_between()] == list(range(1, 11))
    assert [r.index for r in buffer.iter_between(batch=3)] == list(range(1, 11))
    assert list(buffer.iter_between(2000.0, 3000.0)) == []


def test_iter_between_is_snapshot() -> None:
    """迭代是快照式的：开始之后新写入的记录不在本次范围内。"""
    buffer = AlignmentBuffer(capacity=10)
    for i in range(10):
        buffer.put(make_sample(1000.0 + i))

    iterator = buffer.iter_between(batch=4)
    assert next(iterator).index == 1
    buffer.put(make_sample(2000.0))
    assert [r.index for r in iterator] == list(range(2, 11))


def test_wait_new() -> None:
    buffer = AlignmentBuffer(capacity=4)
    assert buffer.wait_new(0, timeout=0.1) is None, "空缓冲应超时返回 None"

    writer = threading.Thread(target=lambda: (time.sleep(0.1), buffer.put(make_sample(1000.0))))
    writer.start()
    record = buffer.wait_new(0, timeout=5.0)
    writer.join(timeout=2.0)
    assert record is not None and record.index == 1
    assert buffer.wait_new(record.index, timeout=0.1) is None

    for i in range(1, 6):
        buffer.put(make_sample(1000.0 + i))
    # 之前已有 1 号，共 6 条，容量 4 → 保留 3..6
    assert buffer.indices() == (3, 6)
    # 游标落后到已被驱逐的记录：返回当前保留的最早一条（不会被无休止等待，
    # 但中间被驱逐的 1、2 在此路径上不可见——需要完整消费请用 iter_between）
    lagging = buffer.wait_new(1, timeout=0.2)
    assert lagging is not None and lagging.index == 3


# ----------------------------------------------------------------------
# 拉流源 → 缓冲的装配
# ----------------------------------------------------------------------
def test_alignment_writer() -> None:
    """AlignmentWriter：正常帧全写，取不到遥测才跳过。"""
    broker = TelemetryBroker(history_maxlen=200, history_interval=0.0)
    buffer = AlignmentBuffer(capacity=10)
    writer = AlignmentWriter(buffer, FrameTelemetryAligner(broker))
    assert writer.buffer is buffer

    # 还没有遥测历史：该帧被跳过（唯一允许丢画面、且原因明确的场合）
    writer(make_frame(1000.0))
    assert (writer.written, writer.skipped) == (0, 1)
    assert len(buffer) == 0

    # 注入一段确定性历史后，逐帧都写进去
    base = time.time() - 1.0
    for i in range(10):
        broker._history.append(
            TelemetrySnapshot(timestamp=base + i * 0.1, latitude_deg=47.0 + i * 0.001)
        )
    frames = [make_frame(base + 0.9 + i * 0.02) for i in range(5)]
    for frame in frames:
        writer(frame)

    assert writer.stats() == (5, 1)
    assert buffer.indices() == (1, 5)

    # 缓冲里存的是"拍摄时刻"的遥测，而不是收到帧的时刻
    last = buffer.latest()
    expected = frames[-1].timestamp - DEFAULT_TELEMETRY_LAG
    assert last.capture_timestamp == pytest.approx(expected)
    assert last.snapshot.timestamp == pytest.approx(expected)


# ----------------------------------------------------------------------
# 参数校验 / 换算 / 统计
# ----------------------------------------------------------------------
def test_capacity_helpers() -> None:
    assert capacity_for(30.0, 180.0) == 5400
    assert DEFAULT_BUFFER_CAPACITY == 5400
    # 720p 未压缩一帧 ≈ 2.64 MiB，5400 帧 ≈ 13.9 GiB
    assert raw_frame_bytes(1280, 720) == 2_764_800
    assert raw_frame_bytes(1280, 720) * 5400 / 1024**3 == pytest.approx(13.90, abs=0.01)


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"capacity": 0}, id="容量为 0"),
        pytest.param({"storage": "png"}, id="未知存储"),
        pytest.param({"jpeg_quality": 0}, id="质量越界"),
        pytest.param({"max_bytes": 0}, id="字节上限为 0"),
    ],
)
def test_rejects_bad_params(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        AlignmentBuffer(**kwargs)


def test_rejects_bad_iter_batch() -> None:
    buffer = AlignmentBuffer(capacity=5)
    with pytest.raises(ValueError):
        next(buffer.iter_between(batch=0))  # 生成器体在首次 next 时才执行


def test_put_none_is_rejected() -> None:
    with pytest.raises(ValueError):
        AlignmentBuffer(capacity=5).put(None)


def test_stats_and_clear() -> None:
    buffer = AlignmentBuffer(capacity=5)
    for i in range(3):
        buffer.put(make_sample(1000.0 + i))

    stats = buffer.stats
    assert (stats.put, stats.frames, stats.evicted, stats.last_index) == (3, 3, 0, 3)
    assert stats.storage == "jpeg"
    assert stats.capacity == 5
    assert stats.bytes > 0

    buffer.clear()
    assert len(buffer) == 0
    assert buffer.latest() is None
    assert buffer.indices() is None
    assert buffer.at(1) is None
