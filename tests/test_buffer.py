"""对齐结果缓冲区的离线测试。

不需要飞控与图传硬件：手工构造 ``AlignedSample`` 后验证缓冲的写入/驱逐/
时间区间查询/多读者跟随/内存上限，以及两条最容易踩的坑——

* **ffmpeg 后端的帧是复用缓冲区的视图**，入缓冲必须拷贝，否则历史帧会被
  后续帧改写；
* ``between()`` 会把命中帧全部解码进内存，批量消费必须用 ``iter_between()``。

运行：

    ./.venv/Scripts/python.exe tests/test_buffer.py
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

# 让 `python tests/test_buffer.py` 与 `python -m tests.test_buffer` 都能跑
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from tests._harness import use_utf8_console  # noqa: E402

from airdrop import (  # noqa: E402
    DEFAULT_BUFFER_CAPACITY,
    DEFAULT_TELEMETRY_LAG,
    AlignedSample,
    AlignmentBuffer,
    TelemetrySnapshot,
    VideoFrame,
    capacity_for,
    raw_frame_bytes,
)


def make_sample(
    capture_timestamp: float,
    *,
    image: np.ndarray | None = None,
    lat: float = 47.0,
    lag: float = DEFAULT_TELEMETRY_LAG,
) -> AlignedSample:
    """手工构造一条对齐结果（等价于 aligner.align 的返回值）。"""
    if image is None:
        image = np.full((8, 8, 3), 128, dtype=np.uint8)
    frame = VideoFrame(
        index=1, image=image, timestamp=capture_timestamp + lag, lag=lag
    )
    return AlignedSample(
        frame=frame,
        snapshot=TelemetrySnapshot(timestamp=capture_timestamp, latitude_deg=lat),
        timestamp=capture_timestamp,
        lag=lag,
        mode="interpolate",
    )


def test_put_and_read() -> None:
    buffer = AlignmentBuffer(capacity=10)
    for i in range(3):
        record = buffer.put(make_sample(1000.0 + i, lat=47.0 + i * 0.1))
        assert record.index == i + 1, record.index
        assert record.image.shape == (8, 8, 3)
        assert record.capture_timestamp == 1000.0 + i
        assert record.received_timestamp == 1000.0 + i + DEFAULT_TELEMETRY_LAG
        assert record.lag == DEFAULT_TELEMETRY_LAG
        assert record.snapshot.latitude_deg == 47.0 + i * 0.1

    assert len(buffer) == 3
    assert buffer.indices() == (1, 3)
    assert buffer.latest_index() == 3
    assert buffer.time_span() == (1000.0, 1002.0)
    assert buffer.latest().index == 3
    assert buffer.at(2).snapshot.latitude_deg == 47.1
    assert buffer.at(99) is None, "不存在的序号应返回 None"
    print("[ok] 写入 / 按序号读取 / 序号与时间范围")


def test_capacity_evicts_oldest() -> None:
    buffer = AlignmentBuffer(capacity=5)
    for i in range(8):
        buffer.put(make_sample(1000.0 + i))

    assert len(buffer) == 5, len(buffer)
    assert buffer.indices() == (4, 8), buffer.indices()  # 保留最新的 5 条
    assert buffer.at(3) is None, "被驱逐的序号应取不到"
    assert buffer.at(4) is not None
    stats = buffer.stats
    assert stats.evicted == 3 and stats.put == 8 and stats.last_index == 8
    print("[ok] 容量满时从最旧一端驱逐（序号不回退）")


def test_images_are_not_overwritten() -> None:
    """ffmpeg 后端的帧复用同一块管道缓冲：入缓冲必须拷贝。

    这里刻意模拟 ``_stream_ffmpeg`` 的写法（同一个 bytearray 反复 reshape
    成 view），如果缓冲区存的是引用而不是副本，读回来的历史帧会全部变成
    最后一帧的颜色。
    """
    width = height = 32
    shared = bytearray(width * height * 3)  # 模拟 ffmpeg 的管道缓冲
    view = np.frombuffer(shared, dtype=np.uint8).reshape(height, width, 3)

    for storage in ("jpeg", "raw"):
        buffer = AlignmentBuffer(capacity=8, storage=storage)
        for i, value in enumerate((30, 90, 150)):
            view[:] = value  # 每帧复用同一块内存
            sample = make_sample(1000.0 + i, image=view)
            assert sample.frame.image is view, "测试前提：帧持有的是视图"
            buffer.put(sample)

        # 管道缓冲已被最后一帧改写
        assert shared[0] == 150
        tolerance = 0 if storage == "raw" else 20  # JPEG 有损
        for index, value in ((1, 30), (2, 90), (3, 150)):
            record = buffer.at(index)
            assert record is not None
            assert abs(float(record.image.mean()) - value) <= tolerance, (
                storage,
                index,
                record.image.mean(),
            )
    print("[ok] 入缓冲的画面不被后续帧覆盖（复用缓冲视图）")


def test_reader_isolation() -> None:
    # raw：读到的是内部数组的只读视图（零拷贝，防误改）
    raw = AlignmentBuffer(capacity=2, storage="raw")
    raw.put(make_sample(1000.0, image=np.full((8, 8, 3), 200, dtype=np.uint8)))
    record = raw.latest()
    assert not record.image.flags.writeable, "raw 模式应返回只读视图"
    try:
        record.image[0, 0] = 0
    except ValueError:
        pass
    else:
        raise AssertionError("raw 模式的视图不应可写")
    # 需要就地修改时先 copy()
    editable = record.copy()
    assert editable.image.flags.writeable
    editable.image[:] = 0
    assert raw.latest().image.mean() > 150, "改动副本不应影响缓冲区"

    # jpeg：每次读取都是新解码的数组，改动同样不影响缓冲区
    jpeg = AlignmentBuffer(capacity=2, storage="jpeg")
    jpeg.put(make_sample(1000.0, image=np.full((16, 16, 3), 200, dtype=np.uint8)))
    decoded = jpeg.latest()
    assert decoded.image.flags.writeable
    decoded.image[:] = 0
    assert jpeg.latest().image.mean() > 150
    print("[ok] 读者拿到的画面与缓冲区隔离（raw 只读 / jpeg 独立数组）")


def test_jpeg_vs_raw() -> None:
    gradient = np.tile(np.arange(64, dtype=np.uint8).reshape(1, 64, 1), (48, 1, 3))
    raw = AlignmentBuffer(capacity=2, storage="raw")
    jpeg = AlignmentBuffer(capacity=2, storage="jpeg", jpeg_quality=80)
    raw.put(make_sample(1000.0, image=gradient))
    jpeg.put(make_sample(1000.0, image=gradient))

    assert raw.stats.bytes == 64 * 48 * 3, raw.stats.bytes
    assert jpeg.stats.bytes < raw.stats.bytes // 4, (
        jpeg.stats.bytes,
        raw.stats.bytes,
    )
    assert np.array_equal(raw.latest().image, gradient), "raw 必须无损"
    decoded = jpeg.latest().image
    assert decoded.shape == gradient.shape
    assert np.abs(decoded.astype(int) - gradient.astype(int)).mean() < 10
    print("[ok] jpeg（体积小、有损）与 raw（无损、体积大）两条存储")


def test_between_and_iter_between() -> None:
    buffer = AlignmentBuffer(capacity=10)
    for i in range(10):
        buffer.put(make_sample(1000.0 + i))

    assert [r.index for r in buffer.between(1002.0, 1005.0)] == [3, 4, 5, 6]
    assert len(buffer.between(start=1007.0)) == 3
    assert len(buffer.between(end=1001.0)) == 2
    assert len(buffer.between()) == 10
    assert len(buffer.between(limit=4)) == 4

    # 分批迭代的结果与一次性查询一致，且内存占用与 batch 同阶
    assert [
        r.index for r in buffer.iter_between(1002.0, 1005.0, batch=2)
    ] == [3, 4, 5, 6]
    assert [r.index for r in buffer.iter_between(batch=3)] == list(range(1, 11))
    assert [r.index for r in buffer.iter_between(start=1008.0)] == [9, 10]
    assert list(buffer.iter_between(2000.0, 3000.0)) == []

    # 快照式语义：迭代开始后新写入的记录不在本次范围内
    iterator = buffer.iter_between(batch=4)
    assert next(iterator).index == 1
    buffer.put(make_sample(2000.0))
    assert [r.index for r in iterator] == list(range(2, 11))
    print("[ok] 时间区间查询（闭区间 / limit / 分批迭代 / 快照语义）")


def test_wait_new() -> None:
    buffer = AlignmentBuffer(capacity=4)
    assert buffer.wait_new(0, timeout=0.1) is None, "空缓冲应超时返回 None"

    writer = threading.Thread(
        target=lambda: (time.sleep(0.1), buffer.put(make_sample(1000.0)))
    )
    writer.start()
    record = buffer.wait_new(0, timeout=5.0)
    writer.join(timeout=2.0)
    assert record is not None and record.index == 1, record
    assert buffer.wait_new(record.index, timeout=0.1) is None, "没有更新数据了"

    for i in range(1, 6):
        buffer.put(make_sample(1000.0 + i))
    # 之前已有 1 号，共 6 条，容量 4 → 保留 3..6
    assert buffer.indices() == (3, 6), buffer.indices()
    lagging = buffer.wait_new(1, timeout=0.2)  # 游标落后于最旧记录
    assert lagging is not None and lagging.index == 6, lagging
    print("[ok] 多读者跟随 wait_new（阻塞唤醒 / 游标 / 落后被驱逐）")


def test_max_bytes_eviction() -> None:
    frame_bytes = 32 * 32 * 3
    buffer = AlignmentBuffer(
        capacity=100, storage="raw", max_bytes=frame_bytes * 3
    )
    for i in range(10):
        buffer.put(
            make_sample(
                1000.0 + i,
                image=np.full((32, 32, 3), 100 + i, dtype=np.uint8),
            )
        )
    stats = buffer.stats
    assert stats.frames == 3, stats.frames
    assert stats.bytes <= frame_bytes * 3, stats.bytes
    assert stats.evicted == 7, stats.evicted
    assert buffer.indices() == (8, 10), buffer.indices()
    print("[ok] max_bytes 字节上限驱逐（与容量上限谁先到听谁）")


def test_validation_and_helpers() -> None:
    # 容量换算：30 fps × 180 s
    assert capacity_for(30.0, 180.0) == 5400
    assert DEFAULT_BUFFER_CAPACITY == 5400
    # 720p 未压缩一帧 ≈ 2.64 MiB，5400 帧 ≈ 13.9 GiB
    assert raw_frame_bytes(1280, 720) == 2_764_800
    assert abs(raw_frame_bytes(1280, 720) * 5400 / 1024**3 - 13.90) < 0.01

    for kwargs in (
        {"capacity": 0},
        {"storage": "png"},
        {"jpeg_quality": 0},
        {"max_bytes": 0},
    ):
        try:
            AlignmentBuffer(**kwargs)
        except ValueError:
            continue
        raise AssertionError(f"{kwargs} 应当抛出 ValueError")

    buffer = AlignmentBuffer(capacity=5)
    try:
        next(buffer.iter_between(batch=0))  # 生成器体在首次 next 时才执行
    except ValueError:
        pass
    else:
        raise AssertionError("batch=0 应当抛出 ValueError")

    for i in range(3):
        buffer.put(make_sample(1000.0 + i))
    stats = buffer.stats
    assert (stats.put, stats.frames, stats.evicted, stats.last_index) == (3, 3, 0, 3)
    assert stats.storage == "jpeg" and stats.capacity == 5, stats
    assert stats.bytes > 0

    buffer.clear()
    assert len(buffer) == 0 and buffer.latest() is None
    assert buffer.indices() is None and buffer.time_span() is None
    assert buffer.at(1) is None

    try:
        buffer.put(None)
    except ValueError:
        pass
    else:
        raise AssertionError("put(None) 应当抛出 ValueError")
    print("[ok] 参数校验 / 容量换算 / clear 与统计")


def main() -> int:
    use_utf8_console()
    tests = [
        test_put_and_read,
        test_capacity_evicts_oldest,
        test_images_are_not_overwritten,
        test_reader_isolation,
        test_jpeg_vs_raw,
        test_between_and_iter_between,
        test_wait_new,
        test_max_bytes_eviction,
        test_validation_and_helpers,
    ]
    for test in tests:
        test()
    print(f"\ntest_buffer: 全部通过（{len(tests)} 组）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
