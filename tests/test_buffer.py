"""对齐结果缓冲的单元测试。

不依赖飞控与图传硬件：手工构造 ``AlignedSample`` / ``VideoFrame`` 后验证写入、驱逐、
时间区间迭代、多读者跟随、字节上限与参数校验，并盯住两条最容易踩的易错点——

* ffmpeg 后端的帧是复用缓冲区的视图，入缓冲必须拷贝，否则历史帧会被后续帧改写；
* 回看历史一律走 ``iter_between()``：分批解码，不会一次把 5400 帧解进内存。
"""

from __future__ import annotations

import threading
import time
from queue import Queue

import numpy as np
import pytest

from airdrop import (
    DEFAULT_BUFFER_CAPACITY,
    DEFAULT_TELEMETRY_LAG,
    AlignedSample,
    AlignmentBuffer,
    AlignmentWriter,
    BufferedFrame,
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
@pytest.mark.parametrize("storage", ["jpeg", "raw"])
def test_freeze_retains_unread_frames_without_copying_images(storage) -> None:
    source = AlignmentBuffer(capacity=3, storage=storage)
    for index in range(3):
        source.put(make_sample(1000 + index), materialize=False)
    frozen = source.freeze(after_index=1)
    assert frozen.latest_index() == 3
    assert frozen._records[0] is source._records[1]
    assert frozen._records[0].payload is source._records[1].payload
    for index in range(3, 8):
        source.put(make_sample(1000 + index), materialize=False)
    assert source.latest_index() == 8
    assert frozen.latest_index() == 3
    assert [frame.index for frame in frozen.iter_between()] == [2, 3]
    with pytest.raises(RuntimeError, match="驱逐"):
        source.freeze(after_index=1)


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


def test_iter_between_survives_concurrent_eviction() -> None:
    """迭代途中并发写入把未取到的记录驱逐了，**也不能静默截断**。

    原来只固定了上界 ``latest_index()``，每批仍去读当前的 ``self._records``：
    游标被驱逐时无声跳过整段；保留下来的记录又全都 ``> last`` 时中途 ``return``，
    调用方拿到一个"看起来完整"的短列表，没有任何告警或计数——而这个方法的
    用途恰恰是"回看历史"（天生是慢消费者），容量打满时必然踩到。
    """
    buffer = AlignmentBuffer(capacity=5)
    for i in range(5):  # 索引 1..5，正好装满
        buffer.put(make_sample(1000.0 + i))

    iterator = buffer.iter_between(batch=1)
    assert next(iterator).index == 1
    # 迭代器还握着游标时，写者把 2..4 驱逐掉（容量 5，只剩 5 和新来的）
    for i in range(5, 20):
        buffer.put(make_sample(1000.0 + i))

    rest = [r.index for r in iterator]
    assert rest == [2, 3, 4, 5], f"快照式迭代应把开始时保留的记录都吐出来，实际 {rest}"


def test_iter_between_honours_start_and_end() -> None:
    buffer = AlignmentBuffer(capacity=10)
    for i in range(10):  # capture 1000..1009，索引 1..10
        buffer.put(make_sample(1000.0 + i))
    assert [r.index for r in buffer.iter_between(start=1002.0, end=1004.0)] == [3, 4, 5]
    assert [r.index for r in buffer.iter_between(start=None, end=1001.0)] == [1, 2]
    assert [r.index for r in buffer.iter_between(start=1008.0)] == [9, 10]
    assert [r.index for r in buffer.iter_between()] == list(range(1, 11))


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


class _WaitObservedCondition(threading.Condition):
    """测试用 Condition：在线程进入 wait 边界时发出通知。"""

    def __init__(self, lock: threading.RLock, entered: Queue[None]) -> None:
        super().__init__(lock)
        self.entered = entered

    def wait(self, timeout: float | None = None) -> bool:
        self.entered.put(None)
        return super().wait(timeout)


class _UnboundedObservedCondition(_WaitObservedCondition):
    """忽略超时，让测试只能靠显式通知结束等待。"""

    def wait(self, timeout: float | None = None) -> bool:
        self.entered.put(None)
        return super().wait(None)


def test_wait_new_pre_cancelled_returns_none_even_when_data_exists() -> None:
    buffer = AlignmentBuffer(capacity=4)
    buffer.put(make_sample(1000.0))
    cancel_event = threading.Event()
    cancel_event.set()

    assert buffer.wait_new(cancel_event=cancel_event) is None
    assert buffer.wait_new() is not None


def test_wait_new_cancel_notifies_an_infinite_waiter() -> None:
    buffer = AlignmentBuffer(capacity=4)
    entered: Queue[None] = Queue()
    buffer._condition = _UnboundedObservedCondition(buffer._lock, entered)
    cancel_event = threading.Event()
    result: list[BufferedFrame | None] = []
    waiter = threading.Thread(
        target=lambda: result.append(buffer.wait_new(timeout=None, cancel_event=cancel_event))
    )
    waiter.start()
    assert entered.get(timeout=2) is None
    cancel_event.set()
    buffer.wake_waiters()
    waiter.join(timeout=2)

    assert not waiter.is_alive()
    assert result == [None]


def test_wait_new_cancellation_at_wait_boundary_cannot_lose_wakeup() -> None:
    buffer = AlignmentBuffer(capacity=4)
    entered = threading.Event()
    release_wait = threading.Event()
    cancel_set = threading.Event()

    class BoundaryCondition(threading.Condition):
        def wait(self, timeout: float | None = None) -> bool:
            entered.set()
            assert release_wait.wait(2)
            return super().wait(timeout)

    buffer._condition = BoundaryCondition(buffer._lock)
    cancel_event = threading.Event()
    result: list[BufferedFrame | None] = []
    waiter = threading.Thread(
        target=lambda: result.append(buffer.wait_new(timeout=None, cancel_event=cancel_event))
    )
    waiter.start()
    assert entered.wait(2)

    def cancel() -> None:
        cancel_event.set()
        cancel_set.set()
        buffer.wake_waiters()

    notifier = threading.Thread(target=cancel)
    notifier.start()
    assert cancel_set.wait(2)
    release_wait.set()
    waiter.join(timeout=2)
    notifier.join(timeout=2)

    assert not waiter.is_alive() and not notifier.is_alive()
    assert result == [None]


def test_wait_new_cancels_only_one_reader_and_buffer_remains_writable() -> None:
    buffer = AlignmentBuffer(capacity=4)
    entered: Queue[None] = Queue()
    buffer._condition = _WaitObservedCondition(buffer._lock, entered)
    cancel_event = threading.Event()
    cancelled: list[BufferedFrame | None] = []
    continuing: list[BufferedFrame | None] = []
    reader_a = threading.Thread(
        target=lambda: cancelled.append(buffer.wait_new(after_index=0, cancel_event=cancel_event))
    )
    reader_b = threading.Thread(target=lambda: continuing.append(buffer.wait_new(after_index=0)))
    reader_a.start()
    reader_b.start()
    assert entered.get(timeout=2) is None
    assert entered.get(timeout=2) is None

    cancel_event.set()
    buffer.wake_waiters()
    reader_a.join(timeout=2)
    assert not reader_a.is_alive()
    assert reader_b.is_alive()
    buffer.put(make_sample(1000.0))
    reader_b.join(timeout=2)
    next_frame = buffer.put(make_sample(1001.0))

    assert not reader_b.is_alive()
    assert cancelled == [None]
    assert continuing[0] is not None and continuing[0].index == 1
    assert next_frame is not None and next_frame.index == 2


def test_wait_new_decodes_outside_the_lock() -> None:
    """``wait_new`` 解码画面时**不得持有**那把 Condition 锁（采集线程也要拿它）。

    720p 的 ``cv2.imdecode`` 要几毫秒；放在锁内等于让采集线程的 ``put()`` 替
    读者的解码买单。实测（720p）：锁内解码时拿锁等待 p95 4.5ms / 最大 5.1ms，
    锁外解码后 p95 0.006ms / 最大 0.018ms。

    这里不用计时判定（会 flaky），而是在 ``_materialize`` 里让**另一个线程**试着
    拿锁：拿得到就说明读线程当时没持锁。
    """
    buffer = AlignmentBuffer(capacity=8)
    buffer.put(make_sample(1000.0))
    original = AlignmentBuffer.__dict__["_materialize"]  # 取 staticmethod 对象本身
    held: list[bool] = []

    def probe(stored):
        got: list[bool] = []

        def try_lock() -> None:
            if buffer._lock.acquire(timeout=0.05):
                buffer._lock.release()
                got.append(True)
            else:
                got.append(False)

        t = threading.Thread(target=try_lock)
        t.start()
        t.join(timeout=2.0)
        held.append(got[0] if got else False)
        return original(stored)

    AlignmentBuffer._materialize = staticmethod(probe)
    try:
        assert buffer.wait_new(0, timeout=1.0) is not None
    finally:
        AlignmentBuffer._materialize = original

    assert held == [True], "wait_new 在解码时仍持有 Condition 锁，会阻塞采集线程的 put()"


def test_wait_new_skips_undecodable_records_instead_of_spinning() -> None:
    """坏记录（解不出画面）应被跳过并计数，而不是把 None 交给调用方空转。

    调用方拿到 None 只能 ``continue`` 且游标不前进 ⇒ 对着同一条坏记录热自旋。
    """
    buffer = AlignmentBuffer(capacity=8)
    for i in range(3):
        buffer.put(make_sample(1000.0 + i))
    # 把 2 号记录的载荷换成解不出来的字节
    stored = buffer._records[1]
    object.__setattr__(stored, "payload", b"not-a-jpeg")

    original = AlignmentBuffer.__dict__["_materialize"]
    AlignmentBuffer._materialize = staticmethod(lambda s: original(s) if s.index != 2 else None)
    try:
        seen: list[int] = []
        cursor = 0
        # 只有 1、3 两条可取（2 号是坏记录），所以取 2 次；第 3 次应当超时返回 None
        for _ in range(2):
            record = buffer.wait_new(cursor, timeout=0.5)
            assert record is not None, "坏记录之后还有好记录，不该返回 None 让调用方空转"
            seen.append(record.index)
            cursor = record.index
        assert buffer.wait_new(cursor, timeout=0.1) is None
    finally:
        AlignmentBuffer._materialize = original

    assert seen == [1, 3], f"应跳过 2 号拿到 1、3，实际 {seen}"
    assert buffer.stats.decode_failures == 1


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


def test_deferred_alignment_writer_ignores_ground_frames_and_opens_after_cutoff() -> None:
    """任务起飞前仍可更新预览源，但不对齐或缓存画面。"""

    class Aligner:
        calls = 0

        def align(self, frame: VideoFrame) -> AlignedSample:
            self.calls += 1
            return make_sample(frame.timestamp - frame.lag)

    aligner = Aligner()
    buffer = AlignmentBuffer(capacity=10)
    writer = AlignmentWriter(buffer, aligner, defer_until_airborne=True)
    latest: VideoFrame | None = None

    for index in range(1, 4):
        latest = VideoFrame(index=index, image=make_frame(10.0).image, timestamp=10.0, lag=0.0)
        writer(latest)

    assert latest is not None and latest.index == 3
    assert aligner.calls == 0
    assert writer.stats() == (0, 0)
    assert len(buffer) == 0

    assert writer.begin_after(lambda: latest.index if latest is not None else None)
    writer(latest)  # sink 延迟到达的地面帧仍必须跳过
    next_frame = VideoFrame(index=4, image=latest.image, timestamp=10.1, lag=0.0)
    writer(next_frame)

    assert aligner.calls == 1
    assert writer.stats() == (1, 0)
    assert buffer.indices() == (1, 1)


def test_deferred_writer_cutoff_snapshot_and_sink_race_are_serialized() -> None:
    """放行快照期间发布的帧被纳入 cutoff，不会穿过门控。"""
    buffer = AlignmentBuffer(capacity=10)

    class Aligner:
        def align(self, frame: VideoFrame) -> AlignedSample:
            return make_sample(frame.timestamp)

    writer = AlignmentWriter(buffer, Aligner(), defer_until_airborne=True)
    snapshot_entered = threading.Event()
    release_snapshot = threading.Event()
    latest_index = [2]
    errors: list[AssertionError] = []

    def snapshot() -> int | None:
        snapshot_entered.set()
        if not release_snapshot.wait(2.0):
            raise AssertionError("未释放 cutoff 快照屏障")
        return latest_index[0]

    def open_writer() -> None:
        try:
            writer.begin_after(snapshot)
        except AssertionError as exc:
            errors.append(exc)

    opener = threading.Thread(target=open_writer)
    opener.start()
    assert snapshot_entered.wait(2.0)

    latest_index[0] = 3
    frame3 = VideoFrame(index=3, image=make_frame(10.0).image, timestamp=10.0, lag=0.0)
    sink_done = threading.Event()

    def delayed_sink() -> None:
        writer(frame3)
        sink_done.set()

    sink = threading.Thread(target=delayed_sink)
    sink.start()
    release_snapshot.set()
    opener.join(2.0)
    sink.join(2.0)

    assert not opener.is_alive() and not sink.is_alive()
    assert sink_done.is_set()
    assert errors == []
    assert len(buffer) == 0  # 新快照 N=3，N 帧必须跳过
    frame4 = VideoFrame(index=4, image=frame3.image, timestamp=10.1, lag=0.0)
    writer(frame4)
    assert buffer.indices() == (1, 1)


def test_deferred_writer_repeat_open_is_idempotent_and_seal_is_terminal() -> None:
    buffer = AlignmentBuffer(capacity=10)

    class Aligner:
        def align(self, frame: VideoFrame) -> AlignedSample:
            return make_sample(frame.timestamp)

    writer = AlignmentWriter(buffer, Aligner(), defer_until_airborne=True)
    assert writer.begin_after(2)
    assert not writer.begin_after(100)
    writer(VideoFrame(index=3, image=make_frame(10.0).image, timestamp=10.0, lag=0.0))
    assert buffer.indices() == (1, 1)

    writer.seal()
    writer.seal()
    assert not writer.begin_after(0)
    writer(VideoFrame(index=4, image=make_frame(10.1).image, timestamp=10.1, lag=0.0))
    assert buffer.indices() == (1, 1)
    assert writer.stats() == (1, 0)


def test_deferred_writer_default_still_accepts_passive_capture() -> None:
    """非任务采集默认保持原有逐帧入缓冲行为。"""

    class Aligner:
        def align(self, frame: VideoFrame) -> AlignedSample:
            return make_sample(frame.timestamp)

    buffer = AlignmentBuffer(capacity=10)
    writer = AlignmentWriter(buffer, Aligner())
    writer(VideoFrame(index=1, image=make_frame(10.0).image, timestamp=10.0, lag=0.0))

    assert buffer.indices() == (1, 1)
    assert writer.stats() == (1, 0)


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
