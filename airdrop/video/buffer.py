"""对齐结果的环形缓冲：把"画面 + 拍摄时刻的遥测"留给其他模块慢慢读。

它解决什么问题
--------------
下游有两类完全相反的消费方式：

* **实时引导**（YOLO/OCR 正在做的事）：只要"当前画面"。所以
  :class:`~airdrop.video.Hm30VideoSource` 的语义是"只保留最新一帧"，慢消费者
  丢帧而不是积压——积压等于给无人机看历史照片。
* **异步/事后消费**（异步推理、落盘、回放、丢包复盘）：需要**回看**刚刚过去的
  一段时间，而且往往不止一个模块要看、各自进度还不一样。

本模块就是这两者之间的接缝：写者（拿帧的那条线程）把对齐结果压进来，读者
（其他模块）按自己的节奏取走，互不影响。缓冲区满时从**最旧**的一端驱逐，
不是丢最新的——丢最新的等于让所有读者一起挨饿。

容量与内存（先看这段再选参数）
----------------------------
默认容量按"30 fps × 3 分钟"算：``DEFAULT_BUFFER_CAPACITY = 5400`` 帧。

**未压缩的 720p 帧是 2.64 MiB**（1280×720×3 字节），5400 帧就是
**≈13.9 GiB**——大多数机器放不进内存。因此默认 ``storage="jpeg"``：720p 在
JPEG 质量 80 下约 100~200 KB/帧，5400 帧约 **0.6~1.0 GiB**，代价是轻微的有损
压缩（对"看画面/找目标"足够，对像素级测量不够）。

需要无损时用 ``storage="raw"``，这时请务必同时设 ``max_bytes``（否则请确认
机器真有十几 GB 空闲内存）。写入第一帧时会按实测帧大小打印一次满载内存估算，
别等到 OOM 才发现。

关于拷贝（重要）
----------------
ffmpeg 后端每帧的 ``image`` 是**复用管道缓冲区的视图**（零拷贝换带宽），下一帧
到达时同一块内存会被覆盖。所以本缓冲在 ``raw`` 模式下显式 ``copy()``、在
``jpeg`` 模式下编码成独立字节串——**存进来的画面不会被后续帧改写**。任何需要跨帧
保留画面的地方都**不能**直接持有 ``VideoFrame.image``。

读取语义
--------
多个读者各自持有一个"上次读到的序号"，用 :meth:`AlignmentBuffer.wait_new`
往前跟：::

    index = 0
    while True:
        record = buffer.wait_new(after_index=index, timeout=1.0)
        if record is None:
            continue          # 还没有新数据
        index = record.index
        roi = record.image[top:bottom, left:right]   # 只读，别就地改
        print(record.snapshot.yaw_deg, record.capture_timestamp)

读到的 ``image``：``jpeg`` 模式下是**新解码出来的数组**（可写，改了不影响
缓冲区）；``raw`` 模式下是**内部数组的只读视图**（零拷贝，需要可写请自行
``.copy()``）。

回看历史一律走 :meth:`AlignmentBuffer.iter_between`：它分批解码，内存占用与批量
大小同阶而与总量无关（一次性把 5400 帧 720p 全解出来是 13.9 GiB，本模块因此
**不提供**"一把返回列表"的接口——真要列表就 ``list(buffer.iter_between(...))``，
那时的内存账由调用方自己认）。
"""

from __future__ import annotations

import bisect
import copy
import logging
import operator
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, replace

import cv2
import numpy as np

from ..telemetry.models import TelemetrySnapshot
from .align import AlignedSample, FrameTelemetryAligner
from .source import VideoFrame

LOGGER = logging.getLogger(__name__)

# 默认容量：按 30 fps 跑满 3 分钟
DEFAULT_FPS = 30.0
DEFAULT_SECONDS = 180.0
DEFAULT_BUFFER_CAPACITY = int(DEFAULT_FPS * DEFAULT_SECONDS)  # 5400 帧
DEFAULT_JPEG_QUALITY = 80

SUPPORTED_STORAGE = frozenset({"jpeg", "raw"})

# bisect 的 key：直接作用在记录列表上，不重建时间列表
_capture_timestamp = operator.attrgetter("capture_timestamp")
_index = operator.attrgetter("index")


def capacity_for(fps: float = DEFAULT_FPS, seconds: float = DEFAULT_SECONDS) -> int:
    """按"帧率 × 秒数"换算容量，例如 30 fps × 180 s → 5400 帧。"""
    if fps <= 0 or seconds <= 0:
        raise ValueError(f"fps/seconds 必须为正: fps={fps} seconds={seconds}")
    return max(int(round(fps * seconds)), 1)


def raw_frame_bytes(width: int, height: int, channels: int = 3) -> int:
    """未压缩一帧占多少字节（720p BGR → 2 764 800 字节 ≈ 2.64 MiB）。"""
    return int(width) * int(height) * int(channels)


@dataclass(frozen=True, slots=True)
class BufferedFrame:
    """缓冲区里的一条记录：一幅画面 + 它拍摄时刻的遥测。

    ``index`` 是**写入序号**（从 1 开始单调递增），不是列表下标——被驱逐的
    记录不会让序号回退，因此它可以安全地当读者的游标。
    """

    index: int
    image: np.ndarray
    capture_timestamp: float
    received_timestamp: float
    lag: float
    snapshot: TelemetrySnapshot
    extrapolated: bool = False
    offset: float = 0.0

    @property
    def width(self) -> int:
        return int(self.image.shape[1])

    @property
    def height(self) -> int:
        return int(self.image.shape[0])

    @property
    def age(self) -> float:
        """该画面拍摄至今过去了多少秒。"""
        return time.time() - self.capture_timestamp

    def copy(self) -> "BufferedFrame":
        """返回一份图像数据独立的副本（需要就地修改画面时用）。"""
        return replace(self, image=self.image.copy())


@dataclass(slots=True)
class BufferStats:
    """缓冲区的运行统计（供日志与容量调参使用）。"""

    storage: str = "jpeg"
    capacity: int = 0
    max_bytes: int | None = None
    frames: int = 0
    put: int = 0
    evicted: int = 0
    bytes: int = 0
    last_index: int = 0


@dataclass(frozen=True, slots=True)
class StoredFrame:
    """缓冲内部的一条记录：图像以 JPEG 字节串或独立的 ndarray 持有。

    原本是模块私有（下划线）；:class:`FlightRecorder` 需要按序号拿到 JPEG 字节
    做"零重编码"落盘，所以把名字公开。这仍是缓冲内部数据——公共读者应该走
    :class:`BufferedFrame`，需要原始字节请走 :meth:`AlignmentBuffer.read_jpeg_bytes`。
    """

    index: int
    capture_timestamp: float
    received_timestamp: float
    lag: float
    snapshot: TelemetrySnapshot
    extrapolated: bool
    offset: float
    payload: bytes | np.ndarray
    nbytes: int


class AlignmentBuffer:
    """线程安全的对齐结果环形缓冲（写者一入，读者多出）。

    参数
    ----
    capacity:
        最多保留多少帧；满了从最旧一端驱逐。默认 5400（30 fps × 3 分钟）。
    storage:
        ``"jpeg"``（默认）把画面编码成 JPEG 字节串，720p 约 0.1~0.2 MB/帧、
        5400 帧约 0.6~1.0 GiB；``"raw"`` 保存未压缩副本，像素级无损，但 720p
        单帧 2.64 MiB，5400 帧 ≈13.9 GiB，请自行确认内存或设 ``max_bytes``。
    jpeg_quality:
        JPEG 质量（1~100），仅 ``storage="jpeg"`` 时有意义。
    max_bytes:
        缓冲占用的**字节上限**（默认不限制）。超出时同样从最旧一端驱逐，
        与 ``capacity`` 谁先到就按谁算。``storage="raw"`` 时强烈建议设置。

    线程安全：内部一把 ``RLock`` + ``Condition``，写者与任意多个读者可以并发；
    JPEG 编解码在锁外进行，不会因为某个读者解码而卡住采集线程。
    """

    def __init__(
        self,
        capacity: int = DEFAULT_BUFFER_CAPACITY,
        *,
        storage: str = "jpeg",
        jpeg_quality: int = DEFAULT_JPEG_QUALITY,
        max_bytes: int | None = None,
    ) -> None:
        if capacity < 1:
            raise ValueError(f"capacity 至少为 1: {capacity}")
        if storage not in SUPPORTED_STORAGE:
            raise ValueError(f"不支持的存储方式: {storage}，可选 {sorted(SUPPORTED_STORAGE)}")
        if not 0 < jpeg_quality <= 100:
            raise ValueError(f"jpeg_quality 取值 1~100: {jpeg_quality}")
        if max_bytes is not None and max_bytes < 1:
            raise ValueError(f"max_bytes 至少为 1: {max_bytes}")

        self._capacity = capacity
        self._storage = storage
        self._jpeg_quality = jpeg_quality
        self._max_bytes = max_bytes

        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        # 用 list 而不是 deque：容量 5400 时，驱逐的 O(n) 移动远小于
        # 索引访问的开销差异，而二分/切片在 list 上都是 O(log n)/O(k)。
        self._records: list[StoredFrame] = []
        self._bytes = 0
        self._next_index = 0
        self._put = 0
        self._evicted = 0
        self._logged_estimate = False

    # ------------------------------------------------------------------
    # 写入接口
    # ------------------------------------------------------------------
    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def storage(self) -> str:
        return self._storage

    def put(self, sample: AlignedSample, *, materialize: bool = True) -> BufferedFrame | None:
        """把一条对齐结果存入缓冲，可选返回解码后的入库记录。

        传入 :meth:`airdrop.alignment.FrameTelemetryAligner.align` 的返回值即可；
        若 ``sample`` 为 None（对齐失败/无遥测）会抛 ``ValueError``，调用方应先
        自己判断。
        """
        if sample is None:
            raise ValueError("sample 不能为 None；请先判断 align() 的返回值")
        payload, nbytes = self._encode(sample.frame.image)

        with self._condition:
            self._next_index += 1
            stored = StoredFrame(
                index=self._next_index,
                capture_timestamp=sample.timestamp,
                received_timestamp=sample.frame.timestamp,
                lag=sample.lag,
                snapshot=sample.snapshot,
                extrapolated=sample.extrapolated,
                offset=sample.offset,
                payload=payload,
                nbytes=nbytes,
            )
            self._records.append(stored)
            self._bytes += nbytes
            self._put += 1
            self._evict_locked()
            self._log_estimate_locked(nbytes)
            self._condition.notify_all()

        if not materialize:
            return None
        frame = self._materialize(stored)
        if frame is None:  # 刚编码的数据不该解不出来
            raise RuntimeError("入库记录无法还原为画面")
        return frame

    def clear(self) -> None:
        """清空缓冲（不影响已发出的记录对象）。"""
        with self._condition:
            self._records.clear()
            self._bytes = 0
            self._condition.notify_all()

    # ------------------------------------------------------------------
    # 读取接口
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        with self._lock:
            return len(self._records)

    @property
    def stats(self) -> BufferStats:
        with self._lock:
            return BufferStats(
                storage=self._storage,
                capacity=self._capacity,
                max_bytes=self._max_bytes,
                frames=len(self._records),
                put=self._put,
                evicted=self._evicted,
                bytes=self._bytes,
                last_index=self._next_index,
            )

    def latest(self) -> BufferedFrame | None:
        """取最新一条；缓冲为空时返回 None。"""
        with self._lock:
            if not self._records:
                return None
            stored = self._records[-1]
        return self._materialize(stored)

    def latest_index(self) -> int:
        """最新写入记录的序号；还没写入过任何记录时返回 0。

        读者可以用它作为"我知道的最新位置"的快照上界。
        """
        with self._lock:
            return self._next_index

    def read_jpeg_bytes(self, index: int) -> bytes | None:
        """按写入序号取出**原始 JPEG 字节串**（**内部 API，专供 FlightRecorder 用**）。

        默认 jpeg 模式下返回的是缓冲在写入时编码的那份字节——**直接落盘
        即可，零重编码**，画质与编码耗时都和缓冲保持一致。raw 存储模式下
        返回 None（因为保存的是未压缩 ndarray，没有"原始 JPEG"可言），
        调用方需要自行把 :attr:`BufferedFrame.image` 重新编码。

        该序号已被驱逐或尚未产生时同样返回 None。
        """
        with self._lock:
            stored = self._find_by_index_locked(index)
        if stored is None:
            return None
        payload = stored.payload
        return payload if isinstance(payload, bytes) else None

    def at(self, index: int) -> BufferedFrame | None:
        """按写入序号取记录；该序号已被驱逐或尚未产生时返回 None。"""
        with self._lock:
            stored = self._find_by_index_locked(index)
        return None if stored is None else self._materialize(stored)

    def indices(self) -> tuple[int, int] | None:
        """当前保留的序号范围 ``(最旧, 最新)``；空缓冲返回 None。"""
        with self._lock:
            if not self._records:
                return None
            return self._records[0].index, self._records[-1].index

    def iter_between(
        self,
        start: float | None = None,
        end: float | None = None,
        *,
        batch: int = 16,
    ) -> Iterator[BufferedFrame]:
        """按时间区间逐帧迭代，分批解码，内存占用与 ``batch`` 同阶。

        整个迭代过程不持有锁，因此采集线程不会被读者的解码拖慢。

        这是**快照式**迭代：只覆盖开始迭代那一刻已入库的记录，之后新写入的
        内容不在本次迭代范围内（要跟随最新数据请用 :meth:`wait_new`）。
        """
        if batch < 1:
            raise ValueError(f"batch 至少为 1: {batch}")
        cursor = self._cursor_before(start)
        last = self.latest_index()
        while cursor <= last:
            records = self._records_after(cursor, batch, last)
            if not records:
                return
            crossed_end = False
            for stored in records:
                if end is not None and stored.capture_timestamp > end:
                    crossed_end = True
                    break
                frame = self._materialize(stored)
                if frame is not None:
                    yield frame
            if crossed_end:
                return
            cursor = records[-1].index

    def wait_new(self, after_index: int = 0, timeout: float | None = None) -> BufferedFrame | None:
        """阻塞等待比 ``after_index`` 更新的记录，返回**最早**满足条件的那条。

        这是给"按序号完整消费"的读者用的——保证不漏中间记录::

            index = 0
            while True:
                record = buffer.wait_new(index, timeout=1.0)
                if record is None:
                    continue
                # record.index == index + 1（除非中间记录被驱逐——见下）
                index = record.index

        超时返回 None（不是出错）；``timeout=None`` 表示一直等。

        **游标落后太多**：若 ``after_index`` 已经早于缓冲里保留的**最早一条**
        （记录被驱逐过），本方法返回**当前保留的最早一条**——不会永远卡在
        已被驱逐的位置，但中间已丢的记录无法在此路径上补回。需要"一帧不漏"
        的消费者（典型如 :class:`~airdrop.record.FlightRecorder`）应当
        用 :meth:`AlignmentBuffer.iter_between` 配合自己的序号游标，而不是
        仅靠 ``wait_new``。
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        stored: StoredFrame | None = None
        with self._condition:
            while True:
                records = self._records
                if records:
                    first_index = records[0].index
                    target = max(after_index + 1, first_index)
                    idx = bisect.bisect_left(records, target, key=_index)
                    if idx < len(records):
                        stored = records[idx]
                        break
                remaining = None
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return None
                self._condition.wait(remaining)
            return self._materialize(stored)

    # ------------------------------------------------------------------
    # 内部实现
    # ------------------------------------------------------------------
    def _encode(self, image: np.ndarray) -> tuple[bytes | np.ndarray, int]:
        """把图像变成可长期持有的独立数据（这是"防覆盖"的关键一步）。"""
        if self._storage == "raw":
            payload = np.array(image, copy=True)  # ffmpeg 后端的帧是复用视图
            return payload, int(payload.nbytes)
        ok, buffer = cv2.imencode(
            ".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), self._jpeg_quality]
        )
        if not ok:  # 理论上不该发生；退化成未压缩也比丢帧好
            LOGGER.warning("JPEG 编码失败，该帧退化为未压缩存储")
            payload = np.array(image, copy=True)
            return payload, int(payload.nbytes)
        data = buffer.tobytes()
        return data, len(data)

    @staticmethod
    def _materialize(stored: StoredFrame) -> BufferedFrame | None:
        """把内部记录还原成对外记录（JPEG 在此解码）。"""
        payload = stored.payload
        if isinstance(payload, np.ndarray):
            # raw：零拷贝，但设为只读，避免读者就地修改污染缓冲区
            image = payload.view()
            image.flags.writeable = False
        else:
            image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                LOGGER.error("缓冲记录 #%d 的 JPEG 解码失败，跳过", stored.index)
                return None
        return BufferedFrame(
            index=stored.index,
            image=image,
            capture_timestamp=stored.capture_timestamp,
            received_timestamp=stored.received_timestamp,
            lag=stored.lag,
            snapshot=copy.copy(stored.snapshot),
            extrapolated=stored.extrapolated,
            offset=stored.offset,
        )

    def _find_by_index_locked(self, index: int) -> StoredFrame | None:
        records = self._records
        if not records:
            return None
        first = records[0].index
        offset = index - first
        if 0 <= offset < len(records):
            stored = records[offset]  # 序号连续，可直接算出下标
            if stored.index == index:
                return stored
        # 兜底：序号理论上连续，但保守起见做一次二分
        idx = bisect.bisect_left(records, index, key=_index)
        if idx < len(records) and records[idx].index == index:
            return records[idx]
        return None

    def _cursor_before(self, start: float | None) -> int:
        """返回"已经消费到哪"的起始游标：最后一条 ``capture_timestamp < start``
        的序号；``start`` 为 None 或早于全部记录时返回 0。"""
        with self._lock:
            records = self._records
            if not records:
                return self._next_index
            if start is None:
                return 0
            pos = bisect.bisect_left(records, start, key=_capture_timestamp)
            if pos <= 0:
                return 0
            return records[pos - 1].index

    def _records_after(self, after_index: int, batch: int, max_index: int) -> list[StoredFrame]:
        """取序号在 ``(after_index, max_index]`` 内、最靠前的至多 ``batch`` 条。"""
        with self._lock:
            records = self._records
            pos = bisect.bisect_right(records, after_index, key=_index)
            return [stored for stored in records[pos : pos + batch] if stored.index <= max_index]

    def _evict_locked(self) -> None:
        """按帧数上限与字节上限从最旧一端驱逐。"""
        records = self._records
        while len(records) > self._capacity:
            self._drop_oldest_locked()
        while self._max_bytes is not None and self._bytes > self._max_bytes:
            if len(records) <= 1:  # 单帧就超限：留着它，否则永远为空
                break
            self._drop_oldest_locked()

    def _drop_oldest_locked(self) -> None:
        dropped = self._records.pop(0)
        self._bytes -= dropped.nbytes
        self._evicted += 1

    def _log_estimate_locked(self, nbytes: int) -> None:
        """第一次入库时按实测帧大小给出满载内存估算，避免 OOM 才发现。"""
        if self._logged_estimate:
            return
        self._logged_estimate = True
        estimate = nbytes * self._capacity
        LOGGER.info(
            "对齐缓冲就绪：容量 %d 帧、存储 %s、单帧约 %.2f MB → 满载约 %.2f GiB%s",
            self._capacity,
            self._storage,
            nbytes / 1024 / 1024,
            estimate / 1024 / 1024 / 1024,
            "（受 max_bytes 限制）" if self._max_bytes else "",
        )


# ----------------------------------------------------------------------
# 拉流源 → 缓冲的接线
# ----------------------------------------------------------------------
class AlignmentWriter:
    """把**每一帧**对齐后写进缓冲；直接挂到拉流源上当 sink 用。

    它只做一件事：``aligner.align(frame)`` → ``buffer.put(sample)``。挂上去以后，
    拉流源采集线程读出的每一帧都会被接管（缓冲内部会拷贝图像，不受"下一帧覆盖
    管道缓冲"的影响），因此不存在"消费者跟不上就丢帧"的问题::

        buffer = AlignmentBuffer(capacity_for(30, 180))
        writer = AlignmentWriter(buffer, FrameTelemetryAligner(broker, lag=0.15))

        with Hm30VideoSource(VideoConfig()) as source:
            source.add_sink(writer)        # 从这一帧起，一帧不落
            ...

    唯一会跳过画面的情况是对齐拿不到遥测——历史为空、等满 ``max_wait`` 仍未追上
    （``on_timeout="drop"``）、或超出 ``max_extrapolation`` 被拒——这类帧计入
    :attr:`skipped`；正常写入计入 :attr:`written`。两个数字分开看，就能区分
    "没写进去"与"跟不上"——后者在这个类里不会发生。

    ``skipped`` 只回答"丢了几帧"，丢的原因看对齐器自己的
    :attr:`~airdrop.video.align.FrameTelemetryAligner.stats`
    （``no_telemetry`` / ``wait_timeouts`` / ``extrapolation_drops``）。

    它跑在采集线程里，所以必须快：``put`` 只做一次 JPEG 编码（720p 几毫秒）；
    若给对齐器设了 ``max_wait``，等待也会发生在这个线程里（见
    :mod:`airdrop.video.align` 的"关于等待"）。
    """

    def __init__(self, buffer: AlignmentBuffer, aligner: FrameTelemetryAligner) -> None:
        self._buffer = buffer
        self._aligner = aligner
        self._lock = threading.Lock()
        self._written = 0
        self._skipped = 0

    def __call__(self, frame: VideoFrame) -> None:
        sample = self._aligner.align(frame)
        if sample is None:
            with self._lock:
                self._skipped += 1
            return
        self._buffer.put(sample, materialize=False)
        with self._lock:
            self._written += 1

    @property
    def buffer(self) -> AlignmentBuffer:
        return self._buffer

    @property
    def aligner(self) -> FrameTelemetryAligner:
        """本 writer 使用的对齐器（丢弃原因的细分统计在它身上）。"""
        return self._aligner

    @property
    def written(self) -> int:
        """已成功写进缓冲的帧数。"""
        with self._lock:
            return self._written

    @property
    def skipped(self) -> int:
        """因取不到遥测而跳过的帧数（唯一会丢画面、且原因明确的场合）。"""
        with self._lock:
            return self._skipped

    def stats(self) -> tuple[int, int]:
        """返回 ``(written, skipped)`` 快照，方便日志里一起打印。"""
        with self._lock:
            return self._written, self._skipped
