from __future__ import annotations

import ctypes
import json
import multiprocessing
import queue
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from fractions import Fraction
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from typing import Any, cast

import av
import cv2
import numpy as np
import pytest
from av.error import ConnectionResetError as PyAvConnectionResetError
from av.error import ExitError as PyAvExitError
from av.error import InvalidDataError as PyAvInvalidDataError
from av.error import TimeoutError as PyAvTimeoutError

from airdrop.config import Config, RecordConfig
from airdrop.record.recorder import FlightRecorder
from airdrop.record.replay import ReplayVideoSource
from airdrop.runtime.components import build_alignment_buffer
from airdrop.telemetry.broker import TelemetryBroker
from airdrop.telemetry.models import TelemetrySnapshot
from airdrop.video.align import AlignedSample, FrameTelemetryAligner
from airdrop.video.buffer import (
    AlignmentBuffer,
    AlignmentWriter,
    BufferedFrame,
    ColdFrameRef,
    StoredFrame,
)
from airdrop.video.cold import TsDataNotReady, TsRangeDecoder, TsSegmentWriter
from airdrop.video.segments import SegmentLifecycle, SegmentState
from airdrop.video.source import (
    Hm30VideoSource,
    VideoConfig,
    VideoFrame,
    VideoStorageError,
    _pyav_input_options,
)


def test_pyav_maps_sitl_protocol_whitelist_and_rejects_unknown_arguments() -> None:
    config = VideoConfig(extra_input_args=("-protocol_whitelist", "file,udp,rtp"))
    assert _pyav_input_options(config) == {"protocol_whitelist": "file,udp,rtp"}
    with pytest.raises(VideoStorageError, match="仅支持 -protocol_whitelist"):
        _pyav_input_options(VideoConfig(extra_input_args=("-foo", "bar")))


def _block_worker_for_test(stop_event) -> None:
    while not stop_event.is_set():
        time.sleep(1)


def _write_large_shared_slot_for_test(slot_name: str, started) -> None:
    slot = SharedMemory(name=slot_name)
    buffer = slot.buf
    assert buffer is not None
    try:
        started.set()
        chunk = b"x" * (1024 * 1024)
        for offset in range(0, slot.size, len(chunk)):
            count = min(len(chunk), slot.size - offset)
            buffer[offset : offset + count] = chunk[:count]
            time.sleep(0.01)
    finally:
        slot.close()


def _parent_guarded_blocker_for_test(started) -> None:
    from airdrop.video.source import _watch_parent_exit

    threading.Thread(target=_watch_parent_exit, daemon=True).start()
    started.set()
    while True:
        time.sleep(1)


def _supervise_parent_guarded_blocker_for_test(started, child_pid) -> None:
    context = multiprocessing.get_context("spawn")
    child = context.Process(target=_parent_guarded_blocker_for_test, args=(started,))
    child.start()
    child_pid.put(child.pid)
    while True:
        time.sleep(1)


def _abrupt_recording_parent_for_test(source_path: str, base_dir: str, ready) -> None:
    from airdrop.record.recorder import FlightRecorder
    from airdrop.telemetry.broker import TelemetryBroker
    from airdrop.telemetry.models import TelemetrySnapshot
    from airdrop.video.align import AlignedSample
    from airdrop.video.buffer import AlignmentBuffer
    from airdrop.video.source import Hm30VideoSource, VideoConfig

    broker = TelemetryBroker()
    buffer = AlignmentBuffer(capacity=200, storage="raw")
    recorder = FlightRecorder(Config(record=RecordConfig(telemetry_hz=0)), base_dir=Path(base_dir))
    source = Hm30VideoSource(VideoConfig(url=source_path, width=64, height=48))
    open_files = recorder._open_files

    class InterruptedIndexWriter:
        def __init__(self, file) -> None:
            self.file = file

        def write(self, line: str) -> int:
            record = json.loads(line)
            if record["index"] == 6:
                split = max(len(line) // 2, 1)
                self.file.write(line[:split])
                self.file.flush()
                process = source._pyav_process
                ready.put(
                    {
                        "flight_dir": str(recorder.flight_dir),
                        "worker_pid": None if process is None else process.pid,
                        "frames": record["index"],
                    }
                )
                while True:
                    time.sleep(1.0)
            return self.file.write(line)

        def flush(self) -> None:
            self.file.flush()

    def open_files_with_fault() -> None:
        open_files()
        index_file = recorder._frames_index_file
        assert index_file is not None
        setattr(recorder, "_frames_index_file", InterruptedIndexWriter(index_file))  # noqa: B010 - 故障注入替换文件句柄

    setattr(recorder, "_open_files", open_files_with_fault)  # noqa: B010 - 在线程启动前注入写入故障
    recorder.start(broker=broker, buffer=buffer, ts_video=True)
    assert recorder.video_directory is not None
    source.set_recording_directory(recorder.video_directory)

    def capture(frame: VideoFrame) -> None:
        snapshot = TelemetrySnapshot(
            timestamp=frame.capture_timestamp,
            latitude_deg=47.0 + frame.index * 1e-5,
            yaw_deg=float(frame.index),
        )
        buffer.put(
            AlignedSample(
                frame=frame,
                snapshot=snapshot,
                timestamp=frame.capture_timestamp,
                lag=frame.lag,
                mode="test",
            ),
            materialize=False,
        )

    source.add_sink(capture)
    source.add_sink(lambda _frame: time.sleep(0.05))
    thread = threading.Thread(target=source._stream_pyav_process, daemon=True)
    thread.start()
    while True:
        time.sleep(1.0)


def _supervise_pyav_udp_blocker_for_test(config, ready, child_pid, slot_names_out) -> None:
    from airdrop.video.source import _pyav_capture_worker

    context = multiprocessing.get_context("spawn")
    slots = [SharedMemory(create=True, size=config.width * config.height * 3) for _ in range(2)]
    names = tuple(slot.name for slot in slots)
    messages = context.Queue(maxsize=2)
    free_slots = context.Queue(maxsize=2)
    for slot_id in range(2):
        free_slots.put(slot_id)
    stop_event = context.Event()
    child = context.Process(
        target=_pyav_capture_worker,
        args=(
            config,
            None,
            1,
            messages,
            free_slots,
            names,
            config.width * config.height * 3,
            stop_event,
        ),
    )
    child.start()
    child_pid.put(child.pid)
    slot_names_out.put(names)
    while True:
        message = messages.get(timeout=1.0)
        if message.get("kind") == "opening":
            ready.set()
            break
    while True:
        time.sleep(1)


def _make_h264(
    path: Path, frames: int = 48, *, gop: int = 48
) -> list[tuple[int, Fraction, np.ndarray]]:
    container = av.open(str(path), mode="w", format="mp4")
    stream = container.add_stream("libx264", rate=12)
    stream.width = 64
    stream.height = 48
    stream.pix_fmt = "yuv420p"
    stream.options = {"bf": "3", "g": str(gop), "keyint_min": str(gop)}
    for index in range(frames):
        image = np.full((48, 64, 3), index * 4 % 256, dtype=np.uint8)
        frame = av.VideoFrame.from_ndarray(image, format="rgb24")
        frame.pts = index
        frame.time_base = Fraction(1, 12)
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()

    source = av.open(str(path), mode="r")
    video = source.streams.video[0]
    decoded: list[tuple[int, Fraction, np.ndarray]] = []
    for packet in source.demux(video):
        for frame in packet.decode():
            if frame.pts is not None and frame.time_base is not None:
                decoded.append(
                    (
                        frame.pts,
                        Fraction(frame.time_base.numerator, frame.time_base.denominator),
                        frame.to_ndarray(format="bgr24"),
                    )
                )
    source.close()
    return decoded


def test_ts_range_decoder_matches_source_with_b_frames_and_reuses_gop(tmp_path) -> None:
    source_path = tmp_path / "source.mp4"
    source_frames = _make_h264(source_path)
    input_container = av.open(str(source_path), mode="r")
    input_stream = input_container.streams.video[0]
    writer = TsSegmentWriter(tmp_path, segment=1, input_stream=input_stream)
    for packet in input_container.demux(input_stream):
        packet.decode()
        writer.mux(packet)
    writer.close()
    input_container.close()

    decoder = TsRangeDecoder(tmp_path, max_cache_bytes=1_000_000)
    decode_calls = 0
    original_put = decoder._put_cache

    def counted_put(key, image) -> None:
        nonlocal decode_calls
        decode_calls += 1
        original_put(key, image)

    decoder._put_cache = counted_put
    try:
        for frame_index in (2, 17, 45):
            pts, time_base, expected = source_frames[frame_index]
            actual = decoder.read(1, pts, time_base.numerator, time_base.denominator)
            assert actual is not None
            assert np.array_equal(actual, expected)
            assert decoder._cache_bytes <= decoder.max_cache_bytes

        before = decode_calls
        pts, time_base, expected = source_frames[30]
        actual = decoder.read(1, pts, time_base.numerator, time_base.denominator)
        assert actual is not None and np.array_equal(actual, expected)
        assert decode_calls == before, "同一 GOP 的连续回读应复用解码缓存"
    finally:
        decoder.close()


def test_iter_range_reports_intermediate_decoded_frames_without_copying_cache(tmp_path) -> None:
    source_path = tmp_path / "source.mp4"
    source_frames = _make_h264(source_path, frames=20, gop=20)
    source = av.open(str(source_path), mode="r")
    writer = TsSegmentWriter(tmp_path, segment=1, input_stream=source.streams.video[0])
    try:
        for packet in source.demux(source.streams.video[0]):
            packet.decode()
            writer.mux(packet)
    finally:
        writer.close()
        source.close()

    decoder = TsRangeDecoder(tmp_path, max_cache_bytes=1)
    observed: list[tuple[Fraction, np.ndarray]] = []

    def collect(frame_time: Fraction, image: np.ndarray) -> None:
        assert not image.flags.writeable
        observed.append((frame_time, image.copy()))

    pts, time_base, expected = source_frames[15]
    try:
        result = list(
            decoder.iter_range(
                1,
                [(pts, time_base.numerator, time_base.denominator)],
                on_decoded=collect,
            )
        )
        assert len(observed) == 16
        for (frame_time, image), (source_pts, source_base, expected_image) in zip(
            observed, source_frames, strict=False
        ):
            assert frame_time == source_pts * source_base
            assert np.array_equal(image, expected_image)
        assert len(result) == 1 and np.array_equal(result[0][1], expected)
        assert decoder._cache_bytes == 0
    finally:
        decoder.close()


def test_iter_range_callback_failure_closes_cursor_and_can_retry(tmp_path) -> None:
    source_path = tmp_path / "source.mp4"
    source_frames = _make_h264(source_path, frames=12, gop=12)
    source = av.open(str(source_path), mode="r")
    writer = TsSegmentWriter(tmp_path, segment=1, input_stream=source.streams.video[0])
    try:
        for packet in source.demux(source.streams.video[0]):
            packet.decode()
            writer.mux(packet)
    finally:
        writer.close()
        source.close()

    decoder = TsRangeDecoder(tmp_path, max_cache_bytes=1)
    pts, time_base, expected = source_frames[8]

    def fail(_frame_time: Fraction, _image: np.ndarray) -> None:
        raise RuntimeError("consumer failed")

    try:
        with pytest.raises(RuntimeError, match="consumer failed"):
            list(
                decoder.iter_range(
                    1,
                    [(pts, time_base.numerator, time_base.denominator)],
                    on_decoded=fail,
                )
            )
        image = decoder.read(1, pts, time_base.numerator, time_base.denominator)
        assert image is not None and np.array_equal(image, expected)
    finally:
        decoder.close()


def test_iter_range_preserves_error_when_container_is_already_unregistered(
    tmp_path, monkeypatch
) -> None:
    source_path = tmp_path / "source.mp4"
    frames = _make_h264(source_path, frames=12, gop=12)
    source = av.open(str(source_path), mode="r")
    writer = TsSegmentWriter(tmp_path, segment=1, input_stream=source.streams.video[0])
    for packet in source.demux(source.streams.video[0]):
        packet.decode()
        writer.mux(packet)
    writer.close()
    source.close()
    decoder = TsRangeDecoder(tmp_path)
    real_open = av.open
    opened = []

    def remember(*args, **kwargs):
        container = real_open(*args, **kwargs)
        opened.append(container)
        return container

    monkeypatch.setattr(av, "open", remember)

    def fail(_time, _image):
        decoder._open_containers.clear()
        raise RuntimeError("原始回调失败")

    pts, tb, _ = frames[3]
    try:
        with pytest.raises(RuntimeError, match="原始回调失败"):
            list(decoder.iter_range(1, [(pts, tb.numerator, tb.denominator)], on_decoded=fail))
    finally:
        for container in opened:
            container.close()
        decoder.close()


def test_direct_iter_range_close_waits_for_active_native_decode(tmp_path, monkeypatch) -> None:
    source_path = tmp_path / "source.mp4"
    frames = _make_h264(source_path, frames=12, gop=12)
    source = av.open(str(source_path), mode="r")
    writer = TsSegmentWriter(tmp_path, segment=1, input_stream=source.streams.video[0])
    for packet in source.demux(source.streams.video[0]):
        packet.decode()
        writer.mux(packet)
    writer.close()
    source.close()
    entered = threading.Event()
    release = threading.Event()
    closed = threading.Event()
    real_open = av.open

    class Packet:
        def __init__(self, packet):
            self.packet = packet

        def __getattr__(self, name):
            return getattr(self.packet, name)

        def decode(self):
            entered.set()
            assert release.wait(timeout=3.0)
            return self.packet.decode()

    class Container:
        def __init__(self, container):
            self.container = container

        def __getattr__(self, name):
            return getattr(self.container, name)

        def demux(self, stream):
            for packet in self.container.demux(stream):
                yield Packet(packet)

        def close(self):
            closed.set()
            self.container.close()

    monkeypatch.setattr(av, "open", lambda *args, **kwargs: Container(real_open(*args, **kwargs)))
    decoder = TsRangeDecoder(tmp_path)
    pts, tb, _ = frames[8]
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            reading = pool.submit(
                list, decoder.iter_range(1, [(pts, tb.numerator, tb.denominator)])
            )
            assert entered.wait(timeout=3.0)
            closing = pool.submit(decoder.close)
            assert decoder._read_cancelled.wait(timeout=3.0)
            assert not closed.is_set(), "原生解码仍在执行时不得跨线程关闭容器"
            release.set()
            with pytest.raises(InterruptedError, match="已取消"):
                reading.result(timeout=3.0)
            closing.result(timeout=3.0)
            assert closed.is_set()
    finally:
        release.set()
        decoder.close()


def test_old_iter_range_keeps_cancelled_token_after_decoder_reopen(tmp_path) -> None:
    source_path = tmp_path / "source.mp4"
    frames = _make_h264(source_path, frames=12, gop=12)
    source = av.open(str(source_path), mode="r")
    writer = TsSegmentWriter(tmp_path, segment=1, input_stream=source.streams.video[0])
    for packet in source.demux(source.streams.video[0]):
        packet.decode()
        writer.mux(packet)
    writer.close()
    source.close()
    decoder = TsRangeDecoder(tmp_path)
    first_pts, first_tb, expected = frames[2]
    pending_read = threading.Event()

    def requests():
        yield first_pts, first_tb.numerator, first_tb.denominator
        pending_read.set()
        raise AssertionError("已取消的旧游标不能继续请求下一帧")

    old = decoder.iter_range(1, requests())
    try:
        _, image = next(old)
        assert np.array_equal(image, expected)
        decoder.close()
        reopened = decoder.read(1, first_pts, first_tb.numerator, first_tb.denominator)
        assert reopened is not None and np.array_equal(reopened, expected)
        with pytest.raises(InterruptedError, match="已取消"):
            next(old)
        assert not pending_read.is_set()
    finally:
        old.close()
        decoder.close()


def test_main_and_frozen_buffer_share_gop_job_with_tiny_cache(tmp_path, monkeypatch) -> None:
    from airdrop.video.gop import _GopJob

    source_path = tmp_path / "source.mp4"
    frames = _make_h264(source_path, frames=20, gop=20)
    source = av.open(str(source_path), mode="r")
    writer = TsSegmentWriter(tmp_path, segment=1, input_stream=source.streams.video[0])
    for packet in source.demux(source.streams.video[0]):
        packet.decode()
        writer.mux(packet)
    writer.close()
    source.close()
    decoder = TsRangeDecoder(tmp_path, max_cache_bytes=1)
    decoded_cache = decoder.decoded_cache
    assert decoded_cache is not None
    buffer = AlignmentBuffer(capacity=30, storage="raw", cold_decoder=decoder, hot_seconds=0.01)
    for index, (pts, tb, image) in enumerate(frames):
        stamp = index / 12
        frame = VideoFrame(
            index=index + 1,
            image=image,
            timestamp=stamp + 0.15,
            lag=0.15,
            media_pts=pts,
            time_base_num=tb.numerator,
            time_base_den=tb.denominator,
            segment=1,
        )
        buffer.put(
            AlignedSample(frame, TelemetrySnapshot(timestamp=stamp), stamp, 0.15, "test"),
            materialize=False,
        )
    frozen = buffer.freeze()
    entered = threading.Event()
    release = threading.Event()
    real_dispatch = _GopJob._dispatch
    real_open = av.open
    opens = []
    first_time = frames[1][0] * frames[1][1]

    def dispatch(job, frame_time, image):
        if frame_time == first_time:
            entered.set()
            assert release.wait(timeout=3.0)
        real_dispatch(job, frame_time, image)

    def counting_open(*args, **kwargs):
        if str(args[0]).endswith("segment-000001.ts"):
            opens.append(args[0])
        return real_open(*args, **kwargs)

    monkeypatch.setattr(_GopJob, "_dispatch", dispatch)
    monkeypatch.setattr(av, "open", counting_open)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            high = pool.submit(buffer.at, 16)
            assert entered.wait(timeout=3.0)
            low = pool.submit(frozen.at, 6)
            coordinator = decoder._gop_coordinator
            assert coordinator is not None
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline:
                jobs = list(coordinator._jobs.values())
                if jobs and len(jobs[0].waiters) == 2:
                    break
                time.sleep(0.01)
            assert jobs and len(jobs[0].waiters) == 2
            release.set()
            low_frame = low.result(timeout=3.0)
            high_frame = high.result(timeout=3.0)
            assert low_frame is not None and np.array_equal(low_frame.image, frames[5][2])
            assert high_frame is not None and np.array_equal(high_frame.image, frames[15][2])
        assert len(opens) == 1
        assert decoded_cache.byte_size == 0, "GOP 命中不能依赖已缓存图像"
    finally:
        release.set()
        frozen.close_cold_storage()
        buffer.close_cold_storage()


def test_shared_gop_close_failure_keeps_job_for_retry(tmp_path, monkeypatch) -> None:
    from airdrop.video.gop import _GopJob

    source_path = tmp_path / "source.mp4"
    frames = _make_h264(source_path, frames=12, gop=12)
    source = av.open(str(source_path), mode="r")
    writer = TsSegmentWriter(tmp_path, segment=1, input_stream=source.streams.video[0])
    for packet in source.demux(source.streams.video[0]):
        packet.decode()
        writer.mux(packet)
    writer.close()
    source.close()
    main = TsRangeDecoder(tmp_path, max_cache_bytes=1)
    other = TsRangeDecoder(tmp_path, decoded_cache=main.decoded_cache)
    coordinator = main.enable_shared_gop()
    other.enable_shared_gop(coordinator)
    pts, tb, expected = frames[5]
    actual = main.read(1, pts, tb.numerator, tb.denominator)
    assert actual is not None and expected is not None
    assert np.array_equal(actual, expected)
    main.close()
    real_stop = _GopJob.stop
    attempts = []

    def fail_once(job):
        attempts.append(1)
        if len(attempts) == 1:
            raise TimeoutError("injected timeout")
        return real_stop(job)

    monkeypatch.setattr(_GopJob, "stop", fail_once)
    with pytest.raises(BaseExceptionGroup, match="GOP 作业未能关闭"):
        other.close()
    assert coordinator._jobs, "关闭失败后必须保留作业以便重试"
    other.close()
    assert not coordinator._jobs


def test_shared_gop_rejects_new_view_while_last_close_is_in_progress(tmp_path, monkeypatch) -> None:
    from airdrop.video.gop import _GopJob

    source_path = tmp_path / "source.mp4"
    frames = _make_h264(source_path, frames=12, gop=12)
    source = av.open(str(source_path), mode="r")
    writer = TsSegmentWriter(tmp_path, segment=1, input_stream=source.streams.video[0])
    for packet in source.demux(source.streams.video[0]):
        packet.decode()
        writer.mux(packet)
    writer.close()
    source.close()
    main = TsRangeDecoder(tmp_path, max_cache_bytes=1)
    other = TsRangeDecoder(tmp_path, decoded_cache=main.decoded_cache)
    newcomer = TsRangeDecoder(tmp_path, decoded_cache=main.decoded_cache)
    coordinator = main.enable_shared_gop()
    other.enable_shared_gop(coordinator)
    pts, tb, expected = frames[5]
    actual = main.read(1, pts, tb.numerator, tb.denominator)
    assert actual is not None and expected is not None
    assert np.array_equal(actual, expected)
    main.close()
    entered = threading.Event()
    release = threading.Event()
    real_stop = _GopJob.stop

    def blocking_stop(job):
        entered.set()
        assert release.wait(timeout=3.0)
        return real_stop(job)

    monkeypatch.setattr(_GopJob, "stop", blocking_stop)
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            closing = pool.submit(other.close)
            assert entered.wait(timeout=3.0)
            with pytest.raises(RuntimeError, match="不能绑定新读取者"):
                newcomer.enable_shared_gop(coordinator)
            release.set()
            closing.result(timeout=3.0)
    finally:
        release.set()
        other.close()
        newcomer.close()


def test_shared_gop_same_pts_waiters_use_one_decode_with_tiny_cache(tmp_path, monkeypatch) -> None:
    from airdrop.video.gop import _GopJob

    source_path = tmp_path / "source.mp4"
    frames = _make_h264(source_path, frames=20, gop=20)
    source = av.open(str(source_path), mode="r")
    writer = TsSegmentWriter(tmp_path, segment=1, input_stream=source.streams.video[0])
    for packet in source.demux(source.streams.video[0]):
        packet.decode()
        writer.mux(packet)
    writer.close()
    source.close()
    main = TsRangeDecoder(tmp_path, max_cache_bytes=1)
    other = TsRangeDecoder(tmp_path, decoded_cache=main.decoded_cache)
    coordinator = main.enable_shared_gop()
    other.enable_shared_gop(coordinator)
    entered = threading.Event()
    release = threading.Event()
    real_dispatch = _GopJob._dispatch
    real_open = av.open
    opens = []

    def dispatch(job, frame_time, image):
        if frame_time == frames[1][0] * frames[1][1]:
            entered.set()
            assert release.wait(timeout=3.0)
        real_dispatch(job, frame_time, image)

    def counting_open(*args, **kwargs):
        if str(args[0]).endswith("segment-000001.ts"):
            opens.append(args[0])
        return real_open(*args, **kwargs)

    monkeypatch.setattr(_GopJob, "_dispatch", dispatch)
    monkeypatch.setattr(av, "open", counting_open)
    pts, tb, expected = frames[15]
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(main.read, 1, pts, tb.numerator, tb.denominator)
            assert entered.wait(timeout=3.0)
            second = pool.submit(other.read, 1, pts, tb.numerator, tb.denominator)
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline:
                jobs = list(coordinator._jobs.values())
                if jobs and len(jobs[0].waiters) == 2:
                    break
                time.sleep(0.01)
            assert jobs and len(jobs[0].waiters) == 2
            release.set()
            first_image = first.result(timeout=3.0)
            second_image = second.result(timeout=3.0)
            assert first_image is not None and second_image is not None and expected is not None
            assert np.array_equal(first_image, expected)
            assert np.array_equal(second_image, expected)
            first_image[:] = 0
            assert np.array_equal(second_image, expected)
        assert len(opens) == 1
        assert main._cache_bytes == 0
    finally:
        release.set()
        main.close()
        other.close()


def test_shared_gop_limits_outstanding_requests_without_silent_drop(tmp_path, monkeypatch) -> None:
    from airdrop.video.gop import _GopJob

    source_path = tmp_path / "source.mp4"
    frames = _make_h264(source_path, frames=20, gop=20)
    source = av.open(str(source_path), mode="r")
    writer = TsSegmentWriter(tmp_path, segment=1, input_stream=source.streams.video[0])
    for packet in source.demux(source.streams.video[0]):
        packet.decode()
        writer.mux(packet)
    writer.close()
    source.close()
    decoder = TsRangeDecoder(tmp_path, max_cache_bytes=1)
    coordinator = decoder.enable_shared_gop()
    entered = threading.Event()
    release = threading.Event()
    real_dispatch = _GopJob._dispatch

    def dispatch(job, frame_time, image):
        if frame_time == frames[1][0] * frames[1][1]:
            entered.set()
            assert release.wait(timeout=3.0)
        real_dispatch(job, frame_time, image)

    monkeypatch.setattr(_GopJob, "_dispatch", dispatch)
    pts, tb, expected = frames[15]
    try:
        with ThreadPoolExecutor(max_workers=9) as pool:
            futures = [pool.submit(decoder.read, 1, pts, tb.numerator, tb.denominator)]
            assert entered.wait(timeout=3.0)
            futures.extend(
                pool.submit(decoder.read, 1, pts, tb.numerator, tb.denominator) for _ in range(7)
            )
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline:
                jobs = list(coordinator._jobs.values())
                if jobs and jobs[0].outstanding == 8:
                    break
                time.sleep(0.01)
            assert jobs and jobs[0].outstanding == 8
            overflow = pool.submit(decoder.read, 1, pts, tb.numerator, tb.denominator)
            with pytest.raises(RuntimeError, match="已达到上限"):
                overflow.result(timeout=3.0)
            release.set()
            for future in futures:
                image = future.result(timeout=3.0)
                assert image is not None and expected is not None
                assert np.array_equal(image, expected)
            assert jobs[0].outstanding == 0
    finally:
        release.set()
        decoder.close()


def test_shared_gop_keeps_temporary_storage_until_last_view_and_job_exit(
    tmp_path, monkeypatch
) -> None:
    from airdrop.video.gop import _GopJob
    from airdrop.video.storage import TemporaryTsStorage

    source_path = tmp_path / "source.mp4"
    frames = _make_h264(source_path, frames=20, gop=20)
    storage = TemporaryTsStorage(parent=tmp_path)
    source = av.open(str(source_path), mode="r")
    writer = TsSegmentWriter(storage.directory, segment=1, input_stream=source.streams.video[0])
    for packet in source.demux(source.streams.video[0]):
        packet.decode()
        writer.mux(packet)
    writer.close()
    source.close()
    lease = storage.acquire_lease()
    main = TsRangeDecoder(storage.directory, max_cache_bytes=1)
    other = TsRangeDecoder(storage.directory, decoded_cache=main.decoded_cache)
    main.bind_storage_lease(lease)
    other.bind_storage_lease(lease)
    lease.close()
    coordinator = main.enable_shared_gop()
    other.enable_shared_gop(coordinator)
    entered = threading.Event()
    release = threading.Event()
    real_dispatch = _GopJob._dispatch

    def dispatch(job, frame_time, image):
        if frame_time == frames[1][0] * frames[1][1]:
            entered.set()
            assert release.wait(timeout=3.0)
        real_dispatch(job, frame_time, image)

    monkeypatch.setattr(_GopJob, "_dispatch", dispatch)
    high_pts, high_tb, _ = frames[15]
    low_pts, low_tb, expected = frames[5]
    try:
        with ThreadPoolExecutor(max_workers=3) as pool:
            high = pool.submit(main.read, 1, high_pts, high_tb.numerator, high_tb.denominator)
            assert entered.wait(timeout=3.0)
            low = pool.submit(other.read, 1, low_pts, low_tb.numerator, low_tb.denominator)
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline:
                jobs = list(coordinator._jobs.values())
                if jobs and len(jobs[0].waiters) == 2:
                    break
                time.sleep(0.01)
            assert jobs and len(jobs[0].waiters) == 2
            storage.producer_stopped()
            assert not storage.finalize()
            closing = pool.submit(main.close)
            closing.result(timeout=3.0)
            assert storage.directory.is_dir()
            with pytest.raises(InterruptedError, match="取消"):
                high.result(timeout=3.0)
            release.set()
            low_image = low.result(timeout=3.0)
            assert low_image is not None and expected is not None
            assert np.array_equal(low_image, expected)
        assert storage.directory.is_dir()
        other.close()
        assert not storage.directory.exists()
    finally:
        release.set()
        main.close()
        other.close()


def test_shared_gop_delivers_decode_failure_to_all_waiters(tmp_path, monkeypatch) -> None:
    from airdrop.video.gop import _GopJob

    source_path = tmp_path / "source.mp4"
    frames = _make_h264(source_path, frames=20, gop=20)
    source = av.open(str(source_path), mode="r")
    writer = TsSegmentWriter(tmp_path, segment=1, input_stream=source.streams.video[0])
    for packet in source.demux(source.streams.video[0]):
        packet.decode()
        writer.mux(packet)
    writer.close()
    source.close()
    main = TsRangeDecoder(tmp_path, max_cache_bytes=1)
    other = TsRangeDecoder(tmp_path, decoded_cache=main.decoded_cache)
    coordinator = main.enable_shared_gop()
    other.enable_shared_gop(coordinator)
    entered = threading.Event()
    release = threading.Event()
    real_dispatch = _GopJob._dispatch

    def fail(job, frame_time, image):
        if frame_time == frames[1][0] * frames[1][1]:
            entered.set()
            assert release.wait(timeout=3.0)
            raise ValueError("模拟 GOP 损坏")
        real_dispatch(job, frame_time, image)

    monkeypatch.setattr(_GopJob, "_dispatch", fail)
    pts, tb, _ = frames[15]
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(main.read, 1, pts, tb.numerator, tb.denominator)
            assert entered.wait(timeout=3.0)
            second = pool.submit(other.read, 1, pts, tb.numerator, tb.denominator)
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline:
                jobs = list(coordinator._jobs.values())
                if jobs and len(jobs[0].waiters) == 2:
                    break
                time.sleep(0.01)
            assert jobs and len(jobs[0].waiters) == 2
            release.set()
            for future in (first, second):
                with pytest.raises(ValueError, match="模拟 GOP 损坏"):
                    future.result(timeout=3.0)
            assert not jobs[0].busy, "失败作业不能永久占据活动额度"
    finally:
        release.set()
        main.close()
        other.close()


@pytest.mark.parametrize("failure", ["construct", "bind"])
def test_shared_gop_worker_construction_failure_rolls_back_storage_lease(
    tmp_path, monkeypatch, failure: str
) -> None:
    from airdrop.video.storage import TemporaryTsStorage

    source_path = tmp_path / "source.mp4"
    frames = _make_h264(source_path, frames=12, gop=12)
    storage = TemporaryTsStorage(parent=tmp_path)
    source = av.open(str(source_path), mode="r")
    writer = TsSegmentWriter(storage.directory, segment=1, input_stream=source.streams.video[0])
    for packet in source.demux(source.streams.video[0]):
        packet.decode()
        writer.mux(packet)
    writer.close()
    source.close()
    lease = storage.acquire_lease()
    decoder = TsRangeDecoder(storage.directory, max_cache_bytes=1)
    decoder.bind_storage_lease(lease)
    lease.close()
    coordinator = decoder.enable_shared_gop()
    before = storage.lease_count
    created = []
    real_init = TsRangeDecoder.__init__
    real_bind = TsRangeDecoder.bind_storage_lease

    def injected_init(instance, *args, **kwargs):
        if not kwargs.get("cache_enabled", True) and failure == "construct":
            raise RuntimeError("模拟作业解码器构造失败")
        real_init(instance, *args, **kwargs)
        if not kwargs.get("cache_enabled", True):
            created.append(instance)

    def injected_bind(instance, given):
        if instance in created and failure == "bind":
            raise RuntimeError("模拟作业租约绑定失败")
        return real_bind(instance, given)

    monkeypatch.setattr(TsRangeDecoder, "__init__", injected_init)
    monkeypatch.setattr(TsRangeDecoder, "bind_storage_lease", injected_bind)
    pts, tb, _ = frames[5]
    try:
        with pytest.raises(RuntimeError, match="模拟作业"):
            decoder.read(1, pts, tb.numerator, tb.denominator)
        assert storage.lease_count == before
        assert not coordinator._jobs
        assert all(not item._reaper.is_alive() for item in created)
    finally:
        decoder.close()
        storage.producer_stopped()
        assert storage.finalize()


def test_cold_watermark_never_accepts_cached_image_without_reading_ts(tmp_path) -> None:
    source_path = tmp_path / "source.mp4"
    frames = _make_h264(source_path, frames=12, gop=12)
    source = av.open(str(source_path), mode="r")
    writer = TsSegmentWriter(tmp_path, segment=1, input_stream=source.streams.video[0])
    for packet in source.demux(source.streams.video[0]):
        packet.decode()
        writer.mux(packet)
    writer.close()
    source.close()
    decoder = TsRangeDecoder(tmp_path, max_cache_bytes=100_000)
    buffer = AlignmentBuffer(capacity=12, storage="raw", cold_decoder=decoder)
    pts, tb, actual = frames[5]
    forged = np.bitwise_xor(actual, np.uint8(255))
    key = decoder._make_cache_key(1, Fraction(pts) * tb, actual.shape[1], actual.shape[0])
    decoder._put_cache(key, forged)
    frame = VideoFrame(
        index=1,
        image=forged,
        timestamp=1.15,
        lag=0.15,
        media_pts=pts,
        time_base_num=tb.numerator,
        time_base_den=tb.denominator,
        segment=1,
    )
    try:
        buffer.put(
            AlignedSample(frame, TelemetrySnapshot(timestamp=1.0), 1.0, 0.15, "test"),
            materialize=False,
        )
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and buffer._cold_error is None:
            time.sleep(0.01)
        with pytest.raises(OSError, match="冷读校验不一致"):
            buffer.raise_if_failed()
    finally:
        with pytest.raises(RuntimeError, match="冷区排空失败"):
            buffer.close_cold_storage()


def test_shared_gop_long_sequential_reads_reuse_cursor_with_tiny_cache(
    tmp_path, monkeypatch
) -> None:
    source_path = tmp_path / "source.mp4"
    frames = _make_h264(source_path, frames=24, gop=24)
    source = av.open(str(source_path), mode="r")
    writer = TsSegmentWriter(tmp_path, segment=1, input_stream=source.streams.video[0])
    for packet in source.demux(source.streams.video[0]):
        packet.decode()
        writer.mux(packet)
    writer.close()
    source.close()
    main = TsRangeDecoder(tmp_path, max_cache_bytes=1)
    other = TsRangeDecoder(tmp_path, decoded_cache=main.decoded_cache)
    coordinator = main.enable_shared_gop()
    other.enable_shared_gop(coordinator)
    real_open = av.open
    opens = []

    def counting_open(*args, **kwargs):
        if str(args[0]).endswith("segment-000001.ts"):
            opens.append(args[0])
        return real_open(*args, **kwargs)

    monkeypatch.setattr(av, "open", counting_open)
    try:
        for index, (pts, tb, expected) in enumerate(frames):
            decoder = main if index % 2 == 0 else other
            image = decoder.read(1, pts, tb.numerator, tb.denominator)
            assert image is not None and np.array_equal(image, expected)
        assert len(opens) == 1
        assert main._cache_bytes == 0
    finally:
        main.close()
        other.close()


def test_shared_gop_idle_jobs_do_not_consume_all_range_slots(tmp_path) -> None:
    source_path = tmp_path / "source.mp4"
    frames = _make_h264(source_path, frames=24, gop=2)
    source = av.open(str(source_path), mode="r")
    writer = TsSegmentWriter(tmp_path, segment=1, input_stream=source.streams.video[0])
    for packet in source.demux(source.streams.video[0]):
        packet.decode()
        writer.mux(packet)
    writer.close()
    source.close()
    main = TsRangeDecoder(tmp_path, max_cache_bytes=1, cursor_idle_timeout_s=30.0)
    other = TsRangeDecoder(tmp_path, decoded_cache=main.decoded_cache, cursor_idle_timeout_s=30.0)
    coordinator = main.enable_shared_gop()
    other.enable_shared_gop(coordinator)
    try:
        assert len(main._get_keyframes(1, tmp_path / "segment-000001.ts")) >= 9
        for index in range(0, 20, 2):
            pts, tb, expected = frames[index]
            actual = main.read(1, pts, tb.numerator, tb.denominator)
            assert actual is not None and np.array_equal(actual, expected)
    finally:
        main.close()
        other.close()


def test_shared_gop_unconsumed_results_pin_bounded_job_slots(tmp_path, monkeypatch) -> None:
    from airdrop.video.gop import _GopJob

    source_path = tmp_path / "source.mp4"
    frames = _make_h264(source_path, frames=24, gop=2)
    source = av.open(str(source_path), mode="r")
    writer = TsSegmentWriter(tmp_path, segment=1, input_stream=source.streams.video[0])
    for packet in source.demux(source.streams.video[0]):
        packet.decode()
        writer.mux(packet)
    writer.close()
    source.close()
    decoder = TsRangeDecoder(tmp_path, max_cache_bytes=1)
    coordinator = decoder.enable_shared_gop()
    assert len(decoder._get_keyframes(1, tmp_path / "segment-000001.ts")) >= 9
    entered = threading.Event()
    release = threading.Event()
    count_lock = threading.Lock()
    count = 0
    real_finish = _GopJob.finish

    def blocked_finish(job, waiter):
        nonlocal count
        with count_lock:
            count += 1
            if count == 8:
                entered.set()
        assert release.wait(timeout=5.0)
        real_finish(job, waiter)

    monkeypatch.setattr(_GopJob, "finish", blocked_finish)
    try:
        with ThreadPoolExecutor(max_workers=9) as pool:
            futures = [
                pool.submit(
                    decoder.read, 1, frames[i][0], frames[i][1].numerator, frames[i][1].denominator
                )
                for i in range(0, 16, 2)
            ]
            assert entered.wait(timeout=5.0)
            assert len(coordinator._jobs) == 8
            pts, tb, expected = frames[16]
            ninth = pool.submit(decoder.read, 1, pts, tb.numerator, tb.denominator)
            assert not ninth.done(), "未消费的结果必须占用有界作业额度"
            release.set()
            ninth_image = ninth.result(timeout=5.0)
            assert ninth_image is not None and expected is not None
            assert np.array_equal(ninth_image, expected)
            for future in futures:
                assert future.result(timeout=5.0) is not None
    finally:
        release.set()
        decoder.close()


def test_shared_gop_retries_if_idle_stop_races_request_registration(tmp_path, monkeypatch) -> None:
    from airdrop.video.gop import _GopJob

    source_path = tmp_path / "source.mp4"
    frames = _make_h264(source_path, frames=12, gop=12)
    source = av.open(str(source_path), mode="r")
    writer = TsSegmentWriter(tmp_path, segment=1, input_stream=source.streams.video[0])
    for packet in source.demux(source.streams.video[0]):
        packet.decode()
        writer.mux(packet)
    writer.close()
    source.close()
    decoder = TsRangeDecoder(tmp_path, max_cache_bytes=1, cursor_idle_timeout_s=0.05)
    coordinator = decoder.enable_shared_gop()
    assert coordinator is not None
    first_pts, tb, first_image = frames[2]
    first_image_result = decoder.read(1, first_pts, tb.numerator, tb.denominator)
    assert first_image_result is not None and first_image is not None
    assert np.array_equal(first_image_result, first_image)
    entered = threading.Event()
    release = threading.Event()
    real_add = _GopJob.add

    def slow_add(job, waiter):
        if not entered.is_set():
            entered.set()
            assert release.wait(timeout=3.0)
        return real_add(job, waiter)

    monkeypatch.setattr(_GopJob, "add", slow_add)
    next_pts, next_tb, expected = frames[8]
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            result = pool.submit(decoder.read, 1, next_pts, next_tb.numerator, next_tb.denominator)
            assert entered.wait(timeout=3.0)
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline:
                jobs = list(coordinator._jobs.values())
                if jobs and jobs[0].stopping:
                    break
                time.sleep(0.01)
            assert jobs and jobs[0].stopping
            release.set()
            image = result.result(timeout=3.0)
            assert image is not None and expected is not None
            assert np.array_equal(image, expected)
    finally:
        release.set()
        decoder.close()


def test_last_gop_view_close_stops_decode_at_next_frame(tmp_path, monkeypatch) -> None:
    from airdrop.video.gop import _GopJob

    source_path = tmp_path / "source.mp4"
    frames = _make_h264(source_path, frames=20, gop=20)
    source = av.open(str(source_path), mode="r")
    writer = TsSegmentWriter(tmp_path, segment=1, input_stream=source.streams.video[0])
    for packet in source.demux(source.streams.video[0]):
        packet.decode()
        writer.mux(packet)
    writer.close()
    source.close()
    decoder = TsRangeDecoder(tmp_path, max_cache_bytes=1)
    coordinator = decoder.enable_shared_gop()
    entered = threading.Event()
    release = threading.Event()
    seen = []
    real_dispatch = _GopJob._dispatch

    def dispatch(job, frame_time, image):
        seen.append(frame_time)
        if frame_time == frames[1][0] * frames[1][1]:
            entered.set()
            assert release.wait(timeout=3.0)
        real_dispatch(job, frame_time, image)

    monkeypatch.setattr(_GopJob, "_dispatch", dispatch)
    pts, tb, _ = frames[15]
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            reading = pool.submit(decoder.read, 1, pts, tb.numerator, tb.denominator)
            assert entered.wait(timeout=3.0)
            closing = pool.submit(decoder.close)
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline:
                jobs = list(coordinator._jobs.values())
                if jobs and jobs[0].stopping:
                    break
                time.sleep(0.01)
            assert jobs and jobs[0].stopping
            release.set()
            with pytest.raises(InterruptedError):
                reading.result(timeout=3.0)
            closing.result(timeout=3.0)
        assert len(seen) == 2, "最后读取者关闭后不应继续解码至原目标"
    finally:
        release.set()
        decoder.close()


def test_cancel_last_pending_request_stops_job_even_if_another_view_remains(
    tmp_path, monkeypatch
) -> None:
    from airdrop.video.gop import _GopJob

    source_path = tmp_path / "source.mp4"
    frames = _make_h264(source_path, frames=20, gop=20)
    source = av.open(str(source_path), mode="r")
    writer = TsSegmentWriter(tmp_path, segment=1, input_stream=source.streams.video[0])
    for packet in source.demux(source.streams.video[0]):
        packet.decode()
        writer.mux(packet)
    writer.close()
    source.close()
    main = TsRangeDecoder(tmp_path, max_cache_bytes=1)
    other = TsRangeDecoder(tmp_path, decoded_cache=main.decoded_cache)
    coordinator = main.enable_shared_gop()
    other.enable_shared_gop(coordinator)
    entered = threading.Event()
    release = threading.Event()
    real_dispatch = _GopJob._dispatch

    def dispatch(job, frame_time, image):
        if not entered.is_set() and frame_time == frames[1][0] * frames[1][1]:
            entered.set()
            assert release.wait(timeout=3.0)
        real_dispatch(job, frame_time, image)

    monkeypatch.setattr(_GopJob, "_dispatch", dispatch)
    pts, tb, expected = frames[15]
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            reading = pool.submit(main.read, 1, pts, tb.numerator, tb.denominator)
            assert entered.wait(timeout=3.0)
            main.close()
            jobs = list(coordinator._jobs.values())
            assert jobs and jobs[0].stopping
            release.set()
            with pytest.raises(InterruptedError):
                reading.result(timeout=3.0)
        image = other.read(1, pts, tb.numerator, tb.denominator)
        assert image is not None and expected is not None
        assert np.array_equal(image, expected)
    finally:
        release.set()
        main.close()
        other.close()


def test_decoder_close_cancels_only_its_active_read_and_allows_reopen(tmp_path) -> None:
    source_path = tmp_path / "source.mp4"
    frames = _make_h264(source_path, frames=20, gop=20)
    source = av.open(str(source_path), mode="r")
    writer = TsSegmentWriter(tmp_path, segment=1, input_stream=source.streams.video[0])
    try:
        for packet in source.demux(source.streams.video[0]):
            packet.decode()
            writer.mux(packet)
    finally:
        writer.close()
        source.close()

    decoder = TsRangeDecoder(tmp_path, max_cache_bytes=1)
    other = TsRangeDecoder(tmp_path, decoded_cache=decoder.decoded_cache)
    entered = threading.Event()
    release = threading.Event()
    original_put = decoder._put_cache

    def blocked_put(key, image) -> None:
        if not entered.is_set():
            entered.set()
            assert release.wait(timeout=3.0)
        original_put(key, image)

    decoder._put_cache = blocked_put
    pts, base, expected = frames[15]
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            read = pool.submit(decoder.read, 1, pts, base.numerator, base.denominator)
            assert entered.wait(timeout=3.0)
            close = pool.submit(decoder.close)
            assert decoder._read_cancelled.wait(timeout=3.0)
            release.set()
            with pytest.raises(InterruptedError, match="已取消"):
                read.result(timeout=3.0)
            close.result(timeout=3.0)
        other_image = other.read(1, pts, base.numerator, base.denominator)
        assert other_image is not None and np.array_equal(other_image, expected)
        image = decoder.read(1, pts, base.numerator, base.denominator)
        assert image is not None and np.array_equal(image, expected)
    finally:
        release.set()
        decoder.close()
        other.close()


def test_ts_writer_rolls_back_if_stream_template_creation_fails(tmp_path, monkeypatch) -> None:
    class BrokenContainer:
        closed = False

        def add_stream_from_template(self, stream):
            del stream
            raise RuntimeError("broken stream template")

        def close(self) -> None:
            self.closed = True

    container = BrokenContainer()
    monkeypatch.setattr(av, "open", lambda *args, **kwargs: container)
    with pytest.raises(RuntimeError, match="broken stream template"):
        TsSegmentWriter(tmp_path, segment=1, input_stream=object())
    assert container.closed
    path = tmp_path / "segment-000001.ts"
    assert not path.exists()


def test_ts_writer_never_overwrites_existing_segment_artifacts(tmp_path) -> None:
    existing = {
        "segment-000001.ts": b"previous video",
        "segment-000001.rap.jsonl": b"previous index\n",
        "segment-000001.complete": b"previous complete marker",
    }
    for name, content in existing.items():
        (tmp_path / name).write_bytes(content)

    with pytest.raises(FileExistsError, match="拒绝覆盖"):
        TsSegmentWriter(tmp_path, segment=1, input_stream=object())

    assert {name: (tmp_path / name).read_bytes() for name in existing} == existing


def test_ts_range_decoder_serializes_concurrent_seeks(tmp_path) -> None:
    source_path = tmp_path / "source.mp4"
    source_frames = _make_h264(source_path, frames=24)
    input_container = av.open(str(source_path), mode="r")
    input_stream = input_container.streams.video[0]
    writer = TsSegmentWriter(tmp_path, segment=1, input_stream=input_stream)
    for packet in input_container.demux(input_stream):
        packet.decode()
        writer.mux(packet)
    writer.close()
    input_container.close()

    decoder = TsRangeDecoder(tmp_path)
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            outputs = list(
                pool.map(
                    lambda index: decoder.read(
                        1,
                        source_frames[index][0],
                        source_frames[index][1].numerator,
                        source_frames[index][1].denominator,
                    ),
                    (3, 4, 5, 6),
                )
            )
        assert all(output is not None for output in outputs)
        for index, output in zip((3, 4, 5, 6), outputs, strict=True):
            assert output is not None
            assert np.array_equal(output, source_frames[index][2])
    finally:
        decoder.close()


def test_ts_range_decoder_read_keeps_cursor_when_gop_exceeds_cache(tmp_path) -> None:
    source_path = tmp_path / "source.mp4"
    source_frames = _make_h264(source_path)
    input_container = av.open(str(source_path), mode="r")
    input_stream = input_container.streams.video[0]
    writer = TsSegmentWriter(tmp_path, segment=1, input_stream=input_stream)
    for packet in input_container.demux(input_stream):
        packet.decode()
        writer.mux(packet)
    writer.close()
    input_container.close()

    decoder = TsRangeDecoder(tmp_path, max_cache_bytes=20_000)
    requests = [
        (
            index,
            source_frames[index][0],
            source_frames[index][1].numerator,
            source_frames[index][1].denominator,
            source_frames[index][2],
        )
        for index in (2, 17, 45)
    ]
    decoded_count = 0
    original_put = decoder._put_cache

    def counted_put(key, image) -> None:
        nonlocal decoded_count
        decoded_count += 1
        original_put(key, image)

    decoder._put_cache = counted_put
    try:
        for index, pts, num, den, expected in requests:
            image = decoder.read(1, pts, num, den)
            assert image is not None
            assert np.array_equal(image, source_frames[index][2])
            assert np.array_equal(image, expected)
        assert decoded_count <= 48, "连续历史读取不应重复解码 GOP 的同一段"
        assert decoder._cache_bytes <= decoder.max_cache_bytes
    finally:
        decoder.close()


def test_ts_range_decoder_reaps_idle_cursor_and_restarts_reaper_after_close(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    decoder = TsRangeDecoder(tmp_path, cursor_idle_timeout_s=0.05)
    released = threading.Event()

    def cursor():
        try:
            yield (1, 1, 1), np.zeros((1, 1, 3), dtype=np.uint8)
        finally:
            released.set()

    active_cursor = cursor()
    next(active_cursor)
    decoder._cursor = active_cursor
    decoder._cursor_last_used = time.monotonic()
    try:
        assert released.wait(timeout=1.0)
        assert decoder._cursor is None

        def iter_range(segment, requests, *, output_size=None):
            del segment, output_size
            yield next(iter(requests)), np.ones((1, 1, 3), dtype=np.uint8)

        monkeypatch.setattr(decoder, "iter_range", iter_range)
        decoder.close()
        assert decoder.read(1, 1, 1, 1) is not None
        assert decoder._reaper.is_alive()
    finally:
        decoder.close()
    assert not decoder._reaper.is_alive()


def test_idle_reaper_failure_blocks_reads_and_keeps_storage_until_close_retry(tmp_path) -> None:
    from airdrop.video.storage import TemporaryTsStorage

    storage = TemporaryTsStorage(parent=tmp_path)
    lease = storage.acquire_lease()
    decoder = TsRangeDecoder(storage.directory, cursor_idle_timeout_s=0.02)
    decoder.bind_storage_lease(lease)
    lease.close()
    storage.producer_stopped()
    assert not storage.finalize()
    failed = threading.Event()

    class Cursor:
        calls = 0

        def close(self):
            self.calls += 1
            if self.calls == 1:
                failed.set()
                raise OSError("模拟游标关闭失败")

    class Container:
        closed = False

        def close(self):
            self.closed = True

    cursor = Cursor()
    container = Container()
    with decoder._lock:
        cast(Any, decoder)._cursor = cursor
        decoder._cursor_last_used = time.monotonic()
        decoder._open_containers[id(container)] = container
    try:
        assert failed.wait(timeout=2)
        decoder._reaper.join(timeout=2)
        assert not decoder._reaper.is_alive()
        assert decoder._reaper_stop.is_set()
        assert decoder._cursor is cursor
        assert storage.directory.exists()
        assert storage.lease_count == 1
        with pytest.raises(RuntimeError, match="关闭未完成") as caught:
            decoder.read(1, 1, 1, 1)
        assert isinstance(caught.value.__cause__, OSError)
        with pytest.raises(RuntimeError, match="关闭未完成"):
            next(decoder.iter_range(1, [(1, 1, 1)]))
        assert not container.closed
        decoder.close()
        assert cursor.calls == 2
        assert container.closed
        assert storage.finalized
    finally:
        if not storage.finalized:
            decoder.close()


def test_ts_range_decoder_serializes_read_with_close_and_joins_each_reaper(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    decoder = TsRangeDecoder(tmp_path, cursor_idle_timeout_s=0.05)
    read_entered = threading.Event()
    release_read = threading.Event()
    second_read_entered = threading.Event()
    read_results: list[np.ndarray | None] = []
    failures: list[RuntimeError] = []
    old_reaper = decoder._reaper
    old_stop = decoder._reaper_stop
    calls = 0

    def iter_range(segment, requests, *, output_size=None):
        del segment, output_size
        nonlocal calls
        calls += 1
        if calls == 1:
            read_entered.set()
            if not release_read.wait(timeout=2.0):
                raise TimeoutError("测试未释放首个读取")
        else:
            second_read_entered.set()
        pending = iter(requests)
        yield next(pending), np.full((1, 1, 3), calls, dtype=np.uint8)
        for request in pending:
            yield request, np.full((1, 1, 3), calls, dtype=np.uint8)

    monkeypatch.setattr(decoder, "iter_range", iter_range)

    def read(pts: int) -> None:
        try:
            read_results.append(decoder.read(1, pts, 1, 1))
        except RuntimeError as exc:
            failures.append(exc)

    reader_a = threading.Thread(target=read, args=(1,))
    closer = threading.Thread(target=decoder.close)
    reader_b = threading.Thread(target=read, args=(2,))
    reader_a.start()
    try:
        assert read_entered.wait(timeout=1.0)
        closer.start()
        reader_b.start()
        assert not second_read_entered.wait(timeout=0.05)
    finally:
        release_read.set()
    reader_a.join(timeout=2.0)
    closer.join(timeout=2.0)
    reader_b.join(timeout=2.0)

    try:
        assert not reader_a.is_alive()
        assert not closer.is_alive()
        assert not reader_b.is_alive()
        assert not failures
        assert second_read_entered.is_set()
        assert len(read_results) == 2
        assert not old_reaper.is_alive()
        assert old_stop.is_set()
        if decoder._reaper is not old_reaper:
            assert decoder._reaper.is_alive()
            assert decoder._reaper_stop is not old_stop
    finally:
        decoder.close()
    assert not decoder._reaper.is_alive()


def test_ts_range_decoder_rewinds_after_oversized_frame_eviction(tmp_path) -> None:
    source_path = tmp_path / "source.mp4"
    source_frames = _make_h264(source_path, frames=12)
    input_container = av.open(str(source_path), mode="r")
    input_stream = input_container.streams.video[0]
    writer = TsSegmentWriter(tmp_path, segment=1, input_stream=input_stream)
    for packet in input_container.demux(input_stream):
        packet.decode()
        writer.mux(packet)
    writer.close()
    input_container.close()

    decoder = TsRangeDecoder(tmp_path, max_cache_bytes=1)
    pts, time_base, expected = source_frames[5]
    try:
        for _ in range(2):
            actual = decoder.read(1, pts, time_base.numerator, time_base.denominator)
            assert actual is not None
            assert np.array_equal(actual, expected)
        assert decoder._cache_bytes == 0
    finally:
        decoder.close()


def test_ts_range_decoder_applies_hot_buffer_resize_rule(tmp_path) -> None:
    import cv2

    source_path = tmp_path / "source.mp4"
    source_frames = _make_h264(source_path, frames=12)
    input_container = av.open(str(source_path), mode="r")
    input_stream = input_container.streams.video[0]
    writer = TsSegmentWriter(tmp_path, segment=1, input_stream=input_stream)
    for packet in input_container.demux(input_stream):
        packet.decode()
        writer.mux(packet)
    writer.close()
    input_container.close()

    decoder = TsRangeDecoder(tmp_path, max_cache_bytes=100_000)
    pts, time_base, expected = source_frames[4]
    actual = decoder.read(1, pts, time_base.numerator, time_base.denominator, output_size=(32, 24))
    assert actual is not None
    assert np.array_equal(actual, cv2.resize(expected, (32, 24)))
    decode_calls = 0
    original_put = decoder._put_cache

    def counted_put(key, image) -> None:
        nonlocal decode_calls
        decode_calls += 1
        original_put(key, image)

    decoder._put_cache = counted_put
    cached = decoder.read(1, pts, time_base.numerator, time_base.denominator, output_size=(32, 24))
    assert cached is not None and np.array_equal(cached, actual)
    assert decode_calls == 0
    decoder.close()


def test_freeze_keeps_call_time_cutoff_when_put_arrives_during_cold_wait(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    decoder = TsRangeDecoder(tmp_path)
    buffer = AlignmentBuffer(
        capacity=8,
        storage="raw",
        cold_decoder=decoder,
        _start_cold_worker=False,
    )

    def put(index: int) -> None:
        image = np.full((2, 2, 3), index, np.uint8)
        buffer.put(
            AlignedSample(
                frame=VideoFrame(index=index, image=image, timestamp=float(index), lag=0),
                snapshot=TelemetrySnapshot(timestamp=float(index)),
                timestamp=float(index),
                lag=0,
                mode="test",
            ),
            materialize=False,
        )

    put(1)
    with buffer._condition:
        first_record = buffer._records[0]
        reference = ColdFrameRef(1, 1, 1, 1)
        buffer._records[0] = replace(
            first_record, payload=reference, cold_locator=reference, nbytes=0
        )
        buffer._hot_bytes -= first_record.nbytes
        buffer._cold_active_index = first_record.index
        buffer._cold_schedule_limit = first_record.index
        buffer._cold_next_schedule = first_record.index + 1
    waiting = threading.Event()
    release = threading.Event()
    original_pending = buffer._cold_pending_through_locked

    def pending(index: int, *, require_verified: bool = False) -> bool:
        if index == 1 and not release.is_set():
            waiting.set()
            return True
        return original_pending(index, require_verified=require_verified)

    monkeypatch.setattr(buffer, "_cold_pending_through_locked", pending)
    frozen_result: list[AlignmentBuffer] = []
    freeze_errors: list[Exception] = []

    def freeze() -> None:
        try:
            frozen_result.append(buffer.freeze(after_index=0))
        except Exception as exc:  # noqa: BLE001 - 收集线程中的冻结异常供测试断言
            freeze_errors.append(exc)

    thread = threading.Thread(target=freeze)
    thread.start()
    try:
        assert waiting.wait(timeout=1.0)
        put(2)
        with buffer._condition:
            buffer._cold_active_index = None
            release.set()
            buffer._condition.notify_all()
        thread.join(timeout=2.0)
        assert not thread.is_alive()
        assert not freeze_errors
        assert len(frozen_result) == 1
        frozen = frozen_result[0]
        assert [record.index for record in frozen._records] == [1]
        assert frozen.stats.last_index == 1
        frozen.close_cold_storage()
    finally:
        release.set()
        thread.join(timeout=2.0)
        buffer.close_cold_storage()
        decoder.close()


def test_active_ts_keyframe_index_tracks_appends(tmp_path) -> None:
    source_path = tmp_path / "source.mp4"
    source_frames = _make_h264(source_path, frames=48, gop=12)
    source_container = av.open(str(source_path), mode="r")
    stream = source_container.streams.video[0]
    writer = TsSegmentWriter(tmp_path, segment=1, input_stream=stream)
    decoder = TsRangeDecoder(tmp_path)
    try:
        packets = source_container.demux(stream)
        for packet in packets:
            packet.decode()
            writer.mux(packet)
            if len(decoder._get_keyframes(1, writer.path)) >= 2:
                break
        before = len(decoder._get_keyframes(1, writer.path))
        for packet in packets:
            packet.decode()
            writer.mux(packet)
        after = len(decoder._get_keyframes(1, writer.path))
        assert after >= before
        assert after > 1
        assert len(source_frames) == 48
    finally:
        writer.close()
        source_container.close()
        decoder.close()


@pytest.mark.parametrize("old_cursor_result", ["eof", "overshoot"])
def test_active_cursor_reopens_segment_after_it_becomes_complete(
    monkeypatch, tmp_path, old_cursor_result: str
) -> None:
    """封口后重开解码器应能读到旧活动游标看不到的尾帧。"""
    segment = tmp_path / "segment-000001.ts"
    segment.write_bytes(b"active-ts-prefix")
    index_path = segment.with_suffix(".rap.jsonl")
    index_path.write_text(
        '{"pts":0,"dts":0,"time_base_num":1,"time_base_den":30}\n'
        '{"pts":1,"dts":1,"time_base_num":1,"time_base_den":30}\n',
        encoding="utf-8",
    )
    complete_path = segment.with_suffix(".complete")
    lifecycle = SegmentLifecycle()
    lifecycle.opened(1)

    class Frame:
        time_base = Fraction(1, 30)
        is_corrupt = False

        def __init__(self, pts: int) -> None:
            self.pts = pts

        def to_ndarray(self, *, format: str) -> np.ndarray:
            assert format == "bgr24"
            return np.full((2, 2, 3), self.pts, dtype=np.uint8)

    class Packet:
        size = 1
        is_corrupt = False

        def __init__(self, frames: list[Frame]) -> None:
            self.frames = frames

        def decode(self) -> list[Frame]:
            return self.frames

    class Container:
        streams = type(
            "Streams", (), {"video": [type("Stream", (), {"time_base": Fraction(1, 30)})()]}
        )()

        def __init__(self, frames: list[Frame]) -> None:
            self.frames = frames

        def seek(self, *args, **kwargs) -> None:
            del args, kwargs

        def demux(self, stream):
            del stream
            yield Packet(self.frames)

        def close(self) -> None:
            return

    opened = 0

    def open_container(*args, **kwargs):
        nonlocal opened
        del args, kwargs
        opened += 1
        if opened == 1:
            frames = [Frame(0)] if old_cursor_result == "eof" else [Frame(0), Frame(2)]
        else:
            frames = [Frame(0), Frame(1)]
        return Container(frames)

    monkeypatch.setattr(av, "open", open_container)
    decoder = TsRangeDecoder(tmp_path, segment_lifecycle=lifecycle)
    try:
        first = decoder.read(1, 0, 1, 30)
        assert first is not None and np.all(first == 0)
        complete_path.touch()

        with pytest.raises(EOFError, match="活动 MPEG-TS"):
            decoder.read(1, 1, 1, 30)
        assert decoder._cursor is None
        lifecycle.closed(1, complete=True)

        recovered = decoder.read(1, 1, 1, 30)
        assert recovered is not None and np.all(recovered == 1)
        assert opened == 2
    finally:
        decoder.close()


def test_completed_segment_still_rejects_a_missing_exact_pts(monkeypatch, tmp_path) -> None:
    segment = tmp_path / "segment-000001.ts"
    segment.write_bytes(b"complete-ts")
    segment.with_suffix(".complete").touch()
    segment.with_suffix(".rap.jsonl").write_text(
        '{"pts":0,"dts":0,"time_base_num":1,"time_base_den":30}\n',
        encoding="utf-8",
    )

    class Frame:
        pts = 0
        time_base = Fraction(1, 30)
        is_corrupt = False

        def to_ndarray(self, *, format: str) -> np.ndarray:
            assert format == "bgr24"
            return np.zeros((2, 2, 3), dtype=np.uint8)

    class Packet:
        size = 1
        is_corrupt = False

        def decode(self) -> list[Frame]:
            return [Frame()]

    class Container:
        streams = type(
            "Streams", (), {"video": [type("Stream", (), {"time_base": Fraction(1, 30)})()]}
        )()

        def seek(self, *args, **kwargs) -> None:
            del args, kwargs

        def demux(self, stream):
            del stream
            yield Packet()

        def close(self) -> None:
            return

    monkeypatch.setattr(av, "open", lambda *args, **kwargs: Container())
    decoder = TsRangeDecoder(tmp_path)
    try:
        with pytest.raises(ValueError, match="找不到精确 PTS"):
            decoder.read(1, 1, 1, 30)
    finally:
        decoder.close()


@pytest.mark.parametrize("damage", ["packet", "frame"])
def test_active_corrupt_segment_reopens_after_growth_and_seal(
    monkeypatch, tmp_path, damage: str
) -> None:
    segment = tmp_path / "segment-000001.ts"
    segment.write_bytes(b"active-prefix")
    segment.with_suffix(".rap.jsonl").write_text(
        '{"pts":0,"dts":0,"time_base_num":1,"time_base_den":30}\n'
        '{"pts":1,"dts":1,"time_base_num":1,"time_base_den":30}\n',
        encoding="utf-8",
    )
    lifecycle = SegmentLifecycle()
    lifecycle.opened(1)

    class Frame:
        time_base = Fraction(1, 30)

        def __init__(self, pts: int, *, corrupt: bool = False) -> None:
            self.pts = pts
            self.is_corrupt = corrupt

        def to_ndarray(self, *, format: str) -> np.ndarray:
            assert format == "bgr24"
            return np.full((2, 2, 3), self.pts, dtype=np.uint8)

    class Packet:
        size = 1

        def __init__(self, frame: Frame | None, *, corrupt: bool = False) -> None:
            self.pts = None if frame is None else frame.pts
            self.is_corrupt = corrupt
            self.frame = frame

        def decode(self) -> list[Frame]:
            return [] if self.frame is None else [self.frame]

    opened = 0

    class Container:
        streams = type(
            "Streams", (), {"video": [type("Stream", (), {"time_base": Fraction(1, 30)})()]}
        )()

        def __init__(self, *, dirty: bool) -> None:
            self.dirty = dirty

        def seek(self, *args, **kwargs) -> None:
            del args, kwargs

        def demux(self, stream):
            del stream
            yield Packet(Frame(0))
            if self.dirty:
                if damage == "packet":
                    yield Packet(None, corrupt=True)
                else:
                    yield Packet(Frame(1, corrupt=True))
            else:
                yield Packet(Frame(1))

        def close(self) -> None:
            return

    def open_container(*args, **kwargs):
        nonlocal opened
        del args, kwargs
        opened += 1
        return Container(dirty=opened == 1)

    monkeypatch.setattr(av, "open", open_container)
    decoder = TsRangeDecoder(tmp_path, segment_lifecycle=lifecycle)
    try:
        first = decoder.read(1, 0, 1, 30)
        assert first is not None and np.all(first == 0)
        with pytest.raises(EOFError, match="活动 MPEG-TS 段"):
            decoder.read(1, 1, 1, 30)
        assert decoder._cursor is None

        with segment.open("ab") as handle:
            handle.write(b"-sealed")
        segment.with_suffix(".complete").touch()
        lifecycle.closed(1, complete=True)
        recovered = decoder.read(1, 1, 1, 30)
        assert recovered is not None and np.all(recovered == 1)
        assert opened == 2
    finally:
        decoder.close()


@pytest.mark.parametrize("state", [SegmentState.UNKNOWN, SegmentState.CLOSED_INCOMPLETE])
def test_corrupt_segment_without_live_writer_fails_without_retry(
    monkeypatch, tmp_path, state: SegmentState
) -> None:
    segment = tmp_path / "segment-000001.ts"
    segment.write_bytes(b"ts")
    segment.with_suffix(".rap.jsonl").write_text(
        '{"pts":0,"dts":0,"time_base_num":1,"time_base_den":30}\n',
        encoding="utf-8",
    )
    lifecycle = SegmentLifecycle()
    if state is SegmentState.CLOSED_INCOMPLETE:
        lifecycle.opened(1)
        lifecycle.closed(1, complete=False)

    class Frame:
        pts = 0
        time_base = Fraction(1, 30)
        is_corrupt = False

    class Packet:
        size = 1
        pts = 0
        is_corrupt = True

        def decode(self) -> list[Frame]:
            pytest.fail("损坏包不应送入解码器")

    class Container:
        streams = type(
            "Streams", (), {"video": [type("Stream", (), {"time_base": Fraction(1, 30)})()]}
        )()

        def seek(self, *args, **kwargs) -> None:
            del args, kwargs

        def demux(self, stream):
            del stream
            yield Packet()

        def close(self) -> None:
            return

    monkeypatch.setattr(av, "open", lambda *args, **kwargs: Container())
    decoder = TsRangeDecoder(tmp_path, segment_lifecycle=lifecycle)
    try:
        with pytest.raises(ValueError, match="包标记为损坏"):
            decoder.read(1, 0, 1, 30)
    finally:
        decoder.close()


def test_cold_worker_waits_for_segment_revision_without_holding_buffer_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lifecycle = SegmentLifecycle()
    lifecycle.opened(1)
    decoder = TsRangeDecoder("unused", segment_lifecycle=lifecycle)
    buffer = AlignmentBuffer(
        capacity=4,
        storage="raw",
        cold_decoder=decoder,
        max_hot_bytes=1024,
    )
    source = Hm30VideoSource()

    class BufferSink:
        _buffer = buffer

        def __call__(self, frame: VideoFrame) -> None:
            del frame

    source.add_sink(BufferSink())
    assert source.segment_lifecycle is lifecycle
    entered_wait = threading.Event()
    release_wait = threading.Event()
    original_wait = lifecycle.wait_for_change

    def wait_for_change(segment: int, revision: int, timeout: float | None = None):
        entered_wait.set()
        if not release_wait.wait(timeout=2.0):
            raise TimeoutError("测试未释放 lifecycle 等待")
        return original_wait(segment, revision, timeout=0.0)

    monkeypatch.setattr(lifecycle, "wait_for_change", wait_for_change)
    attempts = 0

    def read(segment: int, pts: int, num: int, den: int, *, output_size=None):
        nonlocal attempts
        del segment, num, den
        attempts += 1
        if attempts == 1:
            raise TsDataNotReady("模拟活动尾部暂不可读")
        assert output_size == (2, 2)
        return np.full((2, 2, 3), pts, dtype=np.uint8)

    monkeypatch.setattr(decoder, "read", read)
    assert buffer._cold_worker_decoder is not None
    monkeypatch.setattr(buffer._cold_worker_decoder, "read", read)

    def put(index: int) -> None:
        timestamp = index / 30
        image = np.full((2, 2, 3), index, dtype=np.uint8)
        buffer.put(
            AlignedSample(
                frame=VideoFrame(
                    index=index + 1,
                    image=image,
                    timestamp=timestamp,
                    lag=0,
                    media_pts=index,
                    time_base_num=1,
                    time_base_den=30,
                    segment=1,
                ),
                snapshot=TelemetrySnapshot(timestamp=timestamp),
                timestamp=timestamp,
                lag=0,
                mode="test",
            ),
            materialize=False,
        )

    try:
        put(0)
        assert entered_wait.wait(timeout=2.0)
        started = time.monotonic()
        put(1)
        assert time.monotonic() - started < 0.5
        assert buffer.stats.hot_bytes > 0, "暂未验证的图像必须保留在热区"

        lifecycle.progress(1)
        release_wait.set()
        deadline = time.monotonic() + 3.0
        while buffer.stats.cold_pending and time.monotonic() < deadline:
            time.sleep(0.01)
        assert buffer.stats.cold_pending == 0
        assert buffer.stats.cold_verified == 2
        frozen = buffer.freeze(after_index=0)
        assert frozen.segment_lifecycle is lifecycle
        assert frozen._cold_decoder is not None
        assert frozen._cold_decoder.segment_lifecycle is lifecycle
        frozen.close_cold_storage()
    finally:
        release_wait.set()
        buffer.close_cold_storage()


def test_source_requires_all_cold_sinks_to_share_one_segment_lifecycle() -> None:
    first = SegmentLifecycle()
    second = SegmentLifecycle()
    source = Hm30VideoSource()

    class BufferSink:
        def __init__(self, lifecycle: SegmentLifecycle) -> None:
            self._buffer = type("Buffer", (), {"segment_lifecycle": lifecycle})()

        def __call__(self, frame: VideoFrame) -> None:
            del frame

    source.add_sink(BufferSink(first))
    source.add_sink(BufferSink(first))
    assert source.segment_lifecycle is first
    with pytest.raises(RuntimeError, match="必须共享同一"):
        source.add_sink(BufferSink(second))

    explicit = Hm30VideoSource(segment_lifecycle=first)
    with pytest.raises(RuntimeError, match="必须共享同一"):
        explicit.bind_segment_lifecycle(second)


@pytest.mark.parametrize("damage", ["packet", "frame"])
def test_ts_range_decoder_rejects_corrupt_packets_and_frames(
    monkeypatch, tmp_path, damage: str
) -> None:
    segment = tmp_path / "segment-000001.ts"
    segment.write_bytes(b"ts")
    segment.with_suffix(".complete").touch()
    segment.with_suffix(".rap.jsonl").write_text(
        '{"pts":0,"dts":0,"time_base_num":1,"time_base_den":30}\n',
        encoding="utf-8",
    )

    class Frame:
        pts = 0
        time_base = Fraction(1, 30)
        is_corrupt = damage == "frame"

        def to_ndarray(self, *, format: str) -> np.ndarray:
            assert format == "bgr24"
            return np.zeros((2, 2, 3), dtype=np.uint8)

    class Packet:
        size = 1
        is_corrupt = damage == "packet"
        pts = 0

        def decode(self) -> list[Frame]:
            if damage == "packet":
                pytest.fail("损坏包不应被送入解码器")
            return [Frame()]

    class Container:
        streams = type(
            "Streams", (), {"video": [type("Stream", (), {"time_base": Fraction(1, 30)})()]}
        )()

        def seek(self, *args, **kwargs) -> None:
            del args, kwargs

        def demux(self, stream):
            del stream
            yield Packet()

        def close(self) -> None:
            return

    monkeypatch.setattr(av, "open", lambda *args, **kwargs: Container())
    decoder = TsRangeDecoder(tmp_path)
    try:
        with pytest.raises(ValueError, match="标记为损坏"):
            decoder.read(1, 0, 1, 30)
    finally:
        decoder.close()


def test_ts_range_decoder_decodes_delayed_frame_from_flush_packet(monkeypatch, tmp_path) -> None:
    segment = tmp_path / "segment-000001.ts"
    segment.write_bytes(b"ts")
    segment.with_suffix(".rap.jsonl").write_text(
        '{"pts":0,"dts":0,"time_base_num":1,"time_base_den":30}\n',
        encoding="utf-8",
    )

    class Frame:
        time_base = Fraction(1, 30)
        is_corrupt = False

        def __init__(self, pts: int) -> None:
            self.pts = pts

        def to_ndarray(self, *, format: str) -> np.ndarray:
            assert format == "bgr24"
            return np.full((2, 2, 3), self.pts, dtype=np.uint8)

    class Packet:
        def __init__(self, size: int, frame: Frame) -> None:
            self.size = size
            self.frame = frame
            self.is_corrupt = size == 0
            self.pts = frame.pts

        def decode(self) -> list[Frame]:
            return [self.frame]

    flush_packet = Packet(0, Frame(1))

    class Container:
        streams = type(
            "Streams", (), {"video": [type("Stream", (), {"time_base": Fraction(1, 30)})()]}
        )()

        def seek(self, *args, **kwargs) -> None:
            del args, kwargs

        def demux(self, stream):
            del stream
            yield Packet(1, Frame(0))
            yield flush_packet

        def close(self) -> None:
            return

    monkeypatch.setattr(av, "open", lambda *args, **kwargs: Container())
    decoder = TsRangeDecoder(tmp_path)
    try:
        first = decoder.read(1, 0, 1, 30)
        delayed = decoder.read(1, 1, 1, 30)
        assert first is not None and np.all(first == 0)
        assert delayed is not None and np.all(delayed == 1)
        assert flush_packet.size == 0 and flush_packet.is_corrupt
    finally:
        decoder.close()


def test_corrupt_random_access_index_is_not_treated_as_not_ready(tmp_path) -> None:
    segment = tmp_path / "segment-000001.ts"
    segment.write_bytes(b"placeholder")
    segment.with_suffix(".rap.jsonl").write_text('{"pts":"bad"}\n', encoding="utf-8")
    decoder = TsRangeDecoder(tmp_path)
    with pytest.raises(ValueError, match="随机访问索引损坏"):
        decoder._get_keyframes(1, segment)
    decoder.close()


def test_ts_writer_flush_allows_exact_reads_while_segment_is_open(tmp_path) -> None:
    source_path = tmp_path / "source.mp4"
    source_frames = _make_h264(source_path)
    source_container = av.open(str(source_path), mode="r")
    source_stream = source_container.streams.video[0]
    writer = TsSegmentWriter(tmp_path, segment=1, input_stream=source_stream)
    verified: set[int] = set()
    try:
        for packet in source_container.demux(source_stream):
            decoded = packet.decode()
            writer.mux(packet)
            for frame in decoded:
                if frame.pts is None or frame.time_base is None:
                    continue
                time_base = Fraction(frame.time_base.numerator, frame.time_base.denominator)
                index = next(
                    index
                    for index, (pts, source_tb, _) in enumerate(source_frames)
                    if Fraction(pts) * source_tb == Fraction(frame.pts) * time_base
                )
                decoder = TsRangeDecoder(tmp_path, max_cache_bytes=100_000)
                try:
                    actual = decoder.read(1, frame.pts, time_base.numerator, time_base.denominator)
                except EOFError, ValueError:
                    continue
                finally:
                    decoder.close()
                if actual is not None:
                    assert np.array_equal(actual, source_frames[index][2])
                    verified.add(index)
    finally:
        writer.close()
        source_container.close()

    assert verified, "写入尚未关闭时至少应能验证一部分热帧已进入 TS"
    decoder = TsRangeDecoder(tmp_path, max_cache_bytes=1_000_000)
    try:
        for index, (pts, time_base, expected) in enumerate(source_frames):
            actual = decoder.read(1, pts, time_base.numerator, time_base.denominator)
            assert actual is not None
            assert np.array_equal(actual, expected), f"source frame {index}"
    finally:
        decoder.close()


def test_pyav_subprocess_preserves_frames_and_media_identity(tmp_path) -> None:
    source_path = tmp_path / "source.mp4"
    expected_frames = _make_h264(source_path, frames=24, gop=12)
    output_dir = tmp_path / "cold"
    source = Hm30VideoSource(config=VideoConfig(url=str(source_path), width=64, height=48))
    source.set_recording_directory(output_dir)
    published: list[VideoFrame] = []

    def slow_sink(frame: VideoFrame) -> None:
        time.sleep(0.01)
        published.append(frame.copy())

    source.add_sink(slow_sink)
    errors: list[Exception] = []

    def run_capture() -> None:
        try:
            source._stream_pyav_process()
        except Exception as exc:  # noqa: BLE001 - 线程异常由断言统一检查
            errors.append(exc)

    thread = threading.Thread(target=run_capture)
    thread.start()
    deadline = time.monotonic() + 5.0
    while source._pyav_messages is None and time.monotonic() < deadline:
        time.sleep(0.01)
    assert source._pyav_messages is not None
    assert source._pyav_messages._maxsize == 2
    thread.join(timeout=10.0)
    assert not thread.is_alive()
    assert len(errors) == 1 and "输入流结束" in str(errors[0])

    assert len(published) == len(expected_frames)
    assert source.stats.pyav_queue_age_max_s >= 0.005
    assert source.stats.pyav_queue_age_s >= 0.0
    assert all(a.timestamp <= b.timestamp for a, b in zip(published, published[1:], strict=False))
    by_time = {Fraction(pts) * time_base: image for pts, time_base, image in expected_frames}
    decoder = TsRangeDecoder(output_dir, max_cache_bytes=20_000)
    try:
        for frame in published:
            assert frame.media_pts is not None
            assert frame.time_base_num is not None and frame.time_base_den is not None
            key = Fraction(frame.media_pts * frame.time_base_num, frame.time_base_den)
            actual = decoder.read(
                frame.segment,
                frame.media_pts,
                frame.time_base_num,
                frame.time_base_den,
                output_size=(64, 48),
            )
            assert actual is not None
            assert np.array_equal(actual, by_time[key])
        assert (output_dir / "segment-000001.complete").is_file()
        assert len(decoder._get_keyframes(1, output_dir / "segment-000001.ts")) >= 2
    finally:
        decoder.close()


def test_pyav_worker_timestamps_entire_b_frame_batch_before_mux_or_conversion(monkeypatch) -> None:
    from airdrop.video import cold as cold_module
    from airdrop.video import source as source_module

    events: list[str] = []

    class Frame:
        is_corrupt = False
        time_base = Fraction(1, 30)

        def __init__(self, pts: int) -> None:
            self.pts = pts

        def to_ndarray(self, *, format: str) -> np.ndarray:
            assert format == "bgr24"
            events.append(f"pixels:{self.pts}")
            return np.full((2, 2, 3), self.pts, dtype=np.uint8)

    class Packet:
        size = 1
        is_keyframe = True
        is_corrupt = False
        pts = dts = 0
        time_base = Fraction(1, 30)

        def decode(self):
            return [Frame(0), Frame(1)]

    class Stream:
        thread_type = "AUTO"

    class Container:
        streams = type("Streams", (), {"video": [Stream()]})()

        def demux(self, _stream):
            yield Packet()

        def close(self) -> None:
            pass

    class Writer:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def mux(self, _packet) -> None:
            events.append("mux")

        def close(self, *, complete: bool) -> None:
            assert complete

    class Messages:
        def __init__(self) -> None:
            self.rows = []

        def put(self, message, timeout) -> None:
            del timeout
            self.rows.append(message)

    class Stop:
        def is_set(self) -> bool:
            return False

    wall = iter((100.0, 101.0))
    mono = iter((1_000_000_000, 1_010_000_000))
    monkeypatch.setattr("av.open", lambda *_args, **_kwargs: Container())
    monkeypatch.setattr(cold_module, "TsSegmentWriter", Writer)
    monkeypatch.setattr(source_module.time, "time", lambda: (events.append("ready"), next(wall))[1])
    monkeypatch.setattr(
        source_module.time,
        "monotonic_ns",
        lambda: (events.append("ready"), next(mono))[1],
    )
    slots = [SharedMemory(create=True, size=12), SharedMemory(create=True, size=12)]
    free_slots = queue.Queue()
    for slot_id in range(2):
        free_slots.put(slot_id)
    messages = Messages()
    try:
        source_module._pyav_capture_worker(
            VideoConfig(url="udp://127.0.0.1:1234", width=2, height=2),
            "recording",
            1,
            messages,
            free_slots,
            tuple(slot.name for slot in slots),
            12,
            Stop(),
        )
    finally:
        for slot in slots:
            slot.close()
            slot.unlink()

    frames = [message for message in messages.rows if message["kind"] == "frame"]
    assert [message["timestamp"] for message in frames] == [100.0, 101.0]
    assert [message["producer_monotonic_ns"] for message in frames] == [
        1_000_000_000,
        1_010_000_000,
    ]
    assert events.index("mux") == 4
    assert events.index("pixels:0") > events.index("mux")


def test_pyav_worker_skips_packets_before_first_keyframe(monkeypatch) -> None:
    from airdrop.video import cold as cold_module
    from airdrop.video import source as source_module

    class FakePacket:
        def __init__(self, *, keyframe: bool, pts: int | None, corrupt: bool = False) -> None:
            self.size = 1
            self.is_keyframe = keyframe
            self.is_corrupt = corrupt
            self.pts = pts
            self.dts = pts
            self.time_base = Fraction(1, 30)
            self.decoded = False

        def decode(self):
            self.decoded = True
            return []

    class FakeWriter:
        def __init__(self, *_args, **_kwargs) -> None:
            self.packets = []

        def mux(self, packet) -> None:
            self.packets.append(packet)

        def close(self, *, complete: bool) -> None:
            assert complete

    class FakeQueue:
        def __init__(self) -> None:
            self.messages = []

        def put(self, message, timeout) -> None:
            self.messages.append(message)

    class FakeStopEvent:
        def is_set(self) -> bool:
            return False

    skipped = [
        FakePacket(keyframe=False, pts=0, corrupt=True),
        FakePacket(keyframe=False, pts=1),
    ]
    keyframe = FakePacket(keyframe=True, pts=2)
    following = FakePacket(keyframe=False, pts=3)

    class FakeStream:
        thread_type = "AUTO"

    class FakeContainer:
        streams = type("Streams", (), {"video": [FakeStream()]})()

        def demux(self, _stream):
            yield from [*skipped, keyframe, following]

        def close(self) -> None:
            pass

    writer = FakeWriter()
    monkeypatch.setattr("av.open", lambda *_args, **_kwargs: FakeContainer())
    monkeypatch.setattr(cold_module, "TsSegmentWriter", lambda *_args, **_kwargs: writer)
    slot = SharedMemory(create=True, size=12)
    messages = FakeQueue()
    try:
        source_module._pyav_capture_worker(
            VideoConfig(url="udp://127.0.0.1:1234", width=2, height=2),
            "recording",
            1,
            messages,
            queue.Queue(),
            (slot.name,),
            slot.size,
            FakeStopEvent(),
        )
    finally:
        slot.close()
        slot.unlink()

    assert not any(packet.decoded for packet in skipped)
    assert writer.packets == [keyframe, following]
    ready = next(message for message in messages.messages if message["kind"] == "ready")
    assert ready["startup_packets_skipped"] == 2
    assert messages.messages[-1] == {
        "kind": "eof",
        "segment_state": SegmentState.CLOSED_COMPLETE.value,
    }


@pytest.mark.parametrize("damage", ["packet", "frame"])
def test_pyav_worker_damage_after_start_is_reconnectable_and_incomplete(
    tmp_path: Path, monkeypatch, damage: str
) -> None:
    from airdrop.video import cold as cold_module
    from airdrop.video import source as source_module

    class FakeFrame:
        is_corrupt = True

    class FakePacket:
        def __init__(self, *, keyframe: bool, pts: int, corrupt: bool = False) -> None:
            self.size = 1
            self.is_keyframe = keyframe
            self.is_corrupt = corrupt
            self.pts = pts
            self.dts = pts
            self.time_base = Fraction(1, 30)

        def decode(self):
            return [FakeFrame()] if damage == "frame" and self.is_keyframe else []

    class FakeStream:
        thread_type = "AUTO"

    class FakeContainer:
        streams = type("Streams", (), {"video": [FakeStream()]})()

        def demux(self, _stream):
            yield FakePacket(keyframe=True, pts=0)
            if damage == "packet":
                yield FakePacket(keyframe=False, pts=1, corrupt=True)

        def close(self) -> None:
            pass

    class FakeWriter:
        def __init__(self, directory: Path, *, segment: int, input_stream) -> None:
            del input_stream
            self.complete_path = directory / f"segment-{segment:06d}.complete"
            self.complete: bool | None = None

        def mux(self, _packet) -> None:
            pass

        def close(self, *, complete: bool) -> None:
            self.complete = complete
            if complete:
                self.complete_path.touch()

    class FakeQueue:
        def __init__(self) -> None:
            self.messages = []

        def put(self, message, timeout) -> None:
            del timeout
            self.messages.append(message)

    class FakeStopEvent:
        def is_set(self) -> bool:
            return False

    writer = FakeWriter(tmp_path, segment=1, input_stream=None)
    monkeypatch.setattr("av.open", lambda *_args, **_kwargs: FakeContainer())
    monkeypatch.setattr(cold_module, "TsSegmentWriter", lambda *_args, **_kwargs: writer)
    messages = FakeQueue()
    source_module._pyav_capture_worker(
        VideoConfig(url="udp://127.0.0.1:1234", width=2, height=2),
        str(tmp_path),
        1,
        messages,
        queue.Queue(),
        (),
        12,
        FakeStopEvent(),
    )

    assert messages.messages[-1]["kind"] == "stream_error"
    assert damage in messages.messages[-1]["error"] or "损坏" in messages.messages[-1]["error"]
    assert writer.complete is False
    assert not writer.complete_path.exists()


@pytest.mark.parametrize(
    ("read_error", "error_stage", "should_seal", "stop_during_read"),
    [
        (PyAvTimeoutError(110, "timed out"), "read", True, False),
        (PyAvConnectionResetError(104, "connection reset"), "read", True, False),
        (PyAvExitError(1414092869, "Immediate exit requested"), "read", True, False),
        (PyAvExitError(1414092869, "Immediate exit requested"), "read", True, True),
        (PyAvInvalidDataError(1094995529, "invalid data"), "decode", False, False),
        (PyAvExitError(1414092869, "Immediate exit requested"), "decode", False, False),
    ],
)
def test_pyav_worker_seals_normal_read_termination_but_not_invalid_data(
    tmp_path: Path,
    monkeypatch,
    read_error: Exception,
    error_stage: str,
    should_seal: bool,
    stop_during_read: bool,
) -> None:
    from airdrop.video import cold as cold_module
    from airdrop.video import source as source_module

    class FakePacket:
        size = 1
        is_keyframe = True
        is_corrupt = False
        pts = 0
        dts = 0
        time_base = Fraction(1, 30)

        def decode(self):
            if error_stage == "decode":
                raise read_error
            return []

    class FakeStream:
        thread_type = "AUTO"

    class FakeContainer:
        streams = type("Streams", (), {"video": [FakeStream()]})()

        def demux(self, _stream):
            yield FakePacket()
            if error_stage == "read":
                raise read_error

        def close(self) -> None:
            pass

    class FakeWriter:
        def __init__(self, directory: Path, *, segment: int, input_stream) -> None:
            del input_stream
            self.complete_path = directory / f"segment-{segment:06d}.complete"
            self.complete: bool | None = None
            self.packets = []

        def mux(self, packet) -> None:
            self.packets.append(packet)

        def close(self, *, complete: bool) -> None:
            self.complete = complete
            if complete:
                self.complete_path.touch()

    class FakeQueue:
        def __init__(self) -> None:
            self.messages = []

        def put(self, message, timeout) -> None:
            del timeout
            self.messages.append(message)

    class FakeStopEvent:
        def __init__(self) -> None:
            self.checks = 0

        def is_set(self) -> bool:
            self.checks += 1
            return stop_during_read and self.checks >= 3

    writer = FakeWriter(tmp_path, segment=1, input_stream=None)
    monkeypatch.setattr("av.open", lambda *_args, **_kwargs: FakeContainer())
    monkeypatch.setattr(cold_module, "TsSegmentWriter", lambda *_args, **_kwargs: writer)
    messages = FakeQueue()
    source_module._pyav_capture_worker(
        VideoConfig(url="udp://127.0.0.1:1234", width=2, height=2),
        str(tmp_path),
        1,
        messages,
        queue.Queue(),
        (),
        12,
        FakeStopEvent(),
    )

    expected_message = "eof" if stop_during_read else "stream_error"
    assert messages.messages[-1]["kind"] == expected_message
    assert writer.complete is should_seal
    assert writer.complete_path.exists() is should_seal
    assert len(writer.packets) == (0 if error_stage == "decode" else 1)


def test_pyav_worker_eof_before_first_keyframe_does_not_seal_empty_segment(
    tmp_path: Path, monkeypatch
) -> None:
    from airdrop.video import cold as cold_module
    from airdrop.video import source as source_module

    class FakePacket:
        size = 1
        is_keyframe = False
        is_corrupt = False
        pts = 0
        dts = 0
        time_base = Fraction(1, 30)

    class FakeStream:
        thread_type = "AUTO"

    class FakeContainer:
        streams = type("Streams", (), {"video": [FakeStream()]})()

        def demux(self, _stream):
            yield FakePacket()

        def close(self) -> None:
            pass

    class FakeWriter:
        def __init__(self, directory: Path, *, segment: int, input_stream) -> None:
            del input_stream
            self.complete_path = directory / f"segment-{segment:06d}.complete"
            self.complete: bool | None = None

        def mux(self, _packet) -> None:
            pytest.fail("首关键帧前的包不得写入 TS")

        def close(self, *, complete: bool) -> None:
            self.complete = complete
            if complete:
                self.complete_path.touch()

    class FakeQueue:
        def __init__(self) -> None:
            self.messages = []

        def put(self, message, timeout) -> None:
            del timeout
            self.messages.append(message)

    class FakeStopEvent:
        def is_set(self) -> bool:
            return False

    writer = FakeWriter(tmp_path, segment=1, input_stream=None)
    monkeypatch.setattr("av.open", lambda *_args, **_kwargs: FakeContainer())
    monkeypatch.setattr(cold_module, "TsSegmentWriter", lambda *_args, **_kwargs: writer)
    messages = FakeQueue()
    source_module._pyav_capture_worker(
        VideoConfig(url="udp://127.0.0.1:1234", width=2, height=2),
        str(tmp_path),
        1,
        messages,
        queue.Queue(),
        (),
        12,
        FakeStopEvent(),
    )

    assert messages.messages[-1] == {
        "kind": "eof",
        "segment_state": SegmentState.CLOSED_INCOMPLETE.value,
    }
    assert writer.complete is False
    assert not writer.complete_path.exists()


def test_pyav_worker_decodes_delayed_frame_from_corrupt_flush_packet(monkeypatch) -> None:
    from airdrop.video import cold as cold_module
    from airdrop.video import source as source_module

    class FakeFrame:
        pts = 1
        time_base = Fraction(1, 30)
        is_corrupt = False

        def to_ndarray(self, *, format: str) -> np.ndarray:
            assert format == "bgr24"
            return np.ones((2, 2, 3), dtype=np.uint8)

    class FakePacket:
        def __init__(self, *, size: int, keyframe: bool, pts: int | None, corrupt: bool) -> None:
            self.size = size
            self.is_keyframe = keyframe
            self.is_corrupt = corrupt
            self.pts = pts
            self.dts = pts
            self.time_base = Fraction(1, 30)

        def decode(self):
            return [FakeFrame()] if self.size == 0 else []

    class FakeStream:
        thread_type = "AUTO"

    class FakeContainer:
        streams = type("Streams", (), {"video": [FakeStream()]})()

        def demux(self, _stream):
            yield FakePacket(size=1, keyframe=True, pts=0, corrupt=False)
            # PyAV flush 包可能标记 corrupt，但仍需要 decode 取回 B 帧延迟画面。
            yield FakePacket(size=0, keyframe=False, pts=None, corrupt=True)

        def close(self) -> None:
            pass

    class FakeWriter:
        def __init__(self, *_args, **_kwargs) -> None:
            self.packets = []
            self.complete: bool | None = None

        def mux(self, packet) -> None:
            self.packets.append(packet)

        def close(self, *, complete: bool) -> None:
            self.complete = complete

    class FakeQueue:
        def __init__(self) -> None:
            self.messages = []

        def put(self, message, timeout) -> None:
            del timeout
            self.messages.append(message)

    class FakeStopEvent:
        def is_set(self) -> bool:
            return False

    writer = FakeWriter()
    monkeypatch.setattr("av.open", lambda *_args, **_kwargs: FakeContainer())
    monkeypatch.setattr(cold_module, "TsSegmentWriter", lambda *_args, **_kwargs: writer)
    slot = SharedMemory(create=True, size=12)
    free_slots = queue.Queue()
    free_slots.put(0)
    messages = FakeQueue()
    try:
        source_module._pyav_capture_worker(
            VideoConfig(url="udp://127.0.0.1:1234", width=2, height=2),
            "recording",
            1,
            messages,
            free_slots,
            (slot.name,),
            slot.size,
            FakeStopEvent(),
        )
    finally:
        slot.close()
        slot.unlink()

    frames = [message for message in messages.messages if message["kind"] == "frame"]
    assert len(frames) == 1 and frames[0]["pts"] == 1
    assert messages.messages[-1] == {
        "kind": "eof",
        "segment_state": SegmentState.CLOSED_COMPLETE.value,
    }
    assert writer.complete is True
    assert len(writer.packets) == 1


def test_pyav_worker_fails_when_startup_keyframe_has_no_timestamp(monkeypatch) -> None:
    from airdrop.video import source as source_module

    class FakePacket:
        size = 1
        is_keyframe = True
        is_corrupt = False
        pts = None
        dts = None
        time_base = None

        def decode(self):
            pytest.fail("缺少时间戳的关键包不能交给解码器")

    class FakeStream:
        thread_type = "AUTO"

    class FakeContainer:
        streams = type("Streams", (), {"video": [FakeStream()]})()

        def demux(self, _stream):
            yield FakePacket()

        def close(self) -> None:
            pass

    class FakeQueue:
        def __init__(self) -> None:
            self.messages = []

        def put(self, message, timeout) -> None:
            self.messages.append(message)

    class FakeStopEvent:
        def is_set(self) -> bool:
            return False

    monkeypatch.setattr("av.open", lambda *_args, **_kwargs: FakeContainer())
    messages = FakeQueue()
    source_module._pyav_capture_worker(
        VideoConfig(url="udp://127.0.0.1:1234", width=2, height=2),
        None,
        1,
        messages,
        queue.Queue(),
        (),
        12,
        FakeStopEvent(),
    )
    assert messages.messages[-1]["kind"] == "fatal_error"
    assert "首个随机访问包缺少" in messages.messages[-1]["error"]


def test_pyav_worker_bounds_wait_for_first_keyframe(monkeypatch) -> None:
    from airdrop.video import source as source_module

    class FakePacket:
        size = 1
        is_keyframe = False
        is_corrupt = False
        pts = 1
        dts = 1
        time_base = Fraction(1, 30)

    class FakeStream:
        thread_type = "AUTO"

    class FakeContainer:
        streams = type("Streams", (), {"video": [FakeStream()]})()

        def demux(self, _stream):
            yield FakePacket()
            yield FakePacket()

        def close(self) -> None:
            pass

    class FakeQueue:
        def __init__(self) -> None:
            self.messages = []

        def put(self, message, timeout) -> None:
            self.messages.append(message)

    class FakeStopEvent:
        def is_set(self) -> bool:
            return False

    clock = iter((0.0, 31.0))
    monkeypatch.setattr("av.open", lambda *_args, **_kwargs: FakeContainer())
    monkeypatch.setattr(source_module.time, "monotonic", lambda: next(clock))
    messages = FakeQueue()
    source_module._pyav_capture_worker(
        VideoConfig(url="udp://127.0.0.1:1234", width=2, height=2),
        None,
        1,
        messages,
        queue.Queue(),
        (),
        12,
        FakeStopEvent(),
    )
    assert messages.messages[-1]["kind"] == "stream_error"
    assert "超过 30 秒" in messages.messages[-1]["error"]


def test_source_stop_force_terminates_blocked_pyav_worker() -> None:
    context = multiprocessing.get_context("spawn")
    stop_event = context.Event()
    worker = context.Process(target=_block_worker_for_test, args=(stop_event,))
    worker.start()
    source = Hm30VideoSource()
    source.segment_lifecycle.opened(1)
    source._active_pyav_segment = 1
    source._pyav_process = worker
    source._pyav_stop_event = stop_event
    started = time.monotonic()
    source.stop(timeout=1.0)
    assert time.monotonic() - started < 2.0
    assert not worker.is_alive()
    assert source.segment_lifecycle.snapshot(1).state is SegmentState.CLOSED_INCOMPLETE


def test_source_marks_segment_failed_if_worker_cannot_be_terminated() -> None:
    class StuckWorker:
        def is_alive(self) -> bool:
            return True

        def terminate(self) -> None:
            return

        def join(self, timeout: float) -> None:
            del timeout

    source = Hm30VideoSource()
    source.segment_lifecycle.opened(3)
    source._active_pyav_segment = 3
    worker = StuckWorker()
    messages = object()
    slots: list[SharedMemory] = []
    source._pyav_process = worker
    source._pyav_messages = messages
    source._pyav_shared_slots = slots
    with pytest.raises(VideoStorageError, match="禁止继续启动"):
        source._terminate_pyav_worker(0.0)
    assert source.segment_lifecycle.snapshot(3).state is SegmentState.FAILED
    assert source._pyav_process is worker
    assert source._pyav_messages is messages
    assert source._pyav_shared_slots is slots


def test_live_pyav_worker_blocks_starting_another_segment(monkeypatch) -> None:
    class StuckWorker:
        def is_alive(self) -> bool:
            return True

    source = Hm30VideoSource(config=VideoConfig(width=64, height=48))
    worker = StuckWorker()
    source._pyav_process = worker

    def fail_if_started(*_args, **_kwargs):
        pytest.fail("新采集周期不得覆盖仍存活的 PyAV worker")

    monkeypatch.setattr("airdrop.video.source.multiprocessing.get_context", fail_if_started)
    with pytest.raises(VideoStorageError, match="上一 PyAV 工作进程仍存活"):
        source._stream_pyav_process()
    assert source._pyav_process is worker


def test_pyav_shared_memory_creation_failure_releases_prior_slots(monkeypatch) -> None:
    from airdrop.video import source as source_module

    real_shared_memory = SharedMemory
    created: list[SharedMemory] = []

    def create_slot(*, create: bool, size: int) -> SharedMemory:
        if created:
            raise OSError("injected second-slot failure")
        slot = real_shared_memory(create=create, size=size)
        created.append(slot)
        return slot

    monkeypatch.setattr(source_module, "SharedMemory", create_slot)
    source = Hm30VideoSource(config=VideoConfig(width=64, height=48))
    with pytest.raises(OSError, match="second-slot"):
        source._stream_pyav_process()
    assert source._pyav_shared_slots == []
    assert source._pyav_process is None
    with pytest.raises(FileNotFoundError):
        real_shared_memory(name=created[0].name)


def test_killed_large_shared_frame_cannot_block_control_queue_reader() -> None:
    context = multiprocessing.get_context("spawn")
    slot = SharedMemory(create=True, size=64 * 1024 * 1024)
    messages = context.Queue(maxsize=2)
    started = context.Event()
    worker = context.Process(target=_write_large_shared_slot_for_test, args=(slot.name, started))
    worker.start()
    source = Hm30VideoSource()
    source._pyav_process = worker
    source._pyav_stop_event = context.Event()
    assert started.wait(timeout=3.0)
    source._terminate_pyav_worker(1.0)
    started_at = time.monotonic()
    with pytest.raises(queue.Empty):
        messages.get(timeout=0.25)
    assert time.monotonic() - started_at < 0.5
    messages.close()
    slot.close()
    slot.unlink()


def test_pyav_grandchild_exits_when_parent_is_force_killed() -> None:
    context = multiprocessing.get_context("spawn")
    started = context.Event()
    child_pid = context.Queue(maxsize=1)
    parent = context.Process(
        target=_supervise_parent_guarded_blocker_for_test,
        args=(started, child_pid),
    )
    parent.start()
    grandchild_pid = None
    try:
        assert started.wait(timeout=5.0)
        grandchild_pid = child_pid.get(timeout=2.0)
        parent.terminate()
        parent.join(timeout=2.0)
        assert not parent.is_alive()

        kernel32 = ctypes.windll.kernel32
        wait_timeout = 0x00000102

        def process_alive(pid: int) -> bool:
            handle = kernel32.OpenProcess(0x00100000, False, pid)
            if not handle:
                return False
            try:
                return kernel32.WaitForSingleObject(handle, 0) == wait_timeout
            finally:
                kernel32.CloseHandle(handle)

        deadline = time.monotonic() + 5.0
        while process_alive(grandchild_pid) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not process_alive(grandchild_pid)
    finally:
        if parent.is_alive():
            parent.terminate()
            parent.join(timeout=2.0)
        if grandchild_pid is not None:
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.OpenProcess(0x0001, False, grandchild_pid)
            if handle:
                kernel32.TerminateProcess(handle, 1)
                kernel32.CloseHandle(handle)
        child_pid.close()


@pytest.mark.stream
def test_pyav_grandchild_exits_when_parent_dies_during_udp_open() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as receiver:
        receiver.bind(("127.0.0.1", 0))
        port = receiver.getsockname()[1]
    config = VideoConfig(
        url=f"udp://127.0.0.1:{port}",
        width=64,
        height=48,
        read_timeout=30.0,
    )
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    child_pid = context.Queue(maxsize=1)
    slot_names = context.Queue(maxsize=1)
    parent = context.Process(
        target=_supervise_pyav_udp_blocker_for_test,
        args=(config, ready, child_pid, slot_names),
    )
    parent.start()
    grandchild_pid = None
    names: tuple[str, ...] = ()
    try:
        assert ready.wait(timeout=5.0)
        grandchild_pid = child_pid.get(timeout=2.0)
        names = slot_names.get(timeout=2.0)

        kernel32 = ctypes.windll.kernel32
        wait_timeout = 0x00000102

        def process_alive(pid: int) -> bool:
            handle = kernel32.OpenProcess(0x00100000, False, pid)
            if not handle:
                return False
            try:
                return kernel32.WaitForSingleObject(handle, 0) == wait_timeout
            finally:
                kernel32.CloseHandle(handle)

        time.sleep(1.0)
        assert process_alive(grandchild_pid), "UDP 输入应保持阻塞直到父进程被结束"
        parent.terminate()
        parent.join(timeout=2.0)
        assert not parent.is_alive()

        deadline = time.monotonic() + 5.0
        while process_alive(grandchild_pid) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not process_alive(grandchild_pid)
    finally:
        if parent.is_alive():
            parent.terminate()
            parent.join(timeout=2.0)
        for name in names:
            try:
                slot = SharedMemory(name=name)
            except FileNotFoundError:
                continue
            slot.close()
            slot.unlink()
        child_pid.close()
        slot_names.close()


def test_pyav_subprocess_reconnect_uses_new_segment_and_continuous_indices(tmp_path) -> None:
    source_path = tmp_path / "source.mp4"
    frames = _make_h264(source_path, frames=12, gop=6)
    source = Hm30VideoSource(config=VideoConfig(url=str(source_path), width=64, height=48))
    source.set_recording_directory(tmp_path / "cold")
    published: list[VideoFrame] = []
    source.add_sink(lambda frame: published.append(frame.copy()))
    for _ in range(2):
        with pytest.raises(RuntimeError, match="输入流结束"):
            source._stream_pyav_process()
    assert len(published) == len(frames) * 2
    assert [frame.index for frame in published] == list(range(1, len(published) + 1))
    assert {frame.segment for frame in published[: len(frames)]} == {1}
    assert {frame.segment for frame in published[len(frames) :]} == {2}
    for segment in (1, 2):
        assert (tmp_path / "cold" / f"segment-{segment:06d}.complete").is_file()
        state = source.segment_lifecycle.snapshot(segment)
        assert state.state is SegmentState.CLOSED_COMPLETE
        assert state.revision >= 2
    assert (
        source.segment_lifecycle.snapshot(1).revision
        < source.segment_lifecycle.snapshot(2).revision
    )


def test_pyav_subprocess_writer_failure_leaves_segment_unverified(tmp_path) -> None:
    source_path = tmp_path / "source.mp4"
    _make_h264(source_path, frames=12, gop=6)
    output_dir = tmp_path / "not-a-directory"
    output_dir.write_bytes(b"blocked")
    source = Hm30VideoSource(config=VideoConfig(url=str(source_path), width=64, height=48))
    source._recording_directory = output_dir
    with pytest.raises(VideoStorageError, match="FileExistsError"):
        source._stream_pyav_process()
    assert not (output_dir / "segment-000001.complete").exists()


def test_pyav_subprocess_graceful_stop_drains_accepted_frames(tmp_path) -> None:
    source_path = tmp_path / "source.mp4"
    _make_h264(source_path, frames=48, gop=12)
    output_dir = tmp_path / "cold"
    source = Hm30VideoSource(config=VideoConfig(url=str(source_path), width=64, height=48))
    source.set_recording_directory(output_dir)
    first_frame = threading.Event()
    published: list[VideoFrame] = []

    def sink(frame: VideoFrame) -> None:
        published.append(frame.copy())
        first_frame.set()
        time.sleep(0.01)

    source.add_sink(sink)
    errors: list[Exception] = []

    def run_capture() -> None:
        try:
            source._stream_pyav_process()
        except Exception as exc:  # noqa: BLE001 - 线程错误统一检查
            errors.append(exc)

    capture_thread = threading.Thread(target=run_capture)
    capture_thread.start()
    assert first_frame.wait(timeout=5.0)
    source._thread = capture_thread
    source.stop(timeout=5.0)
    assert not capture_thread.is_alive()
    assert errors == []
    assert published
    assert (output_dir / "segment-000001.complete").is_file()


def test_wait_new_reads_verified_cold_frames_with_sequential_cursor(tmp_path, monkeypatch) -> None:
    source_path = tmp_path / "source.mp4"
    source_frames = _make_h264(source_path)
    input_container = av.open(str(source_path), mode="r")
    input_stream = input_container.streams.video[0]
    writer = TsSegmentWriter(tmp_path, segment=1, input_stream=input_stream)
    for packet in input_container.demux(input_stream):
        packet.decode()
        writer.mux(packet)
    writer.close()
    input_container.close()

    decoder = TsRangeDecoder(tmp_path, max_cache_bytes=20_000)
    decoded_cache = decoder.decoded_cache
    assert decoded_cache is not None
    buffer = AlignmentBuffer(
        capacity=10,
        storage="raw",
        cold_decoder=decoder,
        hot_seconds=1.0,
        max_hot_bytes=2_000_000,
    )
    decode_calls = 0
    original_put = decoder._put_cache

    def counted_put(key, image) -> None:
        nonlocal decode_calls
        decode_calls += 1
        original_put(key, image)

    decoder._put_cache = counted_put
    for index, (pts, time_base, image) in enumerate(source_frames):
        capture_timestamp = index / 12
        frame = VideoFrame(
            index=index + 1,
            image=image,
            timestamp=capture_timestamp + 0.15,
            lag=0.15,
            media_pts=pts,
            time_base_num=time_base.numerator,
            time_base_den=time_base.denominator,
            segment=1,
            random_access_dts=0,
            random_access_time_base_num=time_base.numerator,
            random_access_time_base_den=time_base.denominator,
        )
        buffer.put(
            AlignedSample(
                frame=frame,
                snapshot=TelemetrySnapshot(timestamp=capture_timestamp),
                timestamp=capture_timestamp,
                lag=0.15,
                mode="test",
            ),
            materialize=False,
        )

    deadline = time.monotonic() + 5.0
    while buffer.stats.cold_pending and time.monotonic() < deadline:
        time.sleep(0.01)
    assert buffer.stats.cold_pending == 0
    assert buffer.stats.cold_verified == len(source_frames)
    assert buffer.stats.cold_not_ready == 0
    cold_count = sum(isinstance(record.payload, ColdFrameRef) for record in buffer._records)
    assert 20 <= cold_count < len(source_frames), "校验完成后仍保留约 1 秒热帧"
    assert len(buffer) == len(source_frames), "冷存储索引不能按热区容量驱逐历史帧"
    assert (
        sum(record.nbytes for record in buffer._records if isinstance(record.payload, np.ndarray))
        <= 200_000
    )

    cursor = 0
    for index, (_, _, expected) in enumerate(source_frames, start=1):
        record = buffer.wait_new(after_index=cursor, timeout=0)
        assert record is not None
        assert record.index == index
        assert np.array_equal(record.image, expected)
        cursor = record.index
    assert decode_calls <= len(source_frames) * 2
    frozen = buffer.freeze(after_index=0)
    frozen_second = buffer.freeze(after_index=0)
    assert all(isinstance(record.payload, ColdFrameRef) for record in frozen._records)
    assert any(isinstance(record.payload, np.ndarray) for record in buffer._records)
    try:
        for view in (frozen, frozen_second):
            assert view._cold_thread is None and view._cold_worker_decoder is None
            assert (
                view._cold_decoder is not None and view._cold_decoder.decoded_cache is decoded_cache
            )
            for index, (_, _, expected) in enumerate(source_frames, start=1):
                record = view.wait_new(after_index=index - 1, timeout=0)
                assert record is not None and np.array_equal(record.image, expected)
                assert decoded_cache.byte_size <= decoder.max_cache_bytes

        # 最后一个冻结视图仍持有缓存，重复读取末帧应直接命中。
        buffer.close_cold_storage()
        frozen.close_cold_storage()
        assert 0 < decoded_cache.byte_size <= decoder.max_cache_bytes
        monkeypatch.setattr(
            av, "open", lambda *args, **kwargs: pytest.fail("缓存命中时重新打开了 TS")
        )
        record = frozen_second.at(len(source_frames))
        assert record is not None and np.array_equal(record.image, source_frames[-1][2])
    finally:
        frozen.close_cold_storage()
        frozen_second.close_cold_storage()
        buffer.close_cold_storage()
        decoder.close()
    assert decoded_cache.byte_size == 0, "最后一个冷读取租约关闭后缓存应清空"


def test_decoded_cache_is_shared_across_decoders_with_independent_leases(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_path = tmp_path / "source.mp4"
    source_frames = _make_h264(source_path, frames=12)
    source_container = av.open(str(source_path), mode="r")
    source_stream = source_container.streams.video[0]
    writer = TsSegmentWriter(tmp_path, segment=1, input_stream=source_stream)
    for packet in source_container.demux(source_stream):
        packet.decode()
        writer.mux(packet)
    writer.close()
    source_container.close()

    main = TsRangeDecoder(tmp_path, max_cache_bytes=9_216)
    view = TsRangeDecoder(tmp_path, decoded_cache=main.decoded_cache)
    pts, time_base, expected = source_frames[5]
    image = main.read(1, pts, time_base.numerator, time_base.denominator)
    assert image is not None and np.array_equal(image, expected)
    image[:] = 0
    assert main._cache_bytes <= main.max_cache_bytes

    main.close()
    original_open = av.open
    monkeypatch.setattr(av, "open", lambda *args, **kwargs: pytest.fail("缓存命中不应重新解码"))
    cached = view.read(1, pts, time_base.numerator, time_base.denominator)
    assert cached is not None and np.array_equal(cached, expected)
    assert view._cache_bytes <= view.max_cache_bytes

    view.close()
    assert view._cache_bytes == 0, "最后一个解码器租约释放后应清空缓存"
    monkeypatch.setattr(av, "open", original_open)
    reread = main.read(1, pts, time_base.numerator, time_base.denominator)
    assert reread is not None and np.array_equal(reread, expected)
    main.close()


def test_decoded_cache_skips_oversized_images_and_isolates_sessions(tmp_path) -> None:
    from airdrop.video.cache import DecodedFrameCache

    cache = DecodedFrameCache(8)
    cache.acquire()
    key = (cache.storage_identity, str(tmp_path.resolve()), 1, 0, 1, 0, 0)
    image = np.arange(9, dtype=np.uint8).reshape((3, 3))
    cache.put(key, image)
    assert cache.byte_size == 0
    assert cache.get(key) is None

    small = np.arange(4, dtype=np.uint8).reshape((2, 2))
    cache.put(key, small)
    hit = cache.get(key)
    assert hit is not None
    hit[:] = 99
    isolated_hit = cache.get(key)
    assert isolated_hit is not None and np.array_equal(isolated_hit, small)
    other = DecodedFrameCache(8)
    assert other.storage_identity != cache.storage_identity
    other.acquire()
    other.put((other.storage_identity, str(tmp_path.resolve()), 1, 0, 1, 0, 0), image[:2, :2])
    assert other.get(key) is None
    cache.release()
    cache.put(key, small)
    assert cache.byte_size == 0, "零租约时拒绝迟到写入"
    other.release()


def test_disabled_decoder_rejects_shared_cache() -> None:
    from airdrop.video.cache import DecodedFrameCache

    cache = DecodedFrameCache(16)
    with pytest.raises(ValueError, match="禁用缓存的解码器不能绑定共享缓存"):
        TsRangeDecoder("unused", decoded_cache=cache, cache_enabled=False)


def test_decoder_constructor_releases_cache_lease_when_reaper_start_fails(
    tmp_path, monkeypatch
) -> None:
    from airdrop.video.cache import DecodedFrameCache

    cache = DecodedFrameCache(16)
    start_error = RuntimeError("injected reaper start failure")
    original_start = threading.Thread.start

    def fail_reaper_start(thread: threading.Thread) -> None:
        if thread.name == "ts-decoder-cursor-reaper":
            raise start_error
        original_start(thread)

    monkeypatch.setattr(threading.Thread, "start", fail_reaper_start)
    with pytest.raises(RuntimeError, match="injected reaper start failure") as caught:
        TsRangeDecoder(tmp_path, decoded_cache=cache)

    assert caught.value is start_error
    cache.put(("late",), np.zeros((2, 2), dtype=np.uint8))
    assert cache.byte_size == 0


def test_cache_put_and_close_release_are_synchronized(tmp_path, monkeypatch) -> None:
    decoder = TsRangeDecoder(tmp_path, max_cache_bytes=16)
    cache = decoder.decoded_cache
    assert cache is not None
    put_entered = threading.Event()
    allow_put = threading.Event()
    release_entered = threading.Event()
    close_done = threading.Event()
    failures: list[BaseException] = []
    original_put = cache.put
    original_release = decoder._release_cache_lease

    def blocked_put(key, image) -> None:
        put_entered.set()
        if not allow_put.wait(2):
            raise TimeoutError("测试未释放缓存写入")
        original_put(key, image)

    def tracked_release() -> None:
        release_entered.set()
        original_release()

    monkeypatch.setattr(cache, "put", blocked_put)
    monkeypatch.setattr(decoder, "_release_cache_lease", tracked_release)

    def write_cache() -> None:
        try:
            decoder._put_cache(("key",), np.zeros((2, 2), dtype=np.uint8))
        except BaseException as exc:  # noqa: BLE001 - 跨线程测试需回传失败
            failures.append(exc)

    def close_decoder() -> None:
        try:
            decoder.close()
        except BaseException as exc:  # noqa: BLE001 - 跨线程测试需回传失败
            failures.append(exc)
        finally:
            close_done.set()

    writer = threading.Thread(target=write_cache)
    closer = threading.Thread(target=close_decoder)
    writer.start()
    assert put_entered.wait(1)
    closer.start()
    assert release_entered.wait(1)
    assert not close_done.is_set(), "关闭应等待正在执行的缓存写入"
    allow_put.set()
    writer.join(2)
    closer.join(2)
    assert not writer.is_alive() and not closer.is_alive()
    assert not failures
    assert cache.byte_size == 0, "最后租约关闭应清除交错写入的数据"


def test_close_releases_lease_when_cursor_cleanup_fails(tmp_path, monkeypatch) -> None:
    main = TsRangeDecoder(tmp_path)
    cache = main.decoded_cache
    assert cache is not None
    view = TsRangeDecoder(tmp_path, decoded_cache=cache)
    cache.put(("key",), np.ones((2, 2), dtype=np.uint8))
    assert cache.byte_size == 4

    def broken_reset() -> None:
        raise RuntimeError("cursor cleanup failed")

    monkeypatch.setattr(main, "_reset_cursor_locked", broken_reset)
    with pytest.raises(RuntimeError, match="cursor cleanup failed"):
        main.close()
    assert cache.byte_size == 4, "仍有 view 租约时缓存必须保留"
    view.close()
    assert cache.byte_size == 0, "最后 lease 仍须在主 decoder 清理异常后释放"


def test_decoder_cache_key_includes_normalized_directory_and_output_size(tmp_path) -> None:
    decoder = TsRangeDecoder(tmp_path)
    try:
        cache = decoder.decoded_cache
        assert cache is not None
        media_time = Fraction(7, 30)
        original = decoder._make_cache_key(3, media_time, 640, 480)
        decoder.directory = tmp_path / "another-voyage"
        changed_directory = decoder._make_cache_key(3, media_time, 640, 480)
        changed_size = decoder._make_cache_key(3, media_time, 320, 240)
        assert original != changed_directory
        assert original != changed_size
        assert original[0] == changed_directory[0] == cache.storage_identity
    finally:
        decoder.close()


def test_cold_release_visits_only_new_prefix_after_long_history(tmp_path) -> None:
    """冷帧释放游标避免每帧扫描已释放的长期历史。"""
    decoder = TsRangeDecoder(tmp_path)
    buffer = AlignmentBuffer(
        storage="raw", cold_decoder=decoder, _start_cold_worker=False, max_hot_bytes=10_000
    )
    image = np.zeros((1, 1, 3), np.uint8)
    sample = AlignedSample(
        frame=VideoFrame(index=1, image=image, timestamp=1, lag=0),
        snapshot=TelemetrySnapshot(timestamp=1),
        timestamp=1,
        lag=0,
        mode="test",
    )
    buffer.put(sample, materialize=False)
    template = buffer._records[0]
    reference = ColdFrameRef(1, 1, 1, 1)

    class CountingRecords(list):
        reads = 0

        def __getitem__(self, index):
            self.reads += 1
            return super().__getitem__(index)

    history_count = 108_000
    recent_count = 60
    records = [
        replace(
            template,
            index=index,
            payload=reference,
            cold_locator=reference,
            nbytes=1,
        )
        for index in range(1, history_count + 1)
    ]
    records.extend(
        replace(
            template,
            index=index,
            payload=image.copy(),
            cold_locator=reference,
            nbytes=image.nbytes,
        )
        for index in range(history_count + 1, history_count + recent_count + 1)
    )
    with buffer._condition:
        buffer._records = CountingRecords(records)
        buffer._next_index = history_count + recent_count
        buffer._hot_bytes = recent_count * image.nbytes
        buffer._bytes = history_count + recent_count * image.nbytes
        buffer._cold_frame_count = history_count
        buffer._cold_release_index = history_count
        buffer._cold_schedule_limit = buffer._next_index
        buffer._cold_next_schedule = buffer._next_index + 1
        buffer._release_eligible_locked(float("inf"))
        visited = buffer._records.reads

    assert visited == recent_count
    assert buffer.stats.cold_frames == history_count + recent_count
    assert buffer.stats.hot_bytes == 0
    buffer.close_cold_storage()


def test_cold_buffer_clear_resets_hot_state_and_can_be_reused(tmp_path) -> None:
    source_path = tmp_path / "source.mp4"
    source_frames = _make_h264(source_path, frames=12)
    input_container = av.open(str(source_path), mode="r")
    input_stream = input_container.streams.video[0]
    writer = TsSegmentWriter(tmp_path, segment=1, input_stream=input_stream)
    for packet in input_container.demux(input_stream):
        packet.decode()
        writer.mux(packet)
    writer.close()
    input_container.close()

    decoder = TsRangeDecoder(tmp_path)
    buffer = AlignmentBuffer(
        storage="raw", cold_decoder=decoder, hot_seconds=0.1, max_hot_bytes=1_000_000
    )

    def put(index: int) -> None:
        pts, time_base, image = source_frames[index]
        timestamp = index / 12
        buffer.put(
            AlignedSample(
                frame=VideoFrame(
                    index=index + 1,
                    image=image,
                    timestamp=timestamp,
                    lag=0,
                    media_pts=pts,
                    time_base_num=time_base.numerator,
                    time_base_den=time_base.denominator,
                    segment=1,
                ),
                snapshot=TelemetrySnapshot(timestamp=timestamp),
                timestamp=timestamp,
                lag=0,
                mode="test",
            ),
            materialize=False,
        )

    put(0)
    deadline = time.monotonic() + 5
    while buffer.stats.cold_pending and time.monotonic() < deadline:
        time.sleep(0.01)
    buffer.clear()
    assert buffer.stats.hot_bytes == 0
    assert buffer.stats.frames == 0
    assert buffer.stats.cold_verified == 0
    put(0)
    buffer.close_cold_storage()
    assert buffer.stats.hot_bytes == 0
    assert buffer.stats.cold_frames == 1
    buffer.close_cold_storage()


def test_cold_buffer_fails_before_appending_after_fatal_error(tmp_path) -> None:
    decoder = TsRangeDecoder(tmp_path)
    buffer = AlignmentBuffer(storage="raw", cold_decoder=decoder)
    frame = VideoFrame(index=1, image=np.zeros((2, 2, 3), np.uint8), timestamp=1, lag=0)
    sample = AlignedSample(
        frame=frame,
        snapshot=TelemetrySnapshot(timestamp=1),
        timestamp=1,
        lag=0,
        mode="test",
    )
    with buffer._condition:
        buffer._cold_error = "injected failure"
    with pytest.raises(OSError, match="injected failure"):
        buffer.put(sample, materialize=False)
    assert buffer.stats.frames == 0
    assert buffer.stats.last_index == 0
    with buffer._condition:
        buffer._cold_stopping = True
        buffer._condition.notify_all()
    assert buffer._cold_thread is not None
    buffer._cold_thread.join(timeout=1)


def test_cold_storage_close_attempts_both_decoders_after_one_fails(tmp_path) -> None:
    decoder = TsRangeDecoder(tmp_path)
    buffer = AlignmentBuffer(storage="raw", cold_decoder=decoder)
    worker_decoder = buffer._cold_worker_decoder
    assert worker_decoder is not None
    worker_close = worker_decoder.close
    main_closed = False
    main_close = decoder.close

    def failing_worker_close() -> None:
        worker_close()
        raise OSError("模拟校验解码器关闭失败")

    def tracked_main_close() -> None:
        nonlocal main_closed
        main_closed = True
        main_close()

    worker_decoder.close = failing_worker_close
    decoder.close = tracked_main_close
    with pytest.raises(RuntimeError, match="校验解码器关闭失败"):
        buffer.close_cold_storage()

    assert main_closed
    decoder.close()


def test_ts_flight_recorder_replay_and_calibration_reader_round_trip(
    tmp_path, monkeypatch: pytest.MonkeyPatch, caplog, request: pytest.FixtureRequest
) -> None:
    source_path = tmp_path / "source.mp4"
    source_frames = _make_h264(source_path, frames=24, gop=12)
    monkeypatch.setattr(
        cv2,
        "imencode",
        lambda *_args, **_kwargs: pytest.fail("raw TS 路径不得 JPEG 编码"),
    )
    buffer = build_alignment_buffer(capture=True, ts_directory=tmp_path)
    assert buffer is not None
    request.addfinalizer(buffer.close_cold_storage)
    recorder = FlightRecorder(
        Config(record=RecordConfig(telemetry_hz=0)), base_dir=tmp_path / "flights"
    )
    recorder.start(broker=TelemetryBroker(), buffer=buffer, ts_video=True)
    assert recorder.video_directory is not None
    buffer.bind_cold_directory(recorder.video_directory)
    request.addfinalizer(recorder.stop)
    source_container = av.open(str(source_path), mode="r")
    source_stream = source_container.streams.video[0]
    writer = TsSegmentWriter(recorder.video_directory, segment=1, input_stream=source_stream)
    for packet in source_container.demux(source_stream):
        packet.decode()
        writer.mux(packet)
    writer.close()
    source_container.close()

    for index, (pts, time_base, image) in enumerate(source_frames):
        timestamp = index / 12
        buffer.put(
            AlignedSample(
                frame=VideoFrame(
                    index=index + 1,
                    image=image,
                    timestamp=timestamp + 0.15,
                    lag=0.15,
                    media_pts=pts,
                    time_base_num=time_base.numerator,
                    time_base_den=time_base.denominator,
                    segment=1,
                    random_access_dts=0,
                    random_access_time_base_num=time_base.numerator,
                    random_access_time_base_den=time_base.denominator,
                ),
                snapshot=(
                    None
                    if index == 0
                    else TelemetrySnapshot(
                        timestamp=timestamp,
                        latitude_deg=47.0 + index * 1e-5,
                        yaw_deg=float(index),
                    )
                ),
                timestamp=timestamp,
                lag=0.15,
                mode="test",
            ),
            materialize=False,
        )
    recorder.stop()
    deadline = time.monotonic() + 5.0
    while buffer.stats.cold_verified < 1 and time.monotonic() < deadline:
        time.sleep(0.01)
    cold_first = buffer.at(1)
    hot_last = buffer.at(len(source_frames))
    assert buffer.stats.storage == "raw"
    assert buffer.stats.cold_frames > 0
    assert buffer.stats.frames > buffer.stats.cold_frames
    assert buffer.stats.hot_bytes > 0
    assert buffer.read_jpeg_bytes(1) is None
    assert buffer.stats.frames == len(source_frames)
    assert cold_first is not None and hot_last is not None
    assert np.array_equal(cold_first.image, source_frames[0][2])
    assert cold_first.media_pts == source_frames[0][0]
    assert cold_first.snapshot is None
    assert hot_last.index == len(source_frames)
    assert hot_last.media_pts == source_frames[-1][0]
    assert hot_last.capture_timestamp == pytest.approx((len(source_frames) - 1) / 12)
    assert hot_last.snapshot is not None
    assert hot_last.snapshot.yaw_deg == float(len(source_frames) - 1)
    flight_dir = recorder.flight_dir
    assert flight_dir is not None
    assert not (flight_dir / "frames").exists()

    from airdrop.runtime.replay import _build_replay_buffer

    replay = ReplayVideoSource(flight_dir, speed=0.0, strict=True)
    replay_buffer, _output_size = _build_replay_buffer(replay, flight_dir)
    request.addfinalizer(replay_buffer.close_cold_storage)
    replay_writer = AlignmentWriter(replay_buffer, FrameTelemetryAligner(TelemetryBroker()))
    replay.add_sink(replay_writer)
    replayed: list[VideoFrame] = []
    replay.add_sink(lambda frame: replayed.append(frame.copy()))
    request.addfinalizer(replay.stop)
    replay.start()
    assert replay.wait_finished(timeout=10.0)
    replay.stop()
    assert len(replayed) == len(source_frames)
    deadline = time.monotonic() + 5.0
    while replay_buffer.stats.cold_verified < 1 and time.monotonic() < deadline:
        time.sleep(0.01)
    replay_cold_first = replay_buffer.at(1)
    replay_hot_last = replay_buffer.at(len(source_frames))
    assert replay_buffer.stats.storage == "raw"
    assert replay_buffer.stats.frames == len(source_frames)
    assert replay_buffer.stats.cold_frames > 0
    assert replay_buffer.stats.hot_bytes > 0
    assert replay_buffer.read_jpeg_bytes(1) is None
    assert replay_cold_first is not None and replay_hot_last is not None
    assert replay_cold_first.media_pts == source_frames[0][0]
    assert replay_cold_first.capture_timestamp == pytest.approx(0.0)
    assert replay_cold_first.snapshot is None
    assert np.array_equal(replay_cold_first.image, source_frames[0][2])
    assert replay_hot_last.index == len(source_frames)
    assert replay_hot_last.media_pts == source_frames[-1][0]
    assert replay_hot_last.capture_timestamp == pytest.approx((len(source_frames) - 1) / 12)
    assert replay_hot_last.snapshot is not None
    assert replay_hot_last.snapshot.yaw_deg == float(len(source_frames) - 1)
    assert np.array_equal(replay_hot_last.image, source_frames[-1][2])
    aligned = replay.frames[7].aligned_telemetry
    assert aligned is not None
    assert aligned["timestamp"] == pytest.approx(7 / 12)
    assert aligned["latitude_deg"] == pytest.approx(47.00007)
    assert aligned["yaw_deg"] == 7.0
    assert replayed[7].aligned_snapshot is not None
    assert replayed[0].alignment_disabled
    capture_only = FrameTelemetryAligner(TelemetryBroker()).align(replayed[0])
    assert capture_only is not None
    assert capture_only.mode == "capture_only"
    assert capture_only.snapshot is None
    direct_alignment = FrameTelemetryAligner(TelemetryBroker()).align(replayed[7])
    assert direct_alignment is not None
    assert direct_alignment.mode == "recorded"
    assert direct_alignment.snapshot is not None
    assert direct_alignment.snapshot.latitude_deg == pytest.approx(47.00007)
    for actual, expected in zip(replayed, source_frames, strict=True):
        assert actual.media_pts == expected[0]
        assert np.array_equal(actual.image, expected[2])

    from airdrop_calibration.flight_log import iter_flight_frames_paired

    calibrated = list(iter_flight_frames_paired(flight_dir))
    assert len(calibrated) == len(source_frames)
    for (_, name, actual), expected in zip(calibrated, source_frames, strict=True):
        assert name.startswith("ts:1:")
        assert np.array_equal(actual, expected[2])

    index_path = flight_dir / "frames_index.jsonl"
    complete_index = index_path.read_text(encoding="utf-8")
    index_lines = complete_index.splitlines()
    index_lines[-1] = index_lines[-1][:-8]
    index_path.write_text("\n".join(index_lines[:-1]) + "\n" + index_lines[-1], encoding="utf-8")
    events_path = flight_dir / "events.jsonl"
    event_lines = events_path.read_text(encoding="utf-8").splitlines()
    events_path.write_text("\n".join(event_lines[:-1]) + "\n", encoding="utf-8")
    from airdrop_calibration import calibrate as calibration_module

    recovered = calibration_module._validate_full_update_capture(flight_dir, [])
    assert recovered is not None
    assert recovered["scope"] == "verified_prefix"
    assert recovered["complete"] is False
    assert recovered["frame_count"] == len(source_frames) - 1

    prefix_replay = ReplayVideoSource(flight_dir, speed=0.0)
    prefix_frames: list[VideoFrame] = []
    prefix_replay.add_sink(lambda frame: prefix_frames.append(frame.copy()))
    prefix_replay.start()
    assert prefix_replay.wait_finished(timeout=10.0)
    prefix_replay.stop()
    assert len(prefix_frames) == len(source_frames) - 1

    class FailingReader:
        closed = False

        def __enter__(self):
            return self

        def __exit__(self, *exc_info) -> None:
            del exc_info
            self.closed = True

        def read(self, record: dict[str, object]) -> np.ndarray:
            del record
            raise OSError("injected calibration decode failure")

    failing_reader = FailingReader()
    with monkeypatch.context() as patcher:
        patcher.setattr(calibration_module, "TsFrameReader", lambda _path: failing_reader)
        with pytest.raises(ValueError):
            calibration_module._validate_full_update_capture(flight_dir, [])
    assert failing_reader.closed

    index_lines = complete_index.splitlines()
    tail = json.loads(index_lines[-1])
    tail["media_pts"] += 100_000
    index_lines[-1] = json.dumps(tail)
    previous_tail = json.loads(index_lines[-2])
    previous_tail["media_pts"] += 100_000
    index_lines[-2] = json.dumps(previous_tail)
    index_path.write_text("\n".join(index_lines) + "\n", encoding="utf-8")
    strict_replay = ReplayVideoSource(flight_dir, speed=0.0, strict=True)
    strict_replay.start()
    assert strict_replay.wait_finished(timeout=10.0)
    assert strict_replay.stats.state == "error"
    strict_replay.stop()
    damaged_tail = calibration_module._validate_full_update_capture(flight_dir, [])
    assert damaged_tail is not None and damaged_tail["frame_count"] == len(source_frames) - 2
    assert "尾部帧无法验证" in caplog.text
    tolerant_replay = ReplayVideoSource(flight_dir, speed=0.0)
    tolerant_frames: list[VideoFrame] = []
    tolerant_replay.add_sink(lambda frame: tolerant_frames.append(frame.copy()))
    tolerant_replay.start()
    assert tolerant_replay.wait_finished(timeout=10.0)
    tolerant_replay.stop()
    assert len(tolerant_frames) == len(source_frames) - 2
    assert len(list(iter_flight_frames_paired(flight_dir))) == len(source_frames) - 2

    index_lines = complete_index.splitlines()
    tail = json.loads(index_lines[-1])
    tail["aligned_telemetry"]["timestamp"] = "not-a-number"
    tail["aligned_telemetry"]["yaw_deg"] = {"invalid": True}
    index_lines[-1] = json.dumps(tail)
    index_path.write_text("\n".join(index_lines) + "\n", encoding="utf-8")
    invalid_snapshot_replay = ReplayVideoSource(flight_dir, speed=0.0)
    valid_frames: list[VideoFrame] = []
    invalid_snapshot_replay.add_sink(lambda frame: valid_frames.append(frame.copy()))
    invalid_snapshot_replay.start()
    assert invalid_snapshot_replay.wait_finished(timeout=10.0)
    invalid_snapshot_replay.stop()
    assert len(valid_frames) == len(source_frames) - 1


@pytest.mark.integration
def test_forced_parent_termination_keeps_verified_ts_prefix(tmp_path, caplog) -> None:
    source_path = tmp_path / "source.mp4"
    source_frames = _make_h264(source_path, frames=48, gop=12)
    expected_by_time = {Fraction(pts) * time_base: image for pts, time_base, image in source_frames}
    context = multiprocessing.get_context("spawn")
    ready = context.Queue(maxsize=1)
    parent = context.Process(
        target=_abrupt_recording_parent_for_test,
        args=(str(source_path), str(tmp_path / "flights"), ready),
    )
    parent.start()
    try:
        result = ready.get(timeout=20.0)
        flight_dir = Path(result["flight_dir"])
        worker_pid = int(result["worker_pid"])
        parent.terminate()
        parent.join(timeout=5.0)
        assert not parent.is_alive()

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.restype = ctypes.c_void_p
        kernel32.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
        kernel32.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
        process_handle = kernel32.OpenProcess(0x00100000, False, worker_pid)
        assert process_handle, "无法检查 PyAV 子进程生命周期"
        try:
            assert kernel32.WaitForSingleObject(process_handle, 10_000) == 0
        finally:
            kernel32.CloseHandle(process_handle)

        flight_dir = Path(result["flight_dir"])
        video_dir = flight_dir / "video"
        assert (video_dir / "segment-000001.ts").is_file()
        assert not (video_dir / "segment-000001.complete").exists()
        index_bytes = (flight_dir / "frames_index.jsonl").read_bytes()
        assert not index_bytes.endswith(b"\n")
        complete_rows = [
            json.loads(line)
            for line in index_bytes.splitlines(keepends=True)
            if line.endswith(b"\n")
        ]
        assert len(complete_rows) == result["frames"] - 1
        replay = ReplayVideoSource(flight_dir, speed=0.0)
        frames: list[VideoFrame] = []
        replay.add_sink(lambda frame: frames.append(frame.copy()))
        replay.start()
        assert replay.wait_finished(timeout=10.0)
        replay.stop()
        assert 2 <= len(frames) <= result["frames"]
        assert "保留可用前缀" in caplog.text
        for frame_index, frame in enumerate(frames, start=1):
            assert frame.media_pts is not None
            assert frame.time_base_num is not None and frame.time_base_den is not None
            media_time = Fraction(frame.media_pts * frame.time_base_num, frame.time_base_den)
            assert np.array_equal(frame.image, expected_by_time[media_time])
            assert frame.aligned_snapshot is not None
            assert frame.aligned_snapshot.yaw_deg == float(frame_index)
            assert frame.aligned_snapshot.latitude_deg == pytest.approx(47.0 + frame_index * 1e-5)

        from airdrop_calibration.flight_log import iter_flight_frames_paired

        calibrated_frames = list(iter_flight_frames_paired(flight_dir))
        assert len(calibrated_frames) == len(frames)
        for (_, name, image), replayed in zip(calibrated_frames, frames, strict=True):
            assert name.startswith("ts:1:")
            assert np.array_equal(image, replayed.image)

        from airdrop_calibration import calibrate as calibration_module

        recovered = calibration_module._validate_full_update_capture(flight_dir, [])
        assert recovered is not None
        assert recovered["scope"] == "verified_prefix"
        assert recovered["frame_count"] == len(frames)
    finally:
        if parent.is_alive():
            parent.terminate()
            parent.join(timeout=5.0)
        ready.close()
        ready.join_thread()


def test_wait_new_metadata_does_not_materialize_image(monkeypatch: pytest.MonkeyPatch) -> None:
    buffer = AlignmentBuffer(capacity=4, storage="raw")
    image = np.zeros((3, 5, 3), dtype=np.uint8)
    buffer.put(
        AlignedSample(
            frame=VideoFrame(index=1, image=image, timestamp=2.0, lag=0.25),
            snapshot=TelemetrySnapshot(timestamp=1.75, yaw_deg=23.0),
            timestamp=1.75,
            lag=0.25,
            mode="test",
        ),
        materialize=False,
    )

    def fail_if_materialized(stored: StoredFrame) -> BufferedFrame | None:
        del stored
        raise AssertionError("元数据游标不应读取或解码图像")

    monkeypatch.setattr(buffer, "_materialize_record", fail_if_materialized)
    metadata = buffer.wait_new_metadata(timeout=0.1)
    assert metadata is not None
    assert metadata.index == 1
    assert (metadata.width, metadata.height) == (5, 3)
    assert metadata.snapshot is not None and metadata.snapshot.yaw_deg == 23.0


def test_alignment_buffer_advances_watermark_during_active_ts_write(tmp_path) -> None:
    source_path = tmp_path / "source.mp4"
    source_frames = _make_h264(source_path)
    source_container = av.open(str(source_path), mode="r")
    source_stream = source_container.streams.video[0]
    writer = TsSegmentWriter(tmp_path, segment=1, input_stream=source_stream)
    decoder = TsRangeDecoder(tmp_path, max_cache_bytes=100_000)
    buffer = AlignmentBuffer(
        capacity=100,
        storage="raw",
        cold_decoder=decoder,
        hot_seconds=0.5,
        max_hot_bytes=2_000_000,
    )
    reference = {Fraction(pts) * time_base: image for pts, time_base, image in source_frames}
    index = 0
    try:
        for packet in source_container.demux(source_stream):
            decoded = packet.decode()
            writer.mux(packet)
            for frame in decoded:
                if frame.pts is None or frame.time_base is None:
                    raise AssertionError("合成输入解码帧缺少媒体时间")
                time_base = Fraction(frame.time_base.numerator, frame.time_base.denominator)
                image = frame.to_ndarray(format="bgr24")
                if not np.array_equal(image, reference[Fraction(frame.pts) * time_base]):
                    raise AssertionError("源解码参考帧不一致")
                index += 1
                capture_timestamp = (index - 1) / 12
                buffer.put(
                    AlignedSample(
                        frame=VideoFrame(
                            index=index,
                            image=image,
                            timestamp=capture_timestamp + 0.15,
                            lag=0.15,
                            media_pts=frame.pts,
                            time_base_num=time_base.numerator,
                            time_base_den=time_base.denominator,
                            segment=1,
                        ),
                        snapshot=TelemetrySnapshot(timestamp=capture_timestamp),
                        timestamp=capture_timestamp,
                        lag=0.15,
                        mode="test",
                    ),
                    materialize=False,
                )
    finally:
        writer.close()
        source_container.close()

    buffer.close_cold_storage(timeout=5.0)
    assert index == len(source_frames)
    assert buffer.stats.hot_bytes == 0
    assert buffer.stats.cold_frames == len(source_frames)
    for frame_index, (_, _, expected) in enumerate(source_frames, start=1):
        actual = buffer.wait_new(after_index=frame_index - 1, timeout=0)
        assert actual is not None
        assert np.array_equal(actual.image, expected)
    decoder.close()


def test_pyav_sink_failure_is_fatal_for_cold_capture() -> None:
    source = Hm30VideoSource()
    source.add_sink(lambda frame: (_ for _ in ()).throw(OSError("disk full")))

    with pytest.raises(RuntimeError, match="sink 执行失败"):
        source._publish(
            np.zeros((2, 2, 3), dtype=np.uint8),
            media_pts=1,
            time_base_num=1,
            time_base_den=30,
            fail_on_sink_error=True,
        )

    assert source.stats.sink_errors == 1
    assert "disk full" in (source.stats.last_error or "")


def test_pyav_remux_failure_does_not_publish_frame(tmp_path, monkeypatch) -> None:
    class FakeFrame:
        pts = 0
        time_base = Fraction(1, 12)

        def to_ndarray(self, *, format: str) -> np.ndarray:
            assert format == "bgr24"
            return np.zeros((48, 64, 3), dtype=np.uint8)

    class FakePacket:
        size = 1
        is_keyframe = True
        pts = 0
        dts = 0
        time_base = Fraction(1, 12)

        def decode(self) -> list[FakeFrame]:
            return [FakeFrame()]

    class FakeStream:
        thread_type = "AUTO"

    class FakeContainer:
        streams = type("Streams", (), {"video": [FakeStream()]})()

        def demux(self, stream):
            yield FakePacket()

        def close(self) -> None:
            pass

    class FailingWriter:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def mux(self, packet) -> None:
            raise OSError("disk full")

        def close(self) -> None:
            pass

    from airdrop.video import cold

    container = FakeContainer()
    monkeypatch.setattr(av, "open", lambda *args, **kwargs: container)
    monkeypatch.setattr(cold, "TsSegmentWriter", FailingWriter)
    source = Hm30VideoSource()
    source.set_recording_directory(tmp_path)
    published: list[np.ndarray] = []
    monkeypatch.setattr(source, "_publish", lambda image, **kwargs: published.append(image))

    with pytest.raises(VideoStorageError, match="disk full"):
        source._stream_pyav()

    assert published == []


def test_storage_fatal_does_not_enter_network_reconnect_loop(monkeypatch) -> None:
    source = Hm30VideoSource()

    def fail_storage() -> None:
        raise VideoStorageError("TS index failed")

    monkeypatch.setattr(source, "_stream_pyav_process", fail_storage)
    source._run()

    assert source.stats.reconnects == 0
    assert source.stats.state == "stopped"
    assert source.stats.last_error == "TS index failed"


def test_ts_range_decoder_caches_resolved_directory(tmp_path: Path, monkeypatch) -> None:
    """热路径构造缓存键复用规范化目录缓存，且更换目录后同步更新。"""
    from fractions import Fraction

    from airdrop.video.cold import TsRangeDecoder

    first_dir = tmp_path / "run-a"
    second_dir = tmp_path / "run-b"
    first_dir.mkdir()
    second_dir.mkdir()
    decoder = TsRangeDecoder(first_dir)
    try:
        expected_first = str(first_dir.resolve())
        assert decoder.resolved_directory == expected_first
        original_resolve = Path.resolve
        calls = 0

        def counted_resolve(self: Path, *args, **kwargs) -> Path:
            nonlocal calls
            calls += 1
            return original_resolve(self, *args, **kwargs)

        monkeypatch.setattr(Path, "resolve", counted_resolve)
        key = decoder._make_cache_key(1, Fraction(1, 30), 1280, 720)
        assert key[1] == expected_first
        assert calls == 0

        decoder.directory = second_dir
        assert calls == 1
        assert decoder.resolved_directory == str(second_dir.resolve())
        key_after = decoder._make_cache_key(1, Fraction(1, 30), 1280, 720)
        assert key_after[1] == decoder.resolved_directory
        assert calls == 2
    finally:
        decoder.close()
