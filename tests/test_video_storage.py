from __future__ import annotations

import threading
from fractions import Fraction

import av
import numpy as np
import pytest

from airdrop.video import storage
from airdrop.video.buffer import AlignmentBuffer
from airdrop.video.cold import TsRangeDecoder
from airdrop.video.storage import TemporaryTsStorage


def test_finalize_waits_for_producer_and_all_leases(tmp_path) -> None:
    owner = TemporaryTsStorage(parent=tmp_path)
    owner.directory.joinpath("segment.ts").write_bytes(b"data")
    first = owner.acquire_lease()
    second = first.fork()

    assert owner.finalize() is False
    owner.producer_stopped()
    assert owner.finalize() is False
    first.close()
    first.close()
    assert owner.directory.exists()
    assert owner.lease_count == 1

    second.close()
    assert not owner.directory.exists()
    assert owner.finalize() is True
    assert owner.lease_count == 0


def test_decoder_and_gop_job_hold_independent_storage_leases(tmp_path) -> None:
    owner = TemporaryTsStorage(parent=tmp_path)
    buffer_lease = owner.acquire_lease()
    decoder = TsRangeDecoder(owner.directory)
    decoder.bind_storage_lease(buffer_lease)
    job_lease = decoder.fork_storage_lease()
    assert job_lease is not None
    assert owner.lease_count == 3
    owner.producer_stopped()
    assert owner.finalize() is False
    buffer_lease.close()
    decoder.close()
    assert owner.directory.exists()
    assert owner.lease_count == 1
    with pytest.raises(RuntimeError, match="租约已释放"):
        decoder.read(1, 0, 1, 1)
    job_lease.close()
    assert not owner.directory.exists()


def test_decoder_rejects_storage_lease_from_other_directory(tmp_path) -> None:
    owner = TemporaryTsStorage(parent=tmp_path)
    lease = owner.acquire_lease()
    decoder = TsRangeDecoder(tmp_path)
    try:
        with pytest.raises(ValueError, match="目录与 TS 存储租约不一致"):
            decoder.bind_storage_lease(lease)
        assert owner.lease_count == 1
    finally:
        decoder.close()
        lease.close()
        owner.producer_stopped()
        owner.finalize()


def test_decoder_close_error_releases_storage_after_cursor_is_gone(tmp_path, monkeypatch) -> None:
    owner = TemporaryTsStorage(parent=tmp_path)
    lease = owner.acquire_lease()
    decoder = TsRangeDecoder(owner.directory)
    decoder.bind_storage_lease(lease)
    owner.producer_stopped()
    assert owner.finalize() is False
    lease.close()

    def fail_reset() -> None:
        raise RuntimeError("injected cursor close failure")

    monkeypatch.setattr(decoder, "_reset_cursor_locked", fail_reset)
    with pytest.raises(RuntimeError, match="injected cursor close failure"):
        decoder.close()
    assert owner.lease_count == 0
    assert not owner.directory.exists()
    with pytest.raises(RuntimeError, match="关闭未完成"):
        decoder.read(1, 0, 1, 1)


def test_decoder_close_keeps_storage_while_generator_is_still_alive(tmp_path) -> None:
    owner = TemporaryTsStorage(parent=tmp_path)
    lease = owner.acquire_lease()
    decoder = TsRangeDecoder(owner.directory)
    decoder.bind_storage_lease(lease)

    def stubborn_cursor():
        try:
            yield (0, 1, 1), np.zeros((1, 1, 3), dtype=np.uint8)
        except GeneratorExit:
            yield (0, 1, 1), np.zeros((1, 1, 3), dtype=np.uint8)

    cursor = stubborn_cursor()
    next(cursor)
    decoder._cursor = cursor
    owner.producer_stopped()
    assert owner.finalize() is False
    lease.close()
    with pytest.raises(RuntimeError, match="ignored GeneratorExit"):
        decoder.close()
    assert owner.directory.exists()
    assert owner.lease_count == 1
    with pytest.raises(RuntimeError, match="关闭未完成"):
        decoder.read(1, 0, 1, 1)
    decoder.close()
    assert not owner.directory.exists()


def test_decoder_does_not_reuse_closed_lease_after_directory_cleanup_failure(
    tmp_path, monkeypatch
) -> None:
    owner = TemporaryTsStorage(parent=tmp_path)
    lease = owner.acquire_lease()
    decoder = TsRangeDecoder(owner.directory)
    decoder.bind_storage_lease(lease)
    owner.producer_stopped()
    assert owner.finalize() is False
    lease.close()
    real_rmtree = storage.shutil.rmtree

    def fail_cleanup(_path) -> None:
        raise OSError("injected cleanup error")

    monkeypatch.setattr(storage.shutil, "rmtree", fail_cleanup)
    with pytest.raises(OSError, match="injected cleanup error"):
        decoder.close()
    assert decoder._storage_lease is None
    assert owner.lease_count == 0
    assert owner.directory.exists()
    with pytest.raises(RuntimeError, match="关闭未完成"):
        decoder.read(1, 0, 1, 1)
    monkeypatch.setattr(storage.shutil, "rmtree", real_rmtree)
    assert owner.finalize() is True
    decoder.close()


def test_buffer_does_not_leak_lease_when_decoder_binding_fails(tmp_path, monkeypatch) -> None:
    owner = TemporaryTsStorage(parent=tmp_path)
    lease = owner.acquire_lease()
    decoder = TsRangeDecoder(owner.directory)
    buffer = AlignmentBuffer(storage="raw", cold_decoder=decoder)

    def fail_binding(_lease) -> None:
        raise RuntimeError("injected decoder lease failure")

    monkeypatch.setattr(decoder, "bind_storage_lease", fail_binding)
    try:
        with pytest.raises(RuntimeError, match="injected decoder lease failure"):
            buffer.bind_cold_directory(owner.directory, storage_lease=lease)
        assert owner.lease_count == 1
        assert buffer._storage_lease is None
        assert decoder.fork_storage_lease() is None
    finally:
        buffer.close_cold_storage()
        lease.close()
        owner.producer_stopped()
        owner.finalize()


def test_failed_native_container_close_retains_storage_until_same_container_closes(
    tmp_path, monkeypatch
) -> None:
    owner = TemporaryTsStorage(parent=tmp_path)
    path = owner.directory / "segment-000001.ts"
    path.write_bytes(b"synthetic")
    path.with_suffix(".complete").touch()
    lease = owner.acquire_lease()
    decoder = TsRangeDecoder(owner.directory)
    decoder.bind_storage_lease(lease)

    class Frame:
        pts = 0
        time_base = Fraction(1, 1)
        is_corrupt = False

        def to_ndarray(self, *, format: str) -> np.ndarray:
            assert format == "bgr24"
            return np.zeros((1, 1, 3), dtype=np.uint8)

    class Packet:
        size = 1
        is_corrupt = False
        pts = 0

        def decode(self):
            return [Frame()]

    class Container:
        streams = type(
            "Streams", (), {"video": [type("Stream", (), {"time_base": Fraction(1)})()]}
        )()
        close_calls = 0

        def seek(self, *args, **kwargs) -> None:
            pass

        def demux(self, _stream):
            yield Packet()

        def close(self) -> None:
            self.close_calls += 1
            if self.close_calls == 1:
                raise OSError("injected native close failure")

    container = Container()
    monkeypatch.setattr(av, "open", lambda *args, **kwargs: container)
    monkeypatch.setattr(
        decoder, "_get_keyframes", lambda *_args: [(Fraction(0), Fraction(0), None)]
    )
    assert decoder.read(1, 0, 1, 1) is not None
    owner.producer_stopped()
    assert owner.finalize() is False
    lease.close()
    with pytest.raises(OSError, match="injected native close failure"):
        decoder.close()
    assert container.close_calls == 1
    assert owner.directory.exists() and owner.lease_count == 1
    with pytest.raises(RuntimeError, match="不能派生作业租约"):
        decoder.fork_storage_lease()
    decoder.close()
    assert container.close_calls == 2
    assert not owner.directory.exists()
    with pytest.raises(RuntimeError, match="不能派生作业租约"):
        decoder.fork_storage_lease()


def test_finalize_request_rejects_new_owner_lease_but_valid_lease_can_fork(tmp_path) -> None:
    owner = TemporaryTsStorage(parent=tmp_path)
    lease = owner.acquire_lease()
    assert owner.finalize() is False
    with pytest.raises(RuntimeError, match="不能增加租约"):
        owner.acquire_lease()

    frozen = lease.fork()
    lease.close()
    assert owner.directory.exists()
    with pytest.raises(RuntimeError, match="已关闭"):
        lease.fork()
    owner.producer_stopped()
    assert owner.directory.exists()
    frozen.close()
    assert not owner.directory.exists()


def test_cleanup_failure_preserves_directory_and_can_be_retried(tmp_path, monkeypatch) -> None:
    owner = TemporaryTsStorage(parent=tmp_path)
    owner.directory.joinpath("segment.ts").write_bytes(b"data")
    owner.producer_stopped()
    original = storage.shutil.rmtree
    calls = 0

    def fail_once(path) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise PermissionError("busy")
        original(path)

    monkeypatch.setattr(storage.shutil, "rmtree", fail_once)
    with pytest.raises(OSError, match="目录已保留"):
        owner.finalize()
    assert owner.directory.is_dir()
    assert isinstance(owner.cleanup_error, PermissionError)
    assert owner.finalize() is True
    assert not owner.directory.exists()
    assert owner.cleanup_error is None


def test_concurrent_idempotent_lease_close_releases_once(tmp_path) -> None:
    owner = TemporaryTsStorage(parent=tmp_path)
    owner.producer_stopped()
    lease = owner.acquire_lease()
    assert owner.finalize() is False
    threads = [threading.Thread(target=lease.close) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2)
        assert not thread.is_alive()
    assert owner.lease_count == 0
    assert not owner.directory.exists()


def test_directory_prefix_cannot_escape_parent(tmp_path) -> None:
    with pytest.raises(ValueError, match="单一文件名片段"):
        TemporaryTsStorage(parent=tmp_path, prefix="../outside-")
