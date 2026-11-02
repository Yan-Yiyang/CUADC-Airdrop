"""飞行期间非持久 TS 存储目录及其读取租约。"""

from __future__ import annotations

import shutil
import tempfile
import threading
from pathlib import Path


class TsStorageLease:
    """持有临时 TS 目录的独立、幂等租约。"""

    def __init__(self, owner: TemporaryTsStorage) -> None:
        self._owner = owner
        self._lock = threading.Lock()
        self._closed = False

    @property
    def directory(self) -> Path:
        """租约所保护的 TS 目录。"""
        return self._owner.directory

    def close(self) -> None:
        """释放租约；重复调用不会重复扣减引用数。"""
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._owner._release_lease()  # noqa: SLF001 - 租约只通过 owner 更新引用数

    def fork(self) -> TsStorageLease:
        """从仍有效的租约派生冻结视图租约。"""
        with self._lock:
            if self._closed:
                raise RuntimeError("已关闭的 TS 存储租约不能派生新租约")
            return self._owner._fork_lease()  # noqa: SLF001 - 派生必须验证本租约有效

    def __enter__(self) -> TsStorageLease:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        del exc_type, exc, traceback
        self.close()


class TemporaryTsStorage:
    """管理一个临时 TS 目录，等待生产者停止且所有读取租约释放后删除。"""

    def __init__(self, *, parent: str | Path | None = None, prefix: str = "airdrop-ts-") -> None:
        if not prefix or Path(prefix).name != prefix or "/" in prefix or "\\" in prefix:
            raise ValueError("临时 TS 目录前缀必须是单一文件名片段")
        if parent is not None and not Path(parent).is_dir():
            raise FileNotFoundError(f"临时 TS 存储父目录不存在: {parent}")
        self.directory = Path(tempfile.mkdtemp(prefix=prefix, dir=parent)).resolve()
        self._lock = threading.RLock()
        self._lease_count = 0
        self._producer_stopped = False
        self._finalize_requested = False
        self._deleted = False
        self._cleanup_error: OSError | None = None

    @property
    def lease_count(self) -> int:
        with self._lock:
            return self._lease_count

    @property
    def producer_has_stopped(self) -> bool:
        with self._lock:
            return self._producer_stopped

    @property
    def finalized(self) -> bool:
        with self._lock:
            return self._deleted

    @property
    def cleanup_error(self) -> OSError | None:
        with self._lock:
            return self._cleanup_error

    def acquire_lease(self) -> TsStorageLease:
        """增加目录租约；删除请求后的目录不再接受新读者。"""
        with self._lock:
            if self._finalize_requested or self._deleted:
                raise RuntimeError("临时 TS 存储已请求结束，不能增加租约")
            self._lease_count += 1
            return TsStorageLease(self)

    def producer_stopped(self) -> None:
        """确认采集生产者及其文件句柄均已关闭。"""
        with self._lock:
            self._producer_stopped = True
            self._try_delete_locked()

    def finalize(self) -> bool:
        """请求最终删除；门槛未满足时返回 False，满足并删除后返回 True。

        文件系统删除失败时保留目录并抛出异常，后续 finalize() 可重试。
        """
        with self._lock:
            if self._deleted:
                return True
            self._finalize_requested = True
            return self._try_delete_locked()

    def _release_lease(self) -> None:
        with self._lock:
            if self._lease_count <= 0:
                raise RuntimeError("临时 TS 存储租约计数下溢")
            self._lease_count -= 1
            self._try_delete_locked()

    def _fork_lease(self) -> TsStorageLease:
        with self._lock:
            if self._deleted or self._lease_count <= 0:
                raise RuntimeError("TS 存储目录已不可用，不能派生租约")
            self._lease_count += 1
            return TsStorageLease(self)

    def _try_delete_locked(self) -> bool:
        if self._deleted:
            return True
        if not self._finalize_requested or not self._producer_stopped or self._lease_count:
            return False
        try:
            shutil.rmtree(self.directory)
        except OSError as exc:
            self._cleanup_error = exc
            raise OSError(f"临时 TS 存储清理失败，目录已保留: {self.directory}: {exc}") from exc
        self._cleanup_error = None
        self._deleted = True
        return True
