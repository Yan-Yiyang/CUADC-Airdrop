"""历史解码图像的线程安全共享缓存。"""

from __future__ import annotations

import threading
import uuid
from collections import OrderedDict

import numpy as np


class DecodedFrameCache:
    """多个解码器共用单一字节额度的 LRU 图像缓存。

    额度仅统计已驻留图像，不包括解码器在途帧、插入时临时副本或返回给调用方的副本。
    """

    def __init__(self, max_bytes: int) -> None:
        if max_bytes < 1:
            raise ValueError("历史解码缓存上限必须为正")
        self.max_bytes = max_bytes
        self.storage_identity = uuid.uuid4().hex
        self._entries: OrderedDict[tuple[object, ...], np.ndarray] = OrderedDict()
        self._bytes = 0
        self._leases = 0
        self._lock = threading.RLock()

    def acquire(self) -> None:
        """取得一个解码器租约。"""
        with self._lock:
            self._leases += 1

    def release(self) -> None:
        """释放租约；最后一个使用者退出时清空缓存。"""
        with self._lock:
            if self._leases <= 0:
                raise RuntimeError("解码缓存租约计数失配")
            self._leases -= 1
            if self._leases == 0:
                self._entries.clear()
                self._bytes = 0

    def get(self, key: tuple[object, ...]) -> np.ndarray | None:
        """返回隔离于缓存内容的副本。"""
        with self._lock:
            if self._leases == 0:
                return None
            image = self._entries.get(key)
            if image is None:
                return None
            self._entries.move_to_end(key)
            return image.copy()

    def get_readonly(self, key: tuple[object, ...]) -> np.ndarray | None:
        """返回缓存图像的零拷贝只读视图，并更新 LRU 顺序。"""
        with self._lock:
            if self._leases == 0:
                return None
            image = self._entries.get(key)
            if image is None:
                return None
            self._entries.move_to_end(key)
            view = image.view()
            view.flags.writeable = False
            return view

    def put(self, key: tuple[object, ...], image: np.ndarray) -> None:
        """在复制图像前预留额度；超大图像不进入缓存。"""
        size = image.nbytes
        with self._lock:
            if self._leases == 0:
                return
            if size > self.max_bytes:
                return
            previous = self._entries.pop(key, None)
            if previous is not None:
                self._bytes -= previous.nbytes
            while self._entries and self._bytes + size > self.max_bytes:
                _, removed = self._entries.popitem(last=False)
                self._bytes -= removed.nbytes
            owned = image.copy()
            self._entries[key] = owned
            self._bytes += owned.nbytes

    @property
    def byte_size(self) -> int:
        with self._lock:
            return self._bytes
