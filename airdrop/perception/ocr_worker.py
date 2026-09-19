'OCR 独立工作进程：消费裁剪图请求，回传编号。\n\n进程间只传**图像 ndarray**（用 ``multiprocessing`` 的 pickle 走共享内存管道），\n不传 ffmpeg 的复用缓冲视图——那个视图出了原进程就是脏数据。\n\n队列有界，满了**丢最旧**：OCR 落后时宁可少认几个编号，也不能让请求队列无限\n膨胀把内存吃光（编号是"多帧投票"的输入，丢几帧不影响最终统计）。\n'

from __future__ import annotations

import logging
import multiprocessing as mp
import queue
import time
from dataclasses import dataclass, field
from multiprocessing.context import SpawnProcess
from typing import Any

import numpy as np

from .cropproc import CropResult, OcrEngineConfig, OpenCvPostProcess

LOGGER = logging.getLogger(__name__)

__all__ = [
    "OcrRequest",
    "OcrResult",
    "OcrWorkerPool",
    "ocr_worker_main",
    "put_drop_oldest",
]


@dataclass(slots=True)
class OcrRequest:
    """一条 OCR 送检请求（裁剪图 + 溯源信息）。"""

    request_id: int
    frame_index: int
    capture_timestamp: float
    crop: np.ndarray
    # 仅用于回传时对位，worker 不解释
    payload: Any = None


@dataclass(slots=True)
class OcrResult:
    """一条 OCR 结果；``number`` 为 None 表示没读出来。"""

    request_id: int
    frame_index: int
    capture_timestamp: float
    number: int | None
    raw_text: str = ""
    confidence: float = 0.0
    side_px: float = -1.0
    saturation_level: int = 0
    stage: str = ""
    payload: Any = None
    worker_id: int = 0
    elapsed_ms: float = 0.0
    error: str | None = None
    rectified: np.ndarray | None = field(default=None, repr=False)


def put_drop_oldest(bounded_queue: Any, item: Any) -> bool:
    """往有界队列放一项；满了丢**最旧**的一项再放。

    返回是否发生了丢弃。用 ``put_nowait`` + 手动驱逐而不是 ``put(timeout=0)``：
    调用方（拉流/推理线程）**一刻都不能阻塞**在队列上。
    """
    dropped = False
    while True:
        try:
            bounded_queue.put_nowait(item)
            return dropped
        except queue.Full:
            try:
                bounded_queue.get_nowait()
                dropped = True
            except queue.Empty:  # pragma: no cover - 竞态兜底
                return dropped


def ocr_worker_main(
    request_q: Any,
    result_q: Any,
    config_dict: dict[str, Any],
    worker_id: int = 0,
) -> None:  # pragma: no cover - 子进程入口，由集成测试覆盖
    """OCR 工作进程入口（``multiprocessing`` target，必须是模块级函数）。

    进程内自建 :class:`OpenCvPostProcess`（连同 RapidOCR 引擎）——句柄不能跨
    进程继承，每个 worker 各加载一份权重（约 2s、显存几百 MB）。
    """
    logging.basicConfig(
        level=logging.INFO,
        format=f"%(asctime)s %(levelname)-7s ocr{worker_id} %(message)s",
    )
    log = logging.getLogger(f"ocr{worker_id}")
    engine_config = OcrEngineConfig(**config_dict.get("engine", {}))
    post = OpenCvPostProcess(
        color=config_dict.get("color", "blue"),
        ocr_conf_threshold=config_dict.get("ocr_conf_threshold", 0.6),
        engine_config=engine_config,
    )
    log.info("OCR 工作进程 %d 启动（color=%s）", worker_id, post.color)
    post.warmup()

    while True:
        try:
            request = request_q.get(timeout=0.5)
        except queue.Empty:
            continue
        if request is None:
            log.info("OCR 工作进程 %d 收到退出指令", worker_id)
            break
        started = time.perf_counter()
        try:
            crop_result: CropResult = post.recognize(request.crop)
            result = OcrResult(
                request_id=request.request_id,
                frame_index=request.frame_index,
                capture_timestamp=request.capture_timestamp,
                number=crop_result.number,
                raw_text=crop_result.raw_text,
                confidence=crop_result.confidence,
                side_px=crop_result.side_px,
                saturation_level=crop_result.saturation_level,
                stage=crop_result.stage,
                payload=request.payload,
                worker_id=worker_id,
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
                rectified=crop_result.rectified,
            )
        except Exception as exc:
            log.exception("OCR 处理异常 帧#%d", request.frame_index)
            result = OcrResult(
                request_id=request.request_id,
                frame_index=request.frame_index,
                capture_timestamp=request.capture_timestamp,
                number=None,
                payload=request.payload,
                worker_id=worker_id,
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
                error=f"{type(exc).__name__}: {exc}",
            )
        put_drop_oldest(result_q, result)


class OcrWorkerPool:
    """管理 N 个 OCR 工作进程的请求/结果队列（主进程侧）。"""

    def __init__(
        self,
        workers: int = 2,
        *,
        queue_size: int = 500,
        result_queue_size: int = 500,
        config: dict[str, Any] | None = None,
    ) -> None:
        if workers < 1:
            raise ValueError(f"OCR 工作进程数至少为 1: {workers}")
        if queue_size < 1:
            raise ValueError(f"ocr_queue_size 至少为 1: {queue_size}")
        self._workers = int(workers)
        self._config = dict(config or {})
        self._context = mp.get_context("spawn")  # Windows 默认；显式声明意图
        self._request_q: Any = self._context.Queue(maxsize=queue_size)
        self._result_q: Any = self._context.Queue(maxsize=result_queue_size)
        # spawn 上下文的 Process 是 SpawnProcess（`mp.Process` 只是基类名），
        # 按实际类型标注才不会被类型检查判成"装不进去"
        self._processes: list[SpawnProcess] = []
        self._next_id = 0
        self._submitted = 0
        self._dropped = 0
        self._results = 0
        self._started = False

    # ------------------------------------------------------------------
    # 状态
    # ------------------------------------------------------------------
    @property
    def started(self) -> bool:
        return self._started

    @property
    def stats(self) -> dict[str, int]:
        return {
            "workers": self._workers,
            "submitted": self._submitted,
            "dropped": self._dropped,
            "results": self._results,
            "pending": self._submitted - self._results,
        }

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    def start(self) -> "OcrWorkerPool":
        """拉起工作进程（幂等）。"""
        if self._started:
            return self
        for index in range(self._workers):
            process = self._context.Process(
                target=ocr_worker_main,
                args=(self._request_q, self._result_q, self._config, index),
                name=f"ocr-worker-{index}",
                daemon=True,
            )
            process.start()
            self._processes.append(process)
        self._started = True
        LOGGER.info("OCR 工作进程池已启动：%d 个", self._workers)
        return self

    def stop(self, timeout: float = 5.0) -> None:
        """发退出指令并等进程结束（幂等）。"""
        if not self._started:
            return
        for _ in self._processes:
            try:
                self._request_q.put_nowait(None)
            except queue.Full:
                try:  # 队列满：挤掉一个再放
                    self._request_q.get_nowait()
                    self._request_q.put_nowait(None)
                except queue.Empty, queue.Full:  # pragma: no cover
                    pass
        for process in self._processes:
            process.join(timeout)
            if process.is_alive():
                LOGGER.warning("OCR 工作进程 %s 未在 %.1fs 内退出，强制终止", process.name, timeout)
                process.terminate()
                process.join(1.0)
        self._processes.clear()
        self._started = False

    def __enter__(self) -> "OcrWorkerPool":
        return self.start()

    def __exit__(self, *exc_info: object) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # 请求 / 结果
    # ------------------------------------------------------------------
    def submit(
        self,
        crop: np.ndarray,
        *,
        frame_index: int,
        capture_timestamp: float,
        payload: Any = None,
    ) -> int | None:
        """提交一张裁剪图；队列满时丢最旧。返回 ``request_id``（未启动则 None）。"""
        if not self._started:
            return None
        self._next_id += 1
        request = OcrRequest(
            request_id=self._next_id,
            frame_index=frame_index,
            capture_timestamp=capture_timestamp,
            crop=crop,
            payload=payload,
        )
        if put_drop_oldest(self._request_q, request):
            self._dropped += 1
        self._submitted += 1
        return request.request_id

    def poll(self, timeout: float = 0.0) -> OcrResult | None:
        """取一条结果；没有则返回 None（不阻塞，除非给了 timeout）。"""
        try:
            result = self._result_q.get(timeout=timeout) if timeout else self._result_q.get_nowait()
        except queue.Empty:
            return None
        self._results += 1
        return result

    def drain(self) -> list[OcrResult]:
        """把当前可用的结果全部取走。"""
        out: list[OcrResult] = []
        while True:
            result = self.poll()
            if result is None:
                return out
            out.append(result)
