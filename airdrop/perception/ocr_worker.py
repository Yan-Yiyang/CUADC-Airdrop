"""OCR 独立工作进程：消费裁剪图请求，回传编号。

**为什么单独开进程**（旧版实测约束，P5 复测见下）
----------------------------------------------------
旧版的结论是"ffmpeg 解码与 GPU 推理同进程并发会破坏检测结果"，所以把 OCR
挪进独立进程。本架构的解码在 ffmpeg 子进程里，天然规避了那一层冲突；但
**"YOLO 与 RapidOCR 同进程并发是否异常"仍未实测**，据此判断能否降级成线程。
**在复测得出结论之前，进程隔离是保守默认**。

进程间只传**图像 ndarray**（用 ``multiprocessing`` 的 pickle 走共享内存管道），
不传 ffmpeg 的复用缓冲视图——那个视图出了原进程就是脏数据。

队列**无界、不丢弃**
-------------------
请求与结果队列都不设上限：目标可能只在一瞬间清晰可见，**漏掉任何一个送检
请求都可能漏掉目标编号**（编号是坐标解算的输入，丢帧不是"少几个投票"那么轻）。
代价是 OCR 跟不上检测时积压会吃内存，所以积压超过 ``queue_size`` 时打一条
WARNING（跨越阈值只打一次，回落后可再次触发），把"该降检测帧率还是加 OCR
worker"的判断交给操作手；``stats()`` 里的 ``pending`` 也一直可见。
"""

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
        # 结果队列无界：OCR 结果同样不丢弃（没回来的编号晚几拍也会到）
        result_q.put_nowait(result)


class OcrWorkerPool:
    """管理 N 个 OCR 工作进程的请求/结果队列（主进程侧）。

    队列**无界、不丢弃**（见模块 docstring）。``queue_size`` 不再是容量，
    而是**积压告警阈值**：未完成请求数（``submitted - results``）达到它时打
    一条 WARNING，回落到一半以下后允许再次告警。请求慢一点没关系，漏一个
    请求就可能漏掉一个目标编号。
    """

    def __init__(
        self,
        workers: int = 2,
        *,
        queue_size: int = 500,
        config: dict[str, Any] | None = None,
    ) -> None:
        if workers < 1:
            raise ValueError(f"OCR 工作进程数至少为 1: {workers}")
        if queue_size < 1:
            raise ValueError(f"ocr_queue_size 至少为 1: {queue_size}")
        self._workers = int(workers)
        self._config = dict(config or {})
        self._queue_size = int(queue_size)
        self._context = mp.get_context("spawn")  # Windows 默认；显式声明意图
        # maxsize=0 → 无界：put_nowait 永远不会 Full，请求/结果都不丢
        self._request_q: Any = self._context.Queue()
        self._result_q: Any = self._context.Queue()
        # spawn 上下文的 Process 是 SpawnProcess（`mp.Process` 只是基类名），
        # 按实际类型标注才不会被类型检查判成"装不进去"
        self._processes: list[SpawnProcess] = []
        self._next_id = 0
        self._submitted = 0
        self._dropped = 0  # 保留计数：不丢弃策略下恒为 0（一旦非 0 说明有新的丢弃路径）
        self._results = 0
        self._backlog_warned = False
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
        """发退出指令并等进程结束（幂等）。

        退出指令排在已有请求**之后**：worker 会先把积压处理完再退出；
        等不完的（``timeout`` 内没退）会被强制终止，这也是"关闭时不再保证
        处理完积压"的唯一场合。
        """
        if not self._started:
            return
        for _ in self._processes:
            self._request_q.put_nowait(None)
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
        """提交一张裁剪图；请求队列**无界、不丢弃**。返回 ``request_id``（未启动则 None）。"""
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
        self._request_q.put_nowait(request)
        self._submitted += 1
        self._warn_if_backlogged()
        return request.request_id

    def _warn_if_backlogged(self) -> None:
        """积压超过阈值时告警（跨阈值一次，回落后可再告警）。

        跑在提交侧（感知线程），所以只做一次整数比较，不碰 RLock 之外的任何
        资源；真正重的是 OCR，不是这条日志。
        """
        pending = self._submitted - self._results
        if not self._backlog_warned and pending >= self._queue_size:
            self._backlog_warned = True
            LOGGER.warning(
                "OCR 积压 %d 个请求（阈值 %.0f）：**请求不会丢**，但积压期间内存会随裁剪图增长；"
                "若持续增长请降检测帧率或调大 perception.ocr_workers",
                pending,
                self._queue_size,
            )

    def poll(self, timeout: float = 0.0) -> OcrResult | None:
        """取一条结果；没有则返回 None（不阻塞，除非给了 timeout）。"""
        try:
            result = self._result_q.get(timeout=timeout) if timeout else self._result_q.get_nowait()
        except queue.Empty:
            return None
        self._results += 1
        # 积压回落到阈值一半以下时复位告警状态，下次再涨上来会重新提醒
        if self._backlog_warned and (self._submitted - self._results) < max(
            self._queue_size // 2, 1
        ):
            self._backlog_warned = False
        return result

    def drain(self) -> list[OcrResult]:
        """把当前可用的结果全部取走。"""
        out: list[OcrResult] = []
        while True:
            result = self.poll()
            if result is None:
                return out
            out.append(result)
