"""OCR 独立工作进程：消费裁剪图请求，回传编号。

**独立工作进程**
----------------
OCR 使用独立进程运行。请求和结果通过 ``multiprocessing`` 队列序列化传输图像与元数据；
不传 ffmpeg 的复用缓冲视图，因为下一帧到达时该视图会被覆盖。

磁盘积压、有界请求队列
----------------------
提交时将 PNG 无损图像和请求元数据原子写入磁盘；进程队列只传文件引用，队列容量
按 worker 数限制。结果队列也有界并通过阻塞回压保证结果不丢。成功关联结果后才删除
请求文件；异常中断留下的文件保留供诊断或重启恢复。
"""

from __future__ import annotations

import logging
import math
import multiprocessing as mp
import os
import pickle
import queue
import shutil
import sqlite3
import tempfile
import threading
import time
from dataclasses import dataclass
from multiprocessing.context import SpawnProcess
from pathlib import Path
from typing import Any

import cv2
import numpy as np

import airdrop.perception.timing as timing  # noqa: PLR0402 - 避免惰性包循环导入

from .cropproc import CropResult, OcrEngineConfig, OpenCvPostProcess

LOGGER = logging.getLogger(__name__)


def _write_jpeg_atomic(image: np.ndarray, path: Path) -> None:
    """将 OCR 转正结果原子替换到检测裁切图路径。"""
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise OSError(f"OCR 转正图 JPG 编码失败: {path}")
    temporary = path.with_name(f"{path.stem}.rectified.tmp.jpg")
    temporary.write_bytes(encoded.tobytes())
    temporary.replace(path)


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
    crop_path: str
    rectified_path: str | None = None


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
    archive_error: str | None = None
    rectified_written: bool = False
    # 这里不回传 rectified 图像：结果消费端不使用它，避免额外序列化和队列内存占用。
    # 排故要转正图请在**进程内**直接调 OpenCvPostProcess.recognize
    # （tests/test_perception_realdata.py 就是这么做的）。


class _OcrSpoolIndexError(OSError):
    """索引持久化故障，须沿感知主循环的 I/O 致命路径传播。"""


def ocr_worker_main(  # noqa: PLR0912, PLR0915 - 请求处理及诊断保持顺序
    request_q: Any,
    result_q: Any,
    config_dict: dict[str, Any],
    worker_id: int = 0,
    status_q: Any = None,
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
    try:
        engine_config = OcrEngineConfig(**config_dict.get("engine", {}))
        post = OpenCvPostProcess(
            color=config_dict.get("color", "blue"),
            ocr_conf_threshold=config_dict.get("ocr_conf_threshold", 0.6),
            engine_config=engine_config,
        )
        if not post.warmup():
            raise RuntimeError("OCR 模型预热失败，请检查权重、引擎及设备配置")
    except Exception as exc:
        log.exception("OCR 工作进程 %d 初始化失败", worker_id)
        if status_q is not None:
            status_q.put((worker_id, f"{type(exc).__name__}: {exc}"))
        return
    if status_q is not None:
        status_q.put((worker_id, None))
    log.info("OCR 工作进程 %d 初始化完成（color=%s）", worker_id, post.color)
    if timing.ENABLED:
        try:
            engine = post.engine.ensure()
            timing.instrument_call_class(engine.text_det, "ocr_text_det")
            timing.instrument_call_class(engine.text_rec, "ocr_text_rec")
        except Exception:  # 诊断不影响 OCR 初始化
            log.exception("OCR 分段计时不可用，继续正常识别")

    while True:
        try:
            request = request_q.get(timeout=0.5)
        except queue.Empty:
            continue
        if request is None:
            log.info("OCR 工作进程 %d 收到退出指令", worker_id)
            timing.flush()
            break
        if timing.ENABLED:
            timing.context(
                request_id=request.request_id, frame_index=request.frame_index, worker_id=worker_id
            )
            worker_start = timing.now_ns()
        started = time.perf_counter()
        try:
            encoded = np.fromfile(request.crop_path, dtype=np.uint8)
            crop = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if crop is None:
                raise OSError(f"OCR 裁切图读取失败: {request.crop_path}")
            read_end = timing.now_ns() if timing.ENABLED else 0
            crop_result: CropResult = post.recognize(crop)
            recognize_end = timing.now_ns() if timing.ENABLED else 0
            archive_error = None
            rectified_written = False
            if crop_result.rectified is not None and request.rectified_path:
                try:
                    _write_jpeg_atomic(crop_result.rectified, Path(request.rectified_path))
                    rectified_written = True
                except Exception as exc:
                    archive_error = f"{type(exc).__name__}: {exc}"
                    log.exception("OCR 转正图归档失败 帧#%d", request.frame_index)
            if timing.ENABLED:
                timing.emit(
                    "ocr_worker",
                    start_ns=worker_start,
                    read_end_ns=read_end,
                    recognize_end_ns=recognize_end,
                    end_ns=timing.now_ns(),
                    has_number=crop_result.number is not None,
                    stage=crop_result.stage,
                    archive_written=rectified_written,
                )
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
                archive_error=archive_error,
                rectified_written=rectified_written,
                worker_id=worker_id,
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
            )
        except Exception as exc:
            log.exception("OCR 处理异常 帧#%d", request.frame_index)
            if timing.ENABLED:
                timing.emit(
                    "ocr_worker_error",
                    start_ns=worker_start,
                    end_ns=timing.now_ns(),
                    error=f"{type(exc).__name__}: {exc}",
                )
            result = OcrResult(
                request_id=request.request_id,
                frame_index=request.frame_index,
                capture_timestamp=request.capture_timestamp,
                number=None,
                worker_id=worker_id,
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
                error=f"{type(exc).__name__}: {exc}",
            )
        # 结果队列有界；阻塞回压由父进程 poll 解除，不丢弃 OCR 结果。
        while True:
            try:
                result_q.put(result, timeout=0.5)
                break
            except queue.Full:
                continue


class OcrWorkerPool:
    """管理 N 个 OCR 工作进程的请求/结果队列（主进程侧）。

    请求图片与元数据落在磁盘，mp 请求队列只保存有界文件引用。``queue_size`` 是
    **积压告警阈值**：未完成请求数（``submitted - results``）达到它时打
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
        self._request_q: Any = self._context.Queue(maxsize=workers)
        self._result_q: Any = self._context.Queue(maxsize=workers * 2)
        self._status_q: Any = self._context.Queue(maxsize=workers)
        self._used = False
        # spawn 上下文的 Process 是 SpawnProcess（`mp.Process` 只是基类名），
        # 按实际类型标注才不会被类型检查判成"装不进去"
        self._processes: list[SpawnProcess] = []
        self._next_id = 0
        self._submitted = 0
        self._dropped = 0  # 保留计数：不丢弃策略下恒为 0（一旦非 0 说明有新的丢弃路径）
        self._results = 0
        self._backlog_warned = False
        #: 上次报过的存活数（None = 上次是全活）；按数值变化报，见 check_alive
        self._dead_reported: int | None = None
        self._started = False
        spool_root = Path.cwd() / ".local" / "ocr_spool"
        spool_root.mkdir(parents=True, exist_ok=True)
        self._spool_dir = Path(tempfile.mkdtemp(prefix="session-", dir=spool_root))
        self._owns_spool_dir = True
        self._dispatch_thread: threading.Thread | None = None
        self._dispatch_stop = threading.Event()
        self._dispatch_wakeup = threading.Event()
        self._dispatch_id = 1
        self._dispatch_error: str | None = None
        self._dispatch_error_reported = False
        self._cleanup_q: queue.Queue[tuple[int, Path, Path] | None] = queue.Queue(
            maxsize=max(self._queue_size, 1024)
        )
        self._cleanup_thread: threading.Thread | None = None
        self._submit_lock = threading.Lock()
        self._spool_index_lock = threading.Lock()
        self._spool_index: sqlite3.Connection | None = None
        self._spool_index_path: Path | None = None

    # ------------------------------------------------------------------
    # 状态
    # ------------------------------------------------------------------
    @property
    def started(self) -> bool:
        return self._started

    @property
    def alive(self) -> int:
        """当前**活着**的 worker 进程数（运行期可随时取）。

        ⚠ 这个数必须能被外部看见：worker 死亡的路径是真实存在的（CUDA OOM、
        RapidOCR/torch 的原生崩溃、``result_q.put_nowait`` 自身抛错——那行不在
        try 内）。一旦某个 worker 死了，它名下的请求永远不返回 → ``pending``
        单调上升；而积压告警 ``_backlog_warned`` 只打**一条**就置位不再打 ⇒
        全死时表现为"永远零产出"，而操作手从日志里看不出已经死了。
        """
        return sum(1 for process in self._processes if process.is_alive())

    @property
    def stats(self) -> dict[str, int]:
        return {
            "workers": self._workers,
            "alive": self.alive,
            "submitted": self._submitted,
            "dropped": self._dropped,
            "results": self._results,
            "pending": self._submitted - self._results,
            "dispatch_failures": int(self._dispatch_error is not None),
        }

    def check_alive(self) -> int:
        """巡检存活数；掉worker / 存活数变化时打 ERROR（同一数值不重复打，避免刷屏）。

        返回存活数。跑在感知线程里，只是 ``is_alive()`` 的整数比较，很便宜。
        按"数值变化"而不是"首次"来报，是为了不漏掉**恶化**那一档：从 1/2 掉到
        0/2（全死）是新信息，必须再喊一次；而同一个数值每拍重复喊就是刷屏了。
        """
        alive = self.alive
        if self._dispatch_error is not None and not self._dispatch_error_reported:
            self._dispatch_error_reported = True
            LOGGER.error("OCR 磁盘积压派发异常：%s", self._dispatch_error)
        if self._started and alive < self._workers:
            if alive != self._dead_reported:
                self._dead_reported = alive
                LOGGER.error(
                    "OCR 工作进程掉了：%d/%d 存活（pending=%d）。这些 worker 名下的请求"
                    "永远不会有结果，编号会静默缺失；本架次不自动重启进程池，"
                    "落地复盘时看这条日志（存活数再变化时会再报一次）",
                    alive,
                    self._workers,
                    self._submitted - self._results,
                )
        elif alive >= self._workers:
            self._dead_reported = None
        return alive

    def set_spool_directory(self, directory: str | Path) -> None:
        """设置持久积压目录，只能在进程池启动前调用。"""
        if self._started or self._submitted:
            raise RuntimeError("OCR 积压目录必须在提交请求前设置")
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        self._close_spool_index()
        self._spool_dir = path
        self._owns_spool_dir = False
        maximum, pending, first_pending = self._scan_spool_state()
        self._next_id = max(self._next_id, maximum)
        self._submitted = pending
        self._dispatch_id = first_pending or self._next_id + 1

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    def _clear_or_reject_stale_threads(self) -> None:
        """重启前确认上次的派发和清理线程均已退出，并清理已结束的线程引用。"""
        for attr, label in (
            ("_dispatch_thread", "派发"),
            ("_cleanup_thread", "清理"),
        ):
            thread: threading.Thread | None = getattr(self, attr)
            if thread is None:
                continue
            if thread.is_alive() and not self._started:
                raise RuntimeError(f"上次 OCR 磁盘{label}线程尚未退出，不能重新启动")
            if not thread.is_alive():
                setattr(self, attr, None)

    def start(self, timeout: float = 120.0) -> "OcrWorkerPool":
        """并行初始化全部工作进程，收到预热成功确认后返回；失败回滚。

        timeout 是整池初始化的总时限。状态走独立队列，不混入识别结果。
        """
        if not math.isfinite(timeout) or timeout < 0:
            raise ValueError("OCR 初始化超时必须是非负有限秒数")
        self._clear_or_reject_stale_threads()
        if self._started:
            return self
        if self._processes:
            raise RuntimeError("上次 OCR 进程尚未退出，不能重新启动")
        if self._used:
            # 不把上次关闭的哨兵或失败启动的消息交给新进程。
            for channel in (self._request_q, self._result_q, self._status_q):
                channel.close()
            self._request_q = self._context.Queue(maxsize=self._workers)
            self._result_q = self._context.Queue(maxsize=self._workers * 2)
            self._status_q = self._context.Queue(maxsize=self._workers)
            self._cleanup_q = queue.Queue(maxsize=max(self._queue_size, 1024))
            self._submitted = self._results = self._dropped = 0
            self._backlog_warned = False
            maximum, pending_count, first_pending = self._scan_spool_state()
            self._next_id = max(self._next_id, maximum)
            if self._owns_spool_dir and pending_count == 0:
                self._spool_dir = Path(
                    tempfile.mkdtemp(prefix="session-", dir=self._spool_dir.parent)
                )
            self._submitted = pending_count
            self._dispatch_id = first_pending or self._next_id + 1
        self._used = True
        deadline = time.monotonic() + timeout
        try:
            for index in range(self._workers):
                process = self._context.Process(
                    target=ocr_worker_main,
                    args=(self._request_q, self._result_q, self._config, index, self._status_q),
                    name=f"ocr-worker-{index}",
                    daemon=True,
                )
                process.start()
                self._processes.append(process)
            self._wait_ready(deadline)
        except BaseException:
            # 覆盖部分 Process.start 失败、初始化失败、超时及操作手中断。
            for process in self._processes:
                if process.is_alive():
                    process.terminate()
                process.join(1.0)
            self._processes = [p for p in self._processes if p.is_alive()]
            self._close_spool_index()
            raise
        self._started = True
        self._dispatch_error = None
        self._dispatch_error_reported = False
        self._dispatch_stop.clear()
        self._dispatch_wakeup.clear()
        self._dispatch_thread = threading.Thread(
            target=self._dispatch_spool, name="ocr-spool-dispatch", daemon=True
        )
        self._dispatch_thread.start()
        self._cleanup_thread = threading.Thread(
            target=self._cleanup_spool, name="ocr-spool-cleanup", daemon=True
        )
        self._cleanup_thread.start()
        LOGGER.info("OCR 工作进程池已就绪：%d 个", self._workers)
        return self

    def _wait_ready(self, deadline: float) -> None:
        """等待所有不同进程的预热确认，并检测初始化期退出。"""
        ready: set[int] = set()
        while len(ready) < self._workers:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"OCR 初始化超时：{len(ready)}/{self._workers} 就绪")
            try:
                worker_id, error = self._status_q.get(timeout=min(0.1, remaining))
            except queue.Empty:
                if self.alive != self._workers:
                    raise RuntimeError("OCR 初始化期间工作进程退出，未全部就绪") from None
                continue
            if error is not None:
                raise RuntimeError(f"OCR 工作进程 {worker_id} 初始化失败：{error}")
            if worker_id not in range(self._workers):
                raise RuntimeError(f"OCR 初始化收到无效进程编号：{worker_id}")
            ready.add(worker_id)
        if self.alive != self._workers:
            raise RuntimeError("OCR 工作进程在初始化确认后退出")

    def _stop_cleanup_thread(self, deadline: float) -> None:
        """发送退出哨兵并等待异步清理线程在截止时间内排空退出。"""
        cleanup_thread = self._cleanup_thread
        if cleanup_thread is None:
            return
        if cleanup_thread.is_alive():
            while cleanup_thread.is_alive():
                remaining = deadline - time.monotonic()
                try:
                    if remaining <= 0:
                        self._cleanup_q.put_nowait(None)
                    else:
                        self._cleanup_q.put(None, timeout=min(0.1, remaining))
                    break
                except queue.Full:
                    if remaining <= 0:
                        break
                    continue
            cleanup_thread.join(max(0.0, deadline - time.monotonic()))
        if not cleanup_thread.is_alive():
            self._cleanup_thread = None
        else:
            LOGGER.error("OCR 磁盘清理线程未能按时退出；保留线程引用并跳过目录删除")

    def stop(self, timeout: float = 5.0) -> None:
        """发退出指令并等进程结束（幂等）。

        退出指令排在已有请求**之后**：worker 会先把积压处理完再退出；
        等不完的（``timeout`` 内没退）会被强制终止，这也是"关闭时不再保证
        处理完积压"的唯一场合。
        """
        if (
            not self._started
            and not self._processes
            and self._dispatch_thread is None
            and self._cleanup_thread is None
        ):
            self._close_spool_index()
            return
        deadline = time.monotonic() + timeout
        self._dispatch_stop.set()
        self._dispatch_wakeup.set()
        dispatch_thread = self._dispatch_thread
        if dispatch_thread is not None:
            dispatch_thread.join(max(0.0, deadline - time.monotonic()))
            if not dispatch_thread.is_alive():
                self._dispatch_thread = None
            else:
                LOGGER.error("OCR 磁盘派发线程未能按时退出；保留线程引用")
        for _ in self._processes:
            while time.monotonic() < deadline:
                try:
                    self._request_q.put(None, timeout=min(0.1, deadline - time.monotonic()))
                    break
                except queue.Full:
                    continue
        for process in self._processes:
            process.join(max(0.0, deadline - time.monotonic()))
            if process.is_alive():
                LOGGER.warning("OCR 工作进程 %s 未在 %.1fs 内退出，强制终止", process.name, timeout)
                process.terminate()
                process.join(1.0)
        self._processes = [p for p in self._processes if p.is_alive()]
        self._stop_cleanup_thread(deadline)
        self._started = False
        self._dead_reported = None
        if self._submitted > self._results:
            LOGGER.error(
                "OCR 关闭时仍有 %d 条未关联结果；保留磁盘积压目录 %s",
                self._submitted - self._results,
                self._spool_dir,
            )
            self._close_spool_index()
        elif self._dispatch_thread is not None or self._cleanup_thread is not None:
            self._close_spool_index()
        elif self._owns_spool_dir:
            self._close_spool_index()
            if self._spool_dir.exists():
                shutil.rmtree(self._spool_dir, ignore_errors=False)
        else:
            self._close_spool_index()

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
        rectified_path: str | Path | None = None,
    ) -> int | None:
        """先将无损裁切图和请求元数据写盘，再向有界队列提交文件引用。"""
        if not self._started:
            return None
        trace_start = timing.now_ns() if timing.ENABLED else 0
        with self._submit_lock:
            if self._dispatch_error is not None:
                raise OSError(f"OCR 磁盘积压已停止派发：{self._dispatch_error}")
            request_id = self._next_id + 1
            crop_path = self._spool_dir / f"{request_id:012d}.png"
            meta_path = self._spool_dir / f"{request_id:012d}.request"
            temp_meta = meta_path.with_suffix(".request.tmp")
            request_published = False
            try:
                ok, encoded = cv2.imencode(".png", crop)
                if not ok:
                    raise OSError(f"OCR 无损裁切图编码失败: {crop_path}")
                png_end = timing.now_ns() if timing.ENABLED else 0
                temp_path = crop_path.with_suffix(".png.tmp")
                temp_path.write_bytes(encoded.tobytes())
                temp_path.replace(crop_path)
                image_end = timing.now_ns() if timing.ENABLED else 0
                with temp_meta.open("wb") as stream:
                    pickle.dump(
                        (
                            frame_index,
                            capture_timestamp,
                            payload,
                            str(rectified_path) if rectified_path else None,
                        ),
                        stream,
                    )
                    stream.flush()
                    os.fsync(stream.fileno())
                temp_meta.replace(meta_path)
                metadata_end = timing.now_ns() if timing.ENABLED else 0
                request_published = True
                with self._spool_index_lock:
                    index = self._spool_index_connection_locked()
                    with index:
                        index.execute(
                            "INSERT OR IGNORE INTO pending (request_id) VALUES (?)",
                            (request_id,),
                        )
                self._next_id = request_id
                self._submitted += 1
                self._dispatch_wakeup.set()
                if timing.ENABLED:
                    timing.emit(
                        "ocr_submit",
                        request_id=request_id,
                        frame_index=frame_index,
                        start_ns=trace_start,
                        png_end_ns=png_end,
                        image_end_ns=image_end,
                        metadata_end_ns=metadata_end,
                        end_ns=timing.now_ns(),
                    )
            except BaseException as exc:
                if request_published:
                    # sidecar 发布即构成持久提交；索引失败时保留成对文件供恢复扫描。
                    self._next_id = request_id
                    self._submitted += 1
                    self._set_dispatch_error(
                        f"OCR 请求 #{request_id} 已落盘但 SQLite 索引提交失败："
                        f"{type(exc).__name__}: {exc}"
                    )
                    raise OSError(self._dispatch_error) from exc
                temp_meta.unlink(missing_ok=True)
                crop_path.with_suffix(".png.tmp").unlink(missing_ok=True)
                crop_path.unlink(missing_ok=True)
                raise
        self._warn_if_backlogged()
        return request_id

    def _dispatch_spool(self) -> None:
        """顺序读取磁盘请求，只在有限 mp 队列中保留轻量路径引用。"""
        try:
            self._dispatch_spool_loop()
        except (sqlite3.Error, _OcrSpoolIndexError) as exc:
            self._set_dispatch_error(f"OCR 磁盘索引异常：{type(exc).__name__}: {exc}")

    def _dispatch_spool_loop(self) -> None:
        """执行派发；索引错误由外层转换成可巡检的致命状态。"""
        while not self._dispatch_stop.is_set():
            self._dispatch_wakeup.clear()
            with self._spool_index_lock:
                index = self._spool_index_connection_locked()
                row = index.execute(
                    "SELECT request_id FROM pending ORDER BY request_id LIMIT 1"
                ).fetchone()
            if row is None:
                if self._dispatch_stop.is_set():
                    return
                self._dispatch_wakeup.wait(0.02)
                continue
            request_id = int(row[0])
            self._dispatch_id = request_id
            crop_path = self._spool_dir / f"{request_id:012d}.png"
            meta_path = self._spool_dir / f"{request_id:012d}.request"
            if not meta_path.is_file():
                marker = self._spool_dir / f"{request_id:012d}.ack"
                if marker.exists():
                    self._remove_spool_index_id(request_id)
                    continue
                self._set_dispatch_error(
                    f"OCR 请求 #{request_id} 索引待办存在但 sidecar 缺失：{meta_path}"
                )
                return
            if not crop_path.is_file():
                self._set_dispatch_error(f"OCR 请求 #{request_id} 缺少 PNG: {crop_path}")
                return
            try:
                frame_index, capture_timestamp, _, rectified_path = pickle.loads(
                    meta_path.read_bytes()
                )
                request = OcrRequest(
                    request_id, frame_index, capture_timestamp, str(crop_path), rectified_path
                )
                dispatch_start = timing.now_ns() if timing.ENABLED else 0
                # 队列满时复用已加载的请求直接等待空位，避免每 100ms 重复查 SQLite 与读盘。
                while not self._dispatch_stop.is_set():
                    try:
                        self._request_q.put(request, timeout=0.1)
                        break
                    except queue.Full:
                        continue
                else:
                    return
                if timing.ENABLED:
                    timing.emit(
                        "ocr_dispatch",
                        request_id=request_id,
                        frame_index=frame_index,
                        start_ns=dispatch_start,
                        end_ns=timing.now_ns(),
                    )
                self._dispatch_id = request_id + 1
                self._remove_spool_index_id(request_id)
            except _OcrSpoolIndexError:
                raise
            except sqlite3.Error:
                raise
            except Exception as exc:  # noqa: BLE001 - 解码 sidecar 的异常均需显式停止派发
                self._set_dispatch_error(
                    f"OCR 请求 #{request_id} 元数据损坏: {type(exc).__name__}: {exc}"
                )
                return

    def _scan_spool_state(self) -> tuple[int, int, int | None]:
        """流式恢复编号和积压，并在磁盘索引中保留稀疏待办号。"""
        maximum = pending = 0
        first_pending: int | None = None
        self._spool_dir.mkdir(parents=True, exist_ok=True)
        with self._spool_index_lock:
            index = self._spool_index_connection_locked()
            with index:
                index.execute("DELETE FROM pending")
                try:
                    entries = os.scandir(self._spool_dir)
                except FileNotFoundError:
                    return maximum, pending, first_pending
                with entries:
                    for entry in entries:
                        path = Path(entry.name)
                        if path.suffix not in {".request", ".ack"}:
                            continue
                        try:
                            request_id = int(path.stem)
                        except ValueError:
                            continue
                        maximum = max(maximum, request_id)
                        if path.suffix == ".ack":
                            continue
                        marker = self._spool_dir / f"{request_id:012d}.ack"
                        if marker.exists():
                            (self._spool_dir / f"{request_id:012d}.png").unlink(missing_ok=True)
                            (self._spool_dir / entry.name).unlink(missing_ok=True)
                        else:
                            index.execute(
                                "INSERT OR IGNORE INTO pending (request_id) VALUES (?)",
                                (request_id,),
                            )
                            pending += 1
                            first_pending = (
                                request_id
                                if first_pending is None
                                else min(first_pending, request_id)
                            )
        return maximum, pending, first_pending

    def _spool_index_connection_locked(self) -> sqlite3.Connection:
        """取得仅由索引锁保护的本地 SQLite 连接。"""
        path = self._spool_dir / ".dispatch.sqlite3"
        if self._spool_index is not None and self._spool_index_path == path:
            return self._spool_index
        if self._spool_index is not None:
            self._spool_index.close()
            self._spool_index = None
            self._spool_index_path = None
        index = sqlite3.connect(path, timeout=30.0, check_same_thread=False)
        try:
            index.execute("PRAGMA journal_mode=WAL")
            index.execute("PRAGMA synchronous=NORMAL")
            index.execute("PRAGMA temp_store=MEMORY")
            index.execute("CREATE TABLE IF NOT EXISTS pending (request_id INTEGER PRIMARY KEY)")
        except BaseException:
            index.close()
            raise
        self._spool_index = index
        self._spool_index_path = path
        return index

    def _remove_spool_index_id(self, request_id: int) -> None:
        try:
            with self._spool_index_lock:
                index = self._spool_index_connection_locked()
                with index:
                    index.execute("DELETE FROM pending WHERE request_id = ?", (request_id,))
        except sqlite3.Error as exc:
            raise _OcrSpoolIndexError(
                f"OCR 请求 #{request_id} SQLite 索引删除失败：{type(exc).__name__}: {exc}"
            ) from exc

    def _close_spool_index(self) -> None:
        with self._spool_index_lock:
            if self._spool_index is not None:
                self._spool_index.close()
                self._spool_index = None
                self._spool_index_path = None

    def _set_dispatch_error(self, message: str) -> None:
        self._dispatch_error = message
        self._dispatch_stop.set()
        self._dispatch_wakeup.set()
        LOGGER.error("OCR 磁盘请求派发失败，保留原文件：%s", message)

    @staticmethod
    def _unlink_request_files(request_id: int, crop_path: Path, meta_path: Path) -> None:
        """删除已确认请求的输入裁切图和元数据文件；失败仅记日志，不中断确认流程。"""
        try:
            crop_path.unlink(missing_ok=True)
            meta_path.unlink(missing_ok=True)
        except OSError:
            LOGGER.exception("OCR 已确认请求的临时文件清理失败 #%d", request_id)

    def _cleanup_spool(self) -> None:
        """后台异步清理已确认请求的 .png 与 .request 文件，收到 None 哨兵后退出。"""
        cleanup_q = self._cleanup_q
        while True:
            item = cleanup_q.get()
            if item is None:
                return
            request_id, crop_path, meta_path = item
            self._unlink_request_files(request_id, crop_path, meta_path)

    def _recover_acknowledged(self) -> None:
        """清理已完成结果在强制终止窗口留下的输入文件。"""
        for marker in self._spool_dir.glob("*.ack"):
            try:
                request_id = int(marker.stem)
            except ValueError:
                continue
            (self._spool_dir / f"{request_id:012d}.png").unlink(missing_ok=True)
            (self._spool_dir / f"{request_id:012d}.request").unlink(missing_ok=True)
            self._remove_spool_index_id(request_id)

    def _warn_if_backlogged(self) -> None:
        """积压超过阈值时告警（跨阈值一次，回落后可再告警）。

        跑在提交侧（感知线程），所以只做一次整数比较，不碰 RLock 之外的任何
        资源；真正重的是 OCR，不是这条日志。
        """
        pending = self._submitted - self._results
        if not self._backlog_warned and pending >= self._queue_size:
            self._backlog_warned = True
            LOGGER.warning(
                "OCR 积压 %d 个请求（阈值 %.0f）：请求保存在磁盘；"
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
        meta_path = self._spool_dir / f"{result.request_id:012d}.request"
        try:
            _, _, result.payload, _ = pickle.loads(meta_path.read_bytes())
        except Exception as exc:
            LOGGER.exception("读取 OCR 请求元数据失败 #%d；保留磁盘请求", result.request_id)
            raise OSError(f"OCR 请求元数据读取失败 #{result.request_id}") from exc
        return result

    def ack(self, result: OcrResult) -> None:
        """在业务关联和归档成功后确认结果，之后才删除请求文件。"""
        ack_start = timing.now_ns() if timing.ENABLED else 0
        meta_path = self._spool_dir / f"{result.request_id:012d}.request"
        crop_path = self._spool_dir / f"{result.request_id:012d}.png"
        marker = self._spool_dir / f"{result.request_id:012d}.ack"
        if marker.exists():
            self._recover_acknowledged()
            return
        if not meta_path.is_file() or not crop_path.is_file():
            raise OSError(f"OCR 请求文件缺失，拒绝确认 #{result.request_id}")
        temporary = marker.with_suffix(".ack.tmp")
        with temporary.open("wb") as stream:
            stream.write(f"{result.request_id}\n".encode("ascii"))
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(marker)
        self._remove_spool_index_id(result.request_id)
        self._results += 1
        # 积压回落到阈值一半以下时复位告警状态，下次再涨上来会重新提醒
        if self._backlog_warned and (self._submitted - self._results) < max(
            self._queue_size // 2, 1
        ):
            self._backlog_warned = False
        cleanup_thread = self._cleanup_thread
        if cleanup_thread is not None and cleanup_thread.is_alive():
            try:
                self._cleanup_q.put_nowait((result.request_id, crop_path, meta_path))
            except queue.Full:
                self._unlink_request_files(result.request_id, crop_path, meta_path)
        else:
            self._unlink_request_files(result.request_id, crop_path, meta_path)
        if timing.ENABLED:
            timing.emit(
                "ocr_ack",
                request_id=result.request_id,
                frame_index=result.frame_index,
                worker_id=result.worker_id,
                start_ns=ack_start,
                end_ns=timing.now_ns(),
            )

    def drain(self) -> list[OcrResult]:
        """把当前可用的结果全部取走。"""
        out: list[OcrResult] = []
        while True:
            result = self.poll()
            if result is None:
                return out
            out.append(result)
