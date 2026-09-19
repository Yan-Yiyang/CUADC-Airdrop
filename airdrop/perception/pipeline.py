'感知主循环：逐帧 YOLO → （OCR 送检 | cls12 直出）→ :class:`Detection`。\n\n    AlignmentBuffer ──wait_new(一帧不落)──> PerceptionWorker\n                                              ├─ YOLO 检测（全帧）\n                                              ├─ ocr 模式：裁剪 → 按目标去重 → mp.Queue → OCR 进程×N\n                                              └─ cls12 模式：类别直出编号\n                                            OCR 结果回填 ──> Detection ──> 结果队列（消费者取走）\n\n两条铁律\n--------\n1. **输入必须来自缓冲**（``wait_new`` / ``iter_between``），不能用 ``read()``\n   循环——那条路允许丢帧，而"目标可能只出现一瞬"。本类只接受\n   :class:`~airdrop.video.buffer.AlignmentBuffer`。\n2. **OCR 绝不阻塞逐帧检测**。OCR 单帧约 100ms（GPU）而 YOLO 约 29ms，内联\n   OCR 会让有效帧率掉到 1/4。所以裁剪图送进**独立进程**池，检测继续跑；\n   编号晚几帧回来无所谓（同一目标在整个侦查段会被看到几十次）。\n\n去重（``ocr_dedupe_s`` / ``ocr_dedupe_px``）\n--------------------------------------------\n同一目标在连续几十帧里都被检出，每帧都送 OCR 是纯浪费（也是队列被打满的主因）。\n规则：**同一目标**（中心位移 < ``ocr_dedupe_px``）在 ``ocr_dedupe_s`` 秒内只送\n一次。去重在**送检时**按"画面拍摄时刻"判断，所以多目标同帧互不误伤。\n'

from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from ..video.buffer import AlignmentBuffer, BufferedFrame
from .cropproc import OcrEngineConfig
from .detector import Detector, DetectorConfig
from .models import Detection
from .ocr_worker import OcrResult, OcrWorkerPool, put_drop_oldest

LOGGER = logging.getLogger(__name__)

__all__ = ["PerceptionConfigLike", "PerceptionStats", "PerceptionWorker"]


@dataclass(frozen=True, slots=True)
class PerceptionConfigLike:
    """pipeline 需要的参数切片（由 ``airdrop.config.PerceptionConfig`` 装配）。

    单独定义一份是为了让 pipeline 能被离线测试直接构造，而不必拖上整个
    ``Config``（后者会牵到 video/telemetry 的一堆默认值）。
    """

    mode: str = "ocr"
    model_path: str = "models/best2.pt"
    device: str = "0"
    conf_threshold: float = 0.25
    imgsz: int = 1280
    target_color: str = "blue"
    ocr_conf_threshold: float = 0.6
    ocr_workers: int = 2
    ocr_queue_size: int = 500
    ocr_dedupe_s: float = 0.8
    ocr_dedupe_px: float = 30.0
    min_side_px: float = 10.0
    max_side_px: float = 400.0
    engine: OcrEngineConfig = field(default_factory=OcrEngineConfig)

    def __post_init__(self) -> None:
        if self.mode not in ("ocr", "cls12"):
            raise ValueError(f"mode 只支持 ocr/cls12: {self.mode!r}")
        if self.target_color not in ("blue", "red"):
            raise ValueError(f"target_color 只支持 blue/red: {self.target_color!r}")
        if self.ocr_dedupe_s < 0 or self.ocr_dedupe_px < 0:
            raise ValueError("去重阈值不能为负")


@dataclass(slots=True)
class PerceptionStats:
    """感知统计（运行中可随时取）。"""

    frames: int = 0  # 已处理帧数
    detections: int = 0  # 累计检测到的目标数
    with_code: int = 0  # 其中读出了编号的
    submitted: int = 0  # 送 OCR 的裁剪图数
    deduped: int = 0  # 被去重挡下的送检数
    results: int = 0  # 收到的 OCR 结果数
    invalid_side: int = 0  # 像素边长超出有效区间而被丢弃的
    ocr_dropped: int = 0  # OCR 队列满而丢掉的请求数
    lag_frames: int = 0  # 处理落后最新帧多少帧
    last_error: str | None = None


class PerceptionWorker:
    """感知主循环。

    用法::

        worker = PerceptionWorker(config, buffer=buffer)
        worker.start()                      # 起工作线程（ocr 模式会拉起 OCR 进程池）
        ...
        for detection in worker.iter_results():   # 或在别处 drain_results()
            ...
        worker.stop()
    """

    def __init__(
        self,
        config: PerceptionConfigLike,
        *,
        buffer: AlignmentBuffer,
        detector: Detector | None = None,
        pool: OcrWorkerPool | None = None,
        start_index: int = 0,
    ) -> None:
        self._config = config
        self._buffer = buffer
        self._start_index = int(start_index)
        self._detector = detector or Detector(
            DetectorConfig(
                model_path=config.model_path,
                device=config.device,
                conf_threshold=config.conf_threshold,
                imgsz=config.imgsz,
            )
        )
        self._pool = pool
        self._owns_pool = pool is None and config.mode == "ocr"

        self._results: queue.Queue[Detection] = queue.Queue(maxsize=2000)
        self._stats = PerceptionStats()
        self._stats_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._cursor = self._start_index
        # 去重表：[(拍摄时刻, u, v)]，按时间自然有序，过期从头部丢掉
        self._recent: list[tuple[float, float, float]] = []
        #: 送检中的 OCR 请求：request_id → 该帧的基础 Detection（结果回来时回填编号）
        self._pending: dict[int, Detection] = {}
        self._on_detection: Callable[[Detection], None] | None = None

    # ------------------------------------------------------------------
    # 状态
    # ------------------------------------------------------------------
    @property
    def config(self) -> PerceptionConfigLike:
        return self._config

    @property
    def detector(self) -> Detector:
        return self._detector

    @property
    def pool(self) -> OcrWorkerPool | None:
        return self._pool

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def stats(self) -> PerceptionStats:
        with self._stats_lock:
            return replace(self._stats)

    @property
    def lag_frames(self) -> int:
        """处理落后最新入缓冲帧多少帧（全帧处理，仅作监控）。"""
        latest = self._buffer.latest_index()
        with self._stats_lock:
            return max(latest - self._cursor, 0)

    def set_result_callback(self, callback: Callable[[Detection], None] | None) -> None:
        """注册检测结果回调（在工作线程里被调用，必须快）。"""
        self._on_detection = callback

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    def start(self) -> "PerceptionWorker":
        """起工作线程；OCR 模式下同时拉起 OCR 进程池。"""
        if self.running:
            return self
        if self._config.mode == "ocr":
            if self._pool is None:
                self._pool = OcrWorkerPool(
                    workers=self._config.ocr_workers,
                    queue_size=self._config.ocr_queue_size,
                    config={
                        "color": self._config.target_color,
                        "ocr_conf_threshold": self._config.ocr_conf_threshold,
                        "engine": {
                            "model_dir": self._config.engine.model_dir,
                            "det_model": self._config.engine.det_model,
                            "rec_model": self._config.engine.rec_model,
                            "rec_keys": self._config.engine.rec_keys,
                            "use_cuda": self._config.engine.use_cuda,
                            "device_id": self._config.engine.device_id,
                            "text_score": self._config.engine.text_score,
                            "rec_batch_num": self._config.engine.rec_batch_num,
                            "log_level": self._config.engine.log_level,
                        },
                    },
                )
            self._pool.start()
        else:
            # cls12 不需要 OCR：编号直接来自 YOLO 类别
            self._detector.load()

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="perception", daemon=True)
        self._thread.start()
        LOGGER.info(
            "感知线程已启动（mode=%s，buffer 现有 %d 帧）",
            self._config.mode,
            len(self._buffer),
        )
        return self

    def stop(self, timeout: float = 10.0) -> None:
        """停工作线程与 OCR 进程池（幂等）。"""
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout)
            if thread.is_alive():
                LOGGER.warning("感知线程未在 %.1fs 内退出", timeout)
        self._thread = None
        if self._owns_pool and self._pool is not None:
            self._pool.stop()
        self._collect_remaining(timeout=0.5)

    def __enter__(self) -> "PerceptionWorker":
        return self.start()

    def __exit__(self, *exc_info: object) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # 结果
    # ------------------------------------------------------------------
    def poll_result(self, timeout: float = 0.0) -> Detection | None:
        """取一条结果；没有则返回 None。"""
        try:
            return self._results.get(timeout=timeout) if timeout else self._results.get_nowait()
        except queue.Empty:
            return None

    def drain_results(self) -> list[Detection]:
        """把当前可用结果全部取走。"""
        out: list[Detection] = []
        while True:
            item = self.poll_result()
            if item is None:
                return out
            out.append(item)

    def iter_results(self, timeout: float = 0.5) -> Any:
        """迭代结果，直到 :meth:`stop` 且队列排空。"""
        idle_deadline = time.monotonic() + timeout
        while True:
            item = self.poll_result(timeout=min(timeout, 0.05))
            if item is not None:
                idle_deadline = time.monotonic() + timeout
                yield item
                continue
            if self._stop_event.is_set() and self._results.empty():
                return
            if time.monotonic() > idle_deadline:
                return

    # ------------------------------------------------------------------
    # 处理循环
    # ------------------------------------------------------------------
    def _run(self) -> None:
        try:
            while not self._stop_event.is_set():
                frame = self._buffer.wait_new(after_index=self._cursor, timeout=0.5)
                if frame is None:
                    # 缓冲里暂时没有新帧：先把 OCR 结果收回来，保持结果流不断
                    self._collect_remaining(timeout=0.0)
                    continue
                self._cursor = frame.index
                try:
                    self._process_frame(frame)
                except Exception as exc:
                    LOGGER.exception("处理帧 #%d 失败", frame.index)
                    with self._stats_lock:
                        self._stats.last_error = f"{type(exc).__name__}: {exc}"
                self._collect_remaining(timeout=0.0)
        except Exception as exc:
            LOGGER.exception("感知线程异常退出")
            with self._stats_lock:
                self._stats.last_error = f"{type(exc).__name__}: {exc}"

    def _process_frame(self, frame: BufferedFrame) -> None:
        """一帧的完整处理：检测 → 按模式产出。"""
        batch = self._detector.detect(
            frame.image,
            frame_index=frame.index,
            capture_timestamp=frame.capture_timestamp,
            telemetry=frame.snapshot,
            mode=self._config.mode,
        )
        with self._stats_lock:
            self._stats.frames += 1
            self._stats.detections += len(batch.detections)
            self._stats.lag_frames = max(self._buffer.latest_index() - frame.index, 0)

        if self._config.mode == "cls12":
            # 编号已由类别给出，直接产出
            for detection in batch.detections:
                self._emit(detection)
            return

        for detection in batch.detections:
            crop = self._detector.crop(batch, detection)
            if crop.size == 0:
                continue
            if self._should_skip(detection):
                with self._stats_lock:
                    self._stats.deduped += 1
                continue
            self._remember(detection)
            self._submit(detection, crop)

    # ------------------------------------------------------------------
    # 去重
    # ------------------------------------------------------------------
    def _should_skip(self, detection: Detection) -> bool:
        """该目标在去重窗口内是否已经送检过。"""
        config = self._config
        if config.ocr_dedupe_s <= 0 and config.ocr_dedupe_px <= 0:
            return False
        u, v = detection.pixel
        now = detection.capture_timestamp
        self._expire(now)
        for stamp, su, sv in self._recent:
            if now - stamp > config.ocr_dedupe_s:
                continue
            if abs(su - u) <= config.ocr_dedupe_px and abs(sv - v) <= config.ocr_dedupe_px:
                return True
        return False

    def _remember(self, detection: Detection) -> None:
        u, v = detection.pixel
        self._recent.append((detection.capture_timestamp, u, v))

    def _expire(self, now: float) -> None:
        window = self._config.ocr_dedupe_s
        if window <= 0:
            self._recent.clear()
            return
        keep_from = now - window
        if self._recent and self._recent[0][0] < keep_from:
            self._recent = [item for item in self._recent if item[0] >= keep_from]

    # ------------------------------------------------------------------
    # OCR 送检与回填
    # ------------------------------------------------------------------
    def _submit(self, detection: Detection, crop: np.ndarray) -> None:
        pool = self._pool
        if pool is None:  # pragma: no cover - start() 已保证
            return
        payload = detection
        request_id = pool.submit(
            crop,
            frame_index=detection.frame_index,
            capture_timestamp=detection.capture_timestamp,
            payload=payload,
        )
        if request_id is None:
            return
        self._pending[request_id] = payload
        with self._stats_lock:
            self._stats.submitted += 1

    def _collect_remaining(self, timeout: float) -> None:
        """把 OCR 结果取回来，回填成 :class:`Detection` 并产出。"""
        pool = self._pool
        if pool is None:
            return
        while True:
            result = pool.poll(timeout=timeout) if timeout else pool.poll()
            if result is None:
                break
            timeout = 0.0  # 只在第一次调用时允许等待
            self._apply_result(result)

    def _apply_result(self, result: OcrResult) -> None:
        base = self._pending.pop(result.request_id, None)
        if base is None:
            return
        with self._stats_lock:
            self._stats.results += 1
        if result.error is not None:
            with self._stats_lock:
                self._stats.last_error = result.error
            LOGGER.warning("帧 #%d 的 OCR 失败: %s", result.frame_index, result.error)
            self._emit(base, set_code=True)
            return
        side_px = result.side_px
        if result.number is None:
            # 没读出编号：仍然产出（画面里确实有目标），但 code 为空
            self._emit(base, set_code=True, side_px=side_px, raw_text=result.raw_text)
            return
        if not (self._config.min_side_px <= side_px <= self._config.max_side_px):
            # 像素边长不合理 → 不参与坐标解算（见 4.3 有效性门限）
            with self._stats_lock:
                self._stats.invalid_side += 1
            self._emit(base, set_code=True, side_px=side_px, raw_text=result.raw_text)
            return
        self._emit(
            base,
            code=result.number,
            set_code=True,
            side_px=side_px,
            raw_text=result.raw_text,
            ocr_confidence=result.confidence,
            extra={
                "ocr_stage": result.stage,
                "ocr_worker": result.worker_id,
                "saturation_level": result.saturation_level,
            },
        )

    def _emit(
        self,
        base: Detection,
        *,
        code: int | None = None,
        set_code: bool = False,
        side_px: float = 0.0,
        raw_text: str = "",
        ocr_confidence: float = 0.0,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """把一条检测结果交给消费者（队列 + 可选回调）。

        ``set_code=False``（默认）表示**保留 base 上已有的编号**——cls12 模式的
        编号是 YOLO 类别直接给的，不需要（也不能）由 OCR 覆盖。OCR 路径显式传
        ``set_code=True``，这样"没读出来"（code=None）也能如实覆盖掉一个可能
        存在的旧值。
        """
        detection = replace(
            base,
            code=(code if set_code else base.code),
            side_px=side_px,
            raw_text=raw_text,
            ocr_confidence=ocr_confidence,
        )
        if extra:
            detection = replace(detection, extra={**base.extra, **extra})
        with self._stats_lock:
            if detection.code is not None:
                self._stats.with_code += 1
        if put_drop_oldest(self._results, detection):
            LOGGER.debug("结果队列满，丢弃最旧的一条检测")
        callback = self._on_detection
        if callback is not None:
            try:
                callback(detection)
            except Exception:
                LOGGER.exception("检测结果回调异常")
