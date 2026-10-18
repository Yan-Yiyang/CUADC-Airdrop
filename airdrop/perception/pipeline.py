"""感知主循环：逐帧 YOLO → （OCR 送检 | cls12 直出）→ :class:`Detection`。

数据流
------
::

    AlignmentBuffer ──wait_new(一帧不落)──> PerceptionWorker
                                              ├─ YOLO 检测（全帧）
                                              ├─ ocr 模式：裁剪 → 按目标去重 → mp.Queue → OCR 进程×N
                                              └─ cls12 模式：类别直出编号
                                            OCR 结果回填 ──> Detection ──> 结果队列（消费者取走）

两条铁律
--------
1. **输入必须来自缓冲**（``wait_new`` / ``iter_between``），不能用 ``read()``
   循环——那条路允许丢帧，而"目标可能只出现一瞬"。本类只接受
   :class:`~airdrop.video.buffer.AlignmentBuffer`。
2. **OCR 绝不阻塞逐帧检测**。OCR 单帧约 100ms（GPU）而 YOLO 约 29ms，内联
   OCR 会让有效帧率掉到 1/4。所以裁剪图送进**独立进程**池，检测继续跑；
   编号晚几帧回来无所谓（同一目标在整个侦查段会被看到几十次）。

OCR 逐帧送检
------------
默认不启用跨帧去重；显式打开时才按时间/像素窗口合并相邻重复请求。目标可能只在
少数帧中清晰可见，运动模糊会让相邻帧的可识别性不同；**每个 YOLO 检测结果都会
进入无界队列，绝不因为"队列满"而丢掉送检请求或结果**。OCR 落后时队列会积压
（只告警、不丢弃），内存账由 ``ocr_queue_size`` 阈值上的 WARNING 与
:attr:`PerceptionStats` 的 ``submitted - results`` 如实反映。
"""

from __future__ import annotations

import logging
import math
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any, cast

import numpy as np

from ..video.buffer import AlignmentBuffer, BufferedFrame
from .cropproc import OcrEngineConfig
from .detector import Detector, DetectorConfig, FrameSizeMismatchError
from .models import (
    Detection,
    DetectionSnapshot,
    ProcessedFrame,
    snapshot_detection,
    snapshot_pose,
)
from .ocr_worker import OcrResult, OcrWorkerPool

LOGGER = logging.getLogger(__name__)

#: 检测结果队列积压告警阈值（队列本身无界）：消费者（target_result 回调）没在抽干
#: 时只告警一次，提醒确认它还在跑。
_RESULT_BACKLOG_WARN = 2000

#: 纯像素去重（``ocr_dedupe_s <= 0``）时保留的最近位置条数上限——没有时间窗可
#: 淘汰，用条数兜住内存。
_DEDUPE_MAX_RECENT = 64

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
    #: OCR **积压告警阈值**（不是队列容量：请求/结果队列都无界、不丢弃，见 ocr_worker）
    ocr_queue_size: int = 500
    # 仅在显式开启时生效；默认关闭，避免误伤运动模糊下唯一清晰帧。
    # 两个阈值是**与**关系：ocr_dedupe_s = 时间窗（秒），ocr_dedupe_px = 像素半径。
    # 只设其一也有效——只设 px 时不做时间窗判断（最近若干条里位置够近就算重复）。
    ocr_dedupe_s: float = 0.0
    ocr_dedupe_px: float = 0.0
    min_side_px: float = 10.0
    max_side_px: float = 400.0
    engine: OcrEngineConfig = field(default_factory=OcrEngineConfig)
    #: 相机内参/畸变（标定产出）。给了就走 :class:`Detector` 的**预计算 remap**
    #: 快路径（比逐帧 undistort 快 3~5 倍），并在 ``Detection.extra["undistorted"]``
    #: 上打标记，下游 georef 据此不再逐点去畸变。
    #: 曾经这三个字段不存在、装配时也没人传，于是整条快路径是死代码：每帧都退回
    #: 逐点 ``cv2.undistortPoints``，且"帧尺寸与标定不符"这条保护无人把守。
    camera_matrix: np.ndarray | None = None
    dist_coeffs: np.ndarray | None = None
    calib_size: tuple[int, int] | None = None

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
    #: OCR 请求丢弃数——**不丢弃策略下恒为 0**，保留作监控（一旦非 0 说明有新的丢弃路径）
    ocr_dropped: int = 0
    #: OCR 工作进程**存活**数（cls12 模式或未注入池时为 0）。
    #: 小于 workers 说明有 worker 掉了，它名下的请求永远不会有结果——见
    #: :meth:`PerceptionWorker._check_pool_health`。
    ocr_alive: int = 0
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
                camera_matrix=config.camera_matrix,
                dist_coeffs=config.dist_coeffs,
                calib_size=config.calib_size,
            )
        )
        self._pool = pool
        self._owns_pool = pool is None and config.mode == "ocr"

        # 结果队列**无界**：检测结果同样不丢（消费者慢时只积压，不丢目标）。
        # 积压超过 _RESULT_BACKLOG_WARN 时告警一次（见 _emit）。
        self._results: queue.Queue[Detection] = queue.Queue()
        self._results_warned = False
        self._stats = PerceptionStats()
        self._stats_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._drained = threading.Event()
        self._drain_to: int | None = None
        self._drain_ok = False
        self._fatal_error: str | None = None
        self._run_finished = False
        self._thread: threading.Thread | None = None
        self._cursor = self._start_index
        self._input_lock = threading.Lock()
        self._completed_index = self._start_index
        self._cutoff: int | None = None
        # 去重缓存：[(capture_timestamp, u, v)]；**只在启用去重时记录**，
        # 否则它会被每帧的检测结果喂大（见 _process_frame 的说明）。
        self._recent: list[tuple[float, float, float]] = []
        #: 送检中的 OCR 请求：request_id → 该帧的基础 Detection（结果回来时回填编号）
        self._pending: dict[int, Detection] = {}
        self._on_detection: Callable[[Detection], None] | None = None
        self._observer_lock = threading.Lock()
        self._frame_observers: set[Callable[[ProcessedFrame], None]] = set()
        self._result_observers: set[Callable[[DetectionSnapshot], None]] = set()

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

    def add_frame_observer(self, callback: Callable[[ProcessedFrame], None]) -> None:
        """添加整帧旁路观察者；每帧检测和 OCR 送检后调用，包括零检测帧。"""
        with self._observer_lock:
            self._frame_observers.add(callback)

    def remove_frame_observer(self, callback: Callable[[ProcessedFrame], None]) -> None:
        """移除整帧旁路观察者。"""
        with self._observer_lock:
            self._frame_observers.discard(callback)

    def add_result_observer(self, callback: Callable[[DetectionSnapshot], None]) -> None:
        """添加结果旁路观察者，不消费业务结果队列。"""
        with self._observer_lock:
            self._result_observers.add(callback)

    def remove_result_observer(self, callback: Callable[[DetectionSnapshot], None]) -> None:
        """移除结果旁路观察者。"""
        with self._observer_lock:
            self._result_observers.discard(callback)

    @property
    def processing(self) -> bool:
        """检测游标已取走、但整帧处理还未完成。"""
        with self._input_lock:
            return self._completed_index < self._cursor

    def finish_input(self) -> int:
        """固定侦查截止帧（幂等）；新图传仍写原缓冲，感知只排空固定范围。"""
        with self._input_lock:
            if self._cutoff is None:
                self._buffer = self._buffer.freeze(after_index=self._cursor)
                self._cutoff = self._buffer.latest_index()
            return self._cutoff

    def raise_if_failed(self) -> None:
        """将后台尺寸配置错误交给主线程，避免感知已停而任务继续。"""
        with self._stats_lock:
            error = self._fatal_error
        if error is not None:
            raise FrameSizeMismatchError(error)

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------
    def start(self) -> "PerceptionWorker":
        """起工作线程；OCR 模式下同时拉起 OCR 进程池。"""
        if self.running:
            return self
        self.raise_if_failed()
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
        self._run_finished = False
        self._drained.clear()
        self._drain_to = None
        self._drain_ok = False
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
        if thread is not None:
            if thread.is_alive():
                thread.join(timeout)
            if thread.is_alive():
                # 线程没停就别清引用（否则 running 会撒谎、start() 会起第二个）；
                # 线程仍可能提交或收取 OCR，不能由调用方同时操作池。
                LOGGER.warning("感知线程未在 %.1fs 内退出，保留引用（可再调 stop()）", timeout)
                return
            else:
                self._thread = None
        if self._owns_pool and self._pool is not None:
            self._pool.stop()
        self._collect_remaining(timeout=0.5)

    def drain(self, timeout: float = 120.0) -> bool:
        """先停止生产者，再等待既有帧与 OCR 完成；失败或超时返回 False。

        由工作线程在完整处理一帧之后确认，不能用已经提前推进的游标判断在途帧。
        本方法不消费结果队列，调用方仍须抽取结果；生命周期操作应由单一调用方串行执行。
        """
        if not math.isfinite(timeout) or timeout < 0:
            raise ValueError("排空超时必须是非负有限秒数")
        with self._stats_lock:
            self._drained.clear()
            self._drain_ok = False
            self._drain_to = self._buffer.latest_index()
            if not self.running or self._run_finished:
                return False
        if not self._drained.wait(timeout):
            return False
        with self._stats_lock:
            return self._drain_ok

    def _check_drained(self) -> None:
        """仅工作线程在帧处理与结果回填之后确认排空。"""
        if self._drain_to is None:
            return
        with self._stats_lock:
            if self._cursor >= self._drain_to and not self._pending:
                self._drain_ok = self._stats.last_error is None
                self._drained.set()

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

    @property
    def queued_results(self) -> int:
        """结果队列里还没被消费的条数（监控用）。

        队列**无界、不丢弃**：消费者（``MissionRunner`` 的 ``target_result``/``pump``）
        不抽，结果就一直留着等抽——所以收尾时这个数字应该回落到 0。
        """
        return self._results.qsize()

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
                source_buffer = self._buffer
                frame = source_buffer.wait_new(after_index=self._cursor, timeout=0.5)
                if frame is None:
                    # 缓冲里暂时没有新帧：先把 OCR 结果收回来，保持结果流不断
                    self._collect_remaining(timeout=0.0)
                    self._check_pool_health()
                    self._check_drained()
                    continue
                with self._input_lock:
                    # 截止可能发生在等待或解码期间；改从固定快照重新取，不能送检新帧。
                    if source_buffer is not self._buffer:
                        continue
                    self._cursor = frame.index
                try:
                    self._process_frame(frame)
                except FrameSizeMismatchError as exc:
                    with self._stats_lock:
                        self._fatal_error = self._stats.last_error = str(exc)
                    LOGGER.error("%s", exc)
                    return
                except Exception as exc:
                    LOGGER.exception("处理帧 #%d 失败", frame.index)
                    with self._stats_lock:
                        self._stats.last_error = f"{type(exc).__name__}: {exc}"
                self._collect_remaining(timeout=0.0)
                with self._input_lock:
                    self._completed_index = frame.index
                    if self._cutoff is not None and frame.index == self._cutoff:
                        # 尾帧已送检，OCR 持有独立裁剪；释放固定快照中的整帧引用。
                        self._buffer.clear()
                self._check_pool_health()
                self._check_drained()
        except Exception as exc:
            LOGGER.exception("感知线程异常退出")
            with self._stats_lock:
                self._stats.last_error = f"{type(exc).__name__}: {exc}"

        finally:
            with self._stats_lock:
                self._run_finished = True
                self._drained.set()

    def _check_pool_health(self) -> None:
        """巡检 OCR 进程池存活数（有 worker 掉了会打一条节流的 ERROR）。

        worker 死亡的路径很现实（CUDA OOM、RapidOCR/torch 原生崩溃、结果队列
        写入自身抛错）。掉了之后它名下的请求永远不返回，而积压告警只打一条就
        不再打 ⇒ 全死时表现为"永远零产出"，日志里看不出已经死了。这里每拍
        一次整数比较，把故障显式喊出来。
        """
        if self._pool is None:
            return
        # pool 是可注入的测试接缝：自定义实现可能没有存活巡检接口。缺失就跳过
        # （而不是让 AttributeError 打死感知线程——那会静默丢掉后面所有帧）。
        alive = getattr(self._pool, "alive", None)
        if alive is None:
            return
        with self._stats_lock:
            self._stats.ocr_alive = int(alive)
        check = getattr(self._pool, "check_alive", None)
        if check is not None:
            check()

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
            self._notify_frame_observers(frame, batch.image, batch.undistorted, batch.detections)
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
            # 只有去重开着时才记位置缓存：关闭去重时 _expire 不会被调用（它的
            # 唯一调用点在 _should_skip 里），记下来的条目会随每帧检测无界增长。
            if self._dedupe_enabled():
                self._remember(detection)
            self._submit(detection, crop)
        self._notify_frame_observers(frame, batch.image, batch.undistorted, batch.detections)

    def _make_frame_snapshot(
        self,
        frame: BufferedFrame,
        image: np.ndarray,
        undistorted: bool,
        detections: tuple[Detection, ...],
    ) -> ProcessedFrame:
        """复制图像和姿态标量，构造与业务对象无共享可变引用的帧快照。"""
        snapshot = frame.snapshot
        pose = snapshot_pose(snapshot)
        configured_matrix = self._detector.config.camera_matrix
        matrix = None
        if configured_matrix is not None:
            values = np.asarray(configured_matrix, dtype=np.float64).reshape(3, 3)
            if np.isfinite(values).all():
                matrix = cast(
                    tuple[tuple[float, float, float], ...],
                    tuple(tuple(float(value) for value in row) for row in values),
                )
        height, width = image.shape[:2]
        immutable_image = np.frombuffer(image.tobytes(), dtype=image.dtype).reshape(image.shape)
        immutable_image.flags.writeable = False
        return ProcessedFrame(
            frame_index=frame.index,
            received_timestamp=float(frame.received_timestamp),
            capture_timestamp=float(frame.capture_timestamp),
            processed_timestamp=time.time(),
            image=immutable_image,
            image_size=(int(width), int(height)),
            camera_matrix=matrix,
            undistorted=bool(undistorted),
            pose=pose,
            detections=tuple(snapshot_detection(item) for item in detections),
        )

    def _notify_frame_observers(
        self,
        source_frame: BufferedFrame,
        image: np.ndarray,
        undistorted: bool,
        detections: tuple[Detection, ...],
    ) -> None:
        with self._observer_lock:
            observers = tuple(self._frame_observers)
        if not observers:
            return
        frame = self._make_frame_snapshot(source_frame, image, undistorted, detections)
        for observer in observers:
            try:
                observer(frame)
            except Exception:
                LOGGER.exception("处理帧观察者异常")

    # ------------------------------------------------------------------
    # 去重
    # ------------------------------------------------------------------
    def _dedupe_enabled(self) -> bool:
        """是否启用跨帧去重（两个阈值都为 0 = 关闭）。"""
        config = self._config
        return config.ocr_dedupe_s > 0 or config.ocr_dedupe_px > 0

    def _should_skip(self, detection: Detection) -> bool:
        """在启用去重时，跳过相邻时间/位置近似重复的 OCR 请求。

        两个阈值是**与**关系：都要越过才算重复。``ocr_dedupe_s <= 0`` 时不做
        时间窗判断（纯位置去重），``ocr_dedupe_px <= 0`` 时不做位置判断。
        """
        if not self._dedupe_enabled():
            return False
        config = self._config
        u, v = detection.pixel
        now = detection.capture_timestamp
        self._expire(now)
        window = config.ocr_dedupe_s
        radius = config.ocr_dedupe_px
        for stamp, su, sv in self._recent:
            if window > 0 and now - stamp > window:
                continue
            if radius > 0 and abs(su - u) <= radius and abs(sv - v) <= radius:
                return True
        return False

    def _remember(self, detection: Detection) -> None:
        """记住一条已提交送检的目标位置，用于后续最近窗口去重。"""
        u, v = detection.pixel
        self._recent.append((detection.capture_timestamp, u, v))
        if len(self._recent) > _DEDUPE_MAX_RECENT:
            # 没有时间窗时（只按 ocr_dedupe_px 去重）没有"过期"的概念，
            # 用条数上限兜住内存；这条路径过去是被 _expire 无条件 clear() 掉的，
            # 于是"只设 ocr_dedupe_px"会静默完全不去重、且没有任何日志。
            del self._recent[: len(self._recent) - _DEDUPE_MAX_RECENT]

    def _expire(self, now: float) -> None:
        """淘汰超出时间窗口的历史位置，避免缓存无限增长。

        ``ocr_dedupe_s <= 0``（纯位置去重）时**不清空**：改由
        :meth:`_remember` 的条数上限兜底。清空会让刚记下的条目下一拍就没了，
        纯位置去重形同虚设。
        """
        window = self._config.ocr_dedupe_s
        if window <= 0:
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
        # 池侧丢弃计数同步进统计（不丢弃策略下恒为 0；留着是为了万一将来
        # 又出现丢弃路径，监控里能立刻看见）
        with self._stats_lock:
            self._stats.ocr_dropped = int(pool.stats.get("dropped", 0))
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
        # 结果队列无界：不丢弃任何检测结果。积压只告警（跨阈值一次），
        # 提醒确认消费者（target_result/pump）还在抽。
        self._results.put_nowait(detection)
        size = self._results.qsize()
        if not self._results_warned and size >= _RESULT_BACKLOG_WARN:
            self._results_warned = True
            LOGGER.warning(
                "检测结果积压 %d 条（阈值 %d）：**不丢弃**，但消费者可能没在抽干"
                "（检查 MissionRunner 的 target_result 回调是否在跑）",
                size,
                _RESULT_BACKLOG_WARN,
            )
        elif self._results_warned and size < _RESULT_BACKLOG_WARN // 2:
            self._results_warned = False
        callback = self._on_detection
        if callback is not None:
            try:
                callback(detection)
            except Exception:
                LOGGER.exception("检测结果回调异常")
        with self._observer_lock:
            observers = tuple(self._result_observers)
        display_snapshot = snapshot_detection(detection)
        for observer in observers:
            try:
                observer(display_snapshot)
            except Exception:
                LOGGER.exception("检测结果观察者异常")
