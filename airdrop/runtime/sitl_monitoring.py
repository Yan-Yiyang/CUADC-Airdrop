"""SITL 感知性能采样。"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any

LOGGER = logging.getLogger("sitl_recon")


class PerformanceMonitor:
    """每 5 s 采一次性能样本（感知吞吐 / 积压 / 内存），日志一行、报告全量。

    三个"积压"口径要分开看：

    * ``lag_frames``——缓冲里**还没检测**的帧（检测线程追不上图传）；
    * ``pending``——送出去**还没回来**的 OCR 裁剪图（OCR 追不上检测）；
    * ``deduped``——跨帧去重**挡下**的送检（用来还原"不开去重的原始需求"）。

    内存两项靠 psutil（ultralytics 的传递依赖）；取不到就只少这两列，不影响别的。
    """

    def __init__(
        self,
        worker: Any,
        tracker: Any,
        source: Any,
        writer: Any,
        *,
        interval_s: float = 5.0,
    ) -> None:
        self._worker = worker
        self._tracker = tracker
        self._source = source
        self._writer = writer
        self._interval_s = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._started_at = time.monotonic()
        self._previous: dict[str, float] = {}
        self.samples: list[dict[str, Any]] = []
        self._process: Any = None
        try:  # psutil 是可选依赖：拿不到就降级（少内存两列）
            import psutil  # 可选依赖：不在导入期拉起它

            self._process = psutil.Process(os.getpid())
        except Exception:  # noqa: BLE001 - 观测不能影响主流程
            self._process = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="perf-monitor", daemon=True)
        self._thread.start()

    def stop(self) -> list[dict[str, Any]]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0 * self._interval_s)
            self._thread = None
        return self.samples

    def _run(self) -> None:
        while not self._stop.wait(self._interval_s):
            try:
                self._sample()
            except Exception:  # 观测绝不把任务带崩
                LOGGER.exception("性能采样失败")

    def _sample(self) -> None:
        stats = self._worker.stats
        tracker_stats = self._tracker.stats
        video = self._source.stats
        written, _skipped = self._writer.stats()
        now = time.monotonic()
        counters = {
            "frames": float(stats.frames),
            "submitted": float(stats.submitted),
            "results": float(stats.results),
            "deduped": float(stats.deduped),
            "video_frames": float(video.frames),
        }

        def rate(key: str) -> float:
            last = self._previous.get(key)
            last_at = self._previous.get("_at")
            if last is None or last_at is None or now <= last_at:
                return 0.0
            return (counters[key] - last) / (now - last_at)

        sample: dict[str, Any] = {
            "t_s": round(now - self._started_at, 1),
            "frames": stats.frames,
            "detections": stats.detections,
            "with_code": stats.with_code,
            "submitted": stats.submitted,
            "results": stats.results,
            "deduped": stats.deduped,
            "pending": max(stats.submitted - stats.results, 0),
            "queued": self._worker.queued_results,
            "points": tracker_stats.get("points", 0),
            "labeled": tracker_stats.get("labeled", 0),
            "lag_frames": stats.lag_frames,
            "ocr_dropped": stats.ocr_dropped,
            "video_frames": video.frames,
            "buffer_written": written,
            "detection_fps": round(rate("frames"), 1),
            "video_fps": round(rate("video_frames"), 1),
            "ocr_in_per_s": round(rate("submitted"), 1),
            "ocr_out_per_s": round(rate("results"), 1),
            "dedupe_per_s": round(rate("deduped"), 1),
            "rss_mb": None,
            "children_mb": None,
        }
        if self._process is not None:
            try:
                sample["rss_mb"] = round(self._process.memory_info().rss / 2**20)
                children = self._process.children(recursive=True)
                sample["children_mb"] = round(
                    sum(child.memory_info().rss for child in children) / 2**20
                )
            except Exception:  # noqa: BLE001 - 子进程可能刚好退出
                pass
        self._previous = {**counters, "_at": now}
        self.samples.append(sample)
        LOGGER.info(
            "性能 %5.0fs: 检测 %.1f fps（累计 %d 帧 → 检出 %d，读号 %d）|"
            "OCR 入 %.1f / 出 %.1f 次/s（积压 %d，去重挡下 %.1f 次/s）|"
            "缓冲落后 %s 帧|目标点 %d（带编号 %d，队列 %d）|内存 %s + 子进程 %s MB",
            sample["t_s"],
            sample["detection_fps"],
            sample["frames"],
            sample["detections"],
            sample["with_code"],
            sample["ocr_in_per_s"],
            sample["ocr_out_per_s"],
            sample["pending"],
            sample["dedupe_per_s"],
            sample["lag_frames"],
            sample["points"],
            sample["labeled"],
            sample["queued"],
            sample["rss_mb"],
            sample["children_mb"],
        )
