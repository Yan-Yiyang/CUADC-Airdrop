"""起飞前自检：**载入模型 + 视频自检**（正式任务流程的第一步）。

正式流程
--------
``INIT``（遥测 + NED 原点）→ **``PREFLIGHT``**（本模块：载入模型 → 视频自检）→
``WAIT_AIRBORNE``（等飞机在空中）→ ``RECON``（正式任务等操作手在 QGC 启动侦察航线）。

设计
----
"载入模型"与"视频自检"各自都要碰外部资源（GPU/权重文件、ffmpeg 视频子进程、相机标定），
而状态机必须**离线可测**，所以这里把每一步做成**注入**：

* ``model_loaders``：``{"detector": 载入函数, "ocr": ..., "camera": ...}``——键就是
  :class:`~airdrop.config.PreflightConfig` 里的开关名；关掉的项记一条 ``preflight``
  事件（``ok=None``）后跳过；
* ``video_source``：任何有 ``stats.frames``（以及可选 ``latest()``）的对象
  （:class:`~airdrop.video.Hm30VideoSource` / 回放源都行）；
* ``camera_model``：已载入的相机标定（用来核对画面分辨率是否与标定一致）。

失败一律 :class:`PreflightError`（消息里带**检查名**与原因）——状态机据此
``ABORT("preflight_failed:<check>")``。**检查被关掉 ≠ 检查通过**：关掉的项会进事件，
正式任务起飞前核一眼就知道哪几项没把关。
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:  # 仅类型标注：本模块运行时不依赖 config/video
    from .config import PreflightConfig

LOGGER = logging.getLogger(__name__)

__all__ = ["Preflight", "PreflightCheck", "PreflightError", "PreflightLike"]

#: 检查名（同时也是 ``PreflightConfig`` 里的开关名），顺序即执行顺序
CHECK_NAMES = ("detector", "ocr", "camera")


class PreflightError(RuntimeError):
    """起飞前自检失败（消息里带检查名与原因）。"""


@dataclass(frozen=True, slots=True)
class PreflightCheck:
    """一项检查的结果。``ok=None`` 表示该项被配置关掉（**没有把关**，不是通过）。"""

    name: str
    ok: bool | None
    detail: str = ""

    @property
    def skipped(self) -> bool:
        return self.ok is None

    def as_dict(self) -> dict[str, Any]:
        return {"check": self.name, "ok": self.ok, "detail": self.detail}


@runtime_checkable
class PreflightLike(Protocol):
    """状态机眼里的预检（实现方可以是 :class:`Preflight`，也可以是测试用的假对象）。

    ``run()`` 返回各项结果；失败可以直接抛 :class:`PreflightError`，也可以返回
    ``ok=False`` 的项——调用方两种都认。
    """

    def run(self) -> tuple[PreflightCheck, ...]: ...


class Preflight:
    """默认实现：按配置逐项做检查，**每项只做一次**（``run()`` 幂等）。

    参数
    ----
    config:
        :class:`~airdrop.config.PreflightConfig`（各开关 + 视频探测窗口/帧数上限）。
    model_loaders:
        键为 ``"detector"`` / ``"ocr"`` / ``"camera"`` 的载入回调；**开着却没给回调**
        视为接线错误 → 直接失败（不能静默当成通过）。
    video_source:
        视频源；读 ``stats.frames`` 计帧、``latest()`` 取最新帧核对分辨率。
    camera_model:
        已载入的相机标定（有 ``width``/``height`` 时核对画面尺寸）。
    """

    def __init__(
        self,
        config: "PreflightConfig",
        *,
        model_loaders: Mapping[str, Callable[[], Any]] | None = None,
        video_source: Any | None = None,
        camera_model: Any | None = None,
        on_event: Callable[[str, dict[str, Any]], None] | None = None,
        clock: Callable[[], float] = time.time,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._config = config
        self._loaders = dict(model_loaders or {})
        self._video_source = video_source
        self._camera_model = camera_model
        self._on_event = on_event
        self._clock = clock
        self._sleep = sleep
        self._done: tuple[PreflightCheck, ...] | None = None

    # ------------------------------------------------------------------
    def run(self) -> tuple[PreflightCheck, ...]:
        """跑一遍预检（幂等）。失败抛 :class:`PreflightError`。"""
        if self._done is not None:
            return self._done
        checks: list[PreflightCheck] = []
        for name in CHECK_NAMES:
            checks.append(self._load_one(name))
        checks.append(self._check_video())
        self._done = tuple(checks)
        LOGGER.info(
            "起飞前自检通过：%s",
            "、".join(f"{c.name}({c.detail})" if c.detail else c.name for c in checks),
        )
        return self._done

    # ------------------------------------------------------------------
    def _switch(self, name: str) -> bool:
        return bool(getattr(self._config, f"load_{name}", True))

    def _load_one(self, name: str) -> PreflightCheck:
        if not self._switch(name):
            return self._report(PreflightCheck(name, None, "配置里关掉了（未检查）"))
        loader = self._loaders.get(name)
        if loader is None:
            self._report(PreflightCheck(name, False, "没有注入载入回调"))
            raise PreflightError(f"{name}: 这项检查开着但没有注入载入回调")
        try:
            detail = loader()
        except Exception as exc:
            self._report(PreflightCheck(name, False, f"{type(exc).__name__}: {exc}"))
            raise PreflightError(f"{name}: {exc}") from exc
        return self._report(PreflightCheck(name, True, "" if detail is None else str(detail)))

    def _check_video(self) -> PreflightCheck:
        if not self._config.check_video:
            return self._report(PreflightCheck("video", None, "配置里关掉了（未检查）"))
        source = self._video_source
        if source is None:
            self._report(PreflightCheck("video", False, "没有接入视频源"))
            raise PreflightError("video: 没有接入视频源")

        want = int(self._config.video_min_frames)
        window = float(self._config.video_probe_s)
        deadline = self._clock() + window
        frames = self._frames()
        while frames < want and self._clock() < deadline:
            self._sleep(0.1)
            frames = self._frames()
        if frames < want:
            detail = f"{window:.1f}s 内只收到 {frames} 帧（要求 ≥ {want}）"
            self._report(PreflightCheck("video", False, detail))
            raise PreflightError(f"video: {detail}")

        size = self._expected_size()
        if size is not None:
            actual = self._frame_size()
            if actual is not None and actual != size:
                detail = f"画面 {actual[0]}x{actual[1]} 与标定 {size[0]}x{size[1]} 不一致"
                self._report(PreflightCheck("video", False, detail))
                raise PreflightError(f"video: {detail}")
        return self._report(PreflightCheck("video", True, f"{frames} 帧"))

    # ------------------------------------------------------------------
    def _frames(self) -> int:
        stats = getattr(self._video_source, "stats", None)
        try:
            return int(getattr(stats, "frames", 0) or 0)
        except TypeError, ValueError:  # pragma: no cover - 防御性
            return 0

    def _expected_size(self) -> tuple[int, int] | None:
        model = self._camera_model
        if model is None:
            return None
        width = getattr(model, "width", None)
        height = getattr(model, "height", None)
        if width and height:
            return (int(width), int(height))
        return None

    def _frame_size(self) -> tuple[int, int] | None:
        latest = getattr(self._video_source, "latest", None)
        if latest is None:
            return None
        try:
            frame = latest()
        except Exception:
            LOGGER.exception("取最新帧失败（跳过分辨率核对）")
            return None
        if frame is None:
            return None
        width = getattr(frame, "width", None)
        height = getattr(frame, "height", None)
        if width and height:
            return (int(width), int(height))
        return None

    def _report(self, check: PreflightCheck) -> PreflightCheck:
        LOGGER.info("起飞前自检 %s：%s", check.name, check.detail or "通过")
        if self._on_event is not None:
            try:
                self._on_event("preflight", check.as_dict())
            except Exception:
                LOGGER.exception("写入 preflight 事件失败（自检继续）")
        return check
