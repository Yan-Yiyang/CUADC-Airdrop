"""真实与 SITL 共用的组件生命周期。"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from airdrop.mission.states import MissionState

from .options import RunOptions

if TYPE_CHECKING:
    from airdrop.georef import LLARef
    from airdrop.mission.runner import MissionStats, MissionTransition
    from airdrop.mission.targets import TargetTracker
    from airdrop.targeting.models import TargetingResult
    from airdrop.telemetry.broker import TelemetryBroker
    from airdrop.video.buffer import AlignmentBuffer
    from airdrop.video.source import VideoFrame, VideoStats

LOGGER = logging.getLogger(__name__)


class Drainable(Protocol):
    """可排空的感知工作器接口。"""

    @property
    def running(self) -> bool: ...

    def drain(self, timeout: float = 120.0) -> bool: ...

    def start(self) -> object: ...

    def stop(self, timeout: float = 10.0) -> None: ...

    def raise_if_failed(self) -> None: ...


class TelemetryThread(Protocol):
    """飞控线程的任务运行接口。"""

    def connect(self) -> None: ...

    def stop(self, timeout: float = 5.0) -> None: ...


class VideoSource(Protocol):
    """实时视频源接口。"""

    running: bool
    stats: VideoStats

    def add_sink(self, sink: Callable[[VideoFrame], None]) -> None: ...

    def start(self) -> VideoSource: ...

    def stop(self, timeout: float = 5.0) -> None: ...

    def wait_ready(self, timeout: float = 10.0) -> bool: ...


class TaskRunner(Protocol):
    """状态机生命周期接口。"""

    @property
    def state(self) -> MissionState: ...

    @property
    def stats(self) -> MissionStats: ...

    @property
    def history(self) -> tuple[MissionTransition, ...]: ...

    @property
    def origin(self) -> LLARef | None: ...

    def run(self) -> MissionState: ...

    def stop(self, reason: str = "stopped") -> None: ...

    def close(self, *, timeout: float | None = None) -> None: ...


class Recorder(Protocol):
    """飞行记录器接口。"""

    flight_dir: Path | None

    @property
    def is_running(self) -> bool: ...

    def start(self, *, broker: TelemetryBroker, buffer: AlignmentBuffer) -> None: ...

    def stop(self, timeout: float = 5.0) -> None: ...


class DisplayPublisher(Protocol):
    """可独立降级的 QGC 视频发布器。"""

    @property
    def running(self) -> bool: ...

    def start(self) -> object: ...

    def stop(self, timeout: float | None = None) -> None: ...


class Targets(Protocol):
    """目标结果消费接口。"""

    def pump(self) -> int: ...

    @property
    def tracker(self) -> TargetTracker: ...

    def result(self) -> TargetingResult: ...

    def busy(self) -> bool: ...


@dataclass(slots=True)
class LiveComponents:
    """运行时组装完成的资源，允许测试替换外部输入和输出。"""

    thread: TelemetryThread
    source: VideoSource | None = None
    writer: Callable[[VideoFrame], None] | None = None
    worker: Drainable | None = None
    targets: Targets | None = None
    recorder: Recorder | None = None
    display: DisplayPublisher | None = None
    runner: TaskRunner | None = None
    controller: object | None = None
    broker: TelemetryBroker | None = None
    buffer: AlignmentBuffer | None = None
    on_start: Callable[[], int | None] | None = None
    on_finish: Callable[[], None] | None = None


class LiveSession:
    """按统一顺序启动、执行和清理一组运行组件。"""

    def __init__(self, options: RunOptions, components: LiveComponents) -> None:
        self.options = options.validated()
        self.components = components
        if not self.options.run_mission and components.runner is not None:
            raise ValueError("被动采集模式不得创建任务状态机")

    def run(self) -> int:
        """启动资源并返回任务状态码；清理中每一步都独立尝试。"""
        started = time.monotonic()
        status = 0
        try:
            self._start_components()
            status = self._run_mission() if self.options.run_mission else self._run_passive(started)
        except KeyboardInterrupt:
            status = 130
            LOGGER.info("操作手中断运行")
        except Exception:
            LOGGER.exception("任务运行失败")
            status = status or 1
        finally:
            status = self._cleanup(status)
        return status

    def _start_components(self) -> None:
        """按数据依赖顺序启动遥测、记录、感知和视频。"""
        c = self.components
        c.thread.connect()
        if c.recorder is not None:
            if c.broker is None or c.buffer is None:
                raise RuntimeError("启用记录时必须提供遥测代理和帧缓冲")
            c.recorder.start(broker=c.broker, buffer=c.buffer)
        if c.worker is not None:
            c.worker.start()
        if c.source is not None:
            if c.writer is not None:
                c.source.add_sink(c.writer)
            c.source.start()
            if not self.options.recognize:
                if not c.source.wait_ready(10.0):
                    raise RuntimeError("视频采集启动后未收到视频帧")
                if c.source.stats.frames < 1:
                    raise RuntimeError("视频采集未收到任何视频帧")
        if c.display is not None:
            display = c.display
            try:
                display.start()
                if not display.running:
                    raise RuntimeError("QGC 视频发布器启动后未运行")
            except Exception:
                LOGGER.exception("QGC 视频发布启动失败；继续任务并关闭显示")
                try:
                    display.stop()
                except Exception:
                    LOGGER.exception("QGC 视频发布器启动失败后的清理也失败")
                else:
                    c.display = None
        if c.on_start is not None:
            start_result = c.on_start()
            if start_result:
                raise RuntimeError(f"运行前置步骤失败，退出码 {start_result}")

    def _run_mission(self) -> int:
        """等待任务状态机到达明确终态。"""
        runner = self.components.runner
        if runner is None:
            raise RuntimeError("启用任务控制时必须提供任务状态机")
        return 0 if runner.run() is MissionState.DONE else 1

    def _run_passive(self, started: float) -> int:
        """被动运行期间持续检查感知线程，并按时长选项等待。"""
        deadline = None if self.options.max_seconds is None else started + self.options.max_seconds
        while deadline is None or time.monotonic() < deadline:
            if self.components.worker is not None:
                self.components.worker.raise_if_failed()
            source = self.components.source
            if source is not None and not source.running:
                raise RuntimeError("被动采集期间视频源意外停止")
            time.sleep(0.1)
        source = self.components.source
        if source is not None and source.stats.state != "streaming":
            raise RuntimeError(f"被动采集结束时视频源状态异常：{source.stats.state}")
        return 0

    def _cleanup(self, status: int) -> int:  # noqa: PLR0912 - 每个异步资源都需独立释放
        """逐项收尾；任何错误都不跳过后续资源，尤其是 MAVSDK 线程。"""
        c = self.components
        errors: list[BaseException] = []

        def attempt(label: str, callback: Callable[..., object] | None) -> None:
            if callback is None:
                return
            try:
                callback()
            except BaseException as exc:
                errors.append(exc)
                LOGGER.exception("清理步骤失败：%s", label)

        source_stopped = c.source is None
        if c.source is not None:
            try:
                _stop_source(c.source)
                source_stopped = True
            except BaseException as exc:
                errors.append(exc)
                LOGGER.exception("清理步骤失败：source.stop")
        if c.runner is not None:
            runner = c.runner
            if status:
                attempt("runner.stop", lambda: runner.stop("runtime_exit"))
            attempt("runner.close", runner.close)
        worker = c.worker
        if worker is not None:
            if source_stopped:
                attempt("worker.drain", lambda: _require_drain(worker))
            else:
                errors.append(RuntimeError("视频源未确认停止，跳过感知排空确认"))
            try:
                worker.stop()
                if worker.running:
                    raise RuntimeError("感知工作器 stop 后仍处于运行状态")
            except BaseException as exc:
                errors.append(exc)
                LOGGER.exception("清理步骤失败：worker.stop")
        attempt("targets.pump", None if c.targets is None else c.targets.pump)
        if c.display is not None:
            attempt("display.stop", c.display.stop)
        if c.recorder is not None:
            try:
                c.recorder.stop()
                if c.recorder.is_running:
                    raise RuntimeError("记录器 stop 后仍处于运行状态")
            except BaseException as exc:
                errors.append(exc)
                LOGGER.exception("清理步骤失败：recorder.stop")
        if c.on_finish is not None:
            attempt("on_finish", c.on_finish)
        attempt("thread.stop", c.thread.stop)
        if errors:
            LOGGER.error("运行收尾有 %d 项失败", len(errors))
            if status == 0 and any(isinstance(error, KeyboardInterrupt) for error in errors):
                return 130
            return status or 1
        return status


def _require_drain(worker: Drainable) -> None:
    if not worker.drain():
        raise RuntimeError("感知工作器未确认帧与尾请求已排空")


def _stop_source(source: VideoSource) -> None:
    source.stop()
    if source.running:
        raise RuntimeError("视频源 stop 后仍处于运行状态，不能确认尾帧已排空")
