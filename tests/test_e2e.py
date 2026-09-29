"""P10 端到端：**视频回放驱动全链路，不需要任何硬件**。

链路（与实飞同一条代码路径，只有"视频来源"和"检测器"是可替换的）::

    飞行目录 → ReplayVideoSource(原时间轴) → 帧-遥测对齐 → 环形缓冲
             → PerceptionWorker（逐帧）→ Detection（像素 + 编号 + 拍摄时刻遥测）
             → TargetTracker（georef：像素 → NED）→ targeting.analyze（聚类 + 选唯一）
             → MissionRunner（状态机 + 航线规划）→ 飞掠+降落任务

两种口径，都在这个文件里：

* **默认（离线）**：位姿与"检测结果"是合成的（脚本化假检测器给固定像素与编号），
  georef / targeting / planner / runner **全是真的**——验收计划里那句
  "回放全链路产出结果坐标"。
* **``-m realdata``（GPU）**：帧来自真实实战视频（用环境变量
  ``AIRDROP_REALDATA_VIDEO`` 指定）、YOLO 与 OCR 都是真的，
  只有位姿是合成的（那段素材没有遥测）。见文件末尾的说明。

为什么坐标断言能写死
--------------------
让飞机**水平且机头朝下看**（四元数为单位、地面水平），目标像素取**主点**：
主点对应的视线就是相机光轴，水平姿态下它正对地面——于是目标必然落在飞机正下方，
``(north, east, 0)``。默认相机外参 ``t_bc = 0``，所以这个期望值不需要任何标定数据。
"""

from __future__ import annotations

import json
import math
import os
import shutil
import time
import uuid
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from airdrop import (
    MAV_CMD_NAV_LAND,
    AlignmentBuffer,
    AlignmentWriter,
    Config,
    Detection,
    DetectionBatch,
    FrameTelemetryAligner,
    GroundConfig,
    MissionConfig,
    MissionRunner,
    MissionState,
    NedOrigin,
    OverflyConfig,
    PerceptionConfigLike,
    PerceptionTargetSource,
    PerceptionWorker,
    PixelBox,
    PreflightConfig,
    ReplayVideoSource,
    RoutesConfig,
    TargetingConfig,
    TargetTracker,
    TelemetrySnapshot,
    Waypoint,
    default_camera_model,
    load_broker_from_log,
)

# 素材与相机（两处口径共用）
WORK_ROOT = Path(__file__).resolve().parents[1] / ".e2e-test-tmp"
CAMERA = default_camera_model(320, 180)

#: 合成位姿：飞机在 NED (100, 200)、离地 50m、水平、机头朝北；原点海拔 500m
POSE_NORTH = 100.0
POSE_EAST = 200.0
POSE_DOWN = -50.0
ORIGIN_ALT = 500.0
GROUND_ALT = 500.0  # 地面点与原点的**高差为 0** ⇒ ground_z = 0


# ----------------------------------------------------------------------
# 素材构造（按 FlightRecorder 的落盘格式手写；不用 tmp_path，见 AGENTS）
# ----------------------------------------------------------------------
def _jpeg(image: np.ndarray) -> bytes:
    ok, buffer = cv2.imencode(".jpg", image)
    assert ok
    return buffer.tobytes()


def _blank_image(width: int = 320, height: int = 180) -> np.ndarray:
    return np.full((height, width, 3), 60, np.uint8)


def _pose_snapshot(timestamp: float) -> TelemetrySnapshot:
    """水平、朝下、位置固定在 (100, 200) 的合成遥测（四元数为单位四元数）。

    ``in_air=True``：状态机的 ``WAIT_AIRBORNE`` 那道门靠它放行（回放素材里没有
    相对高度，兜底的 ``relative_altitude_m`` 判据在这条素材上永远不满足）。
    """
    return TelemetrySnapshot(
        timestamp=timestamp,
        in_air=True,
        north_m=POSE_NORTH,
        east_m=POSE_EAST,
        down_m=POSE_DOWN,
        vx_m_s=0.0,
        vy_m_s=0.0,
        vz_m_s=0.0,
        roll_deg=0.0,
        pitch_deg=0.0,
        yaw_deg=0.0,
        quaternion_w=1.0,
        quaternion_x=0.0,
        quaternion_y=0.0,
        quaternion_z=0.0,
        origin_latitude_deg=47.0,
        origin_longitude_deg=8.0,
        origin_altitude_m=ORIGIN_ALT,
    )


def _write_flight(
    root: Path,
    *,
    frames: Sequence[np.ndarray] | None = None,
    frame_dt: float = 0.2,
    tlm_dt: float = 0.1,
    t0: float | None = None,
) -> Path:
    """手写一个飞行目录：帧索引 + jpeg + telemetry.jsonl。

    ``t0`` 默认取"现在往前 2 秒"——遥测时间戳与墙钟同一量纲，
    这样 MissionRunner 的链路看门狗（默认 5s 陈旧）在测试里不会误触发。
    """
    images = list(frames) if frames else [_blank_image() for _ in range(4)]
    flight_dir = WORK_ROOT / uuid.uuid4().hex[:12]
    (flight_dir / "frames").mkdir(parents=True)
    start = (time.time() - 2.0) if t0 is None else float(t0)
    lag = 0.15

    index_lines = []
    for position, image in enumerate(images, start=1):
        capture = start + (position - 1) * frame_dt
        filename = f"{position:06d}.jpg"
        payload = _jpeg(image)
        (flight_dir / "frames" / filename).write_bytes(payload)
        index_lines.append(
            json.dumps(
                {
                    "index": position,
                    "filename": filename,
                    "capture_timestamp": capture,
                    "received_timestamp": capture + lag,
                    "lag": lag,
                    "extrapolated": False,
                    "offset": 0.0,
                    "bytes": len(payload),
                }
            )
        )
    (flight_dir / "frames_index.jsonl").write_text("\n".join(index_lines) + "\n", encoding="utf-8")

    span = (len(images) - 1) * frame_dt + 0.5
    stamps = [start - 0.2 + k * tlm_dt for k in range(int(span / tlm_dt) + 1)]
    (flight_dir / "telemetry.jsonl").write_text(
        "\n".join(json.dumps(_pose_snapshot(ts).as_dict()) for ts in stamps) + "\n",
        encoding="utf-8",
    )
    return flight_dir


@pytest.fixture
def workdir() -> Iterator[Path]:
    """工作区内的临时目录；用例结束整棵删掉（不用 ``tmp_path``：见 AGENTS）。"""
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    path = WORK_ROOT / uuid.uuid4().hex[:8]
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


# ----------------------------------------------------------------------
# 假检测器（替掉 YOLO/OCR，其余全是真的）
# ----------------------------------------------------------------------
class ScriptedDetector:
    """按帧号发固定检测结果的假检测器（clsl2 模式，不涉及 OCR 进程池）。

    ``fov`` 是"目标出现在画面里的像素"；默认取主点——见模块 docstring 里
    为什么这样取的坐标可以写死。
    """

    def __init__(
        self,
        codes: dict[int, int | None],
        *,
        pixel: tuple[float, float] | None = None,
        side_px: float = 60.0,
        confidence: float = 0.9,
    ) -> None:
        self.codes = dict(codes)
        self.pixel = pixel
        self.side_px = side_px
        self.confidence = confidence
        self.frames = 0

    def load(self) -> "ScriptedDetector":
        """``PerceptionWorker.start()`` 会调它（真检测器在这里加载权重）。"""
        return self

    def detect(
        self,
        image: Any,
        *,
        frame_index: int = 0,
        capture_timestamp: float = 0.0,
        telemetry: TelemetrySnapshot | None = None,
        mode: str = "",
        undistort: bool | None = None,
    ) -> DetectionBatch:
        self.frames += 1
        code = self.codes.get(frame_index)
        if code is None and frame_index not in self.codes:
            return DetectionBatch(image=image, detections=(), undistorted=True)
        center = self.pixel or CAMERA.principal_point()
        box = PixelBox(center[0] - 30, center[1] - 30, center[0] + 30, center[1] + 30)
        detection = Detection(
            frame_index=frame_index,
            capture_timestamp=capture_timestamp,
            pixel=box.center,
            box=box,
            confidence=self.confidence,
            telemetry=telemetry if telemetry is not None else TelemetrySnapshot(),
            code=code,
            side_px=self.side_px,
            mode=mode,
            extra={"undistorted": True, "class_name": "target"},
        )
        return DetectionBatch(image=image, detections=(detection,), undistorted=True)


# ----------------------------------------------------------------------
# 接线：回放 → 缓冲 → 感知 → 目标点
# ----------------------------------------------------------------------
class Chain:
    """跑完一次"回放 → 缓冲区"，把感知工作线程留在手里。"""

    def __init__(
        self,
        flight_dir: Path,
        detector: Any,
        *,
        config: Config,
        buffer_frames: int = 60,
    ) -> None:
        self.flight_dir = flight_dir
        self.config = config
        self.broker, pacer = load_broker_from_log(flight_dir)
        self.source = ReplayVideoSource(flight_dir, speed=0.0, telemetry=pacer)
        self.aligner = FrameTelemetryAligner(self.broker, max_wait=1.0)
        self.buffer = AlignmentBuffer(buffer_frames, storage="jpeg")
        self.writer = AlignmentWriter(self.buffer, self.aligner)
        self.worker = PerceptionWorker(
            PerceptionConfigLike(mode="cls12", ocr_dedupe_s=0.0, ocr_dedupe_px=0.0),
            buffer=self.buffer,
            detector=detector,
        )
        self.detector = detector
        self.tracker = TargetTracker(config, camera=CAMERA)
        self.source_obj = PerceptionTargetSource(worker=self.worker, tracker=self.tracker)

    # -- 生命周期 ------------------------------------------------------
    def play(self, *, timeout: float = 30.0) -> int:
        """回放到底并等感知把缓冲追平；返回处理过的帧数。"""
        self.worker.start()
        self.source.add_sink(self.writer)
        self.source.start()
        thread = self.source._thread
        assert thread is not None
        thread.join(timeout)
        assert not thread.is_alive(), "回放线程没在超时内结束"
        expected = len(self.source.frames)
        deadline = time.monotonic() + timeout
        while self.worker.stats.frames < expected and time.monotonic() < deadline:
            time.sleep(0.02)
        frames = self.worker.stats.frames
        self.worker.stop()
        assert frames >= expected, f"感知只处理了 {frames}/{expected} 帧"
        return frames

    def consumed(self) -> int:
        return self.source_obj.pump()


def _config(**overrides: Any) -> Config:
    routes = RoutesConfig(
        recon_route=(Waypoint(lat=47.0, lon=8.0, alt_m=60.0),),
        backup_point=Waypoint(lat=47.01, lon=8.01, alt_m=0.0),
        landing_route=(Waypoint(lat=47.0, lon=8.02, alt_m=0.0),),
    )
    params: dict[str, Any] = {
        "routes": routes,
        "overfly": OverflyConfig(heading_deg=90.0, altitude_m=20.0, leg_length_m=200.0),
        # 侦查航线由测试上传（auto）；正式任务默认 operator，由操作手在 QGC 上传并启动
        "mission": MissionConfig(tick_hz=20.0, recon_upload="auto"),
        "ground": GroundConfig(ground_point_alt=GROUND_ALT),
        "targeting": TargetingConfig(eps_m=0.75, min_samples=2),
        # 回放链路里没有相机/模型：起飞前自检的**四项检查全关**（关掉≠通过，只是离线跑通）
        "preflight": PreflightConfig(
            load_detector=False,
            load_ocr=False,
            load_camera=False,
            check_video=False,
        ),
    }
    params.update(overrides)
    return Config(**params).validated()


class StubController:
    """满足 :class:`~airdrop.telemetry.MissionController` 协议的最小假控制器。"""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.uploads: list[tuple[Any, ...]] = []
        self.finished = False

    def upload_mission(self, items: Sequence[Any], /) -> int:
        self.calls.append("upload_mission")
        self.uploads.append(tuple(items))
        return len(items)

    def start_mission(self) -> None:
        self.calls.append("start_mission")
        self.finished = False

    def in_mission_mode(self) -> bool:
        self.calls.append("in_mission_mode")
        return True

    def hold(self) -> None:
        self.calls.append("hold")

    def rtl(self) -> None:
        self.calls.append("rtl")

    def gripper_release(self) -> bool:
        self.calls.append("gripper_release")
        return True

    def request_origin(self) -> NedOrigin | None:
        self.calls.append("request_origin")
        return NedOrigin(lat_deg=47.0, lon_deg=8.0, alt_m=ORIGIN_ALT)

    def mission_finished(self) -> bool:
        self.calls.append("mission_finished")
        return self.finished


class StepClock:
    """每拍推进固定步长的假时钟（让测试里的时间完全可控）。"""

    def __init__(self, start: float | None = None, step: float = 0.05) -> None:
        self.now = time.time() if start is None else float(start)
        self.step = float(step)

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += self.step


# ----------------------------------------------------------------------
# 离线全链路
# ----------------------------------------------------------------------
def _advance(runner: MissionRunner, controller: "StubController", clock: StepClock) -> None:
    """推到 LAND（或 DONE）：先让看门狗看到一次"任务在跑"，再报侦查航线飞完。"""
    runner.update()  # INIT → RECON
    runner.update()  # RECON：``mission_finished()`` 先回一次 False
    controller.finished = True  # 侦查航线飞完
    for _ in range(60):
        runner.update()
        clock.sleep(0.05)
        if runner.state in (MissionState.LAND, MissionState.DONE):
            return


def test_replay_drives_full_chain_to_target_coordinates(workdir: Path) -> None:
    """**验收用例**：回放素材 → 结果坐标；跨帧投票把一帧错读压掉。"""
    flight_dir = _write_flight(workdir)
    config = _config()
    # 4 帧看同一个目标：3 帧读 56、1 帧读 12（一颗"高置信度的错读"）
    detector = ScriptedDetector({1: 56, 2: 56, 3: 56, 4: 12}, confidence=0.9)
    chain = Chain(flight_dir, detector, config=config)

    frames = chain.play()
    assert frames == 4
    written, skipped = chain.writer.stats()
    assert (written, skipped) == (4, 0), "回放的每一帧都要对齐入缓冲（一帧不落）"

    assert chain.consumed() == 4
    assert chain.tracker.stats["no_fix"] == 0

    result = chain.source_obj.result()
    assert result.ok, "4 次观测应当聚成一个候选类"
    assert result.code == 56, "类内编号众数：3 票 56 压过 1 票 12"
    assert result.north_m == pytest.approx(POSE_NORTH, abs=0.05)
    assert result.east_m == pytest.approx(POSE_EAST, abs=0.05)
    assert result.ned is not None and result.ned[2] == 0.0, "目标是地面上的点"

    # 结果坐标 → 飞掠航线 → 上传（状态机与航线规划也是真的）
    controller = StubController()
    clock = StepClock()
    events: list[tuple[str, dict[str, Any]]] = []
    runner = MissionRunner(
        config,
        controller,
        chain.broker,
        target_result=chain.source_obj.result,
        target_busy=chain.source_obj.busy,
        release_judge=None,  # 这一条验的是坐标链路，投放判据由 P8/P9 各自覆盖
        on_event=lambda kind, data: events.append((kind, data)),
        clock=clock,
        sleep=clock.sleep,
    )
    _advance(runner, controller, clock)
    assert runner.state is MissionState.LAND, runner.history[-1].reason
    # 正式流程的前两跳（自检 → 等在空中）也要走：INIT 不再直接跳到侦查
    assert [str(record.to_state) for record in runner.history] == [
        "PREFLIGHT",
        "WAIT_AIRBORNE",
        "RECON",
        "HOLD_PROCESS",
        "OVERFLY",
        "LAND",
    ]

    plan = runner.plan
    assert plan is not None and plan.source == "target"
    assert plan.target_ned[0] == pytest.approx(POSE_NORTH, abs=0.05)
    assert plan.target_ned[1] == pytest.approx(POSE_EAST, abs=0.05)
    assert len(controller.uploads) == 2, "先侦查航线、再飞掠+降落"
    assert [item.command for item in controller.uploads[1]][-1] == MAV_CMD_NAV_LAND

    targeting_event = [data for kind, data in events if kind == "targeting"]
    assert targeting_event and targeting_event[0]["targeting"]["selected"]["code"] == 56


def test_chain_reports_no_result_when_target_is_never_labelled(workdir: Path) -> None:
    """一帧编号都没读出来时 → targeting 不给结果，状态机走**备用点**分支。"""
    flight_dir = _write_flight(workdir)
    chain = Chain(
        flight_dir, ScriptedDetector({1: None, 2: None, 3: None, 4: None}), config=_config()
    )
    chain.play()
    chain.consumed()

    # 观测点本身有效（看到目标了），只是没有编号
    assert len(chain.tracker.points) == 4
    assert chain.tracker.stats["labeled"] == 0
    assert chain.source_obj.result().ok is False

    controller = StubController()
    clock = StepClock()
    runner = MissionRunner(
        _config(),
        controller,
        chain.broker,
        target_result=chain.source_obj.result,
        target_busy=lambda: False,
        release_judge=None,
        clock=clock,
        sleep=clock.sleep,
    )
    runner.update()  # INIT → RECON
    runner.update()  # 先看到一次"任务在跑"
    controller.finished = True
    for _ in range(80):
        runner.update()
        clock.sleep(0.05)
        if runner.plan is not None:
            break
    assert runner.plan is not None and runner.plan.source == "backup", "无编号 → 备用点"


def test_tracker_skips_points_without_attitude_and_without_position() -> None:
    """遥测缺姿态/缺位置时**不猜**：计数并跳过（这些点会污染聚类）。"""
    config = _config()
    tracker = TargetTracker(config, camera=CAMERA)
    good = _pose_snapshot(time.time())

    euler_only = _pose_snapshot(time.time())  # 只有欧拉角（四元数缺失）→ 用兜底路径
    for field in ("quaternion_w", "quaternion_x", "quaternion_y", "quaternion_z"):
        setattr(euler_only, field, None)
    no_attitude = _pose_snapshot(time.time())  # 两种姿态都没有 → 无法解算
    for field in (
        "quaternion_w",
        "quaternion_x",
        "quaternion_y",
        "quaternion_z",
        "roll_deg",
        "pitch_deg",
        "yaw_deg",
    ):
        setattr(no_attitude, field, None)
    no_position = _pose_snapshot(time.time())
    no_position.north_m = None

    def _detection(snapshot: TelemetrySnapshot) -> Detection:
        center = CAMERA.principal_point()
        box = PixelBox(center[0] - 10, center[1] - 10, center[0] + 10, center[1] + 10)
        return Detection(
            frame_index=1,
            capture_timestamp=snapshot.timestamp,
            pixel=box.center,
            box=box,
            confidence=0.9,
            telemetry=snapshot,
            code=56,
            side_px=0.0,
            extra={"undistorted": True},
        )

    assert tracker.add(_detection(no_attitude)) is None
    assert tracker.add(_detection(no_position)) is None
    point = tracker.add(_detection(good))
    assert point is not None and point.code == 56
    # 只有欧拉角时走兜底路径，结果与四元数一致（水平姿态下两者本就等价）
    fallback = tracker.add(_detection(euler_only))
    assert fallback is not None
    assert fallback.north_m == pytest.approx(point.north_m, abs=1e-9)
    assert fallback.east_m == pytest.approx(point.east_m, abs=1e-9)
    assert tracker.stats == {
        "points": 2,
        "labeled": 2,
        "no_fix": 1,
        "attitude_missing": 1,
        "side_mismatch": 0,
    }


def test_tracker_side_check_is_off_by_default() -> None:
    """边长互校**默认关掉**（正式流程不刷日志）；回放优化时才显式给门限。

    同一份"边长与求交深度明显不符"的观测：默认既不计 ``side_mismatch`` 也不发事件；
    ``side_check_tolerance>0`` 时才作诊断（仍然只记不剔点）。
    """
    capture = 1000.0
    snapshot = _pose_snapshot(capture)
    box = PixelBox(140.0, 80.0, 180.0, 120.0)
    detection = Detection(
        frame_index=1,
        capture_timestamp=capture,
        pixel=box.center,
        box=box,
        confidence=0.9,
        telemetry=snapshot,
        code=56,
        # 20 m 处的 1 m 目标在主点附近约 14 px（fx≈277），100 px 差一个数量级 → 必不通过
        side_px=100.0,
        extra={"undistorted": True},
    )

    events: list[str] = []
    default = TargetTracker(
        _config(), camera=CAMERA, on_event=lambda kind, _data: events.append(kind)
    )
    assert default.side_check_tolerance == 0.0, "默认关掉：连计数都不做"
    assert default.add(detection) is not None
    assert default.stats["side_mismatch"] == 0
    assert "side_check" not in events

    enabled = TargetTracker(
        _config(),
        camera=CAMERA,
        side_check_tolerance=0.25,
        on_event=lambda kind, _data: events.append(kind),
    )
    assert enabled.add(detection) is not None, "打开诊断也仍然不剔点"
    assert enabled.stats["side_mismatch"] == 1
    assert "side_check" in events


def test_target_source_busy_follows_perception_backlog() -> None:
    """``busy()``：缓冲没处理完或 OCR 没回来就算"还在干活"。"""

    class _Stats:
        def __init__(self, submitted: int, results: int) -> None:
            self.submitted = submitted
            self.results = results

    class _Worker:
        def __init__(self, lag: int, submitted: int, results: int) -> None:
            self._lag = lag
            self._stats = _Stats(submitted, results)
            self.drained = 0

        @property
        def lag_frames(self) -> int:
            return self._lag

        @property
        def stats(self) -> Any:
            return self._stats

        def drain_results(self) -> list[Any]:
            self.drained += 1
            return []

    tracker = TargetTracker(_config(), camera=CAMERA)
    busy = PerceptionTargetSource(worker=_Worker(3, 0, 0), tracker=tracker)  # type: ignore[arg-type]
    assert busy.busy() is True
    idle = PerceptionTargetSource(worker=_Worker(0, 0, 0), tracker=tracker)  # type: ignore[arg-type]
    assert idle.busy() is False
    ocr_pending = PerceptionTargetSource(worker=_Worker(0, 5, 2), tracker=tracker)  # type: ignore[arg-type]
    assert ocr_pending.busy() is True, "送出去 5 个 OCR 只回来 2 个 → 还没干完"
    # result() 顺手抽干队列（不需要调用方额外 pump）
    assert ocr_pending.result() is not None
    assert ocr_pending.worker.drained == 1  # type: ignore[attr-defined]


# ----------------------------------------------------------------------
# 真实素材全链路（-m realdata）
# ----------------------------------------------------------------------
#: 真实素材视频的环境变量名
REALDATA_VIDEO_ENV = "AIRDROP_REALDATA_VIDEO"


def _realdata_video() -> Path | None:
    """从环境变量读视频路径；没设或文件不存在时返回 None（用例跳过）。"""
    raw = os.environ.get(REALDATA_VIDEO_ENV, "").strip()
    if not raw:
        return None
    path = Path(raw).expanduser()
    return path if path.is_file() else None


VIDEO = _realdata_video()
WEIGHTS = Path("models/best2.pt")
MODELS_DIR = Path("models/ppocr")
#: 目标段里连续几帧（画面里都在目标附近）
REAL_FRAMES = (1840, 1841, 1842, 1843)


class RealYoloOcrDetector:
    """真 YOLO + 真 OCR 的"检测器"，但 **OCR 在进程内跑**。

    ⚠ 有意为之的取舍：``PerceptionWorker`` 在 ``ocr`` 模式下会拉起**独立 OCR 进程池**，
    而受限沙箱里子进程管道不可用（Windows 上 multiprocessing 走命名管道）。所以这里让
    检测器在进程内把"检测 + 裁剪 + OCR"一次做完，再由 pipeline 以 ``cls12`` 模式直出——
    **进程池那条路**由 ``tests/test_perception.py`` 的假池用例覆盖，**OCR 精度**由
    ``tests/test_perception_realdata.py`` 覆盖；这一条验的是"真实像素能不能一路走到坐标"。
    """

    def __init__(self) -> None:
        from airdrop.perception.cropproc import OcrEngineConfig, OpenCvPostProcess
        from airdrop.perception.detector import Detector, DetectorConfig

        self._detector = Detector(DetectorConfig(model_path=str(WEIGHTS), device="0"))
        self._post = OpenCvPostProcess(
            color="blue", ocr_conf_threshold=0.6, engine_config=OcrEngineConfig()
        )
        self.codes: list[int | None] = []

    def load(self) -> "RealYoloOcrDetector":
        self._detector.load()
        return self

    def detect(
        self,
        image: Any,
        *,
        frame_index: int = 0,
        capture_timestamp: float = 0.0,
        telemetry: TelemetrySnapshot | None = None,
        mode: str = "",
        undistort: bool | None = None,
    ) -> DetectionBatch:
        batch = self._detector.detect(
            image,
            frame_index=frame_index,
            capture_timestamp=capture_timestamp,
            telemetry=telemetry,
            mode="ocr",
        )
        detections = []
        for detection in batch.detections:
            crop = self._detector.crop(batch, detection)
            number = self._post.recognize(crop).number if crop.size else None
            self.codes.append(number)
            detections.append(
                Detection(
                    frame_index=detection.frame_index,
                    capture_timestamp=detection.capture_timestamp,
                    pixel=detection.pixel,
                    box=detection.box,
                    confidence=detection.confidence,
                    telemetry=detection.telemetry,
                    code=number,
                    side_px=float(max(detection.box.width, detection.box.height)),
                    mode="ocr",
                    extra={"undistorted": False, "class_name": "target"},
                )
            )
        return DetectionBatch(image=batch.image, detections=tuple(detections), undistorted=False)


@pytest.mark.realdata
def test_real_video_replay_produces_target_coordinates(workdir: Path) -> None:
    """实战视频帧 → YOLO → OCR → georef → 聚类 → 结果坐标。

    **位姿是合成的**（那段素材只有画面没有遥测）：飞机水平、离地 50m、朝下看。
    所以这条用例验的是**链路连通与量纲**（真像素能一路走到地面坐标、编号能读对），
    **不是**坐标精度——精度要靠带遥测的真实素材或标定飞行。
    """
    pytest.importorskip("ultralytics")
    pytest.importorskip("rapidocr")
    if VIDEO is None or not WEIGHTS.is_file():
        pytest.skip(f"未用 {REALDATA_VIDEO_ENV} 指定素材视频，或缺少 YOLO 权重")
    if os.environ.get("AIRDROP_SKIP_GPU"):
        pytest.skip("设了 AIRDROP_SKIP_GPU")

    capture = cv2.VideoCapture(str(VIDEO))
    assert capture.isOpened()
    frames = []
    try:
        for index in REAL_FRAMES:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            assert ok, f"读不到第 {index} 帧"
            frames.append(frame)
    finally:
        capture.release()

    flight_dir = _write_flight(workdir, frames=frames)
    detector = RealYoloOcrDetector()
    # 连续帧之间目标像素会飘，坐标跟着飘（合成位姿没补偿）：聚类半径放宽到 5m，
    # 只为把"同一目标的连续观测"聚到一起——这不是精度验收，别拿它当 eps 的依据。
    config = _config(targeting=TargetingConfig(eps_m=5.0, min_samples=2))
    chain = Chain(flight_dir, detector, config=config)

    assert chain.play() == len(REAL_FRAMES)
    added = chain.consumed()
    assert added >= 1, f"一个目标点都没解算出来（OCR 编号：{detector.codes}）"
    assert 56 in detector.codes, f"目标段应当读出 56，实际 {detector.codes}"

    labeled = [point for point in chain.tracker.points if point.code is not None]
    assert labeled and all(point.code == 56 for point in labeled)
    for point in labeled:
        assert math.isfinite(point.north_m) and math.isfinite(point.east_m)
        # 水平朝下看、离地 50m：目标应当落在飞机附近几十米内（量纲检查）
        assert math.hypot(point.north_m - POSE_NORTH, point.east_m - POSE_EAST) < 200.0

    result = chain.source_obj.result()
    assert result.ok, f"{len(labeled)} 个观测没聚成结果（点：{chain.tracker.stats}）"
    assert result.code == 56
