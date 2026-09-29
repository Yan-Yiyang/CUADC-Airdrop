"""docs/handbook.md §3 的示例逐条真跑：文档里的示例必须真能运行。

为什么单独一个文件
------------------
手册 §3 每个模块都给了一段"最小可运行示例"（离线、不碰硬件）。文档里的代码最容易腐烂：
改了 API、改了默认值，示例照旧印在纸上，读者照着抄就报错。这个文件把那些示例逐条搬成
用例——它们同时也是跨模块的装配冒烟测试测试测试：遥测 → 对齐 → 缓冲 → 解算 → 统计 → 弹道 → 任务 →
记录回放，各跑一遍（细节由各自的 test_*.py 覆盖，这里只保证"按文档抄能跑通"）。

约定
----
* 用例名与被验证的小节一一对应，改了手册那一段就改这里的同名函数；
* 需要临时目录的用例使用本文件的 workdir fixture（工作区内建目录、用完即删）；
* 全部离线：不需要飞控、视频、GPU。
"""

from __future__ import annotations

import json
import shutil
import time
import uuid
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from airdrop import (
    AlignmentBuffer,
    AlignmentWriter,
    BallisticsModel,
    Config,
    Detection,
    DetectionBatch,
    DropConfig,
    DropRecord,
    DropSample,
    FrameTelemetryAligner,
    GroundConfig,
    LLARef,
    MissionConfig,
    MissionRunner,
    NedOrigin,
    OverflyConfig,
    PixelBox,
    PreflightConfig,
    RoutesConfig,
    TargetingConfig,
    TelemetryBroker,
    TelemetrySnapshot,
    VideoFrame,
    Waypoint,
    analyze,
    build_drop_mission,
    build_recon_mission,
    command_name,
    cross_check_by_side,
    default_camera_model,
    fit_ballistics,
    pixel_to_ned,
)
from airdrop.perception import OpenCvPostProcess, PerceptionConfigLike, PerceptionWorker

WORK_ROOT = Path(__file__).resolve().parents[1] / ".handbook-test-tmp"
CAMERA = default_camera_model(320, 180)


@pytest.fixture
def workdir() -> Iterator[Path]:
    """工作区内的临时目录；用例结束整棵删掉（不用 tmp_path：见模块 docstring）。"""
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    path = WORK_ROOT / uuid.uuid4().hex[:8]
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


# ----------------------------------------------------------------------
def ex_telemetry() -> None:
    """遥测：传入 MAVSDK 风格的更新，按时间查询（内插）。"""
    broker = TelemetryBroker(history_maxlen=1200, history_interval=0.0)
    position = SimpleNamespace(north_m=100.0, east_m=200.0, down_m=-50.0)
    velocity = SimpleNamespace(north_m_s=15.0, east_m_s=0.0, down_m_s=0.0)
    snapshot = broker.update_local_position_velocity(position, velocity)
    print("  最新快照:", snapshot.north_m, snapshot.east_m, snapshot.vx_m_s)
    found = broker.get_snapshot_at(snapshot.timestamp, mode="nearest")
    print("  按时间查询:", None if found is None else found.north_m)
    print("  历史区间:", broker.history_span())


# ----------------------------------------------------------------------
def ex_align_and_buffer() -> None:
    """对齐 + 缓冲：帧 → 拍摄时刻遥测 → 环形缓冲。"""
    broker = TelemetryBroker()
    broker.update_local_position_velocity(
        SimpleNamespace(north_m=100.0, east_m=200.0, down_m=-50.0),
        SimpleNamespace(north_m_s=15.0, east_m_s=0.0, down_m_s=0.0),
    )
    aligner = FrameTelemetryAligner(broker, max_wait=0.5)
    buffer = AlignmentBuffer(capacity=8, storage="jpeg")
    writer = AlignmentWriter(buffer, aligner)

    image = np.zeros((180, 320, 3), dtype=np.uint8)
    # 帧的"收到时间" = 拍摄时刻 + 链路延时，于是 capture = timestamp - lag 正好落在遥测上
    writer(
        VideoFrame(index=1, image=image, timestamp=broker.get_snapshot().timestamp + 0.15, lag=0.15)
    )
    record = buffer.latest()
    assert record is not None, "帧没有进缓冲（遥测没覆盖拍摄时刻？）"
    print("  入缓冲 #%d，拍摄时刻遥测 north=%s" % (record.index, record.snapshot.north_m))
    print("  sink 统计 (written, skipped):", writer.stats())


# ----------------------------------------------------------------------
def ex_georef() -> None:
    """坐标解算：水平朝下看 + 主点像素 ⇒ 目标就在飞机正下方。"""
    result = pixel_to_ned(
        CAMERA.principal_point(),
        camera=CAMERA,
        ground_z=0.0,
        position_ned=(100.0, 200.0, -50.0),
        quaternion=(1.0, 0.0, 0.0, 0.0),
    )
    print("  地面交点 NED:", result.ned, "深度 %.1f m" % result.depth_m)
    # 边长法独立估深度：depth ≈ f·S/(side_px·cos_tilt)；取与 50m 自洽的像素边长
    side_px = float(CAMERA.camera_matrix[0, 0]) * 1.0 / result.depth_m
    check = cross_check_by_side(
        side_px=side_px,
        side_m=1.0,
        depth_by_intersection_m=result.depth_m,
        camera=CAMERA,
        quaternion=(1.0, 0.0, 0.0, 0.0),
    )
    print(
        "  边长法交叉验证 ok=%s 相对误差 %.4f（side_px=%.2f）"
        % (check.ok, check.relative_error, side_px)
    )


# ----------------------------------------------------------------------
def ex_targeting() -> None:
    """目标统计：DBSCAN 聚类 + 类内编号众数。"""
    from airdrop import TargetPoint

    points = [
        TargetPoint(north_m=10.0, east_m=20.0, capture_timestamp=1.0, frame_index=1, code=56),
        TargetPoint(north_m=10.2, east_m=20.1, capture_timestamp=1.1, frame_index=2, code=56),
        TargetPoint(north_m=10.1, east_m=19.9, capture_timestamp=1.2, frame_index=3, code=12),
        TargetPoint(north_m=80.0, east_m=90.0, capture_timestamp=1.3, frame_index=4, code=7),
    ]
    result = analyze(points, TargetingConfig(eps_m=0.75, min_samples=2))
    print("  选中编号 %s @ NED (%.2f, %.2f)" % (result.code, result.north_m, result.east_m))
    print(
        "  候选类 %d 个，噪声 %d，剔除 %d"
        % (len(result.clusters), len(result.noise), len(result.rejected))
    )


# ----------------------------------------------------------------------
def ex_ballistics() -> None:
    """弹道 + 投放判据：预测落点、越过目标后强制投放。"""
    model = BallisticsModel(Config().ballistics)
    impact = model.predict_impact((0.0, 0.0, -20.0), (18.0, 0.0, 0.0), ground_z=0.0)
    print("  落点 %s，飞行时间 %.2fs" % (impact.ned, impact.flight_time_s))

    from airdrop import ReleaseJudge

    judge = ReleaseJudge(DropConfig(radius_m=2.0), model, overfly_heading_deg=90.0)
    snapshot = TelemetrySnapshot(timestamp=time.time())
    snapshot.north_m, snapshot.east_m, snapshot.down_m = 0.0, -35.0, -20.0
    snapshot.vx_m_s, snapshot.vy_m_s, snapshot.vz_m_s = 0.0, 18.0, 0.0
    decision = judge.update(snapshot, (0.0, 0.0, 0.0), now=snapshot.timestamp)
    print(
        "  判据: should_release=%s reason=%s 误差 %.2f m"
        % (decision.should_release, decision.reason, decision.horizontal_error_m or -1.0)
    )


# ----------------------------------------------------------------------
def _mission_config() -> Config:
    """手册 §3.8.4 里那段 config = Config(...)（两处要一起改：示例由本文件真跑）。"""
    return Config(
        routes=RoutesConfig(
            recon_route=(
                Waypoint(lat=47.0, lon=8.0, alt_m=60.0),
                Waypoint(lat=47.001, lon=8.0, alt_m=60.0),
            ),
            backup_point=Waypoint(lat=47.05, lon=8.05, alt_m=0.0),
            landing_route=(Waypoint(lat=46.999, lon=8.01, alt_m=0.0),),
        ),
        overfly=OverflyConfig(heading_deg=90.0, altitude_m=20.0, leg_length_m=200.0),
        ground=GroundConfig(ground_point_alt=500.0),
        # 离线示例：侦察航线由示例自己上传（auto）；正式任务用默认的 operator（操作手在 QGC 启动）
        mission=MissionConfig(recon_upload="auto"),
        # 自检四项全关（示例里没有相机/模型）：关掉 ≠ 通过，只是离线跑通
        preflight=PreflightConfig(
            load_detector=False,
            load_ocr=False,
            load_camera=False,
            check_video=False,
        ),
    ).validated()


def ex_mission_plan() -> None:
    """航线规划（纯函数）：侦察任务项 + 飞掠/降落合并任务。"""
    config = _mission_config()
    items = build_recon_mission(config)
    print("  侦察任务项 %d 个，首项命令 %s" % (len(items), command_name(items[0].command)))
    origin = LLARef(lon_deg=8.0, lat_deg=47.0, alt_m=500.0)
    plan = build_drop_mission(config, origin=origin, target_ned=(300.0, 0.0, 0.0))
    print(
        "  投放任务：来源 %s，任务项 %d 个，目标 NED %s"
        % (plan.source, len(plan.items), plan.target_ned)
    )
    print("  飞掠 entry/exit: %.6f / %.6f" % (plan.entry.lat, plan.exit.lat))


# ----------------------------------------------------------------------
class _FakeClock:
    def __init__(self) -> None:
        self.now = time.time()

    def __call__(self) -> float:
        return self.now


class _FakeController:
    """按 MissionController 协议实现的最小假控制器（离线跑状态机）。"""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.finished = False

    def upload_mission(self, items, /):
        self.calls.append("upload_mission")
        return len(items)

    def start_mission(self) -> None:
        self.calls.append("start_mission")
        self.finished = False

    def in_mission_mode(self) -> bool:
        return True

    def hold(self) -> None:
        self.calls.append("hold")

    def rtl(self) -> None:
        self.calls.append("rtl")

    def gripper_release(self) -> bool:
        self.calls.append("gripper_release")
        return True

    def request_origin(self):
        return NedOrigin(lat_deg=47.0, lon_deg=8.0, alt_m=500.0)

    def mission_finished(self) -> bool:
        return self.finished


def ex_mission_runner() -> None:
    """主循环：假控制器 + 假时钟，分钟级任务在毫秒级跑完。"""
    config = _mission_config()
    clock = _FakeClock()
    controller = _FakeController()
    broker = TelemetryBroker()
    broker.update_local_position_velocity(
        SimpleNamespace(north_m=0.0, east_m=-200.0, down_m=-40.0),
        SimpleNamespace(north_m_s=0.0, east_m_s=18.0, down_m_s=0.0),
    )
    broker.update_in_air(True)  # 假飞控报告"已离地"（WAIT_AIRBORNE 那道门读它）
    runner = MissionRunner(config, controller, broker, clock=clock, sleep=lambda _s: None)
    for _ in range(4):
        runner.update()
        clock.now += 0.05
    controller.finished = True
    runner.update()
    print(
        "  状态 %s，上传 %d 次，历史 %d 条"
        % (runner.state, runner.stats.uploads, len(runner.history))
    )
    print(
        "  最后一次转移: %s → %s (%s)"
        % (runner.history[-1].from_state, runner.history[-1].to_state, runner.history[-1].reason)
    )


# ----------------------------------------------------------------------
def ex_drops_fit() -> None:
    """投放记录 + 反演：合成两条样本，反演出 Cd。"""
    from dataclasses import replace

    truth = replace(Config().ballistics, drag_coefficient=0.8)  # 真值 Cd=0.8
    true_model = BallisticsModel(truth)
    records = [
        DropRecord(
            index=i,
            timestamp=1000.0 + i,
            position_ned=(100.0, 200.0, -height),
            velocity_ned=(speed, 0.0, -1.0),
            ground_z=0.0,
            delay_s=0.05,
            ballistics=Config().ballistics,  # 记录里存的是当时的占位参数
        )
        for i, (height, speed) in enumerate([(25.0, 12.0), (45.0, 20.0)], start=1)
    ]
    from airdrop import predict_record_impact

    samples = []
    for record in records:
        impact = predict_record_impact(record, true_model, delay_s=0.05)
        assert impact.ok and impact.ned is not None
        samples.append(DropSample(record=record, impact_ned=impact.ned, label=f"#{record.index}"))

    from airdrop import FitConfig

    result = fit_ballistics(samples, FitConfig())
    print("  ok=%s reliable=%s（拟合参数 %s）" % (result.ok, result.reliable, list(result.fitted)))
    print(
        "  Cd=%.4f（真值 0.8），κ=Cd·A/m=%.5f，RMS=%.1e m"
        % (
            result.parameters.get("drag_coefficient", float("nan")),
            result.drag_k_per_m or float("nan"),
            result.rms_error_m or float("nan"),
        )
    )


# ----------------------------------------------------------------------
def ex_cropproc() -> None:
    """裁剪图后处理（几何部分，不需要 OCR 权重）：五边形定位。"""
    image = np.zeros((320, 320, 3), dtype=np.uint8)
    side = 100
    left, top = 110, 180  # 正方形左上角
    apex_y = int(top - side * (3**0.5) / 2)  # 等边三角形顶角
    pentagon = np.array(
        [
            [left, top + side],
            [left + side, top + side],
            [left + side, top],
            [left + side // 2, apex_y],
            [left, top],
        ],
        dtype=np.int32,
    )
    cv2.fillPoly(image, [pentagon], (255, 0, 0))  # BGR：纯蓝
    post = OpenCvPostProcess(color="blue")
    post.image = image  # edge_detection 用 self.image 作为输入
    approx, ok = post.edge_detection()
    print("  五边形顶点 %d 个，ok=%s（s_min=%d）" % (len(approx), ok, post.s_min))


# ----------------------------------------------------------------------
def ex_perception_worker() -> None:
    """感知主循环（cls12 模式，注入假检测器）：缓冲 → 检测 → Detection。"""
    buffer = AlignmentBuffer(capacity=4, storage="raw")
    snapshot = TelemetrySnapshot(timestamp=time.time())
    snapshot.north_m, snapshot.east_m, snapshot.down_m = 100.0, 200.0, -50.0
    snapshot.vx_m_s = snapshot.vy_m_s = snapshot.vz_m_s = 0.0
    from airdrop import AlignedSample

    for index in (1, 2, 3):
        frame = VideoFrame(
            index=index,
            image=np.zeros((180, 320, 3), dtype=np.uint8),
            timestamp=time.time(),
            lag=0.15,
        )
        buffer.put(
            AlignedSample(
                frame=frame,
                snapshot=snapshot,
                timestamp=frame.timestamp,
                lag=0.15,
                mode="interpolate",
            )
        )

    class _FakeDetector:
        def load(self):
            return self

        def detect(
            self,
            image,
            *,
            frame_index=0,
            capture_timestamp=0.0,
            telemetry=None,
            mode="",
            undistort=None,
        ):
            detection = Detection(
                frame_index=frame_index,
                capture_timestamp=capture_timestamp,
                pixel=(160.0, 90.0),
                box=PixelBox(150.0, 80.0, 170.0, 100.0),
                confidence=0.9,
                telemetry=telemetry,
                code=7,
                mode=mode,
            )
            return DetectionBatch(image=image, detections=(detection,), undistorted=False)

    worker = PerceptionWorker(
        PerceptionConfigLike(mode="cls12"), buffer=buffer, detector=_FakeDetector()
    )
    worker.start()
    try:
        deadline = time.time() + 5.0
        results = []
        while time.time() < deadline and len(results) < 3:
            results.extend(worker.drain_results())
            time.sleep(0.05)
    finally:
        worker.stop()
    print(
        "  结果 %d 条，编号 %s，统计 %s" % (len(results), [d.code for d in results], worker.stats)
    )


# ----------------------------------------------------------------------
def ex_recorder_replay(work: Path) -> None:
    """记录 + 回放：工作区临时目录里录一次（无帧），再打开回放目录。"""
    from airdrop import FlightLog, FlightRecorder, ReplayVideoSource, load_broker_from_log

    config = Config()
    broker = TelemetryBroker()
    buffer = AlignmentBuffer(capacity=4)
    recorder = FlightRecorder(config, base_dir=work)
    recorder.start(broker=broker, buffer=buffer)
    recorder.events.emit("state", from_state="RECON", to_state="HOLD")
    recorder.stop()
    flight_dir = recorder.flight_dir
    assert flight_dir is not None

    # 手工补一份最小素材（录的时候没有帧/遥测）：1 帧 + 3 条遥测
    frames_dir = flight_dir / "frames"
    frames_dir.mkdir(exist_ok=True)
    base = time.time()
    snapshots = []
    for i, north in enumerate((0.0, 10.0, 20.0)):
        snapshot = TelemetrySnapshot(timestamp=base + i * 0.1)
        snapshot.north_m = north
        snapshots.append(snapshot)
    (flight_dir / "telemetry.jsonl").write_text(
        "\n".join(json.dumps(s.as_dict()) for s in snapshots) + "\n", encoding="utf-8"
    )
    jpeg = cv2.imencode(".jpg", np.zeros((180, 320, 3), dtype=np.uint8))[1].tobytes()
    (frames_dir / "000001.jpg").write_bytes(jpeg)
    index_line = {
        "index": 1,
        "filename": "000001.jpg",
        "capture_timestamp": base,
        "received_timestamp": base + 0.15,
        "lag": 0.15,
        "extrapolated": False,
        "offset": 0.0,
        "bytes": len(jpeg),
    }
    (flight_dir / "frames_index.jsonl").write_text(json.dumps(index_line) + "\n", encoding="utf-8")

    log = FlightLog.open(flight_dir)
    assert sum(1 for _ in log.iter_telemetry()) == 3
    assert len(log.frames) == 1
    replay_broker, pacer = load_broker_from_log(flight_dir)
    pacer.publish_until(snapshots[-1].timestamp)
    assert replay_broker.get_snapshot().north_m == 20.0

    # 真的回放一次（speed=0 全速；sink 路径一帧不落）
    received: list[VideoFrame] = []
    source = ReplayVideoSource(flight_dir, speed=0.0)
    source.add_sink(received.append)
    source.start()
    try:
        assert source.wait_ready(timeout=5.0)
        deadline = time.time() + 3.0
        while not received and time.time() < deadline:
            time.sleep(0.05)
        assert received, "回放一帧都没收到"
        # 回放必须保持原拍摄时刻（这正是"回放等价于实飞"的前提）
        assert abs(received[0].capture_timestamp - base) < 1e-9
    finally:
        source.stop()


# ----------------------------------------------------------------------
# 手册 §3 的 11 个示例，一条一个用例
# ----------------------------------------------------------------------
def test_handbook_telemetry_snippet() -> None:
    """§3.1 遥测：传入 MAVSDK 风格的更新，按时间查询。"""
    ex_telemetry()


def test_handbook_align_and_buffer_snippet() -> None:
    """§3.2 对齐 + 缓冲：帧 → 拍摄时刻遥测 → 环形缓冲。"""
    ex_align_and_buffer()


def test_handbook_georef_snippet() -> None:
    """§3.5 坐标解算：水平朝下看 + 主点像素 ⇒ 目标在飞机正下方。"""
    ex_georef()


def test_handbook_targeting_snippet() -> None:
    """§3.6 目标统计：DBSCAN + 类内编号众数。"""
    ex_targeting()


def test_handbook_ballistics_snippet() -> None:
    """§3.7 弹道 + 投放判据。"""
    ex_ballistics()


def test_handbook_mission_plan_snippet() -> None:
    """§3.8 航线规划：侦察航线 + 飞掠/降落合并。"""
    ex_mission_plan()


def test_handbook_mission_runner_snippet() -> None:
    """§3.8 主循环：单拍驱动状态机（假控制器 + 假时钟）。"""
    ex_mission_runner()


def test_handbook_drops_fit_snippet() -> None:
    """§3.7 投放记录 + 反演：合成真值应被收回。"""
    ex_drops_fit()


def test_handbook_cropproc_snippet() -> None:
    """§3.4 五边形几何：合成目标图能过形态判据。"""
    ex_cropproc()


def test_handbook_perception_worker_snippet() -> None:
    """§3.4 感知主循环（假检测器 + 假池，不碰 GPU）。"""
    ex_perception_worker()


def test_handbook_recorder_replay_snippet(workdir: Path) -> None:
    """§3.3 记录 + 回放：录一个目录、补最小素材、按原时间轴放一遍。"""
    ex_recorder_replay(workdir)
