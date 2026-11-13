"""手册的代表性入口示例与文档一致性回归。

保留对齐缓冲、图像轮廓、任务驱动、感知工作器及记录回放五个装配示例。
算法结果由各模块的业务测试验证；这些手写用例不自动提取手册原文。
临时数据使用 workdir fixture，全部离线运行。
"""

from __future__ import annotations

import json
import subprocess
import time
from collections.abc import Callable
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from airdrop import (
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
    PixelBox,
    PreflightConfig,
    RoutesConfig,
    TelemetryBroker,
    TelemetrySnapshot,
    VideoFrame,
    Waypoint,
)
from airdrop.perception import OpenCvPostProcess, PerceptionConfigLike, PerceptionWorker
from tools.check_docs import _repository_hygiene

pytestmark = pytest.mark.integration


def _git(cwd: Path, *args: str) -> str:
    """在隔离临时仓库中执行 Git 命令。"""
    result = subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True, encoding="utf-8"
    )
    return result.stdout.strip()


def _commit_all(root: Path) -> str:
    """提交临时仓库文件并返回提交号。"""
    _git(root, "add", "-A")
    _git(
        root,
        "-c",
        "user.name=Docs Test",
        "-c",
        "user.email=docs@example.invalid",
        "commit",
        "-m",
        "fixture",
    )
    return _git(root, "rev-parse", "HEAD")


def test_doc_hygiene_uses_git_index_and_submodule_index(tmp_path: Path) -> None:
    """现存未跟踪文件不能掩盖死链，子仓目标由子仓索引确认。"""
    root = tmp_path / "main"
    root.mkdir()
    _git(root, "init", "-q")
    (root / "docs").mkdir()
    (root / "docs" / "guide.md").write_text("# guide\n", encoding="utf-8")
    (root / "docs" / "untracked.md").write_text("# local only\n", encoding="utf-8")

    child = root / "vendor" / "tool"
    child.mkdir(parents=True)
    _git(child, "init", "-q")
    (child / "README.md").write_text("# tool\n", encoding="utf-8")
    child_head = _commit_all(child)

    (root / "README.md").write_text(
        "[tracked](docs/guide.md) [untracked](docs/untracked.md) "
        "[child](vendor/tool/README.md) [site](https://example.invalid) [top](#top) "
        "[local](C:/Users/name/file.md)\n",
        encoding="utf-8",
    )
    _git(root, "add", "README.md", "docs/guide.md")
    _git(root, "update-index", "--add", "--cacheinfo", f"160000,{child_head},vendor/tool")
    _git(
        root,
        "-c",
        "user.name=Docs Test",
        "-c",
        "user.email=docs@example.invalid",
        "commit",
        "-m",
        "fixture",
    )

    link_issues, privacy_issues = _repository_hygiene(root)
    assert len(link_issues) == 2
    assert any("untracked.md" in issue for issue in link_issues)
    assert any("C:/Users/name/file.md" in issue for issue in link_issues)
    assert len(privacy_issues) == 1
    assert "本地绝对路径" in privacy_issues[0]


def test_doc_hygiene_scans_public_text_not_runtime_strings(tmp_path: Path) -> None:
    """私人路径和旧命名会被发现，示意路径、协议地址和运行时字符串不会误报。"""
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q")
    (root / "README.md").write_text(
        "示意 C:/...；平台路径 `/usr/bin/python`、`/tmp/generated`；协议 `udp://127.0.0.1:5602`。\n"
        "生成记录 `flights/<架次>/frames/000001.jpg`；素材 D:/captures/flight.mp4。\n",
        encoding="utf-8",
    )
    (root / "module.py").write_text(
        "# C:\\Users\\someone\\private.txt\n"
        '"""旧入口 2024v2。"""\n'
        'PRIVATE = "C:\\\\Users\\\\someone\\\\secret.mp4"\n',
        encoding="utf-8",
    )
    (root / "notes.md").write_text(
        "旧资料曾叫 2024v2；本地素材在 `C:/Users/someone/Videos/flight.mp4`。\n",
        encoding="utf-8",
    )
    _commit_all(root)

    _, privacy_issues = _repository_hygiene(root)
    assert len(privacy_issues) == 5
    assert sum("notes.md" in issue for issue in privacy_issues) == 2
    assert sum("module.py" in issue for issue in privacy_issues) == 2
    assert sum("README.md" in issue for issue in privacy_issues) == 1
    assert any("本地绝对路径" in issue for issue in privacy_issues)
    assert any("2024v2" in issue for issue in privacy_issues)


def test_doc_hygiene_rejects_retired_packages_and_entries(tmp_path: Path) -> None:
    """旧包/入口在公开文本中报错，当前模块路径和运行时字符串保持有效。"""
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q")
    (root / "README.md").write_text(
        "旧引用：`airdrop.alignment`、`airdrop.alignment.FrameTelemetryAligner`、"
        "`airdrop.buffer`、`airdrop.buffer.AlignmentWriter`、`examples/sitl_mission.py`、"
        "`tools/sitl_recon.py`、`drone/coordinates.py`、`drone/target.py`、`drone/cvproc.py`。\n"
        "当前引用：`airdrop.video.buffer`、`airdrop.video.align`。\n",
        encoding="utf-8",
    )
    (root / "module.py").write_text(
        '"""旧注释入口为 drone/cvproc.py。"""\n'
        'RUNTIME_TEXT = "airdrop.alignment examples/sitl_mission.py"\n',
        encoding="utf-8",
    )
    _commit_all(root)

    _, issues = _repository_hygiene(root)
    legacy_issues = [issue for issue in issues if "旧包/入口" in issue]
    assert len(legacy_issues) == 2
    assert any("README.md" in issue for issue in legacy_issues)
    assert any("module.py" in issue for issue in legacy_issues)


@pytest.fixture
def workdir(tmp_path: Path) -> Path:
    """为本用例提供独立素材目录。"""
    return tmp_path


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
        # 离线示例：侦查航线由示例自己上传（auto）；正式任务用默认的 operator（操作手在 QGC 启动）
        mission=MissionConfig(recon_upload="auto"),
        # 自检四项全关（示例里没有相机/模型）：关掉 ≠ 通过，只是离线跑通
        preflight=PreflightConfig(
            load_detector=False,
            load_ocr=False,
            load_camera=False,
            check_video=False,
        ),
    ).validated()


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

    def submit_inline(operation: Callable[[], int], /) -> Future[int]:
        """立即完成假控制器任务，使示例阶段断言不依赖线程调度。"""
        future: Future[int] = Future()
        future.set_result(operation())
        return future

    runner = MissionRunner(
        config,
        controller,
        broker,
        clock=clock,
        sleep=lambda _s: None,
        transfer_submit=submit_inline,
    )
    try:
        for _ in range(4):
            runner.update()
            clock.now += 0.05
        controller.finished = True
        runner.update()
        assert runner.state is MissionState.HOLD_PROCESS
        assert controller.calls.count("upload_mission") == 1
    finally:
        runner.close()
    print(
        "  状态 %s，上传 %d 次，历史 %d 条"
        % (runner.state, runner.stats.uploads, len(runner.history))
    )
    print(
        "  最后一次转移: %s → %s (%s)"
        % (runner.history[-1].from_state, runner.history[-1].to_state, runner.history[-1].reason)
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
    assert ok and len(approx) == 5
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
    assert len(results) == 3
    assert [d.code for d in results] == [7, 7, 7]
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
# 手册 §3 的五个代表性装配示例
def test_handbook_align_and_buffer_snippet() -> None:
    """§3.2 对齐 + 缓冲：帧 → 拍摄时刻遥测 → 环形缓冲。"""
    ex_align_and_buffer()


def test_handbook_mission_runner_snippet() -> None:
    """§3.8 主循环：单拍驱动状态机（假控制器 + 假时钟）。"""
    ex_mission_runner()


def test_handbook_cropproc_snippet() -> None:
    """§3.4 五边形几何：合成目标图能过形态判据。"""
    ex_cropproc()


def test_handbook_perception_worker_snippet() -> None:
    """§3.4 感知主循环（假检测器 + 假池，不碰 GPU）。"""
    ex_perception_worker()


def test_handbook_recorder_replay_snippet(workdir: Path) -> None:
    """§3.3 记录 + 回放：录一个目录、补最小素材、按原时间轴放一遍。"""
    ex_recorder_replay(workdir)
