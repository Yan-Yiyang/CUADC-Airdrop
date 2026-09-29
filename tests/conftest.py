"""pytest 公共装置（fixture）与本地测试流。

本目录的测试全部离线运行：不需要飞控，也不需要 HM30 硬件。要用到"图传"的
用例，由 :class:`StreamSender` 用 ffmpeg 把 ``testsrc`` 编成 H.264/MPEG-TS 推到
本地 UDP，再让 :mod:`airdrop.video` 去拉——路径与真实 RTSP 完全一致（同一套
解复用、同一套低延迟 flag、同一套断流判定与重连逻辑）。

常用命令::

    ./.venv/Scripts/python.exe -m pytest                     # 全部
    ./.venv/Scripts/python.exe -m pytest -m "not stream"     # 跳过要起 ffmpeg 的用例
    ./.venv/Scripts/python.exe -m pytest -k alignment -v     # 只跑对齐相关
    ./.venv/Scripts/python.exe -m pytest --no-cov            # 不统计覆盖率

注意：``stream`` 标记的用例会占用本地 UDP 端口 51234。
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Iterator
from dataclasses import replace
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

TEST_TEMP_ROOTS = (
    ".replay-test-tmp",
    ".calibrate-test-tmp",
    ".e2e-test-tmp",
    ".fit-test-tmp",
    ".handbook-test-tmp",
    ".plan-test-tmp",
    ".pytest-tmp",
    ".world-test-tmp",
)


def _cleanup_test_temp_dirs() -> None:
    """清理测试在工作区根目录创建的临时目录。"""
    for name in TEST_TEMP_ROOTS:
        shutil.rmtree(REPO_ROOT / name, ignore_errors=True)
    for path in REPO_ROOT.glob("pytest-cache-files-*"):
        shutil.rmtree(path, ignore_errors=True)


@pytest.fixture(scope="session", autouse=True)
def cleanup_test_temp_dirs() -> Iterator[None]:
    """测试会话前后清理工作区内的临时目录。"""
    _cleanup_test_temp_dirs()
    try:
        yield
    finally:
        _cleanup_test_temp_dirs()


from airdrop import TelemetryBroker
from airdrop.video.source import Hm30VideoSource, VideoConfig, _find_ffmpeg

# 本地测试流的参数（多个模块共用）
STREAM_PORT = 51234
STREAM_URL = f"udp://127.0.0.1:{STREAM_PORT}"
SRC_W, SRC_H, SRC_FPS = 640, 360, 30
FIRST_FRAME_TIMEOUT = 12.0
BIND_GRACE = 0.8  # 先让接收端 bind 住 UDP 端口，再起发送端


class StreamSender:
    """可反复启停的本地测试流发送端（等价于 HM30 的图传源）。

    用例自己决定何时起、何时停——断流重连那几组要的就是"中途把流掐掉"。
    """

    def __init__(self, executable: str) -> None:
        self._executable = executable
        self._processes: list[subprocess.Popen] = []

    def start(self) -> subprocess.Popen:
        process = subprocess.Popen(
            [
                self._executable,
                "-hide_banner",
                "-loglevel",
                "error",
                "-re",
                "-f",
                "lavfi",
                "-i",
                f"testsrc=size={SRC_W}x{SRC_H}:rate={SRC_FPS}",
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",
                "-tune",
                "zerolatency",
                "-g",
                "15",
                "-pix_fmt",
                "yuv420p",
                "-f",
                "mpegts",
                STREAM_URL,
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self._processes.append(process)
        return process

    @staticmethod
    def stop(process: subprocess.Popen | None) -> None:
        if process is not None and process.poll() is None:
            process.terminate()
            process.wait(timeout=5)

    def stop_all(self) -> None:
        for process in self._processes:
            self.stop(process)


@pytest.fixture(scope="session")
def ffmpeg_exe() -> str:
    """ffmpeg 可执行文件；找不到就跳过需要它的用例。"""
    executable = _find_ffmpeg()
    if not executable:
        pytest.skip("找不到 ffmpeg，跳过图传用例")
    return executable


@pytest.fixture
def sender(ffmpeg_exe: str) -> Iterator[StreamSender]:
    """本地测试流发送端，用例结束后自动清理全部子进程。"""
    helper = StreamSender(ffmpeg_exe)
    yield helper
    helper.stop_all()


@pytest.fixture
def broker() -> TelemetryBroker:
    """已就绪的遥测代理：``history_interval=0``，每次更新都入库，便于确定性断言。"""
    return TelemetryBroker(history_maxlen=200, history_interval=0.0)


@pytest.fixture
def received(broker: TelemetryBroker) -> list:
    """订阅 ``broker`` 之后收到的 yaw 序列，用于验证订阅推送。"""
    items: list = []
    broker.subscribe(lambda snapshot: items.append(snapshot.yaw_deg))
    return items


@pytest.fixture
def stream_config() -> VideoConfig:
    """本地测试流的默认拉流配置（640x360，输出尺寸与源一致）。"""
    return VideoConfig(url=STREAM_URL, width=SRC_W, height=SRC_H)


@pytest.fixture
def make_source():
    """拉流源工厂：用例结束自动 ``stop()``（幂等，用例中途也可以自己停）。"""
    sources: list[Hm30VideoSource] = []

    def factory(config: VideoConfig) -> Hm30VideoSource:
        source = Hm30VideoSource(config)
        sources.append(source)
        return source

    yield factory
    for source in sources:
        source.stop()


@pytest.fixture
def live_source(make_source, sender: StreamSender, stream_config: VideoConfig):
    """起好本地流、并等到首帧的拉流源工厂。

    ``configure`` 回调在 ``start()`` 之前执行，因此可以在第一帧到来之前挂上
    sink（"每帧都经过 sink"那组需要这个时序）。
    """

    def factory(
        configure: Callable[[Hm30VideoSource], None] | None = None,
        **overrides,
    ) -> Hm30VideoSource:
        config = replace(stream_config, **overrides) if overrides else stream_config
        source = make_source(config)
        if configure is not None:
            configure(source)
        source.start()
        time.sleep(BIND_GRACE)
        sender.start()
        assert source.wait_ready(timeout=FIRST_FRAME_TIMEOUT), (
            f"首帧未到达: state={source.stats.state} err={source.stats.last_error}"
        )
        return source

    return factory
