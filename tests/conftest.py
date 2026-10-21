"""pytest 公共装置（fixture）与本地测试流。

默认 ``fast`` 层不启动外部工具、WSL、GPU 或飞控；需要 pytest 文件/目录临时 fixture
的用例自动标为 ``integration``，由 ``offline`` 层运行。该分类读取 fixture 依赖，
不根据测试文件名猜测；纯配置、数学和受控线程用例仍可留在 ``fast``。
``offline`` 也包含显式标记的跨仓集成，
``platform`` 再包含 ffmpeg/UDP 与 WSL，``all`` 还包含真实素材和 SITL。要用到"图传"的
用例，由 :class:`StreamSender` 用 ffmpeg 把 ``testsrc`` 编成 H.264/MPEG-TS 推到
本地 UDP，再让 :mod:`airdrop.video` 去拉——路径与真实 RTSP 完全一致（同一套
解复用、同一套低延迟 flag、同一套断流判定与重连逻辑）。

常用命令::

    ./.venv/Scripts/python.exe -m pytest                     # fast（默认）
    ./.venv/Scripts/python.exe -m pytest --test-tier offline
    ./.venv/Scripts/python.exe -m pytest --test-tier platform
    ./.venv/Scripts/python.exe -m pytest --test-tier all     # 所有层级
    ./.venv/Scripts/python.exe -m pytest --test-tier all -m realdata
    ./.venv/Scripts/python.exe -m pytest -k alignment -v     # 只跑对齐相关

注意：``stream`` 标记的用例会选择一个本地临时 UDP 端口，发送端和接收端使用同一端口。
"""

from __future__ import annotations

import socket
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

# 工具测试独立运行；主仓集成测试只在子模块存在时导入它们。
for repository in ("airdrop-sim", "airdrop-calibration"):
    location = REPO_ROOT / "vendor" / repository
    if location.is_dir():
        sys.path.insert(0, str(location))

from airdrop import TelemetryBroker
from airdrop.video.source import Hm30VideoSource, VideoConfig, _find_ffmpeg

# 本地测试流的参数（多个模块共用）
SRC_W, SRC_H, SRC_FPS = 640, 360, 30
FIRST_FRAME_TIMEOUT = 12.0
BIND_GRACE = 0.8  # 先让接收端 bind 住 UDP 端口，再起发送端

_FILE_INTEGRATION_FIXTURES = {
    "tmp_path",
    "tmp_path_factory",
    "tmpdir",
    "tmpdir_factory",
    "workdir",
    "core_checkout",
}


def pytest_itemcollected(item: pytest.Item) -> None:
    """按实际 fixture 依赖自动把文件/目录测试放入 integration 层。"""
    if isinstance(item, pytest.Function) and _FILE_INTEGRATION_FIXTURES.intersection(
        item.fixturenames
    ):
        item.add_marker(pytest.mark.integration)


def pytest_addoption(parser: pytest.Parser) -> None:
    """注册按运行环境分层的测试选择参数。"""
    parser.addoption(
        "--test-tier",
        choices=("fast", "offline", "platform", "all"),
        default="fast",
        help="选择测试环境层级：fast、offline、platform 或 all",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """按显式 marker 筛选 tier，pytest 自带的 -m 仍作为额外筛选。"""
    tier = config.getoption("--test-tier")
    if not isinstance(tier, str):
        raise pytest.UsageError("--test-tier 必须是 fast、offline、platform 或 all")
    allowed = {
        "fast": set(),
        "offline": {"integration"},
        "platform": {"stream", "network", "integration", "wsl"},
        "all": {"stream", "network", "integration", "wsl", "realdata", "sitl"},
    }[tier]
    deselected: list[pytest.Item] = []
    selected: list[pytest.Item] = []
    for item in items:
        restricted = {"stream", "network", "integration", "wsl", "realdata", "sitl"}
        denied = {mark.name for mark in item.iter_markers() if mark.name in restricted} - allowed
        if denied:
            deselected.append(item)
        else:
            selected.append(item)
    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = selected


class StreamSender:
    """可反复启停的本地测试流发送端（等价于 HM30 的图传源）。

    用例自己决定何时起、何时停——断流重连那几组要的就是"中途把流掐掉"。
    """

    def __init__(self, executable: str, url: str) -> None:
        self._executable = executable
        self._url = url
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
                self._url,
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
def stream_url() -> str:
    """为单个测试选择一个当前可用的本地 UDP 端口。"""
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    return f"udp://127.0.0.1:{port}"


@pytest.fixture
def sender(ffmpeg_exe: str, stream_url: str) -> Iterator[StreamSender]:
    """本地测试流发送端，用例结束后自动清理全部子进程。"""
    helper = StreamSender(ffmpeg_exe, stream_url)
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
def stream_config(stream_url: str) -> VideoConfig:
    """本地测试流的默认拉流配置（640x360，输出尺寸与源一致）。"""
    return VideoConfig(url=stream_url, width=SRC_W, height=SRC_H)


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
