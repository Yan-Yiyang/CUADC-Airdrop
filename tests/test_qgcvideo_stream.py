from __future__ import annotations

import re
import shutil
import socket
import subprocess
import time
from pathlib import Path
from typing import Any

import imageio_ffmpeg
import numpy as np
import pytest

from airdrop.config import QgcVideoConfig
from airdrop.qgcvideo.models import DisplaySnapshot
from airdrop.qgcvideo.publisher import QgcVideoPublisher

pytestmark = pytest.mark.stream


class _SolidComposer:
    def compose(self, snapshot: DisplaySnapshot) -> np.ndarray:
        del snapshot
        frame = np.zeros((990, 1760, 3), dtype=np.uint8)
        frame[:, :, 1] = 100
        return frame


def _snapshot() -> DisplaySnapshot:
    from typing import cast

    return cast(DisplaySnapshot, object())


def _udp_port() -> int:
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.bind(("127.0.0.1", 0))
            port = int(sock.getsockname()[1])
            if port >= 65535:
                continue
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as rtcp:
                try:
                    rtcp.bind(("127.0.0.1", port + 1))
                except OSError:
                    continue
                return port


@pytest.mark.parametrize("transport", ["rtp", "mpegts"])
def test_bundled_ffmpeg_udp_stream_decodes_full_frame_for_late_receiver(
    transport: str, tmp_path: Path
) -> None:
    executable = imageio_ffmpeg.get_ffmpeg_exe()
    if not executable or not Path(executable).is_file() or shutil.which(executable) is None:
        pytest.skip("bundled FFmpeg executable is unavailable")
    port = _udp_port()
    config = QgcVideoConfig(
        enabled=True,
        host="127.0.0.1",
        port=port,
        transport=transport,
        fps=12.0,
        bitrate_kbps=900,
        ffmpeg_executable=executable,
        shutdown_timeout_s=3.0,
    ).validated()
    publisher = QgcVideoPublisher(config, _snapshot, _SolidComposer())
    publisher.start()
    assert publisher.running
    # 接收端晚加入：RTP 必须等到周期 IDR，MPEG-TS 必须能重新同步 TS 包。
    time.sleep(1.6)
    receiver_args = [executable, "-hide_banner", "-loglevel", "error"]
    if transport == "rtp":
        sdp = tmp_path / f"receiver-{port}.sdp"
        sdp.write_text(
            "v=0\n"
            "o=- 0 0 IN IP4 127.0.0.1\n"
            "s=QGC video test\n"
            "c=IN IP4 127.0.0.1\n"
            "t=0 0\n"
            f"m=video {port} RTP/AVP 96\n"
            "a=rtpmap:96 H264/90000\n"
            "a=fmtp:96 packetization-mode=1\n",
            encoding="ascii",
            newline="\n",
        )
        receiver_args.extend(
            [
                "-protocol_whitelist",
                "file,udp,rtp",
                "-localaddr",
                "127.0.0.1",
                "-i",
                str(sdp),
            ]
        )
    else:
        receiver_args.extend(
            [
                "-probesize",
                "200000",
                "-analyzeduration",
                "1000000",
                "-i",
                f"udp://127.0.0.1:{port}?fifo_size=500000&overrun_nonfatal=1",
            ]
        )
    receiver_args.extend(["-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"])
    receiver = subprocess.Popen(
        receiver_args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        stdout, stderr = receiver.communicate(timeout=8.0)
    finally:
        publisher.stop(timeout=3.0)
        if receiver.poll() is None:
            receiver.kill()
            receiver.wait(timeout=1.0)

    assert receiver.returncode == 0, stderr.decode("utf-8", errors="replace")
    assert len(stdout) == 1760 * 990 * 3
    decoded = np.frombuffer(stdout, dtype=np.uint8).reshape((990, 1760, 3))
    assert float(decoded[:, :, 1].mean()) > 30
    assert publisher.stats.encoded > 10


@pytest.mark.parametrize("compose_delay", [0.0, 0.2])
def test_publisher_preserves_wall_time_and_finishes_complete_tail(
    compose_delay: float, tmp_path: Path
) -> None:
    """真实编码器在慢合成时保留媒体时间，EOF 收尾后的末帧仍可严格解码。"""

    class Composer(_SolidComposer):
        def compose(self, snapshot: DisplaySnapshot) -> np.ndarray:
            time.sleep(compose_delay)
            return super().compose(snapshot)

    output = tmp_path / "encoded.ts"
    processes: list[subprocess.Popen] = []

    def factory(command: list[str], **kwargs: Any) -> subprocess.Popen:
        # 只替换最终输出地址，使用生产的输入时钟、同步模式和真实编码器。
        process = subprocess.Popen([*command[:-1], str(output)], **kwargs)
        processes.append(process)
        return process

    config = QgcVideoConfig(enabled=True, transport="mpegts", fps=12.0)
    publisher = QgcVideoPublisher(config, _snapshot, Composer(), subprocess_factory=factory)
    started = time.monotonic()
    publisher.start()
    try:
        time.sleep(4.0)
    finally:
        publisher.stop(timeout=3.0)
    elapsed = time.monotonic() - started

    assert len(processes) == 1
    assert processes[0].returncode == 0
    assert publisher.stats.failures == 0
    assert publisher.stats.encoded >= 10
    decoded = subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-hide_banner",
            "-xerror",
            "-err_detect",
            "explode",
            "-i",
            str(output),
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        check=False,
        timeout=15.0,
    )
    log = decoded.stderr.decode("utf-8", errors="replace")
    assert decoded.returncode == 0, log
    duration = re.search(r"Duration: (\d+):(\d+):(\d+\.\d+)", log)
    assert duration is not None, log
    media_seconds = int(duration[1]) * 3600 + int(duration[2]) * 60 + float(duration[3])
    assert 0.8 <= media_seconds / elapsed <= 1.2, (media_seconds, elapsed, log)


def test_real_encoder_failure_is_recorded_and_restart_finishes_cleanly(tmp_path: Path) -> None:
    """真实编码器报错后有界重启；错误不因后续恢复而消失，正常末帧仍完整。"""
    processes: list[subprocess.Popen] = []

    def factory(command: list[str], **kwargs: Any) -> subprocess.Popen:
        command = [*command[:-1], str(tmp_path / f"encoded-{len(processes)}.ts")]
        if not processes:
            command[command.index("libx264")] = "airdrop_intentional_invalid_encoder"
        process = subprocess.Popen(command, **kwargs)
        processes.append(process)
        return process

    publisher = QgcVideoPublisher(
        QgcVideoConfig(
            enabled=True, transport="mpegts", fps=12.0, restart_delay_s=0.05, max_restarts=1
        ),
        _snapshot,
        _SolidComposer(),
        subprocess_factory=factory,
    )
    publisher.start()
    try:
        deadline = time.monotonic() + 8.0
        while publisher.stats.encoded < 12 and time.monotonic() < deadline:
            time.sleep(0.05)
        assert publisher.stats.encoded >= 12
    finally:
        publisher.stop(timeout=3.0)
    assert publisher.stats.restarts == 1
    assert publisher.stats.failures >= 1
    assert len(processes) == 2
    assert processes[0].returncode != 0
    assert processes[1].returncode == 0
    result = subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-hide_banner",
            "-xerror",
            "-i",
            str(tmp_path / "encoded-1.ts"),
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        check=False,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr.decode("utf-8", errors="replace")
