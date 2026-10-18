from __future__ import annotations

import shutil
import socket
import subprocess
import time
from pathlib import Path

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
