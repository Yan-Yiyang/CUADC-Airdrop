"""使用生产 QGC 发布器测量本地 RTP 时钟和 UDP 包序列。"""

from __future__ import annotations

import itertools
import json
import math
import socket
import struct
import threading
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

OUTPUT_PATH = None
SECONDS = 6.0
FPS = 12.0
COMPOSE_DELAY = 0.125


def _measure(*, seconds: float, fps: float, compose_delay: float) -> dict[str, Any]:
    import numpy as np

    from airdrop.config import QgcVideoConfig
    from airdrop.qgcvideo.publisher import QgcVideoPublisher

    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    packets: list[dict[str, int | float | bool]] = []
    received_datagrams = 0
    raw_headers: list[str] = []
    receiver_errors: list[str] = []
    stopped = threading.Event()

    def collect() -> None:
        nonlocal received_datagrams
        try:
            while not stopped.is_set():
                try:
                    data = receiver.recv(65536)
                except TimeoutError:
                    continue
                received_datagrams += 1
                if len(raw_headers) < 5:
                    raw_headers.append(data[:16].hex())
                if len(data) >= 12 and data[0] >> 6 == 2 and data[1] & 0x7F == 96:
                    packets.append(
                        {
                            "arrival_s": time.monotonic(),
                            "sequence": struct.unpack_from(">H", data, 2)[0],
                            "timestamp": struct.unpack_from(">I", data, 4)[0],
                            "marker": bool(data[1] & 0x80),
                        }
                    )
        except OSError as exc:
            if not stopped.is_set():
                receiver_errors.append(str(exc))

    class Composer:
        def __init__(self) -> None:
            self.image = np.zeros((990, 1760, 3), dtype=np.uint8)
            self.index = 0

        def compose(self, snapshot: Any) -> np.ndarray:
            del snapshot
            if compose_delay:
                time.sleep(compose_delay)
            self.index += 1
            self.image[0, 0, 0] = self.index % 255
            return self.image

    thread: threading.Thread | None = None
    publisher: QgcVideoPublisher | None = None
    started = time.monotonic()
    try:
        receiver.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)
        receiver.bind(("127.0.0.1", 0))
        receiver.settimeout(0.1)
        thread = threading.Thread(target=collect, name="qgc-video-benchmark-receiver", daemon=True)
        config = QgcVideoConfig(
            enabled=True, host="127.0.0.1", port=receiver.getsockname()[1], fps=fps
        ).validated()
        publisher = QgcVideoPublisher(
            config, cast(Any, lambda: SimpleNamespace(replay_mode=True)), Composer()
        )
        thread.start()
        started = time.monotonic()
        try:
            publisher.start()
            if not publisher.running:
                raise RuntimeError(f"QGC 发布器启动失败：{publisher.stats.last_error}")
            time.sleep(seconds)
        finally:
            publisher.stop()
    finally:
        stopped.set()
        try:
            if thread is not None and thread.ident is not None:
                thread.join(timeout=2.0)
        finally:
            receiver.close()
    if thread is not None and thread.is_alive():
        raise TimeoutError("RTP 接收线程未能退出")
    if publisher is None:
        raise RuntimeError("QGC 发布器未能构造")
    if receiver_errors:
        raise RuntimeError(f"RTP 接收线程失败：{receiver_errors[0]}")
    ordered = sorted(packets, key=lambda item: float(item["arrival_s"]))
    markers = [item for item in ordered if item["marker"]]
    if len(markers) < 2:
        raise RuntimeError(
            f"收到的有效 RTP 帧不足：{len(markers)}；UDP包={received_datagrams}，"
            f"样本头={raw_headers}；统计：{asdict(publisher.stats)}"
        )
    arrival_span = float(markers[-1]["arrival_s"]) - float(markers[0]["arrival_s"])
    media_span = (
        (int(markers[-1]["timestamp"]) - int(markers[0]["timestamp"])) & 0xFFFFFFFF
    ) / 90000.0
    gaps = sum(
        max(((int(b["sequence"]) - int(a["sequence"])) & 0xFFFF) - 1, 0)
        for a, b in itertools.pairwise(ordered)
    )
    stats = publisher.stats
    return {
        "compose_delay_s": compose_delay,
        "requested_seconds": seconds,
        "elapsed_s": time.monotonic() - started,
        "requested_fps": fps,
        "arrival_span_s": arrival_span,
        "rtp_media_span_s": media_span,
        "wall_minus_media_s": arrival_span - media_span,
        "rtp_packets": len(ordered),
        "received_datagrams": received_datagrams,
        "sample_headers_hex": raw_headers,
        "marker_frames": len(markers),
        "sequence_gaps": gaps,
        "publisher": asdict(stats),
    }


def main(
    *,
    output_path: Path | None,
    seconds: float = 6.0,
    fps: float = 12.0,
    compose_delay: float = 0.125,
) -> int:
    """依次测量正常与慢合成；仅在本地回环发送，无模型和飞控连接。"""
    if output_path is None:
        raise ValueError("必须指定 --out")
    if output_path.exists():
        raise FileExistsError(f"拒绝覆盖已有诊断结果：{output_path}")
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("seconds 必须是正的有限数")
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps 必须是正的有限数")
    if not math.isfinite(compose_delay) or compose_delay < 0:
        raise ValueError("compose_delay 必须是非负有限数")
    from tools.performance import write_json_new

    result = {
        "format_version": 1,
        "description": "生产 RTP 发布器的本地回环诊断，不连接飞控或外部视频接收器",
        "cases": [
            _measure(seconds=seconds, fps=fps, compose_delay=0.0),
            _measure(seconds=seconds, fps=fps, compose_delay=compose_delay),
        ],
    }
    write_json_new(output_path, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0
