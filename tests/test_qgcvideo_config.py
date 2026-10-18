"""QGC 视频端点配置检查。"""

from dataclasses import replace
from pathlib import Path

import pytest

from airdrop.config import Config, QgcVideoConfig
from airdrop.runtime.mission import prepare_config
from airdrop.runtime.options import RunOptions


def test_qgc_video_defaults_avoid_sitl_input_ports() -> None:
    config = QgcVideoConfig()
    assert (config.host, config.port, config.transport) == ("127.0.0.1", 5602, "rtp")
    assert config.validated() is config
    assert Config().validated().qgc_video == config


@pytest.mark.parametrize(
    "updates",
    [
        {"host": "127.0.0.1;touch"},
        {"host": ""},
        {"port": 0},
        {"port": 65535},
        {"transport": "rtsp"},
        {"fps": float("nan")},
        {"fps": 0.0},
        {"bitrate_kbps": 1},
        {"write_timeout_s": float("inf")},
        {"ffmpeg_executable": " "},
    ],
)
def test_qgc_video_rejects_invalid_configuration(updates: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        replace(QgcVideoConfig(), **updates).validated()


def test_qgc_video_accepts_mpegts_and_rtsp_input_is_not_transport() -> None:
    assert replace(QgcVideoConfig(), transport="mpegts").validated().transport == "mpegts"
    config = replace(Config(), video=replace(Config().video, url="rtsp://camera:8554/live"))
    assert config.validated().video.url == "rtsp://camera:8554/live"


@pytest.mark.parametrize(("port", "transport"), [(5600, "rtp"), (5599, "rtp"), (5600, "mpegts")])
def test_output_udp_port_conflict_is_rejected_before_runtime_creation(
    tmp_path: Path, port: int, transport: str
) -> None:
    options = RunOptions(mode="sitl", run_mission=False, recognize=False, qgc_video=True)
    config = replace(
        Config(),
        video=replace(Config().video, url=str(tmp_path / "not-created-yet.sdp")),
        qgc_video=replace(QgcVideoConfig(), enabled=True, port=port, transport=transport),
    )
    with pytest.raises(ValueError, match="端口.*重叠"):
        prepare_config(options, config)


def test_rtsp_input_port_is_not_treated_as_sdp_udp_port(tmp_path: Path) -> None:
    options = RunOptions(mode="real", run_mission=False, recognize=False, qgc_video=True)
    config = replace(
        Config(),
        video=replace(Config().video, url="rtsp://camera:5600/live"),
        qgc_video=replace(QgcVideoConfig(), enabled=True, port=5600),
    )
    assert prepare_config(options, config).qgc_video.port == 5600


@pytest.mark.parametrize(
    "url",
    [
        "udp://127.0.0.1:5600?fifo_size=1000000",
        "rtp://[::1]:5600?localaddr=::1",
        "udp://@:5600?fifo_size=1000000",
    ],
)
def test_udp_input_query_port_conflict_is_rejected(url: str) -> None:
    options = RunOptions(mode="real", run_mission=False, recognize=False, qgc_video=True)
    config = replace(
        Config(),
        video=replace(Config().video, url=url),
        qgc_video=replace(QgcVideoConfig(), enabled=True, port=5600),
    )
    with pytest.raises(ValueError, match="端口.*重叠"):
        prepare_config(options, config)


def test_existing_sdp_video_port_pair_is_checked(tmp_path: Path) -> None:
    sdp = tmp_path / "capture.sdp"
    sdp.write_text("v=0\nm=video 5700 RTP/AVP 96\n", encoding="utf-8")
    options = RunOptions(mode="real", run_mission=False, recognize=False, qgc_video=True)
    config = replace(
        Config(),
        video=replace(Config().video, url=str(sdp)),
        qgc_video=replace(QgcVideoConfig(), enabled=True, port=5701),
    )
    with pytest.raises(ValueError, match="端口.*重叠"):
        prepare_config(options, config)
