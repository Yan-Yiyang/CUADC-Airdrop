"""统一任务 CLI 与轻量入口回归。"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from airdrop import run
from airdrop.runtime.mission import main as run_mission
from airdrop.runtime.session import LiveComponents

ROOT = Path(__file__).resolve().parents[1]
HEAVY_MODULES = ("torch", "cv2", "mavsdk", "ultralytics", "rapidocr", "onnxruntime")


def _run(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
        check=False,
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )


def test_registry_has_one_mission_and_offline_commands() -> None:
    assert "mission" in run.SUBCOMMANDS
    assert "replay" in run.SUBCOMMANDS
    assert not set(run.SUBCOMMANDS) & {
        "full-mission",
        "sitl",
        "sitl-recon",
        "basic",
        "hm30-video",
        "video-sync",
        "calibration-capture",
    }
    assert run.SUBCOMMANDS["mission"].module == "airdrop.runtime.mission"


@pytest.mark.parametrize(
    "argv",
    [["--help"], ["mission", "--help"], ["replay", "--help"], ["qgc-video-test", "--help"]],
)
def test_help_does_not_load_heavy_libraries(argv: list[str]) -> None:
    code = (
        "import sys; from airdrop.run import main; "
        f"\ntry: main({argv!r})\nexcept SystemExit as exc: assert exc.code == 0\n"
        f"print([name for name in {HEAVY_MODULES!r} if name in sys.modules] or 'NONE')"
    )
    result = _run(code)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.rstrip().endswith("NONE")


def test_mission_options_parse_to_runtime_keywords() -> None:
    values = run.resolve_kwargs(
        "mission",
        [
            "--mode",
            "real",
            "--no-mission-control",
            "--capture",
            "--no-recognize",
            "--record",
            "--max-seconds",
            "20",
            "--backup-point",
            "31.2,121.5,10",
        ],
    )
    assert values["mode"] == "real"
    assert values["run_mission"] is False
    assert values["capture"] is True and values["recognize"] is False
    assert values["max_seconds"] == 20
    assert values["backup_point"] == (31.2, 121.5, 10.0)
    defaults = run.resolve_kwargs("mission", ["--mode", "sitl"])
    assert all(
        defaults[key] is None
        for key in (
            "system_address",
            "rtsp_url",
            "calib_file",
            "recon_plan",
            "land_plan",
            "qgc_video",
            "qgc_host",
            "qgc_port",
            "qgc_transport",
            "qgc_fps",
            "qgc_bitrate_kbps",
        )
    )


def test_real_release_has_explicit_positive_and_negative_flags() -> None:
    assert run.resolve_kwargs("mission", ["--real-release"])["real_release"] is True
    assert run.resolve_kwargs("mission", ["--no-real-release"])["real_release"] is False


def test_qgc_cli_options_resolve_to_config_overrides_and_tool_arguments() -> None:
    mission = run.resolve_kwargs("mission", ["--mode", "real", "--qgc-video", "--qgc-port", "5614"])
    assert mission["qgc_video"] is True
    assert mission["qgc_port"] == 5614
    tool = run.resolve_kwargs(
        "qgc-video-test",
        ["--port", "5616", "--preview", "preview.png", "--ffmpeg", "ffmpeg.exe"],
    )
    assert tool["port"] == 5616
    assert tool["preview_path"] == Path("preview.png")
    assert tool["ffmpeg_executable"] == "ffmpeg.exe"


def test_real_cli_keywords_run_through_injected_factory() -> None:
    calls: list[str] = []

    class Thread:
        def connect(self) -> None:
            calls.append("connect")

        def stop(self, timeout: float = 5.0) -> None:
            del timeout
            calls.append("stop")

    def factory(options, config) -> LiveComponents:
        assert options.mode == "real" and not options.run_mission
        assert config.telemetry.system_address == "udp://127.0.0.1:14540"
        assert config.video.url == "rtsp://example.test/stream"
        assert config.routes.recon_plan == "custom-recon.plan"
        assert config.routes.land_plan == "custom-land.plan"
        assert config.qgc_video.enabled
        assert config.qgc_video.host == "localhost"
        assert config.qgc_video.port == 5612
        return LiveComponents(thread=Thread())

    kwargs = run.resolve_kwargs(
        "mission",
        [
            "--mode",
            "real",
            "--no-mission-control",
            "--no-recognize",
            "--no-record",
            "--max-seconds",
            "0.000001",
            "--system-address",
            "udp://127.0.0.1:14540",
            "--rtsp-url",
            "rtsp://example.test/stream",
            "--recon-plan",
            "custom-recon.plan",
            "--land-plan",
            "custom-land.plan",
            "--qgc-video",
            "--qgc-host",
            "localhost",
            "--qgc-port",
            "5612",
        ],
    )

    assert kwargs["work_dir"] is None
    assert run_mission(**kwargs, components_factory=factory) == 0
    assert calls == ["connect", "stop"]


@pytest.mark.parametrize(
    "argv",
    [
        ["mission", "--mode", "unknown"],
        ["mission", "--max-seconds", "nan"],
        ["mission", "--backup-point", "31,121"],
        ["mission", "--no-capture", "--no-recognize", "--qgc-video"],
        ["replay", "--speed", "fast"],
    ],
)
def test_invalid_arguments_fail_at_cli_boundary(argv: list[str]) -> None:
    result = _run(f"from airdrop.run import main; main({argv!r})")
    assert result.returncode == 2, result.stdout + result.stderr


def test_importing_real_runtime_does_not_import_sitl_or_heavy_libraries() -> None:
    result = _run(
        "import sys; import airdrop.runtime.mission as mission; "
        "assert callable(mission.build_live_components); "
        f"print([name for name in {HEAVY_MODULES!r} if name in sys.modules] or 'NONE'); "
        "print('SITL', 'airdrop.runtime.sitl' in sys.modules)"
    )
    assert result.returncode == 0, result.stderr
    assert "NONE" in result.stdout
    assert "SITL False" in result.stdout


def test_docs_check_subcommand_passes() -> None:
    result = _run("from airdrop.run import main; raise SystemExit(main(['check-docs']))")
    assert result.returncode == 0, result.stdout + result.stderr


def test_fetch_models_missing_source_returns_failure(tmp_path: Path) -> None:
    from tools import fetch_models

    missing = tmp_path / "missing"
    assert (
        fetch_models.main(source_dir=missing, source_ocr_dir=missing, target_dir=tmp_path / "out")
        == 2
    )
