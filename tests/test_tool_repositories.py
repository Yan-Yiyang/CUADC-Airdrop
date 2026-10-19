"""子模块缺失、惰性导入及旧 CLI 转发的真实子解释器回归。"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.integration


@pytest.fixture
def core_checkout(tmp_path: Path) -> Path:
    """复制主仓源码而不带 vendor，避免修改当前子模块状态。"""
    for name in ("airdrop", "tools"):
        shutil.copytree(ROOT / name, tmp_path / name, ignore=shutil.ignore_patterns("__pycache__"))
    return tmp_path


def test_core_help_never_imports_tool_packages(core_checkout: Path) -> None:
    code = """
import importlib.abc
import sys
class RejectTools(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(('airdrop_sim', 'airdrop_calibration')):
            raise AssertionError('帮助不应加载工具: ' + fullname)
        return None
sys.meta_path.insert(0, RejectTools())
import airdrop
from airdrop.run import main
for arguments in (['--help'], ['mission', '--help'], ['replay', '--help']):
    try:
        main(arguments)
    except SystemExit as exc:
        assert exc.code == 0
assert not any(n.startswith(('airdrop_sim', 'airdrop_calibration')) for n in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-X", "utf8", "-c", code],
        cwd=core_checkout,
        env={**os.environ, "PYTHONPATH": str(core_checkout)},
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("command", ["make-world", "calibrate"])
def test_missing_tool_has_actionable_error(core_checkout: Path, command: str) -> None:
    result = subprocess.run(
        [sys.executable, "-X", "utf8", "-m", "airdrop.run", command, "--help"],
        cwd=core_checkout,
        env={**os.environ, "PYTHONPATH": str(core_checkout)},
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
        timeout=60,
    )
    assert result.returncode == 2
    assert "git submodule update --init" in result.stderr
    assert "Traceback" not in result.stderr


def test_legacy_tool_command_forwards_real_arguments(tmp_path: Path) -> None:
    pytest.importorskip("airdrop_sim", reason="兼容转发集成需要仿真子模块")
    output = tmp_path / "worlds"
    result = subprocess.run(
        [
            sys.executable,
            "-X",
            "utf8",
            "-m",
            "airdrop.run",
            "make-world",
            "--rounds",
            "2",
            "--seed",
            "11",
            "--out-dir",
            str(output),
        ],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    import json

    manifest = json.loads((output / "cuadc_recon_strike_r2.manifest.json").read_text("utf-8"))
    assert manifest["seed"] == 11 and manifest["round_no"] == 2
    assert (output / "materials/textures/digit_0.png").is_file()
