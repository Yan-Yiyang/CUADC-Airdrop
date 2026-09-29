"""集中式运行模块（airdrop/run.py）的离线验收。

全部离线：不连飞控、不开相机、不碰 GPU，也没有一个用例真的去跑任务——跑的都是
``--help``、参数校验与"选项 → 关键字"的翻译。

三条核心口径：

1. 子命令注册表完整：每个子命令都有 parser 与 handler；
2. 帮助信息不加载重库：``--help``（顶层与每个子命令）跑在干净子进程里，
   跑完 ``sys.modules`` 里不许有 torch / cv2 / mavsdk / ultralytics / rapidocr；
3. 选项真的进了配置：``run.resolve_kwargs()`` 给出的关键字交给各入口的
   ``build_config()``，配置里确实变了（``dataclasses.replace`` 那条路）。
"""

from __future__ import annotations

import importlib
import inspect
import subprocess
import sys
from pathlib import Path

import pytest

from airdrop import run

REPO_ROOT = Path(__file__).resolve().parents[1]

#: 帮助信息与 ``import airdrop`` 都不许加载的重依赖
HEAVY_MODULES = ("torch", "cv2", "mavsdk", "ultralytics", "rapidocr", "onnxruntime")


def _run_in_subprocess(code: str) -> subprocess.CompletedProcess[str]:
    """在干净解释器里跑一段代码（重库是否被加载只能这样量）。"""
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
        check=False,  # 非零退出由调用方按退出码断言
    )


# ----------------------------------------------------------------------
# 1) 注册表：每个子命令都有 parser 与 handler
# ----------------------------------------------------------------------
def test_every_subcommand_has_a_parser_and_a_handler() -> None:
    parser = run.build_parser()
    # argparse 把子命令注册在 _SubParsersAction.choices 里（名字 -> 子解析器）
    actions = [
        action for action in parser._actions if action.__class__.__name__ == "_SubParsersAction"
    ]
    assert len(actions) == 1, "顶层解析器应当只有一个子命令动作"
    registered = actions[0].choices
    # choices 在桩里是 Iterable | None；这里断言它确实注册过（上面已确认只有一个动作）
    assert registered is not None
    assert set(registered) == set(run.SUBCOMMANDS), "注册表与解析器必须一一对应"

    for name, spec in run.SUBCOMMANDS.items():
        module = importlib.import_module(spec.module)
        handler = getattr(module, spec.handler, None)
        assert callable(handler), f"{name}: {spec.module}.{spec.handler} 不可调用"
        assert callable(getattr(module, "build_config", None)), (
            f"{name}: {spec.module} 缺少 build_config()（纯库模块的约定）"
        )
        # 每个选项的默认值出处必须是模块里真有的常量
        for option in spec.options:
            if option.source:
                assert hasattr(module, option.source), (
                    f"{name}: {spec.module} 里没有常量 {option.source}"
                )


def test_subcommand_list_covers_the_old_entries() -> None:
    """覆盖改造前的全部入口（少一个就是"某个脚本没地方跑了"）。"""
    assert set(run.SUBCOMMANDS) == {
        "full-mission",
        "sitl",
        "replay",
        "basic",
        "hm30-video",
        "video-sync",
        "calibration-capture",
        "calibrate",
        "fit-ballistics",
        "fetch-models",
        "make-world",
        "make-backdrops",
        "fetch-aerial",
        "preview-world",
        "dump-api",
        "check-docs",
    }


def test_help_text_lists_every_subcommand() -> None:
    result = _run_in_subprocess("import airdrop.run; airdrop.run.main(['--help'])")
    assert result.returncode == 0, result.stderr
    for name in run.SUBCOMMANDS:
        assert name in result.stdout, f"--help 里没有列出子命令 {name}"


# ----------------------------------------------------------------------
# 2) 惰性导入：帮助信息不加载重库（核心验收）
# ----------------------------------------------------------------------
_HELP_PROBE = """
import sys
import airdrop.run as runner

code = 0
try:
    code = runner.main({argv!r})
except SystemExit as exc:          # argparse 的 --help 走 SystemExit(0)
    code = 0 if exc.code is None else exc.code
print("EXIT", code)
print("HEAVY", [name for name in {heavy!r} if name in sys.modules] or "NONE")
"""


@pytest.mark.parametrize("argv", [["--help"], *([name, "--help"] for name in run.SUBCOMMANDS)])
def test_help_does_not_load_heavy_libraries(argv: list[str]) -> None:
    """顶层与每个子命令的 ``--help``：退出码 0，且 sys.modules 里没有重依赖。"""
    result = _run_in_subprocess(_HELP_PROBE.format(argv=argv, heavy=HEAVY_MODULES))
    assert result.returncode == 0, result.stderr
    assert "EXIT 0" in result.stdout, result.stdout + result.stderr
    assert "HEAVY NONE" in result.stdout, (
        f"--help（{' '.join(argv)}）加载了重库：{result.stdout.strip()}"
    )


def test_importing_airdrop_is_light() -> None:
    """``import airdrop`` 本身不加载重依赖（惰性导出的底线）。"""
    result = _run_in_subprocess(
        "import sys; import airdrop;"
        f"print('HEAVY', [n for n in {HEAVY_MODULES!r} if n in sys.modules] or 'NONE')"
    )
    assert result.returncode == 0, result.stderr
    assert "HEAVY NONE" in result.stdout, result.stdout


def test_lazy_exports_still_resolve_across_subpackages() -> None:
    """``import airdrop`` 之后，跨子包的名字照样取得出来（惰性导出没漏）。"""
    result = _run_in_subprocess(
        "import airdrop;"
        "names = ('MavsdkThread', 'Config', 'BallisticsModel', 'load_plan', 'MissionState');"
        "print('RESOLVED', [n for n in names if getattr(airdrop, n, None) is None] or 'ALL')"
    )
    assert result.returncode == 0, result.stderr
    assert "RESOLVED ALL" in result.stdout, result.stdout + result.stderr


def test_check_docs_subcommand_is_light_and_passes() -> None:
    """``check-docs`` 也是 handler：跑完不许加载重库，退出码 0（手册与代码一致）。"""
    result = _run_in_subprocess(
        "import sys; import airdrop.run as runner;"
        "code = runner.main(['check-docs']);"
        f"print('HEAVY', [n for n in {HEAVY_MODULES!r} if n in sys.modules] or 'NONE');"
        "raise SystemExit(code)"
    )
    assert "HEAVY NONE" in result.stdout, result.stdout + result.stderr
    assert result.returncode == 0, result.stdout + result.stderr


# ----------------------------------------------------------------------
# 3) 参数校验：未知子命令 / 非法参数以 2 退出
# ----------------------------------------------------------------------
def test_unknown_subcommand_exits_with_two() -> None:
    result = _run_in_subprocess("import airdrop.run; airdrop.run.main(['nope'])")
    assert result.returncode == 2, result.stdout + result.stderr


def test_missing_subcommand_exits_with_two() -> None:
    result = _run_in_subprocess("import airdrop.run; airdrop.run.main([])")
    assert result.returncode == 2, result.stdout + result.stderr


@pytest.mark.parametrize(
    ("argv", "what"),
    [
        (["full-mission", "--telemetry-lag", "abc"], "浮点参数写错"),
        (["full-mission", "--recon-upload", "manual"], "枚举取值不在集合里"),
        (["full-mission", "--unknown-option"], "未知选项"),
        (["sitl", "--target-offset", "300,0"], "三元组少一个数"),
        (["calibrate", "--pattern", "9"], "二元组少一个数"),
        (["make-world", "--sun", "218"], "太阳参数少一个数"),
        (["preview-world", "--round", "abc"], "轮次不是整数"),
        (["replay", "--speed", "fast"], "浮点参数写错"),
    ],
)
def test_invalid_arguments_exit_with_two(argv: list[str], what: str) -> None:
    result = _run_in_subprocess(f"import airdrop.run; airdrop.run.main({argv!r})")
    assert result.returncode == 2, f"{what}: {result.stdout + result.stderr}"


# ----------------------------------------------------------------------
# 4) 选项 → 关键字 → 配置：真的进了 build_config()
# ----------------------------------------------------------------------
def _config_kwargs(module_name: str, kwargs: dict) -> dict:
    """只挑 ``build_config`` 认的关键字（``dry_run``/``record`` 这类是运行期开关）。"""
    module = importlib.import_module(module_name)
    parameters = inspect.signature(module.build_config).parameters
    if any(item.kind is inspect.Parameter.VAR_KEYWORD for item in parameters.values()):
        return dict(kwargs)
    return {key: value for key, value in kwargs.items() if key in parameters}


def test_full_mission_options_override_the_config() -> None:
    from examples import full_mission

    kwargs = run.resolve_kwargs(
        "full-mission",
        [
            "--system-address",
            "udpin://0.0.0.0:14550",
            "--rtsp-url",
            "rtsp://127.0.0.1:8554/test",
            "--telemetry-lag",
            "0.25",
            "--land-plan",
            "routes/land.plan",
            "--recon-upload",
            "auto",
            "--no-video",
            "--no-preflight",
            "--no-require-airborne",
            "--dry-run",
        ],
    )
    config = full_mission.build_config(**_config_kwargs("examples.full_mission", kwargs))

    assert config.telemetry.system_address == "udpin://0.0.0.0:14550"
    assert config.video.url == "rtsp://127.0.0.1:8554/test"
    assert config.video.telemetry_lag == pytest.approx(0.25)
    assert config.routes.land_plan == "routes/land.plan"
    assert not config.routes.landing_route, "走 plan 时不能再带配置航点"
    assert config.mission.recon_upload == "auto"
    assert config.mission.require_airborne is False
    # --no-video / --no-preflight：一个把视频链路关掉，一个把自检各项关掉
    assert not any(
        (
            config.preflight.load_detector,
            config.preflight.load_ocr,
            config.preflight.load_camera,
            config.preflight.check_video,
        )
    )
    # 运行期开关不进 config，但要真的传给了 main()
    assert kwargs["dry_run"] is True and kwargs["use_video"] is False


def test_make_world_options_override_the_config() -> None:
    """make-world 的太阳/天井开关真的进了 build_config()。"""
    from tools import make_world

    kwargs = run.resolve_kwargs(
        "make-world",
        ["--seed", "7", "--sun", "90,12", "--well-arrows", "--wind", "5,2,0"],
    )
    config = make_world.build_config(**_config_kwargs("tools.make_world", kwargs))
    assert config.seed == 7
    assert config.sun_azel_deg == (90.0, 12.0)
    assert config.sun_random is False, "默认不随机；只有给了 --sun-random 才随机"
    assert config.well_arrows is True
    assert config.wind_enu_m_s == (5.0, 2.0, 0.0)

    random_kwargs = run.resolve_kwargs("make-world", ["--sun-random"])
    random_config = make_world.build_config(**_config_kwargs("tools.make_world", random_kwargs))
    assert random_config.sun_random is True

    patches_kwargs = run.resolve_kwargs("make-world", ["--aerial-patches", "3"])
    patches_config = make_world.build_config(**_config_kwargs("tools.make_world", patches_kwargs))
    assert patches_config.aerial_patches == 3


def test_backdrop_options_override_the_config() -> None:
    """make-backdrops / fetch-aerial 的选项真的进了各自的 build_config()。"""
    from tools import fetch_aerial, make_backdrops

    kwargs = run.resolve_kwargs(
        "make-backdrops", ["--seed", "3", "--size", "256", "--parts", "ground"]
    )
    config = make_backdrops.build_config(**_config_kwargs("tools.make_backdrops", kwargs))
    assert config.seed == 3 and config.size == 256 and config.parts == "ground"

    kwargs = run.resolve_kwargs("fetch-aerial", ["--size", "640", "--tile-width-m", "200"])
    config = fetch_aerial.build_config(**_config_kwargs("tools.fetch_aerial", kwargs))
    assert config.size == 640 and config.tile_width_m == pytest.approx(200.0)


def test_sitl_options_override_the_config() -> None:
    from examples import sitl_mission

    kwargs = run.resolve_kwargs(
        "sitl",
        ["--target-offset", "300,10,-2", "--system-address", "udpin://0.0.0.0:14541"],
    )
    config = sitl_mission.build_config(**_config_kwargs("examples.sitl_mission", kwargs))

    assert kwargs["target_offset_ned"] == (300.0, 10.0, -2.0)
    assert config.telemetry.system_address == "udpin://0.0.0.0:14541"
    # 演练档的两条默认：不等起飞、自检全关（--require-airborne 未给 → 保持默认）
    assert config.mission.require_airborne is False
    assert config.mission.recon_upload == "auto"


def test_replay_options_override_the_config() -> None:
    from examples import replay_flight

    kwargs = run.resolve_kwargs(
        "replay", ["--flight", "flights/x", "--speed", "2.5", "--perception-mode", "cls12"]
    )
    config = replay_flight.build_config(**_config_kwargs("examples.replay_flight", kwargs))

    assert kwargs["flight_dir"] == Path("flights/x")
    assert kwargs["speed"] == pytest.approx(2.5)
    assert config.perception.mode == "cls12"


def test_calibrate_options_override_the_config() -> None:
    from tools import calibrate

    kwargs = run.resolve_kwargs(
        "calibrate",
        ["--flight", "flights/y", "--pattern", "7,5", "--square", "0.03", "--strict"],
    )
    settings = calibrate.build_config(**kwargs)

    assert settings.flight_dir == Path("flights/y")
    assert settings.pattern_size == (7, 5)
    assert settings.square_size == pytest.approx(0.03)
    assert settings.strict is True


def test_fit_ballistics_options_override_the_config() -> None:
    from tools import fit_ballistics

    kwargs = run.resolve_kwargs(
        "fit-ballistics", ["--impacts", "flights/z/impacts.jsonl", "--mass-kg", "0.42"]
    )
    settings = fit_ballistics.build_config(mass_kg=kwargs["mass_kg"])

    # --impacts / --drops / --out 是 main() 的运行期入参，--mass-kg 才是配置
    assert kwargs["impacts"] == Path("flights/z/impacts.jsonl")
    assert settings.base.mass_kg == pytest.approx(0.42)
    assert settings.fit_drag_coefficient is fit_ballistics.FIT_DRAG_COEFFICIENT


def test_defaults_come_from_the_module_constants() -> None:
    """不带选项时，关键字必须等于入口模块的常量（默认值不写第二遍）。"""
    from examples import full_mission, sitl_mission
    from tools import calibrate

    kwargs = run.resolve_kwargs("full-mission", [])
    assert kwargs["rtsp_url"] == full_mission.RTSP_URL
    assert kwargs["telemetry_lag_s"] == full_mission.TELEMETRY_LAG_S
    assert kwargs["land_plan"] == full_mission.LAND_PLAN

    sitl = run.resolve_kwargs("sitl", [])
    assert sitl["target_offset_ned"] == sitl_mission.TARGET_OFFSET_NED
    assert sitl["system_address"] == sitl_mission.SYSTEM_ADDRESS

    cal = run.resolve_kwargs("calibrate", [])
    assert cal["pattern_size"] == calibrate.PATTERN_SIZE
    assert cal["flight_dir"] == calibrate.FLIGHT_DIR
