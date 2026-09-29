"""示例目录的"不生锈"检查：每个示例都能导入、都有 ``main()``、导入期不加载重依赖。

示例是给人照抄的，最容易随接口演进而悄悄失效（改了构造参数、搬了模块，示例却没人跑）。
这里只做离线可做的那部分校验：

* 模块能导入（说明 import 的 API 都还在，且没有在导入期连飞控/开视频）；
* 导入不加载重依赖（torch / cv2 / mavsdk / ultralytics / rapidocr）：重活都在
  函数体内，命令行只需要读常量——airdrop.run 的 --help 靠这条保持廉价；
* 有 ``main()``（``examples`` 的约定：``main(**kwargs)`` + 顶部常量 = 默认值）；
* 模块级常量区块存在，且 ``build_config(**覆盖)`` 真的覆盖得动。

不执行 ``main()``：那几个示例要飞控、视频或真实素材，跑起来就不是单测了。
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES_DIR = REPO_ROOT / "examples"

#: 示例模块导入后不许出现在 sys.modules 里的重依赖
HEAVY_MODULES = ("torch", "cv2", "mavsdk", "ultralytics", "rapidocr", "onnxruntime")


def _example_modules() -> list[str]:
    return sorted(
        module.name
        for module in pkgutil.iter_modules([str(EXAMPLES_DIR)])
        if not module.name.startswith("_")
    )


def test_examples_directory_is_not_empty() -> None:
    assert _example_modules(), "examples/ 下没有示例模块"


@pytest.mark.parametrize("name", _example_modules())
def test_example_imports_and_exposes_main(name: str) -> None:
    module = importlib.import_module(f"examples.{name}")

    assert callable(getattr(module, "main", None)), f"examples/{name}.py 缺少 main()"
    assert callable(getattr(module, "build_config", None)), (
        f"examples/{name}.py 缺少 build_config()（命令行覆盖的入口）"
    )
    assert module.__doc__, f"examples/{name}.py 缺少模块 docstring（用法写在里面）"
    assert getattr(module, "__file__", "").endswith(f"{name}.py")


@pytest.mark.parametrize("name", _example_modules())
def test_importing_an_example_does_not_load_heavy_libraries(name: str) -> None:
    """项目约定：示例是纯库模块——重依赖只在函数体内导入。

    命令行只读常量（默认值）与调 ``build_config``，所以导入期必须廉价；
    这条同时保证 ``python -m airdrop.run <子命令> --help`` 不加载重库。
    """
    # 在干净解释器里测：pytest 进程自己可能已经加载过 cv2（别的用例干的）
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            f"import sys; import examples.{name};"
            f"print([m for m in {HEAVY_MODULES!r} if m in sys.modules] or 'NONE')",
        ],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
        check=False,  # 退出码由下面那条断言负责
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().endswith("NONE"), (
        f"examples/{name}.py 导入期加载了重依赖：{result.stdout.strip()}"
    )


@pytest.mark.parametrize("name", _example_modules())
def test_example_documents_its_configuration(name: str) -> None:
    """项目约定：示例的默认值写在文件顶部常量里，命令行解析只在 airdrop/run.py。"""
    source = (EXAMPLES_DIR / f"{name}.py").read_text(encoding="utf-8")

    assert "argparse" not in source, f"examples/{name}.py 不该自己解析命令行（归 airdrop/run.py）"
    assert "if __name__" in source, f"examples/{name}.py 缺少 __main__ 入口"


# ----------------------------------------------------------------------
# 装配逻辑：不碰硬件的部分要真跑（导入检查抓不到 replace() 写错这类问题）
# ----------------------------------------------------------------------
def test_full_mission_builds_a_valid_config() -> None:
    from examples import full_mission

    config = full_mission.build_config()

    assert config.routes.recon_route and config.routes.backup_point
    assert config.routes.landing_route, "没有降落航线就拼不出完整任务"
    assert config.perception.mode == full_mission.PERCEPTION_MODE
    assert config.targeting.selection_rule == full_mission.SELECTION_RULE
    assert config.video.url == full_mission.RTSP_URL
    assert config.video.telemetry_lag == full_mission.TELEMETRY_LAG_S
    assert config.overfly.heading_deg == full_mission.OVERFLY_HEADING_DEG
    # 正式入口的两条默认：自检全开、侦察航线等操作手在 QGC 上传并启动、必须等起飞
    assert config.preflight.load_detector and config.preflight.load_camera
    assert config.preflight.check_video
    assert config.mission.recon_upload == "operator"
    assert config.mission.require_airborne is True


def test_full_mission_build_config_honours_overrides() -> None:
    """build_config(**覆盖) 必须真的覆盖得动（命令行就是走这条路）。"""
    from examples import full_mission

    config = full_mission.build_config(
        system_address="udpin://0.0.0.0:14550",
        rtsp_url="rtsp://127.0.0.1:8554/x",
        telemetry_lag_s=0.3,
        land_plan="routes/land.plan",
        recon_upload="auto",
        use_video=False,
        require_airborne=False,
    )

    assert config.telemetry.system_address == "udpin://0.0.0.0:14550"
    assert config.video.url == "rtsp://127.0.0.1:8554/x"
    assert config.video.telemetry_lag == 0.3
    assert config.routes.land_plan == "routes/land.plan" and not config.routes.landing_route
    assert config.mission.recon_upload == "auto"
    assert config.mission.require_airborne is False
    assert not config.preflight.check_video, "不接视频时视频自检无从做起"


def test_full_mission_default_route_passes_landing_precheck() -> None:
    """默认配置必须真的拼得出任务：降落几何过 PX4 预检（这是最容易踩的坑）。"""
    from airdrop import build_drop_mission
    from airdrop.georef import LLARef
    from examples import full_mission

    config = full_mission.build_config()
    origin = LLARef(lon_deg=8.5456, lat_deg=47.3977, alt_m=500.0)
    plan = build_drop_mission(config, origin=origin, target_ned=(100.0, 200.0, 0.0))

    commands = [item.command for item in plan.items]
    assert commands[:2] == [16, 16], "飞掠段是 [entry, exit] 两个 NAV_WAYPOINT"
    assert 21 in commands, "降落段末尾必须有 NAV_LAND"
    assert plan.source == "target" and plan.landing_count >= 2


def test_calibration_capture_builds_a_valid_config() -> None:
    from examples import calibration_capture

    config = calibration_capture.build_config()

    assert config.telemetry.attitude_rate_hz >= 50.0, "标定建议 ≥50Hz 姿态"
    assert config.video.telemetry_lag == calibration_capture.TELEMETRY_LAG_S
    # 覆盖生效：速率/延时都能从外面改
    tuned = calibration_capture.build_config(telemetry_lag_s=0.2, attitude_rate_hz=100.0)
    assert tuned.video.telemetry_lag == 0.2
    assert tuned.telemetry.attitude_rate_hz == 100.0


def test_sitl_scenario_builds_config_and_synthetic_target() -> None:
    """SITL 演练的合成目标：不连飞控也能验的部分（配置 + 目标合成 + 缓存）。"""
    from types import SimpleNamespace

    from airdrop import TelemetryBroker
    from examples import sitl_mission

    config = sitl_mission.build_config()
    assert config.telemetry.system_address == sitl_mission.SYSTEM_ADDRESS
    assert config.routes.landing_route or config.routes.land_plan, "降落段必须有一个来源"
    assert config.overfly.heading_deg == sitl_mission.OVERFLY_HEADING_DEG
    # 演练档：侦察航线由本包上传；起飞前自检四项全关（SITL 没有相机/模型/视频）
    assert config.mission.recon_upload == "auto"
    assert not any(
        (
            config.preflight.load_detector,
            config.preflight.load_ocr,
            config.preflight.load_camera,
            config.preflight.check_video,
        )
    )
    # 地面演练的临时放行开关：不等起飞（正式任务必须 True，见 airdrop/config.py）
    assert config.mission.require_airborne is False
    assert sitl_mission.build_config(require_airborne=True).mission.require_airborne is True, (
        "override 要真的覆盖得动"
    )

    broker = TelemetryBroker()
    broker.update_local_position_velocity(
        SimpleNamespace(north_m=10.0, east_m=20.0, down_m=-50.0),
        SimpleNamespace(north_m_s=15.0, east_m_s=0.0, down_m_s=0.0),
    )
    targets = sitl_mission.SyntheticTargets((300.0, 0.0, 0.0), broker)

    assert targets.busy() is False, "合成目标永远“处理完了”"
    result = targets.result()
    assert result.ok and result.code == 56
    assert result.north_m == pytest.approx(310.0), "盘旋点 + 偏移"
    assert result.east_m == pytest.approx(20.0)
    assert targets.result() is result, "结果算一次就缓存（判据每拍都会问）"


def test_replay_flight_build_config_honours_overrides() -> None:
    from examples import replay_flight

    default = replay_flight.build_config()
    tuned = replay_flight.build_config(perception_mode="cls12", selection_rule="max")

    assert default.perception.mode == "ocr" and default.targeting.selection_rule == "median"
    assert tuned.perception.mode == "cls12" and tuned.targeting.selection_rule == "max"
    # build_config 的入参必须都是关键字（命令行按关键字覆盖）
    for name, parameter in inspect.signature(replay_flight.build_config).parameters.items():
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY, f"{name} 应当是关键字参数"
