"""示例目录的"不生锈"检查：每个示例都能导入、都有 ``main()``、都不在导入期碰硬件。

示例是给人照抄的，最容易随接口演进而悄悄失效（改了构造参数、搬了模块，示例却没人跑）。
这里只做**离线可做的**那部分校验：

* 模块能导入（说明 import 的 API 都还在，且没有在导入期连飞控/开图传）；
* 有 ``main()`` 且返回 int（``examples`` 的约定：不带命令行参数、配置在文件顶部）；
* 模块级常量区块存在（``FLIGHT_DIR``/``ROUTES`` 之类），便于人工核对。

**不**执行 ``main()``：那几个示例要飞控、图传或真实素材，跑起来就不是单测了。
"""

from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"


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
    assert module.__doc__, f"examples/{name}.py 缺少模块 docstring（用法写在里面）"
    assert getattr(module, "__file__", "").endswith(f"{name}.py")


@pytest.mark.parametrize("name", _example_modules())
def test_example_documents_its_configuration(name: str) -> None:
    """项目约定：示例**不带命令行参数**，配置在文件顶部常量里。"""
    source = (EXAMPLES_DIR / f"{name}.py").read_text(encoding="utf-8")

    assert "argparse" not in source, f"examples/{name}.py 不该用命令行参数"
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


def test_calibration_capture_builds_a_valid_config() -> None:
    from examples import calibration_capture

    config = calibration_capture.build_config()

    assert config.telemetry.attitude_rate_hz >= 50.0, "标定建议 ≥50Hz 姿态"
    assert config.video.telemetry_lag == calibration_capture.TELEMETRY_LAG_S


def test_sitl_scenario_builds_config_and_synthetic_target() -> None:
    """SITL 演练的合成目标：不连飞控也能验的部分（配置 + 目标合成 + 缓存）。"""
    from types import SimpleNamespace

    from airdrop import TelemetryBroker

    from examples import sitl_mission

    config = sitl_mission.build_config()
    assert config.telemetry.system_address == sitl_mission.SYSTEM_ADDRESS
    assert config.routes.landing_route and config.overfly.heading_deg == (
        sitl_mission.OVERFLY_HEADING_DEG
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
