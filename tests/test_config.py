"""配置装配与取值域校验测试。

配置是"起飞前唯一的参数入口"，所以校验必须在启动前把非法值拦下来
（而不是飞到一半才发现 selection_rule 写错）。
"""

from __future__ import annotations

import pytest

from airdrop import (
    AlignConfig,
    BallisticsConfig,
    Config,
    GripperConfig,
    MissionConfig,
    PerceptionConfig,
    TargetingConfig,
    TelemetryConfig,
    VideoConfig,
)


def test_default_config_is_valid() -> None:
    """默认配置必须直接可用（默认值即"能跑"的那一组）。"""
    cfg = Config().validated()
    assert cfg.perception.mode == "ocr"
    assert cfg.targeting.selection_rule == "median"
    assert cfg.align.max_wait == 1.0
    assert cfg.align.on_timeout == "drop"
    assert cfg.drop.radius_m == 2.0
    assert cfg.overfly.altitude_m == 20.0
    assert cfg.video.telemetry_lag > 0


def test_mission_config_defaults_match_plan() -> None:
    """计划 4.7 写死的几个数：HOLD_PROCESS 上限 10s、判据 20Hz、超时兜底。"""
    mission = MissionConfig()
    assert mission.hold_process_max_s == 10.0
    assert mission.tick_hz == 20.0 == Config().drop.evaluation_hz
    assert mission.abort_action == "hold"
    assert mission.takeoff_first and mission.land_last
    assert mission.telemetry_stale_s > Config().align.max_wait, (
        "状态机的链路看门狗要比对齐的超时宽——单帧同步不上不该把整个任务判失败"
    )
    assert Config().gripper.instance == 0


def test_replace_derives_new_config() -> None:
    """replace 派生不修改原配置（frozen dataclass 语义）。"""
    base = Config()
    derived = base.replace(perception=PerceptionConfig(mode="cls12"))
    assert base.perception.mode == "ocr"
    assert derived.perception.mode == "cls12"


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"perception": PerceptionConfig(mode="cls100")}, "perception.mode"),
        ({"perception": PerceptionConfig(target_color="green")}, "target_color"),
        ({"targeting": TargetingConfig(selection_rule="mode")}, "selection_rule"),
        ({"targeting": TargetingConfig(eps_m=0.0)}, "eps_m"),
        ({"perception": PerceptionConfig(ocr_workers=0)}, "ocr_workers"),
        ({"perception": PerceptionConfig(imgsz=16)}, "imgsz"),
        ({"align": AlignConfig(on_timeout="skip")}, "on_timeout"),
        ({"ballistics": BallisticsConfig(wind_source="manual")}, "wind_source"),
        ({"mission": MissionConfig(abort_action="land")}, "abort_action"),
        ({"mission": MissionConfig(tick_hz=0.0)}, "tick_hz"),
        ({"mission": MissionConfig(recon_max_s=-1.0)}, "recon_max_s"),
        ({"mission": MissionConfig(hold_process_min_s=20.0)}, "hold_process_min_s"),
        ({"gripper": GripperConfig(instance=-1)}, "gripper.instance"),
        ({"gripper": GripperConfig(release_settle_s=-0.5)}, "release_settle_s"),
    ],
)
def test_invalid_values_rejected(changes: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        Config(**changes).validated()


def test_invalid_video_config_rejected() -> None:
    """VideoConfig 自己的校验也要被 Config.validated() 带出来。"""
    bad = Config(video=VideoConfig(transport="quic"))
    with pytest.raises(ValueError, match="传输层"):
        bad.validated()


def test_telemetry_config_defaults_match_requirement() -> None:
    """速率下限是内插精度的前提（见 plan 4.x）：位置 ≥10Hz、姿态 ≥30Hz。"""
    tel = TelemetryConfig()
    assert tel.position_velocity_ned_rate_hz >= 10.0
    assert tel.attitude_rate_hz >= 30.0
    assert tel.history_interval > 0.0
