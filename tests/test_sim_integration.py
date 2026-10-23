"""生成工具、manifest 与主仓演练的跨边界契约。"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from airdrop import Config, TelemetryBroker, load_camera_model
from airdrop.mission.plan_file import load_plan
from airdrop.run import resolve_kwargs

pytest.importorskip("airdrop_sim", reason="跨仓契约需要仿真子模块")

from airdrop_sim import make_vehicle, make_world
from airdrop_sim.camera import CameraConfig
from airdrop_sim.stream import write_stream_sdp

from airdrop.runtime import sitl, sitl_analysis, sitl_geometry, sitl_report
from airdrop.runtime.sitl_scene import load_recon_scene

pytestmark = pytest.mark.integration


def _custom_scene(tmp_path: Path):
    """从真实生成入口取得与缺省布局不同的场景，不拼造内部 DTO。"""
    assert (
        make_vehicle.main(
            out_dir=tmp_path,
            seed=7,
            width_px=800,
            height_px=600,
            hfov_rad=1.2,
            mount_translation_frd_m=(1.0, 2.0, 0.5),
            origin_lat_deg=35.0,
            origin_lon_deg=112.0,
            origin_alt_m=120.0,
        )
        == 0
    )
    return load_recon_scene(tmp_path / "simulation.manifest.json")


def test_manifest_drives_camera_truth_and_report(tmp_path: Path) -> None:
    scene = _custom_scene(tmp_path)
    assert scene.seed == 7
    assert scene.origin_lla == (112.0, 35.0, 120.0)
    assert scene.wells_enu != sitl_geometry.WELLS_A_ENU
    calib = sitl_geometry.write_sim_calib(tmp_path / "readback.json", camera=scene.camera)
    camera = load_camera_model(calib)
    assert camera.image_size == (800, 600)
    assert camera.fx == pytest.approx(400 / math.tan(0.6))
    code = scene.holdout_code
    east, north = scene.wells_enu[code]
    # 理想下视相机中心射线沿 FRD +Z；机体位置扣除杆臂后，落点就是独立目标真值。
    detail = [
        {
            "code": code,
            "frame_index": 1,
            "pixel": [400.0, 300.0],
            "telemetry": {
                "north_m": north - 1.0,
                "east_m": east - 2.0,
                "down_m": -20.0,
                "roll_deg": 0.0,
                "pitch_deg": 0.0,
                "yaw_deg": 0.0,
                "quaternion_w": 1.0,
                "quaternion_x": 0.0,
                "quaternion_y": 0.0,
                "quaternion_z": 0.0,
                "origin_altitude_m": scene.origin_lla[2],
            },
        }
    ]
    errors = sitl_analysis._observation_errors(
        detail, origin_lla=scene.origin_lla, calib_file=str(calib), scene=scene
    )
    assert errors["all"]["n"] == 1
    assert errors["all"]["max_m"] < 0.01
    outcome = sitl_analysis._RunOutcome(
        work=tmp_path,
        sdp=tmp_path / "stream.sdp",
        calib=calib,
        lag_s=0.49,
        flown_s=None,
        result=SimpleNamespace(clusters=[], noise=[], selected=None),
        stats={},
        perception=cast(
            Any,
            SimpleNamespace(
                frames=0, detections=0, read_codes=0, last_error=None, snapshot=lambda: {}
            ),
        ),
        video_stats=SimpleNamespace(frames=0, dropped=0, reconnects=0),
        writer_stats=(0, 0),
        detail=[],
        broker=TelemetryBroker(),
        flight_dir=None,
        origin=scene.origin_lla,
        config=Config(),
        scene=scene,
    )
    report = sitl_report._report(outcome)
    assert report["simulation"]["seed"] == 7
    assert report["truth_ned"][str(code)] == pytest.approx([north, east], abs=0.01)
    assert (
        json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))["simulation"]["seed"]
        == 7
    )


def test_scene_is_local_and_default_assembly_uses_operator_route_path(tmp_path: Path) -> None:
    scene = _custom_scene(tmp_path)
    baseline = load_recon_scene()
    default_config = sitl_geometry.build_config()
    local_plan = tmp_path / "local-land.plan"
    local_plan.write_text(
        json.dumps(
            {
                "fileType": "Plan",
                "mission": {
                    "items": [
                        {
                            "type": "SimpleItem",
                            "command": 16,
                            "frame": 3,
                            "params": [0, 0, 0, 0, 35.0, 112.0, 30.0],
                            "autoContinue": True,
                            "doJumpId": 1,
                        }
                    ],
                    "plannedHomePosition": [35.0, 112.0, 0.0],
                },
            }
        ),
        encoding="utf-8",
    )
    config = sitl_geometry.build_config(
        scene=scene,
        route=sitl_geometry.recon_route(scene=scene),
        land_plan=str(local_plan),
    )
    explicit_default_scene = sitl_geometry.build_config(land_plan=str(local_plan))
    default_plan = Path(default_config.routes.land_plan)
    assert default_plan.is_absolute()
    assert default_plan == (
        Path(sitl_geometry.__file__).resolve().parents[2] / "routes" / "land.plan"
    )
    assert Path(explicit_default_scene.routes.land_plan) == local_plan
    assert load_plan(explicit_default_scene.routes.land_plan).items[0].lat == pytest.approx(35.0)
    assert (
        replace(
            explicit_default_scene,
            routes=replace(
                explicit_default_scene.routes, land_plan=default_config.routes.land_plan
            ),
        )
        == default_config
    )
    assert config.routes.backup_point is not None
    assert config.routes.backup_point.lat == pytest.approx(35.0, abs=0.01)
    assert load_recon_scene() == baseline
    assert baseline.origin_lla == sitl_geometry.WORLD_ORIGIN_LLA
    with pytest.raises(ValueError, match="原点"):
        sitl_geometry.build_config(scene=scene)
    with pytest.raises(ValueError, match="覆盖"):
        sitl_geometry.build_config(scene=replace(scene, origin_lla=baseline.origin_lla))
    with pytest.raises(ValueError, match="覆盖"):
        sitl_geometry.build_config(
            scene=replace(
                baseline, camera=replace(baseline.camera, mount_translation_frd_m=(0.0, 0.0, 10.0))
            )
        )
    output = tmp_path / "rejected"
    with pytest.raises(ValueError, match="原点"):
        sitl.main(work_dir=str(output), manifest_path=tmp_path / "simulation.manifest.json")
    assert not output.exists()


def test_round_one_manifest_is_rejected_for_numbered_recon(tmp_path: Path) -> None:
    assert make_vehicle.main(out_dir=tmp_path, round_no=1) == 0
    with pytest.raises(ValueError, match="第二轮"):
        load_recon_scene(tmp_path / "simulation.manifest.json")


def test_duplicate_numbered_targets_cannot_silently_overwrite_truth(tmp_path: Path) -> None:
    _custom_scene(tmp_path)
    path = tmp_path / "simulation.manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    first, second = [
        row for row in payload["truth"] if row["zone"] == "A" and row["code"] is not None
    ][:2]
    second["code"], second["content"] = first["code"], first["content"]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="重复编号"):
        load_recon_scene(path)


def test_vehicle_cli_runs_from_another_directory(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root)
    result = subprocess.run(
        [
            sys.executable,
            "-X",
            "utf8",
            "-m",
            "airdrop.run",
            "make-vehicle",
            "--out-dir",
            "vehicle",
            "--width",
            "800",
            "--height",
            "600",
            "--hfov",
            "1.2",
            "--seed",
            "7",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    scene = load_recon_scene(tmp_path / "vehicle/simulation.manifest.json")
    assert (scene.camera.width_px, scene.camera.height_px) == (800, 600)
    assert scene.seed == 7
    kwargs = resolve_kwargs(
        "mission",
        ["--mode", "sitl", "--manifest", "vehicle/simulation.manifest.json"],
    )
    assert kwargs["manifest_path"] == Path("vehicle/simulation.manifest.json")
    assert kwargs["lag_s"] is None


def test_stream_and_camera_configuration_validate_before_writing(tmp_path: Path) -> None:
    target = tmp_path / "sdp"
    with pytest.raises(ValueError, match="端口"):
        write_stream_sdp(target, port=0)
    assert not target.exists()
    text = write_stream_sdp(target, port=5601).read_text(encoding="ascii")
    assert "m=video 5601 RTP/AVP 96" in text
    assert replace(CameraConfig(), width_px=800).width_px == 800
    calibration = sitl_geometry.write_sim_calib(
        tmp_path / "custom_lag.json", camera=replace(CameraConfig(), telemetry_lag_s=0.17)
    )
    assert load_camera_model(calibration).telemetry_lag == pytest.approx(0.17)


def test_sitl_recon_route_aims_at_the_median_well() -> None:
    """适配航线和备用点从同一场景真值构造。"""
    from airdrop.georef.geo import wgs84_to_ned

    reference = (make_world.WORLD_LON_DEG, make_world.WORLD_LAT_DEG, 0.0)

    def enu(waypoint) -> tuple[float, float]:
        north, east, _down = wgs84_to_ned(waypoint.lon, waypoint.lat, 0.0, reference)
        return east, north

    config = sitl_geometry.build_config()
    scene = load_recon_scene()
    backup_east, backup_north = enu(config.routes.backup_point)
    truth_east = sum(point[0] for point in scene.wells_enu.values()) / len(scene.wells_enu)
    truth_north = sum(point[1] for point in scene.wells_enu.values()) / len(scene.wells_enu)
    assert math.hypot(backup_east - truth_east, backup_north - truth_north) < 0.5


def test_sitl_recon_truth_table_matches_the_world() -> None:
    """真值表（编号 → 天井中心 ENU）必须与入库世界的 A 区布点逐座对得上。"""
    wells = {}
    for well in make_world.wells_for_round(2):
        if well.zone != "A" or well.content is None:
            continue
        payload = well.content[1]
        assert isinstance(payload, tuple)  # 数字靶的内容是 (tens, ones)
        wells[payload[0] * 10 + payload[1]] = well
    assert set(sitl_geometry.WELLS_A_ENU) <= set(wells)
    for code, (east_m, north_m) in sitl_geometry.WELLS_A_ENU.items():
        well = wells[code]
        assert abs(well.east_m - east_m) < 0.05, f"{code} 号天井东向坐标与世界不一致"
        assert abs(well.north_m - north_m) < 0.05, f"{code} 号天井北向坐标与世界不一致"


def test_sitl_recon_route_covers_every_well_and_flies_both_directions() -> None:
    """侦查航线要真的看得见三个靶标，而且有**反向**观测（延时诊断要用）。

    ⚠ 判据随固定翼转弯几何改过一次（2026-09-30）。原来两条要求是"每条扫掠线在
    **2 m** 内压过天井" + "存在一组**同线**反向"。同线反向必然要求 180° 掉头，而掉头的
    横向挪移**恒为 2R**（两段反向圆弧横移之和与圆心角无关，R≈22.9 m ⇒ 46 m）——靶区
    南北只跨 19 m、15 m 高度的视场半幅只有 17.8 m，两条线拉开 46 m 之后第二条线
    **一个靶标都看不见**。两条要求互斥，所以现在改成：

    * 覆盖判据从"2 m"放宽到**相机视场半幅**（真的看得见就算数）；
    * 反向观测不再要求同线，只要求存在一段**东向分量为负**、且同样扫过靶标的腿。
    """
    from airdrop.runtime import sitl_geometry

    def distance_to_leg(point: tuple[float, float], leg) -> float:
        (ax, ay), (bx, by) = leg
        dx, dy = bx - ax, by - ay
        length_sq = dx * dx + dy * dy
        t = (
            0.0
            if length_sq == 0.0
            else max(0.0, min(1.0, ((point[0] - ax) * dx + (point[1] - ay) * dy) / length_sq))
        )
        return math.dist(point, (ax + t * dx, ay + t * dy))

    half_width = sitl_geometry.effective_half_width_m()
    for code, (east_m, north_m) in sitl_geometry.WELLS_A_ENU.items():
        target = (east_m, north_m)
        nearest = min(distance_to_leg(target, leg) for leg in sitl_geometry.RECON_LEGS_ENU)
        assert nearest <= half_width, (
            f"{code} 号天井离航线 {nearest:.1f} m，超出视场半幅 {half_width:.1f} m"
            "（改航线要照着真值表来）"
        )
    # 反向观测：存在一段东向分量为负、且离每个靶标都在视场内的腿
    backward = [
        leg
        for leg in sitl_geometry.RECON_LEGS_ENU
        if leg[1][0] < leg[0][0]
        and all(
            distance_to_leg(truth, leg) <= half_width
            for truth in sitl_geometry.WELLS_A_ENU.values()
        )
    ]
    assert backward, "没有一段东向为负、又能扫到靶标的腿（反向诊断/延时诊断要用）"


def test_sitl_recon_route_and_camera_writers(tmp_path: Path) -> None:
    """航点、模拟相机标定、SDP 的纯函数产物要能被下游真读走。"""
    from airdrop import load_camera_model
    from airdrop.runtime import sitl_geometry

    route = sitl_geometry.recon_route()
    assert route[0].alt_m == pytest.approx(sitl_geometry.TAKEOFF_ALT_M)  # 首项 = 起飞项
    assert all(point.alt_m == pytest.approx(sitl_geometry.RECON_ALT_M) for point in route[1:])

    work = tmp_path / "sitl-recon"
    calib = sitl_geometry.write_sim_calib(work / "camera_calib_sim.json")
    model = load_camera_model(calib)
    assert model.is_calibrated()  # 读得回来（不是缺省模型）
    assert model.fx == pytest.approx(
        (sitl_geometry.CAMERA_WIDTH / 2.0) / math.tan(sitl_geometry.CAMERA_HFOV_RAD / 2.0)
    )
    assert list(model.t_bc) == pytest.approx(list(sitl_geometry.CAMERA_T_BC))

    sdp = sitl_geometry.write_stream_sdp(work / "stream.sdp")
    text = sdp.read_text(encoding="ascii")
    assert f"m=video {sitl_geometry.STREAM_PORT} RTP/AVP 96" in text
    assert "a=rtpmap:96 H264/90000" in text
