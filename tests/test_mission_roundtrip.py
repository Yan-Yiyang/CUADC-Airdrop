"""PX4 mission_raw 上传后的规范化回读验证。"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import math
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from airdrop.config import Config, Waypoint
from airdrop.georef import LLARef
from airdrop.mission.items import (
    MAV_CMD_DO_CHANGE_SPEED,
    MAV_CMD_DO_LAND_START,
    MAV_CMD_DO_SET_CAM_TRIGG_DIST,
    MAV_CMD_IMAGE_STOP_CAPTURE,
    MAV_CMD_NAV_LAND,
    MAV_CMD_NAV_LOITER_TO_ALT,
    MAV_CMD_NAV_WAYPOINT,
    MAV_CMD_VIDEO_STOP_CAPTURE,
    UNSET,
    MissionItem,
)
from airdrop.mission.planner import build_drop_mission
from airdrop.telemetry.controller import ControllerError, DroneController

_RAW_FIELDS = (
    "seq",
    "frame",
    "command",
    "current",
    "autocontinue",
    "param1",
    "param2",
    "param3",
    "param4",
    "x",
    "y",
    "z",
    "mission_type",
)


class _RoundTripDrone:
    """模拟 PX4 mission_raw 存储/下载，并允许测试指定读回变更。"""

    def __init__(self, mutate: Callable[[dict[str, Any]], None] | None = None) -> None:
        self.stored: list[SimpleNamespace] = []
        self.mutate = mutate
        drone = self

        class MissionRaw:
            async def upload_mission(self, items: list[Any]) -> None:
                drone.stored = []
                for item in items:
                    fields = {name: getattr(item, name) for name in _RAW_FIELDS}
                    if fields["frame"] == 0:
                        fields["frame"] = 5
                    elif fields["frame"] == 3:
                        fields["frame"] = 6
                    if fields["command"] == MAV_CMD_NAV_LAND:
                        for name in ("param1", "param2", "param3"):
                            if math.isnan(float(fields[name])):
                                fields[name] = 0.0
                    if drone.mutate is not None:
                        drone.mutate(fields)
                    drone.stored.append(SimpleNamespace(**fields))

            async def download_mission(self) -> list[SimpleNamespace]:
                return list(drone.stored)

        self.mission_raw = MissionRaw()


class _Thread:
    def __init__(self, drone: _RoundTripDrone) -> None:
        self.drone = drone

    def require_drone(self) -> _RoundTripDrone:
        return self.drone

    def submit(self, coroutine: Any) -> concurrent.futures.Future[Any]:
        future: concurrent.futures.Future[Any] = concurrent.futures.Future()
        try:
            future.set_result(asyncio.run(coroutine))
        except BaseException as exc:  # noqa: BLE001 - 与 run_coroutine_threadsafe 一致
            future.set_exception(exc)
        return future


def _upload(
    items: tuple[MissionItem, ...],
    mutate: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[int, _RoundTripDrone]:
    drone = _RoundTripDrone(mutate)
    return DroneController(cast(Any, _Thread(drone))).upload_mission(items), drone


def _ten_item_drop_and_land_mission() -> tuple[MissionItem, ...]:
    """构造两项飞掠、七项进近航点和一项降落的完整任务。"""
    items = [
        MissionItem.waypoint(47.0, 8.0, 20.0, frame=3),
        MissionItem.waypoint(47.001, 8.0, 20.0, frame=0),
    ]
    for index, altitude in enumerate((40.0, 35.0, 30.0, 25.0, 20.0, 15.0, 5.0)):
        items.append(
            MissionItem.waypoint(
                47.002 + index * 0.001,
                8.0,
                altitude,
                frame=3 if index % 2 else 0,
            )
        )
    items.append(
        MissionItem(
            command=MAV_CMD_NAV_LAND,
            lat=47.010,
            lon=8.0,
            frame=3,
            param1=UNSET,
            param2=UNSET,
            param3=UNSET,
            param4=UNSET,
        )
    )
    return tuple(items)


def test_px4_normalized_ten_item_drop_and_land_round_trip() -> None:
    items = _ten_item_drop_and_land_mission()

    count, drone = _upload(items)

    assert count == 10
    assert [item.frame for item in drone.stored if item.command == MAV_CMD_NAV_WAYPOINT] == [
        6,
        5,
        5,
        6,
        5,
        6,
        5,
        6,
        5,
    ]
    land = drone.stored[-1]
    assert (land.param1, land.param2, land.param3) == (0.0, 0.0, 0.0)
    assert math.isnan(land.param4)


def test_built_drop_mission_with_qgc_landing_tail_round_trips_through_px4(
    tmp_path: Path,
) -> None:
    """真实 planner 生成的两项飞掠加八项 QGC 降落尾段可通过规范化回读。"""
    items = [
        {
            "type": "SimpleItem",
            "command": MAV_CMD_NAV_WAYPOINT,
            "frame": 3,
            "doJumpId": index,
            "autoContinue": True,
            "Altitude": 50.0,
            "params": [0.0, 3.0, 0.0, None, 47.0, 8.004 + index * 0.001, 50.0],
        }
        for index in (1,)
    ]
    items.append(
        {
            "type": "ComplexItem",
            "complexItemType": "fwLandingPattern",
            "version": 2,
            "valueSetIsDistance": False,
            "altitudesAreRelative": True,
            "landCoordinate": [46.997, 8.0, 0.0],
            "landingApproachCoordinate": [47.0, 8.0, 40.0],
            "loiterRadius": 75,
            "loiterClockwise": True,
            "useLoiterToAlt": True,
            "useDoChangeSpeed": True,
            "finalApproachSpeed": 9,
            "stopTakingPhotos": True,
            "stopVideoPhotos": True,
        }
    )
    plan_path = tmp_path / "landing.plan"
    plan_path.write_text(
        json.dumps(
            {
                "fileType": "Plan",
                "mission": {
                    "firmwareType": 12,
                    "vehicleType": 1,
                    "plannedHomePosition": [47.0, 8.0, 500.0],
                    "items": items,
                },
            }
        ),
        encoding="utf-8",
    )
    base = Config()
    config = Config(
        routes=replace(
            base.routes,
            landing_route=(),
            land_plan=str(plan_path),
            backup_point=Waypoint(lat=47.0, lon=8.0, alt_m=0.0),
        ),
        overfly=replace(base.overfly, heading_deg=0.0, altitude_m=20.0),
        mission=replace(base.mission, auto_land=True),
        ground=base.ground,
    ).validated()

    plan = build_drop_mission(
        config,
        origin=LLARef(lon_deg=8.0, lat_deg=47.0, alt_m=500.0),
        target_ned=(0.0, 0.0, 0.0),
    )
    assert len(plan.items) == 10
    assert plan.items[2].command == MAV_CMD_NAV_WAYPOINT
    assert plan.items[3].command == MAV_CMD_DO_LAND_START
    assert plan.items[4].command == MAV_CMD_DO_CHANGE_SPEED
    assert plan.items[5].command == MAV_CMD_DO_SET_CAM_TRIGG_DIST
    assert plan.items[6].command == MAV_CMD_IMAGE_STOP_CAPTURE
    assert plan.items[7].command == MAV_CMD_VIDEO_STOP_CAPTURE
    assert plan.items[8].command == MAV_CMD_NAV_LOITER_TO_ALT
    assert plan.items[-1].command == MAV_CMD_NAV_LAND

    count, drone = _upload(plan.items)

    assert count == 10
    assert drone.stored[-1].command == MAV_CMD_NAV_LAND
    assert (drone.stored[-1].param1, drone.stored[-1].param2, drone.stored[-1].param3) == (
        0.0,
        0.0,
        0.0,
    )


@pytest.mark.parametrize(("sent_frame", "returned_frame"), [(0, 6), (3, 5)])
def test_position_frame_does_not_allow_cross_altitude_datum(
    sent_frame: int, returned_frame: int
) -> None:
    item = MissionItem.waypoint(47.0, 8.0, 20.0, frame=sent_frame)

    with pytest.raises(ControllerError, match="frame不符"):
        _upload((item,), lambda fields: fields.__setitem__("frame", returned_frame))


def test_non_position_command_frame_remains_strict() -> None:
    item = MissionItem(command=178, frame=0, param1=1.0)

    with pytest.raises(ControllerError, match="frame不符"):
        _upload((item,), lambda fields: fields.__setitem__("frame", 5))


@pytest.mark.parametrize("parameter", ["param1", "param2", "param3"])
def test_land_unspecified_parameters_normalize_to_zero_but_explicit_values_stay_strict(
    parameter: str,
) -> None:
    default_land = MissionItem(command=MAV_CMD_NAV_LAND, lat=47.0, lon=8.0)
    _upload((default_land,))

    explicit_params: dict[str, Any] = {parameter: 2.0}
    explicit_land = MissionItem(
        command=MAV_CMD_NAV_LAND,
        lat=47.0,
        lon=8.0,
        **explicit_params,
    )
    with pytest.raises(ControllerError, match=parameter):
        _upload(
            (explicit_land,),
            lambda fields: fields.__setitem__(parameter, 0.0),
        )


def test_land_param4_nan_to_zero_is_not_normalized() -> None:
    item = MissionItem(command=MAV_CMD_NAV_LAND, lat=47.0, lon=8.0)

    with pytest.raises(ControllerError, match="param4"):
        _upload((item,), lambda fields: fields.__setitem__("param4", 0.0))


@pytest.mark.parametrize("bad_altitude", [float("nan"), float("inf"), float("-inf")])
def test_positional_readback_rejects_nonfinite_altitude(bad_altitude: float) -> None:
    item = MissionItem.waypoint(47.0, 8.0, 20.0)

    with pytest.raises(ControllerError, match="位置不符"):
        _upload((item,), lambda fields: fields.__setitem__("z", bad_altitude))
