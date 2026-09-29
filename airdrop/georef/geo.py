"""NED ↔ WGS84 坐标换算（纯函数）。

直接移植旧版实现 ``drone/coordinates.py`` 的这两段（计划 4.4 明确要求）：
它已过 pyproj ``Geod`` 大地线真值验证（方位/距离/往返互逆），没有理由重写。

约定
----
``ned`` 是 ``(north, east, down)``，单位米，相对 ``lla_ref`` 定义的 NED 原点；
``lla`` / ``lla_ref`` 是 ``(lon, lat, alt)``，经纬度单位度、高度米——
注意顺序是经度在前（与 pyproj 的 ``always_xy=True`` 一致，也与
:meth:`airdrop.telemetry.models.TelemetrySnapshot` 里的经纬度字段分开存不同，
这里刻意成对传递以免搞混）。

实现方式：把 NED 原点转成 ECEF，再用标准 NED→ECEF 旋转矩阵把 NED 偏移加进去，
最后 ECEF→经纬高。之所以不用"平面近似"（``lat + n/R``），是因为投送距离上
几百米时平面近似的误差会进入米级，而这里没有性能压力。
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
import pyproj

__all__ = ["LLARef", "ned_distance", "ned_to_wgs84", "wgs84_to_ned"]

# ECEF ↔ 大地坐标的坐标框架（模块级建一次，pyproj 的 Transformer 构造不便宜）
_ECEF = pyproj.Proj(proj="geocent", ellps="WGS84", datum="WGS84")
_LLA = pyproj.Proj(proj="latlong", ellps="WGS84", datum="WGS84")
_LLA_TO_ECEF = pyproj.Transformer.from_proj(proj_from=_LLA, proj_to=_ECEF, always_xy=True).transform
_ECEF_TO_LLA = pyproj.Transformer.from_proj(proj_from=_ECEF, proj_to=_LLA, always_xy=True).transform


class LLARef(NamedTuple):
    """NED 原点：``(lon_deg, lat_deg, alt_m)``。"""

    lon_deg: float
    lat_deg: float
    alt_m: float


def _ned_to_ecef_matrix(lon_deg: float, lat_deg: float) -> np.ndarray:
    """标准 NED→ECEF 旋转矩阵（列依次为北/东/地方向的单位向量）。"""
    sin_lat, cos_lat = math.sin(math.radians(lat_deg)), math.cos(math.radians(lat_deg))
    sin_lon, cos_lon = math.sin(math.radians(lon_deg)), math.cos(math.radians(lon_deg))
    return np.array(
        [
            [-sin_lat * cos_lon, -sin_lon, -cos_lat * cos_lon],
            [-sin_lat * sin_lon, cos_lon, -cos_lat * sin_lon],
            [cos_lat, 0.0, -sin_lat],
        ]
    )


def ned_to_wgs84(
    ned: tuple[float, float, float], ref: tuple[float, float, float] | LLARef
) -> tuple[float, float, float]:
    """NED → ``(lon, lat, alt)``。

    ``ecef = ecef_ref + R_ned2ecef @ ned``，R 的列是北/东/地在 ECEF 里的方向。
    """
    north, east, down = (float(ned[0]), float(ned[1]), float(ned[2]))
    lon_ref, lat_ref, alt_ref = (float(ref[0]), float(ref[1]), float(ref[2]))
    x_ref, y_ref, z_ref = _LLA_TO_ECEF(lon_ref, lat_ref, alt_ref)
    rotation = _ned_to_ecef_matrix(lon_ref, lat_ref)
    ecef = np.array([x_ref, y_ref, z_ref]) + rotation @ np.array([north, east, down])
    lon, lat, alt = _ECEF_TO_LLA(float(ecef[0]), float(ecef[1]), float(ecef[2]))
    return float(lon), float(lat), float(alt)


def wgs84_to_ned(
    lon: float,
    lat: float,
    alt: float,
    ref: tuple[float, float, float] | LLARef,
) -> tuple[float, float, float]:
    """``(lon, lat, alt)`` → NED（与 :func:`ned_to_wgs84` 严格互逆）。

    ``ned = R_ned2ecef^T @ (ecef - ecef_ref)``。
    """
    lon_ref, lat_ref, alt_ref = (float(ref[0]), float(ref[1]), float(ref[2]))
    x_ref, y_ref, z_ref = _LLA_TO_ECEF(lon_ref, lat_ref, alt_ref)
    x, y, z = _LLA_TO_ECEF(float(lon), float(lat), float(alt))
    delta = np.array([x - x_ref, y - y_ref, z - z_ref])
    rotation = _ned_to_ecef_matrix(lon_ref, lat_ref)
    north, east, down = rotation.T @ delta
    return float(north), float(east), float(down)


def ned_distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    """两点的三维直线距离（米）。"""
    return math.dist(a, b)
