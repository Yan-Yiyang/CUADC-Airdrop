"""投放记录与实测落点：投弹瞬间的飞机状态 + 事后量出的落点（P12）。

为什么要单独记这一份
--------------------
:class:`~airdrop.config.BallisticsConfig` 里的**质量是称出来的**（台秤/天平实测，不靠反演），
阻力系数才需要投放试验反演——而且数据只识别 ``κ = Cd·A/m``（见
:mod:`airdrop.ballistics.fit`）。反演需要成对的样本：

* **投放侧**——投放那一刻飞机在哪、以什么速度与姿态飞、当时的风是多少。这些数据
  一闪就过去了，事后无法重建，所以必须当场落盘：:class:`DropRecord` 记全位置、
  速度、姿态（欧拉角与四元数**各留一份**）、风、地面高度、NED 原点、当时的弹道
  参数、判据用的前推位置与预测落点。
* **落点侧**——弹体实际落在哪。现场拿手持 GPS 量出来的一般是经纬度，所以本模块按
  该次投放记录里的 **NED 原点**换算成 NED（:func:`match_impacts`），
  **不跨架次混用原点**。测量值以 ``impacts.jsonl`` / ``impacts.csv`` 提供
  （:func:`load_impacts`，模板由 :func:`write_impact_template` 生成）。

⚠ 记录里的落点**不参与实飞**：判据只用模型预测。这份数据的唯一用途是事后反演
（:func:`~airdrop.ballistics.fit.fit_ballistics`），以及复盘时对照"当时预测落点 vs
实际落点"。

姿态为什么也要记
----------------
弹体挂在机身下方，离机点是**挂点**而不是质心；横滚/俯仰会让这个偏移在 NED 里转出
横向分量（0.3m 挂点 + 20° 横滚 ≈ 0.1m）。默认按偏移为零处理（挂点未知就别猜），
但只要在 :class:`~airdrop.ballistics.fit.FitConfig` 里给出
``release_offset_body_m``，正演就会用记录里的姿态把它转到 NED——这是姿态数据在
本模块里的**唯一**用途，别指望它还能修正别的（弹体自身的姿态不影响无控弹道）。
"""

from __future__ import annotations

import csv
import json
import logging
import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from ..config import BallisticsConfig
from ..georef import LLARef, euler_to_matrix, quaternion_to_matrix, wgs84_to_ned
from ..telemetry.models import TelemetrySnapshot
from .model import ZERO_WIND, BallisticsModel

LOGGER = logging.getLogger(__name__)

__all__ = [
    "DROPS_NAME",
    "IMPACTS_CSV_NAME",
    "IMPACTS_JSONL_NAME",
    "DropRecord",
    "DropSample",
    "ImpactMeasurement",
    "append_drop",
    "attitude_matrix",
    "load_drops",
    "load_impacts",
    "match_impacts",
    "predict_record_impact",
    "release_conditions",
    "resolve_wind",
    "write_impact_template",
]

#: 飞行目录里的投放记录文件名（六件套之一，由 :class:`~airdrop.record.FlightRecorder` 写）
DROPS_NAME = "drops.jsonl"
#: 实测落点的默认文件名（与 ``drops.jsonl`` 同目录，手工填写）
IMPACTS_JSONL_NAME = "impacts.jsonl"
IMPACTS_CSV_NAME = "impacts.csv"

#: 测量文件里代表"落点"的字段名（填了经纬度就按经纬度，否则按 NED）
_LONLAT_KEYS = ("lat_deg", "lon_deg")
_NED_KEYS = ("north_m", "east_m")


def _vec3(values: Sequence[float] | None) -> tuple[float, float, float] | None:
    """把 JSON 读回来的三元组转成 ``tuple[float, float, float]``；None 原样返回。"""
    if values is None:
        return None
    if len(values) != 3:
        raise ValueError(f"需要 3 维矢量，收到 {values!r}")
    return (float(values[0]), float(values[1]), float(values[2]))


# ----------------------------------------------------------------------
# 投放记录
# ----------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class DropRecord:
    """一次投放的**原始状态**（投放侧数据，飞后不可重建）。

    ``index`` 是**同一架次内**的投放序号（1 起）。``timestamp`` 是投放指令时刻的
    本地墙钟——与 ``events.jsonl`` / ``telemetry.jsonl`` 同一条时间轴。

    ``delay_s`` 与 ``release_position_ned`` 是**判据当时用的**前推量与预测落点
    （:class:`~airdrop.ballistics.release.ReleaseJudge` 的一阶前推），留着是为了对照
    "当时怎么算的"；反演时用哪个延迟由
    :class:`~airdrop.ballistics.fit.FitConfig` 决定，默认沿用这里的值。
    """

    index: int
    timestamp: float
    position_ned: tuple[float, float, float]
    velocity_ned: tuple[float, float, float]
    ground_z: float = 0.0
    roll_deg: float | None = None
    pitch_deg: float | None = None
    yaw_deg: float | None = None
    quaternion_wxyz: tuple[float, float, float, float] | None = None
    wind_ned: tuple[float, float, float] | None = None
    origin: LLARef | None = None
    target_ned: tuple[float, float, float] | None = None
    reason: str = ""
    delay_s: float = 0.0
    release_position_ned: tuple[float, float, float] | None = None
    predicted_impact_ned: tuple[float, float, float] | None = None
    predicted_flight_time_s: float | None = None
    predicted_error_m: float | None = None
    ballistics: BallisticsConfig = field(default_factory=BallisticsConfig)
    #: 实测落点（NED）——**事后**才有；也可以只写在测量文件里，由 :func:`match_impacts` 合并
    impact_ned: tuple[float, float, float] | None = None
    impact_source: str = ""

    # ------------------------------------------------------------------
    @property
    def euler_deg(self) -> tuple[float, float, float] | None:
        """``(roll, pitch, yaw)``；三个角缺一个就返回 None（不拼半个姿态）。"""
        if self.roll_deg is None or self.pitch_deg is None or self.yaw_deg is None:
            return None
        return (float(self.roll_deg), float(self.pitch_deg), float(self.yaw_deg))

    @property
    def speed_m_s(self) -> float:
        return math.sqrt(sum(value * value for value in self.velocity_ned))

    @property
    def horizontal_speed_m_s(self) -> float:
        return math.hypot(self.velocity_ned[0], self.velocity_ned[1])

    @property
    def height_agl_m(self) -> float:
        """离地高度（正的米数）——投放试验报表里最直观的一列。"""
        return float(self.ground_z - self.position_ned[2])

    @property
    def ground_altitude_m(self) -> float | None:
        """地面平面的海拔（AMSL）= 原点海拔 − ``ground_z``；缺原点时返回 None。

        密度基准用它（见 :meth:`BallisticsModel.predict_impact` 的
        ``ground_altitude_m``）：地面点与原点同高时 ground_z=0，取的就是原点海拔；
        配了地面点则正好是地面点海拔。**不要**用 0 顶替缺失的原点——那等于把地面
        当海平面，高原上会明显高估空气密度。
        """
        if self.origin is None:
            return None
        return float(self.origin.alt_m) - float(self.ground_z)

    @property
    def heading_deg(self) -> float | None:
        """航迹方位（自北顺时针，度）；水平速度太小就返回 None。"""
        north, east = self.velocity_ned[0], self.velocity_ned[1]
        if math.hypot(north, east) < 1e-6:
            return None
        return math.degrees(math.atan2(east, north)) % 360.0

    # ------------------------------------------------------------------
    def with_impact(
        self,
        impact_ned: Sequence[float],
        *,
        source: str = "manual",
    ) -> "DropRecord":
        """回填实测落点（返回新对象；本类是 frozen 的）。"""
        return replace(self, impact_ned=_vec3(impact_ned), impact_source=str(source))

    def as_dict(self) -> dict[str, Any]:
        """落盘/进事件日志用的扁平字典（数值**不四舍五入**——这是原始数据）。"""
        data: dict[str, Any] = {
            "index": int(self.index),
            "timestamp": float(self.timestamp),
            "position_ned": [float(v) for v in self.position_ned],
            "velocity_ned": [float(v) for v in self.velocity_ned],
            "ground_z": float(self.ground_z),
            "reason": self.reason,
            "delay_s": float(self.delay_s),
            "ballistics": asdict(self.ballistics),
        }
        euler = self.euler_deg
        if euler is not None:
            data["attitude_deg"] = {
                "roll": euler[0],
                "pitch": euler[1],
                "yaw": euler[2],
            }
        if self.quaternion_wxyz is not None:
            data["quaternion_wxyz"] = [float(v) for v in self.quaternion_wxyz]
        if self.wind_ned is not None:
            data["wind_ned"] = [float(v) for v in self.wind_ned]
        if self.origin is not None:
            data["origin"] = {
                "lon_deg": float(self.origin.lon_deg),
                "lat_deg": float(self.origin.lat_deg),
                "alt_m": float(self.origin.alt_m),
            }
        if self.target_ned is not None:
            data["target_ned"] = [float(v) for v in self.target_ned]
        if self.release_position_ned is not None:
            data["release_position_ned"] = [float(v) for v in self.release_position_ned]
        if self.predicted_impact_ned is not None:
            data["predicted_impact_ned"] = [float(v) for v in self.predicted_impact_ned]
        if self.predicted_flight_time_s is not None:
            data["predicted_flight_time_s"] = float(self.predicted_flight_time_s)
        if self.predicted_error_m is not None:
            data["predicted_error_m"] = float(self.predicted_error_m)
        if self.impact_ned is not None:
            data["impact_ned"] = [float(v) for v in self.impact_ned]
            data["impact_source"] = self.impact_source
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DropRecord":
        """从 :meth:`as_dict` 的字典还原（缺字段按 None/默认值，不猜数值）。"""
        euler = data.get("attitude_deg") or None
        origin = data.get("origin") or None
        quaternion = data.get("quaternion_wxyz")
        if quaternion is not None and len(quaternion) != 4:
            raise ValueError(f"quaternion_wxyz 需要 4 个分量: {quaternion!r}")
        raw_ballistics = data.get("ballistics") or {}
        base = BallisticsConfig()
        ballistics = BallisticsConfig(
            mass_kg=float(raw_ballistics.get("mass_kg", base.mass_kg)),
            drag_coefficient=float(raw_ballistics.get("drag_coefficient", base.drag_coefficient)),
            cross_area_m2=float(raw_ballistics.get("cross_area_m2", base.cross_area_m2)),
            gravity=float(raw_ballistics.get("gravity", base.gravity)),
            rk_dt=float(raw_ballistics.get("rk_dt", base.rk_dt)),
            air_density_isa=bool(raw_ballistics.get("air_density_isa", base.air_density_isa)),
            wind_source=str(raw_ballistics.get("wind_source", base.wind_source)),
        )
        return cls(
            index=int(data["index"]),
            timestamp=float(data.get("timestamp", 0.0)),
            position_ned=_vec3(data["position_ned"]) or (0.0, 0.0, 0.0),
            velocity_ned=_vec3(data["velocity_ned"]) or (0.0, 0.0, 0.0),
            ground_z=float(data.get("ground_z", 0.0)),
            roll_deg=None if euler is None else float(euler["roll"]),
            pitch_deg=None if euler is None else float(euler["pitch"]),
            yaw_deg=None if euler is None else float(euler["yaw"]),
            quaternion_wxyz=(
                None
                if quaternion is None
                else (
                    float(quaternion[0]),
                    float(quaternion[1]),
                    float(quaternion[2]),
                    float(quaternion[3]),
                )
            ),
            wind_ned=_vec3(data.get("wind_ned")),
            origin=(
                None
                if origin is None
                else LLARef(
                    lon_deg=float(origin["lon_deg"]),
                    lat_deg=float(origin["lat_deg"]),
                    alt_m=float(origin["alt_m"]),
                )
            ),
            target_ned=_vec3(data.get("target_ned")),
            reason=str(data.get("reason", "")),
            delay_s=float(data.get("delay_s", 0.0)),
            release_position_ned=_vec3(data.get("release_position_ned")),
            predicted_impact_ned=_vec3(data.get("predicted_impact_ned")),
            predicted_flight_time_s=(
                None
                if data.get("predicted_flight_time_s") is None
                else float(data["predicted_flight_time_s"])
            ),
            predicted_error_m=(
                None if data.get("predicted_error_m") is None else float(data["predicted_error_m"])
            ),
            ballistics=ballistics,
            impact_ned=_vec3(data.get("impact_ned")),
            impact_source=str(data.get("impact_source", "")),
        )

    @classmethod
    def from_snapshot(  # noqa: PLR0913 - 一次性把投放瞬间要记的量全收进来，拆开会更难对账
        cls,
        snapshot: TelemetrySnapshot,
        *,
        index: int,
        ground_z: float = 0.0,
        wind_ned: Sequence[float] | None = None,
        origin: LLARef | None = None,
        target_ned: Sequence[float] | None = None,
        reason: str = "",
        delay_s: float = 0.0,
        release_position_ned: Sequence[float] | None = None,
        predicted_impact_ned: Sequence[float] | None = None,
        predicted_flight_time_s: float | None = None,
        predicted_error_m: float | None = None,
        ballistics: BallisticsConfig | None = None,
        timestamp: float | None = None,
    ) -> "DropRecord":
        """从遥测快照 + 判据结果组装一条记录。

        位置/速度缺一个就抛 ``ValueError``——**没有状态的投放记录毫无用处**，
        与其写一条查不出原因的残缺记录，不如当场报错（调用方按事件失败处理）。
        """
        if snapshot.north_m is None or snapshot.east_m is None or snapshot.down_m is None:
            raise ValueError("快照里没有 NED 位置，无法记录投放")
        if snapshot.vx_m_s is None or snapshot.vy_m_s is None or snapshot.vz_m_s is None:
            raise ValueError("快照里没有 NED 速度，无法记录投放")
        quaternion = None
        if None not in (
            snapshot.quaternion_w,
            snapshot.quaternion_x,
            snapshot.quaternion_y,
            snapshot.quaternion_z,
        ):
            quaternion = (
                float(snapshot.quaternion_w),  # type: ignore[arg-type]
                float(snapshot.quaternion_x),  # type: ignore[arg-type]
                float(snapshot.quaternion_y),  # type: ignore[arg-type]
                float(snapshot.quaternion_z),  # type: ignore[arg-type]
            )
        return cls(
            index=int(index),
            timestamp=float(snapshot.timestamp if timestamp is None else timestamp),
            position_ned=(
                float(snapshot.north_m),
                float(snapshot.east_m),
                float(snapshot.down_m),
            ),
            velocity_ned=(
                float(snapshot.vx_m_s),
                float(snapshot.vy_m_s),
                float(snapshot.vz_m_s),
            ),
            ground_z=float(ground_z),
            roll_deg=None if snapshot.roll_deg is None else float(snapshot.roll_deg),
            pitch_deg=None if snapshot.pitch_deg is None else float(snapshot.pitch_deg),
            yaw_deg=None if snapshot.yaw_deg is None else float(snapshot.yaw_deg),
            quaternion_wxyz=quaternion,
            wind_ned=_vec3(wind_ned),
            origin=origin,
            target_ned=_vec3(target_ned),
            reason=str(reason),
            delay_s=float(delay_s),
            release_position_ned=_vec3(release_position_ned),
            predicted_impact_ned=_vec3(predicted_impact_ned),
            predicted_flight_time_s=(
                None if predicted_flight_time_s is None else float(predicted_flight_time_s)
            ),
            predicted_error_m=(None if predicted_error_m is None else float(predicted_error_m)),
            ballistics=ballistics or BallisticsConfig(),
        )


# ----------------------------------------------------------------------
# 姿态 / 正演输入
# ----------------------------------------------------------------------
def attitude_matrix(record: DropRecord) -> np.ndarray | None:
    """投放瞬间的**机体→NED** 旋转矩阵；四元数优先、缺了退欧拉角、都没有给 None。

    与 :mod:`airdrop.georef.project` 同一约定（3-2-1，``Rz·Ry·Rx``），直接复用那里的
    构造函数，免得两处各写一份转出不同的符号。
    """
    quaternion = record.quaternion_wxyz
    if quaternion is not None:
        try:
            return quaternion_to_matrix(*quaternion)
        except ValueError:
            LOGGER.warning("投放记录 #%s 的四元数模长为零，改用欧拉角", record.index)
    euler = record.euler_deg
    if euler is not None:
        return euler_to_matrix(*euler)
    return None


def release_conditions(
    record: DropRecord,
    *,
    delay_s: float | None = None,
    offset_body_m: Sequence[float] | None = None,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """正演的初始条件 ``(位置, 速度)``（NED）。

    两个修正项，都是**可关的**、语义写死：

    * ``delay_s``——"记录状态 → 真正离机"的时间，按判据同款**一阶前推**
      （``位置 += 速度 × delay``，不做二次项，理由见
      :mod:`airdrop.ballistics.release`）。默认沿用记录里的 ``delay_s``
      （= 当时判据用的值）。
    * ``offset_body_m``——弹体挂点在**机体系**下的偏移（前-右-下）。用记录里的姿态
      转到 NED 后加到位置上；给了偏移却没有姿态就**报错**（不假装姿态是水平的）。
      默认零偏移，即"离机点按质心算"。
    """
    delay = float(record.delay_s if delay_s is None else delay_s)
    position = record.position_ned
    velocity = record.velocity_ned
    p0: tuple[float, float, float] = (
        position[0] + velocity[0] * delay,
        position[1] + velocity[1] * delay,
        position[2] + velocity[2] * delay,
    )
    if offset_body_m is None or not any(float(v) for v in offset_body_m):
        return p0, velocity
    rotation = attitude_matrix(record)
    if rotation is None:
        raise ValueError(
            f"投放记录 #{record.index} 没有姿态，无法把挂点偏移转到 NED"
            "（要么去掉 offset_body_m，要么补上姿态）"
        )
    offset_ned = rotation @ np.asarray(offset_body_m, dtype=np.float64).reshape(3)
    shifted: tuple[float, float, float] = (
        p0[0] + float(offset_ned[0]),
        p0[1] + float(offset_ned[1]),
        p0[2] + float(offset_ned[2]),
    )
    return (shifted, velocity)


def resolve_wind(record: DropRecord, *, scale: float = 1.0) -> tuple[float, float, float]:
    """这一条记录用于正演的风（NED，m/s）；没有风估计就按零风（**不猜**）。

    ``scale`` 是反演用的比例因子（默认 1.0 = 照用飞控的风估计）。缺风时不做
    "按比例放大零风"这种自欺欺人的事——返回的就是 :data:`~airdrop.ballistics.model.ZERO_WIND`。
    """
    if record.wind_ned is None:
        return ZERO_WIND
    wind = record.wind_ned
    factor = float(scale)
    return (
        float(wind[0]) * factor,
        float(wind[1]) * factor,
        float(wind[2]) * factor,
    )


def predict_record_impact(
    record: DropRecord,
    model: BallisticsModel,
    *,
    delay_s: float | None = None,
    offset_body_m: Sequence[float] | None = None,
    wind_scale: float = 1.0,
    ground_z: float | None = None,
    ground_altitude_m: float | None = None,
):
    """用 ``model`` 正演这一条记录的落点（返回 :class:`~airdrop.ballistics.model.Impact`）。

    参数含义与 :func:`release_conditions` / :func:`resolve_wind` 一一对应；反演
    （:mod:`airdrop.ballistics.fit`）与报表都走这一个入口，免得两处各写一遍初始条件。

    ``ground_altitude_m`` 缺省取 :attr:`DropRecord.ground_altitude_m`（原点海拔 −
    ``ground_z``，即记录里的 GPS 高度基准）；记录没有原点时按 0 处理（视作地面海拔为
    0）——此时高原站点会有密度偏差，反演报告里应把该条记录的原点补齐。
    """
    baseline = record.ground_altitude_m if ground_altitude_m is None else float(ground_altitude_m)
    position, velocity = release_conditions(record, delay_s=delay_s, offset_body_m=offset_body_m)
    return model.predict_impact(
        position,
        velocity,
        ground_z=float(record.ground_z if ground_z is None else ground_z),
        ground_altitude_m=0.0 if baseline is None else float(baseline),
        wind=resolve_wind(record, scale=wind_scale),
    )


# ----------------------------------------------------------------------
# 落点测量
# ----------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class ImpactMeasurement:
    """测量文件里的一条落点：或给经纬度、或给 NED（两者都缺 = 还没量，跳过）。"""

    index: int
    lat_deg: float | None = None
    lon_deg: float | None = None
    alt_m: float | None = None
    north_m: float | None = None
    east_m: float | None = None
    down_m: float | None = None
    note: str = ""

    @property
    def measured(self) -> bool:
        """是否已有可用的落点（经纬度齐 或 NED 水平分量齐）。"""
        return self.has_lonlat or self.has_ned

    @property
    def has_lonlat(self) -> bool:
        return self.lat_deg is not None and self.lon_deg is not None

    @property
    def has_ned(self) -> bool:
        return self.north_m is not None and self.east_m is not None

    @property
    def source(self) -> str:
        if self.has_lonlat:
            return "gps"
        if self.has_ned:
            return "ned"
        return ""

    def validate(self, record: DropRecord) -> None:
        """填了一半就**显式报错**（比默默按另一套坐标算出一堆废数好）。"""
        if self.has_lonlat or self.has_ned:
            return
        filled = any(
            value is not None
            for value in (
                self.lat_deg,
                self.lon_deg,
                self.alt_m,
                self.north_m,
                self.east_m,
                self.down_m,
            )
        )
        if filled:
            raise ValueError(
                f"测量文件里 index={self.index} 的落点填了一半：要么给 "
                "lat_deg+lon_deg（可带 alt_m），要么给 north_m+east_m（可带 down_m）"
            )

    def to_ned(self, record: DropRecord) -> tuple[float, float, float]:
        """换算成 NED（用**该条记录**的原点；缺原点又给的是经纬度就报错）。"""
        lon_deg, lat_deg = self.lon_deg, self.lat_deg
        if lon_deg is not None and lat_deg is not None:
            if record.origin is None:
                raise ValueError(
                    f"index={self.index} 的落点给的是经纬度，但投放记录里没有 NED 原点，"
                    "无法换算——补上 origin，或直接给 north_m/east_m"
                )
            lon, lat = float(lon_deg), float(lat_deg)
            if self.alt_m is None:
                # 只量了平面位置：高度按**原点地面高度**算。误差是二阶的——偏离原点
                # 法线的高度误差要在几百米外才凑得出厘米级水平偏差（250m 处 500m
                # 高度误差 ≈ 2cm），所以不值得为它要求必须量高度。
                north, east, _ = wgs84_to_ned(lon, lat, float(record.origin.alt_m), record.origin)
                return (north, east, float(record.ground_z))
            north, east, down = wgs84_to_ned(lon, lat, float(self.alt_m), record.origin)
            return (north, east, down)
        north_m, east_m = self.north_m, self.east_m
        if north_m is not None and east_m is not None:
            down = float(record.ground_z if self.down_m is None else self.down_m)
            return (float(north_m), float(east_m), down)
        raise ValueError(f"index={self.index} 没有实测落点")


@dataclass(frozen=True, slots=True)
class DropSample:
    """反演的输入单元：一条投放记录 + 一个**实测**落点。

    ``label`` 由调用方给（跨架次时用 ``"<飞行目录名>#<序号>"``），只在报表里用；
    ``index`` 始终是记录本身的序号。
    """

    record: DropRecord
    impact_ned: tuple[float, float, float]
    impact_source: str = ""
    label: str = ""

    @property
    def index(self) -> int:
        return self.record.index

    @property
    def name(self) -> str:
        return self.label or f"#{self.index}"


# ----------------------------------------------------------------------
# 读写
# ----------------------------------------------------------------------
def append_drop(path: str | Path, record: DropRecord) -> None:
    """追加一条投放记录到 JSONL（写完即 flush；一行一次投放）。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record.as_dict(), ensure_ascii=False)
    with open(target, "a", encoding="utf-8") as handle:
        handle.write(line + "\n")
        handle.flush()


def load_drops(path: str | Path) -> tuple[DropRecord, ...]:
    """读回投放记录（``drops.jsonl``）；目录则自动找目录下的 :data:`DROPS_NAME`。

    空行跳过；坏行**显式报错并带上行号**——半条记录悄悄丢掉会让反演少样本而无人察觉。
    """
    target = Path(path)
    if target.is_dir():
        target = target / DROPS_NAME
    if not target.exists():
        raise FileNotFoundError(f"找不到投放记录文件: {target}")
    records: list[DropRecord] = []
    for number, line in enumerate(target.read_text(encoding="utf-8").splitlines(), 1):
        text = line.strip()
        if not text:
            continue
        try:
            records.append(DropRecord.from_dict(json.loads(text)))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{target} 第 {number} 行不是合法的投放记录：{exc}") from exc
    return tuple(records)


def _iter_impact_rows(path: Path) -> Iterator[Mapping[str, Any]]:
    """按后缀选 JSONL / CSV 解析，统一产出字典（CSV 的空串按 None 处理）。"""
    if path.suffix.lower() == ".csv":
        with open(path, encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                yield {
                    key: (None if value is None or str(value).strip() == "" else value)
                    for key, value in row.items()
                    if key is not None
                }
        return
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        text = line.strip()
        if not text:
            continue
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path} 第 {number} 行不是合法 JSON：{exc}") from exc
        if not isinstance(payload, Mapping):
            raise ValueError(f"{path} 第 {number} 行应当是 JSON 对象")
        yield payload


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return float(text)


def load_impacts(path: str | Path) -> tuple[ImpactMeasurement, ...]:
    """读实测落点（``.jsonl`` 或 ``.csv`` 均支持）。

    字段（至少要能凑出一套完整坐标，缺的行会被当成"还没量"跳过）：

    * ``index``（必需）——对应投放记录的序号；
    * 经纬度：``lat_deg`` / ``lon_deg``（``alt_m`` 可选，缺省按该次 ``ground_z``）；
    * 或 NED：``north_m`` / ``east_m``（``down_m`` 可选，缺省按该次 ``ground_z``）。
    """
    target = Path(path)
    if not target.exists():
        raise FileNotFoundError(f"找不到落点测量文件: {target}")
    measurements: list[ImpactMeasurement] = []
    for number, row in enumerate(_iter_impact_rows(target), 1):
        if "index" not in row or row["index"] is None:
            raise ValueError(f"{target} 第 {number} 行缺少 index（对应投放序号）")
        try:
            measurement = ImpactMeasurement(
                index=int(float(str(row["index"]))),
                lat_deg=_optional_float(row.get("lat_deg")),
                lon_deg=_optional_float(row.get("lon_deg")),
                alt_m=_optional_float(row.get("alt_m")),
                north_m=_optional_float(row.get("north_m")),
                east_m=_optional_float(row.get("east_m")),
                down_m=_optional_float(row.get("down_m")),
                note=str(row.get("note") or ""),
            )
        except ValueError as exc:
            raise ValueError(f"{target} 第 {number} 行的落点字段不是数字：{exc}") from exc
        measurements.append(measurement)
    return tuple(measurements)


def match_impacts(
    records: Sequence[DropRecord],
    measurements: Sequence[ImpactMeasurement] = (),
) -> tuple[DropSample, ...]:
    """把测量落点配到投放记录上，产出反演样本。

    * 记录里已经带了 ``impact_ned`` 的，优先用记录里的（.jsonl 直接手改也行）；
    * 其余按 ``index`` 找测量：**同一个序号出现两次就报错**，找不到就跳过
      （"这次没量"是常态，不算错误，缺哪些由调用方自己对账）；
    * 序号在记录里不存在 → **报错**（多半是抄错了行，静默忽略最危险）。
    """
    by_index: dict[int, ImpactMeasurement] = {}
    for measurement in measurements:
        if measurement.index in by_index:
            raise ValueError(f"测量文件里 index={measurement.index} 出现了不止一次")
        by_index[measurement.index] = measurement

    known = {record.index for record in records}
    unknown = sorted(set(by_index) - known)
    if unknown:
        raise ValueError(
            f"测量里有记录中不存在的投放序号 {unknown}；本架次的序号是 {sorted(known)}"
        )

    samples: list[DropSample] = []
    for record in records:
        if record.impact_ned is not None:
            samples.append(
                DropSample(
                    record=record,
                    impact_ned=record.impact_ned,
                    impact_source=record.impact_source or "record",
                )
            )
            continue
        measurement = by_index.get(record.index)
        if measurement is None:
            continue
        # 先校验再判断"量没量"：填了一半的行必须报错，不能被当成"还没量"跳过
        measurement.validate(record)
        if not measurement.measured:
            continue
        samples.append(
            DropSample(
                record=record,
                impact_ned=measurement.to_ned(record),
                impact_source=measurement.source,
            )
        )
    return tuple(samples)


def write_impact_template(path: str | Path, records: Sequence[DropRecord]) -> Path:
    """按投放记录生成待填的落点模板（JSONL），现场量完直接改这个文件。

    每行两套坐标都留出来（填哪套都行，见 :func:`load_impacts`），全空表示"这次没量"。
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for record in records:
        lines.append(
            json.dumps(
                {
                    "index": record.index,
                    "lat_deg": None,
                    "lon_deg": None,
                    "alt_m": None,
                    "north_m": None,
                    "east_m": None,
                    "down_m": None,
                    "timestamp": record.timestamp,
                    "note": "填实测落点：经纬度（推荐）或 NED；全空=这次没量",
                },
                ensure_ascii=False,
            )
        )
    target.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return target
