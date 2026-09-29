"""二次阻力弹道：RK4 定步长积分 → 落点与飞行时间（计划 4.6）。

方程（NED，z 向下为正）::

    m·dv/dt = m·g − ½·ρ(z)·Cd·A·|v − w|·(v − w)

* 重力只有 ``+z`` 分量（往下掉）；
* 阻力相对空气算（``v − w``，``w`` 是风的速度矢量）——这是风会平移落点的唯一原因；
* **密度基准是真实海拔（GPS/原点海拔），不是"地面=海平面"**：``isa_air_density()``
  收的是海拔高度，调用方用 ``ground_altitude_m + 离地高度`` 得到它。
  ``air_density_isa=True`` 时逐级按海拔算 ρ；``False``（默认）在**初始（投放）海拔**
  上算一次、全弹道共用（比海平面常密度更接近真实，见下）。

两处刻意的取舍（都写在代码里免得被当 bug）
------------------------------------------
1. 默认（``air_density_isa=False``）用**投放海拔处的常密度**：一次 ISA 求值折进
   阻力系数。任务高度约 20m 时与逐级变化的差别只有 ~0.2%，而逐级求值要贵 26~30%；
   高海拔站点或大落差投放可打开 ``air_density_isa``。
2. 落点用线性插值定在穿越瞬间：定步长 5ms 直接取"第一个 z ≥ ground_z 的步"会带来
   最多一步的水平误差（10 m/s 平飞时 5cm）；穿越点插值后与解析解对得上。
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass

from ..config import BallisticsConfig
from ..telemetry.models import TelemetrySnapshot

LOGGER = logging.getLogger(__name__)

__all__ = [
    "ZERO_WIND",
    "BallisticsModel",
    "Impact",
    "as_float_triple",
    "isa_air_density",
    "wind_from_snapshot",
]

#: 无风（没有风估计时的降级值）
ZERO_WIND: tuple[float, float, float] = (0.0, 0.0, 0.0)

#: 海平面空气密度（kg/m³，ISA）
SEA_LEVEL_DENSITY = 1.225
#: ISA 对流层密度公式的系数（h 单位米）
_ISA_COEFF = 2.25577e-5
_ISA_EXP = 4.2559
#: 积分时长上限（秒）——防止"一直没落地"的病态输入把主循环拖死
MAX_FLIGHT_TIME_S = 120.0


def as_float_triple(
    values: tuple[float | None, float | None, float | None],
) -> tuple[float, float, float] | None:
    """快照里取三个分量：任一缺失就返回 None，**不拿 0 顶上**。

    飞控快照的数值字段都是 ``float | None``（还没收到值就是 None）。这里的取舍与
    :func:`wind_from_snapshot` 一致：缺值如实返回 None，让调用方决定怎么降级。
    """
    first, second, third = values
    if first is None or second is None or third is None:
        return None
    return (float(first), float(second), float(third))


def isa_air_density(altitude_m: float) -> float:
    """ISA 标准大气下 ``altitude_m``（**海拔**，米）处的空气密度（kg/m³）。

    ⚠ 参数是**海拔高度**（AMSL），不是离地高度：地面在 1500m 高原时，地面处的
    密度是 ``isa_air_density(1500)`` ≈ 1.056，而不是 1.225。调用方拿
    ``ground_altitude_m + 离地高度`` 得到海拔。负值（低于海平面）按 0 处理。
    """
    altitude = max(float(altitude_m), 0.0)
    return SEA_LEVEL_DENSITY * (1.0 - _ISA_COEFF * altitude) ** _ISA_EXP


def wind_from_snapshot(
    snapshot: TelemetrySnapshot,
    config: BallisticsConfig,
) -> tuple[float, float, float] | None:
    """按 ``config.wind_source`` 取风矢量（NED，m/s）。

    * ``"zero"``：直接给 :data:`ZERO_WIND`；
    * ``"telemetry"``：读快照里的飞控风估计（``wind_north/east/down_m_s``）。
      取不到就返回 None——这里刻意不悄悄给 0，否则"没有风估计"与"风确实是 0"
      就分不出来了；降级与记日志交给调用方（:class:`~airdrop.ballistics.release.ReleaseJudge`
      会记一次日志再按零风算）；
    * 其它取值：显式报错，不做隐式回退。
    """
    source = config.wind_source
    if source == "zero":
        return ZERO_WIND
    if source != "telemetry":
        raise ValueError(f"未知的 wind_source: {source!r}（只支持 telemetry / zero）")
    values = (snapshot.wind_north_m_s, snapshot.wind_east_m_s, snapshot.wind_down_m_s)
    return as_float_triple(values)


@dataclass(frozen=True, slots=True)
class Impact:
    """一次落点预测的结果；``ok=False`` 时 ``ned`` 为 None、``reason`` 说明原因。"""

    ok: bool
    ned: tuple[float, float, float] | None
    flight_time_s: float
    speed_m_s: float = 0.0
    reason: str = ""

    @property
    def north_m(self) -> float | None:
        return None if self.ned is None else self.ned[0]

    @property
    def east_m(self) -> float | None:
        return None if self.ned is None else self.ned[1]


class BallisticsModel:
    """无控投放物的弹道预测（二次阻力 + RK4 定步长）。

    参数全部来自 :class:`~airdrop.config.BallisticsConfig`——质量按实测称重填入，
    ``drag_coefficient`` 靠投放试验反演（占位值 0.6 只用于跑通链路），迎风面积是几何量。
    注意方程里 ``Cd``、``A``、``m`` 只以 ``κ = Cd·A/m`` 的形式出现（``self._k``）。
    """

    def __init__(self, config: BallisticsConfig) -> None:
        self._config = config
        if config.mass_kg <= 0:
            raise ValueError(f"质量必须为正: {config.mass_kg}")
        if config.rk_dt <= 0:
            raise ValueError(f"积分步长必须为正: {config.rk_dt}")
        #: 阻力系数 k = ½·Cd·A/m（1/m）
        self._k = 0.5 * config.drag_coefficient * config.cross_area_m2 / config.mass_kg

    @property
    def config(self) -> BallisticsConfig:
        return self._config

    def drag_factor(
        self,
        velocity_rel: Sequence[float],
        height_m: float,
        *,
        ground_altitude_m: float = 0.0,
    ) -> float:
        """``height_m``（离地）处的阻力加速度系数 ``½·ρ·Cd·A/m·|v_rel|``（1/s）。

        这是**点查询**：密度取该处的 ISA 值（海拔 = ``ground_altitude_m + height_m``），
        与 ``air_density_isa`` 开关无关——开关只决定正演里密度是否随高度逐级变化。
        """
        density = isa_air_density(ground_altitude_m + height_m)
        speed = math.sqrt(sum(float(v) ** 2 for v in velocity_rel))
        return self._k * density * speed

    def terminal_velocity(self, height_m: float = 0.0, *, ground_altitude_m: float = 0.0) -> float:
        """``height_m``（离地）处的终端速度 ``v_t = √(2mg / (ρ·Cd·A))``（m/s）。

        同样是点查询、ISA 密度（海拔 = ``ground_altitude_m + height_m``）。
        关闭 ``air_density_isa`` 的常密度正演取的是**投放高度**处的值，
        所以在投放高度上查询它，就是那条常密度弹道的终端速度。
        """
        density = isa_air_density(ground_altitude_m + height_m)
        area_term = self._config.drag_coefficient * self._config.cross_area_m2
        if area_term <= 0:
            return float("inf")  # 无阻力：不存在终端速度
        return math.sqrt(2.0 * self._config.mass_kg * self._config.gravity / (density * area_term))

    # ------------------------------------------------------------------
    # 积分
    # ------------------------------------------------------------------
    def _acceleration(
        self,
        velocity: Sequence[float],
        wind: Sequence[float],
        altitude_m: float,
    ) -> tuple[float, float, float]:
        """当前加速度：重力 + 阻力（密度取 ``altitude_m`` 处的 ISA 值）。

        这是"密度随高度逐级变化"那条路径用的；常密度路径直接用
        :meth:`_acceleration_with_scale`，不经过这里。
        """
        return self._acceleration_with_scale(
            velocity, wind, self._k * isa_air_density(altitude_m), self._config.gravity
        )

    @staticmethod
    def _acceleration_with_scale(
        velocity: Sequence[float],
        wind: Sequence[float],
        drag_scale: float,
        gravity: float,
    ) -> tuple[float, float, float]:
        """用固定密度系数计算加速度，供 RK4 热路径复用。"""
        rel_x = float(velocity[0]) - float(wind[0])
        rel_y = float(velocity[1]) - float(wind[1])
        rel_z = float(velocity[2]) - float(wind[2])
        speed = math.sqrt(rel_x * rel_x + rel_y * rel_y + rel_z * rel_z)
        factor = drag_scale * speed
        return (
            -factor * rel_x,
            -factor * rel_y,
            gravity - factor * rel_z,
        )

    def predict_impact(
        self,
        position_ned: Sequence[float],
        velocity_ned: Sequence[float],
        *,
        ground_z: float = 0.0,
        ground_altitude_m: float = 0.0,
        wind: Sequence[float] = (0.0, 0.0, 0.0),
    ) -> Impact:
        """从 ``position_ned``/``velocity_ned`` 起积分，返回落点与飞行时间。

        终止条件 ``z ≥ ground_z``；穿越点线性插值。落不到地（时长超过
        :data:`MAX_FLIGHT_TIME_S`）或起点已在地面以下都返回 ``ok=False`` 并说明原因
        ——不返回一个编出来的落点。

        ``ground_altitude_m`` 是**地面平面的海拔**（AMSL，来自 GPS 原点/地面点）：
        它是密度的基准——``air_density_isa=True`` 时按 ``ground_altitude_m + 离地
        高度`` 逐级求值；``False`` 时在**初始（投放）海拔** ``ground_altitude_m +
        初始离地高度`` 上求一次常密度，全弹道共用。
        """
        if len(position_ned) != 3 or len(velocity_ned) != 3:
            raise ValueError("position_ned / velocity_ned 必须是 3 维 NED 矢量")
        p = [float(v) for v in position_ned]
        v = [float(v) for v in velocity_ned]
        w = [float(v) for v in wind]
        if p[2] >= ground_z:
            return Impact(False, None, 0.0, 0.0, reason="below_ground")

        dt = self._config.rk_dt
        baseline = float(ground_altitude_m)
        if self._config.air_density_isa:
            drag_scale = None  # 逐级按海拔密度算
        else:
            # 常密度：按初始（投放）海拔算一次；比"海平面常密度"更接近真实
            initial_height = max(ground_z - p[2], 0.0)
            drag_scale = self._k * isa_air_density(baseline + initial_height)
        elapsed = 0.0
        while elapsed < MAX_FLIGHT_TIME_S:
            height = max(ground_z - p[2], 0.0)
            # RK4：k1..k4 分别是位置与速度的斜率
            if drag_scale is None:
                a1 = self._acceleration(v, w, baseline + height)
            else:
                a1 = self._acceleration_with_scale(v, w, drag_scale, self._config.gravity)
            v2 = [v[i] + 0.5 * dt * a1[i] for i in range(3)]
            height2 = max(ground_z - (p[2] + 0.5 * dt * v[2]), 0.0)
            if drag_scale is None:
                a2 = self._acceleration(v2, w, baseline + height2)
            else:
                a2 = self._acceleration_with_scale(v2, w, drag_scale, self._config.gravity)
            v3 = [v[i] + 0.5 * dt * a2[i] for i in range(3)]
            if drag_scale is None:
                a3 = self._acceleration(v3, w, baseline + height2)
            else:
                a3 = self._acceleration_with_scale(v3, w, drag_scale, self._config.gravity)
            v4 = [v[i] + dt * a3[i] for i in range(3)]
            height4 = max(ground_z - (p[2] + dt * v[2]), 0.0)
            if drag_scale is None:
                a4 = self._acceleration(v4, w, baseline + height4)
            else:
                a4 = self._acceleration_with_scale(v4, w, drag_scale, self._config.gravity)

            p_next = [
                p[i] + dt / 6.0 * (v[i] + 2.0 * v2[i] + 2.0 * v3[i] + v4[i]) for i in range(3)
            ]
            v = [v[i] + dt / 6.0 * (a1[i] + 2.0 * a2[i] + 2.0 * a3[i] + a4[i]) for i in range(3)]

            if p_next[2] >= ground_z:
                # 穿越点线性插值：用这一步的竖直位移比例
                span = p_next[2] - p[2]
                ratio = 1.0 if span <= 0 else (ground_z - p[2]) / span
                ratio = min(max(ratio, 0.0), 1.0)
                impact = [p[i] + ratio * (p_next[i] - p[i]) for i in range(3)]
                impact[2] = ground_z
                speed = math.sqrt(sum(value**2 for value in v))
                return Impact(
                    True,
                    (impact[0], impact[1], impact[2]),
                    elapsed + ratio * dt,
                    speed,
                )
            p = p_next
            elapsed += dt

        LOGGER.warning("弹道积分超过 %.0fs 仍未落地，判为失败", MAX_FLIGHT_TIME_S)
        return Impact(False, None, elapsed, 0.0, reason="timeout")
