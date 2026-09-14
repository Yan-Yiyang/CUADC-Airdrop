"""二次阻力弹道：RK4 定步长积分 → 落点与飞行时间（计划 4.6）。

方程（NED，z **向下为正**）::

    m·dv/dt = m·g − ½·ρ(z)·Cd·A·|v − w|·(v − w)

* 重力只有 ``+z`` 分量（往下掉）；
* 阻力相对**空气**算（``v − w``，``w`` 是风的速度矢量）——这是风会平移落点的唯一原因；
* ``ρ`` 默认按 ISA 随高度变化。

两处刻意的取舍（都写在代码里免得被当 bug）
------------------------------------------
1. **ISA 用"离地高度"当海拔**：站点海拔没接入本模块，而 20m 量级下密度只差 ~0.2%，
   对落点的影响可以忽略。要精确就得把地面点海拔传进来（``ground_z`` 已经是相对量）。
2. **落点用线性插值定在穿越瞬间**：定步长 5ms 直接取"第一个 z ≥ ground_z 的步"会带来
   最多一步的水平误差（10 m/s 平飞时 5cm）；穿越点插值后与解析解对得上。
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Sequence

from ..config import BallisticsConfig
from ..telemetry.models import TelemetrySnapshot

LOGGER = logging.getLogger(__name__)

__all__ = [
    "Impact",
    "BallisticsModel",
    "ZERO_WIND",
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


def isa_air_density(height_m: float) -> float:
    """ISA 标准大气下 ``height_m`` 处的空气密度（kg/m³）。

    ``height_m`` 是**离地高度**（见模块 docstring 的取舍 1），负值按 0 处理。
    """
    height = max(float(height_m), 0.0)
    return SEA_LEVEL_DENSITY * (1.0 - _ISA_COEFF * height) ** _ISA_EXP


def wind_from_snapshot(
    snapshot: TelemetrySnapshot,
    config: BallisticsConfig,
) -> tuple[float, float, float] | None:
    """按 ``config.wind_source`` 取风矢量（NED，m/s）。

    * ``"zero"``：直接给 :data:`ZERO_WIND`；
    * ``"telemetry"``：读快照里的飞控风估计（``wind_north/east/down_m_s``）。
      **取不到就返回 None**——这里刻意不悄悄给 0，否则"没有风估计"与"风确实是 0"
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
    if any(value is None for value in values):
        return None
    return (float(values[0]), float(values[1]), float(values[2]))


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

    参数全部来自 :class:`~airdrop.config.BallisticsConfig`——**质量按实测称重填入**，
    ``drag_coefficient`` 靠投放试验反演（占位值 0.6 只用于跑通链路），迎风面积是几何量。
    注意方程里 ``Cd``、``A``、``m`` 只以 ``κ = Cd·A/m`` 的形式出现（``self._k``）。
    """

    def __init__(self, config: BallisticsConfig) -> None:
        self._config = config
        if config.mass_kg <= 0:
            raise ValueError(f"质量必须为正: {config.mass_kg}")
        if config.rk_dt <= 0:
            raise ValueError(f"积分步长必须为正: {config.rk_dt}")
        #: 阻力系数 k = ½·Cd·A/m（1/m），ρ 单独随高度算
        self._k = 0.5 * config.drag_coefficient * config.cross_area_m2 / config.mass_kg

    @property
    def config(self) -> BallisticsConfig:
        return self._config

    def drag_factor(self, velocity_rel: Sequence[float], height_m: float) -> float:
        """阻力加速度系数 ``½·ρ·Cd·A/m·|v_rel|``（1/s）；再乘 ``v_rel`` 就是加速度。"""
        density = (
            isa_air_density(height_m)
            if self._config.air_density_isa
            else SEA_LEVEL_DENSITY
        )
        speed = math.sqrt(sum(float(v) ** 2 for v in velocity_rel))
        return self._k * density * speed

    def terminal_velocity(self, height_m: float = 0.0) -> float:
        """终端速度 ``v_t = √(2mg / (ρ·Cd·A))``（m/s）。"""
        density = (
            isa_air_density(height_m)
            if self._config.air_density_isa
            else SEA_LEVEL_DENSITY
        )
        area_term = self._config.drag_coefficient * self._config.cross_area_m2
        if area_term <= 0:
            return float("inf")          # 无阻力：不存在终端速度
        return math.sqrt(2.0 * self._config.mass_kg * self._config.gravity
                         / (density * area_term))

    # ------------------------------------------------------------------
    # 积分
    # ------------------------------------------------------------------
    def _acceleration(
        self,
        velocity: Sequence[float],
        wind: Sequence[float],
        height_m: float,
    ) -> tuple[float, float, float]:
        """当前加速度：重力 + 阻力。"""
        rel = [float(velocity[i]) - float(wind[i]) for i in range(3)]
        factor = self.drag_factor(rel, height_m)
        gravity = self._config.gravity
        return (
            -factor * rel[0],
            -factor * rel[1],
            gravity - factor * rel[2],
        )

    def predict_impact(
        self,
        position_ned: Sequence[float],
        velocity_ned: Sequence[float],
        *,
        ground_z: float = 0.0,
        wind: Sequence[float] = (0.0, 0.0, 0.0),
    ) -> Impact:
        """从 ``position_ned``/``velocity_ned`` 起积分，返回落点与飞行时间。

        终止条件 ``z ≥ ground_z``；穿越点线性插值。落不到地（时长超过
        :data:`MAX_FLIGHT_TIME_S`）或起点已在地面以下都返回 ``ok=False`` 并说明原因
        ——**不返回一个编出来的落点**。
        """
        if len(position_ned) != 3 or len(velocity_ned) != 3:
            raise ValueError("position_ned / velocity_ned 必须是 3 维 NED 矢量")
        p = [float(v) for v in position_ned]
        v = [float(v) for v in velocity_ned]
        w = [float(v) for v in wind]
        if p[2] >= ground_z:
            return Impact(False, None, 0.0, 0.0, reason="below_ground")

        dt = self._config.rk_dt
        elapsed = 0.0
        while elapsed < MAX_FLIGHT_TIME_S:
            height = max(ground_z - p[2], 0.0)
            # RK4：k1..k4 分别是位置与速度的斜率
            a1 = self._acceleration(v, w, height)
            v2 = [v[i] + 0.5 * dt * a1[i] for i in range(3)]
            a2 = self._acceleration(v2, w, max(ground_z - (p[2] + 0.5 * dt * v[2]), 0.0))
            v3 = [v[i] + 0.5 * dt * a2[i] for i in range(3)]
            a3 = self._acceleration(v3, w, max(ground_z - (p[2] + 0.5 * dt * v[2]), 0.0))
            v4 = [v[i] + dt * a3[i] for i in range(3)]
            a4 = self._acceleration(v4, w, max(ground_z - (p[2] + dt * v[2]), 0.0))

            p_next = [
                p[i] + dt / 6.0 * (v[i] + 2.0 * v2[i] + 2.0 * v3[i] + v4[i])
                for i in range(3)
            ]
            v = [
                v[i] + dt / 6.0 * (a1[i] + 2.0 * a2[i] + 2.0 * a3[i] + a4[i])
                for i in range(3)
            ]

            if p_next[2] >= ground_z:
                # 穿越点线性插值：用这一步的竖直位移比例
                span = p_next[2] - p[2]
                ratio = 1.0 if span <= 0 else (ground_z - p[2]) / span
                ratio = min(max(ratio, 0.0), 1.0)
                impact = [
                    p[i] + ratio * (p_next[i] - p[i]) for i in range(3)
                ]
                impact[2] = ground_z
                speed = math.sqrt(sum(value ** 2 for value in v))
                return Impact(
                    True, (impact[0], impact[1], impact[2]),
                    elapsed + ratio * dt, speed,
                )
            p = p_next
            elapsed += dt

        LOGGER.warning("弹道积分超过 %.0fs 仍未落地，判为失败", MAX_FLIGHT_TIME_S)
        return Impact(False, None, elapsed, 0.0, reason="timeout")
