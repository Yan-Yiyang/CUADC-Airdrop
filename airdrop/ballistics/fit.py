"""弹道参数反演：用投放试验反推质量/阻力/释放延迟（P12）。

正演与反演
----------
正演（forward）就是 :class:`~airdrop.ballistics.model.BallisticsModel`：给出投放瞬间的
位置/速度/风 → RK4 积分 → 落点。**反演（inverse）**是拿它的输出与实测落点的差做
最小二乘，把参数往回拧::

    参数 θ = (Cd, A, δt, 风比例)——**质量不在其中**（它是称出来的输入，见下）
    r_i(θ) = [预测落点(θ) − 实测落点]_x,y      # 每次投放贡献 2 个残差（北、东）
    θ* = argmin Σ |r_i(θ)|²

竖向残差**不参与**：实测落点按定义落在地面上（``ground_z``），拿它去比没有信息量。

三个必须先想清楚的问题
----------------------
1. **质量是称出来的，不反演**。台秤/天平的相对精度轻松到 1e-3，而弹道反演对质量的
   分辨力远不如它——更关键的是：弹道方程里 ``Cd``、``A``、``m`` **只以 ``κ = Cd·A/m``
   的形式出现**（重力与质量无关），单靠落点数据根本分不开这三个量。所以
   :data:`FIT_PARAMETERS` 里**没有** ``mass_kg``：它和（若固定的）迎风面积一起作为
   **已知输入**从 :attr:`FitConfig.base` 进来。
   **数据真正识别到的是 κ = Cd·A/m**（:attr:`FitResult.drag_k_per_m`），一个数；
   给定**实测**质量后才谈得上 ``Cd·A = κ·m``（:attr:`FitResult.drag_area_m2`）。
   ⚠ **质量填错多少，拟合出的 Cd 与 Cd·A 就跟着错多少**（``Cd ∝ m``，实测 1e-6 级线性），
   **只有 κ 不受影响**——这正是"质量必须实测"的数值依据，也是换配重时的换算法：
   形状/气动不变 ⇒ ``Cd·A`` 不变 ⇒ ``κ' = Cd·A/m'``。回归用例
   ``test_identified_quantity_is_kappa_given_the_measured_mass`` 钉住了这三条。
2. **``Cd`` 与 ``A`` 只能固定其一**。既然只有乘积可辨识，"同时拟合 Cd 和迎风面积"
   必然有无穷多组解、残差可以一样小。默认固定面积（几何量，卡尺能量）、只拟合 Cd；
   代码会在两者同时被选中时**直接拒绝**，而不是给出一组看着合理、实则任意的数。
3. **释放延迟与阻力沿航迹方向高度相关**。延迟把整个落点顺航迹前移 ``v·δt``，阻力
   误差把落点顺航迹后移；只有当各次投放的**速度/高度差别够大**（`v` 与飞行时间不是
   同一个线性函数）时两者才分得开。所以 ``fit_release_delay_s`` 默认关，
   开了以后必须看 :attr:`FitResult.correlation` 与 :attr:`FitResult.condition_number`。

还有一条不属于"参数"但同样会毁掉结论的：**常数偏差吸收不掉**。如果实测落点整体偏了
某个固定矢量（测量点坐标错、目标点坐标错、坐标解算偏），任何弹道参数都解释不了它——
这种时候残差不会变小多少。:attr:`FitResult.bias_north_m` / ``bias_east_m`` 就是这个
"吸收不掉的量"，先修它再谈调参。

判据落在哪儿
------------
* :attr:`FitResult.ok`——这次反演数学上是否成立（样本够、不退化、不发散）；
* :attr:`FitResult.reliable`——**结果能不能信**（自由度 ≥ 1、条件数不超限、没参数顶到
  边界）。两者都为真才该把参数写回配置。

没有不确定度的拟合不是拟合
--------------------------
:attr:`FitResult.sigma`（1σ）与 :attr:`FitResult.correlation` 由解处的数值雅可比
（中心差分，**自己算**，不依赖 scipy 内部返回的是"缩放前还是缩放后"的雅可比）给出。
样本极少时 σ 会大得吓人——那就该多投几次，而不是把 σ 当噪声忽略掉。
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from typing import Any

import numpy as np
from scipy.optimize import least_squares

from ..config import BallisticsConfig
from .drops import DropSample, predict_record_impact
from .model import BallisticsModel

LOGGER = logging.getLogger(__name__)

__all__ = [
    "FIT_PARAMETERS",
    "DropResidual",
    "FitConfig",
    "FitResult",
    "fit_ballistics",
]

#: 可拟合的参数名（顺序即内部参数向量顺序）。
#: ⚠ **质量不在这里**：它是**称出来的**（见模块 docstring 的第 1 条）。
FIT_PARAMETERS: tuple[str, ...] = (
    "drag_coefficient",
    "cross_area_m2",
    "release_delay_s",
    "wind_scale",
)

#: 只以 ``Cd·A`` 乘积出现的两个参数——同时拟合必然退化（见模块 docstring）
_DEGENERATE_PAIR = ("drag_coefficient", "cross_area_m2")

#: 每个参数的"特征量级"（数值雅可比的差分步长与 scipy 的 ``x_scale`` 都用它）。
#: 延迟的初值可能是 0，所以这里必须给绝对下限，否则相对步长会退化成 0。
_PARAMETER_SCALES: Mapping[str, float] = {
    "drag_coefficient": 0.1,
    "cross_area_m2": 1e-3,
    "release_delay_s": 0.05,
    "wind_scale": 0.5,
}

#: 预测失败时给最小二乘的惩罚残差（米）——把参数推回可预测区域，别让它停在那儿
_PENALTY_M = 1e4


@dataclass(frozen=True, slots=True)
class FitConfig:
    """反演设置：拟合哪些参数、边界、以及"什么算可信"。

    ``base`` 是**没有被拟合的参数**取值来源，也是拟合的初值来源。它必须带上
    **实测**的质量与（若固定面积）几何面积——见模块 docstring 第 1 条：
    **质量是称出来的输入量，不是待估参数**。``base.air_density_isa`` 决定反演
    正演用的密度模型；密度基准（地面海拔）逐条取自投放记录的原点
    （:attr:`DropRecord.ground_altitude_m`），记录缺原点时退回 0 并在报告里告警。
    """

    base: BallisticsConfig = field(default_factory=BallisticsConfig)

    fit_drag_coefficient: bool = True
    fit_cross_area_m2: bool = False
    fit_release_delay_s: bool = False
    fit_wind_scale: bool = False

    drag_coefficient_bounds: tuple[float, float] = (0.05, 3.0)
    cross_area_bounds: tuple[float, float] = (1e-4, 0.5)
    #: 释放延迟（秒）——与 :class:`~airdrop.config.DropConfig` 的 ``delay_s`` **同口径**，
    #: 拟合出来可以直接回填到那儿
    release_delay_bounds: tuple[float, float] = (-1.0, 2.0)
    wind_scale_bounds: tuple[float, float] = (0.0, 3.0)

    #: 弹体挂点在机体系下的偏移（前-右-下，米）；零 = 按质心离机。见 drops.release_conditions
    release_offset_body_m: tuple[float, float, float] = (0.0, 0.0, 0.0)

    #: 缩放后雅可比的条件数上限：超过就判"参数分不开"，除非 allow_ill_conditioned
    max_condition_number: float = 1e4
    allow_underdetermined: bool = False
    allow_ill_conditioned: bool = False
    #: 留一交叉验证（样本 ≥ 3 才有意义）：反演最容易自欺的地方就是"拟合残差很小"
    leave_one_out: bool = True
    #: scipy 的损失函数；默认线性（不对任何一条测量降权）。只有在确认有粗差、
    #: 且样本足够多时才考虑 ``"soft_l1"``——它会**静默**降低离群点的影响
    loss: str = "linear"
    f_scale: float = 1.0
    max_nfev: int = 200
    diff_step: float = 1e-4

    # ------------------------------------------------------------------
    @property
    def fitted_names(self) -> tuple[str, ...]:
        """本次要拟合的参数名（顺序同 :data:`FIT_PARAMETERS`）。"""
        return tuple(name for name in FIT_PARAMETERS if getattr(self, f"fit_{name}"))

    def bounds_for(self, name: str) -> tuple[float, float]:
        table = {
            "drag_coefficient": self.drag_coefficient_bounds,
            "cross_area_m2": self.cross_area_bounds,
            "release_delay_s": self.release_delay_bounds,
            "wind_scale": self.wind_scale_bounds,
        }
        return table[name]

    def initial_for(self, name: str, samples: Sequence[DropSample]) -> float:
        """初值：弹道量取 ``base``，释放延迟取各条记录里的均值（判据当时用的值）。"""
        if name == "release_delay_s":
            if not samples:
                return 0.0
            return sum(sample.record.delay_s for sample in samples) / len(samples)
        if name == "wind_scale":
            return 1.0
        return float(getattr(self.base, name))

    def with_parameters(self, values: Mapping[str, float]) -> BallisticsConfig:
        """把拟合值写回 ``base``（只认 :data:`FIT_PARAMETERS` 里的名）。

        ``release_delay_s`` 与 ``wind_scale`` 不属于 ``BallisticsConfig``——它们是
        逐次投放的判据参数，由调用方单独传给落点预测函数，所以这里要滤掉。
        """
        updates = {
            name: float(value)
            for name, value in values.items()
            if name in FIT_PARAMETERS and name not in {"release_delay_s", "wind_scale"}
        }
        return replace(self.base, **updates) if updates else self.base


@dataclass(frozen=True, slots=True)
class DropResidual:
    """一次投放的残差（预测 − 实测），按航迹方向分解。

    ``along_track_m > 0`` 表示预测落点比实测**更远**（顺航迹方向多飞了）；
    ``cross_track_m > 0`` 表示偏航迹**右侧**（顺航迹看）。
    这两个分量是投放试验里最有用的读数：沿航迹的系统偏差指向阻力/延迟/速度，
    横向偏差指向风。
    """

    label: str
    index: int
    predicted_ned: tuple[float, float, float] | None
    measured_ned: tuple[float, float, float]
    error_north_m: float
    error_east_m: float
    error_m: float
    along_track_m: float
    cross_track_m: float
    flight_time_s: float | None = None
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "label": self.label,
            "index": self.index,
            "measured_ned": [round(v, 4) for v in self.measured_ned],
            "error_north_m": round(self.error_north_m, 4),
            "error_east_m": round(self.error_east_m, 4),
            "error_m": round(self.error_m, 4),
            "along_track_m": round(self.along_track_m, 4),
            "cross_track_m": round(self.cross_track_m, 4),
        }
        if self.predicted_ned is not None:
            data["predicted_ned"] = [round(v, 4) for v in self.predicted_ned]
        if self.flight_time_s is not None:
            data["flight_time_s"] = round(self.flight_time_s, 4)
        if self.reason:
            data["reason"] = self.reason
        return data


@dataclass(frozen=True, slots=True)
class FitResult:
    """反演结果（含诊断量）。``ok`` 与 ``reliable`` 的含义见模块 docstring。

    **数据识别到的只有一个数**：``drag_k_per_m`` = κ = ``Cd·A/m``（弹道方程里那个系数）。
    给定**实测**质量 ``m`` 之后才有 ``drag_area_m2`` = ``Cd·A`` = κ·m——它是**可迁移**的
    形状/气动量（换配重时 ``κ' = Cd·A/m'``）。质量填错时 Cd 与 Cd·A 同比例错，κ 不变。
    """

    ok: bool
    reliable: bool
    reason: str = ""
    fitted: tuple[str, ...] = ()
    parameters: Mapping[str, float] = field(default_factory=dict)
    ballistics: BallisticsConfig = field(default_factory=BallisticsConfig)
    release_delay_s: float = 0.0
    wind_scale: float = 1.0
    #: **识别量**：κ = Cd·A/m（1/m）——就是弹道方程里的阻力系数，与质量假设无关
    drag_k_per_m: float | None = None
    #: 给定**实测**质量后的 Cd·A（m²）：换配重时按它重算 κ'（质量填错则此值同比例错）
    drag_area_m2: float | None = None
    #: ``Cd·A`` 的 1σ（m²）；没拟合 Cd/面积 时为 None
    drag_area_sigma_m2: float | None = None
    n_samples: int = 0
    n_equations: int = 0
    dof: int = 0
    rms_error_m: float | None = None
    max_error_m: float | None = None
    bias_north_m: float | None = None
    bias_east_m: float | None = None
    per_drop: tuple[DropResidual, ...] = ()
    sigma: Mapping[str, float] = field(default_factory=dict)
    correlation: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    condition_number: float | None = None
    at_bound: tuple[str, ...] = ()
    leave_one_out_rms_m: float | None = None
    leave_one_out_max_m: float | None = None
    leave_one_out_errors_m: tuple[float, ...] = ()
    warnings: tuple[str, ...] = ()
    #: 最小二乘的函数评估次数（不是迭代次数——TRF 内部一步可能评估多次）
    nfev: int = 0

    def as_dict(self) -> dict[str, Any]:
        """JSON 报告（与 :mod:`tools.fit_ballistics` 写出的文件一致）。"""
        return {
            "ok": self.ok,
            "reliable": self.reliable,
            "reason": self.reason,
            "fitted": list(self.fitted),
            "parameters": {k: round(v, 6) for k, v in self.parameters.items()},
            "sigma": {k: round(v, 6) for k, v in self.sigma.items()},
            "correlation": {
                k: {k2: round(v2, 4) for k2, v2 in row.items()}
                for k, row in self.correlation.items()
            },
            "condition_number": (
                None if self.condition_number is None else round(self.condition_number, 3)
            ),
            "at_bound": list(self.at_bound),
            "release_delay_s": round(self.release_delay_s, 6),
            "wind_scale": round(self.wind_scale, 6),
            "drag_area_m2": (None if self.drag_area_m2 is None else round(self.drag_area_m2, 8)),
            "drag_area_sigma_m2": (
                None if self.drag_area_sigma_m2 is None else round(self.drag_area_sigma_m2, 8)
            ),
            "drag_k_per_m": (None if self.drag_k_per_m is None else round(self.drag_k_per_m, 8)),
            "ballistics": asdict(self.ballistics),
            "n_samples": self.n_samples,
            "n_equations": self.n_equations,
            "dof": self.dof,
            "rms_error_m": None if self.rms_error_m is None else round(self.rms_error_m, 4),
            "max_error_m": None if self.max_error_m is None else round(self.max_error_m, 4),
            "bias_north_m": (None if self.bias_north_m is None else round(self.bias_north_m, 4)),
            "bias_east_m": (None if self.bias_east_m is None else round(self.bias_east_m, 4)),
            "per_drop": [item.as_dict() for item in self.per_drop],
            "leave_one_out_rms_m": (
                None if self.leave_one_out_rms_m is None else round(self.leave_one_out_rms_m, 4)
            ),
            "leave_one_out_max_m": (
                None if self.leave_one_out_max_m is None else round(self.leave_one_out_max_m, 4)
            ),
            "leave_one_out_errors_m": [round(value, 4) for value in self.leave_one_out_errors_m],
            "nfev": self.nfev,
            "warnings": list(self.warnings),
        }


# ----------------------------------------------------------------------
# 正演包装
# ----------------------------------------------------------------------
def _predict(
    sample: DropSample,
    config: FitConfig,
    values: Mapping[str, float],
):
    """按参数取值正演一条样本的落点（返回 :class:`~airdrop.ballistics.model.Impact`）。"""
    model = BallisticsModel(config.with_parameters(values))
    return predict_record_impact(
        sample.record,
        model,
        delay_s=float(values.get("release_delay_s", sample.record.delay_s)),
        offset_body_m=config.release_offset_body_m,
        wind_scale=float(values.get("wind_scale", 1.0)),
    )


def _residual_vector(
    samples: Sequence[DropSample],
    config: FitConfig,
    values: Mapping[str, float],
) -> np.ndarray:
    """残差向量 ``[Δnorth, Δeast, ...]``（预测 − 实测），顺序与 ``samples`` 一致。"""
    residuals = np.empty(2 * len(samples), dtype=np.float64)
    for position, sample in enumerate(samples):
        impact = _predict(sample, config, values)
        if not impact.ok or impact.ned is None:
            # 参数跑到"预测不出来"的区域（起点已在地下、积分超时）：给一个大惩罚，
            # 让它退回去而不是停在那儿。真正的失败判据是解处仍然不可预测。
            residuals[2 * position] = _PENALTY_M
            residuals[2 * position + 1] = _PENALTY_M
            continue
        residuals[2 * position] = impact.ned[0] - sample.impact_ned[0]
        residuals[2 * position + 1] = impact.ned[1] - sample.impact_ned[1]
    return residuals


def _step_for(name: str, value: float, bounds: tuple[float, float], ratio: float) -> float:
    """数值雅可比的差分步长（相对 + 绝对下限，绝不越界）。"""
    scale = max(abs(float(value)), _PARAMETER_SCALES[name])
    step = scale * float(ratio)
    span = abs(bounds[1] - bounds[0])
    return max(min(step, 0.25 * span), 1e-9)


def _numeric_jacobian(
    samples: Sequence[DropSample],
    config: FitConfig,
    names: Sequence[str],
    values: list[float],
) -> np.ndarray:
    """中心差分雅可比 ``∂r/∂θ``（m×n）。贴边时自动退成单侧差分。"""
    point = dict(zip(names, values, strict=True))
    n_equations = 2 * len(samples)
    jacobian = np.zeros((n_equations, len(names)), dtype=np.float64)
    for column, name in enumerate(names):
        bounds = config.bounds_for(name)
        step = _step_for(name, values[column], bounds, config.diff_step)
        low = values[column] - step
        high = values[column] + step
        if low < bounds[0]:
            low = float(bounds[0])
            high = min(values[column] + 2.0 * step, float(bounds[1]))
        elif high > bounds[1]:
            high = float(bounds[1])
            low = max(values[column] - 2.0 * step, float(bounds[0]))
        if high - low <= 0.0:
            continue
        forward = dict(point, **{name: high})
        backward = dict(point, **{name: low})
        jacobian[:, column] = (
            _residual_vector(samples, config, forward) - _residual_vector(samples, config, backward)
        ) / (high - low)
    return jacobian


def _along_cross(
    sample: DropSample,
    error_north: float,
    error_east: float,
) -> tuple[float, float]:
    """把水平残差分解成"沿航迹 / 垂直航迹（右为正）"。

    航迹方向取**投放瞬间记录到的水平速度**方向（不是目标连线方向）：投放试验里
    关心的正是"顺着飞机飞的方向多飞了还是少飞了"。
    """
    north, east = sample.record.velocity_ned[0], sample.record.velocity_ned[1]
    norm = math.hypot(north, east)
    if norm < 1e-9:
        return (math.hypot(error_north, error_east), 0.0)
    unit_north, unit_east = north / norm, east / norm
    along = error_north * unit_north + error_east * unit_east
    cross = -error_east * unit_north + error_north * unit_east
    return (along, cross)


def _per_drop(
    samples: Sequence[DropSample],
    config: FitConfig,
    values: Mapping[str, float],
) -> tuple[DropResidual, ...]:
    items = []
    for sample in samples:
        impact = _predict(sample, config, values)
        measured = sample.impact_ned
        if impact.ok and impact.ned is not None:
            error_north = impact.ned[0] - measured[0]
            error_east = impact.ned[1] - measured[1]
            predicted: tuple[float, float, float] | None = impact.ned
            flight_time: float | None = impact.flight_time_s
            reason = ""
        else:
            error_north = _PENALTY_M
            error_east = _PENALTY_M
            predicted = None
            flight_time = None
            reason = impact.reason or "no_prediction"
        along, cross = _along_cross(sample, error_north, error_east)
        items.append(
            DropResidual(
                label=sample.name,
                index=sample.index,
                predicted_ned=predicted,
                measured_ned=measured,
                error_north_m=error_north,
                error_east_m=error_east,
                error_m=math.hypot(error_north, error_east),
                along_track_m=along,
                cross_track_m=cross,
                flight_time_s=flight_time,
                reason=reason,
            )
        )
    return tuple(items)


def _rms(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return math.sqrt(sum(float(v) ** 2 for v in values) / len(values))


def _identified(
    ballistics: BallisticsConfig,
    parameters: Mapping[str, float],
    sigma: Mapping[str, float],
) -> tuple[float, float | None, float]:
    """数据真正识别到的量：``Cd·A``（含 1σ 传播）与 ``κ = Cd·A/m``。

    ``Cd`` 与 ``A`` 只有一个可能是自由参数（同时拟合会被退化检查拒掉），所以一阶
    传播就是"自由那个的 σ × 被固定的那个"。σ 未知（自由度不足）时给 None。
    """
    drag_area = float(ballistics.drag_coefficient) * float(ballistics.cross_area_m2)
    if "drag_coefficient" in parameters and "drag_coefficient" in sigma:
        sigma_area: float | None = float(sigma["drag_coefficient"]) * float(
            ballistics.cross_area_m2
        )
    elif "cross_area_m2" in parameters and "cross_area_m2" in sigma:
        sigma_area = float(sigma["cross_area_m2"]) * float(ballistics.drag_coefficient)
    else:
        sigma_area = None
    mass = float(ballistics.mass_kg)
    k = drag_area / mass if mass > 0 else float("nan")
    return drag_area, sigma_area, k


# ----------------------------------------------------------------------
# 主入口
# ----------------------------------------------------------------------
def fit_ballistics(  # noqa: PLR0911, PLR0912, PLR0915 - 一次反演要把"取数→拟合→诊断→结论"走完
    samples: Sequence[DropSample],
    config: FitConfig | None = None,
) -> FitResult:
    """用"投放状态 + 实测落点"反演弹道参数（最小二乘，TRF + 边界）。

    返回的 :class:`FitResult` 里同时给出拟合值、不确定度、参数相关性、留一交叉验证
    与"每个参数能不能信"的判断——**先看 ``ok``/``reliable``，再看数值**。
    """
    settings = config or FitConfig()
    data = tuple(samples)
    base = settings.base

    if not data:
        return FitResult(
            ok=False,
            reliable=False,
            reason="没有样本：需要至少一条「投放记录 + 实测落点」",
            ballistics=base,
        )

    names = settings.fitted_names
    if not names:
        return FitResult(
            ok=False,
            reliable=False,
            reason="没有选中任何要拟合的参数（FitConfig.fit_* 全为 False）",
            ballistics=base,
            n_samples=len(data),
        )

    # 1) 退化检查：Cd 与迎风面积只以 Cd·A 的乘积出现，同时拟合必然解不唯一
    #    （质量根本不在可拟合集合里——它是称出来的，见模块 docstring 第 1 条）
    both = [name for name in names if name in _DEGENERATE_PAIR]
    if len(both) > 1:
        return FitResult(
            ok=False,
            reliable=False,
            reason=(
                f"参数退化：{both} 在弹道方程里只以 Cd·A 的乘积出现，"
                "同时拟合它们有无穷多组解——固定其中一个（面积是几何量，卡尺能量），"
                "只拟合另一个；数据真正识别到的量是 Cd·A，结果里会直接给出"
            ),
            ballistics=base,
            n_samples=len(data),
        )

    # 2) 风比例需要每条记录都有风估计（缺风的那条按零风算，比例因子对它没有意义）
    if "wind_scale" in names:
        missing = [s.name for s in data if s.record.wind_ned is None]
        if missing:
            return FitResult(
                ok=False,
                reliable=False,
                reason=(
                    f"拟合风比例需要每条记录都有风估计，{missing} 没有"
                    "（缺风的记录按零风正演，比例因子对它无效）"
                ),
                ballistics=base,
                n_samples=len(data),
            )

    # 3) 自由度：每次投放贡献 2 个残差
    n_equations = 2 * len(data)
    if n_equations < len(names) and not settings.allow_underdetermined:
        return FitResult(
            ok=False,
            reliable=False,
            reason=(
                f"欠定：{len(data)} 次投放只有 {n_equations} 个残差，"
                f"要拟合 {len(names)} 个参数——多投几次，或显式 allow_underdetermined"
            ),
            ballistics=base,
            n_samples=len(data),
            n_equations=n_equations,
        )

    # 4) 初值必须在边界内、且每条样本都能预测出来
    initial = {name: settings.initial_for(name, data) for name in names}
    for name in names:
        low, high = settings.bounds_for(name)
        if not low < initial[name] < high:
            return FitResult(
                ok=False,
                reliable=False,
                reason=(
                    f"初值越界：{name}={initial[name]} 不在 ({low}, {high}) 内，"
                    "改 FitConfig 的边界或 base"
                ),
                ballistics=base,
                n_samples=len(data),
                n_equations=n_equations,
            )
    start = np.array([initial[name] for name in names], dtype=np.float64)
    if not np.all(np.isfinite(_residual_vector(data, settings, initial))):
        return FitResult(
            ok=False,
            reliable=False,
            reason="初值下就有样本预测不出来（落点积分失败），先看记录里的 ground_z 与高度",
            ballistics=base,
            n_samples=len(data),
            n_equations=n_equations,
        )

    lower = np.array([settings.bounds_for(name)[0] for name in names])
    upper = np.array([settings.bounds_for(name)[1] for name in names])
    scales = np.array([max(abs(initial[name]), _PARAMETER_SCALES[name]) for name in names])

    def residuals(point: np.ndarray) -> np.ndarray:
        values = dict(zip(names, (float(v) for v in point), strict=True))
        return _residual_vector(data, settings, values)

    solution = least_squares(
        residuals,
        start,
        bounds=(lower, upper),
        x_scale=scales,
        loss=settings.loss,
        f_scale=settings.f_scale,
        max_nfev=settings.max_nfev,
        method="trf",
    )
    values = {name: float(solution.x[i]) for i, name in enumerate(names)}
    ballistics = settings.with_parameters(values)
    delay = float(values.get("release_delay_s", settings.initial_for("release_delay_s", data)))
    wind_scale = float(values.get("wind_scale", 1.0))

    per_drop = _per_drop(data, settings, values)
    if any(item.reason for item in per_drop):
        failed = [item.label for item in per_drop if item.reason]
        return FitResult(
            ok=False,
            reliable=False,
            reason=f"解处仍有样本预测不出来：{failed}（弹道积分失败，不是参数问题）",
            fitted=names,
            parameters=values,
            ballistics=ballistics,
            release_delay_s=delay,
            wind_scale=wind_scale,
            n_samples=len(data),
            n_equations=n_equations,
            per_drop=per_drop,
            # 字段名是 nfev（= 函数评估次数，不是迭代次数）——见 FitResult.nfev
            nfev=int(solution.nfev),
        )

    errors = [item.error_m for item in per_drop]
    rms = _rms(errors)
    bias_north = sum(item.error_north_m for item in per_drop) / len(per_drop)
    bias_east = sum(item.error_east_m for item in per_drop) / len(per_drop)

    # ---- 诊断：雅可比 → 条件数 / σ / 相关性 ----
    jacobian = _numeric_jacobian(data, settings, names, list(solution.x))
    scaled = jacobian * scales.reshape(1, -1)
    condition = float(np.linalg.cond(scaled)) if len(names) > 1 else 1.0
    dof = n_equations - len(names)
    sigma: dict[str, float] = {}
    correlation: dict[str, dict[str, float]] = {}
    warnings: list[str] = []
    missing_origin = [sample.name for sample in data if sample.record.origin is None]
    if missing_origin:
        warnings.append(
            f"这些投放记录没有 NED 原点：{missing_origin}——密度基准退回海平面，"
            "高原站点会高估密度；补上记录里的 origin 再重跑"
        )
    if dof >= 1:
        residual_sum = float(np.sum(np.square(residuals(solution.x))))
        variance = residual_sum / dof
        # 走 SVD 而不是 inv(JᵗJ)：参数几乎完全相关时 JᵗJ 近奇异，numpy 的 inv 会
        # **不报错**地返回一堆垃圾；SVD 能顺手把"哪个方向不可辨识"暴露出来
        # （奇异值小于 s_max×1e-10 的方向按截断处理，σ 因此只是量级参考）。
        _, singular, right = np.linalg.svd(scaled, full_matrices=False)
        threshold = 1e-10 * float(singular[0]) if singular.size else 0.0
        keep = singular > threshold
        if not np.all(keep):
            warnings.append(
                "参数空间里存在完全不可辨识的方向（奇异值差 >1e10 倍）："
                "σ 按截断伪逆给出，只能当量级参考"
            )
        weights = np.zeros_like(singular)
        weights[keep] = 1.0 / np.square(singular[keep])
        covariance = (right.T * weights) @ right * variance
        errors_scaled = np.sqrt(np.abs(np.diag(covariance)))
        sigma = {name: float(errors_scaled[i] * scales[i]) for i, name in enumerate(names)}
        std = errors_scaled
        with np.errstate(divide="ignore", invalid="ignore"):
            corr = covariance / np.outer(std, std)
        correlation = {
            name: {other: float(corr[i, j]) for j, other in enumerate(names)}
            for i, name in enumerate(names)
        }
    else:
        warnings.append(
            f"自由度 {dof} ≤ 0：把参数调到残差为零总能做到，σ 无从谈起——至少要多一次投放的测量"
        )

    at_bound: list[str] = []
    for i, name in enumerate(names):
        low, high = settings.bounds_for(name)
        span = abs(high - low)
        margin = 1e-3 * span
        if solution.x[i] <= low + margin or solution.x[i] >= high - margin:
            at_bound.append(name)
    if at_bound:
        warnings.append(
            f"参数顶到了边界：{at_bound}——多半是物理量本身超出设定范围，"
            "先核对单位与记录，再考虑放宽边界"
        )

    if condition > settings.max_condition_number:
        warnings.append(
            f"缩放雅可比条件数 {condition:.3g} > {settings.max_condition_number:.3g}："
            f"参数分不开（看 correlation），别照抄这组数"
        )

    reliable = (
        dof >= 1
        and condition <= settings.max_condition_number
        and not at_bound
        and solution.success
    )
    if not solution.success:
        warnings.append(f"最小二乘未收敛：{solution.message}")
    if any(item.error_m >= _PENALTY_M / 2 for item in per_drop):
        reliable = False

    # ---- 留一交叉验证 ----
    loo_errors: list[float] = []
    loo_rms: float | None = None
    loo_max: float | None = None
    if settings.leave_one_out:
        if len(data) >= 3:
            inner = replace(settings, leave_one_out=False, allow_ill_conditioned=True)
            inner_ok = True
            for index in range(len(data)):
                rest = [s for i, s in enumerate(data) if i != index]
                if 2 * len(rest) < len(names):
                    inner_ok = False
                    break
                result = fit_ballistics(rest, inner)
                if not result.ok:
                    inner_ok = False
                    break
                held_out = data[index]
                predicted = _predict(held_out, inner, result.parameters)
                if not predicted.ok or predicted.ned is None:
                    inner_ok = False
                    break
                loo_errors.append(
                    math.hypot(
                        predicted.ned[0] - held_out.impact_ned[0],
                        predicted.ned[1] - held_out.impact_ned[1],
                    )
                )
            if inner_ok and loo_errors:
                loo_rms = _rms(loo_errors)
                loo_max = max(loo_errors)
            else:
                warnings.append("留一交叉验证没做成（样本或自由度不够）")
        else:
            warnings.append("样本少于 3 条，跳过留一交叉验证")

    reason = "" if reliable else "收敛了，但按诊断不可信（见 warnings）"
    drag_area, drag_area_sigma, drag_k = _identified(ballistics, values, sigma)
    if condition > settings.max_condition_number and not settings.allow_ill_conditioned:
        return FitResult(
            ok=False,
            reliable=False,
            reason=(
                f"参数不可辨识（条件数 {condition:.3g}）：先只拟合一个参数，"
                "或显式 allow_ill_conditioned=True 看诊断"
            ),
            fitted=names,
            parameters=values,
            ballistics=ballistics,
            release_delay_s=delay,
            wind_scale=wind_scale,
            drag_area_m2=drag_area,
            drag_area_sigma_m2=drag_area_sigma,
            drag_k_per_m=drag_k,
            n_samples=len(data),
            n_equations=n_equations,
            dof=dof,
            rms_error_m=rms,
            max_error_m=max(errors),
            bias_north_m=bias_north,
            bias_east_m=bias_east,
            per_drop=per_drop,
            sigma=sigma,
            correlation=correlation,
            condition_number=condition,
            at_bound=tuple(at_bound),
            warnings=tuple(warnings),
            nfev=int(solution.nfev),
        )

    return FitResult(
        ok=True,
        reliable=bool(reliable),
        reason=reason,
        fitted=names,
        parameters=values,
        ballistics=ballistics,
        release_delay_s=delay,
        wind_scale=wind_scale,
        drag_area_m2=drag_area,
        drag_area_sigma_m2=drag_area_sigma,
        drag_k_per_m=drag_k,
        n_samples=len(data),
        n_equations=n_equations,
        dof=dof,
        rms_error_m=rms,
        max_error_m=max(errors),
        bias_north_m=bias_north,
        bias_east_m=bias_east,
        per_drop=per_drop,
        sigma=sigma,
        correlation=correlation,
        condition_number=condition,
        at_bound=tuple(at_bound),
        leave_one_out_rms_m=loo_rms,
        leave_one_out_max_m=loo_max,
        leave_one_out_errors_m=tuple(loo_errors),
        warnings=tuple(warnings),
        nfev=int(solution.nfev),
    )
