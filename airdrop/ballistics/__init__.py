"""弹道与投放：二次阻力弹道预测 + 实时投放判据 + 投放试验反演（计划 4.6 / P12）。

* :mod:`~airdrop.ballistics.model`：RK4 定步长积分 → 落点与飞行时间（含 ISA 空气密度、
  风平移）；
* :mod:`~airdrop.ballistics.release`：:class:`ReleaseJudge`——每拍预测落点，距离 ≤ 半径
  即投；越过目标后强制投放；一次投放即锁存。
* :mod:`~airdrop.ballistics.drops`：:class:`DropRecord`——投放瞬间的位置/速度/姿态/
  风/预测落点（事后无法重建，所以实飞必须写入磁盘），以及实测落点的读取与配对；
* :mod:`~airdrop.ballistics.fit`：:func:`fit_ballistics`——用投放试验的"投放状态 +
  实测落点"反演弹道参数，并给出可辨识性诊断（退化/相关/留一验证）。

⚠ :class:`~airdrop.config.BallisticsConfig` 里的质量是称出来的（台秤实测，不参与反演），
阻力系数与迎风面积只以 ``Cd·A`` 的乘积影响弹道（单靠落点数据分不开）——固定面积、
用投放试验反演 Cd，流程见 :mod:`airdrop.ballistics.fit` 与 ``tools/fit_ballistics.py``。

⚠ 惰性导出（见 :mod:`airdrop._lazy`）：import airdrop.ballistics 只执行本文件。
``drops`` 会间接拉 georef 的 cv2 依赖，所以排在后面——取个 ``BallisticsModel``
不该顺带加载 OpenCV。
"""

from .._lazy import lazy_dir, lazy_exports

__getattr__ = lazy_exports(__name__, ("model", "release", "drops", "fit"))
__dir__ = lazy_dir(__name__)

__all__ = [
    "DROPS_NAME",
    "FIT_PARAMETERS",
    "IMPACTS_CSV_NAME",
    "IMPACTS_JSONL_NAME",
    "MAX_FLIGHT_TIME_S",
    "SEA_LEVEL_DENSITY",
    "SUMMARY_HZ",
    "ZERO_WIND",
    "BallisticsModel",
    "DropRecord",
    "DropResidual",
    "DropSample",
    "FitConfig",
    "FitResult",
    "Impact",
    "ImpactMeasurement",
    "ReleaseDecision",
    "ReleaseJudge",
    "append_drop",
    "attitude_matrix",
    "fit_ballistics",
    "heading_unit_vector",
    "isa_air_density",
    "load_drops",
    "load_impacts",
    "match_impacts",
    "predict_record_impact",
    "release_conditions",
    "resolve_wind",
    "wind_from_snapshot",
    "write_impact_template",
]
