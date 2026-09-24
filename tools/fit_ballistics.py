"""投放试验反演：读投放记录 + 实测落点 → 优化弹道参数（P12）。

用法（命令行由集中入口解析；本文件顶部的常量就是默认值）::

    ./.venv/Scripts/python.exe -m airdrop.run fit-ballistics --help
    ./.venv/Scripts/python.exe -m airdrop.run fit-ballistics --impacts <落点文件> --mass-kg 0.365

流程
----
1. **称重**：弹体质量用台秤/天平实测，填到 :data:`MEASURED_MASS_KG`（迎风面积按几何量
   填 :data:`MEASURED_CROSS_AREA_M2`）——**它们不参与反演**：反演对质量的分辨力远不如秤，
   而且弹道方程里 Cd/m/A 只以 ``Cd·A/m`` 出现，单靠落点数据分不开；
2. 实飞打开记录（``RECORD=True``），每次投放都会写一条 ``drops.jsonl``：
   投放瞬间的位置/速度/姿态/风、目标点、判据的前推位置与预测落点；
3. 现场量出落点，填进**同一个飞行目录**里的 ``impacts.jsonl``（本工具在没有这个文件时
   会生成一份待填模板）——经纬度或 NED 都行，见
   :func:`~airdrop.ballistics.drops.load_impacts`；
4. 再跑本工具：打印逐次投放的预测/实测对照表，做最小二乘反演，把参数、**识别量 Cd·A**、
   不确定度、相关性、留一验证与"能不能信"的判断一起写进 ``ballistics_fit.json``。

⚠ 读数要先看结论那一行——**参数分不开的时候，最小二乘照样会给你一组漂亮数字**
（模块 :mod:`airdrop.ballistics.fit` 的 docstring 讲了三条典型陷阱）。
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

from airdrop.config import BallisticsConfig, Config, DropConfig

if TYPE_CHECKING:
    # 标注用的类型（`from __future__ import annotations` 下运行时不求值）。
    # 真正的运行期导入仍在函数体内——`airdrop.ballistics` 会间接拉 georef（cv2），
    # 见上面 _LAZY 那条约定。
    from airdrop.ballistics import DropSample, FitConfig, FitResult

LOGGER = logging.getLogger("fit_ballistics")

#: 惰性导入标记（见模块 docstring）：`airdrop.ballistics` 的 drops 会间接拉 georef（cv2），
#: 所以只在真正反演的函数体内导入——`python -m airdrop.run fit-ballistics --help` 不加载它。
_LAZY = "from airdrop.ballistics import ...  # 惰性：重依赖，见模块 docstring"


# ----------------------------------------------------------------------
# 配置（改这里）
# ----------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class FlightInput:
    """一个架次的输入：飞行目录（或直接给 ``drops.jsonl``）+ 可选的测量文件。"""

    path: Path
    #: 实测落点文件；None = 在飞行目录里找 ``impacts.jsonl`` / ``impacts.csv``
    impacts: Path | None = None


#: 空 = 用 ``flights/`` 下**最新**一个含投放记录的目录（会打印用的是哪个）
FLIGHTS: tuple[FlightInput, ...] = ()
#: 记录根目录（``Config().record.dir``）；FLIGHTS 为空时在里面找最新架次
FLIGHTS_ROOT = Path(Config().record.dir)
#: 只跑一个架次时的输入（命令行 ``--drops``）：飞行目录或 ``drops.jsonl``；None = 取最新架次
DROPS: Path | None = None
#: 实测落点文件（命令行 ``--impacts``）；None = 在飞行目录里自动找
IMPACTS: Path | None = None

#: 拟合哪些参数。⚠ 两条硬约束（都不是"随便调调"）：
#: 1. **质量不在其中**——用台秤/天平实测后填 :data:`MEASURED_MASS_KG`，不靠反演：
#:    反演对质量的分辨力远不如秤，而且 Cd/m/A 在方程里只以 Cd·A/m 出现；
#: 2. **Cd 与迎风面积只能固定一个**（只以乘积可辨识），两个都拟合会被直接拒绝。
FIT_DRAG_COEFFICIENT = True
FIT_CROSS_AREA_M2 = False
#: 释放延迟与阻力沿航迹方向高度相关：只有在各次投放的**速度/高度差别够大**时才能分开
FIT_RELEASE_DELAY_S = False
FIT_WIND_SCALE = False

#: 弹体质量（kg）——**实测值**，不参与反演。数据只识别 κ = Cd·A/m，所以质量填错多少，
#: 拟合出的 Cd 与 Cd·A 就跟着错多少（κ 不受影响）。
MEASURED_MASS_KG = 0.365
#: 迎风面积（m²）——几何量（卡尺/投影面积）；FIT_CROSS_AREA_M2=True 时忽略本值
MEASURED_CROSS_AREA_M2 = 0.004

#: 弹体挂点在机体系下的偏移（前-右-下，米）；零 = 按质心离机。给了偏移就必须有姿态记录
RELEASE_OFFSET_BODY_M = (0.0, 0.0, 0.0)

#: 缩放雅可比条件数上限；超过就判"参数分不开"并拒绝给结论
MAX_CONDITION_NUMBER = 1e4
#: 没有测量文件时是否顺手生成待填模板
WRITE_TEMPLATE = True

OUTPUT_PATH = Path("ballistics_fit.json")


# ----------------------------------------------------------------------
# 读数
# ----------------------------------------------------------------------
def _drops_path(path: Path) -> Path:
    """``FlightInput.path`` 既可以是飞行目录，也可以直接是 ``drops.jsonl``。"""
    if path.is_dir():
        return path / "drops.jsonl"
    return path


def _auto_impacts(drops_path: Path) -> Path | None:
    """在投放记录旁边找测量文件（jsonl 优先）。"""
    from airdrop.ballistics import IMPACTS_CSV_NAME, IMPACTS_JSONL_NAME  # 惰性：见模块 docstring

    for name in (IMPACTS_JSONL_NAME, IMPACTS_CSV_NAME):
        candidate = drops_path.parent / name
        if candidate.exists():
            return candidate
    return None


def _latest_flight_dir(root: Path) -> Path:
    """``flights/`` 下最新一个含投放记录的目录（目录名就是时间戳，按名排序即可）。"""
    if not root.is_dir():
        raise FileNotFoundError(f"记录根目录不存在：{root}（先飞一次，或用 FLIGHTS 指定）")
    candidates = sorted(
        (item for item in root.iterdir() if (item / "drops.jsonl").exists()),
        key=lambda item: item.name,
    )
    if not candidates:
        raise FileNotFoundError(f"{root} 下没有任何含 drops.jsonl 的架次")
    return candidates[-1]


def load_samples(
    flights: tuple[FlightInput, ...] | None = None,
) -> tuple[tuple[DropSample, ...], list[dict[str, object]], list[Path]]:
    """读入所有架次并配对实测落点。

    返回 ``(样本, 每架次的账目, 缺测量文件的架次路径)``；样本的 ``label`` 是
    ``"<架次目录名>#<序号>"``，跨架次也不会混。
    """
    from airdrop.ballistics import (  # 惰性：见模块 docstring
        load_drops,
        load_impacts,
        match_impacts,
    )

    chosen = tuple(flights if flights is not None else FLIGHTS)
    if not chosen:
        chosen = (FlightInput(path=_latest_flight_dir(FLIGHTS_ROOT)),)
        print(f"[目录] FLIGHTS 为空，用最新架次：{chosen[0].path}")

    samples: list[DropSample] = []
    ledger: list[dict[str, object]] = []
    missing_files: list[Path] = []
    for item in chosen:
        drops_path = _drops_path(Path(item.path))
        records = load_drops(drops_path)
        impacts_path = (None if item.impacts is None else Path(item.impacts)) or _auto_impacts(
            drops_path
        )
        measurements = () if impacts_path is None else load_impacts(impacts_path)
        if impacts_path is None:
            missing_files.append(drops_path)
        matched = match_impacts(records, measurements)
        label = drops_path.parent.name
        entry: dict[str, object] = {
            "flight": label,
            "drops": len(records),
            "measured": len(matched),
            "missing": sorted({r.index for r in records} - {s.index for s in matched}),
            "impacts": None if impacts_path is None else str(impacts_path),
        }
        ledger.append(entry)
        samples.extend(replace(sample, label=f"{label}#{sample.index}") for sample in matched)
    return tuple(samples), ledger, missing_files


def describe(
    result: FitResult,
    samples: tuple[DropSample, ...],
    *,
    max_condition_number: float = MAX_CONDITION_NUMBER,
) -> str:
    """逐次投放的对照表 + 拟合结论（人看的文本）。"""
    by_name = {sample.name: sample for sample in samples}
    lines: list[str] = []
    lines.append("")
    lines.append(
        "投放 / 序号            高度m   速度m/s  航向°  风(m/s)      判据误差m  反演后误差m  沿/横(m)"
    )
    lines.append("-" * 104)
    for item in result.per_drop:
        sample = by_name.get(item.label)
        record = None if sample is None else sample.record
        judged = None if record is None else record.predicted_error_m
        wind = (
            ""
            if record is None or record.wind_ned is None
            else (f"{record.wind_ned[0]:+.1f}/{record.wind_ned[1]:+.1f}")
        )
        lines.append(
            "{label:<22} {alt:6.1f} {speed:8.1f} {heading:6.1f}  {wind:<11} "
            "{judged:<10} {error:<12.3f} {along:+.3f}/{cross:+.3f}".format(
                label=item.label[:22],
                alt=0.0 if record is None else record.height_agl_m,
                speed=0.0 if record is None else record.horizontal_speed_m_s,
                heading=float("nan")
                if record is None or record.heading_deg is None
                else record.heading_deg,
                wind=wind,
                judged="—" if judged is None else f"{judged:.3f}",
                error=item.error_m,
                along=item.along_track_m,
                cross=item.cross_track_m,
            )
        )
    lines.append("-" * 104)
    lines.append(
        f"样本 {result.n_samples} 次投放 / {result.n_equations} 个残差 / 自由度 {result.dof}"
        f"；拟合参数 {list(result.fitted)}"
    )
    for name in result.fitted:
        value = result.parameters.get(name, float("nan"))
        sigma = result.sigma.get(name)
        lines.append(
            f"  {name:<20} = {value:.6g}" + ("" if sigma is None else f"  ± {sigma:.3g} (1σ)")
        )
    if result.drag_k_per_m is not None:
        # 数据识别到的**只有一个数**：κ = Cd·A/m。Cd·A 是"给定实测质量"后的换算量。
        area = result.ballistics.cross_area_m2
        mass = result.ballistics.mass_kg
        lines.append(
            f"  识别量 κ = Cd·A/m     = {result.drag_k_per_m:.6g} 1/m（数据只识别这一个数）"
        )
        lines.append(
            f"    ⇒ 按**实测**质量 {mass:g} kg 换算：Cd·A = {result.drag_area_m2:.6g} m²"
            + (
                ""
                if result.drag_area_sigma_m2 is None
                else f" ± {result.drag_area_sigma_m2:.3g} (1σ)"
            )
            + f"（面积固定 {area:g} m²）"
        )
        lines.append(
            "    ⚠ 质量填错多少，Cd 与 Cd·A 就跟着错多少（κ 不变）；"
            "换配重 m' 时按 κ' = Cd·A/m' 重算，不必重做试验"
        )
    if result.correlation:
        names = list(result.fitted)
        lines.append("  参数相关性：")
        for i, left in enumerate(names):
            row = result.correlation.get(left, {})
            pairs = [
                f"{right}={row.get(right, float('nan')):+.3f}"
                for j, right in enumerate(names)
                if j > i
            ]
            if pairs:
                lines.append(f"    {left}: " + ", ".join(pairs))
    lines.append(
        f"  条件数 {result.condition_number:.3g}（阈值 {max_condition_number:.0f}）"
        f"，最小二乘评估 {result.nfev} 次"
    )
    if result.rms_error_m is not None:
        lines.append(
            f"  残差：RMS {result.rms_error_m:.3f} m，最大 {result.max_error_m:.3f} m；"
            f"常数偏差 北 {result.bias_north_m:+.3f} / 东 {result.bias_east_m:+.3f} m"
        )
    if result.leave_one_out_rms_m is not None:
        lines.append(
            f"  留一交叉验证：RMS {result.leave_one_out_rms_m:.3f} m，"
            f"最大 {result.leave_one_out_max_m:.3f} m"
        )
    verdict = (
        "可信 → 可以按下面的参数回填配置" if result.reliable else "不可信 → 别回填，先看 warnings"
    )
    lines.append(f"  结论：ok={result.ok} reliable={result.reliable}：{verdict}")
    if result.reason:
        lines.append(f"  原因：{result.reason}")
    for warning in result.warnings:
        lines.append(f"  ⚠ {warning}")
    if result.at_bound:
        lines.append(f"  ⚠ 顶到边界的参数：{list(result.at_bound)}")
    if result.bias_north_m is not None and result.rms_error_m is not None:
        bias = (result.bias_north_m**2 + result.bias_east_m**2) ** 0.5
        if bias > max(0.5, 0.25 * result.rms_error_m):
            lines.append(
                f"  ⚠ 常数偏差 {bias:.3f} m 偏大：弹道参数解释不了固定偏移，"
                "先核对目标点坐标/测量点坐标/坐标解算，再谈调参"
            )
    return "\n".join(lines)


def config_snippet(result: FitResult) -> str:
    """可直接粘回配置的代码片段。"""
    values = result.ballistics
    lines = [
        "BallisticsConfig(",
        f"    mass_kg={values.mass_kg!r},          # ← 实测称重（不参与反演）",
        f"    drag_coefficient={values.drag_coefficient!r},",
        f"    cross_area_m2={values.cross_area_m2!r},   # ← 几何量（与 Cd 只以乘积可辨识）",
        f"    gravity={values.gravity!r},",
        f"    rk_dt={values.rk_dt!r},",
        f"    air_density_isa={values.air_density_isa!r},",
        f"    wind_source={values.wind_source!r},",
        ")",
    ]
    if result.drag_k_per_m is not None:
        lines.append("")
        lines.append(
            f"# 识别量 κ = Cd·A/m = {result.drag_k_per_m:.6g} 1/m（数据只识别这一个数）。"
            f"换配重 m'（形状不变 ⇒ Cd·A 不变）时："
            f"Cd' = κ·m'/A，A = {values.cross_area_m2:g} m²"
        )
    if "release_delay_s" in result.fitted:
        base = DropConfig()
        lines.append("")
        lines.append(
            "DropConfig("
            f"radius_m={base.radius_m!r}, delay_s={result.release_delay_s!r}, "
            f"force_after_pass={base.force_after_pass!r}, "
            f"evaluation_hz={base.evaluation_hz!r})"
        )
    if "wind_scale" in result.fitted:
        lines.append("")
        lines.append(f"# 记录到的风要乘 {result.wind_scale!r}（飞控风估计的标定系数）")
    return "\n".join(lines)


def build_report(
    result: FitResult,
    samples: tuple[DropSample, ...],
    ledger: list[dict[str, object]],
) -> dict[str, object]:
    """JSON 报告：账目 + 散点 + 拟合结果（给脚本/归档用）。"""
    return {
        "flights": ledger,
        "samples": [
            {
                "label": sample.name,
                "index": sample.index,
                "impact_ned": [round(v, 4) for v in sample.impact_ned],
                "impact_source": sample.impact_source,
                "record": sample.record.as_dict(),
            }
            for sample in samples
        ],
        "fit": result.as_dict(),
    }


# ----------------------------------------------------------------------
def build_config(**overrides) -> "FitConfig":
    """按关键字覆盖派生一份反演配置（``dataclasses.replace``；未知字段直接报错）。

    ``overrides`` 里除了 ``FitConfig`` 自己的字段，还可以给 ``mass_kg`` /
    ``cross_area_m2``（这两个是**实测输入**，写进配置里的 ``base``，不参与反演）。
    """
    from airdrop.ballistics import FitConfig  # 惰性：见模块 docstring

    mass_kg = overrides.pop("mass_kg", MEASURED_MASS_KG)
    cross_area_m2 = overrides.pop("cross_area_m2", MEASURED_CROSS_AREA_M2)
    settings = FitConfig(
        # 质量与（固定时的）迎风面积是**已知输入**：质量按实测填，不参与反演
        base=replace(BallisticsConfig(), mass_kg=mass_kg, cross_area_m2=cross_area_m2),
        fit_drag_coefficient=FIT_DRAG_COEFFICIENT,
        fit_cross_area_m2=FIT_CROSS_AREA_M2,
        fit_release_delay_s=FIT_RELEASE_DELAY_S,
        fit_wind_scale=FIT_WIND_SCALE,
        release_offset_body_m=RELEASE_OFFSET_BODY_M,
        max_condition_number=MAX_CONDITION_NUMBER,
    )
    return replace(settings, **overrides) if overrides else settings


def flight_inputs(
    *,
    drops: Path | None = None,
    impacts: Path | None = None,
    root: Path | None = None,
) -> tuple[FlightInput, ...]:
    """把"只跑一个架次"的覆盖（``--drops`` / ``--impacts``）变成 :class:`FlightInput` 序列。

    两个都不给就返回模块常量 :data:`FLIGHTS`（空 = 自动取最新架次）。
    """
    if drops is None and impacts is None:
        return FLIGHTS
    base = Path(FLIGHTS_ROOT if root is None else root) if drops is None else Path(drops)
    if drops is None:
        base = _latest_flight_dir(base)
    return (FlightInput(path=base, impacts=None if impacts is None else Path(impacts)),)


def main(
    flights: tuple[FlightInput, ...] | None = None,
    *,
    drops: Path | None = None,
    impacts: Path | None = None,
    output_path: Path | None = None,
    fit_config: "FitConfig | None" = None,
    **fit_overrides,
) -> int:
    """跑一次反演；返回进程退出码（0 可信 / 2 不可信 / 3 缺测量 / 1 出错）。

    ``fit_overrides`` 原样交给 :func:`build_config`（命令行 ``--mass-kg`` 之类）。
    """
    from airdrop.ballistics import (  # 惰性：见模块 docstring
        IMPACTS_JSONL_NAME,
        fit_ballistics,
        load_drops,
        write_impact_template,
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    target = Path(OUTPUT_PATH if output_path is None else output_path)
    settings = fit_config or build_config(**fit_overrides)
    if flights is None:
        flights = flight_inputs(drops=drops, impacts=impacts)
    if len(settings.fitted_names) == 0:
        print("配置里没有选中任何要拟合的参数（FIT_* 全为 False）", file=sys.stderr)
        return 1

    print("=" * 104)
    print("投放试验反演弹道参数")
    print("=" * 104)
    samples, ledger, missing_files = load_samples(flights)
    for entry in ledger:
        print(
            f"[架次] {entry['flight']}：投放 {entry['drops']} 次，"
            f"有实测落点 {entry['measured']} 次"
            + (f"，缺 {entry['missing']}" if entry["missing"] else "")
            + f"，测量文件 {entry['impacts'] or '（无）'}"
        )
    for drops_path in missing_files:
        if not WRITE_TEMPLATE:
            print(f"[提示] {drops_path.parent} 没有实测落点文件，先补上再跑", file=sys.stderr)
            continue
        template = write_impact_template(
            drops_path.parent / IMPACTS_JSONL_NAME, load_drops(drops_path)
        )
        print(f"[模板] 已生成待填落点：{template}")
    if not samples:
        print(
            "\n没有可用的『投放 + 实测落点』样本：先把实测落点填进模板"
            "（经纬度或 NED 都行），再重跑。",
            file=sys.stderr,
        )
        return 3

    result = fit_ballistics(samples, settings)
    print(describe(result, samples, max_condition_number=settings.max_condition_number))

    report = build_report(result, samples, ledger)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[报告] {target}")
    print("[回填] 把下面这段抄进配置（airdrop/config.py 的默认值或 examples 里的装配处）：")
    print(config_snippet(result))
    if not result.ok:
        return 2
    return 0 if result.reliable else 2


if __name__ == "__main__":
    sys.exit(main())
