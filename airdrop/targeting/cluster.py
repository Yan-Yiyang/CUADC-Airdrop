"""DBSCAN 聚类 + 跨类筛选（计划 4.5 / Q8）。

方法学（每一条都是刻意的，别顺便改）
------------------------------------
* 输入先排序：按 ``(capture_timestamp, frame_index)`` 排好再聚类。DBSCAN 的标签号
  按首次访问顺序分配，"同一个点集换个到达顺序就换个 label"会让结果不可复现。
* 只聚类有限坐标：georef 在视线几乎与地面平行等病态情形会给出 inf/nan，
  这种点会让整批 eps 邻域计算失效，单独放进 ``rejected`` 并计数，绝不静默丢。
* 置信度加权必须归一化：sklearn 的 ``sample_weight`` 是绝对权重，直接进核心点
  判据 ``Σw ≥ min_samples``。实测（sklearn 1.9.1）：权重 ``[1.0, 0.1]``、``min_samples=2``
  时那一对双双被打成噪声。若直接用置信度（恒 <1）当权重，两个 0.9 的点加起来 1.8 < 2，
  真实目标会被整片剔掉。所以权重按均值归一到 1：总质量 = 点数，
  ``min_samples`` 仍然表示"至少看到几次"，相对置信度只调制核心点判据。
* 类标签 = 类内 ``code`` 众数（平票取较小值；全 None → 该类无标签）；
  结果坐标 = 类内均值（加权开关打开时为加权均值）。
* 跨类筛选只在中带编号的类里做：median 取标签的下中位数（偶数个取下中位），
  max 取标签最大。None 取不了中位数，所以无编号的类无论如何都不参与筛选；
  配置里的 ``require_label`` 控制的是"无编号的类要不要留在结果列表里"。
* 同一标签有多个类时（两个目标印着同一个编号）取成员最多者，再比平均置信度，
  再比 ``(north, east)``——结果唯一且可复现，不靠 DBSCAN 的标签序碰运气。
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Sequence

import numpy as np
from sklearn.cluster import DBSCAN

from ..config import TargetingConfig
from .models import Cluster, TargetingResult, TargetPoint

LOGGER = logging.getLogger(__name__)

__all__ = ["analyze", "cluster_points", "select_cluster"]

#: 置信度权重的下限（防止 0/负置信度把权重压成 0，那等于把这个点从质心里抹掉）
MIN_WEIGHT = 1e-6


def _ordered(points: Sequence[TargetPoint]) -> list[TargetPoint]:
    """按 (拍摄时刻, 帧号) 排序，让结果与调用方的到达顺序无关。"""
    return sorted(points, key=lambda p: (p.capture_timestamp, p.frame_index))


def _split_finite(
    points: Sequence[TargetPoint],
) -> tuple[list[TargetPoint], list[TargetPoint]]:
    usable = [p for p in points if p.finite]
    rejected = [p for p in points if not p.finite]
    if rejected:
        LOGGER.warning("剔除 %d 个坐标非有限的目标点（georef 病态）", len(rejected))
    return usable, rejected


def _weights(points: Sequence[TargetPoint], config: TargetingConfig) -> np.ndarray | None:
    """DBSCAN 的样本权重；未开启加权时返回 None。

    归一化到均值 1 的理由见模块 docstring（sklearn 的权重是绝对的）。
    """
    if not config.weight_by_confidence:
        return None
    raw = np.array([max(float(p.confidence), MIN_WEIGHT) for p in points], dtype=np.float64)
    mean = float(raw.mean())
    if mean <= 0.0:  # 理论上到不了（上面已夹住下限）
        return np.ones(len(points), dtype=np.float64)
    return raw / mean


def _labeled_codes(members: Sequence[TargetPoint]) -> list[int]:
    """成员里**有编号**的那些编号。

    ``labeled`` 是 TargetPoint 上的派生属性（``code is not None``），类型检查器
    没法据此收窄 ``code``——所以这里显式判 None，顺带给下面几处复用。
    """
    return [code for point in members if (code := point.code) is not None]


def _mode_code(members: Sequence[TargetPoint]) -> int | None:
    """类内编号众数；平票取较小值；一个有效编号都没有 → None。

    刻意不用置信度加权：计划里加权开关只说"类内均值也变加权均值"，
    众数保持纯计数——一个高置信度的错读不该靠分数压过多数票。
    """
    counts = Counter(_labeled_codes(members))
    if not counts:
        return None
    best = max(counts.values())
    return min(code for code, count in counts.items() if count == best)


def _aggregate(
    label: int,
    members: Sequence[TargetPoint],
    weights: np.ndarray | None,
) -> Cluster:
    """把一簇成员折成一个 :class:`Cluster`（均值/加权均值 + 众数 + 置信度）。"""
    north = np.array([p.north_m for p in members], dtype=np.float64)
    east = np.array([p.east_m for p in members], dtype=np.float64)
    confidence = np.array([float(p.confidence) for p in members], dtype=np.float64)
    if weights is None:
        north_m, east_m = float(north.mean()), float(east.mean())
        mean_confidence = float(confidence.mean())
    else:
        total = float(weights.sum())
        north_m = float((north * weights).sum() / total)
        east_m = float((east * weights).sum() / total)
        mean_confidence = float((confidence * weights).sum() / total)
    return Cluster(
        label=label,
        members=tuple(members),
        north_m=north_m,
        east_m=east_m,
        code=_mode_code(members),
        confidence=mean_confidence,
    )


def cluster_points(
    points: Sequence[TargetPoint],
    config: TargetingConfig,
) -> tuple[list[Cluster], list[TargetPoint], list[TargetPoint]]:
    """DBSCAN 聚类；返回 ``(候选类, 噪声点, 被剔除的非有限点)``。

    坐标用 ``(north, east)`` 两维——目标都在地面上，高度由 georef 单独互校，
    把 z 掺进 eps 只会让噪声（地面交点抖动）影响水平聚类。
    """
    ordered = _ordered(points)
    usable, rejected = _split_finite(ordered)
    if not usable:
        return [], [], rejected

    weights = _weights(usable, config)
    features = np.array([[p.north_m, p.east_m] for p in usable], dtype=np.float64)
    labels = DBSCAN(eps=config.eps_m, min_samples=config.min_samples).fit_predict(
        features, sample_weight=weights
    )

    clusters: list[Cluster] = []
    noise: list[TargetPoint] = []
    for label in sorted({int(value) for value in labels}):
        index = np.flatnonzero(labels == label)
        if label < 0:
            noise.extend(usable[i] for i in index)
            continue
        members = [usable[i] for i in index]
        member_weights = None if weights is None else weights[index]
        clusters.append(_aggregate(label, members, member_weights))
    return clusters, noise, rejected


def select_cluster(
    clusters: Sequence[Cluster],
    config: TargetingConfig,
) -> Cluster | None:
    """按 ``selection_rule`` 选出唯一的候选类；选不出返回 None。

    * 只有带编号的类参与（中位数/最大值都要数字）；
    * ``median``：标签排序后取下中位数（偶数个取靠下的那个）；
    * ``max``：标签最大；
    * 同一标签有多个类时：成员多者优先，再比平均置信度，再比坐标（保证可复现）；
    * 未知规则显式报错，不做隐式回退（计划 Q8）。
    """
    candidates = [cluster for cluster in clusters if cluster.labeled]
    if not candidates:
        return None

    if config.selection_rule == "median":
        labels = sorted(cluster.code for cluster in candidates if cluster.code is not None)
        target = labels[(len(labels) - 1) // 2]  # 偶数个 → 下中位
    elif config.selection_rule == "max":
        target = max(cluster.code for cluster in candidates if cluster.code is not None)
    else:
        raise ValueError(f"未知的 selection_rule: {config.selection_rule!r}（只支持 median / max）")

    same_label = [cluster for cluster in candidates if cluster.code == target]
    same_label.sort(key=lambda c: (-c.count, -c.confidence, c.north_m, c.east_m))
    return same_label[0]


def analyze(
    points: Sequence[TargetPoint],
    config: TargetingConfig,
) -> TargetingResult:
    """一次完整的统计：聚类 → 选唯一结果。实飞与回放走同一入口。

    ``require_label=True``（默认）会把没有有效编号的候选类整个丢掉（计划 4.5
    "忽略无有效编码的类"）；置 False 时它们留在 ``clusters`` 里供人工核对，
    但仍然不会被选中——中位数/最大值都要数字，None 取不了。
    """
    clusters, noise, rejected = cluster_points(points, config)
    kept = clusters
    if config.require_label:
        kept = [cluster for cluster in clusters if cluster.labeled]
        if len(kept) != len(clusters):
            LOGGER.info(
                "忽略 %d 个无有效编号的候选类（require_label=True）",
                len(clusters) - len(kept),
            )
    selected = select_cluster(kept, config)
    result = TargetingResult(
        selection_rule=config.selection_rule,
        clusters=tuple(kept),
        selected=selected,
        noise=tuple(noise),
        rejected=tuple(rejected),
    )
    if selected is None:
        outcome = "无（没有带编号的候选类）"
    else:
        outcome = f"编号 {selected.code} @ ({selected.north_m:.2f}, {selected.east_m:.2f})"
    LOGGER.info(
        "targeting：%d 点 → %d 类（%d 带编号）+ %d 噪声 + %d 剔除；结果 %s",
        len(points),
        len(kept),
        sum(1 for c in kept if c.labeled),
        len(noise),
        len(rejected),
        outcome,
    )
    return result
