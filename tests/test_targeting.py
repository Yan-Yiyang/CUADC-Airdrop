"""targeting 离线单测：DBSCAN 聚类、编号众数、median/max 筛选、噪声与边界。

全部离线：合成点云，不碰 GPU / 飞控 / 相机。
验收口径来自计划 P7：众数 / median / max / 均值 / eps 边界 / 噪声剔除。
"""

from __future__ import annotations

import math

import pytest

from airdrop.config import TargetingConfig
from airdrop.targeting import Cluster, TargetPoint, analyze, cluster_points, select_cluster


def _point(
    north: float,
    east: float,
    code: int | None,
    *,
    timestamp: float = 0.0,
    frame: int = 0,
    confidence: float = 1.0,
) -> TargetPoint:
    return TargetPoint(
        north_m=north,
        east_m=east,
        capture_timestamp=timestamp,
        frame_index=frame,
        code=code,
        confidence=confidence,
    )


def _cloud(north: float, east: float, code: int | None, count: int) -> list[TargetPoint]:
    """在一点附近撒 count 个观测（间隔远小于 eps）。"""
    return [
        _point(north + 0.01 * i, east - 0.01 * i, code, timestamp=float(i), frame=i)
        for i in range(count)
    ]


# ----------------------------------------------------------------------
# 聚类：基本行为
# ----------------------------------------------------------------------
def test_two_targets_are_clustered_apart() -> None:
    points = _cloud(0.0, 0.0, 56, 4) + _cloud(10.0, 5.0, 95, 4)
    result = analyze(points, TargetingConfig())

    assert len(result.clusters) == 2
    assert result.selected is not None
    by_code = {cluster.code: cluster for cluster in result.clusters}
    assert set(by_code) == {56, 95}
    assert by_code[56].count == 4
    assert by_code[56].north_m == pytest.approx(0.015, abs=1e-9)


def test_single_observation_is_noise() -> None:
    """min_samples=2：孤立的单次观测进不了任何类（计划里正是要剔掉野点）。"""
    points = _cloud(0.0, 0.0, 56, 3) + [_point(50.0, 50.0, 95)]
    result = analyze(points, TargetingConfig())

    assert len(result.clusters) == 1
    assert len(result.noise) == 1
    assert result.noise[0].code == 95


def test_all_isolated_points_yield_no_result() -> None:
    points = [_point(0.0, 0.0, 1), _point(100.0, 0.0, 2), _point(0.0, 100.0, 3)]
    result = analyze(points, TargetingConfig())

    assert result.clusters == ()
    assert result.selected is None
    assert not result.ok
    assert result.north_m is None and result.ned is None
    assert len(result.noise) == 3


def test_empty_input_is_safe() -> None:
    result = analyze([], TargetingConfig())
    assert result.selected is None
    assert result.as_dict()["counts"]["points"] == 0


def test_eps_boundary_is_inclusive() -> None:
    """eps 边界实测为闭区间（sklearn 1.9.1）：恰好 eps 距离算邻居，差一点就不算。

    这条钉住的是"eps=0.75 到底含不含 0.75"——它决定相邻目标会不会被并成一个。
    """
    config = TargetingConfig(eps_m=0.75)
    exactly = [_point(0.0, 0.0, 56), _point(0.75, 0.0, 56)]
    assert len(cluster_points(exactly, config)[0]) == 1, "恰好 eps 应当相连"

    just_over = [_point(0.0, 0.0, 56), _point(0.7501, 0.0, 56)]
    clusters, noise, _ = cluster_points(just_over, config)
    assert clusters == [] and len(noise) == 2, "超过 eps 应当各自成噪声"


def test_non_finite_coordinates_are_rejected_not_clustered() -> None:
    """georef 病态时会给 inf/nan：必须单独剔除并计数，不能塞进 DBSCAN。"""
    points = _cloud(0.0, 0.0, 56, 3) + [
        _point(float("inf"), 0.0, 56),
        _point(float("nan"), 0.0, 56),
    ]
    result = analyze(points, TargetingConfig())

    assert len(result.clusters) == 1
    assert len(result.rejected) == 2
    assert all(not point.finite for point in result.rejected)


# ----------------------------------------------------------------------
# 类标签：编号众数
# ----------------------------------------------------------------------
def test_cluster_label_is_code_mode() -> None:
    points = _cloud(0.0, 0.0, 56, 3) + _cloud(0.0, 0.0, 95, 1)
    result = analyze(points, TargetingConfig())
    assert len(result.clusters) == 1
    assert result.clusters[0].code == 56, "众数应当是出现更多的 56"


def test_mode_tie_takes_smaller_code() -> None:
    points = _cloud(0.0, 0.0, 95, 2) + _cloud(0.0, 0.0, 56, 2)
    result = analyze(points, TargetingConfig())
    assert result.clusters[0].code == 56, "平票取较小值"


def test_cluster_without_any_code_has_no_label() -> None:
    points = [_point(0.0, 0.0, None), _point(0.05, 0.0, None)]
    result = analyze(points, TargetingConfig(require_label=False))
    assert len(result.clusters) == 1
    assert result.clusters[0].code is None
    assert result.selected is None, "无编号的类取不了中位数，永远选不中"


def test_require_label_drops_unlabeled_clusters_from_output() -> None:
    points = _cloud(0.0, 0.0, None, 3) + _cloud(20.0, 0.0, 56, 3)
    with_label = analyze(points, TargetingConfig(require_label=True))
    without = analyze(points, TargetingConfig(require_label=False))

    assert len(with_label.clusters) == 1 and with_label.code == 56
    assert len(without.clusters) == 2, "关掉后无编号的类要留在列表里供核对"
    assert without.code == 56, "但它仍然不会被选中"


# ----------------------------------------------------------------------
# 跨类筛选：median / max
# ----------------------------------------------------------------------
def test_median_rule_picks_lower_median_for_even_labels() -> None:
    """三个目标 10/20/30 取中位 20；四个 10/20/30/40 取下中位 20。"""
    odd = _cloud(0.0, 0.0, 10, 2) + _cloud(50.0, 0.0, 20, 2) + _cloud(100.0, 0.0, 30, 2)
    assert analyze(odd, TargetingConfig(selection_rule="median")).code == 20

    even = odd + _cloud(150.0, 0.0, 40, 2)
    assert analyze(even, TargetingConfig(selection_rule="median")).code == 20


def test_max_rule_picks_largest_label() -> None:
    points = _cloud(0.0, 0.0, 10, 2) + _cloud(50.0, 0.0, 20, 2) + _cloud(100.0, 0.0, 30, 2)
    result = analyze(points, TargetingConfig(selection_rule="max"))
    assert result.code == 30
    assert result.north_m == pytest.approx(100.0, abs=0.1)


def test_same_label_prefers_bigger_cluster() -> None:
    """两个目标印着同一编号：取成员多者（不能靠 DBSCAN 标签序碰运气）。"""
    points = _cloud(0.0, 0.0, 56, 2) + _cloud(80.0, 0.0, 56, 5)
    result = analyze(points, TargetingConfig())

    assert len(result.clusters) == 2
    assert result.selected is not None
    assert result.selected.count == 5
    assert result.north_m == pytest.approx(80.0, abs=0.1)


def test_unknown_selection_rule_raises() -> None:
    """未知规则必须显式报错——计划 Q8 明说"代码不做隐式回退"。"""
    clusters = [
        Cluster(
            label=0,
            members=(_point(0.0, 0.0, 56),),
            north_m=0.0,
            east_m=0.0,
            code=56,
            confidence=1.0,
        )
    ]
    with pytest.raises(ValueError, match="selection_rule"):
        select_cluster(clusters, TargetingConfig(selection_rule="mean"))


# ----------------------------------------------------------------------
# 均值 / 加权
# ----------------------------------------------------------------------
def test_centroid_is_plain_mean_by_default() -> None:
    """类内均值就是算术平均（两点必须落在 eps 内，否则根本不成类）。"""
    points = [_point(0.0, 0.0, 56, confidence=1.0), _point(0.2, 0.4, 56, confidence=0.1)]
    result = analyze(points, TargetingConfig())
    assert result.north_m == pytest.approx(0.1)
    assert result.east_m == pytest.approx(0.2)


def test_weighted_centroid_follows_confidence() -> None:
    """开加权后均值偏向高置信度的点（0.9 : 0.1 → 偏向 0 那侧）。"""
    points = [_point(0.0, 0.0, 56, confidence=0.9), _point(0.6, 0.0, 56, confidence=0.1)]
    result = analyze(points, TargetingConfig(weight_by_confidence=True))

    assert result.north_m is not None
    assert result.north_m < 0.15, "应当明显偏向 0.0 那侧（无权重的均值是 0.3）"
    assert result.north_m == pytest.approx(0.06, abs=0.02)


def test_weighted_clustering_still_counts_detections() -> None:
    """回归：sklearn 的 sample_weight 是绝对权重，直接用置信度会把真实目标打成噪声。

    实测（1.9.1）：权重 [1.0, 0.1]、min_samples=2 时那一对双双成为噪声。所以本实现把
    权重按均值归一到 1（总质量 = 点数），min_samples 仍是"至少看到几次"。
    """
    points = [_point(0.0, 0.0, 56, confidence=0.9), _point(0.1, 0.0, 56, confidence=0.9)]
    result = analyze(points, TargetingConfig(weight_by_confidence=True))

    assert len(result.clusters) == 1, "两个 0.9 置信度的观测必须仍能成类"
    assert result.code == 56


def test_confidence_zero_does_not_break_weights() -> None:
    """置信度为 0 的点不能把权重压成 0（那等于从质心里抹掉它）。"""
    points = [
        _point(0.0, 0.0, 56, confidence=0.0),
        _point(0.5, 0.0, 56, confidence=0.0),
    ]
    result = analyze(points, TargetingConfig(weight_by_confidence=True))
    assert len(result.clusters) == 1
    assert result.north_m == pytest.approx(0.25)


# ----------------------------------------------------------------------
# 可复现性
# ----------------------------------------------------------------------
def test_result_is_independent_of_input_order() -> None:
    points = _cloud(0.0, 0.0, 56, 3) + _cloud(30.0, 0.0, 95, 3) + [_point(99.0, 9.0, None)]
    forward = analyze(points, TargetingConfig())
    backward = analyze(list(reversed(points)), TargetingConfig())

    assert forward.selected is not None and backward.selected is not None
    assert forward.code == backward.code
    assert forward.north_m == pytest.approx(backward.north_m)
    assert [c.count for c in forward.clusters] == [c.count for c in backward.clusters]


def test_counts_add_up() -> None:
    points = _cloud(0.0, 0.0, 56, 3) + [_point(80.0, 0.0, 95)]
    counts = analyze(points, TargetingConfig()).as_dict()["counts"]
    assert counts["points"] == 4
    assert counts["clusters"] == 1
    assert counts["labeled_clusters"] == 1
    assert counts["noise"] == 1
    assert counts["rejected"] == 0


def test_distance_used_for_eps_is_horizontal() -> None:
    """eps 只看水平距离：同样的 (north, east) 差、不同的 down 不改变聚类。"""
    near = [
        TargetPoint(north_m=0.0, east_m=0.0, down_m=0.0, capture_timestamp=0.0, code=56),
        TargetPoint(north_m=0.5, east_m=0.0, down_m=100.0, capture_timestamp=1.0, code=56),
    ]
    clusters, _, _ = cluster_points(near, TargetingConfig())
    assert len(clusters) == 1
    assert clusters[0].count == 2
    assert math.isclose(clusters[0].north_m, 0.25)
