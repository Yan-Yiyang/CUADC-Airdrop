"""目标统计：把一批观测点聚成候选目标，按编号 median/max 选出唯一结果。

数据流位置：``Detection`` + georef → :class:`TargetPoint` → 本模块 → 结果坐标
（→ WGS84 → 飞掠任务）。实飞与回放走同一入口（计划 5 章）。

* :mod:`~airdrop.targeting.models`：:class:`TargetPoint` / :class:`Cluster` /
  :class:`TargetingResult`；
* :mod:`~airdrop.targeting.cluster`：DBSCAN 聚类与跨类筛选（方法学写在模块 docstring 里）。
"""

from .cluster import analyze, cluster_points, select_cluster
from .models import Cluster, TargetingResult, TargetPoint

__all__ = [
    "Cluster",
    "TargetPoint",
    "TargetingResult",
    "analyze",
    "cluster_points",
    "select_cluster",
]
