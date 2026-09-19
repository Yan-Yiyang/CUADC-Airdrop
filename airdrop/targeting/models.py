"""目标统计的数据模型：单次观测 → 聚类 → 唯一结果。

坐标一律用 NED（米）——那是 :mod:`airdrop.georef` 的输出；WGS84 由调用方在最后
一步用 ``georef.geo.ned_to_wgs84`` 换算（这里不做，免得把原点配置也拖进来）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

__all__ = ["Cluster", "TargetPoint", "TargetingResult"]


@dataclass(frozen=True, slots=True)
class TargetPoint:
    """一次目标观测（一帧里一个目标 + georef 解算出的地面坐标）。

    ``code`` 为 None 表示看到了目标但没读出编号（OCR 失败或编号不在 00–99）：
    这种点仍然能聚类（它证明那里确实有个目标），但没有编号就参与不了
    median/max 筛选——编号是筛选的依据，取不了中位数。
    """

    north_m: float
    east_m: float
    capture_timestamp: float
    frame_index: int = -1
    code: int | None = None
    confidence: float = 1.0
    down_m: float = 0.0

    @property
    def ned(self) -> tuple[float, float, float]:
        return (self.north_m, self.east_m, self.down_m)

    @property
    def labeled(self) -> bool:
        """是否有有效编号（决定它能不能当筛选依据）。"""
        return self.code is not None

    @property
    def finite(self) -> bool:
        """坐标是否有限——georef 病态时会给 inf/nan，那种点不能进聚类。"""
        return all(math.isfinite(value) for value in (self.north_m, self.east_m))

    def as_dict(self) -> dict[str, Any]:
        return {
            "north_m": self.north_m,
            "east_m": self.east_m,
            "down_m": self.down_m,
            "capture_timestamp": self.capture_timestamp,
            "frame_index": self.frame_index,
            "code": self.code,
            "confidence": self.confidence,
        }


@dataclass(frozen=True, slots=True)
class Cluster:
    """一个候选目标：DBSCAN 聚出来的一簇观测。"""

    label: int
    members: tuple[TargetPoint, ...]
    north_m: float
    east_m: float
    code: int | None
    confidence: float

    @property
    def count(self) -> int:
        return len(self.members)

    @property
    def labeled(self) -> bool:
        return self.code is not None

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "count": self.count,
            "north_m": self.north_m,
            "east_m": self.east_m,
            "code": self.code,
            "confidence": self.confidence,
        }


@dataclass(frozen=True, slots=True)
class TargetingResult:
    """一次统计的完整结果：唯一结果 + 全部候选类 + 被剔除的点。

    ``selected`` 是 None 表示没有可用结果（没看到目标 / 全被剔除 / 没有带编号的类），
    调用方要按"无结果"处理（计划里走备用点分支），不要拿 ``clusters`` 里的东西凑合。
    """

    selection_rule: str = "median"
    clusters: tuple[Cluster, ...] = ()
    selected: Cluster | None = None
    noise: tuple[TargetPoint, ...] = ()
    rejected: tuple[TargetPoint, ...] = ()

    @property
    def ok(self) -> bool:
        return self.selected is not None

    @property
    def north_m(self) -> float | None:
        return None if self.selected is None else self.selected.north_m

    @property
    def east_m(self) -> float | None:
        return None if self.selected is None else self.selected.east_m

    @property
    def code(self) -> int | None:
        return None if self.selected is None else self.selected.code

    @property
    def ned(self) -> tuple[float, float, float] | None:
        """结果坐标（米，NED）；无结果返回 None。"""
        if self.selected is None:
            return None
        return (self.selected.north_m, self.selected.east_m, 0.0)

    def as_dict(self) -> dict[str, Any]:
        """写入磁盘与日志用。``selected`` 单独给一份，便于直接取坐标与编号。"""
        return {
            "selection_rule": self.selection_rule,
            "selected": None if self.selected is None else self.selected.as_dict(),
            "clusters": [cluster.as_dict() for cluster in self.clusters],
            "counts": {
                "points": sum(c.count for c in self.clusters) + len(self.noise),
                "clusters": len(self.clusters),
                "labeled_clusters": sum(1 for c in self.clusters if c.labeled),
                "noise": len(self.noise),
                "rejected": len(self.rejected),
            },
        }
