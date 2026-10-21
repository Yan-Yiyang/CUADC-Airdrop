from __future__ import annotations

import math
import statistics
from collections.abc import Iterable
from pathlib import Path


def percentile_summary(values: Iterable[float]) -> dict[str, float | int | None]:
    """汇总毫秒样本，空输入返回空统计。"""
    items = sorted(float(value) for value in values)
    if not items:
        return {
            "count": 0,
            "mean_ms": None,
            "p50_ms": None,
            "p95_ms": None,
            "p99_ms": None,
            "max_ms": None,
        }

    def percentile(fraction: float) -> float:
        rank = (len(items) - 1) * fraction
        lower = math.floor(rank)
        upper = math.ceil(rank)
        return (
            items[lower]
            if lower == upper
            else items[lower] * (upper - rank) + items[upper] * (rank - lower)
        )

    return {
        "count": len(items),
        "mean_ms": statistics.fmean(items),
        "p50_ms": percentile(0.50),
        "p95_ms": percentile(0.95),
        "p99_ms": percentile(0.99),
        "max_ms": items[-1],
    }


def write_json_new(path: Path, payload: object) -> None:
    """以独占创建方式保存 JSON，防止覆盖已有诊断结果。"""
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
