"""Daily series, trend detection, re-bucketing and paging for KPI history.

Weekly and monthly series are built client-side from correct daily points (no dynamic
SQL): sums for value and count, the *maximum* for risk (the worst day of the period,
consistent with "composite risk is the worst axis").
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

from llm_agent_evals.analytics.common import StatusBucket, StatusSets, as_float, as_int
from llm_agent_evals.business import BusinessConfig
from llm_agent_evals.session import Window

Row = dict[str, Any]

PAGE_SIZE = 31
METRICS = ("value", "count", "risk")


def daily_points(
    rows: list[Row], config: BusinessConfig, metric: str, window: Window
) -> list[dict[str, Any]]:
    """Continuous daily points over the window. Days without rows become zero."""
    if metric not in METRICS:
        raise ValueError(f"metric must be one of {METRICS}")
    sets = StatusSets.from_config(config)
    per_day: dict[str, StatusBucket] = {}
    for row in rows:
        if row.get("day") is None:
            continue
        bucket = per_day.setdefault(str(row["day"])[:10], StatusBucket())
        bucket.add(
            sets,
            as_int(row.get("status_id")),
            as_float(row.get("qty")),
            as_float(row.get("amount")),
        )
    points: list[dict[str, Any]] = []
    current = window.start
    while current <= window.end:
        points.append(
            {
                "date": current.isoformat(),
                "value": _point_value(per_day.get(current.isoformat()), metric),
            }
        )
        current += timedelta(days=1)
    return points


def _point_value(bucket: StatusBucket | None, metric: str) -> float | int:
    if bucket is None:
        return 0 if metric == "count" else 0.0
    if metric == "value":
        return round(bucket.generated_value, 2)
    if metric == "count":
        return int(bucket.generated_qty)
    return bucket.composite_risk_pct()


def trend(points: list[dict[str, Any]], flat_band_pct: float) -> tuple[str, float | None]:
    """Compare the mean of the second half of the window with the first half.

    A dead zone of plus or minus `flat_band_pct` percent reads as "flat" (a business
    parameter, see `BusinessConfig.trend_flat_band_pct`). The change is None when the first
    half averages zero, where a percentage change has no meaning.
    """
    values = [float(p["value"]) for p in points]
    if len(values) < 2:
        return "flat", None
    half = len(values) // 2
    first = sum(values[:half]) / half
    second = sum(values[half:]) / (len(values) - half)
    change = round((second - first) * 100.0 / first, 2) if first > 0 else None
    band = flat_band_pct / 100.0
    if second > first * (1 + band):
        return "rising", change
    if second < first * (1 - band):
        return "falling", change
    return "flat", change


def bucket_key(day: date, granularity: str) -> str:
    if granularity == "month":
        return day.replace(day=1).isoformat()
    if granularity == "week":
        return (day - timedelta(days=day.weekday())).isoformat()
    return day.isoformat()


def rebucket(points: list[dict[str, Any]], kpi: str, granularity: str) -> list[dict[str, Any]]:
    """Aggregate daily points to weeks (starting Monday) or months."""
    if granularity == "day":
        return points
    buckets: dict[str, list[float]] = {}
    for point in points:
        key = bucket_key(date.fromisoformat(point["date"]), granularity)
        buckets.setdefault(key, []).append(float(point["value"]))
    out: list[dict[str, Any]] = []
    for key in sorted(buckets):
        values = buckets[key]
        if kpi == "composite_risk_pct":
            aggregated: float | int = round(max(values), 2)
        elif kpi == "agreement_count":
            aggregated = int(sum(values))
        else:
            aggregated = round(sum(values), 2)
        out.append({"date": key, "value": aggregated})
    return out


def paginate(points: list[dict[str, Any]], page: int, page_size: int = PAGE_SIZE) -> dict[str, Any]:
    """Slice `points` into a page and describe the paging."""
    start = (page - 1) * page_size
    total_pages = max(1, (len(points) + page_size - 1) // page_size)
    return {
        "data": points[start : start + page_size],
        "meta": {
            "page": page,
            "total_pages": total_pages,
            "total_points": len(points),
            "truncated": page < total_pages,
        },
    }
