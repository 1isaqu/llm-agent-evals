"""Portfolio risk entries and the status breakdown, built from `status_rollup` rows.

Rules (kept identical to the system prompt, change both together):

* the denominator of every percentage is the first-installment universe of the
  window: generated + exception + rejected value;
* composite risk is the maximum of the three axes, never their sum;
* an axis outside 0..100% marks `anomaly=true`. Every axis is a slice of a universe that
  contains it, so with non-negative amounts it cannot leave that range. It only happens
  when amounts are negative (reversed rows, credit notes mapped by mistake) or when
  sources with different windows were combined. It is reported as a data-quality alarm
  and never silently corrected. A high value at or below 100% is real risk, not an
  anomaly.
"""

from __future__ import annotations

from typing import Any

from llm_agent_evals.analytics.common import StatusBucket, StatusSets, as_float, as_int
from llm_agent_evals.business import BusinessConfig

Row = dict[str, Any]


def portfolio_entries(
    rows: list[Row], config: BusinessConfig, reference_date: str
) -> list[dict[str, Any]]:
    """One entry per portfolio, sorted by first-installment value (largest first)."""
    sets = StatusSets.from_config(config)
    buckets: dict[str, StatusBucket] = {}
    for row in rows:
        name = str(row.get("portfolio_name") or "").strip()
        if not name:
            continue
        bucket = buckets.setdefault(name, StatusBucket())
        bucket.add(
            sets,
            as_int(row.get("status_id")),
            as_float(row.get("qty")),
            as_float(row.get("amount")),
        )
    entries = [_entry(name, bucket, config, reference_date) for name, bucket in buckets.items()]
    entries.sort(key=lambda e: (-e["first_installment_value"], e["portfolio_name"]))
    return entries


def _entry(
    name: str, bucket: StatusBucket, config: BusinessConfig, reference_date: str
) -> dict[str, Any]:
    axes = bucket.risk_axes()
    composite = max(axes.values())
    return {
        "portfolio_name": name,
        "first_installment_value": round(bucket.generated_value, 2),
        "agreement_count": int(bucket.generated_qty),
        "composite_risk_pct": composite,
        "breakdown": axes,
        "risk_level": config.risk_level(composite),
        "anomaly": any(not 0.0 <= value <= 100.0 for value in axes.values()),
        "reference_date": reference_date,
    }


def status_breakdown(
    rows: list[Row], config: BusinessConfig, reference_date: str
) -> dict[str, Any]:
    """Quantity and first-installment value per status, with explicit generated totals.

    `all_statuses_*` adds every status listed and includes exception and rejected
    value, so it is not the generated total. The generated totals are returned
    separately, so nothing has to be summed by hand to quote "generated".
    """
    known = set(config.rollup_ids)
    totals: dict[int, dict[str, float]] = {}
    for row in rows:
        status_id = as_int(row.get("status_id"))
        if status_id not in known:
            continue
        bucket = totals.setdefault(status_id, {"qty": 0.0, "amount": 0.0})
        bucket["qty"] += as_float(row.get("qty"))
        bucket["amount"] += as_float(row.get("amount"))
    statuses = [
        {
            "status_id": status_id,
            "label": config.label_of(status_id),
            "qty": int(values["qty"]),
            "amount": round(values["amount"], 2),
        }
        for status_id, values in totals.items()
    ]
    statuses.sort(key=lambda s: (-s["amount"], s["status_id"]))
    generated = set(config.generated_ids)
    generated_rows = [s for s in statuses if s["status_id"] in generated]
    return {
        "statuses": statuses,
        "generated_qty": sum(s["qty"] for s in generated_rows),
        "generated_amount": round(sum(s["amount"] for s in generated_rows), 2),
        "all_statuses_qty": sum(s["qty"] for s in statuses),
        "all_statuses_amount": round(sum(s["amount"] for s in statuses), 2),
        "note": (
            "Quantity and first-installment value per status in the window. Generated "
            f"statuses are: {', '.join(config.generated_labels)}. all_statuses_* also "
            "includes exception and rejected value and must not be called generated."
        ),
        "reference_date": reference_date,
    }
