"""Agent x portfolio x status views.

Two views over the same first-installment base:

* the crosstab fixes one side (a portfolio or an agent) and decomposes the other side by
  status dimension (generated, exception, broken, rejected);
* the ranking orders agents by value in one status dimension.
"""

from __future__ import annotations

from typing import Any

from llm_agent_evals.analytics.common import StatusBucket, StatusSets, as_float, as_int
from llm_agent_evals.business import BusinessConfig

Row = dict[str, Any]

DIMENSIONS = ("generated", "exceptions", "broken", "rejected")


def dimension_ids(config: BusinessConfig, dimension: str) -> tuple[int, ...]:
    """Status ids that make up one ranking dimension."""
    mapping = {
        "generated": config.generated_ids,
        "exceptions": config.exception_ids,
        "broken": config.broken_ids,
        "rejected": config.rejected_ids,
    }
    if dimension not in mapping:
        raise ValueError(f"dimension must be one of {DIMENSIONS}")
    return mapping[dimension]


def crosstab_summary(
    rows: list[Row], counterpart: str, config: BusinessConfig, reference_date: str
) -> list[dict[str, Any]]:
    """Decompose the free side. `counterpart` is "agent" or "portfolio"."""
    sets = StatusSets.from_config(config)
    buckets: dict[tuple[str, str], StatusBucket] = {}
    for row in rows:
        key = _counterpart_key(row, counterpart)
        if not key[0]:
            continue
        bucket = buckets.setdefault(key, StatusBucket())
        bucket.add(
            sets,
            as_int(row.get("status_id")),
            as_float(row.get("qty")),
            as_float(row.get("amount")),
        )
    entries = [
        {
            **_counterpart_fields(key, counterpart),
            "agreement_count": int(b.generated_qty),
            "first_installment_value": round(b.generated_value, 2),
            "exception_value": round(b.exception_value, 2),
            "broken_value": round(b.broken_value, 2),
            "rejected_value": round(b.rejected_value, 2),
            "reference_date": reference_date,
        }
        for key, b in buckets.items()
    ]
    entries.sort(key=lambda e: (-e["first_installment_value"], str(next(iter(e.values())))))
    return entries[: config.crosstab_limit]


def _counterpart_key(row: Row, counterpart: str) -> tuple[str, str]:
    if counterpart == "agent":
        return str(row.get("agent_id") or "").strip(), str(row.get("agent_name") or "").strip()
    return str(row.get("portfolio_name") or "").strip(), ""


def _counterpart_fields(key: tuple[str, str], counterpart: str) -> dict[str, str]:
    if counterpart == "agent":
        return {"agent_name": key[1] or key[0], "agent_id": key[0]}
    return {"portfolio_name": key[0]}


def agent_ranking(
    rows: list[Row], dimension: str, config: BusinessConfig, limit: int, reference_date: str
) -> dict[str, Any]:
    """Agents ordered by first-installment value in one status dimension."""
    wanted = set(dimension_ids(config, dimension))
    acc: dict[tuple[str, str], dict[str, float]] = {}
    for row in rows:
        if as_int(row.get("status_id")) not in wanted:
            continue
        key = (str(row.get("agent_id") or "").strip(), str(row.get("agent_name") or "").strip())
        if not key[0]:
            continue
        cell = acc.setdefault(key, {"qty": 0.0, "amount": 0.0})
        cell["qty"] += as_float(row.get("qty"))
        cell["amount"] += as_float(row.get("amount"))
    agents = [
        {
            "agent_name": name or agent_id,
            "agent_id": agent_id,
            "qty": int(cell["qty"]),
            "amount": round(cell["amount"], 2),
        }
        for (agent_id, name), cell in acc.items()
    ]
    agents.sort(key=lambda a: (-a["amount"], a["agent_id"]))
    return {"dimension": dimension, "agents": agents[:limit], "reference_date": reference_date}
