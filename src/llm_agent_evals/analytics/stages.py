"""Payment-plan stage of approved agreements.

The base is *approved* agreements issued in the lookback window ending at the reference
date: plan stage is a property of the live book of agreements, not of the dashboard window.
The stage comes from payment progress (an installment with `paid_amount > 0` is paid):

    settled  every installment paid
    early    at most one installment paid
    late     `plan_late_remaining` or fewer installments left to pay
    middle   everything else

`open_amount` is the sum of the amounts of installments not yet paid.
"""

from __future__ import annotations

from typing import Any

from llm_agent_evals.analytics.common import as_float, as_int
from llm_agent_evals.business import BusinessConfig

Row = dict[str, Any]

STAGES = ("early", "middle", "late", "settled")


def stage_summary(
    rows: list[Row], stage: str | None, config: BusinessConfig, reference_date: str
) -> dict[str, Any]:
    """Totals per stage. With `stage`, also the portfolios holding the most open amount in it."""
    per_stage = {name: {"qty": 0, "open_amount": 0.0} for name in STAGES}
    per_portfolio: dict[str, dict[str, dict[str, float]]] = {}
    for row in rows:
        name = str(row.get("stage") or "").strip()
        if name not in per_stage:
            continue
        qty, amount = as_int(row.get("qty")), as_float(row.get("open_amount"))
        per_stage[name]["qty"] += qty
        per_stage[name]["open_amount"] += amount
        portfolio = str(row.get("portfolio_name") or "").strip()
        if portfolio:
            cells = per_portfolio.setdefault(
                portfolio, {s: {"qty": 0, "open_amount": 0.0} for s in STAGES}
            )
            cells[name]["qty"] += qty
            cells[name]["open_amount"] += amount
    total = sum(int(v["qty"]) for v in per_stage.values())
    out: dict[str, Any] = {
        "stages": {
            name: {
                "qty": int(values["qty"]),
                "open_amount": round(values["open_amount"], 2),
                "share_pct": round(values["qty"] * 100.0 / total, 2) if total else 0.0,
            }
            for name, values in per_stage.items()
        },
        "total_agreements": total,
        "criterion": (
            f"Approved agreements issued in the last {config.plan_lookback_days} days. "
            "Stage by payment progress: settled = everything paid; early = at most one "
            f"installment paid; late = {config.plan_late_remaining} or fewer installments "
            "left; middle = the rest."
        ),
        "reference_date": reference_date,
    }
    if stage:
        out["stage"] = stage
        out["portfolios"] = _top_portfolios(per_portfolio, stage, config.stage_portfolio_limit)
    return out


def _top_portfolios(
    per_portfolio: dict[str, dict[str, dict[str, float]]], stage: str, limit: int
) -> list[dict[str, Any]]:
    entries = []
    for name, cells in per_portfolio.items():
        total = sum(int(c["qty"]) for c in cells.values())
        cell = cells[stage]
        if cell["qty"] <= 0:
            continue
        entries.append(
            {
                "portfolio_name": name,
                "qty": int(cell["qty"]),
                "open_amount": round(cell["open_amount"], 2),
                "share_of_portfolio_agreements_pct": round(cell["qty"] * 100.0 / total, 2),
            }
        )
    entries.sort(key=lambda e: (-e["open_amount"], e["portfolio_name"]))
    return entries[:limit]
