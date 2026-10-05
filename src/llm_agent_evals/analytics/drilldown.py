"""Portfolio drill-down: rankings, maturities and row-level paging.

Row-level drilldowns return agreement ids, amounts, dates and the agent who owns the
agreement. They never carry any debtor attribute, because the data contract has none.
"""

from __future__ import annotations

from typing import Any

from llm_agent_evals.analytics.common import as_float, as_int, ratio_pct
from llm_agent_evals.analytics.portfolio import portfolio_entries
from llm_agent_evals.business import BusinessConfig
from llm_agent_evals.session import Window

Row = dict[str, Any]

ROW_DRILLDOWNS = ("generated", "exception", "rejected", "broken")


def _ranking_envelope(
    kind: str, product_line: str, window: Window, ranking: list[dict[str, Any]], limit: int
) -> dict[str, Any]:
    return {
        "drilldown": kind,
        "product_line": product_line,
        "date_from": window.start.isoformat(),
        "date_to": window.end.isoformat(),
        "total_portfolios_in_period": len(ranking),
        "portfolios_returned": min(len(ranking), limit),
        "ranking": ranking[:limit],
        "truncated": len(ranking) > limit,
    }


def generation_ranking(
    rows: list[Row],
    config: BusinessConfig,
    product_line: str,
    window: Window,
    limit: int,
) -> dict[str, Any]:
    """Every portfolio's generated value and agreement count in an arbitrary window."""
    entries = portfolio_entries(rows, config, window.end.isoformat())
    ranking = [
        {
            "portfolio": e["portfolio_name"],
            "agreement_count": e["agreement_count"],
            "generated_value": e["first_installment_value"],
        }
        for e in entries
    ]
    return _ranking_envelope("generation_ranking", product_line, window, ranking, limit)


def _maturity_fields(row: Row) -> dict[str, Any]:
    issued = as_int(row.get("generated"))
    paid = as_int(row.get("paid_on_time"))
    return {
        "slips_issued": issued,
        "slips_paid_within_grace": paid,
        "amount_maturing": round(as_float(row.get("amount_maturing")), 2),
        "amount_received": round(as_float(row.get("amount_received")), 2),
        "effectiveness_pct": ratio_pct(paid, issued),
    }


def maturities_summary(
    rows: list[Row], portfolio: str, product_line: str, window: Window
) -> dict[str, Any]:
    """Slips whose due date falls in the window, for one portfolio."""
    row = rows[0] if rows else {}
    return {
        "portfolio": portfolio,
        "drilldown": "maturities",
        "product_line": product_line,
        "date_from": window.start.isoformat(),
        "date_to": window.end.isoformat(),
        **_maturity_fields(row),
    }


def maturities_ranking(
    rows: list[Row], product_line: str, window: Window, limit: int
) -> dict[str, Any]:
    """Maturities of every portfolio, largest amount maturing first."""
    ranking = [
        {"portfolio": str(r.get("group_key")), **_maturity_fields(r)}
        for r in rows
        if r.get("group_key")
    ]
    ranking.sort(key=lambda e: (-e["amount_maturing"], e["portfolio"]))
    return _ranking_envelope("maturities_ranking", product_line, window, ranking, limit)


def page_rows(
    rows: list[Row], kind: str, portfolio: str, page: int, page_size: int
) -> dict[str, Any]:
    """One page of agreement rows with explicit paging metadata."""
    total = len(rows)
    start = (page - 1) * page_size
    total_pages = max(1, (total + page_size - 1) // page_size)
    data = [
        {
            "agreement_id": row.get("agreement_id"),
            "portfolio_id": row.get("portfolio_id"),
            "first_installment_value": as_float(row.get("first_installment_value")),
            "total_value": as_float(row.get("total_value")),
            "agent_name": str(row.get("agent_name") or "").strip(),
            "agent_id": str(row.get("agent_id") or "").strip(),
            "issued_on": str(row.get("issued_on") or "")[:10],
            "due_on": str(row.get("due_on") or "")[:10],
            "installments": as_int(row.get("installments")),
        }
        for row in rows[start : start + page_size]
    ]
    return {
        "portfolio": portfolio,
        "drilldown": kind,
        "data": data,
        "meta": {
            "page": page,
            "page_size": page_size,
            "total_rows": total,
            "total_pages": total_pages,
            "truncated": page < total_pages,
        },
    }
