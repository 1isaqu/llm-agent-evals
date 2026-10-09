"""Tools about time: KPI history and the daily pace."""

from __future__ import annotations

from typing import Any

from llm_agent_evals.analytics.effectiveness import (
    CRITERION,
    effectiveness_points,
    effectiveness_total,
)
from llm_agent_evals.analytics.series import daily_points, paginate, rebucket, trend
from llm_agent_evals.context import ToolContext
from llm_agent_evals.registry import Tool, ToolRegistry
from llm_agent_evals.tools.common import (
    date_property,
    find_portfolio,
    line_property,
    not_found_error,
    window_errors,
    window_of,
)

_ROLLUP_KPIS = {
    "generated_value": "value",
    "agreement_count": "count",
    "composite_risk_pct": "risk",
}
KPIS = (*_ROLLUP_KPIS, "slip_effectiveness", "daily_pace")


def _resolve_portfolio(
    ctx: ToolContext, args: dict[str, Any]
) -> tuple[str | None, dict[str, Any] | None]:
    """Return (resolved name or None, error payload or None)."""
    if not args.get("portfolio"):
        return None, None
    entries = ctx.entries(args["product_line"], window_of(args))
    entry = find_portfolio(args["portfolio"], entries)
    if entry is None:
        error = not_found_error(
            "portfolio",
            args["portfolio"],
            "available_portfolios",
            [e["portfolio_name"] for e in entries],
        )
        return None, error
    return entry["portfolio_name"], None


def _rollup_series(ctx: ToolContext, args: dict[str, Any]) -> Any:
    kpi, window = args["kpi"], window_of(args)
    portfolio, error = _resolve_portfolio(ctx, args)
    if error is not None:
        return error
    rows = ctx.source_for(args["product_line"]).daily_rollup(window, portfolio)
    points = rebucket(
        daily_points(rows, ctx.config, _ROLLUP_KPIS[kpi], window), kpi, args["granularity"]
    )
    direction, change = trend(points, ctx.config.trend_flat_band_pct) if points else ("flat", None)
    result: dict[str, Any] = {
        "kpi": kpi,
        "granularity": args["granularity"],
        **paginate(points, args["page"]),
        "trend": direction,
        "change_pct": change,
    }
    if portfolio:
        result["portfolio"] = portfolio
    return result


def _effectiveness_series(ctx: ToolContext, args: dict[str, Any]) -> Any:
    window = window_of(args)
    portfolio, error = _resolve_portfolio(ctx, args)
    if error is not None:
        return error
    rows = ctx.source_for(args["product_line"]).slip_rollup(window, "day", "issued_at", portfolio)
    points = effectiveness_points(rows, args["granularity"])
    result: dict[str, Any] = {
        "kpi": "slip_effectiveness",
        "granularity": args["granularity"],
        **paginate(points, args["page"]),
        "total": effectiveness_total(points),
        "criterion": CRITERION,
    }
    if portfolio:
        result["portfolio"] = portfolio
    return result


def _pace_series(ctx: ToolContext, args: dict[str, Any]) -> Any:
    forecast = ctx.pace.forecast(ctx.source_for(args["product_line"]), ctx.now, ctx.config)
    if forecast.get("ok") is False:
        return forecast
    warning = (
        "kpi='daily_pace' always reflects TODAY. date_from, date_to, granularity and portfolio "
        "are ignored."
    )
    return {
        **forecast,
        "meta": {"page": 1, "total_pages": 1, "truncated": False, "warnings": [warning]},
    }


def _query_kpi_history(ctx: ToolContext, args: dict[str, Any]) -> Any:
    if args["kpi"] in _ROLLUP_KPIS:
        return _rollup_series(ctx, args)
    if args["kpi"] == "slip_effectiveness":
        return _effectiveness_series(ctx, args)
    return _pace_series(ctx, args)


def _get_daily_pace(ctx: ToolContext, args: dict[str, Any]) -> Any:
    return ctx.pace.forecast(ctx.source_for(), ctx.now, ctx.config)


def register(registry: ToolRegistry, lines: tuple[str, ...]) -> None:
    registry.register(
        Tool(
            name="query_kpi_history",
            description=(
                "History of one KPI over a date window, by day, week or month. kpi: "
                "'generated_value' (first-installment value per day), 'agreement_count' "
                "(agreements per day), 'composite_risk_pct' (worst risk axis of the day), "
                "'slip_effectiveness' (slips paid within grace / slips issued, by issue date) or "
                "'daily_pace' (always TODAY, ignores dates). portfolio (optional) narrows "
                "generated_value, agreement_count, composite_risk_pct and slip_effectiveness to "
                "one portfolio. The product line is per call. Use it for trend, evolution, "
                "history and 'versus last week'. slip_effectiveness is NOT conversion: never "
                "label it conversion. Conversion (agreements / right-party contacts) has no daily "
                "series and no per-line or per-portfolio total: it exists only per agent, via "
                "get_agent_performance, list_agents_performance or compare_agents. Say so when "
                "asked for conversion at another grain, and never substitute silently."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "product_line": line_property(lines),
                    "kpi": {"type": "string", "enum": list(KPIS)},
                    "date_from": date_property("First day of the window (YYYY-MM-DD)."),
                    "date_to": date_property("Last day of the window (YYYY-MM-DD)."),
                    "granularity": {
                        "type": "string",
                        "enum": ["day", "week", "month"],
                        "default": "day",
                    },
                    "page": {"type": "integer", "minimum": 1, "maximum": 20, "default": 1},
                    "portfolio": {"type": "string", "minLength": 1, "maxLength": 80},
                },
                "required": ["product_line", "kpi", "date_from", "date_to"],
                "additionalProperties": False,
            },
            handler=_query_kpi_history,
            validator=window_errors,
        )
    )
    registry.register(
        Tool(
            name="get_daily_pace",
            description=(
                "Pace of TODAY (always the current day, whatever the session window is): expected "
                "agreements by hour band against the actual count, the actual total so far, the "
                "expected total and a projection of the close. Use it for 'how is the day going', "
                "'are we on target today', 'projected close'."
            ),
            input_schema={"type": "object", "properties": {}, "additionalProperties": False},
            handler=_get_daily_pace,
        )
    )
