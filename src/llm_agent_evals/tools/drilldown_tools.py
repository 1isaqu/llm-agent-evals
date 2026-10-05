"""Portfolio drill-down tool and the metric explainer."""

from __future__ import annotations

from typing import Any

from llm_agent_evals.analytics.drilldown import (
    generation_ranking,
    maturities_ranking,
    maturities_summary,
    page_rows,
)
from llm_agent_evals.context import ToolContext
from llm_agent_evals.errors import build_tool_error
from llm_agent_evals.registry import Tool, ToolRegistry
from llm_agent_evals.tools.common import (
    ambiguity_warning,
    date_property,
    find_portfolio,
    line_property,
    not_found_error,
    window_errors,
    window_of,
)

DRILLDOWNS = (
    "summary",
    "generated",
    "exception",
    "rejected",
    "broken",
    "maturities",
    "generation_ranking",
)
_NEEDS_PORTFOLIO = {"summary", "generated", "exception", "rejected", "broken"}


def _drill_validator(args: dict[str, Any], ctx: ToolContext) -> list[str]:
    errors = window_errors(args, ctx)
    kind, portfolio = args["drilldown"], args.get("portfolio")
    if kind == "generation_ranking" and portfolio:
        errors.append(
            "drilldown='generation_ranking' only exists as a ranking of all portfolios: omit "
            "portfolio. For one portfolio use query_kpi_history with kpi='generated_value'."
        )
    if kind in _NEEDS_PORTFOLIO and not portfolio:
        errors.append("portfolio is required for this drilldown.")
    return errors


def _drill_down(ctx: ToolContext, args: dict[str, Any]) -> Any:
    line, window, kind = args["product_line"], window_of(args), args["drilldown"]
    source = ctx.source_for(line)
    if kind == "generation_ranking":
        return generation_ranking(
            source.status_rollup(window), ctx.config, line, window, args["page_size"]
        )
    if kind == "maturities" and not args.get("portfolio"):
        rows = source.slip_rollup(window, "portfolio", "due_at", None)
        return maturities_ranking(rows, line, window, args["page_size"])
    entries = ctx.entries(line, window)
    entry = find_portfolio(args["portfolio"], entries)
    if entry is None:
        return not_found_error(
            "portfolio",
            args["portfolio"],
            "available_portfolios",
            [e["portfolio_name"] for e in entries],
        )
    name = entry["portfolio_name"]
    if kind == "summary":
        result = dict(entry)
    elif kind == "maturities":
        rows = source.slip_rollup(window, "none", "due_at", name)
        result = maturities_summary(rows, name, line, window)
    else:
        rows = source.drilldown_rows(kind, name, window)
        result = page_rows(rows, kind, name, args["page"], args["page_size"])
    warning = ambiguity_warning(args["portfolio"], entry, entries)
    if warning:
        result["ambiguity_warning"] = warning
    return result


def _explain_metric(ctx: ToolContext, args: dict[str, Any]) -> Any:
    registry = ctx.metric_registry()
    resolved = registry.resolve(args["term"])
    if resolved is None:
        return build_tool_error(
            "validation",
            hint=f"Unknown term {args['term']!r}. Pick one of available_terms.",
            user_facing="I do not know that term.",
            available_terms=registry.available_terms(),
        )
    return {"term": args["term"], **resolved}


def register(registry: ToolRegistry, lines: tuple[str, ...]) -> None:
    registry.register(
        Tool(
            name="drill_down_portfolio",
            description=(
                "Drill down into portfolios of a date window. drilldown: 'summary' (aggregate "
                "metrics, like get_portfolio_metrics); 'generated', 'exception', 'rejected' and "
                "'broken' (paged agreement rows: ids, values, dates, owning agent; 'broken' is "
                "the strict broken status only); 'maturities' (slips whose due date falls in the "
                "window: how many were issued, how much is maturing, how much was received); "
                "'generation_ranking' (every portfolio ranked by generated value and agreement "
                "count for an arbitrary day or window). portfolio is required for 'summary' and "
                "the four row drilldowns. For 'maturities' omit portfolio to get the ranking of "
                "ALL portfolios in one call. For 'generation_ranking' portfolio is forbidden. "
                "Never loop portfolio by portfolio to build a ranking by hand. page_size (10, 25, "
                "50) also limits ranking length: check total_portfolios_in_period and truncated "
                "before claiming full coverage. The product line is per call."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "product_line": line_property(lines),
                    "portfolio": {"type": "string", "minLength": 1, "maxLength": 80},
                    "date_from": date_property("First day of the window (YYYY-MM-DD)."),
                    "date_to": date_property("Last day of the window (YYYY-MM-DD)."),
                    "drilldown": {"type": "string", "enum": list(DRILLDOWNS), "default": "summary"},
                    "page": {"type": "integer", "minimum": 1, "maximum": 10, "default": 1},
                    "page_size": {"type": "integer", "enum": [10, 25, 50], "default": 25},
                },
                "required": ["product_line", "date_from", "date_to"],
                "additionalProperties": False,
            },
            handler=_drill_down,
            validator=_drill_validator,
        )
    )
    registry.register(
        Tool(
            name="explain_metric",
            description=(
                "Official definition of a KPI, status group or business term (formula, filters, "
                "conventions) from the metric registry. ALWAYS call it before explaining any "
                "formula: never deduce one. An unknown term returns the list of valid terms."
            ),
            input_schema={
                "type": "object",
                "properties": {"term": {"type": "string", "minLength": 1}},
                "required": ["term"],
                "additionalProperties": False,
            },
            handler=_explain_metric,
        )
    )
