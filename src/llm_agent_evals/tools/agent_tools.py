"""Tools that answer questions about agents (collectors)."""

from __future__ import annotations

from typing import Any

from llm_agent_evals.analytics.agents import COMPARE_METRICS, ORDER_BY_FIELDS
from llm_agent_evals.analytics.crosstab import DIMENSIONS, agent_ranking, crosstab_summary
from llm_agent_evals.context import ToolContext
from llm_agent_evals.errors import build_tool_error
from llm_agent_evals.registry import Tool, ToolRegistry
from llm_agent_evals.tools.common import (
    ambiguity_warning,
    date_property,
    find_agent,
    find_agent_exact,
    find_portfolio,
    limit_property,
    line_property,
    matches_excluded,
    not_found_error,
    window_errors,
    window_of,
)


def _get_agent_performance(ctx: ToolContext, args: dict[str, Any]) -> Any:
    agents = ctx.agents()
    entry = find_agent(args["agent_name"], agents)
    if entry is None:
        return not_found_error(
            "agent", args["agent_name"], "available_agents", [a["agent_name"] for a in agents]
        )
    return entry


def _list_agents(ctx: ToolContext, args: dict[str, Any]) -> Any:
    ranked = sorted(ctx.agents(), key=lambda a: a.get(args["order_by"]) or 0, reverse=True)
    return ranked[: args["limit"]]


def _exactly_one_side(args: dict[str, Any], ctx: ToolContext) -> list[str]:
    if bool(args.get("portfolio")) == bool(args.get("agent_name")):
        return ["Provide exactly one of portfolio or agent_name."]
    return []


def _crosstab(ctx: ToolContext, args: dict[str, Any]) -> Any:
    window = ctx.session.window
    if args.get("portfolio"):
        entries = ctx.entries()
        entry = find_portfolio(args["portfolio"], entries)
        if entry is None:
            return not_found_error(
                "portfolio",
                args["portfolio"],
                "available_portfolios",
                [e["portfolio_name"] for e in entries],
            )
        rows = ctx.source_for().crosstab_rollup(window, "portfolio", entry["portfolio_name"])
        result: dict[str, Any] = {"portfolio": entry["portfolio_name"], "detail_by": "agent"}
        result["entries"] = crosstab_summary(rows, "agent", ctx.config, ctx.session.date_to)
        warning = ambiguity_warning(args["portfolio"], entry, entries)
        if warning:
            result["ambiguity_warning"] = warning
    else:
        agents = ctx.agents()
        agent = find_agent(args["agent_name"], agents)
        if agent is None:
            return not_found_error(
                "agent", args["agent_name"], "available_agents", [a["agent_name"] for a in agents]
            )
        rows = ctx.source_for().crosstab_rollup(window, "agent", agent["agent_id"])
        result = {
            "agent_name": agent["agent_name"],
            "agent_id": agent["agent_id"],
            "detail_by": "portfolio",
        }
        result["entries"] = crosstab_summary(rows, "portfolio", ctx.config, ctx.session.date_to)
    result["reference_date"] = ctx.session.date_to
    if not result["entries"]:
        result["note"] = "No agreements for this filter in the window."
    return result


def _ranking(ctx: ToolContext, args: dict[str, Any]) -> Any:
    rows = ctx.source_for().agent_status_rollup(ctx.session.window)
    return agent_ranking(rows, args["dimension"], ctx.config, args["limit"], ctx.session.date_to)


def _compare_validator(args: dict[str, Any], ctx: ToolContext) -> list[str]:
    errors = window_errors(args, ctx)
    keys = [k.strip().lower() for k in args["agent_ids"]]
    if len(set(keys)) != len(keys):
        errors.append("agent_ids has duplicates after normalisation.")
    return errors


def _compare_agents(ctx: ToolContext, args: dict[str, Any]) -> Any:
    line, window = args["product_line"], window_of(args)
    agents = ctx.agents(line, window)
    found: list[dict[str, Any]] = []
    system: list[str] = []
    missing: list[str] = []
    for raw in (k.strip().lower() for k in args["agent_ids"]):
        entry = find_agent_exact(raw, agents)
        if entry is not None:
            found.append(entry)
        elif matches_excluded(raw, ctx.config.excluded_agent_patterns):
            system.append(raw)
        else:
            missing.append(raw)
    if system or missing:
        problems = [f"agent {k!r} is a system account and cannot be compared" for k in system]
        problems += [f"agent {k!r} was not found in the period" for k in missing]
        return build_tool_error(
            "validation",
            hint="; ".join(problems),
            user_facing="One or more agents cannot be compared.",
            errors=problems,
            available_agents=[a["agent_name"] for a in agents],
        )
    result: dict[str, Any] = {
        "agents": [
            {
                "agent_name": e["agent_name"],
                "agent_id": e["agent_id"],
                **{m: e.get(m) for m in args["metrics"]},
            }
            for e in found
        ]
    }
    if args["consolidate_cross_line"]:
        result["meta"] = {
            "warnings": [
                "consolidate_cross_line was requested, but each product line has separate agent "
                "entities per call. Values were NOT summed across lines."
            ]
        }
    return result


def register(registry: ToolRegistry, lines: tuple[str, ...]) -> None:
    registry.register(
        Tool(
            name="get_agent_performance",
            description=(
                "Full entry of one agent by name or agent_id (case-insensitive exact match, then "
                "substring): funnel, agreements, values and conversion in the session window. "
                "Use it when the user asks about a specific agent."
            ),
            input_schema={
                "type": "object",
                "properties": {"agent_name": {"type": "string", "minLength": 1}},
                "required": ["agent_name"],
                "additionalProperties": False,
            },
            handler=_get_agent_performance,
        )
    )
    registry.register(
        Tool(
            name="list_agents_performance",
            description=(
                "Agents ranked by a metric (largest first) in the session window. Use it for "
                "'best agents', 'who converted most', 'team ranking'. conversion_pct is the "
                "official conversion; paid_per_cpc_pct is a different metric."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "order_by": {
                        "type": "string",
                        "enum": list(ORDER_BY_FIELDS),
                        "default": "agreement_value",
                    },
                    "limit": limit_property(10),
                },
                "additionalProperties": False,
            },
            handler=_list_agents,
        )
    )
    registry.register(
        Tool(
            name="get_agent_portfolio_crosstab",
            description=(
                "Cross agents and portfolios in the session window. Provide EXACTLY ONE side: "
                "with portfolio, the portfolio is split by agent; with agent_name, the agent is "
                "split by portfolio. Each row has agreement count, first-installment value and "
                "exception, broken and rejected value. Use it for 'who generates the exceptions "
                "of portfolio X' or 'which portfolios does agent Y work'."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "portfolio": {"type": "string", "minLength": 1},
                    "agent_name": {"type": "string", "minLength": 1},
                },
                "additionalProperties": False,
            },
            handler=_crosstab,
            validator=_exactly_one_side,
        )
    )
    registry.register(
        Tool(
            name="get_agent_ranking_by_dimension",
            description=(
                "Agents ranked by first-installment value in one status dimension of the session "
                "window: generated, exceptions, broken or rejected. Use it for 'who breaks the "
                "most agreements' or 'who has the most rejections'."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "dimension": {"type": "string", "enum": list(DIMENSIONS)},
                    "limit": limit_property(10),
                },
                "required": ["dimension"],
                "additionalProperties": False,
            },
            handler=_ranking,
        )
    )
    registry.register(
        Tool(
            name="compare_agents",
            description=(
                "Compare 2 to 5 agents side by side on the requested metrics, identified by "
                "agent_id. Use it when the user names agents explicitly. Not for a general "
                "ranking (use list_agents_performance). System accounts cannot be compared. "
                "The product line is per call and may differ from the session."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "product_line": line_property(lines),
                    "agent_ids": {
                        "type": "array",
                        "items": {"type": "string", "minLength": 1},
                        "minItems": 2,
                        "maxItems": 5,
                    },
                    "metrics": {
                        "type": "array",
                        "items": {"type": "string", "enum": list(COMPARE_METRICS)},
                        "minItems": 1,
                    },
                    "date_from": date_property("First day of the window (YYYY-MM-DD)."),
                    "date_to": date_property("Last day of the window (YYYY-MM-DD)."),
                    "consolidate_cross_line": {"type": "boolean", "default": False},
                },
                "required": ["product_line", "agent_ids", "metrics", "date_from", "date_to"],
                "additionalProperties": False,
            },
            handler=_compare_agents,
            validator=_compare_validator,
        )
    )
