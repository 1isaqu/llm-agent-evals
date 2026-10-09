"""Tools that answer questions about portfolios and agreement statuses."""

from __future__ import annotations

from typing import Any

from llm_agent_evals.analytics.portfolio import status_breakdown
from llm_agent_evals.analytics.stages import STAGES, stage_summary
from llm_agent_evals.context import ToolContext
from llm_agent_evals.registry import Tool, ToolRegistry
from llm_agent_evals.tools.common import (
    ambiguity_warning,
    find_portfolio,
    limit_property,
    not_found_error,
)

_COMPARE_METRICS = (
    "first_installment_value",
    "agreement_count",
    "composite_risk_pct",
    "exception_pct",
    "broken_pct",
    "rejected_pct",
)


def _get_portfolio_metrics(ctx: ToolContext, args: dict[str, Any]) -> Any:
    entries = ctx.entries()
    name = args["portfolio_name"]
    entry = find_portfolio(name, entries)
    if entry is None:
        return not_found_error(
            "portfolio", name, "available_portfolios", [e["portfolio_name"] for e in entries]
        )
    warning = ambiguity_warning(name, entry, entries)
    return {**entry, "ambiguity_warning": warning} if warning else entry


def _filter_by_risk(ctx: ToolContext, args: dict[str, Any]) -> Any:
    return [e for e in ctx.entries() if e["risk_level"] == args["level"]]


def _filter_by_value(ctx: ToolContext, args: dict[str, Any]) -> Any:
    matching = [e for e in ctx.entries() if e["first_installment_value"] >= args["min_value"]]
    return matching[: args["limit"]]


def _compare_portfolios(ctx: ToolContext, args: dict[str, Any]) -> Any:
    entries = ctx.entries()
    found: list[dict[str, Any]] = []
    missing: list[str] = []
    warnings: list[str] = []
    for raw in args["names"]:
        entry = find_portfolio(raw, entries)
        if entry is None:
            missing.append(raw)
            continue
        if entry not in found:
            found.append(entry)
        warning = ambiguity_warning(raw, entry, entries)
        if warning:
            warnings.append(warning)
    result: dict[str, Any] = {"portfolios": found}
    if args.get("metric"):
        result["metric"] = args["metric"]
    if warnings:
        result["ambiguity_warnings"] = warnings
    if missing:
        result["not_found"] = missing
        result["available_portfolios"] = [e["portfolio_name"] for e in entries]
    return result


def _status_breakdown(ctx: ToolContext, args: dict[str, Any]) -> Any:
    rows = ctx.source_for().status_rollup(ctx.session.window)
    return status_breakdown(rows, ctx.config, ctx.session.date_to)


def _plan_stage(ctx: ToolContext, args: dict[str, Any]) -> Any:
    rows = ctx.source_for().plan_stage_rollup(ctx.session.window.end)
    return stage_summary(rows, args.get("stage"), ctx.config, ctx.session.date_to)


def register(registry: ToolRegistry) -> None:
    registry.register(
        Tool(
            name="get_portfolio_metrics",
            description=(
                "Full metrics entry of one portfolio by name (case-insensitive exact match, "
                "then substring). Use it when the user asks about a specific portfolio. If the "
                "result has an ambiguity_warning, tell the user about the other matches."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "portfolio_name": {
                        "type": "string",
                        "minLength": 1,
                        "description": "Portfolio name or a fragment of it.",
                    }
                },
                "required": ["portfolio_name"],
                "additionalProperties": False,
            },
            handler=_get_portfolio_metrics,
        )
    )
    registry.register(
        Tool(
            name="filter_portfolios_by_risk",
            description="List the portfolios at a risk level (low, medium or high composite risk).",
            input_schema={
                "type": "object",
                "properties": {"level": {"type": "string", "enum": ["low", "medium", "high"]}},
                "required": ["level"],
                "additionalProperties": False,
            },
            handler=_filter_by_risk,
        )
    )
    registry.register(
        Tool(
            name="filter_portfolios_by_value",
            description=(
                "List portfolios whose first-installment value in the session window is at least "
                "min_value, largest first. It reflects the session window only."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "min_value": {"type": "number", "minimum": 0},
                    "limit": limit_property(10),
                },
                "required": ["min_value"],
                "additionalProperties": False,
            },
            handler=_filter_by_value,
        )
    )
    registry.register(
        Tool(
            name="compare_portfolios",
            description=(
                "Compare two or more portfolios side by side. Returns their full entries and the "
                "metric in focus when one is given."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "names": {
                        "type": "array",
                        "items": {"type": "string", "minLength": 1},
                        "minItems": 1,
                        "maxItems": 10,
                    },
                    "metric": {"type": "string", "enum": list(_COMPARE_METRICS)},
                },
                "required": ["names"],
                "additionalProperties": False,
            },
            handler=_compare_portfolios,
        )
    )
    registry.register(
        Tool(
            name="get_status_breakdown",
            description=(
                "Agreements of the session window by status (every status in the catalogue "
                "that counts), with quantity and first-installment value. The result states "
                "generated_amount separately: all_statuses_amount includes exception and "
                "rejected value and is never 'generated'."
            ),
            input_schema={"type": "object", "properties": {}, "additionalProperties": False},
            handler=_status_breakdown,
        )
    )
    registry.register(
        Tool(
            name="get_plan_stage_distribution",
            description=(
                "Approved agreements of the recent past grouped by payment-plan stage: early "
                "(few installments paid), middle, late (few installments left), settled "
                "(everything paid). With a stage, also lists the portfolios holding the most "
                "open amount in it. Independent of the session window."
            ),
            input_schema={
                "type": "object",
                "properties": {"stage": {"type": "string", "enum": list(STAGES)}},
                "additionalProperties": False,
            },
            handler=_plan_stage,
        )
    )
