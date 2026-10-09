"""System prompt rendering.

The prompt template (`system_prompt.md`) holds the rules and the tone. Everything that a
rule quotes from the configuration (status sets, thresholds, window limit, grace days,
product lines) is a `{{PLACEHOLDER}}` filled from `BusinessConfig`, and the tool list is
generated from the registry. The prose therefore cannot say one thing while the code does
another. `render_system_prompt` refuses to return text with an unresolved placeholder.

Relative periods are the classic way for a model to be quietly wrong ("last quarter" computed
from the wrong year). The session context appended to the prompt lists them as explicit
dates, computed once from the injected clock.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from datetime import date, datetime, timedelta
from importlib import resources

from llm_agent_evals.business import BusinessConfig
from llm_agent_evals.errors import ConfigError
from llm_agent_evals.registry import ToolRegistry
from llm_agent_evals.session import ALL_LINES, Session

_PLACEHOLDER = re.compile(r"\{\{[A-Z_]+\}\}")

FORCE_FINAL_NUDGE = (
    "You reached the step or time limit for this query, or repeated a tool call. Do not "
    "call any more tools. Answer now, in the required JSON format, with the data you "
    "already collected. If something is incomplete, say so explicitly in the text field."
)

COMPACTION_NOTICE = (
    "[Automatic summary: context budget] Earlier tool calls already made in this "
    "conversation (their results were removed to fit the budget): {calls}. Call one of "
    "them again only if you still need its figures; an identical second call returns the "
    "earlier result."
)


def load_template() -> str:
    """The prompt template shipped with the package."""
    return resources.files("llm_agent_evals").joinpath("system_prompt.md").read_text("utf-8")


def _quarter_start(day: date) -> date:
    return date(day.year, ((day.month - 1) // 3) * 3 + 1, 1)


def date_anchors(today: date) -> dict[str, tuple[date, date]]:
    """Explicit `(start, end)` dates for the relative periods a user can ask about.

    Open periods (this week, this month, this quarter) end today, never at the end of the
    period, because the end may lie in the future. Weeks start on Monday.
    """
    month_start = today.replace(day=1)
    last_month_end = month_start - timedelta(days=1)
    quarter_start = _quarter_start(today)
    last_quarter_end = quarter_start - timedelta(days=1)
    return {
        "today": (today, today),
        "yesterday": (today - timedelta(days=1), today - timedelta(days=1)),
        "this week": (today - timedelta(days=today.weekday()), today),
        "this month": (month_start, today),
        "last month": (last_month_end.replace(day=1), last_month_end),
        "this quarter": (quarter_start, today),
        "last quarter": (_quarter_start(last_quarter_end), last_quarter_end),
    }


def _status_text(config: BusinessConfig, labels: Sequence[str]) -> str:
    return ", ".join(f"{label} (id {config.status_ids[label]})" for label in labels)


def tool_summary(registry: ToolRegistry) -> str:
    """One bullet per tool: its name and the first sentence of its description."""
    lines = []
    for spec in registry.specs():
        first = re.split(r"(?<=[.!?])\s", spec["description"].strip(), maxsplit=1)[0]
        lines.append(f"- `{spec['name']}`: {first}")
    return "\n".join(lines)


def render_system_prompt(
    config: BusinessConfig,
    registry: ToolRegistry,
    lines: Sequence[str],
    reference_date: str,
) -> str:
    """Fill the template. Raises `ConfigError` if a placeholder is left over."""
    values = {
        "REFERENCE_DATE": reference_date,
        "EXCEPTION_STATUSES": _status_text(config, config.exception),
        "BROKEN_STATUSES": _status_text(config, config.broken),
        "REJECTED_STATUSES": _status_text(config, config.rejected),
        "GENERATED_STATUSES": _status_text(config, config.generated_labels),
        "RISK_LOW_MAX": f"{config.risk_low_max:g}",
        "RISK_MEDIUM_MAX": f"{config.risk_medium_max:g}",
        "MAX_WINDOW_DAYS": str(config.max_window_days),
        "GRACE_DAYS": str(config.grace_days),
        "LINES": ", ".join(lines),
        "TOOL_SUMMARY": tool_summary(registry),
    }
    text = load_template()
    for key, value in values.items():
        text = text.replace("{{" + key + "}}", value)
    leftover = _PLACEHOLDER.search(text)
    if leftover:
        raise ConfigError(f"system prompt has an unresolved placeholder {leftover.group()}")
    return text


def session_context(
    session: Session,
    now: datetime,
    portfolio_count: int,
    lines: Sequence[str],
) -> str:
    """The per-request section appended after the template."""
    scope = (
        f"all product lines ({', '.join(lines)})"
        if session.product_line == ALL_LINES
        else f"product line {session.product_line}"
    )
    out = [
        "## Session context",
        f"- Scope of the session: {scope}.",
        f"- Window selected by the user (a display filter, not a limit on the tools): "
        f"{session.date_from} to {session.date_to}.",
        f"- Portfolios loaded in that window: {portfolio_count}.",
        f"- Current time of the system: {now.strftime('%Y-%m-%d %H:%M')}.",
        "",
        "Relative periods, as explicit dates (use these for `date_from` and `date_to`):",
    ]
    for name, (start, end) in date_anchors(now.date()).items():
        span = start.isoformat() if start == end else f"{start.isoformat()} to {end.isoformat()}"
        out.append(f"- {name}: {span}")
    return "\n".join(out)


def build_system_prompt(
    config: BusinessConfig,
    registry: ToolRegistry,
    lines: Sequence[str],
    session: Session,
    now: datetime,
    portfolio_count: int,
) -> str:
    """Template plus session context: the full system prompt for one request."""
    template = render_system_prompt(config, registry, lines, session.date_to)
    return template + "\n\n" + session_context(session, now, portfolio_count, lines) + "\n"
