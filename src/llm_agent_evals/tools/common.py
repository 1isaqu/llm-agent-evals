"""Helpers shared by the tool modules: name search, window rules and error shapes."""

from __future__ import annotations

import re
from datetime import date
from typing import Any

from llm_agent_evals.context import ToolContext
from llm_agent_evals.errors import build_tool_error
from llm_agent_evals.session import Window

Entries = list[dict[str, Any]]


def date_property(description: str) -> dict[str, Any]:
    return {"type": "string", "format": "date", "description": description}


def line_property(lines: tuple[str, ...]) -> dict[str, Any]:
    return {
        "type": "string",
        "enum": list(lines),
        "description": "Product line to query. It is per call and may differ from the session.",
    }


def limit_property(default: int = 10) -> dict[str, Any]:
    return {
        "type": "integer",
        "minimum": 1,
        "maximum": 50,
        "default": default,
        "description": f"Maximum rows to return (default {default}, at most 50).",
    }


def window_of(args: dict[str, Any]) -> Window:
    return Window(date.fromisoformat(args["date_from"]), date.fromisoformat(args["date_to"]))


def window_errors(args: dict[str, Any], ctx: ToolContext) -> list[str]:
    """Cross-field rules for tools that take date_from and date_to."""
    start, end = date.fromisoformat(args["date_from"]), date.fromisoformat(args["date_to"])
    if end < start:
        return ["date_to must not be before date_from."]
    if end > ctx.today:
        return ["date_to must not be in the future."]
    days = (end - start).days + 1
    limit = ctx.config.max_window_days
    if days > limit:
        return [
            f"The window spans {days} days, more than the {limit} allowed per call. "
            "Shorten the window, or make several calls with consecutive windows."
        ]
    return []


def not_found_error(
    kind: str, name: str, available_key: str, available: list[str]
) -> dict[str, Any]:
    """The single shape for 'that entity does not exist in this period'."""
    return build_tool_error(
        "not_found",
        hint=f"No {kind} matches {name!r} in this period. Do not retry the same name. "
        "Offer the available ones instead.",
        user_facing=f"I could not find {kind} {name!r} in this period.",
        **{available_key: available},
    )


def find_portfolio(name: str, entries: Entries) -> dict[str, Any] | None:
    """Exact match ignoring case, else the first entry whose name contains the text."""
    wanted = (name or "").strip().lower()
    if not wanted:
        return None
    for entry in entries:
        if entry["portfolio_name"].lower() == wanted:
            return entry
    for entry in entries:
        if wanted in entry["portfolio_name"].lower():
            return entry
    return None


def ambiguity_warning(name: str, resolved: dict[str, Any], entries: Entries) -> str | None:
    """A warning when a fragment matched more than one portfolio, else None.

    A substring search silently returns the first match. Without this warning the model
    would present one portfolio as if it were the only candidate and hide the others.
    """
    wanted = (name or "").strip().lower()
    if not wanted or resolved["portfolio_name"].strip().lower() == wanted:
        return None
    others = [
        e["portfolio_name"]
        for e in entries
        if wanted in e["portfolio_name"].lower()
        and e["portfolio_name"] != resolved["portfolio_name"]
    ]
    if not others:
        return None
    return (
        f"The fragment {name!r} also matches: {', '.join(others)}. Returning "
        f"{resolved['portfolio_name']!r} (first match). Tell the user about the other "
        "matches and ask which one they meant."
    )


def find_agent_exact(name: str, agents: Entries) -> dict[str, Any] | None:
    """Exact match on name or id, ignoring case. No fragments: used where a wrong guess
    would silently compare the wrong people."""
    wanted = (name or "").strip().lower()
    for entry in agents:
        if wanted and wanted in (entry["agent_name"].lower(), entry["agent_id"].lower()):
            return entry
    return None


def find_agent(name: str, agents: Entries) -> dict[str, Any] | None:
    """Exact match on name or id ignoring case, else the first substring match."""
    wanted = (name or "").strip().lower()
    if not wanted:
        return None
    exact = find_agent_exact(wanted, agents)
    if exact is not None:
        return exact
    for entry in agents:
        if wanted in entry["agent_name"].lower() or wanted in entry["agent_id"].lower():
            return entry
    return None


def matches_excluded(login: str, patterns: tuple[str, ...]) -> bool:
    """True when `login` matches one of the SQL-style (`%`) exclusion patterns."""
    for pattern in patterns:
        regex = ".*".join(re.escape(part) for part in pattern.lower().split("%"))
        if re.fullmatch(regex, login.strip().lower()):
            return True
    return False
