"""Exception types, the tool error taxonomy and the final-answer sanitiser.

Every tool failure reaches the model as one structured payload built by
`build_tool_error`, never as a traceback and never as an ad-hoc dict. The model
gets a machine-readable `error_type`, a `retryable` flag, a `hint` that tells it
what to do next and a `user_facing` sentence it may repeat to the user.
"""

from __future__ import annotations

import re
from typing import Any


class AgentError(Exception):
    """Base class for errors raised by this package."""


class ConfigError(AgentError):
    """Invalid or missing configuration. The message names the variable, never its value."""


class ModelError(AgentError):
    """The model provider failed (network, HTTP status, malformed response)."""


class DataSourceError(AgentError):
    """A data source failed in a way that may be transient (timeout, dropped connection).

    The run guard retries a tool once when it raises this and reports `upstream_5xx`
    if the retry fails too. Any other exception type is treated as a code bug.
    """


class DataUnavailable(AgentError):
    """A data source cannot serve the requested data (not configured, not provided)."""


# error_type -> default `retryable` flag shown to the model.
ERROR_RETRYABLE_DEFAULT: dict[str, bool] = {
    "validation": True,
    "upstream_timeout": True,
    "upstream_5xx": True,
    "upstream_4xx": False,
    "empty_result": False,
    "tool_disabled": False,
    "step_budget_exceeded": False,
    "wall_clock_exceeded": False,
    "unknown_tool": False,
    "internal_error": False,
    "not_found": False,
    "data_unavailable": False,
    "egress_blocked": False,
}


def build_tool_error(
    error_type: str,
    hint: str,
    user_facing: str,
    retryable: bool | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """Return the single error envelope every tool and the dispatcher use."""
    payload: dict[str, Any] = {
        "ok": False,
        "error_type": error_type,
        "retryable": ERROR_RETRYABLE_DEFAULT.get(error_type, False)
        if retryable is None
        else retryable,
        "hint": hint,
        "user_facing": user_facing,
    }
    payload.update(extra)
    return payload


_SQL_IN_CODE_BLOCK = re.compile(
    r"```[a-zA-Z]*\n[^`]*\b(SELECT|DROP|INSERT|DELETE|UPDATE)\b[^`]*```",
    re.IGNORECASE | re.DOTALL,
)
_TRACEBACK = re.compile(r"Traceback \(most recent call last\)", re.IGNORECASE)
_RAW_ENVELOPE_DUMP = re.compile(r'"(meta|data|errors)"\s*:\s*[\[{]')


def sanitize_final_text(text: str) -> str | None:
    """Return `text` when it is safe to show, or None when it must be refused.

    Refused: SQL inside a fenced code block, a raw dump of a tool envelope and a
    stack trace. The caller degrades to an honest low-confidence answer and never
    echoes the refused content.
    """
    if _SQL_IN_CODE_BLOCK.search(text) or _TRACEBACK.search(text):
        return None
    if _RAW_ENVELOPE_DUMP.search(text):
        return None
    return text
