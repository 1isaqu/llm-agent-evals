"""The response envelope every endpoint returns.

    {"meta": {generated_at, total_rows, sources, filters, run_id}, "data": [...], "errors": [...]}

There is no `success` flag and no top-level status field: a request failed when `errors`
is not empty (and the HTTP status says so too). Keeping one envelope means a client has one
code path for every response, including failures.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any


def build_envelope(
    data: Sequence[Any],
    sources: Sequence[str],
    *,
    now: datetime,
    filters: Mapping[str, Any] | None = None,
    run_id: str = "",
    errors: Sequence[Mapping[str, str]] = (),
) -> dict[str, Any]:
    return {
        "meta": {
            "generated_at": now.isoformat(timespec="seconds"),
            "total_rows": len(data),
            "sources": list(sources),
            "filters": dict(filters or {}),
            "run_id": run_id,
        },
        "data": list(data),
        "errors": [dict(error) for error in errors],
    }


def error_envelope(
    code: str,
    message: str,
    *,
    now: datetime,
    filters: Mapping[str, Any] | None = None,
    run_id: str = "",
) -> dict[str, Any]:
    """An envelope with no data and one error."""
    return build_envelope(
        [], [], now=now, filters=filters, run_id=run_id, errors=[{"code": code, "message": message}]
    )
