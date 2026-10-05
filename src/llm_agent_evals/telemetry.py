"""Safe telemetry: what may be logged about a tool call, and a JSON-lines observer.

The agent reports events to an observer, a plain callable `(event, payload) -> None`. A
payload never carries tool arguments, tool results, user text or model text. For a tool
call it holds the tool name, a hash of the canonical arguments, timing and the guard's
outcome. That is enough to debug a slow or failing tool without putting business data or
personal data into a log.

An observer must never break a request, so `JsonlTelemetry` swallows I/O errors, and the
agent shields itself from observers that raise.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

Observer = Callable[[str, dict[str, Any]], None]


@dataclass(frozen=True)
class ToolCallRecord:
    """Everything that is safe to say about one tool call."""

    tool_name: str
    args_hash: str
    step_index: int | None
    latency_ms: float
    cache_hit: bool
    truncated: bool
    row_count: int | None
    error_type: str | None
    egress_findings: int = 0

    def as_payload(self) -> dict[str, Any]:
        return asdict(self)


class JsonlTelemetry:
    """Append each event as one JSON line. Errors while writing are ignored."""

    def __init__(self, path: str | Path, run_id: str = "") -> None:
        self._path = Path(path)
        self._run_id = run_id

    def __call__(self, event: str, payload: dict[str, Any]) -> None:
        line = {
            "ts": datetime.now(UTC).isoformat(timespec="seconds"),
            "event": event,
            "run_id": payload.get("run_id") or self._run_id,
            **payload,
        }
        try:
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(line, ensure_ascii=False, default=str) + "\n")
        except OSError:
            return


def build_observer(path: str) -> Observer | None:
    """A `JsonlTelemetry` for `AGENT_TELEMETRY_PATH`, or None when the path is empty."""
    return JsonlTelemetry(path) if path else None
