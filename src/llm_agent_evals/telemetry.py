"""Safe telemetry: what may be logged about a tool call, and observers that record it.

The agent reports events to an observer, a plain callable `(event, payload) -> None`. A
payload never carries tool arguments, tool results, user text or model text. For a tool
call it holds the tool name, a hash of the canonical arguments, timing and the guard's
outcome. That is enough to debug a slow or failing tool without putting business data or
personal data into a log.

An observer must never break a request, so `JsonlTelemetry` swallows I/O errors, and the
agent shields itself from observers that raise.

Two observers ship. `JsonlTelemetry` has no dependencies. `LangfuseTelemetry` forwards the same
events to Langfuse and needs the optional `langfuse` extra, imported only when it is switched on.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from llm_agent_evals.errors import ConfigError

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


class LangfuseTelemetry:
    """Forward the safe events to Langfuse, one trace per `run_id`.

    A tool call becomes a `tool` observation named after the tool, every other event a point
    event. Only the scalar fields of a payload are sent, so an event cannot carry arguments or
    data even if a payload one day grows a field that does. `client` is a `langfuse.Langfuse`,
    or anything with `create_trace_id`, `start_observation` and `create_event`.
    """

    def __init__(self, client: Any, run_id: str = "") -> None:
        self._client = client
        self._run_id = run_id

    def __call__(self, event: str, payload: dict[str, Any]) -> None:
        run_id = payload.get("run_id") or self._run_id or "no-run-id"
        error = payload.get("error_type")
        fields = {
            "trace_context": {"trace_id": self._client.create_trace_id(seed=run_id)},
            "metadata": {
                key: value
                for key, value in payload.items()
                if key != "run_id" and isinstance(value, str | int | float | bool | None)
            },
            "level": "ERROR" if error else None,
            "status_message": error,
        }
        if event == "tool_call":
            name = payload.get("tool_name") or event
            self._client.start_observation(name=name, as_type="tool", **fields).end()
        else:
            self._client.create_event(name=event, **fields)


def build_langfuse_observer(env: Mapping[str, str]) -> Observer:
    """A `LangfuseTelemetry` over the official client, configured from the `LANGFUSE_*` variables.

    The keys are handed to the client and never logged. Raises `ConfigError` when a key is
    missing or the `langfuse` extra is not installed, so a typo is loud at start-up.
    """
    public_key, secret_key = env.get("LANGFUSE_PUBLIC_KEY", ""), env.get("LANGFUSE_SECRET_KEY", "")
    if not (public_key and secret_key):
        raise ConfigError(
            "AGENT_TELEMETRY_LANGFUSE needs LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY"
        )
    try:
        from langfuse import Langfuse
    except ImportError as exc:
        raise ConfigError(
            "AGENT_TELEMETRY_LANGFUSE needs the langfuse package: "
            'pip install "llm-agent-evals[langfuse]"'
        ) from exc
    base_url = env.get("LANGFUSE_BASE_URL") or env.get("LANGFUSE_HOST") or None
    return LangfuseTelemetry(
        Langfuse(public_key=public_key, secret_key=secret_key, base_url=base_url)
    )


def combine(*observers: Observer | None) -> Observer | None:
    """One observer that calls each of those given. A failing sink cannot starve the others."""
    active = [observer for observer in observers if observer is not None]
    if len(active) < 2:
        return active[0] if active else None

    def fan_out(event: str, payload: dict[str, Any]) -> None:
        for observer in active:
            try:
                observer(event, payload)
            except Exception:  # noqa: BLE001 - one sink failing must not silence another
                continue

    return fan_out
