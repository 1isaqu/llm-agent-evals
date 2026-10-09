"""Run guard: the runtime protocol around the tool loop.

One `RunState` lives for one agent request. It enforces:

* a ceiling on tool calls (`max_steps`) and on wall-clock time, checked inside
  `dispatch` itself and not only between rounds, because a model can return a batch
  of tool calls in a single turn and the loop only re-checks between turns;
* memoisation of identical calls (same tool, same canonical arguments), with the
  third identical call declaring a loop;
* spiral detection (A, B, A, B) over the last four tool names;
* per-tool disabling after three consecutive failures;
* one retry with backoff for transient data-source failures;
* a token cap on every tool result and a request-level budget that tells the loop
  when to compact the conversation.

Two objects outlive a request and live in `SharedState`: a result cache (short TTL
for errors, longer for successes) and a circuit breaker. Both take an injected clock
and are owned by the caller instead of being module globals, so tests never need to
reset global state or patch the time module.
"""

from __future__ import annotations

import hashlib
import json
import random
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from llm_agent_evals.errors import DataSourceError, DataUnavailable, build_tool_error


@dataclass(frozen=True)
class RunGuard:
    """Per-request limits. The numbers are engineering defaults, tune them freely."""

    max_steps: int = 8
    max_identical_calls: int = 2
    wall_clock_seconds: float = 60.0
    tool_result_token_cap: int = 1500
    tool_budget_tokens: int = 12000
    retry_base_delay_seconds: float = 0.5
    retry_jitter_seconds: float = 0.2


# JSON with digits and punctuation tokenises nearer 3 characters per token than English
# prose does. Using 4 under-estimates the real size of a result.
CHARS_PER_TOKEN_ESTIMATE = 3


def canonical_args(args: dict[str, Any]) -> str:
    return json.dumps(args, sort_keys=True, ensure_ascii=False, default=str)


def call_hash(tool_name: str, args: dict[str, Any]) -> str:
    """Memoisation key inside one request."""
    return hashlib.sha1(f"{tool_name}|{canonical_args(args)}".encode()).hexdigest()


def cache_key(
    tool_name: str, product_line: str, date_from: str, date_to: str, args: dict[str, Any]
) -> str:
    """Cross-request cache key.

    The session window is part of the key because some tools take their window from
    the session instead of from their arguments. Without it, a session looking at one
    month could be served the cached result of another session looking at a different
    month within the TTL.
    """
    raw = f"{tool_name}|{product_line}|{date_from}|{date_to}|{canonical_args(args)}"
    return hashlib.sha1(raw.encode()).hexdigest()


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN_ESTIMATE)


def cap_tool_result(result: Any, cap_tokens: int) -> tuple[Any, bool]:
    """Cut a tool result down to `cap_tokens`.

    The result is never cut in the middle of a structure: it is serialised whole, and
    when it does not fit the model receives a small warning envelope with a text
    preview instead of truncated JSON that would fail to parse on its side.
    """
    serialized = json.dumps(result, ensure_ascii=False, default=str)
    if estimate_tokens(serialized) <= cap_tokens:
        return result, False
    return {
        "_truncated": True,
        "_warning": (
            f"Result too large (about {estimate_tokens(serialized)} tokens) and cut to fit "
            "the budget. Ask for a narrower window or a smaller page."
        ),
        "_preview": serialized[: cap_tokens * CHARS_PER_TOKEN_ESTIMATE],
    }, True


_ROW_KEYS = ("data", "entries", "agents", "portfolios", "ranking", "statuses")


def infer_row_count(result: Any) -> int | None:
    """Best-effort row count for telemetry.

    A list counts its items. A paged result reports its real total. A result that holds a
    list under a well-known key (`data`, `entries`, ...) counts that list, so an empty
    crosstab is 0 and not 1. Any other non-error dict is one entity entry, so it counts
    as 1. An error payload has no rows (None).
    """
    if isinstance(result, list):
        return len(result)
    if not isinstance(result, dict) or result.get("ok") is False:
        return None
    meta = result.get("meta")
    if isinstance(meta, dict):
        for key in ("total_rows", "total_points"):
            if key in meta:
                return meta[key]
    for key in _ROW_KEYS:
        if isinstance(result.get(key), list):
            return len(result[key])
    return 1


def tool_error_type(result: Any) -> str | None:
    """The `error_type` of a tool result that is an error payload, else None.

    Tools report model-caused problems (bad arguments, unknown names) as normal return
    values shaped by `build_tool_error`. Telemetry reads them here so a `validation` or
    `not_found` result is not recorded as a success.
    """
    if isinstance(result, dict) and result.get("ok") is False:
        error_type = result.get("error_type")
        return str(error_type) if error_type else None
    return None


class ToolCache:
    """Result cache shared across requests. Errors expire quickly so a broken
    source is not hammered, but a transient failure is not remembered for long."""

    def __init__(
        self,
        ttl_success_seconds: float = 45.0,
        ttl_error_seconds: float = 6.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.ttl_success_seconds = ttl_success_seconds
        self.ttl_error_seconds = ttl_error_seconds
        self._clock = clock
        self._store: dict[str, tuple[float, Any]] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> Any | None:
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            expires_at, value = entry
            if expires_at < self._clock():
                self._store.pop(key, None)
                return None
            return value

    def set(self, key: str, value: Any, *, error: bool = False) -> None:
        ttl = self.ttl_error_seconds if error else self.ttl_success_seconds
        if ttl <= 0:
            return
        with self._lock:
            self._store[key] = (self._clock() + ttl, value)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()


class CircuitBreaker:
    """Per-tool breaker. Opens when the error rate over the last `window` calls
    exceeds `error_rate_threshold`, after at least `min_sample` calls."""

    def __init__(
        self,
        window: int = 12,
        min_sample: int = 4,
        error_rate_threshold: float = 0.40,
        open_seconds: float = 45.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.window = window
        self.min_sample = min_sample
        self.error_rate_threshold = error_rate_threshold
        self.open_seconds = open_seconds
        self._clock = clock
        self._history: dict[str, deque[bool]] = {}
        self._opened_until: dict[str, float] = {}
        self._lock = threading.Lock()

    def is_open(self, tool_name: str) -> bool:
        with self._lock:
            until = self._opened_until.get(tool_name)
            if until is None:
                return False
            if self._clock() >= until:
                self._opened_until.pop(tool_name, None)
                return False
            return True

    def record(self, tool_name: str, ok: bool) -> None:
        with self._lock:
            history = self._history.setdefault(tool_name, deque(maxlen=self.window))
            history.append(ok)
            if len(history) < self.min_sample:
                return
            error_rate = 1 - (sum(history) / len(history))
            if error_rate > self.error_rate_threshold:
                self._opened_until[tool_name] = self._clock() + self.open_seconds


@dataclass
class SharedState:
    """Objects that outlive a single request."""

    cache: ToolCache = field(default_factory=ToolCache)
    breaker: CircuitBreaker = field(default_factory=CircuitBreaker)


class RunState:
    """Mutable state of one agent request."""

    def __init__(
        self,
        guard: RunGuard | None = None,
        *,
        shared: SharedState | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        jitter: Callable[[float, float], float] = random.uniform,
    ) -> None:
        self.guard = guard or RunGuard()
        self.shared = shared or SharedState()
        self._clock = clock
        self._sleep = sleep
        self._jitter = jitter
        self._started_at = clock()
        self.steps = 0
        self.loop_declared = False
        self.tool_budget_used = 0
        self.last_call_meta: dict[str, Any] = {}
        self._call_counts: dict[str, int] = {}
        self._call_results: dict[str, Any] = {}
        self._fail_counts: dict[str, int] = {}
        self._disabled_tools: set[str] = set()
        self._recent_tools: list[str] = []

    # -- signals read by the agent loop ------------------------------------

    def wall_clock_exceeded(self) -> bool:
        return (self._clock() - self._started_at) >= self.guard.wall_clock_seconds

    def steps_exceeded(self) -> bool:
        return self.steps >= self.guard.max_steps

    def budget_near_limit(self) -> bool:
        """True above 80% of the request token budget: the loop should compact."""
        return self.tool_budget_used >= 0.8 * self.guard.tool_budget_tokens

    def force_final(self) -> bool:
        """True when the loop must stop calling tools and write the final answer."""
        return self.loop_declared or self.steps_exceeded() or self.wall_clock_exceeded()

    # -- dispatch ----------------------------------------------------------

    def dispatch(
        self,
        tool_name: str,
        args: dict[str, Any],
        run_fn: Callable[[], Any],
        *,
        product_line: str = "",
        date_from: str = "",
        date_to: str = "",
    ) -> Any:
        """Run (or memoise, or serve from cache) one tool call under the guard.

        Order of checks: budget exhausted, tool disabled, breaker open, memoised in
        this request, cached across requests, real execution.
        """
        refusal = self._budget_refusal()
        if refusal is not None:
            return refusal
        self.steps += 1
        self._record_spiral(tool_name)
        self.last_call_meta = {
            "step_index": self.steps,
            "cache_hit": False,
            "error_type": None,
            "truncated": False,
            "row_count": None,
        }
        blocked = self._tool_blocked(tool_name)
        if blocked is not None:
            return blocked
        key_in_request = call_hash(tool_name, args)
        memoised = self._memoised(key_in_request)
        if memoised is not None:
            return memoised
        shared_key = cache_key(tool_name, product_line, date_from, date_to, args)
        cached = self.shared.cache.get(shared_key)
        if cached is not None:
            return self._serve_cached(key_in_request, cached)
        return self._execute(tool_name, key_in_request, shared_key, run_fn)

    def reject(self, tool_name: str, error: dict[str, Any]) -> dict[str, Any]:
        """Account for a call that never reached the registry (its arguments were not JSON).

        It costs a step and feeds spiral detection, like any call, but it is neither
        memoised nor cached: caching it under the empty-argument key would answer a later,
        valid call to the same tool with this error.
        """
        refusal = self._budget_refusal()
        if refusal is not None:
            return refusal
        self.steps += 1
        self._record_spiral(tool_name)
        self.last_call_meta = {
            "step_index": self.steps,
            "cache_hit": False,
            "error_type": tool_error_type(error),
            "truncated": False,
            "row_count": None,
        }
        return error

    def _budget_refusal(self) -> dict[str, Any] | None:
        if self.steps_exceeded():
            return self._refuse(
                "step_budget_exceeded",
                f"The limit of {self.guard.max_steps} tool calls for this request was already "
                "reached. Do not call more tools. Write the answer from the data you already "
                "have and tell the user the query was limited.",
                "I consulted the maximum number of data sources allowed for this question.",
            )
        if self.wall_clock_exceeded():
            return self._refuse(
                "wall_clock_exceeded",
                f"The time budget ({self.guard.wall_clock_seconds:g}s) for this request was "
                "already used. Do not call more tools. Write the answer from the data you "
                "already have and tell the user the query was limited by time.",
                "This query took longer than expected and was stopped before it finished.",
            )
        return None

    def _refuse(self, error_type: str, hint: str, user_facing: str) -> dict[str, Any]:
        self.last_call_meta = {
            "step_index": self.steps,
            "cache_hit": False,
            "error_type": error_type,
            "truncated": False,
            "row_count": None,
        }
        return build_tool_error(error_type, hint=hint, user_facing=user_facing)

    def _record_spiral(self, tool_name: str) -> None:
        self._recent_tools.append(tool_name)
        if len(self._recent_tools) < 4:
            return
        first, second, third, fourth = self._recent_tools[-4:]
        if first == third and second == fourth and first != second:
            self.loop_declared = True

    def _tool_blocked(self, tool_name: str) -> dict[str, Any] | None:
        if tool_name in self._disabled_tools:
            self.last_call_meta["error_type"] = "tool_disabled"
            return build_tool_error(
                "tool_disabled",
                hint="This tool failed three times in a row and is disabled for the rest of "
                "this request.",
                user_facing="One of the data sources is unavailable right now.",
            )
        if self.shared.breaker.is_open(tool_name):
            self.last_call_meta["error_type"] = "upstream_5xx"
            return build_tool_error(
                "upstream_5xx",
                hint="This data source had too many recent errors and is paused for a short time.",
                user_facing="The data service is unstable right now.",
                retryable=True,
            )
        return None

    def _memoised(self, key: str) -> dict[str, Any] | None:
        count = self._call_counts.get(key, 0) + 1
        self._call_counts[key] = count
        if count == 1:
            return None
        if count >= self.guard.max_identical_calls + 1:
            self.loop_declared = True
        previous = self._call_results.get(key)
        self.last_call_meta["cache_hit"] = True
        self.last_call_meta["error_type"] = tool_error_type(previous)
        self.last_call_meta["row_count"] = infer_row_count(previous)
        return {
            "memoised_result": previous,
            "note": "You already made this exact call. Use this result, do not repeat the call.",
        }

    def _serve_cached(self, key: str, cached: Any) -> Any:
        self.last_call_meta["cache_hit"] = True
        self.last_call_meta["error_type"] = tool_error_type(cached)
        self.last_call_meta["row_count"] = infer_row_count(cached)
        self._call_results[key] = cached
        self.tool_budget_used += estimate_tokens(
            json.dumps(cached, ensure_ascii=False, default=str)
        )
        return cached

    def _execute(self, tool_name: str, key: str, shared_key: str, run_fn: Callable[[], Any]) -> Any:
        try:
            result = run_fn()
        except DataSourceError:
            self._sleep(self._retry_delay())
            try:
                result = run_fn()
            except Exception as exc:  # noqa: BLE001 - any second failure is reported, not raised
                return self._failure(tool_name, key, shared_key, "upstream_5xx", exc)
        except DataUnavailable as exc:
            return self._failure(tool_name, key, shared_key, "data_unavailable", exc)
        except Exception as exc:  # noqa: BLE001 - a code bug must never escape the loop
            return self._failure(tool_name, key, shared_key, "internal_error", exc)
        return self._success(tool_name, key, shared_key, result)

    def _retry_delay(self) -> float:
        base = self.guard.retry_base_delay_seconds
        spread = self.guard.retry_jitter_seconds
        return max(0.0, base + self._jitter(-spread, spread))

    def _success(self, tool_name: str, key: str, shared_key: str, result: Any) -> Any:
        """A call that returned. A returned error payload (bad arguments, unknown name) is
        still a healthy data source, so the breaker sees a success, but telemetry records
        the error type and the cache keeps it only for the short error TTL."""
        self.shared.breaker.record(tool_name, ok=True)
        result, truncated = cap_tool_result(result, self.guard.tool_result_token_cap)
        error_type = tool_error_type(result)
        self.last_call_meta["truncated"] = truncated
        self.last_call_meta["error_type"] = error_type
        self.last_call_meta["row_count"] = infer_row_count(result)
        self.tool_budget_used += estimate_tokens(
            json.dumps(result, ensure_ascii=False, default=str)
        )
        self._fail_counts[tool_name] = 0
        self._call_results[key] = result
        self.shared.cache.set(shared_key, result, error=error_type is not None)
        return result

    def _failure(
        self, tool_name: str, key: str, shared_key: str, error_type: str, exc: Exception
    ) -> dict[str, Any]:
        self._fail_counts[tool_name] = self._fail_counts.get(tool_name, 0) + 1
        if self._fail_counts[tool_name] >= 3:
            self._disabled_tools.add(tool_name)
        self.shared.breaker.record(tool_name, ok=False)
        result = _failure_payload(error_type, exc)
        self.last_call_meta["error_type"] = error_type
        self._call_results[key] = result
        self.shared.cache.set(shared_key, result, error=True)
        return result


def _failure_payload(error_type: str, exc: Exception) -> dict[str, Any]:
    if error_type == "upstream_5xx":
        return build_tool_error(
            "upstream_5xx",
            hint="Large windows often time out. Try a smaller window.",
            user_facing="The data service is unstable right now.",
            error_detail=type(exc).__name__,
        )
    if error_type == "data_unavailable":
        return build_tool_error(
            "data_unavailable",
            hint="This data is not available from the configured source.",
            user_facing="That data is not available right now.",
            error_detail=type(exc).__name__,
        )
    return build_tool_error(
        "internal_error",
        hint="Internal error while processing this query.",
        user_facing="This query could not be completed right now.",
        error_detail=type(exc).__name__,
        retryable=False,
    )
