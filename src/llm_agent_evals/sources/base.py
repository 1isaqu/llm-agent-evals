"""The data source protocol and three composable wrappers.

A data source answers nine questions, each as a list of plain dict rows. The rows
are *additive*: combining two product lines is concatenation, because every
aggregation in `analytics/` sums rows that share a key. That keeps multi-line
sessions trivial and makes it impossible to average two ratios by accident.

Row shapes (all keys lower case):

    status_rollup      portfolio_name, status_id, qty, amount
    daily_rollup       day, status_id, qty, amount
    agent_rollup       agent_id, agent_name, attempt_count, answer_count, cpc_count,
                       agreement_count, agreement_value, first_installment_value,
                       agreement_count_by_contract, slips_issued, slips_paid,
                       exception_count, exception_value
    agent_status_rollup  agent_id, agent_name, status_id, qty, amount
    crosstab_rollup    agent_id, agent_name, portfolio_name, status_id, qty, amount
    plan_stage_rollup  portfolio_name, stage, qty, open_amount
    slip_rollup        group_key, generated, paid_on_time, amount_maturing, amount_received
    drilldown_rows     agreement_id, portfolio_id, first_installment_value, total_value,
                       agent_name, agent_id, issued_on, due_on, installments
    hourly_counts      day, hour, qty
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from datetime import date
from typing import Any, Protocol, runtime_checkable

from llm_agent_evals.session import Window

Row = dict[str, Any]


@runtime_checkable
class DataSource(Protocol):
    """Read-only access to the analytics rollups. Every method may raise
    `DataSourceError` (transient) or `DataUnavailable` (not provided)."""

    def status_rollup(self, window: Window) -> list[Row]: ...

    def daily_rollup(self, window: Window, portfolio: str | None = None) -> list[Row]: ...

    def agent_rollup(self, window: Window) -> list[Row]: ...

    def agent_status_rollup(self, window: Window) -> list[Row]: ...

    def crosstab_rollup(self, window: Window, by: str, value: str) -> list[Row]: ...

    def plan_stage_rollup(self, end: date) -> list[Row]: ...

    def slip_rollup(
        self, window: Window, group_by: str, date_field: str, portfolio: str | None = None
    ) -> list[Row]: ...

    def drilldown_rows(self, kind: str, portfolio: str, window: Window) -> list[Row]: ...

    def hourly_counts(self, window: Window) -> list[Row]: ...


SOURCE_METHODS = (
    "status_rollup",
    "daily_rollup",
    "agent_rollup",
    "agent_status_rollup",
    "crosstab_rollup",
    "plan_stage_rollup",
    "slip_rollup",
    "drilldown_rows",
    "hourly_counts",
)


class MultiLineSource:
    """Concatenate the rows of several sources (one per product line)."""

    def __init__(self, sources: Sequence[DataSource]) -> None:
        if not sources:
            raise ValueError("MultiLineSource needs at least one source")
        self._sources = list(sources)

    def __getattr__(self, name: str) -> Callable[..., list[Row]]:
        if name not in SOURCE_METHODS:
            raise AttributeError(name)

        def combined(*args: Any, **kwargs: Any) -> list[Row]:
            rows: list[Row] = []
            for source in self._sources:
                rows.extend(getattr(source, name)(*args, **kwargs))
            return rows

        return combined


class CachedSource:
    """Time-limited cache in front of another source.

    The agent loads the portfolio rollup before the first tool call and several tools
    ask for the same rows again. Caching by (method, arguments) keeps that to one query.
    """

    def __init__(
        self,
        inner: DataSource,
        ttl_seconds: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._inner = inner
        self._ttl = ttl_seconds
        self._clock = clock
        self._store: dict[tuple[Any, ...], tuple[float, list[Row]]] = {}

    def clear(self) -> None:
        self._store.clear()

    def __getattr__(self, name: str) -> Callable[..., list[Row]]:
        if name not in SOURCE_METHODS:
            raise AttributeError(name)
        method = getattr(self._inner, name)

        def cached(*args: Any, **kwargs: Any) -> list[Row]:
            key = (name, args, tuple(sorted(kwargs.items())))
            now = self._clock()
            entry = self._store.get(key)
            if entry is not None and entry[0] > now:
                return [dict(row) for row in entry[1]]
            rows = method(*args, **kwargs)
            if self._ttl > 0:
                self._store[key] = (now + self._ttl, [dict(row) for row in rows])
            return rows

        return cached


def merged_sources(
    sources: Mapping[str, DataSource], lines: Sequence[str]
) -> DataSource | MultiLineSource:
    """One source for `lines`: the line's own source, or a multi-line merge."""
    chosen = [sources[line] for line in lines]
    return chosen[0] if len(chosen) == 1 else MultiLineSource(chosen)
