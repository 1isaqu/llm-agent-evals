"""The context object handed to every tool handler."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

from llm_agent_evals.analytics.agents import agent_entries
from llm_agent_evals.analytics.pace import BaselinePace, PaceProvider
from llm_agent_evals.analytics.portfolio import portfolio_entries
from llm_agent_evals.business import BusinessConfig
from llm_agent_evals.metrics import MetricRegistry
from llm_agent_evals.session import ALL_LINES, Session, Window
from llm_agent_evals.sources.base import DataSource, merged_sources


@dataclass
class ToolContext:
    """Everything a tool needs: the session, the sources, the clock and the config.

    `now` is injected, never read from the system clock inside a tool, so a relative
    question ("today", "yesterday", "this quarter") resolves the same way in a test as in
    production. Portfolio and agent entries are computed lazily and cached per
    (product line, window) for the lifetime of the context, which is one agent request.
    """

    session: Session
    config: BusinessConfig
    sources: Mapping[str, DataSource]
    now: datetime
    pace: PaceProvider = field(default_factory=BaselinePace)
    _metrics: MetricRegistry | None = field(default=None, repr=False)
    _cache: dict[tuple[Any, ...], list[dict[str, Any]]] = field(default_factory=dict, repr=False)

    def metric_registry(self) -> MetricRegistry:
        """The metric registry for this context's business config (built on first use)."""
        if self._metrics is None:
            self._metrics = MetricRegistry(self.config)
        return self._metrics

    @property
    def today(self) -> date:
        return self.now.date()

    @property
    def lines(self) -> tuple[str, ...]:
        return tuple(self.sources)

    def source_for(self, line: str | None = None) -> Any:
        """The source for a product line. None means the session's line, 'all' merges them."""
        chosen = line or self.session.product_line
        if chosen == ALL_LINES:
            return merged_sources(self.sources, self.lines)
        return self.sources[chosen]

    def entries(
        self, line: str | None = None, window: Window | None = None
    ) -> list[dict[str, Any]]:
        """Portfolio entries for a line and window (defaults: the session's)."""
        use_line = line or self.session.product_line
        use_window = window or self.session.window
        key = ("portfolios", use_line, use_window)
        if key not in self._cache:
            rows = self.source_for(use_line).status_rollup(use_window)
            self._cache[key] = portfolio_entries(rows, self.config, use_window.end.isoformat())
        return self._cache[key]

    def agents(self, line: str | None = None, window: Window | None = None) -> list[dict[str, Any]]:
        """Agent entries for a line and window (defaults: the session's)."""
        use_line = line or self.session.product_line
        use_window = window or self.session.window
        key = ("agents", use_line, use_window)
        if key not in self._cache:
            rows = self.source_for(use_line).agent_rollup(use_window)
            self._cache[key] = agent_entries(rows, use_window.end.isoformat())
        return self._cache[key]
