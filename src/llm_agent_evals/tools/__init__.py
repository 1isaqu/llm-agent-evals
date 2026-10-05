"""The default tool set: 15 read-only analytics tools."""

from __future__ import annotations

from llm_agent_evals.registry import ToolRegistry
from llm_agent_evals.session import PRODUCT_LINES
from llm_agent_evals.tools import agent_tools, drilldown_tools, history_tools, portfolio_tools


def build_default_registry(lines: tuple[str, ...] = PRODUCT_LINES) -> ToolRegistry:
    """Build the registry. `lines` are the product-line names the per-call argument accepts."""
    registry = ToolRegistry()
    portfolio_tools.register(registry)
    agent_tools.register(registry, lines)
    history_tools.register(registry, lines)
    drilldown_tools.register(registry, lines)
    return registry


__all__ = ["build_default_registry"]
