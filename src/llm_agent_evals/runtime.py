"""Wiring: build an `Agent` from environment settings and a data backend.

Two backends exist. `demo` builds invented in-memory SQLite databases (one per product
line) and pins the clock to `DEMO_NOW`, so the demo data is always "recent". `sqlserver`
opens one read-only connection per product line from the `DB_*` variables and uses the
real clock; it has never been run against a live server. Anything else (your own
`DataSource` objects) goes through `build_agent(sources=...)` directly.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from datetime import datetime
from typing import Any

from llm_agent_evals.agent import Agent
from llm_agent_evals.business import BusinessConfig
from llm_agent_evals.errors import ConfigError
from llm_agent_evals.llm import LLMClient, build_llm
from llm_agent_evals.session import PRODUCT_LINES
from llm_agent_evals.settings import AgentSettings, LLMSettings
from llm_agent_evals.sources.base import CachedSource, DataSource
from llm_agent_evals.sources.demo import DEMO_NOW, build_demo_connections
from llm_agent_evals.sources.sql import SQLITE, SQLSERVER
from llm_agent_evals.sources.sqlsource import SqlDataSource, connect_sqlserver
from llm_agent_evals.telemetry import build_observer

BACKENDS = ("demo", "sqlserver")


def demo_sources(config: BusinessConfig, cache_ttl: float = 60.0) -> dict[str, DataSource]:
    """One cached SQL source per product line over invented in-memory databases."""
    connections = build_demo_connections()
    return {
        line: CachedSource(SqlDataSource(conn, SQLITE, config), cache_ttl)
        for line, conn in connections.items()
    }


def sqlserver_sources(
    env: Mapping[str, str],
    config: BusinessConfig,
    cache_ttl: float = 60.0,
    connect: Callable[..., Any] | None = None,
) -> dict[str, DataSource]:
    """One cached SQL Server source per product line (`DB_NAME_<LINE>` or `DB_NAME`)."""
    return {
        line: CachedSource(
            SqlDataSource(connect_sqlserver(env, line, connect), SQLSERVER, config), cache_ttl
        )
        for line in PRODUCT_LINES
    }


def build_agent(
    env: Mapping[str, str] | None = None,
    *,
    backend: str = "demo",
    sources: Mapping[str, DataSource] | None = None,
    llm: LLMClient | None = None,
    now: Callable[[], datetime] | None = None,
) -> Agent:
    """Create an agent from the environment. Explicit arguments win over the environment."""
    if backend not in BACKENDS:
        raise ConfigError(f"backend must be one of: {', '.join(BACKENDS)}")
    llm_settings = LLMSettings.from_env(env)
    llm_settings.require_ready()
    settings = AgentSettings.from_env(env)
    config = settings.business_config()
    chosen_now = now
    if sources is None:
        if backend == "demo":
            sources = demo_sources(config, settings.cache_ttl_seconds)
            chosen_now = chosen_now or (lambda: DEMO_NOW)
        else:
            sources = sqlserver_sources(
                os.environ if env is None else env, config, settings.cache_ttl_seconds
            )
    return Agent(
        llm or build_llm(llm_settings),
        sources,
        config=config,
        settings=settings,
        observer=build_observer(settings.telemetry_path),
        now=chosen_now or datetime.now,
        max_tokens=llm_settings.max_tokens,
        temperature=llm_settings.temperature,
    )
