"""Small inline fixtures and builders shared by the tests. Every value is invented."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

from llm_agent_evals.agent import Agent
from llm_agent_evals.business import BusinessConfig
from llm_agent_evals.context import ToolContext
from llm_agent_evals.evals.runner import FakeClock
from llm_agent_evals.guards import RunGuard, SharedState
from llm_agent_evals.llm.mock import ScriptedLLM
from llm_agent_evals.session import Session, Window
from llm_agent_evals.settings import AgentSettings
from llm_agent_evals.sources.fixture import FixtureSource

NOW = datetime(2026, 3, 18, 15, 30)
WINDOW = Window.from_iso("2026-03-01", "2026-03-18")
SESSION = Session("auto", WINDOW)

P1, P2 = "Portfolio T-01", "Portfolio T-02"


def status_row(portfolio: str, status_id: int, qty: int, amount: float) -> dict[str, Any]:
    return {"portfolio_name": portfolio, "status_id": status_id, "qty": qty, "amount": amount}


def agent_row(agent_id: str = "agent_01", **overrides: Any) -> dict[str, Any]:
    row = {
        "agent_id": agent_id,
        "agent_name": agent_id.replace("_", " ").title(),
        "attempt_count": 100,
        "answer_count": 50,
        "cpc_count": 20,
        "agreement_count": 5,
        "agreement_value": 1000.0,
        "first_installment_value": 300.0,
        "agreement_count_by_contract": 6,
        "slips_issued": 5,
        "slips_paid": 3,
        "exception_count": 1,
        "exception_value": 100.0,
    }
    row.update(overrides)
    return row


# T-01: generated 500 (ACTIVE 400 + BROKEN 100), exception 100  -> universe 600
# T-02: generated 200 (ACTIVE 200), rejected 100                 -> universe 300
BASE_STATUS = [
    status_row(P1, 10, 4, 400.0),
    status_row(P1, 20, 1, 100.0),
    status_row(P1, 40, 1, 100.0),
    status_row(P2, 10, 2, 200.0),
    status_row(P2, 50, 1, 100.0),
]


def base_sections(**overrides: Any) -> dict[str, Any]:
    sections: dict[str, Any] = {
        "status_rollup": BASE_STATUS,
        "agent_rollup": [agent_row("agent_01"), agent_row("agent_02", cpc_count=40)],
    }
    sections.update(overrides)
    return sections


def fixture_sources(
    auto: Mapping[str, Any] | None = None, consumer: Mapping[str, Any] | None = None
) -> dict[str, FixtureSource]:
    sources = {"auto": FixtureSource(auto if auto is not None else base_sections())}
    if consumer is not None:
        sources["consumer"] = FixtureSource(consumer)
    return sources


def make_context(
    sections: Mapping[str, Any] | None = None,
    *,
    line: str = "auto",
    window: Window = WINDOW,
    now: datetime = NOW,
    config: BusinessConfig | None = None,
    consumer: Mapping[str, Any] | None = None,
) -> ToolContext:
    sources = fixture_sources(sections, consumer)
    return ToolContext(Session(line, window), config or BusinessConfig(), sources, now)


def make_agent(
    script: Sequence[Any],
    sections: Mapping[str, Any] | None = None,
    *,
    consumer: Mapping[str, Any] | None = None,
    guard: RunGuard | None = None,
    settings: AgentSettings | None = None,
    **kwargs: Any,
) -> Agent:
    """An agent over fixture data and a scripted model, with a fake clock and no sleeping."""
    kwargs.setdefault("now", lambda: NOW)
    kwargs.setdefault("clock", FakeClock())
    kwargs.setdefault("sleep", lambda _seconds: None)
    kwargs.setdefault("shared", SharedState())
    return Agent(
        ScriptedLLM(script),
        fixture_sources(sections, consumer),
        guard=guard or RunGuard(retry_base_delay_seconds=0.0, retry_jitter_seconds=0.0),
        settings=settings,
        **kwargs,
    )


def call(name: str, **args: Any) -> dict[str, Any]:
    return {"tool_call": {"name": name, "args": args}}


def final(text: str = "Done.", confidence: str = "high", **extra: Any) -> dict[str, Any]:
    body = {
        "text": text,
        "highlights": [],
        "suggested_actions": [],
        "data_sources": [],
        "confidence": confidence,
    }
    body.update(extra)
    return {"final": body}


def ask(agent: Agent, question: str = "How is it going?", session: Session = SESSION) -> Any:
    return agent.run([{"role": "user", "content": question}], session)
