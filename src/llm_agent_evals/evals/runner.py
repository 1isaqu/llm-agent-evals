"""Run scenarios through the real agent with a scripted (or real) model.

What runs for real: the agent loop, the tool registry and schema validation, the run guard,
the analytics code, the egress guard, the answer parser and the sanitiser. What is
replaced: the model (a `ScriptedLLM` replaying the scenario script, or a real client in
live mode) and the SQL layer (a `FixtureSource` serving the scenario's rollup rows).

In scripted mode this measures the machinery, not the model: the script is the behaviour
the scenario says is correct, so a passing scenario proves the harness and the guards
handle that behaviour, and a scenario declared `should_pass: false` proves the harness
detects a specific mistake. In live mode the script is ignored and a real model is
measured against the same expectations, at real token cost.

Time is injected: the clock the guards read is a fake that advances only when a scenario
asks for it, `sleep` does nothing and "now" is the scenario's own `now`. Runs are
reproducible and fast.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from llm_agent_evals.agent import Agent, AgentResult
from llm_agent_evals.business import BusinessConfig
from llm_agent_evals.errors import AgentError, ModelError
from llm_agent_evals.evals.checks import Violation, evaluate
from llm_agent_evals.evals.scenario import Scenario
from llm_agent_evals.guards import RunGuard, SharedState
from llm_agent_evals.llm.base import LLMClient
from llm_agent_evals.llm.mock import ScriptedLLM, ScriptExhausted
from llm_agent_evals.settings import AgentSettings
from llm_agent_evals.sources.base import DataSource
from llm_agent_evals.sources.fixture import FixtureSource


class FakeClock:
    """A monotonic clock that moves `tick` seconds on every read."""

    def __init__(self, tick: float = 0.0) -> None:
        self._now = 0.0
        self._tick = tick

    def __call__(self) -> float:
        self._now += self._tick
        return self._now


@dataclass(frozen=True)
class ScenarioResult:
    """The outcome of one scenario: what happened and whether it matched the declaration."""

    scenario_id: str
    category: str
    should_pass: bool
    violations: tuple[Violation, ...] = ()
    required: tuple[str, ...] = ()
    tool_calls: int = 0
    skipped: str = ""
    result: AgentResult | None = field(default=None, repr=False)

    @property
    def violation_ids(self) -> tuple[str, ...]:
        return tuple(v.id for v in self.violations)

    def meets_expectation(self) -> bool:
        """A golden scenario must be clean. A self-test must show the declared violations."""
        if self.skipped:
            return True
        if self.should_pass:
            return not self.violations
        if self.required:
            return all(rid in self.violation_ids for rid in self.required)
        return bool(self.violations)


def build_sources(scenario: Scenario) -> dict[str, DataSource]:
    return {line: FixtureSource(sections) for line, sections in scenario.datasets.items()}


def build_agent(scenario: Scenario, llm: LLMClient) -> Agent:
    """An agent wired to the scenario's data, limits and clock."""
    limits = scenario.limits
    settings = AgentSettings(
        max_rounds=int(limits.get("max_rounds", 8)),
        max_steps=int(limits.get("max_steps", 8)),
        wall_clock_seconds=float(limits.get("wall_clock_seconds", 60.0)),
    )
    guard = RunGuard(
        max_steps=settings.max_steps,
        wall_clock_seconds=settings.wall_clock_seconds,
        retry_base_delay_seconds=0.0,
        retry_jitter_seconds=0.0,
    )
    config = BusinessConfig.from_mapping(scenario.config) if scenario.config else BusinessConfig()
    return Agent(
        llm,
        build_sources(scenario),
        config=config,
        settings=settings,
        shared=SharedState(),
        now=lambda: scenario.now,
        clock=FakeClock(float(limits.get("clock_tick_seconds", 0.0))),
        sleep=lambda _seconds: None,
        guard=guard,
    )


def _harness_violation(code: str, message: str) -> Violation:
    return Violation("harness", code, message)


def run_scenario(scenario: Scenario, llm: LLMClient | None = None) -> ScenarioResult:
    """Run one scenario. `llm=None` replays the scripted model; pass a client for live mode."""
    model = llm if llm is not None else ScriptedLLM(scenario.script)
    agent = build_agent(scenario, model)
    base = {
        "scenario_id": scenario.id,
        "category": scenario.category,
        "should_pass": scenario.expect.should_pass,
        "required": tuple(scenario.expect.expect_violations),
    }
    try:
        result = agent.run(scenario.messages, scenario.session, run_id=f"eval-{scenario.id}")
    except ModelError as exc:
        violations = (
            []
            if scenario.expect.model_error
            else [_harness_violation("model_error", f"the model provider failed: {exc}")]
        )
        return ScenarioResult(violations=tuple(violations), **base)
    except ScriptExhausted as exc:
        return ScenarioResult(
            violations=(_harness_violation("script_exhausted", str(exc)),), **base
        )
    except AgentError as exc:
        return ScenarioResult(violations=(_harness_violation("run_failed", str(exc)),), **base)
    violations = list(evaluate(scenario, result, agent.registry.names()))
    if scenario.expect.model_error:
        violations.append(
            _harness_violation(
                "model_error_missing", "the model failure was expected but did not occur"
            )
        )
    return ScenarioResult(
        violations=tuple(violations), tool_calls=len(result.calls), result=result, **base
    )


def run_scenarios(
    scenarios: Sequence[Scenario],
    llm_factory: Callable[[], LLMClient] | None = None,
) -> list[ScenarioResult]:
    """Run every scenario. With `llm_factory` (live mode) self-tests are skipped.

    Self-tests depend on a script that misbehaves on purpose, which a live model does not
    follow, so measuring them live would say nothing.
    """
    results: list[ScenarioResult] = []
    for scenario in scenarios:
        if llm_factory is not None and not scenario.expect.should_pass:
            results.append(
                ScenarioResult(
                    scenario_id=scenario.id,
                    category=scenario.category,
                    should_pass=False,
                    skipped="self-test needs the scripted model",
                )
            )
            continue
        results.append(run_scenario(scenario, llm_factory() if llm_factory else None))
    return results
