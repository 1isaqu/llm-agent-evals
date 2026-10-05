"""Evaluation harness: scenarios, assertion layers, runner, metrics and reports."""

from llm_agent_evals.evals.checks import Violation, evaluate
from llm_agent_evals.evals.metrics import compute_metrics, gate_failures
from llm_agent_evals.evals.runner import ScenarioResult, run_scenario, run_scenarios
from llm_agent_evals.evals.scenario import (
    Expectations,
    Scenario,
    ScenarioError,
    load_scenario,
    load_scenarios,
)

__all__ = [
    "Expectations",
    "Scenario",
    "ScenarioError",
    "ScenarioResult",
    "Violation",
    "compute_metrics",
    "evaluate",
    "gate_failures",
    "load_scenario",
    "load_scenarios",
    "run_scenario",
    "run_scenarios",
]
