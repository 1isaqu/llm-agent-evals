"""Render scenario results as Markdown or JSON."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from llm_agent_evals.evals.runner import ScenarioResult


def result_status(result: ScenarioResult) -> str:
    if result.skipped:
        return "skipped"
    if result.meets_expectation():
        return "ok" if result.should_pass else "detected"
    return "FAIL" if result.should_pass else "MISSED"


def to_dict(
    results: Sequence[ScenarioResult], metrics: Mapping[str, Any], mode: str
) -> dict[str, Any]:
    """A JSON-serialisable report. Contains ids, categories and violations, never answers."""
    return {
        "mode": mode,
        "metrics": dict(metrics),
        "scenarios": [
            {
                "id": r.scenario_id,
                "category": r.category,
                "kind": "golden" if r.should_pass else "self_test",
                "status": result_status(r),
                "tool_calls": r.tool_calls,
                "violations": [{"id": v.id, "message": v.message} for v in r.violations],
            }
            for r in results
        ],
    }


def to_json(results: Sequence[ScenarioResult], metrics: Mapping[str, Any], mode: str) -> str:
    return json.dumps(to_dict(results, metrics, mode), indent=2, ensure_ascii=False) + "\n"


def _fmt(value: Any) -> str:
    return "n/a" if value is None else f"{value:g}"


def to_markdown(results: Sequence[ScenarioResult], metrics: Mapping[str, Any], mode: str) -> str:
    lines = [
        f"# Evaluation report ({mode} mode)",
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "|---|---|",
    ]
    for name in (
        "scenarios",
        "golden",
        "self_tests",
        "skipped",
        "pass_rate",
        "tool_selection_accuracy",
        "param_accuracy",
        "faithfulness",
        "answer_accuracy",
        "steps_per_question",
        "detection_rate",
        "meets_expectation_rate",
    ):
        lines.append(f"| {name} | {_fmt(metrics.get(name))} |")
    lines += ["", "## Scenarios", "", "| Id | Category | Kind | Status | Tool calls | Violations |"]
    lines.append("|---|---|---|---|---|---|")
    for result in results:
        kind = "golden" if result.should_pass else "self-test"
        found = ", ".join(result.violation_ids) or "none"
        lines.append(
            f"| {result.scenario_id} | {result.category} | {kind} | {result_status(result)} | "
            f"{result.tool_calls} | {found} |"
        )
    return "\n".join(lines) + "\n"
