"""Aggregate scenario results into the numbers a CI gate or a report needs.

Every score is computed from the `layer` field of structured violations, never from the
text of a message. Two populations are kept apart:

* golden scenarios (`should_pass: true`) measure the agent. Their scores answer "how many
  scenarios were clean in this layer";
* self-test scenarios (`should_pass: false`) measure the harness. Their score is the
  detection rate: how many of them produced the violations they declare. A harness that
  stops detecting a class of mistake turns this number down, and the golden scores alone
  would never show it.

Skipped scenarios (self-tests in live mode) are excluded from everything.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from llm_agent_evals.evals.runner import ScenarioResult

LAYER_GROUPS: Mapping[str, tuple[str, ...]] = {
    "tool_selection_accuracy": ("tool_selection", "sql_channel"),
    "param_accuracy": ("params",),
    "faithfulness": ("faithfulness", "ground_truth", "false_unavailability"),
    "answer_accuracy": ("answer_text", "outcome"),
}


def _percent(part: int, total: int) -> float | None:
    return round(100.0 * part / total, 2) if total else None


def _clean_in(result: ScenarioResult, layers: tuple[str, ...]) -> bool:
    return not any(v.layer in layers for v in result.violations)


def compute_metrics(results: Sequence[ScenarioResult]) -> dict[str, Any]:
    """Scores over golden scenarios, detection over self-tests, plus per-category counts."""
    active = [r for r in results if not r.skipped]
    golden = [r for r in active if r.should_pass]
    selftests = [r for r in active if not r.should_pass]
    metrics: dict[str, Any] = {
        "scenarios": len(results),
        "skipped": len(results) - len(active),
        "golden": len(golden),
        "self_tests": len(selftests),
        "pass_rate": _percent(sum(1 for r in golden if not r.violations), len(golden)),
        "detection_rate": _percent(
            sum(1 for r in selftests if r.meets_expectation()), len(selftests)
        ),
        "meets_expectation_rate": _percent(
            sum(1 for r in active if r.meets_expectation()), len(active)
        ),
        "steps_per_question": round(sum(r.tool_calls for r in golden) / len(golden), 2)
        if golden
        else None,
    }
    for name, layers in LAYER_GROUPS.items():
        metrics[name] = _percent(sum(1 for r in golden if _clean_in(r, layers)), len(golden))
    metrics["by_category"] = _by_category(golden)
    return metrics


def _by_category(golden: Sequence[ScenarioResult]) -> dict[str, dict[str, int]]:
    table: dict[str, dict[str, int]] = {}
    for result in golden:
        row = table.setdefault(result.category, {"scenarios": 0, "clean": 0})
        row["scenarios"] += 1
        row["clean"] += 0 if result.violations else 1
    return dict(sorted(table.items()))


def gate_failures(metrics: Mapping[str, Any], minimums: Mapping[str, float]) -> list[str]:
    """Descriptions of every minimum that `metrics` does not reach. Empty means the gate passes.

    A metric that is None (no scenarios in its population) fails a gate that asks for it:
    "no evidence" is not a pass.
    """
    failures = []
    for name, minimum in sorted(minimums.items()):
        value = metrics.get(name)
        if value is None:
            failures.append(f"{name}: no scenarios to measure (minimum {minimum:g})")
        elif value < minimum:
            failures.append(f"{name}: {value:g} is below the minimum {minimum:g}")
    return failures
