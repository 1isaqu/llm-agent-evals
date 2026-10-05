"""The shipped scenario set, run end to end with the scripted model."""

import re
from pathlib import Path

import pytest

from llm_agent_evals.evals import compute_metrics, load_scenarios, run_scenarios

ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = load_scenarios(ROOT / "scenarios")
RESULTS = {r.scenario_id: r for r in run_scenarios(SCENARIOS)}
GOLDEN = [s for s in SCENARIOS if s.expect.should_pass]
SELF_TESTS = [s for s in SCENARIOS if not s.expect.should_pass]


def test_the_set_is_large_and_diverse_enough_to_mean_something():
    assert len(GOLDEN) >= 30 and len(SELF_TESTS) >= 10
    categories = {s.category for s in GOLDEN}
    assert categories >= {
        "factual",
        "trap",
        "comparison",
        "refusal",
        "fuzzy",
        "temporal",
        "vague",
        "multi_hop",
        "resilience",
        "adversarial",
        "validation",
        "explanation",
        "injection",
        "multi_line",
        "format",
    }


@pytest.mark.parametrize("scenario", GOLDEN, ids=lambda s: s.id)
def test_every_golden_scenario_is_clean(scenario):
    result = RESULTS[scenario.id]
    assert result.violations == (), [f"{v.id}: {v.message}" for v in result.violations]


@pytest.mark.parametrize("scenario", SELF_TESTS, ids=lambda s: s.id)
def test_every_self_test_shows_exactly_the_violations_it_declares(scenario):
    result = RESULTS[scenario.id]
    assert scenario.expect.expect_violations, "a self-test must name the violation it expects"
    declared = set(scenario.expect.expect_violations)
    assert declared <= set(result.violation_ids), result.violation_ids
    assert result.meets_expectation()


def test_the_metrics_of_the_scripted_run():
    metrics = compute_metrics(list(RESULTS.values()))
    for name in (
        "pass_rate",
        "tool_selection_accuracy",
        "param_accuracy",
        "faithfulness",
        "answer_accuracy",
        "detection_rate",
        "meets_expectation_rate",
    ):
        assert metrics[name] == 100.0, name
    assert 0 < metrics["steps_per_question"] < 3


def test_every_self_test_covers_a_distinct_layer_or_code():
    declared = [rid for s in SELF_TESTS for rid in s.expect.expect_violations]
    layers = {rid.split(":")[0] for rid in declared}
    assert layers >= {
        "tool_selection",
        "sql_channel",
        "params",
        "faithfulness",
        "ground_truth",
        "answer_text",
        "false_unavailability",
        "outcome",
        "harness",
    }


def test_scenario_ids_follow_the_naming_convention_and_files_match_ids():
    for scenario in SCENARIOS:
        assert scenario.id[0] in "SX" and scenario.id[1:].isdigit()
        assert Path(scenario.path).name.startswith(scenario.id.lower() + "_")
        assert (scenario.id[0] == "X") == (not scenario.expect.should_pass)


def _rows(section):
    """Rows of a fixture section, whether it is a plain list or a list of match blocks."""
    if not isinstance(section, list):
        return []
    for item in section:
        if isinstance(item, dict) and "rows" in item:
            yield from item["rows"]
        else:
            yield item


def test_every_scenario_uses_only_invented_names():
    portfolio = re.compile(r"^Portfolio [A-Z]-\d{2}")
    agents = {"agent_01", "agent_02", "agent_03", "agent_04"}
    for scenario in SCENARIOS:
        for sections in scenario.datasets.values():
            for row in _rows(sections.get("status_rollup")):
                assert portfolio.match(row["portfolio_name"]), row["portfolio_name"]
            for name in ("agent_rollup", "agent_status_rollup", "crosstab_rollup"):
                for row in _rows(sections.get(name)):
                    assert row["agent_id"] in agents and row["agent_name"].startswith("Agent 0")
