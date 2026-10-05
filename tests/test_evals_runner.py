import json
from pathlib import Path

import pytest

from llm_agent_evals.evals import (
    Violation,
    compute_metrics,
    gate_failures,
    report,
    run_scenario,
    run_scenarios,
)
from llm_agent_evals.evals.runner import FakeClock, ScenarioResult
from llm_agent_evals.evals.scenario import parse_scenario
from llm_agent_evals.llm import RuleBasedLLM

DATASET = {
    "auto": {
        "status_rollup": [
            {"portfolio_name": "Portfolio T-01", "status_id": 10, "qty": 4, "amount": 400.0},
            {"portfolio_name": "Portfolio T-01", "status_id": 40, "qty": 1, "amount": 100.0},
        ],
        "agent_rollup": [],
    }
}
FINAL = {
    "final": {
        "text": "Portfolio T-01 is at low risk.",
        "highlights": [],
        "suggested_actions": [],
        "data_sources": [],
        "confidence": "high",
    }
}


def scenario(script, expect=None, **extra):
    body = {
        "id": "T1",
        "title": "t",
        "category": "factual",
        "question": "How is Portfolio T-01?",
        "session": {"product_line": "auto", "date_from": "2026-03-01", "date_to": "2026-03-18"},
        "now": "2026-03-18T15:30:00",
        "datasets": DATASET,
        "script": script,
        "expect": expect or {},
    }
    body.update(extra)
    return parse_scenario(body, Path("t1.json"))


def call(name, **args):
    return {"tool_call": {"name": name, "args": args}}


def test_a_clean_scripted_run_has_no_violations():
    result = run_scenario(
        scenario(
            [call("get_portfolio_metrics", portfolio_name="Portfolio T-01"), FINAL],
            {"tools_any_of": ["get_portfolio_metrics"]},
        )
    )
    assert result.violations == () and result.tool_calls == 1 and result.meets_expectation()
    assert result.result.response["confidence"] == "high"


def test_a_script_that_is_too_short_is_a_harness_failure_not_a_pass():
    result = run_scenario(
        scenario([call("get_portfolio_metrics", portfolio_name="Portfolio T-01")])
    )
    assert result.violation_ids == ("harness:script_exhausted",) and not result.meets_expectation()


def test_a_provider_failure_is_a_violation_unless_the_scenario_expects_it():
    unexpected = run_scenario(scenario([{"fail": "down"}]))
    assert unexpected.violation_ids == ("harness:model_error",)
    expected = run_scenario(scenario([{"fail": "down"}], {"model_error": True}))
    assert expected.violations == () and expected.meets_expectation()
    missing = run_scenario(scenario([FINAL], {"model_error": True}))
    assert missing.violation_ids == ("harness:model_error_missing",)


def test_a_session_over_all_lines_runs_on_the_lines_the_dataset_has():
    all_lines = scenario(
        [FINAL], session={"product_line": "all", "date_from": "2026-03-01", "date_to": "2026-03-18"}
    )
    assert run_scenario(all_lines).violations == ()


def test_the_wall_clock_limit_is_driven_by_the_scenario_clock_not_real_time():
    sc = scenario(
        [call("get_status_breakdown"), FINAL],
        limits={"wall_clock_seconds": 5, "clock_tick_seconds": 3},
    )
    result = run_scenario(sc)
    assert result.result.forced_final and result.result.response["confidence"] == "high"
    assert "wall_clock_exceeded" in {c.record.error_type for c in result.result.calls}


def test_the_fake_clock_advances_only_when_read():
    clock = FakeClock(2.0)
    assert (clock(), clock()) == (2.0, 4.0)
    assert FakeClock()() == 0.0


def test_a_self_test_must_show_the_declared_violations():
    script = [FINAL]
    expect = {
        "should_pass": False,
        "tools_any_of": ["get_portfolio_metrics"],
        "expect_violations": ["tool_selection:missing_expected_tool"],
    }
    assert run_scenario(scenario(script, expect)).meets_expectation()
    wrong = {**expect, "expect_violations": ["params:missing_params"]}
    assert not run_scenario(scenario(script, wrong)).meets_expectation()
    no_declared = {"should_pass": False}
    assert not run_scenario(scenario(script, no_declared)).meets_expectation()  # nothing went wrong


def test_live_mode_skips_self_tests_and_runs_the_rest_with_the_given_model():
    golden = scenario(
        [],
        {"tools_any_of": ["filter_portfolios_by_risk"]},
        question="Which portfolios are at high risk?",
    )
    selftest = scenario([FINAL], {"should_pass": False, "tools_any_of": ["x_tool"]}, id="T2")
    results = run_scenarios([golden, selftest], llm_factory=RuleBasedLLM)
    assert results[0].skipped == "" and results[0].violations == ()
    assert results[1].skipped and results[1].meets_expectation()


def test_every_scenario_gets_a_fresh_cache_so_runs_do_not_influence_each_other():
    sc = scenario([call("get_status_breakdown"), FINAL])
    first, second = run_scenario(sc), run_scenario(sc)
    assert first.result.calls[0].record.cache_hit is False
    assert second.result.calls[0].record.cache_hit is False


# -- metrics -------------------------------------------------------------------


def res(sid, category="factual", should_pass=True, violations=(), calls=1, required=()):
    return ScenarioResult(sid, category, should_pass, tuple(violations), tuple(required), calls)


def v(layer, code="x"):
    return Violation(layer, code, "m")


def test_metrics_are_computed_from_layers_and_split_golden_from_self_tests():
    results = [
        res("a"),
        res("b", violations=[v("params")]),
        res("c", violations=[v("faithfulness")]),
        res("d", violations=[v("tool_selection")], calls=3),
        res("e", violations=[v("answer_text")]),
        res("x1", should_pass=False, violations=[v("params")]),
        res("x2", should_pass=False),
    ]
    m = compute_metrics(results)
    assert (m["golden"], m["self_tests"]) == (5, 2)
    assert m["pass_rate"] == 20.0 and m["param_accuracy"] == 80.0 and m["faithfulness"] == 80.0
    assert m["tool_selection_accuracy"] == 80.0 and m["answer_accuracy"] == 80.0
    assert m["steps_per_question"] == 1.4
    assert m["detection_rate"] == 50.0  # x2 declared a failure and showed none
    assert m["by_category"] == {"factual": {"scenarios": 5, "clean": 1}}


def test_a_message_can_never_change_a_score():
    one = [
        res("a", violations=[Violation("params", "missing_params", "about faithfulness and tools")])
    ]
    assert (
        compute_metrics(one)["faithfulness"] == 100.0
        and compute_metrics(one)["param_accuracy"] == 0.0
    )


def test_skipped_scenarios_are_excluded_and_empty_populations_are_none():
    skipped = ScenarioResult("x", "self_test", False, skipped="live")
    m = compute_metrics([skipped])
    assert (
        m["skipped"] == 1
        and m["golden"] == 0
        and m["pass_rate"] is None
        and m["detection_rate"] is None
    )


def test_a_metric_with_no_evidence_fails_a_gate_that_asks_for_it():
    assert gate_failures({"pass_rate": None}, {"pass_rate": 90}) == [
        "pass_rate: no scenarios to measure (minimum 90)"
    ]
    assert gate_failures({"pass_rate": 85.5}, {"pass_rate": 90}) == [
        "pass_rate: 85.5 is below the minimum 90"
    ]
    assert gate_failures({"pass_rate": 100}, {"pass_rate": 90}) == []
    assert gate_failures({}, {}) == []


# -- reports -------------------------------------------------------------------


def test_the_markdown_report_lists_metrics_and_every_scenario():
    results = [
        res("S01"),
        res("S02", violations=[v("params", "missing_params")]),
        res("X01", should_pass=False, violations=[v("params")], required=["params:other"]),
    ]
    text = report.to_markdown(results, compute_metrics(results), "scripted")
    assert "| pass_rate | 50 |" in text
    assert "| S02 | factual | golden | FAIL | 1 | params:missing_params |" in text
    assert "| X01 | factual | self-test | MISSED |" in text


def test_the_json_report_has_ids_and_violations_but_no_answer_text():
    results = [res("S01"), res("S02", violations=[v("params", "missing_params")])]
    data = json.loads(report.to_json(results, compute_metrics(results), "scripted"))
    assert data["mode"] == "scripted" and data["metrics"]["golden"] == 2
    assert data["scenarios"][1]["violations"] == [{"id": "params:missing_params", "message": "m"}]
    assert {s["status"] for s in data["scenarios"]} == {"ok", "FAIL"}


@pytest.mark.parametrize(
    ("result", "word"),
    [
        (res("a"), "ok"),
        (res("a", violations=[v("params")]), "FAIL"),
        (res("a", should_pass=False, violations=[v("params")]), "detected"),
        (res("a", should_pass=False), "MISSED"),
        (ScenarioResult("a", "c", False, skipped="live"), "skipped"),
    ],
)
def test_status_words(result, word):
    assert report.result_status(result) == word
