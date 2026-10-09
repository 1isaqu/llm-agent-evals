"""The assertion layers, one at a time, on hand-built tool calls."""

import pytest

from llm_agent_evals.agent import AgentResult, ExecutedCall
from llm_agent_evals.evals.checks import (
    allowed_numbers,
    check_answer_text,
    check_faithfulness,
    check_false_unavailability,
    check_ground_truth,
    check_outcome,
    check_params,
    check_sql_channel,
    check_tool_selection,
    extract_numbers,
    parse_number,
)
from llm_agent_evals.evals.scenario import Expectations
from llm_agent_evals.telemetry import ToolCallRecord


def executed(name="a_tool", args=None, result=None, *, error_type=None, row_count=1):
    record = ToolCallRecord(name, "h", 1, 0.0, False, False, row_count, error_type)
    return ExecutedCall(name, args or {}, result if result is not None else {}, record)


def ids(violations):
    return [v.id for v in violations]


# -- tool selection ------------------------------------------------------------


def test_tool_selection_flags_missing_forbidden_unexpected_and_excess_calls():
    calls = [executed("a_tool"), executed("b_tool")]
    assert check_tool_selection(calls, Expectations(tools_any_of=("a_tool", "z_tool"))) == []
    assert ids(check_tool_selection(calls, Expectations(tools_any_of=("z_tool",)))) == [
        "tool_selection:missing_expected_tool"
    ]
    assert ids(check_tool_selection(calls, Expectations(tools_all_of=("a_tool", "z_tool")))) == [
        "tool_selection:missing_expected_tool"
    ]
    assert ids(check_tool_selection(calls, Expectations(tools_forbidden=("b_tool",)))) == [
        "tool_selection:forbidden_tool"
    ]
    assert ids(check_tool_selection(calls, Expectations(zero_tool_calls=True))) == [
        "tool_selection:unexpected_tool_calls"
    ]
    assert ids(check_tool_selection(calls, Expectations(max_tool_calls=1))) == [
        "tool_selection:too_many_tool_calls"
    ]


def test_no_expectations_means_no_violations():
    assert check_tool_selection([executed()], Expectations()) == []


# -- SQL channel ---------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "x'; DROP TABLE agreements; --",
        "SELECT * FROM agreements",
        "1 UNION SELECT password",
        "DELETE FROM agreements",
        "INSERT INTO agreements VALUES (1)",
        "UPDATE agreements SET a = 1",
    ],
)
def test_sql_shaped_arguments_are_flagged_wherever_they_are_nested(text):
    call = executed("a_tool", {"filters": [{"name": text}]})
    assert ids(check_sql_channel([call], {"a_tool"})) == ["sql_channel:sql_in_arguments"]


@pytest.mark.parametrize(
    "text", ["Portfolio A-01", "select the best option", "update schedule", "drop-off rate"]
)
def test_ordinary_words_that_resemble_sql_keywords_are_not_flagged(text):
    assert check_sql_channel([executed("a_tool", {"q": text})], {"a_tool"}) == []


def test_a_tool_outside_the_registry_is_flagged():
    assert ids(check_sql_channel([executed("run_sql")], {"a_tool"})) == [
        "sql_channel:tool_not_in_allowlist"
    ]


# -- params --------------------------------------------------------------------


def test_params_need_one_call_that_matches_every_condition():
    calls = [executed("a_tool", {"x": 1, "y": 2}), executed("b_tool", {"x": 1})]
    assert check_params(calls, [{"tool": "a_tool", "contains": {"x": 1, "y": 2}}]) == []
    assert ids(check_params(calls, [{"tool": "a_tool", "contains": {"x": 1, "y": 3}}])) == [
        "params:missing_params"
    ]
    assert ids(check_params(calls, [{"tool": "b_tool", "contains": {"y": 2}}])) == [
        "params:missing_params"
    ]


def test_param_values_compare_case_insensitively_for_strings_and_exactly_otherwise():
    calls = [executed("a_tool", {"name": "Portfolio A-01", "n": 5})]
    assert check_params(calls, [{"contains": {"name": "portfolio a-01"}}]) == []
    assert check_params(calls, [{"contains": {"n": "5"}}])


def test_absent_keys_must_not_be_present_in_the_matching_call():
    calls = [executed("a_tool", {"x": 1, "portfolio": "P"})]
    assert check_params(calls, [{"tool": "a_tool", "contains": {"x": 1}, "absent": ["portfolio"]}])
    assert (
        check_params(calls, [{"tool": "a_tool", "contains": {"x": 1}, "absent": ["other"]}]) == []
    )


# -- faithfulness --------------------------------------------------------------


@pytest.mark.parametrize(
    ("token", "value"),
    [
        ("1,234.50", 1234.5),
        ("23.75%", 23.75),
        ("-31.11%", -31.11),
        ("2.35M", 2.35e6),
        ("12k", 12000.0),
        ("500", 500.0),
        ("1,000", 1000.0),
    ],
)
def test_number_parsing(token, value):
    assert parse_number(token) == pytest.approx(value)


def test_extraction_ignores_dates_identifiers_and_currency_symbols():
    text = "On 2026-03-18 Portfolio A-12 and agent_05 reached $1,234.50 (23.75%)."
    assert [t for t, _ in extract_numbers(text)] == ["12", "1,234.50", "23.75%"]


def test_a_number_that_a_tool_returned_is_traceable_in_any_common_format():
    allowed = [1234.5, 23.75, 2_350_000.0]
    assert check_faithfulness("Value $1,234.50 at 23.75%, about $2.35M.", allowed) == []


def test_an_invented_number_is_flagged_with_its_token():
    violations = check_faithfulness("Risk is 37.5%.", [25.0, 4000.0])
    assert ids(violations) == ["faithfulness:untraceable_number"]
    assert violations[0].details["token"] == "37.5%"


def test_small_numbers_are_not_checked_because_they_cannot_be_told_apart_from_counts():
    assert check_faithfulness("There are 8 agreements across 3 portfolios.", []) == []


def test_a_sum_of_two_seen_numbers_is_accepted_only_for_values_of_at_least_one_hundred():
    assert check_faithfulness("Total $5,000.00.", [3000.0, 2000.0]) == []
    assert check_faithfulness("Total 55.", [30.0, 25.0])  # small: must match a single number
    assert check_faithfulness("Total 150.", [100.0, 50.0]) == []


def test_numbers_inside_strings_of_a_result_and_in_the_question_are_allowed():
    call = executed("a_tool", result={"portfolio_name": "Portfolio A-12", "value": 7.0})
    allowed = allowed_numbers([call], "How is Portfolio Z-99 doing?")
    assert check_faithfulness("Portfolio A-12 and Portfolio Z-99.", allowed) == []


def test_booleans_are_not_numbers():
    call = executed("a_tool", result={"anomaly": True, "n": 40})
    assert allowed_numbers([call], "") == [40.0]


# -- ground truth --------------------------------------------------------------


def test_ground_truth_reads_dotted_paths_through_dicts_and_lists():
    call = executed("a_tool", result={"data": [{"v": 10.0}, {"v": 20.0}], "meta": {"n": 2}})
    spec = lambda path, equals, **kw: {"tool": "a_tool", "path": path, "equals": equals, **kw}  # noqa: E731
    assert check_ground_truth([call], [spec("data.1.v", 20.0), spec("meta.n", 2)]) == []
    assert ids(check_ground_truth([call], [spec("data.1.v", 21.0)])) == [
        "ground_truth:value_mismatch"
    ]
    assert check_ground_truth([call], [spec("data.1.v", 20.05, tolerance_pct=1.0)]) == []


@pytest.mark.parametrize("path", ["data.5.v", "nope", "data.x", "meta.n.deeper"])
def test_ground_truth_reports_an_unreachable_value(path):
    call = executed("a_tool", result={"data": [{"v": 1}], "meta": {"n": 2}})
    assert ids(check_ground_truth([call], [{"tool": "a_tool", "path": path, "equals": 1}])) == [
        "ground_truth:missing_value"
    ]


def test_ground_truth_can_target_the_nth_call_and_booleans_count_as_zero_or_one():
    calls = [
        executed("a_tool", result={"v": 1, "flag": False}),
        executed("a_tool", result={"v": 2, "flag": True}),
    ]
    assert (
        check_ground_truth(calls, [{"tool": "a_tool", "path": "v", "equals": 2, "occurrence": 2}])
        == []
    )
    assert (
        check_ground_truth(
            calls, [{"tool": "a_tool", "path": "flag", "equals": 1, "occurrence": 2}]
        )
        == []
    )
    assert ids(
        check_ground_truth(calls, [{"tool": "a_tool", "path": "v", "equals": 1, "occurrence": 3}])
    ) == ["ground_truth:missing_value"]


# -- answer text ---------------------------------------------------------------


def test_text_rules_forbidden_required_alternatives_and_dates():
    text = "ALERT: Portfolio A-09 looks corrupt on 2026-03-17."
    assert (
        check_answer_text(text, {"contains": ["alert", "corrupt"], "contains_date": "2026-03-17"})
        == []
    )
    assert ids(check_answer_text(text, {"forbidden": ["Corrupt"]})) == [
        "answer_text:forbidden_text"
    ]
    assert ids(check_answer_text(text, {"contains": ["missing"]})) == ["answer_text:missing_text"]
    assert ids(check_answer_text(text, {"contains_any_of": ["x", "y"]})) == [
        "answer_text:missing_text"
    ]
    assert check_answer_text(text, {"contains_any_of": ["x", "ALERT"]}) == []
    assert ids(check_answer_text(text, {"contains_date": "2026-03-18"})) == [
        "answer_text:missing_date"
    ]


# -- false unavailability ------------------------------------------------------


def test_claiming_unavailability_after_a_call_returned_rows_is_flagged():
    calls = [executed(row_count=3)]
    assert ids(check_false_unavailability("Data is not available.", calls, ["not available"])) == [
        "false_unavailability:claims_unavailable_after_data"
    ]


def test_the_claim_is_fine_when_every_call_failed_or_returned_nothing():
    failed = [executed(error_type="upstream_5xx", row_count=None), executed(row_count=0)]
    assert check_false_unavailability("Data is not available.", failed, ["not available"]) == []
    assert check_false_unavailability("All good.", [executed(row_count=3)], ["not available"]) == []


# -- outcome -------------------------------------------------------------------


def result(confidence="high", **kw):
    response = {"text": "t", "confidence": confidence}
    return AgentResult(response=response, **kw)


def test_outcome_checks_confidence_degradation_forced_final_and_observed_errors():
    assert check_outcome(result(), Expectations()) == []
    assert ids(check_outcome(result("high"), Expectations(confidence=("low", "medium")))) == [
        "outcome:confidence_mismatch"
    ]
    assert ids(check_outcome(result(degraded="unsafe_answer"), Expectations())) == [
        "outcome:degraded_mismatch"
    ]
    assert (
        check_outcome(result(degraded="unsafe_answer"), Expectations(degraded="unsafe_answer"))
        == []
    )
    assert ids(check_outcome(result(forced_final=True), Expectations(forced_final=False))) == [
        "outcome:forced_final_mismatch"
    ]
    assert ids(check_outcome(result(), Expectations(tool_errors=("not_found",)))) == [
        "outcome:tool_error_not_observed"
    ]
    seen = result(calls=(executed(error_type="not_found"),))
    assert check_outcome(seen, Expectations(tool_errors=("not_found",))) == []


def test_a_non_json_answer_is_a_violation_unless_the_scenario_expects_it():
    assert ids(check_outcome(result(format_ok=False), Expectations())) == ["outcome:not_json"]
    assert check_outcome(result(format_ok=False), Expectations(format_ok=False)) == []
