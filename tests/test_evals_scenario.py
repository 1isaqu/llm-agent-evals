import json
from pathlib import Path

import pytest

from llm_agent_evals.evals.scenario import (
    ScenarioError,
    load_scenario,
    load_scenarios,
    parse_scenario,
)

DATASET = {
    "auto": {"status_rollup": [{"portfolio_name": "P", "status_id": 10, "qty": 1, "amount": 5.0}]}
}


def raw(**overrides):
    body = {
        "id": "T1",
        "title": "A test scenario",
        "category": "factual",
        "question": "How is it going?",
        "session": {"product_line": "auto", "date_from": "2026-03-01", "date_to": "2026-03-18"},
        "now": "2026-03-18T15:30:00",
        "datasets": DATASET,
        "script": [],
        "expect": {},
    }
    body.update(overrides)
    return body


def parse(**overrides):
    return parse_scenario(raw(**overrides), Path("t1.json"))


def test_a_minimal_scenario_parses_with_defaults():
    scenario = parse()
    assert scenario.id == "T1" and scenario.question == "How is it going?"
    assert scenario.expect.should_pass and scenario.expect.format_ok
    assert scenario.session.product_line == "auto" and scenario.now.hour == 15
    assert scenario.expect.degraded is None and scenario.expect.forced_final is None


def test_messages_replace_the_question_for_multi_turn_scenarios():
    scenario = parse(
        messages=[
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
            {"role": "user", "content": "c"},
        ]
    )
    assert [m.role for m in scenario.messages] == ["user", "assistant", "user"]
    assert scenario.question == "c"


def test_expectations_are_converted_to_the_typed_form():
    scenario = parse(
        expect={
            "should_pass": False,
            "expect_violations": ["params:missing_params"],
            "tools_any_of": ["a_tool"],
            "confidence": "low",
            "max_tool_calls": 2,
            "params": [{"tool": "a_tool", "contains": {"x": 1}}],
            "ground_truth": [{"tool": "a_tool", "path": "v", "equals": 1}],
            "answer": {"forbidden": ["x"]},
        }
    )
    expect = scenario.expect
    assert not expect.should_pass and expect.expect_violations == ("params:missing_params",)
    assert expect.confidence == ("low",) and expect.max_tool_calls == 2
    assert expect.answer == {"forbidden": ["x"]}


@pytest.mark.parametrize(
    "overrides",
    [
        {"tools_forbiden": []},  # top level typo
        {"expect": {"tools_forbiden": ["a_tool"]}},  # expect typo
        {"expect": {"answer": {"forbiden": ["x"]}}},  # answer typo
        {"limits": {"max_step": 3}},  # limits typo
        {"expect": {"params": [{"tool": "a_tool"}]}},  # params without a condition
        {"expect": {"params": [{"contains": {"x": 1}, "equal": 1}]}},
        {"expect": {"ground_truth": [{"tool": "a_tool", "path": "v"}]}},
    ],
)
def test_typos_and_incomplete_specs_are_errors_and_never_silently_ignored(overrides):
    with pytest.raises(ScenarioError):
        parse(**overrides)


@pytest.mark.parametrize(
    "missing", ["id", "title", "category", "session", "now", "script", "expect"]
)
def test_required_fields_are_enforced(missing):
    body = raw()
    del body[missing]
    with pytest.raises(ScenarioError, match=f"missing '{missing}'"):
        parse_scenario(body, Path("t1.json"))


def test_question_or_messages_is_required():
    body = raw()
    del body["question"]
    with pytest.raises(ScenarioError, match="question"):
        parse_scenario(body, Path("t1.json"))


def test_a_bad_now_and_a_bad_session_are_reported():
    with pytest.raises(ScenarioError, match="invalid 'now'"):
        parse(now="soon")
    with pytest.raises(ScenarioError, match="invalid session"):
        parse(session={"product_line": "auto", "date_from": "x", "date_to": "y"})
    with pytest.raises(ScenarioError, match="invalid session"):
        parse(session={"product_line": "auto"})


SESSION = {"product_line": "auto", "date_from": "2026-03-01", "date_to": "2026-03-18"}


def test_an_unknown_session_key_is_an_error_that_names_the_file():
    with pytest.raises(ScenarioError, match=r"t1\.json: unknown key\(s\) in session: date_too"):
        parse(session={**SESSION, "date_too": "2026-03-19"})


def test_a_misspelled_dataset_section_is_an_error_that_names_the_file_and_the_section():
    typo = {"auto": {"agent_rolup": []}}
    with pytest.raises(ScenarioError, match=r"t1\.json: unknown dataset section.*agent_rolup"):
        parse(datasets=typo)


def test_a_misspelled_section_added_by_a_dataset_patch_is_caught_too(tmp_path):
    (tmp_path / "base.json").write_text(json.dumps(DATASET), encoding="utf-8")
    body = raw(dataset="base.json", dataset_patch={"auto": {"status_rolup": []}})
    del body["datasets"]
    with pytest.raises(ScenarioError, match="status_rolup"):
        parse_scenario(body, tmp_path / "t1.json")


def test_a_dataset_line_and_a_session_must_be_objects():
    with pytest.raises(ScenarioError, match="must be an object"):
        parse(datasets={"auto": []})
    with pytest.raises(ScenarioError, match="'session' must be an object"):
        parse(session=["auto"])


def test_a_bad_business_config_is_a_scenario_error_not_a_crash_at_run_time():
    with pytest.raises(ScenarioError, match=r"t1\.json: invalid config: .*grace_dayz"):
        parse(config={"grace_dayz": 3})
    with pytest.raises(ScenarioError, match="invalid config"):
        parse(config={"risk_low_max": 50.0, "risk_medium_max": 10.0})
    assert parse(config={"grace_days": 7}).config == {"grace_days": 7}


def test_the_session_line_must_have_data():
    with pytest.raises(ScenarioError, match="no data for line"):
        parse(
            session={"product_line": "consumer", "date_from": "2026-03-01", "date_to": "2026-03-18"}
        )
    parse(session={"product_line": "all", "date_from": "2026-03-01", "date_to": "2026-03-18"})


def write(directory: Path, name: str, body: dict) -> Path:
    path = directory / name
    path.write_text(json.dumps(body), encoding="utf-8")
    return path


def test_a_dataset_file_is_loaded_relative_to_the_scenario_and_can_be_patched(tmp_path):
    (tmp_path / "datasets").mkdir()
    write(
        tmp_path / "datasets",
        "base.json",
        {"auto": {"status_rollup": [{"a": 1}], "agent_rollup": [{"b": 2}]}},
    )
    body = raw(
        dataset="datasets/base.json",
        dataset_patch={"auto": {"status_rollup": [{"a": 9}]}, "consumer": {"status_rollup": []}},
    )
    del body["datasets"]
    scenario = load_scenario(write(tmp_path, "s.json", body))
    assert scenario.datasets["auto"]["status_rollup"] == [{"a": 9}]
    assert scenario.datasets["auto"]["agent_rollup"] == [{"b": 2}]
    assert "consumer" in scenario.datasets


def test_a_missing_or_malformed_file_is_a_scenario_error(tmp_path):
    with pytest.raises(ScenarioError, match="cannot read"):
        load_scenario(tmp_path / "nope.json")
    (tmp_path / "bad.json").write_text("{", encoding="utf-8")
    with pytest.raises(ScenarioError, match="cannot read"):
        load_scenario(tmp_path / "bad.json")
    (tmp_path / "list.json").write_text("[]", encoding="utf-8")
    with pytest.raises(ScenarioError, match="JSON object"):
        load_scenario(tmp_path / "list.json")


def test_a_directory_loads_in_file_order_and_rejects_duplicate_ids(tmp_path):
    write(tmp_path, "b.json", raw(id="B"))
    write(tmp_path, "a.json", raw(id="A"))
    assert [s.id for s in load_scenarios(tmp_path)] == ["A", "B"]
    write(tmp_path, "c.json", raw(id="A"))
    with pytest.raises(ScenarioError, match="duplicate scenario id"):
        load_scenarios(tmp_path)
    with pytest.raises(ScenarioError, match="not found"):
        load_scenarios(tmp_path / "missing")
