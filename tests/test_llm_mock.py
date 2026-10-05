import json

import pytest

from llm_agent_evals.errors import ModelError
from llm_agent_evals.llm import Message, ModelRequest, RuleBasedLLM, ScriptedLLM, ScriptExhausted
from llm_agent_evals.llm.rulebased import route
from llm_agent_evals.prompt import FORCE_FINAL_NUDGE


def req(*messages, allow_tools=True):
    return ModelRequest(system="s", messages=tuple(messages), allow_tools=allow_tools)


def user(text):
    return Message("user", text)


# -- scripted model ------------------------------------------------------------------------


def test_the_script_is_replayed_in_order_and_requests_are_recorded():
    llm = ScriptedLLM(
        [
            {"tool_call": {"name": "get_x", "args": {"a": 1}}},
            {"final": {"text": "done", "confidence": "high"}},
        ]
    )
    first = llm.complete(req(user("q")))
    assert first.tool_calls[0].name == "get_x" and first.tool_calls[0].args == {"a": 1}
    second = llm.complete(req(user("q")))
    assert json.loads(second.text)["text"] == "done"
    assert len(llm.requests) == 2


def test_a_batch_turn_returns_several_calls_with_distinct_ids():
    llm = ScriptedLLM([{"tool_calls": [{"name": "a_tool"}, {"name": "b_tool", "args": {"x": 1}}]}])
    calls = llm.complete(req(user("q"))).tool_calls
    assert [c.name for c in calls] == ["a_tool", "b_tool"] and len({c.id for c in calls}) == 2


def test_raw_arguments_simulate_a_provider_that_sent_invalid_json():
    llm = ScriptedLLM([{"tool_call": {"name": "a_tool", "raw_args": "{oops"}}])
    (call,) = llm.complete(req(user("q"))).tool_calls
    assert call.args == {} and call.args_error


def test_text_fail_and_callable_turns():
    llm = ScriptedLLM(
        [
            {"text": "not json at all"},
            {"fail": "provider down"},
            lambda request: {"text": f"saw {len(request.messages)} message(s)"},
        ]
    )
    assert llm.complete(req(user("q"))).text == "not json at all"
    with pytest.raises(ModelError, match="provider down"):
        llm.complete(req(user("q")))
    assert llm.complete(req(user("a"), user("b"))).text == "saw 2 message(s)"


def test_running_past_the_script_is_a_harness_failure_not_a_model_failure():
    llm = ScriptedLLM([{"text": "only turn"}])
    llm.complete(req(user("q")))
    with pytest.raises(ScriptExhausted) as caught:
        llm.complete(req(user("q")))
    assert not isinstance(caught.value, ModelError)


def test_an_unrecognised_turn_is_rejected():
    with pytest.raises(ValueError, match="no recognised key"):
        ScriptedLLM([{"mystery": 1}]).complete(req(user("q")))


# -- rule-based model --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("question", "tool"),
    [
        ("how is conversion calculated?", "explain_metric"),
        ("How is the day going?", "get_daily_pace"),
        ("which portfolios are at high risk?", "filter_portfolios_by_risk"),
        ("portfolios on alert", "filter_portfolios_by_risk"),
        ("how many plans are settled by stage", "get_plan_stage_distribution"),
        ("what is the status breakdown", "get_status_breakdown"),
        ("who are the top agents", "list_agents_performance"),
        ("which are the largest portfolios", "filter_portfolios_by_value"),
    ],
)
def test_keywords_route_to_one_tool(question, tool):
    assert route(question.lower())[0] == tool


def test_routing_arguments_are_valid_for_the_schema():
    assert route("high risk portfolios") == ("filter_portfolios_by_risk", {"level": "high"})
    assert route("rank by conversion agents")[1]["order_by"] == "conversion_pct"
    assert route("how is conversion defined")[1] == {"term": "conversion"}


def test_an_unmatched_question_gets_a_low_confidence_answer_and_no_tool_call():
    llm = RuleBasedLLM()
    turn = llm.complete(req(user("what is the weather")))
    assert not turn.tool_calls
    assert json.loads(turn.text)["confidence"] == "low"


def test_after_the_tool_result_it_answers_with_values_taken_from_the_result():
    llm = RuleBasedLLM()
    first = llm.complete(req(user("which agents are top performers")))
    call = first.tool_calls[0]
    result = json.dumps([{"agent_name": "Agent 01", "agreement_value": 1234.5}])
    messages = (
        user("which agents are top performers"),
        Message("assistant", "", (call,)),
        Message("tool", result, tool_call_id=call.id),
    )
    answer = json.loads(llm.complete(req(*messages)).text)
    assert "Agent 01" in answer["text"] and "1234.5" in answer["text"]
    assert answer["data_sources"] == [call.name] and answer["confidence"] == "high"


def test_a_tool_error_is_reported_with_low_confidence():
    llm = RuleBasedLLM()
    call = llm.complete(req(user("high risk portfolios"))).tool_calls[0]
    error = json.dumps(
        {"ok": False, "error_type": "upstream_5xx", "user_facing": "The service is down."}
    )
    messages = (
        user("high risk portfolios"),
        Message("assistant", "", (call,)),
        Message("tool", error, tool_call_id=call.id),
    )
    answer = json.loads(llm.complete(req(*messages)).text)
    assert answer["text"] == "The service is down." and answer["confidence"] == "low"


def test_the_force_final_nudge_is_not_mistaken_for_the_question():
    llm = RuleBasedLLM()
    messages = (user("which agents are top performers"), user(FORCE_FINAL_NUDGE))
    turn = llm.complete(req(*messages, allow_tools=False))
    assert not turn.tool_calls


def test_the_same_conversation_always_produces_the_same_turns():
    one, two = RuleBasedLLM(), RuleBasedLLM()
    conversation = req(user("which portfolios are at high risk?"))
    assert one.complete(conversation) == two.complete(conversation)


def test_a_metric_explanation_quotes_the_formula_and_skips_long_prose_fields():
    llm = RuleBasedLLM()
    call = llm.complete(req(user("how is conversion calculated"))).tool_calls[0]
    result = json.dumps(
        {"term": "conversion", "type": "kpi", "formula": "a / b * 100", "note": "x" * 300}
    )
    messages = (
        user("how is conversion calculated"),
        Message("assistant", "", (call,)),
        Message("tool", result, tool_call_id=call.id),
    )
    text = json.loads(llm.complete(req(*messages)).text)["text"]
    assert text == "conversion: a / b * 100"


def test_a_dict_result_without_a_formula_is_summarised_by_its_short_scalar_fields():
    llm = RuleBasedLLM()
    call = llm.complete(req(user("what is the status breakdown"))).tool_calls[0]
    result = json.dumps(
        {"generated_amount": 700.0, "note": "n" * 200, "ok_flag": True, "rows": [1]}
    )
    messages = (
        user("what is the status breakdown"),
        Message("assistant", "", (call,)),
        Message("tool", result, tool_call_id=call.id),
    )
    text = json.loads(llm.complete(req(*messages)).text)["text"]
    assert text == "Result: generated_amount 700.0."
