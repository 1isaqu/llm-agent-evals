"""The agent loop, with a scripted model and fixture data. No network, no database."""

import json
from datetime import datetime

import pytest

from llm_agent_evals.agent import Agent, coerce_messages
from llm_agent_evals.errors import ConfigError, ModelError
from llm_agent_evals.guards import RunGuard
from llm_agent_evals.llm import Message, ModelTurn, ToolCall
from llm_agent_evals.privacy import REDACTED, EgressGuard
from llm_agent_evals.prompt import FORCE_FINAL_NUDGE
from llm_agent_evals.session import Session, Window
from llm_agent_evals.settings import AgentSettings
from llm_agent_evals.sources.fixture import FixtureSource
from tests.helpers import (
    BASE_STATUS,
    NOW,
    SESSION,
    ask,
    base_sections,
    call,
    final,
    make_agent,
)

REVEAL = "sentinel-question-text"


def tool_messages(request):
    return [m for m in request.messages if m.role == "tool"]


def test_a_tool_call_then_a_final_answer():
    agent = make_agent([call("get_status_breakdown"), final("Generated is 700.", sources=["x"])])
    result = ask(agent)
    assert [c.name for c in result.calls] == ["get_status_breakdown"]
    assert result.response["text"] == "Generated is 700."
    assert result.response["reference_date"] == "2026-03-18"
    assert result.rounds == 2 and not result.forced_final and result.degraded is None
    assert result.format_ok


def test_the_first_request_carries_the_prompt_the_session_context_and_every_tool():
    agent = make_agent([final()])
    ask(agent)
    request = agent.llm.requests[0]
    assert "## Session context" in request.system and "2026-03-01 to 2026-03-18" in request.system
    assert len(request.tools) == 15 and request.allow_tools
    assert request.messages[0] == Message("user", "How is it going?")


def test_the_tool_result_goes_back_to_the_model_as_json_matched_by_call_id():
    agent = make_agent([call("get_status_breakdown"), final()])
    ask(agent)
    second = agent.llm.requests[1]
    assistant, tool = second.messages[-2], second.messages[-1]
    assert assistant.role == "assistant" and assistant.tool_calls[0].name == "get_status_breakdown"
    assert tool.role == "tool" and tool.tool_call_id == assistant.tool_calls[0].id
    assert json.loads(tool.content)["generated_amount"] == 700.0


def test_conversation_history_is_passed_through():
    agent = make_agent([final()])
    history = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "reply"},
        {"role": "user", "content": "second"},
    ]
    agent.run(history, SESSION)
    assert [m.content for m in agent.llm.requests[0].messages] == ["first", "reply", "second"]


def test_coerce_messages_accepts_both_forms():
    messages = coerce_messages([Message("user", "a"), {"role": "assistant", "content": "b"}])
    assert [m.role for m in messages] == ["user", "assistant"]


def test_a_batch_of_calls_is_executed_and_answered_together():
    script = [
        {"tool_calls": [{"name": "get_status_breakdown"}, {"name": "get_plan_stage_distribution"}]},
        final(),
    ]
    agent = make_agent(script, base_sections(plan_stage_rollup=[]))
    result = ask(agent)
    assert len(result.calls) == 2
    assert len(tool_messages(agent.llm.requests[1])) == 2


def test_the_round_limit_forces_a_tool_free_final_round():
    script = [
        call("get_status_breakdown"),
        call("get_agent_performance", agent_name="agent_01"),
        final("Stopped."),
    ]
    agent = make_agent(script, settings=AgentSettings(max_rounds=2, max_steps=9))
    result = ask(agent)
    last = agent.llm.requests[-1]
    assert not last.allow_tools and last.messages[-1].content == FORCE_FINAL_NUDGE
    assert result.forced_final and result.response["text"] == "Stopped."


def test_a_model_that_keeps_calling_tools_on_the_final_round_gets_no_more_calls_run():
    script = [call("get_status_breakdown"), call("get_daily_pace")]
    result = ask(make_agent(script, settings=AgentSettings(max_rounds=1, max_steps=9)))
    assert len(result.calls) == 1 and result.forced_final
    assert result.response["confidence"] == "low" and not result.format_ok


def test_the_step_budget_forces_the_final_round_even_with_rounds_left():
    guard = RunGuard(max_steps=1, retry_base_delay_seconds=0.0)
    script = [
        call("get_status_breakdown"),
        call("get_agent_performance", agent_name="agent_01"),
        final("Limited."),
    ]
    agent = make_agent(script, guard=guard)
    result = ask(agent)
    assert result.forced_final and len(result.calls) == 1
    assert not agent.llm.requests[-1].allow_tools


def test_three_identical_calls_declare_a_loop_and_end_the_run():
    script = [call("get_status_breakdown")] * 3 + [final("Done.")]
    result = ask(make_agent(script))
    assert [c.record.cache_hit for c in result.calls] == [False, True, True]
    assert result.forced_final


def test_a_failing_base_dataset_degrades_the_answer_and_never_asks_the_model():
    events = []
    agent = make_agent(
        [],
        base_sections(status_rollup={"$fail": "timeout"}),
        observer=lambda e, p: events.append(e),
    )
    result = ask(agent)
    assert result.degraded == "dataset_unavailable"
    assert result.response["confidence"] == "low" and "not available" in result.response["text"]
    assert agent.llm.requests == [] and events == ["dataset_error"]


def test_a_missing_section_in_the_base_dataset_also_degrades():
    result = ask(make_agent([], {"agent_rollup": []}))
    assert result.degraded == "dataset_unavailable"


def test_a_model_failure_propagates_and_is_reported_to_the_observer():
    events = []
    agent = make_agent([{"fail": "provider down"}], observer=lambda e, p: events.append(e))
    with pytest.raises(ModelError):
        ask(agent)
    assert "model_error" in events


def test_an_unsafe_final_answer_is_replaced():
    unsafe = final("Run:\n```sql\nSELECT * FROM agreements\n```")
    result = ask(make_agent([unsafe]))
    assert result.degraded == "unsafe_answer" and "SELECT" not in result.response["text"]


def test_a_non_json_final_answer_degrades_to_low_confidence_and_is_marked():
    result = ask(make_agent([{"text": "Portfolio T-01 is fine."}]))
    assert not result.format_ok and result.response["confidence"] == "low"
    assert result.response["text"] == "Portfolio T-01 is fine."


def test_malformed_tool_arguments_cost_a_step_and_get_a_validation_error():
    script = [
        {"tool_call": {"name": "get_status_breakdown", "raw_args": "{oops"}},
        call("get_status_breakdown"),
        final(),
    ]
    result = ask(make_agent(script))
    assert [c.record.error_type for c in result.calls] == ["validation", None]
    assert result.calls[0].result["error_type"] == "validation"
    assert result.calls[1].record.step_index == 2


def test_an_unknown_product_line_is_rejected_up_front():
    with pytest.raises(ConfigError, match="unknown product line"):
        make_agent([final()]).run(
            [Message("user", "q")], Session("nowhere", Window.from_iso("2026-03-01", "2026-03-02"))
        )


def test_an_agent_needs_at_least_one_source():
    with pytest.raises(ConfigError):
        Agent(make_agent([]).llm, {})


def test_a_session_over_all_lines_merges_the_rows_of_every_source():
    consumer = {
        "status_rollup": [
            {"portfolio_name": "Portfolio U-01", "status_id": 10, "qty": 3, "amount": 300.0}
        ]
    }
    agent = make_agent([call("get_status_breakdown"), final()], consumer=consumer)
    session = Session("all", Window.from_iso("2026-03-01", "2026-03-18"))
    result = ask(agent, session=session)
    assert result.calls[0].result["generated_amount"] == 1000.0  # 700 + 300
    assert "all product lines" in agent.llm.requests[0].system


def test_the_clock_for_today_is_injected():
    agent = make_agent([final()], now=lambda: datetime(2030, 1, 15, 8, 0))
    ask(agent)
    assert "2030-01-15" in agent.llm.requests[0].system
    assert agent.current_time() == datetime(2030, 1, 15, 8, 0)


# -- observability ----------------------------------------------------------------------------


def test_telemetry_records_safe_fields_and_never_arguments_results_or_question_text():
    events = []
    script = [call("get_agent_performance", agent_name="agent_01"), final("Agent 01 did well.")]
    agent = make_agent(script, observer=lambda event, payload: events.append((event, payload)))
    ask(agent, question=REVEAL)
    dump = json.dumps(events, default=str)
    assert REVEAL not in dump and "agent_01" not in dump and "Agent 01" not in dump
    names = [e for e, _ in events]
    assert names.count("model_turn") == 2 and "tool_call" in names and names[-1] == "run_end"
    tool = next(p for e, p in events if e == "tool_call")
    assert set(tool) >= {
        "tool_name",
        "args_hash",
        "step_index",
        "latency_ms",
        "cache_hit",
        "truncated",
        "row_count",
        "error_type",
        "egress_findings",
    }
    run_end = events[-1][1]
    assert run_end["steps"] == 1 and run_end["confidence"] == "high"


def test_a_broken_observer_cannot_break_a_request():
    def boom(event, payload):
        raise RuntimeError("telemetry down")

    result = ask(make_agent([call("get_status_breakdown"), final("Fine.")], observer=boom))
    assert result.response["text"] == "Fine."


def test_token_usage_is_summed_over_the_turns():
    class Usage:
        def __init__(self):
            self.turns = [
                ModelTurn(
                    tool_calls=(ToolCall("c1", "get_status_breakdown", {}),),
                    usage={"input": 10, "output": 2},
                ),
                ModelTurn(
                    text=json.dumps({"text": "ok", "confidence": "high"}),
                    usage={"input": 30, "output": 5},
                ),
            ]
            self.requests = []

        def complete(self, request):
            self.requests.append(request)
            return self.turns.pop(0)

    agent = make_agent([])
    agent.llm = Usage()
    assert ask(agent).usage == {"input": 40, "output": 7}


# -- privacy ------------------------------------------------------------------------------------


def leaking_sections():
    return base_sections(
        agent_rollup=[
            {
                "agent_id": "agent_01",
                "agent_name": "Agent 01 (reach me at agent01@example.org)",
                "attempt_count": 10,
                "answer_count": 5,
                "cpc_count": 2,
                "agreement_count": 1,
                "agreement_value": 10.0,
                "first_installment_value": 5.0,
                "agreement_count_by_contract": 1,
                "slips_issued": 1,
                "slips_paid": 1,
                "exception_count": 0,
                "exception_value": 0.0,
            }
        ]
    )


def test_personal_looking_values_are_redacted_before_the_model_sees_them():
    agent = make_agent([call("list_agents_performance"), final()], leaking_sections())
    result = ask(agent)
    shown = tool_messages(agent.llm.requests[1])[0].content
    assert "example.org" not in shown and REDACTED in shown
    assert result.calls[0].record.egress_findings >= 1


def test_the_block_policy_withholds_the_whole_result():
    agent = make_agent(
        [call("list_agents_performance"), final()], leaking_sections(), egress=EgressGuard("block")
    )
    result = ask(agent)
    assert result.calls[0].result["error_type"] == "egress_blocked"
    shown = tool_messages(agent.llm.requests[1])[0].content
    assert "agent01" not in shown


# -- caching and compaction ----------------------------------------------------------------------


class CountingSource(FixtureSource):
    def __init__(self, sections):
        super().__init__(sections)
        self.calls = 0

    def agent_rollup(self, window):
        self.calls += 1
        return super().agent_rollup(window)


def test_a_shared_cache_serves_the_same_call_in_a_later_request():
    from llm_agent_evals.guards import SharedState

    source = CountingSource(base_sections())
    shared = SharedState()
    script = [call("get_agent_performance", agent_name="agent_01"), final()]
    first = Agent(make_agent(script).llm, {"auto": source}, shared=shared, now=lambda: NOW)
    first.run([Message("user", "q")], SESSION)
    second = Agent(make_agent(list(script)).llm, {"auto": source}, shared=shared, now=lambda: NOW)
    result = second.run([Message("user", "q")], SESSION)
    assert source.calls == 1 and result.calls[0].record.cache_hit


def test_the_conversation_is_compacted_when_the_tool_budget_nears_its_limit():
    guard = RunGuard(
        tool_budget_tokens=60, tool_result_token_cap=1000, retry_base_delay_seconds=0.0
    )
    script = [
        call("get_status_breakdown"),
        call("get_agent_performance", agent_name="agent_01"),
        call("get_agent_performance", agent_name="agent_02"),
        final(),
    ]
    agent = make_agent(script, guard=guard)
    result = ask(agent)
    assert result.compactions >= 1
    last = agent.llm.requests[-1]
    assert any("Automatic summary" in m.content for m in last.messages if m.role == "user")
    ids = {c.id for m in last.messages for c in m.tool_calls}
    assert ids == {m.tool_call_id for m in last.messages if m.role == "tool"}


def test_the_base_dataset_is_loaded_once_per_session_window():
    source = CountingSource(base_sections())
    agent = Agent(make_agent([final()]).llm, {"auto": source}, now=lambda: NOW)
    assert agent.run([Message("user", "q")], SESSION).degraded is None
    assert len(BASE_STATUS) == 5
