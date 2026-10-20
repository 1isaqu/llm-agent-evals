"""The Langfuse observer, the observer fan-out and how the switch reaches the agent."""

import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

from llm_agent_evals.errors import ConfigError
from llm_agent_evals.runtime import build_agent
from llm_agent_evals.session import Session, Window
from llm_agent_evals.settings import AgentSettings
from llm_agent_evals.telemetry import LangfuseTelemetry, build_langfuse_observer, combine

SESSION = Session("auto", Window.from_iso("2026-03-01", "2026-03-18"))
KEYS = {"LANGFUSE_PUBLIC_KEY": "pk-lf-test", "LANGFUSE_SECRET_KEY": "sk-lf-test"}
TOOL_CALL = {
    "run_id": "r1",
    "tool_name": "get_status_breakdown",
    "args_hash": "abc",
    "latency_ms": 1.5,
    "cache_hit": False,
    "row_count": None,
    "error_type": None,
}


class FakeLangfuse:
    """Records the calls the observer makes, with the same surface as `langfuse.Langfuse`."""

    def __init__(self, **settings):
        self.settings = settings
        self.calls = []

    def create_trace_id(self, *, seed):
        return f"trace-of-{seed}"

    def start_observation(self, **kwargs):
        self.calls.append(("observation", kwargs))
        return SimpleNamespace(end=lambda: self.calls.append(("end", {})))

    def create_event(self, **kwargs):
        self.calls.append(("event", kwargs))


def install_fake_langfuse(monkeypatch):
    module = ModuleType("langfuse")
    holder = {}

    def factory(**settings):
        holder["client"] = FakeLangfuse(**settings)
        return holder["client"]

    module.Langfuse = factory
    monkeypatch.setitem(sys.modules, "langfuse", module)
    return holder


def test_a_tool_call_becomes_a_tool_observation_in_the_trace_of_its_run():
    client = FakeLangfuse()
    LangfuseTelemetry(client)("tool_call", TOOL_CALL)
    (kind, call), (end, _) = client.calls
    assert (kind, end) == ("observation", "end")
    assert call["name"] == "get_status_breakdown" and call["as_type"] == "tool"
    assert call["trace_context"] == {"trace_id": "trace-of-r1"}
    assert call["level"] is None and call["status_message"] is None
    assert call["metadata"]["args_hash"] == "abc" and "run_id" not in call["metadata"]


def test_other_events_become_point_events_and_errors_are_flagged():
    client = FakeLangfuse()
    observer = LangfuseTelemetry(client, run_id="fallback")
    observer("model_error", {"error_type": "ModelError"})
    ((kind, call),) = client.calls
    assert kind == "event" and call["name"] == "model_error"
    assert call["level"] == "ERROR" and call["status_message"] == "ModelError"
    assert call["trace_context"] == {"trace_id": "trace-of-fallback"}


def test_a_failed_tool_call_is_an_error_observation():
    client = FakeLangfuse()
    LangfuseTelemetry(client)("tool_call", {**TOOL_CALL, "error_type": "validation"})
    assert client.calls[0][1]["level"] == "ERROR"
    assert client.calls[0][1]["status_message"] == "validation"


def test_events_without_any_run_id_share_one_named_trace():
    client = FakeLangfuse()
    LangfuseTelemetry(client)("run_end", {"steps": 1})
    assert client.calls[0][1]["trace_context"] == {"trace_id": "trace-of-no-run-id"}


def test_only_scalar_fields_are_forwarded_so_data_cannot_ride_along():
    client = FakeLangfuse()
    payload = {"steps": 2, "rows": [{"agent": "x"}], "nested": {"a": 1}, "note": "ok"}
    LangfuseTelemetry(client)("run_end", payload)
    assert client.calls[0][1]["metadata"] == {"steps": 2, "note": "ok"}


def test_combine_returns_none_one_observer_or_a_fan_out_that_survives_a_failing_sink():
    seen = []

    def boom(event, payload):
        raise RuntimeError("sink down")

    def keep(event, payload):
        seen.append(event)

    assert combine() is None and combine(None, None) is None
    assert combine(None, keep) is keep
    combine(boom, keep)("run_end", {})
    assert seen == ["run_end"]


def test_the_observer_needs_both_keys():
    with pytest.raises(ConfigError, match="LANGFUSE_PUBLIC_KEY"):
        build_langfuse_observer({"LANGFUSE_PUBLIC_KEY": "pk-lf-test"})


def test_a_missing_package_is_a_clear_start_up_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "langfuse", None)
    with pytest.raises(ConfigError, match=r"llm-agent-evals\[langfuse\]"):
        build_langfuse_observer(KEYS)


def test_the_client_gets_the_keys_and_the_base_url(monkeypatch):
    holder = install_fake_langfuse(monkeypatch)
    build_langfuse_observer({**KEYS, "LANGFUSE_BASE_URL": "http://langfuse.example.invalid"})
    assert holder["client"].settings == {
        "public_key": "pk-lf-test",
        "secret_key": "sk-lf-test",
        "base_url": "http://langfuse.example.invalid",
    }
    build_langfuse_observer({**KEYS, "LANGFUSE_HOST": "http://old.example.invalid"})
    assert holder["client"].settings["base_url"] == "http://old.example.invalid"
    build_langfuse_observer(KEYS)
    assert holder["client"].settings["base_url"] is None


def test_the_switch_is_off_by_default_and_read_from_the_environment():
    assert AgentSettings.from_env({}).telemetry_langfuse is False
    assert AgentSettings.from_env({"AGENT_TELEMETRY_LANGFUSE": "true"}).telemetry_langfuse is True


def test_the_switch_without_keys_stops_the_agent_from_being_built():
    with pytest.raises(ConfigError, match="LANGFUSE_SECRET_KEY"):
        build_agent({"AGENT_TELEMETRY_LANGFUSE": "true", "LANGFUSE_PUBLIC_KEY": "pk-lf-test"})


def test_a_run_reaches_langfuse_without_the_question_or_any_data(monkeypatch):
    holder = install_fake_langfuse(monkeypatch)
    agent = build_agent({"AGENT_TELEMETRY_LANGFUSE": "true", **KEYS})
    question = "Which agents are the top performers?"
    agent.run([{"role": "user", "content": question}], SESSION, run_id="r7")
    calls = holder["client"].calls
    names = [call.get("name") for kind, call in calls if kind != "end"]
    assert "run_end" in names and "model_turn" in names
    assert any(call.get("as_type") == "tool" for kind, call in calls if kind == "observation")
    dump = json.dumps(calls, default=str)
    assert "top performers" not in dump and "Agent 01" not in dump
    assert all(
        call["trace_context"] == {"trace_id": "trace-of-r7"}
        for kind, call in calls
        if kind != "end"
    )


def test_the_jsonl_file_and_langfuse_can_run_together(monkeypatch, tmp_path):
    holder = install_fake_langfuse(monkeypatch)
    path = tmp_path / "events.jsonl"
    env = {"AGENT_TELEMETRY_LANGFUSE": "true", "AGENT_TELEMETRY_PATH": str(path), **KEYS}
    build_agent(env).run([{"role": "user", "content": "high risk portfolios"}], SESSION, "r8")
    assert "run_end" in path.read_text(encoding="utf-8")
    assert any(call.get("name") == "run_end" for _, call in holder["client"].calls)
