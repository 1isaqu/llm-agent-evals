import json

import pytest

from llm_agent_evals.business import BusinessConfig
from llm_agent_evals.errors import ConfigError
from llm_agent_evals.llm import RuleBasedLLM
from llm_agent_evals.runtime import build_agent, demo_sources, sqlserver_sources
from llm_agent_evals.session import Session, Window
from llm_agent_evals.sources.demo import DEMO_NOW

SESSION = Session("auto", Window.from_iso("2026-03-01", "2026-03-18"))


def test_the_default_agent_runs_on_the_demo_data_with_the_mock_model_and_a_pinned_clock():
    agent = build_agent({})
    assert isinstance(agent.llm, RuleBasedLLM) and agent.current_time() == DEMO_NOW
    result = agent.run([{"role": "user", "content": "Which portfolios are at high risk?"}], SESSION)
    assert [c.name for c in result.calls] == ["filter_portfolios_by_risk"]
    assert result.response["confidence"] == "high"


def test_a_real_provider_without_a_key_fails_before_anything_runs():
    with pytest.raises(ConfigError, match="LLM_API_KEY"):
        build_agent({"LLM_PROVIDER": "anthropic", "LLM_MODEL": "m"})


def test_an_unknown_backend_is_rejected():
    with pytest.raises(ConfigError, match="backend"):
        build_agent({}, backend="oracle")


def test_the_business_config_file_reaches_the_agent(tmp_path):
    path = tmp_path / "business.json"
    path.write_text(json.dumps({"max_window_days": 30}), encoding="utf-8")
    assert build_agent({"AGENT_BUSINESS_CONFIG": str(path)}).config.max_window_days == 30


def test_explicit_sources_and_clock_win_over_the_backend():
    from datetime import datetime

    from tests.helpers import base_sections, fixture_sources

    agent = build_agent(
        {}, sources=fixture_sources(base_sections()), now=lambda: datetime(2031, 5, 5)
    )
    assert agent.lines == ("auto",) and agent.current_time() == datetime(2031, 5, 5)


def test_model_settings_reach_the_agent():
    agent = build_agent({"LLM_MAX_TOKENS": "321", "LLM_TEMPERATURE": "0.9"})
    assert agent.max_tokens == 321 and agent.temperature == 0.9


def test_telemetry_goes_to_the_configured_file_without_arguments_or_data(tmp_path):
    path = tmp_path / "events.jsonl"
    agent = build_agent({"AGENT_TELEMETRY_PATH": str(path)})
    agent.run(
        [{"role": "user", "content": "Which agents are the top performers?"}], SESSION, run_id="r1"
    )
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert {line["event"] for line in lines} >= {"model_turn", "tool_call", "run_end"}
    assert all(line["run_id"] == "r1" for line in lines)
    text = path.read_text(encoding="utf-8")
    assert "top performers" not in text and "Agent 01" not in text


def test_an_unwritable_telemetry_path_never_breaks_a_request(tmp_path):
    agent = build_agent({"AGENT_TELEMETRY_PATH": str(tmp_path / "missing-dir" / "events.jsonl")})
    result = agent.run([{"role": "user", "content": "high risk portfolios"}], SESSION)
    assert result.response["confidence"] == "high"


def test_demo_sources_are_cached_per_line():
    sources = demo_sources(BusinessConfig())
    assert set(sources) == {"auto", "consumer"}
    window = Window.from_iso("2026-03-01", "2026-03-18")
    assert sources["auto"].status_rollup(window) == sources["auto"].status_rollup(window)


def test_sqlserver_sources_open_one_connection_per_line_through_an_injected_driver():
    opened = []
    env = {
        "DB_HOST": "db.example.invalid",
        "DB_NAME": "demo",
        "DB_USER": "u",
        "DB_PASSWORD": "change-me",
        "DB_NAME_CONSUMER": "demo_consumer",
    }
    sources = sqlserver_sources(
        env, BusinessConfig(), connect=lambda cs, **kw: opened.append(cs) or object()
    )
    assert set(sources) == {"auto", "consumer"}
    assert "DATABASE=demo;" in opened[0] and "DATABASE=demo_consumer;" in opened[1]
