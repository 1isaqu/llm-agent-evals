import pytest

from llm_agent_evals.errors import ConfigError
from llm_agent_evals.settings import AgentSettings, LLMSettings


def test_defaults_select_the_offline_mock_and_need_no_key():
    settings = LLMSettings.from_env({})
    assert settings.provider == "mock"
    assert not settings.is_real
    settings.require_ready()


def test_real_provider_needs_a_key_and_a_model():
    with pytest.raises(ConfigError, match="LLM_API_KEY"):
        LLMSettings.from_env({"LLM_PROVIDER": "anthropic", "LLM_MODEL": "m"}).require_ready()
    with pytest.raises(ConfigError, match="LLM_MODEL"):
        LLMSettings.from_env({"LLM_PROVIDER": "anthropic", "LLM_API_KEY": "k"}).require_ready()


def test_the_api_key_never_appears_in_repr_or_error_messages():
    secret = "sk-test-not-a-real-key"
    settings = LLMSettings.from_env({"LLM_PROVIDER": "openai_compatible", "LLM_API_KEY": secret})
    assert secret not in repr(settings)
    assert secret not in str(settings)
    with pytest.raises(ConfigError) as caught:
        settings.require_ready()
    assert secret not in str(caught.value)


def test_unknown_provider_is_rejected():
    with pytest.raises(ConfigError, match="LLM_PROVIDER"):
        LLMSettings.from_env({"LLM_PROVIDER": "something-else"})


def test_base_url_defaults_per_provider_and_strips_the_trailing_slash():
    assert LLMSettings.from_env({"LLM_PROVIDER": "anthropic"}).base_url.startswith("https://")
    custom = LLMSettings.from_env(
        {"LLM_PROVIDER": "openai_compatible", "LLM_BASE_URL": "http://localhost:9999/v1/"}
    )
    assert custom.base_url == "http://localhost:9999/v1"


def test_numbers_are_validated_and_temperature_can_be_omitted():
    settings = LLMSettings.from_env({"LLM_TEMPERATURE": "none", "LLM_MAX_TOKENS": "512"})
    assert settings.temperature is None and settings.max_tokens == 512
    with pytest.raises(ConfigError, match="LLM_MAX_TOKENS"):
        LLMSettings.from_env({"LLM_MAX_TOKENS": "0"})
    with pytest.raises(ConfigError, match="LLM_TIMEOUT_SECONDS"):
        LLMSettings.from_env({"LLM_TIMEOUT_SECONDS": "soon"})


def test_agent_settings_defaults_and_switches():
    settings = AgentSettings.from_env({})
    assert settings.max_steps == 8 and settings.chat_enabled
    assert settings.egress_policy == "redact"
    off = AgentSettings.from_env({"AGENT_CHAT_ENABLED": "off", "AGENT_EGRESS_POLICY": "block"})
    assert not off.chat_enabled and off.egress_policy == "block"


def test_agent_settings_reject_bad_values():
    with pytest.raises(ConfigError, match="AGENT_CHAT_ENABLED"):
        AgentSettings.from_env({"AGENT_CHAT_ENABLED": "maybe"})
    with pytest.raises(ConfigError, match="AGENT_EGRESS_POLICY"):
        AgentSettings.from_env({"AGENT_EGRESS_POLICY": "ignore"})
    with pytest.raises(ConfigError, match="AGENT_MAX_STEPS"):
        AgentSettings.from_env({"AGENT_MAX_STEPS": "0"})


def test_business_config_path_is_honoured(tmp_path):
    path = tmp_path / "business.json"
    path.write_text('{"grace_days": 9}', encoding="utf-8")
    settings = AgentSettings.from_env({"AGENT_BUSINESS_CONFIG": str(path)})
    assert settings.business_config().grace_days == 9
    assert AgentSettings().business_config().grace_days == 3
