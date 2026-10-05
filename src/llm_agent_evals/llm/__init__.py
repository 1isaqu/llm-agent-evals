"""Model clients: the neutral types, the scripted mock and the HTTP adapters."""

from __future__ import annotations

import time
from collections.abc import Callable

from llm_agent_evals.errors import ConfigError
from llm_agent_evals.llm.anthropic import AnthropicLLM
from llm_agent_evals.llm.base import LLMClient, Message, ModelRequest, ModelTurn, ToolCall
from llm_agent_evals.llm.mock import ScriptedLLM, ScriptExhausted
from llm_agent_evals.llm.openai_compat import OpenAICompatibleLLM
from llm_agent_evals.llm.rulebased import RuleBasedLLM
from llm_agent_evals.llm.transport import Transport, urllib_transport
from llm_agent_evals.settings import LLMSettings


def build_real_llm(
    settings: LLMSettings,
    transport: Transport = urllib_transport,
    sleep: Callable[[float], None] = time.sleep,
) -> LLMClient:
    """Build the model client selected by `LLM_PROVIDER`.

    Raises `ConfigError` when the provider is `mock` (the mock needs a script, so it is
    constructed directly) or when the key or model is missing. The key is read only from
    the environment through `LLMSettings`.
    """
    if settings.provider == "mock":
        raise ConfigError("LLM_PROVIDER is 'mock': set a real provider to call a model")
    settings.require_ready()
    if settings.provider == "anthropic":
        return AnthropicLLM(settings, transport, sleep)
    return OpenAICompatibleLLM(settings, transport, sleep)


def build_llm(
    settings: LLMSettings,
    transport: Transport = urllib_transport,
    sleep: Callable[[float], None] = time.sleep,
) -> LLMClient:
    """The model client for `LLM_PROVIDER`: the offline rule-based mock, or a real adapter."""
    if settings.provider == "mock":
        return RuleBasedLLM()
    return build_real_llm(settings, transport, sleep)


__all__ = [
    "AnthropicLLM",
    "LLMClient",
    "Message",
    "ModelRequest",
    "ModelTurn",
    "OpenAICompatibleLLM",
    "RuleBasedLLM",
    "ScriptExhausted",
    "ScriptedLLM",
    "ToolCall",
    "build_llm",
    "build_real_llm",
]
