"""Runtime settings read from environment variables.

Nothing is read from files in the repository and no secret has a default. The API
key exists only as the `LLM_API_KEY` environment variable and is never included in
`repr`, logs or error messages.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from llm_agent_evals.business import BusinessConfig
from llm_agent_evals.errors import ConfigError

PROVIDERS = ("mock", "openai_compatible", "anthropic")
EGRESS_POLICIES = ("redact", "block")
_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}

_DEFAULT_BASE_URLS = {
    "openai_compatible": "https://api.openai.com/v1",
    "anthropic": "https://api.anthropic.com",
}


def _read(env: Mapping[str, str], name: str, default: str = "") -> str:
    return (env.get(name) or default).strip()


def _read_int(env: Mapping[str, str], name: str, default: int, minimum: int = 0) -> int:
    raw = _read(env, name)
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer") from exc
    if value < minimum:
        raise ConfigError(f"{name} must be >= {minimum}")
    return value


def _read_float(env: Mapping[str, str], name: str, default: float) -> float:
    raw = _read(env, name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number") from exc


def _read_optional_float(env: Mapping[str, str], name: str, default: float | None) -> float | None:
    """A float, or None when the variable is set to none/off (omit the parameter)."""
    raw = _read(env, name)
    if not raw:
        return default
    if raw.lower() in ("none", "off"):
        return None
    return _read_float(env, name, 0.0)


def _read_bool(env: Mapping[str, str], name: str, default: bool) -> bool:
    raw = _read(env, name).lower()
    if not raw:
        return default
    if raw in _TRUE:
        return True
    if raw in _FALSE:
        return False
    raise ConfigError(f"{name} must be true or false")


@dataclass(frozen=True)
class LLMSettings:
    """Model provider settings. `api_key` is hidden from `repr`."""

    provider: str = "mock"
    api_key: str = ""
    base_url: str = ""
    model: str = ""
    timeout_seconds: float = 30.0
    max_retries: int = 1
    max_tokens: int = 2048
    temperature: float | None = 0.2

    def __repr__(self) -> str:
        key_state = "set" if self.api_key else "unset"
        return (
            f"LLMSettings(provider={self.provider!r}, api_key=<{key_state}>, "
            f"base_url={self.base_url!r}, model={self.model!r})"
        )

    @property
    def is_real(self) -> bool:
        return self.provider != "mock"

    def require_ready(self) -> None:
        """Raise ConfigError when a real provider is selected but not usable."""
        if not self.is_real:
            return
        if not self.api_key:
            raise ConfigError(f"LLM_API_KEY is not set (required for provider {self.provider!r})")
        if not self.model:
            raise ConfigError("LLM_MODEL is not set")

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> LLMSettings:
        source = os.environ if env is None else env
        provider = _read(source, "LLM_PROVIDER", "mock").lower()
        if provider not in PROVIDERS:
            raise ConfigError(f"LLM_PROVIDER must be one of: {', '.join(PROVIDERS)}")
        base_url = _read(source, "LLM_BASE_URL") or _DEFAULT_BASE_URLS.get(provider, "")
        return cls(
            provider=provider,
            api_key=_read(source, "LLM_API_KEY"),
            base_url=base_url.rstrip("/"),
            model=_read(source, "LLM_MODEL"),
            timeout_seconds=_read_float(source, "LLM_TIMEOUT_SECONDS", 30.0),
            max_retries=_read_int(source, "LLM_MAX_RETRIES", 1),
            max_tokens=_read_int(source, "LLM_MAX_TOKENS", 2048, minimum=1),
            temperature=_read_optional_float(source, "LLM_TEMPERATURE", 0.2),
        )


@dataclass(frozen=True)
class AgentSettings:
    """Agent runtime limits and switches."""

    max_rounds: int = 8
    max_steps: int = 8
    wall_clock_seconds: float = 60.0
    egress_policy: str = "redact"
    telemetry_path: str = ""
    telemetry_langfuse: bool = False
    chat_enabled: bool = True
    cache_ttl_seconds: float = 45.0
    business_config_path: str = ""

    def business_config(self) -> BusinessConfig:
        """The business config, from `AGENT_BUSINESS_CONFIG` when set, else defaults."""
        if self.business_config_path:
            return BusinessConfig.from_file(Path(self.business_config_path))
        return BusinessConfig()

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> AgentSettings:
        source = os.environ if env is None else env
        policy = _read(source, "AGENT_EGRESS_POLICY", "redact").lower()
        if policy not in EGRESS_POLICIES:
            raise ConfigError(f"AGENT_EGRESS_POLICY must be one of: {', '.join(EGRESS_POLICIES)}")
        return cls(
            max_rounds=_read_int(source, "AGENT_MAX_ROUNDS", 8, minimum=1),
            max_steps=_read_int(source, "AGENT_MAX_STEPS", 8, minimum=1),
            wall_clock_seconds=_read_float(source, "AGENT_WALL_CLOCK_SECONDS", 60.0),
            egress_policy=policy,
            telemetry_path=_read(source, "AGENT_TELEMETRY_PATH"),
            telemetry_langfuse=_read_bool(source, "AGENT_TELEMETRY_LANGFUSE", False),
            chat_enabled=_read_bool(source, "AGENT_CHAT_ENABLED", True),
            cache_ttl_seconds=_read_float(source, "AGENT_CACHE_TTL_SECONDS", 45.0),
            business_config_path=_read(source, "AGENT_BUSINESS_CONFIG"),
        )
