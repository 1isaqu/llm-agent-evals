"""Shared HTTP call logic for the wire adapters: retries and safe error messages."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from llm_agent_evals.errors import ModelError
from llm_agent_evals.llm.transport import Transport
from llm_agent_evals.settings import LLMSettings

_RETRYABLE_STATUS = {408, 409, 429}
_MAX_ERROR_CHARS = 200


def error_text(payload: Any) -> str:
    """A short, safe description of a provider error body.

    Only the provider's `error.message` (or a truncated string body) is used. Request
    headers, the request body and the API key are never part of an error message.
    """
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict) and error.get("message"):
            return str(error["message"])[:_MAX_ERROR_CHARS]
        if isinstance(error, str):
            return error[:_MAX_ERROR_CHARS]
    if isinstance(payload, str):
        return payload.strip()[:_MAX_ERROR_CHARS]
    return "no error detail"


def post_with_retry(
    transport: Transport,
    url: str,
    headers: dict[str, str],
    body: dict[str, Any],
    settings: LLMSettings,
    sleep: Callable[[float], None] = time.sleep,
) -> Any:
    """POST and return the parsed payload, retrying transient failures.

    Retried: network errors raised by the transport, HTTP 408, 409, 429 and 5xx, up to
    `settings.max_retries` extra attempts with exponential backoff (1s, 2s, 4s, capped
    at 8s). Anything else fails immediately with a `ModelError`.
    """
    attempts = settings.max_retries + 1
    for attempt in range(attempts):
        last = attempt == attempts - 1
        try:
            status, payload = transport(url, headers, body, settings.timeout_seconds)
        except ModelError:
            if last:
                raise
            sleep(min(2.0**attempt, 8.0))
            continue
        if status < 400:
            return payload
        transient = status in _RETRYABLE_STATUS or status >= 500
        if transient and not last:
            sleep(min(2.0**attempt, 8.0))
            continue
        raise ModelError(f"model provider returned HTTP {status}: {error_text(payload)}")
    raise ModelError("model call failed")  # pragma: no cover - the loop always returns or raises
