"""Adapter for OpenAI-style chat completion endpoints (function calling).

Covers any server that speaks the chat-completions protocol: set `LLM_BASE_URL` to its
`/v1` root. Tool calls come back as `message.tool_calls[*].function` with the arguments
serialised as a JSON string. Not exercised against a live provider in this repository.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from typing import Any

from llm_agent_evals.errors import ModelError
from llm_agent_evals.llm.base import Message, ModelRequest, ModelTurn, ToolCall
from llm_agent_evals.llm.http import post_with_retry
from llm_agent_evals.llm.transport import Transport, urllib_transport
from llm_agent_evals.settings import LLMSettings


def _message_to_wire(message: Message) -> dict[str, Any]:
    if message.role == "tool":
        return {"role": "tool", "tool_call_id": message.tool_call_id, "content": message.content}
    if message.role == "assistant":
        wire: dict[str, Any] = {"role": "assistant", "content": message.content or None}
        if message.tool_calls:
            wire["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": json.dumps(call.args)},
                }
                for call in message.tool_calls
            ]
        return wire
    return {"role": "user", "content": message.content}


def to_wire(request: ModelRequest, settings: LLMSettings) -> dict[str, Any]:
    """Translate a neutral request into a chat-completions request body.

    For a final answer (`allow_tools=False`) the `tools` field is omitted, which is the
    most widely supported way to stop an OpenAI-style server from calling tools.
    """
    messages = [{"role": "system", "content": request.system}]
    messages.extend(_message_to_wire(m) for m in request.messages)
    body: dict[str, Any] = {
        "model": settings.model,
        "messages": messages,
        "max_tokens": request.max_tokens,
    }
    if request.temperature is not None:
        body["temperature"] = request.temperature
    if request.tools and request.allow_tools:
        body["tools"] = [
            {
                "type": "function",
                "function": {
                    "name": spec["name"],
                    "description": spec["description"],
                    "parameters": spec["input_schema"],
                },
            }
            for spec in request.tools
        ]
    return body


def _parse_call(raw: dict[str, Any]) -> ToolCall:
    function = raw.get("function") or {}
    arguments = function.get("arguments") or "{}"
    args: dict[str, Any] = {}
    error: str | None = None
    try:
        parsed = json.loads(arguments) if isinstance(arguments, str) else arguments
        if isinstance(parsed, dict):
            args = parsed
        else:
            error = "arguments must be a JSON object"
    except ValueError as exc:
        error = f"arguments are not valid JSON: {exc}"
    return ToolCall(
        id=str(raw.get("id") or ""),
        name=str(function.get("name") or ""),
        args=args,
        args_error=error,
    )


def parse_response(payload: Any) -> ModelTurn:
    """Translate a chat-completions response into a neutral turn."""
    try:
        choice = payload["choices"][0]
        message = choice["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ModelError("unexpected response shape from the model provider") from exc
    calls = tuple(_parse_call(raw) for raw in message.get("tool_calls") or [])
    usage = payload.get("usage") or {}
    return ModelTurn(
        text=str(message.get("content") or ""),
        tool_calls=calls,
        stop_reason=str(choice.get("finish_reason") or ""),
        usage={
            "input": int(usage.get("prompt_tokens") or 0),
            "output": int(usage.get("completion_tokens") or 0),
        },
    )


class OpenAICompatibleLLM:
    """`LLMClient` for chat-completions servers."""

    def __init__(
        self,
        settings: LLMSettings,
        transport: Transport = urllib_transport,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._settings = settings
        self._transport = transport
        self._sleep = sleep

    def complete(self, request: ModelRequest) -> ModelTurn:
        url = f"{self._settings.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self._settings.api_key}",
            "Content-Type": "application/json",
        }
        body = to_wire(request, self._settings)
        payload = post_with_retry(self._transport, url, headers, body, self._settings, self._sleep)
        return parse_response(payload)
