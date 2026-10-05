"""Adapter for the Anthropic Messages API (native tool use).

The Messages API requires strict alternation of `user` and `assistant` roles. The neutral
conversation can hold several consecutive user-side items (tool results, then a nudge
asking for the final answer), so `to_wire` merges consecutive user-side items into one
`user` message, with tool results first. Duplicate consecutive roles cannot be produced.

For a final answer the request keeps its `tools` (a conversation that contains tool use
blocks must declare tools) and sets `tool_choice` to `none`. Not exercised against a live
provider in this repository.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from llm_agent_evals.errors import ModelError
from llm_agent_evals.llm.base import ModelRequest, ModelTurn, ToolCall
from llm_agent_evals.llm.http import post_with_retry
from llm_agent_evals.llm.transport import Transport, urllib_transport
from llm_agent_evals.settings import LLMSettings

API_VERSION = "2023-06-01"


def _merge(wire: list[dict[str, Any]], role: str, blocks: list[dict[str, Any]]) -> None:
    """Append blocks to the last wire message when it has the same role, else start one.

    The API requires every `tool_result` block of a user message to come before any other
    block, so after merging the tool results are moved to the front (order otherwise kept).
    """
    if wire and wire[-1]["role"] == role:
        wire[-1]["content"].extend(blocks)
    else:
        wire.append({"role": role, "content": list(blocks)})
    if role == "user":
        content = wire[-1]["content"]
        content.sort(key=lambda block: block["type"] != "tool_result")


def messages_to_wire(request: ModelRequest) -> list[dict[str, Any]]:
    wire: list[dict[str, Any]] = []
    for message in request.messages:
        if message.role == "assistant":
            blocks: list[dict[str, Any]] = []
            if message.content:
                blocks.append({"type": "text", "text": message.content})
            blocks.extend(
                {"type": "tool_use", "id": call.id, "name": call.name, "input": call.args}
                for call in message.tool_calls
            )
            _merge(wire, "assistant", blocks)
        elif message.role == "tool":
            block = {
                "type": "tool_result",
                "tool_use_id": message.tool_call_id,
                "content": message.content,
            }
            _merge(wire, "user", [block])
        else:
            _merge(wire, "user", [{"type": "text", "text": message.content}])
    return wire


def to_wire(request: ModelRequest, settings: LLMSettings) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": settings.model,
        "max_tokens": request.max_tokens,
        "system": request.system,
        "messages": messages_to_wire(request),
    }
    if request.temperature is not None:
        body["temperature"] = request.temperature
    if request.tools:
        body["tools"] = [
            {"name": s["name"], "description": s["description"], "input_schema": s["input_schema"]}
            for s in request.tools
        ]
        if not request.allow_tools:
            body["tool_choice"] = {"type": "none"}
    return body


def parse_response(payload: Any) -> ModelTurn:
    try:
        blocks = payload["content"]
    except (KeyError, TypeError) as exc:
        raise ModelError("unexpected response shape from the model provider") from exc
    texts: list[str] = []
    calls: list[ToolCall] = []
    for block in blocks:
        if block.get("type") == "text":
            texts.append(str(block.get("text") or ""))
        elif block.get("type") == "tool_use":
            raw_input = block.get("input")
            calls.append(
                ToolCall(
                    id=str(block.get("id") or ""),
                    name=str(block.get("name") or ""),
                    args=raw_input if isinstance(raw_input, dict) else {},
                    args_error=None
                    if isinstance(raw_input, dict)
                    else "input must be a JSON object",
                )
            )
    usage = payload.get("usage") or {}
    return ModelTurn(
        text="".join(texts),
        tool_calls=tuple(calls),
        stop_reason=str(payload.get("stop_reason") or ""),
        usage={
            "input": int(usage.get("input_tokens") or 0),
            "output": int(usage.get("output_tokens") or 0),
        },
    )


class AnthropicLLM:
    """`LLMClient` for the Anthropic Messages API."""

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
        url = f"{self._settings.base_url}/v1/messages"
        headers = {
            "x-api-key": self._settings.api_key,
            "anthropic-version": API_VERSION,
            "content-type": "application/json",
        }
        body = to_wire(request, self._settings)
        payload = post_with_retry(self._transport, url, headers, body, self._settings, self._sleep)
        return parse_response(payload)
