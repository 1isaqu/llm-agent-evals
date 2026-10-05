"""Provider-neutral conversation types and the model client protocol.

The agent loop speaks only these types. Each wire adapter translates them to one
provider's JSON format and back, so the loop, the compaction and the guards exist exactly
once instead of once per provider.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class ToolCall:
    """A tool call requested by the model.

    `args_error` is set when the provider sent arguments that were not valid JSON. The
    loop then answers with a validation error (which counts as a step) instead of
    silently running the tool with empty arguments.
    """

    id: str
    name: str
    args: dict[str, Any] = field(default_factory=dict)
    args_error: str | None = None


@dataclass(frozen=True)
class Message:
    """One conversation message. Roles: user, assistant, tool.

    An assistant message may carry `tool_calls`. A tool message carries the result of
    the call named by `tool_call_id` as JSON text in `content`.
    """

    role: str
    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str = ""


@dataclass(frozen=True)
class ModelRequest:
    """One request to the model. `tools` holds neutral specs (name, description, input_schema).

    `allow_tools=False` asks for a final answer: the adapter must stop the model from
    calling tools, in whatever way the provider supports.
    """

    system: str
    messages: tuple[Message, ...]
    tools: tuple[dict[str, Any], ...] = ()
    allow_tools: bool = True
    max_tokens: int = 2048
    temperature: float | None = 0.2


@dataclass(frozen=True)
class ModelTurn:
    """The model's reply: text, tool calls, or both."""

    text: str = ""
    tool_calls: tuple[ToolCall, ...] = ()
    stop_reason: str = ""
    usage: dict[str, int] = field(default_factory=dict)


class LLMClient(Protocol):
    """Anything that can answer a `ModelRequest`."""

    def complete(self, request: ModelRequest) -> ModelTurn: ...
