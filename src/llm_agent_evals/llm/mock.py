"""A deterministic scripted model for CI and for scenario replay.

The script is a list of turns. Each turn is a dict with exactly one of these keys:

    {"tool_call": {"name": ..., "args": {...}}}      one tool call
    {"tool_calls": [{"name": ..., "args": {...}}]}   a batch of calls in one turn
                                                     (a call with "raw_args" instead of "args"
                                                     simulates arguments that were not valid JSON)
    {"final": {...}}                                 the JSON answer (serialised for you)
    {"text": "..."}                                  raw text, for malformed-output tests
    {"fail": "message"}                              the provider call fails

or a callable `fn(request) -> turn dict`, for turns that depend on what the model saw.

The mock never uses randomness or the network. When the script is exhausted it raises
`ScriptExhausted`, which tells a scenario author that the loop asked for more turns than
the script provides. That is a harness failure, not a model failure, so it is not a
`ModelError`.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from llm_agent_evals.errors import AgentError, ModelError
from llm_agent_evals.llm.base import ModelRequest, ModelTurn, ToolCall

Turn = Mapping[str, Any] | Callable[[ModelRequest], Mapping[str, Any]]


class ScriptExhausted(AgentError):
    """The agent asked the scripted model for more turns than the script holds."""


class ScriptedLLM:
    """Replays a fixed script. `requests` records every request it received."""

    def __init__(self, turns: Sequence[Turn]) -> None:
        self._turns = list(turns)
        self.requests: list[ModelRequest] = []

    def complete(self, request: ModelRequest) -> ModelTurn:
        index = len(self.requests)
        self.requests.append(request)
        if index >= len(self._turns):
            raise ScriptExhausted(
                f"the agent requested model turn {index + 1} but the script has "
                f"{len(self._turns)} turn(s)"
            )
        turn = self._turns[index]
        spec = turn(request) if callable(turn) else turn
        return self._to_turn(spec, index)

    @staticmethod
    def _to_turn(spec: Mapping[str, Any], index: int) -> ModelTurn:
        if "fail" in spec:
            raise ModelError(str(spec["fail"]))
        if "final" in spec:
            return ModelTurn(
                text=json.dumps(spec["final"], ensure_ascii=False), stop_reason="end_turn"
            )
        if "text" in spec:
            return ModelTurn(text=str(spec["text"]), stop_reason="end_turn")
        if "tool_call" in spec:
            calls = [spec["tool_call"]]
        elif "tool_calls" in spec:
            calls = list(spec["tool_calls"])
        else:
            raise ValueError(f"script turn {index} has no recognised key: {sorted(spec)}")
        tool_calls = tuple(_tool_call(index, n, call) for n, call in enumerate(calls))
        return ModelTurn(tool_calls=tool_calls, stop_reason="tool_use")


def _tool_call(turn: int, position: int, spec: Mapping[str, Any]) -> ToolCall:
    call_id = f"call_{turn}_{position}"
    if "raw_args" in spec:
        return ToolCall(
            id=call_id,
            name=spec["name"],
            args={},
            args_error="arguments are not valid JSON",
        )
    return ToolCall(id=call_id, name=spec["name"], args=dict(spec.get("args", {})))
