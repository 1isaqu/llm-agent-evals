"""A keyword-routed mock model: deterministic, offline, no script needed.

`ScriptedLLM` replays a fixed list of turns and is what the evaluation scenarios use. This
second mock answers any question without a script, so the CLI, the HTTP service and a
fresh clone work with no key and no network. It is not a language model and makes no
attempt to look like one:

* it reads the latest user question and picks at most one tool from fixed keyword rules;
* after the tool result it writes a short answer that only repeats values found in the
  result, so every number it states is traceable;
* a question that matches no rule gets a low-confidence answer that says so.

It never uses randomness, the clock or the network, so a given conversation always yields
the same turns.
"""

from __future__ import annotations

import json
from typing import Any

from llm_agent_evals.llm.base import Message, ModelRequest, ModelTurn, ToolCall
from llm_agent_evals.prompt import COMPACTION_NOTICE, FORCE_FINAL_NUDGE

_TERMS = (
    ("composite risk", "composite_risk"),
    ("answer rate", "answer_rate"),
    ("effectiveness", "effectiveness"),
    ("conversion", "conversion"),
    ("exception", "exception"),
    ("ticket", "avg_ticket"),
    ("generated", "generated"),
    ("pace", "daily_pace"),
    ("cpc", "cpc"),
)
_HEADLINE_KEYS = (
    "composite_risk_pct",
    "first_installment_value",
    "agreement_value",
    "conversion_pct",
    "generated_amount",
    "on_pace_pct",
    "amount",
)
_NAME_KEYS = ("portfolio_name", "agent_name", "label", "stage")
_NOTICE_PREFIX = COMPACTION_NOTICE.split("{", 1)[0]


def _question(messages: tuple[Message, ...]) -> str:
    for message in reversed(messages):
        plain = message.role == "user" and message.content != FORCE_FINAL_NUDGE
        if plain and not message.content.startswith(_NOTICE_PREFIX):
            return message.content.lower()
    return ""


def _term_in(question: str) -> str | None:
    return next((term for phrase, term in _TERMS if phrase in question), None)


def route(question: str) -> tuple[str, dict[str, Any]] | None:
    """The `(tool, args)` for a question, or None when no keyword rule matches."""
    term = _term_in(question)
    if term and any(w in question for w in ("how is", "define", "definition", "formula", "mean")):
        return "explain_metric", {"term": term}
    if any(w in question for w in ("pace", "day going", "projected close")):
        return "get_daily_pace", {}
    if "risk" in question or "alert" in question:
        level = "high" if "high" in question else "medium" if "alert" in question else "low"
        return "filter_portfolios_by_risk", {"level": level}
    if any(w in question for w in ("stage", "settled", "open plan")):
        return "get_plan_stage_distribution", {}
    if any(w in question for w in ("status", "breakdown", "pending", "rejected")):
        return "get_status_breakdown", {}
    if any(w in question for w in ("agent", "conversion", "ranking", "top performer")):
        order = "conversion_pct" if "conversion" in question else "agreement_value"
        return "list_agents_performance", {"order_by": order, "limit": 5}
    if any(w in question for w in ("portfolio", "largest", "biggest")):
        return "filter_portfolios_by_value", {"min_value": 0, "limit": 5}
    return None


def _scalars(item: dict[str, Any], limit: int = 3) -> str:
    keys = [k for k in _HEADLINE_KEYS if k in item][:limit]
    return ", ".join(f"{key} {item[key]}" for key in keys)


def _short_scalar(value: Any) -> bool:
    """Numbers and short strings: long prose fields are left out of the one-line summary."""
    if isinstance(value, bool):
        return False
    return isinstance(value, (int, float)) or (isinstance(value, str) and len(value) <= 40)


def _describe(result: Any) -> str:
    if isinstance(result, list):
        if not result:
            return "The tool returned no rows."
        head = result[0] if isinstance(result[0], dict) else {}
        name = next((str(head[k]) for k in _NAME_KEYS if k in head), "the first row")
        return f"The tool returned {len(result)} row(s). First: {name} ({_scalars(head)})."
    if isinstance(result, dict) and "formula" in result:
        return f"{result.get('term', 'metric')}: {result['formula']}"
    if isinstance(result, dict):
        fields = [f"{k} {v}" for k, v in result.items() if _short_scalar(v)]
        return "Result: " + ", ".join(fields[:4]) + "."
    return "The tool returned data."


def _final(question: str, tool_message: Message | None, tool_name: str) -> dict[str, Any]:
    if tool_message is None:
        return {
            "text": "This offline mock model only recognises a few keywords (risk, agents, "
            "status, plan stage, pace, metric definitions). Rephrase or use a real model.",
            "highlights": [],
            "suggested_actions": [{"label": "Portfolios at high risk", "prompt": "high risk?"}],
            "data_sources": [],
            "confidence": "low",
        }
    result = json.loads(tool_message.content)
    failed = isinstance(result, dict) and result.get("ok") is False
    text = result["user_facing"] if failed else _describe(result)
    return {
        "text": text,
        "highlights": [],
        "suggested_actions": [],
        "data_sources": [tool_name],
        "confidence": "low" if failed else "high",
    }


class RuleBasedLLM:
    """Deterministic `LLMClient`. See the module docstring."""

    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []

    def complete(self, request: ModelRequest) -> ModelTurn:
        self.requests.append(request)
        question = _question(request.messages)
        used = [c for m in request.messages for c in m.tool_calls]
        tool_message = next((m for m in reversed(request.messages) if m.role == "tool"), None)
        if request.allow_tools and not used:
            picked = route(question)
            if picked is not None:
                call = ToolCall(id="call_0", name=picked[0], args=picked[1])
                return ModelTurn(tool_calls=(call,), stop_reason="tool_use")
        name = used[-1].name if used else ""
        text = json.dumps(_final(question, tool_message, name), ensure_ascii=False)
        return ModelTurn(text=text, stop_reason="end_turn")
