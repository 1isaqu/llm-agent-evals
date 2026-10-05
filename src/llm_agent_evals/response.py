"""The answer contract and the parser that enforces it.

The model is asked for one JSON object (see the system prompt). Models still break the
format: they wrap it in code fences, write a raw line break inside a string, add prose
around it, or return something that is not an object at all. `parse_final_text` never raises
and never lets a format failure reach the user as an error. It extracts what it can,
coerces the fields to the contract and degrades to `confidence: "low"` when nothing usable
was found.

`degrade_unsafe` replaces an answer whose text carries SQL, a raw tool envelope or a stack
trace. The refused content is never echoed.
"""

from __future__ import annotations

import json
from typing import Any

from llm_agent_evals.errors import sanitize_final_text

CONFIDENCE_LEVELS = ("high", "medium", "low")
HIGHLIGHT_TYPES = ("anomaly", "metric", "portfolio")
MAX_HIGHLIGHTS = 4
MAX_ACTIONS = 3

NO_ANSWER_TEXT = "No answer could be produced for this question."
DATA_UNAVAILABLE_TEXT = "Data is not available for this query right now."
REFUSED_TEXT = (
    "This answer cannot be shown as it came back: it contained content that is not allowed "
    "(SQL, raw data or an internal error). Rephrase the question or try again."
)


def empty_response(text: str, confidence: str = "low") -> dict[str, Any]:
    return {
        "text": text,
        "highlights": [],
        "suggested_actions": [],
        "data_sources": [],
        "confidence": confidence,
    }


def _strip_fence(raw: str) -> str:
    candidate = raw.strip()
    if candidate.startswith("```"):
        candidate = candidate.strip("`")
        if candidate.lower().startswith("json"):
            candidate = candidate[4:]
    return candidate.strip()


def extract_json_object(raw: str) -> Any:
    """Parse the first JSON object found in `raw`, or return None.

    Tolerates code fences, prose before and after the object and literal control
    characters inside strings (`strict=False` accepts everything strict parsing accepts).
    """
    candidate = _strip_fence(raw or "")
    if not candidate.startswith("{"):
        start, end = candidate.find("{"), candidate.rfind("}")
        if start == -1 or end <= start:
            return None
        candidate = candidate[start : end + 1]
    try:
        return json.loads(candidate, strict=False)
    except (ValueError, TypeError):
        return None


def _highlights(items: Any) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label") or "").strip()
        if not label:
            continue
        kind = str(item.get("type") or "").strip().lower()
        highlight = {"type": kind if kind in HIGHLIGHT_TYPES else "metric", "label": label}
        if item.get("value") is not None:
            highlight["value"] = str(item["value"])
        out.append(highlight)
    return out[:MAX_HIGHLIGHTS]


def _actions(items: Any) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label") or "").strip()
        if not label:
            continue
        action = {"label": label}
        prompt = str(item.get("prompt") or "").strip()
        if prompt:
            action["prompt"] = prompt
        out.append(action)
    return out[:MAX_ACTIONS]


def normalize_response(payload: Any, fallback_text: str) -> dict[str, Any]:
    """Coerce a parsed payload to the answer contract. Unusable input becomes low confidence."""
    fallback = fallback_text.strip() or NO_ANSWER_TEXT
    if not isinstance(payload, dict):
        return empty_response(fallback)
    sources = payload.get("data_sources")
    confidence = str(payload.get("confidence") or "").strip().lower()
    return {
        "text": str(payload.get("text") or "").strip() or fallback,
        "highlights": _highlights(payload.get("highlights")),
        "suggested_actions": _actions(payload.get("suggested_actions")),
        "data_sources": [str(s) for s in sources if str(s).strip()]
        if isinstance(sources, list)
        else [],
        "confidence": confidence if confidence in CONFIDENCE_LEVELS else "low",
    }


def parse_final_text(final_text: str) -> tuple[dict[str, Any], bool]:
    """Parse the model's final text. Returns (response, format_ok).

    `format_ok` is False when no JSON object could be read from the text, which the
    evaluation harness reports as a format violation even though the user still gets an
    answer.
    """
    payload = extract_json_object(final_text)
    return normalize_response(payload, fallback_text=final_text), isinstance(payload, dict)


def degrade_unsafe(response: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    """Return (response, was_refused). An unsafe text is replaced, never echoed."""
    clean = sanitize_final_text(response["text"])
    if clean is None:
        refused = empty_response(REFUSED_TEXT)
        refused["data_sources"] = list(response.get("data_sources", []))
        return refused, True
    return {**response, "text": clean}, False
