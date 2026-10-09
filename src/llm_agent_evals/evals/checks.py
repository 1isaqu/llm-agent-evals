"""The assertion layers. Every check returns structured `Violation` objects.

Each violation carries a `layer` and a `code`, and metrics are computed from those fields
alone. No metric is ever derived by searching a message for a word, so rewording a message
cannot change a score.

Layers, in the order they run:

    tool_selection        right tools, no forbidden tools, tool-call count
    sql_channel           only registered tools, and no SQL in any argument
    params                a call to the right tool carried the right arguments
    faithfulness          every number in the answer can be traced to a tool result
    ground_truth          a hand-computed value equals what the analytics code returned
    answer_text           required and forbidden wording, required date
    false_unavailability  the answer claims data is missing after a call returned rows
    outcome               confidence, degradation, forced final, observed tool errors, format
    harness               the run itself failed (script too short, provider failure)

The numeric check is a safety net, not a proof. It catches an invented number. It cannot
catch a real number attached to the wrong label, which is what `answer_text.forbidden` and
`ground_truth` are for.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from llm_agent_evals.agent import AgentResult, ExecutedCall
from llm_agent_evals.evals.scenario import Expectations, Scenario


@dataclass(frozen=True)
class Violation:
    """One failed check. `id` (`layer:code`) is what scenarios refer to."""

    layer: str
    code: str
    message: str
    details: Mapping[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> str:
        return f"{self.layer}:{self.code}"


# -- layer: tool selection ---------------------------------------------------


def check_tool_selection(calls: Sequence[ExecutedCall], expect: Expectations) -> list[Violation]:
    names = {call.name for call in calls}
    out: list[Violation] = []
    if expect.tools_any_of and not names & set(expect.tools_any_of):
        out.append(
            Violation(
                "tool_selection",
                "missing_expected_tool",
                f"none of the expected tools {sorted(expect.tools_any_of)} was called",
                {"called": sorted(names)},
            )
        )
    missing = sorted(set(expect.tools_all_of) - names)
    if missing:
        out.append(
            Violation("tool_selection", "missing_expected_tool", f"tools not called: {missing}", {})
        )
    for tool in expect.tools_forbidden:
        if tool in names:
            out.append(
                Violation("tool_selection", "forbidden_tool", f"forbidden tool {tool!r} called")
            )
    if expect.zero_tool_calls and calls:
        out.append(
            Violation(
                "tool_selection",
                "unexpected_tool_calls",
                f"expected no tool calls, saw {len(calls)}",
            )
        )
    if expect.max_tool_calls is not None and len(calls) > expect.max_tool_calls:
        out.append(
            Violation(
                "tool_selection",
                "too_many_tool_calls",
                f"{len(calls)} tool calls, at most {expect.max_tool_calls} allowed",
            )
        )
    return out


# -- layer: sql channel ------------------------------------------------------

_SQL_SHAPES = re.compile(
    r"\bselect\b.+\bfrom\b|\bdrop\s+table\b|\binsert\s+into\b|\bdelete\s+from\b"
    r"|\bupdate\s+\w+\s+set\b|\bunion\s+select\b|;\s*--|--\s*$",
    re.IGNORECASE | re.DOTALL,
)


def _strings(node: Any) -> Iterable[str]:
    if isinstance(node, str):
        yield node
    elif isinstance(node, Mapping):
        for value in node.values():
            yield from _strings(value)
    elif isinstance(node, (list, tuple)):
        for value in node:
            yield from _strings(value)


def check_sql_channel(calls: Sequence[ExecutedCall], allowed: Iterable[str]) -> list[Violation]:
    """The model may only call registered tools and never put SQL in an argument."""
    registered = set(allowed)
    out: list[Violation] = []
    for call in calls:
        if call.name not in registered:
            out.append(
                Violation("sql_channel", "tool_not_in_allowlist", f"unknown tool {call.name!r}")
            )
        if any(_SQL_SHAPES.search(text) for text in _strings(call.args)):
            out.append(
                Violation(
                    "sql_channel",
                    "sql_in_arguments",
                    f"SQL-shaped text in arguments of {call.name!r}",
                )
            )
    return out


# -- layer: params -----------------------------------------------------------


def _same_value(expected: Any, actual: Any) -> bool:
    if isinstance(expected, str) and isinstance(actual, str):
        return expected.strip().lower() == actual.strip().lower()
    return bool(expected == actual)


def _matches_spec(call: ExecutedCall, spec: Mapping[str, Any]) -> bool:
    if spec.get("tool") and call.name != spec["tool"]:
        return False
    contains = spec.get("contains", {})
    if not all(k in call.args and _same_value(v, call.args[k]) for k, v in contains.items()):
        return False
    return not any(key in call.args for key in spec.get("absent", ()))


def check_params(
    calls: Sequence[ExecutedCall], specs: Iterable[Mapping[str, Any]]
) -> list[Violation]:
    """Each spec needs one call to its tool whose arguments match `contains` and omit `absent`."""
    out: list[Violation] = []
    for spec in specs:
        if not any(_matches_spec(call, spec) for call in calls):
            out.append(
                Violation(
                    "params",
                    "missing_params",
                    f"no call matched the expected arguments {dict(spec)}",
                    {"spec": dict(spec)},
                )
            )
    return out


# -- layer: faithfulness -----------------------------------------------------

_ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
_NUMBER = re.compile(r"(?<![\w.])-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[kKmMbB]\b)?%?")
_SUFFIX = {"k": 1e3, "m": 1e6, "b": 1e9}
MIN_CHECKED_VALUE = 10.0
TOLERANCE = 0.01
MAX_PAIR_TERMS = 150
PAIR_SUM_MIN = 100.0


def parse_number(token: str) -> float | None:
    """`1,234.50`, `23.8%`, `$2.35M` (the `$` is not part of the token) to a float."""
    text = token.rstrip("%").replace(",", "")
    scale = 1.0
    if text and text[-1].lower() in _SUFFIX:
        scale, text = _SUFFIX[text[-1].lower()], text[:-1]
    try:
        return float(text) * scale
    except ValueError:
        return None


def extract_numbers(text: str) -> list[tuple[str, float]]:
    """Numbers written in `text`, ISO dates removed. Returns `(token, value)` pairs."""
    cleaned = _ISO_DATE.sub(" ", text)
    found = []
    for match in _NUMBER.finditer(cleaned):
        value = parse_number(match.group())
        if value is not None:
            found.append((match.group(), value))
    return found


def _flatten_numbers(node: Any, out: list[float]) -> None:
    if isinstance(node, bool):
        return
    if isinstance(node, (int, float)):
        out.append(float(node))
    elif isinstance(node, Mapping):
        for value in node.values():
            _flatten_numbers(value, out)
    elif isinstance(node, (list, tuple)):
        for value in node:
            _flatten_numbers(value, out)


def allowed_numbers(calls: Sequence[ExecutedCall], context_text: str) -> list[float]:
    """Numbers the model legitimately saw: tool results (also inside strings) and the question."""
    out: list[float] = []
    for call in calls:
        _flatten_numbers(call.result, out)
        out.extend(v for s in _strings(call.result) for _, v in extract_numbers(s))
    out.extend(v for _, v in extract_numbers(context_text))
    return out


def _is_close(value: float, target: float) -> bool:
    return abs(value - target) <= max(TOLERANCE * abs(target), 0.5)


def _traceable(value: float, allowed: list[float]) -> bool:
    """A seen number, or the sum of two seen numbers for values of at least `PAIR_SUM_MIN`.

    Small values never match a sum: among a few dozen small counts almost any small number
    is the sum of two of them, which would let an invented percentage through.
    """
    if any(_is_close(value, d) for d in allowed):
        return True
    if abs(value) < PAIR_SUM_MIN:
        return False
    terms = allowed[:MAX_PAIR_TERMS]
    return any(_is_close(value, a + b) for a in terms for b in terms)


def check_faithfulness(
    text: str, allowed: list[float], where: str = "the answer text"
) -> list[Violation]:
    """Every number of at least `MIN_CHECKED_VALUE` must match a seen number (or a sum of two).

    Smaller numbers are skipped: they are mostly ordinals, status ids and counts quoted in
    prose, and cannot be told apart from invented ones by magnitude.
    """
    out: list[Violation] = []
    for token, value in extract_numbers(text):
        if abs(value) < MIN_CHECKED_VALUE or _traceable(value, allowed):
            continue
        out.append(
            Violation(
                "faithfulness",
                "untraceable_number",
                f"{token!r} in {where} does not match any number in the tool results",
                {"token": token, "value": value, "where": where},
            )
        )
    return out


# -- layer: ground truth -----------------------------------------------------


def _resolve(node: Any, path: str) -> Any:
    for part in path.split("."):
        if isinstance(node, list):
            node = node[int(part)]
        elif isinstance(node, Mapping):
            node = node[part]
        else:
            raise KeyError(part)
    return node


def _nth_result(calls: Sequence[ExecutedCall], tool: str, occurrence: int) -> Any:
    matching = [call for call in calls if call.name == tool]
    return matching[occurrence - 1].result


def check_ground_truth(
    calls: Sequence[ExecutedCall], specs: Iterable[Mapping[str, Any]]
) -> list[Violation]:
    """Compare hand-computed values with what the analytics code returned to the model."""
    out: list[Violation] = []
    for spec in specs:
        tool, path = spec["tool"], spec["path"]
        try:
            actual = _resolve(_nth_result(calls, tool, int(spec.get("occurrence", 1))), path)
            actual_value = float(actual)
        except (IndexError, KeyError, TypeError, ValueError):
            out.append(
                Violation(
                    "ground_truth",
                    "missing_value",
                    f"{tool}:{path} is not available in the tool results",
                    {"spec": dict(spec)},
                )
            )
            continue
        expected = float(spec["equals"])
        tolerance = float(spec.get("tolerance_pct", 0.5)) / 100.0
        if abs(actual_value - expected) > max(abs(expected) * tolerance, 1e-9):
            out.append(
                Violation(
                    "ground_truth",
                    "value_mismatch",
                    f"{tool}:{path} is {actual_value}, expected {expected}",
                    {"actual": actual_value, "expected": expected},
                )
            )
    return out


# -- layer: answer text ------------------------------------------------------


def check_answer_text(text: str, answer: Mapping[str, Any]) -> list[Violation]:
    lowered = text.lower()
    out: list[Violation] = []
    for term in answer.get("forbidden", ()):
        if term.lower() in lowered:
            out.append(Violation("answer_text", "forbidden_text", f"answer contains {term!r}"))
    for term in answer.get("contains", ()):
        if term.lower() not in lowered:
            out.append(Violation("answer_text", "missing_text", f"answer lacks {term!r}"))
    options = answer.get("contains_any_of", ())
    if options and not any(term.lower() in lowered for term in options):
        out.append(Violation("answer_text", "missing_text", f"answer has none of {list(options)}"))
    day = answer.get("contains_date")
    if day and day not in text:
        out.append(Violation("answer_text", "missing_date", f"answer lacks the date {day}"))
    return out


# -- layer: false unavailability --------------------------------------------


def check_false_unavailability(
    text: str, calls: Sequence[ExecutedCall], phrases: Iterable[str]
) -> list[Violation]:
    """A call returned rows, and the answer still says the data is not available.

    It works from telemetry (`error_type` and `row_count` as the guard recorded them), so it
    measures what the tools returned, not what the script says it returned. It does not know
    which source a phrase refers to: it is a net for the obvious contradiction.
    """
    had_data = any(c.record.error_type is None and (c.record.row_count or 0) > 0 for c in calls)
    if not had_data:
        return []
    lowered = text.lower()
    return [
        Violation(
            "false_unavailability",
            "claims_unavailable_after_data",
            f"answer says {phrase!r} although a tool call returned rows",
        )
        for phrase in phrases
        if phrase.lower() in lowered
    ]


# -- layer: outcome ----------------------------------------------------------


def check_outcome(result: AgentResult, expect: Expectations) -> list[Violation]:
    out: list[Violation] = []
    confidence = result.response["confidence"]
    if expect.confidence and confidence not in expect.confidence:
        out.append(
            Violation(
                "outcome",
                "confidence_mismatch",
                f"confidence is {confidence!r}, expected one of {list(expect.confidence)}",
            )
        )
    if result.degraded != expect.degraded:
        out.append(
            Violation(
                "outcome",
                "degraded_mismatch",
                f"degraded is {result.degraded!r}, expected {expect.degraded!r}",
            )
        )
    if expect.forced_final is not None and result.forced_final != expect.forced_final:
        out.append(
            Violation(
                "outcome",
                "forced_final_mismatch",
                f"forced_final is {result.forced_final}, expected {expect.forced_final}",
            )
        )
    seen = {c.record.error_type for c in result.calls}
    for error_type in expect.tool_errors:
        if error_type not in seen:
            out.append(
                Violation(
                    "outcome", "tool_error_not_observed", f"no tool call ended in {error_type!r}"
                )
            )
    if expect.format_ok and not result.format_ok:
        out.append(Violation("outcome", "not_json", "the final answer was not a JSON object"))
    return out


def _check_highlight_values(response: Mapping[str, Any], allowed: list[float]) -> list[Violation]:
    """Highlight values are shown as headline figures, so they get the same numeric check."""
    out: list[Violation] = []
    for index, highlight in enumerate(response.get("highlights", [])):
        value = highlight.get("value")
        if isinstance(value, str):
            out.extend(check_faithfulness(value, allowed, f"highlights[{index}].value"))
    return out


def evaluate(
    scenario: Scenario, result: AgentResult, registered_tools: Iterable[str]
) -> list[Violation]:
    """Run every layer that the scenario's expectations switch on."""
    expect = scenario.expect
    text = result.response["text"]
    context = scenario.question + " " + " ".join(m.content for m in scenario.messages)
    violations = [
        *check_tool_selection(result.calls, expect),
        *check_sql_channel(result.calls, registered_tools),
        *check_params(result.calls, expect.params),
        *check_answer_text(text, expect.answer),
        *check_ground_truth(result.calls, expect.ground_truth),
        *check_outcome(result, expect),
    ]
    if expect.answer.get("traceable_numbers"):
        allowed = allowed_numbers(result.calls, context)
        violations.extend(check_faithfulness(text, allowed))
        violations.extend(_check_highlight_values(result.response, allowed))
    phrases = expect.answer.get("false_unavailability_phrases")
    if phrases:
        violations.extend(check_false_unavailability(text, result.calls, phrases))
    return violations
