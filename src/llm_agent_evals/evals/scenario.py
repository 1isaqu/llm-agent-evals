"""Scenario files: what is asked, what data exists, how the model behaves, what is expected.

A scenario is one JSON file (format documented in docs/eval-format.md):

    id, title, category, question | messages, session, now,
    dataset (path) [+ dataset_patch] | datasets (inline), script, expect, limits, config

`load_scenario` is strict. An unknown key (also in `session`, `config` and the dataset
sections), a missing field or a malformed date is a `ScenarioError` that names the file,
because a typo such as `tools_forbiden` would otherwise silently turn a check off and leave
the scenario passing for the wrong reason.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

from llm_agent_evals.business import BusinessConfig
from llm_agent_evals.errors import AgentError, ConfigError
from llm_agent_evals.llm.base import Message
from llm_agent_evals.session import ALL_LINES, Session, Window
from llm_agent_evals.sources.base import SOURCE_METHODS


class ScenarioError(AgentError):
    """A scenario file is malformed."""


_TOP_KEYS = {
    "id",
    "title",
    "category",
    "question",
    "messages",
    "session",
    "now",
    "dataset",
    "dataset_patch",
    "datasets",
    "script",
    "expect",
    "limits",
    "config",
    "notes",
}
_EXPECT_KEYS = {
    "should_pass",
    "expect_violations",
    "tools_any_of",
    "tools_all_of",
    "tools_forbidden",
    "zero_tool_calls",
    "max_tool_calls",
    "params",
    "answer",
    "ground_truth",
    "tool_errors",
    "confidence",
    "degraded",
    "forced_final",
    "model_error",
    "format_ok",
}
_ANSWER_KEYS = {
    "traceable_numbers",
    "forbidden",
    "contains",
    "contains_any_of",
    "contains_date",
    "false_unavailability_phrases",
}
_LIMIT_KEYS = {"max_steps", "max_rounds", "wall_clock_seconds", "clock_tick_seconds"}
_SESSION_KEYS = {"product_line", "date_from", "date_to"}
_PARAM_KEYS = {"tool", "contains", "absent"}
_TRUTH_KEYS = {"tool", "path", "equals", "tolerance_pct", "occurrence"}
_TRUTH_REQUIRED = {"tool", "path", "equals"}


@dataclass(frozen=True)
class Expectations:
    """What a scenario requires of the run. Every field defaults to "no requirement"."""

    should_pass: bool = True
    expect_violations: tuple[str, ...] = ()
    tools_any_of: tuple[str, ...] = ()
    tools_all_of: tuple[str, ...] = ()
    tools_forbidden: tuple[str, ...] = ()
    zero_tool_calls: bool = False
    max_tool_calls: int | None = None
    params: tuple[Mapping[str, Any], ...] = ()
    answer: Mapping[str, Any] = field(default_factory=dict)
    ground_truth: tuple[Mapping[str, Any], ...] = ()
    tool_errors: tuple[str, ...] = ()
    confidence: tuple[str, ...] = ()
    degraded: str | None = None
    forced_final: bool | None = None
    model_error: bool = False
    format_ok: bool = True


@dataclass(frozen=True)
class Scenario:
    id: str
    title: str
    category: str
    messages: tuple[Message, ...]
    session: Session
    now: datetime
    datasets: Mapping[str, Mapping[str, Any]]
    script: tuple[Any, ...]
    expect: Expectations
    limits: Mapping[str, Any] = field(default_factory=dict)
    config: Mapping[str, Any] = field(default_factory=dict)
    path: str = ""

    @property
    def question(self) -> str:
        return next((m.content for m in reversed(self.messages) if m.role == "user"), "")


def _check_keys(found: Mapping[str, Any], allowed: set[str], where: str, path: Path) -> None:
    unknown = sorted(set(found) - allowed)
    if unknown:
        raise ScenarioError(f"{path.name}: unknown key(s) in {where}: {', '.join(unknown)}")


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ScenarioError(f"cannot read {path}: {exc}") from exc


def _tuple(value: Any) -> tuple[Any, ...]:
    return tuple(value) if isinstance(value, list) else ()


def _check_specs(raw: Mapping[str, Any], path: Path) -> None:
    for spec in raw.get("params", ()):
        _check_keys(spec, _PARAM_KEYS, "expect.params", path)
        if not spec.get("contains") and not spec.get("absent"):
            raise ScenarioError(f"{path.name}: a params entry needs 'contains' or 'absent'")
    for spec in raw.get("ground_truth", ()):
        _check_keys(spec, _TRUTH_KEYS, "expect.ground_truth", path)
        missing = sorted(_TRUTH_REQUIRED - set(spec))
        if missing:
            raise ScenarioError(f"{path.name}: ground_truth entry lacks {', '.join(missing)}")


def _expectations(raw: Mapping[str, Any], path: Path) -> Expectations:
    _check_keys(raw, _EXPECT_KEYS, "expect", path)
    answer = raw.get("answer", {})
    _check_keys(answer, _ANSWER_KEYS, "expect.answer", path)
    _check_specs(raw, path)
    confidence = raw.get("confidence", ())
    return Expectations(
        should_pass=bool(raw.get("should_pass", True)),
        expect_violations=_tuple(raw.get("expect_violations")),
        tools_any_of=_tuple(raw.get("tools_any_of")),
        tools_all_of=_tuple(raw.get("tools_all_of")),
        tools_forbidden=_tuple(raw.get("tools_forbidden")),
        zero_tool_calls=bool(raw.get("zero_tool_calls", False)),
        max_tool_calls=raw.get("max_tool_calls"),
        params=_tuple(raw.get("params")),
        answer=dict(answer),
        ground_truth=_tuple(raw.get("ground_truth")),
        tool_errors=_tuple(raw.get("tool_errors")),
        confidence=(confidence,) if isinstance(confidence, str) else _tuple(confidence),
        degraded=raw.get("degraded"),
        forced_final=raw.get("forced_final"),
        model_error=bool(raw.get("model_error", False)),
        format_ok=bool(raw.get("format_ok", True)),
    )


def _session(raw: Mapping[str, Any], path: Path) -> Session:
    if not isinstance(raw, Mapping):
        raise ScenarioError(f"{path.name}: 'session' must be an object")
    _check_keys(raw, _SESSION_KEYS, "session", path)
    try:
        window = Window(date.fromisoformat(raw["date_from"]), date.fromisoformat(raw["date_to"]))
        return Session(str(raw["product_line"]), window)
    except (KeyError, ValueError) as exc:
        raise ScenarioError(f"{path.name}: invalid session: {exc}") from exc


def _messages(raw: Mapping[str, Any], path: Path) -> tuple[Message, ...]:
    if "messages" in raw:
        return tuple(
            Message(role=str(m["role"]), content=str(m["content"])) for m in raw["messages"]
        )
    if "question" not in raw:
        raise ScenarioError(f"{path.name}: needs 'question' or 'messages'")
    return (Message(role="user", content=str(raw["question"])),)


def _apply_patch(
    base: Mapping[str, Mapping[str, Any]], patch: Mapping[str, Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    """Replace whole sections per product line, leaving the rest of the dataset alone."""
    merged = {line: dict(sections) for line, sections in base.items()}
    for line, sections in patch.items():
        merged.setdefault(line, {}).update(sections)
    return merged


def _datasets(raw: Mapping[str, Any], path: Path) -> Mapping[str, Mapping[str, Any]]:
    if "datasets" in raw:
        return dict(raw["datasets"])
    if "dataset" not in raw:
        raise ScenarioError(f"{path.name}: needs 'dataset' (a path) or 'datasets' (inline)")
    loaded = _load_json(path.parent / str(raw["dataset"]))
    if not isinstance(loaded, dict):
        raise ScenarioError(f"{raw['dataset']}: a dataset file must be a JSON object")
    return _apply_patch(loaded, raw.get("dataset_patch", {}))


def _check_datasets(datasets: Mapping[str, Any], path: Path) -> None:
    for line, sections in datasets.items():
        if not isinstance(sections, Mapping):
            raise ScenarioError(f"{path.name}: the dataset for line {line!r} must be an object")
        unknown = sorted(set(sections) - set(SOURCE_METHODS))
        if unknown:
            raise ScenarioError(
                f"{path.name}: unknown dataset section(s) for line {line!r}: {', '.join(unknown)}"
            )


def _config(raw: Any, path: Path) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ScenarioError(f"{path.name}: 'config' must be an object")
    try:
        BusinessConfig.from_mapping(raw)
    except (ConfigError, TypeError) as exc:
        raise ScenarioError(f"{path.name}: invalid config: {exc}") from exc
    return dict(raw)


def parse_scenario(raw: Mapping[str, Any], path: Path) -> Scenario:
    """Validate a decoded scenario and build the `Scenario`."""
    _check_keys(raw, _TOP_KEYS, "scenario", path)
    for required in ("id", "title", "category", "session", "now", "script", "expect"):
        if required not in raw:
            raise ScenarioError(f"{path.name}: missing '{required}'")
    limits = dict(raw.get("limits", {}))
    _check_keys(limits, _LIMIT_KEYS, "limits", path)
    try:
        now = datetime.fromisoformat(str(raw["now"]))
    except ValueError as exc:
        raise ScenarioError(f"{path.name}: invalid 'now': {exc}") from exc
    session = _session(raw["session"], path)
    datasets = _datasets(raw, path)
    _check_datasets(datasets, path)
    if session.product_line != ALL_LINES and session.product_line not in datasets:
        raise ScenarioError(f"{path.name}: dataset has no data for line {session.product_line!r}")
    return Scenario(
        id=str(raw["id"]),
        title=str(raw["title"]),
        category=str(raw["category"]),
        messages=_messages(raw, path),
        session=session,
        now=now,
        datasets=datasets,
        script=tuple(raw["script"]),
        expect=_expectations(raw["expect"], path),
        limits=limits,
        config=_config(raw.get("config", {}), path),
        path=str(path),
    )


def load_scenario(path: str | Path) -> Scenario:
    file = Path(path)
    raw = _load_json(file)
    if not isinstance(raw, dict):
        raise ScenarioError(f"{file.name}: a scenario file must be a JSON object")
    return parse_scenario(raw, file)


def load_scenarios(directory: str | Path) -> list[Scenario]:
    """Every `*.json` directly inside `directory` (datasets live in a subfolder), by file name."""
    root = Path(directory)
    if not root.is_dir():
        raise ScenarioError(f"scenario directory not found: {root}")
    scenarios = [load_scenario(p) for p in sorted(root.glob("*.json"))]
    ids = [s.id for s in scenarios]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise ScenarioError(f"duplicate scenario id(s): {', '.join(duplicates)}")
    return scenarios
