"""The ledger data model and its JSON loader.

A reliability ledger is a status board of failure classes. Every entry says what went
wrong (or could), what fixed it, which test or scenario proves the fix still holds, and what
risk remains. The board lives in data (`ledger/entries.json`) rather than in a hand-edited
page, so it can be checked against the repository (`verify.py`) and rendered (`render.py`).

Statuses:

* `closed`   a fix exists and at least one scenario or test guards it;
* `partial`  mitigated, but something that matters is not covered (the `gap` says what);
* `open`     known and not fixed, queued;
* a separate list of `landmines` holds hazards that are flagged and deliberately not fixed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from llm_agent_evals.errors import AgentError

STATUSES = ("closed", "partial", "open")
EVIDENCE_KINDS = ("scenario", "test", "module")


class LedgerError(AgentError):
    """The ledger file is missing or malformed."""


@dataclass(frozen=True)
class Evidence:
    """A pointer into the repository. `ref` formats: a scenario id (`S02`), a test
    (`tests/test_guards.py::test_name`) or a source path (`src/llm_agent_evals/guards.py`)."""

    kind: str
    ref: str


@dataclass(frozen=True)
class Entry:
    id: str
    failure_class: str
    status: str
    finding: str
    fix: str
    evidence: tuple[Evidence, ...] = ()
    residual_risk: str = ""
    gap: str = ""


@dataclass(frozen=True)
class Landmine:
    id: str
    note: str


@dataclass(frozen=True)
class Ledger:
    title: str
    intro: str
    entries: tuple[Entry, ...]
    landmines: tuple[Landmine, ...] = ()
    methodology: tuple[str, ...] = ()
    constraints: tuple[str, ...] = field(default_factory=tuple)

    def by_status(self, status: str) -> list[Entry]:
        return [entry for entry in self.entries if entry.status == status]


def _evidence(raw: dict[str, Any], where: str) -> Evidence:
    kind = raw.get("kind")
    if kind not in EVIDENCE_KINDS or not raw.get("ref"):
        raise LedgerError(f"{where}: evidence needs a kind in {EVIDENCE_KINDS} and a ref")
    return Evidence(kind=kind, ref=str(raw["ref"]))


def _entry(raw: dict[str, Any]) -> Entry:
    where = str(raw.get("id", "<entry without id>"))
    for required in ("id", "class", "status", "finding", "fix"):
        if not raw.get(required):
            raise LedgerError(f"{where}: missing '{required}'")
    return Entry(
        id=str(raw["id"]),
        failure_class=str(raw["class"]),
        status=str(raw["status"]),
        finding=str(raw["finding"]),
        fix=str(raw["fix"]),
        evidence=tuple(_evidence(e, where) for e in raw.get("evidence", [])),
        residual_risk=str(raw.get("residual_risk", "")),
        gap=str(raw.get("gap", "")),
    )


def parse_ledger(raw: dict[str, Any]) -> Ledger:
    """Build a `Ledger` from decoded JSON. Structure errors raise `LedgerError`."""
    try:
        return Ledger(
            title=str(raw["title"]),
            intro=str(raw.get("intro", "")),
            entries=tuple(_entry(e) for e in raw["entries"]),
            landmines=tuple(
                Landmine(str(m["id"]), str(m["note"])) for m in raw.get("landmines", [])
            ),
            methodology=tuple(str(line) for line in raw.get("methodology", [])),
            constraints=tuple(str(line) for line in raw.get("constraints", [])),
        )
    except (KeyError, TypeError) as exc:
        raise LedgerError(f"ledger structure is invalid: {exc!r}") from exc


def load_ledger(path: str | Path) -> Ledger:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise LedgerError(f"cannot read ledger file {str(path)!r}: {exc}") from exc
    if not isinstance(raw, dict):
        raise LedgerError("the ledger file must contain a JSON object")
    return parse_ledger(raw)
