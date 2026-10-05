"""Check a ledger against the repository.

A ledger that cites evidence nobody checks decays into a list of claims. `verify_ledger`
resolves every evidence reference:

* `scenario`: the id must exist in the scenarios directory;
* `test`: `path::name` must point at a file that defines a function or method of that
  name (found by parsing, not by importing, so a broken test module is still checked);
* `module`: the path must exist.

It also enforces the ledger's own rules: unique ids, a known status, and that a `closed`
entry cites at least one scenario or test, because a source file alone shows where a fix
lives, not that anything still guards it.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable
from pathlib import Path

from llm_agent_evals.ledger.model import STATUSES, Entry, Evidence, Ledger


def _defined_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _check_evidence(entry: Entry, item: Evidence, root: Path, scenario_ids: set[str]) -> list[str]:
    where = f"{entry.id}: {item.kind} {item.ref!r}"
    if item.kind == "scenario":
        return [] if item.ref in scenario_ids else [f"{where} does not exist"]
    if item.kind == "module":
        return [] if (root / item.ref).is_file() else [f"{where} does not exist"]
    path_text, _, name = item.ref.partition("::")
    path = root / path_text
    if not path.is_file():
        return [f"{where}: file {path_text!r} does not exist"]
    if not name:
        return [f"{where}: a test reference needs 'path::test_name'"]
    if name not in _defined_names(path):
        return [f"{where}: {path_text} defines no function named {name!r}"]
    return []


def _check_entry(entry: Entry, root: Path, scenario_ids: set[str]) -> list[str]:
    problems: list[str] = []
    if entry.status not in STATUSES:
        problems.append(f"{entry.id}: unknown status {entry.status!r}")
    if entry.status == "closed" and not any(e.kind in ("scenario", "test") for e in entry.evidence):
        problems.append(f"{entry.id}: a closed entry must cite a scenario or a test")
    if entry.status == "partial" and not entry.gap:
        problems.append(f"{entry.id}: a partial entry must say what the gap is")
    for item in entry.evidence:
        problems.extend(_check_evidence(entry, item, root, scenario_ids))
    return problems


def verify_ledger(ledger: Ledger, root: str | Path, scenario_ids: Iterable[str]) -> list[str]:
    """Every problem found, as readable lines. An empty list means the ledger checks out."""
    base = Path(root)
    known = set(scenario_ids)
    problems: list[str] = []
    ids = [entry.id for entry in ledger.entries] + [m.id for m in ledger.landmines]
    problems.extend(f"duplicate id {i}" for i in sorted({i for i in ids if ids.count(i) > 1}))
    for entry in ledger.entries:
        problems.extend(_check_entry(entry, base, known))
    return problems
