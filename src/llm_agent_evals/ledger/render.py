"""Render a ledger as Markdown or as a self-contained HTML page.

Output is deterministic: no timestamps, no host information. A test regenerates both
documents and compares them with the committed copies, so they cannot drift from
`ledger/entries.json`.

When `scenario_status` is given (scenario id to a status word such as `ok`), each scenario
reference is annotated with how that scenario currently behaves. Test references are listed
as they are: whether a test passes is the test runner's business.
"""

from __future__ import annotations

import html
import re
from collections.abc import Mapping

from llm_agent_evals.ledger.model import STATUSES, Entry, Evidence, Ledger

_HEADINGS = {
    "closed": "Closed",
    "partial": "Partially closed",
    "open": "Open queue",
}
_CODE_SPAN = re.compile(r"`([^`]+)`")
GENERATED_NOTE = (
    "Generated from `ledger/entries.json` by `llm-agent-evals ledger generate`. "
    "Edit the data, not this file."
)


def _html_text(value: str) -> str:
    """Escape `value`, then turn Markdown code spans into `<code>` elements."""
    return _CODE_SPAN.sub(lambda m: f"<code>{m.group(1)}</code>", html.escape(value))


def _ref_text(item: Evidence, scenario_status: Mapping[str, str] | None) -> str:
    if item.kind == "scenario" and scenario_status and item.ref in scenario_status:
        return f"scenario {item.ref} ({scenario_status[item.ref]})"
    return f"{item.kind} {item.ref}"


def summary_counts(ledger: Ledger) -> dict[str, int]:
    counts = {status: len(ledger.by_status(status)) for status in STATUSES}
    counts["landmines"] = len(ledger.landmines)
    return counts


def _entry_markdown(entry: Entry, scenario_status: Mapping[str, str] | None) -> list[str]:
    lines = [
        f"### {entry.id}: {entry.failure_class}",
        "",
        f"- Finding: {entry.finding}",
        f"- Fix: {entry.fix}",
    ]
    if entry.gap:
        lines.append(f"- Gap: {entry.gap}")
    if entry.evidence:
        refs = "; ".join(_ref_text(item, scenario_status) for item in entry.evidence)
        lines.append(f"- Evidence: {refs}")
    if entry.residual_risk:
        lines.append(f"- Residual risk: {entry.residual_risk}")
    return [*lines, ""]


def render_markdown(ledger: Ledger, scenario_status: Mapping[str, str] | None = None) -> str:
    counts = summary_counts(ledger)
    lines = [
        f"# {ledger.title}",
        "",
        f"_{GENERATED_NOTE}_",
        "",
        ledger.intro,
        "",
        f"Status: {counts['closed']} closed, {counts['partial']} partially closed, "
        f"{counts['open']} open, {counts['landmines']} landmines flagged.",
        "",
    ]
    for status in STATUSES:
        entries = ledger.by_status(status)
        lines.append(f"## {_HEADINGS[status]} ({len(entries)})")
        lines.append("")
        if not entries:
            lines += ["Nothing in this state.", ""]
        for entry in entries:
            lines.extend(_entry_markdown(entry, scenario_status))
    if ledger.landmines:
        lines += ["## Landmines flagged, not fixed", ""]
        lines += [f"- {mine.id}: {mine.note}" for mine in ledger.landmines] + [""]
    if ledger.methodology:
        lines += ["## Method", ""] + [f"- {line}" for line in ledger.methodology] + [""]
    if ledger.constraints:
        lines += ["## Hard constraints", ""] + [f"- {line}" for line in ledger.constraints] + [""]
    return "\n".join(lines).rstrip("\n") + "\n"


_CSS = """
:root { color-scheme: light dark; --bg:#fafafa; --fg:#1b1f23; --card:#fff; --line:#d8dee4;
  --closed:#1a7f37; --partial:#9a6700; --open:#cf222e; --muted:#59636e; }
@media (prefers-color-scheme: dark) { :root { --bg:#0d1117; --fg:#e6edf3; --card:#161b22;
  --line:#30363d; --closed:#3fb950; --partial:#d29922; --open:#f85149; --muted:#8d96a0; } }
body { font: 15px/1.5 system-ui, sans-serif; background: var(--bg); color: var(--fg);
  max-width: 60rem; margin: 0 auto; padding: 1.5rem 1rem 3rem; }
h1 { margin-bottom: .25rem; } .note { color: var(--muted); font-size: .85rem; }
.stats { display: flex; gap: .75rem; flex-wrap: wrap; margin: 1rem 0 1.5rem; }
.stat { background: var(--card); border: 1px solid var(--line); border-radius: 8px;
  padding: .5rem .9rem; } .stat b { font-size: 1.4rem; display: block; }
.entry { background: var(--card); border: 1px solid var(--line); border-left-width: 4px;
  border-radius: 8px; padding: .75rem 1rem; margin: .75rem 0; }
.entry.closed { border-left-color: var(--closed); }
.entry.partial { border-left-color: var(--partial); }
.entry.open { border-left-color: var(--open); }
.entry h3 { margin: 0 0 .4rem; font-size: 1rem; } .entry p { margin: .2rem 0; }
.label { font-weight: 600; } code { font-size: .85em; }
"""


def _entry_html(entry: Entry, scenario_status: Mapping[str, str] | None) -> str:
    rows = [
        ("Finding", entry.finding),
        ("Fix", entry.fix),
        ("Gap", entry.gap),
        ("Evidence", "; ".join(_ref_text(i, scenario_status) for i in entry.evidence)),
        ("Residual risk", entry.residual_risk),
    ]
    body = "".join(
        f'<p><span class="label">{label}.</span> {_html_text(value)}</p>'
        for label, value in rows
        if value
    )
    title = _html_text(f"{entry.id}: {entry.failure_class}")
    return f'<div class="entry {entry.status}"><h3>{title}</h3>{body}</div>'


def render_html(ledger: Ledger, scenario_status: Mapping[str, str] | None = None) -> str:
    counts = summary_counts(ledger)
    stats = "".join(
        f'<div class="stat"><b>{counts[key]}</b>{label}</div>'
        for key, label in (
            ("closed", "closed"),
            ("partial", "partially closed"),
            ("open", "open"),
            ("landmines", "landmines"),
        )
    )
    sections = []
    for status in STATUSES:
        entries = ledger.by_status(status)
        cards = "".join(_entry_html(e, scenario_status) for e in entries) or "<p>Nothing here.</p>"
        sections.append(f"<h2>{_HEADINGS[status]} ({len(entries)})</h2>{cards}")
    if ledger.landmines:
        items = "".join(
            f"<li><b>{html.escape(m.id)}.</b> {_html_text(m.note)}</li>" for m in ledger.landmines
        )
        sections.append(f"<h2>Landmines flagged, not fixed</h2><ul>{items}</ul>")
    for heading, lines in (
        ("Method", ledger.methodology),
        ("Hard constraints", ledger.constraints),
    ):
        if lines:
            items = "".join(f"<li>{_html_text(line)}</li>" for line in lines)
            sections.append(f"<h2>{heading}</h2><ul>{items}</ul>")
    return (
        '<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{html.escape(ledger.title)}</title><style>{_CSS}</style></head><body>"
        f"<h1>{html.escape(ledger.title)}</h1>"
        f'<p class="note">{_html_text(GENERATED_NOTE)}</p>'
        f'<p>{_html_text(ledger.intro)}</p><div class="stats">{stats}</div>'
        + "".join(sections)
        + "</body></html>\n"
    )
