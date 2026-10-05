import re
from pathlib import Path

import pytest

from llm_agent_evals.evals import load_scenarios, run_scenarios
from llm_agent_evals.evals.report import result_status
from llm_agent_evals.ledger import (
    LedgerError,
    load_ledger,
    parse_ledger,
    render_html,
    render_markdown,
    summary_counts,
    verify_ledger,
)

ROOT = Path(__file__).resolve().parents[1]
SCENARIO_IDS = [s.id for s in load_scenarios(ROOT / "scenarios")]


def entry(**overrides):
    body = {
        "id": "L-1",
        "class": "A class",
        "status": "closed",
        "finding": "f",
        "fix": "x",
        "evidence": [{"kind": "scenario", "ref": "S01"}],
    }
    body.update(overrides)
    return body


def ledger(*entries, **extra):
    return parse_ledger({"title": "T", "entries": list(entries or [entry()]), **extra})


def problems(*entries, root=ROOT):
    return verify_ledger(ledger(*entries), root, SCENARIO_IDS)


# -- the shipped ledger --------------------------------------------------------


def test_the_shipped_ledger_verifies_against_the_repository():
    shipped = load_ledger(ROOT / "ledger" / "entries.json")
    assert verify_ledger(shipped, ROOT, SCENARIO_IDS) == []
    counts = summary_counts(shipped)
    assert counts["closed"] >= 10 and counts["partial"] >= 3 and counts["open"] >= 1
    assert counts["landmines"] >= 3


def test_every_closed_entry_cites_a_test_or_a_scenario_and_every_partial_one_names_its_gap():
    shipped = load_ledger(ROOT / "ledger" / "entries.json")
    for item in shipped.by_status("closed"):
        assert any(e.kind in ("scenario", "test") for e in item.evidence), item.id
    for item in shipped.by_status("partial"):
        assert item.gap, item.id


def test_the_committed_documents_match_what_the_generator_produces():
    shipped = load_ledger(ROOT / "ledger" / "entries.json")
    status = {
        r.scenario_id: result_status(r) for r in run_scenarios(load_scenarios(ROOT / "scenarios"))
    }
    assert (ROOT / "docs" / "reliability-ledger.md").read_text(encoding="utf-8") == render_markdown(
        shipped, status
    )
    assert (ROOT / "docs" / "reliability-ledger.html").read_text(encoding="utf-8") == render_html(
        shipped, status
    )


# -- verification --------------------------------------------------------------


def test_a_missing_scenario_is_reported():
    out = problems(entry(evidence=[{"kind": "scenario", "ref": "S99"}]))
    assert out == ["L-1: scenario 'S99' does not exist"]


def test_a_test_reference_needs_an_existing_file_and_function():
    ok = {
        "kind": "test",
        "ref": "tests/test_guards.py::test_reject_costs_a_step_and_is_never_cached_or_memoised",
    }
    assert problems(entry(evidence=[ok])) == []
    no_file = {"kind": "test", "ref": "tests/test_nowhere.py::test_x"}
    assert "does not exist" in problems(entry(evidence=[no_file]))[0]
    no_func = {"kind": "test", "ref": "tests/test_guards.py::test_not_there"}
    assert "defines no function named 'test_not_there'" in problems(entry(evidence=[no_func]))[0]
    no_name = {"kind": "test", "ref": "tests/test_guards.py"}
    assert "path::test_name" in problems(entry(evidence=[no_name]))[0]


def test_a_module_reference_must_be_an_existing_file():
    assert (
        problems(
            entry(
                evidence=[
                    {"kind": "scenario", "ref": "S01"},
                    {"kind": "module", "ref": "src/llm_agent_evals/guards.py"},
                ]
            )
        )
        == []
    )
    out = problems(
        entry(
            evidence=[
                {"kind": "scenario", "ref": "S01"},
                {"kind": "module", "ref": "src/llm_agent_evals/gone.py"},
            ]
        )
    )
    assert out == ["L-1: module 'src/llm_agent_evals/gone.py' does not exist"]


def test_a_closed_entry_with_only_a_module_for_evidence_is_rejected():
    out = problems(entry(evidence=[{"kind": "module", "ref": "src/llm_agent_evals/guards.py"}]))
    assert out == ["L-1: a closed entry must cite a scenario or a test"]
    assert problems(entry(status="open", evidence=[])) == []


def test_a_partial_entry_must_say_what_the_gap_is():
    assert problems(entry(status="partial")) == ["L-1: a partial entry must say what the gap is"]
    assert problems(entry(status="partial", gap="not covered")) == []


def test_duplicate_ids_and_unknown_statuses_are_reported():
    out = problems(entry(), entry())
    assert "duplicate id L-1" in out
    assert "unknown status 'fixed'" in problems(entry(status="fixed"))[0]
    both = parse_ledger(
        {"title": "T", "entries": [entry()], "landmines": [{"id": "L-1", "note": "n"}]}
    )
    assert "duplicate id L-1" in verify_ledger(both, ROOT, SCENARIO_IDS)


def test_a_test_module_that_cannot_be_imported_is_still_checked_by_parsing(tmp_path):
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_broken.py").write_text(
        "import not_installed\n\ndef test_a():\n    pass\n", encoding="utf-8"
    )
    ref = {"kind": "test", "ref": "tests/test_broken.py::test_a"}
    assert verify_ledger(ledger(entry(evidence=[ref])), tmp_path, SCENARIO_IDS) == []


def test_methods_inside_test_classes_are_found(tmp_path):
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_cls.py").write_text(
        "class TestThing:\n    def test_method(self):\n        pass\n", encoding="utf-8"
    )
    ref = {"kind": "test", "ref": "tests/test_cls.py::test_method"}
    assert verify_ledger(ledger(entry(evidence=[ref])), tmp_path, SCENARIO_IDS) == []


# -- loading -------------------------------------------------------------------


def test_loading_errors_are_ledger_errors(tmp_path):
    with pytest.raises(LedgerError, match="cannot read"):
        load_ledger(tmp_path / "missing.json")
    (tmp_path / "list.json").write_text("[]", encoding="utf-8")
    with pytest.raises(LedgerError, match="JSON object"):
        load_ledger(tmp_path / "list.json")
    with pytest.raises(LedgerError, match="missing 'fix'"):
        parse_ledger({"title": "T", "entries": [entry(fix="")]})
    with pytest.raises(LedgerError, match="evidence needs a kind"):
        parse_ledger({"title": "T", "entries": [entry(evidence=[{"kind": "hunch", "ref": "x"}])]})
    with pytest.raises(LedgerError, match="structure is invalid"):
        parse_ledger({"entries": []})


# -- rendering -----------------------------------------------------------------


def test_markdown_groups_entries_by_status_and_is_deterministic():
    data = ledger(
        entry(id="L-1"),
        entry(id="L-2", status="open", evidence=[]),
        entry(id="L-3", status="partial", gap="g"),
        landmines=[{"id": "M-1", "note": "careful"}],
        methodology=["m"],
        constraints=["c"],
    )
    text = render_markdown(data)
    assert text == render_markdown(data)
    assert (
        text.index("## Closed (1)")
        < text.index("## Partially closed (1)")
        < text.index("## Open queue (1)")
    )
    assert "1 closed, 1 partially closed, 1 open, 1 landmines flagged." in text
    assert "### L-3: A class" in text and "- **Gap.** g" in text
    assert "## Landmines flagged, not fixed" in text and "careful" in text


def test_scenario_references_show_how_the_scenario_currently_behaves():
    text = render_markdown(ledger(), {"S01": "ok"})
    assert "scenario S01 (ok)" in text
    assert "scenario S01" in render_markdown(ledger())


def test_html_is_self_contained_and_escapes_everything():
    data = parse_ledger(
        {
            "title": "A <b>title</b>",
            "intro": "i",
            "entries": [entry(finding="<script>alert(1)</script> & more")],
        }
    )
    page = render_html(data)
    assert "<script>" not in page and "&lt;script&gt;" in page and "&amp; more" in page
    assert "&lt;b&gt;title" in page
    assert "http://" not in page and "https://" not in page


def test_an_empty_section_says_so():
    assert "Nothing in this state." in render_markdown(ledger())
    assert "Nothing here." in render_html(ledger())


def test_generated_documents_contain_no_dates_or_timestamps():
    shipped = load_ledger(ROOT / "ledger" / "entries.json")
    for text in (render_markdown(shipped), render_html(shipped)):
        assert not re.search(r"\d{4}-\d{2}-\d{2}", text)
