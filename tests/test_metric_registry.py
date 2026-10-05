import json
from pathlib import Path

import pytest

from llm_agent_evals.analytics.agents import agent_entries
from llm_agent_evals.business import DEFAULT_STATUS_IDS, BusinessConfig
from llm_agent_evals.metrics import MetricRegistry
from tests.helpers import agent_row

ROOT = Path(__file__).resolve().parents[1]


def registry(config=None):
    return MetricRegistry(config or BusinessConfig())


def test_the_committed_snapshot_matches_the_code():
    committed = json.loads(
        (ROOT / "docs" / "metric-registry.default.json").read_text(encoding="utf-8")
    )
    assert committed == registry().to_dict(), (
        "docs/metric-registry.default.json is out of date: "
        "run `llm-agent-evals metrics --write docs/metric-registry.default.json`"
    )


def test_every_kpi_has_a_formula_and_a_source():
    for name, entry in registry().to_dict()["kpis"].items():
        assert entry["formula"].strip() and entry["source"].strip(), name


AGENT_LEVEL_KPIS = (
    "attempt_count",
    "answer_count",
    "answer_rate_pct",
    "cpc_rate_pct",
    "conversion_pct",
    "paid_per_cpc_pct",
    "slip_effectiveness_pct",
    "avg_ticket",
    "first_installment_value",
    "agreement_value",
    "agreement_count",
    "agreement_count_by_contract",
)


def test_every_agent_level_kpi_in_the_registry_is_a_field_of_the_agent_entry():
    entry = agent_entries([agent_row("agent_01")], "2026-03-18")[0]
    kpis = registry().to_dict()["kpis"]
    for name in AGENT_LEVEL_KPIS:
        assert name in kpis and name in entry, name
    assert "cpc" in kpis and "cpc_count" in entry


def test_status_sections_follow_the_business_config():
    config = BusinessConfig(status_ids={**DEFAULT_STATUS_IDS, "REJECTED": 77})
    status = registry(config).to_dict()["status"]
    assert status["rejected"]["ids"] == [77]
    assert 77 not in status["generated"]["ids"] and 77 not in status["universe"]["ids"]


def test_caveats_quote_the_configured_numbers():
    config = BusinessConfig(grace_days=9, risk_low_max=11.0, risk_medium_max=22.0)
    caveats = registry(config).to_dict()["caveats"]
    assert "9 days" in caveats["slip_maturity"]
    assert "11.0%" in caveats["thresholds"] and "22.0%" in caveats["thresholds"]


@pytest.mark.parametrize(
    ("term", "resolved"),
    [
        ("CPC", "cpc"),
        ("right party contact", "cpc"),
        ("right-party-contact", "cpc"),
        ("Answer Rate", "answer_rate_pct"),
        ("composite risk", "composite_risk_formula"),
        ("ticket", "avg_ticket"),
    ],
)
def test_terms_are_normalised_and_aliased(term, resolved):
    result = registry().resolve(term)
    assert result["type"] == "kpi" and result == registry().resolve(resolved)


def test_a_status_group_carries_its_caveat():
    exception = registry().resolve("exception")
    assert exception["type"] == "status" and "pending approval" in exception["caveat"]
    assert exception["values"]["ids"] == [40]


def test_an_unknown_term_is_none_and_the_term_list_is_sorted():
    reg = registry()
    assert reg.resolve("vibes") is None
    terms = reg.available_terms()
    assert terms == sorted(terms) and "conversion_pct" in terms and "generated" in terms


def test_composite_risk_is_documented_as_the_maximum():
    formula = registry().resolve("composite_risk")["formula"]
    assert "MAX" in formula and "Never the sum" in formula


def test_conversion_effectiveness_and_paid_per_cpc_are_three_different_entries():
    reg = registry()
    formulas = {
        reg.resolve(t)["formula"]
        for t in ("conversion_pct", "slip_effectiveness_pct", "paid_per_cpc_pct")
    }
    assert len(formulas) == 3
    assert "never call it conversion" in reg.resolve("paid_per_cpc_pct")["formula"]


def test_the_registry_is_independent_between_instances():
    one, two = registry(), registry()
    one.to_dict()["kpis"]["cpc"]["formula"] = "tampered"
    assert two.to_dict()["kpis"]["cpc"]["formula"] != "tampered"
