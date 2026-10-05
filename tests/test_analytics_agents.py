from llm_agent_evals.analytics.agents import (
    COMPARE_METRICS,
    ORDER_BY_FIELDS,
    agent_entries,
)
from tests.helpers import agent_row

REF = "2026-03-18"


def one(**overrides):
    return agent_entries([agent_row("agent_01", **overrides)], REF)[0]


def test_funnel_ratios_follow_the_documented_formulas():
    entry = one(
        attempt_count=200,
        answer_count=100,
        cpc_count=40,
        agreement_count=10,
        agreement_value=5000.0,
        slips_issued=10,
        slips_paid=7,
    )
    assert entry["answer_rate_pct"] == 50.0  # answers / attempts
    assert entry["cpc_rate_pct"] == 40.0  # cpc / answers
    assert entry["avg_ticket"] == 500.0  # value / agreements
    assert entry["conversion_pct"] == 25.0  # agreements / cpc
    assert entry["paid_per_cpc_pct"] == 17.5  # slips paid / cpc
    assert entry["slip_effectiveness_pct"] == 70.0  # slips paid / slips issued


def test_conversion_and_paid_per_cpc_are_two_different_metrics():
    entry = one(cpc_count=25, agreement_count=8, slips_paid=6, slips_issued=8)
    assert entry["conversion_pct"] == 32.0
    assert entry["paid_per_cpc_pct"] == 24.0
    assert entry["conversion_pct"] != entry["paid_per_cpc_pct"]


def test_cpc_is_a_count_and_not_a_percentage():
    entry = one(cpc_count=40)
    assert entry["cpc_count"] == 40 and isinstance(entry["cpc_count"], int)
    assert "cpc_pct" not in entry


def test_two_lines_are_combined_by_summing_counters_then_dividing_once():
    line_a = agent_row("agent_01", cpc_count=10, agreement_count=5)  # 50%
    line_b = agent_row("agent_01", cpc_count=90, agreement_count=9)  # 10%
    combined = agent_entries([line_a, line_b], REF)
    assert len(combined) == 1
    assert combined[0]["conversion_pct"] == 14.0  # 14 / 100, not the mean of 50 and 10
    assert combined[0]["conversion_pct"] != (50.0 + 10.0) / 2


def test_agent_identity_is_case_insensitive_across_rows():
    rows = [agent_row("Agent_01"), agent_row("agent_01")]
    assert len(agent_entries(rows, REF)) == 1


def test_zero_denominators_give_zero_ratios():
    entry = one(attempt_count=0, answer_count=0, cpc_count=0, agreement_count=0)
    assert entry["answer_rate_pct"] == 0.0 and entry["conversion_pct"] == 0.0
    assert entry["avg_ticket"] == 0.0


def test_agreements_may_exceed_cpc_without_being_an_anomaly():
    entry = one(cpc_count=3, agreement_count=5)
    assert entry["conversion_pct"] > 100.0


def test_entries_are_sorted_by_agreement_value_then_id():
    rows = [
        agent_row("agent_02", agreement_value=1.0),
        agent_row("agent_01", agreement_value=1.0),
        agent_row("agent_03", agreement_value=9.0),
    ]
    assert [e["agent_id"] for e in agent_entries(rows, REF)] == ["agent_03", "agent_01", "agent_02"]


def test_rows_without_an_identity_are_ignored():
    assert agent_entries([{"agent_id": "", "agent_name": ""}], REF) == []


def test_every_orderable_and_comparable_field_exists_on_an_entry():
    entry = one()
    assert set(ORDER_BY_FIELDS) <= set(entry)
    assert set(COMPARE_METRICS) <= set(entry)
