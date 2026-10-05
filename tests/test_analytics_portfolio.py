import pytest

from llm_agent_evals.analytics.common import StatusBucket, StatusSets, ratio_pct
from llm_agent_evals.analytics.portfolio import portfolio_entries, status_breakdown
from llm_agent_evals.business import BusinessConfig
from tests.helpers import BASE_STATUS, P1, P2, status_row

CONFIG = BusinessConfig()
REF = "2026-03-18"


def entries(rows=None):
    return {e["portfolio_name"]: e for e in portfolio_entries(rows or BASE_STATUS, CONFIG, REF)}


def test_every_axis_is_a_share_of_the_same_universe():
    p1 = entries()[P1]
    # generated 500 (400 active + 100 broken), exception 100, rejected 0 -> universe 600
    assert p1["first_installment_value"] == 500.0
    assert p1["breakdown"] == {"exception_pct": 16.67, "broken_pct": 16.67, "rejected_pct": 0.0}


def test_composite_risk_is_the_maximum_axis_never_the_sum():
    rows = [
        status_row("P", 10, 4, 400.0),
        status_row("P", 20, 2, 200.0),
        status_row("P", 40, 2, 300.0),
        status_row("P", 50, 1, 100.0),
    ]
    entry = entries(rows)["P"]
    axes = entry["breakdown"]
    assert entry["composite_risk_pct"] == max(axes.values())
    assert entry["composite_risk_pct"] < sum(axes.values())


def test_risk_level_comes_from_the_configured_cutoffs():
    by_name = entries()
    assert by_name[P1]["risk_level"] == "low"
    assert by_name[P2]["risk_level"] == "medium"
    strict = BusinessConfig(risk_low_max=5.0, risk_medium_max=10.0)
    assert portfolio_entries(BASE_STATUS, strict, REF)[0]["risk_level"] == "high"


def test_a_negative_amount_is_flagged_as_an_anomaly_and_not_corrected():
    rows = [status_row("P", 10, 1, -1000.0), status_row("P", 40, 2, 2500.0)]
    entry = entries(rows)["P"]
    assert entry["anomaly"] is True
    assert entry["composite_risk_pct"] == 166.67


def test_a_high_value_at_or_below_the_universe_is_risk_and_not_an_anomaly():
    rows = [status_row("P", 40, 5, 900.0), status_row("P", 10, 1, 100.0)]
    entry = entries(rows)["P"]
    assert entry["composite_risk_pct"] == 90.0 and entry["anomaly"] is False


def test_clean_data_can_never_leave_the_zero_to_hundred_range():
    for amounts in ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 5.0, 5.0), (3.0, 9.0, 1.0)):
        rows = [status_row("P", s, 1, a) for s, a in zip((10, 40, 50), amounts, strict=True)]
        axes = entries(rows)["P"]["breakdown"].values()
        assert all(0.0 <= value <= 100.0 for value in axes)


def test_an_empty_universe_gives_zero_risk_not_a_division_error():
    entry = entries([status_row("P", 10, 1, 0.0)])["P"]
    assert entry["composite_risk_pct"] == 0.0 and entry["first_installment_value"] == 0.0


def test_portfolios_are_sorted_by_value_then_name_and_blank_names_are_dropped():
    rows = [
        status_row("B", 10, 1, 50.0),
        status_row("A", 10, 1, 50.0),
        status_row(" ", 10, 1, 99.0),
    ]
    assert [e["portfolio_name"] for e in portfolio_entries(rows, CONFIG, REF)] == ["A", "B"]


def test_statuses_outside_every_set_do_not_count():
    rows = [status_row("P", 999, 3, 500.0), status_row("P", 10, 1, 100.0)]
    assert entries(rows)["P"]["first_installment_value"] == 100.0


def test_status_breakdown_states_generated_separately_from_all_statuses():
    result = status_breakdown(BASE_STATUS, CONFIG, REF)
    assert result["generated_amount"] == 700.0 and result["generated_qty"] == 7
    assert result["all_statuses_amount"] == 900.0
    assert result["all_statuses_amount"] != result["generated_amount"]
    labels = {s["label"]: s["amount"] for s in result["statuses"]}
    assert labels == {
        "ACTIVE": 600.0,
        "BROKEN": 100.0,
        "PENDING_EXCEPTION": 100.0,
        "REJECTED": 100.0,
    }


def test_status_bucket_arithmetic_is_explicit():
    sets = StatusSets.from_config(CONFIG)
    bucket = StatusBucket()
    bucket.add(sets, 10, 2, 200.0)
    bucket.add(sets, 40, 1, 50.0)
    bucket.add(sets, 50, 1, 50.0)
    assert bucket.universe_value == 300.0
    assert bucket.share_pct(50.0) == 16.67


@pytest.mark.parametrize(
    ("num", "den", "out"), [(1, 3, 33.33), (0, 5, 0.0), (5, 0, 0.0), (5, -1, 0.0)]
)
def test_ratio_pct_rounds_and_never_divides_by_zero(num, den, out):
    assert ratio_pct(num, den) == out
