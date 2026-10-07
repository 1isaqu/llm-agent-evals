"""Crosstab, ranking, plan stages, drill-downs, slip effectiveness and the daily pace."""

from datetime import datetime

import pytest

from llm_agent_evals.analytics.crosstab import (
    agent_ranking,
    crosstab_summary,
    dimension_ids,
)
from llm_agent_evals.analytics.drilldown import (
    generation_ranking,
    maturities_ranking,
    maturities_summary,
    page_rows,
)
from llm_agent_evals.analytics.effectiveness import (
    effectiveness_points,
    effectiveness_total,
)
from llm_agent_evals.analytics.pace import BaselinePace
from llm_agent_evals.analytics.stages import STAGES, stage_summary
from llm_agent_evals.business import BusinessConfig
from llm_agent_evals.errors import DataSourceError
from llm_agent_evals.sources.fixture import FixtureSource
from tests.helpers import WINDOW

CONFIG = BusinessConfig()
REF = "2026-03-18"


def cross_row(agent, portfolio, status, qty, amount):
    return {
        "agent_id": agent,
        "agent_name": agent.title(),
        "portfolio_name": portfolio,
        "status_id": status,
        "qty": qty,
        "amount": amount,
    }


# -- crosstab and ranking ---------------------------------------------------------


def test_crosstab_decomposes_the_free_side_by_status_dimension():
    rows = [
        cross_row("a1", "P", 10, 2, 200.0),
        cross_row("a1", "P", 20, 1, 100.0),
        cross_row("a1", "P", 40, 1, 50.0),
        cross_row("a2", "P", 10, 1, 90.0),
        cross_row("a2", "P", 50, 1, 30.0),
    ]
    by_agent = {e["agent_id"]: e for e in crosstab_summary(rows, "agent", CONFIG, REF)}
    assert by_agent["a1"]["first_installment_value"] == 300.0  # active + broken
    assert by_agent["a1"]["exception_value"] == 50.0
    assert by_agent["a1"]["broken_value"] == 100.0
    assert by_agent["a2"]["rejected_value"] == 30.0


def test_crosstab_by_portfolio_names_the_portfolio_side():
    rows = [cross_row("a1", "P-1", 10, 1, 10.0), cross_row("a1", "P-2", 10, 1, 20.0)]
    entries = crosstab_summary(rows, "portfolio", CONFIG, REF)
    assert [e["portfolio_name"] for e in entries] == ["P-2", "P-1"]


def test_crosstab_is_limited_by_the_configuration():
    rows = [cross_row(f"a{i}", "P", 10, 1, float(i)) for i in range(1, 6)]
    small = BusinessConfig(crosstab_limit=2)
    assert len(crosstab_summary(rows, "agent", small, REF)) == 2


def test_ranking_orders_agents_by_value_in_one_dimension():
    rows = [
        cross_row("a1", "P", 20, 1, 100.0),
        cross_row("a2", "P", 20, 2, 300.0),
        cross_row("a2", "P", 10, 9, 999.0),
    ]
    ranking = agent_ranking(rows, "broken", CONFIG, limit=5, reference_date=REF)
    assert [a["agent_id"] for a in ranking["agents"]] == ["a2", "a1"]
    assert ranking["agents"][0]["amount"] == 300.0 and ranking["dimension"] == "broken"


def test_ranking_respects_the_limit_and_rejects_unknown_dimensions():
    rows = [cross_row(f"a{i}", "P", 40, 1, float(i)) for i in range(1, 6)]
    assert len(agent_ranking(rows, "exceptions", CONFIG, 2, REF)["agents"]) == 2
    with pytest.raises(ValueError):
        dimension_ids(CONFIG, "profit")


# -- plan stages ------------------------------------------------------------------


def stage_row(portfolio, stage, qty, amount):
    return {"portfolio_name": portfolio, "stage": stage, "qty": qty, "open_amount": amount}


def test_stage_totals_and_shares():
    rows = [
        stage_row("P-1", "early", 3, 300.0),
        stage_row("P-1", "late", 1, 50.0),
        stage_row("P-2", "early", 1, 100.0),
        stage_row("P-2", "settled", 3, 0.0),
    ]
    summary = stage_summary(rows, None, CONFIG, REF)
    assert summary["total_agreements"] == 8
    assert summary["stages"]["early"] == {"qty": 4, "open_amount": 400.0, "share_pct": 50.0}
    assert set(summary["stages"]) == set(STAGES) and "portfolios" not in summary


def test_a_stage_filter_lists_the_portfolios_holding_the_most_open_amount():
    rows = [
        stage_row("P-1", "late", 2, 80.0),
        stage_row("P-2", "late", 1, 300.0),
        stage_row("P-2", "early", 3, 10.0),
        stage_row("P-3", "early", 1, 5.0),
    ]
    summary = stage_summary(rows, "late", CONFIG, REF)
    assert [p["portfolio_name"] for p in summary["portfolios"]] == ["P-2", "P-1"]
    assert summary["portfolios"][0]["share_of_portfolio_agreements_pct"] == 25.0


def test_unknown_stage_rows_are_ignored():
    assert (
        stage_summary([stage_row("P", "mystery", 5, 1.0)], None, CONFIG, REF)["total_agreements"]
        == 0
    )


# -- drill-downs ------------------------------------------------------------------


def test_generation_ranking_reports_total_and_truncation():
    rows = [
        {"portfolio_name": f"P-{i}", "status_id": 10, "qty": 1, "amount": float(i)}
        for i in range(1, 6)
    ]
    result = generation_ranking(rows, CONFIG, "auto", WINDOW, limit=3)
    assert result["total_portfolios_in_period"] == 5 and result["portfolios_returned"] == 3
    assert result["truncated"] is True
    assert [r["portfolio"] for r in result["ranking"]] == ["P-5", "P-4", "P-3"]


def maturity_row(key, generated, paid, maturing, received):
    return {
        "group_key": key,
        "generated": generated,
        "paid_on_time": paid,
        "amount_maturing": maturing,
        "amount_received": received,
    }


def test_maturities_are_ranked_by_amount_maturing_with_their_own_effectiveness():
    rows = [maturity_row("P-1", 10, 5, 100.0, 40.0), maturity_row("P-2", 4, 4, 900.0, 900.0)]
    result = maturities_ranking(rows, "auto", WINDOW, limit=10)
    assert [r["portfolio"] for r in result["ranking"]] == ["P-2", "P-1"]
    assert result["ranking"][1]["effectiveness_pct"] == 50.0


def test_maturities_summary_of_an_empty_result_is_all_zeros():
    result = maturities_summary([], "P-1", "auto", WINDOW)
    assert result["slips_issued"] == 0 and result["effectiveness_pct"] == 0.0


def test_row_paging_never_carries_personal_fields():
    rows = [
        {
            "agreement_id": i,
            "portfolio_id": 1,
            "first_installment_value": 10.0,
            "total_value": 30.0,
            "agent_name": "Agent 01",
            "agent_id": "agent_01",
            "issued_on": "2026-03-10 09:00:00",
            "due_on": "2026-03-15",
            "installments": 3,
            "debtor_name": "not a real person",
        }
        for i in range(60)
    ]
    page = page_rows(rows, "generated", "P-1", page=2, page_size=25)
    assert len(page["data"]) == 25 and page["meta"]["total_rows"] == 60
    assert page["meta"] == {
        "page": 2,
        "page_size": 25,
        "total_rows": 60,
        "total_pages": 3,
        "truncated": True,
    }
    assert "debtor_name" not in page["data"][0]
    assert page["data"][0]["issued_on"] == "2026-03-10"


# -- slip effectiveness -----------------------------------------------------------


def test_effectiveness_is_slips_paid_within_grace_over_slips_issued():
    rows = [maturity_row("2026-03-10", 4, 3, 0, 0), maturity_row("2026-03-11", 5, 2, 0, 0)]
    points = effectiveness_points(rows, "day")
    assert points[0] == {
        "date": "2026-03-10",
        "slips_issued": 4,
        "paid_within_grace": 3,
        "effectiveness_pct": 75.0,
    }
    assert effectiveness_total(points) == {
        "slips_issued": 9,
        "paid_within_grace": 5,
        "effectiveness_pct": 55.56,
    }


def test_effectiveness_points_never_use_the_conversion_field_names():
    point = effectiveness_points([maturity_row("2026-03-10", 1, 1, 0, 0)], "day")[0]
    assert "conversion_pct" not in point and "effectiveness_pct" in point


def test_effectiveness_rebuckets_by_week():
    rows = [maturity_row("2026-03-09", 2, 1, 0, 0), maturity_row("2026-03-12", 2, 2, 0, 0)]
    assert effectiveness_points(rows, "week") == [
        {"date": "2026-03-09", "slips_issued": 4, "paid_within_grace": 3, "effectiveness_pct": 75.0}
    ]


# -- daily pace -------------------------------------------------------------------

PROFILE = {9: 2, 10: 3, 11: 3, 12: 1, 14: 2, 15: 3, 16: 2, 17: 1}  # 17 a day


def hourly_source(history_days=5, today_counts=None):
    rows = [
        {"day": f"2026-03-{d:02d}", "hour": h, "qty": q}
        for d in range(17 - history_days + 1, 18)
        for h, q in PROFILE.items()
    ]
    rows += [{"day": "2026-03-18", "hour": h, "qty": q} for h, q in (today_counts or {}).items()]
    return FixtureSource({"hourly_counts": rows})


def test_pace_compares_today_with_the_mean_hourly_profile():
    source = hourly_source(today_counts={9: 2, 10: 2, 11: 4, 12: 1, 14: 1})
    result = BaselinePace().forecast(source, datetime(2026, 3, 18, 15, 30), CONFIG)
    assert result["expected_total"] == 17.0
    assert result["expected_to_date"] == 11.0  # hours 9..14
    assert result["actual_to_date"] == 10.0
    assert result["on_pace_pct"] == 90.91
    assert result["projected_close"] == 15.45  # 10 / (11 / 17)
    assert result["history_days"] == 5


def test_the_current_hour_is_in_progress_and_not_part_of_the_comparison():
    source = hourly_source(today_counts={15: 99})
    result = BaselinePace().forecast(source, datetime(2026, 3, 18, 15, 30), CONFIG)
    band = next(b for b in result["bands"] if b["hour"] == 15)
    assert band["in_progress"] is True and result["actual_to_date"] == 0.0


def test_the_projection_is_withheld_early_in_the_day():
    source = hourly_source(today_counts={})
    result = BaselinePace().forecast(source, datetime(2026, 3, 18, 9, 5), CONFIG)
    assert result["projected_close"] is None


def test_too_little_history_is_a_structured_error_and_not_a_guess():
    source = hourly_source(history_days=2)
    result = BaselinePace().forecast(source, datetime(2026, 3, 18, 15, 30), CONFIG)
    assert result["ok"] is False and result["error_type"] == "empty_result"
    assert result["history_days"] == 2


def test_a_failing_source_propagates_so_the_guard_can_retry():
    source = FixtureSource({"hourly_counts": {"$fail": "timeout"}})
    with pytest.raises(DataSourceError):
        BaselinePace().forecast(source, datetime(2026, 3, 18, 15, 30), CONFIG)
