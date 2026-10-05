from datetime import date

import pytest

from llm_agent_evals.analytics.series import (
    bucket_key,
    daily_points,
    paginate,
    rebucket,
    trend,
)
from llm_agent_evals.business import BusinessConfig
from llm_agent_evals.session import Window

CONFIG = BusinessConfig()
WINDOW = Window.from_iso("2026-03-09", "2026-03-15")  # Monday to Sunday


def daily(day, status, qty, amount):
    return {"day": day, "status_id": status, "qty": qty, "amount": amount}


def test_days_without_rows_become_zero_so_the_series_is_continuous():
    points = daily_points([daily("2026-03-10", 10, 2, 100.0)], CONFIG, "value", WINDOW)
    assert len(points) == 7
    assert [p["value"] for p in points] == [0.0, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    count_points = daily_points([daily("2026-03-10", 10, 2, 100.0)], CONFIG, "count", WINDOW)
    assert count_points[1]["value"] == 2 and count_points[0]["value"] == 0


def test_only_generated_statuses_count_toward_value_and_count():
    rows = [daily("2026-03-10", 10, 2, 100.0), daily("2026-03-10", 40, 5, 900.0)]
    point = daily_points(rows, CONFIG, "value", WINDOW)[1]
    assert point["value"] == 100.0


def test_the_daily_risk_is_the_worst_axis_of_that_day():
    rows = [daily("2026-03-10", 10, 1, 100.0), daily("2026-03-10", 40, 1, 100.0)]
    point = daily_points(rows, CONFIG, "risk", WINDOW)[1]
    assert point["value"] == 50.0


def test_unknown_metric_is_rejected():
    with pytest.raises(ValueError):
        daily_points([], CONFIG, "volume", WINDOW)


@pytest.mark.parametrize(
    ("values", "direction", "change"),
    [
        ([100, 100, 100, 100], "flat", 0.0),
        ([100, 100, 120, 120], "rising", 20.0),
        ([100, 100, 80, 80], "falling", -20.0),
        ([100, 100, 104, 104], "flat", 4.0),
        ([0, 0, 5, 5], "rising", None),
    ],
)
def test_trend_compares_the_second_half_with_the_first(values, direction, change):
    points = [{"date": f"2026-03-{i + 1:02d}", "value": v} for i, v in enumerate(values)]
    assert trend(points) == (direction, change)


def test_a_single_point_has_no_trend():
    assert trend([{"date": "2026-03-01", "value": 5}]) == ("flat", None)


def test_weeks_start_on_monday_and_months_on_the_first():
    assert bucket_key(date(2026, 3, 15), "week") == "2026-03-09"
    assert bucket_key(date(2026, 3, 16), "week") == "2026-03-16"
    assert bucket_key(date(2026, 3, 31), "month") == "2026-03-01"
    assert bucket_key(date(2026, 3, 31), "day") == "2026-03-31"


def test_rebucketing_sums_values_and_counts_but_takes_the_worst_risk():
    points = [
        {"date": "2026-03-09", "value": 10.0},
        {"date": "2026-03-10", "value": 30.0},
        {"date": "2026-03-16", "value": 5.0},
    ]
    assert rebucket(points, "generated_value", "week") == [
        {"date": "2026-03-09", "value": 40.0},
        {"date": "2026-03-16", "value": 5.0},
    ]
    assert rebucket(points, "agreement_count", "week")[0]["value"] == 40
    assert rebucket(points, "composite_risk_pct", "week")[0]["value"] == 30.0


def test_day_granularity_is_returned_unchanged():
    points = [{"date": "2026-03-09", "value": 1}]
    assert rebucket(points, "generated_value", "day") is points


def test_paging_reports_the_real_total_and_whether_it_is_truncated():
    points = [{"date": f"d{i}", "value": i} for i in range(70)]
    first = paginate(points, 1, page_size=31)
    assert len(first["data"]) == 31
    assert first["meta"] == {"page": 1, "total_pages": 3, "total_points": 70, "truncated": True}
    last = paginate(points, 3, page_size=31)
    assert len(last["data"]) == 8 and last["meta"]["truncated"] is False
    assert paginate([], 1)["meta"]["total_pages"] == 1
