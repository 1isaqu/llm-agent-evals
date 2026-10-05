"""Every tool, called through the registry the way the agent calls it."""

import pytest

from llm_agent_evals.business import BusinessConfig
from llm_agent_evals.tools import build_default_registry
from llm_agent_evals.tools.common import (
    ambiguity_warning,
    find_agent,
    find_agent_exact,
    find_portfolio,
    matches_excluded,
    window_errors,
)
from tests.helpers import P1, P2, base_sections, make_context, status_row

REGISTRY = build_default_registry()


def cross(agent, portfolio, status, qty, amount):
    return {
        "agent_id": agent,
        "agent_name": agent.title(),
        "portfolio_name": portfolio,
        "status_id": status,
        "qty": qty,
        "amount": amount,
    }


def sections(**overrides):
    base = base_sections(
        status_rollup=[
            status_row(P1, 10, 4, 400.0),
            status_row(P1, 20, 1, 100.0),
            status_row(P1, 40, 1, 100.0),
            status_row(P2, 10, 2, 200.0),
            status_row(P2, 50, 1, 100.0),
            status_row("Portfolio T-10", 10, 1, 50.0),
        ],
        agent_status_rollup=[
            {
                "agent_id": "agent_01",
                "agent_name": "Agent 01",
                "status_id": 20,
                "qty": 1,
                "amount": 100.0,
            },
            {
                "agent_id": "agent_02",
                "agent_name": "Agent 02",
                "status_id": 20,
                "qty": 2,
                "amount": 300.0,
            },
        ],
        crosstab_rollup=[
            {
                "match": {"by": "portfolio", "value": P1},
                "rows": [cross("agent_01", P1, 10, 4, 400.0), cross("agent_01", P1, 40, 1, 100.0)],
            },
            {
                "match": {"by": "agent", "value": "agent_01"},
                "rows": [cross("agent_01", P1, 10, 4, 400.0)],
            },
        ],
        plan_stage_rollup=[{"portfolio_name": P1, "stage": "late", "qty": 2, "open_amount": 90.0}],
        daily_rollup=[
            {"day": "2026-03-16", "status_id": 10, "qty": 2, "amount": 200.0},
            {"day": "2026-03-17", "status_id": 10, "qty": 1, "amount": 100.0},
        ],
        slip_rollup=[
            {
                "match": {"group_by": "day"},
                "rows": [
                    {
                        "group_key": "2026-03-16",
                        "generated": 4,
                        "paid_on_time": 2,
                        "amount_maturing": 0.0,
                        "amount_received": 0.0,
                    }
                ],
            },
            {
                "match": {"group_by": "portfolio"},
                "rows": [
                    {
                        "group_key": P1,
                        "generated": 5,
                        "paid_on_time": 4,
                        "amount_maturing": 500.0,
                        "amount_received": 400.0,
                    }
                ],
            },
            {
                "match": {"group_by": "none"},
                "rows": [
                    {
                        "group_key": "all",
                        "generated": 5,
                        "paid_on_time": 4,
                        "amount_maturing": 500.0,
                        "amount_received": 400.0,
                    }
                ],
            },
        ],
        drilldown_rows=[
            {
                "match": {"kind": "exception", "portfolio": P1},
                "rows": [
                    {
                        "agreement_id": i,
                        "portfolio_id": 1,
                        "first_installment_value": 10.0 * i,
                        "total_value": 30.0 * i,
                        "agent_name": "Agent 01",
                        "agent_id": "agent_01",
                        "issued_on": "2026-03-10",
                        "due_on": "2026-03-15",
                        "installments": 3,
                    }
                    for i in range(1, 31)
                ],
            }
        ],
        hourly_counts=[{"day": f"2026-03-{d:02d}", "hour": 10, "qty": 2} for d in range(10, 18)],
    )
    base.update(overrides)
    return base


def run(tool, args=None, **ctx):
    sec = ctx.pop("sections", None) or sections()
    return REGISTRY.call(tool, args or {}, make_context(sec, **ctx))


def is_error(result, kind):
    return isinstance(result, dict) and result.get("ok") is False and result["error_type"] == kind


# -- name resolution helpers ------------------------------------------------------------------


def test_portfolio_lookup_prefers_the_exact_name_then_the_first_substring():
    entries = [{"portfolio_name": "Portfolio T-10"}, {"portfolio_name": "Portfolio T-1"}]
    assert find_portfolio("portfolio t-1", entries)["portfolio_name"] == "Portfolio T-1"
    assert find_portfolio("t-1", entries)["portfolio_name"] == "Portfolio T-10"
    assert find_portfolio("", entries) is None and find_portfolio("zzz", entries) is None


def test_the_ambiguity_warning_names_the_other_matches_only_for_fragments():
    entries = [{"portfolio_name": "Portfolio T-01"}, {"portfolio_name": "Portfolio T-02"}]
    assert "Portfolio T-02" in ambiguity_warning("t-0", entries[0], entries)
    assert ambiguity_warning("Portfolio T-01", entries[0], entries) is None
    assert ambiguity_warning("t-01", entries[0], entries) is None


def test_agent_lookup_has_an_exact_only_variant_for_comparisons():
    agents = [
        {"agent_name": "Agent 01", "agent_id": "agent_01"},
        {"agent_name": "Agent 011", "agent_id": "agent_011"},
    ]
    assert find_agent_exact("AGENT_01", agents)["agent_id"] == "agent_01"
    assert find_agent_exact("agent_0", agents) is None
    assert find_agent("agent_0", agents)["agent_id"] == "agent_01"
    assert find_agent_exact("", agents) is None


def test_exclusion_patterns_use_sql_like_wildcards():
    assert matches_excluded("System_Bot", ("system%", "test%"))
    assert matches_excluded("test", ("test%",))
    assert not matches_excluded("agent_01", ("system%", "test%"))
    assert not matches_excluded("my_system", ("system%",))


# -- portfolio tools --------------------------------------------------------------------------


def test_get_portfolio_metrics_by_exact_name_and_by_fragment_with_a_warning():
    exact = run("get_portfolio_metrics", {"portfolio_name": P1})
    assert exact["portfolio_name"] == P1 and "ambiguity_warning" not in exact
    fuzzy = run("get_portfolio_metrics", {"portfolio_name": "t-0"})
    assert fuzzy["portfolio_name"] == P1
    assert P2 in fuzzy["ambiguity_warning"]


def test_an_unknown_portfolio_lists_the_real_ones():
    result = run("get_portfolio_metrics", {"portfolio_name": "Nope"})
    assert is_error(result, "not_found") and P1 in result["available_portfolios"]
    assert result["retryable"] is False


def test_filter_by_risk_returns_only_that_level():
    low = run("filter_portfolios_by_risk", {"level": "low"})
    assert {e["risk_level"] for e in low} == {"low"}
    assert is_error(run("filter_portfolios_by_risk", {"level": "extreme"}), "validation")


def test_filter_by_value_applies_the_minimum_and_the_limit():
    result = run("filter_portfolios_by_value", {"min_value": 100, "limit": 1})
    assert [e["portfolio_name"] for e in result] == [P1]
    assert len(run("filter_portfolios_by_value", {"min_value": 0})) == 3
    assert is_error(run("filter_portfolios_by_value", {"min_value": -1}), "validation")


def test_compare_portfolios_reports_found_missing_and_ambiguous_names():
    result = run(
        "compare_portfolios", {"names": [P1, P1, "Ghost", "t-0"], "metric": "composite_risk_pct"}
    )
    assert [p["portfolio_name"] for p in result["portfolios"]].count(P1) == 1
    assert result["not_found"] == ["Ghost"] and "available_portfolios" in result
    assert result["metric"] == "composite_risk_pct"
    assert "ambiguity_warnings" in result


def test_status_breakdown_separates_generated_from_the_rest():
    result = run("get_status_breakdown")
    assert result["generated_amount"] == 750.0 and result["all_statuses_amount"] == 950.0


def test_plan_stage_distribution_with_and_without_a_stage():
    plain = run("get_plan_stage_distribution")
    assert plain["total_agreements"] == 2 and "portfolios" not in plain
    late = run("get_plan_stage_distribution", {"stage": "late"})
    assert late["portfolios"][0]["portfolio_name"] == P1
    assert is_error(run("get_plan_stage_distribution", {"stage": "final"}), "validation")


# -- agent tools -------------------------------------------------------------------------------


def test_agent_performance_found_and_not_found():
    entry = run("get_agent_performance", {"agent_name": "agent 01"})
    assert entry["agent_id"] == "agent_01" and entry["conversion_pct"] == 25.0
    missing = run("get_agent_performance", {"agent_name": "ghost"})
    assert is_error(missing, "not_found") and "Agent 02" in missing["available_agents"]


def test_agents_are_ranked_by_the_requested_metric_and_the_enum_is_enforced():
    by_cpc = run("list_agents_performance", {"order_by": "cpc_count", "limit": 1})
    assert [a["agent_id"] for a in by_cpc] == ["agent_02"]
    assert is_error(run("list_agents_performance", {"order_by": "salary"}), "validation")
    assert len(run("list_agents_performance")) == 2


def test_the_crosstab_needs_exactly_one_side():
    assert is_error(run("get_agent_portfolio_crosstab", {}), "validation")
    both = run("get_agent_portfolio_crosstab", {"portfolio": P1, "agent_name": "agent_01"})
    assert is_error(both, "validation")


def test_the_crosstab_by_portfolio_and_by_agent():
    by_portfolio = run("get_agent_portfolio_crosstab", {"portfolio": P1})
    assert (
        by_portfolio["detail_by"] == "agent"
        and by_portfolio["entries"][0]["exception_value"] == 100.0
    )
    by_agent = run("get_agent_portfolio_crosstab", {"agent_name": "agent_01"})
    assert by_agent["detail_by"] == "portfolio" and by_agent["entries"][0]["portfolio_name"] == P1
    assert is_error(run("get_agent_portfolio_crosstab", {"portfolio": "Ghost"}), "not_found")
    assert is_error(run("get_agent_portfolio_crosstab", {"agent_name": "ghost"}), "not_found")


def test_an_empty_crosstab_says_so_instead_of_returning_a_bare_list():
    result = run("get_agent_portfolio_crosstab", {"portfolio": P2})
    assert result["entries"] == [] and "No agreements" in result["note"]


def test_agent_ranking_by_dimension():
    result = run("get_agent_ranking_by_dimension", {"dimension": "broken", "limit": 5})
    assert [a["agent_id"] for a in result["agents"]] == ["agent_02", "agent_01"]
    assert is_error(run("get_agent_ranking_by_dimension", {"dimension": "profit"}), "validation")


COMPARE = {
    "product_line": "auto",
    "agent_ids": ["agent_01", "agent_02"],
    "metrics": ["conversion_pct", "cpc_count"],
    "date_from": "2026-03-01",
    "date_to": "2026-03-18",
}


def test_compare_agents_returns_only_the_requested_metrics():
    result = run("compare_agents", COMPARE)
    assert result["agents"][1] == {
        "agent_name": "Agent 02",
        "agent_id": "agent_02",
        "conversion_pct": 12.5,
        "cpc_count": 40,
    }


def test_compare_agents_matches_ids_exactly_and_refuses_system_accounts():
    guessed = run("compare_agents", {**COMPARE, "agent_ids": ["agent_0", "agent_02"]})
    assert is_error(guessed, "validation") and "agent_0" in guessed["hint"]
    system = run("compare_agents", {**COMPARE, "agent_ids": ["agent_01", "system_bot"]})
    assert "system account" in system["hint"]


def test_compare_agents_rejects_duplicates_and_bad_windows():
    assert is_error(
        run("compare_agents", {**COMPARE, "agent_ids": ["agent_01", "AGENT_01"]}), "validation"
    )
    assert is_error(
        run("compare_agents", {**COMPARE, "date_from": "2026-03-18", "date_to": "2026-03-01"}),
        "validation",
    )
    assert is_error(run("compare_agents", {**COMPARE, "date_to": "2026-03-19"}), "validation")


def test_consolidating_across_lines_is_refused_with_a_warning_not_faked():
    result = run("compare_agents", {**COMPARE, "consolidate_cross_line": True})
    assert "NOT summed" in result["meta"]["warnings"][0]


# -- time tools --------------------------------------------------------------------------------


HISTORY = {
    "product_line": "auto",
    "kpi": "generated_value",
    "date_from": "2026-03-12",
    "date_to": "2026-03-18",
}


def test_history_returns_continuous_daily_points_with_a_trend():
    result = run("query_kpi_history", HISTORY)
    assert len(result["data"]) == 7 and result["meta"]["total_points"] == 7
    assert result["trend"] in {"rising", "falling", "flat"}
    assert {p["date"]: p["value"] for p in result["data"]}["2026-03-16"] == 200.0


def test_history_can_be_rebucketed_and_paged():
    weekly = run("query_kpi_history", {**HISTORY, "granularity": "week"})
    assert [p["date"] for p in weekly["data"]] == ["2026-03-09", "2026-03-16"]
    wide = run("query_kpi_history", {**HISTORY, "date_from": "2026-01-01"})
    assert wide["meta"]["total_pages"] == 3 and wide["meta"]["truncated"]
    assert (
        run("query_kpi_history", {**HISTORY, "date_from": "2026-01-01", "page": 3})["meta"][
            "truncated"
        ]
        is False
    )


def test_history_can_be_narrowed_to_a_portfolio_and_reports_unknown_ones():
    narrowed = run("query_kpi_history", {**HISTORY, "portfolio": "t-01"})
    assert narrowed["portfolio"] == P1
    assert is_error(run("query_kpi_history", {**HISTORY, "portfolio": "Ghost"}), "not_found")


def test_slip_effectiveness_series_never_uses_conversion_names():
    result = run("query_kpi_history", {**HISTORY, "kpi": "slip_effectiveness"})
    assert result["kpi"] == "slip_effectiveness" and "NOT conversion" in result["criterion"]
    assert result["data"][0]["effectiveness_pct"] == 50.0
    assert "conversion_pct" not in result["data"][0]


def test_daily_pace_through_the_history_tool_ignores_the_window_and_says_so():
    result = run("query_kpi_history", {**HISTORY, "kpi": "daily_pace", "granularity": "month"})
    assert "ignored" in result["meta"]["warnings"][0] and result["date"] == "2026-03-18"


def test_get_daily_pace_reports_missing_history_as_a_structured_error():
    result = run("get_daily_pace", sections=sections(hourly_counts=[]))
    assert is_error(result, "empty_result")
    assert run("get_daily_pace")["date"] == "2026-03-18"


@pytest.mark.parametrize(
    ("args", "fragment"),
    [
        ({"date_from": "2026-03-18", "date_to": "2026-03-01"}, "must not be before"),
        ({"date_to": "2026-03-19"}, "future"),
        ({"date_from": "2025-01-01"}, "more than the 93 allowed"),
    ],
)
def test_window_rules_are_enforced_before_any_query_runs(args, fragment):
    result = run("query_kpi_history", {**HISTORY, **args})
    assert is_error(result, "validation") and fragment in result["hint"]


def test_the_window_limit_comes_from_the_configuration():
    config = BusinessConfig(max_window_days=10)
    result = run("query_kpi_history", {**HISTORY, "date_from": "2026-03-01"}, config=config)
    assert is_error(result, "validation") and "more than the 10 allowed" in result["hint"]
    ctx = make_context(sections(), config=config)
    assert window_errors({"date_from": "2026-03-10", "date_to": "2026-03-18"}, ctx) == []


# -- drill-down and metric explanation ------------------------------------------------------------


DRILL = {"product_line": "auto", "date_from": "2026-03-01", "date_to": "2026-03-18"}


def test_summary_drilldown_matches_the_portfolio_entry():
    result = run("drill_down_portfolio", {**DRILL, "portfolio": P1})
    assert result["portfolio_name"] == P1 and "composite_risk_pct" in result


def test_row_drilldown_is_paged_with_explicit_totals():
    first = run(
        "drill_down_portfolio",
        {**DRILL, "portfolio": P1, "drilldown": "exception", "page_size": 10},
    )
    assert (
        len(first["data"]) == 10
        and first["meta"]["total_rows"] == 30
        and first["meta"]["truncated"]
    )
    last = run(
        "drill_down_portfolio",
        {**DRILL, "portfolio": P1, "drilldown": "exception", "page_size": 10, "page": 3},
    )
    assert last["meta"]["truncated"] is False and last["data"][0]["agreement_id"] == 21


def test_generation_ranking_covers_every_portfolio_in_one_call():
    result = run(
        "drill_down_portfolio", {**DRILL, "drilldown": "generation_ranking", "page_size": 10}
    )
    assert result["total_portfolios_in_period"] == 3 and result["truncated"] is False
    assert result["ranking"][0]["portfolio"] == P1


def test_maturities_for_all_portfolios_or_for_one():
    ranking = run("drill_down_portfolio", {**DRILL, "drilldown": "maturities"})
    assert ranking["ranking"][0]["effectiveness_pct"] == 80.0
    one = run("drill_down_portfolio", {**DRILL, "drilldown": "maturities", "portfolio": P1})
    assert one["slips_issued"] == 5 and one["amount_received"] == 400.0


def test_drilldown_argument_rules():
    assert is_error(run("drill_down_portfolio", {**DRILL, "drilldown": "generated"}), "validation")
    assert is_error(run("drill_down_portfolio", {**DRILL, "drilldown": "summary"}), "validation")
    both = run(
        "drill_down_portfolio", {**DRILL, "drilldown": "generation_ranking", "portfolio": P1}
    )
    assert is_error(both, "validation") and "generated_value" in both["hint"]
    assert is_error(run("drill_down_portfolio", {**DRILL, "portfolio": "Ghost"}), "not_found")
    assert is_error(
        run("drill_down_portfolio", {**DRILL, "portfolio": P1, "page_size": 7}), "validation"
    )


def test_a_fragment_in_a_drilldown_carries_the_ambiguity_warning():
    result = run("drill_down_portfolio", {**DRILL, "portfolio": "t-0"})
    assert "ambiguity_warning" in result


def test_explain_metric_serves_the_registry_and_its_aliases():
    kpi = run("explain_metric", {"term": "Conversion"})
    assert kpi["type"] == "kpi" and "cpc_count" in kpi["formula"]
    assert run("explain_metric", {"term": "rpc"})["type"] == "kpi"
    status = run("explain_metric", {"term": "generated"})
    assert status["type"] == "status" and "caveat" in status
    unknown = run("explain_metric", {"term": "vibes"})
    assert is_error(unknown, "validation") and "conversion_pct" in unknown["available_terms"]


def test_conversion_names_stay_apart_in_the_registry():
    conversion = run("explain_metric", {"term": "conversion_pct"})
    paid = run("explain_metric", {"term": "paid_per_cpc_pct"})
    assert (
        "never call it conversion" in paid["formula"] and conversion["formula"] != paid["formula"]
    )
