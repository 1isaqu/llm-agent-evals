from datetime import date

import pytest

from llm_agent_evals.business import BusinessConfig
from llm_agent_evals.errors import ConfigError, DataSourceError, DataUnavailable
from llm_agent_evals.session import Window
from llm_agent_evals.sources import (
    SQLITE,
    SQLSERVER,
    CachedSource,
    DataSource,
    FixtureSource,
    MultiLineSource,
    SqlDataSource,
    build_connection_string,
    connect_sqlserver,
    merge_agent_rows,
    merged_sources,
)
from llm_agent_evals.sources.demo import DEMO_NOW, build_demo_connection, build_demo_connections
from llm_agent_evals.sources.schema import CONTRACT_TABLES
from llm_agent_evals.sources.sqlsource import sources_from_connections
from tests.helpers import WINDOW

# -- FixtureSource -------------------------------------------------------------------


def test_a_plain_list_is_returned_for_every_call():
    source = FixtureSource(
        {"status_rollup": [{"portfolio_name": "P", "status_id": 10, "qty": 1, "amount": 5}]}
    )
    assert source.status_rollup(WINDOW) == source.status_rollup(
        Window.from_iso("2026-01-01", "2026-01-02")
    )


def test_rows_are_copies():
    source = FixtureSource({"status_rollup": [{"amount": 1}]})
    source.status_rollup(WINDOW)[0]["amount"] = 99
    assert source.status_rollup(WINDOW)[0]["amount"] == 1


def test_blocks_are_matched_on_call_parameters_first_match_wins():
    source = FixtureSource(
        {
            "crosstab_rollup": [
                {"match": {"by": "portfolio", "value": "P-1"}, "rows": [{"id": 1}]},
                {"match": {"by": "portfolio"}, "rows": [{"id": 2}]},
            ]
        }
    )
    assert source.crosstab_rollup(WINDOW, "portfolio", "p-1") == [{"id": 1}]  # case-insensitive
    assert source.crosstab_rollup(WINDOW, "portfolio", "P-9") == [{"id": 2}]
    assert source.crosstab_rollup(WINDOW, "agent", "x") == []  # no block matches


def test_blocks_can_match_the_window():
    source = FixtureSource(
        {
            "status_rollup": [
                {"match": {"start": "2026-03-17", "end": "2026-03-17"}, "rows": [{"id": "day"}]},
                {"match": {"start": "2026-03-01"}, "rows": [{"id": "month"}]},
            ]
        }
    )
    assert source.status_rollup(Window.from_iso("2026-03-17", "2026-03-17")) == [{"id": "day"}]
    assert source.status_rollup(WINDOW) == [{"id": "month"}]


def test_rows_with_a_day_are_filtered_to_the_window():
    source = FixtureSource({"daily_rollup": [{"day": "2026-03-05"}, {"day": "2026-04-05"}]})
    assert source.daily_rollup(WINDOW) == [{"day": "2026-03-05"}]


def test_a_none_match_value_requires_a_missing_argument():
    source = FixtureSource(
        {
            "slip_rollup": [
                {"match": {"portfolio": None}, "rows": [{"id": "all"}]},
                {"match": {"portfolio": "P-1"}, "rows": [{"id": "one"}]},
            ]
        }
    )
    assert source.slip_rollup(WINDOW, "none", "due_at") == [{"id": "all"}]
    assert source.slip_rollup(WINDOW, "none", "due_at", "P-1") == [{"id": "one"}]


def test_a_fail_section_simulates_a_transient_upstream_error():
    source = FixtureSource({"agent_rollup": {"$fail": "connection reset"}})
    with pytest.raises(DataSourceError, match="connection reset"):
        source.agent_rollup(WINDOW)


def test_a_missing_section_is_data_unavailable_not_an_empty_result():
    with pytest.raises(DataUnavailable):
        FixtureSource({}).hourly_counts(WINDOW)


def test_unknown_sections_are_rejected_at_construction():
    with pytest.raises(ValueError, match="unknown fixture sections"):
        FixtureSource({"status_rolup": []})


def test_the_fixture_satisfies_the_data_source_protocol():
    assert isinstance(FixtureSource({}), DataSource)
    assert FixtureSource({"plan_stage_rollup": []}).plan_stage_rollup(date(2026, 3, 18)) == []


# -- wrappers -------------------------------------------------------------------------


class Counting:
    def __init__(self):
        self.calls = 0

    def status_rollup(self, window):
        self.calls += 1
        return [{"n": self.calls}]


def test_the_cache_serves_repeated_calls_until_the_ttl_expires():
    now = [0.0]
    inner = Counting()
    cached = CachedSource(inner, ttl_seconds=60, clock=lambda: now[0])
    assert cached.status_rollup(WINDOW) == [{"n": 1}]
    assert cached.status_rollup(WINDOW) == [{"n": 1}] and inner.calls == 1
    now[0] = 61
    assert cached.status_rollup(WINDOW) == [{"n": 2}]


def test_the_cache_key_includes_the_arguments():
    inner = Counting()
    cached = CachedSource(inner, ttl_seconds=60)
    cached.status_rollup(WINDOW)
    cached.status_rollup(Window.from_iso("2026-02-01", "2026-02-02"))
    assert inner.calls == 2


def test_cached_rows_cannot_be_corrupted_by_a_caller():
    cached = CachedSource(Counting(), ttl_seconds=60)
    cached.status_rollup(WINDOW)[0]["n"] = 999
    assert cached.status_rollup(WINDOW) == [{"n": 1}]


def test_a_zero_ttl_never_caches_and_clear_empties_the_store():
    inner = Counting()
    never = CachedSource(inner, ttl_seconds=0)
    never.status_rollup(WINDOW), never.status_rollup(WINDOW)
    assert inner.calls == 2
    some = CachedSource(inner, ttl_seconds=60)
    some.status_rollup(WINDOW)
    some.clear()
    some.status_rollup(WINDOW)
    assert inner.calls == 4


def test_the_cache_exposes_only_source_methods():
    with pytest.raises(AttributeError):
        getattr(CachedSource(Counting()), "delete_everything")  # noqa: B009


def test_multi_line_sources_concatenate_rows_and_never_merge_ratios():
    one = FixtureSource({"status_rollup": [{"portfolio_name": "A", "amount": 1}]})
    two = FixtureSource({"status_rollup": [{"portfolio_name": "B", "amount": 2}]})
    merged = MultiLineSource([one, two])
    assert [r["portfolio_name"] for r in merged.status_rollup(WINDOW)] == ["A", "B"]
    with pytest.raises(ValueError):
        MultiLineSource([])


def test_merged_sources_returns_the_single_source_unwrapped():
    only = FixtureSource({})
    assert merged_sources({"auto": only, "consumer": only}, ["auto"]) is only
    assert isinstance(
        merged_sources({"auto": only, "consumer": only}, ["auto", "consumer"]), MultiLineSource
    )


def test_merge_agent_rows_combines_the_separate_aggregates_per_agent():
    effort = [{"agent_id": "agent_01", "agent_name": "Agent 01", "attempt_count": 10}]
    money = [
        {"agent_id": " agent_01 ", "agreement_count": 2},
        {"agent_id": "agent_02", "agreement_count": 1},
    ]
    merged = merge_agent_rows(effort, money)
    assert [r["agent_id"] for r in merged] == ["agent_01", "agent_02"]
    assert merged[0]["attempt_count"] == 10 and merged[0]["agreement_count"] == 2
    assert merged[1]["attempt_count"] == 0 and merged[1]["agent_name"] == ""


def test_merge_agent_rows_skips_rows_without_an_identity():
    assert merge_agent_rows([{"agent_id": "", "attempt_count": 1}, {"attempt_count": 2}]) == []


# -- SqlDataSource --------------------------------------------------------------------


class RaisingConnection:
    def cursor(self):
        raise RuntimeError("driver exploded: host db.example.invalid")


def test_driver_failures_become_data_source_errors_that_keep_the_cause():
    source = SqlDataSource(RaisingConnection(), SQLITE, BusinessConfig())
    with pytest.raises(DataSourceError) as caught:
        source.status_rollup(WINDOW)
    assert isinstance(caught.value.__cause__, RuntimeError)


def test_a_statement_that_is_not_read_only_is_refused_before_it_reaches_the_driver():
    class Spy:
        executed = []

        def cursor(self):
            raise AssertionError("must not be reached")

    source = SqlDataSource(Spy(), SQLITE, BusinessConfig())
    from llm_agent_evals.sources.sql import Query

    with pytest.raises(ConfigError, match="non read-only"):
        source._run(Query("DELETE FROM agreements", ()))


def test_catalog_check_reports_ids_that_the_database_does_not_have():
    conn = build_demo_connection("auto")
    assert SqlDataSource(conn, SQLITE, BusinessConfig()).catalog_problems() == []
    conn.execute("UPDATE agreement_statuses SET label = 'SOMETHING ELSE' WHERE status_id = 10")
    conn.execute("DELETE FROM agreement_statuses WHERE status_id = 20")
    problems = SqlDataSource(conn, SQLITE, BusinessConfig()).catalog_problems()
    assert any("ACTIVE" in p and "SOMETHING ELSE" in p for p in problems)
    assert any("BROKEN" in p and "not in the catalogue" in p for p in problems)


def test_sources_from_connections_builds_one_source_per_line():
    sources = sources_from_connections(build_demo_connections(), SQLITE, BusinessConfig())
    assert set(sources) == {"auto", "consumer"}


# -- SQL Server connection helper ---------------------------------------------------------


def test_the_connection_string_escapes_special_characters():
    text = build_connection_string(
        "ODBC Driver 17 for SQL Server", "db.example.invalid", "demo", "reader", "p;w}=x"
    )
    assert "DRIVER={ODBC Driver 17 for SQL Server}" in text
    assert "PWD={p;w}}=x}" in text
    assert text.endswith("ApplicationIntent=ReadOnly")


ENV = {
    "DB_HOST": "db.example.invalid",
    "DB_NAME": "demo",
    "DB_USER": "reader",
    "DB_PASSWORD": "change-me",
}


def test_connect_uses_the_environment_and_an_injected_driver():
    seen = {}

    def fake_connect(connection_string, **kwargs):
        seen["cs"], seen["kwargs"] = connection_string, kwargs
        return "connection"

    assert connect_sqlserver(ENV, connect=fake_connect) == "connection"
    assert "SERVER=db.example.invalid" in seen["cs"] and "DATABASE=demo" in seen["cs"]
    assert seen["kwargs"]["autocommit"] is True


def test_a_database_per_product_line_is_read_from_its_own_variable():
    seen = []
    env = {**ENV, "DB_NAME_AUTO": "auto_db"}
    connect_sqlserver(env, "auto", lambda cs, **kw: seen.append(cs))
    connect_sqlserver(env, "consumer", lambda cs, **kw: seen.append(cs))
    assert "DATABASE=auto_db" in seen[0] and "DATABASE=demo" in seen[1]


def test_missing_settings_are_named_and_the_password_never_appears():
    with pytest.raises(ConfigError) as caught:
        connect_sqlserver({"DB_PASSWORD": "super-secret-value"}, connect=lambda *a, **k: None)
    message = str(caught.value)
    assert "DB_HOST" in message and "DB_USER" in message
    assert "super-secret-value" not in message


def test_a_missing_pyodbc_is_a_clear_configuration_error(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def deny(name, *args, **kwargs):
        if name == "pyodbc":
            raise ImportError("no pyodbc")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", deny)
    with pytest.raises(ConfigError, match="pyodbc is not installed"):
        connect_sqlserver(ENV)


def test_the_sql_server_dialect_is_selected_for_the_helper_source():
    class Cursor:
        description = [("portfolio_name",), ("status_id",), ("qty",), ("amount",)]
        sql = ""

        def execute(self, sql, params):
            Cursor.sql = sql

        def fetchall(self):
            return []

        def close(self):
            pass

    class Conn:
        def cursor(self):
            return Cursor()

    SqlDataSource(Conn(), SQLSERVER, BusinessConfig()).status_rollup(WINDOW)
    assert "WITH (NOLOCK)" in Cursor.sql


# -- the demo data --------------------------------------------------------------------


def test_the_demo_databases_have_every_contract_table_and_are_deterministic():
    first, second = build_demo_connection("auto"), build_demo_connection("auto")
    for table in CONTRACT_TABLES:
        a = first.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        b = second.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        assert a == b and a > 0, table


def test_the_demo_uses_only_invented_agents_and_portfolios():
    conn = build_demo_connection("consumer")
    logins = {r[0] for r in conn.execute("SELECT login FROM agents")}
    assert logins == {"agent_01", "agent_02", "agent_03", "agent_04", "system_bot", "test_user"}
    labels = {r[0] for r in conn.execute("SELECT DISTINCT portfolio_label FROM debt_attributes")}
    assert labels == {"Portfolio B-01", "Portfolio B-02"}


def test_the_demo_exercises_the_dedup_and_exclusion_rules():
    conn = build_demo_connection("auto")
    rewritten = conn.execute(
        "SELECT COUNT(*) FROM (SELECT agreement_id FROM agreements WHERE installment_no = 0 "
        "GROUP BY agreement_id, portfolio_id HAVING COUNT(*) > 1)"
    ).fetchone()[0]
    assert rewritten >= 1
    source = SqlDataSource(conn, SQLITE, BusinessConfig())
    agents = {
        r["agent_id"] for r in source.agent_rollup(Window.from_iso("2026-03-01", "2026-03-18"))
    }
    assert agents <= {"agent_01", "agent_02", "agent_03", "agent_04"}


def test_the_demo_funnel_is_monotonic_up_to_cpc():
    source = SqlDataSource(build_demo_connection("auto"), SQLITE, BusinessConfig())
    for row in source.agent_rollup(Window.from_iso("2026-03-01", "2026-03-18")):
        assert row["attempt_count"] >= row["answer_count"] >= row["cpc_count"]


def test_demo_now_is_a_fixed_instant():
    assert DEMO_NOW.isoformat() == "2026-03-18T15:30:00"


def test_an_unknown_demo_line_is_rejected():
    with pytest.raises(ValueError):
        build_demo_connection("nope")
