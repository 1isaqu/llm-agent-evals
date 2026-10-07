"""The contract queries, executed on in-memory SQLite and checked as text for SQL Server."""

import re
import sqlite3
from datetime import date

import pytest

from llm_agent_evals.business import BusinessConfig
from llm_agent_evals.session import Window
from llm_agent_evals.sources.schema import create_schema
from llm_agent_evals.sources.sql import (
    SQLITE,
    SQLSERVER,
    ContractQueries,
    UnsafeQuery,
    ensure_read_only,
)
from llm_agent_evals.sources.sqlsource import SqlDataSource

WINDOW = Window.from_iso("2026-03-01", "2026-03-18")
ACTIVE, BROKEN, PAID, EXCEPTION, REJECTED = 10, 20, 30, 40, 50
UNKNOWN_STATUS = 999


class Db:
    """A tiny contract database. Agents: 1 and 2 are ordinary, 3 and 4 are excluded."""

    def __init__(self) -> None:
        self.conn = sqlite3.connect(":memory:")
        create_schema(self.conn)
        self.conn.executemany(
            "INSERT INTO agents VALUES (?,?,?)",
            [
                (1, "Agent 01", "agent_01"),
                (2, "Agent 02", "agent_02"),
                (3, "System Bot", "system_bot"),
                (4, "Test User", "test_user"),
            ],
        )
        self.conn.executemany(
            "INSERT INTO contact_results VALUES (?,?,?,?)",
            [(1, "No answer", 0, 0), (2, "Wrong party", 1, 0), (3, "Right party", 1, 1)],
        )
        self._record = 0

    def agreement(
        self,
        agreement_id,
        portfolio_id=1,
        installment=0,
        amount=100.0,
        *,
        issued="2026-03-10 09:00:00",
        due="2026-03-15 00:00:00",
        status=ACTIVE,
        agent=1,
        paid=None,
        paid_at=None,
    ):
        self._record += 1
        self.conn.execute(
            "INSERT INTO agreements VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                self._record,
                agreement_id,
                portfolio_id,
                installment,
                amount,
                paid,
                issued,
                due,
                paid_at,
                status,
                agent,
            ),
        )

    def label(self, agreement_id, portfolio_id, debt_id, label):
        self.conn.execute(
            "INSERT INTO agreement_debts VALUES (?,?,?)", (agreement_id, portfolio_id, debt_id)
        )
        self.conn.execute("INSERT INTO debt_attributes VALUES (?,?)", (debt_id, label))

    def event(self, agent, result, when="2026-03-10 10:00:00"):
        self.conn.execute(
            "INSERT INTO contact_events (debt_owner_id, result_id, agent_id, occurred_at) "
            "VALUES (1,?,?,?)",
            (result, agent, when),
        )

    def source(self, config=None) -> SqlDataSource:
        return SqlDataSource(self.conn, SQLITE, config or BusinessConfig())


def rows_by(rows, *keys):
    return {tuple(r[k] for k in keys): r for r in rows}


# -- dedup, filters, exclusions, portfolio resolution ---------------------------------


@pytest.fixture
def rollup_db():
    db = Db()
    # 1: a re-written first installment: the latest row wins
    db.agreement(1, amount=100.0, issued="2026-03-10 09:00:00")
    db.agreement(1, amount=120.0, issued="2026-03-10 11:00:00")
    db.label(1, 1, 101, "Portfolio T-01")
    # 2: a later row with a status outside every set must not hide the valid one
    db.agreement(2, amount=50.0, issued="2026-03-11 09:00:00")
    db.agreement(2, amount=55.0, issued="2026-03-11 12:00:00", status=UNKNOWN_STATUS)
    db.label(2, 1, 102, "Portfolio T-01")
    # 3: a re-write outside the window must not hide the in-window row
    db.agreement(3, amount=70.0, issued="2026-03-12 09:00:00")
    db.agreement(3, amount=999.0, issued="2026-04-02 09:00:00")
    db.label(3, 1, 103, "Portfolio T-01")
    # 4 and 5: excluded agents
    db.agreement(4, amount=500.0, agent=3)
    db.agreement(5, amount=500.0, agent=4)
    db.label(4, 1, 104, "Portfolio T-01")
    db.label(5, 1, 105, "Portfolio T-01")
    # 6: the same agreement id in two portfolios resolves to two labels
    db.agreement(6, portfolio_id=1, amount=10.0)
    db.agreement(6, portfolio_id=2, amount=20.0)
    db.label(6, 1, 106, "Portfolio T-01")
    db.label(6, 2, 206, "Portfolio T-02")
    # 7: a debt with no portfolio label is left out of portfolio rollups
    db.agreement(7, amount=300.0)
    db.conn.execute("INSERT INTO agreement_debts VALUES (7, 1, 107)")
    db.conn.execute("INSERT INTO debt_attributes VALUES (107, NULL)")
    return db


def test_dedup_filters_exclusions_and_pair_resolution_in_one_rollup(rollup_db):
    rows = rows_by(rollup_db.source().status_rollup(WINDOW), "portfolio_name", "status_id")
    t01 = rows[("Portfolio T-01", ACTIVE)]
    assert t01["qty"] == 4  # agreements 1, 2, 3 and 6
    assert t01["amount"] == 120.0 + 50.0 + 70.0 + 10.0
    t02 = rows[("Portfolio T-02", ACTIVE)]
    assert (t02["qty"], t02["amount"]) == (1, 20.0)
    assert not any(r["portfolio_name"] is None for r in rows.values())


def test_the_latest_rewrite_wins(rollup_db):
    amounts = {r["portfolio_name"]: r["amount"] for r in rollup_db.source().status_rollup(WINDOW)}
    assert amounts["Portfolio T-01"] == 250.0  # 120 (not 100) is part of it


def test_excluded_agent_patterns_are_configurable(rollup_db):
    keep_everyone = BusinessConfig(excluded_agent_patterns=())
    rows = rows_by(
        rollup_db.source(keep_everyone).status_rollup(WINDOW), "portfolio_name", "status_id"
    )
    assert rows[("Portfolio T-01", ACTIVE)]["amount"] > 1000.0


def test_the_window_end_is_inclusive_of_the_whole_last_day():
    db = Db()
    db.agreement(1, issued="2026-03-18 23:59:59")
    db.label(1, 1, 1, "P")
    db.agreement(2, issued="2026-03-19 00:00:00")
    db.label(2, 1, 2, "P")
    rows = db.source().status_rollup(WINDOW)
    assert rows[0]["qty"] == 1


def test_daily_rollup_groups_by_day_and_can_filter_one_portfolio():
    db = Db()
    for agreement, day, label in ((1, "10", "P-1"), (2, "10", "P-2"), (3, "11", "P-1")):
        db.agreement(agreement, issued=f"2026-03-{day} 09:00:00")
        db.label(agreement, 1, agreement, label)
    all_days = rows_by(db.source().daily_rollup(WINDOW), "day")
    assert all_days[("2026-03-10",)]["qty"] == 2
    only = db.source().daily_rollup(WINDOW, "P-1")
    assert {r["day"]: r["qty"] for r in only} == {"2026-03-10": 1, "2026-03-11": 1}


# -- the funnel, money and grain ------------------------------------------------------


def test_effort_counts_attempts_answers_and_right_party_contacts():
    db = Db()
    for result in (1, 1, 2, 3, 3, 3):
        db.event(1, result)
    db.event(3, 3)  # excluded agent
    db.event(1, 3, when="2026-02-01 10:00:00")  # outside the window
    row = db.source().queries.effort_by_agent(WINDOW)
    effort = db.source()._run(row)
    assert effort == [
        {
            "agent_id": "agent_01",
            "agent_name": "Agent 01",
            "attempt_count": 6,
            "answer_count": 4,
            "cpc_count": 3,
        }
    ]


def test_cpc_requires_both_flags_and_stays_below_the_answer_count():
    db = Db()
    for result in (2, 2, 3):
        db.event(1, result)
    row = db.source()._run(db.source().queries.effort_by_agent(WINDOW))[0]
    assert row["attempt_count"] >= row["answer_count"] >= row["cpc_count"]
    assert (row["answer_count"], row["cpc_count"]) == (3, 1)


def test_slips_are_paid_only_when_paid_within_the_grace_period():
    db = Db()
    due = "2026-03-10 00:00:00"
    db.agreement(
        1, amount=100.0, due=due, paid=100.0, paid_at="2026-03-13 00:00:00"
    )  # day 3: on time
    db.agreement(2, amount=100.0, due=due, paid=100.0, paid_at="2026-03-14 00:00:00")  # day 4: late
    db.agreement(3, amount=100.0, due=due)  # unpaid
    row = db.source()._run(db.source().queries.financial_by_agent(WINDOW))[0]
    assert (row["slips_issued"], row["slips_paid"]) == (3, 1)
    longer = BusinessConfig(grace_days=4)
    wider = db.source(longer)._run(db.source(longer).queries.financial_by_agent(WINDOW))[0]
    assert wider["slips_paid"] == 2


def test_exceptions_count_but_do_not_add_to_the_generated_value():
    db = Db()
    db.agreement(1, amount=100.0, status=ACTIVE)
    db.agreement(2, amount=40.0, status=EXCEPTION)
    row = db.source()._run(db.source().queries.financial_by_agent(WINDOW))[0]
    assert row["first_installment_value"] == 100.0
    assert (row["exception_count"], row["exception_value"]) == (1, 40.0)


def test_agreement_value_adds_every_installment_of_a_generated_agreement():
    db = Db()
    for installment, amount in ((0, 100.0), (1, 150.0), (2, 150.0)):
        db.agreement(1, installment=installment, amount=amount)
    db.agreement(2, amount=40.0, status=EXCEPTION)
    row = db.source()._run(db.source().queries.agreement_value_by_agent(WINDOW))[0]
    assert row["agreement_value"] == 400.0


def test_the_contract_grain_counts_distinct_debts_per_agreement():
    db = Db()
    db.agreement(1)
    for debt in (11, 12, 13):
        db.conn.execute("INSERT INTO agreement_debts VALUES (1, 1, ?)", (debt,))
    db.conn.execute("INSERT INTO agreement_debts VALUES (1, 1, 11)")  # duplicate link
    row = db.source()._run(db.source().queries.contracts_by_agent(WINDOW))[0]
    assert row["agreement_count_by_contract"] == 3


def test_agent_rollup_merges_the_four_aggregates_into_one_row_per_agent():
    db = Db()
    db.agreement(1, amount=100.0)
    db.conn.execute("INSERT INTO agreement_debts VALUES (1, 1, 11)")
    db.event(1, 3)
    (row,) = db.source().agent_rollup(WINDOW)
    assert row["agent_id"] == "agent_01"
    assert row["attempt_count"] == 1 and row["agreement_count"] == 1
    assert row["agreement_value"] == 100.0 and row["agreement_count_by_contract"] == 1


# -- stages, slips, drill-downs, hours ------------------------------------------------


def test_plan_stage_follows_payment_progress():
    db = Db()
    plans = {
        1: (3, 0),
        2: (3, 1),
        3: (4, 2),
        4: (7, 3),
        5: (2, 2),
    }  # agreement: (installments, paid)
    for agreement, (total, paid) in plans.items():
        for installment in range(total):
            db.agreement(
                agreement,
                installment=installment,
                amount=10.0,
                paid=10.0 if installment < paid else None,
            )
        db.label(agreement, 1, agreement, "P-1")
    stages = {r["stage"]: r for r in db.source().plan_stage_rollup(date(2026, 3, 18))}
    assert stages["early"]["qty"] == 2  # 0 and 1 paid
    assert stages["late"]["qty"] == 1  # 2 left of 4 (<= plan_late_remaining)
    assert stages["middle"]["qty"] == 1  # 4 left of 7
    assert stages["settled"]["qty"] == 1
    assert stages["settled"]["open_amount"] == 0.0
    assert stages["early"]["open_amount"] == 30.0 + 20.0


def test_slip_rollup_by_issue_day_and_by_portfolio_on_due_date():
    db = Db()
    db.agreement(
        1,
        issued="2026-03-10 09:00:00",
        due="2026-03-12 00:00:00",
        amount=100.0,
        paid=100.0,
        paid_at="2026-03-12 00:00:00",
    )
    db.agreement(2, issued="2026-03-10 09:00:00", due="2026-03-20 00:00:00", amount=50.0)
    db.label(1, 1, 1, "P-1")
    db.label(2, 1, 2, "P-2")
    by_day = db.source().slip_rollup(WINDOW, "day", "issued_at")
    assert by_day == [
        {
            "group_key": "2026-03-10",
            "generated": 2,
            "paid_on_time": 1,
            "amount_maturing": 150.0,
            "amount_received": 100.0,
        }
    ]
    by_portfolio = rows_by(db.source().slip_rollup(WINDOW, "portfolio", "due_at"), "group_key")
    assert ("P-1",) in by_portfolio and (
        "P-2",
    ) not in by_portfolio  # P-2 is due outside the window
    one = db.source().slip_rollup(WINDOW, "none", "due_at", "P-1")
    assert one[0]["generated"] == 1


def test_drilldown_rows_carry_totals_and_installment_counts():
    db = Db()
    for installment, amount in ((0, 100.0), (1, 80.0), (2, 80.0)):
        db.agreement(1, installment=installment, amount=amount, status=EXCEPTION)
    db.label(1, 1, 1, "P-1")
    (row,) = db.source().drilldown_rows("exception", "P-1", WINDOW)
    assert row["first_installment_value"] == 100.0 and row["total_value"] == 260.0
    assert row["installments"] == 3 and row["agent_id"] == "agent_01"
    assert db.source().drilldown_rows("generated", "P-1", WINDOW) == []
    with pytest.raises(ValueError, match="kind"):
        db.source().queries.drilldown("everything", "P-1", WINDOW)


def test_hourly_counts_group_generated_agreements_by_day_and_hour():
    db = Db()
    for agreement, when in ((1, "09:10"), (2, "09:50"), (3, "11:00")):
        db.agreement(agreement, issued=f"2026-03-10 {when}:00")
    rows = rows_by(db.source().hourly_counts(WINDOW), "hour")
    assert rows[(9,)]["qty"] == 2 and rows[(11,)]["qty"] == 1


def test_agent_status_rollup_and_crosstab():
    db = Db()
    db.agreement(1, amount=10.0, status=ACTIVE, agent=1)
    db.agreement(2, amount=20.0, status=EXCEPTION, agent=2)
    db.label(1, 1, 1, "P-1")
    db.label(2, 1, 2, "P-1")
    by_agent = rows_by(db.source().agent_status_rollup(WINDOW), "agent_id", "status_id")
    assert by_agent[("agent_02", EXCEPTION)]["amount"] == 20.0
    crosstab = db.source().crosstab_rollup(WINDOW, "portfolio", "P-1")
    assert {r["agent_id"] for r in crosstab} == {"agent_01", "agent_02"}
    assert db.source().crosstab_rollup(WINDOW, "agent", "agent_01")[0]["portfolio_name"] == "P-1"
    with pytest.raises(ValueError):
        db.source().queries.crosstab(WINDOW, "debtor", "x")


# -- query construction: dialects, parameters and the read-only guard --------------------

BUILDERS = {
    "status_rollup": lambda q: q.status_rollup(WINDOW),
    "daily_rollup": lambda q: q.daily_rollup(WINDOW, "P-1"),
    "effort_by_agent": lambda q: q.effort_by_agent(WINDOW),
    "financial_by_agent": lambda q: q.financial_by_agent(WINDOW),
    "agreement_value_by_agent": lambda q: q.agreement_value_by_agent(WINDOW),
    "contracts_by_agent": lambda q: q.contracts_by_agent(WINDOW),
    "agent_status_rollup": lambda q: q.agent_status_rollup(WINDOW),
    "crosstab": lambda q: q.crosstab(WINDOW, "agent", "agent_01"),
    "plan_stage": lambda q: q.plan_stage(date(2026, 3, 18)),
    "slip_rollup": lambda q: q.slip_rollup(WINDOW, "portfolio", "due_at", "P-1"),
    "drilldown": lambda q: q.drilldown("generated", "P-1", WINDOW),
    "hourly_counts": lambda q: q.hourly_counts(WINDOW),
}
_TABLES = "agreements|agents|agreement_debts|debt_attributes|contact_events|contact_results"


@pytest.mark.parametrize("name", sorted(BUILDERS))
@pytest.mark.parametrize("dialect", [SQLITE, SQLSERVER], ids=lambda d: d.name)
def test_every_placeholder_has_exactly_one_bound_value(name, dialect):
    query = BUILDERS[name](ContractQueries(BusinessConfig(), dialect))
    assert query.sql.count("?") == len(query.params)
    ensure_read_only(query.sql)


def unhinted_tables(sql):
    """Table references (`FROM t a` / `JOIN t a`) that are not followed by WITH (NOLOCK)."""
    pattern = rf"\b(?:FROM|JOIN)\s+(?:{_TABLES})\b(?!\s+\w*\s*WITH \(NOLOCK\))"
    return re.findall(pattern, sql)


@pytest.mark.parametrize("name", sorted(BUILDERS))
def test_sql_server_text_hints_every_table_with_nolock(name):
    sql = BUILDERS[name](ContractQueries(BusinessConfig(), SQLSERVER)).sql
    assert unhinted_tables(sql) == []
    assert "WITH (NOLOCK)" in sql


def test_the_nolock_check_would_notice_a_missing_hint():
    sql = BUILDERS["status_rollup"](ContractQueries(BusinessConfig(), SQLITE)).sql
    assert unhinted_tables(sql)


def test_sqlite_text_has_no_table_hints():
    sql = BUILDERS["status_rollup"](ContractQueries(BusinessConfig(), SQLITE)).sql
    assert "NOLOCK" not in sql


def test_sql_server_uses_its_own_date_and_grace_functions():
    sql = ContractQueries(BusinessConfig(), SQLSERVER).hourly_counts(WINDOW).sql
    assert "DATEPART(hour" in sql and "CAST(d.issued_at AS DATE)" in sql
    grace = ContractQueries(BusinessConfig(grace_days=7), SQLSERVER).financial_by_agent(WINDOW).sql
    assert "DATEADD(day, 7, d.due_at)" in grace


def test_hostile_text_is_bound_as_a_parameter_never_spliced_into_the_sql():
    hostile = "x'; DROP TABLE agreements; --"
    query = ContractQueries(BusinessConfig(), SQLSERVER).drilldown("generated", hostile, WINDOW)
    assert hostile in query.params
    assert "DROP" not in query.sql
    db = Db()
    assert db.source().drilldown_rows("generated", hostile, WINDOW) == []


def test_business_filters_sit_inside_the_derived_table_before_ranking():
    sql = ContractQueries(BusinessConfig(), SQLITE).status_rollup(WINDOW).sql
    inner = sql.index("FROM agreements a")
    rank = sql.index("ROW_NUMBER()")
    outer_filter = sql.index("x.rn = 1")
    where = sql.index("WHERE a.issued_at")
    assert rank < inner < where < outer_filter


def test_invalid_group_and_date_field_are_rejected():
    queries = ContractQueries(BusinessConfig())
    with pytest.raises(ValueError):
        queries.slip_rollup(WINDOW, "week", "issued_at")
    with pytest.raises(ValueError):
        queries.slip_rollup(WINDOW, "day", "paid_at")


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT 1",
        "select a from t where b = 'delete me'",
        "WITH x AS (SELECT 1) SELECT * FROM x;",
        "SELECT 1 -- DROP TABLE t",
        "SELECT /* DELETE */ 1",
    ],
)
def test_read_only_guard_accepts_selects(sql):
    ensure_read_only(sql)


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM t",
        "UPDATE t SET a = 1",
        "SELECT 1; DROP TABLE t",
        "SELECT * INTO copy FROM t",
        "EXEC sp_who",
        "SELECT 1 /* x */; DELETE FROM t",
        "",
        "-- only a comment",
    ],
)
def test_read_only_guard_rejects_everything_else(sql):
    with pytest.raises(UnsafeQuery):
        ensure_read_only(sql)
