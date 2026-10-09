"""SQL builders over the neutral data contract (see docs/data-contract.md).

Two dialects are supported. SQLite is executed in the test suite against in-memory
databases. SQL Server (T-SQL) is covered by tests on the generated text only and
has not been run against a live server.

Rules every query follows, because they are rules of the domain and not accidents:

* read-only, a single statement, validated by `ensure_read_only` before it runs;
* dedup first: the agreements table can hold several rows for the same
  (agreement_id, portfolio_id, installment_no) after a re-write, so every query keeps
  only the latest row per installment, and the business filters (window, status) sit
  *inside* the derived table, before `ROW_NUMBER()`. Filtering after the rank would let
  a re-write that falls outside the filter win the rank and hide the valid row;
* agreements are counted as rows of that deduplicated first-installment table, which holds
  one row per (agreement_id, portfolio_id). `COUNT(DISTINCT agreement_id)` would merge two
  agreements that share an id across portfolios and fall in the same group;
* aggregates live in separate queries instead of one coupled statement, so each can be
  checked on its own (agent rollups are merged in Python);
* the portfolio label is resolved through `agreement_debts` on the *pair*
  (agreement_id, portfolio_id) with a scalar `MIN(...)` subquery, which keeps one row
  per installment and is deterministic. Joining on agreement_id alone would match the
  wrong portfolio when an id repeats across portfolios;
* excluded agents are removed in SQL with `NOT LIKE` patterns on the agent login.

Parameters are positional (`?`) and are collected in the order they appear in the text.
Status id lists come from `BusinessConfig` as integers and are inlined.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

from llm_agent_evals.business import BusinessConfig
from llm_agent_evals.session import Window

_AGREEMENT_COLUMNS = (
    "agreement_id",
    "portfolio_id",
    "installment_no",
    "amount",
    "paid_amount",
    "issued_at",
    "due_at",
    "paid_at",
    "status_id",
    "agent_id",
)
_DATE_FIELDS = ("issued_at", "due_at")
_GROUP_BY = ("day", "portfolio", "none")
_CROSSTAB_BY = ("portfolio", "agent")


@dataclass(frozen=True)
class Query:
    """A statement plus its positional parameters."""

    sql: str
    params: tuple[Any, ...]


class UnsafeQuery(ValueError):
    """Raised when a statement is not a single read-only SELECT."""


class Dialect(ABC):
    """The handful of places where SQLite and SQL Server differ."""

    name = "ansi"
    table_hint = ""

    @abstractmethod
    def date_of(self, expr: str) -> str:
        """Expression that truncates a timestamp to a date."""

    @abstractmethod
    def hour_of(self, expr: str) -> str:
        """Expression that extracts the hour (0-23) from a timestamp."""

    @abstractmethod
    def paid_within_grace(self, paid_expr: str, due_expr: str, days: int) -> str:
        """Boolean expression: paid on or before due date plus `days`."""

    def bind_date(self, value: date) -> Any:
        """Python value to bind for a date parameter."""
        return value


class SqliteDialect(Dialect):
    name = "sqlite"

    def date_of(self, expr: str) -> str:
        return f"date({expr})"

    def hour_of(self, expr: str) -> str:
        return f"CAST(strftime('%H', {expr}) AS INTEGER)"

    def paid_within_grace(self, paid_expr: str, due_expr: str, days: int) -> str:
        return f"datetime({paid_expr}) <= datetime({due_expr}, '+{int(days)} day')"

    def bind_date(self, value: date) -> str:
        return value.isoformat()


class SqlServerDialect(Dialect):
    """T-SQL. `WITH (NOLOCK)` on every read table, as analytical dashboards usually do."""

    name = "sqlserver"
    table_hint = " WITH (NOLOCK)"

    def date_of(self, expr: str) -> str:
        return f"CAST({expr} AS DATE)"

    def hour_of(self, expr: str) -> str:
        return f"DATEPART(hour, {expr})"

    def paid_within_grace(self, paid_expr: str, due_expr: str, days: int) -> str:
        return f"{paid_expr} <= DATEADD(day, {int(days)}, {due_expr})"


SQLITE = SqliteDialect()
SQLSERVER = SqlServerDialect()


def _int_list(values: tuple[int, ...]) -> str:
    if not values:
        raise ValueError("status id list must not be empty")
    return ", ".join(str(int(v)) for v in values)


class _Params:
    """Collects positional parameters in textual order."""

    def __init__(self, dialect: Dialect) -> None:
        self.values: list[Any] = []
        self._dialect = dialect

    def add(self, value: Any) -> str:
        self.values.append(value)
        return "?"

    def date(self, value: date) -> str:
        return self.add(self._dialect.bind_date(value))


class ContractQueries:
    """Builds every query the analytics layer needs."""

    def __init__(self, config: BusinessConfig, dialect: Dialect = SQLITE) -> None:
        self.config = config
        self.dialect = dialect
        self._hint = dialect.table_hint

    # -- shared fragments ---------------------------------------------------

    def _dedup(
        self,
        p: _Params,
        *,
        window: Window | None,
        date_field: str = "issued_at",
        first_only: bool = True,
        status_ids: tuple[int, ...] | None = None,
    ) -> str:
        """Derived table with the latest row per installment, filters applied before ranking."""
        if date_field not in _DATE_FIELDS:
            raise ValueError(f"date_field must be one of {_DATE_FIELDS}")
        conditions: list[str] = []
        if window is not None:
            low, high = p.date(window.start), p.date(window.end_exclusive)
            conditions.append(f"a.{date_field} >= {low} AND a.{date_field} < {high}")
        if first_only:
            conditions.append(f"a.installment_no = {int(self.config.first_installment_no)}")
        if status_ids is not None:
            conditions.append(f"a.status_id IN ({_int_list(status_ids)})")
        where = " AND ".join(conditions) or "1 = 1"
        inner = ", ".join(f"a.{c}" for c in _AGREEMENT_COLUMNS)
        outer = ", ".join(f"x.{c}" for c in _AGREEMENT_COLUMNS)
        return (
            f"(SELECT {outer} FROM (SELECT {inner}, ROW_NUMBER() OVER ("
            "PARTITION BY a.agreement_id, a.portfolio_id, a.installment_no "
            "ORDER BY a.issued_at DESC, a.record_id DESC) AS rn "
            f"FROM agreements a{self._hint} WHERE {where}) x WHERE x.rn = 1)"
        )

    def _all_rows_cte(self) -> str:
        """CTE with every installment of every agreement, latest row per installment ranked 1."""
        return (
            "allrows AS (SELECT a.agreement_id, a.portfolio_id, a.installment_no, a.amount, "
            "a.paid_amount, ROW_NUMBER() OVER (PARTITION BY a.agreement_id, a.portfolio_id, "
            "a.installment_no ORDER BY a.issued_at DESC, a.record_id DESC) AS rn "
            f"FROM agreements a{self._hint})"
        )

    def _exclusions(self, p: _Params) -> str:
        parts = [
            f" AND LOWER(u.login) NOT LIKE {p.add(pattern.lower())}"
            for pattern in self.config.excluded_agent_patterns
        ]
        return "".join(parts)

    def _portfolio_expr(self, alias: str) -> str:
        h = self._hint
        return (
            "(SELECT MIN(da.portfolio_label) "
            f"FROM agreement_debts ad{h} JOIN debt_attributes da{h} ON da.debt_id = ad.debt_id "
            f"WHERE ad.agreement_id = {alias}.agreement_id "
            f"AND ad.portfolio_id = {alias}.portfolio_id "
            "AND da.portfolio_label IS NOT NULL)"
        )

    def _agents_join(self, alias: str) -> str:
        return f"JOIN agents u{self._hint} ON u.agent_id = {alias}.agent_id"

    def _gen_cte(self, p: _Params, window: Window) -> str:
        dedup = self._dedup(p, window=window, status_ids=self.config.generated_ids)
        join = self._agents_join("d")
        excl = self._exclusions(p)
        return (
            "gen AS (SELECT d.agreement_id, d.portfolio_id, d.agent_id "
            f"FROM {dedup} d {join} WHERE 1 = 1{excl})"
        )

    # -- rollups -----------------------------------------------------------

    def status_rollup(self, window: Window) -> Query:
        p = _Params(self.dialect)
        from_sql = (
            f"FROM {self._dedup(p, window=window, status_ids=self.config.rollup_ids)} d "
            f"{self._agents_join('d')}"
        )
        excl = self._exclusions(p)
        sql = (
            "SELECT s.portfolio_name, s.status_id, COUNT(*) AS qty, "
            "SUM(s.amount) AS amount FROM ("
            "SELECT d.agreement_id, d.status_id, d.amount, "
            f"{self._portfolio_expr('d')} AS portfolio_name {from_sql} WHERE 1 = 1{excl}) s "
            "WHERE s.portfolio_name IS NOT NULL "
            "GROUP BY s.portfolio_name, s.status_id ORDER BY s.portfolio_name, s.status_id"
        )
        return Query(sql, tuple(p.values))

    def daily_rollup(self, window: Window, portfolio: str | None = None) -> Query:
        p = _Params(self.dialect)
        day = self.dialect.date_of("d.issued_at")
        portfolio_col = f", {self._portfolio_expr('d')} AS portfolio_name" if portfolio else ""
        from_sql = (
            f"FROM {self._dedup(p, window=window, status_ids=self.config.rollup_ids)} d "
            f"{self._agents_join('d')}"
        )
        excl = self._exclusions(p)
        portfolio_where = f" WHERE s.portfolio_name = {p.add(portfolio)}" if portfolio else ""
        sql = (
            "SELECT s.day, s.status_id, COUNT(*) AS qty, "
            "SUM(s.amount) AS amount FROM ("
            f"SELECT {day} AS day, d.status_id, d.agreement_id, d.amount{portfolio_col} "
            f"{from_sql} WHERE 1 = 1{excl}) s{portfolio_where} "
            "GROUP BY s.day, s.status_id ORDER BY s.day, s.status_id"
        )
        return Query(sql, tuple(p.values))

    def effort_by_agent(self, window: Window) -> Query:
        p = _Params(self.dialect)
        h = self._hint
        low, high = p.date(window.start), p.date(window.end_exclusive)
        excl = self._exclusions(p)
        sql = (
            "SELECT u.login AS agent_id, u.display_name AS agent_name, "
            "COUNT(*) AS attempt_count, "
            "SUM(CASE WHEN cr.answered = 1 THEN 1 ELSE 0 END) AS answer_count, "
            "SUM(CASE WHEN cr.answered = 1 AND cr.right_party = 1 THEN 1 ELSE 0 END) AS cpc_count "
            f"FROM contact_events ce{h} "
            f"JOIN contact_results cr{h} ON cr.result_id = ce.result_id "
            f"JOIN agents u{h} ON u.agent_id = ce.agent_id "
            f"WHERE ce.occurred_at >= {low} AND ce.occurred_at < {high}{excl} "
            "GROUP BY u.login, u.display_name"
        )
        return Query(sql, tuple(p.values))

    def financial_by_agent(self, window: Window) -> Query:
        p = _Params(self.dialect)
        gen = _int_list(self.config.generated_ids)
        exc = _int_list(self.config.exception_ids)
        within = self.dialect.paid_within_grace("d.paid_at", "d.due_at", self.config.grace_days)
        from_sql = (
            f"FROM {self._dedup(p, window=window, status_ids=self.config.universe_ids)} d "
            f"{self._agents_join('d')}"
        )
        excl = self._exclusions(p)
        sql = (
            "SELECT u.login AS agent_id, u.display_name AS agent_name, "
            f"SUM(CASE WHEN d.status_id IN ({gen}) THEN 1 ELSE 0 END) AS agreement_count, "
            f"SUM(CASE WHEN d.status_id IN ({gen}) THEN d.amount ELSE 0 END) "
            "AS first_installment_value, "
            f"SUM(CASE WHEN d.status_id IN ({gen}) THEN 1 ELSE 0 END) AS slips_issued, "
            f"SUM(CASE WHEN d.status_id IN ({gen}) AND d.paid_amount > 0 AND {within} "
            "THEN 1 ELSE 0 END) AS slips_paid, "
            f"SUM(CASE WHEN d.status_id IN ({exc}) THEN 1 ELSE 0 END) AS exception_count, "
            f"SUM(CASE WHEN d.status_id IN ({exc}) THEN d.amount ELSE 0 END) AS exception_value "
            f"{from_sql} WHERE 1 = 1{excl} GROUP BY u.login, u.display_name"
        )
        return Query(sql, tuple(p.values))

    def agreement_value_by_agent(self, window: Window) -> Query:
        p = _Params(self.dialect)
        gen_cte = self._gen_cte(p, window)
        sql = (
            f"WITH {gen_cte}, {self._all_rows_cte()} "
            "SELECT u.login AS agent_id, u.display_name AS agent_name, "
            "SUM(r.amount) AS agreement_value "
            "FROM gen g JOIN allrows r ON r.agreement_id = g.agreement_id "
            "AND r.portfolio_id = g.portfolio_id AND r.rn = 1 "
            f"{self._agents_join('g')} GROUP BY u.login, u.display_name"
        )
        return Query(sql, tuple(p.values))

    def contracts_by_agent(self, window: Window) -> Query:
        p = _Params(self.dialect)
        gen_cte = self._gen_cte(p, window)
        sql = (
            f"WITH {gen_cte} "
            "SELECT x.agent_id, x.agent_name, COUNT(*) AS agreement_count_by_contract FROM ("
            "SELECT DISTINCT u.login AS agent_id, u.display_name AS agent_name, "
            "ad.agreement_id, ad.portfolio_id, ad.debt_id "
            f"FROM gen g JOIN agreement_debts ad{self._hint} ON ad.agreement_id = g.agreement_id "
            f"AND ad.portfolio_id = g.portfolio_id {self._agents_join('g')}) x "
            "GROUP BY x.agent_id, x.agent_name"
        )
        return Query(sql, tuple(p.values))

    def agent_status_rollup(self, window: Window) -> Query:
        p = _Params(self.dialect)
        from_sql = (
            f"FROM {self._dedup(p, window=window, status_ids=self.config.rollup_ids)} d "
            f"{self._agents_join('d')}"
        )
        excl = self._exclusions(p)
        sql = (
            "SELECT u.login AS agent_id, u.display_name AS agent_name, d.status_id, "
            "COUNT(*) AS qty, SUM(d.amount) AS amount "
            f"{from_sql} WHERE 1 = 1{excl} "
            "GROUP BY u.login, u.display_name, d.status_id ORDER BY u.login, d.status_id"
        )
        return Query(sql, tuple(p.values))

    def crosstab(self, window: Window, by: str, value: str) -> Query:
        if by not in _CROSSTAB_BY:
            raise ValueError(f"by must be one of {_CROSSTAB_BY}")
        p = _Params(self.dialect)
        from_sql = (
            f"FROM {self._dedup(p, window=window, status_ids=self.config.rollup_ids)} d "
            f"{self._agents_join('d')}"
        )
        excl = self._exclusions(p)
        column = "s.portfolio_name" if by == "portfolio" else "s.agent_id"
        sql = (
            "SELECT s.agent_id, s.agent_name, s.portfolio_name, s.status_id, "
            "COUNT(*) AS qty, SUM(s.amount) AS amount FROM ("
            "SELECT d.agreement_id, d.status_id, d.amount, u.login AS agent_id, "
            "u.display_name AS agent_name, "
            f"{self._portfolio_expr('d')} AS portfolio_name {from_sql} WHERE 1 = 1{excl}) s "
            f"WHERE s.portfolio_name IS NOT NULL AND {column} = {p.add(value)} "
            "GROUP BY s.agent_id, s.agent_name, s.portfolio_name, s.status_id "
            "ORDER BY s.agent_id, s.portfolio_name, s.status_id"
        )
        return Query(sql, tuple(p.values))

    def plan_stage(self, end: date) -> Query:
        cfg = self.config
        window = Window(end - timedelta(days=cfg.plan_lookback_days), end)
        p = _Params(self.dialect)
        from_sql = (
            f"FROM {self._dedup(p, window=window, status_ids=cfg.approved_ids)} b "
            f"{self._agents_join('b')} "
            "JOIN progress g ON g.agreement_id = b.agreement_id AND g.portfolio_id = b.portfolio_id"
        )
        excl = self._exclusions(p)
        stage = (
            "CASE WHEN g.paid_installments >= g.total_installments THEN 'settled' "
            "WHEN g.paid_installments <= 1 THEN 'early' "
            f"WHEN g.total_installments - g.paid_installments <= {int(cfg.plan_late_remaining)} "
            "THEN 'late' ELSE 'middle' END"
        )
        progress = (
            "progress AS (SELECT r.agreement_id, r.portfolio_id, COUNT(*) AS total_installments, "
            "SUM(CASE WHEN r.paid_amount > 0 THEN 1 ELSE 0 END) AS paid_installments, "
            "SUM(CASE WHEN COALESCE(r.paid_amount, 0) <= 0 THEN r.amount ELSE 0 END) "
            "AS open_amount FROM allrows r WHERE r.rn = 1 GROUP BY r.agreement_id, r.portfolio_id)"
        )
        sql = (
            f"WITH {self._all_rows_cte()}, {progress} "
            "SELECT s.portfolio_name, s.stage, COUNT(*) AS qty, SUM(s.open_amount) AS open_amount "
            f"FROM (SELECT {self._portfolio_expr('b')} AS portfolio_name, {stage} AS stage, "
            f"g.open_amount {from_sql} WHERE 1 = 1{excl}) s "
            "WHERE s.portfolio_name IS NOT NULL "
            "GROUP BY s.portfolio_name, s.stage ORDER BY s.portfolio_name, s.stage"
        )
        return Query(sql, tuple(p.values))

    def slip_rollup(
        self, window: Window, group_by: str, date_field: str, portfolio: str | None = None
    ) -> Query:
        if group_by not in _GROUP_BY:
            raise ValueError(f"group_by must be one of {_GROUP_BY}")
        p = _Params(self.dialect)
        group_expr = {
            "day": self.dialect.date_of(f"d.{date_field}") if date_field in _DATE_FIELDS else "",
            "portfolio": self._portfolio_expr("d"),
            "none": "'all'",
        }[group_by]
        within = self.dialect.paid_within_grace("d.paid_at", "d.due_at", self.config.grace_days)
        portfolio_col = f", {self._portfolio_expr('d')} AS portfolio_name" if portfolio else ""
        generated = self.config.generated_ids
        dedup = self._dedup(p, window=window, date_field=date_field, status_ids=generated)
        from_sql = f"FROM {dedup} d {self._agents_join('d')}"
        excl = self._exclusions(p)
        wheres: list[str] = []
        if group_by == "portfolio":
            wheres.append("s.group_key IS NOT NULL")
        if portfolio:
            wheres.append(f"s.portfolio_name = {p.add(portfolio)}")
        where = f" WHERE {' AND '.join(wheres)}" if wheres else ""
        sql = (
            "SELECT s.group_key, COUNT(*) AS generated, SUM(s.paid_flag) AS paid_on_time, "
            "SUM(s.amount) AS amount_maturing, SUM(COALESCE(s.paid_amount, 0)) AS amount_received "
            "FROM (SELECT d.amount, d.paid_amount, "
            f"CASE WHEN d.paid_amount > 0 AND {within} THEN 1 ELSE 0 END AS paid_flag, "
            f"{group_expr} AS group_key{portfolio_col} {from_sql} WHERE 1 = 1{excl}) s{where} "
            "GROUP BY s.group_key ORDER BY s.group_key"
        )
        return Query(sql, tuple(p.values))

    def drilldown(self, kind: str, portfolio: str, window: Window) -> Query:
        cfg = self.config
        ids = {
            "generated": cfg.generated_ids,
            "exception": cfg.exception_ids,
            "rejected": cfg.rejected_ids,
            "broken": cfg.broken_ids,
        }.get(kind)
        if ids is None:
            raise ValueError("kind must be one of generated, exception, rejected, broken")
        p = _Params(self.dialect)
        totals = (
            "totals AS (SELECT r.agreement_id, r.portfolio_id, SUM(r.amount) AS total_value, "
            "COUNT(DISTINCT r.installment_no) AS installments FROM allrows r WHERE r.rn = 1 "
            "GROUP BY r.agreement_id, r.portfolio_id)"
        )
        from_sql = (
            f"FROM {self._dedup(p, window=window, status_ids=ids)} d {self._agents_join('d')}"
        )
        excl = self._exclusions(p)
        sql = (
            f"WITH {self._all_rows_cte()}, {totals} "
            "SELECT s.agreement_id, s.portfolio_id, s.first_installment_value, t.total_value, "
            "s.agent_name, s.agent_id, s.issued_on, s.due_on, t.installments FROM ("
            "SELECT d.agreement_id, d.portfolio_id, d.amount AS first_installment_value, "
            "u.display_name AS agent_name, u.login AS agent_id, "
            f"{self.dialect.date_of('d.issued_at')} AS issued_on, "
            f"{self.dialect.date_of('d.due_at')} AS due_on, "
            f"{self._portfolio_expr('d')} AS portfolio_name {from_sql} WHERE 1 = 1{excl}) s "
            "JOIN totals t ON t.agreement_id = s.agreement_id AND t.portfolio_id = s.portfolio_id "
            f"WHERE s.portfolio_name = {p.add(portfolio)} "
            "ORDER BY s.first_installment_value DESC, s.agreement_id"
        )
        return Query(sql, tuple(p.values))

    def hourly_counts(self, window: Window) -> Query:
        p = _Params(self.dialect)
        day = self.dialect.date_of("d.issued_at")
        hour = self.dialect.hour_of("d.issued_at")
        from_sql = (
            f"FROM {self._dedup(p, window=window, status_ids=self.config.generated_ids)} d "
            f"{self._agents_join('d')}"
        )
        excl = self._exclusions(p)
        sql = (
            f"SELECT s.day, s.hour, COUNT(*) AS qty FROM ("
            f"SELECT {day} AS day, {hour} AS hour, d.agreement_id {from_sql} WHERE 1 = 1{excl}) s "
            "GROUP BY s.day, s.hour ORDER BY s.day, s.hour"
        )
        return Query(sql, tuple(p.values))


# -- read-only guard ----------------------------------------------------------

_LITERAL = re.compile(r"'(?:[^']|'')*'")
_LINE_COMMENT = re.compile(r"--[^\n]*")
_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
_FORBIDDEN = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE|MERGE|EXEC|EXECUTE|GRANT|REVOKE|INTO)\b",
    re.IGNORECASE,
)


def ensure_read_only(sql: str) -> None:
    """Raise `UnsafeQuery` unless `sql` is one SELECT (optionally with CTEs).

    Quoted literals and comments are removed first so that a keyword inside a string
    does not trip the check and a comment cannot hide a second statement.
    """
    text = _BLOCK_COMMENT.sub(" ", _LINE_COMMENT.sub(" ", _LITERAL.sub("''", sql))).strip()
    body = text.rstrip(";").strip()
    if ";" in body:
        raise UnsafeQuery("only a single statement is allowed")
    first = body.split(None, 1)[0].upper() if body else ""
    if first not in ("SELECT", "WITH"):
        raise UnsafeQuery("only SELECT statements are allowed")
    match = _FORBIDDEN.search(body)
    if match:
        raise UnsafeQuery(f"statement contains the forbidden keyword {match.group(1).upper()}")
