"""A `DataSource` that runs the contract queries on any DB-API connection."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import date, datetime
from decimal import Decimal
from typing import Any

from llm_agent_evals.business import BusinessConfig
from llm_agent_evals.errors import ConfigError, DataSourceError
from llm_agent_evals.session import Window
from llm_agent_evals.sources.base import Row
from llm_agent_evals.sources.sql import (
    SQLITE,
    ContractQueries,
    Dialect,
    Query,
    UnsafeQuery,
    ensure_read_only,
)

_AGENT_COUNTERS = (
    "attempt_count",
    "answer_count",
    "cpc_count",
    "agreement_count",
    "agreement_value",
    "first_installment_value",
    "agreement_count_by_contract",
    "slips_issued",
    "slips_paid",
    "exception_count",
    "exception_value",
)


def _plain(value: Any) -> Any:
    """Normalise driver types: Decimal to float, dates to ISO strings."""
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, datetime):
        return value.isoformat(sep=" ")
    if isinstance(value, date):
        return value.isoformat()
    return value


class SqlDataSource:
    """Executes `ContractQueries` on a DB-API connection.

    The connection is supplied by the caller and is used only for statements that pass
    `ensure_read_only`. Driver failures become `DataSourceError`, which the run guard
    treats as transient and retries once; the driver's message stays in the exception
    chain for the operator and never reaches the model.
    """

    def __init__(
        self,
        connection: Any,
        dialect: Dialect = SQLITE,
        config: BusinessConfig | None = None,
    ) -> None:
        self._connection = connection
        self.config = config or BusinessConfig()
        self.queries = ContractQueries(self.config, dialect)

    # -- DataSource protocol ------------------------------------------------

    def status_rollup(self, window: Window) -> list[Row]:
        return self._run(self.queries.status_rollup(window))

    def daily_rollup(self, window: Window, portfolio: str | None = None) -> list[Row]:
        return self._run(self.queries.daily_rollup(window, portfolio))

    def agent_rollup(self, window: Window) -> list[Row]:
        parts = (
            self._run(self.queries.effort_by_agent(window)),
            self._run(self.queries.financial_by_agent(window)),
            self._run(self.queries.agreement_value_by_agent(window)),
            self._run(self.queries.contracts_by_agent(window)),
        )
        return merge_agent_rows(*parts)

    def agent_status_rollup(self, window: Window) -> list[Row]:
        return self._run(self.queries.agent_status_rollup(window))

    def crosstab_rollup(self, window: Window, by: str, value: str) -> list[Row]:
        return self._run(self.queries.crosstab(window, by, value))

    def plan_stage_rollup(self, end: date) -> list[Row]:
        return self._run(self.queries.plan_stage(end))

    def slip_rollup(
        self, window: Window, group_by: str, date_field: str, portfolio: str | None = None
    ) -> list[Row]:
        return self._run(self.queries.slip_rollup(window, group_by, date_field, portfolio))

    def drilldown_rows(self, kind: str, portfolio: str, window: Window) -> list[Row]:
        return self._run(self.queries.drilldown(kind, portfolio, window))

    def hourly_counts(self, window: Window) -> list[Row]:
        return self._run(self.queries.hourly_counts(window))

    # -- status catalogue check -------------------------------------------

    def catalog_problems(self) -> list[str]:
        """Compare the configured status ids with the `agreement_statuses` table.

        The catalogue is never used to *define* a set (sets are configuration), only to
        catch a mapping that points at an id the database does not have, or at an id
        whose label differs from the configured one.
        """
        rows = self._run(Query("SELECT status_id, label FROM agreement_statuses", ()))
        catalogue = {int(r["status_id"]): str(r["label"]) for r in rows}
        problems: list[str] = []
        for label, status_id in self.config.status_ids.items():
            if status_id not in catalogue:
                problems.append(
                    f"status {label} maps to id {status_id}, which is not in the catalogue"
                )
            elif catalogue[status_id].strip().upper() != label:
                problems.append(
                    f"status {label} maps to id {status_id}, labelled "
                    f"{catalogue[status_id]!r} in the catalogue"
                )
        return problems

    # -- execution ----------------------------------------------------------

    def _run(self, query: Query) -> list[Row]:
        try:
            ensure_read_only(query.sql)
        except UnsafeQuery as exc:
            raise ConfigError(f"refusing to run a non read-only statement: {exc}") from exc
        cursor = None
        try:
            cursor = self._connection.cursor()
            cursor.execute(query.sql, query.params)
            names = [str(column[0]).lower() for column in cursor.description]
            return [
                {name: _plain(value) for name, value in zip(names, record, strict=True)}
                for record in cursor.fetchall()
            ]
        except Exception as exc:
            raise DataSourceError(f"query failed ({type(exc).__name__}): {exc}") from exc
        finally:
            close = getattr(cursor, "close", None)
            if callable(close):
                close()


def merge_agent_rows(*row_sets: list[Row]) -> list[Row]:
    """Merge per-agent aggregate rows from separate queries into one row per agent."""
    merged: dict[str, Row] = {}
    for rows in row_sets:
        for row in rows:
            agent_id = str(row.get("agent_id") or "").strip()
            if not agent_id:
                continue
            target = merged.setdefault(
                agent_id,
                {"agent_id": agent_id, "agent_name": "", **dict.fromkeys(_AGENT_COUNTERS, 0)},
            )
            if not target["agent_name"]:
                target["agent_name"] = str(row.get("agent_name") or "").strip()
            for counter in _AGENT_COUNTERS:
                if counter in row and row[counter] is not None:
                    target[counter] = row[counter]
    return sorted(merged.values(), key=lambda r: r["agent_id"])


def connect_sqlserver(
    env: Mapping[str, str],
    line: str | None = None,
    connect: Callable[..., Any] | None = None,
) -> Any:
    """Open a read-oriented SQL Server connection from environment variables.

    Variables: `DB_DRIVER`, `DB_HOST`, `DB_NAME` (or `DB_NAME_<LINE>` for one database
    per product line), `DB_USER`, `DB_PASSWORD`. The `pyodbc` package is imported only
    here and only when `connect` is not injected. This function has not been exercised
    against a live server: tests inject a fake `connect`.
    """
    name_var = f"DB_NAME_{line.upper()}" if line else "DB_NAME"
    database = (env.get(name_var) or env.get("DB_NAME") or "").strip()
    required = {
        "DB_HOST": (env.get("DB_HOST") or "").strip(),
        name_var: database,
        "DB_USER": (env.get("DB_USER") or "").strip(),
        "DB_PASSWORD": env.get("DB_PASSWORD") or "",
    }
    missing = [name for name, value in required.items() if not value]
    if missing:
        raise ConfigError(f"missing database settings: {', '.join(missing)}")
    driver = (env.get("DB_DRIVER") or "ODBC Driver 17 for SQL Server").strip()
    connection_string = build_connection_string(
        driver, required["DB_HOST"], database, required["DB_USER"], required["DB_PASSWORD"]
    )
    if connect is None:
        try:
            import pyodbc
        except ImportError as exc:
            raise ConfigError(
                "pyodbc is not installed (pip install 'llm-agent-evals[sqlserver]')"
            ) from exc
        connect = pyodbc.connect
    return connect(connection_string, autocommit=True, timeout=30)


def _odbc_value(value: str) -> str:
    """Brace-escape an ODBC attribute value when it contains special characters."""
    if any(ch in value for ch in ";{}= "):
        return "{" + value.replace("}", "}}") + "}"
    return value


def build_connection_string(driver: str, host: str, database: str, user: str, password: str) -> str:
    return (
        f"DRIVER={{{driver}}};SERVER={_odbc_value(host)};DATABASE={_odbc_value(database)};"
        f"UID={_odbc_value(user)};PWD={_odbc_value(password)};ApplicationIntent=ReadOnly"
    )


def sources_from_connections(
    connections: Mapping[str, Any], dialect: Dialect, config: BusinessConfig
) -> dict[str, SqlDataSource]:
    """One `SqlDataSource` per product line."""
    return {line: SqlDataSource(conn, dialect, config) for line, conn in connections.items()}
