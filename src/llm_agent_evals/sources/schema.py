"""DDL for the data contract, used to build throw-away SQLite databases.

The contract itself is documented in docs/data-contract.md. These statements exist so
tests and the demo can build an in-memory database with the same table and column
names a user maps their own system to. They are SQLite DDL: on a real system the
tables are your own views or tables and nothing here is executed against them.
"""

from __future__ import annotations

import sqlite3

SQLITE_DDL = (
    """CREATE TABLE agents (
        agent_id INTEGER PRIMARY KEY,
        display_name TEXT NOT NULL,
        login TEXT NOT NULL
    )""",
    """CREATE TABLE agreement_statuses (
        status_id INTEGER PRIMARY KEY,
        label TEXT NOT NULL
    )""",
    """CREATE TABLE agreements (
        record_id INTEGER PRIMARY KEY,
        agreement_id INTEGER NOT NULL,
        portfolio_id INTEGER NOT NULL,
        installment_no INTEGER NOT NULL,
        amount REAL NOT NULL,
        paid_amount REAL,
        issued_at TEXT NOT NULL,
        due_at TEXT NOT NULL,
        paid_at TEXT,
        status_id INTEGER NOT NULL,
        agent_id INTEGER NOT NULL
    )""",
    """CREATE TABLE agreement_debts (
        agreement_id INTEGER NOT NULL,
        portfolio_id INTEGER NOT NULL,
        debt_id INTEGER NOT NULL
    )""",
    """CREATE TABLE debts (
        debt_id INTEGER PRIMARY KEY,
        portfolio_id INTEGER NOT NULL,
        contract_no TEXT,
        balance REAL,
        due_at TEXT,
        days_overdue INTEGER
    )""",
    """CREATE TABLE debt_attributes (
        debt_id INTEGER NOT NULL,
        portfolio_label TEXT
    )""",
    """CREATE TABLE contact_results (
        result_id INTEGER PRIMARY KEY,
        label TEXT NOT NULL,
        answered INTEGER NOT NULL,
        right_party INTEGER NOT NULL
    )""",
    """CREATE TABLE contact_events (
        event_id INTEGER PRIMARY KEY,
        debt_owner_id INTEGER NOT NULL,
        result_id INTEGER NOT NULL,
        agent_id INTEGER NOT NULL,
        occurred_at TEXT NOT NULL
    )""",
)

CONTRACT_TABLES = (
    "agents",
    "agreement_statuses",
    "agreements",
    "agreement_debts",
    "debts",
    "debt_attributes",
    "contact_results",
    "contact_events",
)


def create_schema(connection: sqlite3.Connection) -> None:
    """Create all contract tables on a SQLite connection."""
    for statement in SQLITE_DDL:
        connection.execute(statement)
    connection.commit()
