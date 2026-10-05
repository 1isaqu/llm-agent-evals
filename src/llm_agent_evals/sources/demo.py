"""Invented demo data in throw-away in-memory SQLite databases.

Nothing here is real. Every row is produced by arithmetic on the row index, so the
databases are identical on every run and small enough to reason about by hand:

* two product lines (`auto`, `consumer`) with invented portfolios;
* six agents per line, four ordinary and two that the default exclusion patterns
  (`system%`, `test%`) must remove;
* agreements with two to four installments, statuses cycled over the default status
  catalogue, and a couple of re-written first installments so the dedup rule matters;
* contact events whose results cycle over a six-entry catalogue.

`DEMO_NOW` is the fixed "current time" the demo and the tests use.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta

from llm_agent_evals.business import DEFAULT_STATUS_IDS
from llm_agent_evals.sources.schema import create_schema

DEMO_NOW = datetime(2026, 3, 18, 15, 30, 0)

_PORTFOLIOS = {
    "auto": ("Portfolio A-01", "Portfolio A-02", "Portfolio A-03"),
    "consumer": ("Portfolio B-01", "Portfolio B-02"),
}
_AGREEMENT_COUNTS = {"auto": 36, "consumer": 24}

_AGENTS = (
    (1, "Agent 01", "agent_01"),
    (2, "Agent 02", "agent_02"),
    (3, "Agent 03", "agent_03"),
    (4, "Agent 04", "agent_04"),
    (5, "System Bot", "system_bot"),
    (6, "Test User", "test_user"),
)

_CONTACT_RESULTS = (
    (1, "No answer", 0, 0),
    (2, "Wrong number", 1, 0),
    (3, "Promise to pay", 1, 1),
    (4, "Callback requested", 1, 1),
    (5, "Voicemail", 0, 0),
    (6, "Hung up", 1, 0),
)

_STATUS_CYCLE = (
    "ACTIVE",
    "ACTIVE",
    "PAID",
    "ACTIVE",
    "BROKEN",
    "PENDING_EXCEPTION",
    "ACTIVE",
    "REJECTED",
    "PAID_MANUAL",
    "AUTO_BROKEN",
    "ACTIVE",
    "PAID",
)

_REWRITTEN = (3, 14)


def _stamp(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%d %H:%M:%S")


def _agent_for(index: int) -> int:
    if index % 9 == 0:
        return 5
    if index % 11 == 0:
        return 6
    return index % 4 + 1


def _paid_installments(index: int, status_id: int, total: int) -> int:
    """How many installments of an agreement are paid: all, some, or none, by status."""
    if status_id in (DEFAULT_STATUS_IDS["PAID"], DEFAULT_STATUS_IDS["PAID_MANUAL"]):
        return total
    if status_id == DEFAULT_STATUS_IDS["ACTIVE"]:
        return index % total
    return 1 if index % 3 == 0 else 0


def _agreement_rows(index: int, portfolio_id: int, status_id: int) -> list[tuple]:
    issued = datetime.combine(DEMO_NOW.date(), datetime.min.time()) - timedelta(days=index % 12)
    issued += timedelta(hours=8 + (index * 7) % 11, minutes=index % 50)
    first_amount = 100.0 + 20.0 * (index % 5)
    total = 2 + index % 3
    paid_count = _paid_installments(index, status_id, total)
    rows = []
    for installment in range(total):
        due = issued.replace(hour=0, minute=0) + timedelta(days=5 + 30 * installment)
        amount = first_amount if installment == 0 else 120.0 + 15.0 * (index % 6)
        paid_amount, paid_at = None, None
        if installment < paid_count:
            late = timedelta(days=10) if index % 7 == 0 and installment == 0 else timedelta(days=-1)
            paid_amount, paid_at = amount, _stamp(due + late)
        rows.append(
            (
                index + 1000,
                portfolio_id,
                installment,
                amount,
                paid_amount,
                _stamp(issued),
                _stamp(due),
                paid_at,
                status_id,
            )
        )
    return rows


def _insert_agreements(conn: sqlite3.Connection, line: str, portfolios: tuple[str, ...]) -> None:
    record = 0
    for index in range(_AGREEMENT_COUNTS[line]):
        portfolio_index = index % len(portfolios)
        status = DEFAULT_STATUS_IDS[_STATUS_CYCLE[(index * 5 + len(line)) % len(_STATUS_CYCLE)]]
        agent_id = _agent_for(index)
        for row in _agreement_rows(index, portfolio_index + 1, status):
            record += 1
            conn.execute(
                "INSERT INTO agreements VALUES (?,?,?,?,?,?,?,?,?,?,?)", (record, *row, agent_id)
            )
            if row[2] == 0 and index in _REWRITTEN:
                record += 1
                later = datetime.fromisoformat(row[5]) + timedelta(hours=1)
                broken = DEFAULT_STATUS_IDS["BROKEN"]
                conn.execute(
                    "INSERT INTO agreements VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        record,
                        row[0],
                        row[1],
                        0,
                        row[3],
                        None,
                        _stamp(later),
                        row[6],
                        None,
                        broken,
                        agent_id,
                    ),
                )
        conn.execute(
            "INSERT INTO agreement_debts VALUES (?,?,?)",
            (index + 1000, portfolio_index + 1, 5000 + index),
        )
        conn.execute(
            "INSERT INTO debts VALUES (?,?,?,?,?,?)",
            (5000 + index, portfolio_index + 1, f"C-{index:04d}", 900.0 + index, None, 30 + index),
        )
        conn.execute(
            "INSERT INTO debt_attributes VALUES (?,?)", (5000 + index, portfolios[portfolio_index])
        )


def _insert_events(conn: sqlite3.Connection) -> None:
    event_id = 0
    for agent_id in range(1, 5):
        for day_offset in range(12):
            for n in range(6 + agent_id):
                event_id += 1
                moment = datetime.combine(DEMO_NOW.date(), datetime.min.time()) - timedelta(
                    days=day_offset
                )
                moment += timedelta(hours=8 + n, minutes=agent_id)
                result_id = (agent_id + day_offset + n) % 6 + 1
                owner = 9000 + agent_id * 100 + day_offset * 10 + n
                conn.execute(
                    "INSERT INTO contact_events VALUES (?,?,?,?,?)",
                    (event_id, owner, result_id, agent_id, _stamp(moment)),
                )
    for extra in range(3):
        event_id += 1
        conn.execute(
            "INSERT INTO contact_events VALUES (?,?,?,?,?)",
            (event_id, 9900 + extra, 3, 5, _stamp(DEMO_NOW - timedelta(hours=extra + 1))),
        )


def build_demo_connection(line: str = "auto") -> sqlite3.Connection:
    """Build the in-memory demo database of one product line."""
    if line not in _PORTFOLIOS:
        raise ValueError(f"unknown product line {line!r}")
    conn = sqlite3.connect(":memory:")
    create_schema(conn)
    conn.executemany("INSERT INTO agents VALUES (?,?,?)", _AGENTS)
    conn.executemany(
        "INSERT INTO agreement_statuses VALUES (?,?)",
        [(v, k) for k, v in DEFAULT_STATUS_IDS.items()],
    )
    conn.executemany("INSERT INTO contact_results VALUES (?,?,?,?)", _CONTACT_RESULTS)
    _insert_agreements(conn, line, _PORTFOLIOS[line])
    _insert_events(conn)
    conn.commit()
    return conn


def build_demo_connections() -> dict[str, sqlite3.Connection]:
    """One in-memory database per product line."""
    return {line: build_demo_connection(line) for line in _PORTFOLIOS}
