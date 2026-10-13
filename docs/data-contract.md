# Data contract

The agent never sees your database. It sees nine *rollup* methods (the `DataSource` protocol), and
the SQL in `sources/sql.py` implements them over eight neutral tables. You connect your own data
in one of two ways:

1. Map your database to the eight tables (views are fine) and use `SqlDataSource` as it is.
2. Implement `DataSource` yourself and return the rollup rows described in the second half of
   this page. The analytics, the tools and the agent do not care where the rows come from.

All data in this repository is invented.

## Part 1: the eight tables

Names are fixed because the queries use them. Column types are the SQLite types used by the demo
schema (`sources/schema.py`); on SQL Server any compatible type works (`datetime2`, `decimal`, ...).

### `agreements`: one row per installment of an agreement

| Column | Type | Meaning |
|---|---|---|
| `record_id` | integer, unique | Row identity. Only used as the tie-breaker when the same installment was written twice. Any increasing key works. |
| `agreement_id` | integer | Agreement number. Not unique on its own: pair it with `portfolio_id`. |
| `portfolio_id` | integer | Portfolio the agreement belongs to. |
| `installment_no` | integer | `0` is the first installment (the down payment). The value is configurable (`first_installment_no`). |
| `amount` | decimal | Amount due for this installment. |
| `paid_amount` | decimal, null | Amount paid. A value above zero means the installment is paid. |
| `issued_at` | datetime | When the row was issued. This is the date filter column for every windowed query. |
| `due_at` | datetime | Due date of the installment. |
| `paid_at` | datetime, null | When it was paid. |
| `status_id` | integer | Foreign key to `agreement_statuses`. See "Status sets". |
| `agent_id` | integer | Foreign key to `agents`: who owns the agreement. |

The table can hold several rows for the same `(agreement_id, portfolio_id, installment_no)`,
for example when a system re-writes an installment. Every query keeps only the latest one (highest
`issued_at`, then highest `record_id`). The window and status filters are applied *before* that
ranking, inside the derived table. See [docs/decisions.md](decisions.md) for why.

### `agreement_debts`: links agreements to debts

| Column | Type | Meaning |
|---|---|---|
| `agreement_id` | integer | |
| `portfolio_id` | integer | |
| `debt_id` | integer | Foreign key to `debts` and `debt_attributes`. |

One agreement can cover several debts. The pair `(agreement_id, portfolio_id)` is the join key,
never `agreement_id` alone: an id that repeats across portfolios would otherwise match the wrong
one.

### `debt_attributes`: where the portfolio label lives

| Column | Type | Meaning |
|---|---|---|
| `debt_id` | integer | |
| `portfolio_label` | text, null | Human-readable portfolio name, for example `Portfolio A-01`. |

An agreement whose debts all have a null label is left out of every portfolio rollup. If several
debts carry different labels, the alphabetically smallest one is used (`MIN`), which keeps the
result deterministic and one row per installment.

### `debts`: debt / contract detail

| Column | Type |
|---|---|
| `debt_id` | integer |
| `portfolio_id` | integer |
| `contract_no` | text |
| `balance` | decimal |
| `due_at` | datetime |
| `days_overdue` | integer |

Part of the shared contract and created by the demo schema, but no current query reads it. It
is here so that a tool needing balances or ageing has an agreed place to look.

### `contact_events`: one row per outreach attempt

| Column | Type | Meaning |
|---|---|---|
| `event_id` | integer | |
| `debt_owner_id` | integer | Opaque id of the debt owner. Never a name or a document number. |
| `result_id` | integer | Foreign key to `contact_results`. |
| `agent_id` | integer | Foreign key to `agents`. |
| `occurred_at` | datetime | When the attempt happened. Date filter column for effort queries. |

### `contact_results`: the result catalogue

| Column | Type | Meaning |
|---|---|---|
| `result_id` | integer | Join key. Use the surrogate id, not a business code that may repeat. |
| `label` | text | |
| `answered` | bit | 1 when someone picked up. |
| `right_party` | bit | 1 when the person reached was the right one. |

Funnel definitions (all counts):

- `attempt_count` = number of events
- `answer_count` = events whose result has `answered = 1`
- `cpc_count` (right-party contact) = events whose result has `answered = 1 AND right_party = 1`

So `attempt_count >= answer_count >= cpc_count`, always. `cpc_count` is a count, never a
percentage.

### `agreement_statuses`: the status catalogue

| Column | Type |
|---|---|
| `status_id` | integer |
| `label` | text |

Only used by `SqlDataSource.catalog_problems()`, which compares your configured status ids with
the catalogue and reports an id that is missing or labelled differently.

### `agents`

| Column | Type | Meaning |
|---|---|---|
| `agent_id` | integer | Join key. |
| `display_name` | text | Shown in answers, for example `Agent 01`. |
| `login` | text | The public key of the agent, for example `agent_01`. Excluded-agent patterns match against it. |

## Status sets

Status ids are *configuration*, not catalogue facts. `BusinessConfig.status_ids` maps seven labels
to your ids; the demo uses invented numbers (10, 20, 30, ...). The sets are defined by label:

| Label | Meaning |
|---|---|
| `ACTIVE`, `PAID`, `PAID_MANUAL` | approved |
| `BROKEN`, `AUTO_BROKEN` | the agreement broke later, manually or automatically |
| `PENDING_EXCEPTION` | the exception: value pending approval |
| `REJECTED` | refused by a supervisor |

```
generated = approved + BROKEN + AUTO_BROKEN      (a broken agreement was still generated)
universe  = generated + PENDING_EXCEPTION        (the filter inside every query)
rollup    = universe  + REJECTED                 (the portfolio rollup)
```

Only `BROKEN` is on the *broken axis* of the risk calculation; `AUTO_BROKEN` counts as generated
but not as an axis. Change the sets in `BusinessConfig` (or in the JSON file named by
`AGENT_BUSINESS_CONFIG`) and the SQL, the analytics, the system prompt and the metric registry all
follow, because they all read the same object.

## Excluded agents

`BusinessConfig.excluded_agent_patterns` (default `("system%", "test%")`) are SQL `LIKE` patterns
matched case-insensitively against `agents.login`. They are applied in SQL, never after the
fact, so an excluded agent cannot leak into a total.

## Part 2: the nine rollup methods

`DataSource` is a `Protocol`. A method returns a list of plain dicts with lower-case keys. Rows
are additive: combining two product lines is concatenation, because every aggregation sums
rows that share a key and recomputes ratios afterwards.

| Method | Row keys |
|---|---|
| `status_rollup(window)` | `portfolio_name, status_id, qty, amount` |
| `daily_rollup(window, portfolio=None)` | `day, status_id, qty, amount` |
| `agent_rollup(window)` | `agent_id, agent_name, attempt_count, answer_count, cpc_count, agreement_count, agreement_value, first_installment_value, agreement_count_by_contract, slips_issued, slips_paid, exception_count, exception_value` |
| `agent_status_rollup(window)` | `agent_id, agent_name, status_id, qty, amount` |
| `crosstab_rollup(window, by, value)` | `agent_id, agent_name, portfolio_name, status_id, qty, amount`, filtered to one portfolio (`by="portfolio"`) or one agent (`by="agent"`) |
| `plan_stage_rollup(end)` | `portfolio_name, stage, qty, open_amount` with `stage` one of `early, middle, late, settled` |
| `slip_rollup(window, group_by, date_field, portfolio=None)` | `group_key, generated, paid_on_time, amount_maturing, amount_received`; `group_by` is `day`, `portfolio` or `none`; `date_field` is `issued_at` or `due_at` |
| `drilldown_rows(kind, portfolio, window)` | `agreement_id, portfolio_id, first_installment_value, total_value, agent_name, agent_id, issued_on, due_on, installments`; `kind` is `generated, exception, rejected, broken` |
| `hourly_counts(window)` | `day, hour, qty`: generated agreements per hour |

`window` is an inclusive `[start, end]` date range (`session.Window`). A method may raise
`DataSourceError` for a transient failure (the run guard retries once) or `DataUnavailable` when it
cannot serve the data at all.

Definitions every implementation must respect:

- `amount` in the status rollups is the first-installment amount, one per agreement, after
  de-duplication.
- `agreement_count` counts distinct `(agreement_id, portfolio_id)` pairs. Two agreements that share
  an `agreement_id` in different portfolios are two agreements, also when they land in the same
  group. `agreement_count_by_contract` counts `(agreement, portfolio, debt)` triples. The second is only
  for totals; ticket and conversion always use the first.
- `agreement_value` is the sum over *all* installments of generated agreements whose first
  installment was issued in the window.
- A slip is paid within the grace period when `paid_amount > 0` and
  `paid_at <= due_at + grace_days`.
- `plan_stage_rollup` covers approved agreements issued in the `plan_lookback_days` before `end`.
  An installment with `paid_amount > 0` is paid. Stage is `settled` when all are paid, `early`
  when `plan_early_max_paid` or fewer are, `late` when `plan_late_remaining` or fewer are left,
  else `middle`. The first rule that matches wins, in that order.

## Using SQL Server

`sources/sql.py` generates T-SQL as well as SQLite. Differences are isolated in `Dialect`:
`WITH (NOLOCK)` on every table, `CAST(x AS DATE)`, `DATEPART(hour, x)` and
`DATEADD(day, n, x)`.

```python
import os
from llm_agent_evals.business import BusinessConfig
from llm_agent_evals.sources import SQLSERVER, SqlDataSource, connect_sqlserver

config = BusinessConfig()
# reads DB_HOST, DB_NAME (or DB_NAME_AUTO), DB_USER and DB_PASSWORD
conn = connect_sqlserver(os.environ, "auto")
source = SqlDataSource(conn, SQLSERVER, config)
```

The generated T-SQL is unit-tested as text (every table hinted, every placeholder bound, a single
read-only statement), but it has never been executed against a live SQL Server. The SQLite
versions of the same queries are executed in the test suite against small in-memory databases.
`connect_sqlserver` is tested with an injected fake `connect`.

The connection string sets `ApplicationIntent=ReadOnly`, and `SqlDataSource` refuses to execute
any statement that is not a single `SELECT` (or `WITH ... SELECT`). Give the database login only
read permission as well: the guard is a second line of defence.
