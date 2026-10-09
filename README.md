# llm-agent-evals

A tool-calling analytics agent together with the harness that evaluates it: golden scenarios, a
metric registry with caveats, a reliability ledger, a deterministic mock model for CI, and a real
model that plugs in through environment variables. The domain is debt collections (portfolios,
agents, agreements, risk). Most of the code is about making the agent testable and checking what
it does.

## What it does and why it is interesting

An executive asks a question in plain language. The agent calls read-only tools over rollup data
and answers with a small JSON contract (text, highlights, suggested actions, sources, confidence).
The engineering effort goes into making that loop checkable.

The agent loop is written once, over a neutral message type. Provider wire formats live in thin
adapters, so message alternation errors, compaction and the guards exist in one place. A run guard
is enforced inside tool dispatch: it sets step and time ceilings, memoises identical calls, detects
call spirals, can disable a single tool, retries a transient source failure once, caps the size of a
result, shares a cache and has a circuit breaker. Time, sleep, cache and breaker are injected, so
tests need no patching.

A tool is one object that holds its schema and its handler. The schema the model sees also
validates its arguments, and the tool list in the prompt is generated from the registry. Model
mistakes come back as structured errors with a hint, not as stack traces.

`BusinessConfig` drives the SQL, the analytics, the system prompt and the metric registry, so the
four cannot disagree. Some of the rules it carries: composite risk is the maximum axis and never
the sum; conversion, paid-per-CPC and slip effectiveness are three metrics with three names; rows
from two product lines are added before any ratio is computed.

The evaluation harness has eight assertion layers: tool selection, SQL channel, parameters,
numeric faithfulness, ground truth, answer wording, false unavailability and outcome. Violations
are structured (`layer:code`) and scores come from the layer, not from searching a message.
Self-test scenarios script a mistake on purpose and declare the violation the harness must report,
which shows that each layer can fail.

The reliability ledger is data. A verifier resolves every scenario and test it cites, and a test
fails when the generated documents drift.

The data contract has no personal identifier, and an egress guard inspects every tool result
before a model sees it. At runtime the package uses only the standard library, with no provider
SDK, web framework or pydantic.

## Architecture

```
 question
    |
 service.py / wsgi.py   validation, gates, envelope            POST /chat
    |
 agent.py               load base data -> prompt -> loop -> parse answer
    |   model turn <-----------------------------+
    |   tool calls                                |
    v                                             |
 guards.RunState.dispatch   budget, memo, cache, retry, cap
    v
 registry.ToolRegistry      schema validation, cross-field rules
    v
 tools/ (15)  ->  analytics/ (pure functions)  ->  sources/ (DataSource protocol)
                                                      |-- SqlDataSource  SQLite and SQL Server
                                                      |-- FixtureSource  canned rows for scenarios
    v
 privacy.EgressGuard        redact or block, then back to the model

 llm/   ScriptedLLM, RuleBasedLLM (mocks)  |  OpenAICompatibleLLM, AnthropicLLM (urllib)
 evals/ scenarios -> runner -> checks -> metrics -> report       ledger/ entries -> verify -> render
```

More detail in `docs/architecture.md`; the reasons behind the choices in `docs/decisions.md`.

## Quick start

Requires Python 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -e ".[dev]"

llm-agent-evals ask "Which portfolios are at high risk?"
llm-agent-evals eval               # runs all scenarios with the scripted model
python -m pytest
```

With the default settings no key and no network are needed. The agent runs on invented
in-memory data (two product lines, a pinned clock of 2026-03-18 15:30) and the offline
rule-based mock model:

```
$ llm-agent-evals ask "Which portfolios are at high risk?"
The tool returned 1 row(s). First: Portfolio A-01 (composite_risk_pct 52.17, first_installment_value 820.0).

confidence: high
tools: filter_portfolios_by_risk
```

The mock is a keyword router that repeats values from the tool result. It is not a language
model. Its job is to let the pipeline run end to end anywhere.

The HTTP service (plain WSGI, bind it to localhost):

```bash
llm-agent-evals serve --port 8000
curl -s -X POST http://127.0.0.1:8000/chat -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Which portfolios are at high risk?"}],
       "product_line":"auto","date_from":"2026-03-01","date_to":"2026-03-18"}'
```

Other commands: `llm-agent-evals tools` (the 15 tools), `prompt` (the rendered system prompt),
`metrics` (the metric registry), `ledger verify` and `ledger generate`.

## Using a real model

Set environment variables; the key is read nowhere else.

```bash
export LLM_PROVIDER=anthropic          # or openai_compatible
export LLM_API_KEY=...                 # never commit it
export LLM_MODEL=...                   # the model id you want to use
llm-agent-evals ask "How is Portfolio A-01 doing?"
llm-agent-evals eval --live --min pass_rate=80
```

`openai_compatible` works with any server that speaks the chat-completions protocol (set
`LLM_BASE_URL`). Both adapters are tested through an injected transport and against a local fake
provider on `127.0.0.1`. They have not been called against a live provider. `eval --live`
skips the self-tests and costs real tokens.

## Connect your own data

The agent sees nine rollup methods, and `sources/sql.py` implements them over eight neutral tables.
Either map your database to those tables (views are fine) or implement the `DataSource` protocol
yourself. The tables, the columns, the status sets and the row shapes are in
`docs/data-contract.md`. Then calibrate `BusinessConfig` on your own data: every threshold in
it is invented.

```python
import os
from llm_agent_evals.agent import Agent
from llm_agent_evals.business import BusinessConfig
from llm_agent_evals.llm import build_llm
from llm_agent_evals.session import Session, Window
from llm_agent_evals.settings import LLMSettings
from llm_agent_evals.sources import SQLSERVER, CachedSource, SqlDataSource, connect_sqlserver

config = BusinessConfig(risk_low_max=10.0, risk_medium_max=30.0, grace_days=7)  # calibrate these
sources = {
    line: CachedSource(SqlDataSource(connect_sqlserver(os.environ, line), SQLSERVER, config))
    for line in ("auto", "consumer")
}
agent = Agent(build_llm(LLMSettings.from_env()), sources, config=config)
result = agent.run(
    [{"role": "user", "content": "Which portfolios are at high risk?"}],
    Session("auto", Window.from_iso("2026-03-01", "2026-03-18")),
)
print(result.response["text"], result.response["confidence"])
```

`pip install ".[sqlserver]"` adds `pyodbc`. `ask` and `serve` also accept `--backend sqlserver`,
which opens one read-only connection per product line from the `DB_*` variables. The SQL Server
path (T-SQL generation and the connection helper) is tested as text and with an injected driver,
not against a live server. For another database, implement `DataSource` and skip
`sources/sql.py`.

## Configuration

Environment variables (see `.env.example`; the package does not read `.env` files by itself):

| Variable | Default | Meaning |
|---|---|---|
| `LLM_PROVIDER` | `mock` | `mock`, `openai_compatible` or `anthropic` |
| `LLM_API_KEY` | none | Required for a real provider. Never logged, never in an error. |
| `LLM_BASE_URL` | per provider | Must be `http` or `https` |
| `LLM_MODEL` | none | Required for a real provider |
| `LLM_TIMEOUT_SECONDS` | `30` | Per HTTP call |
| `LLM_MAX_RETRIES` | `1` | Extra attempts for 408, 409, 429, 5xx and network errors |
| `LLM_MAX_TOKENS` | `2048` | Output cap per turn |
| `LLM_TEMPERATURE` | `0.2` | A number, or `none` to omit it |
| `AGENT_MAX_ROUNDS` | `8` | Model turns that may call tools |
| `AGENT_MAX_STEPS` | `8` | Tool calls per request |
| `AGENT_WALL_CLOCK_SECONDS` | `60` | Time budget per request |
| `AGENT_CACHE_TTL_SECONDS` | `45` | Result cache of the demo and SQL backends |
| `AGENT_EGRESS_POLICY` | `redact` | `redact` or `block` personal-looking values |
| `AGENT_BUSINESS_CONFIG` | none | Path to a JSON file overriding `BusinessConfig` |
| `AGENT_TELEMETRY_PATH` | none | JSON-lines file of safe events (no arguments, no data) |
| `AGENT_CHAT_ENABLED` | `true` | Feature switch for the HTTP endpoint |
| `DB_DRIVER`, `DB_HOST`, `DB_USER`, `DB_PASSWORD` | placeholders | SQL Server connection |
| `DB_NAME`, `DB_NAME_AUTO`, `DB_NAME_CONSUMER` | placeholders | One database, or one per product line |

`BusinessConfig` parameters. Every default is invented and has to be calibrated on your own data:

| Parameter | Default | Meaning |
|---|---|---|
| `status_ids`, `approved`, `broken`, `auto_broken`, `exception`, `rejected` | ids 10 to 70 | Which status ids mean what |
| `risk_low_max`, `risk_medium_max` | `20.0`, `40.0` | Composite-risk cut-offs for `low` and `medium` |
| `grace_days` | `3` | A slip paid within this many days after the due date counts as on time |
| `excluded_agent_patterns` | `system%`, `test%` | SQL `LIKE` patterns on the agent login, applied in SQL |
| `first_installment_no` | `0` | The installment number of the down payment |
| `max_window_days` | `93` | Longest window a single tool call may span |
| `plan_lookback_days`, `plan_late_remaining` | `150`, `3` | Plan-stage base and the "late" threshold |
| `crosstab_limit`, `stage_portfolio_limit` | `12`, `8` | Rows returned by those views |
| `pace_first_hour`, `pace_last_hour`, `pace_min_history_days` | `9`, `18`, `3` | Daily-pace window |

Guard numbers (`RunGuard`: ceiling of 8 steps, 60 seconds, 1,500 tokens per result, a 12,000-token
request budget) are engineering defaults you can pass to `Agent(guard=...)`.

## Run the tests

```bash
python -m pytest                                  # unit and integration tests, no network
llm-agent-evals eval --min pass_rate=100          # the scenario set
llm-agent-evals ledger verify                     # every ledger reference resolves
llm-agent-evals metrics --check docs/metric-registry.default.json
ruff check . && ruff format --check .
```

The tests use small inline fixtures with invented values. SQL is executed on in-memory SQLite. The
HTTP adapters are also driven end to end against a throwaway fake provider on `127.0.0.1`. Nothing
needs a key or a database. CI (`.github/workflows/ci.yml`) runs the same commands with the mock
model only.

The suite was run locally on Python 3.14. CI runs it on 3.11 to 3.14.

## Project layout

```
src/llm_agent_evals/
  agent.py  guards.py  registry.py  prompt.py  response.py  compaction.py  privacy.py  telemetry.py
  business.py  settings.py  session.py  context.py  errors.py  jsonschema_lite.py  metrics.py
  service.py  wsgi.py  envelope.py  runtime.py  cli.py  system_prompt.md
  analytics/   risk, funnel, series, crosstab, stages, drill-down, effectiveness, pace
  tools/       the 15 tools
  sources/     DataSource protocol, SQL builders, SQL source, fixture source, demo data, schema
  llm/         neutral types, scripted and rule-based mocks, OpenAI-style and Anthropic adapters
  evals/       scenario format, assertion layers, runner, metrics, reports
  ledger/      model, verifier, renderers
scenarios/     32 golden scenarios (S), 13 self-tests (X), datasets/basic.json (invented rows)
ledger/        entries.json, the source of docs/reliability-ledger.md
docs/          architecture, data-contract, decisions, eval-format, ledger, glossary,
               metric-registry.default.json, reliability-ledger.md and .html
tests/         pytest suite
```

## Key design decisions

Short list; the full reasoning is in `docs/decisions.md`.

- Guards are enforced inside `dispatch`, because a model can return a batch of calls in one turn.
- Model mistakes are structured return values (`error_type`, `retryable`, `hint`), not exceptions.
- Analytics are pure functions over rollup rows, so every scenario runs the real analytics with
  canned rows and only the SQL needs a database to be exercised.
- Business filters sit inside the de-duplication subquery, before the row ranking.
- A numeric faithfulness check is a safety net: it catches an invented number, not a wrong label.
  Hand-computed ground truth and forbidden wording cover that gap per scenario.
- The ledger and the metric registry snapshot are generated, and tests fail on drift.

## Limitations

No live model provider and no live SQL Server was used. The two model adapters and the T-SQL are
verified through injected fakes and generated text, so expect to adjust them on first contact.
Scripted scenarios measure the machinery, not a model. A real model is measured only by
`eval --live`, which has never been run here, and there is no judge layer for tone or
completeness. The rule-based mock produces terse answers meant to prove the plumbing, not to read
well.

Every threshold is invented. The risk cut-offs, grace period, window limit and guard budgets show
where calibrated values go, and the agent's statements about risk level mean nothing until they
are calibrated on your own data. The daily pace is a baseline (the mean hourly profile of recent
history), not a forecast model. The `PaceProvider` protocol is where one would plug in.

The egress guard is a safety net. It cannot recognise a name in free text, it can redact a long id
by mistake, and it does not inspect the user's own message. The HTTP layer has no authentication,
rate limiting or TLS, so bind it to localhost or put it behind something that provides them. The
cache and the circuit breaker are per process. The tool descriptions, the prompt and the answers
are in English only. The `debts` table of the contract is not read by any query yet.

## Independence

This is an independent portfolio project. It is not affiliated with any company, it contains no
real data, and every name, number and portfolio in it is invented.

## License

MIT, see `LICENSE`.
