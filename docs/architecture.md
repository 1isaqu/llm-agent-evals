# Architecture

The package has five layers. Each depends only on the ones below it, and nothing below the agent
layer knows that a language model exists.

```
 evaluation   evals/  ledger/  cli.py                       scenarios, assertion layers, metrics, ledger
     |
 service      service.py  wsgi.py  runtime.py               request validation, envelope, wiring
     |
 agent        agent.py  guards.py  registry.py  tools/      the loop, the run guard, the tool registry
              prompt.py  response.py  compaction.py  llm/   prompt, answer contract, model clients
              privacy.py  telemetry.py
     |
 analytics    analytics/  metrics.py  business.py           pure functions over rollup rows
     |
 data         sources/                                       DataSource protocol, SQL builders, fixtures
```

## One request, end to end

```
POST /chat  ->  ChatService.handle
                  gates (enabled? model configured?)  ->  validate and trim messages  ->  Session
                  Agent.run(messages, session)
                    1. load the portfolio rollup for the session           (failure -> degraded answer)
                    2. build the system prompt: rules + tool summary + session context + date anchors
                    3. loop, at most max_rounds + 1 times
                         model turn  <-  ModelRequest(system, messages, tools, allow_tools)
                         for each tool call:
                             RunState.dispatch(...)       step and time ceilings, memo, spiral, cache,
                                                          breaker, retry, result cap
                               ToolRegistry.call(...)     schema validation, cross-field rules
                                 tool handler             analytics over DataSource rows
                             EgressGuard.inspect(result)  redact or block personal-looking values
                             result goes back to the model as a tool message
                         budget near its limit?  ->  compact the oldest tool groups
                         guard says stop (or last round)?  ->  nudge to answer, tools disabled
                    4. parse the last text into the answer contract; replace unsafe text
                  envelope: {meta, data: [answer], errors: []}
```

## Modules

| Module | Responsibility |
|---|---|
| `business.py` | `BusinessConfig`: status sets, risk cut-offs, grace days, window limit, exclusion patterns. Every default is invented. |
| `settings.py` | Environment variables to `LLMSettings` and `AgentSettings`. The only place the API key is read. |
| `session.py` | `Window` (inclusive date range) and `Session` (product line plus window). |
| `sources/base.py` | `DataSource` protocol, `CachedSource`, `MultiLineSource`. |
| `sources/sql.py` | `ContractQueries`: every query as text plus positional parameters, two dialects, `ensure_read_only`. |
| `sources/sqlsource.py` | `SqlDataSource` (runs queries on a DB-API connection), the SQL Server connection helper, merging of per-agent aggregates. |
| `sources/fixture.py` | `FixtureSource`: canned rollup rows, used by every scenario. |
| `sources/demo.py` | Invented in-memory SQLite databases for the demo and the tests. |
| `analytics/` | Portfolio risk entries, agent funnel entries, daily series and trend, crosstab and ranking, plan stages, drill-downs, slip effectiveness, daily pace. Pure functions. |
| `metrics.py` | `MetricRegistry`: status sets, KPI formulas and caveats, built from the config. Served by `explain_metric`. |
| `registry.py` | `Tool` and `ToolRegistry`: schema, description, handler and validator in one object. |
| `jsonschema_lite.py` | A small JSON-schema subset validator with coercion of harmless mistakes. |
| `tools/` | The 15 tools. Each module registers its tools with a registry. |
| `guards.py` | `RunState` (per request), `ToolCache` and `CircuitBreaker` (shared across requests). |
| `privacy.py` | `EgressGuard`: last check before a tool result reaches a model. |
| `prompt.py`, `system_prompt.md` | Template, placeholders filled from the config, tool summary generated from the registry, date anchors. |
| `response.py` | The answer contract: tolerant parsing, coercion, sanitising. |
| `compaction.py` | Replaces old tool-call groups with a short note when the budget nears its limit. |
| `llm/` | Neutral message types, the scripted mock, the rule-based mock, and two HTTP adapters over `urllib`. |
| `agent.py` | `Agent.run` and the loop. |
| `telemetry.py` | Observer protocol and a JSON-lines writer that never carries arguments or data. |
| `service.py`, `wsgi.py`, `envelope.py` | The chat endpoint without a framework. |
| `runtime.py` | Builds an `Agent` from the environment and a backend (`demo` or `sqlserver`). |
| `evals/` | Scenario format, assertion layers, runner, metrics, reports. See [docs/eval-format.md](eval-format.md). |
| `ledger/` | Ledger model, verifier and renderers. See [docs/ledger.md](ledger.md). |

## Extension points

To add a tool, write a handler `(ctx, args) -> result`, wrap it in a `Tool` with a closed input
schema and register it. The schema the model sees, the validation and the code are one object. The
system prompt's tool list is generated from the registry, so it picks the tool up without edits.
Add a scenario for it.

To add a data backend, implement the nine methods of `DataSource` (see
[docs/data-contract.md](data-contract.md)) and pass `{"line_name": source}` to `Agent`. Wrap it in
`CachedSource` if the queries are slow.

To add a model provider, implement `LLMClient.complete(ModelRequest) -> ModelTurn`. The two shipped
adapters are about 130 to 140 lines each and only translate between the neutral types and one wire
format.

To use a different pace model, implement `PaceProvider.forecast(source, now, config)` and pass it
to `Agent(pace=...)`. The shipped baseline is a mean hourly profile, and no fitted artifact is
included.

To change a business definition, edit `BusinessConfig` or point `AGENT_BUSINESS_CONFIG` at a JSON
file. Then run `llm-agent-evals metrics --write docs/metric-registry.default.json`, because a test
fails when the snapshot and the code disagree.

## What is injected, and why

Time, sleeping, the monotonic clock, the result cache, the circuit breaker, the model, the data
sources and the egress guard are all constructor arguments. No test patches a module global or the
system clock, and a scripted scenario run gives the same result every time.
