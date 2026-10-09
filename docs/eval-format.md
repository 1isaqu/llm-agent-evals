# Evaluation format

A *scenario* is one JSON file in `scenarios/`. `llm-agent-evals eval` loads them all, runs each
through the real agent, and reports which expectations held.

```
llm-agent-evals eval                                 # scripted model, Markdown report, exit 1 on a miss
llm-agent-evals eval --format json --out report.json
llm-agent-evals eval --min pass_rate=100 --min faithfulness=100
llm-agent-evals eval --live                          # a real model, see "Live mode"
```

## What runs for real, and what is replaced

| Real | Replaced |
|---|---|
| the agent loop, the run guard, the tool registry and schema validation | the model: a script (default) or a real client (`--live`) |
| the analytics (risk, funnel, trend, stages, pace) | the SQL layer: `FixtureSource` serves the rows the scenario provides |
| the egress guard, the answer parser, the sanitiser | the clock: the scenario's `now`, and a fake monotonic clock |

In scripted mode this measures the machinery. The script is the behaviour the scenario calls
correct. A real model is measured only in live mode.

## Scenario file

```json
{
  "id": "S01",
  "title": "Risk of one portfolio, factual",
  "category": "factual",
  "question": "How risky is Portfolio A-02?",
  "session": {"product_line": "auto", "date_from": "2026-03-01", "date_to": "2026-03-18"},
  "now": "2026-03-18T15:30:00",
  "dataset": "datasets/basic.json",
  "dataset_patch": {"auto": {"agent_rollup": {"$fail": "connection reset"}}},
  "script": [
    {"tool_call": {"name": "get_portfolio_metrics", "args": {"portfolio_name": "Portfolio A-02"}}},
    {"final": {"text": "...", "highlights": [], "suggested_actions": [], "data_sources": [], "confidence": "high"}}
  ],
  "limits": {"max_steps": 8, "max_rounds": 8, "wall_clock_seconds": 60, "clock_tick_seconds": 0},
  "config": {"grace_days": 7},
  "expect": { "...": "see below" },
  "notes": "free text"
}
```

| Key | Required | Meaning |
|---|---|---|
| `id`, `title`, `category` | yes | `S` + number for golden scenarios, `X` + number for self-tests. The file is named `<id lower-case>_<slug>.json`. |
| `question` or `messages` | one of them | A single user question, or a whole conversation (`role` and `content`). |
| `session` | yes | `product_line` (`auto`, `consumer` or `all`) and an inclusive window. |
| `now` | yes | The agent's clock, an ISO timestamp. "Today" in the prompt and in the pace tool. |
| `dataset` / `datasets` | one of them | A path relative to the scenario file, or the rows inline. |
| `dataset_patch` | no | Replaces whole sections per product line on top of the loaded dataset. |
| `script` | yes | The model's turns, see below. Ignored in live mode. |
| `limits` | no | `max_steps`, `max_rounds`, `wall_clock_seconds`, and `clock_tick_seconds` (how far the fake clock moves on each read). |
| `config` | no | `BusinessConfig` overrides. |
| `expect` | yes | Expectations. Every key is optional. |

The loader is strict. An unknown key in the scenario, `session`, `limits`, `config`, `expect` or
`expect.answer` (`tools_forbiden`, `max_step`) is a `ScenarioError` that names the file. So is a
dataset section that is not a `DataSource` method name (`agent_rolup`), because a typo would
otherwise switch a check off or fail only when the scenario runs. `eval` prints the message and
exits with code 2.

### Datasets

A dataset file maps a product line to *sections*. A section is named after a `DataSource` method
(`status_rollup`, `agent_rollup`, ...) and holds either

- a plain list of rows, returned for every call, or
- a list of blocks `{"match": {...}, "rows": [...]}`: the first block whose `match` entries all equal
  the call's parameters supplies the rows. Parameters are the method's own arguments plus `start` and
  `end` (ISO dates) for windowed calls. A call that matches no block gets no rows.

Rows with a `day` key are also filtered to the requested window. A section can be
`{"$fail": "message"}` to make the call raise a transient error. A method with no section raises
`DataUnavailable`, so a scenario states exactly which data it supplies. Row keys are in
`docs/data-contract.md`.

### Script turns

Each turn has exactly one of these keys:

| Turn | Effect |
|---|---|
| `{"tool_call": {"name": ..., "args": {...}}}` | one tool call |
| `{"tool_calls": [{"name": ..., "args": {...}}, ...]}` | a batch in one turn |
| `{"tool_call": {"name": ..., "raw_args": "..."}}` | arguments that were not valid JSON |
| `{"final": {...}}` | the JSON answer (serialised for you) |
| `{"text": "..."}` | raw text, for malformed-output scenarios |
| `{"fail": "message"}` | the provider call fails |

Running past the end of the script is a harness failure (`harness:script_exhausted`), never a pass.

## Expectations

| Key | Layer | Meaning |
|---|---|---|
| `should_pass` | | `false` makes this a self-test. Default `true`. |
| `expect_violations` | | For self-tests: the `layer:code` ids that must appear. |
| `tools_any_of`, `tools_all_of`, `tools_forbidden` | tool_selection | At least one / all / none of these tools called. |
| `zero_tool_calls`, `max_tool_calls` | tool_selection | No calls at all, or at most N. |
| `params` | params | A list of `{tool, contains: {...}, absent: [...]}`. Needs one call to `tool` whose arguments contain every pair (strings compared case-insensitively) and none of the `absent` keys. |
| `answer.traceable_numbers` | faithfulness | Every number of 10 or more in the answer must match a number the model saw. |
| `answer.forbidden`, `contains`, `contains_any_of`, `contains_date` | answer_text | Wording rules, case-insensitive. |
| `answer.false_unavailability_phrases` | false_unavailability | Phrases that must not appear after a call returned rows. |
| `ground_truth` | ground_truth | A list of `{tool, path, equals, tolerance_pct, occurrence}`. A hand-computed value compared with what the tool returned (dotted path, list indices allowed). Booleans count as 0 or 1. Default tolerance 0.5%. |
| `tool_errors` | outcome | These `error_type` values must have been observed. |
| `confidence` | outcome | One of the listed values (a string or a list). |
| `degraded` | outcome | The degradation reason expected (`dataset_unavailable`, `unsafe_answer`), or none. |
| `forced_final` | outcome | Whether the loop had to force a tool-free final round. |
| `model_error` | harness | The provider is expected to fail. |
| `format_ok` | outcome | Default `true`: the final text must be a JSON object. |

## Assertion layers and violation ids

A violation is `Violation(layer, code, message, details)` and is referred to as `layer:code`.

| Layer | Codes |
|---|---|
| `tool_selection` | `missing_expected_tool`, `forbidden_tool`, `unexpected_tool_calls`, `too_many_tool_calls` |
| `sql_channel` | `tool_not_in_allowlist`, `sql_in_arguments` |
| `params` | `missing_params` |
| `faithfulness` | `untraceable_number` |
| `ground_truth` | `value_mismatch`, `missing_value` |
| `answer_text` | `forbidden_text`, `missing_text`, `missing_date` |
| `false_unavailability` | `claims_unavailable_after_data` |
| `outcome` | `confidence_mismatch`, `degraded_mismatch`, `forced_final_mismatch`, `tool_error_not_observed`, `not_json` |
| `harness` | `script_exhausted`, `model_error`, `model_error_missing`, `run_failed` |

The numeric check reads the numbers in the answer text and in each highlight `value`
(`1,234.50`, `23.8%`, `$2.35M`, `-31.11%`) and ignores ISO dates. It compares each value of at
least 10 with every number in the tool results the model saw (also numbers inside strings, such
as the 12 in `Portfolio A-12`) and in the question. A match is within 1% (at least 0.5). Values
of at least 100 may also match the sum of two such numbers. It catches an invented figure. It
cannot catch a real figure on the wrong label, and it reports a total that a model added up from
three or more numbers.

The false-unavailability check uses the guard's own telemetry for each call (`error_type` and
`row_count`), not the script, so it measures what the tools actually returned.

## Metrics

Computed from the `layer` field only.

| Metric | Population | Meaning |
|---|---|---|
| `pass_rate` | golden | share of scenarios with no violation at all |
| `tool_selection_accuracy` | golden | share clean in `tool_selection` and `sql_channel` |
| `param_accuracy` | golden | share clean in `params` |
| `faithfulness` | golden | share clean in `faithfulness`, `ground_truth` and `false_unavailability` |
| `answer_accuracy` | golden | share clean in `answer_text` and `outcome` |
| `steps_per_question` | golden | mean number of tool calls |
| `detection_rate` | self-tests | share that produced the violations they declare |
| `meets_expectation_rate` | all | share that behaved as declared |

A metric with no scenarios in its population is `None`, and a gate that asks for it fails: no
evidence is not a pass.

## Live mode

`llm-agent-evals eval --live` uses the model named by `LLM_PROVIDER`, `LLM_API_KEY` and `LLM_MODEL`.
The script is ignored, so the model chooses the tools itself, and the same expectations judge it.
Self-tests are skipped (they depend on a script that misbehaves on purpose). The exit code is 1 only
when a `--min` gate fails: a live model is expected to miss some expectations, and the report says
which. It costs real tokens. It has not been run against a live provider while building this
repository, and CI never runs it.

## Not built: an LLM judge

There is no layer that asks a second model to grade tone or completeness. It would need a live
model and a rubric, and results that vary between runs do not belong in a deterministic gate. The
ledger lists it as an open item.

## Writing a scenario

1. Copy the closest scenario. Keep invented names (`agent_01`, `Portfolio A-01`).
2. Compute the numbers you assert by hand from the rows you wrote. A `ground_truth` entry is only
   worth anything if you did not get its value by running the code.
3. Add a self-test (`X...`) for any new layer or code, with `expect_violations`.
4. Run `llm-agent-evals eval`. If the scenario is evidence for a ledger entry, add its id there and
   run `llm-agent-evals ledger verify`.
