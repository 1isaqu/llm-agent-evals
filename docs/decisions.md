# Key decisions

Each entry says what was decided, why, and what it costs.

## 1. One loop over a neutral message type

The conversation is a list of `Message(role, content, tool_calls, tool_call_id)`. Wire adapters
translate it to one provider's JSON and back. The loop, the compaction and the guards therefore
exist once. Two providers with two near-identical loops drift apart, and one of them gets a fix
the other never sees.

A second benefit is that a whole class of provider error cannot be built. One API refuses two
consecutive `user` messages, and the loop naturally produces them (a tool result followed by a
"answer now" nudge). The adapter merges consecutive user-side items into one message, tool results
first, so the loop never has to know.

*Cost:* an adapter is a translation layer that can be wrong. The two shipped ones are only tested
through an injected transport.

## 2. Schema, description, handler and validator are one object

A `Tool` carries everything. The schema the model sees is the schema that validates its arguments,
and the tool list in the system prompt is generated from the registry. There is no second copy of
an enum or a description to go out of date. Schemas must be closed (`additionalProperties: false`)
and registration fails at startup otherwise, so a hallucinated argument is rejected instead of
ignored.

## 3. Model mistakes are return values, not exceptions

Unknown tool, bad arguments, an unknown name and a failed cross-field rule all return one structured
payload (`ok: false`, `error_type`, `retryable`, `hint`, `user_facing`). The hint says what to do
next. Exceptions are reserved for code bugs and infrastructure failures, and those belong to the
run guard.

## 4. The guard enforces its ceilings inside `dispatch`

A model can return several tool calls in one turn. A check that runs between turns dispatches the
whole batch even if the limit was reached at the first call. `RunState.dispatch` checks the step
and time budget before every call. The same method owns memoisation, spiral detection, per-tool
disabling after three consecutive failures, one retry with jitter, the result cap and the shared
cache and breaker.

## 5. Injected time, sleep, cache and breaker

The cache and the circuit breaker are objects handed to the agent, not module globals, and every
clock is a parameter. Tests build their own `SharedState` and a fake clock. Nothing patches
`time`. The cost is a longer constructor.

## 6. Analytics work on rollup rows, not on SQL

`DataSource` has nine methods that return plain dicts. Everything interesting (the single
denominator, composite risk, funnel ratios, trend, re-bucketing, stage thresholds, pace) is a pure
function over those rows. The SQL layer is replaceable, and every scenario can run the real
analytics with canned rows. Only the SQL itself needs a database to be exercised.

## 7. Rows are additive across product lines

Two lines are combined by concatenating rows, then recomputing every ratio once. Averaging two
conversion rates is wrong (a 50% line with ten contacts and a 10% line with ninety combine to
14%, not the 30% a plain average of the two rates gives). Because the rows are counters, a merge
cannot be done wrong by accident.

## 8. Risk: one denominator, composite is the maximum

Every risk percentage is a slice of the same universe (generated + exception + rejected value).
Broken value is a subset of generated, so the axes overlap and adding them double counts. A
portfolio's risk is its worst axis. Since every axis is a share of a universe that contains it, an
axis can leave 0 to 100 percent only when amounts are negative or sources disagree, so that is
reported as an `anomaly` and never "fixed".

## 9. Three different ratios keep three different names

Conversion (agreements over right-party contacts), paid-per-CPC and slip effectiveness are easy to
confuse. They have different field names, different tools, and the effectiveness series does not
use conversion field names at all. Conversion is only defined per agent, and the tool descriptions
tell the model to say so rather than substitute another metric.

## 10. De-duplicate first, and filter inside the derived table

A re-written installment would otherwise be counted twice. Every query keeps the latest row per
`(agreement_id, portfolio_id, installment_no)`. The window and status filters sit *inside* the
ranking subquery. With the filter outside, a late re-write that falls outside the filter wins the
rank and hides the valid row.

*Cost:* an agreement whose latest row has a status outside the sets is judged by its latest row
that is inside them. That is the documented behaviour and it is tested.

## 11. Aggregates are separate queries, merged in Python

The agent funnel is four small queries (effort, financial, agreement value, contract grain) merged
per agent, not one coupled statement. Each can be checked alone, and a join cannot multiply rows
across aggregates.

## 12. Two mocks with different jobs

`ScriptedLLM` replays a fixed script and is what scenarios use. `RuleBasedLLM` answers any question
from keyword rules, so the CLI, the HTTP service and a fresh clone work with no key. Neither uses
randomness or the network. Neither pretends to be a model.

## 13. Scenarios measure the machinery; self-tests measure the harness

In scripted mode the model's behaviour is the script, so a passing scenario proves the loop, the
guards, the analytics and the checks handle that behaviour. That says nothing about a real model.
Self-test scenarios script a mistake and declare the violation the harness must report. A harness
that silently stops detecting a class of mistake lowers the detection rate, which no golden score
would show. Live mode runs the same expectations against a real model, at real cost, and skips
the self-tests.

## 14. Violations are structured; scores come from the `layer` field

A check returns `Violation(layer, code, message)`. Metrics group by layer. Rewording a message
cannot change a score, and nothing attributes a failure to a metric by searching for a word.

## 15. The numeric check is a safety net, not a proof

It extracts numbers from the answer and requires every number of 10 or more to match a number the
model actually saw (in a tool result, in a string of a result, or in the question), or, for values
of 100 and above, the sum of two such numbers. It catches an invented number. It cannot catch a
real number on the wrong label, which `forbidden` wording and hand-computed ground truth are for.
The two-number sum is limited to large values because among a few dozen small counts almost any
small number is the sum of two of them.

## 16. Personal data never enters the contract, and a guard checks anyway

The data contract has no name, document, phone or e-mail field. If a user maps a free-text column
by mistake, the egress guard inspects every tool result before a model can see it. It redacts (or
blocks) denied field names and e-mail, phone-like and long-digit shapes. It is a safety net: it
cannot recognise a name in free text.

## 17. Telemetry carries no arguments, results or text

A tool-call event holds the tool name, a hash of the canonical arguments, timing and the guard's
outcome. That is enough to debug a slow tool and never puts business data in a log. Observers are
shielded: a failing observer cannot fail a request.

## 18. The ledger is data

A page of claims decays. The ledger is a JSON file; a verifier resolves every scenario, test and
module it cites; a generator renders Markdown and HTML; a test fails when the committed documents
differ from the generated ones.

## 19. Standard library only at runtime

The model clients use `urllib`, the validator is a small JSON-schema subset, the service is plain WSGI and
the SQL driver is an optional extra imported lazily. The package installs with no dependencies and
the tests run offline. The cost is that none of these is as complete as the library it replaces.
