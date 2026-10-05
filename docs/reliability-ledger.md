# Agent reliability ledger

_Generated from `ledger/entries.json` by `llm-agent-evals ledger generate`. Edit the data, not this file._

Failure classes of a tool-calling analytics agent, what closed each one, and the scenario or test that keeps it closed. An entry is only `closed` when it cites evidence that `llm-agent-evals ledger verify` can resolve.

**Status:** 16 closed, 7 partially closed, 2 open, 6 landmines flagged.

## Closed (16)

### L-01: Formula explained from memory

- **Finding.** A model asked how a metric is calculated answers from general knowledge and states a formula that differs from the one the numbers use.
- **Fix.** `explain_metric` serves definitions from a registry built from the active business configuration. The prompt requires a lookup before any explanation, and a committed snapshot of the registry is compared with the code in a test.
- **Evidence.** scenario S15 (ok); test tests/test_metric_registry.py::test_the_committed_snapshot_matches_the_code; test tests/test_tools.py::test_explain_metric_serves_the_registry_and_its_aliases; module src/llm_agent_evals/metrics.py
- **Residual risk.** Nothing in code forces the lookup before an explanation. The rule lives in the prompt, so a live model can still skip it.

### L-02: One word for three ratios

- **Finding.** Conversion (agreements over right-party contacts), paid-per-CPC and slip effectiveness are different ratios that are easy to call by the same name.
- **Fix.** Separate field names and separate tools. Effectiveness points do not carry conversion field names. Conversion exists only per agent and the tool descriptions say so.
- **Evidence.** scenario S03 (ok); scenario S28 (ok); scenario X05 (detected); test tests/test_analytics_agents.py::test_conversion_and_paid_per_cpc_are_two_different_metrics; test tests/test_analytics_views.py::test_effectiveness_points_never_use_the_conversion_field_names; test tests/test_metric_registry.py::test_conversion_effectiveness_and_paid_per_cpc_are_three_different_entries
- **Residual risk.** Whether free text uses the right label is checked only through forbidden strings that a scenario author thought of.

### L-03: Overlapping risk axes added together

- **Finding.** Exception, broken and rejected shares overlap (broken is part of generated). Adding them double counts and overstates risk.
- **Fix.** Composite risk is the maximum axis, every axis is a share of one common universe, and the registry and the prompt both state it.
- **Evidence.** scenario S02 (ok); test tests/test_analytics_portfolio.py::test_composite_risk_is_the_maximum_axis_never_the_sum; test tests/test_analytics_portfolio.py::test_every_axis_is_a_share_of_the_same_universe
- **Residual risk.** A model can still add axes in prose. The scenario catches the one sum it knows about.

### L-04: Corrupt rows read as extreme risk

- **Finding.** Negative amounts push an axis outside 0 to 100 percent. Reported as risk, it looks like a catastrophe instead of bad data.
- **Fix.** Entries carry an `anomaly` flag when an axis leaves the range. The prompt requires an explicit alert and forbids correcting the number.
- **Evidence.** scenario S04 (ok); test tests/test_analytics_portfolio.py::test_a_negative_amount_is_flagged_as_an_anomaly_and_not_corrected; test tests/test_analytics_portfolio.py::test_clean_data_can_never_leave_the_zero_to_hundred_range
- **Residual risk.** Only corruption that leaves the range is detected. Wrong but plausible values pass.

### L-05: Runaway tool loop

- **Finding.** A model repeats the same call, or alternates between two calls, until the budget is gone.
- **Fix.** Identical calls are memoised and the third declares a loop. An A, B, A, B pattern over the last four calls is detected. Either ends tool use and forces a final answer.
- **Evidence.** scenario S19 (ok); test tests/test_guards.py::test_identical_calls_are_memoised_and_the_third_declares_a_loop; test tests/test_guards.py::test_an_a_b_a_b_spiral_declares_a_loop; test tests/test_agent.py::test_three_identical_calls_declare_a_loop_and_end_the_run; module src/llm_agent_evals/guards.py
- **Residual risk.** A loop through slightly different arguments is bounded only by the step budget.

### L-06: A batch of calls slips past the budget

- **Finding.** A model can return many tool calls in one turn. A check that runs between turns dispatches the whole batch even when the limit is already reached.
- **Fix.** The step and wall-clock ceilings are checked inside `dispatch`, before every call.
- **Evidence.** scenario S20 (ok); test tests/test_guards.py::test_step_budget_is_enforced_inside_dispatch_for_a_batch_of_calls; test tests/test_guards.py::test_wall_clock_budget_is_enforced_inside_dispatch
- **Residual risk.** One slow call can still overrun the wall clock by its own duration. There is no per-call timeout beyond what the database driver provides.

### L-07: Oversized tool results

- **Finding.** A large result either fills the context or, cut in the middle, becomes JSON that the model cannot parse.
- **Fix.** Results are capped as a whole into a warning envelope with a preview. Tool results count against a request budget, and old tool groups are compacted into a short note when it nears the limit.
- **Evidence.** test tests/test_guards.py::test_an_oversized_result_becomes_a_warning_envelope_not_cut_json; test tests/test_guards.py::test_dispatch_applies_the_cap_and_reports_it; test tests/test_compaction.py::test_old_groups_are_replaced_by_one_note_and_the_newest_group_is_kept_intact; test tests/test_agent.py::test_the_conversation_is_compacted_when_the_tool_budget_nears_its_limit
- **Residual risk.** Token counts are a characters-per-token estimate, not a tokenizer.

### L-09: Silent first match

- **Finding.** A name fragment that matches several portfolios returns the first one, presented as if it were the only candidate.
- **Fix.** Results carry an `ambiguity_warning` naming the other matches, and the prompt requires the model to pass it on and ask which one was meant. Agent comparison matches ids exactly, never by fragment.
- **Evidence.** scenario S08 (ok); test tests/test_tools.py::test_the_ambiguity_warning_names_the_other_matches_only_for_fragments; test tests/test_tools.py::test_get_portfolio_metrics_by_exact_name_and_by_fragment_with_a_warning; test tests/test_tools.py::test_compare_agents_matches_ids_exactly_and_refuses_system_accounts
- **Residual risk.** Passing the warning on to the user is a prompt rule. Only the scenario checks it.

### L-10: Guessed names after a not-found

- **Finding.** After a lookup fails, a model tries other names it makes up.
- **Fix.** A not-found result is not retryable and lists the real names. The prompt forbids guessing and tells the model to offer the listed ones.
- **Evidence.** scenario S21 (ok); test tests/test_tools.py::test_an_unknown_portfolio_lists_the_real_ones; test tests/test_tools.py::test_agent_performance_found_and_not_found

### L-12: SQL or internals in the final answer

- **Finding.** The answer contains SQL in a code block, a dump of a raw tool envelope or a stack trace.
- **Fix.** A sanitiser replaces such text with a fixed refusal and never echoes it. The data source rejects any statement that is not a single read-only SELECT before it runs.
- **Evidence.** scenario S24 (ok); scenario S07 (ok); test tests/test_errors.py::test_sanitizer_refuses_sql_stack_traces_and_raw_envelopes; test tests/test_agent.py::test_an_unsafe_final_answer_is_replaced; test tests/test_response.py::test_unsafe_text_is_replaced_and_never_echoed; test tests/test_sql.py::test_read_only_guard_rejects_everything_else
- **Residual risk.** The sanitiser is pattern based. SQL written as plain prose passes.

### L-14: Cached result served to another window

- **Finding.** A cache keyed only by tool and arguments serves one session the result of another session that looks at a different period.
- **Fix.** The session line and window are part of the cache key.
- **Evidence.** test tests/test_guards.py::test_the_cache_key_includes_the_session_window; test tests/test_guards.py::test_different_windows_do_not_share_cached_results; test tests/test_agent.py::test_a_shared_cache_serves_the_same_call_in_a_later_request
- **Residual risk.** The cache is per process and has no invalidation inside its time to live.

### L-15: Relative dates worked out by the model

- **Finding.** A model resolves last quarter or yesterday from its own idea of today and uses the wrong period.
- **Fix.** The agent computes explicit dates for today, yesterday, this week, this month, last month, this quarter and last quarter from an injected clock and lists them in the session context. Windows are validated before any query runs.
- **Evidence.** scenario S09 (ok); scenario S14 (ok); scenario S27 (ok); test tests/test_prompt.py::test_date_anchors_are_computed_from_the_given_day; test tests/test_prompt.py::test_open_periods_end_today_not_at_the_end_of_the_period; test tests/test_tools.py::test_window_rules_are_enforced_before_any_query_runs
- **Residual risk.** The model still has to copy the dates into its arguments. Only scenarios that declare expected parameters check it.

### L-16: Base data failure crashes the request

- **Finding.** The portfolio rollup is loaded before the loop, outside the guard. A timeout there turns into a raw server error.
- **Fix.** Any failure while loading it degrades to a fixed low-confidence answer, and the model is never called.
- **Evidence.** scenario S22 (ok); test tests/test_agent.py::test_a_failing_base_dataset_degrades_the_answer_and_never_asks_the_model; test tests/test_agent.py::test_a_missing_section_in_the_base_dataset_also_degrades

### L-17: Malformed answer from the model

- **Finding.** The model wraps its JSON in code fences, writes a raw line break inside a string, adds prose around it, or does not produce an object.
- **Fix.** A tolerant parser extracts the object and coerces every field to the contract. Anything unusable becomes a low-confidence answer. The harness still reports a format violation.
- **Evidence.** scenario S23 (ok); scenario X08 (detected); test tests/test_response.py::test_code_fences_prose_and_raw_line_breaks_are_tolerated; test tests/test_response.py::test_text_that_is_not_an_object_degrades_to_low_confidence_without_raising

### L-18: Hallucinated or malformed tool arguments

- **Finding.** A model invents an argument, sends a number as text, or sends arguments that are not valid JSON.
- **Fix.** Every schema is closed. A small validator coerces harmless mistakes and reports all remaining problems at once with a hint. Unreadable arguments cost a step and come back as a validation error.
- **Evidence.** scenario S14 (ok); scenario S25 (ok); test tests/test_registry.py::test_open_schemas_are_a_startup_error; test tests/test_registry.py::test_invalid_arguments_come_back_as_a_validation_error_not_an_exception; test tests/test_agent.py::test_malformed_tool_arguments_cost_a_step_and_get_a_validation_error; test tests/test_jsonschema_lite.py::test_all_problems_are_reported_at_once

### L-19: Driver details reach the model

- **Finding.** A database driver message can contain host names or other internals. Passed to the model, it ends up in the answer.
- **Fix.** Driver failures are wrapped, the original stays in the exception chain for the operator, and the model receives only an error type and a generic hint.
- **Evidence.** scenario S12 (ok); test tests/test_guards.py::test_a_second_failure_is_reported_without_internal_detail; test tests/test_sources.py::test_driver_failures_become_data_source_errors_that_keep_the_cause
- **Residual risk.** The error type name of an unexpected exception is included for debugging.

## Partially closed (7)

### L-08: Provider rejects consecutive same-role messages

- **Finding.** A tool result followed by a nudge to answer produces two user-side messages in a row, which one provider format refuses.
- **Fix.** The loop works on a neutral message type. The wire adapter merges consecutive user-side items into one message and puts tool results first, so the error class cannot be built.
- **Gap.** The adapters are tested through an injected transport and have never been run against a live provider.
- **Evidence.** test tests/test_llm_wire.py::test_anthropic_merges_consecutive_user_items_so_roles_always_alternate; test tests/test_llm_wire.py::test_anthropic_never_emits_two_consecutive_messages_with_the_same_role; test tests/test_llm_wire.py::test_anthropic_puts_tool_results_before_text_in_a_merged_user_message

### L-11: Instruction hidden in tool data

- **Finding.** A field returned by a tool (a portfolio label, for example) contains text that tells the model to ignore its rules.
- **Fix.** The prompt states that tool output is data, never an instruction, and gives the trust order. A scenario carries such a label and checks that the answer does not follow it.
- **Gap.** The defence is the prompt only. The scripted model cannot be injected, so the scenario shows the harness would catch an echo, not that a real model resists.
- **Evidence.** scenario S16 (ok)

### L-13: Personal data reaching an external model

- **Finding.** A mapped column that happens to hold free text, or a join that pulls in a contact field, would send personal data to the model provider.
- **Fix.** The data contract has no personal identifier. An egress guard inspects every tool result for denied field names and for email, phone and long-digit shapes, and either redacts them or withholds the whole result.
- **Gap.** A person's name in free text cannot be recognised. Long digit runs such as internal ids can be redacted by mistake. The user's own message is not inspected.
- **Evidence.** scenario S13 (ok); test tests/test_privacy.py::test_fields_that_commonly_hold_personal_data_are_redacted_by_name; test tests/test_agent.py::test_personal_looking_values_are_redacted_before_the_model_sees_them; test tests/test_agent.py::test_the_block_policy_withholds_the_whole_result

### L-20: Confidence that does not match the answer

- **Finding.** An answer that is mostly not found, or based on a failed call, is marked high confidence.
- **Fix.** The prompt ties confidence to the core of the question. Scenarios can declare the confidence they expect, and a self-test shows the harness reports a mismatch.
- **Gap.** No general rule links confidence to tool errors. Only scenarios that declare an expectation are checked.
- **Evidence.** scenario S12 (ok); scenario S21 (ok); scenario X09 (detected)

### L-21: Provider call failures

- **Finding.** A model provider throttles, times out or returns an error.
- **Fix.** Throttling and server errors are retried with capped backoff. Other errors fail fast. Error messages carry the status and a short provider message, never the key, the headers or the request body. The service maps provider failures to a generic 502.
- **Gap.** Retry behaviour is tested against a fake transport. No live provider was called.
- **Evidence.** test tests/test_llm_wire.py::test_a_throttled_call_is_retried_with_backoff_then_succeeds; test tests/test_llm_wire.py::test_error_messages_never_contain_the_key_the_headers_or_the_request_body; test tests/test_service.py::test_a_model_failure_is_502_with_a_generic_message_and_no_provider_text

### L-22: A real number on the wrong label

- **Finding.** The numeric faithfulness check catches an invented number. It cannot catch a real number attached to the wrong name.
- **Fix.** Scenarios add forbidden wording and ground-truth values computed by hand. Self-tests show each layer fires.
- **Gap.** Totals a model adds up from three or more numbers are reported as untraceable.
- **Evidence.** scenario X01 (detected); scenario X05 (detected); scenario X07 (detected); test tests/test_evals_checks.py::test_an_invented_number_is_flagged_with_its_token; test tests/test_evals_checks.py::test_a_sum_of_two_seen_numbers_is_accepted_only_for_values_of_at_least_one_hundred

### L-23: Claiming data is missing after receiving it

- **Finding.** The model says the data is not available although a tool call returned rows.
- **Fix.** A harness layer compares the answer with the guard's own telemetry (error type and row count) and flags the contradiction.
- **Gap.** It matches phrases, not entities. A sentence that mixes a true and a false unavailability claim passes.
- **Evidence.** scenario X06 (detected); test tests/test_evals_checks.py::test_claiming_unavailability_after_a_call_returned_rows_is_flagged

## Open queue (2)

### L-24: Answer quality that no rule can check

- **Finding.** Tone, completeness and whether the answer addresses the question are not covered by any assertion layer.
- **Fix.** Not started. A judge layer would need a live model and a rubric, which this repository does not include.

### L-25: The model itself is not measured in CI

- **Finding.** Scripted scenarios prove the machinery and the guards. They say nothing about how a real model chooses tools or words.
- **Fix.** `llm-agent-evals eval --live` runs the same scenarios against the configured model and reports the same metrics. No scheduled live run exists, so no baseline exists either.
- **Evidence.** module src/llm_agent_evals/evals/runner.py

## Landmines flagged, not fixed

- **M-01.** The result cache and the circuit breaker live in one process. With several workers each keeps its own, so cached results and breaker state are not shared.
- **M-02.** The SQL Server dialect is checked as generated text only. It has never been run against a live server.
- **M-03.** The OpenAI-compatible and Anthropic adapters have never been called against a live provider. Their wire formats were written from the public descriptions.
- **M-04.** The egress guard inspects tool results. The user's own message goes to the model as typed.
- **M-05.** The window limit applies per call. A model can chain calls over consecutive windows, which costs steps but not correctness.
- **M-06.** The default risk cut-offs, grace period and window limit are invented. Every number the agent reports about risk level depends on calibrating them.

## Method

- Every closed entry cites at least one scenario or test. Verification fails when a reference does not resolve.
- Scripted scenarios exercise the real loop, guards and analytics with a model that follows a script. They measure the machinery, not a model.
- Self-test scenarios (X-series) script a deliberate mistake and declare the violation the harness must report, which shows each assertion layer can fail.
- Partial entries say what is not covered. Open entries are known and unfixed.

## Hard constraints

- Data access is read-only. Every statement passes a single-SELECT guard before it runs.
- The data contract has no personal identifiers.
- The API key is read only from the environment and never appears in a message, log or representation.
- Telemetry carries no tool arguments, no data and no text of the conversation.
- Continuous integration uses the scripted and rule-based mock models only.
