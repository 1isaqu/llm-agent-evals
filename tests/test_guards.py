import json

import pytest

from llm_agent_evals.errors import DataSourceError, DataUnavailable, build_tool_error
from llm_agent_evals.guards import (
    CircuitBreaker,
    RunGuard,
    RunState,
    SharedState,
    ToolCache,
    cache_key,
    cap_tool_result,
    estimate_tokens,
    infer_row_count,
)


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def make_state(clock=None, sleeps=None, shared=None, **guard):
    clock = clock or Clock()
    sleeps = sleeps if sleeps is not None else []
    guard.setdefault("retry_base_delay_seconds", 0.5)
    shared = shared or SharedState(
        cache=ToolCache(clock=clock), breaker=CircuitBreaker(clock=clock)
    )
    return RunState(
        RunGuard(**guard),
        shared=shared,
        clock=clock,
        sleep=sleeps.append,
        jitter=lambda low, high: 0.0,
    )


def ok(value=1):
    return lambda: {"data": [value]}


def test_step_budget_is_enforced_inside_dispatch_for_a_batch_of_calls():
    state = make_state(max_steps=2)
    results = [state.dispatch(f"tool_{i}", {}, ok()) for i in range(4)]
    assert [r.get("error_type") for r in results] == [
        None,
        None,
        "step_budget_exceeded",
        "step_budget_exceeded",
    ]
    assert state.steps == 2 and state.force_final()


def test_wall_clock_budget_is_enforced_inside_dispatch():
    clock = Clock()
    state = make_state(clock=clock, wall_clock_seconds=10.0)
    assert "error_type" not in state.dispatch("a_tool", {}, ok())
    clock.advance(11.0)
    refused = state.dispatch("b_tool", {}, ok())
    assert refused["error_type"] == "wall_clock_exceeded"
    assert state.wall_clock_exceeded() and state.force_final()


def test_identical_calls_are_memoised_and_the_third_declares_a_loop():
    state = make_state()
    runs = []

    def run():
        runs.append(1)
        return {"data": [1]}

    first = state.dispatch("a_tool", {"x": 1}, run)
    second = state.dispatch("a_tool", {"x": 1}, run)
    assert second["memoised_result"] == first and len(runs) == 1
    assert not state.loop_declared
    state.dispatch("a_tool", {"x": 1}, run)
    assert state.loop_declared and state.force_final()


def test_argument_order_does_not_defeat_memoisation():
    state = make_state()
    state.dispatch("a_tool", {"x": 1, "y": 2}, ok())
    again = state.dispatch("a_tool", {"y": 2, "x": 1}, ok())
    assert "memoised_result" in again


def test_an_a_b_a_b_spiral_declares_a_loop():
    state = make_state()
    for name, arg in (("a_tool", 1), ("b_tool", 1), ("a_tool", 2), ("b_tool", 2)):
        state.dispatch(name, {"n": arg}, ok())
    assert state.loop_declared


def test_a_b_c_d_is_not_a_spiral():
    state = make_state()
    for name in ("a_tool", "b_tool", "c_tool", "d_tool"):
        state.dispatch(name, {}, ok())
    assert not state.loop_declared


def test_three_consecutive_failures_disable_the_tool_for_the_request():
    state = make_state(max_steps=20)

    def bug():
        raise KeyError("x")

    for n in range(3):
        assert state.dispatch("bad_tool", {"n": n}, bug)["error_type"] == "internal_error"
    assert state.dispatch("bad_tool", {"n": 99}, ok())["error_type"] == "tool_disabled"


def test_a_transient_failure_is_retried_once_with_backoff_and_then_succeeds():
    sleeps = []
    state = make_state(sleeps=sleeps)
    attempts = []

    def flaky():
        attempts.append(1)
        if len(attempts) == 1:
            raise DataSourceError("dropped connection")
        return {"data": [1, 2]}

    result = state.dispatch("a_tool", {}, flaky)
    assert result == {"data": [1, 2]} and len(attempts) == 2
    assert sleeps == [0.5]
    assert state.last_call_meta["error_type"] is None


def test_a_second_failure_is_reported_without_internal_detail():
    state = make_state()

    def down():
        raise DataSourceError("host db.example.invalid refused the connection")

    result = state.dispatch("a_tool", {}, down)
    assert result["error_type"] == "upstream_5xx"
    assert "db.example.invalid" not in json.dumps(result)


def test_a_bug_is_internal_error_and_is_not_retried():
    sleeps = []
    state = make_state(sleeps=sleeps)

    def bug():
        raise ZeroDivisionError("x")

    result = state.dispatch("a_tool", {}, bug)
    assert result["error_type"] == "internal_error" and result["retryable"] is False
    assert sleeps == []
    assert "ZeroDivisionError" in result["error_detail"]


def test_data_unavailable_is_reported_as_such_without_retry():
    sleeps = []
    state = make_state(sleeps=sleeps)

    def missing():
        raise DataUnavailable("no hourly data configured")

    result = state.dispatch("a_tool", {}, missing)
    assert result["error_type"] == "data_unavailable" and sleeps == []
    assert "no hourly data" not in json.dumps(result)


def test_an_oversized_result_becomes_a_warning_envelope_not_cut_json():
    big = {"data": list(range(5000))}
    capped, truncated = cap_tool_result(big, cap_tokens=100)
    assert truncated and capped["_truncated"] is True
    assert "narrower window" in capped["_warning"]
    assert len(capped["_preview"]) <= 100 * 3
    small, flag = cap_tool_result({"data": [1]}, cap_tokens=100)
    assert small == {"data": [1]} and not flag


def test_dispatch_applies_the_cap_and_reports_it():
    state = make_state(tool_result_token_cap=50)
    result = state.dispatch("a_tool", {}, lambda: {"data": list(range(1000))})
    assert result["_truncated"] and state.last_call_meta["truncated"]


def test_the_circuit_breaker_opens_on_a_high_error_rate_and_closes_later():
    clock = Clock()
    breaker = CircuitBreaker(
        window=10, min_sample=5, error_rate_threshold=0.3, open_seconds=60, clock=clock
    )
    for ok_flag in (True, True, False, False, False):
        breaker.record("a_tool", ok_flag)
    assert breaker.is_open("a_tool")
    assert not breaker.is_open("other_tool")
    clock.advance(61)
    assert not breaker.is_open("a_tool")


def test_an_open_breaker_short_circuits_the_call():
    clock = Clock()
    shared = SharedState(
        cache=ToolCache(clock=clock), breaker=CircuitBreaker(min_sample=1, clock=clock)
    )
    shared.breaker.record("a_tool", False)
    state = make_state(clock=clock, shared=shared)
    ran = []
    result = state.dispatch("a_tool", {}, lambda: ran.append(1))
    assert result["error_type"] == "upstream_5xx" and ran == []


def test_a_second_request_is_served_from_the_shared_cache():
    clock = Clock()
    shared = SharedState(cache=ToolCache(60, 8, clock), breaker=CircuitBreaker(clock=clock))
    runs = []

    def run():
        runs.append(1)
        return {"data": [1]}

    first = make_state(clock=clock, shared=shared)
    first.dispatch(
        "a_tool", {}, run, product_line="auto", date_from="2026-03-01", date_to="2026-03-18"
    )
    second = make_state(clock=clock, shared=shared)
    second.dispatch(
        "a_tool", {}, run, product_line="auto", date_from="2026-03-01", date_to="2026-03-18"
    )
    assert len(runs) == 1 and second.last_call_meta["cache_hit"]


def test_the_cache_key_includes_the_session_window():
    same_args = {}
    one = cache_key("a_tool", "auto", "2026-03-01", "2026-03-18", same_args)
    other_window = cache_key("a_tool", "auto", "2026-02-01", "2026-02-28", same_args)
    other_line = cache_key("a_tool", "consumer", "2026-03-01", "2026-03-18", same_args)
    assert len({one, other_window, other_line}) == 3


def test_different_windows_do_not_share_cached_results():
    clock = Clock()
    shared = SharedState(cache=ToolCache(clock=clock), breaker=CircuitBreaker(clock=clock))
    values = iter([1, 2])
    results = []
    for window in (("2026-03-01", "2026-03-18"), ("2026-02-01", "2026-02-28")):
        state = make_state(clock=clock, shared=shared)
        results.append(
            state.dispatch(
                "a_tool",
                {},
                lambda: {"data": [next(values)]},
                date_from=window[0],
                date_to=window[1],
            )
        )
    assert results[0] != results[1]


def test_cached_errors_expire_sooner_than_cached_successes():
    clock = Clock()
    cache = ToolCache(ttl_success_seconds=60, ttl_error_seconds=8, clock=clock)
    cache.set("ok", {"v": 1})
    cache.set("bad", {"v": 2}, error=True)
    clock.advance(9)
    assert cache.get("ok") == {"v": 1} and cache.get("bad") is None
    clock.advance(60)
    assert cache.get("ok") is None


def test_a_zero_ttl_disables_caching():
    cache = ToolCache(ttl_success_seconds=0)
    cache.set("k", {"v": 1})
    assert cache.get("k") is None


def test_a_returned_error_payload_is_recorded_as_an_error_and_cached_briefly():
    clock = Clock()
    shared = SharedState(cache=ToolCache(60, 8, clock), breaker=CircuitBreaker(clock=clock))
    state = make_state(clock=clock, shared=shared)
    payload = build_tool_error("not_found", hint="h", user_facing="u")
    state.dispatch("a_tool", {}, lambda: payload)
    assert state.last_call_meta["error_type"] == "not_found"
    clock.advance(9)
    again = make_state(clock=clock, shared=shared)
    ran = []
    again.dispatch("a_tool", {}, lambda: ran.append(1) or payload)
    assert ran == [1]


def test_reject_costs_a_step_and_is_never_cached_or_memoised():
    state = make_state(max_steps=3)
    error = build_tool_error("validation", hint="h", user_facing="u")
    assert state.reject("a_tool", error) is error
    assert state.steps == 1 and state.last_call_meta["error_type"] == "validation"
    result = state.dispatch("a_tool", {}, ok(7))
    assert result == {"data": [7]}


def test_reject_respects_the_step_budget():
    state = make_state(max_steps=1)
    state.dispatch("a_tool", {}, ok())
    refused = state.reject("a_tool", build_tool_error("validation", hint="h", user_facing="u"))
    assert refused["error_type"] == "step_budget_exceeded"


def test_budget_near_limit_signals_compaction_at_eighty_percent():
    state = make_state(tool_budget_tokens=100, tool_result_token_cap=1000)
    state.dispatch("a_tool", {}, lambda: {"text": "x" * 100})
    assert not state.budget_near_limit()
    state.dispatch("b_tool", {}, lambda: {"text": "y" * 200})
    assert estimate_tokens(json.dumps({"text": "y" * 200})) > 60
    assert state.budget_near_limit()


@pytest.mark.parametrize(
    ("result", "count"),
    [
        ([1, 2, 3], 3),
        ({"data": [1, 2]}, 2),
        ({"entries": []}, 0),
        ({"statuses": [1]}, 1),
        ({"meta": {"total_rows": 40}, "data": [1]}, 40),
        ({"agent_id": "a"}, 1),
        ({"ok": False, "error_type": "x"}, None),
        ("text", None),
    ],
)
def test_row_count_inference(result, count):
    assert infer_row_count(result) == count
