import pytest

from llm_agent_evals.errors import ERROR_RETRYABLE_DEFAULT, build_tool_error, sanitize_final_text


def test_tool_error_has_one_shape_everywhere():
    error = build_tool_error("validation", hint="fix it", user_facing="bad request", errors=["x"])
    assert error == {
        "ok": False,
        "error_type": "validation",
        "retryable": True,
        "hint": "fix it",
        "user_facing": "bad request",
        "errors": ["x"],
    }


def test_retryable_defaults_and_override():
    assert build_tool_error("not_found", hint="h", user_facing="u")["retryable"] is False
    assert build_tool_error("upstream_5xx", hint="h", user_facing="u")["retryable"] is True
    forced = build_tool_error("upstream_5xx", hint="h", user_facing="u", retryable=False)
    assert forced["retryable"] is False
    assert set(ERROR_RETRYABLE_DEFAULT) >= {"validation", "egress_blocked", "unknown_tool"}


def test_unknown_error_types_are_not_retryable():
    assert build_tool_error("brand_new", hint="h", user_facing="u")["retryable"] is False


@pytest.mark.parametrize(
    "text",
    [
        "Here you go:\n```sql\nSELECT * FROM agreements\n```",
        "```\nDROP TABLE agreements\n```",
        "Traceback (most recent call last):\n  File x",
        'Raw: {"meta": {"run_id": "r"}, "data": []}',
        'Raw: {"errors": [1]}',
    ],
)
def test_sanitizer_refuses_sql_stack_traces_and_raw_envelopes(text):
    assert sanitize_final_text(text) is None


@pytest.mark.parametrize(
    "text",
    [
        "Portfolio T-01 is at low risk.",
        "You can select a window with the date picker.",
        "```python\nprint('hello')\n```",
    ],
)
def test_sanitizer_lets_ordinary_text_through(text):
    assert sanitize_final_text(text) == text
