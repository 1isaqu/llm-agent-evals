from datetime import date, datetime

import pytest

from llm_agent_evals.business import DEFAULT_STATUS_IDS, BusinessConfig
from llm_agent_evals.errors import ConfigError
from llm_agent_evals.prompt import (
    build_system_prompt,
    date_anchors,
    load_template,
    render_system_prompt,
    session_context,
    tool_summary,
)
from llm_agent_evals.session import Session, Window
from llm_agent_evals.tools import build_default_registry
from tests.helpers import NOW, SESSION

REGISTRY = build_default_registry()
LINES = ("auto", "consumer")


def render(config=None):
    return render_system_prompt(config or BusinessConfig(), REGISTRY, LINES, "2026-03-18")


def test_the_template_renders_with_no_placeholder_left():
    assert "{{" not in render()


def test_a_leftover_placeholder_is_a_startup_error(monkeypatch):
    monkeypatch.setattr(
        "llm_agent_evals.prompt.load_template", lambda: load_template() + "\n{{MYSTERY}}"
    )
    with pytest.raises(ConfigError, match="MYSTERY"):
        render()


def test_the_prompt_quotes_the_configured_status_sets_and_thresholds():
    config = BusinessConfig(
        status_ids={**DEFAULT_STATUS_IDS, "ACTIVE": 101},
        risk_low_max=12.0,
        risk_medium_max=33.0,
        grace_days=7,
        max_window_days=45,
    )
    text = render(config)
    assert "ACTIVE (id 101)" in text
    assert "12%" in text and "33%" in text
    assert "7 days" in text and "45 days" in text
    assert "ACTIVE (id 10)" not in text


def test_the_prompt_names_the_generated_set_exactly():
    config = BusinessConfig()
    text = render(config)
    for label in config.generated_labels:
        assert label in text.split('**"Generated" means exactly these statuses:**')[1][:300]
    assert "PENDING_EXCEPTION (id 40)" in text


def test_the_tool_summary_lists_every_registered_tool_once():
    summary = tool_summary(REGISTRY)
    for name in REGISTRY.names():
        assert summary.count(f"`{name}`") == 1
    assert summary.count("\n") == len(REGISTRY) - 1


def test_the_response_format_is_documented_in_the_prompt():
    text = render()
    for field in ("highlights", "suggested_actions", "data_sources", "confidence"):
        assert f'"{field}"' in text


@pytest.mark.parametrize(
    ("today", "expected"),
    [
        (
            date(2026, 3, 18),
            {
                "today": ("2026-03-18", "2026-03-18"),
                "yesterday": ("2026-03-17", "2026-03-17"),
                "this week": ("2026-03-16", "2026-03-18"),
                "this month": ("2026-03-01", "2026-03-18"),
                "last month": ("2026-02-01", "2026-02-28"),
                "this quarter": ("2026-01-01", "2026-03-18"),
                "last quarter": ("2025-10-01", "2025-12-31"),
            },
        ),
        (
            date(2026, 1, 1),
            {
                "today": ("2026-01-01", "2026-01-01"),
                "yesterday": ("2025-12-31", "2025-12-31"),
                "this week": ("2025-12-29", "2026-01-01"),
                "this month": ("2026-01-01", "2026-01-01"),
                "last month": ("2025-12-01", "2025-12-31"),
                "this quarter": ("2026-01-01", "2026-01-01"),
                "last quarter": ("2025-10-01", "2025-12-31"),
            },
        ),
        (
            date(2024, 3, 1),
            {
                "today": ("2024-03-01", "2024-03-01"),
                "yesterday": ("2024-02-29", "2024-02-29"),
                "this week": ("2024-02-26", "2024-03-01"),
                "this month": ("2024-03-01", "2024-03-01"),
                "last month": ("2024-02-01", "2024-02-29"),
                "this quarter": ("2024-01-01", "2024-03-01"),
                "last quarter": ("2023-10-01", "2023-12-31"),
            },
        ),
    ],
)
def test_date_anchors_are_computed_from_the_given_day(today, expected):
    anchors = {k: (s.isoformat(), e.isoformat()) for k, (s, e) in date_anchors(today).items()}
    assert anchors == expected


def test_open_periods_end_today_not_at_the_end_of_the_period():
    for name in ("this week", "this month", "this quarter"):
        assert date_anchors(date(2026, 5, 6))[name][1] == date(2026, 5, 6)


def test_the_session_context_lists_the_scope_the_window_and_every_anchor():
    text = session_context(SESSION, NOW, portfolio_count=3, lines=LINES)
    assert "product line auto" in text and "2026-03-01 to 2026-03-18" in text
    assert "Portfolios loaded in that window: 3" in text and "2026-03-18 15:30" in text
    for name in ("today", "yesterday", "this week", "last quarter"):
        assert f"- {name}:" in text


def test_a_session_over_all_lines_says_so():
    text = session_context(
        Session("all", Window.from_iso("2026-03-01", "2026-03-18")), NOW, 0, LINES
    )
    assert "all product lines (auto, consumer)" in text


def test_the_full_prompt_is_template_plus_session_context():
    full = build_system_prompt(BusinessConfig(), REGISTRY, LINES, SESSION, NOW, 2)
    assert full.startswith(render()[:200]) and "## Session context" in full
    assert datetime.fromisoformat("2026-03-18T15:30").strftime("%Y-%m-%d %H:%M") in full
