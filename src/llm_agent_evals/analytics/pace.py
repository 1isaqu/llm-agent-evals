"""Daily pace: how today's agreement generation compares with a normal day.

`PaceProvider` is the extension point. A real deployment can plug in a trained
forecasting model behind it; this package ships a transparent baseline instead of any
fitted artifact. The baseline treats the average hourly profile of recent history as
the expectation:

    expected(hour)      mean agreements generated in that hour on past days
    expected_to_date    sum of expected(hour) over the hours already finished today
    on_pace_pct         actual_to_date / expected_to_date
    projected_close     actual_to_date / (expected_to_date / expected_total)

The projection is withheld while the elapsed share of the expected day is below
`min_share`, because dividing by a tiny share amplifies noise.

"Today" and the current hour come from the clock given to the agent, so results are
reproducible in tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Protocol

from llm_agent_evals.analytics.common import as_float, as_int, ratio_pct
from llm_agent_evals.business import BusinessConfig
from llm_agent_evals.errors import build_tool_error
from llm_agent_evals.session import Window
from llm_agent_evals.sources.base import DataSource


class PaceProvider(Protocol):
    def forecast(self, source: DataSource, now: datetime, config: BusinessConfig) -> dict[str, Any]:
        """Return today's pace as a JSON-serialisable dict (or a tool error payload)."""
        ...


@dataclass(frozen=True)
class BaselinePace:
    """Average-hourly-profile baseline."""

    lookback_days: int = 28
    min_share: float = 0.1

    def forecast(self, source: DataSource, now: datetime, config: BusinessConfig) -> dict[str, Any]:
        today = now.date()
        rows = source.hourly_counts(Window(today - timedelta(days=self.lookback_days), today))
        hours = range(config.pace_first_hour, config.pace_last_hour + 1)
        history, actual = _split_rows(rows, today, set(hours))
        if len(history) < config.pace_min_history_days:
            return build_tool_error(
                "empty_result",
                hint="Not enough history to estimate a normal day. Report that the pace "
                "cannot be estimated yet.",
                user_facing="There is not enough history to estimate today's pace.",
                history_days=len(history),
            )
        expected = {
            h: sum(day.get(h, 0.0) for day in history.values()) / len(history) for h in hours
        }
        return self._summarise(now, hours, expected, actual, len(history))

    def _summarise(
        self,
        now: datetime,
        hours: range,
        expected: dict[int, float],
        actual: dict[int, float],
        history_days: int,
    ) -> dict[str, Any]:
        finished = [h for h in hours if h < now.hour]
        expected_total = sum(expected.values())
        expected_to_date = sum(expected[h] for h in finished)
        actual_to_date = sum(actual.get(h, 0.0) for h in finished)
        share = expected_to_date / expected_total if expected_total > 0 else 0.0
        projected = round(actual_to_date / share, 2) if share >= self.min_share else None
        return {
            "date": now.date().isoformat(),
            "as_of_hour": now.hour,
            "history_days": history_days,
            "expected_total": round(expected_total, 2),
            "expected_to_date": round(expected_to_date, 2),
            "actual_to_date": round(actual_to_date, 2),
            "on_pace_pct": ratio_pct(actual_to_date, expected_to_date),
            "projected_close": projected,
            "bands": [
                {
                    "hour": h,
                    "expected": round(expected[h], 2),
                    "actual": round(actual.get(h, 0.0), 2) if h <= now.hour else None,
                    "in_progress": h == now.hour,
                }
                for h in hours
            ],
            "method": (
                "baseline: mean hourly profile of recent history; projection = actual to date "
                "divided by the expected share of the day already elapsed"
            ),
        }


def _split_rows(
    rows: list[dict[str, Any]], today: date, hours: set[int]
) -> tuple[dict[str, dict[int, float]], dict[int, float]]:
    """Split hourly rows into history (by day) and today's actuals (by hour)."""
    history: dict[str, dict[int, float]] = {}
    actual: dict[int, float] = {}
    for row in rows:
        hour = as_int(row.get("hour"))
        if hour not in hours or row.get("day") is None:
            continue
        day, qty = str(row["day"])[:10], as_float(row.get("qty"))
        if day == today.isoformat():
            actual[hour] = actual.get(hour, 0.0) + qty
        else:
            history.setdefault(day, {})
            history[day][hour] = history[day].get(hour, 0.0) + qty
    return history, actual
