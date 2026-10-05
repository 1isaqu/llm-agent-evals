"""A data source backed by canned rollup rows, used by evaluation scenarios and tests.

Because the fixture sits at the rollup level, the real analytics code (risk
composition, funnel ratios, stage classification, re-bucketing) runs in every
scenario. Only the SQL layer is replaced.

A section named after a `DataSource` method holds either

* a plain list of rows, returned for every call, or
* a list of blocks `{"match": {...}, "rows": [...]}`. The first block whose `match`
  entries all equal the call parameters supplies the rows. Parameters are the
  method's own arguments plus `start` and `end` (ISO dates) for windowed calls.
  A call that matches no block gets no rows.

Rows that carry a `day` key are additionally filtered to the requested window.
A section can be the single entry `{"$fail": "message"}` to simulate a failing
upstream: the call raises `DataSourceError`. A method without a section raises
`DataUnavailable`, so a scenario states exactly which data it supplies.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date
from typing import Any

from llm_agent_evals.errors import DataSourceError, DataUnavailable
from llm_agent_evals.session import Window
from llm_agent_evals.sources.base import SOURCE_METHODS, Row


def _same(expected: Any, actual: Any) -> bool:
    if expected is None or actual is None:
        return expected is None and actual is None
    return str(expected).lower() == str(actual).lower()


def _is_block_list(section: Any) -> bool:
    return (
        isinstance(section, list)
        and bool(section)
        and all(
            isinstance(item, dict) and "rows" in item and set(item) <= {"match", "rows"}
            for item in section
        )
    )


class FixtureSource:
    """Canned rollup rows. See the module docstring for the section format."""

    def __init__(self, sections: Mapping[str, Any]) -> None:
        unknown = sorted(set(sections) - set(SOURCE_METHODS))
        if unknown:
            raise ValueError(f"unknown fixture sections: {', '.join(unknown)}")
        self._sections = dict(sections)

    def status_rollup(self, window: Window) -> list[Row]:
        return self._rows("status_rollup", window)

    def daily_rollup(self, window: Window, portfolio: str | None = None) -> list[Row]:
        return self._rows("daily_rollup", window, portfolio=portfolio)

    def agent_rollup(self, window: Window) -> list[Row]:
        return self._rows("agent_rollup", window)

    def agent_status_rollup(self, window: Window) -> list[Row]:
        return self._rows("agent_status_rollup", window)

    def crosstab_rollup(self, window: Window, by: str, value: str) -> list[Row]:
        return self._rows("crosstab_rollup", window, by=by, value=value)

    def plan_stage_rollup(self, end: date) -> list[Row]:
        return self._rows("plan_stage_rollup", None, end=end.isoformat())

    def slip_rollup(
        self, window: Window, group_by: str, date_field: str, portfolio: str | None = None
    ) -> list[Row]:
        return self._rows(
            "slip_rollup", window, group_by=group_by, date_field=date_field, portfolio=portfolio
        )

    def drilldown_rows(self, kind: str, portfolio: str, window: Window) -> list[Row]:
        return self._rows("drilldown_rows", window, kind=kind, portfolio=portfolio)

    def hourly_counts(self, window: Window) -> list[Row]:
        return self._rows("hourly_counts", window)

    def _rows(self, name: str, window: Window | None, **params: Any) -> list[Row]:
        if name not in self._sections:
            raise DataUnavailable(f"fixture provides no {name!r} section")
        section = self._sections[name]
        if isinstance(section, dict) and "$fail" in section:
            raise DataSourceError(str(section["$fail"]))
        call_params = dict(params)
        if window is not None:
            call_params.update(start=window.start.isoformat(), end=window.end.isoformat())
        rows = self._pick_rows(section, call_params)
        if window is not None:
            rows = [row for row in rows if _day_in_window(row, window)]
        return [dict(row) for row in rows]

    @staticmethod
    def _pick_rows(section: Any, call_params: Mapping[str, Any]) -> list[Row]:
        if not _is_block_list(section):
            return list(section)
        for block in section:
            match = block.get("match", {})
            if all(_same(expected, call_params.get(key)) for key, expected in match.items()):
                return list(block["rows"])
        return []


def _day_in_window(row: Row, window: Window) -> bool:
    if "day" not in row:
        return True
    return window.contains(date.fromisoformat(str(row["day"])[:10]))
