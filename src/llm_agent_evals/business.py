"""Business configuration: status sets, risk cut-offs and operating parameters.

Everything here is an invented default. The status ids, the risk cut-offs, the
grace period and the window limits are placeholders that show where a real
operation plugs its own calibrated values in. Calibrate them on your own data
before trusting any number the agent reports.

The status sets are defined by label so a user maps their own catalogue once
(`status_ids`) and the derived sets stay consistent:

    generated = approved + broken + auto_broken
    universe  = generated + exception
    rollup    = universe + rejected
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from llm_agent_evals.errors import ConfigError

DEFAULT_STATUS_IDS: dict[str, int] = {
    "ACTIVE": 10,
    "BROKEN": 20,
    "PAID": 30,
    "PENDING_EXCEPTION": 40,
    "REJECTED": 50,
    "AUTO_BROKEN": 60,
    "PAID_MANUAL": 70,
}

_LABEL_SET_FIELDS = ("approved", "broken", "auto_broken", "exception", "rejected")


@dataclass(frozen=True)
class BusinessConfig:
    """Operating rules for the analytics layer. All defaults are invented."""

    status_ids: Mapping[str, int] = field(default_factory=lambda: dict(DEFAULT_STATUS_IDS))
    approved: tuple[str, ...] = ("ACTIVE", "PAID", "PAID_MANUAL")
    broken: tuple[str, ...] = ("BROKEN",)
    auto_broken: tuple[str, ...] = ("AUTO_BROKEN",)
    exception: tuple[str, ...] = ("PENDING_EXCEPTION",)
    rejected: tuple[str, ...] = ("REJECTED",)

    risk_low_max: float = 20.0
    risk_medium_max: float = 40.0

    grace_days: int = 3
    excluded_agent_patterns: tuple[str, ...] = ("system%", "test%")
    first_installment_no: int = 0

    max_window_days: int = 93
    plan_lookback_days: int = 150
    plan_late_remaining: int = 3

    crosstab_limit: int = 12
    stage_portfolio_limit: int = 8

    pace_first_hour: int = 9
    pace_last_hour: int = 18
    pace_min_history_days: int = 3

    def __post_init__(self) -> None:
        self._validate_status_sets()
        if not 0 <= self.risk_low_max < self.risk_medium_max:
            raise ConfigError("risk_low_max must be >= 0 and smaller than risk_medium_max")
        for name in ("grace_days", "max_window_days", "plan_lookback_days", "plan_late_remaining"):
            if getattr(self, name) < 0:
                raise ConfigError(f"{name} must not be negative")
        if not 0 <= self.pace_first_hour < self.pace_last_hour <= 23:
            raise ConfigError("pace hours must satisfy 0 <= first < last <= 23")

    def _validate_status_sets(self) -> None:
        if len(set(self.status_ids.values())) != len(self.status_ids):
            raise ConfigError("status_ids must map each label to a distinct id")
        for name in _LABEL_SET_FIELDS:
            for label in getattr(self, name):
                if label not in self.status_ids:
                    raise ConfigError(f"{name} refers to unknown status label {label!r}")

    def ids(self, labels: Iterable[str]) -> tuple[int, ...]:
        """Sorted status ids for the given labels."""
        return tuple(sorted({int(self.status_ids[label]) for label in labels}))

    @property
    def approved_ids(self) -> tuple[int, ...]:
        return self.ids(self.approved)

    @property
    def exception_ids(self) -> tuple[int, ...]:
        return self.ids(self.exception)

    @property
    def rejected_ids(self) -> tuple[int, ...]:
        return self.ids(self.rejected)

    @property
    def broken_ids(self) -> tuple[int, ...]:
        return self.ids(self.broken)

    @property
    def auto_broken_ids(self) -> tuple[int, ...]:
        return self.ids(self.auto_broken)

    @property
    def generated_labels(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys((*self.approved, *self.broken, *self.auto_broken)))

    @property
    def generated_ids(self) -> tuple[int, ...]:
        return self.ids(self.generated_labels)

    @property
    def universe_ids(self) -> tuple[int, ...]:
        return self.ids((*self.generated_labels, *self.exception))

    @property
    def rollup_ids(self) -> tuple[int, ...]:
        return self.ids((*self.generated_labels, *self.exception, *self.rejected))

    def label_of(self, status_id: int) -> str:
        for label, value in self.status_ids.items():
            if value == status_id:
                return label
        return f"STATUS_{status_id}"

    def risk_level(self, composite_risk_pct: float) -> str:
        """Map a composite risk percentage to low, medium or high."""
        if composite_risk_pct <= self.risk_low_max:
            return "low"
        if composite_risk_pct <= self.risk_medium_max:
            return "medium"
        return "high"

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> BusinessConfig:
        """Build a config from a plain mapping (for example a parsed JSON file)."""
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ConfigError(f"unknown business config keys: {', '.join(unknown)}")
        kwargs: dict[str, Any] = {}
        for key, value in data.items():
            if isinstance(value, list):
                kwargs[key] = tuple(value)
            else:
                kwargs[key] = value
        return cls(**kwargs)

    @classmethod
    def from_file(cls, path: str | Path) -> BusinessConfig:
        try:
            raw = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ConfigError(f"cannot read business config file {str(path)!r}: {exc}") from exc
        if not isinstance(raw, dict):
            raise ConfigError("business config file must contain a JSON object")
        return cls.from_mapping(raw)
