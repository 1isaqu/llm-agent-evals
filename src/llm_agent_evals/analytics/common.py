"""Shared building blocks: status sets, the status bucket and ratio helpers.

The single-denominator rule lives here, once. Every percentage the agent reports about
risk is a slice of the *first-installment universe*: generated value plus exception
value plus rejected value. Because every slice is a fraction of the same total, the
slices are comparable between portfolios, and a slice above 100% is impossible unless
the underlying rows are corrupt (negative amounts, mismatched windows).

Broken value is a subset of generated value, so the three risk slices overlap and are
not additive. That is why composite risk is the *maximum* slice and never the sum.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from llm_agent_evals.business import BusinessConfig


@dataclass(frozen=True)
class StatusSets:
    """Status ids grouped by role, resolved once from a `BusinessConfig`."""

    generated: frozenset[int]
    exception: frozenset[int]
    broken: frozenset[int]
    rejected: frozenset[int]

    @classmethod
    def from_config(cls, config: BusinessConfig) -> StatusSets:
        return cls(
            generated=frozenset(config.generated_ids),
            exception=frozenset(config.exception_ids),
            broken=frozenset(config.broken_ids),
            rejected=frozenset(config.rejected_ids),
        )


@dataclass
class StatusBucket:
    """Running totals for one group (a portfolio, a day, an agent)."""

    generated_qty: float = 0.0
    generated_value: float = 0.0
    exception_value: float = 0.0
    broken_value: float = 0.0
    rejected_value: float = 0.0

    def add(self, sets: StatusSets, status_id: int, qty: float, amount: float) -> None:
        if status_id in sets.generated:
            self.generated_qty += qty
            self.generated_value += amount
        if status_id in sets.exception:
            self.exception_value += amount
        if status_id in sets.broken:
            self.broken_value += amount
        if status_id in sets.rejected:
            self.rejected_value += amount

    @property
    def universe_value(self) -> float:
        return self.generated_value + self.exception_value + self.rejected_value

    def share_pct(self, value: float) -> float:
        """`value` as a percentage of the universe, 0 when the universe is empty."""
        universe = self.universe_value
        return round(value * 100.0 / universe, 2) if universe > 0 else 0.0

    def risk_axes(self) -> dict[str, float]:
        return {
            "exception_pct": self.share_pct(self.exception_value),
            "broken_pct": self.share_pct(self.broken_value),
            "rejected_pct": self.share_pct(self.rejected_value),
        }

    def composite_risk_pct(self) -> float:
        return max(self.risk_axes().values())


def ratio_pct(numerator: float, denominator: float) -> float:
    """`numerator / denominator` as a percentage rounded to 2 places; 0 when undefined."""
    if denominator <= 0:
        return 0.0
    return round(numerator * 100.0 / denominator, 2)


def as_int(value: Any) -> int:
    return int(float(value or 0))


def as_float(value: Any) -> float:
    return float(value or 0)
