"""Session value types: a date window and the filters of one chat session."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

PRODUCT_LINES = ("auto", "consumer")
ALL_LINES = "all"


@dataclass(frozen=True)
class Window:
    """An inclusive date window `[start, end]`."""

    start: date
    end: date

    def __post_init__(self) -> None:
        if self.end < self.start:
            raise ValueError("window end is before window start")

    @classmethod
    def from_iso(cls, start: str, end: str) -> Window:
        return cls(date.fromisoformat(start), date.fromisoformat(end))

    @property
    def days(self) -> int:
        return (self.end - self.start).days + 1

    @property
    def end_exclusive(self) -> date:
        return self.end + timedelta(days=1)

    def contains(self, day: date) -> bool:
        return self.start <= day <= self.end

    def trailing(self, days: int) -> Window:
        """A window of `days` days ending at this window's end."""
        return Window(self.end - timedelta(days=days - 1), self.end)

    def __str__(self) -> str:
        return f"{self.start.isoformat()}..{self.end.isoformat()}"


@dataclass(frozen=True)
class Session:
    """What the user is looking at: a product line (or all of them) and a window."""

    product_line: str
    window: Window

    @property
    def date_from(self) -> str:
        return self.window.start.isoformat()

    @property
    def date_to(self) -> str:
        return self.window.end.isoformat()
