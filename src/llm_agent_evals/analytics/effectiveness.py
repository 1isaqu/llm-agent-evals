"""Slip effectiveness: slips paid within the grace period over slips issued.

Effectiveness is *not* conversion. Conversion divides generated agreements by right-party
contacts. Effectiveness divides slips (first-installment payment documents) paid within
the grace period by slips issued. The two answer different questions, use different
tools and must never share a label. The point shape here uses `slips_issued`,
`paid_within_grace` and `effectiveness_pct` so a result cannot be mistaken for the other.

Maturity caveat: a slip counts as unpaid until it is paid, so recent days look worse
than they will once their grace periods end.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from llm_agent_evals.analytics.common import as_int, ratio_pct
from llm_agent_evals.analytics.series import bucket_key

Row = dict[str, Any]

CRITERION = (
    "Slip effectiveness: first-installment slips paid within the grace period divided by "
    "slips issued, by issue date. It is NOT conversion (agreements over right-party contacts). "
    "Recent days are immature: their grace period may not have ended yet."
)


def effectiveness_points(rows: list[Row], granularity: str) -> list[dict[str, Any]]:
    """Daily, weekly or monthly effectiveness points, oldest first."""
    buckets: dict[str, list[int]] = {}
    for row in rows:
        if row.get("group_key") is None:
            continue
        day = date.fromisoformat(str(row["group_key"])[:10])
        counts = buckets.setdefault(bucket_key(day, granularity), [0, 0])
        counts[0] += as_int(row.get("generated"))
        counts[1] += as_int(row.get("paid_on_time"))
    return [
        {
            "date": key,
            "slips_issued": counts[0],
            "paid_within_grace": counts[1],
            "effectiveness_pct": ratio_pct(counts[1], counts[0]),
        }
        for key, counts in sorted(buckets.items())
    ]


def effectiveness_total(points: list[dict[str, Any]]) -> dict[str, Any]:
    issued = sum(p["slips_issued"] for p in points)
    paid = sum(p["paid_within_grace"] for p in points)
    return {
        "slips_issued": issued,
        "paid_within_grace": paid,
        "effectiveness_pct": ratio_pct(paid, issued),
    }
