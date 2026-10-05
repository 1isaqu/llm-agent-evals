"""Agent funnel entries built from `agent_rollup` rows.

Funnel vocabulary (do not swap the words):

    attempt  -> answer (someone picks up) -> CPC (right party reached) -> agreement

`answer_count` is "someone answered". `cpc_count` is "the right party answered" and is a
*count*, never a percentage. `answer_rate_pct = answer_count / attempt_count` and
`cpc_rate_pct = cpc_count / answer_count`. The funnel is monotonic up to CPC
(attempts >= answers >= CPC). Agreements come from another table and can exceed the CPC
count of the same window, which is not an anomaly.

Ratios are recomputed *after* summing rows. When two product lines are combined,
averaging their ratios would be wrong; summing counters and dividing once is right.

Two metrics are kept apart on purpose:

* `conversion_pct = agreement_count / cpc_count`. Generated agreements per right-party
  contact. This is the official conversion.
* `paid_per_cpc_pct = slips_paid / cpc_count`. Slips paid within grace per right-party
  contact. It is a different metric and must never be called conversion.
"""

from __future__ import annotations

from typing import Any

from llm_agent_evals.analytics.common import as_float, as_int, ratio_pct

Row = dict[str, Any]

_INT_FIELDS = (
    "attempt_count",
    "answer_count",
    "cpc_count",
    "agreement_count",
    "agreement_count_by_contract",
    "slips_issued",
    "slips_paid",
    "exception_count",
)
_FLOAT_FIELDS = ("agreement_value", "first_installment_value", "exception_value")


def agent_key(agent_id: str, agent_name: str) -> str:
    """Case-insensitive identity of an agent across rows."""
    return (agent_id or agent_name or "").strip().lower()


def agent_entries(rows: list[Row], reference_date: str) -> list[dict[str, Any]]:
    """One entry per agent, sorted by agreement value (largest first)."""
    acc: dict[str, dict[str, Any]] = {}
    for row in rows:
        agent_id = str(row.get("agent_id") or "").strip()
        agent_name = str(row.get("agent_name") or "").strip()
        key = agent_key(agent_id, agent_name)
        if not key:
            continue
        bucket = acc.setdefault(
            key,
            {
                "agent_id": agent_id,
                "agent_name": agent_name or agent_id,
                **dict.fromkeys(_INT_FIELDS, 0),
                **dict.fromkeys(_FLOAT_FIELDS, 0.0),
            },
        )
        for name in _INT_FIELDS:
            bucket[name] += as_int(row.get(name))
        for name in _FLOAT_FIELDS:
            bucket[name] += as_float(row.get(name))
    entries = [_entry(bucket, reference_date) for bucket in acc.values()]
    entries.sort(key=lambda e: (-e["agreement_value"], e["agent_id"]))
    return entries


def _entry(b: dict[str, Any], reference_date: str) -> dict[str, Any]:
    agreements = b["agreement_count"]
    return {
        "agent_name": b["agent_name"],
        "agent_id": b["agent_id"],
        "attempt_count": b["attempt_count"],
        "answer_count": b["answer_count"],
        "cpc_count": b["cpc_count"],
        "answer_rate_pct": ratio_pct(b["answer_count"], b["attempt_count"]),
        "cpc_rate_pct": ratio_pct(b["cpc_count"], b["answer_count"]),
        "agreement_count": agreements,
        "agreement_count_by_contract": b["agreement_count_by_contract"],
        "agreement_value": round(b["agreement_value"], 2),
        "avg_ticket": round(b["agreement_value"] / agreements, 2) if agreements else 0.0,
        "first_installment_value": round(b["first_installment_value"], 2),
        "slips_issued": b["slips_issued"],
        "slips_paid": b["slips_paid"],
        "slip_effectiveness_pct": ratio_pct(b["slips_paid"], b["slips_issued"]),
        "conversion_pct": ratio_pct(agreements, b["cpc_count"]),
        "paid_per_cpc_pct": ratio_pct(b["slips_paid"], b["cpc_count"]),
        "exception_count": b["exception_count"],
        "exception_value": round(b["exception_value"], 2),
        "reference_date": reference_date,
    }


ORDER_BY_FIELDS = (
    "agreement_value",
    "agreement_count",
    "conversion_pct",
    "paid_per_cpc_pct",
    "cpc_count",
    "attempt_count",
    "first_installment_value",
    "avg_ticket",
    "exception_value",
    "exception_count",
    "cpc_rate_pct",
    "answer_rate_pct",
    "answer_count",
)

COMPARE_METRICS = (
    "attempt_count",
    "cpc_count",
    "answer_rate_pct",
    "agreement_count",
    "conversion_pct",
    "first_installment_value",
)
