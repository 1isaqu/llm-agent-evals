"""The metric registry: official definitions, status sets and caveats.

`explain_metric` serves these entries to the model, and the system prompt tells the
model to look a term up before explaining any formula. The registry is built from the
active `BusinessConfig`, so the status sets and thresholds it states can never differ
from the ones the analytics code applies. A default snapshot is committed under
`docs/metric-registry.default.json` and a test fails when it drifts from this module.
Regenerate it with `llm-agent-evals metrics --write docs/metric-registry.default.json`,
never by hand.
"""

from __future__ import annotations

import re
from typing import Any

from llm_agent_evals.business import BusinessConfig

_ALIASES = {
    "cpc": "cpc",
    "right_party_contact": "cpc",
    "rpc": "cpc",
    "answer": "answer_rate_pct",
    "answer_rate": "answer_rate_pct",
    "contact_rate": "answer_rate_pct",
    "cpc_rate": "cpc_rate_pct",
    "conversion": "conversion_pct",
    "exceptions": "exception",
    "pending_exception": "exception",
    "pending": "exception",
    "composite_risk": "composite_risk_formula",
    "risk": "composite_risk_formula",
    "effectiveness": "slip_effectiveness_pct",
    "slip_effectiveness": "slip_effectiveness_pct",
    "ticket": "avg_ticket",
    "average_ticket": "avg_ticket",
    "pace": "daily_pace",
}

_KPIS: dict[str, dict[str, str]] = {
    "attempt_count": {
        "formula": "number of outreach events in the window",
        "source": "analytics/agents.py (effort query)",
    },
    "answer_count": {
        "formula": "outreach events whose result is flagged answered",
        "source": "analytics/agents.py (effort query)",
    },
    "cpc": {
        "formula": "cpc_count = outreach events flagged answered AND right_party. "
        "A count, never a percentage.",
        "source": "analytics/agents.py (effort query)",
    },
    "answer_rate_pct": {
        "formula": "answer_count / attempt_count * 100. Always label it 'answer rate', "
        "never 'CPC'.",
        "source": "analytics/agents.py (ratio_pct(answer_count, attempt_count))",
    },
    "cpc_rate_pct": {
        "formula": "cpc_count / answer_count * 100",
        "source": "analytics/agents.py (ratio_pct(cpc_count, answer_count))",
    },
    "conversion_pct": {
        "formula": "agreement_count / cpc_count * 100: generated agreements over "
        "right-party contacts. "
        "The denominator is CPC, not slips issued.",
        "source": "analytics/agents.py (ratio_pct(agreement_count, cpc_count))",
        "note": "Only exists at agent grain: there is no daily, portfolio or product-line series.",
    },
    "paid_per_cpc_pct": {
        "formula": "slips_paid / cpc_count * 100. A different metric: never call it conversion.",
        "source": "analytics/agents.py",
    },
    "slip_effectiveness_pct": {
        "formula": "slips paid within the grace period / slips issued * 100, by issue date",
        "source": "analytics/effectiveness.py",
        "note": "Not conversion. Recent days are immature because their grace period may "
        "still be running.",
    },
    "avg_ticket": {
        "formula": "agreement_value / agreement_count",
        "source": "analytics/agents.py",
    },
    "first_installment_value": {
        "formula": "SUM(amount) of installment 0 of generated agreements issued in the window",
        "source": "sources/sql.py (financial_by_agent, status_rollup)",
    },
    "agreement_value": {
        "formula": "SUM(amount) over all installments of generated agreements whose first "
        "installment was issued in the window",
        "source": "sources/sql.py (agreement_value_by_agent)",
    },
    "agreement_count": {
        "formula": "COUNT(DISTINCT agreement_id) with a generated first installment in the window "
        "(agreement grain)",
        "source": "sources/sql.py",
    },
    "agreement_count_by_contract": {
        "formula": "COUNT(DISTINCT agreement_id, portfolio_id, debt_id) through agreement_debts "
        "(contract grain)",
        "source": "sources/sql.py (contracts_by_agent)",
        "note": "Never use it as a denominator. Ticket and conversion use the agreement grain.",
    },
    "generated_value": {
        "formula": "first_installment_value over generated statuses, per day",
        "source": "analytics/series.py",
    },
    "daily_pace": {
        "formula": "Always TODAY according to the agent clock, whatever the session window is. "
        "Baseline estimator: mean hourly profile of recent history.",
        "source": "analytics/pace.py",
    },
}


def _status_section(config: BusinessConfig) -> dict[str, Any]:
    def entry(labels: tuple[str, ...]) -> dict[str, Any]:
        return {"labels": list(labels), "ids": list(config.ids(labels))}

    return {
        "approved": entry(config.approved),
        "exception": entry(config.exception),
        "rejected": entry(config.rejected),
        "broken": entry(config.broken),
        "auto_broken": entry(config.auto_broken),
        "generated": entry(config.generated_labels),
        "universe": entry((*config.generated_labels, *config.exception)),
    }


def _caveats(config: BusinessConfig) -> dict[str, str]:
    generated = ", ".join(config.generated_labels)
    return {
        "exception": (
            "The business word 'exception' means status PENDING_EXCEPTION: value that is "
            "pending approval and has not become a firm slip."
        ),
        "generated": (
            f"GENERATED = {generated}. It includes agreements that broke later: breaking is a "
            "later outcome and does not erase the generation. A broken slip stays in the "
            "issued denominator and never in the paid numerator."
        ),
        "first_installment": (
            f"installment_no = {config.first_installment_no} is the first installment. Never "
            "normalise it to 1."
        ),
        "portfolio": "The portfolio name comes from debt_attributes.portfolio_label, not from "
        "a portfolio table.",
        "agent_id": "agent_id is the agent's public key (agents.login). agent_name is the "
        "display name.",
        "cross_line": (
            "The same agent in two product lines is two separate entities per call. Rollups for "
            "'all' lines sum counters first and recompute ratios once."
        ),
        "grain": (
            "Two grains of 'agreement count': agreement_count (per agreement) is used for "
            "rankings, "
            "ticket and conversion; agreement_count_by_contract (per contract) is for totals only."
        ),
        "answer_vs_cpc": (
            "Three different things: CPC is a count; answer rate = answer_count / attempt_count; "
            "CPC rate = cpc_count / answer_count. Do not confuse a column name with a definition."
        ),
        "denominator": (
            "Every risk percentage uses the same denominator: the first-installment universe of "
            f"the window = generated ({generated}) + exception + rejected value. Each axis is a "
            "slice between 0 and 100% of that universe."
        ),
        "thresholds": (
            f"Risk level from composite risk: low <= {config.risk_low_max}%, medium <= "
            f"{config.risk_medium_max}%, high above. These cut-offs are configuration, not "
            "universal truths. Any axis outside 0..100% is a data anomaly and must be flagged."
        ),
        "anomaly": (
            "An axis outside 0..100% cannot happen with clean data: every axis is a slice of a "
            "universe that contains it. Report it as suspected corruption (negative amounts, "
            "mismatched windows), never as real risk and never corrected."
        ),
        "slip_maturity": (
            f"A slip counts as unpaid until it is paid. With a grace period of {config.grace_days} "
            "days, slips issued in the last days look worse than they will."
        ),
        "excluded_agents": (
            "Agents whose login matches an excluded pattern are removed in SQL, never afterwards: "
            f"{', '.join(config.excluded_agent_patterns)}."
        ),
    }


class MetricRegistry:
    """Status sets, KPI definitions and caveats for one `BusinessConfig`."""

    def __init__(self, config: BusinessConfig) -> None:
        self.config = config
        kpis = {name: dict(entry) for name, entry in _KPIS.items()}
        kpis["composite_risk_formula"] = {
            "formula": "MAX(exception_pct, broken_pct, rejected_pct). Never the sum: the axes "
            "overlap (broken is a subset of generated), so adding them double counts and "
            "overstates risk. A portfolio's risk is its worst axis.",
            "source": "analytics/portfolio.py",
        }
        self._data: dict[str, Any] = {
            "status": _status_section(config),
            "caveats": _caveats(config),
            "kpis": kpis,
        }

    def to_dict(self) -> dict[str, Any]:
        """A JSON-serialisable snapshot of the whole registry."""
        return {key: self._data[key] for key in ("status", "caveats", "kpis")}

    def available_terms(self) -> list[str]:
        data = self.to_dict()
        return sorted(set(data["status"]) | set(data["caveats"]) | set(data["kpis"]))

    def resolve(self, term: str) -> dict[str, Any] | None:
        """Look up a term (KPI, status group or caveat). None when unknown."""
        key = re.sub(r"[\s\-]+", "_", term.strip().lower())
        key = _ALIASES.get(key, key)
        data = self.to_dict()
        if key in data["status"]:
            entry: dict[str, Any] = {"type": "status", "values": data["status"][key]}
            if key in data["caveats"]:
                entry["caveat"] = data["caveats"][key]
            return entry
        if key in data["caveats"]:
            return {"type": "caveat", "text": data["caveats"][key]}
        if key in data["kpis"]:
            return {"type": "kpi", **data["kpis"][key]}
        return None
