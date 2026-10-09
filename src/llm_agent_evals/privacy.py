"""Egress guard: the last check before tool output can reach an external model.

The data contract contains no customer identifiers, so a correctly mapped data
source never produces any. (Agent names and logins are part of the contract and are
sent to the model on purpose.) The guard exists for the case where one appears anyway:
a user maps a column that happens to hold free text, a portfolio label carries an
address, a join pulls in a contact field. It inspects every tool result and either
redacts the offending values or blocks the whole result, depending on policy.

It is a safety net, not a classifier. It looks for field names that commonly hold
personal data and for value shapes: e-mail addresses, phone-like and long digit runs
in strings, and whole numbers of nine or more digits (a phone or tax number stored as
an integer column). Floats are left alone, since amounts are floats. It cannot
recognise a person's name written in free text, so the primary protection stays the
contract: do not expose such columns to the agent.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

DEFAULT_DENIED_FIELDS = frozenset(
    {
        "name",
        "full_name",
        "first_name",
        "last_name",
        "customer_name",
        "debtor_name",
        "email",
        "e_mail",
        "phone",
        "phone_number",
        "mobile",
        "document",
        "document_number",
        "tax_id",
        "national_id",
        "ssn",
        "birth_date",
        "address",
        "street",
        "postal_code",
    }
)

_EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_ISO_TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2})?)?")
_PHONE_LIKE = re.compile(r"(?<!\w)\+?\d(?:[\s().-]?\d){9,}(?!\w)")
_LONG_DIGITS = re.compile(r"(?<!\w)\d{9,}(?!\w)")

REDACTED = "[redacted]"


@dataclass(frozen=True)
class EgressFinding:
    """One thing the guard found. `path` locates it, `kind` says what it looked like."""

    path: str
    kind: str


@dataclass
class EgressResult:
    """Outcome of inspecting one payload."""

    payload: Any
    findings: list[EgressFinding] = field(default_factory=list)
    blocked: bool = False


class EgressGuard:
    """Inspect (and optionally rewrite) tool output before it is sent to a model."""

    def __init__(self, policy: str = "redact", denied_fields: frozenset[str] | None = None) -> None:
        if policy not in ("redact", "block"):
            raise ValueError("policy must be 'redact' or 'block'")
        self.policy = policy
        self.denied_fields = (
            DEFAULT_DENIED_FIELDS if denied_fields is None else frozenset(denied_fields)
        )

    def inspect(self, payload: Any) -> EgressResult:
        findings: list[EgressFinding] = []
        cleaned = self._walk(payload, "$", findings)
        if findings and self.policy == "block":
            return EgressResult(payload=None, findings=findings, blocked=True)
        return EgressResult(payload=cleaned, findings=findings)

    def _walk(self, node: Any, path: str, findings: list[EgressFinding]) -> Any:
        if isinstance(node, dict):
            return {
                key: self._walk_entry(key, value, path, findings) for key, value in node.items()
            }
        if isinstance(node, list):
            return [self._walk(item, f"{path}[{i}]", findings) for i, item in enumerate(node)]
        if isinstance(node, str):
            return self._scrub_string(node, path, findings)
        if isinstance(node, int) and not isinstance(node, bool):
            return self._scrub_integer(node, path, findings)
        return node

    def _walk_entry(self, key: Any, value: Any, path: str, findings: list[EgressFinding]) -> Any:
        child_path = f"{path}.{key}"
        if isinstance(key, str) and key.lower() in self.denied_fields and value not in (None, ""):
            findings.append(EgressFinding(child_path, "field"))
            return REDACTED
        return self._walk(value, child_path, findings)

    @staticmethod
    def _scrub_integer(value: int, path: str, findings: list[EgressFinding]) -> Any:
        if _LONG_DIGITS.fullmatch(str(abs(value))):
            findings.append(EgressFinding(path, "long_digits"))
            return REDACTED
        return value

    @staticmethod
    def _scrub_string(text: str, path: str, findings: list[EgressFinding]) -> str:
        result = text
        for kind, pattern in (
            ("email", _EMAIL),
            ("phone", _PHONE_LIKE),
            ("long_digits", _LONG_DIGITS),
        ):
            scannable = _ISO_TIMESTAMP.sub(lambda m: "x" * len(m.group()), result)
            if pattern.search(scannable):
                findings.append(EgressFinding(path, kind))
                result = _redact_matches(result, scannable, pattern)
        return result


def _redact_matches(original: str, scannable: str, pattern: re.Pattern[str]) -> str:
    """Replace the spans that matched in `scannable`, keeping `original` text elsewhere."""
    pieces: list[str] = []
    cursor = 0
    for match in pattern.finditer(scannable):
        pieces.append(original[cursor : match.start()])
        pieces.append(REDACTED)
        cursor = match.end()
    pieces.append(original[cursor:])
    return "".join(pieces)
