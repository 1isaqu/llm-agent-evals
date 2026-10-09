import pytest

from llm_agent_evals.privacy import REDACTED, EgressGuard


def test_clean_payloads_pass_through_unchanged():
    payload = {"portfolio_name": "Portfolio T-01", "value": 1234.5, "date": "2026-03-18 15:30:00"}
    result = EgressGuard().inspect(payload)
    assert result.payload == payload and not result.findings and not result.blocked


def test_fields_that_commonly_hold_personal_data_are_redacted_by_name():
    result = EgressGuard().inspect({"email": "someone@example.org", "agent_id": "agent_01"})
    assert result.payload == {"email": REDACTED, "agent_id": "agent_01"}
    assert result.findings[0].kind == "field" and result.findings[0].path == "$.email"


def test_an_empty_denied_field_is_not_a_finding():
    result = EgressGuard().inspect({"phone": "", "name": None})
    assert not result.findings


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        ("write to someone@example.org today", "email"),
        ("call +1 555 010 9999 now", "phone"),
        ("document 123456789 on file", "long_digits"),
    ],
)
def test_string_shapes_are_scrubbed_in_free_text(text, kind):
    result = EgressGuard().inspect({"note": text})
    assert REDACTED in result.payload["note"]
    assert kind in {f.kind for f in result.findings}


def test_a_long_digit_run_is_scrubbed_whichever_pattern_sees_it_first():
    result = EgressGuard().inspect({"note": "document 12345678901 on file"})
    assert result.payload["note"] == f"document {REDACTED} on file"
    assert result.findings


def test_a_long_integer_is_redacted_like_the_same_digits_in_a_string():
    result = EgressGuard().inspect({"customer_ref": 12345678901, "as_text": "12345678901"})
    assert result.payload == {"customer_ref": REDACTED, "as_text": REDACTED}
    assert ("$.customer_ref", "long_digits") in {(f.path, f.kind) for f in result.findings}
    assert EgressGuard().inspect([[-12345678901]]).payload == [[REDACTED]]


def test_counts_booleans_floats_and_short_integers_are_not_touched():
    payload = {"qty": 12, "ok": True, "amount": 12345678901.0, "limit": 99999999}
    result = EgressGuard().inspect(payload)
    assert result.payload == payload and not result.findings


def test_timestamps_and_ordinary_numbers_are_not_mistaken_for_phone_numbers():
    payload = {"issued_at": "2026-03-18 15:30:00", "note": "agreement 123456 for 1,500.25 total"}
    assert not EgressGuard().inspect(payload).findings


def test_nested_structures_are_walked_and_paths_are_reported():
    payload = {"rows": [{"ok": 1}, {"address": "1 Example Street"}]}
    result = EgressGuard().inspect(payload)
    assert result.payload["rows"][1]["address"] == REDACTED
    assert result.findings[0].path == "$.rows[1].address"


def test_block_policy_withholds_the_whole_result():
    result = EgressGuard("block").inspect({"rows": [{"email": "a@example.org"}]})
    assert result.blocked and result.payload is None and result.findings


def test_the_input_is_not_mutated():
    payload = {"email": "a@example.org"}
    EgressGuard().inspect(payload)
    assert payload == {"email": "a@example.org"}


def test_an_unknown_policy_is_rejected_and_the_denied_fields_are_configurable():
    with pytest.raises(ValueError):
        EgressGuard("allow")
    custom = EgressGuard(denied_fields=frozenset({"secret_note"}))
    assert custom.inspect({"secret_note": "x"}).payload == {"secret_note": REDACTED}
    assert custom.inspect({"email": "x"}).payload == {"email": "x"}
