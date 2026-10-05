import pytest

from llm_agent_evals.jsonschema_lite import normalize_args, validate

SCHEMA = {
    "type": "object",
    "properties": {
        "count": {"type": "integer", "minimum": 1, "maximum": 10, "default": 5},
        "mode": {"type": "string", "enum": ["a", "b"]},
        "day": {"type": "string", "format": "date"},
        "names": {
            "type": "array",
            "items": {"type": "string", "minLength": 2},
            "minItems": 1,
            "maxItems": 2,
        },
        "ratio": {"type": "number"},
        "flag": {"type": "boolean", "default": False},
    },
    "required": ["mode"],
    "additionalProperties": False,
}


def test_defaults_are_filled_and_valid_input_has_no_errors():
    clean, errors = normalize_args({"mode": "a"}, SCHEMA)
    assert errors == []
    assert clean == {"mode": "a", "count": 5, "flag": False}


def test_all_problems_are_reported_at_once():
    _, errors = normalize_args({"mode": "z", "count": 99, "extra": 1}, SCHEMA)
    joined = " | ".join(errors)
    assert "must be one of" in joined
    assert "must be <= 10" in joined
    assert "is not an allowed argument" in joined


def test_required_property_is_enforced():
    _, errors = normalize_args({}, SCHEMA)
    assert any("mode: is required" in e for e in errors)


@pytest.mark.parametrize(
    ("value", "expected"),
    [("7", 7), (" 7 ", 7), (7.0, 7)],
)
def test_harmless_number_mistakes_are_coerced(value, expected):
    clean, errors = normalize_args({"mode": "a", "count": value}, SCHEMA)
    assert errors == [] and clean["count"] == expected


def test_a_string_is_not_coerced_into_a_boolean_or_a_wrong_number():
    _, errors = normalize_args({"mode": "a", "count": "seven", "flag": "true"}, SCHEMA)
    assert len(errors) == 2


def test_booleans_are_not_numbers():
    _, errors = normalize_args({"mode": "a", "count": True}, SCHEMA)
    assert errors and "expected integer" in errors[0]


@pytest.mark.parametrize("day", ["2026-02-30", "2026-3-1", "20260301", "yesterday"])
def test_dates_must_be_real_iso_dates(day):
    _, errors = normalize_args({"mode": "a", "day": day}, SCHEMA)
    assert any("YYYY-MM-DD" in e for e in errors)


def test_array_bounds_and_item_rules():
    assert validate(["ab"], SCHEMA["properties"]["names"]) == []
    assert validate([], SCHEMA["properties"]["names"])
    assert validate(["ab", "cd", "ef"], SCHEMA["properties"]["names"])
    assert validate(["a"], SCHEMA["properties"]["names"])


def test_non_object_arguments_are_one_clear_error():
    clean, errors = normalize_args(["x"], SCHEMA)
    assert clean == {} and "expected an object" in errors[0]


def test_type_lists_and_null_are_supported():
    schema = {"type": ["string", "null"]}
    assert validate(None, schema) == []
    assert validate("x", schema) == []
    assert validate(3, schema)
