import json

import pytest

from llm_agent_evals.response import (
    MAX_ACTIONS,
    MAX_HIGHLIGHTS,
    NO_ANSWER_TEXT,
    REFUSED_TEXT,
    degrade_unsafe,
    extract_json_object,
    normalize_response,
    parse_final_text,
)

GOOD = {
    "text": "All good.",
    "highlights": [{"type": "metric", "label": "Risk", "value": "12%"}],
    "suggested_actions": [{"label": "Next", "prompt": "What else?"}],
    "data_sources": ["get_portfolio_metrics"],
    "confidence": "high",
}


def test_a_well_formed_object_round_trips():
    response, ok = parse_final_text(json.dumps(GOOD))
    assert ok and response == GOOD


def test_code_fences_prose_and_raw_line_breaks_are_tolerated():
    fenced = "```json\n" + json.dumps(GOOD) + "\n```"
    assert parse_final_text(fenced)[1]
    prose = "Here is my answer: " + json.dumps(GOOD) + " Hope that helps."
    assert parse_final_text(prose)[0]["text"] == "All good."
    raw_break = '{"text": "line one\nline two", "confidence": "medium"}'
    response, ok = parse_final_text(raw_break)
    assert ok and response["text"] == "line one\nline two"


@pytest.mark.parametrize("text", ["Plain sentence.", "", "[1, 2, 3]", "{broken", '"just a string"'])
def test_text_that_is_not_an_object_degrades_to_low_confidence_without_raising(text):
    response, ok = parse_final_text(text)
    assert not ok and response["confidence"] == "low"
    assert response["text"] == (text.strip() or NO_ANSWER_TEXT)
    assert response["highlights"] == [] and response["suggested_actions"] == []


def test_an_invalid_confidence_becomes_low():
    for bad in ("certain", "", None, 5):
        assert normalize_response({"text": "x", "confidence": bad}, "")["confidence"] == "low"
    assert normalize_response({"text": "x", "confidence": " HIGH "}, "")["confidence"] == "high"


def test_a_missing_text_falls_back_to_the_raw_output():
    assert normalize_response({"confidence": "high"}, "raw words")["text"] == "raw words"


def test_highlights_are_cleaned_capped_and_typed():
    payload = {
        "text": "x",
        "highlights": [
            {"type": "weird", "label": "A", "value": 5},
            {"type": "anomaly", "label": "B"},
            {"label": ""},
            "nope",
            {"label": "C"},
            {"label": "D"},
            {"label": "E"},
        ],
    }
    highlights = normalize_response(payload, "")["highlights"]
    assert len(highlights) == MAX_HIGHLIGHTS
    assert highlights[0] == {"type": "metric", "label": "A", "value": "5"}
    assert highlights[1] == {"type": "anomaly", "label": "B"}


def test_suggested_actions_are_cleaned_and_capped():
    payload = {
        "text": "x",
        "suggested_actions": [
            {"label": "One", "prompt": " p "},
            {"label": ""},
            {"label": "Two"},
            {"label": "Three"},
            {"label": "Four"},
        ],
    }
    actions = normalize_response(payload, "")["suggested_actions"]
    assert actions[0] == {"label": "One", "prompt": "p"} and len(actions) == MAX_ACTIONS


def test_data_sources_must_be_a_list_of_non_empty_strings():
    assert normalize_response({"text": "x", "data_sources": ["a", "", " ", 3]}, "")[
        "data_sources"
    ] == ["a", "3"]
    assert normalize_response({"text": "x", "data_sources": "a"}, "")["data_sources"] == []


def test_extract_returns_none_when_there_is_no_object():
    assert extract_json_object("nothing here") is None
    assert extract_json_object("}{") is None


def test_unsafe_text_is_replaced_and_never_echoed():
    unsafe = {**GOOD, "text": "Query:\n```sql\nSELECT * FROM agreements\n```"}
    response, refused = degrade_unsafe(unsafe)
    assert refused and response["text"] == REFUSED_TEXT and "SELECT" not in response["text"]
    assert response["confidence"] == "low" and response["data_sources"] == ["get_portfolio_metrics"]


def test_safe_text_is_kept_as_is():
    response, refused = degrade_unsafe(GOOD)
    assert not refused and response == GOOD
