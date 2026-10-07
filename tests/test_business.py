import json

import pytest

from llm_agent_evals.business import DEFAULT_STATUS_IDS, BusinessConfig
from llm_agent_evals.errors import ConfigError


def test_derived_status_sets_follow_the_definitions():
    config = BusinessConfig()
    ids = DEFAULT_STATUS_IDS
    assert config.approved_ids == tuple(sorted(ids[s] for s in ("ACTIVE", "PAID", "PAID_MANUAL")))
    assert set(config.generated_ids) == set(config.approved_ids) | {
        ids["BROKEN"],
        ids["AUTO_BROKEN"],
    }
    assert set(config.universe_ids) == set(config.generated_ids) | {ids["PENDING_EXCEPTION"]}
    assert set(config.rollup_ids) == set(config.universe_ids) | {ids["REJECTED"]}


def test_auto_broken_is_generated_but_not_on_the_broken_axis():
    config = BusinessConfig()
    assert DEFAULT_STATUS_IDS["AUTO_BROKEN"] in config.generated_ids
    assert DEFAULT_STATUS_IDS["AUTO_BROKEN"] not in config.broken_ids


def test_a_remapped_catalogue_changes_every_derived_set():
    config = BusinessConfig(status_ids={**DEFAULT_STATUS_IDS, "ACTIVE": 101, "REJECTED": 102})
    assert 101 in config.generated_ids
    assert 102 in config.rollup_ids and 102 not in config.generated_ids


def test_label_of_unknown_id_is_stable():
    assert BusinessConfig().label_of(10) == "ACTIVE"
    assert BusinessConfig().label_of(999) == "STATUS_999"


@pytest.mark.parametrize(
    ("risk", "level"),
    [(0.0, "low"), (20.0, "low"), (20.01, "medium"), (40.0, "medium"), (40.01, "high")],
)
def test_risk_level_boundaries_are_inclusive_on_the_lower_band(risk, level):
    assert BusinessConfig().risk_level(risk) == level


def test_unknown_status_label_in_a_set_is_rejected():
    with pytest.raises(ConfigError, match="unknown status label"):
        BusinessConfig(approved=("ACTIVE", "NOT_A_STATUS"))


def test_duplicate_status_ids_are_rejected():
    with pytest.raises(ConfigError, match="distinct"):
        BusinessConfig(status_ids={"ACTIVE": 1, "PAID": 1})


def test_thresholds_must_be_ordered():
    with pytest.raises(ConfigError, match="risk_low_max"):
        BusinessConfig(risk_low_max=50.0, risk_medium_max=40.0)


def test_pace_hours_must_be_ordered():
    with pytest.raises(ConfigError, match="pace hours"):
        BusinessConfig(pace_first_hour=12, pace_last_hour=8)


def test_from_mapping_converts_lists_and_rejects_unknown_keys():
    config = BusinessConfig.from_mapping({"excluded_agent_patterns": ["bot%"], "grace_days": 7})
    assert config.excluded_agent_patterns == ("bot%",)
    assert config.grace_days == 7
    with pytest.raises(ConfigError, match="unknown business config keys: nope"):
        BusinessConfig.from_mapping({"nope": 1})


def test_from_file_reads_json_and_reports_bad_files(tmp_path):
    good = tmp_path / "config.json"
    good.write_text(json.dumps({"max_window_days": 31}), encoding="utf-8")
    assert BusinessConfig.from_file(good).max_window_days == 31
    with pytest.raises(ConfigError, match="cannot read"):
        BusinessConfig.from_file(tmp_path / "missing.json")
    bad = tmp_path / "bad.json"
    bad.write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(ConfigError, match="JSON object"):
        BusinessConfig.from_file(bad)
