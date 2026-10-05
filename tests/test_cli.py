import json
import shutil
from pathlib import Path

import pytest

from llm_agent_evals import __version__, cli

ROOT = Path(__file__).resolve().parents[1]


def run(capsys, *argv):
    code = cli.main(list(argv))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_version(capsys):
    with pytest.raises(SystemExit) as caught:
        cli.main(["--version"])
    assert caught.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_eval_runs_the_shipped_scenarios_and_exits_zero(capsys):
    code, out, _ = run(capsys, "eval", "--scenarios", str(ROOT / "scenarios"))
    assert code == 0
    assert "Evaluation report (scripted mode)" in out and "| pass_rate | 100 |" in out
    assert "| X05 | self_test | self-test | detected |" in out


def test_eval_can_write_a_json_report_to_a_file(capsys, tmp_path):
    target = tmp_path / "report.json"
    code, out, _ = run(
        capsys,
        "eval",
        "--scenarios",
        str(ROOT / "scenarios"),
        "--format",
        "json",
        "--out",
        str(target),
    )
    assert code == 0 and out == ""
    data = json.loads(target.read_text(encoding="utf-8"))
    assert data["metrics"]["golden"] >= 30 and data["mode"] == "scripted"


def test_eval_exits_nonzero_when_a_scenario_misses_its_expectation(capsys, tmp_path):
    scenarios = tmp_path / "scenarios"
    shutil.copytree(ROOT / "scenarios", scenarios)
    broken = next(scenarios.glob("s01_*.json"))
    body = json.loads(broken.read_text(encoding="utf-8"))
    body["expect"]["ground_truth"][0]["equals"] = 1.0
    broken.write_text(json.dumps(body), encoding="utf-8")
    code, out, _ = run(capsys, "eval", "--scenarios", str(scenarios))
    assert code == 1 and "| S01 | factual | golden | FAIL |" in out


def test_eval_gates_on_minimums_and_reports_them_on_stderr(capsys):
    code, _, err = run(
        capsys, "eval", "--scenarios", str(ROOT / "scenarios"), "--min", "pass_rate=101"
    )
    assert code == 1 and "gate: pass_rate: 100 is below the minimum 101" in err
    code, _, _ = run(
        capsys, "eval", "--scenarios", str(ROOT / "scenarios"), "--min", "pass_rate=100"
    )
    assert code == 0


def test_a_malformed_minimum_is_an_error_exit(capsys):
    code, _, err = run(capsys, "eval", "--scenarios", str(ROOT / "scenarios"), "--min", "pass_rate")
    assert code == 2 and "name=number" in err


def test_live_mode_refuses_to_run_with_the_mock_provider(capsys, monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    code, _, err = run(capsys, "eval", "--scenarios", str(ROOT / "scenarios"), "--live")
    assert code == 2 and "LLM_PROVIDER" in err


def test_live_mode_needs_a_key_for_a_real_provider(capsys, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("LLM_MODEL", "m")
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    code, _, err = run(capsys, "eval", "--scenarios", str(ROOT / "scenarios"), "--live")
    assert code == 2 and "LLM_API_KEY" in err


def test_a_missing_scenario_directory_is_an_error_exit(capsys, tmp_path):
    code, _, err = run(capsys, "eval", "--scenarios", str(tmp_path / "nope"))
    assert code == 2 and "not found" in err


def test_ask_answers_from_the_demo_data(capsys):
    code, out, _ = run(capsys, "ask", "Which portfolios are at high risk?")
    assert code == 0 and "confidence: high" in out and "tools: filter_portfolios_by_risk" in out


def test_ask_can_print_the_raw_answer_object(capsys):
    code, out, _ = run(capsys, "ask", "who are the top agents", "--json", "--line", "consumer")
    answer = json.loads(out)
    assert code == 0 and set(answer) >= {"text", "confidence", "reference_date"}
    assert answer["reference_date"] == "2026-03-18"


def test_ask_accepts_an_explicit_window(capsys):
    code, out, _ = run(
        capsys, "ask", "status breakdown", "--from", "2026-03-10", "--to", "2026-03-12", "--json"
    )
    assert code == 0 and json.loads(out)["reference_date"] == "2026-03-12"


def test_serve_wires_the_service_and_the_requested_address(capsys, monkeypatch):
    seen = {}
    monkeypatch.setattr(
        cli, "serve", lambda service, host, port: seen.update(host=host, port=port, service=service)
    )
    code, _, err = run(capsys, "serve", "--port", "8123")
    assert code == 0 and seen["host"] == "127.0.0.1" and seen["port"] == 8123
    assert "http://127.0.0.1:8123" in err
    assert (
        seen["service"]
        .handle(
            {
                "messages": [{"role": "user", "content": "high risk portfolios"}],
                "product_line": "auto",
            }
        )
        .status
        == 200
    )


def test_metrics_prints_writes_and_checks(capsys, tmp_path):
    code, out, _ = run(capsys, "metrics")
    assert code == 0 and json.loads(out)["kpis"]["conversion_pct"]
    target = tmp_path / "registry.json"
    assert run(capsys, "metrics", "--write", str(target))[0] == 0
    assert run(capsys, "metrics", "--check", str(target))[0] == 0
    target.write_text("{}", encoding="utf-8")
    code, _, err = run(capsys, "metrics", "--check", str(target))
    assert code == 1 and "out of date" in err
    assert (
        run(capsys, "metrics", "--check", str(ROOT / "docs" / "metric-registry.default.json"))[0]
        == 0
    )


def test_ledger_verify_passes_on_the_shipped_ledger(capsys):
    code, _, err = run(
        capsys,
        "ledger",
        "verify",
        "--ledger",
        str(ROOT / "ledger" / "entries.json"),
        "--scenarios",
        str(ROOT / "scenarios"),
        "--root",
        str(ROOT),
    )
    assert code == 0 and err == ""


def test_ledger_verify_reports_every_problem_and_exits_nonzero(capsys, tmp_path):
    ledger = tmp_path / "ledger.json"
    ledger.write_text(
        json.dumps(
            {
                "title": "T",
                "entries": [
                    {
                        "id": "L-1",
                        "class": "c",
                        "status": "closed",
                        "finding": "f",
                        "fix": "x",
                        "evidence": [{"kind": "scenario", "ref": "S99"}],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    code, _, err = run(
        capsys,
        "ledger",
        "verify",
        "--ledger",
        str(ledger),
        "--scenarios",
        str(ROOT / "scenarios"),
        "--root",
        str(ROOT),
    )
    assert code == 1 and "scenario 'S99' does not exist" in err


def test_ledger_generate_writes_markdown_and_html(capsys, tmp_path):
    md, html = tmp_path / "ledger.md", tmp_path / "ledger.html"
    code, _, _ = run(
        capsys,
        "ledger",
        "generate",
        "--ledger",
        str(ROOT / "ledger" / "entries.json"),
        "--scenarios",
        str(ROOT / "scenarios"),
        "--out",
        str(md),
        "--html",
        str(html),
    )
    assert code == 0
    assert md.read_text(encoding="utf-8") == (ROOT / "docs" / "reliability-ledger.md").read_text(
        encoding="utf-8"
    )
    assert html.read_text(encoding="utf-8").startswith("<!doctype html>")


def test_tools_lists_every_tool_in_text_and_json(capsys):
    code, out, _ = run(capsys, "tools")
    assert code == 0 and len(out.strip().splitlines()) == 15
    code, out, _ = run(capsys, "tools", "--json")
    assert next(spec["name"] for spec in json.loads(out)) == "get_portfolio_metrics"


def test_prompt_prints_the_rendered_system_prompt(capsys):
    code, out, _ = run(capsys, "prompt", "--reference-date", "2026-03-18")
    assert code == 0 and "2026-03-18" in out and "{{" not in out


def test_an_unknown_command_is_an_argparse_error(capsys):
    with pytest.raises(SystemExit) as caught:
        cli.main(["nonsense"])
    assert caught.value.code == 2
