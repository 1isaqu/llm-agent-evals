"""Command line interface: `llm-agent-evals <command>`.

    eval      run the scenarios (scripted model by default, `--live` for a real model)
    ask       ask one question against the demo data (or SQL Server) with the configured model
    serve     run the HTTP chat service
    metrics   print, write or check the metric registry snapshot
    ledger    generate or verify the reliability ledger
    tools     list the tools the model can call
    prompt    print the rendered system prompt

Every command returns an exit code. `eval`, `metrics --check` and `ledger verify` return 1 when
something is wrong, so they can gate CI.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from datetime import date
from pathlib import Path

from llm_agent_evals import __version__
from llm_agent_evals.business import BusinessConfig
from llm_agent_evals.errors import AgentError
from llm_agent_evals.evals import compute_metrics, gate_failures, load_scenarios, run_scenarios
from llm_agent_evals.evals import report as eval_report
from llm_agent_evals.ledger import load_ledger, render_html, render_markdown, verify_ledger
from llm_agent_evals.llm import build_llm
from llm_agent_evals.metrics import MetricRegistry
from llm_agent_evals.prompt import render_system_prompt
from llm_agent_evals.runtime import BACKENDS, build_agent
from llm_agent_evals.service import ChatService
from llm_agent_evals.session import ALL_LINES, PRODUCT_LINES, Session, Window
from llm_agent_evals.settings import LLMSettings
from llm_agent_evals.tools import build_default_registry
from llm_agent_evals.wsgi import serve

DEFAULT_SCENARIOS = "scenarios"
DEFAULT_LEDGER = "ledger/entries.json"


def _write_or_print(text: str, out: str | None) -> None:
    if out:
        Path(out).write_text(text, encoding="utf-8", newline="\n")
    else:
        sys.stdout.write(text)


def _parse_minimums(pairs: Sequence[str]) -> dict[str, float]:
    minimums: dict[str, float] = {}
    for pair in pairs:
        name, _, value = pair.partition("=")
        try:
            minimums[name.strip()] = float(value)
        except ValueError as exc:
            raise AgentError(f"--min needs name=number, got {pair!r}") from exc
    return minimums


# -- eval ---------------------------------------------------------------------


def cmd_eval(args: argparse.Namespace) -> int:
    scenarios = load_scenarios(args.scenarios)
    if args.live:
        settings = LLMSettings.from_env()
        if not settings.is_real:
            raise AgentError("live mode needs LLM_PROVIDER set to a real provider")
        settings.require_ready()
        results = run_scenarios(scenarios, lambda: build_llm(settings))
    else:
        results = run_scenarios(scenarios)
    metrics = compute_metrics(results)
    mode = "live" if args.live else "scripted"
    render = eval_report.to_json if args.format == "json" else eval_report.to_markdown
    _write_or_print(render(results, metrics, mode), args.out)
    failures = gate_failures(metrics, _parse_minimums(args.min))
    for line in failures:
        print(f"gate: {line}", file=sys.stderr)
    if args.live:
        return 1 if failures else 0
    return 0 if not failures and all(r.meets_expectation() for r in results) else 1


# -- ask / serve -------------------------------------------------------------


def _session_from(args: argparse.Namespace, today: date) -> Session:
    try:
        end = date.fromisoformat(args.to) if args.to else today
        start = date.fromisoformat(args.start) if args.start else end.replace(day=1)
        return Session(args.line, Window(start, end))
    except ValueError as exc:
        raise AgentError(f"invalid window: {exc}") from exc


def cmd_ask(args: argparse.Namespace) -> int:
    agent = build_agent(backend=args.backend)
    session = _session_from(args, agent.current_time().date())
    result = agent.run([{"role": "user", "content": args.question}], session, run_id="cli")
    if args.json:
        print(json.dumps(result.response, indent=2, ensure_ascii=False))
        return 0
    response = result.response
    print(response["text"])
    print(f"\nconfidence: {response['confidence']}")
    if result.calls:
        print("tools: " + ", ".join(call.name for call in result.calls))
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    agent = build_agent(backend=args.backend)
    service = ChatService(agent, llm_settings=LLMSettings.from_env())
    print(f"serving on http://{args.host}:{args.port} (POST /chat, GET /health)", file=sys.stderr)
    try:
        serve(service, args.host, args.port)
    except KeyboardInterrupt:
        print("stopped", file=sys.stderr)
    return 0


# -- metrics / ledger / tools / prompt ----------------------------------------


def _registry_json(config: BusinessConfig) -> str:
    return json.dumps(MetricRegistry(config).to_dict(), indent=2, ensure_ascii=False) + "\n"


def cmd_metrics(args: argparse.Namespace) -> int:
    text = _registry_json(BusinessConfig())
    if args.check:
        current = Path(args.check).read_text(encoding="utf-8")
        if current != text:
            print(
                f"{args.check} is out of date: run `llm-agent-evals metrics --write`",
                file=sys.stderr,
            )
            return 1
        return 0
    _write_or_print(text, args.write)
    return 0


def _scenario_status(directory: str) -> dict[str, str]:
    results = run_scenarios(load_scenarios(directory))
    return {r.scenario_id: eval_report.result_status(r) for r in results}


def cmd_ledger(args: argparse.Namespace) -> int:
    ledger = load_ledger(args.ledger)
    scenarios = load_scenarios(args.scenarios)
    if args.action == "verify":
        problems = verify_ledger(ledger, args.root, [s.id for s in scenarios])
        for line in problems:
            print(line, file=sys.stderr)
        return 1 if problems else 0
    status = _scenario_status(args.scenarios)
    _write_or_print(render_markdown(ledger, status), args.out)
    if args.html:
        Path(args.html).write_text(render_html(ledger, status), encoding="utf-8", newline="\n")
    return 0


def cmd_tools(args: argparse.Namespace) -> int:
    specs = build_default_registry().specs()
    if args.json:
        print(json.dumps(specs, indent=2))
        return 0
    for spec in specs:
        print(f"{spec['name']}: {spec['description'].split('. ')[0].rstrip('.')}.")
    return 0


def cmd_prompt(args: argparse.Namespace) -> int:
    registry = build_default_registry()
    text = render_system_prompt(BusinessConfig(), registry, PRODUCT_LINES, args.reference_date)
    print(text)
    return 0


# -- parser -----------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="llm-agent-evals", description=__doc__.split("\n\n")[0])
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    ev = sub.add_parser("eval", help="run the scenarios")
    ev.add_argument("--scenarios", default=DEFAULT_SCENARIOS)
    ev.add_argument("--format", choices=("md", "json"), default="md")
    ev.add_argument("--out")
    ev.add_argument("--live", action="store_true", help="use the model from LLM_PROVIDER")
    ev.add_argument("--min", action="append", default=[], metavar="METRIC=VALUE")
    ev.set_defaults(func=cmd_eval)

    ask = sub.add_parser("ask", help="ask one question")
    ask.add_argument("question")
    ask.add_argument("--line", default="auto", choices=(*PRODUCT_LINES, ALL_LINES))
    ask.add_argument("--from", dest="start", help="first day, YYYY-MM-DD (default: month start)")
    ask.add_argument("--to", help="last day, YYYY-MM-DD (default: today)")
    ask.add_argument("--backend", choices=BACKENDS, default="demo")
    ask.add_argument("--json", action="store_true")
    ask.set_defaults(func=cmd_ask)

    sv = sub.add_parser("serve", help="run the HTTP chat service")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8000)
    sv.add_argument("--backend", choices=BACKENDS, default="demo")
    sv.set_defaults(func=cmd_serve)

    mt = sub.add_parser("metrics", help="print, write or check the metric registry snapshot")
    mt.add_argument("--write", metavar="PATH")
    mt.add_argument("--check", metavar="PATH")
    mt.set_defaults(func=cmd_metrics)

    lg = sub.add_parser("ledger", help="generate or verify the reliability ledger")
    lg.add_argument("action", choices=("generate", "verify"))
    lg.add_argument("--ledger", default=DEFAULT_LEDGER)
    lg.add_argument("--scenarios", default=DEFAULT_SCENARIOS)
    lg.add_argument("--root", default=".")
    lg.add_argument("--out", help="Markdown output path (default: stdout)")
    lg.add_argument("--html", help="also write the HTML page here")
    lg.set_defaults(func=cmd_ledger)

    tl = sub.add_parser("tools", help="list the tools")
    tl.add_argument("--json", action="store_true")
    tl.set_defaults(func=cmd_tools)

    pr = sub.add_parser("prompt", help="print the rendered system prompt")
    pr.add_argument("--reference-date", default="YYYY-MM-DD")
    pr.set_defaults(func=cmd_prompt)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        code: int = args.func(args)
    except AgentError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return code
