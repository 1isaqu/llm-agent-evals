"""The agent loop: model, tools and guards, written once over neutral messages.

One `Agent.run` call answers one question. The shape of a request:

1. Load the portfolio rollup for the session. If the data source fails here the request
   degrades to an honest "data not available" answer instead of an error.
2. Build the system prompt (rules, tool summary, session context with date anchors).
3. Loop. Each round asks the model for a turn. Tool calls are validated and run under the
   `RunState` guard (step and time ceilings, memoisation, spiral detection, per-tool
   disabling, retry, result cap), their results pass the egress guard, and the guarded
   results go back to the model. A final round without tools is forced when the guard
   says so or when the round limit is reached.
4. Parse the model's last text into the answer contract, refuse unsafe text and return.

The loop knows nothing about a provider. Wire adapters (`llm/`) translate the neutral
messages. Time, sleep, the clock for "today", the cache and the circuit breaker are
injected, so a run is reproducible in tests.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from llm_agent_evals.analytics.pace import BaselinePace, PaceProvider
from llm_agent_evals.business import BusinessConfig
from llm_agent_evals.compaction import compact_messages
from llm_agent_evals.context import ToolContext
from llm_agent_evals.errors import ConfigError, ModelError, build_tool_error
from llm_agent_evals.guards import RunGuard, RunState, SharedState, call_hash
from llm_agent_evals.llm.base import LLMClient, Message, ModelRequest, ModelTurn, ToolCall
from llm_agent_evals.privacy import EgressGuard
from llm_agent_evals.prompt import FORCE_FINAL_NUDGE, build_system_prompt
from llm_agent_evals.registry import ToolRegistry
from llm_agent_evals.response import (
    DATA_UNAVAILABLE_TEXT,
    degrade_unsafe,
    empty_response,
    parse_final_text,
)
from llm_agent_evals.session import ALL_LINES, Session
from llm_agent_evals.settings import AgentSettings
from llm_agent_evals.sources.base import DataSource
from llm_agent_evals.telemetry import Observer, ToolCallRecord
from llm_agent_evals.tools import build_default_registry

DATASET_UNAVAILABLE_TEXT = f"{DATA_UNAVAILABLE_TEXT} Try again in a moment."


@dataclass(frozen=True)
class ExecutedCall:
    """One tool call as the loop executed it. `result` is what the model was shown."""

    name: str
    args: dict[str, Any]
    result: Any
    record: ToolCallRecord


@dataclass(frozen=True)
class AgentResult:
    """The answer plus everything the evaluation harness needs to judge how it was reached."""

    response: dict[str, Any]
    calls: tuple[ExecutedCall, ...] = ()
    raw_final_text: str = ""
    format_ok: bool = True
    rounds: int = 0
    degraded: str | None = None
    forced_final: bool = False
    compactions: int = 0
    usage: Mapping[str, int] = field(default_factory=dict)


def coerce_messages(messages: Sequence[Message | Mapping[str, Any]]) -> list[Message]:
    """Accept `Message` objects or `{"role", "content"}` mappings."""
    out: list[Message] = []
    for item in messages:
        if isinstance(item, Message):
            out.append(item)
        else:
            out.append(Message(role=str(item["role"]), content=str(item["content"])))
    return out


class Agent:
    """Answers questions about the data behind `sources` using `llm` and the tool registry."""

    def __init__(
        self,
        llm: LLMClient,
        sources: Mapping[str, DataSource],
        *,
        config: BusinessConfig | None = None,
        registry: ToolRegistry | None = None,
        settings: AgentSettings | None = None,
        pace: PaceProvider | None = None,
        shared: SharedState | None = None,
        egress: EgressGuard | None = None,
        observer: Observer | None = None,
        now: Callable[[], datetime] = datetime.now,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        guard: RunGuard | None = None,
        max_tokens: int = 2048,
        temperature: float | None = 0.2,
    ) -> None:
        if not sources:
            raise ConfigError("at least one data source is required")
        self.llm = llm
        self.sources = dict(sources)
        self.settings = settings or AgentSettings()
        self.config = config or self.settings.business_config()
        self.lines = tuple(self.sources)
        self.registry = registry or build_default_registry(self.lines)
        self.pace = pace or BaselinePace()
        self.shared = shared or SharedState()
        self.egress = egress or EgressGuard(self.settings.egress_policy)
        self.observer = observer
        self._now, self._clock, self._sleep = now, clock, sleep
        self.guard = guard or RunGuard(
            max_steps=self.settings.max_steps,
            wall_clock_seconds=self.settings.wall_clock_seconds,
        )
        self.max_tokens, self.temperature = max_tokens, temperature
        self._tool_specs = tuple(self.registry.specs())

    def current_time(self) -> datetime:
        """The agent's clock, which anchors "today" for the model and for defaults."""
        return self._now()

    def run(
        self,
        messages: Sequence[Message | Mapping[str, Any]],
        session: Session,
        run_id: str = "",
    ) -> AgentResult:
        """Answer the conversation. Raises `ModelError` when the model provider fails."""
        if session.product_line != ALL_LINES and session.product_line not in self.sources:
            raise ConfigError(f"unknown product line {session.product_line!r}")
        started = self._clock()
        now = self.current_time()
        ctx = ToolContext(session, self.config, self.sources, now, pace=self.pace)
        try:
            portfolio_count = len(ctx.entries())
        except Exception as exc:  # noqa: BLE001 - any failure here must degrade, not crash
            return self._dataset_unavailable(session, exc, run_id)
        system = build_system_prompt(
            self.config, self.registry, self.lines, session, now, portfolio_count
        )
        attempt = _Attempt(self, ctx, system, coerce_messages(messages), run_id)
        try:
            result = attempt.execute()
        except ModelError as exc:
            self._emit("model_error", run_id=run_id, error_type=type(exc).__name__)
            raise
        self._emit(
            "run_end",
            run_id=run_id,
            rounds=result.rounds,
            steps=len(result.calls),
            degraded=result.degraded,
            forced_final=result.forced_final,
            confidence=result.response["confidence"],
            latency_ms=round((self._clock() - started) * 1000, 1),
        )
        return result

    def _dataset_unavailable(self, session: Session, exc: Exception, run_id: str) -> AgentResult:
        self._emit("dataset_error", run_id=run_id, error_type=type(exc).__name__)
        response = empty_response(DATASET_UNAVAILABLE_TEXT)
        response["reference_date"] = session.date_to
        return AgentResult(response=response, degraded="dataset_unavailable")

    def _emit(self, event: str, **payload: Any) -> None:
        if self.observer is None:
            return
        try:
            self.observer(event, payload)
        except Exception:  # noqa: BLE001 - telemetry must never break a request
            return


class _Attempt:
    """The mutable state of one `Agent.run`: conversation, calls made, usage."""

    def __init__(
        self,
        agent: Agent,
        ctx: ToolContext,
        system: str,
        messages: list[Message],
        run_id: str,
    ) -> None:
        self.agent = agent
        self.ctx = ctx
        self.system = system
        self.convo = messages
        self.run_id = run_id
        self.state = RunState(
            agent.guard, shared=agent.shared, clock=agent._clock, sleep=agent._sleep
        )
        self.calls: list[ExecutedCall] = []
        self.usage = {"input": 0, "output": 0}
        self.rounds = 0
        self.compactions = 0
        self.forced = False

    def execute(self) -> AgentResult:
        final_text = self._loop()
        response, format_ok = parse_final_text(final_text)
        response, refused = degrade_unsafe(response)
        response["reference_date"] = self.ctx.session.date_to
        return AgentResult(
            response=response,
            calls=tuple(self.calls),
            raw_final_text=final_text,
            format_ok=format_ok,
            rounds=self.rounds,
            degraded="unsafe_answer" if refused else None,
            forced_final=self.forced,
            compactions=self.compactions,
            usage=dict(self.usage),
        )

    def _loop(self) -> str:
        limit = self.agent.settings.max_rounds
        for round_index in range(limit + 1):
            final = self.state.force_final() or round_index == limit
            turn = self._ask(final)
            if final or not turn.tool_calls:
                self.forced = self.forced or final
                return turn.text
            self._run_tools(turn)
            if self.state.budget_near_limit():
                self.convo = compact_messages(self.convo)
                self.compactions += 1
        return ""

    def _ask(self, final: bool) -> ModelTurn:
        if final:
            self.convo.append(Message(role="user", content=FORCE_FINAL_NUDGE))
        request = ModelRequest(
            system=self.system,
            messages=tuple(self.convo),
            tools=self.agent._tool_specs,
            allow_tools=not final,
            max_tokens=self.agent.max_tokens,
            temperature=self.agent.temperature,
        )
        turn = self.agent.llm.complete(request)
        self.rounds += 1
        for key in self.usage:
            self.usage[key] += int(turn.usage.get(key, 0))
        self.agent._emit(
            "model_turn",
            run_id=self.run_id,
            round=self.rounds,
            final=final,
            tool_call_count=len(turn.tool_calls),
            stop_reason=turn.stop_reason,
        )
        return turn

    def _run_tools(self, turn: ModelTurn) -> None:
        self.convo.append(Message(role="assistant", content=turn.text, tool_calls=turn.tool_calls))
        for call in turn.tool_calls:
            executed = self._execute_call(call)
            self.calls.append(executed)
            content = json.dumps(executed.result, ensure_ascii=False, default=str)
            self.convo.append(Message(role="tool", content=content, tool_call_id=call.id))

    def _execute_call(self, call: ToolCall) -> ExecutedCall:
        clock = self.agent._clock
        started = clock()
        result = self._dispatch(call)
        inspected = self.agent.egress.inspect(result)
        findings = len(inspected.findings)
        result = _egress_blocked_error() if inspected.blocked else inspected.payload
        meta = self.state.last_call_meta
        record = ToolCallRecord(
            tool_name=call.name,
            args_hash=call_hash(call.name, call.args),
            step_index=meta.get("step_index"),
            latency_ms=round((clock() - started) * 1000, 1),
            cache_hit=bool(meta.get("cache_hit", False)),
            truncated=bool(meta.get("truncated", False)),
            row_count=meta.get("row_count"),
            error_type=meta.get("error_type"),
            egress_findings=findings,
        )
        self.agent._emit("tool_call", run_id=self.run_id, **record.as_payload())
        return ExecutedCall(name=call.name, args=dict(call.args), result=result, record=record)

    def _dispatch(self, call: ToolCall) -> Any:
        if call.args_error:
            return self.state.reject(
                call.name,
                build_tool_error(
                    "validation",
                    hint=f"The tool arguments could not be read: {call.args_error}. "
                    "Send a JSON object that matches the tool schema.",
                    user_facing="The request parameters were not valid.",
                ),
            )
        session = self.ctx.session
        return self.state.dispatch(
            call.name,
            call.args,
            lambda: self.agent.registry.call(call.name, call.args, self.ctx),
            product_line=str(call.args.get("product_line") or session.product_line),
            date_from=str(call.args.get("date_from") or session.date_from),
            date_to=str(call.args.get("date_to") or session.date_to),
        )


def _egress_blocked_error() -> dict[str, Any]:
    return build_tool_error(
        "egress_blocked",
        hint="This result contained fields or values that look like personal data and was "
        "withheld. Do not guess its content. Tell the user the data could not be shown.",
        user_facing="That result could not be shown because it may contain personal data.",
    )
