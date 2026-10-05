"""The chat endpoint as a framework-free service.

`ChatService.handle` takes the decoded JSON body and returns a status and an envelope. It
owns the rules that sit between an HTTP request and the agent:

* gates: the feature switch (404), a usable model configuration (503);
* request validation (400): a non-empty list of `user` and `assistant` messages with text,
  the last one from the user, a known product line, ISO dates in the right order;
* trimming: only the last `MAX_MESSAGES` messages, each cut to `MAX_MESSAGE_CHARS`, so a
  long conversation cannot run the token bill up;
* error mapping: a failing model provider is a 502 with a generic message (the provider's
  own text is never forwarded), anything unexpected is a 500 with no detail.

`wsgi.create_app` puts a plain WSGI layer on top. Another framework needs only a handler
that calls `handle`.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

from llm_agent_evals.agent import Agent
from llm_agent_evals.envelope import build_envelope, error_envelope
from llm_agent_evals.errors import ConfigError, ModelError
from llm_agent_evals.llm.base import Message
from llm_agent_evals.session import ALL_LINES, Session, Window
from llm_agent_evals.settings import AgentSettings, LLMSettings

MAX_MESSAGES = 20
MAX_MESSAGE_CHARS = 4000
ROLES = ("user", "assistant")


@dataclass(frozen=True)
class ServiceResponse:
    status: int
    body: dict[str, Any]


class RequestError(ValueError):
    """The request body is not acceptable. The message is safe to show to the caller."""


def _new_run_id() -> str:
    return f"run-{uuid.uuid4().hex[:12]}"


def parse_messages(raw: Any) -> list[Message]:
    """Validate and trim the conversation. Raises `RequestError`."""
    if not isinstance(raw, list) or not raw:
        raise RequestError("messages must be a non-empty list.")
    messages: list[Message] = []
    for item in raw:
        if not isinstance(item, dict) or item.get("role") not in ROLES:
            raise RequestError("each message needs a role of 'user' or 'assistant'.")
        content = item.get("content")
        if not isinstance(content, str) or not content.strip():
            raise RequestError("each message needs non-empty text content.")
        messages.append(Message(role=item["role"], content=content.strip()[:MAX_MESSAGE_CHARS]))
    if messages[-1].role != "user":
        raise RequestError("the last message must come from the user.")
    return messages[-MAX_MESSAGES:]


def _parse_day(value: Any, field: str, default: date) -> date:
    if value in (None, ""):
        return default
    try:
        return date.fromisoformat(str(value).strip())
    except ValueError as exc:
        raise RequestError(f"{field} is not valid: use YYYY-MM-DD.") from exc


class ChatService:
    """Validates a chat request, runs the agent and wraps the answer in the envelope."""

    def __init__(
        self,
        agent: Agent,
        *,
        agent_settings: AgentSettings | None = None,
        llm_settings: LLMSettings | None = None,
        new_run_id: Callable[[], str] = _new_run_id,
    ) -> None:
        self.agent = agent
        self.agent_settings = agent_settings or agent.settings
        self.llm_settings = llm_settings or LLMSettings()
        self._new_run_id = new_run_id

    def handle(self, payload: Any) -> ServiceResponse:
        run_id = self._new_run_id()
        now = self.agent.current_time()
        if not self.agent_settings.chat_enabled:
            return self._fail(404, "chat_disabled", "The chat agent is disabled.", now, run_id)
        try:
            self.llm_settings.require_ready()
        except ConfigError as exc:
            return self._fail(503, "model_not_configured", str(exc), now, run_id)
        try:
            messages, session = self._parse(payload, now.date())
        except RequestError as exc:
            return self._fail(400, "invalid_request", str(exc), now, run_id)
        return self._answer(messages, session, now, run_id)

    def _parse(self, payload: Any, today: date) -> tuple[list[Message], Session]:
        if not isinstance(payload, Mapping):
            raise RequestError("the request body must be a JSON object.")
        messages = parse_messages(payload.get("messages"))
        line = str(payload.get("product_line") or "").strip().lower()
        if line != ALL_LINES and line not in self.agent.sources:
            allowed = ", ".join((*self.agent.lines, ALL_LINES))
            raise RequestError(f"product_line must be one of: {allowed}.")
        start = _parse_day(payload.get("date_from"), "date_from", today)
        end = _parse_day(payload.get("date_to"), "date_to", today)
        if start > end:
            raise RequestError("date_from must not be after date_to.")
        return messages, Session(line, Window(start, end))

    def _answer(
        self, messages: list[Message], session: Session, now: datetime, run_id: str
    ) -> ServiceResponse:
        filters = {
            "product_line": session.product_line,
            "date_from": session.date_from,
            "date_to": session.date_to,
            "provider": self.llm_settings.provider,
            "model": self.llm_settings.model,
        }
        try:
            result = self.agent.run(messages, session, run_id)
        except ModelError:
            return self._fail(502, "model_failed", "The AI service failed. Try again.", now, run_id)
        except Exception:  # noqa: BLE001 - never leak internals through the API
            message = "The request could not be completed."
            return self._fail(500, "internal_error", message, now, run_id)
        lines = self.agent.lines if session.product_line == ALL_LINES else (session.product_line,)
        body = build_envelope([result.response], lines, now=now, filters=filters, run_id=run_id)
        return ServiceResponse(200, body)

    @staticmethod
    def _fail(status: int, code: str, message: str, now: datetime, run_id: str) -> ServiceResponse:
        return ServiceResponse(status, error_envelope(code, message, now=now, run_id=run_id))
