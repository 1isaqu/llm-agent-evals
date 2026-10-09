"""Conversation compaction for the tool loop.

When the tool results of a request approach the token budget, the oldest tool-call groups
are replaced by one short user note listing what was already called. A group is an
assistant message with tool calls plus the tool messages that answer it. Groups are
removed whole: a tool message without its call (or the reverse) is rejected by every
provider.

The note names each call and its arguments and carries no result text. The note has the
user role, and the system prompt ranks user text above tool data, so copying a result into
it would raise instructions hidden in that result to user trust.

The system prompt, every user turn (also those between groups) and the most recent groups
stay untouched. The loop is written once against the neutral message type, so this function
exists once instead of once per provider format.
"""

from __future__ import annotations

from collections.abc import Sequence

from llm_agent_evals.guards import canonical_args
from llm_agent_evals.llm.base import Message
from llm_agent_evals.prompt import COMPACTION_NOTICE

ARGS_CHARS = 100


def tool_groups(messages: Sequence[Message]) -> list[tuple[int, int]]:
    """`(start, end)` index pairs (end exclusive) of each assistant-calls-plus-results group."""
    groups: list[tuple[int, int]] = []
    index = 0
    while index < len(messages):
        message = messages[index]
        if message.role == "assistant" and message.tool_calls:
            end = index + 1
            while end < len(messages) and messages[end].role == "tool":
                end += 1
            groups.append((index, end))
            index = end
        else:
            index += 1
    return groups


def _calls(messages: Sequence[Message], start: int) -> list[str]:
    return [
        f"{call.name}({canonical_args(call.args)[:ARGS_CHARS]})"
        for call in messages[start].tool_calls
    ]


def compact_messages(messages: Sequence[Message], keep_recent_groups: int = 1) -> list[Message]:
    """Return `messages` with all but the newest `keep_recent_groups` tool groups summarised."""
    groups = tool_groups(messages)
    if len(groups) <= keep_recent_groups:
        return list(messages)
    removed = groups[: len(groups) - keep_recent_groups]
    calls: list[str] = []
    for start, _ in removed:
        calls.extend(_calls(messages, start))
    notice = Message(role="user", content=COMPACTION_NOTICE.format(calls="; ".join(calls)))
    dropped = {i for start, end in removed for i in range(start, end)}
    out: list[Message] = []
    noted = False
    for index, message in enumerate(messages):
        if index not in dropped:
            out.append(message)
        elif not noted:
            out.append(notice)
            noted = True
    return out
