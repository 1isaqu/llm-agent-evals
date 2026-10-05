"""Tool registry: schema, handler and validation live in one object.

A tool is defined once, and the schema shown to the model, the validation applied to its
arguments and the code that runs are all derived from that one definition. There is no
second place where a description or an enum could drift out of date.

Policy enforced at registration time (a mistake is a startup error, not a runtime one):

* schemas are closed (`additionalProperties: false`), so a hallucinated argument is
  rejected instead of ignored;
* tool names are lower snake case;
* every `required` property exists.

`ToolRegistry.call` turns every problem the model can cause (unknown tool, malformed or
out-of-range arguments, failed cross-field rule) into a structured `validation` or
`unknown_tool` result with an actionable hint. Exceptions raised by a handler are *not*
caught here: they belong to the run guard, which retries, counts and classifies them.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from llm_agent_evals.errors import build_tool_error
from llm_agent_evals.jsonschema_lite import normalize_args

if TYPE_CHECKING:
    from llm_agent_evals.context import ToolContext

Handler = Callable[["ToolContext", dict[str, Any]], Any]
Validator = Callable[[dict[str, Any], "ToolContext"], list[str]]

_NAME = re.compile(r"^[a-z][a-z0-9_]{2,63}$")


@dataclass(frozen=True)
class Tool:
    """One tool: what the model sees (name, description, schema) and what runs."""

    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Handler
    validator: Validator | None = None


class ToolRegistry:
    """An ordered collection of tools with schema-validated dispatch."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> Tool:
        if not _NAME.match(tool.name):
            raise ValueError(f"invalid tool name {tool.name!r}")
        if tool.name in self._tools:
            raise ValueError(f"duplicate tool {tool.name!r}")
        schema = tool.input_schema
        if schema.get("type") != "object":
            raise ValueError(f"{tool.name}: input schema must describe an object")
        if schema.get("additionalProperties") is not False:
            raise ValueError(f"{tool.name}: input schema must set additionalProperties to false")
        missing = [
            name for name in schema.get("required", []) if name not in schema.get("properties", {})
        ]
        if missing:
            raise ValueError(f"{tool.name}: required properties are not defined: {missing}")
        self._tools[tool.name] = tool
        return tool

    def names(self) -> list[str]:
        return list(self._tools)

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def specs(self) -> list[dict[str, Any]]:
        """Provider-neutral tool specs. Wire adapters convert these to their own format."""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": copy.deepcopy(tool.input_schema),
            }
            for tool in self._tools.values()
        ]

    def call(self, name: str, args: Any, ctx: ToolContext) -> Any:
        """Validate `args` and run the tool. Model mistakes come back as structured errors."""
        tool = self._tools.get(name)
        if tool is None:
            return build_tool_error(
                "unknown_tool",
                hint=f"Unknown tool {name!r}. Use only the tools in the list.",
                user_facing="That request does not match any available data source.",
                available_tools=sorted(self._tools),
            )
        clean, errors = normalize_args(args, tool.input_schema)
        if not errors and tool.validator is not None:
            errors = tool.validator(clean, ctx)
        if errors:
            return build_tool_error(
                "validation",
                hint="Fix the arguments and call the tool again: " + "; ".join(errors),
                user_facing="The request parameters were not valid.",
                errors=errors,
            )
        return tool.handler(ctx, clean)
