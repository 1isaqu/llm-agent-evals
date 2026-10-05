import pytest

from llm_agent_evals.registry import Tool, ToolRegistry
from llm_agent_evals.tools import build_default_registry
from tests.helpers import make_context

CLOSED = {
    "type": "object",
    "properties": {"n": {"type": "integer", "minimum": 1, "default": 1}},
    "additionalProperties": False,
}


def _tool(name="demo_tool", schema=None, handler=None, validator=None):
    return Tool(
        name=name,
        description="Demo.",
        input_schema=schema or CLOSED,
        handler=handler or (lambda ctx, args: {"n": args["n"]}),
        validator=validator,
    )


def test_a_tool_runs_with_defaults_applied():
    registry = ToolRegistry()
    registry.register(_tool())
    assert registry.call("demo_tool", {}, make_context()) == {"n": 1}


@pytest.mark.parametrize("name", ["Demo", "x", "1abc", "has space", "a" * 80])
def test_bad_tool_names_are_a_startup_error(name):
    with pytest.raises(ValueError, match="invalid tool name"):
        ToolRegistry().register(_tool(name=name))


def test_duplicate_tools_are_a_startup_error():
    registry = ToolRegistry()
    registry.register(_tool())
    with pytest.raises(ValueError, match="duplicate"):
        registry.register(_tool())


def test_open_schemas_are_a_startup_error():
    open_schema = {"type": "object", "properties": {}}
    with pytest.raises(ValueError, match="additionalProperties"):
        ToolRegistry().register(_tool(schema=open_schema))


def test_a_required_property_that_is_not_defined_is_a_startup_error():
    schema = {**CLOSED, "required": ["ghost"]}
    with pytest.raises(ValueError, match="not defined"):
        ToolRegistry().register(_tool(schema=schema))


def test_unknown_tool_is_a_structured_error_listing_the_real_ones():
    registry = ToolRegistry()
    registry.register(_tool())
    result = registry.call("nope", {}, make_context())
    assert result["error_type"] == "unknown_tool"
    assert result["available_tools"] == ["demo_tool"]


def test_invalid_arguments_come_back_as_a_validation_error_not_an_exception():
    registry = ToolRegistry()
    registry.register(_tool())
    result = registry.call("demo_tool", {"n": 0, "x": 1}, make_context())
    assert result["error_type"] == "validation" and result["retryable"] is True
    assert len(result["errors"]) == 2


def test_cross_field_validator_runs_after_the_schema():
    registry = ToolRegistry()
    registry.register(
        _tool(validator=lambda args, ctx: ["n must be even"] if args["n"] % 2 else [])
    )
    assert registry.call("demo_tool", {"n": 3}, make_context())["errors"] == ["n must be even"]
    assert registry.call("demo_tool", {"n": 4}, make_context()) == {"n": 4}


def test_handler_exceptions_are_not_swallowed_here():
    registry = ToolRegistry()

    def boom(ctx, args):
        raise RuntimeError("bug")

    registry.register(_tool(handler=boom))
    with pytest.raises(RuntimeError):
        registry.call("demo_tool", {}, make_context())


def test_specs_are_copies():
    registry = ToolRegistry()
    registry.register(_tool())
    spec = registry.specs()[0]
    spec["input_schema"]["properties"].clear()
    assert registry.specs()[0]["input_schema"]["properties"]


def test_default_registry_has_the_fifteen_tools_and_every_schema_is_closed():
    registry = build_default_registry()
    assert len(registry) == 15
    for spec in registry.specs():
        assert spec["input_schema"]["additionalProperties"] is False
        assert spec["description"].strip()


def test_product_line_enum_follows_the_configured_lines():
    registry = build_default_registry(("alpha", "beta"))
    spec = next(s for s in registry.specs() if s["name"] == "query_kpi_history")
    assert spec["input_schema"]["properties"]["product_line"]["enum"] == ["alpha", "beta"]
