"""A small JSON-schema subset validator for tool arguments.

Supported keywords: `type` (string, integer, number, boolean, array, object, null,
or a list of those), `enum`, `required`, `properties`, `additionalProperties: false`,
`items`, `minimum`, `maximum`, `minLength`, `maxLength`, `minItems`, `maxItems`,
`format: "date"` and `default`.

This is not a general validator. It covers exactly what the tool schemas use,
which keeps the runtime free of third-party dependencies. `normalize_args`
coerces the harmless mistakes models make (numbers sent as strings), fills
defaults and reports every remaining problem at once, so the model can fix all of
them in one retry.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from datetime import date
from typing import Any

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

_TYPE_CHECKS: dict[str, Callable[[Any], bool]] = {
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "array": lambda v: isinstance(v, list),
    "object": lambda v: isinstance(v, dict),
    "null": lambda v: v is None,
}


def _type_names(schema: dict[str, Any]) -> list[str]:
    declared = schema.get("type")
    if declared is None:
        return []
    return list(declared) if isinstance(declared, list) else [declared]


def _is_iso_date(value: str) -> bool:
    if not _DATE_RE.match(value):
        return False
    try:
        date.fromisoformat(value)
    except ValueError:
        return False
    return True


def _check_scalar_bounds(value: Any, schema: dict[str, Any], path: str) -> list[str]:
    errors: list[str] = []
    if isinstance(value, bool):
        return errors
    if isinstance(value, (int, float)):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{path}: must be >= {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{path}: must be <= {schema['maximum']}")
    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            errors.append(f"{path}: must have at least {schema['minLength']} characters")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            errors.append(f"{path}: must have at most {schema['maxLength']} characters")
        if schema.get("format") == "date" and not _is_iso_date(value):
            errors.append(f"{path}: must be a date written YYYY-MM-DD")
    return errors


def _check_array(value: list[Any], schema: dict[str, Any], path: str) -> list[str]:
    errors: list[str] = []
    if "minItems" in schema and len(value) < schema["minItems"]:
        errors.append(f"{path}: needs at least {schema['minItems']} items")
    if "maxItems" in schema and len(value) > schema["maxItems"]:
        errors.append(f"{path}: allows at most {schema['maxItems']} items")
    item_schema = schema.get("items")
    if item_schema:
        for index, item in enumerate(value):
            errors.extend(validate(item, item_schema, f"{path}[{index}]"))
    return errors


def _check_object(value: dict[str, Any], schema: dict[str, Any], path: str) -> list[str]:
    errors: list[str] = []
    properties = schema.get("properties", {})
    errors.extend(
        f"{path}.{name}: is required" for name in schema.get("required", []) if name not in value
    )
    if schema.get("additionalProperties") is False:
        errors.extend(
            f"{path}.{name}: is not an allowed argument" for name in value if name not in properties
        )
    for name, sub_schema in properties.items():
        if name in value:
            errors.extend(validate(value[name], sub_schema, f"{path}.{name}"))
    return errors


def validate(value: Any, schema: dict[str, Any], path: str = "$") -> list[str]:
    """Return a list of human-readable problems (empty when the value is valid)."""
    names = _type_names(schema)
    if names and not any(_TYPE_CHECKS[name](value) for name in names):
        return [f"{path}: expected {' or '.join(names)}, got {type(value).__name__}"]
    if "enum" in schema and value not in schema["enum"]:
        allowed = ", ".join(repr(option) for option in schema["enum"])
        return [f"{path}: must be one of {allowed}"]
    errors = _check_scalar_bounds(value, schema, path)
    if isinstance(value, list):
        errors.extend(_check_array(value, schema, path))
    if isinstance(value, dict):
        errors.extend(_check_object(value, schema, path))
    return errors


def _coerce(value: Any, schema: dict[str, Any]) -> Any:
    names = _type_names(schema)
    if isinstance(value, str) and "string" not in names:
        text = value.strip()
        if "integer" in names and re.fullmatch(r"-?\d+", text):
            return int(text)
        if "number" in names and re.fullmatch(r"-?\d+(\.\d+)?", text):
            return float(text)
    is_integral_float = isinstance(value, float) and value.is_integer()
    if is_integral_float and "integer" in names and "number" not in names:
        return int(value)
    return value


def normalize_args(args: Any, schema: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Coerce, fill defaults and validate tool arguments against `schema`."""
    if not isinstance(args, dict):
        return {}, [f"$: expected an object of arguments, got {type(args).__name__}"]
    properties = schema.get("properties", {})
    out: dict[str, Any] = {}
    for name, value in args.items():
        out[name] = _coerce(value, properties[name]) if name in properties else value
    for name, sub_schema in properties.items():
        if name not in out and "default" in sub_schema:
            out[name] = sub_schema["default"]
    return out, validate(out, schema)
