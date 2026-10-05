"""HTTP transport for the model adapters, built on `urllib` (no SDK, no dependency).

A transport is a callable `(url, headers, body, timeout) -> (status, payload)`. It is a
parameter everywhere so tests inject a fake and never touch the network.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any
from urllib.parse import urlparse

from llm_agent_evals.errors import ConfigError, ModelError

Transport = Callable[[str, dict[str, str], dict[str, Any], float], tuple[int, Any]]


def ensure_http_url(url: str) -> None:
    """Only http and https are allowed. `urllib` also understands file: and ftp:, and a
    mistyped or hostile base URL must not turn into a local file read."""
    if urlparse(url).scheme not in ("http", "https"):
        raise ConfigError("LLM_BASE_URL must start with http:// or https://")


def urllib_transport(
    url: str, headers: dict[str, str], body: dict[str, Any], timeout: float
) -> tuple[int, Any]:
    """POST `body` as JSON. Returns the status and the parsed JSON (or raw text)."""
    ensure_http_url(url)
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, _parse(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, _parse(exc.read())
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise ModelError(f"network error while calling the model: {type(exc).__name__}") from exc


def _parse(raw: bytes) -> Any:
    text = raw.decode("utf-8", errors="replace")
    try:
        return json.loads(text)
    except ValueError:
        return text
