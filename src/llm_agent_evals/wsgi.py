"""A stdlib WSGI application around `ChatService`.

Routes: `GET /health` and `POST /chat`. Nothing else is served. The layer is intentionally
thin and has no authentication, no rate limiting and no TLS: bind it to localhost, or put
it behind a proxy that provides those. It exists so the service can run and be tested with
no web framework installed.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping
from typing import Any
from wsgiref.simple_server import make_server

from llm_agent_evals.service import ChatService

MAX_BODY_BYTES = 64 * 1024
_STATUS_TEXT = {
    200: "200 OK",
    400: "400 Bad Request",
    404: "404 Not Found",
    405: "405 Method Not Allowed",
    413: "413 Payload Too Large",
    500: "500 Internal Server Error",
    502: "502 Bad Gateway",
    503: "503 Service Unavailable",
}

StartResponse = Callable[[str, list[tuple[str, str]]], Any]


def _reply(start_response: StartResponse, status: int, body: Mapping[str, Any]) -> list[bytes]:
    payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
    headers = [
        ("Content-Type", "application/json; charset=utf-8"),
        ("Content-Length", str(len(payload))),
        ("Cache-Control", "no-store"),
    ]
    start_response(_STATUS_TEXT.get(status, f"{status} Error"), headers)
    return [payload]


def _problem(code: str, message: str) -> dict[str, Any]:
    return {"meta": {}, "data": [], "errors": [{"code": code, "message": message}]}


def _read_body(environ: Mapping[str, Any]) -> bytes | None:
    """The request body, or None when it is larger than `MAX_BODY_BYTES`."""
    try:
        length = int(environ.get("CONTENT_LENGTH") or 0)
    except ValueError:
        length = 0
    if length > MAX_BODY_BYTES:
        return None
    return environ["wsgi.input"].read(length) if length > 0 else b""


def create_app(
    service: ChatService,
) -> Callable[[Mapping[str, Any], StartResponse], Iterable[bytes]]:
    """Build the WSGI callable."""

    def app(environ: Mapping[str, Any], start_response: StartResponse) -> Iterable[bytes]:
        path, method = environ.get("PATH_INFO", ""), environ.get("REQUEST_METHOD", "GET")
        if path == "/health":
            return _reply(start_response, 200, {"status": "ok"})
        if path != "/chat":
            return _reply(start_response, 404, _problem("not_found", "Unknown path."))
        if method != "POST":
            return _reply(start_response, 405, _problem("method_not_allowed", "Use POST."))
        raw = _read_body(environ)
        if raw is None:
            return _reply(start_response, 413, _problem("too_large", "Request body too large."))
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return _reply(start_response, 400, _problem("invalid_json", "Body is not valid JSON."))
        outcome = service.handle(payload)
        return _reply(start_response, outcome.status, outcome.body)

    return app


def serve(service: ChatService, host: str = "127.0.0.1", port: int = 8000) -> None:
    """Run the app on the stdlib development server until interrupted."""
    with make_server(host, port, create_app(service)) as server:
        server.serve_forever()
