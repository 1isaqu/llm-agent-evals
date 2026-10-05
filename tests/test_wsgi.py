import io
import json

from llm_agent_evals.service import ChatService
from llm_agent_evals.wsgi import MAX_BODY_BYTES, create_app
from tests.helpers import final, make_agent

VALID = {
    "messages": [{"role": "user", "content": "q"}],
    "product_line": "auto",
    "date_from": "2026-03-01",
    "date_to": "2026-03-18",
}


def call_app(app, method="POST", path="/chat", body=None, raw=None, length=None):
    payload = (
        raw if raw is not None else json.dumps(body).encode("utf-8") if body is not None else b""
    )
    environ = {
        "REQUEST_METHOD": method,
        "PATH_INFO": path,
        "CONTENT_LENGTH": str(len(payload) if length is None else length),
        "wsgi.input": io.BytesIO(payload),
    }
    captured = {}

    def start_response(status, headers):
        captured["status"], captured["headers"] = status, dict(headers)

    chunks = app(environ, start_response)
    return captured["status"], captured["headers"], json.loads(b"".join(chunks))


def app_with(script=None):
    service = ChatService(make_agent(script if script is not None else [final("Hello.")]))
    return create_app(service)


def test_health_endpoint():
    status, headers, body = call_app(app_with(), "GET", "/health")
    assert status == "200 OK" and body == {"status": "ok"}
    assert headers["Content-Type"].startswith("application/json")


def test_chat_returns_the_envelope_with_the_right_headers():
    status, headers, body = call_app(app_with(), body=VALID)
    assert status == "200 OK" and body["data"][0]["text"] == "Hello."
    assert headers["Cache-Control"] == "no-store"
    assert int(headers["Content-Length"]) > 0


def test_unknown_paths_and_wrong_methods_are_rejected():
    assert call_app(app_with(), "GET", "/nope")[0].startswith("404")
    status, _, body = call_app(app_with(), "GET", "/chat")
    assert status.startswith("405") and body["errors"][0]["code"] == "method_not_allowed"


def test_invalid_json_is_400():
    status, _, body = call_app(app_with(), raw=b"{not json")
    assert status.startswith("400") and body["errors"][0]["code"] == "invalid_json"
    assert call_app(app_with(), raw=b"\xff\xfe")[0].startswith("400")


def test_an_oversized_body_is_413_and_is_never_read():
    status, _, body = call_app(app_with(), raw=b"{}", length=MAX_BODY_BYTES + 1)
    assert status.startswith("413") and body["errors"][0]["code"] == "too_large"


def test_service_errors_keep_their_status_through_the_wsgi_layer():
    status, _, body = call_app(app_with(), body={**VALID, "product_line": "nowhere"})
    assert status.startswith("400") and body["errors"][0]["code"] == "invalid_request"
    status, _, _ = call_app(app_with([{"fail": "down"}]), body=VALID)
    assert status.startswith("502")


def test_an_empty_body_is_a_400_from_the_service_not_a_crash():
    status, _, _ = call_app(app_with(), raw=b"")
    assert status.startswith("400")
