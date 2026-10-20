"""The Langfuse observer against the real SDK and a local stand-in for the server.

Skipped unless the `langfuse` extra is installed. It proves that the calls the observer makes
exist in the installed SDK and that what the SDK exports carries no question text or data. It
does not prove anything about a real Langfuse server.
"""

import gzip
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

pytest.importorskip("langfuse")
trace_pb2 = pytest.importorskip("opentelemetry.proto.collector.trace.v1.trace_service_pb2")

from llm_agent_evals.runtime import build_agent  # noqa: E402
from llm_agent_evals.session import Session, Window  # noqa: E402

SESSION = Session("auto", Window.from_iso("2026-03-01", "2026-03-18"))


class _Receiver(BaseHTTPRequestHandler):
    bodies: list[bytes] = []

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        if self.headers.get("Content-Encoding") == "gzip":
            body = gzip.decompress(body)
        self.bodies.append(body)
        self.send_response(200)
        self.end_headers()

    def log_message(self, *args):
        pass


def _exported_spans(bodies):
    spans = []
    for body in bodies:
        request = trace_pb2.ExportTraceServiceRequest()
        request.ParseFromString(body)
        spans += [s for rs in request.resource_spans for ss in rs.scope_spans for s in ss.spans]
    return spans


def test_a_run_is_exported_by_the_real_sdk_as_one_trace_without_the_question(monkeypatch):
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    _Receiver.bodies = []
    server = HTTPServer(("127.0.0.1", 0), _Receiver)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    env = {
        "AGENT_TELEMETRY_LANGFUSE": "true",
        "LANGFUSE_PUBLIC_KEY": "pk-lf-local-test",
        "LANGFUSE_SECRET_KEY": "sk-lf-local-test",
        "LANGFUSE_BASE_URL": f"http://127.0.0.1:{server.server_port}",
    }
    agent = build_agent(env)
    question = "Which agents are the top performers?"
    agent.run([{"role": "user", "content": question}], SESSION, run_id="run-42")
    client = agent.observer._client
    client.flush()
    client.shutdown()
    server.shutdown()

    spans = _exported_spans(_Receiver.bodies)
    assert {s.name for s in spans} >= {"model_turn", "run_end", "list_agents_performance"}
    assert {s.trace_id.hex() for s in spans} == {client.create_trace_id(seed="run-42")}
    types = {
        a.value.string_value
        for s in spans
        for a in s.attributes
        if a.key == "langfuse.observation.type"
    }
    assert types == {"tool", "event"}
    dump = json.dumps([[(a.key, a.value.string_value) for a in s.attributes] for s in spans])
    assert "top performers" not in dump and "Agent 01" not in dump
