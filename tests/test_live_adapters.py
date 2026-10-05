"""Both HTTP adapters driven through the real `urllib` transport by a local fake provider.

The fake server speaks the two wire formats as the adapters expect them. This proves the
plumbing (URL, headers, JSON bodies, tool-call round trip, final round) end to end without a
network or a key. It does not prove that a real provider accepts these requests.
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from llm_agent_evals.agent import Agent
from llm_agent_evals.llm import AnthropicLLM, OpenAICompatibleLLM
from llm_agent_evals.settings import LLMSettings
from tests.helpers import NOW, SESSION, fixture_sources

KEY = "sk-test-not-a-real-key"
ANSWER = {
    "text": "Generated value is **$700.00**.",
    "highlights": [],
    "suggested_actions": [],
    "data_sources": ["get_status_breakdown"],
    "confidence": "high",
}


def _openai_reply(body):
    if any(m.get("role") == "tool" for m in body["messages"]):
        message = {"role": "assistant", "content": json.dumps(ANSWER)}
        return {"choices": [{"message": message, "finish_reason": "stop"}]}
    call = {
        "id": "call_1",
        "type": "function",
        "function": {"name": "get_status_breakdown", "arguments": "{}"},
    }
    message = {"role": "assistant", "content": None, "tool_calls": [call]}
    return {"choices": [{"message": message, "finish_reason": "tool_calls"}]}


def _anthropic_reply(body):
    has_result = any(
        isinstance(m["content"], list) and any(b.get("type") == "tool_result" for b in m["content"])
        for m in body["messages"]
    )
    if has_result:
        return {
            "content": [{"type": "text", "text": json.dumps(ANSWER)}],
            "stop_reason": "end_turn",
        }
    block = {"type": "tool_use", "id": "tu_1", "name": "get_status_breakdown", "input": {}}
    return {"content": [block], "stop_reason": "tool_use"}


class _Provider(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        headers = {name.lower(): value for name, value in self.headers.items()}
        self.server.seen.append({"path": self.path, "headers": headers, "body": body})
        reply = (
            _anthropic_reply(body) if self.path.endswith("/v1/messages") else _openai_reply(body)
        )
        payload = json.dumps(reply).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


@pytest.fixture
def provider():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Provider)
    server.seen = []
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield server
    server.shutdown()
    server.server_close()


def run_agent(client):
    agent = Agent(client, fixture_sources(), now=lambda: NOW)
    return agent.run([{"role": "user", "content": "What is the generated value?"}], SESSION)


def settings(provider_name, server):
    base = f"http://127.0.0.1:{server.server_address[1]}"
    url = f"{base}/v1" if provider_name == "openai_compatible" else base
    return LLMSettings(
        provider=provider_name,
        api_key=KEY,
        model="m-1",
        base_url=url,
        timeout_seconds=5.0,
        max_retries=0,
    )


def test_openai_style_round_trip_through_a_tool_call(provider):
    client = OpenAICompatibleLLM(settings("openai_compatible", provider))
    result = run_agent(client)
    assert [c.name for c in result.calls] == ["get_status_breakdown"]
    assert result.response["text"] == "Generated value is **$700.00**." and result.format_ok
    first, second = provider.seen
    assert first["path"] == "/v1/chat/completions"
    assert first["headers"]["authorization"] == f"Bearer {KEY}"
    assert len(first["body"]["tools"]) == 15 and first["body"]["messages"][0]["role"] == "system"
    assert second["body"]["messages"][-1]["role"] == "tool"
    tool_result = json.loads(second["body"]["messages"][-1]["content"])
    assert tool_result["generated_amount"] == 700.0


def test_anthropic_style_round_trip_through_a_tool_call(provider):
    client = AnthropicLLM(settings("anthropic", provider))
    result = run_agent(client)
    assert [c.name for c in result.calls] == ["get_status_breakdown"]
    assert result.response["confidence"] == "high"
    first, second = provider.seen
    assert first["path"] == "/v1/messages" and first["headers"]["x-api-key"] == KEY
    roles = [m["role"] for m in second["body"]["messages"]]
    assert roles == ["user", "assistant", "user"]
    assert second["body"]["messages"][-1]["content"][0]["type"] == "tool_result"


def test_the_key_appears_only_in_the_auth_header_never_in_a_request_body(provider):
    run_agent(OpenAICompatibleLLM(settings("openai_compatible", provider)))
    run_agent(AnthropicLLM(settings("anthropic", provider)))
    for seen in provider.seen:
        assert KEY not in json.dumps(seen["body"])
