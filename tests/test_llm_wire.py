"""The two HTTP adapters, tested through an injected transport. No live provider is called."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from itertools import pairwise

import pytest

from llm_agent_evals.errors import ConfigError, ModelError
from llm_agent_evals.llm import (
    AnthropicLLM,
    Message,
    ModelRequest,
    OpenAICompatibleLLM,
    RuleBasedLLM,
    ToolCall,
    build_llm,
    build_real_llm,
)
from llm_agent_evals.llm import anthropic as anthropic_wire
from llm_agent_evals.llm import openai_compat as openai_wire
from llm_agent_evals.llm.http import error_text, post_with_retry
from llm_agent_evals.llm.transport import ensure_http_url, urllib_transport
from llm_agent_evals.settings import LLMSettings

KEY = "sk-test-not-a-real-key"
SPEC = {
    "name": "get_thing",
    "description": "Get a thing.",
    "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
}


def settings(provider="openai_compatible", **kw):
    base = {
        "provider": provider,
        "api_key": KEY,
        "model": "m-1",
        "base_url": "http://localhost:9",
        "max_retries": 1,
        "timeout_seconds": 5.0,
    }
    base.update(kw)
    return LLMSettings(**base)


class FakeTransport:
    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def __call__(self, url, headers, body, timeout):
        self.calls.append({"url": url, "headers": headers, "body": body, "timeout": timeout})
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def request(**kw):
    messages = kw.pop("messages", (Message("user", "How is it going?"),))
    return ModelRequest(system="Be brief.", messages=messages, tools=(SPEC,), **kw)


OPENAI_OK = (
    200,
    {
        "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 7, "completion_tokens": 3},
    },
)


# -- OpenAI-style wire format -----------------------------------------------------------


def test_openai_request_carries_system_tools_and_sampling():
    body = openai_wire.to_wire(request(), settings())
    assert body["messages"][0] == {"role": "system", "content": "Be brief."}
    assert body["tools"][0]["function"]["parameters"] == SPEC["input_schema"]
    assert body["model"] == "m-1" and body["temperature"] == 0.2 and body["max_tokens"] == 2048


def test_openai_final_round_omits_the_tools_and_temperature_can_be_left_out():
    body = openai_wire.to_wire(request(allow_tools=False, temperature=None), settings())
    assert "tools" not in body and "temperature" not in body


def test_openai_assistant_tool_calls_and_tool_results_round_trip():
    call = ToolCall(id="c1", name="get_thing", args={"a": 1})
    messages = (
        Message("user", "q"),
        Message("assistant", "", (call,)),
        Message("tool", "{}", tool_call_id="c1"),
    )
    wire = openai_wire.to_wire(request(messages=messages), settings())["messages"]
    assert wire[2]["content"] is None
    assert json.loads(wire[2]["tool_calls"][0]["function"]["arguments"]) == {"a": 1}
    assert wire[3] == {"role": "tool", "tool_call_id": "c1", "content": "{}"}


def test_openai_response_with_tool_calls_and_usage():
    payload = {
        "choices": [
            {
                "message": {
                    "content": None,
                    "tool_calls": [
                        {"id": "c1", "function": {"name": "get_thing", "arguments": '{"x": 2}'}}
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2},
    }
    turn = openai_wire.parse_response(payload)
    assert turn.tool_calls == (ToolCall(id="c1", name="get_thing", args={"x": 2}),)
    assert turn.usage == {"input": 5, "output": 2} and turn.stop_reason == "tool_calls"


@pytest.mark.parametrize("arguments", ["{not json", "[1, 2]", '"text"'])
def test_openai_malformed_arguments_are_flagged_and_never_run_with_empty_input(arguments):
    payload = {
        "choices": [
            {
                "message": {
                    "tool_calls": [
                        {"id": "c1", "function": {"name": "get_thing", "arguments": arguments}}
                    ]
                }
            }
        ]
    }
    (call,) = openai_wire.parse_response(payload).tool_calls
    assert call.args == {} and call.args_error


@pytest.mark.parametrize("payload", [{}, {"choices": []}, {"choices": [{}]}, "oops", None])
def test_openai_unexpected_response_shapes_are_model_errors(payload):
    with pytest.raises(ModelError, match="unexpected response shape"):
        openai_wire.parse_response(payload)


def test_openai_client_posts_to_the_chat_completions_url_with_a_bearer_key():
    transport = FakeTransport(OPENAI_OK)
    turn = OpenAICompatibleLLM(settings(), transport).complete(request())
    call = transport.calls[0]
    assert call["url"] == "http://localhost:9/chat/completions"
    assert call["headers"]["Authorization"] == f"Bearer {KEY}" and call["timeout"] == 5.0
    assert turn.text == "hi" and turn.usage == {"input": 7, "output": 3}


# -- Anthropic wire format --------------------------------------------------------------------


def test_anthropic_merges_consecutive_user_items_so_roles_always_alternate():
    call = ToolCall(id="c1", name="get_thing", args={})
    messages = (
        Message("user", "q"),
        Message("assistant", "", (call,)),
        Message("tool", "{}", tool_call_id="c1"),
        Message("user", "answer now"),
    )
    wire = anthropic_wire.messages_to_wire(request(messages=messages))
    assert [m["role"] for m in wire] == ["user", "assistant", "user"]
    kinds = [block["type"] for block in wire[2]["content"]]
    assert kinds == ["tool_result", "text"]


def test_anthropic_puts_tool_results_before_text_in_a_merged_user_message():
    call = ToolCall(id="c1", name="get_thing", args={})
    messages = (
        Message("assistant", "", (call,)),
        Message("user", "note first"),
        Message("tool", "{}", tool_call_id="c1"),
    )
    wire = anthropic_wire.messages_to_wire(request(messages=messages))
    assert [b["type"] for b in wire[1]["content"]] == ["tool_result", "text"]


def test_anthropic_never_emits_two_consecutive_messages_with_the_same_role():
    call = ToolCall(id="c1", name="get_thing", args={})
    messages = (
        Message("user", "a"),
        Message("user", "b"),
        Message("assistant", "x", (call,)),
        Message("tool", "{}", tool_call_id="c1"),
        Message("tool", "{}", tool_call_id="c1"),
        Message("user", "c"),
    )
    roles = [m["role"] for m in anthropic_wire.messages_to_wire(request(messages=messages))]
    assert all(a != b for a, b in pairwise(roles))


def test_anthropic_final_round_keeps_the_tools_and_forbids_calling_them():
    body = anthropic_wire.to_wire(request(allow_tools=False), settings("anthropic"))
    assert body["tools"] and body["tool_choice"] == {"type": "none"}
    open_body = anthropic_wire.to_wire(request(), settings("anthropic"))
    assert "tool_choice" not in open_body and open_body["system"] == "Be brief."


def test_anthropic_response_text_and_tool_use_blocks():
    payload = {
        "content": [
            {"type": "text", "text": "a"},
            {"type": "text", "text": "b"},
            {"type": "tool_use", "id": "t1", "name": "get_thing", "input": {"k": 1}},
        ],
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 4, "output_tokens": 1},
    }
    turn = anthropic_wire.parse_response(payload)
    assert turn.text == "ab" and turn.tool_calls[0].args == {"k": 1}
    assert turn.usage == {"input": 4, "output": 1}


def test_anthropic_non_object_tool_input_is_flagged():
    payload = {"content": [{"type": "tool_use", "id": "t1", "name": "get_thing", "input": "x"}]}
    assert anthropic_wire.parse_response(payload).tool_calls[0].args_error


def test_anthropic_client_sends_its_key_and_version_headers():
    transport = FakeTransport((200, {"content": [{"type": "text", "text": "ok"}]}))
    AnthropicLLM(settings("anthropic", base_url="http://localhost:9"), transport).complete(
        request()
    )
    call = transport.calls[0]
    assert call["url"] == "http://localhost:9/v1/messages"
    assert call["headers"]["x-api-key"] == KEY and "anthropic-version" in call["headers"]


def test_anthropic_unexpected_shape_is_a_model_error():
    with pytest.raises(ModelError):
        anthropic_wire.parse_response({"nope": 1})


# -- retries and safe errors ----------------------------------------------------------------


def test_a_throttled_call_is_retried_with_backoff_then_succeeds():
    sleeps = []
    transport = FakeTransport((429, {"error": {"message": "slow down"}}), OPENAI_OK)
    turn = OpenAICompatibleLLM(settings(), transport, sleeps.append).complete(request())
    assert turn.text == "hi" and sleeps == [1.0] and len(transport.calls) == 2


@pytest.mark.parametrize("status", [408, 409, 500, 502, 503])
def test_transient_statuses_are_retried_until_the_limit(status):
    transport = FakeTransport((status, "x"), (status, "x"))
    with pytest.raises(ModelError, match=f"HTTP {status}"):
        OpenAICompatibleLLM(settings(), transport, lambda _s: None).complete(request())
    assert len(transport.calls) == 2


def test_a_client_error_is_not_retried():
    transport = FakeTransport((400, {"error": {"message": "bad request"}}))
    with pytest.raises(ModelError, match="HTTP 400: bad request"):
        OpenAICompatibleLLM(settings(), transport, lambda _s: None).complete(request())
    assert len(transport.calls) == 1


def test_network_failures_are_retried_too():
    transport = FakeTransport(ModelError("network error"), OPENAI_OK)
    turn = OpenAICompatibleLLM(settings(), transport, lambda _s: None).complete(request())
    assert turn.text == "hi"


def test_zero_retries_means_one_attempt():
    transport = FakeTransport((503, "x"))
    with pytest.raises(ModelError):
        OpenAICompatibleLLM(settings(max_retries=0), transport, lambda _s: None).complete(request())
    assert len(transport.calls) == 1


def test_backoff_doubles_and_is_capped():
    sleeps = []
    transport = FakeTransport(*[(503, "x")] * 6)
    with pytest.raises(ModelError):
        post_with_retry(transport, "http://x", {}, {}, settings(max_retries=5), sleeps.append)
    assert sleeps == [1.0, 2.0, 4.0, 8.0, 8.0]


def test_error_messages_never_contain_the_key_the_headers_or_the_request_body():
    transport = FakeTransport((401, {"error": {"message": "invalid key"}}))
    with pytest.raises(ModelError) as caught:
        OpenAICompatibleLLM(settings(), transport, lambda _s: None).complete(request())
    text = str(caught.value)
    assert KEY not in text and "Bearer" not in text and "How is it going?" not in text


def test_provider_error_text_is_short_and_safe():
    assert error_text({"error": {"message": "x" * 500}}) == "x" * 200
    assert error_text({"error": "plain"}) == "plain"
    assert error_text("  body  ") == "body"
    assert error_text({"other": 1}) == "no error detail"


# -- transport and factories ---------------------------------------------------------------------


@pytest.mark.parametrize("url", ["file:///etc/hosts", "ftp://example.invalid/x", "localhost:8000"])
def test_only_http_and_https_urls_are_allowed(url):
    with pytest.raises(ConfigError, match="http"):
        ensure_http_url(url)
    with pytest.raises(ConfigError):
        urllib_transport(url, {}, {}, 1.0)


class _Handler(BaseHTTPRequestHandler):
    seen = {}

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        _Handler.seen = {
            "body": json.loads(self.rfile.read(length)),
            "path": self.path,
            "auth": self.headers.get("Authorization"),
        }
        status = 429 if self.path.endswith("/limited") else 200
        payload = b"not json" if self.path.endswith("/plain") else b'{"ok": true}'
        self.send_response(status)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):  # silence
        pass


@pytest.fixture
def local_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def test_the_real_transport_posts_json_and_parses_the_reply(local_server):
    status, payload = urllib_transport(
        f"{local_server}/echo", {"Authorization": "Bearer x"}, {"a": 1}, 5.0
    )
    assert (status, payload) == (200, {"ok": True})
    assert _Handler.seen["body"] == {"a": 1} and _Handler.seen["auth"] == "Bearer x"


def test_the_real_transport_returns_error_statuses_and_raw_text(local_server):
    assert urllib_transport(f"{local_server}/limited", {}, {}, 5.0)[0] == 429
    assert urllib_transport(f"{local_server}/plain", {}, {}, 5.0) == (200, "not json")


def test_the_real_transport_reports_an_unreachable_host_as_a_model_error():
    with pytest.raises(ModelError, match="network error"):
        urllib_transport("http://127.0.0.1:1/x", {}, {}, 1.0)


def test_build_real_llm_picks_the_adapter_and_checks_the_configuration():
    assert isinstance(build_real_llm(settings("anthropic")), AnthropicLLM)
    assert isinstance(build_real_llm(settings("openai_compatible")), OpenAICompatibleLLM)
    with pytest.raises(ConfigError, match="mock"):
        build_real_llm(LLMSettings())
    with pytest.raises(ConfigError, match="LLM_API_KEY"):
        build_real_llm(settings("anthropic", api_key=""))


def test_build_llm_returns_the_offline_mock_by_default():
    assert isinstance(build_llm(LLMSettings()), RuleBasedLLM)
    assert isinstance(build_llm(settings("anthropic")), AnthropicLLM)
