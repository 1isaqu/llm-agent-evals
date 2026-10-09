import json

import pytest

from llm_agent_evals.service import (
    MAX_MESSAGE_CHARS,
    MAX_MESSAGES,
    ChatService,
    RequestError,
    parse_messages,
)
from llm_agent_evals.settings import AgentSettings, LLMSettings
from tests.helpers import final, make_agent

VALID = {
    "messages": [{"role": "user", "content": "How is it going?"}],
    "product_line": "auto",
    "date_from": "2026-03-01",
    "date_to": "2026-03-18",
}


def service(script=None, **kw):
    agent = make_agent(script if script is not None else [final("All fine.")])
    return ChatService(agent, new_run_id=lambda: "run-test", **kw), agent


def test_a_valid_request_returns_the_envelope_with_one_answer():
    svc, _ = service()
    response = svc.handle(VALID)
    assert response.status == 200
    body = response.body
    assert body["errors"] == [] and body["meta"]["total_rows"] == 1
    assert body["meta"]["run_id"] == "run-test" and body["meta"]["sources"] == ["auto"]
    assert body["meta"]["filters"]["date_to"] == "2026-03-18"
    assert (
        body["data"][0]["text"] == "All fine." and body["data"][0]["reference_date"] == "2026-03-18"
    )
    assert "success" not in body and "success" not in body["meta"]


def test_dates_default_to_today_on_the_agents_clock():
    svc, _ = service()
    body = svc.handle({**VALID, "date_from": None, "date_to": ""}).body
    assert (
        body["meta"]["filters"]["date_from"] == body["meta"]["filters"]["date_to"] == "2026-03-18"
    )


def test_all_lines_lists_every_source():
    agent = make_agent([final()], consumer={"status_rollup": []})
    svc = ChatService(agent)
    body = svc.handle({**VALID, "product_line": "all"}).body
    assert body["meta"]["sources"] == ["auto", "consumer"]


def test_a_disabled_agent_is_404_and_the_model_is_not_called():
    svc, agent = service(agent_settings=AgentSettings(chat_enabled=False))
    response = svc.handle(VALID)
    assert response.status == 404 and response.body["errors"][0]["code"] == "chat_disabled"
    assert agent.llm.requests == []


def test_a_real_provider_without_a_key_is_503_and_the_message_names_only_the_variable():
    settings = LLMSettings(provider="anthropic", model="m")
    svc, _ = service(llm_settings=settings)
    response = svc.handle(VALID)
    assert response.status == 503 and "LLM_API_KEY" in response.body["errors"][0]["message"]


@pytest.mark.parametrize(
    ("patch", "fragment"),
    [
        ({"messages": []}, "non-empty list"),
        ({"messages": "hi"}, "non-empty list"),
        ({"messages": [{"role": "system", "content": "x"}]}, "role"),
        ({"messages": [{"role": "user", "content": "  "}]}, "non-empty text"),
        ({"messages": [{"role": "user", "content": 5}]}, "non-empty text"),
        (
            {"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]},
            "last message",
        ),
        ({"product_line": "nowhere"}, "product_line must be one of"),
        ({"product_line": None}, "product_line must be one of"),
        ({"date_from": "03/01/2026"}, "date_from is not valid"),
        ({"date_to": "tomorrow"}, "date_to is not valid"),
        ({"date_from": "2026-03-18", "date_to": "2026-03-01"}, "must not be after"),
    ],
)
def test_bad_requests_are_400_with_a_precise_message(patch, fragment):
    svc, agent = service()
    response = svc.handle({**VALID, **patch})
    assert response.status == 400 and response.body["errors"][0]["code"] == "invalid_request"
    assert fragment in response.body["errors"][0]["message"]
    assert agent.llm.requests == []


@pytest.mark.parametrize("body", [None, [], "text", 5])
def test_a_body_that_is_not_an_object_is_400(body):
    assert service()[0].handle(body).status == 400


def test_a_model_failure_is_502_with_a_generic_message_and_no_provider_text():
    svc, _ = service([{"fail": "internal key=SECRET-VALUE rejected"}])
    response = svc.handle(VALID)
    assert response.status == 502
    assert "SECRET-VALUE" not in json.dumps(response.body)


def test_an_unexpected_bug_is_500_with_no_detail():
    svc, agent = service()

    def explode(*args, **kwargs):
        raise RuntimeError("secret internal detail")

    agent.run = explode
    response = svc.handle(VALID)
    assert response.status == 500 and "secret" not in json.dumps(response.body)


def test_long_conversations_are_trimmed_to_the_last_messages_and_each_is_cut():
    history = []
    for i in range(MAX_MESSAGES + 6):
        history += [
            {"role": "user", "content": f"q{i} " + "x" * (MAX_MESSAGE_CHARS + 50)},
            {"role": "assistant", "content": f"a{i}"},
        ]
    history.append({"role": "user", "content": "last"})
    messages = parse_messages(history)
    assert len(messages) <= MAX_MESSAGES and messages[-1].content == "last"
    assert all(len(m.content) <= MAX_MESSAGE_CHARS for m in messages)


def test_a_trimmed_conversation_never_starts_with_an_assistant_turn():
    # Alternating turns that end on a user turn always cut onto an assistant turn.
    alternating = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}"}
        for i in range(MAX_MESSAGES + 1)
    ]
    trimmed = parse_messages(alternating)
    assert trimmed[0].role == "user" and trimmed[-1].content == f"m{MAX_MESSAGES}"
    assert len(trimmed) == MAX_MESSAGES - 1 and trimmed[0].content == "m2"
    only_user = [{"role": "user", "content": f"m{i}"} for i in range(MAX_MESSAGES + 1)]
    assert len(parse_messages(only_user)) == MAX_MESSAGES


def test_parse_messages_strips_whitespace():
    assert parse_messages([{"role": "user", "content": "  hi  "}])[0].content == "hi"
    with pytest.raises(RequestError):
        parse_messages([{"role": "user"}])


def test_every_request_gets_a_fresh_run_id_by_default():
    agent = make_agent([final(), final()])
    svc = ChatService(agent)
    ids = {svc.handle(VALID).body["meta"]["run_id"] for _ in range(2)}
    assert len(ids) == 2 and all(i.startswith("run-") for i in ids)
