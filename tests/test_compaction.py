from llm_agent_evals.compaction import ARGS_CHARS, DIGEST_CHARS, compact_messages, tool_groups
from llm_agent_evals.llm import Message, ToolCall


def group(n: int, result: str = '{"data": [1]}'):
    call = ToolCall(id=f"c{n}", name=f"tool_{n}", args={"n": n})
    return [Message("assistant", "", (call,)), Message("tool", result, tool_call_id=f"c{n}")]


def convo(groups: int, **kw):
    messages = [Message("user", "How are we doing?")]
    for n in range(groups):
        messages.extend(group(n, **kw))
    return messages


def assert_no_orphans(messages):
    ids = {c.id for m in messages for c in m.tool_calls}
    answered = {m.tool_call_id for m in messages if m.role == "tool"}
    assert ids == answered


def test_groups_pair_each_assistant_turn_with_its_tool_messages():
    messages = convo(3)
    assert tool_groups(messages) == [(1, 3), (3, 5), (5, 7)]


def test_a_batch_turn_forms_one_group_with_all_its_results():
    calls = tuple(ToolCall(id=f"b{i}", name="a_tool", args={}) for i in range(2))
    messages = [
        Message("user", "q"),
        Message("assistant", "", calls),
        Message("tool", "{}", tool_call_id="b0"),
        Message("tool", "{}", tool_call_id="b1"),
    ]
    assert tool_groups(messages) == [(1, 4)]


def test_short_conversations_are_left_alone():
    messages = convo(1)
    assert compact_messages(messages) == messages


def test_old_groups_are_replaced_by_one_note_and_the_newest_group_is_kept_intact():
    messages = convo(4)
    compacted = compact_messages(messages)
    assert compacted[0] == messages[0]
    assert compacted[1].role == "user" and "Automatic summary" in compacted[1].content
    assert compacted[2:] == messages[-2:]
    assert_no_orphans(compacted)


def test_the_note_names_every_removed_call_so_the_model_does_not_repeat_it():
    note = compact_messages(convo(4))[1].content
    for n in range(3):
        assert f"tool_{n}" in note
    assert "tool_3" not in note


def test_a_digest_of_each_removed_result_is_kept_but_truncated():
    long_result = '{"data": "' + "z" * 1000 + '"}'
    note = compact_messages(convo(3, result=long_result))[1].content
    assert note.count("z") < 2 * DIGEST_CHARS + 10
    assert "->" in note


def test_long_arguments_are_truncated_in_the_note():
    big = ToolCall(id="c0", name="tool_0", args={"text": "q" * 500})
    messages = [
        Message("user", "q"),
        Message("assistant", "", (big,)),
        Message("tool", "{}", tool_call_id="c0"),
        *group(1),
        *group(2),
    ]
    note = compact_messages(messages)[1].content
    assert note.count("q") <= ARGS_CHARS


def test_keeping_more_groups_removes_fewer():
    compacted = compact_messages(convo(5), keep_recent_groups=3)
    assert len(tool_groups(compacted)) == 3
    assert_no_orphans(compacted)


def test_user_turns_between_groups_are_preserved_in_order():
    messages = [*convo(1), Message("user", "follow-up"), *group(1), *group(2)]
    compacted = compact_messages(messages)
    assert [m.content for m in compacted if m.role == "user" and "summary" not in m.content] == [
        "How are we doing?",
        "follow-up",
    ]
