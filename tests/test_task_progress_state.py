"""项目自增的任务状态回归测试；不调用真实模型。"""

import json

import pytest

from cs329z_hw1 import adapters
from cs329z_hw1.agent.task_progress import TaskProgress
from cs329z_hw1.llm import ScriptedLM
from cs329z_hw1.tokens import count_message_tokens
from cs329z_hw1.types import AgentConfig, ToolCall, ToolResult, ToolSpec


def doc_result(start=1, end=10, *, text="Policy text.", doc_id="policy"):
    return json.dumps({
        "doc_id": doc_id, "start_line": start, "end_line": end,
        "total_lines": 100, "text": text,
    })


def read_call(start=1):
    return ToolCall("read_doc", {"doc_id": "policy", "start_line": start})


def scripted_call(start=1):
    call = read_call(start)
    return {"tool_call": {"name": call.name, "args": call.args}}


def doc_tool(fn):
    return ToolSpec(
        name="read_doc", description="Read a document window.", read_only=True,
        parameters={
            "type": "object",
            "properties": {"doc_id": {"type": "string"}, "start_line": {"type": "integer"}},
            "required": ["doc_id", "start_line"], "additionalProperties": False,
        },
        fn=fn,
    )


def test_search_records_returned_sources_without_marking_them_read():
    progress = TaskProgress(task_id=7)
    call = ToolCall("search_docs", {"query": "approval", "k": 2})
    content = json.dumps({"results": [{
        "doc_id": "policy", "snippet": "Approval policy.", "start_line": 4, "end_line": 9,
    }]})
    evidence_id = progress.record(call, ToolResult("ok", content))
    assert evidence_id == "task-7-e1"
    assert progress.searches[0]["query"] == "approval"
    assert progress.searches[0]["k"] == 2
    assert progress.get_evidence(evidence_id)["source_refs"] == [{
        "kind": "doc_snippet", "doc_id": "policy", "start_line": 4, "end_line": 9,
    }]
    assert progress.read_ids == []
    assert progress.read_windows == []


def test_pagination_and_repeated_windows_keep_distinct_observations():
    progress = TaskProgress()
    first = progress.record(read_call(), ToolResult("ok", doc_result()))
    second = progress.record(read_call(11), ToolResult("ok", doc_result(11, 20)))
    repeated = progress.record(read_call(), ToolResult("ok", doc_result()))
    assert progress.read_ids == ["policy"]
    assert progress.stage_reads == 1
    windows = progress.read_windows
    assert [(w["start_line"], w["end_line"]) for w in windows] == [(1, 10), (11, 20)]
    assert windows[0]["evidence_ids"] == [first, repeated]
    assert windows[1]["evidence_ids"] == [second]
    assert len(progress.evidence) == progress.tool_count == 3


@pytest.mark.parametrize("status", ["unknown_tool", "invalid_args", "denied", "tool_error"])
def test_failure_is_recorded_but_never_marks_a_window_read(status):
    progress = TaskProgress()
    evidence_id = progress.record(read_call(), ToolResult(status, "Failed or refused."))
    assert evidence_id is None
    assert progress.attempts[0]["status"] == status
    assert progress.attempts[0]["args"] == read_call().args
    assert progress.status_counts[status] == 1
    assert progress.read_windows == []
    assert progress.evidence == {}


def test_failed_search_and_successful_retry_are_both_retained():
    progress = TaskProgress()
    call = ToolCall("search_docs", {"query": "approval"})
    progress.record(call, ToolResult("tool_error", "Read failed."))
    progress.record(call, ToolResult("ok", '{"results": []}'))
    assert [s["status"] for s in progress.searches] == ["tool_error", "ok"]
    assert progress.searches[0]["evidence_id"] is None
    assert progress.searches[1]["evidence_id"] in progress.evidence
    assert progress.queries == ["approval"]


@pytest.mark.parametrize("content", [
    "not JSON",
    "[]",
    doc_result(doc_id="a different document"),
    doc_result(2, 10),
    doc_result(1, 101),
    doc_result(True, 10),
])
def test_invalid_or_mismatched_read_metadata_is_not_invented_from_args(content):
    progress = TaskProgress()
    evidence_id = progress.record(read_call(), ToolResult("ok", content))
    assert progress.get_evidence(evidence_id)["source_refs"] == []
    assert progress.read_windows == []
    assert progress.read_ids == []


def test_email_reads_are_distinguished_from_email_search_snippets():
    progress = TaskProgress()
    progress.record(
        ToolCall("search_emails", {"query": "billing"}),
        ToolResult("ok", '{"results": [{"id": "em-1", "snippet": "Billing."}]}'),
    )
    assert progress.read_ids == []
    evidence_id = progress.record(
        ToolCall("read_email", {"email_id": "em-1"}),
        ToolResult("ok", '{"id": "em-1", "body": "Full email."}'),
    )
    assert progress.read_ids == ["em-1"]
    assert progress.get_evidence(evidence_id)["source_refs"] == [
        {"kind": "email", "email_id": "em-1"}
    ]


def test_recorded_arguments_and_returned_snapshots_are_independent():
    progress = TaskProgress()
    call = ToolCall("echo", {"nested": {"value": "original"}})
    evidence_id = progress.record(call, ToolResult("ok", "Original result."))
    call.args["nested"]["value"] = "mutated"
    snapshot = progress.snapshot(include_content=True)
    snapshot["evidence"][evidence_id]["args"]["nested"]["value"] = "also mutated"
    evidence = progress.get_evidence(evidence_id)
    evidence["args"]["nested"]["value"] = "another mutation"
    assert progress.get_evidence(evidence_id)["args"]["nested"]["value"] == "original"
    assert progress.attempts[0]["args"]["nested"]["value"] == "original"
    with pytest.raises(KeyError):
        progress.get_evidence("invented-id")


@pytest.mark.parametrize("max_chars", [256, 1000, 2000, 2400])
def test_prompt_is_bounded_and_excludes_source_text_and_failure_details(max_chars):
    progress = TaskProgress()
    for i in range(50):
        progress.record(
            ToolCall("search_docs", {"query": ('"\\\n' + str(i)) * 1000}),
            ToolResult("ok", '{"results": []}'),
        )
    progress.record(read_call(), ToolResult("ok", doc_result(text="PRIVATE_SOURCE_TEXT")))
    progress.record(read_call(11), ToolResult("tool_error", "UNTRUSTED_FAILURE_TEXT"))
    text = progress.instruction(remaining=3, max_chars=max_chars)
    assert len(text) <= max_chars
    assert "PRIVATE_SOURCE_TEXT" not in text
    assert "UNTRUSTED_FAILURE_TEXT" not in text
    data, _ = json.JSONDecoder().raw_decode(text[text.index("{"):])
    assert data["details_omitted"] is True
    assert len(progress.searches) == 50
    assert progress.tool_count == 52


def test_compaction_keeps_full_task_state_without_extra_agent_calls():
    count = 16
    raw = {}

    def read(doc_id, start_line):
        raw[start_line] = doc_result(start_line, start_line, text="long evidence " * 600)
        return raw[start_line]

    lm = ScriptedLM(
        [scripted_call(i) for i in range(1, count + 1)] + ["Finished."],
        formatter=adapters.run_format_tool_call,
    )
    aux = ScriptedLM(["Earlier history summarized."], repeat_last=True)
    session = adapters.run_agent_session(
        lm, [doc_tool(read)],
        AgentConfig(context_budget=4000, max_turns=count + 1, aux_lm=aux),
    )
    result = session.send("Read the requested document windows.")
    assert result.status == "done"
    assert len(lm.calls) == count + 1
    assert aux.calls, "The fixture must actually exercise compaction."
    assert all(count_message_tokens(messages) <= 4000 for messages in lm.calls)
    assert len(session.task_progress.read_windows) == count
    evidence = session.task_progress.get_evidence("task-1-e1")
    assert evidence["content"] == raw[1]
    assert evidence["truncated"] is True
    events = [e for e in result.transcript if e["type"] == "tool_result"]
    assert events[0]["content"] == evidence["visible_content"]
    assert events[0]["evidence_id"] == evidence["id"]
    assert evidence["id"] in lm.calls[1][-1]["content"]
    assert evidence["visible_content"] in lm.calls[1][-1]["content"]
    assert len(session.task_progress.snapshot()["evidence"]) == count


def test_continue_preserves_task_and_new_request_resets_without_reusing_ids():
    lm = ScriptedLM(
        [scripted_call(), "Done.", scripted_call(11), "Done.", scripted_call(), "Done."],
        formatter=adapters.run_format_tool_call,
    )
    session = adapters.run_agent_session(
        lm, [doc_tool(lambda doc_id, start_line: doc_result(start_line, start_line))],
        AgentConfig(),
    )
    session.send("Read a window.")
    original = session.task_progress
    snapshot = original.snapshot()
    session.send("继续")
    assert session.task_progress is original
    assert len(original.read_windows) == 2
    assert len(snapshot["read_windows"]) == 1
    session.send("Start a different task.")
    assert session.task_progress is not original
    assert len(session.task_progress.read_windows) == 1
    assert set(session.task_progress.evidence) == {"task-3-e1"}
    assert set(original.evidence) == {"task-1-e1", "task-1-e2"}


def test_session_captures_arguments_before_tool_mutates_them():
    def mutate(payload):
        payload["value"] = "changed by tool"
        return "Done."

    tool = ToolSpec("mutate", "Mutate input.", {"type": "object"}, mutate, read_only=True)
    lm = ScriptedLM([
        {"tool_call": {"name": "mutate", "args": {"payload": {"value": "original"}}}},
        "Done.",
    ], formatter=adapters.run_format_tool_call)
    session = adapters.run_agent_session(lm, [tool], AgentConfig())
    session.send("Run the tool.")
    evidence = session.task_progress.get_evidence("task-1-e1")
    assert evidence["args"] == {"payload": {"value": "original"}}
