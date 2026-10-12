"""Prepared docs publish through the real runner/store even when classifiers are unavailable."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from local_operator.harness.types import Message, TextContent, ToolCall
from local_operator.session.attachments import AttachmentStore
from local_operator.supplements import policy
from local_operator.supplements.trigger import snapshot_messages
from tests.unit.supplements.test_prepared import DOC
from tests.unit.supplements.test_runner_generator import _handle, _make_session, _rows


@pytest.mark.asyncio
@pytest.mark.parametrize("mixed", [False, True])
async def test_prepared_document_reaches_durable_attachment_and_live_event(
    tmp_path: Path, mixed: bool
) -> None:
    target = tmp_path / "workflow.html"
    target.write_text(DOC)
    session = _make_session(tmp_path / "session", tmp_path)
    handle = await _handle(session, tmp_path)
    events = []
    session.subscribe(events.append)
    answer = Message.assistant(f"[Workflow]({target})")
    items = snapshot_messages(
        [
            Message.assistant(
                tool_calls=[
                    ToolCall(id="w", name="write", arguments={"path": str(target), "content": DOC})
                ]
            ),
            Message(
                role="tool",
                tool_call_id="w",
                tool_name="write",
                content=[TextContent(text="written")],
            ),
            answer,
        ],
        prepared_writes={str(target): DOC},
    )
    if mixed:
        report = tmp_path / "notes.md"
        report.write_text("Supporting report")
        items[1:1] = snapshot_messages(
            [
                Message.assistant(
                    tool_calls=[ToolCall(id="edit", name="edit", arguments={"path": str(report)})]
                ),
                Message(role="tool", tool_call_id="edit", tool_name="edit"),
            ]
        )
    try:
        await session.transcript.append_message(answer)
        await handle._supplements._job(
            SimpleNamespace(items=items, user_text="Create an interactive workflow document"),
            policy.SupplementSettings.from_values({"supplements": {"graphics": True}}),
        )
        rows = _rows(session.transcript)
        assert len(rows) == 1 and rows[0]["state"] == "done"
        assert rows[0]["files"] and rows[0]["components"]
        assert {item["path"] for item in rows[0]["files"]} == (
            {"workflow.html", "notes.md"} if mixed else {"workflow.html"}
        )
        component = rows[0]["components"][0]
        blob = AttachmentStore().get_bytes(component["attachment"])
        assert blob is not None
        assert b"onclick" in blob[0] and blob[1] == "text/html"
        assert any(event.type == "supplement_progress" and event.components for event in events)
    finally:
        await handle.dispose()
