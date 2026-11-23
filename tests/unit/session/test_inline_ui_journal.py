"""Presentation provenance must survive replay without becoming a wire field."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from local_operator.harness.types import Message, TextContent
from local_operator.session.inline_ui_text import INLINE_UI_V1, project_inline_ui
from local_operator.session.transcript import Transcript

_SOURCE = 'Summary.\n\n```lop-ui\n{"type":"bar","title":"Revenue","data":[["January",6]]}\n```\n'


@pytest.mark.asyncio
async def test_journal_reopen_preserves_provenance_outside_strict_message_payload(
    tmp_path: Path,
) -> None:
    journal = Transcript(tmp_path / "session")
    marked = Message(role="assistant", content=[TextContent(text=_SOURCE)])
    marked._inline_ui_source_version = 1
    literal = Message(role="assistant", content=[TextContent(text=_SOURCE)])
    user = Message(role="user", content=[TextContent(text="Show a sample.")])
    user._inline_ui_source_version = 1  # never serialized for a non-assistant
    await journal.append_messages([user, marked, literal])

    reopened = Transcript(journal.directory)
    rows = {entry.id: json.loads(entry.to_json()) for entry in reopened.entries()}
    assert rows[marked.id]["inline_ui_source_version"] == 1
    assert "inline_ui_source_version" not in rows[literal.id]
    assert "inline_ui_source_version" not in rows[user.id]
    # The previous journal reader only copies these envelope keys and passes
    # the payload into the strict Message model. Its message must still load.
    payload = dict(rows[marked.id]["payload"])
    payload.pop("kind")
    assert Message.model_validate({**payload, "id": marked.id}).text == "Summary.\n\n\n"
    assert "lop-ui" not in json.dumps(payload)
    from local_operator.session.runtime.inline_ui_egress import journal_inline_ui

    envelope = journal_inline_ui(reopened.entries(), {marked.id, literal.id})
    assert envelope and list(envelope["messages"]) == [marked.id]
    assert envelope["messages"][marked.id]["text"] == "Summary.\n\n\n"

    history = {message.id: message for message in reopened.build_llm_history()}
    for original, version in [(marked, 1), (literal, None), (user, None)]:
        message = history[original.id]
        assert isinstance(message, Message)
        assert message._inline_ui_source_version == version
        assert message.model_dump() == original.model_dump()
        assert "inline_ui_source_version" not in message.model_dump_json()
    assert project_inline_ui(
        _SOURCE,
        streaming=False,
        source_version=history[marked.id]._inline_ui_source_version,
        capabilities=frozenset(),
    ) == {"text": "Summary.\n\n\n"}
    rich = project_inline_ui(
        _SOURCE,
        streaming=False,
        source_version=history[literal.id]._inline_ui_source_version,
        capabilities=frozenset({INLINE_UI_V1}),
    )
    assert rich == {"text": _SOURCE}
