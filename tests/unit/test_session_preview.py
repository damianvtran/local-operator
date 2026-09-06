"""The conversation list's last-reply preview, read from the canonical transcript.

Design D19: a sidebar row for a session with a user message, an assistant reply
and a completed tool turn read "No messages yet". It rendered the legacy agent
record's `last_message`, which only the legacy execution path writes, so every
canonical session -- where the conversation actually lives in
`transcript.jsonl` -- was described as empty while displaying the timestamp of
the message it was denying.
"""

from __future__ import annotations

import json
from pathlib import Path

from local_operator.resume import PREVIEW_MAX_CHARS, PREVIEW_SCAN_BYTES, session_preview


def write(session: Path, entries: list[dict[str, object]]) -> None:
    session.mkdir(parents=True, exist_ok=True)
    (session / "transcript.jsonl").write_text(
        "".join(json.dumps(entry) + "\n" for entry in entries), encoding="utf-8"
    )


def message(role: str, text: str, **payload: object) -> dict[str, object]:
    return {
        "id": f"{role}-{abs(hash(text)) % 10**8}",
        "ts": 1.0,
        "type": "message",
        "payload": {"kind": "message", "role": role, "content": [{"text": text}], **payload},
    }


def test_preview_is_the_last_assistant_reply(tmp_path: Path) -> None:
    session = tmp_path / "sessions" / "aaaaaaaaaaaa"
    write(
        session,
        [
            message("user", "first question"),
            message("assistant", "an earlier reply"),
            message("user", "second question"),
            message("assistant", "the most recent reply"),
        ],
    )
    assert session_preview(session) == "the most recent reply"


def test_a_tool_only_turn_does_not_blank_the_row(tmp_path: Path) -> None:
    """The row shows the last thing the model SAID, not the last record it wrote.

    An assistant entry carrying only tool calls has no text; previewing it would
    replace a real sentence with an empty string, which renders exactly like the
    "No messages yet" this fix exists to remove.
    """
    session = tmp_path / "sessions" / "bbbbbbbbbbbb"
    write(
        session,
        [
            message("user", "do the thing"),
            message("assistant", "Working on it."),
            message("assistant", "", tool_calls=[{"id": "call_1", "name": "todo"}]),
            message("tool", "Todo list initialized.", tool_call_id="call_1"),
        ],
    )
    assert session_preview(session) == "Working on it."


def test_a_tool_result_is_never_mistaken_for_a_reply(tmp_path: Path) -> None:
    """`role` is matched exactly: a tool result would preview a directory listing."""
    session = tmp_path / "sessions" / "cccccccccccc"
    write(
        session,
        [
            message("assistant", "Reading the directory."),
            message("tool", "file-a.py\nfile-b.py\nfile-c.py", tool_call_id="call_1"),
        ],
    )
    assert session_preview(session) == "Reading the directory."


def test_preview_is_condensed_and_bounded(tmp_path: Path) -> None:
    session = tmp_path / "sessions" / "dddddddddddd"
    write(session, [message("assistant", "line one\nline two   with     runs")])
    assert session_preview(session) == "line one line two with runs"

    session = tmp_path / "sessions" / "eeeeeeeeeeee"
    write(session, [message("assistant", "z" * 5_000)])
    assert len(session_preview(session)) <= PREVIEW_MAX_CHARS


def test_a_huge_transcript_reads_only_its_tail(tmp_path: Path) -> None:
    """The scan is bounded, and the bound does not cost the answer.

    A conversation list paints one row per session; a pasted file or a base64
    image in one entry must not turn that into reading megabytes per row.
    """
    session = tmp_path / "sessions" / "ffffffffffff"
    write(
        session,
        [
            message("assistant", "an ancient reply nobody should see"),
            message("user", "x" * (PREVIEW_SCAN_BYTES * 3)),
            message("assistant", "the newest reply"),
        ],
    )
    assert (session / "transcript.jsonl").stat().st_size > PREVIEW_SCAN_BYTES
    assert session_preview(session) == "the newest reply"


def test_missing_or_unparseable_transcripts_yield_no_preview(tmp_path: Path) -> None:
    """Tolerant by design: a list must paint, and the caller owns the empty state."""
    missing = tmp_path / "sessions" / "111111111111"
    missing.mkdir(parents=True)
    assert session_preview(missing) == ""

    truncated = tmp_path / "sessions" / "222222222222"
    truncated.mkdir(parents=True)
    (truncated / "transcript.jsonl").write_text('{"type": "mess', encoding="utf-8")
    assert session_preview(truncated) == ""

    # A half-written final line is normal for a session being written right now;
    # the previous complete reply is still the honest answer.
    live = tmp_path / "sessions" / "333333333333"
    write(live, [message("assistant", "the settled reply")])
    with (live / "transcript.jsonl").open("a", encoding="utf-8") as handle:
        handle.write('{"id": "half", "type": "mess')
    assert session_preview(live) == "the settled reply"
