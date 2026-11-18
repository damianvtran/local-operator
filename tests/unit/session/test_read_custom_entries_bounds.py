"""``read_custom_entries``: the message-row shape and the ``min_ts`` walk bound.

WHY THIS FILE EXISTS. The Team Bridge reconcile repair (H5-F3-4) asked this
reader for ``team_bridge_message`` rows, which are only ever persisted as
``message`` rows (payload kind ``custom``), while the reader accepted only
``custom`` entries. Every needle hit was decoded and rejected, ``max_rows`` was
never reached, and the walk read the whole journal on every turn start and ask
tick — 1.07 s warm on a real 688 MB journal, on the event loop, returning
nothing. These tests pin the two options that fix it: ``custom_messages``
(match the row shape the writer actually produces) and ``min_ts`` (stop the
walk once rows are older than the caller's window).
"""

from __future__ import annotations

import json
import time

import pytest

from local_operator.harness.types import CustomMessage
from local_operator.session import transcript as transcript_mod
from local_operator.session.transcript import (
    ENTRY_CUSTOM,
    ENTRY_MESSAGE,
    Transcript,
    TranscriptEntry,
    read_custom_entries,
)

TYPE = "team_bridge_message"


@pytest.mark.asyncio
async def test_message_rows_are_found_only_when_asked_for(tmp_path) -> None:
    """The REAL writer's shape: a CustomMessage lands as a ``message`` row."""

    transcript = Transcript(tmp_path / "s")
    await transcript.append_message(
        CustomMessage(custom_type=TYPE, attribution="user", details={"body": "one"})
    )
    await transcript.append_custom(TYPE, {"body": "custom-entry"})
    await transcript.append_message(
        CustomMessage(custom_type=TYPE, attribution="user", details={"body": "two"})
    )

    as_messages = read_custom_entries(transcript.directory, TYPE, custom_messages=True)
    assert [e.type for e in as_messages] == [ENTRY_MESSAGE, ENTRY_MESSAGE]
    assert [e.payload["details"]["body"] for e in as_messages] == ["two", "one"]

    # The default is unchanged for every existing caller: custom entries only.
    as_custom = read_custom_entries(transcript.directory, TYPE)
    assert [e.type for e in as_custom] == [ENTRY_CUSTOM]
    assert as_custom[0].payload["details"]["body"] == "custom-entry"

    # max_rows still bounds the message walk.
    assert (
        len(read_custom_entries(transcript.directory, TYPE, custom_messages=True, max_rows=1)) == 1
    )


def _message_row(row_id: str, ts: float, custom_type: str, pad: int = 0) -> str:
    payload = {
        "kind": "custom",
        "custom_type": custom_type,
        "attribution": "user",
        "details": {"body": "x" * pad},
    }
    return TranscriptEntry(id=row_id, ts=ts, type=ENTRY_MESSAGE, payload=payload).to_json()


def _write_journal(path, now: float) -> None:
    """~4 MiB of rows older than a day (one bridge row per ~100 KiB), then five
    recent bridge rows — the shape of a long-lived session's journal."""

    lines: list[str] = []
    old = now - 10 * 86400
    for i in range(4000):
        if i % 100 == 0:
            lines.append(_message_row(f"old-{i}", old + i, TYPE))
        else:
            lines.append(_message_row(f"fill-{i}", old + i, "other_type", pad=1024))
    for i in range(5):
        lines.append(_message_row(f"new-{i}", now - 60 + i, TYPE))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_min_ts_excludes_old_rows_and_stops_the_walk(tmp_path, monkeypatch) -> None:
    directory = tmp_path / "s"
    directory.mkdir()
    now = time.time()
    _write_journal(directory / "transcript.jsonl", now)
    assert (directory / "transcript.jsonl").stat().st_size > 4 << 20

    decoded: list[int] = []
    real = transcript_mod._row_at_offset

    def _counting(handle, offset):
        decoded.append(offset)
        return real(handle, offset)

    monkeypatch.setattr(transcript_mod, "_row_at_offset", _counting)

    unbounded = read_custom_entries(directory, TYPE, custom_messages=True)
    assert len(unbounded) == 45
    unbounded_decodes = len(decoded)

    decoded.clear()
    bounded = read_custom_entries(directory, TYPE, custom_messages=True, min_ts=now - 86400)
    assert [e.id for e in bounded] == [f"new-{i}" for i in reversed(range(5))]
    # The walk stopped at the first chunk whose head row is outside the
    # window: at most one chunk's worth (~10) of old bridge rows is decoded,
    # against all 40 for the unbounded walk.
    assert unbounded_decodes == 45
    assert len(decoded) <= 5 + 12, len(decoded)


def test_min_ts_tolerates_rows_without_a_readable_head(tmp_path) -> None:
    """A foreign-serialized row gives no age; the bound must not drop rows."""

    directory = tmp_path / "s"
    directory.mkdir()
    now = time.time()
    foreign = json.dumps(
        {
            "type": ENTRY_MESSAGE,
            "id": "foreign",
            "ts": now,
            "payload": {"kind": "custom", "custom_type": TYPE, "details": {}},
        },
        separators=(",", ":"),
    )
    (directory / "transcript.jsonl").write_text(
        foreign + "\n" + _message_row("ours", now, TYPE) + "\n", encoding="utf-8"
    )
    rows = read_custom_entries(directory, TYPE, custom_messages=True, min_ts=now - 86400)
    assert [e.id for e in rows] == ["ours", "foreign"]


def test_chunk_head_ts_reads_the_first_row_starting_in_the_chunk() -> None:
    row_a = _message_row("a", 100.5, TYPE).encode()
    row_b = _message_row("b", 200.25, TYPE).encode()
    buf = row_a + b"\n" + row_b + b"\n"
    # Chunk starting at file start: the head is row a.
    assert transcript_mod._chunk_head_ts(buf, 0, 0) == 100.5
    # Chunk starting mid-row-a: the first row STARTING in it is row b.
    assert transcript_mod._chunk_head_ts(buf, 0, 5) == 200.25
    # Chunk starting exactly at row b's first byte.
    assert transcript_mod._chunk_head_ts(buf, 0, len(row_a) + 1) == 200.25
    # No complete head after the start: no age.
    assert transcript_mod._chunk_head_ts(buf, 0, len(buf)) is None
