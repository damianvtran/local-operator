"""``read_entry_by_id`` — the pure by-ID read the S-7 recovery loader stands on.

The contract pinned here is the one the consumed-ASK_RESPONSE loader relies on:
a row is addressed by the ENTRY ID (which ``append_message`` guarantees IS the
message's own id — never a minted one), the read is PURE (an absent journal
answers ``None`` and the directory is NOT created), the newest occurrence of an
id wins, malformed lines are skipped individually, and a journal re-serialized
by something other than ``TranscriptEntry.to_json`` is still answered through
the parse walk. Rows here are built with the REAL serializer and the REAL
reader — no hand-rolled JSON, so a format drift cannot hide behind a fixture
that drifted with it.
"""

from __future__ import annotations

import json

from local_operator.session.transcript import (
    TRANSCRIPT_FILENAME,
    TranscriptEntry,
    read_entry_by_id,
)

ENTRY_MESSAGE = "message"


def _row(entry_id: str, *, payload: dict, ts: float = 1.0) -> TranscriptEntry:
    return TranscriptEntry(entry_id, ts, ENTRY_MESSAGE, payload)


def _write(directory, rows: list[TranscriptEntry], *, separators=(",", ":")) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    lines = []
    for entry in rows:
        if separators == (",", ":"):
            lines.append(entry.to_json())
        else:
            lines.append(
                json.dumps(
                    {
                        "id": entry.id,
                        "ts": entry.ts,
                        "type": entry.type,
                        "payload": entry.payload,
                    },
                    separators=separators,
                )
            )
    (directory / TRANSCRIPT_FILENAME).write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_an_absent_directory_answers_none_without_creating_it(tmp_path) -> None:
    directory = tmp_path / "not-yet-materialised"
    assert read_entry_by_id(directory, "anything") is None
    # THE PURITY GUARANTEE ``read_latest_custom_entry`` documents, one reader
    # over: a scan of somebody else's session leaves nothing behind.
    assert not directory.exists()


def test_the_row_with_the_exact_entry_id_is_answered(tmp_path) -> None:
    _write(
        tmp_path,
        [
            _row("alpha", payload={"kind": "message", "body": "one"}),
            _row("beta", payload={"kind": "custom", "custom_type": "ask_response"}),
        ],
    )
    entry = read_entry_by_id(tmp_path, "beta")
    assert entry is not None
    assert entry.id == "beta"
    assert entry.payload["custom_type"] == "ask_response"
    assert read_entry_by_id(tmp_path, "missing") is None


def test_the_newest_occurrence_of_a_duplicated_id_wins(tmp_path) -> None:
    # The cursor locator's documented resolution rule, kept by the by-id
    # reader: scan from the end, first match IS the newest.
    _write(
        tmp_path,
        [
            _row("dup", payload={"body": "old"}, ts=1.0),
            _row("dup", payload={"body": "new"}, ts=2.0),
        ],
    )
    entry = read_entry_by_id(tmp_path, "dup")
    assert entry is not None and entry.payload["body"] == "new"


def test_a_payload_echo_of_an_id_is_not_the_row(tmp_path) -> None:
    # ``{"id":"ghost"`` nested inside a payload sits after a colon, never at a
    # line start — the candidate must not be believed without the parse check.
    _write(
        tmp_path,
        [_row("real", payload={"kind": "message", "nested": {"id": "ghost"}})],
    )
    assert read_entry_by_id(tmp_path, "ghost") is None
    assert read_entry_by_id(tmp_path, "real") is not None


def test_reserialized_rows_are_answered_by_the_parse_walk(tmp_path) -> None:
    # Spaced separators break the compact byte needle; the walk is the
    # authority for exactly this case (mirrors read_latest_custom_entry).
    _write(
        tmp_path,
        [_row("spaced", payload={"kind": "message"})],
        separators=(", ", ": "),
    )
    entry = read_entry_by_id(tmp_path, "spaced")
    assert entry is not None and entry.id == "spaced"


def test_an_escaped_id_spelling_falls_back_to_the_walk(tmp_path) -> None:
    # An id whose escaped spelling differs from its literal one cannot be
    # searched for by bytes at all; the walk must still answer it.
    _write(tmp_path, [_row('we"ird', payload={"kind": "message"})])
    entry = read_entry_by_id(tmp_path, 'we"ird')
    assert entry is not None and entry.id == 'we"ird'


def test_malformed_lines_are_skipped_individually(tmp_path) -> None:
    tmp_path.mkdir(parents=True, exist_ok=True)
    good = _row("good", payload={"kind": "message"}).to_json()
    (tmp_path / TRANSCRIPT_FILENAME).write_text(
        "{not json at all\n" + good + "\n" + good[:20] + "\n",
        encoding="utf-8",
    )
    entry = read_entry_by_id(tmp_path, "good")
    assert entry is not None and entry.id == "good"
