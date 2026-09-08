"""Saved display must be bounded without inventing a canonical replay cut."""

from __future__ import annotations

from pathlib import Path

import pytest

from local_operator.harness.types import Message
from local_operator.session.saved_preview import PREVIEW_BYTES, read_saved_preview
from local_operator.session.transcript import TranscriptEntry, encode_message_payload


def message(identifier: str, text: str) -> TranscriptEntry:
    return TranscriptEntry(identifier, 0, "message", encode_message_payload(Message.user(text)))


def journal(path: Path, entries: list[TranscriptEntry]) -> None:
    (path / "transcript.jsonl").write_text(
        "".join(entry.to_json() + "\n" for entry in entries), encoding="utf-8"
    )


def test_preview_replays_correct_session_and_prunes(tmp_path):
    journal(
        tmp_path,
        [
            message("old", "original tool output"),
            message("new", "Useful saved answer"),
            TranscriptEntry("prune", 0, "prune", {"target": "old", "notice": "Removed"}),
        ],
    )
    result = read_saved_preview(tmp_path)
    assert not result.partial
    assert result.messages[-1].text == "Useful saved answer"
    assert "original tool output" not in result.messages[0].text


def test_preview_reads_only_bounded_suffix(tmp_path, monkeypatch):
    journal(tmp_path, [message("huge", "x" * (PREVIEW_BYTES * 16)), message("tail", "Useful tail")])
    original = Path.open
    reads = []

    class BoundedRead:
        def __init__(self, handle):
            self.handle = handle

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.handle.close()

        def seek(self, *args):
            return self.handle.seek(*args)

        def tell(self):
            return self.handle.tell()

        def read(self, count=-1):
            reads.append(count)
            assert 0 <= count <= PREVIEW_BYTES
            return self.handle.read(count)

    monkeypatch.setattr(Path, "open", lambda path, *a, **kw: BoundedRead(original(path, *a, **kw)))
    result = read_saved_preview(tmp_path)
    assert result.partial
    assert [row.text for row in result.messages] == ["Useful tail"]
    assert reads == [PREVIEW_BYTES]


@pytest.mark.parametrize("suffix", [b'{"type":"prune"', b"not json\n"])
def test_incomplete_or_corrupt_tail_cannot_resurrect_content(tmp_path, suffix):
    journal(tmp_path, [message("saved", "May have been retracted")])
    with (tmp_path / "transcript.jsonl").open("ab") as handle:
        handle.write(suffix)
    result = read_saved_preview(tmp_path)
    assert result.partial and result.messages == []


def test_unresolved_compaction_cut_does_not_use_full_history_fallback(tmp_path):
    journal(
        tmp_path,
        [
            message("saved", "Do not resurrect"),
            TranscriptEntry("compact", 0, "compaction", {"first_kept_entry_id": "missing"}),
        ],
    )
    result = read_saved_preview(tmp_path)
    assert result.partial and result.messages == []


def test_oversized_last_row_reports_unavailable_instead_of_fake_content(tmp_path):
    journal(tmp_path, [message("huge", "x" * (PREVIEW_BYTES * 2))])
    result = read_saved_preview(tmp_path)
    assert result.partial and result.messages == []
