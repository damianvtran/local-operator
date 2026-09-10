"""Saved display must be bounded without inventing a canonical replay cut."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

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


def test_missing_journal_is_not_a_complete_empty_saved_conversation(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_saved_preview(tmp_path / "missing")
    with pytest.raises(FileNotFoundError):
        read_saved_preview(tmp_path)
    assert not (tmp_path / "missing").exists()


def test_existing_empty_journal_is_a_valid_empty_preview(tmp_path):
    (tmp_path / "transcript.jsonl").touch()
    result = read_saved_preview(tmp_path)
    assert result.messages == [] and not result.partial


@pytest.mark.asyncio
async def test_unmaterialized_empty_view_requires_a_discoverable_owner(tmp_path, monkeypatch):
    from local_operator.session.remote import RemoteSession

    record = SimpleNamespace(cwd="/synthetic-owner")
    monkeypatch.setattr(
        "local_operator.mobile.attach_client.find_runtime_record", lambda *args: (record, 12345)
    )

    async def no_takeover():
        raise AssertionError("a preview must not become an execution owner")

    remote = await RemoteSession.saved_preview(
        "unstarted", config_dir=tmp_path, cwd="/other", takeover_factory=no_takeover
    )
    try:
        assert remote.is_cold
        assert remote.frontend_state.cwd == "/synthetic-owner"
        assert remote.display_history_window() == []
        assert not (tmp_path / "sessions" / "unstarted").exists()
    finally:
        await remote.dispose()


@pytest.mark.asyncio
async def test_absent_owner_and_journal_refuse_before_building_facade(tmp_path, monkeypatch):
    from local_operator.session.remote import RemoteSession

    monkeypatch.setattr(
        "local_operator.mobile.attach_client.find_runtime_record", lambda *args: (None, None)
    )

    async def no_takeover():
        raise AssertionError("a missing target must not launch an owner")

    with pytest.raises(FileNotFoundError, match="no longer available"):
        await RemoteSession.saved_preview(
            "deleted", config_dir=tmp_path, cwd="/other", takeover_factory=no_takeover
        )
    assert not (tmp_path / "sessions" / "deleted").exists()


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
