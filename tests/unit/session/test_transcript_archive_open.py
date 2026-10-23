"""What opening an ARCHIVED session does — read-only, and when it cannot decode.

Review round 1 left two holes here, both about a runtime being told something
false about a session it can see:

* m2 — a pure READER that constructed ``Transcript(...)`` restored the archive
  into a plain file for good. Browsing one image on the phone re-inflated a
  280 MB transcript on the serving device, and the compression pass would not be
  allowed to take it back for another 30 days.
* m4 — an archive this build cannot decode opened as an EMPTY conversation. Only
  the next append said otherwise, so the user saw a session that had lost its
  history and the model was handed an empty one.

Both assertions below are about the observable answer a caller gets, not about
which branch ran: the archive survives a read, the rows match the archive, a
write is refused, and an undecodable archive fails the open while the LISTING
still shows the row.
"""

from __future__ import annotations

import gzip
import json
import os
from pathlib import Path

import pytest

from local_operator import resume
from local_operator.session import compression
from local_operator.session.transcript import ReadOnlyTranscript, Transcript


def _row(index: int, role: str, text: str) -> bytes:
    return (
        json.dumps(
            {
                "id": f"row{index:03d}",
                "ts": 1_700_000_000.0 + index,
                "type": "message",
                "payload": {"role": role, "content": [{"kind": "text", "text": text}]},
            }
        ).encode()
        + b"\n"
    )


def _journal(count: int = 6) -> bytes:
    return b"".join(
        _row(index, "user" if index % 2 == 0 else "assistant", f"turn {index}")
        for index in range(count)
    )


def _store(tmp_path: Path) -> Path:
    config = tmp_path / "config"
    (config / "sessions").mkdir(parents=True)
    return config


def _session(config: Path, sid: str, body: bytes | None = None) -> Path:
    directory = config / "sessions" / sid
    directory.mkdir(parents=True)
    (directory / compression.PLAIN_FILENAME).write_bytes(_journal() if body is None else body)
    return directory


def _compress(directory: Path) -> None:
    outcome = compression.compress_session(directory, codec="gzip")
    assert outcome.status == "compressed", outcome


# ---------------------------------------------------------------------------
# m2 — a read-only open leaves the archive alone
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_read_only_open_serves_the_archive_without_restoring_it(
    tmp_path: Path,
) -> None:
    directory = _session(_store(tmp_path), "ro0001")
    plain_view = Transcript(directory)
    try:
        expected = [(entry.id, entry.type) for entry in plain_view.entries()]
    finally:
        plain_view.close()
    _compress(directory)
    archive = compression.archive_path(directory)
    assert archive is not None
    digest_before = archive.read_bytes()

    reader = Transcript(directory, read_only=True)
    try:
        assert [(entry.id, entry.type) for entry in reader.entries()] == expected
        assert reader.build_llm_history()  # the fold the phone actually runs
        assert compression.archive_path(directory) == archive
        assert archive.read_bytes() == digest_before
        assert not (directory / compression.PLAIN_FILENAME).exists()
        with pytest.raises(ReadOnlyTranscript):
            await reader.append_custom("note", {"text": "must not land"})
        with pytest.raises(ReadOnlyTranscript):
            await reader.compact_file()
    finally:
        reader.close()

    # And nothing this open did made the session eligible for a second pass.
    assert compression.archive_path(directory) == archive


def test_a_writer_open_still_restores_the_archive(tmp_path: Path) -> None:
    """The control: the writer path is unchanged, or m2's fix over-reached."""
    directory = _session(_store(tmp_path), "ro0002")
    _compress(directory)
    assert compression.archive_path(directory) is not None

    writer = Transcript(directory)
    try:
        assert writer.entries()
    finally:
        writer.close()

    assert compression.archive_path(directory) is None
    assert (directory / compression.PLAIN_FILENAME).is_file()


# ---------------------------------------------------------------------------
# m4 — an undecodable archive is an ERROR, never an empty conversation
# ---------------------------------------------------------------------------


def _corrupt_archive(directory: Path) -> Path:
    body = gzip.compress(_journal(40))
    archive = directory / (compression.PLAIN_FILENAME + ".gz")
    # Truncated mid-stream: the header is intact, the stream ends early. This is
    # the "archive exists but does not decode" state readers already report.
    archive.write_bytes(body[: len(body) // 2])
    (directory / compression.PLAIN_FILENAME).unlink()
    return archive


def test_a_corrupt_archive_raises_instead_of_opening_empty(tmp_path: Path) -> None:
    config = _store(tmp_path)
    directory = _session(config, "bad001")
    archive = _corrupt_archive(directory)

    with pytest.raises(compression.ArchiveCorrupt) as raised:
        Transcript(directory)

    message = str(raised.value)
    assert archive.name in message, message
    assert "gzip" in message, message
    # The listing still shows the session — the failure is in the OPEN, exactly
    # as a byte-corrupt plain journal behaves.
    assert compression.has_transcript(directory) is True
    rows = resume.recent_session_rows(config)
    assert [row.id for row in rows] == ["bad001"]
    # And the archive was not touched by the failed open.
    assert archive.exists()
    assert (directory / compression.PLAIN_FILENAME).exists() is False


def test_an_undecodable_codec_names_the_archive_and_the_codec(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = _session(_store(tmp_path), "bad002")
    archive = directory / (compression.PLAIN_FILENAME + ".zst")
    archive.write_bytes(b"\x28\xb5\x2f\xfd" + os.urandom(64))
    (directory / compression.PLAIN_FILENAME).unlink()
    # Force the cross-version case on any interpreter: the codec exists on disk,
    # this build has no decoder for it.
    monkeypatch.setattr(compression, "_zstd", None)

    with pytest.raises(compression.ArchiveError) as raised:
        Transcript(directory)

    message = str(raised.value)
    assert archive.name in message, message
    assert "zstd" in message, message
    assert compression.has_transcript(directory) is True


@pytest.mark.asyncio
async def test_the_append_refusal_stays_for_a_held_writer(tmp_path: Path) -> None:
    """A writer opened before the archive went bad must still refuse to append."""
    directory = _session(_store(tmp_path), "bad003")
    writer = Transcript(directory)
    try:
        _corrupt_archive(directory)
        with pytest.raises(OSError, match="cannot restore"):
            await writer.append_custom("note", {"text": "must not land"})
        assert not (directory / compression.PLAIN_FILENAME).exists()
    finally:
        writer.close()
