"""The listing sidecar: a compressed session's row costs a stat, not a decode.

Review round 1 (M1) measured the defect this file pins: after compression, the
per-row listing readers fell through to ``compression.read_tail``, which streams
an archive end to end. On a 210 MB plain transcript the old seek-based preview
read 0.06 MB in 0.45 ms; the new one read 209.7 MB in 19.75 ms, PER ROW, and a
77 MB archive decoded end to end for every row of every cold ``/resume``.

The fix has two halves, and both are asserted here through the REAL surfaces:

* ``read_tail_window`` seeks on a plain journal and reports whether the window
  is truncated, so a source of exactly the window size keeps its first row
  (review n1);
* compression writes ``preview.json`` (schema 1) beside the archive and
  guarantees ``title.json`` exists before it retires the plain file, so a cold
  listing answers the preview, the failure line and the name without opening
  the archive at all. An archive that predates the sidecar still answers through
  the tail read — correctness first, one slow row rather than none.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from local_operator import resume
from local_operator.session import compression


def _row(kind: str, payload: dict[str, Any], row_id: str) -> bytes:
    return (
        json.dumps({"id": row_id, "ts": 1_700_000_000.0, "type": kind, "payload": payload}).encode()
        + b"\n"
    )


def _title_row(text: str) -> bytes:
    return _row(
        "custom",
        {"custom_type": "conversation_name", "details": {"text": text, "user_set": False}},
        "title000",
    )


def _message_row(role: str, text: str, row_id: str) -> bytes:
    return _row(
        "message",
        {"role": role, "content": [{"kind": "text", "text": text}]},
        row_id,
    )


def _incident_row(text: str) -> bytes:
    return _row(
        "custom",
        {"custom_type": "session_incident", "details": {"raw": text}},
        "incident0",
    )


def _session(config: Path, sid: str, body: bytes) -> Path:
    directory = config / "sessions" / sid
    directory.mkdir(parents=True)
    (directory / compression.PLAIN_FILENAME).write_bytes(body)
    return directory


def _store(tmp_path: Path) -> Path:
    config = tmp_path / "config"
    (config / "sessions").mkdir(parents=True)
    return config


def _body(title: str, answer: str, *, incident: str = "") -> bytes:
    """A journal with a title near the HEAD, an incident and a final answer.

    The title is near the head because that is where the auto-name lands, and
    because it is the row that lets the compression pass seed ``title.json``:
    with one, the archived session's name never needs the transcript at all.
    """
    rows = [
        _title_row(title),
        _message_row("user", f"open the {title.lower()}", "row0001"),
        _message_row("assistant", "working on it", "row0002"),
    ]
    if incident:
        rows.append(_incident_row(incident))
    rows.append(_message_row("assistant", answer, "row9999"))
    return b"".join(rows)


# ---------------------------------------------------------------------------
# M1 — the sidecar is written, and a cold listing reads nothing else
# ---------------------------------------------------------------------------


def test_compression_writes_the_listing_sidecar_from_the_plain_tail(tmp_path: Path) -> None:
    config = _store(tmp_path)
    directory = _session(config, "side01", _body("Harbour plan", "the crane arrives at dawn"))

    preview = resume.session_preview(directory)
    failure = resume.session_failure_summary(directory)
    title = resume.stored_session_title(directory)

    outcome = compression.compress_session(directory)
    assert outcome.status == "compressed", outcome

    sidecar = json.loads((directory / compression.PREVIEW_SIDECAR_NAME).read_text())
    assert sidecar["schema"] == compression.PREVIEW_SIDECAR_SCHEMA
    # The sidecar answers EXACTLY what the plain file answered: same function,
    # same window, same rules.
    assert sidecar["preview"] == preview == "the crane arrives at dawn"
    assert sidecar["failure"] == failure == ""
    # And the name survives compression in its own sidecar, which is what lets
    # the archived reader stay on the head window.
    assert resume.stored_session_title(directory) == title == "Harbour plan"
    assert (directory / resume.TITLE_SIDECAR_NAME).is_file()


def test_the_sidecar_pin_between_the_two_modules_holds() -> None:
    """``compression`` must import standalone, so the window size is spelled twice."""
    assert compression.PREVIEW_TAIL_BYTES == resume.PREVIEW_SCAN_BYTES


def test_a_cold_listing_and_preview_pass_makes_zero_archive_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _store(tmp_path)
    bodies = {
        "cold01": _body("Harbour plan", "the crane arrives at dawn"),
        "cold02": _body("Invoice chase", "sent the second reminder", incident="tool timed out"),
        "cold03": _body("Roof repair", "two quotes are in"),
    }
    directories = {sid: _session(config, sid, body) for sid, body in bodies.items()}
    plain = {
        sid: (
            resume.session_preview(directory),
            resume.session_failure_summary(directory),
        )
        for sid, directory in directories.items()
    }
    for directory in directories.values():
        assert compression.compress_session(directory).status == "compressed"

    opens: list[str] = []
    tails: list[str] = []
    heads: list[str] = []
    real_open = compression._open_archive_reader
    real_tail = compression.read_tail
    real_head = compression.read_head

    def counting_open(path: Path, codec: str | None) -> Any:
        opens.append(Path(path).name)
        return real_open(path, codec)

    def counting_tail(directory: Path, limit: int) -> bytes:
        tails.append(Path(directory).name)
        return real_tail(directory, limit)

    def counting_head(directory: Path, limit: int) -> bytes:
        heads.append(Path(directory).name)
        return real_head(directory, limit)

    monkeypatch.setattr(compression, "_open_archive_reader", counting_open)
    monkeypatch.setattr(compression, "read_tail", counting_tail)
    monkeypatch.setattr(compression, "read_head", counting_head)

    rows = resume.recent_session_rows(config)
    previews = {sid: resume.session_preview(directory) for sid, directory in directories.items()}
    failures = {
        sid: resume.session_failure_summary(directory) for sid, directory in directories.items()
    }

    assert sorted(row.id for row in rows) == ["cold01", "cold02", "cold03"]
    assert {row.id: row.name for row in rows} == {
        "cold01": "Harbour plan",
        "cold02": "Invoice chase",
        "cold03": "Roof repair",
    }
    assert previews == {sid: answer for sid, (answer, _failure) in plain.items()}
    assert failures == {sid: failure for sid, (_answer, failure) in plain.items()}
    # THE POINT: not one archive was opened, no tail was read — and no head
    # either, because the opening message was never needed (the title answered).
    assert opens == []
    assert tails == []
    assert heads == []


def test_an_archive_without_a_sidecar_still_answers_through_the_tail(
    tmp_path: Path,
) -> None:
    """A pre-upgrade archive: the fallback is slower, never wrong."""
    config = _store(tmp_path)
    directory = _session(config, "old001", _body("Old session", "an answer from before"))
    expected = resume.session_preview(directory)
    assert compression.compress_session(directory).status == "compressed"
    # Exactly the state a build without the sidecar leaves behind.
    (directory / compression.PREVIEW_SIDECAR_NAME).unlink()

    assert compression.read_listing_preview(directory) is None
    assert resume.session_preview(directory) == expected
    # The name still resolves, from the seeded title sidecar.
    assert resume.stored_session_title(directory) == "Old session"


def test_restore_removes_the_listing_sidecar(tmp_path: Path) -> None:
    """The plain file is authoritative again; a stale preview would outlive it."""
    config = _store(tmp_path)
    directory = _session(config, "rest01", _body("Restore me", "a reply"))
    assert compression.compress_session(directory).status == "compressed"
    assert (directory / compression.PREVIEW_SIDECAR_NAME).is_file()

    assert compression.restore_for_write(directory) is True
    assert not (directory / compression.PREVIEW_SIDECAR_NAME).exists()
    assert compression.read_listing_preview(directory) is None


def test_retention_treats_the_sidecar_as_bookkeeping(tmp_path: Path) -> None:
    """Bookkeeping: it must move neither the activity clock nor the size budget."""
    from local_operator.session import retention

    assert compression.PREVIEW_SIDECAR_NAME in retention._SIDECAR_NAMES
    assert compression.PREVIEW_SIDECAR_NAME not in retention._ACTIVITY_FILES


# ---------------------------------------------------------------------------
# The plain fast path, and the n1 boundary
# ---------------------------------------------------------------------------


class _Recording:
    """A file handle that counts the bytes actually read through it."""

    def __init__(self, handle: Any) -> None:
        self._handle = handle
        self.bytes_read = 0

    def read(self, size: int = -1) -> bytes:
        data = self._handle.read(size)
        self.bytes_read += len(data)
        return data

    def __enter__(self) -> "_Recording":
        return self

    def __exit__(self, *exc: object) -> None:
        self._handle.close()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._handle, name)


def test_a_plain_tail_read_seeks_instead_of_streaming(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _store(tmp_path)
    body = b"".join(_message_row("user", "x" * 2000, f"pad{index:05d}") for index in range(1500))
    directory = _session(config, "seek01", body)
    assert len(body) > compression.PREVIEW_TAIL_BYTES

    recorded: list[_Recording] = []
    real_open = Path.open

    def spying_open(self: Path, *args: Any, **kwargs: Any) -> _Recording:
        handle = real_open(self, *args, **kwargs)
        if self.name == compression.PLAIN_FILENAME:
            wrapper = _Recording(handle)
            recorded.append(wrapper)
            return wrapper
        return handle

    monkeypatch.setattr(Path, "open", spying_open)

    window, truncated = compression.read_tail_window(directory, compression.PREVIEW_TAIL_BYTES)

    assert truncated is True
    assert len(window) == compression.PREVIEW_TAIL_BYTES
    assert recorded and recorded[0].bytes_read == compression.PREVIEW_TAIL_BYTES
    assert recorded[0].bytes_read < len(body)


def _exactly_sized(body_bytes: int) -> bytes:
    """A journal exactly ``body_bytes`` long, whose FIRST row is distinctive."""
    first = _message_row("user", "FIRST-ROW-MARKER", "first001")
    # Each "y" costs exactly one byte in the encoded row, so the empty-text row
    # measures the fixed overhead and the difference is the padding.
    overhead = len(_message_row("assistant", "", "filler01"))
    padding = body_bytes - len(first) - overhead
    assert padding >= 0
    rows = first + _message_row("assistant", "y" * padding, "filler01")
    assert len(rows) == body_bytes, (len(rows), body_bytes)
    return rows


def test_a_window_sized_transcript_keeps_its_first_row(tmp_path: Path) -> None:
    """n1: at exactly the window size the first row is real, not a fragment."""
    config = _store(tmp_path)
    directory = _session(config, "boundary", _exactly_sized(compression.PREVIEW_TAIL_BYTES))
    window, truncated = compression.read_tail_window(directory, compression.PREVIEW_TAIL_BYTES)
    assert len(window) == compression.PREVIEW_TAIL_BYTES
    assert truncated is False
    ids = [entry.get("id") for entry in resume._tail_entries(directory)]
    assert "first001" in ids


def test_a_truncated_window_drops_only_the_fragment(tmp_path: Path) -> None:
    config = _store(tmp_path)
    directory = _session(config, "boundary2", _exactly_sized(compression.PREVIEW_TAIL_BYTES + 1))
    window, truncated = compression.read_tail_window(directory, compression.PREVIEW_TAIL_BYTES)
    assert truncated is True
    ids = [entry.get("id") for entry in resume._tail_entries(directory)]
    assert "first001" not in ids


# ---------------------------------------------------------------------------
# Round 2 M1 — the seed ranks the window that is NEWER
# ---------------------------------------------------------------------------


def _renamed_legacy_body(*, filler_rows: int = 900) -> bytes:
    """A legacy journal: an auto-name at the head, ``/rename`` long after it.

    The filler is what makes the case real. It is >2x ``TITLE_SCAN_BYTES``
    (128 KiB), so the late title is outside the HEAD window and the auto-name is
    outside the TAIL one: the two windows disagree about which row is newest,
    which is the shape a tail-then-head concatenation gets backwards.
    """
    rows = [_title_row("Early auto name")]
    rows.extend(
        _message_row("user", f"filler {index} " + "harbour " * 20, f"fill{index:04d}")
        for index in range(filler_rows)
    )
    rows.append(_title_row("Late rename"))
    rows.append(_message_row("assistant", "renamed near the end", "lastrow"))
    return b"".join(rows)


def test_compression_seeds_the_late_rename_not_the_opening_name(tmp_path: Path) -> None:
    """THE REPRO (review M1): a legacy session renamed mid-conversation.

    Seeding ``title.json`` from ``tail + head`` made ``found[-1]`` the HEAD
    title, so compressing a renamed session permanently renamed it back — and
    the title backfill then skips it, because ``title.json`` now exists. The
    seed must agree with ``stored_session_title``: the last row in FILE order.
    The 260 KB journal here is the reviewer's 602 KB one in miniature, sized so
    the two windows are disjoint.
    """
    config = _store(tmp_path)
    directory = _session(config, "rename01", _renamed_legacy_body())
    # The plain twin's answer, from the window scan's own rule.
    assert resume.stored_session_title(directory) == "Late rename"

    assert compression.compress_session(directory).status == "compressed"

    sidecar = json.loads((directory / resume.TITLE_SIDECAR_NAME).read_text())
    assert sidecar["text"] == "Late rename"
    assert "Early auto name" in sidecar["names"]
    assert resume.stored_session_title(directory) == "Late rename"


def test_the_backfill_scan_ranks_a_late_rename_in_an_archive(tmp_path: Path) -> None:
    """``_scan_all_titles`` must stream an archive, not answer from its head.

    Its caller ranks ``titles[-1]`` as the newest, so a head-only answer for an
    archive names the session with its oldest title and writes that sidecar.
    Only an archive with NO ``title.json`` reaches it, and streaming one once
    ever is the price of the answer being right.
    """
    config = _store(tmp_path)
    directory = _session(config, "rename02", _renamed_legacy_body())
    assert compression.compress_session(directory).status == "compressed"
    # The state that reaches the scan: an archive from a build without the seed.
    (directory / resume.TITLE_SIDECAR_NAME).unlink()

    titles = resume._scan_all_titles(directory)
    assert [text for text, _user_set in titles] == ["Early auto name", "Late rename"]

    # And the caller therefore writes the late name, not the opening one.
    assert resume.backfill_session_titles(config) == 1
    assert resume.stored_session_title(directory) == "Late rename"


# ---------------------------------------------------------------------------
# Round 2 n2 — the sidecar records the cap it cut at
# ---------------------------------------------------------------------------


def test_a_caller_asking_for_more_than_the_sidecar_stores_reads_the_tail(
    tmp_path: Path,
) -> None:
    """A 200-char sidecar cannot answer a 500-char request; the tail can."""
    config = _store(tmp_path)
    long_answer = "the crane arrives at dawn and the second one at dusk " * 8
    plain = _session(config, "wide01", _body("Harbour plan", long_answer))
    archived = _session(config, "wide02", _body("Harbour plan", long_answer))
    expected_narrow = resume.session_preview(plain, max_chars=120)
    expected_wide = resume.session_preview(plain, max_chars=500)
    assert len(expected_wide) > resume.PREVIEW_MAX_CHARS

    assert compression.compress_session(archived).status == "compressed"
    sidecar = json.loads((archived / compression.PREVIEW_SIDECAR_NAME).read_text())
    assert sidecar["max_chars"] == resume.PREVIEW_MAX_CHARS

    # Under the cap: the sidecar answers, byte-identical to the plain twin.
    assert resume.session_preview(archived, max_chars=120) == expected_narrow
    # Over it: the stored text was already cut, so the answer comes from the
    # bounded tail read — the same answer, not a silently shorter one.
    assert resume.session_preview(archived, max_chars=500) == expected_wide
