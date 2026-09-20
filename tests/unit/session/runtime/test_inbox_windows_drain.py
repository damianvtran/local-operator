"""The unlocked inbox drain on a platform where the lock is never held (audit D11).

The writer side is portable and safe on its own — one ``O_APPEND`` write of a
whole line. The *drain* side is where the platform mattered: it read the file
and then emptied it, and the only reason that is safe when the lock could not be
taken is that it stages the remainder through ``os.replace`` instead. Windows
takes the unlocked branch by construction (there is no ``fcntl`` there, so the
lock is never acquired), and it was being sent to the ``ftruncate`` branch —
the one the code's own comment identifies as able to discard a row an appender
wrote between the read and the truncate. At-most-once where the module's
documented contract is at-least-once.

**What is patched, and why it is faithful.** ``os.name`` is set to ``"nt"`` for
exactly the duration of the drain call, so the code under test takes the branch
a Windows process takes. That patch is deliberately narrow: ``pathlib`` reads
``os.name`` when it constructs a path, so leaving it set would turn the next
``Path(...)`` in this process into a ``WindowsPath`` and fail with
``UnsupportedOperation``. The session directory is built before the patch, and
``Path`` instances already created keep their own flavour.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from local_operator.session.runtime import inbox

RACING_ROW = b'{"text":"appended while the drain ran","sender":{},"mode":"mailbox"}\n'


def _write_rows(path: Path, *texts: str) -> bytes:
    """Append rows the way a sender does: whole JSON lines, one ``write`` each."""
    payload = b"".join(
        json.dumps(
            inbox.InboxLine(text=text, sender={"name": "peer"}, wake=False).to_json(),
            separators=(",", ":"),
        ).encode()
        + b"\n"
        for text in texts
    )
    with open(path, "ab") as handle:
        handle.write(payload)
    return payload


def test_an_unlocked_drain_keeps_a_row_appended_while_it_ran(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The defect: the racing row was truncated away instead of staged.

    The race is made real rather than described — the appender's write happens
    between the drain's read and its emptying, which is the window the
    ``ftruncate`` branch loses. It appends ONCE, on the first read: the drain
    reads twice now (see below) and a monkeypatch that fired on both would model
    two racing senders rather than the one this test names.
    """
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    path = inbox.inbox_path(session_dir)
    _write_rows(path, "first", "second")

    original_read_all = inbox._read_all
    appended = False

    def _read_then_race(fd: int) -> bytes:
        nonlocal appended
        raw = original_read_all(fd)
        if not appended:
            appended = True
            # An appender arriving in the window the comment names. Closed
            # before returning, so this stands in for a sender rather than for
            # an open handle that would block the rename on Windows.
            with open(path, "ab") as handle:
                handle.write(RACING_ROW)
        return raw

    monkeypatch.setattr(inbox, "_read_all", _read_then_race)

    with monkeypatch.context() as windows:
        windows.setattr(os, "name", "nt")
        lines = inbox.drain_inbox(session_dir)

    # THE PROPERTY THIS TEST EXISTS FOR: the racing row is delivered, not
    # truncated away.
    #
    # It arrives in THIS batch, which is upstream's latest-bytes re-read doing
    # its job: `latest` is parsed so that a recall marker landing mid-read still
    # withholds its row, and the same re-read is what brings a racing APPEND
    # into the batch instead of leaving it for the next open.
    delivered = [line.text for line in lines]
    assert delivered == ["first", "second", "appended while the drain ran"], delivered
    # And the spool is left EMPTY, which is the whole point rather than a
    # surprise: `_replace_remainder` stages only the bytes written AFTER the
    # bytes that were consumed, and this batch consumed everything `latest`
    # holds — so nothing is left to stage. The row is delivered exactly once:
    # not truncated away (the old `ftruncate` branch) and not duplicated (it
    # would have been, had the remainder been `latest` rather than its tail).
    assert path.read_bytes() == b"", path.read_bytes()


def test_a_clean_drain_still_delivers_and_empties(tmp_path: Path) -> None:
    """The neighbour this change could plausibly regress: an uncontended drain."""
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    path = inbox.inbox_path(session_dir)
    _write_rows(path, "only")

    assert [line.text for line in inbox.drain_inbox(session_dir)] == ["only"]
    assert path.read_bytes() == b""
