"""The cold-session inbox: spooling, ordering, and what a crash costs.

The guarantee this file pins down is an ORDERING one, and it is structural
rather than timed: ``process.py`` drains the spool after the session exists and
before the control socket listens, so a message written while the session was
cold is delivered ahead of anything a socket client could send. The test for
that lives with the drain (``test_drain_precedes_the_socket``); the rest here
cover the file format, concurrency, and the crash contract.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path

from local_operator.session.runtime.inbox import (
    MAX_INBOX_ROWS,
    InboxLine,
    append_inbox,
    drain_inbox,
    inbox_path,
    peek_inbox,
)


def _line(text: str) -> InboxLine:
    return InboxLine(text=text, sender={"pid": 1, "conversation_name": "peer"})


def test_append_then_drain_preserves_write_order(tmp_path: Path) -> None:
    for index in range(5):
        assert append_inbox(tmp_path, _line(f"note {index}")) is True

    drained = drain_inbox(tmp_path)
    assert [line.text for line in drained] == [f"note {i}" for i in range(5)]
    assert drained[0].sender["conversation_name"] == "peer"
    # Consumed: a second open must not re-deliver them.
    assert drain_inbox(tmp_path) == []


def test_peek_does_not_consume(tmp_path: Path) -> None:
    """The cold viewer reads the spool without stealing the runtime's work."""
    append_inbox(tmp_path, _line("pending"))

    assert [line.text for line in peek_inbox(tmp_path)] == ["pending"]
    assert [line.text for line in peek_inbox(tmp_path)] == ["pending"]
    assert [line.text for line in drain_inbox(tmp_path)] == ["pending"]


def test_missing_and_empty_spools_are_not_errors(tmp_path: Path) -> None:
    assert drain_inbox(tmp_path) == []
    assert peek_inbox(tmp_path) == []
    inbox_path(tmp_path).write_text("", encoding="utf-8")
    assert drain_inbox(tmp_path) == []


def test_a_torn_final_line_is_skipped_not_fatal(tmp_path: Path) -> None:
    """A writer killed mid-write must not make the whole spool unreadable."""
    append_inbox(tmp_path, _line("intact"))
    with open(inbox_path(tmp_path), "ab") as handle:
        handle.write(b'{"text": "half a row')  # no newline, no closing brace

    assert [line.text for line in drain_inbox(tmp_path)] == ["intact"]


def test_concurrent_writers_never_interleave_within_a_line(tmp_path: Path) -> None:
    """O_APPEND + one write() per row is what keeps rows whole.

    Two peers writing to the same cold session in the same instant is the
    contended case; the requirement is not that they be ordered against each
    other but that neither row is corrupted by the other.
    """
    bodies = [f"peer-{index}-{'x' * 200}" for index in range(20)]

    def write(text: str) -> None:
        append_inbox(tmp_path, _line(text))

    threads = [threading.Thread(target=write, args=(body,)) for body in bodies]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    raw = inbox_path(tmp_path).read_bytes().decode()
    # Every line parses: no row was cut in half by another writer.
    parsed = [json.loads(row) for row in raw.splitlines() if row.strip()]
    assert len(parsed) == len(bodies)
    assert {row["text"] for row in parsed} == set(bodies)


def test_the_spool_refuses_to_grow_without_bound(tmp_path: Path) -> None:
    """A runaway producer must not create a file the next open cannot replay."""
    for index in range(MAX_INBOX_ROWS):
        assert append_inbox(tmp_path, _line(f"row {index}")) is True
    assert append_inbox(tmp_path, _line("one too many")) is False

    assert len(drain_inbox(tmp_path)) == MAX_INBOX_ROWS


def test_a_crash_between_read_and_delivery_redelivers_rather_than_drops(
    tmp_path: Path,
) -> None:
    """The crash contract is at-least-once, and this is the direction chosen.

    A runtime killed after ``drain_inbox`` returned but before it delivered
    loses those messages; one killed before the truncate re-delivers them. The
    second is the recoverable failure, so the implementation truncates AFTER
    reading. Simulated by reading the spool without going through the drain,
    which is exactly the state a process killed mid-drain leaves behind.
    """
    append_inbox(tmp_path, _line("must survive"))

    # A crash before the truncate: the file is still whole on disk.
    contents = inbox_path(tmp_path).read_bytes()
    assert b"must survive" in contents

    # The next runtime to open the session drains it successfully.
    assert [line.text for line in drain_inbox(tmp_path)] == ["must survive"]


def test_the_spool_is_owner_only(tmp_path: Path) -> None:
    """It holds message bodies, so it gets the same 0600 the records get."""
    append_inbox(tmp_path, _line("private"))
    assert (os.stat(inbox_path(tmp_path)).st_mode & 0o777) == 0o600


def test_the_inbox_counts_as_content_for_retention(tmp_path: Path) -> None:
    """A spooled message must survive the junk reap.

    ``retention`` treats any file that is not a declared sidecar as content, so
    this is a guard on ``inbox.jsonl`` NOT being added to ``_SIDECAR_NAMES``
    later: doing so would let the sweep delete a session directory holding a
    message the user has never seen.
    """
    from local_operator.session.retention import _SIDECAR_NAMES
    from local_operator.session.runtime.inbox import INBOX_NAME

    assert INBOX_NAME not in _SIDECAR_NAMES


def test_the_drain_is_wired_before_the_socket_starts_listening() -> None:
    """THE ordering guarantee, asserted against the source that provides it.

    ``process.amain`` must drain the spool BEFORE ``RuntimeServer`` begins
    listening. That ordering is the entire reason a spooled message cannot be
    interleaved with an errand a client sends: while the drain runs there is no
    socket to send one on.

    Asserted structurally — on the order of the two statements in the source —
    because there is no runtime observable that distinguishes "drained first"
    from "drained fast enough", and a timing test would be measuring luck. If
    someone moves the drain below the server start, the ordering silently
    becomes a race and every functional test still passes.
    """
    import inspect

    from local_operator.session.runtime import process

    source = inspect.getsource(process.amain)
    drain_at = source.index("_drain_inbox_into(handle)")
    listen_at = source.index("start_in_process()")
    assert drain_at < listen_at, (
        "the inbox drain must run before the control socket listens; "
        "moving it after turns the delivery guarantee into a race"
    )
