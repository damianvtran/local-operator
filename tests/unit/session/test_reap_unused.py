"""The junk backfill: remove directories no user ever spoke into.

THE ONLY DESTRUCTIVE CODE IN THIS CHANGE, so these tests are mostly NEGATIVE
CONTROLS — one per way a directory can matter. A false positive here deletes
somebody's conversation, and the asymmetry between the two mistakes (keeping
junk costs disk; deleting work costs the user their work) is what makes every
uncertain answer a keep.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from local_operator.session.retention import (
    EMPTY_DIR_GRACE_SECONDS,
    UNUSED_SCAN_CHARS,
    _is_unused_session,
    reap_unused_sessions,
)

OLD = time.time() - EMPTY_DIR_GRACE_SECONDS * 2


def _age(directory: Path, age: float = OLD) -> None:
    """Re-stamp the directory's mtime.

    Needed after ANY write into it: writing a claim marker or an inbox row
    updates the directory's mtime, which would otherwise put the fixture
    inside the grace window and make the test assert the wrong guard.
    """
    import os

    os.utime(directory, (age, age))


def _session(
    root: Path, name: str, rows: list[dict[str, Any]] | None = None, *, age: float = OLD
) -> Path:
    directory = root / "sessions" / name
    directory.mkdir(parents=True, exist_ok=True)
    if rows is not None:
        (directory / "transcript.jsonl").write_text(
            "\n".join(json.dumps(row) for row in rows), encoding="utf-8"
        )
    import os

    os.utime(directory, (age, age))
    return directory


def _machine_rows() -> list[dict[str, Any]]:
    """What a session that nobody spoke into actually contains.

    Taken from the reference store: a model-route row and a boot incident,
    both written by the harness itself. This is the population the backfill
    exists to remove.
    """
    return [
        {"kind": "custom", "payload": {"custom_type": "active_model_route", "details": {}}},
        {"kind": "message", "payload": {"custom_type": "session_incident", "details": {}}},
    ]


def test_a_machine_only_session_is_reapable(tmp_path: Path) -> None:
    directory = _session(tmp_path, "junk00000001", _machine_rows())

    verdict, reason = _is_unused_session(directory, time.time())

    assert verdict is True
    assert "no user turn" in reason


def test_one_user_turn_saves_the_directory(tmp_path: Path) -> None:
    """The primary control: a human spoke here, so it is a conversation."""
    directory = _session(
        tmp_path,
        "real00000001",
        _machine_rows() + [{"kind": "message", "payload": {"role": "user", "text": "hello"}}],
    )

    verdict, reason = _is_unused_session(directory, time.time())

    assert verdict is False
    assert reason == "has a user turn"


def test_a_scheduled_wake_saves_the_directory(tmp_path: Path) -> None:
    """A wake is the user saying "bring me back here", even with no message."""
    directory = _session(tmp_path, "waked0000001", _machine_rows())
    from local_operator.wakes.store import write_entry

    write_entry(
        tmp_path,
        "waked0000001",
        cwd=str(tmp_path),
        schedules=[{"id": "w1", "message": "later", "next_due_at": 4_102_444_800_000}],
    )

    verdict, reason = _is_unused_session(directory, time.time())

    assert verdict is False
    assert reason == "has scheduled wakes"


def test_spooled_mail_saves_the_directory(tmp_path: Path) -> None:
    """An unread message was accepted on a promise; reaping would break it."""
    directory = _session(tmp_path, "mailed000001", _machine_rows())
    from local_operator.session.runtime.inbox import InboxLine, append_inbox

    append_inbox(directory, InboxLine(text="read me when you open", sender={}))
    _age(directory)

    verdict, reason = _is_unused_session(directory, time.time())

    assert verdict is False
    assert reason == "has an unread spooled message"


def test_a_live_claim_saves_the_directory(tmp_path: Path) -> None:
    """A process owns this right now, whatever is (or is not) written in it."""
    import os

    directory = _session(tmp_path, "claimed00001", _machine_rows())
    from local_operator.session.retention import claim_session

    claim_session(directory, os.getpid())
    _age(directory)

    verdict, reason = _is_unused_session(directory, time.time())

    assert verdict is False
    assert reason == "claimed by a live process"


def test_a_fresh_directory_is_never_a_candidate(tmp_path: Path) -> None:
    """A session created seconds ago is one message away from being real."""
    directory = _session(tmp_path, "fresh0000001", _machine_rows(), age=time.time())

    verdict, reason = _is_unused_session(directory, time.time())

    assert verdict is False
    assert reason == "inside the grace window"


def test_an_over_bound_transcript_without_a_user_turn_is_kept(tmp_path: Path) -> None:
    """FAIL CLOSED: unprovable is not the same as proven empty.

    A transcript longer than the scan window whose head holds no user turn
    could still contain one past the bound. That population is tiny and
    strange; deleting a real conversation over it is not a trade worth making.
    """
    filler = {"kind": "custom", "payload": {"custom_type": "x", "pad": "y" * 500}}
    rows = [filler] * (UNUSED_SCAN_CHARS // 400)
    directory = _session(tmp_path, "toolong00001", rows)
    assert (directory / "transcript.jsonl").stat().st_size > UNUSED_SCAN_CHARS

    verdict, reason = _is_unused_session(directory, time.time())

    assert verdict is False
    assert "fail closed" in reason


def test_an_unreadable_transcript_is_kept(tmp_path: Path, monkeypatch) -> None:
    """A file that could not be opened is not evidence of anything."""
    directory = _session(tmp_path, "unreadable01", _machine_rows())

    def _boom(*_args, **_kwargs):
        raise OSError("permission denied")

    monkeypatch.setattr(Path, "open", _boom)

    verdict, reason = _is_unused_session(directory, time.time())

    assert verdict is False
    assert "fail closed" in reason


def test_the_reap_removes_only_the_junk(tmp_path: Path) -> None:
    """End to end over a store holding one of each population."""
    junk = _session(tmp_path, "junk00000002", _machine_rows())
    real = _session(
        tmp_path,
        "real00000002",
        [{"kind": "message", "payload": {"role": "user", "text": "hi"}}],
    )
    fresh = _session(tmp_path, "fresh0000002", _machine_rows(), age=time.time())

    result = reap_unused_sessions(tmp_path / "sessions")

    assert result.evicted == 1
    assert not junk.exists()
    assert real.exists() and fresh.exists()


def test_a_dry_run_reports_without_deleting(tmp_path: Path) -> None:
    """How this was validated against a copy of a real store before it ran."""
    junk = _session(tmp_path, "junk00000003", _machine_rows())

    result = reap_unused_sessions(tmp_path / "sessions", dry_run=True)

    assert result.evicted == 1
    assert junk.exists(), "a dry run must not delete anything"


def test_the_live_session_is_never_reaped(tmp_path: Path) -> None:
    """The caller's own directory, which it may have created seconds ago."""
    mine = _session(tmp_path, "mine00000001", _machine_rows())

    result = reap_unused_sessions(tmp_path / "sessions", live_dir=mine)

    assert result.evicted == 0
    assert mine.exists()


def test_a_missing_store_is_a_no_op(tmp_path: Path) -> None:
    """First run of a fresh install: no sessions directory yet."""
    result = reap_unused_sessions(tmp_path / "sessions")

    assert result.evicted == 0 and result.errors == 0
