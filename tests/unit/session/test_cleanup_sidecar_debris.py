"""A removal takes the session's OUTSIDE sidecars with it, and only its own.

WHY THIS TEST EXISTS. ``cache/session_files/<id>.json`` and
``wakes/spooled/<id>.json`` + ``.<id>.lock`` are written OUTSIDE the session
directory on purpose (a sidecar inside it would move the directory's mtime and
reorder the recency-ranked session list), so no removal ever contained them and
none ever took them. Measured on the reference host 2026-10-29: 6,767 cache
entries across 3,442 session directories and 4,261 spool locks with ZERO spooled
rows — files keyed to conversations that no longer exist, accumulating forever.

The rule these tests pin is deliberately narrow: a removal clears the sidecars
keyed to THE SESSION IT REMOVED. A sidecar whose session directory still exists
is somebody's live cache or live spooled obligation and is never touched — that
is the failure mode a directory-scanning sweep would have, which is why the fix
clears at removal instead.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from local_operator.session.cleanup import (
    hand_off_session_dir,
    mark_store,
    remove_session_dir,
)
from local_operator.session_files import ledger as session_files_ledger
from local_operator.wakes import spooled as spooled_store
from local_operator.wakes.lock import WakeWriteLock


def _store(root: Path, *session_ids: str) -> Path:
    """A marked store with one real-looking directory per id; returns ``sessions/``."""
    sessions = root / "sessions"
    for session_id in session_ids:
        target = sessions / session_id
        target.mkdir(parents=True)
        (target / "transcript.jsonl").write_text('{"role": "user"}\n', encoding="utf-8")
    mark_store(sessions)
    return sessions


def _sidecars(config_dir: Path, session_id: str) -> dict[str, Path]:
    """Write the three outside sidecars through the real writers; return their paths."""
    cache = session_files_ledger.cache_path(config_dir, session_id)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text('{"session": "%s"}' % session_id, encoding="utf-8")
    assert spooled_store.note_spooled_turn(config_dir, session_id) is True
    row = spooled_store.spooled_path(config_dir, session_id)
    lock = spooled_store.spooled_dir(config_dir) / spooled_store.lock_name(session_id)
    assert row.is_file() and lock.is_file(), (row, lock)
    return {"cache": cache, "row": row, "lock": lock}


def test_a_removal_clears_the_removed_sessions_sidecars(tmp_path: Path) -> None:
    """The reproduction: before the fix all three outlived the directory."""
    sessions = _store(tmp_path, "abc")
    sidecars = _sidecars(tmp_path, "abc")

    assert remove_session_dir(
        sessions / "abc", config_dir=tmp_path, policy="p", reason="r", actor="test"
    )
    assert not (sessions / "abc").exists()
    for name, path in sidecars.items():
        assert not path.exists(), f"the {name} sidecar outlived the session: {path}"


def test_a_different_sessions_sidecars_are_left_alone(tmp_path: Path) -> None:
    """THE BOUND: a sidecar whose session directory still exists is never removed.

    This is what separates clearing-at-removal from a sweep over every file in the
    sidecar directories: "abc" going away says nothing about "def", whose cache is
    the answer to a live conversation's next listing.
    """
    sessions = _store(tmp_path, "abc", "def")
    kept = _sidecars(tmp_path, "def")
    _sidecars(tmp_path, "abc")

    assert remove_session_dir(
        sessions / "abc", config_dir=tmp_path, policy="p", reason="r", actor="test"
    )
    assert (sessions / "def").is_dir()
    for name, path in kept.items():
        assert path.exists(), f"the live session's {name} sidecar was removed: {path}"


def test_a_held_lock_is_left_in_place(tmp_path: Path) -> None:
    """A CONTENDED spool is not touched (``wakes/lock.py``'s own rule).

    A waiter holding the old inode and a new writer creating a fresh one are two
    holders of "the" lock, so the file is only unlinked while this call holds it.
    A busy one belongs to a process still finishing a write, and the spooled row
    behind it is guarded by the same lock — so the row is left too, which is the
    correct answer rather than a compromise: mutating an obligation while its
    writer holds the mutex is what the mutex exists to prevent. Both leftovers are
    inert once the directory is gone (``arm._apply`` refuses a session directory
    that is not there) and the supervisor's own reconcile pass drops a row whose
    session is gone, so the residue is bounded to a lock contended at the exact
    moment of removal.
    """
    sessions = _store(tmp_path, "abc")
    sidecars = _sidecars(tmp_path, "abc")

    holder = WakeWriteLock(
        spooled_store.spooled_dir(tmp_path),
        name=spooled_store.lock_name("abc"),
        timeout_s=0.0,
    )
    with holder:
        assert remove_session_dir(
            sessions / "abc", config_dir=tmp_path, policy="p", reason="r", actor="test"
        )
        assert sidecars["lock"].exists(), "a held lock must not be unlinked from under its holder"
        assert sidecars["row"].exists(), "a held lock's row must not be cleared under it"
    # The removal itself still happened, and the cache entry — which no lock guards
    # — is still cleared: a contended spool is not a reason to leak the scan cache.
    assert not (sessions / "abc").exists()
    assert not sidecars["cache"].exists()


def test_a_removal_with_no_sidecars_still_works(tmp_path: Path) -> None:
    """The ordinary case — a session nothing ever scanned or spooled — is unchanged."""
    sessions = _store(tmp_path, "abc")

    assert remove_session_dir(
        sessions / "abc", config_dir=tmp_path, policy="p", reason="r", actor="test"
    )
    assert not (sessions / "abc").exists()


def test_a_mesh_handoff_clears_the_sidecars_too(tmp_path: Path) -> None:
    """The other exit: a committed move takes the session out of THIS store.

    ``hand_off_session_dir`` renames into ``network/handed-away/`` and deletes
    there; it never calls :func:`remove_session_dir`, so without its own call the
    move would leak exactly the debris a removal no longer does — the session has
    left this device either way.
    """
    sessions = _store(tmp_path, "abc")
    sidecars = _sidecars(tmp_path, "abc")

    outcome = hand_off_session_dir(
        sessions / "abc", config_dir=tmp_path, reason="move", actor="test"
    )

    assert outcome.handed is True, outcome.reason
    assert not (sessions / "abc").exists()
    for name, path in sidecars.items():
        assert not path.exists(), f"the moved session's {name} sidecar stayed behind: {path}"


@pytest.mark.parametrize("session_id", ["abc"])
def test_the_sidecar_clearing_is_bounded_to_three_named_files(
    tmp_path: Path, session_id: str
) -> None:
    """Nothing else in either sidecar directory is touched.

    A neighbouring file in ``cache/session_files/`` (another subsystem's, a
    foreign id's) must survive a removal, so the clearing cannot be read as "empty
    the directory".
    """
    sessions = _store(tmp_path, session_id)
    _sidecars(tmp_path, session_id)
    neighbour_cache = session_files_ledger.cache_path(tmp_path, "someone-else")
    neighbour_cache.write_text("{}", encoding="utf-8")
    neighbour_lock = spooled_store.spooled_dir(tmp_path) / spooled_store.lock_name("someone-else")
    neighbour_lock.write_text("", encoding="utf-8")

    assert remove_session_dir(
        sessions / session_id, config_dir=tmp_path, policy="p", reason="r", actor="test"
    )
    assert neighbour_cache.exists()
    assert neighbour_lock.exists()
