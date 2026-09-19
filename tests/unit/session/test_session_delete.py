"""Explicit deletion: what it removes, what it refuses, and what it leaves alone.

``delete_session`` is the only entry point besides the automatic policy, and it
goes through ``cleanup.remove_session_dir`` — the one ``rmtree`` of a session
directory in this codebase, and the thing ``test_no_session_deletion.py`` walks
the package to enforce. These tests are about the DECISIONS around that call:

* **It removes exactly the addressed directory.** Subagent runs a conversation
  launched live as siblings under ``sessions/`` and are NOT touched. The count
  travels on the outcome so a receipt can say so, and the fixture below proves
  the children are still on disk afterwards — because the natural assumption to
  make about a delete is the opposite one.
* **Every hard guard holds, and the refusal NAMES the guard.** A running
  session, an armed wake and unread spooled mail have three different remedies,
  so one blanket "in use" would send the user to stop a session that was never
  running.
* **``RECENT_KEEP`` does not apply, and neither does the enabled switch.** That
  constant bounds the AUTOMATIC sweep; a person who names a conversation and
  confirms a typed ``yes`` has answered the question it is a proxy for. The
  switch gates the reapers the module docstring's incident is about, and a user
  with cleanup off must still be able to delete a conversation.
* **The stores that mention sessions need no cooperation.** The pin and archive
  indexes prune at read, and the search cache is keyed by ids a caller lists —
  asserted here rather than assumed, because "deletion needs nothing from them"
  is a claim about two other modules.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from local_operator.resume import ORIGIN_SUBAGENT, mark_session_origin
from local_operator.session.archived import read_archived, set_archived
from local_operator.session.cleanup import (
    CLEANUP_LOG_NAME,
    EXPLICIT_DELETE_POLICY,
    delete_session,
    mark_store,
)
from local_operator.session.retention import LIVE_MARKER_NAME
from local_operator.session.search_index import build_index

A = "a" * 12
B = "b" * 12
CHILD = "c" * 12


def _store(tmp_path: Path) -> Path:
    """A MARKED store, which is what ``remove_session_dir`` requires.

    Marked through the real ``mark_store`` rather than by writing the marker
    file, because the marker's name and shape is what the removal policy reads.
    """
    sessions = tmp_path / "sessions"
    sessions.mkdir(parents=True, exist_ok=True)
    mark_store(sessions)
    return sessions


def _session(tmp_path: Path, session_id: str, *, content: str | None = None) -> Path:
    directory = tmp_path / "sessions" / session_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "transcript.jsonl").write_text(
        json.dumps(
            {
                "type": "message",
                "payload": {"role": "user", "content": content or f"about {session_id}"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return directory


def _roster(directory: Path, children: int) -> None:
    """A roster sidecar in the shape the runtime writes, with ``children`` records."""
    (directory / "subagent-roster.v1.json").write_text(
        json.dumps(
            {
                "version": 1,
                "generation": 2,
                "jobs": [],
                "records": [{"session_id": f"{index:012x}"} for index in range(children)],
            }
        ),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# What it removes
# ---------------------------------------------------------------------------


def test_a_deletion_removes_exactly_the_addressed_session(tmp_path: Path) -> None:
    _store(tmp_path)
    victim = _session(tmp_path, A)
    neighbour = _session(tmp_path, B)

    outcome = delete_session(tmp_path, A, actor="test")

    assert outcome.found is True and outcome.deleted is True and outcome.refusal == ""
    assert not victim.exists()
    assert neighbour.is_dir()


def test_a_delegated_run_the_conversation_launched_is_kept(tmp_path: Path) -> None:
    """The blast radius, stated by the machine rather than by a comment.

    The child is a sibling directory, so nothing here had to preserve it — but
    "nothing here removes it" is the property a receipt claims on the user's
    behalf, and the count that receipt prints comes from the roster sidecar the
    runtime writes.
    """
    _store(tmp_path)
    parent = _session(tmp_path, A)
    child = _session(tmp_path, CHILD)
    mark_session_origin(child, ORIGIN_SUBAGENT)
    _roster(parent, 2)

    outcome = delete_session(tmp_path, A, actor="test")

    assert outcome.deleted is True
    assert outcome.children == 2
    assert child.is_dir(), "a delegated run is not part of this blast radius"
    assert not parent.exists()


def test_an_unreadable_roster_reports_no_children_rather_than_failing(tmp_path: Path) -> None:
    """Best-effort in the safe direction: a low count cannot widen the blast radius."""
    _store(tmp_path)
    victim = _session(tmp_path, A)
    (victim / "subagent-roster.v1.json").write_text("{ truncated", encoding="utf-8")

    outcome = delete_session(tmp_path, A, actor="test")
    assert outcome.deleted is True and outcome.children == 0


def test_the_removal_is_recorded_under_its_own_policy(tmp_path: Path) -> None:
    """The cleanup log is the only durable record either kind of removal leaves."""
    _store(tmp_path)
    _session(tmp_path, A)

    delete_session(tmp_path, A, actor="tui")

    records = [
        json.loads(line)
        for line in (tmp_path / "sessions" / CLEANUP_LOG_NAME).read_text().splitlines()
        if line.strip()
    ]
    assert records[-1]["session"] == A
    assert records[-1]["policy"] == EXPLICIT_DELETE_POLICY
    assert records[-1]["actor"] == "tui"


def test_a_dry_run_reports_the_decision_and_removes_nothing(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The confirmation path in both frontends is this call, so it must be honest.

    ``deleted=True`` on a dry run means WOULD have been removed — the contract
    ``remove_session_dir`` already has — and nothing is written: no log record
    and no WARNING, so a rehearsal does not look like an event in the log.
    """
    _store(tmp_path)
    _session(tmp_path, A)

    with caplog.at_level("WARNING"):
        outcome = delete_session(tmp_path, A, actor="test", dry_run=True)

    assert outcome.found is True and outcome.deleted is True
    assert (tmp_path / "sessions" / A).is_dir()
    assert not (tmp_path / "sessions" / CLEANUP_LOG_NAME).exists()
    assert not [record for record in caplog.records if record.levelname == "WARNING"]


# ---------------------------------------------------------------------------
# What it refuses
# ---------------------------------------------------------------------------


def _claim(directory: Path) -> None:
    (directory / LIVE_MARKER_NAME).write_text(str(os.getpid()), encoding="utf-8")


def _lease(directory: Path) -> None:
    (directory / ".execution-lease").write_text(
        json.dumps({"generation": "g", "pid": os.getpid()}), encoding="utf-8"
    )


def _wake(tmp_path: Path, session_id: str) -> Path:
    from local_operator.wakes.store import entry_path

    path = entry_path(tmp_path, session_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}", encoding="utf-8")
    return path


def _mail(directory: Path) -> None:
    (directory / "inbox.jsonl").write_text('{"from":"peer"}\n', encoding="utf-8")


@pytest.mark.parametrize(
    "shape,expected",
    [
        ("claimed", "open in a running session"),
        ("leased", "open in a running session"),
        ("wake", "has a wake armed for it"),
        ("mail", "has unread messages waiting"),
    ],
)
def test_a_hard_guard_refuses_and_names_itself(tmp_path: Path, shape: str, expected: str) -> None:
    _store(tmp_path)
    victim = _session(tmp_path, A)
    if shape == "claimed":
        _claim(victim)
    elif shape == "leased":
        _lease(victim)
    elif shape == "wake":
        _wake(tmp_path, A)
    else:
        _mail(victim)

    outcome = delete_session(tmp_path, A, actor="test")

    assert outcome.found is True
    assert outcome.deleted is False
    assert expected in outcome.refusal, outcome.refusal
    assert victim.is_dir(), "a refusal must not have removed anything"
    assert not (tmp_path / "sessions" / CLEANUP_LOG_NAME).exists()


def test_a_guard_that_cannot_be_evaluated_keeps_the_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fail closed, and say WHICH uncertainty rather than naming a condition.

    Telling a user to stop a session that is not running is a remedy for
    something that did not happen; the fallback sentence says the check could
    not be made instead.
    """
    from local_operator.session import cleanup

    _store(tmp_path)
    victim = _session(tmp_path, A)

    def explode(*args: object, **kwargs: object) -> bool:
        raise RuntimeError("probe unavailable")

    monkeypatch.setattr(cleanup, "_claimed", explode)
    outcome = delete_session(tmp_path, A, actor="test")
    assert outcome.deleted is False
    assert "could not be checked" in outcome.refusal
    assert victim.is_dir()


def test_a_delegated_run_is_not_deletable_through_this_path(tmp_path: Path) -> None:
    """An id the user cannot see is answered as unknown, not deleted.

    Deleting is irreversible and a subagent run is not a conversation anyone
    opened, so the admission is stricter than the pin and archive routes' — the
    asymmetry is argued in ``delete_session``'s docstring.
    """
    _store(tmp_path)
    child = _session(tmp_path, CHILD)
    mark_session_origin(child, ORIGIN_SUBAGENT)

    outcome = delete_session(tmp_path, CHILD, actor="test")

    assert outcome.found is False and outcome.deleted is False
    assert child.is_dir()


@pytest.mark.parametrize("session_id", ["nope", "../agents", "", ".", "a/b", "/tmp"])
def test_an_unknown_or_malformed_id_is_not_found(tmp_path: Path, session_id: str) -> None:
    _store(tmp_path)
    outcome = delete_session(tmp_path, session_id, actor="test")
    assert outcome.found is False and outcome.deleted is False


def test_an_unmarked_store_is_refused_with_a_sentence(tmp_path: Path) -> None:
    """``remove_session_dir`` declines; the caller must not read that as success."""
    (tmp_path / "sessions").mkdir(parents=True, exist_ok=True)
    _session(tmp_path, A)

    outcome = delete_session(tmp_path, A, actor="test")

    assert outcome.found is True and outcome.deleted is False
    assert outcome.refusal, "a declined removal is a refusal, not a silent no-op"
    assert (tmp_path / "sessions" / A).is_dir()


# ---------------------------------------------------------------------------
# The policy limits it is NOT bound by
# ---------------------------------------------------------------------------


def test_recent_keep_does_not_protect_a_session_from_an_explicit_delete(tmp_path: Path) -> None:
    """The recent-N rule bounds the SWEEP. A confirmed delete has answered it."""
    from local_operator.session.cleanup import RECENT_KEEP, CleanupPolicy, run_cleanup

    _store(tmp_path)
    for index in range(RECENT_KEEP + 2):
        _session(tmp_path, f"{index:012x}")
    newest = f"{RECENT_KEEP + 1:012x}"

    # The automatic policy refuses to touch it...
    policy = CleanupPolicy(enabled=True, max_sessions=1)
    plan = run_cleanup(tmp_path, policy, dry_run=True)
    assert newest not in {candidate.session for candidate in plan.removed}

    # ...and the explicit path removes it.
    assert delete_session(tmp_path, newest, actor="test").deleted is True
    assert not (tmp_path / "sessions" / newest).exists()


def test_the_enabled_switch_does_not_gate_an_explicit_delete(tmp_path: Path) -> None:
    """Cleanup is off by default, and the user must still be able to delete.

    The switch exists because a REAPER once removed 225 of an operator's 244
    conversations; it is not consent for a conversation the user names.
    """
    _store(tmp_path)
    _session(tmp_path, A)
    # No config file at all: the default is disabled.
    assert delete_session(tmp_path, A, actor="test").deleted is True
    assert not (tmp_path / "sessions" / A).exists()


# ---------------------------------------------------------------------------
# What it leaves consistent
# ---------------------------------------------------------------------------


def test_the_pin_and_archive_stores_report_nothing_for_a_deleted_session(
    tmp_path: Path,
) -> None:
    """Both prune at READ, so deletion needs no cooperation from either.

    Nothing writes to them here and this test does not either: the entries are
    left on disk deliberately, because a write-time prune is what would couple
    ``cleanup`` to two other modules.
    """
    from local_operator.tui.sidebar_pins import read_pins, set_pin

    _store(tmp_path)
    _session(tmp_path, A)
    set_pin(tmp_path, A, True)
    set_archived(tmp_path, A, True)
    assert read_pins(tmp_path) == [A] and read_archived(tmp_path) == [A]

    assert delete_session(tmp_path, A, actor="test").deleted is True

    assert read_pins(tmp_path) == []
    assert read_archived(tmp_path) == []
    # The records themselves are untouched: the prune is a read.
    assert json.loads((tmp_path / "sidebar-pins.json").read_text()) == [A]
    assert json.loads((tmp_path / "archived-sessions.json").read_text()) == [A]


def test_the_wake_entry_is_pruned_when_the_deletion_happens(tmp_path: Path) -> None:
    """Unreachable while the wake guard holds, and still done.

    The guard is one config write away from being bypassed by hand, and a wake
    entry whose session is gone is an index pointing at nothing. Stubbed at the
    guard so the prune is what is under test rather than the refusal above it.
    """
    from local_operator.session import cleanup

    _store(tmp_path)
    _session(tmp_path, A)
    entry = _wake(tmp_path, A)

    original = cleanup._has_wake
    cleanup._has_wake = lambda config_dir, session: False
    try:
        assert delete_session(tmp_path, A, actor="test").deleted is True
    finally:
        cleanup._has_wake = original

    assert not entry.exists()


def test_a_stale_search_cache_entry_is_inert_after_a_deletion(tmp_path: Path) -> None:
    """The index is built from ids a caller LISTS, and the id is gone from the store."""
    _store(tmp_path)
    _session(tmp_path, A)
    _session(tmp_path, B)
    assert build_index(tmp_path, [A, B]) != {}

    assert delete_session(tmp_path, A, actor="test").deleted is True

    assert A not in build_index(tmp_path, [A, B])
    assert build_index(tmp_path, [A]) == {}
