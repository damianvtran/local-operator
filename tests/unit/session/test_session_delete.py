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


def _dormant_wake(tmp_path: Path, session_id: str) -> Path:
    """The entry ``/stop`` leaves: schedules kept, ``stopped_at`` stamped.

    Written through the same call the stop path makes — ``write_entry`` with
    ``preserve`` carrying the marker (see ``control._mark_wakes_dormant``) — so
    the fixture is the shape the product writes rather than a hand-built file
    that could agree with a wrong idea of what dormancy is.
    """
    from local_operator.wakes.store import write_entry

    schedule = {"id": "w1", "message": "check the parser fix", "next_due_at": 4_102_444_800_000}
    path = write_entry(
        tmp_path,
        session_id,
        cwd=str(tmp_path),
        schedules=[schedule],
        preserve={"stopped_at": 1_700_000_000_000},
    )
    assert path is not None
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
    """A deleted session's wake entry goes with it.

    REACHABLE since UX round 1 (U2), and that is why this no longer stubs the
    guard: a DORMANT entry (the ``stopped_at`` marker ``/stop`` leaves) does not
    refuse, so the real path removes the directory AND drops the index entry. An
    entry whose session is gone is an index pointing at nothing — the `lop wake`
    ghost row.
    """
    _store(tmp_path)
    _session(tmp_path, A)
    entry = _dormant_wake(tmp_path, A)

    assert delete_session(tmp_path, A, actor="test").deleted is True

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


# ---------------------------------------------------------------------------
# The wake guard asks whether a wake can FIRE (UX round 1, U1 and U2)
# ---------------------------------------------------------------------------


def test_a_dormant_wake_does_not_block_the_delete(tmp_path: Path) -> None:
    """A wake the product has put to sleep is not pending work (UX U2).

    The guard used to ask whether the entry FILE exists, and `/stop` never deletes
    one — it stamps ``stopped_at``, which the supervisor skips in every path it
    has. So a conversation in which the user had ever set a reminder became
    permanently undeletable from the moment they stopped it: `/delete` was refused
    with a sentence about a wake that cannot fire, and the two-step flow the design
    record documents as reachable (`/stop`, then `/delete`) was refused too.

    On the head this test was written against, this call returns
    ``deleted=False`` with "That conversation has a wake armed for it." — the
    marker is what discriminates, and the armed case below is the control.
    """
    _store(tmp_path)
    _session(tmp_path, A)
    _dormant_wake(tmp_path, A)

    outcome = delete_session(tmp_path, A, actor="test")

    assert outcome.deleted is True, outcome.refusal
    assert not (tmp_path / "sessions" / A).exists()


def test_an_armed_wake_still_refuses_and_names_a_remedy_that_exists(tmp_path: Path) -> None:
    """The other half of U2, plus U1's copy.

    An ARMED entry is a schedule that will fire, so it is genuinely pending work
    and the delete stays refused. What UX round 1 (U1) found is that the sentence
    named an action with no door: the composer offers no wake command, the wake
    band has no cancel, and `lop wake`'s own copy says there is no cancel. The
    sentence must name the two actions that DO exist — ask the conversation, or
    delete the index entry file, the same remedy the CLI's ghost row names.
    """
    _store(tmp_path)
    _session(tmp_path, A)
    _wake(tmp_path, A)

    outcome = delete_session(tmp_path, A, actor="test")

    assert outcome.deleted is False
    assert "has a wake armed for it" in outcome.refusal
    # THE DOOR IS ADDRESSABLE AND STORE-RELATIVE (design round 2, D8; desktop QA
    # round 4, Q14): a literal ``<session-id>`` template told the user to go and
    # find a filename, and resolving it to a machine-absolute path leaked this
    # host's layout inside a dialog for a conversation the user is looking at.
    # The relative form is the one the CLI's ghost row uses, and the id is still
    # resolved so the file can be found without hunting.
    assert f"wakes/{A}.json" in outcome.refusal, outcome.refusal
    assert "<session-id>" not in outcome.refusal, outcome.refusal
    assert str(tmp_path) not in outcome.refusal, "the host's layout is not the user's business"
    assert "/Users" not in outcome.refusal
    assert "Cancel the wake" not in outcome.refusal, (
        "that action has no surface reachable from the terminal that printed the "
        "sentence: " + outcome.refusal
    )
    assert (tmp_path / "sessions" / A).is_dir()


def test_every_refusal_names_the_conversation_and_never_a_host_path() -> None:
    """THE REFUSAL SET, READ AS A SET (desktop QA round 4, Q14).

    Five sentences can stand in front of an irreversible act — claimed, leased, an
    armed wake, unread mail, and the fallback for a guard that could not be
    evaluated — and each is read by a user looking at one conversation. Two rules
    bind all of them, and they were found one at a time rather than together: no
    template placeholders (D8: the app holds the id, so ``<session-id>`` asked the
    user to go and find a filename) and no machine-absolute paths (Q14: a dialog
    is not the place to learn this host's directory layout). Every one names the
    conversation, which is the thing the user has.
    """
    import re

    from local_operator.session.cleanup import _GUARD_REFUSALS, _GUARD_REFUSAL_FALLBACK

    sentences = list(_GUARD_REFUSALS.values()) + [_GUARD_REFUSAL_FALLBACK]
    assert len(sentences) == 5, "the set this rule is written for"
    for sentence in sentences:
        assert "conversation" in sentence, sentence
        assert "<" not in sentence and ">" not in sentence, sentence
        assert not re.search(r"\s/[A-Za-z]", sentence), f"absolute path in: {sentence}"
        assert str(Path("/Users")) not in sentence


def test_the_rehearsal_sentence_has_exactly_one_source() -> None:
    """R3-2: the sentence cannot drift host by host any more.

    It was retyped verbatim in ``tui/app.py`` twice and in
    ``session/runtime/serving.py`` once — ``label`` was shared, the sentence
    around it was not — so a wording edit in one host would have given the
    terminal and the detached runtime different confirmations for the same
    irreversible act, and only the local host had a test that would have noticed.
    It lives on ``DeleteOutcome.rehearsal`` now, and this walk is the half of the
    pin that keeps it there.

    HONEST ABOUT ITS REACH: the local host's execution is covered by
    ``tests/unit/tui/test_session_archive_commands.py`` (the real composer); the
    other two hosts are pinned structurally — they must ASK for the sentence and
    must not carry it — because driving the detached runtime end to end costs a
    process and this class of drift is a copy-paste, which is what a walk sees.
    """
    import local_operator
    from local_operator.session.cleanup import DeleteOutcome

    package = Path(local_operator.__file__).parent
    for phrase in (
        "/delete removes",
        "and its transcript — it cannot be",
        "it started are kept.",
    ):
        holders = [
            path.relative_to(package).as_posix()
            for path in package.rglob("*.py")
            if phrase in path.read_text(encoding="utf-8", errors="ignore")
        ]
        assert holders == ["session/cleanup.py"], f"{phrase!r} is retyped in {holders}"

    app = (package / "tui" / "app.py").read_text(encoding="utf-8")
    serving = (package / "session" / "runtime" / "serving.py").read_text(encoding="utf-8")
    assert app.count("outcome.rehearsal()") == 2, "both TUI hosts must ask the outcome"
    assert serving.count("outcome.rehearsal()") == 1, "the runtime must ask the outcome"
    assert isinstance(DeleteOutcome(session_id="a", found=True, deleted=False), DeleteOutcome)


def test_the_rehearsal_states_permanence_once_and_names_the_children() -> None:
    """D10 and the children clause, both of which are now this one sentence's.

    "for good — it cannot be undone" said the same thing twice in one breath and
    wrapped ``for good`` across lines at the standard width (design round 2,
    D10); irreversibility is the part a user must take away, so it is the part
    that stays. The children clause is the fact that says what the deletion does
    NOT touch, and it has to be part of the same sentence rather than a
    separately-built tail in each host.
    """
    from local_operator.session.cleanup import DeleteOutcome

    plain = DeleteOutcome(
        session_id="a" * 12, found=True, deleted=False, label="“Retention sweep” (aaaaaaaaaaaa)"
    ).rehearsal()
    assert "it cannot be undone" in plain
    assert "for good" not in plain, "permanence is stated once"
    assert "are kept." not in plain, "no children, no clause"

    with_kids = DeleteOutcome(
        session_id="a" * 12,
        found=True,
        deleted=False,
        label="“Retention sweep” (aaaaaaaaaaaa)",
        children=2,
    ).rehearsal()
    # The clause sits INSIDE the sentence, between the act and the confirmation,
    # so a host cannot append it in a different place or forget it.
    assert (
        "it cannot be undone. 2 subagent run(s) it started are kept. "
        "Run /delete yes to confirm." in with_kids
    )


def test_an_unreadable_wake_entry_keeps_the_session(tmp_path: Path) -> None:
    """FAIL CLOSED, which is the property the rename could have lost.

    The dormancy check reads and parses the entry, so a file that cannot be parsed
    is a file whose state is unknown — and a guard that answered "not armed" there
    would delete a conversation that may have a live schedule. ``store.read_entry``
    is deliberately not used for this: it treats an unreadable file as absent for
    display's sake.
    """
    _store(tmp_path)
    _session(tmp_path, A)
    _wake(tmp_path, A).write_text("{not json at all", encoding="utf-8")

    outcome = delete_session(tmp_path, A, actor="test")

    assert outcome.found is True
    assert outcome.deleted is False
    assert "has a wake armed for it" in outcome.refusal, outcome.refusal
    assert (tmp_path / "sessions" / A).is_dir()


def test_the_unread_mail_refusal_names_how_to_read_them(tmp_path: Path) -> None:
    """UX U4: "read them" named no way to read them.

    The spool drains once, at open (``inbox.drain_inbox``), and no command reads
    another conversation's inbox — so reopening that conversation IS the action,
    and the sentence has to say so.
    """
    _store(tmp_path)
    victim = _session(tmp_path, A)
    _mail(victim)

    outcome = delete_session(tmp_path, A, actor="test")

    assert outcome.deleted is False
    assert "Reopen it to read them" in outcome.refusal, outcome.refusal


def test_the_delete_sentence_names_the_conversation_by_title(tmp_path: Path) -> None:
    """Design round 1 (D2): the rehearsal named a handle that is not on screen.

    The id resolves, but the screen the rehearsal is typed into shows the model and
    the cwd — the id appears only as a dim column inside ``/resume``. So the one
    check a rehearsal exists to enable ("is this the conversation I mean?") had
    nothing to check against, and in the round-1 frame the fixture's id rendered as
    the word ``sess``, reading as a truncation. The label leads with the title.
    """
    from local_operator.resume import write_session_title
    from local_operator.session.cleanup import session_label

    _store(tmp_path)
    directory = _session(tmp_path, A)
    write_session_title(
        directory, "Parser crash on nested frontmatter", user_set=True, past_names=[]
    )

    assert session_label(directory) == f"“Parser crash on nested frontmatter” ({A})"

    outcome = delete_session(tmp_path, A, actor="test", dry_run=True)
    assert outcome.label == f"“Parser crash on nested frontmatter” ({A})"


def test_a_conversation_with_no_title_is_named_as_this_conversation(tmp_path: Path) -> None:
    """No title means no guessed title — the lists name it nothing either.

    ``Untitled conversation`` would be a name the user never chose and cannot
    search for; the id is the only handle that exists, so it stays.
    """
    from local_operator.session.cleanup import session_label

    _store(tmp_path)
    directory = _session(tmp_path, A)

    assert session_label(directory) == f"this conversation ({A})"
