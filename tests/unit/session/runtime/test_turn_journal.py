"""The runtime's own account of a death: a turn journal and a boot record.

WHY THESE TESTS EXIST. On 2026-09-15 a fleet outage killed 36 session runtimes
at 19:41 and every one of them recorded the SAME anonymous verdict, because no
artifact on disk was authored by the process that died. The tests below pin the
three things that fix that, and they are deliberately arranged so that each one
fails if the responsible half is removed:

* a killed runtime leaves an OPEN row naming the turn that was in flight — a
  REAL process, SIGKILLed by exact pid, read back from disk by another process;
* the classifier PREFERS the named cause that evidence supports (an escalated
  sweep, an install-window tear, a crash) and falls back to the legacy rungs
  unchanged when the evidence is absent;
* the interruption reaches the RESUMED session's own context exactly once, and
  re-runs nothing.

The inertness half — that none of this moved ``_should_exit`` or a grace
constant — is pinned in ``test_residency_guard.py``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from local_operator import update
from local_operator.incidents import STALL_BOUND_CAUSE
from local_operator.session.attention import _classify_orphaned_run
from local_operator.session.runtime import journal, registry, stall_watchdog
from local_operator.session.runtime.types import (
    BUILD_DRAIN_OVERDUE_CAUSE,
    BUILD_DRAIN_PROGRESS_S,
    bound_text,
)

#: A child that opens a real journal row, records a real tool boundary, writes
#: its boot record, and then blocks. The test SIGKILLs it: nothing in this
#: module's subject runs on the child's exit path, which is the point — the
#: artifact has to stand on its own after a process that recorded nothing at
#: exit.
_CHILD_SCRIPT = """
import sys, time
from pathlib import Path
from local_operator.session.runtime import journal
from local_operator.update import installed_build

directory = Path(sys.argv[1])
session_id = sys.argv[2]
build = installed_build()
journal.write_boot_record(session_id, build)
writer = journal.TurnJournal(directory, session_id, build)
writer.open_turn(command_id="cmd-1")
writer.note_boundary("bash")
print("ready", flush=True)
time.sleep(60)
"""


def _isolated_env(home: Path) -> dict[str, str]:
    """A child environment with every inherited harness variable stripped.

    ``CMUX_*`` because an inherited workspace id lets a test rename the
    operator's real cmux workspaces; ``LOP_*`` because an inherited
    ``LOP_RUNTIME_*``/``LOP_MOBILE_CHILD_*`` pins a spawned runtime's session and
    provider, so a cell would exercise the operator's own session instead of its
    own. Both families are stripped rather than a few names, so a new one cannot
    quietly re-enter.
    """
    env = {k: v for k, v in os.environ.items() if not k.startswith(("CMUX_", "LOP_"))}
    env["HOME"] = str(home)
    env["LOCAL_OPERATOR_CONFIG_DIR"] = str(home / ".local-operator")
    return env


def _child_config_root(tmp_path: Path) -> Path:
    """The root the spawned child writes its records under.

    Read from the SAME environment the child is spawned with rather than from
    this process's ambient config: the two are different roots by design (that
    is what the isolation is for), and a test that read the ambient one would
    assert about a directory nothing ever wrote to.
    """
    return Path(_isolated_env(tmp_path)["LOCAL_OPERATOR_CONFIG_DIR"])


def _session_directory(root: Path, session_id: str) -> Path:
    """``<root>/sessions/<id>``, the layout ``attention._config_root_for`` reads."""
    directory = root / "sessions" / session_id
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _spawn_mid_turn(directory: Path, session_id: str, tmp_path: Path) -> subprocess.Popen[bytes]:
    """A real process parked with an open journal row.

    Returns the live child; the caller kills it. Nothing here sends a signal to
    anything but this exact pid — a pattern kill on a box holding dozens of live
    agent sessions is the hazard the whole design exists to remove.
    """
    child = subprocess.Popen(
        [sys.executable, "-c", _CHILD_SCRIPT, str(directory), session_id],
        env=_isolated_env(tmp_path),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    assert child.stdout is not None
    line = child.stdout.readline()
    assert line.strip() == b"ready", (line, child.stderr.read() if child.stderr else b"")
    return child


def _kill(child: subprocess.Popen[bytes]) -> int:
    """SIGKILL one pid this test spawned, and reap it."""
    pid = child.pid
    child.kill()
    child.wait(timeout=10)
    return pid


def _pid_from_child() -> int:
    """A pid that has certainly exited — taken from a child we spawned.

    From the kernel rather than invented: writing a fabricated pid into a row
    would make "the owner is gone" a guess about which numbers a test picked,
    and PID reuse would turn it into a silently live owner.
    """
    child = subprocess.Popen(
        [sys.executable, "-c", "pass"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    child.wait(timeout=30)
    return child.pid


# ---------------------------------------------------------------------------
# The row, as the runtime writes it
# ---------------------------------------------------------------------------


def test_a_turn_opens_a_row_and_closes_it(tmp_path: Path) -> None:
    """The row covers the turn: open at the start, closed with a cause at the end."""
    directory = _session_directory(tmp_path, "sess-open-close")
    build = update.BuildStamp(version="0.55.9", source_ref="abc1234def")
    writer = journal.TurnJournal(directory, "sess-open-close", build)

    writer.open_turn(command_id="cmd-7")
    opened = registry.read_turn_journal(directory)
    assert opened is not None
    assert opened["open"] is True
    assert opened["ended_at"] is None
    assert opened["session_id"] == "sess-open-close"
    assert opened["pid"] == os.getpid()
    assert opened["turn_seq"] == 1
    assert opened["command_id"] == "cmd-7"
    # The generation/install root and the build the process loaded, so a
    # successor can say WHAT was running and not merely that something was.
    assert opened["build"] == {"version": "0.55.9", "source_ref": "abc1234def"}
    assert opened["install_root"] == journal._install_root()

    writer.note_boundary("bash")
    bounded = registry.read_turn_journal(directory)
    assert bounded is not None and bounded["last_boundary"] == "bash"

    writer.close_turn("completed")
    closed = registry.read_turn_journal(directory)
    assert closed is not None
    assert closed["open"] is False
    assert closed["ended_at"] is not None
    assert closed["end_cause"] == "completed"

    # A second turn is a NEW row, not an edit of the closed one.
    writer.open_turn()
    second = registry.read_turn_journal(directory)
    assert second is not None
    assert second["turn_seq"] == 2
    assert second["open"] is True


def test_the_row_survives_a_killed_runtime_and_names_the_turn(tmp_path: Path) -> None:
    """THE CASE STUDY'S SHAPE: a real process, SIGKILLed mid-turn.

    Asserted from a DIFFERENT process than the one that wrote it, because that
    is the position every reader of this artifact is actually in — a successor
    runtime, or an operator's shell the next morning.
    """
    session_id = "sess-killed-midturn"
    directory = _session_directory(tmp_path, session_id)
    child = _spawn_mid_turn(directory, session_id, tmp_path)
    killed_pid = _kill(child)

    row = journal.TurnJournalRow.from_json(registry.read_turn_journal(directory))
    assert row is not None
    assert row.open is True, "a killed runtime's row must still say the turn was in flight"
    assert row.pid == killed_pid
    assert row.session_id == session_id
    assert row.turn_seq == 1
    assert row.last_boundary == "bash"
    assert row.end_cause == ""
    assert row.build.get("version") == update.installed_build().version

    # And it reads as a DEATH, not as a live turn: the liveness test is what
    # keeps the other direction honest.
    assert journal.open_row_after_death(directory) is not None

    # The boot record outlives the process too, and names the parent chain the
    # incident investigation could not reconstruct from the corpse.
    record = journal.read_boot_record(killed_pid, root=_child_config_root(tmp_path))
    assert record is not None
    assert record.session_id == session_id
    assert record.kind == journal.BOOT_RECORD_KIND
    assert record.parent_pid == os.getpid(), "the spawned child's parent is this test process"
    assert record.install_root


def test_a_clean_exit_withdraws_the_boot_record(tmp_path: Path) -> None:
    """A surviving record means "stopped without running its own exit ordering"."""
    session_id = "sess-clean-exit"
    directory = _session_directory(tmp_path, session_id)
    build = update.BuildStamp(version="1.0.0", source_ref="aaa")

    record_path = journal.write_boot_record(session_id, build, root=tmp_path)
    assert record_path.parent.name == "host"
    assert journal.read_boot_record(os.getpid(), root=tmp_path) is not None

    journal.clear_boot_record(root=tmp_path)
    assert journal.read_boot_record(os.getpid(), root=tmp_path) is None
    # The row-side artefact is untouched by the record's withdrawal.
    assert registry.read_turn_journal(directory) is None


def test_the_boot_record_is_readable_by_a_successor_for_another_pid(tmp_path: Path) -> None:
    """The missing link in ``app -> backend -> runtime -> runtimes``."""
    session_id = "sess-successor"
    directory = _session_directory(tmp_path, session_id)
    child = _spawn_mid_turn(directory, session_id, tmp_path)
    killed_pid = _kill(child)

    root = _child_config_root(tmp_path)
    record = journal.read_boot_record(killed_pid, root=root)
    assert record is not None, "the successor has to be able to see that this pid existed"
    assert record.kind == journal.BOOT_RECORD_KIND
    assert record.build_stamp() is not None
    # And the STRICTER reader refuses it: a dead pid's row is not this process's
    # evidence (see ``recorded_boot_build``), and the pid alone is a recyclable
    # name.
    assert journal.recorded_boot_build(killed_pid, root=root) is None


def test_the_exit_path_touches_no_filesystem_when_nothing_was_recorded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The instrument must not add work to an exit path that owes it nothing.

    Two reasons, and the second is the sharp one. ``test_process_reaper`` TIMES
    ``_clean_exit`` (it asserts the spread of three elapsed times stays under
    40 ms), so work added there for a host with no record is a timing regression
    on someone else's test. And the withdrawal is not free even when it finds
    nothing: ``registry.unpublish`` resolves ``run_dir()``, which MKDIRS the
    namespace — so an unguarded call makes every clean exit create a directory it
    has nothing to put in.
    """
    from local_operator.session.runtime import process

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setattr(process, "_boot_record_pid", None)

    process._clear_boot_record()

    assert not (
        tmp_path / "cfg" / "run"
    ).exists(), "a process with no boot record must not create (or touch) the namespace"


def test_the_exit_path_withdraws_the_record_this_process_wrote(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """And the positive half: what this process published, it withdraws."""
    from local_operator.session.runtime import process

    root = tmp_path / "cfg"
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    journal.write_boot_record("sess-clean", update.BuildStamp("1.0.0", "aaa"))
    monkeypatch.setattr(process, "_boot_record_pid", os.getpid())
    assert journal.read_boot_record(os.getpid()) is not None

    process._clear_boot_record()

    assert journal.read_boot_record(os.getpid()) is None


# ---------------------------------------------------------------------------
# The WRITER side: what the runtime records, not just what the reader makes of it
# ---------------------------------------------------------------------------


def test_note_exit_normalizes_both_writers_to_one_token(tmp_path: Path) -> None:
    """R3: the two exit writers must land the SAME token.

    ``amain``'s shutdown block passes the signal's own name (``SIGTERM``) while
    the signal drain reaches ``_clean_exit`` with a sentence (``leaving after
    SIGTERM``). The reader keys on the token, so without one vocabulary the
    escalated-sweep rung would hold only by which writer happened to run last —
    and it would answer ``runtime-killed`` ("no stop was asked for") for a
    death where a signal WAS recorded.
    """
    directory = _session_directory(tmp_path, "sess-writer")
    writer = journal.TurnJournal(directory, "sess-writer", update.BuildStamp("1.0.0", "aaa"))
    writer.open_turn(command_id="cmd-9")

    # The drain's spelling.
    writer.note_exit("leaving after SIGTERM")
    row = journal.TurnJournalRow.from_json(registry.read_turn_journal(directory))
    assert row is not None
    assert row.exit_cause == "SIGTERM", row.exit_cause
    assert row.still_open_at_exit is True
    # And the reader agrees, on the row the RUNTIME wrote rather than one a test
    # hand-stamped.
    assert journal.death_verdict(row)[1] == "runtime-shutdown"


def test_a_recorded_signal_survives_a_later_writer_without_one(tmp_path: Path) -> None:
    """R3, the ordering half: a signal is sticky.

    The shutdown block runs after the drain and may have no signal to report
    (``trigger.get("why") or "unknown"``). If that overwrote the token, the same
    escalation would be a named sweep or a crash depending on rung order.
    """
    directory = _session_directory(tmp_path, "sess-sticky")
    writer = journal.TurnJournal(directory, "sess-sticky", update.BuildStamp("1.0.0", "aaa"))
    writer.open_turn()
    writer.note_exit("leaving after SIGKILL")
    writer.note_exit("unknown")

    row = journal.TurnJournalRow.from_json(registry.read_turn_journal(directory))
    assert row is not None
    assert row.exit_cause == "SIGKILL", row.exit_cause
    assert journal.death_verdict(row)[1] == "runtime-shutdown"


def test_a_recorded_bound_outranks_the_install_inference(tmp_path: Path) -> None:
    """Q-2/R3: the row's own last word is narrated ahead of an inference.

    Measured before this: a runtime that cut its turn under the drain's progress
    bound left ``exit_cause`` naming the bound, and the successor narrated
    ``install-mid-update`` instead. That sentence is true — the install HAS moved,
    which is why the drain latched — but it is also the sentence every ordinary
    build handover leaves, so the fact the operator needs (this turn was cut by a
    bound rather than waited out) was invisible in the one account that outlives the
    process. ``lop sessions`` loses the record ~97 ms after the escalation, so that
    row is all there is.
    """
    directory = _session_directory(tmp_path, "sess-overdue")
    writer = journal.TurnJournal(directory, "sess-overdue", update.BuildStamp("1.0.0", "aaa"))
    writer.open_turn(command_id="cmd-over")
    writer.note_exit(BUILD_DRAIN_OVERDUE_CAUSE)

    row = journal.TurnJournalRow.from_json(registry.read_turn_journal(directory))
    assert row is not None
    assert row.exit_cause == BUILD_DRAIN_OVERDUE_CAUSE
    assert journal.install_moved(row), (
        "the fixture must be one where the inference WOULD have won, or this test "
        "proves nothing about the ordering"
    )
    kind, cause, reason = journal.death_verdict(row)
    assert kind == "error"
    assert cause == BUILD_DRAIN_OVERDUE_CAUSE, cause
    assert "no movement" in reason, reason
    assert bound_text(BUILD_DRAIN_PROGRESS_S) in reason, reason


def test_the_killing_token_is_never_answered_by_the_recorded_rung(tmp_path: Path) -> None:
    """MINOR 1 (review round 4): the verdict token must not be its own evidence.

    Rung 2 answers a row's own recorded cause whenever ``CUT_OFF_CAUSES`` knows the
    token, and that table holds the very token the unattributed arm returns —
    ``KILL_CAUSE``, this taxonomy's name for a death nobody recorded an act for.
    Membership alone therefore let such a row be rendered by ITSELF: the sentence a
    runtime is owed when the kill went unattributed, minus the attribution, on the
    arm that cannot name its actor. No writer records the token today (``note_exit``
    is reached only from ``process._clean_exit`` and the direct-dispose path, and
    neither passes it), so this is an invariant held against a future writer rather
    than a live defect — which is exactly why it is pinned here instead of left to
    the reader's prose.

    THE FIXTURE CARRIES NO BUILD STAMP ON PURPOSE. ``install_moved`` (rung 3) answers
    False for a row with no stamp, so rung 2 and the unattributed arm are the only
    rungs left and the assertion is about the exclusion rather than about the rung
    order: with the token admitted to rung 2 the sentence loses its
    ``(unattributed)`` clause, and nothing else in the fixture can supply one.
    """
    from local_operator.incidents import KILL_CAUSE, KILL_UNATTRIBUTED

    directory = _session_directory(tmp_path, "sess-recorded-kill")
    writer = journal.TurnJournal(directory, "sess-recorded-kill")
    writer.open_turn(command_id="cmd-kill")
    writer.note_exit(KILL_CAUSE)

    row = journal.TurnJournalRow.from_json(registry.read_turn_journal(directory))
    assert row is not None
    assert row.exit_cause == KILL_CAUSE, (
        "the writer must record the token verbatim, or this test is about the "
        "normalizer rather than about rung 2"
    )
    assert not journal.install_moved(row), (
        "the fixture must be one no inference claims, or a verdict here would say "
        "nothing about rung 2"
    )

    kind, cause, reason = journal.death_verdict(row)

    assert kind == "error"
    assert cause == KILL_CAUSE, cause
    assert KILL_UNATTRIBUTED in reason, reason


def test_a_non_signal_exit_cause_is_recorded_verbatim(tmp_path: Path) -> None:
    """The normalizer is for signals only: a planned exit keeps its own word."""
    directory = _session_directory(tmp_path, "sess-planned")
    writer = journal.TurnJournal(directory, "sess-planned", update.BuildStamp("1.0.0", "aaa"))
    writer.open_turn()
    writer.note_exit("retiring for 0.56.0")

    row = journal.TurnJournalRow.from_json(registry.read_turn_journal(directory))
    assert row is not None
    assert row.exit_cause == "retiring for 0.56.0"
    # Not a signal, so no sweep claim: the reader must not read "SIG" out of a
    # word that merely starts with those letters.
    assert journal.signal_exit_token("signalled by a peer") == ""
    assert journal.death_verdict(row)[1] in {"runtime-killed", "install-mid-update"}


def test_recorded_boot_build_answers_only_for_this_process(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """R4: the durable second opinion is identity-bound, not pid-bound.

    A pid is a recyclable name and ``run/host`` outlives the processes that
    wrote it, so "the record on disk with my pid" can be a long-dead stranger's.
    Handing THAT build to the import classifier would name an install race for a
    genuine packaging error — the false positive its docstring forbids.
    """
    root = tmp_path / "cfg"
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    monkeypatch.setattr(journal, "_own_record", None)

    # A record for OUR pid that we did not write (a recycled identity).
    stranger = journal.BootRecord(
        pid=os.getpid(), session_id="stranger", started_at=time.time() - 10_000.0
    )
    registry.publish(stranger, root, "run/host")
    assert journal.read_boot_record(os.getpid(), root=None) is not None
    assert journal.recorded_boot_build() is None, "a stranger's record must not answer"

    # Ours, published through the production writer.
    journal.write_boot_record("mine", update.BuildStamp("1.0.0", "aaa"))
    assert journal.recorded_boot_build() is not None


def test_prune_boot_records_drops_dead_pids_and_keeps_live_ones(tmp_path: Path) -> None:
    """R4: ``run/host`` is bounded by AGE, and never at a live runtime's expense."""
    root = tmp_path / "cfg"
    now = time.time()
    an_old_death = _pid_from_child()
    a_fresh_death = _pid_from_child()
    registry.publish(
        journal.BootRecord(
            # Past the retention bound (``registry.REAPED_MAX_AGE_S``), which is
            # the same policy the reaped sidecar uses: evidence is worth one look
            # soon after the death, not indefinite storage.
            pid=an_old_death,
            session_id="dead",
            started_at=now - registry.REAPED_MAX_AGE_S - 60.0,
        ),
        root,
        "run/host",
    )
    registry.publish(
        # A death from minutes ago: still evidence, still kept.
        journal.BootRecord(pid=a_fresh_death, session_id="just-died", started_at=now - 120.0),
        root,
        "run/host",
    )
    # Our own pid is alive by definition, and ancient: age alone must not take it.
    registry.publish(
        journal.BootRecord(
            pid=os.getpid(), session_id="live", started_at=now - registry.REAPED_MAX_AGE_S * 5
        ),
        root,
        "run/host",
    )

    removed = journal.prune_boot_records(root, now=now)
    assert removed == 1
    assert journal.read_boot_record(an_old_death, root=root) is None
    assert journal.read_boot_record(a_fresh_death, root=root) is not None
    assert journal.read_boot_record(os.getpid(), root=root) is not None


def test_prune_boot_records_keeps_the_newest_when_the_count_bound_bites(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """R2-1: the count path evicts by AGE, not by filename.

    The directory glob hands records back in pid-as-string order, which says
    nothing about when a runtime died: evicting in that order deletes the
    FRESHEST death and keeps the oldest, the inversion ``registry._prune_reaped``
    exists to prevent. This arm gives the newest death the SMALLEST pid, so with
    filename order it is the first victim and with age order it is the last
    record standing (reviewer round 2, R2-1).
    """
    root = tmp_path / "cfg"
    # A three-record count bound and no record old enough to age out, so the
    # count path is the only term that can fire.
    monkeypatch.setattr(registry, "REAPED_MAX_FILES", 3)
    now = time.time()
    pids = sorted(_pid_from_child() for _ in range(5))
    for index, pid in enumerate(pids):
        registry.publish(
            journal.BootRecord(pid=pid, session_id=f"s{pid}", started_at=now - index * 60.0),
            root,
            "run/host",
        )

    newest, oldest = pids[0], pids[-1]
    assert journal.prune_boot_records(root, now=now) == 2
    assert (
        journal.read_boot_record(newest, root=root) is not None
    ), "the newest death is the evidence; it must survive a count-bound eviction"
    assert journal.read_boot_record(oldest, root=root) is None


def test_a_closed_row_that_left_a_turn_open_is_not_an_unfinished_turn(tmp_path: Path) -> None:
    """Q10, pinned as a decision rather than left as prose.

    A SIGTERM the drain bound cut exits CLEANLY: the row ends closed with
    ``end_cause="runtime-shutdown"`` and ``still_open_at_exit=true``. The turn
    did end — the runtime ran its own exit ordering and published the cause to
    the attention store, which is the surface and the notice both read — so
    ``open_row_after_death`` refusing it is correct: "a turn was in flight and
    never ended" would be false, and accepting it would give the same run a
    second narrator under a cause the first one already named.
    """
    directory = _session_directory(tmp_path, "sess-drained")
    _write_open_row(
        directory,
        _pid_from_child(),
        open=False,
        ended_at=time.time(),
        end_cause="runtime-shutdown",
        exit_cause="SIGTERM",
        still_open_at_exit=True,
    )
    assert journal.open_row_after_death(directory) is None
    # The evidence is still ON the row for a reader that asks the writer's
    # question rather than the death question.
    row = journal.TurnJournalRow.from_json(registry.read_turn_journal(directory))
    assert row is not None
    assert journal.signal_exit_token(row.exit_cause) == "SIGTERM"
    assert row.still_open_at_exit is True


# ---------------------------------------------------------------------------
# The classifier's preference order
# ---------------------------------------------------------------------------


def _current_build_fields() -> dict[str, str]:
    """The build this process is running, as a row would record it.

    Through ``buildwatch.build_prefix`` — the same seam the runtime stamps its
    rows with — so that "the row's build equals the install" is a statement
    about ONE tree.
    """
    from local_operator import buildwatch

    stamp = update.installed_build(buildwatch.build_prefix())
    return {"version": stamp.version, "source_ref": stamp.source_ref}


def _write_open_row(
    directory: Path,
    pid: int,
    *,
    build: dict[str, str] | None = None,
    alive_until: float | None = None,
    **fields: Any,
) -> journal.TurnJournalRow:
    """An open row whose owner is gone, written the way the runtime writes it.

    ``alive_until`` is the LAST INSTANT THIS RUNTIME IS KNOWN TO HAVE BEEN
    ALIVE — the row's own ``updated_at``/``started_at``, which the tear rung
    binds its comparison to. It defaults to ``now`` because a real row is
    written by a live runtime; the epoch-1970 stamp this fixture used to carry
    was itself the defect reviewer round 1 (R1) filed, since a 1970 timestamp
    makes the install look "older than the runtime" no matter what it is and so
    pinned the unbounded answer.

    The build defaults to the RUNNING install unless a test overrides it: the
    classifier's tear rung compares the row's build against the install on disk,
    so a fabricated build would make every cell read as a tear unless the test
    that means to exercise one says so explicitly.
    """
    stamp = time.time() - 1.0 if alive_until is None else alive_until
    payload: dict[str, Any] = {
        "session_id": directory.name,
        "pid": pid,
        "parent_pid": 1,
        "turn_seq": 4,
        "command_id": "cmd-4",
        "started_at": stamp - 30.0,
        "ended_at": None,
        "open": True,
        "end_cause": "",
        "exit_cause": "",
        "still_open_at_exit": False,
        "last_boundary": "bash",
        "build": _current_build_fields() if build is None else build,
        "install_root": "/tmp/install",
        "updated_at": stamp,
    }
    payload.update(fields)
    registry.write_turn_journal(directory, payload)
    row = journal.TurnJournalRow.from_json(payload)
    assert row is not None
    return row


# NOTE ON ISOLATION. ``LOP_BUILD_PREFIX`` points the install stamp at a fake
# tree, so a developer who has it exported would have every cell here compare a
# row's build against a different root from the one it was stamped with — the
# writer/reader mislabel reviewer round 1 (R2) filed, reproduced from the
# ambient shell. No fixture is needed for it HERE: ``tests/conftest.py``'s
# autouse ``isolate_environment`` clears every ``_AMBIENT_VARS`` name for every
# test in the suite, and that list carries this variable with that exact reason.
# One central scrub, not a second copy per module.


def test_the_classifier_names_a_crash_from_the_runtimes_own_row(tmp_path: Path) -> None:
    """No signal recorded and no install movement: the crash, now evidenced."""
    directory = _session_directory(tmp_path, "sess-crash")
    _write_open_row(directory, _pid_from_child())

    kind, cause, reason = _classify_orphaned_run(directory)
    assert (kind, cause) == ("error", "runtime-killed")
    assert "turn 4 in flight" in reason
    assert "last boundary bash" in reason
    # The pid has to be there: the 19:41 investigation's first question was
    # "which process", and the legacy sentence never carried one.
    written = json.loads((directory / registry.TURN_JOURNAL_NAME).read_text())
    assert f"pid {written['pid']}" in reason


def test_the_classifier_names_the_escalated_sweep(tmp_path: Path) -> None:
    """A row that recorded a signal is a stop sweep that REACHED its target.

    This is the shape the fleet could not name: the runtime was asked to leave,
    wrote the signal down, and was killed before its turn ended.
    """
    directory = _session_directory(tmp_path, "sess-signalled")
    _write_open_row(directory, _pid_from_child(), exit_cause="SIGTERM", still_open_at_exit=True)

    kind, cause, reason = _classify_orphaned_run(directory)
    assert (kind, cause) == ("error", "runtime-shutdown")
    assert "SIGTERM" in reason


def _torn_install(monkeypatch: pytest.MonkeyPatch, *, installed_ago_s: float | None) -> None:
    """Make the on-disk install a DIFFERENT build, last written N seconds ago."""
    monkeypatch.setattr(
        update, "installed_build", lambda *_a, **_k: update.BuildStamp("0.56.0", "cafe9999")
    )
    monkeypatch.setattr(update, "build_marker_age_s", lambda *_a, **_k: installed_ago_s)


def test_the_classifier_names_the_install_window_tear(tmp_path: Path, monkeypatch) -> None:
    """An install that moved while the runtime was ALIVE names the tear.

    R1's bound, positive arm: the install was last written an hour before the
    row's last write, so the runtime is demonstrably alive after the tree under
    it was replaced — the ordering the rung claims.
    """
    directory = _session_directory(tmp_path, "sess-torn")
    _write_open_row(
        directory,
        _pid_from_child(),
        build={"version": "0.55.9", "source_ref": "beef1234"},
        alive_until=time.time(),
    )
    _torn_install(monkeypatch, installed_ago_s=3600.0)

    kind, cause, reason = _classify_orphaned_run(directory)
    assert (kind, cause) == ("error", "install-mid-update")
    assert "0.55.9@beef123" in reason and "0.56.0@cafe999" in reason


def test_the_classifier_does_not_blame_an_install_that_moved_after_the_death(
    tmp_path: Path, monkeypatch
) -> None:
    """R1's regression arm — the 19:41 read, which the unbounded rule got wrong.

    A runtime SIGKILLed at 19:41 with a turn in flight, read at 20:30 by a
    successor running the build that was published in between: the builds
    differ, and yet the death was a kill, not a tear. The row's last write is
    the discriminator — the install is NEWER than the last instant this runtime
    is known to have been alive.
    """
    directory = _session_directory(tmp_path, "sess-killed-then-released")
    killed_at = time.time() - 3000.0
    _write_open_row(
        directory,
        _pid_from_child(),
        build={"version": "0.55.9", "source_ref": "beef1234"},
        alive_until=killed_at,
    )
    # The install landed 200 s after the kill — a release riding the same window.
    _torn_install(monkeypatch, installed_ago_s=2800.0)

    kind, cause, reason = _classify_orphaned_run(directory)
    assert (kind, cause) == ("error", "runtime-killed"), reason
    assert "install-mid-update" not in cause
    assert "turn 4 in flight" in reason


def test_install_moved_refuses_to_guess(tmp_path: Path, monkeypatch) -> None:
    """No install date, or no row timestamp, is no evidence — never a tear."""
    directory = _session_directory(tmp_path, "sess-unbounded")
    row = _write_open_row(
        directory, _pid_from_child(), build={"version": "0.55.9", "source_ref": "beef1234"}
    )
    _torn_install(monkeypatch, installed_ago_s=None)
    assert journal.install_moved(row) is False

    _torn_install(monkeypatch, installed_ago_s=1.0)
    assert journal.install_moved(replace(row, updated_at=0.0, started_at=0.0)) is False


def test_the_classifier_keeps_the_legacy_paths_without_a_row(tmp_path: Path) -> None:
    """No evidence must behave exactly as it did before this change.

    Two arms, and both are the ones the existing taxonomy was built on: a dead
    record with no journal row still reads ``runtime-killed`` with the record's
    own detail, and a session with nothing at all still reads
    ``CUT_OFF_UNKNOWN`` rather than a guess.
    """
    directory = _session_directory(tmp_path, "sess-legacy")
    assert _classify_orphaned_run(directory) == (
        "error",
        "",
        "the turn was cut off and the cause could not be determined",
    )

    dead = _pid_from_child()
    root = tmp_path
    registry.publish(
        registry.SessionRecord(
            pid=dead,
            kind="daemon",
            session_id="sess-legacy",
            conversation_name="legacy",
            cwd=str(tmp_path),
            model_label="mock",
            control_port=0,
            control_key="k" * 8,
        ),
        root,
    )
    kind, cause, reason = _classify_orphaned_run(directory)
    assert (kind, cause) == ("error", "runtime-killed")
    assert f"pid {dead}" in reason


def test_a_live_owner_with_an_open_row_is_not_a_death(tmp_path: Path) -> None:
    """A turn in flight in a LIVE runtime must never read as an interruption."""
    directory = _session_directory(tmp_path, "sess-live")
    writer = journal.TurnJournal(directory, "sess-live", update.BuildStamp("1.0.0", "aaa"))
    writer.open_turn(command_id="cmd-live")

    assert journal.open_row_after_death(directory) is None
    # And the row itself is there, saying the turn is in flight.
    row = journal.TurnJournalRow.from_json(registry.read_turn_journal(directory))
    assert row is not None and row.open is True


def test_the_death_verdict_never_falls_back_to_no_cause(tmp_path: Path) -> None:
    """``CUT_OFF_UNKNOWN`` is unreachable FROM a row, which is the whole point."""
    directory = _session_directory(tmp_path, "sess-verdict")
    row = _write_open_row(directory, _pid_from_child())
    kind, cause, _reason = journal.death_verdict(row)
    assert kind == "error"
    assert cause in {"runtime-killed", "install-mid-update", "runtime-shutdown"}


# ---------------------------------------------------------------------------
# The interruption reaches the resumed session, once, and re-runs nothing
# ---------------------------------------------------------------------------


def _stream_recording(calls: list[Any]) -> Any:
    """A stream double that records a call and yields nothing.

    An async generator, because that is what ``stream_fn`` requires: a plain
    callable returning ``None`` would type-check as a mistake and would raise
    the first time a turn actually ran — which is precisely the turn this suite
    asserts does NOT happen.
    """

    async def stream(request: Any, *args: Any, **kwargs: Any) -> Any:
        calls.append(request)
        return
        yield  # pragma: no cover — the ``yield`` is what makes this a generator

    return stream


def _boot_session(directory: Path, calls: list[Any]) -> Any:
    """A REAL successor session over the killed session's directory."""
    from local_operator.harness.types import ModelSpec
    from local_operator.session.session import Session
    from local_operator.session.transcript import Transcript

    return Session(
        model=ModelSpec(provider="test", model_id="mock"),
        stream_fn=_stream_recording(calls),
        tools=[],
        transcript=Transcript(directory),
        system_blocks_provider=lambda *_args: [],
    )


def _incidents(directory: Path) -> list[Any]:
    """Every ``session_incident`` row, however it was written."""
    from local_operator.session.transcript import Transcript

    return [
        entry
        for entry in Transcript(directory).entries()
        if entry.payload.get("custom_type") == "session_incident"
    ]


@pytest.mark.asyncio
async def test_the_interruption_reaches_the_resumed_session_once(tmp_path: Path) -> None:
    """The trust half: an agent must be able to tell a cut-off from a completion
    in its OWN state, and must be told exactly once."""
    session_id = "sess-resumed"
    directory = _session_directory(tmp_path, session_id)
    # A durable row of work, so the resumed session has real history to read.
    (directory / "transcript.jsonl").write_text(
        '{"id": "seed", "ts": 1, "type": "message", "payload": {"kind": "message", '
        '"role": "user", "content": [{"type": "text", "text": "start the deploy"}]}}\n',
        encoding="utf-8",
    )
    child = _spawn_mid_turn(directory, session_id, tmp_path)
    killed_pid = _kill(child)

    calls: list[Any] = []
    session = _boot_session(directory, calls)
    try:
        restored = session._restored_cut_off
        assert restored is not None, "the boot must carry the interruption into the session"
        kind, cause, reason, token = restored
        assert kind == "error"
        assert cause == "runtime-killed"
        assert f"pid {killed_pid}" in reason
        assert token == f"turn-journal:{session_id}:{killed_pid}:1"

        # The narration is the EXISTING once-per-run path, so it must land once.
        await session.async_init()
        incidents = _incidents(directory)
        assert len(incidents) == 1, [
            (row.payload.get("details") or {}).get("text") for row in incidents
        ]
        text = str((incidents[0].payload.get("details") or {}).get("raw") or "")
        assert "never ended" in text or "disappeared without exiting cleanly" in text

        # It is in the model's OWN context, through the production converter.
        rendered = _rendered_text(session)
        assert "[session incident]" in rendered
        assert "in flight" in rendered

        # AND IT RERAN NOTHING: no prompt, no provider call, no turn.
        assert calls == [], "the interrupted turn must never be replayed"
        assert session._restored_cut_off is None
    finally:
        await session.dispose()

    # A second open of the same session narrates nothing new.
    calls_again: list[Any] = []
    second = _boot_session(directory, calls_again)
    try:
        await second.async_init()
        assert len(_incidents(directory)) == 1, "re-opening re-narrated the interruption"
        assert calls_again == []
    finally:
        await second.dispose()


def _rendered_text(session: Any) -> str:
    """The next turn's model-visible text, through the production converter."""
    from local_operator.session.session import _default_convert_to_llm

    parts: list[str] = []
    for message in _default_convert_to_llm(list(session._context.messages)):
        for block in message.content:
            text = getattr(block, "text", None)
            if text:
                parts.append(str(text))
    return "\n".join(parts)


def test_a_fired_stall_bound_is_narrated_as_its_own_class(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """THE BOUND'S EXIT HAS A CLASS, and it is read off the bound's own dump.

    The defect this closes, measured on this fleet on 2026-09-21: a peer session
    (pid 4698) was ended by the stall bound, its dump was on disk, and its death
    was narrated as ``unattributed`` — which is the taxonomy's word for "no act
    was recorded". That is the worst of the three possible readings, because it
    is not "we could not tell" but "the instrument knew and did not say so"; a
    bound that fires and cannot name itself is indistinguishable to the next
    reader from one that never fired.

    THE DUMP IS THE ONLY PLACE THE CLASS CAN COME FROM. ``faulthandler`` reaches
    its timer from a C thread and calls ``_exit(1)`` there, so no exit hook, no
    journal write and no reaper runs after a fire. That is why the rung reads the
    artifact rather than a record — and why it sits ABOVE every other rung: it is
    written at the instant of death, later than anything the row itself says.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    directory = _session_directory(tmp_path, "sess-stall")
    writer = journal.TurnJournal(directory, "sess-stall", update.BuildStamp("1.0.0", "aaa"))
    writer.open_turn(command_id="cmd-stall")
    row = journal.TurnJournalRow.from_json(registry.read_turn_journal(directory))
    assert row is not None

    dump = stall_watchdog.dump_path(row.pid)
    dump.parent.mkdir(parents=True, exist_ok=True)

    # (a) A HEADER-ONLY FILE IS NOT A FIRE — the hard-death shape, which must keep
    # reaching the arm that says no act was recorded.
    dump.write_text("[stall watchdog] armed\n", encoding="utf-8")
    os.utime(dump, (row.started_at + 1, row.started_at + 1))
    kind, cause, reason = journal.death_verdict(row)
    assert cause != STALL_BOUND_CAUSE, reason

    # (b) A FIRE, WITH THE PROGRESS LINE: the runtime was ended by the composite
    # leg, and the detail says which one so a reader is not sent looking for a
    # loop that never came back.
    dump.write_text(
        f"[stall watchdog] armed\n{stall_watchdog.PROGRESS_MARKER}{STALL_BOUND_CAUSE}: 300s of "
        f"CPU with no progress\n{stall_watchdog.FIRED_MARKER}0:05:00)!\nThread 0x1:\n",
        encoding="utf-8",
    )
    os.utime(dump, (row.started_at + 1, row.started_at + 1))
    kind, cause, reason = journal.death_verdict(row)
    assert kind == "error"
    assert cause == STALL_BOUND_CAUSE, cause
    assert "ITSELF" in reason, reason
    assert "stopped advancing" in reason, reason

    # (c) A FIRE WITH NO PROGRESS LINE: the SILENCE leg, and the detail has to say
    # so — the class is deliberately one token for both legs.
    dump.write_text(
        f"[stall watchdog] armed\n{stall_watchdog.FIRED_MARKER}0:05:00)!\nThread 0x1:\n",
        encoding="utf-8",
    )
    os.utime(dump, (row.started_at + 1, row.started_at + 1))
    kind, cause, reason = journal.death_verdict(row)
    assert cause == STALL_BOUND_CAUSE, cause
    assert "no plane reported" in reason, reason

    # (e) A FIRE OVER A DUMP THAT NAMES A DEAD TICKER: still the SILENCE leg -- the
    # predicate is what fired -- but the narration must carry the second fact, which
    # is the reason the plane had nothing left to report WITH. Without this the
    # automated verdict kept calling a runtime whose reporter had died a runtime
    # whose loop had gone quiet (agent review round 1, MINOR 3): the one surface
    # where the false attribution survived not being a human reading the file.
    dump.write_text(
        f"[stall watchdog] armed\n"
        f"{stall_watchdog.TICK_DEATH_MARKER}{stall_watchdog.WORKLOAD}: RuntimeError: the "
        f"rigged beat raised\n"
        f"{stall_watchdog.FIRED_MARKER}0:05:00)!\nThread 0x1:\n",
        encoding="utf-8",
    )
    os.utime(dump, (row.started_at + 1, row.started_at + 1))
    kind, cause, reason = journal.death_verdict(row)
    assert cause == STALL_BOUND_CAUSE, cause
    assert "REPORTER" in reason, reason
    assert (
        "no plane reported" not in reason
    ), f"a dump recording the ticker's own death is still narrated as a silent loop: {reason}"

    # (d) A DUMP OLDER THAN THIS TURN IS NOT THIS TURN'S. The file is keyed by pid
    # alone, so a recycled pid would otherwise let another runtime's freeze be
    # narrated as this row's death.
    os.utime(dump, (row.started_at - 600, row.started_at - 600))
    assert journal.death_verdict(row)[1] != STALL_BOUND_CAUSE

    # (f) A FIRE THE RUNTIME SURVIVED IS NOT THIS TURN'S DEATH — the THIRD STATE,
    # and the one this change introduces. With a turn in flight the bound dumps and
    # leaves the runtime ALIVE (``stall_watchdog._holds_work``), so a dump carrying
    # ``HELD_MARKER`` is evidence the runtime was STALLED, never evidence about what
    # killed it: the death that left this row is some other death, and calling it
    # ``runtime-stall-bound`` would put the instrument's name on the wrong corpse —
    # the precise falsehood this rung was added to end, in the other direction. The
    # fact still reaches the reader, as a lead on whatever DID answer.
    dump.write_text(
        f"[stall watchdog] armed\n{stall_watchdog.FIRED_MARKER}0:05:00)!\nThread 0x1:\n"
        f"{stall_watchdog.HELD_MARKER}it did not end it\n",
        encoding="utf-8",
    )
    os.utime(dump, (row.started_at + 1, row.started_at + 1))
    kind, cause, reason = journal.death_verdict(row)
    assert (
        cause != STALL_BOUND_CAUSE
    ), f"a fire the runtime SURVIVED was narrated as the death that ended it: {reason}"
    assert journal.HELD_BOUND_LEAD in reason, (
        f"the death says nothing about the bound that fired and held, so a reader "
        f"comparing this with the dump sees two unrelated events: {reason}"
    )

    # AND THE LIST COLUMN KEEPS IT (design review round 1, D3). ``outcome_summary`` is
    # what a LIST renders a reason through, and it splits the sentence at its
    # parenthetical — so while the lead rode inside the brackets a held death and a
    # no-dump death rendered byte-identical there, collapsing three attribution states
    # into two outcomes on the surface a person scans. The lead is a CLAUSE of the
    # sentence now, which is what makes it survive the split.
    from local_operator import incidents

    listed = incidents.outcome_summary(reason)
    assert journal.HELD_BOUND_LEAD in listed, (
        f"the listing column dropped the one fact that says this runtime survived its "
        f"bound: {listed!r}"
    )
    assert listed != incidents.outcome_summary(
        incidents.render_cut_off_reason(incidents.KILL_UNATTRIBUTED)
    ), "a held death must not render as a plain unattributed one"
