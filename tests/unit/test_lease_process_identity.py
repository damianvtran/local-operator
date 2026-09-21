"""A REUSED PID MUST NOT IMPERSONATE THE DEAD OWNER OF A TRANSCRIPT CLAIM.

THE INCIDENT (2026-09-21). The operator could not reopen session ``bfbc971ef537``:
every interface refused, the engage loop spent its whole 30 s deadline
(``launch.DEFAULT_DEADLINE_S``) spawning nothing, and the user was shown the
generic ``the runtime is reconnecting``. The session's runtime had died and left
its claim behind naming pid 1969; pid 1969 was then REUSED by an unrelated live
process. ``session_lease._pid_state`` asks only "is this pid a live process", so
the dead owner's claim read live for as long as the stranger happened to hold the
number — and three consumers were poisoned by it: ``launch._lease_holder`` made
engage wait for a runtime that would never publish, ``acquire_session_lease``
refused the session from every interface, and ``resume.live_runtime_pid``
reported it as already owned.

**A PID IS NOT AN IDENTITY.** The kernel hands the number to the next process
that wants one as soon as the owner is reaped, which is precisely the case the
round-3 U10 zombie fix does not reach: its reasoning is "the pid is not reused
while the corpse lingers", true for an unreaped zombie and false one instant
after it is reaped. The claim therefore has to record something about the
PROCESS — its birth token (start time) — and every caller has to require
liveness AND a token match before calling a holder live.

HOW THESE TESTS STAGE A REUSE. The kernel picks which process gets a recycled
number, so a literal pid-reuse race cannot be staged deterministically. What is
staged instead is the STATE reuse produces, built from measured pieces: a token
is read off a real process, that process is killed and reaped, and the claim is
then written naming a DIFFERENT live process that holds the same number. Every
pid and every token below is real; only the coincidence is arranged. The
arrangement is also the honest general form of the defect — "this pid is live but
is not the process that wrote the claim" — of which pid reuse is one cause
(a migrated container, a copied session directory, and a restored store are
others).

THE FOUR CASES PINNED HERE ARE THE MIXED-GENERATION MATRIX:

1. token matches the live holder  -> LIVE, owner protected (never displaced);
2. token carries no value (legacy claim or mirror) -> today's pid-liveness, so a
   live owner on an older build is never displaced by a newer one;
3. token differs from the live holder -> DEAD, the claim is recoverable;
4. token cannot be verified (unreadable probe, or a token this build cannot
   compare) -> LIVE, because the unsafe direction is calling a live writer dead.
"""

from __future__ import annotations

import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from local_operator import procstate, session_lease
from local_operator import session_lease as lease_mod
from local_operator.resume import live_runtime_pid
from local_operator.session.runtime.launch import _lease_holder
from local_operator.session_lease import (
    LEASE_NAME,
    MIRROR_NAME,
    RECOVERY_LOCK_NAME,
    SessionLeaseHeldError,
    _pid_state,
    acquire_session_lease,
    reap_proven_dead_session_claim,
)

#: The birth token probe is a POSIX one: Windows answers ``None`` (its process
#: identity is a handle, not a process-table entry — see ``procstate``), and a
#: token that can never be read means every claim there stays on the legacy
#: pid-liveness path by construction. The mixed-generation cases below are
#: therefore POSIX-only, and the Windows behaviour they fall back to is pinned
#: by the existing lease/platform tests.
posix_only = pytest.mark.skipif(
    procstate.is_windows(), reason="the birth token probe is POSIX-only by design"
)


def _spawn_live_process() -> subprocess.Popen[bytes]:
    """A live process that is NOT a runtime for the session under test.

    Deliberately not a `lop` child of any kind: it stands in for the stranger
    the kernel gave a dead owner's pid to.
    """
    return subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])


#: The token's resolution, and the reason the staging below has to respect it.
#: macOS answers with ``ps -o lstart=`` — whole seconds — so two processes born
#: in the SAME second are indistinguishable to it (Linux is finer: 10 ms). A
#: real dead owner is a runtime that lived for minutes or hours, so its start
#: second and the second its pid was handed to a stranger are far apart; a
#: harness that spawns and kills an owner inside one tick would stage a collision
#: instead of a reuse. This is the wait that keeps the staging faithful.
_TOKEN_TICK_MARGIN_S = 1.2


def _token_of_a_process_that_is_now_gone() -> str:
    """A REAL birth token, measured, whose process no longer exists.

    This is what a claim left by a dead owner actually holds — a token that was
    true when it was written and can never be true again. Reading it off a live
    process and then reaping that process, rather than inventing a string, is
    the whole point: the token below is one the platform produced, so these tests
    fail for the defect rather than for a fixture the code could never have seen.

    The owner is held alive for one token tick first, so the token it leaves
    behind is the token of a process that had a start second of its own —
    see ``_TOKEN_TICK_MARGIN_S``.
    """
    process = _spawn_live_process()
    try:
        time.sleep(_TOKEN_TICK_MARGIN_S)
        token = procstate.process_birth_token(process.pid)
        assert token is not None, "the platform must answer this, or these tests are vacuous"
        return token
    finally:
        process.kill()
        process.wait(timeout=10)


def _write_claim(session_dir: Path, pid: int, token: str | None, *, generation: str = "forged") -> None:
    """Write the exact on-disk state the incident left: claim plus mirror."""
    session_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "schema": 1,
        "session_id": session_dir.name,
        "generation": generation,
        "pid": pid,
    }
    if token is not None:
        payload["token"] = token
    (session_dir / LEASE_NAME).write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    (session_dir / MIRROR_NAME).write_text(str(pid), encoding="utf-8")


@pytest.fixture(scope="module")
def dead_owner_token() -> str:
    """The token a dead owner left in its claim — measured once per module.

    Module-scoped because producing it costs a token tick (~1.2 s), and every
    test in this module stages the same incident state.
    """
    return _token_of_a_process_that_is_now_gone()


@pytest.fixture
def stranger(dead_owner_token: str) -> subprocess.Popen[bytes]:
    """The live process that now holds the dead owner's pid.

    Depends on ``dead_owner_token`` so it is always spawned AFTER the owner was
    reaped — the ordering is what makes the two tokens differ rather than
    collide (see ``_TOKEN_TICK_MARGIN_S``).
    """
    process = _spawn_live_process()
    try:
        yield process
    finally:
        process.kill()
        process.wait(timeout=10)


# ---------------------------------------------------------------------------
# Case 3: the token differs from the live holder — the claim is stale
# ---------------------------------------------------------------------------


@posix_only
def test_a_reused_pid_does_not_read_as_the_dead_owner(
    tmp_path: Path, stranger, dead_owner_token: str
) -> None:
    """THE REGRESSION. The live process holding this pid is not the writer.

    Every assertion here failed before the fix, and the first one is the one the
    operator saw: ``_lease_holder`` reported a contender, so engage waited for a
    runtime that could not publish instead of spawning one, and the deadline
    expired into ``the runtime is reconnecting``.
    """
    session_id = "reusedpid"
    session_dir = tmp_path / "sessions" / session_id
    dead_token = dead_owner_token
    _write_claim(session_dir, stranger.pid, dead_token)

    # The staging is real, not a string the code could never compare: the live
    # process holds a DIFFERENT token from the one the claim records, and the
    # token itself came off a process that existed.
    assert procstate.process_birth_token(stranger.pid) != dead_token

    # Liveness alone still says live — that is what pid reuse means, and it is
    # why the pid cannot be the question the arbitration asks.
    assert _pid_state(stranger.pid) == "live"

    # ...and the identity question is what says the owner is gone.
    assert _pid_state(stranger.pid, token=dead_token) == "dead"

    # The user's symptom, at its source: engage sees no contender and spawns.
    assert _lease_holder(tmp_path, session_id) is None
    # Attach-vs-spawn: nobody is hosting this session.
    assert live_runtime_pid(tmp_path, session_id) is None
    # And the session is openable again: the generation-fenced recovery path
    # may take the claim.
    lease = acquire_session_lease(session_dir)
    assert lease.pid == os.getpid()
    lease.release()


@posix_only
def test_the_daemon_can_reap_a_reused_pid_claim(
    tmp_path: Path, stranger, dead_owner_token: str
) -> None:
    """The cleanup path learns the same answer as the acquisition path.

    ``reap_proven_dead_session_claim`` is what the daemon calls after discovery
    proves a record's owner gone; a claim whose pid was reused read as protected
    there too, so the row and the claim outlived the session together.
    """
    session_dir = tmp_path / "sessions" / "reapexisting"
    _write_claim(session_dir, stranger.pid, dead_owner_token)

    assert reap_proven_dead_session_claim(session_dir, stranger.pid) is True
    assert not (session_dir / LEASE_NAME).exists()
    assert not (session_dir / MIRROR_NAME).exists()


# ---------------------------------------------------------------------------
# Case 1: the token matches — a genuine live owner is NEVER displaced
# ---------------------------------------------------------------------------

#: The test process is a real process with a real token, which makes it the
#: cheapest honest stand-in for a live owner: nothing else in the suite can be
#: guaranteed to be alive at the assertion.
def _live_owner_claim(session_dir: Path, *, token: str | None) -> None:
    _write_claim(session_dir, os.getpid(), token, generation="genuine")


@posix_only
def test_a_genuine_live_owner_is_still_refused(tmp_path: Path) -> None:
    """THE NEGATIVE INVARIANT. Identity must not displace a practising writer.

    This is the direction that costs a transcript: calling a live owner dead
    lets a second runtime take a claim a working process is still appending to.
    The token is this process's own, so it matches by construction.
    """
    session_id = "genuineowner"
    session_dir = tmp_path / "sessions" / session_id
    owner_token = procstate.process_birth_token(os.getpid())
    assert owner_token is not None
    _live_owner_claim(session_dir, token=owner_token)

    assert _pid_state(os.getpid(), token=owner_token) == "live"
    assert _lease_holder(tmp_path, session_id) == os.getpid()
    assert live_runtime_pid(tmp_path, session_id) == os.getpid()
    with pytest.raises(SessionLeaseHeldError) as excinfo:
        acquire_session_lease(session_dir)
    assert excinfo.value.pid == os.getpid()
    assert "already open in pid" in str(excinfo.value)


@posix_only
def test_a_legacy_claim_without_a_token_is_not_displaced(tmp_path: Path) -> None:
    """MIXED GENERATIONS. A claim written by an older build carries no token.

    Absent token means "this build cannot tell", and the answer then has to stay
    today's pid-liveness verdict — otherwise the first new build to run would
    displace a live owner started by the build before it, which is a forked
    transcript rather than a recovered one.
    """
    session_id = "legacyowner"
    session_dir = tmp_path / "sessions" / session_id
    _live_owner_claim(session_dir, token=None)

    assert _pid_state(os.getpid(), token=None) == "live"
    assert _lease_holder(tmp_path, session_id) == os.getpid()
    assert live_runtime_pid(tmp_path, session_id) == os.getpid()
    with pytest.raises(SessionLeaseHeldError):
        acquire_session_lease(session_dir)


# ---------------------------------------------------------------------------
# Case 4: the token cannot be verified — fail closed
# ---------------------------------------------------------------------------


@posix_only
def test_a_token_that_cannot_be_compared_fails_closed(tmp_path: Path) -> None:
    """An uncomparable token is doubt, and doubt never permits theft.

    A token this build cannot interpret (one written by a future build, or on
    another platform) is not evidence of death. ``is_zombie`` and
    ``pid_liveness`` answer doubt the same way, for the same reason.
    """
    session_id = "opaqueToken"
    session_dir = tmp_path / "sessions" / session_id
    _live_owner_claim(session_dir, token="somefuturetag:1234")

    assert _pid_state(os.getpid(), token="somefuturetag:1234") == "live"
    assert _lease_holder(tmp_path, session_id) == os.getpid()
    assert live_runtime_pid(tmp_path, session_id) == os.getpid()
    with pytest.raises(SessionLeaseHeldError):
        acquire_session_lease(session_dir)


@posix_only
def test_an_unreadable_probe_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The probe itself returning nothing is doubt too, not a licence to take.

    Staged by breaking the probe rather than the token, because "the platform
    could not answer" is the case an unreadable ``/proc``, a forked ``ps`` under
    load, or a Windows host actually produces.
    """
    session_id = "unreadable"
    session_dir = tmp_path / "sessions" / session_id
    _live_owner_claim(session_dir, token="lstart:1")
    monkeypatch.setattr(lease_mod, "process_facts", lambda _pid: None)

    assert _pid_state(os.getpid(), token="lstart:1") == "live"
    assert _lease_holder(tmp_path, session_id) == os.getpid()
    with pytest.raises(SessionLeaseHeldError):
        acquire_session_lease(session_dir)


# ---------------------------------------------------------------------------
# The recovery path still serialises through the recovery lock
# ---------------------------------------------------------------------------


@posix_only
def test_recovery_of_a_reused_pid_claim_still_serialises_on_the_lock(
    tmp_path: Path, stranger, dead_owner_token: str
) -> None:
    """A stale-by-identity claim is recovered the SAME way a corpse's is.

    Identity decides WHETHER the claim may be taken; it must not become a second
    path that takes it. The kernel lock is what makes recovery safe against a
    successor claiming the transcript between an inspection and a takeover, so a
    recoverer that cannot take the lock has to refuse — and the refusal is a
    ``SessionLeaseHeldError``, not a silent no-op.
    """
    session_id = "lockedrecovery"
    session_dir = tmp_path / "sessions" / session_id
    dead_token = dead_owner_token
    _write_claim(session_dir, stranger.pid, dead_token)
    assert _pid_state(stranger.pid, token=dead_token) == "dead"

    held = os.open(session_dir / RECOVERY_LOCK_NAME, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(held, fcntl.LOCK_EX)
        with pytest.raises(SessionLeaseHeldError):
            acquire_session_lease(session_dir)
        # The claim survived the refused takeover, generation included.
        assert json.loads((session_dir / LEASE_NAME).read_text(encoding="utf-8"))["generation"] == "forged"
    finally:
        os.close(held)

    # With the lock free, the very same state IS recoverable.
    lease = acquire_session_lease(session_dir)
    assert lease.pid == os.getpid()
    lease.release()


# ---------------------------------------------------------------------------
# The dense grid: the cheap probe, and why its answer may stay wrong
# ---------------------------------------------------------------------------


@posix_only
def test_the_dense_grid_keeps_the_cheap_answer(
    tmp_path: Path, stranger, dead_owner_token: str
) -> None:
    """``check_zombie=False`` skips the identity proof too, deliberately.

    The engage loop's dense grid polls every 10 ms against a ~23-30 µs budget,
    and the identity proof rides the SAME platform probe as the zombie proof: on
    macOS one ``ps`` fork, measured at 2.4-4.6 ms, i.e. 24-46% of a dense
    iteration. Paying it per pass would eat the dead time the grid exists to
    remove, so the grid keeps the cheap answer and defers the proof to the
    coarse passes — where a wrong "live" costs a bounded wait (the grid is
    capped at ``launch._CONSTRUCTING_WINDOW_S`` = 3 s, after which the loop is
    coarse again and this same probe proves the owner gone) and never
    arbitration, because only the spawned child's ``acquire_session_lease`` may
    take a claim, and that one always proves.

    A stale claim CANNOT wedge the loop this way: the grid is only entered after
    a pass that SAW a contender or a live candidate of its own, and the first
    pass of every engage is coarse — so the pass that decides to spawn always
    pays the proof. That is the property this test pins from the other side, by
    checking the coarse answer on the very same state.
    """
    session_id = "densegrid"
    session_dir = tmp_path / "sessions" / session_id
    dead_token = dead_owner_token
    _write_claim(session_dir, stranger.pid, dead_token)

    assert _pid_state(stranger.pid, check_zombie=False, token=dead_token) == "live"
    assert _lease_holder(tmp_path, session_id, check_zombie=False) == stranger.pid
    # The coarse pass, which is where the decision is taken, knows better.
    assert _lease_holder(tmp_path, session_id) is None


# ---------------------------------------------------------------------------
# The instrument itself: the token has to actually discriminate
# ---------------------------------------------------------------------------


@posix_only
def test_the_birth_token_is_stable_for_one_process_and_differs_across_processes(
    stranger,
) -> None:
    """Prove the instrument before trusting the verdicts above.

    A token that changed between two reads of the same live process would report
    every live owner as stale (a second writer on a live transcript); one that
    was equal for two different processes would report no one as stale (the
    incident). Both directions are checked against real processes.
    """
    mine = procstate.process_birth_token(os.getpid())
    assert mine is not None
    assert procstate.process_birth_token(os.getpid()) == mine
    theirs = procstate.process_birth_token(stranger.pid)
    assert theirs is not None
    assert theirs != mine
    assert procstate.birth_token_verdict(mine, mine) is True
    assert procstate.birth_token_verdict(theirs, mine) is False
    # Doubt, in both of its shapes: no reading, and no way to compare.
    assert procstate.birth_token_verdict(None, mine) is None
    assert procstate.birth_token_verdict(mine, "somefuturetag:1") is None
    assert procstate.birth_token_verdict(theirs, None) is None
