"""A REUSED PID MUST NOT IMPERSONATE THE DEAD OWNER OF A TRANSCRIPT CLAIM.

THE INCIDENT (2026-09-21). The operator could not reopen session ``bfbc971ef537``:
every interface refused, the engage loop spent its whole 30 s deadline
(``launch.DEFAULT_DEADLINE_S``) spawning nothing, and the user was shown the
generic ``the runtime is reconnecting``. The session's runtime had died and left
its claim behind naming pid 1969; pid 1969 was then REUSED by an unrelated live
process. ``session_lease._pid_state`` asked only "is this pid a live process", so
the dead owner's claim read live for as long as the stranger happened to hold the
number — and three consumers were poisoned by it: ``launch._lease_holder`` made
engage wait for a runtime that would never publish, ``acquire_session_lease``
refused the session from every interface, and ``resume.live_runtime_pid``
reported it as already owned.

**A PID IS NOT AN IDENTITY.** The kernel hands the number to the next process
that wants one as soon as the owner is reaped, which is precisely the case the
round-3 U10 zombie fix does not reach: its reasoning is "the pid is not reused
while the corpse lingers", true of an unreaped zombie and false one instant after
it is reaped. A claim therefore records a BIRTH TOKEN — the start time of the
process that wrote it, as the platform renders it (``procstate.birth_token``) —
and every caller requires liveness AND an identity match before calling a holder
live.

HOW THESE TESTS STAGE A REUSE. The kernel picks which process gets a recycled
number, so a literal pid-reuse race cannot be staged deterministically. What is
staged instead is the STATE reuse produces, built from measured pieces: a token
is read off a real process, that process is killed and reaped, and the claim is
then written naming a DIFFERENT live process that holds the same number. Every
pid and every token below is real; only the coincidence is arranged. The
arrangement is also the honest general form of the defect — "this pid is live but
is not the process that wrote the claim" — of which pid reuse is one cause (a
migrated container, a copied session directory and a restored store are others).

THE MIXED-GENERATION MATRIX THESE TESTS PIN:

1. token matches the live holder -> LIVE, owner protected (never displaced);
2. no token (legacy claim, or a Windows writer) -> today's pid-liveness, so a
   live owner on an older build is never displaced by a newer one;
3. token differs from the live holder -> DEAD, the claim is recoverable;
4. token unverifiable (unknown scheme, empty value, unreadable probe) -> LIVE,
   because the unsafe direction is calling a live writer dead.

Cells 3 and 4 must never be conflated: 3 authorises a recovery, 4 forbids one.
"""

from __future__ import annotations

import fcntl
import json
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import pytest

from local_operator import procstate
from local_operator import session_lease
from local_operator import session_lease as lease_mod
from local_operator.resume import live_runtime_pid
from local_operator.session.runtime.launch import _lease_holder
from local_operator.session_lease import (
    LEASE_NAME,
    MIRROR_NAME,
    RECOVERY_LOCK_NAME,
    SessionLeaseHeldError,
    _pid_state,
    _read_claim,
    acquire_session_lease,
    reap_proven_dead_session_claim,
)

#: The birth token probe is POSIX-only: Windows produces none by design (its
#: process identity is a handle, not a process-table entry — see ``procstate``),
#: and a platform that can never read a token keeps every claim on the legacy
#: pid-liveness path by construction. The cells below are therefore POSIX-only,
#: and the fallback they exercise on Windows is pinned by
#: ``test_windows_records_no_birth_and_keeps_todays_verdicts``.
posix_only = pytest.mark.skipif(
    procstate.is_windows(), reason="the birth token probe is POSIX-only by design"
)

#: How long a staged dead owner is held alive before its token is read. The
#: macOS token is `ps -o lstart=` — whole SECONDS — so an owner spawned and
#: killed inside one second would stage a token COLLISION, not a reuse. A real
#: dead owner is a runtime that lived for minutes or hours, so its start second
#: and the second its pid was handed to a stranger are far apart; this is the
#: wait that keeps the staging faithful rather than convenient.
_TOKEN_TICK_MARGIN_S = 1.2


def _spawn_live_process() -> subprocess.Popen[bytes]:
    """A live process that is NOT a runtime for the session under test.

    Deliberately not a `lop` child of any kind: it stands in for the stranger the
    kernel gave a dead owner's pid to.
    """
    return subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])


@pytest.fixture(scope="module")
def dead_birth() -> tuple[str, str]:
    """The (scheme, token) a dead owner left in its claim — measured once.

    Read off a real process and then that process reaped, rather than invented,
    so these tests fail for the defect rather than for a fixture the code could
    never have seen. Module-scoped because producing it costs a token tick.
    """
    process = _spawn_live_process()
    try:
        time.sleep(_TOKEN_TICK_MARGIN_S)
        scheme = procstate.birth_scheme()
        token = procstate.birth_token(process.pid)
        assert scheme is not None and token is not None, "a POSIX host must answer this"
        return (scheme, token)
    finally:
        process.kill()
        process.wait(timeout=10)


@pytest.fixture
def stranger(dead_birth: tuple[str, str]) -> Iterator[subprocess.Popen[bytes]]:
    """The live process that now holds the dead owner's pid.

    Depends on ``dead_birth`` so it is spawned AFTER the owner was reaped; that
    ordering is what makes the two tokens differ rather than collide.
    """
    process = _spawn_live_process()
    try:
        yield process
    finally:
        process.kill()
        process.wait(timeout=10)


def _write_claim(
    session_dir: Path,
    pid: int | None,
    birth: tuple[str | None, str | None] | None = None,
    *,
    generation: str = "forged",
    mirror: bool = True,
) -> None:
    """Write the exact on-disk state the incident left: claim plus mirror.

    ``birth`` is the recorded pair, and ``(None, None)`` writes the fields present
    but EMPTY — "a writer recorded nothing usable", which is not the same claim as
    an older build's, where the fields are absent entirely. Both are doubt for the
    identity question, and both must leave a live holder exactly where it is.
    """
    session_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "schema": 1,
        "session_id": session_dir.name,
        "generation": generation,
    }
    if pid is not None:
        payload["pid"] = pid
    if birth is not None:
        payload["birth_scheme"], payload["birth_token"] = birth
    (session_dir / LEASE_NAME).write_text(
        json.dumps(payload, separators=(",", ":")), encoding="utf-8"
    )
    if mirror and pid is not None:
        (session_dir / MIRROR_NAME).write_text(str(pid), encoding="utf-8")


def _live_owner_claim(session_dir: Path, birth: tuple[str, str] | None) -> None:
    """A claim naming THIS test process — a real, live, self-consistent writer."""
    _write_claim(session_dir, os.getpid(), birth, generation="genuine")


def _own_birth() -> tuple[str, str]:
    scheme = procstate.birth_scheme()
    token = procstate.birth_token(os.getpid())
    assert scheme is not None and token is not None
    return (scheme, token)


# ---------------------------------------------------------------------------
# Cell 3: the token differs from the live holder — the claim is stale
# ---------------------------------------------------------------------------


@posix_only
def test_a_reused_pid_does_not_read_as_the_dead_owner(
    tmp_path: Path, stranger: subprocess.Popen[bytes], dead_birth: tuple[str, str]
) -> None:
    """THE REGRESSION. The live process holding this pid is not the writer.

    Every assertion here failed before the fix, and the first one is the one the
    operator saw: ``_lease_holder`` reported a contender, so engage waited for a
    runtime that could not publish instead of spawning one, and the deadline
    expired into ``the runtime is reconnecting``.
    """
    session_id = "reusedpid"
    session_dir = tmp_path / "sessions" / session_id
    _write_claim(session_dir, stranger.pid, dead_birth)

    # The staging is real, not a string the code could never compare: the live
    # process holds a DIFFERENT token from the one the claim records, and that
    # token came off a process which existed.
    assert procstate.birth_token(stranger.pid) != dead_birth[1]

    # Liveness alone still says live — that is what pid reuse means, and it is
    # why the pid cannot be the question the arbitration asks.
    assert _pid_state(stranger.pid) == "live"

    # ...and the identity question is what says the writer is gone.
    assert _pid_state(stranger.pid, expected_birth=dead_birth) == "dead"

    # The user's symptom, at its source: engage sees no contender and spawns.
    assert _lease_holder(tmp_path, session_id) is None
    # Attach-vs-spawn: nobody is hosting this session.
    assert live_runtime_pid(tmp_path, session_id) is None
    # And the session is openable again: the generation-fenced recovery path may
    # take the claim.
    lease = acquire_session_lease(session_dir)
    assert lease.pid == os.getpid()
    lease.release()


@posix_only
def test_the_reaper_stays_on_pid_evidence_alone(
    tmp_path: Path, stranger: subprocess.Popen[bytes], dead_birth: tuple[str, str]
) -> None:
    """The cleanup path stays on PID evidence — an asymmetry, pinned on purpose.

    ``reap_proven_dead_session_claim`` is what the daemon calls after discovery
    proves a record's owner gone, and it DELETES a claim outright: no kernel lock
    held against a successor, no acquisition. The only evidence that may
    authorise that is the pid question, whose failure mode is a claim that leaks
    rather than a forked transcript. A token mismatch on a LIVE pid is therefore
    not a licence to reap — such a claim is taken over by
    ``acquire_session_lease``, which takes the lock and re-reads the whole claim
    first.
    """
    session_dir = tmp_path / "sessions" / "reapexisting"
    _write_claim(session_dir, stranger.pid, dead_birth)

    assert (
        reap_proven_dead_session_claim(session_dir, stranger.pid) is False
    ), "a live pid's claim must never be reaped, however stale its token looks"
    assert (session_dir / LEASE_NAME).exists()


@posix_only
def test_a_dead_pid_is_dead_whatever_its_token_says(tmp_path: Path) -> None:
    """MONOTONICITY. The token may narrow liveness; it may never widen deadness.

    A pid that is not a process has no writer, so a claim naming it must stay
    recoverable even when its recorded token cannot be read or compared. The
    opposite would make a dead owner's transcript immortal — the same defect
    class, pointing the other way.
    """
    session_id = "deadpid"
    session_dir = tmp_path / "sessions" / session_id
    gone_pid = 2_147_483_600  # RFC 2606 style: far outside any allocatable pid
    _write_claim(session_dir, gone_pid, ("ps-lstart-c-v1", "Thu Jan  1 00:00:00 1970"))

    assert _pid_state(gone_pid, expected_birth=("ps-lstart-c-v1", "x")) == "dead"
    assert _lease_holder(tmp_path, session_id) is None
    assert live_runtime_pid(tmp_path, session_id) is None
    lease = acquire_session_lease(session_dir)
    assert lease.pid == os.getpid()
    lease.release()


# ---------------------------------------------------------------------------
# Cell 1: the token matches — a genuine live owner is NEVER displaced
# ---------------------------------------------------------------------------


@posix_only
def test_a_genuine_live_owner_is_still_refused(tmp_path: Path) -> None:
    """THE NEGATIVE INVARIANT. Identity must not displace a practising writer.

    This is the direction that costs a transcript: calling a live owner dead lets
    a second runtime take a claim a working process is still appending to. The
    token is this process's own, so it matches by construction.
    """
    session_id = "genuineowner"
    session_dir = tmp_path / "sessions" / session_id
    birth = _own_birth()
    _live_owner_claim(session_dir, birth)

    assert _pid_state(os.getpid(), expected_birth=birth) == "live"
    assert _lease_holder(tmp_path, session_id) == os.getpid()
    assert live_runtime_pid(tmp_path, session_id) == os.getpid()
    with pytest.raises(SessionLeaseHeldError) as excinfo:
        acquire_session_lease(session_dir)
    assert excinfo.value.pid == os.getpid()
    assert "already open in pid" in str(excinfo.value)


@posix_only
def test_a_legacy_claim_without_a_token_is_not_displaced(tmp_path: Path) -> None:
    """Cell 2 — MIXED GENERATIONS. A claim from an older build carries no birth.

    Absent identity means "this build cannot tell", and the answer then has to
    stay today's pid-liveness verdict: otherwise the first new build to run would
    displace a live owner started by the build before it, which is a forked
    transcript rather than a recovered one.
    """
    session_id = "legacyowner"
    session_dir = tmp_path / "sessions" / session_id
    _live_owner_claim(session_dir, None)

    assert _lease_holder(tmp_path, session_id) == os.getpid()
    assert live_runtime_pid(tmp_path, session_id) == os.getpid()
    with pytest.raises(SessionLeaseHeldError):
        acquire_session_lease(session_dir)


@posix_only
def test_a_legacy_mirror_with_no_claim_is_not_displaced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The mirror alone is the older build's whole marker, and it still counts.

    ``retention.claim_session`` writes that mirror with no claim beside it, and a
    mirror has no identity available — so the answer stays today's. The identity
    is read from the CLAIM, never from the mirror, whose readers parse it as an
    ``int()`` and read anything else as "no owner" (i.e. as licence for a second
    writer).
    """
    session_id = "mirroronly"
    session_dir = tmp_path / "sessions" / session_id
    session_dir.mkdir(parents=True)
    (session_dir / MIRROR_NAME).write_text(str(os.getpid()), encoding="utf-8")

    assert live_runtime_pid(tmp_path, session_id) == os.getpid()
    with pytest.raises(SessionLeaseHeldError):
        acquire_session_lease(session_dir)


# ---------------------------------------------------------------------------
# Cell 4: the token cannot be verified — fail closed, and NEVER as a mismatch
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "birth",
    [
        pytest.param(("some-future-scheme-v9", "whatever"), id="unknown-scheme"),
        pytest.param(None, id="no-birth-recorded"),
        pytest.param((None, None), id="empty-fields"),
    ],
)
@posix_only
def test_an_unverifiable_token_fails_closed(
    tmp_path: Path, birth: tuple[str | None, str | None] | None
) -> None:
    """Cells 4 and 5: doubt is not death, and the two must never be conflated.

    A token this build cannot compare — written on another platform, by a build
    that measured something else, or not recorded at all — is not evidence of
    death. ``is_zombie`` and ``pid_liveness`` answer doubt the same way, for the
    same reason: a live writer declared dead is a forked transcript, while a dead
    one declared live only leaves the claim where it is.
    """
    session_id = "opaqueToken"
    session_dir = tmp_path / "sessions" / session_id
    _write_claim(session_dir, os.getpid(), birth)

    assert _pid_state(os.getpid(), expected_birth=birth) == "live"
    assert _lease_holder(tmp_path, session_id) == os.getpid()
    assert live_runtime_pid(tmp_path, session_id) == os.getpid()
    with pytest.raises(SessionLeaseHeldError):
        acquire_session_lease(session_dir)


@posix_only
def test_an_unreadable_probe_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The probe itself returning nothing is doubt too, not a licence to take.

    Staged by breaking the probe rather than the token, because "the platform
    could not answer" is what a failed ``ps``, an unreadable ``/proc`` entry, or
    a Windows host actually produces.
    """
    session_id = "unreadable"
    session_dir = tmp_path / "sessions" / session_id
    _live_owner_claim(session_dir, ("ps-lstart-c-v1", "Mon Sep 21 09:53:01 2026"))
    monkeypatch.setattr(lease_mod, "process_sample", lambda _pid: None)

    assert (
        _pid_state(os.getpid(), expected_birth=("ps-lstart-c-v1", "Mon Sep 21 09:53:01 2026"))
        == "live"
    )
    assert _lease_holder(tmp_path, session_id) == os.getpid()
    with pytest.raises(SessionLeaseHeldError):
        acquire_session_lease(session_dir)


@posix_only
def test_a_broken_probe_does_not_reach_the_mirror_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``live_runtime_pid`` fails closed too, on the same evidence."""
    session_id = "unreadablemirror"
    session_dir = tmp_path / "sessions" / session_id
    _live_owner_claim(session_dir, _own_birth())
    monkeypatch.setattr("local_operator.resume.process_sample", lambda _pid: None)

    assert live_runtime_pid(tmp_path, session_id) == os.getpid()


# ---------------------------------------------------------------------------
# The recovery path still serialises through the kernel lock
# ---------------------------------------------------------------------------


@posix_only
def test_recovery_of_a_reused_pid_claim_still_serialises_on_the_lock(
    tmp_path: Path, stranger: subprocess.Popen[bytes], dead_birth: tuple[str, str]
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
    _write_claim(session_dir, stranger.pid, dead_birth)
    assert _pid_state(stranger.pid, expected_birth=dead_birth) == "dead"

    held = os.open(session_dir / RECOVERY_LOCK_NAME, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(held, fcntl.LOCK_EX)
        with pytest.raises(SessionLeaseHeldError):
            acquire_session_lease(session_dir)
        assert (
            json.loads((session_dir / LEASE_NAME).read_text(encoding="utf-8"))["generation"]
            == "forged"
        ), "a refused takeover changed the claim"
    finally:
        os.close(held)

    # With the lock free, the very same state IS recoverable.
    lease = acquire_session_lease(session_dir)
    assert lease.pid == os.getpid()
    lease.release()


@posix_only
def test_the_claim_reread_under_the_lock_is_fenced_on_the_whole_claim(
    tmp_path: Path,
    stranger: subprocess.Popen[bytes],
    dead_birth: tuple[str, str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A successor that re-uses the pid AND changes the token is not stolen from.

    The fence exists so that "the exact generation/process pair changed" is
    detected under the lock; the birth token is part of that pair's identity, so
    a claim whose token moved while the recoverer was inspecting it must be left
    alone. Staged by having the lock holder rewrite the claim's token at the
    moment it hands over the lock — the interleaving the fence exists for,
    without a second process.
    """
    session_id = "tokenmoved"
    session_dir = tmp_path / "sessions" / session_id
    _write_claim(session_dir, stranger.pid, dead_birth)

    @contextmanager
    def rewrite_then_yield(_session_dir: Path) -> Iterator[bool]:
        _write_claim(_session_dir, stranger.pid, ("ps-lstart-c-v1", "Mon Sep 21 10:00:00 2026"))
        yield True

    monkeypatch.setattr(session_lease, "_stale_recovery_right", rewrite_then_yield)
    with pytest.raises(SessionLeaseHeldError):
        acquire_session_lease(session_dir)

    assert _read_claim(session_dir / LEASE_NAME).birth_token == "Mon Sep 21 10:00:00 2026"


@posix_only
def test_reap_refuses_a_live_pid_and_takes_a_dead_one_with_a_token(
    tmp_path: Path, stranger: subprocess.Popen[bytes], dead_birth: tuple[str, str]
) -> None:
    """The daemon's two cells, side by side, on identical-looking claims."""
    live_dir = tmp_path / "sessions" / "reapLive"
    _write_claim(live_dir, os.getpid(), _own_birth())
    assert reap_proven_dead_session_claim(live_dir, os.getpid()) is False
    assert (live_dir / LEASE_NAME).exists()

    dead_dir = tmp_path / "sessions" / "reapDead"
    gone_pid = 2_147_483_600
    _write_claim(dead_dir, gone_pid, dead_birth)
    assert reap_proven_dead_session_claim(dead_dir, gone_pid) is True
    assert not (dead_dir / LEASE_NAME).exists()
    assert not (dead_dir / MIRROR_NAME).exists()


# ---------------------------------------------------------------------------
# The dense grid: the cheap probe, and why its answer may stay wrong
# ---------------------------------------------------------------------------


@posix_only
def test_the_dense_grid_keeps_the_cheap_answer(
    tmp_path: Path, stranger: subprocess.Popen[bytes], dead_birth: tuple[str, str]
) -> None:
    """``check_zombie=False`` skips the identity proof too, deliberately.

    The engage loop's dense grid polls every 10 ms against a ~23-30 µs budget,
    and the identity proof rides the SAME platform probe as the corpse proof: on
    macOS one ``ps`` fork, measured at 2.4-4.6 ms, i.e. 24-46% of a dense
    iteration. Paying it per pass would eat the dead time the grid exists to
    remove, so the grid keeps the cheap answer and the proof is deferred to the
    coarse passes — where a wrong "live" costs a bounded wait (the grid is capped
    at ``launch._CONSTRUCTING_WINDOW_S`` = 3 s, after which the loop is coarse
    again) and never arbitration, because only the spawned child's
    ``acquire_session_lease`` may take a claim, and that one always proves.

    A stale claim cannot wedge the loop this way, and the last assertion is that
    property from the other side: the grid is only entered after a pass that SAW
    a contender or a live candidate of its own, and the FIRST pass of every
    engage is coarse — so the pass that decides to spawn always pays the proof.
    """
    session_id = "densegrid"
    session_dir = tmp_path / "sessions" / session_id
    _write_claim(session_dir, stranger.pid, dead_birth)

    assert _pid_state(stranger.pid, check_zombie=False, expected_birth=dead_birth) == "live"
    assert _lease_holder(tmp_path, session_id, check_zombie=False) == stranger.pid
    # The coarse pass, which is where the decision is taken, knows better.
    assert _lease_holder(tmp_path, session_id) is None


# ---------------------------------------------------------------------------
# The instrument itself: the token has to actually discriminate
# ---------------------------------------------------------------------------


@posix_only
def test_the_birth_token_is_stable_for_one_process_and_differs_across_processes(
    stranger: subprocess.Popen[bytes],
) -> None:
    """Prove the instrument before trusting the verdicts above.

    A token that changed between two reads of the same live process would report
    every live owner as stale (a second writer on a live transcript); one that was
    equal for two different processes would report none as stale (the incident).
    Both directions are checked against real processes, plus the round-trip that
    matters most: the token this process WRITES is the token a reader derives for
    it, on both platforms CI runs (the ``/proc`` branch is Linux's).
    """
    mine = procstate.birth_token(os.getpid())
    assert mine is not None
    assert procstate.birth_token(os.getpid()) == mine
    theirs = procstate.birth_token(stranger.pid)
    assert theirs is not None
    assert theirs != mine

    scheme = procstate.birth_scheme()
    assert scheme is not None
    assert procstate.same_birth(scheme, mine, os.getpid()) is True
    assert procstate.same_birth(scheme, mine, stranger.pid) is False
    # Doubt, in each of its shapes: no reading, unknown scheme, empty value.
    assert procstate.same_birth(scheme, None, os.getpid()) is None
    assert procstate.same_birth("some-future-scheme-v9", mine, os.getpid()) is None
    assert procstate.same_birth(scheme, mine, 2_147_483_600) is None


@posix_only
def test_the_self_token_is_memoised_and_follows_a_forked_pid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The writer's token is cached per pid, and a fork cannot inherit one.

    The token is immutable for the life of a process, so one probe answers every
    later question — but the memo is keyed by pid, because a fork-without-exec
    child inherits this module's globals and would otherwise write a claim naming
    the PARENT as its writer. A failed sample is not cached either: that would
    poison every later claim this process writes.

    **The memo is process-global, so this test has to start it cold.** Without
    the reset below the probe count depends on whether an earlier test already
    warmed it — which is how this assertion first shipped passing for the wrong
    reason (review round 1, M2: green on a cold memo, red on a warm one).
    """
    calls: list[int] = []
    real = procstate.birth_token

    def counted(pid: int) -> str | None:
        calls.append(pid)
        return real(pid)

    monkeypatch.setattr(procstate, "_SELF_BIRTH", None)
    monkeypatch.setattr(procstate, "birth_token", counted)

    first = procstate.self_birth_token()
    assert first is not None
    assert calls == [os.getpid()], calls
    # Warm: the second question is answered from the memo, not by a probe.
    assert procstate.self_birth_token() == first
    assert calls == [os.getpid()], calls

    # A different pid is a different process: the memo must not answer for it.
    monkeypatch.setattr(procstate, "_SELF_BIRTH", (os.getpid() + 1, "inherited-from-a-parent"))
    assert procstate.self_birth_token() == first
    assert calls == [os.getpid(), os.getpid()], calls

    # A failed sample is not cached, so a transient failure cannot poison every
    # later claim this process writes.
    monkeypatch.setattr(procstate, "_SELF_BIRTH", None)
    monkeypatch.setattr(procstate, "birth_token", lambda _pid: None)
    assert procstate.self_birth_token() is None
    assert procstate._SELF_BIRTH is None, "a failed sample was cached"


def test_windows_records_no_birth_and_keeps_todays_verdicts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Cell 2 on Windows, where the platform has no token at all.

    Nothing here can authorise a takeover the current build would not already
    authorise, and nothing refuses one it would already take: no scheme, no
    token, no probe, and a claim that therefore carries no birth fields. The
    probe is asserted NOT to run rather than merely to return nothing — a
    ``ps``-shaped fork on every acquisition is the cost this platform branch
    exists to avoid.
    """
    probed: list[object] = []

    class _Refusing:
        def run(self, argv: object, **_kwargs: object) -> object:
            probed.append(argv)
            raise AssertionError("win32 must not probe for a birth token")

    monkeypatch.setattr(procstate, "_PLATFORM", "win32")
    monkeypatch.setattr(procstate, "subprocess", _Refusing())
    monkeypatch.setattr(procstate, "_windows_liveness", lambda _pid: True)
    monkeypatch.setattr(procstate, "_SELF_BIRTH", None)

    assert procstate.birth_scheme() is None
    assert procstate.birth_token(os.getpid()) is None
    assert procstate.process_samples([os.getpid()]) == {}
    assert probed == []

    assert _pid_state(os.getpid()) == "live"
    assert _pid_state(2_147_483_600) == "live", "win32 reports every pid as live here"

    session_dir = tmp_path / "sessions" / "winclaim"
    lease = acquire_session_lease(session_dir)
    try:
        payload = json.loads((session_dir / LEASE_NAME).read_text(encoding="utf-8"))
        assert "birth_scheme" not in payload and "birth_token" not in payload
    finally:
        lease.release()
