"""Atomic sole-writer leases for durable session transcripts.

The compatibility ``.session.pid`` marker is useful for old readers but cannot
be authoritative because replacing a text file is not acquisition.  This
module stays stdlib-only so resume discovery can consult it without importing
the engine or mobile stack.

Liveness is asked through :func:`local_operator.procstate.is_zombie` rather
than signal 0 alone, because the claim this module arbitrates is only ever
taken over from a holder that is PROVEN dead, and a zombie is exactly the
holder that looks alive forever.  The probe stays a leaf module so this one
keeps its stdlib-only contract.

**A PID IS NOT AN IDENTITY, and that is the second question this module asks**
(2026-09-21). A claim that records only a pid reads live for as long as ANY
process holds that number, and the kernel hands a reaped owner's number to the
next process that wants one — so a dead runtime's claim became permanently
un-takeable and the session could not be opened by any interface. The claim now
records its writer's birth token as well (``birth_scheme``/``birth_token``, see
:func:`local_operator.procstate.same_birth`) and the probe requires liveness AND
an identity match before a holder counts as live. **The token may only ever
NARROW liveness, never widen deadness**: it can turn ``live`` into ``dead``, and
it can never turn ``dead`` or ``uncertain`` into ``live``. That one rule is what
keeps every dead-pid caller here (``reap_proven_dead_session_claim``, the
recovery path) as takeable as it was.
"""

from __future__ import annotations

import json
import os
import secrets
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal, NamedTuple

from local_operator.procstate import (
    O_BINARY,
    birth_scheme,
    birth_token,
    is_zombie,
    pid_liveness,
    process_sample,
    self_birth_token,
)

LEASE_NAME = ".execution-lease"
MIRROR_NAME = ".session.pid"
RECOVERY_LOCK_NAME = ".execution-lease.recovery"


class SessionLeaseHeldError(RuntimeError):
    """Raised when another live or unverifiable process owns a transcript."""

    def __init__(self, session_dir: Path, pid: int | None) -> None:
        owner = f"pid {pid}" if pid is not None else "an unverifiable process"
        super().__init__(
            f"session {session_dir.name} is already open in {owner}; "
            "attach to that session or wait for its owner to exit"
        )
        self.session_dir = session_dir
        self.pid = pid


def _pid_state(
    pid: int,
    *,
    check_zombie: bool = True,
    expected_birth: "tuple[str | None, str | None] | None" = None,
) -> Literal["live", "dead", "uncertain"]:
    """Probe only what the platform can prove; uncertainty never permits theft.

    **An exited-but-unreaped process is DEAD here, not live.** Signal 0
    succeeds against a zombie, so a probe built on it alone reports the corpse
    of a SIGKILLed runtime as a working owner — forever, because the pid is not
    reused while it lingers and nothing may reap it on the owner's behalf. Both
    readers of this verdict require a holder to be *proven dead* before they
    move its claim, so the wrong answer here is not a misleading log line: it
    is a transcript whose sole-writer claim can never be acquired. The operator
    sees a session that no interface will open (`live_runtime_pid` refuses with
    "already open in pid N", where N is the corpse) and no mechanism will
    recover. :func:`local_operator.procstate.is_zombie` documents the incident.

    **The zombie probe is not free, so the caller decides how often it is
    worth spending.** Signal 0 costs ~1 µs and settles two of the three
    answers (a live owner, a pid that is simply gone); the proof costs a `ps`
    fork (2.4-4.6 ms across runs on this host, tracking load) and is therefore
    spent
    only after the cheap probe has already said "live". ``check_zombie=False``
    skips it,
    which is what the engage loop passes while it is polling every 10 ms
    against a claim it believes is mid-construction (`launch._lease_holder`).
    That choice is a LATENCY trade, never a safety one: the verdict it produces
    can be "live" for a corpse only where the caller has already decided to
    wait, and every acquisition still requires the proof before it may take a
    claim.
    """
    if pid <= 0:
        return "uncertain"
    # ONE PROBE, TWO PLATFORMS. `os.kill(pid, 0)` is a liveness question on
    # POSIX and a KILL on Windows (signal 0 falls through to
    # `TerminateProcess`), so the branch lives in `procstate.pid_liveness`
    # rather than here — this module probes the owner of a transcript, and a
    # probe that terminated that owner would destroy the very writer the lease
    # exists to protect. `None` is "the platform could not prove it", which
    # keeps its old "uncertain" meaning: uncertainty never permits theft.
    live = pid_liveness(pid)
    if live is None:
        return "uncertain"
    if not live:
        return "dead"
    if not check_zombie:
        # The caller is on a dense poll cadence and has already accepted that
        # it will wait; see the docstring. A zombie reads as live here, and so
        # does a holder whose identity has not been checked.
        return "live"
    scheme, token = expected_birth if expected_birth is not None else (None, None)
    if not token:
        # No identity was recorded — a claim written by an older build, or on
        # Windows, which produces no token at all. Today's answer, exactly:
        # this is the mixed-generation cell that keeps an older build's live
        # owner safe from a newer build's recovery path.
        return "dead" if is_zombie(pid) else "live"
    # ONE SAMPLE, BOTH ANSWERS, and the sample is what decides. Asking the
    # zombie question with a second `ps` fork would answer it about a different
    # instant than the identity question, which is the TOCTOU this probe's
    # single sample exists to close.
    sample = process_sample(pid)
    if sample is None:
        # Unreadable is DOUBT, never death: the platform could not answer, which
        # is the same answer signal 0 gives an unprovable pid. Fail closed.
        return "live"
    if sample.zombie:
        # A corpse is not a writer, whatever identity it carries. This comes
        # BEFORE the token comparison on purpose: the token may only ever narrow
        # liveness, and returning "live" here for an unverifiable token would
        # instead WIDEN it, turning a proven-dead holder back into a live one.
        return "dead"
    return "dead" if sample.is_birth(scheme, token) is False else "live"


def _read_claim(path: Path) -> "Claim":
    """The claim's four fields, all-``None`` when it says nothing usable.

    ONE parser for one artifact. A second reader beside it is how two callers
    come to disagree about a claim's identity — which is precisely the class of
    bug the birth fields were added to fix, so it must not be reintroduced at the
    parse.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        token = data.get("generation")
        pid = data.get("pid")
        scheme = data.get("birth_scheme")
        birth = data.get("birth_token")
        return Claim(
            generation=str(token) if token else None,
            pid=int(pid) if isinstance(pid, int) else None,
            birth_scheme=str(scheme) if scheme else None,
            birth_token=str(birth) if birth else None,
        )
    except (OSError, ValueError, TypeError):
        return Claim(None, None, None, None)


class Claim(NamedTuple):
    """One ``.execution-lease`` as recorded: who wrote it, and as WHICH process.

    ``birth_scheme``/``birth_token`` are additive and flat, and ``schema`` stays
    1: an older build ignores both, so the same file is a valid claim to every
    generation of reader. Both are ``None`` in a claim written before this
    change (and on Windows), which is the legacy cell — see :func:`_pid_state`.
    """

    generation: str | None
    pid: int | None
    birth_scheme: str | None
    birth_token: str | None

    @property
    def birth(self) -> tuple[str | None, str | None]:
        """The pair the probe compares, spelled once so call sites cannot drift."""
        return (self.birth_scheme, self.birth_token)


@contextmanager
def _stale_recovery_right(session_dir: Path) -> Iterator[bool]:
    """Try to serialize stale takeover with a crash-released kernel lock.

    A persistent side file avoids the same pathname replacement race as the
    lease itself. The kernel lock, not the file's lifetime, is authoritative:
    process death releases it automatically, so recovery cannot become
    immortal. Windows lock failures stay closed because access denial and a
    live contender are intentionally indistinguishable here.
    """
    path = session_dir / RECOVERY_LOCK_NAME
    fd = os.open(path, os.O_CREAT | os.O_RDWR | O_BINARY, 0o600)
    token = secrets.token_hex(16)
    acquired = False
    try:
        if os.name == "nt":
            import msvcrt

            try:
                if os.fstat(fd).st_size == 0:
                    os.write(fd, b"\0")
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                acquired = True
            except OSError:
                yield False
                return
        else:
            import fcntl

            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                acquired = True
            except OSError:
                yield False
                return
        # Metadata is diagnostic only, but records both process identity and a
        # per-attempt token so a surviving file is never mistaken for authority.
        os.ftruncate(fd, 0)
        os.write(
            fd,
            json.dumps({"pid": os.getpid(), "token": token}, separators=(",", ":")).encode(),
        )
        os.fsync(fd)
        yield True
    finally:
        if acquired:
            try:
                if os.name == "nt":
                    import msvcrt

                    os.lseek(fd, 0, os.SEEK_SET)
                    msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError:
                pass
        os.close(fd)


@dataclass(frozen=True)
class SessionLease:
    """One generation claim; release removes only this exact generation."""

    session_dir: Path
    generation: str
    pid: int

    def release(self) -> None:
        """Drop this claim, if it is still this handle's.

        FENCED ON THE GENERATION ALONE, and the birth token is carried but never
        compared — which is a conclusion, not an omission. ``generation`` is 128
        bits of ``secrets.token_hex(16)`` minted per acquisition, so "the claim
        on disk is mine" is already decided with no false-match window at all; a
        token comparison here could only WEAKEN the fence (a same-pid successor
        is already excluded by its fresh generation, and a genuine owner's token
        always matches itself, so the comparison adds nothing but a way to get
        it wrong). Do not "tidy" one in.
        """
        path = self.session_dir / LEASE_NAME
        claim = _read_claim(path)
        if claim.generation != self.generation:
            return
        try:
            path.unlink()
        except OSError:
            return
        # The mirror is never authority. Compare its pid so an old generation
        # cannot erase the compatibility signal written by a successor.
        mirror = self.session_dir / MIRROR_NAME
        try:
            if mirror.read_text(encoding="utf-8").strip() == str(self.pid):
                mirror.unlink()
        except OSError:
            pass


def lease_holder(session_dir: Path) -> tuple[int | None, Literal["none", "live", "uncertain"]]:
    """Who holds ``session_dir``'s transcript lease, WITHOUT taking it.

    Returns ``(pid, verdict)``: ``(None, "none")`` when there is no claim, or the
    claim's holder is PROVEN dead; ``(pid, "live")`` for a live writer; and
    ``(pid_or_None, "uncertain")`` when a claim exists that this platform cannot
    settle (an unparseable claim, an unprovable pid).

    WHY IT MUST NOT ACQUIRE. The callers are guards that decide whether a
    session may be moved or deleted — the mesh move's source-side retire is the
    one this was written for. Probing by acquiring would take the very lease the
    guard is protecting: a live holder would be refused (fine), but a holder that
    had just released would find the GUARD holding the transcript, and the
    runtime the move is waiting on — or the successor it will engage — would then
    fail to start against a lease held by the process asking whether anyone
    holds it. A read of the claim file answers the question and changes nothing.

    "uncertain" is reported separately rather than folded into either answer so
    a guard can fail CLOSED on it: the claim-writer's contract here is that
    uncertainty never permits theft, and deleting a directory is the strongest
    form of theft there is.
    """
    path = session_dir / LEASE_NAME
    if not path.exists():
        mirror = session_dir / MIRROR_NAME
        # The legacy mirror alone, exactly as ``acquire_session_lease`` reads it:
        # an older build that writes only ``.session.pid`` is still a writer.
        try:
            legacy_pid = int(mirror.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            return None, "none"
        state = _pid_state(legacy_pid)
        return (None, "none") if state == "dead" else (legacy_pid, state)
    claim = _read_claim(path)
    if claim.pid is None:
        return None, "uncertain"
    state = _pid_state(claim.pid, expected_birth=claim.birth)
    return (None, "none") if state == "dead" else (claim.pid, state)


def reap_proven_dead_session_claim(session_dir: Path, owner_pid: int) -> bool:
    """Remove only the exact dead owner's lease and compatibility mirror.

    The daemon calls this after discovery proves a record pid is gone. Recovery
    still revalidates under the same kernel lock used by acquisition, because a
    successor may claim the durable transcript between the scan and cleanup.
    Windows remains conservative through ``_pid_state`` and lock acquisition.

    **It stays on the PID question, and the birth token does NOT narrow it.**
    That asymmetry with the acquisition path is deliberate, not an oversight.
    Acquisition takes a claim over through the kernel lock and a whole-claim
    re-read; this function DELETES a claim outright, and the evidence that would
    authorise that here is a token comparison, whose failure mode is a FALSE
    mismatch — the one direction that costs a transcript. A false mismatch would
    let this delete a live writer's protection, with nothing left to fence the
    second writer that follows. So: a pid the platform reports GONE is reaped,
    whatever its token says (a dead pid has no writer, so the ``owner_pid`` this
    was called with IS the writer); a pid that is ALIVE is left alone, whatever
    its token says; and a recycled pid's stale claim is taken over by the path
    that can prove it — ``acquire_session_lease`` — the moment someone opens the
    session. The cost of the asymmetry is a claim that sits on disk slightly
    longer than it could; the cost of the other choice is a forked transcript.
    """
    path = session_dir / LEASE_NAME
    claim = _read_claim(path)
    if claim.pid is None or claim.pid != owner_pid or _pid_state(owner_pid) != "dead":
        return False
    with _stale_recovery_right(session_dir) as may_recover:
        if not may_recover:
            return False
        current = _read_claim(path)
        if current.pid is None or current.pid != owner_pid or current.generation is None:
            return False
        if _pid_state(current.pid) != "dead":
            return False
        # Re-read immediately before unlink so cleanup is generation-fenced even
        # if a future platform changes lock semantics around pathname replacement.
        if _read_claim(path) != current:
            return False
        try:
            path.unlink()
        except OSError:
            return False
        mirror = session_dir / MIRROR_NAME
        try:
            if mirror.read_text(encoding="utf-8").strip() == str(owner_pid):
                mirror.unlink()
        except OSError:
            pass
        return True


def _birth_fields(owner_pid: int) -> dict[str, str]:
    """The claim's additive identity fields for ``owner_pid``, or ``{}``.

    Empty on a platform with no token (Windows) and when the sample fails, which
    leaves that claim on exactly today's pid-liveness path — an unwritten token
    is the legacy cell, and it is deliberately not an error: refusing to write a
    lease because a probe failed would take every session on that host down.

    Sampled for ``owner_pid`` rather than blindly for ``os.getpid()``, so the
    claim's identity describes the pid it names even when a caller passes an
    explicit one. The self case is memoised in ``procstate`` (the token cannot
    change while the process lives), so the write path costs one fork per
    PROCESS on macOS and none on Linux.
    """
    scheme = birth_scheme()
    token = self_birth_token() if owner_pid == os.getpid() else birth_token(owner_pid)
    if scheme is None or token is None:
        return {}
    return {"birth_scheme": scheme, "birth_token": token}


def acquire_session_lease(session_dir: Path, pid: int | None = None) -> SessionLease:
    """Atomically acquire sole-writer ownership, recovering proven-dead claims."""
    owner_pid = os.getpid() if pid is None else pid
    session_dir.mkdir(parents=True, exist_ok=True)
    path = session_dir / LEASE_NAME
    mirror = session_dir / MIRROR_NAME
    # During mixed-version rollout an old writer has only the pid mirror. It is
    # still authoritative when live or uncertain; otherwise a new binary could
    # acquire a lease beside an old binary that knows nothing about leases.
    # No birth token is available on this path by construction: the claim does not
    # exist yet, so there is nothing that recorded the writer's identity. The
    # mirror stays a bare pid (it is read by `resume`, `retention` and `cleanup`
    # as an int, where an unparseable value reads as "no owner" — a second writer
    # against a live one), and identity is recorded in the CLAIM below instead.
    if not path.exists():
        try:
            legacy_pid = int(mirror.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            legacy_pid = None
        if legacy_pid is not None and _pid_state(legacy_pid) != "dead":
            raise SessionLeaseHeldError(session_dir, legacy_pid)
    generation = secrets.token_hex(16)
    payload = json.dumps(
        {
            "schema": 1,
            "session_id": session_dir.name,
            "generation": generation,
            "pid": owner_pid,
            # THE WRITER'S OWN BIRTH, additive and flat, schema unchanged. It is
            # sampled for `owner_pid` (not blindly for `os.getpid()`) so the claim
            # describes the pid it names even when a caller passes one; it is
            # omitted entirely where the platform has no token (Windows), which
            # leaves that claim on today's pid-liveness path.
            **_birth_fields(owner_pid),
        },
        separators=(",", ":"),
    ).encode()

    while True:
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | O_BINARY, 0o600)
        except FileExistsError:
            inspected = _read_claim(path)
            if (
                inspected.pid is None
                or _pid_state(inspected.pid, expected_birth=inspected.birth) != "dead"
            ):
                raise SessionLeaseHeldError(session_dir, inspected.pid)
            with _stale_recovery_right(session_dir) as may_recover:
                if not may_recover:
                    # A live recoverer is indistinguishable from ownership until
                    # it publishes its successor. Fail closed instead of racing it.
                    raise SessionLeaseHeldError(session_dir, _read_claim(path).pid)
                current = _read_claim(path)
                if (
                    current != inspected
                    or current.pid is None
                    or _pid_state(current.pid, expected_birth=current.birth) != "dead"
                ):
                    # The whole claim changed — generation, pid OR birth token —
                    # or became live, or cannot still be proven dead. A changed
                    # token is a changed WRITER (a successor that took the claim
                    # and re-used the pid), not a new number to steal, so the
                    # refusal stands and this recoverer yields to it.
                    raise SessionLeaseHeldError(session_dir, current.pid)
                tombstone = session_dir / f"{LEASE_NAME}.stale.{secrets.token_hex(8)}"
                try:
                    os.replace(path, tombstone)
                    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | O_BINARY, 0o600)
                except OSError:
                    # Windows rename denial and any unexpected successor both
                    # stay closed; neither permits speculative ownership.
                    raise SessionLeaseHeldError(session_dir, current.pid) from None
                finally:
                    try:
                        tombstone.unlink()
                    except OSError:
                        pass
        try:
            os.write(fd, payload)
            os.fsync(fd)
        finally:
            os.close(fd)
        try:
            mirror.write_text(str(owner_pid), encoding="utf-8")
        except OSError:
            pass
        return SessionLease(session_dir, generation, owner_pid)
