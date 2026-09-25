"""A pre-imported runtime interpreter this CONSOLE forked, adopted by its next cold engage.

WHY THIS EXISTS — THE MEASUREMENT
=================================
A cold engage (a brand-new conversation, or switching to one whose runtime is
not running) spawns ``python -m local_operator.session.runtime.process`` and
waits for it to publish. Under the operator's real load (~17-25 live runtimes,
load average 100-140 on 14 cores, measured 2026-09-24) that wait was 3-4 s wall
(``scripts/bench_cold_engage.py``: child spawn -> publish median 2.4-3.0 s).
Attributed with a CPU-clock cProfile of a fresh child:

    interpreter + the runtime's import graph     ~1,100 ms CPU   (~70%)
    session construction (create_session)          ~440 ms CPU
    serve + publish                                  ~15 ms CPU

and on that host 100 ms of CPU costs 1.0-1.7 s of WALL (a pure CPU spin,
measured the same afternoon), because the scheduler is shared with ~1,000
processes. So the only lever is removing CPU from the critical path, and import
pruning cannot do it: session construction needs all but a handful of the ~450
modules it loads. A standby pays that import, and the next cold engage hands its
spawn to the already-warm process instead of forking a fresh interpreter.

WHY THE CHANNEL IS PRIVATE — THE ONE PROPERTY EVERYTHING ELSE FOLLOWS FROM
==========================================================================
**This console forks the standby, and the only channel to it is the socketpair
end that fork inherited.** There is no path, port, lock file or directory any
other process can find, bind or connect to, because there is none at all.

That is not a simplification of the earlier design; it is the correction of a
security defect in it (agent review round 1, R1-1). That revision kept one
standby per MACHINE and reached it over ``<root>/run/standby/standby.sock``.
Same-uid processes cannot be isolated from each other — a model-authored
``bash`` call runs as exactly this uid — so anything writable by this uid can be
bound FIRST by any of them, and the console would then have handed that process
the operator capability (issue #1310's escalation: the capability is what lets
a runtime's own gate be moved from ``ask`` to ``auto``). A peer-pid or
``LOCAL_PEERPID`` check does not close it either: the impostor's pid is just as
real, and ``secrets/peer.py`` records that same-uid peers can only be identified
by LINEAGE. Lineage is what an inherited descriptor IS: the value can reach a
process only if this process's own ``fork``/``exec`` put the descriptor in it,
so the capability cannot reach anything this console did not start. There is no
ownership, mode, path or handshake that could have the same strength.

THE CONSEQUENCE, STATED PLAINLY: one standby per WARMING CONSOLE, not one per
machine. On this host the desktop app's ``lop serve`` daemon is the singleton
that warms (so the desktop surface keeps its spares per machine, exactly as
intended), while each interactive TUI that opens new conversations warms its
own. The earlier machine-wide sharing is not recoverable by any means that keeps
the capability out of a stranger's hands — see the trade-off section of the PR
body — and the mitigations here are: warming is opt-in per process and only
enabled at the TUI and ``serve`` launch points; a spare is warmed at the
console's launch and refilled behind every engage by the supervisor thread
(:func:`notify_engage`), never on an engage's critical path; every standby exits
on adoption, when its console goes away (EOF on the inherited descriptor,
detected immediately), when its root disappears, and on idle where the slot
reaps (the daemon's keeps warm — see :data:`DAEMON_SPARE_DEPTH`).

WHAT IT IS NOT — THE OPERATOR'S CONSTRAINTS, AND HOW EACH IS HELD
================================================================
(a) ONE PER CONSOLE, and never more: the process holds at most one standby, and
    a spawn is refused while one is alive. A console with no standby spawns cold
    exactly as before — the standby is an optimisation, never a dependency.

(b) NEVER A STALE BUILD, NEVER A STALE CONFIG. Adoption is refused — the standby
    exits — when any of these moved since it finished warming:

    * the interpreter the caller would spawn (``launch._spawn_interpreter``,
      i.e. the CURRENT generation after a ``lop-update``) is not in the same venv
      the standby runs;
    * ``update.installed_build`` (version + ``.lop-source`` ref) differs;
    * any loaded ``local_operator`` module file's mtime moved (a same-path
      rebuild, an editable checkout edited under it — measured at 0.2 ms for the
      ~140 files, so it is checked on every request rather than sampled);
    * ``config.yml``'s stat key (``config.config_file_key``) differs from the one
      recorded at warm, for the operator's explicit concern that a spare warmed
      under one configuration must not construct a session under another.

    A requester whose root or warm-sensitive environment differs is DECLINED (it
    keeps waiting for a matching one) rather than retired, because a standby that
    exited on every mismatch would be killed by whichever host engaged next.

(c) INVISIBLE AS A SESSION until it is adopted. The standby publishes nothing a
    session reader lists: no record in ``run/mobile`` (so not in ``lop
    sessions``, the desktop feed or the mobile list), no boot record, no lease,
    no transcript, no analytics row — and it never runs ``create_session``, so it
    never takes the store-maintenance or analytics passes, which start from
    session construction. It does not arm the stall watchdog
    (``stall_watchdog.arm`` has exactly one call site, the runtime ``__main__``
    guard, and a standby reaches it only through :func:`_become_runtime`). Its
    argv is ``-m local_operator.session.runtime.standby``, which the residency
    census (``reclaim.parse_process_row``) matches as a whole word against
    ``RUNTIME_MODULE`` and therefore does NOT count as a runtime.

    AT adoption it must become countable and attributable, because it is now a
    runtime the sweep's orphan protection has to see:

    * the module word and the ``[standby]``/``id=--------`` label are rewritten
      IN PLACE in the process's own argv memory (``_rename_argv``) — same
      lengths, and macOS/Linux ``ps`` read argv straight from that memory
      (verified with ``ps -ww``). Where the rewrite is impossible, the adoption
      is refused rather than leaving an uncountable runtime behind;
    * a BOOT RECORD is written BEFORE construction starts. That is what fixes
      R1-3: the adopted process is older than its session (its ``etime`` counts
      from the warm) and its ``ps -E`` environment block is this console's, so
      ``reclaim``'s young rung and ``session_id_of`` had nothing to read. The
      boot record is the artifact those readers already prefer, and writing it
      one step earlier makes both true for the whole construction window.

(d) COST, measured: max RSS 47-158 MB (median ~130 MB) per standby, sitting at 0%
    CPU in a blocking read. It is warmed on the supervisor's thread, so it costs
    an engage nothing; it exits on adoption, on idle where the slot reaps
    (:data:`IDLE_REAP_S`; the daemon's spares keep warm), and when its console
    goes away.

    HOW LONG THE WARM TAKES, AND WHY IT IS NOT PUT IN A NICER SCHEDULING CLASS
    (QA round 1 QW1, QA round 2 Q2-3). The first revision warmed inside Darwin's
    background band (``PRIO_DARWIN_BG``) for its whole life and claimed "about a
    minute". That was false on the host this exists for: measured at load 180-227,
    standbys spent 16 MINUTES of wall on 0.25-0.29 s of CPU each and had not
    finished, so the feature failed open to a cold spawn exactly when it mattered.
    Round 1 added a CPU-progress test and a hard ceiling on the polite phase, and
    round 2 measured that a 20 s/45 s window STILL produced 125.2 s at load 150.3,
    because the sampling is starved by the very band it is sampling: a process in
    the background band cannot bound its own warm.

    So the band is gone. The warm runs at the host's NORMAL priority, which makes
    its duration the host's own scheduling latency and nothing invented here:
    measured on this host, **p50 2.7 s / max 3.7 s over 7 runs at load 153-166**,
    on 1.7 s of CPU. It is paid on a daemon thread AFTER an engage, so no engage
    ever waits on it. What the design gives up with the band is the claim that a
    speculative spare never competes with live work; what it keeps is that the
    spare is opt-in, lazy, capped per root, and reaped on idle. A slow warm is
    strictly better than a starved one: a late standby still serves the NEXT
    engage, while an abandoned one serves nothing.

    The user-visible consequence, stated rather than implied: a host's FIRST new
    conversation after a boot can be cold, and so can the next one if it comes
    before the warm finishes (see the PR body's per-surface table for both).

    THE COUNT IS CAPPED PER ROOT, FOR THE WHOLE MACHINE (agent review round 3).
    Sharing ONE adoptable spare across consoles is not achievable: a spare is a
    private descriptor to a child of exactly one console (that is the R1-1 fix),
    so serving another console's engage would need a rendezvous between two
    same-uid processes — and any path one of them can bind is a path an impostor
    can bind first: the impostor then receives a real console's operator
    capability, which is the escalation R1-1 proved. The narrow reason is worth
    stating exactly, because the broad version of it is FALSE and this codebase
    contradicts it: same-uid peers CAN be authenticated — ``secrets/peer.py`` does
    it with the kernel's own attestation — but that names the peer to a SERVER. It
    gives a CLIENT nothing to compare a candidate server against, and the console
    is the client here, so it cannot tell a real warmer from an impostor. There is
    no code-identity check available either (both are the same interpreter), and
    privilege separation would be a redesign. So the count is capped: an ``flock``
    slot per root, kernel-released on death, holding TWO slots at most for the
    whole machine — the daemon's (a singleton, and the desktop surface must not
    lose its spare to whichever TUI started first) and one shared by every other
    console. Each holder keeps at most :data:`DAEMON_SPARE_DEPTH` /
    :data:`TUI_SPARE_DEPTH` private spares. A console that cannot take its slot
    retries at :data:`SLOT_RETRY_S` and spawns cold until it wins; that is the
    trade, and it is what turns ~20 spares at ~136 MB each into two slots' worth.

    A CONSEQUENCE FOR THE CONSTRAINTS ABOVE: the desktop app's ``lop serve``
    daemon keeps its own spares, and every TUI on the root shares the other
    slot, so only the first TUI to warm gets the win and the rest spawn cold
    while it is held. That is a per-console cost too — a time cost rather than a
    memory one — and the body's cap section measures it rather than implying it.

THE SUPERVISOR — INVARIANT P, AND THE DEAD-ENDS IT CLOSES
========================================================
While a warming console is live (``_WARMING`` and not disabled and its root
exists), at every instant either **(a)** a ready spare is in hand, **(b)** a warm
is in flight, or **(c)** a retry is scheduled at a deadline ≤ ``RETRY_CEIL_S``.
Every transition out of (a)/(b) ends in (c) BY CONSTRUCTION: every clear of a
tracked spare funnels through :func:`note_spare_gone`, the one choke point, which
schedules the refill. The first version of this module had no such owner, and
five separate guards could each leave a live console with no spare and NOTHING
scheduled — the production state this section was written for (the daemon held
its slot with no ``[standby]`` child, indefinitely):

    S1  a warm that FAILED cleared the slot and stopped; the next engage re-warmed.
    S2  a standby that exited within ``REWARM_MIN_LIFE_S`` of its spawn was
        "exited at once; not re-warming" — reproduced: 90 s with no replacement.
    S3  a second such exit inside 60 s was refused, not deferred — permanent.
    S4  a DECLINED spare (warm-sensitive environment mismatch) was kept alive
        forever while ``ensure_warm`` refused to warm beside it.
    S5  a slot held by another console was never re-claimed.
    S6  ``_spawn_standby`` raising after the claim was swallowed; no retry.
    S8  a consumed spare's replacement chain could land in S1/S2/S3 — and the
        consumed entry itself stayed tracked, so the NEXT engage read EOF off
        its closed channel and RETIRED it: opening a second chat SIGTERMed the
        first chat's still-live runtime (measured on an isolated daemon: the
        adopted runtime killed ~1 s into the next engage, which took 7.8 s
        against 6.5 s cold).

What replaces them: ONE supervisor thread per warming console owns every fork;
``note_spare_gone`` schedules the refill for every clear; a young death or a
failed attempt is RETRIED under a doubling backoff (1 s → 30 s, reset when a
spare reaches READY or a refill lives a normal life) instead of being dropped; a
warm that never finishes inside ``WARM_DEADLINE_S`` is retired by exact pid; a
spare declined ``DECLINE_RETIRE_N`` times is retired and refilled (a standing
mismatch means it can never serve its own console); a slot claim is retried at
``SLOT_RETRY_S``; and an adopted spare LEAVES the pool immediately — it carries
a FLAG set from inside the adoption lock, so a concurrent attempt racing for the
same spare finds the flag and moves to the next candidate — so no path can
signal a process that is now a session's runtime. The daemon keeps
``DAEMON_SPARE_DEPTH`` spares and never idle-reaps them; every other console
keeps one and keeps ``IDLE_REAP_S``. The idle policy travels to the child as
``STANDBY_IDLE_ENV`` (additive; a child from an older build keeps its 900 s).

RIG NOTE (agent review round 1, R1-6, and it is now structural rather than a
policy): a rig never leaves a standby behind, because a standby's life is tied
to the process that warmed it — when the rig ends, its descriptor closes and the
standby sees EOF and exits. A rig that keeps its ROOT while its warming process
stays alive can still hold one for up to :data:`IDLE_REAP_S`; the root-gone check
covers the case where the root is deleted first.

WHAT IT DOES NOT FIX: session construction itself (~440 ms CPU before this PR's
config memo, ~180 ms after) still runs after adoption, because it depends on the
session id, cwd and model the engage names. See the PR body for the bound that
leaves on a host at load 100+.
"""

from __future__ import annotations

import ctypes
import json
import logging
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: The module a standby runs as. SAME LENGTH as ``types.RUNTIME_MODULE`` on
#: purpose: adoption rewrites this word in place (see ``_rename_argv``), and an
#: in-place rewrite cannot grow the string. ``test_standby`` pins the lengths.
STANDBY_MODULE = "local_operator.session.runtime.standby"

#: The argv flag carrying the INHERITED descriptor number the standby listens on.
#: A descriptor number is not a secret: the descriptor itself is what nobody else
#: has, which is the whole property this design rests on. Spelled once, like
#: ``approval.OPERATOR_FD_FLAG``, because the spawner and the child are different
#: processes and a drift between them is a silent "no standby".
#: The argv word naming the descriptor this process must read its capability from.
#:
#: THE SAME SPELLING AS ``approval.OPERATOR_FD_FLAG``, deliberately (agent review
#: round 2, nit). A forked child's exec argv ends with ``--operator-fd <n>``; an
#: adopted standby is that same process with the same job, so its row must say the
#: same thing — and it can, because the NUMBER it was started with is where the
#: capability ends up: ``_commit_adoption`` moves it there with ``dup2`` once the
#: private channel is closed. Before adoption the number names the channel instead,
#: which no reader can mistake for a runtime: the only reader of that descriptor is
#: the runtime itself, out of its own ``sys.argv`` and never out of a process
#: listing, and the census requires the ``-m`` module word this process has not yet
#: taken. (The module that reads it is named nowhere here on purpose: the seam test
#: that keeps the capability's blast radius to a handful of modules scans for the
#: name, and a comment is enough to trip it — which is how this paragraph was
#: written the second time.)
#:
#: Equal in LENGTH is what makes the swap possible at all: ``_rename_argv`` rewrites
#: in place and refuses when the two words differ in size (which is how this was
#: found — a 12-byte ``--standby-fd`` cannot become a 13-byte ``--operator-fd``).
STANDBY_FD_FLAG = "--operator-fd"

#: How long a standby waits for an adoption before exiting on its own, for the
#: slots that reap on idle at all. Long enough that an operator who opens new
#: conversations every few minutes always finds one warm; short enough that a
#: console gives the memory back within the quarter hour. NOT universal any more:
#: the desktop daemon's slot keeps its spares warm for the life of the console
#: (see ``DAEMON_SPARE_DEPTH``), and the window travels to the child in
#: ``STANDBY_IDLE_ENV``. A standby is no longer shared between consoles, so this
#: is purely the memory bound rather than an availability policy.
IDLE_REAP_S = 900.0

#: How long the engage side waits for a standby's answer before giving up and
#: spawning cold. A healthy standby answers in single-digit ms (it is parked in a
#: read); the bound exists for one that is wedged, so a wedged standby costs at
#: most this on top of today's cold spawn.
ADOPT_TIMEOUT_S = 2.0

#: Environment switch that turns the whole mechanism off — no warm and no
#: adoption — for an operator who wants the cold spawn exactly, and for the test
#: suite, which sets it process-wide in ``tests/conftest.py``: warming is opt-in
#: per process (:func:`enable_warming`, called only at the TUI and ``serve`` CLI
#: launch points), but the suite drives exactly those launch points through
#: ``cli.main()``, and a standby is a detached process by design, so without the
#: switch a run leaves live interpreters behind (measured, and why it is there).
DISABLE_ENV = "LOP_RUNTIME_STANDBY_DISABLED"

#: The keys of the spawn contract a standby is allowed to receive. The same
#: names ``launch._spawn_runtime`` writes into a cold child's environment, and
#: nothing else: a standby must end up in exactly the state a cold child starts
#: in.
CONTRACT_KEYS = (
    "LOP_MOBILE_CHILD_CWD",
    "LOP_MOBILE_CHILD_RESUME",
    "LOP_MOBILE_CHILD_PROVIDER",
    "LOP_MOBILE_CHILD_MODEL",
    "LOP_MOBILE_CHILD_EFFORT",
    "LOP_MODEL_SELECTION_OVERRIDE",
    "LOP_RUNTIME_DEFER_MATERIALISE",
)

#: The modules warmed beyond the runtime's own top-level imports and
#: ``session_factory._WARM_IMPORTS``: the smallest set whose transitive closure
#: covers every module a first session construction imports lazily (a greedy
#: cover over the 199 modules measured, 76 of them ours). A module missing here
#: is not a correctness problem — it is imported on first use exactly as before —
#: only a slower adoption.
_WARM_EXTRA: tuple[str, ...] = (
    "local_operator.classification",
    "local_operator.tools.registry",
    "local_operator.ui_console",
    "local_operator.tui.notify",
    "local_operator.secrets.access",
    "local_operator.agents",
    "local_operator.guides",
    "local_operator.session.attention",
    "local_operator.update",
    "local_operator.wakes.store",
    "local_operator.browser_bridge.resources",
    "local_operator.config_watch",
    "local_operator.context_files",
    "local_operator.fork",
    "local_operator.mcp.resources",
    "local_operator.model.prices",
    "local_operator.secrets.session",
    "local_operator.session.cleanup",
    "local_operator.session.runtime.journal",
    "local_operator.variables",
)

#: Environment names a WARMED interpreter may already have acted on, so a
#: requester whose values differ must get a cold child instead. Measured, not
#: guessed: an ``os.environ`` read hook over the whole warm recorded exactly
#: ``HOME``, ``LANG``/``LANGUAGE``/``LC_*``, ``ARCHFLAGS``, ``OTEL_*``,
#: ``PYDANTIC_*``, ``PYTHON*``/``_PYTHON*`` — and nothing else; every product
#: variable (``LOCAL_OPERATOR_*``, ``LOP_*``) is read at CALL time, after
#: adoption has installed the requester's environment. ``TMPDIR``/``TZ`` are
#: added because ``tempfile`` and ``time`` cache them on first use, and
#: ``LOP_BUILD_PREFIX`` because the warm's build stamp is read through it. Every
#: other name (``PATH``, API keys, a terminal's variables, the desktop token) is
#: simply replaced by the requester's value before the runtime starts, which is
#: exactly what a cold child would have inherited.
_WARM_SENSITIVE_NAMES = frozenset(
    {"HOME", "LANG", "LANGUAGE", "ARCHFLAGS", "TMPDIR", "TZ", "LOP_BUILD_PREFIX"}
)
_WARM_SENSITIVE_PREFIXES = ("LC_", "OTEL_", "PYDANTIC_", "PYTHON", "_PYTHON")


def _venv_of(interpreter: str) -> str:
    """The environment an interpreter path runs in, comparable across spellings.

    NOT ``realpath(interpreter)``: a standby is exec'd through the branded
    hardlink beside the interpreter (``procname.spawn_identity``), so its
    ``sys.executable`` is ``<venv>/bin/Local Operator`` while the requester names
    ``<venv>/bin/python3`` — and a venv's ``python`` is itself a symlink to the
    base interpreter, which every venv on the machine shares. The venv directory
    is the identity that decides which build is imported.
    """
    return os.path.realpath(Path(interpreter).parent.parent)


def disabled() -> bool:
    """Off by switch, and off where an inherited descriptor cannot be trusted.

    The private channel is a POSIX socketpair; Windows has no ``pass_fds``, and
    its inheritance rules would hand the console's descriptor to any child it
    starts. Windows therefore keeps the cold spawn.
    """
    if os.name != "posix":
        return True
    return os.environ.get(DISABLE_ENV, "") not in ("", "0", "false", "no", "off")


def _warm_sensitive(env: "dict[str, str] | os._Environ[str]") -> dict[str, str]:
    return {
        key: value
        for key, value in env.items()
        if (key in _WARM_SENSITIVE_NAMES or key.startswith(_WARM_SENSITIVE_PREFIXES))
        and key not in CONTRACT_KEYS
        and key != "LOP_RUNTIME_ADOPT_SESSION"
    }


# ---------------------------------------------------------------------------
# The console side: one standby, one private descriptor, no path anywhere
# ---------------------------------------------------------------------------

#: Whether THIS process warms standbys. Off by default: only a long-lived
#: interface host turns it on (:func:`enable_warming`), so a unit test, a script,
#: a ``lop exec`` or a benchmark that engages a runtime never leaves an idle
#: interpreter behind it. A list, not a bare global, so the flag is mutated
#: rather than rebound.
_WARMING: list[bool] = [False]

#: The spares this process holds — READY, or warming with the supervisor on the
#: clock — at most :data:`DAEMON_SPARE_DEPTH` / :data:`TUI_SPARE_DEPTH` of them.
#: Only the supervisor (:func:`_spawn_one`, append) and :func:`note_spare_gone`
#: (remove, for every clear) mutate it, under ``_LOCK``.
_POOL: list["_Standby"] = []

#: The root this console warms for, set once by :func:`enable_warming`.
_ROOT: list["Path | None"] = [None]

#: Serialises the pool and the scheduling state between the engage path, the
#: supervisor and the tests that drive them.
_LOCK = threading.Lock()

#: Serialises reads and writes on the spare CHANNELS (one lock for the whole
#: pool: there are at most two spares, and an adoption owns its channel for the
#: whole handshake). WHY IT EXISTS: readiness is a one-byte peek with a zero
#: timeout, the supervisor peeks every tick, and without this lock a peek during
#: an adoption could swallow the first byte of the adoption's REPLY — the reader
#: then sees a torn frame and a healthy spare is retired.
_CHANNEL_LOCK = threading.Lock()

#: When (``time.monotonic``) the supervisor may next attempt a spawn. Every
#: clear of a spare and every failed attempt pushes it; the invariant is that a
#: console whose pool is empty and whose root exists always has a deadline of at
#: most :data:`RETRY_CEIL_S` (or the root park probe) ahead — see the module
#: docstring's invariant P.
_NEXT_AT: list[float] = [0.0]

#: Consecutive failed warm attempts, for the doubling backoff. Reset when a
#: spare reaches READY or a refill lives out :data:`REWARM_MIN_LIFE_S`: the
#: backoff exists for a root that cannot warm, not for the normal
#: consume-and-replace cycle.
_ATTEMPTS: list[int] = [0]

#: Wakes the supervisor: the engage path's nudge, and the first fill.
_NUDGE = threading.Event()

#: The one supervisor thread, started by :func:`enable_warming`.
_SUPERVISOR: list["threading.Thread | None"] = [None]

#: The desktop daemon's slot. It is a singleton per root (``lop serve``), and it
#: serves the surface the operator opens conversations from, so it never competes
#: with the TUIs.
SLOT_DAEMON = "daemon"

#: The slot every OTHER console on the root shares.
SLOT_TUI = "tui"

#: ``(root, slot)`` -> the lock descriptor this process holds, for its lifetime.
#: Keyed by ROOT as well as slot because the invariant is per root (agent review
#: round 3, m3-1): an isolated-root session must never hold or miss the operator's
#: slot, and the key carrying only the slot was right by accident — one process
#: never held two roots' slots at once solely because ``ensure_warm`` refuses a
#: root that is not ``config_dir()``.
_SLOTS: dict[tuple[str, str], int] = {}

#: Serialises the check-then-set on ``_SLOTS`` (round 3, n3-2). Its own lock, not
#: ``_LOCK``: ``_take_slot`` does file I/O while holding it, and ``_LOCK`` is the
#: engage path's critical section.
_SLOT_LOCK = threading.Lock()

#: This process's slot, set once by :func:`enable_warming`.
_ROLE: list[str] = [SLOT_TUI]


class _Standby:
    """A forked, warming interpreter and the private descriptor to it.

    ``sock`` is this process's end of the socketpair whose other end the child
    inherited at ``exec``. Nothing else on the machine holds it, which is why
    possession of it is the whole authentication story (see the module
    docstring): the capability is handed over this descriptor and nowhere else.
    """

    def __init__(
        self,
        proc: "subprocess.Popen[bytes]",
        sock: socket.socket,
        root: Path,
        slot: str = SLOT_TUI,
    ) -> None:
        self.proc = proc
        self.sock = sock
        self.root = root
        #: The root slot this spare occupies, so its replacement asks for the same
        #: one rather than quietly taking the other.
        self.slot = slot
        #: Set once the child says it is warm. Kept as a cached answer so the
        #: engage path never blocks on a read that has not arrived yet.
        self.ready = False
        #: Set the instant THIS spare's adoption handshake succeeded, under
        #: ``_CHANNEL_LOCK``, before anything outside it can run (agent review
        #: round 1, B1). From that moment the pid is a session's runtime, not a
        #: spare: :func:`_retire` refuses it, :func:`note_spare_gone` can never
        #: terminate it, and a concurrent attempt that still finds it tracked
        #: sees this flag and moves on to the next candidate.
        self.adopted = False
        #: When the fork happened (``_now()``): the young-death backoff and the
        #: wedged-warm deadline are both measured from it.
        self.spawned_at = _now()
        #: Consecutive declines from THIS spare, and the last reason, for the
        #: S4 retirement (:data:`DECLINE_RETIRE_N`).
        self.declines = 0
        self.decline_reason = ""

    def alive(self) -> bool:
        return self.proc.poll() is None

    def close(self) -> None:
        """Drop the descriptor. Never raises; called on every exit path."""
        try:
            self.sock.close()
        except OSError:
            pass


def enable_warming(root: "Path | None" = None, *, daemon: bool = False) -> None:
    """Make this process a warmer, and start the supervisor that keeps a spare in hand.

    Called by the TUI and the ``serve`` daemon at their launch points. From here
    on the supervisor owns every fork: it fills the pool immediately (so the FIRST
    new conversation already finds a standby), refills behind every engage, and
    never leaves the pool empty with nothing scheduled (invariant P, module
    docstring). Never raises: a console that cannot warm simply spawns cold.

    ``daemon`` picks the slot AND with it the depth and idle policy (see
    :func:`_take_slot`, :data:`DAEMON_SPARE_DEPTH`): the daemon is a singleton and
    keeps its own spares for the life of the console, every other console shares
    ``SLOT_TUI``, keeps one and reaps it on idle. It is declared HERE, once per
    process, rather than passed to each warm: the engage path merely nudges
    (:func:`notify_engage`), and a per-call parameter would need every caller to
    remember which kind of console it is.
    """
    if disabled() or os.environ.get("LOP_MOBILE_CHILD_RESUME"):
        return
    try:
        from local_operator.paths import config_dir
        from local_operator.session.runtime.launch import _spawn_interpreter

        target = root if root is not None else config_dir()
        # A real directory path or nothing: a standby outlives its warmer, so a
        # root that is a test double (or a relative path resolved against some
        # later cwd) would leave a process bound to a directory nobody owns.
        if not isinstance(target, Path) or not target.is_absolute():
            return
        # The standby resolves its own root from the environment it inherits, and
        # a console has exactly one. An engage for a different root is declined
        # by ``try_adopt`` rather than served from a spare warmed for another
        # store.
        if Path(target) != config_dir():
            return
        # Resolvable NOW, so a process that could not spawn at all stays off
        # rather than starting a supervisor that fails every attempt. The
        # supervisor re-resolves per attempt on purpose: after a ``lop-update``
        # the next replacement must be the new generation's interpreter.
        _spawn_interpreter()
    except Exception:  # noqa: BLE001 — a missing warm is a slower first engage
        logger.debug("could not resolve a standby target", exc_info=True)
        return
    _WARMING[0] = True
    _ROLE[0] = SLOT_DAEMON if daemon else SLOT_TUI
    _ROOT[0] = Path(target)
    _start_supervisor()


def _start_supervisor() -> None:
    """Start (or just wake) the one thread that owns every fork."""
    thread = _SUPERVISOR[0]
    if thread is not None and thread.is_alive():
        _NUDGE.set()
        return
    thread = threading.Thread(target=_supervisor_main, name="lop-standby-supervisor", daemon=True)
    _SUPERVISOR[0] = thread
    thread.start()
    # The first fill does not wait out a tick: a warm started now is warm sooner.
    _NUDGE.set()


def notify_engage() -> None:
    """Wake the spare supervisor after an engage's spawn decision.

    Called from ``launch.engage_runtime`` where the old ``warm_in_background``
    was, still AFTER the spawn and still off the critical path — but it no longer
    forks: the supervisor owns every fork, so a refill has exactly one owner and
    it is the one that holds invariant P (module docstring).
    """
    if not _WARMING[0] or disabled():
        return
    _NUDGE.set()


#: How long a standby which exits without being adopted must have lived before
#: its replacement is warmed at once rather than under the backoff. A death
#: younger than this is the shape a deleted root or a broken tree produces (a
#: spare that cannot live), and it is the one class that must not spin: its retry
#: is SCHEDULED under the backoff — never dropped, which is what the first
#: version of this module did (one such death left a live console holding the
#: slot with no spare and nothing scheduled until some later engage).
REWARM_MIN_LIFE_S = 5.0

#: Retry backoff, doubling from base to ceiling: 1, 2, 4, 8, 16, 30, 30…
#: Applies to every failed attempt class (a young death, a failed warm, a wedged
#: spare, a spawn that raised); a consumed spare and a refill that lived a normal
#: life are replaced at once.
RETRY_BASE_S = 1.0
RETRY_CEIL_S = 30.0

#: A spare not READY within this of its spawn is wedged: retired by exact pid and
#: replaced. ~24x the measured worst warm (2.7 s p50 / 3.7 s max at load
#: 153-166), so it cannot fire on a merely slow host; it bounds the exposure of a
#: warm that will never finish.
WARM_DEADLINE_S = 90.0

#: Consecutive declines from the SAME spare before it is retired and refilled.
#: One decline is normally about the requester (a foreign venv, a differing
#: environment) and the spare must stay for its own console; a standing mismatch
#: means this console's spare can never serve its console, which is the state the
#: retirement exists for.
DECLINE_RETIRE_N = 3

#: How long before the supervisor re-attempts a slot another console holds, and
#: before it re-probes a root that is gone (or warming that is switched off).
SLOT_RETRY_S = 30.0
ROOT_PROBE_S = 60.0

#: How often the supervisor wakes between fills. The same thread detects a
#: spare's death, so this bounds how much of a replacement's budget is detection
#: (0.25 s against a warm of seconds).
SUPERVISOR_TICK_S = 0.25

#: Spares each console keeps in hand, per slot. The daemon keeps TWO: its spare is
#: the desktop first-send budget, and a second new chat must not land in the
#: replacement's warm window (measured: a send racing the replacement warm paid
#: 2688-3119 ms). What depth 2 covers is exactly THE SECOND chat: four
#: back-to-back new chats measured 3 adopted / 1 cold at depth 2 against 2/4 at
#: depth 1, so a third send in rapid succession can still race both refills —
#: that residue belongs to the desktop draft pre-engage, not to more spares
#: (agent review round 1, Q3). It is this module's one memory trade, +~130 MB
#: (median spare RSS) while the second spare lives. Every other console keeps one.
DAEMON_SPARE_DEPTH = 2
TUI_SPARE_DEPTH = 1

#: The env var carrying the idle window to the standby child (seconds; 0 or less
#: = never reap on idle). Additive: a child from an older build ignores it and
#: keeps :data:`IDLE_REAP_S`, and a console from an older build never sets it.
#: Set at spawn; consumed only by ``_await_request``.
STANDBY_IDLE_ENV = "LOP_STANDBY_IDLE_S"


def _now() -> float:
    """The console-side clock. One indirection so tests can drive the schedule."""
    return time.monotonic()


def _depth_for(slot: str) -> int:
    """How many private spares a console warming ``slot`` keeps."""
    return DAEMON_SPARE_DEPTH if slot == SLOT_DAEMON else TUI_SPARE_DEPTH


def _idle_for(slot: str) -> float:
    """The idle window a spare for ``slot`` is spawned with: 0 = keep warm."""
    return 0.0 if slot == SLOT_DAEMON else IDLE_REAP_S


def _supervisor_main() -> None:
    """The supervisor loop: wake, reconcile, sleep until the next deadline or tick."""
    while True:
        _NUDGE.wait(_supervisor_wait())
        _NUDGE.clear()
        try:
            _supervise_once()
        except BaseException:  # noqa: BLE001 — the supervisor must never die
            logger.debug("standby supervisor tick failed", exc_info=True)


def _supervisor_wait() -> float:
    """Seconds until the supervisor should run again."""
    with _LOCK:
        pool_short = len(_POOL) < _depth_for(_ROLE[0])
        due = _NEXT_AT[0] - _now()
    if pool_short:
        return max(0.0, min(SUPERVISOR_TICK_S, due))
    return SUPERVISOR_TICK_S


def _supervise_once() -> None:
    """One reconcile pass: prune and probe the pool, then fill it if due."""
    if not _WARMING[0] or disabled():
        # Parked: warming was switched off, or is off because this process is a
        # runtime child. Re-probe rather than spin.
        _push_deadline(ROOT_PROBE_S)
        return
    root = _ROOT[0]
    if root is None:
        return
    if not root.is_dir():
        # ``root-gone``: a replacement would exit for the same reason, so the
        # loop breaker here is the CAUSE (re-probe), not a fork per backoff tick.
        _push_deadline(ROOT_PROBE_S)
        return
    with _LOCK:
        snapshot = list(_POOL)
    for warm in snapshot:
        if not warm.alive():
            note_spare_gone(warm, "exited")
            continue
        if not warm.ready and _ready(warm):
            logger.info("standby up for %s", root)
        elif not warm.ready and _now() - warm.spawned_at > WARM_DEADLINE_S:
            logger.warning(
                "standby for %s never warmed in %.0fs; retiring and refilling",
                root,
                _now() - warm.spawned_at,
            )
            note_spare_gone(warm, "warm-wedged", terminate=True)
    _fill(root, _ROLE[0])


def _fill(root: Path, slot: str) -> None:
    """Spawn spares up to the role's depth, no sooner than ``_NEXT_AT``.

    Only the supervisor calls this, so there is no concurrent spawner to
    serialise against; the loop re-reads the state each pass because spawning and
    scheduling both move it.
    """
    depth = _depth_for(slot)
    while True:
        with _LOCK:
            if len(_POOL) >= depth:
                return
            if _now() < _NEXT_AT[0]:
                return
        from local_operator.paths import config_dir

        if Path(root) != config_dir():
            # The same rule ``ensure_warm`` has always had: the child resolves its
            # root from the environment it inherits, so warming for another store
            # would warm the wrong one. Park and re-probe.
            _push_deadline(ROOT_PROBE_S)
            return
        if not _claim_slot(root, slot):
            return
        _spawn_one(root, slot)


def _claim_slot(root: Path, slot: str) -> bool:
    """Take this root's slot if not held; a failed claim sets the slow retry.

    RETRYABLE, not "cold for the lifetime" (the old S5 dead-end): the slot is the
    kernel's ``flock`` and its holder is another console, which can die at any
    time. A console that cannot take it re-probes at :data:`SLOT_RETRY_S`.
    """
    if _take_slot(root, slot):
        return True
    logger.info(
        "the %s standby slot for %s is held by another console; retrying in %.0fs",
        slot,
        root,
        SLOT_RETRY_S,
    )
    _push_deadline(SLOT_RETRY_S)
    return False


def _spawn_one(root: Path, slot: str) -> None:
    """One spawn attempt: append a spare to the pool, or schedule the retry."""
    from local_operator.session.runtime.launch import _spawn_interpreter

    try:
        interpreter = _spawn_interpreter()
        warm = _spawn_standby(root, interpreter, slot, idle_s=_idle_for(slot))
    except BaseException:  # noqa: BLE001 — a failed spawn is a retry, never a crash
        # The claim goes back ONLY when nothing else is held (m3-2, refined by
        # agent review round 1, M2). The m3-2 rationale — "a slot held by a
        # console with no spare is a win nobody gets" — is exactly the EMPTY
        # pool; at depth 2 a holder with a live spare keeps its slot, or another
        # console wins the cap and warms beside a holder that already has one.
        with _LOCK:
            empty = not _POOL
        if empty:
            _release_slot(root, slot)
        _note_attempt_failure("spawn-failed")
        return
    with _LOCK:
        _POOL.append(warm)
        count = len(_POOL)
    logger.info("warmed a standby for %s (%d/%d)", root, count, _depth_for(slot))


def _note_attempt_failure(reason: str) -> float:
    """Record a failed warm attempt, schedule the next one, return the delay.

    Doubling from :data:`RETRY_BASE_S` to :data:`RETRY_CEIL_S` per consecutive
    failure; the schedule RESETS when a spare reaches READY or a refill lives a
    normal life, so the backoff is for a root that cannot warm, not for the
    normal consume-and-replace cycle.
    """
    with _LOCK:
        _ATTEMPTS[0] = min(_ATTEMPTS[0] + 1, 16)
        delay = min(RETRY_BASE_S * (2 ** (_ATTEMPTS[0] - 1)), RETRY_CEIL_S)
        _NEXT_AT[0] = _now() + delay
    return delay


def _push_deadline(delay: float) -> None:
    """Set the next-attempt deadline at least ``delay`` from now.

    A parking cadence for states that are not retryable failures (root gone,
    warming switched off, a slot another console holds): it pushes the deadline
    OUT rather than resetting it, so repeated ticks cannot pull a re-probe in.
    """
    with _LOCK:
        _NEXT_AT[0] = max(_NEXT_AT[0], _now() + delay)


def note_spare_gone(spare: "_Standby", reason: str, *, terminate: bool = False) -> None:
    """THE ONE CHOKE POINT for every clear of a tracked spare.

    This is where invariant P (module docstring) is held: every call ends with a
    refill scheduled or already due, so no transition — an adoption, a death, a
    failed warm, a retirement, a torn handshake — can leave the console with
    nothing ready, nothing warming and nothing on the clock. The first version of
    this module had no such owner; five separate guards could each drop a spare
    without replacing it (S1-S6/S8).

    The reason's CLASS picks the schedule. A spare that was consumed, or lived
    out :data:`REWARM_MIN_LIFE_S`, is replaced AT ONCE (the normal
    consume-and-replace cycle, and it resets the backoff). Everything else — a
    young death, a failed warm, a wedged warm, a torn handshake, a thrice-declined
    spare — goes through the doubling backoff, so a root that cannot keep a spare
    cannot fork-storm either.

    ``terminate`` ends the child by exact pid for the clears where it may still be
    alive and unusable (a wedged warm, a failed handshake, a declined spare). It
    is NEVER set for an adoption: that process is a session's runtime now, and
    signalling it is the very defect this choke point exists to prevent.
    """
    if terminate and spare.adopted:
        # HARD RULE (B1): a spare that has been adopted IS a session's runtime —
        # its pid is not a spare's any more, whatever reason a caller passes.
        # The flag is checked HERE, at the one function that can reach
        # ``_retire``, and again inside ``_retire`` itself, because this is the
        # promise the module exists to keep.
        logger.debug("not retiring an adopted spare (%s)", reason)
        terminate = False
    if terminate:
        _retire(spare)
    spare.close()
    with _LOCK:
        # Rebuilt by IDENTITY rather than ``list.remove`` (agent review round 1,
        # B1 round: the session-deletion guard read ``_POOL.remove`` as a
        # filesystem removal — the receiver is a module global, not a literal
        # container, which is the only shape that guard can tell apart). A
        # comprehension over the list is the same operation, spelled where no
        # reader has to guess what the receiver is.
        _POOL[:] = [entry for entry in _POOL if entry is not spare]
    life = _now() - spare.spawned_at
    retry_class = (
        reason
        in (
            "warm-failed",
            "warm-wedged",
            "bad-handshake",
            "adoption-failed",
            "declined-thrice",
        )
        or (reason == "exited" and life < REWARM_MIN_LIFE_S)
        or (reason.startswith("retired") and life < REWARM_MIN_LIFE_S)
    )
    if reason == "adopted":
        # A spare was consumed: success, so the backoff resets and the
        # replacement is DUE NOW (the supervisor's next pass forks it).
        with _LOCK:
            _ATTEMPTS[0] = 0
            _NEXT_AT[0] = _now()
        logger.info("standby adopted; a replacement is scheduled")
        return
    if not retry_class:
        with _LOCK:
            _ATTEMPTS[0] = 0
            _NEXT_AT[0] = _now()
        logger.info("standby gone (%s) after %.1fs; replacing now", reason, life)
        return
    delay = _note_attempt_failure(reason)
    logger.info("standby gone (%s) after %.1fs; retrying in %.1fs", reason, life, delay)


def ensure_warm(root: Path, interpreter: str, slot: str | None = None) -> None:
    """One synchronous fill attempt: spawn a spare now if the pool is short.

    The supervisor's per-attempt primitive, and the function the tests drive
    directly. It is NOT on the engage path any more: ``launch.engage_runtime``
    calls :func:`notify_engage` and the single supervisor thread forks, so a
    refill has exactly one owner and it is the one that holds invariant P
    (module docstring). Never raises.
    """
    if not _WARMING[0] or disabled() or os.environ.get("LOP_MOBILE_CHILD_RESUME"):
        return
    try:
        from local_operator.paths import config_dir

        if Path(root) != config_dir():
            return
        slot = _ROLE[0] if slot is None else slot
        with _LOCK:
            if len(_POOL) >= _depth_for(slot):
                return
        # THE CAP, and it is a hard one: one warming console per slot per root
        # for the whole machine, however many consoles are running; each holder
        # keeps at most its role's depth of private spares. A console that cannot
        # take its slot goes cold until the supervisor's next retry (2.15 s p50
        # at load 153-166, measured), which is the price this module trades for a
        # bounded memory ceiling.
        if not _take_slot(Path(root), slot):
            return
        try:
            warm = _spawn_standby(Path(root), interpreter, slot, idle_s=_idle_for(slot))
        except BaseException:
            # Same refinement as ``_spawn_one`` (M2): release only when this was
            # the console's only spare — then a slot that produced nothing is a
            # win nobody gets; with a live spare in hand the slot is doing its
            # job and stays.
            with _LOCK:
                empty = not _POOL
            if empty:
                _release_slot(Path(root), slot)
            raise
        if warm is not None:
            with _LOCK:
                _POOL.append(warm)
    except Exception:  # noqa: BLE001 — a missing warm is a slower next engage, never a failure
        logger.debug("could not warm a standby for %s", root, exc_info=True)


def _take_slot(root: Path, slot: str) -> bool:
    """Claim this root's ``slot`` for this process, or report that it is taken.

    ONE ADOPTABLE SPARE IS NOT POSSIBLE, WHICH IS WHY THIS IS A CAP INSTEAD
    (agent review round 3, after `d2 <https://github.com/damianvtran/local-operator/pull/1538>`_
    measured three consoles on one root warming three spares at ~136 MB each).
    A spare is a private descriptor to a child of exactly one console — that is
    the R1-1 fix, and the capability reaches no process that console did not fork.
    Making one spare adoptable by the OTHER consoles on the root therefore needs a
    rendezvous between two same-uid processes, and a path any same-uid process can
    bind is a path an impostor can bind first: the impostor then receives a real
    console's operator capability, which is the escalation R1-1 proved. The narrow
    reason is worth stating exactly, because the broad version of it is FALSE and
    this codebase contradicts it: same-uid peers CAN be authenticated —
    ``secrets/peer.py`` does it with the kernel's own attestation — but that names
    the peer to a SERVER. It gives a CLIENT nothing to compare a candidate server
    against, and the console is the client here, so it cannot tell a real warmer
    from an impostor. There is no code-identity check available either (both are
    the same interpreter), and privilege separation would be a redesign. So the
    count is capped instead.

    ``flock`` and not a pid file: the kernel releases the lock when the holding
    process dies, so a crash or a ``kill -9`` cannot leave a root permanently
    without a spare, and there is no stale pid to misread as a live owner.
    ``O_CLOEXEC`` so the spare's own child cannot hold the slot its parent's death
    should free. Never raises: a console that cannot take a slot simply spawns
    cold, which is the behaviour this whole module already degrades to.

    IT FAILS CLOSED (round 3, M3-1). The first version returned ``True`` on any
    ``OSError``, and the reviewer reproduced what that means: with
    ``chmod 0o500 root/run`` two consoles on one root both warmed, neither held a
    lock, and the cap silently stopped applying on exactly the roots that are
    under pressure (a read-only home, ``ENOSPC`` — this host hit 100% twice in a
    day — ``EMFILE``). A cap that disappears when the disk is full is not a cap.
    """
    key = (str(root), slot)
    with _SLOT_LOCK:
        if key in _SLOTS:
            return True
        try:
            import fcntl

            directory = root / "run"
            directory.mkdir(parents=True, exist_ok=True)
            descriptor = os.open(
                str(directory / f"standby-{slot}.lock"),
                os.O_CREAT | os.O_RDWR | getattr(os, "O_CLOEXEC", 0),
                0o600,
            )
        except (ImportError, OSError):
            # Cold, not uncapped. "Cannot express the cap here" and "do not warm
            # here" are the same answer; the ImportError arm is unreachable anyway
            # (no fcntl implies a non-POSIX platform, where ``disabled`` has
            # already stopped warming), so both are answered the same way.
            logger.warning(
                "no standby slot available for %s on %s; this console will spawn cold",
                slot,
                root,
                exc_info=True,
            )
            return False
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(descriptor)
            logger.info(
                "this root's %s standby slot is held by another console; "
                "engages here will spawn cold (%s)",
                slot,
                root,
            )
            return False
        _SLOTS[key] = descriptor
        return True


def _release_slot(root: Path, slot: str) -> None:
    """Give up this root's slot — a claim that produced no spare is a wasted slot.

    Called when the spawn itself fails (round 3, m3-2) AND the pool is empty —
    the empty pool is where the rationale holds: a console that claims the root's
    slot and then cannot warm would hold it with nothing to offer, for the life
    of the process, since ``_SLOTS`` is per process and nothing else ever releases
    it. With a live spare in hand (depth 2) the slot stays (agent review round 1,
    M2): what the count promises is "at most one HOLDER per slot, each holding at
    most its role's depth of private spares".
    """
    with _SLOT_LOCK:
        descriptor = _SLOTS.pop((str(root), slot), None)
    if descriptor is not None:
        try:
            os.close(descriptor)
        except OSError:
            pass


def _release_slots() -> None:
    """Give up every slot this process holds. Tests only; exit releases them anyway."""
    with _SLOT_LOCK:
        descriptors = list(_SLOTS.values())
        _SLOTS.clear()
    for descriptor in descriptors:
        try:
            os.close(descriptor)
        except OSError:
            pass


def _spawn_standby(
    root: Path,
    interpreter: str,
    slot: str = SLOT_TUI,
    *,
    idle_s: "float | None" = None,
) -> "_Standby":
    """Fork the warming interpreter, handing it ONE end of a private socketpair.

    ``pass_fds`` is what makes the other end unreachable by anything else: it is
    the only descriptor that survives the child's ``exec`` (``close_fds`` closes
    every other one), and the console keeps its own end with ``O_CLOEXEC`` set so
    no later child of this process — a tool subprocess, an ``exec --background``
    worker — inherits it either.

    ``idle_s`` is the child's idle window (0 = keep warm): the slot's policy
    unless the caller overrides it. It travels in ``STANDBY_IDLE_ENV``, an
    additive variable an older child ignores (keeping its 900 s) and the
    requester's adopted environment drops (``_apply_environment`` replaces the
    whole environment at adoption).
    """
    from local_operator import procname
    from local_operator.interpreter import SAFE_PATH_FLAG
    from local_operator.paths import CONFIG_DIR_ENV
    from local_operator.procstate import detached_popen_kwargs

    console_end, child_end = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    child_fd = child_end.fileno()
    env = dict(os.environ)
    for key in CONTRACT_KEYS + ("LOP_RUNTIME_ADOPT_SESSION",):
        env.pop(key, None)
    # Pinned, because the child resolves its root from the environment and a
    # warmer's own environment is the only one it should serve. Safe to pin: the
    # config dir is read at call time, not by the warm, and adoption replaces
    # this whole environment with the requester's anyway (``_apply_environment``).
    env[CONFIG_DIR_ENV] = str(root)
    env[STANDBY_IDLE_ENV] = str(int(_idle_for(slot) if idle_s is None else idle_s))
    # The label is the ANON session label with ``[standby]`` in place of
    # ``[session]`` and a placeholder id: same lengths as the real label, so the
    # adoption rename is in place.
    label = procname.LABEL_SESSION_ANON.replace("[session]", "[standby]")
    if interpreter != sys.executable:
        argv0, executable = procname.spawn_identity_for_interpreter(
            label, interpreter, id="--------"
        )
    else:
        argv0, executable = procname.spawn_identity(label, id="--------")
    try:
        proc = subprocess.Popen(  # noqa: S603 — fixed argv, no shell
            [argv0, SAFE_PATH_FLAG, "-m", STANDBY_MODULE, STANDBY_FD_FLAG, str(child_fd)],
            executable=executable,
            env=env,
            # Detached stdio: the standby has no session to speak for yet, and its
            # own console is a full-screen app whose terminal a stray write would
            # paint over. On adoption ``_redirect_output`` points both at the
            # session's capture, which is where a runtime's own traceback must
            # land (see R1-2 in the module docstring of ``_become_runtime``).
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            pass_fds=(child_fd,),
            **detached_popen_kwargs(),
        )
    except BaseException:
        # A failed spawn must not leak either end of the socketpair.
        child_end.close()
        console_end.close()
        raise
    child_end.close()
    return _Standby(proc, console_end, root, slot)


#: What a standby may say before it is asked anything, as ONE byte (the console
#: reads it with a zero timeout, so the frame must arrive whole and mean something
#: on its own). ``READY`` is the normal case; ``FAILED`` says the warm raised, and
#: the console logs it and spawns cold.
_READY = b"R"
_FAILED = b"E"


class AdoptedRuntime:
    """The ``Popen``-shaped handle the engage loop expects, backed by a real child.

    It is a REAL ``Popen`` this process forked (through the standby), so ``pid``
    is the console's own child, and the console's own record of "I started the
    runtime behind this record" is telling the truth about it — and ``poll()``
    reports the child's actual status and reaps it, instead of the previous
    design's zombie probe guessing at a pid that was never ours.
    """

    #: ``launch`` and the engage loop read these to treat an adopted candidate
    #: exactly like a forked one.
    lop_adopted_standby = True

    def __init__(self, proc: "subprocess.Popen[bytes]", capture: Path | None) -> None:
        self._proc = proc
        self.lop_capture_path = capture

    @property
    def pid(self) -> int:
        return self._proc.pid

    def poll(self) -> "int | None":
        return self._proc.poll()

    @property
    def returncode(self) -> "int | None":
        return self._proc.returncode


def adoption_possible() -> bool:
    """Whether a warmed standby for this process is ready to take a spawn.

    A PURE probe for the caller that must decide BEFORE it builds anything: the
    engage path mints a capability handoff for whichever route it takes, and a
    console with no standby (the common case — ``lop exec``, a test, the phone
    daemon) should not mint one it will not use. Never blocks past a zero-timeout
    read: :func:`_ready` reads a byte that was already written when the warm
    finished.

    It can clear a standby whose warm failed — and schedules the refill at the
    same moment (invariant P) — which is the only side effect and is the same one
    :func:`try_adopt` would have caused a moment later.
    """
    if disabled():
        return False
    with _LOCK:
        snapshot = list(_POOL)
    for warm in snapshot:
        if _ready(warm):
            return True
    return False


def try_adopt(
    root: Path,
    interpreter: str,
    env: dict[str, str],
    capture: Path,
    cap_fd: "int | None",
) -> "AdoptedRuntime | None":
    """Hand a cold child's whole spawn to a ready standby, or ``None``.

    ``env`` is EXACTLY the environment ``launch._spawn_runtime`` would have given
    the cold child, and ``cap_fd`` is the child end of the operator-capability
    handoff, passed by ``SCM_RIGHTS`` so the value travels on a descriptor exactly
    as it does into a cold child (``harness/approval.py``): the adopted runtime
    reads it with the same ``--operator-fd`` reader, and the capability never
    touches argv, the environment or a file.

    WHO MAY ADOPT, and why there is no check here to look for: only this process
    holds the descriptor this request is written to (see the module docstring).
    The value reaches a process only through this process's own fork, so there is
    no peer to authenticate — and nothing to steal the capability with.

    ON SUCCESS THE SPARE LEAVES THE POOL. The console clears its tracking entry
    and schedules the replacement through :func:`note_spare_gone`, so nothing
    that runs later can find — or signal — a process that is now a session's
    runtime. The first revision kept the entry ("consumed") until the next engage
    read EOF off its closed channel and RETIRED it: opening a second chat
    SIGTERMed the first chat's live runtime, and the replacement the console
    needed was the cold spawn of that second chat (invariant P's S8).

    THE FLAG IS FOR THE CONCURRENT CASE (agent review round 1, B1). Two attempts
    can both pass the membership check before either finishes; the loser then
    holds a channel whose child is a runtime. ``warm.adopted`` is set inside this
    lock and read inside it, so the loser skips that spare — and tries the next
    candidate — instead of reporting a torn handshake and retiring a live pid,
    which is what ``note_spare_gone(..., terminate=True)`` used to do there.

    Never raises: a standby that is not ready, one that declined, a torn reply,
    or a wedged one that misses :data:`ADOPT_TIMEOUT_S` all answer ``None``, and
    the caller does exactly what it did before this module existed.
    """
    if disabled():
        return None
    with _LOCK:
        candidates = list(_POOL)
    for warm in candidates:
        if Path(root) != warm.root:
            continue
        with _LOCK:
            if warm not in _POOL:
                continue
        if not warm.alive():
            note_spare_gone(warm, "exited")
            continue
        adopted: "AdoptedRuntime | None" = None
        #: Set when the standby answered a well-formed refusal that leaves it
        #: ALIVE and waiting. A DECLINE IS NOT A FAILURE (QA round 2, Q2-2): the
        #: standby declines requests whose venv, root or warm-sensitive
        #: environment differ from what it warmed under, precisely so that it can
        #: serve the right one later — and the console that declined is usually
        #: the one that warmed it (a TUI and the desktop daemon on one root
        #: legitimately differ). Killing it on the way out threw away the spare
        #: its own host had paid for, measured twice: a foreign venv and a
        #: differing ``PYTHONHASHSEED`` each retired the console's own standby.
        declined = False
        retired = ""
        failed = False
        reply: Any = None
        # THE WHOLE HANDSHAKE IS ONE CRITICAL SECTION (``_CHANNEL_LOCK``): the
        # supervisor peeks the same socket every tick, and a peek landing between
        # the request and its reply would swallow a byte of the frame.
        with _CHANNEL_LOCK:
            if warm.adopted:
                # A CONCURRENT ATTEMPT WON THIS SPARE between our membership
                # check above and this lock (agent review round 1, B1): the pid
                # is a session's runtime now. Never handshake with it and never
                # signal it — try the NEXT candidate instead. At depth 2 that is
                # the other ready spare, so the loser of a race still adopts
                # rather than falling back to a cold spawn.
                continue
            if not _ready_locked(warm):
                continue
            try:
                cwd = os.getcwd()
            except OSError:
                cwd = ""
            request = {
                "op": "adopt",
                "root": str(root),
                "interpreter": str(interpreter),
                "env": dict(env),
                "cwd": cwd,
                "capture": str(capture),
                "has_cap_fd": cap_fd is not None,
            }
            try:
                # The request sets its own deadline (ADOPT_TIMEOUT_S): the reply
                # is a real round trip to a process that may be mid-warm.
                warm.sock.settimeout(ADOPT_TIMEOUT_S)
                # ONE MESSAGE: the frame and the capability descriptor together,
                # so the child's first read gets the length prefix and the fd in
                # the same ``recvmsg`` (see :func:`_recv_request`). A separate
                # marker byte would be indistinguishable from the start of that
                # prefix.
                body = json.dumps(request).encode("utf-8")
                payload = len(body).to_bytes(4, "big") + body
                if cap_fd is not None:
                    socket.send_fds(warm.sock, [payload], [cap_fd])
                else:
                    socket.send_fds(warm.sock, [payload], [])
                reply = _recv(warm.sock)
                if isinstance(reply, dict) and reply.get("ok"):
                    # THE FLAG IS SET INSIDE THE LOCK, at the instant the winner
                    # is known (B1): every later reader — this thread's own
                    # clear, the supervisor, a concurrent attempt that passed
                    # membership before we did — must see one truth about this
                    # pid, and it is "no longer a spare".
                    warm.adopted = True
                    adopted = AdoptedRuntime(warm.proc, capture)
                elif isinstance(reply, dict) and reply.get("reason"):
                    # LOGGED, not swallowed (QA round 2, Q2-5): the reason is why
                    # the standby's own docstrings promise a retirement reads as
                    # ``config-moved`` rather than as an unexplained EOF, and
                    # until this line existed the console logged nothing at all.
                    if reply.get("retire"):
                        retired = str(reply["reason"])
                        logger.info(
                            "runtime standby retired (%s); a replacement will be warmed",
                            retired,
                        )
                    else:
                        declined = True
                        logger.info(
                            "runtime standby declined this spawn (%s); keeping it warm",
                            reply["reason"],
                        )
                else:
                    failed = True
                    logger.debug("runtime standby answered %r; spawning cold", reply)
            except (OSError, ValueError):
                failed = True
                logger.debug("standby adoption failed; spawning cold", exc_info=True)
        # OUTSIDE the channel lock: every clear logs, schedules the refill, and
        # the terminate path can block for its SIGTERM.
        if adopted is not None:
            note_spare_gone(warm, "adopted")
            return adopted
        if declined:
            _note_decline(warm, str(reply.get("reason")) if isinstance(reply, dict) else "")
            return None
        if retired:
            note_spare_gone(warm, f"retired ({retired})")
            return None
        if failed:
            with _LOCK:
                left_the_pool = warm not in _POOL
            if warm.adopted or left_the_pool:
                # THE LOSER OF A RACE, not a broken spare (B1): somebody else's
                # adoption took this pid out of the pool while we held it, and
                # our channel died because the child is a runtime now, not
                # because the handshake failed. Nothing here is ours to clear or
                # to signal — try the next candidate, and fall through to the
                # caller's cold spawn only when there is none.
                continue
            # A torn channel, a wedged child, an answer that was not a frame:
            # this spare cannot serve, so it goes — by exact pid — and the refill
            # is scheduled (the old code retired it and warmed NOTHING until the
            # next engage).
            note_spare_gone(warm, "adoption-failed", terminate=True)
            return None
    return None


def _note_decline(warm: "_Standby", reason: str) -> None:
    """Count a decline from THIS spare; retire it at :data:`DECLINE_RETIRE_N`.

    One decline is normally about the REQUESTER (a foreign venv, a differing
    environment) and the spare must stay: the next engage may well be the right
    one. But a console's own spare that keeps declining its own console can never
    serve it — the warmer and the only requester are the same process — so past
    the threshold it is retired and refilled against the CURRENT state instead
    (S4: without this, a declined spare sat out its whole idle window while the
    console held a slot it could not use).
    """
    warm.declines += 1
    warm.decline_reason = reason
    if warm.declines < DECLINE_RETIRE_N:
        return
    logger.info(
        "standby declined %d times in a row (%s); retiring it and refilling",
        warm.declines,
        reason,
    )
    note_spare_gone(warm, "declined-thrice", terminate=True)


def _ready(warm: _Standby) -> bool:
    """Whether the child has said it is warm. Cached, and never blocks past now.

    A zero timeout, because this runs on the engage path: a standby whose warm is
    still in flight must cost the engage nothing at all. The byte was written
    when the warm finished, so a warm standby's answer is already in the socket
    buffer and this read returns immediately. Takes ``_CHANNEL_LOCK`` so the peek
    can never swallow a byte of a concurrent adoption's reply (see :func:`try_adopt`).
    """
    if warm.ready:
        return True
    with _CHANNEL_LOCK:
        return _ready_locked(warm)


def _ready_locked(warm: "_Standby") -> bool:
    """``_ready`` with ``_CHANNEL_LOCK`` already held (the adoption path's shape).

    A clear from here schedules the refill like every other (invariant P). It does
    NOT terminate: closing our end is the child's exit signal, and the terminate
    path waits for a SIGTERM, which must not happen under the channel lock.
    """
    if warm.ready:
        return True
    try:
        warm.sock.settimeout(0)
        answer = warm.sock.recv(1)
    except (BlockingIOError, TimeoutError, OSError):
        return False
    if answer == _READY:
        warm.ready = True
        with _LOCK:
            _ATTEMPTS[0] = 0
        return True
    if answer == _FAILED:
        logger.warning("the runtime standby could not warm; a replacement will be warmed")
        note_spare_gone(warm, "warm-failed")
    elif answer:
        logger.debug("unexpected standby handshake %r", answer)
        note_spare_gone(warm, "bad-handshake")
    else:
        # EOF before READY: the child died, or its channel closed. Same class as
        # an observed exit; the death may simply not be visible in poll() yet.
        note_spare_gone(warm, "exited")
    return False


def _retire(warm: _Standby) -> None:
    """End a standby that cannot serve, by EXACT pid of a child this process owns.

    Never by name: this fleet runs ~25 agents whose own children carry similar
    argv, and an unscoped kill has already taken out another session's process
    tree once. SIGTERM first because that is what a standby expects; SIGKILL only
    if it ignores one, and only for the pid this process forked.

    A flagged spare is refused (B1): whatever path got here — a direct call from
    a clear, a test teardown, a future caller — this pid is a session's runtime
    now, and signalling it is the exact harm this module exists to remove.
    """
    if getattr(warm, "adopted", False):
        return
    try:
        if warm.proc.poll() is None:
            warm.proc.terminate()
            try:
                warm.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                warm.proc.kill()
    except Exception:  # noqa: BLE001 - a retire is best-effort BY CONTRACT
        # Not just OSError, and not fatal: this runs on the engage path (a
        # refused or timed-out adoption) and on the console's way out. A standby
        # that cannot be signalled is still harmless - it holds one end of a
        # socketpair nothing else has, and it exits on its own when this console
        # closes the other end, when its root goes away, or on idle.
        logger.debug("could not end standby %s", getattr(warm, "pid", "?"), exc_info=True)


def reset_for_tests() -> None:
    """End every tracked spare and clear the flags. Tests only — production never calls it."""
    with _LOCK:
        spares = list(_POOL)
        _POOL.clear()
        _WARMING[0] = False
        _ATTEMPTS[0] = 0
        _NEXT_AT[0] = 0.0
    for warm in spares:
        # No scheduling here: this call is the deliberate stop, not a clear that
        # must be replaced (``_WARMING`` goes off first, and the supervisor parks
        # on it).
        warm.close()
        _retire(warm)
    # The slots go with it, or a suite that disables warming and re-enables it
    # would keep a lock it never released (and, in-process, would then be its own
    # "another console").
    _release_slots()


# ---------------------------------------------------------------------------
# Wire: one length-prefixed JSON object each way, over the private descriptor
# ---------------------------------------------------------------------------


def _send(sock: socket.socket, payload: dict[str, Any]) -> None:
    body = json.dumps(payload).encode("utf-8")
    sock.sendall(len(body).to_bytes(4, "big") + body)


def _recv(sock: socket.socket) -> Any:
    header = _read_exact(sock, 4)
    size = int.from_bytes(header, "big")
    if size > 1 << 20:
        raise ValueError("standby frame too large")
    return json.loads(_read_exact(sock, size).decode("utf-8"))


def _read_exact(sock: socket.socket, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = sock.recv(size - len(chunks))
        if not chunk:
            raise ValueError("standby connection closed mid-frame")
        chunks.extend(chunk)
    return bytes(chunks)


# ---------------------------------------------------------------------------
# The standby side
# ---------------------------------------------------------------------------


def _loaded_tree_stamp() -> dict[str, int]:
    """``{file: mtime_ns}`` for every loaded ``local_operator`` module file."""
    stamp: dict[str, int] = {}
    for name, module in list(sys.modules.items()):
        if not name.startswith("local_operator"):
            continue
        path = getattr(module, "__file__", None)
        if not path:
            continue
        try:
            stamp[path] = os.stat(path).st_mtime_ns
        except OSError:
            stamp[path] = -1
    return stamp


def _tree_moved(recorded: dict[str, int]) -> bool:
    """Whether any file the warm IMPORTED has changed since it was imported.

    The recorded files only, re-stat'ed: a module first imported AFTER the warm
    (the guard's own lazy imports, a timer's) is not a change to what was warmed,
    and comparing whole stamps read that as a moved tree — measured: the first
    idle check retired every standby with ``tree-moved`` on an untouched checkout.
    """
    for path, mtime in recorded.items():
        try:
            now = os.stat(path).st_mtime_ns
        except OSError:
            now = -1
        if now != mtime:
            return True
    return False


class _Warmth:
    """What the standby recorded when it finished warming: the adoption guards.

    Two different questions, answered separately because they want different
    reactions:

    * :meth:`stale` — has anything this interpreter's imports depend on moved
      since the warm? If so the standby is worthless to EVERY requester, so it
      EXITS (the console warms a replacement against what is current).
    * the per-request checks in :func:`_await_request` — is this requester one a
      cold child of THIS interpreter would serve identically? If not, the standby
      DECLINES and keeps waiting, because exiting on a mismatch would end a spare
      whose console merely engaged from another directory.
    """

    def __init__(self, root: Path) -> None:
        from local_operator import update
        from local_operator.config import CONFIG_FILE_NAME, config_file_key

        self.root = root
        self.config_file = root / CONFIG_FILE_NAME
        self.config_key = config_file_key(self.config_file)
        self.build = update.installed_build(os.environ.get("LOP_BUILD_PREFIX") or None)
        self.tree = _loaded_tree_stamp()
        self.venv = os.path.realpath(sys.prefix)

    def stale(self) -> str:
        """Why no requester may have this standby any more, or ``""``."""
        from local_operator import update
        from local_operator.config import config_file_key

        if not self.root.is_dir():
            # A deleted root (a finished test rig, a removed sandbox): nothing
            # will ever adopt here again, and the process must not outlive it.
            return "root-gone"
        # THE GENERATION. ``lop-update`` installs a new generation beside this
        # one and flips ``current``; this interpreter's own files do not change,
        # so neither the tree stamp nor ``installed_build`` of its own prefix
        # would notice. What a cold spawn would run now is the question.
        current = update.current_interpreter()
        if current is not None and _venv_of(str(current)) != self.venv:
            return "generation-moved"
        if update.installed_build(os.environ.get("LOP_BUILD_PREFIX") or None) != self.build:
            return "build-moved"
        if _tree_moved(self.tree):
            return "tree-moved"
        if config_file_key(self.config_file) != self.config_key:
            return "config-moved"
        return ""


def _rename_argv(old: bytes, new: bytes) -> bool:
    """Rewrite one argv word IN PLACE, in the memory ``ps`` reads it from.

    WHY. The residency census identifies a runtime by the ``-m RUNTIME_MODULE``
    word in its argv (``reclaim.parse_process_row``). An adopted standby IS a
    runtime and must be counted — the sweep's protection of attached runtimes and
    its orphan reclaim both depend on the census — but it was exec'd as
    ``-m ...standby``. macOS's ``ps`` reads the argument area of the process
    (``KERN_PROCARGS2``) and Linux reads ``/proc/<pid>/cmdline``, both backed by
    the same memory ``_NSGetArgv``/``argv`` point at, so an equal-length
    overwrite is visible to every reader (verified with ``ps -ww`` on this host).
    ``sys.orig_argv`` keeps the original, so nothing in-process changes.

    Returns False when the platform offers no way to find argv; the caller then
    refuses adoption rather than leaving an uncountable runtime.
    """
    if len(old) != len(new):
        return False
    try:
        if sys.platform == "darwin":
            libc = ctypes.CDLL(None)
            libc._NSGetArgc.restype = ctypes.POINTER(ctypes.c_int)
            libc._NSGetArgv.restype = ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))
            argc = libc._NSGetArgc().contents.value
            argv = libc._NSGetArgv().contents
            addresses = [argv[i] for i in range(argc)]
        elif sys.platform.startswith("linux"):
            return _rename_argv_linux(old, new)
        else:
            return False
        renamed = False
        for address in addresses:
            if not address:
                continue
            word = ctypes.string_at(address)
            at = word.find(old)
            if at >= 0:
                ctypes.memmove(address + at, new, len(new))
                renamed = True
        return renamed
    except Exception:  # noqa: BLE001 — no argv access means "cannot rename"
        logger.debug("argv rename unavailable", exc_info=True)
        return False


def _rename_argv_linux(old: bytes, new: bytes) -> bool:
    """Linux: argv is the contiguous region ``/proc/self/stat`` fields 48-49 name."""
    with open("/proc/self/stat", "rb") as handle:
        fields = handle.read().rsplit(b")", 1)[1].split()
    start, end = int(fields[45]), int(fields[46])
    with open("/proc/self/mem", "r+b", buffering=0) as mem:
        mem.seek(start)
        area = mem.read(end - start)
        at = area.find(old)
        if at < 0:
            return False
        mem.seek(start + at)
        mem.write(new)
    return True


def _standby_fd_from_argv(argv: "list[str]") -> "int | None":
    """The inherited descriptor number, or ``None`` when this is not a standby.

    A NUMBER, not a value: the descriptor is the secret, and a number that names
    nothing in another process's table is useless there.
    """
    for index, item in enumerate(argv):
        if item == STANDBY_FD_FLAG and index + 1 < len(argv):
            candidate = argv[index + 1]
            break
        if item.startswith(f"{STANDBY_FD_FLAG}="):
            candidate = item.partition("=")[2]
            break
    else:
        return None
    return int(candidate) if candidate.lstrip("-").isdigit() else None


def _recv_request(sock: socket.socket) -> "tuple[dict[str, Any], list[int]]":
    """One request frame AND the descriptors that came with it.

    THE DESCRIPTOR RIDES THE SAME MESSAGE AS THE FRAME'S FIRST BYTES, not a second
    message with a marker byte: a byte sent alongside the ancillary data is
    indistinguishable from the start of the frame's length prefix, so the reader
    saw a torn frame (found by this change's own round-1 tests, which is what the
    tests are for). The kernel attaches an ``SCM_RIGHTS`` set to the first byte of
    the message it was sent with, so reading the 4-byte header with ``recv_fds``
    gets both, and the body follows with a plain read.

    Raises ``ValueError`` on EOF — the console having gone away, which the caller
    reads as this process's exit signal.
    """
    header = b""
    fds: list[int] = []
    while len(header) < 4:
        chunk, received, _flags, _addr = socket.recv_fds(
            sock, 4 - len(header), max(1, 4 - len(fds))
        )
        if not chunk:
            for fd in fds:
                os.close(fd)
            raise ValueError("standby channel closed")
        header += chunk
        fds.extend(received)
    size = int.from_bytes(header, "big")
    if size > 1 << 20:
        raise ValueError("standby frame too large")
    body = b""
    while len(body) < size:
        chunk = sock.recv(size - len(body))
        if not chunk:
            raise ValueError("standby channel closed")
        body += chunk
    return json.loads(body.decode("utf-8")), fds


def _close_all(fds: "list[int]") -> None:
    for fd in fds:
        try:
            os.close(fd)
        except OSError:
            pass


def _idle_reap_seconds() -> float:
    """This standby's idle window, from the console that forked it.

    ``STANDBY_IDLE_ENV`` is set at spawn and consumed only here: 0 or less means
    KEEP WARM (the daemon's slot, where the spare is the whole first-send
    budget), any positive value is a deadline. Absent, empty or unparseable — an
    older console, or a child newer than its console — keeps
    :data:`IDLE_REAP_S`, so the mixed-generation fleet is indistinguishable from
    the old behaviour on every path.
    """
    raw = os.environ.get(STANDBY_IDLE_ENV, "")
    if not raw:
        return IDLE_REAP_S
    try:
        return float(raw)
    except ValueError:
        return IDLE_REAP_S


def _await_request(sock: socket.socket) -> "dict[str, Any] | None":
    """Warm, then wait for an adoption this standby may take.

    Returns the request to serve, or ``None`` when the standby should leave
    quietly (idle, its console gone, or its build/root/config moved). Raises only
    for a genuine programming error: every condition a normal machine produces is
    handled here, because a standby that dies noisily would be a worse citizen
    than one that simply is not there.
    """
    from local_operator.paths import config_dir

    root = config_dir()
    # NO SCHEDULING BAND, deliberately: see the module docstring's cost section.
    # A process in Darwin's background band cannot bound its own warm (the
    # sampling is starved by the thing it samples), and an unbounded warm is the
    # failure the operator sees, so this runs at the host's normal priority and
    # its duration is the host's own scheduling latency.
    try:
        _warm()
    except BaseException:  # noqa: BLE001 — a failed warm means "no standby"
        logger.debug("standby warm failed", exc_info=True)
        _send_byte(sock, _FAILED)
        return None
    # The guard snapshot is a handful of stats and two small file reads, after the
    # imports, at the same priority.
    warmth = _Warmth(root)
    warm_env = _warm_sensitive(os.environ)
    _send_byte(sock, _READY)
    idle_s = _idle_reap_seconds()
    # ``0`` or less is KEEP-WARM (the daemon's slot): no deadline at all, so an
    # idle afternoon between conversations still finds this spare ready. Every
    # other slot keeps the 900 s memory bound.
    deadline = None if idle_s <= 0 else time.monotonic() + idle_s
    while True:
        if deadline is None:
            wait = 30.0
        else:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            wait = min(remaining, 30.0)
        sock.settimeout(wait)
        try:
            request, fds = _recv_request(sock)
        except (TimeoutError, socket.timeout):
            # A stale warm is worth nothing: checked between waits too, so a
            # standby does not sit on a superseded build for the whole idle
            # window after a ``lop-update``.
            if warmth.stale():
                return None
            continue
        except (OSError, ValueError):
            # EOF here is the console having gone away — the descriptor it held
            # was the only one, so its close is this process's exit signal.
            return None
        if not isinstance(request, dict) or request.get("op") != "adopt":
            _close_all(fds)
            continue
        reason, retire = _refusal(warmth, warm_env, request)
        if reason:
            _close_all(fds)
            # Sent even on the way out: the console logs the reason, and an
            # unexplained EOF is what a retirement used to look like.
            _refuse(sock, reason, retire=retire)
            if retire:
                return None
            continue
        if request.get("has_cap_fd") and not fds:
            _refuse(sock, "capability-descriptor-missing")
            continue
        cap_fd = fds[0] if fds else -1
        _close_all(fds[1:])
        # LAST, and after every check that can decline: this is the one step that
        # cannot be undone, and it is taken only for a request this standby is
        # actually going to serve.
        if not _rename_argv(STANDBY_MODULE.encode(), _runtime_module().encode()):
            # No way to make this process countable by the census: leaving it as a
            # runtime would hide it from the sweep, which is worse than no spare.
            _refuse(sock, "argv-rename-unavailable")
            return None
        return {**request, "cap_fd": cap_fd}


def _refusal(
    warmth: _Warmth, warm_env: dict[str, str], request: dict[str, Any]
) -> "tuple[str, bool]":
    """Why this standby must not take this request, or ``("", False)``.

    Returns ``(reason, retire)``. ``retire`` means the standby is worthless to
    EVERYONE (an input it warmed against moved) so it should exit; a reason with
    ``retire`` False is a decline that leaves it waiting, because it is still the
    right spare for the host that warmed it — a TUI and the desktop daemon
    legitimately differ in environment, and a spare that exited on every mismatch
    would be killed by whichever of the two engaged next.

    The reason is SENT to the console either way, so a retirement appears in that
    host's log as ``config-moved`` rather than as an unexplained EOF.
    """
    stale = warmth.stale()
    if stale:
        return stale, True
    if str(request.get("root") or "") != str(warmth.root):
        return "other-root", False
    if _venv_of(str(request.get("interpreter") or "")) != warmth.venv:
        return "other-venv", False
    env = request.get("env")
    env = env if isinstance(env, dict) else {}
    if _warm_sensitive(env) != warm_env:
        # The requester's interpreter settings, locale, home or build prefix
        # differ from the ones this process imported under: only a fresh
        # interpreter is equivalent.
        return "other-environment", False
    # The one step that cannot be undone is taken by the CALLER, as the last
    # precondition before it commits: this function only answers questions.
    return "", False


def _refuse(sock: socket.socket, reason: str, *, retire: bool = False) -> None:
    """Tell the console why this request was refused, and whether we are leaving.

    ``retire`` is what lets the console tell the two apart without guessing
    (QA round 2, Q2-2/Q2-5): a decline means "still warm, ask again", a retirement
    means "this spare is worthless, replace me" — and both now appear in the
    console's log with their reason.
    """
    try:
        _send(sock, {"ok": False, "reason": reason, "retire": retire})
    except OSError:
        pass


def _send_byte(sock: socket.socket, byte: bytes) -> None:
    """One byte, in one write: the handshake the console reads with a zero timeout."""
    try:
        sock.sendall(byte)
    except OSError:
        pass


def _commit_adoption(sock: socket.socket, request: dict[str, Any]) -> int:
    """Become the runtime's process: argv, name, environment, boot record, reply.

    Called OUTSIDE the caller's error handling, deliberately (agent review round
    1, R1-2). Everything up to the previous line is a standby that may fail
    quietly; from here this process is the session's runtime, and a failure must
    be as loud as it is in a cold child — non-zero exit and a traceback on stderr,
    which by then is the session's capture file that ``engage_runtime`` reads for
    its actionable startup reason.

    Returns the descriptor number carrying the capability (``-1`` for none).

    THREE renames, and the inequalities between them are deliberate. The MODULE
    word was renamed by ``_await_request`` as the last precondition and it REFUSES
    on failure, because the census matches it (``parse_process_row``) and a runtime
    no census can see is worse than no spare. The two below are COSMETIC and their
    return is ignored: no reader keys on ``[standby]``/``[session]`` or on
    ``id=``, so a platform that cannot rewrite argv still gets a working runtime
    (agent review round 2, nit).
    """
    env = request.get("env")
    env = env if isinstance(env, dict) else {}
    # Recover the fd number this process was STARTED with, before the channel is
    # closed: the capability is moved onto it below so the row `ps` shows is
    # literally the one a cold child has.
    try:
        standby_fd = sock.fileno()
    except (OSError, ValueError):
        standby_fd = -1
    _rename_argv(b"[standby]", b"[session]")
    _rename_argv(b"id=--------", _label_id(env).encode())
    cap_fd = int(request.get("cap_fd", -1))
    _apply_environment(env)
    _change_directory(str(request.get("cwd") or ""))
    _redirect_output(str(request.get("capture") or ""))
    # BEFORE the session is constructed, so the sweep can see this runtime's
    # session and its true age for the whole construction window (R1-3). The
    # runtime rewrites this same record at its own boot boundary.
    _write_adoption_boot_record(env)
    _send(sock, {"ok": True, "pid": os.getpid()})
    # The private descriptor has served its purpose, and the runtime must not hold
    # one nothing will ever write to again: close it before any tool subprocess
    # could be started.
    try:
        sock.close()
    except OSError:
        pass
    # MAKE THE ROW TRUE, not merely similar (agent review round 2, nit). The argv
    # already carries ``--operator-fd <n>`` (see :data:`STANDBY_FD_FLAG`), and what
    # makes that statement TRUE is moving the capability onto the very descriptor
    # number this process was started with, now that the private channel (which
    # held it) is closed. So an adopted runtime's ``ps`` row and the descriptor it
    # actually reads its capability from agree — byte for byte, with no rename
    # needed on a word whose length could not change anyway.
    if cap_fd >= 0 and standby_fd >= 0 and cap_fd != standby_fd:
        try:
            os.dup2(cap_fd, standby_fd)
            os.close(cap_fd)
            cap_fd = standby_fd
        except OSError:
            # WARNING, not DEBUG: after this the row names a descriptor that holds
            # the closed channel rather than the capability. The runtime still reads
            # the right one (from ``sys.argv``, never from a listing), so this is
            # cosmetic — but it is a wrong statement about the process, and the
            # whole point of the change was to stop making those quietly.
            logger.warning(
                "could not inherit the capability onto descriptor %s; the ps row names it",
                standby_fd,
                exc_info=True,
            )
    return cap_fd


def _write_adoption_boot_record(env: dict[str, Any]) -> None:
    """Publish this process's boot record now, before it constructs anything.

    ``run/host``'s record is the artifact ``reclaim`` and the roster already
    prefer for attribution, so writing it one step earlier is what makes an
    adopted runtime visible to them exactly as a forked one is. Best-effort, for
    the reason ``journal.write_boot_record``'s own contract gives: a runtime whose
    boot record cannot be written must still run its turns.
    """
    session_id = str(env.get("LOP_MOBILE_CHILD_RESUME") or "")
    if not session_id:
        return
    try:
        from local_operator import update
        from local_operator.session.runtime.journal import write_boot_record

        build = update.installed_build(os.environ.get("LOP_BUILD_PREFIX") or None)
        write_boot_record(session_id, build, cwd=str(env.get("LOP_MOBILE_CHILD_CWD") or ""))
    except Exception:  # noqa: BLE001 — instrumentation, never a boot gate
        logger.debug("adopted runtime could not write its boot record", exc_info=True)


def _runtime_module() -> str:
    from local_operator.session.runtime.types import RUNTIME_MODULE

    return RUNTIME_MODULE


def _warm() -> None:
    """Import what a runtime child and a first session construction import.

    Exactly the work a cold child pays before it can publish, minus anything that
    depends on the session: nothing here opens the store, reads a transcript,
    takes a lease or starts a background pass (verified with an audit hook over
    the whole warm: zero opens or listings under the config root). The tokenizer
    rides along for the reason ``warm_session_imports`` gives.

    AT NORMAL PRIORITY, and that is the correction of two rounds' worth of trying
    to be polite first (QA round 1 QW1, round 2 Q2-3). The background band made
    the warm unbounded — 16 minutes of wall on 0.25 s of CPU at load 180-227, and
    still 125.2 s at load 150.3 with a 20 s/45 s window and a CPU-progress test —
    because a starved process cannot reliably measure or bound its own progress.
    A slow warm is strictly better than a starved one: a late standby still serves
    the NEXT engage, while an abandoned one serves nothing. The politeness lives
    in the design instead: warming is opt-in, it starts on a daemon thread after
    an engage, there is one spare per host, and it is reaped on idle.
    """
    import importlib

    # NOT ``local_operator.session.runtime.process`` itself: adoption runs that
    # module as ``__main__`` (see :func:`_become_runtime`), exactly as ``python
    # -m`` does in a cold child, and ``runpy`` warns when the module it is about
    # to run is already imported. Its own top-level imports are warmed by name.
    import local_operator.session.runtime.server  # noqa: F401
    import local_operator.session.runtime.serving  # noqa: F401
    import local_operator.session.runtime.stall_watchdog  # noqa: F401
    from local_operator.session_factory import _WARM_IMPORTS

    # ``_WARM_IMPORTS`` directly rather than ``warm_session_imports()``: that
    # helper also starts the BYTECODE warm, a subprocess that belongs to a
    # long-lived host (``bytecode.warm_bytecode_cache_in_background``'s own
    # docstring says a runtime child must not start it).
    for name in _WARM_IMPORTS + _WARM_EXTRA:
        try:
            importlib.import_module(name)
        except Exception:  # noqa: BLE001 — a warm-up must never be the failure
            logger.debug("standby warm skipped %s", name, exc_info=True)
    try:
        from local_operator.compaction.tokens import warm_tokenizer

        warm_tokenizer()
    except Exception:  # noqa: BLE001
        logger.debug("standby tokenizer warm skipped", exc_info=True)


def _label_id(env: dict[str, Any]) -> str:
    """``id=<first 8 of the session>``, padded to the placeholder's width.

    The same 8 characters ``launch._spawn_runtime`` puts in a cold child's label,
    so ``ps`` and Activity Monitor show an adopted runtime exactly as they show a
    cold one. Padded rather than truncated because the rename is in place and must
    keep the length.
    """
    ident = str(env.get("LOP_MOBILE_CHILD_RESUME") or "")[:8]
    return ("id=" + ident).ljust(len("id=--------"), "-")


def _apply_environment(env: dict[str, Any]) -> None:
    """Replace this process's environment with the one a cold child would get.

    WHOLE, not merged: a cold child inherits exactly ``env`` and nothing else, and
    an adopted standby must be indistinguishable from it — a key the warmer had
    and the requester does not would otherwise survive into the session (the
    ``LOP_*`` leakage AGENTS.md "Isolating a run" documents). Warm-sensitive names
    already matched (see ``_WARM_SENSITIVE_NAMES``), so nothing replaced here was
    read by an import that already ran.

    THE ``ps -E`` VIEW IS NOT REPAIRED HERE, and cannot be: ``os.environ`` and
    ``putenv`` do not rewrite the argument area ``ps -Eww`` reads. That is why the
    boot record is written at adoption (R1-3): reclaim and the roster read the
    record for attribution instead of the process's environment.
    """
    wanted = {str(k): str(v) for k, v in env.items() if isinstance(v, str)}
    for key in list(os.environ):
        if key not in wanted:
            del os.environ[key]
    for key, value in wanted.items():
        if os.environ.get(key) != value:
            os.environ[key] = value


def _change_directory(cwd: str) -> None:
    """The requester's working directory, which a cold child inherits.

    ``launch._spawn_runtime`` passes no ``cwd=``, so the cold child starts where
    the engaging host stands; the session's OWN directory rides the contract
    (``LOP_MOBILE_CHILD_CWD``) and is what the session uses. Best-effort: a
    vanished directory leaves the standby where it was, which only matters to code
    that reads ``os.getcwd()`` before the session applies its own.
    """
    if not cwd:
        return
    try:
        os.chdir(cwd)
    except OSError:
        logger.debug("could not enter the requester's directory", exc_info=True)


def _redirect_output(capture: str) -> None:
    """Point stdout/stderr at the caller's capture file, as a cold spawn does.

    ``engage_runtime`` reads that file to report WHY a candidate died (the
    actionable startup reasons), so an adopted standby must write its construction
    failures to the same place a cold child would — and so must a runtime that
    crashes hours later.
    """
    if not capture:
        return
    try:
        # ``O_CREAT`` as well as append, at the mode the spawn path's ``mkstemp``
        # makes: the capture is this process's log, and a runtime whose traceback
        # went nowhere because the file was missing would be the R1-2 defect in a
        # second shape.
        fd = os.open(capture, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
    except OSError:
        return
    try:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(fd, 1)
        os.dup2(fd, 2)
    finally:
        os.close(fd)


def _become_runtime(operator_fd: int) -> int:
    """Run the runtime module as ``__main__``: from here on, a cold child.

    Called OUTSIDE the standby's error handling, on purpose: this is the point
    after which a failure belongs to the session rather than to a speculative
    spare, so the exception propagates and the process exits NON-ZERO with its
    traceback in the capture file — which is what lets the console report
    "connect a provider" instead of a generic error after three retries (agent
    review round 1, R1-2). The earlier revision caught it here and returned 0 with
    an empty capture, so every adopted failure read as a success.

    ``runpy.run_module(..., run_name="__main__")`` is the machinery ``python -m``
    itself uses, so the runtime's ``__main__`` guard runs unchanged — brand, ARM
    THE STALL BOUND, ``main()`` — and ``stall_watchdog.arm`` keeps its one call
    site (pinned by ``test_the_only_arm_site_is_the_runtime_entry_point``). A
    standby never arms it: the warm is not a session.

    The operator capability arrives exactly as it does for a cold child — as the
    NUMBER of a descriptor in ``sys.argv`` (``--operator-fd <n>``), read and
    closed by the capability reader in ``harness/approval.py``, from
    ``process.main``. The descriptor is the one ``SCM_RIGHTS`` delivered over this
    console's private channel, i.e. the child end of the console's own handoff.

    ``SystemExit`` from the guard's ``sys.exit(main())`` propagates to this
    process's own exit, as it would in a cold child.
    """
    import runpy

    from local_operator.harness.approval import OPERATOR_FD_FLAG

    sys.argv = [sys.argv[0]] + (
        [OPERATOR_FD_FLAG, str(operator_fd)] if operator_fd is not None and operator_fd >= 0 else []
    )
    runpy.run_module(_runtime_module(), run_name="__main__", alter_sys=True)
    return 0


def main() -> int:
    """The standby entry point: wait, then become a runtime or leave quietly."""
    from local_operator import procname

    procname.brand_this_process()
    fd = _standby_fd_from_argv(sys.argv[1:])
    if fd is None:
        logger.debug("standby started without an inherited descriptor; nothing to do")
        return 0
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM, fileno=fd)
    try:
        request = _await_request(sock)
    except Exception:  # noqa: BLE001 — before adoption a failure is just "no standby"
        logger.debug("standby exited on an error while waiting", exc_info=True)
        return 0
    if request is None:
        return 0
    # PAST THIS LINE A FAILURE IS THE SESSION'S, not the spare's: no handler here,
    # so the exception reaches stderr (the capture) and the exit status is non-zero.
    operator_fd = _commit_adoption(sock, request)
    return _become_runtime(operator_fd)


if __name__ == "__main__":
    sys.exit(main())
