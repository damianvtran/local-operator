"""Spawn long-lived runtimes out of the macOS responsibility scope of whoever started us.

WHY THIS MODULE EXISTS — the 2026-10-09 incident. The operator force-quit a hung
Local Operator app from Activity Monitor at 17:11:30; within 17 ms a SIGTERM
sweep hit nine session/exec runtimes that were detached in the POSIX sense
(own session and process group, ppid 1). The sweep came from the macOS
force-quit path, and it reached those runtimes because they still sat in the
APP'S RESPONSIBILITY SCOPE: macOS attributes every process a "responsible"
process whose identity is inherited at spawn and is **not** severed by
``setsid`` (``start_new_session=True``) or by reparenting. Nothing in this
product ever disavowed that inheritance, so killing or crashing the desktop
was able to terminate runtimes it was never meant to own — a violation of the
operator's requirement, whose word for it is "kill the UI, never the sessions".

THE LEVER. ``responsibility_spawnattrs_setdisclaim`` (libSystem, present and
working on macOS 27.0.1 — verified on this host, and the reason this module
feature-detects rather than assumes) is a ``posix_spawn`` attribute: with it
set, the child is responsible for ITSELF instead of inheriting its spawner's
responsibility chain. ``subprocess`` exposes no hook to set spawn attributes,
so the disclaimed path is a direct ``posix_spawn`` call; everything else about
the spawn mirrors what the call sites had (``subprocess.Popen`` semantics:
fresh session, close-fds-except-redirections, stdio redirection, and
``restore_signals`` — SIGPIPE/SIGXFZ/SIGXFSZ reset to default via
``POSIX_SPAWN_SETSIGDEF``; agent review round 1, R1-4).

WHAT WE COULD MEASURE HERE, AND WHAT WE COULD NOT. Measured live on this host:
a child spawned the ordinary way (fork/exec, ``start_new_session=True``)
resolves ``responsible_path`` to the app at the root of its chain in tccd's
attribution rows, while the same child spawned SETSID|CLOEXEC_DEFAULT +
disclaim either resolves responsibility to itself or carries no ``responsible``
block at all — the app link is gone (full evidence on the PR). NOT reproduced
from a scripted context: the force-quit sweep itself. ``LQForceQuit`` (the
libquit primitive Activity Monitor used) kills a LaunchServices app instance
and does NOT sweep even an un-disclaimed detached child in our rig, so the
sweep appears to need context we cannot stage headlessly (the hung-GUI-app
termination flow). This module is the source-level fix for the mechanism the
incident's evidence points at; the verification story is stated in the PR.

FALLBACK SEMANTICS — feature-detected, never silent. Any missing symbol, a
non-darwin host, or a spawn shape this module cannot express (PIPE stdio,
``close_fds=False``) falls back to ``subprocess.Popen`` with
``procstate.detached_popen_kwargs()`` — today's exact behaviour — and logs WHY
once per process. The one case that is logged loudly is a symbol that has
disappeared on darwin: that is an OS change, and a runtime that silently
stopped disclaiming would put the incident back with nothing to notice it.

SPAWN-CHAIN EVIDENCE (the same incident's second half). The runtime's signal
receipt could not name any sender, so a sweep read as "SIGTERM … from an
unidentified sender". Every spawn this module makes records the chain it was
born into (spawner pid + argv0, plus the spawner's own chain where it has one)
into ``LOP_SPAWN_CHAIN`` — bounded per entry (``MAX_ENTRY_CHARS``) and in hops
(``MAX_CHAIN``) — which the runtime reads into its boot record and snapshots
into its signal receipt. Each entry carries the member's liveness AT RECORD
TIME (``alive_at_spawn``); the arrival snapshot adds a second reading
(``alive_now``). The renderer names the app at the root of the chain only when
the two readings conspire — recorded alive, gone at arrival — so the clause
means "this runtime outlived the app", never "the app explains this death";
see ``incidents.render_signal_receipt_detail``.

WHAT THE DISCLAIM COSTS: TCC GRANTS (agent review round 1, R1-2). Severing the
responsibility chain also severs TCC-grant INHERITANCE. A disclaimed runtime —
and everything it spawns — no longer borrows the spawning app's grants (Files &
Folders / Full Disk for protected folders — the class measured here; Screen
Recording and AppleEvents automation were NOT measured, because a probe would
prompt on the operator's screen, and may or may not behave the same), so on a
host where the grant lives on the app identity, an interpreter without its own
grant can lose protected-folder access inside a disclaimed runtime. Measured
on this host as a controlled A/B (same parent and spawn shape, only the
disclaim toggled): an unbranded interpreter's read of ``~/Desktop`` was denied
under the disclaim where the same spawn without it read fine; the product's own
branded interpreter binaries kept access — a per-binary grant property, not a
guarantee. The trade is accepted deliberately: the standing requirement ("kill
the UI, never the sessions") makes a runtime swept with the UI the worse
failure, and the grant is a one-time, per-host decision (the same statement
lives in ``docs/EXEC.md`` and on the PR).
"""

from __future__ import annotations

import ctypes
import errno
import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from typing import Any, Mapping, Sequence, cast

logger = logging.getLogger(__name__)

#: The environment variable carrying the spawn chain into a runtime. Additive:
#: an older child ignores it, and an older build's spawn simply omits it.
ENV_SPAWN_CHAIN = "LOP_SPAWN_CHAIN"

#: How many members a recorded chain keeps. The immediate spawner plus a few
#: ancestors is enough to reach an app-level ancestor (serve → app is one hop);
#: a longer walk would record terminal/loginwindow noise nobody renders.
MAX_CHAIN = 8

#: How much of one member's recorded command line a chain keeps. The chain
#: rides every spawn's environment and every boot record on disk, while a
#: ``ps`` command column is unbounded (measured: 26,921 chars for one ancestor,
#: agent review round 1, R1-3) and can carry secrets among its arguments. The
#: renderer matches a ``.app/Contents/MacOS/`` prefix and renders only the
#: bundle name (design round 1, D3), so a longer prefix buys nothing — this cap
#: bounds what one chain adds to an environment (≤ ``MAX_CHAIN`` × this) and
#: what a command line can leak into the artifacts.
MAX_ENTRY_CHARS = 256

#: Appended when a recorded command is cut to ``MAX_ENTRY_CHARS`` so a reader
#: of the boot record can tell "the command ended here" from "the record
#: ended here" (agent review round 2, R2-N3). The marker is INSIDE the budget:
#: a bounded entry is never longer than ``MAX_ENTRY_CHARS``.
_TRUNCATION_MARK = "…"

#: Largest pid a POSIX ``pid_t`` (a signed 32-bit int on every supported
#: platform) can hold. A larger int in a recorded chain is corruption, and
#: handing it to ``os.kill`` raises ``OverflowError`` rather than ``OSError``.
_PID_MAX = 2**31 - 1


def _bound_command(command: object) -> str:
    """``command`` as a string no longer than ``MAX_ENTRY_CHARS``, marked when cut."""
    text = str(command or "")
    if len(text) <= MAX_ENTRY_CHARS:
        return text
    return text[: MAX_ENTRY_CHARS - len(_TRUNCATION_MARK)] + _TRUNCATION_MARK


#: ``POSIX_SPAWN_SETSID`` / ``POSIX_SPAWN_CLOEXEC_DEFAULT`` / ``POSIX_SPAWN_SETSIGDEF``
#: from XNU's spawn.h — present in the SDK via ``sys/spawn.h`` (``spawn.h``
#: includes it; re-verified on this host), so they are literals here. SETSID
#: mirrors ``start_new_session=True``; CLOEXEC_DEFAULT gives the child exactly
#: ``close_fds=True`` semantics — every descriptor beyond 0/1/2 and the file
#: actions' own is closed, measured on this host: an ``adddup2`` SOURCE is
#: closed in the child and an ``addinherit_np`` fd stays open; SETSIGDEF makes
#: the signal-default set below take effect.
_POSIX_SPAWN_SETSID = 0x0400
_POSIX_SPAWN_CLOEXEC_DEFAULT = 0x4000
_POSIX_SPAWN_SETSIGDEF = 0x0004

#: The signals ``subprocess.Popen(restore_signals=True)`` resets to ``SIG_DFL``
#: in the child (CPython's ``_Py_RestoreSignals``): SIGPIPE always, and
#: SIGXFZ/SIGXFSZ where the platform defines them — macOS defines the last,
#: not SIGXFZ, so this resolves to ``(13, 25)`` here. Building it from
#: ``signal`` keeps the platform rule in one place.
_RESTORED_SIGNALS: tuple[int, ...] = tuple(
    signum
    for name in ("SIGPIPE", "SIGXFZ", "SIGXFSZ")
    if (signum := getattr(signal, name, None)) is not None
)

#: Set once per process: the loud-fallback has already been logged. A warning
#: per spawn would spam a serve's log with one line per runtime.
_fallback_logged = False

_libc_cache: Any = None
_libc_loaded = False

#: ctypes prototypes, set once per process. WITHOUT THESE, ctypes passes every
#: integer as a 32-bit C int and every str/bytes by its own guessing rules; the
#: functions below take pointers, shorts and C bools, and the on-host probe
#: that established the mechanism set its prototypes explicitly — the module
#: must not rely on ABI luck the probe did not. (Missing symbols are skipped;
#: `_fallback_reason` refuses the path that needs them.)
_PROTOTYPES: tuple[tuple[str, Any, list[Any]], ...] = (
    (
        "posix_spawn",
        ctypes.c_int,
        [
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_char_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_char_p),
            ctypes.POINTER(ctypes.c_char_p),
        ],
    ),
    ("posix_spawnattr_init", ctypes.c_int, [ctypes.c_void_p]),
    ("posix_spawnattr_destroy", ctypes.c_int, [ctypes.c_void_p]),
    ("posix_spawnattr_setflags", ctypes.c_int, [ctypes.c_void_p, ctypes.c_short]),
    # Darwin's ``sigset_t`` is a 32-bit mask (``sys/_types.h``:
    # ``__darwin_sigset_t`` is ``__uint32_t``), so ``c_uint32`` is the exact
    # spelling of the ``const sigset_t *`` this takes.
    (
        "posix_spawnattr_setsigdefault",
        ctypes.c_int,
        [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)],
    ),
    ("posix_spawn_file_actions_init", ctypes.c_int, [ctypes.c_void_p]),
    ("posix_spawn_file_actions_destroy", ctypes.c_int, [ctypes.c_void_p]),
    (
        "posix_spawn_file_actions_adddup2",
        ctypes.c_int,
        [ctypes.c_void_p, ctypes.c_int, ctypes.c_int],
    ),
    (
        "posix_spawn_file_actions_addopen",
        ctypes.c_int,
        [ctypes.c_void_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_uint16],
    ),
    ("posix_spawn_file_actions_addinherit_np", ctypes.c_int, [ctypes.c_void_p, ctypes.c_int]),
    ("posix_spawn_file_actions_addchdir_np", ctypes.c_int, [ctypes.c_void_p, ctypes.c_char_p]),
    ("responsibility_spawnattrs_setdisclaim", ctypes.c_int, [ctypes.c_void_p, ctypes.c_bool]),
)


def _fallback_reason() -> str | None:
    """Why the disclaimed path cannot run here, or ``None`` when it can.

    Checked before EVERY spawn: the check is a handful of ``getattr`` calls on
    a cached library handle, and re-checking keeps the module honest on a host
    where the first spawn happened under a different condition.
    """
    if sys.platform != "darwin":
        return "not macOS"
    libc = _load_libc()
    if libc is None:
        return "libSystem.B.dylib could not be loaded"
    for name in (
        "posix_spawn",
        "posix_spawnattr_init",
        "posix_spawnattr_setflags",
        "posix_spawnattr_setsigdefault",
        "posix_spawn_file_actions_init",
        "posix_spawn_file_actions_adddup2",
        "posix_spawn_file_actions_addopen",
        "posix_spawn_file_actions_addinherit_np",
        "posix_spawn_file_actions_addchdir_np",
        "responsibility_spawnattrs_setdisclaim",
    ):
        if not hasattr(libc, name):
            return f"missing symbol: {name}"
    return None


def _load_libc() -> Any:
    """The one non-constant security-relevant dependency, loaded lazily.

    ``responsibility_spawnattrs_setdisclaim`` is reachable from ``libSystem``
    on every macOS where the function exists; the handle is cached because a
    spawn path may run per-turn, and the prototypes are applied here so every
    call site passes arguments the way the ABI expects (see ``_PROTOTYPES``).
    """
    global _libc_cache, _libc_loaded
    if not _libc_loaded:
        _libc_loaded = True
        try:
            libc: Any = ctypes.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True)
        except OSError:  # pragma: no cover - libSystem always exists on darwin
            libc = None
        if libc is not None:
            for name, restype, argtypes in _PROTOTYPES:
                fn = getattr(libc, name, None)
                if fn is not None:
                    fn.restype = restype
                    fn.argtypes = argtypes
        _libc_cache = libc
    return _libc_cache


def _log_fallback(reason: str) -> None:
    """Say once per process why a spawn is NOT disclaimed. Never silent.

    LOUD (warning) for the cases that mean an OS change on darwin: a symbol is
    missing, or the kernel refused the spawn flags. Everything else (a
    non-macOS host, a spawn shape this module cannot express) is the
    platform's ordinary shape and logs at debug.
    """
    global _fallback_logged
    if _fallback_logged:
        return
    _fallback_logged = True
    if reason.startswith("missing symbol") or reason.startswith("posix_spawn refused"):
        logger.warning(
            "macOS responsibility disclaim unavailable (%s); falling back to "
            "subprocess detachment — long-lived runtimes cannot be disclaimed "
            "on this host (see local_operator/macos_disclaim.py for the "
            "2026-10-09 incident this protects against)",
            reason,
        )
    else:
        logger.debug("responsibility disclaim not applicable (%s); using subprocess", reason)


class SpawnUnsupportedError(Exception):
    """The requested spawn shape cannot be expressed as posix_spawn attributes.

    Private protocol between :func:`spawn_disclaimed` and its helpers; never
    escapes the public function, which degrades to ``subprocess.Popen`` when it
    sees one.
    """


# ---------------------------------------------------------------------------
# The spawn-chain evidence
# ---------------------------------------------------------------------------


def parse_spawn_chain(raw: object) -> list[dict[str, Any]] | None:
    """The chain from its recorded form — a JSON string or an already-parsed list.

    Both spellings exist by design: the environment variable carries a JSON
    string, while a boot record or receipt read back from disk hands over the
    parsed list. Malformed input yields ``None`` rather than a partial chain: a
    truncated read is not evidence, and the render side treats absent as "no
    evidence" (it says nothing extra) — which is the fail-closed reading.

    Entries are NORMALISED here, so every reader sees the same facts: the pid
    validated, the command bounded to ``MAX_ENTRY_CHARS`` (bounds an
    environment written by any producer, this build or not — agent review
    round 1, R1-3), and the optional liveness readings (``alive_at_spawn``,
    ``alive_now``) preserved as tri-states when the writer recorded them.
    """
    if not raw:
        return None
    if isinstance(raw, str):
        try:
            data = json.loads(raw)
        except ValueError:
            return None
    else:
        data = raw
    if not isinstance(data, list):
        return None
    entries: list[dict[str, Any]] = []
    for item in data[:MAX_CHAIN]:
        if not isinstance(item, dict):
            continue
        pid = item.get("pid")
        argv0 = item.get("argv0")
        if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
            continue
        entry: dict[str, Any] = {"pid": pid, "argv0": _bound_command(argv0)}
        for key in ("alive_at_spawn", "alive_now"):
            if key in item:
                value = item.get(key)
                # Anything that is not a bool or None is no evidence — read as
                # None, never as a reason to drop an otherwise valid entry.
                entry[key] = value if isinstance(value, bool) or value is None else None
        entries.append(entry)
    return entries or None


def spawn_chain_facts() -> list[dict[str, Any]] | None:
    """The chain THIS process was spawned with, from its environment.

    What the boot record stores: the facts recorded at spawn — including each
    member's liveness AT RECORD TIME (``alive_at_spawn``), taken when the
    spawner wrote the chain, and preserved here. The arrival half of the
    renderer's gate is added only when a signal actually arrives — see
    :func:`snapshot_spawn_chain`.
    """
    try:
        return parse_spawn_chain(os.environ.get(ENV_SPAWN_CHAIN))
    except Exception:  # noqa: BLE001 — attribution must never fail a boot or a signal
        return None


def _self_entry() -> dict[str, Any]:
    """This process, as a chain entry: the spawner the child records first."""
    argv0 = ""
    try:
        argv0 = os.path.basename(sys.argv[0] or "") or sys.executable
    except Exception:  # noqa: BLE001
        argv0 = sys.executable or ""
    return {"pid": os.getpid(), "argv0": _bound_command(argv0)}


def _process_table() -> dict[int, tuple[int, str]]:
    """One bounded ``ps`` snapshot: ``pid -> (ppid, command)`` for the whole table.

    ONE CALL FOR THE WHOLE WALK, not one per hop, and that is a measured
    requirement rather than a preference: the walk below runs on the spawn path,
    and the per-hop ``ps -p`` shape measured **~1.5 s for a 5-hop chain** on this
    fleet-loaded host — enough to blow the tight engage deadlines the runtime
    launch loop runs under (it cost the engagement tests their 5 s budgets the
    day it landed). A single ``ps -axo`` pays one fork instead of one per hop.
    Best-effort throughout: a slow or failing probe costs the chain, never the
    spawn.
    """
    try:
        out = subprocess.run(  # noqa: S603, S607 — fixed argv, no shell
            ["ps", "-axo", "pid=,ppid=,command="],
            capture_output=True,
            text=True,
            timeout=2,
        )
    except Exception:  # noqa: BLE001
        return {}
    table: dict[int, tuple[int, str]] = {}
    for raw in (out.stdout or "").splitlines():
        parts = raw.strip().split(None, 2)
        if len(parts) < 2:
            continue
        try:
            pid = int(parts[0])
            ppid = int(parts[1])
        except ValueError:
            continue
        table[pid] = (ppid, parts[2] if len(parts) > 2 else "")
    return table


#: The ancestor walk costs one ``ps`` fork, and THIS process's ancestors do not
#: change while it lives — so the first walk is kept and every later spawn in
#: the same process reads it for free. Keyed by (start_ppid, cap); the spawn
#: path is the only caller and it always starts from ``os.getppid()``.
_walk_cache: dict[tuple[int, int], list[dict[str, Any]]] = {}


def _walk_parent_chain(start_ppid: int, *, cap: int) -> list[dict[str, Any]]:
    """Walk up from ``start_ppid`` collecting ancestors, bounded by ``cap``.

    Stops at launchd (pid 1), on an unreadable pid, or when the cap is hit —
    and at a pid seen twice, so a pathological ppid cycle cannot loop. This is
    how a serve spawned directly by the desktop app records the app: the app
    is one hop above it and is still alive at spawn time.

    ONE process-table read serves the whole walk (:func:`_process_table`), and
    the result is memoised for this process's lifetime: the per-hop ``ps``
    shape this replaced measured ~1.5 s for 5 hops under fleet load — enough to
    blow the engagement loop's 5 s budgets — and even the single-snapshot form
    spends a fork a serve cannot amortise over N spawns. The facts recorded are
    "the lineage at spawn time", which for this process cannot change.

    Recorded commands are bounded to ``MAX_ENTRY_CHARS`` per entry (agent
    review round 1, R1-3): a ``ps`` command column is unbounded and can carry
    secrets, and the readers match a ``.app/Contents/MacOS/`` prefix the bound
    never truncates in practice.
    """
    key = (start_ppid, cap)
    cached = _walk_cache.get(key)
    if cached is not None:
        return cached

    table = _process_table()
    chain: list[dict[str, Any]] = []
    if table:
        seen: set[int] = set()
        pid = start_ppid
        while pid and pid > 1 and pid not in seen and len(chain) < cap:
            seen.add(pid)
            info = table.get(pid)
            if info is None:
                break
            ppid, command = info
            chain.append({"pid": pid, "argv0": _bound_command(command)})
            pid = ppid
    _walk_cache[key] = chain
    return chain


def _chain_for_child() -> list[dict[str, Any]] | None:
    """The chain to record for a child about to be spawned. Never raises.

    The immediate spawner first; then whichever of the two sources exists:
    the INHERITED chain (each lop process prepends itself, so a chain deepens
    down the tree) or, when there is none — the desktop app never sets this
    variable — a ppid walk at spawn time, which is what captures the app
    ancestor for a serve the app started directly.

    Every kept entry is annotated with its liveness at THIS moment
    (``alive_at_spawn``) — probed fresh for every member, inherited entries
    included, because an inherited reading describes an earlier moment (the
    spawner's own spawn) and what the renderer gates on is "alive when the
    chain was recorded for THIS child". The reading travels with the chain, so
    the boot record — which stores exactly this chain — carries it too; the
    arrival half is :func:`snapshot_spawn_chain`.
    """
    try:
        chain = [_self_entry()]
        inherited = spawn_chain_facts()
        if inherited:
            chain.extend(inherited)
        else:
            chain.extend(_walk_parent_chain(os.getppid(), cap=MAX_CHAIN - 1))
        # COPY each entry before annotating: the ppid-walk entries are the
        # memoised dicts in ``_walk_cache`` (this process's lineage, which
        # must stay a pure record of ancestry), and concurrent spawns from the
        # launch threads would otherwise write their readings into the same
        # objects between one thread's probe and its ``json.dumps`` (agent
        # review round 2, R2-3).
        chain = [dict(entry) for entry in chain[:MAX_CHAIN]]
        for entry in chain:
            entry["alive_at_spawn"] = _pid_liveness(entry.get("pid"))
        return chain
    except Exception:  # noqa: BLE001 — never fail a spawn over attribution
        return None


def _env_with_chain(env: Mapping[str, str] | None) -> dict[str, str]:
    """The child's environment with the chain recorded, or the input unchanged."""
    base = dict(env if env is not None else os.environ)
    chain = _chain_for_child()
    if chain:
        try:
            base[ENV_SPAWN_CHAIN] = json.dumps(chain, separators=(",", ":"))
        except (TypeError, ValueError):  # pragma: no cover - entries are scalars
            base.pop(ENV_SPAWN_CHAIN, None)
    else:
        # No chain could be built: drop any inherited spelling rather than let
        # a value this process cannot vouch for reach a grandchild as if it
        # described that spawn.
        base.pop(ENV_SPAWN_CHAIN, None)
    return base


def _pid_liveness(pid: object) -> bool | None:
    """Whether a chain member is still running: True / False / None (unknown).

    Type-validated here, then DELEGATED to :func:`local_operator.procstate.pid_liveness`
    — the one home for the platform's probe. The delegation is a correctness
    fix, not indirection for its own sake (agent review round 1, R1-1):
    ``os.kill(pid, 0)`` is a liveness probe on POSIX and a KILL on Windows
    (CPython routes any signal but Ctrl-C/Ctrl-Break to ``TerminateProcess``),
    and a probe that reports True after killing its subject is the worst shape
    there is. ``procstate`` owns the win32 ``OpenProcess`` branch for exactly
    that reason; this module must not re-implement the trap. ``None`` stays the
    fail-closed value — the renderer says nothing it cannot attest — and a pid
    that is not a positive int a ``pid_t`` can hold never reaches any platform
    probe (an int past 2**31-1 would raise ``OverflowError`` out of ``os.kill``
    instead of reading as unknown; agent review round 2, R2-N2). Imported
    lazily (as ``_fallback_popen`` does) so importing this module stays
    stdlib-only for the signal path.
    """
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0 or pid > _PID_MAX:
        return None
    from local_operator.procstate import pid_liveness

    return pid_liveness(pid)


def snapshot_spawn_chain(chain: list[dict[str, Any]] | None = None) -> list[dict[str, Any]] | None:
    """The chain with each member's liveness at THIS instant, as ``alive_now``.

    Called at signal arrival — the only moment the arrival half of the
    renderer's gate can exist. The recorded half (``alive_at_spawn``) travels
    on the entry itself (:func:`_chain_for_child`); this snapshot PRESERVES it
    and adds the arrival probe, so ``incidents`` can render the clause only
    when a member was recorded alive and is gone now (design round 1, D1).
    Every reading is measured, never inferred; ``None`` marks a probe that
    could not be made.
    """
    if chain is None:
        chain = spawn_chain_facts()
    if not chain:
        return None
    return [{**entry, "alive_now": _pid_liveness(entry.get("pid"))} for entry in chain]


# ---------------------------------------------------------------------------
# The spawn itself
# ---------------------------------------------------------------------------


class DisclaimedProcess:
    """The ``subprocess.Popen`` subset the wired call sites use, for a disclaimed child.

    Deliberately NOT a Popen subclass: a caller that needs Popen-only machinery
    (pipes, ``communicate()``) must fail visibly on the posix_spawn path rather
    than silently get a half-object — which is also why the spawn helper
    refuses unsupported stdio up front instead of emulating it. What IS
    supported is what the three runtime spawn sites use: ``pid``, ``poll()``,
    ``wait(timeout=None)``, ``returncode``, ``kill()``/``terminate()``/
    ``send_signal()``, and plain attribute assignment (``lop_capture_path``).

    ``wait`` and ``poll`` are serialised on an internal lock, the role
    ``Popen._waitpid_lock`` plays for a real child: the launch path parks a
    reaper thread on ``wait()`` while the engage loop calls ``poll()``, and two
    concurrent ``waitpid`` calls on one pid are exactly the race the lock
    exists to remove.
    """

    def __init__(self, pid: int, args: Sequence[str]) -> None:
        self.pid = pid
        self.args = list(args)
        self.returncode: int | None = None
        self._lock = threading.Lock()

    def _reap_when_exited(self) -> int | None:
        """Non-blocking reap; ``None`` while the child still runs."""
        with self._lock:
            if self.returncode is not None:
                return self.returncode
            try:
                wpid, status = os.waitpid(self.pid, os.WNOHANG)
            except ChildProcessError:
                # Not our child (something else reaped it, or it was never
                # ours — impossible for a pid posix_spawn returned). There is
                # no exit status to report; -1 keeps ``poll`` honest as
                # "exited, status unavailable" instead of hanging a waiter.
                self.returncode = -1
                return self.returncode
            if wpid == 0:
                return None
            self.returncode = os.waitstatus_to_exitcode(status)
            return self.returncode

    def poll(self) -> int | None:
        """The exit code once the child exits, ``None`` while it runs."""
        return self._reap_when_exited()

    def wait(self, timeout: float | None = None) -> int:
        """Block until the child exits; raise ``TimeoutExpired`` past ``timeout``.

        The no-timeout form is what the reaper thread parks on; the timeout
        form steps in 50 ms probes so one ``waitpid`` never blocks past the
        deadline a caller stated.
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            code = self._reap_when_exited()
            if code is not None:
                return code
            if deadline is not None and time.monotonic() >= deadline:
                assert timeout is not None  # a deadline exists only when one was stated
                raise subprocess.TimeoutExpired(self.args, timeout)
            time.sleep(0.05)

    def kill(self) -> None:
        if self.returncode is None:
            os.kill(self.pid, 9)

    def terminate(self) -> None:
        if self.returncode is None:
            os.kill(self.pid, 15)

    def send_signal(self, sig: int) -> None:
        os.kill(self.pid, sig)


def _stdio_action(fa: Any, fd: int, target: Any, *, is_stdin: bool) -> None:
    """One file action for one of 0/1/2. Raises ``SpawnUnsupportedError`` on shapes.

    The mapping is ``subprocess.Popen``'s, restricted to the forms the runtime
    spawn sites use: ``DEVNULL`` opens /dev/null at the slot, a file object (or
    an int fd) is dup2'd into it, ``STDOUT`` (stderr only) duplicates the
    already-arranged fd 1, and ``None`` inherits. ``PIPE`` is refused rather
    than approximated.
    """
    libc = _load_libc()
    if target is None:
        return
    if target is subprocess.PIPE:
        # Refused FIRST and explicitly: ``PIPE`` is the int -1, so the generic
        # fd branch below would silently ask for ``dup2(-1, fd)``.
        raise SpawnUnsupportedError("PIPE stdio is not expressible as file actions")
    if target is subprocess.DEVNULL:
        flags = os.O_RDWR if is_stdin else os.O_WRONLY
        rc = libc.posix_spawn_file_actions_addopen(ctypes.byref(fa), fd, b"/dev/null", flags, 0)
        if rc != 0:
            raise OSError(rc, os.strerror(rc))
        return
    if target is subprocess.STDOUT:
        rc = libc.posix_spawn_file_actions_adddup2(ctypes.byref(fa), 1, fd)
        if rc != 0:
            raise OSError(rc, os.strerror(rc))
        return
    if isinstance(target, bool):
        # ``bool`` is an ``int``; neither a descriptor nor a stream object.
        raise SpawnUnsupportedError("unsupported stdio target: bool")
    fileno: int
    if isinstance(target, int):
        fileno = target
    else:
        try:
            fileno = int(target.fileno())
        except (AttributeError, OSError, ValueError) as exc:
            raise SpawnUnsupportedError(
                f"unsupported stdio target: {type(target).__name__}"
            ) from exc
    rc = libc.posix_spawn_file_actions_adddup2(ctypes.byref(fa), fileno, fd)
    if rc != 0:
        raise OSError(rc, os.strerror(rc))


def _spawn_via_posix_spawn(
    argv: Sequence[str],
    *,
    executable: str,
    cwd: str | None,
    env: Mapping[str, str],
    stdin: Any,
    stdout: Any,
    stderr: Any,
    pass_fds: Sequence[int],
    close_fds: bool,
) -> int:
    """posix_spawn with SETSID + CLOEXEC_DEFAULT + SIGDEF + responsibility disclaim.

    Returns the child pid. Raises ``SpawnUnsupportedError`` for shapes to
    degrade, ``OSError`` for refusals the caller should see (missing binary),
    and lets attribute-level failures surface as ``OSError`` too — the caller
    decides which errnos degrade rather than raise.
    """
    libc = _load_libc()
    if not close_fds:
        raise SpawnUnsupportedError("close_fds=False is not expressible here")

    attr = ctypes.c_void_p()
    fa = ctypes.c_void_p()
    attr_inited = False
    fa_inited = False
    try:
        rc = libc.posix_spawnattr_init(ctypes.byref(attr))
        if rc != 0:
            raise OSError(rc, os.strerror(rc))
        attr_inited = True
        rc = libc.posix_spawn_file_actions_init(ctypes.byref(fa))
        if rc != 0:
            raise OSError(rc, os.strerror(rc))
        fa_inited = True
        flags = _POSIX_SPAWN_SETSID | _POSIX_SPAWN_CLOEXEC_DEFAULT | _POSIX_SPAWN_SETSIGDEF
        rc = libc.posix_spawnattr_setflags(ctypes.byref(attr), flags)
        if rc != 0:
            raise OSError(rc, os.strerror(rc))
        rc = libc.responsibility_spawnattrs_setdisclaim(ctypes.byref(attr), True)
        if rc != 0:
            raise OSError(rc, os.strerror(rc))
        # ``restore_signals`` PARITY (agent review round 1, R1-4): Popen resets
        # SIGPIPE/SIGXFZ/SIGXFSZ to default in the child; a raw posix_spawn
        # otherwise lets the parent's SIG_IGN dispositions survive execve
        # (measured on this host: a ``/bin/sh`` child kept SIGPIPE, SIGXFSZ
        # ignored). Darwin's sigset_t is the 32-bit mask the prototype above
        # spells; bit = signal number - 1.
        restored_bits = 0
        for signum in _RESTORED_SIGNALS:
            restored_bits |= 1 << (signum - 1)
        sigdefaults = ctypes.c_uint32(restored_bits)
        rc = libc.posix_spawnattr_setsigdefault(ctypes.byref(attr), ctypes.byref(sigdefaults))
        if rc != 0:
            raise OSError(rc, os.strerror(rc))

        # Order matters for ``stderr=STDOUT``: fd 1 is arranged first, then
        # duplicated onto 2 — the same order Popen applies them.
        _stdio_action(fa, 0, stdin, is_stdin=True)
        _stdio_action(fa, 1, stdout, is_stdin=False)
        _stdio_action(fa, 2, stderr, is_stdin=False)

        for fd in pass_fds:
            rc = libc.posix_spawn_file_actions_addinherit_np(ctypes.byref(fa), int(fd))
            if rc != 0:
                raise OSError(rc, os.strerror(rc))

        if cwd is not None:
            rc = libc.posix_spawn_file_actions_addchdir_np(ctypes.byref(fa), os.fsencode(cwd))
            if rc != 0:
                raise OSError(rc, os.strerror(rc))

        argv_c = (ctypes.c_char_p * (len(argv) + 1))(*[os.fsencode(str(a)) for a in argv], None)
        env_items = [f"{k}={v}" for k, v in env.items()]
        env_c = (ctypes.c_char_p * (len(env_items) + 1))(
            *[os.fsencode(item) for item in env_items], None
        )
        pid = ctypes.c_int(0)
        rc = libc.posix_spawn(
            ctypes.byref(pid),
            os.fsencode(executable),
            ctypes.byref(fa),
            ctypes.byref(attr),
            argv_c,
            env_c,
        )
        if rc != 0:
            raise OSError(rc, os.strerror(rc))
        return pid.value
    finally:
        # Destroy only what was initialised; a failed init must not leak the
        # other half, and destroy on an uninitialised opaque is undefined.
        if fa_inited:
            libc.posix_spawn_file_actions_destroy(ctypes.byref(fa))
        if attr_inited:
            libc.posix_spawnattr_destroy(ctypes.byref(attr))


#: Errnos where the kernel refused the ATTRIBUTES (not the program): an OS
#: change. These degrade to the fallback instead of surfacing, because the
#: caller's error handling is written for "the child could not start", not for
#: "this host no longer supports our spawn flags".
_ATTRIBUTE_ERRNOS = frozenset(
    {errno.EINVAL, errno.ENOSYS, errno.ENOTSUP} | {getattr(errno, "EOPNOTSUPP", 0)}
)


def spawn_disclaimed(
    argv: Sequence[str],
    *,
    executable: str | None = None,
    cwd: str | None = None,
    env: Mapping[str, str] | None = None,
    stdin: Any = None,
    stdout: Any = None,
    stderr: Any = None,
    pass_fds: Sequence[int] = (),
    close_fds: bool = True,
) -> Any:
    """Spawn ``argv`` detached AND out of this process's macOS responsibility scope.

    THE ONE PUBLIC HELPER for every long-lived runtime spawn (exec workers,
    session runtimes, standby warms). Returns a ``subprocess.Popen`` on the
    fallback path and a :class:`DisclaimedProcess` on the posix_spawn path —
    both satisfy what the call sites use (``pid``, ``poll()``, ``wait()``).

    Behaviour is ``Popen``'s plus two things Popen cannot do:

    * on macOS with the symbols present: ``posix_spawn`` with
      ``POSIX_SPAWN_SETSID | POSIX_SPAWN_CLOEXEC_DEFAULT |
      POSIX_SPAWN_SETSIGDEF`` and ``responsibility_spawnattrs_setdisclaim`` —
      the child leads its own session, inherits NO responsibility from this
      process (the app scope at the root of our chain no longer reaches it),
      and gets the signal dispositions Popen's ``restore_signals`` resets
      (SIGPIPE, SIGXFSZ; agent review round 1, R1-4);
    * always: the spawn chain is recorded into the child's environment
      (``LOP_SPAWN_CHAIN``) so a later signal can be rendered with the lineage
      it arrived against. Entries are bounded (``MAX_ENTRY_CHARS``); if the
      environment nonetheless crosses ARG_MAX, the chain is shed once and the
      spawn retried rather than failed (agent review round 1, R1-3).

    Fallback is today's exact behaviour (``subprocess.Popen`` +
    ``procstate.detached_popen_kwargs()``) whenever the platform lacks the
    lever, and it is logged (see :func:`_log_fallback`). Never silent.
    """
    executable = executable if executable is not None else str(argv[0])
    child_env = _env_with_chain(env)

    def start(spawn_env: Mapping[str, str]) -> Any:
        """One complete spawn decision (disclaimed path, or its fallback) for ``spawn_env``."""
        reason = _fallback_reason()
        if reason is not None:
            _log_fallback(reason)
            return _fallback_popen(
                argv,
                executable=executable,
                cwd=cwd,
                env=spawn_env,
                stdin=stdin,
                stdout=stdout,
                stderr=stderr,
                pass_fds=pass_fds,
                close_fds=close_fds,
            )
        try:
            pid = _spawn_via_posix_spawn(
                argv,
                executable=executable,
                cwd=cwd,
                env=spawn_env,
                stdin=stdin,
                stdout=stdout,
                stderr=stderr,
                pass_fds=pass_fds,
                close_fds=close_fds,
            )
        except SpawnUnsupportedError as exc:
            _log_fallback(str(exc))
            return _fallback_popen(
                argv,
                executable=executable,
                cwd=cwd,
                env=spawn_env,
                stdin=stdin,
                stdout=stdout,
                stderr=stderr,
                pass_fds=pass_fds,
                close_fds=close_fds,
            )
        except OSError as exc:
            if exc.errno in _ATTRIBUTE_ERRNOS:
                # The attributes were refused, not the program: an OS change.
                _log_fallback(f"posix_spawn refused the spawn flags (errno {exc.errno})")
                return _fallback_popen(
                    argv,
                    executable=executable,
                    cwd=cwd,
                    env=spawn_env,
                    stdin=stdin,
                    stdout=stdout,
                    stderr=stderr,
                    pass_fds=pass_fds,
                    close_fds=close_fds,
                )
            if exc.errno == errno.ENOENT:  # match Popen's exception class
                raise FileNotFoundError(exc.errno, exc.strerror, str(argv[0])) from exc
            raise
        return DisclaimedProcess(pid, list(argv))

    try:
        return start(child_env)
    except OSError as exc:
        if exc.errno != errno.E2BIG or ENV_SPAWN_CHAIN not in child_env:
            raise
        # THE CHAIN MUST NEVER FAIL A SPAWN THAT WOULD OTHERWISE START (agent
        # review rounds 1 and 2, R1-3 / R2-1). E2BIG means the environment
        # crossed the kernel's ARG_MAX; the chain this module added is the
        # only part of it we own, so shed exactly that and try once more. The
        # shed wraps BOTH the posix_spawn path and the Popen fallback: the
        # fallback receives the same chain-bearing env, and a caller env a few
        # KB under the limit fails there too (reproduced in round 2). A second
        # E2BIG then means the caller's own environment is over the limit — the
        # same failure ``Popen`` would have raised — and propagates from the
        # retry. Entries are bounded (``MAX_ENTRY_CHARS``), so this path takes
        # a pathological environment, not a long command line; it is logged
        # because the child loses its recorded lineage.
        logger.warning(
            "spawn chain dropped: environment over ARG_MAX (E2BIG); retrying without %s",
            ENV_SPAWN_CHAIN,
        )
        shed_env = {key: value for key, value in child_env.items() if key != ENV_SPAWN_CHAIN}
        return start(shed_env)


def _fallback_popen(
    argv: Sequence[str],
    *,
    executable: str,
    cwd: str | None,
    env: Mapping[str, str],
    stdin: Any,
    stdout: Any,
    stderr: Any,
    pass_fds: Sequence[int],
    close_fds: bool,
) -> subprocess.Popen[bytes]:
    """Today's spawn, unchanged: subprocess + the per-platform detach kwargs."""
    from local_operator.procstate import detached_popen_kwargs

    kwargs: dict[str, Any] = dict(
        executable=executable,
        cwd=cwd,
        env=dict(env),
        stdin=stdin,
        stdout=stdout,
        stderr=stderr,
        close_fds=close_fds,
    )
    if pass_fds:
        kwargs["pass_fds"] = tuple(pass_fds)
    kwargs.update(detached_popen_kwargs())
    # ``Popen``'s generic binds from the ARGV (strings) even though every wired
    # call site hands over binary stdio; the annotation states the CALLER
    # contract — bytes/opaque handles — not the argv spelling. The only Popen
    # surface these call sites touch is pid/poll/wait/kill, where the pair is
    # identical.
    spawned = subprocess.Popen(list(argv), **kwargs)  # noqa: S603 — fixed argv, no shell
    return cast("subprocess.Popen[bytes]", spawned)
