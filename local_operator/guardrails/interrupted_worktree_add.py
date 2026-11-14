"""Finish what a SIGKILLed ``git worktree add`` could not: git's own cleanup.

WHY. ``git worktree add`` locks the new worktree's registration with the reason
``initializing`` before it checks the tree out, and drops that lock when the add
ends. On an error or a catchable signal (SIGTERM, SIGINT) git's ``remove_junk``
handler deletes the half-made tree and its registration instead. SIGKILL cannot
be caught, and every bash-tool stop path (timeout, memory guard, pad watchdog,
abort, cancel) SIGKILLs the command's whole process group
(``tools/builtin.py`` ``_kill``). So the lock, and usually a partial checkout
with no ``index``, stayed behind. The dangerous-delete guard then ranks that
tree ``locked`` (HIGH), its creator could not remove it, and 14 such
registrations had piled up by 2026-11-13.

WHAT. After the harness killed a call's group, :func:`settle_after_kill` does
what git would have done, for the trees THAT call was adding:

* checkout incomplete (no ``index``, or ``index.lock`` present):
  ``git worktree remove -f -f`` (git refuses a locked tree with a single
  ``-f``), which removes the tree and its registration, as ``remove_junk`` would;
* checkout complete (``index`` present, no ``index.lock``): only
  ``git worktree unlock``. The tree is usable; only the unlock was missed.

CONSTRAINTS (fail closed: anything unproven is left alone):

* Only the exact ``worktree`` creation targets the guard's own resolver reads
  out of the killed command's literal text. A script or variable-driven add is
  not covered; ``scripts/worktree_janitor.py`` still reports it as STALE.
* ``<path>/.git`` names a registration whose ``gitdir`` points back at exactly
  that path. A path that is some other tree, or no tree, is not ours to touch.
* The registration's ``locked`` file reads exactly ``initializing`` and was
  written at or after the call's spawn: git wrote it during THIS call, so a
  lock on a pre-existing tree, or a reason someone chose, is never touched.
  A command passing ``--lock`` asked to keep its lock, so it is skipped.
* The process group is gone and no live process has its cwd inside the tree,
  so no escaped (``setsid``/``nohup``) add is still checking out. A process
  whose cwd the kernel hides (Linux, non-dumpable: ``(sd-pam)`` on every host
  with a systemd user session) counts as inside unless :class:`SpawnSnapshot`
  proves it has not run since before the tree existed.
* Branches are never deleted. POSIX only. Never raises: a miss leaves exactly
  what the harness left before this module existed.
"""

from __future__ import annotations

import contextlib
import logging
import os
import subprocess
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from local_operator.guardrails.provenance import ProcMark

logger = logging.getLogger(__name__)

#: The literal git writes (``builtin/worktree.c`` ``add_worktree``). Matched
#: exactly: any other reason is a lock someone meant to keep.
GIT_ADD_LOCK_REASON = "initializing"

#: How long the settle waits for the killed group to vanish. SIGKILL is not
#: deferrable, so the group is normally gone once asyncio reaps the leader
#: (measured well under 1 s); past this the settle gives up, fail closed.
_GROUP_GONE_TIMEOUT_S = 5.0
#: Bound on each git invocation. ``worktree remove`` walks the partial tree.
_GIT_TIMEOUT_S = 60.0


def _read(path: str) -> str | None:
    try:
        with open(path, encoding="utf-8") as handle:
            return handle.read()
    except (OSError, UnicodeDecodeError):
        return None


def _registration(tree: str) -> str | None:
    """The admin dir of the linked worktree at ``tree``, if it points back at it."""
    dotgit = os.path.join(tree, ".git")
    if os.path.islink(dotgit) or not os.path.isfile(dotgit):
        return None
    text = (_read(dotgit) or "").strip()
    if not text.startswith("gitdir: "):
        return None
    admin = os.path.join(tree, text[len("gitdir: ") :])
    back = (_read(os.path.join(admin, "gitdir")) or "").strip()
    if not back or os.path.realpath(back) != os.path.realpath(dotgit):
        return None
    return os.path.realpath(admin)


@dataclass(frozen=True)
class SpawnSnapshot:
    """What was true just BEFORE the call spawned, for proving a hidden pid out.

    WHY: Linux will not reveal the cwd of a non-dumpable same-uid process —
    not ``readlink``, not ``stat`` (both EACCES; measured on ``(sd-pam)``
    2026-11-14) — so the cwd inventory cannot say where it is, and treating it
    as "inside" meant no interrupted add ever settled on a host with a systemd
    user session. ``/proc/<pid>/status`` and ``/proc/<pid>/stat`` stay readable.

    THE PROOF, per hidden pid, for a tree in ``absent``:

    1. the tree path did not exist when the snapshot ended (checked after the
       marks were read), so every directory in it was created later, and a cwd
       inside it needs a ``chdir`` after the marks were read;
    2. the process has exactly one thread (``Threads: 1``) at the snapshot
       AND at settle. ``State`` and the switch counts are the leader thread's
       alone, but the cwd is shared by every thread, so a worker thread can
       ``chdir`` in under a parked leader whose mark never moves (reproduced
       in PR #3980 round 1). With one thread the leader's counters are the
       process's, and it cannot grow a thread without running itself;
    3. at settle the pid has the same start time (same process, not a reused
       pid) and the same context-switch total, and it was not ``State: R`` at
       either read. ``State: S`` only says "declared asleep": a task sets it
       just BEFORE calling ``schedule()``, so it is not proof of being off
       the CPU. What proves it did not run is the count: the kernel bumps a
       task's count each time it is switched OUT (``__schedule``, prev !=
       next), so a process that ran after the snapshot has been switched out
       since — and changed the count — unless it is still on the CPU at the
       settle's read. ``State`` is printed before the counts in one read, so
       a wake-up between them cannot pass.

    Anything else (a pid absent from the snapshot, a changed counter, a
    second thread, a ``State: R`` read, an unreadable mark, a target that
    already existed) stays "might be inside". OUTSIDE THE MODEL: (a) another
    actor renaming or bind-mounting an existing directory — one a hidden
    process already sits in — onto the absent target path during the call,
    which needs no ``chdir`` by that process (step 1 assumes the tree's
    directories are new); (b) a process that ran during the call and is
    still on the CPU, caught at the settle's single read between declaring
    sleep and being switched out — one read cannot tell "about to sleep"
    from "asleep". The remaining window — a process that ``chdir``s in after
    the settle's inventory — is the one the inventory always had.
    """

    absent: frozenset[str] = frozenset()
    marks: Mapping[int, ProcMark] = field(default_factory=dict)

    def cleared(self, pid: int) -> bool:
        from local_operator.guardrails import provenance

        before = self.marks.get(pid)
        if before is None:
            return False
        now = provenance._proc_mark(pid)
        return (
            now is not None
            and before.threads == 1
            and now.threads == 1
            and not before.on_cpu
            and not now.on_cpu
            and now.start == before.start
            and now.switches == before.switches
        )


def _targets(command: str, cwd: str | None, env: Mapping[str, str]) -> list[str]:
    """The exact ``worktree`` creation targets the guard's resolver reads."""
    from local_operator.guardrails.resolve import resolve_command

    home = env.get("HOME") or os.path.expanduser("~")
    cwd_n = os.path.normpath(os.path.abspath(os.path.expanduser(cwd))) if cwd else None
    resolution = resolve_command(command, cwd=cwd_n, env=env, home=home)
    trees = [c.path for c in resolution.created if c.kind == "worktree" and c.path]
    return list(dict.fromkeys(trees))


def snapshot_before_spawn(
    command: str, *, cwd: str | None, env: Mapping[str, str]
) -> SpawnSnapshot | None:
    """Record the :class:`SpawnSnapshot` for a call that may add a worktree.

    Called by the bash tool right before it spawns. Cheap and never raises:
    ``None`` (no proof available) unless this is a Linux ``/proc`` host and
    the command names a ``worktree`` add target that does not exist yet. Only
    same-uid pids whose cwd is hidden get a mark, so the cost is one
    ``/proc`` scan plus two small reads per hidden pid.
    """
    if os.name != "posix" or "worktree" not in command or "--lock" in command:
        return None
    try:
        from local_operator.guardrails import provenance

        trees = _targets(command, cwd, env)
        if not trees or not os.path.isdir("/proc/self"):
            return None
        marks: dict[int, ProcMark] = {}
        uid = os.getuid()
        for entry in os.scandir("/proc"):
            if not entry.name.isdecimal():
                continue
            try:
                if entry.stat().st_uid != uid:
                    continue
                os.readlink(os.path.join(entry.path, "cwd"))
            except PermissionError:
                mark = provenance._proc_mark(int(entry.name))
                if mark is not None:
                    marks[int(entry.name)] = mark
            except OSError:
                continue
        # After the marks (proof step 1): absent now means created after them.
        absent = frozenset(t for t in trees if not os.path.lexists(t))
        return SpawnSnapshot(absent=absent, marks=marks) if absent else None
    except Exception as exc:  # noqa: BLE001 — a missing snapshot only withholds a proof
        logger.debug("interrupted worktree add: spawn snapshot failed: %s", exc)
        return None


def _live_cwd_inside(tree: str, snapshot: SpawnSnapshot | None) -> bool | None:
    # Reuse the guard's cwd inventory, but read it fresh: a cached inventory
    # taken before an escaped add started would miss it. None = unknown,
    # including a hidden pid the snapshot cannot prove out.
    from local_operator.guardrails import provenance

    provenance._cwd_inventory_cache = None
    cleared = snapshot.cleared if snapshot is not None and tree in snapshot.absent else None
    return provenance._live_cwd_inside(os.path.realpath(tree), cleared)


def _git(admin: str, env: Mapping[str, str], *args: str) -> bool:
    common = os.path.realpath(os.path.join(admin, "..", ".."))
    try:
        proc = subprocess.run(
            ["git", f"--git-dir={common}", "worktree", *args],
            capture_output=True,
            text=True,
            env=dict(env),
            timeout=_GIT_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.debug("interrupted worktree add: git %s failed: %s", args, exc)
        return False
    if proc.returncode != 0:
        logger.debug("interrupted worktree add: git %s: %s", args, proc.stderr.strip())
    return proc.returncode == 0


def settle_after_kill(
    command: str,
    *,
    cwd: str | None,
    env: Mapping[str, str],
    spawned_at: float,
    snapshot: SpawnSnapshot | None = None,
) -> list[str]:
    """Undo or finish this call's own interrupted adds; returns what it did."""
    if os.name != "posix" or "worktree" not in command or "--lock" in command:
        return []

    done: list[str] = []
    try:
        for tree in _targets(command, cwd, env):
            admin = _registration(tree)
            if admin is None:
                continue
            locked = os.path.join(admin, "locked")
            if (_read(locked) or "").strip() != GIT_ADD_LOCK_REASON:
                continue
            with contextlib.suppress(OSError):
                if os.stat(locked).st_mtime < spawned_at:
                    continue  # a lock older than this call is not this call's
            if not os.path.exists(locked):
                continue
            live = _live_cwd_inside(tree, snapshot)
            if live is not False:
                # Logged so a tree left locked says why (busy vs unprovable).
                logger.debug("interrupted worktree add: %s left, live cwd=%s", tree, live)
                continue
            complete = os.path.isfile(os.path.join(admin, "index")) and not os.path.exists(
                os.path.join(admin, "index.lock")
            )
            if complete:
                if _git(admin, env, "unlock", tree):
                    done.append(f"unlocked {tree}")
            elif _git(admin, env, "remove", "-f", "-f", tree):
                done.append(f"removed {tree}")
    except Exception as exc:  # noqa: BLE001 — cleanup is best-effort, never a crash
        logger.debug("interrupted worktree add settle failed: %s", exc)
    if done:
        logger.info("interrupted git worktree add settled: %s", "; ".join(done))
    return done


def _group_gone(pgid: int, timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return True
        except PermissionError:
            # macOS answers EPERM, not ESRCH, while the killed members are
            # zombies not yet reaped (measured: the first probe right after
            # the SIGKILL). Keep polling; the deadline still fails closed.
            pass
        except OSError:
            return False
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.05)


def settle_in_background(
    command: str,
    *,
    cwd: str | None,
    env: Mapping[str, str],
    spawned_at: float,
    pgid: int | None,
    snapshot: SpawnSnapshot | None = None,
) -> threading.Thread | None:
    """Run :func:`settle_after_kill` off the event loop once ``pgid`` is gone.

    Called from the bash tool's ``_kill`` right after the group SIGKILL. A
    daemon thread: ``_kill`` is synchronous and its ~20 call sites assume it
    returns at once, and if ``lop`` exits first the lock simply stays (the
    pre-existing behaviour). Returns the thread (tests join it) or ``None``
    when there is nothing to do.
    """
    if pgid is None or os.name != "posix" or "worktree" not in command:
        return None
    env_copy = dict(env)

    def _run() -> None:
        if _group_gone(pgid, _GROUP_GONE_TIMEOUT_S):
            settle_after_kill(
                command, cwd=cwd, env=env_copy, spawned_at=spawned_at, snapshot=snapshot
            )

    thread = threading.Thread(target=_run, name="interrupted-worktree-add", daemon=True)
    thread.start()
    return thread
