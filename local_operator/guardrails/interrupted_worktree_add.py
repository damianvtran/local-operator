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
  so no escaped (``setsid``/``nohup``) add is still checking out.
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


def _live_cwd_inside(tree: str) -> bool | None:
    # Reuse the guard's cwd inventory, but read it fresh: a cached inventory
    # taken before an escaped add started would miss it. None = unknown.
    from local_operator.guardrails import provenance

    provenance._cwd_inventory_cache = None
    return provenance._live_cwd_inside(os.path.realpath(tree))


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
    command: str, *, cwd: str | None, env: Mapping[str, str], spawned_at: float
) -> list[str]:
    """Undo or finish this call's own interrupted adds; returns what it did."""
    if os.name != "posix" or "worktree" not in command or "--lock" in command:
        return []
    from local_operator.guardrails.resolve import resolve_command

    done: list[str] = []
    try:
        home = env.get("HOME") or os.path.expanduser("~")
        cwd_n = os.path.normpath(os.path.abspath(os.path.expanduser(cwd))) if cwd else None
        resolution = resolve_command(command, cwd=cwd_n, env=env, home=home)
        trees = [c.path for c in resolution.created if c.kind == "worktree" and c.path]
        for tree in dict.fromkeys(trees):
            admin = _registration(tree)
            if admin is None:
                continue
            locked = os.path.join(admin, "locked")
            if (_read(locked) or "").strip() != GIT_ADD_LOCK_REASON:
                continue
            with contextlib.suppress(OSError):
                if os.stat(locked).st_mtime < spawned_at:
                    continue  # a lock older than this call is not this call's
            if not os.path.exists(locked) or _live_cwd_inside(tree) is not False:
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
            settle_after_kill(command, cwd=cwd, env=env_copy, spawned_at=spawned_at)

    thread = threading.Thread(target=_run, name="interrupted-worktree-add", daemon=True)
    thread.start()
    return thread
