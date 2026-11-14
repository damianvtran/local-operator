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
* When that cwd check cannot say "no" — a live cwd inside, or a process whose
  cwd the kernel hides (Linux, non-dumpable: ``(sd-pam)`` on every host with
  a systemd user session) — NOTHING is deleted. The settle only runs
  ``git worktree unlock``, leaves the partial tree and its registration in
  place, and the bash result tells the agent to remove its own path (see
  :func:`left_in_place_notice`). PR #3980 rounds 1-3 showed no sound proof
  that a hidden-cwd process is outside the tree (``State`` and the switch
  counters lag a task that is about to run or has just run), so deletion
  stays gated on a cwd inventory that can answer "no"; unlocking deletes no
  working file and so needs no such proof.
* Branches are never deleted. POSIX only. Never raises: a miss leaves exactly
  what the harness left before this module existed.
"""

from __future__ import annotations

import contextlib
import logging
import os
import shlex
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
#: How long the bash tool waits for the settle before returning its result,
#: so a left-in-place notice can ride that result (``tools/builtin.py``). Covers
#: the group-gone wait plus an unlock or a small remove; past it the result goes
#: out without the notice and the settle still finishes on its thread.
NOTICE_WAIT_S = _GROUP_GONE_TIMEOUT_S + 10.0
#: Prefix of a ``settle_after_kill`` entry for a tree unlocked but kept because
#: the cwd check could not clear it; :func:`left_in_place_notice` reads it.
_LEFT_PREFIX = "unlocked, left in place: "


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
            if not os.path.exists(locked):
                continue
            complete = os.path.isfile(os.path.join(admin, "index")) and not os.path.exists(
                os.path.join(admin, "index.lock")
            )
            if complete:
                # A finished checkout only missed git's own unlock: finish it,
                # whoever may be using the tree (unlocking deletes nothing).
                if _git(admin, env, "unlock", tree):
                    done.append(f"unlocked {tree}")
            elif _live_cwd_inside(tree) is not False:
                # Removal needs a cwd inventory that says "no"; this one could
                # not (a live cwd inside, or a hidden one). Only drop the lock:
                # `git worktree unlock` deletes the `locked` file and nothing
                # else, and git itself (not a raw unlink) keeps doing the
                # bookkeeping. Prune stays safe: `_registration` proved
                # `<tree>/.git` exists and points back, so `git worktree prune`
                # keeps the registration. Without the lock the dangerous-delete
                # guard ranks the tree on its own state (a hidden pid is still
                # `busy`), not `locked`.
                if _git(admin, env, "unlock", tree):
                    done.append(f"{_LEFT_PREFIX}{tree}")
            elif _git(admin, env, "remove", "-f", "-f", tree):
                done.append(f"removed {tree}")
    except Exception as exc:  # noqa: BLE001 — cleanup is best-effort, never a crash
        logger.debug("interrupted worktree add settle failed: %s", exc)
    if done:
        logger.info("interrupted git worktree add settled: %s", "; ".join(done))
    return done


def left_in_place_notice(done: list[str]) -> str | None:
    """The bash-result line for trees the settle unlocked but could not remove.

    Model-facing tool output, so English by design (AGENTS.md i18n). It names
    the exact command because the agent created the path in this very call and
    is the one placed to clean it up; the harness deliberately did not.
    """
    trees = [entry[len(_LEFT_PREFIX) :] for entry in done if entry.startswith(_LEFT_PREFIX)]
    if not trees:
        return None
    return "\n".join(
        (  # i18n: ignore model-facing tool output
            f"The interrupted `git worktree add` left a half-made worktree at {tree}. "
            "It was unlocked but left in place, because a process's working "
            "directory couldn't be checked (or is inside it). Clean up your own "
            "path with: "
            f"git worktree remove --force {shlex.quote(tree)}"
        )
        for tree in trees
    )


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


class SettleThread(threading.Thread):
    """The daemon thread running one call's settle; keeps what it did.

    ``done`` is written once, by this thread, before it ends; read it only
    after a ``join`` that returned with the thread dead (:meth:`notice`).
    """

    def __init__(
        self,
        command: str,
        *,
        cwd: str | None,
        env: Mapping[str, str],
        spawned_at: float,
        pgid: int,
    ) -> None:
        super().__init__(name="interrupted-worktree-add", daemon=True)
        self._args = (command, cwd, dict(env), spawned_at, pgid)
        self.done: list[str] = []

    def run(self) -> None:
        command, cwd, env, spawned_at, pgid = self._args
        if _group_gone(pgid, _GROUP_GONE_TIMEOUT_S):
            self.done = settle_after_kill(command, cwd=cwd, env=env, spawned_at=spawned_at)

    def notice(self, timeout_s: float = NOTICE_WAIT_S) -> str | None:
        """Wait up to ``timeout_s`` for the settle; its notice, if any (blocking)."""
        self.join(timeout_s)
        if self.is_alive():
            logger.debug("interrupted worktree add settle still running; no notice")
            return None
        return left_in_place_notice(self.done)


def settle_in_background(
    command: str,
    *,
    cwd: str | None,
    env: Mapping[str, str],
    spawned_at: float,
    pgid: int | None,
) -> SettleThread | None:
    """Run :func:`settle_after_kill` off the event loop once ``pgid`` is gone.

    Called from the bash tool's ``_kill`` right after the group SIGKILL. A
    daemon thread: ``_kill`` is synchronous and its ~20 call sites assume it
    returns at once, and if ``lop`` exits first the lock simply stays (the
    pre-existing behaviour). The bash tool's result tail waits on it, bounded
    (:meth:`SettleThread.notice`), so a left-in-place notice reaches the
    agent in the same result. Returns the thread or ``None`` when there is
    nothing to do.
    """
    if pgid is None or os.name != "posix" or "worktree" not in command:
        return None
    thread = SettleThread(command, cwd=cwd, env=env, spawned_at=spawned_at, pgid=pgid)
    thread.start()
    return thread
