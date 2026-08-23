"""Replace this process with a fresh launch of the same install.

WHY THIS EXISTS
---------------
``/reload`` used to reboot MCP/credentials/model inside the *old*
interpreter. After ``lop update`` (or any other package replace) that
cannot pick up the new wheel. ``/update`` and ``/reload`` therefore share
one helper that ends the TUI cleanly and then replaces the process.

WHY NOT ``os.execv`` FROM INSIDE THE TUI
----------------------------------------
Textual owns the alternate screen. Exec-ing over a live ``OperatorApp``
leaves the terminal half-raw. ``run_tui`` already returns
``app.return_code`` after ``run_async()`` and restores the terminal in
that teardown — that is the seam. The handler stashes a :class:`RestartPlan`,
exits with :data:`REEXEC_CODE`, and ``cli.main`` calls :func:`replace_self`
only after the TUI has released the tty.

``75`` is ``EX_TEMPFAIL``: not 0/1/130, which ``run_tui`` already uses
for success, error, and Ctrl-C.

POSIX uses ``os.execvpe`` so the same PID comes back with cwd/env
preserved by definition. Windows console shims make ``os.execv``
unreliable, so that backend is ``Popen`` + ``os._exit(0)``. One helper,
two OS backends — no detached babysitter unless a file-lock on
``uv tool upgrade`` later forces it.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass

#: Private exit from the TUI that means "replace this process", not "the
#: user quit". ``run_tui`` suppresses the ``session ended — resume with:``
#: hint on this code: that line is for a human staying in the shell.
REEXEC_CODE = 75


@dataclass(frozen=True)
class RestartPlan:
    """What the next process should look like.

    ``argv`` is the rewritten launch line. ``resume_id`` is recorded so
    tests (and a future Windows helper) can see the session the plan
    targeted without re-parsing argv.
    """

    argv: list[str]
    resume_id: str | None = None


def _drop_flag(argv: list[str], name: str) -> list[str]:
    """Remove ``name`` and, when it takes a value, the following token."""
    out: list[str] = []
    skip = False
    for token in argv:
        if skip:
            skip = False
            continue
        if token == name:
            skip = True
            continue
        if token.startswith(f"{name}="):
            continue
        out.append(token)
    return out


def plan_argv(
    original: list[str] | None = None,
    *,
    resume_id: str | None = None,
) -> list[str]:
    """``sys.argv`` as launched, with ``--resume`` rewritten.

    A transcript on disk (the same id ``resume_hint()`` advertises) is
    injected so the new process reopens this conversation. No transcript
    yet — first turn never persisted — strips any ``--resume`` so the new
    process is a cold launch, matching what today's in-process ``/reload``
    already accepts when it cannot rebind.

    A previous ``--resume`` is dropped before the new one is written so
    the flag cannot appear twice. ``--hosting`` / ``--model`` / ``--agent``
    / ``--tui`` / ``--debug`` / ``--train`` ride along because they were
    already on the launch line; we do not invent them.
    """
    launched = list(original if original is not None else sys.argv)
    if not launched:
        launched = [sys.argv[0] if sys.argv else "lop"]

    rewritten = _drop_flag(launched, "--resume")
    if resume_id:
        rewritten.extend(["--resume", resume_id])
    return rewritten


def make_plan(
    original: list[str] | None = None,
    *,
    resume_id: str | None = None,
) -> RestartPlan:
    return RestartPlan(argv=plan_argv(original, resume_id=resume_id), resume_id=resume_id)


_pending: RestartPlan | None = None


def stash_plan(plan: RestartPlan) -> None:
    """Park a plan for ``cli.main`` to consume after the TUI has torn down.

    The handler cannot ``exec`` from inside Textual (alternate screen). It
    exits 75 and leaves the plan here so ``replace_self`` runs only after
    ``run_tui`` has restored the terminal.
    """
    global _pending
    _pending = plan


def take_plan() -> RestartPlan | None:
    """Return and clear the parked plan (``None`` if the TUI never stashed)."""
    global _pending
    plan = _pending
    _pending = None
    return plan


def _on_windows() -> bool:
    """Indirection so tests can flip the backend without patching ``sys``."""
    return sys.platform == "win32"


def _replace_windows(argv: list[str], env: dict[str, str]) -> None:
    """Spawn a sibling then exit. ``os.execv`` on Windows console shims is unreliable."""
    flags = 0
    create_new = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    if isinstance(create_new, int):
        flags = create_new
    subprocess.Popen(  # noqa: S603 — relaunch of our own argv
        argv,
        cwd=os.getcwd(),
        env=env,
        close_fds=True,
        creationflags=flags,
    )
    os._exit(0)


def _replace_posix(argv: list[str], env: dict[str, str]) -> None:
    os.execvpe(argv[0], argv, env)


def replace_self(plan: RestartPlan) -> None:
    """Never returns on success. POSIX execs; Windows spawns then exits."""
    argv = list(plan.argv)
    if not argv:
        raise RuntimeError("RestartPlan.argv is empty")
    env = os.environ.copy()
    if _on_windows():
        _replace_windows(argv, env)
        return
    _replace_posix(argv, env)
