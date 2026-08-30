"""The one interface every spawn backend implements, and what it carries.

Deliberately free of any backend import: the registry decides which backend is
live, and a backend that needed this module to know about it would make "support
another emulator" a two-file edit. Mirrors ``multiplexer/types.py``, whose rules
this package inherits wholesale.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from local_operator.terminals import EnvMap

__all__ = ["CALL_TIMEOUT_S", "EnvMap", "ForkLaunch", "SpawnBackend", "env_or_process"]

#: How long any one spawn call that must be WAITED on may take. Most launches
#: are fire-and-forget and need no bound, but two are not: the cmux surface
#: placement has to read back the id of the surface it just made, and the
#: WezTerm mux form has to learn whether it failed so the fallback form can run.
#: Both are bounded so an unreachable socket or a hung mux server costs a couple
#: of seconds and a fallback receipt, never the fork.
CALL_TIMEOUT_S = 5.0


def env_or_process(env: EnvMap | None) -> EnvMap:
    """``env`` when given, else the live process environment.

    One helper rather than ``env if env is not None else os.environ`` written at
    every entry point, because the fallback is what production uses and a call
    site that forgot it would silently detect nothing.
    """
    return os.environ if env is None else env


@dataclass(frozen=True)
class ForkLaunch:
    """Everything a backend needs to open the fork, decided by the caller.

    Frozen: what gets launched is settled once, at the point the fork was taken,
    so no backend can alter the command, the directory or the session it opens.
    """

    #: The FORK's 12-hex session id — never the parent's. This is what
    #: ``--resume`` takes and what the fallback receipt names.
    session_id: str

    #: Absolute path to the real launcher (``~/.local/bin/lop``), never the
    #: python interpreter running it. See ``broadcast.resume_executable``.
    executable: str

    #: The restore-and-idle argv, ``[executable, "--resume", session_id]``,
    #: built by ``broadcast.resume_argv`` so the safety boundary documented
    #: there has one definition. The fork's opening message rides a sidecar, not
    #: this argv.
    argv: tuple[str, ...]

    #: The PARENT session's working directory, not ``os.getcwd()`` at call time.
    #: Two reasons, and they are independent: a restore into ``$HOME`` would
    #: reopen the conversation pointing at the wrong project, and a different cwd
    #: changes the environment block AND which ``createIf`` tools resolve —
    #: altering the tool inventory that rides the cached prompt prefix, so the
    #: fork would miss the provider cache for a reason unrelated to forking.
    cwd: str

    #: Human-facing label for a window or workspace. Never model-generated text
    #: without sanitising: this string can reach a shell-adjacent surface.
    title: str = "local-operator"


@runtime_checkable
class SpawnBackend(Protocol):
    """Somewhere this app can open a window running a forked session.

    Two methods and no state. Backends are stateless by design: the terminal
    identity they need comes from the environment, which cannot change under a
    running process, so an instance would only be a place for a stale copy of it
    to live.
    """

    #: Stable identifier, used in log lines and in the fork receipt the user
    #: reads ("opened in a new cmux workspace"). Matches the emulator's own name.
    name: str

    def detect(self, env: EnvMap) -> bool:
        """Whether a fork opened from HERE should use this backend.

        Must be cheap and must never raise. Detection runs on the ``/fork``
        command's own path, and a backend author's bug should cost that
        backend's feature rather than the user's fork.
        """
        ...

    def spawn(self, launch: ForkLaunch, env: EnvMap) -> bool:
        """Open the window. True when it was started, False on any failure.

        Best-effort: returns False rather than raising, so the caller's only job
        is to fall through to the receipt. "Started" is as much as this can
        honestly promise — the child is detached and never waited on, so a
        window that opens and then dies still reports True.
        """
        ...
