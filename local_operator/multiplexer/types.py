"""The one interface every multiplexer backend implements, and what it carries.

Deliberately stdlib-only and free of any backend import: the registry decides
which backend is live, and a backend that needed this module to know about it
would make "add a multiplexer" a two-file edit (see the package docstring).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping, Protocol, runtime_checkable

#: The environment a backend reads to decide whether it is the host. Passed in
#: rather than read from ``os.environ`` inside each backend so detection is
#: testable without mutating process state — a test that has to
#: ``monkeypatch.setenv`` five variables to exercise one branch tends not to
#: exercise the other four.
EnvMap = Mapping[str, str]


@dataclass(frozen=True)
class SessionBinding:
    """What a pane needs to reopen this conversation later.

    Frozen because it is a description of a fact, not a builder: every field
    is decided once at broadcast time, and a backend that could mutate it
    could publish something the safety rules above never sanctioned.

    ``argv`` is the FULL launch line and is always a restore-and-idle command
    (see the package docstring). ``executable`` is spelled separately because
    the structured launch command cmux stores wants the launcher path on its
    own, and re-deriving it by slicing ``argv[0]`` in the backend would put
    that decision in three places.
    """

    #: The 12-hex session directory name under ``sessions/``. This is what
    #: ``--resume`` takes and what identifies the conversation on disk.
    session_id: str

    #: Absolute path to the real launcher (``~/.local/bin/lop``), never the
    #: python interpreter running it. See :func:`.broadcast.resume_executable`
    #: for why that distinction is load-bearing.
    executable: str

    #: The restore-and-idle argv, ``[executable, "--resume", session_id]``.
    argv: tuple[str, ...]

    #: Working directory to restore into. The user's cwd is part of what a
    #: pane was doing; a restore into ``$HOME`` would reopen the conversation
    #: pointing at the wrong project.
    cwd: str

    #: Human-facing label for multiplexer UI that shows one (cmux's restore
    #: prompt). Never model-generated text — see the pane-id sanitising note on
    #: ``_SAFE_PANE_ID`` in ``markers``: this string can reach a
    #: shell-adjacent surface.
    name: str = "local-operator"


@runtime_checkable
class MultiplexerBackend(Protocol):
    """A multiplexer this app can publish a resume binding to.

    Three methods and no state. Backends are stateless by design: the pane
    identity they need comes from the environment, which cannot change under a
    running process, so an instance would only be a place for a stale copy of
    it to live.
    """

    #: Stable identifier, used in log lines and in the marker files. Matches
    #: the multiplexer's own name (``cmux``, ``tmux``, ...).
    name: str

    def detect(self, env: EnvMap) -> bool:
        """Whether THIS process is running inside this multiplexer.

        Must be cheap and must never raise. A backend that shells out to
        answer this would put a subprocess on the startup path of every
        session on every host, including the ones running no multiplexer at
        all.
        """
        ...

    def publish(self, binding: SessionBinding, env: EnvMap) -> bool:
        """Record ``binding`` against this pane. True when it was published.

        Best-effort: returns False rather than raising on any failure, so the
        caller's only job is to log. See the package docstring — this must
        never be able to stop a session from starting.
        """
        ...

    def retire(self, binding: SessionBinding, env: EnvMap) -> bool:
        """Drop this pane's binding, so a NEW shell here does not replay it.

        Called on a clean exit. The asymmetry with :meth:`publish` is the
        point: a crash leaves the binding in place (that is the whole
        feature), while a user who quit deliberately has ended the session and
        must not have it reopened behind them.
        """
        ...


def env_or_process(env: EnvMap | None) -> EnvMap:
    """``env`` when given, else the live process environment.

    One helper rather than ``env if env is not None else os.environ`` written
    at every entry point, because the fallback is what production uses and a
    call site that forgot it would silently detect nothing.
    """
    return os.environ if env is None else env
