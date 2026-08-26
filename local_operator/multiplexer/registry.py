"""Which backend, if any, hosts this process.

The registry is the reason "support another multiplexer" is a new module and
not an edit to a chain of ifs: a backend is appended to :data:`_BACKENDS` and
everything else — broadcast, retire, the TUI wiring, the tests — is unchanged.
"""

from __future__ import annotations

from local_operator.multiplexer.cmux import CmuxBackend
from local_operator.multiplexer.markers import (
    ScreenBackend,
    TmuxBackend,
    WezTermBackend,
    ZellijBackend,
)
from local_operator.multiplexer.types import EnvMap, MultiplexerBackend, env_or_process

#: Every known backend, in DETECTION ORDER, which is deliberate rather than
#: alphabetical. Multiplexers nest: a tmux session is routinely run inside a
#: cmux surface, and both sets of environment variables are then present. The
#: outermost host is the one that survives a crash and restores panes, so it
#: is the one worth publishing to, and cmux is therefore asked first. Within
#: the marker backends the order does not matter — their variables do not
#: co-occur in practice — but they keep a stable order so a host that somehow
#: has two always resolves the same way.
_BACKENDS: tuple[MultiplexerBackend, ...] = (
    CmuxBackend(),
    TmuxBackend(),
    ZellijBackend(),
    WezTermBackend(),
    ScreenBackend(),
)


def backends() -> tuple[MultiplexerBackend, ...]:
    """Every registered backend. Exposed for tests and for documentation."""
    return _BACKENDS


def active_backend(env: EnvMap | None = None) -> MultiplexerBackend | None:
    """The backend hosting this process, or None when there is no multiplexer.

    None is the ORDINARY case, not an error: most sessions run in a plain
    terminal, and everything downstream treats a missing backend as "nothing
    to publish" rather than as a failure to report.

    A backend whose ``detect`` raises is skipped rather than allowed to
    propagate. Detection is the very first thing that runs on a startup path
    that must never break, and a backend author's bug should cost that
    backend's feature, not the user's session.
    """
    source = env_or_process(env)
    for backend in _BACKENDS:
        try:
            if backend.detect(source):
                return backend
        except Exception:  # noqa: BLE001 — see docstring: never break startup
            continue
    return None
