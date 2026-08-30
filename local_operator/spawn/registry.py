"""Which backend, if any, can open a window for a fork here.

The registry is why "support another terminal" is a new module rather than an
edit to a chain of ifs: a backend is appended to :data:`_BACKENDS` and the
command, the receipt and the tests are unchanged.
"""

from __future__ import annotations

from local_operator.spawn.apple import ITerm2Backend, TerminalAppBackend
from local_operator.spawn.cmux import PLACEMENT_WORKSPACE, CmuxBackend
from local_operator.spawn.ghostty import GhosttyBackend
from local_operator.spawn.kitty import KittyBackend
from local_operator.spawn.types import EnvMap, SpawnBackend, env_or_process
from local_operator.spawn.wezterm import WezTermBackend


def _ordered_backends(cmux_placement: str) -> tuple[SpawnBackend, ...]:
    """Every backend in DETECTION ORDER, which is deliberate, not alphabetical.

    **cmux is first, and that is the load-bearing part of this order.** cmux
    embeds ghostty and exports ``GHOSTTY_RESOURCES_DIR`` in every surface, so a
    ghostty-first order would detect ghostty inside a cmux session and open a
    stray OS window instead of the sidebar workspace the user expects. Verified
    on a real host: both ``CMUX_SURFACE_ID`` and ``GHOSTTY_RESOURCES_DIR`` are
    set at once in an ordinary cmux surface. ``multiplexer/registry.py`` orders
    itself the same way for the same reason.

    The rest do not co-occur in practice, but they keep a stable order so a host
    that somehow presents two markers always resolves the same way.
    """
    return (
        CmuxBackend(cmux_placement),
        GhosttyBackend(),
        KittyBackend(),
        WezTermBackend(),
        ITerm2Backend(),
        TerminalAppBackend(),
    )


def backends(cmux_placement: str = PLACEMENT_WORKSPACE) -> tuple[SpawnBackend, ...]:
    """Every registered backend. Exposed for tests and for documentation."""
    return _ordered_backends(cmux_placement)


def active_backend(
    env: EnvMap | None = None, *, cmux_placement: str = PLACEMENT_WORKSPACE
) -> SpawnBackend | None:
    """The backend that should open a fork from here, or None.

    None is an ORDINARY answer, not an error: a bare tty, a plain ssh session,
    an unrecognised emulator. The caller prints the fallback receipt, the fork
    still exists, and ``lop --resume <id>`` still reaches it.

    A backend whose ``detect`` raises is SKIPPED and the scan continues to the
    next one. Detection runs on the ``/fork`` command's own path, and a backend
    author's bug should cost that backend's feature rather than the user's fork.
    """
    source = env_or_process(env)
    for backend in _ordered_backends(cmux_placement):
        try:
            if backend.detect(source):
                return backend
        except Exception:  # noqa: BLE001 — see docstring: never break the fork
            continue
    return None
