"""The trigger sources, and the one place to add another.

A source is a small module exposing a module-level ``SOURCE`` object that
implements the :class:`~local_operator.wakes.triggers.TriggerSource` protocol
(``name``, ``enabled(values)``, ``evaluate(ctx)``). Adding one is: write the
module, add it to ``_BUILTIN`` below, add its tests. Nothing in the registry,
the record layer, the supervisor or any rendered surface changes — that
generality is the point of this package.

Each module must stay stdlib-only at module scope for the same reason the
package's own modules do: :func:`load_builtin` runs inside the wake supervisor,
a ~40 MB process whose whole justification is that it does not carry the
harness. Heavier reads (the runtime registry, the projects store) belong in
function-local imports inside ``evaluate``, where a failure costs one source's
pass and never the process.
"""

from __future__ import annotations

from local_operator.wakes.triggers import register
from local_operator.wakes.triggers.sources import project_staleness

#: Every built-in source module, in registration order (the registry sorts by
#: name at evaluation time, so this list is only the roster).
_BUILTIN = (project_staleness,)

_LOADED = False


def load_builtin() -> None:
    """Register every built-in source exactly once per process."""
    global _LOADED
    if _LOADED:
        return
    _LOADED = True
    for module in _BUILTIN:
        register(module.SOURCE)


def reset_loaded() -> None:
    """Drop the load latch so a fresh :func:`load_builtin` can re-register.

    TEST SUPPORT, exactly like ``triggers._reset_registry`` (which calls this):
    production loads once per process, but a test that resets the registry
    needs the built-ins to come back through the lazy loader on its next
    evaluation — with this latch left set, the second load is a no-op against
    an empty registry and every sweep after the first test silently finds
    nothing.
    """
    global _LOADED
    _LOADED = False
