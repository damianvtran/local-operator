"""Typed point-in-time tool readiness for owning session callers.

WHY THIS EXISTS (architecture decision, 2026-10-13). ``SessionSpec.tools`` is
the run's whole reach, but it is a DECLARATION: nothing in it discovers,
connects or waits. MCP deliberately returns at its 250 ms startup gate and
finishes slower servers (``npx``/``uvx`` spawns) on background continuations, so
a caller that reads the inventory right after ``open_session`` can catch it
empty and prompt a model whose first request answers every call with
"Tool not found" (measured: the pilot turn failed exactly that way when the
prompt went out ~10 ms after open). Until this module there was no public typed
way to ask "are these tools reachable, and is discovery still moving?" --
callers hand-rolled a manager wait plus an inventory read, and the production
arm carried its own copy. This is the sanctioned replacement: a small owning
``Session`` API returning this immutable result.

WHAT IT GUARANTEES, AND WHAT IT DOES NOT.

* **Point-in-time, not a lifetime guarantee.** Underlying servers can drop
  later; ``ready`` answers "reachable at the settle point", nothing more.
* **Bounded; timeouts never cancel discovery.** The wait reuses the existing
  ``McpManager.wait_settled`` continuation wait, which leaves slow servers
  connecting on their own task. A timed-out call reports ``pending`` and a
  later fresh call can still return ``ready``.
* **Settled is not reachable; failed/filtered is never ready.** Only the LIVE
  enforced inventory decides; a failed server, a typo, or a name excluded by
  the declaration reads ``missing`` and never widens the inventory.
* **Cancellation is preserved.** A caller cancelled while waiting sees
  ``CancelledError``; the discovery lifecycle it was waiting on is untouched.
* **Owning sessions only.** An attached viewer is a facade onto a runtime in
  another process and cannot wait on its discovery; it does not get this API
  (see ``OwningSessionProtocol``), and must not pretend it can.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

#: The three point-in-time answers a readiness check can report.
#:
#: * ``ready`` -- every required name is in the LIVE enforced inventory.
#: * ``missing`` -- at least one name is not reachable and discovery is
#:   either settled or absent (nothing is going to add it).
#: * ``pending`` -- at least one name is not reachable YET while discovery is
#:   still in flight; a later check (or a longer bound) may still succeed.
ToolReadinessStatus = Literal["ready", "missing", "pending"]


@dataclass(frozen=True)
class ToolReadiness:
    """The immutable answer to ``Session.wait_for_tools``.

    ``required`` is the caller's deduplicated request order; ``present`` and
    ``missing`` partition it against the LIVE inventory read after the wait;
    ``discovery_settled`` says whether the owning session's MCP discovery had
    fully settled when the snapshot was taken (``None`` when no wait was
    performed — either every name was already present, or there is no
    discovery manager to wait on).
    """

    status: ToolReadinessStatus
    required: tuple[str, ...]
    #: The names in ``required`` reachable at snapshot time (live inventory).
    present: tuple[str, ...]
    #: The names in ``required`` NOT reachable at snapshot time (disjoint
    #: from ``present`` and, together with it, exactly ``required``).
    missing: tuple[str, ...]
    #: ``True``/``False`` from the bounded continuation wait; ``None`` when no
    #: wait ran. ``False`` with missing names is ``pending``, never ready.
    discovery_settled: bool | None = None

    @property
    def all_present(self) -> bool:
        """True when every required name is reachable (same as ``ready``)."""

        return not self.missing
