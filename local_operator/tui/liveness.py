"""The TUI's honest liveness term — what a viewer may say about its owner.

WHY THIS MODULE EXISTS. `is_cold` is the TUI's ONLY liveness term today
(``app.py:3059``, ``:6805``, ``:6903``, ``:6912``, ``:7330``, ``:7526``,
``:9048``; ``session_navigation.py:58``), and it is a purely LOCAL predicate:
``self._client is None or not self._client.connected or not
self._ready_for_events``. It has no round trip in it, so a viewer whose owner
stopped answering — a SIGSTOPped runtime, a machine mid-swap — keeps reading
live for as long as its socket stays open, because the kernel keeps the
connection and nothing observable locally changes. #1624 fixed exactly that on
the wire by publishing ``verified_at`` and refusing to claim live without it;
this module is the reader half, so the same honesty reaches the operator's own
screen instead of living only in a frame nobody reads (RR1-4).

THE TERM. ``AttachedSession.verified_at`` — when a round trip to the owner was
last ANSWERED — aged by the reader against ``LIVE_FRESHNESS_BUDGET_S``, plus
``attaching``, which is what lets a viewer tell **"coming"** from **"never"**.

NO ROUND TRIP LIVES HERE, and that is a constraint rather than an omission. The
read path answers in 170.7-350.6 ms and the warm receipt in 6.3-100.6 ms today;
a reader that dialled to look honest would trade the operator's latency target
for truthfulness he already has. Everything below is a clock comparison against
state the session already holds, and :func:`owner_liveness` is synchronous and
I/O-free by construction — ``test_the_reader_never_touches_the_owner`` is that
property, expressed as a test rather than as a promise.

THE COPY IS NOT MINE. :data:`LIVENESS_TEXT` is deliberately the single place the
words live, so the designer's strings are a substitution and not a refactor.
Nothing outside this module should spell a liveness word.
"""

from __future__ import annotations

import enum
import time
from typing import Any

from local_operator.session.runtime.types import LIVE_FRESHNESS_BUDGET_S

__all__ = [
    "LIVENESS_TEXT",
    "LIVE_FRESHNESS_BUDGET_S",
    "OwnerLiveness",
    "liveness_text",
    "owner_liveness",
]


class OwnerLiveness(enum.Enum):
    """What the viewer may honestly say about the session's owner.

    Ordered by precedence in :func:`owner_liveness`, and the order is the
    contract: a fresh answer outranks everything, and "coming" is claimed ONLY
    when a live claim cannot be made — which is the distinction the ``attaching``
    token exists for ("the runtime has accepted us and its state has not arrived
    yet" is not "the runtime is gone").
    """

    #: An answered round trip inside the budget: the owner is there.
    LIVE = "live"
    #: A dial is retained and its canonical state has not arrived yet. COMING,
    #: explicitly not NEVER — the whole reason the token survived review.
    COMING = "coming"
    #: The owner answered once, and has not answered for longer than the budget.
    #: Stale is not dead, and must never be drawn as live.
    STALE = "stale"
    #: No answered round trip, and nothing arriving: the honest "never".
    NEVER = "never"


#: THE ONE PLACE THE WORDS LIVE. These are placeholders chosen to be obviously
#: provisional rather than copy: the designer's three (or four) strings replace
#: the values here and nothing else moves. A caller must read them through
#: :func:`liveness_text` rather than reaching for a literal, so a reword cannot
#: miss a site — the same discipline the status band already uses for its
#: readings ("glyph must not imply a live reading, which is why the WORD beside
#: it is the part …", ``widgets/status_line.py``).
LIVENESS_TEXT: dict[OwnerLiveness, str] = {
    OwnerLiveness.LIVE: "live",
    OwnerLiveness.COMING: "attaching",
    OwnerLiveness.STALE: "unresponsive since",
    OwnerLiveness.NEVER: "no owner",
}


def liveness_text(state: OwnerLiveness) -> str:
    """The word for ``state``, from the single copy table above."""
    return LIVENESS_TEXT[state]


def owner_liveness(
    session: Any,
    *,
    now: float | None = None,
    budget: float = LIVE_FRESHNESS_BUDGET_S,
) -> OwnerLiveness:
    """Classify what ``session`` may say about its owner RIGHT NOW.

    ``verified_at`` is read with ``getattr`` and a ``None`` default: an owner-side
    ``Session`` has no stamp (it is not a viewer and never dials), and neither
    does a facade that has never had an answered round trip. Both are honestly
    :attr:`OwnerLiveness.NEVER` rather than an error — the term must be usable on
    every session the status band can be asked to render.

    ``attaching`` is likewise optional, and is consulted only when a live claim
    cannot be made. A resync on a session whose owner answered a moment ago is
    ``LIVE``: the stamp is the evidence, and treating a mid-refresh viewer as
    absent is the conflation ``owner_reachable`` exists to prevent.

    ``now``/``budget`` are parameters so the classification is testable without a
    clock or a runtime. Neither is a hook for behaviour: this function reads
    nothing but its argument and the clock.
    """
    stamp = getattr(session, "verified_at", None)
    if stamp is not None:
        if (now if now is not None else time.time()) - stamp <= budget:
            return OwnerLiveness.LIVE
        return OwnerLiveness.COMING if getattr(session, "attaching", False) else OwnerLiveness.STALE
    if getattr(session, "attaching", False):
        return OwnerLiveness.COMING
    return OwnerLiveness.NEVER
