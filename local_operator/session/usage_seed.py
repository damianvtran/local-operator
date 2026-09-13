"""Seeding a status readout from the usages a transcript already recorded.

The one implementation of "which past provider reading may be shown", shared by
the two callers that need it and previously each had its own answer:

* the OWNER path, which seeds ``Session._last_usage`` from the replayed
  transcript at construction (``session.py``);
* the COLD viewer, which has no runtime to ask and no ``Session`` above it, so
  it seeds the same fields on the canonical state it publishes
  (``attached.py``). A resume whose checkpoint row is missing — the common case
  on a desktop, which detaches between requests and so never writes one — opened
  with an empty context and no spend for a conversation that might be heavily
  used.

Extracted rather than duplicated because the two must agree on WHICH reading
counts. The boundary rule (do not seed from a reading the newest compaction or
prune invalidated) is the load-bearing part, and a second copy of it is how the
two surfaces would come to disagree about the same conversation.

Import-light on purpose: this is arithmetic over already-parsed data, so it
pulls the harness's ``Usage`` model and the conversation-selection dataclass and
nothing from the TUI, the server or the config layer.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from local_operator.harness.types import Usage
from local_operator.session.model_selection import StoredModelSelection

logger = logging.getLogger(__name__)


def parse_usage(payload: Mapping[str, Any]) -> Usage | None:
    """One persisted ``usage`` payload as a :class:`Usage`, or ``None``.

    A transcript row is data from a previous process and may predate a field, so
    a payload that no longer validates is dropped rather than raised: a status
    readout must not be able to stop a session from opening. ``None`` simply
    falls through to the next-newest reading, and then to the local estimate.
    """
    try:
        return Usage.model_validate(payload)
    except Exception:
        logger.debug("dropping unparseable persisted usage payload", exc_info=True)
        return None


def last_reported_usage(usages: Sequence[Usage | None]) -> Usage | None:
    """The newest provider-reported :class:`Usage` in ``usages``, or ``None``.

    Scans BACKWARDS and stops at the first hit: the newest reading is the only
    one that describes the context as it now stands, and a resumed conversation
    can hold hundreds of entries to walk past.

    **Refuses any reading recorded before the newest compaction**, and that
    exception is the whole reason this is a function rather than a one-line
    scan. A compacted transcript replays as a summary marker followed by the
    KEPT WINDOW, and those kept messages still carry the ``usage`` they were
    given BEFORE the pass — figures describing a context that no longer exists,
    which nothing supersedes when the session compacted and then exited.

    Seeding from one is not a small error. Measured on a transcript that
    compacted at 900k of a 1M window, the reading came back 900_000 against a
    real 1_707 — 527x over, installed as EXACT so the correct local estimate
    could never replace it, and handed to ``should_compact``, which would then
    rewrite the user's history on the first turn after the resume.
    Under-reporting was the bug this seeding fixed; this is the same lie
    pointing the other way, and the compaction consequence makes it the more
    expensive of the two.

    The rule cannot be expressed on the replayed list alone. The marker sits at
    the HEAD of it and the kept window FOLLOWS it, so "stop scanning backwards
    at the marker" reads exactly backwards — the stale messages come first — and
    "any marker disqualifies everything" throws away the legitimate case: a
    session that compacted and then ran ten more turns has a perfectly good
    newest reading, and refusing it would send every such resume back to the
    local estimate for no reason.

    So the boundary is taken from the TRANSCRIPT, whose entries are in append
    order and therefore say which readings were recorded after the pass —
    :func:`~local_operator.session.transcript.usages_since_newest_shrink`
    returns exactly those; a history with no compaction returns all of them,
    which is the ordinary path.

    ``None`` means "no usable reading here", a real state and distinct from
    zero: a brand-new session, a conversation of nothing but user messages, a
    provider that reports no usage, or a compacted history with no completed
    turn since the pass. Callers must not collapse the two — a confident 0 on a
    resumed session is the empty-context lie this exists to prevent — and
    falling through to the local estimate is the right answer for all of them.
    """
    for usage in reversed(usages):
        if usage is not None:
            return usage
    return None


def seed_reported_usage(payloads: Iterable[Mapping[str, Any]]) -> Usage | None:
    """The newest usable reading among ``payloads``, which must be in append
    order (oldest first) — the order the transcript stores them in.

    The convenience the two seeding callers share: parse every payload, drop the
    ones that no longer validate, and take the newest survivor. Kept here beside
    the parser and the backwards scan so a caller cannot accidentally reverse
    the sequence, which would silently seed from the OLDEST reading.
    """
    return last_reported_usage([parse_usage(payload) for payload in payloads])


def reading_identity(
    usage: Usage | None, *, fallback: StoredModelSelection | None
) -> tuple[str, str] | None:
    """The ``(provider, model_id)`` that produced ``usage``, or ``None``.

    The usage's OWN stamp decides when it carries one: the failover layer stamps
    the spec that actually served the call, so a receipt knows which model it was
    measured on even when the session has since switched. Rows written before
    that stamp existed fall back to the conversation's saved selection — durable
    evidence of the model this conversation ran on, which is what makes an
    unstamped receipt attributable at all.

    ``None`` is the honest answer when neither exists, and callers must treat it
    as "cannot be attributed" rather than as a match: a reading nobody can name
    a model for is not a reading any denominator may be applied to.
    """
    if usage is not None:
        provider = str(usage.provider or "")
        model_id = str(usage.model_id or "")
        if provider and model_id:
            return (provider, model_id)
    if fallback is not None and fallback.provider and fallback.model_id:
        return (fallback.provider, fallback.model_id)
    return None


def _unknown_context_window() -> int:
    """``configure.UNKNOWN_CONTEXT_WINDOW``, imported where it is used.

    A function rather than a module constant because ``model.configure`` is the
    heavy model layer (its own module comment says so) and this module is
    deliberately import-light: the value is only needed once a window is about to
    be believed.
    """
    from local_operator.model.configure import UNKNOWN_CONTEXT_WINDOW

    return int(UNKNOWN_CONTEXT_WINDOW)


def reading_window(
    usage: Usage | None,
    *,
    fallback: StoredModelSelection | None,
    spec: Any,
) -> int | None:
    """The denominator this reading may be divided by, or ``None``.

    Non-``None`` only when the reading is attributable
    (:func:`reading_identity`), the model it names is the model that will run
    (``spec``'s own identity), the window was RESOLVED rather than defaulted
    (``spec.context_metadata_resolved``), and the window VALUE itself is
    vouched for — see below.

    ``context_metadata_resolved`` alone is NOT enough, and assuming it was is how
    this function first shipped a wrong denominator. ``UNKNOWN_CONTEXT_WINDOW``
    (``configure.py``) is 128_000 — a PLACEHOLDER — and ``context_spec_for_access``
    writes it TOGETHER WITH ``context_metadata_resolved: True`` on its two
    ordinary unresolved paths: the selected account resolved to nothing
    (``access is None``) and a catalogue row carrying no positive window. A real
    receipt of 322_546 tokens would then be divided by the placeholder and the band
    would print a MEASURED ``252.0%/128k`` for a conversation whose true budget is
    larger — the ``268.2%`` class of defect this module exists to prevent,
    reintroduced through the denominator instead of the numerator.

    So a window equal to that placeholder is refused UNLESS the model's own
    metadata corroborates it: ``default_context_window`` is the row's documented
    default budget, so a spec that reports 128_000 as its default is a model
    that really does serve 128k (the ``settings`` path in
    ``context_spec_for_access`` prefers exactly that value). In every placeholder
    case the field is ``None``, so the two are distinguishable without guessing.

    A ``None`` here is not a refusal to show the tokens — the caller still has
    the numerator — it is the band's honest ``window unknown`` state: absolute
    tokens and no arc, which is what the strip already renders.
    """
    identity = reading_identity(usage, fallback=fallback)
    if identity is None or spec is None:
        return None
    if (str(getattr(spec, "provider", "") or ""), str(getattr(spec, "model_id", "") or "")) != (
        identity
    ):
        return None
    if not getattr(spec, "context_metadata_resolved", False):
        return None
    window = int(getattr(spec, "context_window", 0) or 0)
    if window <= 0:
        return None
    if window == _unknown_context_window():
        documented_default = int(getattr(spec, "default_context_window", 0) or 0)
        if documented_default != window:
            return None
    return window
