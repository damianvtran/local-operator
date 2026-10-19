"""Channel-ledger records for the classification layer's decision calls.

WHY THIS MODULE EXISTS
======================

The classification service owns its own ``httpx`` client and posts straight to the
decision endpoints (``vendors.py``). It never builds a ``ChatRequest``, so it never
passes through ``SessionStreamFn._record_stream`` — the one funnel the request
ledger observes — and until this module existed its cost was a log line and nothing
else (``session_factory._log_classification_cost``). The consequence was general,
not specific to any one front end: a run with classification active (the default)
could not certify its all-purpose request count or its spend, because up to
``legs x walks`` paid requests per user message were invisible to every ledger.

The session ledger already had everything needed to receive them — a
``classification`` channel (``channel_spend.CHANNELS``), a display label, and the
one ingest every surface reads (``Session.record_channel_spend``). Nothing emitted
to it. This module is the emitter's record shape; ``ClassificationService`` calls it
from the single place every decision call passes through
(``ClassificationService._call_cascade``).

ONE ROW PER LEG ATTEMPT, NOT PER SERVICE CALL; ``units`` IS THE WIRE REQUESTS
=============================================================================

A "classification call" from a caller's side (``recommend_resources`` / ``decide``)
can place up to ``DEFAULT_CASCADE_ATTEMPTS x len(legs)`` vendor requests inside one
1500 ms deadline, and each leg is a different provider, model and (for the Radient
leg) a different billing relationship. A row per service call would have to pick one
of them and could not count the rest, which is exactly the "cannot certify the
request count" defect this fixes. The leg attempt is also the unit the cascade
already counts (``DecisionVendorError.attempts``), so the ledger and the log agree.

``units`` is the number of HTTP requests that attempt actually placed, not a constant
1: the 401/403 ladder in ``vendors.decide`` re-sends up to twice inside one attempt,
and a leg with no credential places none. :class:`AttemptMeter` counts them at the one
place a request leaves the process (``vendors._HttpDecisionVendor._send``) through a
``ContextVar``, so the count is right when the message path and a monitor ``decide``
share a leg instance concurrently — a counter on the shared leg would cross-attribute.
An attempt that placed no request (no credential, cancelled while still resolving
one) records NOTHING: a row is a claim that a request was made.

THE RULES THE ROWS KEEP (inherited from ``channel_spend``, restated where they bite)
===================================================================================

* **Unknown is never zero.** A leg that answered without stating an amount — neither
  ``usage.cost`` nor a top-level ``cost`` object (Radient's route sends the latter) —
  records ``amount_micro=None`` (the record's ``__post_init__`` then forces the basis to
  ``not_tracked``). A stated ``0`` is a statement and survives as ``0``.
* **A failure the vendor refused is presumed unbilled** — ``status="failed"`` with no
  amount is the ledger's documented presumption and does not degrade the session's
  knowledge. That covers non-200 refusals and transport failures.
* **A request we abandoned may still have been charged.** A timeout, a session
  dispose, or a caller's cancel interrupts a request that may already be at the
  vendor, so it records ``status="cancelled"`` with no amount: countable as an
  ATTEMPT, and the session total honestly reads "partial" rather than claiming the
  money was zero. The same holds for a response lost after the request was sent
  (``DecisionVendorError.outcome_lost``: a read timeout or a reset mid-body).
* **A 200 we could not parse was served, so it may have been billed.** It records
  ``status="ok"`` with no amount for the same reason (``detail`` says why), not
  ``failed``: the "unbilled failure" presumption does not hold for a request the
  vendor completed.
* **A cache hit places no request and records nothing.** That falls out of the call
  site (the cache is consulted before the cascade), not out of anything here.

``cost_source`` is ``server_reported`` for the Radient leg (first-party server figure)
and ``provider_reported`` for the others (a third party's own ``usage.cost``). The
shared ``radient_cost.channel_record_from_cost`` converter is still deliberately NOT
used: it stamps ``server_reported`` on every row that carries an amount, which would
mislabel two of the three legs. (Its PARSER, ``cost_from_payload``, IS the one
adapter for the Radient ``cost`` object — ``vendors._parse_response`` reads the
amount and basis through it, since measured 2026-10-16 the Radient decision route
does send that object; only the record-stamping converter is bypassed.)

This module is a LEAF over ``channel_spend`` (itself a leaf), so importing it does
not drag the session package in. It is imported only by the service, which the
factory imports lazily; a session built with ``values.classification.auto`` false
never imports it.
"""

from __future__ import annotations

import contextvars
import logging
from collections.abc import Callable
from typing import Any

from local_operator.classification.types import (
    DecisionResponse,
    DecisionSchemaError,
    DecisionVendorError,
)
from local_operator.clients.radient_cost import micro_from_amount
from local_operator.model_ids import canonical_model_id
from local_operator.session.channel_spend import (
    BASIS_BILLED,
    COST_SOURCE_NONE,
    COST_SOURCE_PROVIDER,
    COST_SOURCE_SERVER,
    STATUS_CANCELLED,
    STATUS_FAILED,
    STATUS_OK,
    ChannelSpendRecord,
    new_record_id,
    now_ms,
)

logger = logging.getLogger(__name__)

#: The ledger channel (``channel_spend.CHANNELS``) these rows belong to.
CLASSIFICATION_CHANNEL = "classification"

#: The unit one row counts. ``calls`` is ``radient_cost._UNIT_SPELLINGS``' plural for
#: the wire's ``call``; summing it over a channel row gives the number of vendor
#: requests the layer placed, which is the figure a request-count certificate needs.
CALL_UNIT = "calls"

#: Where a record goes: ``Session.record_channel_spend`` in production. Bound late
#: (the service is built before the session exists), ``None`` until then.
SpendSink = Callable[[ChannelSpendRecord], None]

#: ``detail`` for an attempt interrupted before the vendor answered.
DETAIL_INTERRUPTED = "interrupted-in-flight"

#: ``detail`` for a response that was lost after the request was sent.
DETAIL_OUTCOME_LOST = "response-lost"

#: ``detail`` for a 200 whose body could not be used (served, so possibly billed).
DETAIL_UNUSABLE_RESPONSE = "unusable-response"

#: ``detail`` for our own request shape being rejected (a 400/422 naming a question).
DETAIL_SCHEMA = "schema-rejected"

#: ``detail`` for any other fault out of a leg (a bug in a leg, not weather).
DETAIL_ERROR = "error"

#: ``DecisionVendorError.kind`` of a 200 the leg could not parse. The vendor SERVED
#: it, which is the one failure shape the ledger's "an unreported failure was not
#: charged" presumption does not cover.
_SERVED_KIND = "response"


class AttemptMeter:
    """How many HTTP requests ONE leg attempt placed.

    A plain mutable counter held in a ``ContextVar`` for the duration of the attempt:
    the leg increments it at the moment a request leaves the process, and the
    service reads it after the attempt settles — including when the attempt was
    cancelled, which is the case a return value or an exception attribute cannot
    carry. Context-local rather than an attribute of the leg because a service's legs
    are shared by every caller of that service.
    """

    __slots__ = ("posts",)

    def __init__(self) -> None:
        self.posts = 0


_METER: contextvars.ContextVar[AttemptMeter | None] = contextvars.ContextVar(
    "classification_attempt_meter", default=None
)


def begin_attempt() -> tuple[AttemptMeter, contextvars.Token[AttemptMeter | None]]:
    """Open a meter for one leg attempt; pass the token to :func:`end_attempt`."""
    meter = AttemptMeter()
    return meter, _METER.set(meter)


def end_attempt(token: contextvars.Token[AttemptMeter | None]) -> None:
    """Close the meter opened by :func:`begin_attempt`."""
    _METER.reset(token)


def note_post() -> None:
    """Record that the running leg attempt is placing one HTTP request.

    The contract every :class:`~local_operator.classification.types.DecisionVendor`
    keeps: call this immediately before the request leaves the process. A no-op
    outside an attempt (a leg used on its own, as ``resolve_vendor`` probes one).
    """
    meter = _METER.get()
    if meter is not None:
        meter.posts += 1


def _cost_source(vendor: str) -> str:
    """Who stated the figure: Radient's own server, or a third party's receipt."""
    return COST_SOURCE_SERVER if vendor == "radient" else COST_SOURCE_PROVIDER


def answered_record(
    response: DecisionResponse, *, posts: int, requested_model: str = ""
) -> ChannelSpendRecord:
    """The row for a leg that answered: the vendor's own figure, or unknown.

    The row's ``model`` is the CANONICAL billing identity:
    ``response.cost_model`` (the id the server says it priced the amount from)
    when stated, else the leg's requested model, else the vendor's echo. The
    vendor echo itself stays in the free-text ``detail`` when it differs, so a
    row can still be reconciled against the wire without making that echo a
    second ledger identity.

    ``response.cost_usd is None`` (neither source stated an amount) stays an unknown
    amount — never ``0`` — so the session total is a lower bound rather than a
    claim.

    ``billing_basis`` is the leg's OWN basis when it stated one (Radient's
    top-level ``cost`` object labels a settled charge ``billed`` and an unsettled
    one ``estimated`` — a quote must never print as a charge), else today's
    ``billed`` default, which is what a third-party ``usage.cost`` is: the request's
    own receipt. Kept additive: ``cost_basis`` is optional on
    :class:`DecisionResponse`, so nothing that predates it changes behaviour.
    """

    # Channel rows persist in SQLite's signed INTEGER column. Decimal conversion
    # avoids float multiplication overflow, but an integer beyond the storage range
    # is still not a usable receipt. Keep it unknown, never truncate or invent zero.
    amount = micro_from_amount(response.cost_usd)
    if amount is not None and not 0 <= amount <= (1 << 63) - 1:
        amount = None
    billed_model = canonical_model_id(response.cost_model or requested_model or response.model)
    # Evidence only: the echo is not the billing identity. ``detail`` is the
    # record's free-text field (see ChannelSpendRecord) and carries no status
    # semantics, so a later reconciler can still see that the wire said
    # ``jev-1.13.0`` while the money was grouped under ``jev-1.13``.
    detail = f"vendor-model={response.model}" if response.model != billed_model else ""
    return ChannelSpendRecord(
        record_id=new_record_id(CLASSIFICATION_CHANNEL),
        ts_ms=now_ms(),
        channel=CLASSIFICATION_CHANNEL,
        provider=response.vendor,
        model=billed_model,
        units=posts,
        unit=CALL_UNIT,
        amount_micro=amount,
        billing_basis=response.cost_basis or BASIS_BILLED,
        cost_source=_cost_source(response.vendor) if amount is not None else COST_SOURCE_NONE,
        status=STATUS_OK,
        detail=detail,
    )


def failed_record(
    vendor: str, model: str, error: BaseException, *, posts: int
) -> ChannelSpendRecord:
    """The row for a leg attempt that raised instead of answering.

    A schema rejection and every non-200 / refused transport failure are ``failed``
    (the ledger presumes an unreported failure unbilled). Two shapes are NOT: a 200
    we could not parse (the vendor served it) is ``ok`` with an unknown amount, and a
    response lost after the request was sent is ``cancelled`` with an unknown amount —
    in both the vendor may have billed a request we have no figure for.
    """
    status, detail = STATUS_FAILED, DETAIL_ERROR
    if isinstance(error, DecisionSchemaError):
        detail = DETAIL_SCHEMA
    elif isinstance(error, DecisionVendorError):
        if error.kind == _SERVED_KIND:
            status, detail = STATUS_OK, DETAIL_UNUSABLE_RESPONSE
        elif error.outcome_lost:
            status, detail = STATUS_CANCELLED, DETAIL_OUTCOME_LOST
        else:
            detail = error.kind
    return ChannelSpendRecord(
        record_id=new_record_id(CLASSIFICATION_CHANNEL),
        ts_ms=now_ms(),
        channel=CLASSIFICATION_CHANNEL,
        provider=vendor,
        model=canonical_model_id(model),
        units=posts,
        unit=CALL_UNIT,
        amount_micro=None,
        status=status,
        detail=detail,
    )


def interrupted_record(vendor: str, model: str, *, posts: int) -> ChannelSpendRecord:
    """The row for an attempt we abandoned (timeout, dispose, cancel) mid-request.

    ``cancelled`` with no amount: counted as an attempt, and — unlike a refused
    request — possibly charged, so it degrades the session's knowledge to partial.
    """
    return ChannelSpendRecord(
        record_id=new_record_id(CLASSIFICATION_CHANNEL),
        ts_ms=now_ms(),
        channel=CLASSIFICATION_CHANNEL,
        provider=vendor,
        model=canonical_model_id(model),
        units=posts,
        unit=CALL_UNIT,
        amount_micro=None,
        status=STATUS_CANCELLED,
        detail=DETAIL_INTERRUPTED,
    )


def emit(sink: SpendSink | None, build: Callable[[], ChannelSpendRecord]) -> None:
    """Hand ONE record to the bound sink, best-effort.

    ``None`` is "no session bound yet" (the service was built, nothing ingests) and
    skips quietly. Never raises: ``Session.record_channel_spend`` already swallows its
    own faults, but the sink is an injectable callable and a lost spend row must not
    be able to fail a classification, let alone the turn that asked for it. The call
    adds no await of its own; its synchronous bookkeeping still consumes CPU. The
    sink owns persistence/publication. An unbound sink does not evaluate ``build``.
    """
    if sink is None:
        return
    try:
        # Construction belongs inside this guard too: a corrupt receipt or a record
        # factory fault must not replace a vendor answer, original error or cancel.
        sink(build())
    except Exception:  # noqa: BLE001 — a lost spend row is not a failed classification
        logger.debug("classification: channel-spend emission failed", exc_info=True)


def model_of(vendor: Any) -> str:
    """The model id a leg will send, or ``""`` for a double that publishes none."""
    return str(getattr(vendor, "model_id", "") or "")


__all__ = [
    "CALL_UNIT",
    "CLASSIFICATION_CHANNEL",
    "DETAIL_ERROR",
    "DETAIL_INTERRUPTED",
    "DETAIL_OUTCOME_LOST",
    "DETAIL_SCHEMA",
    "DETAIL_UNUSABLE_RESPONSE",
    "AttemptMeter",
    "SpendSink",
    "answered_record",
    "begin_attempt",
    "emit",
    "end_attempt",
    "failed_record",
    "interrupted_record",
    "model_of",
    "note_post",
]
