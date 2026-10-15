"""READ-ONLY reads of agent messages, for the desktop plane (Team Bridge READ slice).

WHY THIS MODULE EXISTS. The connector (``team_bridge.connector``) is the only
thing that fetches, screens and ACKs an arrival, and it does that to WAKE a
session. A desktop that wants to show the person what arrived has no path at all
today: every read of the wire in this package also mutates local state
(``pull.inbox`` → ``Store.merge_page`` rewrites the connector's inbox checkpoint
and walk terminal, so a UI page would move the connector's cursor; ``pull.ack``
writes an ack intent). So this module exists to give ONE surface, the Read slice
of RFC §6.1, a way to read the wire page and the native screening tables and
return metadata — and to do it without any of those writes.

WHAT IT IS NOT. No decision, no signing, no capability bump of ``team_bridge``,
no wake, no screening call, no ack intent, no claim, no checkpoint, no
``merge_page``/``merge_refs``, and no message text. This slice returns
METADATA ONLY: ids, states, hashes and coarse screening codes. Returning a body
needs the shared content-admission predicate (owner standing ∧ validated envelope
∧ native receipt bound to the fetched bytes ∧ currency rules ∧ a confirmed read
intent); that predicate is not implemented yet, so no surface here may return one
— a route that did would serve an unreviewed body, which is the defect the
screening pipeline exists to prevent.

TWO STANDING CONTEXTS, and the difference is a real one, not a label:

- :class:`ReceiveReadContext` — the machine is enabled, it holds a committed
  owner whose consent record is present, and the captured credential row IS that
  owner's row. This is receipt/receive standing.
- :class:`HistoryReadContext` — the same authenticated participant and the same
  committed namespace, but the local feature switch is OFF (paused or disabled:
  one local key, so one reason): the caller may still read what a legitimate
  participant already received, and may not start anything new. Reads need no
  designated session, so the connector's ``no_designation`` fence code is
  deliberately NOT part of this decision.

The two contexts differ in STANDING, not in SOURCE: both answer from the same
authenticated wire page, and history mode reads no retained local rows. The
context is the receipt the caller is answered under (``mode``/``read_only``/
``history_reason``), not a second data path — this slice serves no stored
history shape.

A read is never a receive: neither context grants delivery, wake, screening or a
decision, and no path here widens paused history into new traffic.

NEVER ``str(exception)``. The refusal codes this module raises are translated by
the ROUTE through a closed table of fixed sentences, because the underlying
``PullRefused``/``StoreRefused``/``TeamBridgeTypeError`` messages can carry
server-supplied text (upstream codes are passed through the existing
``_raise_bridge`` mapping, which is where §6.5 codes belong).

IMPORT DISCIPLINE. This module is imported INSIDE the route handler, never at the
route module's scope: it reaches ``client``/``httpx`` through its callers'
imports, and the route rides ``lop serve``'s boot path.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlsplit

from local_operator.team_bridge import pull
from local_operator.team_bridge.pull import PullRefused
from local_operator.team_bridge.screening import HOLD_CODES
from local_operator.team_bridge.store import Store, StoreRefused
from local_operator.team_bridge.types import (
    InboxPage,
    OwnerSnapshot,
    TeamBridgeTypeError,
    project_inbox_page,
)

#: The read surface's version, reported as a receipt (``read_capability``) so a
#: renderer can tell WHICH read shape it is talking to. It is not a gate here:
#: the advertised capability key is ``team_bridge_read`` in the features map.
READ_CAPABILITY = 1

#: The slice's availability receipt (spec M-11 / §7): which read-family operations
#: EXIST on this build, stated so a renderer decides what to draw instead of
#: probing operations that are not there. M0 ships the metadata inbox only; the
#: other flags turn true only when their operations exist and pass their own
#: gates. It rides the team-bridge STATUS answer next to ``capability``, so one
#: response can drive the whole surface decision.
READ_AVAILABILITY: Mapping[str, bool] = {
    "metadata_inbox": True,
    "screened_detail": False,
    "read_ack": False,
    "local_events": False,
}

READ_MODE_RECEIVE = "receive"
READ_MODE_HISTORY = "history"

#: The coarse reasons a read runs on retained-history standing. CLOSED, and
#: deliberately the connector fence's own vocabulary for the same conditions
#: (``connector.wake._producer_fence``): the local switch is ONE key, so paused
#: and disabled are one reason, and a second spelling for it is how two fences
#: start disagreeing. The fence's OTHER conditions (``consent_revoked``,
#: ``owner_changed``) are refusals rather than history: the consent record is what
#: names the enrollment this namespace came from, and reading history under an
#: owner the store no longer holds is the stale answer the fence exists to stop.
HISTORY_REASONS = ("bridge_off",)

#: The screening states a row may report. CLOSED, and deliberately THREE: M0
#: reports what this machine RECORDED, never what it concluded.
#:
#: ``allowed`` is NOT one of them (review R-3 / M-12a). "Allowed" is an admission
#: claim, and admission needs the validated-envelope + currency + confirmed-intent
#: predicate this slice does not implement; a raw ``screening_receipts`` row is
#: bound to none of those, so deriving ``allowed`` from its verdict would be this
#: surface asserting a permission it cannot back. A receipt — whatever its
#: verdict — therefore leaves the row ``unreviewed``. ``blocked`` stays in the
#: vocabulary because M1's ack/receipt-bound path owns it: nothing in this slice
#: produces it, and a bare receipt ``reject`` is not the authoritative record
#: that would.
SCREENING_STATES = ("unreviewed", "held", "blocked")

#: What a hold code outside :data:`~local_operator.team_bridge.screening.HOLD_CODES`
#: is reported as. Connector-written hold rows carry free-form values
#: (``ack_not_confirmed:pending``, ``envelope_refused:<code>``), and those strings
#: can quote provider prose — they are not display copy.
UNKNOWN_HOLD_CODE = "unknown_hold"

DEFAULT_PAGE = 50
MAX_PAGE = 100

#: ID and cursor grammars, as PATTERNS for the route's strict query model: they
#: are checked BEFORE any I/O, because these values are interpolated into an
#: upstream path/query by the transport (``client.message``/``ack`` build
#: ``/messaging/messages/{id}``) and a ``..`` or ``?`` in an id rewrites the
#: request. Reject, never coerce. The cursor is opaque base64 of ``(sort_key,
#: id)`` (§6.3) — a POSITION only, so nothing here decodes or interprets it.
MESSAGE_ID_PATTERN = r"^bmsg_[A-Za-z0-9_-]{1,64}$"
THREAD_ID_PATTERN = r"^bth_[A-Za-z0-9_-]{1,64}$"
ACCOUNT_ID_PATTERN = r"^acc_[A-Za-z0-9_-]{1,64}$"
TENANT_ID_PATTERN = r"^tnt_[A-Za-z0-9_-]{1,64}$"
CURSOR_PATTERN = r"^[A-Za-z0-9_+/=-]{1,512}$"

#: The closed state filter the upstream accepts (§6.1: the DELIVERY state).
DELIVERY_STATES = ("unread", "read", "working", "done", "blocked")


class ReadRefused(Exception):
    """A typed refusal from the read path. ``code`` is the whole payload.

    The message is NOT carried: the route owns the fixed sentence per code, so a
    store/pull message that quotes server text can never reach a response.
    """

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class ReadContext:
    """Who a read is for, and which standing it runs under."""

    owner: OwnerSnapshot
    access: Any
    mode: str = READ_MODE_RECEIVE
    read_only: bool = False

    @property
    def account_id(self) -> str:
        return str(self.owner.account_id)

    @property
    def tenant_id(self) -> str:
        return str(self.owner.tenant_id)


@dataclass(frozen=True)
class ReceiveReadContext(ReadContext):
    """Receive/receipt standing: enabled, consented, committed owner, own row."""

    mode: str = READ_MODE_RECEIVE


@dataclass(frozen=True)
class HistoryReadContext(ReadContext):
    """Retained-history standing: the local switch is off; reads only, no new work."""

    reason: str = HISTORY_REASONS[0]
    mode: str = READ_MODE_HISTORY
    read_only: bool = True


@dataclass(frozen=True)
class ReadInbox:
    """One read page: the context it was read under, rows, cursor and counts."""

    context: ReadContext
    items: tuple[Mapping[str, Any], ...] = field(default_factory=tuple)
    next_cursor: str | None = None
    counts: Mapping[str, int] = field(default_factory=dict)


def _read_state(call: Callable[[], Any]) -> Any:
    """Run a READ-ONLY store accessor, mapping a damaged store to a refusal (R-7).

    The read-only accessors raise ``StoreRefused`` (a ``RuntimeError``) for a file
    they cannot read. This path's vocabulary is CLOSED, so the mapping belongs
    here ONCE rather than at each call site: without it a damaged
    ``state.sqlite3`` would leave the route as an untyped 500, and the code is
    the machine-readable half of the refusal.
    """

    try:
        return call()
    except StoreRefused as failure:
        raise ReadRefused(failure.code) from failure


def wire_for(config_dir: Path) -> Any:
    """The wire client for ONE read request, resolved at CALL time.

    WHY NOT ``pull._client_for`` (review R-2). That helper hands back the class
    object ``pull`` bound when it was first imported, while
    :func:`resolve_read_context` builds its client from ``client.TeamBridgeClient``
    at call time. ONE request therefore built TWO clients by two rules, and they
    could disagree — measured as ``destination_mismatch`` for any run where
    ``pull`` was imported before the fixtures patched the client module. Both
    builders now go through this one function, so a request has one construction
    rule; the route builds the wire once and hands the same object to both.
    """

    from local_operator.team_bridge import client as tb_client

    return tb_client.TeamBridgeClient(config_dir)


def _origin(url: str) -> str:
    parsed = urlsplit(str(url or ""))
    return f"{parsed.scheme}://{parsed.netloc}" if parsed.scheme and parsed.netloc else ""


def _bridge_enabled(config_dir: Path) -> bool:
    """The local switch, read from DISK (never a long-lived settings object).

    ``False`` for a torn read: the switch is what separates receive standing from
    retained-history standing, and a read that cannot tell must not claim the
    stronger one. Nothing here grants content, so the weaker context is the safe
    direction — it reports ``read_only``.
    """

    from local_operator.config import read_config_values
    from local_operator.team_bridge.settings import TeamBridgeSettings

    try:
        values = read_config_values(config_dir)
    except Exception:  # noqa: BLE001 — a torn read is not "still enabled"
        return False
    if values is None:
        return False
    return bool(TeamBridgeSettings.from_values(values).enabled)


def _receive_fence(store: Store, owner: OwnerSnapshot, access: Any) -> str:
    """The receive-standing fence, minus DESIGNATION. ``""`` when intact.

    Reads need no designated session (designation is what makes an arrival WAKE
    somebody), so this is the connector fence's other checks and no more: the
    local switch, the committed owner and its generation, the consent record, and
    the captured credential row. Returns the fence's own code for the first one
    that fails, which is what a caller renders as the coarse reason.
    """

    if not _bridge_enabled(store.config_dir):
        return "bridge_off"
    live = _read_state(store.active_owner_readonly)
    if live is None or live != owner:
        return "owner_changed"
    if _read_state(lambda: store.owner_consent_receipt_readonly(owner)) is None:
        return "consent_revoked"
    if int(getattr(access, "credential_id", -1)) != int(owner.credential_id):
        return "credential_changed"
    return ""


def _recheck(context: ReadContext, store: Store) -> None:
    """The post-await fence: the read must not answer for a moved owner.

    Re-run after the wire call, because every ``await`` is a boundary where the
    operator (or another process) can re-enroll, sign out or revoke. Both modes
    must stop here: retained history is history for THIS owner, and reading on
    behalf of a namespace that has since moved is exactly the stale-return this
    fence exists to refuse.
    """

    live = _read_state(store.active_owner_readonly)
    if live is None or live != context.owner:
        raise ReadRefused("owner_changed")
    if _read_state(lambda: store.owner_consent_receipt_readonly(context.owner)) is None:
        raise ReadRefused("consent_revoked")


async def resolve_read_context(
    config_dir: Path, *, expected_account_id: str, wire: Any = None
) -> ReadContext:
    """Resolve the read's namespace and standing, or refuse with a typed code.

    The captured account is the DECLARED one or nothing: the desktop plane's
    anti-replay receipt, the same rule the settings/allowlist writes carry. The
    committed owner is then read from the store and must be the row the captured
    credential belongs to — a read may not be answered for an account that
    happens to be signed in when the request was captured for another.

    The store is read READ-ONLY (review R-1/M-9): resolving whether a namespace
    exists must not create one. ``wire`` is the request's single client when the
    caller already built it (:func:`wire_for`), else this call builds one.
    """

    client = wire if wire is not None else wire_for(config_dir)
    access = await client.current_access()  # raises the sign-in / refusal states
    if str(getattr(access, "account_id", "")) != str(expected_account_id):
        raise ReadRefused("account_mismatch")
    if str(getattr(access, "kind", "")) != "oauth":
        raise ReadRefused("not_signed_in")
    store = Store(config_dir)
    owner = _read_state(store.active_owner_readonly)
    if owner is None:
        raise ReadRefused("history_context_unavailable")
    if int(getattr(access, "credential_id", -1)) != int(owner.credential_id):
        # The signed-in row is not this owner's row: there is no retained
        # namespace to read for the declared account, and guessing one is the
        # fallback the contract forbids.
        raise ReadRefused("history_context_unavailable")
    reason = _receive_fence(store, owner, access)
    if reason == "":
        return ReceiveReadContext(owner=owner, access=access)
    if reason not in HISTORY_REASONS:
        # A fence condition that is not history: report it rather than reading
        # under a standing the machine does not hold.
        raise ReadRefused(reason)
    return HistoryReadContext(owner=owner, access=access, reason=reason)


async def read_inbox_page(
    context: ReadContext,
    *,
    config_dir: Path,
    wire: Any = None,
    after: str | None = None,
    limit: int = DEFAULT_PAGE,
    state: str | None = None,
    thread_id: str | None = None,
    sender_account_id: str | None = None,
) -> ReadInbox:
    """ONE validated inbox page, projected to metadata. NO BRIDGE-STORE WRITES.

    The whole page must bind to THIS owner before anything is returned: a page
    that carries another tenant's or another endpoint's item is refused whole
    (``wrong_tenant``/``wrong_recipient``), never partially rendered, and nothing
    is persisted either way. The transport is ``client.inbox`` — the same call
    ``pull.inbox`` makes — and the difference from that function is exactly the
    missing ``merge_page``: the connector's checkpoint and walk terminal are not
    this surface's to move.
    """

    if limit < 1 or limit > MAX_PAGE:
        raise ReadRefused("bad_limit")
    store = Store(config_dir)
    # ONE wire per request, by the SAME rule the context resolved under (review
    # R-2): the caller's when the route already built one, else the call-time
    # factory here. The destination guard below still checks the CONCRETE object
    # it is handed, whichever construction produced it.
    client = wire if wire is not None else wire_for(config_dir)
    try:
        # The CONCRETE destination must still be the committed hub, checked
        # before any request: a client pointed at another origin is a different
        # trust boundary, not a read.
        pull._guard_destination(context.owner, client)  # noqa: SLF001 — the shared guard
    except PullRefused as failure:
        raise ReadRefused(failure.code) from failure
    result = await client.inbox(
        tenant_id=context.owner.tenant_id,
        after=after,
        limit=limit,
        state=state,
        thread_id=thread_id,
        sender_account_id=sender_account_id,
        access=context.access,
    )
    try:
        page: InboxPage = project_inbox_page(result)
    except TeamBridgeTypeError as failure:
        raise ReadRefused("malformed_page") from failure
    for item in page.items:
        if item.tenant_id != context.owner.tenant_id:
            raise ReadRefused("wrong_tenant")
        # The SAME binding invariant the connector enforces before it persists a
        # page (``Store._check_binding``): the recipient endpoint is MANDATORY and
        # exact, the account is compared when the server serves one. An item
        # addressed elsewhere — or nowhere — refuses the whole page here exactly
        # as ``pull.inbox`` refuses it, or this surface would render a row the
        # connector drops.
        if item.recipient_endpoint_id != context.owner.endpoint_id:
            raise ReadRefused("wrong_recipient")
        recipient = item.recipient_account_id
        if recipient is not None and recipient != context.owner.account_id:
            raise ReadRefused("wrong_recipient")
    _recheck(context, store)
    rows = tuple(_row(store, context.owner, item) for item in page.items)
    return ReadInbox(
        context=context,
        items=rows,
        next_cursor=page.next_cursor,
        counts=_counts(rows),
    )


def _screening_state(store: Store, owner: OwnerSnapshot, message_id: str) -> Mapping[str, Any]:
    """This machine's screening state for ONE message — code only, never prose.

    A RECEIPT IS NOT CONSULTED AT ALL (review R-3 / M-12a). Its verdict would
    have to be read as an admission (``allow``) or a block (``reject``), and a raw
    ``screening_receipts`` row is bound to neither the fetched bytes nor the
    envelope nor a currency check — so a surface that rendered it would be
    asserting content standing this slice cannot back. Whatever the receipt says,
    the row is ``unreviewed``; ``held`` comes from the native hold row, which is
    what this machine actually RECORDED. A hold row's code is passed through only
    when it is one of :data:`~local_operator.team_bridge.screening.HOLD_CODES`;
    connector-written rows carry free-form values that can quote provider text, so
    anything else is reported as ``unknown_hold``. The screener's private reason
    is never read. The whole read is READ-ONLY (review R-1).
    """

    hold = _read_state(lambda: store.screening_hold_readonly(owner, message_id))
    if hold is not None:
        code = str(hold.get("hold_code") or "")
        retry = hold.get("retry_at")
        return {
            "state": "held",
            "hold_code": code if code in HOLD_CODES else UNKNOWN_HOLD_CODE,
            "retry_at": float(retry) if isinstance(retry, (int, float)) else None,
        }
    return {"state": "unreviewed", "hold_code": None, "retry_at": None}


def _row(store: Store, owner: OwnerSnapshot, item: Any) -> Mapping[str, Any]:
    """One allowlisted metadata row: ids, states, coarse codes. Never text."""

    return {
        "message_id": item.message_id,
        "thread_id": item.thread_id,
        "message_type": item.message_type,
        "origin": item.origin,
        "seq": item.seq,
        "delivery": {"state": item.delivery_state, "state_at": item.state_at},
        "sender": {
            "account_id": item.sender_account_id,
            "endpoint_id": item.sender_endpoint_id,
        },
        "recipient": {
            "account_id": item.recipient_account_id,
            "endpoint_id": item.recipient_endpoint_id,
        },
        "request": {
            "state": item.request_state,
            "expires_at": item.request_expires_at,
            "ref": item.request_ref,
        },
        "screening": _screening_state(store, owner, item.message_id),
    }


def _counts(rows: tuple[Mapping[str, Any], ...]) -> Mapping[str, int]:
    """Counts over THIS page, no double-counting.

    ``unread`` is the upstream DELIVERY state of the page's rows; ``held`` and
    ``blocked`` are this machine's screening states for the same rows. A held
    message is one that never reached the session, so it is unread too — the
    held count is a SUBSET of the unread count, never added to it.
    """

    return {
        "items": len(rows),
        "unread": sum(1 for row in rows if row["delivery"]["state"] == "unread"),
        "held": sum(1 for row in rows if row["screening"]["state"] == "held"),
        "blocked": sum(1 for row in rows if row["screening"]["state"] == "blocked"),
    }


def inbox_view(result: ReadInbox) -> Mapping[str, Any]:
    """The serialisable answer: what a UI may render, and what it must not expect.

    ``body_included``/``screening_detail_included`` are stated rather than
    implied: a renderer must not read a metadata row's absence of text as a
    broken payload, and the two flags are the honest boundary of this slice.
    ``mode``/``read_only``/``history_reason`` are the standing receipt: an answer
    read under retained history says so, so a surface cannot present a read-only
    answer as a live inbox session.
    """

    context = result.context
    return {
        "mode": context.mode,
        "read_only": context.read_only,
        "history_reason": getattr(context, "reason", None),
        "read_capability": READ_CAPABILITY,
        "account_id": context.owner.account_id,
        "tenant_id": context.owner.tenant_id,
        "owner": {
            "endpoint_id": context.owner.endpoint_id,
            "hub_origin": _origin(context.owner.canonical_base_url),
            "enrollment_generation": int(context.owner.enrollment_generation),
        },
        "items": [dict(row) for row in result.items],
        "next_cursor": result.next_cursor,
        "counts": dict(result.counts),
        "body_included": False,
        "screening_detail_included": False,
        # §7: local invalidation events are NOT part of this slice, and a surface
        # told "maybe later" would poll. Said once, as a fact.
        "local_events": False,
    }


__all__ = [
    "ACCOUNT_ID_PATTERN",
    "READ_AVAILABILITY",
    "CURSOR_PATTERN",
    "DEFAULT_PAGE",
    "DELIVERY_STATES",
    "HISTORY_REASONS",
    "HistoryReadContext",
    "MAX_PAGE",
    "MESSAGE_ID_PATTERN",
    "READ_CAPABILITY",
    "READ_MODE_HISTORY",
    "READ_MODE_RECEIVE",
    "ReadContext",
    "ReadInbox",
    "ReadRefused",
    "ReceiveReadContext",
    "SCREENING_STATES",
    "TENANT_ID_PATTERN",
    "THREAD_ID_PATTERN",
    "UNKNOWN_HOLD_CODE",
    "inbox_view",
    "read_inbox_page",
    "resolve_read_context",
    "wire_for",
]
