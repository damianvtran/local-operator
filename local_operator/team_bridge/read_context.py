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
``merge_page``/``merge_refs``. Metadata is the DEFAULT answer: ids, states,
hashes and coarse screening codes.

MESSAGE TEXT IS ADMITTED, NEVER IMPLIED. The ONE way a body or an excerpt leaves
this module is :func:`admit_read_content` (RFC §4, review S-RSA-3): the fetched
item's envelope must validate against this owner and carry the hash the server
served, a native receipt must re-derive to the SAME bytes and envelope digest
under the still-published policy and this owner's generation, and a CONFIRMED
read intent must carry that exact tuple. A raw ``screening_receipts`` row is NOT
proof, and neither is a delivery state: those are the RECORD that screening
happened, and the predicate is what turns it into an admission. Anything short of
it leaves the message metadata-only (``content: null``) — a route that returned
bytes without it would serve an unreviewed body, which is the defect the
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

from local_operator.harness.message_types import DISPLAY_CONTEXT_BOUNDS
from local_operator.team_bridge import pull
from local_operator.team_bridge.pull import PullRefused
from local_operator.team_bridge.screening import HOLD_CODES, Envelope, validate_envelope
from local_operator.team_bridge.store import (
    RECEIPT_CORRUPT,
    Store,
    StoreRefused,
    receipt_authority,
)
from local_operator.team_bridge.types import (
    InboxPage,
    MessageRef,
    OwnerSnapshot,
    TeamBridgeTypeError,
    project_inbox_page,
    project_message_ref,
)

#: The read surface's version, reported as a receipt (``read_capability``) so a
#: renderer can tell WHICH read shape it is talking to. It is not a gate here:
#: the advertised capability key is ``team_bridge_read`` in the features map.
READ_CAPABILITY = 1

#: The slice's availability receipt (spec M-11 / §7): which read-family operations
#: EXIST on this build, stated so a renderer decides what to draw instead of
#: probing operations that are not there. M1a ships the detail read and the list
#: excerpts, both only through the shared admission predicate, so
#: ``screened_detail`` is true; ``read_ack`` turns true only when the ACK gate and
#: its POST exist (M1b), and ``local_events`` is deliberately outside this family.
#: It rides the team-bridge STATUS answer next to ``capability``, so one response
#: can drive the whole surface decision.
READ_AVAILABILITY: Mapping[str, bool] = {
    "metadata_inbox": True,
    "screened_detail": True,
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

#: The upstream delivery states in which CONTENT may be admitted (S-RSA-3, §4.6).
#: Closed, and deliberately not ``unread``: an item nobody has taken up yet has no
#: read standing to show its bytes, and ``blocked`` is a native refusal no local
#: proof may overrule. Read from the AUTHENTICATED upstream answer, never inferred
#: from a local receipt.
ADMITTABLE_DELIVERY_STATES = ("read", "working", "done")

#: How admitted text is labelled on the wire (§4). Both are constants because a
#: renderer keys on them: bridge text is INERT third-party data written by another
#: organization's member's agent, never instructions and never markup to unfurl.
CONTENT_TRUST = "untrusted_external"
CONTENT_FORMAT = "plain_text"

#: The list excerpt's bound: the SAME window the wake payload already ships
#: (§11.4's shared display bound). Imported from the constant's own module rather
#: than from the connector, because this module rides the route's import path and
#: the connector does not. A longer preview would be a second, unreviewed window
#: on the same bytes.
EXCERPT_MAX_CHARS = DISPLAY_CONTEXT_BOUNDS["excerpt"]

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
class ContentAdmission:
    """The verdict of the shared content predicate, and the text it admitted.

    ``reason`` is a CLOSED code from this module — never provider prose, never
    ``str(exception)``, and never rendered as an error: it is why a surface with
    no text is not a broken payload. The body rides ``text`` only when
    ``admitted``; the private carrier the envelope keeps off ``repr`` is read
    exactly once, there.
    """

    admitted: bool
    reason: str | None = None
    envelope: Envelope | None = None
    text: str | None = None


def admit_read_content(
    context: ReadContext, item: Mapping[str, Any], proof: Mapping[str, Any] | None
) -> ContentAdmission:
    """THE shared content-admission predicate (S-RSA-3) — PURE: no I/O, no await.

    Every term is required, and the ORDER is part of the contract: the envelope is
    authenticated against the FETCHED bytes first, so no stored row can be used to
    justify serving bytes that do not hash to what the server said it sent.

    1. ``validate_envelope`` on the exact fetched item: closed message type, the
       body's utf8 hash equal to the served ``body_hash``, the per-type size cap,
       and the thread/recipient bindings against THIS owner.
    2. The AUTHENTICATED delivery state must be read/working/done. ``unread``
       means no uptake happened and ``blocked`` is a native refusal; neither is
       inferred away from a local proof row.
    3. No native hold/bounce row: a hold is this machine's record that the item
       never reached the session, so it outranks any receipt.
    4. Policy, receipt and read intent must all be present.
    5. The receipt's own authority (:func:`receipt_authority`): published policy,
       admission revision, epoch and creator_generation, with decoded closed
       categories and a closed verdict.
    6. Verdict ``allow``, and its ``body_sha256`` AND ``envelope_digest`` equal to
       the recomputed ones. The digest covers sender/type/request as well as the
       body, so a changed sender under an old body hash denies — which is exactly
       why a raw receipt row can never be the proof.
    7. A CONFIRMED ``target=read`` intent whose whole immutable tuple (verdict,
       model, prompt version, template, body hash, categories) equals the receipt.
       Pending, conflicting or absent: metadata only, even beside an allow
       receipt, because the read is not recorded as taken up.

    ``proof`` is ONE entry of the snapshot
    :meth:`~local_operator.team_bridge.store.Store.content_proof_readonly`
    returned, so the decision reads one linearized moment.
    """

    envelope, code = validate_envelope(context.owner, item)
    if envelope is None:
        return ContentAdmission(False, code or "envelope_refused")
    if envelope.delivery_state not in ADMITTABLE_DELIVERY_STATES:
        return ContentAdmission(False, "delivery_not_admitted")
    proof = proof or {}
    if proof.get("hold") is not None:
        return ContentAdmission(False, "held")
    policy = proof.get("policy")
    receipt = proof.get("receipt")
    intent = proof.get("intent")
    if not isinstance(policy, Mapping) or not isinstance(receipt, Mapping):
        return ContentAdmission(False, "proof_missing")
    authority, damage = receipt_authority(context.owner, policy, receipt)
    if authority is None:
        return ContentAdmission(False, damage or RECEIPT_CORRUPT)
    if authority["verdict"] != "allow":
        return ContentAdmission(False, "not_allowed")
    if str(authority["body_sha256"] or "") != str(envelope.body_sha256):
        return ContentAdmission(False, "body_mismatch")
    if str(authority["envelope_digest"] or "") != str(envelope.envelope_digest):
        return ContentAdmission(False, "envelope_mismatch")
    if not isinstance(intent, Mapping):
        return ContentAdmission(False, "read_intent_missing")
    if not _intent_matches(authority, intent):
        return ContentAdmission(False, "read_intent_mismatch")
    return ContentAdmission(True, None, envelope=envelope, text=envelope.private_body())


def _intent_matches(authority: Mapping[str, Any], intent: Mapping[str, Any]) -> bool:
    """Does this read intent carry the receipt's whole immutable tuple?

    Only ``confirmed`` counts (S-RSA-3's term 5): a pending intent says the ACK
    was started, not that this participant took the message up, and a conflicting
    one says two writers disagree about which tuple it was. Both are metadata-only
    answers, and neither may be repaired here.
    """

    if str(intent.get("state") or "") != "confirmed":
        return False
    for term in ("verdict", "model", "prompt_version", "template_sha256", "body_sha256"):
        if str(authority.get(term) or "") != str(intent.get(term) or ""):
            return False
    categories = intent.get("categories")
    if not isinstance(categories, list):
        # The snapshot decodes that column and answers ``None`` for text that is
        # not the array it claims to be: unreadable proof cannot admit.
        return False
    return list(authority["categories"]) == list(categories)


def _content_object(admission: ContentAdmission, *, excerpt: bool) -> Mapping[str, Any] | None:
    """Admitted text as the object a renderer draws, or ``None`` for no content.

    The list form carries a bounded EXCERPT and the detail form the full body
    (M1a): one predicate, two windows, and the excerpt is the same cut the wake
    payload already ships. ``truncated`` is stated rather than inferred from the
    length, and the body hash rides along so a caller that later fetches the
    detail can tell whether these are the same bytes.
    """

    if not admission.admitted or admission.text is None:
        return None
    text = admission.text
    digest = str(admission.envelope.body_sha256) if admission.envelope is not None else ""
    if excerpt:
        cut = text[:EXCERPT_MAX_CHARS]
        return {
            "excerpt": cut,
            "excerpt_chars": len(cut),
            "truncated": len(cut) < len(text),
            "content_trust": CONTENT_TRUST,
            "content_format": CONTENT_FORMAT,
            "content_sha256": digest,
        }
    return {
        "text": text,
        "content_chars": len(text),
        "truncated": False,
        "content_trust": CONTENT_TRUST,
        "content_format": CONTENT_FORMAT,
        "content_sha256": digest,
    }


@dataclass(frozen=True)
class ReadDetail:
    """ONE message as the detail read answers it: metadata plus, maybe, its body.

    ``admission`` is carried rather than flattened so the caller can report
    WHETHER a body was admitted without reading the text: the view's flags are
    about the payload, never about the message.
    """

    context: ReadContext
    item: MessageRef
    #: The same allowlisted row shape a list row has, with the BODY in place of
    #: the excerpt. Built once by :func:`read_message_detail`, so the view layer
    #: has no second chance to decide what "content" means.
    row: Mapping[str, Any]


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
    # The RAW items ride along with the projected ones: the content predicate has
    # to see the exact mapping the server sent (its body bytes, its sender block,
    # its request block), while the projection is what binds the row to this
    # owner. ``project_inbox_page`` has already refused a non-list ``items``.
    raw_items = tuple(result.get("items") or ())
    for item in page.items:
        _bind_ref(context, item)
    # ONE proof snapshot for the WHOLE page (S-RSA-3), read after the wire call
    # and before the post-await fence: every row about to be serialized comes from
    # one consistent database image, and a fence that trips here denies the whole
    # answer rather than half of it.
    proofs = _read_state(
        lambda: store.content_proof_readonly(
            context.owner, [item.message_id for item in page.items]
        )
    )
    # No await between this snapshot, the admission and the serialization: the
    # invalidation checks for the context about to be answered run HERE, so a
    # re-enrollment during the wire call denies content instead of being answered
    # under the old owner's standing.
    _recheck(context, store)
    rows = tuple(
        _row(context, item, raw, proofs.get(item.message_id))
        for item, raw in zip(page.items, raw_items)
    )
    return ReadInbox(
        context=context,
        items=rows,
        next_cursor=page.next_cursor,
        counts=_counts(rows),
    )


def _screening_state(hold: Mapping[str, Any] | None) -> Mapping[str, Any]:
    """This machine's screening state for ONE message — code only, never prose.

    A RECEIPT IS NOT CONSULTED AT ALL (review R-3 / M-12a). Its verdict would have
    to be read as an admission (``allow``) or a block (``reject``), and a raw
    ``screening_receipts`` row is bound to neither the fetched bytes nor the
    envelope nor a currency check — so a surface that rendered it would be
    asserting content standing this slice cannot back. Whatever the receipt says,
    the row is ``unreviewed``; ``held`` comes from the native hold row, which is
    what this machine actually RECORDED. A hold row's code is passed through only
    when it is one of :data:`~local_operator.team_bridge.screening.HOLD_CODES`;
    connector-written rows carry free-form values that can quote provider text, so
    anything else is reported as ``unknown_hold``. The screener's private reason
    is never read.

    The hold row arrives from the CALLER's one proof snapshot (S-RSA-3) rather
    than from a reader of its own: a page must not open a connection per row, and
    the state a row reports has to be the state its admission was decided on.
    """

    if hold is not None:
        code = str(hold.get("hold_code") or "")
        retry = hold.get("retry_at")
        return {
            "state": "held",
            "hold_code": code if code in HOLD_CODES else UNKNOWN_HOLD_CODE,
            "retry_at": float(retry) if isinstance(retry, (int, float)) else None,
        }
    return {"state": "unreviewed", "hold_code": None, "retry_at": None}


def _row(
    context: ReadContext,
    item: MessageRef,
    raw: Mapping[str, Any],
    proof: Mapping[str, Any] | None,
    *,
    excerpt: bool = True,
) -> Mapping[str, Any]:
    """One allowlisted row: ids, states, coarse codes, and admitted content.

    ``content`` is the ONLY field that can carry third-party text, and it is built
    from the shared predicate's verdict (S-RSA-3): a row whose proof does not
    admit answers ``content: null`` while still reporting its metadata, so a
    renderer draws "there is a message here, this build will not show you its
    text" instead of an empty payload.
    """

    admission = admit_read_content(context, raw, proof)
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
        "screening": _screening_state((proof or {}).get("hold")),
        "content": _content_object(admission, excerpt=excerpt),
    }


def _bind_ref(context: ReadContext, item: MessageRef) -> None:
    """The binding invariant ONE item must satisfy to be THIS owner's (§6.2).

    The SAME rule the connector enforces before it persists a page
    (``Store._check_binding``): the recipient endpoint is MANDATORY and exact, the
    account is compared when the server serves one. An item addressed elsewhere —
    or nowhere — refuses the whole answer here exactly as ``pull.inbox`` refuses
    it, or this surface would render a row the connector drops.
    """

    if item.tenant_id != context.owner.tenant_id:
        raise ReadRefused("wrong_tenant")
    if item.recipient_endpoint_id != context.owner.endpoint_id:
        raise ReadRefused("wrong_recipient")
    recipient = item.recipient_account_id
    if recipient is not None and recipient != context.owner.account_id:
        raise ReadRefused("wrong_recipient")


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


async def read_message_detail(
    context: ReadContext,
    *,
    config_dir: Path,
    message_id: str,
    tenant_id: str,
    wire: Any = None,
) -> ReadDetail:
    """ONE message, with its body ONLY through the shared predicate (M1a).

    Same transport ``pull`` uses for a single message, and the same binding rule
    as a page, with the two things this surface must NOT do stated as structure:
    no ``merge_refs``/checkpoint write, and no value returned from the store that
    a reader can mistake for the body. A message whose proof does not admit
    answers its metadata and ``content: null`` — held, blocked, unreviewed and
    not-yet-read all land there, with this machine's coarse screening state and
    never the screener's reason.
    """

    store = Store(config_dir)
    client = wire if wire is not None else wire_for(config_dir)
    try:
        # The CONCRETE destination must still be the committed hub, checked before
        # any request: a client pointed at another origin is a different trust
        # boundary, not a read.
        pull._guard_destination(context.owner, client)  # noqa: SLF001 — the shared guard
    except PullRefused as failure:
        raise ReadRefused(failure.code) from failure
    result = await client.message(
        message_id,
        tenant_id=tenant_id,
        access=context.access,
    )
    raw: Mapping[str, Any] = result if isinstance(result, Mapping) else {}
    try:
        item: MessageRef = project_message_ref(raw)
    except TeamBridgeTypeError as failure:
        raise ReadRefused("malformed_page") from failure
    if item.message_id != message_id:
        # The id rides the PATH, so an answer for another message is not this
        # read at all: the same bind-the-answer-to-the-request rule a page gets,
        # one level below the grammar that already refused a rewritable id.
        raise ReadRefused("malformed_page")
    _bind_ref(context, item)
    proofs = _read_state(lambda: store.content_proof_readonly(context.owner, [item.message_id]))
    # No await between the snapshot, the admission and the serialization (§4):
    # the standing checks for the context about to be answered run HERE.
    _recheck(context, store)
    return ReadDetail(
        context=context,
        item=item,
        row=_row(context, item, raw, proofs.get(item.message_id), excerpt=False),
    )


def _owner_receipt(context: ReadContext) -> Mapping[str, Any]:
    """The owner half of every read answer: who this read was made under (§3).

    One function for both surfaces, because a renderer that saw a different
    receipt on a detail than on the page it came from could not tell a switch of
    account from a bug. ``hub_origin`` is the ORIGIN, never the canonical URL: the
    receipt says which hub, not which path a caller may build.
    """

    return {
        "endpoint_id": context.owner.endpoint_id,
        "hub_origin": _origin(context.owner.canonical_base_url),
        "enrollment_generation": int(context.owner.enrollment_generation),
    }


def detail_view(result: ReadDetail) -> Mapping[str, Any]:
    """The single-message answer: metadata always, content only when admitted.

    ``body_included``/``screening_detail_included`` say whether THIS payload
    carries text, so a renderer can tell "this build will not show you the bytes"
    from "there is nothing here". ``content`` repeats the row's own object rather
    than inventing a second shape for the same bytes: a list excerpt and a detail
    body differ only in which window they were cut to, and the field names say
    which one a caller is holding.
    """

    context = result.context
    content = result.row.get("content")
    return {
        "mode": context.mode,
        "read_only": context.read_only,
        "history_reason": getattr(context, "reason", None),
        "read_capability": READ_CAPABILITY,
        "account_id": context.owner.account_id,
        "tenant_id": context.owner.tenant_id,
        "owner": _owner_receipt(context),
        "item": dict(result.row),
        "content": dict(content) if isinstance(content, Mapping) else None,
        "body_included": isinstance(content, Mapping),
        "screening_detail_included": isinstance(content, Mapping),
        # §7: local invalidation events are NOT part of this slice, and a surface
        # told "maybe later" would poll. Said once, as a fact.
        "local_events": False,
    }


def inbox_view(result: ReadInbox) -> Mapping[str, Any]:
    """The serialisable answer: what a UI may render, and what it must not expect.

    ``body_included``/``screening_detail_included`` are stated rather than
    implied: a renderer must not read a metadata row's absence of text as a
    broken payload, and the two flags are the honest boundary of this slice — with
    the excerpt predicate in place they say whether ANY row on this page carried
    content, which is a fact about the page and not about the messages in it.
    ``mode``/``read_only``/``history_reason`` are the standing receipt: an answer
    read under retained history says so, so a surface cannot present a read-only
    answer as a live inbox session.
    """

    context = result.context
    has_content = any(row.get("content") for row in result.items)
    return {
        "mode": context.mode,
        "read_only": context.read_only,
        "history_reason": getattr(context, "reason", None),
        "read_capability": READ_CAPABILITY,
        "account_id": context.owner.account_id,
        "tenant_id": context.owner.tenant_id,
        "owner": _owner_receipt(context),
        "items": [dict(row) for row in result.items],
        "next_cursor": result.next_cursor,
        "counts": dict(result.counts),
        "body_included": has_content,
        "screening_detail_included": has_content,
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
    "ADMITTABLE_DELIVERY_STATES",
    "CONTENT_FORMAT",
    "CONTENT_TRUST",
    "ContentAdmission",
    "EXCERPT_MAX_CHARS",
    "ReadContext",
    "ReadDetail",
    "ReadInbox",
    "ReadRefused",
    "ReceiveReadContext",
    "SCREENING_STATES",
    "TENANT_ID_PATTERN",
    "THREAD_ID_PATTERN",
    "UNKNOWN_HOLD_CODE",
    "admit_read_content",
    "detail_view",
    "inbox_view",
    "read_inbox_page",
    "read_message_detail",
    "resolve_read_context",
    "wire_for",
]
