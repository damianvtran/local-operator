"""Explicit one-shot orchestrators for the H2 pull/ack primitives.

WHAT THIS IS NOT. No timers, no background poll, no model call, no session or
supervisor import, no approval decision, no automatic state progression. Every
function here is invoked explicitly by a caller (H3/H4 later) and performs AT
MOST ONE wire request.

THE FOUR LAWS THIS MODULE ENFORCES, from the H2 handoff:

1. **Exact owner AND destination.** Every dispatch takes the pinned
   ``OAuthAccess`` and resolves the committed ``OwnerSnapshot`` by its FULL
   namespace (hub base, account, tenant, endpoint) BEFORE the request, then
   checks the concrete wire client's own base against the committed hub. A
   mismatched or missing pinned access, or a client pointed elsewhere,
   refuses locally; there is no model-key fallback and no sibling credential
   search — ever.
2. **Generation fencing on both sides of the await.** The owner's committed
   generation is checked before dispatch and re-checked after the await;
   a completion that returns after an invalidate is refused locally and
   cannot populate the new run's mirror. An already-sent remote mutation is
   never "recalled": its refs-only intent stays under the OLD owner for
   explicit reconciliation (no fabricated distributed atomicity).
3. **Intent before dispatch.** An ack records its write-once intent (the
   EXACT screening tuple + client_ref) in the store in its own transaction
   BEFORE the HTTP call, so a crash between intent and dispatch replays the
   identical request instead of re-screening. A lost response is retried by
   :func:`flush_ack` with the SAME tuple — never a replacement.
4. **Ambiguity is not a verdict.** Only a definite HTTP 409 write-once
   refusal (or a validated success carrying a real delivery state) may end an
   intent; transport failures, 401/403/429/503 answers and malformed 200s
   keep the intent PENDING and replayable, however many times it must be
   flushed.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, cast
from urllib.parse import urlsplit

from local_operator.providers.auth_store import OAuthAccess
from local_operator.team_bridge.client import TeamBridgeClient
from local_operator.team_bridge.store import (
    ACK_CONFIRMED,
    ACK_CONFLICT,
    ACK_PENDING,
    Store,
    StoreRefused,
)
from local_operator.team_bridge.types import (
    DELIVERY_STATES,
    AckTarget,
    FeedSeq,
    InboxCursor,
    InboxPage,
    MessageRef,
    OwnerSnapshot,
    ScreeningReceipt,
    TeamBridgeTypeError,
    project_events_batch,
    project_inbox_page,
    project_message_ref,
)


class WireClient(Protocol):
    """The four wire methods the orchestrators use (tests inject fakes).

    A Protocol, not the concrete :class:`TeamBridgeClient`: H2's tests need a
    substitute that records calls, and a subclass would drag the credential
    machinery into every test.
    """

    #: The concrete dispatch destination. REQUIRED: the Q1 guard refuses a
    #: wire without it rather than silently skipping the destination check.
    base_url: str

    async def inbox(self, **kwargs: Any) -> dict[str, Any]: ...

    async def message(self, message_id: str, **kwargs: Any) -> dict[str, Any]: ...

    async def ack(self, message_id: str, **kwargs: Any) -> dict[str, Any]: ...

    async def events(self, **kwargs: Any) -> dict[str, Any] | None: ...


__all__ = [
    "AckOutcome",
    "PullRefused",
    "ack",
    "drain_feed_refs",
    "events",
    "finish_resync",
    "flush_ack",
    "inbox",
    "message",
    "recover_ack",
]

#: Server caps the inbox page at 100; 50 is the contract default. A caller
#: may go lower, never higher.
_MAX_PAGE = 100

#: The ACK route's KNOWN terminal 409 codes, derived from the frozen server
#: (internal/controllers/messaging_ack.go @ 24b0067a: "screening_required …
#: or invalid_transition (regression, illegal jump, a disagreeing repeat
#: verdict, or a sender acking)" — both are business refusals of THIS exact
#: tuple; replaying it cannot converge). Any OTHER 409 code is additive and
#: unproven: it must stay PENDING for explicit reconciliation, never be
#: promoted to a permanent conflict (QA R2 H2-Q4).
_KNOWN_TERMINAL_ACK_409 = ("invalid_transition", "screening_required")


class PullRefused(RuntimeError):
    """A local refusal. ``code`` is stable and machine-readable."""

    def __init__(self, code: str, message: str = "") -> None:
        super().__init__(message or code)
        self.code = code


@dataclass(frozen=True)
class AckOutcome:
    """One flush's result. ``state`` mirrors the store's intent lifecycle."""

    intent_id: str
    state: str  # pending | confirmed | conflict
    delivery_state: str | None
    error_code: str | None
    retry_after: float | None


def _guard_owner(store: Store, owner: OwnerSnapshot, access: OAuthAccess | None) -> None:
    """Refuse anything but the EXACT pinned identity of the committed owner.

    The access record the request carries must be the committed credential
    row, the same account, and an OAuth row; the owner is resolved by the
    FULL namespace (hub base, account, tenant, endpoint) through
    :meth:`Store.resolve_owner`, so a future multi-namespace store cannot
    dispatch through a credential-only guess (QA H2-Q5). ``None`` access —
    the model-routing cascade a generic caller would resolve — is refused
    here: H2 has no fallback path, by design.
    """

    if access is None:
        raise PullRefused("no_access", "a pinned OAuthAccess is required; no fallback exists")
    if access.account_id != owner.account_id:
        raise PullRefused("owner_mismatch", "pinned access account differs from owner")
    if access.kind != "oauth":
        raise PullRefused("not_oauth", "the bridge requires the OAuth credential row")
    committed = store.resolve_owner(
        credential_id=access.credential_id,
        account_id=owner.account_id,
        tenant_id=owner.tenant_id,
        endpoint_id=owner.endpoint_id,
        canonical_base_url=owner.canonical_base_url,
    )
    if committed is None:
        raise PullRefused(
            "not_enrolled",
            "no committed owner for this exact credential/hub/account/tenant/endpoint",
        )
    if committed != owner:
        raise PullRefused("owner_mismatch", "committed owner differs from the passed snapshot")


def _guard_destination(owner: OwnerSnapshot, wire: WireClient) -> None:
    """The CONCRETE dispatch destination must match the committed hub.

    ``owner.canonical_base_url`` was recorded at enrollment; the wire client's
    own base is TRUSTED CONFIGURATION (env/config), never server payload, and
    the shared origin is the trust boundary. An injected fake with no base is
    test configuration and exempt; a REAL client pointed at another hub is
    refused BEFORE any request (QA H2-Q1). When one side names no path the
    other's path is accepted; when both name paths they must agree.
    """

    wire_base = getattr(wire, "base_url", None)
    if not isinstance(wire_base, str) or not wire_base:
        # FAIL CLOSED (QA R2 nit): a wire that cannot name its destination has
        # no evidence to compare against the committed hub — it must not
        # dispatch on trust. The injected-test seam is explicit: fixtures
        # carry the owner's hub as their base_url like a real client does.
        raise PullRefused("no_destination_evidence", "wire client exposes no base_url")
    owner_url = urlsplit(owner.canonical_base_url)
    wire_url = urlsplit(wire_base)
    if (owner_url.scheme.lower(), owner_url.netloc.lower()) != (
        wire_url.scheme.lower(),
        wire_url.netloc.lower(),
    ):
        raise PullRefused("destination_mismatch", "wire base origin differs from the committed hub")
    owner_path = owner_url.path.rstrip("/")
    wire_path = wire_url.path.rstrip("/")
    if owner_path and wire_path and owner_path != wire_path:
        raise PullRefused("destination_mismatch", "wire base path differs from the committed hub")


def _generation_intact(store: Store, owner: OwnerSnapshot) -> bool:
    current = store.resolve_owner(
        credential_id=owner.credential_id,
        account_id=owner.account_id,
        tenant_id=owner.tenant_id,
        endpoint_id=owner.endpoint_id,
        canonical_base_url=owner.canonical_base_url,
    )
    return current is not None and current.enrollment_generation == owner.enrollment_generation


def _client_for(config_dir: Path, client: WireClient | None) -> WireClient:
    """The injected fake or a real client for this config dir."""

    if client is not None:
        return client
    # cast: TeamBridgeClient satisfies the protocol's call shapes, but its
    # concrete keyword-only signatures are not proven structure-matching.
    return cast(WireClient, TeamBridgeClient(config_dir))


def _project_or_refuse(page: Mapping[str, Any]) -> InboxPage:
    try:
        return project_inbox_page(page)
    except TeamBridgeTypeError as failure:
        # A malformed page is a REFUSAL, never a partial adoption: the caller
        # must not persist a checkpoint over an answer it could not parse.
        raise PullRefused("malformed_page", str(failure)) from failure


async def inbox(
    owner: OwnerSnapshot,
    *,
    config_dir: Path,
    access: OAuthAccess | None,
    after: InboxCursor | None = None,
    limit: int | None = None,
    state: str | None = None,
    thread_id: str | None = None,
    sender_account_id: str | None = None,
    client: WireClient | None = None,
) -> InboxPage:
    """ONE inbox page, projected and merged (refs + checkpoint atomically).

    The query names are the server's exact ones (§6.1), and ``tenant_id`` is
    the caller's committed namespace — never a per-item guess. The query itself
    is scoped by the committed tenant of the owner (:meth:`Store.resolve_owner`
    already proved it). ``state`` filters the DELIVERY state the caller asks
    about; nothing here derives admission from a page — merging refs is
    metadata, not delivery. The whole page must project cleanly and every item
    must bind to THIS owner before any row persists (QA H2-Q1/Q3); the store
    transaction makes the adoption atomic.
    """

    store = Store(config_dir)
    _guard_owner(store, owner, access)
    if limit is not None and (limit < 1 or limit > _MAX_PAGE):
        raise PullRefused("bad_limit", f"limit must be within 1..{_MAX_PAGE}")
    wire = _client_for(config_dir, client)
    _guard_destination(owner, wire)
    result = await wire.inbox(
        tenant_id=owner.tenant_id,
        after=None if after is None else str(after),
        limit=limit,
        state=state,
        thread_id=thread_id,
        sender_account_id=sender_account_id,
        access=access,
    )
    page = _project_or_refuse(result)
    if not _generation_intact(store, owner):
        raise PullRefused("stale_generation", "owner changed while the page was in flight")
    try:
        store.merge_page(owner, page.items, page.next_cursor)
    except StoreRefused as failure:
        # The store's own refusals (wrong_recipient / wrong_tenant) surface as
        # this module's refusal type, one spelling for callers. Nothing was
        # persisted: merge_page is one transaction and it rolled back.
        if failure.code in ("wrong_recipient", "wrong_tenant"):
            raise PullRefused(failure.code, str(failure)) from failure
        raise
    return page


async def message(
    owner: OwnerSnapshot,
    message_id: str,
    *,
    config_dir: Path,
    access: OAuthAccess | None,
    client: WireClient | None = None,
) -> MessageRef:
    """ONE message read: the same projection + fencing as a page, refs only.

    The single-message envelope is the SAME item type as an inbox row
    (bridge_read.go), so the identical binding rules apply: a foreign or
    unbound recipient refuses before persistence.
    """

    store = Store(config_dir)
    _guard_owner(store, owner, access)
    wire = _client_for(config_dir, client)
    _guard_destination(owner, wire)
    result = await wire.message(message_id, tenant_id=owner.tenant_id, access=access)
    try:
        ref = project_message_ref(result)
    except TeamBridgeTypeError as failure:
        raise PullRefused("malformed_message", str(failure)) from failure
    if ref.message_id != message_id:
        raise PullRefused("mismatched_id", "server answered about a different message")
    if ref.tenant_id != owner.tenant_id:
        raise PullRefused("wrong_tenant", "item tenant differs from owner")
    if not _generation_intact(store, owner):
        raise PullRefused("stale_generation", "owner changed while the read was in flight")
    try:
        store.merge_refs(owner, [ref])
    except StoreRefused as failure:
        # Same single spelling as inbox: the store's binding refusals become
        # PullRefused and nothing was persisted (one rolled-back transaction).
        if failure.code in ("wrong_recipient", "wrong_tenant"):
            raise PullRefused(failure.code, str(failure)) from failure
        raise
    return ref


async def ack(
    owner: OwnerSnapshot,
    message_id: str,
    target: AckTarget,
    receipt: ScreeningReceipt,
    *,
    config_dir: Path,
    access: OAuthAccess | None,
    client_ref: str | None = None,
    client: WireClient | None = None,
) -> AckOutcome:
    """Record the intent, then make ONE ack request with the EXACT tuple.

    The intent is durable BEFORE dispatch; :func:`flush_ack` maps outcomes
    without ever overwriting the write-once verdict: a definite 409 refusal
    marks the intent ``conflict`` (terminal, not re-sent); every ambiguous
    outcome — transport failure, sign-in refusal, 429, 503, malformed success
    — leaves it ``pending`` for an exact replay; a generation change after the
    await leaves it ``pending`` under the OLD owner.
    """

    store = Store(config_dir)
    _guard_owner(store, owner, access)
    ref = client_ref or str(uuid.uuid4())
    intent_id = store.record_ack_intent(owner, message_id, target, receipt, ref)
    return await flush_ack(owner, intent_id, config_dir=config_dir, access=access, client=client)


async def flush_ack(
    owner: OwnerSnapshot,
    intent_id: str,
    *,
    config_dir: Path,
    access: OAuthAccess | None,
    client: WireClient | None = None,
) -> AckOutcome:
    """Replay ONE pending intent with its exact stored tuple (crash recovery).

    There is deliberately no variant that re-screens or replaces the tuple: a
    different verdict is a NEW write-once conflict (the server keeps the first
    outcome), not a retry. Only a definite business refusal ends an intent;
    every ambiguity keeps it pending and replayable with the SAME tuple+ref.
    """

    store = Store(config_dir)
    _guard_owner(store, owner, access)
    intent = store.ack_intent(owner, intent_id)
    if intent is None:
        # Missing/corrupt intent: the verdict is NOT reconstructable from a
        # later read (the wire exposes only delivery state/time), so this is
        # a HOLD, never a fabricated tuple.
        raise PullRefused("no_intent", "ack intent is absent or unreadable; hold")
    if intent["state"] == ACK_CONFIRMED:
        return AckOutcome(intent_id, ACK_CONFIRMED, None, None, None)
    if intent["state"] == ACK_CONFLICT:
        return AckOutcome(intent_id, ACK_CONFLICT, None, intent.get("error_code"), None)
    receipt = intent["receipt"]
    if receipt is None:
        raise PullRefused("no_intent", "intent carries no screening tuple; hold")

    def _commit_result(
        *, state: str, error_code: str | None, retry_after: float | None = None
    ) -> None:
        """Record ONE outcome; a fence refusal is a STALE completion, surfaced.

        The store checks the caller's generation inside its own write
        transaction (QA R3 H2-Q5), so this translation is the only path a
        post-await invalidation can take — the intent stays pending, and the
        caller must reconcile under the new generation explicitly.
        """

        try:
            store.mark_ack_result(
                owner, intent_id, state=state, error_code=error_code, retry_after=retry_after
            )
        except StoreRefused as failure:
            if failure.code in ("stale_generation", "not_enrolled"):
                raise PullRefused(
                    "stale_generation", "owner changed before the ack result could commit"
                ) from failure
            raise

    wire = _client_for(config_dir, client)
    _guard_destination(owner, wire)
    try:
        result = await wire.ack(
            intent["message_id"],
            tenant_id=owner.tenant_id,
            state=intent["target"],
            screening=receipt,
            client_ref=intent["client_ref"],
            access=access,
        )
    except Exception as failure:  # noqa: BLE001 - classified below; never blindly re-sent
        code = getattr(failure, "code", None) or type(failure).__name__
        http_status = getattr(failure, "http_status", None)
        retry_after = getattr(failure, "retry_after_seconds", None)
        if not _generation_intact(store, owner):
            # Stale completion: the intent stays PENDING under the old owner
            # for explicit reconciliation. We do not mark it against the new
            # generation and we never claim the remote write did not happen.
            raise PullRefused(
                "stale_generation", "owner changed while the ack was in flight"
            ) from failure
        if http_status == 409 and str(code) in _KNOWN_TERMINAL_ACK_409:
            # The ONLY terminal arm: a KNOWN business refusal of this exact
            # tuple (disagreeing verdict, illegal move, wrong party) that
            # replaying can never converge. A 409 carrying a code this client
            # does not know is NOT proven terminal (QA R2 H2-Q4).
            _commit_result(state=ACK_CONFLICT, error_code=str(code))
            return AckOutcome(intent_id, ACK_CONFLICT, None, str(code), retry_after)
        # Everything else is AMBIGUITY, not a verdict: transport failure,
        # sign-in refusal, 429/503 — keep the exact pending intent replayable,
        # retaining the error code and retry hint.
        _commit_result(state=ACK_PENDING, error_code=str(code), retry_after=retry_after)
        return AckOutcome(intent_id, ACK_PENDING, None, str(code), retry_after)
    delivery = result.get("delivery") if isinstance(result, Mapping) else None
    delivery_state = (
        delivery.get("state")
        if isinstance(delivery, Mapping) and isinstance(delivery.get("state"), str)
        else None
    )
    if not _generation_intact(store, owner):
        raise PullRefused("stale_generation", "owner changed while the ack was in flight")
    if delivery_state != intent["target"]:
        # The frozen §7.2 machine answers a 200 for THIS ack only with the
        # target itself (the move, or a same-state no-op repeat; anything
        # else is a 409). An enum-valid but different state — e.g. `unread`
        # answering a requested `read` — is CONTRADICTORY: it cannot prove
        # the transition (QA R2 H2-Q4), and a missing state proves nothing
        # either. Both keep the intent replayable; no fabricated read/done.
        code = "malformed_ack" if delivery_state not in DELIVERY_STATES else "result_mismatch"
        _commit_result(state=ACK_PENDING, error_code=code)
        return AckOutcome(intent_id, ACK_PENDING, None, code, None)
    _commit_result(state=ACK_CONFIRMED, error_code=None)
    return AckOutcome(intent_id, ACK_CONFIRMED, delivery_state, None, None)


async def recover_ack(
    owner: OwnerSnapshot,
    message_id: str,
    target: AckTarget,
    *,
    config_dir: Path,
    access: OAuthAccess | None,
    client_ref: str | None = None,
    client: WireClient | None = None,
) -> AckOutcome:
    """Resume a PENDING intent whose id the caller never received (R5/Q7).

    The crash window this closes: ``record_ack_intent`` committed but the
    process died before ``ack`` returned the id, so a plain re-ack with the
    default auto-UUID ref would be refused as a write-once conflict and the
    row would be unreachable. Lookup is by the exact-owner namespace plus
    (message, target); the STORED tuple and STORED ref are replayed — there
    is no path that re-screens or mutates the immutable intent. A caller that
    names a DIFFERENT ref is refused: silently swapping the ref would make
    one intent carry two identities.
    """

    store = Store(config_dir)
    _guard_owner(store, owner, access)
    intent = store.find_pending_intent(owner, message_id, target)
    if intent is None:
        raise PullRefused("no_intent", "no pending intent for this message+target; hold")
    if client_ref is not None and client_ref != intent["client_ref"]:
        raise PullRefused(
            "ref_mismatch", "the stored intent's ref is immutable; a different ref is refused"
        )
    return await flush_ack(
        owner, intent["intent_id"], config_dir=config_dir, access=access, client=client
    )


async def events(
    owner: OwnerSnapshot,
    *,
    config_dir: Path,
    access: OAuthAccess | None,
    after: FeedSeq | None = None,
    wait: int | None = None,
    client: WireClient | None = None,
) -> dict[str, Any] | None:
    """ONE feed read. ``None`` = the server's 204 timeout (H4 owns lifetime).

    The frozen DTO answers ``cursor`` as a canonical decimal STRING and each
    event as a ``{type, ids, at}`` refs-only object; :func:`project_events_batch`
    normalizes both strictly and never stringifies unknown keys (QA H2-Q2).
    A 200 with an empty ``events`` list and a fresh cursor is the RESYNC
    signal (§6.7) — recorded as such through :func:`record_feed_batch`, not
    inferred here.
    """

    store = Store(config_dir)
    _guard_owner(store, owner, access)
    wire = _client_for(config_dir, client)
    _guard_destination(owner, wire)
    # ``wait`` is forwarded only when set, so a wire double (or an older
    # client) without the keyword keeps working for plain one-shot reads.
    extra: dict[str, Any] = {} if wait is None else {"wait": int(wait)}
    result = await wire.events(
        tenant_id=owner.tenant_id,
        after=None if after is None else int(after),
        access=access,
        **extra,
    )
    if result is None:
        return None
    try:
        cursor, refs, resync = project_events_batch(result)
    except TeamBridgeTypeError as failure:
        raise PullRefused("malformed_events", str(failure)) from failure
    if not _generation_intact(store, owner):
        raise PullRefused("stale_generation", "owner changed while the feed read was in flight")
    store.record_feed_batch(owner, cursor, refs, resync_required=resync)
    return {"cursor": int(cursor), "refs": list(refs), "resync": resync}


def drain_feed_refs(owner: OwnerSnapshot, *, config_dir: Path, upto_seq: int) -> list[str]:
    """Durable refs at/below ``upto_seq`` for the H4 drainer (no network)."""

    return Store(config_dir).drain_feed_refs(owner, upto_seq)


def finish_resync(owner: OwnerSnapshot, checkpoint: InboxCursor, *, config_dir: Path) -> None:
    """Mark the walk complete — REFUSED unless every page was merged.

    The walk itself is H4's (it drives the pagination loop and calls this);
    H2 owns the witness (:meth:`Store.finish_resync` refuses a resync whose
    last merged page was not terminal) so an early call cannot record an
    unfinished checkpoint as complete.
    """

    Store(config_dir).finish_resync(owner, checkpoint)
