"""Explicit one-shot orchestrators for the H2 pull/ack primitives.

WHAT THIS IS NOT. No timers, no background poll, no model call, no session or
supervisor import, no approval decision, no automatic state progression. Every
function here is invoked explicitly by a caller (H3/H4 later) and performs AT
MOST ONE wire request.

THE THREE LAWS THIS MODULE ENFORCES, from the H2 handoff:

1. **Exact owner.** Every dispatch takes the pinned ``OAuthAccess`` and
   compares it to the committed ``OwnerSnapshot`` (credential row, account,
   tenant, endpoint) BEFORE the request. A mismatched or missing pinned
   access refuses locally; there is no model-key fallback and no sibling
   credential search — ever.
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
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, cast

from local_operator.providers.auth_store import OAuthAccess
from local_operator.team_bridge.client import TeamBridgeClient
from local_operator.team_bridge.store import ACK_CONFIRMED, ACK_CONFLICT, Store
from local_operator.team_bridge.types import (
    AckTarget,
    FeedSeq,
    InboxCursor,
    InboxPage,
    MessageRef,
    OwnerSnapshot,
    ScreeningReceipt,
    TeamBridgeTypeError,
    feed_seq,
    project_inbox_page,
    project_message_ref,
)


class WireClient(Protocol):
    """The four wire methods the orchestrators use (tests inject fakes).

    A Protocol, not the concrete :class:`TeamBridgeClient`: H2's tests need a
    substitute that records calls, and a subclass would drag the credential
    machinery into every test.
    """

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
]

#: Server caps the inbox page at 100; 50 is the contract default. A caller
#: may go lower, never higher.
_MAX_PAGE = 100


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

    This is the host-account bound from the handoff: the access record the
    request will carry must be the committed credential row AND the same
    account/tenant the namespace names. ``None`` access (the model-routing
    cascade a generic caller would resolve) is refused here — H2 has no
    fallback path, by design.
    """

    if access is None:
        raise PullRefused("no_access", "a pinned OAuthAccess is required; no fallback exists")
    committed = store.active_owner(credential_id=access.credential_id)
    if committed is None:
        raise PullRefused("not_enrolled", "no committed owner for this credential row")
    if committed != owner:
        raise PullRefused("owner_mismatch", "committed owner differs from the passed snapshot")
    if access.account_id != owner.account_id:
        raise PullRefused("owner_mismatch", "pinned access account differs from owner")
    if access.kind != "oauth":
        raise PullRefused("not_oauth", "the bridge requires the OAuth credential row")


def _generation_intact(store: Store, owner: OwnerSnapshot) -> bool:
    current = store.active_owner(credential_id=owner.credential_id)
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

    The query names are the server's exact ones (§6.1). ``state`` filters the
    DELIVERY state the caller asks about; nothing here derives admission from
    a page — merging refs is metadata, not delivery.
    """

    store = Store(config_dir)
    _guard_owner(store, owner, access)
    if limit is not None and (limit < 1 or limit > _MAX_PAGE):
        raise PullRefused("bad_limit", f"limit must be within 1..{_MAX_PAGE}")
    wire = _client_for(config_dir, client)
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
    for ref in page.items:
        if ref.recipient_endpoint_id and ref.recipient_endpoint_id != owner.endpoint_id:
            raise PullRefused("wrong_endpoint", "item is addressed to a different endpoint")
    if not _generation_intact(store, owner):
        raise PullRefused("stale_generation", "owner changed while the page was in flight")
    store.merge_page(owner, page.items, page.next_cursor)
    return page


async def message(
    owner: OwnerSnapshot,
    message_id: str,
    *,
    config_dir: Path,
    access: OAuthAccess | None,
    client: WireClient | None = None,
) -> MessageRef:
    """ONE message read: the same projection + fencing as a page, refs only."""

    store = Store(config_dir)
    _guard_owner(store, owner, access)
    wire = _client_for(config_dir, client)
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
    store.merge_refs(owner, [ref])
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

    Refusals map to the intent's lifecycle without ever overwriting a
    write-once verdict: a 409-shaped refusal marks the intent ``conflict``
    (code retained) and is NOT re-sent; a generation change after the await
    leaves the intent ``pending`` for explicit reconciliation under the OLD
    owner.
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
    outcome), not a retry.
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
    wire = _client_for(config_dir, client)
    try:
        result = await wire.ack(
            intent["message_id"],
            tenant_id=owner.tenant_id,
            state=intent["target"],
            screening=receipt,
            client_ref=intent["client_ref"],
            access=access,
        )
    except Exception as failure:  # noqa: BLE001 - mapped below; never re-sent blindly
        code = getattr(failure, "code", None) or type(failure).__name__
        retry_after = getattr(failure, "retry_after_seconds", None)
        if not _generation_intact(store, owner):
            # Stale completion: the intent stays PENDING under the old owner
            # for explicit reconciliation. We do not mark it against the new
            # generation and we never claim the remote write did not happen.
            raise PullRefused(
                "stale_generation", "owner changed while the ack was in flight"
            ) from failure
        store.mark_ack_result(owner, intent_id, state=ACK_CONFLICT, error_code=str(code))
        return AckOutcome(intent_id, ACK_CONFLICT, None, str(code), retry_after)
    delivery = result.get("delivery") if isinstance(result, Mapping) else None
    delivery_state = (
        delivery.get("state")
        if isinstance(delivery, Mapping) and isinstance(delivery.get("state"), str)
        else None
    )
    if not _generation_intact(store, owner):
        raise PullRefused("stale_generation", "owner changed while the ack was in flight")
    store.mark_ack_result(owner, intent_id, state=ACK_CONFIRMED, error_code=None)
    return AckOutcome(intent_id, ACK_CONFIRMED, delivery_state, None, None)


async def events(
    owner: OwnerSnapshot,
    *,
    config_dir: Path,
    access: OAuthAccess | None,
    after: FeedSeq | None = None,
    client: WireClient | None = None,
) -> dict[str, Any] | None:
    """ONE feed read. ``None`` = the server's 204 timeout (H4 owns lifetime).

    A 200 with an empty ``events`` list and a fresh cursor is the RESYNC
    signal (§6.7) — recorded as such through :func:`record_feed_batch`, not
    inferred here.
    """

    store = Store(config_dir)
    _guard_owner(store, owner, access)
    wire = _client_for(config_dir, client)
    result = await wire.events(
        tenant_id=owner.tenant_id, after=None if after is None else int(after), access=access
    )
    if result is None:
        return None
    refs = result.get("events")
    cursor_raw = result.get("cursor")
    refs_list = [str(ref) for ref in refs] if isinstance(refs, list) else []
    resync = isinstance(refs, list) and not refs_list
    cursor = (
        feed_seq(cursor_raw)
        if isinstance(cursor_raw, int) and not isinstance(cursor_raw, bool)
        else None
    )
    if cursor is None:
        raise PullRefused("malformed_events", "feed answer carried no integer cursor")
    if not _generation_intact(store, owner):
        raise PullRefused("stale_generation", "owner changed while the feed read was in flight")
    store.record_feed_batch(owner, cursor, refs_list, resync_required=resync)
    return {"cursor": int(cursor), "refs": refs_list, "resync": resync}


def drain_feed_refs(owner: OwnerSnapshot, *, config_dir: Path, upto_seq: int) -> list[str]:
    """Durable refs at/below ``upto_seq`` for the H4 drainer (no network)."""

    return Store(config_dir).drain_feed_refs(owner, upto_seq)


def finish_resync(owner: OwnerSnapshot, checkpoint: InboxCursor, *, config_dir: Path) -> None:
    """Mark the walk complete ONLY when every page was merged (caller's law)."""

    Store(config_dir).finish_resync(owner, checkpoint)
