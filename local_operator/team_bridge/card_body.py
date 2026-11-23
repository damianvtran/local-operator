"""Resolve the admitted body behind ONE inbound Team Bridge card, for display.

WHY THIS EXISTS. A journal card is metadata only, so every surface used to say
"Message body unavailable" even for a message the agent had read. This module
answers, for one card, what each of its messages is to THIS viewer: the body
the agent was given, still pending, held, refused, not admitted, or
unavailable (with a closed reason). It is the single resolver the desktop route
and the phone relay route both call, so the two can never disagree.

KEYED BY ENTRY, NOT BY MESSAGE ID. The caller names a transcript row it can
already read; the message ids come from that row's validated
``bridge_receipt_source``. Nobody can probe an arbitrary message id through
this path.

RESOLUTION ORDER (design §4), per message:

1. the local display store (``display_bodies``) for the live owner;
2. a local hold row -> ``held`` (coarse class + age only, never a code/reason);
3. a non-terminal body reservation -> ``pending``;
4. the hub's ADMITTED-ONLY history read (``conversation_messages`` with this
   endpoint), the same read ``channel_history`` makes. The hub authorises it
   (membership + per-(message, endpoint) admission) and a null body comes back
   with the hub's own ``body_state``.

WHAT THIS MUST NEVER CALL: ``conversation_body`` (``/messages/{id}/body``).
That is the screener's pre-admission classification fetch and serves UNSCREENED
bytes; showing it would bypass screening. A test pins that it is never reached.

``refused`` is only ever the hub's ``rejected``; this machine never derives a
moderation outcome (``tool._annotate_body_state`` has the same law).

LOCAL-ONLY MODE (``local_only=True``, design round 1 D1). The TUI resolves every
card as it MOUNTS so a collapsed header can show the excerpt or the state, and
a transcript can hold hundreds of cards: that pass must cost no network at all.
It answers steps 1-3 from local files under a local standing check (live
owner, same account, consent receipt, the owner's Radient row present) and
reports every remaining message as ``unavailable``/``not_loaded`` — "open to
load", never a guess. Opening the card runs the full resolver (step 4).

HUB DELETES (review R2). On a full resolve the hub read also covers messages
the local store answered, so a message deleted on the hub (an explicit deleted
marker, or a single-message card whose exact-seq read no longer returns it)
drops its local row and answers ``unavailable``/``deleted``. A hub that cannot
be reached never overrides a local row.

ACCOUNT FENCE. A card whose receipt names another account/tenant/endpoint than
the live owner answers ``other_account`` and reads nothing. Observing sign-out,
an account mismatch or the bridge turned off purges the display store
(revocation is live: it takes effect at the next view).
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "CardBodyNotFound",
    "ITEM_LIMIT",
    "resolve_card_body",
]

#: At most this many messages of a coalesced card are resolved per request; the
#: rest are counted in ``more`` (a coalesced card can name up to 200 ids).
ITEM_LIMIT = 20
#: The hub's history page ceiling for the backfill read.
_HUB_PAGE_MAX = 50
#: Transcript entry ids are host-minted; this bounds what reaches the reader.
ENTRY_ID = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")

_TRANSIENT_HOLD_CLASSES = frozenset({"screening_unavailable", "delivery_pending"})


class CardBodyNotFound(LookupError):
    """No readable Team Bridge card row by that entry id (the route's 404)."""


def _item(message_id: str, conversation_id: str, state: str, **extra: Any) -> dict[str, Any]:
    return {"message_id": message_id, "conversation_id": conversation_id, "state": state, **extra}


def _card_source(sessions_root: Path, session_id: str, entry_id: str) -> dict[str, Any] | None:
    """The validated receipt source of a Team Bridge card row, else None/raise."""

    from local_operator.harness.message_types import TEAM_BRIDGE_MESSAGE_TYPE
    from local_operator.session.transcript import read_entry_by_id

    from .display_receipt import projected_display_receipt

    if not ENTRY_ID.fullmatch(entry_id):
        raise CardBodyNotFound("entry")
    entry = read_entry_by_id(sessions_root / session_id, entry_id)
    if entry is None:
        raise CardBodyNotFound("entry")
    payload = entry.payload
    if payload.get("custom_type") != TEAM_BRIDGE_MESSAGE_TYPE:
        raise CardBodyNotFound("entry")
    details = payload.get("details")
    if not isinstance(details, dict):
        raise CardBodyNotFound("entry")
    # A pre-receipt card carries no typed locator; the answer is "nothing to
    # resolve", never a parse of its model-facing text.
    projected = projected_display_receipt(details)
    source = projected.get("bridge_receipt_source") if projected else None
    if not source:
        return None
    from .card_projection import _THREAD_ROOT_LINE

    # A thread reply is served by the replies read, not the top-level history
    # page, so its absence from that page proves nothing (see _hub_backfill).
    body = details.get("body")
    return {
        **source,
        "is_thread_reply": bool(isinstance(body, str) and _THREAD_ROOT_LINE.search(body)),
        "entry_ts": entry.ts,
    }


def _card_facts(source: dict[str, Any]) -> dict[str, Any]:
    """The answer's optional ``card`` object: facts of the card ROW itself.

    WHY. An open card shows when the message arrived and whether it is a thread
    reply next to the body, and the phone/desktop rows carry neither (the relay
    transcript row has no timestamp; the thread marker lives only in the
    model-facing carriage, which no client may render). Both come from the
    card's own journal row — no hub read, no body text — so they ride every
    answer for a found card, including local-only and every unavailable path,
    and cost nothing beyond the row read the resolver already makes.

    ``received_at`` is the journal entry's time (when this machine recorded the
    delivery), ISO-8601 UTC; it is omitted if the row's ``ts`` is unusable.
    Additive within ``v: 1``: an older client ignores the key.
    """
    from datetime import UTC, datetime

    facts: dict[str, Any] = {"thread_reply": bool(source.get("is_thread_reply"))}
    ts = source.get("entry_ts")
    if isinstance(ts, (int, float)) and not isinstance(ts, bool):
        try:
            stamp = datetime.fromtimestamp(float(ts), tz=UTC)
        except (OverflowError, OSError, ValueError):
            stamp = None
        if stamp is not None:
            facts["received_at"] = stamp.isoformat().replace("+00:00", "Z")
    return facts


def _hub_state(row: dict[str, Any]) -> str:
    from .tool import _hub_body_state  # the one reducer channel_history uses

    reduced = _hub_body_state(row.get("body_state"))
    state = reduced["state"] if reduced else "not_admitted"
    return {"rejected": "refused"}.get(state, state)


def _refusal_reason(failure: BaseException) -> str:
    """Map a hub failure to the closed reason vocabulary; never its text."""

    from .client import BridgeRefusal, BridgeSignInRequired

    if isinstance(failure, BridgeSignInRequired):
        return "signed_out"
    if isinstance(failure, BridgeRefusal):
        if failure.http_status in (403, 404):
            return "not_member"
        if failure.http_status in (405, 501):
            return "hub_unsupported"
    return "offline"


async def resolve_card_body(
    config_dir: Path,
    session_id: str,
    entry_id: str,
    *,
    sessions_root: Path | None = None,
    wire: Any = None,
    local_only: bool = False,
) -> dict[str, Any]:
    """The ``{v:1, items:[...], more:n, card:{...}}`` answer for one card (design §4).

    ``card`` (additive, optional for readers): ``{received_at?: ISO-8601 UTC,
    thread_reply: bool}`` from the card's own journal row (:func:`_card_facts`);
    absent only on the nothing-to-resolve answer for a pre-receipt card.

    Raises :class:`CardBodyNotFound` for an entry that is not a Team Bridge card
    in that session. ``wire`` is the test seam for the hub client.
    ``local_only`` reads local files only (see LOCAL-ONLY MODE above).
    """

    from .display_bodies import DisplayBodyStore
    from .store import Store, StoreRefused, _owner_key

    config_dir = Path(config_dir)
    root = Path(sessions_root) if sessions_root is not None else config_dir / "sessions"
    source = await asyncio.to_thread(_card_source, root, session_id, entry_id)
    if not source:
        return {"v": 1, "items": [], "more": 0}
    conversation_id = str(source["conversation_id"])
    all_ids = [str(mid) for mid in source["message_ids"]]
    ids = all_ids[:ITEM_LIMIT]
    more = len(all_ids) - len(ids)

    card = _card_facts(source)

    def answer(items: list[dict[str, Any]]) -> dict[str, Any]:
        return {"v": 1, "items": items, "more": more, "card": dict(card)}

    def every(state: str, **extra: Any) -> dict[str, Any]:
        return answer([_item(mid, conversation_id, state, **extra) for mid in ids])

    store = Store(config_dir)
    display = DisplayBodyStore(config_dir)
    try:
        owner = await asyncio.to_thread(store.active_owner_readonly)
    except StoreRefused:
        owner = None
    if owner is None:
        return every("unavailable", reason="signed_out")
    if any(
        source[key] != getattr(owner, key) for key in ("account_id", "tenant_id", "endpoint_id")
    ):
        # The card was received under another owner. Its content is not this
        # viewer's to see, and probing the hub for it would be a cross-account read.
        return every("unavailable", reason="other_account")
    key = _owner_key(owner)

    if local_only:
        return answer(
            await asyncio.to_thread(_local_items, config_dir, owner, key, ids, conversation_id)
        )

    # Standing first: a signed-out or switched viewer must not be served even
    # the local copy (revocation is live). An unreachable hub is not a
    # revocation, so local rows still serve while the hub items read offline.
    from .read_context import (
        HistoryReadContext,
        ReadRefused,
        resolve_read_context,
        wire_for,
    )

    client = wire if wire is not None else wire_for(config_dir)
    context = None
    standing_reason = "offline"
    try:
        context = await resolve_read_context(
            config_dir, expected_account_id=str(owner.account_id), wire=client
        )
    except ReadRefused as refused:
        if refused.code == "account_mismatch":
            await asyncio.to_thread(display.purge)
            return every("unavailable", reason="other_account")
        if refused.code in ("consent_revoked", "owner_changed"):
            # Revocation: the retained copies are no longer this viewer's.
            await asyncio.to_thread(display.purge)
            return every("unavailable", reason="signed_out")
        if refused.code != "state_unreadable":
            # not_signed_in / history_context_unavailable / credential_changed:
            # standing cannot be proven, so nothing (local or hub) is served.
            return every("unavailable", reason="signed_out")
        standing_reason = "offline"
    except Exception as failure:  # noqa: BLE001 — closed reasons only, never text
        reason = _refusal_reason(failure)
        if reason == "signed_out":
            await asyncio.to_thread(display.purge)
            return every("unavailable", reason="signed_out")
        standing_reason = reason
    if isinstance(context, HistoryReadContext):
        # The bridge is turned off (opt-out): retained local copies go, and the
        # hub backfill below serves display only, writing nothing back.
        await asyncio.to_thread(display.purge)

    local = await asyncio.to_thread(display.bodies, key, ids)
    resolved: dict[str, dict[str, Any]] = {}
    for mid in ids:
        row = local.get(mid)
        if row is not None and row["conversation_id"] == conversation_id:
            resolved[mid] = _item(
                mid, conversation_id, "admitted", body=row["body"], source="local"
            )
    misses = [mid for mid in ids if mid not in resolved]

    if misses:
        try:
            holds = await asyncio.to_thread(store.hold_states_readonly, owner, misses)
        except (StoreRefused, OSError):
            holds = {}
        try:
            receiver = await asyncio.to_thread(store.receiver_state_readonly, owner, misses)
        except (StoreRefused, OSError):
            receiver = None
        pending = receiver.get("pending_admission", frozenset()) if receiver else frozenset()
        now = time.time()
        for mid in misses:
            hold = holds.get(mid)
            if hold is not None:
                resolved[mid] = _item(mid, conversation_id, "held", hold=_hold_view(hold, now))
            elif mid in pending:
                resolved[mid] = _item(mid, conversation_id, "pending")
        misses = [mid for mid in ids if mid not in resolved]

    if misses and context is None:
        for mid in misses:
            resolved[mid] = _item(mid, conversation_id, "unavailable", reason=standing_reason)
        misses = []

    local_hits = [mid for mid in ids if resolved.get(mid, {}).get("source") == "local"]
    if context is not None and (misses or local_hits):
        hub = await _hub_backfill(
            client,
            context,
            display,
            key,
            conversation_id,
            misses,
            verify=local_hits,
            # R1: the exact single-row read is right only when the card IS one
            # message; a coalesced card's seq is its NEWEST message, so one
            # local hit plus an older miss must read the whole page.
            single=len(all_ids) == 1,
            absence_is_deletion=len(all_ids) == 1 and not source.get("is_thread_reply"),
            newest_seq=int(source["conversation_seq"]),
            write=not isinstance(context, HistoryReadContext),
            config_dir=config_dir,
        )
        resolved.update(hub)
    return answer([resolved[mid] for mid in ids])


def _local_items(
    config_dir: Path, owner: Any, key: str, ids: list[str], conversation_id: str
) -> list[dict[str, Any]]:
    """Steps 1-3 from local files only, under a LOCAL standing check (D1).

    The standing that the full path proves with ``current_access`` is checked
    here from what is on disk: the consent receipt for the live owner and the
    owner's own Radient row. Either missing reads ``signed_out`` — the same
    answer the full path gives — and nothing local is served. Bridge off with
    rows still present purges them (the opt-out hooks normally already did).
    """

    from .display_bodies import (
        DisplayBodyStore,
        purge_if_bridge_off,
        radient_sign_in_state,
    )
    from .store import Store, StoreRefused

    def every(reason: str) -> list[dict[str, Any]]:
        return [_item(mid, conversation_id, "unavailable", reason=reason) for mid in ids]

    store = Store(config_dir)
    try:
        consent = store.owner_consent_receipt_readonly(owner)
    except (StoreRefused, OSError):
        # Unreadable is not "signed out": the open runs the full proof.
        return every("not_loaded")
    signed_in = radient_sign_in_state(config_dir, getattr(owner, "credential_id", None))
    if signed_in is None:
        return every("not_loaded")
    if consent is None or not signed_in:
        return every("signed_out")
    purge_if_bridge_off(config_dir)
    local = DisplayBodyStore(config_dir).bodies(key, ids)
    out: dict[str, dict[str, Any]] = {}
    for mid in ids:
        row = local.get(mid)
        if row is not None and row["conversation_id"] == conversation_id:
            out[mid] = _item(mid, conversation_id, "admitted", body=row["body"], source="local")
    misses = [mid for mid in ids if mid not in out]
    if misses:
        try:
            holds = store.hold_states_readonly(owner, misses)
        except (StoreRefused, OSError):
            holds = {}
        try:
            receiver = store.receiver_state_readonly(owner, misses)
        except (StoreRefused, OSError):
            receiver = None
        pending = receiver.get("pending_admission", frozenset()) if receiver else frozenset()
        now = time.time()
        for mid in misses:
            hold = holds.get(mid)
            if hold is not None:
                out[mid] = _item(mid, conversation_id, "held", hold=_hold_view(hold, now))
            elif mid in pending:
                out[mid] = _item(mid, conversation_id, "pending")
            else:
                # Not known locally: the hub might serve it, but only an OPEN
                # may ask (no network on mount).
                out[mid] = _item(mid, conversation_id, "unavailable", reason="not_loaded")
    return [out[mid] for mid in ids]


def _hold_view(hold: dict[str, Any], now: float) -> dict[str, Any]:
    """The coarse hold shape both paths serve: class + age, never a code."""

    held: dict[str, Any] = {
        "class": (
            "transient" if hold.get("hold_class") in _TRANSIENT_HOLD_CLASSES else "awaiting_review"
        )
    }
    created = hold.get("created_at")
    if isinstance(created, (int, float)):
        held["held_for_s"] = max(0, int(now - created))
    return held


def _hub_deleted(row: dict[str, Any]) -> bool:
    """An explicit hub deletion marker on a history row (forward-compatible)."""

    state = row.get("body_state")
    return (
        row.get("deleted") is True
        or bool(row.get("deleted_at"))
        or (isinstance(state, dict) and state.get("state") == "deleted")
    )


async def _hub_backfill(
    client: Any,
    context: Any,
    display: Any,
    owner_key: str,
    conversation_id: str,
    misses: list[str],
    *,
    verify: list[str] = (),  # type: ignore[assignment]
    single: bool,
    absence_is_deletion: bool = False,
    newest_seq: int,
    write: bool,
    config_dir: Path,
) -> dict[str, dict[str, Any]]:
    """One admitted-only history page ending at the card's newest seq.

    A single-message card's seq is exact, so ``limit=1`` reads precisely that
    row. A coalesced card only knows its NEWEST seq and other members' messages
    may interleave inside the window, so it reads a full page (the hub's 50
    cap) and filters to the card's own ids; an id outside the page reads
    ``unavailable`` rather than guessing.

    ``verify`` are ids the LOCAL store already answered. They are only checked
    for a hub deletion (R2) and answer only when deleted — then their local row
    is dropped; any failure or other answer leaves the local body standing.

    ABSENCE AS DELETION is deliberately narrow (``absence_is_deletion``): a
    single-message, top-level card whose exact read (``before_seq=seq+1,
    limit=1``) came back with an OLDER row, i.e. the hub walked past the
    message's position and it was not there. An empty page, a coalesced window
    or a thread reply (served by the replies read, not this page) proves
    nothing, and deleting the operator's copy on a guess is the worse error.
    """

    from . import pull
    from .read_context import ReadRefused, _recheck
    from .store import Store

    out: dict[str, dict[str, Any]] = {}

    def all_unavailable(reason: str) -> dict[str, dict[str, Any]]:
        return {mid: _item(mid, conversation_id, "unavailable", reason=reason) for mid in misses}

    owner = context.owner
    try:
        pull._guard_destination(owner, client)  # noqa: SLF001 — the shared guard
    except pull.PullRefused:
        return all_unavailable("offline")
    try:
        page = await client.conversation_messages(
            conversation_id,
            tenant_id=owner.tenant_id,
            endpoint_id=owner.endpoint_id,
            before_seq=newest_seq + 1,
            limit=1 if single else _HUB_PAGE_MAX,
            access=context.access,
        )
    except Exception as failure:  # noqa: BLE001 — closed reasons only
        return all_unavailable(_refusal_reason(failure))
    try:
        # Every await is a boundary where the owner can move; never answer for
        # a namespace that changed while the request was in flight.
        _recheck(context, Store(config_dir))
    except ReadRefused:
        return all_unavailable("signed_out")
    rows = page.get("messages") if isinstance(page, dict) else None
    by_id = {
        row["id"]: row
        for row in (rows if isinstance(rows, list) else [])
        if isinstance(row, dict) and isinstance(row.get("id"), str)
    }
    skipped_past = absence_is_deletion and any(
        isinstance(row.get("seq"), int) and row["seq"] < newest_seq for row in by_id.values()
    )
    deleted: list[str] = []
    for mid in [*misses, *verify]:
        row = by_id.get(mid)
        if row is not None and row.get("conversation_id", conversation_id) != conversation_id:
            row = None
        if (row is None and skipped_past) or (row is not None and _hub_deleted(row)):
            deleted.append(mid)
            out[mid] = _item(mid, conversation_id, "unavailable", reason="deleted")
    if deleted:
        # The hub no longer serves it: neither may the local copy (design §3
        # "delete on hub deleted").
        await asyncio.to_thread(display.delete, owner_key, deleted)
    for mid in misses:
        if mid in out:
            continue
        row = by_id.get(mid)
        if row is None or row.get("conversation_id", conversation_id) != conversation_id:
            out[mid] = _item(mid, conversation_id, "unavailable")
            continue
        body = row.get("body")
        digest = row.get("body_hash")
        if isinstance(body, str) and isinstance(digest, str):
            if hashlib.sha256(body.encode("utf-8")).hexdigest() != digest:
                # Bytes that do not match their own committed digest are not
                # the admitted message; never display them.
                out[mid] = _item(mid, conversation_id, "unavailable")
                continue
            if write:
                raw_sender = row.get("sender")
                sender = raw_sender if isinstance(raw_sender, dict) else {}
                await asyncio.to_thread(
                    display.record,
                    owner_key,
                    message_id=mid,
                    conversation_id=conversation_id,
                    body=body,
                    body_sha256=digest,
                    sender_kind=str(sender.get("kind") or "unknown"),
                    source="hub",
                )
            out[mid] = _item(mid, conversation_id, "admitted", body=body, source="hub")
            continue
        state = _hub_state(row)
        out[mid] = _item(mid, conversation_id, state)
    return out
