"""Lend a login's renewal token to ONE node; the owner claims before it renews (B1).

THE DECISION (Damian, 2026-11-13, C5): model logins whose access token is short
(Anthropic ~8 h, Radient ~55 min) are lent to a node WITH their renewal token —
"Lend the node the renewal token too" — so node sessions survive a Mac that is away
for longer than one access token. Design: ``docs/design/mesh-credential-renewal.md``
§B and the 2026-11-14 decisions in §D (frozen at ``4509988f``).

WHAT B1 IS, AND WHY IT IS SAFE ALONE. This module holds the OWNER side of the
protocol plus the node's STORE-ONLY side:

* **lend** (R1): an eligible node's ``refresh_take`` under the standing
  ``device`` authority (no per-lend operator signature) writes a ``lent`` record on
  the owner's row and answers the current pair, a per-lend fingerprint salt and the
  family epoch;
* **claim** (R3): before the owner presents a lent login's refresh token it asks the
  node "are you renewing?" (``renewal_claim``); a ``standby`` answer authorises ONE
  owner POST for :data:`CLAIM_FRESH_S`, after which the owner pushes the new pair
  (``renewal_push``);
* **write-back** (§B.4): a node's ``refresh_return{act: done}`` is adopted with a
  compare-and-set on the fingerprint of the token it started from, under the
  conflict rule (the owner's own successful rotation wins);
* **awaiting handback** (R4): a claim nobody answers within :data:`CLAIM_WAIT_S`
  makes the owner WAIT rather than present a token the node may have rotated, and
  says so by naming the node.

The node in B1 never renews: its custody row carries a blocking delegation marker
(:func:`node_custody_marker`), so the ordinary refresh path refuses to present it.
And no build advertises :data:`RENEWAL_CAP_STRING` yet (it is deliberately NOT in
``wire.LINK_CAPABILITIES``): a node that cannot renew would only make the Mac wait
under R4 for nothing, so nothing is lent in production until B2 lands the node's
renewal gate and adds the advertisement. Tests advertise it explicitly.

THE TOKEN NEVER LEAVES THESE FUNCTIONS EXCEPT INSIDE A LINK RECORD. Fingerprints are
HMAC-SHA256 under a random per-lend salt, so neither side compares, stores or logs a
raw token to tell which pair a write-back forked from; audit events carry ids and
counts only (``audit.FORBIDDEN_DETAIL_KEYS`` refuses token fields by name).

Stdlib only at import, like the rest of the credentials package.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import secrets
import threading
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: Advertised by a build whose NODE can renew a lent login (B2). Gated on, never
#: inferred from an answer: an owner only lends to a link that negotiated it, and an
#: owner only sends a claim over one. Not in ``wire.LINK_CAPABILITIES`` in B1 — see
#: the module docstring.
RENEWAL_CAP_STRING = "credential-renewal-v1"

#: The providers whose renewal token is lent (§B.1). Both rotate their refresh token
#: and are treated as revoking the whole family on reuse (§D D2/B0). OpenAI's
#: ~10-day access token is already covered by the 24 h hold; kimi is device-bound.
RENEWABLE_PROVIDERS: frozenset[str] = frozenset({"anthropic", "radient"})

#: The owner-row record state of an automatic (device-authority) lend. NOT one of
#: ``auth_store._REFRESH_DELEGATION_LIVE_STATES``: a lent login does not block the
#: owner's own refresh — the owner claims first instead (R3).
LENT_STATE = "lent"

AUTHORITY_DEVICE = "device"
AUTHORITY_SIGNED = "signed"

#: ``W`` (R4): how long the owner's relay waits for a claim answer. The redial
#: backoff cap ``owner_wait_s`` reasons from; to be re-sized from the B0
#: wake-to-link-up measurement.
CLAIM_WAIT_S = 60.0

#: How long a ``standby`` answer authorises the owner's ONE POST. Long enough for
#: the session that asked to reach its POST; short enough that a node which later
#: lost its link cannot have reached its own lease (``L`` = 900 s) in between.
CLAIM_FRESH_S = 120.0

#: How long a REQUEST PATH waits for a claim before it moves on (manager ruling,
#: 2026-11-14: "the user-facing request path must never sit on it"). The claim keeps
#: running in the background up to :data:`CLAIM_WAIT_S`; the turn says "checking
#: with node2" and continues on its failover chain.
CLAIM_INLINE_S = 1.5

#: How often one process may start a background claim for the same login.
CLAIM_RETRY_S = 5.0

#: The return-only cap (§D D4): the longest a node may hold a renewed pair it has
#: not written back. Also when an unanswered owner wait clears on its own (R4).
RETURN_CAP_S = 7 * 86400.0

#: How many adopted write-back fingerprints a lent record keeps for replay
#: detection (§B.4 case 4). Bounded: a replay older than this many adoptions reads
#: as a conflict, which adopts nothing either — only the receipt's name differs.
ADOPTED_FPS_MAX = 8

#: The renewal counter's WIRE names. NEVER ``epoch``: the relay reads a top-level
#: ``epoch`` on every peer frame as the NETWORK epoch and refuses a higher one as
#: "the network has rotated" — so a renewal counter spelled ``epoch`` worked only
#: while it happened to sit at or below the network's (found by the write-back cell
#: at renewal epoch 2 on a network at epoch 1). Answers use the same names, so one
#: spelling covers the whole exchange. Stored records keep ``epoch``: they never
#: cross the wire as a frame.
WIRE_EPOCH = "renewal_epoch"
WIRE_PARENT_EPOCH = "parent_renewal_epoch"


# ---------------------------------------------------------------------------
# Settings (config readers; no /settings row until B2 makes them observable)
# ---------------------------------------------------------------------------


def lend_renewal_enabled(root: Path | None = None) -> bool:
    """``network.credentials.lend_renewal`` (default ON: the operator's standing C5
    decision of 2026-11-13). Off means no new lends. A garbled value reads as off:
    a typo must not lend a renewal token."""
    from local_operator.network import store

    raw = store.read_config(("network", "credentials", "lend_renewal"), True, root)
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        return raw.strip().lower() in ("1", "true", "yes", "on")
    return False


def renewal_authority(root: Path | None = None) -> str:
    """``network.credentials.renewal_authority``: ``device`` (default) or ``signed``.

    ``signed`` keeps slice 1's per-delegation operator signature as the only way to
    lend renewal (``mint_refresh_delegation``); automatic lends are then refused. An
    unknown value reads as ``signed`` — the stricter mode.
    """
    from local_operator.network import store

    raw = str(store.read_config(("network", "credentials", "renewal_authority"), "device", root))
    return AUTHORITY_DEVICE if raw.strip().lower() == AUTHORITY_DEVICE else AUTHORITY_SIGNED


# ---------------------------------------------------------------------------
# Fingerprints
# ---------------------------------------------------------------------------


def new_salt() -> str:
    return secrets.token_hex(16)


def fingerprint(salt: str, token: Any) -> str:
    """A keyed fingerprint of ``token`` under ``salt``; ``""`` for no token.

    Keyed so the value is useless as a lookup for the token and safe to store in the
    lend record and compare on both sides. Never logged.
    """
    if not isinstance(token, str) or not token or not salt:
        return ""
    try:
        key = bytes.fromhex(salt)
    except ValueError:
        return ""
    return hmac.new(key, token.encode("utf-8"), hashlib.sha256).hexdigest()[:32]


# ---------------------------------------------------------------------------
# R1 — eligibility
# ---------------------------------------------------------------------------


def lend_refusal(
    *,
    provider: str,
    native_incarnation: str | None,
    holder_scope: str,
    link_capabilities: Any,
    trusted: bool,
    root: Path | None,
) -> str:
    """Why ``provider`` may NOT be lent with its renewal token to this caller, or ``""``.

    A mismatch is not an error (§B.3 R1): the node simply keeps the access-only hold.
    The reason is an id for the audit trail, never shown as a failure.
    """
    if provider not in RENEWABLE_PROVIDERS:
        return "provider_not_renewable"
    if not lend_renewal_enabled(root):
        return "lend_renewal_off"
    if renewal_authority(root) != AUTHORITY_DEVICE:
        return "authority_signed"
    if not native_incarnation:
        # D5: only a lop-native sign-in. An imported or externally shared row may be
        # renewed by another tool too, and two renewers end a rotating family.
        return "not_native_login"
    if holder_scope != "device":
        return "not_device_scoped"
    if RENEWAL_CAP_STRING not in tuple(link_capabilities or ()):
        return "node_cannot_renew"
    if not trusted:
        return "network_untrusted"
    return ""


def _network_trusted(network_id: str, root: Path | None) -> bool:
    try:
        from local_operator.network import store

        return str(store.load(network_id, root).trust or "active") == "active"
    except Exception:  # noqa: BLE001 — cannot tell = not trusted (fail closed)
        return False


def lent_record(data: dict[str, Any]) -> dict[str, Any] | None:
    """The owner row's ``lent`` record, or ``None``."""
    from local_operator.providers.auth_store import REFRESH_DELEGATE_KEY

    record = data.get(REFRESH_DELEGATE_KEY) if isinstance(data, dict) else None
    if isinstance(record, dict) and record.get("state") == LENT_STATE:
        return record
    return None


# ---------------------------------------------------------------------------
# Copy (the f-string pattern of the readiness/status surfaces; see the PR's i18n note)
# ---------------------------------------------------------------------------


def awaiting_handback_message(provider: str, node_label: str) -> str:
    """R4, visible (§D D2): the owner waits, names the node, and offers the way out."""
    return (
        f"Waiting for {node_label} to hand back your {provider} login. It comes back as "
        f"soon as {node_label} reconnects. Or sign in again here (/login {provider})."
    )


def checking_message(provider: str, node_label: str) -> str:
    """The request path's sentence while a claim runs in the background."""
    return (
        f"Checking with {node_label} before renewing your {provider} login "
        f"(it is lent to {node_label}); this request continues without waiting."
    )


def returned_notice(node_label: str, provider: str, stamps: list[int]) -> str:
    """The adoption notice (§B.7): what the node did while the Mac was away."""
    times = ", ".join(time.strftime("%H:%M", time.gmtime(ms / 1000.0)) for ms in stamps[-5:])
    count = len(stamps)
    if not count:
        return f"{node_label} handed back your {provider} login."
    plural = "time" if count == 1 else "times"
    return (
        f"{node_label} renewed your {provider} login {count} {plural} while this Mac was "
        f"away (at {times} UTC)."
    )


# ---------------------------------------------------------------------------
# Owner — lend (``refresh_take`` under the device authority)
# ---------------------------------------------------------------------------


def owner_take(broker: Any, link: Any, key: str, by: str) -> dict[str, Any] | None:
    """Lend ``key``'s renewal token to ``by``, or ``None`` to leave the frame to slice 1.

    ``None`` means "not an automatic lend": the row carries a signed (slice-1)
    record, so the existing signed handler answers it unchanged. Any refusal here is
    a broker refusal reply.
    """
    from local_operator.providers.auth_store import REFRESH_DELEGATE_KEY

    entry = broker._entry(key)  # noqa: SLF001
    provider = str((entry.provider if entry else "") or key)
    if entry is None or entry.owner_device != broker.self_device:
        return broker._refuse(  # noqa: SLF001
            link, key, provider, by, "not_owner", "this device does not hold this login"
        )
    store = broker._auth_store_instance()  # noqa: SLF001
    row_id = broker._credential_id_for(key)  # noqa: SLF001
    if row_id is None:
        return broker._refuse(  # noqa: SLF001
            link, key, provider, by, "no_local_credential", "no stored login to lend"
        )
    row = store.get_credential(row_id)
    data = dict(getattr(row, "data", {}) or {})
    existing = data.get(REFRESH_DELEGATE_KEY)
    if isinstance(existing, dict) and existing.get("state") != LENT_STATE:
        return None  # a signed delegation: slice 1's handler owns it
    holder = entry.holder(by)
    native = store.native_login_incarnation(row_id)
    reason = lend_refusal(
        provider=provider,
        native_incarnation=native,
        holder_scope=holder.scope if holder is not None else "",
        link_capabilities=getattr(link, "capabilities", ()),
        trusted=_network_trusted(broker.network_id, broker.root),
        root=broker.root,
    )
    if reason:
        return broker._refuse(  # noqa: SLF001
            link,
            key,
            provider,
            by,
            "renewal_not_lent",
            "this login is lent for access only; its renewal stays on the owner",
            reason=reason,
        )
    token = store._presented_refresh_token(data)  # noqa: SLF001
    if not token:
        return broker._refuse(  # noqa: SLF001
            link, key, provider, by, "grant_invalid", "this login holds no renewal token"
        )
    now_ms = int(time.time() * 1000)
    outcome: dict[str, Any] = {}

    def write(payload: dict[str, Any]) -> bool:
        current = payload.get(REFRESH_DELEGATE_KEY)
        if isinstance(current, dict):
            if current.get("state") != LENT_STATE:
                outcome["code"] = "signed"
                return False
            if str(current.get("device") or "") != by:
                # R1: ONE renewer per login. The renewer moves only through the owner
                # (claim back, then lend) — that move is B2's (§D D1); until then a
                # second node keeps the access-only hold.
                outcome["code"] = "already_lent"
                outcome["device"] = str(current.get("device") or "")
                return False
            record = dict(current)
        else:
            record = {
                "state": LENT_STATE,
                "authority": AUTHORITY_DEVICE,
                "device": by,
                # Read by the owner fence, which sees the row and nothing else.
                "key": key,
                "provider": provider,
                "epoch": 0,
                "issued_at": now_ms / 1000.0,
                "salt": new_salt(),
                # The row's LOGIN incarnation column (a sign-in moves it; a refresh
                # does not): the write-back's lineage check (§B.4 case 0).
                "login_incarnation": str(native or ""),
            }
        record["lent_fp"] = fingerprint(
            record["salt"], store._presented_refresh_token(payload)
        )  # noqa: SLF001
        record["last_contact_ms"] = now_ms
        record["device_name"] = _member_name(broker, by)
        payload[REFRESH_DELEGATE_KEY] = record
        outcome["record"] = dict(record)
        return True

    if not store._update_payload(row_id, write, moves_write_stamp=True):  # noqa: SLF001
        if outcome.get("code") == "already_lent":
            return broker._refuse(  # noqa: SLF001
                link,
                key,
                provider,
                by,
                "renewal_not_lent",
                "this login's renewal is lent to another device; this one keeps the "
                "access-only hold",
                reason="already_lent",
            )
        if outcome.get("code") == "signed":
            return None
        return broker._refuse(  # noqa: SLF001
            link, key, provider, by, "internal", "the lend could not be recorded"
        )
    record = outcome["record"]
    fresh = dict(getattr(store.get_credential(row_id), "data", {}) or {})
    broker._audit(  # noqa: SLF001
        "credential.renewal_lent",
        actor=broker.self_device,
        subject=by,
        detail={
            "credential_key": key,
            "act": broker.self_device,
            "sub": by,
            "epoch": int(record.get("epoch") or 0),
        },
    )
    return material_reply(key, provider, record, fresh, kind="refresh_take")


def material_reply(
    key: str, provider: str, record: dict[str, Any], data: dict[str, Any], *, kind: str
) -> dict[str, Any]:
    """The lend/push/stale answer: the CURRENT pair plus what custody needs to fork from it."""
    from local_operator.providers.auth_store import AuthStore

    return {
        "kind": kind,
        "ok": True,
        "key": key,
        "provider": provider,
        "authority": AUTHORITY_DEVICE,
        WIRE_EPOCH: int(record.get("epoch") or 0),
        "salt": str(record.get("salt") or ""),
        "login_incarnation": str(record.get("login_incarnation") or ""),
        "refresh": AuthStore._presented_refresh_token(data),  # noqa: SLF001
        "access": data.get("access"),
        "expires": data.get("expires"),
        "token_type": data.get("token_type") or "Bearer",
        "email": data.get("email"),
        "account_id": data.get("account_id"),
    }


def _member_name(broker: Any, device: str) -> str:
    try:
        from local_operator.network import store

        member = store.load(broker.network_id, broker.root).member(device)
        return str(getattr(member, "name", "") or "") if member is not None else ""
    except Exception:  # noqa: BLE001
        return ""


# ---------------------------------------------------------------------------
# Owner — write-back adoption and the conflict rule (§B.4)
# ---------------------------------------------------------------------------


def owner_return(
    broker: Any, link: Any, key: str, by: str, frame: dict[str, Any]
) -> dict[str, Any]:
    """Adopt a node's write-back, or say why not. Called for a ``lent`` record only.

    The four cases of §B.4, in the order that makes each unambiguous:

    0. the lend is not this node's, or the login moved (incarnation) → ``renewal_stale``;
    1. the node renewed from the pair the owner still holds → ADOPT;
    4. an epoch the owner has already adopted (a replayed ``done``) → ``renewal_stale``;
    2. the owner renewed and the node did not → ``stale`` + the owner's current pair;
    3. both claim a renewal → the OWNER'S pair wins, ``renewal_conflict`` is written.
    """
    from local_operator.providers.auth_store import (
        AWAITING_WRITEBACK_KEY,
        REFRESH_DELEGATE_KEY,
        REFRESH_SEND_UNCONFIRMED_KEY,
    )

    store = broker._auth_store_instance()  # noqa: SLF001
    row_id = broker._credential_id_for(key)  # noqa: SLF001
    entry = broker._entry(key)  # noqa: SLF001
    provider = str((entry.provider if entry else "") or key)
    epoch = int(frame.get(WIRE_EPOCH) or 0)
    parent_epoch = int(frame.get(WIRE_PARENT_EPOCH) or 0)
    parent_fp = str(frame.get("parent_fp") or "")
    incarnation = str(frame.get("login_incarnation") or "")
    receipts = [r for r in (frame.get("receipts") or []) if isinstance(r, dict)][:64]
    outcome: dict[str, Any] = {"case": "stale_refused"}
    # The login's lineage NOW: a sign-in on this device since the lend moved it, and
    # the node's family is then another login's (case 0, refused).
    native = str(store.native_login_incarnation(row_id) or "") if row_id is not None else ""

    def write(payload: dict[str, Any]) -> bool:
        record = payload.get(REFRESH_DELEGATE_KEY)
        if (
            not isinstance(record, dict)
            or record.get("state") != LENT_STATE
            or str(record.get("device") or "") != by
            or str(record.get("login_incarnation") or "") != native
            or incarnation != native
            or store._grant_dead_at(payload) is not None  # noqa: SLF001
        ):
            return False
        salt = str(record.get("salt") or "")
        current_fp = fingerprint(salt, store._presented_refresh_token(payload))  # noqa: SLF001
        owner_epoch = int(record.get("epoch") or 0)
        record = dict(record)
        record["last_contact_ms"] = int(time.time() * 1000)
        if parent_fp and parent_fp == current_fp and epoch > parent_epoch == owner_epoch:
            new_refresh = frame.get("refresh")
            if not isinstance(new_refresh, str) or not new_refresh:
                return False
            payload["refresh"] = new_refresh
            payload.pop("refresh_token", None)
            if frame.get("access") is not None:
                payload["access"] = frame.get("access")
            if frame.get("expires") is not None:
                payload["expires"] = frame.get("expires")
            payload.pop(AWAITING_WRITEBACK_KEY, None)
            payload.pop(REFRESH_SEND_UNCONFIRMED_KEY, None)
            record["epoch"] = epoch
            record["lent_fp"] = fingerprint(salt, new_refresh)
            # The keyed fingerprints of the last write-backs ADOPTED (never the
            # tokens): how a replay is told from a conflict once the owner has
            # renewed again past the adopted pair (case 4 below).
            adopted = [str(fp) for fp in (record.get("adopted_fps") or []) if fp]
            record["adopted_fps"] = [*adopted, record["lent_fp"]][-ADOPTED_FPS_MAX:]
            record.pop("claim", None)
            outcome["case"] = "adopted"
        elif (
            epoch <= owner_epoch
            and epoch > parent_epoch
            and isinstance(frame.get("refresh"), str)
            and fingerprint(salt, frame.get("refresh"))
            in {current_fp, *(record.get("adopted_fps") or [])}
        ):
            # Case 4: a REPLAY of a write-back already adopted. The epochs alone
            # cannot say so — an owner that rotated once is also at epoch 1, and a
            # node renewing from the same parent reports epoch 1 too (case 3) — so
            # the pair decides: a replay carries a pair the owner holds or adopted.
            # Both branches adopt nothing; only the receipt differs.
            return False
        elif epoch == parent_epoch:
            outcome["case"] = "stale"  # case 2 (or nothing renewed): re-lend the owner's
        else:
            outcome["case"] = "conflict"  # case 3: the owner's own rotation wins
        payload[REFRESH_DELEGATE_KEY] = record
        outcome["record"] = record
        return True

    acquired = _take_lease(store, row_id)
    try:
        written = row_id is not None and store._update_payload(  # noqa: SLF001
            row_id, write, moves_write_stamp=True
        )
    finally:
        if acquired:
            store._release_refresh_lease(row_id)  # noqa: SLF001
    detail = {"credential_key": key, "act": by, "sub": broker.self_device, "epoch": epoch}
    if not written:
        broker._audit(  # noqa: SLF001
            "credential.renewal_stale",
            actor=broker.self_device,
            subject=by,
            detail={**detail, "code": "renewal_stale"},
        )
        return broker._refuse(  # noqa: SLF001
            link,
            key,
            provider,
            by,
            "renewal_stale",
            "this write-back is older than the login this device holds, or names a lend "
            "that has ended; nothing was adopted",
        )
    record = outcome["record"]
    fresh = dict(getattr(store.get_credential(row_id), "data", {}) or {})
    case = outcome["case"]
    if case == "adopted":
        stamps = [int(r.get("at_ms") or 0) for r in receipts if int(r.get("at_ms") or 0) > 0]
        broker._audit(  # noqa: SLF001
            "credential.renewal_returned",
            actor=broker.self_device,
            subject=by,
            detail={**detail, "renewals": len(stamps), "last_at": max(stamps) if stamps else 0},
        )
        for receipt in receipts:
            broker._audit(  # noqa: SLF001
                "credential.renewed_by_peer",
                actor=broker.self_device,
                subject=by,
                detail={
                    "credential_key": key,
                    "act": by,
                    "sub": broker.self_device,
                    "epoch": int(receipt.get("epoch") or 0),
                    "at": int(receipt.get("at_ms") or 0),
                },
            )
        notify_returned(
            broker.root,
            network_id=broker.network_id,
            node=by,
            node_label=str(record.get("device_name") or by),
            key=key,
            provider=provider,
            stamps=stamps,
        )
        return {"kind": "refresh_return", "act": "done", "ok": True, WIRE_EPOCH: epoch}
    if case == "conflict":
        broker._audit(  # noqa: SLF001
            "credential.renewal_conflict",
            actor=broker.self_device,
            subject=by,
            detail={**detail, "code": "owner_wins"},
        )
    reply = material_reply(key, provider, record, fresh, kind="refresh_return")
    reply.update(act="stale", code="owner_wins" if case == "conflict" else "stale")
    return reply


def _take_lease(store: Any, row_id: int | None) -> bool:
    """The store's cross-process refresh lease, so no owner process POSTs mid-adoption."""
    if row_id is None:
        return False
    try:
        return bool(store._try_refresh_lease(row_id))  # noqa: SLF001
    except Exception:  # noqa: BLE001 — the payload CAS still guards the write
        return False


def notify_returned(
    root: Path | None,
    *,
    network_id: str,
    node: str,
    node_label: str,
    key: str,
    provider: str,
    stamps: list[int],
) -> None:
    """ONE Notifications item per ``(node, key)``, updated on each write-back. Never raises."""
    if not stamps:
        return
    try:
        from local_operator.notifications.store import NotificationStore
        from local_operator.paths import config_dir

        NotificationStore(root if root is not None else config_dir()).notify(
            "network-credentials",
            title=returned_notice(node_label, provider, stamps),
            body_md=(
                f"{node_label} kept your {provider} login renewed while this Mac was "
                "away and handed it back when they reconnected; nothing to do."
            ),
            priority="normal",
            key=f"renewal-returned:{network_id}:{node}:{key}",
            session_title="lop network credentials",
        )
    except Exception:  # noqa: BLE001 — a notice is never a reason to fail an adoption
        logger.debug("renewal notice could not be written", exc_info=True)


# ---------------------------------------------------------------------------
# Owner — the claim (R3/R4), run by the relay's ``credential_claim`` local op
# ---------------------------------------------------------------------------

_claim_locks: dict[tuple[str, str], threading.Lock] = {}
_claim_locks_guard = threading.Lock()


def owner_claim(broker: Any, key: str, *, wait_s: float | None = None) -> dict[str, Any]:
    """Ask the renewing node before the owner presents ``key``'s refresh token.

    Writes the answer onto the lent record, where every owner process's fence reads
    it (:func:`owner_fence_verdict`):

    * ``standby`` (the node has not renewed and is at the owner's epoch) → one owner
      POST is authorised for :data:`CLAIM_FRESH_S`;
    * ``renewed`` (the node answered with a write-back) → adopted here; usually no
      owner POST is needed at all;
    * no answer within ``wait_s`` → ``awaiting_handback`` (R4): the owner does not
      present, and says so.

    One claim per login per relay at a time; a second caller waits for the first.
    """
    from local_operator.network.credentials.types import PEER_BROKER_OP

    if wait_s is None:
        wait_s = CLAIM_WAIT_S  # read at call time, so a test can shorten W
    lock_key = (str(broker.network_id), key)
    with _claim_locks_guard:
        lock = _claim_locks.setdefault(lock_key, threading.Lock())
    with lock:
        store = broker._auth_store_instance()  # noqa: SLF001
        row_id = broker._credential_id_for(key)  # noqa: SLF001
        row = store.get_credential(row_id) if row_id is not None else None
        record = lent_record(dict(getattr(row, "data", {}) or {}))
        if row_id is None or record is None:
            return {"state": "not_lent"}
        claim = record.get("claim") or {}
        now_ms = int(time.time() * 1000)
        if (
            claim.get("state") == "standby"
            and now_ms - int(claim.get("at_ms") or 0) < CLAIM_FRESH_S * 1000
            and _claim_epoch(claim) == int(record.get("epoch") or 0)
        ):
            return {"state": "standby", "fresh": True}
        node = str(record.get("device") or "")
        owner_epoch = int(record.get("epoch") or 0)
        client = getattr(broker, "_client", None)
        reply = None
        if client is not None:
            try:
                reply = _ask_with_budget(
                    client,
                    node,
                    {
                        "op": PEER_BROKER_OP,
                        "kind": "renewal_claim",
                        "key": key,
                        WIRE_EPOCH: owner_epoch,
                    },
                    wait_s,
                )
            except Exception:  # noqa: BLE001 — no answer is R4, never a crash
                reply = None
        detail = reply.get("detail") if isinstance(reply, dict) else None
        answer = str(detail.get("state") or "") if isinstance(detail, dict) else ""
        if answer == "renewed" and isinstance(detail, dict):

            class _Link:  # the node's write-back, carried by the claim answer
                device_id = node
                capabilities = (RENEWAL_CAP_STRING,)

            adopted = owner_return(broker, _Link(), key, node, {**detail, "act": "done"})
            state = "renewed" if adopted.get("ok") and adopted.get("act") == "done" else "standby"
            if state == "renewed":
                _record_claim(store, row_id, {"state": "renewed", "at_ms": now_ms})
                _audit_claim(broker, key, node, "renewed")
                return {"state": "renewed"}
        if answer in ("standby", "renewed"):
            node_epoch = int((detail or {}).get(WIRE_EPOCH) or 0)
            if node_epoch < owner_epoch:
                # The node missed a push: send the owner's pair before it may stand by.
                push_to_node(broker, key)
            _record_claim(
                store, row_id, {"state": "standby", "at_ms": now_ms, "epoch": owner_epoch}
            )
            _audit_claim(broker, key, node, "standby")
            return {"state": "standby"}
        if answer == "none":
            # The node holds no custody (wiped it, or never stored it): nothing it
            # holds can be newer than the owner's pair, so the owner may renew.
            _record_claim(
                store, row_id, {"state": "standby", "at_ms": now_ms, "epoch": owner_epoch}
            )
            _audit_claim(broker, key, node, "no_custody")
            return {"state": "standby"}
        _record_claim(store, row_id, {"state": "awaiting_handback", "at_ms": now_ms})
        _audit_claim(broker, key, node, "no_answer")
        return {"state": "awaiting_handback", "node": node}


def _ask_with_budget(client: Any, node: str, frame: dict[str, Any], wait_s: float) -> Any:
    """``_ask_owner_directly`` bounded by ``wait_s`` (R4's ``W``), on a worker thread."""
    box: dict[str, Any] = {}

    def run() -> None:
        try:
            box["reply"] = client._ask_owner_directly(  # noqa: SLF001
                node, frame, required_capability=RENEWAL_CAP_STRING
            )
        except Exception:  # noqa: BLE001
            box["reply"] = None

    worker = threading.Thread(target=run, name="mesh-renewal-claim", daemon=True)
    worker.start()
    worker.join(wait_s)
    return box.get("reply")


def _record_claim(store: Any, row_id: int, claim: dict[str, Any]) -> None:
    from local_operator.providers.auth_store import REFRESH_DELEGATE_KEY

    def write(payload: dict[str, Any]) -> bool:
        record = payload.get(REFRESH_DELEGATE_KEY)
        if not isinstance(record, dict) or record.get("state") != LENT_STATE:
            return False
        record = dict(record)
        record["claim"] = dict(claim)
        if claim.get("state") in ("standby", "renewed"):
            record["last_contact_ms"] = int(claim.get("at_ms") or 0)
        payload[REFRESH_DELEGATE_KEY] = record
        return True

    try:
        store._update_payload(row_id, write)  # noqa: SLF001
    except Exception:  # noqa: BLE001
        logger.debug("claim outcome could not be recorded", exc_info=True)


def _audit_claim(broker: Any, key: str, node: str, outcome: str) -> None:
    broker._audit(  # noqa: SLF001
        "credential.renewal_claim",
        actor=broker.self_device,
        subject=node,
        detail={"credential_key": key, "act": broker.self_device, "sub": node, "outcome": outcome},
    )


def push_to_node(broker: Any, key: str) -> bool:
    """Send the owner's current pair to the renewing node (after an owner POST). Never raises."""
    from local_operator.network.credentials.types import PEER_BROKER_OP

    try:
        store = broker._auth_store_instance()  # noqa: SLF001
        row_id = broker._credential_id_for(key)  # noqa: SLF001
        data = dict(getattr(store.get_credential(row_id), "data", {}) or {})
        record = lent_record(data)
        entry = broker._entry(key)  # noqa: SLF001
        if record is None or entry is None:
            return False
        frame = material_reply(key, str(entry.provider or key), record, data, kind="renewal_push")
        frame["op"] = PEER_BROKER_OP
        reply = _ask_with_budget(
            broker._client, str(record.get("device") or ""), frame, CLAIM_WAIT_S  # noqa: SLF001
        )
        detail = reply.get("detail") if isinstance(reply, dict) else None
        return bool(isinstance(detail, dict) and detail.get("ok"))
    except Exception:  # noqa: BLE001 — a lost push leaves the node waiting, never presenting
        logger.debug("renewal push failed", exc_info=True)
        return False


def note_owner_posted(store: Any, credential_id: int, data: dict[str, Any]) -> bool:
    """After an OWNER POST of a lent login: bump the epoch, consume the claim.

    The node's copy is now spent; its next claim answer (or this push) carries the
    owner's new pair. Returns whether the row is lent (the caller then pushes).
    """
    from local_operator.providers.auth_store import REFRESH_DELEGATE_KEY

    if lent_record(data) is None:
        return False

    def write(payload: dict[str, Any]) -> bool:
        record = payload.get(REFRESH_DELEGATE_KEY)
        if not isinstance(record, dict) or record.get("state") != LENT_STATE:
            return False
        record = dict(record)
        record["epoch"] = int(record.get("epoch") or 0) + 1
        record["lent_fp"] = fingerprint(
            str(record.get("salt") or ""), store._presented_refresh_token(payload)  # noqa: SLF001
        )
        record.pop("claim", None)
        payload[REFRESH_DELEGATE_KEY] = record
        return True

    return bool(store._update_payload(credential_id, write))  # noqa: SLF001


# ---------------------------------------------------------------------------
# Owner — the fence verdict, read in EVERY owner process before a POST
# ---------------------------------------------------------------------------


def _claim_epoch(claim: dict[str, Any]) -> int:
    """The epoch a recorded claim answered for; ``-1`` when it names none.

    NOT ``int(claim.get("epoch") or -1)``: epoch 0 is the first lend's epoch and is
    falsy, so that spelling read every first-epoch ``standby`` as "no epoch" and the
    fence never allowed the owner's POST it had just authorised.
    """
    raw = claim.get("epoch")
    if isinstance(raw, bool) or not isinstance(raw, int):
        return -1
    return raw


def owner_fence_verdict(data: dict[str, Any], *, now_ms: int | None = None) -> tuple[str, str]:
    """``(verdict, node_label)`` for a lent row: ``allow`` / ``claim`` / ``awaiting``.

    ``""`` verdict = not lent (the fence does nothing). Pure: reads the row only.
    """
    record = lent_record(data)
    if record is None:
        return "", ""
    now = int(time.time() * 1000) if now_ms is None else now_ms
    label = str(record.get("device_name") or record.get("device") or "the node")
    claim = record.get("claim") or {}
    state = claim.get("state")
    at_ms = int(claim.get("at_ms") or 0)
    if (
        state == "standby"
        and 0 <= now - at_ms < CLAIM_FRESH_S * 1000
        and _claim_epoch(claim) == int(record.get("epoch") or 0)
    ):
        return "allow", label
    if state == "awaiting_handback":
        last = int(record.get("last_contact_ms") or 0)
        if last and now - last >= RETURN_CAP_S * 1000:
            # R4 clears on its own at the return cap (§D reconciliation): past it the
            # node has certainly deleted anything newer than this pair.
            return "allow", label
        return "awaiting", label
    return "claim", label


class OwnerLendFence:
    """The owner fence ``auth_store.set_lend_fence`` installs in EVERY owner process.

    Consulted only for a LENT row, right before this process would present its
    refresh token. The request path never sits on the claim (manager ruling,
    2026-11-14): the claim runs on a background thread through this device's relay
    (``credential_claim``, up to :data:`CLAIM_WAIT_S`), the fence waits at most
    :data:`CLAIM_INLINE_S` for it, and otherwise answers "checking with <node>"
    so the turn moves on; the claim's outcome lands on the row, where the next
    refresh reads it.
    """

    def __init__(self, root: Path | None) -> None:
        self.root = root

    def __call__(self, credential_id: int, data: dict[str, Any]) -> str | None:
        verdict, label = owner_fence_verdict(data)
        if verdict in ("", "allow"):
            return None
        record = lent_record(data) or {}
        key = str(record.get("key") or "")
        provider = str(record.get("provider") or key or "this")
        done = request_claim(self.root, key)
        if verdict == "awaiting":
            _note_awaiting(self.root, provider, record)
            return awaiting_handback_message(provider, label)
        if done is not None and done.wait(CLAIM_INLINE_S):
            outcome = _claim_outcomes.get((_root_key(self.root), key), "")
            if outcome == "standby":
                return None
            if outcome == "renewed":
                # The node's newer pair was adopted: nothing to POST. The caller
                # re-reads the row on its next resolve.
                return f"{label} handed back a newer {provider} login; it is in use now"
            if outcome == "awaiting_handback":
                _note_awaiting(self.root, provider, record)
                return awaiting_handback_message(provider, label)
        return checking_message(provider, label)

    def posted(self, credential_id: int, data: dict[str, Any]) -> None:
        """After this owner renewed a lent login: bump the epoch, push to the node."""
        from local_operator.providers.auth_store import AuthStore

        record = lent_record(data)
        if record is None:
            return
        try:
            store = AuthStore(config_dir=self.root) if self.root is not None else AuthStore()
            try:
                note_owner_posted(store, credential_id, data)
            finally:
                store.close()
        except Exception:  # noqa: BLE001 — the next claim pushes what this missed
            logger.debug("lent-login epoch bump failed", exc_info=True)
        _relay_op_async(self.root, "credential_lend_push", str(record.get("key") or ""))


_claim_threads: dict[tuple[str, str], threading.Event] = {}
_claim_outcomes: dict[tuple[str, str], str] = {}
_claim_started: dict[tuple[str, str], float] = {}
_claim_guard = threading.Lock()


def _root_key(root: Path | None) -> str:
    return str(root) if root is not None else ""


def request_claim(root: Path | None, key: str) -> threading.Event | None:
    """Start (or join) ONE background ``credential_claim`` for ``key``; its done event.

    Rate-limited per key (:data:`CLAIM_RETRY_S`) so a burst of requests does not
    queue claims behind each other on the relay.
    """
    if not key:
        return None
    slot = (_root_key(root), key)
    with _claim_guard:
        running = _claim_threads.get(slot)
        if running is not None and not running.is_set():
            return running
        if time.monotonic() - _claim_started.get(slot, -1e9) < CLAIM_RETRY_S and running:
            return running
        done = threading.Event()
        _claim_threads[slot] = done
        _claim_started[slot] = time.monotonic()

    def run() -> None:
        try:
            answer = _relay_op(root, "credential_claim", key, timeout=CLAIM_WAIT_S + 5.0)
            _claim_outcomes[slot] = str((answer or {}).get("state") or "")
        finally:
            done.set()

    threading.Thread(target=run, name="mesh-renewal-claim-ask", daemon=True).start()
    return done


def _relay_op(
    root: Path | None, op: str, key: str, *, timeout: float = 10.0
) -> dict[str, Any] | None:
    """One local op on this device's relay; ``None`` when it does not answer."""
    from local_operator.network import relay, store

    try:
        record = store.find_own_relay(root) if root is not None else store.find_own_relay()
        if record is None:
            return None
        reply = relay.control_request(record, op, timeout=timeout, key=key)
    except Exception:  # noqa: BLE001 — no answer is the R4 path, never a crash
        return None
    detail = (reply or {}).get("detail") if isinstance(reply, dict) else None
    return detail if isinstance(detail, dict) else None


def _relay_op_async(root: Path | None, op: str, key: str) -> None:
    threading.Thread(target=_relay_op, args=(root, op, key), name=f"mesh-{op}", daemon=True).start()


def _note_awaiting(root: Path | None, provider: str, record: dict[str, Any]) -> None:
    """Publish ``awaiting_handback`` for the status surfaces (§D D2). Never raises."""
    try:
        from local_operator.network.credentials import effective

        effective.note_awaiting_handback(
            root,
            provider,
            node=str(record.get("device") or ""),
            node_label=str(record.get("device_name") or record.get("device") or ""),
        )
    except Exception:  # noqa: BLE001
        logger.debug("awaiting_handback could not be published", exc_info=True)


# ---------------------------------------------------------------------------
# Node — store-only custody (B1). B2 adds the renewal gate.
# ---------------------------------------------------------------------------

#: The custody block on the NODE's row. A ``REFRESH_DELEGATE_KEY`` record in a LIVE
#: state with no ceiling, so ``auth_store`` refuses every POST of this row
#: (``RefreshDelegatedError``): in B1 the node only STORES the lent pair.
CUSTODY_KEY = "renewal_custody"


def node_custody_marker(owner: str) -> dict[str, Any]:
    return {"state": "active", "device": owner, "custody": "renewal-b1", "not_after": None}


def node_store(client: Any, key: str, owner: str, detail: dict[str, Any]) -> bool:
    """Write (or replace) the node's custody of ``key`` from an owner lend/push/stale answer."""
    from local_operator.network.credentials import sync as sync_mod
    from local_operator.providers.auth_store import REFRESH_DELEGATE_KEY

    refresh = detail.get("refresh")
    if not isinstance(refresh, str) or not refresh:
        return False
    salt = str(detail.get("salt") or "")
    epoch = int(detail.get(WIRE_EPOCH) or 0)
    payload: dict[str, Any] = {
        "access": detail.get("access"),
        "refresh": refresh,
        "expires": detail.get("expires"),
        "token_type": detail.get("token_type") or "Bearer",
        "login_incarnation": str(detail.get("login_incarnation") or ""),
        sync_mod.MESH_ORIGIN_KEY: {
            "owner_device": owner,
            "key": key,
            "grant_id": f"rl_{salt[:12]}",
            "epoch": epoch,
        },
        REFRESH_DELEGATE_KEY: node_custody_marker(owner),
        CUSTODY_KEY: {
            "owner": owner,
            "key": key,
            "epoch": epoch,
            "parent_epoch": epoch,
            "salt": salt,
            "parent_fp": fingerprint(salt, refresh),
            "received_wall": time.time(),
            "receipts": [],
        },
    }
    for field in ("email", "account_id"):
        if detail.get(field):
            payload[field] = detail.get(field)
    try:
        client._auth_store().upsert_credential(  # noqa: SLF001
            str(detail.get("provider") or key), payload
        )
    except Exception:  # noqa: BLE001 — a write failure is a refusal, never a crash
        return False
    return True


def node_custody(store: Any, key: str, owner: str) -> tuple[int, dict[str, Any]] | None:
    """``(row_id, data)`` of the node's custody row for ``key`` from ``owner``, or ``None``."""
    try:
        rows = store.list_credentials()
    except Exception:  # noqa: BLE001
        return None
    for row in rows:
        data = dict(getattr(row, "data", {}) or {})
        custody = data.get(CUSTODY_KEY)
        if (
            isinstance(custody, dict)
            and str(custody.get("key") or "") == key
            and str(custody.get("owner") or "") == owner
        ):
            return int(row.id), data
    return None


def node_take(client: Any, key: str) -> dict[str, Any]:
    """Ask the owner for ``key``'s renewal token under the device authority, and store it."""
    from local_operator.network.credentials.types import PEER_BROKER_OP

    owner = client.owner_of(key)
    if not owner:
        return {"ok": False, "code": "not_a_holder"}
    reply = client._ask_owner_directly(  # noqa: SLF001
        owner,
        {"op": PEER_BROKER_OP, "kind": "refresh_take", "key": key, "authority": AUTHORITY_DEVICE},
        required_capability=RENEWAL_CAP_STRING,
    )
    detail = reply.get("detail") if isinstance(reply, dict) else None
    if not isinstance(detail, dict):
        return {"ok": False, "code": "owner_offline"}
    if detail.get("kind") == "error" or detail.get("ok") is False:
        return {
            "ok": False,
            "code": str(detail.get("code") or "internal"),
            "message": str(detail.get("message") or ""),
            "reason": str(detail.get("reason") or ""),
        }
    if not node_store(client, key, owner, detail):
        return {"ok": False, "code": "internal", "message": "custody could not be stored"}
    return {"ok": True, "epoch": int(detail.get(WIRE_EPOCH) or 0)}


def _node_client(server: Any) -> Any:
    from local_operator.network.credentials.client import MeshCredentialClient

    return MeshCredentialClient.for_relay(server)


def node_on_claim(server: Any, link: Any, frame: dict[str, Any]) -> dict[str, Any]:
    """The node's answer to an owner claim. B1 never renews, so it is ``standby``.

    ``none`` when this node holds no custody for the key from THIS owner (the owner
    is the authenticated link, never the frame).
    """
    key = str(frame.get("key") or "")
    owner = str(getattr(link, "device_id", "") or "")
    client = _node_client(server)
    found = (
        node_custody(client._auth_store(), key, owner) if client is not None else None
    )  # noqa: SLF001
    if found is None:
        return {"kind": "renewal_claim", "ok": True, "state": "none"}
    custody = found[1].get(CUSTODY_KEY) or {}
    return {
        "kind": "renewal_claim",
        "ok": True,
        "state": "standby",
        WIRE_EPOCH: int(custody.get("epoch") or 0),
    }


def node_on_push(server: Any, link: Any, frame: dict[str, Any]) -> dict[str, Any]:
    """Replace this node's custody with the owner's newer pair (after an owner POST)."""
    key = str(frame.get("key") or "")
    owner = str(getattr(link, "device_id", "") or "")
    client = _node_client(server)
    if client is None or client.owner_of(key) != owner:
        return {"kind": "renewal_push", "ok": False, "code": "not_owner"}
    found = node_custody(client._auth_store(), key, owner)  # noqa: SLF001
    if found is not None:
        current = int((found[1].get(CUSTODY_KEY) or {}).get("epoch") or 0)
        if int(frame.get(WIRE_EPOCH) or 0) <= current:
            return {"kind": "renewal_push", "ok": True, WIRE_EPOCH: current, "applied": False}
    ok = node_store(client, key, owner, frame)
    return {"kind": "renewal_push", "ok": ok, WIRE_EPOCH: int(frame.get(WIRE_EPOCH) or 0)}


def node_write_back(client: Any, key: str) -> dict[str, Any]:
    """Send this node's custody back to the owner (``refresh_return{act: done}``).

    B1 nodes never renew, so a B1 write-back carries the lent pair and the owner
    answers ``stale`` (nothing to adopt). It exists here because the sender and the
    owner's adoption are one contract; B2 calls it on reconnect after renewing.
    """
    from local_operator.network.credentials.types import PEER_BROKER_OP

    owner = client.owner_of(key)
    store = client._auth_store()  # noqa: SLF001
    found = node_custody(store, key, owner)
    if found is None:
        return {"ok": False, "code": "no_custody"}
    data = found[1]
    custody = data.get(CUSTODY_KEY) or {}
    frame = {
        "op": PEER_BROKER_OP,
        "kind": "refresh_return",
        "act": "done",
        "key": key,
        "authority": AUTHORITY_DEVICE,
        WIRE_EPOCH: int(custody.get("epoch") or 0),
        WIRE_PARENT_EPOCH: int(custody.get("parent_epoch") or 0),
        "parent_fp": str(custody.get("parent_fp") or ""),
        "login_incarnation": str(data.get("login_incarnation") or ""),
        "refresh": store._presented_refresh_token(data),  # noqa: SLF001
        "access": data.get("access"),
        "expires": data.get("expires"),
        "receipts": list(custody.get("receipts") or []),
    }
    # ONE retry on no answer: the owner's own push to this node dials it, and the
    # relay keeps one link per pair, so a write-back can start on the link that dial
    # is replacing and get nothing back. The write-back is idempotent on the owner
    # (a repeat is case 4, ``renewal_stale``), so asking twice is safe.
    reply = None
    for _attempt in range(2):
        reply = client._ask_owner_directly(  # noqa: SLF001
            owner, frame, required_capability=RENEWAL_CAP_STRING
        )
        if isinstance(reply, dict):
            break
        time.sleep(0.5)
    detail = reply.get("detail") if isinstance(reply, dict) else None
    if not isinstance(detail, dict):
        return {"ok": False, "code": "owner_offline"}
    if detail.get("act") == "stale" and detail.get("ok"):
        node_store(client, key, owner, detail)
        return {"ok": True, "state": "stale", "code": str(detail.get("code") or "")}
    if detail.get("ok") and detail.get("act") == "done":
        return {"ok": True, "state": "adopted"}
    return {"ok": False, "code": str(detail.get("code") or "internal")}


# ---------------------------------------------------------------------------
# Owner — logout of a lent login (§D D3)
# ---------------------------------------------------------------------------

#: Where a person signs out of each renewable provider when lop cannot revoke there.
PROVIDER_SIGN_OUT: dict[str, str] = {
    "anthropic": "sign out of Anthropic on claude.ai",
    "radient": "sign out of Radient on its website",
}


def logout_revoke_lent(store: Any, providers: list[str], definition: Any) -> list[str]:
    """Revoke every LENT login of ``providers`` at the provider; the lines to print.

    Runs BEFORE ``lop logout`` deletes the rows. The node holding the lent pair is
    told at its next contact (B2); meanwhile only a provider-side revoke ends the
    family. ``definition.revoke_token`` is used when the provider has one. When it
    does not (every provider today), the lines say what the node can still do and
    where to sign out instead — they never claim a revoke happened. Each outcome is
    recorded as ``credential.provider_revoked`` (``revoked`` / ``failed`` /
    ``unsupported``). Never raises.
    """
    import asyncio

    from local_operator.network.credentials import offline_hold_s

    lines: list[str] = []
    revoke = getattr(definition, "revoke_token", None)
    for provider in providers:
        try:
            rows = store.list_credentials(provider)
        except Exception:  # noqa: BLE001
            continue
        for row in rows:
            data = dict(getattr(row, "data", {}) or {})
            record = lent_record(data)
            if record is None:
                continue
            node = str(record.get("device_name") or record.get("device") or "the node")
            key = str(record.get("key") or provider)
            outcome = "unsupported"
            if revoke is not None:
                try:
                    outcome = "revoked" if asyncio.run(revoke(data)) else "failed"
                except Exception:  # noqa: BLE001 — a failed revoke is reported, not raised
                    outcome = "failed"
            _audit_provider_revoked(getattr(store, "_config_dir", None), key, provider, outcome)
            if outcome == "revoked":
                lines.append(
                    f"Revoked your {provider} login at the provider: {node}'s lent copy no "
                    "longer works."
                )
                continue
            hold_h = int(offline_hold_s(getattr(store, "_config_dir", None)) // 3600)
            where = PROVIDER_SIGN_OUT.get(provider, f"sign out of {provider} at the provider")
            reason = (
                "the provider refused the revoke"
                if outcome == "failed"
                else "lop cannot revoke this login at the provider"
            )
            lines.append(
                f"{node} holds a lent copy of this {provider} login and {reason}: {node} can "
                f"keep using it until it reconnects or for up to {hold_h} h; to end it now, "
                f"{where}."
            )
    return lines


def _audit_provider_revoked(root: Any, key: str, provider: str, outcome: str) -> None:
    """``credential.provider_revoked`` in every network this device is in. Never raises."""
    try:
        from local_operator.network import audit as audit_mod
        from local_operator.network import store as net_store

        log = audit_mod.AuditLog(root) if root is not None else audit_mod.AuditLog()
        for record in net_store.list_networks(root):
            log.record(
                audit_mod.AuditEvent(
                    event="credential.provider_revoked",
                    actor=record.self_device_id,
                    network_id=record.network_id,
                    detail={"credential_key": key, "provider": provider, "outcome": outcome},
                )
            )
        log.flush()
    except Exception:  # noqa: BLE001 — a log that cannot be written is not a refusal
        logger.debug("provider revoke could not be audited", exc_info=True)
