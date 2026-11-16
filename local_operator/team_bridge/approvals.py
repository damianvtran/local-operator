"""The desktop approvals surface: pending requests, human batch decide, the ledger.

WHY THIS MODULE EXISTS. Requirement: approvals as a first-class flow — a
teammate's agent sends a ``request``, the operator sees "N requests waiting",
approves or denies several at once, and every decision is audited. The hub has
all four routes (``GET /messaging/requests``, the single and batch decide,
``GET /messaging/decisions``) and this package already had the executor
(:func:`~local_operator.team_bridge.decide.decide_batch_from_human_selection`),
but nothing called it. This module is the harness half the desktop routes in
``server/routes/desktop_team_bridge.py`` call; it owns no transport of its own.

THE TRUST BOUNDARY — DECIDE AUTHORITY IS HUMAN-ONLY. "Answered by the human" is
CLIENT-ASSERTED (TEAM-BRIDGE.md §10.1): the server checks the receipt's shape
and binding, never that a person tapped. So the human proof has to live HERE,
and it is an OPERATOR PRESENCE SIGNATURE, not the desktop bearer:

- the bearer is NOT a same-uid boundary. ``agent_shell.DAEMON_BEARER_ENVS``
  only keeps it out of an agent shell's ENVIRONMENT; the desktop persists it
  in a 0600 file any ordinary same-uid shell can read (security review
  round 1, S-1 — measured). ``agent_shell`` says it itself: only a
  presence-gated key proves a person.
- every batch therefore carries one operator signature over a digest of the
  EXACT selection (account, tenant, sorted ``(message_id, word)`` pairs) and a
  daemon-minted, single-use, 30 s challenge (:func:`verify_selection_proof`),
  verified BEFORE any hub call. The signing key is the same operator key the
  runtime's card path verifies (``operator.verify.signed_message`` /
  ``signature_verdict``); on an ``operator-presence`` host every signature
  costs an OS gesture (Touch ID), which is the human act. A host whose anchor
  is file-only or absent is refused (``operator_presence_required``): there a
  signature proves nothing against a same-uid process, and the per-item ask
  card stays the way to decide.
- this module is imported ONLY by the desktop route family (behind
  ``require_desktop``). No model tool, agent loop or connector imports it; a
  test pins that. That keeps the door narrow; it is the signature, not the
  import graph, that makes it human-only.
- the selection is EXPLICIT: every item names one ``message_id`` and one word
  (approve | deny). Nothing here expands "all", defaults a word, or derives a
  selection from server state — §10.2's select-and-approve is per item.
- every item must resolve to a REAL open bridge ask this machine raised for
  that request, in the designated CoS session, under the CURRENT owner binding
  (``authority_fence`` + ``scope_matches_runtime`` — the exact checks the
  single decide path runs). The ask id is what the server records as the
  receipt, so a receipt is never minted for an ask that does not exist.

NOTHING DECIDED TWICE. After the hub answers, each item it settled (decided, or
already terminal server-side) gets a ``withdrawn`` row in its ask's log. That
event is terminal-on-write and injects nothing (asks/store.py ``fold``), and —
unlike ``answered`` — it never reaches the decide-at-consumption seam, so the
ask card can no longer answer it (every answer re-folds the log from disk and
refuses a withdrawn ask, with copy naming the desktop decision). The reverse
direction is the list read: a request whose local ask is already answered is
not listed, and once that answer reaches the hub it is no longer pending there.

ONE RACE IS NOT FULLY CONSISTENT, and is accepted (review R1-5): a card answer
that lands WHILE a batch is awaiting the hub outranks the batch's later
``withdrawn`` in the fold, so the card's word is injected into the session,
while the hub keeps the batch's word (the card's own decide gets
``request_already_decided``, and finds the batch's intent row already recorded,
so it sends nothing). The hub's one-decision rule means nothing is decided
twice and the audit names the batch; only the session's local view can
disagree. The window is one human on two surfaces during one hub round trip.

CRASH / LOST-RESPONSE SAFETY (review R1-1). Each item is recorded in the SAME
durable decide-intent ledger the card path uses (``decide_intents``) BEFORE the
hub call, tagged ``consumption_row_id = "batch:…"`` so it is never mistaken for
a card row (whose retry is the card's own explicit Retry). An intent left
``pending``/``signin`` by a lost response or a kill is re-sent with its FROZEN
word in the next signed batch (a re-send is safe: the hub answers
``request_already_decided`` and that settles the ask). A request the hub
already decided but whose ask is still open locally is settled from the
decision ledger on the next list read (:func:`reconcile_with_ledger`). The
list never POSTs: the intent ledger's law is that no read, poll or boundary
re-sends a decision — only a human-signed act does.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import secrets
import threading
import time
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

logger = logging.getLogger(__name__)

#: The batch bound the hub enforces (§10.2), re-stated where the route can
#: refuse BEFORE any lookup; the client re-checks it (``BATCH_DECIDE_MAX``).
BATCH_MAX = 50

#: Hub per-item codes after which the request can never be decided again, so
#: the local ask is moot and is settled too. Transient codes
#: (``bridge_disabled``, ``messaging_unavailable``) leave the ask open: the
#: request is still decidable and the human may retry.
TERMINAL_ITEM_ERRORS = frozenset(
    {"request_already_decided", "request_withdrawn", "request_revoked", "request_expired"}
)

#: The fold statuses that mean "the human already answered this on the card":
#: such a request is excluded from the pending list (decision in flight or
#: recorded) and refused per item by the batch.
_ANSWERED_STATUSES = frozenset({"answered", "late"})

#: The fold statuses an answer is still admissible in (a timed-out ask leaves
#: the request pending server-side, §10.3; an answer folds to ``late``).
_DECIDABLE_STATUSES = frozenset({"open", "timed_out"})

#: Who settled the ask, as recorded on the ``withdrawn`` row's ``by``.
SETTLED_BY = {"surface": "desktop", "via": "team_bridge_batch_decide"}

_NAME_MAX = 120
_CONTROL = re.compile(r"[\x00-\x1f\x7f-\x9f]")


@dataclass(frozen=True)
class BridgeAsk:
    """One local bridge ask, located by the request's message id."""

    session_id: str
    ask_id: str
    status: str
    #: The question text the ask card shows: ``<sender>: <excerpt>``, built by
    #: ``asks_bridge.bridge_request_question`` from the SCREENED excerpt at
    #: issuance (flattened, bounded, tag-neutralised). Never the raw body.
    text: str = ""


def _bridge_message_ids(row: Mapping[str, Any]) -> list[str]:
    """The ``question_id`` carriage of one ask row (the bridge predicate)."""

    return [
        str(question.get("question_id") or "")
        for question in (row.get("questions") or ())
        if isinstance(question, Mapping) and str(question.get("question_id") or "")
    ]


def bridge_asks_by_message(config_dir: Path | str) -> dict[str, BridgeAsk]:
    """Map each request message id to its most relevant local ask.

    Read from the derived cross-session index (``asks.store.index_asks``) —
    one file read per session with ask activity, no runtime. A request can have
    more than one ask over time ("Review again" re-raises, §10.3); a decidable
    one wins over an answered one, which wins over anything else, newest first
    within each (the index is already newest-first). The decide path re-checks
    the winner against the session's LOG, so a stale index row can only cost a
    per-item refusal, never authorize anything.
    """

    from local_operator.asks import store as asks_store

    rank = {status: 0 for status in _DECIDABLE_STATUSES}
    rank.update({status: 1 for status in _ANSWERED_STATUSES})
    found: dict[str, BridgeAsk] = {}
    for row in asks_store.index_asks(config_dir):
        status = str(row.get("status") or "")
        texts = {
            str(q.get("question_id") or ""): str(q.get("question") or "")
            for q in (row.get("questions") or ())
            if isinstance(q, Mapping)
        }
        for message_id in _bridge_message_ids(row):
            candidate = BridgeAsk(
                session_id=str(row.get("session_id") or ""),
                ask_id=str(row.get("ask_id") or ""),
                status=status,
                text=texts.get(message_id, ""),
            )
            current = found.get(message_id)
            if current is None or rank.get(status, 2) < rank.get(current.status, 2):
                found[message_id] = candidate
    return found


def _clean_name(value: Any) -> str | None:
    """A display name bounded to one line; the hub already sanitized it (§3.2)."""

    if not isinstance(value, str):
        return None
    text = _CONTROL.sub(" ", unicodedata.normalize("NFC", value)).strip()
    return text[:_NAME_MAX] or None


def _mapping(value: Any) -> Mapping[str, Any]:
    """A wire sub-object, or an empty one: a missing or odd block reads as absent."""

    return value if isinstance(value, Mapping) else {}


def _str_or_none(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def request_rows(
    items: Sequence[Any], asks: Mapping[str, BridgeAsk], *, tenant_id: str
) -> list[dict[str, Any]]:
    """Project hub request items to REFERENCE fields only.

    Allowlisted: ids, the request lifecycle (state, expiry), the server-resolved
    sender identity (§3.2: account name + email, never the endpoint's free-text
    label) and the local ask reference the UI joins to the ask card. The message
    BODY is never read — it is untrusted third-party text, and the screened
    excerpt a human decides on already rides the ask card — and no screening
    rationale exists on this path to leak. Items for another tenant, items with
    NO tenant (the projection is a fence, so absence fails closed — review S-4)
    and items whose local ask is already answered are dropped.

    ``excerpt`` (review S-2): on a DECIDABLE row only, the ask card's own
    question text — ``<sender>: <screened excerpt>`` as issued — so a batch is
    never blind: two requests from one sender are told apart by what they ask.
    It comes from the LOCAL ask, which was screened before it was raised; the
    hub's body is still never read here. Non-decidable rows carry ``None``.
    """

    rows: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, Mapping):
            continue
        message = _mapping(item.get("message"))
        thread = _mapping(item.get("thread"))
        sender = _mapping(item.get("sender"))
        message_id = _str_or_none(message.get("id"))
        if message_id is None:
            continue
        item_tenant = _str_or_none(message.get("tenant_id")) or _str_or_none(
            thread.get("tenant_id")
        )
        if item_tenant != tenant_id:
            continue
        request = _mapping(message.get("request"))
        ask = asks.get(message_id)
        if ask is not None and ask.status in _ANSWERED_STATUSES:
            continue
        decidable = ask is not None and ask.status in _DECIDABLE_STATUSES
        rows.append(
            {
                "message_id": message_id,
                "thread_id": _str_or_none(message.get("thread_id"))
                or _str_or_none(thread.get("id")),
                "created_at": _str_or_none(message.get("created_at")),
                "title": None,  # the hub serves no title; the body is never read here
                "excerpt": (ask.text or None) if decidable and ask is not None else None,
                "request": {
                    "state": _str_or_none(request.get("state")),
                    "expires_at": _str_or_none(request.get("expires_at")),
                },
                "sender": {
                    "account_id": _str_or_none(sender.get("account_id")),
                    "endpoint_id": _str_or_none(sender.get("endpoint_id")),
                    "name": _clean_name(sender.get("name")),
                    "email": _str_or_none(sender.get("email")),
                },
                "ask": (
                    {"session_id": ask.session_id, "ask_id": ask.ask_id, "status": ask.status}
                    if ask is not None
                    else None
                ),
                "decidable": decidable,
            }
        )
    return rows


def ledger_rows(items: Sequence[Any], *, tenant_id: str) -> list[dict[str, Any]]:
    """Project decision-ledger rows (§10.4) to references: who, what, decision, when.

    The decider's ``note`` and the request ``body_hash`` stay on the hub; the
    ``decided_via`` claim is passed through and labelled CLIENT-ASSERTED by the
    field name the UI renders it under. A row naming ANOTHER tenant is dropped
    (review S-3): the hub's tenant scoping is the control, this is the same
    defence-in-depth fence the list applies. The ledger wire does not always
    carry ``tenant_id``, so only a present, different value drops a row.
    """

    rows: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, Mapping):
            continue
        row_tenant = _str_or_none(item.get("tenant_id"))
        if row_tenant is not None and row_tenant != tenant_id:
            continue
        request_id = _str_or_none(item.get("request_id"))
        if request_id is None:
            continue
        ask_ref = _mapping(item.get("ask_ref"))

        def person(raw: Any) -> dict[str, Any]:
            raw = _mapping(raw)
            return {
                "account_id": _str_or_none(raw.get("account_id")),
                "name": _clean_name(raw.get("name")),
                "email": _str_or_none(raw.get("email")),
            }

        rows.append(
            {
                "request_id": request_id,
                "thread_id": _str_or_none(item.get("thread_id")),
                "decision": _str_or_none(item.get("state")),
                "decided_at": _str_or_none(item.get("decided_at")),
                "requested_by": person(item.get("requested_by")),
                "decided_by": person(item.get("decided_by")),
                "decided_via_claimed": _str_or_none(item.get("decided_via")),
                "ask_id": _str_or_none(ask_ref.get("ask_id")),
            }
        )
    return rows


def _verified_ask(
    config_dir: Path, ask: BridgeAsk, message_id: str
) -> tuple[str, Mapping[str, Any] | None]:
    """Re-read ONE ask from its session's LOG: ``("", scope)`` or ``(code, None)``.

    The index located it; the log is the truth. Exactly one question must carry
    this ``question_id`` and an ORIGINAL ``bridge`` scope block (a scopeless
    legacy ask is non-authorizing, as everywhere else on the decide path).
    """

    from local_operator.asks import store as asks_store

    events = asks_store.read_events(asks_store.session_dir(config_dir, ask.session_id))
    record = next(
        (
            row
            for row in asks_store.fold(events, asks_store.now_ms(), session_id=ask.session_id)
            if str(row.get("ask_id")) == ask.ask_id
        ),
        None,
    )
    if record is None:
        return "no_open_ask", None
    status = str(record.get("status") or "")
    if status in _ANSWERED_STATUSES:
        return "answered_on_card", None
    if status not in _DECIDABLE_STATUSES:
        return "no_open_ask", None
    matched = [
        q
        for q in (record.get("questions") or ())
        if isinstance(q, Mapping) and str(q.get("question_id") or "") == message_id
    ]
    if len(matched) != 1 or not isinstance(matched[0].get("bridge"), Mapping):
        return "owner_changed", None
    return "", dict(matched[0]["bridge"])


@dataclass(frozen=True)
class Selection:
    """One explicitly selected item, as the human submitted it."""

    message_id: str
    word: str  # "approve" | "deny"


# -- the operator presence proof over the exact selection (review S-1) --------
#
# WHY THESE PRIMITIVES AND NO NEW CRYPTO. The runtime's card path already has
# the whole mechanism: a daemon-minted single-use challenge, the operator key
# signing ``operator.verify.signed_message(action, session_id, request_id,
# challenge)`` (``operator.sign.sign_challenge`` — the call that raises the OS
# presence prompt), and ``operator.verify.signature_verdict`` against the
# root-owned anchor. The batch reuses all three verbatim. What it binds:
#
# * ``action`` = ``decide`` (the bridge decide class, ``harness/approval``);
# * ``session_id`` = :data:`SIGNING_SESSION`, a fixed label no real session id
#   can equal, because a batch spans sessions and is not one session's act;
# * ``request_id`` = ``batch:<selection digest>`` — the digest covers the
#   account, the tenant and the SORTED ``(message_id, word)`` pairs, so a
#   signature over one selection cannot carry another;
# * ``challenge`` = minted HERE, kept server-side with the digest it was minted
#   for, popped on first use (pop-then-verify, the runtime's replay rule).
#
# A runtime's challenge and this one live in different stores, so a signature
# minted for either can never be presented to the other.

#: The fixed signing "session" of a batch decide (see the block above).
SIGNING_SESSION = "team-bridge:approvals-batch"

#: The signed action class: a bridge decide (both words are authority-bearing,
#: ``harness.approval.frame_authority``).
SIGNING_ACTION = "decide"

#: The challenge lifetime. The runtime's own value and rationale
#: (``session/runtime/server._CHALLENGE_TTL_S``, design rev. 2 §2.3: 30 s, one
#: human gesture); expiry is checked at USE, so a slow prompt is refused and the
#: surface asks again rather than a stale challenge being honoured.
CHALLENGE_TTL_S = 30.0

#: Unspent challenges held at once, the runtime's ``_MAX_LIVE_CHALLENGES``: a
#: bearer holder can mint in a loop (minting prompts nobody), so the store is
#: bounded; expired entries are pruned first, so a real caller is never refused.
MAX_LIVE_CHALLENGES = 64

#: The selection-digest schema; bump if the canonical form ever changes.
_DIGEST_VERSION = 1


class PresenceRefused(Exception):
    """A batch refused for want of a valid operator presence proof."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class _Minted:
    digest: str
    account_id: str
    tenant_id: str
    expires_at: float  # time.monotonic()


_challenges: dict[str, _Minted] = {}
_challenges_lock = threading.Lock()


def selection_digest(*, account_id: str, tenant_id: str, selection: Sequence[Selection]) -> str:
    """SHA-256 over the canonical selection: what one signature authorises.

    Canonical JSON (sorted keys, no whitespace) with the pairs SORTED by message
    id, so the submitted order does not change the digest but any added,
    dropped or flipped item does.
    """

    canonical = {
        "v": _DIGEST_VERSION,
        "account_id": account_id,
        "tenant_id": tenant_id,
        "items": sorted([item.message_id, item.word] for item in selection),
    }
    raw = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def signing_request_id(digest: str) -> str:
    """The ``request_id`` the signed message binds for one selection."""

    return f"batch:{digest}"


def presence_anchor() -> Any:
    """This host's operator anchor, ONLY when it is presence-gated.

    ``operator-presence`` is the one level at which a signature costs a human
    gesture; ``operator-file-only`` lets any same-uid process sign, and with no
    usable anchor there is nothing to verify against — both refuse with
    ``operator_presence_required`` before any challenge is minted or prompt
    raised (the per-item card path stays available there).
    """

    from local_operator import operator as operator_authority

    level, loaded = operator_authority.authority_level_load()
    if level != operator_authority.LEVEL_OPERATOR_PRESENCE or loaded.anchor is None:
        raise PresenceRefused("operator_presence_required")
    return loaded.anchor


def mint_challenge(*, account_id: str, tenant_id: str, digest: str) -> str:
    """Mint one single-use challenge bound to this selection digest."""

    now = time.monotonic()
    with _challenges_lock:
        for key in [k for k, v in _challenges.items() if v.expires_at < now]:
            _challenges.pop(key, None)
        if len(_challenges) >= MAX_LIVE_CHALLENGES:
            raise PresenceRefused("operator_challenge_limit")
        challenge = secrets.token_hex(32)
        _challenges[challenge] = _Minted(
            digest=digest,
            account_id=account_id,
            tenant_id=tenant_id,
            expires_at=now + CHALLENGE_TTL_S,
        )
    return challenge


def sign_selection(config_root: Path, *, challenge: str, digest: str) -> dict[str, str]:
    """Sign one minted challenge with the operator key. THIS CALL PROMPTS.

    The same ``sign_challenge`` the CLI's ``lop operator sign`` and the card
    path's ``attach_client`` call: on a presence host the OS shows its Touch ID
    sheet and blocks until the person answers, so callers run it off the event
    loop. A dismissed sheet or a missing key is ``operator_signature_unavailable``.
    """

    from local_operator.operator.keychain import KeyBackendError
    from local_operator.operator.sign import sign_challenge

    try:
        signature = sign_challenge(
            challenge=challenge,
            purpose=SIGNING_ACTION,
            config_root=config_root,
            session_id=SIGNING_SESSION,
            request_id=signing_request_id(digest),
        )
    except KeyBackendError:
        raise PresenceRefused("operator_signature_unavailable") from None
    return signature.as_json()


def verify_selection_proof(
    proof: Mapping[str, Any] | None,
    *,
    account_id: str,
    tenant_id: str,
    selection: Sequence[Selection],
) -> None:
    """Admit the batch only on a valid presence signature over THIS selection.

    Raises :class:`PresenceRefused`: ``operator_presence_required`` (host
    level), ``operator_signature_required`` (no proof), ``operator_signature_invalid``
    (unknown/spent challenge, a different selection, account or tenant, or a
    signature the anchor does not verify) or ``operator_signature_expired``.
    The challenge is POPPED before verification, so every proof has exactly one
    use whether it verifies or not (the runtime's replay rule).
    """

    from local_operator.operator.verify import signature_verdict

    anchor = presence_anchor()
    if not isinstance(proof, Mapping) or not proof:
        raise PresenceRefused("operator_signature_required")
    challenge = str(proof.get("challenge") or "")
    with _challenges_lock:
        minted = _challenges.pop(challenge, None)
    if minted is None:
        raise PresenceRefused("operator_signature_invalid")
    if time.monotonic() > minted.expires_at:
        raise PresenceRefused("operator_signature_expired")
    digest = selection_digest(account_id=account_id, tenant_id=tenant_id, selection=selection)
    if (minted.digest, minted.account_id, minted.tenant_id) != (digest, account_id, tenant_id):
        raise PresenceRefused("operator_signature_invalid")
    verdict = signature_verdict(
        action=SIGNING_ACTION,
        session_id=SIGNING_SESSION,
        request_id=signing_request_id(digest),
        challenge=challenge,
        signature_hex=proof.get("sig"),
        operator_spki=anchor.spki,
        operator_key_id=str(proof.get("key_id") or ""),
        operator_cert=None,
        device_spki=None,
        now=int(time.time()),
    )
    if verdict is not True:
        raise PresenceRefused("operator_signature_invalid")


# -- the batch, recorded before it is sent (review R1-1) ----------------------

#: The ``consumption_row_id`` prefix that marks a decide-intent row as a BATCH
#: row. Card rows carry the ask's response-row id instead, and keep their own
#: explicit-Retry rule; only batch rows are re-sent by a later signed batch.
BATCH_INTENT_PREFIX = "batch:"

_RETRY_POOL = ("pending", "signin")


def _owner_key(owner: Any) -> str:
    """``BridgeRuntime.key()``'s construction: the intent ledger's owner key."""

    from .decide_intents import scope_owner_key

    return scope_owner_key(
        {
            "base_url": getattr(owner, "canonical_base_url", ""),
            "account_id": getattr(owner, "account_id", ""),
            "tenant_id": getattr(owner, "tenant_id", ""),
        }
    )


def _runtime_view(config_dir: Path, owner: Any, session_id: str) -> Any:
    """A runtime-shaped view of one ask's session for the shared fences."""

    return SimpleNamespace(
        config_dir=config_dir,
        owner=owner,
        tenant_id=str(owner.tenant_id),
        session_id=session_id,
    )


def _batch_leftovers(intents: Any, owner_key: str) -> list[Any]:
    """Batch intents a lost response or a kill left unconfirmed, oldest first."""

    return [
        intent
        for intent in intents.pending(owner_key)
        if intent.consumption_row_id.startswith(BATCH_INTENT_PREFIX)
    ]


async def _settle_and_cancel(config_dir: Path, ask: BridgeAsk) -> bool:
    settled = settle_ask(config_dir, ask)
    if settled:
        from local_operator.asks.dismiss import _cancel_deadline_wake

        try:
            # The ask's hidden deadline row would otherwise fire an
            # ask_timeout turn for a request that is already decided.
            await _cancel_deadline_wake(config_dir, ask.session_id, ask.ask_id)
        except Exception:  # noqa: BLE001 — the decision stands; a stale
            # wake row only fires a notice for a withdrawn ask
            logger.warning("approvals: deadline wake not cancelled", exc_info=True)
    return settled


def _receipt_status(intent: Any) -> str | None:
    try:
        receipt = json.loads(intent.receipt_json or "null")
    except ValueError:
        return None
    return _str_or_none(receipt.get("status")) if isinstance(receipt, Mapping) else None


async def decide_selection(
    client: Any,
    selection: Sequence[Selection],
    *,
    config_dir: Path,
    owner: Any,
    access: Any,
    expected_account_id: str,
    batch_ref: str,
) -> dict[str, list[dict[str, Any]]]:
    """Decide the human's explicit selection; per-item results in submitted order.

    The CALLER has already verified the presence proof over exactly this
    selection (:func:`verify_selection_proof`). Returns ``{"results",
    "recovered"}``. Each result is ``{message_id, decision, sent, ask_settled,
    status|error}``: ``status`` is the hub's ``approved|denied``; ``error`` is a
    hub code, or a LOCAL refusal (``no_open_ask``, ``answered_on_card``,
    ``owner_changed``, ``not_designated``, ``feature_off``,
    ``consent_revoked``, ``earlier_decision_unconfirmed``) for an item that was
    never sent. ``sent`` is True for every item that went out, including one
    the hub's answer omitted (review R1-2: it may be committed). ``recovered``
    reports batch intents from an earlier lost response, re-sent here with
    their FROZEN word in the same single hub call.
    """

    from .authority import authority_fence, scope_matches_runtime
    from .client import BridgeRefusal, BridgeSignInRequired
    from .decide import decide_batch_from_human_selection, human_decision
    from .decide_intents import (
        DecideIntents,
        DecideIntentsRefused,
        _validated_for_transport,
    )

    asks = bridge_asks_by_message(config_dir)
    intents = DecideIntents(config_dir)
    owner_key = _owner_key(owner)
    consumed_at = datetime.now(UTC).isoformat()
    results: dict[str, dict[str, Any]] = {}
    # message_id -> (ask, intent, recovered?) for everything that goes out.
    outgoing: dict[str, tuple[BridgeAsk | None, Any, bool]] = {}
    for item in selection:
        result: dict[str, Any] = {
            "message_id": item.message_id,
            "decision": item.word,
            "sent": False,
            "ask_settled": False,
        }
        results[item.message_id] = result
        ask = asks.get(item.message_id)
        if ask is None:
            result["error"] = "no_open_ask"
            continue
        code, scope = _verified_ask(config_dir, ask, item.message_id)
        if code or scope is None:
            result["error"] = code or "owner_changed"
            continue
        # The SAME fence + original-scope equality the single decide path runs
        # (decide_intents), against a runtime-shaped view of the ask's session.
        runtime = _runtime_view(config_dir, owner, ask.session_id)
        refusal = authority_fence(runtime)
        if refusal is not None:
            result["error"] = str(refusal.get("code") or "owner_changed")
            continue
        if not scope_matches_runtime(scope, runtime):
            result["error"] = "owner_changed"
            continue
        # DURABLE BEFORE THE POST (R1-1): first record wins, exactly as on the
        # card path, so a re-selection never mints a second word for one ask.
        intent, created = intents.record(
            owner_key,
            ask_id=ask.ask_id,
            question_id=item.message_id,
            word=item.word,
            generation=str(scope.get("generation") or ""),
            scope_json=json.dumps(dict(scope), sort_keys=True),
            consumption_row_id=batch_ref,
            consumed_at=consumed_at,
        )
        if not created and intent.state in ("sent", "refused"):
            # The hub already answered this item; only the local settle was
            # lost. Settle now; nothing is sent again.
            result["sent"] = True
            status = _receipt_status(intent) if intent.state == "sent" else None
            if status is not None:
                result["status"] = status
            else:
                result["error"] = str(intent.error_code or "request_already_decided")
            result["ask_settled"] = await _settle_and_cancel(config_dir, ask)
            continue
        if not created and intent.word != item.word:
            # An earlier signed word for this ask is still unconfirmed; it is
            # re-sent below and wins (first record wins), so this one is not.
            result["error"] = "earlier_decision_unconfirmed"
            continue
        outgoing[item.message_id] = (ask, intent, False)

    recovered: dict[str, dict[str, Any]] = {}
    for intent in _batch_leftovers(intents, owner_key):
        if intent.question_id in outgoing:
            continue  # re-selected with the same word: it rides as a result
        if len(outgoing) >= BATCH_MAX:
            break  # the rest wait for the next signed batch
        ask = asks.get(intent.question_id)
        session_id = ask.session_id if ask is not None and ask.ask_id == intent.ask_id else ""
        try:
            scope = json.loads(intent.scope_json or "{}")
            session_id = session_id or str(scope.get("session_id") or "")
            _validated_for_transport(intent, _runtime_view(config_dir, owner, session_id))
        except (DecideIntentsRefused, ValueError, AttributeError):
            intents.mark_refused(intent.intent_id, "owner_changed")
            continue
        recovered[intent.question_id] = {
            "message_id": intent.question_id,
            "decision": intent.word,
            "sent": False,
            "ask_settled": False,
        }
        located = ask if ask is not None and ask.ask_id == intent.ask_id else None
        outgoing[intent.question_id] = (located, intent, True)

    if not outgoing:
        return {
            "results": [results[item.message_id] for item in selection],
            "recovered": list(recovered.values()),
        }

    eligible = [
        human_decision(message_id=mid, ask_id=intent.ask_id, word=intent.word)
        for mid, (_ask, intent, _rec) in outgoing.items()
    ]
    try:
        answer = await decide_batch_from_human_selection(
            client,
            eligible,
            tenant_id=str(owner.tenant_id),
            expected_account_id=expected_account_id,
            access=access,
        )
    except BridgeSignInRequired:
        for _ask, intent, _rec in outgoing.values():
            intents.mark_signin(intent.intent_id)
        raise
    except BridgeRefusal as exc:
        # Whole-call failure (unreachable, timeout, a whole-batch refusal): the
        # hub may or may not have committed. The rows stay in the retry pool and
        # the next signed batch re-sends them; a committed one answers
        # ``request_already_decided`` there, and the list's ledger read settles
        # its ask before that.
        code = str(getattr(exc, "code", "") or "bridge_unreachable")
        for _ask, intent, _rec in outgoing.values():
            intents.mark_transient(intent.intent_id, code)
        raise

    answered: set[str] = set()
    for outcome in answer.get("results") or ():
        if not isinstance(outcome, Mapping):
            continue
        message_id = str(outcome.get("message_id") or "")
        if message_id not in outgoing or message_id in answered:
            continue  # the hub cannot add an item the human did not select
        answered.add(message_id)
        ask, intent, is_recovered = outgoing[message_id]
        result = (recovered if is_recovered else results)[message_id]
        result["sent"] = True
        status = _str_or_none(outcome.get("status"))
        error = _str_or_none(outcome.get("error"))
        if status is not None:
            result["status"] = status
            intents.mark_sent(intent.intent_id, dict(outcome))
        else:
            result["error"] = error or "messaging_unavailable"
            if error in TERMINAL_ITEM_ERRORS:
                intents.mark_refused(intent.intent_id, str(error))
            else:
                # Transient per item (``bridge_disabled``,
                # ``messaging_unavailable``, unknown): the request is still
                # decidable, so the ask stays open and the intent stays pending.
                intents.mark_transient(intent.intent_id, str(error or "messaging_unavailable"))
                continue
        if ask is not None and ask.status in _DECIDABLE_STATUSES:
            result["ask_settled"] = await _settle_and_cancel(config_dir, ask)
    for message_id, (_ask, intent, is_recovered) in outgoing.items():
        if message_id in answered:
            continue
        # Sent, but the hub's answer omitted it (R1-2): it may be committed, so
        # ``sent`` stays True and the intent stays pending for the next batch.
        result = (recovered if is_recovered else results)[message_id]
        result["sent"] = True
        result["error"] = "messaging_unavailable"
    return {
        "results": [results[item.message_id] for item in selection],
        "recovered": list(recovered.values()),
    }


async def reconcile_with_ledger(
    config_dir: Path, owner: Any, ledger_items: Sequence[Any]
) -> list[str]:
    """Settle local asks for requests the decision ledger says are DECIDED.

    The lost-response half of R1-1 that needs no POST: the hub committed a
    decision (any surface, this account) but the local ask is still open, so it
    is invisible to the pending list yet still card-answerable. Each ledger row
    (tenant-fenced) whose request has a decidable local ask, verified against
    the log and naming THIS owner's account and tenant in its original scope,
    gets the ``withdrawn`` settle; a batch intent still pending for it is marked
    sent with the ledger row as its receipt. Returns the settled message ids.
    """

    from .decide_intents import DecideIntents

    tenant_id = str(owner.tenant_id)
    asks = bridge_asks_by_message(config_dir)
    intents = DecideIntents(config_dir)
    owner_key = _owner_key(owner)
    pending = {intent.question_id: intent for intent in _batch_leftovers(intents, owner_key)}
    settled: list[str] = []
    for item in ledger_items:
        if not isinstance(item, Mapping):
            continue
        request_id = _str_or_none(item.get("request_id"))
        row_tenant = _str_or_none(item.get("tenant_id"))
        if request_id is None or (row_tenant is not None and row_tenant != tenant_id):
            continue
        ask = asks.get(request_id)
        if ask is None or ask.status not in _DECIDABLE_STATUSES:
            continue
        code, scope = _verified_ask(config_dir, ask, request_id)
        if code or scope is None:
            continue
        if (str(scope.get("account_id") or ""), str(scope.get("tenant_id") or "")) != (
            str(owner.account_id),
            tenant_id,
        ):
            continue
        intent = pending.get(request_id)
        if intent is not None and intent.ask_id == ask.ask_id:
            intents.mark_sent(
                intent.intent_id,
                {"message_id": request_id, "status": _str_or_none(item.get("state"))},
            )
        if await _settle_and_cancel(config_dir, ask):
            settled.append(request_id)
    return settled


def has_batch_leftovers(config_dir: Path, owner: Any) -> bool:
    """Whether any batch intent for this owner is still unconfirmed."""

    from .decide_intents import DecideIntents

    return bool(_batch_leftovers(DecideIntents(config_dir), _owner_key(owner)))


def settle_ask(config_dir: Path | str, ask: BridgeAsk) -> bool:
    """Append the ``withdrawn`` row that retires a decided request's ask.

    Safe with or without a live owner: the append is one ``O_APPEND`` write
    under the store's flock, and every reader (the live queue included) re-folds
    the log from disk, so a live runtime refuses a later card answer and its next
    index refresh agrees. The derived index row is updated here too so the
    cross-session views stop counting it now; the log stays the truth.
    """

    from local_operator.asks import store as asks_store

    directory = asks_store.session_dir(config_dir, ask.session_id)
    now = asks_store.now_ms()
    payload = {
        "v": asks_store.EVENT_SCHEMA,
        "kind": asks_store.EVENT_WITHDRAWN,
        "ask_id": ask.ask_id,
        "at": now,
        "by": dict(SETTLED_BY),
    }
    if not asks_store.append_event(directory, payload):
        logger.warning("approvals: could not settle ask %s in %s", ask.ask_id, ask.session_id)
        return False
    try:
        entry = asks_store.read_entry(config_dir, ask.session_id)
        if entry is not None:
            rows = [dict(row) for row in entry.get("asks") or ()]
            touched = False
            for row in rows:
                if str(row.get("ask_id")) == ask.ask_id:
                    row["status"] = asks_store.STATUS_WITHDRAWN
                    touched = True
            if touched:
                asks_store.write_entry(
                    config_dir, ask.session_id, cwd=str(entry.get("cwd") or ""), asks=rows
                )
    except Exception:  # noqa: BLE001 — the log is the truth; the index heals
        logger.warning("approvals: index rewrite failed for %s", ask.session_id, exc_info=True)
    return True


def parse_selection(raw: Iterable[Mapping[str, Any]]) -> list[Selection]:
    """The validated, de-duplicated selection, or ``ValueError`` (whole refusal)."""

    selection: list[Selection] = []
    seen: set[str] = set()
    for item in raw:
        message_id = str(item.get("message_id") or "")
        word = str(item.get("decision") or "")
        if word not in ("approve", "deny"):
            raise ValueError("decision is approve | deny")
        if message_id in seen:
            raise ValueError("a message id appears twice in one selection")
        seen.add(message_id)
        selection.append(Selection(message_id=message_id, word=word))
    if not selection:
        raise ValueError("a selection carries at least one item")
    if len(selection) > BATCH_MAX:
        raise ValueError(f"a selection carries at most {BATCH_MAX} items")
    return selection


__all__ = [
    "BATCH_MAX",
    "SETTLED_BY",
    "TERMINAL_ITEM_ERRORS",
    "BridgeAsk",
    "Selection",
    "bridge_asks_by_message",
    "decide_selection",
    "ledger_rows",
    "parse_selection",
    "request_rows",
    "settle_ask",
]
