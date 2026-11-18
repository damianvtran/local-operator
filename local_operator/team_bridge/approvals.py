"""The desktop approvals surface: pending requests, human batch decide, the ledger.

WHY THIS MODULE EXISTS. Requirement: approvals as a first-class flow — a
teammate's agent sends a ``request``, the operator sees "N requests waiting",
approves or denies several at once, and every decision is audited. The hub has
all four routes (``GET /messaging/requests``, the single and batch decide,
``GET /messaging/decisions``) and this package already had the executor
(:func:`~local_operator.team_bridge.decide.decide_batch_from_human_selection`),
but nothing called it. This module is the harness half the desktop routes in
``server/routes/desktop_team_bridge.py`` call; it owns no transport of its own.

THE TRUST BOUNDARY — THE DESKTOP BEARER PLUS AN EXPLICIT SELECTION, NO OS
PROMPT. "Answered by the human" is CLIENT-ASSERTED (TEAM-BRIDGE.md §10.1): the
server checks the receipt's shape and binding, never that a person tapped. An
ordinary approval is authorized here by the desktop bearer (``require_desktop``)
and by the human naming every item and its word; it is audited by the durable
intent ledger and the hub's decision ledger. It deliberately raises NO Touch ID
or other OS prompt, and needs no biometric hardware.

WHY THE PRESENCE SIGNATURE WAS REMOVED (decision a-856f, 2026-11-15). Until
then every batch carried an operator presence signature over the exact
selection, minted and signed through ``…/requests/decide/authorize`` (an OS
Touch ID sheet per decision). The operator ruled that out: "I don't want to have
to approve every action with Touch ID that's a terrible user experience" — and a
gate that needs biometrics also blocks every Windows, Linux or keyless Mac host.
What the gate bought did not justify that cost:

- it never covered same-user processes (PR #4099 security round 2, S-9): the
  hub's OAuth tokens sit in ``~/.local-operator/auth.db`` (0600, no keychain),
  and any same-uid process can POST the hub's decide directly with no gesture.
  It bound callers of the daemon API only.
- the gesture proved presence, never informed consent: the OS sheet cannot
  carry this project's copy and showed no selection.

THE TRADE-OFF, STATED (for security review). The bearer is NOT a same-uid
boundary: ``agent_shell.DAEMON_BEARER_ENVS`` keeps it out of an agent shell's
ENVIRONMENT, but the desktop persists it in a 0600 file any ordinary same-uid
shell can read (security round 1, S-1 — measured). A same-uid process holding
it can therefore decide a request the operator has a local ask for — exactly
what it could already do with ``auth.db``. The residual is unchanged in kind;
closing it for real needs the HUB to verify an operator signature on every
human decision (a protocol follow-up). Presence signing stays for
authority-RAISING operator actions (``operator.sign`` — e.g. loosening a gate),
which are not ordinary approvals.

The mechanism:

- this module is imported ONLY by the desktop route family (behind
  ``require_desktop``). No model tool, agent loop or connector imports it; a
  test pins that, so no model or agent path can reach the batch decide
  in-process; an out-of-process caller still needs the bearer (S-9 above is
  the named residual).
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

ONE WORD PER ASK (PR #4099 design (c); replaces round 1's implicit carriage).
The word for an ask is fixed by the FIRST human act that records an intent,
and an intent is sent only by an act on the surface that recorded it: a card
grant for that ask, or a batch decide that names that ``(message_id, word)``
pair. Any later human act naming the other word is
refused BEFORE it writes anything — ``/authorize`` and ``/decide`` answer 409
``earlier_decision_unconfirmed`` (:func:`conflicting_selection`) and the card
door refuses ``decided_from_list`` (``Session.bridge_decide``). An unconfirmed
batch word is NEVER carried by a later batch that does not select it (round 2
sent such a leftover under a digest that did not name it — S-7/R2-1); the list
row shows it as ``unsent_decision`` instead, and only that word can be selected.

ONE RACE REMAINS, narrowed and accepted (review R1-5, re-stated for round 2).
The card path checks the fold and then appends ``answered``; the batch records
its intent and then re-checks the fold just before the POST
(:func:`decide_selection`). Those two check-then-act sequences can still
interleave by milliseconds: a card answer whose check ran before the batch's
record, and whose append lands after the batch's re-check, injects its word
locally while the hub receives the batch's (the card's own consumption then
finds the batch's intent row and sends nothing). The hub's one-decision rule
means nothing is decided twice and the audit names the batch; only the
session's local view can disagree. Before round 2's remediation that window
stayed open until the NEXT batch; now it is one human on two surfaces within
the batch's own record→POST gap. Closing it fully means reordering the card
path's append and record steps, which is out of this PR's scope.

CRASH / LOST-RESPONSE SAFETY (review R1-1). Each item is recorded in the SAME
durable decide-intent ledger the card path uses (``decide_intents``) BEFORE the
hub call, tagged ``consumption_row_id = "batch:…"`` so it is never mistaken for
a card row (whose retry is the card's own explicit Retry). An intent left
``pending``/``signin`` by a lost response or a kill resolves one of two ways:
if the hub committed it, the decision ledger settles it on the next list read
(:func:`reconcile_with_ledger`); if not, the row stays decidable with its
frozen word and the human selects it again (an explicit re-send; the
hub answers ``request_already_decided`` for a committed one). Nothing — no
read, poll, boundary or unrelated batch — re-sends a decision on its own.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from .decide_intents import BATCH_CONSUMPTION_PREFIX

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
    items: Sequence[Any],
    asks: Mapping[str, BridgeAsk],
    *,
    tenant_id: str,
    unsent: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Project hub request items to REFERENCE fields only.

    Allowlisted: ids, the request lifecycle (state, expiry), the server-resolved
    sender identity (§3.2: account name + email, never the endpoint's free-text
    label) and the local ask reference the UI joins to the ask card. The message
    BODY is never read — it is untrusted third-party text, and the screened
    excerpt a human decides on already rides the ask card — and no screening
    rationale exists on this path to leak. Items for another tenant, items with
    NO tenant (the projection is a fence, so absence fails closed — review S-4)
    items with NO local ask (screening rejected the request, or has not
    screened it yet — gap 2) and items whose local ask is already answered are
    dropped.

    ``excerpt`` (review S-2): on a DECIDABLE row only, the ask card's own
    question text — ``<sender>: <screened excerpt>`` as issued — so a batch is
    never blind: two requests from one sender are told apart by what they ask.
    It comes from the LOCAL ask, which was screened before it was raised; the
    hub's body is still never read here. Non-decidable rows carry ``None``.

    ``unsent_decision`` (design (c), exit 9): the word of a signed batch the hub
    has not confirmed for this row's ask (``unsent``, from
    :func:`unsent_decisions`), else ``None``. Only that word may be selected;
    the other is refused ``earlier_decision_unconfirmed``.
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
        # NO LOCAL ASK, NO ROW (Team Bridge inbox scope, gap 2). A request
        # appears in Approvals only once this machine raised an ask for it,
        # which happens only after screening admitted it. A hub request
        # screening REJECTED (that lives in Holds, if anywhere) or has not
        # screened YET has no ask; listing it showed the operator a row they
        # could neither decide nor act on, and an unscreened body has no
        # screened excerpt to show. It appears on a later read once screened.
        if ask is None or ask.status in _ANSWERED_STATUSES:
            continue
        decidable = ask.status in _DECIDABLE_STATUSES
        rows.append(
            {
                "message_id": message_id,
                "thread_id": _str_or_none(message.get("thread_id"))
                or _str_or_none(thread.get("id")),
                "created_at": _str_or_none(message.get("created_at")),
                "title": None,  # the hub serves no title; the body is never read here
                "excerpt": (ask.text or None) if decidable else None,
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
                "ask": {"session_id": ask.session_id, "ask_id": ask.ask_id, "status": ask.status},
                "decidable": decidable,
                "unsent_decision": (unsent or {}).get(message_id) if decidable else None,
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


# -- no presence proof: ordinary approvals never prompt (decision a-856f) -----
#
# This block used to hold the batch's operator presence primitives: a minted
# single-use challenge store, a one-prompt-at-a-time gate with a dismissal
# cool-down, ``sign_selection`` (which raised the OS Touch ID sheet) and
# ``verify_selection_proof``. They existed only to put a presence signature on
# every batch, and decision a-856f retired that for ordinary approvals (see the
# module docstring for why, and for the same-uid trade-off). With nothing left
# to sign or verify they were removed rather than left as dead authority code a
# reader could mistake for a live gate. Presence signing for authority-RAISING
# actions lives in ``operator.sign`` and is untouched.

#: The selection-digest schema; bump if the canonical form ever changes.
_DIGEST_VERSION = 1


def selection_digest(*, account_id: str, tenant_id: str, selection: Sequence[Selection]) -> str:
    """SHA-256 over the canonical selection: a stable name for one exact batch.

    No longer a signing input. ``…/decide/authorize`` (the compatibility shim)
    still returns it as ``selection_digest`` so the shipped desktop keeps its
    response shape. Canonical JSON (sorted keys, no whitespace) with the pairs
    SORTED by message id: the submitted order does not change it, but any added,
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


# -- the batch, recorded before it is sent (review R1-1) ----------------------

#: The ``consumption_row_id`` prefix that marks a decide-intent row as a BATCH
#: row (defined in the shared ledger so the card door can see it too). Card
#: rows carry the ask's response-row id instead and keep their own
#: explicit-Retry rule; a batch row is sent only by a later batch that names it.
BATCH_INTENT_PREFIX = BATCH_CONSUMPTION_PREFIX


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


def unsent_decisions(config_dir: Path, owner: Any, asks: Mapping[str, BridgeAsk]) -> dict[str, str]:
    """``{message_id: word}`` for asks whose batch word the hub has not confirmed.

    The list's ``unsent_decision`` (design (c), exit 9): the human must see that
    a row already carries a word, and that only that word can be selected.
    Keyed by the located ask, so a "Review again" re-raise (a new ask id) is a
    fresh decision, as everywhere else in the intent ledger.
    """

    from .decide_intents import DecideIntents

    by_ask = {
        (intent.ask_id, intent.question_id): intent.word
        for intent in _batch_leftovers(DecideIntents(config_dir), _owner_key(owner))
    }
    return {
        message_id: by_ask[(ask.ask_id, message_id)]
        for message_id, ask in asks.items()
        if (ask.ask_id, message_id) in by_ask
    }


def conflicting_selection(
    config_dir: Path, owner: Any, selection: Sequence[Selection]
) -> list[dict[str, str]]:
    """Selected items naming the OTHER word than an unconfirmed batch intent.

    One word per ask (design (c)): the ``/authorize`` shim refuses these too
    (the shipped desktop calls it first), and ``/decide`` refuses the WHOLE batch on them before any
    intent write or hub call. Each entry names the item and the word that is
    still unconfirmed for it: ``{message_id, unsent_decision}``.
    """

    asks = bridge_asks_by_message(config_dir)
    unsent = unsent_decisions(config_dir, owner, asks)
    return [
        {"message_id": item.message_id, "unsent_decision": unsent[item.message_id]}
        for item in selection
        if item.message_id in unsent and unsent[item.message_id] != item.word
    ]


#: Bound on ``client_ref`` (hex), echoed on the published authorize frame.
CLIENT_REF_MAX = 32


def prompt_items(config_dir: Path, selection: Sequence[Selection]) -> list[dict[str, Any]]:
    """What a batch decision is FOR, resolved by the daemon from the LOCAL asks.

    The payload of the ``team_bridge_authorize`` feed frame (S-6), which the
    ``/authorize`` compatibility shim still publishes (as ``signed``, with no
    prompt behind it — decision a-856f) so the shipped desktop keeps naming
    what each confirmation was for. Each item is
    ``{message_id, word, sender, excerpt}``. ``sender`` and ``excerpt`` come
    from the ask card's own line (``<sender>: <screened excerpt>``, composed at
    issuance by ``asks_bridge.bridge_request_question``), split at its first
    ``": "`` — the caller supplies neither, so a caller cannot make a frame
    look like it is for something else. An item with no local ask carries
    ``None`` for both. Never the hub body, a screening rationale or a body hash.
    The line is bounded at issuance, and a selection is at most 50 items.
    """

    asks = bridge_asks_by_message(config_dir)
    items: list[dict[str, Any]] = []
    for item in selection:
        ask = asks.get(item.message_id)
        sender: str | None = None
        excerpt: str | None = None
        if ask is not None and ask.text:
            head, sep, tail = ask.text.partition(": ")
            sender, excerpt = (head, tail) if sep else (None, ask.text)
        items.append(
            {
                "message_id": item.message_id,
                "word": item.word,
                "sender": sender,
                "excerpt": excerpt,
            }
        )
    return items


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

    The CALLER (the desktop route, behind the desktop bearer) has already
    refused the whole batch on a conflicting re-selection
    (:func:`conflicting_selection`). No presence proof is checked: ordinary
    approvals never prompt (decision a-856f, module docstring). Returns
    ``{"results"}``. Each result is ``{message_id, decision, sent,
    ask_settled, status|error}``: ``status`` is the hub's ``approved|denied``;
    ``error`` is a hub code, or a LOCAL refusal (``no_open_ask``,
    ``answered_on_card``, ``owner_changed``, ``not_designated``,
    ``feature_off``, ``consent_revoked``, ``earlier_decision_unconfirmed``) for
    an item that was never sent. ``sent`` is True for every item that went out,
    including one the hub's answer omitted (review R1-2: it may be committed).

    EXACTLY the selected pairs go out (B1/B3). There is no ``recovered`` list
    any more: an earlier unconfirmed batch word is never carried by a batch
    that does not name it (S-7/R2-1); it is re-sent only when the human
    selects it again, with the same word.
    """

    from .authority import authority_fence, scope_matches_runtime
    from .client import BridgeRefusal, BridgeSignInRequired
    from .decide import decide_batch_from_human_selection, human_decision
    from .decide_intents import DecideIntents

    asks = bridge_asks_by_message(config_dir)
    intents = DecideIntents(config_dir)
    owner_key = _owner_key(owner)
    consumed_at = datetime.now(UTC).isoformat()
    results: dict[str, dict[str, Any]] = {}
    # message_id -> (ask, intent) for everything that goes out.
    outgoing: dict[str, tuple[BridgeAsk, Any]] = {}
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
            # BACKSTOP ONLY. The route refuses a conflicting re-selection WHOLE
            # before it reaches here (design (c)); this arm covers a row that
            # appeared between that check and this record (a race, or a card
            # row). The earlier word is NOT sent: only what the human selected
            # goes out, and this item goes nowhere.
            result["error"] = "earlier_decision_unconfirmed"
            continue
        outgoing[item.message_id] = (ask, intent)

    # RE-CHECK THE FOLD BETWEEN RECORD AND POST (design (c), exit 8). A card
    # answer can land after the per-item check above; if it did, its word
    # stands and this item must not reach the hub. The row is refused so it
    # never sits in the unsent pool either. The residual window (a card append
    # landing after THIS check) is the race named in the module docstring.
    for message_id in list(outgoing):
        ask, intent = outgoing[message_id]
        code, _scope = _verified_ask(config_dir, ask, message_id)
        if code == "answered_on_card":
            intents.mark_refused(intent.intent_id, "answered_on_card")
            results[message_id]["error"] = "answered_on_card"
            del outgoing[message_id]

    if not outgoing:
        return {"results": [results[item.message_id] for item in selection]}

    eligible = [
        human_decision(message_id=mid, ask_id=intent.ask_id, word=intent.word)
        for mid, (_ask, intent) in outgoing.items()
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
        for _ask, intent in outgoing.values():
            intents.mark_signin(intent.intent_id)
        raise
    except BridgeRefusal as exc:
        # Whole-call failure (unreachable, timeout, a whole-batch refusal): the
        # hub may or may not have committed. The rows stay unconfirmed: a
        # committed one is settled by the list's ledger read; an uncommitted one
        # stays decidable with its frozen word (``unsent_decision``) until the
        # human selects it again. Nothing re-sends it on its own.
        code = str(getattr(exc, "code", "") or "bridge_unreachable")
        for _ask, intent in outgoing.values():
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
        ask, intent = outgoing[message_id]
        result = results[message_id]
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
        if ask.status in _DECIDABLE_STATUSES:
            result["ask_settled"] = await _settle_and_cancel(config_dir, ask)
    for message_id in outgoing:
        if message_id in answered:
            continue
        # Sent, but the hub's answer omitted it (R1-2): it may be committed, so
        # ``sent`` stays True and the intent stays unconfirmed.
        result = results[message_id]
        result["sent"] = True
        result["error"] = "messaging_unavailable"
    return {"results": [results[item.message_id] for item in selection]}


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
