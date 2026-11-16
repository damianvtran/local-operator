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
and binding, never that a person tapped. The claim is honest only because the
code that drives ``/decide`` is reachable from a human surface and nowhere
else, so:

- this module is imported ONLY by the desktop route family, which sits behind
  ``require_desktop`` (the desktop bearer AND an allowlisted Origin — the same
  door the signed ``bridge_decide`` answer path rides from the ask card). No
  model tool, agent loop or connector imports it; a test pins that.
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
refuses a withdrawn ask). An answer the human already gave on the card
outranks a later withdrawal in the fold, and the hub's one-decision-per-request
rule (``request_already_decided``) backs both orders. The reverse direction is
the list read: a request whose local ask is already answered is not listed,
and once that answer reaches the hub it is no longer pending there either.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
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
        for message_id in _bridge_message_ids(row):
            candidate = BridgeAsk(
                session_id=str(row.get("session_id") or ""),
                ask_id=str(row.get("ask_id") or ""),
                status=status,
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
    rationale exists on this path to leak. Items for another tenant, items that
    are not requests, and items whose local ask is already answered are dropped.
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
        if item_tenant is not None and item_tenant != tenant_id:
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


def ledger_rows(items: Sequence[Any]) -> list[dict[str, Any]]:
    """Project decision-ledger rows (§10.4) to references: who, what, decision, when.

    The decider's ``note`` and the request ``body_hash`` stay on the hub; the
    ``decided_via`` claim is passed through and labelled CLIENT-ASSERTED by the
    field name the UI renders it under.
    """

    rows: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, Mapping):
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


async def decide_selection(
    client: Any,
    selection: Sequence[Selection],
    *,
    config_dir: Path,
    owner: Any,
    access: Any,
    expected_account_id: str,
) -> list[dict[str, Any]]:
    """Decide the human's explicit selection; per-item results in submitted order.

    Each result is ``{message_id, decision, status|error, sent, ask_settled}``:
    ``status`` is the hub's ``approved|denied``; ``error`` is either a hub code
    or a LOCAL refusal (``no_open_ask``, ``answered_on_card``,
    ``owner_changed``, ``not_designated``, ``feature_off``,
    ``consent_revoked``) for an item that was never sent. ``sent`` says which.
    Only items that passed every local check reach ONE batch call.
    """

    from local_operator.asks.dismiss import _cancel_deadline_wake

    from .authority import authority_fence, scope_matches_runtime
    from .decide import decide_batch_from_human_selection, human_decision

    asks = bridge_asks_by_message(config_dir)
    results: dict[str, dict[str, Any]] = {}
    eligible: list[Any] = []
    located: dict[str, BridgeAsk] = {}
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
        if code:
            result["error"] = code
            continue
        # The SAME fence + original-scope equality the single decide path runs
        # (decide_intents), against a runtime-shaped view of the ask's session.
        runtime = SimpleNamespace(
            config_dir=config_dir,
            owner=owner,
            tenant_id=str(owner.tenant_id),
            session_id=ask.session_id,
        )
        refusal = authority_fence(runtime)
        if refusal is not None:
            result["error"] = str(refusal.get("code") or "owner_changed")
            continue
        if not scope_matches_runtime(scope, runtime):
            result["error"] = "owner_changed"
            continue
        eligible.append(
            human_decision(message_id=item.message_id, ask_id=ask.ask_id, word=item.word)
        )
        located[item.message_id] = ask

    if eligible:
        answer = await decide_batch_from_human_selection(
            client,
            eligible,
            tenant_id=str(owner.tenant_id),
            expected_account_id=expected_account_id,
            access=access,
        )
        for outcome in answer.get("results") or ():
            if not isinstance(outcome, Mapping):
                continue
            message_id = str(outcome.get("message_id") or "")
            if message_id not in located:
                continue  # the hub cannot add an item the human did not select
            result = results[message_id]
            result["sent"] = True
            status = _str_or_none(outcome.get("status"))
            error = _str_or_none(outcome.get("error"))
            if status is not None:
                result["status"] = status
            else:
                result["error"] = error or "messaging_unavailable"
            if status is not None or error in TERMINAL_ITEM_ERRORS:
                ask = located[message_id]
                result["ask_settled"] = settle_ask(config_dir, ask)
                if result["ask_settled"]:
                    try:
                        # The ask's hidden deadline row would otherwise fire an
                        # ask_timeout turn for a request that is already decided.
                        await _cancel_deadline_wake(config_dir, ask.session_id, ask.ask_id)
                    except Exception:  # noqa: BLE001 — the decision stands; a stale
                        # wake row only fires a notice for a withdrawn ask
                        logger.warning("approvals: deadline wake not cancelled", exc_info=True)
        for message_id in located:
            result = results[message_id]
            if not result["sent"]:
                # Selected and sent, but the hub's answer omitted it: say so
                # rather than claim a decision nobody confirmed.
                result["error"] = "messaging_unavailable"
    return [results[item.message_id] for item in selection]


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
