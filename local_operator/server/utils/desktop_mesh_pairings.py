"""Mesh pairing cards for the desktop's Mesh approvals (``features.mesh_pairing``).

WHY THIS EXISTS. Pairing two devices needed a terminal on both: the joiner's person
typed ``lop network join --confirm <code>`` and the inviter's ran ``lop network
confirm``. The operator asked that pairing NOT require terminal commands and be done
through the app's Mesh approvals. This module turns the two parked-ceremony records
into card rows and carries a card's answer to the one place that decides it:
- inviter side (``PendingPairing``, written by the relay's pair listener): the answer
  goes to the RELAY as ``net_pair_confirm`` with a ``code``, and the relay compares it
  against its own derivation and spends the invite's attempt budget (``relay.
  _pair_confirm_by_code``) — this module never compares an inviter code itself;
- joiner side (``PendingJoin``, written by the ``join --park`` process): the answer is
  written through ``network.pairing_answers``, the same writer the CLI uses.

READS NEVER WRITE. Listing is folded from the relay (``net_pair_pending
include_stages``) or, when no relay answers, from the files — read-only, unlike
``store.pending_pairings`` which clears expired rows — the same discipline as
``approval_rows``. Nothing here creates a directory on a machine in no network.

THE ANSWER IS GATED BY AN APP ATTESTATION (``network.pairing_attest``), consumed
BEFORE anything is compared, so a bearer-only POST (an agent) is 403
``pairing_needs_app`` having spent no attempt and written no decision.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from local_operator.network import pairing_answers, pairing_attest
from local_operator.network.types import (
    PAIR_STAGE_AWAITING_CONFIRMATION,
    MeshRefusal,
    PendingJoin,
    PendingPairing,
)

#: Joiner card stage once its local match was handed to the parker.
JOINER_WAITING = "waiting_for_other_device"

_OFFER_KEYS = ("key", "label", "kind", "share")


def _bound_store(root: Path | None) -> Any:
    """The store calls ``pairing_answers`` makes, bound to THIS backend's root."""
    from local_operator.network import store

    return SimpleNamespace(
        pending_joins=lambda: store.pending_joins(root),
        join_answer=lambda invite_id: store.join_answer(invite_id, root),
        save_join_answer=lambda answer: store.save_join_answer(answer, root),
    )


def _has_pending_dir(root: Path | None) -> bool:
    from local_operator.network import store

    return store.pending_dir(root).is_dir()


def _inviter_from_files(root: Path | None, now: float) -> list[PendingPairing]:
    """Open inviter records straight from disk, clearing NOTHING (a read)."""
    from local_operator.network import store

    found: list[PendingPairing] = []
    for path in sorted(store.pending_dir(root).glob("*.pending.json")):
        data = store._read_json(path)  # noqa: SLF001 — read-only sibling of pending_pairing
        if not isinstance(data, dict):
            continue
        try:
            row = PendingPairing.from_json(data)
        except (TypeError, ValueError):
            continue
        if row.is_open(now):
            found.append(row)
    return sorted(found, key=lambda row: row.issued_at)


def inviter_records(root: Path | None, *, now: float | None = None) -> list[PendingPairing]:
    """Every open inviter ceremony, both stages: the relay's view, else the files."""
    moment = time.time() if now is None else now
    if not _has_pending_dir(root):
        return []
    from local_operator.server.utils import desktop_mesh

    try:
        live = desktop_mesh._call(  # noqa: SLF001 — the plane's one relay client
            root, "net_pair_pending", timeout=5.0, include_stages=True
        )
    except MeshRefusal:
        live = None
    rows = live.get("value") if isinstance(live, dict) else None
    if not isinstance(rows, list):
        return _inviter_from_files(root, moment)
    out: list[PendingPairing] = []
    for data in rows:
        if not isinstance(data, dict):
            continue
        try:
            row = PendingPairing.from_json(data)
        except (TypeError, ValueError):
            continue
        if row.is_open(moment):
            out.append(row)
    return out


def joiner_records(root: Path | None) -> list[PendingJoin]:
    """Every parked join that could still take an answer (open window, live parker)."""
    if not _has_pending_dir(root):
        return []
    return pairing_answers.live_parked_joins(_bound_store(root))


def first_open_pairing_id(root: Path | None) -> str | None:
    """The oldest open card's id, files only — the feed's 1 s probe (no relay call)."""
    if not _has_pending_dir(root):
        return None
    now = time.time()
    candidates: list[tuple[float, str]] = [
        (row.issued_at, f"inviter:{row.invite_id}") for row in _inviter_from_files(root, now)
    ]
    candidates += [(row.issued_at, f"joiner:{row.invite_id}") for row in joiner_records(root)]
    return min(candidates)[1] if candidates else None


def _offers(state: str, items: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "state": state,
        "items": [{key: item.get(key) for key in _OFFER_KEYS} for item in items or []],
    }


def _inviter_attempts_left(root: Path | None, row: PendingPairing) -> int:
    from local_operator.network import store
    from local_operator.network.invite import PAIRING_MAX_FORGIVEN_FAILURES

    try:
        record = store.load(row.network_id, root)
    except (FileNotFoundError, ValueError, TypeError):
        return 0
    invite = record.invite(row.invite_id)
    spent = invite.attempts if invite is not None else PAIRING_MAX_FORGIVEN_FAILURES
    return max(0, PAIRING_MAX_FORGIVEN_FAILURES - spent)


def _answerable(unavailable: str, *, nonce: str, stage: str) -> bool:
    return not unavailable and bool(nonce) and stage == PAIR_STAGE_AWAITING_CONFIRMATION


def pairing_rows(root: Path | None, service: Any) -> dict[str, Any]:
    """``GET /v1/desktop/mesh/pairings``: ``{pairings: [row, ...]}``, oldest first.

    ``can_answer_here`` is true only on a host with an app-attestation listener AND
    for a card in a stage that takes an answer; ``answer_unavailable`` names why
    not at the host level (``unsupported_host`` | ``no_app_attestation``; ``""``).
    """
    now = time.time()
    unavailable = pairing_attest.answer_unavailable(service)
    rows: list[dict[str, Any]] = []
    for inv in inviter_records(root, now=now):
        rows.append(
            {
                "id": f"inviter:{inv.invite_id}",
                "side": "inviter",
                "stage": inv.stage,
                "invite_id": inv.invite_id,
                "network_id": inv.network_id,
                "network_name": inv.network_name,
                "peer_device_id": inv.joiner_device_id,
                "peer_name": inv.joiner_name,
                "role": inv.role,
                # THIS device's own code: the card shows it for the OTHER device's
                # person to type there. Not a secret (design F4).
                "code": inv.sas,
                "fingerprint": inv.fingerprint,
                "offers": _offers(inv.offer_state, list(inv.offer)),
                "expires_at": inv.expires_at,
                "seconds_left": int(inv.seconds_left(now)),
                "attempts_left": _inviter_attempts_left(root, inv),
                "nonce": inv.nonce,
                "can_answer_here": _answerable(unavailable, nonce=inv.nonce, stage=inv.stage),
                "answer_unavailable": unavailable,
                "issued_at": inv.issued_at,
            }
        )
    store = _bound_store(root)
    for join in joiner_records(root):
        waiting = store.join_answer(join.invite_id) is not None
        stage = JOINER_WAITING if waiting else PAIR_STAGE_AWAITING_CONFIRMATION
        rows.append(
            {
                "id": f"joiner:{join.invite_id}",
                "side": "joiner",
                "stage": stage,
                "invite_id": join.invite_id,
                "network_id": join.network_id,
                "network_name": join.network_name,
                "peer_device_id": join.inviter_device_id,
                "peer_name": join.inviter_name,
                "role": "",
                "code": join.sas,
                "fingerprint": join.fingerprint,
                "offers": _offers(join.offer_state, list(join.offers)),
                "expires_at": join.expires_at,
                "seconds_left": int(join.seconds_left(now)),
                "attempts_left": pairing_answers.strikes_left(join.invite_id, join.nonce),
                "nonce": join.nonce,
                "can_answer_here": _answerable(
                    unavailable,
                    nonce=join.nonce,
                    # ``--verify`` parks ask for the fingerprint, which the card's
                    # six-digit input cannot carry: those stay terminal-only.
                    stage="" if join.verify else stage,
                ),
                "answer_unavailable": unavailable,
                "issued_at": join.issued_at,
            }
        )
    rows.sort(key=lambda row: row["issued_at"])
    for row in rows:
        row.pop("issued_at")
    return {"pairings": rows}


def _find(root: Path | None, side: str, invite_id: str) -> dict[str, Any] | None:
    # The host-level availability does not change which row is found.
    for row in pairing_rows(root, None)["pairings"]:
        if row["side"] == side and row["invite_id"] == invite_id:
            return row
    return None


def answer(
    root: Path | None,
    service: Any,
    *,
    side: str,
    invite_id: str,
    decision: str,
    code: str,
    shares: list[str] | None,
    nonce: str,
    deadline: int,
) -> dict[str, Any]:
    """``POST .../pairings/{side}/{invite_id}/answer`` — attestation first, then act.

    ORDER IS THE SECURITY PROPERTY: find the live card (404/409 consume nothing) →
    require a deadline inside the ceremony → CONSUME a matching app attestation (none:
    403 ``pairing_needs_app``; nothing compared, no attempt spent, no decision) →
    only then hand the answer to its decider. Refusal codes: ``no_pending_pairing``
    (404), ``pairing_expired`` / ``pairing_not_ready`` (409), ``pairing_needs_app``
    (403). A wrong code is a 409 ``sas_mismatch`` carrying ``attempts_left`` and the
    card stays open; see the route for the mapping.
    """
    if side not in pairing_attest.SIDES:
        raise MeshRefusal("no_pending_pairing", "there is no pairing of that kind")
    if pairing_attest.answer_unavailable(service):
        # NEVER A DOWNGRADE: a host that cannot attest refuses outright.
        raise MeshRefusal("pairing_needs_app", "answering a pairing here needs the signed app")
    row = _find(root, side, invite_id)
    if row is None:
        # The ceremony ended (expiry, answer, abandonment): drop any attestation
        # that was made for it so nothing about it outlives the card.
        pairing_attest.discard(service, side=side, invite_id=invite_id)
        raise MeshRefusal("no_pending_pairing", "that pairing is no longer waiting here")
    if not nonce or nonce != row["nonce"]:
        raise MeshRefusal("pairing_expired", "that pairing card is out of date")
    if side == "inviter" and row["stage"] != PAIR_STAGE_AWAITING_CONFIRMATION:
        raise MeshRefusal("pairing_not_ready", "the other device has not typed this code yet")
    now = time.time()
    if not (now < deadline <= math.ceil(float(row["expires_at"]))):
        raise MeshRefusal("pairing_needs_app", "the app's approval does not cover this pairing")
    if not pairing_attest.consume(
        service,
        side=side,
        invite_id=invite_id,
        nonce=nonce,
        decision=decision,
        code=code,
        shares=shares,
        deadline=deadline,
    ):
        raise MeshRefusal("pairing_needs_app", "answering a pairing needs the signed app")
    if side == "inviter":
        from local_operator.server.utils import desktop_mesh

        fields: dict[str, Any] = {
            "invite_id": invite_id,
            "decision": "admit" if decision == "admit" else "decline",
            "code": code if decision == "admit" else "",
            "nonce": nonce,
            "answered_by": "desktop",
        }
        if shares is not None:
            fields["shares"] = list(shares)
        reply = desktop_mesh._call(root, "net_pair_confirm", timeout=10.0, **fields)  # noqa: SLF001
        status = str(reply.get("status") or "sent")
        left = int(reply.get("attempts_left") or 0)
        if decision != "admit" and status == "sent":
            status = "refused"
    else:
        outcome = pairing_answers.answer_card(
            _bound_store(root),
            invite_id,
            decision=decision,
            code=code,
            nonce=nonce,
            answered_by="desktop",
        )
        status, left = outcome.status, outcome.attempts_left
    if status == "sas_mismatch":
        raise PairingMismatch(left)
    return {"status": status, "attempts_left": left}


class PairingMismatch(MeshRefusal):
    """A wrong code on an open card: 409 with ``attempts_left``; the card stays."""

    def __init__(self, attempts_left: int) -> None:
        super().__init__("sas_mismatch", "that code does not match the other device's")
        self.attempts_left = attempts_left
