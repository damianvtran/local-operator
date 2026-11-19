"""The JOINER's answer to a parked pairing — one checker for the CLI and the app.

WHY A SHARED MODULE. A parked join (``lop network join --park``) waits for its
person's answer in a file (``store.save_join_answer``); two surfaces now write that
answer: ``lop network join --confirm <code>`` (the same-user terminal fallback) and the
desktop's Mesh approvals card (``POST /v1/desktop/mesh/pairings/joiner/<invite>/answer``,
after an app attestation). Both must apply the SAME check — the typed value against
this device's own derivation, constant time — and write the SAME record, or the two
doors would come to admit different things. The CLI keeps its own sentences and its
"the code chooses the ceremony" selection; what it shares is below.

THE CHECK HERE IS A TYPO GUARD, NOT THE BOUNDARY. Both ends derive the same six digits
and this device already holds them, so a local match proves only that the person read
the other screen correctly. The comparison that decides is the inviter's, on the wire
(``net_pair_ready``) and then its own person's. A local mismatch therefore sends
nothing to the peer and leaves the ceremony open.

CARD STRIKES. The card counts its own wrong codes per ceremony (``PAIR_CARD_STRIKES``)
and declines at the bound, so the joiner card behaves like the inviter's. They are the
CARD's strikes only: ``lop network join --confirm`` keeps its own retry rule, and the
count is forgotten when the ceremony ends (answered, declined, or no longer parked). The count is
in THIS process's memory, deliberately: it has no security value (this device already
knows the code), and writing it into the ``PendingJoin`` file would race the parker,
which rewrites that same file with its outcome. A ``lop serve`` restart resets it.
"""

from __future__ import annotations

import hmac
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from local_operator.network.types import JoinAnswer, MeshRefusal

#: Wrong codes the joiner card tolerates per ceremony before it declines — the same
#: number the inviter's invite budget forgives (``invite.PAIRING_MAX_FORGIVEN_FAILURES``)
#: so both cards read alike.
PAIR_CARD_STRIKES = 3

_strikes: dict[tuple[str, str], int] = {}
_strikes_lock = threading.Lock()


def parked_code_matches(pending: Any, typed: str) -> bool:
    """Whether the human's value is this ceremony's own derivation.

    ``--verify`` compares the 160-bit fingerprint as the prompt path does, and strips
    the same separators; otherwise it is the six digits through the shared
    ``sas_matches`` (constant time, and a value that is not six digits is a mismatch
    rather than a near miss).
    """
    from local_operator.network.handshake import sas_matches

    if pending.verify:
        typed_fp = typed.replace("-", "").replace(" ", "").upper()
        derived_fp = str(pending.fingerprint).replace("-", "").upper()
        return bool(typed_fp) and hmac.compare_digest(derived_fp, typed_fp)
    return sas_matches(str(pending.sas), typed)


def process_alive(pid: int) -> bool:
    """Whether a parker is still there (``procstate.pid_alive``, fail-closed)."""
    from local_operator import procstate

    if pid <= 0:
        return False
    return procstate.pid_alive(pid)


def live_parked_joins(store: Any, *, alive: Callable[[int], bool] = process_alive) -> list[Any]:
    """The parked ceremonies that could still take an answer, newest last.

    Finished records are dropped here rather than by the store: this is the caller
    that knows the difference between "that ceremony has an outcome, report it" and
    "that ceremony is gone", and it is the only place both are asked at once.
    ``alive`` is a parameter so the CLI keeps its own patchable probe.
    """
    return [
        row
        for row in store.pending_joins()
        if row.status == "awaiting_confirmation" and row.is_open() and alive(row.pid)
    ]


def submit(store: Any, row: Any, *, decision: str, code: str, answered_by: str) -> JoinAnswer:
    """Write the answer the parked ceremony reads. The ONE writer for both surfaces."""
    answer = JoinAnswer(
        invite_id=row.invite_id,
        decision="admit" if decision == "admit" else "decline",
        code=code if decision == "admit" else "",
        answered_by=answered_by,
    )
    store.save_join_answer(answer)
    return answer


@dataclass(frozen=True)
class CardOutcome:
    """What a card answer did: ``status`` plus the strikes still allowed."""

    status: str
    attempts_left: int


def _prune_strikes(
    live: set[tuple[str, str]] | None = None, *, ended: tuple[str, str] | None = None
) -> None:
    """Forget the strikes of ceremonies that are over (review round 1, N4).

    ``live`` is every ``(invite_id, nonce)`` still parked and answerable; ``ended`` is
    the ceremony this call just answered for good. Without this the map grew by one
    entry per wrongly-answered joiner ceremony for the life of ``lop serve``.
    """
    with _strikes_lock:
        for key in [
            key for key in _strikes if (live is not None and key not in live) or key == ended
        ]:
            del _strikes[key]


def strikes_left(invite_id: str, nonce: str) -> int:
    with _strikes_lock:
        return max(0, PAIR_CARD_STRIKES - _strikes.get((invite_id, nonce), 0))


def answer_card(
    store: Any,
    invite_id: str,
    *,
    decision: str,
    code: str,
    nonce: str,
    answered_by: str,
    alive: Callable[[int], bool] = process_alive,
) -> CardOutcome:
    """The Mesh approvals card's joiner answer. Never blocks on the other device.

    Statuses: ``waiting_for_other_device`` (a local match was handed to the parker,
    which now sends ``net_pair_ready`` and waits for the inviter's person — the card
    reads the outcome from ``PendingJoin.status``), ``sas_mismatch`` (nothing sent,
    the ceremony stays open), ``refused`` (a decline, or the strike bound reached:
    the parker sends ``declined_local``, which consumes the invite on the inviter).
    Refusals: ``no_pending_pairing`` (nothing parked under that invite here),
    ``pairing_expired`` (window closed, parker gone, or a stale nonce).
    """
    rows = {row.invite_id: row for row in store.pending_joins()}
    row = rows.get(invite_id)
    if row is None:
        raise MeshRefusal(
            "no_pending_pairing", "no pairing is parked on this device for that invite"
        )
    parked = live_parked_joins(store, alive=alive)
    live = {r.invite_id for r in parked}
    _prune_strikes({(r.invite_id, r.nonce) for r in parked})
    if invite_id not in live or not row.nonce or nonce != row.nonce:
        raise MeshRefusal(
            "pairing_expired",
            "that pairing is no longer open on this device, so nothing was answered",
        )
    if store.join_answer(invite_id) is not None:
        # Already answered (a double click, or the CLI got there first): report the
        # wait rather than writing a second answer the parker would never read.
        return CardOutcome("waiting_for_other_device", strikes_left(invite_id, nonce))
    key = (invite_id, nonce)
    if decision != "admit":
        left = strikes_left(invite_id, nonce)
        submit(store, row, decision="decline", code="", answered_by=answered_by)
        _prune_strikes(ended=key)
        return CardOutcome("refused", left)
    if not parked_code_matches(row, code):
        with _strikes_lock:
            spent = _strikes.get(key, 0) + 1
            _strikes[key] = spent
        left = max(0, PAIR_CARD_STRIKES - spent)
        if left == 0:
            submit(store, row, decision="decline", code="", answered_by=answered_by)
            _prune_strikes(ended=key)
            return CardOutcome("refused", 0)
        return CardOutcome("sas_mismatch", left)
    left = strikes_left(invite_id, nonce)
    submit(store, row, decision="admit", code=code, answered_by=answered_by)
    _prune_strikes(ended=key)
    return CardOutcome("waiting_for_other_device", left)
