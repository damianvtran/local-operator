"""Wire a TUI Team Bridge card to the shared card-body resolver.

WHY a separate module: ``TeamBridgeBlock`` must stay a pure widget (it is built
in tests and replay with no session at all), and the replay presenter and the
live receipt path both need the SAME rule for "can this card fetch its body".
The rule is: only a local conversation with an on-disk journal can, because the
resolver is keyed by a transcript row and reads that conversation's own
directory. A remote/peer view gets ``None`` and keeps today's card copy.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

#: The loader a card calls for its resolver answer: ``local_only=True`` on
#: mount (local files, no network — design round 1 D1), ``False`` on open.
CardBodyLoader = Callable[..., Awaitable[dict[str, Any]]]

# -- re-read policy -----------------------------------------------------------
#
# WHY A CARD KEEPS ASKING WHILE IT IS OPEN. The resolver answers ``pending``
# from the moment the card is posted until the receiving session takes the
# body at its next turn, and that hand-off is often long. Measured on the
# operator's machine (n=4005 receipts, post -> hand-off): p50 45 s, p90 21 min,
# p99 100 min, max 8 h. A card that read once and kept the first answer said
# "Waiting to be delivered" forever (the reported bug: a received message whose
# body never appeared, even hours after delivery, until closed and reopened).
#
# So an OPEN card whose answer is not final re-reads on a backoff — 5, 10, 20,
# 30 s, then every 60 s — with NO overall give-up: any cap below the observed
# 8 h maximum would reintroduce the bug for exactly the slow tail. The cost is
# bounded by what is on screen (only expanded cards poll) and a ``pending`` or
# ``held`` re-read is a LOCAL read in core (resolved from the receiver store
# before any hub backfill); only misses/offline reach the hub, at most once a
# minute per open card. Closing the card, unmounting it, or a final answer
# stops the loop. Nothing is cached past an open, so revocation stays live.
#
# The desktop (local-operator-ui) and the phone web client
# (``mobile/web/src/lib/bridge-card-reread.ts``) apply the same table.

#: Delays before each successive re-read of a non-final answer, in seconds;
#: past the end of the table every re-read waits :data:`REREAD_STEADY_S`.
REREAD_BACKOFF_S: tuple[float, ...] = (5.0, 10.0, 20.0, 30.0)
REREAD_STEADY_S = 60.0

#: ``unavailable`` reasons that can still change by themselves: Radient was
#: unreachable, or only the local-only mount pass has answered so far.
_TRANSIENT_UNAVAILABLE = frozenset({"offline", "not_loaded"})


def reread_delay(attempt: int) -> float:
    """Seconds to wait before re-read number ``attempt`` (0-based)."""

    if 0 <= attempt < len(REREAD_BACKOFF_S):
        return REREAD_BACKOFF_S[attempt]
    return REREAD_STEADY_S


def answer_is_final(answer: Any) -> bool:
    """Whether a resolver answer can no longer change by waiting.

    Non-final: no answer / a malformed one (a failed read), any item
    ``pending``, any item ``held`` (a transient hold clears when screening
    runs; a review hold can still be released or refused), and ``unavailable``
    for ``offline``/``not_loaded``. Everything else — admitted, refused,
    not_admitted, the other unavailable reasons, and any state a newer core
    invents — is final, so an unknown state fails quiet instead of polling.
    """

    if not isinstance(answer, dict) or not isinstance(answer.get("items"), list):
        return False
    for item in answer["items"]:
        if not isinstance(item, dict):
            continue
        state = item.get("state")
        if state in ("pending", "held"):
            return False
        if state == "unavailable" and item.get("reason") in _TRANSIENT_UNAVAILABLE:
            return False
    return True


# -- context shown with the body ----------------------------------------------


def card_conversation(details: Any) -> dict[str, Any]:
    """Where a card's message was sent, for the open card's context line.

    From the row's VALIDATED display receipt only (``kind`` in dm/group/channel
    and an optional ``label``), never the carriage's model-facing text. The
    time and the thread-reply marker come from the resolver answer's ``card``
    object instead (``team_bridge.card_body._card_facts``), which every
    client shares.
    """

    from local_operator.team_bridge.display_receipt import projected_display_receipt

    details = details if isinstance(details, dict) else {}
    receipt = projected_display_receipt(details).get("bridge_receipt")
    conversation = receipt.get("conversation") if isinstance(receipt, dict) else None
    conversation = conversation if isinstance(conversation, dict) else {}
    return {"kind": conversation.get("kind"), "label": conversation.get("label")}


def card_body_loader(session: Any, entry_id: str) -> CardBodyLoader | None:
    """A coroutine factory for ``entry_id`` in ``session``, or None."""

    if not entry_id:
        return None
    transcript = getattr(session, "_transcript", None)
    directory = getattr(transcript, "directory", None)
    if directory is None:
        return None
    try:
        directory = Path(directory)
    except TypeError:
        return None
    sessions_root = directory.parent
    # ``<config>/sessions/<id>``: the config dir is the sessions root's parent,
    # the same derivation the desktop host uses (``root / "sessions"``).
    config_dir = sessions_root.parent
    session_id = directory.name

    async def load(*, local_only: bool = False) -> dict[str, Any]:
        from local_operator.team_bridge.card_body import resolve_card_body

        return await resolve_card_body(
            config_dir,
            session_id,
            entry_id,
            sessions_root=sessions_root,
            local_only=local_only,
        )

    return load
