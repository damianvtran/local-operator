"""The DURABLE half of a park that happens on ANOTHER device.

WHY THIS MODULE EXISTS. A session this device started can be offloaded to a
peer, and when a tool call there parks for approval, nothing on this machine
was holding that fact: the TUI's park notice (``OperatorApp._note_remote_parks``)
is a toast plus one OS banner, and a banner dies with its dismissal. An operator
who was not looking at the screen — the 2026-09-28 offload test on
``cloud-node-1`` is the measured case — learned nothing at all, and the park sat
for hours until someone attached a viewer. The operator's inbox
(``docs/design/notifications-inbox.md`` §1) is the surface that exists for
exactly that: a durable, unread-until-read row every surface can show.

ONE STORE, ONE ROW SHAPE, NO SECOND KILL SWITCH. The write goes through
:meth:`local_operator.notifications.store.NotificationStore.notify` — the same
door ``Session._notify_inbox`` uses for a LOCAL park (``Session.raise_gate_item``)
and for the agent's ``notify`` tool — so the read/dismiss marks, the same-topic
fold, the priority ladder and the store's own retention are the ones already
shipped. The kill switch is the same one: ``notify_enabled()``.

THE ROW IS KEYED BY THE PEER SESSION AND IS STABLE PER EPISODE. The key carries
the DEVICE as well as the kind because a session id is meant to be unique across
the mesh but the park detector already refuses to assume it
(``session/peer_rows.park_edges``): two devices minting one id are two
conversations, and a key on the id alone would let one device's park overwrite
the other's row. A re-park after the row was dismissed is the SAME key and
therefore an update — one operator action clears one episode, never a queue of
identical rows.

WHAT THIS DOES NOT DO, stated here so no reader assumes it. The item is a
notification, not a control: it does not allow the call. An allow has to be
signed by a key the parked runtime can verify (``harness/approval.py``,
``operator/verify.py``), so the remedy an unconfigured peer needs is the anchor
install the item's body names — and a peer with no authority stays unable to
allow, which is the negative control beside this module's tests.

BEST-EFFORT BY CONTRACT. A row is chrome beside a parked tool call; nothing
here may raise. Every failure is logged and returns ``None``/``False``, because
the alternative — an attention write that can break the poll it rides — would
cost the park notice as well as the row.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: The parked-gate priority, the local gate item's (``Session.raise_gate_item``):
#: the agent is blocked on a person, which outranks any informational update.
PRIORITY = "high"

#: The key's namespace. ``gate:`` is the local park's (``Session.raise_gate_item``
#: writes ``gate:<kind>:<wait id>``); a remote park is a different row in the
#: same namespace rather than a second vocabulary, and the prefix is what tells
#: a reader (and the tests) which device the countdown belongs to.
KEY_PREFIX = "remote-gate"


def item_key(device_id: str, session_id: str, kind: str) -> str:
    """The stable key of one conversation's current park of ``kind`` on ``device_id``.

    THREE TERMS, each answering a different collision. The DEVICE because a
    session id is meant to be unique across the mesh but ``park_edges`` refuses
    to assume it (two devices minting one id are two conversations); the
    SESSION because two conversations can be parked on one device at once and a
    row is about one of them (``session_id``/``session_title`` are the store's
    own columns); the KIND because an ask and an approval are two remedies.
    """
    return f"{KEY_PREFIX}:{device_id}:{session_id}:{kind}"


def _inbox_reads_here() -> bool:
    """Whether this machine's inbox should hold a park row at all.

    The ``notify`` tool's own switch (``builtin.notify_enabled``), asked the way
    ``Session._notify_inbox_callable`` asks it: ``LOP_NOTIFY=0`` means the
    operator has said this machine raises no items, and a park is not exempt.
    Imported lazily because ``tools.builtin`` loads the whole tool surface and
    this module is reached from a five-second poll.
    """
    from local_operator.tools import builtin

    return builtin.notify_enabled()


def raise_park_item(
    root: Path,
    *,
    session_id: str,
    device_id: str,
    kind: str,
    device_name: str = "",
    name: str = "",
) -> str | None:
    """Create or update the row for one park on ``device_id``.

    Returns the store's action (``created``/``updated``/``absorbed``/
    ``unchanged``/``revived``) for the caller's log, or ``None`` when the row
    could not be written — the two answers a caller needs, and no exception.
    """
    if not session_id or not device_id or not kind:
        return None
    try:
        if not _inbox_reads_here():
            logger.debug("remote park item: an inbox that raises no items was asked to")
            return None
        from local_operator.notifications.store import NotificationStore
        from local_operator.tui.notify import (
            remote_park_item_body,
            remote_park_item_title,
        )

        label = device_name or device_id
        store = NotificationStore(root)
        outcome = store.notify(
            session_id=session_id,
            title=remote_park_item_title(label, kind),
            body_md=remote_park_item_body(label, kind),
            priority=PRIORITY,
            key=item_key(device_id, session_id, kind),
            # The conversation's name where the peer chose one, so two parks on
            # one device are tellable apart in a list of rows.
            session_title=name,
        )
        _reopen(store, session_id, device_id, kind)
        logger.info(
            "remote park item %s for %s on %s (%s)",
            outcome.action,
            session_id,
            label,
            kind,
        )
        return outcome.action
    except Exception:  # noqa: BLE001 — a lost row must not cost the park notice
        logger.warning(
            "remote park item: could not raise the %s row for %s on %s",
            kind,
            session_id,
            device_id,
            exc_info=True,
        )
        return None


def _reopen(store: Any, session_id: str, device_id: str, kind: str) -> None:
    """Make the key's row UNREAD and UNDISMISSED again, if it had been settled.

    THE COLLAPSE RULE WOULD OTHERWISE SWALLOW A SECOND PARK (``store.notify``'s
    own contract): a keyed write whose text is unchanged is ``unchanged`` when
    the row was read and ``absorbed`` when it was dismissed, and the store only
    revives a dismissed row on a PRIORITY escalation — a park is always
    ``high``, so it never escalates. Left alone, a conversation that parks,
    settles, then parks again the same way would raise exactly one row and the
    second wait would be invisible: the defect this module exists to fix,
    reproduced by its own collapse.

    THE CALLER RAISES ON AN EDGE, never on a poll that merely found the park
    still live (``OperatorApp._note_remote_parks`` fires this once per episode),
    which is what keeps an operator who read the row from being pushed back to
    unread while the call they read about is still waiting.

    Never raises: the caller's try owns the failure, and a row that could not be
    reopened is still a row.
    """
    try:
        row = store.get_by_key(session_id, item_key(device_id, session_id, kind))
        if row is None:
            return
        if row["read"]:
            store.apply("unread", ids=[(str(row["id"]), int(row["revision"]))], via="remote-park")
            row = store.get_by_key(session_id, item_key(device_id, session_id, kind))
            if row is None:
                return
        if row["dismissed"]:
            store.apply(
                "undismiss", ids=[(str(row["id"]), int(row["revision"]))], via="remote-park"
            )
    except Exception:  # noqa: BLE001 — a row that could not be reopened is still a row
        logger.warning(
            "remote park item: could not reopen the %s row for %s on %s",
            kind,
            session_id,
            device_id,
            exc_info=True,
        )


def clear_park_item(
    root: Path,
    *,
    session_id: str,
    device_id: str,
    kind: str,
    via: str = "remote-park",
) -> bool:
    """Dismiss the row for a park that is no longer live.

    THE CARD IS WITHDRAWN FOR THE SAME REASON THE TOAST IS
    (``OperatorApp._note_remote_parks``): a row that still says "waiting" after
    the answer landed is a lie the operator cannot correct from the row itself.
    Dismissed rather than deleted so the record and its read mark stay in the
    store's history (``store.apply``'s ``dismiss`` is exactly that).

    ``False`` covers every non-write: no row for this key, a row whose revision
    moved between the read and the write (the token-bound rule, R10 — someone
    else's newer revision wins), and any store failure. Never raises.
    """
    try:
        from local_operator.notifications.store import NotificationStore

        store = NotificationStore(root)
        row = store.get_by_key(session_id, item_key(device_id, session_id, kind))
        if row is None:
            return False
        result: Any = store.apply(
            "dismiss",
            ids=[(str(row["id"]), int(row["revision"]))],
            via=via,
        )
        return bool(result.changed)
    except Exception:  # noqa: BLE001 — the row is stale chrome, never a gate
        logger.warning(
            "remote park item: could not clear the %s row for %s on %s",
            kind,
            session_id,
            device_id,
            exc_info=True,
        )
        return False


def write_transitions(
    root: Path,
    ended: Sequence[Mapping[str, str]],
    fresh: Sequence[Mapping[str, str]],
) -> None:
    """Apply one read's worth of park transitions: clears first, then raises.

    ONE function for the pair, because the order is a rule rather than an
    accident: a park that ends and a park that starts for the SAME key in one
    read (a re-park inside one poll) must end up unread and present, and
    clearing after the raise would dismiss the row that describes the live park.

    Both arguments are plain dicts rather than ``RemotePark``/key objects so the
    caller assembles them from whatever its detector produced (the TUI hands
    ``(device_id, session_id, kind, device_name, name)`` per edge) and this
    module stays free of the sidebar's types.
    """
    for one in ended:
        clear_park_item(
            root,
            session_id=one.get("session_id", ""),
            device_id=one.get("device_id", ""),
            kind=one.get("kind", ""),
        )
    for one in fresh:
        raise_park_item(
            root,
            session_id=one.get("session_id", ""),
            device_id=one.get("device_id", ""),
            kind=one.get("kind", ""),
            device_name=one.get("device_name", ""),
            name=one.get("name", ""),
        )
