"""A finished turn or a ``notify`` on ANOTHER device, raised on the session's origin.

WHY THIS MODULE EXISTS (remote-session-parity, 2026-11-11). A session the
operator starts on their head device and runs on a mesh node (``/new remote``,
an offload, a move) finishes its turns and calls ``notify`` ON THE NODE: the
completion lands in the node's ``attention.db`` and the item in the node's
``notifications.db``, where nobody is looking. The measured case: the head's
``notifications.db`` held 0 rows for the node's ``notify`` item
``n_xr44lx4zb5gi3mfn``, and its ``attention.db`` 0 completions for any node
session. A local session raises both on the operator's own machine; this is the
hop that makes a remote one do the same.

IT EXTENDS THE PARK PATH, never a second one. A remote approval park already
reaches the origin (``notifications/remote_park.py`` plus
``session/peer_rows.park_edges``) by reading the SAME peer listing the sidebar
polls; this adds two more facts to those rows and two more edge detectors
beside ``park_edges``:

* OWNER SIDE (:func:`stamp_rows_for_viewer`, called from the relay's catalogue
  op): a row gains ``attention: {token, outcome}`` for the owner's newest
  UNACKNOWLEDGED completion and ``notice: {key, revision, priority, title}`` for
  the newest unread ``notify`` item — ONLY when the asking link's device is the
  session's origin (``stamp.origin.source_device``), and only on a link whose
  viewer advertised ``wire.REMOTE_ATTENTION_V1``. Another member of the mesh
  sees neither: a completion is the operator's fact about THEIR session, and a
  co-member listing the node's sessions is not that operator.
* ORIGIN SIDE (:class:`RemoteAttentionWatcher`): one edge per completion token
  and per notice revision, written through the SAME
  ``NotificationStore.notify`` door every other item uses — the kill switch
  (``notify_enabled``), the read/dismiss marks, the priority ladder and the
  store's retention are the shipped ones.

THE CLEAR IS THE EXISTING RECEIPT. Opening the session from the origin already
routes its read receipt to the owner (``net_session_receipt`` →
``attention.acknowledge_session_receipt``); an acknowledged completion is no
longer "unacknowledged", so the owner stops stamping it and the next read sees
the key leave — which marks the origin's done row READ (the operator saw it
elsewhere), never deletes it.

IDEMPOTENT BY CONSTRUCTION, which is what lets two hosts run it. The TUI and the
always-on ``lop serve`` process (and the desktop's own backend) may all watch at
once, and every one of them starts with an empty episode map after a restart.
Each write carries the completion token (or notice revision) as the store's
``episode``: a second writer of the same token is ``unchanged`` — no new row, no
unread flip, no banner — and only a NEW token or revision moves the row. The
idempotence lives in the store's digest, so no cross-process lock or ledger is
needed.

PRIORITY. A completion is ``normal`` (the brief's rule, and the same rung a
local completion's inbox row would sit on): the reel and badge carry it, and the
item ladder never banners ``normal``. A notice passes its own priority through,
so a ``high`` notice on the node is a banner-eligible ``high`` row here, raised
through the item ladder (``notifications.delivery.deliver_item``) whose
per-revision claim keeps the TUI, the desktop feed and this writer to one
banner between them.

WHAT THIS DOES NOT DO. Phone push reads ``attention.db``, not this inbox, and
nothing here writes the origin's ``attention.db`` — a remote completion does not
reach the phone yet (the next slice; stated in the PR, not claimed). And the
completion carries no end time: the owner's completion store records none, so
no ``ended_at`` is invented here.

BEST-EFFORT BY CONTRACT, like the park module: nothing here raises.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: Key namespaces of the two origin-side rows. ``remote-gate`` is the park's
#: (``remote_park.KEY_PREFIX``); these sit beside it so a reader can tell a
#: forwarded row from a local one by its key alone.
DONE_PREFIX = "remote-done"
NOTICE_PREFIX = "remote-notice"

#: A completion's row priority: informational, the turn is not blocked on anyone.
DONE_PRIORITY = "normal"

#: Owner-side items NEVER forwarded as a notice. ``gate:`` rows are parks, which
#: reach the origin through ``pending`` (``remote_park``) — forwarding them here
#: too would put two rows on one park. ``remote-*`` rows are THIS module's and
#: the park module's own writes on a device that is itself an origin: echoing
#: them back would loop between two devices that each started a session on the
#: other. ``team-bridge-`` rows are the node's agent-message notices
#: (``builtin._RESERVED_NOTIFY_KEY_PREFIX``), which belong to that device's
#: connector, not to the session's operator.
NOT_FORWARDED_PREFIXES: tuple[str, ...] = (
    "gate:",
    "remote-gate:",
    f"{DONE_PREFIX}:",
    f"{NOTICE_PREFIX}:",
    "team-bridge-",
)

#: How many unread items per session the owner scans for the newest forwardable
#: one. Bounded so a session that raised a burst of excluded rows cannot make the
#: catalogue op read its whole inbox; the store's own per-hour new-item cap is
#: the population this bounds.
_NOTICE_SCAN = 10


def done_key(device_id: str, session_id: str) -> str:
    """One row per (device, session): a new completion UPDATES it, never stacks.

    The device is in the key for the reason ``remote_park.item_key`` gives: a
    session id meant to be mesh-unique is still not assumed to be.
    """
    return f"{DONE_PREFIX}:{device_id}:{session_id}"


def notice_key(device_id: str, session_id: str, key: str) -> str:
    """One row per OWNER item: a keyed ``notify`` that updates there updates here."""
    return f"{NOTICE_PREFIX}:{device_id}:{session_id}:{key}"


# ---------------------------------------------------------------------------
# Owner side: what a catalogue row carries for its origin
# ---------------------------------------------------------------------------


def stamp_rows_for_viewer(
    root: Path, rows: Sequence[dict[str, Any]], viewer_device_id: str
) -> None:
    """Add ``attention``/``notice`` to the rows whose ORIGIN is the viewer. In place.

    The caller (``RelayServer._op_catalog``) has already checked the viewer's
    capability; this decides per row. A row qualifies only when its stamp's
    ``origin.source_device`` equals ``viewer_device_id`` — equality on the
    authenticated link's device id, so a member cannot ask for another device's
    completions by naming it. One ``attention.db`` snapshot for every qualifying
    row (``state_many``) and one bounded inbox page per qualifying row; a device
    with no peer-originated sessions pays one dict scan and nothing else.

    Never raises: a listing is not an error path, and a store that cannot be
    read leaves the rows exactly as the build before this one wrote them.
    """
    if not viewer_device_id:
        return
    mine = [
        row
        for row in rows
        if isinstance(row.get("origin"), dict)
        and str(row["origin"].get("source_device") or "") == viewer_device_id
        and str(row.get("session_id") or "")
    ]
    if not mine:
        return
    try:
        from local_operator.session.attention import AttentionStore

        identities = [f"session/{row['session_id']}" for row in mine]
        states = AttentionStore(Path(root) / "attention.db").state_many(identities)
    except Exception:  # noqa: BLE001 — a listing without attention is still a listing
        logger.debug("remote attention: completion read failed", exc_info=True)
        states = {}
    store: Any = None
    try:
        from local_operator.notifications.store import NotificationStore

        store = NotificationStore(root)
    except Exception:  # noqa: BLE001
        logger.debug("remote attention: inbox open failed", exc_info=True)
    for row in mine:
        session_id = str(row["session_id"])
        state = states.get(f"session/{session_id}") or {}
        token = state.get("completion_token")
        # ``notify`` False is the completion's own "do not tell anyone" (§14 of
        # the attention contract): honoured here exactly as the local legs do.
        if token and state.get("unseen") and state.get("notify", True):
            row["attention"] = {"token": str(token), "outcome": str(state.get("kind") or "")}
        notice = _newest_notice(store, session_id) if store is not None else None
        if notice is not None:
            row["notice"] = notice


def _newest_notice(store: Any, session_id: str) -> dict[str, Any] | None:
    """The newest unread, forwardable item of ``session_id``, as its wire block."""
    try:
        items, _cursor = store.list_items(state="unread", session_id=session_id, limit=_NOTICE_SCAN)
    except Exception:  # noqa: BLE001
        logger.debug("remote attention: inbox read failed for %s", session_id, exc_info=True)
        return None
    for item in items:
        key = str(item.get("key") or "")
        if key.startswith(NOT_FORWARDED_PREFIXES):
            continue
        return {
            # A keyless item is its own topic on the owner (every write is a new
            # row), so its id stands in for the key — the same "one row per
            # owner item" on the origin.
            "key": key or str(item.get("id") or ""),
            "revision": int(item.get("revision") or 1),
            "priority": str(item.get("priority") or "normal"),
            "title": str(item.get("title") or ""),
        }
    return None


# ---------------------------------------------------------------------------
# Origin side: the rows
# ---------------------------------------------------------------------------


def _inbox_reads_here() -> bool:
    """The ``notify`` kill switch, asked the way ``remote_park`` asks it."""
    from local_operator.tools import builtin

    return builtin.notify_enabled()


def _location(device_label: str) -> str:
    """ "runs on <device>" — the park card's own floor sentence, reused (keyed copy)."""
    from local_operator.tui.notify import remote_session_location

    return remote_session_location(device_label)


def raise_done_item(
    root: Path,
    *,
    session_id: str,
    device_id: str,
    token: str,
    outcome: str = "complete",
    device_name: str = "",
    name: str = "",
) -> str | None:
    """Create or update the origin's row for one finished turn on ``device_id``.

    The TITLE is the conversation's name (the local completion banner's own
    title rule) and the BODY is the same house sentence a local completion of
    that kind carries, then where it ran. ``token`` is the store's ``episode``:
    the same completion re-raised is ``unchanged`` (no unread flip), a new one
    is an update that is unread again. Returns the store's action, or ``None``
    when nothing was written.
    """
    if not session_id or not device_id or not token:
        return None
    try:
        if not _inbox_reads_here():
            return None
        from local_operator.notifications.store import NotificationStore
        from local_operator.tui import notify as notify_copy

        label = device_name or device_id
        body = notify_copy.BODIES.get(outcome or "complete", notify_copy.BODY_COMPLETE)
        store = NotificationStore(root)
        existing = store.get_by_key(session_id, done_key(device_id, session_id))
        if existing is not None and str(existing.get("anchor_id") or "") == token:
            # THIS completion is already the row's (a restarted watcher, or a
            # second host). Answered before the write rather than left to the
            # digest, because the digest also covers the title: a session
            # renamed since would otherwise turn a repeat into an update and
            # flip a row the operator had read back to unread.
            return "unchanged"
        result = store.notify(
            session_id=session_id,
            title=name or notify_copy.BACKGROUND_FALLBACK_TITLE,
            body_md=f"{body}\n\n{_location(label)}",
            priority=DONE_PRIORITY,
            key=done_key(device_id, session_id),
            anchor_id=token,
            session_title=name,
            episode=token,
        )
        if result.action == "absorbed":
            # A NEW completion on a row the operator dismissed: the store keeps
            # an unescalated update dismissed, but a dismissed "it finished"
            # must not swallow the next finish. Same reasoning as
            # ``remote_park._reopen``; the episode makes this reachable only
            # for a new token.
            _undismiss(store, session_id, done_key(device_id, session_id))
        return str(result.action)
    except Exception:  # noqa: BLE001 — chrome; never the failure
        logger.warning(
            "remote attention: could not raise done row for %s", session_id, exc_info=True
        )
        return None


def settle_done_item(root: Path, *, session_id: str, device_id: str, token: str) -> bool:
    """Mark the done row READ when its completion was acknowledged on the owner.

    READ, not dismissed: the operator saw the conversation (that is what an
    acknowledgement is), and a read row stays in the reel's history exactly as a
    locally-read one does. Bound to ``token``: a row that already describes a
    NEWER completion is left alone.
    """
    try:
        from local_operator.notifications.store import NotificationStore

        store = NotificationStore(root)
        row = store.get_by_key(session_id, done_key(device_id, session_id))
        if row is None or row["read"] or str(row.get("anchor_id") or "") != token:
            return False
        result: Any = store.apply(
            "read", ids=[(str(row["id"]), int(row["revision"]))], via="remote-attention"
        )
        return bool(result.changed)
    except Exception:  # noqa: BLE001
        logger.debug(
            "remote attention: could not settle done row for %s", session_id, exc_info=True
        )
        return False


def raise_notice_item(
    root: Path,
    *,
    session_id: str,
    device_id: str,
    key: str,
    revision: int,
    priority: str,
    title: str,
    device_name: str = "",
    name: str = "",
    deliver: bool = True,
) -> str | None:
    """Mirror one owner ``notify`` item onto the origin's inbox.

    The title is the agent's own (already redacted by the owner's ``notify``
    before it was stored there); the body says where it ran. ``priority`` passes
    through — an unknown word degrades to ``normal`` rather than failing the
    write. A deliverable ``high`` write is handed to the item ladder off-thread
    (``deliver``), so it banners exactly once machine-wide.
    """
    if not session_id or not device_id or not key or not title:
        return None
    try:
        if not _inbox_reads_here():
            return None
        from local_operator.notifications.store import PRIORITIES, NotificationStore

        if priority not in PRIORITIES:
            priority = "normal"
        label = device_name or device_id
        store = NotificationStore(root)
        row_key = notice_key(device_id, session_id, key)
        result = store.notify(
            session_id=session_id,
            title=title,
            body_md=_location(label),
            priority=priority,
            key=row_key,
            session_title=name,
            episode=f"rev:{int(revision)}",
        )
        # An ``absorbed`` notice (the operator dismissed this row and the owner's
        # item was updated without escalating) STAYS dismissed: that is the
        # store's rule for a local keyed ``notify`` too, and a forwarded one must
        # not be harder to set aside than a local one. An escalation revives it
        # through the store itself.
        if deliver and priority == "high":
            _deliver_async(root, result, session_id=session_id, title=title, body=_location(label))
        return str(result.action)
    except Exception:  # noqa: BLE001
        logger.warning("remote attention: could not raise notice for %s", session_id, exc_info=True)
        return None


def _undismiss(store: Any, session_id: str, key: str) -> None:
    row = store.get_by_key(session_id, key)
    if row is not None and row["dismissed"]:
        store.apply(
            "undismiss", ids=[(str(row["id"]), int(row["revision"]))], via="remote-attention"
        )
    row = store.get_by_key(session_id, key)
    if row is not None and row["read"]:
        store.apply("unread", ids=[(str(row["id"]), int(row["revision"]))], via="remote-attention")


def _deliver_async(root: Path, outcome: Any, *, session_id: str, title: str, body: str) -> None:
    """Hand a ``high`` revision to the item ladder WITHOUT blocking the watcher.

    ``deliver_item`` sleeps out the banner grace before it claims, so it runs on
    a daemon thread. ``route_item`` refuses a non-deliverable action itself
    (``unchanged`` from a second watcher never banners), and the store's
    per-revision claim makes the TUI's sweep, the desktop feed and this one
    raise at most one banner between them.
    """

    def run() -> None:
        try:
            from local_operator.notifications.delivery import deliver_item

            deliver_item(
                root,
                item_id=str(outcome.item_id),
                revision=int(outcome.revision),
                priority=str(outcome.priority),
                action=str(outcome.action),
                session_id=session_id,
                title=title,
                body=body,
            )
        except Exception:  # noqa: BLE001
            logger.debug("remote attention: banner delivery failed", exc_info=True)

    threading.Thread(target=run, name="remote-notice-banner", daemon=True).start()


# ---------------------------------------------------------------------------
# The watcher both hosts share
# ---------------------------------------------------------------------------


class RemoteAttentionWatcher:
    """Edge state for one host process; :meth:`observe` is the whole surface.

    ONE helper for both hosts (the TUI's park poll and ``lop serve``'s
    :func:`run_forever`), so they cannot drift. The maps are per process by
    design — the store's ``episode`` digest is what makes a fresh map (a restart,
    a second host) harmless.
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self._done: dict[tuple[str, str], str] = {}
        self._notice: dict[tuple[str, str], tuple[str, int]] = {}

    def observe(self, rows: Iterable[Any], unanswered: Iterable[str] = ()) -> None:
        """Apply one listing read's completion and notice edges. Never raises."""
        try:
            from local_operator.session.peer_rows import completion_edges, notice_edges

            rows = tuple(rows)
            silent = tuple(unanswered)
            done, done_state = completion_edges(self._done, rows, unanswered=silent)
            for key, token in self._done.items():
                if key not in done_state:
                    settle_done_item(self.root, session_id=key[1], device_id=key[0], token=token)
            self._done = done_state
            for edge in done:
                raise_done_item(
                    self.root,
                    session_id=edge.session_id,
                    device_id=edge.device_id,
                    token=edge.token,
                    outcome=edge.outcome,
                    device_name=edge.device_name,
                    name=edge.name,
                )
            notices, notice_state = notice_edges(self._notice, rows, unanswered=silent)
            self._notice = notice_state
            for notice in notices:
                raise_notice_item(
                    self.root,
                    session_id=notice.session_id,
                    device_id=notice.device_id,
                    key=notice.key,
                    revision=notice.revision,
                    priority=notice.priority,
                    title=notice.title,
                    device_name=notice.device_name,
                    name=notice.name,
                )
        except Exception:  # noqa: BLE001 — a poll that cannot forward is still a poll
            logger.debug("remote attention watch failed", exc_info=True)


#: ``lop serve``'s poll cadence: the TUI park watch's (``REMOTE_PARK_POLL_S``),
#: over the same 20 s-cached listing, so serve adds no relay load the TUI did
#: not already pay — the two share ``peer_rows``' process-local cache only within
#: one process, so with both up the relay sees at most one listing per TTL each.
SERVE_POLL_S = 5.0


async def run_forever(root: Path, *, interval_s: float = SERVE_POLL_S) -> None:
    """The always-on host: ``lop serve`` runs this so no TUI has to be open.

    A device in no mesh short-circuits in ``read_listing`` (no relay record, no
    call), so every install outside a mesh pays one directory check per poll.
    Cancelled by the server's lifespan teardown.
    """
    from local_operator.session.peer_rows import read_listing

    watcher = RemoteAttentionWatcher(root)
    while True:
        try:
            rows, unanswered = await asyncio.to_thread(read_listing, Path(root))
            await asyncio.to_thread(watcher.observe, rows, [peer.device_id for peer in unanswered])
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            logger.debug("remote attention serve poll failed", exc_info=True)
        await asyncio.sleep(interval_s)


# ---------------------------------------------------------------------------
# The node's notify sentence
# ---------------------------------------------------------------------------


def forwarded_notice_sentence(root: Path, session_id: str) -> str:
    """The clause a peer-placed session's ``notify`` result adds, or ``""``.

    Model-facing English (a tool result, like the sentence it extends), so not
    catalogued. A bare "Notified" on the node read as if the operator had been
    told; the item is in THIS device's inbox and reaches the operator only when
    the origin forwards it, so the model is told exactly that. ``""`` for every
    session not placed here by another device — an ordinary local session's
    result is unchanged byte for byte.
    """
    try:
        from local_operator.session.placement import read_stamp

        stamp = read_stamp(Path(root), session_id)
        if stamp is None or stamp.placement.mode != "peer":
            return ""
        origin = str((stamp.origin or {}).get("source_device") or "")
        if not origin:
            return ""
        here, there = _device_names(Path(root), origin)
        return f" Posted on {here}; it will be forwarded to {there}, where this session started."
    except Exception:  # noqa: BLE001
        return ""


def _device_names(root: Path, origin: str) -> tuple[str, str]:
    """``(this device's name, the origin's name)`` from the membership records."""
    here = "this device"
    there = origin
    try:
        from local_operator.network import store as net_store

        for record in net_store.list_networks(root):
            for member in record.active_members():
                if member.device_id == record.self_device_id and member.name:
                    here = member.name
                if member.device_id == origin and member.name:
                    there = member.name
    except Exception:  # noqa: BLE001
        pass
    return here, there
