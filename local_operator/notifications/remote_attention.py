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
  UNACKNOWLEDGED completion and ``notices: [{key, revision, priority, title},
  ...]`` for the newest :data:`NOTICES_ON_WIRE` unread ``notify`` items — ONLY
  when the asking link's device is the session's origin
  (``stamp.origin.source_device``), and only on a link whose viewer advertised
  ``wire.REMOTE_ATTENTION_V1``. Another member of the mesh
  sees neither: a completion is the operator's fact about THEIR session, and a
  co-member listing the node's sessions is not that operator.
* ORIGIN SIDE (:class:`RemoteAttentionWatcher`): one edge per completion token
  and per (owner item, revision), written through the SAME
  ``NotificationStore.notify`` door every other item uses — the kill switch
  (``notify_enabled``), the read/dismiss marks, the priority ladder and the
  store's retention are the shipped ones.

THE CLEAR IS THE EXISTING RECEIPT. Opening the session from the origin already
routes its read receipt to the owner (``net_session_receipt`` →
``attention.acknowledge_session_receipt``); an acknowledged completion is no
longer "unacknowledged", so the owner stops stamping it, and any listing read
that shows the session without that claim marks the origin's done row READ (the
operator saw it elsewhere), never deletes it. The settle reads the store and
the CURRENT listing, never process memory, so a receipt that landed while no
watcher ran still clears the row on the next read (QA round 1, Q1).

IDEMPOTENT BY CONSTRUCTION, which is what lets two hosts run it. The TUI and the
always-on ``lop serve`` process (and the desktop's own backend) may all watch at
once, and every one of them starts with an empty episode map after a restart.
Each row records its episode identity — the completion token, or ``rev:<N>``
for a notice — as its ``anchor_id`` and the store's ``episode``; a writer whose
episode is already the row's (or, for a notice, older than it) returns
``unchanged`` BEFORE writing — no new row, no unread flip, no banner — and only a
NEW token or a HIGHER revision moves the row. The check reads the row rather
than relying on the content digest alone, because the digest also covers the
title and the "runs on <device>" body, which a renamed session or a host that
read a nameless row would change (review round 1, m1). No cross-process lock or
ledger is needed.

PRIORITY AND BANNERS. A completion's ROW is ``normal`` (the reel and badge carry
it, and the item ladder never banners ``normal``), but a NEW completion also
raises the same clickable OS banner a local background completion raises
(review round 1, M1: a remote completion must be as visible as a local one).
That banner is :func:`_deliver_done_banner`, claimed per revision through
``NotificationStore.claim_delivery`` so the TUI's watcher and ``lop serve``'s
raise exactly one between them; its click carries the session id to ``lop
resume-click``, whose ladder (desktop viewer, desktop launch, TUI viewer, then
a terminal running ``lop --resume <id>``) opens a peer session by id — the same
id ``lop --resume`` resolves to a remote viewer. A notice passes its own priority through,
so a ``high`` notice on the node is a banner-eligible ``high`` row here, raised
through the item ladder (``notifications.delivery.deliver_item``) whose
per-revision claim keeps the TUI, the desktop feed and this writer to one
banner between them.

WHAT THIS DOES NOT DO. Phone push reads ``attention.db``, not this inbox
(``mobile/push_worker.py``: its completion scan is built on ``AttentionStore``
records, cursor and deferral), and nothing here writes the origin's
``attention.db`` — a remote completion does not reach the phone yet; the
follow-up is a second push source in that worker reading ``remote-done:`` rows.
The completion carries no end time: the owner's completion store records
none, so no ``ended_at`` is invented here.

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

#: How many notices one row carries for its origin, newest first (review round 1,
#: M2: carrying only the newest lost every earlier unread notice, ``high`` ones
#: included). WHY 8: the owner's inbox is never read on the node, so its unread
#: items persist and EVERY catalogue read for the origin carries the full list
#: for every peer-originated session — at most 8 x (the owner store's 120-char
#: ``KEY_MAX`` key + 120-char ``TITLE_MAX`` title + small fields), measured at
#: ≤2.5 KB of JSON per session (2,453 B worst case), which keeps a listing of tens
#: of such sessions well inside a relay frame. A notice is lost only when more
#: than 8 newer forwardable notices land between two origin reads; the origin
#: reads every 5 s against a 20 s listing cache, so that is >8 notices inside
#: ~20 s from one session — a runaway loop the store's own
#: ``NEW_ITEMS_PER_HOUR`` (30) refuses within the hour anyway. An origin that
#: was OFFLINE across more than 8 notices does miss the oldest (stated in the PR's
#: known limits). ``projection.NOTICES_MAX`` is the read-side twin; a test pins
#: them equal.
NOTICES_ON_WIRE = 8

#: How many unread items per session the owner scans to fill that list. Larger
#: than the list so a burst of EXCLUDED rows (parks, bridge notices) cannot crowd
#: the forwardable ones out, and bounded so the catalogue op never reads a whole
#: inbox.
_NOTICE_SCAN = 4 * NOTICES_ON_WIRE


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
    """Add ``attention``/``notices`` to the rows whose ORIGIN is the viewer. In place.

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
    # The caller checked the viewer's capability, so this origin collects
    # forwarded items: recorded for the node's ``notify`` sentence (m2).
    record_forwarding_origin(Path(root), viewer_device_id)
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
        states = None
    store: Any = None
    try:
        from local_operator.notifications.store import NotificationStore

        store = NotificationStore(root)
    except Exception:  # noqa: BLE001
        logger.debug("remote attention: inbox open failed", exc_info=True)
    for row in mine:
        session_id = str(row["session_id"])
        if states is None:
            # A FAILED READ IS NOT "NOTHING UNSEEN" (review round 2, R2-m2): a
            # row with no ``attention`` is what an acknowledgement looks like,
            # and the origin settles its done row on it — losing the badge of a
            # completion nobody saw. Saying "unknown" makes the origin leave
            # its row alone until a read that could see the store.
            row["attention_unknown"] = True
            state: dict[str, Any] = {}
        else:
            state = states.get(f"session/{session_id}") or {}
        token = state.get("completion_token")
        # ``notify`` False is the completion's own "do not tell anyone" (§14 of
        # the attention contract): honoured here exactly as the local legs do.
        if token and state.get("unseen") and state.get("notify", True):
            row["attention"] = {"token": str(token), "outcome": str(state.get("kind") or "")}
        notices = _newest_notices(store, session_id) if store is not None else []
        if notices:
            row["notices"] = notices


def _newest_notices(store: Any, session_id: str) -> list[dict[str, Any]]:
    """The newest unread, forwardable items of ``session_id``, as wire blocks.

    At most :data:`NOTICES_ON_WIRE`, newest content first (the store's
    ``updated_at DESC`` order), so a reader that cuts the list keeps the newest.
    """
    try:
        items, _cursor = store.list_items(state="unread", session_id=session_id, limit=_NOTICE_SCAN)
    except Exception:  # noqa: BLE001
        logger.debug("remote attention: inbox read failed for %s", session_id, exc_info=True)
        return []
    out: list[dict[str, Any]] = []
    for item in items:
        key = str(item.get("key") or "")
        if key.startswith(NOT_FORWARDED_PREFIXES):
            continue
        out.append(
            {
                # A keyless item is its own topic on the owner (every write is a new
                # row), so its id stands in for the key — the same "one row per
                # owner item" on the origin.
                "key": key or str(item.get("id") or ""),
                "revision": int(item.get("revision") or 1),
                "priority": str(item.get("priority") or "normal"),
                "title": str(item.get("title") or ""),
            }
        )
        if len(out) >= NOTICES_ON_WIRE:
            break
    return out


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
    deliver: bool = True,
) -> str | None:
    """Create or update the origin's row for one finished turn on ``device_id``.

    The TITLE is the conversation's name (the local completion banner's own
    title rule) and the BODY is the same house sentence a local completion of
    that kind carries, then where it ran. ``token`` is the store's ``episode``:
    the same completion re-raised is ``unchanged`` (no unread flip), a new one
    is an update that is unread again. A new completion also raises the
    completion banner off-thread (``deliver``; :func:`_deliver_done_banner`).
    Returns the store's action, or ``None`` when nothing was written.
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
        if deliver and result.action in ("created", "updated", "revived", "absorbed"):
            # Re-read: ``_undismiss`` may have moved the row, and the banner is
            # claimed against the row's CURRENT revision.
            row = store.get_by_key(session_id, done_key(device_id, session_id))
            if row is not None:
                kind = outcome if outcome in notify_copy.CONTEXTS else "complete"
                _deliver_done_banner(
                    root,
                    item_id=str(row["id"]),
                    revision=int(row["revision"]),
                    session_id=session_id,
                    title=name,
                    body=body,
                    subtitle=notify_copy.CONTEXTS.get(kind, ""),
                )
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


def settle_unclaimed_done_items(
    root: Path, rows: Iterable[Any], *, listed_at_ms: int | None = None
) -> int:
    """Mark READ every unread done row the CURRENT listing no longer claims.

    QA round 1, Q1: the settle used to fire only when a key left the watcher's
    in-memory map, so a receipt that landed while no watcher ran (the TUI
    closed, ``lop serve`` restarting) left the row unread for good — a fresh
    process starts with an empty map and never sees the key leave. This reads
    the store's unread ``remote-done:`` rows (one query) and settles each whose
    session IS in this listing with no ``attention`` claim for the row's token:
    the owner acknowledged it. A session absent from the listing — a silent
    device, a deleted session — is left alone, because absence is not an
    acknowledgement. A row whose session claims a DIFFERENT token is also left:
    the same read raises that newer completion as an update. Returns how many
    rows were settled. Never raises.

    ONLY A LISTING NEWER THAN THE ROW MAY SETTLE IT (QA round 2, Q3).
    ``listed_at_ms`` is the wall-clock moment the listing's relay read STARTED
    (``peer_rows.listing_read_at_ms``); a row whose ``updated_at`` is not older
    than that was written from a later read than this one, so this listing's
    "no claim" predates the completion rather than acknowledging it. Without
    the check, ``lop serve``'s 20 s-cached listing, fetched just before a turn
    finished, marked the row the TUI had just raised READ ~5 s later. Both
    times are this device's own clock, so no skew between devices enters.
    ``None`` — a listing of unknown age — settles nothing. A row whose owner
    could not read its attention store (``remote_attention_unknown``) is left
    alone too (R2-m2).
    """
    if listed_at_ms is None:
        return 0
    try:
        listed: dict[str, str] = {}
        for row in rows:
            device_id = str(getattr(row, "owner_device", "") or "")
            session_id = str(getattr(row, "id", "") or "")
            if not device_id or not session_id:
                continue
            if getattr(row, "remote_attention_unknown", False):
                continue
            claim = getattr(row, "remote_attention", None)
            listed[done_key(device_id, session_id)] = str(claim[0]) if claim else ""
        if not listed:
            return 0
        from local_operator.notifications.store import NotificationStore

        store = NotificationStore(root)
        settled = 0
        for item in store.unread_with_key_prefix(f"{DONE_PREFIX}:"):
            key = str(item.get("key") or "")
            if key not in listed:
                continue
            if listed[key]:
                # Still claimed (this token, or a newer one ``raise_done_item``
                # moves the row to on this same read): not acknowledged.
                continue
            if int(item.get("updated_at") or 0) >= int(listed_at_ms):
                # Written from a read newer than this listing (Q3, above).
                continue
            result: Any = store.apply(
                "read",
                ids=[(str(item["id"]), int(item["revision"]))],
                via="remote-attention",
            )
            settled += int(bool(result.changed))
        return settled
    except Exception:  # noqa: BLE001
        logger.debug("remote attention: done settle failed", exc_info=True)
        return 0


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
        existing = store.get_by_key(session_id, row_key)
        if existing is not None and _notice_rev(existing) >= int(revision):
            # THIS revision (or a newer one) is already the row's: a restarted
            # watcher, a second host, or a host reading a stale listing. Decided
            # on the revision BEFORE the write rather than left to the digest
            # (review round 1, m1): the digest also covers the "runs on
            # <device>" body, so a host that read a nameless row would turn a
            # repeat into an update and flip a READ row back to unread, or raise
            # a second ``high`` banner revision for one owner revision.
            return "unchanged"
        result = store.notify(
            session_id=session_id,
            title=title,
            body_md=_location(label),
            priority=priority,
            key=row_key,
            anchor_id=f"rev:{int(revision)}",
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


def _notice_rev(row: dict[str, Any]) -> int:
    """The owner revision a notice row describes (its ``rev:<N>`` anchor), else 0.

    0 for a row written before the anchor was recorded, so the first write
    after an upgrade still goes through the store's digest as before.
    """
    anchor = str(row.get("anchor_id") or "")
    if not anchor.startswith("rev:"):
        return 0
    try:
        return int(anchor[4:])
    except ValueError:
        return 0


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


#: The claimant a done banner records in ``delivered_by``: distinct from the
#: item ladder's ``runtime`` so a release can never undo another leg's claim.
DONE_BANNER_CLAIMANT = "remote-done"


def _deliver_done_banner(
    root: Path,
    *,
    item_id: str,
    revision: int,
    session_id: str,
    title: str,
    body: str,
    subtitle: str = "",
    notify: Any = None,
    background: bool = True,
) -> None:
    """Raise the completion banner for one NEW remote completion, at most once.

    WHY NOT THE ITEM LADDER (review round 1, M1). ``deliver_item`` banners only a
    ``high`` row, and both of its other rungs — the desktop feed and the TUI's
    banner tick — read ``undelivered(priority="high")``, so handing a ``normal``
    done row to it (or promoting the row to ``high``, which would also reorder
    the reel and badge) either raises nothing or changes what the row IS. A
    local background completion is announced by its own leg
    (``OperatorApp._deliver_background_completion``) with a claim, a clickable
    ``detached_notify`` and the house title/body; this is that leg for a
    completion the origin only knows through its inbox row.

    THE CLAIM IS THE ROW'S ``claim_delivery``, so the TUI's watcher and ``lop
    serve``'s — both of which see the same new token — raise ONE banner
    between them, and a row the operator already read (or dismissed) before
    the claim is refused by the store. A notifier that started nothing hands
    the claim back, the local leg's rule: a watermark must not assert a banner
    nobody received.

    GATES AND PRIVACY mirror the local leg: the notifications switch and the
    rig guard (``delivery.gates_refuse``) refuse BEFORE the claim, so a gated
    revision stays claimable for a process with the gates open; the title is
    the conversation's name only when ``session_names_in_notifications`` allows
    it (else the brand), and the body is the house sentence for the outcome,
    never model-written text. ``session_id`` makes the banner clickable: it
    rides to ``lop resume-click``, which opens the peer session by id.

    Runs on a daemon thread by default so a slow notifier never stalls the
    poll; ``background=False`` is the tests' synchronous seam.
    """

    def run() -> None:
        try:
            from local_operator.notifications.delivery import gates_refuse
            from local_operator.notifications.store import NotificationStore
            from local_operator.tui import notify as notify_copy

            if gates_refuse():
                return
            store = NotificationStore(root)
            row = store.get(item_id)
            if row is None or int(row["revision"]) != int(revision):
                # Superseded by a newer completion: that write raises its own.
                return
            # ``per_revision``: one banner per completion with no cooldown, the
            # local leg's per-token rule (R2-M1; see ``claim_delivery``).
            if not store.claim_delivery(item_id, DONE_BANNER_CLAIMANT, per_revision=True):
                return
            banner_title = (
                notify_copy.sanitize_text(title) or notify_copy.BACKGROUND_FALLBACK_TITLE
                if notify_copy.session_names_in_notifications()
                else notify_copy.APP_NAME
            )
            send = notify if notify is not None else notify_copy.detached_notify
            try:
                started = bool(
                    send(
                        notify_copy.argv_safe(banner_title),
                        notify_copy.argv_safe(body),
                        session_id=session_id,
                        subtitle=subtitle,
                    )
                )
            except Exception:  # noqa: BLE001 — a banner is chrome; the row is durable
                started = False
            if not started:
                store.release_delivery(item_id, int(revision), DONE_BANNER_CLAIMANT)
        except Exception:  # noqa: BLE001
            logger.debug("remote attention: done banner failed", exc_info=True)

    if not background:
        run()
        return
    threading.Thread(target=run, name="remote-done-banner", daemon=True).start()


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
        self._notice: dict[tuple[str, str, str], int] = {}

    def observe(
        self,
        rows: Iterable[Any],
        unanswered: Iterable[str] = (),
        *,
        listed_at_ms: int | None = None,
    ) -> None:
        """Apply one listing read's completion and notice edges. Never raises.

        ``listed_at_ms`` is when the listing's read started (wall clock, ms);
        omitted, it is looked up for ``rows`` from ``peer_rows``' cache, which
        answers only for the very tuple ``read_listing`` handed out — so both
        TUI callers and ``lop serve`` get it without threading a timestamp
        through the sidebar. Unknown means no settle (Q3).
        """
        try:
            from local_operator.session.peer_rows import (
                completion_edges,
                listing_read_at_ms,
                notice_edges,
            )

            if listed_at_ms is None:
                listed_at_ms = listing_read_at_ms(self.root, rows)
            rows = tuple(rows)
            silent = tuple(unanswered)
            done, done_state = completion_edges(self._done, rows, unanswered=silent)
            self._done = done_state
            settle_unclaimed_done_items(self.root, rows, listed_at_ms=listed_at_ms)
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


#: Where an OWNER records the origins that have asked for these fields over a
#: ``REMOTE_ATTENTION_V1`` link: ``{device_id: first-seen epoch}`` directly
#: under the config root — NOT in the networks directory, whose every ``*.json``
#: is read (and a non-record quarantined) as a network record. WHY A FILE
#: (review round 1, m2): a link's capabilities live only in the relay process,
#: while the ``notify`` sentence is composed in the session process; the
#: sentence must promise forwarding only to an origin that has shown it
#: collects it. Written once per (process, origin) — a new
#: origin, not every catalogue read — so it costs no steady-state I/O.
ORIGINS_FILE = "remote-attention-origins.json"
_recorded_origins: set[tuple[str, str]] = set()
_origins_lock = threading.Lock()


def _origins_path(root: Path) -> Path:
    return Path(root) / ORIGINS_FILE


def record_forwarding_origin(root: Path, device_id: str) -> None:
    """Note that ``device_id`` reads forwarded completions/notices. Never raises."""
    if not device_id:
        return
    marker = (str(Path(root)), device_id)
    if marker in _recorded_origins:
        return
    try:
        import json
        import os
        import time

        with _origins_lock:
            path = _origins_path(root)
            known = forwarding_origins(root)
            if device_id not in known:
                known[device_id] = time.time()
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
                # 0600 like its siblings (``attention.db``, ``notifications.db``):
                # it names the mesh devices this one serves (review round 2,
                # R2-m3). Created with the mode rather than chmod-ed after, so
                # no umask-wide window exists; ``os.replace`` keeps the mode.
                fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    handle.write(json.dumps(known, sort_keys=True))
                os.chmod(tmp, 0o600)
                os.replace(tmp, path)
            _recorded_origins.add(marker)
    except Exception:  # noqa: BLE001 — a marker is a hint for one sentence
        logger.debug("remote attention: could not record origin", exc_info=True)


def forwarding_origins(root: Path) -> dict[str, float]:
    """The recorded origins (see :data:`ORIGINS_FILE`), or ``{}``. Never raises."""
    try:
        import json

        data = json.loads(_origins_path(root).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 — absent or unreadable is "none recorded"
        return {}
    if not isinstance(data, dict):
        return {}
    return {str(k): float(v) for k, v in data.items() if isinstance(v, (int, float))}


def forwarded_notice_sentence(root: Path, session_id: str, key: str = "") -> str:
    """The clause a peer-placed session's ``notify`` result adds, or ``""``.

    Model-facing English (a tool result, like the sentence it extends), so not
    catalogued. A bare "Notified" on the node read as if the operator had been
    told; the item is in THIS device's inbox and reaches the operator only when
    the origin forwards it, so the model is told which of the two happens.
    "Will be forwarded" is said ONLY for an origin recorded as collecting
    forwarded items (:func:`record_forwarding_origin`); any other origin — an
    older build, or one that has not read this device since updating — gets
    "it stays on <this device>" (review round 1, m2). ``""`` for every session
    not placed here by another device, and for a ``key`` this module never
    forwards — an ordinary local session's result is unchanged byte for byte.
    """
    try:
        if key and key.startswith(NOT_FORWARDED_PREFIXES):
            return ""
        from local_operator.session.placement import read_stamp

        stamp = read_stamp(Path(root), session_id)
        if stamp is None or stamp.placement.mode != "peer":
            return ""
        origin = str((stamp.origin or {}).get("source_device") or "")
        if not origin:
            return ""
        here, there = _device_names(Path(root), origin)
        if origin in forwarding_origins(Path(root)):
            return (
                f" Posted on {here}; it will be forwarded to {there}, where this session started."
            )
        return (
            f" Posted on {here}; it stays on {here}, because {there}, where this session"
            " started, is not known to collect forwarded notices."
        )
    except Exception:  # noqa: BLE001
        return ""


def _device_names(root: Path, origin: str) -> tuple[str, str]:
    """``(this device's name, the origin's name)`` from the membership records.

    An origin that is no member here (it left, or was never in this network)
    reads "the device where this session started" rather than its raw device id
    (QA round 1, Q2): an id names nothing the operator would recognise.
    """
    here = "this device"
    there = "its origin device"
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
