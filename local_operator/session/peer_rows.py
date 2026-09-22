"""Sessions OTHER devices hold, as rows THIS device's list can render.

WHY THIS MODULE EXISTS, and why it is not a second projection. ``mesh-ui.md``
§1.3 landed the sidebar's ``⇄`` locality mark, its per-device heading and the
picker's ``locality``/``owner_*`` columns against a CONTRACT FIXTURE, because the
producer was not in that slice — the review recorded it twice (round 4 MINOR 3:
"the ``⇄`` has no remote-row producer"; design round 1: "the producer is not
merely a rendering follow-up; it is what makes the slice's own action
observable"). ``/new remote <peer>`` is the one WRITE the slice ships, and
without this module the session it creates exists nowhere the user can see it.
This is the missing producer: the relay's peer projection, turned into the
``SessionRow``s every surface already knows how to paint.

READ-ONCE, BOUNDED, AND NEVER A DIAL FROM A FRAME. The rows come from
``RelayPeerCatalog``, which is ONE control call to THIS device's own relay (the
relay fans out to the peers over the links it already holds) — never one socket
per row, and never from the paint path. Two things keep it off the frame:

* a TTL (``_TTL_S``), because the sidebar polls every two seconds and a listing
  that dials peers does not belong on a two-second timer;
* the zero-peer short circuit, which is the same property
  ``network/projection.py`` states for itself: a device with no relay record
  reads its own disk and returns nothing, issuing no call at all. That is what
  keeps a machine outside any mesh — every existing install, and every test and
  capture — byte-identical to before.

NOTHING HERE RAISES. A listing that cannot read the projection is a listing
without peer rows, not a broken sidebar: the failure this file must not repeat is
a swallowed read rendering as "there is nothing there", which is why an empty
result is the honest answer and the caller has no error path to forget.
"""

from __future__ import annotations

import time
from pathlib import Path

from local_operator.resume import UNTITLED_CONVERSATION, SessionRow

#: How long one projection answer is reused. Chosen against the sidebar's own
#: two-second poll: long enough that a peer listing is a rare event, short enough
#: that a session created on a peer (``/new remote``) surfaces while the user is
#: still looking at the list it should appear in.
_TTL_S = 20.0

#: ``config root`` → ``(monotonic read time, rows)``. Keyed by root so an
#: isolated ``LOCAL_OPERATOR_CONFIG_DIR`` (every test and capture) cannot read a
#: real install's answer, and module-level so the sidebar's poll thread and the
#: app's own guard share ONE read rather than two.
_CACHE: dict[str, tuple[float, tuple[SessionRow, ...]]] = {}


def clear_cache() -> None:
    """Forget every cached read. For tests: a fixture's rows must not leak on."""
    _CACHE.clear()


def peer_session_rows(
    root: Path | None = None,
    *,
    now: float | None = None,
    ttl_s: float = _TTL_S,
    catalog: object | None = None,
) -> tuple[SessionRow, ...]:
    """Every session another device is holding, as rows for THIS device's list.

    Returns ``()`` — not ``None``, and without raising — for every reason there is
    nothing to show: no network on this device, no relay record, a relay that did
    not answer, or a projection that came back empty. The caller paints the list
    it has.

    ``catalog`` is the injection seam for a test or a future transport-owned
    catalogue (``network/projection.PeerCatalog``); production passes nothing and
    gets this device's own relay.
    """
    key = "" if root is None else str(root)
    moment = time.monotonic() if now is None else now
    cached = _CACHE.get(key)
    if cached is not None and ttl_s > 0 and moment - cached[0] < ttl_s:
        return cached[1]
    rows = _read(root, catalog)
    _CACHE[key] = (moment, rows)
    return rows


def peer_session_row(session_id: str, root: Path | None = None) -> SessionRow | None:
    """The CACHED row for ``session_id``, or ``None``. Never reads, never dials.

    Deliberately cache-only: this answers "is this id a session on another
    device", which the resume path asks BEFORE it decides whether to boot a
    session, and a guard that dialled would put a peer's latency in front of
    every ``/resume``. It does not even refresh the cache — that is the sidebar
    poll's job (``peer_session_rows`` on its own cadence), and a miss here is
    simply a miss: the local path is unchanged either way.
    """
    cached = _CACHE.get("" if root is None else str(root))
    if cached is None:
        return None
    for row in cached[1]:
        if row.id == session_id:
            return row
    return None


def _read(root: Path | None, catalog: object | None) -> tuple[SessionRow, ...]:
    """One projection read, or ``()``. Every failure is an empty list."""
    if catalog is None:
        try:
            from local_operator.network import store
        except Exception:  # pragma: no cover - the mesh package is always importable
            return ()
        try:
            if store.find_own_relay(root) is None:
                # NO RELAY, NO WORK: the zero-peer property, measured as "did
                # this process issue a call" rather than asserted in a comment.
                return ()
        except Exception:  # noqa: BLE001 - an unreadable store is no projection
            return ()
        try:
            from local_operator.network.projection import RelayPeerCatalog

            catalog = RelayPeerCatalog(root)
        except Exception:  # noqa: BLE001
            return ()
    try:
        peers = {peer.device_id: peer for peer in catalog.peers()}  # type: ignore[attr-defined]
        raw = catalog.rows()  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001 - a refused or timed-out relay is no rows
        return ()
    if not peers:
        return ()
    rows: list[SessionRow] = []
    for peer_row in raw:
        session_id = str(getattr(peer_row, "session_id", "") or "")
        if not session_id:
            continue
        facts = peers.get(str(getattr(peer_row, "device_id", "") or ""))
        if facts is None:
            # A row from a device the peers answer did not include: the two
            # halves of one projection disagree, so this device cannot say where
            # the session lives and must not guess a heading for it.
            continue
        rows.append(
            SessionRow(
                session_id,
                float(getattr(peer_row, "started", 0.0) or 0.0),
                # ONE NAME FOR ONE CONDITION, shared with the local half
                # (design round 2, D14). This fell back to the session's own
                # 12-hex id while a nameless row THIS device holds paints
                # ``Untitled conversation`` (``session/catalog.py``), so one
                # missing name read two ways on the same screen — and the id
                # was on the row a reader can least resolve by looking around
                # them.
                str(getattr(peer_row, "conversation_name", "") or UNTITLED_CONVERSATION),
                live_state=_live_state(peer_row),
                pending=getattr(peer_row, "pending", None),
                kind=str(getattr(peer_row, "kind", "") or ""),
                locality="remote",
                owner_device=facts.device_id,
                owner_device_name=facts.name,
                reachable=bool(facts.reachable),
                unreachable_reason=str(facts.reason or ""),
            )
        )
    return tuple(rows)


def _live_state(peer_row: object) -> str:
    """The peer's own state token, in THIS list's vocabulary.

    ``SessionRow.live_state`` is the token ``row_state_mark`` ranks: ``busy``,
    ``idle``, ``attached``, ``wedged`` or ``""``. The federated row's ``state``
    is the peer's own word for the same thing (``to_row_json``), so it is passed
    through when it is one of the four, and otherwise derived from the two
    booleans the transport does define — a session another terminal is watching
    is ``attached``, and a session with work in flight is ``busy``. An
    unrecognised word becomes ``""``: a cold row, which renders as "no claim"
    rather than as a state this list invented.
    """
    state = str(getattr(peer_row, "state", "") or "")
    if state in ("busy", "idle", "attached", "wedged"):
        return state
    if bool(getattr(peer_row, "detached", False)):
        return "attached"
    if bool(getattr(peer_row, "busy", False)):
        return "busy"
    return ""
