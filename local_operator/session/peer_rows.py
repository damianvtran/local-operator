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
from typing import NamedTuple

from local_operator.resume import UNTITLED_CONVERSATION, SessionRow
from local_operator.session.catalog import live_state_from_flags

#: How long one projection answer is reused. Chosen against the sidebar's own
#: two-second poll: long enough that a peer listing is a rare event, short enough
#: that a session created on a peer (``/new remote``) surfaces while the user is
#: still looking at the list it should appear in.
_TTL_S = 20.0

#: ``config root`` → ``(monotonic read time, rows, unanswered peers)``. Keyed by
#: root so an isolated ``LOCAL_OPERATOR_CONFIG_DIR`` (every test and capture)
#: cannot read a real install's answer, and module-level so the sidebar's poll
#: thread and the app's own guard share ONE read rather than two. The unanswered
#: peers ride in the same entry because they are one answer: the relay reports
#: the device that did not reply WITH the rows the others did, and a second read
#: for the second half would be the second staleness rule this module avoids.
_CACHE: dict[str, tuple[float, tuple[SessionRow, ...], tuple[UnansweredPeer, ...]]] = {}


def clear_cache() -> None:
    """Forget every cached read. For tests: a fixture's rows must not leak on."""
    _CACHE.clear()


class UnansweredPeer(NamedTuple):
    """A device in this device's networks that did not answer a listing read.

    THE DEVICE THAT IS GONE NEEDS A NAME OF ITS OWN (UX round 3, U16). The
    relay already reports this — ``_fan_out_catalog`` contributes a
    ``reachable: false`` block with a reason and NO rows for a peer that does not
    reply, precisely so "the device exists and is switched off" is sayable — and
    this module used to DROP it, because it only ever returned rows. The sidebar
    reads rows, so a peer that stopped answering lost its whole section, and the
    user's six sessions disappeared with nothing said: "my peer has no
    sessions" and "my peer is gone" were one picture.

    ``reason`` is the relay's own sentence. It is carried rather than formatted
    away because the tooltip on a REAL row shows it (``session_sidebar``'s
    ``location`` line) and the two must say the same thing about the same state.
    """

    device_id: str
    name: str
    reason: str


def peer_session_rows(
    root: Path | None = None,
    *,
    now: float | None = None,
    ttl_s: float = _TTL_S,
    catalog: object | None = None,
) -> tuple[SessionRow, ...]:
    """Every session another device is holding, as rows for THIS device's list.

    ONE ROW PER SESSION, and that is a property of the SESSION rather than of the
    read (QA round 1, Q1): a session lives on ONE device, and a device that shares
    two networks with this one is still one device holding it. The relay's fan-out
    asks every shared network's member table, so a session on such a device is
    reported once per shared network — two rows for one conversation, four for two
    — and every reader of this function inherited it (the chat list de-duplicated
    by accident because it keys rows by id, the search did not, and
    ``peer_catalogue`` counted memberships instead of sessions). The de-duplication
    belongs HERE rather than in each reader: this function is the published answer
    to "what does the mesh hold", and a consumer should not have to know how many
    networks two devices happen to share.

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
    return _read_all(root, catalog, moment, key)[0]


def unanswered_peers(
    root: Path | None = None,
    *,
    now: float | None = None,
    ttl_s: float = _TTL_S,
    catalog: object | None = None,
) -> tuple[UnansweredPeer, ...]:
    """Peers this device could not reach for the last listing read.

    One answer with :func:`peer_session_rows`, not a second read: the relay
    reports the device that did not reply beside the rows the others did, so both
    functions read the same cache entry and a caller that asks for both cannot
    see a peer listed and missing at the same instant.

    IT DOES NOT REQUIRE A PRIOR POLL, and that is deliberate:
    ``peer_session_row`` is cache-only because it sits in front of every
    ``/resume``, but a selector that answered "no peers are silent" because
    nobody had polled yet would be the same silent-empty failure this function
    exists to remove. A cold call reads, exactly as an empty cache makes
    :func:`peer_session_rows` read.

    The result is a peer the relay NAMED as unanswered — nothing here guesses
    from an absent section, which would report every peer with no sessions as
    gone.
    """
    key = "" if root is None else str(root)
    moment = time.monotonic() if now is None else now
    cached = _CACHE.get(key)
    if cached is not None and ttl_s > 0 and moment - cached[0] < ttl_s:
        return cached[2]
    return _read_all(root, catalog, moment, key)[1]


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


def _read_all(
    root: Path | None, catalog: object | None, moment: float, key: str
) -> tuple[tuple[SessionRow, ...], tuple[UnansweredPeer, ...]]:
    """One projection read, cached under ``key``, as ``(rows, unanswered)``.

    The ONE read both public functions hand out. It is a function rather than
    two because the two halves are one relay answer — a caller that asked for the
    rows and then for the silent peers must not be able to see two different
    fan-outs of a mesh that is moving underneath it.
    """
    rows, unanswered = _read(root, catalog)
    _CACHE[key] = (moment, rows, unanswered)
    return rows, unanswered


def _read(
    root: Path | None, catalog: object | None
) -> tuple[tuple[SessionRow, ...], tuple[UnansweredPeer, ...]]:
    """One projection read, or ``((), ())``. Every failure is an empty answer."""
    if catalog is None:
        try:
            from local_operator.network import store
        except Exception:  # pragma: no cover - the mesh package is always importable
            return (), ()
        try:
            if store.find_own_relay(root) is None:
                # NO RELAY, NO WORK: the zero-peer property, measured as "did
                # this process issue a call" rather than asserted in a comment.
                return (), ()
        except Exception:  # noqa: BLE001 - an unreadable store is no projection
            return (), ()
        try:
            from local_operator.network.projection import RelayPeerCatalog

            catalog = RelayPeerCatalog(root)
        except Exception:  # noqa: BLE001
            return (), ()
    try:
        peers = {peer.device_id: peer for peer in catalog.peers()}  # type: ignore[attr-defined]
        raw = catalog.rows()  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001 - a refused or timed-out relay is no rows
        return (), ()
    if not peers:
        return (), ()
    rows: list[SessionRow] = []
    # Keyed by id, and the FIRST row for an id wins: the duplicates this collapses
    # are one device's own answer repeated per shared network, so they are the same
    # row and the order decides nothing. A row from a DIFFERENT device claiming an
    # id already taken is the routing ambiguity the mesh exists to prevent
    # (``relay._op_session_create``), not a case this function can resolve — the
    # first one the projection named wins, and the id stays one row.
    seen: set[str] = set()
    for peer_row in raw:
        session_id = str(getattr(peer_row, "session_id", "") or "")
        if not session_id or session_id in seen:
            continue
        facts = peers.get(str(getattr(peer_row, "device_id", "") or ""))
        if facts is None:
            # A row from a device the peers answer did not include: the two
            # halves of one projection disagree, so this device cannot say where
            # the session lives and must not guess a heading for it.
            continue
        seen.add(session_id)
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
    # THE PEERS THAT DID NOT ANSWER, and the exclusion is by DEVICE rather than
    # by row: a peer the relay marked unreachable contributes no rows, so the two
    # sets are disjoint by construction — but a device that both answered an
    # earlier read and failed this one would otherwise be reported twice, once as
    # a section with rows and once as a silent heading. `answered` is what makes
    # that impossible without asking the relay twice.
    answered = {row.owner_device for row in rows}
    unanswered = tuple(
        UnansweredPeer(device_id=device_id, name=str(facts.name or ""), reason=str(facts.reason))
        for device_id, facts in peers.items()
        if not facts.reachable and device_id not in answered
    )
    return tuple(rows), unanswered


def _live_state(peer_row: object) -> str:
    """The peer's own state token, in THIS list's vocabulary.

    ``SessionRow.live_state`` is the token ``row_state_mark`` ranks: ``busy``,
    ``idle``, ``attached``, ``wedged`` or ``""``. The federated row's ``state`` is
    the peer's own word for the same thing (``to_row_json``), so it is passed
    through when it is one of the four. Two of the peer's other words are decided
    here instead:

    * ``stored`` — the relay's word for a session that exists on the peer with NO
      runtime behind it (``RelayPeerCatalog._stored_rows``, which stamps each such
      row ``detached: True`` because nothing can be watching a session that is not
      running). That is the same condition the local half paints with an empty
      ``live_state`` — a cold row, no claim — so it must not reach the flag ladder
      below: that ladder answers for a RUNNING session, and letting a stored row
      into it made the peer group claim a live, watched session that does not
      exist.
    * anything else — derived from the two booleans the transport does define,
      through the SAME reading the local half uses
      (``session.catalog.live_state_from_flags``), so the two halves of one list
      cannot answer "is this running, and is anyone watching it" differently. In
      particular ``detached`` means NOBODY IS WATCHING: a live but unwatched peer
      session is ``idle``, never ``attached``.
    """
    state = str(getattr(peer_row, "state", "") or "")
    if state in ("busy", "idle", "attached", "wedged"):
        return state
    if state == "stored":
        return ""
    return live_state_from_flags(peer_row)
