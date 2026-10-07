"""Sessions OTHER devices hold, as rows THIS device's list can render.

WHY THIS MODULE EXISTS, and why it is not a second projection. ``mesh-ui.md``
§1.3 landed the sidebar's ``⇄`` locality mark and the picker's
``locality``/``owner_*`` columns against a CONTRACT FIXTURE, because the
producer was not in that slice — the review recorded it twice (round 4 MINOR 3:
"the ``⇄`` has no remote-row producer"; design round 1: "the producer is not
merely a rendering follow-up; it is what makes the slice's own action
observable"). ``/new remote <peer>`` is the one WRITE the slice ships, and
without this module the session it creates exists nowhere the user can see it.
This is the missing producer: the relay's peer projection, turned into the
``SessionRow``s every surface already knows how to paint. (The per-device
HEADING those rows once filed under retired with the operator's convergence: a
remote row takes the ordinary bins — see ``tui/session_sidebar._unpinned_rank``.)

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

TWO FIELDS ARE RESOLVED HERE, at the producer, so every surface that paints a
remote row gets one answer: the membership's network NAME
(``owner_network_name``, for the tooltip's device-AND-network clause —
``_network_names`` carries its why) and the row's ordering birth
(``created_at``, stamped from the peer's ``started`` claim — the row
construction carries that why).
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Mapping
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


def read_listing(
    root: Path | None = None,
    *,
    now: float | None = None,
    ttl_s: float = _TTL_S,
    catalog: object | None = None,
) -> tuple[tuple[SessionRow, ...], tuple[UnansweredPeer, ...]]:
    """ONE listing read, handing back BOTH halves: the rows and the silence.

    WHY THIS IS A FUNCTION RATHER THAN TWO CALLS (agent review round 1, R-1).
    The two halves are one relay answer, and a caller that needs both used to
    ask twice — ``peer_session_rows`` then ``unanswered_peers`` — which is only
    the same read while the first one completes inside the TTL: the cache entry
    carries the moment the read STARTED (``_read_all``), and a listing that
    spends its own documented budget (``relay.LISTING_CLIENT_TIMEOUT_S``, which
    equals ``_TTL_S``) is exactly the case this exists for — a member that
    black-holes, i.e. the silence a resolution miss must consult. The second
    call then re-dialled, and the re-dial's ROWS were discarded by
    ``unanswered_peers``, so an id the listing had just been seen to hold could
    answer "not a peer's". Both failures are impossible here by construction:
    one call, one read, one answer.

    ``ttl_s=0`` on a resolution miss is the ONE FORCED READ those seams pay, so
    both halves describe the same fan-out. Callers that want one half keep
    :func:`peer_session_rows` and :func:`unanswered_peers`, which are now this
    function's two projections.
    """
    key = "" if root is None else str(root)
    moment = time.monotonic() if now is None else now
    cached = _CACHE.get(key)
    if cached is not None and ttl_s > 0 and moment - cached[0] < ttl_s:
        return cached[1], cached[2]
    return _read_all(root, catalog, moment, key)


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
    return read_listing(root, now=now, ttl_s=ttl_s, catalog=catalog)[0]


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
    return read_listing(root, now=now, ttl_s=ttl_s, catalog=catalog)[1]


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


def seed_peer_row(root: Path | None, row: SessionRow) -> None:
    """Merge a KNOWINGLY-LIVE peer row into the cache, for resolutions to find.

    WHY THIS EXISTS (the operator-blocking defect, 2026-10-06). ``/new remote
    <peer>`` is answered by the peer's mint: the id in that reply names a
    conversation that exists. But every other local route resolves ids through
    the cached listing (``peer_session_row`` above), and a listing read a moment
    before the create cannot contain the id — so the message sent to the
    just-created conversation was refused with "This conversation no longer
    exists, so your message wasn't sent", and the id resolved only when the
    next federated sidebar read landed. The create's own reply is authoritative
    local knowledge, so the route that holds it merges the row in here: the very
    next resolution finds it with zero wire cost (the miss path's live read —
    ``remote_open.remote_row_for``'s ``ttl_s=0`` — remains the fallback for
    every id no reply announced).

    MERGE; A ONE-ROW ENTRY NEVER ANSWERS A LISTING. A cache entry is this
    module's ONE answer about the whole mesh, so it is only ever built from a
    read — except here, where a row is KNOWN rather than read. With a listing
    already cached, the row merges into it, keeping its age (``moment``) and its
    unanswered peers: the row is visible on the next poll, and the poll's own
    refresh schedule is untouched. With NO listing yet, the entry is stamped
    ALREADY-STALE (``-inf``): the cache-only lookup answers from it, and every
    TTL-respecting read — the sidebar's poll, the silent-peer read — sees it
    expire immediately and pays its read in full, exactly as it would on a cold
    cache. That is the property a one-row entry must never be able to break:
    "device D holds this row" is what the create proved; "and nothing else" it
    did not.

    SAME DEVICE, SAME ID REPLACES — one conversation, one row, the same
    (device, id) key ``_read`` de-duplicates by — while a same-id row for
    ANOTHER device is the different conversation ``_read`` keeps separate.
    ``unanswered`` rides as-is when merging: it is a fact about the last LISTING
    read, and a create is not one.
    """
    key = "" if root is None else str(root)
    device = str(row.owner_device or "")
    cached = _CACHE.get(key)
    if cached is None:
        _CACHE[key] = (float("-inf"), (row,), ())
        return
    moment, rows, unanswered = cached
    merged: list[SessionRow] = []
    replaced = False
    for existing in rows:
        if existing.id == row.id and str(existing.owner_device or "") == device:
            merged.append(row)
            replaced = True
        else:
            merged.append(existing)
    if not replaced:
        merged.append(row)
    _CACHE[key] = (moment, tuple(merged), unanswered)


def select_peer_session(
    target: str, rows: Iterable[SessionRow]
) -> tuple[SessionRow | None, tuple[SessionRow, ...]]:
    """Resolve one id-or-name target to a SINGLE fetched peer row. Pure; no I/O.

    THE ONE SPELLING of "which session did they mean" for every surface that
    reaches a peer's sessions from here (the pilot verbs and the session-plane
    verbs on ``lop network sessions`` today; the tools that import this helper
    directly in the companion slice). The semantics mirror the local resolver
    (``mobile/peer_send.resolve_peer_target``) minus the tiers that name local
    rows only — a pid, the stored-session fallback, and the team role
    vocabulary — none of which exist on the rows this reads.

    The tiers, in order:

    * an EXACT whole-value match — conversation name first, then session id —
      resolves silently: a full name or id is evidence, not a guess. More than
      one exact match is REFUSED with the candidates rather than picked
      between (the wrong-recipient hazard the local exact tier exists for);
    * a case-insensitive SUBSTRING of either field resolves when it is the only
      match, and is refused WITH the candidates when it is not.

    Returns ``(row, ())`` for a resolution, ``(None, candidates)`` for an
    ambiguity (EXACT-tier ambiguities are ordered by the field that matched,
    then by input order; substring ambiguities keep input order), and
    ``(None, ())`` for no match.

    THE CWD-BASENAME ARM IS ABSENT BY DATA, not by choice: the local resolver
    also matches a row's working-directory basename, and the row shape this
    module publishes (``resume.SessionRow``) carries no cwd — the projection
    read at ``_read`` drops it — so there is no third field to read here. A
    third address field is a row/wire change, not a selector one.
    """
    needle = (target or "").strip().lower()
    ordered = list(rows)
    if not needle:
        return None, ()
    exact = [
        (rank, order, row)
        for order, row in enumerate(ordered)
        if (rank := _exact_field_rank(row, needle)) is not None
    ]
    if len(exact) > 1:
        exact.sort(key=lambda item: (item[0], item[1]))
        return None, tuple(row for _rank, _order, row in exact)
    if len(exact) == 1:
        return exact[0][2], ()
    matches = tuple(row for row in ordered if _address_contains(row, needle))
    if len(matches) > 1:
        return None, matches
    if len(matches) == 1:
        return matches[0], ()
    return None, ()


def _address_fields(row: SessionRow) -> tuple[str, str]:
    """The fields a target matches against, in precedence order.

    ``name`` then ``id`` — the local resolver's field order with the cwd arm
    dropped, because a peer row has no cwd to read (see
    :func:`select_peer_session`). The order is load-bearing in the exact tier:
    it is the rank a multiple-exact-match refusal sorts its candidates by.
    """
    return (str(row.name or ""), str(row.id or ""))


def _address_contains(row: SessionRow, needle: str) -> bool:
    """Whether ``needle`` is a SUBSTRING of any addressed field.

    ``needle`` arrives lowercased by the caller.
    """
    return any(needle in field.lower() for field in _address_fields(row))


def _exact_field_rank(row: SessionRow, needle: str) -> int | None:
    """The precedence rank of the first field that EQUALS ``needle``, else None.

    Whole-value equality, not containment (``manager`` must not exactly match
    ``team: manager``), and case-insensitively: a name is typed by a person.
    ``needle`` arrives already lowercased by the caller.
    """
    for rank, field in enumerate(_address_fields(row)):
        if field.lower() == needle:
            return rank
    return None


class RemotePark(NamedTuple):
    """One LIVE parked request on a peer device, as the origin sees it.

    The origin cannot answer a remote ALLOW — the OWNER takes one only as a
    signature from a device the operator has paired with it (a challenge minted
    there, answered with its key), so the requesting device's own gesture is
    refused by the owner's runtime by design. What a park is worth at the origin
    is the FACT that a person is needed on ``device``: this is the tuple every
    origin surface composes from — the toast, the OS banner, and the card hint's
    device name.

    ``device_name`` can be empty (a peer that never reported a name), so a
    reader falls back to ``device_id`` exactly as the lifecycle router does
    (``app._remote_owner_facts``' label). ``name`` is the conversation title,
    which is what the banner's TITLE should say while its body says the state.
    """

    session_id: str
    device_id: str
    device_name: str
    kind: str
    name: str


def park_edges(
    previous: Mapping[tuple[str, str], str],
    rows: Iterable[SessionRow],
    *,
    unanswered: Iterable[str] = (),
) -> tuple[tuple[RemotePark, ...], dict[tuple[str, str], str]]:
    """New park EPISODES among ``rows``, and the map to pass back next read.

    Pure: no writes, no dials, no cache. Edges are read off the SAME peer rows
    the sidebar already polls (``_TTL_S``-cached, one relay call per TTL), so a
    park announces itself on the origin with no wire change and no new timer —
    the freshness budget is the caller's poll plus that row cache.

    ONE EPISODE PER (device, session, kind), and the key carries the DEVICE as
    well as the id because the two are not interchangeable: a session id is
    meant to be unique across the mesh (``relay._op_session_create`` keeps two
    devices from minting one id), but if one ever appears twice the two rows
    are two conversations — a map keyed on the id alone would let the first
    device's park swallow the second's notice, the same cross-device collapse
    ``_read`` refuses to make.

    * appear — a key or a kind the previous read did not carry: fire once.
    * kind change (ask -> approval) — a new episode, because it is a new
      remedy (the approval needs a presence gesture; an ask does not).
    * clear — the key leaves the returned map; the caller owns the card that
      said otherwise (the notice is withdrawn, never left to contradict the
      answer that ended the park).
    * re-park after a clear — the key is absent from ``previous``, so it fires
      again: one notice per episode, and a second park IS a second episode.
    * SILENCE — ``unanswered`` names the devices that did not answer this read
      (``peer_rows.unanswered_peers``), and a key belonging to one of them is
      CARRIED into the returned map rather than cleared: a refused or
      timed-out read is not the answer landing, and treating it as one would
      withdraw the card and then re-announce the same live park the moment the
      peer recovered (agent review round 1, MINOR-1). When the device answers
      again, an absent key is a real clear and a present one with the same kind
      is no edge — so the episode survives the outage without a second notice.
      A WHOLE-RELAY failure is the RESIDUAL, stated rather than hidden:
      ``_read``'s empty answer delivers neither rows nor names, so an outage and
      an all-clear are indistinguishable at this seam and this helper READS IT
      AS A CLEAR — the caller withdraws the card, and the same park fires again
      as a second episode once the relay answers. Distinguishing the two needs
      an answered/not-answered signal out of ``_read`` itself; that is recorded
      as the follow-up (round-2 review NIT-1, PR body), not claimed as handled.

    THE STORED-HALF CAVEAT, and it is the discriminator rather than a filter
    here: a park counts only when ``pending`` is present AND ``live_state`` is
    non-empty. The catalogue's STORED half translates an unread completion
    into ``pending:"ask"`` (``relay.py``; the row's state reads ``stored``),
    and ``_live_state`` maps that word to ``""`` because no runtime is behind
    the row — treating it as a park would announce a turn that finished on the
    peer hours ago and needs nobody. A LIVE parked row always carries a state
    word from the four ``_live_state`` passes through, so the discriminator
    cannot drop a real park; the helper's tests pin both fixture shapes.
    """
    current: dict[tuple[str, str], str] = {}
    fresh: dict[tuple[str, str], RemotePark] = {}
    for row in rows:
        kind = str(row.pending or "")
        if not kind or not row.live_state:
            continue
        key = (str(row.owner_device or ""), row.id)
        current[key] = kind
        fresh[key] = RemotePark(
            session_id=row.id,
            device_id=str(row.owner_device or ""),
            device_name=str(row.owner_device_name or ""),
            kind=kind,
            name=str(row.name or ""),
        )
    state = dict(current)
    silent = {str(device_id) for device_id in unanswered}
    for key, kind in previous.items():
        if key not in state and key[0] in silent:
            state[key] = kind
    edges = tuple(fresh[key] for key, kind in current.items() if previous.get(key) != kind)
    return edges, state


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


def _started_epoch(peer_row: object) -> float:
    """The peer's ``started`` claim as an epoch — or ``0.0``, NO claim, for anything else.

    ONE READING FOR BOTH consumers in ``_read`` (the row's ``mtime`` and its
    ``created_at``), because the two must not disagree about when the peer says
    the row began.

    WHY A BOOL IS REFUSED RATHER THAN COERCED (operator report).
    The live half of a federated listing published ``SessionRecord.started`` —
    the "has run a real turn" BOOL — under this key, and ``float(True)`` is
    ``1.0``: an epoch second into 1970, rendered by the desktop sidebar as
    "56y" and filed under "Older". ``bool`` is excluded explicitly because
    ``isinstance(True, int)`` is True: a claim that is not a number is NO
    claim, and it lands where a missing key lands — ``0.0``, "an unknown start
    sorts last" — never minted into an epoch. The type check is also what
    keeps this reader's "nothing here raises" contract for a string claim,
    where a bare ``float()`` would have raised into the sidebar's poll.
    """
    value = getattr(peer_row, "started", 0.0)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    return float(value or 0.0)


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
    # The membership names for the tooltip's device-AND-network clause, read
    # ONCE per listing (this whole read is TTL-cached above). Guarded on
    # ``root`` because a root-less call is a test's injected-catalogue call and
    # must not wander into the developer's real store; production always names
    # its root.
    network_names = _network_names(root) if root is not None else {}
    rows: list[SessionRow] = []
    # KEYED BY ``(owner_device, session_id)``, AND THE FIRST ROW FOR A KEY WINS (review
    # round 1, MINOR 2). The duplicates this collapses are one device's own answer
    # repeated per shared network, so they are the same row and the order decides
    # nothing. Keying on the id ALONE was global across devices: two devices reporting
    # one id — which is the permanent routing ambiguity the mesh exists to prevent
    # (``relay._op_session_create``) — collapsed to a single row silently, filed under
    # the first device's heading, and a user's two conversations looked like one. With
    # the device in the key every case this function is FOR still collapses (same
    # device, same id, one row per membership) and the cross-device collision stays
    # two rows, which is the routing ambiguity being visible rather than hidden.
    seen: set[tuple[str, str]] = set()
    for peer_row in raw:
        session_id = str(getattr(peer_row, "session_id", "") or "")
        device_id = str(getattr(peer_row, "device_id", "") or "")
        if not session_id or (device_id, session_id) in seen:
            continue
        facts = peers.get(device_id)
        if facts is None:
            # A row from a device the peers answer did not include: the two
            # halves of one projection disagree, so this device cannot say where
            # the session lives and must not guess a heading for it.
            continue
        seen.add((device_id, session_id))
        rows.append(
            SessionRow(
                session_id,
                _started_epoch(peer_row),
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
                owner_network_name=network_names.get((facts.device_id, facts.network_id), ""),
                reachable=bool(facts.reachable),
                unreachable_reason=str(facts.reason or ""),
                # THE START IT CLAIMS IS ALSO ITS ORDERING BIRTH (operator
                # convergence, 2026-10-05). The federated row carries no
                # conversation birth, and ranking every remote row at
                # ``created_at=0`` parked each one at the BOTTOM of its bin —
                # a soft form of the per-device segregation the merged bins
                # exist to remove. ``started`` is the only per-row time the
                # wire has, it is already this row's ``mtime`` (its age
                # column), and an unknown start — missing, or a claim that is
                # not a number (``_started_epoch``) — sorts last, the honest
                # direction. See ``resume.SessionRow.created_at``.
                created_at=_started_epoch(peer_row),
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


def _network_names(root: Path) -> dict[tuple[str, str], str]:
    """``(device_id, network_id)`` → the membership's network NAME.

    WHY THE READER RESOLVES IT. The federated row carries the network's ID
    only, and a 32-hex id is not a name a person reads — the tooltip's location
    clause reads "the device AND the network" (operator convergence,
    2026-10-05), so a name has to come from somewhere. This device's own
    membership records are the only store that can answer, and they are read
    through ``known_peers`` (``network/peers.py``), the module that already
    resolves "a peer device as a person sees it" for ``/new remote``'s
    autofill, rather than by re-deriving the record layout here. A device in
    two networks yields one entry per membership, which is why the key carries
    the network as well as the device.

    THE FAILURE IS AN EMPTY ANSWER, as everywhere in this module: an unreadable
    store contributes no names (the clause is then omitted), never a broken
    listing.
    """
    try:
        from local_operator.network.peers import known_peers

        return {
            (peer.device_id, peer.network_id): peer.network_name
            for peer in known_peers(root)
            if peer.network_name
        }
    except Exception:  # noqa: BLE001 — a listing never raises for a name it lacks
        return {}


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
