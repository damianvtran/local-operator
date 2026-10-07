"""The producer that gives the sidebar's ``↗`` rows something to paint.

What these pin, in the order the design cares about:

1. **The zero-peer property is measured, not asserted.** A device with no relay
   record issues NO call at all — the spy counts them, because "no rows appeared"
   is satisfied by a producer that dials every two seconds and finds nothing.
2. **The row carries the mobility fields**, so the ordinary bins, the ``↗``/``↛``
   locality marks, the tooltip's device · network clause and the unreachable reason all have a
   producer rather than a fixture — including the two the reader resolves
   (``owner_network_name`` from this device's own membership record, and
   ``created_at`` from the peer's ``started`` claim, which is what orders a
   remote row among local ones).
3. **The read is cached and bounded**, because the sidebar polls every two
   seconds and this is the only thing on that path that talks to the network.
4. **Nothing raises.** A refused, timed-out or unreadable projection is an empty
   tuple — never an exception on a paint path, and never a swallowed failure
   rendered as a fact.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from local_operator.network import store
from local_operator.resume import SessionRow
from local_operator.session import peer_rows as peer_rows_mod
from local_operator.session.peer_rows import (
    RemotePark,
    clear_cache,
    park_edges,
    peer_session_row,
    peer_session_rows,
    seed_peer_row,
    select_peer_session,
)


class _Facts:
    def __init__(
        self,
        device_id: str,
        name: str,
        *,
        reachable: bool,
        reason: str = "",
        network_id: str = "",
    ) -> None:
        self.device_id = device_id
        self.name = name
        self.reachable = reachable
        self.reason = reason
        self.network_id = network_id


class _Row:
    """One federated session row, as ``RelayPeerCatalog._load`` builds it."""

    def __init__(
        self,
        session_id: str,
        device_id: str,
        *,
        name: str = "Some conversation",
        state: str = "idle",
        busy: bool = False,
        detached: bool = False,
        pending: str | None = None,
        started: float = 1000.0,
    ) -> None:
        self.session_id = session_id
        self.device_id = device_id
        self.conversation_name = name
        self.state = state
        self.busy = busy
        self.detached = detached
        self.pending = pending
        self.started = started
        self.kind = "tui"


class _Catalog:
    """A ``PeerCatalog`` stand-in that counts the calls made to it."""

    def __init__(self, peers: list[_Facts], rows: list[_Row], *, boom: bool = False) -> None:
        self._peers = peers
        self._rows = rows
        self._boom = boom
        self.calls = 0

    def peers(self):  # noqa: ANN201
        self.calls += 1
        if self._boom:
            raise RuntimeError("the relay went away")
        return list(self._peers)

    def rows(self):  # noqa: ANN201
        if self._boom:
            raise RuntimeError("the relay went away")
        return list(self._rows)


class _Member:
    """One member row of a ``NetworkRecord``, as ``known_peers`` reads it."""

    def __init__(self, device_id: str, name: str = "", *, active: bool = True) -> None:
        self.device_id = device_id
        self.name = name
        self.role = "drive"
        self.kind = "device"
        self.active = active


class _Record:
    """One network record, as ``store.list_networks`` answers it."""

    def __init__(
        self, network_id: str, name: str, self_device_id: str, members: list[_Member]
    ) -> None:
        self.network_id = network_id
        self.name = name
        self.self_device_id = self_device_id
        self.members = members

    def active_members(self) -> list[_Member]:
        return [member for member in self.members if member.active]


@pytest.fixture(autouse=True)
def _fresh_cache() -> None:
    """Every test starts with an empty cache: a fixture's rows must not leak on."""
    clear_cache()


def _with_relay(monkeypatch: pytest.MonkeyPatch, present: bool = True) -> list[str]:
    """Stand in for ``store.find_own_relay``, recording that it was asked."""
    seen: list[str] = []

    def find(root: Path | None = None) -> object | None:
        seen.append("asked")
        return object() if present else None

    monkeypatch.setattr(store, "find_own_relay", find)
    return seen


def test_a_device_with_no_relay_issues_no_call_at_all(monkeypatch: pytest.MonkeyPatch) -> None:
    """The zero-peer property, measured on the call rather than on the result.

    Counted on the CATALOGUE's construction and use, because "no rows appeared"
    is equally satisfied by a producer that dials every two seconds and finds
    nothing — which is the failure this pins.
    """
    from local_operator.network import projection

    built: list[object] = []

    class _Spy:
        def __init__(self, root: object = None) -> None:
            built.append(self)
            raise AssertionError("a device in no mesh must not build a projection reader")

    monkeypatch.setattr(projection, "RelayPeerCatalog", _Spy)
    asked = _with_relay(monkeypatch, present=False)
    assert peer_session_rows() == ()
    assert asked == ["asked"], "the relay record must be looked for, not assumed"
    assert built == [], "no catalogue may be built when there is no relay record"


def test_a_nameless_peer_row_reads_like_every_other_nameless_row() -> None:
    """ONE SPELLING FOR ONE CONDITION (design round 2, D14).

    The peer-side fallback was the session's own 12-hex id while a nameless row
    THIS device holds paints ``Untitled conversation`` — so one missing name read
    two ways on one screen, and the id was on the row a reader can least resolve
    by looking around them (a remote row is the one row no local list can
    explain). Both halves now read the shared constant.
    """
    from local_operator.resume import UNTITLED_CONVERSATION

    catalog = _Catalog(
        [_Facts("d_aa", "radiant-m4", reachable=True)],
        [_Row("s_1", "d_aa", name="")],
    )
    rows = peer_session_rows(catalog=catalog)
    assert [row.name for row in rows] == [UNTITLED_CONVERSATION]


def test_rows_carry_every_field_the_surfaces_read() -> None:
    catalog = _Catalog(
        [
            _Facts("d_aa", "radiant-m4", reachable=True),
            _Facts("d_bb", "", reachable=False, reason="connect_failed"),
        ],
        [
            _Row("s_1", "d_aa", name="Mesh transport identity", state="busy"),
            _Row("s_2", "d_bb", name="Phone portal deploy", state="idle"),
        ],
    )
    rows = peer_session_rows(catalog=catalog)
    assert [row.id for row in rows] == ["s_1", "s_2"]
    first = rows[0]
    assert first.is_remote and first.locality == "remote"
    assert first.owner_device == "d_aa"
    assert first.owner_device_name == "radiant-m4"
    assert first.owner_label == "radiant-m4"
    assert first.live_state == "busy"
    assert first.reachable is True
    # The unreachable half: the reason travels WITH the row, which is what the
    # sidebar's tooltip reads.
    second = rows[1]
    assert second.reachable is False
    assert second.unreachable_reason == "connect_failed"
    assert second.owner_label == "d_bb"[:8], "an unnamed device falls back to its id's tail"


def test_a_remote_row_orders_by_the_peers_started_claim() -> None:
    """The row's ordering birth is the only per-row time the wire carries.

    ``session/catalog`` ranks by ``-created_at``; a remote row left at zero
    parked at the BOTTOM of its bin — the soft form of the per-device
    segregation the merged bins removed (operator convergence, 2026-10-05) —
    so the producer stamps ``started``. ``started`` is ALSO the row's ``mtime``
    (its age column), so the two cannot disagree about when the peer says this
    session began.
    """
    catalog = _Catalog(
        [_Facts("d_aa", "radiant-m4", reachable=True)],
        [_Row("s_1", "d_aa", started=4242.0)],
    )
    row = peer_session_rows(catalog=catalog)[0]
    assert row.created_at == 4242.0
    assert row.mtime == 4242.0


def test_the_membership_name_rides_the_row_for_the_tooltips_device_clause(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """DEVICE AND NETWORK on the hover: the reader resolves the network's NAME.

    The federated row carries the network's id only, and an id is not a name a
    person reads — the tooltip's clause is "the device AND the network", so the
    producer resolves it (``_network_names``) from THIS device's own membership
    record, through the same ``known_peers`` the ``/new remote`` autofill reads.
    A membership that cannot be read leaves the clause empty — no name is no
    claim, never a guessed one.
    """
    catalog = _Catalog(
        [_Facts("d_aa", "radiant-m4", reachable=True, network_id="n_1")],
        [_Row("s_1", "d_aa")],
    )
    monkeypatch.setattr(
        store,
        "list_networks",
        lambda root=None: [
            _Record("n_1", "devmesh", "d_self", [_Member(device_id="d_aa", name="radiant-m4")])
        ],
    )
    # A real (temporary) root: the resolution reads THIS device's own store, and
    # the guard exists so a root-less call — a test's injected-catalogue call —
    # never wanders into the developer's real one.
    row = peer_session_rows(root=tmp_path, catalog=catalog)[0]
    assert row.owner_network_name == "devmesh"
    # The same row through a store that knows no membership: the clause is
    # omitted rather than guessed. ``clear_cache`` because the read above is
    # TTL-cached under the same root.
    monkeypatch.setattr(store, "list_networks", lambda root=None: [])
    clear_cache()
    row = peer_session_rows(root=tmp_path, catalog=catalog)[0]
    assert row.owner_network_name == ""


def test_a_peer_answer_vocabulary_is_mapped_not_invented() -> None:
    """The peer's own ``state`` passes through when this list knows the word.

    Everything else falls to the transport's own booleans, through the SAME
    reading the local half uses (``session.catalog.live_state_from_flags``) —
    there is no second vocabulary here, invented or otherwise.
    """
    catalog = _Catalog(
        [_Facts("d_aa", "radiant-m4", reachable=True)],
        [
            _Row("s_1", "d_aa", state="wedged"),
            _Row("s_2", "d_aa", state="attached"),
            # A word this list does not know: the booleans decide, and
            # ``detached: False`` says a terminal is watching it.
            _Row("s_3", "d_aa", state="philosophising"),
            # No token at all, and no viewer: a runtime nobody is watching is
            # IDLE — the ``detached`` bit names the absence of a viewer, so it
            # can never be the ``attached`` token.
            _Row("s_4", "d_aa", state="", detached=True),
            _Row("s_5", "d_aa", state="", busy=True),
        ],
    )
    states = [row.live_state for row in peer_session_rows(catalog=catalog)]
    assert states == ["wedged", "attached", "attached", "idle", "busy"]


def test_a_stored_peer_row_is_never_painted_open() -> None:
    """``detached: True`` means NOBODY IS WATCHING, never "Open" (UX round 5).

    The relay's stored half (``RelayPeerCatalog._stored_rows``) stamps every row
    it mints ``state: "stored"``, ``detached: True`` — a session that exists on
    the peer with NO runtime behind it, which is the same condition the local
    half paints with an empty ``live_state``. Read backwards, that bit made the
    peer half claim every one of them was live: ``○`` in the sidebar and "Open"
    in the tooltip, on the same screen where a session THIS device holds with no
    runtime paints cold. The cell is asserted through the RENDERED pair rather
    than the token alone, because the token is what the glyph and the tooltip
    are computed from.
    """
    from local_operator.session.catalog import CatalogEntry
    from local_operator.tui.widgets.session_picker import (
        ATTACHED_MARKER,
        row_state_mark,
    )

    catalog = _Catalog(
        [_Facts("d_aa", "radiant-m4", reachable=True)],
        [_Row("s_1", "d_aa", name="Moved here last week", state="stored", detached=True)],
    )
    (row,) = peer_session_rows(catalog=catalog)
    assert row.live_state == "", "a session with no runtime is a cold row"
    assert CatalogEntry(row).status != "Open"
    glyph, _ink = row_state_mark(row, 0)
    assert glyph != ATTACHED_MARKER, "the peer group must not claim a viewer it does not have"
    assert glyph == "", "a cold row draws no state mark"


def test_a_live_peer_row_nobody_watches_is_idle_not_cold() -> None:
    """The other half of the same bit: a RUNNING unwatched session is ``idle``.

    ``state: "live"`` is the registry's word for a session whose pid is alive and
    heartbeating (``runtime/registry.py``), so the row must keep the mark and the
    sentence the local half gives the same session — a runtime with nothing to
    report — rather than falling through to the cold/no-claim case ``stored``
    gets.
    """
    from local_operator.session.catalog import CatalogEntry
    from local_operator.tui.widgets.session_picker import IDLE_MARKER, row_state_mark

    catalog = _Catalog(
        [_Facts("d_aa", "radiant-m4", reachable=True)],
        [_Row("s_1", "d_aa", name="Review the mesh brief", state="live", detached=True)],
    )
    (row,) = peer_session_rows(catalog=catalog)
    assert row.live_state == "idle"
    assert CatalogEntry(row).status == "Ready"
    assert row_state_mark(row, 0) == (IDLE_MARKER, "muted")


def test_a_row_from_a_device_the_peers_answer_lacks_is_dropped() -> None:
    """The two halves of one projection disagreeing cannot buy a guessed heading."""
    catalog = _Catalog(
        [_Facts("d_aa", "radiant-m4", reachable=True)],
        [_Row("s_1", "d_aa"), _Row("s_2", "d_ghost")],
    )
    assert [row.id for row in peer_session_rows(catalog=catalog)] == ["s_1"]


def test_the_read_is_cached_for_its_ttl(monkeypatch: pytest.MonkeyPatch) -> None:
    _with_relay(monkeypatch)
    catalog = _Catalog([_Facts("d_aa", "radiant-m4", reachable=True)], [_Row("s_1", "d_aa")])
    first = peer_session_rows(catalog=catalog, now=0.0)
    again = peer_session_rows(catalog=catalog, now=peer_rows_mod._TTL_S / 2)
    assert first == again
    assert catalog.calls == 1, "a second poll inside the TTL must not re-ask the relay"
    # Past the TTL it asks again — a session created on the peer has to surface
    # while the user is still looking at the list it belongs in.
    peer_session_rows(catalog=catalog, now=peer_rows_mod._TTL_S * 2)
    assert catalog.calls == 2


def test_a_refused_projection_is_an_empty_tuple_not_an_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _with_relay(monkeypatch)
    assert peer_session_rows(catalog=_Catalog([], [], boom=True)) == ()


def test_an_empty_peer_answer_is_an_empty_tuple(monkeypatch: pytest.MonkeyPatch) -> None:
    _with_relay(monkeypatch)
    assert peer_session_rows(catalog=_Catalog([], [_Row("s_1", "d_aa")])) == ()


def test_the_lookup_is_cache_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """The resume guard must never dial: a cold cache is a miss, not a read."""
    _with_relay(monkeypatch)
    catalog = _Catalog([_Facts("d_aa", "radiant-m4", reachable=True)], [_Row("s_1", "d_aa")])
    assert peer_session_row("s_1") is None, "nothing has been read yet"
    assert catalog.calls == 0

    peer_session_rows(catalog=catalog)
    found = peer_session_row("s_1")
    assert found is not None and found.owner_device == "d_aa"
    assert catalog.calls == 1, "the lookup itself must not add a read"
    assert peer_session_row("s_missing") is None


def test_an_isolated_root_does_not_read_another_installs_answer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cache is keyed by config root, so a test cannot inherit a real answer."""
    _with_relay(monkeypatch)
    catalog = _Catalog([_Facts("d_aa", "radiant-m4", reachable=True)], [_Row("s_1", "d_aa")])
    peer_session_rows(tmp_path / "one", catalog=catalog)
    other = _Catalog([_Facts("d_bb", "pixel-8", reachable=True)], [_Row("s_9", "d_bb")])
    rows = peer_session_rows(tmp_path / "two", catalog=other)
    assert [row.id for row in rows] == ["s_9"]


# ---------------------------------------------------------------------------
# UX round 3, U16 — the peer that did not answer, which only ever produced rows
# ---------------------------------------------------------------------------


def _silent_peer_catalog() -> _Catalog:
    """One peer that did not answer, and one that did with a single row."""
    return _Catalog(
        [
            _Facts(
                "d_silent",
                "pixel-8",
                reachable=False,
                reason="connect_failed:ConnectionRefusedError",
            ),
            _Facts("d_live", "radiant-m4", reachable=True),
        ],
        [_Row("row-1", "d_live")],
    )


def test_a_peer_that_did_not_answer_is_reported_though_it_has_no_rows() -> None:
    """THE FACT §8.3 WAS DROPPING. A peer that does not answer contributes no
    rows — right, and it used to contribute nothing else either, because this
    module only ever returned rows: the sidebar's section was built from them, so
    the whole tier vanished and the list read as complete. The relay reports the
    device with ``reachable: false`` and its reason in the SAME answer; this is
    the half that carries it to a surface."""
    catalog = _silent_peer_catalog()
    assert [row.id for row in peer_session_rows(Path("/tmp/x"), catalog=catalog)] == ["row-1"]
    unanswered = peer_rows_mod.unanswered_peers(Path("/tmp/x"), catalog=catalog)
    assert [(peer.device_id, peer.name, peer.reason) for peer in unanswered] == [
        ("d_silent", "pixel-8", "connect_failed:ConnectionRefusedError")
    ]


def test_the_rows_and_the_silent_peers_come_from_one_read() -> None:
    """They are one relay answer, so a caller that asks for both must not be able
    to see two different fan-outs of a mesh that is moving underneath it."""
    catalog = _silent_peer_catalog()
    peer_session_rows(Path("/tmp/x"), catalog=catalog)
    peer_rows_mod.unanswered_peers(Path("/tmp/x"), catalog=catalog)
    assert catalog.calls == 1, "the second call re-read the catalogue"


def test_a_peer_that_answered_contributes_nothing_and_a_row_excludes_its_device() -> None:
    """Two ways to be wrong here, both pinned: reporting a peer that simply has no
    sessions as "gone" (it contributed no rows, but it answered), and reporting a
    device twice — once as a section with rows, once as a silent heading.

    Each case gets its OWN root: the cache is keyed by root for its TTL, so a
    second read under the same root would answer the first one's question.
    """
    only_live = _Catalog([_Facts("d_live", "radiant-m4", reachable=True)], [])
    assert peer_rows_mod.unanswered_peers(Path("/tmp/answered"), catalog=only_live) == ()

    # A device the peers answer marks unreachable but WHICH ALSO produced a row:
    # the row is the section, and the heading already carries `(unreachable)`.
    both = _Catalog(
        [_Facts("d_mixed", "pixel-8", reachable=False, reason="asked, and it did not answer")],
        [_Row("row-1", "d_mixed")],
    )
    assert len(peer_session_rows(Path("/tmp/mixed"), catalog=both)) == 1
    assert peer_rows_mod.unanswered_peers(Path("/tmp/mixed"), catalog=both) == ()


def test_a_cold_cache_is_not_an_answer_about_silent_peers() -> None:
    """``peer_session_row`` is deliberately cache-only (it fronts every
    ``/resume``), but a selector that answered "no peer is silent" merely because
    nobody had polled yet would be the silent-empty failure U16 is about. A cold
    call reads."""
    catalog = _silent_peer_catalog()
    assert peer_rows_mod.unanswered_peers(Path("/tmp/x"), catalog=catalog)
    assert catalog.calls == 1


def test_a_device_in_two_networks_contributes_one_row_per_session() -> None:
    """QA round 1, Q1: a session belongs to a DEVICE, so a row cannot be per-network.

    The relay's fan-out walks this device's networks and each network's members, so a
    device that shares two networks with this one used to answer once per network and
    have its rows appended twice: measured with three real paired devices on loopback,
    two conversations on such a device arrived as FOUR rows — in the listing, in the
    search, and in ``peer_catalogue``'s count — while the sidebar (which groups its own
    rows by id) said two. The duplication is fed in directly here because the property
    is this producer's: whatever the relay hands over, one session is one row.
    """
    catalog = _Catalog(
        [_Facts("d_aa", "radiant-m4", reachable=True)],
        [
            _Row("s_1", "d_aa", name="Shared conversation"),
            _Row("s_2", "d_aa", name="Another one"),
            # ...and the same device's answer again, over its SECOND shared network.
            _Row("s_1", "d_aa", name="Shared conversation"),
            _Row("s_2", "d_aa", name="Another one"),
        ],
    )
    rows = peer_session_rows(catalog=catalog)
    assert [row.id for row in rows] == ["s_1", "s_2"]
    assert [row.name for row in rows] == ["Shared conversation", "Another one"]

    # THE CACHE ANSWERS THE SAME WAY, because it stores this producer's own tuple
    # rather than the relay's raw rows — a second read that re-expanded them would put
    # the duplication back on the sidebar's two-second poll.
    clear_cache()
    assert [row.id for row in peer_session_rows(catalog=catalog)] == ["s_1", "s_2"]
    reads = catalog.calls
    assert [row.id for row in peer_session_rows(catalog=catalog)] == ["s_1", "s_2"]
    assert catalog.calls == reads, "the second read is the cache's, not a new fan-out"


def test_two_devices_reporting_one_id_stay_two_rows() -> None:
    """Review round 1, MINOR 2: the de-duplication is keyed by DEVICE, not by id.

    Keying on the id alone made the collapse global across devices, so a genuine
    cross-device collision — the one routing ambiguity the mesh exists to prevent
    (``relay._op_session_create`` mints on the destination for exactly this reason) —
    dropped a row silently: measured as ``[('same-id', 'd_bbb\\u2026')]``, with the second
    device's row gone and the survivor filed under the first device's heading, so two
    conversations read as one. Everything this producer is FOR happens INSIDE one
    device (its own answer repeated once per shared network), so the device in the key
    keeps that fix and leaves the ambiguity visible as the two rows it is.
    """
    catalog = _Catalog(
        [
            _Facts("d_aa", "radiant-m4", reachable=True),
            _Facts("d_bbb", "build-box", reachable=True),
        ],
        [
            _Row("same-id", "d_aa", name="This device's copy"),
            _Row("same-id", "d_bbb", name="The other device's copy"),
        ],
    )
    rows = peer_session_rows(catalog=catalog)
    assert [(row.id, row.owner_device) for row in rows] == [
        ("same-id", "d_aa"),
        ("same-id", "d_bbb"),
    ], "a cross-device id collision is a routing ambiguity, not a row to drop"

    # ...AND THE CASE THIS FIX IS FOR STILL COLLAPSES, asserted in the same test so a
    # later relaxation of the key cannot trade one of the two behaviours for the other.
    clear_cache()
    repeated = _Catalog(
        [_Facts("d_aa", "radiant-m4", reachable=True)],
        [_Row("s_1", "d_aa"), _Row("s_1", "d_aa")],
    )
    assert [row.id for row in peer_session_rows(catalog=repeated)] == ["s_1"]


# ---------------------------------------------------------------------------
# Park edges: the episode model the origin's notice rides.
#
# These drive the PURE helper with constructed rows, plus two cells through the
# real producer so the stored-half discriminator is pinned against the shape
# ``RelayPeerCatalog`` actually mints rather than against a hand-guessed one.


def _parked_row(
    session_id: str,
    device_id: str = "d_aa",
    *,
    name: str = "Some conversation",
    kind: str | None = "approval",
    device_name: str = "radiant-m4",
    live_state: str = "busy",
) -> SessionRow:
    """One row for the pure matrix: only the fields ``park_edges`` reads are set."""
    return SessionRow(
        id=session_id,
        mtime=0.0,
        name=name,
        pending=kind,
        live_state=live_state,
        locality="remote",
        owner_device=device_id,
        owner_device_name=device_name,
    )


def test_the_park_edge_matrix_appear_clear_repark_and_kind_change() -> None:
    """One notice per EPISODE, and the four transitions that define one.

    Appear (fires), the same park polled again (silent), the park clearing
    (already answered — the map drops the key, and the caller withdraws the
    card that claimed otherwise), and a re-park (a NEW episode, because a
    second park is a second thing a person must do).
    """
    parked = _parked_row("s_1")
    edges, state = park_edges({}, [parked])
    assert [(edge.session_id, edge.kind) for edge in edges] == [("s_1", "approval")]

    edges, state = park_edges(state, [parked])
    assert edges == (), "the same park polled again is the same episode"

    idle = _parked_row("s_1", kind=None, live_state="idle")
    edges, cleared = park_edges(state, [idle])
    assert edges == () and cleared == {}, "an answered park leaves the map"

    edges, _ = park_edges(cleared, [parked])
    assert [edge.session_id for edge in edges] == ["s_1"], "a re-park re-arms"


def test_a_silent_device_carries_its_park_instead_of_clearing_it() -> None:
    """Only a device that ANSWERED can clear its own park (review round 1, MINOR-1).

    ``unanswered_peers`` names the devices the relay reported as unreachable, and
    a refused or timed-out read returns no rows for them. Reading that as "the
    park cleared" withdrew the card for a park that may still be live and then
    re-announced it as a SECOND episode when the peer recovered; the keys of a
    silent device are therefore carried, and the first answered read that does
    not carry them performs the real clear.
    """
    parked = _parked_row("s_1")
    edges, state = park_edges({}, [parked])
    assert [edge.session_id for edge in edges] == ["s_1"]

    # The peer goes silent: no rows for it, and the read says so by name.
    edges, silent = park_edges(state, [], unanswered=["d_aa"])
    assert edges == ()
    assert silent == state, "a silent read must neither clear the park nor re-fire it"

    # Silence for a DIFFERENT device carries nothing: that device's absence is
    # an ordinary clear, or one dead peer would freeze every other peer's map.
    edges, other = park_edges(state, [], unanswered=["d_zzz"])
    assert edges == () and other == {}, "only the silent device's keys are carried"

    # The peer answers again with the park gone: THAT is the clear.
    edges, cleared = park_edges(silent, [], unanswered=())
    assert edges == () and cleared == {}

    # And an answered read that carries the SAME park is the same episode, not
    # a re-announcement: the carried key compares equal and stays quiet.
    edges, _ = park_edges(silent, [parked], unanswered=())
    assert edges == (), "a carried episode re-announced itself on recovery"


def test_a_kind_change_is_a_new_episode() -> None:
    """ask -> approval is a new remedy: the approval needs a presence gesture.

    Polling the OLD kind's map against a row whose kind moved must fire again —
    the user was told to answer, and the session now needs something different
    from them.
    """
    edges, state = park_edges({}, [_parked_row("s_1", kind="ask")])
    assert [edge.kind for edge in edges] == ["ask"]

    edges, _ = park_edges(state, [_parked_row("s_1", kind="approval")])
    assert [edge.kind for edge in edges] == ["approval"]


def test_a_park_edge_carries_the_device_and_the_conversation_name() -> None:
    """Every surface composes from this tuple: device label, kind, title.

    ``device_name`` and ``name`` are what the banner's title and body and the
    toast's copy are built from, so the row's identity must travel whole rather
    than as an id the caller would have to look up again.
    """
    row = _parked_row("s_1", name="Rename the deploy job", device_name="radiant-m4")
    edges, _ = park_edges({}, [row])
    assert edges == (
        RemotePark(
            session_id="s_1",
            device_id="d_aa",
            device_name="radiant-m4",
            kind="approval",
            name="Rename the deploy job",
        ),
    )


def test_two_devices_parking_one_id_are_two_episodes() -> None:
    """The key carries the DEVICE, same as the producer's de-duplication.

    A map keyed on the session id alone lets the first device's park swallow
    the second's notice — the cross-device collapse ``_read`` refuses to make,
    reintroduced one layer up where it would be silent.
    """
    rows = [
        _parked_row("same-id", "d_aa", name="This device's copy"),
        _parked_row("same-id", "d_bbb", name="The other device's copy"),
    ]
    edges, state = park_edges({}, rows)
    assert [(edge.device_id, edge.session_id) for edge in edges] == [
        ("d_aa", "same-id"),
        ("d_bbb", "same-id"),
    ]
    edges, _ = park_edges(state, rows)
    assert edges == (), "each device's episode is its own"


def test_a_stored_unread_completion_is_not_a_park() -> None:
    """The §2 discriminator, pinned against the shape a legacy peer still ships.

    A producer that predates the 2026-10-07 correction minted ``state:
    "stored"``, ``detached: True`` for a session with no runtime AND translated
    an unread COMPLETION into ``pending: "ask"``; the row-level reader drops
    that claim now (``network.types.row_needs_claim``), and this fixture seeds
    it directly so ``park_edges``'s own discriminator stays covered for any row
    that already carries the legacy shape — a cached row ingested before the
    fix, or a peer whose build is older than this test. ``_live_state`` maps
    ``stored`` to the empty string, so the row is refused THROUGH the real
    helper — a completion that already finished needs nobody, and announcing it
    as a park would page a person for an answered turn.
    """
    catalog = _Catalog(
        [_Facts("d_aa", "radiant-m4", reachable=True)],
        [_Row("s_stored", "d_aa", state="stored", detached=True, pending="ask")],
    )
    (row,) = peer_session_rows(catalog=catalog)
    assert row.live_state == "", "the stored half is a cold row"
    assert row.pending == "ask", "...the seeded legacy claim the discriminator must still refuse"
    edges, state = park_edges({}, [row])
    assert edges == () and state == {}


def test_a_live_parked_row_is_a_park_through_the_producer() -> None:
    """The other half of the same fixture pair: a live parked turn DOES fire.

    ``state: "busy"`` plus the record's ``pending`` is the shape the two
    measured approval pilots carried; it must survive the same read that
    refuses the stored row, or the discriminator has traded one half for the
    other.
    """
    catalog = _Catalog(
        [_Facts("d_aa", "radiant-m4", reachable=True)],
        [
            _Row(
                "s_live",
                "d_aa",
                name="Backfill the audit log",
                state="busy",
                busy=True,
                pending="approval",
            )
        ],
    )
    (row,) = peer_session_rows(catalog=catalog)
    assert row.live_state == "busy"
    edges, _ = park_edges({}, [row])
    assert [(edge.session_id, edge.kind, edge.device_name) for edge in edges] == [
        ("s_live", "approval", "radiant-m4")
    ]


# ---------------------------------------------------------------------------
# The create-on-peer seed: an id that was just minted resolves NOW
# ---------------------------------------------------------------------------


def _seeded_row(
    session_id: str, device_id: str = "d_aa", *, name: str = "Untitled conversation"
) -> SessionRow:
    """The row the create route builds from a peer's create reply."""
    return SessionRow(
        session_id,
        1234.0,
        name,
        locality="remote",
        owner_device=device_id,
        owner_device_name="radiant-m4",
        created_at=1234.0,
    )


def test_a_seed_merges_into_the_listing_that_exists(monkeypatch: pytest.MonkeyPatch) -> None:
    """The next resolution finds the created id with no read — the defect's fix.

    The operator's timeline: the sidebar's federated read filled the cache, the
    create minted ``s_new`` on the peer, and nothing seeded it, so every
    resolution inside the TTL missed. The seed is what the create route does
    now, and the cache-only lookup — the one every route resolves through —
    must find it immediately; the sidebar's own listing carries it too.
    """
    _with_relay(monkeypatch)
    catalog = _Catalog([_Facts("d_aa", "radiant-m4", reachable=True)], [_Row("s_old", "d_aa")])
    peer_session_rows(catalog=catalog, now=0.0)
    seed_peer_row(None, _seeded_row("s_new"))
    assert catalog.calls == 1, "a seed is knowledge, not a read: it never dials"
    found = peer_session_row("s_new")
    assert found is not None and found.owner_device == "d_aa"
    # ...and the same one answer the sidebar paints: merged, not a second listing.
    assert [row.id for row in peer_session_rows(catalog=catalog, now=1.0)] == ["s_old", "s_new"]


def test_a_seed_with_no_listing_yet_never_becomes_a_listing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A one-row entry must not paint "nothing else exists" for a TTL.

    With no listing cached, the seed's entry is stamped ALREADY-STALE: the
    cache-only lookup answers from it, and the first LISTING read still pays its
    read in full — the mesh's answer is the mesh's, never a seed's.
    """
    _with_relay(monkeypatch)
    seed_peer_row(None, _seeded_row("s_new"))
    assert peer_session_row("s_new") is not None, "the resolution the seed exists for"
    catalog = _Catalog(
        [_Facts("d_aa", "radiant-m4", reachable=True)],
        [_Row("s_old", "d_aa"), _Row("s_new", "d_aa")],
    )
    assert [row.id for row in peer_session_rows(catalog=catalog)] == ["s_old", "s_new"]
    assert catalog.calls == 1, "a TTL-respecting read must still read"


def test_a_seed_keeps_the_entrys_age_and_refresh_schedule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Merging a row is not a reason to extend every other row's staleness."""
    _with_relay(monkeypatch)
    catalog = _Catalog([_Facts("d_aa", "radiant-m4", reachable=True)], [_Row("s_old", "d_aa")])
    peer_session_rows(catalog=catalog, now=0.0)
    seed_peer_row(None, _seeded_row("s_new"))
    # Inside the TTL the seeded row rides the existing entry...
    rows = peer_session_rows(catalog=catalog, now=peer_rows_mod._TTL_S / 2)
    assert [row.id for row in rows] == ["s_old", "s_new"]
    assert catalog.calls == 1, "the seed must not have reset the entry's clock"
    # ...and past it the schedule is exactly what it was: one full read.
    peer_session_rows(catalog=catalog, now=peer_rows_mod._TTL_S * 2)
    assert catalog.calls == 2


def test_a_reseed_replaces_one_row_and_never_duplicates_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same device, same id is one conversation — ``_read``'s key, on the write side."""
    _with_relay(monkeypatch)
    catalog = _Catalog([_Facts("d_aa", "radiant-m4", reachable=True)], [_Row("s_1", "d_aa")])
    peer_session_rows(catalog=catalog, now=0.0)
    seed_peer_row(None, _seeded_row("s_new", name="first"))
    seed_peer_row(None, _seeded_row("s_new", name="second"))
    rows = peer_session_rows(catalog=catalog, now=1.0)
    assert [(row.id, row.name) for row in rows] == [
        ("s_1", "Some conversation"),
        ("s_new", "second"),
    ]


def test_a_seed_is_keyed_by_root_and_device(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The cache's two keys hold: the config root, and (device, id) within it."""
    _with_relay(monkeypatch)
    one, two = tmp_path / "one", tmp_path / "two"
    catalog = _Catalog([_Facts("d_aa", "radiant-m4", reachable=True)], [_Row("s_1", "d_aa")])
    peer_session_rows(one, catalog=catalog, now=0.0)
    seed_peer_row(one, _seeded_row("s_new"))
    assert peer_session_row("s_new", one) is not None
    assert peer_session_row("s_new", two) is None, "another root must not read this seed"
    # A same-id row from ANOTHER device is a different conversation, kept separate.
    seed_peer_row(one, _seeded_row("s_new", "d_bb"))
    rows = peer_session_rows(one, catalog=catalog, now=1.0)
    assert [(row.id, row.owner_device) for row in rows] == [
        ("s_1", "d_aa"),
        ("s_new", "d_aa"),
        ("s_new", "d_bb"),
    ]


# ---------------------------------------------------------------------------
# The target selector: which session did "that name" mean.
#
# The pure half of the pilot family's id-or-name resolution
# (``sessions-remote-tools.md`` §3B): no mesh, no I/O — the rows are fetched by
# the caller and the decision is made here, so every tier is asserted against
# constructed rows rather than against a relay that would have to exist first.


def _selectable_row(
    session_id: str,
    *,
    name: str = "Some conversation",
    device_id: str = "d_aa",
    device_name: str = "radiant-m4",
) -> SessionRow:
    """One row for the selector matrix: only its two address fields matter."""
    return SessionRow(
        id=session_id,
        mtime=0.0,
        name=name,
        locality="remote",
        owner_device=device_id,
        owner_device_name=device_name,
    )


def test_the_selector_resolves_an_exact_name_or_id_in_either_case() -> None:
    """An address the picker prints resolves silently — name first, then id.

    The ordering of the tiers is what the callers' refusals are built on: an
    exact value is evidence, so it never reaches the ambiguity list, and a
    caller that typed 'PILOT' gets the same session as one that typed 'pilot'
    (a conversation name is typed by a person, not by a machine).
    """
    by_name = _selectable_row("9e7d35e4e41f", name="pilot")
    other = _selectable_row("aaaa11112222", name="unrelated")
    assert select_peer_session("pilot", [by_name, other]) == (by_name, ())
    assert select_peer_session("PILOT", [by_name, other]) == (by_name, ())
    assert select_peer_session("9e7d35e4e41f", [by_name, other]) == (by_name, ())


def test_an_exact_match_wins_over_a_substring_namesake() -> None:
    """The wrong-recipient defect the exact tier exists for (§3B).

    ``pilot`` also appears inside ``Pilot notes``, and the substring tier alone
    would have refused the pair (or, when only the namesake contained it,
    silently delivered to the wrong session). The whole-value match must
    resolve to the row that IS ``pilot``.
    """
    exact = _selectable_row("9e7d35e4e41f", name="pilot")
    namesake = _selectable_row("aaaa11112222", name="Pilot notes")
    assert select_peer_session("pilot", [exact, namesake]) == (exact, ())


def test_two_exact_matches_are_candidates_ordered_by_the_field_that_matched() -> None:
    """Two whole-value matches is the one exact-tier state that refuses.

    Picking either would be the silent wrong-recipient class; the candidates
    are ordered name-match first, then id-match, and then by input order, so a
    refusal listing them reads the same way twice.
    """
    by_name = _selectable_row("aaaa11112222", name="9e7d35e4e41f")
    by_id = _selectable_row("9e7d35e4e41f", name="something else")
    # ``9e7d35e4e41f`` is one row's NAME and another row's ID: two exact
    # matches, and neither may be picked.
    assert select_peer_session("9e7d35e4e41f", [by_id, by_name]) == (
        None,
        (by_name, by_id),
    )


def test_a_substring_resolves_alone_and_lists_every_candidate_when_not() -> None:
    """One substring hit resolves; two are refused WITH both rows.

    The candidate list is the refusal's payload — a person (or a script) picks
    a new spelling from it rather than guessing at the tail of a name they
    half-remember.
    """
    alpha = _selectable_row("aaaa11112222", name="release checklist")
    beta = _selectable_row("bbbb33334444", name="checklist for the audit")
    assert select_peer_session("release", [alpha, beta]) == (alpha, ())
    assert select_peer_session("checklist", [alpha, beta]) == (None, (alpha, beta))


def test_an_unknown_target_is_no_match_and_no_candidates() -> None:
    """The third state the CLI needs apart: nothing matched AT ALL.

    ``(None, ())`` is what the caller renders as the family's own
    ``session_unknown`` (which distinguishes "the peer answered and holds no
    such conversation" from "the peer did not answer") — so an empty target and
    a miss must both land here without raising, and an empty candidate tuple is
    the signal that nothing was refused because nothing was ambiguous.
    """
    row = _selectable_row("aaaa11112222", name="unrelated")
    assert select_peer_session("nothing-like-it", [row]) == (None, ())
    assert select_peer_session("", [row]) == (None, ())
    assert select_peer_session("unrelated", []) == (None, ())


def test_a_cwd_basename_is_not_an_address_because_a_peer_row_carries_no_cwd() -> None:
    """The deliberate third-arm omission, pinned so a future wire change sees it.

    The local resolver also matches a row's working-directory basename; the row
    shape this selector reads (``SessionRow``) has no cwd field — the peer
    projection drops it — so there is nothing to match a worktree token
    against. If a cwd ever rides the row, this cell fails and the arm (plus its
    sole-match rule, ``peer_send._EXACT_WEAK_RANK``) belongs beside it; until
    then, a basename spelling simply is not an address.
    """
    assert "cwd" not in SessionRow._fields
    row = _selectable_row("aaaa11112222", name="unrelated")
    assert select_peer_session("sessions-remote-0b39", [row]) == (None, ())
