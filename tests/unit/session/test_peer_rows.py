"""The producer that gives the sidebar's ``⇄`` rows something to paint.

What these pin, in the order the design cares about:

1. **The zero-peer property is measured, not asserted.** A device with no relay
   record issues NO call at all — the spy counts them, because "no rows appeared"
   is satisfied by a producer that dials every two seconds and finds nothing.
2. **The row carries the mobility fields**, so the section heading, the ``⇄`` mark
   and the unreachable reason all have a producer rather than a fixture.
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
from local_operator.session import peer_rows as peer_rows_mod
from local_operator.session.peer_rows import (
    clear_cache,
    peer_session_row,
    peer_session_rows,
)


class _Facts:
    def __init__(self, device_id: str, name: str, *, reachable: bool, reason: str = "") -> None:
        self.device_id = device_id
        self.name = name
        self.reachable = reachable
        self.reason = reason


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
    # sidebar's tooltip and the section heading read.
    second = rows[1]
    assert second.reachable is False
    assert second.unreachable_reason == "connect_failed"
    assert second.owner_label == "d_bb"[:8], "an unnamed device falls back to its id's tail"


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
