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
    """The peer's own ``state`` passes through when this list knows the word."""
    catalog = _Catalog(
        [_Facts("d_aa", "radiant-m4", reachable=True)],
        [
            _Row("s_1", "d_aa", state="wedged"),
            _Row("s_2", "d_aa", state="attached"),
            # An unrecognised token is a COLD row, not a new state this list
            # invented for a word it does not know.
            _Row("s_3", "d_aa", state="philosophising"),
            # No token at all: the transport's own booleans decide.
            _Row("s_4", "d_aa", state="", detached=True),
            _Row("s_5", "d_aa", state="", busy=True),
        ],
    )
    states = [row.live_state for row in peer_session_rows(catalog=catalog)]
    assert states == ["wedged", "attached", "", "attached", "busy"]


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
