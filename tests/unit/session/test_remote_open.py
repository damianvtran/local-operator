"""``remote_row_for``'s resolution contract: cache first, a MISS paid LIVE.

WHY THIS FILE EXISTS, and why the miss path is the whole story. A conversation
created ON A PEER must be resolvable by every local route the moment the create
answers — the peer mints the id, the create route returns it, and the message
the user sends to it next goes through exactly this function. The operator hit
the failure live (2026-10-05): ``/new remote <peer>`` returned, and ``GET
events``, ``command-entities`` and ``POST messages`` on that id all 404'd at
~50 ms after the create — each one refused with "This conversation no longer
exists, so your message wasn't sent." The id only resolved when the next
federated sidebar read landed, ~47 s later.

The cause was this function's miss path asking the CACHED listing
(``peer_session_rows``): the listing's TTL (``peer_rows._TTL_S``) exists to keep
the sidebar's two-second poll off the wire, and the answer it returns on a miss
is guaranteed stale BY CONSTRUCTION — the id is not in it, and it cannot have
become complete since it was read. So the miss now pays a genuine read
(``ttl_s=0``), and the caching that remains is only the part a resolution may
legitimately reuse: a row the listing HELD still answers without a dial.

The catalogue is injected the way the producer's own tests inject it (patching
``projection.RelayPeerCatalog``, the one construction ``peer_rows._read`` makes
for a root with a relay record), so every layer above the relay — the TTL
check, ``_read_all``, the miss path, the local-directory guard — is production
code. All three helper classes come from ``test_peer_rows``' fixtures by
import, the shared-fixture shape ``tests/unit/network/test_projection.py`` uses.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import pytest

from local_operator.network import projection, store
from local_operator.session import peer_rows as peer_rows_mod
from local_operator.session.peer_rows import peer_session_row
from local_operator.session.remote_open import remote_row_for
from tests.unit.session.test_peer_rows import _Catalog, _Facts, _Row


@pytest.fixture(autouse=True)
def _fresh_cache() -> Iterator[None]:
    """The cache is module-level and keyed by root: no test may inherit one."""
    peer_rows_mod.clear_cache()
    yield
    peer_rows_mod.clear_cache()


def _relay(monkeypatch: pytest.MonkeyPatch, catalog: _Catalog) -> _Catalog:
    """Put a relay record behind the root, and answer its reads from ``catalog``."""
    monkeypatch.setattr(store, "find_own_relay", lambda root=None: object())
    monkeypatch.setattr(projection, "RelayPeerCatalog", lambda root: catalog)
    return catalog


def test_a_miss_inside_the_listing_ttl_is_answered_live(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """THE OPERATOR'S DEFECT: an id minted after the last read resolved to nothing.

    The timeline, from the live trace: the sidebar's federated read filled the
    cache; the create minted ``s_new`` ON THE PEER; every resolution of it
    inside the TTL was answered from the listing read BEFORE the id existed.
    The clock is pinned, so "inside the TTL" is a fact of the test rather than
    a hope about how fast this machine is.
    """
    monkeypatch.setattr(peer_rows_mod, "time", SimpleNamespace(monotonic=lambda: 100.0))
    root = tmp_path / "root"
    _relay(
        monkeypatch,
        _Catalog([_Facts("d_aa", "radiant-m4", reachable=True)], [_Row("s_old", "d_aa")]),
    )
    assert [row.id for row in peer_rows_mod.peer_session_rows(root)] == [
        "s_old"
    ], "the listing must be cached to reproduce the window"

    # The create landed on the peer: a live read from this moment includes s_new.
    after = _relay(
        monkeypatch,
        _Catalog(
            [_Facts("d_aa", "radiant-m4", reachable=True)],
            [_Row("s_old", "d_aa"), _Row("s_new", "d_aa")],
        ),
    )

    row = remote_row_for("s_new", root)
    assert row is not None and row.id == "s_new", (
        "the id the create answered with resolved to nothing: the miss path reused a "
        "listing that cannot contain an id minted after it was read"
    )
    assert after.calls == 1, "the miss is paid with exactly ONE live read"
    assert (
        peer_session_row("s_new", root) is not None
    ), "and the live answer is merged back, so the next resolution needs no read"


def test_a_listed_row_resolves_without_a_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CACHE FIRST stays true: a row the listing holds must not pay anything.

    The fix widens the miss path, and nothing else: the ordinary resolution (a
    row the sidebar's read already holds) keeps its zero-wire answer, which is
    the property every ``/resume`` and archive/delete path relies on.
    """
    root = tmp_path / "root"
    catalog = _relay(
        monkeypatch,
        _Catalog([_Facts("d_aa", "radiant-m4", reachable=True)], [_Row("s_1", "d_aa")]),
    )
    assert [row.id for row in peer_rows_mod.peer_session_rows(root)] == ["s_1"]
    row = remote_row_for("s_1", root)
    assert row is not None and row.owner_device == "d_aa"
    assert catalog.calls == 1, "the cached answer must not trigger a second read"


def test_a_local_directory_keeps_the_peer_off_the_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A directory this device holds is not a peer question, and never dials."""
    root = tmp_path / "root"
    (root / "sessions" / "aaaaaaaaaaaa").mkdir(parents=True)
    catalog = _relay(
        monkeypatch,
        _Catalog([_Facts("d_aa", "radiant-m4", reachable=True)], [_Row("s_1", "d_aa")]),
    )
    assert remote_row_for("aaaaaaaaaaaa", root) is None
    assert catalog.calls == 0, "no catalogue may be consulted for an id this device holds"


def test_an_id_nobody_holds_is_none_at_the_cost_of_one_live_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """THE FIX'S COST, stated: a genuinely unknown id pays ONE live read.

    That is the read this path always documented ("ONE read only when this
    device holds no directory for the id"); what changed is that it can no
    longer be answered by a listing that provably cannot contain the id.
    """
    root = tmp_path / "root"
    _relay(
        monkeypatch,
        _Catalog([_Facts("d_aa", "radiant-m4", reachable=True)], [_Row("s_old", "d_aa")]),
    )
    assert [row.id for row in peer_rows_mod.peer_session_rows(root)] == ["s_old"]
    after = _relay(
        monkeypatch,
        _Catalog([_Facts("d_aa", "radiant-m4", reachable=True)], [_Row("s_old", "d_aa")]),
    )
    assert remote_row_for("s_missing", root) is None
    assert after.calls == 1, "one live read — and still the honest None"
