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
from typing import Any

import pytest

from local_operator.network import projection, store
from local_operator.resume import UNNAMED_DEVICE
from local_operator.session import peer_rows as peer_rows_mod
from local_operator.session.peer_rows import UnansweredPeer, peer_session_row
from local_operator.session.remote_open import (
    PeerSessionUnresolved,
    open_remote_viewer,
    remote_row_and_silence,
    remote_row_for,
    unreachable_peer_sentence,
    unresolved_peer_sentence,
)
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


# ---------------------------------------------------------------------------
# `open_remote_viewer`'s miss: None means "not a peer's", never "nobody said"
# ---------------------------------------------------------------------------


async def _never_take_over() -> Any:
    """A remote viewer never takes over: the owner is the peer, not this device."""
    raise RuntimeError("a remote viewer never takes over a session")


class _JumpingClock:
    """A monotonic clock that jumps PAST the listing TTL on every reading.

    THE INSTRUMENT R-1 NEEDED. The "no second dial" property used to be an
    argument about durations: the second call's freshness test is "(now − the
    moment the first read STARTED) < TTL", and a listing that spends its own
    documented budget (``relay.LISTING_CLIENT_TIMEOUT_S`` == ``_TTL_S`` == 20 s)
    sits exactly on that edge. With a real clock the two-call shape passes this
    file's cells only because the injected catalogue answers instantly, so the
    cells could not see the defect they were asserting away. Jumping the clock
    100 s per reading makes the two-call shape re-dial DETERMINISTICALLY — no
    sleeping, no flake — while the one-read shape is a single call and cannot
    care what the clock says.

    Patched onto the ``peer_rows`` MODULE (its only use of ``time`` is the
    listing moment), so the global ``time`` module is never touched.
    """

    def __init__(self) -> None:
        self.readings = 0

    def monotonic(self) -> float:
        self.readings += 1
        return 1_000.0 + 100.0 * self.readings


@pytest.mark.asyncio
async def test_a_silent_device_refuses_the_viewer_instead_of_answering_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """THE ANSWER EVERY CALLER READS AS "NOT A PEER'S".

    ``None`` here sends the CLI's ``--resume`` to the LOCAL cold viewer below it
    — a viewer for a conversation this device does not hold, whose first write
    would engage a runtime HERE under somebody else's id (the two-writer case
    INV-1 forbids). When the read that missed also reports a device that did not
    answer, the honest answer is a refusal that says so, not that ``None``.
    """
    root = tmp_path / "root"
    catalog = _relay(
        monkeypatch,
        _Catalog(
            [
                _Facts(
                    "d_silent",
                    "build-box",
                    reachable=False,
                    reason="connect_failed:ConnectionRefusedError",
                ),
                _Facts("d_live", "radiant-m4", reachable=True),
            ],
            [],
        ),
    )

    with pytest.raises(PeerSessionUnresolved) as raised:
        await open_remote_viewer("s_missing", config_dir=root, takeover=_never_take_over)

    assert raised.value.session_id == "s_missing"
    assert raised.value.code == "session_unresolved"
    assert [peer.name for peer in raised.value.unanswered] == ["build-box"]
    assert str(raised.value) == unresolved_peer_sentence(
        "s_missing", (UnansweredPeer("d_silent", "build-box", ""),)
    )
    # R-7: assert the OWNERSHIP SHAPE, not a string that merely happens to be
    # absent. The positive form is what the composer must produce, so a leak
    # phrased some other way ("the conversation lives on build-box") fails here.
    assert "may be on that device" in str(raised.value)
    assert "did not answer" in str(raised.value)
    assert " is on " not in str(raised.value), "silence was turned into an ownership claim"
    assert "no longer" not in str(raised.value), "an unprovable absence was stated as a deletion"
    assert catalog.calls == 1, "the refusal must ride the read that already missed"


@pytest.mark.asyncio
async def test_a_miss_whose_own_read_outlasts_the_ttl_still_issues_no_second_dial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R-1: ONE READ, BOTH HALVES — held by construction, not by duration.

    The rows and the silence come from a single ``read_listing(ttl_s=0)``, so a
    listing that spends its whole documented budget (the black-hole member this
    state exists for) cannot make the consult re-dial — and cannot make it drop
    the rows that read fetched, which is the worse half: an id the listing HELD
    would otherwise answer "not a peer's".
    """
    root = tmp_path / "root"
    monkeypatch.setattr(peer_rows_mod, "time", _JumpingClock())
    catalog = _relay(
        monkeypatch,
        _Catalog(
            [
                _Facts(
                    "d_silent",
                    "build-box",
                    reachable=False,
                    reason="probe_timeout",
                ),
                _Facts("d_live", "radiant-m4", reachable=True),
            ],
            [],
        ),
    )

    with pytest.raises(PeerSessionUnresolved):
        await open_remote_viewer("s_missing", config_dir=root, takeover=_never_take_over)

    assert catalog.calls == 1, "the consult issued a SECOND fan-out"


@pytest.mark.asyncio
async def test_a_row_the_read_holds_is_never_lost_to_the_silence_consult(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R-1's second half: the re-dial used to DISCARD the rows it had read.

    A healed listing between two reads answered ``None`` for an id the second
    read contained — on the pool path, the shared 404 and the renderer's
    deleted-conversation claim about a conversation the read had in hand. One
    call cannot produce that state: the row and the silence are halves of one
    answer, so a row present means the seam resolves.
    """
    root = tmp_path / "root"
    monkeypatch.setattr(peer_rows_mod, "time", _JumpingClock())
    _relay(
        monkeypatch,
        _Catalog(
            [_Facts("d_peer", "build-box", reachable=True)],
            [_Row("s_missing", "d_peer")],
        ),
    )

    row, silent = remote_row_and_silence("s_missing", root)

    assert row is not None and row.id == "s_missing"
    assert silent == ()


@pytest.mark.asyncio
async def test_the_two_halves_of_a_miss_come_from_one_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The seam's own contract: one call, one dial, both facts."""
    root = tmp_path / "root"
    catalog = _relay(
        monkeypatch,
        _Catalog(
            [_Facts("d_silent", "build-box", reachable=False, reason="probe_timeout")],
            [],
        ),
    )

    assert remote_row_and_silence("s_missing", root) == (
        None,
        (UnansweredPeer("d_silent", "build-box", "probe_timeout"),),
    )
    assert catalog.calls == 1


@pytest.mark.asyncio
async def test_a_local_directory_consults_no_silence_at_all(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The residue corner: no read means no silence to report.

    R-3 measured the two-call shape answering 409 here (from a stale entry, cold
    or warm) where the note claimed 404. There is no read, so there is no silence
    this device can honestly report.
    """
    root = tmp_path / "root"
    (root / "sessions" / "s_here").mkdir(parents=True)
    catalog = _relay(
        monkeypatch,
        _Catalog(
            [_Facts("d_silent", "build-box", reachable=False, reason="probe_timeout")],
            [],
        ),
    )

    assert remote_row_and_silence("s_here", root) == (None, ())
    assert catalog.calls == 0, "a directory this device holds is not a peer question"


@pytest.mark.asyncio
async def test_a_listing_that_answered_still_answers_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No silence ⇒ the seam's contract is unchanged: ``None``, one read, no raise."""
    root = tmp_path / "root"
    catalog = _relay(
        monkeypatch,
        _Catalog([_Facts("d_live", "radiant-m4", reachable=True)], [_Row("s_1", "d_live")]),
    )

    assert await open_remote_viewer("s_missing", config_dir=root, takeover=_never_take_over) is None
    assert catalog.calls == 1


# ---------------------------------------------------------------------------
# the sentence: the copy rules the design round measured (D1-D5), each one a
# way to be dishonest that the composer must not take
# ---------------------------------------------------------------------------


def test_the_sentence_headlines_the_silence_and_never_names_a_holder() -> None:
    """D1/D2: what the reader is told FIRST is what the devices DID.

    "could not be resolved" is this product's own wording for NOT FOUND (the UI
    captions its *missing* state that way), so opening with it would state the
    reading this whole code exists to prevent. And the sentence must not raise
    the deletion idea in order to deny it: the positive claim already carries
    the truth (D2).
    """
    sentence = unresolved_peer_sentence(
        "a1b2c3d4e5f6", (UnansweredPeer("d_1", "build-box", "probe_timeout"),)
    )
    assert sentence == (
        "a1b2c3d4e5f6: build-box did not answer, so this conversation may be on "
        "that device. Retry once the connection is back; /network doctor "
        "diagnoses the link."
    )
    assert "could not be resolved" not in sentence
    assert "gone" not in sentence
    assert "no longer" not in sentence
    assert " is on " not in sentence
    # D3: the action names the control and the word that holds in both branches.
    assert "Retry once the connection is back" in sentence


def test_the_plural_says_one_of_them_and_coordinates_the_names() -> None:
    """D4: device names are free text, so a comma join lets one device read as two."""
    two = unresolved_peer_sentence(
        "a1b2c3d4e5f6",
        (
            UnansweredPeer("d_1", "build-box", "probe_timeout"),
            UnansweredPeer("d_2", "radiant-m4", "connect_failed"),
        ),
    )
    assert "build-box and radiant-m4 did not answer" in two
    assert "one of them" in two
    assert "that device" not in two
    assert " is on " not in two

    three = unresolved_peer_sentence(
        "a1b2c3d4e5f6",
        (
            UnansweredPeer("d_1", "a", "x"),
            UnansweredPeer("d_2", "b", "y"),
            UnansweredPeer("d_3", "c", "z"),
        ),
    )
    assert "a, b and c did not answer" in three

    # A name that CONTAINS the separator: the reader must not count three.
    awkward = unresolved_peer_sentence(
        "a1b2c3d4e5f6",
        (
            UnansweredPeer("d_1", "build-box, spare", "x"),
            UnansweredPeer("d_2", "radiant-m4", "y"),
        ),
    )
    assert "build-box, spare and radiant-m4 did not answer" in awkward


def test_a_device_the_membership_never_named_still_appears() -> None:
    """R-6/Q1: ``peer_rows`` builds ``name=str(facts.name or "")``, so this is live.

    Dropping the name would compose ``build-box,  did not answer`` — a double
    space and one fewer silent device than there is.
    """
    sentence = unresolved_peer_sentence(
        "a1b2c3d4e5f6",
        (
            UnansweredPeer("d_1", "build-box", "x"),
            UnansweredPeer("d_2", "", "y"),
        ),
    )
    assert UNNAMED_DEVICE in sentence
    assert "build-box and unnamed device did not answer" in sentence
    assert "  " not in sentence, "an empty name left a gap in the list"


def test_the_composer_refuses_an_empty_device_list() -> None:
    """D5: the degenerate input composes a sentence naming nobody, so it is refused.

    Not reachable from either raise site (both guard on non-empty), but the type
    is public and its ``str()`` is what a surface prints.
    """
    with pytest.raises(ValueError, match="at least one silent device"):
        unresolved_peer_sentence("a1b2c3d4e5f6", ())


def test_the_unresolved_state_rides_the_refusal_family() -> None:
    """R-4: a CLI surface renders a ``MeshRefusal``; anything else is a traceback.

    ``_pilot_act`` (``network/cli.py``) resolves the id inside
    ``open_remote_viewer`` — it is the one caller that passes no ``row=`` — and
    catches only ``TimeoutError``/``ConnectionError``, so this exception can
    escape into that module's ``main``, which prints a refusal for ``MeshRefusal``
    and re-raises everything else. Family membership is therefore part of the
    contract, not a convenience.
    """
    from local_operator.network.types import MeshRefusal

    error = PeerSessionUnresolved(
        "a1b2c3d4e5f6", (UnansweredPeer("d_1", "build-box", "probe_timeout"),)
    )
    assert isinstance(error, MeshRefusal)
    assert error.code == "session_unresolved"
    assert error.sentence == str(error)
    assert "build-box did not answer" in error.sentence


def test_the_unreachable_sentence_names_an_unnamed_device_in_both_places() -> None:
    """R2-3 (design D6): a row carrying neither name nor id still reads as one sentence.

    The named form is asserted at the routes; this is the neither-named form,
    which only the composer's own fallback (``owner_device_name or owner_device or
    UNNAMED_DEVICE``) keeps from rendering ``/network doctor  diagnoses the link.``
    — two spaces and no device to go with the one the first half just supplied.
    """
    from local_operator.resume import SessionRow

    row = SessionRow(
        "a1b2c3d4e5f6",
        1789400000.0,
        "build box chat",
        locality="remote",
        reachable=False,
        unreachable_reason="connect_failed:ConnectionRefusedError",
    )
    assert row.owner_device == "" and row.owner_device_name == ""

    sentence = unreachable_peer_sentence("a1b2c3d4e5f6", row)

    assert f"is on {UNNAMED_DEVICE}, which is unreachable" in sentence, sentence
    assert f"/network doctor {UNNAMED_DEVICE} diagnoses the link." in sentence, sentence
    assert "  " not in sentence, sentence
