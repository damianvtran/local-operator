"""``/resume`` on a session ANOTHER device owns (QA round 10, Q-R10-2).

WHAT THIS PINS, AND WHY IT NEEDS TWO REAL RELAYS. ``/resume``'s factory looks a
session up in THIS machine's store, so without its guard a peer's id boots a
brand-new LOCAL session under that id — the same conversation id then exists on
two devices at once, which is the permanent routing ambiguity the mobility design
exists to prevent, and the user is told nothing. The guard is one lookup into the
producer's cache, and that cache is filled by ``peer_session_rows`` against a
live relay, so a test that stubs either half proves nothing about the guard: the
state it must recognise is "this id is a row of the federated listing", which is
exactly the answer the ENVELOPE bug made empty (Q-R10-1). With the cache empty
this test's own assertion fails and the app mints the duplicate — which is the
before/after pair this file was written from.

The rig is the shared two-relay one: two config roots, two identities, two
relays on loopback, the product's own pairing ceremony, and a real dial. The app
is the real ``OperatorApp``; only its session factory is the pilot's fake, as it
is in every other TUI test.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from local_operator.network import relay, store
from local_operator.session.peer_rows import clear_cache, peer_session_rows
from tests.unit.network.test_relay_e2e import (  # noqa: F401 — fixtures by import
    _pair,
    devices,
)

REMOTE_ID = "8dbb1d07f3a3"

Devices = tuple[relay.RelayServer, relay.RelayServer, str, int]


@pytest.fixture()
def root(tmp_path: Path) -> Path:
    """The parent the two relays' config roots are built under.

    ``test_relay_e2e``'s ``devices`` fixture asks for a ``root``; it is defined in
    ``tests/unit/network/conftest.py``, which is not on this directory's conftest
    chain, so the same one-liner is declared here. Same body as that fixture, on
    purpose: this file is outside that package and the two must not diverge.
    """
    return tmp_path


@pytest.fixture()
def peer_pair(request: pytest.FixtureRequest) -> Devices:
    """The shared two-relay fixture, under a name these tests can take.

    Requested by NAME rather than imported into a signature: pytest registers
    ``test_relay_e2e``'s ``devices`` fixture here by importing it, and a test
    parameter sharing that name is a redefinition flake8 refuses (F811).
    """
    pair: Devices = request.getfixturevalue("devices")
    return pair


@pytest.fixture(autouse=True)
def _no_cached_rows() -> Any:
    """The producer caches by config root; a test must not inherit another's."""
    clear_cache()
    yield
    clear_cache()


def _listen(server: relay.RelayServer) -> tuple[str, int]:
    """Give a device a listener, so the OTHER device can dial it."""
    host, port = server.bind()
    server.bind_control()
    server.start()
    return str(host), int(port)


def _seed(root: Path, session_id: str) -> None:
    """A session the peer holds: a directory with the one activity file that ranks it."""
    directory = root / "sessions" / session_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "transcript.jsonl").write_text("", encoding="utf-8")


def _notices(app: Any) -> list[str]:
    from local_operator.tui.widgets.transcript import NoticeBlock, TranscriptView

    return [
        block._text
        for block in app.query_one(TranscriptView).blocks()
        if isinstance(block, NoticeBlock)
    ]


@pytest.mark.asyncio
async def test_resume_of_a_peers_session_names_the_device_and_mints_nothing(
    peer_pair: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The guard speaks, and NO directory appears under the peer's id."""
    server_a, server_b, _host_a, _port_a = peer_pair
    record, _host, _port = _pair(peer_pair, monkeypatch, role="drive")
    host_b, port_b = _listen(server_b)
    _seed(server_b.root, REMOTE_ID)
    link, reason = server_a.dial(record.network_id, host=f"{host_b}:{port_b}", epoch=record.epoch)
    assert link is not None, reason
    # The app resolves its root from the AMBIENT config dir — the same way a user
    # runs it — so the test points it at the viewer's own root.
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(server_a.root))
    try:
        # THE SIDEBAR'S OWN READ, through the production producer and a live
        # relay. This is the state the guard is written against; with the reply
        # envelope unread (Q-R10-1) this assertion is what fails, and the guard
        # below is then unreachable.
        assert [row.id for row in peer_session_rows(server_a.root)] == [REMOTE_ID]

        from local_operator.tui.app import OperatorApp
        from tests.unit.tui.test_app_pilot import FakeSession, _factory

        #: WHETHER THE RESUME FACTORY RAN. This is the assertion that carries the
        #: finding: the duplicate session Q-R10-2 is about is minted BY THAT
        #: FACTORY, so "the guard returned before the factory was reached" is the
        #: mechanism, and a flag records it here rather than a directory existing
        #: (the pilot's fake builds no directory, so a `not exists` check alone
        #: would pass on the broken tree as well).
        launched: list[str | None] = []

        def _resume_factory(resume_id: str | None = None) -> Any:
            launched.append(resume_id)
            return _factory(FakeSession())

        # A RESUME-CAPABLE LAUNCHER, which is what `_cmd_resume` requires before
        # it reaches any of its own logic (`_resume_factory is None` answers
        # "resume requires a resume-capable launcher" and returns).
        app = OperatorApp(lambda: _factory(FakeSession()), resume_factory=_resume_factory)
        async with app.run_test(size=(100, 30)) as pilot:
            for _ in range(40):
                await pilot.pause()
                if app._session is not None:
                    break
            app._run_slash_command(f"/resume {REMOTE_ID}")
            await pilot.pause()
            await pilot.pause()

            shown = " ".join(_notices(app))
            assert (
                server_b.identity.name in shown
            ), f"the guard did not name the owning device: {shown!r}"
            assert REMOTE_ID in shown, shown
            # SLICE V: the guard now OPENS the session on the peer rather than
            # announcing it (mesh build plan §0 finding 2) — the device is still
            # named, and the assertions below (no local factory, no local
            # directory) are the ones that pin "nothing was minted here".
            assert f"opened {REMOTE_ID} on" in shown, shown

        # AND THE FACTORY WAS NEVER REACHED. The other half of the guard's job: a
        # local session under the peer's id is minted by the resume factory, so
        # "the factory did not run" is the mechanism, and the directory check
        # below is the same fact on disk for a launcher that does.
        assert launched == [], (
            "resuming a peer's session reached the resume factory, which is the "
            "duplicate session Q-R10-2 is about"
        )
        assert not (server_a.root / "sessions" / REMOTE_ID).exists()
        stored = store.list_networks(server_a.root)
        assert [row.network_id for row in stored] == [record.network_id]
        link.close("test")
    finally:
        server_b.stop()


@pytest.mark.asyncio
async def test_the_guard_speaks_without_the_sidebar_ever_having_polled(
    peer_pair: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    """UX round 3, U20: the SAME guard, on a COLD cache, must still speak.

    The test above warms the producer's cache first — ``peer_session_rows`` is
    called before the app runs — and that is the surface which hid this: the cache
    is filled by the SIDEBAR's poll, so a user who has not opened the sidebar this
    session got no guard at all. Measured A/B on one build and one id: cold, the
    composer cleared and nothing was said, twice; after opening the sidebar for a
    few seconds, the identical command printed the guard's sentence.

    Nothing here warms the cache, and the assertion is the guard's own sentence.
    The refresh is gated on this device holding no directory for the id, so the
    read this test exercises is the one a LOCAL resume never pays.
    """
    server_a, server_b, _host_a, _port_a = peer_pair
    record, _host, _port = _pair(peer_pair, monkeypatch, role="drive")
    host_b, port_b = _listen(server_b)
    _seed(server_b.root, REMOTE_ID)
    link, reason = server_a.dial(record.network_id, host=f"{host_b}:{port_b}", epoch=record.epoch)
    assert link is not None, reason
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(server_a.root))
    try:
        # PRECONDITION: the cache really is cold. Read through the module's own
        # state rather than by calling the producer, which would warm it.
        from local_operator.session import peer_rows as peer_rows_mod

        assert peer_rows_mod._CACHE == {}, "the cache is warm, so this test proves nothing"
        assert not (server_a.root / "sessions" / REMOTE_ID).is_dir()

        from local_operator.tui.app import OperatorApp
        from tests.unit.tui.test_app_pilot import FakeSession, _factory

        launched: list[str | None] = []

        def _resume_factory(resume_id: str | None = None) -> Any:
            launched.append(resume_id)
            return _factory(FakeSession())

        app = OperatorApp(lambda: _factory(FakeSession()), resume_factory=_resume_factory)
        async with app.run_test(size=(100, 30)) as pilot:
            for _ in range(40):
                await pilot.pause()
                if app._session is not None:
                    break
            app._run_slash_command(f"/resume {REMOTE_ID}")
            await pilot.pause()
            await pilot.pause()
            shown = " ".join(_notices(app))
            assert server_b.identity.name in shown, (
                "the guard was silent on a cold cache — which is the U20 defect, "
                f"byte-for-byte: {shown!r}"
            )
            # SLICE V: the guard now OPENS the session on the peer rather than
            # announcing it (mesh build plan §0 finding 2) — the device is still
            # named, and the assertions below (no local factory, no local
            # directory) are the ones that pin "nothing was minted here".
            assert f"opened {REMOTE_ID} on" in shown, shown

        assert launched == [], "the guard let a peer's id reach the resume factory"
        assert not (server_a.root / "sessions" / REMOTE_ID).exists()
        link.close("test")
    finally:
        server_b.stop()


@pytest.mark.asyncio
async def test_picking_a_peer_row_from_the_sidebar_names_the_device_not_the_local_store(
    peer_pair: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    """UX round 5, U27: the LIST PICK asks the same guard ``/resume`` asks.

    Measured on the branch, in ONE frame and two lines apart: with the sidebar
    focused on a row under the ``⇄ pixel-8`` heading, ``enter`` printed

        ! Could not open conversation: This conversation is no longer available

    while ``/resume <the same id>`` printed

        ! 8dbb1d07f3a3 is running on pixel-8 — /network sessions --peer pixel-8 …

    The session exists only on the peer, so the pick told the user their
    conversation was gone while it was running on the other machine — and it took
    the local-not-found path the ``/resume`` arm exists to avoid. The pick posts
    ``SessionSidebar.Selected(id)`` (``widgets/session_sidebar.py``,
    ``action_select``, which is what ``enter`` on the list runs), and this posts
    exactly that message: the act the round measured, through the handler the app
    wires it to.

    TWO ASSERTIONS CARRY THE FINDING, and the second is the one a fix that merely
    added a sentence would fail: the peer's name reaches the transcript, AND
    nothing was started — the navigation must not run at all, because a
    navigation that runs prints the local store's sentence over the guard's.
    """
    server_a, server_b, _host_a, _port_a = peer_pair
    record, _host, _port = _pair(peer_pair, monkeypatch, role="drive")
    host_b, port_b = _listen(server_b)
    _seed(server_b.root, REMOTE_ID)
    link, reason = server_a.dial(record.network_id, host=f"{host_b}:{port_b}", epoch=record.epoch)
    assert link is not None, reason
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(server_a.root))
    try:
        # THE ROW THE USER PICKED has to be in the list they picked it from: the
        # sidebar's own read, through the production producer and a live relay.
        assert [row.id for row in peer_session_rows(server_a.root)] == [REMOTE_ID]

        from local_operator.tui.app import OperatorApp
        from local_operator.tui.widgets.session_sidebar import SessionSidebar
        from tests.unit.tui.test_app_pilot import FakeSession, _factory

        launched: list[str | None] = []

        def _resume_factory(resume_id: str | None = None) -> Any:
            launched.append(resume_id)
            return _factory(FakeSession())

        app = OperatorApp(lambda: _factory(FakeSession()), resume_factory=_resume_factory)
        async with app.run_test(size=(100, 30)) as pilot:
            for _ in range(40):
                await pilot.pause()
                if app._session is not None:
                    break
            app.post_message(SessionSidebar.Selected(REMOTE_ID))
            await pilot.pause()
            await pilot.pause()

            shown = " ".join(_notices(app))
            assert server_b.identity.name in shown, (
                "the pick did not name the device holding the session: " f"{shown!r}"
            )
            assert REMOTE_ID in shown, shown
            # SLICE V: the guard now OPENS the session on the peer rather than
            # announcing it (mesh build plan §0 finding 2) — the device is still
            # named, and the assertions below (no local factory, no local
            # directory) are the ones that pin "nothing was minted here".
            assert f"opened {REMOTE_ID} on" in shown, shown
            # THE LOCAL STORE'S SENTENCE IS THE DEFECT, so its absence is the
            # assertion — on the pick path it was the whole answer.
            assert "no longer available" not in shown, shown
            assert "Could not open conversation" not in shown, shown
            # AND NOTHING WAS STARTED. A navigation would have printed the local
            # failure above and left the app mid-transition.
            assert (
                app._sidebar_navigation.requested_id == ""
            ), "the pick started a navigation for a session this device does not have"

        assert launched == [], "picking a peer's row reached the resume factory"
        assert not (server_a.root / "sessions" / REMOTE_ID).exists()
        link.close("test")
    finally:
        server_b.stop()
