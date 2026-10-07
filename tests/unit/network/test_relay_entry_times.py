"""The entry-time declaration ACROSS THE MESH HOP, on two real relays.

WHY THIS FILE EXISTS, and why the in-process cells cannot replace it (QA round 1,
Q1). ``network/dial.py``'s ``AUTH_FIELDS`` is a CLOSED allowlist on the owner-dial
path: the relay copies a viewer's per-connection declarations through it and
nothing else. A name missing from that tuple is not refused — it is silently
dropped, so the owner reads no declaration, keeps ``conn.entry_times`` False, and
strips the ``{entry id: ts}`` join it had just built. The result is a page that is
honest but POORER than the viewer can render, with nothing logged: every row
``unstated`` (negotiated) or ``served`` (not), and never ``entry``.

Every owner-side cell that attaches IN PROCESS is green through that defect,
because the drop happens in a hop they do not cross — which is exactly how it
shipped. So the test has to cross it: two real relays, a real session on the
owner, and the viewer declaring on one side and reading the page on the other.

The owner's runtime here is REAL (a ``ServingSessionHandle`` over a real journal),
not a projection fake: the capability is advertised only to a handle that can
actually page history, and a fake that cannot would make the advertisement — and
therefore the whole cell — vacuous.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from local_operator.network import relay
from tests.unit.network.test_relay_e2e import (  # noqa: F401 — fixtures by import
    _pair,
    devices,
)
from tests.unit.network.test_session_plane import (
    SESSION,
    _dial_to,
    _seed_journal,
    _stop_all,
    _StreamClient,
    _viewer,
    _warm,
)

#: The auth field this file is about: the name the VIEWER declares and the
#: relay allowlist must forward. Spelled once, from its own parts, so a grep
#: for the literal cannot silently miss the cell that exists to pin it.
ENTRY_TIMES_AUTH_FIELD = "display_history_" + "entry_times"

#: The frame ops that can carry a display page on the way in.
_PAGE_OPS = ("frontend_sync", "welcome", "snapshot")

Devices = tuple[relay.RelayServer, relay.RelayServer, str, int]


@pytest.fixture()
def peer_pair(request: pytest.FixtureRequest) -> Devices:
    """The shared two-relay fixture, under a name this module's tests can take.

    Requested by NAME rather than imported into a test signature: pytest registers
    ``test_relay_e2e``'s fixture here by importing it, and a test parameter with
    the same name as that module-level import is a redefinition flake8 refuses
    (F811) — so the fixture itself is the shared one, not a second copy of it.
    """
    pair: Devices = request.getfixturevalue("devices")
    return pair


class _Runtime:
    """A runtime this file started, and how to stop it.

    Its own class rather than the shared ``_Served``: that one is typed for
    ``test_session_plane``'s projection fake, and this handle is a real
    ``ServingSessionHandle`` — a nominal mismatch pyright is right to refuse.
    The dictionary the callers share is typed ``Any`` at both ends.
    """

    def __init__(self, handle: Any, runtime: Any) -> None:
        self.handle = handle
        self.runtime = runtime

    def stop(self) -> None:
        try:
            self.runtime.close()
        except Exception:  # noqa: BLE001 — teardown must not mask a failure
            pass


def _serve_windowed(monkeypatch: pytest.MonkeyPatch, root: Path) -> dict[str, Any]:
    """``engage_runtime`` on ``root``, stood in for by a runtime over a REAL session.

    The same shape as ``test_session_plane``'s ``_serve``, with the one difference
    this file needs: the handle is a real ``ServingSessionHandle``, so the owner
    advertises the display-window capabilities and can build the join. A
    projection fake has no ``history_page``, so the runtime would advertise
    neither capability and the cell would prove nothing about the hop.
    """
    served: dict[str, Any] = {}
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))

    async def engage(session_id: str, cwd: str, work: Any, **kwargs: Any) -> Any:
        monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
        if session_id not in served:
            from local_operator.session.runtime.server import RuntimeServer
            from local_operator.session.runtime.serving import ServingSessionHandle
            from tests.e2e.harness import ScriptedStream, build_session

            directory = root / "sessions" / session_id
            session = build_session(directory, ScriptedStream([]), cwd=root)
            handle = ServingSessionHandle(session, asyncio.get_running_loop(), cwd=str(root))
            runtime = RuntimeServer(handle, kind="tui")
            runtime.start()
            served[session_id] = _Runtime(handle, runtime)
        from local_operator.session.runtime import registry

        async with asyncio.timeout(20):
            while not any(
                record.session_id == session_id and status == "live"
                for record, status in registry.scan(root)
            ):
                await asyncio.sleep(0.01)
        return None

    monkeypatch.setattr("local_operator.session.runtime.launch.engage_runtime", engage)
    return served


def _page_from(client: _StreamClient, *, rounds: int = 80) -> dict[str, Any]:
    """The first display page this viewer receives, or a failure naming the frames.

    Read from the WIRE rather than from the owner's objects: the claim under test
    is what a viewer on the far side of the hop actually received.
    """
    seen: list[str] = []
    for _ in range(rounds):
        frame = client.recv()
        assert frame is not None, f"the stream went quiet; frames seen: {seen}"
        seen.append(str(frame.get("op")))
        if frame.get("op") in _PAGE_OPS:
            data = frame.get("data") or {}
            page = data.get("display_history")
            if isinstance(page, dict):
                return page
    raise AssertionError(f"no display page arrived; frames seen: {seen}")


def test_the_entry_time_declaration_survives_the_relay_hop(
    peer_pair: Devices,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Q1, end to end: owner ships the join, viewer on the other device reads it.

    Three facts, one wire path. The owner BUILT a join (its own journal has the
    rows). The viewer DECLARED it can read one. And the page it received carries
    that join, keyed by the owner's own ids with the owner's own entry times —
    not the serve stamp, whose value is 70 million seconds away from these
    fixtures' journal.

    Without the allowlist entry the middle fact never arrives, the owner strips
    what it built, and this page comes back with no ``entry_times`` at all.
    """
    server_a, server_b, host_a, port_a = peer_pair
    record, _host, _port = _pair(peer_pair, monkeypatch, role="drive")
    # The OWNER holds the conversation; the VIEWER is B, whose relay dials A.
    ids = _seed_journal(server_a.root, SESSION, ["the first question", "the second"])
    served = _serve_windowed(monkeypatch, server_a.root)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(server_b.root))
    client = None
    try:
        _warm(server_a.root, SESSION)
        _viewer(server_b)
        assert _dial_to(server_b, record, host_a, port_a) is not None

        client = _StreamClient(server_b.root)
        opened = client.open_stream(
            server_a.identity.device_id,
            SESSION,
            events=True,
            frontend_state=True,
            display_window=True,
            **{ENTRY_TIMES_AUTH_FIELD: True},
            surface="terminal",
        )
        assert opened["op"] == "ack", opened

        page = _page_from(client)
        assert page.get("status", "ok") == "ok", page
        join = page.get("entry_times")
        assert isinstance(
            join, dict
        ), "the declaration did not survive the hop: the page carries no join at all"
        assert set(join) == set(ids), (join, ids)
        # The VALUES are the owner's journal times, not this device's clock.
        assert sorted(join.values()) == [1_700_000_000.0, 1_700_000_001.0]
    finally:
        if client is not None:
            client.close()
        _stop_all(served)


def test_a_viewer_that_does_not_declare_still_gets_no_join_across_the_hop(
    peer_pair: Devices,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The NEGATIVE control for the cell above, and it must not be inferred.

    A viewer that asks for the window but not for the vocabulary gets the same
    page WITHOUT the join — the strip still holds after the allowlist change, so
    adding the names did not turn the capability into an unconditional emit.
    Without this half, a page that simply always carried the join would pass the
    positive cell.
    """
    server_a, server_b, host_a, port_a = peer_pair
    record, _host, _port = _pair(peer_pair, monkeypatch, role="drive")
    ids = _seed_journal(server_a.root, SESSION, ["the first question", "the second"])
    served = _serve_windowed(monkeypatch, server_a.root)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(server_b.root))
    client = None
    try:
        _warm(server_a.root, SESSION)
        _viewer(server_b)
        assert _dial_to(server_b, record, host_a, port_a) is not None

        client = _StreamClient(server_b.root)
        opened = client.open_stream(
            server_a.identity.device_id,
            SESSION,
            events=True,
            frontend_state=True,
            display_window=True,
            surface="terminal",
        )
        assert opened["op"] == "ack", opened

        page = _page_from(client)
        assert [
            row["id"] for row in page["messages"]
        ] == ids, "the fixture served no rows, so the absence below would be vacuous"
        assert "entry_times" not in page, "the join leaked to a viewer that did not ask"
    finally:
        if client is not None:
            client.close()
        _stop_all(served)
