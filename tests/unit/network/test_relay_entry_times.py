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

The owner's runtime here is REAL, on a LOOP OF ITS OWN
(``test_session_plane._serve_real_sessions``): the capability is advertised only
to a handle that can actually page history, and — the part a first draft of this
file got wrong — ``ServingSessionHandle`` publishes the loop it was built on for
the runtime to hop to, so building it on the CALLER's loop leaves the hop
addressing a loop that ``asyncio.run`` buries as soon as the engage returns. The
welcome then cannot land and the dial expires its ``WELCOME_TIMEOUT_S``, which
reads as a load-correlated flake rather than as a rig defect (agent review round
2, R2-1: 3/46 failures, failing runs 11.5-14.4 s against 1.2-1.7 s passing).
"""

from __future__ import annotations

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
    _RealServed,
    _seed_journal,
    _serve_real_sessions,
    _stop_real,
    _StreamClient,
    _viewer,
    _warm,
)

#: The auth field this file is about: the name the VIEWER declares and the relay
#: allowlist must forward. Spelled once, from its own parts, so a grep for the
#: literal cannot silently miss the cell that exists to pin it.
ENTRY_TIMES_AUTH_FIELD = "display_history_entry_times"

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


def _journal_times(served: dict[str, _RealServed], ids: list[str]) -> dict[str, float]:
    """The owner's OWN journal times for ``ids``, read off its disk.

    The expected values are taken from the owner rather than restated as
    constants: the claim is that the wire carries the journal's instants, and a
    fixture that merely echoed a literal back would be asserting that the test's
    own arithmetic is consistent with itself.
    """
    wanted = set(ids)
    return {
        row["id"]: row["ts"] for row in served[SESSION].transcript_entries() if row["id"] in wanted
    }


def test_the_entry_time_declaration_survives_the_relay_hop(
    peer_pair: Devices,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Q1, end to end: owner ships the join, viewer on the other device reads it.

    Three facts, one wire path. The owner BUILT a join (its own journal has the
    rows). The viewer DECLARED it can read one. And the page it received carries
    that join, keyed by the owner's own ids with the owner's own entry times —
    not the serve stamp, whose value is ~90 million seconds from these fixtures'
    journal, so no serve-clock guess can satisfy the comparison.

    Without the allowlist entry the middle fact never arrives, the owner strips
    what it built, and this page comes back with no ``entry_times`` at all.
    """
    server_a, server_b, host_a, port_a = peer_pair
    record, _host, _port = _pair(peer_pair, monkeypatch, role="drive")
    # The OWNER holds the conversation; the VIEWER is B, whose relay dials A.
    ids = _seed_journal(server_a.root, SESSION, ["the first question", "the second"])
    served = _serve_real_sessions(monkeypatch, server_a.root)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(server_b.root))
    client = None
    try:
        _warm(server_a.root, SESSION)
        _viewer(server_b)
        assert _dial_to(server_b, record, host_a, port_a) is not None
        expected = _journal_times(served, ids)
        assert len(expected) == len(
            ids
        ), f"the owner's journal lost the fixture rows: {served[SESSION].transcript_entries()}"

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
        assert join == expected, "the wire's join is not the owner's own journal"
    finally:
        if client is not None:
            client.close()
        _stop_real(served)


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
    served = _serve_real_sessions(monkeypatch, server_a.root)
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
        _stop_real(served)
