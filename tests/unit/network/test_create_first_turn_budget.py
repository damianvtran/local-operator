"""A create's first-turn receipt on a REAL two-relay pair, per turn duration.

WHY THIS FILE EXISTS. #3507 made ``lop network sessions --peer X --create
--prompt …`` report how the first turn ended, and kept one honest line for an
owner that cannot settle it: ``turn: running on X — this receipt cannot see how
it ends``. Live, that line fired on SAME-BUILD pairs (2026-11-06 23:01Z, both
devices on 0.101.4; 2026-11-07 05:39Z on 0.101.7) for turns that went on to
complete. The cause is a WAIT BUDGET, not a capability gap: the owning relay
waited ``op_wait_s`` (10 s) for a ``wait_complete`` reply that, by design, only
arrives after the whole turn, and a fresh runtime's first model turn on a
loaded machine routinely outlives ten seconds. Silence past the budget was then
reported in the same words as an owner that genuinely cannot settle a turn.

THE RIG is the shared two-relay fixture (``devices``) with the owning side's
runtime a real ``RuntimeServer`` over a real socket, whose handle runs a turn
of a duration this file controls. The receipt is rendered by the CLI's own
handler (``_cmd_sessions``) from the reply the owning relay sent across the
link, so every hop between the turn and the printed line is production code.

The budget is shrunk (``op_wait_s`` and ``FIRST_TURN_WAIT_S``) so a "slow"
turn costs a second of test time, not a minute; the ORDER between them is what
the cells assert, not the absolute numbers.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import pytest

from local_operator.network import relay
from local_operator.session.errors import FirstTurnFailed
from tests.unit.network import test_session_plane as plane
from tests.unit.network.test_relay_e2e import (  # noqa: F401 — fixtures by import
    _pair,
    devices,
)

Devices = tuple[relay.RelayServer, relay.RelayServer, str, int]


class _TimedHandle(plane._Handle):  # noqa: SLF001
    """A handle that HONOURS ``wait_complete`` and runs a turn of known length.

    ``wait_complete`` in the signature is what makes the runtime's prompt op
    treat this owner as a settling one (``server.py``'s ``honours_wait`` probe),
    exactly like the production ``ServingSessionHandle``.
    """

    delay_s = 0.0
    failure = ""

    async def prompt(  # noqa: ANN201
        self,
        text: str,
        images: Any = None,
        command_id: str | None = None,
        wait_complete: bool = False,
    ):
        receipt = await self._record("prompt", text)
        if wait_complete:
            await asyncio.sleep(self.delay_s)
            if self.failure:
                raise FirstTurnFailed(self.failure)
        return receipt


def _pair_of(request: pytest.FixtureRequest) -> Devices:
    """The shared two-relay fixture, requested by NAME (``devices`` is imported
    for registration only — a same-named parameter would be F811)."""
    pair: Devices = request.getfixturevalue("devices")
    return pair


def _create_receipt(
    peer_pair: Devices,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    handle_type: type[plane._Handle],  # noqa: SLF001
) -> tuple[dict[str, Any], str]:
    """Create with a first prompt across the pair; return (reply detail, receipt)."""
    server_a, server_b, _host_a, _port_a = peer_pair
    record, _host, _port = _pair(peer_pair, monkeypatch, role="drive")
    # THE HISTORICAL BUDGET, shrunk: before this fix the first turn was waited
    # for ``op_wait_s``; the slow cell's turn outlives it.
    monkeypatch.setattr(server_b, "settings", replace(server_b.settings, op_wait_s=1.0))
    host_b, port_b = plane._listen(server_b)  # noqa: SLF001
    monkeypatch.setattr(plane, "_Handle", handle_type)
    served = plane._serve(monkeypatch, server_b.root)  # noqa: SLF001
    try:
        link = plane._dial_to(server_a, record, host_b, port_b)  # noqa: SLF001
        reply = link.request(
            {
                "op": "net_session_create",
                "req": 7,
                "locality": "remote",
                "cwd": str(server_b.root),
                "name": "",
                "prompt": "Reply with exactly BASE_OK",
            },
            timeout=60.0,
        )
        link.close("test")
    finally:
        plane._stop_all(served)  # noqa: SLF001
    assert reply is not None and reply["op"] == "ack", reply
    detail = reply["detail"]

    from local_operator.network import cli as network_cli

    # THE REQUESTER'S OWN RELAY PASSES THE PEER'S DETAIL THROUGH UNCHANGED
    # (``_ctl_peer_create``), so its answer is this reply; the handler below is
    # the one argv dispatches to.
    monkeypatch.setattr(network_cli, "_relay_answer", lambda *_a, **_k: {"ok": True, **detail})
    args: Any = SimpleNamespace(
        peer="device-b",
        create=True,
        name="",
        prompt="Reply with exactly BASE_OK",
        cwd="",
        model="",
        profile="",
        agent="",
        agent_id="",
        team="",
        effort="",
        yolo=False,
        json=False,
        **{key: "" for key in ("stop", "engage", "send", "steer", "slash", "peek")},
    )
    capsys.readouterr()
    assert network_cli._cmd_sessions(args) == 0  # noqa: SLF001
    receipt = capsys.readouterr().out
    print(f"\n--- receipt ({handle_type.__name__}) ---\n{receipt}")  # rig evidence (-s)
    return detail, receipt


def test_a_turn_that_outlives_op_wait_still_completes(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """THE LIVE DEFECT: a normal turn longer than ``op_wait_s`` reads as completed.

    On main the owning relay gives up at ``op_wait_s`` and the receipt says it
    "cannot see how it ends" for a same-build owner whose turn then completes.
    """

    class _Slowish(_TimedHandle):
        delay_s = 2.0

    detail, receipt = _create_receipt(_pair_of(request), monkeypatch, capsys, _Slowish)
    assert detail["turn"]["state"] == "settled", detail
    assert "turn: completed on device-b" in receipt, receipt
    assert "cannot see" not in receipt, receipt


def test_a_fast_turn_completes(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class _Fast(_TimedHandle):
        delay_s = 0.05

    detail, receipt = _create_receipt(_pair_of(request), monkeypatch, capsys, _Fast)
    assert detail["turn"]["state"] == "settled", detail
    assert "turn: completed on device-b" in receipt, receipt


def test_a_turn_past_the_first_turn_budget_says_it_stopped_waiting(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Past the budget the receipt says it STOPPED WAITING, never that it cannot see.

    A same-build owner CAN settle the turn; the requester merely stopped
    listening. The line names how long it waited and the read that follows up.
    """

    class _Slow(_TimedHandle):
        delay_s = 3.0

    monkeypatch.setattr(relay, "FIRST_TURN_WAIT_S", 1.5, raising=False)
    detail, receipt = _create_receipt(_pair_of(request), monkeypatch, capsys, _Slow)
    assert detail["turn"]["state"] == "running", detail
    session_id = detail["session_id"]
    assert "cannot see" not in receipt, receipt
    assert "turn: still running on device-b after 1.5 s" in receipt, receipt
    assert f"--peek {session_id}" in receipt, receipt


def test_a_failed_turn_reports_its_cause(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class _Failing(_TimedHandle):
        delay_s = 0.2
        failure = "'anthropic' is rate-limited"

    detail, receipt = _create_receipt(_pair_of(request), monkeypatch, capsys, _Failing)
    assert detail["turn"]["state"] == "failed", detail
    assert "turn failed on device-b" in receipt, receipt
    assert "rate-limited" in receipt, receipt


def test_a_genuinely_old_owner_still_says_it_cannot_see(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """NEGATIVE CONTROL: an owner that never hears ``wait_complete`` keeps the line.

    ``plane._Handle``'s ``prompt`` has no ``wait_complete`` parameter, so the
    runtime acks at the append without the ``settled`` marker — the one case
    where this side genuinely cannot see the end.
    """
    old_owner = plane._Handle  # noqa: SLF001
    detail, receipt = _create_receipt(_pair_of(request), monkeypatch, capsys, old_owner)
    assert detail["turn"]["state"] == "running", detail
    assert "this receipt cannot see how it ends" in receipt, receipt
    assert "still running" not in receipt, receipt


def test_the_first_turn_wait_fits_inside_the_requesters_hop() -> None:
    """The budget arithmetic: long enough for a normal first turn, never past the hop.

    A fresh create gets the full ``FIRST_TURN_WAIT_S`` (well past the 10 s
    ``op_wait_s`` that misreported same-build turns); a slow engage eats into it
    so the reply still lands inside the requester's ``engage_hop_bound_s``; and
    the floor is the historical ``op_wait_s``, never less.
    """
    op_wait = relay.wire.OP_WAIT_S
    hop = relay.engage_hop_bound_s(op_wait)
    assert relay.first_turn_wait_s(100.0, op_wait, 100.0) == relay.FIRST_TURN_WAIT_S
    assert relay.FIRST_TURN_WAIT_S >= 3 * op_wait
    assert relay.FIRST_TURN_WAIT_S + relay.FIRST_TURN_REPLY_MARGIN_S <= hop
    # A 20 s engage leaves hop - margin - 20 for the turn.
    clipped = relay.first_turn_wait_s(100.0, op_wait, 120.0)
    assert clipped == hop - relay.FIRST_TURN_REPLY_MARGIN_S - 20.0
    assert 20.0 + clipped + relay.FIRST_TURN_REPLY_MARGIN_S == hop
    # An engage that already spent the hop still gets the historical floor.
    assert relay.first_turn_wait_s(100.0, op_wait, 100.0 + hop) == op_wait
