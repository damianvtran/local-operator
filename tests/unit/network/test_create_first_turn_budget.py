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
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
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


@contextmanager
def _link_to_owner(
    peer_pair: Devices,
    monkeypatch: pytest.MonkeyPatch,
    handle_type: type[plane._Handle],  # noqa: SLF001
) -> Iterator[tuple[relay.PeerLink, relay.RelayServer]]:
    """A live A→B link with B serving runtimes of ``handle_type``; yields (link, B)."""
    server_a, server_b, _host_a, _port_a = peer_pair
    record, _host, _port = _pair(peer_pair, monkeypatch, role="drive")
    # THE HISTORICAL BUDGET, shrunk: before this fix the first turn was waited
    # for ``op_wait_s``; the slow cells' turns outlive it.
    monkeypatch.setattr(server_b, "settings", replace(server_b.settings, op_wait_s=1.0))
    host_b, port_b = plane._listen(server_b)  # noqa: SLF001
    monkeypatch.setattr(plane, "_Handle", handle_type)
    served = plane._serve(monkeypatch, server_b.root)  # noqa: SLF001
    link = plane._dial_to(server_a, record, host_b, port_b)  # noqa: SLF001
    try:
        yield link, server_b
    finally:
        link.close("test")
        plane._stop_all(served)  # noqa: SLF001


def _create_frame(req: int) -> dict[str, Any]:
    return {
        "op": "net_session_create",
        "req": req,
        "locality": "remote",
        "cwd": "",
        "name": "",
        "prompt": "Reply with exactly BASE_OK",
    }


def _create_receipt(
    peer_pair: Devices,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    handle_type: type[plane._Handle],  # noqa: SLF001
) -> tuple[dict[str, Any], str]:
    """Create with a first prompt across the pair; return (reply detail, receipt)."""
    with _link_to_owner(peer_pair, monkeypatch, handle_type) as (link, server_b):
        frame = {**_create_frame(7), "cwd": str(server_b.root)}
        reply = link.request(frame, timeout=60.0)
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
    lines = receipt.splitlines()
    assert "turn: still running on device-b after 1.5 s — its reply shows here:" in (lines), receipt
    # D1: the follow-up is ALONE on an indented line, with no backticks a shell
    # would run as a command substitution when the line is pasted.
    assert f"  lop network sessions --peer device-b --peek {session_id}" in lines, receipt
    assert "`" not in receipt, receipt


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

    ``due`` is the create's arrival plus its slow-op deadline. A fresh create gets
    the full ``FIRST_TURN_WAIT_S`` (well past the 10 s ``op_wait_s`` that
    misreported same-build turns); a slow engage eats into it; and the
    ``op_wait_s`` floor never pushes the wait past ``due`` (review MINOR-1).
    """
    op_wait = relay.wire.OP_WAIT_S
    hop = relay.engage_hop_bound_s(op_wait)
    deadline = relay.create_op_deadline_s(op_wait)
    assert deadline == hop - relay.FIRST_TURN_REPLY_MARGIN_S
    slack = relay.FIRST_TURN_FINISH_SLACK_S
    arrival = 100.0
    due = arrival + deadline
    assert relay.first_turn_wait_s(due, op_wait, arrival) == relay.FIRST_TURN_WAIT_S
    assert relay.FIRST_TURN_WAIT_S >= 3 * op_wait
    assert relay.FIRST_TURN_WAIT_S + slack <= deadline
    # A 20 s engage leaves the rest of the deadline, less the finish slack.
    assert relay.first_turn_wait_s(due, op_wait, arrival + 20.0) == deadline - 20.0 - slack
    # 50 s gone: the floor is min(op_wait_s, remaining), so it still ends before due.
    late = relay.first_turn_wait_s(due, op_wait, arrival + 50.0)
    assert late == deadline - 50.0 - slack and arrival + 50.0 + late <= due
    # Nothing left: the shortest wait that still READS the owner's reply
    # (review MINOR-2 on #3547), never 0, which sent the prompt blind.
    assert relay.first_turn_wait_s(due, op_wait, due + 3.0) == relay.FIRST_TURN_MIN_WAIT_S
    assert 0 < relay.FIRST_TURN_MIN_WAIT_S < slack


def test_the_help_names_the_relays_first_turn_wait() -> None:
    """``--prompt``'s help spells the wait out (D2); it must stay the relay's number.

    Read from the parser's SOURCE because the CLI imports the relay lazily, which
    is why the help spells the number rather than formatting the constant.
    """
    import inspect

    from local_operator.network import cli as network_cli

    source = inspect.getsource(network_cli.add_parser)
    assert f"still running after up to {relay.FIRST_TURN_WAIT_S:g} s" in source


def _sessions_help(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], columns: int
) -> str:
    """``lop network sessions --help`` as argparse prints it ``columns`` wide."""
    import argparse

    from local_operator.network import cli as network_cli

    monkeypatch.setenv("COLUMNS", str(columns))
    parser = argparse.ArgumentParser(prog="lop")
    network_cli.add_parser(parser.add_subparsers(dest="command"))
    capsys.readouterr()
    with pytest.raises(SystemExit):
        parser.parse_args(["network", "sessions", "--help"])
    return capsys.readouterr().out


def test_the_help_says_runtime_start_up_shortens_the_wait(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """NIT-2 on #3547: the README says the 45 s is less the runtime's start-up, and
    the help said only "up to 45 s" — so a receipt reading "after 31 s" looked like
    a broken promise. The rendered help says what the budget really is, at 80
    columns, with no line past them."""
    rendered = _sessions_help(monkeypatch, capsys, 80)
    flat = " ".join(rendered.split())
    assert "still running after up to 45 s, less any time spent starting its runtime" in flat
    assert all(len(line) <= 80 for line in rendered.splitlines()), [
        line for line in rendered.splitlines() if len(line) > 80
    ]


def _still_running_receipt(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], peer: str
) -> list[str]:
    """The receipt the CLI prints for an owner's "still running after 45 s" reply."""
    from local_operator.network import cli as network_cli

    detail = {
        "session_id": "3f139cd73849",
        "admitted": True,
        "detail": "prompt admitted",
        "turn": {"state": "running", "waited_s": relay.FIRST_TURN_WAIT_S},
    }
    monkeypatch.setattr(network_cli, "_relay_answer", lambda *_a, **_k: {"ok": True, **detail})
    args: Any = SimpleNamespace(
        peer=peer,
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
    return capsys.readouterr().out.splitlines()


@pytest.mark.parametrize("peer", ["device-b", "macbook-pro", "damians-mac-mini"])
def test_the_still_running_receipt_fits_80_columns(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], peer: str
) -> None:
    """D4 on #3547: the line above the ``--peek`` command was 83-93 columns for
    ordinary peer names, so it wrapped on an 80-column terminal right above the
    line meant to be copied. Every line of the receipt now fits."""
    lines = _still_running_receipt(monkeypatch, capsys, peer)
    print("\n".join(f"{len(line):3d} | {line}" for line in lines))  # evidence (-s)
    assert f"turn: still running on {peer} after 45 s — its reply shows here:" in lines
    assert f"  lop network sessions --peer {peer} --peek 3f139cd73849" in lines
    assert all(len(line) <= 80 for line in lines), lines


def test_with_no_time_left_the_owner_still_reads_the_turns_reply(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """MINOR-2 on #3547: an engage that ate the whole budget left a wait of 0, and
    ``exchange(timeout_s=0)`` sent the prompt without reading anything. The
    receipt then claimed "admitted" and "still running … after 0 s" for a turn
    that had in fact completed at once. With the 1 s floor the owner reads the
    reply: a fast turn reports completed, and no receipt names a 0 s wait."""

    class _Fast(_TimedHandle):
        delay_s = 0.0

    # "The engage left nothing": the create reads its remaining lane time here.
    monkeypatch.setattr(relay, "slow_op_remaining_s", lambda: 0.0)
    detail, receipt = _create_receipt(_pair_of(request), monkeypatch, capsys, _Fast)
    turn = detail["turn"]
    assert "after 0 s" not in receipt, receipt
    assert turn.get("waited_s", relay.FIRST_TURN_MIN_WAIT_S) >= relay.FIRST_TURN_MIN_WAIT_S
    assert turn["state"] == "settled", detail
    assert "turn: completed on device-b" in receipt, receipt


# ---------------------------------------------------------------------------
# The link stays live while a first turn runs (review MAJOR-1/2, QA Q1/Q2)
# ---------------------------------------------------------------------------


class _ThreeSecondTurn(_TimedHandle):
    delay_s = 3.0


def _scale_the_hop(monkeypatch: pytest.MonkeyPatch, server_b: relay.RelayServer) -> float:
    """Shrink the requester's 60 s hop to 4 s, keeping every ratio; return it.

    The create's slow-op deadline is read at relay construction, so it is
    re-derived here — ONLY where the create is already on the slow lane, so this
    rig cannot put a build that dispatches it inline onto the lane by itself.
    """
    # ``op_wait_s`` 2 s is the bound every OTHER caller on the link uses here, and
    # it is the floor of the pre-fix budget: two inline 2.5 s waits (the second
    # floored at 2 s) cannot both answer inside the 4 s hop, which is Q1.
    monkeypatch.setattr(server_b, "settings", replace(server_b.settings, op_wait_s=2.0))
    monkeypatch.setattr(relay, "ENGAGE_DEADLINE_S", 4.0)
    monkeypatch.setattr(relay, "FIRST_TURN_WAIT_S", 2.5)
    monkeypatch.setattr(relay, "FIRST_TURN_REPLY_MARGIN_S", 0.5)
    monkeypatch.setattr(relay, "FIRST_TURN_FINISH_SLACK_S", 0.5, raising=False)
    slow = server_b._slow_ops  # noqa: SLF001
    if "net_session_create" in slow:
        monkeypatch.setitem(
            slow, "net_session_create", relay.create_op_deadline_s(server_b.settings.op_wait_s)
        )
    return relay.engage_hop_bound_s(server_b.settings.op_wait_s)


def test_two_concurrent_slow_creates_on_one_link_both_get_receipts(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Q1 / MAJOR-1: the second create used to queue behind the first's wait.

    Inline on the reader, create #2 started only when #1 answered, then took a
    fresh budget of its own, and its requester (waiting the hop from when it
    SENT) gave up: no receipt for a session that existed and whose turn ran.
    """
    with _link_to_owner(_pair_of(request), monkeypatch, _ThreeSecondTurn) as (link, server_b):
        hop = _scale_the_hop(monkeypatch, server_b)
        replies: dict[int, Any] = {}

        def _create(req: int) -> None:
            replies[req] = link.request(
                {**_create_frame(req), "cwd": str(server_b.root)}, timeout=hop
            )

        threads = [threading.Thread(target=_create, args=(req,)) for req in (101, 102)]
        # STAGGERED BY 0.3 s, past #1's engage: this rig's stand-in engage
        # (``plane._serve``) can wedge when two in-process runtimes publish at
        # once, which is the fake's limit, not the relay's (production engages
        # spawn processes). The overlap that matters is the two TURN waits, and
        # #2 still arrives while #1's turn is held.
        for thread in threads:
            thread.start()
            time.sleep(0.3)
        for thread in threads:
            thread.join(timeout=hop + 5.0)
    for req in (101, 102):
        reply = replies.get(req)
        assert reply is not None and reply["op"] == "ack", (req, replies)
        assert reply["detail"]["turn"]["state"] == "running", (req, reply)
        assert "waited_s" in reply["detail"]["turn"], (req, reply)


def test_the_link_answers_other_ops_while_a_first_turn_runs(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Q2 / MAJOR-2: a ping and a history read are not held behind the turn.

    Each is asked 0.2 s into a 3 s first turn with ``op_wait_s`` (2 s) as its
    bound — the budget every other caller on a link uses. Inline, the reader was
    parked in the create until the turn's wait ended, so both timed out and the
    peer read as unreachable.
    """
    pair = _pair_of(request)
    plane._seed_journal(pair[1].root, plane.SESSION, ["an earlier question"])  # noqa: SLF001
    with _link_to_owner(pair, monkeypatch, _ThreeSecondTurn) as (link, server_b):
        hop = _scale_the_hop(monkeypatch, server_b)
        bound = server_b.settings.op_wait_s
        created: dict[str, Any] = {}
        creator = threading.Thread(
            target=lambda: created.update(
                reply=link.request({**_create_frame(201), "cwd": str(server_b.root)}, timeout=hop)
            )
        )
        creator.start()
        time.sleep(0.2)
        began = time.monotonic()
        pong = link.request({"op": "ping", "req": 202}, timeout=bound)
        ping_s = time.monotonic() - began
        began = time.monotonic()
        page = link.request(
            {
                "op": "net_session_history",
                "req": 203,
                "locality": "remote",
                "session_id": plane.SESSION,
                "limit": 50,
            },
            timeout=bound,
        )
        peek_s = time.monotonic() - began
        creator.join(timeout=hop + 5.0)
    assert pong is not None and pong["op"] == "ack", (pong, ping_s)
    assert page is not None and page["op"] == "ack", (page, peek_s)
    assert ping_s < bound and peek_s < bound, (ping_s, peek_s)
    reply = created.get("reply")
    assert reply is not None and reply["detail"]["turn"]["state"] == "running", created
