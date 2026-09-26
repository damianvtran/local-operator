"""The joining device's two-phase pair: ``--park`` opens it, ``--confirm`` answers it.

WHY THIS IS ITS OWN FILE. ``test_pairing_confirm.py`` covers the INVITER's parked
question — the relay parks it in a file because a launchd daemon has no terminal, and
``lop network confirm`` answers it. This is the mirror on the joining device, and it is
the half an AGENT drives, so the properties under test are the ones the human gate
rests on:

* a parked pairing sends NOTHING to the peer until a person's transcription arrives
  (the socket is held by the process that dialled, and the frame that tells the inviter
  a human is present is the frame the answer releases);
* the confirmation is checked against THIS device's own derivation, and a mistyped
  digit leaves the ceremony open rather than spending the invite;
* the invocation that answers is a different process from the one that dialled, which
  is the whole shape — and the tool that shells both carries only the value the caller
  supplied for the second one.

The humans are supplied by the test, which is what a test is allowed to do: the
inviter's person is the real parked question answered through the relay's own control
op, and the joiner's person is a second invocation of the CLI exactly as an agent would
make it.
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from argparse import Namespace
from typing import Any

import pytest

from local_operator.harness.types import ToolContext
from local_operator.network import cli as net_cli
from local_operator.network import invite as invite_mod
from local_operator.network import relay, store
from local_operator.network import tool as net_tool
from local_operator.network import types, wire
from local_operator.network.handshake import (
    Credential,
    Handshake,
    pair_abort_frame,
    pair_timeout_seconds,
    sas_matches,
)
from tests.unit.network.test_relay_e2e import (  # noqa: F401 — fixtures by import
    _answer_confirmation,
    _init_network,
    devices,
)


def _park_args(**overrides: Any) -> Namespace:
    """The Namespace argparse builds for ``join @<token> --park --json``."""
    base: dict[str, Any] = {
        "confirm": "",
        "park": True,
        "sas_stdin": False,
        "verify": False,
        "emit_sas": False,
        "name": "",
        "host": "",
        "json": True,
        "token": "",
    }
    base.update(overrides)
    return Namespace(**base)


def _confirm_args(code: str) -> Namespace:
    """The Namespace argparse builds for ``join --confirm <code> --json``."""
    return _park_args(confirm=code, park=False)


def _minted(server: relay.RelayServer, *, role: str = "drive", ttl_s: float = 600.0) -> Any:
    """A network on ``server`` with one invite, the way ``lop network invite`` does it.

    The hosts come from ``--hosts``' own argument, so the envelope names an endpoint a
    JOINEER can dial without a ``--host`` override: the agent path has no override, and
    an invite that did not carry its endpoint would make the tool unusable against a
    loopback test relay.
    """
    record = _init_network(server)
    state = store.load_secrets(record.network_id, server.root)
    minted = invite_mod.mint(
        record,
        state.secret,
        role=role,
        ttl_s=ttl_s,
        hosts=[f"127.0.0.1:{server.settings.port}"],
    )
    record.invites.append(minted.record)
    store.save(record, server.root)
    store.save_invite_token(minted.record.invite_id, minted.token, server.root)
    return record, minted


def _park(server_b: relay.RelayServer, *, host: str, port: int, minted: Any) -> Any:
    """Phase one, driven exactly as ``lop network join @token --park`` drives it."""
    return net_cli._join_one(  # noqa: SLF001 — the CLI's own driver, run as the CLI runs it
        host=f"{host}:{port}",
        token=minted.token,
        envelope=minted.envelope,
        identity=server_b.identity,
        settings=relay.NetworkSettings(port=0, listen_address="127.0.0.1"),
        args=_park_args(name=server_b.identity.name),
        wire=wire,
        Handshake=Handshake,
        Credential=Credential,
        pair_abort_frame=pair_abort_frame,
        pair_timeout_seconds=pair_timeout_seconds,
        sas_matches=sas_matches,
        invite_mod=invite_mod,
        store=store,
        relay_mod=relay,
    )


def _wait_for_parked(root: Any, *, timeout: float = 15.0) -> types.PendingJoin:
    """The parked ceremony phase one writes, waited for by the second invocation."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        rows = store.pending_joins(root)
        if rows:
            return rows[-1]
        time.sleep(0.05)
    raise AssertionError("the parking invocation never wrote a pending-join record")


def test_a_parked_pairing_joins_only_after_the_code_arrives(
    devices: tuple[relay.RelayServer, relay.RelayServer, str, int],  # noqa: F811
) -> None:
    """The ceremony, with the parker in a thread because it WAITS for its answer.

    The order of the assertions is the property: nothing has been sent while the
    ceremony is parked, a wrong code changes nothing at all, and the right one — the
    transcription this device derived, which is what a person types after comparing the
    two screens — completes a pair over real sockets.
    """
    server_a, server_b, host, port = devices
    record, minted = _minted(server_a)

    parked: dict[str, Any] = {}
    failures: list[BaseException] = []

    def _park_it() -> None:
        try:
            parked["result"] = _park(server_b, host=host, port=port, minted=minted)
        except BaseException as exc:  # noqa: BLE001 — reported below, not swallowed
            failures.append(exc)

    thread = threading.Thread(target=_park_it, daemon=True)
    thread.start()
    row = _wait_for_parked(server_b.root)
    try:
        assert row.status == "awaiting_confirmation"
        assert len(row.sas) == 6 and row.sas.isdigit()
        assert row.invite_id == minted.record.invite_id
        assert row.fingerprint
        assert net_cli._process_alive(row.pid)  # noqa: SLF001 — the ceremony's holder
        assert thread.is_alive(), "a parked pairing must WAIT rather than finish"

        # NOTHING HAS BEEN SENT: no transcription reached the inviter, so its own
        # parked question does not exist, the joiner holds no network, and the
        # network still has exactly one member.
        assert server_a._ctl_pair_pending({}) == []  # noqa: SLF001 — the CLI's control op
        assert store.list_networks(server_b.root) == []
        assert len(store.load(record.network_id, server_a.root).active_members()) == 1

        # A SECOND PARK ON THE SAME INVITE IS REFUSED, and before it dials: two sockets
        # for one invite would race for the single admission the inviter grants, and the
        # first parker would be left waiting on an answer written for the second.
        with pytest.raises(types.MeshRefusal) as duplicate:
            net_cli._cmd_join(_park_args(token=minted.token))  # noqa: SLF001
        assert duplicate.value.code == "pairing_already_parked"
        assert store.pending_joins(server_b.root)[-1].pid == row.pid

        # A MISTYPED DIGIT IS NOT A SPENT INVITE: it is refused by name, the ceremony
        # stays open, and the peer is still untouched.
        wrong = f"{(int(row.sas) + 1) % 1000000:06d}"
        with pytest.raises(types.MeshRefusal) as refused:
            net_cli._cmd_join(_confirm_args(wrong))  # noqa: SLF001
        assert refused.value.code == "sas_mismatch"
        assert store.pending_joins(server_b.root)[-1].status == "awaiting_confirmation"
        assert server_a._ctl_pair_pending({}) == []  # noqa: SLF001

        answered: dict[str, Any] = {}

        def _inviter_person() -> None:
            answered.update(_answer_confirmation(server_a) or {})

        answer_thread = threading.Thread(target=_inviter_person, daemon=True)
        answer_thread.start()
        try:
            # THE JOINER'S PERSON: the code this device printed, read on its own
            # screen and typed into a SECOND invocation — the second process is the
            # shape, and it is what makes this the two-phase pair rather than a prompt.
            code = net_cli._cmd_join(_confirm_args(row.sas))  # noqa: SLF001
        finally:
            answer_thread.join(20)
            thread.join(20)

        assert not failures, f"the parking side raised: {failures[0]!r}"
        assert code == 0
        assert answered, "the inviter's person never saw the parked question"
        lines, payload = parked["result"]
        assert payload["network_id"] == record.network_id
        assert payload["device_id"] == server_b.identity.device_id
        assert any("joined" in line for line in lines)
        # The joining device now holds the network, the inviter counts two members, and
        # the parked record is gone rather than lingering as a live ceremony.
        assert [item.network_id for item in store.list_networks(server_b.root)] == [
            record.network_id
        ]
        assert len(store.load(record.network_id, server_a.root).active_members()) == 2
        assert store.pending_joins(server_b.root) == []
    finally:
        thread.join(5)


def test_an_unanswered_park_gives_up_and_joins_nothing(
    devices: tuple[relay.RelayServer, relay.RelayServer, str, int],  # noqa: F811
) -> None:
    """``JoinParkUnanswered`` is the design's exit ``3``: a human was required, and
    nobody came. The window is ``min(remaining, 180)``, so a short invite is what makes
    this test seconds rather than minutes.
    """
    server_a, server_b, host, port = devices
    record, minted = _minted(server_a, ttl_s=2.0)

    with pytest.raises(types.JoinParkUnanswered) as refused:
        _park(server_b, host=host, port=port, minted=minted)
    assert refused.value.code == "pairing_unanswered"
    assert "read back" in refused.value.sentence or "confirm" in refused.value.sentence
    rows = store.pending_joins(server_b.root)
    assert rows and rows[-1].status == "unanswered"
    assert rows[-1].error_code == "pairing_unanswered"
    assert store.list_networks(server_b.root) == []
    assert len(store.load(record.network_id, server_a.root).active_members()) == 1


def test_the_join_exit_code_three_reaches_the_caller(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The exception-to-exit-code mapping, asserted on the dispatcher itself: ``3`` for
    a missing human, ``1`` for everything else, and the same ``--json`` body either way.
    """

    def _unanswered(args: Namespace) -> int:
        del args
        raise types.JoinParkUnanswered("pairing_unanswered", "nobody answered in time")

    def _refused(args: Namespace) -> int:
        del args
        raise types.MeshRefusal("declined", "this device declined the pairing")

    monkeypatch.setitem(net_cli._HANDLERS, "join", _unanswered)  # noqa: SLF001
    assert net_cli.main(Namespace(network_command="join", json=True)) == 3
    body = json.loads(capsys.readouterr().out)
    assert body == {
        "ok": False,
        "code": "pairing_unanswered",
        "message": "nobody answered in time",
    }

    monkeypatch.setitem(net_cli._HANDLERS, "join", _refused)  # noqa: SLF001
    assert net_cli.main(Namespace(network_command="join", json=True)) == 1


def test_the_cli_parses_the_two_phases_as_written(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The flags exist on the REAL parser, so the guide's command lines are commands."""
    import argparse

    parser = argparse.ArgumentParser(prog="lop")
    subparsers = parser.add_subparsers(dest="subcommand")
    net_cli.add_parser(subparsers)
    park = parser.parse_args(["network", "join", "@invite.token", "--park", "--json"])
    assert park.park is True and park.confirm == "" and park.token == "@invite.token"
    answer = parser.parse_args(["network", "join", "--confirm", "481926", "--json"])
    assert answer.confirm == "481926" and answer.park is False
    assert answer.token == ""


def test_the_tool_parks_a_pairing_and_confirms_it_only_with_the_users_code(
    devices: tuple[relay.RelayServer, relay.RelayServer, str, int],  # noqa: F811
) -> None:
    """The agent path end to end, through the tool, with the REAL CLI as a subprocess.

    This is the transcript the brief asks for: a tool call starts the pairing and hands
    back the code, nothing is joined, and a SECOND tool call carrying the code the user
    read back finishes it. The pin is in the middle of it — the park's own result is
    not a confirmation, and there is no field through which it could become one.
    """
    server_a, server_b, host, port = devices
    record, minted = _minted(server_a)

    def _call(args: dict[str, Any]) -> Any:
        async def _run() -> Any:
            return await net_tool.execute_network("call-1", args, None, None, ToolContext(cwd="."))

        return asyncio.run(_run())

    started = _call({"action": "join", "token": minted.token})
    assert not started.is_error, _digest(started)
    digest = _digest(started)
    row = _wait_for_parked(server_b.root)
    # The code the agent is told to show the user is THIS device's derivation, rendered
    # the way a person reads it.
    assert row.sas in digest.replace(" ", "")
    assert f"{row.sas[:3]} {row.sas[3:]}" in digest
    assert "pairing started" in digest
    assert "nothing is joined until the person confirms" in digest

    # A PARKED PAIRING HAS SENT NOTHING, and the ceremony is held by a REAL process —
    # the one the second call has to reach. The inviter's own parked question is the
    # sharpest form of "nothing was sent": the relay only parks one when a
    # transcription has ARRIVED, so an empty queue is proof that the frame carrying the
    # human step is still unsent.
    assert store.list_networks(server_b.root) == []
    assert len(store.load(record.network_id, server_a.root).active_members()) == 1
    assert server_a._ctl_pair_pending({}) == []  # noqa: SLF001 — the CLI's own control op
    assert row.pid != os.getpid()
    assert net_cli._process_alive(row.pid)  # noqa: SLF001

    answered: dict[str, Any] = {}
    answer_thread = threading.Thread(
        target=lambda: answered.update(_answer_confirmation(server_a) or {}), daemon=True
    )
    answer_thread.start()
    try:
        finished = _call({"action": "join", "confirm": row.sas})
    finally:
        answer_thread.join(20)

    assert not finished.is_error, _digest(finished)
    assert answered, "the inviter's person never saw the parked question"
    assert finished.details is not None
    assert "joined" in _digest(finished) and record.network_id in _digest(finished)
    # The record goes with the ceremony, and that is the whole of what is asserted about
    # its END here. The parked process's EXIT is deliberately not: the parker is our own
    # child, and once its stdout pipe is gone it can sit as a zombie until we reap it,
    # where a liveness probe SUCCEEDS for a corpse — so an "it is gone" assertion is a
    # statement about this process's reaping, not about the product, and it measured the
    # difference on CI (passed on macOS, failed on Linux for exactly that reason). The
    # two directions that ARE product properties are asserted above, where the holder is
    # alive while parked (L172) and a second park is refused as `pairing_already_parked`
    # (L187).
    assert store.pending_joins(server_b.root) == []
    assert [item.network_id for item in store.list_networks(server_b.root)] == [record.network_id]
    # Which is what the agent's own verification step then reports.
    verified = _call({"action": "ls"})
    assert not verified.is_error
    assert record.name in _digest(verified)


def _digest(result: Any) -> str:
    return "\n".join(part.text for part in result.content)
