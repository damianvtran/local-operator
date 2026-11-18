"""Mesh pairing from the app's Mesh approvals — the relay half (design §3.2, §3.5).

Operator request (Damian): "Make sure pairing does NOT require running terminal
commands, ideally it should be through the UI mesh approvals." These cells pin the
relay mechanics that make a card answer safe, over REAL loopback relays and the CLI's
own joining driver (``test_relay_e2e`` helpers); only the two people are simulated:

* the person window is ``min(remaining, 600 s)`` (was 180 s, which lapsed on the
  operator);
* a stage-1 (``awaiting_joiner``) record exists before the joiner transcribes, is
  hidden from legacy callers, refuses every answer, and a decision written during
  stage 1 is cleared BEFORE the stage-2 flip (the stale-decision race);
* a code-carrying ``net_pair_confirm`` is compared BY THE RELAY (caller ``matched``
  ignored); a mismatch spends exactly one of the invite's three attempts, survives a
  reload, writes no decision and keeps the socket open; the exhausting one consumes;
* at most four inviter ceremonies park per relay, the fifth is ``pairing_busy`` and
  spends nothing;
* expiry clears the card, frees the slot and changes no membership;
* ``SO_KEEPALIVE`` is set on both pair sockets.
"""

from __future__ import annotations

import socket
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from local_operator.network import cli as net_cli
from local_operator.network import handshake as hs_mod
from local_operator.network import invite as invite_mod
from local_operator.network import relay, store, types
from tests.unit.network.test_relay_e2e import _init_network, _join, _mint_invite
from tests.unit.network.test_relay_e2e import devices as _devices_fixture

# Re-exported under pytest's lookup name: the shared two-relay loopback fixture.
devices = _devices_fixture

Devices = tuple[relay.RelayServer, relay.RelayServer, str, int]


def _hold_the_joiner(monkeypatch: pytest.MonkeyPatch) -> tuple[threading.Event, list[str]]:
    """The joiner's person, who has NOT typed yet: ``_read_code`` blocks on an event.

    While it blocks the inviter sits in stage 1. ``typed`` lets a cell choose the
    digits that will be typed when the event fires (``[]`` = the right ones).
    """
    release = threading.Event()
    typed: list[str] = []

    def held_read_code(args: Any, derived: str, fingerprint: str) -> str:
        del args, fingerprint
        release.wait(60)
        return typed[0] if typed else derived

    monkeypatch.setattr(net_cli, "_read_code", held_read_code)
    return release, typed


def _join_in_thread(
    server_b: relay.RelayServer, **kwargs: Any
) -> tuple[threading.Thread, dict[str, Any]]:
    out: dict[str, Any] = {}

    def run() -> None:
        try:
            out["result"] = _join(server_b, **kwargs)
        except BaseException as exc:  # noqa: BLE001 — asserted by the caller
            out["error"] = exc

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread, out


def _wait_row(server: relay.RelayServer, stage: str, timeout: float = 20.0) -> dict[str, Any]:
    deadline = time.time() + timeout
    while time.time() < deadline:
        rows = server._ctl_pair_pending({"include_stages": True})  # noqa: SLF001
        for row in rows:
            if row.get("stage") == stage:
                return row
        time.sleep(0.05)
    raise AssertionError(f"no {stage} row appeared")


def test_the_person_window_is_the_invite_life_up_to_ten_minutes() -> None:
    assert hs_mod.PAIR_PARK_CEILING_S == 600.0
    assert hs_mod.pair_timeout_seconds(600.0) == 600.0
    assert hs_mod.pair_timeout_seconds(3600.0) == 600.0
    assert hs_mod.pair_timeout_seconds(42.0) == 42.0
    assert not hasattr(hs_mod, "PAIR_CONFIRM_TIMEOUT_S"), "one owner: the old name is gone"


def test_keepalive_is_set_on_a_real_socket() -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        assert hs_mod.set_pair_keepalive(sock) is True
        assert sock.getsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE) == 1
        if hasattr(socket, "TCP_KEEPIDLE"):
            assert sock.getsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE) == 60
        if hasattr(socket, "TCP_KEEPINTVL"):
            assert sock.getsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL) == 20
    finally:
        sock.close()

    class Refuses:
        def setsockopt(self, *_args: Any) -> None:
            raise OSError("no")

    assert hs_mod.set_pair_keepalive(Refuses()) is False, "a refusing socket never raises"


def test_stage_one_card_is_hidden_from_legacy_callers_refuses_answers_and_its_decision_is_cleared(
    devices: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    """THE STALE-DECISION RACE. A decision written while the joiner has not typed must
    never be read by the stage-2 poll; the flip clears it under the record lock."""
    server_a, server_b, host, port = devices
    record = _init_network(server_a)
    token, envelope = _mint_invite(server_a, record)
    release, _typed = _hold_the_joiner(monkeypatch)
    keepalive: list[str] = []
    real = hs_mod.set_pair_keepalive

    def spy_relay(sock: Any) -> bool:
        keepalive.append("inviter")
        return real(sock)

    def spy_cli(sock: Any) -> bool:
        keepalive.append("joiner")
        return real(sock)

    monkeypatch.setattr(relay, "set_pair_keepalive", spy_relay)
    monkeypatch.setattr(hs_mod, "set_pair_keepalive", spy_cli)
    thread, out = _join_in_thread(server_b, host=host, port=port, token=token, envelope=envelope)
    try:
        row = _wait_row(server_a, types.PAIR_STAGE_AWAITING_JOINER)
        invite_id = row["invite_id"]
        assert row["nonce"] and row["sas"] and row["transcribed"] == ""
        assert (
            server_a._ctl_pair_pending({}) == []
        ), "a legacy CLI must not see stage 1"  # noqa: SLF001
        for frame in (
            {"invite_id": invite_id, "decision": "admit", "matched": True},
            {"invite_id": invite_id, "decision": "admit", "code": row["sas"]},
            {"invite_id": invite_id, "decision": "decline", "code": ""},
        ):
            with pytest.raises(types.MeshRefusal) as excinfo:
                server_a._ctl_pair_confirm(frame)  # noqa: SLF001
            assert excinfo.value.code == "pairing_not_ready", frame
        assert store.pair_decision(invite_id, server_a.root) is None
        # An out-of-band ADMIT written during stage 1 (the race the guard closes).
        store.save_pair_decision(
            types.PairDecision(invite_id=invite_id, decision="admit", matched=True),
            server_a.root,
        )
        stage_one_nonce = row["nonce"]
        release.set()
        stage_two = _wait_row(server_a, types.PAIR_STAGE_AWAITING_CONFIRMATION)
        assert stage_two["nonce"] != stage_one_nonce, "the flip must mint a fresh nonce"
        time.sleep(0.6)  # three poll ticks: a surviving stale admit would be read here
        assert store.pending_pairing(invite_id, server_a.root) is not None
        after = store.load(record.network_id, server_a.root)
        assert after.member(server_b.identity.device_id) is None, "stale admit was read"
        server_a._ctl_pair_confirm(  # noqa: SLF001
            {"invite_id": invite_id, "decision": "decline", "code": ""}
        )
    finally:
        release.set()
        thread.join(30)
    assert isinstance(out.get("error"), types.MeshRefusal)
    assert {"inviter", "joiner"} <= set(keepalive), keepalive


def test_a_card_code_is_compared_by_the_relay_and_a_mismatch_keeps_the_ceremony_open(
    devices: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    server_a, server_b, host, port = devices
    record = _init_network(server_a)
    token, envelope = _mint_invite(server_a, record)
    release, _typed = _hold_the_joiner(monkeypatch)
    release.set()
    thread, out = _join_in_thread(server_b, host=host, port=port, token=token, envelope=envelope)
    try:
        row = _wait_row(server_a, types.PAIR_STAGE_AWAITING_CONFIRMATION)
        invite_id, right = row["invite_id"], row["sas"]
        wrong = "000000" if right != "000000" else "111111"
        # A caller-asserted ``matched: true`` is IGNORED once a code is present.
        reply = server_a._ctl_pair_confirm(  # noqa: SLF001
            {"invite_id": invite_id, "decision": "admit", "code": wrong, "matched": True}
        )
        assert reply["status"] == "sas_mismatch" and reply["attempts_left"] == 2
        assert store.pair_decision(invite_id, server_a.root) is None, "a mismatch decided"
        # Survives a reload: the count is on the invite row.
        reloaded = store.load(record.network_id, server_a.root).invite(invite_id)
        assert reloaded is not None and reloaded.attempts == 1
        assert store.pending_pairing(invite_id, server_a.root) is not None, "card closed"
        assert "result" not in out and "error" not in out, "the socket was ended"
        reply = server_a._ctl_pair_confirm(  # noqa: SLF001
            {"invite_id": invite_id, "decision": "admit", "code": right, "nonce": row["nonce"]}
        )
        assert reply["status"] == "sent" and reply["decision"] == "admit"
    finally:
        thread.join(30)
    assert "error" not in out, out.get("error")
    after = store.load(record.network_id, server_a.root)
    assert after.member(server_b.identity.device_id) is not None
    assert after.invites[0].state == "consumed"


def test_the_third_card_mismatch_spends_the_budget_and_consumes_the_invite(
    devices: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    server_a, server_b, host, port = devices
    record = _init_network(server_a)
    token, envelope = _mint_invite(server_a, record)
    release, _typed = _hold_the_joiner(monkeypatch)
    release.set()
    thread, out = _join_in_thread(server_b, host=host, port=port, token=token, envelope=envelope)
    try:
        row = _wait_row(server_a, types.PAIR_STAGE_AWAITING_CONFIRMATION)
        wrong = "000000" if row["sas"] != "000000" else "111111"
        frame = {"invite_id": row["invite_id"], "decision": "admit", "code": wrong}
        statuses = [server_a._ctl_pair_confirm(dict(frame)) for _ in range(3)]  # noqa: SLF001
        assert [s["status"] for s in statuses] == ["sas_mismatch", "sas_mismatch", "refused"]
        assert [s["attempts_left"] for s in statuses] == [2, 1, 0]
    finally:
        thread.join(30)
    assert isinstance(out.get("error"), types.MeshRefusal)
    after = store.load(record.network_id, server_a.root)
    assert after.member(server_b.identity.device_id) is None
    assert after.invites[0].state == "consumed"
    assert after.invites[0].attempts == 3, "the exhausting refusal must not charge twice"


def test_a_fifth_parked_ceremony_is_busy_and_spends_nothing(
    devices: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    server_a, server_b, host, port = devices
    record = _init_network(server_a)
    token, envelope = _mint_invite(server_a, record)
    release, _typed = _hold_the_joiner(monkeypatch)
    release.set()
    assert relay.PAIR_PARKED_CEREMONY_CAP == 4
    server_a._pair_parked = relay.PAIR_PARKED_CEREMONY_CAP  # noqa: SLF001 — four parked
    with pytest.raises(types.MeshRefusal) as excinfo:
        _join(server_b, host=host, port=port, token=token, envelope=envelope)
    assert excinfo.value.code == "pairing_busy"
    after = store.load(record.network_id, server_a.root)
    assert after.invites[0].state == "minted", "a busy refusal consumed the invite"
    assert after.invites[0].attempts == 0, "a busy refusal spent an attempt"
    assert store.pending_pairings(server_a.root) == []
    assert server_a._pair_parked == relay.PAIR_PARKED_CEREMONY_CAP  # noqa: SLF001

    # The SAME token works once a slot frees.
    server_a._pair_parked = 0  # noqa: SLF001
    thread, out = _join_in_thread(server_b, host=host, port=port, token=token, envelope=envelope)
    try:
        row = _wait_row(server_a, types.PAIR_STAGE_AWAITING_CONFIRMATION)
        server_a._ctl_pair_confirm(  # noqa: SLF001
            {"invite_id": row["invite_id"], "decision": "admit", "code": row["sas"]}
        )
    finally:
        thread.join(30)
    assert "error" not in out, out.get("error")
    # The listener thread finishes its post-admission work (link hand-off, peer
    # contact) after the joiner returns, so the slot frees a moment later.
    deadline = time.time() + 10
    while time.time() < deadline and server_a._pair_parked:  # noqa: SLF001
        time.sleep(0.05)
    assert server_a._pair_parked == 0, "the slot was not freed"  # noqa: SLF001


def test_expiry_clears_the_card_frees_the_slot_and_changes_no_membership(
    devices: Devices, monkeypatch: pytest.MonkeyPatch, root: Path
) -> None:
    server_a, server_b, host, port = devices
    record = _init_network(server_a)
    state = store.load_secrets(record.network_id, server_a.root)
    minted = invite_mod.mint(record, state.secret, role="drive", ttl_s=4.0)
    record.invites.append(minted.record)
    store.save(record, server_a.root)
    members_before = sorted(
        m.device_id for m in store.load(record.network_id, server_a.root).members
    )
    release, _typed = _hold_the_joiner(monkeypatch)  # the joiner never types in time
    thread, out = _join_in_thread(
        server_b, host=host, port=port, token=minted.token, envelope=minted.envelope
    )
    try:
        row = _wait_row(server_a, types.PAIR_STAGE_AWAITING_JOINER)
        assert row["expires_at"] - time.time() <= 4.5
        deadline = time.time() + 20
        while time.time() < deadline and (
            store.pending_pairing(row["invite_id"], server_a.root) is not None
            or server_a._pair_parked  # noqa: SLF001
        ):
            time.sleep(0.1)
        assert store.pending_pairing(row["invite_id"], server_a.root) is None, "card outlived"
        assert server_a._pair_parked == 0  # noqa: SLF001
    finally:
        release.set()
        thread.join(30)
    after = store.load(record.network_id, server_a.root)
    assert sorted(m.device_id for m in after.members) == members_before
    assert after.invites[0].attempts == 0, "a delay spent an attempt"
    # The joiner was not admitted: it holds no record of the network.
    assert not store.record_path(record.network_id, server_b.root).exists(), out
