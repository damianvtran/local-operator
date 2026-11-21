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
from local_operator.network.credentials import placement as placement_mod
from local_operator.network.credentials.types import BROKER_CAPABILITY
from tests.unit.network.test_pair_offer import _seed
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
        nonce = row["nonce"]
        for frame in (
            {"invite_id": invite_id, "decision": "admit", "matched": True},
            {
                "invite_id": invite_id,
                "decision": "admit",
                "code": row["sas"],
                "nonce": nonce,
                "shares": [],
            },
            {"invite_id": invite_id, "decision": "decline", "code": "", "nonce": nonce},
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
            {"invite_id": invite_id, "decision": "decline", "code": "", "nonce": stage_two["nonce"]}
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
            {
                "invite_id": invite_id,
                "decision": "admit",
                "code": wrong,
                "matched": True,
                "nonce": row["nonce"],
                "shares": [],
            }
        )
        assert reply["status"] == "sas_mismatch" and reply["attempts_left"] == 2
        assert store.pair_decision(invite_id, server_a.root) is None, "a mismatch decided"
        # Survives a reload: the count is on the invite row.
        reloaded = store.load(record.network_id, server_a.root).invite(invite_id)
        assert reloaded is not None and reloaded.attempts == 1
        assert store.pending_pairing(invite_id, server_a.root) is not None, "card closed"
        assert "result" not in out and "error" not in out, "the socket was ended"
        reply = server_a._ctl_pair_confirm(  # noqa: SLF001
            {
                "invite_id": invite_id,
                "decision": "admit",
                "code": right,
                "nonce": row["nonce"],
                "shares": [],
            }
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
        frame = {
            "invite_id": row["invite_id"],
            "decision": "admit",
            "code": wrong,
            "nonce": row["nonce"],
            "shares": [],
        }
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
            {
                "invite_id": row["invite_id"],
                "decision": "admit",
                "code": row["sas"],
                "nonce": row["nonce"],
                "shares": [],
            }
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


# ---------------------------------------------------------------------------
# Round-1 remediation (security S-R1-1..4, code M2): each cell FAILS on 7e88f31c.
# ---------------------------------------------------------------------------


def _card(row: dict[str, Any], **extra: Any) -> dict[str, Any]:
    """The desktop route's code-carrying frame for ``row`` (the card's answer)."""
    frame = {
        "invite_id": row["invite_id"],
        "decision": "admit",
        "code": row["sas"],
        "nonce": row["nonce"],
    }
    frame.update(extra)
    return frame


def _gate_the_decision_poll(monkeypatch: pytest.MonkeyPatch) -> threading.Event:
    """Hold the listener just before it starts polling for a decision.

    The latch race is "an answer lands between the exhausting decline and the
    listener's next 0.2 s poll"; holding the poll makes that gap as wide as the cell
    needs instead of hoping a thread schedule reproduces it.
    """
    gate = threading.Event()
    real = relay.RelayServer._await_pairing_decision  # noqa: SLF001

    def gated(self: relay.RelayServer, pending: types.PendingPairing, window: float) -> Any:
        gate.wait(60)
        return real(self, pending, window)

    monkeypatch.setattr(relay.RelayServer, "_await_pairing_decision", gated)
    return gate


def test_a_card_admit_must_name_its_shares_and_an_empty_list_grants_nothing(
    devices: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    """S-R1-1: a card answer's selection is EXPLICIT. Absent is refused (it used to
    fall through to "every offered credential"); ``[]`` admits with zero grants and
    no broker capability, with a real credential on offer."""
    server_a, server_b, host, port = devices
    _seed(server_a.root, "openai", {"refresh": "r", "access": "a", "email": "damian@example.com"})
    record = _init_network(server_a)
    token, envelope = _mint_invite(server_a, record)
    release, _typed = _hold_the_joiner(monkeypatch)
    release.set()
    thread, out = _join_in_thread(server_b, host=host, port=port, token=token, envelope=envelope)
    try:
        row = _wait_row(server_a, types.PAIR_STAGE_AWAITING_CONFIRMATION)
        assert [item["key"] for item in row["offer"]] == ["openai"], "the premise: an offer"
        with pytest.raises(types.MeshRefusal) as excinfo:
            server_a._ctl_pair_confirm(_card(row))  # noqa: SLF001 — no ``shares`` key
        assert excinfo.value.code == "bad_request"
        assert store.pair_decision(row["invite_id"], server_a.root) is None
        reply = server_a._ctl_pair_confirm(_card(row, shares=[]))  # noqa: SLF001
        assert reply["status"] == "sent" and reply["shares"] == []
    finally:
        thread.join(30)
    assert "error" not in out, out.get("error")
    member = store.load(record.network_id, server_a.root).member(server_b.identity.device_id)
    assert member is not None and member.active
    assert BROKER_CAPABILITY not in member.capabilities
    document = placement_mod.PlacementDocument.load(
        record.network_id, server_a.root, self_device=server_a.identity.device_id
    )
    assert document.entry("openai") is None, "an empty selection served a credential"


def test_after_the_budget_is_spent_no_later_answer_is_recorded_on_any_ingress(
    devices: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    """S-R1-2 / M1: the exhausting decline is TERMINAL. A correct code (card) or a
    legacy ``matched`` y/N arriving before the listener's next poll is refused and
    overwrites nothing; the invite is consumed and nobody is admitted."""
    server_a, server_b, host, port = devices
    record = _init_network(server_a)
    token, envelope = _mint_invite(server_a, record)
    release, _typed = _hold_the_joiner(monkeypatch)
    release.set()
    gate = _gate_the_decision_poll(monkeypatch)
    thread, out = _join_in_thread(server_b, host=host, port=port, token=token, envelope=envelope)
    try:
        row = _wait_row(server_a, types.PAIR_STAGE_AWAITING_CONFIRMATION)
        invite_id = row["invite_id"]
        wrong = "000000" if row["sas"] != "000000" else "111111"
        statuses = [
            server_a._ctl_pair_confirm(_card(row, code=wrong, shares=[]))["status"]  # noqa: SLF001
            for _ in range(3)
        ]
        assert statuses == ["sas_mismatch", "sas_mismatch", "refused"]
        for late in (
            _card(row, shares=[]),
            {"invite_id": invite_id, "decision": "admit", "matched": True},
        ):
            with pytest.raises(types.MeshRefusal) as excinfo:
                server_a._ctl_pair_confirm(late)  # noqa: SLF001
            # S-R2-2: a SECOND answer gets the outcome-neutral code; the decision on
            # disk could have been an admit.
            assert excinfo.value.code == "pairing_answered", late
        decided = store.pair_decision(invite_id, server_a.root)
        assert decided is not None and decided.decision == "decline"
        assert decided.reason == "sas_mismatch", "the exhausting decline was overwritten"
    finally:
        gate.set()
        thread.join(30)
    assert isinstance(out.get("error"), types.MeshRefusal)
    after = store.load(record.network_id, server_a.root)
    assert after.member(server_b.identity.device_id) is None
    assert after.invites[0].state == "consumed"


def test_cleanup_after_a_re_park_leaves_the_new_ceremonys_card_and_decision(
    devices: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    """S-R1-3: an ending ceremony removes only ITS records. A second ceremony that
    parked the same invite key (the retryable-refusal window) keeps its card and its
    decision through the first one's cleanup."""
    server_a, server_b, host, port = devices
    record = _init_network(server_a)
    token, envelope = _mint_invite(server_a, record)
    release, _typed = _hold_the_joiner(monkeypatch)
    release.set()
    successor: dict[str, Any] = {}

    def answer_then_be_replaced(
        self: relay.RelayServer, pending: types.PendingPairing, window: float
    ) -> types.PairDecision:
        del window
        # Another ceremony parks the same invite key and gets a decision before this
        # one's cleanup runs. Built via ``from_json`` so the cell runs (and fails for
        # the right reason) on a build without the ``ceremony`` field.
        newer = dict(pending.to_json(), nonce="ee" * 16, ceremony="successor")
        store.save_pending_pairing(types.PendingPairing.from_json(newer), self.root)
        store.save_pair_decision(
            types.PairDecision.from_json(
                {
                    "invite_id": pending.invite_id,
                    "decision": "decline",
                    "reason": "declined",
                    "ceremony": "successor",
                }
            ),
            self.root,
        )
        successor["invite_id"] = pending.invite_id
        return types.PairDecision(invite_id=pending.invite_id, decision="decline")

    monkeypatch.setattr(relay.RelayServer, "_await_pairing_decision", answer_then_be_replaced)
    thread, out = _join_in_thread(server_b, host=host, port=port, token=token, envelope=envelope)
    thread.join(30)
    assert isinstance(out.get("error"), types.MeshRefusal)
    deadline = time.time() + 10
    while time.time() < deadline and server_a._pair_parked:  # noqa: SLF001
        time.sleep(0.05)
    assert server_a._pair_parked == 0  # noqa: SLF001 — both finallys have run
    invite_id = successor["invite_id"]
    card = store.pending_pairing(invite_id, server_a.root)
    assert card is not None and card.nonce == "ee" * 16, "the new ceremony's card was removed"
    decided = store.pair_decision(invite_id, server_a.root)
    assert decided is not None, "the new ceremony's decision was removed"


def test_both_inviter_stages_share_one_absolute_deadline(
    devices: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    """S-R1-4: stage 2 cannot outlive stage 1. The deadline is fixed once, when the
    ceremony starts; the stage-2 flip used to start a fresh window. An invite that
    outlives the ten-minute ceiling is the case where the two differ (with less left
    than the ceiling, both windows end at the invite's own expiry)."""
    server_a, server_b, host, port = devices
    record = _init_network(server_a)
    state = store.load_secrets(record.network_id, server_a.root)
    minted = invite_mod.mint(record, state.secret, role="drive", ttl_s=3600.0)
    record.invites.append(minted.record)
    store.save(record, server_a.root)
    token, envelope = minted.token, minted.envelope
    release, _typed = _hold_the_joiner(monkeypatch)
    thread, out = _join_in_thread(server_b, host=host, port=port, token=token, envelope=envelope)
    try:
        stage_one = _wait_row(server_a, types.PAIR_STAGE_AWAITING_JOINER)
        time.sleep(1.5)  # the joiner's person takes a moment to type
        release.set()
        stage_two = _wait_row(server_a, types.PAIR_STAGE_AWAITING_CONFIRMATION)
        assert stage_two["expires_at"] == pytest.approx(stage_one["expires_at"], abs=0.01)
        server_a._ctl_pair_confirm(_card(stage_two, decision="decline", code=""))  # noqa: SLF001
    finally:
        release.set()
        thread.join(30)
    assert isinstance(out.get("error"), types.MeshRefusal)


def test_four_real_ceremonies_fill_the_cap_and_a_fifth_is_busy(
    devices: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    """M2: the cap counts REAL parked ceremonies (the counter is never set by hand):
    four joins held in stage 1 take four slots, the fifth is ``pairing_busy`` and
    spends nothing, and every slot frees when the four end."""
    server_a, server_b, host, port = devices
    record = _init_network(server_a)
    minted = [
        _mint_invite(server_a, store.load(record.network_id, server_a.root)) for _ in range(5)
    ]
    release, _typed = _hold_the_joiner(monkeypatch)
    joins = [
        _join_in_thread(server_b, host=host, port=port, token=token, envelope=envelope)
        for token, envelope in minted[:4]
    ]
    try:
        deadline = time.time() + 20
        while time.time() < deadline and server_a._pair_parked < 4:  # noqa: SLF001
            time.sleep(0.05)
        assert server_a._pair_parked == relay.PAIR_PARKED_CEREMONY_CAP  # noqa: SLF001
        token, envelope = minted[4]
        with pytest.raises(types.MeshRefusal) as excinfo:
            _join(server_b, host=host, port=port, token=token, envelope=envelope)
        assert excinfo.value.code == "pairing_busy"
        fifth = store.load(record.network_id, server_a.root).invite(envelope.invite_id)
        assert fifth is not None and fifth.state == "minted" and fifth.attempts == 0
    finally:
        # End the four: let the joiners type, then decline each stage-2 card.
        release.set()
        declined: set[str] = set()
        deadline = time.time() + 30
        while time.time() < deadline and any(thread.is_alive() for thread, _out in joins):
            for row in server_a._ctl_pair_pending({}):  # noqa: SLF001 — stage 2 only
                if row["invite_id"] in declined:
                    continue
                server_a._ctl_pair_confirm(_card(row, decision="decline", code=""))  # noqa: SLF001
                declined.add(row["invite_id"])
            time.sleep(0.05)
        for thread, _out in joins:
            thread.join(5)
    deadline = time.time() + 10
    while time.time() < deadline and server_a._pair_parked:  # noqa: SLF001
        time.sleep(0.05)
    assert server_a._pair_parked == 0  # noqa: SLF001


def test_a_legacy_decline_naming_sas_mismatch_consumes_as_before(
    devices: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    """N2: the listener believes ``sas_mismatch`` only when the budget is spent. A
    legacy y/N frame that copies ``reason: sas_mismatch`` is a decline: it consumes
    the invite and spends no attempt, as it did before the card path existed."""
    server_a, server_b, host, port = devices
    record = _init_network(server_a)
    token, envelope = _mint_invite(server_a, record)
    release, _typed = _hold_the_joiner(monkeypatch)
    release.set()
    thread, out = _join_in_thread(server_b, host=host, port=port, token=token, envelope=envelope)
    try:
        row = _wait_row(server_a, types.PAIR_STAGE_AWAITING_CONFIRMATION)
        server_a._ctl_pair_confirm(  # noqa: SLF001
            {"invite_id": row["invite_id"], "decision": "decline", "reason": "sas_mismatch"}
        )
    finally:
        thread.join(30)
    assert isinstance(out.get("error"), types.MeshRefusal)
    invite = store.load(record.network_id, server_a.root).invite(row["invite_id"])
    assert invite is not None and invite.state == "consumed", "a decline was retried"
    assert invite.attempts == 0, "a decline spent an attempt"


# ---------------------------------------------------------------------------
# Security review round 2, S-R2-1 (and agent review round 2 m1/m2/n1): onboarding's
# pre-approval is its own record, bound to the invite and the joiner, consumed once
# at the stage-2 flip; no unstamped decision ever answers a ceremony.
# ---------------------------------------------------------------------------


def _invite_of(server: relay.RelayServer, record: types.NetworkRecord) -> str:
    return store.load(record.network_id, server.root).invites[-1].invite_id


def _pre_approve(
    server: relay.RelayServer, record: types.NetworkRecord, invite_id: str, joiner: str
) -> None:
    """What ``onboard.OnboardRun._write_pre_approval`` writes for an approved run."""
    store.save_pre_approval(
        types.PairPreApproval(
            invite_id=invite_id,
            network_id=record.network_id,
            joiner_device_id=joiner,
            answered_by="approval:ap_test0001",
            expires_at=time.time() + 600.0,
        ),
        server.root,
    )


def _codes_in_order(monkeypatch: pytest.MonkeyPatch, *answers: str | None) -> threading.Event:
    """The joiner's person across successive joins: ``None`` types the right digits,
    ``"WRONG"`` six digits that differ from them (a real, well-formed mismatch).

    The FIRST join's read blocks on the returned event, so a cell can act while that
    ceremony sits in stage 1; later reads answer at once.
    """
    first = threading.Event()
    queue = list(answers)
    calls: list[int] = []

    def read_code(args: Any, derived: str, fingerprint: str) -> str:
        del args, fingerprint
        calls.append(1)
        if len(calls) == 1:
            first.wait(60)
        answer = queue.pop(0) if queue else None
        if answer == "WRONG":
            return _wrong(derived)
        return derived if answer is None else answer

    monkeypatch.setattr(net_cli, "_read_code", read_code)
    return first


def _wrong(code: str) -> str:
    return "000000" if code != "000000" else "111111"


def test_a_previous_ceremonys_unstamped_admit_is_not_a_pre_answer_for_the_next(
    devices: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Agent review round 2, m1 (the S-R2-1 repro). Ceremony A parks; an unstamped
    admit lands during A's stage 1; A is refused retryably (a wire mismatch) and B
    re-parks the same invite BEFORE A's cleanup runs. B must not read that admit:
    the old "unstamped and written before B started" rule kept it and admitted B
    with no person answering."""
    server_a, server_b, host, port = devices
    record = _init_network(server_a)
    token, envelope = _mint_invite(server_a, record)
    invite_id = _invite_of(server_a, record)
    real_clear = relay.RelayServer._clear_ceremony  # noqa: SLF001
    ceremony_a: list[str] = []
    deferred: list[tuple[Any, ...]] = []
    # Joiner A's thread returns as soon as it reads its refusal; the relay side of
    # ceremony A reaches ``_clear_ceremony`` on its own thread, possibly later on a
    # slow runner (merge-gate run 44451819486). Wait for the interception itself
    # rather than inferring it from ``thread_a.join()``.
    a_cleanup_intercepted = threading.Event()

    def defer_a(self: relay.RelayServer, network_id: str, invite: str, ceremony: str) -> None:
        if ceremony_a and ceremony == ceremony_a[0]:
            deferred.append((self, network_id, invite, ceremony))
            a_cleanup_intercepted.set()
            return
        real_clear(self, network_id, invite, ceremony)

    monkeypatch.setattr(relay.RelayServer, "_clear_ceremony", defer_a)
    release_a = _codes_in_order(monkeypatch, "WRONG", None)
    thread_a, out_a = _join_in_thread(
        server_b, host=host, port=port, token=token, envelope=envelope
    )
    row_a = _wait_row(server_a, types.PAIR_STAGE_AWAITING_JOINER)
    live_a = store.pending_pairing(invite_id, server_a.root)
    assert live_a is not None and live_a.ceremony
    ceremony_a.append(live_a.ceremony)
    store.save_pair_decision(
        types.PairDecision(invite_id=invite_id, decision="admit", matched=True), server_a.root
    )
    # A's person types a wrong code: a retryable wire mismatch.
    release_a.set()
    thread_a.join(30)
    assert isinstance(out_a.get("error"), types.MeshRefusal), out_a
    assert row_a["invite_id"] == invite_id
    assert a_cleanup_intercepted.wait(30), "the premise: A's cleanup was intercepted before B parks"
    assert deferred, "the premise: A's cleanup has not run when B parks"
    thread_b, out_b = _join_in_thread(
        server_b, host=host, port=port, token=token, envelope=envelope
    )
    try:
        # A kept admit is read on B's FIRST poll, so B's stage-2 card can vanish before
        # a reader sees it: watch for the admission itself as well as the card.
        row_b: dict[str, Any] | None = None
        deadline = time.time() + 20.0
        while row_b is None and time.time() < deadline:
            admitted = store.load(record.network_id, server_a.root).member(
                server_b.identity.device_id
            )
            assert admitted is None, "B read A's unstamped admit"
            rows = server_a._ctl_pair_pending({"include_stages": True})  # noqa: SLF001
            row_b = next(
                (r for r in rows if r.get("stage") == types.PAIR_STAGE_AWAITING_CONFIRMATION),
                None,
            )
            time.sleep(0.05)
        assert row_b is not None, "B never reached its stage-2 card"
        time.sleep(0.6)  # three poll ticks: a kept admit would be read here
        after = store.load(record.network_id, server_a.root)
        assert after.member(server_b.identity.device_id) is None, "B read A's unstamped admit"
        server_a._ctl_pair_confirm(  # noqa: SLF001
            {"invite_id": invite_id, "decision": "decline", "code": "", "nonce": row_b["nonce"]}
        )
    finally:
        thread_b.join(30)
        for args in deferred:
            real_clear(*args)
    assert isinstance(out_b.get("error"), types.MeshRefusal)
    assert store.load(record.network_id, server_a.root).member(server_b.identity.device_id) is None


def test_an_automated_retry_after_a_stage_one_failure_still_uses_the_pre_approval(
    devices: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The reverse case of m1: a ceremony that fails at stage 1 must leave onboarding's
    pre-approval for the invite's retry (it used to delete the pre-answer, and the
    retry then waited out its window and admitted nothing). The retry is admitted with
    no card answer, and the pre-approval is consumed exactly once."""
    server_a, server_b, host, port = devices
    record = _init_network(server_a)
    token, envelope = _mint_invite(server_a, record)
    invite_id = _invite_of(server_a, record)
    _pre_approve(server_a, record, invite_id, server_b.identity.device_id)
    release = _codes_in_order(monkeypatch, "WRONG", None)
    release.set()
    thread, out = _join_in_thread(server_b, host=host, port=port, token=token, envelope=envelope)
    thread.join(30)
    assert isinstance(out.get("error"), types.MeshRefusal), "the premise: a stage-1 failure"
    assert store.pre_approval(invite_id, server_a.root) is not None, "stage-1 cleanup ate it"
    thread, out = _join_in_thread(server_b, host=host, port=port, token=token, envelope=envelope)
    thread.join(30)
    assert "error" not in out, out.get("error")
    member = store.load(record.network_id, server_a.root).member(server_b.identity.device_id)
    assert member is not None and member.active
    assert store.pre_approval(invite_id, server_a.root) is None, "a pre-approval is single use"
    confirmed = [
        row for row in server_a.audit.tail(limit=500) if row.get("event") == "pairing_confirmed"
    ]
    assert confirmed and confirmed[-1]["detail"]["answered_by"] == "approval:ap_test0001"


def test_the_flip_consumes_a_matching_pre_approval_once_as_a_stamped_decision(
    devices: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Single use, under the flip's lock: by the time stage 2 is visible the
    pre-approval file is gone and a decision stamped with THIS ceremony stands in its
    place; a generic decision left on disk beforehand was cleared, not kept."""
    server_a, server_b, host, port = devices
    record = _init_network(server_a)
    token, envelope = _mint_invite(server_a, record)
    invite_id = _invite_of(server_a, record)
    _pre_approve(server_a, record, invite_id, server_b.identity.device_id)
    store.save_pair_decision(
        types.PairDecision(invite_id=invite_id, decision="decline", reason="declined"),
        server_a.root,
    )
    release, _typed = _hold_the_joiner(monkeypatch)
    release.set()
    gate = _gate_the_decision_poll(monkeypatch)
    thread, out = _join_in_thread(server_b, host=host, port=port, token=token, envelope=envelope)
    try:
        _wait_row(server_a, types.PAIR_STAGE_AWAITING_CONFIRMATION)
        live = store.pending_pairing(invite_id, server_a.root)
        assert live is not None
        assert store.pre_approval(invite_id, server_a.root) is None, "not consumed at the flip"
        decided = store.pair_decision(invite_id, server_a.root)
        assert decided is not None and decided.decision == "admit"
        assert decided.ceremony == live.ceremony and decided.answered_by == "approval:ap_test0001"
        assert decided.shares == [], "an approval shares nothing"
    finally:
        gate.set()
        thread.join(30)
    assert "error" not in out, out.get("error")
    assert store.load(record.network_id, server_a.root).member(server_b.identity.device_id)


def test_a_pre_approval_for_another_joiner_is_refused_and_the_card_parks(
    devices: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A different joiner never matches: the pre-approval is left unconsumed, nothing
    is admitted, and the ceremony waits on an answerable stage-2 card (what a brand-new
    host with no pre-approval also gets)."""
    server_a, server_b, host, port = devices
    record = _init_network(server_a)
    token, envelope = _mint_invite(server_a, record)
    invite_id = _invite_of(server_a, record)
    _pre_approve(server_a, record, invite_id, "d_" + "c" * 32)
    release, _typed = _hold_the_joiner(monkeypatch)
    release.set()
    thread, out = _join_in_thread(server_b, host=host, port=port, token=token, envelope=envelope)
    try:
        row = _wait_row(server_a, types.PAIR_STAGE_AWAITING_CONFIRMATION)
        time.sleep(0.6)
        assert (
            store.load(record.network_id, server_a.root).member(server_b.identity.device_id) is None
        ), "another joiner's approval admitted this one"
        assert store.pair_decision(invite_id, server_a.root) is None
        kept = store.pre_approval(invite_id, server_a.root)
        assert kept is not None and kept.joiner_device_id == "d_" + "c" * 32
        assert server_a._ctl_pair_pending({}), "the card is not offered for an answer"
        server_a._ctl_pair_confirm(  # noqa: SLF001
            {"invite_id": invite_id, "decision": "decline", "code": "", "nonce": row["nonce"]}
        )
    finally:
        thread.join(30)
    assert isinstance(out.get("error"), types.MeshRefusal)


def test_only_a_decision_stamped_with_the_live_ceremony_answers_stage_two(
    devices: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The poll reads a decision only when it carries THIS ceremony's id: an unstamped
    out-of-band admit written during stage 2 is ignored, a stamped one is honoured."""
    server_a, server_b, host, port = devices
    record = _init_network(server_a)
    token, envelope = _mint_invite(server_a, record)
    invite_id = _invite_of(server_a, record)
    release, _typed = _hold_the_joiner(monkeypatch)
    release.set()
    thread, out = _join_in_thread(server_b, host=host, port=port, token=token, envelope=envelope)
    try:
        _wait_row(server_a, types.PAIR_STAGE_AWAITING_CONFIRMATION)
        live = store.pending_pairing(invite_id, server_a.root)
        assert live is not None and live.ceremony
        store.save_pair_decision(
            types.PairDecision(invite_id=invite_id, decision="admit", matched=True),
            server_a.root,
        )
        time.sleep(0.6)
        assert (
            store.load(record.network_id, server_a.root).member(server_b.identity.device_id) is None
        ), "an unstamped decision answered the ceremony"
        store.save_pair_decision(
            types.PairDecision(
                invite_id=invite_id, decision="admit", matched=True, ceremony=live.ceremony
            ),
            server_a.root,
        )
    finally:
        thread.join(30)
    assert "error" not in out, out.get("error")
    assert store.load(record.network_id, server_a.root).member(server_b.identity.device_id)


def test_a_code_frame_with_no_share_list_spends_no_attempt(
    devices: Devices, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Agent review round 2, n1: the share-list shape is judged BEFORE the code
    compare, so a malformed control-socket frame with a wrong code spends nothing."""
    server_a, server_b, host, port = devices
    record = _init_network(server_a)
    token, envelope = _mint_invite(server_a, record)
    invite_id = _invite_of(server_a, record)
    release, _typed = _hold_the_joiner(monkeypatch)
    release.set()
    thread, out = _join_in_thread(server_b, host=host, port=port, token=token, envelope=envelope)
    try:
        row = _wait_row(server_a, types.PAIR_STAGE_AWAITING_CONFIRMATION)
        with pytest.raises(types.MeshRefusal) as excinfo:
            server_a._ctl_pair_confirm(_card(row, code=_wrong(row["sas"])))  # noqa: SLF001
        assert excinfo.value.code == "bad_request"
        invite = store.load(record.network_id, server_a.root).invite(invite_id)
        assert invite is not None and invite.attempts == 0, "a malformed frame spent an attempt"
        server_a._ctl_pair_confirm(  # noqa: SLF001
            {"invite_id": invite_id, "decision": "decline", "code": "", "nonce": row["nonce"]}
        )
    finally:
        thread.join(30)
    assert isinstance(out.get("error"), types.MeshRefusal)


def test_a_pre_approval_matches_only_its_invite_network_and_named_joiner(tmp_path: Path) -> None:
    approval = types.PairPreApproval(
        invite_id="i_1",
        network_id="n_1",
        joiner_device_id="d_a",
        answered_by="approval:ap_x",
        expires_at=time.time() + 60,
    )
    assert approval.matches(invite_id="i_1", network_id="n_1", joiner_device_id="d_a")
    assert not approval.matches(invite_id="i_2", network_id="n_1", joiner_device_id="d_a")
    assert not approval.matches(invite_id="i_1", network_id="n_2", joiner_device_id="d_a")
    assert not approval.matches(invite_id="i_1", network_id="n_1", joiner_device_id="d_b")
    unbound = types.PairPreApproval.from_json(dict(approval.to_json(), joiner_device_id=""))
    assert not unbound.matches(invite_id="i_1", network_id="n_1", joiner_device_id="")
    # It expires with its invite, and an expired one is removed as it is read.
    store.save_pre_approval(
        types.PairPreApproval.from_json(dict(approval.to_json(), expires_at=time.time() - 1)),
        tmp_path,
    )
    assert store.pre_approval("i_1", tmp_path) is None
    assert not store.pre_approval_path("i_1", tmp_path).exists()


def test_a_second_answer_is_refused_with_outcome_neutral_copy() -> None:
    """S-R2-2: the earlier answer may have admitted the device, so the keyed sentence
    for ``pairing_answered`` must not say "nothing was approved"."""
    from local_operator.network import pairing_copy

    sentence = pairing_copy.refusal("pairing_answered")
    assert sentence == "This pairing is no longer open. This answer wasn't recorded."
    assert "approved" not in sentence
