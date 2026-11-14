"""An ending reply may settle only the local copy instance sent before its I/O.

The byte generation stays equal across local-only/default and a re-copy. Holding
an actual wipe reply until after that re-copy must not erase its pending receipt
or overwrite a newer present ACK. No signed withdrawal is bypassed by this rig.
"""

from __future__ import annotations

import ipaddress
import json
import socket
import threading
from pathlib import Path
from typing import Any

import pytest

from local_operator.network import store
from local_operator.network.credentials import sync
from local_operator.network.credentials.messages import render_copy_revoke_notice
from tests.unit.network import conftest as net
from tests.unit.network.test_copy_dispatch_removal import (
    _d,
    _deliver_without_ack,
    _HeldAcks,
)
from tests.unit.network.test_credentials_real_link import _lop_network
from tests.unit.network.test_relay_e2e import devices  # noqa: F401
from tests.unit.network.test_secret_copies import (  # noqa: F401
    KEY,
    SECRET_NAME,
    _member_has,
    secret_mesh,
)


@pytest.fixture(autouse=True)
def _loopback_only(monkeypatch: pytest.MonkeyPatch) -> None:
    connect = socket.socket.connect

    def checked(sock: socket.socket, address: Any) -> Any:
        if isinstance(address, tuple):
            assert (
                address[0] == "localhost" or ipaddress.ip_address(address[0]).is_loopback
            ), "wipe identity tests allow only loopback TCP"
        return connect(sock, address)

    monkeypatch.setattr(socket.socket, "connect", checked)


def test_no_copy_notice_does_not_deny_current_dispatch_tracking() -> None:
    lines = render_copy_revoke_notice("device-b", KEY, copied=False, wiped=False)
    assert lines == [
        "no copy of 'STORE_COPY_TOKEN' is recorded on device-b, so there is no wipe to send"
    ]


def test_loopback_guard_is_active() -> None:
    with socket.socket() as sock, pytest.raises(AssertionError, match="only loopback TCP"):
        sock.connect(("192.0.2.1", 443))


def _recopy_while_wipe_reply_is_held(
    mesh: Any, monkeypatch: pytest.MonkeyPatch, *, late: bool, lose_ack: bool
) -> dict[str, Any]:
    sync.credentials_sync_step(mesh.a, mesh.member)
    assert net.wait_for(
        lambda: sync.SyncState.load(mesh.network_id, mesh.a.root).ack_for(mesh.member, KEY)
        is not None
    )
    assert net.wait_for(lambda: mesh.member not in mesh.engine_a._inflight_members)
    old = sync.SyncState.load(mesh.network_id, mesh.a.root).ack_for(mesh.member, KEY)
    assert old is not None
    old_gen = old["gen"]
    # Isolate one ending, not the transport: drive the normal exchange explicitly
    # after the CLI changes policy, with no competing periodic owner exchange.
    monkeypatch.setattr(mesh.engine_a, "enqueue_owner_exchange", lambda device: "scheduled")
    reached = threading.Event()
    release = threading.Event()
    original = mesh.engine_a._record_wipe_reply
    errors: list[BaseException] = []

    def hold(*args: Any, **kwargs: Any) -> None:
        reached.set()
        assert release.wait(net.NET_WAIT_BUDGET_S), "the owned wipe reply was not released"
        original(*args, **kwargs)

    monkeypatch.setattr(mesh.engine_a, "_record_wipe_reply", hold)
    assert _lop_network("credential", "mark", SECRET_NAME, "local-only") == 0

    def exchange() -> None:
        try:
            mesh.engine_a._exchange_blocking(mesh.member)
        except BaseException as exc:
            errors.append(exc)

    worker = threading.Thread(target=exchange, name="owned-wipe-identity")
    worker.start()
    try:
        assert reached.wait(net.NET_WAIT_BUDGET_S), "the real wipe reply missed its barrier"
        assert not _member_has(mesh.b.root, SECRET_NAME), "the member must really have wiped"
        if not late:
            release.set()
            worker.join(net.NET_WAIT_BUDGET_S)
            assert not worker.is_alive()
        held = _HeldAcks(monkeypatch, mesh.b.root) if lose_ack else None
        assert _lop_network("credential", "mark", SECRET_NAME, "default") == 0
        result = mesh.engine_b.pull_now(KEY)
        assert _member_has(mesh.b.root, SECRET_NAME), result
        if held is not None:
            assert held.held.wait(net.NET_WAIT_BUDGET_S)
        before = sync.SyncState.load(mesh.network_id, mesh.a.root)
        current = before.generation(KEY)
        assert current is not None and current["gen"] == old_gen
        assert (before.dispatch_for(mesh.member, KEY) is not None) == lose_ack
        release.set()
        worker.join(net.NET_WAIT_BUDGET_S)
        assert not worker.is_alive() and not errors, errors
        state = sync.SyncState.load(mesh.network_id, mesh.a.root)
        ending = state.ack_for(mesh.member, KEY)
        outcome = {
            "late_reply": late,
            "lost_recopy_ack": lose_ack,
            "old_gen": old_gen,
            "recopy_gen": current["gen"],
            "pending_after": state.dispatch_for(mesh.member, KEY) is not None,
            "member_has_copy": _member_has(mesh.b.root, SECRET_NAME),
            "owner_wiped": bool(ending and ending.get("wiped")),
            "open_copy_count": len(state.open_copies(mesh.member)),
        }
        print(json.dumps(outcome, sort_keys=True))
        return outcome
    finally:
        release.set()
        worker.join(net.NET_WAIT_BUDGET_S)
        assert not worker.is_alive(), "owned wipe thread leaked"


@pytest.mark.parametrize("late", [False, True])
@pytest.mark.parametrize("lose_ack", [False, True])
def test_a_recopy_survives_an_old_equal_generation_wipe_reply(
    secret_mesh: Any, monkeypatch: pytest.MonkeyPatch, late: bool, lose_ack: bool  # noqa: F811
) -> None:
    result = _recopy_while_wipe_reply_is_held(
        secret_mesh, monkeypatch, late=late, lose_ack=lose_ack
    )
    assert result["member_has_copy"]
    assert result["pending_after"] == lose_ack, result
    assert result["open_copy_count"] == 1, result
    if not lose_ack:
        assert not result["owner_wiped"], result
    mesh = secret_mesh
    record = store.load(mesh.network_id, mesh.a.root)
    assert _lop_network("member", "rm", record.name, mesh.b.identity.name) == 0
    assert not _member_has(mesh.b.root, SECRET_NAME)
    state = sync.SyncState.load(mesh.network_id, mesh.a.root)
    assert state.open_copies(mesh.member) == []
    print("ACTUAL_MEMBER_RM recopy_absent=true open_copies=0")


def test_bypassing_the_conditional_check_reproduces_the_loss(
    secret_mesh: Any,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # A negative control of THIS mechanism, not a different older code path.
    monkeypatch.setattr(sync.SyncState, "ending_is_current", lambda *args: True, raising=False)
    result = _recopy_while_wipe_reply_is_held(secret_mesh, monkeypatch, late=True, lose_ack=True)
    assert result["member_has_copy"] and not result["pending_after"], result
    assert result["owner_wiped"] and result["open_copy_count"] == 0, result
    mesh = secret_mesh
    record = store.load(mesh.network_id, mesh.a.root)
    capsys.readouterr()
    assert _lop_network("member", "rm", record.name, mesh.b.identity.name, "--json") == 0
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["copies"] == 0, receipt
    assert _member_has(mesh.b.root, SECRET_NAME), "the bypass must expose the original leak"


def test_local_copy_versions_survive_reload_and_do_not_depend_on_clock(tmp_path: Path) -> None:
    state = sync.SyncState("n_test", root=tmp_path)
    state.record_dispatch("d_b", KEY, gen=1, digest=_d(1), at=100.0)
    first = state.copy_version("d_b", KEY)
    assert first is not None
    state.record_dispatch("d_b", KEY, gen=1, digest=_d(1), at=100.0)
    second = state.copy_version("d_b", KEY)
    assert second is not None and second[0] > first[0] and second[1] != first[1]
    state.record_ack("d_b", KEY, gen=1, digest=_d(1), at=100.0)
    third = state.copy_version("d_b", KEY)
    assert third is not None and third[0] > second[0] and third[1] != second[1]
    state.save()
    restored = sync.SyncState.load("n_test", tmp_path)
    assert restored.copy_version("d_b", KEY) == third
    assert not restored.ending_is_current("d_b", KEY, first)
    assert restored.ending_is_current("d_b", KEY, third)
    # Other keys/devices and refused stale transitions must not invalidate this
    # ending, and a file-loss counter reset cannot recreate the same identity.
    restored.record_dispatch("d_other", KEY, gen=1, digest=_d(1), at=100.0)
    restored.record_dispatch("d_b", "secret:OTHER", gen=1, digest=_d(1), at=100.0)
    assert restored.ending_is_current("d_b", KEY, third)
    # A lower dispatch is refused only when a higher dispatch already exists;
    # after a normal ACK the dispatch was settled, so recreate a valid one first.
    restored.record_dispatch("d_b", KEY, gen=3, digest=_d(3), at=100.0)
    fourth = restored.copy_version("d_b", KEY)
    restored.record_dispatch("d_b", KEY, gen=2, digest=_d(2), at=100.0)
    assert restored.ending_is_current("d_b", KEY, fourth)
    restored.record_ack("d_b", KEY, gen=3, digest=_d(3), at=100.0)
    fifth = restored.copy_version("d_b", KEY)
    restored.record_ack("d_b", KEY, gen=2, digest=_d(2), at=100.0)
    assert restored.ending_is_current("d_b", KEY, fifth)
    reset = sync.SyncState("n_test", root=tmp_path)
    reset.record_dispatch("d_b", KEY, gen=1, digest=_d(1), at=100.0)
    assert reset.copy_version("d_b", KEY) != first


@pytest.mark.parametrize("ending", ["tick", "removal"])
def test_a_recopy_dispatched_after_planning_is_not_settled_by_that_wipe(
    secret_mesh: Any, monkeypatch: pytest.MonkeyPatch, ending: str  # noqa: F811
) -> None:
    mesh = secret_mesh
    sync.credentials_sync_step(mesh.a, mesh.member)
    assert net.wait_for(
        lambda: sync.SyncState.load(mesh.network_id, mesh.a.root).ack_for(mesh.member, KEY)
        is not None
    )
    assert net.wait_for(lambda: mesh.member not in mesh.engine_a._inflight_members)
    monkeypatch.setattr(mesh.engine_a, "enqueue_owner_exchange", lambda device: "scheduled")
    old = sync.SyncState.load(mesh.network_id, mesh.a.root).ack_for(mesh.member, KEY)
    assert old is not None
    link = mesh.b._ensure_link(mesh.a.identity.device_id)
    assert link is not None
    if ending == "tick":
        assert _lop_network("credential", "mark", SECRET_NAME, "local-only") == 0
    dial_name = "_ensure_link" if ending == "tick" else "_ensure_link_with_reason"
    dial = getattr(mesh.a, dial_name)
    held_reply: list[dict[str, Any]] = []

    def recopy_before_dial(*args: Any, **kwargs: Any) -> Any:
        # The list already exists. Restore the seam before doing any more I/O;
        # a same-generation copy is sent, but not applied until the wipe ends.
        monkeypatch.setattr(mesh.a, dial_name, dial)
        if ending == "tick":
            assert _lop_network("credential", "mark", SECRET_NAME, "default") == 0
        reply = link.request(
            {
                "op": "net_broker",
                "kind": "copy",
                "network_id": mesh.network_id,
                "from_device": mesh.member,
                "from_device_name": mesh.b.identity.name,
                "req": mesh.b._next_relay_req(),
                "key": KEY,
                "gen": old["gen"],
                "held": 0,
                "held_digest": "",
            },
            timeout=sync.COPY_TIMEOUT_S,
        )
        assert isinstance(reply, dict)
        detail = reply.get("detail")
        assert isinstance(detail, dict) and detail.get("kind") == "copy"
        assert detail["gen"] == old["gen"], "the schedule must not bump byte generation"
        held_reply.append(detail)
        assert sync.SyncState.load(mesh.network_id, mesh.a.root).dispatch_for(mesh.member, KEY)
        return dial(*args, **kwargs)

    monkeypatch.setattr(mesh.a, dial_name, recopy_before_dial)
    if ending == "tick":
        mesh.engine_a._exchange_blocking(mesh.member)
    else:
        result = mesh.engine_a.deliver_removal_endings(mesh.member)
        assert result["wiped"] == 0, result
    assert len(held_reply) == 1
    assert not _member_has(mesh.b.root, SECRET_NAME), "the old copy really was wiped"
    state = sync.SyncState.load(mesh.network_id, mesh.a.root)
    assert state.dispatch_for(mesh.member, KEY) is not None, "the old reply erased a NEW send"
    held = _HeldAcks(monkeypatch, mesh.b.root)
    assert (
        mesh.engine_b.apply_copy_reply(KEY, held_reply[0], owner=mesh.a.identity.device_id)
        == "applied"
    )
    assert held.held.is_set() and _member_has(mesh.b.root, SECRET_NAME)
    assert len(sync.SyncState.load(mesh.network_id, mesh.a.root).open_copies(mesh.member)) == 1
    print(f"PLANNING_RACE {ending=} generation_unchanged=true pending=true copy_present=true")
    record = store.load(mesh.network_id, mesh.a.root)
    assert _lop_network("member", "rm", record.name, mesh.b.identity.name) == 0
    assert not _member_has(mesh.b.root, SECRET_NAME)
    assert sync.SyncState.load(mesh.network_id, mesh.a.root).open_copies(mesh.member) == []


def _separate_absent_ack(mesh: Any) -> sync.SyncState:
    state = sync.SyncState.load(mesh.network_id, mesh.a.root)
    sent = state.dispatch_for(mesh.member, KEY)
    assert sent is not None
    link = mesh.b._ensure_link(mesh.a.identity.device_id)
    assert link is not None
    reply = link.request(
        {
            "op": "net_broker",
            "kind": "ack",
            "network_id": mesh.network_id,
            "from_device": mesh.member,
            "from_device_name": mesh.b.identity.name,
            "req": mesh.b._next_relay_req(),
            "key": KEY,
            "gen": sent["gen"],
            "digest": sent["digest"],
            "value_state": "absent",
        },
        timeout=sync.ANNOUNCE_TIMEOUT_S,
    )
    assert isinstance(reply, dict) and reply.get("detail", {}).get("kind") == "ack"
    assert _member_has(mesh.b.root, SECRET_NAME), "an ACK is not a physical wipe"
    return sync.SyncState.load(mesh.network_id, mesh.a.root)


def test_a_separate_absent_ack_cannot_settle_an_unidentified_copy(
    secret_mesh: Any, monkeypatch: pytest.MonkeyPatch  # noqa: F811
) -> None:
    _deliver_without_ack(secret_mesh, monkeypatch)
    after = _separate_absent_ack(secret_mesh)
    assert after.dispatch_for(secret_mesh.member, KEY) is not None
    assert not (after.ack_for(secret_mesh.member, KEY) or {}).get("wiped")


def test_bypassing_identity_allows_the_separate_ack_to_forge_an_ending(
    secret_mesh: Any, monkeypatch: pytest.MonkeyPatch  # noqa: F811
) -> None:
    _deliver_without_ack(secret_mesh, monkeypatch)
    monkeypatch.setattr(sync.SyncState, "ending_is_current", lambda *args: True)
    after = _separate_absent_ack(secret_mesh)
    assert after.dispatch_for(secret_mesh.member, KEY) is None
    assert (after.ack_for(secret_mesh.member, KEY) or {}).get("wiped") is True


@pytest.mark.parametrize("legacy", ["absent", "malformed"])
@pytest.mark.parametrize("ending", ["tick", "removal"])
def test_a_legacy_copy_is_identified_when_planned_and_confirms_its_ending(
    secret_mesh: Any,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    legacy: str,
    ending: str,
) -> None:
    mesh = secret_mesh
    if ending == "removal":
        _deliver_without_ack(mesh, monkeypatch)
    else:
        sync.credentials_sync_step(mesh.a, mesh.member)
        assert net.wait_for(
            lambda: sync.SyncState.load(mesh.network_id, mesh.a.root).ack_for(mesh.member, KEY)
            is not None
        )
    assert net.wait_for(lambda: mesh.member not in mesh.engine_a._inflight_members)
    monkeypatch.setattr(mesh.engine_a, "enqueue_owner_exchange", lambda device: "scheduled")
    state = sync.SyncState.load(mesh.network_id, mesh.a.root)
    payload = state.to_json()
    if legacy == "absent":
        payload.pop("copy_versions", None)
    else:
        payload["copy_versions"] = {mesh.member: {KEY: {"revision": True, "identity": "bad"}}}
    # Only our synthetic ledger is rewritten; the open copy is real and durable.
    state.path.write_text(json.dumps(payload), encoding="utf-8")
    assert sync.SyncState.load(mesh.network_id, mesh.a.root).copy_version(mesh.member, KEY) is None
    if ending == "removal":
        record = store.load(mesh.network_id, mesh.a.root)
        capsys.readouterr()
        assert _lop_network("member", "rm", record.name, mesh.b.identity.name) == 0
        out = capsys.readouterr().out
        assert "was deleted (the ending is confirmed)" in out, out
    else:
        assert _lop_network("credential", "mark", SECRET_NAME, "local-only") == 0
        mesh.engine_a._exchange_blocking(mesh.member)
    after = sync.SyncState.load(mesh.network_id, mesh.a.root)
    assert not _member_has(mesh.b.root, SECRET_NAME)
    assert after.copy_version(mesh.member, KEY) is not None
    assert (after.ack_for(mesh.member, KEY) or {}).get("wiped") is True
    assert after.open_copies(mesh.member) == []
    if ending == "tick":

        def no_redial(*args: Any, **kwargs: Any) -> Any:
            pytest.fail("a confirmed legacy ending must not be re-sent on the next tick")

        monkeypatch.setattr(mesh.a, "_ensure_link", no_redial)
        mesh.engine_a._exchange_blocking(mesh.member)
        capsys.readouterr()
        assert (
            _lop_network("credential", "revoke", KEY, "--from", mesh.b.identity.name, "--json") == 0
        )
        receipt = json.loads(capsys.readouterr().out)
        assert receipt["copy"] == {"confirmed": True, "wiped": True, "wipe": "done"}, receipt
    print(f"LEGACY_ENDING {legacy=} {ending=} confirmed=true open_copies=0")
