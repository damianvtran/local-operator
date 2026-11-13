"""A copy the owner SENT but whose ACK never arrived still ends at member removal.

THE DEFECT. ``owner_copy`` returned the value and wrote nothing about WHO it was
sent to; only ``owner_ack`` recorded a recipient, and ``deliver_removal_endings``
derived its work from those ack rows alone. So when the member stored the copy
and its ack was lost (held, dropped, a link that died after the apply), the
owner's ``member rm`` found nothing owed: the receipt said nothing about copies
and the mesh-origin credential stayed usable on the removed device, which is
never contacted again.

THE INVARIANT this file pins: before a ``copy`` (or ``in_sync``) answer leaves
the owner, a DISPATCH row naming the recipient, key, generation and digest is
durable. It is NOT an ack: it never satisfies ``_ack_matches`` (so reconcile
keeps running until the member's real ack lands), a real ack at or past its
generation settles it, and a removal ending treats it as an open copy.

Real path throughout: two relays on loopback roots (``secret_mesh``), the real
copy exchange, and the real ``lop network member rm`` control op.
"""

from __future__ import annotations

import ipaddress
import json
import socket
import threading
from typing import Any

import pytest

from local_operator.network import store
from local_operator.network.credentials import placement, sync
from local_operator.network.credentials.messages import render_removal_endings
from local_operator.network.credentials.sync import SyncState
from tests.unit.network import conftest as net_fixtures
from tests.unit.network.test_credentials_real_link import _lop_network  # noqa: F401
from tests.unit.network.test_relay_e2e import devices  # noqa: F401 — fixtures by import
from tests.unit.network.test_secret_copies import (  # noqa: F401 — fixtures by import
    KEY,
    SECRET_NAME,
    _member_has,
    secret_mesh,
)


@pytest.fixture(autouse=True)
def _loopback_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """These fixtures must never dial a provider or the operator's live mesh."""
    connect = socket.socket.connect

    def checked(sock: socket.socket, address: Any) -> Any:
        if isinstance(address, tuple):
            host = address[0]
            assert (
                host == "localhost" or ipaddress.ip_address(host).is_loopback
            ), "copy-dispatch tests allow only loopback TCP"
        return connect(sock, address)

    monkeypatch.setattr(socket.socket, "connect", checked)


def test_the_loopback_guard_is_wired_before_any_case() -> None:
    # A documentation-only negative guard is not evidence: exercise it without
    # ever reaching connect(2) for this reserved non-loopback address.
    with socket.socket() as sock, pytest.raises(AssertionError, match="only loopback TCP"):
        sock.connect(("192.0.2.1", 443))


class _HeldAcks:
    """Hold the MEMBER's ack after its durable apply; never let it reach the owner."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, member_root: Any) -> None:
        self.held = threading.Event()
        self.frames: list[tuple[str, int, str]] = []
        real = sync.SyncEngine._send_ack

        def hold(engine: Any, network_id: str, owner: str, key: str, gen: int, digest: str):
            if str(engine._root) == str(member_root):  # noqa: SLF001 — the member's engine
                self.frames.append((key, gen, digest))
                self.held.set()
                return None
            return real(engine, network_id, owner, key, gen, digest)

        monkeypatch.setattr(sync.SyncEngine, "_send_ack", hold)


def _deliver_without_ack(mesh: Any, monkeypatch: pytest.MonkeyPatch) -> _HeldAcks:
    acks = _HeldAcks(monkeypatch, mesh.b.root)
    code = sync.credentials_sync_step(mesh.a, mesh.member)
    assert code in ("scheduled", "in_sync"), code
    assert net_fixtures.wait_for(lambda: _member_has(mesh.b.root, SECRET_NAME))
    assert acks.held.wait(30), "the member never reached its ack"
    state = SyncState.load(mesh.network_id, mesh.a.root)
    # The precondition the defect needs: the owner holds NO ack row for B.
    assert state.ack_for(mesh.member, KEY) is None
    return acks


def _remove(mesh: Any) -> str:
    record = store.load(mesh.network_id, mesh.a.root)
    assert _lop_network("member", "rm", record.name, mesh.b.identity.name) == 0
    return ""


def test_a_delivered_copy_with_a_held_ack_is_wiped_by_member_rm(
    secret_mesh: Any,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    mesh = secret_mesh
    _deliver_without_ack(mesh, monkeypatch)
    capsys.readouterr()
    _remove(mesh)
    out = capsys.readouterr().out
    assert net_fixtures.wait_for(
        lambda: not _member_has(mesh.b.root, SECRET_NAME)
    ), f"the removal left the delivered copy on B: {out}"
    row = SyncState.load(mesh.network_id, mesh.a.root).ack_for(mesh.member, KEY) or {}
    assert row.get("wiped") is True, row
    assert "1 copy on " in out and "was deleted (the ending is confirmed)" in out, out


def test_an_unreachable_member_with_an_unacked_copy_reads_unconfirmed(
    secret_mesh: Any,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    mesh = secret_mesh
    _deliver_without_ack(mesh, monkeypatch)
    for link in list(mesh.a.links.values()):
        if getattr(link, "device_id", "") == mesh.member:
            link.close("test: the member becomes unreachable")
    with store.mutate(mesh.network_id, mesh.a.root) as record:
        member_row = record.member(mesh.member)
        assert member_row is not None
        member_row.endpoints = ["127.0.0.1:1"]
        store.save(record, mesh.a.root)
    capsys.readouterr()
    _remove(mesh)
    out = capsys.readouterr().out
    assert _member_has(mesh.b.root, SECRET_NAME), "no contact, no wipe"
    assert "could NOT be confirmed deleted" in out, out
    # Sent is not the same claim as held: the receipt says the delivery itself
    # was never acknowledged, so it is not read as a confirmed copy.
    assert "never acknowledged" in out, out


def test_a_pending_dispatch_never_reads_as_in_sync(
    secret_mesh: Any, monkeypatch: pytest.MonkeyPatch  # noqa: F811
) -> None:
    """Reconcile keeps running: the dispatch row is not an ack."""
    mesh = secret_mesh
    _deliver_without_ack(mesh, monkeypatch)
    state = SyncState.load(mesh.network_id, mesh.a.root)
    dispatched = state.dispatch_for(mesh.member, KEY)
    assert dispatched is not None and int(dispatched["gen"]) > 0, state.to_json()
    current = state.generation(KEY) or {}
    assert not sync._ack_matches(
        state.ack_for(mesh.member, KEY), int(current["gen"]), str(current["digest"])
    )
    # The cheap tick gate still finds this member owed work (announce again).
    engine = sync.sync_for_relay(mesh.a)
    assert engine is not None
    _network_id, announces = engine._pending_announces(mesh.member)  # noqa: SLF001
    assert any(item.get("key") == KEY for item in announces), announces


def test_a_normal_acknowledged_copy_leaves_no_dispatch_row(
    secret_mesh: Any,  # noqa: F811
) -> None:
    mesh = secret_mesh
    code = sync.credentials_sync_step(mesh.a, mesh.member)
    assert code in ("scheduled", "in_sync"), code
    assert net_fixtures.wait_for(
        lambda: SyncState.load(mesh.network_id, mesh.a.root).ack_for(mesh.member, KEY) is not None
    )
    assert net_fixtures.wait_for(
        lambda: SyncState.load(mesh.network_id, mesh.a.root).dispatch_for(mesh.member, KEY) is None
    ), "the ack must settle the dispatch it confirms"


# ---------------------------------------------------------------------------
# The ledger rules, on the document itself
# ---------------------------------------------------------------------------


def _d(n: int) -> str:
    """A well-formed digest (the ledger drops any other shape at load)."""
    return sync._DIGEST_PREFIX + f"{n:x}" * sync._DIGEST_HEX_LENGTH


def _state(tmp_path: Any) -> SyncState:
    return SyncState("n_test", root=tmp_path)


def test_a_dispatch_never_regresses_and_a_stale_ack_does_not_settle_it(tmp_path: Any) -> None:
    state = _state(tmp_path)
    state.record_dispatch("d_b", KEY, gen=3, digest=_d(3), at=10.0)
    state.record_dispatch("d_b", KEY, gen=2, digest=_d(2), at=11.0)
    assert (state.dispatch_for("d_b", KEY) or {}).get("gen") == 3
    state.record_ack("d_b", KEY, gen=2, digest=_d(2), at=12.0)
    assert state.dispatch_for("d_b", KEY) is not None, "an older ack confirms nothing newer"
    state.record_ack("d_b", KEY, gen=3, digest=_d(3), at=13.0)
    assert state.dispatch_for("d_b", KEY) is None


def test_a_dispatch_after_a_wipe_stays_open_until_a_newer_ack(tmp_path: Any) -> None:
    """A re-copy after a wiped ending is a NEW copy: the old wiped row cannot close it."""
    state = _state(tmp_path)
    state.record_ack("d_b", KEY, gen=3, digest=_d(3), at=10.0, wiped=True)
    state.record_dispatch("d_b", KEY, gen=3, digest=_d(3), at=11.0)
    assert state.dispatch_for("d_b", KEY) is not None
    assert [row["key"] for row in state.open_copies("d_b")] == [KEY]
    # A wipe-ack for an OLDER generation must not close the newer dispatch.
    state.record_ack("d_b", KEY, gen=2, digest=_d(2), at=12.0, wiped=True)
    assert state.dispatch_for("d_b", KEY) is not None
    assert (state.ack_for("d_b", KEY) or {}).get("gen") == 3


def test_the_dispatch_ledger_round_trips_and_drops_corrupt_rows(tmp_path: Any) -> None:
    state = _state(tmp_path)
    state.record_dispatch("d_b", KEY, gen=4, digest=_d(4), at=10.0)
    payload = state.to_json()
    payload["dispatched"]["d_b"]["secret:BAD"] = {"gen": 0, "digest": "x"}
    payload["dispatched"]["d_c"] = "not a mapping"
    path = state.path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    loaded = SyncState.load("n_test", root=tmp_path)
    assert loaded.dispatch_for("d_b", KEY) == {"gen": 4, "digest": _d(4), "at": 10}
    assert loaded.dispatch_for("d_b", "secret:BAD") is None
    assert "d_c" not in loaded.dispatched


def test_the_receipt_names_unacknowledged_copies_without_calling_them_confirmed() -> None:
    lines = render_removal_endings("device-b", copies=2, wiped=0, unacknowledged=1)
    text = "\n".join(lines)
    assert "2 of 2 copies on device-b could NOT be confirmed deleted" in text, text
    assert "1 copy on device-b was sent but its delivery was never acknowledged" in text, text
    confirmed = "\n".join(render_removal_endings("device-b", copies=1, wiped=1, unacknowledged=1))
    assert "never acknowledged" not in confirmed, confirmed


def _member_copy_request(mesh: Any, **fields: Any) -> dict[str, Any]:
    """Send ONE real copy request from the member B to the owner A, over the link."""
    link = mesh.b._ensure_link(mesh.a.identity.device_id)  # noqa: SLF001 — the one dial seam
    assert link is not None
    frame = {
        "op": "net_broker",
        "kind": "copy",
        "network_id": mesh.network_id,
        "from_device": mesh.member,
        "from_device_name": mesh.b.identity.name,
        "req": mesh.b._next_relay_req(),  # noqa: SLF001
        **fields,
    }
    reply = link.request(frame, timeout=15.0)
    assert isinstance(reply, dict), reply
    return reply


def test_refused_and_malformed_copy_requests_record_no_dispatch(
    secret_mesh: Any,  # noqa: F811
) -> None:
    """The dispatch row is written only AFTER every authorisation check passes."""
    mesh = secret_mesh
    for fields in (
        {"gen": 1, "held": 0},  # no key: malformed
        {"key": "secret:NOT_SHARED", "gen": 1, "held": 0},  # not a holder of it
        {"key": KEY, "gen": 1, "held": sync.GEN_SAFE_MAX},  # adoption bound refused
    ):
        reply = _member_copy_request(mesh, **fields)
        detail = reply.get("detail") or {}
        assert detail.get("kind") == "error", (fields, reply)
    state = SyncState.load(mesh.network_id, mesh.a.root)
    assert state.dispatched == {}, state.dispatched


@pytest.mark.parametrize("as_json", [False, True])
def test_offline_owner_removal_reports_unacknowledged_dispatches(
    secret_mesh: Any,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    as_json: bool,
) -> None:
    mesh = secret_mesh
    _deliver_without_ack(mesh, monkeypatch)
    mesh.a.stop()
    capsys.readouterr()
    args = ("--json",) if as_json else ()
    record = store.load(mesh.network_id, mesh.a.root)
    assert _lop_network("member", "rm", record.name, mesh.b.identity.name, *args) == 0
    out = capsys.readouterr().out
    assert _member_has(mesh.b.root, SECRET_NAME), "no owner relay means no wipe exchange"
    if as_json:
        receipt = json.loads(out)
        assert receipt["copies"] == 1 and receipt["wiped"] == 0, receipt
        assert receipt["unacknowledged"] == 1, receipt
    else:
        assert "could NOT be confirmed deleted" in out, out
        assert "1 copy on " in out and "delivery was never acknowledged" in out, out


def test_timeout_and_unconfirmed_lines_name_the_unacknowledged_set() -> None:
    text = "\n".join(
        render_removal_endings("device-b", copies=3, wiped=1, timed_out=1, unacknowledged=1)
    )
    assert "1 of 3 copies on device-b timed out" in text, text
    assert "1 of 3 copies on device-b could NOT be confirmed deleted" in text, text
    assert "1 copy on device-b was sent but its delivery was never acknowledged" in text, text
    assert "of them" not in text, text
    plural = "\n".join(render_removal_endings("device-b", copies=2, wiped=0, unacknowledged=2))
    assert "2 copies on device-b were sent but their delivery was never acknowledged" in plural


@pytest.mark.parametrize("failure", ["no_placement", "unreadable"])
def test_early_removal_counts_include_unacknowledged(
    secret_mesh: Any, monkeypatch: pytest.MonkeyPatch, failure: str  # noqa: F811
) -> None:
    mesh = secret_mesh
    if failure == "no_placement":
        monkeypatch.setattr(mesh.engine_a, "_placement", lambda: None)
    else:

        def unreadable(*_args: Any, **_kwargs: Any) -> Any:
            raise OSError("synthetic unreadable copy ledger")

        monkeypatch.setattr(SyncState, "load", unreadable)
    assert mesh.engine_a.deliver_removal_endings(mesh.member) == {
        "copies": 0,
        "wiped": 0,
        "timed_out": 0,
        "unacknowledged": 0,
    }


@pytest.mark.parametrize("as_json", [False, True])
def test_revoke_tracks_and_ends_a_dispatch_only_copy(
    secret_mesh: Any,  # noqa: F811
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    as_json: bool,
) -> None:
    mesh = secret_mesh
    _deliver_without_ack(mesh, monkeypatch)
    capsys.readouterr()
    args = ("--json",) if as_json else ()
    assert _lop_network("credential", "revoke", KEY, "--from", mesh.b.identity.name, *args) == 0
    out = capsys.readouterr().out
    if as_json:
        receipt = json.loads(out)
        assert receipt["copy"] == {
            "confirmed": False,
            "wiped": False,
            "wipe": "queued",
            "unacknowledged": True,
        }, receipt
    else:
        assert "no copy" not in out, out
        assert "delivery was never acknowledged" in out, out
        assert "wipe notice" in out, out
    document = mesh.engine_a._placement()  # noqa: SLF001 — the normal tick's cheap gate
    assert mesh.engine_a._has_pending_wipes(document, mesh.member)  # noqa: SLF001
    mesh.engine_a._exchange_blocking(mesh.member)  # noqa: SLF001 — normal exchange, real link
    assert net_fixtures.wait_for(lambda: not _member_has(mesh.b.root, SECRET_NAME))
    state = SyncState.load(mesh.network_id, mesh.a.root)
    ending = state.ack_for(mesh.member, KEY)
    assert ending is not None and ending["wiped"] is True
    assert state.dispatch_for(mesh.member, KEY) is None
    assert not mesh.engine_a._has_pending_wipes(document, mesh.member)  # noqa: SLF001


@pytest.mark.parametrize("selector", ["gate", "frames"])
def test_revoke_dispatch_is_selected_by_the_normal_tick(
    secret_mesh: Any, monkeypatch: pytest.MonkeyPatch, selector: str  # noqa: F811
) -> None:
    mesh = secret_mesh
    _deliver_without_ack(mesh, monkeypatch)
    assert _lop_network("credential", "revoke", KEY, "--from", mesh.b.identity.name) == 0
    document = mesh.engine_a._placement()  # noqa: SLF001
    if selector == "gate":
        assert mesh.engine_a._has_pending_wipes(document, mesh.member)  # noqa: SLF001
    else:
        _network, frames = mesh.engine_a._pending_announces(mesh.member)  # noqa: SLF001
        assert any(frame["key"] == KEY and frame["value_state"] == "absent" for frame in frames)


@pytest.mark.parametrize("reason", ["inactive", "local_only", "not_copyable", "no_value"])
def test_each_copy_policy_refusal_over_the_link_records_no_dispatch(
    secret_mesh: Any, reason: str  # noqa: F811
) -> None:
    mesh = secret_mesh
    # Establish the authenticated link before changing standing, so the inactive
    # case tests an already admitted request too, not just a refused handshake.
    assert mesh.b._ensure_link(mesh.owner) is not None  # noqa: SLF001
    if reason == "inactive":
        with store.mutate(mesh.network_id, mesh.a.root) as record:
            member = record.member(mesh.member)
            assert member is not None
            member.lifecycle = "expired"
            store.save(record, mesh.a.root)
    elif reason == "local_only":
        with sync.mutate(mesh.network_id, mesh.a.root) as state:
            state.record_mark(KEY, "local-only")
    elif reason == "not_copyable":
        with placement.mutate(mesh.network_id, mesh.a.root, self_device=mesh.owner) as document:
            entry = document.entry(KEY)
            assert entry is not None
            entry.kind = "oauth-rotating"
    else:
        mesh.store_a.delete(SECRET_NAME)
    reply = _member_copy_request(mesh, key=KEY, gen=1, held=0)
    if reason == "inactive":
        # The real transport refuses the now-inactive member before dispatching
        # to owner_copy; do not bypass that stronger backstop to reach its guard.
        assert reply.get("op") == "error", reply
        assert "broker_credential" in str(reply.get("message")), reply
    else:
        detail = reply.get("detail") or {}
        assert detail.get("kind") == "error", reply
        assert detail.get("code") == reason, reply
    state = SyncState.load(mesh.network_id, mesh.a.root)
    assert state.dispatched == {}, state.dispatched
    assert state.acks == {}, state.acks
    assert not _member_has(mesh.b.root, SECRET_NAME)


def test_a_wipe_ack_from_another_device_cannot_close_a_dispatch(tmp_path: Any) -> None:
    """Rows are keyed by the AUTHENTICATED recipient: only its own ending closes it."""
    state = _state(tmp_path)
    state.record_dispatch("d_b", KEY, gen=3, digest=_d(3), at=10.0)
    state.record_ack("d_other", KEY, gen=9, digest=_d(9), at=11.0, wiped=True)
    assert state.dispatch_for("d_b", KEY) is not None
    assert [row["key"] for row in state.open_copies("d_b")] == [KEY]
