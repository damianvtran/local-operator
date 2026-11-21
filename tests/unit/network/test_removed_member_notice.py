"""G9-B: a member removed while OFFLINE learns it at reconnect, and drops what it borrowed.

THE GAP (measured on main with the real-path cell below). ``lop network member rm``
of a device that is offline at the time (the lost or stolen laptop) never reaches
it: the rotation frame that names it removed has nowhere to go, and every later
handshake it makes is refused SILENTLY (the no-oracle rule), which it cannot tell
from an owner that is asleep. So it kept serving its borrowed MCP/provider bearers
from the offline hold until the hold ended (24 h by default).

THE FIX: a listener answers a dialer that PROVES the exact key stored on its
tombstoned row with one fixed, signed "membership ended" frame; the dialer verifies
it against a CURRENT member's stored key, over this handshake's transcript, and
reduces its own authority at once. Every other dialer keeps the silence it had.

The security owner's constraints, one cell (or more) each:

* proof-gated, no oracle: ``test_non_proven_dialers_walk_one_path_and_get_silence``
  (random key for a tombstoned id, a wrong key, a never-member: same branch, one
  signature check each, the same refusal, nothing on the socket);
* transcript-bound, current-member signer: the ``test_dialer_*`` cells (replay from
  another handshake, a non-member signer, an unknown key, a REMOVED signer, an
  epoch not newer than ours, a malformed frame — none reaches the drop path);
* local and reduce-only, one audit row:
  ``test_an_offline_removed_member_learns_at_reconnect_and_drops_its_lends``;
* fixed minimal content: ``test_the_notice_is_fixed_and_minimal``;
* rate limit per key: ``test_a_second_dial_inside_the_window_gets_silence``;
* 7-day tombstone bound: ``test_a_tombstone_past_seven_days_gets_silence``;
* the remover never dials the removed member: asserted inside the real-path cell.

Real path for the reconnect: the repo's ``held`` rig (two loopback relays, real
pairing, real ``credential share``, a GitHub App token and an MCP bearer lent to B;
grant 2 s, hold 6 s). Listener cells drive ``RelayServer._run_inbound_handshake``
over a socketpair with the real ``Handshake`` code on the dialing end. Synthetic ids.
"""

from __future__ import annotations

import json
import socket
import threading
import time
from pathlib import Path
from secrets import token_bytes
from typing import Any, cast

import pytest

from local_operator.network import audit as audit_mod
from local_operator.network import handshake as hs_mod
from local_operator.network import identity, relay, store, types, wire
from local_operator.network.credentials import github as github_mod
from local_operator.network.credentials import mcp_state
from local_operator.network.credentials.types import BrokerError
from tests.unit.network import conftest as net_fixtures
from tests.unit.network.test_credentials_github import (  # noqa: F401 — fixtures by import
    _no_secret_broker_daemon,
    app_keypair,
    env,
    github_api,
    owner,
)
from tests.unit.network.test_credentials_hold_github_mcp import (  # noqa: F401
    MCP_KEY,
    _past_window,
    held,
)
from tests.unit.network.test_credentials_owner import idp, stub_provider  # noqa: F401
from tests.unit.network.test_credentials_real_link import (  # noqa: F401
    _lop_network,
    mesh,
)
from tests.unit.network.test_relay_e2e import (  # noqa: F401
    _pair,
    devices,
    serve_shaped_relay,
)

# ---------------------------------------------------------------------------
# Real path: the lost laptop comes back
# ---------------------------------------------------------------------------


def _remove_while_offline(rig: Any, monkeypatch: pytest.MonkeyPatch) -> tuple[Any, list[str]]:
    """Stop B's relay, ``member rm`` it on A, bring B back: the lost-laptop shape.

    Copied from PR #4393's ``test_member_rm_ends_lends.py`` (not imported, so this
    file does not depend on that PR). Also returns every ``expected_device`` A
    dialled from the removal on, to prove A never dials the removed member.
    """
    net = rig.mesh
    net.b.stop()
    assert net_fixtures.wait_for(lambda: not any(link.alive for link in net.a.links.values()))
    a_dials: list[str] = []
    original_dial = net.a.dial

    def spy(*args: Any, **kwargs: Any) -> Any:
        a_dials.append(str(kwargs.get("expected_device") or ""))
        return original_dial(*args, **kwargs)

    monkeypatch.setattr(net.a, "dial", spy)
    net.a._ctl_member_rm({"network": net.network_id, "device_id": net.borrower})  # noqa: SLF001
    server = serve_shaped_relay(
        net.b.root,
        monkeypatch,
        settings=relay.NetworkSettings(port=net.b.settings.port, listen_address="127.0.0.1"),
        identity=net.b.identity,
        audit=net.b.audit,
    )
    server.bind()
    server.bind_control()
    server.start()
    return server, a_dials


def _self_row_active(net: Any) -> bool:
    record = store.load(net.network_id, net.b.root)
    me = record.self_member()
    return bool(me is not None and me.active)


def _audit_rows(root: Path, event: str) -> list[dict[str, Any]]:
    path = root / "network" / "audit.jsonl"
    if not path.exists():
        return []
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return [row for row in rows if row.get("event") == event]


def test_an_offline_removed_member_learns_at_reconnect_and_drops_its_lends(
    held: Any, monkeypatch: pytest.MonkeyPatch  # noqa: F811
) -> None:
    """RED on main: B came back still a member in its own eyes and served the MCP hold."""
    client, net = held.client, held.mesh
    before = store.load(net.network_id, net.b.root)
    server, a_dials = _remove_while_offline(held, monkeypatch)
    try:
        # B dials A on its own (membership loop); A proves nothing to anyone else.
        learned = net_fixtures.wait_for(lambda: not _self_row_active(net), timeout_s=20.0)
        assert learned, "B reconnected and still reads itself an active member"
        assert client._self_membership_active() is False  # noqa: SLF001
        # Every borrowed bearer is refused, inside the grant window AND past it,
        # with no held serve in between.
        for key, provider in (
            (MCP_KEY, "mcp-oauth"),
            (github_mod.GITHUB_KEY, github_mod.GITHUB_KEY),
        ):
            refused = client.request_grant_sync(key, provider=provider)
            assert isinstance(refused, BrokerError) and refused.code == "not_a_holder", refused
        _past_window(held.mcp)
        for key, provider in (
            (MCP_KEY, "mcp-oauth"),
            (github_mod.GITHUB_KEY, github_mod.GITHUB_KEY),
        ):
            refused = client.request_grant_sync(key, provider=provider)
            assert isinstance(refused, BrokerError) and refused.code == "not_a_holder", refused
        rows = mcp_state.load(net.network_id, net.b.root)
        assert all(row.get("state") != "held" for row in rows.values()), rows
        # The App token B held is DELETEd at GitHub (the drop path's revoke).
        assert net_fixtures.wait_for(
            lambda: not held.api.live(held.gh.access_token), timeout_s=10.0
        )
        # ONE audit row on B, naming the member that proved it.
        net.b.audit.flush()
        ended = _audit_rows(net.b.root, "membership_ended")
        assert len(ended) == 1, ended
        assert ended[0]["actor"] == net.a.identity.device_id
        assert ended[0]["detail"]["credentials_dropped"] is True
        # REDUCE-ONLY: no rejoin, no rotation, nothing deleted beyond the drop.
        after = store.load(net.network_id, net.b.root)
        assert after.epoch == before.epoch
        assert {m.device_id for m in after.members} == {m.device_id for m in before.members}
        assert after.stale == "refused_by_peers"
        assert net.b.identity.device_id in after.removed_ids
        assert store.require_secrets(net.network_id, net.b.root).epoch == before.epoch
    finally:
        server.stop()
    # A never dialled the removed member: the notice is only ever an answer.
    assert net.borrower not in a_dials, a_dials


# ---------------------------------------------------------------------------
# Listener: the real relay handler, a real dialing Handshake over a socketpair
# ---------------------------------------------------------------------------


class Net:
    def __init__(self, root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        a_root = root / "a"
        self.a_identity = identity.mint(a_root, name="owner-a")
        self.b_identity = identity.mint(root / "b", name="removed-b")
        self.c_identity = identity.mint(root / "c", name="stranger-c")
        self.d_identity = identity.mint(root / "d", name="member-d")
        self.root = a_root
        monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(a_root))
        self.server = relay.RelayServer(
            settings=relay.NetworkSettings(port=0, listen_address="127.0.0.1"),
            identity=self.a_identity,
            audit=audit_mod.AuditLog(a_root),
        )
        record = types.NetworkRecord(
            network_id=store.new_network_id(),
            name="notice-net",
            created_by=self.a_identity.device_id,
            self_device_id=self.a_identity.device_id,
            self_role="admin",
            self_capabilities=sorted(types.capabilities_for_role("admin")),
            listen={"address": "127.0.0.1", "port": 0, "advertised": []},
        )
        self.secret_e1 = wire.b64u(token_bytes(32))
        state = types.SecretState(network_id=record.network_id, epoch=1, secret=self.secret_e1)
        for ident, role, via in (
            (self.a_identity, "admin", "self"),
            (self.b_identity, "drive", "invite"),
            (self.d_identity, "drive", "invite"),
        ):
            relay.admit(
                record,
                device_id=ident.device_id,
                public_key=ident.public_key,
                name=ident.name,
                role=role,
                capabilities=sorted(types.capabilities_for_role(role)),
                added_by=self.a_identity.device_id,
                added_via=via,
                root=a_root,
                persist=False,
            )
        store.save(record, a_root)
        store.save_secrets(state, a_root)
        relay.remove_member(
            record,
            state,
            device_id=self.b_identity.device_id,
            by=self.a_identity.device_id,
            root=a_root,
        )
        self.network_id = record.network_id
        assert store.load(self.network_id, a_root).epoch == 2
        #: (code, sentence) of every refusal the listener audited, unfolded (the
        #: audit itself coalesces repeats inside a window).
        self.refusals: list[tuple[str, str]] = []
        original = self.server._audit_handshake_refusal  # noqa: SLF001

        def spy(refusal: Any, *args: Any, **kwargs: Any) -> Any:
            self.refusals.append((refusal.code, str(refusal)))
            return original(refusal, *args, **kwargs)

        monkeypatch.setattr(self.server, "_audit_handshake_refusal", spy)

    def dial(
        self, signer: Any, *, claim: str = "", epoch: int = 1
    ) -> tuple[hs_mod.Handshake, bytes, Exception | None]:
        """One dial at A's real inbound handler. Returns B's handshake and A's raw reply."""
        # TCP on loopback, not ``socketpair``: the handler sets TCP_NODELAY, which an
        # AF_UNIX pair refuses (OSError -> a quiet close before the hello is read).
        with socket.create_server(("127.0.0.1", 0)) as listener:
            ours = socket.create_connection(listener.getsockname()[:2], timeout=5.0)
            theirs, _ = listener.accept()
        errors: list[Exception] = []

        def serve() -> None:
            try:
                self.server._run_inbound_handshake(theirs, ("127.0.0.1", 40000))  # noqa: SLF001
            except Exception as exc:  # noqa: BLE001 — surfaced to the cell
                errors.append(exc)

        thread = threading.Thread(target=serve, daemon=True)
        thread.start()
        hs = hs_mod.Handshake.new(
            role="dialer",
            identity=signer,
            network_id=self.network_id,
            epoch=epoch,
            instance_id="i_dialer",
            session_protocol=relay._session_protocol(),  # noqa: SLF001
            capabilities=list(wire.LINK_CAPABILITIES),
        )
        if claim:
            hs.self_device_id = claim
        try:
            hs.send_hello(ours)
            reader = wire.FrameReader(ours)
            hs.read_challenge(reader, wire.deadline_in(5.0))
            key = wire.epoch_key(self.secret_e1, self.network_id, 1)
            hs.send_auth(ours, hs_mod.Credential("epoch", 1, key))
            thread.join(10)
            assert not thread.is_alive()
            ours.settimeout(5.0)
            chunks = [bytes(reader._buffer)]  # noqa: SLF001 — anything already buffered
            while True:
                chunk = ours.recv(65536)
                if not chunk:
                    break
                chunks.append(chunk)
        finally:
            ours.close()
        return hs, b"".join(chunks), (errors[0] if errors else None)

    def expect(self) -> hs_mod.DialerExpectation:
        a = self.a_identity
        return hs_mod.DialerExpectation(
            device_id=a.device_id,
            public_key_for=lambda device: a.public_key if device == a.device_id else None,
        )


@pytest.fixture()
def net(root: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    rig = Net(root, monkeypatch)
    try:
        yield rig
    finally:
        rig.server.stop()


def _frame(raw: bytes) -> dict[str, Any]:
    line, _, rest = raw.partition(b"\n")
    assert rest == b"", raw
    return json.loads(line)


@pytest.fixture()
def verifies(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Every Ed25519 verification the handshake module ran, by label."""
    seen: list[str] = []
    original = hs_mod._verify_signature  # noqa: SLF001

    def spy(public_key: str, signature: str, digest: bytes, **kwargs: Any) -> None:
        seen.append(kwargs.get("label", b"lop-mesh-auth-v1\x00").decode().rstrip("\x00"))
        original(public_key, signature, digest, **kwargs)

    monkeypatch.setattr(hs_mod, "_verify_signature", spy)
    return seen


def test_a_proven_removed_member_gets_one_verifiable_notice(net: Net) -> None:
    hs, raw, error = net.dial(net.b_identity)
    assert error is None
    frame = _frame(raw)
    with pytest.raises(hs_mod.MembershipEnded) as ended:
        hs.read_welcome(cast(wire.FrameReader, _Lines([frame])), 0.0, net.expect())
    assert ended.value.code == "membership_ended"
    assert ended.value.epoch == 2 and ended.value.signer == net.a_identity.device_id


def test_the_notice_is_fixed_and_minimal(net: Net) -> None:
    hs, raw, _ = net.dial(net.b_identity)
    frame = _frame(raw)
    # No reason text, no remover beyond whom B dialled, no member list.
    assert set(frame) == {"net", "v", "epoch", "sig"}
    assert frame["net"] == "removed" and frame["epoch"] == 2
    for word in ("remov", "admin", "owner", net.a_identity.device_id, net.a_identity.name):
        assert word not in frame["sig"]


@pytest.mark.parametrize(
    "case", ["tombstoned_id_random_key", "never_member", "active_id_wrong_key"]
)
def test_non_proven_dialers_walk_one_path_and_get_silence(
    net: Net, verifies: list[str], case: str
) -> None:
    """Until the key proof succeeds a tombstoned id and a random key look identical.

    Non-members (a tombstoned id with a key nobody stored for it, a never-member)
    walk the same branch (``_probe_tombstone``): exactly ONE auth-label
    verification and the same ``not_a_member`` refusal. A live id with the wrong
    key never reaches that branch (step 7, ``bad_signature``, unchanged). All three
    get NOTHING on the socket: byte-for-byte today's silent close.
    """
    if case == "tombstoned_id_random_key":
        _hs, raw, _ = net.dial(net.c_identity, claim=net.b_identity.device_id)
        expected = hs_mod.REASON_MEMBER
    elif case == "never_member":
        _hs, raw, _ = net.dial(net.c_identity)
        expected = hs_mod.REASON_MEMBER
    else:
        _hs, raw, _ = net.dial(net.c_identity, claim=net.d_identity.device_id)
        expected = hs_mod.REASON_SIGNATURE
    assert raw == b"", raw
    assert verifies == ["lop-mesh-auth-v1"], verifies
    assert [code for code, _ in net.refusals] == [expected]
    assert net.server._removal_notices == {}  # noqa: SLF001 — no notice was even considered


def test_proven_and_unproven_take_the_same_refusal_and_verification_count(
    net: Net, verifies: list[str]
) -> None:
    """The ONLY difference a proven key makes is the reply frame, decided after the refusal."""
    _hs, unproven_raw, _ = net.dial(net.c_identity, claim=net.b_identity.device_id)
    unproven = list(verifies)
    verifies.clear()
    _hs, proven_raw, _ = net.dial(net.b_identity)
    assert unproven == verifies == ["lop-mesh-auth-v1"]
    # Same refusal code AND sentence for both: the audit cannot tell them apart either.
    assert len(net.refusals) == 2 and net.refusals[0] == net.refusals[1], net.refusals
    assert net.refusals[0][0] == hs_mod.REASON_MEMBER
    assert unproven_raw == b"" and _frame(proven_raw)["net"] == "removed"


def test_a_second_dial_inside_the_window_gets_silence(net: Net) -> None:
    _hs, first, _ = net.dial(net.b_identity)
    assert _frame(first)["net"] == "removed"
    _hs, second, _ = net.dial(net.b_identity)
    assert second == b"", "the rate limit let a second notice out"
    # Past the window the next proven dial is answered again.
    with net.server._removal_notices_lock:  # noqa: SLF001
        for key in list(net.server._removal_notices):  # noqa: SLF001
            net.server._removal_notices[key] -= relay.REMOVAL_NOTICE_INTERVAL_S + 1  # noqa: SLF001
    _hs, third, _ = net.dial(net.b_identity)
    assert _frame(third)["net"] == "removed"


def test_a_tombstone_past_seven_days_gets_silence(net: Net) -> None:
    assert relay.REMOVAL_NOTICE_WINDOW_S >= 7 * 24 * 3600
    with store.mutate(net.network_id, net.root) as record:
        row = record.member(net.b_identity.device_id)
        assert row is not None and row.removed_at is not None
        row.removed_at = time.time() - relay.REMOVAL_NOTICE_WINDOW_S - 60
        store.save(record, net.root)
    _hs, raw, _ = net.dial(net.b_identity)
    assert raw == b""
    # The row itself is still kept (90-day prune), only the notice window ended.
    assert store.load(net.network_id, net.root).member(net.b_identity.device_id) is not None


def test_a_device_that_left_is_never_noticed(net: Net) -> None:
    with store.mutate(net.network_id, net.root) as record:
        row = record.member(net.b_identity.device_id)
        assert row is not None
        row.removed_by = net.b_identity.device_id
        store.save(record, net.root)
    _hs, raw, _ = net.dial(net.b_identity)
    assert raw == b""


def test_the_tombstone_holds_no_secret(net: Net) -> None:
    row = store.load(net.network_id, net.root).member(net.b_identity.device_id)
    assert row is not None and row.public_key == net.b_identity.public_key
    blob = json.dumps(row.to_json() if hasattr(row, "to_json") else row.__dict__, default=str)
    assert net.secret_e1 not in blob


# ---------------------------------------------------------------------------
# Dialer: only a real notice from a current member, on THIS handshake, counts
# ---------------------------------------------------------------------------


class _Lines:
    """A ``FrameReader`` stand-in that returns pre-read frames."""

    def __init__(self, frames: list[dict[str, Any]]) -> None:
        self._frames = list(frames)

    def read_line(self, _deadline: float | None) -> dict[str, Any]:
        return self._frames.pop(0)


def _fresh_notice(net: Net) -> tuple[hs_mod.Handshake, dict[str, Any]]:
    hs, raw, _ = net.dial(net.b_identity)
    return hs, _frame(raw)


def test_dialer_refuses_a_notice_replayed_from_another_handshake(net: Net) -> None:
    _old_hs, old_frame = _fresh_notice(net)
    net.server._removal_notices.clear()  # noqa: SLF001
    new_hs, _new_frame = _fresh_notice(net)
    with pytest.raises(types.HandshakeRefusal) as refused:
        new_hs.verify_removal_notice(old_frame, net.expect())
    assert not isinstance(refused.value, hs_mod.MembershipEnded)
    assert refused.value.code == hs_mod.REASON_SIGNATURE


def test_dialer_refuses_a_notice_signed_by_an_unknown_key(net: Net) -> None:
    hs, frame = _fresh_notice(net)
    digest = hs_mod.removal_digest(
        hs.transcript_bytes,
        network_id=net.network_id,
        removed_device_id=net.b_identity.device_id,
        epoch=2,
    )
    frame["sig"] = wire.b64u(net.c_identity.sign(hs_mod.REMOVED_SIG_LABEL + digest))
    with pytest.raises(types.HandshakeRefusal) as refused:
        hs.verify_removal_notice(frame, net.expect())
    assert not isinstance(refused.value, hs_mod.MembershipEnded)
    assert refused.value.code == hs_mod.REASON_SIGNATURE


def test_dialer_refuses_a_notice_from_a_non_member(net: Net) -> None:
    hs, frame = _fresh_notice(net)
    nobody = hs_mod.DialerExpectation(
        device_id=net.a_identity.device_id, public_key_for=lambda _d: None
    )
    with pytest.raises(types.HandshakeRefusal) as refused:
        hs.verify_removal_notice(frame, nobody)
    assert refused.value.code == hs_mod.REASON_MEMBER


def test_dialer_refuses_a_notice_from_a_removed_signer(net: Net) -> None:
    """A tombstoned on B's own record: the relay's real key reader returns no key."""
    hs, frame = _fresh_notice(net)
    b_view = store.load(net.network_id, net.root)
    a_row = b_view.member(net.a_identity.device_id)
    assert a_row is not None
    a_row.removed_at = time.time()
    a_row.lifecycle = "expired"
    expect = hs_mod.DialerExpectation(
        device_id=net.a_identity.device_id,
        public_key_for=relay._active_key_reader(b_view),  # noqa: SLF001
    )
    with pytest.raises(types.HandshakeRefusal) as refused:
        hs.verify_removal_notice(frame, expect)
    assert not isinstance(refused.value, hs_mod.MembershipEnded)
    assert refused.value.code == hs_mod.REASON_MEMBER


def test_dialer_refuses_a_notice_from_a_device_it_did_not_dial(net: Net) -> None:
    hs, frame = _fresh_notice(net)
    other = hs_mod.DialerExpectation(
        device_id=net.c_identity.device_id, public_key_for=lambda _d: net.a_identity.public_key
    )
    with pytest.raises(types.HandshakeRefusal) as refused:
        hs.verify_removal_notice(frame, other)
    assert refused.value.code == hs_mod.REASON_PEER_IDENTITY


def test_dialer_refuses_a_notice_whose_epoch_is_not_newer(net: Net) -> None:
    hs, frame = _fresh_notice(net)
    hs.epoch = 2  # this device already holds the epoch the notice names
    with pytest.raises(types.HandshakeRefusal) as refused:
        hs.verify_removal_notice(frame, net.expect())
    assert refused.value.code == hs_mod.REASON_EPOCH


@pytest.mark.parametrize(
    "mutate",
    [
        lambda f: f.update({"reason": "lost laptop"}),
        lambda f: f.update({"members": []}),
        lambda f: f.pop("sig"),
        lambda f: f.update({"sig": "AAAA"}),
        lambda f: f.update({"epoch": "2"}),
        lambda f: f.update({"epoch": 3}),
    ],
)
def test_dialer_refuses_a_malformed_or_altered_notice(net: Net, mutate: Any) -> None:
    hs, frame = _fresh_notice(net)
    mutate(frame)
    with pytest.raises(types.HandshakeRefusal) as refused:
        hs.verify_removal_notice(frame, net.expect())
    assert not isinstance(refused.value, hs_mod.MembershipEnded)


def test_a_notice_signature_never_verifies_as_a_welcome_proof_or_auth(net: Net) -> None:
    """Domain separation: the notice signature is useless under the other two labels."""
    hs, frame = _fresh_notice(net)
    digest = hs_mod.removal_digest(
        hs.transcript_bytes,
        network_id=net.network_id,
        removed_device_id=net.b_identity.device_id,
        epoch=2,
    )
    for label in (hs_mod.ACCEPT_SIG_LABEL, b"lop-mesh-auth-v1\x00"):
        with pytest.raises(types.HandshakeRefusal):
            hs_mod._verify_signature(  # noqa: SLF001
                net.a_identity.public_key, frame["sig"], digest, label=label
            )


# ---------------------------------------------------------------------------
# Dialer relay: only a VERIFIED notice reaches the drop path
# ---------------------------------------------------------------------------


class BSide:
    """B's real relay and record (A active in it), dialling a scripted listener."""

    def __init__(self, root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        b_root = root / "b-side"
        self.root = b_root
        self.b_identity = identity.mint(b_root, name="removed-b")
        self.a_identity = identity.mint(root / "a-side", name="owner-a")
        self.c_identity = identity.mint(root / "c-side", name="stranger-c")
        monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(b_root))
        self.server = relay.RelayServer(
            settings=relay.NetworkSettings(port=0, listen_address="127.0.0.1"),
            identity=self.b_identity,
            audit=audit_mod.AuditLog(b_root),
        )
        record = types.NetworkRecord(
            network_id=store.new_network_id(),
            name="notice-net",
            created_by=self.a_identity.device_id,
            self_device_id=self.b_identity.device_id,
            self_role="drive",
            self_capabilities=sorted(types.capabilities_for_role("drive")),
            listen={"address": "127.0.0.1", "port": 0, "advertised": []},
        )
        self.secret = wire.b64u(token_bytes(32))
        state = types.SecretState(network_id=record.network_id, epoch=1, secret=self.secret)
        for ident, role in ((self.a_identity, "admin"), (self.b_identity, "drive")):
            relay.admit(
                record,
                device_id=ident.device_id,
                public_key=ident.public_key,
                name=ident.name,
                role=role,
                capabilities=sorted(types.capabilities_for_role(role)),
                added_by=self.a_identity.device_id,
                added_via="invite",
                root=b_root,
                persist=False,
            )
        store.save(record, b_root)
        store.save_secrets(state, b_root)
        self.network_id = record.network_id

    def dial_scripted(self, signer: Any) -> str:
        """B dials a listener that answers its auth with a notice signed by ``signer``."""
        listener = socket.create_server(("127.0.0.1", 0))
        errors: list[BaseException] = []

        def answer() -> None:
            try:
                conn, _ = listener.accept()
                with conn:
                    reader = wire.FrameReader(conn)
                    hs = hs_mod.Handshake.new(
                        role="listener",
                        identity=self.a_identity,
                        network_id=self.network_id,
                        # B's epoch, so the hello is accepted; the notice names 2.
                        epoch=1,
                        instance_id="i_scripted",
                        session_protocol=relay._session_protocol(),  # noqa: SLF001
                    )
                    hs.read_hello(reader, wire.deadline_in(5.0))
                    hs.send_challenge(conn, hs_mod.StaticPolicy(networks={self.network_id: True}))
                    auth = reader.read_line(wire.deadline_in(5.0))
                    hs.auth_core = {k: v for k, v in auth.items() if k not in ("sig", "mac")}
                    digest = hs_mod.removal_digest(
                        hs.transcript_bytes,
                        network_id=self.network_id,
                        removed_device_id=self.b_identity.device_id,
                        epoch=2,
                    )
                    sig = signer.sign(hs_mod.REMOVED_SIG_LABEL + digest)
                    hs_mod._send_frame(  # noqa: SLF001
                        conn,
                        {
                            "net": "removed",
                            "v": hs_mod.LINK_VERSION,
                            "epoch": 2,
                            "sig": wire.b64u(sig),
                        },
                    )
            except BaseException as exc:  # noqa: BLE001 — surfaced to the cell
                errors.append(exc)

        thread = threading.Thread(target=answer, daemon=True)
        thread.start()
        try:
            host, port = listener.getsockname()[:2]
            _link, code = self.server.dial(
                self.network_id,
                host=f"{host}:{port}",
                epoch=1,
                timeout_s=5.0,
                expected_device=self.a_identity.device_id,
            )
        finally:
            thread.join(10)
            listener.close()
        assert not errors, errors
        return code

    def self_active(self) -> bool:
        me = store.load(self.network_id, self.root).self_member()
        return bool(me is not None and me.active)


@pytest.fixture()
def b_side(root: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    rig = BSide(root, monkeypatch)
    try:
        yield rig
    finally:
        rig.server.stop()


def test_relay_applies_a_verified_notice_once_and_reduces_only(b_side: BSide) -> None:
    before = store.load(b_side.network_id, b_side.root)
    assert b_side.dial_scripted(b_side.a_identity) == "membership_ended"
    after = store.load(b_side.network_id, b_side.root)
    me = after.self_member()
    assert me is not None and not me.active and me.removed_by == b_side.a_identity.device_id
    assert after.stale == "refused_by_peers" and after.epoch == before.epoch
    assert store.require_secrets(b_side.network_id, b_side.root).secret == b_side.secret
    b_side.server.audit.flush()
    assert len(_audit_rows(b_side.root, "membership_ended")) == 1


def test_relay_ignores_a_forged_notice_from_a_non_member(b_side: BSide) -> None:
    code = b_side.dial_scripted(b_side.c_identity)
    assert code == hs_mod.REASON_SIGNATURE
    assert b_side.self_active(), "a forged notice reached the drop path"
    b_side.server.audit.flush()
    assert _audit_rows(b_side.root, "membership_ended") == []
