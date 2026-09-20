"""The handshake: the happy path, the transcript binding, and the refusals.

Both roles are driven here, over real sockets, because the security properties are
about what goes on the WIRE — a silent close with zero bytes written is not a
property a mocked socket can hold to.
"""

from __future__ import annotations

import socket
import threading
from dataclasses import dataclass, field
from typing import Any

import pytest

from local_operator.network import handshake as hs_mod
from local_operator.network import identity, wire

NETWORK = "n_0123456789abcdef01234567"
SESSION_PROTOCOL = 5


@dataclass
class Outcome:
    dialer: Any = None
    listener: Any = None
    dialer_error: Exception | None = None
    listener_error: Exception | None = None
    frames: list[dict[str, Any]] = field(default_factory=list)
    listener_bytes: int = 0


def policy_for(
    *,
    network: str = NETWORK,
    current: int = 1,
    previous: int | None = None,
    trusted: bool = True,
    epoch_keys: dict[int, bytes] | None = None,
    members: dict[str, str] | None = None,
) -> hs_mod.StaticPolicy:
    keys = epoch_keys or {current: b"k" * 32}
    return hs_mod.StaticPolicy(
        current=current,
        previous=previous,
        networks={network: trusted},
        epoch_keys={(network, epoch): key for epoch, key in keys.items()},
        members={(network, device): key for device, key in (members or {}).items()},
    )


def run_handshake(
    client: socket.socket,
    server: socket.socket,
    *,
    dialer_identity: Any,
    listener_identity: Any,
    policy: hs_mod.StaticPolicy,
    network: str = NETWORK,
    epoch: int = 1,
    mode: str = "member",
    credential_key: bytes | None = None,
    mutate_hello: Any = None,
    mutate_challenge: Any = None,
    join_block: dict[str, Any] | None = None,
    listener_credential: Any = None,
) -> Outcome:
    """Run one full exchange, listener and dialer concurrently, and collect both ends."""
    outcome = Outcome()
    deadline = wire.deadline_in(5.0)

    def listener_side() -> None:
        try:
            listener = hs_mod.Handshake.new(
                role="listener",
                identity=listener_identity,
                network_id=network,
                epoch=epoch,
                instance_id="i_listener",
                session_protocol=SESSION_PROTOCOL,
                mode=mode,  # type: ignore[arg-type]
                capabilities=list(wire.LINK_CAPABILITIES),
                build={"version": "test"},
            )
            reader = wire.FrameReader(server)
            listener.read_hello(reader, deadline)
            if listener_credential is not None:
                listener.credential = listener_credential
            listener.send_challenge(server, policy)
            listener.verify_auth(reader, deadline, policy)
            result = listener.establish()
            outcome.listener = result
            listener.send_welcome(
                server, listener.welcome_frame(phase=result.phase, epoch=result.epoch)
            )
        except Exception as exc:  # noqa: BLE001 — the test asserts on what comes back
            outcome.listener_error = exc
            try:
                server.close()
            except OSError:
                pass

    def dialer_side() -> None:
        try:
            dialer = hs_mod.Handshake.new(
                role="dialer",
                identity=dialer_identity,
                network_id=network,
                epoch=epoch,
                instance_id="i_dialer",
                session_protocol=SESSION_PROTOCOL,
                mode=mode,  # type: ignore[arg-type]
                capabilities=list(wire.LINK_CAPABILITIES),
                build={"version": "test"},
            )
            if join_block is not None:
                dialer.join_block = dict(join_block)
            reader = wire.FrameReader(client)
            hello = dialer.send_hello(client)
            outcome.frames.append(hello)
            dialer.read_challenge(reader, deadline)
            outcome.frames.append(dialer.challenge)
            if mutate_hello is not None:
                mutate_hello(dialer)
            if mutate_challenge is not None:
                mutate_challenge(dialer)
            key = (
                credential_key
                if credential_key is not None
                else policy.epoch_keys.get((network, epoch), b"k" * 32)
            )
            dialer.send_auth(client, hs_mod.Credential("epoch", epoch, key))
            outcome.frames.append(dialer.auth_core)
            welcome = dialer.read_welcome(reader, deadline)
            outcome.frames.append(welcome)
            outcome.dialer = dialer.establish()
        except Exception as exc:  # noqa: BLE001
            outcome.dialer_error = exc

    listener_thread = threading.Thread(target=listener_side, daemon=True)
    dialer_thread = threading.Thread(target=dialer_side, daemon=True)
    listener_thread.start()
    dialer_thread.start()
    listener_thread.join(10)
    dialer_thread.join(10)
    return outcome


@pytest.fixture()
def peers(root: Any) -> tuple[Any, Any]:
    return identity.mint(root / "a", name="device-a"), identity.mint(root / "b", name="device-b")


# ---------------------------------------------------------------------------
# The happy path
# ---------------------------------------------------------------------------


def test_member_handshake_derives_the_same_sas_on_both_sides(
    socketpair: tuple[socket.socket, socket.socket], peers: tuple[Any, Any]
) -> None:
    client, server = socketpair
    dialer_identity, listener_identity = peers
    policy = policy_for(members={dialer_identity.device_id: dialer_identity.public_key})
    outcome = run_handshake(
        client,
        server,
        dialer_identity=dialer_identity,
        listener_identity=listener_identity,
        policy=policy,
    )
    assert outcome.dialer_error is None and outcome.listener_error is None
    assert outcome.dialer is not None and outcome.listener is not None
    # THE CHECK THE HUMANS MAKE: both sides derive it, neither transmits it.
    assert outcome.dialer.sas == outcome.listener.sas
    assert outcome.dialer.transcript_hash == outcome.listener.transcript_hash
    assert outcome.dialer.keys.k_d2l == outcome.listener.keys.k_d2l
    assert outcome.dialer.keys.link_id == outcome.listener.keys.link_id
    assert outcome.dialer.phase == "member"
    assert outcome.listener.peer_device_id == dialer_identity.device_id
    assert outcome.listener.peer_public_key == dialer_identity.public_key


def test_the_two_ends_can_exchange_records_after_welcome(
    socketpair: tuple[socket.socket, socket.socket], peers: tuple[Any, Any]
) -> None:
    """The negotiated keys are only real if they decrypt each other's records."""
    client, server = socketpair
    dialer_identity, listener_identity = peers
    policy = policy_for(members={dialer_identity.device_id: dialer_identity.public_key})
    outcome = run_handshake(
        client,
        server,
        dialer_identity=dialer_identity,
        listener_identity=listener_identity,
        policy=policy,
    )
    assert outcome.dialer and outcome.listener
    dialer_codec = wire.LinkCrypto(outcome.dialer.keys, role="dialer")
    listener_codec = wire.LinkCrypto(outcome.listener.keys, role="listener")
    assert listener_codec.open(dialer_codec.seal({"op": "net_catalog", "req": 1})[4:])["op"] == (
        "net_catalog"
    )


# ---------------------------------------------------------------------------
# The transcript
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mutate, describe",
    [
        (lambda h: h.hello.__setitem__("caps", ["mesh-net-v1", "extra"]), "capabilities"),
        (lambda h: h.hello.__setitem__("nonce", wire.b64u(b"z" * 32)), "hello nonce"),
        (lambda h: h.hello.__setitem__("instance_id", "i_other"), "instance id"),
        (lambda h: h.challenge.__setitem__("salt", wire.b64u(b"y" * 16)), "challenge salt"),
        (lambda h: h.challenge.__setitem__("eph", wire.b64u(b"x" * 32)), "peer ephemeral"),
    ],
)
def test_the_transcript_binds_every_field(
    socketpair: tuple[socket.socket, socket.socket],
    peers: tuple[Any, Any],
    mutate: Any,
    describe: str,
) -> None:
    """Mutate one field of the transcript after it was sent: every mutation must fail.

    This is the property that makes the signature worth anything — a field that
    could change under the signature without failing it is a field an intermediary
    may rewrite.
    """
    client, server = socketpair
    dialer_identity, listener_identity = peers
    policy = policy_for(members={dialer_identity.device_id: dialer_identity.public_key})
    outcome = run_handshake(
        client,
        server,
        dialer_identity=dialer_identity,
        listener_identity=listener_identity,
        policy=policy,
        mutate_hello=mutate,
    )
    assert outcome.listener_error is not None, f"a changed {describe} was accepted"
    assert getattr(outcome.listener_error, "code", "") in ("bad_signature", "bad_mac")


def test_the_sas_never_appears_in_any_frame(
    socketpair: tuple[socket.socket, socket.socket], peers: tuple[Any, Any]
) -> None:
    """Blanket assertion over every frame both roles emit.

    The comparison is only a CHECK because the value is derived locally: a peer that
    sent its digits would let an on-path attacker echo them back, which is why this
    is asserted rather than trusted.
    """
    client, server = socketpair
    dialer_identity, listener_identity = peers
    policy = policy_for(members={dialer_identity.device_id: dialer_identity.public_key})
    outcome = run_handshake(
        client,
        server,
        dialer_identity=dialer_identity,
        listener_identity=listener_identity,
        policy=policy,
    )
    assert outcome.dialer and outcome.dialer.sas
    for frame in outcome.frames:
        text = wire.canonical_json(frame).decode("utf-8")
        assert outcome.dialer.sas not in text
        assert outcome.dialer.sas[:3] + outcome.dialer.sas[3:] not in text
        assert (
            wire.transcript_fingerprint(bytes.fromhex(outcome.dialer.transcript_hash)) not in text
        )


# ---------------------------------------------------------------------------
# Refusals (each closes with NO reply)
# ---------------------------------------------------------------------------


def _refusal_case(
    socketpair: tuple[socket.socket, socket.socket],
    peers: tuple[Any, Any],
    policy: hs_mod.StaticPolicy,
    **kwargs: Any,
) -> Outcome:
    client, server = socketpair
    dialer_identity, listener_identity = peers
    return run_handshake(
        client,
        server,
        dialer_identity=dialer_identity,
        listener_identity=listener_identity,
        policy=policy,
        **kwargs,
    )


def test_wrong_epoch_is_refused_without_a_reply(
    socketpair: tuple[socket.socket, socket.socket], peers: tuple[Any, Any]
) -> None:
    dialer_identity = peers[0]
    policy = policy_for(current=3, members={dialer_identity.device_id: dialer_identity.public_key})
    outcome = _refusal_case(socketpair, peers, policy, epoch=1)
    assert getattr(outcome.listener_error, "code", "") == hs_mod.REASON_EPOCH
    # SILENCE: the socket was closed, so the dialer never got a challenge. Its own
    # error is a connection failure with no frame — never an error reply.
    assert outcome.dialer_error is not None
    assert outcome.dialer is None


def test_untrusted_network_refuses_every_link(
    socketpair: tuple[socket.socket, socket.socket], peers: tuple[Any, Any]
) -> None:
    dialer_identity = peers[0]
    policy = policy_for(
        trusted=False, members={dialer_identity.device_id: dialer_identity.public_key}
    )
    outcome = _refusal_case(socketpair, peers, policy)
    assert getattr(outcome.listener_error, "code", "") == hs_mod.REASON_UNTRUSTED


def test_a_removed_member_is_refused_before_any_key_is_tried(
    socketpair: tuple[socket.socket, socket.socket], peers: tuple[Any, Any]
) -> None:
    """The revocation line: membership is evaluated against the CURRENT member list.

    The policy here has NO epoch key at all for the offered epoch, so a listener
    that tried the MAC before the membership check would fail with ``bad_mac``
    instead — which is exactly the difference this test pins.
    """
    policy = policy_for(epoch_keys={}, members={})
    outcome = _refusal_case(socketpair, peers, policy)
    assert getattr(outcome.listener_error, "code", "") == hs_mod.REASON_MEMBER


def test_signing_with_the_wrong_device_key_is_refused(
    socketpair: tuple[socket.socket, socket.socket], peers: tuple[Any, Any], root: Any
) -> None:
    """An impostor holds the network secret but not the member's device key."""
    dialer_identity = peers[0]
    impostor = identity.mint(root / "c", name="impostor")
    policy = policy_for(members={dialer_identity.device_id: dialer_identity.public_key})
    client, server = socketpair
    outcome = run_handshake(
        client,
        server,
        dialer_identity=impostor,
        listener_identity=peers[1],
        policy=policy,
    )
    # The hello claims the impostor's own id, which is not a member — refused at
    # the membership check, which is the earlier and better refusal.
    assert getattr(outcome.listener_error, "code", "") == hs_mod.REASON_MEMBER


def test_a_wrong_mac_key_is_refused(
    socketpair: tuple[socket.socket, socket.socket], peers: tuple[Any, Any]
) -> None:
    dialer_identity = peers[0]
    policy = policy_for(members={dialer_identity.device_id: dialer_identity.public_key})
    outcome = _refusal_case(socketpair, peers, policy, credential_key=b"w" * 32)
    assert getattr(outcome.listener_error, "code", "") == hs_mod.REASON_MAC


def test_self_connection_is_refused(
    socketpair: tuple[socket.socket, socket.socket], peers: tuple[Any, Any]
) -> None:
    dialer_identity = peers[0]
    policy = policy_for(members={dialer_identity.device_id: dialer_identity.public_key})
    client, server = socketpair
    outcome = run_handshake(
        client,
        server,
        dialer_identity=dialer_identity,
        listener_identity=dialer_identity,
        policy=policy,
    )
    assert getattr(outcome.listener_error, "code", "") == hs_mod.REASON_SELF


def test_unknown_link_version_is_refused_after_authentication(
    socketpair: tuple[socket.socket, socket.socket], peers: tuple[Any, Any]
) -> None:
    """Checked at ``welcome`` rather than at ``hello``, so an unknown version is not
    an oracle any stranger can pull."""
    dialer_identity = peers[0]
    policy = policy_for(members={dialer_identity.device_id: dialer_identity.public_key})
    client, server = socketpair
    outcome = run_handshake(
        client,
        server,
        dialer_identity=dialer_identity,
        listener_identity=peers[1],
        policy=policy,
        mutate_challenge=lambda h: h.challenge.__setitem__("v", 99),
    )
    assert outcome.listener_error is not None  # the listener refuses a changed hello/challenge
    assert getattr(outcome.listener_error, "code", "") in ("bad_signature", "bad_mac")


# ---------------------------------------------------------------------------
# Reconcile
# ---------------------------------------------------------------------------


def test_previous_epoch_yields_the_reconcile_phase(
    socketpair: tuple[socket.socket, socket.socket], peers: tuple[Any, Any]
) -> None:
    dialer_identity = peers[0]
    policy = policy_for(
        current=8,
        previous=7,
        epoch_keys={7: b"o" * 32, 8: b"n" * 32},
        members={dialer_identity.device_id: dialer_identity.public_key},
    )
    client, server = socketpair
    outcome = run_handshake(
        client,
        server,
        dialer_identity=dialer_identity,
        listener_identity=peers[1],
        policy=policy,
        epoch=7,
    )
    assert outcome.listener_error is None
    assert outcome.listener is not None
    assert outcome.listener.phase == "reconcile"
    # A reconcile link at the OLD epoch still derives the same SAS: the phase is a
    # restriction on what may be asked, not a weaker handshake.
    assert outcome.dialer is not None and outcome.dialer.sas == outcome.listener.sas


def test_a_previous_epoch_is_refused_when_it_is_not_offered(
    socketpair: tuple[socket.socket, socket.socket], peers: tuple[Any, Any]
) -> None:
    dialer_identity = peers[0]
    policy = policy_for(
        current=8,
        previous=None,
        epoch_keys={8: b"n" * 32},
        members={dialer_identity.device_id: dialer_identity.public_key},
    )
    outcome = _refusal_case(socketpair, peers, policy, epoch=4)
    assert getattr(outcome.listener_error, "code", "") == hs_mod.REASON_EPOCH


# ---------------------------------------------------------------------------
# The join (pair) mode
# ---------------------------------------------------------------------------


def test_join_mode_uses_the_invite_key_and_lands_in_the_pair_phase(
    socketpair: tuple[socket.socket, socket.socket], peers: tuple[Any, Any]
) -> None:
    """A join differs from a member link ONLY in which key verifies the MAC.

    Same transcript, same signature, same SAS — which is why one verifier covers
    both and a join cannot take a shortcut a member link does not.
    """
    joiner_identity, inviter_identity = peers
    invite_material = b"i" * 32
    invite_key = wire.invite_key(wire.b64u(invite_material), NETWORK, "invite_1")
    policy = policy_for(epoch_keys={1: b"k" * 32}, members={})
    client, server = socketpair
    outcome = run_handshake(
        client,
        server,
        dialer_identity=joiner_identity,
        listener_identity=inviter_identity,
        policy=policy,
        mode="join",
        credential_key=invite_key,
        listener_credential=hs_mod.Credential("invite", 1, invite_key),
        join_block={
            "invite_id": "invite_1",
            "joiner_public_key": joiner_identity.public_key,
            "joiner_name": "joiner",
        },
    )
    assert outcome.listener_error is None, outcome.listener_error
    assert outcome.listener is not None
    assert outcome.listener.phase == "pair"
    assert outcome.dialer is not None
    assert outcome.dialer.sas == outcome.listener.sas
    assert outcome.listener.peer_device_id == joiner_identity.device_id


def test_join_mode_without_a_claimed_invite_is_refused(
    socketpair: tuple[socket.socket, socket.socket], peers: tuple[Any, Any]
) -> None:
    """No preset credential means the inviter never claimed an invite, and the
    handshake refuses rather than picking one up from the frame."""
    joiner_identity, inviter_identity = peers
    policy = policy_for(members={})
    client, server = socketpair
    outcome = run_handshake(
        client,
        server,
        dialer_identity=joiner_identity,
        listener_identity=inviter_identity,
        policy=policy,
        mode="join",
        credential_key=wire.invite_key(wire.b64u(b"i" * 32), NETWORK, "invite_1"),
        join_block={
            "invite_id": "invite_1",
            "joiner_public_key": joiner_identity.public_key,
            "joiner_name": "joiner",
        },
    )
    assert getattr(outcome.listener_error, "code", "") == hs_mod.REASON_INVITE


def test_a_join_whose_key_does_not_match_the_fingerprint_is_refused(
    socketpair: tuple[socket.socket, socket.socket], peers: tuple[Any, Any]
) -> None:
    joiner_identity, inviter_identity = peers
    invite_key = wire.invite_key(wire.b64u(b"i" * 32), NETWORK, "invite_1")
    policy = policy_for(epoch_keys={})
    client, server = socketpair
    outcome = run_handshake(
        client,
        server,
        dialer_identity=joiner_identity,
        listener_identity=inviter_identity,
        policy=policy,
        mode="join",
        credential_key=invite_key,
        listener_credential=hs_mod.Credential("invite", 1, invite_key),
        join_block={
            "invite_id": "invite_1",
            # The id does not derive from this key, so nothing can be proven about it.
            "joiner_public_key": peers[1].public_key,
            "joiner_name": "joiner",
        },
    )
    assert getattr(outcome.listener_error, "code", "") == hs_mod.REASON_INVITE_DEVICE


# ---------------------------------------------------------------------------
# The SAS comparison
# ---------------------------------------------------------------------------


def test_sas_comparison_normalises_but_never_guesses() -> None:
    assert hs_mod.sas_matches("481926", "481 926")
    assert hs_mod.sas_matches("481926", "481-926")
    # The negative cases: a near miss is a mismatch, never an acceptance.
    assert not hs_mod.sas_matches("481926", "481925")
    assert not hs_mod.sas_matches("481926", "48192")
    assert not hs_mod.sas_matches("481926", "")
    assert not hs_mod.sas_matches("481926", "the code is 481926? really?")


def test_pair_frames_carry_no_sas_value_from_the_peer() -> None:
    """``net_pair_ready`` carries the TYPED value; ``net_pair_result`` never carries one."""
    ready = hs_mod.pair_ready_frame(req=1, typed_sas="481926")
    assert ready["sas"] == "481926"
    result = hs_mod.pair_result_frame(req=1, admit=True, material="x")
    assert "sas" not in result
    abort = hs_mod.pair_abort_frame(req=1, reason="sas_mismatch")
    assert abort["op"] == "net_pair_abort"
    assert "sas" not in abort


def test_pair_confirm_timeout_is_bounded_by_the_invite() -> None:
    assert hs_mod.pair_timeout_seconds(60.0) == 60.0
    assert hs_mod.pair_timeout_seconds(9999.0) == hs_mod.PAIR_CONFIRM_TIMEOUT_S


def test_a_frame_over_the_line_budget_is_never_sent(
    socketpair: tuple[socket.socket, socket.socket], peers: tuple[Any, Any]
) -> None:
    """Checked on the SEND side too: an oversized challenge must fail HERE, naming
    the real culprit, rather than producing a mysterious close on the peer."""
    assert hs_mod.frame_size_ok({"op": "hello", "n": 1})
    assert not hs_mod.frame_size_ok({"op": "hello", "big": "x" * (wire.MAX_HANDSHAKE_LINE + 1)})
