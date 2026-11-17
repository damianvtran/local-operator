"""The DIALER must authenticate the LISTENER (mesh trust audit G1-G4).

WHY THIS FILE EXISTS. The listener proves nothing to the dialer on the member
handshake: ``challenge`` carries no signature or MAC, the dialer copies its
``device_id`` unverified (``Handshake.read_challenge``), ``welcome`` sits outside the
transcript and ``read_welcome`` derives link keys without checking anything. The
relay's dial path (``RelayServer.dial``) then admits the link under the CLAIMED id
and never compares it with ``expected_device``. A listener holding neither the
claimed member's private key nor the epoch key therefore becomes that member on
the dialer, with everything a member link carries: the member-table pull it answers
(G2), the next epoch secret on ``member rm`` (G3), and the endpoints it names in
``welcome``, persisted onto the victim's row (G4).

THE SHAPE. One real dialer — a ``RelayServer`` with a persisted network record, the
same object ``lop network serve`` runs — dials a loopback socket served by a FAKE
listener built from the real ``Handshake`` code (the audit's ``mitm_spike.py``). The
fake either skips every check (an impostor with no key and no epoch) or runs the
genuine listener half under some member's real identity. Every cell asserts what the
dialer must REFUSE; ``test_a_genuine_listener_is_still_admitted`` is the negative
control proving this harness can establish an honest link at all, so a refusal in
another cell is the dialer's decision and not a broken rig.

Synthetic keys only: identities are minted into ``tmp_path``; no live mesh.
"""

from __future__ import annotations

import socket
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from secrets import token_bytes
from typing import Any

import pytest

from local_operator.network import audit as audit_mod
from local_operator.network import handshake as hs_mod
from local_operator.network import identity, relay, store, types, wire

POISONED_ENDPOINT = "203.0.113.66:4500"
HONEST_ENDPOINT_HOST = "127.0.0.1"


# ---------------------------------------------------------------------------
# The fake listener
# ---------------------------------------------------------------------------


@dataclass
class FakeListener:
    """One accepted connection, answered with the real ``Handshake`` code.

    ``policy is None`` is the IMPOSTOR: it reads the dialer's auth frame and never
    verifies it (it holds no epoch key and no member key to verify with), claims
    ``claim_device_id`` in its challenge, and welcomes the dialer anyway. With a
    ``policy`` it is a GENUINE listener for ``listener_identity`` — it verifies the
    dialer exactly as ``RelayServer._run_inbound_handshake`` does.

    ``tamper_welcome`` runs on the welcome frame AFTER ``welcome_frame`` has built
    it (i.e. after any signature a fixed build adds), modelling an on-path attacker
    rewriting the plaintext frame.

    After the handshake it serves records: every frame the dialer sends is kept in
    ``received`` and ``net_member_list`` is answered with ``member_rows``.
    """

    listener_identity: Any
    network_id: str
    claim_device_id: str
    policy: hs_mod.StaticPolicy | None = None
    endpoints: list[str] = field(default_factory=list)
    tamper_welcome: Callable[[dict[str, Any]], None] | None = None
    #: Replaces ``welcome.proof`` with whatever this returns, given the listener's
    #: handshake, the welcome frame and the dialer's raw auth frame: how a cell models
    #: a FORGED proof (own key, wrong epoch key, the dialer's own tags reflected).
    forge_proof: (
        Callable[[hs_mod.Handshake, dict[str, Any], dict[str, Any]], dict[str, str]] | None
    ) = None
    #: The build stamp the challenge reports (the dialer names it on ``listener_unproven``).
    build: dict[str, Any] = field(default_factory=dict)
    #: A rotation statement the challenge carries (``challenge.rotation``): what a
    #: ROTATED listener sends so a dialer holding its pre-rotation row can follow the
    #: hop (G1D-1). The impostor sends it verbatim, modelling a replay of the public
    #: statement by a party that does not hold the new key.
    rotation: dict[str, Any] = field(default_factory=dict)
    auth_frame: dict[str, Any] = field(default_factory=dict)
    member_rows: list[dict[str, Any]] = field(default_factory=list)
    received: list[dict[str, Any]] = field(default_factory=list)
    errors: list[BaseException] = field(default_factory=list)
    listener_keys: Any = None
    handshake_done: threading.Event = field(default_factory=threading.Event)
    _stop: threading.Event = field(default_factory=threading.Event)

    def start(self) -> tuple[str, int]:
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.bind(("127.0.0.1", 0))
        self._server.listen(1)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        host, port = self._server.getsockname()[:2]
        return str(host), int(port)

    def stop(self) -> None:
        self._stop.set()
        for sock in (getattr(self, "_conn", None), self._server):
            if sock is not None:
                try:
                    sock.close()
                except OSError:
                    pass
        self._thread.join(5)

    def _run(self) -> None:
        try:
            self._server.settimeout(10)
            conn, _ = self._server.accept()
            self._conn = conn
            reader = wire.FrameReader(conn)
            deadline = wire.deadline_in(5.0)
            hs = (
                self._impostor(conn, reader, deadline)
                if self.policy is None
                else self._genuine(conn, reader, deadline)
            )
            codec = hs.codec()
            self.listener_keys = hs.establish().keys
            self.handshake_done.set()
            self._serve(conn, reader, codec)
        except BaseException as exc:  # noqa: BLE001 — surfaced by the cells
            self.errors.append(exc)
            self.handshake_done.set()

    def _impostor(self, conn: socket.socket, reader: Any, deadline: float) -> hs_mod.Handshake:
        hs = hs_mod.Handshake.new(
            role="listener",
            identity=self.listener_identity,
            network_id=self.network_id,
            epoch=1,
            instance_id="i_fake_listener",
            session_protocol=1,
        )
        hs.hello = reader.read_line(deadline)
        hs.peer_device_id = str(hs.hello["device_id"])
        hs.self_device_id = self.claim_device_id
        hs.salt = token_bytes(16)
        hs.challenge = hs_mod.build_challenge(
            epoch=int(hs.hello.get("epoch") or 1),
            device_id=self.claim_device_id,
            instance_id="i_fake_listener",
            eph_public=hs.eph_public,
            nonce=hs.nonce,
            salt=hs.salt,
            capabilities=[hs_mod.MESH_NET_V1],
            build=dict(self.build),
        )
        if self.rotation:
            hs.challenge["rotation"] = dict(self.rotation)
        hs_mod._send_frame(conn, hs.challenge)
        auth = reader.read_line(deadline)  # never verified: no key to verify with
        self.auth_frame = dict(auth)
        hs.auth_core = {k: v for k, v in auth.items() if k not in ("sig", "mac")}
        hs.phase = "member"
        self._send_welcome(conn, hs, epoch=int(hs.auth_core.get("epoch") or 1))
        return hs

    def _genuine(self, conn: socket.socket, reader: Any, deadline: float) -> hs_mod.Handshake:
        assert self.policy is not None
        hs = hs_mod.Handshake.new(
            role="listener",
            identity=self.listener_identity,
            network_id=self.network_id,
            epoch=self.policy.current_epoch,
            instance_id="i_genuine_listener",
            session_protocol=1,
            build=dict(self.build),
        )
        hs.read_hello(reader, deadline)
        hs.self_rotation = dict(self.rotation)
        hs.send_challenge(conn, self.policy)
        self.auth_frame = dict(hs.verify_auth(reader, deadline, self.policy))
        result = hs.establish()
        self._send_welcome(conn, hs, epoch=result.epoch)
        return hs

    def _send_welcome(self, conn: socket.socket, hs: hs_mod.Handshake, *, epoch: int) -> None:
        frame = hs.welcome_frame(phase="member", epoch=epoch, endpoints=list(self.endpoints))
        if self.forge_proof is not None:
            frame["proof"] = self.forge_proof(hs, frame, self.auth_frame)
        if self.tamper_welcome is not None:
            self.tamper_welcome(frame)
        hs.send_welcome(conn, frame)

    def _serve(self, conn: socket.socket, reader: Any, codec: Any) -> None:
        while not self._stop.is_set():
            try:
                payload = reader.read_record_payload(wire.deadline_in(0.5))
            except Exception:  # noqa: BLE001 — timeout or EOF; the stop flag decides
                if self._stop.is_set():
                    return
                time.sleep(0.02)
                continue
            frame = codec.open(payload)
            self.received.append(frame)
            if frame.get("op") == "net_member_list" and frame.get("req") is not None:
                conn.sendall(
                    codec.seal(
                        {
                            "op": "ack",
                            "req": frame["req"],
                            "detail": {
                                "network_id": self.network_id,
                                "epoch": 1,
                                "members": list(self.member_rows),
                            },
                        }
                    )
                )


# ---------------------------------------------------------------------------
# The dialer: a real relay with a persisted network
# ---------------------------------------------------------------------------


@dataclass
class Mesh:
    server: relay.RelayServer
    record: types.NetworkRecord
    secret: str
    victim: Any  # admin member X: the device the dialer means to reach
    other: Any  # drive member Y: a real, different member
    attacker: Any  # no row at all
    root: Path


@pytest.fixture()
def mesh(root: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Mesh]:
    dialer_root = root / "dialer"
    dialer_identity = identity.mint(dialer_root, name="dialer")
    victim = identity.mint(root / "victim", name="victim-admin")
    other = identity.mint(root / "other", name="other-member")
    attacker = identity.mint(root / "attacker", name="attacker")
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(dialer_root))
    server = relay.RelayServer(
        settings=relay.NetworkSettings(port=0, listen_address="127.0.0.1"),
        identity=dialer_identity,
        audit=audit_mod.AuditLog(dialer_root),
    )
    assert server.root == dialer_root

    record = types.NetworkRecord(
        network_id=store.new_network_id(),
        name="trust-net",
        created_by=dialer_identity.device_id,
        self_device_id=dialer_identity.device_id,
        self_role="admin",
        self_capabilities=sorted(types.capabilities_for_role("admin")),
        listen={"address": "127.0.0.1", "port": 0, "advertised": []},
    )
    secret = wire.b64u(token_bytes(32))
    state = types.SecretState(network_id=record.network_id, epoch=1, secret=secret)
    for ident, role, via in (
        (dialer_identity, "admin", "self"),
        (victim, "admin", "invite"),
        (other, "drive", "invite"),
    ):
        relay.admit(
            record,
            device_id=ident.device_id,
            public_key=ident.public_key,
            name=ident.name,
            role=role,
            capabilities=sorted(types.capabilities_for_role(role)),
            added_by=dialer_identity.device_id,
            added_via=via,
            endpoints=["127.0.0.1:1"] if ident is not dialer_identity else None,
            root=dialer_root,
            persist=False,
        )
    store.save(record, dialer_root)
    store.save_secrets(state, dialer_root)
    try:
        yield Mesh(server, record, secret, victim, other, attacker, dialer_root)
    finally:
        server.stop()


@pytest.fixture()
def dialer_key_derivations(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Every link-key derivation the DIALER completed, by the peer id it named.

    ``Handshake.establish`` is where ``link_keys`` runs; the fake listener calls it
    too, so only ``role == "dialer"`` counts. A fixed dialer refuses BEFORE this.
    """
    seen: list[str] = []
    original = hs_mod.Handshake.establish

    def spy(self: hs_mod.Handshake) -> Any:
        result = original(self)
        if self.role == "dialer":
            seen.append(result.peer_device_id)
        return result

    monkeypatch.setattr(hs_mod.Handshake, "establish", spy)
    return seen


def _genuine_policy(mesh: Mesh) -> hs_mod.StaticPolicy:
    """What a real member of the network holds: the epoch key and the member table."""
    network_id = mesh.record.network_id
    return hs_mod.StaticPolicy(
        current=1,
        previous=None,
        networks={network_id: True},
        epoch_keys={(network_id, 1): wire.epoch_key(mesh.secret, network_id, 1)},
        members={(network_id, mesh.server.identity.device_id): mesh.server.identity.public_key},
    )


def _dial(mesh: Mesh, fake: FakeListener, *, expected_device: str) -> tuple[Any, str]:
    host, port = fake.start()
    try:
        return mesh.server.dial(
            mesh.record.network_id,
            host=f"{host}:{port}",
            epoch=mesh.record.epoch,
            timeout_s=5.0,
            expected_device=expected_device,
        )
    finally:
        fake.handshake_done.wait(5)


def _links(mesh: Mesh) -> list[Any]:
    return [
        link for link in mesh.server.links.values() if link.network_id == mesh.record.network_id
    ]


def _row(mesh: Mesh, device_id: str) -> Any:
    return store.load(mesh.record.network_id, mesh.root).member(device_id)


# ---------------------------------------------------------------------------
# Negative control: the rig CAN establish an honest link
# ---------------------------------------------------------------------------


def test_a_genuine_listener_is_still_admitted(
    mesh: Mesh, dialer_key_derivations: list[str]
) -> None:
    fake = FakeListener(
        listener_identity=mesh.victim,
        network_id=mesh.record.network_id,
        claim_device_id=mesh.victim.device_id,
        policy=_genuine_policy(mesh),
        endpoints=["127.0.0.1:4777"],
    )
    try:
        link, reason = _dial(mesh, fake, expected_device=mesh.victim.device_id)
        assert not fake.errors, fake.errors
        assert link is not None, f"an honest member link was refused: {reason}"
        assert link.device_id == mesh.victim.device_id
        # ``establish`` is idempotent and the dial path calls it more than once
        # (``read_welcome``, ``dial``, ``codec``), so the control asks WHO, not how often.
        assert set(dialer_key_derivations) == {mesh.victim.device_id}
        assert "127.0.0.1:4777" in _row(mesh, mesh.victim.device_id).endpoints
    finally:
        fake.stop()


# ---------------------------------------------------------------------------
# G1: the dialer authenticates the listener
# ---------------------------------------------------------------------------


def test_an_impostor_with_no_member_key_and_no_epoch_is_refused(
    mesh: Mesh, dialer_key_derivations: list[str]
) -> None:
    """The audit spike: no key, no epoch, claims admin X — and must get nothing."""
    fake = FakeListener(
        listener_identity=mesh.attacker,
        network_id=mesh.record.network_id,
        claim_device_id=mesh.victim.device_id,
        endpoints=[POISONED_ENDPOINT],
    )
    try:
        link, reason = _dial(mesh, fake, expected_device=mesh.victim.device_id)
        keys_match = (
            fake.listener_keys is not None
            and link is not None
            and fake.listener_keys.k_d2l == link.result.keys.k_d2l
        )
        assert link is None and reason != "ok", (
            f"IMPERSONATION SUCCEEDED: dial returned reason={reason!r}, link device "
            f"{getattr(link, 'device_id', None)!r}, attacker k_d2l == dialer k_d2l: {keys_match}"
        )
        assert dialer_key_derivations == [], (
            f"the dialer derived link keys with an unauthenticated listener as "
            f"{dialer_key_derivations}"
        )
        assert _links(mesh) == [], "an unauthenticated link is in the dialer's link table"
        assert (
            POISONED_ENDPOINT not in _row(mesh, mesh.victim.device_id).endpoints
        ), "the impostor's welcome endpoints were persisted onto the victim's row (G4)"
    finally:
        fake.stop()


def test_a_real_member_answering_for_a_different_device_is_refused(
    mesh: Mesh, dialer_key_derivations: list[str]
) -> None:
    """A genuine member Y (real key, real epoch key) answers a dial meant for X."""
    fake = FakeListener(
        listener_identity=mesh.other,
        network_id=mesh.record.network_id,
        claim_device_id=mesh.other.device_id,
        policy=_genuine_policy(mesh),
        endpoints=[POISONED_ENDPOINT],
    )
    try:
        link, reason = _dial(mesh, fake, expected_device=mesh.victim.device_id)
        # The rig ran: the genuine listener sent its challenge. The fixed dialer then
        # hangs up BEFORE its auth frame (``Handshake.refuse_unexpected_peer``), so the
        # listener's only error is that EOF and it never saw an auth to complete on —
        # which is what keeps an honest Y from evicting its real link to this device.
        assert fake.auth_frame == {}, "the dialer sent its auth to the wrong device"
        assert all(isinstance(e, ConnectionError) for e in fake.errors), fake.errors
        assert link is None and reason != "ok", (
            f"the dialer meant {mesh.victim.device_id} and admitted "
            f"{getattr(link, 'device_id', None)!r} (reason={reason!r})"
        )
        assert dialer_key_derivations == [], dialer_key_derivations
        assert _links(mesh) == []
        assert POISONED_ENDPOINT not in _row(mesh, mesh.other.device_id).endpoints
        assert POISONED_ENDPOINT not in _row(mesh, mesh.victim.device_id).endpoints
    finally:
        fake.stop()


def test_welcome_endpoints_rewritten_after_signing_are_refused(
    mesh: Mesh, dialer_key_derivations: list[str]
) -> None:
    """The genuine X listener; an on-path attacker rewrites ``welcome.endpoints``."""

    def rewrite(frame: dict[str, Any]) -> None:
        frame["endpoints"] = [POISONED_ENDPOINT]

    fake = FakeListener(
        listener_identity=mesh.victim,
        network_id=mesh.record.network_id,
        claim_device_id=mesh.victim.device_id,
        policy=_genuine_policy(mesh),
        endpoints=["127.0.0.1:4777"],
        tamper_welcome=rewrite,
    )
    try:
        link, reason = _dial(mesh, fake, expected_device=mesh.victim.device_id)
        assert not fake.errors, fake.errors
        assert (
            POISONED_ENDPOINT not in _row(mesh, mesh.victim.device_id).endpoints
        ), "a welcome endpoint rewritten in flight was persisted onto the member row (G4)"
        assert (
            link is None and reason != "ok"
        ), f"a tampered welcome was accepted (reason={reason!r})"
        assert dialer_key_derivations == [], dialer_key_derivations
    finally:
        fake.stop()


# ---------------------------------------------------------------------------
# G2 / G3: what an impersonated link would carry
# ---------------------------------------------------------------------------


def test_an_impostor_cannot_introduce_or_redirect_member_rows(mesh: Mesh) -> None:
    """G2: the post-dial member pull must not adopt rows from an unverified link."""
    planted = types.MemberRecord(
        device_id=mesh.attacker.device_id,
        public_key=mesh.attacker.public_key,
        name="planted-admin",
        role="admin",
        capabilities=sorted(types.capabilities_for_role("admin")),
        added_by=mesh.victim.device_id,
        endpoints=[POISONED_ENDPOINT],
    ).to_json()
    redirected = _row(mesh, mesh.other.device_id).to_json()
    redirected["endpoints"] = [POISONED_ENDPOINT]
    fake = FakeListener(
        listener_identity=mesh.attacker,
        network_id=mesh.record.network_id,
        claim_device_id=mesh.victim.device_id,
        member_rows=[planted, redirected],
    )
    try:
        _dial(mesh, fake, expected_device=mesh.victim.device_id)
        assert (
            _row(mesh, mesh.attacker.device_id) is None
        ), "an impersonated admin link planted a new member row (G2)"
        assert (
            POISONED_ENDPOINT not in _row(mesh, mesh.other.device_id).endpoints
        ), "an impersonated link redirected a third member's endpoints (G2)"
    finally:
        fake.stop()


def test_member_rm_never_sends_the_new_secret_to_an_impostor(
    mesh: Mesh, monkeypatch: pytest.MonkeyPatch
) -> None:
    """G3: the rotation on ``member rm`` goes to every live link — not an impostor's."""
    from local_operator.network.credentials import sync as cred_sync

    # The removal-endings exchange dials the removed member; nothing answers it in
    # this rig and it is not the property under test.
    monkeypatch.setattr(cred_sync, "sync_for_relay", lambda server: None)
    fake = FakeListener(
        listener_identity=mesh.attacker,
        network_id=mesh.record.network_id,
        claim_device_id=mesh.victim.device_id,
    )
    try:
        _dial(mesh, fake, expected_device=mesh.victim.device_id)
        mesh.server._ctl_member_rm(
            {"network": mesh.record.network_id, "device_id": mesh.other.device_id}
        )
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and not any(
            f.get("op") == "net_epoch" for f in fake.received
        ):
            time.sleep(0.05)
        leaked = [f for f in fake.received if f.get("op") == "net_epoch" and f.get("secret")]
        assert not leaked, (
            f"the impostor received the rotated epoch secret on member rm (G3): "
            f"epoch={leaked[0].get('epoch')!r}, secret present={bool(leaked[0].get('secret'))}"
        )
    finally:
        fake.stop()


# ---------------------------------------------------------------------------
# The design's matrix (docs/design/mesh-listener-auth.md §7), cell by cell.
# Each names its T-number; T19/T20 are the design's path (c), out of scope here.
# ---------------------------------------------------------------------------


def _wrong_key_proof(signer: Any) -> Callable[..., dict[str, str]]:
    """A proof signed by ``signer`` and MACed under a key the dialer never used."""

    def forge(hs: hs_mod.Handshake, frame: dict[str, Any], _auth: dict[str, Any]) -> dict[str, str]:
        wrong = hs_mod.Credential("epoch", 1, token_bytes(32))
        return hs_mod.welcome_proof(signer, wrong, hs.transcript_bytes, frame)

    return forge


def _assert_refused(mesh: Mesh, reason: str, code: str, derivations: list[str]) -> None:
    assert reason == code, f"expected {code!r}, the dialer answered {reason!r}"
    assert derivations == [], f"link keys derived before refusal: {derivations}"
    assert _links(mesh) == []


def test_t1_impostor_without_any_key_is_listener_unproven(
    mesh: Mesh, dialer_key_derivations: list[str]
) -> None:
    fake = FakeListener(
        listener_identity=mesh.attacker,
        network_id=mesh.record.network_id,
        claim_device_id=mesh.victim.device_id,
        endpoints=[POISONED_ENDPOINT],
    )
    try:
        _link, reason = _dial(mesh, fake, expected_device=mesh.victim.device_id)
        _assert_refused(mesh, reason, "listener_unproven", dialer_key_derivations)
    finally:
        fake.stop()


def test_t2_impostor_signing_with_its_own_key_is_bad_signature(
    mesh: Mesh, dialer_key_derivations: list[str]
) -> None:
    fake = FakeListener(
        listener_identity=mesh.attacker,
        network_id=mesh.record.network_id,
        claim_device_id=mesh.victim.device_id,
        forge_proof=_wrong_key_proof(mesh.attacker),
    )
    try:
        _link, reason = _dial(mesh, fake, expected_device=mesh.victim.device_id)
        _assert_refused(mesh, reason, "bad_signature", dialer_key_derivations)
    finally:
        fake.stop()


def test_t3_member_key_without_the_current_epoch_key_is_bad_mac(
    mesh: Mesh, dialer_key_derivations: list[str]
) -> None:
    """A removed device keeps its Ed25519 key but not the rotated epoch secret."""
    fake = FakeListener(
        listener_identity=mesh.victim,
        network_id=mesh.record.network_id,
        claim_device_id=mesh.victim.device_id,
        forge_proof=_wrong_key_proof(mesh.victim),
    )
    try:
        _link, reason = _dial(mesh, fake, expected_device=mesh.victim.device_id)
        _assert_refused(mesh, reason, "bad_mac", dialer_key_derivations)
    finally:
        fake.stop()


@pytest.mark.parametrize("field_name", ["endpoints", "phase"])
def test_t4_a_field_rewritten_after_a_valid_proof_is_bad_signature(
    mesh: Mesh, dialer_key_derivations: list[str], field_name: str
) -> None:
    def rewrite(frame: dict[str, Any]) -> None:
        frame[field_name] = [POISONED_ENDPOINT] if field_name == "endpoints" else "reconcile"

    fake = FakeListener(
        listener_identity=mesh.victim,
        network_id=mesh.record.network_id,
        claim_device_id=mesh.victim.device_id,
        policy=_genuine_policy(mesh),
        endpoints=["127.0.0.1:4777"],
        tamper_welcome=rewrite,
    )
    try:
        _link, reason = _dial(mesh, fake, expected_device=mesh.victim.device_id)
        assert not fake.errors, fake.errors
        _assert_refused(mesh, reason, "bad_signature", dialer_key_derivations)
        assert POISONED_ENDPOINT not in _row(mesh, mesh.victim.device_id).endpoints
    finally:
        fake.stop()


def test_t5_the_dialers_own_auth_tags_reflected_as_proof_do_not_verify(
    mesh: Mesh, dialer_key_derivations: list[str]
) -> None:
    """Label separation: ``lop-mesh-auth-v1`` tags never verify as an accept proof."""

    def reflect(_hs: Any, _frame: Any, auth: dict[str, Any]) -> dict[str, str]:
        return {"sig": str(auth["sig"]), "mac": str(auth["mac"])}

    fake = FakeListener(
        listener_identity=mesh.attacker,
        network_id=mesh.record.network_id,
        claim_device_id=mesh.server.identity.device_id,
        forge_proof=reflect,
    )
    try:
        # The reflection only means anything against the dialer's OWN key, so the
        # cell aims the dial at itself-as-expected and the step-4 self-link guard
        # would win; instead verify the proof directly against the dialer's key.
        host, port = fake.start()
        sock = socket.create_connection((host, port), timeout=5)
        hs = hs_mod.Handshake.new(
            role="dialer",
            identity=mesh.server.identity,
            network_id=mesh.record.network_id,
            epoch=1,
            instance_id="i_dialer",
            session_protocol=1,
        )
        hs.send_hello(sock)
        reader = wire.FrameReader(sock)
        deadline = wire.deadline_in(5.0)
        hs.read_challenge(reader, deadline)
        hs.send_auth(
            sock,
            hs_mod.Credential("epoch", 1, wire.epoch_key(mesh.secret, mesh.record.network_id, 1)),
        )
        frame = reader.read_line(deadline)
        sock.close()
        assert frame["proof"]["sig"] == fake.auth_frame["sig"]
        digest = hs_mod.accept_digest(hs.transcript_bytes, frame)
        with pytest.raises(types.HandshakeRefusal) as caught:
            hs_mod._verify_signature(  # noqa: SLF001 — the primitive under test
                mesh.server.identity.public_key,
                frame["proof"]["sig"],
                digest,
                label=hs_mod.ACCEPT_SIG_LABEL,
            )
        assert caught.value.code == "bad_signature"
        # And the labels are pairwise distinct from the dialer's.
        labels = {
            hs_mod.ACCEPT_DIGEST_LABEL,
            hs_mod.ACCEPT_SIG_LABEL,
            hs_mod.ACCEPT_MAC_LABEL,
            b"lop-mesh-auth-v1\x00",
            b"lop-mesh-mac-v1\x00",
        }
        assert len(labels) == 5
        assert dialer_key_derivations == []
    finally:
        fake.stop()


def test_t6_an_honest_member_at_the_targets_address_is_refused_and_marked_suspect(
    mesh: Mesh, dialer_key_derivations: list[str]
) -> None:
    fake = FakeListener(
        listener_identity=mesh.other,
        network_id=mesh.record.network_id,
        claim_device_id=mesh.other.device_id,
        policy=_genuine_policy(mesh),
        endpoints=[POISONED_ENDPOINT],
    )
    host, port = fake.start()
    endpoint = f"{host}:{port}"
    with store.mutate(mesh.record.network_id, mesh.root) as record:
        victim_row = record.member(mesh.victim.device_id)
        assert victim_row is not None
        victim_row.endpoints = [endpoint, "127.0.0.1:1"]
        store.save(record, mesh.root)
    try:
        _link, reason = mesh.server.dial(
            mesh.record.network_id,
            host=endpoint,
            epoch=mesh.record.epoch,
            timeout_s=5.0,
            expected_device=mesh.victim.device_id,
        )
        fake.handshake_done.wait(5)
        _assert_refused(mesh, reason, "peer_identity_mismatch", dialer_key_derivations)
        assert fake.auth_frame == {}, "refused after sending auth: the listener completes"
        # The challenge is UNAUTHENTICATED, so the row is untouched (review M1): the
        # address is only tried after the member's others, in memory.
        victim = _row(mesh, mesh.victim.device_id)
        assert victim.endpoints == [endpoint, "127.0.0.1:1"]
        assert endpoint in mesh.server._suspect_endpoints[mesh.victim.device_id]  # noqa: SLF001
        assert POISONED_ENDPOINT not in _row(mesh, mesh.other.device_id).endpoints
    finally:
        fake.stop()


def test_t6b_listener_unproven_does_not_demote_the_endpoint(mesh: Mesh) -> None:
    """An old member's expected answer must not strand it after it updates."""
    fake = FakeListener(
        listener_identity=mesh.attacker,
        network_id=mesh.record.network_id,
        claim_device_id=mesh.victim.device_id,
    )
    host, port = fake.start()
    endpoint = f"{host}:{port}"
    with store.mutate(mesh.record.network_id, mesh.root) as record:
        victim_row = record.member(mesh.victim.device_id)
        assert victim_row is not None
        victim_row.endpoints = [endpoint]
        store.save(record, mesh.root)
    try:
        _link, reason = mesh.server.dial(
            mesh.record.network_id,
            host=endpoint,
            timeout_s=5.0,
            expected_device=mesh.victim.device_id,
        )
        fake.handshake_done.wait(5)
        assert reason == "listener_unproven"
        assert _row(mesh, mesh.victim.device_id).endpoints == [endpoint]
    finally:
        fake.stop()


def test_t7_an_impersonated_admin_writes_nothing_and_is_never_pulled(
    mesh: Mesh, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[Any] = []
    original = relay.adopt_members
    monkeypatch.setattr(
        relay, "adopt_members", lambda *a, **k: calls.append(k) or original(*a, **k)
    )
    path = store.record_path(mesh.record.network_id, mesh.root)
    before = path.read_bytes()
    fake = FakeListener(
        listener_identity=mesh.attacker,
        network_id=mesh.record.network_id,
        claim_device_id=mesh.victim.device_id,
        member_rows=[_row(mesh, mesh.other.device_id).to_json()],
    )
    try:
        _link, reason = _dial(mesh, fake, expected_device=mesh.victim.device_id)
        assert reason == "listener_unproven"
        assert path.read_bytes() == before, "the record changed on an impersonated dial"
        assert calls == [], "adopt_members ran for an unverified supplier"
        assert not any(f.get("op") == "net_member_list" for f in fake.received)
    finally:
        fake.stop()


def test_t9_an_impersonated_dial_never_displaces_the_real_link(
    mesh: Mesh, monkeypatch: pytest.MonkeyPatch
) -> None:
    genuine = FakeListener(
        listener_identity=mesh.victim,
        network_id=mesh.record.network_id,
        claim_device_id=mesh.victim.device_id,
        policy=_genuine_policy(mesh),
    )
    impostor = FakeListener(
        listener_identity=mesh.attacker,
        network_id=mesh.record.network_id,
        claim_device_id=mesh.victim.device_id,
        forge_proof=_wrong_key_proof(mesh.attacker),
    )
    try:
        real, reason = _dial(mesh, genuine, expected_device=mesh.victim.device_id)
        assert real is not None, reason
        connected: list[str] = []
        original = mesh.server.redial.connected
        monkeypatch.setattr(
            mesh.server.redial, "connected", lambda d: connected.append(d) or original(d)
        )
        fake_link, reason = _dial(mesh, impostor, expected_device=mesh.victim.device_id)
        assert fake_link is None and reason == "bad_signature"
        assert _links(mesh) == [real], "the real outbound link was displaced"
        assert real.alive and not real.closing
        assert connected == [], "an unverified dial reached redial.connected"
    finally:
        impostor.stop()
        genuine.stop()


# -- T10/T18/T22: the rollout path — an OLD dialer reaching a NEW listener ----------


def _old_dialer_links_in(
    mesh: Mesh, build: dict[str, Any], *, caps: list[str] | None = None
) -> tuple[Any, dict[str, Any]]:
    """``other`` on a pre-proof build dials the real relay; returns (its link there, welcome).

    The old dialer is the real ``Handshake`` minus ``verify_welcome``: it reads the
    welcome the way every build before this one did (frame name, version, phase,
    endpoints) and ignores the unknown ``proof`` key.
    """
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.bind(("127.0.0.1", 0))
    server_sock.listen(1)
    client = socket.create_connection(server_sock.getsockname()[:2], timeout=5)
    conn, addr = server_sock.accept()
    server_sock.close()
    listener = threading.Thread(
        target=mesh.server._run_inbound_handshake, args=(conn, addr), daemon=True  # noqa: SLF001
    )
    listener.start()
    hs = hs_mod.Handshake.new(
        role="dialer",
        identity=mesh.other,
        network_id=mesh.record.network_id,
        epoch=1,
        instance_id="i_old_dialer",
        session_protocol=1,
        build=build,
        # ``caps=LINK_CAPABILITIES`` models an UPDATED member dialling in first: its
        # hello advertises ``listener-proof-v1`` inside the MAC-covered transcript.
        capabilities=caps,
    )
    hs.send_hello(client)
    reader = wire.FrameReader(client)
    deadline = wire.deadline_in(5.0)
    hs.read_challenge(reader, deadline)
    hs.send_auth(
        client,
        hs_mod.Credential("epoch", 1, wire.epoch_key(mesh.secret, mesh.record.network_id, 1)),
    )
    welcome = reader.read_line(deadline)
    assert welcome["net"] == "welcome"
    hs.phase = welcome.get("phase", "member")
    hs.establish()  # the OLD dialer: no proof check at all
    listener.join(5)
    deadline_links = time.monotonic() + 3.0
    while time.monotonic() < deadline_links:
        held = mesh.server._link_for(mesh.other.device_id)  # noqa: SLF001
        if held is not None:
            mesh._old_client = client  # type: ignore[attr-defined]
            return held, welcome
        time.sleep(0.02)
    client.close()
    raise AssertionError("the new listener did not admit the old dialer's link")


def test_t10_t18_new_listener_admits_an_old_dialer_and_update_finds_that_link(
    mesh: Mesh,
) -> None:
    link, welcome = _old_dialer_links_in(mesh, {"version": "0.100.0"})
    try:
        assert "proof" in welcome, "the new listener must always attach its proof"
        assert link.peer_verified is True, "the listener verified the old dialer's auth"
        # ``meshupdate.update_peer`` asks ``_ensure_link_with_reason`` first, which
        # returns ANY live link — so the roll rides this inbound one without a dial.
        found, reason = mesh.server._ensure_link_with_reason(mesh.other.device_id)  # noqa: SLF001
        assert found is link and reason == ""
    finally:
        mesh._old_client.close()  # type: ignore[attr-defined]


def test_t22_inbound_link_keeps_the_sticky_state_and_names_the_verified_build(
    mesh: Mesh,
) -> None:
    mesh.server._unproven_listeners[mesh.other.device_id] = {  # noqa: SLF001
        "build": {"version": "0.99.0"},
        "endpoint": "127.0.0.1:9",
        "at": time.time(),
    }
    link, _welcome = _old_dialer_links_in(mesh, {"version": "0.100.0"})
    try:
        row = mesh.server._listener_proof_row(mesh.other.device_id, link)  # noqa: SLF001
        assert row["state"] == "unproven", "an inbound-only link cleared the listener fact"
        assert row["build"] == {"version": "0.100.0"} and row["build_source"] == "hello"
    finally:
        mesh._old_client.close()  # type: ignore[attr-defined]


def test_t22_a_verified_outbound_link_clears_the_sticky_state(mesh: Mesh) -> None:
    mesh.server._unproven_listeners[mesh.victim.device_id] = {  # noqa: SLF001
        "build": {"version": "0.99.0"},
        "endpoint": "127.0.0.1:9",
        "at": time.time(),
    }
    fake = FakeListener(
        listener_identity=mesh.victim,
        network_id=mesh.record.network_id,
        claim_device_id=mesh.victim.device_id,
        policy=_genuine_policy(mesh),
    )
    try:
        link, reason = _dial(mesh, fake, expected_device=mesh.victim.device_id)
        assert link is not None, reason
        assert mesh.victim.device_id not in mesh.server._unproven_listeners  # noqa: SLF001
        row = mesh.server._listener_proof_row(mesh.victim.device_id, link)  # noqa: SLF001
        assert row["state"] == "verified"
    finally:
        fake.stop()


def test_t11_an_old_listener_is_refused_naming_the_build_it_reports(
    mesh: Mesh, dialer_key_derivations: list[str]
) -> None:
    fake = FakeListener(
        listener_identity=mesh.victim,
        network_id=mesh.record.network_id,
        claim_device_id=mesh.victim.device_id,
        policy=_genuine_policy(mesh),
        build={"version": "0.101.40", "source_ref": "/private/path/never/rendered"},
        tamper_welcome=lambda frame: frame.pop("proof"),
    )
    try:
        _link, reason = _dial(mesh, fake, expected_device=mesh.victim.device_id)
        _assert_refused(mesh, reason, "listener_unproven", dialer_key_derivations)
        sticky = mesh.server._unproven_listeners[mesh.victim.device_id]  # noqa: SLF001
        assert sticky["build"] == {"version": "0.101.40"}, "source_ref must never be kept"
    finally:
        fake.stop()


def test_t11_the_refusal_sentence_names_the_challenge_build(tmp_path: Path) -> None:
    hs = hs_mod.Handshake.new(
        role="dialer",
        identity=identity.mint(tmp_path / "self", name="self"),
        network_id="n",
        epoch=1,
        instance_id="i",
        session_protocol=1,
    )
    hs.self_device_id = "d_self"
    hs.challenge = {"device_id": "d_peer", "build": {"version": "0.101.40"}}
    with pytest.raises(types.HandshakeRefusal) as caught:
        hs.verify_welcome(
            {"net": "welcome", "v": hs_mod.LINK_VERSION, "device_id": "d_peer", "epoch": 1},
            hs_mod.DialerExpectation(device_id="d_peer", public_key_for=lambda _d: None),
        )
    assert caught.value.code == "listener_unproven"
    assert "0.101.40" in caught.value.sentence


@pytest.mark.parametrize(
    "proof",
    [None, "not-a-dict", {"sig": "!!!", "mac": "%%%"}, {"sig": "", "mac": ""}, {"sig": "AA"}],
)
def test_a_malformed_proof_is_a_refusal_never_a_crash(proof: Any, tmp_path: Path) -> None:
    hs = hs_mod.Handshake.new(
        role="dialer",
        identity=identity.mint(tmp_path / "self", name="self"),
        network_id="n",
        epoch=1,
        instance_id="i",
        session_protocol=1,
    )
    hs.challenge = {"device_id": "d_peer"}
    frame: dict[str, Any] = {"net": "welcome", "v": hs_mod.LINK_VERSION, "device_id": "d_peer"}
    if proof is not None:
        frame["proof"] = proof
    with pytest.raises(types.HandshakeRefusal) as caught:
        hs.verify_welcome(frame, hs_mod.DialerExpectation(device_id="d_peer"))
    assert caught.value.code == "listener_unproven"


def test_an_empty_expected_device_refuses_rather_than_trusting_the_claim(
    mesh: Mesh, dialer_key_derivations: list[str]
) -> None:
    fake = FakeListener(
        listener_identity=mesh.victim,
        network_id=mesh.record.network_id,
        claim_device_id=mesh.victim.device_id,
        policy=_genuine_policy(mesh),
    )
    try:
        _link, reason = _dial(mesh, fake, expected_device="")
        _assert_refused(mesh, reason, "peer_identity_mismatch", dialer_key_derivations)
    finally:
        fake.stop()


# -- T12-T15: join, over a loopback pair of real Handshake halves ---------------------


@dataclass
class JoinRig:
    inviter: Any
    joiner: Any
    network_id: str
    invite_key: bytes


@pytest.fixture()
def join_rig(root: Path) -> JoinRig:
    return JoinRig(
        inviter=identity.mint(root / "inviter", name="inviter"),
        joiner=identity.mint(root / "joiner", name="joiner"),
        network_id=store.new_network_id(),
        invite_key=token_bytes(32),
    )


def _join(
    rig: JoinRig,
    *,
    listener_identity: Any,
    listener_key: bytes,
    forge_inviter_key: str = "",
    claim_device_id: str = "",
) -> tuple[hs_mod.Handshake, hs_mod.Handshake | None, BaseException | None]:
    """Run one join handshake; the listener may hold a wrong invite key or a wrong identity.

    The listener half skips ``verify_auth`` when it does not hold the joiner's invite
    key (that is the attacker without the token): it can still send a welcome, signed
    with the key it has and MACed with the key it has.
    """
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.bind(("127.0.0.1", 0))
    server_sock.listen(1)
    client = socket.create_connection(server_sock.getsockname()[:2], timeout=5)
    conn, _ = server_sock.accept()
    server_sock.close()
    out: dict[str, Any] = {}

    def listen() -> None:
        try:
            reader = wire.FrameReader(conn)
            deadline = wire.deadline_in(5.0)
            hs = hs_mod.Handshake.new(
                role="listener",
                identity=listener_identity,
                network_id=rig.network_id,
                epoch=1,
                instance_id="i_inviter",
                session_protocol=1,
                mode="join",
            )
            if claim_device_id:
                hs.self_device_id = claim_device_id
            hs.credential = hs_mod.Credential("invite", 1, listener_key)
            hs.read_hello(reader, deadline)
            hs.send_challenge(conn, hs_mod.StaticPolicy(networks={rig.network_id: True}))
            try:
                hs.verify_auth(
                    reader, deadline, hs_mod.StaticPolicy(networks={rig.network_id: True})
                )
            except types.HandshakeRefusal:
                # No token: it cannot verify the joiner, and welcomes it anyway.
                hs.phase = "pair"
            frame = hs.welcome_frame(phase="pair", epoch=1, network_name="net")
            frame.pop("proof", None)
            if forge_inviter_key:
                frame["inviter"]["public_key"] = forge_inviter_key
            frame["proof"] = hs_mod.welcome_proof(
                listener_identity,
                hs_mod.Credential("invite", 1, listener_key),
                hs.transcript_bytes,
                frame,
            )
            hs.send_welcome(conn, frame)
            out["listener"] = hs
        except BaseException as exc:  # noqa: BLE001
            out["listener_error"] = exc

    thread = threading.Thread(target=listen, daemon=True)
    thread.start()
    dialer = hs_mod.Handshake.new(
        role="dialer",
        identity=rig.joiner,
        network_id=rig.network_id,
        epoch=1,
        instance_id="i_joiner",
        session_protocol=1,
        mode="join",
    )
    dialer.join_block = {
        "invite_id": "inv_1",
        "joiner_public_key": rig.joiner.public_key,
        "joiner_name": "joiner",
    }
    error: BaseException | None = None
    try:
        dialer.send_hello(client)
        reader = wire.FrameReader(client)
        deadline = wire.deadline_in(5.0)
        dialer.read_challenge(reader, deadline)
        dialer.send_auth(client, hs_mod.Credential("invite", 1, rig.invite_key))
        dialer.read_welcome(
            reader, deadline, hs_mod.DialerExpectation(device_id=rig.inviter.device_id)
        )
    except types.HandshakeRefusal as exc:
        error = exc
    finally:
        thread.join(5)
        client.close()
        conn.close()
    return dialer, out.get("listener"), error


def test_t12_a_fake_inviter_without_the_token_is_bad_mac(join_rig: JoinRig) -> None:
    dialer, _listener, error = _join(
        join_rig, listener_identity=join_rig.inviter, listener_key=token_bytes(32)
    )
    assert isinstance(error, types.HandshakeRefusal) and error.code == "bad_mac"
    assert not dialer.auth_verified and not dialer.shared


def test_t13_a_token_holder_with_a_different_key_is_invite_device_mismatch(
    root: Path, join_rig: JoinRig
) -> None:
    attacker = identity.mint(root / "attacker", name="attacker")
    _dialer, _listener, error = _join(
        join_rig,
        listener_identity=attacker,
        listener_key=join_rig.invite_key,
        claim_device_id=join_rig.inviter.device_id,
    )
    assert isinstance(error, types.HandshakeRefusal)
    assert error.code == "invite_device_mismatch"


def test_t13b_a_token_holder_answering_as_itself_is_invite_wrong_inviter(
    root: Path, join_rig: JoinRig
) -> None:
    attacker = identity.mint(root / "attacker", name="attacker")
    _dialer, _listener, error = _join(
        join_rig, listener_identity=attacker, listener_key=join_rig.invite_key
    )
    assert isinstance(error, types.HandshakeRefusal)
    assert error.code == "invite_wrong_inviter"


def test_t14_an_honest_join_verifies_and_the_sas_is_unchanged(join_rig: JoinRig) -> None:
    dialer, listener, error = _join(
        join_rig, listener_identity=join_rig.inviter, listener_key=join_rig.invite_key
    )
    assert error is None and listener is not None
    assert dialer.auth_verified and dialer.peer_public_key == join_rig.inviter.public_key
    result = dialer.establish()
    assert result.peer_verified is True and result.phase == "pair"
    # T is hello/challenge/auth only — the proof is an extra tag, not a transcript
    # element — so both sides derive the same digits by the unchanged formula.
    assert dialer.sas == listener.sas
    expected = wire.sas_code(
        dialer.shared,
        hs_mod.transcript_digest(
            hs_mod.transcript(dialer.hello, dialer.challenge, dialer.auth_core)
        ),
    )
    assert dialer.sas == expected
    assert b"proof" not in dialer.transcript_bytes


def test_t15_a_pair_result_inviter_row_with_another_key_is_refused() -> None:
    from types import SimpleNamespace

    from local_operator.network import cli as net_cli

    envelope = SimpleNamespace(inviter_device_id="d_inviter")
    handshake = SimpleNamespace(auth_verified=True, peer_public_key="KEY_VERIFIED")
    answer = {"members": [{"device_id": "d_inviter", "public_key": "KEY_OTHER"}]}
    with pytest.raises(types.MeshRefusal) as caught:
        net_cli._pin_inviter_key(answer, envelope, handshake)  # noqa: SLF001
    assert caught.value.code == "invite_wrong_inviter"
    answer["members"][0]["public_key"] = "KEY_VERIFIED"
    net_cli._pin_inviter_key(answer, envelope, handshake)  # noqa: SLF001


# -- T16/T17: the structural gate and its backup --------------------------------------


def test_t16_admit_link_refuses_an_unverified_link(mesh: Mesh) -> None:
    from types import SimpleNamespace

    a, b = socket.socketpair()
    try:
        link: Any = SimpleNamespace(
            peer_verified=False, sock=a, network_id=mesh.record.network_id, device_id="d_x"
        )
        assert mesh.server._admit_link(link, outbound=True) == "unverified_link"  # noqa: SLF001
        assert mesh.server.links == {}
        assert a.fileno() == -1, "a refused link's socket must be closed"
    finally:
        a.close()
        b.close()


def test_t17_adopt_members_refuses_an_unverified_supplier(mesh: Mesh) -> None:
    record = store.load(mesh.record.network_id, mesh.root)
    planted = types.MemberRecord(
        device_id=mesh.attacker.device_id,
        public_key=mesh.attacker.public_key,
        name="planted",
        role="admin",
        capabilities=sorted(types.capabilities_for_role("admin")),
        added_by=mesh.victim.device_id,
    ).to_json()
    assert relay.adopt_members(
        record, [planted], sender_device_id=mesh.victim.device_id, supplier_verified=False
    ) == (False, [])
    assert record.member(mesh.attacker.device_id) is None
    # Control: the same rows from a VERIFIED admin supplier are adopted.
    assert relay.adopt_members(
        record, [planted], sender_device_id=mesh.victim.device_id, supplier_verified=True
    ) == (True, [mesh.attacker.device_id])


def test_note_peer_endpoints_writes_nothing_unverified(mesh: Mesh) -> None:
    mesh.server._note_peer_endpoints(  # noqa: SLF001
        mesh.record.network_id, mesh.victim.device_id, [POISONED_ENDPOINT], verified=False
    )
    assert POISONED_ENDPOINT not in _row(mesh, mesh.victim.device_id).endpoints


# -- T21/T23: surfacing ---------------------------------------------------------------


def test_t21_the_unproven_fact_survives_later_redial_failures_and_prints_once(
    mesh: Mesh,
) -> None:
    from local_operator.network import cli as net_cli

    fake = FakeListener(
        listener_identity=mesh.attacker,
        network_id=mesh.record.network_id,
        claim_device_id=mesh.victim.device_id,
        build={"version": "0.101.40"},
    )
    try:
        _link, reason = _dial(mesh, fake, expected_device=mesh.victim.device_id)
        assert reason == "listener_unproven"
    finally:
        fake.stop()
    for _ in range(3):
        mesh.server.redial.failed(
            mesh.victim.device_id, cause="connect_failed:ConnectionRefusedError"
        )
    rows = {row["device_id"]: row for row in mesh.server.peer_status()}
    victim_row = rows[mesh.victim.device_id]
    proof = victim_row["listener_proof"]
    assert proof["state"] == "unproven" and proof["build"] == {"version": "0.101.40"}
    assert proof["build_source"] == "challenge"
    assert rows[mesh.other.device_id]["listener_proof"]["state"] == "unknown"
    line = net_cli._listener_proof_line(victim_row)  # noqa: SLF001
    # "reports" (a challenge claim), never "older" (UX U4); both remedies (S-4).
    core = (
        "victim-admin reports build 0.101.40, which cannot prove its identity when this "
        "device connects to it, so this device no longer opens connections to it. "
        "Connections it opens to this device still work."
    )
    assert line.strip() == (
        f"{core} Update it: run `lop network update victim-admin` from this device, or, "
        "if victim-admin cannot connect to this device, run `lop update` on victim-admin "
        "itself."
    )
    assert net_cli._listener_proof_line(rows[mesh.other.device_id]) == ""  # noqa: SLF001
    finding = mesh.server._listener_proof_finding(_row(mesh, mesh.victim.device_id))  # noqa: SLF001
    assert finding is not None and finding["severity"] == "warn"
    # ``detail`` is a code like every doctor row (U8); the remedy is said once (U5).
    assert finding["detail"] == "listener_unproven"
    assert finding["message"] == core
    assert finding["remedies"] == [
        "run `lop network update victim-admin` from this device",
        "or, if victim-admin cannot connect to this device, run `lop update` on "
        "victim-admin itself",
    ]


def test_t23_both_gloss_tables_read_the_new_codes() -> None:
    from local_operator import resume

    for code in ("listener_unproven", "peer_identity_mismatch"):
        assert code in resume.PEER_REASON_WORDS and code in resume.DOCTOR_DETAIL_WORDS
        assert resume.peer_reason_words(code) == resume.PEER_REASON_WORDS[code]
        assert resume.doctor_detail_words(code) == resume.DOCTOR_DETAIL_WORDS[code]
        assert code not in resume.peer_reason_words(code)


# -- Risks 4 and 9: canonicalisation round-trip and the line budget -------------------


def test_a_real_welcome_round_trips_the_wire_and_fits_the_line_budget(mesh: Mesh) -> None:
    fake = FakeListener(
        listener_identity=mesh.victim,
        network_id=mesh.record.network_id,
        claim_device_id=mesh.victim.device_id,
        policy=_genuine_policy(mesh),
        endpoints=["127.0.0.1:4777", "[::1]:4777"],
    )
    captured: list[bytes] = []
    original = hs_mod._send_frame  # noqa: SLF001

    def capture(sock: Any, frame: dict[str, Any]) -> None:
        if frame.get("net") == "welcome":
            captured.append(wire.encode_line(frame))
        original(sock, frame)

    try:
        hs_mod._send_frame = capture  # type: ignore[assignment]  # noqa: SLF001
        link, reason = _dial(mesh, fake, expected_device=mesh.victim.device_id)
        assert link is not None, reason
    finally:
        hs_mod._send_frame = original  # type: ignore[assignment]  # noqa: SLF001
        fake.stop()
    assert captured and len(captured[0]) < wire.MAX_HANDSHAKE_LINE
    assert len(captured[0]) < 4096


# ---------------------------------------------------------------------------
# Round 1 remediation (PR #4179): L1 rotation hop, L2 no unauthenticated demotion,
# L3 the sticky record yields to a live proof
# ---------------------------------------------------------------------------


def _rotated(mesh: Mesh, tmp_path: Path) -> tuple[Any, dict[str, Any]]:
    """The victim's successor identity and the statement its OLD key signed for it."""
    new = identity.mint(tmp_path / "victim-rotated", name=mesh.victim.name)
    return new, identity.rotation_statement(mesh.victim, new, mesh.record.network_id)


def _only_endpoint(mesh: Mesh, endpoint: str) -> None:
    with store.mutate(mesh.record.network_id, mesh.root) as record:
        row = record.member(mesh.victim.device_id)
        assert row is not None
        row.endpoints = [endpoint]
        store.save(record, mesh.root)


def test_l1_a_rotated_member_answering_under_its_new_id_is_followed_not_refused(
    mesh: Mesh, tmp_path: Path
) -> None:
    """G1D-1 / S-1: the dialer holds the PRE-rotation row; the member answers as A'.

    RED before round 1: refused ``peer_identity_mismatch`` and the member's only
    address was removed, so the queued ``net_identity_rotate`` could never flush.
    """
    new, statement = _rotated(mesh, tmp_path)
    fake = FakeListener(
        listener_identity=new,
        network_id=mesh.record.network_id,
        claim_device_id=new.device_id,
        policy=_genuine_policy(mesh),
        rotation=statement,
    )
    host, port = fake.start()
    endpoint = f"{host}:{port}"
    _only_endpoint(mesh, endpoint)
    try:
        link, reason = mesh.server.dial(
            mesh.record.network_id,
            host=endpoint,
            timeout_s=5.0,
            expected_device=mesh.victim.device_id,
        )
        fake.handshake_done.wait(5)
        assert link is not None, reason
        assert link.device_id == new.device_id and link.peer_verified is True
        row = _row(mesh, new.device_id)
        assert row is not None and row.public_key == new.public_key
        assert _row(mesh, mesh.victim.device_id) is not None, "the old id must still resolve"
        assert mesh.victim.device_id in row.previous_ids
        assert endpoint in row.endpoints, "the rotated member's address was demoted"
    finally:
        fake.stop()


def test_l1_a_replayed_rotation_statement_without_the_new_key_is_refused(
    mesh: Mesh, tmp_path: Path, dialer_key_derivations: list[str]
) -> None:
    """The statement is public (every table carries it); replaying it proves nothing.

    It gets the impostor exactly what claiming the victim's own id already gets (our
    auth frame), then fails the proof under the statement's new key, and neither
    rotates the row nor removes the member's only address.
    """
    new, statement = _rotated(mesh, tmp_path)
    fake = FakeListener(
        listener_identity=mesh.attacker,
        network_id=mesh.record.network_id,
        claim_device_id=new.device_id,
        rotation=statement,
        forge_proof=_wrong_key_proof(mesh.attacker),
    )
    host, port = fake.start()
    endpoint = f"{host}:{port}"
    _only_endpoint(mesh, endpoint)
    try:
        _link, reason = mesh.server.dial(
            mesh.record.network_id,
            host=endpoint,
            timeout_s=5.0,
            expected_device=mesh.victim.device_id,
        )
        fake.handshake_done.wait(5)
        _assert_refused(mesh, reason, "bad_signature", dialer_key_derivations)
        assert _row(mesh, new.device_id) is None, "an unproven hop rotated the row"
        assert _row(mesh, mesh.victim.device_id).endpoints == [endpoint]
    finally:
        fake.stop()


def test_l1_a_statement_not_signed_by_the_stored_key_is_refused_before_auth(
    mesh: Mesh, tmp_path: Path, dialer_key_derivations: list[str]
) -> None:
    """A hop the STORED key never signed is a wrong device: refused pre-auth."""
    new = identity.mint(tmp_path / "attacker-new", name="victim-admin")
    forged = identity.rotation_statement(mesh.attacker, new, mesh.record.network_id)
    forged["old_device_id"] = mesh.victim.device_id
    fake = FakeListener(
        listener_identity=new,
        network_id=mesh.record.network_id,
        claim_device_id=new.device_id,
        rotation=forged,
    )
    try:
        _link, reason = _dial(mesh, fake, expected_device=mesh.victim.device_id)
        _assert_refused(mesh, reason, "peer_identity_mismatch", dialer_key_derivations)
        assert fake.auth_frame == {}, "our auth left before the hop was verified"
        assert _row(mesh, new.device_id) is None
    finally:
        fake.stop()


def _garbage_proof(_hs: Any, _frame: Any, _auth: Any) -> dict[str, str]:
    """What a keyless party can send for free: 64 random bytes and 32 more."""
    return {"sig": wire.b64u(token_bytes(64)), "mac": wire.b64u(token_bytes(32))}


@pytest.mark.parametrize(
    ("claim", "forge", "code"),
    [
        ("other", None, "peer_identity_mismatch"),
        ("victim", _garbage_proof, "bad_signature"),
    ],
)
def test_l2_a_keyless_impostor_never_removes_a_members_only_address(
    mesh: Mesh, claim: str, forge: Any, code: str
) -> None:
    """M1 / S-2: one unauthenticated frame must not strand a single-address member.

    RED before round 1: both cells emptied the row (``ENDPOINTS_AFTER []``).
    """
    claimed = mesh.other.device_id if claim == "other" else mesh.victim.device_id
    fake = FakeListener(
        listener_identity=mesh.attacker,
        network_id=mesh.record.network_id,
        claim_device_id=claimed,
        forge_proof=forge,
    )
    host, port = fake.start()
    endpoint = f"{host}:{port}"
    _only_endpoint(mesh, endpoint)
    try:
        _link, reason = mesh.server.dial(
            mesh.record.network_id,
            host=endpoint,
            timeout_s=5.0,
            expected_device=mesh.victim.device_id,
        )
        fake.handshake_done.wait(5)
        assert reason == code
        assert _row(mesh, mesh.victim.device_id).endpoints == [endpoint]
        assert endpoint in mesh.server._suspect_endpoints[mesh.victim.device_id]  # noqa: SLF001
    finally:
        fake.stop()


def test_l2_bad_mac_still_demotes_but_never_the_last_address(mesh: Mesh) -> None:
    """``bad_mac`` follows a signature under the STORED key: authenticated evidence."""
    for spare in (["127.0.0.1:1"], []):
        fake = FakeListener(
            listener_identity=mesh.victim,
            network_id=mesh.record.network_id,
            claim_device_id=mesh.victim.device_id,
            forge_proof=_wrong_key_proof(mesh.victim),
        )
        host, port = fake.start()
        endpoint = f"{host}:{port}"
        with store.mutate(mesh.record.network_id, mesh.root) as record:
            row = record.member(mesh.victim.device_id)
            assert row is not None
            row.endpoints = [endpoint, *spare]
            store.save(record, mesh.root)
        try:
            _link, reason = mesh.server.dial(
                mesh.record.network_id,
                host=endpoint,
                timeout_s=5.0,
                expected_device=mesh.victim.device_id,
            )
            fake.handshake_done.wait(5)
            assert reason == "bad_mac"
            after = _row(mesh, mesh.victim.device_id).endpoints
            assert after == (spare or [endpoint]), (spare, after)
        finally:
            fake.stop()


def test_l2_a_suspect_address_is_dialled_after_the_others_and_still_dialled_alone(
    mesh: Mesh,
) -> None:
    """Suspects reorder dials, never remove them: the fall-through review M1 asked for."""
    # The suspect is a bare listening socket nobody accepts on: if the probe raced it,
    # a connection would be queued in its backlog, which ``accept`` below would find.
    wrong = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    wrong.bind(("127.0.0.1", 0))
    wrong.listen(4)
    wrong_host, wrong_port = wrong.getsockname()[:2]
    good = FakeListener(
        listener_identity=mesh.victim,
        network_id=mesh.record.network_id,
        claim_device_id=mesh.victim.device_id,
        policy=_genuine_policy(mesh),
    )
    good_host, good_port = good.start()
    suspect, healthy = f"{wrong_host}:{wrong_port}", f"{good_host}:{good_port}"
    with store.mutate(mesh.record.network_id, mesh.root) as record:
        row = record.member(mesh.victim.device_id)
        assert row is not None
        row.endpoints = [suspect, healthy]
        store.save(record, mesh.root)
    mesh.server._suspect_endpoints[mesh.victim.device_id] = {suspect}  # noqa: SLF001
    try:
        link, reason = mesh.server._ensure_link_with_reason(  # noqa: SLF001
            mesh.victim.device_id, probe_timeout_s=5.0
        )
        assert link is not None, reason
        assert link.peer_addr == healthy
        wrong.setblocking(False)
        with pytest.raises(BlockingIOError):
            wrong.accept()  # nothing queued: the suspect was deferred, not raced
    finally:
        wrong.close()
        good.stop()
    for live in _links(mesh):
        live.close("test")
    alone = FakeListener(
        listener_identity=mesh.victim,
        network_id=mesh.record.network_id,
        claim_device_id=mesh.victim.device_id,
        policy=_genuine_policy(mesh),
    )
    host, port = alone.start()
    only = f"{host}:{port}"
    _only_endpoint(mesh, only)
    mesh.server._suspect_endpoints[mesh.victim.device_id] = {only}  # noqa: SLF001
    deadline = time.monotonic() + 3.0
    while mesh.server._link_for(mesh.victim.device_id) is not None:  # noqa: SLF001
        assert time.monotonic() < deadline, "the first link never closed"
        time.sleep(0.02)
    try:
        link, reason = mesh.server._ensure_link_with_reason(  # noqa: SLF001
            mesh.victim.device_id, probe_timeout_s=5.0
        )
        assert link is not None, reason
        assert only not in mesh.server._suspect_endpoints[mesh.victim.device_id]  # noqa: SLF001
    finally:
        alone.stop()


def test_l3_an_updated_member_dialling_in_first_clears_the_unproven_record(
    mesh: Mesh,
) -> None:
    """U1 / m1 / S-3 / Q-1: the warning must go once the remedy worked.

    RED before round 1: only an OUTBOUND admission cleared the record, and a member
    connected inbound gets no redial, so ``peers`` kept saying "update it".
    """
    mesh.server._unproven_listeners[mesh.other.device_id] = {  # noqa: SLF001
        "build": {"version": "0.99.0"},
        "endpoint": "127.0.0.1:9",
        "at": time.time(),
    }
    link, _welcome = _old_dialer_links_in(
        mesh, {"version": "0.100.0"}, caps=list(wire.LINK_CAPABILITIES)
    )
    try:
        assert "listener-proof-v1" in link.capabilities
        assert mesh.other.device_id not in mesh.server._unproven_listeners  # noqa: SLF001
        rows = {row["device_id"]: row for row in mesh.server.peer_status()}
        assert rows[mesh.other.device_id]["listener_proof"]["state"] == "verified"
    finally:
        mesh._old_client.close()  # type: ignore[attr-defined]


def test_l3_a_stale_unproven_address_does_not_contradict_a_live_verified_link(
    mesh: Mesh,
) -> None:
    """A doctor probe of a stale address must not re-set the record (S-3)."""
    genuine = FakeListener(
        listener_identity=mesh.victim,
        network_id=mesh.record.network_id,
        claim_device_id=mesh.victim.device_id,
        policy=_genuine_policy(mesh),
    )
    stale = FakeListener(
        listener_identity=mesh.attacker,
        network_id=mesh.record.network_id,
        claim_device_id=mesh.victim.device_id,
        build={"version": "0.99.0"},
    )
    try:
        link, reason = _dial(mesh, genuine, expected_device=mesh.victim.device_id)
        assert link is not None, reason
        _link, reason = _dial(mesh, stale, expected_device=mesh.victim.device_id)
        assert reason == "listener_unproven"
        assert mesh.victim.device_id not in mesh.server._unproven_listeners  # noqa: SLF001
        finding = mesh.server._listener_proof_finding(  # noqa: SLF001
            _row(mesh, mesh.victim.device_id)
        )
        assert finding is None
    finally:
        genuine.stop()
        stale.stop()


def test_n1_a_welcome_before_auth_is_a_refusal_not_an_assertion(tmp_path: Path) -> None:
    """``python -O`` strips asserts; the mis-ordered call must still be refused."""
    signer = identity.mint(tmp_path / "peer", name="peer")
    hs = hs_mod.Handshake.new(
        role="dialer",
        identity=identity.mint(tmp_path / "self", name="self"),
        network_id="n",
        epoch=1,
        instance_id="i",
        session_protocol=1,
    )
    hs.challenge = {"device_id": signer.device_id}
    frame: dict[str, Any] = {
        "net": "welcome",
        "v": hs_mod.LINK_VERSION,
        "device_id": signer.device_id,
        "epoch": 1,
    }
    frame["proof"] = hs_mod.welcome_proof(
        signer, hs_mod.Credential("epoch", 1, token_bytes(32)), hs.transcript_bytes, frame
    )
    with pytest.raises(types.HandshakeRefusal) as caught:
        hs.verify_welcome(
            frame,
            hs_mod.DialerExpectation(
                device_id=signer.device_id, public_key_for=lambda _d: signer.public_key
            ),
        )
    assert caught.value.code == hs_mod.REASON_FRAME
