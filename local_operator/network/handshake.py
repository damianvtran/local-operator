"""The mutually-authenticated handshake, and the SAS the two humans compare.

Five frames, both roles in one file so the transcript construction is written
ONCE. That is the whole reason this is not two modules: the transcript is the
security boundary between them, and two implementations of it are two
implementations that can disagree.

    hello      dialer  → listener   (ephemeral public, nonce, device claim)
    challenge  listener → dialer    (same, plus a salt)
    auth       dialer  → listener   (sig over the transcript, MAC over the same)
    welcome    listener → dialer    (the LAST plaintext frame; phase + nets, and the
                                     listener's ``proof`` over the transcript + welcome)
    …then AEAD records for everything else, including the pair ceremony

BOTH SIDES PROVE THEMSELVES. The dialer's ``auth`` is checked by the listener
(:meth:`Handshake.verify_auth`); the listener's ``welcome.proof`` is checked by the
dialer (:meth:`Handshake.verify_welcome`) BEFORE any welcome field is read — see
``docs/design/mesh-listener-auth.md``. Until that check existed the dialer copied the
listener's claimed id out of the unsigned ``challenge`` and anyone who answered the
dial became that member (mesh trust audit G1).

MEMBER MODE differs from JOIN mode only in which key verifies the MAC: the
current (or previous) epoch key, or the invite key. Everything else — the
transcript, the signature, the SAS — is identical, which is why one verifier
covers both and why a join cannot take a shortcut a member link does not.

THE VERIFICATION ORDER IS THE SECURITY PROPERTY. Each step is checked in the
order §6.2 pins, each failure closes the socket with NO reply frame and writes a
local audit record, and the reason never distinguishes on the wire:

1. frame shape, known LINK version, line size
2. the network is one this install is in
3. ``trust == "active"`` (an untrusted network refuses every link)
4. ``epoch ∈ {current, current-1}``
5. the peer is not this device (``self_link``)
6. a member row exists, is not a tombstone, and IS EVALUATED AGAINST THE CURRENT
   EPOCH'S MEMBER LIST — this is the line that makes revocation real, and it runs
   before any key is tried
7. the signature verifies against the stored public key
8. the MAC verifies: for a join, under the INVITE key from the token, which is
   the joiner's only credential at that point; for a member, under the key of
   the epoch the AUTH FRAME NAMES (step 4 has already bounded that to
   {current, current-1}), and a previous-epoch answer is what puts the link in
   the ``reconcile`` phase

THE SAS IS DERIVED, NEVER TRANSMITTED. Its value is absent from every frame: a
peer that sent it would let an on-path attacker echo the inviter's own digits
back at it, and the human comparison would be a round trip rather than a check.
"""

from __future__ import annotations

import hmac
import socket
from collections.abc import Callable
from dataclasses import dataclass, field
from secrets import token_bytes
from typing import Any, Literal, Protocol, cast

from local_operator.network.identity import DeviceIdentity
from local_operator.network.types import (
    MESH_PROTOCOL_VERSION,
    HandshakeRefusal,
    HandshakeResult,
    LinkPhase,
    MeshRefusal,
    PairingRefusal,
)
from local_operator.network.wire import (
    MAX_HANDSHAKE_LINE,
    MESH_NET_V1,
    ROLE_DIALER,
    ZLIB_RECORDS_V1,
    FrameReader,
    LinkCrypto,
    Role,
    b64u,
    canonical_json,
    check_link_version,
    deadline_in,
    encode_line,
    link_keys,
    negotiate_capabilities,
    sas_code,
    sha256,
    unb64u,
)

#: The LINK protocol number, named here for the frames this module builds so a
#: reader sees one spelling. It is ``types.MESH_PROTOCOL_VERSION`` and nothing
#: else: two constants for one wire number is how a bump gets half-applied.
LINK_VERSION = MESH_PROTOCOL_VERSION

#: The whole ceremony's budget, applied as ONE absolute deadline rather than per
#: read: a peer that answers every read just inside a per-read timeout could
#: otherwise hold the handshake open indefinitely.
HANDSHAKE_TIMEOUT_S = 10.0

TRANSCRIPT_PREFIX = b"lop-mesh-v1\x00"

#: The longest a person may take to compare and confirm, as a ceiling on the
#: invite's own remaining life (``pair_timeout_seconds``). It was 180 s: a pairing
#: answered from the app's Mesh approvals routinely needs the person to walk to
#: the other device, and the old cap lapsed under the operator while the invite
#: itself still had minutes left. 600 s equals the default invite TTL
#: (``invite.DEFAULT_TTL_S``), so a default invite now gets its whole life, while a
#: ``--expires 24h`` token still cannot hold a pre-admission sealed socket open for
#: hours. The anti-grind property rests on the 3-attempt budget
#: (``invite.PAIRING_MAX_FORGIVEN_FAILURES``), not on this time — a longer window
#: buys no extra guesses. Rename from ``PAIR_CONFIRM_TIMEOUT_S`` keeps ONE owner:
#: every wait that involves a person reads ``pair_timeout_seconds``.
PAIR_PARK_CEILING_S = 600.0

#: How long the JOINER waits for the share list before it shows its screens
#: without one (``handshake.pair_offer_frame``). The owner sends the offer
#: immediately after ``welcome``, with NO human step in between, so this is a
#: rendezvous bound rather than a person's budget — and it is deliberately small:
#: a new owner that fails to send promptly would otherwise make every new joiner
#: pay the bound. A frame that arrives after it is still tolerated by the joining
#: side's first-result read, so a slow sender costs the list on screen, never the
#: ceremony. Never extended by anything the peer says.
PAIR_OFFER_WAIT_S = 2.0

Mode = Literal["member", "join"]

#: Refusal reasons the wire never sees.
REASON_FRAME = "malformed_frame"
REASON_VERSION = "protocol_mismatch"
REASON_UNKNOWN_NETWORK = "unknown_network"
REASON_UNTRUSTED = "untrusted"
REASON_EPOCH = "epoch_stale"
REASON_SELF = "self_link"
REASON_MEMBER = "not_a_member"
REASON_SIGNATURE = "bad_signature"
REASON_MAC = "bad_mac"
REASON_INVITE = "invite_invalid"
REASON_INVITE_EPOCH = "invite_epoch_stale"
REASON_INVITE_DEVICE = "invite_device_mismatch"
#: The DIALER's refusals of a listener (``verify_welcome``). ``listener_unproven``
#: is a welcome with no proof at all: an old build, or an impostor that cannot make
#: one. ``peer_identity_mismatch`` is a listener answering as a different device
#: than the one this device meant to reach. ``invite_wrong_inviter`` is the join
#: spelling of the same thing (``invite.REASON_INVITER``, kept identical on purpose).
REASON_LISTENER_UNPROVEN = "listener_unproven"
REASON_PEER_IDENTITY = "peer_identity_mismatch"
REASON_INVITE_INVITER = "invite_wrong_inviter"

#: Domain-separation labels for the listener's welcome proof. THREE labels, none of
#: them the dialer's (``lop-mesh-auth-v1`` / ``lop-mesh-mac-v1``): the dialer's own
#: auth sig/MAC replayed as a ``proof`` must never verify (reflection), and the
#: digest label keeps the proof's input disjoint from the auth digest even though
#: both are built from the same transcript.
ACCEPT_DIGEST_LABEL = b"lop-mesh-accept-v1\x00"
ACCEPT_SIG_LABEL = b"lop-mesh-accept-sig-v1\x00"
ACCEPT_MAC_LABEL = b"lop-mesh-accept-mac-v1\x00"


def lp(chunk: bytes) -> bytes:
    """Length-prefix one transcript field: ``len(x).to_bytes(4, "big") + x``.

    Unambiguous framing INSIDE the hash, so concatenating two fields can never
    produce the same transcript as concatenating another pair (the classic
    ``"a"+"bc" == "ab"+"c"`` bound).
    """
    return len(chunk).to_bytes(4, "big") + chunk


def transcript(hello: dict[str, Any], challenge: dict[str, Any], auth_core: Any) -> bytes:
    """``T`` — the bytes both sides sign, MAC and derive keys from.

    THE TRANSCRIPT MUST BE IDENTICAL ON BOTH SIDES, and that is why the role byte
    below is the INITIATOR's marker rather than each side's own role. §6.2's
    notation reads as ``b"D"`` for a dialer and ``b"L"`` for a listener, but a
    per-side byte cannot live inside ``T``: the signature is over ``sha256(T)``, so
    a listener hashing ``b"L"`` while the dialer hashes ``b"D"`` can never verify
    the other's signature and the handshake fails 100% of the time. That is not a
    theory — it is what the first implementation did, and the test suite caught it
    at once.

    Reflection is still prevented, by the two mechanisms that do not need the two
    sides to disagree about anything:

    * frame POSITION is bound — ``hello`` is by construction the initiator's frame
      and ``challenge`` the acceptor's — so a peer that echoes an initiator's own
      hello back at it hashes that hello in the challenge slot;
    * the per-direction KEY SPLIT (``link_keys``) makes a record reflected at its
      author decrypt under the opposite direction's key, which fails the tag.
    """
    return b"".join(
        (
            TRANSCRIPT_PREFIX,
            lp(ROLE_DIALER),
            lp(canonical_json(hello)),
            lp(canonical_json(challenge)),
            lp(canonical_json(auth_core)),
        )
    )


def transcript_digest(transcript_bytes: bytes) -> bytes:
    return sha256(transcript_bytes)


def welcome_core(frame: dict[str, Any]) -> dict[str, Any]:
    """``W``: the welcome frame minus its ``proof`` — what the proof commits to.

    EVERY other field is covered, ``endpoints``/``phase``/``inviter``/``nets``
    included, so an on-path rewrite of any of them breaks the proof (G4). Computed
    from the PARSED frame on the dialer side and from the built dict on the
    listener side; ``canonical_json`` makes the two byte-identical because every
    welcome value is a str/int/list/dict (no floats).
    """
    return {key: value for key, value in frame.items() if key != "proof"}


def accept_digest(transcript_bytes: bytes, welcome: dict[str, Any]) -> bytes:
    """``digest_L = sha256(label + lp(T) + lp(W))`` — the listener proof's input.

    T is the full transcript (both hellos' worth of nonces, ephemerals, ids, epochs,
    caps, builds and the auth core), so a recorded proof never verifies on a new
    handshake: the dialer's nonce and ephemeral are fresh every time. Never cached.
    """
    return sha256(
        ACCEPT_DIGEST_LABEL + lp(transcript_bytes) + lp(canonical_json(welcome_core(welcome)))
    )


def welcome_proof(
    identity: DeviceIdentity,
    credential: Credential,
    transcript_bytes: bytes,
    welcome: dict[str, Any],
) -> dict[str, str]:
    """The listener's ``proof``: Ed25519 over ``digest_L`` + HMAC under the dialer's key.

    The signature proves the listener holds the device key on its member row; the
    MAC proves it holds the SAME epoch (or invite) key the dialer MACed with — so a
    removed device still holding its own Ed25519 key fails once the network has
    rotated, which is the revocation property. The MAC is deliberately NOT under a
    derived link key: the signature already binds the link keys (``th`` and both
    ephemerals are inside T), and mixing a transport key into an identity proof is a
    key-confusion risk. One helper for the relay, the CLI and test fixtures, so no
    hand-built welcome grows its own spelling of the proof.
    """
    digest = accept_digest(transcript_bytes, welcome)
    return {
        "sig": b64u(identity.sign(ACCEPT_SIG_LABEL + digest)),
        "mac": b64u(hmac.new(credential.key, ACCEPT_MAC_LABEL + digest, "sha256").digest()),
    }


def _send_frame(sock: socket.socket, frame: dict[str, Any]) -> None:
    """Write one handshake frame, refusing to send an oversized one.

    Checked on the SEND side as well as the receive side: a listener that answered
    with a challenge past the line budget would be the one to blame for the link
    dying, and the peer's reader would (correctly) refuse it — so failing here
    names the real culprit instead of producing a mysterious close.
    """
    payload = encode_line(frame)
    if len(payload) > MAX_HANDSHAKE_LINE:
        raise HandshakeRefusal(
            REASON_FRAME,
            f"a handshake frame of {len(payload)} bytes exceeds the {MAX_HANDSHAKE_LINE}-byte "
            "limit; not sending it",
        )
    sock.sendall(payload)


# ---------------------------------------------------------------------------
# Frames
# ---------------------------------------------------------------------------


def build_hello(
    *,
    mode: Mode,
    network_id: str,
    epoch: int,
    device_id: str,
    instance_id: str,
    eph_public: bytes,
    nonce: bytes,
    session_protocol: int,
    capabilities: list[str],
    build: dict[str, Any],
    join: dict[str, Any] | None = None,
    endpoints: list[str] | None = None,
) -> dict[str, Any]:
    frame: dict[str, Any] = {
        "net": "hello",
        "v": LINK_VERSION,
        "protocol": session_protocol,
        "mode": mode,
        "network_id": network_id,
        "epoch": epoch,
        "device_id": device_id,
        "instance_id": instance_id,
        "eph": b64u(eph_public),
        "nonce": b64u(nonce),
        "caps": capabilities,
        "build": build,
    }
    # THE DIALER'S OWN ENDPOINTS RIDE THE HELLO, and they are why a listener can
    # record a member row that `_ensure_link` can dial later. Without them the
    # only address a listener ever learned was the OBSERVED source address of the
    # connection, which is an ephemeral port that is closed the moment the link
    # drops — so a paired peer was permanently unreachable (QA round 1, F-2).
    # The hello is inside the transcript, so this field is authenticated before
    # anything acts on it, and both sides hash the SAME bytes (the listener folds
    # in the frame it received), so the SAS is unaffected.
    if endpoints:
        frame["endpoints"] = list(endpoints)
    if mode == "join":
        # A join carries the invitation id and the joiner's own device key; the
        # MAC over the transcript is what proves possession of the token, so the
        # token itself never crosses the wire.
        frame["join"] = join or {}
    return frame


#: How many endpoints a handshake may declare, and how long one may be. The
#: list lands in a durable member row that `_ensure_link` dials, so an
#: unbounded or absurd one is a dial storm waiting to happen: a peer can only
#: ever be asked to dial what it learned here.
MAX_DECLARED_ENDPOINTS = 8
MAX_ENDPOINT_CHARS = 255


def clean_endpoints(value: Any) -> list[str]:
    """Validate a peer's declared endpoints: ``host:port`` strings, bounded.

    Anything malformed is DROPPED rather than refused: a peer that declares one
    bad endpoint and one good one is still reachable, and refusing the whole
    handshake over a cosmetic field would be a worse failure than ignoring it.
    The value is used only to fill a member row, never as authority — the link
    is authenticated by the key, not by the address it claims.
    """
    if not isinstance(value, (list, tuple)):
        return []
    cleaned: list[str] = []
    for item in value:
        text = str(item).strip()
        if not text or len(text) > MAX_ENDPOINT_CHARS:
            continue
        address, _, port_text = text.rpartition(":")
        if not address or not port_text.isdigit() or not 0 < int(port_text) <= 65535:
            continue
        if text not in cleaned:
            cleaned.append(text)
        if len(cleaned) >= MAX_DECLARED_ENDPOINTS:
            break
    return cleaned


def build_join_block(*, invite_id: str, joiner_public_key: str, joiner_name: str) -> dict[str, Any]:
    return {
        "invite_id": invite_id,
        "joiner_public_key": joiner_public_key,
        "joiner_name": joiner_name,
    }


def build_challenge(
    *,
    epoch: int,
    device_id: str,
    instance_id: str,
    eph_public: bytes,
    nonce: bytes,
    salt: bytes,
    capabilities: list[str],
    build: dict[str, Any],
) -> dict[str, Any]:
    # A rotated listener's ``rotation`` statement is added by ``send_challenge``.
    return {
        "net": "challenge",
        "v": LINK_VERSION,
        "epoch": epoch,
        "device_id": device_id,
        "instance_id": instance_id,
        "eph": b64u(eph_public),
        "nonce": b64u(nonce),
        "salt": b64u(salt),
        "caps": capabilities,
        "build": build,
    }


def build_auth_core(*, mode: Mode, device_id: str, epoch: int) -> dict[str, Any]:
    """The ``auth`` frame WITHOUT ``sig``/``mac`` — the third transcript field.

    ``locality: "remote"`` is carried here for the same reason every forwarded
    frame carries it: the receiver's authorisation decisions must be able to see
    that this arrived over a peer link, and a link's own auth frame is where that
    fact is most obviously true.
    """
    return {
        "net": "auth",
        "v": LINK_VERSION,
        "mode": mode,
        "device_id": device_id,
        "epoch": epoch,
        "locality": "remote",
    }


@dataclass(frozen=True)
class Credential:
    """What the dialer MACs the transcript with, and what the listener checks.

    Member mode uses the epoch key (current first, previous for reconcile); join
    mode uses the INVITE key, because the joiner's only credential at that point
    is the token — it does not hold a network secret yet.
    """

    label: Literal["epoch", "invite"]
    epoch: int
    key: bytes


class ListenerPolicy(Protocol):
    """Every question the handshake is allowed to ask about membership.

    Deliberately a protocol with a handful of methods rather than a relay object:
    the verification ORDER is the security property, and a test can only prove the
    order if it can answer each question independently — including the ones it
    would never reach (a policy whose ``member_public_key`` would explode the test
    on a removed device is how "checked before any key is tried" is proven rather
    than asserted).
    """

    @property
    def current_epoch(self) -> int: ...

    @property
    def previous_epoch(self) -> int | None: ...

    def network_known(self, network_id: str) -> bool: ...

    def trust_active(self, network_id: str) -> bool: ...

    def epoch_key(self, network_id: str, epoch: int) -> bytes | None: ...

    def active_member_public_key(self, network_id: str, device_id: str) -> str | None: ...


@dataclass
class StaticPolicy:
    """A policy a test (or a doctor probe) can fill in without a relay."""

    current: int = 1
    previous: int | None = None
    networks: dict[str, bool] = field(default_factory=dict)  # network_id -> trust active
    epoch_keys: dict[tuple[str, int], bytes] = field(default_factory=dict)
    members: dict[tuple[str, str], str] = field(default_factory=dict)  # (net, device) -> pubkey

    @property
    def current_epoch(self) -> int:
        return self.current

    @property
    def previous_epoch(self) -> int | None:
        return self.previous

    def network_known(self, network_id: str) -> bool:
        return network_id in self.networks

    def trust_active(self, network_id: str) -> bool:
        return bool(self.networks.get(network_id))

    def epoch_key(self, network_id: str, epoch: int) -> bytes | None:
        return self.epoch_keys.get((network_id, epoch))

    def active_member_public_key(self, network_id: str, device_id: str) -> str | None:
        return self.members.get((network_id, device_id))


@dataclass(frozen=True)
class DialerExpectation:
    """Who the DIALER meant to reach, and where it reads that device's key from.

    WHY THIS EXISTS: a dialer that learns the peer's id from the peer is a dialer
    anyone can answer (G1). ``device_id`` is this device's OWN record of the target
    (the member row it is dialling, or the invite token's ``inviter_device_id``),
    and ``public_key_for`` reads the ACTIVE member row's stored key — never a key
    from a frame. Join mode leaves ``public_key_for`` unset: the joiner has no
    member table yet, so the key comes from ``welcome.inviter.public_key`` and is
    accepted only when its fingerprint equals ``device_id`` (the token's inviter).
    """

    device_id: str
    public_key_for: Callable[[str], str | None] | None = None


LINK_VERSION = MESH_PROTOCOL_VERSION


# ---------------------------------------------------------------------------
# The state machine
# ---------------------------------------------------------------------------


@dataclass
class Handshake:
    """One side of one handshake. Both roles, one object, one transcript.

    The state is explicit rather than hidden inside two role classes because the
    transcript is a function of ALL of it: a field that one role holds and the
    other forgets is a transcript mismatch, and a mismatch that only shows up as
    "the other side refused" is the hardest kind of bug to read from a log.
    """

    role: Role
    identity: DeviceIdentity
    mode: Mode
    network_id: str
    epoch: int
    instance_id: str
    session_protocol: int
    capabilities: list[str] = field(default_factory=lambda: [MESH_NET_V1])
    build: dict[str, Any] = field(default_factory=dict)
    self_device_id: str = ""
    link_id: bytes = b""
    #: Minted here and NEVER on the wire; an AAD field, so a record cannot be
    #: transplanted between two links of the same pair.
    eph_private: Any = None  # X25519PrivateKey
    eph_public: bytes = b""
    nonce: bytes = b""
    salt: bytes = b""
    hello: dict[str, Any] = field(default_factory=dict)
    challenge: dict[str, Any] = field(default_factory=dict)
    auth_core: dict[str, Any] = field(default_factory=dict)
    shared: bytes = b""
    peer_device_id: str = ""

    #: True once the PEER has proved the id in ``peer_device_id`` rather than declared
    #: it: on the listener, :meth:`verify_auth` matched the dialer's sig and MAC; on the
    #: dialer, :meth:`verify_welcome` matched the listener's welcome proof against the
    #: stored key of the device it meant to reach. It becomes
    #: ``HandshakeResult.peer_verified``, which ``_admit_link`` requires. Also read by
    #: the refusal audit (``relay._audit_handshake_refusal``), which may not name a
    #: device as the actor on the strength of a claim: an attacker chooses the id it
    #: declares.
    auth_verified: bool = False
    peer_instance_id: str = ""
    peer_public_key: str = ""
    peer_capabilities: list[str] = field(default_factory=list)
    peer_build: dict[str, Any] = field(default_factory=dict)
    #: Where THIS device says peers can reach it (``advertise_hosts`` plus
    #: detected addresses), carried on the hello we send.
    self_endpoints: list[str] = field(default_factory=list)
    #: Where the PEER says it can be reached. Set from the peer's hello when we
    #: listen and from its ``welcome`` when we dial, because the two roles learn
    #: different frames first.
    peer_endpoints: list[str] = field(default_factory=list)
    phase: LinkPhase = "member"
    credential: Credential | None = None
    join_block: dict[str, Any] = field(default_factory=dict)
    #: The typed SAS, for the joining side's transcription step only.
    typed_sas: str = ""
    #: The welcome frame as verified (dialer side only; ``{}`` until
    #: :meth:`verify_welcome` passes). Join callers read ``inviter``/``network``
    #: from here rather than a second parse of an unverified copy.
    welcome: dict[str, Any] = field(default_factory=dict)
    #: LISTENER: this device's own rotation statement (``MemberRecord.rotation_proof``
    #: on its self row, signed by its PREVIOUS key), sent in the challenge so a dialer
    #: still holding the pre-rotation row can follow the hop (mesh-listener-auth §6.1,
    #: G1D-1). ``{}`` for a device that never rotated, and on join.
    self_rotation: dict[str, Any] = field(default_factory=dict)
    #: DIALER: the listener's rotation statement once :meth:`refuse_unexpected_peer`
    #: VERIFIED it against the stored key of the device this dial meant to reach.
    #: ``{}`` unless that check passed. Read by :meth:`verify_welcome` (the key the
    #: proof must verify under) and by ``RelayServer.dial`` (which applies the hop
    #: through ``apply_device_rotation``, the ``net_identity_rotate`` code path).
    peer_rotation: dict[str, Any] = field(default_factory=dict)

    # -- construction -------------------------------------------------------

    @classmethod
    def new(
        cls,
        *,
        role: Role,
        identity: DeviceIdentity,
        network_id: str,
        epoch: int,
        instance_id: str,
        session_protocol: int,
        mode: Mode = "member",
        capabilities: list[str] | None = None,
        build: dict[str, Any] | None = None,
        endpoints: list[str] | None = None,
    ) -> Handshake:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey

        private = X25519PrivateKey.generate()
        public = private.public_key().public_bytes(
            encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw
        )
        return cls(
            role=role,
            identity=identity,
            mode=mode,
            network_id=network_id,
            epoch=epoch,
            instance_id=instance_id,
            session_protocol=session_protocol,
            capabilities=list(capabilities or [MESH_NET_V1]),
            build=dict(build or {}),
            self_device_id=identity.device_id,
            self_endpoints=clean_endpoints(endpoints or []),
            # Overwritten by `establish` with the DERIVED id; this placeholder exists
            # only so the dataclass field is never unset before the handshake completes.
            link_id=b"",
            eph_private=private,
            eph_public=public,
            nonce=token_bytes(32),
        )

    # -- dialer steps -------------------------------------------------------

    def send_hello(self, sock: socket.socket) -> dict[str, Any]:
        join = None
        if self.mode == "join":
            join = dict(self.join_block)
        self.hello = build_hello(
            mode=self.mode,
            network_id=self.network_id,
            epoch=self.epoch,
            device_id=self.self_device_id,
            instance_id=self.instance_id,
            eph_public=self.eph_public,
            nonce=self.nonce,
            session_protocol=self.session_protocol,
            capabilities=self.capabilities,
            build=self.build,
            join=join,
            endpoints=self.self_endpoints,
        )
        _send_frame(sock, self.hello)
        return self.hello

    def read_challenge(self, reader: FrameReader, deadline: float) -> dict[str, Any]:
        frame = reader.read_line(deadline)
        if frame.get("net") != "challenge":
            raise HandshakeRefusal(
                REASON_FRAME, "the other device did not answer with a challenge frame"
            )
        for key in ("eph", "nonce", "salt", "device_id", "instance_id"):
            if not frame.get(key):
                raise HandshakeRefusal(
                    REASON_FRAME, f"the challenge frame is missing its {key!r} field"
                )
        self.challenge = frame
        self.peer_device_id = str(frame["device_id"])
        self.peer_instance_id = str(frame["instance_id"])
        self.peer_capabilities = [str(cap) for cap in frame.get("caps") or []]
        self.peer_build = dict(frame.get("build") or {})
        return frame

    def send_auth(self, sock: socket.socket, credential: Credential) -> dict[str, Any]:
        self.credential = credential
        self.auth_core = build_auth_core(
            mode=self.mode, device_id=self.self_device_id, epoch=self.epoch
        )
        digest = transcript_digest(transcript(self.hello, self.challenge, self.auth_core))
        signature = self.identity.sign(b"lop-mesh-auth-v1\x00" + digest)
        mac = hmac.new(credential.key, b"lop-mesh-mac-v1\x00" + digest, "sha256").digest()
        frame = dict(self.auth_core)
        frame["sig"] = b64u(signature)
        frame["mac"] = b64u(mac)
        _send_frame(sock, frame)
        return frame

    def read_welcome(
        self, reader: FrameReader, deadline: float, expect: DialerExpectation
    ) -> dict[str, Any]:
        """Read ``welcome`` — the last plaintext frame — VERIFY it, then finish.

        ``expect`` is REQUIRED, with no default: a dialer that does not say who it
        meant to reach cannot verify who answered, and an optional argument is how
        a new call site would quietly reopen G1.
        """
        frame = reader.read_line(deadline)
        self.verify_welcome(frame, expect)
        return frame

    def refuse_unexpected_peer(self, expect: DialerExpectation) -> None:
        """Refuse BEFORE ``send_auth`` when the challenge names another device.

        The challenge is unauthenticated, so a match here proves nothing (that is
        :meth:`verify_welcome`'s job), but a MISMATCH is decisive: an honest listener
        names its own id, so either the address is stale or the peer is lying, and
        this device would refuse at step 5 anyway. Refusing here, before the auth
        frame, matters for the honest case: a listener that receives a valid auth
        completes its side and registers the link, and its duplicate fence
        (``register_link``, newest wins) then EVICTS the real, older link from this
        device — so a stale address pointing at member C would cost the working
        link to C. It also keeps this device's auth tag away from the wrong party.
        """
        claimed = str(self.challenge.get("device_id") or "")
        if expect.device_id and claimed == expect.device_id:
            return
        if self._accept_rotation_hop(claimed, expect):
            return
        if self.mode == "join":
            raise HandshakeRefusal(
                REASON_INVITE_INVITER,
                "the device that answered is not the one that issued this invite",
            )
        raise HandshakeRefusal(
            REASON_PEER_IDENTITY,
            f"the device that answered as {claimed or 'unknown'} is not "
            f"{expect.device_id or 'the device this dial meant to reach'}; the address "
            "on that member row is stale or redirected. No link was made",
        )

    def _accept_rotation_hop(self, claimed: str, expect: DialerExpectation) -> bool:
        """Whether the challenge's mismatched id is the PROVEN successor of the target.

        WHY (G1D-1): a member that rotated its device key while this device was
        offline answers at its old address under its NEW id. Refusing that as a
        wrong device split the mesh for good — the queued ``net_identity_rotate``
        rides a link that could now never form. The listener therefore carries its
        rotation statement in the challenge, and this lets the dial continue to
        ``send_auth`` only when ALL of these hold, each against what THIS device
        already holds rather than the frame:

        * member mode with a stored-key reader (a joiner has no member table);
        * ``old_device_id`` is the device this dial meant to reach and
          ``new_device_id`` is the id the challenge names;
        * the statement names this network (signed inside the body);
        * ``verify_rotation_statement`` passes against the STORED key of the target
          (the old id is that key's fingerprint, the new id the carried key's, and
          ``sig_old`` verifies).

        WHY SENDING AUTH IS SAFE HERE: the statement is public (it rides on every
        copy of the member table), so passing this check proves nothing about who
        is answering — exactly as a matching id proves nothing. It only lets the
        handshake reach :meth:`verify_welcome`, which then requires the proof to
        verify under the statement's ``new_public_key`` and the MAC under this
        side's credential. A party that replays the statement gets the same auth
        frame an impostor claiming the target's own id already gets, and fails at
        step 7 (``bad_signature``), which demotes nothing (``_note_listener_refusal``).
        One hop only: the self row carries the LAST statement, so a device that
        rotated twice while this one was offline still needs the table route.
        """
        from local_operator.network.identity import verify_rotation_statement

        statement = self.challenge.get("rotation")
        if (
            self.mode != "member"
            or not isinstance(statement, dict)
            or not expect.device_id
            or expect.public_key_for is None
            or not claimed
            or claimed == self.self_device_id
        ):
            return False
        if (
            statement.get("old_device_id") != expect.device_id
            or statement.get("new_device_id") != claimed
            or statement.get("network_id") != self.network_id
        ):
            return False
        stored = expect.public_key_for(expect.device_id)
        if not stored:
            return False
        try:
            verify_rotation_statement(statement, stored)
        except (HandshakeRefusal, KeyError, TypeError, ValueError):
            return False
        self.peer_rotation = dict(statement)
        return True

    def verify_welcome(self, frame: dict[str, Any], expect: DialerExpectation) -> None:
        """The dialer's half of mutual authentication: steps 1-9, in this order.

        THREAT: whoever answers a dial — an on-path attacker, a poisoned endpoint, a
        removed device, or an honest member at a stale address — would otherwise
        become the member this device meant to reach, and receive the member-table
        pull (G2), the rotated epoch secret (G3) and a durable endpoint redirect
        (G4). Every check runs against what THIS device already holds: the stored
        member key (or, for a join, the token's inviter fingerprint) and the
        credential it MACed with. Nothing from the frame is trusted until step 9,
        and each failure raises before ``establish`` derives link keys.

        NO-DOWNGRADE WITHOUT A VERSION BUMP: the proof is required unconditionally
        on every welcome, whatever the challenge advertises, so there is nothing to
        negotiate and nothing to strip. ``MESH_PROTOCOL_VERSION`` stays 1 so an OLD
        dialer still links to a NEW listener (it ignores the unknown ``proof`` key)
        — that inbound link is how a not-yet-updated member receives its update.
        """
        # STEP 1: frame name and link version (as before).
        if frame.get("net") != "welcome":
            raise HandshakeRefusal(
                REASON_FRAME, "the other device did not answer with a welcome frame"
            )
        # The peer's LINK version is checked here rather than at hello, so an
        # unknown version is a refusal AFTER authentication instead of an oracle
        # any stranger can pull.
        check_link_version(frame.get("v"), mine=LINK_VERSION)
        claimed = str(self.challenge.get("device_id") or "")
        # STEP 2: a proof is present and decodes. Absent = an old listener or an
        # impostor that cannot make one; the dialer cannot tell which, and refuses
        # both — the reason names the build the challenge REPORTED (a claim).
        proof = frame.get("proof")
        sig_bytes = mac_bytes = b""
        if isinstance(proof, dict):
            try:
                sig_bytes = unb64u(str(proof.get("sig") or ""))
                mac_bytes = unb64u(str(proof.get("mac") or ""))
            except ValueError:
                sig_bytes = mac_bytes = b""
        if len(sig_bytes) != 64 or len(mac_bytes) != 32:
            version = str((self.challenge.get("build") or {}).get("version") or "unknown")
            if self.mode == "join":
                # A JOINER IS NOT A MEMBER: it cannot run `lop network update`, and a
                # fresh token from the same not-yet-updated inviter fails the same way,
                # so the only way forward is updating the INVITER (review N2, UX U2).
                # The build is the challenge's claim, so it is "reports", never "older".
                raise HandshakeRefusal(
                    REASON_LISTENER_UNPROVEN,
                    f"the inviting device reports build {version} and could not prove its "
                    "identity, so this device stopped before joining; nothing was shared. "
                    "On the inviting device, run `lop update`, then mint a new invite there "
                    "with `lop network invite` and join again",
                )
            # No literal ``<name>`` (N-2): this layer knows only the claimed id, so the
            # sentence names both remedies without one; ``peers``/``doctor`` name it.
            raise HandshakeRefusal(
                REASON_LISTENER_UNPROVEN,
                f"{claimed or 'the other device'} did not prove its identity: its welcome "
                f"carried no valid signature. It reports build {version}, or it is not the "
                "device it claims to be. This device no longer links to peers it cannot "
                "verify; update it from a member (`lop network update`), or run "
                "`lop update` on that device",
            )
        # STEP 3: the welcome agrees with the challenge about who is speaking.
        if str(frame.get("device_id") or "") != claimed or not isinstance(frame.get("epoch"), int):
            raise HandshakeRefusal(
                REASON_FRAME, "the welcome frame names a different device than the challenge"
            )
        # STEP 4: never link to ourselves.
        if claimed == self.self_device_id:
            raise HandshakeRefusal(REASON_SELF, "the other end of this dial is this device")
        # STEP 5: the claimed id only SELECTS which stored key to verify against; it
        # must be the device this side meant to reach. The evidence is step 7.
        # The ONE exception is a rotation hop :meth:`refuse_unexpected_peer` already
        # verified against the target's stored key (G1D-1); the proof below must then
        # verify under that statement's new key, so nothing is taken on trust.
        rotated = bool(self.peer_rotation) and claimed == self.peer_rotation.get("new_device_id")
        if (claimed != expect.device_id and not rotated) or not expect.device_id:
            if self.mode == "join":
                raise HandshakeRefusal(
                    REASON_INVITE_INVITER,
                    "the device that answered is not the one that issued this invite",
                )
            raise HandshakeRefusal(
                REASON_PEER_IDENTITY,
                f"the device that answered as {claimed or 'unknown'} is not "
                f"{expect.device_id or 'the device this dial meant to reach'}; the address "
                "on that member row is stale or redirected. No link was made",
            )
        # STEP 6: the verification key, from THIS device's own record — never the frame
        # (join: the frame's key, accepted ONLY because its fingerprint is the token's).
        if self.mode == "join":
            inviter = frame.get("inviter")
            key = str(inviter.get("public_key") or "") if isinstance(inviter, dict) else ""
            try:
                fingerprint_ok = bool(key) and _derived_id(key) == expect.device_id
            except ValueError:
                fingerprint_ok = False
            if not fingerprint_ok:
                raise HandshakeRefusal(
                    REASON_INVITE_DEVICE,
                    "the inviter's key is not the one this invite was issued by",
                )
        elif rotated:
            # The successor's key, bound to ``claimed`` by fingerprint and to the
            # target's STORED key by ``sig_old`` (both checked before our auth left).
            key = str(self.peer_rotation.get("new_public_key") or "")
        else:
            stored = expect.public_key_for(claimed) if expect.public_key_for else None
            if not stored:
                raise HandshakeRefusal(
                    REASON_MEMBER, "the other device is not an active member of this network"
                )
            key = stored
        digest = accept_digest(self.transcript_bytes, frame)
        # STEP 7: the listener's signature against that key.
        _verify_signature(
            key,
            b64u(sig_bytes),
            digest,
            label=ACCEPT_SIG_LABEL,
            sentence=(
                "the listener's signature over the handshake transcript does not verify "
                "against the public key on its member record"
            ),
        )
        # STEP 8: the MAC under the SAME credential this side MACed its auth with —
        # constant-time, so the comparison leaks nothing about the expected tag.
        # A refusal, not an ``assert``: ``python -O`` strips asserts, and a mis-ordered
        # call must still end as a named refusal rather than an AttributeError (N1).
        if self.credential is None:
            raise HandshakeRefusal(
                REASON_FRAME, "the welcome arrived before this device sent its auth frame"
            )
        expected_mac = hmac.new(self.credential.key, ACCEPT_MAC_LABEL + digest, "sha256").digest()
        if not hmac.compare_digest(expected_mac, mac_bytes):
            raise HandshakeRefusal(
                REASON_MAC,
                "the listener could not prove it holds this network's current secret",
            )
        # STEP 9: only now is anything read from the frame.
        self.auth_verified = True
        self.peer_public_key = key
        self.welcome = dict(frame)
        self.phase = cast(LinkPhase, str(frame.get("phase", "member")))
        # Where the peer can be reached — now under the listener's signature and
        # MAC, so a rewritten endpoint fails step 7/8 instead of being persisted.
        self.peer_endpoints = clean_endpoints(frame.get("endpoints"))
        self.establish()

    # -- listener steps -----------------------------------------------------

    def read_hello(self, reader: FrameReader, deadline: float) -> dict[str, Any]:
        return self.accept_hello(reader.read_line(deadline))

    def accept_hello(self, frame: dict[str, Any]) -> dict[str, Any]:
        """Validate and fold in a hello frame.

        Public and separate from the read because the accept path PEEKS the first
        line (to learn which network and mode it claims, before a policy object can
        exist) and then hands the same frame here — one implementation of "what a
        hello means", so the peeking path cannot drift from the reading one.
        """
        if frame.get("net") != "hello":
            raise HandshakeRefusal(REASON_FRAME, "the first frame was not a hello frame")
        check_link_version(frame.get("v"), mine=LINK_VERSION)
        if frame.get("network_id") != self.network_id:
            # A network this install is not in is refused with the same silence as
            # everything else; naming the difference would let a stranger probe
            # which networks exist here.
            raise HandshakeRefusal(REASON_UNKNOWN_NETWORK, "no such network on this device")
        if frame.get("mode") != self.mode:
            raise HandshakeRefusal(
                REASON_FRAME,
                "the other device offered a different handshake mode than the one this link "
                "was opened for",
            )
        if frame.get("device_id") and str(frame["device_id"]) == self.identity.device_id:
            # Step 5 of the verification order. An accidental self-connection (a
            # member row for a device dialling its own advertised endpoint) is
            # otherwise a confusing silent no-op, and naming it costs one comparison.
            raise HandshakeRefusal(REASON_SELF, "a device dialled its own endpoint and was refused")
        for key in ("eph", "nonce", "device_id", "instance_id"):
            if not frame.get(key):
                raise HandshakeRefusal(REASON_FRAME, f"the hello frame is missing its {key!r}")
        self.hello = frame
        self.peer_device_id = str(frame["device_id"])
        self.peer_instance_id = str(frame["instance_id"])
        self.peer_capabilities = [str(cap) for cap in frame.get("caps") or []]
        self.peer_build = dict(frame.get("build") or {})
        self.peer_endpoints = clean_endpoints(frame.get("endpoints"))
        self.join_block = dict(frame.get("join") or {})
        return frame

    def send_challenge(self, sock: socket.socket, policy: ListenerPolicy) -> dict[str, Any]:
        """Validate everything the inviter must before showing a challenge.

        Epoch, trust and the invite all belong HERE rather than after the
        challenge: the challenge is the first frame that tells an attacker
        anything, and a listener that challenged first and validated second would
        hand a stranger a fresh X25519 public key and salt before deciding whether
        it would speak to them at all.
        """
        if not policy.network_known(self.network_id):
            raise HandshakeRefusal(REASON_UNKNOWN_NETWORK, "no such network on this device")
        if not policy.trust_active(self.network_id):
            raise HandshakeRefusal(
                REASON_UNTRUSTED,
                "this network is marked untrusted on this device; re-admit it with "
                "`lop network trust` before it accepts links",
            )
        if self.mode == "join":
            # THE CALLER HAS ALREADY VALIDATED THE INVITE before this frame is
            # built, and the challenge is the FIRST frame an attacker learns
            # anything from, so nothing about the token may be decided after it.
            # Note what the caller has NOT done: written `redeemed`. That happens
            # once the auth frame proves the peer holds the invite material (see
            # ``RelayServer._run_inbound_handshake``), because a peer that sends a
            # hello and vanishes has proved nothing. Validating here as well would
            # be a second implementation of the same decision (and a second place
            # for the two to disagree), so the only check left is that the
            # credential it derived actually arrived.
            if self.credential is None:
                raise HandshakeRefusal(
                    REASON_INVITE,
                    "a join handshake was started without a validated invite",
                )
            self.phase = "pair"
        else:
            if self.epoch not in (policy.current_epoch, policy.previous_epoch):
                raise HandshakeRefusal(
                    REASON_EPOCH,
                    "the other device is at an epoch this one no longer accepts; it must "
                    "reconcile or re-pair",
                )
        self.salt = token_bytes(16)
        self.challenge = build_challenge(
            epoch=policy.current_epoch,
            device_id=self.self_device_id,
            instance_id=self.instance_id,
            eph_public=self.eph_public,
            nonce=self.nonce,
            salt=self.salt,
            capabilities=self.capabilities,
            build=self.build,
        )
        if self.mode == "member" and self.self_rotation:
            # Our rotation statement, for a dialer that still holds our previous row
            # (G1D-1; ``Handshake._accept_rotation_hop``). INSIDE the transcript, so
            # the welcome proof commits to it, and harmless to an old dialer, which
            # hashes the parsed challenge as it arrived and ignores unknown keys.
            self.challenge["rotation"] = dict(self.self_rotation)
        _send_frame(sock, self.challenge)
        return self.challenge

    def send_welcome(self, sock: socket.socket, frame: dict[str, Any]) -> None:
        """Send the last PLAINTEXT frame. Everything after it rides in a record."""
        _send_frame(sock, frame)

    def verify_auth(
        self, reader: FrameReader, deadline: float, policy: ListenerPolicy
    ) -> dict[str, Any]:
        """Steps 3-9 of §6.2, in order. Any failure raises and the caller closes."""
        frame = reader.read_line(deadline)
        if frame.get("net") != "auth":
            raise HandshakeRefusal(REASON_FRAME, "the third frame was not an auth frame")
        signature = frame.get("sig")
        mac = frame.get("mac")
        if not signature or not mac:
            raise HandshakeRefusal(REASON_FRAME, "the auth frame carried no signature or MAC")
        self.auth_core = {k: v for k, v in frame.items() if k not in ("sig", "mac")}
        if self.auth_core.get("device_id") != self.peer_device_id:
            raise HandshakeRefusal(
                REASON_FRAME,
                "the auth frame names a different device than the hello frame did",
            )
        if self.auth_core.get("mode") != self.mode:
            raise HandshakeRefusal(REASON_FRAME, "the auth frame changed the handshake mode")
        if self.auth_core.get("epoch") != policy.current_epoch and self.mode == "member":
            # A member link may authen at the previous epoch only to reconcile.
            if self.auth_core.get("epoch") != policy.previous_epoch:
                raise HandshakeRefusal(
                    REASON_EPOCH, "the auth frame names an epoch this device does not accept"
                )

        # STEP 6, before any key is used: membership is evaluated against the
        # CURRENT epoch's member list, and a tombstoned device resolves to None
        # here no matter which epoch's key it holds. This is what makes revocation
        # take effect without visiting the removed device.
        if self.mode == "member":
            public_key = policy.active_member_public_key(self.network_id, self.peer_device_id)
            if public_key is None:
                raise HandshakeRefusal(
                    REASON_MEMBER, "the other device is not a member of this network"
                )
        else:
            public_key = str(self.join_block.get("joiner_public_key") or "")
            if not public_key:
                raise HandshakeRefusal(
                    REASON_FRAME, "the join request carried no public key to verify against"
                )
            if _derived_id(public_key) != self.peer_device_id:
                raise HandshakeRefusal(
                    REASON_INVITE_DEVICE,
                    "the joining device's id is not the fingerprint of the public key it sent",
                )
        self.peer_public_key = public_key

        digest = transcript_digest(transcript(self.hello, self.challenge, self.auth_core))
        # STEP 7: the signature, against the STORED key. An id is a name; the key
        # on the member row is the authority, which is why a copied device key is
        # indistinguishable while a changed one is detected.
        _verify_signature(public_key, signature, digest)

        # STEP 8: the MAC, under the key of the epoch THIS FRAME NAMES — step 4 has
        # already bounded that to {current, current-1} — which is what puts a
        # previous-epoch link in the reconcile phase.
        if self.mode == "join":
            assert self.credential is not None
            keys = [self.credential]
        else:
            epoch = int(self.auth_core.get("epoch") or policy.current_epoch)
            current = policy.epoch_key(self.network_id, epoch)
            keys = []
            if current is not None:
                keys.append(Credential("epoch", epoch, current))
        matched: Credential | None = None
        for credential in keys:
            expected = hmac.new(credential.key, b"lop-mesh-mac-v1\x00" + digest, "sha256").digest()
            if hmac.compare_digest(expected, unb64u(str(mac))):
                matched = credential
                break
        if matched is None:
            raise HandshakeRefusal(
                REASON_MAC,
                "the other device could not prove it holds this network's current secret",
            )
        # THE PEER HAS NOW PROVED THE ID IT DECLARED: the MAC is over a transcript
        # that carries ``device_id``, computed with the key of the epoch it named,
        # so everything this side has accepted about the peer's identity is proven
        # from here on. It is a flag rather than a re-derivation at the audit site
        # because the audit asks a question the handshake's own state answers exactly
        # once (relay._audit_handshake_refusal), and "was this id proven" must not
        # become a second implementation of the MAC check.
        self.auth_verified = True
        if self.mode == "member":
            self.phase = (
                "reconcile"
                if int(self.auth_core.get("epoch") or policy.current_epoch) != policy.current_epoch
                else "member"
            )
        else:
            self.phase = "pair"
        self.credential = matched
        self.establish()
        return frame

    # -- shared -------------------------------------------------------------

    def establish(self) -> HandshakeResult:
        """Compute ``Z``, the link keys and the SAS. Idempotent.

        The peer's ephemeral comes from the frame the OTHER side authored: the
        listener read it in ``hello``, the dialer in ``challenge``. Getting this
        backwards is not a subtle bug — every key would be derived from this
        side's own ephemeral and both sides would silently disagree — which is
        why the choice is one line here rather than a parameter at four call
        sites.
        """
        from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PublicKey

        peer_frame = self.hello if self.role == "listener" else self.challenge
        peer_public = unb64u(str(peer_frame.get("eph") or ""))
        if len(peer_public) != 32:
            raise HandshakeRefusal(
                REASON_FRAME, "the other device's ephemeral key is not a 32-byte X25519 key"
            )
        shared = self.eph_private.exchange(X25519PublicKey.from_public_bytes(peer_public))
        self.shared = shared
        digest = transcript_digest(transcript(self.hello, self.challenge, self.auth_core))
        # No link_id argument: it is DERIVED from (Z, th) so both sides agree by
        # construction (a per-side id would be part of every record's AAD and every
        # tag would fail).
        keys = link_keys(shared, digest)
        self.link_id = keys.link_id
        return HandshakeResult(
            role=self.role,
            peer_device_id=self.peer_device_id,
            peer_instance_id=self.peer_instance_id,
            peer_public_key=self.peer_public_key,
            network_id=self.network_id,
            epoch=int(self.auth_core.get("epoch") or self.epoch),
            phase=self.phase,
            link_id=self.link_id.hex(),
            keys=keys,
            sas=sas_code(shared, digest),
            transcript_hash=digest.hex(),
            session_protocol=int(self.hello.get("protocol") or self.session_protocol),
            peer_build=self.peer_build,
            peer_capabilities=self.peer_capabilities,
            local_capabilities=list(self.capabilities),
            # Listener: set by ``verify_auth`` (sig + MAC against the stored key).
            # Dialer: set by ``verify_welcome`` (the listener's proof against the
            # stored key, and peer == expected). ``_admit_link`` refuses False.
            # Reported, never re-derived, so this field cannot disagree with it.
            peer_verified=self.auth_verified,
        )

    @property
    def transcript_bytes(self) -> bytes:
        return transcript(self.hello, self.challenge, self.auth_core)

    @property
    def sas(self) -> str:
        if not self.shared:
            self.establish()
        return sas_code(self.shared, transcript_digest(self.transcript_bytes))

    def codec(self) -> LinkCrypto:
        """The record codec for this link, once the handshake has completed."""
        result = self.establish()
        # COMPRESSION IS THE INTERSECTION, never a local choice (wire.py "Record
        # compression"): ``self.capabilities`` is what THIS end put in its hello or
        # challenge, ``peer_capabilities`` what the other end did, and both sit in
        # the MAC-covered transcript. An old peer advertises neither, so its link
        # gets the codec it always had.
        negotiated = negotiate_capabilities(self.capabilities, self.peer_capabilities)
        return LinkCrypto(result.keys, role=self.role, compression=ZLIB_RECORDS_V1 in negotiated)

    def welcome_frame(
        self,
        *,
        phase: LinkPhase,
        epoch: int,
        nets: list[dict[str, Any]] | None = None,
        capabilities: list[str] | None = None,
        members_digest: str = "",
        network_name: str = "",
        endpoints: list[str] | None = None,
    ) -> dict[str, Any]:
        """The listener's last plaintext frame. The SAS is deliberately absent."""
        frame: dict[str, Any] = {
            "net": "welcome",
            "v": LINK_VERSION,
            "phase": phase,
            "device_id": self.self_device_id,
            "instance_id": self.instance_id,
            "epoch": epoch,
            "session_protocol": self.session_protocol,
            "members_digest": members_digest,
        }
        if capabilities:
            frame["capabilities"] = capabilities
        if phase == "pair":
            frame["network"] = {"network_id": self.network_id, "name": network_name}
            # ``public_key`` lets the JOINER verify the proof: it has no member table
            # yet, and accepts this key only because its fingerprint is the token's
            # ``inviter_device_id`` (design §5). An old joiner ignores the extra key.
            frame["inviter"] = {
                "device_id": self.self_device_id,
                "name": self.identity.name,
                "public_key": self.identity.public_key,
            }
        if endpoints:
            # How the dialer learns where to reach US. Outside the transcript (so the
            # SAS is unaffected) but INSIDE the proof below, so a rewrite fails.
            # A build that does not know this key reads it as decoration.
            frame["endpoints"] = list(endpoints)
        if nets is not None:
            frame["nets"] = nets
        # THE LISTENER'S PROOF, last, over the transcript and every field above. Only
        # a listener that has VERIFIED the dialer's auth holds the matched credential
        # to MAC with; any other caller (a test impostor, a mis-ordered call) sends
        # no proof and a current dialer refuses it ``listener_unproven``.
        if self.role == "listener" and self.auth_verified and self.credential is not None:
            frame["proof"] = welcome_proof(
                self.identity, self.credential, self.transcript_bytes, frame
            )
        return frame


# ---------------------------------------------------------------------------
# Small crypto helpers the two verification steps need
# ---------------------------------------------------------------------------


def _derived_id(public_key_b64: str) -> str:
    from local_operator.network.identity import device_id_for

    return device_id_for(unb64u(public_key_b64))


def _verify_signature(
    public_key_b64: str,
    signature_b64: str,
    digest: bytes,
    *,
    label: bytes = b"lop-mesh-auth-v1\x00",
    sentence: str = (
        "the other device's signature over the handshake transcript does not verify "
        "against the public key on its member record"
    ),
) -> None:
    """Ed25519 verify under ``label``; a decode error is a refusal, never a crash.

    ``label`` defaults to the DIALER's auth label; the listener's welcome proof
    passes ``ACCEPT_SIG_LABEL`` so neither side's signature can be reflected as the
    other's.
    """
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    try:
        Ed25519PublicKey.from_public_bytes(unb64u(public_key_b64)).verify(
            unb64u(signature_b64), label + digest
        )
    except (InvalidSignature, ValueError) as exc:
        raise HandshakeRefusal(REASON_SIGNATURE, sentence) from exc


# ---------------------------------------------------------------------------
# The pair ceremony's frames and the SAS comparison
# ---------------------------------------------------------------------------


def pair_ready_frame(*, req: int, typed_sas: str) -> dict[str, Any]:
    """The joiner's HUMAN-typed code, sent to the inviter.

    It is a transcription, not a yes: a yes/no lets two people each press ``y``
    without comparing anything, and a value received FROM the peer would let a
    relay in the middle echo the inviter's own digits back at it.
    """
    return {"op": "net_pair_ready", "req": req, "sas": typed_sas}


def pair_abort_frame(
    *, req: int, reason: str, detail: str = "", spent: bool = False
) -> dict[str, Any]:
    """The inviter's refusal of a pairing, with the SENTENCE it refused with.

    ``reason`` is the machine code and stays the contract; ``detail`` is the refusing
    device's own sentence, which is the only thing that can explain a refusal the
    joiner cannot see the reason for. Without it every refusal read as
    ``the pairing was refused (device_id_conflict)`` — a code with no remedy, on the
    one screen where the operator has just done something and needs to know what to
    do next (QA round 3, Q-R3-3: the admitting device's message named the burned id
    and the remedy, and all of it was dropped one line before the wire).

    Bounded, because this frame travels inside the handshake's line budget and a
    sentence is not allowed to be the reason a link cannot open.
    """
    frame: dict[str, Any] = {"op": "net_pair_abort", "req": req, "reason": reason}
    text = " ".join(str(detail or "").split())[:PAIR_ABORT_DETAIL_MAX]
    if text:
        frame["detail"] = text
    if spent:
        # ADDITIVE, AND ONLY WHEN TRUE (QA round 1, Q3): the refusing device consumed
        # the invite, so the joiner's remedy must not say "run it again with the SAME
        # token". ``reason`` stays the contract an older joiner keys on; a build that
        # does not know this key keeps its old (retry) sentence.
        frame["spent"] = True
    return frame


def pair_result_frame(
    *,
    req: int,
    admit: bool,
    network: dict[str, Any] | None = None,
    member: dict[str, Any] | None = None,
    members: list[dict[str, Any]] | None = None,
    members_digest: str = "",
    material: str = "",
    rotations: dict[str, str] | None = None,
    shares: list[str] | None = None,
    reduced: list[str] | None = None,
    reason: str = "",
) -> dict[str, Any]:
    """The inviter's admission answer — and the ONLY frame that carries material.

    ``material`` is the current epoch's network secret, and this frame is the
    member's one chance to learn it. It is sent after the member row is written,
    so a member this device has admitted is durable even if the link dies in the
    next millisecond.

    ``members`` is the FULL member list, and it is here because of a gap the
    design left: ``welcome`` carries the inviter's id and name but not its public
    key, so a joiner that received only its own row could never verify a later
    handshake from the inviter. The list is the same shape ``net_epoch`` carries,
    which also means the joiner's persistence code is one path rather than two.

    ``shares`` is the FINAL granted set — the receipt the joiner shows its person
    ("available here: …") and the one fact that reconciles the two screens when a
    share was dropped. Old joiners read named keys only and ignore it; it is
    empty for mixed pairs, where no list ever travelled.

    ``reduced`` names the keys the owner's PERSON removed at the confirm screen
    (the deliberate half of the offer-to-final delta). The joiner needs it to
    keep "the owner chose not to" apart from "the grant failed", which carry
    different receipt sentences (UX round 1, U2). It is additive: an older
    joiner ignores the key, and a relay that predates it sends nothing — both
    then read the whole delta as failed, which is the pre-fix behaviour.
    """
    frame: dict[str, Any] = {"op": "net_pair_result", "req": req, "admit": admit}
    if admit:
        frame["network"] = network or {}
        frame["member"] = member or {}
        frame["members"] = list(members or [])
        frame["members_digest"] = members_digest
        frame["secret"] = material
        frame["rotations"] = rotations or {}
        frame["shares"] = list(shares or [])
        frame["reduced"] = list(reduced or [])
    else:
        frame["reason"] = reason
    return frame


def pair_offer_frame(*, items: list[dict[str, Any]], digest: str) -> dict[str, Any]:
    """The owner's share list — the FIRST sealed record after ``welcome``.

    Sent before anything waits on a human, so both ends can show the same list
    before the joiner types: the owner needs it to decide what to lend, and the
    joiner needs it to see what it will borrow. The ``digest`` is sha256 over the
    CANONICAL PROJECTION of the items (``credentials.offers.digest_of``, whose
    docstring owns the additive-evolution rule): a checksum
    that pins the two screens to one list, NOT a signature — the record's AEAD
    under the pair link keys is the integrity (the frame never existed pre-auth,
    and a tamper without those keys fails the AEAD, which is a hard refusal).

    ``items`` is display data in the design's shape (``credentials/offers.py``
    validates it on both ends): ``{key, kind, label, share}`` per row, labels
    masked by the builder.
    """
    return {"op": "net_pair_offer", "items": [dict(item) for item in items], "digest": digest}


def parse_pair_offer(frame: dict[str, Any]) -> list[dict[str, Any]]:
    """Validate a ``net_pair_offer`` and return its items, or refuse the pairing.

    STRICT, never tolerant: this is a sealed frame from a peer that proved the
    invite, and a frame this side cannot use is a protocol fact — a silent
    downgrade to "no offer" would erase the difference between an old peer (which
    cannot have sent this) and a broken one. The digest is recomputed from the
    bytes as received; a mismatch means the two ends disagree about the list,
    which is exactly what the digest exists to catch.
    """
    from local_operator.network.credentials import offers

    try:
        items = offers.validate_items(frame.get("items"))
    except ValueError as exc:
        raise MeshRefusal(
            "protocol_error", f"the other device's share list is not usable: {exc}"
        ) from exc
    digest = str(frame.get("digest") or "")
    if not digest or digest != offers.digest_of(items):
        raise MeshRefusal(
            "protocol_error",
            "the other device's share list failed its digest check; the pairing is refused",
        )
    return items


def sas_matches(derived: str, typed: str) -> bool:
    """Compare a typed code with this side's own derivation, in constant time.

    ``normalize_sas`` reduces the typed text to six digits or to nothing; anything
    that is not six digits is a mismatch, never a near miss.
    """
    from local_operator.network.wire import normalize_sas

    candidate = normalize_sas(typed)
    if not candidate:
        return False
    return hmac.compare_digest(derived, candidate)


def sas_mismatch_sentence() -> str:
    """What B prints when the codes disagreed: the DIAGNOSIS only.

    THE REMEDY LIVES ONCE, in ``PAIRING_REMEDIES["sas_mismatch"]`` — the
    caps-consistent "SAME token" advice, which is what the forgiving budget
    (Q-XH-6) actually licenses. This sentence used to carry a second,
    differently-worded copy of the same advice, so the composed refusal said it
    twice in two voices (~470 cols; design/UX round 1, Q-R1-2). It now states
    only what happened; ``refusal_from_pairing`` appends the one remedy.
    """
    return "the codes did not match — the other device did not admit this machine"


def pair_timeout_seconds(remaining_s: float) -> float:
    """Whichever fires first: what is LEFT of the invite, or the confirm budget.

    The parameter is ``remaining_s`` since agent review round 1 (NIT 1) because all
    three call sites — the joiner's read, the parked question's window and the
    listener's wait — pass ``invite.remaining_seconds`` / ``relay._remaining_of``. A
    parameter named ``ttl_s`` invited a reader to assume the MINTED duration was still
    being passed, which is exactly the confusion that produced the overstated prompt.
    """
    return min(remaining_s, PAIR_PARK_CEILING_S)


#: TCP keepalive timing for a socket a pairing ceremony parks on (see
#: :func:`set_pair_keepalive`): first probe after 60 s idle, then every 20 s.
PAIR_KEEPALIVE_IDLE_S = 60
PAIR_KEEPALIVE_INTERVAL_S = 20
PAIR_KEEPALIVE_PROBES = 3


def set_pair_keepalive(sock: Any) -> bool:
    """Turn on ``SO_KEEPALIVE`` (idle 60 s / interval 20 s where supported) for a pair socket.

    WHY. A pairing answered from the app's Mesh approvals can park its sealed socket
    for up to :data:`PAIR_PARK_CEILING_S` while a person walks between devices, and
    NAT / tunnel idle timers on the path routinely drop a silent flow well inside ten
    minutes. Kernel keepalive probes keep the mapping warm without putting anything
    on the wire protocol.

    WHY NOT AN APP-LEVEL HEARTBEAT FRAME. An inviter running an older build reads
    the first sealed frame after its offer as ``net_pair_ready`` and treats any other
    op as ``protocol_error`` — which CONSUMES the invite. Keepalive is invisible to
    the peer's code, so it is the only liveness signal that is safe across versions.

    GUARDED PER PLATFORM: ``TCP_KEEPIDLE`` is Linux (and newer macOS), macOS spells
    the idle knob ``TCP_KEEPALIVE``; a platform or socket type that refuses one of
    them still gets whatever it accepted. Never raises: a missing knob must not be
    the reason a pairing fails. Returns whether ``SO_KEEPALIVE`` itself was set.
    """
    socket_mod = socket
    try:
        sock.setsockopt(socket_mod.SOL_SOCKET, socket_mod.SO_KEEPALIVE, 1)
    except (OSError, AttributeError, TypeError, ValueError):
        return False
    idle_opt = getattr(socket_mod, "TCP_KEEPIDLE", None) or getattr(
        socket_mod, "TCP_KEEPALIVE", None
    )
    for option, value in (
        (idle_opt, PAIR_KEEPALIVE_IDLE_S),
        (getattr(socket_mod, "TCP_KEEPINTVL", None), PAIR_KEEPALIVE_INTERVAL_S),
        (getattr(socket_mod, "TCP_KEEPCNT", None), PAIR_KEEPALIVE_PROBES),
    ):
        if option is None:
            continue
        try:
            sock.setsockopt(socket_mod.IPPROTO_TCP, option, value)
        except (OSError, AttributeError, TypeError, ValueError):
            continue
    return True


#: How much of a refusing device's own sentence travels in ``net_pair_abort``. The
#: frame is sealed inside the handshake's line budget, and a long explanation must
#: never be the reason a pairing dial fails.
PAIR_ABORT_DETAIL_MAX = 400

#: What an operator can DO about each pairing refusal.
#:
#: A CODE IS NOT A REMEDY. ``device_id_conflict`` is the one that cost a round: the
#: admitting device's own sentence named the burned id and the way out, the joiner was
#: told the code and nothing else, and the way out that actually exists is not the one
#: a reader would guess — a removed device's id is burned on every device that saw the
#: rotation (``removed_ids``, §4.2) and NO invite revives it, while
#: `lop network trust <net> --active` re-admits a network marked untrusted after a
#: panic and does not restore a removed member. SAID HERE, once, next to the frame
#: that carries the refusal, because a remedy three screens away from the error is a
#: remedy nobody executes.
PAIRING_REMEDIES: dict[str, str] = {
    # THE ONE REMEDY AN OPERATOR GETS WRONG HERE, so it is stated and not implied: a
    # burned id is not revived by anything on the admitting device, and the verb that
    # LOOKS like the fix (`trust --active`) belongs to a different incident. Measured
    # (QA round 3, Q-R3-3): the joiner was told `device_id_conflict` and nothing else,
    # and a fresh invite plus `trust --active` is what an operator tries next.
    "device_id_conflict": (
        "a removed id is burned on every device that saw the rotation, so no invite "
        "revives it — and `lop network trust <network> --active` does not restore a "
        "removed member (it re-admits a NETWORK marked untrusted after a panic). The "
        "way back is a new identity on this device: `lop network identity rotate "
        "--json`, then a fresh invite"
    ),
    "invite_already_used": (
        "mint a new one on a member of that network: `lop network invite --role drive --json`"
    ),
    "invite_in_use": (
        "that token is being redeemed right now: wait for it to finish, or mint a fresh invite"
    ),
    "invite_epoch_stale": (
        "the token predates the network's current epoch: mint a fresh invite on a member"
    ),
    "unknown_invite": (
        "the device being dialled does not know this token: mint a fresh invite on it, or "
        "dial the device that issued this one"
    ),
    "invite_device_mismatch": (
        "the token is bound to another device id: mint an invite for THIS device, or use "
        "the device it was bound to"
    ),
    "untrusted": (
        "that device has marked the network untrusted: an admin runs "
        "`lop network trust <network> --active` there"
    ),
    # BOTH OF THESE ARE RETRYABLE, AND SAY SO. Neither a mismatch inside the forgiving
    # budget nor a delay consumes the invite on the admitting device (Q-XH-6) — a
    # person who read six digits off another screen slowly, or fat-fingered one, was
    # told to "start over with a fresh invite", which is a fresh token minted on the
    # device they are NOT sitting at. The sentence now says what is true: the SAME
    # token works, and it is the repeated failures that spend it. The last clause
    # covers the case where it genuinely was the final failure, so the advice cannot
    # send someone round a loop.
    "sas_mismatch": (
        "the two screens disagreed: run the join again with the SAME token and compare "
        "the code on both devices before typing — an invite is spent only after repeated "
        "failures, so if it is refused as already used, mint a fresh one"
    ),
    "declined_remote": "ask its operator",
    "timeout": (
        "nobody typed the code in time: run it again with the SAME token — a delay does "
        "not spend an invite — and answer the prompt on both devices"
    ),
}


#: The remedy when the refusing device says the invite is SPENT (``spent`` on the
#: abort frame): the SAME-token advice of ``sas_mismatch``/``timeout`` would send the
#: person round a loop the inviter has already closed.
PAIRING_SPENT_REMEDY = (
    "this invitation is used up: mint a fresh invite on the inviting device and pair "
    "with that one"
)


def refusal_from_pairing(reason: str, *, detail: str = "", spent: bool = False) -> PairingRefusal:
    """Map a pair-phase abort reason onto a refusal a CLI can print.

    The sentence is composed from three things, in this order: what the refusing
    device said (``detail``, the most specific fact available), the reason's own
    sentence, and then the REMEDY — because a refusal an operator cannot act on is a
    refusal that gets retried in the dark (Q-R3-3).
    """
    sentences = {
        "sas_mismatch": sas_mismatch_sentence(),
        "declined_local": "this device declined the pairing",
        "declined_remote": "the other device declined",
        "timeout": "the pairing timed out before both people confirmed",
        "invite_already_used": "that invite has already been used",
        "invite_in_use": "that invite is already being redeemed",
    }
    remote = " ".join(str(detail or "").split())
    base = sentences.get(reason) or (
        f"the pairing was refused: {remote}" if remote else f"the pairing was refused ({reason})"
    )
    # The refusing device's own words are KEPT BESIDE the mapped sentence when both
    # exist: they are the only place a joiner can learn which id or which device the
    # refusal was about.
    if remote and remote not in base:
        base = f"{base} — {remote}"
    remedy = PAIRING_REMEDIES.get(reason, "")
    if spent and reason in ("sas_mismatch", "timeout"):
        remedy = PAIRING_SPENT_REMEDY
    return PairingRefusal(reason, f"{base}. {remedy}" if remedy else base)


def frame_size_ok(frame: dict[str, Any]) -> bool:
    """Whether a frame is inside the handshake line budget once encoded.

    Checked on the frames this side SENDS too: a listener that answered with an
    oversized challenge would be the one to blame for the link dying, and the
    peer's reader would (correctly) refuse it.
    """
    return len(canonical_json(frame)) <= MAX_HANDSHAKE_LINE


def handshake_deadline(seconds: float = HANDSHAKE_TIMEOUT_S) -> float:
    return deadline_in(seconds)
