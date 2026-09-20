"""The mutually-authenticated handshake, and the SAS the two humans compare.

Five frames, both roles in one file so the transcript construction is written
ONCE. That is the whole reason this is not two modules: the transcript is the
security boundary between them, and two implementations of it are two
implementations that can disagree.

    hello      dialer  → listener   (ephemeral public, nonce, device claim)
    challenge  listener → dialer    (same, plus a salt)
    auth       dialer  → listener   (sig over the transcript, MAC over the same)
    welcome    listener → dialer    (the LAST plaintext frame; phase + nets)
    …then AEAD records for everything else, including the pair ceremony

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
8. the MAC verifies against the current epoch key, else the previous one
   (which admits the peer in the ``reconcile`` phase and nothing else)

THE SAS IS DERIVED, NEVER TRANSMITTED. Its value is absent from every frame: a
peer that sent it would let an on-path attacker echo the inviter's own digits
back at it, and the human comparison would be a round trip rather than a check.
"""

from __future__ import annotations

import hmac
import socket
from dataclasses import dataclass, field
from secrets import token_bytes
from typing import Any, Literal, Protocol, cast

from local_operator.network.identity import DeviceIdentity
from local_operator.network.types import (
    MESH_PROTOCOL_VERSION,
    HandshakeRefusal,
    HandshakeResult,
    LinkPhase,
    PairingRefusal,
)
from local_operator.network.wire import (
    MAX_HANDSHAKE_LINE,
    MESH_NET_V1,
    ROLE_DIALER,
    FrameReader,
    LinkCrypto,
    Role,
    b64u,
    canonical_json,
    check_link_version,
    deadline_in,
    encode_line,
    link_keys,
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

#: How long both humans have to compare and confirm. Bounded also by the
#: invite's own ``ttl_s``; whichever fires first consumes the invite.
PAIR_CONFIRM_TIMEOUT_S = 180.0

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

    def read_welcome(self, reader: FrameReader, deadline: float) -> dict[str, Any]:
        """Read ``welcome`` — the last plaintext frame — and finish the handshake."""
        frame = reader.read_line(deadline)
        if frame.get("net") != "welcome":
            raise HandshakeRefusal(
                REASON_FRAME, "the other device did not answer with a welcome frame"
            )
        # The peer's LINK version is checked here rather than at hello, so an
        # unknown version is a refusal AFTER authentication instead of an oracle
        # any stranger can pull.
        check_link_version(frame.get("v"), mine=LINK_VERSION)
        self.phase = cast(LinkPhase, str(frame.get("phase", "member")))
        self.peer_public_key = self._peer_key_from_welcome(frame)
        # The listener's ``welcome`` is where the DIALER learns where the peer can
        # be reached. ``welcome`` is the last plaintext frame and is OUTSIDE the
        # transcript (only hello/challenge/auth are hashed), so this field cannot
        # perturb the SAS the two humans compare.
        self.peer_endpoints = clean_endpoints(frame.get("endpoints"))
        self.establish()
        return frame

    def _peer_key_from_welcome(self, frame: dict[str, Any]) -> str:
        """What the dialer can know about the listener's key from ``welcome``.

        The listener's public key is NOT in ``welcome``: the dialer already
        verified the listener's identity by the fact that it could produce a valid
        ``auth``-verified transcript and, in member mode, a MAC under the network
        epoch key. It is left empty on purpose rather than filled with an
        unverified value that a later reader might treat as evidence.
        """
        return ""

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
            # THE CALLER HAS ALREADY CLAIMED THE INVITE before this frame is built:
            # validating here as well would be a second implementation of the same
            # decision (and a second place for the two to disagree), so the only
            # check left is that the credential it derived actually arrived.
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

        # STEP 8: the MAC. Current epoch first; previous only for member mode,
        # which is what puts the link in the reconcile phase.
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
        return LinkCrypto(result.keys, role=self.role)

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
            frame["inviter"] = {"device_id": self.self_device_id, "name": self.identity.name}
        if endpoints:
            # How the dialer learns where to reach US. Sent here rather than in
            # the challenge because ``welcome`` is outside the transcript, so a
            # build that does not know this key reads it as decoration.
            frame["endpoints"] = list(endpoints)
        if nets is not None:
            frame["nets"] = nets
        return frame


# ---------------------------------------------------------------------------
# Small crypto helpers the two verification steps need
# ---------------------------------------------------------------------------


def _derived_id(public_key_b64: str) -> str:
    from local_operator.network.identity import device_id_for

    return device_id_for(unb64u(public_key_b64))


def _verify_signature(public_key_b64: str, signature_b64: str, digest: bytes) -> None:
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    try:
        Ed25519PublicKey.from_public_bytes(unb64u(public_key_b64)).verify(
            unb64u(signature_b64), b"lop-mesh-auth-v1\x00" + digest
        )
    except (InvalidSignature, ValueError) as exc:
        raise HandshakeRefusal(
            REASON_SIGNATURE,
            "the other device's signature over the handshake transcript does not verify "
            "against the public key on its member record",
        ) from exc


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


def pair_abort_frame(*, req: int, reason: str, detail: str = "") -> dict[str, Any]:
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
    """
    frame: dict[str, Any] = {"op": "net_pair_result", "req": req, "admit": admit}
    if admit:
        frame["network"] = network or {}
        frame["member"] = member or {}
        frame["members"] = list(members or [])
        frame["members_digest"] = members_digest
        frame["secret"] = material
        frame["rotations"] = rotations or {}
    else:
        frame["reason"] = reason
    return frame


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
    """What B prints when the codes disagreed. Named once because it is a promise."""
    return (
        "the codes did not match — the other device did not admit this machine. "
        "Do not retry: ask for a new invite."
    )


def pair_timeout_seconds(ttl_s: float) -> float:
    """Whichever fires first: the invite's own life, or the confirm budget."""
    return min(ttl_s, PAIR_CONFIRM_TIMEOUT_S)


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
    "sas_mismatch": (
        "the two screens disagreed: start over with a fresh invite and compare the code "
        "on both devices before typing"
    ),
    "declined_remote": "the other device declined: ask its operator",
    "timeout": "run it again and answer the prompt on both devices",
}


def refusal_from_pairing(reason: str, *, detail: str = "") -> PairingRefusal:
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
