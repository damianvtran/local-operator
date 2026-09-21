"""The wire: encodings, the key schedule, the AEAD record codec, framing.

One module owns every byte that crosses a peer link, because the security of a
hand-rolled protocol comes from pinning each byte in one place. The primitives
are Ed25519, X25519, HKDF-SHA256 and AES-256-GCM, all from ``cryptography``
(already a dependency of the secret store); **no cryptographic primitive is
implemented here** — this is composition, and every domain-separation string,
salt and nonce derivation is spelled out rather than improvised.

WHY NOT TLS. Python's stdlib ``ssl`` exposes no certificate verification
callback, so mutual TLS with per-device certificates is not expressible without
either a per-network CA (whose private key every member would hold, which cannot
distinguish devices and whose revocation would be no better than the epoch check
the mesh needs anyway) or ``CERT_NONE`` (encryption without authentication).
Framing, multiplexing, keepalive, reconnect and backpressure have to be written
either way, because the payload is a deadline-bound, coalescible event stream
rather than a byte stream.

THE IMPORT RULE. ``cryptography`` is imported inside the functions that need it.
This module is imported by ``local_operator/network/cli.py`` for argument
registration on the CLI startup path, and the repository's import-graph guard
exists precisely to stop that path growing a crypto stack.

ENCODINGS, AND WHICH ONE FOR WHAT:
* unpadded base64url for binary material on the wire — public keys, epoch keys,
  nonces, tags, signatures. Never hex where size matters.
* Crockford base32 (lowercase, ``I``/``L``/``O``/``U`` excluded) for anything a
  HUMAN transcribes: invite ids when printed, device fingerprints, and the
  160-bit transcript fingerprint. Never base64 where a person types.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import socket
import time
from dataclasses import dataclass
from typing import Any, Literal

from local_operator.network.types import HandshakeRefusal, LinkPhase, MeshRefusal

# ---------------------------------------------------------------------------
# Limits and timing defaults (module-level beside their readers, per AGENTS.md's
# "adding a configuration key" rule: one default, one home)
# ---------------------------------------------------------------------------

#: A handshake frame is JSON on one line and nothing in the ceremony comes close
#: to this; the cap exists so a peer cannot make the listener buffer without
#: limit before any authentication has happened.
#:
#: A PRE-AUTH BOUND, AND ONLY THAT: it does not bound the relay's loopback control
#: socket, whose replies are post-auth, peer-to-peer and grow with the mesh (a
#: federated catalogue is one row per session per device). That path reads under
#: ``dial.MAX_SESSION_FRAME_BYTES``, through ``dial.LineReader`` — see
#: ``relay.control_request`` — and applying THIS number to it is what made a busy
#: mesh's listing report a wedged relay (QA round 8, Q-R8-1).
MAX_HANDSHAKE_LINE = 16 * 1024

#: A single AES-GCM record's plaintext ceiling. Checked against the LENGTH
#: PREFIX before a byte is allocated, so an 8 MiB claim followed by four bytes of
#: traffic is a close rather than an allocation.
MAX_RECORD_BYTES = 8 * 1024 * 1024

#: Keepalive cadence and the idle budget that closes a link.
KEEPALIVE_S = 30.0
LINK_IDLE_S = 120.0

#: Reconnect backoff bounds: 1 s floor with ±25 % jitter, capped at
#: ``RECONNECT_MAX_S`` and reset by a successful handshake. A link is only ever
#: established by a fresh handshake — there is no resume and no session ticket.
RECONNECT_MIN_S = 1.0
RECONNECT_MAX_S = 60.0
RECONNECT_JITTER = 0.25

#: How long a RELIABLE producer waits for its op before failing it with a
#: sentence rather than dropping it.
OP_WAIT_S = 10.0

#: Per-link queue budget: frames first, then bytes; whichever bites first. The
#: DROPPABLE class is coalesced down first, so a peer that is two seconds behind
#: costs one pending projection rather than four hundred stale repaints.
QUEUE_FRAMES = 256
QUEUE_BYTES = 8 * 1024 * 1024

#: How many unanswered RELIABLE ops a peer may have in flight before it gets one
#: ``error: net_backpressure`` and then a close. Unbounded inbound queues are how
#: a peer turns a relay into swap.
MAX_INFLIGHT = 64

#: Live links this relay will hold. Sized for the spine's real topologies
#: (1-3 peers) with room for a few networks on one device; a relay that is being
#: dialled by two hundred devices is a problem the operator needs to see, not one
#: to absorb.
MAX_LINKS = 32

#: Per-direction role bytes, used BOTH as the transcript's role byte and as the
#: record AAD's direction byte. One alphabet, so a reader cannot misread one for
#: the other.
ROLE_DIALER = b"D"
ROLE_LISTENER = b"L"

Role = Literal["dialer", "listener"]

RECORD_AAD_PREFIX = b"lop-mesh-rec-v1\x00"
LOCAL_PREFIX_LEN = 4  # uint32_be length prefix on every record

#: Link feature strings. A feature is used only when BOTH sides advertise it;
#: absent means "old peer" and unknown is ignored, which is what lets a new
#: capability ship without moving either protocol version.
LINK_CAPABILITIES: tuple[str, ...] = (
    "mesh-net-v1",
    "session-mobility-v1",
    "credential-broker-v1",
    "compute-pool-v1",
)
MESH_NET_V1 = "mesh-net-v1"


# ---------------------------------------------------------------------------
# Encodings
# ---------------------------------------------------------------------------

_CROCKFORD = "0123456789abcdefghjkmnpqrstvwxyz"


def b64u(data: bytes) -> str:
    """Unpadded base64url — the only encoding for binary material on the wire."""
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def unb64u(text: str) -> bytes:
    """Decode unpadded base64url, tolerating padding a peer added by hand."""
    padded = text + "=" * (-len(text) % 4)
    return base64.urlsafe_b64decode(padded.encode("ascii"))


def crockford(data: bytes) -> str:
    """Lowercase Crockford base32, unpadded.

    Crockford's alphabet drops ``I``, ``L``, ``O`` and ``U`` so a transcribed
    string cannot be misread for a digit or a rude word — which is the whole
    reason a human-facing value uses it instead of base64.
    """
    bits = 0
    value = 0
    out: list[str] = []
    for byte in data:
        value = (value << 8) | byte
        bits += 8
        while bits >= 5:
            bits -= 5
            out.append(_CROCKFORD[(value >> bits) & 0x1F])
    if bits:
        out.append(_CROCKFORD[(value << (5 - bits)) & 0x1F])
    return "".join(out)


def canonical_json(payload: Any) -> bytes:
    """The one serialisation the transcript and every signature commit to.

    Sorted keys, no whitespace, UTF-8: canonical, so two implementations that
    disagree about nothing still produce the same bytes.
    """
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
        "utf-8"
    )


def sha256(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def hex64(data: bytes) -> str:
    return data.hex()


# ---------------------------------------------------------------------------
# The key schedule
# ---------------------------------------------------------------------------


def epoch_key(material: str, network_id: str, epoch: int) -> bytes:
    """``HKDF-SHA256(ikm=secret, salt=sha256(network_id), info=…||epoch, 32)``.

    Derived rather than stored so a rotation only ever has to distribute ONE
    32-byte value, and so an epoch number that disagrees with the secret shows up
    as a MAC failure rather than as a silent identity confusion.
    """
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF

    hkdf = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=sha256(network_id.encode("utf-8")),
        info=b"lop-mesh-epoch-v1\x00" + str(epoch).encode("ascii"),
    )
    return hkdf.derive(unb64u(material))


def invite_key(material: str, network_id: str, invite_id: str) -> bytes:
    """The key an invite token is MAC'd with: one token, one key.

    The ``invite_id`` is in the info string so two invites minted from the same
    epoch secret during the same second cannot share a key — and so a token
    cannot be re-labelled with another invite's id.
    """
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF

    hkdf = HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=network_id.encode("utf-8"),
        info=b"lop-mesh-invite-v1\x00" + invite_id.encode("ascii"),
    )
    return hkdf.derive(unb64u(material))


def invite_mac(key: bytes, payload: bytes) -> bytes:
    """``HMAC-SHA256(invite_key, b"lop-invite-v1\\x00" || payload_bytes)``."""
    return hmac.new(key, b"lop-invite-v1\x00" + payload, hashlib.sha256).digest()


def auth_signature_message(transcript_hash: bytes) -> bytes:
    """What the device key signs: a domain-separated transcript hash."""
    return b"lop-mesh-auth-v1\x00" + transcript_hash


def auth_mac_message(transcript_hash: bytes) -> bytes:
    """What the epoch (or invite) key MACs — the same hash, another domain.

    The two domain strings are different so a signature over a transcript can
    never be replayed as a MAC over the same transcript, which is the mistake a
    shared prefix would allow.
    """
    return b"lop-mesh-mac-v1\x00" + transcript_hash


def auth_mac(key: bytes, transcript_hash: bytes) -> bytes:
    return hmac.new(key, auth_mac_message(transcript_hash), hashlib.sha256).digest()


@dataclass(frozen=True)
class LinkKeys:
    """Per-direction keys and nonce prefixes for one link, and nothing else.

    ``link_id`` is minted at the handshake from 16 random bytes and **never
    crosses the wire**: it is an AAD field, so two links between the same two
    devices in the same second cannot have a record transplanted from one to the
    other.
    """

    link_id: bytes
    k_d2l: bytes
    k_l2d: bytes
    iv_d: bytes
    iv_l: bytes

    def send_params(self, role: Role) -> tuple[bytes, bytes, bytes]:
        """``(key, nonce_prefix, direction_byte)`` for a record this side sends."""
        if role == "dialer":
            return self.k_d2l, self.iv_d, ROLE_DIALER
        return self.k_l2d, self.iv_l, ROLE_LISTENER

    def receive_params(self, role: Role) -> tuple[bytes, bytes, bytes]:
        """The sender's parameters as the RECEIVER sees them."""
        other: Role = "listener" if role == "dialer" else "dialer"
        return self.send_params(other)


def link_id_for(shared: bytes, transcript_hash: bytes) -> bytes:
    """The link's identifier: DERIVED from the handshake, never assigned.

    A per-side random id is the obvious implementation and it is wrong: ``link_id``
    is part of the AAD of every record, so two sides that each picked their own
    would fail every tag — and the failure would present as a MAC error, sending
    anyone debugging it after a crypto bug that does not exist. Deriving it from
    the same (Z, th) both sides already agree on makes the two identical by
    construction.
    """
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF

    return HKDF(
        algorithm=hashes.SHA256(),
        length=16,
        salt=link_salt(transcript_hash, shared),
        info=b"lop-mesh-linkid-v1",
    ).derive(shared)


def link_salt(transcript_hash: bytes, shared: bytes) -> bytes:
    """The one salt both the key schedule and the id derivation use."""
    return sha256(b"lop-mesh-salt-v1\x00" + transcript_hash + shared)


def link_keys(shared: bytes, transcript_hash: bytes, link_id: bytes | None = None) -> LinkKeys:
    """The link's key schedule, exactly as §6.3 pins it.

    ``salt = sha256(b"lop-mesh-salt-v1\\x00" || th || Z)`` — the ephemeral shared
    secret appears in the salt AND as the IKM, and the transcript hash is bound in
    beside it, so a key is a function of both the key exchange and everything the
    two sides said.

    The 72-byte output is split in order: two 32-byte directional keys and two
    4-byte nonce prefixes, giving each direction its own key AND its own nonce
    space. That split is what makes a reflection attack impossible: a peer that
    echoes another's records back at it hands them a record encrypted under the
    key for the OPPOSITE direction, which fails the tag.

    ``link_id`` defaults to the DERIVED one (:func:`link_id_for`) and the argument
    exists only so a test can prove the AAD binds it.
    """
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF

    salt = link_salt(transcript_hash, shared)
    okm = HKDF(
        algorithm=hashes.SHA256(),
        length=72,
        salt=salt,
        info=b"lop-mesh-link-v1",
    ).derive(shared)
    return LinkKeys(
        link_id=link_id if link_id is not None else link_id_for(shared, transcript_hash),
        k_d2l=okm[0:32],
        k_l2d=okm[32:64],
        iv_d=okm[64:68],
        iv_l=okm[68:72],
    )


def sas_code(shared: bytes, transcript_hash: bytes) -> str:
    """The six digits both humans compare, derived and NEVER transmitted.

    ``HKDF(ikm=Z, salt=sha256(b"lop-mesh-sas-v1\\x00" || th), info=…, 8)`` then
    ``% 1_000_000``. Note the salt differs from the link keys' — it binds the
    transcript hash but not the shared secret, so the digits are a function of the
    transcript in a way that is cheap to state and cheap to test.

    Its strength is ~20 bits per human interaction, and that is stated rather
    than implied: what makes it useful is everything around it (a successful
    grind still needs TWO humans, a failed comparison burns the invite, and a
    fresh invite is a fresh human action). The 160-bit fingerprint below is the
    strong check.
    """
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF

    okm = HKDF(
        algorithm=hashes.SHA256(),
        length=8,
        salt=sha256(b"lop-mesh-sas-v1\x00" + transcript_hash),
        info=b"lop-mesh-sas-v1",
    ).derive(shared)
    return f"{int.from_bytes(okm, 'big') % 1_000_000:06d}"


def sas_display(code: str) -> str:
    """``481926`` → ``481 926``, the form a human reads off a screen."""
    if len(code) != 6:
        return code
    return f"{code[:3]} {code[3:]}"


def transcript_fingerprint(transcript_hash: bytes) -> str:
    """The leading 20 bytes of the transcript hash as Crockford base32, grouped.

    ``K7QM-3XPD-…-9T2B`` — 160 bits in eight groups of four. This is the value
    ``--verify`` makes the thing a human compares, and the reason the six digits
    are allowed to be "only" ~20 bits: it costs one paste.
    """
    text = crockford(transcript_hash[:20]).upper()
    return "-".join(text[index : index + 4] for index in range(0, len(text), 4))


def normalize_sas(text: str) -> str:
    """Reduce a typed SAS to its six digits, or ``""`` when it is not one.

    Spaces and dashes are stripped (people type ``481 926``), and anything else
    is a refusal rather than a guess: a transcription that is not six digits is
    not a near miss, it is a different value.
    """
    # ONLY the separators people actually type are removed, and the result must be
    # exactly six digits. Extracting digits from anywhere would accept "the code is
    # 481926 — confirm?" as a match, which turns a comparison the human is supposed
    # to be making into a parser's guess.
    candidate = text.replace(" ", "").replace("-", "").replace("\u00a0", "").strip()
    return candidate if len(candidate) == 6 and candidate.isdigit() else ""


# ---------------------------------------------------------------------------
# The AEAD record codec
# ---------------------------------------------------------------------------


class LinkCrypto:
    """Seals and opens the records of ONE link, one codec per side.

    Nonces are DERIVED, never transmitted: ``nonce = iv_x || uint64_be(seq)`` with
    ``seq`` starting at 0 per direction. A counter is strictly stronger than a
    random nonce for a single link and needs no entropy at frame rate.

    Every failure is fatal to the link: an ``InvalidTag``, an AAD mismatch, an
    impossible sequence number (a gap, or past 2⁴⁰) raises :class:`LinkCryptoError`
    and the caller closes. There is no resynchronisation and no "recover and
    continue" — a protocol with a partial-trust phase has a partial-trust attack.
    """

    #: Past this many records on one link the sequence number stops being safe to
    #: keep incrementing; the link is closed and re-handshaken, which is free.
    MAX_SEQ = 2**40

    def __init__(self, keys: LinkKeys, *, role: Role) -> None:
        self._keys = keys
        # ANNOTATED, not inferred: pyright widens an attribute assigned from a
        # ``Literal``-typed parameter to ``str``, so the ``role`` property below
        # would promise a ``Role`` it cannot prove. The annotation is the promise.
        self._role: Role = role
        self._key, self._iv, self._direction = keys.send_params(role)
        recv_key, recv_iv, recv_direction = keys.receive_params(role)
        self._recv_key = recv_key
        self._recv_iv = recv_iv
        self._recv_direction = recv_direction
        self._send_seq = 0
        self._recv_seq = 0

    @property
    def role(self) -> Role:
        return self._role

    @property
    def sent(self) -> int:
        return self._send_seq

    @property
    def received(self) -> int:
        return self._recv_seq

    def seal(self, frame: dict[str, Any]) -> bytes:
        """One record: ``uint32_be(len) || AES-256-GCM(plaintext)``."""
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM

        plaintext = json.dumps(
            frame, sort_keys=False, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        if len(plaintext) > MAX_RECORD_BYTES:
            raise LinkCryptoError(
                f"a frame of {len(plaintext)} bytes exceeds the {MAX_RECORD_BYTES}-byte record "
                "limit; the link is closed rather than fragmented"
            )
        sequence = self._send_seq
        payload = AESGCM(self._key).encrypt(
            self._nonce(self._iv, sequence),
            plaintext,
            self._aad(self._direction, sequence),
        )
        self._send_seq += 1
        return len(payload).to_bytes(LOCAL_PREFIX_LEN, "big") + payload

    def open(self, payload: bytes) -> dict[str, Any]:
        """Decrypt one record's payload (length prefix already consumed)."""
        from cryptography.exceptions import InvalidTag
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM

        sequence = self._recv_seq
        if sequence >= self.MAX_SEQ:
            raise LinkCryptoError(
                "this link has carried more records than its sequence numbers can name; "
                "reconnect with a fresh handshake"
            )
        try:
            plaintext = AESGCM(self._recv_key).decrypt(
                self._nonce(self._recv_iv, sequence),
                payload,
                self._aad(self._recv_direction, sequence),
            )
        except InvalidTag as exc:
            # The message is deliberately the same for a tampered record, a
            # replayed one and one from another link: which of the three happened
            # is not the peer's business, and the local audit record keeps the
            # distinction for the operator.
            raise LinkCryptoError(
                "a record failed authentication: the link is closed and nothing in it is repaired"
            ) from exc
        self._recv_seq += 1
        try:
            frame = json.loads(plaintext.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            raise LinkCryptoError(
                "a record decrypted to something that is not a JSON frame; the link is closed"
            ) from exc
        if not isinstance(frame, dict):
            raise LinkCryptoError("a record decrypted to a JSON value that is not an object")
        return frame

    @staticmethod
    def _nonce(iv: bytes, sequence: int) -> bytes:
        return iv + sequence.to_bytes(8, "big")

    def _aad(self, direction: bytes, sequence: int) -> bytes:
        return RECORD_AAD_PREFIX + self._keys.link_id + direction + sequence.to_bytes(8, "big")


class LinkCryptoError(Exception):
    """A record failed to authenticate or to parse. Always fatal to the link."""


# ---------------------------------------------------------------------------
# Framing
# ---------------------------------------------------------------------------


class FrameReader:
    """Reads handshake LINES and post-handshake RECORDS from one socket.

    One buffer for both, because a peer may pipeline: the challenge can arrive in
    the same TCP segment as the tail of our own write, and a reader that assumed
    "one read is one frame" would lose bytes. Byte-oriented sockets have no
    message boundaries, and every framing bug in a hand-rolled protocol starts
    with forgetting that.
    """

    def __init__(self, sock: socket.socket, *, buffered: bytes = b"") -> None:
        """``buffered`` is a PREVIOUS reader's leftover, handed over rather than lost.

        ONE SOCKET HAS TWO PHASES — the JSON-line handshake and the sealed record
        stream after it — and the reader that finishes the first may have pulled
        bytes of the second off the socket already (``_fill`` reads in 64 KiB
        chunks). Dropping that object therefore drops those bytes, and the frame
        they belong to is then decrypted out of sequence: the receiver sees a
        ``LinkCryptoError`` on a link whose handshake was perfectly good, seconds
        after it was established. That is a real hazard for ANY peer that speaks
        immediately after the handshake — a pull, a rotation, a session op — and it
        is why this constructor takes the buffer instead of a fresh reader silently
        starting empty.
        """
        self._sock = sock
        self._buffer = bytearray(buffered)

    def read_line(self, deadline: float | None) -> dict[str, Any]:
        """One JSON-lines frame, bounded by :data:`MAX_HANDSHAKE_LINE`."""
        while True:
            newline = self._buffer.find(b"\n")
            if newline >= 0:
                raw = bytes(self._buffer[:newline])
                del self._buffer[: newline + 1]
                return _parse_line(raw, limit=MAX_HANDSHAKE_LINE)
            if len(self._buffer) > MAX_HANDSHAKE_LINE:
                raise LinkCryptoError(
                    f"a handshake frame exceeded the {MAX_HANDSHAKE_LINE}-byte limit"
                )
            self._fill(deadline)

    def read_record_payload(self, deadline: float | None) -> bytes:
        """One record's ciphertext, with the length prefix checked FIRST.

        The check happens before a byte is allocated, so a peer cannot ask this
        process for 8 MiB of buffer by writing four bytes.
        """
        header = self._read_exactly(LOCAL_PREFIX_LEN, deadline)
        length = int.from_bytes(header, "big")
        if length > MAX_RECORD_BYTES:
            raise LinkCryptoError(
                f"a peer announced a {length}-byte record, over the {MAX_RECORD_BYTES}-byte "
                "limit; the link is closed"
            )
        if length == 0:
            raise LinkCryptoError("a peer announced an empty record, which is never valid")
        return self._read_exactly(length, deadline)

    def _read_exactly(self, count: int, deadline: float | None) -> bytes:
        while len(self._buffer) < count:
            self._fill(deadline)
        chunk = bytes(self._buffer[:count])
        del self._buffer[:count]
        return chunk

    def _fill(self, deadline: float | None) -> None:
        timeout = None if deadline is None else max(0.0, deadline - time.monotonic())
        if timeout is not None and timeout == 0.0:
            raise TimeoutError("the peer did not answer within the deadline")
        self._sock.settimeout(timeout)
        try:
            chunk = self._sock.recv(65536)
        except TimeoutError as exc:
            raise TimeoutError("the peer did not answer within the deadline") from exc
        if not chunk:
            raise ConnectionError("the peer closed the connection")
        self._buffer.extend(chunk)

    def pending(self) -> bytes:
        """Buffered bytes (a test's window onto pipelining, and a reader's on reuse)."""
        return bytes(self._buffer)


def _parse_line(raw: bytes, *, limit: int) -> dict[str, Any]:
    if len(raw) > limit:
        raise LinkCryptoError(f"a frame exceeded the {limit}-byte limit")
    try:
        frame = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise LinkCryptoError("a frame was not valid UTF-8 JSON") from exc
    if not isinstance(frame, dict):
        raise LinkCryptoError("a frame was not a JSON object")
    return frame


def encode_line(frame: dict[str, Any]) -> bytes:
    """A handshake frame as it goes on the wire: compact JSON, one line."""
    return (
        json.dumps(frame, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    ).encode("utf-8")


def deadline_in(seconds: float) -> float:
    """An absolute monotonic deadline, which is what every reader here wants.

    Relative timeouts applied per read make a slow peer able to extend an
    exchange indefinitely; one deadline for the whole ceremony cannot be
    extended.
    """
    return time.monotonic() + seconds


# ---------------------------------------------------------------------------
# Version and capability negotiation
# ---------------------------------------------------------------------------


def check_link_version(peer_version: Any, *, mine: int) -> None:
    """Refuse an unknown LINK version, naming both numbers.

    Checked AFTER authentication (the version lives in a frame either side may
    send unauthenticated, so refusing early would be an oracle) and never
    silently downgraded: the same discipline the session record already uses to
    report a build skew rather than fail silently.
    """
    if peer_version == mine:
        return
    raise HandshakeRefusal(
        "protocol_mismatch",
        f"the two builds speak different mesh link protocols (this device: {mine}, "
        f"the other: {peer_version}). Update the older one.",
    )


def negotiate_capabilities(mine: list[str], theirs: list[str]) -> list[str]:
    """The features both sides advertise, in this build's order.

    Absent means "old peer" and unknown is ignored, so a capability can ship
    without either protocol version moving — which is the whole reason link
    features are strings.
    """
    offered = set(theirs)
    return [capability for capability in mine if capability in offered]


# ---------------------------------------------------------------------------
# Keepalive
# ---------------------------------------------------------------------------


def keepalive_frame(req: int) -> dict[str, Any]:
    """``ping`` is a REUSED ``ControlOp``, not a new op: the vocabulary already
    has a liveness probe and inventing a second one would be a second thing for
    both sides to agree about."""
    return {"op": "ping", "req": req, "locality": "remote"}


def is_keepalive(frame: dict[str, Any]) -> bool:
    return frame.get("op") == "ping"


def is_bye(frame: dict[str, Any]) -> bool:
    return frame.get("op") == "net_bye"


def refusal_frame(req: Any, sentence: str) -> dict[str, Any]:
    """What a peer is told when something is refused.

    Deliberately vague at the boundary: never which of membership, epoch or
    capability failed. The real cause goes to the local audit log, where the
    operator can see it and a remote attacker cannot. ``MeshRefusal`` carries both
    halves for exactly this reason.
    """
    return {"op": "error", "req": req, "message": sentence}


def error_from(refusal: MeshRefusal, req: Any) -> dict[str, Any]:
    return refusal_frame(req, refusal.sentence)


def phase_is_reconcile(phase: LinkPhase) -> bool:
    return phase == "reconcile"
