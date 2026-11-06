"""ES256 verification for operator authority (issue #1310, revision 2).

WHY THIS MODULE EXISTS APART FROM THE SIGNER. Verification runs inside the
runtime, in the process that also serves the control socket, and it is reached
from ``harness/approval``'s seam through ONE function that takes a VERDICT —
``harness/approval`` is stdlib-only by design (see its module docstring), so the
crypto must not leak into its import graph. Everything here therefore imports
``cryptography`` lazily, inside the functions, and a host that never verifies an
operator signature never pays for the import.

WHAT IS VERIFIED. A signature over a domain-separated, length-prefixed message
(:func:`signed_message`). The framing is not decoration: without the length
prefixes an attacker who controls one field could shift bytes between adjacent
fields and produce a second, valid-looking message for a different action.
Without the domain tag they could replay a signature from any other protocol
this key signs. Both are the reason a signed message is built in exactly one
place and read by exactly two callers (the signer and this module).

TWO KINDS OF SIGNER, ONE ROOT OF TRUST:

* the OPERATOR key itself, whose public half is the root-owned anchor
  (:mod:`local_operator.operator.trust`);
* a DEVICE key (the phone), which proves itself with an operator-signed
  certificate (:func:`verify_device_cert`) that is checked against that same
  anchored key. A device key is therefore only as good as the operator's
  willingness to have certified it, and an operator can revoke one by removing
  it from the anchor's device list.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import struct
from dataclasses import dataclass
from typing import Any

#: The domain tag. Versioned so a future message shape cannot be confused with
#: this one by a verifier that happens to accept both during a migration.
DOMAIN = b"lop-operator-v1\x00"

#: The domain tag for a device certificate. Separate from :data:`DOMAIN` so a
#: signature harvested over a device certificate can never be presented as a
#: signature over an action.
DEVICE_CERT_DOMAIN = b"lop-operator-device-v1\x00"

#: The domain tag for an APPROVAL DECISION (remote-onboarding §2.4, F4). A third
#: tag beside the two above, for the same reason they are separate from each
#: other: a signature harvested over an operator action or a device certificate
#: can never be replayed as an approval decision, and the reverse. The payload's
#: ONE builder lives with the approval store (``network/approvals.signed_payload``
#: — it needs this module's length-prefix framing without importing the operator
#: package at read paths); the TAG lives here so every domain this key signs is
#: declared in one module.
APPROVAL_DOMAIN = b"lop-approval-v1\x00"

#: The domain tag for a Team Bridge OWNER RELEASE (B4b DESIGN §3.2, operator
#: decision a-3f46; security review S-8 / decision D7). A fourth tag, NOT a new
#: member of :data:`ACTIONS`: ``signed_message`` binds a session/challenge pair
#: that a release does not have, and a release must never be presentable as a
#: loosen/approve/decide signature or the reverse. The payload's ONE builder is
#: ``team_bridge.hold_release.release_payload`` (length-prefixed with this
#: module's framing); the TAG lives here so every domain this key signs is
#: declared in one module.
RELEASE_DOMAIN = b"lop-bridge-release-v1\x00"

#: A signature is DER-encoded ECDSA over a P-256 key: 8 (sequence) + up to 2*33
#: of integers. Bounded rather than exact, because DER length varies with the
#: leading byte of each integer; the bound exists so a megabyte of junk cannot
#: reach the parser.
MAX_SIGNATURE_BYTES = 80

#: A device certificate is a small JSON envelope. Bounded for the same reason.
MAX_CERT_BYTES = 4096

#: Which authority-increasing actions a signature may name. The runtime derives
#: this from the frame it is deciding (never from a field the caller chose), so
#: a signature minted for one action cannot be presented as the other.
ACTIONS = ("loosen", "approve", "decide")


def _lp(value: str) -> bytes:
    """Length-prefix one field: 4-byte big-endian length, then UTF-8 bytes."""
    raw = value.encode("utf-8", "surrogatepass")
    return struct.pack(">I", len(raw)) + raw


def signed_message(
    *,
    action: str,
    session_id: str,
    request_id: str,
    challenge: str,
) -> bytes:
    """The ONE message a signature covers.

    Every field is bound into it, which is what makes a signature
    per-action, per-session and per-challenge rather than a bearer token:

    * ``action`` — a signature for answering a card cannot loosen the gate;
    * ``session_id`` — a signature harvested on one session is worthless on
      another, so a relay forwarding frames between two sessions cannot profit;
    * ``request_id`` — a card approval names the card, so an approval cannot be
      moved to a different question;
    * ``challenge`` — the runtime mints it per request and single-uses it, which
      is what removes replay as a category (a captured signature has no second
      use, because its challenge is spent).
    """
    return DOMAIN + _lp(action) + _lp(session_id) + _lp(request_id) + _lp(challenge)


def key_id_for(spki: bytes) -> str:
    """The stable identifier for a public key: truncated SHA-256 of its SPKI.

    Derived rather than stored so the anchor and every frame agree without a
    registry, and truncated to 32 hex characters because the value rides frames
    and a full digest buys nothing an attacker can use — the SIGNATURE is the
    authentication, and this field only routes which key to try.
    """
    return hashlib.sha256(spki).hexdigest()[:32]


def spki_fp(spki: bytes) -> str:
    """The SHORT display fingerprint of a public key: 64 bits, hex, grouped.

    ``9A3C-EF12-3456-7890`` — the value an approval card puts beside the full
    ``key_id`` so a human can compare two screens the way they compare a device
    fingerprint. Cosmetic and derived, never authority (the card's check is the
    digest + signature): the same split the device fingerprint draws in
    ``network/identity.py``.
    """
    text = hashlib.sha256(spki).hexdigest().upper()[:16]
    return "-".join(text[index : index + 4] for index in range(0, len(text), 4))


def decode_point(spki: bytes) -> Any | None:
    """An ``EllipticCurvePublicKey`` from a 65-byte uncompressed P-256 point.

    ``None`` for anything else, including the compressed and hybrid forms: the
    anchor stores the uncompressed point and nothing else, so accepting a second
    encoding here would be a second thing to keep in step for no gain.
    """
    if len(spki) != 65 or spki[0] != 0x04:
        return None
    from cryptography.hazmat.primitives.asymmetric import ec

    try:
        return ec.EllipticCurvePublicKey.from_encoded_point(ec.SECP256R1(), spki)
    except ValueError:
        return None


def verify_signature(*, spki: bytes, message: bytes, signature: bytes) -> bool:
    """Whether ``signature`` is a valid ES256 signature over ``message``.

    ``cryptography`` is imported here rather than at module import: the runtime
    must be able to serve a session on a host where the library is absent, and a
    host with no anchor never reaches this. Every failure mode — a malformed
    point, a malformed DER signature, a wrong key — is ``False`` rather than an
    exception, because a raise here would travel as a runtime error frame and
    describe a different problem than "this signature is not the operator's".
    """
    if not signature or len(signature) > MAX_SIGNATURE_BYTES:
        return False
    public = decode_point(spki)
    if public is None:
        return False
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import ec

    try:
        public.verify(signature, message, ec.ECDSA(hashes.SHA256()))
    except InvalidSignature:
        return False
    except Exception:  # noqa: BLE001 — a malformed DER sig raises ValueError/TypeError
        return False
    return True


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _unb64(text: str) -> bytes | None:
    padded = text + "=" * (-len(text) % 4)
    try:
        return base64.urlsafe_b64decode(padded.encode("ascii"))
    except (binascii.Error, ValueError):
        return None


@dataclass(frozen=True)
class DeviceCert:
    """The operator-signed statement that a device key belongs to this operator.

    ``not_after`` is a Unix timestamp rather than a date string so expiry is one
    integer comparison and needs no calendar handling on either side.
    """

    device_id: str
    spki: bytes
    label: str
    issued_at: int
    not_after: int

    def payload(self) -> bytes:
        """The canonical bytes a certificate signature covers.

        Sorted keys and no whitespace, so the signer and the verifier cannot
        disagree about the encoding of the same statement — a pretty-printed
        ``json.dumps`` on one side is the classic way two implementations end up
        rejecting each other's valid certificates.
        """
        return DEVICE_CERT_DOMAIN + json.dumps(
            {
                "device_id": self.device_id,
                "issued_at": self.issued_at,
                "label": self.label,
                "not_after": self.not_after,
                "spki": _b64(self.spki),
                "v": 1,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

    def encode(self, *, signature: bytes) -> str:
        """The wire form: ``<b64 payload>.<b64 signature>``."""
        return f"{_b64(self.payload())}.{_b64(signature)}"


def read_device_cert(certificate: str) -> DeviceCert | None:
    """The certificate's STATEMENT, parsed and bounded, but NOT yet verified.

    Split from :func:`verify_device_cert` because two callers need different
    halves of the answer: the verifier needs the public point, and the runtime
    needs the ``device_id`` so a REVOKED device can be refused — and revocation
    is a property of the anchor's list, not of the certificate. Parsing without
    verifying here keeps that division honest: nothing that reads a device id
    from this is thereby trusting the certificate, and the caller that acts on it
    must have verified first.
    """
    if not certificate or len(certificate) > MAX_CERT_BYTES:
        return None
    payload_b64, _, _signature_b64 = certificate.partition(".")
    if not payload_b64:
        return None
    payload = _unb64(payload_b64)
    if payload is None or not payload.startswith(DEVICE_CERT_DOMAIN):
        return None
    try:
        body = json.loads(payload[len(DEVICE_CERT_DOMAIN) :].decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return None
    if not isinstance(body, dict) or body.get("v") != 1:
        return None
    not_after = body.get("not_after")
    device_id = body.get("device_id")
    label = body.get("label")
    issued_at = body.get("issued_at")
    raw_spki = body.get("spki")
    if not isinstance(not_after, int) or isinstance(not_after, bool):
        return None
    if not isinstance(issued_at, int) or isinstance(issued_at, bool):
        return None
    if not isinstance(device_id, str) or not isinstance(label, str):
        return None
    if not isinstance(raw_spki, str):
        return None
    spki = _unb64(raw_spki)
    if spki is None or decode_point(spki) is None:
        return None
    return DeviceCert(
        device_id=device_id,
        spki=spki,
        label=label,
        issued_at=issued_at,
        not_after=not_after,
    )


def verify_device_cert(
    certificate: str,
    *,
    operator_spki: bytes,
    now: int,
    parsed: DeviceCert | None = None,
) -> bytes | None:
    """The device's public point, or ``None`` when the certificate does not hold.

    This is the whole of the device trust story: the phone never exports its
    private key, the machine stores a certificate the OPERATOR signed, and a
    device signature is only reached through a certificate that verifies under
    the pinned operator key and has not expired. A forged certificate — the
    negative control in the acceptance matrix — has no operator signature and
    dies here.

    ``parsed`` lets a caller that already read the statement (:func:`read_device_
    cert`, for the device id) avoid parsing it twice; the SIGNATURE is checked
    either way, so passing it can never skip a check.
    """
    if not certificate or len(certificate) > MAX_CERT_BYTES:
        return None
    payload_b64, _, signature_b64 = certificate.partition(".")
    if not payload_b64 or not signature_b64:
        return None
    payload = _unb64(payload_b64)
    signature = _unb64(signature_b64)
    if payload is None or signature is None:
        return None
    if not payload.startswith(DEVICE_CERT_DOMAIN):
        return None
    if not verify_signature(spki=operator_spki, message=payload, signature=signature):
        return None
    if parsed is None:
        parsed = read_device_cert(certificate)
    if parsed is None:
        return None
    if now > parsed.not_after or now < parsed.issued_at:
        # BOTH directions: a certificate from the future is as much a lie as an
        # expired one, and a skewed clock is not a reason to accept either.
        return None
    return parsed.spki


def signature_verdict(
    *,
    action: str,
    session_id: str,
    request_id: str,
    challenge: str,
    signature_hex: object,
    operator_spki: bytes | None,
    operator_key_id: str,
    operator_cert: object,
    device_spki: bytes | None,
    now: int,
) -> bool | None:
    """Whether an offered operator/device signature admits an increasing frame.

    ``None`` means NO SIGNATURE WAS OFFERED, which the seam reads as "this
    source said nothing" and combines with the other sources; ``False`` means
    one WAS offered and did not hold, which is a refusal that no other source
    may paper over (an attacker who presents a bad signature must not be
    admitted by a capability they do not have — and, symmetrically, a
    capability holder who also offers a broken signature is not made worse off,
    because the sources are combined in :func:`local_operator.harness.approval.
    admit_increasing`).

    One function rather than the verification steps at the call site, because
    the runtime's job is to map a frame to a verdict and this module's job is to
    decide it: a second caller that reassembled these steps could disagree about
    which key a certificate is checked against.
    """
    if not isinstance(signature_hex, str) or not signature_hex:
        return None
    if operator_spki is None:
        # No anchor: the signature cannot be placed against a pinned key, so a
        # well-formed one is refused rather than skipped. A runtime with no
        # anchor is the state a freshly-installed host is in, and reporting it
        # as "not offered" would let a present-but-unverifiable signature read
        # as absent.
        return False
    try:
        signature = bytes.fromhex(signature_hex)
    except ValueError:
        return False
    if not isinstance(operator_key_id, str) or not operator_key_id:
        return False

    signer_spki = operator_spki
    if operator_cert is not None:
        # THE DEVICE'S POINT MUST COME FROM THE CALLER, and ``None`` is a REFUSAL
        # rather than "not looked up". This used to re-verify the certificate
        # here whenever the caller passed no point, and that fallback silently
        # defeated the whole revocation list: the caller
        # (``RuntimeServer._device_cert_point``) is where the certificate is
        # checked against the anchor AND where the anchor's revocation list is
        # consulted, and it answers a revoked device with ``None``. Re-verifying
        # in that case returned the device's point anyway and admitted the
        # signature — caught by the phone's revoked-device cell in
        # ``tests/unit/session/runtime/test_approval_authority_seam.py``, which is
        # the negative control that exists for exactly this class of bypass.
        #
        # One resolution point, and ONLY the point travels. A caller that does not
        # resolve the certificate therefore refuses every device signature rather
        # than accepting one it has not checked — the fail-closed direction.
        if device_spki is None:
            return False
        signer_spki = device_spki
    elif operator_key_id != key_id_for(operator_spki):
        # An anchored-key signature that names a different key id than the
        # anchor's is refused before the verify, so a frame cannot probe which
        # keys the anchor knows.
        return False

    message = signed_message(
        action=action, session_id=session_id, request_id=request_id, challenge=challenge
    )
    return verify_signature(spki=signer_spki, message=message, signature=signature)


# ---------------------------------------------------------------------------
# Refresh delegation (mesh-refresh-delegation.md §3.7g)
# ---------------------------------------------------------------------------

#: The domain tag for a REFRESH DELEGATION: the statement by which the operator
#: lets ONE named node present one login's refresh family while the owner is
#: away. A fifth tag, and deliberately NOT a member of :data:`ACTIONS` or a new
#: approval ``KIND``: a delegation is multi-use and time-bounded (like a device
#: certificate) where an approval is a single-use transition, and folding one
#: into the other would either weaken the fence or widen the approval payload
#: (design §3.8). The same stub signs revocation notices — a deny statement is
#: the same document with ``relay_deny`` set and ``not_after`` at the revoke
#: moment — so ONE verifier answers both and a deny can only ever reduce.
REFRESH_GRANT_DOMAIN = b"lop-refresh-grant-v1\x00"

#: A delegation statement is a small JSON envelope. Bounded for the same reason
#: :data:`MAX_CERT_BYTES` is: nothing that big can be a real statement, and an
#: unbounded parse is free work for whoever sent it.
MAX_REFRESH_GRANT_BYTES = 8192

#: The signed field order, fixed here so the signer and every verifier derive
#: the same bytes. Ordered rather than sorted by JSON: length-prefixed fields
#: leave no separator to shift a byte across (the argument
#: ``approvals.signed_payload`` states for its own framing), and a fixed tuple is
#: the only place the order can be got wrong.
REFRESH_GRANT_FIELDS: tuple[str, ...] = (
    "network_id",
    "owner_device",
    "owner_spki_fp",
    "node_device",
    "node_spki_fp",
    "key",
    "provider",
    "credential_kind",
    "login_incarnation",
    "family_epoch_at_issue",
    "grant_id",
    "issued_at",
    "not_after",
    "max_refreshes",
    "exclusive_holder",
    "relay_deny",
)


def _grant_field(value: object) -> bytes:
    """One length-prefixed delegation field. Booleans and ints get ONE spelling each.

    Reuses this module's own :func:`_lp` framing so the delegation cannot drift
    from the other operator-signed messages, and canonicalises the two non-string
    types here because ``"True"``/``"1"`` must never both be accepted for the same
    field.
    """
    if isinstance(value, bool):
        return _lp("1" if value else "0")
    return _lp(str(value))


def refresh_grant_payload(fields: Any) -> bytes:
    """The canonical bytes a delegation signature covers — the ONE builder.

    ``fields`` is any mapping with :data:`REFRESH_GRANT_FIELDS`; a missing field
    is a ``KeyError`` rather than a zero, because a statement that silently
    omitted ``not_after`` must not verify as one that expired at 0.
    """
    return REFRESH_GRANT_DOMAIN + b"".join(
        _grant_field(fields[name]) for name in REFRESH_GRANT_FIELDS
    )


@dataclass(frozen=True)
class RefreshGrant:
    """A parsed delegation (or revocation) statement, NOT yet verified.

    ``not_after`` is a Unix timestamp so expiry is one integer comparison, the
    same choice :class:`DeviceCert` makes. ``login_incarnation`` binds the
    statement to one login *lineage*, so a re-login on the owner invalidates the
    delegation by a VALUE the owner already has rather than by a clock.
    ``node_spki_fp`` binds the node's identity KEY, not merely its device id, so
    a rotated node key must be re-granted (the device-certificate lesson).
    """

    network_id: str
    owner_device: str
    owner_spki_fp: str
    node_device: str
    node_spki_fp: str
    key: str
    provider: str
    credential_kind: str
    login_incarnation: str
    family_epoch_at_issue: int
    grant_id: str
    issued_at: int
    not_after: int
    max_refreshes: int
    exclusive_holder: bool
    relay_deny: bool

    def fields(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in REFRESH_GRANT_FIELDS}

    def payload(self) -> bytes:
        """The SIGNED message: domain tag plus length-prefixed fields."""
        return refresh_grant_payload(self.fields())

    def body(self) -> dict[str, Any]:
        return {"v": 1, **{name: getattr(self, name) for name in REFRESH_GRANT_FIELDS}}

    def envelope(self) -> bytes:
        """The STORED/WIRE envelope: domain tag plus canonical JSON of the body.

        Not the signed bytes: the signature covers :meth:`payload`, and a
        verifier re-derives that from the parsed body (below), so an envelope
        someone edited cannot be made to verify by editing it — the re-derived
        message differs and the signature check fails. Keeping the two forms
        apart is what makes that true.
        """
        return REFRESH_GRANT_DOMAIN + json.dumps(
            self.body(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")

    def encode(self, *, signature: bytes) -> str:
        """The wire form: ``<b64 envelope>.<b64 signature>``."""
        return f"{_b64(self.envelope())}.{_b64(signature)}"


def read_refresh_grant(statement: str) -> RefreshGrant | None:
    """The statement's CONTENT, parsed and bounded, but NOT verified.

    Split from :func:`verify_refresh_grant` for the same reason
    :func:`read_device_cert` is split from :func:`verify_device_cert`: a relay
    needs the ``grant_id``/``node_device`` to route a revocation, and nothing
    that reads those fields is thereby trusting the statement.
    """
    if not statement or len(statement) > MAX_REFRESH_GRANT_BYTES:
        return None
    payload_b64, _, _signature_b64 = statement.partition(".")
    if not payload_b64:
        return None
    payload = _unb64(payload_b64)
    if payload is None or not payload.startswith(REFRESH_GRANT_DOMAIN):
        return None
    rest = payload[len(REFRESH_GRANT_DOMAIN) :]
    # The signed message is length-prefixed, not JSON, so the body is NOT
    # recoverable from it: the payload travels as the JSON envelope a signer
    # emits. Reading it back means parsing that envelope from the decoded text
    # the signer wrote, which is the only place it exists.
    return _refresh_grant_from_json(rest)


def _refresh_grant_from_json(raw: bytes) -> RefreshGrant | None:
    try:
        body = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return None
    if not isinstance(body, dict) or body.get("v") != 1:
        return None
    strings = (
        "network_id",
        "owner_device",
        "owner_spki_fp",
        "node_device",
        "node_spki_fp",
        "key",
        "provider",
        "credential_kind",
        "login_incarnation",
        "grant_id",
    )
    ints = ("family_epoch_at_issue", "issued_at", "not_after", "max_refreshes")
    bools = ("exclusive_holder", "relay_deny")
    for name in strings:
        if not isinstance(body.get(name), str):
            return None
    for name in ints:
        value = body.get(name)
        if not isinstance(value, int) or isinstance(value, bool):
            return None
    for name in bools:
        if not isinstance(body.get(name), bool):
            return None
    return RefreshGrant(**{name: body[name] for name in REFRESH_GRANT_FIELDS})


def verify_refresh_grant(
    statement: str,
    *,
    operator_spki: bytes | None,
    now: float,
    network_id: str = "",
    node_device: str = "",
    node_spki_fp: str = "",
    key: str = "",
    parsed: RefreshGrant | None = None,
) -> RefreshGrant | None:
    """The statement, or ``None`` when it does not hold for THIS device.

    Three things are checked and each one is a fence, so none is a caller's job:

    * **the signature**, against the operator anchor — never a senders' claim
      and never a device key. This is why a revocation may be relayed by any
      member: the authority is the signature, not who carried the frame.
    * **the name bindings** supplied by the caller (``network_id``,
      ``node_device``, ``node_spki_fp``, ``key``). A statement about another
      device's login answers ``None`` here rather than being refused later by a
      caller that might forget.
    * **the window**, and the DIRECTION differs per kind: a grant must be
      ``issued_at <= now < not_after``, while a deny statement is *by design* at
      or past its own ``not_after`` (it is minted with ``not_after = now``), so
      it is accepted from its ``issued_at`` onward with no upper bound. A clock
      behind the issuer refuses both, because a statement from the future is as
      much a lie as an expired one (the device-certificate rule).
    """
    if operator_spki is None:
        return None
    if not statement or len(statement) > MAX_REFRESH_GRANT_BYTES:
        return None
    payload_b64, _, signature_b64 = statement.partition(".")
    if not payload_b64 or not signature_b64:
        return None
    payload = _unb64(payload_b64)
    signature = _unb64(signature_b64)
    if payload is None or signature is None or not payload.startswith(REFRESH_GRANT_DOMAIN):
        return None
    if parsed is None:
        parsed = _refresh_grant_from_json(payload[len(REFRESH_GRANT_DOMAIN) :])
    if parsed is None:
        return None
    # The signature covers the RE-DERIVED message, never the bytes as they
    # arrived: a tampered envelope body changes the derived fields and dies
    # here, which is the whole reason the builder is one function.
    if not verify_signature(spki=operator_spki, message=parsed.payload(), signature=signature):
        return None
    if network_id and parsed.network_id != network_id:
        return None
    if node_device and parsed.node_device != node_device:
        return None
    if node_spki_fp and parsed.node_spki_fp != node_spki_fp:
        return None
    if key and parsed.key != key:
        return None
    if now < parsed.issued_at:
        return None
    if not parsed.relay_deny and now >= parsed.not_after:
        return None
    return parsed
