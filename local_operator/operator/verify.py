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
ACTIONS = ("loosen", "approve")


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
