"""Verify a remote server's signed statement that a workload belongs to ONE network.

WHAT THIS MODULE IS FOR. A remote run leaves this device, and the answer that
comes back is only useful if it can be tied to the tenant the request named. The
server signs, with Ed25519, a canonical JSON statement binding the workload to a
network and to THIS device's key; this module is the client-side check that the
statement is the server's, is about the network and device the caller asked
about, and is current. Everything it needs arrives as arguments — the statement,
the public keys, the caller's expectations and the clock — so the verification is
a pure function with no I/O and no ambient state.

WHY THE CANONICAL BYTES ARE RE-SERIALISED AND COMPARED. Verifying the signature
proves the bytes are the signer's; it says nothing about what those bytes MEAN to
this parser. Two encodings of the same object (a different key order, an escaped
form, a duplicated field resolved last-wins) would verify but parse differently
in another reader, so the statement is refused unless re-serialising the parsed
values reproduces the signed bytes exactly. That is why the field order in
:data:`STATEMENT_FIELDS` is part of the contract rather than a formatting
choice.

WHY THE CHECKS HAVE A FIXED ORDER. Recognisability comes before meaning: a
caller with no expectation configured, or a statement signed by a key this
device does not hold, is refused before any parsing work, and each later rule is
only meaningful once the earlier ones hold. The codes are stable strings a caller
branches on, in the same shape as :mod:`local_operator.remote_compute.resolve`.

WHAT IS NOT CHECKED HERE, deliberately: whether the KEY ID is the server's
CURRENT key (a rotated key still signs statements that verified when issued —
freshness is the window's job, and re-binding to "current" would fail a
statement the far side signed minutes ago), and whether the workload still
exists. Both are the caller's, and neither is answerable from the statement.

IMPORTS. ``local_operator.network.types`` is imported at module scope for the
refusal family every mesh surface shares; it is stdlib-only and already on the
CLI startup path. ``cryptography`` is imported inside the verifying function
only: it is a real dependency, but this module is imported by the run-create
path, not by capability negotiation.
"""

from __future__ import annotations

import base64
import binascii
import dataclasses
import json
import re
from collections.abc import Mapping
from typing import Any

from local_operator.network.types import MeshRefusal

#: The statement's fields, IN THE ORDER THEY ARE SIGNED. Re-serialising the
#: parsed values with :func:`json.dumps` reproduces the signed bytes only
#: because this order is the contract's, so it is a data structure here rather
#: than seven literals repeated across the parse and the canonicalisation.
STATEMENT_FIELDS: tuple[str, ...] = (
    "v",
    "workload_id",
    "account_id",
    "network_id",
    "device_key",
    "issued_at",
    "expires_at",
)

#: The statement version this verifier understands. A newer version is refused
#: rather than parsed leniently: the whole point of the version field is that a
#: reader which does not know a format cannot decide it is safe.
STATEMENT_VERSION = 1

#: Refusal codes, spelled once because a caller (and a test) branches on the
#: string. Same vocabulary shape as ``remote_compute.resolve``: a machine code
#: plus a sentence for the operator.
CODE_MISSING_EXPECTATION = "remote_compute_attestation_missing_expectation"
CODE_UNKNOWN_KEY_ID = "remote_compute_attestation_unknown_key_id"
CODE_INVALID_KEY = "remote_compute_attestation_invalid_key"
CODE_MALFORMED_PAYLOAD = "remote_compute_attestation_malformed_payload"
CODE_BAD_SIGNATURE = "remote_compute_attestation_bad_signature"
CODE_UNSUPPORTED_VERSION = "remote_compute_attestation_unsupported_version"
CODE_NON_CANONICAL = "remote_compute_attestation_non_canonical"
CODE_NETWORK_MISMATCH = "remote_compute_attestation_network_mismatch"
CODE_DEVICE_MISMATCH = "remote_compute_attestation_device_mismatch"
CODE_ISSUED_IN_FUTURE = "remote_compute_attestation_issued_in_future"
CODE_EXPIRED = "remote_compute_attestation_expired"
CODE_WINDOW_INVERTED = "remote_compute_attestation_window_inverted"
CODE_WINDOW_TOO_LONG = "remote_compute_attestation_window_too_long"
CODE_MALFORMED_KEY_SET = "remote_compute_attestation_malformed_key_set"
CODE_DUPLICATE_KEY_ID = "remote_compute_attestation_duplicate_key_id"

#: The unpadded URL-safe base64 alphabet, matched with ``fullmatch``. The stdlib
#: decoder DISCARDS characters outside the alphabet instead of refusing them, so
#: the shape is checked here first: two spellings that decode to the same bytes
#: are one thing signed and another thing read.
_B64URL_PATTERN = re.compile(r"[A-Za-z0-9_-]+")

#: Ed25519's fixed sizes. A key of another length is not a key this verifier can
#: use, and cryptography's own ``ValueError`` would say so only as "bad length".
_PUBLIC_KEY_BYTES = 32
_SIGNATURE_BYTES = 64


class AttestationRefusal(MeshRefusal):
    """A statement refused, with a stable ``code`` and a sentence for the operator.

    A ``MeshRefusal`` subclass so the ``code``/``sentence`` shape every consumer
    already parses is unchanged, and a DISTINCT class because an attestation
    refusal is not a membership refusal: a handler that admits work to a network
    must not swallow "the server's statement does not verify" under the same
    ``except`` as "this device is not a member".

    The sentence names the key id, the version and the network id, which are
    public identifiers. It NEVER echoes a signature, a public key or a payload:
    those are bytes an operator cannot act on, and a log line is forever.
    """


@dataclasses.dataclass(frozen=True)
class VerifiedAttestation:
    """A statement whose signature, canonical form, tenant and window all held.

    The fields are exactly what the statement binds, and ``device_key`` is the
    one this device presented — the caller's own key, echoed back only after it
    matched. Frozen because this object is the evidence a run is attributed to a
    network, and evidence that can be edited afterwards is not evidence.
    """

    workload_id: str
    account_id: str
    network_id: str
    device_key: bytes
    issued_at: int
    expires_at: int
    key_id: str


def verify_attestation(
    *,
    payload_b64: str,
    signature: str,
    key_id: str,
    keys: Mapping[str, bytes],
    expect_network_id: str,
    expect_device_key: bytes,
    now: int,
    max_skew: int = 60,
    max_window: int = 3600,
) -> VerifiedAttestation:
    """Verify one statement against this device's expectations and clock.

    ``keys`` is ``{key_id: 32-byte Ed25519 public key}``, as
    :func:`parse_attestation_keys` returns it. The key is selected BY ID and no
    other: trying every key in the set would let a statement verified under one
    identity answer for another, and it would hide a key-id mistake behind a
    success. ``now``, ``max_skew`` and ``max_window`` are arguments rather than a
    clock read so the window rules are testable at their exact boundaries.
    """
    _require_expectation(expect_network_id, expect_device_key)
    public_key = _select_key(keys, key_id)

    payload_bytes = _decode_payload(payload_b64)
    _verify_signature(public_key, signature, payload_bytes, key_id)

    statement = _parse_statement(payload_bytes)
    _require_supported_version(statement)
    _require_canonical(statement, payload_bytes)
    _require_network(statement, expect_network_id)
    device_key = _require_device_key(statement, expect_device_key)
    _require_window(statement, now=now, max_skew=max_skew, max_window=max_window)

    return VerifiedAttestation(
        workload_id=statement["workload_id"],
        account_id=statement["account_id"],
        network_id=statement["network_id"],
        device_key=device_key,
        issued_at=statement["issued_at"],
        expires_at=statement["expires_at"],
        key_id=key_id,
    )


def parse_attestation_keys(body: Any) -> dict[str, bytes]:
    """Turn a discovery response body into ``{key_id: public key}``.

    The shape is ``{"keys": [{"key_id", "alg": "Ed25519", "public_key",
    "status"}]}``, and every part of it is validated rather than assumed: a key
    set is what decides which signatures are the server's, so a body this
    function accepted loosely would become the trust anchor. ``status`` is
    admitted for both ``current`` and ``previous`` because a rotation leaves
    statements signed by the outgoing key inside their window; it is rejected
    only when it is a value the contract does not define, which is how a future
    "revoked" is noticed instead of trusted.

    Extra fields are ignored: the contract may grow, and an unknown field this
    verifier does not read cannot change what it checks.
    """
    if not isinstance(body, Mapping):
        raise AttestationRefusal(
            CODE_MALFORMED_KEY_SET,
            "the attestation key set must be a JSON object with a `keys` list",
        )
    rows = body.get("keys")
    if not isinstance(rows, list):
        raise AttestationRefusal(
            CODE_MALFORMED_KEY_SET,
            "the attestation key set must carry a `keys` list",
        )

    keys: dict[str, bytes] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise AttestationRefusal(
                CODE_MALFORMED_KEY_SET,
                f"key entry {index} is not a JSON object",
            )
        key_id = row.get("key_id")
        if not isinstance(key_id, str) or not key_id:
            raise AttestationRefusal(
                CODE_MALFORMED_KEY_SET,
                f"key entry {index} has no `key_id`",
            )
        if row.get("alg") != "Ed25519":
            raise AttestationRefusal(
                CODE_MALFORMED_KEY_SET,
                f"key {key_id!r} is not an Ed25519 key; this verifier supports no other",
            )
        status = row.get("status")
        if status not in ("current", "previous"):
            raise AttestationRefusal(
                CODE_MALFORMED_KEY_SET,
                f"key {key_id!r} carries the status {status!r}, which is not "
                "'current' or 'previous'; refusing a key set this verifier cannot read",
            )
        if key_id in keys:
            raise AttestationRefusal(
                CODE_DUPLICATE_KEY_ID,
                f"key id {key_id!r} appears twice in the key set; one id must name "
                "exactly one key, or a signature could be checked against the wrong one",
            )
        public_key = row.get("public_key")
        try:
            decoded = _decode_b64u(public_key)
        except ValueError as exc:
            raise AttestationRefusal(
                CODE_MALFORMED_KEY_SET,
                f"key {key_id!r} carries a `public_key` that is not unpadded " "URL-safe base64",
            ) from exc
        if len(decoded) != _PUBLIC_KEY_BYTES:
            raise AttestationRefusal(
                CODE_MALFORMED_KEY_SET,
                f"key {key_id!r} decodes to {len(decoded)} bytes; an Ed25519 public "
                f"key is {_PUBLIC_KEY_BYTES}",
            )
        keys[key_id] = decoded
    return keys


def _require_expectation(expect_network_id: str, expect_device_key: bytes) -> None:
    """Refuse an empty expectation BEFORE reading the statement.

    A verifier with nothing to compare against would accept any well-signed
    statement, which is worse than refusing: the caller would believe a check
    happened. An expectation that cannot be compared is a caller bug, so it is
    named as one rather than reported as a bad statement.
    """
    if not isinstance(expect_network_id, str) or not expect_network_id:
        raise AttestationRefusal(
            CODE_MISSING_EXPECTATION,
            "no expected network id was supplied; a statement is only meaningful "
            "against the network this device asked about",
        )
    if not isinstance(expect_device_key, bytes) or not expect_device_key:
        raise AttestationRefusal(
            CODE_MISSING_EXPECTATION,
            "no expected device key was supplied; a statement is only meaningful "
            "against the key this device presented",
        )


def _select_key(keys: Mapping[str, bytes], key_id: str) -> bytes:
    """The public key named by ``key_id``, or a refusal — never a search.

    NO TRY-EVERY-KEY FALLBACK. A loop over the set would answer "which of these
    verified" and call it success, so a statement signed by a key the caller did
    not name would pass, and so would one carrying a key id the server never
    published.
    """
    if not isinstance(key_id, str) or key_id not in keys:
        raise AttestationRefusal(
            CODE_UNKNOWN_KEY_ID,
            f"the attestation names key id {key_id!r}, which is not in the " "published key set",
        )
    public_key = keys[key_id]
    if not isinstance(public_key, bytes) or len(public_key) != _PUBLIC_KEY_BYTES:
        raise AttestationRefusal(
            CODE_INVALID_KEY,
            f"key {key_id!r} is not a {_PUBLIC_KEY_BYTES}-byte Ed25519 public key; "
            "the statement cannot be checked against it",
        )
    return public_key


def _decode_payload(payload_b64: str) -> bytes:
    """The exact bytes the server signed, or a refusal.

    The bytes are decoded ONCE and both verified and parsed from that same
    result: decoding twice is how a verifier and a parser come to disagree about
    what was signed.
    """
    try:
        return _decode_b64u(payload_b64)
    except ValueError as exc:
        raise AttestationRefusal(
            CODE_MALFORMED_PAYLOAD,
            "the attestation payload is not unpadded URL-safe base64",
        ) from exc


def _verify_signature(
    public_key: bytes, signature_b64: str, payload_bytes: bytes, key_id: str
) -> None:
    """Refuse unless ``signature_b64`` is this key's signature over those bytes."""
    # IMPORTED HERE: this module is on the run-create path, and only a
    # verification needs the crypto backend. See the module docstring.
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    try:
        signature_bytes = _decode_b64u(signature_b64)
    except ValueError as exc:
        raise AttestationRefusal(
            CODE_BAD_SIGNATURE,
            f"the attestation's signature for key {key_id!r} is not unpadded " "URL-safe base64",
        ) from exc
    if len(signature_bytes) != _SIGNATURE_BYTES:
        raise AttestationRefusal(
            CODE_BAD_SIGNATURE,
            f"the attestation's signature for key {key_id!r} is "
            f"{len(signature_bytes)} bytes; an Ed25519 signature is {_SIGNATURE_BYTES}",
        )
    try:
        Ed25519PublicKey.from_public_bytes(public_key).verify(signature_bytes, payload_bytes)
    except InvalidSignature as exc:
        raise AttestationRefusal(
            CODE_BAD_SIGNATURE,
            f"the attestation does not verify against key {key_id!r}: the signature "
            "is not this key's signature over these exact bytes",
        ) from exc


def _parse_statement(payload_bytes: bytes) -> dict[str, Any]:
    """Parse the signed bytes into the seven typed fields, or refuse.

    TYPES ARE CHECKED, not assumed: ``bool`` is an ``int`` in Python, so
    ``"v": true`` would otherwise pass every arithmetic rule that follows, and a
    string ``issued_at`` would raise a ``TypeError`` from a comparison long after
    the refusal should have been made.
    """
    try:
        statement = json.loads(payload_bytes)
    except (ValueError, UnicodeDecodeError) as exc:
        raise AttestationRefusal(
            CODE_MALFORMED_PAYLOAD,
            "the attestation payload is not valid UTF-8 JSON",
        ) from exc
    if not isinstance(statement, dict):
        raise AttestationRefusal(
            CODE_MALFORMED_PAYLOAD,
            "the attestation payload is not a JSON object",
        )
    missing = [field for field in STATEMENT_FIELDS if field not in statement]
    if missing:
        raise AttestationRefusal(
            CODE_MALFORMED_PAYLOAD,
            "the attestation payload omits " + ", ".join(missing),
        )
    for field in ("workload_id", "account_id", "network_id", "device_key"):
        if not isinstance(statement[field], str):
            raise AttestationRefusal(
                CODE_MALFORMED_PAYLOAD,
                f"the attestation field {field!r} is not a string",
            )
    for field in ("v", "issued_at", "expires_at"):
        value = statement[field]
        if not isinstance(value, int) or isinstance(value, bool):
            raise AttestationRefusal(
                CODE_MALFORMED_PAYLOAD,
                f"the attestation field {field!r} is not an integer",
            )
    return statement


def _require_supported_version(statement: Mapping[str, Any]) -> None:
    """Refuse a statement this verifier does not know how to read."""
    if statement["v"] != STATEMENT_VERSION:
        raise AttestationRefusal(
            CODE_UNSUPPORTED_VERSION,
            f"the attestation is version {statement['v']}, and this client "
            f"understands version {STATEMENT_VERSION}",
        )


def _require_canonical(statement: Mapping[str, Any], payload_bytes: bytes) -> None:
    """Refuse bytes that are not this object's one canonical spelling.

    The signature only proves the bytes are the signer's. A statement with a
    different key order, an escaped string, or an extra field would verify here
    and read differently in another parser — so the parsed values are
    re-serialised in :data:`STATEMENT_FIELDS` order and the result must be the
    signed bytes exactly. The extra-field refusal falls out of this: the rebuilt
    object carries the seven fields and nothing else.
    """
    rebuilt = json.dumps(
        {field: statement[field] for field in STATEMENT_FIELDS},
        separators=(",", ":"),
    ).encode("utf-8")
    if rebuilt != payload_bytes:
        raise AttestationRefusal(
            CODE_NON_CANONICAL,
            "the attestation is validly signed but is not its own canonical form "
            "(field order, escaping or an unexpected field differs from the signed "
            "spelling); a statement two readers parse differently is refused",
        )


def _require_network(statement: Mapping[str, Any], expect_network_id: str) -> None:
    """Refuse a statement about a different network than the one asked about."""
    if statement["network_id"] != expect_network_id:
        raise AttestationRefusal(
            CODE_NETWORK_MISMATCH,
            "this device asked about network "
            f"{expect_network_id!r} and the attestation names "
            f"{statement['network_id']!r}",
        )


def _require_device_key(statement: Mapping[str, Any], expect_device_key: bytes) -> bytes:
    """Refuse a statement about a different device key; return the matched key.

    The statement carries the joining device's key, so this is what makes the
    signature an answer TO THIS DEVICE rather than about the workload in general.
    """
    try:
        device_key = _decode_b64u(statement["device_key"])
    except ValueError as exc:
        raise AttestationRefusal(
            CODE_MALFORMED_PAYLOAD,
            "the attested device key is not unpadded URL-safe base64",
        ) from exc
    if device_key != expect_device_key:
        raise AttestationRefusal(
            CODE_DEVICE_MISMATCH,
            "the attestation binds a different device key than the one this device " "presented",
        )
    return device_key


def _require_window(
    statement: Mapping[str, Any], *, now: int, max_skew: int, max_window: int
) -> None:
    """Refuse a statement that is not current, or that claims too long a life.

    ``expires_at`` is EXCLUSIVE (``now == expires_at`` is expired) so the window
    is half-open and two readings an instant apart cannot both call it live.
    ``issued_at`` is admitted up to ``max_skew`` seconds ahead, because the
    signer's clock and this one are not the same clock; the window length is
    bounded separately so a far-future ``issued_at`` cannot be used to buy a
    longer life than ``max_window``.
    """
    issued_at = statement["issued_at"]
    expires_at = statement["expires_at"]
    if issued_at > now + max_skew:
        raise AttestationRefusal(
            CODE_ISSUED_IN_FUTURE,
            f"the attestation was issued at {issued_at}, more than {max_skew}s "
            f"ahead of this device's clock ({now})",
        )
    if expires_at <= now:
        raise AttestationRefusal(
            CODE_EXPIRED,
            f"the attestation expired at {expires_at}, at or before this device's "
            f"clock ({now})",
        )
    if expires_at <= issued_at:
        raise AttestationRefusal(
            CODE_WINDOW_INVERTED,
            f"the attestation expires at {expires_at}, not after it was issued at " f"{issued_at}",
        )
    if expires_at - issued_at > max_window:
        raise AttestationRefusal(
            CODE_WINDOW_TOO_LONG,
            f"the attestation claims {expires_at - issued_at}s, longer than the "
            f"{max_window}s this client accepts",
        )


def _decode_b64u(text: object) -> bytes:
    """Decode unpadded URL-safe base64, refusing anything that is not exactly that.

    STRICTER THAN ``network.wire.unb64u``, and the difference is deliberate: that
    helper tolerates padding a peer added by hand, and the stdlib decoder it
    calls DISCARDS characters outside the alphabet rather than refusing them. For
    handshake material a lenient decode is a convenience; for signed bytes it is
    a hole, because two different strings would then be "the same" signature. A
    ``ValueError`` here is translated by the caller into its own refusal code.
    """
    if not isinstance(text, str) or not _B64URL_PATTERN.fullmatch(text):
        raise ValueError("not unpadded URL-safe base64")
    try:
        return base64.b64decode(
            (text + "=" * (-len(text) % 4)).encode("ascii"), altchars=b"-_", validate=True
        )
    except (binascii.Error, ValueError) as exc:
        raise ValueError("not unpadded URL-safe base64") from exc


__all__ = [
    "CODE_BAD_SIGNATURE",
    "CODE_DEVICE_MISMATCH",
    "CODE_DUPLICATE_KEY_ID",
    "CODE_EXPIRED",
    "CODE_INVALID_KEY",
    "CODE_ISSUED_IN_FUTURE",
    "CODE_MALFORMED_KEY_SET",
    "CODE_MALFORMED_PAYLOAD",
    "CODE_MISSING_EXPECTATION",
    "CODE_NETWORK_MISMATCH",
    "CODE_NON_CANONICAL",
    "CODE_UNKNOWN_KEY_ID",
    "CODE_UNSUPPORTED_VERSION",
    "CODE_WINDOW_INVERTED",
    "CODE_WINDOW_TOO_LONG",
    "STATEMENT_FIELDS",
    "STATEMENT_VERSION",
    "AttestationRefusal",
    "VerifiedAttestation",
    "parse_attestation_keys",
    "verify_attestation",
]
