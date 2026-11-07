"""Byte encodings for the Vault agent arm. Pure: no I/O, no clock, no store.

Every function here reproduces a frozen contract encoding exactly, and the unit
suite pins it to the Unseal Authority's own cross-repo vector
(``internal/protocol/testdata/wire-v2/agent-resolve-vector.json`` at
vault-unseal-authority ``0794dee2``) plus VAULT.md's V5 and FP1/FP2. Change
nothing here without a contract change: the authority verifies these bytes, and
an encoding that differs by one byte is a ``403`` on every resolve with no hint
as to why.

THE TYPED ENCODING (VAULT.md §4.1). Each field carries a one-byte tag and a
big-endian length or width, so two lanes cannot produce different bytes for the
same fact and no field can bleed into its neighbour:

* ``s(text)``  = ``0x01 ‖ u32be(len) ‖ utf-8 bytes``
* ``n(int)``   = ``0x02 ‖ u64be(value)``
* ``y(bytes)`` = ``0x04 ‖ u32be(len) ‖ bytes``

ENDPOINT BYTES ARE NEVER NORMALISED. The proof binds the endpoint id the request
supplies, byte for byte; the authority compares it to the grant's exactly. A
lower-cased or trimmed endpoint is a different message and fails verification
(the vector deliberately uses ``bep_Raw-Mixed_123``).
"""

from __future__ import annotations

import base64
import hashlib
import re
import struct

#: Outer domain of the operation commitment (VAULT.md §4.1, "radient-vault-op-v2").
OP_DOMAIN = "radient-vault-op-v2"
#: Outer domain of the agent proof; deliberately NOT the op domain, so a proof can
#: never be replayed as a human operation signature or vice versa.
AGENT_PROOF_DOMAIN = "radient-vault-agent-proof-v1"
#: Domain of the delegation-key fingerprint (VAULT.md §6.20 item 2).
FINGERPRINT_DOMAIN = "radient-vault-delegation-key-v1"

#: ``ledger_version`` is a u64 on the wire. It arrives as a DECIMAL STRING because
#: a JSON number above 2^53 rounds silently (the authority's vector uses 2^53+1).
_U64_MAX = (1 << 64) - 1
_DECIMAL = re.compile(r"^(0|[1-9][0-9]{0,19})$")

#: SHA-256 of the empty parameter encoding: ``resolve`` carries no op parameters,
#: and the commitment still binds their (empty) hash so the shape is uniform.
_EMPTY_PARAMS_HASH = hashlib.sha256(b"").digest()


def s(text: str) -> bytes:
    """Tagged UTF-8 string field."""

    raw = text.encode("utf-8")
    return b"\x01" + struct.pack(">I", len(raw)) + raw


def n(value: int) -> bytes:
    """Tagged unsigned 64-bit integer field. Refuses a value outside u64."""

    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= _U64_MAX:
        raise ValueError("n() takes an unsigned 64-bit integer")
    return b"\x02" + struct.pack(">Q", value)


def y(raw: bytes) -> bytes:
    """Tagged byte-string field."""

    return b"\x04" + struct.pack(">I", len(raw)) + bytes(raw)


def parse_ledger_version(text: object) -> int:
    """Parse route 27's ``ledger_version`` (a canonical decimal STRING) as a u64.

    A JSON number is refused outright, not coerced: the server sends a string on
    purpose, and a float that came through any JSON layer may already be the
    wrong epoch, which would sign a different commitment.
    """

    if not isinstance(text, str) or not _DECIMAL.fullmatch(text):
        raise ValueError("ledger_version must be a canonical decimal string")
    value = int(text)
    if value > _U64_MAX:
        raise ValueError("ledger_version exceeds 2^64-1")
    return value


def resolve_operation_encoding(
    *,
    tenant_id: str,
    vault_id: str,
    item_id: str,
    authority_challenge_id: str,
    ledger_version: int,
) -> bytes:
    """The full typed preimage of a ``resolve`` commitment.

    ``authority_challenge_id`` is the AUTHORITY's 32-hex row id from route 27,
    never the 64-hex local ``challenge_id`` that rides in ``X-Vault-Challenge``:
    the authority id is "the ONLY id inside the commitment header" (VAULT.md
    §4.1, the ceremony-ordering amendment).
    """

    return (
        s(OP_DOMAIN)
        + s("resolve")
        + s(tenant_id)
        + s(vault_id)
        + s(item_id)
        + s(authority_challenge_id)
        + n(ledger_version)
        + y(_EMPTY_PARAMS_HASH)
    )


def resolve_commitment(
    *,
    tenant_id: str,
    vault_id: str,
    item_id: str,
    authority_challenge_id: str,
    ledger_version: int,
) -> bytes:
    """SHA-256 of :func:`resolve_operation_encoding` (32 bytes)."""

    return hashlib.sha256(
        resolve_operation_encoding(
            tenant_id=tenant_id,
            vault_id=vault_id,
            item_id=item_id,
            authority_challenge_id=authority_challenge_id,
            ledger_version=ledger_version,
        )
    ).digest()


def agent_proof_message(
    *, commitment: bytes, delegation_id: str, endpoint_id: str, timestamp: int
) -> bytes:
    """The bytes the delegation key signs (VAULT.md §4.1 step 4).

    ``endpoint_id`` is used EXACTLY as supplied; see the module docstring.
    ``timestamp`` is Unix seconds; the authority accepts ±60 s inclusive.
    """

    if len(commitment) != 32:
        raise ValueError("commitment must be 32 bytes")
    return (
        s(AGENT_PROOF_DOMAIN)
        + y(commitment)
        + s(delegation_id)
        + s(endpoint_id)
        + n(timestamp)
    )


def b64url(raw: bytes) -> str:
    """Unpadded base64url, the wire form of the public key and the proof."""

    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def key_fingerprint(public_key: bytes) -> str:
    """64 lowercase hex: SHA-256 over the domain-tagged raw public key.

    For DISPLAY and route-29 SELECTION only, never evidence: the authority
    verifies a proof against the stored raw key, not against this string.
    """

    if len(public_key) != 32:
        raise ValueError("an Ed25519 public key is 32 bytes")
    return hashlib.sha256(s(FINGERPRINT_DOMAIN) + y(public_key)).hexdigest()


def fingerprint_grid(fingerprint: str) -> list[str]:
    """The fingerprint as 4 lines of 4 four-hex groups (VAULT.md §6.20 item 2).

    The same grouping the trusted approval page and the console show, so a person
    compares like with like across three surfaces.
    """

    if not re.fullmatch(r"[0-9a-f]{64}", fingerprint):
        raise ValueError("fingerprint must be 64 lowercase hex characters")
    groups = [fingerprint[i : i + 4] for i in range(0, 64, 4)]
    return [" ".join(groups[row * 4 : row * 4 + 4]) for row in range(4)]
