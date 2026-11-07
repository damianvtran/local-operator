"""H1: the agent-proof encodings reproduce the contract's vectors byte for byte.

PROVENANCE. ``fixtures/agent-resolve-vector.json`` is a byte-identical copy of
vault-unseal-authority ``internal/protocol/testdata/wire-v2/agent-resolve-vector.json``
at ``0794dee2`` (unchanged at ``0f289374``), sha256 pinned below so a silent edit
of the copy fails here rather than passing against a drifted vector. V5 and
FP1/FP2 are VAULT.md's own vectors (agent-server ``docs/VAULT.md``). The seed in
the vector is PUBLIC SYNTHETIC (00..1f), not a credential.
"""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path
from typing import Any

import pytest
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from local_operator.vault_agent import keys, wire

FIXTURE = Path(__file__).parent / "fixtures" / "agent-resolve-vector.json"
FIXTURE_SHA256 = "9a77f2782fb110e99815232169db6733342c14038b04a6b0d445fd59cac722ff"


def _vector() -> dict[str, Any]:
    raw = FIXTURE.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == FIXTURE_SHA256
    return json.loads(raw)


def _decode_fields(encoding: bytes) -> list[Any]:
    """Recover the vector's inputs from its own operation encoding, never guess them."""

    fields: list[Any] = []
    i = 0
    while i < len(encoding):
        tag = encoding[i]
        if tag in (1, 4):
            (length,) = struct.unpack(">I", encoding[i + 1 : i + 5])
            fields.append(encoding[i + 5 : i + 5 + length])
            i += 5 + length
        elif tag == 2:
            fields.append(struct.unpack(">Q", encoding[i + 1 : i + 9])[0])
            i += 9
        else:  # pragma: no cover - a malformed fixture
            raise AssertionError(f"unexpected tag {tag}")
    return fields


@pytest.fixture(scope="module")
def vector_case() -> dict[str, Any]:
    v = _vector()
    _, _op, tenant, vault, item, chal, ledger, _params = _decode_fields(
        bytes.fromhex(v["operation_encoding_hex"])
    )
    message = bytes.fromhex(v["proof_message_hex"])
    (ts,) = struct.unpack(">Q", message[-8:])
    return {
        "v": v,
        "tenant": tenant.decode(),
        "vault": vault.decode(),
        "item": item.decode(),
        "chal": chal.decode(),
        "ledger": ledger,
        "ts": ts,
    }


def _commit(c: dict[str, Any]) -> bytes:
    return wire.resolve_commitment(
        tenant_id=c["tenant"],
        vault_id=c["vault"],
        item_id=c["item"],
        authority_challenge_id=c["chal"],
        ledger_version=c["ledger"],
    )


def test_operation_encoding_and_commitment_match_the_authority_vector(vector_case) -> None:
    c = vector_case
    encoding = wire.resolve_operation_encoding(
        tenant_id=c["tenant"],
        vault_id=c["vault"],
        item_id=c["item"],
        authority_challenge_id=c["chal"],
        ledger_version=c["ledger"],
    )
    assert encoding.hex() == c["v"]["operation_encoding_hex"]
    assert _commit(c).hex() == c["v"]["commitment_hex"]
    # The vector's ledger is 2^53+1: the value float parsing silently breaks.
    assert c["ledger"] == (1 << 53) + 1
    assert wire.parse_ledger_version(str(c["ledger"])) == c["ledger"]


def test_proof_message_public_key_and_signature_match_the_authority_vector(vector_case) -> None:
    c = vector_case
    v = c["v"]
    message = wire.agent_proof_message(
        commitment=_commit(c),
        delegation_id="vdl_a",
        endpoint_id=v["endpoint_id_raw"],
        timestamp=c["ts"],
    )
    assert message.hex() == v["proof_message_hex"]
    seed = bytes.fromhex(v["public_fixture_seed_hex"])
    assert keys.public_key_from_seed(seed).hex() == v["public_key_hex"]
    proof = keys.sign_proof(seed, message)
    assert proof == v["signature_wire"]
    # The deterministic RFC 8032 signature, hex-for-hex, not merely "a valid one".
    padded = proof + "=" * (-len(proof) % 4)
    import base64

    assert base64.urlsafe_b64decode(padded).hex() == v["signature_hex"]


def test_v5_commitment_and_message() -> None:
    commitment = wire.resolve_commitment(
        tenant_id="tnt_000000000000000000000001",
        vault_id="vlt_000000000000000000000001",
        item_id="vit_000000000000000000000001",
        authority_challenge_id="vch_000000000000000000000020",
        ledger_version=3,
    )
    assert commitment.hex() == "e073016b441cc1a8fbcdf9b1e2668c2d57e2799a5ebd5f41100593799a6dae30"
    message = wire.agent_proof_message(
        commitment=commitment,
        delegation_id="vdl_000000000000000000000001",
        endpoint_id="bep_000000000000000000000001",
        timestamp=1792992030,
    )
    assert len(message) == 145
    assert message.hex() == (
        "010000001c72616469656e742d7661756c742d6167656e742d70726f6f662d76310400000020"
        "e073016b441cc1a8fbcdf9b1e2668c2d57e2799a5ebd5f41100593799a6dae30010000001c76646c"
        "5f303030303030303030303030303030303030303030303031010000001c6265705f303030303030"
        "30303030303030303030303030303030303102000000006adee31e"
    )


def test_fp1_fp2_and_the_4x4_grid() -> None:
    fp1_pk = bytes.fromhex("d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a")
    fp1 = wire.key_fingerprint(fp1_pk)
    assert fp1 == "18e1d0fe43fd1e7fed445079b2bb27d6a9bc8f575247baafe0be84d3fe3edf9f"
    assert wire.key_fingerprint(bytes(range(0xA0, 0xC0))) == (
        "0b4e97dc3a66d04095e3a1dd20332e77544fa60508d2a1a116b7638f736069fd"
    )
    assert len(wire.s(wire.FINGERPRINT_DOMAIN) + wire.y(fp1_pk)) == 73
    assert wire.fingerprint_grid(fp1) == [
        "18e1 d0fe 43fd 1e7f",
        "ed44 5079 b2bb 27d6",
        "a9bc 8f57 5247 baaf",
        "e0be 84d3 fe3e df9f",
    ]


def _verifies(public_key: bytes, proof: str, message: bytes) -> bool:
    import base64

    signature = base64.urlsafe_b64decode(proof + "=" * (-len(proof) % 4))
    try:
        Ed25519PublicKey.from_public_bytes(public_key).verify(signature, message)
    except InvalidSignature:
        return False
    return True


def test_negative_controls_are_refused(vector_case) -> None:
    """Each control must FAIL, or the PASS assertions above prove nothing."""

    c = vector_case
    v = c["v"]
    seed = bytes.fromhex(v["public_fixture_seed_hex"])
    pk = keys.public_key_from_seed(seed)
    commitment = _commit(c)
    endpoint = v["endpoint_id_raw"]

    def message(endpoint_id: str = endpoint, ts: int = c["ts"]) -> bytes:
        return wire.agent_proof_message(
            commitment=commitment, delegation_id="vdl_a", endpoint_id=endpoint_id, timestamp=ts
        )

    proof = keys.sign_proof(seed, message())
    assert _verifies(pk, proof, message())  # the positive, so the negatives mean something
    flipped = message()[:-1] + bytes([message()[-1] ^ 1])
    assert not _verifies(pk, proof, flipped)
    assert not _verifies(pk, proof, message(endpoint_id=endpoint.lower()))
    assert not _verifies(pk, proof, message(ts=c["ts"] + 1))
    other = keys.public_key_from_seed(bytes([0x42]) * 32)
    assert not _verifies(other, proof, message())


@pytest.mark.parametrize("bad", [3, "03", "-1", "1.0", "", "18446744073709551616", None])
def test_ledger_version_parsing_refuses_non_canonical_input(bad) -> None:
    with pytest.raises(ValueError):
        wire.parse_ledger_version(bad)


def test_ledger_version_accepts_u64_max() -> None:
    assert wire.parse_ledger_version("18446744073709551615") == (1 << 64) - 1
