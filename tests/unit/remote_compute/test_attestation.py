"""Client-side attestation verification: the golden vector and every refusal path.

ISOLATION: none of these tests read the operator's config, mesh store or clock.
Every input is the committed golden vector (``fixtures/attestation_golden.json``)
or a value DERIVED from it here, and ``now`` is always an argument. That is the
point of a verifier with no ambient state: a tampered statement can be produced
without a server, so each refusal is driven directly rather than simulated over
HTTP.

THE FIXTURE IS A VECTOR, NOT A COPY OF THE SERVER'S SOURCE. Its two seeds are
what make every derived case possible: a statement re-signed with the fixture's
signer seed is "validly signed but wrong" in exactly one dimension, so each test
below differs from the golden case in ONE way and a failure names the rule that
broke.
"""

from __future__ import annotations

import base64
import dataclasses
import json
from pathlib import Path
from typing import Any

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from local_operator.remote_compute.attestation import (
    CODE_BAD_SIGNATURE,
    CODE_DEVICE_MISMATCH,
    CODE_DUPLICATE_KEY_ID,
    CODE_EMPTY_KEY_SET,
    CODE_EXPIRED,
    CODE_INVALID_KEY,
    CODE_ISSUED_IN_FUTURE,
    CODE_MALFORMED_KEY_SET,
    CODE_MALFORMED_PAYLOAD,
    CODE_MISSING_EXPECTATION,
    CODE_NETWORK_MISMATCH,
    CODE_NON_CANONICAL,
    CODE_UNKNOWN_KEY_ID,
    CODE_UNSUPPORTED_VERSION,
    CODE_WINDOW_INVERTED,
    CODE_WINDOW_TOO_LONG,
    CODE_WORKLOAD_MISMATCH,
    AttestationKey,
    AttestationRefusal,
    VerifiedAttestation,
    parse_attestation_keys,
    public_keys,
    verify_attestation,
)

FIXTURE = Path(__file__).parent / "fixtures" / "attestation_golden.json"


def _b64u(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _canonical(statement: dict[str, Any]) -> bytes:
    """The one signed spelling of a statement, as the contract defines it."""
    return json.dumps(statement, separators=(",", ":")).encode("utf-8")


def _public_key(seed_hex: str) -> bytes:
    private = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(seed_hex))
    return private.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    )


def _sign(seed_hex: str, payload_bytes: bytes) -> str:
    private = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(seed_hex))
    return _b64u(private.sign(payload_bytes))


@pytest.fixture(scope="module")
def vector() -> dict[str, Any]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def keys(vector: dict[str, Any]) -> dict[str, bytes]:
    return {vector["signer_key_id"]: bytes.fromhex(vector["verifier_key_hex"])}


@pytest.fixture(scope="module")
def device_key(vector: dict[str, Any]) -> bytes:
    return _public_key(vector["device_seed_hex"])


def _verify(vector: dict[str, Any], keys: dict[str, bytes], **overrides: Any):
    """The golden case, with any argument replaced by keyword for one test."""
    arguments: dict[str, Any] = {
        "payload_b64": vector["payload_b64"],
        "signature": vector["signature_b64"],
        "key_id": vector["signer_key_id"],
        "keys": keys,
        "expect_workload_id": vector["payload"]["workload_id"],
        "expect_network_id": vector["payload"]["network_id"],
        "expect_device_key": _public_key(vector["device_seed_hex"]),
        "now": vector["payload"]["issued_at"] + 1,
    }
    arguments.update(overrides)
    return verify_attestation(**arguments)


def _signed(vector: dict[str, Any], statement: dict[str, Any]) -> dict[str, str]:
    """A statement signed with the fixture's own signer seed."""
    payload_bytes = _canonical(statement)
    return {
        "payload_b64": _b64u(payload_bytes),
        "signature": _sign(vector["signer_seed_hex"], payload_bytes),
    }


# ---------------------------------------------------------------------------
# The golden vector
# ---------------------------------------------------------------------------


def test_fixture_is_internally_consistent(vector: dict[str, Any]) -> None:
    """The vector must reproduce its own canonical bytes, signature and device key.

    A fixture whose parts disagree would make every test below prove the wrong
    thing — a passing verifier against a broken vector is the one failure mode
    with no symptom.
    """
    statement = vector["payload"]
    assert _canonical(statement) == vector["canonical"].encode("utf-8")
    assert _b64u(_canonical(statement)) == vector["payload_b64"]
    assert bytes.fromhex(vector["verifier_key_hex"]) == _public_key(vector["signer_seed_hex"])
    assert _sign(vector["signer_seed_hex"], _canonical(statement)) == vector["signature_b64"]
    assert _b64u(_public_key(vector["device_seed_hex"])) == statement["device_key"]


def test_golden_vector_verifies(vector: dict[str, Any], keys: dict[str, bytes]) -> None:
    verified = _verify(vector, keys)
    assert isinstance(verified, VerifiedAttestation)
    assert verified.workload_id == vector["payload"]["workload_id"]
    assert verified.account_id == vector["payload"]["account_id"]
    assert verified.network_id == vector["payload"]["network_id"]
    assert verified.device_key == _public_key(vector["device_seed_hex"])
    assert verified.issued_at == vector["payload"]["issued_at"]
    assert verified.expires_at == vector["payload"]["expires_at"]
    assert verified.key_id == vector["signer_key_id"]


def test_verified_evidence_cannot_be_edited(vector: dict[str, Any], keys: dict[str, bytes]) -> None:
    verified = _verify(vector, keys)
    with pytest.raises(dataclasses.FrozenInstanceError):
        verified.network_id = "n_ffffffffffffffffffffffff"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Key selection: by id, never by search
# ---------------------------------------------------------------------------


def test_an_unpublished_key_id_is_refused(vector: dict[str, Any], keys: dict[str, bytes]) -> None:
    with pytest.raises(AttestationRefusal) as caught:
        _verify(vector, keys, key_id="attest-test-unknown")
    assert caught.value.code == CODE_UNKNOWN_KEY_ID


def test_a_named_key_that_did_not_sign_is_not_substituted(
    vector: dict[str, Any], keys: dict[str, bytes]
) -> None:
    """Both keys are published, the statement names the wrong one: no fallback.

    This is the try-every-key regression. The signer's key IS in the set, so a
    verifier that looped over the keys would succeed here; naming an id that did
    not sign must fail instead.
    """
    set_with_both = dict(keys)
    set_with_both["attest-test-2"] = _public_key(vector["device_seed_hex"])
    with pytest.raises(AttestationRefusal) as caught:
        _verify(vector, set_with_both, key_id="attest-test-2")
    assert caught.value.code == CODE_BAD_SIGNATURE


def test_a_key_of_the_wrong_length_is_refused(vector: dict[str, Any]) -> None:
    with pytest.raises(AttestationRefusal) as caught:
        _verify(vector, {vector["signer_key_id"]: b"\x00" * 31})
    assert caught.value.code == CODE_INVALID_KEY


# ---------------------------------------------------------------------------
# The signed bytes themselves
# ---------------------------------------------------------------------------


def test_a_tampered_payload_byte_is_refused(vector: dict[str, Any], keys: dict[str, bytes]) -> None:
    payload_bytes = base64.urlsafe_b64decode(
        vector["payload_b64"] + "=" * (-len(vector["payload_b64"]) % 4)
    )
    tampered = payload_bytes[:-1] + bytes([payload_bytes[-1] ^ 0x01])
    with pytest.raises(AttestationRefusal) as caught:
        _verify(vector, keys, payload_b64=_b64u(tampered))
    assert caught.value.code == CODE_BAD_SIGNATURE


def test_a_signature_over_a_different_payload_is_refused(
    vector: dict[str, Any], keys: dict[str, bytes]
) -> None:
    other = dict(vector["payload"])
    other["workload_id"] = "wl_ffffffffffffffffffffffff"
    with pytest.raises(AttestationRefusal) as caught:
        _verify(vector, keys, signature=_sign(vector["signer_seed_hex"], _canonical(other)))
    assert caught.value.code == CODE_BAD_SIGNATURE


def test_a_non_canonical_statement_is_refused_even_when_signed(
    vector: dict[str, Any], keys: dict[str, bytes]
) -> None:
    """Same values, different bytes — re-signed with the real seed, so only the
    canonical rule can catch it."""
    reordered = {key: vector["payload"][key] for key in ("workload_id", "v")}
    reordered.update(
        {key: value for key, value in vector["payload"].items() if key not in reordered}
    )
    signed = _signed(vector, reordered)
    with pytest.raises(AttestationRefusal) as caught:
        _verify(vector, keys, **signed)
    assert caught.value.code == CODE_NON_CANONICAL


def test_a_payload_that_is_not_base64_is_refused(
    vector: dict[str, Any], keys: dict[str, bytes]
) -> None:
    with pytest.raises(AttestationRefusal) as caught:
        _verify(vector, keys, payload_b64="not base64!")
    assert caught.value.code == CODE_MALFORMED_PAYLOAD


def test_a_validly_signed_non_json_payload_is_refused(
    vector: dict[str, Any], keys: dict[str, bytes]
) -> None:
    payload_bytes = b"not json"
    with pytest.raises(AttestationRefusal) as caught:
        _verify(
            vector,
            keys,
            payload_b64=_b64u(payload_bytes),
            signature=_sign(vector["signer_seed_hex"], payload_bytes),
        )
    assert caught.value.code == CODE_MALFORMED_PAYLOAD


def test_a_boolean_version_is_not_an_integer_version(
    vector: dict[str, Any], keys: dict[str, bytes]
) -> None:
    """``True == 1`` in Python, so a naive ``v == 1`` would admit it."""
    statement = dict(vector["payload"])
    statement["v"] = True
    with pytest.raises(AttestationRefusal) as caught:
        _verify(vector, keys, **_signed(vector, statement))
    assert caught.value.code == CODE_MALFORMED_PAYLOAD


@pytest.mark.parametrize("version", [0, 2])
def test_another_statement_version_is_refused(
    vector: dict[str, Any], keys: dict[str, bytes], version: int
) -> None:
    statement = dict(vector["payload"])
    statement["v"] = version
    with pytest.raises(AttestationRefusal) as caught:
        _verify(vector, keys, **_signed(vector, statement))
    assert caught.value.code == CODE_UNSUPPORTED_VERSION


def test_a_refusal_never_echoes_the_signature(
    vector: dict[str, Any], keys: dict[str, bytes]
) -> None:
    with pytest.raises(AttestationRefusal) as caught:
        _verify(vector, keys, signature=_sign(vector["signer_seed_hex"], b"other bytes"))
    assert vector["signature_b64"] not in str(caught.value)
    assert vector["verifier_key_hex"] not in str(caught.value)


def test_a_block_of_base64_longer_than_a_signature_is_refused(
    vector: dict[str, Any], keys: dict[str, bytes]
) -> None:
    """The encoded length is checked BEFORE decoding: 64 KiB is not a signature."""
    with pytest.raises(AttestationRefusal) as caught:
        _verify(vector, keys, signature="A" * 64 * 1024)
    assert caught.value.code == CODE_BAD_SIGNATURE
    assert "86" in str(caught.value)


@pytest.mark.parametrize(
    "signature",
    [
        _b64u(b"\x00" * 63),
        _b64u(b"\x00" * 65),
        _b64u(b"\x00" * 64) + "==",
    ],
)
def test_a_signature_of_the_wrong_encoded_length_is_refused(
    vector: dict[str, Any], keys: dict[str, bytes], signature: str
) -> None:
    with pytest.raises(AttestationRefusal) as caught:
        _verify(vector, keys, signature=signature)
    assert caught.value.code == CODE_BAD_SIGNATURE


def test_a_payload_longer_than_the_bound_is_refused_before_decoding(
    vector: dict[str, Any], keys: dict[str, bytes]
) -> None:
    """Refused while it is still a string, so the peer cannot size the decode."""
    with pytest.raises(AttestationRefusal) as caught:
        _verify(vector, keys, payload_b64="A" * 4097)
    assert caught.value.code == CODE_MALFORMED_PAYLOAD
    assert "4096" in str(caught.value)


def test_a_non_canonical_base64_spelling_is_refused(
    vector: dict[str, Any], keys: dict[str, bytes]
) -> None:
    """Trailing bits ``base64`` ignores: same 64 bytes, a different spelling.

    The last character of an 86-character signature carries four bits that no
    byte uses, so changing one of them decodes to the SAME signature and would
    otherwise be accepted. Re-encoding and comparing is the only way to see it.
    """
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
    last = alphabet.index(vector["signature_b64"][-1])
    other = alphabet[last | 1]
    assert other != vector["signature_b64"][-1]
    signature = vector["signature_b64"][:-1] + other
    with pytest.raises(AttestationRefusal) as caught:
        _verify(vector, keys, signature=signature)
    assert caught.value.code == CODE_BAD_SIGNATURE
    assert "canonical" in str(caught.value) or "base64" in str(caught.value)


# ---------------------------------------------------------------------------
# Every string field is printable ASCII, with no quote and no backslash
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("account_id", "acct_\x1b[31mEVIL"),
        ("account_id", "acct_000\x00"),
        ("network_id", "n_\u2028x"),
        ("workload_id", "wl_caf\u00e9"),
        ("device_key", "/etc\x7f"),
        ("workload_id", 'wl_"quoted"'),
        ("account_id", "acct_back\\\\slash"),
    ],
)
def test_a_control_character_in_a_string_field_is_refused(
    vector: dict[str, Any], keys: dict[str, bytes], field: str, value: str
) -> None:
    """V1: a validly signed statement is still refused for what it spells.

    ``json`` round-trips every one of these escapes byte-identically, so the
    canonical rule cannot see them and the signature is genuinely the
    signer's. Each statement below is re-signed with the fixture's own seed for
    that reason: the ONLY rule that may refuse it is the printable-ASCII one.
    """
    statement = dict(vector["payload"])
    statement[field] = value
    with pytest.raises(AttestationRefusal) as caught:
        _verify(vector, keys, **_signed(vector, statement))
    assert caught.value.code == CODE_MALFORMED_PAYLOAD
    assert "printable ASCII" in str(caught.value)


# ---------------------------------------------------------------------------
# The statement must be about THIS network and THIS device
# ---------------------------------------------------------------------------


def test_a_statement_about_another_workload_is_refused(
    vector: dict[str, Any], keys: dict[str, bytes]
) -> None:
    with pytest.raises(AttestationRefusal) as caught:
        _verify(vector, keys, expect_workload_id="wl_ffffffffffffffffffffffff")
    assert caught.value.code == CODE_WORKLOAD_MISMATCH


def test_a_statement_about_another_network_is_refused(
    vector: dict[str, Any], keys: dict[str, bytes]
) -> None:
    with pytest.raises(AttestationRefusal) as caught:
        _verify(vector, keys, expect_network_id="n_ffffffffffffffffffffffff")
    assert caught.value.code == CODE_NETWORK_MISMATCH


def test_a_statement_about_another_device_key_is_refused(
    vector: dict[str, Any], keys: dict[str, bytes]
) -> None:
    other_seed = "03" * 32
    with pytest.raises(AttestationRefusal) as caught:
        _verify(vector, keys, expect_device_key=_public_key(other_seed))
    assert caught.value.code == CODE_DEVICE_MISMATCH


@pytest.mark.parametrize(
    "overrides",
    [
        {"expect_workload_id": ""},
        {"expect_workload_id": "wl_0123456789abcdef01234567", "expect_network_id": ""},
        {"expect_network_id": "n_0123456789abcdef01234567", "expect_device_key": b""},
    ],
)
def test_an_empty_expectation_is_refused_before_the_statement_is_read(
    vector: dict[str, Any], keys: dict[str, bytes], overrides: dict[str, Any]
) -> None:
    """Refused even with a VALID signature, so the refusal cannot be the signature's."""
    with pytest.raises(AttestationRefusal) as caught:
        _verify(vector, keys, **overrides)
    assert caught.value.code == CODE_MISSING_EXPECTATION


def test_a_re_signed_statement_for_another_workload_is_refused(
    vector: dict[str, Any], keys: dict[str, bytes]
) -> None:
    """Re-signed with the vector's own seed: only the workload rule can refuse it."""
    statement = dict(vector["payload"])
    statement["workload_id"] = "wl_ffffffffffffffffffffffff"
    with pytest.raises(AttestationRefusal) as caught:
        _verify(vector, keys, **_signed(vector, statement))
    assert caught.value.code == CODE_WORKLOAD_MISMATCH


def test_the_network_is_checked_before_the_workload(
    vector: dict[str, Any], keys: dict[str, bytes]
) -> None:
    """Both wrong is a TENANT refusal: the codes must not be interchangeable."""
    statement = dict(vector["payload"])
    statement["workload_id"] = "wl_ffffffffffffffffffffffff"
    statement["network_id"] = "n_ffffffffffffffffffffffff"
    with pytest.raises(AttestationRefusal) as caught:
        _verify(vector, keys, **_signed(vector, statement))
    assert caught.value.code == CODE_NETWORK_MISMATCH


# ---------------------------------------------------------------------------
# Freshness: the half-open window and the skew allowance
# ---------------------------------------------------------------------------


def test_an_expired_statement_is_refused(vector: dict[str, Any], keys: dict[str, bytes]) -> None:
    with pytest.raises(AttestationRefusal) as caught:
        _verify(vector, keys, now=vector["payload"]["expires_at"] + 1)
    assert caught.value.code == CODE_EXPIRED


def test_the_window_is_half_open_so_now_at_expiry_is_expired(
    vector: dict[str, Any], keys: dict[str, bytes]
) -> None:
    with pytest.raises(AttestationRefusal) as caught:
        _verify(vector, keys, now=vector["payload"]["expires_at"])
    assert caught.value.code == CODE_EXPIRED


def test_the_last_live_second_is_accepted(vector: dict[str, Any], keys: dict[str, bytes]) -> None:
    assert _verify(vector, keys, now=vector["payload"]["expires_at"] - 1).expires_at


def test_a_statement_issued_too_far_ahead_is_refused(
    vector: dict[str, Any], keys: dict[str, bytes]
) -> None:
    now = vector["payload"]["issued_at"] - 61
    with pytest.raises(AttestationRefusal) as caught:
        _verify(vector, keys, now=now)
    assert caught.value.code == CODE_ISSUED_IN_FUTURE


def test_the_skew_boundary_is_inclusive(vector: dict[str, Any], keys: dict[str, bytes]) -> None:
    """``max_skew`` ahead is tolerated; one second more is not."""
    now = vector["payload"]["issued_at"] - 60
    assert _verify(vector, keys, now=now).issued_at == vector["payload"]["issued_at"]
    with pytest.raises(AttestationRefusal) as caught:
        _verify(vector, keys, now=now - 1)
    assert caught.value.code == CODE_ISSUED_IN_FUTURE


def test_a_custom_max_skew_moves_the_boundary(
    vector: dict[str, Any], keys: dict[str, bytes]
) -> None:
    now = vector["payload"]["issued_at"] - 5
    assert _verify(vector, keys, now=now, max_skew=5).issued_at
    with pytest.raises(AttestationRefusal) as caught:
        _verify(vector, keys, now=now, max_skew=4)
    assert caught.value.code == CODE_ISSUED_IN_FUTURE


def _windowed(vector: dict[str, Any], window: int) -> dict[str, Any]:
    statement = dict(vector["payload"])
    statement["expires_at"] = statement["issued_at"] + window
    return statement


def test_a_window_of_exactly_the_maximum_is_accepted(
    vector: dict[str, Any], keys: dict[str, bytes]
) -> None:
    statement = _windowed(vector, 3600)
    verified = _verify(vector, keys, **_signed(vector, statement))
    assert verified.expires_at - verified.issued_at == 3600


def test_a_window_one_second_longer_is_refused(
    vector: dict[str, Any], keys: dict[str, bytes]
) -> None:
    statement = _windowed(vector, 3601)
    with pytest.raises(AttestationRefusal) as caught:
        _verify(vector, keys, **_signed(vector, statement))
    assert caught.value.code == CODE_WINDOW_TOO_LONG


def test_a_zero_length_window_is_refused(vector: dict[str, Any], keys: dict[str, bytes]) -> None:
    """``expires_at == issued_at`` is a window that never opened.

    ``now`` is placed BEFORE ``issued_at`` so the window has not closed yet:
    otherwise the expiry rule fires first and the inversion is never reached,
    and the rule under test would go unexercised.
    """
    statement = _windowed(vector, 0)
    with pytest.raises(AttestationRefusal) as caught:
        _verify(vector, keys, now=statement["issued_at"] - 1, **_signed(vector, statement))
    assert caught.value.code == CODE_WINDOW_INVERTED


def test_a_window_that_closed_is_expired_not_inverted(
    vector: dict[str, Any], keys: dict[str, bytes]
) -> None:
    """Ordering pinned: a statement both closed and inverted reports ``expired``."""
    statement = _windowed(vector, 0)
    with pytest.raises(AttestationRefusal) as caught:
        _verify(vector, keys, **_signed(vector, statement))
    assert caught.value.code == CODE_EXPIRED


# ---------------------------------------------------------------------------
# Key discovery
# ---------------------------------------------------------------------------


def _discovery_body(status: str = "current") -> dict[str, Any]:
    return {
        "keys": [
            {
                "key_id": "k1",
                "alg": "Ed25519",
                "public_key": _b64u(b"\x11" * 32),
                "status": status,
            }
        ]
    }


@pytest.mark.parametrize("status", ["current", "previous"])
def test_discovery_parses_either_published_status(status: str) -> None:
    parsed = parse_attestation_keys(_discovery_body(status))
    assert parsed == {"k1": AttestationKey(public_key=b"\x11" * 32, status=status)}
    assert parsed["k1"].status == status


def test_a_previous_only_key_set_is_accepted() -> None:
    """SV2: rotation publishes the outgoing key as ``previous`` while its
    statements are still inside their window, so this must not be refused."""
    parsed = parse_attestation_keys(_discovery_body("previous"))
    assert set(parsed) == {"k1"}


def test_discovery_keeps_two_keys_apart() -> None:
    body = _discovery_body()
    body["keys"].append(
        {
            "key_id": "k2",
            "alg": "Ed25519",
            "public_key": _b64u(b"\x22" * 32),
            "status": "previous",
        }
    )
    assert public_keys(parse_attestation_keys(body)) == {
        "k1": b"\x11" * 32,
        "k2": b"\x22" * 32,
    }


def test_discovery_ignores_fields_it_does_not_read() -> None:
    body = _discovery_body()
    body["keys"][0]["added_at"] = 1760000000
    body["next_cursor"] = "page-2"
    assert public_keys(parse_attestation_keys(body)) == {"k1": b"\x11" * 32}


def test_discovery_refuses_an_empty_key_set() -> None:
    """A server that published no keys has not answered; it has not said "trust
    nothing" in a way a verifier may act on."""
    with pytest.raises(AttestationRefusal) as caught:
        parse_attestation_keys({"keys": []})
    assert caught.value.code == CODE_EMPTY_KEY_SET


def test_discovery_refuses_two_ids_naming_one_key() -> None:
    """SV3: the id is the selector, so two ids over one key make it select nothing."""
    body = _discovery_body()
    body["keys"].append(dict(body["keys"][0], key_id="k2"))
    with pytest.raises(AttestationRefusal) as caught:
        parse_attestation_keys(body)
    assert caught.value.code == CODE_DUPLICATE_KEY_ID
    assert "same public key" in str(caught.value)


def test_discovery_refuses_another_algorithm() -> None:
    body = _discovery_body()
    body["keys"][0]["alg"] = "RSA"
    with pytest.raises(AttestationRefusal) as caught:
        parse_attestation_keys(body)
    assert caught.value.code == CODE_MALFORMED_KEY_SET


def test_discovery_refuses_a_short_key() -> None:
    body = _discovery_body()
    body["keys"][0]["public_key"] = _b64u(b"\x11" * 16)
    with pytest.raises(AttestationRefusal) as caught:
        parse_attestation_keys(body)
    assert caught.value.code == CODE_MALFORMED_KEY_SET


def test_discovery_refuses_a_duplicate_id() -> None:
    body = _discovery_body()
    body["keys"].append(dict(body["keys"][0], public_key=_b64u(b"\x22" * 32)))
    with pytest.raises(AttestationRefusal) as caught:
        parse_attestation_keys(body)
    assert caught.value.code == CODE_DUPLICATE_KEY_ID


def test_discovery_refuses_an_unknown_status() -> None:
    with pytest.raises(AttestationRefusal) as caught:
        parse_attestation_keys(_discovery_body("revoked"))
    assert caught.value.code == CODE_MALFORMED_KEY_SET


@pytest.mark.parametrize(
    "body",
    [{}, {"keys": "nope"}, {"keys": [1]}, {"keys": [{"alg": "Ed25519"}]}, ["keys"]],
)
def test_discovery_refuses_a_body_it_cannot_read(body: Any) -> None:
    with pytest.raises(AttestationRefusal) as caught:
        parse_attestation_keys(body)
    assert caught.value.code == CODE_MALFORMED_KEY_SET


def test_a_parsed_key_set_verifies_the_golden_vector(vector: dict[str, Any]) -> None:
    """The two halves fit: discovery's output is what verify_attestation takes."""
    keys = public_keys(
        parse_attestation_keys(
            {
                "keys": [
                    {
                        "key_id": vector["signer_key_id"],
                        "alg": "Ed25519",
                        "public_key": _b64u(bytes.fromhex(vector["verifier_key_hex"])),
                        "status": "current",
                    }
                ]
            }
        )
    )
    assert _verify(vector, keys).key_id == vector["signer_key_id"]
