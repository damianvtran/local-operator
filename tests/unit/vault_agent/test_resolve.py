"""H5-H8: pickup, refusal map, single shot and the exact endpoint, through a stub transport."""

from __future__ import annotations

import base64
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from local_operator.vault_agent import eligibility, keys, wire
from local_operator.vault_agent.resolve import (
    UNAVAILABLE,
    VaultAgentRefusal,
    parse_reference,
    pick_up,
    resolve_item,
)

from .conftest import (
    AUTHORITY_CHALLENGE,
    ENDPOINT,
    ITEM,
    LOCAL_CHALLENGE,
    SEED,
    TENANT,
    VAULT,
    Transport,
    make_client,
    mint_answer,
)

NOW = 1_800_000_000


def _resolve(config_dir: Path, transport: Transport):
    return resolve_item(
        config_dir,
        vault_id=VAULT,
        item_id=ITEM,
        client_factory=lambda _root: make_client(transport),
        clock=lambda: NOW + 0.9,
    )


# -- H5: pickup ---------------------------------------------------------------


def _row(**overrides):
    row = {
        "delegation_id": "vdl_x",
        "tenant_id": TENANT,
        "endpoint_id": ENDPOINT,
        "key_fingerprint": keys.fingerprint_for_seed(SEED),
        "expires_at": "2026-11-10T00:00:00Z",
        "state": "live",
    }
    row.update(overrides)
    return row


def test_pickup_follows_a_dotted_cursor_and_picks_the_exact_match() -> None:
    fp = keys.fingerprint_for_seed(SEED)
    cursor = "vdl_" + "a" * 32 + "." + "b" * 16
    transport = Transport(
        answers=[
            (
                200,
                {
                    "delegations": [_row(delegation_id="vdl_wrong_ep", endpoint_id="bep_other")],
                    "next_cursor": cursor,
                },
            ),
            (
                200,
                {
                    "delegations": [
                        _row(delegation_id="vdl_wrong_fp", key_fingerprint="0" * 64),
                        _row(delegation_id="vdl_match"),
                        _row(
                            delegation_id="vdl_revoked",
                            state="revoked",
                            expires_at="2027-01-01T00:00:00Z",
                        ),
                    ],
                    "next_cursor": None,
                },
            ),
        ]
    )
    grant = pick_up(make_client(transport), tenant_id=TENANT, endpoint_id=ENDPOINT, fingerprint=fp)
    assert grant is not None and grant.delegation_id == "vdl_match"
    assert transport.calls[1]["params"] == {"tenant_id": TENANT, "cursor": cursor}
    assert all(call["allow_redirects"] is False for call in transport.calls)


def test_pickup_takes_the_latest_expiry_of_several_matches() -> None:
    fp = keys.fingerprint_for_seed(SEED)
    transport = Transport(
        answers=[
            (
                200,
                {
                    "delegations": [
                        _row(delegation_id="vdl_early", expires_at="2026-11-08T00:00:00Z"),
                        _row(delegation_id="vdl_late", expires_at="2026-11-12T00:00:00Z"),
                        _row(delegation_id="vdl_mid", expires_at="2026-11-09T00:00:00Z"),
                    ],
                    "next_cursor": None,
                },
            )
        ]
    )
    grant = pick_up(make_client(transport), tenant_id=TENANT, endpoint_id=ENDPOINT, fingerprint=fp)
    assert grant is not None and grant.delegation_id == "vdl_late"


def test_pickup_with_no_match_is_none() -> None:
    fp = keys.fingerprint_for_seed(SEED)
    transport = Transport(
        answers=[
            (200, {"delegations": [_row(endpoint_id="bep_raw-mixed_123")], "next_cursor": None})
        ]
    )
    # A case-folded endpoint is NOT ours (H8 at the selection step too).
    assert (
        pick_up(make_client(transport), tenant_id=TENANT, endpoint_id=ENDPOINT, fingerprint=fp)
        is None
    )


# -- the happy path, H8 exact endpoint ------------------------------------------


def test_resolve_mints_signs_and_calls_route_15_with_exact_bytes(config_dir, open_gate) -> None:
    transport = Transport(
        answers=[(200, mint_answer()), (200, {"value": "  canary-value\n", "expires_at": None})]
    )
    resolved = _resolve(config_dir, transport)
    assert resolved.value == "  canary-value\n"  # not trimmed

    mint, call = transport.calls
    assert mint["method"] == "POST" and mint["url"].endswith("/v1/vaults/challenges")
    assert mint["json"] == {"op": "resolve", "vault_id": VAULT, "item_id": ITEM}
    assert call["url"].endswith(f"/v1/vault-items/{ITEM}/resolve")
    assert call["params"] == {"as": "agent", "endpoint_id": ENDPOINT, "vault_id": VAULT}
    headers = call["headers"]
    assert headers["X-Vault-Challenge"] == LOCAL_CHALLENGE
    assert headers["X-Vault-Delegation"] == "vdl_a"
    assert headers["X-Vault-Agent-Ts"] == str(NOW)
    assert headers["Authorization"] == "Bearer person-token"

    # The proof verifies over the EXACT endpoint and the AUTHORITY challenge id.
    commitment = wire.resolve_commitment(
        tenant_id=TENANT,
        vault_id=VAULT,
        item_id=ITEM,
        authority_challenge_id=AUTHORITY_CHALLENGE,
        ledger_version=(1 << 53) + 1,
    )
    message = wire.agent_proof_message(
        commitment=commitment, delegation_id="vdl_a", endpoint_id=ENDPOINT, timestamp=NOW
    )
    proof = headers["X-Vault-Agent-Proof"]
    signature = base64.urlsafe_b64decode(proof + "=" * (-len(proof) % 4))
    Ed25519PublicKey.from_public_bytes(keys.public_key_from_seed(SEED)).verify(signature, message)


@pytest.mark.parametrize(
    "bad",
    [
        {"tenant_id": "tnt_other"},
        {"challenge_id": "vch_" + "1" * 32},
        {"authority_challenge_id": "vch_" + "2" * 64},
        {"ledger_version": 42},
        {"ledger_version": "042"},
    ],
)
def test_a_malformed_mint_is_refused_before_route_15(config_dir, open_gate, bad) -> None:
    transport = Transport(answers=[(200, mint_answer(**bad))])
    with pytest.raises(VaultAgentRefusal) as info:
        _resolve(config_dir, transport)
    assert info.value.code == ("wrong_tenant" if "tenant_id" in bad else "protocol_error")
    assert len(transport.calls) == 1


# -- H6: the refusal map ---------------------------------------------------------


@pytest.mark.parametrize(
    ("status", "body", "code", "phrase"),
    [
        (
            403,
            {"code": "assertion_invalid", "details": {"reason": "proof_invalid"}},
            "proof_invalid",
            "clock",
        ),
        (403, {"code": "vault_no_access"}, "vault_no_access", "no longer a member"),
        (401, {"code": "identity_required"}, "identity_required", "lop login radient"),
        (410, {"code": "secret_expired"}, "secret_expired", "expired"),
        (404, {"code": "not_found"}, "not_found", "No such Vault item"),
        (
            403,
            {"code": "conversation_vaults_disabled"},
            "conversation_vaults_disabled",
            "aren't available yet",
        ),
        (503, {"code": "vault_unavailable"}, "unavailable", "isn't available yet"),
    ],
)
def test_route_15_refusals_map_to_a_remedy(
    config_dir, open_gate, status, body, code, phrase
) -> None:
    transport = Transport(answers=[(200, mint_answer()), (status, body)])
    with pytest.raises(VaultAgentRefusal) as info:
        _resolve(config_dir, transport)
    assert info.value.code == code
    assert phrase in str(info.value)
    # The grant survives every refusal but delegation_required.
    assert eligibility.read_grant(
        config_dir,
        tenant_id=TENANT,
        endpoint_id=ENDPOINT,
        fingerprint=keys.fingerprint_for_seed(SEED),
    )


def test_delegation_required_clears_the_cached_grant(config_dir, open_gate) -> None:
    transport = Transport(answers=[(200, mint_answer()), (403, {"code": "delegation_required"})])
    with pytest.raises(VaultAgentRefusal) as info:
        _resolve(config_dir, transport)
    assert info.value.code == "delegation_required"
    assert "lop vault connect" in str(info.value)
    assert not (eligibility.vault_dir(config_dir) / eligibility.GRANT_NAME).exists()


@pytest.mark.parametrize("code", ["vault_not_available", "plan_required"])
def test_a_plan_refusal_closes_the_gate_record(config_dir, open_gate, code) -> None:
    transport = Transport(answers=[(403, {"code": code})])
    with pytest.raises(VaultAgentRefusal) as info:
        _resolve(config_dir, transport)
    assert info.value.code == code
    record = eligibility._read_json(
        eligibility.vault_dir(config_dir) / eligibility.ELIGIBILITY_NAME
    )
    assert record is not None and record["eligible"] is False and record["closed_by"] == code


def test_a_dark_mint_is_unavailable_not_an_error_loop(config_dir, open_gate) -> None:
    transport = Transport(answers=[(503, {"code": "vault_unavailable"})])
    with pytest.raises(VaultAgentRefusal) as info:
        _resolve(config_dir, transport)
    assert str(info.value) == UNAVAILABLE
    assert len(transport.calls) == 1


# -- H7: single shot -------------------------------------------------------------


def test_a_503_on_route_15_is_never_retried(config_dir, open_gate) -> None:
    transport = Transport(answers=[(200, mint_answer()), (503, {"code": "vault_unavailable"})])
    with pytest.raises(VaultAgentRefusal):
        _resolve(config_dir, transport)
    # One mint, one resolve: no second presentation of the burned challenge.
    assert [c["url"].rsplit("/", 1)[-1] for c in transport.calls] == ["challenges", "resolve"]
    assert not transport.answers


# -- preconditions refuse before any HTTP ------------------------------------------


def test_no_grant_refuses_before_any_call(config_dir, open_gate) -> None:
    eligibility.clear_grant(config_dir)
    transport = Transport()
    with pytest.raises(VaultAgentRefusal) as info:
        _resolve(config_dir, transport)
    assert info.value.code == "not_connected"
    assert transport.calls == []


def test_a_closed_gate_refuses_before_any_call(config_dir, monkeypatch) -> None:
    monkeypatch.setattr(eligibility, "gate", lambda _root: None)
    transport = Transport()
    with pytest.raises(VaultAgentRefusal) as info:
        _resolve(config_dir, transport)
    assert info.value.code == "not_available"
    assert transport.calls == []


@pytest.mark.parametrize(
    "link",
    [
        "https://evil.example/dashboard/vaults/vlt_abc/items/vit_def",
        "http://console.radienthq.com/dashboard/vaults/vlt_abc/items/vit_def",
        "https://console.radienthq.com:8443/dashboard/vaults/vlt_abc/items/vit_def",
        "https://u@console.radienthq.com/dashboard/vaults/vlt_abc/items/vit_def",
        "https://console.radienthq.com.evil/dashboard/vaults/vlt_abc/items/vit_def",
    ],
)
def test_parse_reference_refuses_a_link_from_any_other_origin(link) -> None:
    """Security S-5: the console's exact origin, nothing else."""
    with pytest.raises(VaultAgentRefusal) as info:
        parse_reference(None, link)
    assert info.value.code == "invalid_reference"


def test_a_damaged_key_is_a_refusal_with_a_remedy(config_dir, open_gate, monkeypatch) -> None:
    from local_operator.secrets.errors import SecretCorrupt

    def damaged(endpoint, base=None):
        raise SecretCorrupt("damaged")

    monkeypatch.setattr(keys, "load_seed", damaged)
    transport = Transport()
    with pytest.raises(VaultAgentRefusal) as info:
        _resolve(config_dir, transport)
    assert info.value.code == "key_unreadable"
    assert "lop vault disconnect" in str(info.value) and transport.calls == []


def test_a_non_nfc_id_is_refused_before_route_15(config_dir, open_gate, monkeypatch) -> None:
    """Security S-6: the authority signs NFC bytes; a non-NFC endpoint id could
    never verify, so it is refused locally rather than normalised."""
    decomposed = "bep_cafe\u0301"
    gate = eligibility.Gate(tenant_id=TENANT, endpoint_id=decomposed, tenant_name="Acme")
    monkeypatch.setattr(eligibility, "gate", lambda _root: gate)
    monkeypatch.setattr(keys, "load_seed", lambda endpoint, base=None: SEED)
    fingerprint = keys.fingerprint_for_seed(SEED)
    eligibility.write_grant(
        config_dir,
        eligibility.Grant("vdl_a", "2999-01-01T00:00:00Z", TENANT, decomposed, fingerprint),
    )
    transport = Transport(answers=[(200, mint_answer())])
    with pytest.raises(VaultAgentRefusal) as info:
        _resolve(config_dir, transport)
    assert info.value.code == "invalid_id"
    assert [c["url"].rsplit("/", 1)[-1] for c in transport.calls] == ["challenges"]


def test_parse_reference_accepts_a_console_item_link() -> None:
    url = "https://console.radienthq.com/dashboard/vaults/vlt_abc/items/vit_def"
    assert parse_reference(None, url) == ("vlt_abc", "vit_def")
    assert parse_reference("vlt_x", "vit_y") == ("vlt_x", "vit_y")
    with pytest.raises(VaultAgentRefusal):
        parse_reference("vlt_x", "vit_y/../../admin")
    with pytest.raises(VaultAgentRefusal):
        parse_reference(None, "vit_y")
