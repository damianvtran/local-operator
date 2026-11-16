"""#3066 m3/Q5: cooldowns narrow actual owner rows, not the whole cascade.

Use the real share CLI, owner AuthStore and authenticated two-relay grant path.
The ambient API key is synthetic and is read through the provider registry, not
injected as a broker reply. No request may broaden a real row exclusion into an
ambient credential. Unknown inventory must retain that conservative fence.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import time
from pathlib import Path
from typing import Any

import pytest

from local_operator.network.credentials.state import PlacementState
from local_operator.network.credentials.types import BrokerError, Grant
from local_operator.providers import registry
from local_operator.providers.auth_store import AuthStore
from tests.unit.network.test_credentials_real_link import (  # noqa: F401 — fixture registration
    STUB_PROVIDER,
    _borrower_client,
    _pulled_or_held,
    _share,
    devices,
    idp,
    mesh,
    stub_provider,
)

_FIXTURES = (devices, idp, mesh, stub_provider)
ENV = "LOP_CASCADE_SYNTHETIC_API_KEY"
AMBIENT = "synthetic-ambient-not-a-live-key"


def _ambient(monkeypatch: pytest.MonkeyPatch) -> None:
    definition = registry.get_provider_definition(STUB_PROVIDER)
    assert definition is not None
    monkeypatch.setitem(
        registry._BY_ID, STUB_PROVIDER, dataclasses.replace(definition, env_keys=ENV)
    )
    monkeypatch.setenv(ENV, AMBIENT)


def _rest(client: Any, owner: str, rows: list[int]) -> list[dict[str, Any]]:
    for row in rows:
        client.state.cool_down(
            STUB_PROVIDER,
            owner_device=owner,
            credential_id=row,
            block_scope="",
            until=time.time() + 600,
            label="synthetic-account",
        )
    client._save_state()
    return json.loads(json.dumps(client.state.cooldowns(STUB_PROVIDER, owner_device=owner)))


def test_m3_env_does_not_shadow_a_lendable_stored_row(
    mesh: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    _ambient(monkeypatch)
    auth = AuthStore(config_dir=mesh.a.root)
    try:
        auth.upsert_credential(
            STUB_PROVIDER,
            {
                "type": "oauth",
                "access": "synthetic-second",
                "refresh": "unused",
                "expires": int(time.time() * 1000) + 600_000,
                "email": "second@example.test",
            },
        )
        oauth_ids = [
            r.id for r in auth.list_credentials(STUB_PROVIDER) if r.credential_type == "oauth"
        ]
        assert len(oauth_ids) == 2
        auth.upsert_credential(STUB_PROVIDER, {"type": "api_key", "key": "synthetic-stored"})
        stored = next(
            r for r in auth.list_credentials(STUB_PROVIDER) if r.credential_type == "api_key"
        )
        # Ordinary local precedence must stay env-before-tier6, with and without
        # explicitly passing the new default. No excluded OAuth row is refreshed.
        local = asyncio.run(
            auth.get_oauth_access(STUB_PROVIDER, read_only=True, exclude_credential_ids=oauth_ids)
        )
        assert local is not None and local.credential_id == 0
        assert local.access_token == AMBIENT
        _share(mesh)
        assert _pulled_or_held(mesh, STUB_PROVIDER)
        client = _borrower_client(mesh)
        before = _rest(client, mesh.owner, oauth_ids)
        result = client.request_grant_sync(STUB_PROVIDER, session_id="m3-stored")
        assert isinstance(result, Grant), getattr(result, "code", type(result).__name__)
        assert result.credential_ref.credential_id == stored.id
        assert client.state.cooldowns(STUB_PROVIDER, owner_device=mesh.owner) == before
        assert len(mesh.idp.posts) == 0
        explicit = asyncio.run(
            auth.get_oauth_access(
                STUB_PROVIDER,
                read_only=True,
                exclude_credential_ids=oauth_ids,
                persisted_kinds=None,
            )
        )
        assert explicit is not None
        assert (
            json.dumps(dataclasses.asdict(local), sort_keys=True).encode()
            == json.dumps(dataclasses.asdict(explicit), sort_keys=True).encode()
        )
        monkeypatch.delenv(ENV)
        client.grants.drop(STUB_PROVIDER)
        no_env = client.request_grant_sync(STUB_PROVIDER, session_id="m3-no-env-control")
        assert isinstance(no_env, Grant), getattr(no_env, "code", type(no_env).__name__)
        assert no_env.credential_ref.credential_id == stored.id
        assert len(mesh.idp.posts) == 0
        print(
            "WIRE_RESULT",
            json.dumps(
                {
                    "case": "m3",
                    "selected_stored_row": result.credential_ref.credential_id,
                    "excluded": oauth_ids,
                    "idp_posts": len(mesh.idp.posts),
                }
            ),
        )
    finally:
        auth.close()


def test_q5_stale_ids_do_not_hide_the_ordinary_env_share(
    mesh: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    _ambient(monkeypatch)
    auth = AuthStore(config_dir=mesh.a.root)
    try:
        # Grant the share while the owner still has its login, then log out:
        # this is QA6463826030's stale-rest shape, not a new env-only share.
        _share(mesh)
        assert _pulled_or_held(mesh, STUB_PROVIDER)
        old_ids = [r.id for r in auth.list_credentials(STUB_PROVIDER)]
        for row in old_ids:
            auth.delete_credential(row)
        assert auth.list_credentials(STUB_PROVIDER) == []
        client = _borrower_client(mesh)
        baseline = client.request_grant_sync(STUB_PROVIDER, session_id="q5-no-exclusion")
        assert isinstance(baseline, Grant), getattr(baseline, "code", type(baseline).__name__)
        assert baseline.credential_ref.credential_id == 0
        client.grants.drop(STUB_PROVIDER)
        before = _rest(client, mesh.owner, old_ids)
        result = client.request_grant_sync(STUB_PROVIDER, session_id="q5-stale-only")
        assert isinstance(result, Grant), getattr(result, "code", type(result).__name__)
        assert result.credential_ref.credential_id == 0
        assert result.access_token == baseline.access_token
        assert client.state.cooldowns(STUB_PROVIDER, owner_device=mesh.owner) == before
        # Re-open the persisted ledger, not merely this client's in-memory view:
        # normalization is a resolve choice, never a rewrite of holder evidence.
        persisted = PlacementState.load(mesh.network_id, mesh.b.root)
        assert persisted.cooldowns(STUB_PROVIDER, owner_device=mesh.owner) == before
        assert len(mesh.idp.posts) == 0
        print(
            "WIRE_RESULT",
            json.dumps(
                {
                    "case": "q5",
                    "ordinary_row": 0,
                    "stale_rows_retained": len(before),
                    "idp_posts": len(mesh.idp.posts),
                }
            ),
        )
    finally:
        auth.close()


@pytest.mark.parametrize("with_stale_id", [False, True])
def test_real_owner_row_exclusions_never_escape_to_ambient(
    mesh: Any, monkeypatch: pytest.MonkeyPatch, with_stale_id: bool
) -> None:
    _ambient(monkeypatch)
    auth = AuthStore(config_dir=mesh.a.root)
    try:
        rows = [r.id for r in auth.list_credentials(STUB_PROVIDER)]
        _share(mesh)
        assert _pulled_or_held(mesh, STUB_PROVIDER)
        client = _borrower_client(mesh)
        before = _rest(client, mesh.owner, rows + ([999_999] if with_stale_id else []))
        result = client.request_grant_sync(STUB_PROVIDER, session_id="real-exclusion")
        assert isinstance(result, BrokerError)
        assert result.code == "quota_blocked"
        assert client.state.cooldowns(STUB_PROVIDER, owner_device=mesh.owner) == before
        assert len(mesh.idp.posts) == 0
    finally:
        auth.close()


@pytest.mark.parametrize("failure", ["unknown", "error"])
def test_unknown_inventory_retains_the_nonrow_fence(
    mesh: Any, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    _ambient(monkeypatch)
    auth = AuthStore(config_dir=mesh.a.root)
    try:
        _share(mesh)
        assert _pulled_or_held(mesh, STUB_PROVIDER)
        rows = [r.id for r in auth.list_credentials(STUB_PROVIDER)]
        for row in rows:
            auth.delete_credential(row)
        client = _borrower_client(mesh)
        before = _rest(client, mesh.owner, rows)
        reads: list[str] = []

        def inventory(self: AuthStore, provider: str) -> frozenset[int] | None:
            # Fail ONLY the normalization's metadata read. The real selector
            # still sees its truthful empty pool; no grant reply is forged.
            reads.append(provider)
            if failure == "error":
                raise OSError("synthetic inventory unavailable")
            return None

        monkeypatch.setattr(AuthStore, "current_credential_ids", inventory)
        result = client.request_grant_sync(STUB_PROVIDER, session_id="unknown-inventory")
        assert reads, "the unknown-inventory control did not reach the normalization read"
        assert isinstance(result, BrokerError)
        assert result.code == "no_local_credential"
        assert client.state.cooldowns(STUB_PROVIDER, owner_device=mesh.owner) == before
        assert len(mesh.idp.posts) == 0
    finally:
        auth.close()


@pytest.mark.parametrize("nonrow", ["env", "fallback", "runtime", "config"])
def test_only_opted_in_resolves_skip_nonrow_tiers(
    request: pytest.FixtureRequest, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, nonrow: str
) -> None:
    request.getfixturevalue("stub_provider")
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    if nonrow == "env":
        _ambient(monkeypatch)
    auth = AuthStore(config_dir=tmp_path)
    try:
        auth.upsert_credential(STUB_PROVIDER, {"type": "api_key", "key": "synthetic-row"})
        row = auth.list_credentials(STUB_PROVIDER)[0]
        calls: list[str] = []

        def fallback(provider: str) -> str:
            calls.append(provider)
            return "synthetic-fallback"

        if nonrow == "fallback":
            auth.set_fallback_resolver(STUB_PROVIDER, fallback)
            # An ordinary fallback is reached only with no stored winner.
            auth.block_credential(row.id, STUB_PROVIDER)
        elif nonrow == "runtime":
            auth._runtime_overrides[STUB_PROVIDER] = "synthetic-runtime"
        elif nonrow == "config":
            auth._config_overrides[STUB_PROVIDER] = "synthetic-config"
        default = asyncio.run(auth.get_oauth_access(STUB_PROVIDER, read_only=True))
        explicit = asyncio.run(
            auth.get_oauth_access(STUB_PROVIDER, read_only=True, persisted_kinds=None)
        )
        assert default == explicit
        if nonrow in {"runtime", "config"}:
            assert default is None
        else:
            assert default is not None and default.credential_id == 0
            assert explicit is not None
            assert (
                json.dumps(dataclasses.asdict(default), sort_keys=True).encode()
                == json.dumps(dataclasses.asdict(explicit), sort_keys=True).encode()
            )
        before_calls = len(calls)
        restricted = asyncio.run(
            auth.get_oauth_access(
                STUB_PROVIDER, read_only=True, persisted_kinds=frozenset({"oauth", "api_key"})
            )
        )
        if nonrow == "fallback":
            assert restricted is None  # a blocked row never opens the fallback tier
        else:
            assert restricted is not None and restricted.credential_id == row.id
        assert len(calls) == before_calls
    finally:
        auth.close()


def test_inventory_distinguishes_empty_from_closed_without_decrypting_rows(
    request: pytest.FixtureRequest, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request.getfixturevalue("stub_provider")
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    auth = AuthStore(config_dir=tmp_path)
    try:
        assert auth.current_credential_ids(STUB_PROVIDER) == frozenset()
        auth.upsert_credential(STUB_PROVIDER, {"type": "api_key", "key": "synthetic-metadata"})
        row = auth.list_credentials(STUB_PROVIDER)[0]

        def no_bearer_read(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError("row inventory must not decode bearer payloads")

        monkeypatch.setattr(auth, "_row_to_credential", no_bearer_read)
        assert auth.current_credential_ids(STUB_PROVIDER) == frozenset({row.id})
        auth.close()
        assert auth.current_credential_ids(STUB_PROVIDER) is None
    finally:
        auth.close()
