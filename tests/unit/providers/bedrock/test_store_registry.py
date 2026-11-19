"""T7/T8: seal hook, identity dedupe, registry wiring, mesh lending, import guard."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from local_operator.providers.auth_store import AuthStore
from local_operator.providers.bedrock import connection as bconn

# Built from parts so the fixture values read as obviously synthetic.
SECRET = "fake-bedrock-key-" + "PLAINTEXT" + "-0001"
PAIR_SECRET = "fake-pair-secret-" + "PLAINTEXT" + "-0002"
ACCOUNT = "access-key:" + "AKIDFAKE" + "@us-east-1"


def _secret_names(base: Path) -> set[str]:
    from local_operator.secrets.access import open_store

    return {record.name for record in open_store(base).list()}


def _db_bytes(base: Path) -> bytes:
    data = b""
    for path in base.glob("auth.db*"):
        data += path.read_bytes()
    return data


def test_raw_key_row_is_sealed_and_never_persisted_in_plaintext(tmp_path: Path) -> None:
    store = AuthStore(tmp_path / "auth.db", config_dir=tmp_path)
    try:
        row = store.upsert_credential("bedrock", {"type": "api_key", "key": SECRET})
        key = row.data["key"]
        conn = bconn.parse(key)
        assert conn.method == "api-key" and conn.ref and bconn.REF_PATTERN.fullmatch(conn.ref)
        assert SECRET not in key
        store._conn.commit()
        assert SECRET.encode() not in _db_bytes(tmp_path)
        assert any(conn.ref in name for name in _secret_names(tmp_path))
        from local_operator.providers.registry import provider_secret_value

        assert provider_secret_value(conn.ref, base=tmp_path) == SECRET
    finally:
        store.close()


def test_access_key_pair_is_sealed_and_replace_and_delete_remove_refs(tmp_path: Path) -> None:
    store = AuthStore(tmp_path / "auth.db", config_dir=tmp_path)
    try:
        ref_placeholder = "BEDROCK_SECRET_" + "0" * 32
        payload = {
            "type": "api_key",
            "source": "login",
            "key": f"bedrock://access-key?region=us-east-1&ref={ref_placeholder}",
            "account_id": ACCOUNT,
            "secret": {"access_key_id": "AKIDFAKE", "secret_access_key": PAIR_SECRET},
        }
        first = store.upsert_credential("bedrock", dict(payload))
        first_ref = bconn.parse(first.data["key"]).ref
        assert first_ref and first_ref != ref_placeholder
        assert "secret" not in first.data
        store._conn.commit()
        assert PAIR_SECRET.encode() not in _db_bytes(tmp_path)

        # Same identity re-saved: one row, new ref, old ref deleted.
        second = store.upsert_credential("bedrock", dict(payload))
        second_ref = bconn.parse(second.data["key"]).ref
        assert second_ref and second.id == first.id and second_ref != first_ref
        names = _secret_names(tmp_path)
        assert any(second_ref in n for n in names)
        assert not any(first_ref in n for n in names)
        assert len(store.list_credentials("bedrock")) == 1

        store.delete_credential(second.id)
        assert not any(second_ref in n for n in _secret_names(tmp_path))
        assert store.list_credentials("bedrock") == []
    finally:
        store.close()


def test_profile_connections_dedupe_on_profile_and_region(tmp_path: Path) -> None:
    store = AuthStore(tmp_path / "auth.db", config_dir=tmp_path)
    try:
        row = {
            "type": "api_key",
            "source": "login",
            "key": "bedrock://profile?name=dev&region=us-east-1",
        }
        a = store.upsert_credential("bedrock", dict(row))
        b = store.upsert_credential("bedrock", dict(row))
        c = store.upsert_credential(
            "bedrock", {**row, "key": "bedrock://profile?name=dev&region=eu-west-1"}
        )
        assert a.id == b.id and c.id != a.id
        assert a.data["key"] == row["key"]  # nothing sealed for a profile
    finally:
        store.close()


def test_registry_row_and_helpers() -> None:
    from local_operator.providers.registry import (
        get_provider_definition,
        is_anthropic_messages_wire,
    )

    definition = get_provider_definition("bedrock")
    assert definition is not None
    assert definition.wire == "bedrock" and definition.base_url is None
    assert definition.name == "Amazon Bedrock"
    assert is_anthropic_messages_wire("bedrock") and is_anthropic_messages_wire("anthropic")
    assert not is_anthropic_messages_wire("openai-compat")


def test_client_factory_builds_the_bedrock_client() -> None:
    from local_operator.harness.types import ModelSpec
    from local_operator.providers.bedrock.client import BedrockAnthropicClient
    from local_operator.providers.clients import client_for_spec

    spec = ModelSpec(provider="bedrock", model_id="global.anthropic.claude-opus-5-5")
    assert isinstance(client_for_spec(spec), BedrockAnthropicClient)


def test_bedrock_is_never_lent_over_the_mesh() -> None:
    from local_operator.network.credentials.types import is_device_bound_provider

    assert is_device_bound_provider("bedrock")
    assert is_device_bound_provider("kimi")
    assert not is_device_bound_provider("anthropic")


def test_catalogue_prices_and_effort_on_prefixed_ids() -> None:
    from local_operator.model.effort import supported_efforts
    from local_operator.model.registry import bedrock_models, get_model_info

    opus_global = get_model_info("bedrock", "global.anthropic.claude-opus-5-5")
    opus_us = bedrock_models["us.anthropic.claude-opus-5-5"]
    assert (opus_global.input_price, opus_global.output_price) == (4.0, 20.0)
    assert (opus_us.input_price, opus_us.output_price, opus_us.cache_reads_price) == (
        4.4,
        22.0,
        0.22,
    )
    assert bedrock_models["global.anthropic.claude-sonnet-5-5"].cache_reads_price == 0.10
    assert opus_global.context_window == 1_000_000
    assert supported_efforts("us.anthropic.claude-opus-5-5")


def test_family_resolver_describes_discovered_unshipped_ids() -> None:
    from local_operator.model.configure import _FAMILY_MODEL_RESOLVERS

    info = _FAMILY_MODEL_RESOLVERS["bedrock"]("us.anthropic.claude-haiku-5-5")
    assert info is not None and info.input_price == 0.0 and (info.context_window or 0) > 0


def test_importing_registry_and_clients_leaves_botocore_unloaded() -> None:
    env = {k: v for k, v in os.environ.items() if not k.startswith("CMUX_")}
    code = (
        "import sys, local_operator.providers.clients, local_operator.providers.registry;"
        "from local_operator.providers.registry import get_provider_definition;"
        "get_provider_definition('bedrock').env_keys();"
        "print('botocore' in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=120
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip().splitlines()[-1] == "False"


def test_same_access_key_in_two_regions_keeps_both_rows_and_refs(tmp_path: Path) -> None:
    """R3/Q2: Region is part of the access-key identity, like the profile branch."""
    store = AuthStore(tmp_path / "auth.db", config_dir=tmp_path)
    try:
        refs = []
        for region in ("us-east-1", "eu-west-1"):
            row = store.upsert_credential(
                "bedrock",
                {
                    "type": "api_key",
                    "source": "login",
                    # Placeholder ref (replaced by the seal) keeps this test about
                    # identity only, independent of the Q6 ref-minting change.
                    "key": f"bedrock://access-key?region={region}&ref=BEDROCK_SECRET_{'0' * 32}",
                    "account_id": "access-key:" + "AKIDFAKE",  # no Region in account_id
                    "secret": {"access_key_id": "AKIDFAKE", "secret_access_key": PAIR_SECRET},
                },
            )
            refs.append(bconn.parse(row.data["key"]).ref)
        rows = store.list_credentials("bedrock", include_disabled=True)
        regions = [bconn.parse(r.data["key"]).region for r in rows]
        # ``region`` is Optional on the parsed descriptor; every row here was
        # written with one, so a None is itself a failure, not a value to skip.
        assert None not in regions
        assert sorted(r for r in regions if r is not None) == ["eu-west-1", "us-east-1"]
        names = _secret_names(tmp_path)
        assert all(any(ref in name for name in names) for ref in refs)
    finally:
        store.close()


def test_seal_mints_the_ref_for_an_unsealed_access_key_descriptor(tmp_path: Path) -> None:
    """Q6: no placeholder ref needed; the hook mints a fresh BEDROCK_SECRET_ ref."""
    store = AuthStore(tmp_path / "auth.db", config_dir=tmp_path)
    try:
        row = store.upsert_credential(
            "bedrock",
            {
                "type": "api_key",
                "key": "bedrock://access-key?region=us-east-2",
                "account_id": ACCOUNT,
                "secret": {"access_key_id": "AKIDFAKE", "secret_access_key": PAIR_SECRET},
            },
        )
        conn = bconn.parse(row.data["key"])
        assert conn.region == "us-east-2" and conn.ref and bconn.REF_PATTERN.fullmatch(conn.ref)
        # Persisted/resolver paths still refuse a sealed descriptor without a ref.
        import pytest

        with pytest.raises(bconn.DescriptorError):
            bconn.parse("bedrock://access-key?region=us-east-2")
        with pytest.raises(bconn.DescriptorError):
            bconn.parse("bedrock://access-key?region=us-east-2&ref=OTHER", require_ref=False)
    finally:
        store.close()


def test_bedrock_rows_name_their_route_and_leave_first_party_names_resolving() -> None:
    """Merge gate 44521486072: copying first-party names onto the Bedrock rows
    made ``Claude Opus 5`` ambiguous, so ``model/naming.py`` refused it and the
    status band, pinned-fallback notice and picker showed raw selectors (and the
    picker showed indistinguishable duplicate rows). Each Bedrock row now names
    its route, the way multi-route rows already do (``GLM-5.2 (Token Plan)``)."""
    from local_operator.model.naming import model_label
    from local_operator.model.registry import bedrock_models

    assert bedrock_models["global.anthropic.claude-opus-5-5"].name == "Claude Opus 5.5 (Bedrock)"
    assert bedrock_models["eu.anthropic.claude-opus-5-5"].name == "Claude Opus 5.5 (Bedrock EU)"
    assert bedrock_models["anthropic.claude-opus-5-5"].name == "Claude Opus 5.5 (Bedrock in-Region)"
    names = [row.name for row in bedrock_models.values()]
    assert len(names) == len(set(names))
    assert model_label("anthropic", "claude-opus-5").full == "Claude Opus 5"
    assert model_label("bedrock", "us.anthropic.claude-opus-5").full == "Claude Opus 5 (Bedrock US)"
