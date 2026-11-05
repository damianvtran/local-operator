"""Account provenance: the per-call ``account_ref`` stamp and ``account.json``.

The ref is the one thing a ledger or a terminal host may use to attribute usage
to an account, so these pin that it is stable, opaque and absent where nothing
durable identifies the account, and that the host file follows every change that
moves it and never carries a secret or an identity label.
"""

from __future__ import annotations

import json
import os
import stat
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from local_operator.config import ConfigManager
from local_operator.harness.types import ModelSpec, StreamUsageEvent, Usage
from local_operator.providers.account_provenance import (
    ACCOUNT_FILE_NAME,
    account_ref,
    refresh_account_file,
)
from local_operator.providers.auth_store import AuthStore
from local_operator.providers.failover import _stamp_serving_spec
from local_operator.session.frontend_state import FrontendUsage, _aggregate_usage

EMAIL = "someone@example.invalid"
ORG_NAME = "Example Org Name"
SECRETS = ("synthetic-access-token", "synthetic-refresh-token", "sk-synthetic-key")


def _oauth(org_id: str = "org-1", **extra: object) -> dict[str, object]:
    return {
        "type": "oauth",
        "access": SECRETS[0],
        "refresh": SECRETS[1],
        "expires": 4_000_000_000_000,
        "email": EMAIL,
        "org_id": org_id,
        "org_name": ORG_NAME,
        **extra,
    }


@pytest.fixture()
def config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "cfg"
    root.mkdir()
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    return root


@pytest.fixture()
def store(config: Path) -> Iterator[AuthStore]:
    auth = AuthStore(db_path=config / "auth.db")
    yield auth
    auth.close()


def _file(config: Path) -> dict[str, Any]:
    return json.loads((config / ACCOUNT_FILE_NAME).read_text())


# -- the ref --------------------------------------------------------------


def test_oauth_ref_is_stable_across_rows_and_opaque() -> None:
    first = account_ref("anthropic", kind="oauth", credential_id=3, org_id="org-1")
    relogin = account_ref("anthropic", kind="oauth", credential_id=9, org_id="org-1")
    other = account_ref("anthropic", kind="oauth", credential_id=3, org_id="org-2")
    assert first == relogin != other
    assert first is not None and len(first) == 16 and int(first, 16) >= 0
    assert "org-1" not in first
    # Falls back to the account id when the IdP gave no org.
    assert account_ref("openai", kind="oauth", credential_id=1, account_id="acct") == account_ref(
        "openai", kind="oauth", credential_id=2, account_id="acct"
    )


def test_ref_depends_on_provider_and_login_flavour_is_canonicalised() -> None:
    assert account_ref("anthropic", kind="oauth", credential_id=1, org_id="x") != account_ref(
        "openai", kind="oauth", credential_id=1, org_id="x"
    )
    assert account_ref("openai-device", kind="oauth", credential_id=1, org_id="x") == account_ref(
        "openai", kind="oauth", credential_id=1, org_id="x"
    )


def test_api_key_ref_is_per_row_and_ignores_identity_fields() -> None:
    one = account_ref("openai", kind="api_key", credential_id=4)
    assert one == account_ref("openai", kind="api_key", credential_id=4, org_id="ignored")
    assert one != account_ref("openai", kind="api_key", credential_id=5)
    assert one != account_ref("openai", kind="oauth", credential_id=4)


def test_env_credentials_have_no_ref() -> None:
    assert account_ref("anthropic", kind="api_key", credential_id=0) is None
    assert account_ref("anthropic", kind="oauth", credential_id=0, org_id="org-1") is None


def test_oauth_without_idp_identity_is_keyed_on_its_single_row() -> None:
    ref = account_ref("kimi", kind="oauth", credential_id=7)
    assert ref is not None
    assert ref != account_ref("kimi", kind="api_key", credential_id=7)


# -- the per-call stamp ---------------------------------------------------


def _stamp(access: object, key: str | None = None) -> Usage:
    spec = ModelSpec(provider="anthropic", model_id="claude-test")
    out = _stamp_serving_spec(
        StreamUsageEvent(usage=Usage(usd_cost=1)), spec, access=access, key=key
    )
    assert isinstance(out, StreamUsageEvent)
    return out.usage


def test_stamp_carries_ref_for_oauth_and_key_rows_not_env() -> None:
    oauth = _stamp(SimpleNamespace(kind="oauth", credential_id=3, org_id="org-1", account_id=None))
    assert oauth.account_ref == account_ref(
        "anthropic", kind="oauth", credential_id=3, org_id="org-1"
    )
    keyed = _stamp(SimpleNamespace(kind="api_key", credential_id=5))
    assert keyed.account_ref == account_ref("anthropic", kind="api_key", credential_id=5)
    assert (
        _stamp(SimpleNamespace(kind="api_key", credential_id=0), key="sk-env").account_ref is None
    )
    assert _stamp(None).account_ref is None


def test_stamp_reads_identity_from_the_raw_row_when_flat_fields_are_absent() -> None:
    usage = _stamp(
        SimpleNamespace(kind="oauth", credential_id=3, raw={"org_id": "org-1"}, org_id=None)
    )
    assert usage.account_ref == account_ref(
        "anthropic", kind="oauth", credential_id=3, org_id="org-1"
    )


def test_stamp_failure_never_drops_the_event(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_a: object, **_k: object) -> None:
        raise RuntimeError("synthetic")

    monkeypatch.setattr("local_operator.providers.account_provenance.account_ref", boom)
    event = StreamUsageEvent(usage=Usage(usd_cost=1))
    out = _stamp_serving_spec(event, ModelSpec(provider="anthropic", model_id="m"), access=None)
    assert out is event


# -- serializers and aggregates ------------------------------------------


def test_unset_account_ref_is_omitted_from_every_usage_serializer() -> None:
    from local_operator.session.frontend_state import _FrozenFrontendUsage, _FrozenUsage

    for cls in (Usage, FrontendUsage, _FrozenUsage, _FrozenFrontendUsage):
        assert "account_ref" not in cls().model_dump()
        assert cls(account_ref="abc").model_dump()["account_ref"] == "abc"


def test_aggregate_keeps_a_single_account_and_drops_a_mixed_one() -> None:
    same = _aggregate_usage([Usage(account_ref="a"), Usage(account_ref="a")])
    assert same.account_ref == "a"
    assert _aggregate_usage([Usage(account_ref="a"), Usage(account_ref="b")]).account_ref is None
    assert _aggregate_usage([Usage(account_ref="a"), Usage()]).account_ref is None


def test_job_fold_drops_account_when_calls_from_two_accounts_merge() -> None:
    from local_operator.harness.jobs import _merge_accounting_component

    grouped: dict[Any, Usage] = {}
    for ref in ("a", "a", "b"):
        _merge_accounting_component(
            grouped, Usage(provider="anthropic", model_id="m", usd_cost=1.0, account_ref=ref)
        )
    assert len(grouped) == 1
    assert next(iter(grouped.values())).account_ref is None


def test_wire_aggregate_names_one_account_only_when_all_calls_agree() -> None:
    from local_operator.harness.wire import _usage_aggregate as fold

    one = fold([{"usage": {"account_ref": "a"}}, {"usage": {"account_ref": "a"}}])
    assert one is not None and one["account_ref"] == "a"
    mixed = fold([{"usage": {"account_ref": "a"}}, {"usage": {"account_ref": "b"}}])
    assert mixed is not None and "account_ref" not in mixed


# -- account.json ---------------------------------------------------------


def test_file_follows_login_and_has_no_secrets_or_labels(config: Path, store: AuthStore) -> None:
    ConfigManager(config).set_config_value("hosting", "anthropic")
    row = store.upsert_credential("anthropic", _oauth())
    document = _file(config)
    expected = account_ref("anthropic", kind="oauth", credential_id=row.id, org_id="org-1")
    assert document["version"] == 1
    assert document["default"] == {
        "provider": "anthropic",
        "account_ref": expected,
        "kind": "oauth",
        "org_type": None,
    }
    assert document["providers"] == {
        "anthropic": {"account_ref": expected, "kind": "oauth", "org_type": None}
    }
    text = (config / ACCOUNT_FILE_NAME).read_text()
    for forbidden in (EMAIL, ORG_NAME, "org-1", *SECRETS):
        assert forbidden not in text
    assert stat.S_IMODE(os.stat(config / ACCOUNT_FILE_NAME).st_mode) == 0o600


def test_file_follows_logout(config: Path, store: AuthStore) -> None:
    ConfigManager(config).set_config_value("hosting", "anthropic")
    store.upsert_credential("anthropic", _oauth())
    assert _file(config)["default"] is not None
    store.delete_credentials_for_provider("anthropic")
    document = _file(config)
    assert document["default"] is None and document["providers"] == {}


def test_file_follows_disable(config: Path, store: AuthStore) -> None:
    ConfigManager(config).set_config_value("hosting", "anthropic")
    first = store.upsert_credential("anthropic", _oauth("org-1"))
    second = store.upsert_credential("anthropic", _oauth("org-2"))
    assert first.id != second.id
    assert _file(config)["default"]["account_ref"] == account_ref(
        "anthropic", kind="oauth", credential_id=first.id, org_id="org-1"
    )
    store.disable_credential(first.id, "operator")
    assert _file(config)["default"]["account_ref"] == account_ref(
        "anthropic", kind="oauth", credential_id=second.id, org_id="org-2"
    )


def test_file_follows_a_default_model_change(config: Path, store: AuthStore) -> None:
    manager = ConfigManager(config)
    manager.set_config_value("hosting", "anthropic")
    store.upsert_credential("anthropic", _oauth())
    store.upsert_credential("openai", {"type": "api_key", "key": SECRETS[2]})
    assert _file(config)["default"]["provider"] == "anthropic"
    manager.set_config_value("hosting", "openai")
    document = _file(config)
    assert document["default"]["provider"] == "openai"
    assert document["default"]["kind"] == "api_key"
    assert set(document["providers"]) == {"anthropic", "openai"}


def test_org_type_comes_only_from_a_stored_openai_plan_slug(config: Path, store: AuthStore) -> None:
    ConfigManager(config).set_config_value("hosting", "openai")
    store.upsert_credential("openai", _oauth("acct-1", org_name="plus"))
    assert _file(config)["default"]["org_type"] == "plus"
    store.delete_credentials_for_provider("openai")
    store.upsert_credential("openai", _oauth("acct-1", org_name="Some Organisation"))
    assert _file(config)["default"]["org_type"] is None


def test_startup_refresh_writes_when_missing_and_is_idempotent(
    config: Path, store: AuthStore
) -> None:
    ConfigManager(config).set_config_value("hosting", "anthropic")
    store.upsert_credential("anthropic", _oauth())
    (config / ACCOUNT_FILE_NAME).unlink()
    assert refresh_account_file(config) is True
    assert refresh_account_file(config) is False
    (config / ACCOUNT_FILE_NAME).write_text("{}")
    assert refresh_account_file(config) is True
    assert _file(config)["version"] == 1


def test_startup_refresh_creates_nothing_in_a_bare_config_dir(tmp_path: Path) -> None:
    bare = tmp_path / "bare"
    assert refresh_account_file(bare) is False
    assert not bare.exists()


def test_read_only_store_writes_no_file(config: Path, store: AuthStore) -> None:
    store.upsert_credential("anthropic", _oauth())
    (config / ACCOUNT_FILE_NAME).unlink()
    reader = AuthStore.open_read_only(config / "auth.db")
    try:
        reader._publish_account_file()
    finally:
        reader.close()
    assert not (config / ACCOUNT_FILE_NAME).exists()
