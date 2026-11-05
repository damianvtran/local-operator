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
    one = account_ref("openai", kind="api_key", credential_id=4, store_uid="s1")
    assert one == account_ref(
        "openai", kind="api_key", credential_id=4, store_uid="s1", org_id="ignored"
    )
    assert one != account_ref("openai", kind="api_key", credential_id=5, store_uid="s1")
    assert one != account_ref("openai", kind="oauth", credential_id=4, store_uid="s1")


def test_row_derived_ref_is_namespaced_by_store_and_guarded_by_created_at() -> None:
    def ref(store_uid: str = "store-a", created_at: int = 10) -> str | None:
        return account_ref(
            "openai", kind="api_key", credential_id=1, store_uid=store_uid, created_at=created_at
        )

    assert ref() != ref(store_uid="store-b")
    assert ref() != ref(created_at=11)


def test_row_derived_ref_needs_a_store_uid_and_a_positive_row_id() -> None:
    assert account_ref("openai", kind="api_key", credential_id=1) is None
    assert account_ref("openai", kind="api_key", credential_id=1, store_uid="") is None
    assert account_ref("kimi", kind="oauth", credential_id=1) is None
    # A mesh borrower's synthetic (negative) id must never become a ref.
    assert account_ref("openai", kind="api_key", credential_id=-9, store_uid="s") is None
    assert account_ref("kimi", kind="oauth", credential_id=-9, store_uid="s") is None
    # ...but an OAuth identity is global, so owner and borrower agree on it.
    assert account_ref("openai", kind="oauth", credential_id=-9, org_id="o") == account_ref(
        "openai", kind="oauth", credential_id=4, org_id="o"
    )


def test_env_credentials_have_no_ref() -> None:
    assert account_ref("anthropic", kind="api_key", credential_id=0) is None
    assert account_ref("anthropic", kind="oauth", credential_id=0, org_id="org-1") is None


def test_oauth_without_idp_identity_is_keyed_on_its_single_row() -> None:
    ref = account_ref("kimi", kind="oauth", credential_id=7, store_uid="s")
    assert ref is not None
    assert ref != account_ref("kimi", kind="api_key", credential_id=7, store_uid="s")


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
    keyed = _stamp(SimpleNamespace(kind="api_key", credential_id=5, account_ref="0123456789abcdef"))
    assert keyed.account_ref == "0123456789abcdef"
    # No store-minted ref on the access: a key has no global identity to recompute.
    assert _stamp(SimpleNamespace(kind="api_key", credential_id=5)).account_ref is None
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
    inventory = [{"account_ref": expected, "kind": "oauth", "org_type": None}]
    assert document["default"] == {
        "provider": "anthropic",
        "account_ref": expected,
        "kind": "oauth",
        "org_type": None,
        "accounts": inventory,
    }
    assert document["providers"] == {
        "anthropic": {
            "account_ref": expected,
            "kind": "oauth",
            "org_type": None,
            "accounts": inventory,
        }
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
    ref_1 = account_ref("anthropic", kind="oauth", credential_id=first.id, org_id="org-1")
    ref_2 = account_ref("anthropic", kind="oauth", credential_id=second.id, org_id="org-2")
    # Two enabled accounts: which serves is a per-session choice, so the file says unknown.
    both = _file(config)["default"]
    assert both["account_ref"] is None
    assert [a["account_ref"] for a in both["accounts"]] == [ref_1, ref_2]
    store.disable_credential(first.id, "operator")
    assert _file(config)["default"]["account_ref"] == ref_2


def test_file_follows_a_default_model_change(config: Path, store: AuthStore) -> None:
    manager = ConfigManager(config)
    manager.set_config_value("hosting", "anthropic")
    store.upsert_credential("anthropic", _oauth())
    store.upsert_credential("openai", {"type": "api_key", "key": SECRETS[2]})
    assert _file(config)["default"]["provider"] == "anthropic"
    manager.set_config_value("hosting", "openai")
    document = _file(config)
    assert document["default"]["provider"] == "openai"
    # A lone stored key without ``source="login"`` is outranked by an env var: unknown.
    assert document["default"]["kind"] is None and document["default"]["account_ref"] is None
    assert len(document["default"]["accounts"]) == 1
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


# -- store namespacing ----------------------------------------------------


def _row_ref(auth: AuthStore, credential_id: int) -> str | None:
    row = auth.get_credential(credential_id)
    assert row is not None
    return auth._account_ref_for_row(row)  # noqa: SLF001


@pytest.fixture()
def other_store(tmp_path: Path) -> Iterator[AuthStore]:
    other = AuthStore(db_path=tmp_path / "other" / "auth.db")
    yield other
    other.close()


def test_independent_stores_give_different_refs_for_the_same_row_id(
    store: AuthStore, other_store: AuthStore
) -> None:
    other = other_store
    assert store.store_uid and other.store_uid and store.store_uid != other.store_uid
    first = store.upsert_credential("openai", {"type": "api_key", "key": "sk-a"})
    second = other.upsert_credential("openai", {"type": "api_key", "key": "sk-b"})
    assert first.id == second.id == 1
    assert _row_ref(store, first.id) != _row_ref(other, second.id)
    # Identityless OAuth is row-derived too.
    kimi_a = store.upsert_credential("kimi", {"type": "oauth", "access": "a", "expires": 1})
    kimi_b = other.upsert_credential("kimi", {"type": "oauth", "access": "b", "expires": 1})
    assert _row_ref(store, kimi_a.id) != _row_ref(other, kimi_b.id)


def test_store_uid_is_stable_across_opens_and_read_only_never_creates_it(
    config: Path, store: AuthStore
) -> None:
    uid = store.store_uid
    store.close()
    again = AuthStore(db_path=config / "auth.db")
    try:
        assert again.store_uid == uid
    finally:
        again.close()
    import sqlite3

    legacy = config / "legacy.db"
    conn = sqlite3.connect(legacy)
    conn.execute("CREATE TABLE auth_credentials (id INTEGER PRIMARY KEY)")
    conn.commit()
    conn.close()
    reader = AuthStore.open_read_only(legacy)
    try:
        assert reader.store_uid is None
    finally:
        reader.close()
    conn = sqlite3.connect(legacy)
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    conn.close()
    assert "auth_store_meta" not in tables


def test_relogin_of_identityless_oauth_keeps_the_ref(config: Path, store: AuthStore) -> None:
    first = store.upsert_credential("kimi", {"type": "oauth", "access": "a", "expires": 1})
    ref = _row_ref(store, first.id)
    assert ref is not None
    again = store.upsert_credential("kimi", {"type": "oauth", "access": "b", "expires": 2})
    assert again.id == first.id
    assert _row_ref(store, again.id) == ref


def test_oauth_access_carries_the_stores_ref(config: Path, store: AuthStore) -> None:
    import asyncio

    row = store.upsert_credential("openai", {"type": "api_key", "key": "sk-a"})
    access = asyncio.run(store.get_oauth_access("openai"))
    assert access is not None and access.account_ref == _row_ref(store, row.id)
    assert access.account_ref is not None


# -- serialized publication -----------------------------------------------


def test_a_stale_publisher_cannot_resurrect_a_logged_out_account(
    config: Path, store: AuthStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    import threading

    import local_operator.config as config_module

    ConfigManager(config).set_config_value("hosting", "anthropic")
    store.upsert_credential("anthropic", _oauth())
    assert _file(config)["default"] is not None

    real_read = config_module.read_config_values
    rows_read = threading.Event()
    release = threading.Event()

    def gated(config_dir: Path) -> Any:
        if threading.current_thread().name == "stale-writer":
            # The stale publisher has already read the (still signed-in) rows.
            rows_read.set()
            assert release.wait(10)
        return real_read(config_dir)

    monkeypatch.setattr(config_module, "read_config_values", gated)
    (config / ACCOUNT_FILE_NAME).unlink()

    stale = threading.Thread(
        target=lambda: refresh_account_file(config), name="stale-writer", daemon=True
    )
    stale.start()
    assert rows_read.wait(10)

    logout_done = threading.Event()

    def logout() -> None:
        store.delete_credentials_for_provider("anthropic")
        logout_done.set()

    logout_thread = threading.Thread(target=logout, name="logout", daemon=True)
    logout_thread.start()
    # The logout's own publish queues behind the paused publisher's lock.
    assert not logout_done.wait(0.5)
    release.set()
    stale.join(10)
    logout_thread.join(10)
    assert not stale.is_alive() and not logout_thread.is_alive()
    document = _file(config)
    assert document["providers"] == {} and document["default"] is None


def test_deleting_one_credential_is_also_serialized_behind_a_stale_publisher(
    config: Path, store: AuthStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    import threading

    import local_operator.config as config_module

    ConfigManager(config).set_config_value("hosting", "openai")
    row = store.upsert_credential("openai", {"type": "api_key", "key": SECRETS[2]})
    real_read = config_module.read_config_values
    rows_read, release = threading.Event(), threading.Event()

    def gated(config_dir: Path) -> Any:
        if threading.current_thread().name == "stale-writer":
            rows_read.set()
            assert release.wait(10)
        return real_read(config_dir)

    monkeypatch.setattr(config_module, "read_config_values", gated)
    (config / ACCOUNT_FILE_NAME).unlink()
    stale = threading.Thread(
        target=lambda: refresh_account_file(config), name="stale-writer", daemon=True
    )
    stale.start()
    assert rows_read.wait(10)
    done = threading.Event()

    def delete() -> None:
        store.delete_credential(row.id)
        done.set()

    deleter = threading.Thread(target=delete, daemon=True)
    deleter.start()
    assert not done.wait(0.5)
    release.set()
    stale.join(10)
    deleter.join(10)
    document = _file(config)
    assert document["providers"] == {} and document["default"] is None


def test_the_publisher_takes_no_caller_supplied_rows(config: Path, store: AuthStore) -> None:
    import inspect

    assert "rows" not in inspect.signature(refresh_account_file).parameters
    with pytest.raises(TypeError):
        refresh_account_file(config, [])  # type: ignore[call-arg, misc]


def test_publication_leaves_a_private_lock_beside_the_file(config: Path, store: AuthStore) -> None:
    store.upsert_credential("anthropic", _oauth())
    lock = config / ".account-file.lock"
    assert lock.is_file() and stat.S_IMODE(os.stat(lock).st_mode) == 0o600


def test_a_symlinked_lock_is_refused_and_nothing_is_written(config: Path, store: AuthStore) -> None:
    store.upsert_credential("anthropic", _oauth())
    (config / ACCOUNT_FILE_NAME).unlink()
    (config / ".account-file.lock").unlink()
    elsewhere = config / "elsewhere"
    elsewhere.write_text("")
    (config / ".account-file.lock").symlink_to(elsewhere)
    assert refresh_account_file(config) is False
    assert not (config / ACCOUNT_FILE_NAME).exists()


def test_a_held_lock_times_out_without_raising(
    config: Path, store: AuthStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    from local_operator.config_writer import nonblocking_file_claim
    from local_operator.providers import account_provenance

    monkeypatch.setattr(account_provenance, "_LOCK_WAIT_S", 0.05)
    store.upsert_credential("anthropic", _oauth())
    (config / ACCOUNT_FILE_NAME).unlink()
    with nonblocking_file_claim(config / ".account-file.lock") as held:
        assert held
        assert refresh_account_file(config) is False
    assert not (config / ACCOUNT_FILE_NAME).exists()
    assert refresh_account_file(config) is True


def test_a_bare_config_dir_gets_no_lock_file(tmp_path: Path) -> None:
    bare = tmp_path / "bare"
    bare.mkdir()
    assert refresh_account_file(bare) is False
    assert list(bare.iterdir()) == []


# -- the published account follows the cascade ----------------------------


def _providers(config: Path) -> dict[str, Any]:
    return _file(config)["providers"]


def test_login_key_beats_an_older_plain_key_like_the_real_cascade(
    config: Path, store: AuthStore
) -> None:
    import asyncio

    ConfigManager(config).set_config_value("hosting", "openai")
    store.upsert_credential("openai", {"type": "api_key", "key": "sk-old"})
    newer = store.upsert_credential(
        "openai", {"type": "api_key", "key": "sk-new", "source": "login"}
    )
    access = asyncio.run(store.get_oauth_access("openai"))
    assert access is not None and access.credential_id == newer.id
    default = _file(config)["default"]
    assert default["account_ref"] is not None
    assert default["account_ref"] == access.account_ref
    # The older plain key stays in the inventory, after the serving one.
    assert len(default["accounts"]) == 2
    assert default["accounts"][0]["account_ref"] == access.account_ref


def test_two_oauth_rows_are_undeterminable_but_both_inventoried(
    config: Path, store: AuthStore
) -> None:
    store.upsert_credential("anthropic", _oauth("org-1"))
    store.upsert_credential("anthropic", _oauth("org-2"))
    entry = _providers(config)["anthropic"]
    assert entry["account_ref"] is None and entry["kind"] is None and entry["org_type"] is None
    assert [a["account_ref"] for a in entry["accounts"]] == [
        account_ref("anthropic", kind="oauth", credential_id=1, org_id="org-1"),
        account_ref("anthropic", kind="oauth", credential_id=2, org_id="org-2"),
    ]


def test_only_a_tier_six_key_is_unknown_but_listed(config: Path, store: AuthStore) -> None:
    ConfigManager(config).set_config_value("hosting", "openai")
    store.upsert_credential("openai", {"type": "api_key", "key": "sk-plain"})
    entry = _providers(config)["openai"]
    assert entry["account_ref"] is None and entry["kind"] is None
    assert len(entry["accounts"]) == 1 and entry["accounts"][0]["kind"] == "api_key"
    assert _file(config)["default"]["account_ref"] is None


def test_oauth_outranks_a_login_key(config: Path, store: AuthStore) -> None:
    store.upsert_credential("openai", {"type": "api_key", "key": "sk-l", "source": "login"})
    store.upsert_credential("openai", _oauth("acct-1"))
    entry = _providers(config)["openai"]
    assert entry["kind"] == "oauth"
    assert entry["account_ref"] == account_ref(
        "openai", kind="oauth", credential_id=2, org_id="acct-1"
    )
    assert [a["kind"] for a in entry["accounts"]] == ["oauth", "api_key"]


def test_disabled_rows_are_excluded_from_entry_and_inventory(
    config: Path, store: AuthStore
) -> None:
    first = store.upsert_credential("anthropic", _oauth("org-1"))
    store.upsert_credential("anthropic", _oauth("org-2"))
    store.disable_credential(first.id, "operator")
    entry = _providers(config)["anthropic"]
    assert entry["account_ref"] == account_ref(
        "anthropic", kind="oauth", credential_id=2, org_id="org-2"
    )
    assert len(entry["accounts"]) == 1
