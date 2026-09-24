"""Registry tests: legacy hosting resolution, field semantics, env keys."""

from __future__ import annotations

from typing import Any

import pytest

from local_operator.providers.registry import (
    PROVIDER_REGISTRY,
    credential_file_names,
    env_key_name,
    get_provider_definition,
    list_login_providers,
    resolve_env_key,
)

LEGACY_HOSTING_NAMES = [
    "radient",
    "openai",
    "anthropic",
    "google",
    "mistral",
    "ollama",
    "openrouter",
    "deepseek",
    "kimi",
    "alibaba",
    "xai",
    "zai",
]


@pytest.mark.parametrize("hosting", LEGACY_HOSTING_NAMES + ["test", "noop"])
def test_every_legacy_hosting_name_resolves(hosting: str) -> None:
    definition = get_provider_definition(hosting)
    assert definition is not None, f"legacy --hosting name must resolve: {hosting}"


def test_unknown_provider_returns_none() -> None:
    assert get_provider_definition("definitely-not-a-provider") is None


def test_zai_oauth_shares_the_zai_credential_row_and_catalogue() -> None:
    """The sign-in mints a durable key for the SAME provider, exactly as
    ``xai-oauth`` does -- not a second catalogue."""
    zai_oauth = get_provider_definition("zai-oauth")
    zai = get_provider_definition("zai")
    assert zai_oauth is not None and zai is not None
    assert zai_oauth.store_credentials_as == "zai"
    assert zai_oauth.base_url == zai.base_url
    assert zai_oauth.login is not None
    # No refresh: the minted key never expires, so there is nothing to refresh.
    assert zai_oauth.refresh_token is None


def test_registry_ids_unique() -> None:
    ids = [p.id for p in PROVIDER_REGISTRY]
    assert len(ids) == len(set(ids))


def test_openai_definition_oauth_fields() -> None:
    definition = get_provider_definition("openai")
    assert definition is not None
    assert definition.login is not None
    assert definition.refresh_token is not None
    assert definition.callback_port == 1455
    assert definition.paste_code_flow is False
    assert definition.wire == "openai-compat"
    # The device variant aliases into the same credential row.
    device = get_provider_definition("openai-device")
    assert device is not None
    assert device.store_credentials_as == "openai"


def test_zai_definition() -> None:
    """Z.AI is an API-key provider on the CODING-plan base URL.

    The base URL is asserted because the general `/api/paas/v4` endpoint accepts
    the same key but bills the account balance instead of coding-plan quota — a
    silent wrong-budget bug rather than a visible failure.
    """
    definition = get_provider_definition("zai")
    assert definition is not None
    assert definition.login is not None
    assert definition.env_keys == "ZAI_API_KEY"
    assert definition.wire == "openai-compat"
    assert definition.base_url == "https://api.z.ai/api/coding/paas/v4"
    # Search vocabulary only \u2014 nothing ROUTES on these (that is what the
    # registry's own docstring promises), but the picker must offer the name
    # users came here for, which is the model family rather than the company.
    assert set(definition.search_aliases) == {"glm", "zhipu", "bigmodel", "z-ai"}


def test_anthropic_definition() -> None:
    definition = get_provider_definition("anthropic")
    assert definition is not None
    assert definition.callback_port == 54545
    assert definition.paste_code_flow is True
    assert definition.wire == "anthropic"
    assert definition.base_url == "https://api.anthropic.com"


def test_kimi_definition() -> None:
    definition = get_provider_definition("kimi")
    assert definition is not None
    assert definition.base_url == "https://api.moonshot.cn/v1"
    assert definition.login is not None  # RFC 8628 device code
    assert definition.callback_port is None  # no loopback server


def test_xai_pair() -> None:
    key_provider = get_provider_definition("xai")
    oauth_provider = get_provider_definition("xai-oauth")
    assert key_provider is not None and oauth_provider is not None
    assert oauth_provider.store_credentials_as == "xai"
    assert key_provider.login is not None  # paste-key login
    assert key_provider.env_keys == "XAI_API_KEY"


def test_ollama_allows_missing_api_key() -> None:
    definition = get_provider_definition("ollama")
    assert definition is not None
    assert definition.allows_missing_api_key is True
    assert definition.base_url == "http://localhost:11434/v1"


def test_test_provider_is_mock_wire() -> None:
    definition = get_provider_definition("test")
    assert definition is not None
    assert definition.wire == "mock"


def test_list_login_providers_includes_local_setup_but_not_test_host() -> None:
    ids = {p.id for p in list_login_providers()}
    assert {"openai", "anthropic", "kimi", "xai", "xai-oauth"} <= ids
    assert {"ollama", "lmstudio", "vllm", "llamacpp", "openai-compatible"} <= ids
    assert "test" not in ids


def test_env_key_resolution_plain_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deepseek-test")
    assert resolve_env_key("deepseek") == "sk-deepseek-test"
    assert env_key_name("deepseek") == "DEEPSEEK_API_KEY"


def test_env_key_resolution_callable_form(monkeypatch: pytest.MonkeyPatch) -> None:
    """Anthropic's callable resolver prefers the OAuth token var."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "raw-key")
    monkeypatch.delenv("ANTHROPIC_OAUTH_TOKEN", raising=False)
    assert resolve_env_key("anthropic") == "raw-key"
    monkeypatch.setenv("ANTHROPIC_OAUTH_TOKEN", "oauth-token")
    assert resolve_env_key("anthropic") == "oauth-token"
    # Callable resolvers have no single display name.
    assert env_key_name("anthropic") is None


def test_credential_file_names_covers_both_env_keys_forms() -> None:
    """The legacy credential file's key names, for BOTH ``env_keys`` forms.

    ``env_key_name`` answers only the plain-string form, so a reader built on it
    alone silently drops ``anthropic`` — the sole callable-form provider, and the
    one whose key a user is most likely to have set with ``lop credential
    update``. That is not hypothetical: it is how the phone's model sheet came
    back empty on an install the desktop listed 18 models for.
    """
    assert credential_file_names("deepseek") == ["DEEPSEEK_API_KEY"]
    # The callable form, which `env_key_name` cannot answer.
    assert env_key_name("anthropic") is None
    assert "ANTHROPIC_API_KEY" in credential_file_names("anthropic")
    # Alias-aware: a login flavour holds its key under the provider it stores as.
    assert "XAI_API_KEY" in credential_file_names("xai-oauth")
    # A provider with no key name at all, and an unknown id, are both empty
    # rather than raising — callers iterate over the result unconditionally.
    assert credential_file_names("ollama") == []
    assert credential_file_names("no-such-provider") == []


def test_every_registry_provider_with_a_string_key_resolves_a_file_name() -> None:
    """No provider that declares a key name may resolve to nothing.

    The class-level guard behind the ``anthropic`` defect: a provider that
    declares how it is configured but resolves no file name is invisible to the
    legacy credential rung.
    """
    for definition in PROVIDER_REGISTRY:
        if definition.env_keys is None:
            continue
        assert credential_file_names(definition.id), definition.id


def test_env_key_resolution_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    assert resolve_env_key("mistral") is None
    assert resolve_env_key("unknown-provider") is None


def test_alibaba_token_plan_is_separate_from_payg_dashscope() -> None:
    """The Token Plan is a distinct product from DashScope pay-as-you-go:
    its own region-locked endpoint, its own env key, findable by the names a
    user would actually type."""
    from local_operator.providers.registry import get_provider_definition

    definition = get_provider_definition("alibaba-token-plan")
    assert definition is not None
    assert (
        definition.base_url
        == "https://token-plan.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1"
    )
    assert definition.env_keys == "ALIBABA_TOKEN_PLAN_API_KEY"
    # Aliases are picker search vocabulary, not id lookups.
    assert "tokenplan" in definition.search_aliases

    dashscope = get_provider_definition("alibaba")
    assert dashscope is not None and dashscope is not definition


def test_token_plan_oauth_row_spends_the_api_key_on_the_wire() -> None:
    """The OAuth login stores two tokens with different jobs; the wire bearer
    must be the pasted sk-sp key, never the management token."""
    from local_operator.providers.registry import (
        _token_plan_wire_key,
        get_provider_definition,
    )

    assert _token_plan_wire_key({"api_key": "sk-sp-x", "access": "mgmt"}) == "sk-sp-x"
    # Hand-written rows without the embedded key still resolve to something.
    assert _token_plan_wire_key({"access": "mgmt"}) == "mgmt"

    oauth_variant = get_provider_definition("alibaba-token-plan-oauth")
    assert oauth_variant is not None
    assert oauth_variant.store_credentials_as == "alibaba-token-plan"


class TestOAuthHostSplit:
    """Providers serving OAuth and API keys from DIFFERENT hosts.

    Kimi is the case: the coding-plan OAuth grant is only accepted at
    ``api.kimi.com/coding/v1`` -- which is where ``k3`` lives -- while
    ``KIMI_API_KEY`` belongs to the mainland ``api.moonshot.cn`` platform and
    401s there. Verified live against both hosts.
    """

    def test_kimi_declares_the_coding_plan_host_for_oauth(self) -> None:
        from local_operator.providers.registry import get_provider_definition

        kimi = get_provider_definition("kimi")
        assert kimi is not None
        assert kimi.base_url == "https://api.moonshot.cn/v1"
        assert kimi.oauth_base_url == "https://api.kimi.com/coding/v1"

    def test_an_oauth_bearer_is_sent_to_the_oauth_host(self) -> None:
        """Listing the subscription's models is worthless if inference then
        sends them to the API-key host, where they 404."""
        from local_operator.providers.auth_store import OAuthAccess
        from local_operator.providers.clients import OpenAICompatClient

        client = OpenAICompatClient(
            base_url="https://api.moonshot.cn/v1",
            oauth_base_url="https://api.kimi.com/coding/v1",
        )
        oauth = OAuthAccess(access_token="tok", credential_id=1, kind="oauth")
        api_key = OAuthAccess(access_token="sk-x", credential_id=2, kind="api_key")

        assert client._request_base_url(oauth) == "https://api.kimi.com/coding/v1"
        assert client._request_base_url(api_key) == "https://api.moonshot.cn/v1"
        assert client._request_base_url(None) == "https://api.moonshot.cn/v1"

    def test_the_registry_base_on_a_spec_does_not_suppress_the_oauth_host(self) -> None:
        """`build_model_spec` copies `definition.base_url` onto EVERY spec, so a
        naive "did the spec pin a base?" test disables the OAuth host for every
        request -- which is how a live k3 call reached the API-key platform and
        401'd despite all of this being wired up. Only a base the registry did
        NOT supply counts as a deliberate override.
        """
        from local_operator.model.configure import build_model_spec
        from local_operator.providers.clients import OpenAICompatClient, client_for_spec

        def oauth_host_of(spec: Any) -> str | None:
            client = client_for_spec(spec)
            assert isinstance(client, OpenAICompatClient)
            return client._oauth_base_url

        spec = build_model_spec("kimi", "k3")
        assert spec.base_url == "https://api.moonshot.cn/v1"  # the registry's own
        assert oauth_host_of(spec) == "https://api.kimi.com/coding/v1"

        # A genuine override (a gateway) still wins and is never second-guessed.
        assert oauth_host_of(spec.model_copy(update={"base_url": "https://gw.internal/v1"})) is None

    def test_zai_oauth_shares_the_zai_credential_row_and_catalogue(self) -> None:
        """The sign-in mints a durable key for the SAME provider, exactly as
        ``xai-oauth`` does -- not a second catalogue."""
        from local_operator.providers.registry import get_provider_definition

        zai_oauth = get_provider_definition("zai-oauth")
        zai = get_provider_definition("zai")
        assert zai_oauth is not None and zai is not None
        assert zai_oauth.store_credentials_as == "zai"
        assert zai_oauth.base_url == zai.base_url
        assert zai_oauth.login is not None
        # No refresh: the minted key never expires, so there is nothing to refresh.
        assert zai_oauth.refresh_token is None


def _seed_store_file(root: Any, payload: bytes) -> Any:
    """Put a file at the secret store's path, valid or not, so the reader runs."""
    from local_operator.secrets.keys import store_path

    path = store_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return path


def _initialized_then_damaged(root: Any, payload: bytes) -> Any:
    """A REAL store, then damaged — the only shape that reaches the sqlite family.

    Order matters, and not for tidiness: an *uninitialized* store path makes
    ``open_store`` fail closed with ``SecretStoreError`` ("no secret store found"),
    which the reader has always caught — so a rig that damages a store it never
    created proves nothing. QA's 500 needed a store that HAD been written, whose
    master key is still beside it, and then had its database corrupted.
    """
    from local_operator.providers.registry import (
        store_provider_key,
        stored_provider_env_keys,
    )

    store_provider_key("OPENROUTER_API_KEY", "fixture-key", base=root)
    assert stored_provider_env_keys(root) == {"OPENROUTER_API_KEY"}, "a real store, readable"
    return _seed_store_file(root, payload)


def test_a_damaged_secret_store_reads_as_no_provider_rows(tmp_path: Any) -> None:
    """Q2-1: the reader's promise has to cover SQLITE'S exception family too.

    ``stored_provider_env_keys`` caught ``(SecretStoreError, OSError,
    ValueError)``, and a damaged store raises ``sqlite3.DatabaseError`` ("file is
    not a database") — which is not an ``OSError``. So the reader a desktop
    ``GET`` reaches through ``ProviderController.persisted_providers`` raised
    straight into the route and answered **500** for the very store state this
    function's docstring promises to survive: "an absent, locked or damaged store
    yields an EMPTY set — never an error". One fix here covers both surfaces (the
    same gap 500'd ``GET /v1/credentials``).
    """
    from local_operator.providers.registry import stored_provider_env_keys

    _initialized_then_damaged(tmp_path, b"not-a-store-at-all")
    assert stored_provider_env_keys(tmp_path) == set()


def test_a_locked_secret_store_reads_as_no_provider_rows(tmp_path: Any) -> None:
    """The other fault mode QA measured: ``chmod 000`` raises ``OperationalError``.

    A permission-denied store is the "locked" case the docstring names first, and
    it arrives from sqlite as ``sqlite3.OperationalError`` — again not an
    ``OSError``, so the same gap covered it.
    """
    import os

    from local_operator.providers.registry import stored_provider_env_keys

    path = _initialized_then_damaged(tmp_path, b"")
    # The store is only LOCKED, not overwritten: the payload above is empty, so the
    # bytes on disk are still a valid database and the failure can only be the
    # permission check.
    os.chmod(path, 0o000)
    try:
        assert stored_provider_env_keys(tmp_path) == set()
    finally:
        # The tmp_path teardown needs to traverse what it created.
        os.chmod(path, 0o600)


def test_a_store_connection_used_from_another_thread_still_raises(
    tmp_path: Any, monkeypatch: Any
) -> None:
    """The one sqlite failure that is NOT an unreadable store: a caller BUG.

    ``sqlite3.ProgrammingError`` is re-raised inside the family's own clause order
    (it subclasses ``DatabaseError``, so the narrow clause must come first), the
    same precedent ``usable_providers`` sets: a connection crossing threads must
    not be dressed as "no provider rows", because that is a bug nobody would ever
    find.
    """
    import sqlite3

    from local_operator.providers.registry import stored_provider_env_keys

    _seed_store_file(tmp_path, b"")

    def raiser(_root: Any) -> Any:
        raise sqlite3.ProgrammingError("SQLite objects created in a thread can only be used in it")

    monkeypatch.setattr("local_operator.secrets.access.open_store", raiser)
    with pytest.raises(sqlite3.ProgrammingError):
        stored_provider_env_keys(tmp_path)


def test_a_row_whose_blob_columns_hold_text_reads_as_no_provider_rows(tmp_path: Any) -> None:
    """R3-1: a malformed ROW is a store state the reader must survive too.

    A store that is otherwise intact, with one row's ``ciphertext`` holding text
    instead of a BLOB, raised ``TypeError: string argument without an encoding``
    out of the decryptor's ``bytes(row[5])`` — so the reader answered nothing about
    that store except an exception, and the live route answered 500. The reader's
    contract is "no provider rows I can see", for every shape it cannot read, which
    is what this asserts.

    The deeper fix belongs in ``secrets.store._decode`` (validate the byte columns
    and raise ``SecretCorrupt``, so ``_enumerate`` reports the row through
    ``damaged_records`` — the treatment ``_read_meta_int`` already got); that is the
    secrets layer's contract, and it is recorded on the clause rather than fixed
    here.
    """
    import sqlite3

    from local_operator.providers.registry import (
        store_provider_key,
        stored_provider_env_keys,
    )
    from local_operator.secrets.keys import store_path

    store_provider_key("OPENROUTER_API_KEY", "fixture-key", base=tmp_path)
    assert stored_provider_env_keys(tmp_path) == {"OPENROUTER_API_KEY"}, "a real store, readable"

    connection = sqlite3.connect(store_path(tmp_path))
    with connection:
        connection.execute("UPDATE secrets SET ciphertext = 'not-bytes'")
    connection.close()

    assert stored_provider_env_keys(tmp_path) == set()
