"""The ``createIf`` gate's probe matrix — sync, local, and never raising.

Design D9: the gate answers "can this machine reach ANY image provider" from
sqlite rows, the encrypted store and the environment only; no socket may open
on a session-build path. The deliberate divergence between rungs — Radient
lights from persisted rows ONLY, FAL/OpenAI accept a stored row OR an exported
key — is pinned through a REAL ``AuthStore`` where it matters, the way
``test_cascade_persisted_only`` pins the STT half.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pytest

from local_operator.imagegen import availability
from local_operator.providers.auth_store import AuthStore

ENV_VARS = ("RADIENT_API_KEY", "FAL_API_KEY", "OPENAI_API_KEY")


@pytest.fixture()
def config_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A config root with no provider vars in the environment.

    Every test below states what it DOES have; ambient exports from the
    developer's shell would otherwise answer for it.
    """
    for var in ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    root = tmp_path / "config"
    root.mkdir()
    return root


def _store(config_root: Path) -> AuthStore:
    return AuthStore(db_path=config_root / "auth.db", config_dir=config_root)


@pytest.fixture()
def store(config_root: Path) -> Iterator[AuthStore]:
    auth = _store(config_root)
    yield auth
    auth.close()


# ---------------------------------------------------------------------------
# Radient: persisted rows only (the STT/mobile rule)
# ---------------------------------------------------------------------------


def test_radient_env_never_lights_the_gate(
    store: AuthStore, config_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RADIENT_API_KEY", "ambient-not-a-login")
    assert availability.radient_available(config_root) is False


def test_radient_lights_from_a_stored_row(store: AuthStore, config_root: Path) -> None:
    store.upsert_credential(
        "radient",
        {
            "type": "oauth",
            "refresh": "r1",
            "access": "a1",
            "expires": 4_000_000_000_000,
        },
    )
    assert availability.radient_available(config_root) is True


# ---------------------------------------------------------------------------
# FAL and OpenAI: login row -> provider store row -> exported key
# ---------------------------------------------------------------------------


def test_fal_reads_the_exported_key_when_nothing_is_stored(
    config_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert availability.fal_key(config_root) is None
    monkeypatch.setenv("FAL_API_KEY", "exported-key")
    assert availability.fal_key(config_root) == "exported-key"


def test_fal_login_row_wins_over_the_export(
    store: AuthStore, config_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FAL_API_KEY", "exported-key")
    store.upsert_credential("fal", {"type": "api_key", "source": "login", "key": "stored-key"})
    assert availability.fal_key(config_root) == "stored-key"


def test_fal_store_row_answers_when_no_login_row_exists(
    config_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[str] = []

    def fake_provider_secret(name: str, *, base: Path | None = None) -> str | None:
        seen.append(name)
        return "store-row-key" if name == "FAL_API_KEY" else None

    monkeypatch.setattr(
        "local_operator.providers.registry.provider_secret_value", fake_provider_secret
    )
    assert availability.fal_key(config_root) == "store-row-key"
    assert seen == ["FAL_API_KEY"]


def test_openai_oauth_rows_never_answer_only_api_keys_do(
    store: AuthStore, config_root: Path
) -> None:
    # A ChatGPT OAuth row is not valid at the images API; the probe must
    # ignore it exactly as the call-time twin does.
    store.upsert_credential(
        "openai-key",
        {"type": "oauth", "refresh": "r1", "access": "chatgpt-token", "expires": 4_000_000_000_000},
    )
    assert availability.openai_images_key(config_root) is None

    store.upsert_credential("openai-key", {"type": "api_key", "source": "login", "key": "sk-test"})
    assert availability.openai_images_key(config_root) == "sk-test"


def test_openai_store_row_then_export(config_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "local_operator.providers.registry.provider_secret_value",
        lambda name, *, base=None: "store-row-key" if name == "OPENAI_API_KEY" else None,
    )
    assert availability.openai_images_key(config_root) == "store-row-key"

    monkeypatch.setattr(
        "local_operator.providers.registry.provider_secret_value",
        lambda name, *, base=None: None,
    )
    monkeypatch.setenv("OPENAI_API_KEY", "exported-key")
    assert availability.openai_images_key(config_root) == "exported-key"


# ---------------------------------------------------------------------------
# The gate itself
# ---------------------------------------------------------------------------


def test_reachable_is_the_any_of_matrix(
    store: AuthStore, config_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert availability.image_provider_reachable(config_root) is False

    monkeypatch.setenv("FAL_API_KEY", "fk")
    assert availability.image_provider_reachable(config_root) is True
    monkeypatch.delenv("FAL_API_KEY")

    monkeypatch.setenv("OPENAI_API_KEY", "ok")
    assert availability.image_provider_reachable(config_root) is True
    monkeypatch.delenv("OPENAI_API_KEY")

    store.upsert_credential(
        "openai",
        {"type": "oauth", "refresh": "r", "access": "a", "expires": 4_000_000_000_000},
    )
    assert availability.image_provider_reachable(config_root) is True

    monkeypatch.setenv("GOOGLE_AI_STUDIO_API_KEY", "gk")
    assert availability.image_provider_reachable(config_root) is True
    monkeypatch.delenv("GOOGLE_AI_STUDIO_API_KEY")

    monkeypatch.setenv("XAI_API_KEY", "xk")
    assert availability.image_provider_reachable(config_root) is True
    monkeypatch.delenv("XAI_API_KEY")

    store.upsert_credential(
        "radient", {"type": "oauth", "refresh": "r", "access": "a", "expires": 4_000_000_000_000}
    )
    assert availability.image_provider_reachable(config_root) is True


def test_probes_never_raise(config_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A broken probe reads as "not available" — never takes a session down."""

    class _Broken:
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("store unavailable")

    monkeypatch.setattr(availability, "_open_store", _Broken)
    assert availability.radient_available(config_root) is False
    assert availability.fal_key(config_root) is None
    assert availability.openai_images_key(config_root) is None
    assert availability.openai_subscription_grant(config_root) is False
    assert availability.google_key(config_root) is None
    assert availability.xai_available(config_root) is False
    assert availability.image_provider_reachable(config_root) is False


@pytest.mark.asyncio
async def test_the_async_twin_agrees_with_the_sync_probe(
    store: AuthStore, config_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The OpenAI probe and its call-time twin read the same credential class.

    An availability answer about one credential class and a request sent with
    another is the defect this pair exists to prevent (the STT lane's own
    wording): both must take ``api_key`` rows and both must fall back to the
    exported key.
    """
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert availability.openai_images_key(config_root) is None
    assert await availability.openai_call_key(store) is None

    store.upsert_credential("openai-key", {"type": "api_key", "source": "login", "key": "sk-x"})
    assert availability.openai_images_key(config_root) == "sk-x"
    assert await availability.openai_call_key(store) == "sk-x"


# ---------------------------------------------------------------------------
# The subscription grant: OAuth rows only, the inverse of the images rule
# ---------------------------------------------------------------------------


def test_openai_subscription_probe_reads_oauth_rows_only(
    store: AuthStore, config_root: Path
) -> None:
    # A platform api_key is NOT a subscription grant (that row belongs to
    # openai-key); the OAuth grant the images API rejects is exactly what the
    # Codex-backend rung spends. The two probes must never blur into one.
    store.upsert_credential("openai-key", {"type": "api_key", "source": "login", "key": "sk-x"})
    assert availability.openai_subscription_grant(config_root) is False

    store.upsert_credential(
        "openai",
        {
            "type": "oauth",
            "refresh": "r1",
            "access": "chatgpt-token",
            "expires": 4_000_000_000_000,
        },
    )
    assert availability.openai_subscription_grant(config_root) is True


def test_openai_subscription_env_never_lights_the_gate(
    config_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A grant is stored by a sign-in, never exported; an ambient platform key
    # must not advertise this rung.
    monkeypatch.setenv("OPENAI_API_KEY", "exported-key")
    assert availability.openai_subscription_grant(config_root) is False


@pytest.mark.asyncio
async def test_the_subscription_access_twin_serves_the_stored_grant(store: AuthStore) -> None:
    assert await availability.openai_sub_access(store) is None

    store.upsert_credential(
        "openai",
        {
            "type": "oauth",
            "refresh": "r1",
            "access": "chatgpt-token",
            "expires": 4_000_000_000_000,
            "account_id": "acc-1",
        },
    )
    access = await availability.openai_sub_access(store)
    assert access is not None
    assert access.kind == "oauth"
    assert access.access_token == "chatgpt-token"
    assert access.account_id == "acc-1"


# ---------------------------------------------------------------------------
# Google: login row -> provider store row -> exported key (the FAL shape)
# ---------------------------------------------------------------------------


def test_google_reads_the_exported_key_when_nothing_is_stored(
    config_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert availability.google_key(config_root) is None
    monkeypatch.setenv("GOOGLE_AI_STUDIO_API_KEY", "exported-key")
    assert availability.google_key(config_root) == "exported-key"


def test_google_login_row_wins_over_the_export(
    store: AuthStore, config_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GOOGLE_AI_STUDIO_API_KEY", "exported-key")
    store.upsert_credential("google", {"type": "api_key", "source": "login", "key": "gkey"})
    assert availability.google_key(config_root) == "gkey"


def test_google_store_row_answers_when_no_login_row_exists(
    config_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[str] = []

    def fake_provider_secret(name: str, *, base: Path | None = None) -> str | None:
        seen.append(name)
        return "store-row-key" if name == "GOOGLE_AI_STUDIO_API_KEY" else None

    monkeypatch.setattr(
        "local_operator.providers.registry.provider_secret_value", fake_provider_secret
    )
    assert availability.google_key(config_root) == "store-row-key"
    assert seen == ["GOOGLE_AI_STUDIO_API_KEY"]


@pytest.mark.asyncio
async def test_the_google_async_twin_agrees_with_the_sync_probe(
    store: AuthStore, config_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("GOOGLE_AI_STUDIO_API_KEY", raising=False)
    assert availability.google_key(config_root) is None
    assert await availability.google_call_key(store) is None

    store.upsert_credential("google", {"type": "api_key", "source": "login", "key": "gkey"})
    assert availability.google_key(config_root) == "gkey"
    assert await availability.google_call_key(store) == "gkey"


# ---------------------------------------------------------------------------
# xAI: key rows OR the Grok OAuth grant (both store under "xai"), or env
# ---------------------------------------------------------------------------


def test_xai_lights_from_an_api_key_row(store: AuthStore, config_root: Path) -> None:
    assert availability.xai_available(config_root) is False
    store.upsert_credential("xai", {"type": "api_key", "source": "login", "key": "xk"})
    assert availability.xai_available(config_root) is True


def test_xai_lights_from_an_oauth_grant(store: AuthStore, config_root: Path) -> None:
    store.upsert_credential(
        "xai",
        {"type": "oauth", "refresh": "r", "access": "a", "expires": 4_000_000_000_000},
    )
    assert availability.xai_available(config_root) is True


def test_xai_lights_from_the_export(config_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("XAI_API_KEY", "exported")
    assert availability.xai_available(config_root) is True
    monkeypatch.delenv("XAI_API_KEY")
    assert availability.xai_available(config_root) is False


@pytest.mark.asyncio
async def test_the_xai_bearer_twin_prefers_the_store_then_env(
    store: AuthStore, config_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XAI_API_KEY", "exported")
    assert await availability.xai_call_bearer(store) == "exported"

    store.upsert_credential("xai", {"type": "api_key", "source": "login", "key": "xk"})
    assert await availability.xai_call_bearer(store) == "xk"
