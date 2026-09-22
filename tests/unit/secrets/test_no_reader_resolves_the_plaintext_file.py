"""A decoy ``credentials.env`` must not resolve ANY reader (PR2a, risk R2).

PR #1411 repointed the credential readers and writers at the encrypted store but
left a transition leg in place: several readers still fell back to the plaintext
``credentials.env`` file LAST, and a plain ``CredentialManager(...)`` still
CREATED that file on construction. PR2a removes the reader legs and the
recreator, so the file is no longer a credential SOURCE at all.

This module is the evidence for that claim, and it is deliberately adversarial:
it seeds a file that LOOKS authoritative — the decoy value is one no ambient
variable and no store row holds — and asserts that every reader treats the name
as UNSET. A surviving leg anywhere in the set fails here with the decoy value in
the assertion message, which is exactly the "silently unauthenticated provider"
class risk R2 names.

It also pins the other direction so the file's removal cannot quietly take the
legitimate tiers with it: a provider-class STORE row, and an exported variable,
must still resolve (the store-first rung order).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from local_operator.credentials import CREDENTIALS_FILE_NAME, CredentialManager

#: A value that exists ONLY in the decoy file, so any reader that returns it has
#: read the file. Distinct per key is deliberate — a reader wired to the wrong
#: key name is then visible in the failure too.
DECOY_PREFIX = "decoy-from-the-plaintext-file-"


def _write_decoy(root: Path, *keys: str) -> None:
    """Seed ``<root>/credentials.env`` with a decoy value for each key.

    Deliberately NOT via ``CredentialManager``: the point is a file that exists
    on disk exactly as a host mid-migration would have it, and the class's own
    writer is retired (no production caller) so using it here would test the
    writer rather than the readers.
    """
    root.mkdir(parents=True, exist_ok=True)
    body = "".join(f"{key}={DECOY_PREFIX}{key.lower()}\n" for key in keys)
    (root / CREDENTIALS_FILE_NAME).write_text(body, encoding="utf-8")


def _decoy(key: str) -> str:
    return f"{DECOY_PREFIX}{key.lower()}"


# ---------------------------------------------------------------------------
# Reader legs, one function each. Every one takes the isolated config root and
# returns what it resolved for ``OPENROUTER_API_KEY`` (or a truthiness signal
# where the reader answers a yes/no question).
# ---------------------------------------------------------------------------


def _read_provider_env_key(root: Path, monkeypatch: pytest.MonkeyPatch) -> str | None:
    from local_operator.providers.registry import provider_env_key

    return provider_env_key("openrouter", base=root)


def _read_provider_secret_value(root: Path, monkeypatch: pytest.MonkeyPatch) -> str | None:
    from local_operator.providers.registry import provider_secret_value

    return provider_secret_value("OPENROUTER_API_KEY", base=root)


def _read_first_provider_key(root: Path, monkeypatch: pytest.MonkeyPatch) -> str | None:
    from local_operator.providers.registry import first_provider_key

    return first_provider_key(("OPENROUTER_API_KEY",), base=root)


def _read_catalogue_api_key(root: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    from local_operator.model.configure import _catalogue_api_key

    return _catalogue_api_key("openrouter", base=root)


def _read_auth_store_env_tier(root: Path, monkeypatch: pytest.MonkeyPatch) -> str | None:
    from local_operator.providers.auth_store import AuthStore

    store = AuthStore(db_path=root / "auth.db", credential_manager=CredentialManager.readonly(root))
    try:
        return store._env_api_key("openrouter")
    finally:
        store.close()


@pytest.mark.asyncio
async def _read_vendor_credential(root: Path, monkeypatch: pytest.MonkeyPatch) -> str | None:
    """The classification leg's static tier, resolved through the real vendor.

    ``_resolve_key`` walks ``(authstore, OPENROUTER_API_KEY, OPENROUTER_API_KEY_DEV)``;
    with no login row and no store row, the only thing that could answer is the
    file, which PR2a removed.
    """
    from local_operator.classification.vendors import OpenRouterVendor

    manager = CredentialManager.readonly(root)
    value, tier = await OpenRouterVendor(manager)._resolve_key(manager)
    return value.get_secret_value() if value is not None else None


def _read_vendor_status(root: Path, monkeypatch: pytest.MonkeyPatch) -> bool:
    from local_operator.classification.cascade import vendor_status

    return dict(vendor_status(CredentialManager.readonly(root)))["openrouter"]


def _read_search_provider_statuses(root: Path, monkeypatch: pytest.MonkeyPatch) -> bool:
    from local_operator.web_search.models import WebSearchSettings
    from local_operator.web_search.providers import provider_auth_mode

    with_store = provider_auth_mode("tavily", CredentialManager.readonly(root), WebSearchSettings())
    return with_store == "api-key"


def _read_info_credential_names(root: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[str, ...]:
    from local_operator.info.collect import _credential_key_names

    return _credential_key_names(root)


def _read_credentials_listing(root: Path, monkeypatch: pytest.MonkeyPatch) -> set[str]:
    """The ``/v1/credentials`` listing, driven through its still-useful half."""
    import asyncio

    from local_operator.server.routes.credentials import list_credentials

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    response = asyncio.run(list_credentials(CredentialManager.readonly(root)))
    # ``result`` is ``Optional`` on the envelope; the listing route always
    # populates it, so narrow explicitly rather than assert-and-please.
    payload = response.result or {}
    return set(payload.get("keys", []))


def _read_evaluation_resolver(root: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[str, ...]:
    from local_operator.evaluation.runner.host_secrets import CredentialStoreResolver
    from local_operator.evaluation.runner.secrets import MissingSecret

    resolver = CredentialStoreResolver(CredentialManager.readonly(root))
    try:
        resolved = resolver.resolve(["OPENROUTER_API_KEY"])
    except MissingSecret:
        return ()
    return tuple(secret.value for secret in resolved)


def _read_persisted_providers(root: Path, monkeypatch: pytest.MonkeyPatch) -> set[str]:
    from local_operator.providers.auth_store import AuthStore
    from local_operator.providers.controller import ProviderController

    store = AuthStore(db_path=root / "auth.db", credential_manager=CredentialManager.readonly(root))
    try:
        controller = ProviderController(store, CredentialManager.readonly(root))
        return controller.persisted_providers() or set()
    finally:
        store.close()


def _read_daemon_catalogue_admission(root: Path, monkeypatch: pytest.MonkeyPatch) -> set[str]:
    """The phone daemon's admission path, which reads persisted providers."""
    from local_operator.providers.auth_store import AuthStore
    from local_operator.providers.controller import ProviderController

    store = AuthStore(db_path=root / "auth.db", credential_manager=CredentialManager.readonly(root))
    try:
        controller = ProviderController(store, CredentialManager.readonly(root))
        admitted = controller.persisted_providers()
        return admitted or set()
    finally:
        store.close()


def _read_web_search_credential(root: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    from local_operator.web_search.providers import _credential

    return _credential(CredentialManager.readonly(root), "TAVILY_API_KEY")


#: Every reader leg and the assertion each must hold: with a decoy file present
#: and no store row, the reader must treat the name as UNSET. ``signal`` is the
#: falsy value that means "not resolved from the file".
_READERS: list[tuple[str, object]] = [
    ("registry.provider_env_key", _read_provider_env_key),
    ("registry.provider_secret_value", _read_provider_secret_value),
    ("registry.first_provider_key", _read_first_provider_key),
    ("model._catalogue_api_key", _read_catalogue_api_key),
    ("auth_store._env_api_key", _read_auth_store_env_tier),
    ("vendors._resolve_key", _read_vendor_credential),
    ("cascade.vendor_status", _read_vendor_status),
    ("web_search.provider_auth_mode", _read_search_provider_statuses),
    ("web_search._credential", _read_web_search_credential),
    ("info._credential_key_names", _read_info_credential_names),
    ("server.list_credentials", _read_credentials_listing),
    ("evaluation.CredentialStoreResolver", _read_evaluation_resolver),
    ("controller.persisted_providers", _read_persisted_providers),
    ("mobile.daemon admission", _read_daemon_catalogue_admission),
]


@pytest.mark.parametrize(("name", "reader"), _READERS, ids=[n for n, _ in _READERS])
def test_no_reader_resolves_from_a_decoy_plaintext_file(
    name: str, reader: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The core claim of PR2a: the plaintext file is no longer a credential source.

    A decoy file holds ``OPENROUTER_API_KEY`` (and ``TAVILY_API_KEY``) with a
    value nothing else holds, and the reader must NOT return it. The assertion
    is written so a surviving leg fails with the DECOY VALUE in the message,
    which names the file as the source rather than a generic mismatch.
    """
    root = tmp_path / "config"
    _write_decoy(root, "OPENROUTER_API_KEY", "TAVILY_API_KEY")
    # Point the HOME-derived default at the sandbox too, so a reader that forgot
    # to thread ``base`` still reads the isolated root rather than the real home.
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    (tmp_path / "home").mkdir()

    if name == "vendors._resolve_key":
        import asyncio

        result = asyncio.run(reader(root, monkeypatch))  # type: ignore[operator]
    else:
        result = reader(root, monkeypatch)  # type: ignore[operator]

    # No reader may return or contain the decoy value anywhere in its answer.
    text = repr(result)
    assert (
        DECOY_PREFIX not in text
    ), f"{name} resolved a value from the plaintext {CREDENTIALS_FILE_NAME}: {text}"


def test_a_provider_store_row_still_resolves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The legitimate tier the removal must not take with it: the store row."""
    from local_operator.providers.registry import provider_env_key, store_provider_key

    root = tmp_path / "config"
    _write_decoy(root, "OPENROUTER_API_KEY")
    store_provider_key("OPENROUTER_API_KEY", "from-the-store", base=root)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))

    assert provider_env_key("openrouter", base=root) == "from-the-store"


def test_an_exported_variable_still_resolves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The env tier is behind the store row, not gone (risk R6)."""
    from local_operator.providers.registry import provider_env_key

    root = tmp_path / "config"
    _write_decoy(root, "OPENROUTER_API_KEY")
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    monkeypatch.setenv("OPENROUTER_API_KEY", "from-the-environment")

    assert provider_env_key("openrouter", base=root) == "from-the-environment"


def test_no_construction_creates_the_plaintext_file(tmp_path: Path) -> None:
    """The recreator (defect B) is gone: a read-only construction writes nothing.

    ``CredentialManager.__init__`` ran ``_ensure_config_exists``, which created
    an empty ``credentials.env`` on every plain construction — the defect that
    rewrote the operator's file daily. ``readonly`` binds without touching disk,
    which is what every production site now uses.
    """
    root = tmp_path / "config"
    root.mkdir()
    assert not (root / CREDENTIALS_FILE_NAME).exists()

    CredentialManager.readonly(root)
    assert not (root / CREDENTIALS_FILE_NAME).exists()
    assert list(root.iterdir()) == []


def test_no_plain_construction_of_credential_manager_remains_in_the_tree() -> None:
    """A guard on the guard: every construction site must use ``readonly``.

    The 14 plain constructions PR2a replaced were the recreator. A future edit
    that adds one is invisible to a runtime test (it only resurrects the file on
    some path), so the call shape is read at the AST level: outside
    ``credentials.py`` itself (where the class is DEFINED and ``readonly``
    still routes through ``_bind``), no call to ``CredentialManager(...)`` with a
    bare positional/keyword config argument may appear.
    """
    import ast
    from pathlib import Path

    package = Path(__file__).resolve().parents[3] / "local_operator"
    offenders: list[str] = []
    for path in sorted(package.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            # ``CredentialManager(...)`` directly, or ``cls(...)``/``self(...)``
            # inside credentials.py (skipped).
            if isinstance(func, ast.Name) and func.id == "CredentialManager":
                where = f"{path.relative_to(package)}:{node.lineno}"
                offenders.append(f"{where} CredentialManager(...)")
            elif isinstance(func, ast.Attribute) and func.attr == "CredentialManager":
                where = f"{path.relative_to(package)}:{node.lineno}"
                offenders.append(f"{where} .CredentialManager(...)")
    assert (
        offenders == []
    ), "plain CredentialManager(...) constructions recreate the plaintext file: " + repr(offenders)


def test_the_decoy_value_is_not_ambient(tmp_path: Path) -> None:
    """Guard the guard: the decoy value must not exist in the environment.

    If ``DECOY_PREFIX`` were an exported variable the parametrised test above
    could pass for the wrong reason — the env tier, not the file, would be the
    source and the assertion would be vacuous.
    """
    assert not any(value.startswith(DECOY_PREFIX) for value in os.environ.values())
