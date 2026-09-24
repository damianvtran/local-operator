"""A decoy ``credentials.env`` must not resolve ANY reader (PR2a, risk R2).

PR #1411 repointed the credential readers and writers at the encrypted store but
left a transition leg in place: several readers still fell back to the plaintext
``credentials.env`` file LAST, and a plain ``CredentialManager(...)`` still
CREATED that file on construction. PR2a removes the reader legs and the
recreator and PR2b deletes ``credentials.py`` entirely, so the file is no
longer a credential SOURCE at all — the only code that touches it is the
migration's own reader in ``local_operator.secrets.legacy_env``.

This module is the evidence for that claim, and it is deliberately adversarial:
it seeds a file that LOOKS authoritative — the decoy value is one no ambient
variable and no store row holds — and asserts that every reader treats the name
as UNSET.

Two shapes of reader need two shapes of assertion, and the distinction is the
point of the sweep:

* a reader that returns a **value** is asserted against the decoy STRING, so its
  failure names the file as the source;
* a reader that answers a **presence question** — a bool, a set/tuple of NAMES —
  cannot express "I read the file" as a value at all, and asserting `decoy not in
  repr(result)` against it can never fail. Those are asserted FALSY instead
  (`not result` / `result == ()` / `result == set()`), which is what the pre-fix
  base violates: it returned the file's names, or `True`.

Every leg that a REMOVED file read was reachable from fails on the pre-fix base; the
falsification run is recorded on the PR. One leg is deliberately a CONTROL rather
than a removed read — ``registry.provider_secret_value`` was already store-only on
the base (its only plaintext mention is a comment), so it passes there too; it is
kept in ``_READERS`` to pin the store-only shape it must retain, not as
discrimination evidence, and the docstrings and the PR body say "every removed
leg" (16 of 17) rather than "every leg". A leg that no callable reaches — a leg
removed from a path nothing exercises — is removal this sweep cannot see, so
named production helpers exist for each leg it drives.

It also pins the other direction so the file's removal cannot quietly take the
legitimate tiers with it: a provider-class STORE row, and an exported variable,
must still resolve (the store-first rung order).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, cast

import pytest

from local_operator.secrets.legacy_env import CREDENTIALS_FILE_NAME

#: A value that exists ONLY in the decoy file, so any reader that returns it has
#: read the file. Distinct per key is deliberate — a reader wired to the wrong
#: key name is then visible in the failure too.
DECOY_PREFIX = "decoy-from-the-plaintext-file-"


def _write_decoy(root: Path, *keys: str) -> None:
    """Seed ``<root>/credentials.env`` with a decoy value for each key.

    Deliberately NOT via the retired module: the point is a file that exists
    on disk exactly as a host mid-migration would have it, and the class that
    used to write it is deleted (PR2b), so there is no writer left to test.
    """
    root.mkdir(parents=True, exist_ok=True)
    body = "".join(f"{key}={DECOY_PREFIX}{key.lower()}\n" for key in keys)
    (root / CREDENTIALS_FILE_NAME).write_text(body, encoding="utf-8")


def _decoy(key: str) -> str:
    return f"{DECOY_PREFIX}{key.lower()}"


def _config_manager(root: Path):
    """A ``ConfigManager`` bound to the sandbox root.

    The ``/v1/credentials`` route takes a ``ConfigManager`` now that PR2b deleted
    the ``CredentialManager`` whose ``config_dir`` it used to read.
    """
    from local_operator.config import ConfigManager

    return ConfigManager(root)


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

    store = AuthStore(db_path=root / "auth.db", config_dir=root)
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

    vendor = OpenRouterVendor(root)
    value, tier = await vendor._resolve_key(root)
    return value.get_secret_value() if value is not None else None


def _read_vendor_status(root: Path, monkeypatch: pytest.MonkeyPatch) -> bool:
    from local_operator.classification.cascade import vendor_status

    return dict(vendor_status(root))["openrouter"]


def _read_search_provider_statuses(root: Path, monkeypatch: pytest.MonkeyPatch) -> bool:
    from local_operator.web_search.models import WebSearchSettings
    from local_operator.web_search.providers import provider_auth_mode

    with_store = provider_auth_mode("tavily", root, WebSearchSettings())
    return with_store == "api-key"


def _read_info_credential_names(root: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[str, ...]:
    from local_operator.info.collect import _credential_key_names

    return _credential_key_names(root)


def _read_credentials_listing(root: Path, monkeypatch: pytest.MonkeyPatch) -> set[str]:
    """The ``/v1/credentials`` listing, driven through its still-useful half."""
    import asyncio

    from local_operator.server.routes.credentials import list_credentials

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    response = asyncio.run(list_credentials(_config_manager(root)))
    # ``result`` is ``Optional`` on the envelope; the listing route always
    # populates it, so narrow explicitly rather than assert-and-please.
    payload = response.result or {}
    return set(payload.get("keys", []))


def _read_evaluation_resolver(root: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[str, ...]:
    from local_operator.evaluation.runner.host_secrets import CredentialStoreResolver
    from local_operator.evaluation.runner.secrets import MissingSecret

    resolver = CredentialStoreResolver(root)
    try:
        resolved = resolver.resolve(["OPENROUTER_API_KEY"])
    except MissingSecret:
        return ()
    return tuple(secret.value for secret in resolved)


def _read_persisted_providers(root: Path, monkeypatch: pytest.MonkeyPatch) -> set[str]:
    from local_operator.providers.auth_store import AuthStore
    from local_operator.providers.controller import ProviderController

    store = AuthStore(db_path=root / "auth.db", config_dir=root)
    try:
        controller = ProviderController(store, root)
        return controller.persisted_providers() or set()
    finally:
        store.close()


def _read_daemon_catalogue_admission(root: Path, monkeypatch: pytest.MonkeyPatch) -> set[str]:
    """The phone daemon's admission path, which reads persisted providers."""
    from local_operator.providers.auth_store import AuthStore
    from local_operator.providers.controller import ProviderController

    store = AuthStore(db_path=root / "auth.db", config_dir=root)
    try:
        controller = ProviderController(store, root)
        admitted = controller.persisted_providers()
        return admitted or set()
    finally:
        store.close()


def _read_web_search_credential(root: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    from local_operator.web_search.providers import _credential

    return _credential(root, "TAVILY_API_KEY")


def _read_session_factory_credential(root: Path, monkeypatch: pytest.MonkeyPatch) -> str | None:
    """The knowledge/embedder key resolver ``_setup_knowledge`` wires in.

    Driven through the NAMED production helper ``_knowledge_credential`` — with
    the decoy file present and no store row and no export, it must resolve
    nothing. Before PR2a this leg reached the plaintext file, so a decoy read
    here would return the decoy value.
    """
    from local_operator.session_factory import _knowledge_credential

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    return _knowledge_credential("OPENROUTER_API_KEY", root)


def _read_auth_cli_stored_names(root: Path, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """The store-row name log ``list_logins`` prints.

    ``list_logins`` is a printer, so the sweep drives the NAMED reader
    ``stored_login_key_names`` that produces its names. Before PR2a the printed
    set included a ``credentials.env`` loop, so the file's keys appeared here.
    """
    from local_operator.providers.auth_cli import stored_login_key_names

    return stored_login_key_names(root)


def _read_tool_digest(root: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """The per-call singleflight digest ``execute_web_search`` builds.

    Driven through the NAMED production helper ``_search_digest_key``: the digest
    covers EXPORTED credential values only, so a key present only in the decoy
    file must not change it. Before PR2a the digest read the plaintext file
    directly, so the two digests (decoy present vs absent) differed.
    """
    from local_operator.web_search.models import WebSearchSettings
    from local_operator.web_search.service import WebSearchService
    from local_operator.web_search.tool import _search_digest_key

    for name in (
        "TAVILY_API_KEY",
        "OPENROUTER_API_KEY",
        "EXA_API_KEY",
        "PARALLEL_API_KEY",
        "DEEPSEEK_API_KEY",
        "PERPLEXITY_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    settings = WebSearchSettings(providers=["tavily"])
    service = WebSearchService(settings, root)
    params = _digest_params()
    return _search_digest_key(service, settings, params)[2]


def _digest_params():
    """A minimal validated params object for the digest helper."""
    from local_operator.web_search.tool import WebSearchParams

    return WebSearchParams(query="decoy digest probe", max_results=3)


#: Every reader leg and how the assertion reads its answer.
#:
#: ``"value"`` — the reader returns a credential VALUE, so the decoy string
#: must not appear in its repr (a surviving leg fails naming the decoy).
#: ``"falsy"`` — the reader answers a presence question (bool, set/tuple of
#: NAMES), where the decoy string can never appear, so the assertion is that the
#: answer is EMPTY. On the pre-fix base these returned the file's names or
#: ``True``; a truthiness assertion is what makes that visible.
_READERS: list[tuple[str, object, str]] = [
    ("registry.provider_env_key", _read_provider_env_key, "value"),
    ("registry.provider_secret_value", _read_provider_secret_value, "value"),
    ("registry.first_provider_key", _read_first_provider_key, "value"),
    ("model._catalogue_api_key", _read_catalogue_api_key, "value"),
    ("auth_store._env_api_key", _read_auth_store_env_tier, "value"),
    ("vendors._resolve_key", _read_vendor_credential, "value"),
    ("cascade.vendor_status", _read_vendor_status, "falsy"),
    ("web_search.provider_auth_mode", _read_search_provider_statuses, "falsy"),
    ("web_search._credential", _read_web_search_credential, "value"),
    ("info._credential_key_names", _read_info_credential_names, "falsy"),
    ("server.list_credentials", _read_credentials_listing, "falsy"),
    ("evaluation.CredentialStoreResolver", _read_evaluation_resolver, "value"),
    ("controller.persisted_providers", _read_persisted_providers, "falsy"),
    ("mobile.daemon admission", _read_daemon_catalogue_admission, "falsy"),
    ("session_factory._knowledge_credential", _read_session_factory_credential, "value"),
    ("auth_cli.stored_login_key_names", _read_auth_cli_stored_names, "falsy"),
    ("web_search.tool._search_digest_key", _read_tool_digest, "value"),
]


@pytest.mark.parametrize(
    ("name", "reader", "shape"),
    _READERS,
    ids=[n for n, _, _ in _READERS],
)
def test_no_reader_resolves_from_a_decoy_plaintext_file(
    name: str,
    reader: object,
    shape: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The core claim of PR2a: the plaintext file is no longer a credential source.

    A decoy file holds ``OPENROUTER_API_KEY`` (and ``TAVILY_API_KEY``) with a
    value nothing else holds, and the reader must NOT answer from it. A VALUE
    reader is asserted against the decoy string, so a surviving leg fails naming
    the file as the source; a PRESENCE reader (bool, names) is asserted EMPTY,
    because that is the only answer shape in which a file read is visible — on
    the pre-fix base it returned the file's names or ``True``.
    """
    root = tmp_path / "config"
    _write_decoy(root, "OPENROUTER_API_KEY", "TAVILY_API_KEY")
    # Point the HOME-derived default at the sandbox too, so a reader that forgot
    # to thread ``base`` still reads the isolated root rather than the real home.
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    (tmp_path / "home").mkdir()

    if name == "vendors._resolve_key":
        import asyncio

        result = asyncio.run(reader(root, monkeypatch))  # type: ignore[operator]
    else:
        result = reader(root, monkeypatch)  # type: ignore[operator]

    if shape == "falsy":
        assert not result, (
            f"{name} answered from the plaintext {CREDENTIALS_FILE_NAME}: "
            f"expected an empty/presence-negative answer, got {result!r}"
        )
        return
    # A VALUE reader: the decoy value must appear nowhere in its answer.
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


# ---------------------------------------------------------------------------
# Three legs the sweep drives through production callables rather than through
# a private function directly: the session's embedder key resolver, the
# ``list_logins`` status printout, and the web-search tool's per-call digest.
# Each is asserted as an EQUIVALENCE, which is the strongest form available for
# a path whose output is a digest or a printout: the answer must be identical
# whether or not the decoy file exists. On the pre-fix base each differs, so
# each test discriminates.
# ---------------------------------------------------------------------------


def _embedder_backend_credential(root: Path, *, store_value: str | None = None) -> str | None:
    """The embedder key a REAL ``_setup_knowledge`` run hands its backend.

    ``default_backend_from_env`` is stood in for so the resolver the production
    path actually built can be called directly — the assertion is about that
    closure, not about a re-implementation of it. The agent registry is faked
    down to ``list_agents``, the only method the hint builder reaches.
    """
    import asyncio

    import local_operator.skills.api as skills_api
    from local_operator.session_factory import _setup_knowledge

    if store_value is not None:
        from local_operator.providers.registry import store_provider_key

        store_provider_key("OPENROUTER_API_KEY", store_value, base=root)

    class _Registry:
        """Only ``list_agents`` is reached: the hint builder calls nothing else."""

        def list_agents(self) -> list[object]:
            return []

    captured: dict[str, object] = {}

    def _capture(resolver, base_url=None):
        from local_operator.skills.embeddings import LocalEmbedder

        captured["resolver"] = resolver
        return LocalEmbedder()

    real = skills_api.default_backend_from_env
    skills_api.default_backend_from_env = _capture
    try:
        asyncio.run(_setup_knowledge(root, cast(Any, _Registry()), []))
    finally:
        skills_api.default_backend_from_env = real

    resolver = captured.get("resolver")
    assert resolver is not None, "_setup_knowledge built no embedder backend"
    return resolver("OPENROUTER_API_KEY")  # type: ignore[operator]


def _list_logins_output(root: Path) -> str:
    """What ``lop login status`` prints for the store/environment tiers."""
    import contextlib
    import io

    from local_operator.providers.auth_cli import list_logins
    from local_operator.providers.auth_store import AuthStore

    store = AuthStore(db_path=root / "auth.db", config_dir=root)
    try:
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            list_logins(store, root)
    finally:
        store.close()
    return buffer.getvalue()


def _search_tool_digest(root: Path, *, store_value: str | None = None) -> str:
    """The singleflight digest a real ``execute_web_search`` call builds.

    A stand-in web-IO owner records the key and serves an empty response, so
    the tool's own digest path runs and nothing reaches the network.
    """
    import asyncio

    from local_operator.harness.types import ToolContext
    from local_operator.web_search import tool
    from local_operator.web_search.models import SearchResponse, WebSearchSettings

    if store_value is not None:
        from local_operator.providers.registry import store_provider_key

        store_provider_key("TAVILY_API_KEY", store_value, base=root)

    class _Owner:
        def __init__(self) -> None:
            self.keys: list[tuple[object, ...]] = []

        async def singleflight(self, key, run):
            self.keys.append(key)
            return SearchResponse(provider="tavily", auth_mode="test", sources=[])

    owner = _Owner()
    context = ToolContext(cwd=str(root), web_io=owner)  # type: ignore[arg-type]
    real_load = tool.load_search_settings
    tool.load_search_settings = lambda manager: WebSearchSettings(providers=["tavily"])
    try:
        result = asyncio.run(
            tool.execute_web_search(
                "digest-probe", {"query": "decoy digest probe"}, context=context
            )
        )
    finally:
        tool.load_search_settings = real_load
    assert not result.is_error, result.text
    assert owner.keys, "the tool did not take the singleflight path"
    return str(owner.keys[0][2])


def test_the_embedder_key_is_the_same_with_and_without_the_decoy_plaintext_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The knowledge/embedder leg of the sweep, driven through ``_setup_knowledge``.

    Equivalence rather than a falsy check: the resolver must answer the same
    thing whether or not the decoy file exists. Pre-PR2a it answered the decoy
    when the file was there, so the two runs differed.
    """
    root = tmp_path / "config"
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    root.mkdir(parents=True, exist_ok=True)
    without = _embedder_backend_credential(root)
    _write_decoy(root, "OPENROUTER_API_KEY")
    with_decoy = _embedder_backend_credential(root)

    assert without is None
    assert with_decoy == without


def test_the_embedder_key_still_resolves_a_provider_store_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The store rung the embedder kept (R6): a provider row still resolves."""
    root = tmp_path / "config"
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    _write_decoy(root, "OPENROUTER_API_KEY")

    assert _embedder_backend_credential(root, store_value="from-the-store") == "from-the-store"


def test_login_status_lists_no_key_from_the_decoy_plaintext_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``list_logins`` must print no key that only the decoy file holds."""
    root = tmp_path / "config"
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    _write_decoy(root, "OPENROUTER_API_KEY", "TAVILY_API_KEY")

    output = _list_logins_output(root)
    assert DECOY_PREFIX not in output
    assert "OPENROUTER_API_KEY" not in output
    assert "TAVILY_API_KEY" not in output
    assert "(none)" in output


def test_login_status_still_lists_a_provider_store_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The store rung ``list_logins`` kept: a store row is named, as a name only."""
    from local_operator.providers.registry import store_provider_key

    root = tmp_path / "config"
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    _write_decoy(root, "OPENROUTER_API_KEY")
    store_provider_key("OPENROUTER_API_KEY", "from-the-store", base=root)

    output = _list_logins_output(root)
    assert "secret store  OPENROUTER_API_KEY=<set>" in output
    assert "from-the-store" not in output


def test_the_search_digest_is_the_same_with_and_without_the_decoy_plaintext_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The web-search tool's per-call digest must not move with the file.

    The digest scopes duplicate-work suppression, so it covers EXPORTED values
    only. Equivalence again: pre-PR2a the tool digested the plaintext value, so
    adding the decoy file changed the key.
    """
    root = tmp_path / "config"
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    root.mkdir(parents=True, exist_ok=True)
    without = _search_tool_digest(root)
    _write_decoy(root, "TAVILY_API_KEY")
    with_decoy = _search_tool_digest(root)

    assert with_decoy == without


def test_the_search_digest_ignores_a_provider_store_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Deliberate: store rows resolve identically per config root, so they are
    not digested — only the per-call environment is."""
    root = tmp_path / "config"
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    root.mkdir(parents=True, exist_ok=True)
    without = _search_tool_digest(root)
    with_store = _search_tool_digest(root, store_value="from-the-store")

    assert with_store == without


def test_the_search_digest_follows_an_exported_variable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The env tier the digest DOES cover (R6), proven by movement."""
    root = tmp_path / "config"
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    root.mkdir(parents=True, exist_ok=True)
    without = _search_tool_digest(root)
    monkeypatch.setenv("TAVILY_API_KEY", "from-the-environment")
    with_export = _search_tool_digest(root)

    assert with_export != without


def test_no_reader_creates_the_plaintext_file(tmp_path: Path) -> None:
    """The recreator (defect B) is gone: reading a credential writes nothing.

    ``CredentialManager.__init__`` ran ``_ensure_config_exists``, which created
    an empty ``credentials.env`` on every plain construction — the defect that
    rewrote the operator's file daily. PR2b deletes the class outright, so the
    shape to pin is that the READERS leave the root untouched: the file must not
    come back on a host that has migrated.
    """
    from local_operator.providers.registry import provider_env_key
    from local_operator.secrets.legacy_env import read_credentials

    root = tmp_path / "config"
    root.mkdir()
    assert not (root / CREDENTIALS_FILE_NAME).exists()

    provider_env_key("openrouter", base=root)
    assert read_credentials(root) == {}


def test_the_credentials_module_is_gone_from_the_tree() -> None:
    """A guard on the guard: nothing may import the deleted module.

    PR2a replaced the 14 plain constructions that recreated the file; PR2b
    deletes ``local_operator/credentials.py`` entirely, so the shape to pin is
    that no production module imports it and the file itself is absent — a
    re-introduction would otherwise only be visible as a resurrected
    ``credentials.env`` on some path nothing exercises. Read at the AST level
    rather than by grep so a ``from . import credentials`` spelling counts too.
    """
    import ast
    from pathlib import Path

    package = Path(__file__).resolve().parents[3] / "local_operator"
    assert not (
        package / "credentials.py"
    ).exists(), "local_operator/credentials.py is back; PR2b deleted it"
    offenders: list[str] = []
    for path in sorted(package.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            module = getattr(node, "module", None) or ""
            if isinstance(node, ast.ImportFrom) and (
                module == "local_operator.credentials" or module == "credentials"
            ):
                offenders.append(f"{path.relative_to(package)}:{node.lineno} from {module}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "local_operator.credentials":
                        offenders.append(f"{path.relative_to(package)}:{node.lineno} import")
    assert offenders == [], "the deleted credentials module is imported: " + repr(offenders)


def test_the_module_is_not_importable() -> None:
    """The deleted module has no importable surface left.

    Belt and braces beside the AST sweep above: an import that the sweep's
    ``ImportFrom``/``Import`` handling missed — a dynamic ``importlib`` call, a
    string in a plugin registry — still fails here, because the MODULE is gone.
    """
    import importlib

    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("local_operator.credentials")


def test_the_decoy_value_is_not_ambient(tmp_path: Path) -> None:
    """Guard the guard: the decoy value must not exist in the environment.

    If ``DECOY_PREFIX`` were an exported variable the parametrised test above
    could pass for the wrong reason — the env tier, not the file, would be the
    source and the assertion would be vacuous.
    """
    assert not any(value.startswith(DECOY_PREFIX) for value in os.environ.values())
