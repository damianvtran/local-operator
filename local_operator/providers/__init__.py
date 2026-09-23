"""Provider / auth / failover layer of the rewritten harness.

See ``docs/REWRITE.md`` section B and ``docs/recon/ScoutProviders.md``. The
public surface consumed by other streams:

- :mod:`local_operator.providers.registry` — ``ProviderDefinition`` and
  ``PROVIDER_REGISTRY`` (every legacy ``--hosting`` name resolves).
- :mod:`local_operator.providers.auth_store` — SQLite credential store and the
  7-step API-key resolution cascade (``AuthStore``).
- :mod:`local_operator.providers.clients` — httpx wire clients streaming into
  harness ``StreamEvent``s.
- :mod:`local_operator.providers.failover` — credential rotation, model
  fallback chains, backoff math, ``stream_with_failover``.
- :mod:`local_operator.model.configure` — ``create_stream_fn`` builds the
  ``LoopConfig.stream_fn`` from an ``AuthStore``.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # the names `__getattr__` serves, for type checkers only
    from local_operator.providers.auth_store import (  # noqa: F401
        AuthStore,
        OAuthAccess,
        StoredCredential,
    )
    from local_operator.providers.clients import (  # noqa: F401
        AnthropicClient,
        GoogleClient,
        MockClient,
        OpenAICompatClient,
        WireClient,
    )
    from local_operator.providers.failover import (  # noqa: F401
        AuthRetryKeyState,
        ProviderError,
        ProviderErrorKind,
        RetrySettings,
        backoff_delay_ms,
        classify_provider_error,
        expand_fallback_candidates,
        is_timeout_error,
        is_transient_error,
        resolve_chain,
        resolve_next_key,
        stream_with_failover,
        wrap_transport_error,
    )
    from local_operator.providers.registry import (  # noqa: F401
        PROVIDER_REGISTRY,
        ProviderDefinition,
        env_key_name,
        get_provider_definition,
        list_login_providers,
        resolve_env_key,
    )

#: Which submodule each re-exported name lives in, and why they are resolved
#: LAZILY rather than imported above.
#:
#: Importing ANY ``local_operator.providers.<sub>`` runs this ``__init__``
#: first, and the eager re-exports made the whole provider stack -- the httpx
#: wire clients, failover, the auth store -- the price of every such import.
#: ``model.registry`` needs only ``providers.local`` (stdlib-only), and it sits
#: on BOTH startup paths: the TUI's status line reaches it through
#: ``model.naming``, and ``lop serve`` reaches it through the server schemas.
#: Measured on the TUI import, the provider layer was ~190 ms of it, paid
#: before first paint (backend load report B-F10). Nothing in this tree imports
#: these names from the package -- every caller names the submodule -- so the
#: re-exports are kept for compatibility, not for any caller's hot path.
#:
#: PEP 562 ``__getattr__`` keeps ``from local_operator.providers import
#: AuthStore`` working exactly as before, deferring the cost to first access --
#: the same pattern as ``local_operator/mcp/__init__.py``. Do not "tidy" this
#: back into module-scope imports; ``tests/unit/test_import_graph.py`` pins the
#: startup graphs that would regress.
_EXPORTS: dict[str, str] = {
    **dict.fromkeys(("AuthStore", "OAuthAccess", "StoredCredential"), "auth_store"),
    **dict.fromkeys(
        ("WireClient", "AnthropicClient", "GoogleClient", "MockClient", "OpenAICompatClient"),
        "clients",
    ),
    **dict.fromkeys(
        (
            "RetrySettings",
            "AuthRetryKeyState",
            "ProviderError",
            "ProviderErrorKind",
            "backoff_delay_ms",
            "classify_provider_error",
            "expand_fallback_candidates",
            "is_timeout_error",
            "is_transient_error",
            "resolve_chain",
            "resolve_next_key",
            "stream_with_failover",
            "wrap_transport_error",
        ),
        "failover",
    ),
    **dict.fromkeys(
        (
            "PROVIDER_REGISTRY",
            "ProviderDefinition",
            "env_key_name",
            "get_provider_definition",
            "list_login_providers",
            "resolve_env_key",
        ),
        "registry",
    ),
}


def __getattr__(name: str) -> Any:
    """Resolve a deferred re-export on first access, then cache it in the module dict.

    Only names in :data:`_EXPORTS` are resolved here. A SUBMODULE reached as an
    attribute (``local_operator.providers.clients``) needs no help: once any
    code has imported it, the import system has already bound it on this
    package, and before that it was never reachable as an attribute either --
    the eager imports only bound the four submodules they happened to name.
    """
    submodule = _EXPORTS.get(name)
    if submodule is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(f"{__name__}.{submodule}"), name)
    globals()[name] = value
    return value


__all__ = [
    "PROVIDER_REGISTRY",
    "AnthropicClient",
    "AuthRetryKeyState",
    "AuthStore",
    "GoogleClient",
    "MockClient",
    "OAuthAccess",
    "OpenAICompatClient",
    "ProviderDefinition",
    "ProviderError",
    "ProviderErrorKind",
    "backoff_delay_ms",
    "classify_provider_error",
    "env_key_name",
    "expand_fallback_candidates",
    "get_provider_definition",
    "is_timeout_error",
    "is_transient_error",
    "list_login_providers",
    "resolve_chain",
    "resolve_env_key",
    "resolve_next_key",
    "stream_with_failover",
    "wrap_transport_error",
]
