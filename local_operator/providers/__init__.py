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
    WireClient,  # noqa: F401
)
from local_operator.providers.failover import (  # noqa: F401
    AuthRetryKeyState,
    ProviderError,
    RetrySettings,  # noqa: F401
    backoff_delay_ms,
    expand_fallback_candidates,
    resolve_chain,
    resolve_next_key,
    stream_with_failover,
)
from local_operator.providers.registry import (
    PROVIDER_REGISTRY,
    ProviderDefinition,
    env_key_name,
    get_provider_definition,
    list_login_providers,
    resolve_env_key,
)

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
    "backoff_delay_ms",
    "env_key_name",
    "expand_fallback_candidates",
    "get_provider_definition",
    "list_login_providers",
    "resolve_chain",
    "resolve_env_key",
    "resolve_next_key",
    "stream_with_failover",
]
