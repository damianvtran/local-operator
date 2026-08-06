"""Credential rotation, model fallback chains, and failover streaming.

Three independent tiers from the provider-rotation design (§5):

- **Tier 1** — a/b/c credential rotation inside one provider
  (:func:`resolve_next_key`): initial resolve / force-refresh same account /
  rotate to a sibling. 403 and usage-limit errors skip the refresh step
  (a valid-but-denied token cannot be fixed by refreshing). Attempted keys
  are tracked in a set and capped at 64 so sibling pools cannot loop.
- **Tier 2** — model fallback chains (:func:`resolve_chain` /
  :func:`expand_fallback_candidates`), configured in config.yml
  ``values.retry.fallbackChains``.
- **High level** — :func:`stream_with_failover` composes both: rotate the
  credential, then walk the chain, backing off between attempts.
"""

from __future__ import annotations

import asyncio
import dataclasses
import inspect
import random
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from local_operator.harness.types import (
    AbortSignal,
    ChatRequest,
    ModelSpec,
    RenderedStreamError,
    StreamEvent,
)

if TYPE_CHECKING:  # import cycle: both modules import this one at runtime
    from local_operator.providers.auth_store import OAuthAccess
    from local_operator.providers.clients import WireClient

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ProviderError(RenderedStreamError):
    """A provider call failed. ``status`` is the HTTP status when known.

    ``retryable`` reflects whether the SAME credential may succeed again
    (429/5xx/network); auth errors are retryable only via rotation.

    A provider's answer, not a defect: ``RenderedStreamError`` tells the loop
    that ``__str__`` below is the complete diagnosis, so it logs the line
    without a stack.
    """

    def __init__(
        self,
        status: int | None,
        message: str,
        *,
        retryable: bool = False,
        retry_after_ms: int | None = None,
        auth_error: bool = False,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.message = message
        self.retryable = retryable
        self.retry_after_ms = retry_after_ms
        self.auth_error = auth_error

    def __str__(self) -> str:
        prefix = f"HTTP {self.status}: " if self.status is not None else ""
        return f"{prefix}{self.message}"


def is_auth_error(error: BaseException) -> bool:
    """401/403-class failures (bearer rejected or denied)."""
    if isinstance(error, ProviderError):
        return error.auth_error or error.status in (401, 403)
    return False


def is_usage_limit_error(error: BaseException) -> bool:
    """429 or provider-reported quota/rate exhaustion."""
    if not isinstance(error, ProviderError):
        return False
    if error.status == 429:
        return True
    lowered = error.message.lower()
    return any(
        marker in lowered
        for marker in (
            "usage",
            "quota",
            "rate limit",
            "rate_limit",
            "limit reached",
            "insufficient",
        )
    )


def is_invalidated_credential_error(error: BaseException) -> bool:
    """The credential was EXPLICITLY revoked by the IdP — soft-delete worthy.

    Narrow on purpose (PR-03): only true invalidation signals qualify —
    anthropic ``invalid_request_error`` + ``revoked``, openai
    ``token_revoked``/``invalid_grant``, or a generic ``revoked`` /
    ``invalid_grant`` payload. An ordinary expired-token 401 is NOT
    invalidated: it goes through the refresh step (b) and the row stays
    enabled. Generic "invalid"/"unauthorized"/"expired" 401s never
    soft-delete.
    """
    if not isinstance(error, ProviderError):
        return False
    if error.status != 401:
        return False
    lowered = error.message.lower()
    if "invalid_grant" in lowered or "token_revoked" in lowered:
        return True
    # anthropic surfaces revocation as invalid_request_error + "revoked";
    # "revoked" alone is the generic marker both shapes share.
    return "revoked" in lowered


def is_direct_credential_rotation_error(error: BaseException) -> bool:
    """Skip the refresh-same-account step for these: refreshing a
    valid-but-denied token cannot help, so rotate through the pool."""
    return is_usage_limit_error(error) or (isinstance(error, ProviderError) and error.status == 403)


def retry_after_ms_from_error(error: BaseException) -> int | None:
    if isinstance(error, ProviderError):
        return error.retry_after_ms
    return None


# ---------------------------------------------------------------------------
# Tier 1 — a/b/c credential rotation
# ---------------------------------------------------------------------------

AUTH_RETRY_MAX_ATTEMPTS = 64


@dataclasses.dataclass
class ApiKeyResolveContext:
    """What the resolver needs to pick a key: ``error is None`` ⇒ initial
    resolve; ``last_chance`` ⇒ rotate to a sibling credential."""

    last_chance: bool
    error: BaseException | None = None
    previous_key: str | None = None


ApiKeyResolver = Callable[[ApiKeyResolveContext], Awaitable[str | None] | str | None]


@dataclasses.dataclass
class AuthRetryKeyState:
    """Mutable rotation state shared across attempts for one request.

    ``legacy_auth_switch_used`` carries the legacy auth-switch semantics: an
    ORDINARY 401 gets exactly one refresh-same-account plus one sibling
    switch, then rotation is exhausted. Usage-limit/403 errors skip the
    refresh step and may cycle every distinct sibling instead.
    """

    attempted_keys: set[str] = dataclasses.field(default_factory=set)
    last_key: str | None = None
    refreshed_current: bool = False
    legacy_auth_switch_used: bool = False
    attempts: int = 0


async def _call_resolver(resolver: ApiKeyResolver, ctx: ApiKeyResolveContext) -> str | None:
    """Call a resolver that may be sync or async and normalise the result."""
    result = resolver(ctx)
    if inspect.isawaitable(result):
        return await result
    return result


async def resolve_next_key(
    state: AuthRetryKeyState,
    resolver: ApiKeyResolver,
    error: BaseException | None = None,
    *,
    signal: AbortSignal | None = None,
) -> str | None:
    """Next key to try, or ``None`` when rotation is exhausted.

    - ``error is None`` → (a) initial resolve (cheap, cached token OK).
    - ``error`` → (b) force-refresh the same account, then (c) rotate to a
      sibling — except 403/usage-limit skip (b) entirely.

    Termination: key already attempted, ≥64 attempts, resolver gives up, the
    signal aborted, or — for an ordinary 401 — the single refresh + single
    sibling switch (``legacy_auth_switch_used``) has been spent.
    """
    if signal is not None and signal.aborted:
        return None
    if state.attempts >= AUTH_RETRY_MAX_ATTEMPTS:
        return None
    state.attempts += 1

    async def _accept(key: str | None) -> str | None:
        if key is None or key in state.attempted_keys:
            return None
        state.attempted_keys.add(key)
        state.last_key = key
        return key

    if error is None:
        return await _accept(
            await _call_resolver(resolver, ApiKeyResolveContext(last_chance=False))
        )

    direct_rotation = is_direct_credential_rotation_error(error)
    # Ordinary 401: one refresh + one sibling switch, then stop
    # (legacyAuthSwitchUsed). Usage-limit/403 may keep cycling siblings.
    if not direct_rotation and state.legacy_auth_switch_used:
        return None

    # (b) Force-refresh the same account — skipped for valid-but-denied errors.
    if not state.refreshed_current and not direct_rotation:
        state.refreshed_current = True
        key = await _accept(
            await _call_resolver(
                resolver,
                ApiKeyResolveContext(last_chance=False, error=error, previous_key=state.last_key),
            )
        )
        if key is not None:
            return key

    # (c) Rotate to a sibling credential.
    if not direct_rotation:
        state.legacy_auth_switch_used = True
    return await _accept(
        await _call_resolver(
            resolver,
            ApiKeyResolveContext(last_chance=True, error=error, previous_key=state.last_key),
        )
    )


# ---------------------------------------------------------------------------
# Tier 2 — model fallback chains
# ---------------------------------------------------------------------------

DEFAULT_CHAIN_KEY = "default"


def _chain_specificity(key: str, selector: str) -> int | None:
    """Higher = more specific; ``None`` = no match."""
    if key == selector:
        return 1 << 30  # exact beats every wildcard
    if key.endswith("/*"):
        prefix = key[:-2]
        if selector.startswith(prefix + "/"):
            return len(prefix)
    return None


def resolve_chain(selector: str, chains: Mapping[str, Sequence[str]]) -> list[str] | None:
    """Pick the fallback chain for ``selector`` by specificity:
    exact ``provider/model`` → longest matching wildcard prefix → ``default``.
    """
    best_key: str | None = None
    best_score = -1
    for key in chains:
        if key == DEFAULT_CHAIN_KEY:
            continue
        score = _chain_specificity(key, selector)
        if score is not None and score > best_score:
            best_score = score
            best_key = key
    if best_key is None:
        if DEFAULT_CHAIN_KEY in chains:
            return list(chains[DEFAULT_CHAIN_KEY])
        return None
    return list(chains[best_key])


def expand_fallback_candidates(selector: str, chain: Sequence[str]) -> list[str]:
    """Materialize chain entries into concrete selectors.

    - Plain entries are kept as-is.
    - ``provider/*`` keeps the failing model id and swaps the provider.
    - Id-prefixed wildcards (``openrouter/google/*``) re-prefix the bare id.
    The current selector is never emitted as its own fallback.
    """
    _, _, bare_id = selector.partition("/")
    candidates: list[str] = []
    for entry in chain:
        if entry.endswith("/*"):
            candidate = f"{entry[:-1]}{bare_id}"
        else:
            candidate = entry
        if candidate and candidate != selector and candidate not in candidates:
            candidates.append(candidate)
    return candidates


# ---------------------------------------------------------------------------
# Backoff
# ---------------------------------------------------------------------------

BACKOFF_CAP_MS = 8000
BACKOFF_JITTER_FRACTION = 0.25


def backoff_delay_ms(base_delay_ms: int, attempt: int, *, rng: random.Random | None = None) -> int:
    """``min(base * 2^(attempt-1), 8000)`` with 25% downward jitter."""
    raw = min(base_delay_ms * (2 ** max(0, attempt - 1)), BACKOFF_CAP_MS)
    jitter_source = rng or random
    return max(0, int(raw - raw * BACKOFF_JITTER_FRACTION * jitter_source.random()))


@dataclasses.dataclass(frozen=True)
class RetrySettings:
    """The ``values.retry.*`` config surface (defaults from the established harness)."""

    enabled: bool = True
    max_retries: int = 10
    base_delay_ms: int = 500
    model_fallback: bool = True
    fallback_chains: Mapping[str, Sequence[str]] = dataclasses.field(default_factory=dict)

    @staticmethod
    def from_settings(settings: Mapping[str, Any] | None) -> "RetrySettings":
        retry = (settings or {}).get("retry", {}) if isinstance(settings, Mapping) else {}
        if not isinstance(retry, Mapping):
            retry = {}
        chains = retry.get("fallbackChains", retry.get("fallback_chains", {}))
        if not isinstance(chains, Mapping):
            chains = {}
        return RetrySettings(
            enabled=bool(retry.get("enabled", True)),
            max_retries=int(retry.get("maxRetries", retry.get("max_retries", 10))),
            base_delay_ms=int(retry.get("baseDelayMs", retry.get("base_delay_ms", 500))),
            model_fallback=bool(retry.get("modelFallback", retry.get("model_fallback", True))),
            fallback_chains=chains,
        )


def parse_selector(selector: str) -> tuple[str, str]:
    provider, _, model_id = selector.partition("/")
    return provider, model_id


def spec_for_selector(base: ModelSpec, selector: str) -> ModelSpec:
    """Clone ``base`` with the provider/model swapped in (knobs carry over)."""
    provider, model_id = parse_selector(selector)
    return base.model_copy(update={"provider": provider, "model_id": model_id})


# ---------------------------------------------------------------------------
# High-level streaming with failover
# ---------------------------------------------------------------------------

# Factories may be sync (the usual case) or async, so the driver awaits when
# it has to. ``WireClient`` is quoted: clients.py imports this module.
ClientFactory = Callable[[ModelSpec], "WireClient | Awaitable[WireClient]"]


@runtime_checkable
class FailoverAuthStore(Protocol):
    """The credential-store slice the failover driver needs.

    Structural rather than the concrete ``AuthStore``: that module imports
    this one, and hosts (plus test doubles) supply only these members.
    """

    async def get_api_key(
        self, provider: str, session_id: str | None = None, *, force_refresh: bool = False
    ) -> str | None: ...  # pragma: no cover

    def rotate_sibling(
        self,
        provider: str,
        session_id: str | None,
        error: BaseException,
        api_key: str | None = None,
    ) -> bool: ...  # pragma: no cover


@runtime_checkable
class OAuthAccessSource(Protocol):
    """A store that can also hand back the identity-carrying OAuth record.

    Deliberately NOT a subclass of :class:`FailoverAuthStore`: the record is
    an optional capability, so the driver tests for this ONE member with an
    ``isinstance`` against this runtime-checkable protocol. A store exposing
    only ``get_api_key`` yields bare bearers instead.
    """

    async def get_oauth_access(
        self, provider: str, session_id: str | None = None, *, force_refresh: bool = False
    ) -> "OAuthAccess | None": ...  # pragma: no cover


def _selector_for_request(request: ChatRequest) -> str:
    return f"{request.model.provider}/{request.model.model_id}"


async def _abortable_sleep(delay_ms: int, signal: AbortSignal | None) -> None:
    """Backoff sleep that returns early when the abort signal fires (PR-19)."""
    if delay_ms <= 0:
        return
    if signal is None:
        await asyncio.sleep(delay_ms / 1000)
        return
    loop = asyncio.get_running_loop()
    sleeper = loop.create_task(asyncio.sleep(delay_ms / 1000))
    abort_waiter = loop.create_task(signal.wait())
    try:
        done, _pending = await asyncio.wait(
            {sleeper, abort_waiter}, return_when=asyncio.FIRST_COMPLETED
        )
    finally:
        sleeper.cancel()
        abort_waiter.cancel()
    if abort_waiter in done:
        raise ProviderError(None, signal.reason or "aborted", retryable=False)


async def stream_with_failover(
    request: ChatRequest,
    auth: FailoverAuthStore,
    settings: Mapping[str, Any] | None,
    client_for: ClientFactory,
    *,
    session_id: str | None = None,
    signal: AbortSignal | None = None,
) -> AsyncIterator[StreamEvent]:
    """Stream one provider call with tier-1 + tier-2 failover.

    ONE rotation path (PR-04/05): failures are classified here, but every
    key decision — refresh-same-account (b), sibling rotation (c), the
    ordinary-401 single-switch cap — is delegated to
    :func:`resolve_next_key`. The driver never rotates directly.

    Transport-retryable errors (429/5xx/timeout/network) consume an
    INDEPENDENT retry budget (``retry.max_retries``) with backoff on the
    SAME credential before any rotation (PR-06); the budget resets when the
    credential changes.

    Events are forwarded as they arrive — once anything has been yielded, a
    mid-stream failure is re-raised (partial output cannot be replayed).
    Raises :class:`ProviderError` with the last failure when every option is
    spent.
    """
    retry = RetrySettings.from_settings(settings)
    primary_selector = _selector_for_request(request)

    selectors = [primary_selector]
    if retry.enabled and retry.model_fallback:
        chain = resolve_chain(primary_selector, retry.fallback_chains)
        if chain:
            selectors.extend(expand_fallback_candidates(primary_selector, chain))

    last_error: ProviderError | None = None
    clients: dict[str, "WireClient"] = {}
    rng = random.Random()

    for selector in selectors:
        if signal is not None and signal.aborted:
            raise ProviderError(None, signal.reason or "aborted", retryable=False)

        provider, _model_id = parse_selector(selector)
        spec = (
            request.model
            if selector == primary_selector
            else spec_for_selector(request.model, selector)
        )
        client = clients.get(selector)
        if client is None:
            built = client_for(spec)
            client = await built if inspect.isawaitable(built) else built
            clients[selector] = client
        current_request = (
            request if selector == primary_selector else request.model_copy(update={"model": spec})
        )

        state = AuthRetryKeyState()
        error: BaseException | None = None
        access: "OAuthAccess | None" = None  # credential record for this attempt
        current_token: str | None = None
        transport_retries = 0
        retry_same_key = False

        while state.attempts <= AUTH_RETRY_MAX_ATTEMPTS:
            if signal is not None and signal.aborted:
                raise ProviderError(None, signal.reason or "aborted", retryable=False)
            if not retry_same_key:
                access = await _resolve_access_for_provider(
                    auth, provider, session_id, state, error
                )
                token = access.access_token if access is not None else None
                if token != current_token:
                    current_token = token
                    transport_retries = 0  # fresh credential ⇒ fresh budget
                if access is None and error is not None:
                    break  # rotation exhausted for this provider
                if access is None and not _provider_allows_missing(provider):
                    last_error = last_error or ProviderError(
                        None, f"No API key configured for provider '{provider}'", retryable=False
                    )
                    break
                error = None
            retry_same_key = False
            key = access.access_token if access is not None else None

            forwarded_any = False
            try:
                async for event in client.stream(current_request, key, oauth_access=access):
                    forwarded_any = True
                    yield event
                return  # clean completion
            except asyncio.CancelledError:
                raise
            except ProviderError as exc:
                if forwarded_any:
                    raise  # partial output already reached the caller
                last_error = exc
                if not retry.enabled:
                    raise
                if exc.retryable and transport_retries < retry.max_retries:
                    # Transport-retryable: back off on the SAME credential
                    # BEFORE any rotation (PR-06).
                    transport_retries += 1
                    delay = max(
                        exc.retry_after_ms or 0,
                        backoff_delay_ms(retry.base_delay_ms, transport_retries, rng=rng),
                    )
                    await _abortable_sleep(delay, signal)
                    retry_same_key = True
                    continue
                if exc.retryable or exc.auth_error or exc.status in (401, 403):
                    # Delegate: (b) refresh same account, then (c) rotate —
                    # resolve_next_key owns the decision (PR-04/05).
                    error = exc
                    continue
                break  # non-retryable for this provider
            except Exception as exc:  # network errors et al.
                if forwarded_any:
                    raise
                wrapped = ProviderError(None, str(exc), retryable=True)
                last_error = wrapped
                if not retry.enabled:
                    raise wrapped from exc
                if transport_retries < retry.max_retries:
                    transport_retries += 1
                    await _abortable_sleep(
                        backoff_delay_ms(retry.base_delay_ms, transport_retries, rng=rng), signal
                    )
                    retry_same_key = True
                    continue
                error = wrapped
                continue

        # Provider exhausted — walk on to the next fallback selector.

    if last_error is not None:
        raise last_error
    raise ProviderError(None, f"Failover exhausted for '{primary_selector}'", retryable=False)


async def _resolve_access_for_provider(
    auth: FailoverAuthStore,
    provider: str,
    session_id: str | None,
    state: AuthRetryKeyState,
    error: BaseException | None,
) -> "OAuthAccess | None":
    """Bridge AuthStore into the a/b/c resolver shape, returning the
    :class:`~local_operator.providers.auth_store.OAuthAccess` record (or
    ``None``) so wire clients get identity headers alongside the bearer."""
    # Presence test, not a nominal one: stores exposing only get_api_key take
    # the bare-bearer path and get wrapped at the bottom of this function.
    oauth_store = auth if isinstance(auth, OAuthAccessSource) else None
    records: dict[str, "OAuthAccess"] = {}

    async def _access(*, force_refresh: bool = False) -> "OAuthAccess | None":
        if oauth_store is None:
            return None
        # force_refresh is only passed when set so stores declaring the bare
        # (provider, session_id) signature keep working.
        if force_refresh:
            return await oauth_store.get_oauth_access(provider, session_id, force_refresh=True)
        return await oauth_store.get_oauth_access(provider, session_id)

    async def resolver(ctx: ApiKeyResolveContext) -> str | None:
        try:
            if ctx.error is None:
                record = await _access()
                if record is None:
                    return await auth.get_api_key(provider, session_id)
            elif ctx.last_chance:
                auth.rotate_sibling(provider, session_id, ctx.error, api_key=ctx.previous_key)
                record = await _access()
                if record is None:
                    return await auth.get_api_key(provider, session_id)
            else:
                record = await _access(force_refresh=True)
                if record is None:
                    return await auth.get_api_key(provider, session_id, force_refresh=True)
        except Exception:
            return None
        if record is None:
            return None
        records[record.access_token] = record
        return record.access_token

    token = await resolve_next_key(state, resolver, error)
    if token is None:
        return None
    record = records.get(token)
    if record is None:
        # Auth stores without get_oauth_access (test fakes) yield bare keys;
        # wrap them so clients see one uniform shape.
        from local_operator.providers.auth_store import OAuthAccess

        record = OAuthAccess(access_token=token, credential_id=0, kind="api_key")
    return record


def _provider_allows_missing(provider: str) -> bool:
    """Providers that self-authenticate (ollama/test) need no key at all."""
    from local_operator.providers.registry import get_provider_definition

    definition = get_provider_definition(provider)
    return bool(definition and definition.allows_missing_api_key)
