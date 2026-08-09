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
import time
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


def provider_error_summary(error: BaseException) -> str:
    """Stable, user-readable reason retained while a credential is blocked.

    Provider payloads vary from a useful sentence to a bare HTTP code. Collapse
    whitespace and classify the cases an operator can act on, while preserving a
    bounded provider detail when it adds information. This string is persisted in
    the credential backoff row, so a later session can explain why every account
    is unavailable instead of replacing the original diagnosis with "temporary".
    """
    if not isinstance(error, ProviderError):
        detail = " ".join(str(error).split())[:240]
        return detail or "provider error"

    detail = " ".join(error.message.split())[:240]
    lowered = detail.lower()
    if is_auth_error(error):
        category = "authentication failed"
    elif is_usage_limit_error(error):
        if any(marker in lowered for marker in ("quota", "usage", "insufficient")):
            category = "usage quota exhausted"
        else:
            category = "rate limit reached"
    elif error.status is not None and error.status >= 500:
        category = "provider service error"
    else:
        category = "provider request failed"

    status = f" (HTTP {error.status})" if error.status is not None else ""
    if not detail or category in lowered:
        return f"{category}{status}"
    return f"{category}{status}: {detail}"


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
SUPPORTED_EFFORTS = frozenset({"minimal", "low", "medium", "high", "xhigh", "max"})


@dataclasses.dataclass(frozen=True)
class FallbackTarget:
    """One resolved cascade entry.

    ``selector`` keeps routing compatible with the existing
    ``provider/model`` chain format. ``effort`` is optional because many
    providers either do not expose reasoning levels or should use their model
    default.
    """

    selector: str
    effort: str | None = None


def _chain_specificity(key: str, selector: str) -> int | None:
    """Higher = more specific; ``None`` = no match."""
    if key == selector:
        return 1 << 30  # exact beats every wildcard
    if key.endswith("/*"):
        prefix = key[:-2]
        if selector.startswith(prefix + "/"):
            return len(prefix)
    return None


def resolve_chain(selector: str, chains: Mapping[str, Sequence[Any]]) -> list[Any] | None:
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


def _fallback_target(entry: Any) -> FallbackTarget | None:
    """Normalize a legacy selector string or a provider/model/effort mapping."""
    effort: str | None = None
    if isinstance(entry, str):
        selector = entry.strip()
    elif isinstance(entry, Mapping):
        provider = str(entry.get("provider") or "").strip()
        model = str(entry.get("model") or entry.get("model_id") or "").strip()
        selector = str(entry.get("selector") or "").strip()
        if not selector and provider and model:
            selector = f"{provider}/{model}"
        raw_effort = entry.get("effort")
        if raw_effort is not None:
            effort = str(raw_effort).strip().lower()
            if effort not in SUPPORTED_EFFORTS:
                return None
    else:
        return None
    provider, model_id = parse_selector(selector)
    if not provider or not model_id:
        return None
    return FallbackTarget(selector=selector, effort=effort)


def expand_fallback_targets(selector: str, chain: Sequence[Any]) -> list[FallbackTarget]:
    """Materialize configured entries into unique provider/model/effort targets.

    ``provider/*`` keeps the failing model id. A mapping may explicitly repeat
    the current selector with a different effort; that is a real fallback
    route, while an unchanged legacy string is still suppressed.
    """
    _, _, bare_id = selector.partition("/")
    targets: list[FallbackTarget] = []
    for entry in chain:
        target = _fallback_target(entry)
        if target is None:
            continue
        if target.selector.endswith("/*"):
            target = dataclasses.replace(target, selector=f"{target.selector[:-1]}{bare_id}")
        if target.selector == selector and target.effort is None:
            continue
        if target not in targets:
            targets.append(target)
    return targets


def expand_fallback_candidates(selector: str, chain: Sequence[Any]) -> list[str]:
    """Backward-compatible selector-only view of :func:`expand_fallback_targets`."""
    return [target.selector for target in expand_fallback_targets(selector, chain)]


# ---------------------------------------------------------------------------
# Backoff
# ---------------------------------------------------------------------------

BACKOFF_CAP_MS = 8000
BACKOFF_JITTER_FRACTION = 0.25
# Interactive sessions must never disappear into a provider's quota-reset
# window. A 429 can carry Retry-After values of many minutes or hours; sleeping
# that duration before credential rotation makes the TUI look hung and prevents
# configured fallback models from running. Short throttles get one same-key
# retry, while long waits rotate or surface immediately.
MAX_USAGE_RETRY_AFTER_MS = 30_000
MAX_SAME_CREDENTIAL_USAGE_RETRIES = 1


def backoff_delay_ms(base_delay_ms: int, attempt: int, *, rng: random.Random | None = None) -> int:
    """``min(base * 2^(attempt-1), 8000)`` with 25% downward jitter."""
    raw = min(base_delay_ms * (2 ** max(0, attempt - 1)), BACKOFF_CAP_MS)
    jitter_source = rng or random
    return max(0, int(raw - raw * BACKOFF_JITTER_FRACTION * jitter_source.random()))


def _same_credential_retry_allowed(
    error: ProviderError,
    transport_retries: int,
    retry: "RetrySettings",
) -> bool:
    if not error.retryable:
        return False
    if not is_usage_limit_error(error):
        return transport_retries < retry.max_retries
    if (error.retry_after_ms or 0) > MAX_USAGE_RETRY_AFTER_MS:
        return False
    return transport_retries < min(retry.max_retries, MAX_SAME_CREDENTIAL_USAGE_RETRIES)


@dataclasses.dataclass(frozen=True)
class RetrySettings:
    """The ``values.retry.*`` config surface."""

    enabled: bool = True
    max_retries: int = 10
    base_delay_ms: int = 500
    model_fallback: bool = True
    usage_aware_fallback: bool = False
    usage_reserve_percent: float = 10.0
    fallback_chains: Mapping[str, Sequence[Any]] = dataclasses.field(default_factory=dict)

    @staticmethod
    def from_settings(settings: Mapping[str, Any] | None) -> "RetrySettings":
        retry = (settings or {}).get("retry", {}) if isinstance(settings, Mapping) else {}
        if not isinstance(retry, Mapping):
            retry = {}
        chains = retry.get("fallbackChains", retry.get("fallback_chains", {}))
        if not isinstance(chains, Mapping):
            chains = {}
        reserve = retry.get("usageReservePercent", retry.get("usage_reserve_percent", 10.0))
        try:
            reserve_percent = min(100.0, max(0.0, float(reserve)))
        except (TypeError, ValueError):
            reserve_percent = 10.0
        return RetrySettings(
            enabled=bool(retry.get("enabled", True)),
            max_retries=int(retry.get("maxRetries", retry.get("max_retries", 10))),
            base_delay_ms=int(retry.get("baseDelayMs", retry.get("base_delay_ms", 500))),
            model_fallback=bool(retry.get("modelFallback", retry.get("model_fallback", True))),
            usage_aware_fallback=bool(
                retry.get("usageAwareFallback", retry.get("usage_aware_fallback", False))
            ),
            usage_reserve_percent=reserve_percent,
            fallback_chains=chains,
        )


RouteChangeHandler = Callable[[FallbackTarget, str], Awaitable[None] | None]


@dataclasses.dataclass
class FailoverRouteState:
    """Session-sticky fallback route with a primary-probe cooldown.

    A successful fallback stays active for later model calls in the same user
    message. At later message boundaries, quota-aware preflight may return to
    the primary only after ``primary_retry_at_ms``. Without that suppression a
    healthy quota endpoint would make a transport-broken primary consume the
    full prompt once per user message before failing over again.
    """

    active: FallbackTarget | None = None
    active_spec: ModelSpec | None = None
    primary_spec: ModelSpec | None = None
    on_change: RouteChangeHandler | None = None
    primary_retry_at_ms: int = 0

    async def activate(
        self,
        target: FallbackTarget,
        reason: str,
        *,
        cooldown_ms: int = 0,
        spec: ModelSpec | None = None,
    ) -> None:
        if cooldown_ms > 0:
            self.primary_retry_at_ms = max(
                self.primary_retry_at_ms,
                int(time.time() * 1000) + cooldown_ms,
            )
        if spec is not None:
            self.active_spec = spec
        if self.active == target:
            return
        self.active = target
        if self.on_change is None:
            return
        result = self.on_change(target, reason)
        if inspect.isawaitable(result):
            await result

    def primary_retry_due(self, now_ms: int | None = None) -> bool:
        now = int(time.time() * 1000) if now_ms is None else now_ms
        return now >= self.primary_retry_at_ms

    def clear(self) -> None:
        self.active = None
        self.active_spec = None
        self.primary_retry_at_ms = 0


def parse_selector(selector: str) -> tuple[str, str]:
    provider, _, model_id = selector.partition("/")
    return provider, model_id


def spec_for_target(
    base: ModelSpec,
    target: FallbackTarget,
    *,
    catalogue_credential: tuple[str | None, bool, str | None] | None = None,
) -> ModelSpec:
    """Build the fallback model's OWN spec, then carry only sampling choices.

    Cloning the primary spec kept its base URL, context window and capabilities;
    a cross-provider fallback could therefore send an OpenAI model to the
    Anthropic endpoint. Model metadata and transport identity belong to the
    target, while temperature/top-p remain session preferences.
    """
    from local_operator.model.configure import build_model_spec

    target_spec = build_model_spec(
        *parse_selector(target.selector),
        catalogue_credential=catalogue_credential,
    )
    return target_spec.model_copy(
        update={
            "temperature": base.temperature,
            "top_p": base.top_p,
            "reasoning_effort": target.effort,
        }
    )


def spec_for_selector(base: ModelSpec, selector: str) -> ModelSpec:
    """Backward-compatible selector-only wrapper."""
    return spec_for_target(base, FallbackTarget(selector))


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
    route_state: FailoverRouteState | None = None,
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
    primary_target = FallbackTarget(primary_selector, request.model.reasoning_effort)

    targets = [primary_target]
    if retry.enabled and retry.model_fallback:
        chain = resolve_chain(primary_selector, retry.fallback_chains)
        if chain:
            for candidate in expand_fallback_targets(primary_selector, chain):
                if candidate not in targets:
                    targets.append(candidate)
    if route_state is not None and route_state.active in targets:
        targets = targets[targets.index(route_state.active) :]

    last_error: ProviderError | None = None
    last_failed_provider: str | None = None
    clients: dict[tuple[str, str | None], "WireClient"] = {}
    rng = random.Random()

    for target in targets:
        selector = target.selector
        if signal is not None and signal.aborted:
            raise ProviderError(None, signal.reason or "aborted", retryable=False)

        provider, _model_id = parse_selector(selector)
        route_key = (selector, target.effort)
        client = clients.get(route_key)
        spec = request.model
        current_request = request
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
                    last_error = ProviderError(
                        None, f"No API key configured for provider '{provider}'", retryable=False
                    )
                    break
                error = None
                credential = (
                    (
                        access.access_token,
                        access.kind == "oauth",
                        access.account_id or access.org_id,
                    )
                    if access is not None
                    else (None, False, None)
                )
                spec = await asyncio.to_thread(
                    spec_for_target,
                    request.model,
                    target,
                    catalogue_credential=credential,
                )
                current_request = request.model_copy(update={"model": spec})
                if target == primary_target:
                    if route_state is not None:
                        route_state.primary_spec = spec
                elif route_state is not None:
                    cooldown_ms = (
                        max(60_000, last_error.retry_after_ms or 0) if last_error else 60_000
                    )
                    diagnosis = (
                        provider_error_summary(last_error) if last_error else "provider unavailable"
                    )
                    await route_state.activate(
                        target,
                        f"{last_failed_provider or request.model.provider} {diagnosis}",
                        cooldown_ms=cooldown_ms,
                        spec=spec,
                    )
            if client is None:
                built = client_for(spec)
                client = await built if inspect.isawaitable(built) else built
                clients[route_key] = client
            retry_same_key = False
            key = access.access_token if access is not None else None

            forwarded_any = False
            try:
                async for event in client.stream(current_request, key, oauth_access=access):
                    forwarded_any = True
                    yield event
                if route_state is not None and target == primary_target:
                    route_state.clear()
                return  # clean completion
            except asyncio.CancelledError:
                raise
            except ProviderError as exc:
                if forwarded_any:
                    raise  # partial output already reached the caller
                last_error = exc
                if not retry.enabled:
                    raise
                if _same_credential_retry_allowed(exc, transport_retries, retry):
                    # 5xx/network-style failures use the configured budget.
                    # Rate limits retry once only when the advertised delay is
                    # short; long quota resets rotate or surface immediately.
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
        last_failed_provider = provider

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
