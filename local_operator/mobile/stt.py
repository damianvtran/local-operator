"""Phone-facing speech-to-text: availability, dispatch, and failure copy.

This module is the daemon's whole view of the voice path, and it is deliberately
the ONE call site for the cascade session's resolver (``local_operator.stt``).
Two coroutines are the surface:

* :func:`stt_availability` — what the phone is told about voice input:
  ``{"available": bool, "path": str | None, "reason": str}``. The PATH is the
  cascade resolver's answer when their module is in the tree; it is then
  filtered for EXECUTABILITY — the table's ``servable`` flag, whether this build
  carries a way to run the token at all (``clients.stt.backend_ready``), and the
  persisted-credential rule (below). The filter is not priority logic: the
  resolver's choice is otherwise passed through verbatim, and when it names a
  token the phone cannot execute the answer becomes ``available: false`` with
  the row's own reason ("a mic that appears and fails is worse than one that
  does not").

  THE PERSISTED-CREDENTIAL RULE: availability keys on stored credentials only.
  The daemon is usually launched by a service manager whose environment the
  phone's user never chose, so an ambient ``RADIENT_API_KEY`` (or
  ``ELEVENLABS_API_KEY``) must not add voice input to a surface reachable over a
  tunnel. This is the model sheet's own precedent ("Only PERSISTED credentials
  authorize a listing here", ``daemon._list_models``). The CALL may still
  resolve store-first at call time; it is the ADVERTISING that stays strict.

  It never raises: an unreadable credential store or a resolver fault degrades
  to last-known-good (within the TTL) or to a hidden mic — never to a 500 on a
  list repaint.

* :func:`transcribe_audio` — the daemon route's only entry point. Reads
  availability, dispatches through ``clients/stt.py``, and raises
  ``SttBackendUnavailable`` for anything the route answers 503 to; upstream
  failures arrive as ``APIError`` and are classified by
  :func:`describe_stt_failure`.

WHY A LOCAL BASELINE EXISTS. Until the cascade module lands, ``local_operator.stt``
does not exist in the tree, and the Radient leg (ours, live) must still work —
so an absent resolver falls back to "first executable row in the operator-fixed
cascade order". The resolver, once present, ALWAYS wins. The fallback is the
transitional bridge, marked here so the eventual removal is one edit.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import re
import time
from typing import Any, Callable, Optional

from local_operator.clients._http import APIError
from local_operator.clients.stt import (
    STT_BACKENDS,
    SttBackendUnavailable,
    SttOutcome,
    backend_ready,
    resolve_backend_key,
    transcribe_with_backend,
)

logger = logging.getLogger(__name__)

#: How long one availability answer is served before the resolver (and, for the
#: BYO rows, the credential store) is consulted again. The list SSE repaints on
#: every scan tick; without a cache the daemon would read the encrypted store
#: per repaint, and the design's budget for that read is "once, occasionally".
STT_AVAILABILITY_TTL_S = 30.0

#: The resolver seam's module + symbol. The freeze the cascade session was
#: settling is SETTLED here (agents review convergence, B1): the module/symbol
#: below are the cascade's resolver as shipped, and :func:`_read_resolution`
#: reads its ``AudioPathResolution`` (path-driven; see there).
RESOLVER_MODULE = "local_operator.stt.cascade"
RESOLVER_ATTR = "resolve_audio_path"

_cached_answer: Optional[dict[str, Any]] = None
_cached_at: float = 0.0


def _reset_availability_cache() -> None:
    """Drop the memoised answer. Tests only — the daemon never clears it."""
    global _cached_answer, _cached_at
    _cached_answer = None
    _cached_at = 0.0


def _unavailable(reason: str, path: Optional[str] = None) -> dict[str, Any]:
    return {"available": False, "path": path, "reason": reason}


async def _call_maybe_async(fn: Callable[..., Any], /, **kwargs: Any) -> Any:
    """Call ``fn`` without parking the loop, tolerating a sync or async callee.

    The resolver's freeze does not promise one or the other, and a blocking
    SQLite/HTTP read on the daemon's event loop is the failure this shape
    removes: the CALL always runs on a worker thread, and a returned awaitable
    is awaited on the loop (it has not started yet, so nothing was lost).
    """
    result = await asyncio.to_thread(fn, **kwargs)
    if inspect.isawaitable(result):
        result = await result
    return result


def _cascade_resolver() -> Optional[Callable[..., Any]]:
    """The cascade resolver, or ``None`` when this tree predates it.

    ANY exception during the probe reads as absent — the daemon must keep
    serving the mic from whatever this build CAN run, and a half-importable
    resolver is not one it can vouch for.
    """
    try:
        module = __import__(RESOLVER_MODULE, fromlist=[RESOLVER_ATTR])
        resolver = getattr(module, RESOLVER_ATTR, None)
    except Exception as exc:  # noqa: BLE001 — deliberately fail-closed
        logger.debug("cascade STT resolver unavailable: %s", exc)
        return None
    return resolver if callable(resolver) else None


def _read_resolution(resolution: Any) -> tuple[Optional[str], bool, str]:
    """Read ``(token, available, reason)`` off the resolver's answer.

    SETTLED AGAINST THE CASCADE'S ``AudioPathResolution`` (the freeze this
    module's header reserved for the cascade session; agents review
    convergence, B1). The resolver owns ``path`` — an ``AudioPath`` member or
    its token string — and availability is DERIVED from it, never read off a
    field: ``AudioPath.NONE`` (the token ``"none"``) is the resolver saying
    "no path"; a token this build maps is available; a path this build cannot
    map is still a resolver CHOICE, reported available with no token so the
    caller answers "named a path this build cannot run".

    A shape with NO ``path`` at all is a resolver this build does not know,
    and it RAISES — a failed resolver takes the caller's last-known-good path
    and never an advertised yes. Attribute reads rather than an import of
    their class, so the seam stays readable from one place.
    """
    raw = getattr(resolution, "path", None)
    if raw is None:
        raise ValueError(f"unrecognised STT resolution shape: {type(resolution).__name__}")
    if isinstance(raw, str):
        text: Optional[str] = raw
    else:
        # Enum spellings: the StrEnum the cascade ships hits the branch above;
        # ``value``/``name`` cover a plain Enum or a shape-light double.
        text = None
        for attr in ("value", "name"):
            candidate = getattr(raw, attr, None)
            if isinstance(candidate, str):
                text = candidate
                break
    if text is None:
        raise ValueError(f"unrecognised STT resolution shape: {type(resolution).__name__}")
    reason = str(getattr(resolution, "reason", "") or "")
    if text.strip().lower() == "none":
        return None, False, reason
    return resolve_backend_key(raw), True, reason


def _stored_byo_credential(provider_id: str, config_root: Any) -> bool:
    """Whether a provider-class STORE ROW exists for ``provider_id``'s env key.

    Store row only, deliberately: ``provider_env_key``'s process-env leg is not
    a phone-facing gate (see the module docstring). A store that cannot be read
    answers ``False`` — "cannot tell" must not advertise.
    """
    from local_operator.providers.registry import env_key_name, stored_provider_env_keys

    env_key = env_key_name(provider_id)
    if not env_key:
        return False
    try:
        return env_key in stored_provider_env_keys(config_root)
    except Exception as exc:  # noqa: BLE001 — a broken store hides the mic, never 500s
        logger.warning("STT availability: provider store unreadable: %s", exc)
        return False


def _radient_persisted(config_root: Any) -> bool:
    """Whether Radient has a STORED credential (auth.db or provider row).

    The same reader the model sheet uses (``ProviderController.persisted_providers``),
    so the phone cannot advertise voice input on a credential the model sheet
    would refuse to list. ``None`` — an unopenable store — reads as not
    persisted; the honest direction for advertising.
    """
    from contextlib import closing

    from local_operator.providers.auth_store import AuthStore
    from local_operator.providers.controller import ProviderController

    try:
        # The DB path is derived from the root the caller named, matching
        # ``resolve_radient_credential``'s own construction: ``config_dir`` alone
        # would relocate only the env-tier store rows and leave the DB at the
        # HOME-derived default, which is exactly the split a "which store
        # authorized this listing" check must not have.
        db_path = (config_root / "auth.db") if config_root is not None else None
        store = AuthStore(db_path, config_dir=config_root)
    except Exception as exc:  # noqa: BLE001 — see the caller's contract
        logger.warning("STT availability: credential store unreadable: %s", exc)
        return False
    try:
        with closing(store):
            persisted = ProviderController(store, config_root).persisted_providers()
    except Exception as exc:  # noqa: BLE001
        logger.warning("STT availability: persisted-provider read failed: %s", exc)
        return False
    if persisted is None:
        return False
    return bool({"radient", "radient-key"} & set(persisted))


def _provider_label(provider_id: str) -> str:
    """The registry's human name for ``provider_id`` (the id when it has none)."""
    try:
        from local_operator.providers.registry import get_provider_definition

        definition = get_provider_definition(provider_id)
        if definition is not None and definition.name:
            return definition.name.split(" (")[0]
    except Exception:  # noqa: BLE001 — a label must never break availability
        pass
    return provider_id


def path_provider_label(path: Any) -> str:
    """Human provider label for a path token ("upstream" when none/unknown).

    The daemon route interpolates this into the upstream-refusal sentences, so
    a provider-credit answer names the provider whose account ran dry rather
    than the generic "upstream".
    """
    key = resolve_backend_key(path) if path else None
    if key is None:
        return "upstream"
    provider = STT_BACKENDS[key].provider
    return _provider_label(provider) if provider else "upstream"


def _executability(key: str, config_root: Any) -> tuple[bool, str]:
    """Whether the phone can actually RUN ``key``, and why not when it cannot."""
    row = STT_BACKENDS[key]
    if not row.servable:
        return False, row.reason or "Voice input is not available on the phone for this provider."
    if not backend_ready(key):
        # The BYO rungs need the cascade executor; without it in this tree the
        # rung is not executable, and availability never advertises what cannot
        # run (see clients/stt.py::byo_executor).
        return False, "Bring-your-own voice providers are not available in this build yet."
    if row.provider == "radient":
        if not _radient_persisted(config_root):
            return False, "Sign in to Radient to use voice input on the phone."
        return True, ""
    if row.provider:
        if not _stored_byo_credential(row.provider, config_root):
            label = _provider_label(row.provider)
            return False, f"No {label} API key is stored on this machine."
    return True, ""


async def _baseline_availability(config_root: Any) -> dict[str, Any]:
    """The pre-cascade bridge: first executable row, operator-fixed order.

    Used ONLY while ``local_operator.stt.cascade`` is absent from the tree (see
    the module docstring). It is deliberately the smallest possible claim: it
    walks the table in the operator-fixed cascade order and returns the first
    row that is servable, runnable by this build, and backed by a persisted
    credential. No live listing, no probing — executability only.
    """
    for key in STT_BACKENDS:
        ok, _why = _executability(key, config_root)
        if ok:
            return {"available": True, "path": key, "reason": ""}
    return _unavailable("No voice input provider is set up on this machine.")


async def _compute_availability(
    *,
    resolver: Optional[Callable[..., Any]],
    config_root: Any,
    store: Any,
) -> dict[str, Any]:
    """One fresh answer. Raises only for a FAILED resolver call (never absent)."""
    from local_operator.env import resolve_radient_api_base_url
    from local_operator.paths import config_dir as default_config_dir

    root = config_root if config_root is not None else default_config_dir()
    fn = resolver if resolver is not None else _cascade_resolver()
    if fn is None:
        return await _baseline_availability(root)

    resolution = await _call_maybe_async(
        fn,
        config_dir=root,
        base_url=resolve_radient_api_base_url(),
        store=store,
    )
    token, available, reason = _read_resolution(resolution)
    if not available:
        return _unavailable(reason or "Voice input is not available on this machine.")
    if token is None:
        # The resolver named a token this build does not know (or none at all).
        # Advertising a path we cannot map would be advertising something that
        # cannot run, which is the one thing the filter exists to prevent.
        return _unavailable("Voice input resolution named a path this build cannot run.")
    ok, why = _executability(token, root)
    if not ok:
        return _unavailable(why, path=token)
    return {"available": True, "path": token, "reason": ""}


async def stt_availability(
    *,
    resolver: Optional[Callable[..., Any]] = None,
    config_root: Any = None,
    store: Any = None,
    refresh: bool = False,
) -> dict[str, Any]:
    """The phone-facing answer: ``{available, path, reason}``, TTL-cached.

    ``resolver`` is the test seam (a callable with the frozen resolver
    signature); production passes nothing and the cascade module is imported
    lazily. ``refresh`` forces one fresh computation. Never raises.
    """
    global _cached_answer, _cached_at
    now = time.monotonic()
    if not refresh and _cached_answer is not None and (now - _cached_at) <= STT_AVAILABILITY_TTL_S:
        return dict(_cached_answer)
    try:
        answer = await _compute_availability(
            resolver=resolver, config_root=config_root, store=store
        )
    except Exception as exc:  # noqa: BLE001 — the contract is "never raises"
        logger.warning("STT availability: resolver failed: %s", exc)
        # Last-known-good, but only while it is still fresh; past its TTL a
        # stale yes would be a mic that appears and fails.
        if _cached_answer is not None and (now - _cached_at) <= STT_AVAILABILITY_TTL_S:
            return dict(_cached_answer)
        return _unavailable("Voice input is unavailable right now.")
    _cached_answer = dict(answer)
    _cached_at = time.monotonic()
    return dict(answer)


async def transcribe_audio(
    audio: bytes,
    mime: str,
    *,
    language: Optional[str] = None,
    prompt: Optional[str] = None,
    model: Optional[str] = None,
    backend: Optional[str] = None,
    availability: Optional[dict[str, Any]] = None,
    resolver: Optional[Callable[..., Any]] = None,
    config_root: Any = None,
    store: Any = None,
) -> SttOutcome:
    """The daemon route's ONE entry point: availability, then dispatch.

    ``backend`` is the test seam that overrides which token runs (it does not
    bypass the availability gate); ``availability`` overrides the read for a
    caller that already has one. Raises :class:`SttBackendUnavailable` when no
    executable path exists (the route answers 503), and passes upstream typing
    through unchanged for :func:`describe_stt_failure`.
    """
    answer = (
        availability
        if availability is not None
        else await stt_availability(resolver=resolver, config_root=config_root, store=store)
    )
    if not answer.get("available"):
        raise SttBackendUnavailable(
            "Voice input isn't available on this machine.",
            path=str(answer.get("path") or ""),
            reason=str(answer.get("reason") or ""),
        )
    path = backend if backend is not None else str(answer.get("path") or "")
    if resolve_backend_key(path) is None:
        raise SttBackendUnavailable("Voice input isn't available on this machine.", path=path)
    outcome = await transcribe_with_backend(
        path,
        audio,
        mime,
        language=language,
        prompt=prompt,
        model=model,
        config_root=config_root,
        store=store,
    )
    if not outcome.path:
        outcome = SttOutcome(
            text=outcome.text, provider=outcome.provider, model=outcome.model, path=path
        )
    return outcome


# ---------------------------------------------------------------------------
# Failure copy — a deliberate MIRROR of the desktop route's classifier.
# ---------------------------------------------------------------------------

# The sentences below are the desktop transcription route's own wording
# (``server/routes/transcription.py``), which is the surface that has been
# through the incident reviews for this exact upstream: a Radient 402 is a
# credit refusal, a provider quota marker is a 402 too, a 4xx is attributed by
# the body's envelope rather than by the status, and everything upstream that
# is not one of those is a 502 "not our fault". The copy is duplicated HERE
# rather than imported because that file is the cascade session's to change and
# its classifiers are private; the two copies must move together, and the tests
# on both sides pin the sentences.

#: Upstream text that means "the provider refused for want of credit". Mirrored
#: from the desktop route; the list is deliberately short and quota-specific so
#: an ordinary provider fault is not mislabelled as a billing problem.
PROVIDER_CREDIT_MARKERS = (
    "insufficient_quota",
    "insufficient quota",
    "exceeded your current quota",
    # The two vendors' own machine codes, added with the desktop table (voicing
    # S2): ElevenLabs reports an exhausted quota as ``status: "quota_exceeded"``
    # and OpenAI reports one as ``code: "credit_balance_exhausted"``. They
    # survive a translated or reworded message, which matters because the
    # condition arrives on a status that says nothing about credit. Kept in the
    # SAME ORDER as the desktop list so the two stay diff-identical.
    "quota_exceeded",
    "credit_balance_exhausted",
    "no credits remaining",
    "out of credits",
    "insufficient credits",
    "credit balance is too low",
)

#: Statuses that mean the request itself was refused (bad credential, unroutable
#: path, refused parameter). WHO refused it comes from the body — see
#: ``_refusing_side``.
PROVIDER_REJECTION_STATUSES = frozenset({400, 401, 403, 404, 422})

#: Statuses only Radient's own edge produces on this path.
RADIENT_EDGE_STATUSES = frozenset({401, 403, 404})

_RADIENT_ERROR_ENVELOPE = re.compile(r'\A\s*\{\s*"detail"\s*:')
_PROVIDER_ERROR_ENVELOPE = re.compile(r'\A\s*\{[^}]*?"error"\s*:', re.DOTALL)


def _upstream_clause(exc: APIError) -> str:
    """Render the upstream status and body as a trailing diagnostic clause.

    A 2xx is worded differently because it is not self-evidently a failure:
    Radient reports some provider failures in the body of a 200.
    """
    if exc.body:
        if exc.status_code is not None and 200 <= exc.status_code < 300:
            return f" Radient reported an error (HTTP {exc.status_code}): {exc.body}"
        return f" Upstream responded {exc.status_code}: {exc.body}"
    return f" Upstream responded {exc.status_code} with no body."


def _refusing_side(exc: APIError) -> str:
    """Name the hop that refused a 4xx request, from the body rather than the status.

    Radient's own edge is FastAPI and emits a ``detail`` envelope; a provider
    fault arrives in the provider's own envelope (a top-level ``error`` key).
    With no recognisable envelope the status alone attributes it: Radient's edge
    statuses to Radient, everything else to the provider it relayed to. The
    recorded residual (both directions) is documented on the desktop route; this
    mirror keeps it for the same reason — guessing a third signal no evidence
    supports is how the original misdirection happened.
    """
    body = exc.body or ""
    if _RADIENT_ERROR_ENVELOPE.match(body):
        return "radient"
    if _PROVIDER_ERROR_ENVELOPE.match(body):
        return "provider"
    return "radient" if exc.status_code in RADIENT_EDGE_STATUSES else "provider"


def _radient_rejection_detail(exc: APIError) -> str:
    """Say that Radient refused the request, and name the fix for the status."""
    if exc.status_code == 401:
        return (
            "Transcription is unavailable: Radient refused this app's credentials. "
            "Sign in again in the app; if it keeps failing, the daemon's Radient "
            "API key is invalid or has expired." + _upstream_clause(exc)
        )
    if exc.status_code == 403:
        return (
            "Transcription is unavailable: Radient refused the request (403). A 403 "
            "is a permission answer rather than a rejected credential, so signing in "
            "again is not the fix; check what the account is entitled to use, and "
            "the upstream's own words below." + _upstream_clause(exc)
        )
    if exc.status_code == 404:
        return (
            "Transcription is unavailable: the Radient endpoint the daemon is "
            "configured to call was not found. Check the daemon's Radient API base "
            "URL." + _upstream_clause(exc)
        )
    return "Transcription is unavailable: Radient rejected the request." + _upstream_clause(exc)


def describe_stt_failure(exc: APIError, provider: str = "") -> tuple[int, dict[str, str]]:
    """Map an upstream transcription failure onto ``(status, JSON body)``.

    The mobile route's own classifier, mirroring the desktop one sentence for
    sentence: 402 for a balance/quota refusal (Radient's own or a provider
    credit marker), 502 for everything upstream that is not ours, with the
    refusal attributed to the hop the BODY names. Returns a body carrying
    ``error`` (the copy) and ``code`` (the category, additive — every existing
    client reads ``error``).
    """
    if exc.status_code is None:
        # A transport failure never reached the upstream; the client's own text
        # ("Connection refused", "timed out") is the whole diagnostic.
        return 502, {"error": str(exc), "code": "stt_upstream"}

    if exc.status_code == 402:
        return 402, {
            "error": (
                "Transcription is unavailable: your Radient credit balance is too low. "
                "Add credits to continue." + _upstream_clause(exc)
            ),
            "code": "stt_quota",
        }

    body = (exc.body or "").lower()
    if any(marker in body for marker in PROVIDER_CREDIT_MARKERS):
        return 402, {
            "error": (
                f"Transcription is unavailable: the {provider} provider has run out of "
                f"credits. Switch to another provider, or add credits to your Radient "
                f"account." + _upstream_clause(exc)
            ),
            "code": "stt_quota",
        }

    if exc.status_code in PROVIDER_REJECTION_STATUSES:
        if _refusing_side(exc) == "radient":
            return 502, {
                "error": _radient_rejection_detail(exc),
                "code": "stt_upstream",
            }
        if exc.body is None:
            return 502, {
                "error": (
                    "The transcription request was refused upstream (HTTP "
                    f"{exc.status_code}) and the upstream sent no body saying by "
                    "whom. The daemon only authenticates to Radient, so this is "
                    f"either Radient's edge or the {provider} provider it called."
                    + _upstream_clause(exc)
                ),
                "code": "stt_upstream",
            }
        return 502, {
            "error": (
                f"The {provider} provider rejected the transcription request."
                + _upstream_clause(exc)
            ),
            "code": "stt_upstream",
        }

    return 502, {
        "error": "Transcription failed upstream." + _upstream_clause(exc),
        "code": "stt_upstream",
    }
