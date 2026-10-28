"""The credential-confined request builder for ``POST /v1/saver/prepare`` (§7, S-31, S-32).

WHY A DEDICATED BUILDER. Saver's whole safety claim to the user is one sentence:
"Saver never sends your AI provider's credentials to Radient." That is a
property of what bytes leave this process for the Radient host, so it is
enforced HERE, at the one place a Saver request is built, rather than trusted to
callers:

- its own ``httpx.AsyncClient(trust_env=False, follow_redirects=False)`` — the
  ``tunnels/setup.py`` precedent. Never a provider ``WireClient`` (which would add
  ``x-api-key``/``anthropic-beta``/its own bearer), never the session's shared
  pool, never ambient proxy or ``NETRC`` configuration;
- headers are built from an ALLOWLIST: ``Content-Type``, ``Authorization`` (the
  Radient bearer from ``resolve_radient_credential`` and nothing else),
  ``Idempotency-Key``, ``X-Radient-Client-Session-Id`` and the tier hint
  ``X-Radient-Saver-Model``. ``x-api-key`` and any ``anthropic-beta`` header can
  therefore never be present;
- the bearer is compared against EVERY stored provider credential except
  Radient's own; a match refuses the call (a provider key pasted into the
  Radient slot by mistake must not travel);
- an exact-host check (``canonical_radient_destination``): a non-canonical
  ``RADIENT_API_BASE_URL`` uploads nothing unless the operator set the same
  explicit opt-in organisation calls use (``RADIENT_ORG_ALLOW_NONCANONICAL_BASE``);
- a 3xx is NEVER followed: it is an answer like any other non-200, so the
  history and the bearer go to exactly one URL.

The response body is read as RAW transport bytes through a hard cap BEFORE any
JSON decoding, and no content decoder is ever built (§6.1 check 1, contract r6
S3-7); an error body is never read at all: only the status code leaves this
module (S-31: the server's text is never interpolated anywhere).
"""

from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from local_operator.saver.validate import MAX_RESPONSE_BYTES, SaverRejected, read_body

#: §6 latency budget: hard timeout 3 s, connect 0.5 s.
TIMEOUT = httpx.Timeout(3.0, connect=0.5)
PREPARE_PATH = "/saver/prepare"
HEADER_SESSION = "X-Radient-Client-Session-Id"
HEADER_TIER = "X-Radient-Saver-Model"
HEADER_IDEMPOTENCY = "Idempotency-Key"
#: The grammar the platform's correlation headers use; anything else is dropped.
_HEADER_VALUE = re.compile(r"[A-Za-z0-9._:/-]{1,128}")


class SaverRefused(Exception):
    """The request was not sent: no usable credential, or a refused destination.

    ``reason`` is a closed enum value (``signed_out``, ``noncanonical_base``,
    ``credential_conflict``).
    """

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class SaverHTTPError(Exception):
    """A non-200 answer. Carries the status and ``Retry-After`` only, never the body."""

    def __init__(self, status: int, retry_after_s: float | None = None) -> None:
        super().__init__(f"HTTP {status}")
        self.status = status
        self.retry_after_s = retry_after_s


@dataclass(frozen=True)
class PreparedResponse:
    payload: Any
    elapsed_ms: float


def prepare_url(base_url: str) -> str:
    """``<base>/saver/prepare``; ``base_url`` already carries the ``/v1`` segment."""
    return base_url.rstrip("/") + PREPARE_PATH


def destination_allowed(base_url: str) -> bool:
    """The exact-host rule, shared with organisation calls."""
    from local_operator.providers.radient_credentials import (
        org_oauth_destination_allowed,
    )

    return org_oauth_destination_allowed(base_url)


def stored_provider_secrets(config_dir: Path | None) -> set[str]:
    """Every NON-Radient provider credential value, for the S-32 comparison.

    Both places the inference cascade reads a provider key from: the auth
    store's rows (API keys and OAuth access/refresh tokens) and each registered
    provider's environment variables. Read in-process and compared in-process;
    nothing here is logged or returned beyond the comparison. Best effort per
    source: a store that cannot be read contributes nothing, and the allowlisted
    header builder still guarantees no provider header is ever attached.
    """
    from contextlib import closing

    from local_operator.providers.auth_store import AuthStore
    from local_operator.providers.registry import PROVIDER_REGISTRY, resolve_env_key

    radient = ("radient", "radient-key")
    values: set[str] = set()
    try:
        db_path = (config_dir / "auth.db") if config_dir is not None else None
        with closing(AuthStore(db_path) if db_path is not None else AuthStore()) as store:
            for row in store.list_credentials(include_disabled=True):
                if row.provider in radient:
                    continue
                for key in ("key", "access", "refresh", "api_key", "token"):
                    value = row.data.get(key)
                    if isinstance(value, str) and value:
                        values.add(value)
    except Exception:  # noqa: BLE001 — best effort, see the docstring
        pass
    for definition in PROVIDER_REGISTRY:
        if definition.id in radient:
            continue
        try:
            value = resolve_env_key(definition.id)
        except Exception:  # noqa: BLE001 — best effort, see the docstring
            continue
        if value:
            values.add(value)
    return values


def build_headers(
    bearer: str,
    *,
    session_id: str | None,
    idempotency_key: str | None,
    tier_hint: str | None,
    forbidden: Iterable[str] = (),
) -> dict[str, str]:
    """The ONLY headers a Saver request carries (S-32).

    Raises :class:`SaverRefused` when the bearer is empty or equals a stored
    provider credential.
    """
    if not bearer:
        raise SaverRefused("signed_out")
    if bearer in set(forbidden):
        raise SaverRefused("credential_conflict")
    headers = {
        "Content-Type": "application/json",
        # Contract r5 S3-1: identity only. httpx offers gzip/deflate by default
        # and Cloudflare may compress JSON for a client that does; refusing
        # compression removes the decompression-bomb surface outright.
        "Accept-Encoding": "identity",
        "Authorization": f"Bearer {bearer}",
    }
    for name, value in (
        (HEADER_SESSION, session_id),
        (HEADER_IDEMPOTENCY, idempotency_key),
        (HEADER_TIER, tier_hint),
    ):
        text = (value or "").strip()
        if text and _HEADER_VALUE.fullmatch(text):
            headers[name] = text
    return headers


async def _read_capped(response: httpx.Response, max_bytes: int) -> bytes:
    """The body, refused past ``max_bytes`` (R) while it streams (§6.1 check 1).

    Any ``Content-Encoding`` other than identity is refused UNREAD (contract r5
    S3-1): not one body byte is pulled, so no decoder exists to be fed. The
    chain turns that refusal into a process-wide pause (r6 S3-7 b).
    """
    encoding = response.headers.get("content-encoding", "").strip().lower()
    if encoding not in ("", "identity"):
        raise SaverRejected("bad_encoding")
    return await _read_raw_capped(response, max_bytes)


async def _read_raw_capped(response: httpx.Response, max_bytes: int) -> bytes:
    """RAW transport bytes, refused past ``max_bytes`` (contract r6 S3-7 a).

    ``aiter_raw`` and never ``aiter_bytes``/``aread``/``.content``: those make
    httpx pick a decoder from the response's OWN ``Content-Encoding``, which is
    exactly the attacker input the identity rule excludes. Its gzip decoder
    inflates each 64 KiB transport read whole, so a 65 KB gzip arrives as ONE
    64 MiB chunk before any counter could trip (measured, httpx 0.28.1). Under
    identity raw bytes ARE the body, so counting them loses nothing; and if
    the header check above ever regressed, an encoded body would still be
    capped at its wire size and then fail strict JSON, never inflated.
    """
    chunks: list[bytes] = []
    total = 0
    async for chunk in response.aiter_raw():
        total += len(chunk)
        if total > max_bytes:
            raise SaverRejected("too_large")
        chunks.append(chunk)
    return b"".join(chunks)


def _retry_after(response: httpx.Response) -> float | None:
    value = response.headers.get("retry-after")
    if not value:
        return None
    try:
        seconds = float(value)
    except ValueError:
        return None
    return seconds if 0 <= seconds <= 3600 else None


BearerSource = Callable[[bool], Awaitable[str]]


class SaverClient:
    """One process-wide Saver HTTP client; per-session state lives in the chain.

    ``bearer`` is called with ``force_refresh`` and returns the Radient bearer
    (``""`` when signed out). ``forbidden`` returns the stored provider secrets
    the bearer must not equal. Both are injectable so tests can prove the
    header rule without a real store.
    """

    def __init__(
        self,
        base_url: str,
        *,
        bearer: BearerSource,
        forbidden: Callable[[], Iterable[str]] = lambda: (),
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.base_url = base_url
        self._bearer = bearer
        self._forbidden = forbidden
        # trust_env=False: no proxy, NETRC or CA overrides from the environment
        # can redirect or observe this request. follow_redirects=False: a 3xx is
        # an answer, never a second destination for the history and the bearer.
        self._http = httpx.AsyncClient(
            trust_env=False,
            follow_redirects=False,
            timeout=TIMEOUT,
            transport=transport,
        )

    async def refresh(self) -> None:
        """Force ONE refresh of the Radient bearer through the shared store (§6, 401).

        The single-flight lock lives in the store, so this never races the
        inference path for the same grant. The refreshed token is what the next
        :meth:`prepare` resolves; nothing is retried from here.
        """
        await self._bearer(True)

    async def aclose(self) -> None:
        await self._http.aclose()

    async def prepare(
        self,
        body: dict[str, Any],
        *,
        session_id: str | None,
        idempotency_key: str | None,
        tier_hint: str | None,
        max_response_bytes: int = MAX_RESPONSE_BYTES,
    ) -> PreparedResponse:
        """One prepare.

        ``max_response_bytes`` is R for this upload
        (:func:`~local_operator.saver.validate.response_cap`). Raises
        :class:`SaverRefused`, :class:`SaverHTTPError`, :class:`SaverRejected`
        (size/encoding/JSON) or an ``httpx`` transport error.
        """
        import time

        if not destination_allowed(self.base_url):
            raise SaverRefused("noncanonical_base")
        bearer = await self._bearer(False)
        headers = build_headers(
            bearer,
            session_id=session_id,
            idempotency_key=idempotency_key,
            tier_hint=tier_hint,
            forbidden=self._forbidden(),
        )
        content = json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        started = time.perf_counter()
        request = self._http.build_request(
            "POST", prepare_url(self.base_url), content=content, headers=headers
        )
        response = await self._http.send(request, stream=True)
        try:
            if response.status_code != 200:
                # The body is NEVER read: a 3xx Location, an error sentence and a
                # proxy page are all equally untrusted (S-31).
                raise SaverHTTPError(response.status_code, _retry_after(response))
            raw = await _read_capped(response, max_response_bytes)
        finally:
            await response.aclose()
        return PreparedResponse(
            read_body(raw, max_bytes=max_response_bytes),
            (time.perf_counter() - started) * 1000,
        )


def radient_bearer_source(config_dir: Path | None, base_url: str) -> BearerSource:
    """The production bearer: ``resolve_radient_credential`` through the shared store.

    ``force_refresh`` is the §6 "401 ⇒ one refresh through the shared
    single-flight Radient refresh store" rung: it asks the same store the
    inference path uses, so two refreshes never race for one grant.
    """

    async def _resolve(force_refresh: bool) -> str:
        from local_operator.providers.radient_credentials import (
            _radient_auth_store,
            resolve_radient_credential,
        )

        if not force_refresh:
            secret = await resolve_radient_credential(config_dir, base_url)
            return secret.get_secret_value()
        store = _radient_auth_store(config_dir)
        try:
            value = await store.get_api_key("radient", read_only=True, force_refresh=True)
        finally:
            store.close()
        return value or ""

    return _resolve
