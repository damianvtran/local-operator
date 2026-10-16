"""Mint one Gemini Live ephemeral token (session-voice.md R2).

The lop server holds the Google key; Electron main holds the Live socket. This
module is the only place the two meet: it exchanges the operator's AI Studio key
for a single-use token whose setup is locked to :mod:`voice.live_config`, over
plain httpx REST (``POST v1beta/auth_tokens``). It adds no dependency (no
``google-genai``).

NEITHER SECRET IS EVER LOGGED, and the shape of this module is how that holds:

* the key travels only in the ``x-goog-api-key`` header, never in a URL (a URL
  is what access logs and exception reprs print);
* every failure is raised as :class:`LiveTokenError` with a sentence built here,
  with the key AND any token-shaped value in the upstream body masked
  (``redact_secrets``), and ``from None`` so the httpx exception, whose repr
  carries the request, is not chained into a traceback;
* nothing here calls ``logger`` with a value from the request or the response.
  ``tests/unit/voice/test_live_token.py`` captures every log record at DEBUG
  through a mint and a refusal and asserts neither secret appears.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

import httpx

from local_operator.clients._http import redact_secrets
from local_operator.voice import live_config

#: Google's Gemini API root. Overridable per call for tests only; a test passes
#: an ``httpx.MockTransport`` client and this host is never contacted.
GEMINI_API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

#: The mint is one small JSON round trip; a slow answer should fail the take,
#: not hold the push-to-talk press for a minute.
MINT_TIMEOUT_S = 15.0


class LiveTokenError(RuntimeError):
    """The mint failed. ``str(exc)`` is safe to show and to log: no secret is in it."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class LiveToken:
    """What the route returns to main. ``token`` is the secret; nothing else is."""

    token: str
    expires_at: str
    new_session_expires_at: str
    model: str

    def __repr__(self) -> str:
        # A dataclass repr would print the token wherever this object is logged
        # or lands in a traceback's locals.
        return (
            f"LiveToken(token=<redacted>, expires_at={self.expires_at!r}, "
            f"new_session_expires_at={self.new_session_expires_at!r}, model={self.model!r})"
        )


async def mint(
    api_key: str,
    *,
    now: datetime,
    client: Optional[httpx.AsyncClient] = None,
    base_url: str = GEMINI_API_BASE_URL,
) -> LiveToken:
    """Exchange ``api_key`` for one locked, single-use Live token."""
    body = live_config.token_request_body(now)
    url = f"{base_url.rstrip('/')}/auth_tokens"
    headers = {"x-goog-api-key": api_key, "Content-Type": "application/json"}
    try:
        if client is not None:
            response = await client.post(
                url, json=body, headers=headers, timeout=httpx.Timeout(MINT_TIMEOUT_S)
            )
        else:
            async with httpx.AsyncClient() as owned:
                response = await owned.post(
                    url, json=body, headers=headers, timeout=httpx.Timeout(MINT_TIMEOUT_S)
                )
    except httpx.HTTPError as exc:
        # The class name only: an httpx error's text can carry the request.
        raise LiveTokenError(f"Google could not be reached ({type(exc).__name__}).") from None
    payload = _json_object(response)
    if response.status_code >= 400:
        raise LiveTokenError(
            _refusal_sentence(response.status_code, payload, secrets=[api_key]),
            status_code=response.status_code,
        )
    token = payload.get("name") if payload is not None else None
    if not isinstance(token, str) or not token:
        raise LiveTokenError("Google answered without a token.", status_code=response.status_code)
    return LiveToken(
        token=token,
        expires_at=str(payload.get("expireTime") or body["expireTime"]),
        new_session_expires_at=str(
            payload.get("newSessionExpireTime") or body["newSessionExpireTime"]
        ),
        model=live_config.LIVE_MODEL,
    )


def _json_object(response: httpx.Response) -> dict[str, Any] | None:
    try:
        payload = response.json()
    except ValueError:
        return None
    return payload if isinstance(payload, dict) else None


def _refusal_sentence(status: int, payload: dict[str, Any] | None, *, secrets: list[str]) -> str:
    """Google's own error message, masked, or the status alone.

    Only ``error.message`` is quoted, never the whole body, and it is masked
    against the key: an upstream that echoes the request must not echo it here.
    """
    error = (payload or {}).get("error")
    message = error.get("message") if isinstance(error, dict) else None
    if isinstance(message, str) and message.strip():
        detail = redact_secrets(message.strip()[:300], secrets)
        return f"Google refused the token request (HTTP {status}): {detail}"
    return f"Google refused the token request (HTTP {status})."


__all__ = ["GEMINI_API_BASE_URL", "LiveToken", "LiveTokenError", "mint"]
