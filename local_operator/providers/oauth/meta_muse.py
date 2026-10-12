"""Unofficial Muse Code subscription authentication, never Meta PAYG.

Protocol observations are second-party, not a supported Meta integration:
OpenCodex issue #3191 and its device-login implementation (September 2026).
This independently implemented flow exchanges the account bearer for a static
LLM inference key. The account bearer must not be persisted or used for chat.
"""

from __future__ import annotations

import asyncio
import json
import math
import re
import time
from typing import Any, Awaitable, Callable, TypeVar
from urllib.parse import parse_qsl, urlsplit

import httpx

from local_operator.harness.types import AbortSignal
from local_operator.providers.oauth.callback_server import (
    LoginCallbacks,
    LoginCancelledError,
    LoginError,
    LoginTimeoutError,
    maybe_await,
    report_flow_details,
    report_safely,
)
from local_operator.providers.oauth.device_code import (
    DevicePollResult,
    poll_device_code_flow,
)

CLIENT_ID = "1031625952748946"  # Public identifier in Meta's device approval URL.
AUTHORIZATION_URL = "https://auth.meta.com/oidc/device/authorization/"
TOKEN_URL = "https://auth.meta.com/oidc/device/token/"
KEY_URL = "https://api.meta.ai/muse-code/key"
NOTICE = (
    "Unofficial Muse Code subscription integration; unsupported by Meta outside its own CLI. "
    "Subscription eligibility is checked at login, not a guarantee of future billing. "
    "This route never falls back to Meta pay-as-you-go. "
    "Muse Contributor can use prompts, code and context for training by default. "
    "Disable Contributor in Meta/Muse settings before proprietary or customer use; "
    "lop does not change or verify that account setting."
)
MAX_BODY_BYTES = 64 * 1024
REQUEST_TIMEOUT_S = 20.0
KEY_PATTERN = re.compile(r"LLM\|\d+\|[A-Za-z0-9_-]{10,}")
T = TypeVar("T")


def validate_muse_key(value: str) -> str:
    """Accept the observed key grammar without accepting an ordinary PAYG key."""
    key = value.strip()
    if not KEY_PATTERN.fullmatch(key):
        raise LoginError(
            "Expected a Muse Code LLM subscription key; ordinary Meta API keys cannot be used."
        )
    return key


def _number(value: Any, default: float, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    number = float(value)
    return min(maximum, max(minimum, number)) if math.isfinite(number) else default


def _text(payload: dict[str, Any], name: str) -> str:
    value = payload.get(name)
    return value if isinstance(value, str) else ""


def _approval_url(value: str, device_code: str, user_code: str) -> str:
    # A provider response must not turn the device code into a phishing link
    # or a credential-bearing URL that every host will then print/log.
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError:
        raise LoginError("Meta returned an invalid device approval URL; restart login.") from None
    if (
        parsed.scheme != "https"
        or parsed.hostname != "auth.meta.com"
        or parsed.username
        or parsed.password
        or port not in (None, 443)
        or parsed.fragment
        or device_code in value
        or any(
            key not in ("user_code", "code", "client_id")
            or val != (CLIENT_ID if key == "client_id" else user_code)
            for key, val in parse_qsl(parsed.query, keep_blank_values=True)
        )
    ):
        raise LoginError("Meta returned an invalid device approval URL; restart login.")
    return value


async def _bounded(awaitable: Awaitable[T], signal: AbortSignal | None, budget: float) -> T:
    """Cancel and reap the HTTP task when the user cancels or its budget ends."""
    task = asyncio.ensure_future(awaitable)
    abort = asyncio.create_task(signal.wait()) if signal is not None else None
    try:
        if signal is not None and signal.aborted:
            raise LoginCancelledError("Muse Code login cancelled")
        done, _ = await asyncio.wait(
            {task, abort} if abort is not None else {task},
            timeout=budget,
            return_when=asyncio.FIRST_COMPLETED,
        )
        if abort is not None and abort in done:
            raise LoginCancelledError("Muse Code login cancelled")
        if task not in done:
            raise LoginTimeoutError("Muse Code request timed out; restart login.")
        return await task
    finally:
        for pending in (task, abort):
            if pending is not None and not pending.done():
                pending.cancel()
        pending_tasks: list[asyncio.Future[Any]] = [task]
        if abort is not None:
            pending_tasks.append(abort)
        await asyncio.gather(*pending_tasks, return_exceptions=True)


async def _request(
    http: httpx.AsyncClient,
    url: str,
    *,
    signal: AbortSignal | None,
    budget: float = REQUEST_TIMEOUT_S,
    **kwargs: Any,
) -> tuple[int, dict[str, Any], httpx.Headers]:
    async def read() -> tuple[int, dict[str, Any], httpx.Headers]:
        try:
            # Streaming bounds the decoded body too; Content-Length alone does
            # not protect against chunked responses or compressed expansion.
            async with http.stream("POST", url, follow_redirects=False, **kwargs) as response:
                body = bytearray()
                async for chunk in response.aiter_bytes():
                    if len(body) + len(chunk) > MAX_BODY_BYTES:
                        raise LoginError(
                            "Muse Code authentication response exceeded its size limit."
                        )
                    body.extend(chunk)
                try:
                    payload = json.loads(body)
                except (ValueError, UnicodeError):
                    payload = {}
                return (
                    response.status_code,
                    payload if isinstance(payload, dict) else {},
                    response.headers,
                )
        except httpx.HTTPError:
            # URLs and echoed bodies can carry credential material. Preserve no
            # raw HTTP exception text in the CLI, operation snapshot or logs.
            raise LoginError(
                "Muse Code authentication request failed; check your connection and retry."
            ) from None

    return await _bounded(read(), signal, budget)


async def login_meta_muse(
    callbacks: LoginCallbacks,
    *,
    signal: AbortSignal | None = None,
    http_client: httpx.AsyncClient | None = None,
    open_browser: Callable[[str], object] | None = None,
) -> dict[str, Any]:
    """Approve the device, verify subscription entitlement, return its inference key.

    AuthStore seals the returned key before persistence. There is deliberately
    no refresh function: the minted key is static and a refusal requires a new
    explicit login, not an unobservable account-token exchange or PAYG fallback.
    """
    if callbacks.displays_warnings:
        await report_safely(callbacks.on_warning or callbacks.on_progress, NOTICE)
    # A host that does not display warning prose gets the complete disclosure
    # in its approval instructions, not a misleading generic warning state.
    owns_http = http_client is None
    http = http_client or httpx.AsyncClient(timeout=REQUEST_TIMEOUT_S)
    headers = {"Accept": "application/json", "x-api-version": "1.0.0"}
    try:
        status, auth, _ = await _request(
            http, AUTHORIZATION_URL, signal=signal, headers=headers, data={"client_id": CLIENT_ID}
        )
        if status != 200:
            raise LoginError(
                f"Muse Code device authorization failed (HTTP {status}); restart login."
            )
        device_code, user_code = _text(auth, "device_code"), _text(auth, "user_code")
        if (
            not device_code
            or len(device_code) > 4096
            or not re.fullmatch(r"[A-Za-z0-9-]{1,64}", user_code)
            or device_code == user_code
        ):
            raise LoginError(
                "Meta returned an invalid device authorization response; restart login."
            )
        ttl = _number(auth.get("expires_in"), 900, 1, 1800)
        interval = _number(auth.get("interval"), 5, 1, 60)
        url = _approval_url(
            _text(auth, "verification_uri_complete")
            or _text(auth, "verification_uri")
            or "https://auth.meta.com/oauth/device/",
            device_code,
            user_code,
        )
        deadline = time.monotonic() + ttl
        await report_flow_details(callbacks, user_code=user_code, expires_in=ttl)
        instructions = f"Enter code: {user_code} in the browser."
        if not callbacks.displays_warnings:
            # Unknown/desktop hosts may not render warning callbacks. Keep the
            # complete disclosure there; terminal hosts already showed it once.
            instructions = f"{NOTICE}\n{instructions}"
        if callbacks.on_auth_url is not None:
            await maybe_await(callbacks.on_auth_url(url, instructions=instructions))
        # The TUI callback renders, but does not open, this URL. Keep the same
        # injected opener ownership as loopback flows; desktop passes a no-op
        # because its renderer opens the operation URL. Failed local opening
        # remains actionable through the always-visible approval link.
        import webbrowser

        opener = open_browser if open_browser is not None else webbrowser.open
        try:
            opened = await _bounded(maybe_await(opener(url)), signal, REQUEST_TIMEOUT_S)
        except LoginCancelledError:
            raise
        except Exception:
            await report_safely(
                callbacks.on_warning or callbacks.on_progress,
                "Browser could not open. Open the displayed approval page to continue.",
            )
        else:
            if opened is False:
                await report_safely(
                    callbacks.on_warning or callbacks.on_progress,
                    "Browser could not open. Open the displayed approval page to continue.",
                )
        intervals: list[float] = []

        async def poll() -> DevicePollResult[str]:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise LoginTimeoutError("Muse Code device approval expired; restart login.")
            code, token, response_headers = await _request(
                http,
                TOKEN_URL,
                signal=signal,
                budget=min(REQUEST_TIMEOUT_S, remaining),
                headers=headers,
                data={
                    "client_id": CLIENT_ID,
                    "device_code": device_code,
                    "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                },
            )
            error = _text(token, "error")
            if code == 200 and not error:
                access = _text(token, "access_token")
                if not access:
                    return DevicePollResult.failed("Meta returned no account token; restart login.")
                return DevicePollResult.complete(access)
            if error == "authorization_pending":
                return DevicePollResult.pending()
            if error == "slow_down" or code == 429:
                try:
                    wait = float(response_headers.get("retry-after", "0"))
                except ValueError:
                    wait = 0
                if math.isfinite(wait) and wait > 0:
                    intervals.append(min(60.0, wait))
                return DevicePollResult.slow_down()
            if error == "access_denied":
                return DevicePollResult.failed(
                    "Muse Code approval denied; restart login when ready."
                )
            if error == "expired_token":
                return DevicePollResult.failed("Muse Code device code expired; restart login.")
            return DevicePollResult.failed(
                f"Muse Code device token request failed (HTTP {code}); restart login."
            )

        account_token = await poll_device_code_flow(
            poll,
            interval_seconds=interval,
            expires_in_seconds=max(0, deadline - time.monotonic()),
            signal=signal,
            on_progress=callbacks.on_progress,
            interval_holder=intervals,
        )
        status, minted, _ = await _request(
            http,
            KEY_URL,
            signal=signal,
            headers={**headers, "Authorization": f"Bearer {account_token}"},
            json={"onboard": True},
        )
        if status != 200:
            raise LoginError(
                f"Muse Code subscription key exchange failed (HTTP {status}); restart login."
            )
        # Fail closed: a usable key with require_payment is NOT evidence of a
        # subscription-covered request. Never advertise that as a successful
        # subscription login or print an untrusted action_url from the response.
        if minted.get("is_subs_active") is not True or minted.get("require_payment") is not False:
            raise LoginError(
                "Meta did not confirm an active Muse Code subscription "
                "without payment requirements. "
                "Check your plan at https://dev.meta.ai and retry; no PAYG route was selected."
            )
        key = validate_muse_key(_text(minted, "api_key"))
        account_id = _text(minted, "user_id")
        email = _text(minted, "user_email").strip().lower()
        # Identity is public metadata, not another token sink. Reject reflected
        # credentials and control text before it can enter auth.db/status UIs.
        if (
            len(account_id) > 256
            or any(ord(c) < 32 for c in account_id)
            or account_token in account_id
            or key in account_id
        ):
            account_id = ""
        if (
            len(email) > 254
            or not re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", email)
            or account_token.lower() in email
            or key.lower() in email
        ):
            email = ""
        if not account_id and not email:
            raise LoginError(
                "Meta returned no account identity for the subscription key; restart login."
            )
        from local_operator.providers.meta_muse_usage import subscription_observation

        observation = subscription_observation(minted.get("subs_usage"))
        return {
            **({"subscription_observation": observation} if observation is not None else {}),
            "type": "oauth",
            "access": key,
            "source": "device-approval",
            "billing_provenance": "muse-code-subscription",
            **({"account_id": account_id} if account_id else {}),
            **({"email": email} if email else {}),
        }
    finally:
        if owns_http:
            await http.aclose()
