"""RFC 8628 device-authorization polling, shared by Kimi and xAI logins.

Ported from omp ``registry/oauth/device-code.ts``. Semantics: minimum poll
interval 1 s; every ``slow_down`` adds 5 s to the interval; ``expired`` and
``denied`` are terminal; a dedicated timeout message calls out WSL/VM clock
drift because that is the most common real cause.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Awaitable, Callable, Generic, Literal, TypeVar

from local_operator.harness.types import AbortSignal
from local_operator.providers.oauth.callback_server import (
    LoginCancelledError,
    LoginError,
    LoginTimeoutError,
)

T = TypeVar("T")

MIN_INTERVAL_SECONDS = 1.0
SLOW_DOWN_INCREMENT_SECONDS = 5.0

DevicePollStatus = Literal["pending", "slow_down", "complete", "failed"]


class DevicePollResult(Generic[T]):
    """One poll outcome. ``status`` discriminates:

    - ``complete`` — ``value`` holds the token response.
    - ``pending`` — user has not authorized yet.
    - ``slow_down`` — provider asked for slower polling (+5 s).
    - ``failed`` — terminal; ``message`` explains (expired/denied/...).
    """

    __slots__ = ("status", "value", "message")

    def __init__(self, status: DevicePollStatus, value: T | None = None, message: str = "") -> None:
        self.status = status
        self.value = value
        self.message = message

    @staticmethod
    def pending() -> "DevicePollResult[T]":
        return DevicePollResult("pending")

    @staticmethod
    def slow_down() -> "DevicePollResult[T]":
        return DevicePollResult("slow_down")

    @staticmethod
    def complete(value: T) -> "DevicePollResult[T]":
        return DevicePollResult("complete", value=value)

    @staticmethod
    def failed(message: str) -> "DevicePollResult[T]":
        return DevicePollResult("failed", message=message)


PollFn = Callable[[], Awaitable[DevicePollResult[T]]]


async def poll_device_code_flow(
    poll_fn: PollFn[T],
    *,
    interval_seconds: float = 5.0,
    expires_in_seconds: float = 900.0,
    signal: AbortSignal | None = None,
    on_progress: Callable[[str], Awaitable[None] | None] | None = None,
) -> T:
    """Poll ``poll_fn`` until it completes, fails, expires, or is aborted.

    ``poll_fn`` must classify provider errors itself (authorization_pending →
    ``pending``, slow_down → ``slow_down``, expired_token/access_denied →
    ``failed``) and return :class:`DevicePollResult`.
    """
    interval = max(MIN_INTERVAL_SECONDS, float(interval_seconds))
    deadline = time.monotonic() + float(expires_in_seconds)
    first = True

    while True:
        if signal is not None and signal.aborted:
            raise LoginCancelledError(signal.reason or "Login cancelled")

        if not first:
            # Sleep in small slices so abort/expiry stay responsive.
            wait_until = time.monotonic() + interval
            while True:
                remaining = wait_until - time.monotonic()
                if remaining <= 0:
                    break
                if time.monotonic() >= deadline:
                    raise LoginTimeoutError()
                if signal is not None:
                    try:
                        await asyncio.wait_for(signal.wait(), timeout=min(remaining, 1.0))
                    except TimeoutError:
                        continue
                    raise LoginCancelledError(signal.reason or "Login cancelled")
                await asyncio.sleep(min(remaining, 1.0))
        first = False

        if time.monotonic() >= deadline:
            raise LoginTimeoutError()

        result = await poll_fn()
        if result.status == "complete":
            return result.value  # type: ignore[return-value]
        if result.status == "failed":
            raise LoginError(result.message or "Device authorization failed")
        if result.status == "slow_down":
            interval += SLOW_DOWN_INCREMENT_SECONDS
        if on_progress is not None:
            note = on_progress(f"Waiting for authorization ({int(interval)}s poll)…")
            if asyncio.iscoroutine(note):
                await note
