"""Cancellation-safe ownership across desktop and local setup entry points.

Keep the existing desktop drain semantics in one place: shielding alone lets a
cancelled response release its mutation lease while blocking installation still
runs. Local CLI/TUI setup must not import a server router to share that rule.
"""

from __future__ import annotations

import asyncio
from typing import Any


async def join_owned(operation: "asyncio.Task[dict[str, Any]]") -> dict[str, Any]:
    """Await an owned operation, joining it even when THIS waiter is cancelled.

    The repeated shield-and-check loop is the point, and a bare
    ``await asyncio.shield(task)`` is NOT equivalent: shielding protects the TASK
    from cancellation, but the ``await`` itself is still interruptible, so a
    cancelled waiter returns while the transaction keeps running — and a second
    cancellation (a shutdown arriving twice, a client that goes away and the
    route task being torn down afterwards) would leave it unobserved with its
    filesystem work in flight. Here the waiter drains the task first and only
    then propagates the cancellation, and the operation's own exception is
    re-raised (``task.result()``) rather than swallowed — a failed move must be
    reported, not hidden behind a cancellation.
    """
    cancelled = False
    result: dict[str, Any]
    while True:
        try:
            result = await asyncio.shield(operation)
            break
        except asyncio.CancelledError:
            cancelled = True
            if operation.done():
                result = operation.result()
                break
    if cancelled:
        raise asyncio.CancelledError
    return result
