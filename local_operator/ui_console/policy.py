"""Console settings mirrored by the desktop host's admission consumer.

The desktop owns PTYs and enforces these values, even without a backend daemon.
These constants define the backend settings contract, not a second quota ledger.
The live ceiling is subject to managed resource admission and is never a promise
of arbitrary child RSS or permission to terminate already-running work.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

DEFAULT_LIVE_LIMIT = 128
DEFAULT_OWNER_LIMIT = 8
DEFAULT_USER_RESERVED = 16


def effective_policy(values: Mapping[str, Any]) -> dict[str, int]:
    """Clamp hand-edited YAML identically to the desktop consumer."""
    raw = values.get("console", {})
    data = raw if isinstance(raw, Mapping) else {}

    def clamp(key: str, default: int, low: int, high: int) -> int:
        value = data.get(key, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return default
        try:
            return max(low, min(high, int(value)))
        except (OverflowError, ValueError):
            return default

    live = clamp("max_live", DEFAULT_LIVE_LIMIT, 32, 256)
    return {
        "max_live": live,
        "max_owner": clamp("max_owner", DEFAULT_OWNER_LIMIT, 1, 32),
        "user_reserved": min(live // 4, clamp("user_reserved", DEFAULT_USER_RESERVED, 4, 32)),
    }
