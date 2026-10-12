"""Safe allowance receipts shared by Muse key mint and passive Responses SSE.

The observed wire shape is subscription.{window,weekly} on an SSE event named
response.subscription_usage, and subs_usage on key mint (OpenCodex observations,
September 2026). There is no verified read-only quota endpoint. We never spend
an inference request to refresh usage, nor treat unchanged integer percentages
as proof that a tiny request was free or correctly attributed.
"""

from __future__ import annotations

import math
import time
from typing import Any


def subscription_observation(value: object) -> dict[str, Any] | None:
    """Keep only finite allowance numbers; drop opaque tier IDs and all tokens."""
    if not isinstance(value, dict):
        return None
    receipt: dict[str, Any] = {"observedAtMs": int(time.time() * 1000)}
    for name in ("window", "weekly"):
        window = value.get(name)
        if not isinstance(window, dict):
            continue
        percent = window.get("used_percent", window.get("usedPercent"))
        if (
            isinstance(percent, bool)
            or not isinstance(percent, (int, float))
            or not math.isfinite(percent)
            or percent < 0
        ):
            continue
        safe: dict[str, Any] = {"usedPercent": percent}
        reset = window.get("resets_at", window.get("resetsAtMs"))
        if (
            isinstance(reset, (int, float))
            and not isinstance(reset, bool)
            and math.isfinite(reset)
            and 0 <= reset < 10**13
        ):
            # The observed wire uses UNIX seconds; accept already-normalized
            # milliseconds only when their magnitude rules out seconds.
            safe["resetsAtMs"] = int(reset * 1000 if reset < 10**11 else reset)
        duration = window.get("window_duration_mins", window.get("windowDurationMins"))
        if (
            isinstance(duration, (int, float))
            and not isinstance(duration, bool)
            and math.isfinite(duration)
            and 0 < duration <= 525600
        ):
            safe["windowDurationMins"] = duration
        receipt[name] = safe
    return receipt if "window" in receipt or "weekly" in receipt else None
