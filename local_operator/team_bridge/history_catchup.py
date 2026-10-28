"""Bounded, requested-page admission through the connector's screening spine.

The server's explicit ``not_screened`` status is the sole eligibility signal:
absence means own/pre-join/old-server metadata, never permission to fetch a body.
Only confirmed hub verdicts advance the contentless page cursor. A held oldest
candidate deliberately blocks later candidates so another read can resume it.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from .connector.conversation import handle_conversation_available
from .store import Store
from .types import ConversationFeedRef

CATCHUP_MESSAGES_MAX = 10
CATCHUP_AGE_S = 30 * 24 * 60 * 60
# This budget covers credentials, all model/status/admit calls and the refresh;
# cancellation keeps the current candidate unadvanced, and its claim is released
# by the shared handler's finally. It is not ten independent provider timeouts.
CATCHUP_TIMEOUT_S = 45.0


def _masked_page(page: dict[str, Any]) -> dict[str, Any]:
    """Never serialize unscreened/rejected bytes, even in a malformed answer."""

    return {
        **page,
        "messages": [
            {**row, "body": None}
            if isinstance(row, dict)
            and row.get("agent_delivery_status") in ("not_screened", "rejected")
            else row
            for row in page["messages"]
        ],
    }


def _created_at(row: dict[str, Any]) -> float | None:
    value = row.get("created_at")
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        # Naive wall time is not age evidence; never guess the host's timezone.
        return parsed.timestamp() if parsed.tzinfo is not None else None
    except (ValueError, OverflowError):
        return None


async def catch_up_history_page(
    runtime: Any,
    channel_id: str,
    page: Any,
    *,
    limit: int,
    before_seq: int | None,
    after_seq: int | None,
) -> Any:
    """Return fresh authorized history, never bytes from screening's fetch.

    The request bounds AND returned sequence window identify the cursor. A
    moving latest-page window cannot inherit a cursor past a newly exposed
    older row. The enrollment generation is independently bound by Store.
    Retained-history reads in paused/read-only mode must not start admission.
    """

    if not isinstance(page, dict) or not isinstance(page.get("messages"), list):
        return page
    page = _masked_page(page)
    candidates = [
        row for row in page["messages"]
        if isinstance(row, dict) and row.get("agent_delivery_status") == "not_screened"
    ]
    if not candidates:
        return page
    summary: dict[str, Any] = {"attempted": 0, "confirmed": 0, "state": "complete"}
    if runtime.admission != "admitted":
        return {**page, "catchup": {**summary, "state": "held", "code": "not_admitted"}}
    # A bad sequence cannot silently sort after legitimate work and disappear
    # behind its cursor; malformed eligibility metadata holds the entire page.
    if any(type(row.get("seq")) is not int or row["seq"] < 1 for row in candidates):
        return {**page, "catchup": {**summary, "state": "held", "code": "invalid_metadata"}}
    candidates.sort(key=lambda row: row["seq"])
    window = [
        [row.get("seq"), row.get("id")]
        for row in page["messages"]
        if isinstance(row, dict)
    ]
    page_window = hashlib.sha256(
        json.dumps([before_seq, after_seq, limit, window], separators=(",", ":")).encode()
    ).hexdigest()
    config_dir = Path(runtime.config_dir)
    store = Store(config_dir)
    now = time.time()
    try:
        async with asyncio.timeout(CATCHUP_TIMEOUT_S):
            cursor = store.history_catchup_cursor(runtime.owner, channel_id, page_window)
            access = None
            for row in candidates:
                seq = row["seq"]
                if seq <= cursor:
                    continue
                created = _created_at(row)
                if created is None or created > now:
                    summary.update(state="held", code="invalid_age")
                    break
                if now - created > CATCHUP_AGE_S:
                    summary["age_skipped"] = summary.get("age_skipped", 0) + 1
                    continue
                if summary["attempted"] >= CATCHUP_MESSAGES_MAX:
                    summary.update(state="limited", code="message_limit")
                    break
                # Ref validation is the connector's own locator contract. No
                # body, display label or owner-chosen identity enters this ref.
                ref = ConversationFeedRef(
                    conversation_id=channel_id,
                    message_id=str(row.get("id") or ""),
                    conversation_seq=seq,
                    mention=False,
                ).validated()
                if access is None:
                    access = await runtime.client.current_access()
                summary["attempted"] += 1
                outcome = await handle_conversation_available(
                    runtime.owner, ref, config_dir=config_dir,
                    auth_store=runtime.client._auth_store(), access=access,
                    wire=runtime.client, admission_only=True,
                )
                if outcome.get("action") not in ("admitted", "rejected"):
                    # Handler codes may incorporate a remote refusal. Keep
                    # this summary closed rather than reflecting arbitrary
                    # server/provider text into the reading agent's context.
                    summary.update(state="held", code="admission_held")
                    break
                # Count before the durable write: if the owner moves at this
                # boundary, the verdict still happened and needs a fresh read.
                summary["confirmed"] += 1
                store.advance_history_catchup_cursor(runtime.owner, channel_id, page_window, seq)
                cursor = seq
            if summary["confirmed"]:
                # Rejected status is server authority too. Neither a local
                # receipt nor handler result may manufacture page bodies/status.
                fresh = await runtime.client.conversation_messages(
                    channel_id, tenant_id=runtime.tenant_id, limit=limit,
                    before_seq=before_seq, after_seq=after_seq,
                    endpoint_id=runtime.owner.endpoint_id,
                )
                if not isinstance(fresh, dict) or not isinstance(fresh.get("messages"), list):
                    raise ValueError("invalid history page")
                page = _masked_page(fresh)
                summary["refreshed"] = True
    except TimeoutError:
        summary.update(state="held", code="time_limit")
    except Exception:
        # Errors from credentials, screening or storage can include untrusted
        # data. Keep them out of the tool result; the old metadata page is safe.
        summary.update(state="held", code="catchup_unconfirmed")
    return {**page, "catchup": summary}
