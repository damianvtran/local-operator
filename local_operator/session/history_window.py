"""Opt-in display pages of the owner's canonical durable replay.

The journal may contain hundreds of MB of ignored host checkpoints while its
conversation is only a few KB. Reuse the resident canonical replay, not a second
index or the live model context. Signed cursors name a replay cut, never a path;
appends preserve that cut, while prune/compaction/folding require reconciliation.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import sys
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from local_operator.harness.types import AgentMessage, Message

if TYPE_CHECKING:
    from local_operator.session.transcript import Transcript

DISPLAY_HISTORY_CAPABILITY = "display-history-window-v1"
DISPLAY_HISTORY_MESSAGES = 120
DISPLAY_HISTORY_BYTES = 512 * 1024

# Per OWNER, not per viewer: a fleet of sessions must not retain a full replay
# per speculative attach. Admission examines only the already bounded page,
# never walks the whole canonical history merely to decide whether to cache it.
DISPLAY_PAGE_CACHE_ENTRIES = 4
DISPLAY_PAGE_CACHE_BYTES = 2 * 1024 * 1024
_CACHE_CONTAINER_ALLOWANCE = 4096


class DisplayHistoryWindow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["ok", "reset", "full_required"] = "ok"
    conversation_id: str
    owner_epoch: str
    history_generation: int
    through_id: str | None
    messages: list[AgentMessage] = Field(default_factory=list)
    before_token: str | None = None
    snapshot_token: str | None = None
    has_more: bool = False
    total_message_count: int = 0
    theme_turn_count: int = 0
    opener_text: str = ""
    start: int = 0
    # Only seed identities already durable at this cut. These are NOT painted
    # IDs: older unseen messages must remain pageable without being suppressed.
    durable_seed_ids: list[str] = Field(default_factory=list)
    durable_seed_tool_ids: list[str] = Field(default_factory=list)


def _sign(payload: dict[str, Any], key: bytes) -> str:
    raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    data = base64.urlsafe_b64encode(raw).decode()
    return data + "." + hmac.new(key, data.encode(), hashlib.sha256).hexdigest()


def _verify(token: str, key: bytes) -> dict[str, Any]:
    if len(token) > 4096:
        raise ValueError("invalid history token")
    try:
        data, signature = token.split(".")
        expected = hmac.new(key, data.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(signature, expected):
            raise ValueError("invalid history token")
        value = json.loads(base64.urlsafe_b64decode(data))
        if not isinstance(value, dict):
            raise ValueError("invalid history token")
        return value
    except (ValueError, TypeError) as exc:
        raise ValueError("invalid history token") from exc


class _DisplayWindowCache:
    """Private detached page templates; no model/context consumer sees them."""

    def __init__(self) -> None:
        self.entries: OrderedDict[tuple[object, ...], tuple[DisplayHistoryWindow, int]] = (
            OrderedDict()
        )
        self.retained_bytes = 0

    def get(self, key: tuple[object, ...]) -> DisplayHistoryWindow | None:
        cached = self.entries.get(key)
        if cached is None:
            return None
        self.entries.move_to_end(key)
        # Rendering and seed annotation mutate returned models. A shared cached
        # Message would poison later attaches and violate replay's ownership.
        return cached[0].model_copy(deep=True)

    def put(self, key: tuple[object, ...], page: DisplayHistoryWindow) -> None:
        size = _retained_size((key, page), DISPLAY_PAGE_CACHE_BYTES - _CACHE_CONTAINER_ALLOWANCE)
        if size is None:
            return
        previous = self.entries.pop(key, None)
        if previous is not None:
            self.retained_bytes -= previous[1]
        while self.entries and (
            len(self.entries) >= DISPLAY_PAGE_CACHE_ENTRIES
            or self.retained_bytes + size > DISPLAY_PAGE_CACHE_BYTES - _CACHE_CONTAINER_ALLOWANCE
        ):
            _, (_, removed) = self.entries.popitem(last=False)
            self.retained_bytes -= removed
        self.entries[key] = (page.model_copy(deep=True), size)
        self.retained_bytes += size


def _retained_size(value: object, limit: int) -> int | None:
    """Bound the page's reachable data, including keys, tool metadata and media.

    This is a small page walk, not a full-history admission pass (the latter
    doubled cold 20k-row replay CPU in the measured prototype). Shared immutable
    values count once within a page and conservatively again across cache entries.
    The fixed allowance covers the four LRU nodes and bookkeeping. Framework
    class/schema objects are process-global, not retained by this cache.
    """
    pending = [value]
    seen: set[int] = set()
    total = 0
    while pending:
        item = pending.pop()
        identity = id(item)
        if identity in seen:
            continue
        seen.add(identity)
        total += sys.getsizeof(item)
        if total > limit:
            return None
        if isinstance(item, BaseModel):
            pending.extend(
                (
                    item.__dict__,
                    item.__pydantic_fields_set__,
                    item.__pydantic_extra__,
                    item.__pydantic_private__,
                )
            )
        elif isinstance(item, dict):
            pending.extend(item)
            pending.extend(item.values())
        elif isinstance(item, (list, tuple, set, frozenset)):
            pending.extend(item)
    return total


def display_window(
    transcript: Transcript,
    *,
    conversation_id: str,
    owner_epoch: str,
    through_id: str | None,
    before: str | None = None,
    anchor: str = "",
    max_messages: int = DISPLAY_HISTORY_MESSAGES,
    max_wire_bytes: int = DISPLAY_HISTORY_BYTES,
    durable_seed_tools: frozenset[str] = frozenset(),
) -> DisplayHistoryWindow:
    """Reuse exact bounded display requests, never the mutable public replay.

    A signed ``before`` token owns its cut, irrespective of the newer cursor
    the caller passes beside it. The token itself is part of the key, so no
    validation is bypassed: only the exact request already validated at this
    generation/epoch can hit. Invalid/reset requests are never retained.
    """
    cache = transcript._display_window_cache
    if cache is None:
        cache = transcript._display_window_cache = _DisplayWindowCache()
    key = (
        conversation_id,
        owner_epoch,
        transcript._history_generation,
        transcript._history_page_key,
        None if before is not None else through_id,
        before,
        anchor,
        max_messages,
        max_wire_bytes,
        durable_seed_tools,
    )
    cached = cache.get(key)
    if cached is not None:
        return cached
    page = _capture_display_window(
        transcript,
        conversation_id=conversation_id,
        owner_epoch=owner_epoch,
        through_id=through_id,
        before=before,
        anchor=anchor,
        max_messages=max_messages,
        max_wire_bytes=max_wire_bytes,
        durable_seed_tools=durable_seed_tools,
    )
    # Replay creates fresh models, but nested JSON-ish tool/provider payloads
    # may still share children with journal dictionaries. The mutable display
    # response must own those children even on a cold or non-admitted request.
    # Copy only the selected page, never the whole canonical history.
    page = page.model_copy(deep=True)
    if page.status != "reset":
        cache.put(key, page)
    return page


def _capture_display_window(
    transcript: Transcript,
    *,
    conversation_id: str,
    owner_epoch: str,
    through_id: str | None,
    before: str | None = None,
    anchor: str = "",
    max_messages: int = DISPLAY_HISTORY_MESSAGES,
    max_wire_bytes: int = DISPLAY_HISTORY_BYTES,
    durable_seed_tools: frozenset[str] = frozenset(),
) -> DisplayHistoryWindow:
    """Return complete replay rows, with tool call/result groups kept together.

    A single group larger than the byte budget explicitly requests the existing
    full local replay path. It is never replaced by a prefix or marked complete.
    The authenticated attach has already established access to that transcript.
    """
    generation = transcript._history_generation
    envelope: dict[str, Any] = dict(
        conversation_id=conversation_id,
        owner_epoch=owner_epoch,
        history_generation=generation,
        through_id=through_id,
    )
    claims = None
    if before is not None:
        claims = _verify(before, transcript._history_page_key)
        if claims.get("conversation_id") != conversation_id:
            raise ValueError("history token belongs to another conversation")
        if (
            claims.get("owner_epoch") != owner_epoch
            or claims.get("history_generation") != generation
        ):
            return DisplayHistoryWindow(status="reset", **envelope)
        through_id = claims.get("through_id")
        envelope["through_id"] = through_id
    try:
        history = transcript.build_llm_history(through_id=through_id) if through_id else []
    except ValueError:
        return DisplayHistoryWindow(status="reset", **envelope)
    # These identities belong to the SAME durable cut as the page. Derive
    # them before discarding the full replay, including on an oversized page;
    # a subscribing session must not reconstruct history a second time.
    seed_tool_ids = (
        [
            str(message.tool_call_id)
            for message in history
            if isinstance(message, Message)
            and message.role == "tool"
            and message.tool_call_id in durable_seed_tools
        ]
        if durable_seed_tools
        else []
    )
    total = len(history)
    end = int(claims["position"]) if claims is not None else total
    if end < 0 or end > total:
        raise ValueError("invalid history page position")
    # Message.tool_results are rendered on their preceding call's card. Do not
    # split that group across pages, or a settled call looks interrupted until
    # an unrelated scroll fetch happens to supply the result.
    result_positions = {
        message.tool_call_id: index
        for index, message in enumerate(history)
        if isinstance(message, Message) and message.role == "tool"
    }
    boundaries = [0]
    paired_through = -1
    for index, message in enumerate(history):
        if index > paired_through and index and getattr(message, "role", "") != "tool":
            boundaries.append(index)
        for call in getattr(message, "tool_calls", ()):
            paired_through = max(paired_through, result_positions.get(call.id, index))
    boundaries.append(total)
    if anchor:
        wanted = next(
            (
                index
                for index, message in enumerate(history)
                if message.id == anchor
                or any("tool:" + call.id == anchor for call in getattr(message, "tool_calls", ()))
            ),
            None,
        )
        if wanted is None:
            return DisplayHistoryWindow(status="reset", **envelope)
        # Include the anchor and a real viewport's worth after it. The TUI can
        # page in either direction using the snapshot token without guessing IDs.
        end = next((b for b in boundaries if b >= min(total, wanted + max_messages // 2)), total)
    start = end
    selected: list[AgentMessage] = []
    used = 0
    for left in reversed([b for b in boundaries if b < end]):
        group = history[left:start]
        # Match the socket's JSON encoding, including escaped non-ASCII text;
        # a character budget or UTF-8-only estimate understates that frame.
        cost = sum(len(json.dumps(m.model_dump(mode="json")).encode()) + 1 for m in group)
        if used + cost > max_wire_bytes - 8192 or len(selected) + len(group) > max_messages:
            if not selected or (
                anchor
                and not any(
                    m.id == anchor
                    or any("tool:" + c.id == anchor for c in getattr(m, "tool_calls", ()))
                    for m in selected
                )
            ):
                return DisplayHistoryWindow(
                    status="full_required", durable_seed_tool_ids=seed_tool_ids, **envelope
                )
            break
        selected[0:0] = group
        used += cost
        start = left
        if len(selected) >= max_messages:
            break
    token_fields = dict(envelope)
    snapshot_token = _sign(dict(token_fields, position=total), transcript._history_page_key)
    before_token = (
        _sign(dict(token_fields, position=start), transcript._history_page_key) if start else None
    )
    return DisplayHistoryWindow(
        **envelope,
        messages=selected,
        durable_seed_tool_ids=seed_tool_ids,
        before_token=before_token,
        snapshot_token=snapshot_token,
        has_more=start > 0,
        total_message_count=total,
        start=start,
        theme_turn_count=sum(getattr(m, "role", "") in ("user", "assistant") for m in history),
        opener_text=next(
            (m.text[:256] for m in history if isinstance(m, Message) and m.role == "user"), ""
        ),
    )
