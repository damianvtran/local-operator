"""Opt-in display pages of the runtime's canonical durable replay.

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

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator

from local_operator.harness.types import AgentMessage, Message
from local_operator.session.transcript import (
    audit_slice,
    collect_prunes,
    context_cut_index,
    context_preserved_turn_ids,
    first_message_index,
)

if TYPE_CHECKING:
    from local_operator.session.transcript import Transcript

DISPLAY_HISTORY_CAPABILITY = "display-history-window-v1"

#: Separate from the capability above, and it MUST stay separate. That one is a
#: presence flag with no version handshake, so it cannot express "this runtime
#: also pages pre-compaction history". :class:`DisplayHistoryWindow` forbids
#: extra fields, so a runtime that emitted ``audit``/``audit_available`` to a
#: viewer built before those fields existed would fail that viewer's validation
#: and turn into a FAILED ATTACH — not a degraded one. Mixed builds against one
#: sessions directory are routine here (the global uv-tool runtime is updated
#: independently of a repo checkout's venv), so this is a live rollout hazard
#: rather than a theoretical one. The runtime emits the new fields only to a
#: viewer that negotiated this string.
DISPLAY_HISTORY_AUDIT_CAPABILITY = "display-history-audit-v1"
DISPLAY_HISTORY_MESSAGES = 120
DISPLAY_HISTORY_BYTES = 512 * 1024

# Per RUNTIME, not per viewer: a fleet of sessions must not retain a full replay
# per speculative attach. Admission examines only the already bounded page,
# never walks the whole canonical history merely to decide whether to cache it.
DISPLAY_PAGE_CACHE_ENTRIES = 4
DISPLAY_PAGE_CACHE_BYTES = 2 * 1024 * 1024
_CACHE_CONTAINER_ALLOWANCE = 4096


class DisplayHistoryWindow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["ok", "reset", "full_required"] = "ok"
    conversation_id: str
    # Wire compat for the ``owner_epoch`` → ``runtime_epoch`` rename. Reads
    # accept BOTH keys, but the wire keeps emitting ``owner_epoch`` this
    # release: the DTO is ``extra="forbid"`` and crosses the socket, a
    # pre-rename viewer's DTO knows only ``owner_epoch``, and mixed-version
    # attach is supported — emitting the new key would fail that viewer's
    # ``model_validate`` on every page and break the attach outright (worse
    # than the degrade the rename was hedging against). The field therefore
    # stays named ``owner_epoch`` — ``model_dump`` emits the field name on
    # every wire route — and flips to ``runtime_epoch`` (name, emit, and
    # dropping the old validation choice together) in the release after this
    # PR, once no pre-rename viewer can attach.
    owner_epoch: str = Field(validation_alias=AliasChoices("owner_epoch", "runtime_epoch"))
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
    #: This page carries PRE-COMPACTION rows: real history the model can no
    #: longer see. Its ``start`` is a journal coordinate, not a position in the
    #: context replay, so a consumer must not compare the two (see
    #: ``AttachedSession.load_older_display_page``, whose contiguity check is a
    #: context-coordinate assertion and is skipped for these pages).
    audit: bool = False
    #: Older rows exist behind the context cut and are reachable by paging.
    #: Lets the viewer say "earlier history above" at the moment the context
    #: phase drains, rather than claiming the conversation starts there.
    audit_available: bool = False

    @model_validator(mode="before")
    @classmethod
    def _collapse_epoch_alias(cls, data: Any) -> Any:
        """Accept the epoch under either key, including both at once.

        ``AliasChoices`` consumes only the FIRST match and the model forbids
        extras, so a payload carrying ``owner_epoch`` AND ``runtime_epoch``
        raised ``extra_forbidden`` on the second key instead of validating.
        That is the one shape a producer mid-rename can emit — the old key for
        viewers that have not flipped, the new one for those that have — and a
        hard failure there breaks the attach outright. Collapse the pair to
        the single key the field reads.

        ``runtime_epoch`` wins when both are present: a payload that carries it
        comes from the newer producer, so its value is the authoritative one,
        and the alternative is honouring a key the rename is retiring. Delete
        with ``AliasChoices`` on the flip release.

        A ``mode="before"`` validator is handed the CALLER's object, not a
        pydantic-owned copy, so popping and assigning here would mutate the
        dict the caller passed to ``model_validate`` in place — the caller
        would lose ``runtime_epoch`` and see ``owner_epoch`` silently
        overwritten. Today's sole caller passes a fresh dict, but a caller
        that reuses a payload (validating one dict against two models, or
        logging it after validation) would be corrupted. Copy before
        rewriting.
        """
        if isinstance(data, dict) and "runtime_epoch" in data:
            data = dict(data)
            data["owner_epoch"] = data.pop("runtime_epoch")
        return data

    @property
    def runtime_epoch(self) -> str:
        """Runtime-internal name for the epoch while the wire key is held back.

        The wire emits ``owner_epoch`` this release (see the field above), so
        the FIELD carries the old name while the rename's internal callers —
        the capture path's parameter and claims, plus the tests — already use
        ``runtime_epoch``. This bridge keeps both names readable until the
        flip release swaps the field name and drops this property. Drop after
        the release that follows this PR.
        """
        return self.owner_epoch


#: Fields the audit capability introduced. Stripped for a viewer that did not
#: negotiate :data:`DISPLAY_HISTORY_AUDIT_CAPABILITY`; see its comment for why
#: emitting them unconditionally is an attach failure rather than noise.
AUDIT_WIRE_FIELDS = ("audit", "audit_available")


def strip_audit_fields(payload: dict[str, Any], *, audit_capable: bool) -> dict[str, Any]:
    """Drop the audit fields from a serialized page for an older viewer.

    THE one place that decision is implemented, mutating in place and returning
    the same dict so it composes with either kind of caller. A page reaches the
    wire by THREE routes, and every one of them must strip:

    * the attach sync push frame (``server.py``, ``_handle_auth``),
    * the ``history_page`` RPC (``_dispatch_payload``),
    * the ``frontend_sync`` RPC (``_dispatch_payload``), which an older viewer
      calls on every history refresh.

    Stripping in only some of them yields a viewer that attaches cleanly and
    then fails later — on its first scroll up, or on the first refresh after
    the runtime appends a row — which is a worse failure than any single route
    breaking on its own. Round 1 review found the third route unstripped after
    this docstring had asserted there were two: if a fourth is ever added,
    update this list in the same commit.
    """
    if not audit_capable:
        for name in AUDIT_WIRE_FIELDS:
            payload.pop(name, None)
    return payload


def wire_payload(window: DisplayHistoryWindow, *, audit_capable: bool) -> dict[str, Any]:
    """Serialize a page for one viewer, honouring what that viewer negotiated."""
    return strip_audit_fields(window.model_dump(mode="json"), audit_capable=audit_capable)


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

    Deliberately an OVER-estimate: ``sys.getsizeof`` charges per-object CPython
    overhead (measured ~7.5x a pickled page — 91,960 accounted against 12,172
    serialized), so the effective retention ceiling is well under the nominal
    2 MiB. That direction is the safe one for a per-runtime budget multiplied
    across a fleet, but anyone re-tuning the constant should size it against
    ACCOUNTED bytes rather than expecting a wire-sized figure.
    The fixed allowance covers the four LRU nodes and bookkeeping. Framework
    class/schema objects are process-global, not retained by this cache.
    """
    pending = [value]
    # Identity-keyed, and safe only because nothing here is freed mid-walk: the
    # page and its key are held by `pending`/the caller for the whole traversal,
    # so no id() can be recycled into a false "already counted".
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
    # Held at ``owner_epoch`` for one release: the runtime-side callers pass
    # this by keyword from ``session.py``, whose owner is the remaining
    # owner→runtime identifier pass, not this wire-compat PR. The value flows
    # into the ``runtime_epoch`` DTO field and token claims below.
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
        # The paging PHASE needs no key term of its own: it is carried inside
        # the signed ``before`` token, which is already keyed here, and the two
        # phases mint structurally different token payloads. So a context page
        # and an audit page can never collide on one key, and — because the
        # phase is only ever read from a signature this runtime minted — a viewer
        # cannot ask for audit rows without having been handed a cursor to them.
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
        runtime_epoch=owner_epoch,
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


#: Marks a backward cursor as belonging to the audit phase rather than to the
#: context replay. Carried inside the SIGNED token, so a viewer cannot ask for
#: an audit page it was not handed a cursor for.
_AUDIT_PHASE = "audit"


def _audit_entry_cursor(transcript: Transcript) -> str | None:
    """Entry id the audit phase resumes from, or ``None`` if nothing precedes.

    Returns an ENTRY ID, never a journal index, and the token signs that id for
    the same reason ``TranscriptPage`` addresses rows by id: ``compact_file``
    rewrites the journal in place, so an integer offset minted before a
    compaction points at a different row afterwards \u2014 it does not fail, it
    silently lies. An id that no longer resolves is detectable, and the resolve
    failure is answered with ``status="reset"`` so the viewer re-syncs.
    """
    entries = transcript._entries
    cut = context_cut_index(entries, quiet=True)
    first = first_message_index(entries)
    # Nothing precedes the context replay: this conversation really does begin
    # where the context phase ends, and the chain terminates there as before.
    if first is None or first >= cut:
        return None
    return entries[cut].id if cut < len(entries) else _AUDIT_TAIL_CURSOR


#: The audit window's first cursor when the context cut sits past the last
#: journal row (an empty kept suffix). Distinct from an entry id so it cannot
#: collide with one.
_AUDIT_TAIL_CURSOR = "\x00audit-tail"


def _hoisted_turn_ids(transcript: Transcript) -> frozenset[str]:
    """Cached :func:`context_preserved_turn_ids` for the current journal state.

    Cached because it is asked once per audit page and is not cheap: it runs the
    same shed/cap filter the context replay does, over a scan of the whole entry
    list, and measured 88 ms on the 231 MB reference journal — which would have
    dominated a page that otherwise costs ~8 ms and put this feature outside its
    budget on exactly the sessions it exists for.

    Keyed on ``_history_generation``, which the transcript already bumps on
    every compaction and prune — the only events that can change which turns are
    hoisted. An append cannot, because only the LATEST compaction's payload is
    read and appending does not create one.
    """
    generation = transcript._history_generation
    cached = transcript._audit_hoisted_cache
    if cached is not None and cached[0] == generation:
        return cached[1]
    resolved = frozenset(context_preserved_turn_ids(transcript._entries))
    transcript._audit_hoisted_cache = (generation, resolved)
    return resolved


def _capture_audit_window(
    transcript: Transcript,
    *,
    envelope: dict[str, Any],
    claims: dict[str, Any],
    max_messages: int,
    max_wire_bytes: int,
) -> DisplayHistoryWindow:
    """One backward page of PRE-COMPACTION history.

    Deliberately a separate function from :func:`_capture_display_window`
    rather than a branch inside it, mirroring the split in
    :func:`~local_operator.session.transcript.replay_entries`: the two phases
    answer different questions ("what may the model see" vs "what did the
    conversation contain") and share no coordinate space. This one's ``start``
    is a JOURNAL index; that one's is a position in the context replay. Fusing
    them would make every later edit have to remember which coordinate it is
    holding, and the first one to forget reintroduces the unreachable-history
    defect from the other side.

    The counts this page does NOT touch are as load-bearing as the rows it
    returns. ``total_message_count``, ``theme_turn_count`` and ``opener_text``
    stay CONTEXT-derived and are left at their defaults here, because the TUI
    reads the first as a monotonic context-growth signal for its presentation
    cache and the second as the growth gate for conversation retitling.
    Inflating them by an audit depth of 17,000 rows would invalidate every
    cached presentation and re-fire the retitle gate across every session on
    the machine.
    """
    entries = transcript._entries
    cursor = str(claims.get("end_entry_id") or "")
    if cursor == _AUDIT_TAIL_CURSOR:
        end_index = len(entries)
    else:
        end_index = next((i for i, entry in enumerate(entries) if entry.id == cursor), -1)
        # The signed id no longer resolves: ``compact_file`` replaced the file
        # under an outstanding cursor. Reset is the honest answer — the viewer
        # re-syncs and re-pages rather than being handed rows from a coordinate
        # space that no longer exists.
        if end_index < 0:
            return DisplayHistoryWindow(status="reset", **envelope)
    first = first_message_index(entries)
    if first is None or end_index <= first:
        return DisplayHistoryWindow(
            **envelope,
            messages=[],
            before_token=None,
            has_more=False,
            start=end_index,
            audit=True,
            audit_available=False,
        )
    prunes = collect_prunes(entries)
    # The one place the two phases can OVERLAP. The context page re-emits the
    # latest compaction's ``preserved_user_turns`` at its head, under the
    # ORIGINAL row ids — and those same rows sit below the cut, where this
    # phase replays them in place. Delivered twice, the TUI's mount-time id
    # dedupe drops the second, so the visible symptom is a row that is silently
    # MISSING from the audit page rather than one shown twice. Suppress the
    # audit copy: the reader has already been shown that row by the context
    # page it scrolled through to get here.
    hoisted = _hoisted_turn_ids(transcript)
    messages, indices, window_start = audit_slice(
        entries,
        transcript._attachments,
        end_index=end_index,
        limit=max_messages,
        prunes=prunes,
    )
    if hoisted:
        kept = [
            (message, index)
            for message, index in zip(messages, indices)
            if message.id not in hoisted
        ]
        messages = [message for message, _ in kept]
        indices = [index for _, index in kept]
    # Trim from the LEFT to fit the frame budget, so the page stays contiguous
    # with the cursor it was fetched for and the next cursor is simply the row
    # this page begins at. Matches the context phase, which also drops whole
    # groups off its left edge rather than truncating prose.
    used = 0
    keep = 0
    for offset in range(len(messages) - 1, -1, -1):
        cost = len(json.dumps(messages[offset].model_dump(mode="json")).encode()) + 1
        if used + cost > max_wire_bytes - 8192 and keep:
            break
        used += cost
        keep += 1
    if not keep and messages:
        # A single row exceeds the whole frame budget. The context phase
        # answers this with ``full_required``, which escalates to a local
        # replay; that escalation is meaningless here (it would rebuild the
        # model's history, which does not contain this row at all). Return the
        # oversized row alone instead — one row over budget is a frame the
        # transport will refuse, but returning nothing strands the chain.
        keep = 1
    trimmed = bool(keep) and keep < len(messages)
    if keep:
        messages = messages[len(messages) - keep :]
        indices = indices[len(indices) - keep :]
    # Where the NEXT page resumes. Normally the window's own left edge, so the
    # chain advances by a whole window even when this page delivered nothing —
    # which happens for real, when every row in the window was a hoisted
    # preserved turn the context phase already showed. Minting the cursor from
    # the surviving rows instead would re-mint the cursor this page was fetched
    # with, and the chain would spin on one window forever: a hang, not a wrong
    # answer.
    #
    # The exception is a page the BYTE budget trimmed. There the rows below the
    # kept ones were dropped for size, not because they were already shown, so
    # the next page must resume at the first row kept or they are skipped.
    start = indices[0] if (trimmed and indices) else window_start
    more = start > first
    before_token = (
        _sign(
            dict(envelope, phase=_AUDIT_PHASE, end_entry_id=entries[start].id),
            transcript._history_page_key,
        )
        if more
        else None
    )
    return DisplayHistoryWindow(
        **envelope,
        messages=messages,
        before_token=before_token,
        has_more=more,
        start=start,
        audit=True,
        audit_available=more,
    )


def _capture_display_window(
    transcript: Transcript,
    *,
    conversation_id: str,
    runtime_epoch: str,
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
        runtime_epoch=runtime_epoch,
        history_generation=generation,
        through_id=through_id,
    )
    claims = None
    if before is not None:
        claims = _verify(before, transcript._history_page_key)
        if claims.get("conversation_id") != conversation_id:
            raise ValueError("history token belongs to another conversation")
        # Wire compat: tokens minted before the ``owner_epoch`` →
        # ``runtime_epoch`` rename carry the old claims key, and a mismatch
        # here reads as status="reset" — a silent window reset plus a
        # full-replay refetch, not an error — so read old-then-new. Drop the
        # ``owner_epoch`` read after the release that follows this PR.
        claims_epoch = claims.get("owner_epoch")
        if claims_epoch is None:
            claims_epoch = claims.get("runtime_epoch")
        if claims_epoch != runtime_epoch or claims.get("history_generation") != generation:
            return DisplayHistoryWindow(status="reset", **envelope)
        through_id = claims.get("through_id")
        envelope["through_id"] = through_id
        if claims.get("phase") == _AUDIT_PHASE:
            return _capture_audit_window(
                transcript,
                envelope=envelope,
                claims=claims,
                max_messages=max_messages,
                max_wire_bytes=max_wire_bytes,
            )
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
                # One group is larger than the whole frame budget, so the
                # reader escalates to a local replay of the model's history.
                # That replay reaches the start of the CONTEXT phase and stops
                # there — so hand it the audit cursor on the way out, or the
                # pre-compaction rows become unreachable for exactly the
                # sessions most likely to contain an oversized group. The
                # cursor stays valid across the escalation because it names an
                # entry id, not a position in the chain being abandoned.
                audit_entry = _audit_entry_cursor(transcript)
                return DisplayHistoryWindow(
                    status="full_required",
                    durable_seed_tool_ids=seed_tool_ids,
                    before_token=(
                        _sign(
                            dict(envelope, phase=_AUDIT_PHASE, end_entry_id=audit_entry),
                            transcript._history_page_key,
                        )
                        if audit_entry is not None
                        else None
                    ),
                    audit_available=audit_entry is not None,
                    **envelope,
                )
            break
        selected[0:0] = group
        used += cost
        start = left
        if len(selected) >= max_messages:
            break
    token_fields = dict(envelope)
    snapshot_token = _sign(dict(token_fields, position=total), transcript._history_page_key)
    # The audit handoff. When the context replay is drained, the conversation
    # does NOT necessarily begin here: everything before the compaction cut is
    # still on disk, and before this it was simply unaddressable (measured on
    # the reference journal: 327 of 17,349 rows reachable). Mint a cursor into
    # the audit phase instead of terminating the chain.
    audit_entry = _audit_entry_cursor(transcript) if not start else None
    audit_available = audit_entry is not None
    if start:
        before_token = _sign(dict(token_fields, position=start), transcript._history_page_key)
    elif audit_entry is not None:
        before_token = _sign(
            dict(token_fields, phase=_AUDIT_PHASE, end_entry_id=audit_entry),
            transcript._history_page_key,
        )
    else:
        before_token = None
    return DisplayHistoryWindow(
        **envelope,
        messages=selected,
        durable_seed_tool_ids=seed_tool_ids,
        before_token=before_token,
        snapshot_token=snapshot_token,
        has_more=before_token is not None,
        total_message_count=total,
        start=start,
        audit_available=audit_available,
        theme_turn_count=sum(getattr(m, "role", "") in ("user", "assistant") for m in history),
        opener_text=next(
            (m.text[:256] for m in history if isinstance(m, Message) and m.role == "user"), ""
        ),
    )
