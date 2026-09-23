"""The session runtime: make a live session reachable over a control socket.

Every interactive ``lop`` process (the TUI, and ``exec`` when it runs
attached) hosts one of these. It does three things:

1. **Publishes** the discovery record (see :mod:`.registry`) and rewrites it
   on a heartbeat — cheap enough that a machine with no daemon installed
   pays nothing but one small file write every 15 seconds.
2. **Listens** on a random loopback port for control connections,
   authenticating each with the record's key (constant-time
   compare — the key is the whole credential).
3. **Bridges**: folds the session's event stream into the phone projection
   (:class:`~local_operator.mobile.projection.ProjectionFold`) and pushes a
   repaint on change; applies clients' requests to the session through a
   host-provided :class:`SessionHandle`.

This was ``mobile/registrant.py`` and the class was ``Registrant``. Nothing
about it is phone-specific except the projection fold, which is an injected
:class:`ProjectionSink` collaborator built lazily on the first daemon dial
when none is supplied: the phone daemon, an attach terminal, and — later —
wakes and background automations are all VIEWERS of one session runtime.
``Registrant`` remains as an alias at the bottom of this module and at the old
import path, so no call site had to change in the move.

The handle indirection exists because the two runtime kinds drive their
session differently: the TUI must route mutations through Textual's message
pump and thread (``call_from_thread``), while an exec-mode host can call the
session directly. The runtime speaks to the handle, never to Textual.

Threading: the control socket server runs on its own thread with its own
event loop — the TUI's loop must never block on a phone, and clients'
requests (model switches, aborts) must land even while the TUI is mid-repaint.
All session mutations funnel through the handle, whose contract is "callable
from the runtime's loop, serialized by the implementor".
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import contextlib
import copy
import hmac
import inspect
import json
import logging
import os
import secrets
import threading
import time
import weakref
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Mapping, Protocol, cast

if TYPE_CHECKING:
    from pathlib import Path

    from local_operator.harness.types import ImageContent

from local_operator.harness.approval import (
    AUTHORITY_OPS,
    admit_increasing,
    frame_authority,
    handshake_proof,
    handshake_proof_ok,
    is_wire_hex,
    operator_nonce,
    request_proof_ok,
    signature_target,
)
from local_operator.mobile.projection import ProjectionFold
from local_operator.mobile.types import SessionProjection
from local_operator.operator import report_operator_authority
from local_operator.operator import verify as operator_verify
from local_operator.operator.trust import (
    ANCHOR_REFRESH_S,
    AnchorCache,
    AnchorLoad,
    OperatorAnchor,
    anchor_path,
    device_is_revoked,
)
from local_operator.paths import config_dir
from local_operator.session.attachments import AttachmentStore
from local_operator.session.frontend_state import FRONTEND_CAPABILITY
from local_operator.session.runtime import stall_watchdog
from local_operator.session.runtime.publication import PublicationGate
from local_operator.session.runtime.registry import RecordPublisher
from local_operator.session.runtime.types import (
    ATTACH_MAX_CLIENTS,
    DESKTOP_WATCH_CAPABILITY,
    DESKTOP_WATCH_LEASE_S,
    EVENT_MUTE_CAPABILITY,
    EVENT_MUTE_DROP_TYPES,
    EXCLUSIVE_MOVE_CAPABILITY,
    HEARTBEAT_INTERVAL_S,
    OPERATOR_SIGNATURE_CAPABILITY,
    ClientKind,
    ClientLocality,
    SessionRecord,
)
from local_operator.session.transcript import (
    _ATTACHMENT_FLOOR_BYTES,
    ATTACHMENT_KEY,
    durable_conversation_path,
)

logger = logging.getLogger(__name__)


def image_blocks(images: list[dict[str, str]] | None) -> list["ImageContent"]:
    """Decode the wire's [{data_b64, mime_type}] into BOUNDED ImageContent blocks.

    Bad entries are dropped, not fatal: a paste that half-decoded should cost
    that one image, not the whole prompt. Empty input yields an empty list,
    which ``_submit_prompt`` treats exactly like no images. This is part of
    the mobile contract — both handles use it.

    Every entry is bounded at the INGEST edge here, the same call the composer
    makes on a paste, because this is the equivalent seam: a phone camera roll
    produces 4032x3024 photos and a phone screenshot 2206x266, and a provider
    refuses an image over 2000 pixels on its long edge once a request carries
    more than twenty of them. Forwarding verbatim was not lossless, it was
    deferred: the block lands in the HISTORY, so every later request re-sends
    it. The render seam did repair oversize blocks afterwards, which made this
    degraded rather than broken — but a repair on every render is work paid
    repeatedly for a bound that costs nothing once, at entry.

    CPU-bound (~315 ms for a 20 MP image), so an async caller must use
    :func:`image_blocks_in_thread` instead.
    """
    if not images:
        return []
    from local_operator.harness.types import ImageContent
    from local_operator.imaging import bound_image_for_model
    from local_operator.media import sniff_image

    out: list[ImageContent] = []
    for item in images:
        if not isinstance(item, dict):
            logger.debug("mobile image dropped: not a dict (%r)", type(item).__name__)
            continue
        # The client's declared ``mime_type`` is deliberately NOT read: the
        # format is decided by CONTENT below, and the wire mime comes back from
        # the bound. A phone that mislabels a HEIC as image/png would otherwise
        # pick an encoder for a format the bytes are not.
        data = item.get("data_b64") or item.get("data") or ""
        if not data:
            logger.debug("mobile image dropped: no data_b64/data")
            continue
        try:
            raw = base64.b64decode(data, validate=True)
        except (ValueError, binascii.Error):
            logger.debug("mobile image dropped: data_b64 is not valid base64")
            continue
        # Format comes from the CONTENT, never the client's mime_type: a phone
        # that mislabels a HEIC as image/png would otherwise pick the encoder
        # for a format the bytes are not.
        info = sniff_image(raw)
        if info is None:
            logger.debug("mobile image dropped: unrecognised image format")
            continue
        try:
            payload, wire_mime, _summary = bound_image_for_model(raw, info)
        except ValueError as error:
            # Undecodable, a decompression bomb, or too large to send with no
            # decoder available. Same contract as every other bad entry here:
            # this image is dropped, the rest of the prompt proceeds.
            logger.debug("mobile image dropped: %s", error)
            continue
        out.append(
            ImageContent(
                data=base64.b64encode(payload).decode("ascii"),
                mime_type=wire_mime,
            )
        )
    return out


async def image_blocks_in_thread(images: list[dict[str, str]] | None) -> list["ImageContent"]:
    """:func:`image_blocks` off the event loop.

    The bound decodes and re-encodes, which is CPU-bound and unbounded in
    duration by an event loop's standards, so every async caller goes through
    here — the same discipline the composer's paste path follows.

    Returns WITHOUT suspending when there is nothing to decode, and that is
    load-bearing rather than an optimisation. ``asyncio.to_thread`` always
    yields to the loop, so an unconditional hop turns every image-less prompt
    and steer into a suspension point — and these callers reserve a producer
    identity and queue in FIFO order right after this call, so a yield here
    lets concurrently submitted turns interleave and be admitted out of order
    (three gathered prompts admitted 1,3,2). The overwhelming majority of
    prompts carry no images and must keep the await-free path they had.
    """

    if not images:
        return []
    return await asyncio.to_thread(image_blocks, images)


#: A prompt payload past 1 MB is a bug, not a prompt — the line limit the
#: control socket reader enforces.
_MAX_LINE_BYTES = 1 << 20

#: How long the per-connection sync task waits for the ON-LOOP frontend bind
#: before falling back to the off-loop one (``_serve_frontend_sync``).
#:
#: 100 ms, from the healthy owner's own distribution rather than a guess: a
#: healthy owner binds in 4.7-5.0 ms p50 and at most ~15 ms p95, so a healthy attach
#: never reaches this grace and its bytes on the wire are unchanged. An owner
#: busy inside a synchronous step of a turn is the case it exists for, and there
#: the wait has no ceiling of its own — the parked hop is what measured 15.0 s.
_ONLOOP_BIND_GRACE_S = 0.1

#: The identity fields the identity-only welcome carries (``_slim_welcome_frame``).
#:
#: ``session_id`` and ``pid`` are REQUIRED rather than a selection: the client
#: rebuilds a ``SessionProjection`` from this payload, that dataclass requires
#: both, and it overwrites ``pid`` from its own discovery record immediately
#: after. The rest are the identity a welcome is read for, and every one of them
#: is a plain scalar the handle's seed already holds — no transcript, no roster,
#: no fold.
_WELCOME_IDENTITY_FIELDS = (
    "session_id",
    "pid",
    "kind",
    "conversation_name",
    "cwd",
    "model_label",
    "model_selector",
    "effort",
    "effort_ladder",
)


def _frame_line_bytes(frame: dict[str, Any], *, payload: bytes | None = None) -> int:
    """Encoded bytes this frame occupies on the wire, with the delimiter counted.

    ONE definition, deliberately: the same arithmetic decides what the ceiling
    may emit, what the relay guard must degrade, and what compaction may merge
    — on THIS module's send path. Two spellings of it is how a producer's
    "sendable" drifts from a reader's "readable", the failure this whole family
    of guards exists to prevent.

    It is NOT yet the only spelling in the tree, and saying it was would be a
    claim this file cannot keep. ``session/frontend_state.py`` derives the same
    number inline twice — its result-envelope reserve, and ``oversized_frame_report``'s
    ``+1`` — against the very limit this module hands it as an argument, and it
    cannot import this name because ``runtime.server`` imports THAT module, so the
    import would close a cycle (its own comment at ``MODEL_CATALOGUE_FLOOR_ROWS``
    pins the value for exactly that reason). Nothing is broken today: the
    arithmetic is identical. Unifying those two onto one rule — by hosting it
    where both can import it — is a follow-up, not something this docstring can
    assert into being.

    Counting the newline is CONSERVATIVE, not a requirement of the readers. The
    boundary was probed against an asyncio ``StreamReader`` created with a buffer
    limit of ``L`` — the limit is a property of the reader (``open_connection(limit=…)``),
    not an argument to ``readline()``: a payload of exactly ``L`` bytes plus its
    ``\\n`` is RETURNED, and only a payload longer than ``L`` raises
    ``ValueError("Separator is found, but chunk is longer than limit")``. Keeping
    the delimiter in the count buys one byte of margin on a boundary where being
    wrong costs the whole connection, which is worth more than the byte; it is
    not a claim about where the limit sits.

    ``payload`` is for a caller that has ALREADY encoded the frame: ``_send_to``
    serializes to write, so measuring it here would serialize every frame twice
    on the per-client repaint path — measured at 2.3 ms for a 188 KB frame on
    this host, against 1.3 us for a small one. Reusing the bytes keeps the rule
    in ONE place without paying for it twice.
    """
    if payload is None:
        payload = json.dumps(frame).encode()
    return len(payload) + 1


def _frame_size_without_delta(frame: dict[str, Any]) -> int:
    """Encoded size of ``frame`` counting its ``delta`` as empty (plus the "\\n").

    Compaction needs a candidate frame's encoded length, but the expensive part
    of a ``message_update`` is the accumulated ``message`` — hundreds of KB that
    a merge does not change. Measuring the whole frame per merge is what made
    compaction quadratic (67.4 MB and 152 ms for one 64-frame pass). Splitting
    the measurement lets the caller add the delta's escaped bytes itself, which
    is exact because JSON escaping is per-character: the encoded length of
    ``a + b`` is the encoded length of ``a`` plus that of ``b``.

    The delta is blanked rather than removed so the key, its quotes and its
    comma are all still counted — only the VALUE's bytes are the caller's to
    add.
    """
    data = frame.get("data")
    if not isinstance(data, dict) or "delta" not in data:
        return _frame_line_bytes(frame)
    probe = {**frame, "data": {**data, "delta": ""}}
    # ``- 2`` removes the two quotes of the blanked delta: the caller adds the
    # real value back including its own quotes.
    return _frame_line_bytes(probe) - 2


def _mergeable_delta_key(payload: Mapping[str, Any]) -> str | None:
    """The in-flight stream one queued frame carries a fragment of, or ``None``.

    Compaction folds ADJACENT frames of the same stream into one, and the only
    thing that decides "same stream" is this key. Two families are delta-grade
    and mergeable:

    * ``message_update`` — the assistant's visible text, keyed by the message it
      accumulates into;
    * ``reasoning_delta`` — the model's private reasoning, keyed by the message
      it belongs to (the field is ``message_id``, not a whole ``message``: a
      reasoning frame carries no message, which is also why it is cheap to
      merge).

    The FAMILY is part of the key, so a text fragment and a reasoning fragment
    can never fold together — merging the model's thinking into the answer
    being painted would corrupt the transcript on the viewer's screen, and the
    two arrive interleaved.

    Everything else returns ``None`` and is left alone: a frame that must not
    merge is never compared to its neighbour at all.
    """
    kind = payload.get("type")
    if kind == "message_update":
        return f"message_update:{(payload.get('message') or {}).get('id') or ''}"
    if kind == "reasoning_delta":
        return f"reasoning_delta:{payload.get('message_id') or ''}"
    return None


#: Line bytes the shedding stage deliberately leaves UNSPENT.
#:
#: The residual share is exact, so spending all of it parks the frame a handful
#: of bytes under the limit — measured at 11 on a 2 MiB text result. A frame
#: flush against the line is a frame the NEXT per-frame field flips from
#: "briefly clipped text" to the emptied fallback beneath it, and this codebase
#: has already paid for that lesson once (``JOB_TEXT_FRAME_BUDGET_CHARS``
#: carries the 13-byte precedent). Per-row tool text is the right place to buy
#: the room from: it is a preview by construction, and a few hundred characters
#: spread across a roster are invisible where an emptied payload is not.
_FIT_SHED_RESERVE_BYTES = 4 * 1024


def _reference_image_payloads(
    value: Any, store: AttachmentStore, *, key: str, floor: int
) -> tuple[Any, int]:
    """Copy ``value`` with oversized inline image payloads replaced by references.

    ``key`` and ``floor`` are the durable encoder's own ``ATTACHMENT_KEY`` and
    ``_ATTACHMENT_FLOOR_BYTES``, imported rather than restated: the live pass
    must externalize exactly the blocks the transcript would, under exactly the
    key it writes, or a viewer resolves one shape and a resume the other.

    Returns ``(value_or_copy, moved)``. The input is NEVER mutated: one producer
    frame is handed to every recipient by :meth:`RuntimeServer._relay_on_loop`,
    so a pass that rewrote it in place would have the first connection decide
    what every later connection sends — and the later connection would then
    enqueue a reference for a frame nobody measured. Only the spine that
    actually changed is copied, which is ``filter_update_trajectories``'s
    precedent for the same reason.

    An image block is identified by carrying a LONG ``data`` string alongside
    either the ``image`` type or a ``mime_type``, rather than by ``data`` alone
    the way the durable encoder does it. The durable encoder has to key on
    ``data`` because it dumps with ``exclude_defaults`` and so loses the block's
    discriminant; the live wire dumps the models whole, so the discriminant is
    right there to check.

    That NARROWS the exposure of a tool's free-form ``details`` payload rather
    than removing it, and the difference is worth stating: a ``details`` blob
    carrying BOTH a long ``data`` string and a string ``mime_type`` is still
    rewritten into a reference (and resolves back to the same bytes on every in
    repo client, so it is harmless today). It is unreachable in this tree for
    the reason QA checked rather than by construction — the only ``details``
    producer that emits ``mime_type`` is ``read``'s image path, which carries no
    ``data``. A future producer that puts a mime type beside a payload under
    ``data`` should expect this pass to see it as an image block.

    A store that refuses the write (read-only home, full disk) leaves the
    payload inline. The frame then stays large and the guard degrades it
    honestly; a half-reference with no resolvable digest would instead render
    as an unavailable image on every client.
    """
    if isinstance(value, dict):
        data = value.get("data")
        if (
            isinstance(data, str)
            and len(data) >= floor
            and (value.get("type") == "image" or isinstance(value.get("mime_type"), str))
        ):
            ref = store.put(data, str(value.get("mime_type") or "image/png"))
            if ref is None:
                return value, 0
            block = {field_name: item for field_name, item in value.items() if field_name != "data"}
            block[key] = ref.digest
            block["mime_type"] = ref.mime_type
            return block, 1
        moved = 0
        copied: dict[str, Any] = {}
        for field_name, item in value.items():
            fresh, count = _reference_image_payloads(item, store, key=key, floor=floor)
            moved += count
            copied[field_name] = fresh
        return (copied, moved) if moved else (value, 0)
    if isinstance(value, list):
        moved = 0
        items: list[Any] = []
        for item in value:
            fresh, count = _reference_image_payloads(item, store, key=key, floor=floor)
            moved += count
            items.append(fresh)
        return (items, moved) if moved else (value, 0)
    return value, 0


def _map_tool_results(
    value: Any, transform: Callable[[dict[str, Any]], dict[str, Any]]
) -> tuple[Any, int]:
    """Copy ``value`` rewriting every payload-bearing ``tool_execution_end`` result.

    ``transform`` must return a NEW dict: the caller's input frame is shared with
    every recipient of the relay, exactly as above. Recursing rather than
    reaching for known paths is what makes this work for both envelopes the
    chokepoint sees — an ``event`` frame carries the end directly, a
    ``frontend_update`` carries one inside ``changes["live_events"]`` or a job's
    ``job_trajectory_appends``.
    """
    if isinstance(value, dict):
        if value.get("type") == "tool_execution_end" and isinstance(value.get("result"), dict):
            return {**value, "result": transform(value["result"])}, 1
        moved = 0
        copied: dict[str, Any] = {}
        for field_name, item in value.items():
            fresh, count = _map_tool_results(item, transform)
            moved += count
            copied[field_name] = fresh
        return (copied, moved) if moved else (value, 0)
    if isinstance(value, list):
        moved = 0
        items: list[Any] = []
        for item in value:
            fresh, count = _map_tool_results(item, transform)
            moved += count
            items.append(fresh)
        return (items, moved) if moved else (value, 0)
    return value, 0


def _shed_tool_result_payloads(frame: dict[str, Any], cap_bytes: int) -> dict[str, Any]:
    """Bound the result payloads of the tool ends in ``frame``, on a copy.

    The share is RESIDUAL, measured the way :func:`_frame_size_without_delta`
    measures: the frame is re-measured with every such result's payload blanked
    (empty ``content``, ``details`` None) and what is left under ``cap_bytes`` is
    what the payloads may spend. A fixed budget would be wrong in both
    directions here — the frame's other content (a compacted transcript row, a
    deep roster) is not this pass's to spend, and what it leaves varies by
    orders of magnitude.

    The bounding itself is ``frontend_state._bound_live_result_in_place``, the
    same machinery the in-flight seed uses, so a card that sheds here and a card
    that sheds on reconnect elide the same thing and say the same thing about
    it. The event is never dropped: the EVENT is what settles the card, which is
    why the seed keeps its rows too.

    Two fallbacks, in order, before the frame is handed to the guard. The
    residual share can only buy each row its legible text floor if it is at
    least that large, and below it the bound overshoots by the floor it just
    promised; when the bounded frame still does not fit, an EMPTIED result is
    tried, because a settled card with nothing in it still beats the notice,
    which loses the event and with it the settling. Only a frame with no tool
    end at all — or one too large regardless of them — reaches the guard.

    The emptied form is tried in BOTH places the bounded one can fail, including
    the band where the share itself is non-positive. There the whole frame is
    within ``_FIT_SHED_RESERVE_BYTES`` of the line: the bounded form cannot be
    afforded, but the frame that fits is exactly the emptied one, and returning
    the original instead degraded a delta that had a settled-card form available
    (`degraded: True`, which costs the viewer a full re-sync).
    """
    from local_operator.session.frontend_state import _bound_live_result_in_place

    emptied, count = _map_tool_results(
        frame, lambda result: {**result, "content": [], "details": None}
    )
    if not count:
        return frame
    emptied_size = _frame_line_bytes(emptied)
    residual = cap_bytes - emptied_size - _FIT_SHED_RESERVE_BYTES
    if residual <= 0:
        return emptied if emptied_size <= cap_bytes else frame
    share = residual // count

    def bounded(result: dict[str, Any]) -> dict[str, Any]:
        # ``_bound_live_result_in_place`` edits its argument and its blocks in
        # place, and the argument here is a slice of the shared producer frame.
        fresh = copy.deepcopy(result)
        _bound_live_result_in_place(fresh, share=share)
        return fresh

    shed, _ = _map_tool_results(frame, bounded)
    if _frame_line_bytes(shed) <= cap_bytes:
        return shed
    if emptied_size <= cap_bytes:
        return emptied
    return shed


def fit_frame_for_wire(frame: dict[str, Any], cap_bytes: int) -> dict[str, Any]:
    """Return ``frame`` prepared for the socket line, or the honest stand-in.

    WHY A FIT PASS IN FRONT OF THE GUARD. :func:`relay_frame_or_degraded` is
    honest but destructive: it sheds the WHOLE frame, so an oversized
    ``tool_execution_end`` never reaches the viewer, the live tool card never
    settles (``session/attached.py`` carries ``_pending_tool_ends`` and the
    ``⊘ interrupted`` fallback for exactly that stranded card), and the viewer
    falls back to a full re-sync. Measured on the operator's own session bytes
    (digest ``b8758f0a``, a 317,726-byte page render, base64 through the real
    store): one ``message`` with two of them serializes to an 847,600-byte
    frame, and two such rows to 1,695,113 bytes — 1.62x the 1 MiB line
    ``start_server(..., limit=_MAX_LINE_BYTES)`` enforces.

    The asymmetry that says this is the right layer: the DURABLE path already
    solved the shape. ``transcript._externalize_attachments`` moves any block
    over ``_ATTACHMENT_FLOOR_BYTES`` into the content-addressed store and leaves
    ``{"attachment": <digest>, "mime_type": ...}``, which every frontend
    already resolves — the desktop route, the phone daemon, and the resume path
    all read that key. The live event stream was the only route still shipping
    the base64.

    Ordered stages, cheapest first, each re-measured because each can be enough:

    1. Fits already -> returned UNCHANGED (identity), so the common path costs
       the single ``json.dumps`` it always cost.
    2. Image payloads moved to the attachment store, under the durable path's
       own key. Never partial: a store that refuses the write keeps the inline
       payload and the frame simply stays large.
    3. A payload-bearing ``tool_execution_end`` still oversized -> its result
       payloads bounded on a copy, so the EVENT survives to settle the card
       while only its payload is replaced by the existing honest marker.
    4. :func:`relay_frame_or_degraded`, unchanged in behaviour and authority and
       still the terminal step. Only a frame that is genuinely unfittable — an
       image-free oversize, a store that cannot be written, a payload the
       residual share cannot buy down — reaches it, and it still says so at
       ERROR because that is now an alarm rather than the normal path.
    """
    original = _frame_line_bytes(frame)
    if original <= cap_bytes:
        return frame
    op = frame.get("op", "frame")
    referenced, moved = _reference_image_payloads(
        frame, AttachmentStore(), key=ATTACHMENT_KEY, floor=_ATTACHMENT_FLOOR_BYTES
    )
    if moved:
        if _frame_line_bytes(referenced) <= cap_bytes:
            logger.info(
                "session runtime: fitted an oversized %s frame to the socket line: "
                "%d -> %d bytes, %d image payload(s) moved to the attachment store",
                op,
                original,
                _frame_line_bytes(referenced),
                moved,
            )
            return referenced
        frame = referenced
    shed = _shed_tool_result_payloads(frame, cap_bytes)
    if shed is not frame:
        # Name only what actually happened: the shed stage runs on frames with
        # no image payload at all (a 2 MB text result is the measured case), and
        # "0 image payload(s) moved" in the one log line an operator can find
        # reads as a bug in the fit rather than as the stage that did the work.
        what = "tool result payload(s) bounded in place of degrading the event"
        if moved:
            what = f"{moved} image payload(s) moved to the attachment store, then {what}"
        logger.info(
            "session runtime: fitted an oversized %s frame to the socket line: "
            "%d -> %d bytes; %s",
            op,
            original,
            _frame_line_bytes(shed),
            what,
        )
        frame = shed
    return relay_frame_or_degraded(frame, cap_bytes)


def _compose_reuses_key(retained: dict[str, Any], incoming: dict[str, Any]) -> bool:
    """Whether ``incoming`` is a DIFFERENT call that reused a compose key.

    ``argument_bytes`` is the running size of the arguments dictated so far, so
    within one call it only ever grows. A decrease therefore cannot happen on
    the call that is already holding the slot — it proves the key was recycled
    by a new call, and folding the two together would splice one call's
    progress onto another's row.

    This is the second half of the step-boundary defence, and it is deliberately
    independent of the first: the boundary reset needs a ``turn_start`` to be
    present in the SAME compaction batch, which holds for a viewer stalled
    across a step but not for one whose queue happens to contain only compose
    frames from either side of it. Monotonicity needs no boundary frame at all.
    Both are cheap, and the failure they guard — a snapshot folded backwards
    past an execution start — is the exact stranded row this fold exists to
    remove.

    Missing or non-integer counts are treated as NOT a reuse: the fold is the
    safe default here (an equal or absent count is what an un-throttled repeat
    of the same call looks like), and refusing on every malformed frame would
    reopen the overflow this method exists to close.
    """
    previous = (retained.get("data") or {}).get("argument_bytes")
    current = (incoming.get("data") or {}).get("argument_bytes")
    if not isinstance(previous, int) or not isinstance(current, int):
        return False
    return current < previous


def relay_frame_or_degraded(frame: dict[str, Any], cap_bytes: int) -> dict[str, Any]:
    """Return ``frame``, or a small stand-in when it cannot be read.

    WHY A FRAME THIS BIG IS NOT MERELY LARGE. The attach client dials with
    ``limit=_READ_LIMIT_BYTES`` (the same 1 MiB as ``cap_bytes``), so an
    oversized line makes its ``readline`` raise ``LimitOverrunError``. That is
    unrecoverable rather than lossy — not because the bytes stay in the buffer
    (``readline`` DRAINS them: it deletes through the separator when it found
    one and clears the buffer when it did not, the same fact ``_on_connection``'s
    INBOUND prose below depends on), but because a frame the client never sees whole is
    one it cannot apply: a relay frame is a delta carrying a sequence, and the
    client's own contract says resuming after a silent gap is drift. The pump
    therefore dies, and the viewer paints "owner sent a frame too large to
    read" — the message the operator hit — losing the whole connection over one
    delta. ``frontend_sync`` has been guarded since it caused exactly this
    (see ``oversized_frame_report``); ``frontend_update`` and ``event`` were
    not, and a single 1,039,374-byte transcript row produced a 1,129,319-byte
    ``event`` frame that killed the socket.

    DEGRADING ONE DELTA IS SAFE; KILLING THE SOCKET IS NOT. Both relay frame
    types are deltas over state the viewer can recover by other means: it holds
    durable history and re-syncs through ``frontend_sync``, which carries the
    canonical snapshot. Dropping the payload costs one animation step that the
    next snapshot supersedes. Killing the connection costs the session.

    THE STAND-IN MUST STILL BE A VALID FRAME OF ITS OWN OP, which is why this
    is not one generic placeholder:

    * ``event`` — the payload is deserialized by ``deserialize_event``, which
      REQUIRES ``type``. A bare marker raises there, and although the client
      swallows callback failures (so the pump survives), the viewer would then
      silently miss the frame with nothing said. It degrades to a ``notice``
      instead: a real event type, rendered, and honest about what happened.
    * ``frontend_update`` — canonical deltas are NOT replacement-safe. The
      client checks ``epoch``/``sequence`` and closes the connection on a gap
      by design, precisely so canonical state can never drift. A placeholder
      that dropped the sequence would trip that check and kill the connection
      this function exists to save, so the sequencing fields are carried
      through and only the oversized BODY is shed. The viewer sees a
      well-sequenced delta whose payload says it was degraded.

    Cost on the hot path: one ``json.dumps`` per frame, measured at ~48µs on a
    20 KB frame (about 41% of the send's own serialization). That is an
    ADDITIONAL serialization, not a reused one — the send does its own — and it
    is accepted deliberately: the alternative is a frame the viewer cannot read
    at all, which costs the connection.
    """
    encoded = _frame_line_bytes(frame)
    if encoded <= cap_bytes:
        return frame
    op = frame.get("op", "frame")
    # NAME WHAT GREW, not just how big it got. The size alone cannot answer the
    # only actionable question this line raises — which field needs bounding —
    # and the cost is one extra serialization of a handful of fields on a path
    # that is already the slow one: the frame is over the limit and cannot be
    # sent at all. The attribution helper has existed for the connect-time
    # ``frontend_sync`` path all along; this warning went without it, which is
    # how one machine accumulated 44,681 of these lines naming no cause.
    from local_operator.session.frontend_state import largest_frame_fields

    fields = largest_frame_fields(frame)
    logger.error(
        "session runtime: %s frame is %d bytes, over the %d-byte socket line limit%s; "
        "relaying a degraded placeholder instead of killing the connection "
        "(the viewer recovers this state through frontend_sync + durable history)",
        op,
        encoded,
        cap_bytes,
        f"; largest fields: {fields}" if fields else "",
    )
    text = (
        "A live update was too large to send and was dropped; "
        "the view refreshes from the session's own history."
    )
    if op == "frontend_update":
        data = frame.get("data")
        data = data if isinstance(data, dict) else {}
        # Sequencing preserved (see above): shed the body, never the epoch or
        # the sequence number the client's gap check depends on.
        return {
            "op": op,
            "data": {
                "epoch": data.get("epoch"),
                "sequence": data.get("sequence"),
                "degraded": True,
                "degraded_reason": text,
            },
        }
    return {
        "op": op,
        "data": {"type": "notice", "text": text, "kind": "warning"},
    }


# A projection is replaceable state. If a peer cannot accept one within this
# bound, dropping that peer is safer than blocking authority-bearing ACKs for
# every healthy front end. Daemon and legacy attach clients still receive
# full projections; full-TUI clients (events + frontend_state) do not, so
# they get a longer bound — a 1 s stall on a TUI reflow was dropping a
# healthy viewer and synthesising a false "interrupted".
_SEND_TIMEOUT_S = 1.0
_TUI_SEND_TIMEOUT_S = 5.0
#: How long shutdown waits for a shielded frontend bind to land before the
#: runtime's loop can disappear — see ``RuntimeServer._drain_abandoned_binds``.
#: The same order as ``RuntimeServer.close``'s own bounded join (2.0s).
_ABANDONED_BIND_DRAIN_S = 2.0
# Raw events are lossless only while a follower keeps pace. One bounded FIFO per
# event client prevents a non-reader from retaining an unbounded stream before
# its active drain reaches the timeout; overflow drops that client so it can
# reconnect through durable history + canonical frontend_sync instead of drift.
_EVENT_QUEUE_MAX = 64

#: Drop reasons that are an ordinary part of a client's life, kept at INFO: a
#: close the runtime itself was asked for (``runtime shutdown``), a peer that
#: closed first, and a daemon dial that superseded its own predecessor. Every
#: OTHER reason means the runtime removed a client that had not asked to leave —
#: an attach-cap eviction, a queue overflow, a send timeout — which is exactly
#: the event a viewer learns about only as a cold facade, so those are logged at
#: WARNING. Derived from the call sites rather than guessed: see the eleven
#: ``_drop_client`` callers, and keep this list beside any new one.
#:
#: ``frontend requested but unsupported`` is deliberately NOT in this set, and
#: the call is a decision rather than an oversight (review n1). It reads like a
#: client-caused drop, but the level is chosen by what the user sees, and what
#: they see is identical to an eviction: a viewer that asked to be kept live is
#: cut off and reads cold next. The cause is also permanent rather than
#: transient — a runtime whose handler has no ``subscribe_frontend`` refuses
#: every reconnect the same way — so burying it at INFO would make the one
#: recurring reason a viewer keeps going cold the one reason the log does not
#: show without turning INFO on for the whole runtime.
_GRACEFUL_DROP_REASONS = frozenset(
    {"runtime shutdown", "reader eof", "reader reset", "daemon replaced"}
)

# Ops whose answer is structured data (a typed slash result, a cancel count)
# rather than a one-line receipt: they reply with a ``result`` frame so the
# invoker renders the outcome locally instead of the owner's transcript
# printing it.
#: How long ``announce_stop`` will wait for its frame to be handed to the
#: transport on the thread-hosted path. Bounds a courtesy write against a
#: stalled viewer: the caller is the TUI's event loop during a /stop, so this
#: is a frozen-UI budget, not a delivery guarantee. A viewer that misses the
#: frame degrades to the pre-announcement behaviour; the stop is unaffected.
_ANNOUNCE_WRITE_TIMEOUT_S = 0.25

#: How long a thread-mode serve loop waits between close-latch re-checks.
#: ``_request_close`` wakes the loop directly (``_wake_close_wait``), so this is
#: a BACKSTOP rather than the mechanism: work is signalled, not polled — the
#: same choice ``analytics/recorder.py`` makes when it wakes its writer with a
#: queue sentinel instead of sleeping on a flag. A signal can still be missed (a
#: close that lands before the loop published its event, or a loop that has
#: already stopped), and a serve loop that parks forever would hang the join in
#: ``close()`` and leave the listener bound, so the wait keeps a timeout.
_CLOSE_WAIT_BACKSTOP_S = 0.2

#: How long ``wait_until_published`` waits for ``start()``'s record to land.
#: ``start()`` hands the work to a thread and returns, so the caller's next
#: statement still sees ``control_port == 0`` and a record file that does not
#: exist yet. This is therefore a bound on a FAILURE rather than on the normal
#: path — the thread's first act is a loopback bind, measured in single-digit
#: milliseconds — and it is deliberately generous, because the two errors are
#: not symmetric: a caller that gives up early reports a control surface that is
#: about to exist, while one that waits an extra few seconds only delays a boot
#: that is already broken.
_PUBLISH_WAIT_TIMEOUT_S = 15.0

#: Sentinel for ``_retire_if_pristine``'s single-hop re-check. The re-check and
#: the stop go to the session's loop TOGETHER (see that method), so "work
#: arrived" has to come back as a value rather than as an early return, and the
#: value crosses a thread boundary — a module-level singleton is comparable by
#: identity from either side, where a local one would only work by luck.
_WORK_ARRIVED = object()

#: How long ``_shutdown_impl`` waits for requests the reader loops have ALREADY
#: admitted to finish answering, before it closes the sockets they answer on.
#:
#: Five seconds, and the number is chosen from the two bounds it sits between:
#: a hop into another thread's event loop is milliseconds when that loop is
#: healthy (measured: a TUI hop returns in 0.0-0.1 s), and every client
#: speaking to this socket gives a reply 15 s (``attach_client.ACK_TIMEOUT_S``)
#: — so a shutdown that waits a few seconds gets the ack out while still being
#: far below the caller's own patience. The wait exists because the ``stop`` op
#: is answered by a hook that tears this runtime down (see
#: ``_await_in_flight_requests``); a longer grace would buy nothing (a request
#: still parked after 5 s is parked on something that is not going to answer)
#: and would delay every genuine shutdown by that much.
_SHUTDOWN_REPLY_GRACE_S = 5.0

_PAYLOAD_OPS = {
    "slash_result",
    "cancel_subagents",
    "job_trajectory",
    "fork_snapshot",
    "credential",
    "mcp_credentials",
    # Session code memory: the desktop canvas panel's list/create/update/delete
    # verbs over the session's live eval-kernel namespace. A payload op rather
    # than a receipt op because its answer IS the data the panel renders (and a
    # `busy`/refusal state it must not paint as an empty list). Additive on the
    # wire: an older runtime does not list it in `_PAYLOAD_OPS`, so it answers
    # `unknown op`, which the viewer reports as `unsupported` — the honest
    # "this backend cannot read code memory" the panel has a sentence for.
    "variables",
    # The §6 redaction forward: the ONE other op that carries a secret's value,
    # and only in its own named field. It is a distinct op from ``credential``
    # on purpose — the handler registers the value with the runtime's redactor,
    # it does not store a credential — so it must never be reachable through
    # that verb table. Old runtimes answer unknown-op, which the viewer's sink
    # treats as a failure and (correctly) fails closed on.
    "register_secret_redaction",
    "history_page",
    "frontend_sync",
    "record_shell",
}

#: Ops a connection may run while its canonical ``frontend_sync`` is still being
#: built and is not on the wire yet.
#:
#: The bind is a cross-thread hop onto the session's loop, so a session inside a
#: synchronous step of a turn parks it for the length of that step. Running the
#: bind inline therefore made a follower's socket DEAF rather than slow: the
#: reader loop had not started, so nothing the client sent was read at all — not
#: even a ``ping`` — while a daemon or phone dial to the same runtime kept
#: answering (measured: review round 2, UX U6). ``_on_connection`` now starts the
#: bind as a task and enters the reader loop first, which is what makes this set
#: necessary: a connection that is reachable but not yet AUTHORITATIVE must not be
#: allowed to act on state it has not been told about.
#:
#: So: health, and the FOUR ways to regain control of a turn — ``stop``,
#: ``abort``, ``steer`` and ``cancel``. The first three (with ``cancel``, whose
#: ``immediate`` mode routes to ``abort``) are the kill switch in its rungs, and
#: withholding them until a sync lands would deny a supervisor the ability to
#: stop a runaway session precisely when its loop is stuck — the situation this
#: set exists for.
_SYNC_PRIORITY_OPS = frozenset({"ping", "stop", "abort", "steer", "cancel"})

#: Ops exempt from their connection's op CHAIN. A different question from
#: :data:`_SYNC_PRIORITY_OPS`: that set decides what may run before the sync
#: lands (ADMISSION), this one decides what may run without waiting for an
#: earlier op to finish (ORDERING).
#:
#: ``ping`` alone, and the argument is that its answer cannot depend on session
#: state: it reports that the runtime's loop is alive and serving. Chaining it
#: made it report something else entirely — measured over a real socket (review
#: round 1, UX U3), a ``ping`` sent after a parked ``steer`` on the SAME
#: connection went unanswered for 8-15 s, so the one request a surface speaks to
#: ask "are you there" was queued behind a mutation. Everything else keeps its
#: place in the chain, because ordering is what stops two mutations interleaving
#: and only a liveness probe has no state to be ordered against.
_UNCHAINED_OPS = frozenset({"ping"})


#: Connection-LOCAL ops admitted alongside the priority set above. Not a widening
#: of it: each mutates only this connection's own relay state and never touches
#: the session, so none can act on a connection that has not yet been made
#: authoritative — which is the whole reason the priority set is closed. They
#: must be admitted, because the dial path re-asserts some of them immediately
#: after reading the welcome and BEFORE it awaits the sync frame, each under its
#: own bound; refusing those would turn every reconnect of a parked viewer into
#: an error frame.
#:
#: READ THE GATES PER MEMBER; DO NOT SUMMARISE THE SET. This paragraph carried a
#: count and a class for four consecutive review rounds and was wrong every time
#: (the artifact held a different set each round: ``watch_job``/``unwatch_job``
#: were called daemon-gated although they carry no shape gate at all, and the
#: attach-gated members went unnamed). So, from the ``_on_request`` arms:
#:
#: * ``watch`` / ``unwatch`` — gated to a REGISTERED DAEMON, and SILENTLY so:
#:   the arm runs only for ``conn.kind == "daemon"`` on a registered writer; an
#:   attach client's frame is accepted and changes nothing, deliberately, because
#:   only the daemon's count is ever cleared. Present here for the daemon's dial.
#: * ``watch_job`` / ``unwatch_job`` — NO SHAPE GATE AT ALL. The arm validates
#:   only that ``job_id`` is a non-empty string and then adds or discards it on
#:   this connection's own ``watched_jobs``; attach and daemon are treated
#:   identically.
#: * ``desktop_watch`` — REFUSED BY SHAPE: a registered ``kind == "attach"``
#:   connection whose ``surface == "desktop"``, plus boolean ``visible`` /
#:   ``can_notify``; anything else gets the error frame.
#: * ``viewer_watch`` — REFUSED BY SHAPE: a registered ``kind == "attach"``
#:   connection and a boolean ``displaying``.
#: * ``event_mute`` / ``event_unmute`` — REFUSED BY SHAPE: attach-only, because
#:   the relay they mute is never sent to a daemon at all.
#:
#: The dial path depends on the event mute, ``viewer_watch`` and ``desktop_watch``
#: — three of the four refusal-by-shape members enumerated above — which
#: ``session/attached.py`` re-asserts right after the welcome, exactly in the
#: window where the canonical sync is still in flight. The ``watch`` and
#: ``watch_job`` families are here for the daemon and child-page dials that send
#: them, not for that reconnect.
#:
#: Their ``_dispatch`` push exemption is what makes them a MIRROR of an existing
#: decision rather than a second one, and it covers ``watch``, ``unwatch``,
#: ``watch_job``, ``unwatch_job``, ``event_mute`` and ``event_unmute`` — every
#: member of this set except ``desktop_watch`` and ``viewer_watch``, which are
#: exempt from nothing there because they answer with a receipt rather than a
#: repaint.
_SYNC_LOCAL_OPS = frozenset(
    {
        "watch",
        "unwatch",
        "watch_job",
        "unwatch_job",
        "desktop_watch",
        "viewer_watch",
        "event_mute",
        "event_unmute",
    }
)


def _running_loop() -> asyncio.AbstractEventLoop | None:
    """The loop this thread is running, or ``None`` on a plain thread.

    The single spelling of "which loop am I on" for the two places on
    ``RuntimeServer`` that must answer it — ``_on_runtime_loop`` (is the caller
    THIS runtime's owner?) and ``_handle_call_on_session_loop`` (is the caller
    already on the SESSION's?). A bare ``get_running_loop()`` in either would
    raise on a plain thread, and both have legitimate plain-thread callers:
    ``announce_stop`` documents one, and any synchronous wrapper around a
    handle call is another.
    """
    try:
        return asyncio.get_running_loop()
    except RuntimeError:
        return None


async def _maybe_await(result: Any) -> Any:
    """``result``, awaited first if it is awaitable.

    A handle method may be a plain ``def`` returning a value (``subscribe``
    hands back an unsubscribe closure) or an ``async def`` (``subscribe_frontend``
    hands back a subscription). The code that HOPS a call has to preserve
    whichever shape it was given, and it must not impose one on the other: a
    hop that returned a coroutine object to be awaited by the caller would run
    the body on the wrong loop, and a hop that insisted on an awaitable would
    break the synchronous half of the same protocol.
    """
    if inspect.isawaitable(result):
        return await result
    return result


#: Every control op that can reach an authority-INCREASING sink (issue #1310).
#:
#: The set itself lives in ``harness/approval.py`` beside the class predicate,
#: because the CONSOLE reads it too (it presents the capability on exactly these
#: frames) and a set that drifted between the two ends would leave a route the
#: client believes it authorised and the server believes is ordinary. What is
#: asserted here is the correspondence with THIS module's dispatch: every op
#: whose dispatch reaches ``SessionHandle.slash`` / ``slash_images`` /
#: ``run_slash_authoritative`` / ``approval_answer`` — the only ways to
#: ``_approvals_slash``, ``_set_approve_all`` or a card approval — must appear in
#: the set, and ``test_approval_authority_seam.py`` re-derives that group from
#: this file's source so a route added in a new op fails the suite instead of
#: shipping an unguarded way to loosen a running gate.
_AUTHORITY_OPS = AUTHORITY_OPS

#: How long a minted challenge stays usable. Short on purpose (revision 2, §2.3
#: spells 30 s): the window between a surface asking for a challenge and signing
#: it is one human gesture, and a long window is a long time for a captured
#: challenge to be spent by somebody else. Expiry is checked at USE, not at mint,
#: so a challenge that outlives a slow prompt is refused rather than silently
#: honoured — the surface asks again, which costs one more prompt only in the
#: case where the first one took half a minute.
_CHALLENGE_TTL_S = 30.0

#: How many challenges one connection may hold. Bounded because minting is an
#: ordinary op and therefore unauthenticated beyond the record key: without a
#: cap, a same-uid child could mint challenges in a loop and grow this runtime's
#: memory without limit. Eight is far more than a real surface needs (it asks for
#: one per action, signs it, and consumes it).
_MAX_CHALLENGES_PER_CONN = 8

#: Bound on unspent challenges for the WHOLE runtime, not per connection.
#:
#: The per-connection cap alone is not a bound on a subject that can dial freely:
#: the session record's ``control_key`` — which is what the very predicate this op
#: exists for refuses to accept as authority — is readable by the subject, so it
#: can open as many connections as it likes and hold the per-connection maximum on
#: each (agent review round 6, R6-4). The design already accepts denial rather than
#: escalation from that subject, so this is not an escalation either; it is the
#: difference between a bounded and an unbounded one. Set well above any real
#: surface's need — one human gesture is one challenge — and expired entries are
#: pruned before it is consulted, so an idle runtime never refuses a real caller.
_MAX_LIVE_CHALLENGES = 64

#: How long a runtime may keep trusting the revocation list it read at first need.
#:
#: THE SAME OBJECT THE PRODUCT QUOTES, not a second literal that happens to agree
#: (agent review round 7, M-2). ``trust.ANCHOR_REFRESH_S`` is what the design doc
#: and `lop operator devices --revoke`'s receipt state to an operator, so a runtime
#: enforcing its own copy could tell someone a window it does not honour — a
#: security claim outrunning the code, which is the class this round exists to
#: remove. Bound here as a module attribute as well, because a test shrinks the
#: window to zero by patching ONE name and this is the name the cache is built from.
_ANCHOR_REFRESH_S = ANCHOR_REFRESH_S

#: How long a verified device certificate is remembered. The certificate is
#: checked lazily and per certificate string; a TTL rather than a permanent cache
#: because the anchor's revocation list can change under a long-running session.
_DEVICE_CERT_TTL_S = 300.0

#: Bound on the device-certificate cache. Keyed by an attacker-chosen string, so
#: an uncapped map is a memory leak with an on-demand trigger; clearing wholesale
#: on overflow is enough, because the entries are pure recomputable answers.
_MAX_CACHED_DEVICE_CERTS = 32


def _accepts_kw(fn: Any, name: str) -> bool:
    """Whether ``fn`` takes the keyword ``name``.

    Cached because it is asked on every routed slash command and
    ``inspect.signature`` is not cheap. Keyed by the underlying function so
    bound methods of the same class share one answer. A handle whose signature
    cannot be read (a C callable, an exotic mock) is treated as not accepting
    it: the caller then uses the narrow call, which every implementation has
    always supported.

    Generalised from a ``locality``-only probe when ``slash_consumers`` became
    the second optional keyword on the same call. A handle is an INJECTED
    collaborator — the TUI's, the runtime's, and several test doubles all
    implement ``run_slash_authoritative`` — so each new keyword must stay
    optional in the protocol rather than force a lockstep change across every
    implementation. Two independent probes rather than one combined answer:
    a handle may well accept one keyword and not the other.
    """
    target = getattr(fn, "__func__", fn)
    by_name = _KEYWORD_SUPPORT.get(target)
    if by_name is not None:
        cached = by_name.get(name)
        if cached is not None:
            return cached
    else:
        by_name = {}
        _KEYWORD_SUPPORT[target] = by_name
    try:
        params = inspect.signature(target).parameters
    except (TypeError, ValueError):
        answer = False
    else:
        answer = name in params or any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
        )
    by_name[name] = answer
    return answer


#: Memo for ``_accepts_kw``, keyword name → answer per function. Keyed weakly
#: so a handle class that goes away (a per-test double) does not pin its
#: function objects in memory.
_KEYWORD_SUPPORT: "weakref.WeakKeyDictionary[Any, dict[str, bool]]" = weakref.WeakKeyDictionary()


#: Rows one ``job_trajectory`` reply may carry. The whole retained window is
#: 500 events with no size bound per event, which is what overflows the frame
#: limit in the first place, so the viewer pages rather than asking for all of
#: it: 120 rows of ordinary tool traffic sit far inside ``_MAX_LINE_BYTES``
#: while keeping the round trips for a full window in single digits.
_TRAJECTORY_PAGE_MAX = 120


@dataclass(frozen=True)
class AckDetail:
    """An op's ack: the one-line receipt, plus state the CALLER must verify.

    Most ops answer with a sentence and a sentence is all their caller needs.
    The receipt op cannot be one of them: its caller has to know whether this
    call actually moved the read watermark, and the projection it would
    otherwise read cannot tell it. ``frontend_update`` is delivered on the
    connection's event queue while the ack is written directly, so a follower
    resolves its ack a whole writer ahead of the state that ack produced -- the
    honest receipt reads as a lost one (agent review round 1, R4). Carrying the
    state the owner computed, on the ack itself, is what makes "verify, never
    assume" possible on an attached session at all.
    """

    detail: str
    attention: dict[str, Any]


@dataclass
class _ClientConn:
    """One authenticated control connection in the runtime's registry.

    Multiplexing is already half-there (frames carry caller-chosen ``req``
    ids), so multi-front-end needs N concurrent connections on the ONE
    socket rather than a second protocol: the daemon plus up to
    ``ATTACH_MAX_CLIENTS`` attach terminals. ``last_seen`` is the LRU clock
    for attach eviction — stamped on every request so the least-recently-ACTIVE
    follower is the one dropped when the cap is hit.
    """

    writer: asyncio.StreamWriter
    kind: ClientKind
    #: Whether the human on the other end is at this machine. Declared in the
    #: auth frame; see ``ClientLocality``. Only ops that act on the USER's
    #: surroundings (an OAuth browser tab) read it.
    locality: ClientLocality = "local"
    #: Which host is on the other end. ``"desktop"`` is declared in the attach
    #: frame; the remaining fields are that host's presentation state, which
    #: only it reports and only it is scored on.
    surface: str = "terminal"
    desktop_visible: bool = False
    desktop_can_notify: bool = False
    desktop_seen: float = 0.0
    #: Whether a TERMINAL attach is currently DISPLAYING this session.
    #:
    #: A multiplexing TUI keeps the outgoing session's connection open when the
    #: user switches away (the outgoing source is retained for the sidebar), so
    #: "connected" stopped implying "on screen" the moment one viewer could
    #: show several sessions. Routing a parked gate on the connection alone
    #: therefore suppressed the out-of-band toast for a card painted into a
    #: viewer showing something else.
    #:
    #: DEFAULTS TRUE, which is what makes this safe to add: a client that never
    #: sends ``viewer_watch`` (every build before this field, and every
    #: non-TUI attach) keeps counting exactly as it did. Only a client that
    #: explicitly declares it switched away is discounted, so the change can
    #: never invent a silent session out of an older viewer.
    terminal_displaying: bool = True
    #: Which action-carrying slash receipts this client renders itself (see
    #: ``SLASH_ACTION_RECEIPTS``). ``None`` means the auth frame omitted the
    #: field, i.e. a client built before it existed — the runtime then
    #: completes the action on its behalf. An EMPTY frozenset is a different
    #: fact from ``None`` only in provenance, not in effect: both mean "this
    #: type was not declared", which is the one rule the completion path
    #: applies.
    slash_consumers: frozenset[str] | None = None
    last_seen: float = field(default_factory=time.monotonic)
    # Frames on one TCP stream must stay ordered, while unrelated streams must
    # never queue behind its backpressure.
    send_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    # v4: this attach client asked for the raw AgentEvent relay in its auth
    # frame. Daemon connections never set it; a v3 attach client that omitted
    # the flag keeps projection-only behaviour.
    wants_events: bool = False
    #: An attach connection's raw event relay is MUTED: it asked (see
    #: ``EVENT_MUTE_CAPABILITY``) to stop receiving delta-grade frames until it
    #: unmutes. Per connection, not per session — the same owner serves each
    #: viewer's own interest, so a parked viewer's mute never slows the one on
    #: screen. Flipped by the ``event_mute``/``event_unmute`` ops (handled in
    #: the control loop, which owns ``conn``) and read in ``_relay_on_loop``.
    events_muted: bool = False
    # v5 canonical state is attach-only and independently negotiated so daemon
    # projection bytes never gain frontend frames.
    wants_frontend: bool = False
    #: The per-connection operator proof material (issue #1310). The client
    #: offers a NONCE in its auth frame; this runtime answers with a random salt
    #: and a proof over both, and later demands the same construction on an
    #: authority-increasing request. Neither value is secret, and both die with
    #: the connection, so a proof seen on the wire is worthless on another one.
    #: Empty when the client offered no nonce — an old console, or one that
    #: holds no capability for this runtime — which is the fail-closed state.
    operator_nonce: str = ""
    operator_salt: str = ""
    #: The operator-signed device certificate this connection declared in its
    #: auth frame, or "" when it declared none. NOT yet verified at the point it
    #: is stored — ``_device_cert_point`` is what resolves it under the anchor,
    #: with the TTL cache and the revocation check — and every reader goes
    #: through that, so an unverified string here can only change an answer to
    #: ``False``. Kept per connection rather than per frame because the report
    #: needs it before a frame arrives; the SIGNATURE still arrives on the frame
    #: that claims authority, and is judged there.
    device_certificate: str = ""
    #: The per-action challenges this connection has been minted and not yet
    #: spent, keyed by ``(action, request_id)``. Per CONNECTION for the same
    #: reason the nonce is: a challenge is authority-bearing material, and one
    #: minted for a connection must not be spendable on another (a relay, or an
    #: attacker that merely read the record, would otherwise be able to have a
    #: challenge minted here and present the signature it harvested there).
    #: Consumed by ``pop`` on the first frame that uses it, which is the replay
    #: defence; expired entries are pruned on the next mint so a client that asks
    #: and never signs cannot grow this map
    #: (``_MAX_CHALLENGES_PER_CONN`` bounds it either way).
    operator_challenges: dict[tuple[str, str], tuple[str, float]] = field(default_factory=dict)
    #: This viewer negotiated ``display-history-audit-v1`` and can therefore be
    #: sent the audit fields on a display page. A property of the CONNECTION,
    #: so it is read where the connection is known and never inferred from the
    #: frame; see ``DISPLAY_HISTORY_AUDIT_CAPABILITY`` for what emitting them
    #: to a viewer that did not negotiate would do.
    audit_history: bool = False
    frontend_ready: bool = False
    #: True only while ``_push_to`` is writing THIS connection's welcome. It is
    #: what tells ``_readable_frame`` whether an unreadable projection has a
    #: canonical sync coming behind it on the same connection (the welcome does;
    #: a mid-stream repaint does not), because the frame itself cannot say —
    #: "projection" is the op for both. Scoped to the send rather than latched
    #: afterwards so it can never be read as "this connection has had a welcome
    #: at some point".
    sending_welcome: bool = False
    # Updates can be scheduled back to this loop while the owner-loop
    # subscription call is returning. Hold them until the sync frame is queued;
    # dropping them creates an immediate sequence hole at every busy join.
    frontend_pending: list[dict[str, Any]] = field(default_factory=list)
    # Flipped only AFTER welcome + canonical frontend_sync are queued. Raw
    # events begin behind that boundary on the same FIFO, so a joining client
    # cannot see transcript animation ahead of the snapshot that seeded it.
    events_ready: bool = False
    #: The canonical frontend sync for this connection is still being built and is
    #: not on the wire yet. Distinct from ``frontend_ready``, which is the
    #: RECIPIENT set's gate ("frontend frames may be enqueued now"): this one is the
    #: ADMISSION gate, and it is what ``_on_request`` consults to decide whether a
    #: connection may run a session-facing op at all. Set before the sync task is
    #: created and cleared when that task settles, so the two can never disagree
    #: about a LIVE connection (a failed sync drops the client).
    frontend_sync_pending: bool = False
    #: The task that binds this viewer and queues its sync frame — see
    #: ``RuntimeServer._serve_frontend_sync``. Held so ``_drop_client`` can cancel
    #: it, the same contract ``event_writer_task`` keeps, because a connection that
    #: goes away mid-bind owes the session an unsubscribe and must not queue a
    #: frame onto a socket nobody owns.
    frontend_sync_task: asyncio.Task[None] | None = None
    event_queue: asyncio.Queue[dict[str, Any]] = field(
        default_factory=lambda: asyncio.Queue(maxsize=_EVENT_QUEUE_MAX)
    )
    # Exactly one writer drains the queue, so delivery stays ordered without a
    # task per event. Held for shutdown and slow-client eviction.
    event_writer_task: asyncio.Task[None] | None = None
    frontend_unsubscribe: Callable[[], None] | None = None
    #: This connection's CURRENT bind generation, and the mechanism that keeps a
    #: late on-loop bind from relaying into a connection the off-loop fallback
    #: already bound (``_serve_frontend_sync``).
    #:
    #: An attach that outlasts ``_ONLOOP_BIND_GRACE_S`` is served off-loop, but
    #: the on-loop bind it abandoned is SHIELDED and cannot be cancelled — it
    #: still lands in the store and still calls the relay. Without a stamp the
    #: same connection would see every delta twice, which is not a slow frame but
    #: a wrong one: the client's exact-``+1`` check reads the duplicate as a gap
    #: and redials. The generated callback compares the token it was stamped with
    #: against this field, so bumping it retires every callback created before the
    #: bump — the fallback's own included, if a THIRD attempt ever supersedes it.
    bind_token: int = 0
    #: The chain of ops this connection has ADMITTED: each waits for the one
    #: before it, so ordering is preserved, and none of them parks the reader —
    #: which is what lets a ``ping`` be answered while a mutation is still in
    #: flight. See ``RuntimeServer._dispatch_frame`` for the measured failure
    #: the shape fixes (a parked ``steer`` made the whole connection mute).
    op_chain: asyncio.Task[None] | None = None
    #: Strong references to those tasks. A bare ``create_task`` can be collected
    #: mid-await, which would leave an op half-run and its reply never written —
    #: the same reason ``RuntimeServer._event_sends`` exists.
    op_tasks: set[asyncio.Task[None]] = field(default_factory=set)
    # Job ids whose trajectory deltas this connection wants (``watch_job``).
    # Empty by default and per-connection by necessity: the snapshot ships no
    # trajectories at all (they overflow ``_MAX_LINE_BYTES``), so a viewer
    # opts in only for the child page a reader actually opened. A second
    # viewer watching a different child must not widen this one's stream.
    watched_jobs: set[str] = field(default_factory=set)


class SessionHandle(Protocol):
    """What the runtime needs from its host application.

    Every method is awaited on the RUNTIME'S loop (its own thread); the
    implementor guarantees any hop the session needs (for the TUI: Textual's
    ``call_from_thread``; for an owned session: ``run_coroutine_threadsafe``
    back to the daemon loop). Methods return a short human-readable receipt
    that becomes the ``ack`` detail — the same line the TUI would print as a
    notice.
    """

    @property
    def session_projection_seed(self) -> SessionProjection:
        """The projection skeleton: identity fields the runtime folds onto."""
        ...

    def subscribe(self, on_projection: Callable[[], None]) -> Callable[[], None]:
        """Feed the fold from the host's event stream; call ``on_projection``
        (thread-safe) whenever the projection changed and should be pushed.
        Returns an unsubscribe callable."""
        ...

    async def prompt(
        self,
        text: str,
        images: list[dict[str, str]] | None = None,
        command_id: str | None = None,
    ) -> str: ...
    async def steer(self, text: str, images: list[dict[str, str]] | None = None) -> str: ...
    async def abort(self) -> str: ...
    async def set_model(self, provider: str, model_id: str) -> str: ...
    async def set_effort(self, effort: str) -> str: ...

    async def slash(self, command: str, args: str) -> str: ...
    async def new_conversation(self) -> str: ...
    async def resume_session(self, session_id: str) -> str: ...
    async def approval_answer(self, request_id: str, approved: bool, remember: bool) -> str: ...
    async def ask_answer(  # noqa: E301
        self, request_id: str, value: str, question_index: int | None = None
    ) -> str: ...

    async def refresh(self) -> None:
        """Re-read session state into the projection (post-resume, rename,
        model change): the runtime pushes whatever changed."""
        ...

    # -- the kill switch: graceful self-stop (optional, probed) --------------
    # request_stop() -> None: deny parked gates, abort the turn, dispose the
    # session and begin the runtime's own shutdown, so the ``stop`` control
    # op ends this runtime the way SIGTERM would. Probed with getattr like
    # every optional capability below so reduced handles (tests, older
    # bridges) keep satisfying the protocol: a handle without it answers the
    # ``stop`` op with the unknown-op error, and the caller's escalation
    # ladder (session/runtime/control.py) proceeds to identity-confirmed
    # SIGTERM — which the runtime's signal handler has always honoured.
    # Sync and non-raising by contract: called ON the runtime loop from the
    # dispatch, and a stop that faults here is still a stop.
    #
    # -- v4 optional capabilities (probed with getattr, never required) -------
    # subscribe_events(on_event) -> unsubscribe: feed the host session's raw
    #   AgentEvent stream, serialized (``model_dump(mode="json")``) on the
    #   host's own loop, to ``on_event`` (thread-safe). The runtime relays
    #   the dicts to event-subscribed attach clients. Optional so old hosts
    #   and reduced test handles keep working — without it, attach clients
    #   simply get v3 projection-only behaviour.
    # recall_steer(command_id) -> str: unsend the queued steering message the
    #   follower submitted under ``command_id``; raises when it already
    #   drained. Optional for the same reason.
    # receive_peer_message(text, *, mode="mailbox", wake=False, sender=None)
    #   -> str: deliver a message from another local lop session (`lop send`).
    #   Optional (getattr-probed in _dispatch) so reduced test handles and
    #   non-interactive exec hosts that never wired it keep working — a handle
    #   lacking it answers "this session cannot receive peer messages".
    # cancel_gracefully() -> str: stop the turn at the POST-TOOL boundary
    #   instead of cutting the running tool (Session.request_graceful_cancel).
    #   Serves the ``cancel`` op's default mode. Deliberately distinct from
    #   ``abort`` rather than a parameter on it: abort's contract is "stop now,
    #   the human will repair the mess", and a supervised agent has no human to
    #   repair a half-finished push. Optional and getattr-probed so hosts that
    #   cannot honour a boundary (a reduced handle, an older bridge) say so
    #   plainly instead of silently doing the destructive thing.


class ProjectionSink(Protocol):
    """What the runtime needs from a projection collaborator.

    The phone renders from a :class:`SessionProjection` snapshot that the
    runtime broadcasts on change; :class:`ProjectionFold` is the production
    implementation. The runtime only ever READS ``projection`` (to serialize
    a frame) and calls ``set_pending`` (the reduced-handle bridge for gate
    cards), so that is the whole contract — narrow enough that a test can
    hand in a stub and a future runtime with no phone can hand in nothing.

    Read-only is a REAL constraint, not a description: the runtime re-dates the
    band's age through the handle's fold (the one the events are fed into) at
    frame build, and an injected sink must not do that from its own state. A
    fold that is never fed has no phase, and writing its emptiness over a shared
    projection object is what served every attached phone ``null`` for every
    phase (review round 4, BLOCKER 1).
    """

    @property
    def projection(self) -> SessionProjection: ...

    def set_pending(self, pending: Any) -> None: ...


def has_durable_history(session: Any) -> bool:
    """Whether this session's transcript already holds a REAL conversation turn.

    The seed signal for the record's ``started`` bit at two call sites that
    face the same question — ``RuntimeServer.__init__`` (a resumed boot must
    publish ``started=True`` before any turn runs in the NEW process) and
    ``TuiSessionHandle.rebind`` (a ``/resume`` mid-flight re-seeds the bit for
    the swapped identity). Thin wrapper over
    :func:`~local_operator.session.transcript.durable_conversation_path`,
    where the discriminator lives with the row shapes it reads: only a plain
    ``Message`` row counts — a CustomMessage row (a quiet-dialled
    ``peer_message`` note, a wake prompt) is persisted WITHOUT a turn running,
    and counting one seeds ``started=True`` on a session whose owner never
    typed, after which a peer ``--wake`` or broadcast drives an assistant
    turn into it (QA Q4). Read off the session's declared
    ``transcript_path`` rather than ``session.transcript.path`` — the same
    read, through the session's own contract instead of two privates deep;
    a session shape without one (a reduced host) answers False, the
    conservative "unstarted" direction a first real turn immediately
    corrects.
    """
    path = getattr(session, "transcript_path", None)
    if path is None:
        return False
    return durable_conversation_path(path)


class RuntimeServer:
    """One per interactive process. Construct, ``start()``, ``close()``."""

    def __init__(
        self,
        handle: SessionHandle,
        *,
        kind: str = "tui",
        projection_sink: ProjectionSink | None = None,
        operator_cap: bytes | None = None,
        operator_anchor: OperatorAnchor | None = None,
    ) -> None:
        #: The capability this runtime demands for an authority-INCREASING
        #: control request, or ``None`` when nothing handed one over (issue
        #: #1310). Minted by whichever process started this runtime — the
        #: detached spawn hands it over on an inherited descriptor, the TUI
        #: passes the one it minted for its own in-process gate — and held ONLY
        #: here. It is deliberately not a ``SessionRecord`` field and not on the
        #: handle's projection: everything published in the record is readable
        #: under this same uid, which is the defect this exists to close.
        #:
        #: ``None`` is a supported, fail-closed state rather than a bug: a
        #: runtime started by an older console, or by a background spawn with no
        #: console at all, keeps serving every ordinary operation and refuses
        #: every loosening (see ``_authority_admitted``).
        self._operator_cap = operator_cap
        #: The operator ANCHOR, read once and pinned in memory (revision 2).
        #: ``AnchorCache`` caches the FAILED load as well as the successful one,
        #: which is the point: re-reading per frame would let a same-uid subject
        #: race the read, and an attacker who could make the anchor unreadable
        #: could otherwise force a disk read on every frame it sends.
        #:
        #: ``operator_anchor`` is the INJECTION SEAM and mirrors ``operator_cap``:
        #: a caller that has already resolved an anchor (or a test that must not
        #: write to the root-owned path, which by construction it cannot) hands
        #: one in. It is a CONSTRUCTOR ARGUMENT rather than an environment lookup
        #: on purpose — an env-var anchor is the substitution attack this whole
        #: design exists to prevent, so the seam is a value the caller passes and
        #: never a name the process reads (see
        #: ``test_the_anchor_path_cannot_be_redirected``).
        self._anchor_cache = (
            AnchorCache(
                load=AnchorLoad(
                    anchor=operator_anchor,
                    path=anchor_path(),
                    root_owned=True,
                    reason="injected by the caller",
                    exists=True,
                )
            )
            if operator_anchor is not None
            else AnchorCache(refresh_s=_ANCHOR_REFRESH_S)
        )
        #: Verified device certificates, by certificate string, with a TTL — see
        #: ``_device_cert_point`` for why this is lazy, bounded and short-lived.
        #: ``(point, deadline, device_id)``: the id is kept so REVOCATION can be
        #: re-checked on a cache hit — see the method, and R6-1.
        self._device_certs: dict[str, tuple[bytes | None, float, str]] = {}
        #: Every unspent operator challenge in this runtime, by challenge string,
        #: with its deadline — the AGGREGATE bound the per-connection maximum
        #: cannot be (see ``_MAX_LIVE_CHALLENGES``). Pruned on mint and on
        #: consume, so it never needs a timer and never outlives what it counts.
        self._live_challenges: dict[str, float] = {}
        #: Live state mirrored into the discovery record. Held here rather
        #: than read off the record so the publish is one assignment and the
        #: fields have a defined value before the record exists.
        self._pending: str | None = None
        #: True once this session has run at least one real turn. Starts
        #: False for every fresh boot (a ``/new`` session sitting in the
        #: composer) and flips True the first time a turn actually runs — the
        #: hook lives in ``Session._run_turn_pipeline``. One exception at
        #: birth, below: a boot that RESUMED a conversation with history
        #: seeds True from the transcript, because those turns already ran
        #: under an earlier process.
        self._started = False
        self._busy = False
        #: ``LEAVING_ON_SIGNAL`` once the signal drain has committed this
        #: runtime to an exit, else ``""``. Held on the server like the other
        #: live-state fields above so one assignment publishes it, and set from
        #: the drain (``RuntimeServer.note_leaving``) rather than read off the
        #: handle: this is the runtime's own decision to leave, which no handle
        #: predicate knows.
        self._leaving = ""
        #: The update window this runtime has opened (``SessionRecord.updating``),
        #: kept here so :meth:`note_updating` can dedupe like :meth:`note_leaving`.
        #: The HANDLE owns the window's admission behaviour and its lock; this is
        #: only the record's copy, written from the one place that owns the record.
        self._updating = ""
        #: The pair a window FAILED to move to, for :attr:`SessionRecord.update_failed`.
        self._update_failed = ""
        #: Subagent trajectory counts, ``None`` until the handle answers the
        #: probe at least once. Starting at ``None`` rather than 0 is what
        #: makes a runtime whose handle cannot report indistinguishable from
        #: one that predates the fields — both are honestly "unreported", and
        #: neither is a measured zero.
        self._subagents_running: int | None = None
        self._subagents_queued: int | None = None
        #: True until a terminal attaches. A freshly spawned runtime genuinely
        #: has no viewer, so this starts True rather than False — the old
        #: default had every new runtime claiming a terminal it had never had.
        self._detached = True
        self._desktop_delivery = False
        self._handle = handle
        # Back-reference so the handle can publish record state it alone knows
        # about — today the parked-gate ``pending`` bit, which originates deep
        # inside the approval gate and has to reach the discovery record for
        # `lop sessions` and the picker to show it. Set defensively: reduced
        # handles in tests are plain objects and must not fail on an attribute
        # assignment they never asked for.
        try:
            handle._registrant = self  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001 — an unwritable handle simply cannot publish
            logger.debug("handle does not accept a registrant back-reference", exc_info=True)
        # With the back-reference in place the handle can answer "is anyone
        # watching?", which is what the model needs in its prompt so a
        # detached session does not ask a question nobody can answer.
        installer = getattr(handle, "_install_interactivity_probe", None)
        if callable(installer):
            try:
                installer()
            except Exception:  # noqa: BLE001 — a probe is never worth a runtime
                logger.debug("could not install the interactivity probe", exc_info=True)
        seed = handle.session_projection_seed
        seed.kind = kind
        # The projection fold is an OPTIONAL, injected collaborator. A caller
        # that already owns a fold may hand it in and the runtime uses it
        # as-is — only tests do today; every production constructor call
        # (the TUI at ``tui/app.py``, the owned-session process) passes none.
        # Given none, nothing is built until a client that consumes
        # projection semantics (the mobile daemon) actually dials, so a
        # runtime that only ever serves a follower terminal or fires a wake
        # constructs no fold at all. Welcomes and repaints before that moment
        # serialize the seed directly: the seed IS the object the handle's
        # own fold mutates, so the bytes on the wire are identical either
        # way — and the band's age is re-dated through that same handle fold
        # at every frame build, because the runtime's own sink fold is never
        # fed and must not write state it does not own (review round 4,
        # BLOCKER 1). See ``_ensure_projection_sink`` and
        # ``_projection_payload``.
        self._projection_sink: ProjectionSink | None = projection_sink
        #: How many times this runtime built a fold on its own. Observable
        #: for tests and for the "did a headless runtime pay for a fold?"
        #: question; 0 after a lifetime with no daemon client is the claim.
        self.projection_sinks_built: int = 0
        # What build this runtime is running, stamped once at construction:
        # the answer cannot change while the process lives, and the record is
        # the channel an attach client reads it from before it dials.
        #
        # Imported function-locally although ``update`` is stdlib-only, because
        # server.py sits near the CLI startup path and the house style there
        # (see ``app.py``'s update imports) is to keep it off the import graph.
        from local_operator.update import installed_build, process_install_root

        # ``LOP_BUILD_PREFIX`` is the e2e stage's seam only (see
        # ``process._build_prefix``): the boot stamp and the reaper's re-read
        # must come from the SAME root, or a runtime under test would compare
        # a real install against a fake marker and retire at once.
        build = installed_build(os.environ.get("LOP_BUILD_PREFIX") or None)
        #: What this process LOADED, kept for the runtime's self-refresh check:
        #: ``process._should_refresh`` compares it against the install on disk
        #: on the reaper's tick and retires an idle runtime whose build has
        #: moved under it (design-runtime-autorefresh §3.2). Frozen dataclass,
        #: so the boot snapshot cannot drift toward the disk value it is
        #: compared against.
        self._boot_build = build
        self._record = SessionRecord(
            pid=os.getpid(),
            kind=kind,  # type: ignore[arg-type]
            session_id=seed.session_id,
            conversation_name=seed.conversation_name,
            cwd=seed.cwd,
            model_label=seed.model_label,
            # THE ONE-SHOT "an update applied" FACT, and this is the only writer
            # that can publish it: the marker the outgoing runtime left was
            # consumed at boot (``process._consume_update_marker``) BEFORE this
            # server existed, so it waits on the handle and is seeded onto the
            # record here. ``""`` for every ordinary boot, which is also what a
            # runtime too old to carry the attribute reads as.
            updated=getattr(handle, "applied_update", "") or "",
            control_port=0,  # stamped when the listener binds
            control_key=secrets.token_hex(32),
            # Independent capabilities, each gated by its own condition. The
            # desktop watch surface is unconditional (this server always serves
            # it), while the rest are advertised only when the handle actually
            # implements them -- advertising one the handle cannot honour is
            # worse than omitting it, because the client then negotiates a
            # surface that is not there.
            capabilities=(
                [DESKTOP_WATCH_CAPABILITY]
                # Unconditional, unlike the handle-gated entries below: the
                # mute is a property of the RELAY this server always runs, not
                # of anything the handle implements.
                + [EVENT_MUTE_CAPABILITY]
                + ([FRONTEND_CAPABILITY] if hasattr(handle, "subscribe_frontend") else [])
                + (["completion-ack-v1"] if hasattr(handle, "acknowledge_attention") else [])
                + (["display-history-window-v1"] if hasattr(handle, "history_page") else [])
                # The exclusive-move fence is advertised ONLY by a handle that
                # carries the safe retirement latch, because the fence's promise
                # is a re-check at that latch (``begin_retire``). A reduced
                # handle lacking it would advertise a guarantee it cannot keep,
                # and the desktop gates the move on seeing this exact string —
                # so a partial owner must NOT have it.
                + ([EXCLUSIVE_MOVE_CAPABILITY] if hasattr(handle, "begin_retire") else [])
                # A SECOND string for the same op, because the one above is a
                # bare presence flag with no version handshake and cannot say
                # "this owner also pages pre-compaction history". The page
                # model forbids extra fields, so emitting the audit fields to a
                # viewer built before them fails that viewer's validation and
                # breaks the attach outright. See
                # ``DISPLAY_HISTORY_AUDIT_CAPABILITY``.
                + (["display-history-audit-v1"] if hasattr(handle, "history_page") else [])
                # ADVERTISED UNCONDITIONALLY (revision 2, §2.3). The runtime can
                # always VERIFY an operator or device signature: the anchor is a
                # file it reads, and the public half is all verification needs.
                # Deliberately NOT gated on the anchor being installed, which
                # would make a host mid-onboarding look like a host that cannot
                # accept a signature at all — a client that then declined to ask
                # for a challenge would report "not supported" rather than "not
                # yet installed", and the refusal copy tells the reader to
                # install it. An uninstalled anchor fails the verification
                # (``signature_verdict`` returns False for a signature with no
                # anchor to place it against), which is the honest refusal.
                + [OPERATOR_SIGNATURE_CAPABILITY]
            ),
            # A runtime is born with no terminal watching it. Stamped at
            # construction rather than left to the first transition, because
            # the window before a viewer attaches is exactly when a detached
            # runtime is most interesting to look at.
            detached=True,
            # The heartbeat republishes this same dataclass, so the stamp
            # rides every rewrite without a second code path.
            version=build.version,
            source_ref=build.source_ref,
            # The tree THIS runtime imports from. Under the generation layout a
            # process belongs to exactly one generation and the record is where
            # that is written down; ``lop install prune`` reads it so a tree a
            # live session is still reading is never deleted. Resolved, so a
            # process launched through the pointer records the generation it
            # really got rather than the mutable path it came in through.
            install_root=process_install_root(),
        )
        # A resumed conversation has ALREADY run its turns under an earlier
        # process, and the record must say so from its FIRST publish. Until
        # the owner's first turn here, the bit would otherwise read False and
        # peers would treat a working session as a composer window (QA Q3:
        # after a terminal restart, `lop --resume <sid>` engaged a child whose
        # record said ``started=false``, making the idle session invisible to
        # broadcasts and degrading peer wakes to quiet notes until the owner
        # typed once). Seeded from the session's own durable transcript — the
        # same signal ``TuiSessionHandle.rebind`` uses — so a true ``/new``
        # (no message rows) still boots as the composer window.
        #
        # ``_session`` IS AN ATTRIBUTE ON SOME HANDLES AND A METHOD ON OTHERS,
        # and the difference is load-bearing rather than cosmetic. The owned
        # shapes (``ServingSessionHandle``, the exec handle) assign the Session
        # once, so ``getattr`` returns it; ``TuiSessionHandle._session`` is a
        # METHOD — the phone has to follow a ``/new``/``/resume`` swap, so the
        # TUI reads it per call. Probing only the attribute therefore bound a
        # function, read ``transcript_path`` off it, got ``None`` and reported
        # ``False`` for EVERY TUI window, including the very case this seed
        # exists for: a TUI booted on an existing conversation
        # (``lop --resume <sid>``, whose first session is adopted before any
        # ``rebind``) published ``started=false`` over hundreds of transcript
        # rows, and the peer gate then refused a send to it with a sentence
        # that was false about it (review round 1, F-1). Calling it is
        # GUARDED because a TUI whose session has not been bound yet raises
        # ``RuntimeError("session is still starting")``, and a runtime whose
        # handle cannot answer yet keeps the conservative False that its first
        # real turn corrects — the same direction as a reduced test host with
        # no session at all. Direct field writes, not ``set_record_started``:
        # nothing is published yet, and this is a derivation at birth, not the
        # per-turn signal.
        owned_session = getattr(handle, "_session", None)
        if callable(owned_session):
            try:
                owned_session = owned_session()
            except Exception:  # noqa: BLE001 — a boot that cannot answer yet is not an error
                owned_session = None
        try:
            # The DERIVATION is inside the guard too, not just the call. The
            # reader ends at ``durable_conversation_path``, which catches only
            # ``OSError``: a handle answering with something that is not a path
            # would raise ``TypeError`` straight out of this constructor. A
            # runtime that cannot boot is a far worse failure than one record
            # starting conservatively false until the owner's first turn.
            if owned_session is not None and has_durable_history(owned_session):
                self._started = True
                self._record.started = True
        except Exception:  # noqa: BLE001 — a host that cannot answer keeps the conservative False
            # WARNING, not debug (review round 3, N2): where this fires on a
            # RESUMED session the record publishes ``started=false`` over a
            # conversation that has run turns, which is the false-refusal shape
            # of the F-1 defect — a peer send to it is answered with "no user
            # message has been sent in it". Benign on a reduced host, and a
            # signal worth seeing when it is not.
            logger.warning(
                "could not read the resumed session's history at boot; " "publishing started=false",
                exc_info=True,
            )
        self._publisher: RecordPublisher | None = None
        #: The config dir this runtime was STARTED in, captured by ``start`` /
        #: ``start_in_process`` and handed to the publisher. The record path is
        #: ``<config dir>/run/mobile/<pid>.json``, so resolving it when the
        #: worker thread eventually reaches ``_serve`` lets a thread the OS did
        #: not schedule promptly publish into whatever directory is current by
        #: then — another session's, under a pid-keyed filename. Left as None
        #: until a start path runs, so a server that is never started keeps the
        #: publisher's own default.
        self._config_root: Path | None = None
        self._server: asyncio.AbstractServer | None = None
        self._unsubscribe_events: Callable[[], None] | None = None
        # Strong references to the one event writer per subscribed client. A
        # bare create_task can be collected mid-flight, which drops frames.
        self._event_sends: set[asyncio.Task[None]] = set()
        #: Requests the reader loops have ADMITTED and not yet answered, and the
        #: event that fires when the count returns to zero. ``_shutdown_impl``
        #: waits on it so a reply the runtime has already admitted reaches its
        #: socket — see ``_await_in_flight_requests`` for why a shutdown can
        #: otherwise beat its own ack, and why the counter (rather than one
        #: long-lived Event) is what makes a later request wait on its own
        #: admission rather than on an older one's completion.
        self._in_flight_requests = 0
        self._in_flight_idle: asyncio.Event | None = None
        #: Frontend binds abandoned by a connection that died mid-bind, held
        #: until they land so their release callback can still fire (nothing
        #: awaits them any more — see ``_release_when_landed``).
        self._abandoned_binds: set[asyncio.Task[Any]] = set()
        # N authenticated connections keyed by id(writer): one daemon (a new
        # daemon dial evicts the old — that IS its reconnect story) plus up to
        # ATTACH_MAX_CLIENTS attach clients. A single _writer could not carry
        # the phone bridge and a follower terminal at once.
        self._clients: dict[int, _ClientConn] = {}
        #: The attach connection that reserved an EXCLUSIVE move, or ``None``.
        #: Set on this loop in the same synchronous step that counts the other
        #: observers, so a viewer arriving after the count cannot be missed:
        #: ``_on_connection`` refuses a new attach while the fence is held. It
        #: is cleared on a definite refusal and left set once retirement
        #: commits, because a retiring runtime must not admit a facade that
        #: would then engage a successor from its own stale cwd.
        self._exclusive_move_fence: _ClientConn | None = None
        #: Whether the retirement LATCH has committed for this runtime. Read by
        #: the exclusive-move fence release: ``request_stop`` can raise after
        #: ``begin_retire`` has already committed, and that is precisely the
        #: state the retained fence exists for — the runtime is going away and a
        #: facade admitted now would engage the successor from its own cwd
        #: (review round 2, N6). Monotonic: a runtime never un-latches.
        self._retirement_committed = False
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._unsubscribe: Callable[[], None] | None = None
        self._closed = threading.Event()
        #: Publication latch state, set by ``_serve`` once its boot prologue has
        #: SETTLED — the record is on disk with its bound port, or the prologue
        #: failed and there will never be one. Guarded by ``_publication_lock``
        #: because the reader is the CALLER's thread and the writer is the
        #: runtime's own; see :meth:`wait_until_published` for why a caller needs
        #: either, and :meth:`_settle_publication` for why the waiters are
        #: ``PublicationGate``s rather than executor threads.
        self._publication_settled = False
        self._publication_gates: list[PublicationGate] = []
        self._publication_lock = threading.Lock()
        #: Loop-side wake for ``_closed``. Created ON the runtime loop by
        #: ``_closed_wait``, awaited there and set by ``_request_close`` from
        #: whatever thread latches the close, via ``call_soon_threadsafe`` — an
        #: ``asyncio.Event`` may not be touched from a foreign thread directly.
        #: ``None`` outside thread mode (nothing parks, so nothing to wake) and
        #: until ``_closed_wait`` publishes it; a close that lands first finds
        #: the latch already set and never parks at all.
        self._close_event: asyncio.Event | None = None
        self._push_scheduled = False
        # One warning per contiguous run of oversized frames, not one per
        # frame: a busy session repaints ~30x/s and a per-frame warning is the
        # log flood the cap exists to prevent. Reset when a frame fits again.
        self._frame_cap_warned = False
        #: Frontend binds whose connection went away MID-BIND, held until they
        #: land so their registration can be released — see
        #: ``_release_when_landed``. Nothing else awaits these tasks, and an
        #: unreferenced task can be collected mid-flight, in which case its
        #: release never fires and the session keeps a subscriber for the life of
        #: the process.
        self._abandoned_binds: set[asyncio.Task[Any]] = set()
        # The delayed repaint must be owned like the heartbeat. A bare task can
        # still be sleeping when close tears down the runtime loop, producing
        # an orphan warning and proving teardown returned before its work ended.
        self._push_task: asyncio.Task[None] | None = None
        self._heartbeat_task: asyncio.Task[None] | None = None
        self._attention_task: asyncio.Task[None] | None = None
        # Shutdown is represented by one owner-loop task so synchronous close,
        # awaited close, and the thread runner can converge without cancelling
        # loop-owned objects from whichever thread happened to request teardown.
        self._shutdown_task: asyncio.Task[None] | None = None
        # -- front-end accounting (the child reaper's inputs, §4) --------------
        # Phone SSE watchers, fed by the daemon's watch/unwatch pushes. Floored
        # at 0: a daemon restart redials without unwatching, and a counter that
        # went negative would read as "watchers" to an == 0 check forever.
        #
        # READ THIS BEFORE YOU TOUCH THE COUNTER. It is SERVER-GLOBAL state
        # mutated from THREE per-connection contexts, and that mismatch — not
        # any one line — is what has produced four separate defects in this
        # predicate across four review rounds:
        #
        #   1. removal          `_drop_client` never released a dead daemon's
        #                       count, so a phantom viewer suppressed toasts
        #                       forever (R5).
        #   2. second removal   `_drop_client` is documented to run TWICE on
        #                       one connection; an unconditional release on the
        #                       late call wiped the REPLACEMENT's live count
        #                       (R7).
        #   3. request          a `watch`/`unwatch` buffered behind a parked op
        #                       still arrives after its connection is evicted,
        #                       because closing the writer does not stop the
        #                       `StreamReader` (R8) — and an `attach` client's
        #                       `watch` incremented a count only a `daemon`
        #                       drop could clear (R9).
        #
        # Every instance failed the same way: the count outlives, or is stolen
        # from, the connection it describes. Each is now defended separately,
        # which is why three guards say "is this connection still registered?"
        # in three places.
        #
        # THE DURABLE FIX IS STRUCTURAL, and deliberately not taken here: hold
        # the count on `_ClientConn` and fold over the live registry, exactly
        # as `watching_surfaces` already does for `attach` clients. A dropped
        # connection then removes its own contribution BY CONSTRUCTION — no
        # zeroing, no registry guards, and every one of these spellings becomes
        # unrepresentable rather than separately defended. It was scoped out of
        # this release as too large for a review round; do it before adding a
        # fourth mutation site, not after the fifth defect.
        self.phone_watchers: int = 0
        # Latched True on the first watch/unwatch EVER received. Until then
        # watcher count is UNKNOWN (an old daemon never sends the ops), and
        # unknown must be treated as "present" — a new child under an old
        # daemon must not reap a session a phone is actively watching. The
        # latch never resets: once a watch-capable daemon has spoken, absence
        # of the op means absence of watchers.
        self.watch_supported: bool = False

    # -- lifecycle -----------------------------------------------------------

    def start(self) -> None:
        """Bind the listener, publish the record, start the heartbeat and the
        event feed — on a dedicated thread with its own loop. Idempotent.

        This is the path for EVERY kind whose session lives in this process:
        the TUI (which started here because Textual owns its loop) and, since
        the serving plane was decoupled from the workload, the daemon and exec
        children too (:func:`process.amain`, ``exec_control``). Their callers
        owe a :meth:`wait_until_published` before reading the record.
        """
        if self._thread is not None:
            return
        # PIN THE DISCOVERY DIRECTORY HERE, at the moment the caller asks this
        # runtime to announce itself. ``start`` only hands the work to a thread,
        # so this is the last point at which the answer is still the caller's:
        # a runner loaded enough to delay that thread past ``start``'s return
        # (and past the runtime's own ``close``, whose join is bounded) had
        # ``_serve`` create its publisher against a config dir the process had
        # already moved on from, publishing — and then deleting — a record in
        # a directory this runtime never started in. See
        # ``RecordPublisher.__init__`` for the other half of this invariant.
        self._config_root = config_dir()
        # The thread name is a runtime-observable diagnostic (py-spy, thread
        # dumps, `ps -M`) and deliberately keeps its mobile-era spelling: this
        # move changes no behaviour, and renaming it would silently invalidate
        # anyone's saved grep. Rename it only alongside RUN_DIRNAME, if ever.
        self._thread = threading.Thread(target=self._run, name="lop-mobile-registrant", daemon=True)
        self._thread.start()

    async def start_in_process(self) -> None:
        """The same startup as :meth:`start` but on the CALLER'S running loop.

        For a host whose session already lives on that loop and which is not a
        daemon or exec child: an in-process TUI host, ``lop serve``'s reload
        worker, a test. Those two kinds USED to choose this path, and reversing
        that choice is the whole of this change — the docstring here read "a
        second loop would force every handle call through a cross-thread hop for
        no benefit", which was measured and is wrong: the hop IS the benefit,
        because in process one synchronous step of a turn parks the listener,
        the welcome, ``ping`` and the heartbeat together, so a working session
        reads as ``wedged`` and a fresh dial is never served. See
        :func:`process.amain` for the readings.

        Kept rather than deleted, deliberately: the TUI's and the suite's
        in-process hosts are real, and the two modes share ``_serve`` so this
        is a caller-side choice rather than a second implementation.
        """
        if self._server is not None:
            return
        # The pin is LOAD-BEARING on this path too, not defence in depth:
        # `_serve` reaches its first yield — `await asyncio.start_server` —
        # BEFORE it builds the publisher, so a config dir that moves while this
        # await is suspended is otherwise resolved by `RecordPublisher` inside
        # `_serve`, and the record lands in a directory this runtime never
        # started in. (The caller-visible half of the same rule is
        # `RuntimeServer.record_path`.)
        self._config_root = config_dir()
        self._loop = asyncio.get_running_loop()
        await self._serve()

    async def wait_until_published(self, *, timeout: float = _PUBLISH_WAIT_TIMEOUT_S) -> bool:
        """Wait for :meth:`start`'s record, and answer whether it exists.

        ``start()`` only HANDS the work to a thread. Everything the record
        carries — the listener's port above all — is stamped on that thread, so
        a caller that reads ``record.control_port`` (or ``record_path``) on the
        next line reads the constructor's ``0`` and names a file nothing has
        written. That is not theoretical: ``exec_control.start_exec_control``
        builds the endpoint line a SUPERVISOR is handed from exactly those two
        fields, and the line exists so a supervisor can find this session —
        ``port=0`` plus a record path that may not exist breaks the one thing
        it is for. ``process.amain`` reads the record less eagerly (its parent
        polls ``launch.py``'s wait-for-record loop), but it asks for the same
        ordering and gets it from the same call rather than from a second
        convention.

        Returns True when the record was published, False when the boot
        prologue settled without publishing (a bind failure) or the bound
        expired. The two are ONE answer on purpose: both mean "there is no
        control surface to talk to", and a caller that needs to tell them apart
        has the runtime's own log for it.

        IT SETTLES AT THE END OF THE BOOT PROLOGUE, not at the instant the
        record lands, and that is a deliberately stronger guarantee: by the time
        this returns on the success path the boot registrations have come back
        from the session's loop and the heartbeat has been started. "The record
        exists" is not the same claim — a caller released at publication could
        hand the loop straight to a turn, starving the registration hop that
        ``_serve`` is still awaiting, and the runtime would publish a record and
        then never beat (measured: `wedged` at t=46 s with no client attached,
        i.e. the defect this change removes). ``_serve`` states the ordering
        where it is enforced.

        THE WAIT COSTS NO THREAD, which is a correction rather than a nicety.
        The obvious spelling — ``asyncio.to_thread(settled.wait, timeout)`` on a
        ``threading.Event`` — was measured at fleet depth and left a SECOND
        thread parked in the loop's default executor for the life of the
        process: 12 runtimes went from 1 thread each to 3, not 2, which is
        exactly the per-runtime cost §5.2 of the audit exists to watch. A
        ``PublicationGate`` binds the WAITER's loop and hops the open onto it
        with ``call_soon_threadsafe`` — the same class the deferred MCP wiring
        already uses for this exact cross-thread latch — so the wait is one
        parked coroutine and no thread at all. It is awaited here with
        ``asyncio.wait_for``, so a runtime that never publishes cannot park its
        caller for the life of the process either.
        """
        gate: PublicationGate | None = None
        with self._publication_lock:
            # Registered under the lock, and the settle path CLEARS the list
            # under the same lock, so a gate handed over here is guaranteed to
            # be opened: there is no window in which the prologue settles
            # between this check and this append.
            if not self._publication_settled:
                gate = PublicationGate()
                self._publication_gates.append(gate)
        if gate is not None:
            try:
                await asyncio.wait_for(gate.wait(), timeout=timeout)
            except TimeoutError:
                # A TIMED-OUT WAITER TAKES ITS GATE BACK OUT. The settle path
                # clears the list wholesale, so a gate left behind here is not a
                # leak in the ordinary case — but it IS one on the timeout path
                # this branch exists for: a runtime whose prologue never settles
                # keeps the list it was never cleared from, and a host that
                # starts many runtimes that never publish walks a list of dead
                # gates on every one of them. Under the same lock, and only
                # while the latch is still closed, so a concurrent settle cannot
                # race the removal.
                with self._publication_lock:
                    if not self._publication_settled:
                        with contextlib.suppress(ValueError):
                            self._publication_gates.remove(gate)
                return False
        # ``_publisher`` is written on the runtime's own thread BEFORE the latch
        # settles, so a settled latch is a read of it that has already happened
        # — the latch is the synchronisation, not a hint.
        return self._publisher is not None

    def _settle_publication(self) -> None:
        """Open the publication latch, from the runtime's own loop.

        Called on EVERY way out of ``_serve``'s boot prologue, a failed bind
        included: a caller waiting here must be released by a runtime that will
        never serve as well as by one that is about to, or it waits out its
        whole bound for an answer ``_serve`` already has. Which of the two it
        was is ``_publisher``'s business, not this latch's.

        The gates are taken OUT of the list under the lock before they are
        opened, so a second settle (a re-entered prologue cannot happen today,
        but a latch that only works once is a latch nobody can reuse) cannot
        open them twice, and a late waiter cannot be added to a list no one will
        walk again.
        """
        with self._publication_lock:
            self._publication_settled = True
            gates, self._publication_gates = self._publication_gates, []
        for gate in gates:
            gate.set()

    def announce_stop(self) -> None:
        """Tell every attached viewer this session is ending DELIBERATELY.

        THE single emitter of the ``stopping`` frame, called by both triggers:
        the control-op path (a peer's ``lop stop``, another TUI's
        ``/stop all``) and the owner's own bare ``/stop``, which runs its
        teardown locally and never dispatches an op. Round 3 found that second
        route silent, so a follower watching an owner that stopped itself saw
        a plain EOF and took over the session the user had just ended — U2-4
        surviving on a different path.

        Must be called BEFORE the teardown that closes these sockets. Safe
        twice: a viewer reads the frame only as "the disconnect coming next
        is deliberate", so a duplicate is a no-op.

        There is deliberately no on-disk fallback. A wakeless session has no
        wake-index entry at all (``write_entry`` removes the file when a
        session has no schedules), so the wire is the only channel covering
        every session — which is why the marker approach was dropped.

        Safe from any thread, like :meth:`close`, and best-effort by
        contract: a viewer that never receives it degrades to the pre-round-2
        behaviour, which is strictly better than a stop failing because one
        socket was slow.
        """
        if self._closed.is_set():
            return
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        frame = {"op": "stopping", "session_id": self._record.session_id}
        if self._on_runtime_loop():
            # An in-process runtime shares the TUI's loop, so the owner's own
            # /stop arrives HERE, from a synchronous caller whose very next
            # statement tears the sockets down. Awaiting a drain is therefore
            # not available and scheduling a task is too late — the teardown
            # would win the race. ``write`` is synchronous (it buffers into
            # the transport), and a transport closed afterwards still flushes
            # what it holds, so writing inline is what actually gets the frame
            # to the viewer ahead of the EOF it must explain.
            self._write_now(frame)
            return
        # The THREAD-HOSTED path, which is the one production takes: the TUI
        # hosts its registrant with `.start()`, and the caller is a coroutine
        # on the TUI's own event loop. Awaiting a drain here froze that loop
        # for up to two seconds against a viewer whose receive window was full
        # (round-4 MINOR-3, the #401 class) — on a path whose whole contract
        # is that announcing must never make a stop slower.
        #
        # The same reasoning the inline branch rests on applies once the write
        # is on the right thread: ``write`` only buffers, and a transport
        # closed afterwards still flushes what it holds. So hand the write to
        # the runtime's loop and wait only for it to have BEEN WRITTEN, with a
        # bound far below any user-perceptible pause. Missing that bound costs
        # a viewer its explanation, never the stop.
        written = threading.Event()

        def _write_and_signal() -> None:
            try:
                self._write_now(frame)
            finally:
                written.set()

        try:
            loop.call_soon_threadsafe(_write_and_signal)
        except RuntimeError:
            # Loop already closing: nothing is listening that could care.
            return
        if not written.wait(timeout=_ANNOUNCE_WRITE_TIMEOUT_S):
            logger.debug("stop announcement did not reach viewers before the teardown")

    async def announce_retiring(
        self, reason: str, *, to: str = "", draining: bool = False, leaving: str = ""
    ) -> None:
        """Tell attached viewers this runtime is leaving so a NEWER build can
        take its place — a planned refresh, not a stop and not a death.

        A sibling of :meth:`announce_stop` rather than a reuse of it, and that
        distinction is the whole point: a viewer that reads ``stopping`` parks
        in the stopped state and tells the user ``/resume`` reopens it, which
        is the wrong story for a runtime that retired only because
        ``lop-update`` ran. ``retiring`` says "a fresh runtime is owed; engage
        one" and the viewer does so eagerly (``AttachedSession._go_cold(refresh=
        True)``). Additive on the wire: an old viewer ignores the unknown op,
        sees the EOF, and runs its ordinary recovery (cold after 8 s) — the
        pre-refresh behaviour, so no ``PROTOCOL_VERSION`` bump.

        ``draining`` is the runtime's OWN verdict, and it is the only place
        that verdict can come from: it is True when the departure was
        committed while work was still in flight (:func:`process._begin_drain`
        announces and latches in one step, so everything after this frame is
        refused until the drain empties), False for the idle handover
        (:func:`process._refresh_for` and :meth:`_retire_if_pristine`), which
        leaves in about a second and refuses nothing. A viewer that must say
        something to the operator reads THIS field: asking itself instead asks
        a state that is already cold by the time it can look, which is how a
        notice meant for the drain ended up painted on every idle handover as
        well (QA round 3, Q-1).

        Sent to ATTACH clients only. The phone daemon's projection path stays
        byte-identical, and the daemon already handles owner exit by adopting
        the next record it sees.

        ``leaving`` is that same commit's OTHER rendering, and this method is
        where both are written — which is the reconciliation PR #1108 required.
        #1108 landed the drain on the runtime's side and made this frame's
        ``draining`` flag the authoritative word for the APP; this branch had
        added ``SessionRecord.leaving`` for the FLEET surfaces. Two renderings,
        one fact, so one writer: the flag stays authoritative for "is a drain in
        force", the phrase carries the trigger's own words for the surfaces an
        operator reads, and a caller cannot publish one without the other
        because the record is written here, before the frame goes out, from the
        same call that sends it. ``process._commit_to_leaving`` is the only
        caller that passes both.

        AND THE FRAME CARRIES THE PHRASE TOO, which is the half the phrase
        existed for and did not yet have (design round 3, D6). ``draining`` says
        only THAT a drain is in force; it cannot say WHICH trigger committed it,
        and the app's notice is a SENTENCE about the trigger — it promised a
        newer build, so a runtime terminated mid-turn told the operator it was
        switching to a build that does not exist and is not coming. The phrase
        is the same string the fleet surfaces print (``SessionRecord.leaving``,
        written two lines up), so both renderings of the commit leave this one
        method and a viewer that must speak can quote the trigger instead of
        inferring it. Additive like ``draining`` was — a runtime older than the
        key sends no ``leaving``, and its frame is read off the fields it DOES
        carry (``reason``/``to``: ``stale-build`` for the released build
        handover, ``shutdown-drain`` — ``process._SIGNAL_DRAIN_REASON`` — for the
        signal drain this branch added before it added this key). That reader is
        :func:`types.leaving_phrase_for_frame`, and it exists because the
        simpler rule — "no phrase means the build handover" — was true of every
        RELEASED runtime and false of this branch's own intermediate builds,
        which signal-drained into it (design round 4, D9; agent review round 4,
        MAJOR-1).

        Awaited (unlike ``announce_stop``) because its one caller is the
        reaper on the runtime's own loop, which has time to drain: the exit
        follows this frame, and a viewer that receives it late merely goes
        cold the slow way. ON that loop, not from anywhere: the send path below
        owns the loop's ``send_lock`` and the connections' writers, and
        ``_send_to`` refuses a foreign-loop caller outright rather than parking
        it forever — so this method HOPS, and a caller may now sit on either
        plane.

        THE HOP IS NEW, and it is a repair rather than a convenience. Its
        callers (``process._commit_to_leaving``, ``process._refresh_for``) run
        on the SESSION's loop — the reaper and the signal drain both do — which
        was the owner loop only while ``daemon``/``exec`` served in process.
        Once the serving plane moved to its own thread (``start()``), the send
        below met ``_send_to``'s foreign-loop refusal, and the refusal was
        SWALLOWED: ``_commit_to_leaving`` catches it at DEBUG ("a viewer that
        misses this goes cold the slow way"). Viewers would simply stop
        learning that a runtime was retiring, and the drain would proceed — the
        quietest possible failure, which is why the hop lives here rather than
        in a caller's discipline. ``announce_stop`` already carries this
        contract; the two announces now behave the same from any thread.
        """
        loop = self._loop
        if loop is not None and not loop.is_closed() and not self._on_runtime_loop():
            # Hop, never block: ``run_coroutine_threadsafe`` plus an awaited
            # ``wrap_future``, so the session's loop is not parked for as long
            # as the owner's is busy draining viewers. A caller with no owner
            # loop to hop to (a server that never started, and therefore has no
            # connections to reach) falls through to the body, which is what
            # that case did before this hop existed.
            await asyncio.wrap_future(
                asyncio.run_coroutine_threadsafe(
                    self._announce_retiring_on_loop(
                        reason, to=to, draining=draining, leaving=leaving
                    ),
                    loop,
                )
            )
            return
        await self._announce_retiring_on_loop(reason, to=to, draining=draining, leaving=leaving)

    async def _announce_retiring_on_loop(
        self, reason: str, *, to: str = "", draining: bool = False, leaving: str = ""
    ) -> None:
        """The retiring announcement, on the runtime's own loop.

        Split from :meth:`announce_retiring` — which carries the reasoning — so
        that exactly one place owns the thread question and the body below can
        assume it is where it must be, as ``announce_stop``'s caller does.
        """
        if self._closed.is_set():
            return
        if draining and leaving:
            # RECORD FIRST, FRAME SECOND, and they are one commit rather than
            # two publications of one fact (PR #1108 reconciliation). The frame's
            # ``draining`` flag is what the app paints its notice from at frame
            # receipt; ``SessionRecord.leaving`` is what the fleet surfaces read
            # (`lop sessions`, ``/info``, the catalogue, the stop ladder's
            # refusal). Writing the record here — inside the same call that
            # sends the flag, before the send — is what makes the two impossible
            # to disagree: every caller that announces a drain for the app has
            # already published the same drain for the fleet, and the only way
            # to send the frame is through this method.
            #
            # A caller that passes ``draining=True`` and no phrase has nothing
            # to publish (the fallback paths that never latched do exactly
            # that); a caller that passes a phrase without the flag announced no
            # drain and is ignored on purpose — the flag is the authority for
            # "is a drain in force".
            self.note_leaving(leaving)
        frame: dict[str, Any] = {
            "op": "retiring",
            "session_id": self._record.session_id,
            "reason": reason,
            "from": self._boot_build.label(),
            "to": to,
            "draining": bool(draining),
            # The trigger's own words, for the sentence a viewer paints: see the
            # ``leaving`` paragraph above. ``""`` means THIS FRAME NAMED NO
            # TRIGGER — it is what every runtime older than this key sends, this
            # branch's own pre-D6 builds included, and the viewer answers it by
            # reading the ``reason``/``to`` above and only then falling back to a
            # sentence that is true of any drain (design round 4, D9). It is
            # deliberately not "the build handover": that reading painted a
            # signalled runtime with the build sentence (agent review round 4,
            # MAJOR-1).
            "leaving": leaving,
            # THE UPDATE WINDOW, additive like ``draining`` and ``leaving`` above,
            # and it is the key that makes an IDLE handover speakable at all: that
            # rung sends ``draining=False``, so before this key a viewer had
            # nothing to paint while the one handover that QUEUES messages was in
            # flight. ``""`` for every departure that is not a window, and for
            # every runtime older than this key.
            #
            # READ OFF THE RECORD rather than taken as an argument, and that is a
            # correction rather than a shortcut. A parameter here would be a second
            # copy of a field the server already holds, and — worse — a caller whose
            # ``announce_retiring`` predates the parameter would take a TypeError
            # inside the ``except Exception`` that guards a viewer's writer, so the
            # whole announcement would be swallowed by the failure path meant for
            # something else (measured: ``test_process_refresh``'s fake registrant
            # lost its only frame that way). The window is published on the record
            # BEFORE the announce — that ordering is the window's own contract — so
            # the record is the one place both ends can read it from.
            "updating": self._record.updating,
        }
        viewers = [conn for conn in list(self._clients.values()) if conn.kind == "attach"]
        await asyncio.gather(*(self._send_to(conn, frame) for conn in viewers))

    def _write_now(self, frame: dict[str, Any]) -> None:
        """Buffer one frame to every viewer without awaiting a drain.

        PRECONDITION: must run ON the runtime's event loop. Both callers
        satisfy it — the in-process branch is already there, and the
        thread-hosted branch hands this to the loop with
        ``call_soon_threadsafe`` — and it is what makes skipping
        ``conn.send_lock`` sound: no other coroutine can be mid-write at that
        instant, so a partially-written frame is impossible, and taking the
        lock would require awaiting, which the synchronous caller cannot do.
        Called from any other thread the lock-free write would be unsafe
        (round-4 NIT-2: the guarantee belongs to the call site, not the
        method, and saying so is what stops the next reuse from breaking it).

        This is the ONE write that does not go through :meth:`_send_to`, and it
        may only stay that way because ``stopping`` is constant-size by
        construction — a few hundred bytes regardless of session state.
        ``retiring`` does NOT come through here: it carries free-text fields and
        is sent via :meth:`_send_to`, under the ceiling, like everything else.
        Reusing this for anything that can grow would put an unreadable line on
        the wire with no ceiling in front of it.
        """
        for conn in list(self._clients.values()):
            try:
                conn.writer.write(json.dumps(frame).encode() + b"\n")
            except Exception:  # noqa: BLE001 — announcing is best-effort
                logger.debug("stop announcement write failed", exc_info=True)

    def close(self) -> None:
        """Unpublish and shut down. Safe from any thread, safe twice.

        Thread-hosted runtimes are joined before returning. In-process hosts
        should prefer :meth:`aclose`; when ``close`` is called on their owning
        loop it schedules cleanup instead of deadlocking that loop waiting for
        itself.
        """
        self._request_close()
        loop = self._loop
        if self._thread is not None:
            # The thread runner observes the close latch and performs teardown
            # itself. Joining it is both simpler and race-free when close lands
            # while the thread is still publishing its loop reference.
            if threading.current_thread() is not self._thread:
                self._thread.join(timeout=2.0)
            return
        if loop is None or loop.is_closed():
            return
        if self._on_runtime_loop():
            self._ensure_shutdown_task()
            return
        shutdown = self._shutdown_on_loop()
        try:
            asyncio.run_coroutine_threadsafe(shutdown, loop).result(timeout=2.0)
        except RuntimeError:
            # run_coroutine_threadsafe does not consume the coroutine when the
            # loop wins the close race, so close it to avoid a false leak warning.
            shutdown.close()
            logger.debug("runtime loop exited during shutdown", exc_info=True)
        except TimeoutError:
            logger.debug("runtime shutdown did not finish before timeout", exc_info=True)

    async def aclose(self) -> None:
        """Await complete teardown on the owning loop.

        Repeated calls still join the original cleanup task; merely observing
        the cross-thread closed flag is not proof that loop-owned work ended.
        """
        self._request_close()
        if not self._on_runtime_loop():
            raise RuntimeError("RuntimeServer.aclose() must run on its owning event loop")
        await self._shutdown_on_loop()

    async def aclose_remote(self) -> None:
        """Tear down from a loop that does NOT own this runtime: a thread hop.

        :meth:`aclose` is deliberately owner-loop-only — it raises off that
        loop rather than parking a foreign caller forever — but the daemon and
        exec runtimes now publish and serve from their OWN thread
        (:meth:`start`), so their teardown callers are the SESSION's loop and
        the raise is a regression rather than a guard. It was already latent at
        all three call sites; one of them (``process._clean_exit``'s, which sits
        on the path that withdraws the boot record) does not wrap it, so the
        raise aborted the exit BEFORE the record was withdrawn — leaving a
        record for the reaper to read as a runtime that stopped without running
        its own exit ordering, which is exactly what that record is scanned as.

        :meth:`close` already carries every property this needs: synchronous,
        safe from any thread, safe twice, and bounded (a 2 s join of the
        runtime thread, and a bounded coroutine wait on the in-process path).
        Awaiting it off-loop is the whole of the difference.
        """
        await asyncio.to_thread(self.close)

    def _request_close(self) -> None:
        """Latch closure and detach the host feed exactly once."""
        if self._closed.is_set():
            return
        self._closed.set()
        self._wake_close_wait()
        if self._unsubscribe is not None:
            try:
                self._unsubscribe()
            except Exception:  # noqa: BLE001 — shutdown must not raise
                logger.debug("runtime unsubscribe failed", exc_info=True)
            self._unsubscribe = None
        if self._unsubscribe_events is not None:
            try:
                self._unsubscribe_events()
            except Exception:  # noqa: BLE001 — shutdown must not raise
                logger.debug("runtime event unsubscribe failed", exc_info=True)
            self._unsubscribe_events = None

    def _wake_close_wait(self) -> None:
        """Wake the thread-mode serve loop parked on its close event.

        Safe from any thread, like every other caller of ``_request_close``:
        ``call_soon_threadsafe`` is the only thread-safe way to touch another
        loop's objects, and on the loop's own thread it merely schedules for the
        next iteration. Both guards are load-bearing rather than defensive —
        ``_close_event`` is None outside thread mode and until ``_closed_wait``
        publishes it, and a loop that has already stopped raises ``RuntimeError``
        from ``call_soon_threadsafe``. The 2.0 s join in ``close()`` covers both,
        so this must never raise into a close.
        """
        event = self._close_event
        loop = self._loop
        if event is None or loop is None or loop.is_closed():
            return
        with contextlib.suppress(RuntimeError):
            loop.call_soon_threadsafe(event.set)

    def _on_runtime_loop(self) -> bool:
        return self._loop is not None and _running_loop() is self._loop

    async def _handle_call_on_session_loop(
        self, call: Callable[..., Any], /, *args: Any, **kwargs: Any
    ) -> Any:
        """Run one SYNCHRONOUS handle call on the loop that owns the session.

        The ``SessionHandle`` protocol has always specified this shape — *"every
        method is awaited on the RUNTIME'S loop … the implementor guarantees any
        hop the session needs"* — and ``ServingSessionHandle`` now keeps that
        promise for its own asynchronous surface with ``@_on_session_loop``. A
        ``def`` cannot: there is nowhere in a synchronous method to await, and
        blocking on the other loop is exactly what this change forbids. So the
        few ``def``s whose work belongs to the session's loop are hopped HERE,
        by their caller, and this helper is the whole of that path:

        * ``subscribe`` / ``subscribe_events`` — the two boot registrations,
          whose bodies fold history and seed the projection's clocks and state;
        * ``is_pristine`` / ``may_refresh`` / ``begin_retire`` / ``request_stop``
          — the retire and kill-switch probes, where ``begin_retire`` in
          particular is a latch whose value is that it commits in the same
          synchronous step that checks, so it is hopped as ONE call rather than
          sampled and then committed;
        * ``has_admitted_command`` — the dedupe probe, which reads the
          transcript;
        * ``register_secret_redaction`` / ``cancel_subagents_count`` — the two
          ``_dispatch`` arms that write session state. Added in review: the
          first writes the redaction set the bash and eval redactors read, and
          the second runs ``Session.cancel_subagents``, whose task creation and
          loop-bound ``AsyncJobManager`` abort belong to the session's loop — on
          the runtime's thread the cancel's execution ran on the wrong loop and
          the manager swallowed its own cross-loop ``RuntimeError`` at WARNING
          while the op still acked success. The completion the ``await task``
          tracks is the manager's own, so what is lost is the caller's — the
          count the user reads is not evidence the cancel ran where the job
          lives.

        ``reannounce_pending`` is the one named method that does NOT come
        through here, deliberately: one of its four in-tree callers
        (``_drop_client``) is synchronous, so it cannot await a hop, and it is a
        read-then-NOTIFY whose notify path is the registrant's own
        thread-safe-by-design callback surface (the same shape the TUI kind has
        always used). See ``ServingSessionHandle.reannounce_pending``.

        Hop, never block: ``run_coroutine_threadsafe`` plus ``await
        asyncio.wrap_future`` — never ``.result()``, which would freeze the
        runtime's loop for as long as the session's is busy and re-create the
        coupling this whole change removes. An awaitable RESULT is awaited on
        the session's loop too, so a caller may pass an ``async def`` handle
        method (the TUI's ``request_stop`` is one) and receive its value.

        A no-op in the two cases where there is nothing to hop: the caller is
        already on the session's loop (an in-process host, where the note above
        stays true), or the handle publishes no loop at all. That second case is
        what keeps this additive — a handle that owns its own hopping (the TUI's,
        which uses Textual's ``call_from_thread``) declares no ``session_loop``
        and keeps the behaviour it shipped with, and a reduced test host is
        driven inline exactly as before.
        """
        loop = getattr(self._handle, "session_loop", None)
        if loop is None or loop.is_closed() or loop is _running_loop():
            # THE CLOSED-LOOP CASE RUNS INLINE, deliberately, and it is what
            # makes this helper and ``ServingSessionHandle._on_session_loop``
            # agree: ``run_coroutine_threadsafe`` raises ``Event loop is closed``
            # from the caller's side, which is a bare crash where the handle's
            # own refusal is what the dispatcher knows how to render. So a dead
            # loop takes the same path a handle without a loop takes — run it
            # here and let the handle refuse it.
            #
            # WHICH MEANS EVERY BODY REACHABLE THROUGH HERE MUST REFUSE, and that
            # is why the two ``def``s this helper hops for the ``_dispatch`` arms
            # call ``_check_loop_thread`` in their own bodies (review round 2,
            # MINOR-1 / QA Q4): a synchronous method cannot carry the decorator,
            # so without that call the closed-loop case executed the write on the
            # runtime's thread and acked success — the plane round 1 took these
            # two off. ``_resolve_pending`` carries one for the same reason (UX
            # U8): it is the body on the decorator half that touches the loop
            # itself, and its ``call_soon_threadsafe`` was raising the asyncio
            # internal in the middle of a gate tap.
            return await _maybe_await(call(*args, **kwargs))

        async def _invoke() -> Any:
            return await _maybe_await(call(*args, **kwargs))

        return await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(_invoke(), loop))

    def _open_mcp_wiring_gate(self) -> None:
        """Tell the session's deferred MCP wiring that the record is published.

        The latch lives on the HANDLE (``ServingSessionHandle.
        mcp_publication_gate``), read the same way this class reads every other
        optional handle capability — ``subscribe_events``, ``refresh_attention``,
        ``is_busy`` — so a handle that has none is inert rather than an error.
        That covers every registrant constructed for a viewer or a test, and it
        is why the gate is a handle attribute rather than a RuntimeServer
        parameter: the runtime did not create the session and has no other
        business naming its wiring.

        WHAT THE LATCH IS, AND WHAT IT GUARANTEES: it is a
        :class:`~local_operator.session.runtime.publication.PublicationGate`,
        which binds the loop its waiting task runs on and hops with
        ``call_soon_threadsafe`` when it is opened from anywhere else. So this
        method is correct from the runtime's own thread (thread mode, via
        ``start()``) as well as from the session's — the cross-thread case is
        the latch's business rather than a caller's, which is the point of
        using that class instead of a bare ``asyncio.Event``.

        WHY IT EXISTS: the deferred wiring task's first instruction is a
        synchronous import of the MCP SDK. A task starts at the loop's next free
        instant, which in ``process.amain`` is the inbox drain BEFORE this
        publisher runs, so on a machine with a server declared the import took
        the loop for its full duration inside the pre-publication window and the
        record waited behind it (measured +2.3 s, 14 of 14 runs). Setting the
        latch here moves the wiring to the far side of publication, which is
        what ``serving.spawn_owned_session`` states the design already promised.
        """
        gate = getattr(self._handle, "mcp_publication_gate", None)
        if gate is not None:
            gate.set()

    # -- the runtime's own loop -----------------------------------------------

    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        try:
            loop.run_until_complete(self._serve())
        except Exception:  # noqa: BLE001 — a dead runtime must not kill the host
            logger.warning("session runtime loop died", exc_info=True)
        finally:
            loop.close()

    async def _serve(self) -> None:
        # HOW STRONG THE CAPABILITY'S BOUNDARY IS ON THIS HOST, reported rather
        # than assumed, and reported HERE so every host says it exactly once (the
        # helper latches): on Linux with ``ptrace_scope=0`` and on Windows a
        # same-uid process can read this one's memory, so there the capability
        # raises the cost of the attack instead of closing it. See
        # ``harness/approval.operator_cap_guarantee`` and the residual section of
        # ``docs/design/approval-authority.md``.
        report_operator_authority()
        try:
            # Port 0: the OS picks; the record carries the number. Binding
            # loopback only is the security invariant of the whole design.
            self._server = await asyncio.start_server(
                self._on_connection, host="127.0.0.1", port=0, limit=_MAX_LINE_BYTES
            )
            port = self._server.sockets[0].getsockname()[1]
            self._record.control_port = port
            self._publisher = RecordPublisher(self._record, self._config_root)
            # BOTH BOOT REGISTRATIONS ARE PERFORMED ON THE SESSION'S LOOP. The
            # append itself is thread-tolerant, but the body of the same call is
            # not: ``subscribe`` folds the session's history, reconciles the
            # streaming flag and the clocks, and seeds the projection's state
            # (``ServingSessionHandle.subscribe``), all of which read and write
            # session state that belongs to the loop below. Under ``start()`` the
            # two planes have different loops, so each registration is hopped and
            # awaited here — one ``run_coroutine_threadsafe`` each, before the
            # first beat is written, which is also what makes the projection
            # correct from the first push rather than corrected by the first
            # event.
            #
            # INSIDE THE GUARDED PROLOGUE ON PURPOSE, and that is load-bearing
            # rather than tidy: the latch below is what a caller's
            # ``wait_until_published`` waits on, and the hop here WAITS ON THE
            # SESSION'S LOOP. Releasing the latch before this point would hand a
            # caller a runtime whose registrations are still in flight, so a
            # turn that blocks the session's loop immediately after boot —
            # exactly what the regression test does — could starve this hop
            # forever, and the runtime would then never start its heartbeat: the
            # serving plane would be up, the record published, and the beat
            # stale. Measured: that is a `wedged` reading at t=46 s with no
            # client attached, i.e. the defect this change exists to remove.
            self._unsubscribe = await self._handle_call_on_session_loop(
                self._handle.subscribe, self._schedule_push
            )
            # v4: hosts that can serialize their event stream feed the relay.
            # Probed, not required — a handle without the capability leaves
            # attach clients on v3 projection-only behaviour, never broken.
            subscribe_events = getattr(self._handle, "subscribe_events", None)
            if callable(subscribe_events):
                try:
                    subscribe = cast(
                        Callable[[Callable[[dict[str, Any]], None]], Callable[[], None]],
                        subscribe_events,
                    )
                    self._unsubscribe_events = await self._handle_call_on_session_loop(
                        subscribe, self._relay_event
                    )
                except Exception:  # noqa: BLE001 — relay is additive, never a gate
                    logger.debug("event relay subscribe failed", exc_info=True)
            heartbeat = asyncio.ensure_future(self._heartbeat_loop())
            self._heartbeat_task = heartbeat
            if hasattr(self._handle, "refresh_attention"):
                self._attention_task = asyncio.create_task(self._attention_loop())
        finally:
            # RELEASE THE DEFERRED MCP WIRING ON EVERY WAY OUT OF THIS PROLOGUE,
            # not only the happy one. The record is written inside
            # ``RecordPublisher.__init__``, so on the success path this is
            # genuinely post-publication; a bind that raises reaches here too,
            # and that case matters because ``_run`` (thread mode) swallows the
            # exception and the process lives on — with no record and, without
            # this, a latch shut for the session's life. MCP late beats MCP
            # never, and the release cannot mask the failure: the exception
            # still propagates.
            # See ``RuntimeServer._open_mcp_wiring_gate``.
            self._open_mcp_wiring_gate()
            # THE SERVING LATCH OPENS HERE, on every way out and in the same
            # place. On the success path the plane is genuinely serving by now —
            # the record is published, the registrations have returned from the
            # session's loop and the heartbeat has been started — which is the
            # stronger and more useful guarantee a ``start()`` caller needs; on
            # the failure path it means the callers stop waiting for a runtime
            # that is never coming up, and ``_publisher`` still tells the two
            # apart. See ``RuntimeServer._settle_publication``.
            self._settle_publication()
        if self._thread is not None:
            # Thread mode owns the loop: park here until closed. In-process
            # mode returns so the caller's loop keeps running its own work —
            # the caller then owns cancelling the heartbeat (close() does).
            await self._closed_wait()
            await self._shutdown_on_loop()

    async def _closed_wait(self) -> None:
        """Park until the close latch flips, woken by :meth:`_request_close`.

        This used to re-check the latch on a 200 ms ``sleep``, so ``close()`` —
        which joins this thread — inherited the remainder of whatever interval
        it landed in as pure latency, on every close. The event is created HERE,
        on the loop that awaits it, and published to the cross-thread writer
        immediately before parking; a close that beat the publication has
        already set ``_closed``, so the loop condition below is false and it
        never parks. The timeout is the backstop, not the read path; see
        ``_CLOSE_WAIT_BACKSTOP_S``.
        """
        event = asyncio.Event()
        self._close_event = event
        while not self._closed.is_set():
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(event.wait(), timeout=_CLOSE_WAIT_BACKSTOP_S)

    def _ensure_shutdown_task(self) -> asyncio.Task[None]:
        """Create the one teardown task; called only on the runtime loop."""
        task = self._shutdown_task
        if task is None:
            task = asyncio.create_task(self._shutdown_impl())
            self._shutdown_task = task
        return task

    async def _shutdown_on_loop(self) -> None:
        """Join idempotent teardown from a coroutine on the runtime loop."""
        await asyncio.shield(self._ensure_shutdown_task())

    def _admit_request(self) -> None:
        """Count one request as in flight. Called on the runtime's loop."""
        self._in_flight_requests += 1
        if self._in_flight_requests == 1:
            # A FRESH event per span, so a request admitted after an earlier one
            # finished waits on its OWN admission: an event that was left set
            # would make ``_await_in_flight_requests`` return immediately and
            # silently drop the guarantee it exists for.
            self._in_flight_idle = asyncio.Event()

    def _release_request(self) -> None:
        """Release one admitted request and wake the shutdown fence at zero."""
        self._in_flight_requests -= 1
        if not self._in_flight_requests and self._in_flight_idle is not None:
            self._in_flight_idle.set()

    async def _await_in_flight_requests(self) -> None:
        """Let admitted requests answer before the sockets they answer on close.

        WHY THIS IS A FENCE AND NOT A COURTESY. The ``stop`` op is answered by a
        host hook that tears its own runtime down: ``TuiSessionHandle.request_stop``
        schedules the app's teardown, the teardown closes the registrant, and
        :meth:`_shutdown_impl` drops every client — including the one waiting
        for the ack of the op that is doing the stopping. Losing that race turns
        a successful graceful stop into a client-side ``OwnerAckTimeout``, and
        the ladder then escalates to the signal rung on a session that had
        already ended politely. Measured on 2026-09-19 with a trace of
        ``_send_to``/``_shutdown_impl``/``_drop_client``: the shutdown ran at
        +1.035 s and dropped the connection with reason ``runtime shutdown``
        BEFORE the ack was written, and the client read
        ``ConnectionError('runtime closed the connection')``.

        THE RACE IS NOT NEW — the guarantee was. Until ``mobile/tui_handle``
        stopped blocking its own loop on the hop into Textual, the serving loop
        was held by that hop for the whole dispatch, so the teardown (which
        needs this loop to progress) could not overtake the ack. That was
        accidental, and removing it was the point of the change; this restores
        the guarantee deliberately, as the thing the runtime actually owes its
        client: **a request it has admitted is answered before the socket
        closes.**

        BOUNDED, because what this waits for may be another thread's event loop
        (a hop into Textual): a genuinely wedged app must not turn ``close``
        into a hang. After the grace the shutdown proceeds and drops the
        connection — the pre-existing behaviour — with a warning naming how many
        requests were still open.

        THE GRACE IS BOUNDED AGAINST THE CLIENT, NOT AGAINST THE CALLER, and
        that distinction is the whole reason the two numbers differ. Every
        client speaking to this socket gives a reply 15 s
        (``attach_client.ACK_TIMEOUT_S``), so five seconds is what gets an ack
        out while staying well inside its patience. A thread-hosted runtime's
        own caller waits less — ``aclose_remote`` awaits ``close``, whose join
        of the runtime thread is 2 s (``daemon=True``) — so THIS WAIT CAN
        OUTLIVE THE CALLER THAT ASKED FOR THE SHUTDOWN, and that is safe rather
        than an oversight: ``close`` already documents returning with the thread
        still finishing its own teardown, the thread owns this fence, and what
        the ack needs in order to land is the LOOP still running — not the
        caller still waiting. Matching the grace to the join instead would trade
        a live client's reply for the caller's tidiness. Measured in composition
        (``aclose_remote`` → ``close``, a real thread-hosted runtime with one
        admitted ``stop`` parked in its hook): a 1.5 s park — inside the grace —
        has the caller return at 1.51 s with the ack already written and nothing
        left in flight; a 6.0 s park — beyond both bounds — has the caller return
        at its 2.00 s join while the fence runs on to its grace, warns, and then
        proceeds to drop the connection. In that second case the parked op is
        left to unwind on its own, exactly as it was before this fence existed
        (``_dispatch_frame`` documents why op tasks are never cancelled), so the
        composition costs nothing that was not already spent.
        """
        idle = self._in_flight_idle
        if idle is None or not self._in_flight_requests:
            return
        try:
            await asyncio.wait_for(idle.wait(), timeout=_SHUTDOWN_REPLY_GRACE_S)
        except (asyncio.TimeoutError, TimeoutError):
            logger.warning(
                "session runtime: closing with %d request(s) still in flight; "
                "a reply may not reach its client",
                self._in_flight_requests,
            )

    async def _shutdown_impl(self) -> None:
        """Cancel and join every object owned by the runtime event loop."""
        # FIRST, and before anything else here touches a connection: a reply the
        # runtime has already admitted goes out. See
        # :meth:`_await_in_flight_requests` for the measured failure this
        # prevents (a ``stop`` whose ack lost a race with its own teardown) and
        # for why the wait is bounded.
        await self._await_in_flight_requests()
        if self._heartbeat_task is not None:
            self._heartbeat_task.cancel()
        if self._attention_task is not None:
            self._attention_task.cancel()
        if self._push_task is not None:
            self._push_task.cancel()
        for task in list(self._event_sends):
            task.cancel()
        self._event_sends.clear()
        if self._server is not None:
            self._server.close()
        clients = list(self._clients.values())
        for conn in clients:
            self._drop_client(conn, reason="runtime shutdown")
        await self._await_push_shutdown()
        heartbeat = self._heartbeat_task
        if heartbeat is not None:
            await asyncio.gather(heartbeat, return_exceptions=True)
            self._heartbeat_task = None
        if self._attention_task is not None:
            await asyncio.gather(self._attention_task, return_exceptions=True)
            self._attention_task = None
        if self._server is not None:
            await self._server.wait_closed()
            self._server = None
        if clients:
            await asyncio.gather(
                *(conn.writer.wait_closed() for conn in clients), return_exceptions=True
            )
        # A SHIELDED BIND OUTLIVES ITS CONNECTION, deliberately: the shield is what
        # stops a cancel from aborting a registration halfway, so the bind is left
        # to land and its done-callback releases what it registered
        # (``_release_when_landed``). Give those a bounded chance to land BEFORE
        # the loop goes away — a task still pending at loop close is destroyed
        # with a warning, and its release never runs.
        #
        # THE RESIDUAL IS DISCLOSED RATHER THAN ARGUED AWAY: a bind parked on a
        # session loop that is itself stuck cannot be made to land, and it must
        # NOT be cancelled — cancelling it is the half-registered leak the shield
        # exists to prevent (#1327 round 1, F3), since a cancelled bind's
        # done-callback cannot tell whether a subscription got registered. So a
        # shutdown that races a STUCK session loop can still destroy one bind
        # pending and leave that one subscriber to the process's end. Bounded
        # here so the ordinary case — a bind in flight, the session's loop
        # healthy — always drains.
        await self._drain_abandoned_binds()
        if self._publisher is not None:
            self._publisher.close()
            self._publisher = None

    async def _drain_abandoned_binds(self, timeout: float = _ABANDONED_BIND_DRAIN_S) -> None:
        """Let shielded binds land before the runtime's loop can disappear.

        WAITERS, NOT CANCELLERS, and that is the whole design: see the call site
        (and ``_release_when_landed``) for why a cancelled bind can leak the
        subscription it registered. Bounded, because this runs during shutdown
        and the loop a bind is waiting on may never answer.
        """
        pending = [task for task in self._abandoned_binds if not task.done()]
        if not pending:
            return
        _done, still = await asyncio.wait(pending, timeout=timeout)
        if still:
            logger.debug(
                "session runtime: %d frontend bind(s) still in flight at shutdown; each "
                "releases its subscription when it lands, or is dropped with the process",
                len(still),
            )

    async def _await_push_shutdown(self) -> None:
        """Join the coalesced repaint before its owning loop can disappear."""
        task = self._push_task
        if task is None:
            return
        await asyncio.gather(task, return_exceptions=True)
        if self._push_task is task:
            self._push_task = None
        self._push_scheduled = False

    async def _attention_loop(self) -> None:
        """Reconcile read receipts without making a liveness heartbeat a read."""
        previous: Any = None
        while not self._closed.is_set():
            await asyncio.sleep(1.0)
            if not self._clients:
                continue
            try:
                state = await cast(Any, self._handle).refresh_attention()
                if state != previous:
                    previous = state
                    self._schedule_push()
            except Exception:
                logger.debug("runtime receipt reconciliation deferred", exc_info=True)

    async def _heartbeat_loop(self) -> None:
        # THE TWO MEASURED FACTS THIS BEAT PUBLISHES, and the clock they are
        # measured against: ``previous`` is (wall, this process's own CPU time)
        # as of the last beat, so the pair below separates a runtime that burned
        # its core from one the host descheduled. Both are read in-process — no
        # ``ps``/``lsof`` fork per tick, which is the lesson
        # ``control.py`` records at 201 forks per probe.
        previous = (time.monotonic(), time.process_time())
        while not self._closed.is_set():
            # THE WHOLE CYCLE IS GUARDED, INCLUDING THE STAMP, and this is the
            # previous change's own defect one plane over: ``_watch_stall_beats``
            # exists because a reporter that dies takes the plane's evidence with it
            # (the bound fires one deadline later on a healthy runtime, and a
            # ``faulthandler`` dump cannot show it — a dead TASK has no thread and no
            # frame). The WORKLOAD tick got that supervision in #1419; this tick, the
            # SERVING plane's only sign of life, was started as a bare
            # ``ensure_future`` and every statement before the record write below was
            # unguarded — so a raise from ``beat`` ended the serving plane's reporter
            # for the life of the process, with nothing observed, nothing logged and
            # nothing recorded. The dump would then show what an idle healthy process
            # shows, which is exactly the reading the census cannot settle.
            #
            # ``CancelledError`` IS NOT CAUGHT and needs no arm of its own: it is a
            # ``BaseException`` since 3.8, so a shutdown (``close`` cancels this task)
            # still propagates while everything else is logged, RECORDED through
            # ``note_tick_death`` — the instrument that names a dead reporter in the
            # dump — and left running. The tick is NOT re-stamped on the failure path,
            # deliberately: a stamp would claim this plane reported when it did not,
            # and the honest fail-safe is the one #1419 states — a plane whose
            # reporter is truly gone goes unreported and the bound fires on its
            # deadline, with the reason in the dump rather than one leg quietly
            # switched off.
            try:
                await asyncio.sleep(HEARTBEAT_INTERVAL_S)
                if self._closed.is_set():
                    return
                previous_wall, previous_cpu = previous
                moment = (time.monotonic(), time.process_time())
                lag_s, cpu_since_beat_s = moment[0] - previous_wall, moment[1] - previous_cpu
                previous = moment
                # PROGRESS, REPORTED TO THE STALL BOUND. This loop is the serving
                # plane's own sign of life, and it carries ONE stamp — the workload's
                # is its own (``process._beat_stall_watchdog``). The timer is
                # re-armed for the earliest of the two deadlines, so a tick here
                # cannot mask a parked workload loop; that is the whole point of
                # tracking them apart (see ``stall_watchdog``). Deliberately before
                # the record write below rather than after it: the bound must be
                # restarted by THIS LOOP HAVING RUN, not by the write having
                # succeeded — a failed write is self-healing and must not look like a
                # stall.
                stall_watchdog.beat(stall_watchdog.SERVING)
            except Exception:  # noqa: BLE001 — the guard's job is to keep the plane reported
                logger.warning(
                    "runtime heartbeat: the SERVING plane's tick failed; the loop keeps "
                    "running so the plane is still reported, and the death is recorded in "
                    "the stall dump",
                    exc_info=True,
                )
                # THROUGH THE TOTAL WRITE (``process._record_tick_death``'s sibling
                # contract): a raise from here would end the very supervision this arm
                # exists to keep, and this write reaches the dump's own path lookup.
                try:
                    stall_watchdog.note_tick_death(
                        stall_watchdog.SERVING,
                        "the serving plane's heartbeat raised; the loop continued, so the "
                        "plane is reported but this tick stamped nothing",
                    )
                except Exception:  # noqa: BLE001 — a diagnostic never ends a reporter
                    logger.debug("could not record the serving tick's death", exc_info=True)
                continue
            try:
                # The FLOOR for the record's ``busy`` bit, not its fix: the
                # handle republishes at every turn boundary (the session's
                # ``on_turn_settled`` hook), and this only bounds how stale a
                # missed publish can get to one heartbeat. A 15 s-stale bit is
                # still wrong for the picker; it is right for a `lop sessions`
                # run hours later, which is the case the hook's absence cost.
                #
                # Reads the same predicate the handle publishes
                # (``is_conversationally_active``) rather than ``is_busy``:
                # this loop runs every 15 s forever, so a floor that disagreed
                # with the turn-boundary publisher would not merely be stale,
                # it would OVERWRITE the correct value within one heartbeat and
                # re-pin the spinner on every idle session holding a background
                # job. Falls back to ``is_busy`` only for a handle predating the
                # split, where the older answer is the only one available.
                probe = getattr(self._handle, "is_conversationally_active", None)
                if not callable(probe):
                    probe = getattr(self._handle, "is_busy", None)
                if callable(probe):
                    self.set_busy(bool(probe()))
                # The FLOOR for the subagent counts, for the same reason and
                # with the same hazard as ``busy`` directly above: this loop
                # runs forever, so it MUST read the same predicate the
                # transition publisher does (``subagent_counts`` on the
                # handle, which ``_publish_busy`` also calls) or it would
                # overwrite a correct value within one heartbeat instead of
                # merely bounding staleness. A handle that does not implement
                # the probe leaves the counts at ``None`` — unreported, which
                # is the honest answer and not a zero.
                counts = getattr(self._handle, "subagent_counts", None)
                if callable(counts):
                    # Shape-checked rather than unpacked blind: the probe is
                    # duck-typed off the handle, so a handle that answers with
                    # something else must leave the counts unreported instead
                    # of raising inside the loop that also refreshes the
                    # heartbeat — a heartbeat that stops is a runtime the whole
                    # fleet reads as wedged.
                    reported = counts()
                    if isinstance(reported, tuple) and len(reported) == 2:
                        self.set_subagents(reported[0], reported[1])
                # A dead renderer can leave its main-process socket alive.
                # Expiring the lease must reroute a parked gate even when no
                # TCP disconnect arrives to trigger the ordinary detach path.
                self._republish_detached()
                seed = self._handle.session_projection_seed
                if self._publisher is not None:
                    self._publisher.heartbeat(
                        session_id=seed.session_id,
                        conversation_name=seed.conversation_name,
                        model_label=seed.model_label,
                        cwd=seed.cwd,
                        # Carried explicitly rather than trusted to survive on
                        # the record object: ``self._record is
                        # publisher.record`` today, so omitting it happens to
                        # work — but that identity is an implementation
                        # detail, and one rebuild or copied publish away from
                        # silently dropping the bit and making a working
                        # session broadcast-invisible.
                        started=self._started,
                        beat_lag_s=lag_s,
                        cpu_since_beat_s=cpu_since_beat_s,
                    )
            except Exception:  # noqa: BLE001 — a missed heartbeat is self-healing
                logger.debug("runtime heartbeat failed", exc_info=True)

    # -- connections -----------------------------------------------------------

    def _frontend_relay(self, conn: _ClientConn, token: int) -> Callable[[Any], None]:
        """A canonical-delta relay STAMPED with the bind generation it belongs to.

        One factory for both bind attempts (on-loop and off-loop) so the two
        cannot drift in what they put on the wire — the frame is built here, in
        the only place that builds it.

        The stamp is the whole mechanism behind ``_ClientConn.bind_token``: an
        attach that outlives the grace is served off-loop, but the shielded
        on-loop bind it abandoned still lands in the store and still calls this
        callback. Its token is stale by then, so it returns early — without the
        check the SAME connection would receive every delta twice, and a client
        reading an exact-``+1`` stream treats a duplicate as a gap and redials.
        """

        def on_update(update: Any) -> None:
            if conn.bind_token != token:
                return
            payload = update.model_dump(mode="json") if hasattr(update, "model_dump") else update
            self._relay_frontend_to(conn, payload)

        return on_update

    async def _bind_with_grace(self, bind_task: asyncio.Task[Any]) -> Any | None:
        """The on-loop bind's subscription if it answers inside the grace, else ``None``.

        ``None`` means HAND OFF, never "failed": the caller binds off-loop
        instead (``_serve_frontend_sync``), which is the whole point of the grace.
        A genuine bind failure is NOT swallowed here — it is re-raised out of
        ``bind_task.result()`` so the caller drops the connection exactly as it
        always did.

        A METHOD RATHER THAN AN INLINE ``wait_for`` because of the second exit.
        ``wait_for`` expiring cancels the SHIELD it created, not the task it
        shielded, so when the timeout's own callback runs before the shield's
        completion callback the awaiting task is told the grace expired while
        ``bind_task`` is ALREADY DONE and holding a live subscription. Handing off
        there would bind this connection twice and then release the landed one,
        and it would mark a HEALTHY owner window-less — the display window being
        the one thing the on-loop path carries that the off-loop one cannot.
        """
        try:
            return await asyncio.wait_for(asyncio.shield(bind_task), _ONLOOP_BIND_GRACE_S)
        except TimeoutError:
            if bind_task.done() and not bind_task.cancelled():
                return bind_task.result()
            return None

    async def _serve_frontend_sync(
        self,
        conn: _ClientConn,
        frame: dict[str, Any],
        subscribe_frontend: Callable[..., Any],
    ) -> None:
        """Bind one viewer and queue its canonical ``frontend_sync`` frame.

        A per-connection task rather than inline work in ``_on_connection``; the
        reason it is deferred at all is at its creation site. It owns three
        invariants:

        * **The frame order is the frame order it always was.** Registration and
          snapshot capture still happen atomically inside the handle call, the
          sync frame is queued *before* ``frontend_ready`` opens, and every
          update that lands while this task is in flight waits in
          ``conn.frontend_pending`` — so a repaint cannot overtake the state it is
          a delta against. The task changes WHEN the bind runs, never the
          sequence of frames it produces.
        * **A failed bind does not leave a half-open connection.** Inline, a raise
          here escaped ``_on_connection`` — there is no caller to catch it, this is
          a ``client_connected_cb`` — so asyncio logged it as an unhandled
          exception in the callback while the connection stayed registered, never
          read and never dropped. As a task it has an owner: the failure drops the
          connection and releases its subscription.
        * **A busy owner does not hold the bind.** The on-loop bind is given
          ``_ONLOOP_BIND_GRACE_S``; past it this task binds OFF-LOOP through
          ``subscribe_frontend_nowait`` and queues the sync without a display
          window. What that buys is the whole point of the path — measured 15.0 s
          of control attach against a blocked owner, for a bind whose register
          half never needed that loop at all.

        ``subscribe_frontend`` arrives as a PARAMETER, resolved and
        capability-checked at the creation site: dropping a connection whose
        handle cannot bind belongs on the connection path, where the socket still
        exists to be closed, not inside a task. The off-loop capability is read
        here instead, and its ABSENCE is not a refusal: a single-plane handle (the
        TUI kind) simply keeps today's behaviour and waits the hop out.
        """
        # Declared before the ``try`` so the failure path can hand the bind task
        # to ``_release_when_landed`` even when the raise happened before it was
        # created (a capability check, a sync-payload build).
        bind_task: asyncio.Task[Any] | None = None
        try:

            from local_operator.session.frontend_state import (
                FrontendSubscription,
                oversized_frame_report,
                sync_wire_payload,
            )
            from local_operator.session.history_window import (
                strip_audit_fields,
                wire_payload,
            )

            window_requested = bool(frame.get("display_window")) and (
                "display-history-window-v1" in self._record.capabilities
            )
            # Negotiated exactly like ``display_window``: the viewer opts in by
            # declaring the flag, and an older viewer that cannot name it never
            # receives the fields it would reject.
            conn.audit_history = bool(frame.get("display_history_audit")) and (
                "display-history-audit-v1" in self._record.capabilities
            )

            # THE FRONTEND BIND IS A SESSION-LOOP CALL, for a stronger reason
            # than the two boot registrations: ``subscribe_frontend``'s first
            # act is ``refresh_from_session``, a PUBLISH that moves canonical
            # state and its sequence numbers, so running it on the runtime's
            # thread is a write to the publishing store from the wrong plane.
            # The handle's own seam takes care of it
            # (``@_on_session_loop``), so this call needs no hop of its own —
            # and deliberately has none, because two mechanisms for one hop is
            # how a later reader concludes that one of them is redundant and
            # removes the wrong one.
            #
            # It is also the one hop that can be SLOW, and what that slowness
            # costs is the viewer's canonical state and nothing else: this call
            # is made from ``_serve_frontend_sync``, a per-connection task, so
            # the connection is already in its reader loop and being served
            # while the hop is parked — and the accept, the heartbeat and every
            # other connection keep flowing for the same reason, because the
            # wait is on this task rather than on the runtime's loop.
            #
            # THE HOP IS BOUNDED NOW, BUT BY A FALLBACK TRIGGER RATHER THAN BY A
            # FAILURE BUDGET, and the difference is what keeps the old reasoning
            # intact. A budget belongs to a CALLER waiting for an answer; the only
            # caller here is a task nothing awaits, so letting the bind FAIL at a
            # deadline buys nothing and costs a live viewer its connection
            # (``mobile/tui_handle._on_app``'s unbounded branch carries that
            # measurement: a viewer dialled into a busy terminal was welcomed and
            # then killed at 10.01 s with ``owner exited`` while the app was
            # merely busy). Exceeding ``_ONLOOP_BIND_GRACE_S`` therefore changes
            # WHO binds, never whether: past the grace this task binds off-loop
            # and the abandoned hop is released as it lands (``bind_token``, then
            # ``_release_when_landed``). A healthy owner answers in 4.7-5.0 ms
            # p50, far inside the grace, so the path taken there is unchanged.
            #
            # The seam is SINGLE for both handle shapes, and that is why no hop
            # is added here: ``ServingSessionHandle.subscribe_frontend`` carries
            # ``@_on_session_loop`` (daemon/exec), and the TUI handle hops
            # through its own ``_on_app`` — a handle that publishes no
            # ``session_loop`` is served inline by
            # ``_handle_call_on_session_loop``, so the TUI kind keeps the hop it
            # already had rather than gaining a second.
            async def bind(on_update: Callable[[Any], None]) -> Any:
                outcome = (
                    subscribe_frontend(on_update, display_window=True)
                    if window_requested
                    else subscribe_frontend(on_update)
                )
                if inspect.isawaitable(outcome):
                    outcome = await outcome
                return outcome

            # THE BIND IS NOT CANCELLABLE ONCE IT IS RUNNING. The subscription is
            # created INSIDE the awaited handle call — on the session's loop, one
            # message before the reply that carries it back here — so cancelling
            # this task mid-bind can land on either side of that registration.
            # Shielding lets it land, and the handler below releases what it
            # registered; with only one of the two halves a cancelled bind leaks a
            # subscriber the session keeps for the life of the process. That is
            # also what makes ``_drop_client``'s ordering — cancel this task, then
            # release the recorded subscription — safe rather than lucky.
            bind_task = asyncio.ensure_future(bind(self._frontend_relay(conn, conn.bind_token)))
            subscription: FrontendSubscription | None
            try:
                subscription = await self._bind_with_grace(bind_task)
            except asyncio.CancelledError:
                self._release_when_landed(bind_task)
                raise
            if subscription is None:
                bind_off_loop = getattr(self._handle, "subscribe_frontend_nowait", None)
                if not callable(bind_off_loop):
                    # A single-plane handle keeps today's behaviour: wait the hop
                    # out. The grace is then a delay and nothing else, which is
                    # the honest cost of the only handle that cannot be served
                    # this way (the TUI kind, whose subscribe goes through the
                    # app's own loop).
                    subscription = await asyncio.shield(bind_task)
                else:
                    # RETIRE THE ABANDONED ATTEMPT BEFORE IT REGISTERS. The bump
                    # makes every callback stamped before it (the parked on-loop
                    # bind's, when it lands) return early instead of relaying a
                    # second copy of every delta into this connection.
                    conn.bind_token += 1
                    # ...and its subscription is released as it lands. Registered
                    # BEFORE the off-loop call, not after: the released callback is
                    # attached to ``bind_task`` either way, and doing it first means
                    # a raise or a cancellation anywhere below cannot leave a
                    # subscription nobody will receive.
                    self._release_when_landed(bind_task)
                    outcome = bind_off_loop(self._frontend_relay(conn, conn.bind_token))
                    if inspect.isawaitable(outcome):
                        outcome = await outcome
                    subscription = cast(FrontendSubscription, outcome)
            assert subscription is not None, "neither bind path produced a subscription"
            sync = subscription.sync
            # Trajectories are stripped here and re-fetched per job through
            # ``job_trajectory``; see ``sync_wire_payload`` for why the frame
            # cannot carry them.
            sync_payload = sync_wire_payload(sync)
            # The sync frame carries the FIRST display page, so it is one of
            # the THREE places the audit fields reach the wire (the others are
            # the ``history_page`` and ``frontend_sync`` RPCs). All three strip
            # through the same helper; stripping in only some would produce a
            # viewer that attaches cleanly and then fails on its first scroll
            # or on its first post-append refresh.
            if isinstance(sync_payload.get("display_history"), dict):
                strip_audit_fields(
                    sync_payload["display_history"], audit_capable=conn.audit_history
                )
            conn.frontend_unsubscribe = subscription.unsubscribe
            # Registration and snapshot capture happened synchronously on the
            # authoritative loop. Mark ready only after queuing that snapshot;
            # later updates therefore cannot overtake it.
            sync_frame = {"op": "frontend_sync", "data": sync_payload}
            # An oversized sync is unreadable, not merely large: the client's
            # readline raises and its pump dies, so the viewer waits out its
            # full sync timeout and then degrades to a cold session with no
            # roster and no todos. That looked exactly like a slow owner for
            # one release. Say so loudly and name the field responsible, so
            # the next unbounded list is one log line to find rather than a
            # profiling session.
            oversize = oversized_frame_report(sync_frame, _MAX_LINE_BYTES)
            if oversize is not None and sync.display_history is not None:
                # The budget covers the WHOLE frame, not just history. A busy
                # canonical state may leave too little room even for our page.
                # Request the existing exact local replay, never truncate prose.
                fallback = sync.display_history.model_copy(
                    update={
                        "status": "full_required",
                        "messages": [],
                        "durable_seed_ids": [],
                        "before_token": None,
                        "snapshot_token": None,
                    }
                )
                sync_payload["display_history"] = wire_payload(
                    fallback, audit_capable=conn.audit_history
                )
                oversize = oversized_frame_report(sync_frame, _MAX_LINE_BYTES)
            if oversize is not None:
                # Still over the line limit even with the display window
                # reduced. Nothing more can be shed HERE, and the guarantee the
                # socket needs lives in ``_send_to``'s ceiling: it substitutes an
                # ``error`` frame naming the size instead of writing a line this
                # client cannot read (which used to kill its pump and cost the
                # user the whole session). Reported here as well because this is
                # where the field responsible is known — the next unbounded list
                # should cost one log line to find, not a profiling session.
                #
                # What follows is deliberately unchanged and deliberately not
                # prettier: the deltas queued below are applied against a base
                # that never arrived, so the client refuses the first one by its
                # own sequence rule (``attach_client`` raises "frontend state
                # gap") and goes cold at once. That is the honest end for
                # canonical state too large to send — fast, named, and no
                # different in effect from the dead socket it replaces.
                logger.error(
                    "session runtime: frontend_sync does not fit — %s — and will be "
                    "replaced by an error frame at the write",
                    oversize,
                )
            await self._send_to(conn, sync_frame)
            conn.frontend_ready = True
            for pending in conn.frontend_pending:
                self._enqueue_client_frame(conn, {"op": "frontend_update", "data": pending})
            conn.frontend_pending.clear()
            if conn.wants_events:
                # Raw events ride the same FIFO as the canonical deltas and are
                # seeded by the sync frame just queued, so the gate opens only
                # now — a joiner must not receive transcript animation ahead of
                # the snapshot it animates.
                conn.events_ready = True
        except asyncio.CancelledError:
            # Cancellation is the connection going away (``_drop_client``), not a
            # bind failure: re-raise so the task settles as cancelled and the
            # subscription teardown stays in ``_drop_client``'s hands.
            raise
        except Exception as exc:  # noqa: BLE001 — one client's bind must not take the runtime down
            logger.warning(
                "session runtime: frontend bind failed for %s — dropping the connection",
                conn.surface,
                exc_info=True,
            )
            # A BIND THAT LANDS LATE STILL OWNS A SUBSCRIPTION (review round 2,
            # U7). The bind task is shielded, so a failure of THIS await does not
            # abort it: if it goes on to register a subscriber, that subscription
            # has no owner left to record it — the caller is unwinding — and the
            # session would keep pushing canonical state to it for the life of
            # the app. Measured on the timeout path this patch removes (BASELINE
            # subscribers = 2 → AFTER = 3, once per timed-out bind, compounding
            # with each redial). The same release the cancellation path uses
            # (F3) therefore runs here too: releasing a subscription nobody
            # received is always correct, and it makes the leak impossible by
            # construction rather than by the absence of a timeout.
            if bind_task is not None:
                self._release_when_landed(bind_task)
            # AND THE CLIENT IS TOLD WHAT HAPPENED (review round 2, U6). The
            # drop below closes a socket, and a closed socket reads to the far
            # side as "owner exited" — which is false and alarming: the process
            # is alive, this one connection could not be bound. Announced BEFORE
            # the drop, the same ordering ``stop``'s ``stopping`` frame uses and
            # for the same reason (there is no socket left afterwards); it is
            # additive on the wire, so an older client that does not know the op
            # falls back to exactly today's copy.
            await self._announce_bind_failure(conn, exc)
            self._drop_client(conn, reason="frontend bind failed")
        finally:
            # Opened by ``_on_connection`` before this task was created, and
            # cleared on every exit — including the drop above, where the
            # connection is gone and the gate no longer matters, and the
            # cancellation case, where ``_drop_client`` has already removed it.
            conn.frontend_sync_pending = False

    async def _announce_bind_failure(self, conn: _ClientConn, exc: BaseException) -> None:
        """Tell a viewer its connection could not be bound, before closing it.

        The frame is unsolicited (no ``req``) because the failure belongs to the
        CONNECTION, not to a request: it is the same shape as the ``stopping``
        and ``retiring`` announcements, and it is carried the same way — the
        client turns it into the reason string it reports when the socket closes
        moments later, so the person reads "owner could not prepare this
        session's interface" instead of "owner exited".

        Best-effort: a failure to announce must never stop the drop that
        follows, and the send path drops the client itself when the write fails.
        """
        try:
            await self._send_to(conn, {"op": "bind_failed", "message": str(exc)[:400]})
        except Exception:  # noqa: BLE001 — announcing is best-effort
            logger.debug("bind-failure announcement write failed", exc_info=True)

    def _release_when_landed(self, bind_task: asyncio.Task[Any]) -> None:
        """Release a bind's eventual subscription once nothing will receive it.

        Two callers, one contract. ``_drop_client`` cancels this connection's
        bind task, and the reason a cancelled bind cannot leave a live subscriber
        is STRUCTURAL rather than argued: the bind is shielded, so the cancel
        cannot abort it half-registered (``_serve_frontend_sync``), and this runs
        on the cancellation path to release whatever did register. The off-loop
        fallback reaches here having SUPERSEDED the same bind (``conn.bind_token``)
        — the shielded attempt still lands, and its subscription is just as
        unreachable. ``_drop_client``'s ordering follows from it — cancel first,
        then release the recorded subscription, so nothing is released twice.

        A DONE-CALLBACK rather than an await, and the difference matters twice
        over. ``_drop_client`` runs from the reader loop and from shutdown, so
        awaiting the bind there would park the teardown — and the loop with it —
        for as long as the abandoned hop takes, up to a whole hop budget, which is
        exactly the coupling this change exists to remove. And a callback cannot be
        skipped: a second cancellation racing the first would interrupt an await at
        the moment the release has to happen, while a callback registered on the
        task survives it.

        The task is held in ``self._abandoned_binds`` until it lands, because
        nothing else awaits it now: an unreferenced task can be collected
        mid-flight, and then its release never fires.
        """
        self._abandoned_binds.add(bind_task)

        def release(completed: asyncio.Task[Any]) -> None:
            self._abandoned_binds.discard(completed)
            if completed.cancelled():
                return
            error = completed.exception()
            if error is not None:
                logger.debug("frontend bind failed after its connection went away", exc_info=error)
                return
            unsubscribe = getattr(completed.result(), "unsubscribe", None)
            if callable(unsubscribe):
                try:
                    unsubscribe()
                except Exception:  # noqa: BLE001 — connection cleanup must finish
                    logger.debug("late frontend unsubscribe failed", exc_info=True)

        bind_task.add_done_callback(release)

    async def _on_connection(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        """One control connection. Auth is the first frame: ``{"key": ...}``
        within a short deadline, constant-time compared. Anything else closes
        without a reply — an open port that answers wrong keys with errors is
        an oracle, however small.

        Protocol v2 carries N connections: one ``daemon`` plus up to
        ``ATTACH_MAX_CLIENTS`` ``attach`` followers. A new daemon dial still
        REPLACES the old one (that is its reconnect story, preserved from the
        single-writer era); a further attach dial past the cap evicts the
        least-recently-seen attach client. Connection close is detected by
        the reader loop's ``finally``; the cap only guards leaked-but-open
        sockets liveness detection cannot see.
        """
        peer = writer.get_extra_info("peername")
        try:
            line = await asyncio.wait_for(reader.readline(), timeout=5.0)
            frame = json.loads(line.decode("utf-8", "replace"))
        except (TimeoutError, ValueError, UnicodeDecodeError):
            writer.close()
            return
        key = frame.get("key", "")
        if not isinstance(key, str) or not hmac.compare_digest(key, self._record.control_key):
            logger.warning("mobile control: rejected bad key from %s", peer)
            writer.close()
            return
        # Absent client field means daemon: an OLD daemon dialing a NEW
        # runtime must land on the class it always had, or every rolling
        # upgrade would demote the phone bridge to a follower.
        raw_kind = frame.get("client", "daemon")
        kind: ClientKind = "attach" if raw_kind == "attach" else "daemon"
        # Absent means LOCAL, matching every client that exists today: the
        # listener is loopback-only, so anything that dialed is on this
        # machine. A relay forwarding a remote device's commands is the one
        # caller that must say ``"remote"``, and an old client that never
        # heard of the field keeps the behaviour it always had.
        locality: ClientLocality = "remote" if frame.get("locality") == "remote" else "local"
        # v4: only attach clients may subscribe to the raw event relay. The
        # daemon's projection path must stay byte-identical, so a daemon auth
        # carrying the flag (there is none today) is deliberately ignored.
        wants_events = kind == "attach" and bool(frame.get("events"))
        wants_frontend = kind == "attach" and bool(frame.get("frontend_state"))
        # Which action-carrying receipts this client consumes itself. Parsed
        # ONCE here rather than per command: the answer cannot change for the
        # life of a connection. Absent (the common case today, and every
        # client built before the field) stays ``None`` so the completion path
        # can tell "did not declare" from "declared nothing" in the logs, even
        # though both admit. Advisory only — a malformed value degrades to
        # ``None`` and never refuses the connection, because a client that
        # cannot attach is strictly worse than one whose request gets run
        # twice-proof treatment it did not ask for.
        raw_consumers = frame.get("slash_consumers")
        slash_consumers: frozenset[str] | None = (
            frozenset(str(item) for item in raw_consumers)
            if isinstance(raw_consumers, (list, tuple))
            else None
        )
        # The client's half of the handshake (issue #1310). Only the SHAPE is
        # checked here — a nonce is not a credential, and an ill-shaped one
        # degrades to "this client asked for no handshake", which refuses rather
        # than admits. Nothing is remembered across connections.
        raw_nonce = frame.get("operator_nonce")
        client_nonce = raw_nonce if is_wire_hex(raw_nonce) else ""
        # The salt is minted per connection and only when there is a nonce to
        # bind it to: a connection that will never be offered a proof does not
        # need one minted.
        server_salt = operator_nonce() if client_nonce else ""
        # THE PAIRED-DEVICE DECLARATION (stage D). A relay says here that a phone
        # has been paired with THIS machine, and the runtime verifies the
        # certificate under the pinned anchor before it means anything — an
        # UNVERIFIED string changes no behaviour, so a forged one is refused
        # rather than denied service. Nothing crosses on it: the certificate is
        # public data, and the private half that could make it useful is on the
        # phone.
        #
        # It is DECLARED rather than derived from the frames because one report
        # needs the answer before any frame arrives: whether this connection can
        # carry out a loosening, which decides whether `/approvals` tells the
        # phone it may switch the gate or tells it to find another surface
        # (``_connection_may_loosen``).
        raw_device = frame.get("operator_device")
        device_certificate = (
            raw_device if isinstance(raw_device, str) and 0 < len(raw_device) <= 4096 else ""
        )

        if wants_frontend and FRONTEND_CAPABILITY not in self._record.capabilities:
            writer.close()
            return

        if kind == "attach" and self._exclusive_move_fence is not None:
            # REGISTRATION FENCE (review R3). An exclusive move has reserved
            # this runtime for one facade and is retiring it; a facade admitted
            # here would engage the successor from its OWN cwd, which is exactly
            # the contradictory-successor race the fence exists to prevent. The
            # check is one synchronous read on this loop, placed BEFORE the
            # insertion below so it cannot race the reservation.
            writer.close()
            return
        if kind == "daemon":
            # At most ONE daemon connection — a new dial evicts the old, which
            # is also the reconnect path after a daemon restart.
            for other in [
                c for c in self._clients.values() if c.kind == "daemon" and c.writer is not writer
            ]:
                self._drop_client(other, reason="daemon replaced")
        else:
            # Attach cap with LRU eviction: the least-recently-seen follower
            # goes. Sending on the evicted socket first (a goodbye) is not
            # worth the failure modes — its reader loop is still alive and
            # will observe the close as EOF, which is the attach screen's
            # owner-death signal minus a corpse.
            attaches = [c for c in self._clients.values() if c.kind == "attach"]
            if len(attaches) >= ATTACH_MAX_CLIENTS:
                victim = min(attaches, key=lambda c: c.last_seen)
                self._drop_client(victim, reason="attach cap")

        conn = _ClientConn(
            writer=writer,
            kind=kind,
            locality=locality,
            surface=(
                "desktop" if kind == "attach" and frame.get("surface") == "desktop" else "terminal"
            ),
            slash_consumers=slash_consumers,
            wants_events=wants_events,
            wants_frontend=wants_frontend,
            operator_nonce=client_nonce,
            operator_salt=server_salt,
            device_certificate=device_certificate,
        )
        self._clients[id(writer)] = conn
        # A terminal arriving flips ``detached`` (round 1, U2: it was computed
        # only inside a pending transition, so every session claimed a viewer
        # it might never have had). Cheap and de-duplicated — see
        # ``_republish_detached``.
        if kind == "attach":
            self._republish_detached()
        if kind == "daemon":
            # The daemon is the one client that renders projections (attach
            # clients read the welcome for identity only), so its arrival is
            # when a lazily-built fold earns its keep.
            self._ensure_projection_sink()
        await self._push_to(conn)  # the welcome: a full projection, unprompted
        if conn.wants_frontend:
            subscribe_frontend = getattr(self._handle, "subscribe_frontend", None)
            if not callable(subscribe_frontend):
                self._drop_client(conn, reason="frontend requested but unsupported")
                return
            # THE BIND RUNS IN ITS OWN TASK, so this connection's reader loop is
            # already serving by the time the heavyweight part starts.
            #
            # It used to run inline, which made a follower's socket DEAF rather
            # than slow: the bind is a cross-thread hop onto the session's loop,
            # so a session inside a synchronous step of a turn held the connection
            # before it could read anything at all — no ``ping``, no ``stop``, no
            # ``steer`` — while a daemon dial and a phone dial to the same runtime
            # both answered in 0.00 s (measured; review round 2, UX U6). The reader
            # loop must therefore start FIRST.
            #
            # The welcome above stays synchronous and hop-free deliberately: it is
            # an identity frame read from the cached seed, so it costs no
            # session-loop access and there is nothing to defer.
            #
            # ``_on_request`` admits :data:`_SYNC_PRIORITY_OPS` while this task is
            # pending and refuses everything else, so the connection is reachable
            # without yet being authoritative.
            conn.frontend_sync_pending = True
            conn.frontend_sync_task = asyncio.create_task(
                self._serve_frontend_sync(conn, frame, subscribe_frontend)
            )
        elif conn.wants_events:
            # A client that asked for BOTH the frontend and events waits for the
            # sync task to open this gate (``_serve_frontend_sync``); one that
            # asked only for events has no snapshot to be seeded behind and starts
            # now.
            #
            # THE STREAM ITSELF STAYS INLINE — the relay, its writer task and the
            # ``epoch``/``sequence`` check a client closes a gapped connection over
            # are untouched. What moved is only the moment this gate opens, because
            # that is the part that has to be ordered after the sync frame: a
            # joiner receiving events first would fail its own gap check.
            conn.events_ready = True
        # A FLOOR on the bytes discarded by the CURRENT oversized line, not the
        # exact figure — see where it is incremented. Doubles as the "already
        # reported this one" flag, since a run always starts at zero: one
        # oversized frame raises REPEATEDLY — once per limit-sized chunk,
        # measured at 7 raises for a 10 MB line — so a log call per raise would
        # turn one bad message into a burst that buries the fact that they were
        # all the same frame.
        overrun_bytes = 0
        try:
            while not self._closed.is_set():
                try:
                    line = await reader.readline()
                except ValueError:
                    # AN OVERSIZED INBOUND LINE MUST NOT BE A FATAL ERROR.
                    # ``start_server(..., limit=_MAX_LINE_BYTES)`` makes
                    # ``readline`` raise ``ValueError`` (wrapping
                    # ``LimitOverrunError``) for a line past the limit, and the
                    # raise happens HERE, outside the ``json.loads`` try below
                    # that looks like it covers it. Uncaught it escaped the
                    # ``ConnectionResetError``/``BrokenPipeError`` handler, the
                    # reader loop died, and the ``finally`` dropped the
                    # connection — so a client sending one large frame lost its
                    # whole session and the operator had to ``lop --resume``.
                    # A pasted screenshot over ~780 KB of source does it, which
                    # is how this was found.
                    #
                    # CONTINUING IS SAFE HERE and is NOT the infinite loop the
                    # sibling ``viewer_server`` guard warns about. The
                    # difference is ``readline`` vs ``readuntil``: ``readuntil``
                    # leaves the offending bytes in the buffer so the next read
                    # re-raises forever, while ``readline`` catches its own
                    # ``LimitOverrunError`` and DRAINS — deleting through the
                    # separator when one was found and clearing the buffer when
                    # it was not (CPython ``asyncio/streams.py``). So the next
                    # read starts at the following frame, and the connection
                    # survives to carry the user's NEXT message, which is the
                    # behaviour the operator actually missed.
                    #
                    # LOUDLY, at error: the frame is gone and whoever sent it
                    # is owed the reason. ``AttachClient`` now refits images
                    # before sending (``fit_request_frame``), so a line
                    # reaching this point means an old client, a non-attach
                    # peer, or a bug — each of which is worth one greppable
                    # line rather than a vanished session. (How OFTEN it fires
                    # is the next paragraph's subject — once per run of
                    # consecutive discards, not once per frame; an earlier
                    # version of this paragraph claimed the latter and was
                    # wrong, review round 3, MINOR-2.)
                    #
                    # ONCE PER RUN of consecutive discards, not once per frame,
                    # and it names the limit rather than the size because it is
                    # the row that arrives FIRST: a frame only marginally over
                    # produces a single raise whose separator lands inside the
                    # discarded chunk, so no read succeeds until the peer sends
                    # something else and the summary below may not come for a
                    # while (QA round 1, Q2). The operator learns a frame was
                    # dropped, and why, from this row alone.
                    #
                    # BE PRECISE ABOUT THE GAP, because an earlier version of
                    # this comment claimed the line "always fires" and it does
                    # not. The gate is ``overrun_bytes``, which clears only on a
                    # successful read, so an unbroken run of marginal frames
                    # logs this row for the FIRST one and nothing for the rest
                    # — measured at 4 marginal frames with no readable traffic
                    # between: 1 row, 3 silent. A large frame self-clears the
                    # latch through its own readable tail, which is why the run
                    # looks unconditional in testing (QA round 2, Q4).
                    #
                    # Kept as a run-level latch rather than made per-frame: the
                    # two cases are distinguishable only by the wording of
                    # CPython's own ``LimitOverrunError`` ("Separator is found"
                    # vs "is not found"), and pinning diagnostics to an
                    # undocumented message string trades a quiet log for a
                    # silent breakage on the next CPython. The frames are still
                    # discarded and the session still survives on every one of
                    # them — this is a diagnostic gap, not a delivery one — and
                    # the byte total below still accounts for the whole run.
                    if overrun_bytes == 0:
                        logger.error(
                            "session runtime: dropping an inbound frame from %s client %s "
                            "over the %d-byte line limit; the frame is discarded and the "
                            "connection kept — the sender must resize its payload",
                            conn.kind,
                            conn.writer.get_extra_info("peername"),
                            _MAX_LINE_BYTES,
                        )
                    # A FLOOR, and the summary below says so in words. CPython's
                    # ``readline`` clears ``len(self._buffer)`` on the
                    # no-separator path, and that buffer is ``>= limit`` — in
                    # practice more, because the overrun is only detected once a
                    # whole transport chunk has landed. The discarded amount is
                    # therefore not reachable from here without instrumenting the
                    # transport, and measured runs put the shortfall at 6.7-18.8%
                    # (QA round 1, Q1). Reporting a floor AS a floor is honest;
                    # reporting it as a total is what made the number wrong.
                    overrun_bytes += _MAX_LINE_BYTES
                    continue
                if overrun_bytes:
                    # ``overrun_bytes`` ALONE, without this line's length. The
                    # read that succeeds here is usually the tail of the
                    # oversized frame, but not always: when the separator landed
                    # inside a discarded chunk there is no tail, and this is the
                    # peer's next, innocent message — whose bytes were being
                    # added to the bad frame's total, attributing an unrelated
                    # message's size to it (QA round 1, Q2).
                    #
                    # Naming the size is what makes the log actionable: "over 1
                    # MiB" does not say whether to trim one screenshot or five.
                    # "at least" is what makes it TRUE — see the increment above.
                    logger.error(
                        "session runtime: discarded at least %d bytes of oversized inbound "
                        "frame from %s client %s; the connection is still up and the next "
                        "message will be read normally",
                        overrun_bytes,
                        conn.kind,
                        conn.writer.get_extra_info("peername"),
                    )
                    overrun_bytes = 0
                if not line:
                    self._drop_client(conn, reason="reader eof")
                    return  # client hung up
                try:
                    frame = json.loads(line.decode("utf-8", "replace"))
                except ValueError:
                    continue
                conn.last_seen = time.monotonic()
                self._dispatch_frame(conn, frame)
        except (ConnectionResetError, BrokenPipeError):
            self._drop_client(conn, reason="reader reset")
            return
        finally:
            self._drop_client(conn, reason="reader eof")

    def _dispatch_frame(self, conn: _ClientConn, frame: dict[str, Any]) -> None:
        """Run one admitted request OFF this connection's reader loop.

        WHY THE READER NO LONGER AWAITS IT. The reader used to
        ``await self._on_request(...)``, which made a connection strictly serial
        — right for ORDERING (two mutations must not interleave) but wrong for
        LIVENESS: while one op is parked inside a hop into the session's loop,
        the reader cannot read the next frame, so everything else that client
        sent waits behind it. Measured over a real socket (review round 1, UX
        U3): a parked ``steer`` left ``ping`` on that same connection unanswered
        for 8-15 s — the one request a surface speaks to ask "are you there",
        queued behind a mutation. New connections were never affected (a fresh
        dial, its ping and its refusals all answered in 0.00 s), so "always
        connectable" held while "prioritize the health check" did not.

        ORDERING IS PRESERVED BY CHAINING, not by concurrency: each op waits for
        the one admitted before it, so a connection's frames still run one at a
        time and in arrival order. What changes is only that the READER stays
        free while it waits, and therefore keeps answering whatever else that
        client sends.

        The chain link is waited on with :func:`asyncio.wait`, which REPORTS a
        task's outcome rather than raising it: the earlier op's failure is its
        own business (``_on_request`` answers its own errors with an error
        frame), and an ordering link must never be an error channel that takes
        the next request down with it.

        ``ping`` is exempt from the chain (see :data:`_UNCHAINED_OPS`): a health
        check queued behind a mutation answers the wrong question, and measured
        (review round 1, UX U3) it did exactly that — 8-15 s of silence on a
        connection whose only sin was a parked ``steer``. An exempt op also does
        NOT become the chain head, and that is load-bearing rather than tidiness:
        the head is what the next mutation waits on, so letting a ping (which
        finishes at once) take it would let the mutation admitted after the ping
        overtake the one still in flight — reproduced before this line existed,
        where a parked ``steer`` was overtaken by the next ``steer`` because a
        ``ping`` had been admitted in between. The head therefore stays with the
        last CHAINED op, and the chain is transitive: waiting on the head is
        waiting on everything admitted before it.

        Cancellation is deliberately NOT propagated to these tasks. An op parked
        on a dead connection is left to unwind on its own, exactly as before
        this change (``_drop_client``'s ``phone_watchers`` note depends on that:
        an evicted daemon parked inside ``_on_request`` returns on its own), and
        ``conn.op_tasks`` holds the strong references meanwhile.
        """
        # ``isinstance`` FIRST, and it is load-bearing rather than defensive: a
        # frame that parses as a bare JSON scalar (``12345``, ``null``,
        # ``"str"`` — reachable when an oversized line is discarded and its
        # surviving TAIL happens to parse) reaches here as an int/None/str, and
        # ``.get`` on one raises. That exception would escape the reader loop's
        # ``ConnectionResetError``/``BrokenPipeError`` handler and take the
        # connection down — the exact session death
        # ``test_a_junk_scalar_frame_does_not_kill_the_connection`` exists to
        # prevent, which the reader loop already guards against INSIDE
        # ``_on_request``. This check must therefore never become a second,
        # crashing gate in front of it: a non-dict frame is chained like any
        # other and left to the dispatch to reject.
        chained = not (isinstance(frame, dict) and frame.get("op") in _UNCHAINED_OPS)
        previous = conn.op_chain if chained else None

        async def run() -> None:
            if previous is not None:
                await asyncio.wait({previous})
            # ADMITTED BEFORE IT RUNS, and released when it settles however it
            # settles: this span is what ``_shutdown_impl`` waits on so the
            # reply to an admitted request is written before the socket it
            # belongs to is closed.
            self._admit_request()
            try:
                await self._on_request(frame, conn)
            finally:
                self._release_request()

        task = asyncio.create_task(run())
        if chained:
            conn.op_chain = task
        conn.op_tasks.add(task)
        task.add_done_callback(lambda completed: self._op_settled(conn, completed))

    def _op_settled(self, conn: _ClientConn, completed: asyncio.Task[None]) -> None:
        """Retire a dispatched op and consume its outcome.

        The strong reference goes away here, and an exception is LOGGED rather
        than left for the garbage collector: ``_on_request`` answers its own
        failures with an error frame, so anything that escapes it is a bug in
        the dispatch path, and an unretrieved task exception would surface much
        later as an asyncio "never retrieved" warning naming no connection.
        """
        conn.op_tasks.discard(completed)
        if completed.cancelled():
            return
        error = completed.exception()
        if error is not None:
            logger.warning(
                "session runtime: an admitted request failed outside its own "
                "error frame for %s client %s",
                conn.kind,
                conn.writer.get_extra_info("peername"),
                exc_info=error,
            )

    def _drop_client(self, conn: _ClientConn, *, reason: str = "unspecified") -> None:
        """Remove one connection from the registry and close its socket.

        The ONLY removal path: reader-loop exit, shutdown, daemon eviction,
        and attach-cap eviction all funnel here so the registry can never
        retain an entry whose socket is closed (the reaper counts them)."""
        # DID THIS CALL ACTUALLY REMOVE THE CONNECTION? `_drop_client` is
        # designed to run TWICE on one connection — `_send_to` drops a client
        # whose send failed, and that connection's reader loop later observes
        # the close and drops it again from its `finally`, which the docstring
        # there calls "a no-op second removal". Anything below that mutates
        # SERVER-GLOBAL state has to honour that contract, or the late second
        # call reaches across to whatever connection replaced this one.
        was_registered = self._clients.pop(id(conn.writer), None) is not None
        # One line per actual removal, at ONE level chosen by WHY it happened.
        #
        # A view that went cold and was told to reselect used to leave nothing
        # in the log to read: the removal was recorded at INFO beside every
        # routine close, so the reason a watching terminal lost its owner — the
        # attach cap evicting it, an event queue overflowing, a send timing out
        # — could not be found without turning INFO on for the whole runtime.
        # The reasons below that mean "we dropped a client that did not ask to
        # leave" are therefore WARNING, and the ones that are an ordinary part
        # of a client's life stay INFO. The reader loop's `finally` always calls
        # again after a send-path drop; that second call is a no-op and must not
        # look like a second failure (DEBUG only).
        peer = conn.writer.get_extra_info("peername")
        if not was_registered:
            log = logger.debug
        elif reason in _GRACEFUL_DROP_REASONS:
            log = logger.info
        else:
            log = logger.warning
        log(
            "session runtime: dropped %s client %s (events=%s frontend=%s surface=%s): %s",
            conn.kind,
            peer,
            conn.wants_events,
            conn.wants_frontend,
            conn.surface,
            reason,
        )
        # The other half of the ``detached`` transition: the last terminal
        # leaving is precisely when the picker must start saying "nobody is
        # watching this". Published from the ONE removal path so no exit route
        # (reader-loop end, shutdown, eviction) can miss it.
        if conn.kind == "attach":
            self._republish_detached()
        # `phone_watchers` is the daemon CONNECTION's state kept in a
        # server-global counter, and only an `unwatch` op decrements it — an op
        # the daemon sends from an SSE generator's `finally`, in the process
        # that just died. So a daemon restart while a phone is watching left
        # the +1 behind forever: the new daemon's `_reconcile` replays `watch`
        # (a second +1), the phone's eventual close sends ONE `unwatch`, and
        # the residue reports a viewer nobody can see. Every parked approval on
        # that session then sends no toast and the model is told a human is
        # watching — round 3's B1 failure mode, restored by a third route
        # (round 5, R5).
        #
        # Zeroed rather than decremented: the count belongs to the connection
        # that reported it, a new daemon re-announces every session it watches,
        # and at most one daemon connection exists at a time.
        #
        # GUARDED ON `was_registered`, because at most one daemon connection
        # exists but its DROPS are not unique. An evicted daemon parked inside
        # `_on_request` unwinds only when its await returns — by then the
        # replacement has dialled, replayed `watch`, and owns the counter, so
        # an unconditional zero here wiped a LIVE watcher (round 6, R7). That
        # failed OPEN, unlike the leak it replaced: a phone being looked at
        # reported nobody watching, so every parked approval toasted a card
        # already on the user's screen and the model was told no one could
        # answer. Derived from the registry rather than asserted, the same way
        # the `attach` half above computes `detached`.
        if conn.kind == "daemon" and was_registered:
            self.phone_watchers = 0
        # The frontend bind's own task goes first, and for the same reason as the
        # writer above: it is a connection-owned task that must not outlive the
        # connection. Ordering matters here — cancel it BEFORE releasing the
        # recorded subscription, so the shielded bind cannot register an
        # unsubscribe nobody will ever call. ``_release_when_landed`` is the other
        # half of that guarantee, and ``is not current_task`` covers the bind
        # dropping its own connection when it fails.
        sync_task = conn.frontend_sync_task
        conn.frontend_sync_task = None
        if sync_task is not None and sync_task is not asyncio.current_task():
            sync_task.cancel()
        task = conn.event_writer_task
        conn.event_writer_task = None
        if task is not None:
            self._event_sends.discard(task)
            # A send timeout drops its own connection from inside this task.
            # Cancelling self here would interrupt the cleanup path at its next
            # await and leave the stream close only half-observed.
            if task is not asyncio.current_task():
                task.cancel()
        frontend_unsubscribe = conn.frontend_unsubscribe
        conn.frontend_unsubscribe = None
        if frontend_unsubscribe is not None:
            try:
                frontend_unsubscribe()
            except Exception:  # noqa: BLE001 — connection cleanup must finish
                logger.debug("frontend client unsubscribe failed", exc_info=True)
        try:
            conn.writer.close()
        except Exception:  # noqa: BLE001
            pass

    def set_record_pending(self, pending: str | None) -> None:
        """Record that this session is waiting for a PERSON (or no longer is).

        Named for the RECORD it writes, distinct from ``set_pending`` below,
        which carries a ``PendingRequest`` into the projection so a front end
        can paint the card. Two different consumers: that one is "show the
        user this question", this one is "tell the machine a person is owed".

        ``"approval"`` / ``"ask"`` / ``None``. Republished immediately rather
        than waiting for the 15 s heartbeat, because the whole value of the
        field is that a user hunting for "what is that 283 MB process doing"
        finds the answer at once.
        """
        if self._pending == pending:
            return
        self._pending = pending
        self._republish()

    def _republish_detached(self) -> None:
        """Refresh the record when the attached-terminal count crosses 0↔1.

        De-duplicated on the resulting BOOLEAN rather than on the count: a
        second terminal attaching to a session that already had one changes
        nothing a reader can see, and republishing for it would put a staged
        write on every connection churn.
        """
        detached = not bool(self._visible_attach_surfaces())
        delivery = bool(self.notification_surfaces())
        if detached == self._detached and delivery == self._desktop_delivery:
            return
        self._detached = detached
        self._desktop_delivery = delivery
        self._republish()
        if detached and self._pending:
            # A GATE WAS OPENED WHILE SOMEBODY WAS WATCHING, and they have now
            # closed the terminal. The routing decision was made once, at
            # announce time, and correctly sent no toast then — so without
            # this the question waits up to 24 h and the user is never told
            # (round 3, B2). "I approved something, shut the laptop, came back
            # to a session that had been waiting all day" is the ordinary
            # shape of it. Re-announcing on the transition is what turns the
            # one-shot decision into a live one.
            announce = getattr(self._handle, "reannounce_pending", None)
            if callable(announce):
                try:
                    announce()
                except Exception:  # noqa: BLE001 — a toast never breaks teardown
                    logger.debug("could not re-announce the parked gate", exc_info=True)

    def set_busy(self, busy: bool) -> None:
        """Record whether a turn is running, for the picker's liveness marker."""
        if self._busy == busy:
            return
        self._busy = busy
        self._republish()

    def note_leaving(self, phrase: str) -> None:
        """Publish that this runtime has committed to leave, and is finishing
        work in flight first (``LEAVING_ON_SIGNAL``; see
        ``SessionRecord.leaving`` for why the record carries it).

        Written THROUGH to the record in the same synchronous step as the
        assignment, exactly like :meth:`set_record_started`: a reader between
        here and the next heartbeat must already see it, and the whole point of
        the field is the window BEFORE the exit — a marker that arrived with the
        ordinary 15 s heartbeat would leave up to a seventh of the drain
        invisible, which is most of the window it exists to describe.

        Deduped like :meth:`set_busy`, and it matters more here: the drain calls
        this once, but a repeat signal or a second drain arm on the same runtime
        must not put a staged write and rename on the far side of a signal.
        """
        if self._leaving == phrase:
            return
        self._leaving = phrase
        self._record.leaving = phrase
        self._republish()

    def note_updating(self, pair: str) -> None:
        """Publish that an UPDATE WINDOW is open for ``pair`` (``""`` clears it).

        THE ONE WRITER of ``SessionRecord.updating``, so the record and the handle's
        admission state cannot drift: ``serving.ServingSessionHandle.begin_update``
        and ``end_update`` both reach it through the server they hold, and a runtime
        whose handle is not this server's (a reduced host) simply never publishes.

        Written THROUGH to the record in the same synchronous step as the
        assignment, like :meth:`note_leaving` and for a sharper version of its
        reason: the window is about a second long, so a field that waited for the
        15 s heartbeat would be published only AFTER the handover it describes had
        ended — and the surfaces that read a record would never once see a session
        mid-update, which is the whole feature.

        Deduped like :meth:`set_busy`. Idempotent on the clear, because both
        ``end_update`` and :meth:`note_update_failed` close a window and neither can
        tell whether the other already has.
        """
        if self._updating == pair:
            return
        self._updating = pair
        self._record.updating = pair
        if pair:
            # A NEW WINDOW SUPERSEDES THE LAST FAILURE (agent review round 1, NIT 4).
            # Without this the record keeps describing an abandoned move for the rest
            # of the process's life — a fleet row that says "update failed" about a
            # session which has since moved on, or is moving right now — because
            # nothing else clears the field: the success arm's exit takes the whole
            # record away, and the abandon arm is what writes it. The field is
            # re-published by ``note_update_failed`` if THIS attempt fails too.
            self._record.update_failed = ""
        self._republish()

    async def note_update_failed(self, pair: str, bound: float = 0.0) -> None:
        """Publish that the window for ``pair`` ran out of its bound.

        THE FAILURE HAS TO BE REPORTABLE, which is the operator's own requirement
        ("indicate that the update failed so that it can be reported as an issue and
        addressed"), and it is stated twice on purpose, because the two surfaces
        answer different questions and either alone is a hole:

        * the RECORD (``update_failed``) is what a front end that was not watching
          at the time can still read — the TUI's fleet row, ``lop sessions``, the
          phone's projection, the desktop feed. It is also what says the runtime is
          still SERVING, which is the part a person acts on;
        * the INCIDENT ROW is the durable account in the conversation, carrying
          ``types.UPDATE_FAILED_CAUSE`` so every surface that repeats a cause can
          render it as a sentence (``incidents.CUT_OFF_CAUSES``). Without it the
          bounded window would be exactly the silent failure the bound was written
          to prevent — the shape of QA round 1, Q-2, where the overdue handover
          shipped with a token nothing could render.

        NEVER RAISES. The caller is the rung that has just decided to KEEP this
        runtime serving, and a runtime that stayed is a successful outcome even if
        its own bookkeeping could not be written.
        """
        self._updating = ""
        self._record.updating = ""
        self._update_failed = pair
        self._record.update_failed = pair
        self._republish()

        session = getattr(self._handle, "_session", None)
        journal = getattr(session, "journal_incident", None)
        if not callable(journal):
            return
        write_incident = cast(Callable[..., Awaitable[None]], journal)
        from local_operator import incidents
        from local_operator.session.runtime.types import UPDATE_FAILED_CAUSE

        # THE BOUND THE CALLER ACTUALLY ENFORCED, which is why it is a parameter
        # (design review round 1, D2): two arms publish this token with two different
        # bounds, and the shared sentence names none of them, so the failure reports
        # its own number here or it reports no number at all. Rendering the window's
        # constant instead told the operator that an update which had spent fifteen
        # minutes in the drain gave up "within 5s".
        rendered = incidents.render_cut_off_reason(
            UPDATE_FAILED_CAUSE, detail=incidents.update_failed_detail(pair, bound)
        )
        try:
            await write_incident(UPDATE_FAILED_CAUSE, token=UPDATE_FAILED_CAUSE, rendered=rendered)
        except Exception:  # noqa: BLE001 — a failure notice never breaks the runtime
            logger.warning("could not journal the failed update", exc_info=True)

    def set_record_started(self, started: bool) -> None:
        """Record that this session has run at least one real turn.

        One-way, and enforced: once a turn has run, ``False`` is IGNORED —
        the caller is the per-turn hook in ``_run_turn_pipeline``, for which
        ``False`` can only ever be a mistake (no code path un-runs a turn).
        The only legitimate way the bit drops again is a session-identity
        swap, which goes through :meth:`reset_record_started` instead.
        """
        if not started or self._started:
            return
        self._started = True
        # Write through to the record NOW, not only on the next republish: the
        # publisher serializes ``self._record``, and a caller reading the record
        # between here and the republish (or a republish that never comes, e.g.
        # no publisher yet) must already see the flipped bit.
        self._record.started = True
        self._republish()

    def reset_record_started(self, started: bool) -> None:
        """Re-seed the ``started`` bit for a NEW session identity.

        NOT the turn-running signal :meth:`set_record_started` answers: a
        TUI's registrant outlives ``/new`` and ``/resume``
        (``TuiSessionHandle.rebind`` re-points it at the new session), so the
        bit must be re-derived from the NEW session's own durable history
        instead of inherited from the old one. A ``/new`` after a working
        conversation must drop back to ``False`` — the composer window this
        flag exists for — while a ``/resume`` must read ``True``, because that
        conversation already ran turns and a peer wake could always reach it.
        Both directions are legal HERE only because the identity changed.
        """
        self._started = started
        self._record.started = started
        self._republish()

    def set_subagents(self, running: int | None, queued: int | None) -> None:
        """Record this runtime's subagent trajectory counts.

        Deduped exactly like :meth:`set_busy`, and for the same reason: the
        driver is ``_publish_busy`` on every session event, so the steady-state
        cost has to be one comparison rather than a staged write.

        ``None`` is accepted and republished as ``None`` on purpose — a handle
        that cannot answer the probe (a ``kind="tui"`` runtime whose handle
        does not implement it) must publish "I do not report" rather than a
        fabricated zero, which is the distinction ``/info``'s lower-bound
        caveat is built on.
        """
        if self._subagents_running == running and self._subagents_queued == queued:
            return
        self._subagents_running = running
        self._subagents_queued = queued
        self._republish()

    def _republish(self) -> None:
        """Refresh the discovery record with the current live state.

        Through ``RecordPublisher.heartbeat``, which is already the one way
        this process rewrites its record — a second publish path here would be
        a second thing that can disagree about the record's contents.

        Best-effort: publishing is one small staged write and rename, and a
        failure costs a marker rather than a session. Called on every
        transition rather than left to the 15 s heartbeat because the value of
        these fields is that they are current when somebody looks.
        """
        publisher = getattr(self, "_publisher", None)
        if publisher is None:
            return
        try:
            publisher.heartbeat(
                pending=self._pending,
                busy=self._busy,
                leaving=self._leaving,
                started=self._started,
                detached=not bool(self._visible_attach_surfaces()),
                subagents_running=self._subagents_running,
                subagents_queued=self._subagents_queued,
            )
        except Exception:  # noqa: BLE001 — a stale marker is not worth an exception
            logger.debug("could not republish the session record", exc_info=True)

    def attach_clients(self) -> int:
        """Live terminal viewers or leased desktop delivery surfaces.

        The reaper must not keep an idle runtime forever because a crashed
        renderer left its HTTP proxy connection behind. Connection-cap eviction
        still counts raw sockets separately, so expired leases cannot bypass it.
        """
        return sum(
            1
            # SNAPSHOT BEFORE ITERATING (C8): ``attach_clients`` is read by the
            # reaper AND by the handle (``serving``'s attach-surface count) from
            # the session's loop, while this dict is mutated on the runtime's.
            # The reaper treats a raise as "no viewers", so the failure mode is a
            # runtime that keeps itself alive forever rather than an error.
            for c in list(self._clients.values())
            if c.kind == "attach"
            and (
                c.surface != "desktop"
                or (self._desktop_lease_live(c) and (c.desktop_visible or c.desktop_can_notify))
            )
        )

    def _desktop_lease_live(self, conn: _ClientConn) -> bool:
        return (
            conn.surface == "desktop"
            and time.monotonic() - conn.desktop_seen < DESKTOP_WATCH_LEASE_S
        )

    def _visible_attach_surfaces(self) -> set[str]:
        """Which KINDS of surface are actually being LOOKED AT right now.

        The difference between this and :meth:`notification_surfaces` is the
        whole of rung 1 of the notification ladder. This one answers "is a
        person reading this session"; that one answers "could a banner reach
        them somewhere on this machine". Using the reachability predicate as a
        SUPPRESSION predicate is what made "this machine can banner" read as "a
        human is reading X": with the panel on X and the window behind another
        app, every OS surface went quiet while nobody was looking.

        A DESKTOP CONNECTION'S VISIBILITY IS NO LONGER THE RENDERER'S
        ``document.visibilityState``. That value says nothing sound about the
        window: a window behind another app is "visible", and a window Electron
        is throttling can report "hidden" while the user is reading it. So when
        a machine-wide delivery presence exists, ITS window state is the
        authority, and it can both grant and deny. When no presence exists — an
        older app, or none at all — the per-session flag below stands
        unchanged, which is what keeps this additive: an old UI's behaviour is
        byte-identical, and only a new one's occluded-window case changes (a
        banner IS raised, per the design's matrix).

        A TERMINAL ATTACH IS COUNTED ONLY WHILE IT IS DISPLAYING THIS SESSION,
        for the same reason the desktop leg consults window state rather than
        the socket. A multiplexing TUI retains the outgoing session's
        connection when the user switches away, so a live attach stopped
        meaning "on screen" and a gate parked behind one waited in silence for
        a card painted into a viewer showing something else. ``viewer_watch``
        carries that fact; a client that never sends it stays counted, so an
        older viewer behaves exactly as before.
        """
        # SNAPSHOT BEFORE ITERATING. ``_clients`` is mutated by the runtime's
        # own loop on every connect and disconnect, and this predicate is now
        # reached from the SESSION's loop too (the handle's ``_republish`` and
        # ``attach_clients`` callbacks), so a `RuntimeError: dictionary changed
        # size during iteration` is reachable. The fallout would not look like a
        # crash: the failure is caught by the caller and answers "no viewer",
        # i.e. a RESIDENCY decision taken on a failed read, which is how a
        # runtime with a viewer gets reaped.
        return {
            "desktop" if conn.surface == "desktop" else "attach"
            for conn in list(self._clients.values())
            if conn.kind == "attach"
            and (
                (conn.surface != "desktop" and conn.terminal_displaying)
                or (
                    conn.surface == "desktop"
                    and self._desktop_lease_live(conn)
                    and self._desktop_visible(conn)
                )
            )
        }

    def _desktop_visible(self, conn: _ClientConn) -> bool:
        """Whether this session is on a desktop window somebody is looking at.

        See :meth:`_visible_attach_surfaces` for why the machine-wide answer
        wins where it exists and the renderer's flag is the fallback where it
        does not.
        """
        try:
            from local_operator.session.runtime.presence import desktop_presence

            presence = desktop_presence(getattr(self, "_config_root", None) or config_dir())
        except Exception:  # noqa: BLE001 — a presence read must never break a gate
            logger.debug("could not read the desktop presence", exc_info=True)
            return conn.desktop_visible
        if not presence.present:
            return conn.desktop_visible
        record = getattr(self, "_record", None)
        session_id = str(getattr(record, "session_id", "") or "")
        return presence.attended and presence.session_id == session_id

    def notification_surfaces(self) -> frozenset[str]:
        """Delivery reachability is independent of a person viewing a session."""
        return (
            frozenset({"desktop"})
            if any(
                conn.kind == "attach" and self._desktop_lease_live(conn) and conn.desktop_can_notify
                # SNAPSHOT BEFORE ITERATING (C8). This reader is called BY THE
                # HANDLE (``serving._notification_surfaces``), i.e. from the
                # session's loop, while the runtime's loop registers and drops
                # clients in the same dict. A ``RuntimeError: dictionary changed
                # size during iteration`` is not a crash here — it is caught and
                # answered as "nobody can be notified", which silently costs a
                # parked approval its toast.
                for conn in list(self._clients.values())
            )
            else frozenset()
        )

    def watching_surfaces(self) -> frozenset[str]:
        """Which KINDS of surface have a HUMAN watching this session right now.

        Notification routing needs the kind, not the count: a question goes to
        whatever is actually watching, and only falls out to the OS when
        nothing is.

        **A ``daemon`` connection is NOT somebody watching.** ``"daemon"`` is
        the default kind for an auth frame with no ``client`` field, which is
        exactly what the mobile daemon's ADOPTION dial sends — and that dial
        covers every session on the machine and is held open permanently
        (`mobile/daemon.py::_dial`). Counting it meant that on any machine
        running ``lop mobile`` no parked approval ever sent a notification,
        the gate held ~283 MB for 24 h, and the model was told a human was
        watching (round 3, B1). ``process.py::_viewer_attached`` reads this
        same table and counts only ``"attach"`` for exactly this reason.

        **A PHONE THAT IS BEING LOOKED AT REGISTERS THROUGH ``watch``.**
        ``phone_watchers`` is incremented by the ``watch`` control op, which
        the daemon pushes when a session's SSE subscriber count crosses 0↔N
        (`mobile/daemon.py::notify_watch_transition`) — i.e. exactly when a
        person opens or closes the session on their phone. That is the signal
        production already produces, and reading anything else is how round 3
        traded B1's false positive for a false negative: the fix introduced a
        parallel `note_viewer_active` mechanism that NOTHING called, so a user
        reading the session on their phone got a desktop toast for a card
        already on their screen, and the model was told nobody could answer
        (round 4, R1/Q1).

        ``watch_supported`` guards the mixed-version case for us: it latches
        on the first ``watch``/``unwatch`` ever seen, so a daemon too old to
        send the op leaves it False and this reports no phone rather than
        inventing one. That matches the reaper's reading of the same pair.

        Deliberately the live connection table rather than a cached flag:
        surfaces come and go constantly, and a stale answer here means a
        notification delivered to a surface that has gone away.
        """
        watching = self._visible_attach_surfaces()
        if self.watch_supported and self.phone_watchers > 0:
            # Reported as ``viewer`` rather than ``daemon`` so a reader cannot
            # confuse "a relay is connected" (true of every session on a
            # machine running `lop mobile`) with "a person is looking".
            watching.add("viewer")
        return frozenset(watching)

    def _authority_admitted(self, frame: dict[str, Any], conn: _ClientConn) -> bool:
        """Whether this frame may reach an authority-INCREASING sink.

        True for every frame that is not in :data:`_AUTHORITY_OPS` — the
        overwhelming majority of traffic, and the set whose authorization really
        is the record key alone. For an increasing frame, the answer comes from
        :func:`local_operator.harness.approval.admit_increasing`, which consults
        BOTH sources: THIS CONNECTION's proof of the runtime's spawn capability,
        and a verified OPERATOR or DEVICE signature (issue #1310 revision 2; the
        run-scoped supervisor credential joins them in stage E).

        The connection is taken rather than reached for because both sources are
        bound to it: the client's nonce came in on the auth frame that created
        ``conn`` and the salt was minted for it, so a capability proof is only
        ever valid where it was produced — and a challenge is minted per
        connection for the same reason, so a signature harvested on one socket
        cannot be presented on another.

        The classification is by OP plus the fields that op carries, and it is
        deliberately NOT by the handle method or by the resulting value: a frame
        is judged before anything is dispatched, so a refused request cannot
        have had a partial effect on the way to being refused.
        """
        if frame.get("op") not in _AUTHORITY_OPS:
            return True
        authority = frame_authority(frame)
        if authority is None or authority == "ordinary":
            return True
        # TWO SOURCES, COMBINED IN THE STDLIB-ONLY MODULE (revision 2). The
        # runtime maps the frame to two facts and the PREDICATE answers: the
        # spawn capability's per-connection proof (the interactive console, which
        # must stay prompt-free), and a verdict on an operator/device signature
        # (the attached pane, the desktop backend for a session it did not start,
        # the CLI for a background-started run, and — stage D — the phone). The
        # crypto that produces the verdict stays in ``local_operator.operator
        # .verify``, which imports ``cryptography`` lazily; this method never
        # learns how a signature is checked.
        capability = request_proof_ok(
            supplied=frame.get("operator_cap"),
            held=self._operator_cap,
            client_nonce=conn.operator_nonce,
            server_salt=conn.operator_salt,
        )
        signature = self._operator_signature_verdict(frame, conn)
        admitted = admit_increasing(capability=capability, signature=signature)
        if not admitted:
            # Logged because the two refusals have different causes and only
            # one of them is an attack: a capability miss is a follower or a
            # model-authored child, a FALSE signature verdict is somebody
            # presenting a signature that did not hold.
            logger.info(
                "control: refused an authority-increasing %s (capability=%s signature=%s) on "
                "session %s",
                frame.get("op"),
                capability,
                signature,
                self._record.session_id,
            )
        return admitted

    def _operator_signature_verdict(self, frame: dict[str, Any], conn: _ClientConn) -> bool | None:
        """``True``/``False`` when a signature was offered, ``None`` when not.

        THE CHALLENGE IS CONSUMED HERE, before verification, and that is the
        replay defence rather than a detail: a popped challenge cannot be
        presented a second time, so a captured signature (and the ``operator_sig``
        that carries it) has exactly one use. Pop-then-fail also means a client
        whose signature was refused must ask for a NEW challenge, which it does
        on its next attempt — the alternative, verifying first and popping on
        success, would let a flood of replays re-verify forever and would let a
        race present one challenge twice.

        A missing challenge is a refusal rather than "not offered": a frame that
        CARRIES a signature is claiming authority, and a claim with no live
        challenge behind it is exactly what a replay looks like.
        """
        target = signature_target(frame)
        if target is None:
            return None
        supplied = frame.get("operator_sig")
        if supplied is None:
            return None
        action, request_id = target
        entry = conn.operator_challenges.pop((action, request_id), None)
        if entry is None:
            return False
        challenge, expires_at = entry
        if time.monotonic() > expires_at:
            return False
        # The runtime-wide count follows the CONSUME as well as the mint: a
        # challenge handed back here is spent, and leaving it in the aggregate
        # would let a subject that never signs slowly fill the runtime's budget
        # with dead entries it minted itself.
        self._live_challenges.pop(challenge, None)
        loaded = self._anchor_cache.get()
        anchor = loaded.anchor if loaded.usable else None
        device_spki = self._device_cert_point(frame.get("operator_cert"), anchor)
        return operator_verify.signature_verdict(
            action=action,
            session_id=self._record.session_id,
            request_id=request_id,
            challenge=challenge,
            signature_hex=supplied,
            operator_spki=anchor.spki if anchor is not None else None,
            operator_key_id=frame.get("operator_key_id") or "",
            operator_cert=frame.get("operator_cert"),
            device_spki=device_spki,
            now=int(time.time()),
        )

    def _device_cert_point(self, certificate: object, anchor: Any) -> bytes | None:
        """The device point behind a certificate, verified once per TTL.

        A SHORT TTL rather than a permanent cache, and lazily rather than at
        startup, for the two facts the design names: most sessions never see a
        device frame, and a certificate can be REVOKED (an edit to the root-owned
        anchor) — an unlimited cache would keep honouring a revoked phone for the
        lifetime of a session that can run for days. The cache is keyed by the
        certificate string, so a different certificate is always verified rather
        than matched against a stale answer.
        """
        if not isinstance(certificate, str) or not certificate:
            return None
        if anchor is None:
            return None
        cached = self._device_certs.get(certificate)
        now = time.monotonic()
        if cached is not None and cached[1] > now:
            # REVOCATION IS RE-CHECKED ON EVERY USE, even on a cache hit (agent
            # review round 6, R6-1 — the second half of it, and the half the
            # certificate TTL alone does not cover). What is cached is the
            # EXPENSIVE half: the signature and expiry verification, which cannot
            # change. Whether the certificate's device is REVOKED can, and it is
            # the operator's own action — so it is read from the anchor every time
            # rather than frozen for `_DEVICE_CERT_TTL_S`. Without this, a device
            # revoked while a runtime was running kept acting for up to five
            # minutes after the anchor re-read that was meant to stop it.
            point, _deadline, device_id = cached
            if point is None:
                return None
            return None if device_is_revoked(anchor, device_id) else point
        parsed = operator_verify.read_device_cert(certificate)
        point = operator_verify.verify_device_cert(
            certificate, operator_spki=anchor.spki, now=int(time.time()), parsed=parsed
        )
        device_id = parsed.device_id if parsed is not None else ""
        if point is not None:
            # A certificate that names a REVOKED device is refused here, at the
            # point its identity is known: ``verify_device_cert`` deliberately
            # knows nothing about revocation (it checks a signature and an
            # expiry), and the revocation list is a property of the anchor.
            if device_is_revoked(anchor, device_id):
                point = None
        self._device_certs[certificate] = (point, now + _DEVICE_CERT_TTL_S, device_id)
        if len(self._device_certs) > _MAX_CACHED_DEVICE_CERTS:
            # Bounded: a client can present a new certificate string on every
            # frame, and an unbounded cache keyed by attacker-chosen strings is a
            # memory leak with an on-demand trigger.
            self._device_certs.clear()
        return point

    def _connection_may_loosen(self, frame: dict[str, Any], conn: _ClientConn) -> bool | None:
        """Whether THIS connection has PROVEN it may loosen this session's gate.

        ``True``, or ``None`` for "it has not said", which the sentence builders
        read as the conservative branch. Never ``False``: a follower and a
        capable console must not be indistinguishable by accident, and the
        distinction that matters is "proved" vs "did not".

        WHY NOT ``_authority_admitted`` ON A SYNTHETIC FRAME. That predicate
        reads the proof off the ``frame`` it is handed, and a frame constructed
        here has none — so it answered a constant ``False`` and told a console
        that had just loosened the gate that loosening "has to come from the
        window that started it" (agent review round 3, R3-1 = UX U10 = QA Q6:
        measured on production objects, on both the desktop and the phone).
        Passing the REQUEST frame instead is not a fix either: the request is
        ordinary, so an ordinary op would answer "may loosen" for a follower.

        What CAN be verified here is the HANDSHAKE proof: HMAC over THIS
        connection's nonce and salt, computable only by a process holding the
        capability. A client that spawned this runtime has both; a follower, an
        impostor holding the rewritten record, and a relay forwarding someone
        else's frames do not — the proof is bound to the connection's own nonce
        and salt, so another connection's proof does not verify here.

        AND THE OPERATOR SOURCE (revision 2). A LOCAL connection to a runtime with
        a usable anchor may loosen too — it asks for a challenge, signs it, and
        the signature costs one presence gesture. That is precisely the capability
        this revision restores, so reporting ``None`` here would tell an attached
        pane that loosening has to come from the window that started the session
        while the pane is about to do it successfully.

        ``local`` was once the whole of the condition, and the paragraph that said so
        described stage D as unlanded (it was read by exactly the person reasoning
        about a phone's refusal, so it is corrected rather than left: UX round 6,
        U8). Stage D IS on this branch, and what widened is one call down: see
        ``_local_operator_available``, which answers ``True`` for a REMOTE
        connection whose device certificate verifies under the anchor — the phone,
        whose authority does not depend on who spawned the runtime.
        """
        if self._local_operator_available(frame, conn):
            return True
        if self._operator_cap is None:
            return None
        supplied = frame.get("operator_handshake")
        if not is_wire_hex(supplied) or not conn.operator_nonce or not conn.operator_salt:
            # Not proved: an unchanged client (the field is optional and
            # additive), a follower, or a relay. The conservative sentence is
            # the right answer for all three, and for a capable client on an
            # older build it is only cosmetically wrong until it updates.
            return None
        if not handshake_proof_ok(
            supplied=supplied,
            held=self._operator_cap,
            client_nonce=conn.operator_nonce,
            server_salt=conn.operator_salt,
        ):
            return None
        return True

    def _local_operator_available(self, frame: dict[str, Any], conn: _ClientConn) -> bool:
        """Whether THIS connection could carry out a loosening by signing.

        Reads the anchor through the runtime's own cache, so the answer is the
        same anchor the seam will verify against and cannot drift from it. The
        server's own capability is deliberately NOT consulted: it is the
        SPAWNER's proof, and the point of the revision is that a connection
        without it can still hold authority.

        THREE ANSWERS, and the third is what stage D widened — this predicate is
        the single line the earlier revision named as the one that would:

        * LOCAL, on a host with a usable anchor: yes. One presence gesture, and
          an attached pane or the desktop backend is precisely the capability
          this revision restores.
        * REMOTE without a paired device: no. The relay cannot mint a signature,
          and answering yes would put ``/approvals auto`` in a report on a
          surface that cannot carry it out — the dead end UX round 2 removed.
        * REMOTE with a device certificate that VERIFIES under the anchor: yes.
          That is the phone, and it is the whole point of the stage: its
          authority does not depend on who spawned the runtime.
        """
        del frame
        if conn.locality == "local":
            return self._anchor_cache.get().usable
        return self._paired_device_can_sign(conn)

    def _paired_device_can_sign(self, conn: _ClientConn) -> bool:
        """Whether this connection declared a device certificate the anchor vouches for.

        Goes through ``_device_cert_point`` rather than a verification of its own,
        and that is the load-bearing part: that helper is where the certificate is
        checked under the anchored key, where the anchor's REVOCATION list is
        consulted, and where the TTL cache lives. A second verification path here
        could answer "yes" for a phone the seam would then refuse — the same class
        of drift the refusal copy's single-sentence rule exists to prevent.
        """
        loaded = self._anchor_cache.get()
        anchor = loaded.anchor if loaded.usable else None
        if anchor is None or not conn.device_certificate:
            return False
        return self._device_cert_point(conn.device_certificate, anchor) is not None

    def _expire_challenges(self, conn: _ClientConn, now: float) -> None:
        """Drop this connection's spent-by-time challenges before minting another.

        Pruning on MINT rather than on a timer: there is no background task here
        on purpose (a runtime that woke on a timer to sweep a per-connection map
        would be paying for every idle viewer), and the failure mode pruning
        prevents — a client that asks and never signs — is bounded by
        ``_MAX_CHALLENGES_PER_CONN`` in any case.

        The runtime-wide map is pruned on the same two events (a mint and a
        consume), and it needs no timer for the same reason: every entry it holds
        is one of the entries in some connection's map, so anything it can forget
        has already been forgotten here, and any VALUABLE entry has a deadline.

        THE RE-INSERTION BELOW CANNOT GROW THAT MAP (agent review round 7, N-2),
        which is worth the half-sentence because it reads as though it could: it
        refreshes deadlines only for entries ALREADY COUNTED AT MINT — a challenge
        in this connection's map was inserted into the runtime map by the mint that
        created it, under the ``_MAX_LIVE_CHALLENGES`` check — and the second loop
        drops the expired ones. So the map's size is what the bound says it is,
        whether or not this runs.
        """
        stale = [key for key, (_, deadline) in conn.operator_challenges.items() if deadline < now]
        for key in stale:
            conn.operator_challenges.pop(key, None)
        for challenge, deadline in [
            (value[0], value[1]) for value in conn.operator_challenges.values()
        ]:
            self._live_challenges[challenge] = deadline
        for challenge in [
            challenge for challenge, deadline in self._live_challenges.items() if deadline < now
        ]:
            self._live_challenges.pop(challenge, None)

    async def _on_request(self, frame: dict[str, Any], conn: _ClientConn) -> None:
        # A FRAME THAT IS NOT AN OBJECT MUST NOT REACH `.get`, and the guard is
        # HERE rather than at the reader's parse because this is the line that
        # actually dereferences it: `op`/`req` are read BEFORE the `try` below,
        # so an `AttributeError` from them escapes this method entirely, sails
        # past the reader loop's `ConnectionResetError`/`BrokenPipeError`
        # handler, and reaches the `finally` that drops the client — killing the
        # session for a junk frame, which is the exact death the oversized-line
        # fix exists to prevent (review round 1, MAJOR-2).
        #
        # `json.loads` is the source: it succeeds on any JSON SCALAR, so a
        # discarded oversized line whose surviving tail happens to read `12345`,
        # `null`, `true`, `"str"` or `[1,2]` parses cleanly and arrives here as
        # an int/None/bool/str/list. That tail is now reachable in the ordinary
        # course of events precisely because the reader loop survives an
        # oversized frame instead of dying on it.
        #
        # Guarding at this seam rather than at the parse also covers the other
        # callers — a future dispatch route, and the tests that drive this
        # method directly — so the "discard and keep the connection" promise the
        # reader loop's log line makes is total rather than route-specific
        # (review round 1, NIT-2).
        #
        # DROPPED, not answered: an error reply is keyed by `req`, and a frame
        # that is not an object carries no `req` to answer on. One DEBUG line,
        # because the sender is a broken or ancient peer rather than the
        # operator, and the reader loop already logged the oversized frame that
        # produced the tail at `error`.
        if not isinstance(frame, dict):
            logger.debug(
                "session runtime: ignoring a non-object frame (%s) from %s client %s",
                type(frame).__name__,
                conn.kind,
                conn.writer.get_extra_info("peername"),
            )
            return
        op = str(frame.get("op") or "")
        req = frame.get("req")
        try:
            # A CONNECTION THAT IS STILL BINDING IS REACHABLE BUT NOT YET
            # AUTHORITATIVE. ``_on_connection`` starts this connection's reader
            # loop before the canonical ``frontend_sync`` is on the wire (the
            # why is there: the bind takes the cross-thread hop into a busy app
            # loop, and running it inline used to leave the socket mute, so a
            # user could neither steer nor stop the session they were looking
            # at). The price of that reachability is this gate: while the sync
            # is pending, the connection may run :data:`_SYNC_PRIORITY_OPS` —
            # health, and the four ways to regain control of a turn (``stop``,
            # ``abort``, ``steer`` and ``cancel``, which joined the set in review
            # round 1) — plus the connection-local bookkeeping in
            # :data:`_SYNC_LOCAL_OPS` that the dial path itself sends before it
            # awaits the sync.
            #
            # Everything else is REFUSED, through the ordinary error-frame path
            # below rather than by running it or silently dropping it: a
            # ``prompt`` or a ``slash`` admitted here would let a client act on a
            # connection that has not yet been told what state it is acting on.
            # Refusing is also not a dead end — the flag clears the moment the
            # sync settles, so the client's retry succeeds.
            if (
                conn.frontend_sync_pending
                and op not in _SYNC_PRIORITY_OPS
                and op not in _SYNC_LOCAL_OPS
            ):
                raise ValueError(
                    "this viewer is still connecting to the session; the request was not "
                    "run — retry once the interface has connected"
                )

            # THE TWO REFUSALS, AND WHY THE READINESS ONE RUNS FIRST (rebase onto
            # main, 2026-09-19). Upstream added the sync-pending gate above at the
            # same seam this branch added the authority check to. They answer
            # different questions: that one is "is this connection authoritative
            # yet at all?", this one is "may THIS caller remove the gate?". Running
            # the readiness gate first keeps its semantics intact — a follower
            # that has not received the sync gets the retryable connect copy
            # rather than the authority copy, which would be wrong advice for an
            # op it will be allowed to run a moment later. The authority check is
            # NOT skipped for what the priority and connection-local sets admit:
            # ``approval_answer`` is in neither today, and if a future base puts it
            # there, the check below still sees it.
            #
            # THE ONE SEAM WHERE THE GATE'S AUTHORITY IS DECIDED (issue #1310).
            #
            # `control_key` — published 0600 in the session record — is the
            # whole authorization story for ORDINARY operations and stays that
            # way. An authority-INCREASING one additionally demands the
            # per-session operator capability, which exists only in the memory
            # of the process that started this runtime and in the console that
            # typed the command. A model-authored `bash` call runs as this same
            # uid, can read the record, and can dial this loopback port; it
            # cannot hold a value that was never written anywhere it can read.
            #
            # HERE rather than at each sink, because every route in the tree —
            # the daemon's HTTP command surface, the phone relay, a peer send,
            # a follower terminal, the CLI — arrives at the handle through this
            # method. A second dispatch route added later is covered by
            # construction, and `tests/unit/session/runtime/
            # test_approval_authority_seam.py` fails if one appears that reaches
            # a sink without being classified below.
            #
            # The refusal is raised rather than answered inline so it reuses the
            # existing `{"op": "error"}` reply the branches below already
            # produce: one shape for the client to surface, and the copy names
            # the one-step remedies (see OPERATOR_AUTHORITY_REQUIRED_NOTICE).
            #
            # A TYPED refusal, not a bare `ValueError`, and the code is what
            # makes it survivable: every route that carries a control request
            # (the desktop command surface, the desktop card route, the relay,
            # the attach screen) can name this outcome instead of guessing from
            # the message, and the copy reaches the operator verbatim rather
            # than being reported as "the runtime is unreachable" or "the
            # question expired" (agent review round 1 R1-2 = design D1 = UX U4 =
            # QA Q1, from four independent rounds on the same defect).
            if not self._authority_admitted(frame, conn):
                logger.warning(
                    "session runtime: refused an authority-increasing request "
                    "(op %r) from %s at %s",
                    op,
                    conn.kind,
                    conn.writer.get_extra_info("peername"),
                )
                from local_operator.session.errors import (
                    OperatorAuthorityRequired,
                    OperatorAuthorityUnconfigured,
                )

                # WHICH REFUSAL, from the op rather than from prose: a card
                # answer is refused as a card (the question is still parked, and
                # a deny works from here), a slash is refused as a command. The
                # far side rebuilds the same sentence from this token, so the
                # copy never travels as text (UX round 2, U8).
                #
                # AND WHICH HOST, from the anchor rather than from a guess (UX
                # round 6, U1/U2): with no USABLE anchor the two named remedies
                # cannot run, so the refusal has to name the command that lands
                # one. ``usable`` is exactly the predicate the seam above used to
                # decide this caller could not be admitted, read from the same
                # cached load, so the sentence and the decision cannot disagree.
                anchored = self._anchor_cache.get().usable
                raise (
                    OperatorAuthorityRequired(trigger=op)
                    if anchored
                    else OperatorAuthorityUnconfigured(trigger=op)
                )
            # Attach clients are followers: rebinding the owner's conversation
            # from a follower terminal surprises the user AT THAT TERMINAL's
            # owner. The error frame is the reply — the attach screen surfaces
            # it like any other rejected op. The daemon keeps both ops (the
            # phone's resume button rides them).
            if conn.kind == "attach" and op in ("new_conversation", "resume_session"):
                raise ValueError(
                    "attached front ends cannot rebind the session; detach and /resume instead"
                )
            if op in ("watch_job", "unwatch_job"):
                # Trajectory subscription for ONE child page, per connection.
                # Handled here rather than in ``_dispatch`` because it mutates
                # this connection's own state and never touches the session:
                # the dispatcher deliberately has no ``conn``.
                job_id = str(frame.get("job_id") or "")
                if not job_id:
                    raise ValueError("job_id must be a non-empty string")
                if op == "watch_job":
                    conn.watched_jobs.add(job_id)
                else:
                    conn.watched_jobs.discard(job_id)
                detail = f"watching {len(conn.watched_jobs)} job(s)"
            elif op == "desktop_watch":
                if (
                    conn.kind != "attach"
                    or conn.surface != "desktop"
                    or id(conn.writer) not in self._clients
                ):
                    raise ValueError("desktop visibility requires a live desktop attach connection")
                visible, can_notify = frame.get("visible"), frame.get("can_notify")
                if type(visible) is not bool or type(can_notify) is not bool:
                    raise ValueError("desktop visibility fields must be booleans")
                conn.desktop_visible = visible
                conn.desktop_can_notify = can_notify
                conn.desktop_seen = time.monotonic()
                self._republish_detached()
                detail = "desktop lease renewed"
            elif op == "viewer_watch":
                # A MULTIPLEXING TERMINAL SAYS WHETHER IT IS STILL SHOWING US.
                #
                # The desktop leg above answers the same question with window
                # state; a terminal had no way to answer it at all, so a viewer
                # that switched away kept its retained connection counted as a
                # person reading this session and every parked gate behind it
                # went silent.
                #
                # No lease and no timestamp, unlike ``desktop_watch``: this is
                # EDGE-TRIGGERED state a viewer sets when it switches, not a
                # liveness claim that has to expire. Socket close remains the
                # liveness signal, exactly as before.
                if conn.kind != "attach" or id(conn.writer) not in self._clients:
                    raise ValueError("viewer watch requires a live attach connection")
                displaying = frame.get("displaying")
                if type(displaying) is not bool:
                    raise ValueError("viewer watch field must be a boolean")
                conn.terminal_displaying = displaying
                # The 1<->0 transition this can cause is exactly the one
                # ``reannounce_pending`` exists for: a gate that opened while
                # somebody was watching, whose watcher has now looked away.
                self._republish_detached()
                detail = f"viewer {'watching' if displaying else 'away'}"
            elif op in ("watch", "unwatch"):
                # The reaper's phone-watcher signal (§2.8). watch_supported
                # latches on the FIRST op seen so a mixed-version child never
                # mistakes silence for zero watchers. Deliberately OUTSIDE the
                # registry guard below: it is a version signal, not a count,
                # and a frame proves the daemon speaks the op whenever it
                # arrived.
                self.watch_supported = True
                # ONLY A REGISTERED CONNECTION MAY MOVE THE COUNT. The reader
                # loop is strictly serial — `readline()` then `await
                # _on_request(...)` — so anything the daemon sent before it
                # died is still in the socket buffer while an op is parked.
                # `_drop_client` closes the WRITER, but the `StreamReader`
                # keeps yielding those buffered lines, so this runs on a
                # connection already evicted from the registry. A dying
                # daemon's `unwatch` (pushed from the SSE generator's
                # `finally`) then wiped the REPLACEMENT daemon's live count
                # (round 7, R8).
                #
                # Gated on `conn.kind` too: only the daemon's count is ever
                # cleared (`_drop_client` zeroes for `kind == "daemon"`), so an
                # attach client's `watch` would increment something nothing can
                # clear — a permanent phantom viewer (round 7, R9).
                if conn.kind != "daemon" or id(conn.writer) not in self._clients:
                    pass
                elif op == "watch":
                    self.phone_watchers += 1
                else:
                    self.phone_watchers = max(0, self.phone_watchers - 1)
                detail = f"watchers: {self.phone_watchers}"
            elif op == "retire_if_pristine":
                # The counterpart to the eager engage: a viewer starts a
                # runtime at MOUNT now (so the band can show the model, the
                # MCP roster and the context reading immediately), and this is
                # how that runtime is handed back when the viewer leaves
                # without ever using it — the user quit the TUI, or `/resume`d
                # onto a different session.
                #
                # Handled HERE rather than in ``_dispatch`` for the same reason
                # ``watch`` is: the decision reads this connection's identity,
                # and the dispatcher deliberately has no ``conn``.
                #
                # CONDITIONAL AT THE RUNTIME, never at the caller, and that is
                # the whole design. The viewer cannot safely decide this:
                # between its own "looks empty" read and the stop arriving, a
                # wake can fire, a peer's `lop send` can open a turn, or a
                # second terminal can attach and start typing. Asking the
                # runtime to judge ITSELF shrinks that window to the runtime's
                # own loop — and what remains of it is closed in
                # ``_retire_if_pristine``: the ONE await between the decision
                # and the stop is the ``stopping`` broadcast, the pristine
                # check is repeated after it, and anything that still lands
                # inside is aborted by the stop the same way ``stop`` and
                # SIGTERM abort a turn (review round 1, MINOR-3).
                #
                # Two independent reasons to refuse, and both matter:
                #
                # * STILL OBSERVED — another attach client is connected. This
                #   is the long-forgotten-TUI case: a session left open for
                #   hours stays pristine, and if a second terminal is watching
                #   it (or a peer is about to send it an instruction) the
                #   runtime behind that terminal must not vanish because THIS
                #   viewer quit. The leaving viewer is excluded from the count
                #   — its own connection is still registered while this op is
                #   dispatched, so counting it would make every retirement
                #   refuse itself.
                # * NOT PRISTINE — something durable exists (a transcript row,
                #   an armed wake, live work). ``is_busy`` is not enough here;
                #   see ``ServingSessionHandle.is_pristine`` for why a finished
                #   conversation is idle but emphatically not disposable.
                #
                # Refusing is always safe: the runtime stays up and the
                # ordinary residency drain (``process._should_exit``) reaps it
                # seconds later once nobody is attached. This op only makes
                # that immediate for the one case where waiting is pointless.
                observers = self._other_observers(conn)
                if observers > 0:
                    detail = f"kept: {observers} viewer(s) still attached"
                else:
                    detail = await self._retire_if_pristine(leaving=conn)
            elif op == "retire_now":
                # ``/move``: the viewer changed the directory this session
                # works in, and the cwd is baked in at spawn
                # (``LOP_MOBILE_CHILD_CWD``), so the only way to honour it is
                # to retire and let the viewer engage a successor at the new
                # path.
                #
                # A SIBLING of ``refresh_if_idle`` rather than a reuse of it,
                # because the two differ in exactly one term and it is the
                # gating one: a refresh is owed only when the build on disk has
                # moved, whereas a move is owed whenever the user asked for it.
                # Everything downstream is deliberately identical — the same
                # idle predicate, the same ``retiring`` announcement, the same
                # re-check after it — so a moved runtime and a refreshed one
                # leave by one path and a viewer needs no new frame to
                # understand either.
                #
                # Handled HERE rather than in ``_dispatch`` for the same reason
                # its two siblings are: these are lifecycle ops that must not
                # trigger the post-ack refresh (the exemption list below), and
                # a dispatcher that has no ``conn`` cannot make that call.
                # EXCLUSIVITY (review R3). A move is honoured by retiring,
                # and every facade attached at that moment engages its own
                # successor from its OWN ``_cwd`` -- so a sibling desktop
                # window or terminal that never learned the new target asks
                # for a runtime in the OLD directory, and whichever engage
                # wins decides where the session actually works. The bounded
                # answer is to refuse the move while another ACTUAL attach is
                # registered rather than ship a cross-facade propagation
                # protocol this release does not have.
                exclusive = bool(frame.get("exclusive"))
                if exclusive and EXCLUSIVE_MOVE_CAPABILITY not in self._record.capabilities:
                    # Fail CLOSED. An old owner ignores the unknown field and
                    # would retire unguarded, so the desktop only sends it
                    # after reading this capability off the record; reaching
                    # here means a caller sent it blind, and the honest answer
                    # is a refusal, never a legacy retire.
                    detail = "kept: this runtime cannot move exclusively; /reload first"
                elif exclusive:
                    # Reserved BEFORE the first await, on this loop: the
                    # check-and-reserve is one synchronous step so a viewer
                    # attaching afterwards cannot slip behind the count and
                    # be missed (``_on_connection`` honours the fence).
                    self._exclusive_move_fence = conn
                    committed = False
                    try:
                        observers = self._other_observers(conn)
                        if observers > 0:
                            detail = (
                                "kept: This session is open in another terminal or attached "
                                "client. Disconnect that client, then move again."
                            )
                        else:
                            detail = await self._retire_for("moved", exclusive_owner=conn)
                            committed = detail == "retiring"
                    finally:
                        # CLEARED ON EVERY DEFINITE REFUSAL, and the funnel is
                        # this ``finally`` rather than the two exits above it:
                        # ``_retire_for`` has refusal returns of its own (not
                        # idle, no graceful stop, work arriving before the
                        # latch), and every one of them is a runtime that is
                        # STAYING ALIVE and must therefore admit viewers again.
                        # Leaving one of those paths latched would refuse every
                        # later attach for the rest of this runtime's life — an
                        # outage caused by a refused move. It is RETAINED only
                        # once retirement has really committed, because from
                        # then on a facade admitted here would engage the
                        # successor from its own cwd after the move's owner is
                        # gone. ``_retirement_committed`` is the latch's own
                        # record of that, which is how a ``request_stop`` that
                        # raises AFTER committing keeps the fence (N6).
                        if not committed and not self._retirement_committed:
                            self._exclusive_move_fence = None
                else:
                    detail = await self._retire_for("moved")
            elif op == "refresh_if_idle":
                # The viewer-side belt for the runtime's own self-refresh
                # (design-runtime-autorefresh §3.3): a `lop --resume` in the
                # seconds after `lop-update`, before the reaper has noticed,
                # binds to a stale idle owner. Rather than paint a notice and
                # wait ~15-40 s for the reaper, the viewer asks the runtime to
                # retire NOW if it is idle, and re-engages a fresh one on the
                # ``retiring`` frame. Same predicate as the reaper
                # (``ServingSessionHandle.may_refresh``), same announce, same
                # re-check after it; no stagger, because one viewer asking for
                # one runtime is not the sixteen-at-once storm the stagger
                # bounds. Answers ``retiring`` or ``kept: <reason>``; the
                # viewer paints its busy notice only on a ``kept: busy``.
                #
                # Handled here rather than in ``_dispatch`` for symmetry with
                # ``retire_if_pristine``: both are lifecycle ops that must not
                # trigger the post-ack refresh (the exemption list below).
                detail = await self._refresh_if_idle()
            elif op in ("event_mute", "event_unmute"):
                # Raw-event interest, per connection, for a viewer that stops
                # painting this session while parked (see
                # ``EVENT_MUTE_CAPABILITY``). Handled here rather than in
                # ``_dispatch`` for the same reason ``watch_job`` is: it
                # mutates this connection's own relay state and never touches
                # the session, and the dispatcher deliberately has no ``conn``.
                #
                # Delta-grade frames ONLY, and the set is deliberately the
                # same one the viewer's parked ``EventController`` discards
                # app-side: muting anything a client still needs for state
                # (turn boundaries, gates, notices) would change what a
                # reveal can reconstruct, and the unmute path rebuilds live
                # text from history plus the canonical seed either way.
                # Idempotent: re-asserting the current state is how a
                # reconnected parked viewer gets its mute back.
                #
                # Attach-only, like ``desktop_watch``: the relay this mutes is
                # never sent to daemon connections, so a daemon sending it is
                # a client bug and the error frame is the honest reply.
                if conn.kind != "attach":
                    raise ValueError("event muting requires an attach connection")
                conn.events_muted = op == "event_mute"
                detail = "delta-grade events muted" if conn.events_muted else "events resumed"
            elif op == "operator_challenge":
                # THE ORDINARY OP THAT LETS A SURFACE THAT IS NOT THE SOCKET PEER
                # SIGN (revision 2, §2.3). Ordinary by construction — it grants
                # nothing on its own; the signature it is used to produce is what
                # carries authority — so it needs no proof of its own and rides
                # the record key like every other control op.
                #
                # Handled HERE rather than in ``_dispatch`` because the binding
                # is per CONNECTION and the dispatcher deliberately has no
                # ``conn`` (the same reason ``watch_job`` and ``event_mute`` are
                # here). Two things are bound beyond that: the action, which
                # chooses what the signature will be accepted FOR, and the
                # request id, which for an ``approve`` is the card and for a
                # loosening is whatever the client chose. All four are baked into
                # the signed message, so a challenge cannot be moved between them.
                action = str(frame.get("action") or "")
                if action not in ("loosen", "approve"):
                    raise ValueError("action must be 'loosen' or 'approve'")
                request_id = frame.get("request_id", "")
                if not isinstance(request_id, str):
                    raise ValueError("request_id must be a string")
                now = time.monotonic()
                self._expire_challenges(conn, now)
                if len(conn.operator_challenges) >= _MAX_CHALLENGES_PER_CONN:
                    raise ValueError("too many unspent operator challenges on this connection")
                if len(self._live_challenges) >= _MAX_LIVE_CHALLENGES:
                    # THE AGGREGATE BOUND (agent review round 6, R6-4). The
                    # per-connection maximum alone bounds a socket, and the
                    # subject this predicate gates can dial as many as it likes;
                    # this is the term that makes the volume bounded rather than
                    # merely denied. A real surface raises one prompt per human
                    # gesture, so reaching this is the abuse it exists for.
                    raise ValueError("too many unspent operator challenges on this runtime")
                challenge = secrets.token_hex(32)
                conn.operator_challenges[(action, request_id)] = (
                    challenge,
                    now + _CHALLENGE_TTL_S,
                )
                self._live_challenges[challenge] = now + _CHALLENGE_TTL_S
                await self._send_to(
                    conn,
                    {
                        # AN ``ack`` FRAME, NOT AN ``operator_challenge`` ONE, and
                        # that is a protocol requirement rather than a style
                        # choice: ``AttachClient``'s reader routes replies by
                        # ``req`` but only admits the op names ``ack``, ``error``
                        # and ``result`` (``attach_client.py``), so a reply
                        # carrying a NEW op would fall through every branch and
                        # tear down the whole connection — taking the caller's
                        # in-flight request with it, which is exactly the "old
                        # front end must keep working" guarantee the additive
                        # design exists to keep. The two fields are additive on
                        # an existing frame shape, so a client built before the
                        # op existed (which never asks for a challenge) is
                        # unaffected, and a client that does ask reads them off
                        # the ack.
                        "op": "ack",
                        "req": req,
                        "challenge": challenge,
                        "expires_s": int(_CHALLENGE_TTL_S),
                    },
                )
                return
            elif op in _PAYLOAD_OPS:
                # Structured-answer ops reply with a ``result`` frame whose
                # ``data`` the invoker renders locally (a slash command's typed
                # outcome, a cancel's authoritative count) rather than a
                # one-line receipt that would paint in the owner's transcript.
                data = await self._dispatch_payload(
                    op,
                    frame,
                    conn.locality,
                    conn.slash_consumers,
                    audit_capable=conn.audit_history,
                    # Whether THIS connection PROVED it may loosen the gate,
                    # or ``None`` for "it has not said" (design round 2 D10, UX
                    # round 2 U9; corrected in agent review round 3, R3-1): the
                    # reports a routed command returns must not offer a command
                    # this connection would be refused, and must not deny one it
                    # could carry.
                    may_loosen=self._connection_may_loosen(frame, conn),
                )
                await self._send_to(conn, {"op": "result", "req": req, "data": data})
                await self._handle.refresh()
                await self._push()
                return
            else:
                duplicate = await self._already_admitted(op, frame)
                if duplicate:
                    # A retry of an errand this transcript already owns (a
                    # sender that crashed after the row was durable, a wake
                    # re-fired by a restarted supervisor). Acked, not executed:
                    # the caller's outcome is "delivered", which is true, and
                    # nothing is appended twice. See
                    # ``ServingSessionHandle.has_admitted_command``.
                    detail = "already admitted"
                    extra: dict[str, Any] = {}
                else:
                    outcome = await self._dispatch(op, frame)
                    # An op may answer with state as well as with a sentence
                    # (``AckDetail``): the extra fields ride THIS frame rather
                    # than a follow-up push, because the caller of the receipt op
                    # has to verify what that op did and its own projection is
                    # delivered by a different writer.
                    detail, extra = (
                        (outcome.detail, {"attention": outcome.attention})
                        if isinstance(outcome, AckDetail)
                        else (outcome, {})
                    )
                await self._send_to(
                    conn,
                    {
                        "op": "ack",
                        "req": req,
                        "detail": detail,
                        "duplicate": duplicate,
                        **extra,
                    },
                )
                if not duplicate:
                    await self._handle.refresh()
                    await self._push()
                return
            await self._send_to(conn, {"op": "ack", "req": req, "detail": detail})
            # Mutations change the projection; push what every front end
            # should see. ``stop`` is exempt: its whole job is to END the
            # session, so a post-ack refresh would re-read a host that is
            # mid-dispose (the TUI's handle raises "session is still
            # starting" the moment its session reference drops) — the ack is
            # the reply and the ladder's exit-wait is the confirmation.
            #
            # ``retire_if_pristine`` is exempt for the same reason WHEN it
            # retired. It is listed unconditionally rather than by outcome
            # because the refuse path has nothing to push either: a refusal
            # changed no state at all, so the skipped refresh costs nothing
            # and the retire path is spared the mid-dispose read.
            if op not in (
                "watch",
                "unwatch",
                "watch_job",
                "unwatch_job",
                "stop",
                "retire_if_pristine",
                "refresh_if_idle",
                # Same exemption, same reason: it either retired (nothing to
                # push, and the handle is mid-dispose) or refused (nothing
                # changed, so there is nothing to push).
                "retire_now",
                # Connection-local relay toggles, like ``watch`` above: they
                # mutate only this connection's OWN event interest, never the
                # session, so a refresh has nothing new to see and there is
                # no changed state to push.
                "event_mute",
                "event_unmute",
            ):
                await self._handle.refresh()
                await self._push()
        except Exception as exc:  # noqa: BLE001 — the error IS the reply
            from local_operator.session.errors import (
                AttachmentUnavailable,
                OperatorAuthorityRequired,
                ProfileRegistryUnavailable,
                RuntimeRetiring,
            )

            frame = {"op": "error", "req": req, "message": str(exc)[:400]}
            if isinstance(
                exc,
                (
                    AttachmentUnavailable,
                    OperatorAuthorityRequired,
                    ProfileRegistryUnavailable,
                    RuntimeRetiring,
                ),
            ):
                # Category, not arbitrary prose, certifies this as a repairable
                # admission rejection to older/newer attach clients alike.
                frame["error_code"] = exc.code
                if isinstance(exc, OperatorAuthorityRequired) and exc.trigger:
                    # A token from a closed set, never text: the far side picks
                    # the sentence that matches the frame it refused (a refused
                    # command vs a refused card).
                    frame["error_trigger"] = exc.trigger
            if isinstance(exc, RuntimeRetiring) and exc.trigger:
                # WHICH DEPARTURE, as one of the enumerated tokens — the same
                # shape as ``error_count`` below, and for the same reason: the
                # far side rebuilds the sentence from the category, so the only
                # thing that may ride along is a value from a closed set. An
                # older client drops the unknown field and rebuilds the sentence
                # it has always rebuilt, which is what such a client's own
                # runtime means (design round 4, D10).
                frame["error_trigger"] = exc.trigger
            if isinstance(exc, ProfileRegistryUnavailable) and exc.count is not None:
                # The count rides as its own INTEGER field so the attach client
                # can rebuild the actionable wording locally. Without it the
                # client reconstructs from the bare code and renders the
                # countless sentence, which is the unactionable message this
                # detail exists to replace -- and ``/team`` attach is one of
                # the two surfaces that motivated it. An integer carries no
                # path, host or identity, so it does not widen what
                # ``session/errors.py`` admits across this boundary.
                frame["error_count"] = exc.count
            await self._send_to(conn, frame)
            await self._push()

    def _other_observers(self, leaving: _ClientConn) -> int:
        """Attach clients other than the one asking to leave.

        The residency predicate's term 3 with the leaving viewer excluded:
        its own connection is still registered while its op is dispatched,
        so counting it would make every retirement refuse itself.
        """
        return sum(
            1 for other in self._clients.values() if other.kind == "attach" and other is not leaving
        )

    async def _retire_if_pristine(self, *, leaving: _ClientConn) -> str:
        """Stop this runtime iff nothing has ever happened in its session.

        The caller (``_on_request``) has already established that no OTHER
        viewer is attached; this half answers "is there anything here worth
        keeping". Split out so the observer check reads next to the connection
        it inspects while the disposal ordering stays beside the ``stop`` op it
        mirrors. Both checks are repeated after the one await below.

        Every failure path RETURNS rather than raises, and every one of them
        keeps the runtime alive. A wrong "retire" ends a session the user may
        still want; a wrong "keep" costs one idle process for the few seconds
        the residency drain needs to notice nobody is attached. Those are not
        symmetric, so the uncertain answer is always "keep".
        """
        h = self._handle
        pristine = getattr(h, "is_pristine", None)
        if not callable(pristine):
            # An older runtime, or a reduced test handle, that never grew the
            # probe. Unknown state is not an invitation to stop it.
            return "kept: this runtime cannot judge itself pristine"
        try:
            # HOPPED, all three of these: ``is_pristine`` and ``request_stop``
            # read and mutate the session (`_note_deliberate_stop`,
            # `_deny_pending_gates`, then the host hook), and this method runs on
            # the RUNTIME's loop. For a handle that publishes no loop (the TUI's,
            # a reduced test handle) the helper runs the call inline exactly as
            # it did before.
            if not await self._handle_call_on_session_loop(pristine):
                return "kept: session has work or history"
        except Exception as exc:  # noqa: BLE001 — uncertainty keeps the runtime
            logger.debug("pristine probe failed; keeping runtime", exc_info=True)
            return f"kept: pristine probe failed ({exc})"
        request_stop = getattr(h, "request_stop", None)
        if not callable(request_stop):
            return "kept: this runtime cannot stop itself gracefully"
        # The same announcement the ``stop`` op makes, for the same reason: a
        # follower must be able to tell a deliberate stop from owner death.
        #
        # This broadcast is the one await between the decision and the stop,
        # and it is a real one: ``_broadcast`` drains each client's writer
        # under its send lock (1 s cap), and the loop can run another
        # connection's reader meanwhile — a daemon-class ``peer_message`` or
        # ``prompt`` can open a turn in that gap. So the probe is asked AGAIN
        # after it, from the same loop step as ``request_stop()``. A turn that
        # starts between this re-check and the stop is aborted by the stop
        # itself (``ServingSessionHandle.dispose`` aborts in-flight work and
        # flushes the transcript), which is exactly what a ``stop`` op or a
        # SIGTERM racing a turn already does; the message is not lost, it is
        # persisted and the sender's next engage starts a fresh runtime.
        await self._broadcast({"op": "stopping", "session_id": self._record.session_id})
        # The observer count is repeated too: a viewer that attached DURING
        # the broadcast was not on the recipient list, so it never heard
        # `stopping` and would read the exit as owner death — and it is a
        # user who just opened this session (review round 2, MINOR-2).
        late = self._other_observers(leaving)
        if late > 0:
            return f"kept: {late} viewer(s) attached while stopping was announced"
        # ONE HOP, NOT TWO, and that is the point of the pair below rather than
        # tidiness: the re-check and the stop have to happen in the same
        # synchronous step on the session's loop, or a turn admitted between two
        # hops is stopped without having been seen. The helper awaits a single
        # callable, so the pair goes over together and the window the comment
        # above describes stays exactly as wide as it was written to be.

        def _recheck_and_stop() -> Any:
            if not pristine():
                # Refusing AFTER announcing is safe: ``stopping`` only latches
                # the disconnect REASON in an attach client (attach_client.py),
                # and does nothing unless the socket then closes. The only
                # attach client here is the leaving viewer (the caller counted
                # the others and found none), and it is about to close its own
                # socket anyway.
                return _WORK_ARRIVED
            return request_stop()

        try:
            result = await self._handle_call_on_session_loop(_recheck_and_stop)
        except Exception as exc:  # noqa: BLE001
            logger.debug("pristine re-check failed; keeping runtime", exc_info=True)
            return f"kept: pristine probe failed ({exc})"
        if result is _WORK_ARRIVED:
            return "kept: work arrived while stopping was announced"
        return "retired"

    async def _refresh_if_idle(self) -> str:
        """Retire this runtime for the build on disk iff it is idle and stale.

        The viewer-driven twin of the reaper's ``process._refresh_for``, with
        the stagger removed (see the dispatch comment). Every failure path
        RETURNS ``kept: …`` rather than raising, and every one keeps the
        runtime alive — the same asymmetry ``_retire_if_pristine`` records: a
        wrong "retire" aborts nothing (the predicate is idle by construction)
        but costs the user a cold start they did not need, a wrong "keep"
        costs the reaper's next check.

        TWO ANSWERS ARE MORE PRECISE THAN THE QUESTION THEY REPLACE, because
        the caller reads this answer as a STATE rather than as a verdict:

        * A runtime that is ALREADY DRAINING its own exit says so instead of
          reporting the ``kept: busy`` its work in flight would otherwise
          produce. Those are different facts — "still working, moves when it
          ends" versus "leaving whatever you do next" — and only the second
          is true of a signalled runtime inside ``SIGNAL_DRAIN_S``.
        * ``kept: build on disk matches`` is NOT returned for an install that
          has moved but not settled. This is the answer ``lop refresh`` needs
          most precisely: its own documentation says its first run is
          ``lop-update``, so the operator calls it INSIDE the settle window,
          and calling that "already current" (with a zero exit status) tells a
          rotating script the fleet is done when every member of it is about
          to be retired (D1/M2, PR #1141). ``_build_changed`` deliberately
          folds "same stamp" and "not settled yet" into ``None`` — it answers
          "may I act" — so ``pending_build`` asks the settle question
          separately and the two cases answer differently.
        """
        from local_operator import buildwatch
        from local_operator.session.runtime import process as process_mod

        if self._leaving:
            return "kept: already leaving"
        newer = process_mod._build_changed(self._boot_build)
        if newer is None:
            # The sentences are module constants, not literals: the caller
            # routes on them, so a reword here must break that match loudly
            # rather than fall through to its generic ``kept`` branch.
            if buildwatch.pending_build(self._boot_build) is not None:
                return buildwatch.KEPT_UNSETTLED
            return buildwatch.KEPT_MATCHES
        logger.info(
            "session runtime: viewer asked for a refresh; build on disk is %s, loaded %s",
            newer.label(),
            self._boot_build.label(),
        )
        answer = await self._retire_for("stale-build", to=newer.label())
        if answer == "retiring":
            # NAME THE BUILD IT LEAVES FOR, because "which of these is still on
            # the old build" is the question this op exists to answer and a
            # version-only label cannot answer it (same-version rebuilds are
            # this host's common drift — see ``proves_a_move``). The frame
            # already carries ``to``; a caller that wanted it previously had to
            # re-read the marker itself, which is a second, racier read of a
            # fact this process just committed to. Suffixing rather than
            # replacing keeps every consumer that reads the ``retiring``
            # PREFIX working (the TUI's bind-path refresh, and an attach client
            # passing the answer through verbatim); only ``retire_now``, whose
            # callers compare the whole string, is left alone.
            return f"retiring to {newer.label()}"
        return answer

    def _retire_detail(self, to: str) -> str:
        """``" (old → new)"`` for a retirement reason, when both stamps exist.

        The build pair is what makes the reason actionable later: "the runtime
        retired" says nothing about which build it left for.
        """
        boot = getattr(self, "_boot_build", None)
        if boot is None or not to:
            return ""
        return f" ({boot.label()} → {to})"

    async def _retire_for(
        self,
        reason_label: str,
        *,
        to: str = "",
        exclusive_owner: _ClientConn | None = None,
    ) -> str:
        """Retire this runtime iff it is idle, announcing ``reason_label``.

        The shared body of the two viewer-driven retirements — a stale build
        (``refresh_if_idle``) and a directory change (``retire_now``). They
        differ only in what makes the retirement OWED, which each caller
        establishes before calling; everything after that decision is the same
        ordering and must stay so, because a second copy of it is how one path
        grows a re-check the other lacks.

        Every failure path RETURNS ``kept: …`` rather than raising, and every
        one keeps the runtime alive. The asymmetry ``_retire_if_pristine``
        records holds here too: a wrong "retire" costs a cold start nobody
        asked for, a wrong "keep" costs one more check.
        """
        h = self._handle
        may_refresh = getattr(h, "may_refresh", None)
        if not callable(may_refresh):
            return "kept: this runtime cannot judge itself idle"
        try:
            reason = str(await self._handle_call_on_session_loop(may_refresh) or "")
        except Exception as exc:  # noqa: BLE001 — uncertainty keeps the runtime
            return f"kept: idle probe failed ({exc})"
        if reason:
            return f"kept: {reason}"
        request_stop = getattr(h, "request_stop", None)
        if not callable(request_stop):
            return "kept: this runtime cannot stop itself gracefully"
        await self.announce_retiring(reason_label, to=to)
        if exclusive_owner is not None and self._other_observers(exclusive_owner) > 0:
            # THE FENCE IS RE-CHECKED AT THE LATCH, not only at admission. The
            # announcement above is an await, and the initial count is a sample:
            # without this a viewer that attached during it would be counted by
            # nobody and the move would commit with a sibling already registered
            # — the R3 race in its narrowest form. ``_on_connection`` refuses new
            # attaches while the fence is held, so this recheck plus that gate
            # close the window from both sides. A refusal here RELEASES the
            # fence: nothing was retired, so the runtime must admit viewers
            # again.
            self._exclusive_move_fence = None
            return (
                "kept: This session is open in another terminal or attached "
                "client. Disconnect that client, then move again."
            )
        # The ONE await between decision and stop, so the final check is a LATCH
        # and not another sample: a ``prompt`` admitted in this gap would open a
        # turn that ``request_stop`` then aborts one await later. ``begin_retire``
        # commits the runtime in the same synchronous step that checks it, so
        # from here the admissions refuse and the retirement is clean by
        # construction (design §5.1). The cut-off CAUSE is always
        # ``runtime-retired``: ``reason_label`` names the trigger for the log and
        # the wire announcement, while the cause is the vocabulary token a
        # restored session renders.
        begin_retire = getattr(h, "begin_retire", None)
        if callable(begin_retire):
            # HOPPED, and the hop is what preserves the property the comment
            # above claims: ``begin_retire`` commits in the same synchronous
            # step that checks, and the helper hands the WHOLE call to the
            # session's loop — so check-and-commit is still one step, just one
            # step over there rather than over here. Splitting it into a
            # sampled re-check followed by a separate commit would be the
            # window this latch exists to close.
            if not await self._handle_call_on_session_loop(
                begin_retire, "runtime-retired", self._retire_detail(to)
            ):
                return "kept: work arrived while retiring was announced"
            # THE LATCH HAS COMMITTED, recorded here rather than derived by the
            # caller from this function's return value: the stop below can
            # raise, and a raise must not read as "nothing committed" (review
            # round 2, N6).
            self._retirement_committed = True
        else:
            # A reduced/older handle without the latch keeps today's re-check
            # rather than retiring unguarded.
            try:
                reason = str(await self._handle_call_on_session_loop(may_refresh) or "")
            except Exception as exc:  # noqa: BLE001
                return f"kept: idle probe failed ({exc})"
            if reason:
                return f"kept: {reason} (arrived while retiring was announced)"
        logger.info("session runtime: retiring (%s)", reason_label)
        await self._handle_call_on_session_loop(request_stop)
        return "retiring"

    async def _already_admitted(self, op: str, frame: dict[str, Any]) -> bool:
        """Is this a retry of a turn the transcript already carries?

        Only ``prompt`` carries a durable, append-only identity, so only it can
        be answered from the transcript. ``steer`` is deliberately excluded:
        its idempotency is the handle's own reservation map, and a steer is not
        an append-only user row to match against.

        Optional capability, probed — a reduced handle without it simply never
        reports a duplicate, which is the pre-idempotency behaviour; the same
        probe failing for any other reason answers ``False`` for the same
        reason (never fail a turn over a dedupe probe).

        ASYNC, and hopped, because the read is the transcript's and the
        transcript belongs to the session's loop — the same rule as the retire
        probes below, through the same helper.
        """
        if op != "prompt":
            return False
        command_id = str(frame.get("command_id") or "")
        if not command_id:
            return False
        checker = getattr(self._handle, "has_admitted_command", None)
        if not callable(checker):
            return False
        try:
            return bool(await self._handle_call_on_session_loop(checker, command_id))
        except Exception:  # noqa: BLE001 — never fail a turn over a dedupe probe
            logger.debug("admitted-command probe failed", exc_info=True)
            return False

    async def _dispatch(self, op: str, frame: dict[str, Any]) -> str | AckDetail:
        from local_operator.mobile.types import validate_control_frame

        validate_control_frame(frame)
        h = self._handle
        if op == "acknowledge_attention":
            token = frame.get("completion_token")
            if not isinstance(token, str) or len(token) != 36:
                raise ValueError("completion_token must identify the rendered completion")
            acknowledge = getattr(self._handle, "acknowledge_attention", None)
            if not callable(acknowledge):
                raise ValueError("completion acknowledgements unavailable; update the owner")
            state = await cast(Any, acknowledge)(token)
            self._schedule_push()
            # The store's own answer, computed in the same write transaction that
            # decided the receipt. A handle that returns nothing (a test double,
            # or an owner whose ack is a bare op) answers with no state rather
            # than a fabricated one: see ``AckDetail`` for why the caller must
            # then stay inconclusive.
            return AckDetail("completion acknowledged", state if isinstance(state, dict) else {})
        if op == "ping":
            return "pong"
        if op == "snapshot":
            await self._push()
            return "snapshot sent"
        if op == "prompt":
            images = frame.get("images")
            fields: dict[str, Any] = {"images": images}
            if "command_id" in inspect.signature(h.prompt).parameters:
                fields["command_id"] = frame.get("command_id")
            return await h.prompt(frame["text"], **fields)
        if op == "steer":
            fields = {"images": frame.get("images")}
            if "command_id" in inspect.signature(h.steer).parameters:
                fields["command_id"] = frame.get("command_id")
            return await h.steer(frame["text"], **fields)
        if op == "abort":
            return await h.abort()
        if op == "cancel":
            # Two rungs, one op, because the CHOICE is the feature. The default
            # is graceful: a supervisor cancelling a sentinel mid-``git push``
            # must not tear the push in half, and defaulting the other way makes
            # the dangerous behaviour the one you get by not thinking. Asking
            # for ``immediate`` is asking for ``abort``, so it routes there
            # rather than growing a second implementation of "stop now".
            #
            # Optional capability, getattr-probed like every other addition to
            # this dispatch: a reduced test handle or an older bridge answers
            # the unknown-op error, and a caller that needs a stop regardless
            # falls back to ``abort`` (or to the stop ladder). Additive on the
            # wire for the same reason ``peer_message`` and ``stop`` were, so
            # no PROTOCOL_VERSION bump.
            if str(frame.get("mode", "graceful")) == "immediate":
                return await h.abort()
            cancel = getattr(h, "cancel_gracefully", None)
            if not callable(cancel):
                raise ValueError("this session cannot cancel at a tool boundary")
            typed_cancel = cast(Callable[[], Awaitable[str]], cancel)
            return await typed_cancel()
        if op == "set_model":
            provider = str(frame.get("provider", ""))
            model_id = str(frame.get("model_id", ""))
            effort = frame.get("effort")
            if effort:
                # Optional capability, getattr-probed like every other addition
                # to this dispatch, and for the reason the ``cancel`` arm states:
                # the handle is duck-typed across several owners (the serving
                # session, a TUI-hosted one, and a long tail of test doubles),
                # so a third POSITIONAL argument would break every handler that
                # implements the two-argument call the protocol declares. An
                # owner without this method keeps answering ``set_model``, and
                # the level then degrades to the model's own default instead of
                # failing a switch the user asked for.
                with_effort = getattr(h, "set_model_effort", None)
                if callable(with_effort):
                    typed_model = cast(Callable[[str, str, str], Awaitable[str]], with_effort)
                    return await typed_model(provider, model_id, str(effort))
            return await h.set_model(provider, model_id)
        if op == "set_effort":
            return await h.set_effort(str(frame.get("effort", "")))
        if op == "complete_aside":
            complete_aside = getattr(h, "complete_aside", None)
            if not callable(complete_aside):
                raise ValueError("this owner cannot run off-record requests")
            result = complete_aside(list(frame.get("turns") or []))
            if not inspect.isawaitable(result):
                raise ValueError("owner complete_aside operation must be awaitable")
            return await result
        if op == "slash":
            images = frame.get("images")
            if images:
                slash_images = getattr(h, "slash_images", None)
                if not callable(slash_images):
                    raise ValueError("this owner cannot route slash-command images")
                typed_slash_images = cast(
                    Callable[[str, str, list[dict[str, str]]], Awaitable[str]],
                    slash_images,
                )
                return await typed_slash_images(
                    str(frame.get("command", "")),
                    str(frame.get("args", "")),
                    images,
                )
            # Old daemon/reduced handles predate attachment-bearing slash ops;
            # preserve their two-argument call shape when no pixels ride the frame.
            return await h.slash(str(frame.get("command", "")), str(frame.get("args", "")))
        if op == "new_conversation":
            return await h.new_conversation()
        if op == "resume_session":
            return await h.resume_session(str(frame.get("session_id", "")))
        if op == "approval_answer":
            return await h.approval_answer(
                frame["request_id"],
                frame["approved"],
                frame.get("remember", False),
            )
        if op == "recall_steer":
            # v4: follower Esc-recall parity. Optional capability — an owner
            # host that predates it answers with the unknown-op error, which
            # the follower surfaces as "cannot recall here".
            recall = getattr(h, "recall_steer", None)
            if not callable(recall):
                raise ValueError("this owner cannot recall queued steering")
            typed_recall = cast(Callable[[str], Awaitable[str]], recall)
            return await typed_recall(str(frame.get("command_id", "")))
        if op == "ask_answer":
            # ``question_index`` is the question the phone was DISPLAYING when
            # the user tapped (U8 guard): the handle rejects the answer if the
            # picker has since advanced past it. Optional — an older client that
            # omits it falls back to answering the current question, the
            # pre-guard behaviour.
            raw_index = frame.get("question_index")
            question_index = int(raw_index) if isinstance(raw_index, (int, float)) else None
            return await h.ask_answer(
                frame["request_id"],
                frame["value"],
                question_index=question_index,
            )
        if op == "adopt_aside":
            adopt = getattr(h, "adopt_aside", None)
            if not callable(adopt):
                raise ValueError("this owner cannot adopt an aside")
            result = adopt(list(frame.get("messages") or []))
            if not inspect.isawaitable(result):
                raise ValueError("owner adopt_aside operation must be awaitable")
            return await result
        if op == "peer_message":
            # THE RECEIVE-SIDE HALF OF THE UNENGAGED GATE, and the only one that
            # holds against a sender this build does not control. Resolution
            # refuses such a target and ``deliver_peer_message`` refuses it
            # again, but a sender on an OLDER build reads a pre-field record's
            # missing ``started`` key as True (``SessionRecord.from_json``) and
            # dials anyway — and a peer row written through this op becomes the
            # OPENING row of a conversation whose owner has not typed yet, which
            # is the symptom this gate exists for. Refusing HERE makes the
            # guarantee independent of the sender's build: the dispatch turns
            # the ValueError into an ``error`` frame, ``peer_client`` raises it
            # as a RuntimeError, and both send surfaces render
            # ``could not deliver: ...``.
            #
            # ``self._started`` IS the published bit (``set_record_started``
            # flips both together), so what this refuses is exactly what
            # ``lop sessions`` reports as an unengaged composer window. The
            # import is in-function and INSIDE the refusal branch: this module
            # keeps its import weight off the happy path, and peer_send is only
            # needed to name the refusal. The label follows the ADDRESS — a
            # dial arrives at this runtime's PID, so the pid is what the sender
            # typed (or what ``lop sessions`` showed it), not the session id it
            # never named; both come from ``unengaged_label`` so all four
            # refusal sites share one grammar.
            if not self._started:
                from local_operator.mobile.peer_send import (
                    unengaged_label,
                    unengaged_refusal,
                )

                raise ValueError(
                    unengaged_refusal(
                        unengaged_label(pid=self._record.pid, session_id=self._record.session_id)
                    )
                )
            # Cross-session `lop send` delivery. Optional capability — an owner
            # host that predates peer messaging (or a non-interactive exec host
            # that never wired it) answers with a clear error, which the sender
            # surfaces as "this session cannot receive peer messages". Same
            # getattr guard as recall_steer above.
            receive = getattr(h, "receive_peer_message", None)
            if not callable(receive):
                raise ValueError("this session cannot receive peer messages")
            typed_receive = cast(Callable[..., Awaitable[str]], receive)
            return await typed_receive(
                frame["text"],
                mode=str(frame.get("mode", "mailbox")),
                wake=bool(frame.get("wake", False)),
                sender=frame.get("sender") or {},
            )
        if op == "stop":
            # PR 3 (the kill switch): the graceful rung of the stop ladder
            # (session/runtime/control.py). The plan is deny parked gates →
            # abort/dispose the session → release the lease → unpublish the
            # record → exit, executed by the host's ``request_stop`` hook
            # (ServingSessionHandle.request_stop owns the ordering); all this
            # dispatch does is trigger it and ack, so the ack reaching the
            # caller means "the stop is underway", not "it finished" — the
            # ladder's timeout decides what a slow exit costs.
            #
            # Optional capability, probed: a host that predates the hook (a
            # reduced test handle, an old TUI bridge) answers with an error,
            # which the ladder treats as a scheduled miss and proceeds to
            # identity-confirmed SIGTERM. Additive on the wire — no
            # PROTOCOL_VERSION bump, for the same reason ``peer_message``
            # needed none: an old runtime's unknown-op error is exactly the
            # answer the ladder is built to continue from.
            request_stop = getattr(h, "request_stop", None)
            if not callable(request_stop):
                raise ValueError("this owner cannot stop itself gracefully")
            # Tell every attached viewer the disconnect they are about to see
            # is DELIBERATE, before the session goes away. Without this a
            # follower cannot distinguish a stop from owner death and its
            # recovery takes over the session a user just ended — republishing
            # a live record for a cold session (U2-4). Announced BEFORE the
            # hook runs because the hook's own teardown closes these sockets.
            # An old viewer ignores the unknown frame, so this stays additive.
            await self._broadcast({"op": "stopping", "session_id": self._record.session_id})
            # HOPPED: ``request_stop`` mutates the session (``_note_deliberate_stop``,
            # ``_deny_pending_gates``) and then runs the host hook, so it belongs
            # on the session's loop — a kill switch is the worst place to run a
            # mutation from the wrong thread. For a handle that does not publish
            # a loop (the TUI's) the helper runs it inline exactly as before.
            result = await self._handle_call_on_session_loop(request_stop)
            # The host's own line when it gives one (a TUI owner names the
            # session and the reopen command), else the bare progress word.
            return str(result) if isinstance(result, str) and result else "stopping"
        raise ValueError(f"unknown op: {op!r}")

    async def _dispatch_payload(
        self,
        op: str,
        frame: dict[str, Any],
        locality: ClientLocality = "local",
        consumers: frozenset[str] | None = None,
        audit_capable: bool = False,
        may_loosen: bool | None = None,
    ) -> Any:
        """Structured-answer ops: the return value becomes the ``result`` data.

        ``consumers`` is the calling connection's declared
        ``slash_consumers`` set (``None`` when the auth frame omitted it),
        carried here for the same reason ``locality`` is: it is a property of
        the CONNECTION, not of the frame, and only the handle can act on it.
        ``audit_capable`` is a third of the same kind — whether this viewer
        negotiated ``display-history-audit-v1`` — and it decides whether a
        display page may carry the audit fields at all.
        """
        h = self._handle
        if op == "fork_snapshot":
            # Fork destinations are resumed from this machine's session store.
            # A foreign viewer must not receive a local id it cannot reach, nor
            # select another parent's path through the authenticated owner.
            if locality != "local":
                raise ValueError("fork requires a terminal on the session's machine")
            if set(frame) - {"op", "req", "message"}:
                raise ValueError("fork_snapshot accepts only a message")
            message = frame.get("message", "")
            if not isinstance(message, str):
                raise ValueError("fork message must be text")
            snapshot = getattr(h, "fork_snapshot", None)
            if not callable(snapshot):
                raise ValueError("this owner cannot fork; update it and retry /fork")
            result = snapshot(message)
            if inspect.isawaitable(result):
                result = await result
            return result
        if op == "slash_result":
            run = getattr(h, "run_slash_authoritative", None)
            if not callable(run):
                raise ValueError("this owner cannot run typed slash results")
            args: list[Any] = [
                str(frame.get("command", "")),
                str(frame.get("args", "")),
                list(frame.get("images") or []),
            ]
            # ``locality`` and ``consumers`` are passed only to handles that
            # accept them. A handle is an injected collaborator — the TUI's,
            # the runtime's, and test doubles all implement this — so widening
            # the call unconditionally would break every implementation that
            # has not been updated. The probe keeps each parameter OPTIONAL in
            # the protocol rather than forcing a lockstep change, which is the
            # same capability-probe stance the ops above take with ``getattr``.
            #
            # ``consumers`` is what lets the handle decide whether the CLIENT
            # will submit an action-carrying receipt's request or whether the
            # runtime must do it. It is the connection's declaration, so it is
            # read here where the connection is known rather than threaded
            # through the frame.
            kwargs: dict[str, Any] = {}
            if _accepts_kw(run, "locality"):
                kwargs["locality"] = locality
            if _accepts_kw(run, "consumers"):
                kwargs["consumers"] = consumers
            if _accepts_kw(run, "may_loosen"):
                # The connection's own property, read where the connection is
                # known — the same reason ``consumers`` is (design round 2 D10,
                # UX round 2 U9).
                kwargs["may_loosen"] = may_loosen

            result = run(*args, **kwargs)
            if inspect.isawaitable(result):
                result = await result
            return result
        if op == "mcp_credentials":
            from local_operator.mcp.credentials import MCPCredentials

            if locality == "remote":
                return {"code": "remote_client", "saved_ids": [], "failed_ids": []}
            try:
                body = MCPCredentials.model_validate(frame.get("body"))
            except Exception:
                # Pydantic diagnostics can include invalid raw values. Never
                # let its exception enter the generic RPC error serializer.
                raise ValueError("Invalid MCP credential fields") from None
            operation = getattr(h, "mcp_credentials_op", None)
            if not callable(operation):
                raise ValueError("Update the backend for secure MCP key entry")
            answer = operation(
                body.model_dump(mode="json")
                | {"values": {key: value.get_secret_value() for key, value in body.values.items()}}
            )
            # Both shapes accepted, exactly as ``credential`` above does it: a
            # handle may implement the verb synchronously (the in-process session
            # does) and a routed runtime asynchronously, and the caller must not
            # care which.
            if inspect.isawaitable(answer):
                answer = await answer
            return answer
        if op == "credential":
            # Validated HERE because the payload path does not run
            # ``validate_control_frame`` the way ``_dispatch`` does (a
            # pre-existing gap for every payload op). Only this op is wired,
            # deliberately: it is the one that carries a secret, and widening
            # the validator over ``slash_result`` is a behaviour change for
            # every routed slash that belongs in its own review.
            from local_operator.mobile.types import validate_control_frame

            validate_control_frame(frame)
            # A DEDICATED op rather than a `slash_result` with the secret in
            # its `args` string. The value must never sit in a general-purpose
            # field that other code paths echo, log, or put in a transcript —
            # `args` is the same field that carries `/goal` text. Here the
            # secret has exactly one named home and one consumer.
            #
            # The store lives on the owner because that is where the agent's
            # bash commands run and read it from the environment; a follower
            # storing it locally would tell the model about a key that no tool
            # on the executing side can use.
            #
            # Optional capability, getattr-probed like `cancel` above: a
            # reduced handle or an older runtime answers the unknown-op error
            # rather than silently accepting a secret it will not store.
            credential = getattr(h, "credential_op", None)
            if not callable(credential):
                raise ValueError("this owner cannot hold session credentials")
            action = str(frame.get("action", ""))
            if action == "store" and locality == "remote":
                # Same locality rule the `/mcp` grant verbs apply
                # (``mcp/grants.py::REMOTE_GRANT_NOTICE``): a secret pasted on
                # a RELAYED client — a phone, in a future relay topology —
                # would be typed on a device the desktop's environment was
                # never meant to trust, and then injected into every bash
                # command here. Loopback attach clients are ``local`` by
                # construction, so no user today is refused; the gate exists
                # so the relay, when it lands, does not inherit a write it
                # never opted into (review round 1, R4). The read verbs stay
                # open: they return key NAMES only, never a value.
                return {"ok": False, "reason": "remote-client"}
            result = credential(
                action,
                str(frame.get("key", "")),
                str(frame.get("value", "")),
            )
            if inspect.isawaitable(result):
                result = await result
            return result
        if op == "variables":
            # Validated HERE for the same reason ``credential`` is: the payload
            # path does not run ``validate_control_frame``, and this op both
            # writes into a live namespace and reads values back out of it. The
            # frame carries no session identity — the owner answers for the
            # session it IS (``ServingSessionHandle.variables_op``), so a viewer
            # cannot reach another conversation's code memory by naming it.
            from local_operator.mobile.types import validate_control_frame

            validate_control_frame(frame)
            # Optional capability, getattr-probed like ``credential``: a reduced
            # handle (a test host, a TUI-owned handle) cannot answer it. The
            # refusal is worded as the unknown-op error the transport already
            # raises for an op an older runtime does not list at all, because
            # that is the same fact seen from the viewer's side — and the viewer
            # classifies ``unsupported`` on exactly that string. A descriptive
            # sentence of its own would reach the panel as a 503 instead of its
            # "update the backend" state.
            variables = getattr(h, "variables_op", None)
            if not callable(variables):
                raise ValueError("unknown op: 'variables'")
            result = variables(
                str(frame.get("action", "")),
                str(frame.get("key", "")),
                str(frame.get("value", "")),
                str(frame.get("type", "")),
            )
            if inspect.isawaitable(result):
                result = await result
            return result
        if op == "register_secret_redaction":
            # The viewer→runtime half of §6: the viewer's registration answered
            # the broker's notice because this runtime registered nothing (it
            # booted before a store existed at its config root, §13), and the
            # value has to land in THIS process's VariableStore — the one the
            # bash and eval redactors read. Validated for the same reason
            # ``credential`` is: the payload path skips the control-frame
            # validator, and this op carries a secret's value.
            from local_operator.mobile.types import validate_control_frame

            validate_control_frame(frame)
            # getattr-probed like every optional capability: a reduced or older
            # handle answers the unknown-op error rather than silently accepting
            # a value it will not scrub. The handler only writes the value into
            # the redaction set — never a credential, never an announcement,
            # never a log line, never the event stream.
            register = getattr(h, "register_secret_redaction", None)
            if not callable(register):
                raise ValueError("this owner cannot register a redaction")
            # HOPPED. This body WRITES the session's redaction set — the one the
            # bash and eval redactors read — so it belongs on the loop that owns
            # it, like every other mutating handle call. The helper resolves an
            # awaitable result too, so a handle whose registration is an
            # ``async def`` still works.
            await self._handle_call_on_session_loop(register, str(frame.get("value", "")))
            return True
        if op == "cancel_subagents":
            cancel = getattr(h, "cancel_subagents_count", None)
            if not callable(cancel):
                raise ValueError("this owner cannot cancel subagents")
            # HOPPED, and this one is not a tidiness call: the body runs
            # ``Session.cancel_subagents``, which creates the cancellation task
            # with ``asyncio.ensure_future`` and aborts a loop-bound
            # ``AsyncJobManager`` signal. On the runtime's thread that is a task
            # created on the wrong loop and a foreign-thread ``Event.set()``,
            # and the manager's ``RuntimeError: got Future … attached to a
            # different loop`` is SWALLOWED by its own
            # ``logger.warning("job %s task raised on cancel")`` while this op
            # still acks success — the cancel's execution left parked on the
            # wrong loop with its failure swallowed, and the caller shown a
            # count. NOT "cancels nothing": the instrument that found this
            # watched the child's ``CancelledError`` land and the job row settle
            # in BOTH arms, so the claim that survives is the plane and the
            # swallowed failure (review round 2, QA Q3 / design D-8). The
            # stronger reading needs a child parked on the manager's abort
            # signal rather than on a sleep, which no rig here built. This is
            # the operator's second-Esc path.
            result = await self._handle_call_on_session_loop(cancel)
            return result if isinstance(result, int) else 0
        if op == "record_shell":
            from local_operator.harness.types import ToolResult

            record = getattr(h, "record_shell", None)
            command = frame.get("command")
            if not callable(record) or not isinstance(command, str) or not command.strip():
                raise ValueError("invalid shell receipt")
            result = ToolResult.model_validate(frame.get("result"))
            if (
                result.tool_name != "bash"
                or not result.tool_call_id
                or len(result.tool_call_id) > 128
            ):
                raise ValueError("invalid shell receipt identity")
            outcome = record(command, result)
            if inspect.isawaitable(outcome):
                await outcome
            return "accepted"
        if op == "frontend_sync":
            from local_operator.session.frontend_state import (
                FrontendSubscription,
                oversized_frame_report,
                sync_wire_payload,
            )
            from local_operator.session.history_window import strip_audit_fields

            capture = getattr(h, "subscribe_frontend", None)
            if not callable(capture):
                raise ValueError("canonical history refresh is unavailable")
            outcome = capture(lambda _update: None, display_window=True)
            subscription = cast(
                FrontendSubscription, await outcome if inspect.isawaitable(outcome) else outcome
            )
            try:
                payload = sync_wire_payload(subscription.sync)
                # The THIRD wire route, and the one an old viewer uses most:
                # ``AttachedSession._refresh_display_history`` calls this op on
                # every history refresh, which fires whenever the frontend's
                # ``history_generation`` moves — i.e. as soon as a new owner
                # appends a row. ``audit``/``audit_available`` are ordinary
                # model fields, so they serialize on an UNCOMPACTED page too;
                # missing the strip here is a hard ValidationError on the
                # viewer's nested ``DisplayHistoryWindow`` (``extra='forbid'``)
                # rather than a compaction-only or a racy failure.
                if isinstance(payload.get("display_history"), dict):
                    strip_audit_fields(payload["display_history"], audit_capable=audit_capable)
                # Keep the existing live subscription. This temporary capture
                # only supplies an atomic cut; it must not multiply observers.
                response = {"op": "result", "req": frame.get("req"), "data": payload}
                if oversized_frame_report(response, _MAX_LINE_BYTES) is not None:
                    window = subscription.sync.display_history
                    if window is not None:
                        # Re-serialized from the model, so it needs the same
                        # strip as the page above rather than inheriting it.
                        payload["display_history"] = strip_audit_fields(
                            window.model_copy(
                                update={
                                    "status": "full_required",
                                    "messages": [],
                                    "durable_seed_ids": [],
                                    "durable_seed_tool_ids": [],
                                    "before_token": None,
                                    "snapshot_token": None,
                                }
                            ).model_dump(mode="json"),
                            audit_capable=audit_capable,
                        )
                    if oversized_frame_report(response, _MAX_LINE_BYTES) is not None:
                        raise ValueError("canonical refresh exceeds the transport frame limit")
                return payload
            finally:
                subscription.unsubscribe()
        if op == "history_page":
            fetch = getattr(h, "history_page", None)
            before = frame.get("before")
            anchor = frame.get("anchor", "")
            if not callable(fetch) or not isinstance(before, str) or not isinstance(anchor, str):
                raise ValueError("invalid history page request")
            if len(before) > 4096 or len(anchor) > 512:
                raise ValueError("invalid history page request")
            result = fetch(before, anchor)
            payload = await result if inspect.isawaitable(result) else result
            # Same contract as the sync frame: a viewer that did not negotiate
            # the audit capability must not be handed fields its page model
            # forbids.
            if isinstance(payload, dict):
                from local_operator.session.history_window import strip_audit_fields

                strip_audit_fields(payload, audit_capable=audit_capable)
            return payload
        if op == "job_trajectory":
            # The other half of the frame-size fix: the attach snapshot omits
            # trajectories, so a viewer opening a child page pulls that one
            # job's window here, in pages. Optional capability — an older
            # runtime answers unknown-op and the viewer degrades to "trajectory
            # unavailable" rather than rendering the child as empty.
            fetch = getattr(h, "job_trajectory", None)
            if not callable(fetch):
                raise ValueError("this owner cannot serve job trajectories")
            job_id = str(frame.get("job_id") or "")
            if not job_id:
                raise ValueError("job_id must be a non-empty string")
            offset = max(0, int(frame.get("offset") or 0))
            requested = int(frame.get("limit") or _TRAJECTORY_PAGE_MAX)
            limit = max(1, min(requested, _TRAJECTORY_PAGE_MAX))
            result = fetch(job_id, offset, limit)
            if inspect.isawaitable(result):
                result = await result
            return result
        raise ValueError(f"unknown op: {op!r}")

    # -- v5 frontend state relay ----------------------------------------------

    def _relay_frontend_to(self, conn: _ClientConn, data: dict[str, Any]) -> None:
        loop = self._loop
        if loop is None or self._closed.is_set():
            return
        try:
            loop.call_soon_threadsafe(self._relay_frontend_to_on_loop, conn, data)
        except RuntimeError:
            pass

    def _relay_frontend_to_on_loop(self, conn: _ClientConn, data: dict[str, Any]) -> None:
        if id(conn.writer) not in self._clients:
            return
        from local_operator.session.frontend_state import filter_update_trajectories

        # Per-connection, and applied on THIS loop rather than at the producer:
        # one canonical update fans out to every client, each of which has its
        # own open child page (or none).
        data = filter_update_trajectories(
            data, conn.watched_jobs.__contains__, line_limit_bytes=_MAX_LINE_BYTES
        )
        if not conn.frontend_ready:
            if len(conn.frontend_pending) >= _EVENT_QUEUE_MAX:
                # A join that cannot install its boundary before this many
                # canonical edges is already stale. Drop and let it reconnect
                # to one fresh snapshot rather than retain an unbounded suffix.
                self._drop_client(conn, reason="frontend pending overflow before ready")
                return
            conn.frontend_pending.append(data)
            return
        self._enqueue_client_frame(conn, {"op": "frontend_update", "data": data})

    def _enqueue_client_frame(self, conn: _ClientConn, frame: dict[str, Any]) -> None:
        """Queue one state/event frame on the connection's sole ordered FIFO."""
        # THE chokepoint for both relay frame types: ``frontend_update`` and
        # ``event`` both arrive here, so guarding once covers both and no
        # future relay caller can bypass it by forgetting to check. (The
        # ``frontend_sync`` frame does not come through here — it is written
        # directly at connect time and carries its own report.)
        # The fit pass sits in front of the guard so the guard is the last
        # resort rather than the normal path for a payload-bearing frame.
        frame = fit_frame_for_wire(frame, _MAX_LINE_BYTES)
        try:
            conn.event_queue.put_nowait(frame)
        except asyncio.QueueFull:
            # A full FIFO is not always a slow CLIENT: a provider can hand the
            # session a whole turn of token deltas in one loop tick, which
            # overflows any bound before the drain writes a byte. Those frames
            # are losslessly coalescible (each ``message_update`` carries the
            # full accumulated message and its append-only ``delta``), so
            # compact first and drop only a client that stays full — the
            # genuinely slow reader the bound exists for.
            compacted = self._compact_event_queue(conn)
            if compacted:
                try:
                    conn.event_queue.put_nowait(frame)
                    compacted = True
                except asyncio.QueueFull:
                    compacted = False
            if not compacted:
                self._drop_client(conn, reason=f"event queue overflow ({_EVENT_QUEUE_MAX} frames)")
                return
        if conn.event_writer_task is None:
            task = asyncio.create_task(self._drain_event_queue(conn))
            conn.event_writer_task = task
            self._event_sends.add(task)
            task.add_done_callback(self._event_sends.discard)

    # -- v4 event relay --------------------------------------------------------

    def _relay_event(self, data: dict[str, Any]) -> None:
        """Thread-safe relay entry: events fire on the HOST's thread (the
        Textual loop for a TUI owner), and everything relay-ordered must run
        on the runtime loop. ``call_soon_threadsafe`` from one producer
        thread preserves emission order, which is the whole relay contract."""
        loop = self._loop
        if loop is None or self._closed.is_set():
            return
        try:
            loop.call_soon_threadsafe(self._relay_on_loop, data)
        except RuntimeError:  # loop closing
            pass

    def _relay_on_loop(self, data: dict[str, Any]) -> None:
        """Fan one serialized AgentEvent out to event-subscribed attach clients.

        Folding into the live-turn tracker and snapshotting the recipient set
        happen SYNCHRONOUSLY here — that, plus the seed block in
        ``_on_connection``, is what makes a mid-turn join gapless (see the
        comment there). Each connection owns one bounded FIFO writer: a slow
        follower cannot delay healthy peers or accumulate tasks indefinitely,
        while a healthy follower receives every frame in emission order.

        Daemon connections are never in the recipient set: the phone's
        projection path is byte-identical to v3 by construction.
        """
        if self._closed.is_set():
            return
        # A muted connection (``EVENT_MUTE_CAPABILITY``) is filtered PER EVENT,
        # not dropped from the recipient snapshot: its interest is the delta
        # grade only, and turn boundaries, gates, notices and so on must keep
        # flowing or a parked viewer's state would silently rot while muted.
        event_type = str(data.get("type") or "")
        # Snapshotted for the same reason as ``_visible_attach_surfaces``: this
        # runs on the runtime's loop today, but the iteration is over a dict the
        # session's loop can mutate (see that method), and one line is cheaper
        # than the reasoning that would have to hold forever.
        recipients = [
            conn
            for conn in list(self._clients.values())
            if conn.kind == "attach"
            and conn.wants_events
            and conn.events_ready
            and not (conn.events_muted and event_type in EVENT_MUTE_DROP_TYPES)
        ]
        if not recipients:
            return
        frame = {"op": "event", "data": data}
        for conn in recipients:
            # Raw events and canonical deltas share this FIFO. Their producer
            # callbacks are scheduled in session order, so a follower cannot
            # observe transcript animation ahead of the state edge that caused it.
            self._enqueue_client_frame(conn, frame)

    def _compact_event_queue(self, conn: _ClientConn) -> bool:
        """Fold the delta-grade and compose frame families in place.

        Merges runs of same-stream ``message_update`` and ``reasoning_delta``
        frames, and keeps only the NEWEST ``tool_call_compose`` per
        ``tool_call_id``. Which frames belong to one stream is
        :func:`_mergeable_delta_key`'s rule, and it is the only family-specific
        thing here: the size accounting below is delta-sized and therefore
        family-agnostic.

        Both are lossless by construction. For ``message_update`` the later
        event's ``message`` already contains the earlier one's text, and
        concatenating ``delta`` preserves the append contract UIs rely on. For
        ``reasoning_delta`` there is no accumulated payload at all — the frame
        carries one fragment, a ``message_id`` both frames agree on, and the
        same concatenation reproduces the two fragments in arrival order. This
        matters as much as it does for text: reasoning arrives once per token,
        and a long-thinking turn is thousands of frames, so without the fold a
        stalled viewer's FIFO fills with incompressible reasoning frames and is
        dropped — the same failure the compose fold below was written for. For
        ``tool_call_compose`` the argument is the one ``_fold_live_event``
        (``frontend_state.py``) already relies on for the reconnect seed: a
        compose frame is a SNAPSHOT of a call being dictated (``tool_name``,
        CUMULATIVE ``argument_bytes``, ``intent``), never a delta, so an older
        frame carries nothing the newer one lacks. Returns whether any room was
        freed. Runs synchronously on the runtime loop, so the drain task cannot
        observe a half-compacted queue.

        WHY THE COMPOSE FOLD EXISTS. Without it these frames are
        incompressible — each carries a distinct ``tool_call_id`` and a growing
        ``argument_bytes`` — so a viewer stalled during a tool-argument
        dictation fills its whole FIFO with frames compaction cannot touch,
        frees nothing, and is dropped with ``event queue overflow``. That is
        not hypothetical: at three concurrent calls the measured compose rate
        is 16.0 frames/s, which overflows ``_EVENT_QUEUE_MAX`` in ~4.0 s —
        BEFORE ``_TUI_SEND_TIMEOUT_S`` (5.0 s) can be reached, so the timeout
        raised from 1.0 s specifically to stop dropping stalled TUI viewers was
        bypassed for exactly the multi-call case it was meant to protect. The
        dropped viewer re-attaches, its seed re-mounts the composing rows, and
        nothing ever adopts or retires them: the user sees several tool cards
        frozen at one identical elapsed time on calls that in fact ran fine.
        Folding bounds the queue at ONE entry per in-flight call regardless of
        how long the dictation runs, which closes the class; raising
        ``_EVENT_QUEUE_MAX`` would only buy a linear factor against an
        unbounded dictation and re-open the memory question the bound exists
        for.

        THE COMPOSE FOLD REPLACES IN PLACE AND NEVER APPENDS. The newer frame
        takes the retained slot's FIFO position, and the fold is abandoned for
        a call as soon as any OTHER frame for that same ``tool_call_id`` goes
        by (``tool_execution_start``/``_update``/``_end``). Both halves are
        load-bearing: appending would let a compose frame arrive after the
        start that ends composition, and folding across an intervening start
        would move a compose frame BACK past it. Either way the viewer adopts
        out of order and can re-mount a composing row for a call that is
        already executing — the stranded row this fold exists to prevent.

        A COMPOSE KEY IS ONLY UNIQUE WITHIN ONE MODEL STEP, so two further
        guards keep a recycled key from folding across one. The id may be a
        placeholder (``compose:{index}``) derived from ``tool_states``, which
        ``harness/loop.py`` scopes to a single ``_model_turn`` — one STEP of a
        turn, not the whole run — so ``compose:0`` recurs on every tool-calling
        step. A step boundary (``turn_start``/``turn_end``) clears every slot,
        and independently a fold is refused when ``argument_bytes`` goes
        BACKWARDS, which cannot happen within a call and so proves the key was
        reused. Only clearing on ``agent_start``/``agent_end`` guarded the run
        boundary while leaving the step boundary — the common one — open
        (review R1).

        A MERGE THAT WOULD NOT FIT IS REFUSED, and that check cannot be
        skipped on the grounds that everything in this queue already passed
        ``relay_frame_or_degraded``. It did — individually. Merging is the one
        operation here that makes a frame BIGGER than anything the guard was
        shown, so it can assemble individually-legal frames into an
        unreadable one downstream of the only guard: 20 frames of 908,157 B
        merged to 1,060,157 B and killed a real pump, and at 1 KiB deltas a
        full 64-frame queue merged to 1,109,802 B — within 20 KB of the frame
        that killed the operator's socket. The trigger is precisely the burst
        this method exists to absorb.

        REFUSED, not degraded, because refusing is lossless: both frames are
        individually sendable, so keeping them apart costs one queue slot and
        no content, while degrading would throw away deltas the viewer needs.
        Refusing can leave nothing compacted, and that is the honest answer —
        the caller then drops a client it genuinely cannot serve, which is the
        documented slow-reader path rather than a silently broken socket.
        """
        frames: list[dict[str, Any]] = []
        while True:
            try:
                frames.append(conn.event_queue.get_nowait())
            except asyncio.QueueEmpty:
                break
            conn.event_queue.task_done()
        compacted: list[dict[str, Any]] = []
        # Escaped byte length of the DELTA currently held by ``compacted[-1]``.
        # Tracked incrementally so a merge never re-measures the deltas already
        # folded in; see the size check below for why this is exact.
        merged_delta_bytes: list[int] = []
        # ``tool_call_id`` -> index in ``compacted`` of the compose frame
        # retained for that call. An INDEX, not a list position to append at:
        # the newest snapshot overwrites the slot so the row's FIFO mount point
        # is the call's FIRST compose frame, unchanged. Only the snapshot's
        # contents refresh early, and they refresh an already-mounted row.
        #
        # INDEX-PARALLEL WITH ``merged_delta_bytes``, and a replacement here
        # deliberately does NOT touch that list. Safe only because a compose
        # frame carries no ``delta``: the outgoing and incoming frames both
        # contribute 0, so the two lists stay aligned AND correctly valued. A
        # future fold over any frame family that does carry a delta must update
        # ``merged_delta_bytes[slot]`` too, or the ``message_update`` size
        # accounting below silently under-counts and re-opens the oversize bug
        # that accounting exists to prevent.
        compose_slot: dict[str, int] = {}
        for frame in frames:
            if frame.get("op") == "event":
                event_data = frame.get("data") or {}
                event_type = event_data.get("type")
                call_id = str(event_data.get("tool_call_id") or "")
                if event_type == "tool_call_compose" and call_id:
                    slot = compose_slot.get(call_id)
                    if slot is not None and not _compose_reuses_key(compacted[slot], frame):
                        # REPLACE IN PLACE. The frame already passed
                        # ``relay_frame_or_degraded`` individually at enqueue,
                        # and a replacement is not a concatenation, so unlike
                        # the ``message_update`` merge below this cannot
                        # assemble an oversized frame and needs no size check.
                        compacted[slot] = frame
                        continue
                    compose_slot[call_id] = len(compacted)
                elif call_id:
                    # Any OTHER frame for this call ends its composition. Stop
                    # folding it, so a later compose frame appends after this
                    # one instead of moving back in front of it.
                    compose_slot.pop(call_id, None)
                elif event_type in {"turn_start", "turn_end", "agent_start", "agent_end"}:
                    # STEP boundary, not merely a turn boundary. Placeholder
                    # compose keys are index-derived (``compose:{index}``,
                    # ``harness/loop.py``) off ``tool_states``, which is local
                    # to ``_model_turn`` — and ``_model_turn`` runs once PER
                    # STEP, inside ``run()``'s ``while has_more_tool_calls``
                    # loop. So ``compose:0`` recurs on every tool-calling step,
                    # while ``agent_start``/``agent_end`` bracket the whole run.
                    # Resetting only on those left the COMMON boundary open:
                    # step 2's snapshot folded backwards past step 1's start
                    # and end (review R1). ``turn_start`` is the exact frame
                    # ``_model_turn`` emits after creating ``tool_states``, so
                    # it is the boundary the key's lifetime is defined by.
                    #
                    # The ``elif call_id`` branch above does NOT cover this:
                    # under the placeholder regime the compose frame is keyed
                    # ``compose:0`` while start/end carry the provider's real
                    # id, so the pop misses and the slot stays live.
                    compose_slot.clear()
            previous = compacted[-1] if compacted else None
            if (
                previous is not None
                and frame.get("op") == "event"
                and previous.get("op") == "event"
            ):
                data = frame.get("data") or {}
                prior = previous.get("data") or {}
                merge_key = _mergeable_delta_key(data)
                if merge_key is not None and merge_key == _mergeable_delta_key(prior):
                    # SIZE THE DELTA, NOT THE WHOLE FRAME. Re-dumping the
                    # merged frame re-serializes the unchanged accumulated
                    # ``message`` — hundreds of KB — on every merge, which is
                    # quadratic in queue depth: one 64-frame compaction
                    # serialized 67.4 MB and took 152 ms on the runtime loop.
                    # A ``reasoning_delta`` frame has no ``message`` to re-dump,
                    # so the same arithmetic is simply cheap there; it is the
                    # SAME arithmetic, which is what keeps the reasoning family
                    # from needing an accounting of its own.
                    # That loop also owns the ``_SEND_TIMEOUT_S`` sends, so the
                    # stall pushed a healthy peer's 0.90 s drain past 1.0 s and
                    # dropped it — manufacturing the very false disconnect this
                    # guard exists to prevent.
                    #
                    # Exact, not an estimate: JSON escaping is per-character,
                    # so the escaped length of ``a + b`` is exactly the escaped
                    # length of ``a`` plus that of ``b``. The merged frame is
                    # THIS frame carrying its own delta with everything already
                    # folded into ``compacted[-1]`` prepended, so its encoded
                    # size is this frame's size plus those accumulated escaped
                    # bytes. Measuring ``frame`` rather than the merge result
                    # is what makes this O(one frame) per merge — and because
                    # it is THIS frame's own ``message``, a message that grew
                    # between frames is sized correctly rather than estimated
                    # from a stale one. Verified equal to a full re-dump on
                    # adversarial payloads (quotes, backslashes, control
                    # characters, emoji, U+2028) and thousands of random ones.
                    prior_delta_bytes = merged_delta_bytes[-1]
                    frame_bytes = _frame_size_without_delta(frame) + len(
                        json.dumps(str(data.get("delta", ""))).encode()
                    )
                    if frame_bytes + prior_delta_bytes <= _MAX_LINE_BYTES:
                        merged = dict(data)
                        merged["delta"] = str(prior.get("delta", "")) + str(data.get("delta", ""))
                        compacted[-1] = {"op": "event", "data": merged}
                        # ACCUMULATES: the new head carries the prior head's
                        # whole delta plus its own, so the next merge must be
                        # measured against both. Overwriting this with only the
                        # newly-folded delta under-counts every merge after the
                        # second and emitted a 1,048,795-byte frame — over the
                        # cap this method exists to respect.
                        merged_delta_bytes[-1] = prior_delta_bytes + (
                            len(json.dumps(str(data.get("delta", ""))).encode()) - 2
                        )
                        continue
            compacted.append(frame)
            # Seeded with THIS frame's own delta, not zero: the running total
            # is "escaped bytes of the delta ``compacted[-1]`` currently
            # holds", and a later merge prepends all of it. Seeding zero
            # under-counted the first merge of every run. Only the delta is
            # serialized here, never the accumulated message.
            own_delta = (frame.get("data") or {}).get("delta", "")
            merged_delta_bytes.append(len(json.dumps(str(own_delta)).encode()) - 2)
        for frame in compacted:
            conn.event_queue.put_nowait(frame)
        return len(compacted) < len(frames)

    async def _drain_event_queue(self, conn: _ClientConn) -> None:
        """Drain one follower's raw event FIFO until it is dropped."""
        try:
            while id(conn.writer) in self._clients and not self._closed.is_set():
                frame = await conn.event_queue.get()
                try:
                    await self._send_to(conn, frame)
                finally:
                    conn.event_queue.task_done()
                if id(conn.writer) not in self._clients:
                    return
        except asyncio.CancelledError:
            pass
        finally:
            if conn.event_writer_task is asyncio.current_task():
                conn.event_writer_task = None

    # -- pushes ----------------------------------------------------------------

    def _schedule_push(self) -> None:
        """Called by the host on projection change, from ANY thread. Coalesce
        bursts (a streaming assistant row changes 30×/s) into one repaint per
        runtime-loop tick — pushes are snapshots, so intermediate states
        carry no information."""
        if self._loop is None or self._closed.is_set():
            return
        try:
            self._loop.call_soon_threadsafe(self._push_soon)
        except RuntimeError:  # loop closing
            pass

    def _push_soon(self) -> None:
        # A callback may already be queued when close flips the cross-thread
        # event. Recheck here so shutdown cannot create new work behind itself.
        if self._closed.is_set() or self._push_scheduled:
            return
        self._push_scheduled = True
        self._push_task = asyncio.create_task(self._push_later())

    async def _push_later(self) -> None:
        try:
            # One short delay lets the current event batch fold before snapshot.
            await asyncio.sleep(0.05)
            if not self._closed.is_set():
                await self._push()
        finally:
            self._push_scheduled = False

    def _projection_recipients(self) -> list[_ClientConn]:
        """Who still wants projection *repaints*.

        A client that declared ``events`` AND ``frontend_state`` is a full-TUI
        viewer: it consumes ``frontend_sync`` / ``frontend_update`` / ``event``
        and discards projections (``AttachedSession._dial`` installs
        ``lambda _projection: None``). Pushing 100–900 KB × ~20 Hz onto that
        socket is what filled the kernel buffer and tripped ``_SEND_TIMEOUT_S``.
        The welcome (``_push_to``) still goes to everyone — ``AttachClient.connect``
        reads it as the identity check. Daemon and legacy v3/v4-only attach
        clients keep receiving repaints byte-for-byte.

        Desktop uses the same two flags and the same no-op projection
        callback (confirmed: ``desktop_sessions.py`` consumes events +
        frontend_state, nothing keys on ``op == "projection"`` after welcome).
        """
        return [
            conn
            for conn in list(self._clients.values())
            if not (conn.wants_events and conn.wants_frontend)
        ]

    async def _push(self) -> None:
        """Broadcast projection repaints, preserving daemon bytes exactly.

        Full-TUI attach clients are skipped (see ``_projection_recipients``).
        Phone daemon frames stay byte-identical.
        """
        recipients = self._projection_recipients()
        if not recipients:
            # Detached owners and full-TUI-only viewers have nobody consuming
            # repaints. Avoid building a large payload at every coalesced tick;
            # welcome frames and all subscribed update cadences are unchanged.
            #
            # The warning latch still has to be released: no frame was emitted,
            # so nothing degraded, and leaving it set would swallow the log line
            # for the NEXT genuine degradation after a quiet period — exactly
            # what the once-per-episode latch exists to preserve.
            self._frame_cap_warned = False
            return
        # BUILD AND CAP THE FRAME OFF THE EVENT LOOP. ``_projection_payload``
        # runs ``cap_projection_frame``, which serialises a payload that can sit
        # near the 1 MB wire cap; on the loop that is measured to park the whole
        # runtime — 13 of 50 runtime-stall dumps in one 24 h window hold the loop
        # thread in ``json.dumps -> _frame_bytes -> cap_projection_frame``, and
        # one of those fired the 300 s stall bound and killed the runtime. The
        # build is a pure function of the projection the fold publishes, so it
        # belongs in a worker: ``_push`` is a coalesced repaint, never a
        # request/response, so nothing waits on it within a turn.
        #
        # ONLY THE BUILD MOVES. ``_send_to`` stays on the loop and says why
        # itself (the writer and its lock are loop-owned objects), and the
        # per-connection frames are derived from ``ordinary`` on the loop after
        # the hop returns, so the wire bytes are unchanged.
        ordinary = await asyncio.to_thread(self._projection_payload)
        await asyncio.gather(
            *(self._send_to(conn, self._projection_frame(conn, ordinary)) for conn in recipients)
        )

    def _projection_payload(self) -> dict[str, Any]:
        """The broadcast frame, capped to the soft size limit.

        The daemon's control reader drops any line past 1 MB, so an oversized
        projection is a silently lost repaint — and a flood of them starves
        the daemon loop for every other session. ``cap_projection_frame``
        degrades optional payload tiers (subagent text previews, transcript
        expand details, then the transcript tail) until the frame fits, so a
        busy deep-roster session degrades gracefully instead of wedging the
        whole relay. The projection itself is never mutated.
        """
        from local_operator.mobile.projection import cap_projection_frame

        # RE-DATE BEFORE SERIALIZING, through the fold the EVENTS REACH. The
        # band's age is a reading taken when the phase last moved, and this frame
        # is the moment it becomes an answer to "how long has this been going" —
        # for whoever is attached now, including a phone that just attached
        # mid-phase.
        #
        # It must be the HANDLE's fold, not ``self._projection_sink``: that sink
        # is the runtime's own lazily-built fold over the handle's projection
        # OBJECT, and nothing ever feeds it events — the feeders are the handles'
        # own folds. Re-dating through it stamped its empty state over the live
        # age, so every pushed frame carried ``null`` and the phone withheld its
        # clock for every phase, watched edges included (review round 4, BLOCKER
        # 1). Probed rather than required: a reduced handle has no fold, and the
        # sink contract deliberately stays read-only plus ``set_pending``.
        redate = getattr(self._handle, "redate_from_phase", None)
        if callable(redate):
            redate()
        sink = self._projection_sink
        projection = sink.projection if sink is not None else self._handle.session_projection_seed
        data, degraded = cap_projection_frame(projection)
        if degraded and not self._frame_cap_warned:
            self._frame_cap_warned = True
            logger.warning(
                "session runtime: projection frame for session %s exceeded the "
                "soft cap; degrading optional payload tiers to fit",
                self._record.session_id,
            )
        elif not degraded:
            self._frame_cap_warned = False
        return {"op": "projection", "data": data}

    def _projection_frame(self, conn: _ClientConn, ordinary: dict[str, Any]) -> dict[str, Any]:
        # Projection is exclusively the mobile renderer. Full terminal clients
        # authenticate with this welcome but consume no semantic overlays from it.
        return ordinary

    async def _push_to(self, conn: _ClientConn) -> None:
        """The welcome form of a push: one frame to one connection.

        The ONE frame that may also carry the operator capability's handshake
        proof (issue #1310) — deliberately here and not in ``_projection_frame``,
        which every repaint goes through: the proof is per CONNECTION and belongs
        to the frame that decides whether the client will present anything at
        all. A client that sees no proof (this runtime holds no capability, or
        the client offered no nonce) presents nothing, which is the fail-closed
        reading of a runtime nobody handed one to.

        WHICH frame is the welcome is ``_welcome_frame``'s decision, not this
        method's: a full-TUI/desktop attach reads the projection for its
        identity and discards the payload, and that payload is built and capped
        inline on the serving loop.
        """
        conn.sending_welcome = True
        try:
            frame = self._welcome_frame(conn)
            proof = self._welcome_operator_proof(conn)
            if proof is not None:
                # Salt alongside the proof, because the client needs both
                # nonces to build its own proof and only the PROOF is
                # credential-shaped: the salt is a value it just chose for a
                # connection that will not outlive this list.
                frame["operator_salt"] = conn.operator_salt
                frame["operator_proof"] = proof
            await self._send_to(conn, frame)
        finally:
            conn.sending_welcome = False

    def _welcome_frame(self, conn: _ClientConn) -> dict[str, Any]:
        """The welcome for THIS connection: identity-only, or the full projection.

        The split is by who READS the payload, which is a property of the client
        rather than of the daemon. A connection that asked for the canonical
        frontend AND the raw event stream is a full terminal or the desktop, and
        the only client of that shape builds ``AttachClient`` with
        ``on_projection = lambda _projection: None`` — it consumes the welcome
        for identity and nothing else. Phone and daemon connections asked for
        no canonical frontend, are the clients that render the projection, and
        keep their welcome byte-identical.

        ``None`` from ``_slim_welcome_frame`` (a reduced handle with no seed)
        falls back to the full projection rather than guessing at identity.
        """
        if conn.wants_events and conn.wants_frontend:
            slim = self._slim_welcome_frame()
            if slim is not None:
                return slim
        return self._projection_frame(conn, self._projection_payload())

    def _slim_welcome_frame(self) -> dict[str, Any] | None:
        """The identity-only welcome, or ``None`` when the handle has no seed.

        WHY IT EXISTS. ``_projection_payload`` builds and CAPS a whole projection
        for the welcome, and the attach clients above read none of it — every byte
        past the identity is serialized, walked by the cap tiers and discarded.
        On the fixtures that is 0.7 ms of CPU, which is NOT why this exists: it
        exists because ``_push_to`` calls it INLINE on the serving loop, once per
        connection, and a field dump has caught ``cap_projection_frame`` on that
        thread's stack (``lop-mobile-registrant`` ← ``_push_to`` ←
        ``_on_connection``, runtime-stall-42983.log). A frame nobody reads is not
        worth any chance of that.

        The client accepts this op (``attach_client`` treats ``welcome`` exactly
        as it treats ``projection``) and tolerates the missing keys, which is
        what makes the payload a subset rather than a new contract.
        """
        seed = getattr(self._handle, "session_projection_seed", None)
        if seed is None:
            return None
        return {
            "op": "welcome",
            "data": {name: getattr(seed, name, None) for name in _WELCOME_IDENTITY_FIELDS},
        }

    def _welcome_operator_proof(self, conn: _ClientConn) -> str | None:
        """This connection's handshake proof, or ``None`` when there is none to give.

        MUTUAL, and that direction is the point: the console must be able to
        tell a real runtime from an endpoint that merely has the record's
        ``control_port`` written into it. A rewritten record points the console
        at an impostor, and an impostor cannot compute this proof — it does not
        hold the capability — so the console presents nothing. See
        ``harness/approval._proof`` for the attack this closes.
        """
        if not conn.operator_nonce or not conn.operator_salt:
            return None
        if self._operator_cap is None:
            return None
        return handshake_proof(
            self._operator_cap, client_nonce=conn.operator_nonce, server_salt=conn.operator_salt
        )

    async def _broadcast(self, frame: dict[str, Any]) -> None:
        # Copy the registry: a send failure drops its own entry, and mutating
        # the dict mid-iteration is exactly the failure being handled.
        await asyncio.gather(*(self._send_to(conn, frame) for conn in list(self._clients.values())))

    async def _send_to(self, conn: _ClientConn, frame: dict[str, Any]) -> None:
        """One frame to one connection. A failed send drops ONLY that client
        from the registry (never retried — the reader loop will observe the
        close and its finally is a no-op second removal).

        THE CEILING. Every frame the runtime emits reaches the socket through
        here (``_write_now``'s ``stopping`` announcement is the one exception and
        is constant-size by construction; ``retiring`` carries free-text fields
        and goes through here like everything else), so the line limit is
        enforced here rather than trusted to each family's own guard. A frame
        past ``_MAX_LINE_BYTES`` is not merely large: the peer dials with the
        SAME limit, its ``readline`` raises, and its pump dies — the viewer
        then paints "owner sent a frame too large to read" and the session
        cannot be opened at all. ``cap_projection_frame`` is a SOFT cap that
        degrades optional tiers and can still return an over-limit payload
        (measured at 1,254,249 B against this limit on a 256-sibling roster),
        so the hard bound has to live at the write.

        The check costs one length comparison, not a second serialization: the
        frame is encoded here to be written, and ``_frame_line_bytes`` is handed
        those bytes so the ONE definition of the wire-size rule is used without
        dumping the frame twice on the ~30/s repaint path. What replaces an
        oversized frame depends on what the frame IS — see
        :meth:`_readable_frame`.

        ON THE RUNTIME'S OWN LOOP, and that is enforced rather than assumed: the
        writer and the lock below are loop-owned objects, so a foreign loop
        cannot finish this coroutine once the lock is contended. The why, and the
        CI failure it is there to prevent, are at the guard.
        """
        if not self._on_runtime_loop():
            # CROSS-LOOP PRECONDITION, checked rather than trusted. Both objects
            # the send touches belong to the runtime's loop: ``conn.writer`` is a
            # StreamWriter whose transport and drain waiter were created on it,
            # and ``conn.send_lock`` is an asyncio.Lock, which binds itself to the
            # FIRST loop that contends it (``_get_loop``) and is then unusable
            # from any other.
            #
            # Why this is a hard error and not merely slow: when the lock is
            # contended, the foreign loop parks a waiter future of ITS OWN, and
            # the owner's ``Lock.release()`` completes it with ``set_result`` from
            # the wrong thread — which schedules the callback with plain
            # ``call_soon``, i.e. an append to the other loop's ready deque with
            # no self-pipe write. A loop already parked in ``select()`` is never
            # woken, so the await NEVER returns: the awaiting task, and the
            # process running it, are wedged for good. Measured, not inferred —
            # with the lock genuinely held on the runtime loop, a foreign-loop
            # await of this path was still parked 3.5 s after the lock was
            # released (see the PR's reproduction). That is the shape behind ten
            # cancelled CI shard jobs: the stall watchdog fired on one xdist
            # worker while its siblings armed and never fired.
            #
            # And when the lock is UNCONTESTED the failure is quieter but still
            # real: the fast path never creates a waiter, so the send appears to
            # work while silently binding ``send_lock`` to the foreign loop, after
            # which the runtime's own next contention raises inside the runtime.
            # Either way the honest answer is to fail at the call site, where the
            # mistake is, instead of parking a loop that owns a live session.
            # Mirrors the same precondition on ``aclose``.
            raise RuntimeError(
                "RuntimeServer._send_to() must run on its owning event loop "
                "(conn.send_lock and conn.writer belong to it)"
            )
        timeout = (
            _TUI_SEND_TIMEOUT_S if conn.wants_events and conn.wants_frontend else _SEND_TIMEOUT_S
        )
        async with conn.send_lock:
            try:
                payload = json.dumps(frame).encode()
                close_reason: str | None = None
                size = _frame_line_bytes(frame, payload=payload)
                if size > _MAX_LINE_BYTES:
                    replacement, close_reason = self._readable_frame(conn, frame, size)
                    if replacement is None:
                        return
                    payload = json.dumps(replacement).encode()
                    if _frame_line_bytes(replacement, payload=payload) > _MAX_LINE_BYTES:
                        # Not reachable for the substitutions this file builds
                        # (all are constant-size), but a frame that cannot be
                        # read must never be written, so the guarantee is kept
                        # by construction rather than by arithmetic.
                        logger.error(
                            "session runtime: dropped an unsendable %s frame for session %s "
                            "(%d bytes, over the %d-byte line limit) — its degraded "
                            "replacement does not fit either",
                            replacement.get("op"),
                            self._record.session_id,
                            _frame_line_bytes(replacement, payload=payload),
                            _MAX_LINE_BYTES,
                        )
                        return
                conn.writer.write(payload + b"\n")
                await asyncio.wait_for(conn.writer.drain(), timeout=timeout)
                if close_reason is not None:
                    # AFTER the drain, so the substitute frame — the peer's only
                    # trace of why — is actually on the wire before the close.
                    self._drop_client(conn, reason=close_reason)
            except TimeoutError:
                self._drop_client(conn, reason=f"send timeout ({timeout:.1f}s)")
            except (ConnectionResetError, BrokenPipeError, OSError) as exc:
                self._drop_client(conn, reason=f"send failed: {type(exc).__name__}")

    def _readable_frame(
        self, conn: _ClientConn, frame: dict[str, Any], size: int
    ) -> tuple[dict[str, Any] | None, str | None]:
        """What to write instead of an unreadable frame: ``(frame, close_reason)``.

        By op family, because the honest answer depends on what the frame MEANS
        to the peer — and on whether there is anybody waiting for it:

        * replaceable state the peer re-receives anyway → substitute it;
        * an answer to a request → tell the requester it cannot be sent;
        * a frame that is the BASE of a stream the peer cannot be told about →
          substitute, then close, because a peer waiting on a frame that is
          never coming reports the owner as slow/unresponsive 15 s later, which
          is exactly the misdiagnosis this whole class of bug is made of;
        * already-degraded relay traffic → degrade again (a belt);
        * anything else → drop the frame and keep the connection.

        ``None`` as the frame means "write nothing"; the connection survives
        unless ``close_reason`` is set.
        """
        op = frame.get("op")
        if op in ("projection", "welcome"):
            if not conn.sending_welcome:
                # A REPAINT, not a welcome, so there is no canonical sync behind
                # it to restore what a blank would cost — and dropping an
                # unreadable frame is what the daemon has always done with one
                # (it catches the ValueError and continues; its own comment names
                # the consequence). Pushes are full snapshots, so dropping one
                # applies nothing half-way, and the ERROR below carries op, size
                # and limit.
                #
                # THE RESIDUAL IS REAL, and is written down rather than glossed:
                # this branch fires only when a projection is STILL over the
                # ceiling after every cap tier, which is a property of the
                # SESSION — tier 6's identity rows did not fit, so the label
                # column alone is too large — not a transient spike. The next
                # repaint is therefore over the limit too, nothing supersedes
                # this frame, and the phone holds its last good projection and
                # stops updating live with NO client-side surface (it never sees
                # these frames, so it cannot warn); the only trace is an ERROR per
                # drop at repaint rate. That is the daemon's documented
                # ``stale``/skip behaviour, unchanged.
                #
                # It is still the right choice: a stall is recoverable once the
                # state shrinks or the session is reset, while an identity-only
                # payload would REPLACE the phone's good state with an empty
                # session, restorable only by the daemon's version fence (a lower
                # ``version`` is fenced out — ``mobile/daemon.py``'s staleness
                # check and ``mobile/web/src/store.ts``) — not ours to lean on.
                logger.error(
                    "session runtime: dropped an unreadable %s repaint for session %s "
                    "(%d bytes, over the %d-byte line limit) — a blank one would not "
                    "be restorable; the session stops updating live until a repaint "
                    "fits",
                    op,
                    self._record.session_id,
                    size,
                    _MAX_LINE_BYTES,
                )
                return None, None
            # The WELCOME. Only the mobile renderer consumes a projection at all,
            # and for every other client the identity fields are the whole
            # payload — and the canonical state genuinely follows on the
            # ``frontend_sync`` this same connect path sends next, which is what
            # makes blanking the collections here lossless for the terminal and
            # one lost repaint for the phone (a daemon already holding a
            # projection fences the lower version out entirely).
            logger.error(
                "session runtime: refusing to write an unreadable %s frame for session %s "
                "(%d bytes, over the %d-byte line limit) — sending the identity-only "
                "welcome instead; the canonical state follows on frontend_sync",
                op,
                self._record.session_id,
                size,
                _MAX_LINE_BYTES,
            )
            return {"op": "projection", "data": self._identity_projection()}, None
        if op in ("result", "frontend_sync") or "req" in frame:
            # An answer the requester is waiting for. Telling it the answer
            # cannot be sent is the same contract the ``frontend_sync`` RPC
            # already keeps by raising (see ``_dispatch_client``), and the
            # requester learns instead of the socket dying under it.
            #
            # The connect-time PUSH form of ``frontend_sync`` (no ``req``) has
            # nobody to tell, and that is the one case where substituting the
            # answer is not enough: the viewer sits on a base that never comes,
            # applies no delta (its first one would be refused as a sequence
            # gap) and reports "the runtime is not responding" when its envelope
            # expires — the exact misdiagnosis of a hard bug this code base
            # spent two rounds removing. So THIS one closes: the peer fails at
            # once with a disconnect instead of blaming the owner's liveness,
            # and the ERROR lines here and at the connect path (where the field
            # responsible is known) carry the diagnosis.
            logger.error(
                "session runtime: refusing to write an unreadable %s reply for session %s "
                "(%d bytes, over the %d-byte line limit) — replying with an error frame",
                op,
                self._record.session_id,
                size,
                _MAX_LINE_BYTES,
            )
            replacement = {
                "op": "error",
                "req": frame.get("req"),
                "message": (
                    f"{op or 'frame'} is {size:,} bytes, over the "
                    f"{_MAX_LINE_BYTES:,}-byte socket line limit; the owner cannot send it"
                ),
            }
            if "req" in frame:
                return replacement, None
            return replacement, f"unsendable {op} ({size} bytes)"
        if op in ("event", "frontend_update"):
            # Already fitted at enqueue (``_enqueue_client_frame`` routes both
            # relay families through ``fit_frame_for_wire``, whose terminal step
            # IS this guard), so this is a belt rather than the guard — but the
            # same rule holds: never write what the peer cannot read. A frame
            # that arrives here over the limit is one the fit pass could not
            # refit (an image-free oversize, an unwritable store), and the guard
            # says so at ERROR on the way through.
            return relay_frame_or_degraded(frame, _MAX_LINE_BYTES), None
        logger.error(
            "session runtime: dropped an unsendable %s frame for session %s "
            "(%d bytes, over the %d-byte line limit) with no readable substitute",
            op,
            self._record.session_id,
            size,
            _MAX_LINE_BYTES,
        )
        return None, None

    def _identity_projection(self) -> dict[str, Any]:
        """The smallest projection that still identifies the session.

        A welcome's non-negotiable job is the client's identity check: the
        projection names the conversation the owner is REALLY hosting, and
        ``AttachClient.connect`` refuses anything else or anything malformed.
        The rest of a projection is render payload — no terminal client
        consumes it (``_projection_frame`` returns it untouched and the TUI's
        projection callback is a no-op) and every other viewer receives the
        canonical snapshot on the ``frontend_sync`` that follows immediately.
        Empty collections rather than omitted keys, so the frame stays a valid
        projection of its own op for a client that rebuilds it field by field.
        """
        seed = self._handle.session_projection_seed

        def identity(field: str, fallback: str = "") -> str:
            return str(getattr(seed, field, "") or fallback)

        return {
            "session_id": self._record.session_id,
            "pid": self._record.pid,
            "kind": identity("kind", self._record.kind),
            "conversation_name": identity("conversation_name", self._record.conversation_name),
            "cwd": identity("cwd", self._record.cwd),
            "model_label": identity("model_label", self._record.model_label),
            "transcript": [],
            "todos": [],
            "subagents": [],
            "pending": None,
        }

    async def _send(self, frame: dict[str, Any]) -> None:
        """Broadcast alias kept for the pre-v2 call shape (tests, hosts that
        grabbed a reference before the bump)."""
        await self._broadcast(frame)

    # -- host-facing helpers ----------------------------------------------------

    def _ensure_projection_sink(self) -> ProjectionSink:
        """Return the sink, building the default :class:`ProjectionFold` on
        first need. Idempotent; the counter records only builds this runtime
        performed itself, never an injected sink."""
        sink = self._projection_sink
        if sink is None:
            sink = ProjectionFold(self._handle.session_projection_seed)
            self._projection_sink = sink
            self.projection_sinks_built += 1
            logger.info(
                "session runtime: built projection fold for session %s",
                self._record.session_id,
            )
        return sink

    @property
    def record(self) -> SessionRecord:
        """The discovery record this runtime publishes.

        Read-only by intent — the runtime owns every mutation (the port is
        stamped when the listener binds, the heartbeat rewrites the liveness
        fields), and a caller that reassigned it would leave the publisher
        writing a record nobody else holds.

        Exposed because a host that starts a runtime and must then TELL
        someone where it is (``exec --control`` prints the endpoint to stderr)
        otherwise has to re-read the file the runtime just wrote, or reach for
        ``_record``. Deliberately not the control key by a separate accessor:
        the key rides the record, and the record's 0600 file is the whole
        authorization model, so nothing should be encouraged to copy it out.
        """
        return self._record

    @property
    def record_path(self) -> Path:
        """The file this runtime published its discovery record to.

        Read off the PUBLISHER, never recomputed from the config dir. The two
        only agree by circumstance: the record's directory is pinned when the
        runtime is asked to start, while ``_serve`` reaches its first yield —
        ``await asyncio.start_server`` — before it builds the publisher. A
        config dir that moves during that yield therefore leaves a
        ``config_dir()`` recomputation naming a ``<pid>.json`` no runtime ever
        wrote, and a caller that prints or dials this path has to have the file
        that exists (QA round 2, Q2 forced exactly that window).

        Raises rather than returning ``None``, like :attr:`record`'s habit of
        answering directly: asking an unstarted runtime where its record is has
        no useful answer, and a caller cannot print ``None`` as a path.
        """
        publisher = self._publisher
        if publisher is None:
            raise RuntimeError("this runtime has not published a record yet")
        return publisher.path

    @property
    def projection_sink(self) -> ProjectionSink | None:
        """The sink in use, or ``None`` while no consumer has needed one."""
        return self._projection_sink

    @property
    def fold(self) -> ProjectionFold:
        """The default fold, built on demand. Handles that reach for this want
        the full :class:`ProjectionFold` surface (subagent details, todos);
        an injected sink that is not one is a programming error here."""
        sink = self._ensure_projection_sink()
        if not isinstance(sink, ProjectionFold):
            raise TypeError("an injected projection sink is not a ProjectionFold")
        return sink

    def set_pending(self, pending: Any | None) -> None:
        """Set mobile pending state and canonical gate state for reduced hosts.

        Production TUI handles publish directly when their real widget mounts;
        this bridge keeps owned/reduced handles on the same contract.
        """
        self._ensure_projection_sink().set_pending(pending)
        frontend = getattr(self._handle, "_frontend", None)
        mutate = getattr(frontend, "mutate", None)
        if callable(mutate):
            payload = pending.to_json() if pending is not None else None
            if payload is not None:
                # Same stamp as `ServingSessionHandle._publish_pending_gate`,
                # through the handle's own helper so the privacy gate cannot
                # hold on one publication site and not the other. Reduced hosts
                # reach the gate contract through here, and a desktop banner
                # raised from one of them must be as triageable as any other.
                namer = getattr(self._handle, "_notifiable_session_name", None)
                payload["session_name"] = namer() if callable(namer) else ""
            mutate(pending_gate=payload)
        self._schedule_push()


#: The pre-move name. ``Registrant`` was mobile-era vocabulary for the same
#: object; it stays bound here (and re-exported from
#: ``local_operator.mobile.registrant``) so the rename costs no call site and
#: an out-of-tree caller keeps working. New code should use ``RuntimeServer``.
Registrant = RuntimeServer
