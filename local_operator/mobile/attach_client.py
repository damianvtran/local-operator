"""The attach client: a follower terminal's half of the control socket.

Every interactive ``lop`` process hosts a session runtime
(:class:`~local_operator.session.runtime.server.RuntimeServer`)
whose loopback socket is the phone's window onto the session. This module is
the SAME socket seen from a second terminal: ``/resume`` of a session another
process owns dials that owner and renders its projection repaints, steering
through the same ops the phone uses. One socket, N front ends — a second
protocol would drift from the first, so there is none.

Design constraints baked in:

- **No auto-reconnect.** Owner death (socket EOF) is terminal for the
  CONNECTION — never papered over by redialing a pid that may have been
  reused. The callback fires once and the client is dead. What the HOST does
  next changed in v4: ``RemoteSession`` runs a silent reattach-or-takeover
  loop (re-discover the owner, or become it through the normal resume
  factory) instead of showing a decision card, but each loop iteration still
  builds a FRESH client against a freshly discovered record.
- **Identity over pid trust.** ``live_session_owner`` cannot probe pids on
  Windows, and a recycled pid anywhere defeats pid trust. After auth the
  registrant sends a full projection unprompted; the client requires that
  projection's ``session_id`` to match the one the user asked for before
  declaring the attach good. A mismatch means the owner rebound away — the
  caller surfaces the graceful refusal copy.
- **Protocol gate before dialing.** A v1 registrant treats ANY authenticated
  dial as THE daemon and evicts the real one; requiring ``record.protocol
  >= 2`` turns that hazard into the graceful degradation path instead.

Stdlib-only plus the mobile wire types: the CLI imports this lazily on the
owned-resume branch only, keeping ``resume.py`` and the startup path light.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import uuid
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Callable, NamedTuple, Sequence

from local_operator.mobile.types import (
    PROTOCOL_VERSION,
    ContinuationCommand,
    SessionProjection,
    SessionRecord,
    _projection_from_json,
)
from local_operator.session.runtime.registry import scan
from local_operator.session.runtime.types import DESKTOP_WATCH_CAPABILITY

#: How long to wait for an ack/error matching a request id. Mirrors the
#: daemon's ``request`` timeout: long enough for a turn-boundary op (prompt
#: acquires the turn lock) on a busy owner, short enough that a wedged owner
#: surfaces as an error rather than a hang.
ACK_TIMEOUT_S = 15.0

#: Disconnect reason marking a DELIBERATE stop, as opposed to owner death.
#: Consumers compare against this exact string to decide whether to recover.
STOPPED_REASON = "owner stopped the session"

#: Disconnect reason for a PLANNED refresh: the owner retired itself because
#: ``lop-update`` put a newer build on disk, and the next engage should spawn
#: from it. Distinct from :data:`STOPPED_REASON` on purpose — a stop parks the
#: viewer in the stopped state (``/resume`` reopens it); a refresh means "a
#: fresh runtime is owed, engage one now". Nothing was interrupted: the owner
#: retires only when idle (``OwnedSessionHandle.may_refresh``).
RETIRING_REASON = "owner retired for a newer build"

#: Maximum bytes in one frame. Must equal the server's ``_MAX_LINE_BYTES``:
#: the writer refuses to exceed it and the reader refuses to read past it, so
#: two different numbers would mean a frame the owner considers sendable is one
#: this client cannot read.
_READ_LIMIT_BYTES = 1 << 20

#: Disconnect reason for a frame too large to read. Distinct from owner death
#: because the remedy is different in kind: the owner is alive and healthy, and
#: what failed is our ability to parse what it sent. Kept a named constant so a
#: host can tell the two apart rather than matching on prose.
OVERSIZED_FRAME_REASON = "owner sent a frame too large to read"

logger = logging.getLogger(__name__)


class _OversizedFrame(Exception):
    """A readline overrun, re-raised under its own type so the pump's outer
    handler cannot confuse it with a ``ValueError`` a frame CALLBACK raised —
    the two mean opposite things (owner sent too much vs. this side refused
    what it sent) and were logged as the same overrun before this existed."""


class OversizedRequest(ValueError):
    """This side refused to SEND a frame the owner could not have read.

    The outbound twin of :class:`_OversizedFrame`, and the whole point is that
    it is raised INSTEAD of a write. Writing an over-limit line does not fail:
    it succeeds, the owner's ``readline`` raises on the far side, and the
    connection dies — so the caller's request is answered by a dead socket
    rather than an error, which is exactly the session death this guards
    (pasting a large screenshot exited ``lop`` and forced ``lop --resume``).

    A ``ValueError`` and not a ``ConnectionError``, deliberately: the socket is
    healthy and redialling fixes nothing. The TUI's prompt worker surfaces the
    message as a notice and puts the draft back in the composer, which is what
    makes the refusal actionable rather than a loss.
    """


class _RefitReport(NamedTuple):
    """What the wire refit did to ONE image, for the surface that must say so.

    ``marker`` is the number the USER sees on their composer chip, not the
    image's wire position \u2014 see :func:`_refit_images` for why the two differ.

    ``downscaled`` is the only field that decides whether anything is said at
    all. The refit's common outcome is a codec swap that keeps every pixel
    (the composer already bounds a paste to 1024px, so the descending rungs are
    unreachable for ordinary messages), and a notice on that would be pure
    noise on a routine gesture. Losing pixels is different in kind: measured at
    the 768px rung it costs ~40% edge energy and makes digits misread, which is
    exactly the failure a user pastes a screenshot to avoid (design round 1,
    D2). So the dimensions are carried to be shown, and the codec is not.
    """

    marker: int
    width: int
    height: int
    source_width: int
    source_height: int

    @property
    def downscaled(self) -> bool:
        """Did the refit cost PIXELS, as opposed to merely changing codec?

        Pixel counts rather than the ``WxH`` strings, matching
        ``editor._was_downscaled``: the mark is a claim about fidelity and has
        to be tested as one.
        """
        if not all((self.width, self.height, self.source_width, self.source_height)):
            return False
        return self.width * self.height < self.source_width * self.source_height


#: Slack left over the MEASURED frame overhead, to absorb the difference
#: between the empty-image frame and the encoded one.
#:
#: The overhead itself is no longer guessed — :func:`_frame_overhead_bytes`
#: serialises the real frame with its images emptied, so the op name, the
#: ``req`` id, any command id and the user's ACTUAL text are counted rather
#: than bounded. A fixed 64 KiB reserve was the bug: ``clipboard.py`` admits
#: pastes up to ``MAX_CLIPBOARD_TEXT_BYTES`` (1 MiB), so any prompt whose text
#: passed ~64 KiB had its images fitted against a budget that was never
#: available, the post-refit re-measure caught the overflow, and the whole
#: message was refused — including the ~780 KB screenshot this module's
#: docstring names as the motivating bug (review round 1, MAJOR-1).
#:
#: What remains is per-image JSON punctuation the emptied frame does not carry:
#: the ``data_b64``/``mime_type`` keys, quotes, braces and commas, ~40 bytes an
#: image plus base64's own padding. 4 KiB covers a 16-attachment message an
#: order of magnitude over, and the exact frame is still measured again after
#: the refit — so this only has to be non-negative, not precise.
_FRAME_ENCODING_SLACK_BYTES = 4 * 1024


#: Below this much room for ALL the images, the text is the problem and the
#: refusal must say so rather than blaming an attachment.
#:
#: The refit's tightest rung is a 384px JPEG at quality 85, which lands around
#: 20-30 KB of base64 for ordinary content. 32 KiB is therefore "not even one
#: image at its smallest could fit here": under it the per-image refusal is
#: guaranteed and would quote a share rounding to ``0.0 MB``, pointing the user
#: at a screenshot when only shortening the prompt can help.
_MIN_VIABLE_IMAGE_BUDGET_BYTES = 32 * 1024


#: The refit that the CURRENT task performed, published for the caller that has
#: to tell the user about it. See :func:`taken_refit_report`.
#:
#: A context variable rather than a return value because the refit happens four
#: call frames below the surface that renders transcripts — ``fit_request_frame``
#: is reached through ``AttachClient._request_frame`` → ``send_command`` →
#: ``RemoteSession.prompt``, each of which returns a receipt string with no room
#: for a second value, and widening all four signatures to carry a UI detail
#: through the transport would put presentation concerns in three layers that
#: currently have none.
#:
#: Context propagates the RIGHT way for this: a value set in a coroutine is
#: visible to the task that awaited it, while a task spawned with
#: ``create_task`` gets a COPY — so two concurrent sends can never read each
#: other's report. Verified, not assumed. The reader consumes it, so a stale
#: report cannot outlive the send that produced it.
_REFIT_REPORT: ContextVar[tuple[_RefitReport, ...] | None] = ContextVar(
    "local_operator_attach_refit_report", default=None
)


async def fit_request_frame(frame: dict[str, Any]) -> dict[str, Any]:
    """Return ``frame`` sized to fit the socket, refitting its images if needed.

    THE BUG THIS CLOSES. The owner's control socket reads one JSON line per
    frame with ``limit=_MAX_LINE_BYTES``; a line over that makes its
    ``readline`` raise and (before the sibling fix in ``RuntimeServer``) killed
    the reader loop and the connection. Nothing on this side checked, so
    ``prompt``/``steer``/``slash`` put the user's images on the wire fully
    base64-encoded and unguarded, and ONE pasted screenshot over ~780 KB of
    source severed the socket at the moment of sending. Measured on this
    machine: a 1672x941 render is 1.23 MB of base64 after the composer's own
    ingest bound, and two ordinary screenshots together are 1.1 MB, so the
    frames that break it are ordinary rather than pathological.

    WHY REFIT RATHER THAN REFUSE. Pasting screenshots is a routine gesture, and
    the sizes above are routine too — a hard rejection would break the feature
    to protect the transport. So each image is re-encoded to fit
    (:func:`~local_operator.imaging.refit_image_to_budget`, JPEG at full
    resolution first, downscale only if that is not enough). Only an image that
    cannot fit even at its tightest rung is refused, named by the MARKER NUMBER
    on the user's own composer chip so they know which attachment to drop.

    WHY THE BUDGET IS SHARED ACROSS THE IMAGES. The line limit applies to the
    whole frame, so N images have to fit TOGETHER; refitting each against the
    full limit would pass N times and still overflow. What is left after the
    frame's MEASURED overhead is therefore split between them — see
    :func:`_refit_images` for why the split reclaims rather than divides flat.

    A COROUTINE because the refit decodes and re-encodes images — ~315 ms for a
    20 MP frame — and every caller is on an event loop. The work goes to a
    thread; a frame with no images (the overwhelming majority: every ack, every
    watch, every projection request) returns after one length check without
    ever reaching it.

    Publishes what the refit COST to :data:`_REFIT_REPORT` for the front end to
    render — see :func:`taken_refit_report`. Always, including the empty tuple
    when nothing was resized, so a reader cannot mistake a previous send's
    report for this one's.

    Raises :class:`OversizedRequest` when the frame cannot be made to fit,
    which is strictly better than the alternative: the caller learns its
    request was not sent while the connection is still alive.
    """
    images = frame.get("images")
    encoded_size = len(json.dumps(frame).encode()) + 1  # the socket writes a "\n" too
    if encoded_size <= _READ_LIMIT_BYTES:
        # The common path, and the ONLY one that leaves the report untouched:
        # nothing was refitted, so there is nothing for a caller to consume and
        # a `set` here would cost every ack and projection request a context
        # write. `taken_refit_report` treats "absent" as "nothing happened".
        return frame
    if not isinstance(images, list) or not images:
        # Nothing bulky to shrink, so the text itself is over the limit. Say so
        # with both numbers rather than truncating: a prompt silently cut in
        # half is worse than one that was not sent, because the damage is
        # invisible until the model answers about the wrong thing.
        raise OversizedRequest(
            f"this message is {_megabytes(encoded_size)} and the limit is "
            f"{_megabytes(_READ_LIMIT_BYTES)}; shorten it and send again"
        )
    # MEASURED, not reserved. Serialising the frame with its images emptied
    # counts the op, the ``req``, any command id and the user's real text, so a
    # 100 KiB prompt beside a screenshot is budgeted for instead of being
    # refused against a 64 KiB constant that was never checked against it
    # (review round 1, MAJOR-1).
    overhead = _frame_overhead_bytes(frame)
    budget = max(0, _READ_LIMIT_BYTES - overhead - _FRAME_ENCODING_SLACK_BYTES)
    if budget < _MIN_VIABLE_IMAGE_BUDGET_BYTES:
        # THE TEXT IS THE BULK, and no image of any size would change that. The
        # per-image refusal below would technically fire, but it would blame the
        # attachment and quote a "0.0 MB budget" the user can do nothing with —
        # telling them to remove a screenshot when shortening the prompt is the
        # only thing that can work. Raised before the refit rather than after so
        # a doomed re-encode of several megabytes is not spent to reach the same
        # answer.
        # "no room" rather than a figure once the remainder rounds away: a
        # sentence ending "leaving only 0 KB" reads as a bug in the message.
        room = f"only {_megabytes(budget)}" if budget >= 1024 else "no room"
        raise OversizedRequest(
            f"this message's text alone fills {_megabytes(overhead)} of the "
            f"{_megabytes(_READ_LIMIT_BYTES)} limit, leaving {room} for its "
            "attachments; shorten the text or send the images on their own"
        )
    fitted, report = await asyncio.to_thread(_refit_images, images, budget)
    candidate = {**frame, "images": fitted}
    refitted_size = len(json.dumps(candidate).encode()) + 1
    if refitted_size > _READ_LIMIT_BYTES:
        # Every image reached its tightest rung and the frame is STILL over, so
        # the text is the bulk. Measured on the real frame rather than assumed,
        # so the refusal names the size the user can actually act on.
        raise OversizedRequest(
            f"this message is {_megabytes(refitted_size)} even after its images were "
            f"resized, and the limit is {_megabytes(_READ_LIMIT_BYTES)}; shorten the "
            "text or send fewer attachments"
        )
    _REFIT_REPORT.set(report)
    logger.warning(
        "attach client: %s frame was %d bytes, over the %d-byte line limit; "
        "resized %d image(s) to %d bytes so the message could be sent; "
        "%d of them lost pixels",
        frame.get("op", "request"),
        encoded_size,
        _READ_LIMIT_BYTES,
        len(images),
        refitted_size,
        sum(1 for entry in report if entry.downscaled),
    )
    return candidate


def taken_refit_report() -> tuple[_RefitReport, ...]:
    """CONSUME what the last refit on this task cost, for the surface showing it.

    Taken rather than read, so one report is rendered exactly once: the front
    end asks after a send returns, and a later send that refits nothing must not
    find this one still sitting there and mark an untouched image as resized.

    Empty when the send needed no refit at all, which is the overwhelmingly
    common case — callers can treat empty as "say nothing".
    """
    report = _REFIT_REPORT.get()
    _REFIT_REPORT.set(None)
    return report or ()


def _frame_overhead_bytes(frame: dict[str, Any]) -> int:
    """Bytes this frame costs with its images emptied — everything but payloads.

    The images are replaced by EMPTY BLOCKS rather than dropped, so the keys and
    punctuation of the list itself are counted; only the base64 the refit is
    about to resize is excluded. Serialising the real frame is what makes the
    budget track the user's actual text instead of a constant that a 100 KiB
    paste silently invalidates.

    Falls back to the emptied-list encoding if a block is not a dict — those are
    passed through untouched by the refit, so counting them as empty would
    under-budget by their own size; they are kept verbatim here for that reason.
    """
    images = frame.get("images")
    if not isinstance(images, list):
        return len(json.dumps(frame).encode()) + 1
    hollow = [{**image, "data_b64": ""} if isinstance(image, dict) else image for image in images]
    return len(json.dumps({**frame, "images": hollow}).encode()) + 1


def _megabytes(size_bytes: int) -> str:
    """``size_bytes`` in the scale a person reads sizes in.

    ONE unit system across the whole sentence family. The text refusal printed
    raw byte counts (``1,341,208 bytes``) while the image one printed MB, so two
    refusals raised from the same function asked the user to compare figures in
    different scales (design round 1, D1).

    MB down to 0.1 and KB below it, because a one-decimal MB renders every small
    figure as ``0.0 MB`` — and the figures that go small here are exactly the
    per-image budgets a refusal is trying to explain. A budget the sentence
    prints as zero tells the user their image did not fit in nothing, which is
    not a fact they can act on.
    """
    megabytes = size_bytes / (1024 * 1024)
    if megabytes < 0.1:
        return f"{size_bytes / 1024:.0f} KB"
    return f"{megabytes:.1f} MB"


def _refit_images(
    images: list[Any], budget_bytes: int
) -> tuple[list[Any], tuple[_RefitReport, ...]]:
    """Fit every image block into ``budget_bytes`` of base64 TOTAL, or refuse by name.

    Runs in a thread (see :func:`fit_request_frame`). A block that is not a
    dict, or carries no ``data_b64``, is passed through untouched: the owner
    already drops unusable blocks (``_images_from_wire``), and inventing a
    refusal for one here would fail a send over something the owner would have
    ignored. Its bytes still count against the budget, because they still ride
    the frame.

    THE BUDGET IS RECLAIMED, NOT DIVIDED FLAT. An equal ``budget // len(images)``
    share is the obvious split and it wastes the frame: a 32x32 icon reserved
    the same share as a 1600x1000 screenshot and handed nothing back, so the
    screenshot lost half its pixels while 59% of the frame went unspent (review
    round 1, MINOR-2). Images are therefore fitted SMALLEST FIRST, each against
    an equal share of what REMAINS divided by the images still to come, and
    whatever a small image does not spend is immediately available to the
    larger ones behind it. Smallest-first is what makes the reclaim monotone:
    an image under its share always passes untouched, so the leftover only ever
    grows as the walk proceeds.

    This keeps the property the flat split was chosen for \u2014 a message shrinks
    evenly rather than refusing its last attachment \u2014 because an image that
    genuinely needs more than its share still only gets its share when every
    other image is equally hungry.

    Returns the fitted blocks in their ORIGINAL order (the wire order the owner
    binds to markers), plus one :class:`_RefitReport` per image that was
    actually re-encoded.
    """
    from local_operator.imaging import (
        ImageUnreadable,
        refit_image_to_budget,
        sniff_image,
    )

    fitted: dict[int, Any] = {}
    report: list[_RefitReport] = []
    # Wire position -> the number on the user's composer chip. They are NOT the
    # same: marker numbers do not renumber when an attachment is deleted, so
    # after pasting three images and backspacing two the survivor is `[Image
    # #3]` while its wire position is 1 \u2014 and the refusal said "image 1",
    # pointing at a chip that is not on screen (design round 1, D4). The block
    # carries its own marker when the composer knows one; position is the
    # fallback for producers that do not set it (the phone relay).
    markers = {
        position: (
            int(image["marker"])
            if isinstance(image, dict)
            and isinstance(image.get("marker"), int)
            and image["marker"] > 0
            else position + 1
        )
        for position, image in enumerate(images)
    }
    # Smallest first, by the bytes each block actually contributes. The ORDER of
    # the walk only; `fitted` is re-assembled by wire position below.
    order = sorted(
        range(len(images)),
        key=lambda position: len(_image_payload(images[position])),
    )
    remaining = budget_bytes
    for walked, position in enumerate(order):
        image = images[position]
        data_b64 = _image_payload(image)
        # Everything still to fit, this one included, shares what is left.
        share = max(0, remaining) // max(1, len(order) - walked)
        if not isinstance(image, dict) or not data_b64:
            # Passed through, but still charged: these bytes ride the frame
            # whether or not this function can do anything about them.
            fitted[position] = image
            remaining -= len(data_b64)
            continue
        mime_type = str(image.get("mime_type") or "image/png")
        try:
            result = refit_image_to_budget(data_b64, mime_type, share)
        except ImageUnreadable as exc:
            # A DIFFERENT SENTENCE FROM "too large", deliberately: these bytes
            # would not have been sendable at any size, so telling the user to
            # shrink them sends them off fixing the wrong thing.
            raise OversizedRequest(
                f"image {markers[position]} could not be sent because {exc}; "
                "remove it and send again"
            ) from exc
        if result is None:
            # NAMES THE CEILING THAT ACTUALLY APPLIED, and measures the IMAGE.
            # `len(data_b64)` is the base64, which inflates 4/3 \u2014 reporting it
            # told a user their 2.4 MB screenshot was "3.2 MB", overstating by a
            # third against the number their file manager shows. And a size with
            # no scale beside it answers nothing: "remove it" is the only move
            # the sentence leaves, when the real ceiling is a per-image share
            # that says whether cropping or sending it alone would work (design
            # round 1, D1).
            detail = (
                f"this message's {_megabytes(share)} per-image budget "
                f"({len(images)} attachments)"
                if len(images) > 1
                else f"this message's {_megabytes(share)} budget"
            )
            raise OversizedRequest(
                f"image {markers[position]} is {_megabytes(_decoded_size(data_b64))} and will "
                f"not fit in {detail} even at its smallest size; remove it and send again"
            )
        refitted_b64, refitted_mime = result
        fitted[position] = {**image, "data_b64": refitted_b64, "mime_type": refitted_mime}
        remaining -= len(refitted_b64)
        if refitted_b64 != data_b64:
            # Only a block that actually changed is reported, and only its
            # DIMENSIONS decide whether the user is told (see `_RefitReport`).
            # Sniffed from the bytes on both sides rather than trusting the
            # rung: the ladder's first rung re-encodes at unchanged dimensions,
            # so the codec-only case must measure as "not downscaled".
            source = sniff_image(_decode_quietly(data_b64))
            delivered = sniff_image(_decode_quietly(refitted_b64))
            if source is not None and delivered is not None:
                report.append(
                    _RefitReport(
                        marker=markers[position],
                        width=delivered.width or 0,
                        height=delivered.height or 0,
                        source_width=source.width or 0,
                        source_height=source.height or 0,
                    )
                )
    return [fitted[position] for position in range(len(images))], tuple(report)


def _image_payload(image: Any) -> str:
    """The base64 an image block contributes to the frame, or ``""``."""
    if not isinstance(image, dict):
        return ""
    data_b64 = image.get("data_b64") or ""
    return data_b64 if isinstance(data_b64, str) else ""


def _decoded_size(data_b64: str) -> int:
    """The IMAGE's size in bytes, from its base64 length, without decoding it.

    Base64 is 4 characters per 3 bytes plus padding, so the decoded size is
    what the user recognises as their file and the encoded length is a third
    larger. Arithmetic rather than a decode because this runs on a refusal
    path holding megabytes that are about to be discarded.
    """
    padding = data_b64[-2:].count("=") if data_b64 else 0
    return max(0, (len(data_b64) * 3) // 4 - padding)


def _decode_quietly(data_b64: str) -> bytes:
    """``data_b64`` decoded, or empty bytes \u2014 a sniff failure is never fatal.

    Only feeds :func:`~local_operator.imaging.sniff_image` for the report, and
    a report is a convenience: bytes that will not decode here have already
    been through the refit successfully, so the right outcome is one less
    caption, never a failed send.
    """
    try:
        return base64.b64decode(data_b64, validate=True)
    except Exception:  # noqa: BLE001 \u2014 a missing caption must never fail a send
        return b""


def find_owner_record(config_dir: Path, session_id: str) -> tuple[SessionRecord | None, int | None]:
    """Locate the discovery record of the live process hosting ``session_id``.

    Returns ``(record, owner_pid)``. The normal case matches a record whose
    ``session_id`` equals the ask. The rebind-race fallback: the record's
    session_id is re-stamped only every heartbeat (15s), so when no record
    matches but the claim marker names a live pid, that pid's record is
    returned anyway and the welcome projection's identity check (in
    :meth:`AttachClient.connect`) arbitrates — a stale match costs one refused
    dial, never a wrong attach.

    ``(None, pid)`` means an owner exists but no usable record does (old
    binary, registrant failed to start): the caller degrades gracefully.
    ``(None, None)`` means no owner at all.
    """
    from local_operator.resume import live_session_owner

    owner = live_session_owner(config_dir, session_id)
    if owner is None:
        return None, None
    best: SessionRecord | None = None
    fallback: SessionRecord | None = None
    try:
        for record, state in scan(config_dir):
            if state != "live":
                continue
            if record.pid == owner and record.session_id == session_id:
                best = record
                break
            if record.pid == owner:
                fallback = record
    except OSError:
        return None, owner
    if best is not None and best.protocol >= 2:
        return best, owner
    # A v1 record is not dialable (see module docstring) — report the owner so
    # the caller can print the refusal naming it.
    if best is not None or fallback is not None:
        return None, owner
    return None, owner


class AttachClient:
    """One authenticated ``attach`` connection to a live session's registrant.

    The host supplies projection/disconnect callbacks (and, in v4 events mode,
    raw event + sync callbacks). All fire on the client's reader task, so a UI
    host must marshal widget work onto its message pump. The client is
    single-use: after ``on_disconnected`` it is dead by design.
    """

    def __init__(
        self,
        on_projection: Callable[[SessionProjection], None],
        on_disconnected: Callable[[str], None],
        *,
        events: bool = False,
        on_event: Callable[[dict[str, Any]], None] | None = None,
        frontend_state: bool = False,
        display_window: bool = False,
        locality: str = "local",
        slash_consumers: Sequence[str] | None = None,
        on_frontend_sync: Callable[[dict[str, Any]], None] | None = None,
        on_frontend_update: Callable[[dict[str, Any]], None] | None = None,
        surface: str = "terminal",
    ) -> None:
        self._surface = surface
        self._on_projection = on_projection
        self._on_disconnected = on_disconnected
        if locality not in ("local", "remote"):
            raise ValueError("attach locality must be local or remote")
        # A phone viewer reaches this loopback socket through a relay. Keep
        # the physical user's locality explicit; a local socket is not proof
        # that browser-opening or desktop-only actions are appropriate.
        self._locality = locality
        # v4 events mode: subscribe to the owner's raw AgentEvent relay. The
        # callbacks receive the WIRE dicts — deserialization back into concrete
        # AgentEvent subclasses is RemoteSession's job, so this transport stays
        # pydantic-free and cheap to import (module docstring contract).
        self._events = events
        self._on_event = on_event
        self._frontend_state = frontend_state
        self._display_window = display_window
        # Which action-carrying slash receipts THIS client renders itself (see
        # ``SLASH_ACTION_RECEIPTS``). Declaring them is what stops the runtime
        # from also submitting the request: a client that says nothing is
        # treated as one built before the field, whose request the runtime
        # completes on its behalf. ``None`` is therefore meaningfully
        # different from ``[]`` only to a reader of the frame -- both mean the
        # type was not declared, which is the one rule the runtime applies.
        self._slash_consumers = list(slash_consumers) if slash_consumers is not None else None
        self._on_frontend_sync = on_frontend_sync
        self._on_frontend_update = on_frontend_update
        self._frontend_epoch: str | None = None
        self._frontend_sequence: int | None = None
        self._reader: asyncio.StreamReader | None = None
        self._writer: asyncio.StreamWriter | None = None
        self._reader_task: asyncio.Task[None] | None = None
        self._pending: dict[Any, asyncio.Future[dict[str, Any]]] = {}
        self._req_seq = 0
        self._session_id = ""
        self._attention_supported = False
        self._connected = False

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def supports_completion_ack(self) -> bool:
        return self.connected and self._attention_supported

    async def connect(self, record: SessionRecord, session_id: str) -> None:
        """Dial, authenticate as an attach client, and verify identity.

        Raises on any failure (dial refused, auth rejected, welcome identity
        mismatch, protocol too old): the caller collapses every one to the
        graceful refusal copy — a user re-running the command is the only
        retry mechanism, by design.
        """
        if self._surface == "desktop" and DESKTOP_WATCH_CAPABILITY not in record.capabilities:
            raise ConnectionError("This session needs a runtime update before desktop attachment.")
        if record.protocol < 2:
            raise ConnectionError(f"owner runs protocol v{record.protocol}; attach needs >= 2")
        self._session_id = session_id
        self._attention_supported = "completion-ack-v1" in record.capabilities
        try:
            reader, writer = await asyncio.open_connection(
                "127.0.0.1", record.control_port, limit=_READ_LIMIT_BYTES
            )
        except OSError as exc:
            raise ConnectionError(f"owner socket unreachable: {exc}") from exc
        self._reader = reader
        self._writer = writer
        # The owner needs both facts and they answer different questions:
        # ``locality`` (upstream) says where the client physically is, while
        # ``surface`` says which interface it presents — a desktop watcher
        # negotiates notification/visibility leases that a plain attach does not.
        auth: dict[str, Any] = {
            "key": record.control_key,
            "client": "attach",
            "locality": self._locality,
        }
        if self._surface == "desktop":
            auth["surface"] = "desktop"
        if self._events:
            # v4 capability flag. A v3 owner ignores unknown auth fields and
            # simply never sends event frames — the caller gates on
            # ``record.protocol >= 4`` before relying on the relay.
            auth["events"] = True
        if self._frontend_state:
            auth["frontend_state"] = True
        if self._display_window and "display-history-window-v1" in record.capabilities:
            auth["display_window"] = True
        if self._slash_consumers is not None:
            # Additive and advisory, exactly the shape ``events`` and
            # ``frontend_state`` are: an older owner ignores the unknown auth
            # field and behaves as it always did, and no PROTOCOL_VERSION bump
            # is warranted for a field nobody is required to read.
            auth["slash_consumers"] = list(self._slash_consumers)
        writer.write(json.dumps(auth).encode() + b"\n")
        await writer.drain()
        # The welcome projection doubles as the identity check: it names the
        # conversation the OWNER is actually hosting right now, which is the
        # fact the user cares about and the one a pid cannot prove.
        try:
            first = await asyncio.wait_for(reader.readline(), timeout=ACK_TIMEOUT_S)
        except TimeoutError as exc:
            raise ConnectionError("owner did not send its state") from exc
        except ValueError as exc:
            # The same overrun the pump handles, on the WELCOME frame — the one
            # read that happens before the pump exists. Named rather than left
            # to surface as a bare ValueError from a connect() call, because
            # every caller of connect() already collapses ConnectionError into
            # its refusal copy.
            raise ConnectionError(OVERSIZED_FRAME_REASON) from exc
        if not first:
            raise ConnectionError("owner closed the connection")
        try:
            frame = json.loads(first.decode("utf-8", "replace"))
        except ValueError as exc:
            raise ConnectionError("owner sent a malformed frame") from exc
        if frame.get("op") not in ("projection", "welcome"):
            raise ConnectionError(f"owner replied {frame.get('op')!r}, not its state")
        projection = _projection_from_json(frame.get("data") or {}, record)
        if projection.session_id != session_id:
            raise ConnectionError(f"owner moved to another conversation ({projection.session_id})")
        self._connected = True
        self._reader_task = asyncio.get_running_loop().create_task(self._pump())
        # Deliver the welcome synchronously so the host paints before any
        # later repaint can race it.
        self._on_projection(projection)

    async def _pump(self) -> None:
        """Read frames until EOF; route projections and match acks by req id."""
        reader = self._reader
        assert reader is not None
        reason = "owner exited"
        try:
            while True:
                try:
                    line = await reader.readline()
                except ValueError as exc:
                    # ``StreamReader.readline`` raises ValueError (via
                    # LimitOverrunError) when one frame exceeds the connection's
                    # ``limit``. It is NOT a transport failure and it is not
                    # recoverable by reading on: the oversized line stays in the
                    # buffer, so every subsequent read raises the same way.
                    #
                    # Before this it fell through as an unhandled task exception
                    # that killed the pump silently, and the host — which only
                    # ever learns about a dead connection through
                    # ``on_disconnected`` — kept waiting for a sync that could
                    # never arrive, timed out after 15 s, and degraded to a
                    # runtime-less cold session. A hard bug that presented as a
                    # slow owner. Report it as its own reason so the host can
                    # say what actually happened instead of blaming the owner's
                    # liveness.
                    #
                    # Caught HERE, around the read alone, and not around the
                    # whole loop: the frame callbacks below raise ValueError
                    # too (a follower store refusing an update as "not the next
                    # state sequence", pydantic validation of a frame), and
                    # when the outer handler owned every ValueError those were
                    # logged as "owner sent a frame larger than the limit" — a
                    # 104 KB frame reported as an overrun (#573's viewer).
                    raise _OversizedFrame() from exc
                if not line:
                    break
                try:
                    frame = json.loads(line.decode("utf-8", "replace"))
                except ValueError:
                    continue
                op = frame.get("op")
                if op in ("projection", "welcome"):
                    try:
                        # pid=0 record: the attach screen keys nothing on the
                        # record's pid (it reads the projection's own), and
                        # building a fake record per repaint would imply the
                        # record carries truth it does not.
                        self._on_projection(
                            _projection_from_json(
                                frame.get("data") or {},
                                SessionRecord(
                                    pid=0,
                                    kind="tui",
                                    session_id="",
                                    conversation_name="",
                                    cwd="",
                                    model_label="",
                                    control_port=0,
                                    control_key="",
                                    protocol=PROTOCOL_VERSION,
                                ),
                            )
                        )
                    except Exception:  # noqa: BLE001 — a malformed push must not kill the pump
                        continue
                elif op == "event":
                    # v4 relay frame. Deliver the raw dict; a callback failure
                    # must not kill the pump (same contract as projections).
                    if self._on_event is not None:
                        try:
                            self._on_event(frame.get("data") or {})
                        except Exception:  # noqa: BLE001
                            continue
                elif op == "frontend_sync":
                    data = frame.get("data") or {}
                    epoch = data.get("epoch")
                    sequence = data.get("sequence")
                    if not isinstance(epoch, str) or not isinstance(sequence, int):
                        raise ConnectionError("malformed frontend sync")
                    self._frontend_epoch = epoch
                    self._frontend_sequence = sequence
                    if self._on_frontend_sync is not None:
                        self._on_frontend_sync(data)
                elif op == "frontend_update":
                    data = frame.get("data") or {}
                    epoch = data.get("epoch")
                    sequence = data.get("sequence")
                    expected = (
                        (self._frontend_sequence + 1)
                        if self._frontend_sequence is not None
                        else None
                    )
                    if epoch != self._frontend_epoch or sequence != expected:
                        # Deltas are not replacement-safe: every sequence must
                        # arrive. Closing forces a fresh v5 snapshot rather than
                        # continuing with silently incomplete canonical state.
                        raise ConnectionError(
                            f"frontend state gap: expected {self._frontend_epoch}/{expected}, "
                            f"got {epoch}/{sequence}"
                        )
                    self._frontend_sequence = sequence
                    if self._on_frontend_update is not None:
                        self._on_frontend_update(data)
                elif op == "stopping":
                    # The owner is ending this session ON PURPOSE (a /stop
                    # anywhere: this viewer, another TUI's /stop all, a shell
                    # lop stop). Carried in the disconnect reason rather than a
                    # new callback because every consumer already reads that
                    # string, and the EOF it precedes is moments away — a
                    # viewer that mistakes it for owner death takes over a
                    # session the user just ended (U2-4).
                    reason = STOPPED_REASON
                elif op == "retiring":
                    # A planned refresh (design-runtime-autorefresh §3.2). Same
                    # carrier as ``stopping`` — the disconnect reason — and for
                    # the same reason: the EOF is moments away and every host
                    # already reads that string. The host goes cold at once
                    # and re-engages rather than chasing a record for 8 s.
                    reason = RETIRING_REASON
                elif op in ("ack", "error", "result"):
                    future = self._pending.pop(frame.get("req"), None)
                    if future is not None and not future.done():
                        future.set_result(frame)
        except _OversizedFrame:
            reason = OVERSIZED_FRAME_REASON
            logger.error(
                "attach client: owner sent a frame larger than the %d-byte line limit; "
                "the connection cannot continue",
                _READ_LIMIT_BYTES,
            )
        except (ConnectionResetError, BrokenPipeError):
            reason = "owner connection reset"
        except ConnectionError as exc:
            # A frame the owner sent was refused by this side — malformed, a
            # sequence gap, or a sync a host callback rejected. The socket is
            # healthy; the STATE is not, and the host's disconnect handler is
            # the one place that can decide between redialling for a fresh
            # snapshot and giving up. Named so the log says which. Ordered
            # BEFORE the bare ``OSError`` clause below because
            # ``ConnectionError`` is one, and the two resets above are
            # ``ConnectionError`` subclasses in turn.
            reason = str(exc) or "owner sent a frame this client refused"
            logger.warning("attach client: %s; the connection cannot continue", reason)
        except OSError:
            reason = "owner connection reset"
        except Exception as exc:  # noqa: BLE001 — a callback failure must still report
            # A host callback raised something else (a store rejecting an
            # update, a validation error). Same treatment: the pump cannot
            # continue past a frame it could not apply, and silence here is
            # the "slow owner" bug shape above wearing a different exception.
            reason = f"owner frame could not be applied: {exc}"
            logger.warning("attach client: %s; the connection cannot continue", reason)
        finally:
            self._connected = False
            for future in self._pending.values():
                if not future.done():
                    future.set_exception(ConnectionError(reason))
            self._pending.clear()
            try:
                self._writer.close()  # type: ignore[union-attr]
            except Exception:  # noqa: BLE001
                pass
            self._on_disconnected(reason)

    # -- requests ---------------------------------------------------------------

    async def _request(self, op: str, **fields: Any) -> str:
        """Send one op and await its ack detail (or raise its error message)."""
        reply = await self._request_frame(op, **fields)
        if reply.get("op") == "error":
            from local_operator.session.errors import admission_error

            known = admission_error(str(reply.get("error_code", "")), reply.get("error_count"))
            if known is not None:
                raise known
            raise RuntimeError(str(reply.get("message", "request failed")))
        return str(reply.get("detail", ""))

    async def request_ack_with_duplicate(self, op: str, **fields: Any) -> tuple[str, bool]:
        """Send one op and report ``(detail, duplicate)`` from its ack.

        The idempotency seam for :func:`session.runtime.launch.engage_runtime`.
        A retried errand (the sender crashed after the runtime admitted its
        row, a supervisor re-fired a wake) must not append a second copy, so
        the runtime answers ``duplicate: true`` for a ``command_id`` its
        transcript already owns and the caller reports "already delivered"
        rather than delivering again. An older runtime simply omits the field,
        which reads as False — the pre-idempotency behaviour.
        """
        reply = await self._request_frame(op, **fields)
        if reply.get("op") == "error":
            from local_operator.session.errors import admission_error

            known = admission_error(str(reply.get("error_code", "")), reply.get("error_count"))
            if known is not None:
                raise known
            raise RuntimeError(str(reply.get("message", "request failed")))
        return str(reply.get("detail", "")), bool(reply.get("duplicate", False))

    async def _request_frame(self, op: str, **fields: Any) -> dict[str, Any]:
        """Send one op and return its whole reply frame.

        The shared body of :meth:`_request` and
        :meth:`request_ack_with_duplicate`, which differ only in how much of
        the reply they keep.
        """
        if not self._connected or self._writer is None:
            raise ConnectionError("not attached")
        self._req_seq += 1
        req = self._req_seq
        # FITTED BEFORE THE FUTURE IS REGISTERED, because this can raise
        # `OversizedRequest` and a future parked in `_pending` for a request
        # that was never written is never resolved by anything: it sits there
        # until the connection closes, then takes the teardown's
        # `ConnectionError` with nobody awaiting it — an "exception was never
        # retrieved" log for a refusal the caller had already handled cleanly.
        frame = await fit_request_frame({"op": op, "req": req, **fields})
        future: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self._pending[req] = future
        try:
            self._writer.write(json.dumps(frame).encode() + b"\n")
            await self._writer.drain()
            return await asyncio.wait_for(future, timeout=ACK_TIMEOUT_S)
        except (ConnectionResetError, BrokenPipeError, OSError) as exc:
            raise ConnectionError(f"owner connection lost: {exc}") from exc
        finally:
            self._pending.pop(req, None)

    async def _request_payload(self, op: str, **fields: Any) -> Any:
        """Send one op and await its structured ``result`` payload.

        The sibling of :meth:`_request` for ops whose answer is data rather
        than a receipt line. The registrant replies with ``{"op": "result",
        "data": ...}``; an error frame raises the same way.
        """
        if not self._connected or self._writer is None:
            raise ConnectionError("not attached")
        self._req_seq += 1
        req = self._req_seq
        # Fitted before the future is registered, for the reason spelled out in
        # :meth:`_request_frame`: a refusal must leave nothing parked in
        # ``_pending``.
        frame = await fit_request_frame({"op": op, "req": req, **fields})
        future: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self._pending[req] = future
        try:
            self._writer.write(json.dumps(frame).encode() + b"\n")
            await self._writer.drain()
            reply = await asyncio.wait_for(future, timeout=ACK_TIMEOUT_S)
        except (ConnectionResetError, BrokenPipeError, OSError) as exc:
            # No `_pending.pop` here: the `finally` below runs on this path too,
            # so the second call was always a no-op on an already-popped id
            # (review round 1, MINOR-3).
            raise ConnectionError(f"owner connection lost: {exc}") from exc
        finally:
            self._pending.pop(req, None)
        if reply.get("op") == "error":
            from local_operator.session.errors import admission_error

            known = admission_error(str(reply.get("error_code", "")), reply.get("error_count"))
            if known is not None:
                raise known
            raise RuntimeError(str(reply.get("message", "request failed")))
        return reply.get("data")

    async def prompt(
        self,
        text: str,
        *,
        command_id: str | None = None,
        images: list[dict[str, str]] | None = None,
    ) -> str:
        return await self._request(
            "prompt",
            command_id=command_id or str(uuid.uuid4()),
            text=text,
            images=list(images or []),
        )

    async def send_command(self, command: ContinuationCommand, *, streaming: bool = False) -> str:
        """Submit natural text using the latest owner projection.

        The retained command id rides both paths. If an idle projection races a
        turn start, the owner's busy rejection is reconciled once as steering;
        no second prompt is created and reconnect retries keep one identity.
        """
        if command.session_id != self._session_id:
            raise ValueError("command belongs to another conversation")
        if streaming:
            return await self.steer(
                command.text, command_id=command.command_id, images=command.images
            )
        try:
            return await self.prompt(
                command.text,
                command_id=command.command_id,
                images=command.images,
            )
        except RuntimeError as exc:
            if "already streaming" not in str(exc):
                raise
            return await self.steer(
                command.text, command_id=command.command_id, images=command.images
            )

    async def steer(
        self,
        text: str,
        *,
        command_id: str | None = None,
        images: list[dict[str, str]] | None = None,
    ) -> str:
        return await self._request(
            "steer",
            command_id=command_id or str(uuid.uuid4()),
            text=text,
            images=list(images or []),
        )

    async def desktop_watch(self, *, visible: bool, can_notify: bool) -> str:
        """Renew this attach connection's desktop lease, not the phone counter."""
        return await self._request("desktop_watch", visible=visible, can_notify=can_notify)

    async def abort(self) -> str:
        return await self._request("abort")

    async def acknowledge_attention(self, token: str) -> str:
        if not self._attention_supported:
            raise RuntimeError("update the owner to acknowledge completions")
        return await self._request("acknowledge_attention", completion_token=token)

    async def request_stop(self) -> str:
        """Ask the owner to stop itself — the follower's bare ``/stop``.

        The graceful rung of the kill switch, dialled from the viewer that
        is looking at the session rather than by a third party: the owner's
        runtime runs deny-gates → dispose → unpublish → exit. An owner too
        old to know the op answers the standard unknown-op error, which the
        caller surfaces as the upgrade hint — the follower never escalates
        to a signal against its own owner (that decision belongs to the
        owner's machine, through ``lop stop`` or ``/stop <target>``).
        """
        return await self._request("stop")

    async def retire_if_pristine(self) -> str:
        """Offer the owner back when this viewer never used it.

        The counterpart to the eager engage a viewer performs at mount. The
        answer is the RUNTIME's, not ours: it retires only if nothing durable
        ever happened in the session and no other viewer is still attached
        (see ``RuntimeServer._retire_if_pristine``). We ask; it decides.

        The returned detail says which branch ran ("retired", or "kept: …")
        and is logged rather than shown — a viewer shutting down has no
        surface left to paint on, and a refusal is a normal outcome rather
        than an error. An owner too old to know the op answers the standard
        unknown-op error, which the caller treats the same way: leave it to
        the ordinary residency drain.
        """
        return await self._request("retire_if_pristine")

    async def record_shell(self, command: str, result: dict[str, Any]) -> str:
        return str(await self._request_payload("record_shell", command=command, result=result))

    async def frontend_sync(self) -> Any:
        """Refresh the canonical cut without abandoning admitted RPCs."""
        return await self._request_payload("frontend_sync")

    async def history_page(self, before: str, anchor: str = "") -> Any:
        """Read a signed canonical display page on the authenticated socket."""
        return await self._request_payload("history_page", before=before, anchor=anchor)

    async def request_refresh(self) -> str:
        """Ask a STALE owner to retire now if it is idle, so the next engage
        runs the build on disk.

        The viewer-side belt for the runtime's own reaper-driven refresh: a
        resume in the seconds after ``lop-update`` binds before the runtime
        has noticed the change, and without this the viewer would either wait
        for it or warn. As with ``retire_if_pristine`` the answer is the
        runtime's ("retiring", or "kept: …"); an owner too old to know the op
        answers the unknown-op error, which the caller reads as kept.
        """
        return await self._request("refresh_if_idle")

    async def retire_now(self) -> str:
        """Ask an IDLE owner to retire so a successor can start elsewhere.

        ``/move``'s transport. The session's working directory is fixed when
        its runtime is spawned, so changing it means retiring the current
        runtime and engaging one at the new path — and the runtime leaves by
        the ``retiring`` route rather than the ``stopping`` one, which is the
        whole reason this is a distinct op. A stop tells the viewer the SESSION
        ended (it parks, and ``/resume`` is the way back); ``retiring`` tells it
        a successor is owed and it should engage one, which is exactly what a
        move wants and what the build-refresh path already does.

        As with ``retire_if_pristine`` and ``request_refresh`` the answer is the
        runtime's own ("retiring", or "kept: …"): it re-asks its idle predicate
        and refuses if work arrived. An owner too old to know the op answers the
        standard unknown-op error, which the caller surfaces as a refusal to
        move rather than moving anyway.
        """
        return await self._request("retire_now")

    async def job_trajectory(self, job_id: str, offset: int = 0, limit: int = 120) -> Any:
        """Fetch one page of a child job's retained events from the owner.

        The attach snapshot carries no trajectories — a busy session's retained
        events do not fit the socket's 1 MiB line limit — so the subagent page
        pulls its own rows when a reader opens it.
        """
        return await self._request_payload(
            "job_trajectory", job_id=job_id, offset=offset, limit=limit
        )

    async def watch_job(self, job_id: str) -> str:
        """Subscribe this connection to one job's live trajectory appends.

        Per-connection by design: deltas for jobs nobody is looking at are
        dropped at the owner, which is what keeps a 100-child roster's event
        stream bounded for a viewer reading one page.
        """
        return await self._request("watch_job", job_id=job_id)

    async def unwatch_job(self, job_id: str) -> str:
        """Stop receiving one job's trajectory appends (the page closed)."""
        return await self._request("unwatch_job", job_id=job_id)

    async def slash(
        self,
        command: str,
        args: str,
        images: list[dict[str, str]] | None = None,
    ) -> str:
        return await self._request("slash", command=command, args=args, images=images or [])

    async def slash_result(
        self,
        command: str,
        args: str,
        images: list[dict[str, str]] | None = None,
    ) -> Any:
        """Run one shared slash command on the owner; return its typed outcome.

        The ``result`` frame carries a :class:`SlashResult` payload the invoker
        renders locally, replacing the synthetic ``ran /…`` receipt that left
        the follower's terminal with a transport message while the real answer
        painted in the owner's.
        """
        return await self._request_payload(
            "slash_result", command=command, args=args, images=images or []
        )

    async def fork_snapshot(self, message: str = "") -> dict[str, Any]:
        """Copy the authenticated owner's committed history, without interrupting it."""
        result = await self._request_payload("fork_snapshot", message=message)
        if not isinstance(result, dict) or not result.get("fork_id"):
            raise ValueError("owner returned no fork; retry /fork")
        return result

    async def credential(self, action: str, key: str = "", value: str = "") -> Any:
        """Run one ``/credential`` verb against the owner's variable store.

        ``value`` carries a SECRET for the ``store`` action and is empty for
        every other verb. It has its own named field rather than riding the
        generic ``args`` string of ``slash_result`` so that the one place a
        secret appears on the wire is the one place that must handle it
        carefully — nothing echoes, logs, or transcribes this field.
        """
        return await self._request_payload("credential", action=action, key=key, value=value)

    async def adopt_aside(self, messages: list[dict[str, Any]]) -> str:
        """Fork an aside exchange into the conversation on the authoritative owner."""
        return await self._request("adopt_aside", messages=messages)

    async def cancel_subagents(self) -> int:
        """Cancel every running subagent on the owner; return the REAL count."""
        value = await self._request_payload("cancel_subagents")
        try:
            return int(value)
        except (TypeError, ValueError):
            return -1

    async def complete_aside(self, turns: list[dict[str, Any]]) -> str:
        return await self._request("complete_aside", turns=turns)

    async def set_model(self, provider: str, model_id: str) -> str:
        return await self._request("set_model", provider=provider, model_id=model_id)

    async def set_effort(self, effort: str) -> str:
        return await self._request("set_effort", effort=effort)

    async def approval_answer(self, request_id: str, approved: bool) -> str:
        return await self._request(
            "approval_answer", request_id=request_id, approved=approved, remember=False
        )

    async def ask_answer(
        self, request_id: str, value: str, *, question_index: int | None = None
    ) -> str:
        fields: dict[str, Any] = {"request_id": request_id, "value": value}
        if question_index is not None:
            # The stale-answer guard (U8): name the question that was on
            # screen when the user answered, so an advanced picker refuses it.
            fields["question_index"] = question_index
        return await self._request("ask_answer", **fields)

    async def recall_steer(self, command_id: str) -> str:
        """Unsend the queued steer submitted under ``command_id`` (v4)."""
        return await self._request("recall_steer", command_id=command_id)

    async def detach(self) -> None:
        """Close the connection from our side. ``on_disconnected`` still fires
        (the pump observes EOF) so the host's teardown runs one path."""
        self._connected = False
        if self._writer is not None:
            try:
                self._writer.close()
            except Exception:  # noqa: BLE001
                pass

    def abandon(self) -> None:
        """Close WITHOUT reporting the closure as a disconnect.

        Both ``detach`` and ``close`` still end in ``on_disconnected`` (the
        pump observes EOF or its cancellation) — right for a host that is
        exiting or leaving, whose teardown runs one path, and wrong for a host
        that is ABANDONING a connection it judged unusable and intends to keep
        running: there the callback reads as owner loss and starts a recovery
        that redials the same runtime. The refused-sync path in
        ``RemoteSession`` is that case.
        """
        self._on_disconnected = lambda _reason: None
        self.close()

    def close(self) -> None:
        """Synchronous teardown for hosts without a loop (app exit paths)."""
        self._connected = False
        if self._reader_task is not None:
            self._reader_task.cancel()
            self._reader_task = None
        if self._writer is not None:
            try:
                self._writer.close()
            except Exception:  # noqa: BLE001
                pass


async def continue_command(
    config_dir: Path,
    command: ContinuationCommand,
    *,
    deadline_s: float = ACK_TIMEOUT_S,
    on_projection: Callable[[SessionProjection], None] | None = None,
) -> tuple[AttachClient, str]:
    """Deliver one retained command to whichever host wins the session lease.

    Every contender may start a candidate. The atomic transcript lease, never
    a check before spawning, decides authority. Losing candidates exit and the
    producer redials the published winner with the unchanged command id.

    That arbitration now lives in :func:`session.runtime.launch.engage_runtime`,
    which is the ONE place any caller starts a runtime — this function's own
    spawn-and-poll loop was the prototype for it and has been deleted rather
    than left as a second implementation that could drift. The phone keeps its
    connected :class:`AttachClient` (it streams the reply), so the dial happens
    here after the engage guarantees a runtime exists.
    """
    from local_operator.session.runtime.launch import PromptErrand, engage_runtime

    await engage_runtime(
        command.session_id,
        str(Path.home()),
        PromptErrand(
            text=command.text,
            images=list(command.images),
            command_id=command.command_id,
        ),
        config_dir=config_dir,
        deadline_s=deadline_s,
    )
    # The command is admitted; what remains is the phone's live view of the
    # turn it started. A record must exist now (engage_runtime only returns
    # once one answered), so a miss here is a runtime that died in the gap and
    # is reported as the same timeout the caller already handles.
    record, _ = await asyncio.to_thread(find_owner_record, config_dir, command.session_id)
    if record is None:
        raise TimeoutError("Couldn’t continue this conversation. Try again.")
    client = AttachClient(
        on_projection or (lambda projection: None),
        lambda reason: None,
    )
    try:
        await client.connect(record, command.session_id)
    except (ConnectionError, RuntimeError, TimeoutError) as exc:
        client.close()
        raise TimeoutError("Couldn’t continue this conversation. Try again.") from exc
    return client, "prompt admitted"
