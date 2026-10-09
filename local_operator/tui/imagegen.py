"""Image-generation live progress — the TUI surface's ONE mapping place.

The programme's frozen contract has exactly ONE image tool, ``generate_image``
(image-to-image rides ``source_image_path`` on the same call, so there is no
``generate_altered_image``), and its live states move through one vocabulary:
``queued -> running -> done | failed | cancelled``, plus the interim
``cancelling``. Everything in this module exists so that this surface has ONE
place to change when the harness lane freezes the wire details:

- :data:`IMAGE_GEN_TOOLS` / :func:`is_image_gen_tool` — the detection set. The
  card keys its whole variant off this predicate, so a rename reaches every
  renderer by editing one line.
- :func:`live_from_details` — the ADAPTER. Structured live fields (queue
  position, progress fraction, log tail, provider error payload, artifact
  reference) will ride the existing tool-execution update events'
  ``details`` mapping; their exact key names freeze later. The key names below
  are PROVISIONAL and deliberately live only here: when the freeze lands, this
  module's key constants move and no widget is touched. The adapter never
  invents a value — an absent or malformed field renders as the reduced state.
- :func:`imagegen_state_word` — the state vocabulary, mapped from the card's
  own lifecycle states in one place. The live card paints the live arms
  (``queued`` / ``running`` / ``cancelling``); the settled arms are the
  collapsed row's receipt vocabulary, kept here so the whole state machine has
  exactly one definition.
- :func:`progress_graphic` — the generating-image graphic: a determinate block
  bar when a fraction is known, otherwise a canvas of empty cells that the
  shimmer band crosses. It rides :mod:`local_operator.tui.shimmer`'s existing
  machinery, so the ``display.shimmer`` setting and the
  ``LOCAL_OPERATOR_NO_SHIMMER`` kill switch flatten it to a still, legible
  frame exactly as they do every other animated surface.

The module imports NO widget code on purpose: ``tool_card`` imports this one,
so the direction is one-way (importing the card from here would be a cycle).
Inks come straight from the theme, the same way :mod:`local_operator.tui.shimmer`
resolves its own.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from rich.style import Style
from rich.text import Text

from local_operator.ansi import strip_control_sequences
from local_operator.tui import theme as theme_mod
from local_operator.tui.shimmer import shimmer_text

__all__ = [
    "IMAGE_GEN_TOOLS",
    "ImagegenLive",
    "LIVE_STATE_WORDS",
    "PROGRESS_CELLS",
    "PROGRESS_EMPTY",
    "PROGRESS_FILLED",
    "STATE_CANCELLING",
    "imagegen_state_word",
    "is_image_gen_tool",
    "live_from_details",
    "progress_graphic",
    "provider_error_note",
]

#: The detection set. ONE exported constant: a future alias, a rename, or a
#: second call shape is a one-line edit here and no renderer changes.
IMAGE_GEN_TOOLS: frozenset[str] = frozenset({"generate_image"})


def is_image_gen_tool(tool_name: str) -> bool:
    """Whether ``tool_name`` names the image-generation tool.

    Compared case-insensitively, the way the card's other name tests are
    (``send``), so a producer that shouts the name still reaches the variant.
    """
    return tool_name.strip().lower() in IMAGE_GEN_TOOLS


# ---------------------------------------------------------------------------
# State vocabulary
# ---------------------------------------------------------------------------

STATE_QUEUED = "queued"
STATE_RUNNING = "running"
STATE_CANCELLING = "cancelling"
STATE_DONE = "done"
STATE_FAILED = "failed"
STATE_CANCELLED = "cancelled"

#: The arms a LIVE card may paint. The settled arms are deliberately NOT
#: paintable as live words: a running card that claimed ``done`` would preempt
#: the settle's own receipt, so a provider that reports a terminal state while
#: the call still runs falls back to the card's own word until the result
#: actually lands (and prints the real outcome).
LIVE_STATE_WORDS = (STATE_QUEUED, STATE_RUNNING, STATE_CANCELLING)

#: The card's own lifecycle states -> the frozen vocabulary. ``composing`` and
#: ``waiting`` have no arm on purpose: a call the model is still dictating is
#: BEFORE the vocabulary's first state (nothing has been queued), and a call
#: parked on an approval gate is the card's own ``waiting`` presentation, not
#: a state the provider has seen. Both keep the stock card rendering.
_CARD_STATE_WORDS: dict[str, str] = {
    "queued": STATE_QUEUED,
    "running": STATE_RUNNING,
    "success": STATE_DONE,
    "error": STATE_FAILED,
    "interrupted": STATE_CANCELLED,
}

#: Producer words normalized to the frozen vocabulary. PROVISIONAL with the
#: rest of the wire mapping: an unknown word reads as "no state reported" and
#: the card falls back to its own, which is the honest reduced state.
_PROVIDER_STATE_WORDS: dict[str, str] = {
    "queued": STATE_QUEUED,
    "pending": STATE_QUEUED,
    "waiting": STATE_QUEUED,
    "running": STATE_RUNNING,
    "in_progress": STATE_RUNNING,
    "in-progress": STATE_RUNNING,
    "processing": STATE_RUNNING,
    "generating": STATE_RUNNING,
    "cancelling": STATE_CANCELLING,
    "canceling": STATE_CANCELLING,
    "stopping": STATE_CANCELLING,
    "done": STATE_DONE,
    "completed": STATE_DONE,
    "complete": STATE_DONE,
    "succeeded": STATE_DONE,
    "success": STATE_DONE,
    "failed": STATE_FAILED,
    "failure": STATE_FAILED,
    "error": STATE_FAILED,
    "cancelled": STATE_CANCELLED,
    "canceled": STATE_CANCELLED,
    "aborted": STATE_CANCELLED,
}


def imagegen_state_word(card_state: str, provider_state: str | None = None) -> str | None:
    """The frozen state word this card should paint, or ``None``.

    ``provider_state`` (already normalized by :func:`live_from_details`) wins
    while it is a LIVE arm, because it is the one word that can show an
    interim the card's own lifecycle cannot name — a provider-side queue the
    card is already ``running`` through, or an in-flight ``cancelling``. A
    provider word from the settled end is ignored here (see
    :data:`LIVE_STATE_WORDS`); the settle paints the outcome itself.
    """
    if provider_state in LIVE_STATE_WORDS:
        return provider_state
    return _CARD_STATE_WORDS.get(card_state)


# ---------------------------------------------------------------------------
# The adapter: structured live fields -> one view
# ---------------------------------------------------------------------------
#
# PROVISIONAL WIRE KEYS. The harness lane freezes the exact names shortly;
# until then these are the adapter's whole vocabulary, and this block is the
# only place a freeze (or a rename) lands. Each key is read defensively and a
# value that does not fit its shape reads as absent — the card renders the
# reduced state rather than a number nobody sent.

#: Queue depth from the provider — "the number of requests ahead of yours",
#: present only while the call is queued (FAL's own documented semantics).
#: Zero is a REAL reading (nothing ahead), so the guard tolerates it
#: deliberately rather than dropping a producible state (reviewer F2); a
#: negative, bool or non-int value still reads as absent.
_QUEUE_POSITION_KEY = "queue_position"
#: Progress as a FRACTION of the work, 0..1. Percent-scale shapes are NOT
#: accepted here: a value in ``2..100`` is ambiguous (percent? step index?),
#: and a bar drawn from a misread number is worse than no bar. If the freeze
#: defines a percent field, the conversion lands on this line and nowhere else.
_FRACTION_KEY = "progress_fraction"
#: Per-request log tail: a string (split on newlines) or a sequence of rows.
_LOG_KEY = "log_lines"
#: The caller's state word for this call, normalized against
#: :data:`_PROVIDER_STATE_WORDS`.
_STATE_KEY = "state"
#: The provider's error payload. TWO keys, frozen by the harness lane (manager,
#: 2026-10-08): ``error`` is a stable platform sentence that is safe to paint
#: as-is — the card renders it VERBATIM and never substitutes a sentence of its
#: own — and ``error_type`` is the structured code beside it (FAL's own code
#: when it exists, or one of the platform codes ``media_rejected |
#: media_failed | media_rate_limited | media_unavailable``).
_ERROR_KEY = "error"
#: The structured code beside ``error``: parsed so the surface can branch on
#: STRUCTURE (the cancel conflict below) instead of parsing prose. Kept raw.
_ERROR_TYPE_KEY = "error_type"

#: A cancel that finds the generation already finished comes back as a
#: CONFLICT with this type, not as a failure: the surface's words for it are
#: :data:`ALREADY_FINISHED_NOTE` and it must never wear the error ink or glyph.
ALREADY_FINISHED_TYPE = "media_already_completed"
#: The ONE definition of what that conflict says on a surface.
ALREADY_FINISHED_NOTE = "already finished"

#: The two kinds :func:`provider_error_note` can answer with: a real failure
#: (danger ink, ``✗`` lead) or the already-finished conflict (neutral text).
ERROR_KIND_DANGER = "error"
ERROR_KIND_FINISHED = "finished"
#: A reference to the produced artifact. Parsed so the field has its one
#: normalization place; NOT rendered in v1 — the finished image mounts under
#: the settled card through the transcript's existing image-block machinery,
#: and this module must not start a second transport for it.
_ARTIFACT_KEY = "artifact_ref"

#: How many log-tail rows the view keeps (the live body's own tail cap; a
#: producer that sends more gets the end, with the drop counted).
LOG_TAIL_LIMIT = 20


@dataclass(frozen=True)
class ImagegenLive:
    """One snapshot of a live imagegen call's structured fields.

    Snapshot semantics, matching ``ToolCard.set_partial_detail``'s documented
    contract: a producer re-sends the current state on every update and stops
    sending a field once it stops applying (a queue position stops appearing
    once the request starts), so each call REPLACES the view rather than
    merging into it.
    """

    state: str | None = None
    queue_position: int | None = None
    fraction: float | None = None
    log_lines: tuple[str, ...] = ()
    #: Rows the producer sent beyond :data:`LOG_TAIL_LIMIT`. Exact by
    #: construction: the adapter reads the whole payload's LENGTH and only the
    #: tail's content, so the count never describes bytes it did not see.
    log_dropped: int = 0
    #: The provider's error sentence, VERBATIM (whitespace-trimmed, control
    #: sequences stripped — the card's paint boundary).
    error: str | None = None
    #: The structured code beside :attr:`error`, raw; the surface branches on
    #: it (see :data:`ALREADY_FINISHED_TYPE`) rather than parsing the sentence.
    error_type: str | None = None
    artifact_ref: str | None = None
    #: Every recognized key this snapshot carried. A tuple (not a set) to keep
    #: the frozen dataclass hashable; read by tests as "what did this update
    #: actually carry".
    keys: tuple[str, ...] = ()


def live_from_details(details: object) -> ImagegenLive:
    """Map an update or result ``details`` mapping to the surface's view.

    Never raises and never invents: every field is read through a shape check,
    and a value that fails it is the same as an absent one. The mapping this
    builds is what the live body and the settled failure body render from, so
    the whole field vocabulary has exactly one reader.
    """
    if not isinstance(details, Mapping):
        return ImagegenLive()
    seen: list[str] = []

    state: str | None = None
    raw_state = details.get(_STATE_KEY)
    if isinstance(raw_state, str) and raw_state.strip():
        state = _PROVIDER_STATE_WORDS.get(raw_state.strip().lower())
        seen.append(_STATE_KEY)

    queue_position: int | None = None
    raw_queue = details.get(_QUEUE_POSITION_KEY)
    # ``>= 0`` is deliberate: the field counts requests AHEAD of ours, and 0
    # is "nothing ahead", not a malformed value (reviewer F2).
    if isinstance(raw_queue, int) and not isinstance(raw_queue, bool) and raw_queue >= 0:
        queue_position = raw_queue
        seen.append(_QUEUE_POSITION_KEY)

    fraction: float | None = None
    raw_fraction = details.get(_FRACTION_KEY)
    if isinstance(raw_fraction, (int, float)) and not isinstance(raw_fraction, bool):
        value = float(raw_fraction)
        if 0.0 <= value <= 1.0:
            fraction = value
            seen.append(_FRACTION_KEY)

    log_lines: tuple[str, ...] = ()
    log_dropped = 0
    raw_logs = details.get(_LOG_KEY)
    rows: list[str] | None = None
    if isinstance(raw_logs, str):
        rows = raw_logs.splitlines()
    elif (
        isinstance(raw_logs, Sequence)
        and not isinstance(raw_logs, (str, bytes))
        and all(isinstance(row, str) for row in raw_logs)
    ):
        rows = list(raw_logs)
    if rows is not None:
        # Bound BEFORE cleaning: a payload that printed a hundred thousand
        # rows must cost the same as one that printed twenty (the card's own
        # `set_partial_detail` ingest bound, applied at this reader instead of
        # deferred to the painter).
        log_dropped = max(0, len(rows) - LOG_TAIL_LIMIT)
        log_lines = tuple(strip_control_sequences(row) for row in rows[-LOG_TAIL_LIMIT:])
        seen.append(_LOG_KEY)

    error: str | None = None
    raw_error = details.get(_ERROR_KEY)
    if isinstance(raw_error, str):
        # Clean first, assign only when something survives: a payload that was
        # nothing but whitespace or escape sequences reads as ABSENT (``None``),
        # not as an empty string — every caller's truthiness test already
        # treats them the same, and ``None`` is the one shape the dataclass
        # documents.
        cleaned = _clean_error_text(raw_error)
        if cleaned:
            error = cleaned
    if error:
        seen.append(_ERROR_KEY)

    error_type: str | None = None
    raw_error_type = details.get(_ERROR_TYPE_KEY)
    if isinstance(raw_error_type, str) and raw_error_type.strip():
        error_type = raw_error_type.strip()
        seen.append(_ERROR_TYPE_KEY)

    artifact_ref: str | None = None
    raw_artifact = details.get(_ARTIFACT_KEY)
    if isinstance(raw_artifact, str) and raw_artifact.strip():
        artifact_ref = raw_artifact.strip()
        seen.append(_ARTIFACT_KEY)

    return ImagegenLive(
        state=state,
        queue_position=queue_position,
        fraction=fraction,
        log_lines=log_lines,
        log_dropped=log_dropped,
        error=error,
        error_type=error_type,
        artifact_ref=artifact_ref,
        keys=tuple(seen),
    )


def _clean_error_text(text: str) -> str:
    """The provider's own words, minus control sequences and edge whitespace.

    Control sequences go because this text is painted on every frame and an
    erase-display inside it clears the terminal (the same boundary rule the
    card's names and outputs follow). Nothing else is altered: the sentence
    the operator reads on failure is the provider's, verbatim.
    """
    return strip_control_sequences(text).strip()


def provider_error_note(
    error: str | None, error_type: str | None, output_text: str = ""
) -> tuple[str, str] | None:
    """``(text, kind)`` for the provider's error slot, or ``None``.

    TWO shapes ride this reader because the harness lane's frozen contract
    says so (manager, 2026-10-08): a real failure carries the platform's own
    sentence, painted VERBATIM — never a generic sentence of this module's
    own, and never expected to be vendor prose — and a cancel that found the
    generation already finished carries ``error_type:
    media_already_completed``, which is NOT a failure: its words are
    :data:`ALREADY_FINISHED_NOTE` and its kind says so, so no surface can
    accidentally paint it with an error's ink or glyph.

    ``None`` covers both "nothing was carried" and "the standard body already
    carries it": the failure body promotes the result's head line and prints
    the captured output, so a payload already on screen must not print twice.
    The comparison is whitespace-normalized on both sides — the card stores
    collapsed sentences while output keeps its own spacing, so a like-for-like
    test is the one that works for both shapes.
    """
    finished = (error_type or "").strip().lower() == ALREADY_FINISHED_TYPE
    if finished:
        text = ALREADY_FINISHED_NOTE
    elif error:
        text = error
    else:
        return None
    if output_text and " ".join(text.split()) in " ".join(output_text.split()):
        return None
    return (text, ERROR_KIND_FINISHED if finished else ERROR_KIND_DANGER)


# ---------------------------------------------------------------------------
# The graphic
# ---------------------------------------------------------------------------

#: Cells in the graphic — BOTH branches, so freezing a fraction in never
#: reflows the row: the bar lands in exactly the footprint the canvas was
#: animating in. Ten because the splice band (twelve cells wide, including its
#: shoulders) only TRAVELS across a canvas wider than itself; over the brief's
#: five cells the whole row lights and darkens as one pulse instead of a
#: gradient crossing it (measured on the band maths, not assumed).
PROGRESS_CELLS = 10

#: The bar's fill/track pair — the HOUSE pair (`█`/`░`; `usage_panel.py`'s
#: ``BAR_FILLED``/``BAR_EMPTY`` and every other proportion graphic in the
#: product). The parallelogram pair ``▰``/``▱`` this started on was the only
#: occurrence in the tree and sits outside WGL4's Block-Elements subset, so it
#: taught a second fill language for the same read; one vocabulary, in the
#: repertoire the product already trusts for fills (design review round 1,
#: D1). A future pair change is a one-line edit here.
PROGRESS_FILLED = "█"
PROGRESS_EMPTY = "░"


def progress_graphic(fraction: float | None, time_ms: float | None = None) -> Text:
    """The generating-image graphic: determinate bar, else shimmering canvas.

    With ``fraction`` known the graphic is frozen and precise — ``████░░░░░░
    42%`` — because a real number exists and motion would only smear it.
    Without one it is a canvas of empty cells that the shimmer band crosses,
    which is the honest indeterminate read: work is happening, nobody has said
    how much. The band comes from :func:`~local_operator.tui.shimmer.shimmer_text`,
    so the settings flag and the env kill switch flatten it to a still dim
    frame for CI and snapshots, and a blurred terminal keeps the last phase
    frozen rather than rendering a different shape.

    ``time_ms`` is the phase, injected exactly as the shimmer takes it, so
    tests and capture scripts pin a frame deterministically.
    """
    if fraction is not None:
        # Fill and label derive from ONE number — the label's own percent,
        # floor-divided into cells. Rounding them separately let 0.99 paint
        # ten filled cells under a "99%" label (reviewer F3): the bar read
        # ahead of its own number. The floor keeps the fill at or below the
        # label, so the bar can never claim more progress than the percent
        # beside it; only a true 100% fills the last cell.
        percent = round(fraction * 100)
        filled = max(0, min(PROGRESS_CELLS, percent * PROGRESS_CELLS // 100))
        bar = Text()
        bar.append(
            PROGRESS_FILLED * filled,
            style=Style(color=theme_mod.semantic_color("accent"), bold=True),
        )
        bar.append(PROGRESS_EMPTY * (PROGRESS_CELLS - filled), style=_dim_style())
        bar.append(
            f" {percent}%",
            style=Style(color=theme_mod.semantic_color("muted")),
        )
        return bar
    canvas = PROGRESS_EMPTY * PROGRESS_CELLS
    return shimmer_text(canvas, time_ms)


def _dim_style() -> Style:
    return Style(color=theme_mod.semantic_color("dim"))
