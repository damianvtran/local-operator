"""snapcompact — deterministic bitmap-image archival for vision models.

Instead of asking an LLM to summarize discarded history, the serialized
conversation is rendered onto PNG frames of pixel-font text that vision models
read back directly. The whole pass is local and deterministic: no LLM call, no API key,
no network.

Contract (see ``docs/REWRITE.md`` §C snapcompact bullets):

- ``Archive { frames, text, text_head, text_tail, middle_kinds, page_digests }``
  is stored in the ``CompactionEntry.preserve_data`` so later compactions
  re-render from
  :attr:`Archive.text` (folded in via ``previous_text``) rather than carrying
  old PNGs forward.
- Tool results are truncated to 2000 chars, useless-flagged results AND their
  paired calls are dropped, head/tail plain-text edges surround the imaged
  middle (foveation: :data:`HQ_EDGE_FRAMES` HQ frames at each edge).
- Replay is decided per middle page at pass time (``middle_kinds``): a page
  whose source text stays cheaper than its frame even under the worst
  provider/local tokenizer divergence the repo has measured
  (:data:`CROSS_RULER_SLOPE_MARGIN`) replays as that exact text instead of
  the PNG — same characters, no OCR step. Pages where the frame is the
  cheaper carrier keep their image, and ``page_digests`` pins each page
  string to the frame rendered from it, so a replay can prove the text it
  sends is what the frame drew. Legacy archives (no kinds) replay as frames
  throughout.
- Per-provider frame shapes (Anthropic ``11on16-bw`` @1568/1932, OpenAI
  ``8on22-bw`` @1568, Google ``8on22-bw`` @2048). Budgets: at most
  :data:`MAX_FRAMES` frames, ~:data:`FRAME_TOKEN_ESTIMATE` tokens per frame.
- Requires image input support; :func:`strategy_for_model` falls back to
  ``context-full`` otherwise.

Rendering uses a bundled 5x7 bitmap font (crisp at small sizes, deterministic)
and a stdlib-only grayscale PNG encoder (:mod:`local_operator.compaction.png`),
so no imaging library is needed at install time.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator

from local_operator.harness.types import ImageContent, Message, ModelSpec, TextContent

from .api import TOOL_ARGS_MAX_CHARS, TOOL_RESULT_MAX_CHARS
from .mine import (
    CARRY_FRESH_TOKEN_ALLOWANCE,
    CARRY_TOKEN_BUDGET,
    carry_digest_lines,
    merge_carry,
    mine_carry,
    parse_carry,
    render_carry,
)
from .png import encode_grayscale_png
from .tokens import _CHARS_PER_TOKEN_FALLBACK, estimate_tokens

# ---------------------------------------------------------------------------
# Constants (snapcompact)
# ---------------------------------------------------------------------------

#: Hard upper bound on archive frames carried per compaction. Oldest middle
#: frames are dropped first. This is a ceiling, not the default: see
#: :data:`DEFAULT_MAX_FRAMES` for why the pass renders far fewer.
MAX_FRAMES = 80

#: Conservative per-frame token estimate used for context budgeting — the
#: upper bound across shapes (``FRAME_TOKEN_ESTIMATE``). Prefer
#: :func:`frame_token_estimate_for` when the reader is known: providers bill
#: images very differently (a Gemini frame is 1,120 tokens, an Anthropic
#: 1932px frame ~5,000), and budgeting every frame at the ceiling makes the
#: pass drop history it could afford to keep.
FRAME_TOKEN_ESTIMATE = 5024

#: How much dearer than its local estimate a provider's tokenizer may bill a
#: page's TEXT before the frame is the cheaper carrier — the bound the
#: per-page replay flip must clear (:func:`_middle_page_kinds`). The flip is a
#: cross-ruler comparison by nature (a local number against a provider
#: figure), and ``compaction.tokens`` forbids that without a bound: fitted
#: over a real session, ``claude-opus-5`` bills 1.685x this module's estimate
#: (``claude-opus-4-8`` 1.622x; an OpenAI control 1.036x, GLM 1.019x) and
#: per-request ratios run up to 1.90x
#: (``docs/evidence/compaction-ruler/slope_fit.py``). A page clears the
#: margin only when its text is cheaper under the WORST measured divergence,
#: so no measured family can make a flip a loss; families without a
#: measurement keep that band on the image, the conservative side. One bound
#: for every reader, not a per-model slope table: ``tokens.py`` settled that
#: argument (a multiplier table rots on the next retokenization).
CROSS_RULER_SLOPE_MARGIN = 1.9

#: High-quality frames kept at each chronological edge of a foveated archive
#: (``HQ_EDGE_FRAMES``).
HQ_EDGE_FRAMES = 3

#: Default frame budget for a compaction pass — the number of frames replay
#: will actually send. ``history_blocks`` foveates any archive middle beyond
#: ``2 * HQ_EDGE_FRAMES + 2`` frames down to the HQ edges, so a pass that
#: renders more than this spends ~430 ms of raster+deflate per frame on
#: pages no model ever sees. Measured on a live 600k-token session: 80 frames
#: rendered (~35 s), 6 replayed. Content beyond the budget is dropped from
#: the archive text with an explicit ``[... N chars of oldest history
#: dropped]`` marker — the same fate omp gives it ("About N characters of
#: older middle history dropped to fit archive budget"), and strictly more
#: honest than rendering it into frames that are then silently elided.
DEFAULT_MAX_FRAMES = 2 * HQ_EDGE_FRAMES + 2

#: Maximum snapcompact image base64 carried in every rebuilt provider request
#: (``FRAME_DATA_BYTES_BUDGET``).
FRAME_DATA_BYTES_BUDGET = 3_000_000

#: Per-tool-call cap across the whole serialized argument list
#: (``TOOL_CALL_MAX_CHARS``).
TOOL_CALL_MAX_CHARS = 2000

__all__ = [
    "MAX_FRAMES",
    "DEFAULT_MAX_FRAMES",
    "FRAME_TOKEN_ESTIMATE",
    "HQ_EDGE_FRAMES",
    "FRAME_DATA_BYTES_BUDGET",
    "Shape",
    "Archive",
    "resolve_shape",
    "render_frame",
    "serialize_for_snapcompact",
    "compact_to_archive",
    "history_blocks",
    "strategy_for_model",
    "estimate_archive_tokens",
    "frame_token_estimate_for",
    "replayed_frame_count",
    "CROSS_RULER_SLOPE_MARGIN",
    "carry_token_budget",
]


# ---------------------------------------------------------------------------
# Shapes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Shape:
    """One eval-validated frame shape: glyph geometry plus the page size.

    ``glyph_w``/``glyph_h`` are the bundled font's glyph box; ``advance`` is
    the horizontal cell pitch (tracking) and ``line_pitch`` the vertical pitch
    (leading). ``chars_per_line``/``lines_per_frame`` follow from the square
    ``page_width_px`` frame edge.
    """

    id: str
    glyph_w: int
    glyph_h: int
    advance: int
    line_pitch: int
    page_width_px: int

    @property
    def chars_per_line(self) -> int:
        return self.page_width_px // self.advance

    @property
    def lines_per_frame(self) -> int:
        return self.page_width_px // self.line_pitch

    @property
    def capacity(self) -> int:
        """Characters that fit on one frame."""
        return self.chars_per_line * self.lines_per_frame


def _shape(glyph_w: int, glyph_h: int, advance: int, line_pitch: int, width: int) -> Shape:
    return Shape(
        id=f"{advance}on{line_pitch}-bw@{width}",
        glyph_w=glyph_w,
        glyph_h=glyph_h,
        advance=advance,
        line_pitch=line_pitch,
        page_width_px=width,
    )


#: Anthropic high-res readers (Opus 4.7+, Fable/Mythos) ingest larger frames
#: at the same bill — 1932px sweet spot under the 4,784 visual-token cap.
_ANTHROPIC_LARGE_RE = re.compile(r"claude.*(fable|mythos)|claude-?opus-?4[.-][7-9]", re.IGNORECASE)


def resolve_shape(provider: str, model_id: str) -> Shape:
    """Frame shape for a reader: geometry follows the provider family.

    - ``anthropic``: ``11on16-bw`` — 8x13 glyphs on an 11px advance, 1568px
      frames (1932px for high-res Claude lines).
    - ``openai``: ``8on22-bw`` @1568 (area-proportional patch billing).
    - ``google``: ``8on22-bw`` @2048 (Gemini bills a flat budget per image).
    - anything else: ``8on22-bw`` @1568.
    """
    p = (provider or "").lower()
    if "anthropic" in p or "bedrock" in p or "claude" in p:
        width = 1932 if _ANTHROPIC_LARGE_RE.search(model_id or "") else 1568
        return _shape(8, 13, 11, 16, width)
    if "google" in p or "gemini" in p or "vertex" in p:
        return _shape(8, 13, 8, 22, 2048)
    if "openai" in p or "codex" in p or "azure" in p:
        return _shape(8, 13, 8, 22, 1568)
    return _shape(8, 13, 8, 22, 1568)


def frame_token_estimate_for(provider: str, model_id: str) -> int:
    """What ONE archive frame costs in the named provider's visual tokens.

    Providers do not bill images alike, and pricing every frame at the
    cross-provider ceiling (:data:`FRAME_TOKEN_ESTIMATE` = 5024) is not
    conservative — it is wrong in a user-visible way: the compaction receipt
    and the status band price the replayed archive with this number, so a
    Gemini archive of six 1,120-token frames was reported as 30k tokens of
    context that the provider then billed at 6.7k. Formulas mirror omp's
    ``familyBilling``, which verified them against live bills:

    - Anthropic: ceil(edge/28)² 28px patches, capped at 4,784 patches, +5%
      margin (1568px → 3,293; 1932px → 5,000 — 69² = 4,761 patches, under
      the cap; the cap binds from 1,933px, where ceil(edge/28) reaches 70).
    - Google: flat 1,120 tokens per image (``media_resolution`` HIGH),
      regardless of pixel size.
    - OpenAI: ceil(edge/32)² 32px patches × 1.2 flagship multiplier, capped
      at 10,000 patches (1568px → 2,882).
    - Unknown families: Anthropic's formula, the safe ceiling.
    """
    shape = resolve_shape(provider, model_id)
    edge = shape.page_width_px
    p = (provider or "").lower()
    if "google" in p or "gemini" in p or "vertex" in p:
        return 1120
    if "openai" in p or "codex" in p or "azure" in p:
        patches = min((-(-edge // 32)) ** 2, 10_000)
        return -(-(patches * 12) // 10)  # ceil(patches * 1.2)
    patches = min((-(-edge // 28)) ** 2, 4784)
    return -(-(patches * 105) // 100)  # ceil(patches * 1.05)


#: Fraction of the context window the archive's two VERBATIM text edges may
#: occupy together. The edges are the one part of a replayed archive the token
#: budget cannot trim (it only drops imaged pages), so an unbounded edge is a
#: floor that scales with the frame SHAPE instead of the window: for the
#: default ``HQ_EDGE_FRAMES * capacity`` the pair is ~31.5k tokens for the
#: Anthropic 1932px reader (capacity 21000) and ~35.7k for the Gemini reader
#: (capacity 23808), whether the window is 1M or 64k. Capping the pair at 15%
#: keeps them a minority of the post-compaction context — the imaged middle
#: carries the bulk, at ~4x the density per token — while still preserving
#: enough exact head/tail text for the model to anchor the oldest and newest
#: turns. Chosen to sit under the default 0.5*window archive budget with room
#: for frames beside the edges.
#:
#: The cap binds — i.e. trims below the shape default — whenever the default
#: pair exceeds 15% of the window, which is ``default_pair_chars / 0.6`` in
#: window tokens. For the 1932px reader (pair 126000 chars → 31.5k tokens)
#: that threshold is ~210k, so its NATIVE 200k window is trimmed slightly
#: (63000→60000 chars/edge, a deliberate 15.75%→15% haircut); the Gemini
#: reader (238k threshold) is likewise trimmed at 200k, and every shape is
#: trimmed at 96k/128k. Only windows above the shape's threshold (e.g. any
#: window for the Sonnet/OpenAI 13916-capacity readers up to 139k, or ≥210k
#: for the 1932px reader) keep full-default edges. This is the intended
#: behaviour, not an accident: the whole point is that the edges follow the
#: window, so a reader whose default already exceeds its 15% share gives a
#: little verbatim text back to the imaged middle.
EDGE_WINDOW_FRACTION = 0.15


def _edge_chars_for(shape: Shape, context_window: int | None) -> int:
    """Chars kept VERBATIM at each archive edge, bounded by the window.

    Defaults to ``HQ_EDGE_FRAMES * shape.capacity`` — enough source text to
    fill the high-quality edge frames replay renders — but never lets the two
    edges together exceed :data:`EDGE_WINDOW_FRACTION` of ``context_window``.
    Without the cap the edges are a fixed, un-trimmable floor set by the frame
    shape, not the window (see :data:`EDGE_WINDOW_FRACTION`): on a small window
    the two edges alone can outrun the whole archive budget, so the frame loop
    drops every page and the pass STILL commits a context above its own
    target. Bounding here moves the excess oldest-edge text into the imaged
    region, where the budget can actually act on it.

    Priced in the estimator's fallback char density (``chars ≈ 4 × tokens``,
    :data:`~local_operator.compaction.tokens._CHARS_PER_TOKEN_FALLBACK`) so the
    cap is a nominal token share rather than a raw char count that would mean
    something different per tokenizer. It is only NOMINAL: the budget loop that
    consumes these edges prices them through the real tokenizer
    (:func:`estimate_archive_tokens`), and token-dense content (code, JSON,
    long unbroken runs) tokenizes below 4 chars/token, so on such content the
    edges can occupy more than 15% of the REAL budget. Ordinary prose sits near
    4, and the frame budget still bounds total size, so the approximation is
    deliberate.

    Returns at least one page's worth (``shape.capacity``) so a tiny window
    never collapses the edges to nothing — an archive with no verbatim anchor
    is worse than a small one. Below that floor the archive can still exceed
    ``0.5 * window`` (one page per edge is ~5k tokens, above a 24k window's 12k
    budget): the floor RAISES the window at which the pass fits its budget, it
    does not make it fit universally, and the frame budget remains the ultimate
    bound on total archive size.
    """
    default = HQ_EDGE_FRAMES * shape.capacity
    if not context_window or context_window <= 0:
        return default
    # Half the pair's token allowance, converted to chars, is the per-edge cap.
    pair_token_budget = int(context_window * EDGE_WINDOW_FRACTION)
    per_edge_chars = (pair_token_budget // 2) * _CHARS_PER_TOKEN_FALLBACK
    # Never below one page: some verbatim anchor always beats none.
    return max(shape.capacity, min(default, per_edge_chars))


def carry_token_budget() -> int:
    """Tokens an archive may reserve for its carry digest.

    A one-line indirection over :data:`~local_operator.compaction.mine.CARRY_TOKEN_BUDGET`
    so the reservation is named once; ``compact_to_archive`` charges it only when
    a carry can exist (see the budget loop).
    """
    return CARRY_TOKEN_BUDGET


# ---------------------------------------------------------------------------
# Bundled 5x7 bitmap font (printable ASCII; unknown chars render as a box)
# ---------------------------------------------------------------------------

#: One glyph = 7 rows, each a 5-bit row encoded as two hex digits.
_FONT_DATA: dict[str, str] = {
    " ": "00000000000000",
    "!": "04040404040004",
    '"': "0a0a0000000000",
    "#": "0a0a1f0a1f0a0a",
    "$": "040f140e051e04",
    "%": "18190204081303",
    "&": "0c12140815120d",
    "'": "04040000000000",
    "(": "02040808080402",
    ")": "08040202020408",
    "*": "000a041f040a00",
    "+": "0004041f040400",
    ",": "00000000060408",
    "-": "0000001f000000",
    ".": "00000000000606",
    "/": "01020204080810",
    "0": "0e11131519110e",
    "1": "040c040404040e",
    "2": "0e11010608101f",
    "3": "0e11010601110e",
    "4": "02060a121f0202",
    "5": "1f101e0101110e",
    "6": "0608101e11110e",
    "7": "1f010204080808",
    "8": "0e11110e11110e",
    "9": "0e11110f01020c",
    ":": "00000600000600",
    ";": "00000600060408",
    "<": "02040810080402",
    "=": "00001f001f0000",
    ">": "08040201020408",
    "?": "0e110102040004",
    "@": "0e11171517100e",
    "A": "0e11111f111111",
    "B": "1e11111e11111e",
    "C": "0e11101010110e",
    "D": "1c12111111121c",
    "E": "1f10101e10101f",
    "F": "1f10101e101010",
    "G": "0e11101711110f",
    "H": "1111111f111111",
    "I": "0e04040404040e",
    "J": "0702020202120c",
    "K": "11121418141211",
    "L": "1010101010101f",
    "M": "111b1515111111",
    "N": "11111915131111",
    "O": "0e11111111110e",
    "P": "1e11111e101010",
    "Q": "0e11111115120d",
    "R": "1e11111e141211",
    "S": "0f10100e01011e",
    "T": "1f040404040404",
    "U": "1111111111110e",
    "V": "11111111110a04",
    "W": "1111111515150a",
    "X": "11110a040a1111",
    "Y": "11110a04040404",
    "Z": "1f01020408101f",
    "[": "0e08080808080e",
    "\\": "10080804020201",
    "]": "0e02020202020e",
    "^": "040a1100000000",
    "_": "0000000000001f",
    "`": "08040000000000",
    "a": "00000e010f110f",
    "b": "1010161911111e",
    "c": "00000e1010110e",
    "d": "01010d1311110f",
    "e": "00000e111f100e",
    "f": "0609081c080808",
    "g": "000f11110f010e",
    "h": "10101619111111",
    "i": "04000c0404040e",
    "j": "0200060202120c",
    "k": "10101214181412",
    "l": "0c04040404040e",
    "m": "00001a15151111",
    "n": "00001619111111",
    "o": "00000e1111110e",
    "p": "00001e111e1010",
    "q": "00000d130f0101",
    "r": "00001619101010",
    "s": "00000e100e011e",
    "t": "08081c08080906",
    "u": "0000111111130d",
    "v": "00001111110a04",
    "w": "0000111115150a",
    "x": "0000110a040a11",
    "y": "000011110f010e",
    "z": "00001f0204081f",
    "{": "02040408040402",
    "|": "04040404040404",
    "}": "08040402040408",
    "~": "00000815020000",
}

_BOX_GLYPH = "1f11111111111f"

_FONT: dict[str, tuple[int, ...]] = {
    ch: tuple(int(data[i : i + 2], 16) for i in range(0, 14, 2)) for ch, data in _FONT_DATA.items()
}
_BOX: tuple[int, ...] = tuple(int(_BOX_GLYPH[i : i + 2], 16) for i in range(0, 14, 2))


def _glyph_for(ch: str) -> tuple[int, ...]:
    return _FONT.get(ch) or _BOX


def _wrap_lines(text: str, cols: int) -> list[str]:
    """Hard-wrap ``text`` to ``cols`` characters, preserving line breaks."""
    out: list[str] = []
    for line in text.split("\n"):
        if not line:
            out.append("")
            continue
        while len(line) > cols:
            out.append(line[:cols])
            line = line[cols:]
        out.append(line)
    return out or [""]


def render_frame(text_chunk: str, shape: Shape) -> bytes:
    """Render one page of text as a white-on-black PNG frame.

    The frame is ``shape.page_width_px`` wide and hugs the text rows actually
    printed (``rows * line_pitch`` tall). Glyphs come from the bundled 5x7
    bitmap font scaled into the cell — crisp and deterministic, like the
    native pixel-font renderer.
    """
    lines = _wrap_lines(text_chunk, shape.chars_per_line)[: max(1, shape.lines_per_frame)]
    width = shape.page_width_px
    height = len(lines) * shape.line_pitch
    buf = bytearray(width * height)

    # Integer pixel replication: each font bit scales to sx x sy pixels.
    sx = max(1, shape.glyph_w // 5)
    sy = max(1, shape.glyph_h // 7)

    for row_idx, line in enumerate(lines):
        y0 = row_idx * shape.line_pitch + 1  # 1px top bearing inside the pitch
        for col_idx, ch in enumerate(line):
            glyph = _glyph_for(ch)
            x0 = col_idx * shape.advance + 1  # 1px left bearing
            for gy, bits in enumerate(glyph):
                if not bits:
                    continue
                yy = y0 + gy * sy
                for gx in range(5):
                    if bits & (0x10 >> gx):
                        xx = x0 + gx * sx
                        for dx in range(sx):
                            for dy in range(sy):
                                buf[(yy + dy) * width + xx + dx] = 255

    return encode_grayscale_png(width, height, bytes(buf))


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

_BLANK_RUN_RE = re.compile(r"\n{3,}")


def _is_useless_result(message: Message) -> bool:
    """Useless-flagged tool result (errors win). The flag rides in
    ``provider_payload['useless']`` with ``details`` as a lenient fallback —
    ``Message`` forbids extra fields, so this mirrors ``pruning._is_useless``.
    """
    if message.is_error:
        return False
    payload = message.provider_payload
    if not isinstance(payload, dict):
        return False
    if payload.get("useless"):
        return True
    details = payload.get("details")
    return isinstance(details, dict) and bool(details.get("useless"))


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}\n[... {len(text) - max_chars} more characters truncated]"


def _content_text(message: Message) -> str:
    """Joined text content; image blocks become ``[image]`` placeholders."""
    parts: list[str] = []
    for block in message.content:
        if isinstance(block, TextContent):
            parts.append(block.text)
        elif isinstance(block, ImageContent):
            parts.append("[image]")
    return "".join(parts)


def _render_tool_call(call: Any) -> str:
    """``name(args)`` — each argument value truncated, then the whole list."""
    entries: list[str] = []
    for key, value in (call.arguments or {}).items():
        try:
            rendered = json.dumps(value, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            rendered = str(value)
        entries.append(f"{key}={_truncate(rendered, TOOL_ARGS_MAX_CHARS)}")
    args = _truncate(", ".join(entries), TOOL_CALL_MAX_CHARS)
    return f"{call.name}({args})"


def serialize_for_snapcompact(messages: Sequence[Message]) -> str:
    """Deterministic transcript of ``messages`` for frame rendering.

    Serialization rules: role headers; tool results truncated to
    :data:`TOOL_RESULT_MAX_CHARS` with an explicit ``[... N more characters
    truncated]`` tail; tool arguments truncated to :data:`TOOL_ARGS_MAX_CHARS`;
    useless-flagged tool results AND their paired calls dropped entirely;
    image content replaced by ``[image]``; runs of blank lines collapsed.
    """
    drop_ids: set[str] = set()
    for message in messages:
        if message.role == "tool" and _is_useless_result(message) and message.tool_call_id:
            drop_ids.add(message.tool_call_id)

    blocks: list[str] = []
    for message in messages:
        if message.role == "user":
            text = _content_text(message)
            if text.strip():
                blocks.append(f"[User]\n{text}")
        elif message.role == "assistant":
            parts: list[str] = []
            text = _content_text(message)
            calls = [c for c in message.tool_calls if c.id not in drop_ids]
            if text.strip():
                parts.append(f"[Assistant]\n{text}")
            if calls:
                parts.append(
                    "[Assistant tool calls] " + "; ".join(_render_tool_call(c) for c in calls)
                )
            if parts:
                blocks.append("\n".join(parts))
        else:  # tool result
            if message.tool_call_id and message.tool_call_id in drop_ids:
                continue
            text = _content_text(message)
            if not text.strip():
                continue
            name = message.tool_name or "tool"
            prefix = f"[Tool ERROR: {name}]" if message.is_error else f"[Tool result: {name}]"
            blocks.append(f"{prefix}\n{_truncate(text, TOOL_RESULT_MAX_CHARS)}")

    return _BLANK_RUN_RE.sub("\n\n", "\n\n".join(blocks))


# ---------------------------------------------------------------------------
# Archive
# ---------------------------------------------------------------------------


class Archive(BaseModel):
    """Frame archive persisted under ``CompactionEntry.preserve_data``.

    ``text`` is the bounded kept source (oldest to newest) — the single source
    re-rendered on every later compaction; old PNGs are never carried forward.
    ``text_head``/``text_tail`` are the plain-text edges kept verbatim around
    the middle. ``middle_kinds`` records how that middle replays, one ``t``
    (plain text) or ``i`` (image frame) per page, oldest first; ``""`` means a
    legacy archive whose frames all replay as images.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    frames: list[bytes] = Field(default_factory=list)
    text: str = ""
    text_head: str = ""
    text_tail: str = ""
    shape_id: str = ""
    truncated_chars: int = 0
    #: One ``t`` (plain text) or ``i`` (image frame) per middle page, oldest
    #: first — the replay carrier the pass chose for that page (see
    #: :func:`_middle_page_kinds`). Deliberately ADDITIVE: a missing ``""`` or
    #: a spelling this reader cannot reconcile with the stored pages replays
    #: every frame, which is exactly what archives written before this field
    #: existed must keep doing. Frames are still rendered for text pages, so a
    #: build that does not know the field replays the archive complete (at the
    #: old cost), and any replay that cannot re-derive the text pages degrades
    #: to the same all-image form rather than lose content.
    middle_kinds: str = ""
    #: One :func:`_page_digest` per stored frame, oldest first. The replay
    #: gate requires these to match the re-derived pages before the policy
    #: path is taken, so the text a replay sends is PROVEN to be the strings
    #: the frames were rendered from — a page COUNT alone is not proof (a
    #: reader with different pagination or shape metrics can keep the count
    #: while shifting every boundary). Additive like ``middle_kinds``: absent
    #: or short means the archive replays all-image, which is also what every
    #: pre-policy archive does.
    page_digests: list[str] = Field(default_factory=list)
    #: The never-page-truncated, TIER-RANKED carry digest (see
    #: :mod:`local_operator.compaction.mine`). This is the field that stops the
    #: cliff: each pass mines the pages it is about to DROP into this digest and
    #: folds the previous pass's carry in, so material that has been carried is
    #: no longer lost when its page goes — successive passes stop re-truncating
    #: the same history. It lives OUTSIDE :attr:`text` deliberately: ``text`` is
    #: the next pass's fold source and re-enters the edge/paginate/drop path, so
    #: a carry folded into it would be truncated again; kept separate it is
    #: replayed as leading plain text and merged forward untouched.
    #:
    #: Additive and optional, exactly like ``middle_kinds``: absent means the
    #: archive carries no digest (every pre-carry archive, and any pass whose
    #: mined material was empty), which replays byte-identically to before.
    carry: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # ``frames`` is raw PNG bytes in memory and base64 text on disk, and the
    # two directions MUST live together on the model that owns the field.
    # They did not: the session dumped base64 by hand while revival went
    # through pydantic's lax str->bytes coercion, which UTF-8-ENCODES the
    # base64 text instead of base64-DECODING it. The frame silently became the
    # ASCII bytes OF its own base64, history_blocks encoded that a second time,
    # and every request after a compaction shipped base64(base64(png)) as
    # image/png — provider 400 "does not represent a valid image", which the
    # session's image-rejection backstop then answered by dropping the entire
    # compacted history. The same coercion also made _base64_size and
    # estimate_archive_tokens over-report a revived archive by 4/3, since they
    # measure the inflated frame.
    @field_validator("frames", mode="before")
    @classmethod
    def _decode_frames(cls, value: Any) -> Any:
        """Base64-decode persisted (``str``) frames; pass live ``bytes`` through.

        The type IS the discriminator: JSON revival yields ``str``, in-process
        construction by :func:`compact_to_archive` yields real PNG ``bytes``.
        ``validate=True`` makes garbage a ``binascii.Error`` (a ``ValueError``,
        so pydantic reports a validation error) rather than a frame that is
        wrong but plausible.
        """
        if not isinstance(value, list):
            return value
        return [
            base64.b64decode(item, validate=True) if isinstance(item, str) else item
            for item in value
        ]

    @field_serializer("frames", when_used="json")
    def _encode_frames(self, frames: list[bytes]) -> list[str]:
        """The one canonical persisted form: ``model_dump(mode="json")``.

        Scoped to json mode so ``model_dump()`` still hands callers the real
        bytes; without it pydantic tries to UTF-8-decode PNG bytes.
        """
        return [base64.b64encode(frame).decode("ascii") for frame in frames]

    @field_serializer("created_at", when_used="json")
    def _encode_created_at(self, created_at: datetime) -> str:
        """``isoformat()`` (``+00:00``), not pydantic's ``Z`` shorthand.

        Archives already persisted carry the offset form, and this dump is what
        overwrites them on the next compaction; both parse, but one spelling per
        file keeps transcript diffs and any external reader honest.
        """
        return created_at.isoformat()


def _paginate(text: str, shape: Shape) -> list[str]:
    """Split ``text`` into frame pages, breaking on line boundaries.

    Two budgets: ``capacity`` chars is the byte ceiling, ``lines_per_frame``
    lines is the real page break. render_frame draws at most lines_per_frame
    wrapped lines, so a char-only budget silently slices short-line transcripts
    (role headers, code, JSON) — ~70% of a realistic page — before drawing,
    with no marker that history is missing. Bounding by both keeps every
    serialized line in exactly one page.
    """
    cols = shape.chars_per_line
    capacity = shape.capacity
    pages: list[str] = []
    current: list[str] = []
    current_len = 0
    for line in _wrap_lines(text, cols):
        cost = len(line) + 1  # + newline separator
        if current and (current_len + cost > capacity or len(current) >= shape.lines_per_frame):
            pages.append("\n".join(current))
            current, current_len = [], 0
        current.append(line)
        current_len += cost
    if current:
        pages.append("\n".join(current))
    return pages


def _middle_page_kinds(pages: Sequence[str], provider: str, model_id: str) -> str:
    """One ``t``/``i`` per middle page: text unless a frame bills cheaper.

    A cross-ruler comparison, bounded: the text side is a number from the
    local ruler (``estimate_tokens``) times :data:`CROSS_RULER_SLOPE_MARGIN`
    — the worst provider/local divergence the repo has measured — and the
    frame side is the same per-provider billing ruler the pass's budget loop
    and the session's receipt price frames with. A page replays as plain text
    only when it is cheaper as text under the worst measured divergence, else
    as its PNG. Every character survives either way — the frames are literal
    renders of these pages, and the text pages are the very strings the
    frames were drawn from, so a flip changes the carrier, never the content.
    The margin, not the frame price, decides the band between the two: a page
    that is cheaper as text on the local ruler alone keeps its frame, the
    conservative side for readers whose tokenizer we have not measured.
    """
    per_frame = frame_token_estimate_for(provider, model_id)
    kinds: list[str] = []
    for page in pages:
        if estimate_tokens(Message.assistant(page)) * CROSS_RULER_SLOPE_MARGIN < per_frame:
            kinds.append("t")
        else:
            kinds.append("i")
    return "".join(kinds)


def _page_digest(page: str) -> str:
    """A page's 64-bit identity for the replay gate — see ``page_digests``.

    Integrity, not security: it pins the exact page STRING the pass paginated
    and rendered, so a reader whose pagination or shape metrics differ from
    the writer's can never take the policy path with pages the frames were
    not drawn from (the same page COUNT is not proof of the same page
    strings). Sha256 truncated to 16 hex characters — wide enough that no
    realistic pagination drift collides, short enough to leave the persisted
    payload geometry-neutral.
    """
    return hashlib.sha256(page.encode("utf-8")).hexdigest()[:16]


def compact_to_archive(
    messages: Sequence[Message],
    provider: str,
    model_id: str,
    previous_text: str | None = None,
    *,
    max_frames: int = DEFAULT_MAX_FRAMES,
    context_window: int | None = None,
    previous_carry: str | None = None,
) -> Archive:
    """Run one snapcompact pass over discarded ``messages``.

    The serialized history is appended to the accumulated archive source
    (``previous_text`` — re-rendered from ``Archive.text`` each round) and
    laid out as plain text at both chronological edges with the middle imaged.
    When the imaged middle overflows the frame budget, the OLDEST middle pages
    are dropped (with a marker) — mirrors how iterative summaries fade the
    oldest detail. ``context_window`` caps the archive by TOKENS as well as
    frame count, so the pass that exists to get under the threshold cannot
    itself overflow it on the next turn.

    ``max_frames`` defaults to :data:`DEFAULT_MAX_FRAMES` — exactly the
    number of frames ``history_blocks`` will replay before foveation elides
    the interior. The pass used to render up to :data:`MAX_FRAMES` (80) and
    let replay throw 74 of them away, which is where a manual ``/compact``
    spent most of its minute: ~430 ms of raster + deflate per frame, on the
    event loop, for pages no model ever saw — plus ~10 MB of dead base64 in
    the transcript entry. The dropped content is not lost silently: it leaves
    the archive text with an explicit ``[... N chars of oldest history
    dropped]`` marker, and the un-imaged source is what ``Archive.text``
    carries forward for the next pass.
    """
    shape = resolve_shape(provider, model_id)
    max_frames = max(1, min(max_frames, MAX_FRAMES))

    serialized = serialize_for_snapcompact(messages)
    if previous_text:
        archive_text = f"{previous_text}\n\n{serialized}" if serialized else previous_text
    else:
        archive_text = serialized

    # The two verbatim text edges are the archive's HARD floor: replay sends
    # them uncompressed, the frame budget loop below can only drop IMAGED
    # pages, and nothing else in the pass trims them. At the default
    # HQ_EDGE_FRAMES * capacity they are ~31.5k tokens for the Anthropic 1932px
    # reader (pair 126000 chars) — and MORE than the whole archive budget on a
    # small window, where the loop drops every frame and the pass still
    # overshoots its own 0.5*window budget by 2x (measured: a 64k window gives
    # a 32k budget against a ~31.5k edge-only replay). That is the same "a pass
    # meant to get under the line ends up above it" failure the frame budget
    # exists to prevent, one layer down. So bound the per-edge char count to a
    # window-proportional share BEFORE slicing (see EDGE_WINDOW_FRACTION for
    # which windows this trims): the trimmed-away oldest edge content is not
    # lost, it flows into image_text and is imaged (or dropped with the
    # standard marker) like any other middle history.
    edge_chars = _edge_chars_for(shape, context_window)
    if len(archive_text) <= 2 * edge_chars:
        return Archive(
            frames=[],
            text=archive_text,
            text_head=archive_text,
            text_tail="",
            shape_id=shape.id,
        )

    text_head = archive_text[:edge_chars]
    text_tail = archive_text[len(archive_text) - edge_chars :]
    image_text = archive_text[edge_chars : len(archive_text) - edge_chars]

    pages = _paginate(image_text, shape)
    truncated_chars = 0
    dropped_pages: list[str] = []
    if len(pages) > max_frames:
        dropped_pages.extend(pages[: len(pages) - max_frames])
        truncated_chars = sum(len(p) for p in dropped_pages)
        pages = pages[len(pages) - max_frames :]

    # Token budget: drop oldest middle pages until the replayed archive fits a
    # reserve-adjusted share of the window. Frames are priced with the actual
    # provider's billing (frame_token_estimate_for), not the cross-provider
    # ceiling — the ceiling made a Gemini archive look 4.5x its billed size
    # and dropped history the window could afford. The text edges are
    # tokenized ONCE outside the loop: they are fixed-size, and the previous
    # per-iteration re-estimate tokenized ~126k chars of unchanged text each
    # round through tiktoken.
    per_frame = frame_token_estimate_for(provider, model_id)
    # The carry reservation, charged as a REAL rather than a fixed cost. A
    # carried-forward digest is priced at its actual size (the char budget is a
    # ceiling, and a converged same-session carry sits well under it — measured
    # ~2.4k chars), plus the bounded fresh allowance only when this pass has
    # pages to mine. Charging a flat :data:`CARRY_TOKEN_BUDGET` cost a prose-only
    # pass two pages (measured: cw=60000 dropped 4 → 2 pages for a carry the
    # miner kept nothing from) and could drive a small window's budget negative,
    # breaking the ``max(per_frame, …)`` floor (review round 1, MINOR 5). With no
    # prior carry and nothing to mine the reservation is zero.
    prior_carry_text = previous_carry or ""
    carry_reserve = estimate_tokens(Message.assistant(prior_carry_text)) if prior_carry_text else 0
    if pages:
        carry_reserve += CARRY_FRESH_TOKEN_ALLOWANCE
    if pages and context_window and context_window > 0:
        # Budget floored at one frame, as before the reservation existed: a pass
        # must still be able to try to get under the line on a small window.
        budget = max(per_frame, int(context_window * 0.5) - carry_reserve)
        edge_tokens = estimate_archive_tokens(
            Archive(frames=[], text_head=text_head, text_tail=text_tail)
        )
        while pages and edge_tokens + len(pages) * per_frame > budget:
            truncated_chars += len(pages[0])
            dropped_pages.append(pages[0])
            pages = pages[1:]
        # The reservation is charged by a page DROP being possible at all; once
        # the loop has run, the carry's own cost is covered by whatever the loop
        # gave up. Nothing more is subtracted, so a pass that never drops keeps
        # its full budget (see the MINOR 5 note above).

    # ONE cumulative marker for everything both passes dropped, appended
    # after both have run. One marker per pass double-counted: the budget
    # pass's cumulative figure already included the frame-cap pass's, so a
    # reader summing the two markers counted the first drop twice — and each
    # marker re-renders into every later archive via Archive.text. The
    # marker's few tokens are deliberately not folded back into edge_tokens;
    # the budget is an estimate with a 2x reserve, not an invoice.
    if truncated_chars:
        text_head += f"\n[... {truncated_chars} chars of oldest history dropped]"

    # The carry: mine the pages this pass is about to DROP into a bounded,
    # tier-ranked digest and fold the previous pass's carry in. This is the
    # step that stops the cliff — without it the dropped pages are re-folded
    # from ``Archive.text`` next pass and re-truncated again, so the oldest
    # material is lost a little more on every pass. With it, material that has
    # been mined survives as plain text even after its page is dropped. The
    # carry is stored OUTSIDE ``text`` (see ``Archive.carry``), so it never
    # re-enters the edge/paginate/drop path that ``previous_text`` does.
    #
    # Only the DROPPED pages are mined, not the kept ones: the kept pages stay
    # readable as pages (text or frames), so mining them too would double the
    # archive's readable payload for no retention gain. Pages dropped by EITHER
    # budget (the frame cap or the token budget) are covered.
    dropped_text = "\n".join(dropped_pages)
    fresh = mine_carry(dropped_text) if dropped_text else mine_carry("")
    # ``parse_carry`` reads the archive's own ``render_carry`` form (plain
    # headings); the caller always hands this function the archive's exact
    # ``carry`` field, so no narrowing is needed.
    prior = parse_carry(prior_carry_text)
    merged = merge_carry(prior, fresh)
    carry = render_carry(merged)

    # Replay policy, decided per page while the pages are in hand: text when
    # the page stays cheaper even under the worst measured provider/local
    # divergence (see _middle_page_kinds), else its frame. Frames are STILL
    # rendered for every page, deliberately, even the pages the policy
    # replays as text: a build that knows nothing of middle_kinds replays all
    # frames — complete content, old cost — and a replay that cannot
    # re-derive the pages from text degrades to the same all-image form. The
    # dead-frame carriage costs render time once and bytes at rest; dropping
    # it is a follow-up for after the fleet is known to read kinds.
    middle_kinds = _middle_page_kinds(pages, provider, model_id)
    page_digests = [_page_digest(page) for page in pages]

    frames = [render_frame(page, shape) for page in pages]
    # Pages carry no trailing newline; joining without one glues the last
    # line of page N onto the first of page N+1, and Archive.text is the
    # source re-rendered on every later pass, so the corruption compounds.
    kept_middle = "\n".join(pages)
    return Archive(
        frames=frames,
        text=text_head + kept_middle + text_tail,
        text_head=text_head,
        text_tail=text_tail,
        shape_id=shape.id,
        truncated_chars=truncated_chars,
        middle_kinds=middle_kinds,
        page_digests=page_digests,
        carry=carry,
    )


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------


def _base64_size(frame: bytes) -> int:
    return (len(frame) + 2) // 3 * 4


def _retained_image_positions(
    sizes: Sequence[int], max_frame_data_bytes: int
) -> tuple[list[int], int]:
    """Positions (ascending) of the images a replay sends, and how many it omits.

    Byte budget first: drop the OLDEST over budget, keeping the newest — the
    newest frame itself always survives. Then foveation: once more than
    ``2 * HQ_EDGE_FRAMES + 2`` survive, keep only the first and last
    :data:`HQ_EDGE_FRAMES` and count the interior as omitted. One definition
    for both replays (legacy all-image and the per-page policy), so "which
    frames replay" cannot drift between them.
    """
    kept: list[int] = []
    total = 0
    for index in range(len(sizes) - 1, -1, -1):
        size = sizes[index]
        if kept and total + size > max_frame_data_bytes:
            break
        kept.insert(0, index)
        total += size
    omitted = len(sizes) - len(kept)
    if len(kept) > 2 * HQ_EDGE_FRAMES + 2:
        omitted += len(kept) - 2 * HQ_EDGE_FRAMES
        kept = kept[:HQ_EDGE_FRAMES] + kept[-HQ_EDGE_FRAMES:]
    return kept, omitted


def _middle_pages(archive: Archive) -> list[str] | None:
    """The middle's pages, exactly as the frames were rendered from them, or
    ``None`` when this reader cannot re-derive them.

    Frames are renders of ``_paginate(image_text, shape)``, and ``image_text``
    is the stored ``text`` between its two verbatim edges, so re-paginating
    that slice reproduces the very page strings the frames were drawn from.
    That equality is what lets a replay send a page's TEXT instead of its PNG
    without changing one character — and it is re-verified structurally, not
    assumed: any inconsistency (missing or foreign ``shape_id``, edges that
    are not exact prefix/suffix slices, a page count that disagrees with the
    stored frames) returns ``None``, which degrades the replay to the
    all-image path. Refusing to guess is what keeps a malformed archive on
    the one path that cannot lose content.
    """
    shape = _shape_from_id(archive.shape_id)
    if shape is None:
        return None
    head, tail = archive.text_head, archive.text_tail
    if not archive.text.startswith(head) or not archive.text.endswith(tail):
        return None
    middle = archive.text[len(head) : len(archive.text) - len(tail)]
    pages = _paginate(middle, shape)
    if len(pages) != len(archive.frames):
        return None
    return pages


def _policy_pages(archive: Archive) -> list[str] | None:
    """The re-derived pages IF this archive replays under the policy, else ``None``.

    THE single gate, and the reason four surfaces cannot disagree: the
    replay (:func:`history_blocks`), the reading guide
    (:func:`archive_summary`), the estimator (:func:`estimate_archive_tokens`)
    and the session's receipt (:func:`replayed_frame_count`) all read their
    policy/no-policy decision here, so an archive that is about to DEGRADE —
    an unreconcilable kinds spelling, pages the reader cannot re-derive, a
    page count that disagrees with the stored frames, a shape whose
    pagination no longer reproduces the pages the pass rendered — degrades on
    all four at once. The digest check is what makes that last one exact: it
    binds the re-derived page strings, not just their count, to the frames
    (see ``page_digests``); a same-count boundary shift is refused.
    """
    kinds = archive.middle_kinds
    if not kinds or set(kinds) - {"t", "i"}:
        return None
    if len(archive.page_digests) != len(archive.frames):
        return None
    if len(kinds) != len(archive.frames):
        return None
    pages = _middle_pages(archive)
    if pages is None or len(pages) != len(kinds):
        return None
    if [_page_digest(page) for page in pages] != list(archive.page_digests):
        return None
    return pages


def replayed_frame_count(archive: Archive) -> int:
    """How many image frames a replay of this archive carries.

    One definition for the surfaces that must agree about it: the reading
    guide (:func:`archive_summary`) says this many frames, and the session's
    receipt prices this many. Read through :func:`_policy_pages`, so a policy
    archive that will actually replay under the policy counts its image-kind
    pages, and EVERY other archive — legacy (no kinds), unreconcilable
    spellings, unprovable pages, a digest mismatch — counts every stored
    frame, exactly as the replay will send them. Deliberately blind to the
    byte-budget/foveation drops inside :func:`history_blocks`: the guide and
    the receipt have never counted those (they describe the archive's own
    carriage, not one replay's rations), and teaching every caller the drop
    arithmetic is how they stop agreeing.
    """
    pages = _policy_pages(archive)
    if pages is None:
        return len(archive.frames)
    return archive.middle_kinds.count("i")


def _policy_middle_blocks(
    archive: Archive, pages: list[str], max_frame_data_bytes: int
) -> list[dict[str, Any]]:
    """Blocks for a per-page policy archive: text pages as text, image pages
    as frames, in transcript order.

    Adjacent pages of the same material merge into one block; omitted images
    (byte budget / foveation, counted cumulatively) leave a single
    ``[N frames elided]`` text block at the position of the first omitted
    page, so the marker stands where the missing pages were.
    """
    kinds = archive.middle_kinds
    image_pages = [index for index, kind in enumerate(kinds) if kind == "i"]
    kept_positions, omitted = _retained_image_positions(
        [_base64_size(archive.frames[index]) for index in image_pages], max_frame_data_bytes
    )
    kept_pages = {image_pages[position] for position in kept_positions}

    blocks: list[dict[str, Any]] = []
    suffix = "\n-------------- imaged middle below\n" if kept_pages else ""
    if archive.text_head:
        blocks.append({"kind": "text", "text": archive.text_head + suffix})
    note_emitted = False
    text_run: list[str] = []
    frame_run: list[bytes] = []

    def flush_text() -> None:
        if text_run:
            blocks.append({"kind": "text", "text": "\n".join(text_run)})
            text_run.clear()

    def flush_frames() -> None:
        if frame_run:
            blocks.append(
                {
                    "kind": "images",
                    "frames": [base64.b64encode(frame).decode("ascii") for frame in frame_run],
                }
            )
            frame_run.clear()

    for index, kind in enumerate(kinds):
        if kind == "t":
            flush_frames()
            text_run.append(pages[index])
        elif index in kept_pages:
            flush_text()
            frame_run.append(archive.frames[index])
        elif not note_emitted:
            flush_text()
            flush_frames()
            blocks.append({"kind": "text", "text": f"[{omitted} frames elided]"})
            note_emitted = True
    flush_text()
    flush_frames()
    if archive.text_tail:
        blocks.append({"kind": "text", "text": archive.text_tail})
    return blocks


def history_blocks(
    archive: Archive, max_frame_data_bytes: int = FRAME_DATA_BYTES_BUDGET
) -> list[dict[str, Any]]:
    """Ordered archive blocks for context rebuild, oldest to newest:
    plain text at the oldest edge → the middle → plain text at the newest
    edge (``[{'kind': 'text', ...}, {'kind': 'images', ...}, ...]``).

    The middle replays per page under :attr:`Archive.middle_kinds`: a page
    marked ``t`` rides as PLAIN TEXT — the exact characters its frame renders
    — and the rest ride as image frames. Archives without kinds, and archives
    whose kinds cannot be reconciled with their pages, replay as frames
    throughout, byte for byte the pre-policy behaviour. Foveation and the byte
    budget apply to the images either way: the OLDEST over-budget frames are
    omitted (the newest kept), and more than ``2 * HQ_EDGE_FRAMES + 2``
    survivors foveate to the first and last :data:`HQ_EDGE_FRAMES`, the
    interior elided behind a ``[N frames elided]`` marker.
    """
    kinds = archive.middle_kinds
    pages = _policy_pages(archive)
    if pages is not None and len(pages) == len(kinds):
        return _with_carry_block(
            archive, _policy_middle_blocks(archive, pages, max_frame_data_bytes)
        )

    # Legacy all-image replay — byte-for-byte what this function always did.
    kept_positions, omitted = _retained_image_positions(
        [_base64_size(frame) for frame in archive.frames], max_frame_data_bytes
    )
    kept = [archive.frames[index] for index in kept_positions]

    blocks: list[dict[str, Any]] = []
    if archive.text_head:
        suffix = "\n-------------- imaged middle below\n" if kept else ""
        blocks.append({"kind": "text", "text": archive.text_head + suffix})
    if omitted:
        blocks.append({"kind": "text", "text": f"[{omitted} frames elided]"})
    if kept:
        blocks.append(
            {"kind": "images", "frames": [base64.b64encode(f).decode("ascii") for f in kept]}
        )
    if archive.text_tail:
        blocks.append({"kind": "text", "text": archive.text_tail})
    return _with_carry_block(archive, blocks)


def _with_carry_block(archive: Archive, blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Prepend the carry digest as the OLDEST readable block, when present.

    The carry is the tier-ranked digest of material this archive's passes have
    dropped (see :mod:`local_operator.compaction.mine`). It replays FIRST so the
    model reads the retained key facts — operator asks, decisions, identifiers
    — before the frames and edges, exactly as a chronological oldest-first read
    would. ``Archive.carry`` empty (every pre-carry archive, and a pass that
    mined nothing) returns the blocks untouched, so those replays stay byte for
    byte what they were; every existing caller that does not read the carry
    keeps its old output.
    """
    carry = (archive.carry or "").strip()
    if not carry:
        return blocks
    # Trailing newline ON the block, not implied by the join: rendered marker
    # parts are separate content blocks a provider may concatenate with no
    # separator, so without it the digest's last line would glue onto the
    # oldest edge's first line (measured: "...helper[User]..."). Ending the
    # block keeps the readable text block sequence well-formed however a host
    # joins them.
    return [{"kind": "text", "text": carry + "\n"}, *blocks]


# ---------------------------------------------------------------------------
# Integration helpers
# ---------------------------------------------------------------------------


def strategy_for_model(model_spec: ModelSpec) -> Literal["snapcompact", "context-full"]:
    """snapcompact iff the model can read images back; else context-full."""
    return "snapcompact" if model_spec.supports_images else "context-full"


#: ``Shape.id`` grammar: ``<advance>on<line_pitch>-bw@<width>``. Parsed back
#: when a caption must describe an archive rendered by a DIFFERENT model than
#: the one reading it (mid-session switches): the frames' own shape_id is the
#: durable truth about their geometry; the current model's shape is not.
_SHAPE_ID_RE = re.compile(r"^(\d+)on(\d+)-bw@(\d+)$")


def _shape_from_id(shape_id: str) -> Shape | None:
    """Revive the geometry a persisted archive was rendered with, or None."""
    match = _SHAPE_ID_RE.match(shape_id or "")
    if not match:
        return None
    advance, line_pitch, width = (int(g) for g in match.groups())
    if not (advance and line_pitch and width):
        return None
    return _shape(8, 13, advance, line_pitch, width)


def archive_summary(archive: Archive, provider: str, model_id: str) -> str:
    """Deterministic summary text for a snapcompact pass — NO LLM call.

    This is the text slot of the compaction entry when the archive carries the
    real history. It mirrors omp's ``snapcompact-summary.md``: instructions
    for reading the replayed archive (text edges verbatim; the middle as text
    or pixel-font frames per the archive's policy), not a paraphrase of the
    content. The paraphrase is
    what the frames replace — producing one anyway meant shipping the whole
    discarded history to a provider and waiting on the reply, which made the
    "no LLM call, no network" pass 20–50 s slower than the local work it
    fronted, and is exactly why ``/compact`` here took a minute while omp's
    is near-instant.

    Deliberately structural (frame count, grid geometry, truncation note):
    every fact is derivable from the archive, so the summary can never
    contradict it, and hosts that render the text slot get an honest caption
    rather than an unlabelled apology.
    """
    # The frames were rendered with the archive's OWN shape; after a model
    # switch the current reader's shape can differ, and a caption describing
    # the wrong grid is worse than none. Fall back to the reader's shape only
    # for archives predating shape_id.
    shape = _shape_from_id(archive.shape_id) or resolve_shape(provider, model_id)
    frames = replayed_frame_count(archive)
    if frames == 0:
        middle = ", all as plain text"
    elif frames == len(archive.frames):
        middle = ", with the middle rendered as pixel-font image frames"
    else:
        middle = ", with the middle partly as plain text and partly as pixel-font image frames"
    lines = [
        "Resume prior conversation. Earlier turns are archived below, oldest to",
        f"newest{middle}.",
        "",
        "Reading the archive:",
        "- Plain text: verbatim transcript; rely on it exactly.",
    ]
    if frames:
        plural = "s" if frames != 1 else ""
        detail = (
            "each is one page of the transcript"
            if frames == len(archive.frames)
            else "each is one of the middle pages carried as an image"
        )
        lines.append(
            f"- {frames} image frame{plural}: {detail}, a grid up to"
            f" {shape.chars_per_line} characters wide and {shape.lines_per_frame} rows"
            " tall, read left to right, top to bottom. No word wrap; words may"
            " break across rows."
        )
    if archive.truncated_chars:
        lines.append(
            f"- About {archive.truncated_chars:,} characters of the oldest middle"
            " history were dropped to fit the archive budget (marked inline)."
        )
    lines.append(
        "- If an exact earlier detail matters and a section is unclear, re-derive"
        " it from the workspace (re-read files, re-run commands) rather than guess."
    )
    # A bounded digest of the carry, so the text slot is a REAL digest and not
    # only reading instructions: the model gets the retained asks, decisions and
    # identifiers without OCR-ing the frames. Bounded hard (2,000 chars) and
    # derived entirely from the archive, so it can never contradict the frames
    # and never costs a model call (the pass stays local/deterministic).
    if archive.carry:
        digest = carry_digest_lines(parse_carry(archive.carry))
        if digest:
            lines.append("")
            lines.append("Retained digest from earlier history (still on disk above):")
            lines.extend(digest)
    return "\n".join(lines)


def estimate_archive_tokens(archive: Archive) -> int:
    """Context cost of replaying ``archive``: the plain text the replay carries
    (edges plus any middle pages carried as text) and the conservative
    per-frame visual-token estimate for the frames it sends
    (:func:`replayed_frame_count`)."""
    frames = replayed_frame_count(archive)
    text = archive.text_head + archive.text_tail
    if frames != len(archive.frames):
        pages = _policy_pages(archive)
        if pages is not None:
            text = (
                archive.text_head
                + "\n".join(page for page, kind in zip(pages, archive.middle_kinds) if kind == "t")
                + archive.text_tail
            )
    # The carry rides as leading plain text (see ``history_blocks``), so it is
    # priced with the text ruler beside the edges — the reservation in
    # ``compact_to_archive``'s budget loop is what keeps this honest before the
    # carry exists.
    if archive.carry:
        text = archive.carry + "\n" + text
    text_tokens = estimate_tokens(Message.assistant(text)) if text else 0
    return text_tokens + frames * FRAME_TOKEN_ESTIMATE
