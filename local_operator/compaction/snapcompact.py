"""snapcompact — deterministic bitmap-image archival for vision models.

Instead of asking an LLM to summarize discarded history, the serialized
conversation is rendered onto PNG frames of pixel-font text that vision models
read back directly. The whole pass is local and deterministic: no LLM call, no API key,
no network.

Contract (see ``docs/REWRITE.md`` §C snapcompact bullets):

- ``Archive { frames, text, text_head, text_tail }`` is stored in the
  ``CompactionEntry.preserve_data`` so later compactions re-render from
  :attr:`Archive.text` (folded in via ``previous_text``) rather than carrying
  old PNGs forward.
- Tool results are truncated to 2000 chars, useless-flagged results AND their
  paired calls are dropped, head/tail plain-text edges surround the imaged
  middle (foveation: :data:`HQ_EDGE_FRAMES` HQ frames at each edge).
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
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator

from local_operator.harness.types import ImageContent, Message, ModelSpec, TextContent

from .api import TOOL_ARGS_MAX_CHARS, TOOL_RESULT_MAX_CHARS
from .png import encode_grayscale_png
from .tokens import estimate_tokens

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

    - Anthropic: ceil(edge/28)² 28px patches, capped at 4,784 visual tokens,
      +5% margin (1568px → 3,293; 1932px → 5,024).
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
    the imaged middle.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    frames: list[bytes] = Field(default_factory=list)
    text: str = ""
    text_head: str = ""
    text_tail: str = ""
    shape_id: str = ""
    truncated_chars: int = 0
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


def compact_to_archive(
    messages: Sequence[Message],
    provider: str,
    model_id: str,
    previous_text: str | None = None,
    *,
    max_frames: int = DEFAULT_MAX_FRAMES,
    context_window: int | None = None,
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

    edge_chars = HQ_EDGE_FRAMES * shape.capacity
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
    if len(pages) > max_frames:
        dropped = pages[: len(pages) - max_frames]
        truncated_chars = sum(len(p) for p in dropped)
        pages = pages[len(pages) - max_frames :]
        text_head += f"\n[... {truncated_chars} chars of oldest history dropped]"

    # Token budget: drop oldest middle pages until the replayed archive fits a
    # reserve-adjusted share of the window. Frames are priced with the actual
    # provider's billing (frame_token_estimate_for), not the cross-provider
    # ceiling — the ceiling made a Gemini archive look 4.5x its billed size
    # and dropped history the window could afford. The text edges are
    # tokenized ONCE outside the loop: they are fixed-size (the truncation
    # marker aside), and the previous per-iteration re-estimate tokenized
    # ~126k chars of unchanged text each round through tiktoken.
    per_frame = frame_token_estimate_for(provider, model_id)
    if pages and context_window and context_window > 0:
        budget = max(per_frame, int(context_window * 0.5))
        edge_tokens = estimate_archive_tokens(
            Archive(frames=[], text_head=text_head, text_tail=text_tail)
        )
        while pages and edge_tokens + len(pages) * per_frame > budget:
            truncated_chars += len(pages[0])
            pages = pages[1:]
            text_head += f"\n[... {truncated_chars} chars of oldest history dropped]"
        # The marker lines appended above are a handful of tokens; they are
        # deliberately not folded back into edge_tokens — the budget is an
        # estimate with a 2x reserve, not an invoice.

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
    )


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------


def _base64_size(frame: bytes) -> int:
    return (len(frame) + 2) // 3 * 4


def history_blocks(
    archive: Archive, max_frame_data_bytes: int = FRAME_DATA_BYTES_BUDGET
) -> list[dict[str, Any]]:
    """Ordered archive blocks for context rebuild, oldest to newest:
    plain text at the oldest edge → imaged middle → plain text at the newest
    edge (``[{'kind': 'text', ...}, {'kind': 'images', ...}, ...]``).

    Foveation keeps the NEWEST frames (oldest are
    omitted), and an archive middle beyond the HQ edges keeps only the first
    and last :data:`HQ_EDGE_FRAMES` frames, eliding the interior with a
    ``[N frames elided]`` marker.
    """
    # Byte budget: drop the OLDEST frames over budget (keeps the newest).
    kept: list[bytes] = []
    total = 0
    for frame in reversed(archive.frames):
        size = _base64_size(frame)
        if kept and total + size > max_frame_data_bytes:
            break
        kept.insert(0, frame)
        total += size
    omitted = len(archive.frames) - len(kept)

    # Foveation: HQ edges only once the middle outruns 2*HQ_EDGE + 2 frames.
    if len(kept) > 2 * HQ_EDGE_FRAMES + 2:
        omitted += len(kept) - 2 * HQ_EDGE_FRAMES
        kept = kept[:HQ_EDGE_FRAMES] + kept[-HQ_EDGE_FRAMES:]

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
    return blocks


# ---------------------------------------------------------------------------
# Integration helpers
# ---------------------------------------------------------------------------


def strategy_for_model(model_spec: ModelSpec) -> Literal["snapcompact", "context-full"]:
    """snapcompact iff the model can read images back; else context-full."""
    return "snapcompact" if model_spec.supports_images else "context-full"


def archive_summary(archive: Archive, provider: str, model_id: str) -> str:
    """Deterministic summary text for a snapcompact pass — NO LLM call.

    This is the text slot of the compaction entry when the archive carries the
    real history. It mirrors omp's ``snapcompact-summary.md``: instructions
    for reading the replayed archive (text edges verbatim, the middle as
    pixel-font frames), not a paraphrase of the content. The paraphrase is
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
    shape = resolve_shape(provider, model_id)
    middle = (
        ", with the middle rendered as pixel-font image frames"
        if archive.frames
        else ", all as plain text"
    )
    lines = [
        "Resume prior conversation. Earlier turns are archived below, oldest to",
        f"newest{middle}.",
        "",
        "Reading the archive:",
        "- Plain text: verbatim transcript; rely on it exactly.",
    ]
    if archive.frames:
        plural = "s" if len(archive.frames) != 1 else ""
        lines.append(
            f"- {len(archive.frames)} image frame{plural}: each is one page of the"
            f" transcript, a grid up to {shape.chars_per_line} characters wide and"
            f" {shape.lines_per_frame} rows tall, read left to right, top to bottom."
            " No word wrap; words may break across rows."
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
    return "\n".join(lines)


def estimate_archive_tokens(archive: Archive) -> int:
    """Context cost of replaying ``archive``: head/tail text tokens plus the
    conservative per-frame visual-token estimate."""
    text = archive.text_head + archive.text_tail
    text_tokens = estimate_tokens(Message.assistant(text)) if text else 0
    return text_tokens + len(archive.frames) * FRAME_TOKEN_ESTIMATE
