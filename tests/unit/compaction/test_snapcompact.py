"""snapcompact: shapes, serialization, rendering, archiving, replay."""

import base64
import io
import json
from datetime import datetime, timezone
from typing import cast

import pytest
from PIL import Image
from pydantic import ValidationError

from local_operator.compaction.api import TOOL_ARGS_MAX_CHARS, TOOL_RESULT_MAX_CHARS
from local_operator.compaction.snapcompact import (
    CROSS_RULER_SLOPE_MARGIN,
    DEFAULT_MAX_FRAMES,
    EDGE_WINDOW_FRACTION,
    FRAME_DATA_BYTES_BUDGET,
    FRAME_TOKEN_ESTIMATE,
    HQ_EDGE_FRAMES,
    MAX_FRAMES,
    Archive,
    _middle_page_kinds,
    _page_digest,
    archive_summary,
    compact_to_archive,
    estimate_archive_tokens,
    frame_token_estimate_for,
    history_blocks,
    render_frame,
    replayed_frame_count,
    resolve_shape,
    serialize_for_snapcompact,
    strategy_for_model,
)
from local_operator.compaction.tokens import estimate_tokens
from local_operator.harness.types import (
    ImageContent,
    Message,
    ModelSpec,
    TextContent,
    ToolCall,
)

# The session's dump helper, exercised HERE because it is the other half of
# Archive's base64 contract: neither direction is testable alone.
from local_operator.session.session import _archive_to_json

# ---------------------------------------------------------------------------
# Shapes
# ---------------------------------------------------------------------------


def test_shape_table_per_provider():
    """Anthropic 11-on-16 @1568, OpenAI/unknown 8-on-22 @1568, Google @2048."""
    anthropic = resolve_shape("anthropic", "claude-sonnet-4-5")
    assert (anthropic.glyph_w, anthropic.glyph_h) == (8, 13)
    assert anthropic.advance == 11
    assert anthropic.line_pitch == 16
    assert anthropic.page_width_px == 1568
    assert anthropic.chars_per_line == 1568 // 11
    assert anthropic.lines_per_frame == 1568 // 16

    openai = resolve_shape("openai", "gpt-5.5")
    assert (openai.glyph_w, openai.glyph_h) == (8, 13)
    assert openai.advance == 8
    assert openai.line_pitch == 22
    assert openai.page_width_px == 1568

    google = resolve_shape("google", "gemini-3.5-flash")
    assert google.page_width_px == 2048
    assert google.advance == 8 and google.line_pitch == 22

    unknown = resolve_shape("mistral", "mistral-large")
    assert unknown.page_width_px == 1568
    assert unknown.advance == 8 and unknown.line_pitch == 22


def test_shape_anthropic_large_model_gets_1932():
    """High-res Claude lines (Opus 4.7+, Fable/Mythos) read 1932px frames."""
    large = resolve_shape("anthropic", "claude-opus-4-8")
    assert large.page_width_px == 1932
    assert large.advance == 11


def test_shape_capacity_consistent():
    shape = resolve_shape("anthropic", "claude-sonnet-4-5")
    assert shape.capacity == shape.chars_per_line * shape.lines_per_frame


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def _tool_message(
    call_id: str, text: str, *, useless: bool = False, is_error: bool = False
) -> Message:
    """Tool-result message; useless rides in provider_payload per the REWRITE contract."""
    return Message(
        role="tool",
        content=[TextContent(text=text)],
        tool_call_id=call_id,
        tool_name="bash",
        is_error=is_error,
        provider_payload={"useless": useless} if useless else None,
    )


def test_serialize_role_headers_and_tool_result_truncation():
    big = "x" * (TOOL_RESULT_MAX_CHARS + 500)
    messages = [
        Message.user("hello"),
        Message(
            role="assistant",
            content=[TextContent(text="working")],
            tool_calls=[ToolCall(id="c1", name="bash", arguments={"cmd": "ls"})],
        ),
        _tool_message("c1", big),
    ]
    out = serialize_for_snapcompact(messages)
    assert "[User]" in out and "hello" in out
    assert "[Assistant]" in out and "working" in out
    assert "[Tool result: bash]" in out
    # Truncation marker names the exact elided char count.
    assert f"[... {len(big) - TOOL_RESULT_MAX_CHARS} more characters truncated]" in out
    kept = out.split("[Tool result: bash]\n")[1].split("\n[... ")[0]
    assert len(kept) == TOOL_RESULT_MAX_CHARS


def test_serialize_tool_args_truncated():
    huge = "y" * (TOOL_ARGS_MAX_CHARS + 1000)
    messages = [
        Message(
            role="assistant",
            content=[],
            tool_calls=[ToolCall(id="c2", name="write", arguments={"content": huge})],
        ),
        _tool_message("c2", "done"),
    ]
    out = serialize_for_snapcompact(messages)
    assert "more characters truncated" in out
    assert huge not in out
    assert "done" in out


def test_serialize_drops_useless_result_and_paired_call():
    messages = [
        Message(
            role="assistant",
            content=[],
            tool_calls=[
                ToolCall(id="keep", name="read", arguments={"path": "a.py"}),
                ToolCall(id="gone", name="search", arguments={"q": "zzz"}),
            ],
        ),
        _tool_message("keep", "file contents here"),
        _tool_message("gone", "zero matches", useless=True),
    ]
    out = serialize_for_snapcompact(messages)
    assert "read(" in out and "file contents here" in out
    assert "search(" not in out
    assert "zero matches" not in out


def test_serialize_error_with_useless_flag_survives():
    """Errors win: a useless-flagged error is never dropped."""
    messages = [
        Message(
            role="assistant", content=[], tool_calls=[ToolCall(id="e1", name="bash", arguments={})]
        ),
        _tool_message("e1", "boom", useless=True, is_error=True),
    ]
    out = serialize_for_snapcompact(messages)
    assert "[Tool ERROR: bash]" in out
    assert "boom" in out


def test_serialize_replaces_images_and_collapses_blank_runs():
    messages = [
        Message(role="user", content=[TextContent(text="see:"), ImageContent(data="AAAA")]),
        Message.user("after\n\n\n\n\nblank"),
    ]
    out = serialize_for_snapcompact(messages)
    assert "[image]" in out
    assert "\n\n\n" not in out


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def test_render_frame_valid_png_with_shape_dimensions():
    shape = resolve_shape("anthropic", "claude-sonnet-4-5")
    png = render_frame("hello world\nsecond line", shape)
    with Image.open(io.BytesIO(png)) as img:
        assert img.format == "PNG"
        assert img.size == (shape.page_width_px, 2 * shape.line_pitch)
        pixels = img.load()
        assert pixels is not None
        # White-on-black: ink on the first glyph row, black at the bottom edge.
        ink = sum(1 for x in range(0, shape.page_width_px, 3) if cast(int, pixels[x, 3]) > 128)
        assert ink > 0
        assert pixels[0, img.size[1] - 1] == 0


def test_render_frame_height_hugs_text():
    shape = resolve_shape("google", "gemini-3.5-flash")
    png = render_frame("one\ntwo\nthree", shape)
    with Image.open(io.BytesIO(png)) as img:
        assert img.size == (shape.page_width_px, 3 * shape.line_pitch)


def test_render_frame_deterministic_and_wraps():
    shape = resolve_shape("openai", "gpt-5.5")
    long_line = "a" * (shape.chars_per_line + 10)
    assert render_frame(long_line, shape) == render_frame(long_line, shape)
    with Image.open(io.BytesIO(render_frame(long_line, shape))) as img:
        assert img.size[1] == 2 * shape.line_pitch


# ---------------------------------------------------------------------------
# Compaction
# ---------------------------------------------------------------------------


def _conversation_messages(rounds: int, payload: int = 400) -> list[Message]:
    out: list[Message] = []
    for i in range(rounds):
        out.append(Message.user(f"question {i}"))
        out.append(
            Message(
                role="assistant",
                content=[TextContent(text=f"answer {i}")],
                tool_calls=[ToolCall(id=f"call-{i}", name="bash", arguments={"cmd": f"step{i}"})],
            )
        )
        out.append(_tool_message(f"call-{i}", f"output-{i} " + "z" * payload))
    return out


#: One full page of high-entropy filler (a full-width 71-line page at the
#: openai shape): ~7.3k tokens, well above the 2,882-token frame bill, so
#: content built from it keeps its images under the replay policy.
_DENSE_RUN = ("ZXCVBNMASDFGHJKLqwertyuiop1234567890mMnBvC" * 800)[:13_916]


def _dense_messages(rounds: int) -> list[Message]:
    """User turns whose pages each bill above one frame (images kept)."""
    return [Message.user(f"dense {i} " + _DENSE_RUN) for i in range(rounds)]


def _constructed_pages(kinds: str) -> list[str]:
    """One page per kind char: exactly ``lines_per_frame`` short lines each."""
    shape = resolve_shape("openai", "gpt-5.5")
    return [
        "\n".join(f"page{index} line {row}" for row in range(shape.lines_per_frame))
        for index in range(len(kinds))
    ]


def _constructed_archive(kinds: str) -> Archive:
    """An archive whose middle is one page per kind char, re-derivable exactly.

    Built by hand rather than through ``compact_to_archive`` so the replay
    tests can pin arbitrary kind sequences: each page is exactly
    ``lines_per_frame`` short lines of the openai shape — the page-break
    condition ``_paginate`` uses — so ``_middle_pages`` reproduces the pages
    by construction, and the digests are written the way a pass writes them.
    Frames are real renders, so a replay's carrier is checkable per page.
    """
    shape = resolve_shape("openai", "gpt-5.5")
    pages = _constructed_pages(kinds)
    head = "HEAD-" + "h" * 40
    tail = "TAIL-" + "t" * 40
    middle = "\n".join(pages)
    return Archive(
        frames=[render_frame(page, shape) for page in pages],
        text=head + middle + tail,
        text_head=head,
        text_tail=tail,
        shape_id=shape.id,
        middle_kinds=kinds,
        page_digests=[_page_digest(page) for page in pages],
    )


def test_compact_small_history_keeps_all_text():
    """A transcript fitting in the two text edges renders no frames."""
    messages = _conversation_messages(2, payload=50)
    archive = compact_to_archive(messages, "anthropic", "claude-sonnet-4-5")
    assert archive.frames == []
    assert archive.text_head
    assert archive.text == archive.text_head  # everything kept verbatim
    assert archive.shape_id.startswith("11on16-bw@")


def test_compact_large_history_frames_and_edges():
    messages = _conversation_messages(200)
    archive = compact_to_archive(messages, "anthropic", "claude-sonnet-4-5")
    assert archive.frames, "middle should be imaged"
    assert archive.text_head and archive.text_tail
    assert len(archive.frames) <= MAX_FRAMES
    # The replay policy records one kind per middle page, whether or not the
    # page keeps its image — chat-shaped pages here flip to text.
    assert len(archive.middle_kinds) == len(archive.frames)
    shape = resolve_shape("anthropic", "claude-sonnet-4-5")
    edge = HQ_EDGE_FRAMES * shape.capacity
    assert len(archive.text_head) == edge
    assert len(archive.text_tail) == edge
    # Re-render source: previous text folds in ahead of the new history.
    again = compact_to_archive(
        _conversation_messages(2), "anthropic", "claude-sonnet-4-5", previous_text=archive.text
    )
    assert again.text.startswith(archive.text[:200])


def test_compact_caps_frames_at_default_budget():
    """Oldest middle pages drop with a marker once the frame budget overflows.

    The default budget is DEFAULT_MAX_FRAMES — the number replay actually
    sends — not the MAX_FRAMES ceiling: rendering 80 frames so foveation
    could throw 74 of them away is where a manual /compact spent ~35 of its
    60 seconds."""
    shape = resolve_shape("openai", "gpt-5.5")
    # Far more text than the default budget's frames can hold.
    messages = [
        Message.user(f"turn {i} " + "w" * shape.capacity) for i in range(DEFAULT_MAX_FRAMES + 40)
    ]
    archive = compact_to_archive(messages, "openai", "gpt-5.5")
    assert len(archive.frames) == DEFAULT_MAX_FRAMES
    assert archive.truncated_chars > 0
    assert "oldest history dropped]" in archive.text_head


def test_compact_max_frames_is_a_ceiling_an_explicit_ask_can_reach():
    """An explicit ``max_frames`` above the default still renders (bounded by
    MAX_FRAMES) — the knob exists for archives meant to be read exhaustively."""
    shape = resolve_shape("openai", "gpt-5.5")
    messages = [Message.user(f"turn {i} " + "w" * shape.capacity) for i in range(20)]
    archive = compact_to_archive(messages, "openai", "gpt-5.5", max_frames=MAX_FRAMES)
    assert DEFAULT_MAX_FRAMES < len(archive.frames) <= MAX_FRAMES


def test_edges_default_to_frame_shape_without_a_window():
    """No ``context_window`` → the verbatim edges keep their full default size.

    This is the wide-window / unknown-window path: the edge cap only ever
    SHRINKS the default, never grows it, so an archive built without a window
    (or on a 1M-token model where the default already fits its share) is byte
    for byte what it was before the cap existed. Guards against the cap
    silently changing the common case.
    """
    shape = resolve_shape("anthropic", "claude-sonnet-4-5")
    messages = _conversation_messages(200)
    archive = compact_to_archive(messages, "anthropic", "claude-sonnet-4-5")
    assert len(archive.text_head) == HQ_EDGE_FRAMES * shape.capacity
    assert len(archive.text_tail) == HQ_EDGE_FRAMES * shape.capacity


def test_small_window_caps_verbatim_edges_to_its_share():
    """A tight window shrinks the verbatim edges to ``EDGE_WINDOW_FRACTION``.

    The regression this locks down: the edges are the archive's un-trimmable
    floor (the frame-budget loop can only drop imaged pages), and at the
    default ``HQ_EDGE_FRAMES * capacity`` they are ~31.5k tokens for an
    Anthropic 1932px reader REGARDLESS of the window. On a small window that
    floor alone exceeds the whole ``0.5 * window`` archive budget, so a pass
    meant to get the context under the line commits it above the line instead.
    Capping the edges to a window share is what keeps them a minority of the
    post-compaction context, and the trimmed oldest-edge text is not lost — it
    flows into the imaged middle.
    """
    # 128k: the window's per-edge share works out ABOVE one page and below
    # the shape default, so the cap binds without hitting the floor — the
    # regime this test is about. (Below ~70k the per-edge share drops under
    # one page and the floor takes over; that path is its own test.)
    window = 128_000
    shape = resolve_shape("anthropic", "claude-opus-4-8")
    default_edge = HQ_EDGE_FRAMES * shape.capacity
    messages = [Message.user(f"turn {i} " + "w" * shape.capacity) for i in range(40)]
    archive = compact_to_archive(messages, "anthropic", "claude-opus-4-8", context_window=window)
    # Each edge is strictly smaller than the shape default. The per-edge
    # budget is checked on text_tail, the clean edge: text_head carries the
    # appended "[... N chars dropped]" truncation marker, which is a handful
    # of chars ON TOP of the bounded slice, so only the tail is exactly the
    # raw window-share slice.
    assert len(archive.text_head) < default_edge
    assert len(archive.text_tail) < default_edge
    per_edge_char_budget = (int(window * EDGE_WINDOW_FRACTION) // 2) * 4
    assert len(archive.text_tail) <= per_edge_char_budget
    # And it really is the window share, not the floor: strictly more than one
    # page, so this exercises the cap rather than the min() safety net.
    assert len(archive.text_tail) > shape.capacity


def test_small_window_archive_replay_fits_its_budget():
    """The whole point: replay cost stays under the ``0.5 * window`` budget.

    Before the edge cap, a 64k-token window produced a ~31.5k-token edge-only
    replay against a 32k-token budget — the frame loop dropped every page and
    the pass STILL overshot. With the edges bounded, the replayed archive
    (text edges + imaged frames) fits. NB 64k is the FLOOR regime for the
    1932px reader (per-edge share works out at one page); the cap-regime
    budget fit is covered by ``test_cap_regime_archive_replay_fits_its_budget``.
    """
    window = 64_000
    shape = resolve_shape("anthropic", "claude-opus-4-8")
    messages = [Message.user(f"turn {i} " + "w" * shape.capacity) for i in range(60)]
    archive = compact_to_archive(messages, "anthropic", "claude-opus-4-8", context_window=window)
    budget = max(frame_token_estimate_for("anthropic", "claude-opus-4-8"), int(window * 0.5))
    assert estimate_archive_tokens(archive) <= budget


def test_cap_regime_archive_replay_fits_its_budget():
    """Budget fit in the CAP regime (edges trimmed below default, above floor).

    At 128k the 1932px per-edge share is 38400 chars — strictly between the
    one-page floor (21000) and the shape default (63000) — so this exercises
    the window-proportional cap doing its job, not the floor override, and
    proves the resulting archive still fits ``0.5 * window``. This is the
    regime the fix actually changes; the 64k test above only reaches the floor.
    """
    window = 128_000
    shape = resolve_shape("anthropic", "claude-opus-4-8")
    # The 128k per-edge char share sits strictly between floor and default.
    assert shape.capacity < 38_400 < HQ_EDGE_FRAMES * shape.capacity
    messages = [Message.user(f"turn {i} " + "w" * shape.capacity) for i in range(80)]
    archive = compact_to_archive(messages, "anthropic", "claude-opus-4-8", context_window=window)
    assert shape.capacity < len(archive.text_tail) < HQ_EDGE_FRAMES * shape.capacity
    budget = max(frame_token_estimate_for("anthropic", "claude-opus-4-8"), int(window * 0.5))
    assert estimate_archive_tokens(archive) <= budget


def test_native_200k_window_trims_the_high_res_opus_edges():
    """The 1932px reader's edges ARE trimmed at its native 200k window.

    Documents the intended, easily-missed behaviour flagged in review: the
    Opus-1932 default edge pair is 31.5k tokens = 15.75% of a 200k window,
    just over the 15% share, so ``_edge_chars_for`` trims each edge from 63000
    to 60000 chars. This is not a regression — the cap is designed to make the
    edges follow the window — but "large windows unchanged" is only true ABOVE
    the shape's threshold (~210k for this reader), so the flagship 200k path is
    guarded here rather than assumed untouched.
    """
    window = 200_000
    shape = resolve_shape("anthropic", "claude-opus-4-8")
    default_edge = HQ_EDGE_FRAMES * shape.capacity  # 63000
    messages = [Message.user(f"turn {i} " + "w" * shape.capacity) for i in range(80)]
    archive = compact_to_archive(messages, "anthropic", "claude-opus-4-8", context_window=window)
    # Trimmed to the 15% share: (200000 * 0.15 // 2) * 4 = 60000, below default.
    assert len(archive.text_tail) == 60_000
    assert len(archive.text_tail) < default_edge
    # A wider window ABOVE the ~210k threshold keeps the full default edge.
    wide = compact_to_archive(messages, "anthropic", "claude-opus-4-8", context_window=400_000)
    assert len(wide.text_tail) == default_edge


def test_edge_cap_keeps_at_least_one_page_of_verbatim_text():
    """A pathologically small window never collapses the edges to nothing.

    Some verbatim anchor always beats none: even when the window share works
    out below one page, the floor is ``shape.capacity`` chars per edge, and
    the frame budget remains the mechanism that bounds total archive size.
    """
    shape = resolve_shape("anthropic", "claude-opus-4-8")
    messages = [Message.user(f"turn {i} " + "w" * shape.capacity) for i in range(40)]
    archive = compact_to_archive(messages, "anthropic", "claude-opus-4-8", context_window=8_000)
    assert len(archive.text_head) >= shape.capacity
    assert len(archive.text_tail) >= shape.capacity


def test_compact_frames_are_valid_pngs():
    messages = _conversation_messages(60, payload=2000)
    archive = compact_to_archive(messages, "anthropic", "claude-sonnet-4-5")
    assert archive.frames
    for frame in archive.frames:
        with Image.open(io.BytesIO(frame)) as img:
            assert img.format == "PNG"
            assert img.size[0] == 1568


# ---------------------------------------------------------------------------
# Replay (history_blocks)
# ---------------------------------------------------------------------------


def _large_archive() -> Archive:
    """A real pass whose pages ALL flip to text under the carrier margin.

    Sized so every page's local estimate (~600-860 tokens) clears the margin
    threshold (``per_frame / CROSS_RULER_SLOPE_MARGIN`` ≈ 1,733 tokens for
    the anthropic shape) with room. The earlier fixture (200 rounds at
    ``payload=400``) sat at ~1,900 and the margin now keeps its frames —
    tests that need the all-TEXT regime use this fixture, tests that need
    images build dense content or a constructed archive.
    """
    return compact_to_archive(
        _conversation_messages(700, payload=100), "anthropic", "claude-sonnet-4-5"
    )


def test_history_blocks_order_text_images_text():
    """An all-image archive replays edges → frames → edges, byte for byte."""
    archive = _constructed_archive("iii")
    blocks = history_blocks(archive)
    assert blocks[0]["kind"] == "text"
    assert blocks[0]["text"].startswith(archive.text_head)
    kinds = [b["kind"] for b in blocks]
    assert kinds == ["text", "images", "text"]
    assert blocks[-1]["text"] == archive.text_tail
    frames = blocks[1]["frames"]
    assert frames and all(isinstance(f, str) for f in frames)
    with Image.open(io.BytesIO(base64.b64decode(frames[0]))) as img:
        assert img.format == "PNG"


def test_history_blocks_policy_replays_text_pages_as_text():
    """A mixed policy archive: text pages ride as text, image pages as frames."""
    page_text = _constructed_pages("tiit")
    archive = _constructed_archive("tiit")
    blocks = history_blocks(archive)
    assert [b["kind"] for b in blocks] == ["text", "text", "images", "text", "text"]
    assert blocks[0]["text"] == archive.text_head + "\n-------------- imaged middle below\n"
    assert blocks[1]["text"] == page_text[0]
    assert blocks[3]["text"] == page_text[3]
    assert blocks[4]["text"] == archive.text_tail
    assert blocks[2]["frames"] == [
        base64.b64encode(archive.frames[1]).decode("ascii"),
        base64.b64encode(archive.frames[2]).decode("ascii"),
    ]


def test_history_blocks_policy_all_text_middle_is_char_complete():
    """Nothing is dropped in a flip: every archived character is still sent."""
    archive = _large_archive()
    assert set(archive.middle_kinds) == {"t"}
    blocks = history_blocks(archive)
    assert [b["kind"] for b in blocks] == ["text", "text", "text"]
    assert blocks[0]["text"] == archive.text_head
    assert blocks[2]["text"] == archive.text_tail
    assert blocks[0]["text"] + blocks[1]["text"] + blocks[2]["text"] == archive.text


def test_history_blocks_policy_all_image_matches_legacy():
    """An all-image policy archive replays exactly as the legacy path does.

    Up to foveation: with more images than the HQ budget a policy archive
    puts the elision marker where the missing pages stood (after its first
    kept frames), while the legacy path puts it before all frames — both
    orders are pinned in ``test_history_blocks_foveates_long_middle``.
    """
    archive = _constructed_archive("iii")
    legacy = archive.model_copy(update={"middle_kinds": ""})
    assert history_blocks(archive) == history_blocks(legacy)


def test_history_blocks_degrades_to_images_on_unreconcilable_kinds():
    archive = _constructed_archive("tiit")
    legacy = history_blocks(archive.model_copy(update={"middle_kinds": ""}))
    for broken in ("ti", "tixx", "iiiiii", "TTii"):
        patched = archive.model_copy(update={"middle_kinds": broken})
        assert history_blocks(patched) == legacy
        # The guide and the replay disagreeing about which frames a degrade
        # carries was round-1 review MINOR-1: the surfaces share one gate.
        assert replayed_frame_count(patched) == len(patched.frames)
        assert "image frame" in archive_summary(patched, "openai", "gpt-5.5")
    # A missing/foreign shape cannot re-derive the pages either, nor can text
    # whose edges no longer slice it, nor digests that do not match — the
    # gate binds the page STRINGS, not just their count.
    no_shape = archive.model_copy(update={"shape_id": "foreign"})
    assert history_blocks(no_shape) == legacy
    assert replayed_frame_count(no_shape) == len(no_shape.frames)
    assert "image frame" in archive_summary(no_shape, "openai", "gpt-5.5")
    assert history_blocks(archive.model_copy(update={"text": archive.text + "XYZ"})) == legacy
    tampered = archive.model_copy(update={"page_digests": ["deadbeef"] * 4})
    assert history_blocks(tampered) == legacy
    assert replayed_frame_count(tampered) == len(tampered.frames)
    dropped = archive.model_copy(update={"page_digests": []})
    assert history_blocks(dropped) == legacy


def test_replayed_frame_count():
    assert replayed_frame_count(_constructed_archive("tiit")) == 2
    assert replayed_frame_count(_constructed_archive("iiii")) == 4  # all-image policy
    legacy = _constructed_archive("iiii").model_copy(update={"middle_kinds": ""})
    assert replayed_frame_count(legacy) == 4  # legacy: every stored frame replays
    junk = _constructed_archive("tiit").model_copy(update={"middle_kinds": "tixx"})
    assert replayed_frame_count(junk) == 4  # junk spelling degrades to legacy
    short = _constructed_archive("tiit").model_copy(update={"middle_kinds": "ti"})
    assert replayed_frame_count(short) == 4  # length mismatch degrades
    no_digests = _constructed_archive("tiit").model_copy(update={"page_digests": []})
    assert replayed_frame_count(no_digests) == 4  # unproven pages degrade


def test_history_blocks_elides_over_budget():
    """A tight byte budget keeps only the NEWEST frames (oldest elided)."""
    archive = _constructed_archive("iiiii")
    assert len(archive.frames) > 1
    blocks = history_blocks(archive, max_frame_data_bytes=1)  # under one frame
    images = [b for b in blocks if b["kind"] == "images"]
    assert len(images) == 1 and len(images[0]["frames"]) == 1
    elided = [b for b in blocks if b["kind"] == "text" and "frames elided]" in b["text"]]
    assert elided and f"[{len(archive.frames) - 1} frames elided]" in elided[0]["text"]


def test_history_blocks_foveates_long_middle():
    """Interior frames beyond the HQ edges collapse to an elision marker.

    Reached through the replay of an archive PERSISTED by an older build,
    which could carry up to 80 frames; a fresh pass renders only what replay
    sends. The policy path and the legacy path must both foveate — the legacy
    twin places the marker between the head and all kept frames, which is the
    order every pre-policy archive replayed with."""
    archive = _constructed_archive("i" * 10)
    assert len(archive.frames) > 2 * HQ_EDGE_FRAMES + 2

    blocks = history_blocks(archive)
    assert [b["kind"] for b in blocks] == ["text", "images", "text", "images", "text"]
    assert "4 frames elided]" in blocks[2]["text"]
    assert [len(b["frames"]) for b in blocks if b["kind"] == "images"] == [
        HQ_EDGE_FRAMES,
        HQ_EDGE_FRAMES,
    ]

    legacy = archive.model_copy(update={"middle_kinds": ""})
    legacy_blocks = history_blocks(legacy)
    assert [b["kind"] for b in legacy_blocks] == ["text", "text", "images", "text"]
    assert "4 frames elided]" in legacy_blocks[1]["text"]
    assert len(legacy_blocks[2]["frames"]) == 2 * HQ_EDGE_FRAMES


def test_history_blocks_text_only_archive():
    archive = compact_to_archive(
        _conversation_messages(2, payload=50), "anthropic", "claude-sonnet-4-5"
    )
    blocks = history_blocks(archive)
    assert [b["kind"] for b in blocks] == ["text"]
    assert blocks[0]["text"] == archive.text_head


# ---------------------------------------------------------------------------
# Persistence round trip (the base64 contract)
# ---------------------------------------------------------------------------


def test_persisted_archive_replays_real_pngs():
    """The FULL production path, with frames: the live archive is dumped to
    JSON, revived, and replayed.

    This is the path every request after a compaction takes (the marker carries
    the dump, in-process as well as on resume), and it was broken while every
    frame test above stayed green: the dump encoded base64 and revival ran
    pydantic's lax ``str``->``bytes`` coercion, which UTF-8-ENCODED that text
    instead of decoding it, so ``history_blocks`` encoded a second time and the
    provider was handed ``base64(base64(png))`` labelled ``image/png``.
    ``6956424f`` (ASCII ``iVBO``) is the corrupt magic to watch for.
    """
    archive = _constructed_archive("ii")
    assert archive.frames

    revived = Archive.model_validate(_archive_to_json(archive))

    assert revived.frames == archive.frames  # byte-for-byte, not merely decodable
    assert revived.middle_kinds == archive.middle_kinds
    assert revived.page_digests == archive.page_digests
    frames = [b for b in history_blocks(revived) if b["kind"] == "images"][0]["frames"]
    assert base64.b64decode(frames[0]).startswith(b"\x89PNG\r\n\x1a\n")
    with Image.open(io.BytesIO(base64.b64decode(frames[0]))) as img:
        assert img.format == "PNG"


def test_archive_json_dump_keeps_its_persisted_shape():
    """``_archive_to_json`` delegates to the model now; the emitted shape is a
    contract both ways — new keys are additive with replay-safe defaults, and
    payloads written before them keep loading."""
    archive = _large_archive()
    payload = _archive_to_json(archive)
    assert set(payload) == {
        "frames",
        "text",
        "text_head",
        "text_tail",
        "shape_id",
        "truncated_chars",
        "middle_kinds",
        "page_digests",
        "created_at",
    }
    assert payload["middle_kinds"] == archive.middle_kinds
    assert payload["page_digests"] == archive.page_digests
    assert payload["frames"] == [base64.b64encode(f).decode("ascii") for f in archive.frames]
    assert payload["created_at"] == archive.created_at.isoformat()
    assert json.loads(json.dumps(payload)) == payload

    # A payload written before the replay-policy keys existed: both default
    # to the all-image replay (no kinds -> no policy path).
    older = {k: v for k, v in payload.items() if k not in ("middle_kinds", "page_digests")}
    revived = Archive.model_validate(older)
    assert revived.middle_kinds == ""
    assert revived.page_digests == []


def test_malformed_persisted_frame_is_a_validation_error():
    """Garbage in the frames list must FAIL loudly. Silent coercion is what
    turned a defect into a 400 the session answered by dropping the history."""
    payload = _archive_to_json(_large_archive())
    with pytest.raises(ValidationError):
        Archive.model_validate({**payload, "frames": ["not base64 at all!!"]})


# ---------------------------------------------------------------------------
# Integration helpers
# ---------------------------------------------------------------------------


def test_frame_token_estimate_follows_provider_billing():
    """Per-family image billing (mirrors omp's ``familyBilling``, verified
    against live bills there): Anthropic patch pricing capped at 4,784 visual
    tokens +5%, OpenAI 32px patches x1.2, Gemini a flat 1,120 per image. The
    cross-provider ceiling (5024) stays exported for callers with no reader
    at hand, but pricing a Gemini frame at it overstated the archive 4.5x."""
    # 1932px: 69^2 = 4,761 patches, UNDER the 4,784 cap; x1.05 -> 5,000.
    assert frame_token_estimate_for("anthropic", "claude-fable-5") == 5000
    assert frame_token_estimate_for("anthropic", "claude-sonnet-4-5") == 3293  # 1568px
    assert frame_token_estimate_for("openai", "gpt-5.5") == 2882  # 49^2 * 1.2
    assert frame_token_estimate_for("google", "gemini-3") == 1120  # flat HIGH budget
    # Unknown families take Anthropic's formula at 1568px: the safe ceiling.
    assert frame_token_estimate_for("mystery", "model-x") == 3293
    # Every estimate stays under the exported ceiling.
    for provider, model in [
        ("anthropic", "claude-fable-5"),
        ("openai", "gpt-5.5"),
        ("google", "gemini-3"),
    ]:
        assert frame_token_estimate_for(provider, model) <= FRAME_TOKEN_ESTIMATE


def test_archive_summary_is_deterministic_and_structural():
    """The snapcompact text slot is built locally from the archive — no LLM
    call (the branch's contract, and the 20-50s the manual pass used to spend).
    Every claim in it derives from the archive, so it can never contradict
    the frames it captions."""
    archive = _constructed_archive("ii")
    summary = archive_summary(archive, "anthropic", "claude-sonnet-4-5")
    assert summary == archive_summary(archive, "anthropic", "claude-sonnet-4-5")
    assert str(len(archive.frames)) in summary
    assert "image frame" in summary
    shape = resolve_shape("openai", "gpt-5.5")
    assert str(shape.chars_per_line) in summary

    # A policy archive whose pages all flip to text does not describe frames
    # that never replay — and one that mixes says so.
    flipped = _large_archive()
    assert set(flipped.middle_kinds) == {"t"}
    flipped_summary = archive_summary(flipped, "anthropic", "claude-sonnet-4-5")
    assert "image frame" not in flipped_summary
    assert "plain text" in flipped_summary
    mixed = archive_summary(_constructed_archive("ti"), "openai", "gpt-5.5")
    assert "partly as plain text" in mixed

    # A text-only archive does not describe frames it does not have.
    small = compact_to_archive(
        _conversation_messages(2, payload=50), "anthropic", "claude-sonnet-4-5"
    )
    text_only = archive_summary(small, "anthropic", "claude-sonnet-4-5")
    assert "image frame" not in text_only

    # A truncated archive says so.
    shape_o = resolve_shape("openai", "gpt-5.5")
    big = compact_to_archive(
        [Message.user(f"t{i} " + "w" * shape_o.capacity) for i in range(DEFAULT_MAX_FRAMES + 40)],
        "openai",
        "gpt-5.5",
    )
    assert big.truncated_chars > 0
    assert "dropped" in archive_summary(big, "openai", "gpt-5.5")


def test_strategy_for_model_both_branches():
    vision = ModelSpec(provider="anthropic", model_id="claude-sonnet-4-5", supports_images=True)
    assert strategy_for_model(vision) == "snapcompact"
    blind = ModelSpec(provider="ollama", model_id="llama3", supports_images=False)
    assert strategy_for_model(blind) == "context-full"


def test_middle_kinds_flips_chat_pages_to_text_and_keeps_dense_images():
    """Per-page policy: text where it clears the margin, image where not."""
    sparse = _large_archive()
    assert sparse.frames
    assert set(sparse.middle_kinds) == {"t"}
    # Frames are still rendered for the text pages: an archive read by a
    # build that predates middle_kinds replays complete, at the old cost.
    assert len(sparse.frames) == len(sparse.middle_kinds)

    dense = compact_to_archive(_dense_messages(12), "anthropic", "claude-sonnet-4-5")
    assert dense.frames
    assert dense.middle_kinds.count("i") >= len(dense.frames) - 1


def test_middle_page_kinds_keeps_pages_the_slope_margin_does_not_clear():
    """The flip must clear the worst measured provider/local divergence.

    ``CROSS_RULER_SLOPE_MARGIN`` exists because the flip compares a local
    estimate against a provider figure (see the constant): a page cheaper
    only on the local ruler keeps its frame."""
    per_frame = frame_token_estimate_for("anthropic", "claude-sonnet-4-5")
    filler = "ZXCVBNMASDFGHJKLqwertyuiop1234567890mMnBvC" * 100
    small, large = filler[:1900], filler[:4200]
    small_tokens = estimate_tokens(Message.assistant(small))
    large_tokens = estimate_tokens(Message.assistant(large))
    # Premises, so the test fails loudly if tokenization drifts out of band:
    # the small page clears the margin; the large one is cheaper on the LOCAL
    # ruler alone and only the margin keeps its frame.
    assert small_tokens * CROSS_RULER_SLOPE_MARGIN < per_frame
    assert large_tokens * CROSS_RULER_SLOPE_MARGIN >= per_frame
    assert large_tokens < per_frame
    kinds = _middle_page_kinds([small, large], "anthropic", "claude-sonnet-4-5")
    assert kinds == "ti"


def test_estimate_archive_tokens():
    """The estimator prices the REPLAY: text carried plus frames sent."""
    all_image = _constructed_archive("iii")
    edges = estimate_tokens(Message.assistant(all_image.text_head + all_image.text_tail))
    assert estimate_archive_tokens(all_image) == edges + 3 * FRAME_TOKEN_ESTIMATE

    # A fully text-carried middle drops the frame term: the estimate is just
    # the archived text, because the flipped pages join back to it exactly.
    flipped = _large_archive()
    assert replayed_frame_count(flipped) == 0
    assert estimate_archive_tokens(flipped) == estimate_tokens(Message.assistant(flipped.text))

    # Legacy archives price every stored frame, exactly as before.
    legacy = all_image.model_copy(update={"middle_kinds": ""})
    assert estimate_archive_tokens(legacy) == edges + 3 * FRAME_TOKEN_ESTIMATE


def test_archive_created_at_set():
    archive = compact_to_archive(
        _conversation_messages(1, payload=50), "anthropic", "claude-sonnet-4-5"
    )
    assert archive.created_at.tzinfo is not None
    assert archive.created_at <= datetime.now(timezone.utc)


def test_default_budget_constant():
    assert FRAME_DATA_BYTES_BUDGET == 3_000_000
