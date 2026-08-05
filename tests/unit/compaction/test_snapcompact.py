"""snapcompact: shapes, serialization, rendering, archiving, replay."""

import base64
import io
from datetime import datetime, timezone

from PIL import Image

from local_operator.compaction.snapcompact import (
    FRAME_DATA_BYTES_BUDGET,
    FRAME_TOKEN_ESTIMATE,
    HQ_EDGE_FRAMES,
    MAX_FRAMES,
    Archive,
    compact_to_archive,
    estimate_archive_tokens,
    history_blocks,
    render_frame,
    resolve_shape,
    serialize_for_snapcompact,
    strategy_for_model,
)
from local_operator.compaction.api import TOOL_ARGS_MAX_CHARS, TOOL_RESULT_MAX_CHARS
from local_operator.harness.types import (
    ImageContent,
    Message,
    ModelSpec,
    TextContent,
    ToolCall,
)

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


def _tool_message(call_id: str, text: str, *, useless: bool = False, is_error: bool = False) -> Message:
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
        Message(role="assistant", content=[], tool_calls=[ToolCall(id="e1", name="bash", arguments={})]),
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
        # White-on-black: ink on the first glyph row, black at the bottom edge.
        ink = sum(1 for x in range(0, shape.page_width_px, 3) if pixels[x, 3] > 128)
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
    shape = resolve_shape("anthropic", "claude-sonnet-4-5")
    edge = HQ_EDGE_FRAMES * shape.capacity
    assert len(archive.text_head) == edge
    assert len(archive.text_tail) == edge
    # Re-render source: previous text folds in ahead of the new history.
    again = compact_to_archive(
        _conversation_messages(2), "anthropic", "claude-sonnet-4-5", previous_text=archive.text
    )
    assert again.text.startswith(archive.text[:200])


def test_compact_caps_frames_at_max():
    """Oldest middle pages drop with a marker once the frame budget overflows."""
    shape = resolve_shape("openai", "gpt-5.5")
    # Far more text than MAX_FRAMES frames can hold.
    messages = [Message.user(f"turn {i} " + "w" * shape.capacity) for i in range(MAX_FRAMES + 40)]
    archive = compact_to_archive(messages, "openai", "gpt-5.5")
    assert len(archive.frames) == MAX_FRAMES
    assert archive.truncated_chars > 0
    assert "oldest history dropped]" in archive.text_head


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
    return compact_to_archive(_conversation_messages(200), "anthropic", "claude-sonnet-4-5")


def test_history_blocks_order_text_images_text():
    archive = _large_archive()
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


def test_history_blocks_elides_over_budget():
    """A tight byte budget keeps only the NEWEST frames (oldest elided)."""
    archive = _large_archive()
    assert len(archive.frames) > 1
    blocks = history_blocks(archive, max_frame_data_bytes=1)  # under one frame
    images = [b for b in blocks if b["kind"] == "images"]
    assert len(images) == 1 and len(images[0]["frames"]) == 1
    elided = [b for b in blocks if b["kind"] == "text" and "frames elided]" in b["text"]]
    assert elided and f"[{len(archive.frames) - 1} frames elided]" in elided[0]["text"]


def test_history_blocks_foveates_long_middle():
    """Interior frames beyond the HQ edges collapse to an elision marker."""
    shape = resolve_shape("anthropic", "claude-sonnet-4-5")
    page = "f" * (shape.capacity // 2)
    archive = compact_to_archive(
        [Message.user(f"t{i} {page}") for i in range(200)],
        "anthropic",
        "claude-sonnet-4-5",
    )
    assert len(archive.frames) > 2 * HQ_EDGE_FRAMES + 2
    blocks = history_blocks(archive)
    kinds = [b["kind"] for b in blocks]
    assert kinds == ["text", "text", "images", "text"]  # marker between head and frames
    assert "frames elided]" in blocks[1]["text"]
    kept = blocks[2]["frames"]
    assert len(kept) == 2 * HQ_EDGE_FRAMES


def test_history_blocks_text_only_archive():
    archive = compact_to_archive(_conversation_messages(2, payload=50), "anthropic", "claude-sonnet-4-5")
    blocks = history_blocks(archive)
    assert [b["kind"] for b in blocks] == ["text"]
    assert blocks[0]["text"] == archive.text_head


# ---------------------------------------------------------------------------
# Integration helpers
# ---------------------------------------------------------------------------


def test_strategy_for_model_both_branches():
    vision = ModelSpec(provider="anthropic", model_id="claude-sonnet-4-5", supports_images=True)
    assert strategy_for_model(vision) == "snapcompact"
    blind = ModelSpec(provider="ollama", model_id="llama3", supports_images=False)
    assert strategy_for_model(blind) == "context-full"


def test_estimate_archive_tokens():
    archive = _large_archive()
    tokens = estimate_archive_tokens(archive)
    assert tokens >= len(archive.frames) * FRAME_TOKEN_ESTIMATE
    # Frames dominate; the text edges add a bounded positive remainder.
    assert tokens - len(archive.frames) * FRAME_TOKEN_ESTIMATE > 0


def test_archive_created_at_set():
    archive = compact_to_archive(_conversation_messages(1, payload=50), "anthropic", "claude-sonnet-4-5")
    assert archive.created_at.tzinfo is not None
    assert archive.created_at <= datetime.now(timezone.utc)


def test_default_budget_constant():
    assert FRAME_DATA_BYTES_BUDGET == 3_000_000
