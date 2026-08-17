"""End-to-end tests for the builtin tools against a temp working directory.

Covers the review findings RT-27..RT-32 explicitly: subprocess lifecycle
(abort/timeout/pre-abort), the ToolResult invariant sweep, pydantic
ValidationError containment, truncation shape, unexpected-exception safety,
and range-beyond-EOF.
"""

from __future__ import annotations

import asyncio
import base64
import io
import os
import random
import struct
import threading
import time
import zlib
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from local_operator.harness.types import (
    AbortSignal,
    AgentTool,
    ImageContent,
    TextContent,
    ToolContext,
    ToolResult,
)
from local_operator.tools import builtin
from local_operator.tools.registry import create_tools


@pytest.fixture
def context(tmp_path: Path) -> ToolContext:
    return ToolContext(cwd=str(tmp_path), session_id="unit-test")


@pytest.fixture
def tools(context: ToolContext) -> dict[str, AgentTool]:
    return {tool.name: tool for tool in create_tools(context)}


async def _call(
    tools: dict[str, AgentTool], name: str, args: dict[str, Any], context: ToolContext
) -> ToolResult:
    tool = tools[name]
    return await tool.execute("call-1", args, None, None, context)  # type: ignore[operator]


class RecordingApproval:
    """Records every approval request; configurable grant/deny."""

    def __init__(self, approve: bool = True) -> None:
        self.approve = approve
        self.requests: list[tuple[str, str]] = []

    async def __call__(self, tier: str, description: str) -> bool:
        self.requests.append((tier, description))
        return self.approve


class _RecordingContext(ToolContext):
    """ToolContext with the approval recorder DECLARED so tests can read it back."""

    recorder: RecordingApproval


def _context_with_approval(tmp_path: Path, approve: bool = True) -> _RecordingContext:
    approval = RecordingApproval(approve)
    return _RecordingContext(
        cwd=str(tmp_path),
        session_id="unit-test",
        request_approval=approval,
        recorder=approval,
    )


# ---------------------------------------------------------------------------
# bash
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bash_echo_and_streams(tools, context) -> None:
    result = await _call(tools, "bash", {"command": "echo hello && echo bad 1>&2"}, context)
    assert result.is_error is False
    assert "hello" in result.text
    assert "bad" in result.text
    assert "exit code: 0" in result.text


@pytest.mark.asyncio
async def test_bash_nonzero_exit_reported(tools, context) -> None:
    result = await _call(tools, "bash", {"command": "exit 3"}, context)
    assert "exit code: 3" in result.text


@pytest.mark.asyncio
async def test_bash_non_interactive_env_applied(tools, context) -> None:
    result = await _call(tools, "bash", {"command": 'echo "$CI:$NO_COLOR:$TERM"'}, context)
    assert "1:1:dumb" in result.text


@pytest.mark.asyncio
async def test_bash_timeout_kills_and_marks(tools, context) -> None:
    result = await _call(tools, "bash", {"command": "sleep 5", "timeout": 0.2}, context)
    assert "TIMEOUT" in result.text
    assert result.is_error is False


@pytest.mark.asyncio
async def test_bash_timeout_rejects_zero_and_huge(tools, context) -> None:
    zero = await _call(tools, "bash", {"command": "echo hi", "timeout": 0}, context)
    assert zero.is_error is True
    assert "invalid arguments" in zero.text
    huge = await _call(tools, "bash", {"command": "echo hi", "timeout": 99999}, context)
    assert huge.is_error is True
    assert "invalid arguments" in huge.text


@pytest.mark.asyncio
async def test_bash_timeout_kills_descendants_and_keeps_partial_output(tools, context) -> None:
    # RT-27: the timeout must kill the whole process group (the background
    # child included) and still return the output produced before the kill.
    marker = context.cwd + "/timeout-child.pid"
    cmd = f"(sleep 30 & echo $! > {marker}; echo started; sleep 30) & wait"
    result = await _call(tools, "bash", {"command": cmd, "timeout": 0.6}, context)
    assert "TIMEOUT" in result.text
    assert "started" in result.text  # partial output preserved

    # The descendant must be gone: its pid must not be alive anymore.
    await asyncio.sleep(0.1)
    pid = int(Path(marker).read_text().strip())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


@pytest.mark.asyncio
async def test_bash_abort_kills_process_group(tools, context) -> None:
    # RT-27: a mid-run abort kills the session group, descendants included.
    marker = context.cwd + "/abort-child.pid"
    cmd = f"sleep 30 & echo $! > {marker}; sleep 30"
    signal = AbortSignal()

    async def abort_soon() -> None:
        await asyncio.sleep(0.5)
        signal.abort("stop")

    abort_task = asyncio.create_task(abort_soon())
    result = await tools["bash"].execute("c", {"command": cmd}, signal, None, context)
    await abort_task

    assert result.is_error is True
    assert "aborted" in result.text and "stop" in result.text

    await asyncio.sleep(0.1)
    pid = int(Path(marker).read_text().strip())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


@pytest.mark.asyncio
async def test_bash_pre_aborted_signal_spawns_no_child(tools, context) -> None:
    # RT-27/RT-01: an already-aborted signal returns immediately and leaves
    # no child process behind.
    signal = AbortSignal()
    signal.abort("early")
    marker = context.cwd + "/should-not-exist.pid"
    cmd = f"sleep 30 & echo $! > {marker}; sleep 30"
    result = await tools["bash"].execute("c", {"command": cmd}, signal, None, context)
    assert result.is_error is True
    assert "aborted" in result.text
    assert not Path(marker).exists()  # the command never ran


@pytest.mark.asyncio
async def test_bash_streams_updates_while_running(tools, context) -> None:
    # RT-19: accumulated output reaches on_update while the command runs.
    updates: list[str] = []

    def on_update(update) -> None:
        from local_operator.harness.types import TextContent

        updates.append("".join(b.text for b in update.content if isinstance(b, TextContent)))

    cmd = "echo part-one; sleep 0.7; echo part-two; sleep 0.7"
    result = await tools["bash"].execute("c", {"command": cmd}, None, on_update, context)
    assert result.is_error is False
    assert updates, "expected at least one tool_execution_update payload"
    assert any("part-one" in u for u in updates)


@pytest.mark.asyncio
async def test_bash_large_output_truncated(tools, context) -> None:
    # RT-12/RT-30: one combined budget, head+tail survive, marker present,
    # result never exceeds the limit.
    cmd = "python3 -c \"import sys; sys.stdout.write('A' * 60000)\""
    result = await _call(tools, "bash", {"command": cmd}, context)
    assert "truncated" in result.text.lower()
    assert builtin.BASH_TRUNCATION_MARKER.strip() in result.text
    stdout_section = result.text.split("--- stdout ---\n", 1)[1].split("\n--- stderr ---")[0]
    assert stdout_section.startswith("A" * 1000)  # head prefix survives
    assert stdout_section.rstrip().endswith("A" * 1000)  # tail suffix survives
    assert result.text.count("A") < 60000
    # The single combined budget holds across both streams.
    assert len(stdout_section) <= builtin.BASH_OUTPUT_LIMIT_CHARS


@pytest.mark.asyncio
async def test_bash_empty_command_is_error(tools, context) -> None:
    result = await _call(tools, "bash", {"command": "   "}, context)
    assert result.is_error is True


@pytest.mark.asyncio
async def test_bash_executes_without_tool_level_prompt(tmp_path) -> None:
    # The write/exec approval gate is the LOOP's (it fires after
    # tool_execution_start and sees the pending call). The tool itself must
    # NOT prompt a second time: one gate per action, no tier-named prompt.
    context = _context_with_approval(tmp_path, approve=True)
    tools = {t.name: t for t in create_tools(context)}
    result = await _call(tools, "bash", {"command": "echo ok"}, context)
    assert result.is_error is False
    assert context.recorder.requests == []


# ---------------------------------------------------------------------------
# read / write / edit roundtrip
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_write_read_edit_roundtrip(tools, context, tmp_path) -> None:
    target = tmp_path / "doc.txt"

    wrote = await _call(
        tools, "write", {"path": "doc.txt", "content": "line one\nline two\n"}, context
    )
    assert wrote.is_error is False
    assert target.read_text() == "line one\nline two\n"

    read = await _call(tools, "read", {"path": "doc.txt"}, context)
    assert "line one" in read.text and "line two" in read.text

    edited = await _call(
        tools,
        "edit",
        {"path": "doc.txt", "old_text": "line two", "new_text": "LINE 2"},
        context,
    )
    assert edited.is_error is False
    assert target.read_text() == "line one\nLINE 2\n"


@pytest.mark.asyncio
async def test_write_creates_parents(tools, context, tmp_path) -> None:
    await _call(tools, "write", {"path": "a/b/c.txt", "content": "deep"}, context)
    assert (tmp_path / "a" / "b" / "c.txt").read_text() == "deep"


@pytest.mark.asyncio
async def test_edit_missing_text_is_error(tools, context) -> None:
    await _call(tools, "write", {"path": "f.txt", "content": "abc"}, context)
    result = await _call(
        tools,
        "edit",
        {"path": "f.txt", "old_text": "nothere", "new_text": "x"},
        context,
    )
    assert result.is_error is True


@pytest.mark.asyncio
async def test_edit_ambiguous_requires_replace_all(tools, context, tmp_path) -> None:
    await _call(tools, "write", {"path": "dup.txt", "content": "foo\nfoo\n"}, context)

    ambiguous = await _call(
        tools,
        "edit",
        {"path": "dup.txt", "old_text": "foo", "new_text": "bar"},
        context,
    )
    assert ambiguous.is_error is True
    assert (tmp_path / "dup.txt").read_text() == "foo\nfoo\n"  # untouched

    all_replaced = await _call(
        tools,
        "edit",
        {"path": "dup.txt", "old_text": "foo", "new_text": "bar", "replace_all": True},
        context,
    )
    assert all_replaced.is_error is False
    assert (tmp_path / "dup.txt").read_text() == "bar\nbar\n"


@pytest.mark.asyncio
async def test_read_missing_path_is_error(tools, context) -> None:
    result = await _call(tools, "read", {"path": "ghost.txt"}, context)
    assert result.is_error is True


@pytest.mark.asyncio
async def test_read_line_range(tools, context) -> None:
    await _call(tools, "write", {"path": "r.txt", "content": "a\nb\nc\n"}, context)
    result = await _call(tools, "read", {"path": "r.txt", "range": "2-3"}, context)
    assert "2" in result.text and "b" in result.text and "c" in result.text
    assert "a\n" not in result.text


@pytest.mark.asyncio
async def test_read_range_beyond_eof_is_useless(tools, context) -> None:
    # RT-32: a range past the last line is useless, not an error.
    await _call(tools, "write", {"path": "short.txt", "content": "a\nb\n"}, context)
    result = await _call(tools, "read", {"path": "short.txt", "range": "50-60"}, context)
    assert result.useless is True
    assert result.is_error is False
    assert result.details is not None
    assert result.details.get("useless") is True


@pytest.mark.asyncio
async def test_read_large_file_capped_with_footer(tools, context, tmp_path) -> None:
    # RT-06: files over the budget render the head plus a footer naming the
    # exact call that continues. The binding cap is now CHARS, not the 2,000-
    # line cap: 2,000 lines of source is ~80 KB, which measured at ~20k tokens
    # for a single read — the line cap was never a context budget.
    lines = [f"line {i}" for i in range(1, 2501)]
    (tmp_path / "big.txt").write_text("\n".join(lines))
    result = await _call(tools, "read", {"path": "big.txt"}, context)
    assert result.is_error is False
    assert "line 1" in result.text
    assert "line 2500" not in result.text
    assert len(result.text) <= builtin.READ_OUTPUT_LIMIT_CHARS + 400  # body + footer
    # The footer must name a concrete, usable continuation, not just report a
    # loss: an agent that cannot tell how to get the rest re-reads or guesses.
    assert "read(path=" in result.text and 'range="' in result.text

    # The range genuinely continues past wherever the cap landed.
    more = await _call(tools, "read", {"path": "big.txt", "range": "2001-2500"}, context)
    assert "line 2500" in more.text


@pytest.mark.asyncio
async def test_read_refuses_oversized_file(tools, context, tmp_path) -> None:
    # RT-06: stat-first refusal above 2MB with an actionable message.
    big = tmp_path / "huge.bin"
    with big.open("wb") as fh:
        fh.write(b"x" * (builtin.READ_FILE_LIMIT_BYTES + 1))
    result = await _call(tools, "read", {"path": "huge.bin"}, context)
    assert result.is_error is True
    assert "too large" in result.text.lower()
    assert "bash" in result.text


# ---------------------------------------------------------------------------
# read: images
# ---------------------------------------------------------------------------


def _write_png(
    path: Path, size: tuple[int, int], noise: str | None = None, colours: int = 0
) -> Path:
    """A real PNG on disk.

    ``noise`` defeats PNG compression, which is what drives the file over the
    byte budget and reaches the lossy rung — a flat fill never gets close.
    ``smooth`` noise is photographic and compresses better as JPEG; ``sharp``
    noise over a small ``colours`` palette is the inverse, the case where PNG
    wins and the lossy rung must decline itself.

    Saved with maximum compression on purpose: PIL's default settings
    round-trip an image it wrote itself to BYTE-IDENTICAL output, which would
    silently make any "was this forwarded verbatim?" assertion vacuous. Real
    PNGs on disk come from other encoders, so this is also the honest fixture.
    """
    image = Image.new("RGB", size, (10, 60, 120))
    if noise:
        rng = random.Random(1234)
        pixels = image.load()
        assert pixels is not None
        palette = [
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (255, 255, 0),
            (0, 255, 255),
            (255, 0, 255),
            (255, 255, 255),
            (0, 0, 0),
        ][:colours]
        for y in range(size[1]):
            if noise == "sharp":
                for x in range(size[0]):
                    pixels[x, y] = rng.choice(palette)
                continue
            for x in range(0, size[0], 4):
                value = rng.randint(0, 255)
                for offset in range(4):
                    pixels[x + offset, y] = (value, (value * 3) % 256, (value + 77) % 256)
    image.save(path, format="PNG", optimize=True, compress_level=9)
    return path


def _image_blocks(result: ToolResult) -> list[ImageContent]:
    return [block for block in result.content if isinstance(block, ImageContent)]


@pytest.mark.asyncio
async def test_read_png_returns_caption_then_image_block(tools, context, tmp_path) -> None:
    # The whole point of the feature: the model receives the pixels, not
    # "Binary file not readable as text". The caption leads because every
    # text-only consumer (ToolResult.text, compaction, the TUI row) sees that
    # and nothing else, and a bare image says neither what nor whether.
    source = _write_png(tmp_path / "shot.png", (320, 200))
    result = await _call(tools, "read", {"path": "shot.png"}, context)

    assert result.is_error is False
    assert [type(block) for block in result.content] == [TextContent, ImageContent]
    caption, image = result.content
    assert isinstance(caption, TextContent) and isinstance(image, ImageContent)
    assert "shot.png" in caption.text
    assert "image/png" in caption.text and "320x200" in caption.text
    assert image.mime_type == "image/png"
    # In-bounds images go over the wire byte-for-byte: a re-encode can only
    # lose fidelity for an image the model sees at its original size.
    assert base64.b64decode(image.data) == source.read_bytes()


@pytest.mark.asyncio
async def test_read_png_over_the_edge_cap_is_resized(tools, context, tmp_path) -> None:
    # Pixels, not bytes, are what an image costs in context (~w*h/750 tokens),
    # and Anthropic resizes past 1568 anyway while billing the resized count —
    # so anything above the cap is upload the model never benefits from.
    _write_png(tmp_path / "wide.png", (3000, 1500))
    result = await _call(tools, "read", {"path": "wide.png"}, context)

    assert result.is_error is False
    (image,) = _image_blocks(result)
    with Image.open(io.BytesIO(base64.b64decode(image.data))) as delivered:
        assert max(delivered.size) == builtin.READ_IMAGE_MAX_EDGE
        assert delivered.size == (1568, 784)
    # What the model sees and what is on disk now differ; the caption must say
    # so or a later `ls -l` looks like it contradicts the read.
    assert "1568x784" in result.text
    assert "source 3000x1500 image/png" in result.text


@pytest.mark.asyncio
async def test_read_photographic_png_falls_back_to_jpeg(tools, context, tmp_path) -> None:
    # PNG is the right default (screenshots of small text are what this tool
    # reads), but it is a bad photographic codec: the lossy rung exists so an
    # image PNG cannot compress does not ride to the provider at several MB.
    _write_png(tmp_path / "photo.png", (2000, 1500), noise="smooth")
    result = await _call(tools, "read", {"path": "photo.png"}, context)

    assert result.is_error is False
    (image,) = _image_blocks(result)
    assert image.mime_type == "image/jpeg"
    # The rung is only allowed to fire when it wins; taking a lossy encode
    # that is also bigger would be strictly worse on both axes.
    lossless = io.BytesIO()
    with Image.open(tmp_path / "photo.png") as original:
        original.resize((1568, 1176), Image.Resampling.LANCZOS).save(lossless, format="PNG")
    assert len(base64.b64decode(image.data)) < len(lossless.getvalue())
    # base64 inflates by 4/3 and Anthropic rejects an image block over 5 MB.
    assert len(image.data) < 5_000_000


@pytest.mark.asyncio
async def test_read_keeps_png_when_jpeg_would_be_bigger(tools, context, tmp_path) -> None:
    # The mirror of the case above, and not hypothetical: sharp noise over an
    # 8-colour palette measures 1145 KiB as PNG (over the budget, so the lossy
    # rung fires) against 1704 KiB as quality-85 JPEG. Taking JPEG on the way
    # past the budget would then be worse on BOTH axes — bigger and lossy — so
    # the rung has to measure rather than assume it wins.
    _write_png(tmp_path / "sharp.png", (1568, 1176), noise="sharp", colours=8)
    result = await _call(tools, "read", {"path": "sharp.png"}, context)

    assert result.is_error is False
    (image,) = _image_blocks(result)
    assert image.mime_type == "image/png"
    assert len(base64.b64decode(image.data)) > builtin.READ_IMAGE_MAX_BYTES


@pytest.mark.asyncio
async def test_read_image_without_pillow_is_forwarded_verbatim(
    tools, context, tmp_path, monkeypatch
) -> None:
    # Pillow reaches a default install only as a pillow-heif dependency, and
    # that is the most platform-fragile wheel here. With no decoder there is
    # no resize and no validation, but a screenshot the model can look at
    # still beats a paragraph explaining why it cannot.
    source = _write_png(tmp_path / "shot.png", (320, 200))
    monkeypatch.setattr(builtin, "pillow_image_module", lambda: None)
    result = await _call(tools, "read", {"path": "shot.png"}, context)

    assert result.is_error is False
    (image,) = _image_blocks(result)
    assert base64.b64decode(image.data) == source.read_bytes()
    assert "without resizing" in result.text


@pytest.mark.asyncio
async def test_read_large_image_without_pillow_is_refused(
    tools, context, tmp_path, monkeypatch
) -> None:
    # The byte cap is the only bound still enforceable with no decoder, so it
    # becomes the line. Forwarding an unbounded unvalidated blob is how a
    # session ends up wedged behind a provider that refuses it.
    _write_png(tmp_path / "fat.png", (2400, 1800), noise="smooth")
    monkeypatch.setattr(builtin, "pillow_image_module", lambda: None)
    result = await _call(tools, "read", {"path": "fat.png"}, context)

    assert result.is_error is True
    assert _image_blocks(result) == []
    assert str(builtin.READ_IMAGE_MAX_BYTES) in result.text


@pytest.mark.asyncio
async def test_read_heic_without_pillow_heif_refuses_rather_than_forwarding(
    tools, context, tmp_path, monkeypatch
) -> None:
    # No provider accepts HEIC, so forwarding it verbatim would GUARANTEE the
    # refusal rather than risk it. Transcoding is the only way to send one,
    # and transcoding is exactly what is unavailable here.
    (tmp_path / "pic.heic").write_bytes(b"\x00\x00\x00\x1cftypheic" + b"\x00" * 64)
    monkeypatch.setattr(builtin, "heif_image_module", lambda: None)
    result = await _call(tools, "read", {"path": "pic.heic"}, context)

    assert result.is_error is True
    assert _image_blocks(result) == []
    assert "images" in result.text


@pytest.mark.asyncio
async def test_read_refuses_a_decompression_bomb_from_the_header(tools, context, tmp_path) -> None:
    # A bomb is small on disk by construction, so the byte cap cannot see it
    # coming and only the dimensions can. media.sniff_image reads those from
    # the IHDR, which is what lets the refusal land BEFORE a decode allocates
    # 3.6 GB of RGBA — hence a forged header rather than a real 30000px file.
    small = _write_png(tmp_path / "seed.png", (8, 8)).read_bytes()
    forged = bytearray(small)
    struct.pack_into(">II", forged, 16, 30000, 30000)
    struct.pack_into(">I", forged, 29, zlib.crc32(bytes(forged[12:29])) & 0xFFFFFFFF)
    (tmp_path / "bomb.png").write_bytes(bytes(forged))

    result = await _call(tools, "read", {"path": "bomb.png"}, context)
    assert result.is_error is True
    assert _image_blocks(result) == []
    assert "30000x30000" in result.text


@pytest.mark.asyncio
async def test_read_non_image_binary_is_unchanged(tools, context, tmp_path) -> None:
    # The image branch must not swallow the binary refusal it sits in front of.
    (tmp_path / "blob.bin").write_bytes(b"\x00\x01\x02payload")
    result = await _call(tools, "read", {"path": "blob.bin"}, context)
    assert result.is_error is True
    assert "Binary file not readable as text" in result.text
    assert _image_blocks(result) == []


@pytest.mark.asyncio
async def test_read_corrupt_image_errors_without_an_image_block(tools, context, tmp_path) -> None:
    # Load-bearing, not defensive: Anthropic answers an undecodable image with
    # `Could not process image`, and the bad block is already in the
    # transcript by then, so every later request in the session dies on it
    # too. The decode here is the only place that failure is still recoverable.
    intact = _write_png(tmp_path / "shot.png", (320, 200)).read_bytes()
    (tmp_path / "truncated.png").write_bytes(intact[: len(intact) // 2])
    result = await _call(tools, "read", {"path": "truncated.png"}, context)

    assert result.is_error is True
    assert _image_blocks(result) == []
    assert "truncated.png" in result.text and "as an image" in result.text


@pytest.mark.asyncio
async def test_read_classifies_by_magic_bytes_not_extension(tools, context, tmp_path) -> None:
    # A `.png` holding an HTML error page is the realistic version of this: it
    # is readable text and must never be shipped as an image. An extensionless
    # screenshot is the mirror case — still a screenshot.
    (tmp_path / "fake.png").write_text("<html><body>404 not found</body></html>\n")
    fake = await _call(tools, "read", {"path": "fake.png"}, context)
    assert fake.is_error is False
    assert _image_blocks(fake) == []
    assert "404 not found" in fake.text

    _write_png(tmp_path / "screenshot", (64, 48))
    bare = await _call(tools, "read", {"path": "screenshot"}, context)
    assert bare.is_error is False
    assert len(_image_blocks(bare)) == 1


@pytest.mark.asyncio
async def test_read_unsupported_image_format_names_it(tools, context, tmp_path) -> None:
    # No provider takes BMP, so the extension is the only evidence left once
    # the sniff declines. "Binary file not readable as text" reads as a bug in
    # read to a caller who can plainly see a .bmp.
    Image.new("RGB", (32, 32), (0, 0, 0)).save(tmp_path / "pic.bmp")
    result = await _call(tools, "read", {"path": "pic.bmp"}, context)
    assert result.is_error is True
    assert _image_blocks(result) == []
    assert "image/bmp" in result.text


@pytest.mark.asyncio
async def test_read_image_above_the_text_cap_is_still_read(tools, context, tmp_path) -> None:
    # The 2 MB text cap exists because bytes become context; an image's cost
    # is its pixels and is already bounded by the resize. Refusing a 2 MB
    # screenshot with "use bash (head/tail)" helped nobody.
    _write_png(tmp_path / "fat.png", (2400, 1800), noise="smooth")
    assert (tmp_path / "fat.png").stat().st_size > builtin.READ_FILE_LIMIT_BYTES
    result = await _call(tools, "read", {"path": "fat.png"}, context)
    assert result.is_error is False
    assert len(_image_blocks(result)) == 1


@pytest.mark.asyncio
async def test_read_image_reports_that_range_was_ignored(tools, context, tmp_path) -> None:
    # Dropping the argument silently would leave the model believing it read a
    # slice of something.
    _write_png(tmp_path / "shot.png", (64, 48))
    result = await _call(tools, "read", {"path": "shot.png", "range": "1-10"}, context)
    assert result.is_error is False
    assert len(_image_blocks(result)) == 1
    assert "'range' does not apply" in result.text


@pytest.mark.asyncio
async def test_read_directory_listing(tools, context, tmp_path) -> None:
    (tmp_path / "sub").mkdir()
    (tmp_path / "file.txt").write_text("x")
    result = await _call(tools, "read", {"path": "."}, context)
    assert result.is_error is False
    assert "Directory listing" in result.text
    assert "sub/" in result.text and "file.txt" in result.text


@pytest.mark.asyncio
async def test_read_skill_url_via_resolver(tmp_path) -> None:
    def resolver(url: str) -> str | None:
        if url == "skill://demo":
            return "SKILL MARKDOWN BODY"
        return None

    context = ToolContext(cwd=str(tmp_path), session_id="s", resolve_internal_url=resolver)
    tools = {t.name: t for t in create_tools(context)}

    hit = await tools["read"].execute("c", {"path": "skill://demo"}, None, None, context)
    assert hit.is_error is False
    assert "SKILL MARKDOWN BODY" in hit.text

    miss = await tools["read"].execute("c", {"path": "skill://nope"}, None, None, context)
    assert miss.is_error is True


@pytest.mark.asyncio
async def test_read_skill_url_without_resolver(tmp_path) -> None:
    context = ToolContext(cwd=str(tmp_path), session_id="s")  # no resolver installed
    tools = {t.name: t for t in create_tools(context)}
    result = await tools["read"].execute("c", {"path": "skill://x"}, None, None, context)
    assert result.is_error is True


# ---------------------------------------------------------------------------
# path safety and approval tiers (RT-09/RT-10/RT-14/RT-29)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_write_inside_workspace_never_prompts(tmp_path) -> None:
    # Write-tier escalation lives in the loop; inside the workspace the tool
    # must run clean with zero approval callbacks.
    context = _context_with_approval(tmp_path, approve=True)
    tools = {t.name: t for t in create_tools(context)}
    result = await tools["write"].execute(
        "c", {"path": "ok.txt", "content": "x"}, None, None, context
    )
    assert result.is_error is False
    assert context.recorder.requests == []


@pytest.mark.asyncio
async def test_read_outside_workspace_still_escalates(tmp_path) -> None:
    # Read-tier OUTSIDE-workspace escalation remains a tool-level gate (the
    # loop only gates write/exec tiers).
    workspace = tmp_path / "ws"
    outside = tmp_path / "outside"
    workspace.mkdir()
    outside.mkdir()
    (outside / "secret.txt").write_text("x")
    context = _context_with_approval(workspace, approve=True)
    tools = {t.name: t for t in create_tools(context)}
    result = await tools["read"].execute(
        "c", {"path": "../outside/secret.txt"}, None, None, context
    )
    assert result.is_error is False
    tier, description = context.recorder.requests[0]
    assert tier == "read"
    assert description.startswith("[outside workspace] ")
    assert str((outside / "secret.txt").resolve()) in description

    deny = _context_with_approval(workspace, approve=False)
    tools = {t.name: t for t in create_tools(deny)}
    result = await tools["read"].execute("c", {"path": "../outside/secret.txt"}, None, None, deny)
    assert result.is_error is True


@pytest.mark.asyncio
async def test_edit_inside_workspace_never_prompts(tmp_path) -> None:
    (tmp_path / "keep.txt").write_text("alpha\n")
    context = _context_with_approval(tmp_path, approve=True)
    tools = {t.name: t for t in create_tools(context)}
    result = await tools["edit"].execute(
        "c",
        {"path": "keep.txt", "old_text": "alpha", "new_text": "beta"},
        None,
        None,
        context,
    )
    assert result.is_error is False
    assert (tmp_path / "keep.txt").read_text() == "beta\n"
    assert context.recorder.requests == []


@pytest.mark.asyncio
async def test_read_glob_grep_never_prompt_inside_workspace(tmp_path) -> None:
    # RT-29: read-tier tools stay silent inside the workspace.
    (tmp_path / "a.txt").write_text("needle\n")
    context = _context_with_approval(tmp_path, approve=True)
    tools = {t.name: t for t in create_tools(context)}

    await tools["read"].execute("c", {"path": "a.txt"}, None, None, context)
    await tools["glob"].execute("c", {"pattern": "*.txt"}, None, None, context)
    await tools["grep"].execute("c", {"pattern": "needle"}, None, None, context)
    await tools["todo"].execute("c", {"op": "view"}, None, None, context)
    assert context.recorder.requests == []


@pytest.mark.asyncio
async def test_read_outside_workspace_requires_approval(tmp_path) -> None:
    # RT-09: read-tier escalates to a prompt outside the workspace.
    workspace = tmp_path / "ws"
    workspace.mkdir()
    secret = tmp_path / "secret.txt"
    secret.write_text("hush\n")

    approved = _context_with_approval(workspace, approve=True)
    tools = {t.name: t for t in create_tools(approved)}
    ok = await tools["read"].execute("c", {"path": str(secret)}, None, None, approved)
    assert ok.is_error is False
    tier, description = approved.recorder.requests[0]
    assert tier == "read"
    assert description.startswith("[outside workspace] ")

    denied = _context_with_approval(workspace, approve=False)
    tools = {t.name: t for t in create_tools(denied)}
    blocked = await tools["read"].execute("c", {"path": str(secret)}, None, None, denied)
    assert blocked.is_error is True


# ---------------------------------------------------------------------------
# glob / grep
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_glob_matches_and_sorts(tools, context, tmp_path) -> None:
    (tmp_path / "b.txt").write_text("x")
    (tmp_path / "a.txt").write_text("x")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "c.txt").write_text("x")

    result = await _call(tools, "glob", {"pattern": "**/*.txt"}, context)
    assert result.is_error is False
    assert result.useless is False
    assert "a.txt" in result.text and "b.txt" in result.text and "sub/c.txt" in result.text


@pytest.mark.asyncio
async def test_glob_sorts_before_slicing(tools, context, tmp_path) -> None:
    # RT-13: collect all, sort, then slice — the cap keeps the FIRST 500 in
    # sorted order, so 'a...' names always win.
    for i in range(20):
        (tmp_path / f"z{i:02d}.txt").write_text("x")
    (tmp_path / "aaa.txt").write_text("x")
    result = await _call(tools, "glob", {"pattern": "*.txt"}, context)
    body = result.text.split(":\n", 1)[1].splitlines()
    assert body[0] == "aaa.txt"


@pytest.mark.asyncio
async def test_glob_rejects_absolute_and_parent_patterns(tools, context) -> None:
    # RT-14: clean is_error results, never a ValueError escape.
    for pattern in ("/etc/passwd", "../secrets/*", ".."):
        result = await _call(tools, "glob", {"pattern": pattern}, context)
        assert result.is_error is True
        assert "relative" in result.text.lower()


@pytest.mark.asyncio
async def test_glob_no_matches_is_useless(tools, context) -> None:
    result = await _call(tools, "glob", {"pattern": "*.nomatch"}, context)
    assert result.useless is True
    assert result.is_error is False


@pytest.mark.asyncio
async def test_grep_finds_matches(tools, context, tmp_path) -> None:
    (tmp_path / "one.py").write_text("alpha = 1\nbeta = 2\n")
    (tmp_path / "two.py").write_text("gamma = 3\n")

    result = await _call(tools, "grep", {"pattern": "beta"}, context)
    assert result.is_error is False
    assert result.useless is False
    assert "one.py:2:beta = 2" in result.text


@pytest.mark.asyncio
async def test_grep_include_filter(tools, context, tmp_path) -> None:
    (tmp_path / "code.py").write_text("needle\n")
    (tmp_path / "notes.md").write_text("needle\n")
    result = await _call(tools, "grep", {"pattern": "needle", "include": "*.py"}, context)
    assert "code.py:1:needle" in result.text
    assert "notes.md" not in result.text


@pytest.mark.asyncio
async def test_grep_prunes_dot_and_vendor_dirs(tools, context, tmp_path) -> None:
    # RT-07: .git (and friends) are pruned; their contents never match.
    git = tmp_path / ".git"
    git.mkdir()
    (git / "config").write_text("needle\n")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "lib.js").write_text("needle\n")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("needle\n")

    result = await _call(tools, "grep", {"pattern": "needle"}, context)
    assert "src/app.py:1:needle" in result.text
    assert ".git" not in result.text
    assert "node_modules" not in result.text


@pytest.mark.asyncio
async def test_grep_skips_oversized_files_with_footer(tools, context, tmp_path) -> None:
    # RT-07: per-file 1MB cap, with the skipped count in the footer.
    (tmp_path / "small.py").write_text("needle\n")
    (tmp_path / "big.py").write_text("needle\n" * 200000)  # > 1MB
    result = await _call(tools, "grep", {"pattern": "needle"}, context)
    assert "small.py:1:needle" in result.text
    assert "big.py" not in result.text.split(":\n", 1)[1]
    assert "1 file(s) skipped" in result.text


@pytest.mark.asyncio
async def test_grep_invalid_regex_is_error(tools, context) -> None:
    result = await _call(tools, "grep", {"pattern": "(unclosed"}, context)
    assert result.is_error is True


# ---------------------------------------------------------------------------
# todo
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_todo_lifecycle(tools, context) -> None:
    init = await _call(tools, "todo", {"op": "init", "items": ["one", "two"]}, context)
    assert init.is_error is False

    done = await _call(tools, "todo", {"op": "done", "items": ["one"]}, context)
    assert done.is_error is False

    view = await _call(tools, "todo", {"op": "view"}, context)
    assert "one" in view.text and "two" in view.text
    assert "[x]" in view.text


@pytest.mark.asyncio
async def test_todo_done_unknown_is_error(tools, context) -> None:
    await _call(tools, "todo", {"op": "init", "items": ["a"]}, context)
    result = await _call(tools, "todo", {"op": "done", "items": ["ghost"]}, context)
    assert result.is_error is True


@pytest.mark.asyncio
async def test_todo_without_session_id_stores_on_context(tmp_path) -> None:
    # RT-18: no session id -> the list rides on the context object itself,
    # never under a shared "" key in the module table.
    bare_a = ToolContext(cwd=str(tmp_path))
    bare_b = ToolContext(cwd=str(tmp_path))
    tools_a = {t.name: t for t in create_tools(bare_a)}
    tools_b = {t.name: t for t in create_tools(bare_b)}

    await tools_a["todo"].execute("c", {"op": "init", "items": ["mine"]}, None, None, bare_a)
    view_a = await tools_a["todo"].execute("c", {"op": "view"}, None, None, bare_a)
    view_b = await tools_b["todo"].execute("c", {"op": "view"}, None, None, bare_b)
    assert "mine" in view_a.text
    assert view_b.useless is True  # a different bare context sees nothing
    assert "" not in builtin.TODO_STORE


@pytest.mark.asyncio
async def test_todo_view_empty_is_useless(tools, context) -> None:
    # fresh context/session so the in-memory store is empty
    fresh = ToolContext(cwd=".", session_id="fresh-empty")
    t = {x.name: x for x in create_tools(fresh)}
    result = await t["todo"].execute("c", {"op": "view"}, None, None, fresh)
    assert result.useless is True


# ---------------------------------------------------------------------------
# wake
# ---------------------------------------------------------------------------


class _FakeScheduler:
    """Minimal stand-in exposing the surface the wake tool reads."""

    def __init__(self) -> None:
        self._schedules: list[Any] = []

    @property
    def schedules(self) -> list[Any]:
        return self._schedules

    async def update(self, schedules) -> None:
        self._schedules = list(schedules)


def test_wake_builder_returns_none_without_scheduler(tmp_path) -> None:
    # RT-17: createIf — no scheduler on the context, no wake tool at all.
    assert builtin.build_wake_tool(ToolContext(cwd=str(tmp_path))) is None
    assert "wake" not in {t.name for t in create_tools(ToolContext(cwd=str(tmp_path)))}

    with_scheduler = ToolContext(cwd=str(tmp_path), session_id="s", wake_scheduler=_FakeScheduler())
    tool = builtin.build_wake_tool(with_scheduler)
    assert tool is not None and tool.name == "wake"


@pytest.mark.asyncio
async def test_wake_create_list_cancel(tmp_path) -> None:
    scheduler = _FakeScheduler()
    context = ToolContext(cwd=str(tmp_path), session_id="s", wake_scheduler=scheduler)
    tools = {t.name: t for t in create_tools(context)}

    created = await tools["wake"].execute(
        "c", {"op": "create", "message": "standup", "in": "30m"}, None, None, context
    )
    assert created.is_error is False
    assert len(scheduler.schedules) == 1
    schedule_id = scheduler.schedules[0].id

    listed = await tools["wake"].execute("c", {"op": "list"}, None, None, context)
    assert schedule_id in listed.text

    cancelled = await tools["wake"].execute(
        "c", {"op": "cancel", "id": schedule_id}, None, None, context
    )
    assert cancelled.is_error is False
    assert scheduler.schedules == []


@pytest.mark.asyncio
async def test_wake_list_shows_duration_grammar(tmp_path) -> None:
    # RT-26: repeat intervals render in duration grammar (1h), not seconds.
    scheduler = _FakeScheduler()
    context = ToolContext(cwd=str(tmp_path), session_id="s", wake_scheduler=scheduler)
    tools = {t.name: t for t in create_tools(context)}
    await tools["wake"].execute(
        "c",
        {"op": "create", "message": "hourly", "in": "10m", "every": "1h"},
        None,
        None,
        context,
    )
    listed = await tools["wake"].execute("c", {"op": "list"}, None, None, context)
    assert "every 1h" in listed.text
    assert "3600s" not in listed.text


@pytest.mark.asyncio
async def test_wake_create_requires_timing(tmp_path) -> None:
    scheduler = _FakeScheduler()
    context = ToolContext(cwd=str(tmp_path), session_id="s", wake_scheduler=scheduler)
    tools = {t.name: t for t in create_tools(context)}
    result = await tools["wake"].execute(
        "c", {"op": "create", "message": "hi"}, None, None, context
    )
    assert result.is_error is True


# ---------------------------------------------------------------------------
# argument validation and error safety (RT-29/RT-31)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pydantic_validation_errors_are_clean(tools, context) -> None:
    # RT-29: every tool returns 'invalid arguments:' lines, never a traceback.
    cases = {
        "bash": {"timeout": "soon"},
        "read": {"range": 5},
        "write": {"path": "x", "content": "y", "extra": 1},
        "edit": {"path": "x", "old_text": "a"},
        "glob": {"pattern": 7},
        "grep": {"case": "yes"},
        "todo": {"op": "bogus"},
    }
    for name, args in cases.items():
        result = await _call(tools, name, args, context)
        assert result.is_error is True, name
        assert result.text.startswith("invalid arguments:"), name
        assert "Traceback" not in result.text, name


@pytest.mark.asyncio
async def test_unexpected_exception_becomes_error_result(tools, context, monkeypatch) -> None:
    # RT-31: force a genuine internal RuntimeError; the guard converts it.
    monkeypatch.setattr(Path, "exists", lambda self: (_ for _ in ()).throw(RuntimeError("boom")))
    result = await _call(tools, "read", {"path": "ghost.txt"}, context)
    assert result.is_error is True
    assert "failed unexpectedly" in result.text


# ---------------------------------------------------------------------------
# ToolResult invariant sweep (RT-28)
# ---------------------------------------------------------------------------

#: (tool name, args, needs_scheduler) — one representative call per tool,
#: chosen to exercise success AND the useless/error shapes.
_SWEEP_CASES: list[tuple[str, dict[str, Any]]] = [
    ("bash", {"command": "echo sweep"}),
    ("read", {"path": "sweep.txt"}),
    ("read", {"path": "ghost-sweep.txt"}),
    ("read", {"path": "sweep.txt", "range": "900-999"}),
    ("write", {"path": "sweep.txt", "content": "a\nb\n"}),
    ("edit", {"path": "sweep.txt", "old_text": "a", "new_text": "c"}),
    ("edit", {"path": "sweep.txt", "old_text": "zzz", "new_text": "c"}),
    ("glob", {"pattern": "*.txt"}),
    ("glob", {"pattern": "*.nomatch-sweep"}),
    ("grep", {"pattern": "sweep-me"}),
    ("grep", {"pattern": "zzz_no_such_sweep"}),
    ("todo", {"op": "init", "items": ["sweep"]}),
    ("todo", {"op": "view"}),
    ("wake", {"op": "list"}),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_name,args", _SWEEP_CASES, ids=lambda v: str(v)[:60])
async def test_tool_result_invariants(tmp_path, tool_name, args) -> None:
    # RT-28: useless XOR is_error on every result a tool can produce, and
    # useless always carries details['useless'].
    scheduler = _FakeScheduler()
    context = ToolContext(cwd=str(tmp_path), session_id="sweep", wake_scheduler=scheduler)
    tools = {t.name: t for t in create_tools(context)}
    (tmp_path / "sweep.txt").write_text("a\nb\nsweep-me\n")

    result = await tools[tool_name].execute("c", args, None, None, context)

    assert isinstance(result, ToolResult)
    assert result.tool_call_id == "c"
    assert result.tool_name == tool_name
    assert result.text  # never an empty block (providers reject those)
    assert not (
        result.useless and result.is_error
    ), f"{tool_name}: useless and is_error are mutually exclusive"
    if result.useless:
        assert isinstance(result.details, dict)
        assert result.details.get("useless") is True


# ---------------------------------------------------------------------------
# edit: multi-hunk, whitespace tolerance, anchor_line
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_edit_multi_hunk_applies_all_in_one_call(tools, context, tmp_path) -> None:
    await _call(
        tools,
        "write",
        {"path": "m.py", "content": "alpha\nmiddle\nbeta\nmiddle\ngamma\n"},
        context,
    )
    result = await _call(
        tools,
        "edit",
        {
            "path": "m.py",
            "edits": [
                {"old_text": "alpha", "new_text": "ALPHA"},
                {"old_text": "beta", "new_text": "BETA"},
                {"old_text": "gamma", "new_text": "GAMMA"},
            ],
        },
        context,
    )
    assert result.is_error is False
    assert (tmp_path / "m.py").read_text() == "ALPHA\nmiddle\nBETA\nmiddle\nGAMMA\n"
    assert "3 hunk(s)" in result.text


@pytest.mark.asyncio
async def test_read_classification_and_bytes_share_the_write_transaction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A writer cannot swap the file between read sniffing and its byte snapshot."""
    path = tmp_path / "snapshot.txt"
    path.write_text("before")
    sniff_entered = threading.Event()
    release_sniff = threading.Event()
    real_sniff = builtin.sniff_image_file

    def blocked_sniff(target: str):
        # Capture classification, then give the writer a deterministic window.
        # Without one shared transaction it commits `after` during the sleep,
        # so the read's earlier checks describe different returned bytes.
        info = real_sniff(target)
        sniff_entered.set()
        assert release_sniff.wait(timeout=2)
        time.sleep(0.05)
        return info

    def writer() -> None:
        assert sniff_entered.wait(timeout=2)
        release_sniff.set()
        builtin._write_file_result(path, "after")

    monkeypatch.setattr(builtin, "sniff_image_file", blocked_sniff)
    writer_thread = threading.Thread(target=writer)
    writer_thread.start()
    context = ToolContext(cwd=str(tmp_path))
    result = await builtin.execute_read(
        "read-snapshot",
        {"path": "snapshot.txt", "raw": True},
        None,
        None,
        context,
    )
    await asyncio.to_thread(writer_thread.join, 2)

    assert not writer_thread.is_alive()
    assert not result.is_error
    assert "before" in result.text
    assert "after" not in result.text
    assert path.read_text() == "after"


@pytest.mark.asyncio
async def test_concurrent_edits_share_one_file_transaction(
    tools,
    context,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Separate AgentLoops cannot both read the same original and lose one edit."""
    path = tmp_path / "shared.txt"
    path.write_text("alpha\nbeta\n")

    real_match = builtin._match_windows

    def delayed_match(content: str, old_text: str):
        # Both unlocked transactions read the original before this sleep and
        # then overwrite one another. The process-wide path stripe makes the
        # second transaction enter only after the first has committed.
        time.sleep(0.05)
        return real_match(content, old_text)

    monkeypatch.setattr(builtin, "_match_windows", delayed_match)
    first, second = await asyncio.gather(
        _call(
            tools,
            "edit",
            {"path": "shared.txt", "old_text": "alpha", "new_text": "ALPHA"},
            context,
        ),
        _call(
            tools,
            "edit",
            {"path": "shared.txt", "old_text": "beta", "new_text": "BETA"},
            context,
        ),
    )

    assert not first.is_error and not second.is_error
    assert path.read_text() == "ALPHA\nBETA\n"


@pytest.mark.asyncio
async def test_hardlink_aliases_share_one_file_transaction(
    tools,
    context,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Distinct path spellings of one inode cannot lose a concurrent edit."""
    path = tmp_path / "shared.txt"
    alias = tmp_path / "alias.txt"
    path.write_text("alpha\nbeta\n")
    os.link(path, alias)

    real_match = builtin._match_windows

    def delayed_match(content: str, old_text: str):
        time.sleep(0.05)
        return real_match(content, old_text)

    monkeypatch.setattr(builtin, "_match_windows", delayed_match)
    first, second = await asyncio.gather(
        _call(
            tools,
            "edit",
            {"path": "shared.txt", "old_text": "alpha", "new_text": "ALPHA"},
            context,
        ),
        _call(
            tools,
            "edit",
            {"path": "alias.txt", "old_text": "beta", "new_text": "BETA"},
            context,
        ),
    )

    assert not first.is_error and not second.is_error
    assert path.read_text() == "ALPHA\nBETA\n"
    assert alias.read_text() == "ALPHA\nBETA\n"


@pytest.mark.asyncio
async def test_edit_whitespace_tolerant_match_reindents(tools, context, tmp_path) -> None:
    """old_text written at the wrong indentation still matches, and the
    replacement is re-indented to the FILE's level — the edit written from a
    structural summary or memory works instead of erroring."""
    await _call(
        tools,
        "write",
        {"path": "t.py", "content": "class A:\n    def foo(self):\n        return 1\n"},
        context,
    )
    # The model wrote the body at 2 spaces while the file uses 8.
    result = await _call(
        tools,
        "edit",
        {
            "path": "t.py",
            "edits": [
                {"old_text": "def foo(self):\n  return 1", "new_text": "def foo(self):\n  return 2"}
            ],
        },
        context,
    )
    assert result.is_error is False
    assert (tmp_path / "t.py").read_text() == ("class A:\n    def foo(self):\n        return 2\n")


@pytest.mark.asyncio
async def test_edit_anchor_line_disambiguates(tools, context, tmp_path) -> None:
    await _call(tools, "write", {"path": "d.txt", "content": "foo\nfoo\nfoo\n"}, context)
    result = await _call(
        tools,
        "edit",
        {"path": "d.txt", "old_text": "foo", "new_text": "WON", "anchor_line": 3},
        context,
    )
    assert result.is_error is False
    assert (tmp_path / "d.txt").read_text() == "foo\nfoo\nWON\n"


@pytest.mark.asyncio
async def test_edit_rejects_both_forms_at_once(tools, context, tmp_path) -> None:
    await _call(tools, "write", {"path": "x.txt", "content": "abc\n"}, context)
    result = await _call(
        tools,
        "edit",
        {
            "path": "x.txt",
            "old_text": "abc",
            "new_text": "zzz",
            "edits": [{"old_text": "abc", "new_text": "zzz"}],
        },
        context,
    )
    assert result.is_error is True
    assert (tmp_path / "x.txt").read_text() == "abc\n"


@pytest.mark.asyncio
async def test_edit_tolerant_ambiguity_still_errors(tools, context, tmp_path) -> None:
    """Tolerance widens matching, so its ambiguity discipline matters more:
    two strip-equal candidates with no anchor and no replace_all refuse."""
    await _call(tools, "write", {"path": "amb.txt", "content": "  foo\nbar\n  foo\n"}, context)
    result = await _call(
        tools,
        "edit",
        {"path": "amb.txt", "old_text": "foo", "new_text": "X"},
        context,
    )
    # Exact match fails (file lines are indented); tolerant matches twice.
    assert result.is_error is True
    assert "2 places" in result.text


# ---------------------------------------------------------------------------
# read: Python structural summaries
# ---------------------------------------------------------------------------


def _summary_py() -> str:
    body = "\n".join(f"    # filler {i}" for i in range(120))
    return (
        '"""Module doc."""\n'
        "import os\n"
        "from pathlib import Path\n"
        "\n"
        "def helper(one, two=2) -> int:\n"
        '    """Helper doc."""\n'
        f"{body}\n"
        "    return one + two\n"
        "\n"
        "class Widget(Base):\n"
        '    """Widget doc."""\n'
        "\n"
        "    def render(self) -> str:\n"
        "        return 'w'\n"
        "\n"
        "    async def load(self):\n"
        "        pass\n"
    )


@pytest.mark.asyncio
async def test_read_python_structural_summary_default(tools, context, tmp_path) -> None:
    (tmp_path / "big.py").write_text(_summary_py())
    result = await _call(tools, "read", {"path": "big.py"}, context)
    assert result.is_error is False
    assert "structural summary" in result.text
    assert "def helper(one, two=2) -> int" in result.text
    assert "class Widget(Base):" in result.text
    assert "async def load(self)" in result.text
    assert '"Widget doc.' in result.text
    assert "[imports: 2 (elided)]" in result.text
    # Line ranges ride every symbol so the footer's range advice is actionable.
    assert "L5-" in result.text
    # Bodies are elided — the filler is the proof it is not the raw body.
    assert "filler 50" not in result.text
    assert "bodies elided" in result.text


@pytest.mark.asyncio
async def test_read_raw_and_range_bypass_summary(tools, context, tmp_path) -> None:
    (tmp_path / "big.py").write_text(_summary_py())
    raw = await _call(tools, "read", {"path": "big.py", "raw": True}, context)
    assert "filler 50" in raw.text
    ranged = await _call(tools, "read", {"path": "big.py", "range": "1-3"}, context)
    assert "Module doc" in ranged.text
    assert "def helper" not in ranged.text


@pytest.mark.asyncio
async def test_read_summary_falls_back_on_syntax_error(tools, context, tmp_path) -> None:
    broken = "def broken(:\n" + "\n".join(f"# pad {i}" for i in range(100)) + "\n"
    (tmp_path / "broken.py").write_text(broken)
    result = await _call(tools, "read", {"path": "broken.py"}, context)
    assert "structural summary" not in result.text
    assert "def broken(:" in result.text


@pytest.mark.asyncio
async def test_read_short_python_file_stays_raw(tools, context, tmp_path) -> None:
    (tmp_path / "small.py").write_text("def a():\n    return 1\n")
    result = await _call(tools, "read", {"path": "small.py"}, context)
    assert "structural summary" not in result.text
    assert "def a():" in result.text


# ---------------------------------------------------------------------------
# grep/glob: context lines, skip, gitignore
# ---------------------------------------------------------------------------


@pytest.fixture
def python_engine(monkeypatch):
    """Pin the pure-Python scan so filesystem assertions are deterministic
    regardless of whether the host running the tests has ripgrep."""
    monkeypatch.setenv("LOCAL_OPERATOR_GREP_ENGINE", "python")


@pytest.mark.usefixtures("python_engine")
@pytest.mark.asyncio
async def test_grep_context_lines_render_groups(tools, context, tmp_path) -> None:
    # Matches at lines 2 and 7 with -C1 leave a real gap (lines 4-5 unsent),
    # which is what a `--` group separator is FOR; adjacent context blocks
    # render contiguously by design, like sed output.
    (tmp_path / "c.txt").write_text("one\ntwo MATCH\nthree\nfour\nfive\nsix\nseven MATCH\neight\n")
    result = await _call(tools, "grep", {"pattern": "MATCH", "context_lines": 1}, context)
    assert result.is_error is False
    assert "c.txt:2:two MATCH" in result.text
    assert "c.txt:1-one" in result.text  # context: dash separator
    assert "c.txt:3-three" in result.text
    assert "--" in result.text  # groups 1-3 and 6-8 are disjoint
    assert "c.txt:7:seven MATCH" in result.text


@pytest.mark.usefixtures("python_engine")
@pytest.mark.asyncio
async def test_grep_skip_paginates_matches(tools, context, tmp_path) -> None:
    (tmp_path / "p.txt").write_text("".join(f"hit {i}\n" for i in range(10)))
    page1 = await _call(tools, "grep", {"pattern": "hit"}, context)
    assert "p.txt:1:hit 0" in page1.text
    page2 = await _call(tools, "grep", {"pattern": "hit", "skip": 3}, context)
    assert "p.txt:1:hit 0" not in page2.text
    assert "p.txt:4:hit 3" in page2.text
    assert "skipped 3" in page2.text


@pytest.mark.usefixtures("python_engine")
@pytest.mark.asyncio
async def test_grep_respects_gitignore(tools, context, tmp_path) -> None:
    (tmp_path / ".gitignore").write_text("gen/\n*.log\n")
    (tmp_path / "src.txt").write_text("needle here\n")
    (tmp_path / "gen" / "out.txt").parent.mkdir()
    (tmp_path / "gen" / "out.txt").write_text("needle ignored\n")
    (tmp_path / "app.log").write_text("needle ignored\n")
    result = await _call(tools, "grep", {"pattern": "needle"}, context)
    assert "src.txt:1" in result.text
    assert "gen/out.txt" not in result.text
    assert "app.log" not in result.text


@pytest.mark.usefixtures("python_engine")
@pytest.mark.asyncio
async def test_grep_gitignore_negation_un_ignores(tools, context, tmp_path) -> None:
    (tmp_path / ".gitignore").write_text("gen/\n!gen/keep.txt\n")
    (tmp_path / "gen").mkdir()
    (tmp_path / "gen" / "keep.txt").write_text("needle\n")
    result = await _call(tools, "grep", {"pattern": "needle"}, context)
    # git semantics: an ignored DIRECTORY cannot be re-included by a child
    # negation — keep.txt stays out, matching git's own behaviour.
    assert "keep.txt" not in result.text


@pytest.mark.asyncio
async def test_grep_ripgrep_engine_matches_python_contract(tools, context, tmp_path) -> None:
    """With ripgrep present, the native engine must satisfy the same shape
    contract as the Python one: rel paths without './', path:line:text, and
    the 1MB-skip footer recovered from the walked list."""
    import shutil

    if shutil.which("rg") is None:
        pytest.skip("ripgrep not installed")
    (tmp_path / "small.py").write_text("needle\n")
    big = tmp_path / "big.py"
    big.write_text("needle\n" + "x" * (1024 * 1024 + 10))
    result = await _call(tools, "grep", {"pattern": "needle"}, context)
    assert "small.py:1:needle" in result.text
    assert "./small.py" not in result.text
    assert "1 file(s) skipped over the 1MB cap" in result.text


@pytest.mark.asyncio
async def test_glob_respects_gitignore_unless_pattern_names_it(tools, context, tmp_path) -> None:
    (tmp_path / ".gitignore").write_text("dist/\n")
    (tmp_path / "dist").mkdir()
    (tmp_path / "dist" / "out.js").write_text("built")
    (tmp_path / "src.js").write_text("src")
    broad = await _call(tools, "glob", {"pattern": "**/*.js"}, context)
    assert "src.js" in broad.text
    assert "dist/out.js" not in broad.text
    named = await _call(tools, "glob", {"pattern": "dist/*.js"}, context)
    assert "dist/out.js" in named.text


# ---------------------------------------------------------------------------
# bash: steering detach vs real abort
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_bash_steering_cancellation_backgrounds_the_command(tmp_path) -> None:
    """A steering cancel (task cancelled, signal NOT aborted) detaches the
    command into a tracked background job instead of killing it: the tool
    returns a result naming the job, and the job later reports the exit code
    and output of the process that was allowed to finish."""
    from local_operator.harness.jobs import AsyncJobManager

    manager = AsyncJobManager()
    context = ToolContext(cwd=str(tmp_path), session_id="bg", jobs=manager)
    tools = {t.name: t for t in create_tools(context)}

    task = asyncio.create_task(
        _call(
            tools,
            "bash",
            {"command": "sleep 0.6 && echo finished-marker"},
            context,
        )
    )
    await asyncio.sleep(0.2)  # let the command start
    task.cancel()
    result = await task  # the tool swallows the steering cancel and answers

    assert result.is_error is False
    assert "continues in the background" in result.text
    assert result.details is not None
    job_id = result.details["job_id"]
    job = manager.get(job_id)
    assert job is not None and job.type == "bash"

    async def settle():
        while job.status == "running":
            await asyncio.sleep(0.05)

    await settle()
    assert job.status == "completed"
    assert "exit code: 0" in (job.result_text or "")
    assert "finished-marker" in (job.result_text or "")


@pytest.mark.asyncio
async def test_bash_real_abort_still_kills(tmp_path) -> None:
    """A genuine abort (Ctrl+C / jobs cancel: signal.aborted) kills the
    process group; the cancellation propagates as before."""
    from local_operator.harness.jobs import AsyncJobManager
    from local_operator.harness.types import AbortSignal

    manager = AsyncJobManager()
    context = ToolContext(cwd=str(tmp_path), session_id="bg2", jobs=manager)
    tools = {t.name: t for t in create_tools(context)}
    sig = AbortSignal()

    async def run_bash() -> ToolResult:
        return await tools["bash"].execute(
            "c",
            {"command": "sleep 5 && echo should-not-run"},
            sig,
            None,
            context,
        )

    task = asyncio.create_task(run_bash())
    await asyncio.sleep(0.2)
    sig.abort("interrupted")
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    await asyncio.sleep(0.1)
    assert manager.list() == []  # nothing was backgrounded


@pytest.mark.asyncio
async def test_cancel_backgrounded_bash_kills_process_group(tmp_path) -> None:
    """Manager cancel immediately cancels the detached runner; cleanup must
    still kill/reap the start-new-session process before it can create a marker."""
    from local_operator.harness.jobs import AsyncJobManager

    manager = AsyncJobManager()
    context = ToolContext(cwd=str(tmp_path), session_id="bg-cancel", jobs=manager)
    tools = {t.name: t for t in create_tools(context)}
    task = asyncio.create_task(
        _call(
            tools,
            "bash",
            {"command": "sleep 1; touch should-not-exist"},
            context,
        )
    )
    await asyncio.sleep(0.15)
    task.cancel()
    result = await task
    assert result.details is not None
    job_id = result.details["job_id"]
    assert await manager.cancel(job_id) is True
    await asyncio.sleep(1.2)
    assert not (tmp_path / "should-not-exist").exists()
    job = manager.get(job_id)
    assert job is not None and job.status == "cancelled"


@pytest.mark.asyncio
async def test_glob_respects_nested_gitignore(tools, context, tmp_path) -> None:
    nested = tmp_path / "packages" / "a"
    nested.mkdir(parents=True)
    (nested / ".gitignore").write_text("generated/\n")
    (nested / "generated").mkdir()
    (nested / "generated" / "hidden.py").write_text("x")
    (nested / "visible.py").write_text("x")
    result = await _call(tools, "glob", {"pattern": "**/*.py"}, context)
    assert "packages/a/visible.py" in result.text
    assert "packages/a/generated/hidden.py" not in result.text


@pytest.mark.asyncio
async def test_edit_exact_match_preserves_tabs_verbatim(tools, context, tmp_path) -> None:
    makefile = tmp_path / "Makefile"
    makefile.write_text("target:\n\told\nnext:\n", newline="")
    result = await _call(
        tools,
        "edit",
        {"path": "Makefile", "old_text": "\told\n", "new_text": "\tnew\n"},
        context,
    )
    assert result.is_error is False
    assert makefile.read_bytes() == b"target:\n\tnew\nnext:\n"


@pytest.mark.asyncio
async def test_edit_exact_match_preserves_requested_trailing_newline(
    tools, context, tmp_path
) -> None:
    path = tmp_path / "exact.txt"
    path.write_text("needle-after", newline="")
    result = await _call(
        tools,
        "edit",
        {"path": "exact.txt", "old_text": "needle", "new_text": "replacement\n"},
        context,
    )
    assert result.is_error is False
    assert path.read_bytes() == b"replacement\n-after"


@pytest.mark.asyncio
async def test_edit_tolerant_match_keeps_crlf_line_endings(tools, context, tmp_path) -> None:
    path = tmp_path / "crlf.txt"
    path.write_bytes(b"if ok:\r\n\told()\r\nnext()\r\n")
    result = await _call(
        tools,
        "edit",
        {
            "path": "crlf.txt",
            "old_text": "    old()\n",
            "new_text": "    new()\n    added()\n",
        },
        context,
    )
    assert result.is_error is False
    assert path.read_bytes() == b"if ok:\r\n\tnew()\r\n\tadded()\r\nnext()\r\n"
