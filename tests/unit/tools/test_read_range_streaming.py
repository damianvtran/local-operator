"""Ranged ``read`` streams the window; the whole-file cap still applies.

The 2 MiB cap on ``read`` is a CONTEXT budget for a whole-file read, and it
used to be applied before the range was even parsed: ``read(path, range=...)``
on a 2.2 MB file returned "File too large to read" and agents fell back to
``bash sed -n 'A,Bp'``, losing the numbering, the clamp footer and the
``range`` key compaction supersedes on. These tests pin the replacement:

- a ranged read of an oversized file returns the exact lines;
- the streamed split agrees with ``str.splitlines`` byte for byte, odd line
  endings included, because it feeds the same small-file path;
- the whole-file read still refuses an oversized file, with advice that now
  points at a range on the SAME path;
- every other branch (image, NUL binary, range past EOF) survives;
- nothing in the read loads the file whole — asserted by instrumenting the
  open/read calls, which is the regression guard for the whole change.
"""

from __future__ import annotations

import builtins
import io
import re
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from local_operator import media
from local_operator.harness.types import (
    AgentTool,
    ImageContent,
    ToolContext,
    ToolResult,
)
from local_operator.tools import builtin
from local_operator.tools.registry import create_tools

#: Every line break ``str.splitlines`` recognises, cycled through the mixed
#: fixture so one file exercises the whole separator set.
_SEPARATORS = ["\n", "\r\n", "\r", "\v", "\f", "\x1c", "\x1d", "\x1e", "\x85", "\u2028", "\u2029"]


def _plain_lines(count: int) -> list[str]:
    return [f"{i:06d} " + "x" * 44 for i in range(1, count + 1)]


PLAIN_LINES = _plain_lines(45_000)
PLAIN_BYTES = ("\n".join(PLAIN_LINES) + "\n").encode("utf-8")


def _mixed_text(count: int) -> str:
    parts: list[str] = []
    for i in range(1, count + 1):
        body = f"{i:06d}" + "y" * 40
        if i % 7 == 0:
            # Multi-byte characters mid-line: the incremental decoder must not
            # split one across a chunk boundary.
            body += " é漢"
        parts.append(body + _SEPARATORS[i % len(_SEPARATORS)])
    return "".join(parts)


MIXED_BYTES = _mixed_text(48_000).encode("utf-8")


@pytest.fixture
def context(tmp_path: Path) -> ToolContext:
    return ToolContext(cwd=str(tmp_path), session_id="range-test")


@pytest.fixture
def tools(context: ToolContext) -> dict[str, AgentTool]:
    return {tool.name: tool for tool in create_tools(context)}


async def _call(
    tools: dict[str, AgentTool], name: str, args: dict[str, Any], context: ToolContext
) -> ToolResult:
    return await tools[name].execute("call-1", args, None, None, context)  # type: ignore[operator]


def _reference_lines(data: bytes) -> list[str]:
    """``str.splitlines`` over the whole decoded file: the semantics to match."""
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        text = data.decode("utf-8", errors="replace")
    return text.splitlines()


def _reference_rendering(data: bytes, spec: str, path: Path) -> str:
    """What the PRE-streaming implementation rendered for the same bytes.

    Rebuilt from the same primitives the whole-file path uses — the full
    ``splitlines`` slice, numbered from the window start, then clamped — so it
    is an independent expression of the selection semantics the streamed path
    had to reproduce. It deliberately does NOT call the streamed code: the
    regression it guards is a streamed read that sizes the number column from
    the lines it kept instead of from the window, or that stops keeping lines
    at a place the clamp can see.
    """
    lines = _reference_lines(data)
    start, end = builtin._parse_line_range(spec)
    selected = lines[start - 1 : end]
    assert selected, "fixture must select at least one line"
    return builtin._clamp_file_body(builtin._number_lines(selected, start), path, start, len(lines))


_NUMBERED_RE = re.compile(r"\s*(\d+)\| (.*)$")


def _parse_numbered(text: str) -> list[tuple[int, str]]:
    """Pull ``(line number, body)`` out of a numbered listing."""
    rows: list[tuple[int, str]] = []
    for line in text.splitlines():
        match = _NUMBERED_RE.fullmatch(line)
        assert match, f"unexpected listing row: {line!r}"
        rows.append((int(match.group(1)), match.group(2)))
    return rows


# ---------------------------------------------------------------------------
# the bug itself
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ranged_read_of_oversized_text_file_returns_exact_lines(
    tools, context, tmp_path
) -> None:
    path = tmp_path / "big.txt"
    path.write_bytes(PLAIN_BYTES)
    assert path.stat().st_size > builtin.READ_FILE_LIMIT_BYTES

    result = await _call(tools, "read", {"path": "big.txt", "range": "40000-40020"}, context)

    assert result.is_error is False
    # The range key is what compaction supersedes on; a streamed read that
    # dropped it would let a 1-100 read blank an unrelated 900-1000 read.
    assert result.details == {"path": str(path), "range": "40000-40020"}
    assert _parse_numbered(result.text) == [(n, PLAIN_LINES[n - 1]) for n in range(40000, 40021)]


@pytest.mark.asyncio
async def test_whole_file_read_of_oversized_text_file_still_refuses(
    tools, context, tmp_path
) -> None:
    path = tmp_path / "big.txt"
    path.write_bytes(PLAIN_BYTES)

    result = await _call(tools, "read", {"path": "big.txt"}, context)

    assert result.is_error is True
    assert "too large" in result.text.lower()
    # The old advice sent the model to a DIFFERENT, smaller file, which is not
    # what it has: the range works on the file it was holding all along.
    assert "Pass a 'range'" in result.text
    assert "bash" in result.text
    assert "smaller file" not in result.text


@pytest.mark.asyncio
async def test_open_ended_range_of_oversized_file_clips_like_the_whole_file_path(
    tools, context, tmp_path
) -> None:
    # ``range="44000-"`` on a 2.4 MB file asks for the whole tail. The kept
    # window is bounded by the clamp budget rather than by the file, so this
    # also pins that a truncated scan renders the SAME bytes as a full one.
    path = tmp_path / "big.txt"
    path.write_bytes(PLAIN_BYTES)

    result = await _call(tools, "read", {"path": "big.txt", "range": "44000-"}, context)

    assert result.is_error is False
    assert result.text == _reference_rendering(PLAIN_BYTES, "44000-", path)
    assert "not shown" in result.text and 'range="' in result.text


# ---------------------------------------------------------------------------
# semantics: the split must be str.splitlines, not "\n"
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        b"",
        b"a",
        b"a\n",
        b"\n",
        b"\n\n",
        b"a\n\n",
        b"\r",
        b"\r\n",
        b"\r\n\r\n",
        b"a\rb\rc",
        b"a\r\nb\nc\rd",
        b"a\r\n\r",
        b"x\vy\fz",
        b"\x1c\x1d\x1e",
        "\x85".encode(),
        "a\u2028b\u2029c".encode(),
        b"trailing without a break",
        b"mixed \xe2\x80\xa8 and \xff\xfe broken",
        "é漢\u2028".encode(),
        b"latin1 caf\xe9\nnext\n",
    ],
)
async def test_ranged_read_matches_splitlines_on_odd_line_endings(
    tools, context, tmp_path, payload: bytes
) -> None:
    path = tmp_path / "odd.txt"
    path.write_bytes(payload)

    result = await _call(tools, "read", {"path": "odd.txt", "range": "1-999"}, context)

    expected = _reference_lines(payload)
    assert result.is_error is False
    if not expected:
        # An empty file has no line 1: unchanged behaviour, not a crash.
        assert "is beyond end of file" in result.text
        return
    assert _parse_numbered(result.text) == [(n, line) for n, line in enumerate(expected, start=1)]


@pytest.mark.asyncio
async def test_oversized_file_with_mixed_line_endings_matches_splitlines(
    tools, context, tmp_path
) -> None:
    path = tmp_path / "mixed.txt"
    path.write_bytes(MIXED_BYTES)
    assert path.stat().st_size > builtin.READ_FILE_LIMIT_BYTES

    result = await _call(tools, "read", {"path": "mixed.txt", "range": "43000-43030"}, context)

    expected = _reference_lines(MIXED_BYTES)
    assert result.is_error is False
    assert _parse_numbered(result.text) == [(n, expected[n - 1]) for n in range(43000, 43031)]


@pytest.mark.asyncio
async def test_small_file_ranged_read_is_byte_identical_to_the_whole_file_path(
    tools, context, tmp_path
) -> None:
    # A small file must render exactly what it rendered before it was streamed,
    # including the window that spills past the output budget ("1-3000" here):
    # that is the case where a premature stop in the scan, or a number column
    # sized from the kept lines, would change the bytes.
    data = ("\n".join(_plain_lines(3000)) + "\n").encode("utf-8")
    path = tmp_path / "small.txt"
    path.write_bytes(data)
    assert len(data) < builtin.READ_FILE_LIMIT_BYTES

    for spec in ("1-3000", "100-200", "2999-", "3000-3000"):
        result = await _call(tools, "read", {"path": "small.txt", "range": spec}, context)
        assert result.is_error is False
        assert result.text == _reference_rendering(data, spec, path), spec


@pytest.mark.parametrize("chunk", [1, 3, 65536])
@pytest.mark.parametrize(
    "payload",
    [
        b"",
        b"a",
        b"a\n",
        # No trailing break: the file's last line exists and must still be
        # counted when the window stops before it.
        b"a\nb",
        b"a\r\nb",
        b"a\rb\r",
        b"x\vy\fz",
        "\x85a\u2028b".encode(),
        b"a\n\n",
        b"\r\n\r\n",
        b"\xff\xfe a\nb",
    ],
)
def test_stream_text_window_agrees_with_splitlines_for_every_window(
    monkeypatch: pytest.MonkeyPatch, chunk: int, payload: bytes
) -> None:
    """The scan must agree with ``str.splitlines`` for EVERY window it is given.

    The line outside the window is the interesting one: it must still be
    COUNTED, because the clamp footer reports how many lines remain in the
    file. That is why "is the last line open?" cannot be read off the retained
    body — that body is empty for exactly those lines, and reading it there
    lost the final line of a file with no trailing break.
    """
    monkeypatch.setattr(builtin, "_RANGED_READ_CHUNK_BYTES", chunk)
    expected = _reference_lines(payload)
    for start, end in ((1, None), (1, 1), (2, 4), (3, 3), (5, None), (len(expected) + 1, None)):
        binary, kept, window, total = builtin._stream_text_window(io.BytesIO(payload), start, end)
        windowed = expected[max(start, 1) - 1 : end]
        assert binary is False
        assert total == len(expected), (payload, chunk, start, end)
        assert window == len(windowed), (payload, chunk, start, end)
        assert kept == windowed, (payload, chunk, start, end)


@pytest.mark.asyncio
async def test_line_count_is_right_when_the_file_has_no_trailing_break(
    tools, context, tmp_path
) -> None:
    # End-to-end shape of the same bug: the clamp footer's total comes from the
    # scan, and a last line with no break behind it is still a line.
    data = "\n".join(_plain_lines(3000)).encode("utf-8")
    assert not data.endswith(b"\n")
    (tmp_path / "no_tail.txt").write_bytes(data)

    result = await _call(tools, "read", {"path": "no_tail.txt", "range": "1-2000"}, context)

    assert result.is_error is False
    assert "of 3000 lines not shown" in result.text


# ---------------------------------------------------------------------------
# every other branch survives
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ranged_read_past_eof_is_still_useless_not_an_error(tools, context, tmp_path) -> None:
    path = tmp_path / "big.txt"
    path.write_bytes(PLAIN_BYTES)

    result = await _call(tools, "read", {"path": "big.txt", "range": "999999-1000000"}, context)

    assert result.is_error is False
    assert result.useless is True
    assert "is beyond end of file" in result.text
    assert result.details == {"path": str(path), "useless": True}


@pytest.mark.asyncio
async def test_ranged_read_of_binary_file_still_errors(tools, context, tmp_path) -> None:
    (tmp_path / "blob.bin").write_bytes(b"\x00" * 100 + b"text\n")
    result = await _call(tools, "read", {"path": "blob.bin", "range": "1-10"}, context)
    assert result.is_error is True
    assert "Binary file not readable as text" in result.text


@pytest.mark.asyncio
async def test_ranged_read_of_oversized_binary_still_errors(tools, context, tmp_path) -> None:
    # Classification now runs off the first chunk, so this reports what the
    # file IS instead of the old size refusal — and without reading the rest.
    path = tmp_path / "blob.bin"
    path.write_bytes(b"\x00" * 4096 + b"z" * builtin.READ_FILE_LIMIT_BYTES)
    assert path.stat().st_size > builtin.READ_FILE_LIMIT_BYTES

    result = await _call(tools, "read", {"path": "blob.bin", "range": "1-10"}, context)

    assert result.is_error is True
    assert "Binary file not readable as text" in result.text


@pytest.mark.asyncio
async def test_binary_classification_still_sees_only_the_first_8000_bytes(
    tools, context, tmp_path
) -> None:
    # The head window is load-bearing for both paths: a NUL at offset 8000 is
    # text (a NUL deeper in a file is content the caller asked for), a NUL
    # before it is binary.
    text_with_late_nul = b"a" * 8000 + b"\x00" + b"tail\n"
    (tmp_path / "late.txt").write_bytes(text_with_late_nul)
    result = await _call(tools, "read", {"path": "late.txt", "range": "1-1"}, context)
    assert result.is_error is False
    assert _parse_numbered(result.text) == [(1, "a" * 8000 + "\x00" + "tail")]

    (tmp_path / "early.txt").write_bytes(b"a" * 7999 + b"\x00" + b"tail\n")
    result = await _call(tools, "read", {"path": "early.txt", "range": "1-1"}, context)
    assert result.is_error is True
    assert "Binary file not readable as text" in result.text


@pytest.mark.asyncio
async def test_ranged_read_of_image_still_returns_the_image_block(tools, context, tmp_path) -> None:
    Image.new("RGB", (64, 48), (10, 60, 120)).save(tmp_path / "shot.png", format="PNG")

    result = await _call(tools, "read", {"path": "shot.png", "range": "1-10"}, context)

    assert result.is_error is False
    assert [type(block) for block in result.content][-1] is ImageContent
    assert "'range' does not apply" in result.text


@pytest.mark.asyncio
async def test_ranged_read_of_oversized_image_reports_the_image_cap(
    tools, context, tmp_path
) -> None:
    # A range does not get an image past the IMAGE cap, and the advice stays
    # the image advice ("resize it"), never the text one.
    path = tmp_path / "fat.png"
    Image.new("RGB", (3000, 2200), (7, 7, 7)).save(path, format="PNG")
    path.write_bytes(path.read_bytes() + b"\x00" * (builtin.READ_IMAGE_LIMIT_BYTES + 1))
    assert path.stat().st_size > builtin.READ_IMAGE_LIMIT_BYTES

    result = await _call(tools, "read", {"path": "fat.png", "range": "1-10"}, context)

    assert result.is_error is True
    assert "Resize it first" in result.text


# ---------------------------------------------------------------------------
# the memory bound
# ---------------------------------------------------------------------------


class _ReadRecorder:
    """Wrap a binary file object and record every ``read`` it is asked for."""

    def __init__(self, label: str, handle: Any, calls: list[tuple[str, int, int]]) -> None:
        self._label = label
        self._handle = handle
        self._calls = calls

    def read(self, size: int = -1) -> bytes:
        data = self._handle.read(size)
        self._calls.append((self._label, size, len(data)))
        return data

    def __getattr__(self, name: str) -> Any:
        return getattr(self._handle, name)

    def __enter__(self) -> "_ReadRecorder":
        self._handle.__enter__()
        return self

    def __exit__(self, *exc: object) -> Any:
        return self._handle.__exit__(*exc)


def _instrument_reads(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, int, int]]:
    """Record every read ``read`` makes, and refuse whole-file loads outright.

    The refusal is the point: a ranged read that went back to
    ``Path.read_bytes()`` would fail here rather than silently pass a
    correctness assertion while holding the file in memory.
    """
    calls: list[tuple[str, int, int]] = []
    real_path_open = Path.open
    # ``media`` calls the builtin ``open`` from its own globals, so it has no
    # attribute to patch yet: the fallback is what a fresh module namespace
    # resolves to, and setting the attribute then shadows the builtin for it.
    real_media_open = getattr(media, "open", builtins.open)

    def wrap_open(label: str, module_open: Any) -> Any:
        def _open(*args: Any, **kwargs: Any) -> _ReadRecorder:
            return _ReadRecorder(label, module_open(*args, **kwargs), calls)

        return _open

    def _forbidden(name: str) -> Any:
        def _boom(*args: Any, **kwargs: Any) -> bytes:
            raise AssertionError(f"{name} loaded the whole file into memory")

        return _boom

    monkeypatch.setattr(Path, "open", wrap_open("path.open", real_path_open))
    monkeypatch.setattr(media, "open", wrap_open("media.open", real_media_open), raising=False)
    monkeypatch.setattr(Path, "read_bytes", _forbidden("Path.read_bytes"))
    monkeypatch.setattr(Path, "read_text", _forbidden("Path.read_text"))
    return calls


@pytest.mark.asyncio
async def test_ranged_read_never_loads_the_whole_file(
    tools, context, tmp_path, monkeypatch
) -> None:
    path = tmp_path / "big.txt"
    path.write_bytes(PLAIN_BYTES)
    size = path.stat().st_size
    calls = _instrument_reads(monkeypatch)

    result = await _call(tools, "read", {"path": "big.txt", "range": "1-5"}, context)

    assert result.is_error is False, result.text
    streamed = [call for call in calls if call[0] == "path.open"]
    assert streamed, "the ranged read made no instrumented read at all"
    # One bounded chunk per iteration, never a size the file dictates.
    assert all(size_arg == builtin._RANGED_READ_CHUNK_BYTES for _, size_arg, _ in streamed)
    assert max(length for _, _, length in streamed) <= builtin._RANGED_READ_CHUNK_BYTES
    # ... and it is a real forward pass over the file, not a lucky short read.
    assert sum(length for _, _, length in streamed) == size
    # The image sniff reads its header window and no more.
    assert all(length <= media._SNIFF_BYTES for label, _, length in calls if label == "media.open")


@pytest.mark.asyncio
async def test_open_ended_ranged_read_stays_bounded_on_a_large_file(
    tools, context, tmp_path, monkeypatch
) -> None:
    path = tmp_path / "big.txt"
    path.write_bytes(PLAIN_BYTES)
    size = path.stat().st_size
    calls = _instrument_reads(monkeypatch)

    result = await _call(tools, "read", {"path": "big.txt", "range": "44000-"}, context)

    assert result.is_error is False
    streamed = [call for call in calls if call[0] == "path.open"]
    assert max(length for _, _, length in streamed) <= builtin._RANGED_READ_CHUNK_BYTES
    assert sum(length for _, _, length in streamed) == size
    # The window is the whole tail of the file; what is retained is bounded by
    # the clamp budget, not by the window.
    snapshot = builtin._read_ranged_snapshot(path, 44000, None)
    assert snapshot.window_lines == 1001 and snapshot.total_lines == 45_000
    assert sum(len(line) for line in snapshot.lines) <= builtin._RANGED_KEEP_CHARS + 200
    assert snapshot.window_lines > len(snapshot.lines)
