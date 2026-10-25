"""Read dedup: an unchanged, still-resident re-read is answered with a pointer.

Every acceptance case of the feature is here, each against the REAL read tool
(``execute_read``) and the real ``ReadLedger`` over a live message list, except
the Session-level end-to-end proof in
``tests/unit/session/test_read_dedup_session.py``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from local_operator.compaction.pruning import (
    MIN_PRUNE_TOKENS,
    SUPERSEDED_NOTICE,
    _blank,
    _is_pruned,
    prune_tool_outputs,
)
from local_operator.harness.types import Message, TextContent, ToolContext, ToolResult
from local_operator.tools import read_dedup
from local_operator.tools.builtin import execute_read
from local_operator.tools.read_dedup import DEDUP_DETAILS_KEY, LedgerEntry, ReadLedger


class Harness:
    """A live history plus a ledger reading it — what a Session provides.

    ``read`` appends the result exactly as the loop does
    (``Message.tool_result``), so residency is judged against real rows.
    """

    def __init__(self, cwd: Path, replay_bound: int = 8 * 1024) -> None:
        self.messages: list[Message] = []
        self.ledger = ReadLedger(lambda: self.messages, replay_bound=replay_bound)
        self.context = ToolContext(cwd=str(cwd), read_ledger=self.ledger)
        self.calls = 0

    async def read(self, path: str, **args: Any) -> ToolResult:
        self.calls += 1
        result = await execute_read(
            f"call_{self.calls:04d}", {"path": path, **args}, context=self.context
        )
        self.messages.append(Message.tool_result(result))
        return result


def text(result: ToolResult) -> str:
    return "".join(b.text for b in result.content if isinstance(b, TextContent))


def is_pointer(result: ToolResult) -> bool:
    return DEDUP_DETAILS_KEY in (result.details or {})


@pytest.fixture
def src(tmp_path: Path) -> Path:
    path = tmp_path / "mod.txt"
    path.write_text("".join(f"line {i} with some text\n" for i in range(1, 61)))
    return path


@pytest.fixture
def h(tmp_path: Path) -> Harness:
    return Harness(tmp_path)


@pytest.mark.asyncio
async def test_covered_span_dedups_and_points_at_the_earlier_call(h: Harness, src: Path) -> None:
    first = await h.read(str(src), range="10-40")
    assert not is_pointer(first) and "line 25 with" in text(first)

    second = await h.read(str(src), range="20-30")
    assert is_pointer(second)
    body = text(second)
    assert body.startswith(f"[unchanged since your earlier read of {src} lines 20-30")
    assert "call_0001"[-8:] in body and "raw=true" in body
    marker = second.details[DEDUP_DETAILS_KEY]
    assert marker["of"] == "call_0001" and marker["range"] == "20-30"
    assert marker["saved_chars"] > 0


@pytest.mark.asyncio
async def test_whole_file_read_covers_any_range_and_a_repeat(h: Harness, src: Path) -> None:
    await h.read(str(src))
    assert is_pointer(await h.read(str(src), range="55-"))
    assert is_pointer(await h.read(str(src)))
    # Relative spelling of the same realpath dedups too.
    assert is_pointer(await h.read("mod.txt", range="1-3"))


@pytest.mark.asyncio
async def test_partial_overlap_does_not_dedup(h: Harness, src: Path) -> None:
    await h.read(str(src), range="10-20")
    result = await h.read(str(src), range="15-25")
    assert not is_pointer(result)
    assert "line 25 with" in text(result)


@pytest.mark.asyncio
async def test_open_ended_range_needs_an_eof_entry(h: Harness, src: Path) -> None:
    await h.read(str(src), range="10-20")
    assert not is_pointer(await h.read(str(src), range="12-"))
    # A window that ran short of its request reached EOF: now "50-" is covered.
    await h.read(str(src), range="40-999")
    assert is_pointer(await h.read(str(src), range="50-"))
    assert is_pointer(await h.read(str(src), range="45-2000"))


@pytest.mark.asyncio
async def test_same_size_different_bytes_does_not_dedup(h: Harness, src: Path) -> None:
    await h.read(str(src))
    before = src.stat()
    data = src.read_bytes()
    src.write_bytes(data.replace(b"line 7 ", b"LINE 7 "))
    # Same size AND the old mtime restored: only the hash can tell.
    os.utime(src, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert src.stat().st_size == before.st_size
    result = await h.read(str(src), range="5-9")
    assert not is_pointer(result)
    assert "LINE 7" in text(result)


@pytest.mark.asyncio
async def test_pruned_earlier_result_does_not_dedup(h: Harness, src: Path) -> None:
    await h.read(str(src), range="1-40")
    _blank(h.messages[0], SUPERSEDED_NOTICE)
    result = await h.read(str(src), range="1-10")
    assert not is_pointer(result)
    assert "line 3 with" in text(result)


@pytest.mark.asyncio
async def test_replaced_history_after_compaction_does_not_dedup(h: Harness, src: Path) -> None:
    await h.read(str(src), range="1-40")
    # Compaction REPLACES the live list (marker + kept tail); the ledger
    # reads through the accessor, so the old result is simply not found.
    h.messages = [Message(role="user", content=[TextContent(text="[summary]")])]
    assert not is_pointer(await h.read(str(src), range="1-10"))


@pytest.mark.asyncio
async def test_only_the_head_the_replay_bound_keeps_is_recorded(
    tmp_path: Path, src: Path
) -> None:
    """The wire elides the middle of a result over the replay bound, so the
    model never saw those lines: only the contiguous head counts."""
    h = Harness(tmp_path, replay_bound=300)
    await h.read(str(src), range="1-40")
    (entry,) = h.ledger.candidates(os.path.realpath(src), 1, 1)
    assert 1 <= entry.end < 10 and not entry.eof
    assert is_pointer(await h.read(str(src), range=f"1-{entry.end}"))
    assert not is_pointer(await h.read(str(src), range="1-10"))
    assert not is_pointer(await h.read(str(src), range="35-40"))  # tail: not contiguous


@pytest.mark.asyncio
async def test_raw_true_is_the_escape_hatch(h: Harness, src: Path) -> None:
    await h.read(str(src), range="1-40")
    result = await h.read(str(src), range="1-10", raw=True)
    assert not is_pointer(result) and "line 3 with" in text(result)


@pytest.mark.asyncio
async def test_internal_urls_never_dedup(h: Harness, tmp_path: Path) -> None:
    h.context = ToolContext(
        cwd=str(tmp_path),
        read_ledger=h.ledger,
        resolve_internal_url=lambda url: "guide body\n" * 20,
        scratchpad_dir=str(tmp_path / "pad"),
    )
    (tmp_path / "pad").mkdir()
    (tmp_path / "pad" / "notes.md").write_text("note line\n" * 30)
    for target in ("guide://x", "skill://y", "scratchpad://notes.md"):
        await h.read(target)
        again = await h.read(target)
        assert not is_pointer(again), target
    assert len(h.ledger) == 0


@pytest.mark.asyncio
async def test_errors_images_and_summaries_are_never_recorded(h: Harness, tmp_path: Path) -> None:
    missing = await h.read(str(tmp_path / "nope.txt"))
    assert missing.is_error
    from PIL import Image

    Image.new("RGB", (4, 4)).save(tmp_path / "a.png")
    await h.read(str(tmp_path / "a.png"))
    py = tmp_path / "big.py"
    py.write_text("".join(f"def f{i}():\n    return {i}\n" for i in range(60)))
    summary = await h.read(str(py))
    assert summary.details and summary.details.get("summary")
    await h.read(str(tmp_path))  # a directory listing
    assert len(h.ledger) == 0
    # ...so a ranged read after a summary returns content, never a pointer.
    assert not is_pointer(await h.read(str(py), range="1-5"))


@pytest.mark.asyncio
async def test_a_clamped_read_records_only_the_lines_it_showed(h: Harness, tmp_path: Path) -> None:
    """The motivating measurement's biggest bucket: a re-request of lines a
    clamped read NEVER showed is not a re-read, and must return content."""
    huge = tmp_path / "huge.txt"
    huge.write_text("".join(f"{i:04d} " + "y" * 115 + "\n" for i in range(1, 301)))
    clamped = await h.read(str(huge))
    assert "[truncated at" in text(clamped)
    (entry,) = h.ledger.candidates(os.path.realpath(huge), 1, 1)
    # Clamp (8 KiB) then the replay bound (8 KiB, footer pushes it over) keep
    # roughly the first half of the clamp's head; whatever it is, it is a
    # strict head of the file, never EOF, and every recorded line is shown.
    assert 10 < entry.end < 60 and not entry.eof
    assert f"{entry.end:04d} yyy" in text(clamped)
    assert is_pointer(await h.read(str(huge), range=f"2-{entry.end}"))
    following = await h.read(str(huge), range=f"{entry.end}-{entry.end + 1}")
    assert not is_pointer(following)
    assert not is_pointer(await h.read(str(huge)))  # the whole file was never shown


@pytest.mark.asyncio
async def test_one_enormous_line_is_never_recorded(h: Harness, tmp_path: Path) -> None:
    wide = tmp_path / "wide.txt"
    wide.write_text("z" * 20_000 + "\nshort\n")
    await h.read(str(wide))
    await h.read(str(wide), range="1-1")
    assert len(h.ledger) == 0


def test_mirrored_clamp_constant_matches_the_tool() -> None:
    from local_operator.tools.builtin import READ_OUTPUT_LIMIT_CHARS

    assert read_dedup.READ_CLAMP_CHARS == READ_OUTPUT_LIMIT_CHARS


@pytest.mark.asyncio
async def test_pointer_is_skipped_when_it_would_be_no_shorter(h: Harness, tmp_path: Path) -> None:
    tiny = tmp_path / "tiny.txt"
    tiny.write_text("a\n")
    await h.read(str(tiny))
    assert not is_pointer(await h.read(str(tiny)))


@pytest.mark.asyncio
async def test_no_ledger_means_unchanged_behaviour(tmp_path: Path, src: Path) -> None:
    context = ToolContext(cwd=str(tmp_path))
    first = await execute_read("a", {"path": str(src)}, context=context)
    second = await execute_read("b", {"path": str(src)}, context=context)
    assert text(first) == text(second)


@pytest.mark.asyncio
async def test_pointer_never_supersedes_the_result_it_points_at(h: Harness, src: Path) -> None:
    """The load-bearing interaction: supersede keys on ``details['path']``.

    A pointer carrying the path would make compaction blank the ORIGINAL
    (same path, a later result) and leave the pointer dangling. Run the real
    prune pass with the idle flush forced so no cache guard can hide it.
    """
    await h.read(str(src))
    pointer = await h.read(str(src))
    assert is_pointer(pointer)
    assert "path" not in pointer.details
    original = h.messages[0]
    assert (len(text_of(original)) // 4) >= MIN_PRUNE_TOKENS  # a real candidate
    prune_tool_outputs(h.messages, now_ms=10**13, last_activity_ms=0)
    assert not _is_pruned(original), "the pointer erased the result it points at"
    assert "line 30 with" in text_of(original)
    # And a third read still dedups against it.
    assert is_pointer(await h.read(str(src), range="2-4"))


def text_of(message: Message) -> str:
    return "".join(b.text for b in message.content if isinstance(b, TextContent))


@pytest.mark.asyncio
async def test_parallel_batch_twin_is_not_evicted(h: Harness, src: Path) -> None:
    """Two reads in one batch: the second runs before the first's row lands.
    It must return content, and must NOT forget the first entry, which
    becomes a valid dedup target once the batch's results are appended."""
    first = await execute_read("p1", {"path": str(src)}, context=h.context)
    second = await execute_read("p2", {"path": str(src)}, context=h.context)
    assert not is_pointer(second)
    h.messages += [Message.tool_result(first), Message.tool_result(second)]
    assert is_pointer(await h.read(str(src), range="1-2"))


def test_blank_marker_matches_pruning_is_pruned() -> None:
    """``read_dedup._is_blanked`` duplicates ``pruning._is_pruned`` (the tool
    layer must not import compaction); pin the two together."""
    for payload in (
        None,
        {},
        {"pruned": True},
        {"details": {"pruned": True}},
        {"details": {"path": "x"}},
        {"details": "junk"},
    ):
        message = Message(role="tool", content=[], tool_call_id="x", provider_payload=payload)
        assert read_dedup._is_blanked(message) == _is_pruned(message), payload


def test_ledger_is_bounded_and_coverage_is_newest_first() -> None:
    fp = read_dedup.FileFingerprint(size=1, mtime_ns=0, sha256="h")
    ledger = ReadLedger(lambda: [], replay_bound=100, max_entries=3)
    for i in range(5):
        ledger.record(LedgerEntry("/f", 1, 10, False, fp, f"c{i}"))
    assert len(ledger) == 3
    assert [e.tool_call_id for e in ledger.candidates("/f", 2, 5)] == ["c4", "c3", "c2"]
