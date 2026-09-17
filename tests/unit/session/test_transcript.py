"""Transcript tests: append-only JSONL, replay, and the compaction boundary."""

from __future__ import annotations

import contextlib
import json
import shutil
from collections import deque
from pathlib import Path
from typing import Any

import pytest

from local_operator.harness.types import (
    CustomMessage,
    Message,
    TextContent,
    ToolCall,
    ToolContext,
    ToolResult,
    Usage,
)
from local_operator.mcp.tool_bridge import format_mcp_result
from local_operator.session import transcript as transcript_module
from local_operator.session.attachments import AttachmentStore
from local_operator.session.transcript import (
    ENTRY_MESSAGE,
    TRANSCRIPT_FILENAME,
    Transcript,
    TranscriptEntry,
    TranscriptPage,
    read_latest_custom,
    read_latest_custom_entry,
    read_replay_suffix,
    read_transcript_page,
    replay_entries,
)


@pytest.fixture
def transcript(tmp_path):
    return Transcript(tmp_path / "sess")


@pytest.mark.asyncio
async def test_append_message_writes_jsonl(transcript):
    message = Message.user("hello")
    entry = await transcript.append_message(message)
    assert entry.type == ENTRY_MESSAGE
    assert entry.id == message.id  # entry id IS the message id

    lines = transcript.path.read_text().splitlines()
    assert len(lines) == 1
    raw = json.loads(lines[0])
    assert raw["type"] == "message"
    assert raw["payload"]["role"] == "user"
    assert raw["payload"]["content"][0]["text"] == "hello"


@pytest.mark.asyncio
async def test_usage_cost_round_trips_and_old_rows_default_to_unreported(transcript):
    """Provider receipts are durable, while pre-receipt transcripts still load.

    ``exclude_defaults`` omits an absent cost from new rows exactly as old builds
    did. A reported zero must remain present because it means billed-as-free, not
    "the provider did not say".
    """
    charged = Message.assistant("charged")
    charged.usage = Usage(input_tokens=12, output_tokens=3, usd_cost=0.00125)
    free = Message.assistant("free")
    free.usage = Usage(input_tokens=4, output_tokens=1, usd_cost=0.0)
    await transcript.append_message(charged)
    await transcript.append_message(free)

    raw_rows = [json.loads(line) for line in transcript.path.read_text().splitlines()]
    assert raw_rows[0]["payload"]["usage"]["usd_cost"] == pytest.approx(0.00125)
    assert raw_rows[1]["payload"]["usage"]["usd_cost"] == 0.0

    replayed = Transcript(transcript.directory).build_llm_history()
    assert isinstance(replayed[0], Message)
    assert replayed[0].usage is not None
    assert replayed[0].usage.usd_cost == pytest.approx(0.00125)
    assert isinstance(replayed[1], Message)
    assert replayed[1].usage is not None
    assert replayed[1].usage.usd_cost == 0.0

    # This is the exact usage shape written before ``usd_cost`` existed. Pydantic
    # must supply the default rather than rejecting the whole assistant message.
    old = Message.model_validate(
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "old"}],
            "usage": {"input_tokens": 9, "output_tokens": 2},
        }
    )
    assert old.usage is not None
    assert old.usage.usd_cost is None


@pytest.mark.asyncio
async def test_append_recreates_a_vanished_directory(tmp_path):
    """The session must survive its directory being deleted underneath it.

    A sibling process's startup sweep (or a user tidying ``sessions/`` by
    hand) can remove a directory that still looks empty — the gap between
    Session construction and the first turn is as long as the user takes to
    type. The old behaviour was fatal: the first append raised
    ``FileNotFoundError: .../transcript.jsonl`` and the whole session died.
    The append now recreates the directory and rebuilds the file from the
    in-memory entries, so nothing already appended is lost either.
    """
    directory = tmp_path / "sess"
    transcript = Transcript(directory)
    await transcript.append_message(Message.user("before the deletion"))

    shutil.rmtree(directory)  # what the racing sweep used to do

    await transcript.append_message(Message.assistant("after the deletion"))

    lines = transcript.path.read_text().splitlines()
    assert len(lines) == 2  # rebuilt complete, not truncated to the new row
    texts = [json.loads(line)["payload"]["content"][0]["text"] for line in lines]
    assert texts == ["before the deletion", "after the deletion"]


@pytest.mark.asyncio
async def test_append_rebuilds_when_only_the_file_vanished(tmp_path):
    """The quieter variant of the vanished-directory wound (review R1-1).

    Deleting just ``transcript.jsonl`` with the directory intact never
    raises: ``"a"`` mode recreates the file, so the append "succeeds" while
    the file silently holds one row and memory holds the whole session — a
    resume would then replay a single message as if the rest never
    happened. The append must notice the file is gone and rebuild it
    complete from the in-memory entries.
    """
    directory = tmp_path / "sess"
    transcript = Transcript(directory)
    await transcript.append_message(Message.user("one"))
    await transcript.append_message(Message.assistant("two"))

    transcript.path.unlink()  # the user tidied the file, not the directory

    await transcript.append_message(Message.user("three"))

    lines = transcript.path.read_text().splitlines()
    assert len(lines) == 3  # rebuilt complete, not restarted at one row
    texts = [json.loads(line)["payload"]["content"][0]["text"] for line in lines]
    assert texts == ["one", "two", "three"]


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_point", ["write", "flush"])
async def test_failed_append_never_enters_memory_index_or_later_rebuild(
    tmp_path, monkeypatch, failure_point
):
    """A rejected producer row must not resurrect through a later rebuild."""
    directory = tmp_path / "sess"
    transcript = Transcript(directory)
    await transcript.append_message(
        Message.user("admitted", id="admitted"),
        producer_command_id="admitted",
    )
    real_open = Path.open
    fail_once = True

    class FailingHandle:
        def __init__(self, handle):  # noqa: ANN001
            self._handle = handle

        def __enter__(self):
            self._handle.__enter__()
            return self

        def __exit__(self, *args):  # noqa: ANN002, ANN202
            return self._handle.__exit__(*args)

        def write(self, value):  # noqa: ANN001, ANN201
            if failure_point == "write":
                self._handle.write(value[: max(1, len(value) // 2)])
                raise OSError("injected write failure")
            return self._handle.write(value)

        def flush(self):
            if failure_point == "flush":
                raise OSError("injected flush failure")
            return self._handle.flush()

        def fileno(self):
            return self._handle.fileno()

    def failing_open(path, *args, **kwargs):  # noqa: ANN001, ANN202
        nonlocal fail_once
        handle = real_open(path, *args, **kwargs)
        if fail_once and path == transcript.path and args and args[0] == "a":
            fail_once = False
            return FailingHandle(handle)
        return handle

    monkeypatch.setattr(Path, "open", failing_open)
    with pytest.raises(OSError, match=failure_point):
        await transcript.append_message(
            Message.user("failed", id="failed"),
            producer_command_id="failed",
        )

    assert [entry.id for entry in transcript.entries()] == ["admitted"]
    assert not transcript.has_admitted_command("failed")
    assert [entry.id for entry in Transcript(directory).entries()] == ["admitted"]

    transcript.path.unlink()
    await transcript.append_message(
        Message.user("later", id="later"),
        producer_command_id="later",
    )
    reopened = Transcript(directory)
    assert [entry.id for entry in reopened.entries()] == ["admitted", "later"]
    assert not reopened.has_admitted_command("failed")


@pytest.mark.asyncio
@pytest.mark.parametrize("command_kind", ["prompt", "steer"])
async def test_failed_first_rebuild_is_retryable_without_resurrection(
    tmp_path, monkeypatch, command_kind
):
    """An open failure before the first row leaves no disk or memory claim."""
    directory = tmp_path / "sess"
    transcript = Transcript(directory)
    real_open = Path.open
    fail_once = True

    def failing_open(path, *args, **kwargs):  # noqa: ANN001, ANN202
        nonlocal fail_once
        if fail_once and path == transcript.path and args and args[0] == "w":
            fail_once = False
            raise OSError("injected open failure")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", failing_open)
    command_id = f"retry-{command_kind}"
    failed = Message.user("failed", id=command_id)
    with pytest.raises(OSError, match="open failure"):
        await transcript.append_message(failed, producer_command_id=command_id)

    assert transcript.entries() == []
    assert not transcript.path.exists()
    assert not transcript.has_admitted_command(command_id)

    await transcript.append_message(failed, producer_command_id=command_id)
    reopened = Transcript(directory)
    assert [entry.id for entry in reopened.entries()] == [command_id]
    assert reopened.has_admitted_command(command_id)


def test_only_valid_user_message_rows_claim_producer_markers(tmp_path) -> None:
    directory = tmp_path / "sess"
    directory.mkdir()
    valid_payload = {
        "kind": "message",
        "role": "user",
        "content": [{"text": "valid"}],
        "producer_command_id": "valid",
    }
    invalid_payloads = [
        {**valid_payload, "producer_command_id": "missing-kind", "kind": None},
        {**valid_payload, "producer_command_id": "custom-kind", "kind": "custom"},
        {**valid_payload, "producer_command_id": "assistant", "role": "assistant"},
        {**valid_payload, "producer_command_id": "system", "role": "system"},
        {**valid_payload, "producer_command_id": "malformed-content", "content": "text"},
        {**valid_payload, "producer_command_id": "malformed-block", "content": [42]},
        {**valid_payload, "producer_command_id": "   "},
    ]
    entries = [
        TranscriptEntry(id="valid-row", ts=1, type=ENTRY_MESSAGE, payload=valid_payload),
        *[
            TranscriptEntry(
                id=f"invalid-{index}", ts=2 + index, type=ENTRY_MESSAGE, payload=payload
            )
            for index, payload in enumerate(invalid_payloads)
        ],
        TranscriptEntry(
            id="import-collision",
            ts=20,
            type="custom",
            payload={**valid_payload, "producer_command_id": "import-collision"},
        ),
    ]
    (directory / "transcript.jsonl").write_text(
        "".join(entry.to_json() + "\n" for entry in entries), encoding="utf-8"
    )

    transcript = Transcript(directory)

    assert transcript.has_admitted_command("valid")
    for command_id in [
        "missing-kind",
        "custom-kind",
        "assistant",
        "system",
        "malformed-content",
        "malformed-block",
        "import-collision",
    ]:
        assert not transcript.has_admitted_command(command_id)


@pytest.mark.asyncio
async def test_reloads_from_disk(tmp_path):
    directory = tmp_path / "sess"
    first = Transcript(directory)
    await first.append_message(Message.user("one"))
    await first.append_message(Message.assistant("two"))

    reopened = Transcript(directory)
    assert len(reopened.entries()) == 2
    history = reopened.build_llm_history()
    assert [m.text for m in history if isinstance(m, Message)] == ["one", "two"]


@pytest.mark.asyncio
async def test_malformed_lines_dropped_individually(tmp_path):
    directory = tmp_path / "sess"
    store = Transcript(directory)
    await store.append_message(Message.user("good"))
    with store.path.open("a") as handle:
        handle.write("{not json\n")
        handle.write(
            json.dumps({"id": "x", "ts": 1, "type": "message"}) + "\n"
        )  # missing payload ok

    reopened = Transcript(directory)
    assert len(reopened.entries()) == 2  # corrupt line dropped, rest survives


@pytest.mark.asyncio
async def test_backward_pages_reach_start_with_stable_ids_and_skip_malformed_rows(tmp_path):
    directory = tmp_path / "sess"
    transcript = Transcript(directory)
    expected = []
    for index in range(235):
        entry = await transcript.append_message(Message.user(f"row {index}"))
        expected.append(entry.id)
    with transcript.path.open("a") as handle:
        handle.write("{malformed\n")

    seen = []
    cursor = None
    while True:
        page = read_transcript_page(directory, before_id=cursor, limit=100)
        seen[0:0] = [entry.id for entry in page.entries]
        if not page.has_more:
            break
        cursor = page.entries[0].id

    assert seen == expected


@pytest.mark.asyncio
async def test_backward_page_reconciles_a_cursor_removed_by_replacement(tmp_path):
    directory = tmp_path / "sess"
    transcript = Transcript(directory)
    for index in range(5):
        await transcript.append_message(Message.user(f"old {index}"))
    stale_cursor = transcript.entries()[2].id
    replacement = TranscriptEntry("replacement", 1.0, ENTRY_MESSAGE, {"role": "user"})
    transcript.path.write_text(replacement.to_json() + "\n")

    page = read_transcript_page(directory, before_id=stale_cursor, limit=100)

    assert page.reconciled is True
    assert [entry.id for entry in page.entries] == ["replacement"]
    assert page.has_more is False


def test_backward_page_reports_missing_transcript_without_creating_it(tmp_path):
    directory = tmp_path / "missing"
    with pytest.raises(FileNotFoundError):
        read_transcript_page(directory)
    assert not directory.exists()


def test_backward_page_rejects_two_cursors_and_a_zero_limit(tmp_path):
    directory, ids = _variant_journal(tmp_path, "one-row")
    with pytest.raises(ValueError):
        read_transcript_page(directory, before_id=ids[0], through_id=ids[0])
    with pytest.raises(ValueError):
        read_transcript_page(directory, limit=0)


# --- The backward page read: differential equivalence and structural cost ----
#
# ``read_transcript_page`` reads BACKWARD from EOF so the desktop open path
# costs a page instead of a journal. The window alignment is the whole risk of
# that rewrite: which row a cursor includes or excludes, and whether rows
# appended after a snapshot cursor are skipped. ``_forward_transcript_page``
# below is the pre-rewrite implementation kept verbatim as the ORACLE — the
# forward page is the contract — and these tests are what prove the two agree
# on every shape a journal can take. Do not delete the oracle because the
# forward reader is gone from the module: a differential test whose reference
# was re-derived from the code under test proves nothing.


def _forward_transcript_page(
    directory: str | Path,
    *,
    before_id: str | None = None,
    through_id: str | None = None,
    limit: int = 100,
) -> TranscriptPage:
    """The pre-rewrite forward implementation, kept as the differential oracle."""
    if before_id is not None and through_id is not None:
        raise ValueError("choose before_id or through_id, not both")
    if limit < 1:
        raise ValueError("limit must be at least 1")
    path = Path(directory) / TRANSCRIPT_FILENAME
    if not path.exists():
        raise FileNotFoundError(path)
    retained: deque[TranscriptEntry] = deque(maxlen=limit + 1)
    found = before_id is None and through_id is None
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            entry = TranscriptEntry.from_json(line)
            if entry is None:
                continue
            if before_id is not None and entry.id == before_id:
                found = True
                break
            retained.append(entry)
            if through_id is not None and entry.id == through_id:
                found = True
                break
    if through_id is not None and not found:
        return TranscriptPage((), False, True)
    if before_id is not None and not found:
        tail = _forward_transcript_page(directory, limit=limit)
        return TranscriptPage(tail.entries, tail.has_more, True)
    rows = tuple(retained)
    return TranscriptPage(entries=rows[-limit:], has_more=len(rows) > limit)


def _page_signature(page: TranscriptPage) -> tuple[Any, ...]:
    """Everything the contract promises, exactly as a caller can observe it."""
    return (
        tuple(entry.id for entry in page.entries),
        tuple(entry.to_json() for entry in page.entries),
        page.has_more,
        page.reconciled,
    )


def _row_line(index: int, *, pad: int = 0) -> str:
    return TranscriptEntry(
        f"row-{index:05d}",
        1.0 + index,
        ENTRY_MESSAGE,
        {"role": "user", "content": f"row {index}" + "x" * pad},
    ).to_json()


def _variant_journal(
    tmp_path: Path, variant: str, *, rows: int = 2400, pad: int = 1024
) -> tuple[Path, list[str]]:
    """A session dir for ``variant``, plus the ids of its VALID rows in order.

    One builder for every shape the backward reader must agree with the forward
    one on: a clean tail, a cursor mid-journal, a cursor on the last row, a
    cursor that is absent, rows appended after a cursor, malformed rows, a
    blank line, a torn trailing line, a file without a trailing newline, a row
    larger than one 1 MiB chunk, a single row, and an empty journal.
    """
    directory = tmp_path / variant
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / TRANSCRIPT_FILENAME
    ids = [f"row-{index:05d}" for index in range(rows)]
    lines = [_row_line(index, pad=pad) for index in range(rows)]
    if variant == "huge-row":
        # Larger than _BACKWARD_CHUNK_BYTES on purpose: the walker must join the
        # carried tail once, not re-split it per chunk (that is how the first
        # version of the replay reader went quadratic).
        huge = TranscriptEntry(
            "row-huge",
            5.0,
            "custom",
            {"custom_type": "bulk", "pad": "x" * (1 << 21)},
        ).to_json()
        lines = lines[: rows // 2] + [huge] + lines[rows // 2 :]
    if variant == "dirty":
        lines = lines[:6] + ["", "{ not json at all", "   "] + lines[6:]
    if variant == "torn":
        # A row cut off mid-flight by a crash: no terminating newline, so the
        # walker's leading fragment is not a row a replay could ever use.
        lines = lines + ['{"id": "row-torn"']
    if variant == "one-row":
        lines, ids = lines[:1], ids[:1]
    if variant == "empty":
        lines = []
    trailing_newline = variant not in ("no-newline", "torn")
    body = "\n".join(lines)
    path.write_text(body + ("\n" if lines and trailing_newline else ""), encoding="utf-8")
    return directory, ids


def _page_cases(ids: list[str]) -> list[dict[str, Any]]:
    """Cursor shapes, including both "row appended after the snapshot" halves."""
    if not ids:
        return [{}, {"limit": 3}, {"before_id": "row-00000"}, {"through_id": "row-00000"}]
    middle = ids[len(ids) // 2]
    return [
        {},  # bare tail
        {"limit": 1},
        {"limit": 3},
        {"limit": 500},
        {"before_id": ids[-1]},  # cursor on the newest row
        {"before_id": middle},
        {"before_id": ids[0]},  # cursor on the oldest row
        {"before_id": ids[0], "limit": 1},
        {"before_id": "no-such-row"},  # evicted cursor -> reconciled tail
        {"through_id": ids[-1]},  # inclusive cut on the newest row
        {"through_id": middle},  # AND rows newer than the cut must be skipped
        {"through_id": ids[0]},
        {"through_id": ids[0], "limit": 3},
        {"through_id": "no-such-row"},  # missing cut -> empty + reconciled
        {"through_id": "row-huge"},  # cursor inside a larger-than-chunk row
        {"before_id": "row-huge"},
    ]


@pytest.mark.parametrize(
    "variant", ["plain", "dirty", "torn", "no-newline", "huge-row", "one-row", "empty"]
)
def test_backward_page_read_matches_the_forward_oracle(tmp_path, variant):
    """Same rows, same order, same JSON, same flags — for every cursor shape."""
    directory, ids = _variant_journal(tmp_path, variant)
    for kwargs in _page_cases(ids):
        reference = _forward_transcript_page(directory, **kwargs)
        page = read_transcript_page(directory, **kwargs)
        assert _page_signature(page) == _page_signature(reference), (variant, kwargs)


class _CountingHandle:
    """Delegating file wrapper that counts every byte handed to the reader.

    Counting on the way OUT is what makes the measurement implementation-
    agnostic: the forward reader iterates lines (``__next__``) and the backward
    reader pulls chunks (``read``), and both land in the same counter. A text
    handle reports characters, which equal bytes on these ASCII fixtures.
    """

    def __init__(self, handle: Any, counter: list[int]) -> None:
        self._handle = handle
        self._counter = counter

    def read(self, size: int = -1) -> Any:
        data = self._handle.read(size)
        self._counter[0] += len(data)
        return data

    def __next__(self) -> Any:
        line = next(self._handle)
        self._counter[0] += len(line)
        return line

    def __iter__(self) -> "_CountingHandle":
        return self

    def __enter__(self) -> "_CountingHandle":
        self._handle.__enter__()
        return self

    def __exit__(self, *exc: Any) -> Any:
        return self._handle.__exit__(*exc)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._handle, name)


@contextlib.contextmanager
def _counted_reads(monkeypatch, path: Path):
    """Count the bytes ``read_transcript_page`` pulls out of exactly ``path``."""
    counter = [0]
    real_open = Path.open

    def counting_open(self: Path, *args: Any, **kwargs: Any) -> Any:
        handle = real_open(self, *args, **kwargs)
        return _CountingHandle(handle, counter) if self == path else handle

    monkeypatch.setattr(Path, "open", counting_open)
    yield counter


def test_tail_page_read_cost_is_the_page_not_the_journal(tmp_path, monkeypatch):
    """The open path's cost must be the page, and NOT a function of the journal.

    STRUCTURAL, not a clock: what is asserted is bytes handed to the reader on
    its way out of ``Path.open``, which no amount of machine load can move. A
    9.4 MB journal whose tail page is ~1% of the file must be read for a page
    plus the chunk that carries it, and the SAME instrument pointed at the
    forward implementation — below, in this test — reads the whole journal and
    breaks the bound. That is what makes this a guard rather than a description
    of whatever the current code happens to do.
    """
    directory, ids = _variant_journal(tmp_path, "plain", rows=10000, pad=900)
    path = directory / TRANSCRIPT_FILENAME
    size = path.stat().st_size
    with _counted_reads(monkeypatch, path) as counted:
        page = read_transcript_page(directory, limit=100)
    page_bytes = sum(len(entry.to_json().encode("utf-8")) for entry in page.entries)
    assert [entry.id for entry in page.entries] == ids[-100:]
    # A page, the chunk that carries it, and at most one chunk of boundary
    # slack: ``_iter_complete_lines_backward`` reads whole chunks backward.
    assert counted[0] <= page_bytes + 2 * transcript_module._BACKWARD_CHUNK_BYTES
    # ... and never anything like the journal, however long the journal is.
    assert counted[0] * 4 < size

    with _counted_reads(monkeypatch, path) as oracle_counted:
        reference = _forward_transcript_page(directory, limit=100)
    assert _page_signature(reference) == _page_signature(page)
    # The old read IS the journal: it needs the whole file to answer the same
    # question, so it fails the bound above by a factor of ~2.5 on this file.
    assert oracle_counted[0] * 4 >= size


def test_cursor_page_read_cost_is_the_page_plus_the_walk_to_it(tmp_path, monkeypatch):
    """The OTHER page shape this change moves: a ``before_id`` page mid-journal.

    Same structural instrument as the tail case, bounded the other way round.
    Reading backward to a cursor must cost the page, the bytes AFTER that cursor
    (which the walk has to pass through on its way down), and one chunk of
    boundary slack — and must not touch the history BEFORE the cursor at all,
    which is most of this file and is exactly what the forward reader paid for.
    Both halves are asserted, and the forward implementation is measured here
    too so the bound cannot rot into a description.
    """
    directory, ids = _variant_journal(tmp_path, "plain", rows=10000, pad=900)
    path = directory / TRANSCRIPT_FILENAME
    size = path.stat().st_size
    cursor = ids[9000]  # ~1000 rows from the tail: one long scroll back
    rows = [raw for raw in path.read_bytes().split(b"\n") if raw.strip()]
    row_ids = [json.loads(raw)["id"] for raw in rows]
    cursor_index = row_ids.index(cursor)
    tail_bytes = sum(len(raw) + 1 for raw in rows[cursor_index:])  # cursor row + newer
    head_bytes = size - tail_bytes

    with _counted_reads(monkeypatch, path) as counted:
        page = read_transcript_page(directory, before_id=cursor, limit=100)
    page_bytes = sum(len(entry.to_json().encode("utf-8")) for entry in page.entries)
    assert [entry.id for entry in page.entries] == ids[8900:9000]
    assert counted[0] <= page_bytes + tail_bytes + 2 * transcript_module._BACKWARD_CHUNK_BYTES
    # The prefix before the cursor — ~90% of this journal — is never read.
    assert counted[0] < head_bytes

    with _counted_reads(monkeypatch, path) as oracle_counted:
        reference = _forward_transcript_page(directory, before_id=cursor, limit=100)
    assert _page_signature(reference) == _page_signature(page)
    # ... while the forward read has to walk that prefix, so it exceeds the same
    # bound this test asserts: that is the guard's teeth for this shape.
    assert oracle_counted[0] >= head_bytes
    assert oracle_counted[0] > page_bytes + tail_bytes + 2 * transcript_module._BACKWARD_CHUNK_BYTES


def test_backward_page_answers_a_duplicated_id_with_its_newest_row(tmp_path):
    """A repeated cursor id resolves to the NEWER row, deliberately (review R1-2).

    Ids are ``uuid4().hex`` on the append path, so a repeat means a corrupted or
    concatenated journal rather than anything the codebase can write. The
    forward reader answered with whichever occurrence it MET first — the oldest;
    reading backward meets the newest first, and matching the old answer would
    mean walking to the file's start whenever an id repeats, giving up the whole
    optimisation for the one case that is already anomalous. So the newest
    occurrence is the boundary, and this test pins that rather than leaving it to
    the direction of the scan. Row order is unaffected and the cursor row is
    still never returned, so a duplicate cannot make a caller loop.
    """
    directory = tmp_path / "dupes"
    directory.mkdir()

    def _dup_row(content: str) -> str:
        return TranscriptEntry(
            "dup", 1.0, ENTRY_MESSAGE, {"role": "user", "content": content}
        ).to_json()

    written = [
        _row_line(0),
        _row_line(1),
        _row_line(2),
        _dup_row("dup older"),
        _row_line(3),
        _dup_row("dup newer"),
        _row_line(4),
    ]
    (directory / TRANSCRIPT_FILENAME).write_text("\n".join(written) + "\n", encoding="utf-8")

    # before_id excludes its row: the page ends at the row before the NEWEST dup.
    before = read_transcript_page(directory, before_id="dup")
    assert [entry.id for entry in before.entries] == [
        "row-00000",
        "row-00001",
        "row-00002",
        "row-00003",
    ]
    assert before.has_more is False and before.reconciled is False
    assert "dup" not in {entry.id for entry in before.entries}

    # through_id includes its row: the page's newest row is the NEWER dup.
    through = read_transcript_page(directory, through_id="dup")
    assert [entry.id for entry in through.entries] == [
        "row-00000",
        "row-00001",
        "row-00002",
        "dup",
        "row-00003",
        "dup",
    ]
    assert json.loads(through.entries[-1].to_json())["payload"]["content"] == "dup newer"
    assert json.loads(through.entries[3].to_json())["payload"]["content"] == "dup older"

    assert [
        entry.id for entry in read_transcript_page(directory, before_id="dup", limit=1).entries
    ] == ["row-00003"]


@pytest.mark.asyncio
async def test_custom_entries_ignored_by_replay(transcript):
    await transcript.append_message(Message.user("hi"))
    await transcript.append_custom("wake_schedules", {"schedules": []})
    history = transcript.build_llm_history()
    assert len(history) == 1
    assert transcript.latest_custom("wake_schedules") == {"schedules": []}


@pytest.mark.asyncio
async def test_latest_custom_backward_scan(transcript):
    await transcript.append_custom("wake_schedules", {"v": 1})
    await transcript.append_custom("wake_schedules", {"v": 2})
    await transcript.append_custom("other", {"v": 99})
    assert transcript.latest_custom("wake_schedules") == {"v": 2}
    assert transcript.latest_custom("missing") is None


@pytest.mark.asyncio
async def test_compaction_replay_boundary(transcript):
    """Latest compaction wins: summary marker + entries from first_kept onward;
    nothing before the cut replays."""
    m1 = Message.user("early one")
    m2 = Message.user("early two")
    m3 = Message.user("kept")
    await transcript.append_message(m1)
    await transcript.append_message(m2)
    entry3 = await transcript.append_message(m3)
    await transcript.append_compaction("SUMMARY-TEXT", entry3.id, tokens_before=5000)
    m4 = Message.assistant("after")
    await transcript.append_message(m4)

    history = transcript.build_llm_history()
    assert len(history) == 3
    marker = history[0]
    assert isinstance(marker, CustomMessage)
    assert marker.custom_type == "compaction_summary"
    assert marker.details["summary"] == "SUMMARY-TEXT"
    assert isinstance(history[1], Message) and history[1].text == "kept"
    assert isinstance(history[2], Message) and history[2].text == "after"
    # Nothing before the cut leaked in.
    assert all(not (isinstance(m, Message) and "early" in m.text) for m in history)


@pytest.mark.asyncio
async def test_latest_compaction_wins(transcript):
    """Two compactions: only the newest boundary applies."""
    m1 = Message.user("a")
    e1 = await transcript.append_message(m1)
    m2 = Message.user("b")
    e2 = await transcript.append_message(m2)
    m3 = Message.user("c")
    await transcript.append_message(m3)

    await transcript.append_compaction("FIRST", e2.id, 100)
    await transcript.append_compaction("SECOND", e1.id, 200)

    history = transcript.build_llm_history()
    marker = history[0]
    assert isinstance(marker, CustomMessage)
    assert marker.details["summary"] == "SECOND"
    texts = [m.text for m in history[1:]]
    assert texts == ["a", "b", "c"]


@pytest.mark.asyncio
async def test_compaction_with_missing_first_kept_replays_full_history(transcript):
    """If first_kept_entry_id cannot be found, replay falls back to the FULL
    history. The old fallback (compaction_index + 1) pointed past the kept
    window and silently dropped every message compaction promised to keep;
    replaying too much is recoverable at the next compaction, amnesia is not."""
    await transcript.append_message(Message.user("before"))
    await transcript.append_compaction("S", "no-such-entry", 10)
    await transcript.append_message(Message.user("after"))
    history = transcript.build_llm_history()
    assert [m.text for m in history if isinstance(m, Message)] == ["before", "after"]


@pytest.mark.asyncio
async def test_message_round_trip_keeps_provider_payload(transcript):
    """Tool-result metadata rides in provider_payload and must survive replay."""
    tool_msg = Message(
        role="tool",
        tool_call_id="c1",
        tool_name="read",
        provider_payload={"details": {"path": "/tmp/x"}, "useless": False},
    )
    await transcript.append_message(tool_msg)
    history = transcript.build_llm_history()
    assert len(history) == 1
    restored = history[0]
    assert isinstance(restored, Message)
    assert restored.provider_payload == {"details": {"path": "/tmp/x"}, "useless": False}
    assert restored.tool_call_id == "c1"


@pytest.mark.asyncio
async def test_spilled_mcp_metadata_round_trip_omits_duplicate_text(
    transcript, tmp_path, monkeypatch
):
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "cfg"))
    body = "large MCP row\n" * 1_000
    result = format_mcp_result(
        {
            "content": [{"type": "text", "text": body}],
            "structuredContent": {"rowCount": 1_000},
            "isError": False,
        },
        "c1",
        "mcp__demo_rows",
        ToolContext(session_id="transcript-spill"),
    )
    message = Message(
        role="tool",
        content=result.content,
        tool_call_id=result.tool_call_id,
        tool_name=result.tool_name,
        provider_payload={"details": result.details},
    )
    await transcript.append_message(message)

    restored = transcript.build_llm_history()[0]
    assert isinstance(restored, Message)
    assert restored.provider_payload is not None
    details = restored.provider_payload["details"]
    assert details["server_result"]["content"] == []
    assert details["server_result"]["structuredContent"] == {"rowCount": 1_000}
    assert details["spill"]["handle"].startswith("spill://")
    assert body not in transcript.path.read_text()


@pytest.mark.asyncio
async def test_custom_message_round_trip(transcript):
    custom = CustomMessage(custom_type="skill_prompt", details={"name": "s"})
    await transcript.append_message(custom)
    history = transcript.build_llm_history()
    assert len(history) == 1
    restored = history[0]
    assert isinstance(restored, CustomMessage)
    assert restored.custom_type == "skill_prompt"


def test_entry_from_json_rejects_bad_rows():
    assert TranscriptEntry.from_json("nonsense") is None
    assert (
        TranscriptEntry.from_json('{"id": "a", "type": "message"}') is not None
    )  # payload defaults
    assert TranscriptEntry.from_json('{"ts": 1, "type": "message"}') is None  # missing id


@pytest.mark.asyncio
async def test_message_batch_fsyncs_once_off_loop_and_indexes_after_commit(tmp_path, monkeypatch):
    """Disk spikes cannot park sibling sessions; one closed batch is one commit."""
    import asyncio
    import os
    import threading

    loop_thread = threading.get_ident()
    calls = []
    real_fsync = os.fsync

    def fsync(fd):
        calls.append(threading.get_ident())
        real_fsync(fd)

    monkeypatch.setattr(os, "fsync", fsync)
    # Production async constructors use a worker: birth is durable before
    # live discovery, while each later journal batch still has ONE fsync.
    transcript = await asyncio.to_thread(Transcript, tmp_path / "batch")
    assert len(calls) == 1
    assert all(thread != loop_thread for thread in calls)
    calls.clear()
    messages = [Message.user(str(index)) for index in range(12)]
    rows = await transcript.append_messages(messages)
    assert len(calls) == 1
    assert all(thread != loop_thread for thread in calls)
    assert [row.id for row in rows] == [message.id for message in messages]
    assert all(transcript.has_entry(message.id) for message in messages)
    newest = transcript.latest_user_entry()
    assert newest is not None and newest.id == messages[-1].id
    assert [row.id for row in Transcript(transcript.directory).entries()] == [
        m.id for m in messages
    ]


@pytest.mark.asyncio
async def test_cancelled_write_settles_before_successor_and_publishes_durable_rows(
    tmp_path, monkeypatch
):
    """Cancellation cannot release the journal lock while its syscall is live."""
    import asyncio
    import threading

    transcript = Transcript(tmp_path / "cancelled")
    started = threading.Event()
    release = threading.Event()
    original = transcript._write_entries

    def write(rows, *, preserve_mtime: bool = False):
        if rows[0].id == "first":
            started.set()
            assert release.wait(10), "test did not release the disk worker"
        original(rows, preserve_mtime=preserve_mtime)

    monkeypatch.setattr(transcript, "_write_entries", write)
    first = asyncio.create_task(transcript.append_message(Message.user("one", id="first")))
    assert await asyncio.to_thread(started.wait, 10)
    first.cancel()
    second = asyncio.create_task(transcript.append_message(Message.user("two", id="second")))
    assert not transcript.has_entry("first")
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await first
    await second
    assert [entry.id for entry in transcript.entries()] == ["first", "second"]
    assert [entry.id for entry in Transcript(transcript.directory).entries()] == ["first", "second"]


@pytest.mark.asyncio
async def test_cancelled_fold_cannot_replace_a_successors_new_message(tmp_path, monkeypatch):
    """Append and atomic file compaction share the same cancellation boundary."""
    import asyncio
    import threading

    transcript = Transcript(tmp_path / "fold-cancel")
    old = Message(role="tool", content=[TextContent(text="large " * 2000)], tool_call_id="old")
    await transcript.append_message(old)
    await transcript.append_prune(old.id, "pruned")
    entered = threading.Event()
    release = threading.Event()
    original = transcript._replace_file

    def replace(payload):
        entered.set()
        assert release.wait(10), "test did not release the file replacement"
        original(payload)

    monkeypatch.setattr(transcript, "_replace_file", replace)
    fold = asyncio.create_task(transcript.compact_file(min_reclaim_bytes=0))
    assert await asyncio.to_thread(entered.wait, 10)
    fold.cancel()
    append = asyncio.create_task(transcript.append_message(Message.user("new", id="new")))
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await fold
    await append
    reopened = Transcript(transcript.directory)
    assert reopened.has_entry("new")
    assert transcript.has_entry("new")
    history = reopened.build_llm_history()
    assert any(isinstance(message, Message) and message.text == "new" for message in history)


# -- backward suffix replay ---------------------------------------------------
#
# ``read_replay_suffix`` exists so a legacy attach stops paying for a whole-file
# JSON parse when the replay only uses rows from the latest compaction's kept
# window. The property that makes that safe is EQUIVALENCE with the whole-file
# parse, so every shape the replay handles specially is asserted here against
# ``Transcript(...).build_llm_history()`` rather than against expected values:
# if the semantics move, both sides move together and the suffix must follow.


def _dump(history):
    return [message.model_dump(mode="json") for message in history]


async def _journal(directory: Path, *, shape: str) -> list[str]:
    """A journal whose kept window sits behind enough bulk to span chunks."""
    transcript = Transcript(directory)
    ids: list[str] = []
    for n in range(8):
        ids.append((await transcript.append_message(Message.user(f"early q{n}"))).id)
        ids.append((await transcript.append_message(Message.assistant(f"early a{n}"))).id)
    # Bookkeeping rows larger than the reader's chunk, so a single row spans
    # a chunk boundary and the incomplete-head handling is exercised.
    for index in range(3):
        await transcript.append_custom("bulk", {"index": index, "pad": "x" * (1 << 20)})
    if shape != "no-compaction":
        for n in range(6):
            ids.append((await transcript.append_message(Message.user(f"kept q{n}"))).id)
            ids.append((await transcript.append_message(Message.assistant(f"kept a{n}"))).id)
        preserved = (
            [{"id": ids[2], "text": "early q1"}] if shape == "compaction+preserved" else None
        )
        await transcript.append_compaction("summary", ids[-12], 100, preserved_user_turns=preserved)
        for n in range(2):
            ids.append((await transcript.append_message(Message.user(f"post q{n}"))).id)
            ids.append((await transcript.append_message(Message.assistant(f"post a{n}"))).id)
        if shape in ("compaction+prunes", "folded"):
            await transcript.append_prune(ids[-6], "[pruned]")
            await transcript.append_prune(ids[-1], "[pruned]")
        if shape == "folded":
            await transcript.compact_file(min_reclaim_bytes=0)
    await transcript.append_custom("checkpoint", {"epoch": "e1"})
    return ids


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "shape",
    ["no-compaction", "compaction", "compaction+prunes", "compaction+preserved", "folded"],
)
async def test_replay_suffix_matches_whole_file_parse(tmp_path, shape):
    directory = tmp_path / "sess"
    ids = await _journal(directory, shape=shape)
    full = Transcript(directory)
    store = AttachmentStore(directory)

    suffix = read_replay_suffix(directory, checkpoint_types="checkpoint")
    assert _dump(replay_entries(suffix.entries, store)) == _dump(full.build_llm_history())
    assert suffix.checkpoint == full.latest_custom("checkpoint")
    if shape != "no-compaction":
        # The point of the reader: it must NOT have read the early history.
        assert suffix.bytes_read < (directory / "transcript.jsonl").stat().st_size
    else:
        # No compaction means no boundary to stop at: whole file, honestly.
        assert suffix.bytes_read == (directory / "transcript.jsonl").stat().st_size

    # Cuts at the tail, inside the kept window, and (for compacted journals)
    # BEFORE the marker — where a compaction above the cut must be ignored
    # exactly as the whole-file replay ignores it after slicing at the cursor.
    cuts = [ids[-1], ids[-3]] + ([ids[-6]] if shape != "no-compaction" else [])
    for cut in cuts:
        cut_suffix = read_replay_suffix(directory, through_id=cut)
        assert cut_suffix.through_present
        assert _dump(replay_entries(cut_suffix.entries, store, through_id=cut)) == _dump(
            full.build_llm_history(through_id=cut)
        ), (shape, cut)


@pytest.mark.asyncio
async def test_replay_suffix_reports_an_unknown_cursor_instead_of_cutting(tmp_path):
    """A cursor not in the journal is an answer, never a silent tail cut.

    The strict-cut contract: the caller decides whether an absent cursor means
    "legacy best-effort, replay everything" or "the negotiated window is gone".
    The reader's job is to have read far enough to KNOW it is absent.
    """
    directory = tmp_path / "sess"
    await _journal(directory, shape="compaction")
    suffix = read_replay_suffix(directory, through_id="not-a-row")
    assert not suffix.through_present
    assert suffix.bytes_read == (directory / "transcript.jsonl").stat().st_size


@pytest.mark.asyncio
async def test_a_type_that_may_be_absent_never_defeats_the_suffix_stop(tmp_path):
    """Review R1-2: an opportunistic type must not gate the reader's stop.

    The cold path wants the frontend checkpoint AND the spend record out of one
    pass. Requiring both meant a pre-ledger journal — which by definition has no
    spend row — could never satisfy the stop, so the reader walked to the start
    of the file: measured on the operator's store, session ``28a800c6a783``,
    4,194,304 bytes became **16,701,655** (the whole 16.7 MB journal) on every
    cold open, plus the ~1.23x-weight parse it pins.

    The assertion is structural rather than a byte budget: adding a type that is
    absent must not change how far the reader reads, and a row of that type
    inside the window must still come out of the same pass.
    """
    directory = tmp_path / "sess"
    await _journal(directory, shape="compaction")
    size = (directory / "transcript.jsonl").stat().st_size

    required_only = read_replay_suffix(directory, checkpoint_types=("checkpoint",))
    with_absent = read_replay_suffix(
        directory,
        checkpoint_types=("checkpoint",),
        opportunistic_types=("session_spend.v1",),
    )
    assert required_only.bytes_read < size, "the reader must not read the whole journal"
    assert with_absent.bytes_read == required_only.bytes_read
    assert "session_spend.v1" not in with_absent.checkpoints

    # And the opportunistic row IS collected when the backward scan passes it —
    # including from the tail, which is where a per-call record always is.
    await Transcript(directory).append_custom("session_spend.v1", {"version": 1, "micro": 7})
    with_row = read_replay_suffix(
        directory,
        checkpoint_types=("checkpoint",),
        opportunistic_types=("session_spend.v1",),
    )
    assert with_row.checkpoints["session_spend.v1"] == {"version": 1, "micro": 7}
    assert with_row.checkpoints["checkpoint"] == {"epoch": "e1"}
    assert with_row.bytes_read < size + 4096, "a row at the tail must not cost a full read"


def test_replay_suffix_absent_journal_is_empty_history(tmp_path):
    """Same answer as ``Transcript(directory)``: no file means nothing yet."""
    suffix = read_replay_suffix(tmp_path / "absent")
    assert suffix.entries == () and suffix.bytes_read == 0
    assert (
        replay_entries(suffix.entries, None) == Transcript(tmp_path / "absent").build_llm_history()
    )


# --- Audit replay mode -------------------------------------------------------
#
# The seam these tests guard: ``replay_entries`` answers two different
# questions, and the first one must not move. ``mode="context"`` is what the
# MODEL sees and is bounded by compaction; ``mode="audit"`` is what the
# CONVERSATION contained and is bounded by nothing. Fusing them is how the
# display window came to page only post-compaction rows.


async def _compacted(directory: Path, *, compactions: int = 1, rows_each: int = 4):
    """Journal with ``compactions`` cuts, each preserving the turn before it."""
    transcript = Transcript(directory)
    written: list[Message] = []
    for cut in range(compactions):
        batch = [Message.user(f"cut {cut} row {i}") for i in range(rows_each)]
        await transcript.append_messages(batch)
        written.extend(batch)
        await transcript.append_compaction(
            f"summary {cut}",
            batch[-1].id,
            500,
            preserved_user_turns=[{"id": batch[0].id, "text": batch[0].text}],
        )
    tail = [Message.assistant("after the last cut")]
    await transcript.append_messages(tail)
    written.extend(tail)
    return transcript, written


@pytest.mark.asyncio
async def test_context_mode_is_byte_identical_to_build_llm_history(tmp_path):
    """The regression guard for the entire audit change.

    ``mode="context"`` is the default and must remain what this function has
    always produced, compaction cut and preserved turns included. If this ever
    fails, the model's context has changed — which the audit feature is
    explicitly not allowed to do.
    """
    transcript, _ = await _compacted(tmp_path / "sess", compactions=3)
    expected = transcript.build_llm_history()
    replayed = replay_entries(transcript.entries(), transcript._attachments, mode="context")
    assert [m.model_dump(mode="json") for m in replayed] == [
        m.model_dump(mode="json") for m in expected
    ]
    # And the default is context, so no existing caller changed behaviour.
    default = replay_entries(transcript.entries(), transcript._attachments)
    assert [m.model_dump(mode="json") for m in default] == [
        m.model_dump(mode="json") for m in expected
    ]


@pytest.mark.asyncio
async def test_audit_mode_returns_every_message_row_on_disk(tmp_path):
    """Audit mode's contract: nothing the journal holds is unreachable."""
    transcript, written = await _compacted(tmp_path / "sess", compactions=3)
    audited = replay_entries(transcript.entries(), transcript._attachments, mode="audit")
    rows = [m for m in audited if getattr(m, "custom_type", None) != "compaction_summary"]
    assert [m.id for m in rows] == [m.id for m in written]
    # Context mode, on the same journal, reaches almost none of them — which is
    # the defect that made this mode necessary.
    assert len(transcript.build_llm_history()) < len(rows)


@pytest.mark.asyncio
async def test_audit_mode_does_not_reinject_preserved_user_turns(tmp_path):
    """Preserved turns are COPIES of rows audit mode already replays.

    Re-injecting them would duplicate every preserved turn under its ORIGINAL
    id, and the TUI's mount-time id dedupe would then swallow the second — so
    the visible symptom is a MISSING row, not a doubled one.
    """
    transcript, written = await _compacted(tmp_path / "sess", compactions=3)
    audited = replay_entries(transcript.entries(), transcript._attachments, mode="audit")
    ids = [m.id for m in audited]
    assert len(ids) == len(set(ids))
    for message in written:
        assert ids.count(message.id) == 1


@pytest.mark.asyncio
async def test_audit_mode_emits_one_marker_per_compaction_in_place(tmp_path):
    """Every cut is shown where it happened, not hoisted to the front.

    Context mode keeps only the latest cut and puts its marker at the head. A
    reader auditing a session with 48 compactions needs to see where each one
    fell; one marker at the boundary would misrepresent the other 47 as
    ordinary history.
    """
    transcript, _ = await _compacted(tmp_path / "sess", compactions=3, rows_each=2)
    audited = replay_entries(transcript.entries(), transcript._attachments, mode="audit")
    positions = [
        index
        for index, m in enumerate(audited)
        if getattr(m, "custom_type", None) == "compaction_summary"
    ]
    assert len(positions) == 3
    assert positions != sorted(positions)[:1] * 3  # not all hoisted to one spot
    assert positions[0] > 0, "the first marker sits after the rows it followed"
    assert positions == sorted(positions)


@pytest.mark.asyncio
async def test_audit_slice_applies_a_prune_journalled_after_its_window(tmp_path):
    """The one cross-row dependency a bounded window can miss.

    A prune is always appended AFTER the row it targets, so a prune for a row
    inside the slice can sit past ``end_index``. Deriving the prune map from
    the slice would replay tool output compaction already blanked.
    """
    from local_operator.session.transcript import audit_slice

    directory = tmp_path / "sess"
    transcript = Transcript(directory)
    target = Message.assistant("secret tool output")
    await transcript.append_message(target)
    filler = [Message.user(f"later {i}") for i in range(30)]
    await transcript.append_messages(filler)
    # The prune lands at the very END of the journal, far past the window.
    await transcript.append_prune(target.id, "[pruned]")

    entries = transcript.entries()
    end = next(i for i, e in enumerate(entries) if e.id == filler[0].id)
    rows, _indices, _start = audit_slice(entries, transcript._attachments, end_index=end, limit=50)
    replayed = next(m for m in rows if m.id == target.id)
    assert isinstance(replayed, Message)
    assert replayed.text == "[pruned]"
    assert (replayed.provider_payload or {}).get("pruned") is True


@pytest.mark.asyncio
async def test_audit_slice_never_begins_inside_a_tool_group(tmp_path):
    """A window opening on a tool RESULT renders a settled call as interrupted.

    The result row is drawn on the card of the call above it, so a page whose
    first row is a result would show a card with no call until some unrelated
    scroll happened to fetch it.
    """
    from local_operator.session.transcript import audit_slice

    transcript = Transcript(tmp_path / "sess")
    call = Message.assistant(
        "calling", tool_calls=[ToolCall(id="call-1", name="probe", arguments={})]
    )
    result = Message.tool_result(
        ToolResult(tool_call_id="call-1", tool_name="probe", content=[TextContent(text="done")])
    )
    rows = [Message.user("before"), call, result, Message.user("after")]
    await transcript.append_messages(rows)
    entries = transcript.entries()
    end = next(i for i, e in enumerate(entries) if e.id == rows[-1].id)
    # limit=1 would otherwise land the window exactly on the result row.
    messages, _indices, _start = audit_slice(
        entries, transcript._attachments, end_index=end, limit=1
    )
    assert messages[0].id == call.id
    assert [m.id for m in messages] == [call.id, result.id]


class TestDurableConversationPath:
    """The ``started``-bit seed discriminator: a REAL turn's row counts, a
    CustomMessage row persisted without a turn (QA Q4) does not."""

    #: The quiet-dial note exactly as ``append_messages`` persists it: a
    #: message row whose payload kind is ``custom`` (QA Q4's repro row).
    _PEER_NOTE = (
        '{"id":"p1","ts":1,"type":"message","payload":{"kind":"custom",'
        '"custom_type":"peer_message","attribution":"user","details":{"text":"hi"}}}'
    )

    def _write(self, tmp_path: Path, rows: list[str]) -> Path:
        path = tmp_path / "transcript.jsonl"
        path.write_text("".join(row + "\n" for row in rows))
        return path

    def test_a_real_user_turn_counts(self, tmp_path: Path) -> None:
        from local_operator.session.transcript import durable_conversation_path

        row = '{"id":"m1","ts":1,"type":"message","payload":{"kind":"message","role":"user"}}'
        path = self._write(tmp_path, [row])
        assert durable_conversation_path(path) is True

    def test_a_peer_note_only_transcript_does_not_count(self, tmp_path: Path) -> None:
        """QA Q4: a quiet-dialled peer note persists as a message row (kind
        ``custom``) with NO turn running; treating it as history seeds
        ``started=True`` and a peer wake then drives a turn into a session
        the owner never typed in."""
        from local_operator.session.transcript import durable_conversation_path

        path = self._write(
            tmp_path,
            [
                self._PEER_NOTE,
                '{"id":"t1","ts":2,"type":"custom","payload":{"custom_type":"title"}}',
            ],
        )
        assert durable_conversation_path(path) is False

    def test_peer_notes_plus_a_real_turn_counts(self, tmp_path: Path) -> None:
        from local_operator.session.transcript import durable_conversation_path

        row = '{"id":"m1","ts":2,"type":"message","payload":{"kind":"message","role":"assistant"}}'
        path = self._write(tmp_path, [self._PEER_NOTE, row])
        assert durable_conversation_path(path) is True

    def test_a_legacy_row_without_a_kind_marker_counts(self, tmp_path: Path) -> None:
        """``kind`` arrived with producer admission; a row predating it IS a
        plain Message — the custom writer always tagged its rows."""
        from local_operator.session.transcript import durable_conversation_path

        path = self._write(
            tmp_path, ['{"id":"m1","ts":1,"type":"message","payload":{"role":"user"}}']
        )
        assert durable_conversation_path(path) is True

    def test_missing_and_torn_files_read_as_unstarted(self, tmp_path: Path) -> None:
        from local_operator.session.transcript import durable_conversation_path

        assert durable_conversation_path(tmp_path / "absent.jsonl") is False
        torn = self._write(tmp_path, ["{torn", "not json"])
        assert durable_conversation_path(torn) is False


@pytest.mark.asyncio
async def test_search_spend_rows_survive_a_resume(tmp_path, transcript):
    """A resumed conversation recovers the search spend its tool rows recorded.

    Driven through the REAL persistence path — harness tool-result bookkeeping,
    encode, file, replay — because what can break is not the read: it is whether
    ``ToolResult.details`` reaches the row at all, and whether a SECOND session
    built on the same directory can see it. A test that handed the accessor a
    hand-made payload would prove neither.

    The spend itself is real money that the model-token accounting never saw
    (``web_search`` bills separately), so the ledger that displays it is
    process-wide and starts empty in a new process; this read is the only thing
    that stands between a resumed session and reporting a search-heavy
    conversation as free.
    """
    result = ToolResult(
        tool_call_id="call-1",
        tool_name="web_search",
        content=[TextContent(text="Snippets are intentionally capped.")],
        details={
            "provider": "deepseek",
            "search_cost": {
                "usd": 0.0031,
                "basis": "token estimate",
                "priced_from_usage": False,
                "session_usd": 0.0031,
                "session_searches": 1,
                "provider_searches": 1,
            },
        },
    )
    await transcript.append_message(Message.tool_result(result))
    transcript.flush()

    # A SECOND Transcript on the same directory is the resume: nothing is shared
    # but the file, so the row that comes back is the one that was persisted.
    resumed = Transcript(tmp_path / "sess")
    rows = resumed.search_spend_rows()
    assert len(rows) == 1
    assert rows[0]["provider"] == "deepseek"
    assert rows[0]["usd"] == pytest.approx(0.0031)
    assert rows[0]["basis"] == "token estimate"

    # And through a real Session, which is the accessor the host actually calls
    # on adopt. ``_make_session`` is the harness this module's siblings use; the
    # point here is only that the seeding happens at construction and is
    # available before any turn runs.
    from tests.unit.session.test_cut_off_turns import _make_session

    session = _make_session(tmp_path / "sess")
    assert session.restored_search_spend() == tuple(rows)


@pytest.mark.asyncio
async def test_a_pruned_search_row_keeps_its_cost(tmp_path, transcript):
    """Pruning blanks a tool result's CONTENT, not its cost bookkeeping.

    Compaction blanks hundreds of tool rows on a long conversation, and the
    spend recovery reads those rows. If a prune took the details with the
    content, a resumed session would quietly report a fraction of what it spent
    — with no mark to say so — which is the loss this read exists to prevent.
    """
    result = ToolResult(
        tool_call_id="call-1",
        tool_name="web_search",
        content=[TextContent(text="a page of snippets")],
        details={
            "provider": "tavily",
            "search_cost": {
                "usd": 0.0080,
                "basis": "tavily credits",
                "priced_from_usage": True,
            },
        },
    )
    entry = await transcript.append_message(Message.tool_result(result))
    await transcript.append_prune(entry.id, "[pruned]")
    await transcript.compact_file(min_reclaim_bytes=0)

    rows = Transcript(tmp_path / "sess").search_spend_rows()
    assert rows and rows[0]["provider"] == "tavily"
    assert rows[0]["usd"] == pytest.approx(0.0080)
    assert "a page of snippets" not in transcript.path.read_text()


@pytest.mark.asyncio
async def test_a_reads_cost_row_is_recovered_with_its_own_ledger_key(tmp_path, transcript):
    """Resume recovery must find ``read_cost`` rows, and key them as the tool did.

    Round 1's review found the recovery reading only ``search_cost``, so a
    read-heavy conversation came back with its searches and none of its reads.
    Round 2's found the replacement deriving the key from a ``provider`` field a
    read result does not carry, so the restored row landed under ``:read``
    instead of the live ``deepseek:read`` -- one kind of spend split across two
    rows on resume.

    Driven through the real persistence path, like the search-row test beside it:
    what can break is whether ``ToolResult.details`` reaches the row.
    """
    result = ToolResult(
        tool_call_id="call-1",
        tool_name="web_read",
        content=[TextContent(text="Answer.")],
        details={
            "pages": 4,
            "read_cost": {
                "ledger_provider": "deepseek:read",
                "usd": 0.002,
                "basis": "tokens at list price",
                "session_usd": 0.002,
                "session_searches": 0,
                "session_reads": 1,
                "reads": 1,
            },
        },
    )
    await transcript.append_message(Message.tool_result(result))
    transcript.flush()

    resumed = Transcript(tmp_path / "sess")
    rows = resumed.search_spend_rows()

    assert len(rows) == 1
    assert rows[0]["provider"] == "deepseek:read"
    assert rows[0]["kind"] == "read"
    assert rows[0]["usd"] == pytest.approx(0.002)


@pytest.mark.asyncio
async def test_usages_since_newest_shrink_matches_the_method(tmp_path: Path) -> None:
    """The cold reader's entry point and the owner's method are ONE rule.

    ``usages_since_newest_shrink`` exists because a cold viewer holds only the
    journal SUFFIX it read (``read_replay_suffix``), never a ``Transcript`` — and
    it must still apply the same compaction/prune boundary the owner applies when
    seeding a status readout. If the two ever drift, one conversation reports two
    different contexts depending on which surface opened it, and the reading that
    survives the drift picks the compaction trigger's figure (``Session._last_usage``)
    for a session the desktop would describe differently.

    Asserted at every boundary state the scan can meet, with the reading list
    itself pinned so the equality cannot pass on two empty answers.
    """
    from local_operator.session.transcript import usages_since_newest_shrink

    def both(transcript: Transcript) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        entries = transcript.entries()
        return usages_since_newest_shrink(entries), transcript.usages_since_compaction()

    transcript = Transcript(tmp_path / "sess")
    await transcript.append_message(Message.user("q1"))
    await transcript.append_message(Message.assistant("a1", usage=Usage(context_tokens=100_000)))
    await transcript.append_message(Message.user("q2"))
    kept = await transcript.append_message(
        Message.assistant("a2", usage=Usage(context_tokens=200_000))
    )

    # No shrink yet: every reading in the file is on the live side.
    module, method = both(transcript)
    assert [usage["context_tokens"] for usage in module] == [100_000, 200_000]
    assert module == method

    # A compaction moves the boundary for both readers: readings preceding the
    # marker describe the pre-pass context.
    await transcript.append_compaction("summary of q1/a1", kept.id, 300_000)
    await transcript.append_message(Message.assistant("a3", usage=Usage(context_tokens=300_000)))
    module, method = both(transcript)
    assert [usage["context_tokens"] for usage in module] == [300_000]
    assert module == method

    # A prune moves it again, without a marker of its own.
    await transcript.append_prune(kept.id, "[pruned]")
    module, method = both(transcript)
    assert module == [] and method == module

    # Folding the prune journal away (the on-disk form of the same file) is
    # semantically invisible: the boundary stays where the prune was.
    await transcript.compact_file(min_reclaim_bytes=1)
    module, method = both(Transcript(tmp_path / "sess"))
    assert module == [] and method == module


# --- the one-row metadata read (perf/session-load-central-cache) --------------
#
# ``read_latest_custom_entry``/``read_latest_custom`` exist because seven call
# sites answered a ONE-ROW question by constructing a whole ``Transcript`` — a
# full JSON decode of the journal. Measured against a clean ``origin/main``
# worktree at ``bf67bf699`` with ``scripts/bench_session_page.py`` (median of 3
# samples per operation, host load average 179-260, recorded per worker in the
# output — the full table is in ``docs/evidence/session-load-central-cache``):
# 2286.7 ms to construct and 2059.5 ms to read on the 261 MB conversation,
# 580.7/546.3 ms on the 96 MB one, and two of those seven sites are on the
# desktop OPEN path. Two things make the swap safe and both are tested here: the
# DIFFERENTIAL (it must answer exactly what the resident object answered) and the
# STRUCTURAL cost (it must not pay for the rows above its match).


#: Sentinel for "this key is absent from the row" — distinct from any real value,
#: including ``None``, because a row with no ``details`` key and a row whose
#: ``details`` is null are two different rows to both readers.
_OMIT = object()


def _custom_row(
    row_id: str,
    custom_type: Any = _OMIT,
    *,
    details: Any = _OMIT,
    ts: float = 1.0,
) -> str:
    """One ``custom`` journal row, with either key omitted when not given."""
    payload: dict[str, Any] = {}
    if custom_type is not _OMIT:
        payload["custom_type"] = custom_type
    if details is not _OMIT:
        payload["details"] = details
    return TranscriptEntry(row_id, ts, "custom", payload).to_json()


def _message_row(row_id: str) -> str:
    return TranscriptEntry(
        row_id, 0.5, ENTRY_MESSAGE, {"role": "user", "content": row_id}
    ).to_json()


def _mixed_custom_journal(directory: Path) -> None:
    """Every shape the two readers must agree on, in one journal.

    Three custom types interleaved with message rows, so "newest wins" and
    "a different type does not stop the scan" are both exercised; a repeated
    type; a row with NO ``custom_type`` key (which indexes under ``""`` in
    ``_index_entry``); a ``details`` value that is not a mapping (where both
    implementations must fail identically, because a caller cannot be allowed to
    tell which one answered it); and a malformed line, which both skip.
    """
    directory.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for index in range(5):
        lines.append(_message_row(f"m{index}"))
        lines.append(
            _custom_row(
                f"todo-{index}",
                "todo_snapshot",
                details={"items": [{"n": index}], "version": index},
                ts=10.0 + index,
            )
        )
        lines.append(
            _custom_row(
                f"checkpoint-{index}",
                "frontend_state_checkpoint_v1",
                details={"state": {"cwd": f"/work/{index}"}},
                ts=20.0 + index,
            )
        )
        lines.append(
            _custom_row(
                f"roster-{index}", "subagent_roster", details={"records": [{"id": index}]}, ts=30.0
            )
        )
    # Newest rows for each type, so an oldest-wins bug is visible rather than
    # hidden behind a single occurrence.
    lines.append(_custom_row("todo-last", "todo_snapshot", details={"items": ["last"]}, ts=99.0))
    lines.append(_custom_row("untyped", _OMIT, details={"anything": True}, ts=98.0))
    lines.append(_custom_row("not-a-mapping", "odd", details=["a", "list"], ts=97.0))
    lines.append(_custom_row("no-details", "bare", ts=96.0))
    # A MESSAGE row carrying a ``custom_type`` key, and it is LAST so a backward
    # scan meets it first. ``_index_entry`` indexes custom rows only, so the
    # resident object ignores this one entirely; a scan that filtered on the
    # payload alone would answer it instead of the newest real row. That is not a
    # hypothetical row: any writer that puts a ``custom_type`` field in a
    # message payload produces it, and the two implementations must agree about
    # what it means.
    lines.append(
        TranscriptEntry(
            "decoy",
            100.0,
            ENTRY_MESSAGE,
            {"role": "user", "content": "decoy", "custom_type": "todo_snapshot"},
        ).to_json()
    )
    lines.append("{not json at all")
    lines.append("")
    (directory / TRANSCRIPT_FILENAME).write_text("\n".join(lines) + "\n", encoding="utf-8")


def _answer(call: Any) -> Any:
    """A call's result OR the exception it raised, in one comparable value."""
    try:
        return ("value", call())
    except Exception as exc:  # noqa: BLE001 — the point is to compare the failure too
        return ("raised", type(exc).__name__, str(exc))


@pytest.mark.parametrize(
    "custom_type",
    [
        "todo_snapshot",
        "frontend_state_checkpoint_v1",
        "subagent_roster",
        "",
        "odd",
        "bare",
        "never_written",
    ],
)
def test_the_one_row_read_matches_the_resident_transcript(tmp_path, custom_type):
    """The differential, with the resident ``Transcript`` as the reference.

    ``Transcript`` is the reference because it IS what every call site used until
    now: ``_index_entry`` keeps the LAST custom row per type as it walks the
    journal forward, so "newest wins" is that object's rule and a backward scan
    has to reproduce it. Both projections are compared, and both the VALUE and
    the EXCEPTION, so a ``details`` value that is not a mapping cannot be papered
    over by a reader that answers ``{}`` where the resident object raises.
    """
    directory = tmp_path / "sess"
    _mixed_custom_journal(directory)
    resident = Transcript(directory, defer_materialise=True)

    assert _answer(lambda: read_latest_custom(directory, custom_type)) == _answer(
        lambda: resident.latest_custom(custom_type)
    )
    assert _answer(lambda: read_latest_custom_entry(directory, custom_type)) == _answer(
        lambda: resident.latest_custom_entry(custom_type)
    )


def test_the_newest_match_wins_and_the_scan_does_not_stop_on_another_type(tmp_path):
    """The two ways a backward scan can answer the wrong row, pinned separately."""
    directory = tmp_path / "sess"
    _mixed_custom_journal(directory)

    assert read_latest_custom(directory, "todo_snapshot") == {"items": ["last"]}
    entry = read_latest_custom_entry(directory, "todo_snapshot")
    assert entry is not None and entry.id == "todo-last"
    # The type that appears BETWEEN the tail and ``todo-last`` must not end the
    # scan: the very next row after ``todo-last`` is a different type.
    assert read_latest_custom(directory, "subagent_roster") == {"records": [{"id": 4}]}
    # A row whose details key is absent is `{}`, as the resident object has it.
    assert read_latest_custom(directory, "bare") == {}


def test_a_metadata_read_never_creates_the_directory(tmp_path):
    """A pure read of somebody else's session leaves nothing behind.

    ``Transcript`` mkdirs its directory on construction, which is why the TUI's
    child reads had to borrow ``defer_materialise``; this reader opens the journal
    read-only, so the guarantee is structural rather than opt-in. The phantom-
    directory defect that rule descends from is in ``todo_panel``'s docstring.
    """
    missing = tmp_path / "never-created"

    assert read_latest_custom(missing, "todo_snapshot") is None
    assert read_latest_custom_entry(missing, "todo_snapshot") is None
    assert not missing.exists()


def test_a_metadata_read_decodes_only_the_rows_above_the_match(tmp_path, monkeypatch):
    """A spy on ``TranscriptEntry.from_json``: the walk stops at the newest match.

    THE MUTATION THIS CATCHES: dropping the early return — or filtering
    ``custom_type`` only after assembling every row — turns the decode count back
    into the whole journal, which is the 2.5-3.0 s this change removes on a real
    conversation. Counted at the DECODE rather than in bytes or milliseconds: a
    count is identical on an idle laptop and a wedged CI runner, while a wall
    clock here measures the machine (AGENTS.md §Timing).
    """
    directory = tmp_path / "sess"
    directory.mkdir()
    rows = [_message_row(f"m{index:04d}") for index in range(4000)]
    rows.append(_custom_row("the-match", "todo_snapshot", details={"items": []}, ts=99.0))
    rows.extend(_message_row(f"tail{index}") for index in range(2))
    (directory / TRANSCRIPT_FILENAME).write_text("\n".join(rows) + "\n", encoding="utf-8")

    decoded = 0
    real = transcript_module.TranscriptEntry.from_json

    def counting(line: str) -> Any:
        nonlocal decoded
        decoded += 1
        return real(line)

    monkeypatch.setattr(transcript_module.TranscriptEntry, "from_json", staticmethod(counting))
    entry = read_latest_custom_entry(directory, "todo_snapshot")

    assert entry is not None and entry.id == "the-match"
    assert decoded <= 5, f"the walk decoded {decoded} rows for a 3-row tail"
    assert decoded >= 3, "the match and the rows above it must have been decoded"


def test_a_metadata_read_near_the_head_costs_the_journal_and_is_still_correct(
    tmp_path, monkeypatch
):
    """The honest worst case, pinned so it cannot be mistaken for a regression.

    A legacy ``subagent_roster`` row written once near byte zero reaches the file
    start, so that read costs what today's whole-file parse costs — never worse,
    and it answers correctly rather than reporting a bounded "unknown". This is
    the case the design refuses to add a ``max_bytes`` knob for: a ceiling here
    would have to invent an answer for a caller that today always gets one.
    """
    directory = tmp_path / "sess"
    directory.mkdir()
    rows = [_custom_row("ancient", "subagent_roster", details={"records": []}, ts=1.0)]
    rows.extend(_message_row(f"m{index:04d}") for index in range(500))
    (directory / TRANSCRIPT_FILENAME).write_text("\n".join(rows) + "\n", encoding="utf-8")

    decoded = 0
    real = transcript_module.TranscriptEntry.from_json

    def counting(line: str) -> Any:
        nonlocal decoded
        decoded += 1
        return real(line)

    monkeypatch.setattr(transcript_module.TranscriptEntry, "from_json", staticmethod(counting))
    entry = read_latest_custom_entry(directory, "subagent_roster")

    assert entry is not None and entry.id == "ancient"
    assert decoded == 501, "the walk must reach the file start to answer honestly"


def test_a_byte_corrupt_journal_is_read_where_the_resident_object_raises(tmp_path):
    """The one NAMED divergence, pinned rather than left in a docstring.

    A journal whose newest matching row holds one invalid byte — what an
    interrupted append truncated mid-character leaves — is answered here: the
    damaged byte decodes to U+FFFD under ``errors="replace"`` and the row still
    parses. The resident ``Transcript`` decodes through a strict ``read_text`` and
    raises ``UnicodeDecodeError`` for the same file.

    The direction is deliberate (a damaged journal is answered, not turned into a
    500 on the desktop open), and a differential that asserted EQUALITY for this
    shape would be asserting the wrong thing — which is why the equality matrix
    above covers every well-formed journal and the divergence gets this test
    instead.
    """
    directory = tmp_path / "sess"
    directory.mkdir()
    raw = (
        TranscriptEntry(
            "torn",
            2.0,
            "custom",
            {"custom_type": "todo_snapshot", "details": {"x": "MARKER-VALUE"}},
        )
        .to_json()
        .encode("utf-8")
    )
    assert b'"MARKER-VALUE"' in raw
    (directory / TRANSCRIPT_FILENAME).write_bytes(
        raw.replace(b'"MARKER-VALUE"', b'"MARK\xffER-VALUE"') + b"\n"
    )

    with pytest.raises(UnicodeDecodeError):
        Transcript(directory, defer_materialise=True)

    assert read_latest_custom(directory, "todo_snapshot") == {"x": "MARK\ufffdER-VALUE"}
