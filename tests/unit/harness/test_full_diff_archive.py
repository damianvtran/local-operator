"""The COMPLETE write/edit diff beyond the 200-row inline preview, end to end.

Operator report: "There seems to be an issue with reviewing diffs, can you make
sure that we can see all of them with the git style diffs green and red
highlighted ... this was in a subagent view on the block of the dead work".
Root cause: the inline preview keeps 200 rows / 120 KiB and every row past it
was discarded at the producer, so no surface could show it.

These drive the REAL Session with the real write/edit tools and the session's
own redactor, and read back through the one shared page loader
(``tools.diff_archive.load_page``) every reader (desktop routes) uses. Every
secret case scans ALL stored bytes: the transcript, the archive store, and the
live end events.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest

from local_operator.harness.types import (
    ModelSpec,
    StreamEndEvent,
    StreamTextDelta,
    StreamToolCallDelta,
    ToolContext,
    ToolExecutionEndEvent,
)
from local_operator.session.session import Session
from local_operator.session.transcript import Transcript
from local_operator.tools import diff_archive
from local_operator.tools.builtin import (
    DIFF_TRUNCATED_ROWS_KEY,
    build_edit_tool,
    build_write_tool,
)
from local_operator.variables import VariableStore

pytestmark = pytest.mark.asyncio

SESSION_ID = "fd0000000001"


def _provider(calls: list[tuple[str, str, dict[str, Any]] | None]):
    """Scripted provider: a call per request, ``None`` (or past the end) ends the turn."""
    requests: list[Any] = []

    def provider(request, signal=None):
        index = len(requests)
        requests.append(request)

        async def generate():
            step = calls[index] if index < len(calls) else None
            if step is not None:
                call_id, name, args = step
                yield StreamToolCallDelta(
                    index=0, id=call_id, name=name, argument_delta=json.dumps(args)
                )
                yield StreamEndEvent(stop_reason="toolUse")
            else:
                yield StreamTextDelta(delta="done")
                yield StreamEndEvent(stop_reason="stop")

        return generate()

    return provider


async def _run(
    tmp_path: Path,
    calls: list[tuple[str, str, dict[str, Any]] | None],
    *,
    secrets: tuple[str, ...] = (),
    redact: bool = True,
) -> tuple[Session, list[Any]]:
    variables = VariableStore(cwd=str(tmp_path))
    for value in secrets:
        assert variables.register_redaction(value)
    session = Session(
        model=ModelSpec(provider="test", model_id="synthetic", context_window=100000),
        stream_fn=_provider(calls),
        tools=[build_write_tool(), build_edit_tool()],
        transcript=Transcript(tmp_path / "sessions" / SESSION_ID),
        system_blocks_provider=lambda *_: [],
        yolo=True,
        cwd=str(tmp_path),
        variables=variables if redact else None,
    )
    events: list[Any] = []
    session.subscribe(events.append)
    await session.prompt("go")
    return session, events


def _session_dir(tmp_path: Path) -> Path:
    return tmp_path / "sessions" / SESSION_ID


def _reconstruct(
    session_dir: Path, call_id: str, *, limit: int = 200
) -> tuple[list[str], list[dict[str, Any]]]:
    """Walk the server-issued cursor to the end and re-join every row.

    Asserts the v1 continuation contract on every page: the exact requested
    cursor is echoed, the WHOLE response (CRUD envelope included, JSON-escaped)
    fits the byte ceiling, fragments keep their row's marker, the cursor always
    moves forward, and ``None`` arrives only at the end.
    """
    rows: list[str] = []
    pages: list[dict[str, Any]] = []
    start: int | None = 0
    offset = 0
    while start is not None:
        page = diff_archive.load_page(
            session_dir, SESSION_ID, call_id, start=start, offset=offset, limit=limit
        )
        pages.append(page)
        assert page["state"] == "archived", page
        assert (page["start"], page["offset"]) == (start, offset)
        assert page["clipped_rows"] == []
        assert 1 <= len(page["rows"]) <= limit
        envelope = {"status": 200, "message": "Desktop session result.", "result": page}
        assert len(json.dumps(envelope)) <= diff_archive.PAGE_MAX_BYTES
        for i, fragment in enumerate(page["rows"]):
            if i == 0 and offset:
                assert rows[-1][:1] == fragment[:1]
                rows[-1] += fragment[1:]
            else:
                rows.append(fragment)
        next_start, next_offset = page["next_start"], page["next_offset"]
        if next_start is None:
            assert next_offset is None
            assert start + len(page["rows"]) == page["total_rows"]
        elif next_offset:
            assert next_start == start + len(page["rows"]) - 1
            if next_start == start:
                assert next_offset > offset, "same-row continuation must advance"
        else:
            assert next_offset == 0 and next_start == start + len(page["rows"])
        start, offset = next_start, next_offset or 0
    assert len(rows) == pages[-1]["total_rows"]
    return rows, pages


def _all_rows(tmp_path: Path, call_id: str) -> list[str]:
    rows, pages = _reconstruct(_session_dir(tmp_path), call_id)
    assert all(page["next_offset"] in (0, None) for page in pages), "no row was split"
    return rows


def _details(session: Session) -> list[dict[str, Any]]:
    return [
        cast(dict[str, Any], m.provider_payload)["details"]
        for m in session.transcript.build_llm_history()
        if m.role == "tool"
    ]


def _every_stored_byte(tmp_path: Path, session: Session, events: list[Any]) -> str:
    """Transcript + archive store + live end events, as one searchable blob.

    The assistant row's tool-call ARGUMENTS are the model's own input (a
    different, separately scrubbed channel), so only TOOL-RESULT rows count.
    """
    rows = [
        line
        for line in session.transcript.path.read_text().splitlines()
        if json.loads(line).get("payload", {}).get("role") == "tool"
    ]
    store = tmp_path / "config" / diff_archive.DIFF_ARCHIVE_DIRNAME
    stored = [p.read_text(encoding="utf-8") for p in store.glob("*")] if store.exists() else []
    live = [e.model_dump_json() for e in events if isinstance(e, ToolExecutionEndEvent)]
    assert rows and live
    # Archive rows are JSON-encoded; decode so a non-ASCII escape cannot hide one.
    decoded = []
    for text in stored:
        for line in text.split("\n"):
            try:
                decoded.append(str(json.loads(line)))
            except ValueError:
                decoded.append(line)
    return "\n".join(rows + live + stored + decoded)


@pytest.fixture(autouse=True)
def _config(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))


def _big(n: int, tag: str = "") -> str:
    lines = [f"value_{index} = {index}{tag}" for index in range(n)]
    lines[0] = "FIRST_SENTINEL = 1"
    lines[n // 2] = "MIDDLE_SENTINEL = 2"
    lines[-1] = "FINAL_SENTINEL = 3"
    return "\n".join(lines) + "\n"


def _literal(height: int, tag: str) -> list[str]:
    """A synthetic registered literal (not a credential shape), as its lines."""
    return [f"SYNTH-{tag}-{n:02d}-QKV{n * 7919:06d}" for n in range(height)]


async def test_write_over_preview_archives_complete_diff_live_and_persisted(tmp_path):
    target = tmp_path / "big.py"
    session, events = await _run(
        tmp_path, [("w-big", "write", {"path": str(target), "content": _big(500)})]
    )
    details = _details(session)[0]
    assert details[DIFF_TRUNCATED_ROWS_KEY] > 0
    desc = details[diff_archive.DIFF_FULL_KEY]
    assert set(desc) == {"v", "state", "rows", "digest"}
    assert desc["v"] == 1 and desc["state"] == "archived" and desc["rows"] == 503
    assert len(desc["digest"]) == 32
    # Live: the end event already carries the same descriptor (no refresh).
    ends = [e for e in events if isinstance(e, ToolExecutionEndEvent)]
    assert ends[0].result.details[diff_archive.DIFF_FULL_KEY] == desc
    rows = _all_rows(tmp_path, "w-big")
    assert len(rows) == 503
    assert rows[:3] == ["--- ", "+++ ", "@@ -0,0 +1,500 @@"]
    assert rows[3] == "+FIRST_SENTINEL = 1"
    assert "+MIDDLE_SENTINEL = 2" in rows
    assert rows[-1] == "+FINAL_SENTINEL = 3"
    # The preview is unchanged in shape: still capped, still a prefix of the whole.
    assert details["diff"] == rows[: len(details["diff"])]
    assert "_diff_source" not in session.transcript.path.read_text()


async def test_real_edit_over_preview_keeps_signs_and_earlier_call_is_immutable(tmp_path):
    target = tmp_path / "e.py"
    original = _big(600)
    target.write_text(original)
    body = "".join(original.splitlines(keepends=True)[1:-1])
    session, _ = await _run(
        tmp_path,
        [
            (
                "e-1",
                "edit",
                {
                    "path": str(target),
                    "old_text": body,
                    "new_text": body.replace(" = ", " := "),
                },
            ),
            ("e-2", "write", {"path": str(target), "content": "short\n"}),
        ],
    )
    first = _details(session)[0]
    assert first.get(diff_archive.DIFF_FULL_KEY, {}).get("state") == "archived", first
    rows = _all_rows(tmp_path, "e-1")
    assert "-MIDDLE_SENTINEL = 2" in rows and "+MIDDLE_SENTINEL := 2" in rows
    # The final changed row and the trailing context both survive the cut.
    assert "+value_598 := 598" in rows and rows[-1] == " FINAL_SENTINEL = 3"
    assert sum(r.startswith("+") for r in rows[2:]) == first["added"]
    assert sum(r.startswith("-") for r in rows[2:]) == first["removed"]
    # A later write AND a direct filesystem mutation leave e-1's evidence alone:
    # nothing reads the current file.
    target.write_text("mutated on disk\n")
    assert _all_rows(tmp_path, "e-1") == rows


@pytest.mark.parametrize("at", [0, 195, 199, 300, 396, 590])
async def test_tall_registered_literal_at_preview_and_page_seams_never_stored(tmp_path, at):
    """A 10-line literal at the preview cut (row ~200), page seams and the end."""
    parts = _literal(10, "P")
    secret = "\n".join(parts)
    lines = [f"line_{index}" for index in range(600)]
    lines[at : at + 10] = parts
    # A duplicate of one literal line ELSEWHERE: the projection must not let
    # either copy through on the strength of the other.
    lines.append(parts[3])
    target = tmp_path / "s.txt"
    session, events = await _run(
        tmp_path,
        [("s-1", "write", {"path": str(target), "content": "\n".join(lines) + "\n"})],
        secrets=(secret,),
    )
    blob = _every_stored_byte(tmp_path, session, events)
    assert [i for i, part in enumerate(parts) if part in blob] == []
    rows = _all_rows(tmp_path, "s-1")
    assert len(rows) == len(lines) + 3  # header pair + one hunk header
    assert all(r[:1] in ("-", "+", " ", "@") for r in rows)


async def test_edit_inside_tall_literal_archive_publishes_no_fragment(tmp_path):
    """The architect's case A, past the preview: edit the literal's middle line."""
    parts = _literal(10, "E")
    pad = [f"filler {index}" for index in range(450)]
    target = tmp_path / "cfg.txt"
    target.write_text("\n".join(pad[:300] + parts + pad[300:]) + "\n")
    edited = target.read_text().replace("filler ", "FILLER ")
    edited = edited.replace(parts[5], parts[5] + "-X")
    session, events = await _run(
        tmp_path,
        [("c-1", "write", {"path": str(target), "content": edited})],
        secrets=("\n".join(parts),),
    )
    blob = _every_stored_byte(tmp_path, session, events)
    assert [i for i, part in enumerate(parts) if part in blob] == []
    assert _details(session)[0][diff_archive.DIFF_FULL_KEY]["state"] == "archived"


@pytest.mark.parametrize("separator", ["\r\n", "\f"])
async def test_value_registered_with_files_own_separator_never_stored(tmp_path, separator):
    parts = _literal(4, "S")
    secret = separator.join(parts)
    lines = [f"row {index}" for index in range(450)]
    lines[320:324] = parts
    target = tmp_path / "sep.txt"
    session, events = await _run(
        tmp_path,
        [("p-1", "write", {"path": str(target), "content": separator.join(lines) + separator})],
        secrets=(secret,),
    )
    blob = _every_stored_byte(tmp_path, session, events)
    assert [i for i, part in enumerate(parts) if part in blob] == []


async def test_private_key_block_past_preview_never_stored(tmp_path):
    body = [f"MIIEv{index:03d}QIBADANBgkqhkiG9w0BAQEFAASC" for index in range(12)]
    # Armour assembled at runtime so this source file never holds a key shape.
    dashes = "-" * 5
    kind = "PRIVATE" + " KEY"
    pem = [f"{dashes}BEGIN {kind}{dashes}", *body, f"{dashes}END {kind}{dashes}"]
    lines = [f"doc {index}" for index in range(400)]
    lines[350:350] = pem
    target = tmp_path / "k.txt"
    session, events = await _run(
        tmp_path, [("k-1", "write", {"path": str(target), "content": "\n".join(lines) + "\n"})]
    )
    blob = _every_stored_byte(tmp_path, session, events)
    assert [row for row in body if row in blob] == []
    state = _details(session)[0][diff_archive.DIFF_FULL_KEY]["state"]
    assert state in {"archived", "withheld"}


async def test_no_redactor_archives_nothing_and_says_unavailable(tmp_path):
    target = tmp_path / "big.py"
    session, _ = await _run(
        tmp_path,
        [("w-1", "write", {"path": str(target), "content": _big(500)})],
        redact=False,
    )
    desc = _details(session)[0][diff_archive.DIFF_FULL_KEY]
    assert desc == {"v": 1, "state": "unavailable"}
    assert not (tmp_path / "config" / diff_archive.DIFF_ARCHIVE_DIRNAME).exists()
    page = diff_archive.load_page(_session_dir(tmp_path), SESSION_ID, "w-1")
    assert page["state"] == "unavailable" and page["rows"] == []


async def test_over_archive_ceiling_is_oversized_with_no_prefix_stored(tmp_path):
    target = tmp_path / "huge.txt"
    content = "".join(f"r{index}\n" for index in range(diff_archive.ARCHIVE_MAX_ROWS))
    session, _ = await _run(tmp_path, [("h-1", "write", {"path": str(target), "content": content})])
    desc = _details(session)[0][diff_archive.DIFF_FULL_KEY]
    assert desc == {"v": 1, "state": "oversized"}
    store = tmp_path / "config" / diff_archive.DIFF_ARCHIVE_DIRNAME
    assert not store.exists() or not any(store.iterdir())


async def test_small_diff_is_inline_and_records_no_descriptor(tmp_path):
    target = tmp_path / "small.txt"
    session, _ = await _run(tmp_path, [("i-1", "write", {"path": str(target), "content": "a\n"})])
    assert diff_archive.DIFF_FULL_KEY not in _details(session)[0]
    page = diff_archive.load_page(_session_dir(tmp_path), SESSION_ID, "i-1")
    assert page["state"] == "inline" and page["rows"] == []


async def test_membership_paging_and_eviction(tmp_path):
    target = tmp_path / "big.py"
    await _run(tmp_path, [("w-big", "write", {"path": str(target), "content": _big(500)})])
    session_dir = _session_dir(tmp_path)
    with pytest.raises(diff_archive.DiffArchiveNotFound):
        diff_archive.load_page(session_dir, SESSION_ID, "nope")
    with pytest.raises(diff_archive.DiffArchiveNotFound):
        # The header binds the artifact to its session.
        diff_archive.load_page(session_dir, "fd0000000009", "w-big")
    for start, limit in ((-1, 10), (0, 0), (0, 201), (503, 10)):
        with pytest.raises(diff_archive.DiffPageRangeError):
            diff_archive.load_page(session_dir, SESSION_ID, "w-big", start=start, limit=limit)
    last = diff_archive.load_page(session_dir, SESSION_ID, "w-big", start=502, limit=1)
    assert last["rows"] == ["+FINAL_SENTINEL = 3"] and last["next_start"] is None
    mid = diff_archive.load_page(session_dir, SESSION_ID, "w-big", start=300, limit=5)
    assert mid["hunk_header"] == "@@ -0,0 +1,500 @@" and mid["next_start"] == 305
    for path in (tmp_path / "config" / diff_archive.DIFF_ARCHIVE_DIRNAME).iterdir():
        path.unlink()
    assert diff_archive.load_page(session_dir, SESSION_ID, "w-big")["state"] == "evicted"


async def test_direct_execute_without_the_seam_keeps_carrier_private(tmp_path):
    """A result that never reaches the loop has no descriptor and no public rows."""
    result = await build_write_tool().execute(
        "x-1",
        {"path": str(tmp_path / "d.txt"), "content": _big(500)},
        None,
        None,
        ToolContext(cwd=str(tmp_path)),
    )
    carried = result._diff_source
    assert carried is not None and isinstance(carried[3], tuple) and len(carried[3]) == 503
    assert diff_archive.DIFF_FULL_KEY not in (result.details or {})
    assert "MIDDLE_SENTINEL" not in result.model_dump_json()
    assert result.model_copy()._diff_source is None


def _fake_row(
    tmp_path: Path,
    call_id: str,
    desc: dict[str, Any],
    tool: str = "write",
    *,
    entry_id: str | None = None,
) -> Path:
    session_dir = _session_dir(tmp_path)
    session_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "role": "tool",
        "tool_call_id": call_id,
        "tool_name": tool,
        "provider_payload": {"details": {"diff_full": desc, DIFF_TRUNCATED_ROWS_KEY: 1}},
    }
    with (session_dir / "transcript.jsonl").open("a") as handle:
        handle.write(
            json.dumps(
                {
                    "id": entry_id or f"m-{call_id}",
                    "ts": 0,
                    "type": "message",
                    "payload": payload,
                }
            )
            + "\n"
        )
    return session_dir


#: Rows that cannot fit one 120 KiB page, each built to stress one encoding
#: worst case: plain ASCII, astral + combining code points, JSON escapes (tab,
#: backslash, quote, NUL and other controls cost up to 6 bytes each).
_HUGE_BODIES = {
    "ascii": "x" * 400_000,
    "astral-combining": ("e\u0301\U0001f600\u00e9" * 30_000),
    "json-escapes": ('\t\\"\x00\x01' * 50_000),
}


@pytest.mark.parametrize("kind", sorted(_HUGE_BODIES))
async def test_huge_rows_are_served_whole_across_pages(tmp_path, kind):
    body = _HUGE_BODIES[kind]
    rows = ["--- ", "+++ ", "@@ -1,2 +1,3 @@", " ctx", "+" + body, "-" + body[::-1], "+" + body]
    desc = diff_archive.write_archive(rows, session_id=SESSION_ID, call_id="j", tool_name="write")
    assert desc["state"] == "archived", desc
    session_dir = _fake_row(tmp_path, "j", desc)
    for limit in (200, 1):
        got, pages = _reconstruct(session_dir, "j", limit=limit)
        assert got == rows, "every stored character is recovered, in order"
        assert any(page["next_offset"] for page in pages), "the huge rows were split"
    # The final row is huge too: the last page ends exactly at the end.
    assert pages[-1]["next_start"] is None and pages[-1]["next_offset"] is None


async def test_continuation_cursor_rejects_offsets_it_never_issued(tmp_path):
    rows = ["--- ", "+++ ", "@@ -0,0 +1,2 @@", "+ab", "+" + "y" * 300_000]
    desc = diff_archive.write_archive(rows, session_id=SESSION_ID, call_id="j", tool_name="write")
    session_dir = _fake_row(tmp_path, "j", desc)
    first = diff_archive.load_page(session_dir, SESSION_ID, "j")
    # A partial final fragment: its row, a positive offset, marker kept.
    assert first["next_start"] == 4 and first["next_offset"] > 0
    assert first["rows"][-1].startswith("+y")
    assert first["start"] + len(first["rows"]) - 1 == first["next_start"]
    resumed = diff_archive.load_page(
        session_dir, SESSION_ID, "j", start=4, offset=first["next_offset"]
    )
    assert resumed["rows"][0] == "+" + "y" * len(resumed["rows"][0][1:])
    assert resumed["hunk_header"] == "@@ -0,0 +1,2 @@"
    for start, offset in ((3, 2), (3, 5), (4, 300_000), (4, -1), (5, 0)):
        with pytest.raises(diff_archive.DiffPageRangeError):
            diff_archive.load_page(session_dir, SESSION_ID, "j", start=start, offset=offset)
    # offset 1 into "+ab" is inside the body: a valid (if unusual) cursor.
    inside = diff_archive.load_page(session_dir, SESSION_ID, "j", start=3, offset=1, limit=1)
    assert inside["rows"] == ["+b"] and inside["next_start"] == 4 and inside["next_offset"] == 0


async def test_unservable_surrogate_rows_are_withheld_not_altered(tmp_path):
    desc = diff_archive.write_archive(
        ["--- ", "+++ ", "@@ -0,0 +1 @@", "+bad\udcff"],
        session_id=SESSION_ID,
        call_id="u",
        tool_name="write",
    )
    assert desc == {"v": 1, "state": "withheld"}


async def test_two_distinct_rows_for_one_call_id_fail_closed(tmp_path):
    desc = diff_archive.write_archive(
        ["--- ", "+++ ", "@@ -0,0 +1 @@", "+x"],
        session_id=SESSION_ID,
        call_id="dup",
        tool_name="write",
    )
    session_dir = _fake_row(tmp_path, "dup", desc)
    assert diff_archive.load_page(session_dir, SESSION_ID, "dup")["rows"] == [
        "--- ",
        "+++ ",
        "@@ -0,0 +1 @@",
        "+x",
    ]
    # The SAME entry written twice is still one row.
    line = (session_dir / "transcript.jsonl").read_text()
    (session_dir / "transcript.jsonl").write_text(line + line)
    assert diff_archive.load_page(session_dir, SESSION_ID, "dup")["state"] == "archived"
    # A DIFFERENT row with the same call id: neither capture is served.
    _fake_row(tmp_path, "dup", desc, entry_id="m-dup-2")
    page = diff_archive.load_page(session_dir, SESSION_ID, "dup")
    assert page["state"] == "unavailable" and page["rows"] == []
    assert page["next_start"] is None and page["next_offset"] is None


async def test_real_session_reusing_a_call_id_never_serves_the_wrong_capture(tmp_path):
    """Two prompts whose provider reuses ONE call id, mutating the same file."""
    target = tmp_path / "same.py"
    calls = [
        ("dup-id", "write", {"path": str(target), "content": _big(500)}),
        None,  # first prompt ends
        ("dup-id", "write", {"path": str(target), "content": _big(450, "!")}),
    ]
    session, _ = await _run(tmp_path, calls)
    first = _all_rows(tmp_path, "dup-id")
    assert "+MIDDLE_SENTINEL = 2" in first
    # Second prompt on the SAME session: the provider reuses the call id.
    await session.prompt("again")
    details = _details(session)
    assert len(details) == 2
    page = diff_archive.load_page(_session_dir(tmp_path), SESSION_ID, "dup-id")
    assert page["state"] == "unavailable" and page["rows"] == []


async def test_non_diff_tool_and_digest_from_another_call_are_not_found(tmp_path):
    desc = diff_archive.write_archive(
        ["--- ", "+++ ", "@@ -0,0 +1 @@", "+x"],
        session_id=SESSION_ID,
        call_id="a",
        tool_name="write",
    )
    session_dir = _fake_row(tmp_path, "b", desc, tool="bash")
    with pytest.raises(diff_archive.DiffArchiveNotFound):
        diff_archive.load_page(session_dir, SESSION_ID, "b")
    # A row naming ANOTHER call's digest (a forged or replayed descriptor)
    # does not serve that call's rows.
    _fake_row(tmp_path, "c", desc)
    with pytest.raises(diff_archive.DiffArchiveNotFound):
        diff_archive.load_page(session_dir, SESSION_ID, "c")
