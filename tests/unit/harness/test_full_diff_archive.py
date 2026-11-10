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


def _provider(calls: list[tuple[str, str, dict[str, Any]]]):
    requests: list[Any] = []

    def provider(request, signal=None):
        index = len(requests)
        requests.append(request)

        async def generate():
            if index < len(calls):
                call_id, name, args = calls[index]
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
    calls: list[tuple[str, str, dict[str, Any]]],
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


def _all_rows(tmp_path: Path, call_id: str) -> list[str]:
    rows: list[str] = []
    start: int | None = 0
    while start is not None:
        page = diff_archive.load_page(_session_dir(tmp_path), SESSION_ID, call_id, start=start)
        assert page["state"] == "archived", page
        assert page["start"] == start
        assert page["clipped_rows"] == []
        rows.extend(page["rows"])
        assert page["next_start"] in (None, start + len(page["rows"]))
        start = page["next_start"]
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


def _fake_row(tmp_path: Path, call_id: str, desc: dict[str, Any], tool: str = "write") -> Path:
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
            json.dumps({"id": f"m-{call_id}", "ts": 0, "type": "message", "payload": payload})
            + "\n"
        )
    return session_dir


async def test_page_byte_budget_counts_json_escaping_and_clips_one_huge_row(tmp_path):
    worst = "+" + "\t\\\u00e9\U0001f600" * 20000  # every char escapes on the wire
    rows = ["--- ", "+++ ", "@@ -0,0 +1,3 @@", "+a", worst, "+z"]
    desc = diff_archive.write_archive(rows, session_id=SESSION_ID, call_id="j", tool_name="write")
    session_dir = _fake_row(tmp_path, "j", desc)
    first = diff_archive.load_page(session_dir, SESSION_ID, "j")
    assert first["rows"] == rows[:4] and first["next_start"] == 4
    huge = diff_archive.load_page(session_dir, SESSION_ID, "j", start=4)
    assert huge["clipped_rows"] == [4] and huge["rows"][0].startswith("+")
    assert len(json.dumps(huge["rows"][0])) <= diff_archive.PAGE_MAX_BYTES
    assert worst.startswith(huge["rows"][0])
    assert huge["next_start"] == 5 and huge["hunk_header"] == "@@ -0,0 +1,3 @@"
    tail = diff_archive.load_page(session_dir, SESSION_ID, "j", start=5)
    assert tail["rows"] == ["+z"] and tail["next_start"] is None


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
