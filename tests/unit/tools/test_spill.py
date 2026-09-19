"""Tests for the bounded spill store and the truncate-then-expand path.

The properties worth defending here are the ones that fail QUIETLY when they
break:

- the store's total-bytes ceiling actually holds (an unbounded spill directory
  is the exact failure that filled this workstation's disk);
- a live session can still expand its own recent output after other writes;
- a footer names a call that WORKS, so an agent expands instead of re-running;
- error and non-zero-exit text survives truncation in preference to stdout;
- a failed spill degrades to plain truncation instead of failing a tool call.
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from local_operator.harness.types import AgentTool, ToolContext, ToolResult
from local_operator.tools import builtin, spill
from local_operator.tools.registry import create_tools


@pytest.fixture(autouse=True)
def isolated_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the whole app at a tmp config dir.

    Autouse because a test that forgets it writes into the developer's real
    ``~/.local-operator`` — the store honours the override precisely so that
    cannot happen, and a leaked spill file is invisible until it is not.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.delenv(spill.SPILL_MAX_BYTES_ENV, raising=False)
    return tmp_path


@pytest.fixture
def context(tmp_path: Path) -> ToolContext:
    return ToolContext(cwd=str(tmp_path), session_id="spill-test")


@pytest.fixture
def tools(context: ToolContext) -> dict[str, AgentTool]:
    return {tool.name: tool for tool in create_tools(context)}


async def _call(
    tools: dict[str, AgentTool], name: str, args: dict[str, Any], context: ToolContext
) -> ToolResult:
    return await tools[name].execute("call-1", args, None, None, context)  # type: ignore[operator]


def _lines(count: int, prefix: str = "line") -> str:
    return "\n".join(f"{prefix} {i}" for i in range(1, count + 1))


# ---------------------------------------------------------------------------
# handle parsing
# ---------------------------------------------------------------------------


def test_parse_handle_accepts_bare_and_query_forms() -> None:
    digest = "a" * 32
    bare = spill.parse_handle(f"spill://{digest}")
    assert bare is not None and bare.digest == digest and bare.query == ""
    searched = spill.parse_handle(f"spill://{digest}?q=Err.*or")
    assert searched is not None and searched.query == "Err.*or"
    # The bare handle is what a footer must quote, even for a search ref.
    assert searched.handle == f"spill://{digest}"


@pytest.mark.parametrize(
    "bad",
    [
        "skill://demo",  # another scheme must fall through, not be adopted
        "spill://short",
        "spill://" + "g" * 32,  # non-hex
        "spill://" + "a" * 31,
        "/tmp/file.txt",
    ],
)
def test_parse_handle_rejects_non_handles(bad: str) -> None:
    assert spill.parse_handle(bad) is None


# ---------------------------------------------------------------------------
# store: write / read / search
# ---------------------------------------------------------------------------


def test_write_then_read_range_round_trips() -> None:
    store = spill.get_store()
    meta = store.write(_lines(500), tool_name="bash", session_id="s1")
    assert meta is not None
    assert meta.lines == 500 and meta.complete is True

    read = store.read_lines(meta.handle, 10, 12)
    assert read is not None
    selected, total = read
    assert total == 500
    assert selected == ["line 10", "line 11", "line 12"]


def test_write_is_content_addressed_and_idempotent() -> None:
    store = spill.get_store()
    first = store.write("same text\nhere", tool_name="bash", session_id="s1")
    second = store.write("same text\nhere", tool_name="grep", session_id="s2")
    assert first is not None and second is not None
    assert first.handle == second.handle
    # Identical output written twice must cost ONE entry, not two.
    assert store.entry_count() == 1


def test_concurrent_atomic_replaces_use_distinct_temps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Concurrent replaces of one digest cannot rename another writer's temp."""
    workers = 8
    barrier = threading.Barrier(workers)
    real_replace = spill.os.replace

    def synchronized_replace(source, destination) -> None:  # noqa: ANN001
        # Hold every writer after staging but before the rename. A
        # deterministic temp gives all eight the same source; unique temps let
        # every atomic replace complete.
        barrier.wait(timeout=5)
        real_replace(source, destination)

    monkeypatch.setattr(spill.os, "replace", synchronized_replace)
    target = tmp_path / "same.txt"
    payload = b"identical" * 10_000
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(spill._atomic_write_bytes, target, payload) for _ in range(workers)
        ]
        for future in futures:
            future.result(timeout=10)

    assert target.read_bytes() == payload


def test_store_serializes_install_and_evict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Entry installation and its eviction sweep are one store transaction."""
    store = spill.get_store()
    workers = 8
    state_lock = threading.Lock()
    active = 0
    peak = 0
    real_write_entry = spill.SpillStore._write_entry

    def delayed_write_entry(self, digest, data, meta) -> None:  # noqa: ANN001
        nonlocal active, peak
        with state_lock:
            active += 1
            peak = max(peak, active)
        try:
            time.sleep(0.02)
            real_write_entry(self, digest, data, meta)
        finally:
            with state_lock:
                active -= 1

    monkeypatch.setattr(spill.SpillStore, "_write_entry", delayed_write_entry)
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                store.write,
                f"entry {index}\n" + ("x" * 10_000),
                tool_name="bash",
                session_id=f"session-{index}",
            )
            for index in range(workers)
        ]
        metas = [future.result(timeout=10) for future in futures]

    assert all(meta is not None for meta in metas)
    assert peak == 1


def test_atomic_write_cleans_partial_temp_when_write_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A disk-write failure cannot leak an untracked delete=False temp."""
    temp_path = tmp_path / ".forced.tmp"

    class FailingTemp:
        name = str(temp_path)

        def __enter__(self):
            temp_path.write_bytes(b"partial")
            return self

        def __exit__(self, *_args) -> bool:
            return False

        def write(self, _data: bytes) -> None:
            raise OSError("disk full")

    monkeypatch.setattr(
        spill.tempfile,
        "NamedTemporaryFile",
        lambda **_kwargs: FailingTemp(),
    )

    with pytest.raises(OSError, match="disk full"):
        spill._atomic_write_bytes(tmp_path / "target.txt", b"complete")
    assert not temp_path.exists()


def test_stat_returns_metadata_without_content() -> None:
    store = spill.get_store()
    meta = store.write(_lines(30), tool_name="grep", session_id="s1")
    assert meta is not None
    looked_up = store.stat(meta.handle)
    assert looked_up is not None
    assert looked_up.lines == 30
    assert looked_up.tool_name == "grep"
    assert looked_up.session_id == "s1"


def test_stat_of_unknown_handle_is_none_not_an_error() -> None:
    assert spill.get_store().stat("spill://" + "b" * 32) is None


def test_search_reports_line_numbers_and_total() -> None:
    store = spill.get_store()
    text = _lines(200) + "\nTraceback: boom\n" + _lines(50, "tail")
    meta = store.write(text, tool_name="bash", session_id="s1")
    assert meta is not None

    found = store.search(meta.handle, "Traceback")
    assert found is not None
    matches, total_matches, total_lines = found
    assert total_matches == 1
    assert matches[0][0] == 201  # the line number an agent then reads around
    assert "boom" in matches[0][1]
    assert total_lines == 251


def test_search_caps_returned_matches_but_reports_the_true_total() -> None:
    store = spill.get_store()
    meta = store.write(_lines(1000, "match"), tool_name="bash", session_id="s1")
    assert meta is not None
    found = store.search(meta.handle, "match", limit=10)
    assert found is not None
    matches, total_matches, _lines_count = found
    # A pattern matching everything must not reintroduce the unbounded read
    # the store exists to prevent, but must still tell the truth about size.
    assert len(matches) == 10
    assert total_matches == 1000


# ---------------------------------------------------------------------------
# the ceiling — the half omp gets wrong
# ---------------------------------------------------------------------------


def test_total_ceiling_holds_when_written_far_past_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Write ~40x the ceiling and assert the store never exceeds it.
    ceiling = 200_000
    monkeypatch.setenv(spill.SPILL_MAX_BYTES_ENV, str(ceiling))
    store = spill.get_store()

    peak = 0
    for i in range(80):
        payload = f"entry {i}\n" + ("x" * 100_000)
        store.write(payload, tool_name="bash", session_id=f"session-{i}")
        peak = max(peak, store.total_bytes())
        assert store.total_bytes() <= ceiling, f"ceiling breached after write {i}"

    assert peak > 0  # the loop really did store things
    assert store.total_bytes() <= ceiling
    # 8 MB of input must not have left 8 MB on disk.
    assert store.total_bytes() < 80 * 100_000 / 10


def test_ceiling_holds_even_when_one_session_writes_everything(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Session protection must not become a hole in the ceiling: a single
    # session outrunning the whole budget still gets evicted down to it. The
    # grace window is negotiable; the disk is not.
    monkeypatch.setenv(spill.SPILL_MAX_BYTES_ENV, "150000")
    store = spill.get_store()
    for i in range(30):
        store.write(f"same-session {i}\n" + "y" * 60_000, tool_name="bash", session_id="live")
    assert store.total_bytes() <= 150_000


def test_live_session_keeps_its_recent_entry_while_others_are_evicted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The regression this guards: an agent is told "expand spill://X", writes a
    # few more outputs in the same turn, and X is gone when it follows the
    # instruction. Other sessions' entries must be evicted FIRST, so the live
    # session's working set survives as long as the ceiling allows.
    monkeypatch.setenv(spill.SPILL_MAX_BYTES_ENV, "500000")
    store = spill.get_store()

    stale = []
    for i in range(6):
        meta = store.write(f"stale {i}\n" + "z" * 60_000, tool_name="bash", session_id=f"other-{i}")
        assert meta is not None
        stale.append(meta)
    mine = store.write("mine\n" + "m" * 60_000, tool_name="bash", session_id="live")
    assert mine is not None

    # Now push hard from the live session; the live entry must survive while
    # the other sessions' older entries are the ones reclaimed.
    for i in range(4):
        store.write(f"more {i}\n" + "w" * 60_000, tool_name="bash", session_id="live")

    assert store.stat(mine.handle) is not None, "live session lost its own recent handle"
    assert any(store.stat(m.handle) is None for m in stale), "nothing was evicted at all"
    assert store.total_bytes() <= 500_000


def test_per_entry_cap_clips_one_pathological_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Without a per-entry cap, one runaway command evicts the whole store on
    # its way in — the ceiling holds while everything useful disappears.
    monkeypatch.setattr(spill, "SPILL_ENTRY_LIMIT_BYTES", 50_000)
    store = spill.get_store()
    meta = store.write("HEAD\n" + ("q" * 500_000) + "\nTAIL", tool_name="bash", session_id="s1")
    assert meta is not None
    assert meta.bytes <= 50_000
    assert meta.complete is False  # honestly flagged as a partial copy

    read = store.read_lines(meta.handle, 1, None)
    assert read is not None
    text = "\n".join(read[0])
    assert text.startswith("HEAD")  # head kept
    assert text.endswith("TAIL")  # and the tail, where the answer usually is


def test_read_of_an_evicted_handle_explains_itself(
    tools: dict[str, AgentTool], context: ToolContext
) -> None:
    # A bounded store evicts by design, so this is an ordinary outcome and the
    # message has to tell the agent what to do rather than read as a fault.
    import asyncio

    result = asyncio.run(_call(tools, "read", {"path": "spill://" + "c" * 32}, context))
    assert result.is_error is True
    assert "no longer available" in result.text
    assert "Re-run" in result.text


# ---------------------------------------------------------------------------
# degradation — a broken store must never fail a tool call
# ---------------------------------------------------------------------------


def test_unwritable_store_degrades_to_plain_truncation(
    monkeypatch: pytest.MonkeyPatch, context: ToolContext
) -> None:
    def boom(*_args: object, **_kwargs: object) -> None:
        raise OSError("read-only file system")

    monkeypatch.setattr(Path, "mkdir", boom)
    text = _lines(5000)
    body, details = builtin.spill_truncate(text, "bash", context)

    # No handle, but still a valid truncated body with both ends intact.
    assert details is None
    assert body.startswith("line 1\n")
    assert body.rstrip().endswith("line 5000")
    assert len(body) <= builtin.TOOL_OUTPUT_LIMIT_CHARS
    assert builtin.BASH_TRUNCATION_MARKER.strip() in body


def test_write_returns_none_rather_than_raising_on_oserror(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = spill.get_store()
    monkeypatch.setattr(
        spill.SpillStore,
        "_write_entry",
        lambda *_a, **_k: (_ for _ in ()).throw(OSError("disk full")),
    )
    assert store.write("some text", tool_name="bash", session_id="s") is None


def test_corrupt_sidecar_is_invisible_rather_than_fatal() -> None:
    store = spill.get_store()
    meta = store.write(_lines(20), tool_name="bash", session_id="s1")
    assert meta is not None
    (store.root / f"{meta.digest}.json").write_text("{not json", encoding="utf-8")
    assert store.stat(meta.handle) is None  # degraded, not raised


# ---------------------------------------------------------------------------
# truncation shape: head AND tail, errors preferentially
# ---------------------------------------------------------------------------


def test_truncation_keeps_head_and_tail_and_snaps_to_line_boundaries(
    context: ToolContext,
) -> None:
    text = _lines(5000)
    body, details = builtin.spill_truncate(text, "bash", context)
    assert details is not None

    assert body.startswith("line 1\n")
    head, _marker, rest = body.partition(builtin.BASH_TRUNCATION_MARKER.strip())
    # A cut through the middle of a line renders a different, wrong value —
    # 'line 123' truncated to 'line 12' is a plausible-looking lie.
    for line in head.splitlines():
        assert line == "" or line.split()[-1].isdigit()
    assert "line 5000" in rest


def test_footer_names_a_call_that_actually_resolves(
    tools: dict[str, AgentTool], context: ToolContext
) -> None:
    # The whole contract: whatever range the footer prints must work verbatim.
    import asyncio
    import re

    text = _lines(4000)
    body, details = builtin.spill_truncate(text, "bash", context)
    assert details is not None
    handle = details["spill"]["handle"]
    assert handle in body

    match = re.search(r'read\(path="(spill://[0-9a-f]{32})", range="(\d+-\d+)"\)', body)
    assert match, f"footer must print a concrete read() call, got:\n{body[-600:]}"
    assert match.group(1) == handle

    expanded = asyncio.run(
        _call(tools, "read", {"path": match.group(1), "range": match.group(2)}, context)
    )
    assert expanded.is_error is False
    # It returns content the truncated body did NOT contain — the first page
    # of the gap — and it comes back WHOLE, not truncated again.
    first_elided = int(match.group(2).split("-")[0])
    assert f"line {first_elided}" in expanded.text
    assert f"line {first_elided}" not in body
    assert "Continue with" not in expanded.text, "a suggested page must not re-truncate"


# ---------------------------------------------------------------------------
# single-line payloads: JSON results, and no line space to elide in
#
# Every MCP tool result is ONE marshalled JSON string, so the line-based
# elision above has no line to snap to and no line span to state. What it did
# with that shape was splice `[-1 of 1 lines elided — they are lines 2-0 of the
# saved output]` into the middle of the document: a span that cannot exist
# (``splitlines()`` sees one line), inside a payload a caller parses with
# ``json.loads``. Observed on Minerva risk-assessment readbacks on 2026-09-17.
# ---------------------------------------------------------------------------


def _json_payload(records: int = 40) -> str:
    """One marshalled JSON line, the shape every MCP tool result arrives in."""
    return json.dumps(
        {
            "assessment_id": "ra_1",
            "status": "running",
            "notes_markdown": "n" * 4000,
            "tasks": [
                {
                    "task_id": f"task_{i:03d}",
                    "status": "complete",
                    "evidence_ids": [f"ev_{i:03d}"],
                    "resolution_comment": "c" * 400,
                }
                for i in range(records)
            ],
            "evidence_id_index": [
                {"evidence_id": f"ev_{i:03d}", "title": f"Evidence {i}"} for i in range(records)
            ],
        }
    )


def test_single_line_json_result_is_still_parseable(context: ToolContext) -> None:
    text = _json_payload()
    assert "\n" not in text, "this test is about the one-line shape"

    body, details = builtin.spill_truncate(text, "get_assessment", context)
    assert details is not None

    # THE property: a caller can parse the result. raw_decode at the head — the
    # footer is appended after the document and is the only trailing text.
    parsed, end = json.JSONDecoder().raw_decode(body)
    assert end < len(body), "the footer must follow the document, not be spliced into it"
    assert parsed["assessment_id"] == "ra_1"
    # Keys survive: which key gets dropped must not be decided by sort order,
    # because that is how the field a reader wants is the one that goes.
    assert {"status", "notes_markdown", "tasks", "evidence_id_index"} <= set(parsed)

    # A trim says so in band, under this harness's own key. It is deliberately
    # NOT `_truncated`: that key belongs to the Minerva toolproxy, whose shape is
    # `{"_truncated": true, "reason": ...}` and whose signal must survive a
    # harness trim rather than be overwritten by it (review round 1, F4).
    assert '"_elided"' in json.dumps(parsed)
    assert '"_truncated"' not in json.dumps(parsed)

    # The lie is gone: no impossible span, and no call that cannot resolve.
    assert "-1 of" not in body
    assert "lines 2-0" not in body
    assert 'range="' not in body, "a one-line payload has no line range to page to"
    assert details["spill"]["handle"] in body
    assert "?q=<regex>" in body, "the search form is the route that works here"


def test_single_line_non_json_result_gets_no_line_span(context: ToolContext) -> None:
    body, details = builtin.spill_truncate("x" * 40_000, "fetch_page", context)
    assert details is not None
    assert builtin.BASH_TRUNCATION_MARKER.strip() in body
    assert "-1 of" not in body and "lines 2-0" not in body
    assert 'range="' not in body
    assert "?q=<regex>" in body


def test_json_elision_shortens_prose_before_it_drops_structure() -> None:
    value = {
        "status": "complete",
        "prose": "p" * 20_000,
        "tasks": [{"task_id": f"task_{i}", "title": f"T{i}"} for i in range(50)],
    }
    out = builtin._elide_json(json.dumps(value), 2_000)
    assert out is not None
    parsed = json.loads(out)
    assert set(parsed) >= {"status", "prose", "tasks"}
    # A shortened string keeps BOTH ends around a marker that states the count:
    # a head-only cut with a bare `...[truncated]` said nothing about the scale of
    # the loss and destroyed whatever the field concluded with (review round 1,
    # F3) — for a disposition or a finding summary that is the part that matters.
    elided = re.search(r"\.\.\.\[(\d+) of (\d+) chars elided\]\.\.\.", parsed["prose"])
    assert elided, parsed["prose"][:120]
    assert int(elided.group(2)) == 20_000, "the marker states the ORIGINAL length"
    kept_chars_expected = len(parsed["prose"]) - len(elided.group(0))
    assert (
        int(elided.group(1)) == 20_000 - kept_chars_expected
    ), "the count is the chars actually dropped"
    kept_chars = len(parsed["prose"]) - len(elided.group(0))
    assert parsed["prose"].startswith("p" * (kept_chars // 3))
    assert parsed["prose"].endswith("p" * (kept_chars // 3)), "the tail is where a conclusion lives"
    # Every task that is still listed is listed whole; the drop is counted, and
    # the count must reconcile with what was kept.
    kept = parsed["tasks"][:-1]
    assert all(set(task) == {"task_id", "title"} for task in kept)
    marker = parsed["tasks"][-1]["_elided"]
    assert marker["total_items"] == 50
    assert marker["omitted_items"] == 50 - len(kept)


def test_json_elision_never_declines_a_json_payload() -> None:
    # JSON that no rung can fit — short leaves, so there is nothing to shorten,
    # and object keys are never dropped — must still come back as JSON. Declining
    # sent it down the head+tail path, whose marker lands inside the document and
    # hands the caller a payload `json.loads` rejects: the defect this branch
    # exists to remove (review round 1, F1b/F1c).
    for payload in (list(range(5_000)), "a bare string", 12345):
        out = builtin._elide_json(json.dumps(payload), 16)
        assert out is not None
        assert len(out) <= 16 or out == "{}"
        assert json.loads(out)  # or :: it parses

    # Only a payload that is NOT a document shape declines, so the line path
    # keeps its job: prose, logs, and a truncated fragment are readable with a
    # spliced marker, and a document is not.
    assert builtin._elide_json("not json at all", 16) is None
    assert builtin._elide_json("2026-09-17T01:23:45Z starting up\n" * 40, 16) is None


def test_elision_span_declines_a_payload_with_no_interior_line() -> None:
    text = "y" * 40_000
    head, tail = builtin._clip_head_tail(text, 8_000)
    assert builtin._elision_span(text, head, tail) is None
    # ...and still reports a real span when there is one, so the footer's
    # suggested range keeps meaning what it meant before.
    lines = _lines(5_000)
    lhead, ltail = builtin._clip_head_tail(lines, 8_000)
    span = builtin._elision_span(lines, lhead, ltail)
    assert span is not None
    total, first, last = span
    assert total == 5_000 and first <= last


def test_stderr_survives_preferentially_when_the_command_failed() -> None:
    budget = 10_000
    stdout = "o" * 100_000
    stderr = "e" * 100_000

    ok_out, ok_err = builtin._stream_budgets(stdout, stderr, budget, failed=False)
    bad_out, bad_err = builtin._stream_budgets(stdout, stderr, budget, failed=True)

    assert ok_out + ok_err <= budget and bad_out + bad_err <= budget
    # On failure the diagnostic stream gets the larger share; on success it
    # must not be able to crowd stdout out.
    assert bad_err > ok_err
    assert bad_err > bad_out
    assert ok_err <= budget // 2
    # Neither stream is ever budgeted to zero: an empty section reads as "there
    # was no output", which is a different and wrong claim.
    assert ok_out >= 1 and ok_err >= 1 and bad_out >= 1 and bad_err >= 1


def test_small_stderr_is_never_truncated_even_beside_a_huge_stdout() -> None:
    # The common failing-build shape: megabytes of compiler chatter on stdout
    # and the one line that matters on stderr.
    stdout = "o" * 500_000
    stderr = "error: undefined reference to `main'\n"
    out_budget, err_budget = builtin._stream_budgets(stdout, stderr, 8000, failed=True)
    assert err_budget >= len(stderr)
    assert out_budget > 0


@pytest.mark.asyncio
async def test_failing_command_keeps_its_error_text_in_context(
    tools: dict[str, AgentTool], context: ToolContext
) -> None:
    # End to end: a command that floods stdout and then fails must still show
    # the failure. This is the regression that makes aggressive truncation
    # safe — an agent that cannot see the error re-runs the command.
    cmd = (
        'python3 -c "import sys;'
        "sys.stdout.write('noise\\n' * 40000);"
        "sys.stderr.write('FATAL: the real problem\\n');"
        'sys.exit(3)"'
    )
    result = await _call(tools, "bash", {"command": cmd}, context)
    assert "exit code: 3" in result.text
    assert "FATAL: the real problem" in result.text
    assert len(result.text) < builtin.TOOL_OUTPUT_LIMIT_CHARS * 2


@pytest.mark.asyncio
async def test_bash_spill_details_carry_the_handle_and_stay_prunable(
    tools: dict[str, AgentTool], context: ToolContext
) -> None:
    cmd = "python3 -c \"print('x' * 80 + '\\n', end='')\" ; " "python3 -c \"print('y\\n' * 20000)\""
    result = await _call(tools, "bash", {"command": cmd}, context)
    assert result.details is not None
    handle = result.details["spill"]["handle"]
    assert handle.startswith(spill.SPILL_SCHEME)
    # details never reaches a provider, so the handle costs no prompt tokens.
    assert handle not in str(result.details.get("useless", ""))
    assert spill.get_store().stat(handle) is not None


# ---------------------------------------------------------------------------
# expansion through the existing read tool
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_read_expands_a_spill_by_range_with_line_numbers(
    tools: dict[str, AgentTool], context: ToolContext
) -> None:
    meta = spill.get_store().write(_lines(3000), tool_name="bash", session_id="spill-test")
    assert meta is not None
    result = await _call(tools, "read", {"path": meta.handle, "range": "1500-1503"}, context)
    assert result.is_error is False
    assert "1500| line 1500" in result.text
    assert "1503| line 1503" in result.text
    assert "line 1504" not in result.text
    assert "of 3000" in result.text


@pytest.mark.asyncio
async def test_read_searches_within_a_spill(
    tools: dict[str, AgentTool], context: ToolContext
) -> None:
    text = _lines(2000) + "\nAssertionError: expected 3 got 4\n" + _lines(500, "after")
    meta = spill.get_store().write(text, tool_name="bash", session_id="spill-test")
    assert meta is not None

    found = await _call(tools, "read", {"path": f"{meta.handle}?q=AssertionError"}, context)
    assert found.is_error is False
    assert "2001|" in found.text
    assert "expected 3 got 4" in found.text
    # Searching must be cheap — that is the entire reason it exists.
    assert len(found.text) < 2000


@pytest.mark.asyncio
async def test_search_with_no_hits_is_useless_not_an_error(
    tools: dict[str, AgentTool], context: ToolContext
) -> None:
    meta = spill.get_store().write(_lines(100), tool_name="bash", session_id="spill-test")
    assert meta is not None
    result = await _call(tools, "read", {"path": f"{meta.handle}?q=zzz-nope"}, context)
    assert result.is_error is False
    assert result.useless is True
    assert result.details is not None and result.details.get("useless") is True


@pytest.mark.asyncio
async def test_invalid_regex_is_a_correctable_message_not_a_crash(
    tools: dict[str, AgentTool], context: ToolContext
) -> None:
    meta = spill.get_store().write(_lines(10), tool_name="bash", session_id="spill-test")
    assert meta is not None
    result = await _call(tools, "read", {"path": f"{meta.handle}?q=[unclosed"}, context)
    assert result.is_error is True
    assert "invalid regex" in result.text


@pytest.mark.asyncio
async def test_unranged_expansion_is_itself_bounded(
    tools: dict[str, AgentTool], context: ToolContext
) -> None:
    # Expanding "the whole thing" must not undo the truncation that created
    # the handle, or the store becomes a way to blow the context on purpose.
    meta = spill.get_store().write(_lines(50_000), tool_name="bash", session_id="spill-test")
    assert meta is not None
    result = await _call(tools, "read", {"path": meta.handle}, context)
    assert result.is_error is False
    assert len(result.text) <= builtin.READ_OUTPUT_LIMIT_CHARS + 600
    assert "Continue with" in result.text


@pytest.mark.asyncio
async def test_malformed_handle_is_rejected_with_the_expected_shape(
    tools: dict[str, AgentTool], context: ToolContext
) -> None:
    result = await _call(tools, "read", {"path": "spill://not-a-digest"}, context)
    assert result.is_error is True
    assert "Malformed spill handle" in result.text


@pytest.mark.asyncio
async def test_grep_spills_matches_beyond_the_display_cap(
    tools: dict[str, AgentTool], context: ToolContext, tmp_path: Path
) -> None:
    # 'capped at 200' used to be a dead end: match 201 was unreachable without
    # re-running a narrower grep. Now the full list is behind the handle.
    target = tmp_path / "many.txt"
    target.write_text("\n".join(f"needle {i}" for i in range(1, 1201)))
    result = await _call(tools, "grep", {"pattern": "needle", "path": "many.txt"}, context)

    assert result.is_error is False
    assert result.details is not None, "an over-cap grep must publish a handle"
    handle = result.details["spill"]["handle"]

    expanded = await _call(tools, "read", {"path": f"{handle}?q=needle 1150"}, context)
    assert "needle 1150" in expanded.text
    assert "needle 1150" not in result.text  # genuinely beyond the display cap


@pytest.mark.asyncio
async def test_glob_spills_the_full_list_beyond_its_cap(
    tools: dict[str, AgentTool], context: ToolContext, tmp_path: Path
) -> None:
    for i in range(1, 701):
        (tmp_path / f"f{i:04d}.dat").write_text("x")
    result = await _call(tools, "glob", {"pattern": "*.dat"}, context)
    assert result.is_error is False
    assert result.details is not None
    handle = result.details["spill"]["handle"]

    # The tail of a sorted listing is exactly what the count cap used to hide.
    expanded = await _call(tools, "read", {"path": f"{handle}?q=f0699"}, context)
    assert "f0699.dat" in expanded.text
    assert "f0699.dat" not in result.text


@pytest.mark.asyncio
async def test_output_that_fits_gets_no_handle_and_no_footer(
    tools: dict[str, AgentTool], context: ToolContext
) -> None:
    # An expansion hint on output that was never truncated is pure noise the
    # model has to read past on every ordinary call.
    result = await _call(tools, "bash", {"command": "echo hello"}, context)
    assert "hello" in result.text
    assert spill.SPILL_SCHEME not in result.text
    assert result.details is None
    assert spill.get_store().entry_count() == 0


# ---------------------------------------------------------------------------
# compaction interaction
# ---------------------------------------------------------------------------


def test_spilled_results_are_prunable_and_pin_no_bytes() -> None:
    """A spilled result must stay elidable by compaction.

    Skill reads are protected from pruning because a pruned skill gets re-read
    in a loop. A spilled output is the opposite case: the bytes live on disk,
    the handle lives in ``details`` (which never reaches a provider), so
    blanking the content in the transcript loses nothing recoverable and must
    be allowed.
    """
    from local_operator.compaction.pruning import _is_prunable
    from local_operator.harness.types import Message

    message = Message(
        role="tool",
        tool_name="bash",
        content=[],
        provider_payload={"details": {"spill": {"handle": "spill://" + "d" * 32}}},
    )
    assert _is_prunable(message) is True


def test_token_estimator_survives_special_token_literals_in_tool_output() -> None:
    """Tool output is untrusted text and may contain a tokenizer control
    literal. tiktoken raises on those by default, which crashed the estimator
    — and the estimator runs every turn from pruning and the threshold check.
    """
    from local_operator.compaction.tokens import count_text_tokens, estimate_tokens
    from local_operator.harness.types import Message

    evil = "build output\n<|endoftext|>\nmore output"
    assert count_text_tokens(evil) > 0
    assert estimate_tokens(Message.user(evil)) > 0


def test_store_directory_honours_the_config_override(tmp_path: Path) -> None:
    # A spill escaping the override would leave litter nothing cleans up, and
    # would break the promise that an isolated run touches ONE directory.
    expected = Path(os.environ["LOCAL_OPERATOR_CONFIG_DIR"]) / spill.SPILL_DIRNAME
    assert spill.spill_dir() == expected
    meta = spill.get_store().write("x\ny", tool_name="bash", session_id="s")
    assert meta is not None
    assert (expected / f"{meta.digest}.txt").exists()


# ---------------------------------------------------------------------------
# review round 1: the payload shapes where the "the body parses" invariant did
# not hold yet. Each of these was reproduced on the pre-remediation revision as
# a body with the truncation marker spliced into it, which is the exact harm the
# structured branch exists to remove.
# ---------------------------------------------------------------------------


def test_bom_prefixed_json_result_is_still_parseable(context: ToolContext) -> None:
    # `json.loads` rejects a BOM in `str` input, so a BOM'd payload was
    # classified "not JSON" and spliced. Realistic: the BOM survives transport.
    text = "\ufeff" + json.dumps({"assessment_id": "ra_1", "notes_markdown": "n" * 20_000})
    body, details = builtin.spill_truncate(text, "get_assessment", context)
    assert details is not None
    parsed, end = json.JSONDecoder().raw_decode(body)
    assert end < len(body), "the footer must follow the document"
    assert parsed["assessment_id"] == "ra_1"


def test_json_no_rung_can_fit_still_parses(context: ToolContext) -> None:
    # Object KEYS are never shortened (which keys survive is a meaning decision),
    # so enough key bytes exhaust every rung. The answer is an envelope, not a
    # decline that sends the payload to the head+tail path.
    # Five keys of 20,000 characters each: keys are dropped, never shortened, so
    # the smallest rung still carries 60,000 characters of key bytes.
    payload = {("key_%d_" % index) + "k" * 20_000: index for index in range(5)}
    body, details = builtin.spill_truncate(json.dumps(payload), "get_assessment", context)
    assert details is not None
    parsed, end = json.JSONDecoder().raw_decode(body)
    assert end < len(body)
    assert parsed[builtin.ELISION_MARKER_KEY]["reason"] == "too_large"
    assert parsed[builtin.ELISION_MARKER_KEY]["top_level"] == "object"
    assert parsed[builtin.ELISION_MARKER_KEY]["keys"] == 5


def test_top_level_scalar_json_still_parses(context: ToolContext) -> None:
    # A top-level scalar is not a container to trim, so it is either shortened
    # into a shorter scalar or answered with the envelope — both parse. What it
    # must never be is a splice, which is what declining produced.
    body, details = builtin.spill_truncate(json.dumps("s" * 40_000), "fetch_page", context)
    assert details is not None
    parsed, end = json.JSONDecoder().raw_decode(body)
    assert end < len(body)
    assert isinstance(parsed, (str, dict))

    # The envelope is reachable for a scalar too, when even the shortest form of
    # the value exceeds the budget.
    out = builtin._elide_json(json.dumps("s" * 40_000), 20)
    assert out is not None
    marker = json.loads(out)[builtin.ELISION_MARKER_KEY]
    assert marker is True or marker["top_level"] == "scalar", marker


def test_every_read_call_the_chars_footer_prints_resolves(
    tools: dict[str, AgentTool], context: ToolContext
) -> None:
    # The discipline `test_footer_names_a_call_that_actually_resolves` applies to
    # the range branch, applied to EVERY route this branch prints: a footer that
    # names a call which cannot resolve teaches the model the handle is useless.
    import asyncio

    payload = {
        "assessment_id": "ra_1",
        "evidence_id_index": [
            {"evidence_id": f"ev_{index:04d}", "title": f"Evidence {index}"} for index in range(400)
        ],
    }
    body, details = builtin.spill_truncate(json.dumps(payload), "get_assessment", context)
    assert details is not None
    calls = re.findall(
        r'read\(path="(spill://[0-9a-f]{32})(\?q=<regex>)?"(, range="(\d+-\d+)")?\)', body
    )
    assert calls, f"the footer must print at least one route:\n{body[-400:]}"
    for handle, search, _, span in calls:
        args: dict[str, Any] = {"path": f"{handle}?q=ev_0001" if search else handle}
        if span:
            args["range"] = span
        result = asyncio.run(_call(tools, "read", args, context))
        assert result.is_error is False
        assert "beyond the end" not in result.text, f"printed call cannot resolve: {args}"
        assert "range " not in result.text.split("\n")[0], result.text[:200]


def test_multi_line_stored_copy_keeps_the_paging_route(context: ToolContext) -> None:
    # The second route is dropped only where it cannot work. A stored copy with
    # line structure still gets it: an unranged read pages it 200 lines at a time.
    text = json.dumps({"records": [{"n": index} for index in range(400)]}, indent=2)
    body, details = builtin.spill_truncate(text, "bash", context)
    assert details is not None
    assert f'read(path="{details["spill"]["handle"]}")' in body


def test_harness_trim_never_clobbers_the_toolproxy_marker() -> None:
    # The proxy writes `{"_truncated": true, "reason": "max_depth"}`. A harness
    # trim must not overwrite it, or a reader cannot tell which layer elided what
    # or why — and this is the payload family the investigation is about.
    payload: dict[str, Any] = {"_truncated": True, "reason": "max_depth"}
    payload.update({f"field_{index:03d}": "x" * 40 for index in range(300)})
    out = builtin._elide_json(json.dumps(payload), 2_000)
    assert out is not None
    parsed = json.loads(out)
    assert parsed["_truncated"] is True, "the upstream marker survives"
    assert parsed["reason"] == "max_depth", "and so does the reason beside it"
    assert builtin.ELISION_MARKER_KEY in parsed, "and the harness trim is still visible"
    assert '"_truncated": {"' not in out, "the harness never writes the proxy's shape"


def test_over_cap_store_copy_is_not_advertised_as_full(context: ToolContext) -> None:
    # The store caps an entry; when it did, the chars footer still said "SAVED in
    # full" while its own note said the copy was head+tail. A footnote that
    # contradicts itself is worse than no footnote.
    text = json.dumps({"assessment_id": "ra_1", "blob": "x" * (5 * 1024 * 1024)})
    body, details = builtin.spill_truncate(text, "get_assessment", context)
    assert details is not None
    assert details["spill"]["complete"] is False, "the entry must be over the store cap"
    assert "SAVED in full" not in body
    assert "SAVED" in body
    assert "per-entry store cap" in body, "and the note still says so"


def test_non_ascii_payload_keeps_its_characters() -> None:
    # `ensure_ascii=True` turns each non-ASCII character into six, so a CJK
    # payload spent its budget on escapes and served roughly half the content,
    # as `\u65e5\u672c\u8a9e` rather than 日本語.
    text = json.dumps(
        {"records": [{"n": "日本語のデータ" * 30} for _ in range(20)]}, ensure_ascii=False
    )
    out = builtin._elide_json(text, 8_000)
    assert out is not None
    assert len(out) <= 8_000
    assert "日本語" in out, "the reader gets the characters, not their escapes"


# ---------------------------------------------------------------------------
# review round 2: payloads that ARE JSON but that CPython's parser refuses, and
# the small honesty gaps around them
# ---------------------------------------------------------------------------


def _parseable_head(body: str) -> Any:
    """The document at the head of a served body, or a failure that names why."""
    parsed, end = json.JSONDecoder().raw_decode(body.lstrip("\ufeff"))
    assert end > 0
    return parsed


def test_json_the_stdlib_refuses_is_never_spliced(context: ToolContext) -> None:
    # `json.loads` is stricter than JSON: it refuses a number past its
    # int-conversion guard (4,301 digits, CVE-2020-10735), a document nested past
    # the recursion limit, and a document followed by text. All three are JSON a
    # model's own parser accepts, so declining to the head+tail path handed the
    # model the exact signature this module removes (review round 2, F1).
    # A 20,000-deep document cannot be BUILT by json.dumps either (the encoder hits
    # the same recursion limit), so it is spelled out as text.
    shapes = {
        "4301-digit integer": '{\n  "notes": "'
        + "n" * 4_000
        + '",\n  "big": '
        + "1" * 4_301
        + "\n}",
        "40000-digit integer": "1" * 40_000,
        "nesting depth 20000": "[" * 20_000 + "1" + "]" * 20_000,
        "JSON + trailing text": json.dumps({"assessment_id": "ra_1", "notes": "n" * 9_000})
        + "\nnot-json trailing region",
    }

    for name, text in shapes.items():
        body, details = builtin.spill_truncate(text, "get_assessment", context)
        assert details is not None
        parsed = _parseable_head(body)
        assert parsed is not None, name
        assert (
            builtin.BASH_TRUNCATION_MARKER not in body
        ), f"{name}: the marker must never be spliced into a payload that is JSON-shaped"
        # The recovery route is still the footer's, so the model can get the rest.
        assert details["spill"]["handle"] in body


def test_lone_surrogate_escape_never_reaches_the_body_raw(context: ToolContext) -> None:
    # `\ud800` decodes to a lone surrogate. With ensure_ascii=False it would reach
    # the body raw, and every plain UTF-8 write of that body — including the one
    # this agent does when it persists a result — raises UnicodeEncodeError
    # (review round 2, F3).
    text = json.dumps({"a": "\ud800", "blob": "b" * 20_000})
    body, details = builtin.spill_truncate(text, "get_assessment", context)
    assert details is not None
    body.encode("utf-8")  # must not raise
    parsed = _parseable_head(body)
    assert parsed is not None
    assert "blob" in parsed


def test_partial_last_line_is_not_described_as_an_unbroken_copy(
    tools: dict[str, AgentTool], context: ToolContext
) -> None:
    # Withholding the next range is right; calling the copy "one unbroken line"
    # was not, because a multi-line entry reaches that branch whenever the page
    # starts at its last line and that line is longer than the clip (round 2, F2).
    import asyncio

    text = "short first line\n" + "y" * 9_000
    body, details = builtin.spill_truncate(text, "bash", context)
    assert details is not None
    handle = details["spill"]["handle"]
    page = asyncio.run(_call(tools, "read", {"path": handle, "range": "2-2"}, context))
    assert page.is_error is False
    assert "one unbroken line" not in page.text, page.text[-400:]
    assert "only partly shown" in page.text, page.text[-400:]


def test_string_marker_count_is_the_chars_actually_dropped() -> None:
    # The count identity, across the sizes where the marker is a large share of the
    # budget. Review round 3's N1 correction: the marker-only branch (room == 0)
    # always reported `total`, so the round-2 finding about it was wrong and is
    # recorded as such in the code; this grid pins the property that IS load-bearing
    # — dropped == total - kept — instead of the inert branch case.
    for total, string_limit in [(5_000, 20), (1_000, 40), (200, 60), (100_000, 120), (12, 8)]:
        value = "v" * total
        out = builtin._elide_string_middle(value, string_limit)
        elided = re.search(r"\.\.\.\[(\d+) of (\d+) chars elided\]\.\.\.", out)
        assert elided, (total, string_limit, out)
        kept_content = len(out) - len(elided.group(0))
        assert int(elided.group(2)) == total, (total, string_limit)
        assert int(elided.group(1)) == total - kept_content, (total, string_limit, out)


def test_one_line_copy_search_route_says_what_it_returns(context: ToolContext) -> None:
    # Review round 2, N2: for a one-line copy `?q=` is the ONLY route, and it
    # returns that whole line — there is no "read around them" step to budget for.
    payload = json.dumps({"assessment_id": "ra_1", "notes": "n" * 20_000})
    body, details = builtin.spill_truncate(payload, "get_assessment", context)
    assert details is not None
    assert "?q=<regex>" in body
    assert "read around them" not in body, "a one-line copy has no lines to read around"
    assert "the whole line comes back" in body


def test_python_repr_output_is_not_treated_as_a_json_document(context: ToolContext) -> None:
    # `json.JSONDecodeError` is the "not a JSON document" signal; a bare `{`/`[` is
    # not. `eval` hands its printed body straight to spill_truncate, so `print(rows)`
    # on a list or dict of records — Python's repr, single quotes — is the busiest
    # payload this function sees, and treating a leading bracket as evidence of a
    # document collapsed 10-12 KB of readable output to a 167-byte stub with a
    # `not_parseable` reason that was false (review round 3, F1).
    shapes = {
        "list of dicts": repr(
            [{"id": index, "name": f"record {index}", "note": "n" * 200} for index in range(80)]
        ),
        "list of strings": repr([f"row {index} " + "x" * 300 for index in range(100)]),
        "dict": repr({index: "v" * 200 for index in range(120)}),
        "bracketed log": "[INFO] starting\n"
        + "".join(f"[INFO] step {index} done\n" for index in range(700)),
    }
    for name, text in shapes.items():
        assert len(text) > 8_192, name
        body, details = builtin.spill_truncate(text, "eval", context)
        assert details is not None
        assert '"not_parseable"' not in body, f"{name}: readable output is not a refused document"
        # The line path kept readable content from both ends, as it did before the
        # F1 fix went in.
        assert len(body) > 4_000, (name, len(body))
        assert body.lstrip().startswith(text.lstrip()[:20]), (name, body[:80])
        head, _ = builtin._clip_head_tail(text, 4_000 - len(builtin.BASH_TRUNCATION_MARKER))
        assert head[:64] in body, name

    # …and a genuine document still takes the structured branch.
    payload = json.dumps({"assessment_id": "ra_1", "rows": [{"n": "n" * 40} for _ in range(200)]})
    body, _ = builtin.spill_truncate(payload, "eval", context)
    parsed_document, _ = json.JSONDecoder().raw_decode(body.lstrip("\ufeff"))
    assert parsed_document["assessment_id"] == "ra_1"


def test_numeric_scalar_variants_still_get_the_envelope(context: ToolContext) -> None:
    # Review round 3, F2: the old shape test compared the whole head against a
    # numeric charset, so one trailing space or newline defeated it and the payload
    # was spliced into the number. The exception TYPE does not care about
    # whitespace — both variants raise the plain ValueError — so both are closed.
    for suffix in ("", "\n", " "):
        text = "1" * 9_000 + suffix
        body, details = builtin.spill_truncate(text, "eval", context)
        assert details is not None
        assert builtin.BASH_TRUNCATION_MARKER not in body, repr(suffix)
        parsed = _parseable_head(body)  # the footer follows the document
        assert parsed[builtin.ELISION_MARKER_KEY]["reason"] == "not_parseable", repr(suffix)


def test_document_trailer_is_kept_not_dropped(context: ToolContext) -> None:
    # Review round 3, F3: the text after a parsed document is outside it, so it can
    # be kept losslessly rather than "elided out" — dropping it silently was the
    # failure mode, because the only signal was an aggregate char count a reader
    # attributes to the elided JSON fields.
    document = json.dumps({"assessment_id": "ra_1", "notes": "n" * 9_000})
    body, details = builtin.spill_truncate(
        document + "\nTRAILER-TEXT-THAT-MATTERS", "eval", context
    )
    assert details is not None
    assert "TRAILER-TEXT-THAT-MATTERS" in body
    # …and the document before it is still a clean parseable prefix.
    parsed, end = json.JSONDecoder().raw_decode(body.lstrip("\ufeff"))
    assert end < len(body)
    assert parsed["assessment_id"] == "ra_1"
