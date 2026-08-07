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

import os
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
