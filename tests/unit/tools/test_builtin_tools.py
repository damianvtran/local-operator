"""End-to-end tests for the builtin tools against a temp working directory.

Covers the review findings RT-27..RT-32 explicitly: subprocess lifecycle
(abort/timeout/pre-abort), the ToolResult invariant sweep, pydantic
ValidationError containment, truncation shape, unexpected-exception safety,
and range-beyond-EOF.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

from local_operator.harness.types import AbortSignal, AgentTool, ToolContext, ToolResult
from local_operator.tools import builtin
from local_operator.tools.registry import create_tools


@pytest.fixture
def context(tmp_path: Path) -> ToolContext:
    return ToolContext(cwd=str(tmp_path), session_id="unit-test")


@pytest.fixture
def tools(context: ToolContext) -> dict[str, AgentTool]:
    return {tool.name: tool for tool in create_tools(context)}


async def _call(tools: dict, name: str, args: dict, context: ToolContext) -> ToolResult:
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


def _context_with_approval(tmp_path: Path, approve: bool = True) -> ToolContext:
    approval = RecordingApproval(approve)
    context = ToolContext(cwd=str(tmp_path), session_id="unit-test", request_approval=approval)
    # Stash the recorder on the context object for the tests.
    context.recorder = approval  # type: ignore[attr-defined]
    return context


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


@pytest.mark.asyncio
async def test_read_binary_detected(tools, context, tmp_path) -> None:
    (tmp_path / "blob.bin").write_bytes(b"\x00\x01\x02payload")
    result = await _call(tools, "read", {"path": "blob.bin"}, context)
    assert result.is_error is True
    assert "Binary" in result.text


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
        self._schedules: list = []

    @property
    def schedules(self) -> list:
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
_SWEEP_CASES: list[tuple[str, dict]] = [
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
