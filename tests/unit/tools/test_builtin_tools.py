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
import re
import struct
import subprocess
import threading
import time
import zlib
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from local_operator import imaging
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


@pytest.fixture(autouse=True)
def _clean_todo_store():
    """The todo store is MODULE state keyed by session id, so a list left behind
    here is visible to every later test in the process — including a session
    test whose turn the continuation guardrail would then refuse to end."""
    yield
    builtin.TODO_STORE.pop("unit-test", None)


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
async def test_bash_background_returns_a_job_id_without_waiting(tmp_path) -> None:
    """`background=True` hands the command to a job instead of blocking a turn.

    The command sleeps far longer than this call may take, so returning fast
    is only possible if it really was detached rather than awaited.
    """
    from local_operator.harness.jobs import AsyncJobManager

    manager = AsyncJobManager()
    ctx = ToolContext(cwd=str(tmp_path), session_id="bgbash", jobs=manager)
    tool = builtin.build_bash_tool()
    loop = asyncio.get_running_loop()
    started = loop.time()
    result = await tool.execute(  # type: ignore[operator]
        "c1",
        {"command": "echo starting; sleep 30", "background": True, "timeout": 120},
        None,
        None,
        ctx,
    )
    elapsed = loop.time() - started
    assert result.is_error is False
    assert elapsed < 5.0, f"background bash blocked for {elapsed:.1f}s"
    details = result.details or {}
    job_id = str(details["job_id"])
    assert details["backgrounded"] is True

    # Output reaches the peek buffer while the command is still running.
    deadline = loop.time() + 10
    seen = ""
    while loop.time() < deadline:
        window = manager.read_output(job_id)
        assert window is not None
        seen = window[0]
        if "starting" in seen:
            break
        await asyncio.sleep(0.05)
    assert "starting" in seen
    running = manager.get(job_id)
    assert running is not None and running.status == "running"

    await manager.cancel(job_id)
    await manager.dispose()


@pytest.mark.asyncio
async def test_bash_background_cancelled_before_start_kills_the_process_group(tmp_path) -> None:
    """Cancel with ZERO intervening awaits must still kill the child.

    `register` only schedules the runner, so a cancel in the same event-loop
    turn never enters `_detached` and never reaches its cleanup — the process
    was spawned before `register` and would survive, reparented to init. The
    existing process-group test misses this because it sleeps before
    cancelling, which lets the runner start.
    """
    from local_operator.harness.jobs import AsyncJobManager

    manager = AsyncJobManager()
    ctx = ToolContext(cwd=str(tmp_path), session_id="bgleak", jobs=manager)
    tool = builtin.build_bash_tool()
    marker = f"sleep {random.randint(700, 899)}"
    result = await tool.execute(  # type: ignore[operator]
        "c1", {"command": marker, "background": True, "timeout": 900}, None, None, ctx
    )
    job_id = str((result.details or {})["job_id"])

    def _alive() -> int:
        found = subprocess.run(["pgrep", "-f", marker], capture_output=True, text=True)
        return len([pid for pid in found.stdout.split() if pid])

    assert _alive() >= 1, "precondition: the command should be running"
    await manager.cancel(job_id)  # no awaits in between
    await asyncio.sleep(1.0)
    assert _alive() == 0, "process survived a cancel that landed before the runner started"
    await manager.dispose()


@pytest.mark.asyncio
async def test_bash_background_survives_dispose_without_leaking(tmp_path) -> None:
    """Session teardown right after backgrounding must not orphan the child."""
    from local_operator.harness.jobs import AsyncJobManager

    manager = AsyncJobManager()
    ctx = ToolContext(cwd=str(tmp_path), session_id="bgdisp", jobs=manager)
    tool = builtin.build_bash_tool()
    marker = f"sleep {random.randint(900, 1099)}"
    await tool.execute(  # type: ignore[operator]
        "c1", {"command": marker, "background": True, "timeout": 1200}, None, None, ctx
    )
    await manager.dispose()  # no awaits in between
    await asyncio.sleep(1.0)
    found = subprocess.run(["pgrep", "-f", marker], capture_output=True, text=True)
    assert [pid for pid in found.stdout.split() if pid] == [], "dispose orphaned the child"


@pytest.mark.asyncio
async def test_bash_background_without_a_job_manager_is_refused(tools, context) -> None:
    """Refused, not silently run in the foreground: a caller that asked not to
    block must never be handed a call that blocks for the whole timeout."""
    result = await _call(tools, "bash", {"command": "echo hi", "background": True}, context)
    assert result.is_error is True
    assert "job manager" in result.text


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


# -- interpreter resolution (#629) ------------------------------------------
#
# The tool is NAMED bash, so models write bash; /bin/sh on macOS is bash 3.2 in
# POSIX mode and rejects `<(...)` at parse time. These pin the resolution order
# and prove the real spawn path runs bash syntax.


def test_resolve_bash_shell_prefers_the_configured_value(monkeypatch) -> None:
    monkeypatch.setattr(builtin.shutil, "which", lambda name: "/path/which/bash")
    assert builtin.resolve_bash_shell("/opt/custom/bash") == "/opt/custom/bash"
    # Surrounding whitespace is a typo, not an interpreter.
    assert builtin.resolve_bash_shell("  /opt/custom/bash \n") == "/opt/custom/bash"


def test_resolve_bash_shell_falls_back_to_bash_on_path(monkeypatch) -> None:
    monkeypatch.setattr(builtin.shutil, "which", lambda name: "/path/which/bash")
    assert builtin.resolve_bash_shell(None) == "/path/which/bash"
    # Empty and blank both mean "unset": the registry's empty_unsets row and a
    # hand-edited `shell: "  "` must resolve the same way.
    assert builtin.resolve_bash_shell("") == "/path/which/bash"
    assert builtin.resolve_bash_shell("   ") == "/path/which/bash"


def test_resolve_bash_shell_last_resort_is_sh(monkeypatch) -> None:
    monkeypatch.setattr(builtin.shutil, "which", lambda name: None)
    assert builtin.resolve_bash_shell(None) == builtin.BASH_SHELL_FALLBACK == "/bin/sh"


def test_resolve_bash_shell_expands_a_tilde(monkeypatch) -> None:
    """`~/bin/bash` is a plausible thing to type into the Kind.TEXT settings
    row, and nothing else on the read path expands it — unexpanded it would
    reach execve verbatim and fail every call."""
    monkeypatch.setenv("HOME", "/home/tester")
    assert builtin.resolve_bash_shell("~/bin/bash") == "/home/tester/bin/bash"
    assert builtin.resolve_bash_shell("  ~/bin/bash  ") == "/home/tester/bin/bash"


def _set_configured_shell(monkeypatch, tmp_path, value: str):
    """Point `bash.shell` at `value` in an isolated config dir."""
    from local_operator.config import ConfigManager

    config_home = tmp_path / "config"
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config_home))
    ConfigManager(config_home).set_config_value("bash", {"shell": value})


@pytest.mark.asyncio
async def test_bash_reports_a_missing_configured_shell_as_a_tool_error(
    tools, context, tmp_path, monkeypatch
) -> None:
    """A `bash.shell` that does not exist must be a normal error result naming
    the path and the key — not the generic boundary's `failed unexpectedly`
    plus a traceback tail, which never said which key broke every call."""
    _set_configured_shell(monkeypatch, tmp_path, "/nonexistent/qa-shell")

    result = await _call(tools, "bash", {"command": "echo hi"}, context)

    assert result.is_error is True
    assert "bash.shell" in result.text
    assert "/nonexistent/qa-shell" in result.text
    assert "failed unexpectedly" not in result.text


@pytest.mark.asyncio
async def test_bash_reports_a_non_executable_configured_shell_as_a_tool_error(
    tools, context, tmp_path, monkeypatch
) -> None:
    """The same misconfiguration arrives as PermissionError rather than
    FileNotFoundError when the path exists but has no exec bit, which is why
    the handler catches OSError rather than one subclass."""
    not_executable = tmp_path / "not-executable"
    not_executable.write_text("#!/bin/sh\necho nope\n")
    not_executable.chmod(0o644)
    _set_configured_shell(monkeypatch, tmp_path, str(not_executable))

    result = await _call(tools, "bash", {"command": "echo hi"}, context)

    assert result.is_error is True
    assert "bash.shell" in result.text
    assert str(not_executable) in result.text
    # `Traceback` cannot discriminate here: the boundary emits only the last
    # 2000 chars of the formatted traceback, which cuts the header off, so the
    # literal string is absent from the pre-fix output too.
    assert "failed unexpectedly" not in result.text


@pytest.mark.asyncio
async def test_bash_blames_the_working_directory_not_bash_shell(tmp_path, monkeypatch) -> None:
    """`create_subprocess_exec` raises the same OSError family for a deleted
    `cwd=` as for a bad argv[0], so the handler must discriminate on
    `exc.filename`. With `bash.shell` unset and the session directory gone —
    an ordinary condition after a removed worktree or a reclaimed tmpdir — the
    error must name the directory and never mention a key the operator never
    set, whose two prescribed fixes cannot repair a missing directory."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "empty-config"))
    gone = tmp_path / "deleted-cwd"
    gone.mkdir()
    gone_context = ToolContext(cwd=str(gone), session_id="unit-test")
    gone_tools = {tool.name: tool for tool in create_tools(gone_context)}
    gone.rmdir()

    result = await _call(gone_tools, "bash", {"command": "echo hi"}, gone_context)

    assert result.is_error is True
    assert str(gone) in result.text
    assert "bash.shell" not in result.text
    assert "failed unexpectedly" not in result.text


_HOST_BASH = builtin.shutil.which("bash")


@pytest.mark.asyncio
@pytest.mark.skipif(_HOST_BASH is None, reason="no bash on this host")
async def test_bash_runs_process_substitution(tools, context) -> None:
    """The issue's own reproduction: `comm -12 <(echo a) <(echo a)` must print
    `a` and exit 0 under the default (auto-resolved) interpreter."""
    result = await _call(tools, "bash", {"command": "comm -12 <(echo a) <(echo a)"}, context)
    assert result.is_error is False
    assert "exit code: 0" in result.text
    assert "--- stdout ---\na\n" in result.text
    assert "syntax error" not in result.text


@pytest.mark.asyncio
async def test_bash_honours_configured_shell(tools, context, tmp_path, monkeypatch) -> None:
    """`bash.shell` from config is what gets spawned, read at CALL time.

    Pointed at a tiny script rather than /bin/sh so the assertion is a
    positive marker rather than "process substitution failed", which would
    also be true of a broken default."""
    from local_operator.config import ConfigManager

    marker = tmp_path / "marker-shell"
    marker.write_text('#!/bin/sh\necho MARKER-SHELL "$@"\n')
    marker.chmod(0o755)
    config_home = tmp_path / "config"
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config_home))
    ConfigManager(config_home).set_config_value("bash", {"shell": str(marker)})

    result = await _call(tools, "bash", {"command": "ignored"}, context)
    assert result.is_error is False
    assert "MARKER-SHELL -c ignored" in result.text

    # Clearing the key on disk takes effect on the very next call: no cache.
    ConfigManager(config_home).set_config_value("bash", {"shell": ""})
    result = await _call(tools, "bash", {"command": "echo plain"}, context)
    assert "MARKER-SHELL" not in result.text
    assert "plain" in result.text


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
    image.save(path, format="PNG", compress_level=3)
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
    # so anything above the ingest cap is upload the model never benefits from.
    # The cap is IMAGE_INGEST_MAX_EDGE (1024), below the 1568 correctness
    # ceiling: the band between them is billed pixel area that measured no
    # legibility gain on document content (see imaging.IMAGE_INGEST_MAX_EDGE).
    _write_png(tmp_path / "wide.png", (3000, 1500))
    result = await _call(tools, "read", {"path": "wide.png"}, context)

    assert result.is_error is False
    (image,) = _image_blocks(result)
    with Image.open(io.BytesIO(base64.b64decode(image.data))) as delivered:
        assert max(delivered.size) == builtin.READ_IMAGE_MAX_EDGE
        assert delivered.size == (1024, 512)
    # What the model sees and what is on disk now differ; the caption must say
    # so or a later `ls -l` looks like it contradicts the read.
    assert "1024x512" in result.text
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
        original.resize((1024, 768), Image.Resampling.LANCZOS).save(lossless, format="PNG")
    assert len(base64.b64decode(image.data)) < len(lossless.getvalue())
    # base64 inflates by 4/3 and Anthropic rejects an image block over 5 MB.
    assert len(image.data) < 5_000_000


@pytest.mark.asyncio
async def test_read_stays_inside_the_byte_budget_on_incompressible_input(
    tools, context, tmp_path
) -> None:
    # The PNG-wins-over-budget rung (an image PNG compresses better than JPEG
    # AND that still blows IMAGE_MAX_BYTES) is no longer reachable through
    # `read`: bounding to IMAGE_INGEST_MAX_EDGE caps the delivered area at
    # 1024x1024, where the rung's two conditions never hold together: PNG is
    # never BOTH the smaller encode AND over the budget. Figures deliberately
    # not restated (they were wrong three times); run
    # scripts/measure_ingest_lossy_rung.py for the sweep.
    # It is still live on the REPAIR path at 1568 and is covered there by
    # tests/unit/test_imaging.py::test_a_repair_keeps_png_when_jpeg_would_be_bigger.
    #
    # What `read` must still guarantee is this: the hardest-to-compress input
    # lands inside the budget rather than riding to the provider at several MB.
    _write_png(tmp_path / "sharp.png", (1568, 1176), noise="sharp", colours=8)
    result = await _call(tools, "read", {"path": "sharp.png"}, context)

    assert result.is_error is False
    (image,) = _image_blocks(result)
    assert len(base64.b64decode(image.data)) <= builtin.READ_IMAGE_MAX_BYTES
    with Image.open(io.BytesIO(base64.b64decode(image.data))) as delivered:
        assert max(delivered.size) <= builtin.READ_IMAGE_MAX_EDGE


@pytest.mark.asyncio
async def test_read_image_without_pillow_is_forwarded_verbatim(
    tools, context, tmp_path, monkeypatch
) -> None:
    # Pillow reaches a default install only as a pillow-heif dependency, and
    # that is the most platform-fragile wheel here. With no decoder there is
    # no resize and no validation, but a screenshot the model can look at
    # still beats a paragraph explaining why it cannot.
    source = _write_png(tmp_path / "shot.png", (320, 200))
    monkeypatch.setattr(imaging, "pillow_image_module", lambda: None)
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
    monkeypatch.setattr(imaging, "pillow_image_module", lambda: None)
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
    monkeypatch.setattr(imaging, "heif_image_module", lambda: None)
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
async def test_read_missing_skill_path_suggests_protocol(tools, context, tmp_path) -> None:
    result = await _call(
        tools,
        "read",
        {"path": "/tmp/custom/skills/team-workflow/SKILL.md"},
        context,
    )
    assert result.is_error is True
    assert "Path does not exist" in result.text
    assert "Skills are virtual resources loaded via `skill://`" in result.text
    assert "`skill://team-workflow`" in result.text


@pytest.mark.asyncio
async def test_read_missing_plain_path_keeps_plain_error(tools, context, tmp_path) -> None:
    result = await _call(
        tools,
        "read",
        {"path": "nonexistent.txt"},
        context,
    )
    assert result.is_error is True
    assert "Path does not exist" in result.text
    assert "skill://" not in result.text


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
        assert "skill://" not in result.text


@pytest.mark.asyncio
async def test_glob_redirects_absolute_skill_search_to_protocol(tools, context) -> None:
    result = await _call(
        tools,
        "glob",
        {"pattern": "/Users/example/**/minerva-support-workspace/SKILL.md"},
        context,
    )

    assert result.is_error is True
    assert "relative" in result.text.lower()
    assert "Do not scan the filesystem for SKILL.md" in result.text
    assert "`skill://minerva-support-workspace`" in result.text


@pytest.mark.asyncio
async def test_glob_does_not_redirect_skill_md_suffix(tools, context) -> None:
    result = await _call(
        tools,
        "glob",
        {"pattern": "/tmp/catalog/README-NOTSKILL.md"},
        context,
    )

    assert result.is_error is True
    assert "relative" in result.text.lower()
    assert "skill://" not in result.text
    assert "Do not scan" not in result.text


@pytest.mark.asyncio
async def test_glob_uses_placeholder_for_unsafe_skill_name(tools, context) -> None:
    result = await _call(
        tools,
        "glob",
        {"pattern": "/Users/example/**/unsafe name/SKILL.md"},
        context,
    )

    assert result.is_error is True
    assert "`skill://<name>`" in result.text
    assert "unsafe name" not in result.text


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


@pytest.mark.asyncio
async def test_todo_done_marks_every_named_item(tools, context) -> None:
    """Regression: ``done`` used to honour ``items[0]`` and silently ignore the
    rest, so a model closing three items watched two of them stay open."""
    await _call(tools, "todo", {"op": "init", "items": ["a", "b", "c"]}, context)

    done = await _call(tools, "todo", {"op": "done", "items": ["a", "b", "c"]}, context)

    assert done.is_error is False
    assert "3/3" in done.text
    view = await _call(tools, "todo", {"op": "view"}, context)
    assert "[ ]" not in view.text
    assert builtin.open_todos("unit-test") == []


@pytest.mark.asyncio
async def test_todo_done_partial_match_keeps_hits_and_names_misses(tools, context) -> None:
    """The error has to be self-correcting: which text missed, and what is open."""
    await _call(tools, "todo", {"op": "init", "items": ["a", "b"]}, context)

    result = await _call(tools, "todo", {"op": "done", "items": ["a", "ghost"]}, context)

    assert result.is_error is True
    assert "'ghost'" in result.text
    assert "- [ ] b" in result.text  # the open item, so no second `view` call
    # The hit is NOT rolled back: real progress must survive a mistyped sibling.
    assert [item["text"] for item in builtin.open_todos("unit-test")] == ["b"]


@pytest.mark.asyncio
async def test_todo_add_appends_and_skips_duplicates(tools, context) -> None:
    """``add`` is the op the guardrail exists for: a requirement arriving
    mid-turn is recordable without rewriting the list, and a retry is a no-op."""
    await _call(tools, "todo", {"op": "init", "items": ["a"]}, context)

    added = await _call(tools, "todo", {"op": "add", "items": ["b", "c"]}, context)
    again = await _call(tools, "todo", {"op": "add", "items": ["b"]}, context)

    assert added.is_error is False and "2 item(s)" in added.text
    assert again.is_error is False and "nothing added" in again.text
    assert [item["text"] for item in builtin.open_todos("unit-test")] == ["a", "b", "c"]


@pytest.mark.asyncio
async def test_todo_add_without_items_is_error(tools, context) -> None:
    result = await _call(tools, "todo", {"op": "add", "items": []}, context)
    assert result.is_error is True


@pytest.mark.asyncio
async def test_todo_block_requires_a_reason(tools, context) -> None:
    """A blocked item with no reason is indistinguishable from abandoned work."""
    await _call(tools, "todo", {"op": "init", "items": ["ship it"]}, context)

    bare = await _call(tools, "todo", {"op": "block", "items": ["ship it"]}, context)
    with_reason = await _call(
        tools,
        "todo",
        {"op": "block", "items": ["ship it"], "reason": "needs the user's call on the domain"},
        context,
    )

    assert bare.is_error is True and "reason" in bare.text
    assert with_reason.is_error is False
    # Blocked is NOT open for the guardrail — that is what makes it an honest
    # stop rather than a way to end a turn on unfinished work.
    assert builtin.open_todos("unit-test") == []
    # A flat init lives in the single implicit "Todos" phase; walk into it.
    assert (
        builtin.TODO_STORE["unit-test"][0]["items"][0]["reason"]
        == "needs the user's call on the domain"
    )


@pytest.mark.asyncio
async def test_todo_drop_abandons_without_claiming_completion(tools, context) -> None:
    await _call(tools, "todo", {"op": "init", "items": ["a", "b"]}, context)

    dropped = await _call(tools, "todo", {"op": "drop", "items": ["b"]}, context)

    assert dropped.is_error is False
    assert builtin.TODO_STORE["unit-test"][0]["items"][1]["status"] == "dropped"
    assert [item["text"] for item in builtin.open_todos("unit-test")] == ["a"]


@pytest.mark.asyncio
async def test_todo_view_renders_all_four_statuses(tools, context) -> None:
    """The panel renders the same four marks; a status the view cannot spell
    would leave the model and the user reading different lists."""
    await _call(tools, "todo", {"op": "init", "items": ["p", "d", "b", "x"]}, context)
    await _call(tools, "todo", {"op": "done", "items": ["d"]}, context)
    await _call(
        tools, "todo", {"op": "block", "items": ["b"], "reason": "waiting on legal"}, context
    )
    await _call(tools, "todo", {"op": "drop", "items": ["x"]}, context)

    view = await _call(tools, "todo", {"op": "view"}, context)

    assert view.text.splitlines() == [
        "- [ ] p",
        "- [x] d",
        "- [~] b — blocked: waiting on legal",
        "- [-] x",
    ]


@pytest.mark.asyncio
async def test_open_todos_is_pending_only_and_copies(tools, context) -> None:
    """The guardrail's single definition of "open", and it hands out copies:
    the ops mutate item dicts in place."""
    await _call(tools, "todo", {"op": "init", "items": ["a", "b"]}, context)
    await _call(tools, "todo", {"op": "done", "items": ["a"]}, context)

    snapshot = builtin.open_todos("unit-test")
    snapshot[0]["text"] = "mutated"

    assert builtin.open_todos("unit-test") == [{"text": "b", "status": "pending"}]
    assert builtin.open_todos("no-such-session") == []


@pytest.mark.asyncio
async def test_todo_fingerprint_tracks_the_whole_list(tools, context) -> None:
    """The latch compares the FULL list: a change among settled items is still
    a change, and the pending subsequence alone cannot see it."""
    await _call(tools, "todo", {"op": "init", "items": ["a", "b"]}, context)
    await _call(tools, "todo", {"op": "done", "items": ["a"]}, context)
    before = builtin.todo_fingerprint("unit-test")

    await _call(tools, "todo", {"op": "drop", "items": ["a"]}, context)

    # The fingerprint is now a 3-tuple carrying phase identity (design §5.2); a
    # flat init reports the implicit "Todos" phase name for every item.
    assert before == (("Todos", "a", "done"), ("Todos", "b", "pending"))
    assert builtin.todo_fingerprint("unit-test") == (
        ("Todos", "a", "dropped"),
        ("Todos", "b", "pending"),
    )
    assert builtin.todo_fingerprint("no-such-session") == ()


# --- phased todos -----------------------------------------------------------


@pytest.mark.asyncio
async def test_todo_phased_init_builds_phases(tools, context) -> None:
    """A phased ``init`` writes one group per phase, its texts becoming the same
    pending item dicts every other op already handles."""
    result = await _call(
        tools,
        "todo",
        {
            "op": "init",
            "phases": [
                {"phase": "Foundation", "items": ["scaffold", "wire config"]},
                {"phase": "Verification", "items": ["run gate"]},
            ],
        },
        context,
    )

    assert result.is_error is False and "3 item(s) across 2 phase(s)" in result.text
    store = builtin.TODO_STORE["unit-test"]
    assert [phase["name"] for phase in store] == ["Foundation", "Verification"]
    assert [item["text"] for item in store[0]["items"]] == ["scaffold", "wire config"]


@pytest.mark.asyncio
async def test_todo_flat_init_is_one_implicit_phase(tools, context) -> None:
    """The back-compat lever: a flat init lives in one implicit \"Todos\" phase,
    so an existing caller sees the identical list."""
    await _call(tools, "todo", {"op": "init", "items": ["a", "b"]}, context)

    store = builtin.TODO_STORE["unit-test"]
    assert len(store) == 1
    assert store[0]["name"] == "Todos"
    assert [item["text"] for item in store[0]["items"]] == ["a", "b"]


@pytest.mark.asyncio
async def test_todo_init_rejects_both_phases_and_items(tools, context) -> None:
    """``phases`` and flat ``items`` are two spellings of one list; accepting
    both is ambiguous about which wins, so it must fail loud."""
    result = await _call(
        tools,
        "todo",
        {"op": "init", "phases": [{"phase": "A", "items": ["x"]}], "items": ["y"]},
        context,
    )

    assert result.is_error is True and "not both" in result.text


@pytest.mark.asyncio
async def test_todo_add_into_new_and_existing_and_implicit_phase(tools, context) -> None:
    """``add`` appends into the named phase, lazily creating it; with no phase
    it targets the implicit \"Todos\"; dedupe is WITHIN the target phase."""
    await _call(tools, "todo", {"op": "init", "items": ["a"]}, context)

    into_new = await _call(tools, "todo", {"op": "add", "items": ["x"], "phase": "Extra"}, context)
    into_existing = await _call(
        tools, "todo", {"op": "add", "items": ["y"], "phase": "Extra"}, context
    )
    into_implicit = await _call(tools, "todo", {"op": "add", "items": ["b"]}, context)

    assert into_new.is_error is False and "1 item(s)" in into_new.text
    assert into_existing.is_error is False
    assert into_implicit.is_error is False
    store = builtin.TODO_STORE["unit-test"]
    assert [phase["name"] for phase in store] == ["Todos", "Extra"]
    assert [item["text"] for item in store[0]["items"]] == ["a", "b"]
    assert [item["text"] for item in store[1]["items"]] == ["x", "y"]


@pytest.mark.asyncio
async def test_todo_add_dedupe_is_per_phase(tools, context) -> None:
    """The same text pending in another phase is a different task; dedupe must
    not collapse across phases and silently drop it."""
    await _call(
        tools,
        "todo",
        {"op": "init", "phases": [{"phase": "A", "items": ["shared"]}]},
        context,
    )

    added = await _call(tools, "todo", {"op": "add", "items": ["shared"], "phase": "B"}, context)

    assert added.is_error is False and "1 item(s)" in added.text
    assert [item["text"] for item in builtin.open_todos("unit-test")] == ["shared", "shared"]


@pytest.mark.asyncio
async def test_todo_done_by_phase_resolves_every_open_item_idempotently(tools, context) -> None:
    """A phase target resolves every currently-open item in it at once, and a
    re-issue on a settled phase resolves nothing and reports it cleanly."""
    await _call(
        tools,
        "todo",
        {
            "op": "init",
            "phases": [
                {"phase": "Foundation", "items": ["scaffold", "wire config"]},
                {"phase": "Verification", "items": ["run gate"]},
            ],
        },
        context,
    )

    first = await _call(tools, "todo", {"op": "done", "phase": "Foundation"}, context)
    again = await _call(tools, "todo", {"op": "done", "phase": "Foundation"}, context)

    assert first.is_error is False and "scaffold" in first.text and "wire config" in first.text
    assert again.is_error is False and "No open items in phase 'Foundation'" in again.text
    # Only Foundation was resolved; Verification is untouched and still open.
    assert [item["text"] for item in builtin.open_todos("unit-test")] == ["run gate"]


@pytest.mark.asyncio
async def test_todo_block_and_drop_by_phase(tools, context) -> None:
    """``block``/``drop`` accept a phase target too; block still requires a
    reason, and drop leaves no open work behind in the phase."""
    await _call(
        tools,
        "todo",
        {"op": "init", "phases": [{"phase": "P", "items": ["a", "b"]}]},
        context,
    )

    bare_block = await _call(tools, "todo", {"op": "block", "phase": "P"}, context)
    blocked = await _call(
        tools, "todo", {"op": "block", "phase": "P", "reason": "waiting on legal"}, context
    )

    assert bare_block.is_error is True and "reason" in bare_block.text
    assert blocked.is_error is False
    # Blocked is not open; the guardrail sees an honest stop.
    assert builtin.open_todos("unit-test") == []
    assert all(item["status"] == "blocked" for item in builtin.TODO_STORE["unit-test"][0]["items"])


@pytest.mark.asyncio
async def test_todo_done_by_phase_unknown_phase_is_error(tools, context) -> None:
    await _call(tools, "todo", {"op": "init", "items": ["a"]}, context)
    result = await _call(tools, "todo", {"op": "done", "phase": "Ghost"}, context)
    assert result.is_error is True and "Ghost" in result.text


@pytest.mark.asyncio
async def test_todo_done_by_items_searches_across_phases(tools, context) -> None:
    """A text form must find its item wherever it lives, so the model need not
    know which phase holds it; the first-not-in-target idempotency survives."""
    await _call(
        tools,
        "todo",
        {
            "op": "init",
            "phases": [
                {"phase": "A", "items": ["one"]},
                {"phase": "B", "items": ["two"]},
            ],
        },
        context,
    )

    done = await _call(tools, "todo", {"op": "done", "items": ["two"]}, context)
    # Idempotent re-issue: closing an already-closed text is not an error.
    again = await _call(tools, "todo", {"op": "done", "items": ["two"]}, context)

    assert done.is_error is False and "two" in done.text
    assert again.is_error is False
    assert [item["text"] for item in builtin.open_todos("unit-test")] == ["one"]


@pytest.mark.asyncio
async def test_todo_view_groups_by_phase_with_progress(tools, context) -> None:
    """``view`` echoes phase headers with per-phase (done/total) so the receipt
    mirrors the dock panel."""
    await _call(
        tools,
        "todo",
        {
            "op": "init",
            "phases": [
                {"phase": "Foundation", "items": ["scaffold", "wire config"]},
                {"phase": "Verification", "items": ["run gate"]},
            ],
        },
        context,
    )
    await _call(tools, "todo", {"op": "done", "items": ["scaffold"]}, context)

    view = await _call(tools, "todo", {"op": "view"}, context)

    # Header spelling mirrors the dock panel's ``PhaseName · done/total`` exactly
    # (U5) — one grammar across both surfaces.
    assert view.text.splitlines() == [
        "Foundation · 1/2",
        "- [x] scaffold",
        "- [ ] wire config",
        "Verification · 0/1",
        "- [ ] run gate",
    ]


@pytest.mark.asyncio
async def test_todo_view_single_implicit_phase_is_headerless(tools, context) -> None:
    """The single implicit \"Todos\" phase renders HEADERLESS, byte-identical to
    the pre-phases flat output."""
    await _call(tools, "todo", {"op": "init", "items": ["a", "b"]}, context)

    view = await _call(tools, "todo", {"op": "view"}, context)

    assert view.text.splitlines() == ["- [ ] a", "- [ ] b"]


def test_todo_as_phases_coerces_legacy_flat_list() -> None:
    """The one coercion every reader uses: a legacy flat list becomes one
    implicit \"Todos\" phase; an already-phased list passes through."""
    flat = [{"text": "a", "status": "pending"}]
    coerced = builtin._as_phases(flat)
    assert coerced == [{"name": "Todos", "items": [{"text": "a", "status": "pending"}]}]

    phased = [{"name": "P", "items": [{"text": "b", "status": "pending"}]}]
    assert builtin._as_phases(phased) is phased


def test_todo_fingerprint_moves_on_phase_rename_and_item_move() -> None:
    """A 3-tuple fingerprint sees phase-level movement a 2-tuple was blind to."""
    builtin.TODO_STORE["unit-test"] = [
        {"name": "A", "items": [{"text": "x", "status": "pending"}]},
    ]
    before = builtin.todo_fingerprint("unit-test")

    # A phase rename is movement.
    builtin.TODO_STORE["unit-test"][0]["name"] = "B"
    renamed = builtin.todo_fingerprint("unit-test")

    assert before == (("A", "x", "pending"),)
    assert renamed == (("B", "x", "pending"),)
    assert before != renamed


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
async def test_create_transition_keeps_the_path_transaction_stripe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A second writer cannot bypass the creator after the inode appears."""
    path = tmp_path / "new.txt"
    created = threading.Event()
    release_first = threading.Event()
    state_lock = threading.Lock()
    active = 0
    peak = 0
    real_locked = builtin._write_file_result_locked

    def observed_locked(target: Path, content: str):
        nonlocal active, peak
        with state_lock:
            active += 1
            peak = max(peak, active)
        try:
            result = real_locked(target, content)
            if content == "first":
                created.set()
                assert release_first.wait(timeout=2)
            return result
        finally:
            with state_lock:
                active -= 1

    monkeypatch.setattr(builtin, "_write_file_result_locked", observed_locked)
    first = asyncio.create_task(asyncio.to_thread(builtin._write_file_result, path, "first"))
    assert await asyncio.to_thread(created.wait, 1)
    second = asyncio.create_task(asyncio.to_thread(builtin._write_file_result, path, "second"))
    try:
        await asyncio.sleep(0.05)
        assert peak == 1
        assert not second.done(), "existing-inode stripe bypassed the creator"
    finally:
        release_first.set()
    await asyncio.gather(first, second)

    assert path.read_text() == "second"


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


# ---------------------------------------------------------------------------
# edit: the not-found refusal (closest-region diagnostics)
# ---------------------------------------------------------------------------
#
# The failure these cover is a hunk written from a STALE copy of the file: the
# paragraph still exists, reworded, and the file may even hold a later
# revision of it. The refusal is therefore a diagnosis — it shows the file's
# current text and never applies anything — and these tests pin both halves of
# that contract: the facts it must carry, and the bounds that stop a failing
# edit on a big file from flooding the context it is trying to help.
#
# They assert FACTS (a line number, an overlap figure, the presence of the
# difference pair, whether the two rows actually differ) rather than whole
# sentences of copy, so a legitimate rewording cannot turn into four red tests
# (review round 1, R7). Where the wording IS the subject — the label, the
# frame note — the marker asserted is the shortest one that means it.

_DRIFT_DOC = (
    "Intro paragraph: how a stage holds and releases its key material.\n"
    "Second line of background that the hunk does not touch at all.\n"
    "- **Key release, custody and the possession handshake (S-5, S2-12).** A stage's key is\n"
    "  released only after its quote verifies.\n"
    "Trailing paragraph about something else entirely, for padding.\n"
    "Last line of the document.\n"
)

#: The same sentence as the caller still has it: pre-review wording, and the
#: marker rolled back.
_DRIFT_STALE = (
    "- **Key release and hygiene (S-5).** Session keys are released to a stage only\n"
    "  after its quote verifies."
)

#: ``123| the text`` rows, ready to be checked against the file on disk.
_QUOTED_LINE_RE = re.compile(r"^\s+(\d+)\| (.*)$", re.M)
_READ_RANGE_RE = re.compile(r'range="(\d+)-(\d+)"')


def _assert_quotes_match_the_disk(message: str, path: Path) -> None:
    """Every quoted line must be that line of the FILE, at the number shown.

    This is review round 1's R1 in test form: the diagnostics are built from
    the on-disk content, so a quoted row can be checked against the bytes the
    caller is about to `read`. A report computed from the in-memory batch state
    fails here as soon as an earlier hunk of the same call matched.
    """
    quoted = _QUOTED_LINE_RE.findall(message)
    assert quoted, f"nothing quoted to check:\n{message}"
    disk = path.read_text().splitlines()
    for number, text in quoted:
        index = int(number)
        assert 1 <= index <= len(disk), f"quoted line {index} of a {len(disk)}-line file"
        assert disk[index - 1].startswith(
            text.rstrip("…")[:60]
        ), f"line {index} quoted as {text!r} but the file says {disk[index - 1]!r}"
    for low, high in _READ_RANGE_RE.findall(message):
        assert 1 <= int(low) <= int(high) <= len(disk), f"read range {low}-{high} outside the file"


@pytest.mark.asyncio
async def test_edit_not_found_quotes_the_files_line_at_its_number(tools, context, tmp_path) -> None:
    """The refusal names a line, quotes the FILE's text there, and names its metric."""
    path = tmp_path / "doc.md"
    path.write_text(_DRIFT_DOC)
    result = await _call(
        tools,
        "edit",
        {"path": "doc.md", "old_text": _DRIFT_STALE, "new_text": "REPLACED"},
        context,
    )
    assert result.is_error is True
    assert "hunk 1: old_text not found" in result.text
    assert re.search(r"closest text — file lines? \d+(-\d+)? \(\d+% word overlap\)", result.text)
    # The window is the drifted paragraph (lines 3-4), and every row quoted is
    # that line of the file — the whole point of the change.
    assert "file lines 3-4" in result.text
    _assert_quotes_match_the_disk(result.text, path)
    # The actionable call leads its own line with its range intact: the card
    # clips rows from the right, and a range cut mid-argument cannot be pasted
    # back (design round 1, D2).
    assert re.search(r'^  read\(range="\d+-\d+", path="[^"]+"\)', result.text, re.M)


@pytest.mark.asyncio
async def test_edit_diagnostics_agree_with_the_file_on_disk(tools, context, tmp_path) -> None:
    """An earlier hunk matching must not move the lines the report names.

    Eight lines on disk; hunk 1 inserts a line (so the in-memory text is nine
    lines and everything after line 3 shifts), hunk 2 is a stale copy of disk
    line 7. Computing the report against the in-memory state named line 8 of
    the file and suggested reading past its end (review R1 / QA Q1); the
    caller's next move is a `read` of the disk, so the report must describe the
    disk.
    """
    path = tmp_path / "mixed.txt"
    path.write_text("".join(f"line {i} of the document\n" for i in range(1, 9)))
    result = await _call(
        tools,
        "edit",
        {
            "path": "mixed.txt",
            "edits": [
                {
                    "old_text": "line 3 of the document\n",
                    "new_text": "line 3 of the document\nan inserted line\n",
                },
                {"old_text": "line 7 of the documnt (typo, stale)", "new_text": "x"},
            ],
        },
        context,
    )
    assert result.is_error is True
    assert "1 of 2 hunks did not match" in result.text
    # Every quoted row and the read range are checked against the file itself.
    _assert_quotes_match_the_disk(result.text, path)
    assert "line 7 of the document" in result.text
    # And the frame is stated, because the two views genuinely differ here.
    assert "Note:" in result.text and "nothing was written" in result.text


@pytest.mark.asyncio
async def test_edit_frame_note_is_absent_when_nothing_matched_earlier(
    tools, context, tmp_path
) -> None:
    """No earlier hunk matched, so there is nothing to disclaim."""
    path = tmp_path / "plain.txt"
    path.write_text("alpha\nbeta\ngamma\ndelta\n")
    result = await _call(
        tools, "edit", {"path": "plain.txt", "old_text": "zzz nope", "new_text": "x"}, context
    )
    assert result.is_error is True
    assert "Note:" not in result.text


@pytest.mark.asyncio
async def test_edit_not_found_shows_the_first_difference(tools, context, tmp_path) -> None:
    """The differing pair is the file's real line, and the two rows DO differ."""
    path = tmp_path / "doc.md"
    path.write_text(_DRIFT_DOC)
    result = await _call(
        tools,
        "edit",
        {"path": "doc.md", "old_text": _DRIFT_STALE, "new_text": "REPLACED"},
        context,
    )
    assert "first difference — your line 1 vs file line 3:" in result.text
    rows = [
        line
        for line in result.text.splitlines()
        if line.startswith("    - ") or line.startswith("    + ")
    ]
    assert len(rows) == 2
    yours, theirs = (row[6:] for row in rows)
    assert yours != theirs
    assert theirs.startswith("- **Key release, custody")
    assert re.search(r"identical for the first \d+ characters, then they diverge", result.text)


@pytest.mark.parametrize("cut", [77, 231])
@pytest.mark.asyncio
async def test_edit_difference_rows_differ_inside_the_visible_excerpt(
    cut: int, tools, context, tmp_path
) -> None:
    """The card clips rows from the RIGHT, so the pair must diverge near the start.

    Both rows used to be windowed at character 0, which rendered them
    byte-identical whenever the divergence sat past the clip (design round 1,
    D1; QA round 1, Q3 — reproduced at character 231). The excerpt now leads
    with ~24 characters of run-up, so the divergence is inside the first ~30
    characters of what is actually painted.
    """
    path = tmp_path / "wide.txt"
    filler = "a" * 400
    path.write_text(filler[:cut] + "FILE-SIDE" + filler[cut:] + "\n")
    hunk = filler[:cut] + "HUNK-SIDE" + filler[cut:]
    result = await _call(
        tools, "edit", {"path": "wide.txt", "old_text": hunk, "new_text": "x"}, context
    )
    rows = [
        line
        for line in result.text.splitlines()
        if line.startswith("    - ") or line.startswith("    + ")
    ]
    assert len(rows) == 2, result.text
    yours, theirs = (row[6:] for row in rows)
    assert yours != theirs, "the two rows rendered identically"
    assert yours[:40] != theirs[:40], "the divergence is past the visible part of the row"


@pytest.mark.asyncio
async def test_edit_not_found_says_no_close_match_without_inventing_a_region(
    tools, context, tmp_path
) -> None:
    """Nothing similar anywhere: say so, and do not fabricate a region."""
    path = tmp_path / "doc.md"
    path.write_text(_DRIFT_DOC)
    result = await _call(
        tools,
        "edit",
        {
            "path": "doc.md",
            "old_text": "quokka nimbus walrus zephyr marmot pelican saxophone",
            "new_text": "x",
        },
        context,
    )
    assert result.is_error is True
    assert "no close match" in result.text
    assert "closest text" not in result.text
    assert not _QUOTED_LINE_RE.search(result.text)

    # One shared word is enough to have a nearest line, and the label must stay
    # honest: the overlap is still nowhere near a match.
    shared = await _call(
        tools,
        "edit",
        {
            "path": "doc.md",
            "old_text": "quokka nimbus walrus zephyr marmot pelican stage",
            "new_text": "x",
        },
        context,
    )
    assert "no close match" in shared.text
    assert re.search(r"nearest text is line \d+ \(\d+% word overlap\)", shared.text)
    assert not _QUOTED_LINE_RE.search(shared.text)


@pytest.mark.asyncio
async def test_edit_no_overlap_is_not_reported_as_an_empty_file(tools, context, tmp_path) -> None:
    """A file full of text that shares no words is not an empty file."""
    path = tmp_path / "text.txt"
    path.write_text("alpha beta gamma delta\nepsilon zeta eta theta\n")
    result = await _call(
        tools, "edit", {"path": "text.txt", "old_text": "quokka", "new_text": "x"}, context
    )
    assert result.is_error is True
    assert "no close match" in result.text
    # The claim the search actually established, not one it did not.
    assert "there is no text" not in result.text


@pytest.mark.asyncio
async def test_edit_anchor_line_is_echoed_back(tools, context, tmp_path) -> None:
    """The one hint the caller gave the tool is repeated in the refusal.

    It is free advice: the caller said which line it meant, and the old message
    offered "the range around line N" back (review round 1, R4).
    """
    path = tmp_path / "anchor.txt"
    path.write_text("alpha\nbeta\ngamma\ndelta\n")
    result = await _call(
        tools,
        "edit",
        {"path": "anchor.txt", "old_text": "zzz nope", "new_text": "x", "anchor_line": 3},
        context,
    )
    assert result.is_error is True
    # The suggestion is a real call whose range CONTAINS the anchor line, not a
    # fixed range at the head of the file (review round 2, R9) — asserted as a
    # fact about the range so a wording change does not break it (R7).
    hint = re.search(r'read\(range="(\d+)-(\d+)", path="[^"]+"\)', result.text)
    assert hint, result.text
    first, last = (int(value) for value in hint.groups())
    assert first <= 3 <= last
    assert "anchor" in result.text


@pytest.mark.asyncio
async def test_edit_anchor_line_is_echoed_beside_a_close_region(tools, context, tmp_path) -> None:
    """With a region to point at, the anchor is still repeated (R4/R7)."""
    path = tmp_path / "anchor2.txt"
    path.write_text("alpha\nbeta\nthe quick brown fox jumps over the lazy dog\ndelta\n")
    result = await _call(
        tools,
        "edit",
        {
            "path": "anchor2.txt",
            "old_text": "the quick brown cat jumps over the lazy dog",
            "new_text": "x",
            "anchor_line": 3,
        },
        context,
    )
    assert result.is_error is True
    assert re.search(r"anchor[^\n]*\b3\b", result.text), result.text


@pytest.mark.asyncio
async def test_edit_batch_reports_every_refusal_and_writes_nothing(
    tools, context, tmp_path
) -> None:
    """One round trip names all the bad hunks, and the good ones stay unwritten.

    The old engine returned on the FIRST failing hunk, so a caller with two
    broken hunks paid two round trips to learn what one message can say.
    """
    path = tmp_path / "batch.txt"
    path.write_text("alpha line\nbeta line\ngamma line\n")
    before = path.read_bytes()
    result = await _call(
        tools,
        "edit",
        {
            "path": "batch.txt",
            "edits": [
                {"old_text": "alpha line", "new_text": "ALPHA"},
                {"old_text": "missing one\n", "new_text": "x"},
                {"old_text": "beta line", "new_text": "BETA"},
                {"old_text": "missing two\n", "new_text": "y"},
            ],
        },
        context,
    )
    assert result.is_error is True
    assert "2 of 4 hunks" in result.text
    assert "hunk 2: old_text not found" in result.text
    assert "hunk 4: old_text not found" in result.text
    assert "hunk 1" not in result.text and "hunk 3" not in result.text
    # Byte-identical: the atomicity claim is checked, not restated in prose.
    assert path.read_bytes() == before


@pytest.mark.asyncio
async def test_edit_ambiguous_refusal_lists_the_match_lines(tools, context, tmp_path) -> None:
    """The count was always there; the line numbers were the missing half."""
    path = tmp_path / "amb.txt"
    path.write_text("  foo\nbar\n  foo\n")
    result = await _call(
        tools,
        "edit",
        {"path": "amb.txt", "old_text": "foo", "new_text": "X"},
        context,
    )
    assert result.is_error is True
    assert "old_text matches 2 places (lines 1, 3)" in result.text
    assert "give anchor_line" in result.text
    # And the header is about the reason that actually happened: the hunk
    # matched, twice, which "did not match" would contradict (QA round 1, Q2).
    assert "matched more than one place" in result.text
    assert "did not match" not in result.text


@pytest.mark.asyncio
async def test_edit_mixed_refusal_reasons_are_named_in_the_header(tools, context, tmp_path) -> None:
    """A batch that fails for two reasons says both, and counts each."""
    path = tmp_path / "mixed.txt"
    path.write_text("foo\nbar\nfoo\n")
    result = await _call(
        tools,
        "edit",
        {
            "path": "mixed.txt",
            "edits": [
                {"old_text": "foo", "new_text": "X"},
                {"old_text": "nothing like this at all", "new_text": "Y"},
            ],
        },
        context,
    )
    assert result.is_error is True
    assert "2 of 2 hunks did not apply" in result.text
    assert "1 did not match" in result.text
    assert "1 matched more than one place" in result.text


def _failing_batch(count: int) -> list[dict[str, str]]:
    return [
        {"old_text": f"absent hunk number {i} about quokkas and walruses", "new_text": "x"}
        for i in range(count)
    ]


@pytest.mark.asyncio
async def test_edit_refusal_is_bounded_and_names_what_it_suppressed(
    tools, context, tmp_path
) -> None:
    """Ten refusals stay under the cap, are counted, and are never half-shown."""
    path = tmp_path / "many.txt"
    path.write_text("alpha beta gamma delta epsilon\n" * 5)
    result = await _call(
        tools,
        "edit",
        {"path": "many.txt", "edits": _failing_batch(10)},
        context,
    )
    assert result.is_error is True
    assert "10 of 10 hunks" in result.text
    assert len(result.text) <= builtin._EDIT_MAX_MESSAGE_CHARS
    detailed = builtin._EDIT_MAX_DETAILED_HUNKS
    compact = builtin._EDIT_MAX_COMPACT_HUNKS
    assert result.text.count("old_text not found") == detailed + compact
    # The suppression line names exactly the hunks that got no block at all, and
    # every refusal is accounted for once. Asserted as the number plus the
    # suppression claim, so the tail's wording stays free (review round 2, R7).
    tail = result.text.splitlines()[-1]
    assert str(10 - detailed - compact) in tail and "suppress" in tail


@pytest.mark.asyncio
async def test_edit_refusal_stays_bounded_for_a_long_pattern_in_a_large_file(
    tools, context, tmp_path
) -> None:
    """A 200-line pattern against a ~5k-line file still returns a bounded message.

    This is the shape that would be unbounded by default: the window is 200
    lines long, only a handful of them are ever quoted, and the note says how
    many there were.
    """
    rng = random.Random(11)
    words = [f"w{i}" for i in range(400)]
    lines = [" ".join(rng.choice(words) for _ in range(12)) for _ in range(5000)]
    path = tmp_path / "big.txt"
    path.write_text("\n".join(lines) + "\n")
    pattern = lines[2400:2600]
    drifted = "\n".join(
        " ".join(line.split()[1:]) if i % 4 == 0 else line for i, line in enumerate(pattern)
    )
    result = await _call(
        tools,
        "edit",
        {"path": "big.txt", "old_text": drifted, "new_text": "x"},
        context,
    )
    assert result.is_error is True
    assert len(result.text) <= builtin._EDIT_MAX_MESSAGE_CHARS
    quoted = _QUOTED_LINE_RE.findall(result.text)
    assert 0 < len(quoted) <= builtin._EDIT_MAX_WINDOW_LINES
    assert f"of {len(pattern)}" in result.text


@pytest.mark.asyncio
async def test_edit_summarises_refusals_past_the_detailed_cap_compactly(
    tools, context, tmp_path
) -> None:
    """Past the detailed cap each refusal still gets one line naming a place."""
    path = tmp_path / "mixed.txt"
    path.write_text(_DRIFT_DOC)
    edits = [{"old_text": _DRIFT_STALE, "new_text": "x"}] + [
        {"old_text": f"absent hunk {i} about quokkas and walruses", "new_text": "y"}
        for i in range(7)
    ]
    result = await _call(tools, "edit", {"path": "mixed.txt", "edits": edits}, context)
    assert result.is_error is True
    assert "8 of 8 hunks" in result.text
    assert result.text.count("old_text not found (exact and whitespace-tolerant") == (
        builtin._EDIT_MAX_DETAILED_HUNKS
    )
    # The first hunk the cap turned into a one-liner still names a line.
    first_compact = builtin._EDIT_MAX_DETAILED_HUNKS + 1
    assert re.search(
        rf"hunk {first_compact}: old_text not found — (closest text at line \d+|"
        rf"no close match \(nearest line \d+)",
        result.text,
    )
    tail = result.text.splitlines()[-1]
    suppressed = 8 - builtin._EDIT_MAX_DETAILED_HUNKS - builtin._EDIT_MAX_COMPACT_HUNKS
    assert str(suppressed) in tail and "suppress" in tail


@pytest.mark.asyncio
async def test_edit_worst_case_refusal_fits_the_expanded_card(tools, context, tmp_path) -> None:
    """The most blocks this message can carry must still FIT the expanded card.

    This is the constraint that sets ``_EDIT_MAX_DETAILED_HUNKS`` (design round
    1, D4): the card paints at most ``EXPAND_MAX_LINES`` rows of a message and
    silently drops the rest, so a message whose tail falls past that is a
    diagnosis the reader cannot reach by any key. Two full reports plus the
    compact lines plus the tail is the worst shape the caps allow; a third
    report is ~15 more rows and does not fit.
    """
    from local_operator.tui.widgets.tool_card import EXPAND_MAX_LINES

    vocabulary = [f"token{i}" for i in range(60)]
    body: list[str] = []
    starts: list[int] = []
    for seed in range(6):
        paragraph_rng = random.Random(seed)
        starts.append(len(body))
        body.extend(" ".join(paragraph_rng.choice(vocabulary) for _ in range(40)) for _ in range(8))
        body.append("")

    def drift(paragraph: list[str]) -> str:
        return "\n".join(
            " ".join(line.split()[: max(1, len(line.split()) - 3)]) if i % 2 == 0 else line
            for i, line in enumerate(paragraph)
        )

    path = tmp_path / "worst.txt"
    path.write_text("\n".join(body) + "\n")
    # THREE full-size candidates, so a third detailed block would cost 15 more
    # rows rather than a two-line no-close-match stub.
    edits = [
        {"old_text": drift(body[starts[index] : starts[index] + 6]), "new_text": "x"}
        for index in (1, 3, 5)
    ] + [{"old_text": f"absent hunk {i} with a few words", "new_text": "y"} for i in range(3)]
    result = await _call(tools, "edit", {"path": "worst.txt", "edits": edits}, context)
    assert result.is_error is True
    assert result.text.count("old_text not found (exact and whitespace-tolerant") == (
        builtin._EDIT_MAX_DETAILED_HUNKS
    )
    assert len(result.text.splitlines()) <= EXPAND_MAX_LINES


def test_edit_label_follows_the_printed_percent(monkeypatch: pytest.MonkeyPatch) -> None:
    """The label is decided by the number the message prints (review R5).

    0.296 rounds to ``30%``, which is the label threshold — so the region gets
    the ``closest text`` label, not ``no close match`` next to ``30% word
    overlap``. This drives the real report builders with a forced score, so it
    fails if either of them goes back to comparing the raw float.
    """
    monkeypatch.setattr(builtin, "_closest_regions", lambda scan, pattern, limit=3: [(0, 1, 0.296)])
    scan = builtin._FileScan("one line of text\nanother line of text\n")
    detailed = "\n".join(builtin._edit_not_found_report(scan, 1, "some hunk text", "/tmp/x"))
    compact = "\n".join(builtin._edit_not_found_compact(scan, 1, "some hunk text"))
    for message in (detailed, compact):
        assert "30% word overlap" in message
        assert "no close match" not in message


@pytest.mark.asyncio
async def test_edit_single_line_file_uses_singular_copy(tools, context, tmp_path) -> None:
    """``1 line``, not ``1 lines`` — and a one-line region is ``file line 7``."""
    path = tmp_path / "one.txt"
    path.write_text("only line\n")
    singular = await _call(
        tools, "edit", {"path": "one.txt", "old_text": "zzz nope", "new_text": "x"}, context
    )
    assert "— 1 line," in singular.text
    assert "1 lines" not in singular.text

    path.write_text(_DRIFT_DOC)
    one_line = await _call(
        tools,
        "edit",
        {
            "path": "one.txt",
            "old_text": "Trailing paragraph about something else entirly",
            "new_text": "x",
        },
        context,
    )
    assert "closest text — file line 5" in one_line.text
    assert "file lines 5-5" not in one_line.text


def test_edit_overlap_label_agrees_with_the_printed_percent() -> None:
    """One rounded number decides the label AND what gets printed (review R5).

    The raw float decided the label while the rounded integer got printed, so a
    region at 0.296 printed ``30% word overlap`` under a ``no close match``
    label. Both now read ``_overlap_percent``.
    """
    for value, expected in (
        (0.296, "closest"),
        (0.294, "no close"),
        (0.30, "closest"),
        (0.0, "no close"),
    ):
        printed = builtin._overlap_percent(value)
        label = "closest" if printed >= builtin._EDIT_OVERLAP_LABEL_PERCENT else "no close"
        assert label == expected, (value, printed)
        assert builtin._percent(value) == f"{printed}%"
    # A percent printed under the "no close match" label is always below it.
    assert builtin._overlap_percent(0.294) < builtin._EDIT_OVERLAP_LABEL_PERCENT


# The tolerant pass as it was before the index: the reference the prefilter
# must agree with, kept here on purpose. It is the ONLY definition of the
# semantics — the fast one is an optimisation of this, so a disagreement is a
# bug in the fast one.
def _naive_tolerant_windows(content: str, old_text: str) -> list[tuple[int, int, str]]:
    windows: list[tuple[int, int, str]] = []
    start = content.find(old_text)
    while start != -1:
        windows.append((start, start + len(old_text), old_text))
        start = content.find(old_text, start + 1)
    if windows:
        return windows

    file_lines = content.splitlines(keepends=True)
    offsets: list[int] = []
    at = 0
    for line in file_lines:
        offsets.append(at)
        at += len(line)
    old_lines = old_text.splitlines()
    if not old_lines or len(old_lines) > len(file_lines):
        return []

    for i in range(len(file_lines) - len(old_lines) + 1):
        window = file_lines[i : i + len(old_lines)]
        if all(f.rstrip("\r\n").strip() == o.strip() for f, o in zip(window, old_lines)):
            last_line = window[-1]
            end = offsets[i + len(old_lines) - 1] + len(last_line)
            windows.append((offsets[i], end, "".join(window)))
    return windows


def _tolerant_corpus() -> list[tuple[str, str]]:
    """Patterns that exercise the seeded prefilter's edges, with a file each."""
    lf = "alpha\n    beta\n\ngamma\nalpha\nbeta\n"
    crlf = "alpha\r\n    beta\r\n\r\ngamma\r\nalpha\r\nbeta\r\n"
    repeats = "x\nblock\nsame\nsame\nx\nblock\nsame\nsame\nend\n"
    cases = [
        # Indent drift in both directions, and a blank interior line.
        (lf, "    alpha\nbeta"),
        (lf, "alpha\n\tbeta\n\ngamma"),
        # CRLF file against an LF hunk (and the reverse spelling).
        (crlf, "    alpha\nbeta"),
        (crlf, "alpha\n  beta\n\ngamma"),
        # Repeated blocks: every candidate start must still be found.
        (repeats, "block\nsame\nsame"),
        # Pattern longer than the file.
        (repeats, "block\nsame\nsame\nend\nmore"),
        # Blank first line, and a pattern that is only a blank line.
        (lf, "\ngamma"),
        (lf, "\n"),
        # Nothing that matches at all, plus a file with no newline at the end.
        (lf, "nothing\nhere"),
        ("tail-without-newline", "tail-without-newline"),
        # Seed line present elsewhere with the WRONG neighbours: the indexed
        # pass must reject the start the seed alone cannot rule out. Both
        # pattern lines occur the same number of times, so the seed is the
        # first of them, and only one of its occurrences has the right tail.
        # The indent on the second line is what keeps pass 1 (exact) out of
        # the way so pass 2 actually runs.
        ("a\n  b\na\nc\n  b\nd\n", "a\nb"),
        ("p\n  q\nx\np\ny\n  q\n", "p\nq"),
    ]
    return cases


def test_tolerant_prefilter_matches_the_naive_reference() -> None:
    """The seeded index is an optimisation, so it must be indistinguishable."""
    for content, old_text in _tolerant_corpus():
        expected = _naive_tolerant_windows(content, old_text)
        actual = builtin._match_windows(content, old_text)
        assert actual == expected, (content, old_text, expected, actual)


def test_tolerant_prefilter_agrees_over_a_generated_corpus() -> None:
    """Property-style version of the check above, over generated shapes."""
    rng = random.Random(4242)
    vocabulary = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta"]
    for _ in range(120):
        line_count = rng.randint(0, 30)
        lines = []
        for _ in range(line_count):
            indent = " " * rng.choice([0, 0, 1, 2, 4, 8]) + ("\t" if rng.random() < 0.2 else "")
            body = " ".join(rng.choice(vocabulary) for _ in range(rng.randint(0, 4)))
            lines.append(indent + body)
        ending = rng.choice(["\n", "\r\n"])
        content = ending.join(lines) + (ending if lines and rng.random() < 0.8 else "")
        start = rng.randrange(0, max(line_count, 1))
        width = rng.randint(1, 4)
        old_text = ending.join(lines[start : start + width])
        if rng.random() < 0.35:
            # Drift the pattern's indentation, the way a model writing from
            # memory does.
            old_text = "\n".join(
                (" " * rng.choice([0, 2, 4, 8])) + line.strip() for line in old_text.splitlines()
            )
        assert builtin._match_windows(content, old_text) == _naive_tolerant_windows(
            content, old_text
        ), (content, old_text)


class _ScoringCounter:
    """Counts how many candidate windows the full metric is run over."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.count = 0
        real = builtin._dice

        def counting(pattern, window):  # type: ignore[no-untyped-def]
            self.count += 1
            return real(pattern, window)

        monkeypatch.setattr(builtin, "_dice", counting)


def test_closest_region_search_scores_a_bounded_number_of_windows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A structural bound, not a stopwatch: the work cannot grow with file size.

    A timing assertion here would encode this laptop's core speed into the
    test (see AGENTS.md, "Timing, flakes, and how to assert that something is
    fast"). What actually matters is that the exhaustive O(file_lines x
    pattern_lines) scan is gone, and that is a fact about how many windows are
    scored.
    """
    pattern = [f"absent prose line {i} with several words" for i in range(40)]
    counts = []
    for line_count in (60, 6000):
        content = "\n".join(f"filler line {i} of the document" for i in range(line_count))
        counter = _ScoringCounter(monkeypatch)
        builtin._closest_regions(builtin._FileScan(content), pattern)
        counts.append(counter.count)
    ceiling = builtin._EDIT_MAX_ANCHORS * builtin._EDIT_MAX_CANDIDATE_STARTS
    assert all(count <= ceiling for count in counts), counts
    # 100x the file, the same search: the prefilter is what makes that true.
    assert counts[0] == counts[1]


def _counting_file_scan(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Every ``_FileScan`` construction, in order (review round 2, R11)."""
    built: list[str] = []
    real = builtin._FileScan

    class Counting(real):  # type: ignore[misc, valid-type]
        def __init__(self, content: str) -> None:
            built.append(content)
            super().__init__(content)

    monkeypatch.setattr(builtin, "_FileScan", Counting)
    return built


@pytest.mark.asyncio
async def test_edit_builds_one_scan_per_frame_not_one_per_hunk(
    tools, context, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Frames are built per content version, not per hunk.

    The diagnostics describe the file on disk while matching runs against the
    in-memory state, so a failing batch legitimately needs two views. What it
    must not do is rebuild them per hunk: a 10-hunk failure costs the same as a
    2-hunk one.
    """
    built = _counting_file_scan(monkeypatch)
    path = tmp_path / "many.txt"
    path.write_text("".join(f"line {i} of the document\n" for i in range(1, 21)))
    edits = [
        {"old_text": "line 2 of the document\n", "new_text": "line 2 of the document\nmore\n"}
    ] + [{"old_text": f"absent hunk {i} about quokkas", "new_text": "y"} for i in range(10)]
    result = await _call(tools, "edit", {"path": "many.txt", "edits": edits}, context)
    assert result.is_error is True
    assert len(built) == 2, built


@pytest.mark.asyncio
async def test_edit_pins_the_disk_frame_across_an_alternating_batch(
    tools, context, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The on-disk frame survives a batch that applies and fails alternately.

    Review round 2 (R11): with the two frames sharing one cache, an
    apply-then-fail batch evicted the disk frame each time a hunk wrote, so
    every later failure rebuilt it as well. Pinned in its own slot, the disk
    frame is built EXACTLY once however the batch alternates, and the matching
    frame is rebuilt only when a hunk actually writes — which is the honest
    bound the comments state, and the bound asserted below.
    """
    original_text = "".join(f"line {i} of the document\n" for i in range(1, 31))
    built = _counting_file_scan(monkeypatch)
    path = tmp_path / "alternating.txt"
    path.write_text(original_text)
    applies = [
        {"old_text": f"line {n} of the document\n", "new_text": f"line {n} revised\n"}
        for n in (3, 7, 11)
    ]
    # An apply FIRST, so the matching frame can never be mistaken for the disk
    # frame by content: from hunk 2 on, the two frames differ by construction.
    edits = [
        applies[0],
        {"old_text": "absent hunk about quokkas", "new_text": "y"},
        applies[1],
        {"old_text": "another absent hunk about quokkas", "new_text": "y"},
        applies[2],
        {"old_text": "a third absent hunk about quokkas", "new_text": "y"},
    ]
    result = await _call(tools, "edit", {"path": "alternating.txt", "edits": edits}, context)
    assert result.is_error is True
    disk_builds = [content for content in built if content == original_text]
    assert len(disk_builds) == builtin._EDIT_MAX_DISK_SCANS, built
    # At most the pinned disk frame plus one matching frame per content version.
    assert len(built) <= 1 + len(applies) + builtin._EDIT_MAX_DISK_SCANS, built


_AMBIGUOUS_LINES_RE = re.compile(r"old_text matches \d+ places \(lines ([\d, …]+)\)")


def _listed_match_lines(message: str) -> list[int]:
    """The line numbers an ambiguity refusal lists, as integers."""
    found = _AMBIGUOUS_LINES_RE.search(message)
    assert found, message
    return [int(value) for value in re.findall(r"\d+", found.group(1))]


@pytest.mark.asyncio
async def test_edit_ambiguous_list_names_the_files_lines_not_the_batchs(
    tools, context, tmp_path
) -> None:
    """Review round 2, R8: the match list is the FILE's, like every other fact.

    An earlier hunk of the same call had already inserted a line in memory, so
    the batch's own numbering is shifted; the caller's next move is a `read` of
    the file (or an `anchor_line` against it), so the listed numbers have to be
    the file's.
    """
    path = tmp_path / "amb.txt"
    path.write_text("x\nfoo\ny\nfoo\nz\n")
    edits = [
        {"old_text": "x\n", "new_text": "x\ninserted\n"},
        {"old_text": "foo", "new_text": "bar"},
    ]
    result = await _call(tools, "edit", {"path": "amb.txt", "edits": edits}, context)
    assert result.is_error is True
    disk = path.read_text().splitlines()
    listed = _listed_match_lines(result.text)
    assert listed == [2, 4], result.text
    # Each listed number is a line that really holds this text.
    assert all(disk[number - 1] == "foo" for number in listed), listed
    # And the frame note is present, because the frames really do differ here.
    assert "Note: earlier hunks" in result.text


@pytest.mark.asyncio
async def test_edit_ambiguous_row_says_when_the_batch_created_the_duplicate(
    tools, context, tmp_path
) -> None:
    """A duplicate this call created is named as such, never as a file line."""
    path = tmp_path / "amb2.txt"
    path.write_text("x\nfoo\ny\nz\n")
    edits = [
        {"old_text": "x\n", "new_text": "x\nfoo\n"},
        {"old_text": "foo", "new_text": "bar"},
    ]
    result = await _call(tools, "edit", {"path": "amb2.txt", "edits": edits}, context)
    assert result.is_error is True
    disk = path.read_text().splitlines()
    # The file holds one occurrence (line 2); the batch's second copy is what
    # made the hunk ambiguous, and the message says so instead of listing a
    # line the file does not have.
    assert "in-memory" in result.text, result.text
    assert "1 place in the file on disk (line 2)" in result.text, result.text
    assert disk[1] == "foo"


@pytest.mark.asyncio
async def test_edit_hint_prints_the_callers_path_spelling(tools, context, tmp_path) -> None:
    """Design round 2, D2 residual: the hint's path is the caller's own.

    The message is for the caller and quotes a `read` they will re-use, so it
    prints the spelling they passed. The resolved absolute path of this fixture
    is much longer and pushed the range argument off the card's row at every
    width, which is the defect this fixes.
    """
    nested = tmp_path / "local_operator" / "tools"
    nested.mkdir(parents=True)
    (nested / "builtin.py").write_text("".join(f"line {i} of the document\n" for i in range(1, 30)))
    caller_path = "local_operator/tools/builtin.py"
    result = await _call(
        tools,
        "edit",
        # A drifted hunk, so the report carries a closest region AND a hint.
        {"path": caller_path, "old_text": "line 7 of the documnt (typo)", "new_text": "y"},
        context,
    )
    assert result.is_error is True
    hint = next(line for line in result.text.splitlines() if line.strip().startswith("read("))
    assert f'path="{caller_path}"' in hint, hint
    assert str(tmp_path) not in hint, "the hint must not carry the resolved absolute path"
    # range first: at the narrowest supported width it is the argument that
    # survives the card's right-truncation. The header line keeps the resolved
    # path, which is the one place an absolute path is the right answer.
    assert re.match(r'^  read\(range="\d+-\d+", path="[^"]+"\)', hint), hint
    assert str(tmp_path) in result.text.splitlines()[1]


@pytest.mark.asyncio
async def test_edit_no_region_hint_never_invents_a_range(tools, context, tmp_path) -> None:
    """Review round 2, R9: with nothing to point at, no range is invented.

    The old hint emitted the head of the file (and, on an empty file, the
    inverted `1-0`) — a place the failure gave no reason to look at.
    """
    path = tmp_path / "big.txt"
    path.write_text("".join(f"line {i} of the document\n" for i in range(1, 5001)))
    result = await _call(
        tools,
        "edit",
        {"path": "big.txt", "old_text": "quokka marsupial taxonomy", "new_text": "y"},
        context,
    )
    assert result.is_error is True
    assert "no close match" in result.text
    assert not re.search(r'range="\d+-\d+"', result.text), result.text

    empty = tmp_path / "empty.txt"
    empty.write_text("")
    result = await _call(
        tools,
        "edit",
        {"path": "empty.txt", "old_text": "anything at all", "new_text": "y"},
        context,
    )
    assert result.is_error is True
    assert 'range="1-0"' not in result.text
    assert not re.search(r'range="\d+-\d+"', result.text), result.text


@pytest.mark.asyncio
async def test_edit_no_region_hint_uses_the_anchor_line_when_given(
    tools, context, tmp_path
) -> None:
    """With an anchor, the suggested range brackets it and stays in the file."""
    path = tmp_path / "big2.txt"
    path.write_text("".join(f"line {i} of the document\n" for i in range(1, 5001)))
    result = await _call(
        tools,
        "edit",
        {
            "path": "big2.txt",
            "old_text": "quokka marsupial taxonomy",
            "new_text": "y",
            "anchor_line": 4000,
        },
        context,
    )
    assert result.is_error is True
    found = re.search(r'read\(range="(\d+)-(\d+)", path="[^"]+"\)', result.text)
    assert found, result.text
    first, last = (int(value) for value in found.groups())
    assert first <= 4000 <= last <= 5000
    # No quoted rows and no closest region: the file resembles nothing, so the
    # only number in the message is the range around the caller's own anchor.
    assert "closest text" not in result.text


def test_own_text_on_disk_is_an_identity_test_not_a_metric() -> None:
    """The gate itself (review round 3, R13): text, never a similarity figure.

    ``_overlap_percent`` is bag-of-words, so a punctuation-only rewrite of the
    same words scores exactly 100% and the clause ("this is this hunk's own
    text") would be false. This pins the gate directly, because the two
    repro shapes are also covered by the no-difference-pair guard and would
    therefore survive a metric-based gate on their own.
    """
    disk = "alpha, beta gamma\nmoved line\n"
    current = "alpha, beta gamma\nmoved line\nrewritten\n"
    # Same word bag, different text: NOT the hunk's own text.
    assert builtin._own_text_on_disk(disk, current, "alpha beta gamma\n") is False
    # A one-word drop from a 120-token line: raw overlap 0.9958, which ROUNDS to
    # 100% and so fired the old percentage gate. The window is still not the
    # hunk's text, which is what the gate has to answer.
    long_line = " ".join(f"token{i}" for i in range(1, 121))
    without_one = " ".join(word for word in long_line.split() if word != "token60")
    assert (
        builtin._own_text_on_disk(
            long_line + "\nrewritten\n", long_line + "\nrewritten\n", without_one + "\n"
        )
        is False
    )
    # The genuine case: byte-identical on disk, moved on in the batch.
    assert builtin._own_text_on_disk("target text\nother\n", "other\n", "target text\n") is True
    # Frames identical: nothing was consumed, so there is no clause to make.
    assert builtin._own_text_on_disk("target text\n", "target text\n", "target text\n") is False


@pytest.mark.asyncio
async def test_edit_self_match_clause_needs_text_identity_not_a_similarity(
    tools, context, tmp_path
) -> None:
    """Review round 3, R13 / design round 2, D8: identity, not 100% overlap.

    A bag-of-words figure reaches 100% for a punctuation-only rewrite or a
    reordered list, which is exactly when the clause ("this is this hunk's own
    text") was false — and false in the same block as the difference pair that
    disproved it.
    """
    # (a) punctuation only: the same word bag, different text. Hunk 1 APPLIES,
    # so the frames differ and only the identity test can withhold the clause —
    # the shape both streams reproduced.
    punctuation = tmp_path / "punctuation.txt"
    punctuation.write_text("alpha, beta gamma\nother line\n")
    result = await _call(
        tools,
        "edit",
        {
            "path": "punctuation.txt",
            "edits": [
                {"old_text": "other line\n", "new_text": "OTHER LINE\n"},
                {"old_text": "alpha beta gamma\n", "new_text": "y\n"},
            ],
        },
        context,
    )
    assert result.is_error is True
    assert "100% word overlap" in result.text, result.text
    assert "unchanged on disk" not in result.text, result.text
    # ... and the block stays self-consistent: the pair that disproves the
    # identity is right there, which is what made the clause self-contradictory.
    assert "first difference" in result.text, result.text
    assert punctuation.read_text() == "alpha, beta gamma\nother line\n"

    # (b) one word dropped from a 120-token line: the designer's shape, where
    # raw overlap is 0.9958 and therefore ROUNDS to 100% — the case that made a
    # percentage gate fire on text that is not the hunk's.
    long_line = " ".join(f"token{i}" for i in range(1, 121))
    without_one = " ".join(word for word in long_line.split() if word != "token60")
    body = f"{long_line}\na second line, so the batch has something to apply\n"
    drifted = tmp_path / "drifted.txt"
    drifted.write_text(body)
    result = await _call(
        tools,
        "edit",
        {
            "path": "drifted.txt",
            "edits": [
                {
                    "old_text": "a second line, so the batch has something to apply\n",
                    "new_text": "a second line, rewritten\n",
                },
                {"old_text": without_one + "\n", "new_text": "y\n"},
            ],
        },
        context,
    )
    assert result.is_error is True
    assert "100% word overlap" in result.text, result.text
    assert "unchanged on disk" not in result.text, result.text
    assert drifted.read_text() == body


@pytest.mark.asyncio
async def test_edit_self_match_clause_is_short_and_leads_with_the_cause(
    tools, context, tmp_path
) -> None:
    """Design round 2, D9: the clause must fit a narrow card, cause first.

    At 60 columns the message body has ~50 visible cells and the card clips
    every row from the right, so a 149-character clause lost the reason
    entirely.
    """
    line = "the whole quote chain must verify before any plaintext leaves the stage\n"
    path = tmp_path / "short.txt"
    path.write_text(line)
    result = await _call(
        tools,
        "edit",
        {
            "path": "short.txt",
            "edits": [
                {"old_text": line, "new_text": "the replacement chain must verify\n"},
                {"old_text": line, "new_text": "another chain must verify\n"},
            ],
        },
        context,
    )
    assert result.is_error is True
    clause = next(row.strip() for row in result.text.splitlines() if "unchanged on disk" in row)
    assert len(clause) <= 70, clause
    assert clause.startswith("(") and "earlier hunks" in clause[:30], clause


@pytest.mark.asyncio
async def test_edit_consumed_hunk_says_its_text_is_still_on_disk(tools, context, tmp_path) -> None:
    """Review round 2, R10: "not found" must not sit beside 100% of your own text.

    Hunk 1 rewrites the paragraph in memory; hunk 2 sends the original wording.
    The disk still holds hunk 2's text exactly, so the report has to name the
    real situation — an earlier hunk of this call consumed it.
    """
    paragraph = "AAA BBB CCC the target paragraph\n"
    path = tmp_path / "consumed.txt"
    path.write_text(paragraph)
    edits = [
        {"old_text": paragraph, "new_text": "AAA BBB CCC the replacement paragraph\n"},
        {"old_text": paragraph, "new_text": "AAA BBB CCC another paragraph\n"},
    ]
    result = await _call(tools, "edit", {"path": "consumed.txt", "edits": edits}, context)
    assert result.is_error is True
    assert "100% word overlap" in result.text
    assert re.search(r"earlier hunk[^\n]*this call", result.text), result.text
    assert path.read_text() == paragraph


@pytest.mark.asyncio
async def test_edit_suppression_tail_uses_the_headers_word_for_a_mixed_batch(
    tools, context, tmp_path
) -> None:
    """Review round 2, R12: the tail does not name a reason the header avoided."""
    path = tmp_path / "mixed.txt"
    # "alpha" twice, so hunk 1 is ambiguous; then hunks that match nothing.
    path.write_text(
        "alpha\nalpha\nbeta\n" + "".join(f"line {i} of the document\n" for i in range(1, 20))
    )
    edits = [{"old_text": "alpha", "new_text": "ALPHA"}] + [
        {"old_text": f"absent hunk {i} about quokkas", "new_text": "y"} for i in range(8)
    ]
    result = await _call(tools, "edit", {"path": "mixed.txt", "edits": edits}, context)
    assert result.is_error is True
    assert "did not apply" in result.text and "matched more than one place" in result.text
    tail = result.text.splitlines()[-1]
    suppressed = 9 - builtin._EDIT_MAX_DETAILED_HUNKS - builtin._EDIT_MAX_COMPACT_HUNKS
    assert str(suppressed) in tail, tail
    assert "refused" in tail, tail
