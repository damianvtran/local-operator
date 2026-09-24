"""The bash long-sleep guard: refuse a foreground call that is mostly a sleep.

Two properties, as for the search guard. It must fire on the MEASURED shapes —
the commands a child used on 2026-09-24 to poll a background pytest run while
its parent's hub notes waited out every sleep (session ``f7318cc06bdd``) — and
it must never fire on the sleeps a model writes legitimately: a short settle
after a kill, a readiness loop, a background job, a sleep the parser cannot
read as a literal. The allow list is the contract, so each shape is a named case.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from local_operator.harness.jobs import AsyncJobManager
from local_operator.harness.types import ToolContext
from local_operator.tools import builtin, sleep_guard

C = sleep_guard.check_long_sleep

#: Verbatim from the child's transcript (``~/.local-operator/sessions/
#: f7318cc06bdd/transcript.jsonl``), each a FOREGROUND bash call. The last two
#: are the calls the parent's notes arrived during and were only read after.
MEASURED = [
    'sleep 1500; tail -c 250 "$LOCAL_OPERATOR_SCRATCHPAD/suite-final2.log"; echo; '
    'echo "--- pyright ---"; tail -3 "$LOCAL_OPERATOR_SCRATCHPAD/pyright-final.log"; uptime',
    'sleep 1500; tail -c 250 "$LOCAL_OPERATOR_SCRATCHPAD/suite-final2.log"; echo; uptime',
    'sleep 1800; tail -c 200 "$LOCAL_OPERATOR_SCRATCHPAD/suite-final3.log"; echo; '
    "sysctl vm.swapusage | tail -1; uptime",
    'sleep 900; tail -c 400 "$LOCAL_OPERATOR_SCRATCHPAD/suite-final.log"; echo; echo "---"; uptime',
    'sleep 600; head -3 "$LOCAL_OPERATOR_SCRATCHPAD/suite-final3.log"; '
    'tail -c 200 "$LOCAL_OPERATOR_SCRATCHPAD/suite-final3.log"; echo; sysctl vm.swapusage; uptime',
    'sleep 300; echo "waited 5 min"; uptime',
    'sleep 180; cat "$LOCAL_OPERATOR_SCRATCHPAD/suite-final2.log"',
]


@pytest.mark.parametrize("command", MEASURED)
def test_the_measured_commands_from_the_live_child_are_refused(command: str) -> None:
    message = C(command)
    assert message is not None, command
    # The refusal names the replacement, not just the problem.
    assert "background: true" in message
    assert "`wait`" in message
    assert "jobs op='peek'" in message
    assert sleep_guard.ALLOW_ENV in message


#: The same child's SHORT sleeps, which must keep working.
MEASURED_ALLOWED = [
    "kill -TERM 23076 23031 2>/dev/null; sleep 3; ps -p 23076,23031 -o pid,command 2>&1 | tail -2",
    'cd ~/wt && pkill -f "wt" 2>/dev/null; sleep 2; pgrep -fl "pyright|pytest" | grep -c wt',
    "for i in $(seq 1 60); do curl -fsS http://127.0.0.1:13380/v1/capabilities >/dev/null "
    "&& break; sleep 1; done",
]


@pytest.mark.parametrize("command", MEASURED_ALLOWED)
def test_the_short_sleeps_from_the_same_child_still_run(command: str) -> None:
    assert C(command) is None, command


BLOCKED = [
    "sleep 121",
    "sleep 25m",
    "sleep 1h",
    "sleep infinity",
    "/bin/sleep 600",
    "sleep 60; sleep 70",  # the foreground total is what holds the call
    "cd x && sleep 600 && tail f",
    "sleep 1 && sleep 1000 && x",
    "while true; do tail f; done; sleep 900",  # after the loop closes: top level again
    "sleep 1500\ntail -c 250 log",
    "sleep 900 2>/dev/null; tail log",
    "sleep 900 | cat",  # the pipeline waits for its first stage
]


@pytest.mark.parametrize("command", BLOCKED)
def test_long_foreground_sleeps_are_refused(command: str) -> None:
    assert C(command) is not None, command


NEVER_BLOCKED = [
    # at or under the threshold
    "sleep 120",
    "sleep 2; pgrep x",
    "sleep 0.5",
    "sleep 1m",
    # backgrounded by the shell: holds nothing
    "sleep 900 &",
    "x & sleep 5",
    "(sleep 600; echo) &",
    "{ sleep 600; echo; } &",
    # inside a compound: may itself be backgrounded, and a loop sleep is a poll
    # interval the guard cannot weigh against the loop's own exit
    "while true; do sleep 300; done",
    "until test -f x; do sleep 300; done",
    "for i in 1 2; do\n sleep 900\ndone",
    "if true; then sleep 900; fi",
    # not a literal duration
    'sleep "$N"',
    "sleep $((60*30))",
    "sleep 1m 30x",
    # a LATER pipeline stage, and a `2>&1` the shared splitter reads as `&`
    "yes | sleep 900",
    "sleep 900 2>&1; tail log",
    # not a sleep command at all
    "timeout 900 sleep 900",
    'echo "sleep 900"',
    'git commit -m "sleep 900; tail"',
    "cat <<EOF\nsleep 900\nEOF",
    # the escape hatch, read per segment off the command itself
    f"{sleep_guard.ALLOW_ENV}=1 sleep 900; tail f",
]


@pytest.mark.parametrize("command", NEVER_BLOCKED)
def test_legitimate_sleeps_are_not_blocked(command: str) -> None:
    assert C(command) is None, command


def test_a_falsy_grant_does_not_open_the_hatch() -> None:
    assert C(f"{sleep_guard.ALLOW_ENV}=0 sleep 900") is not None


def test_an_inherited_grant_is_not_consulted(monkeypatch: pytest.MonkeyPatch) -> None:
    """The grant is per call, read off the command. An exported value would make
    it silently global and invisible in the transcript."""
    monkeypatch.setenv(sleep_guard.ALLOW_ENV, "1")
    assert C("sleep 900") is not None


def test_the_threshold_is_the_one_constant() -> None:
    limit = sleep_guard.LONG_SLEEP_THRESHOLD_SECONDS
    assert C(f"sleep {int(limit)}") is None
    assert C(f"sleep {int(limit) + 1}") is not None


# --- through the real bash tool --------------------------------------------


async def _bash(args: dict[str, object], context: ToolContext):
    tool = builtin.build_bash_tool()
    return await tool.execute("c", args, None, None, context)  # type: ignore[operator]


@pytest.mark.asyncio
async def test_the_bash_tool_refuses_before_spawning(tmp_path) -> None:
    """Refused at the top of ``execute_bash``: nothing runs, nothing waits."""
    marker = tmp_path / "ran"
    context = ToolContext(cwd=str(tmp_path), session_id="sg")
    started = time.perf_counter()
    result = await _bash({"command": f"touch {marker}; sleep 1500; tail -c 250 log"}, context)
    assert time.perf_counter() - started < 5.0
    assert result.is_error is True
    assert "sleeps 1500 s (25 min) in the foreground" in result.text
    assert not marker.exists()


@pytest.mark.asyncio
async def test_a_background_call_is_never_refused(tmp_path) -> None:
    """The replacement the refusal names must itself be allowed."""
    manager = AsyncJobManager()
    context = ToolContext(cwd=str(tmp_path), session_id="sg-bg", jobs=manager)
    result = await _bash({"command": "sleep 1500; echo done", "background": True}, context)
    try:
        assert result.is_error is False, result.text
        assert result.details is not None and result.details.get("backgrounded") is True
    finally:
        await manager.cancel(result.details["job_id"]) if result.details else None
        await manager.dispose()


@pytest.mark.asyncio
async def test_the_escape_hatch_runs_the_command(tmp_path) -> None:
    context = ToolContext(cwd=str(tmp_path), session_id="sg-hatch")
    # A long sleep would stall the test, so prove the grant on a command whose
    # sleep is long only on paper: the timeout ends it, and the result is a
    # timeout rather than the refusal.
    result = await asyncio.wait_for(
        _bash(
            {"command": f"{sleep_guard.ALLOW_ENV}=1 sleep 900", "timeout": 0.5},
            context,
        ),
        timeout=30,
    )
    assert "in the foreground" not in result.text
    assert result.text.startswith("TIMEOUT after 0.5s")
