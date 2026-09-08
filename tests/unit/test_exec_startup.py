"""Bounded startup adapters retain actual team/profile and loop semantics."""

import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from local_operator.exec_mode import (
    ExecArgs,
    build_worker_argv,
    job_status,
    resolve_prompt,
)
from local_operator.exec_startup import apply_startup, resolve_startup
from local_operator.session.goal_loop import GoalLoop
from local_operator.teams import TeamEditFields, TeamMember, TeamRegistry


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    import os

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "config"))
    for key in list(os.environ):
        if key.startswith("CMUX_"):
            monkeypatch.delenv(key)


def test_real_team_preflight_and_attachment(tmp_path):
    team = TeamRegistry(tmp_path / "config").create_team(
        TeamEditFields(
            name="release",
            manager="manager",
            members=[TeamMember(role="coder")],
            instructions="review first",
            project="headless work",
        )
    )
    args = ExecArgs(team="release", profile="reviewer", goal="clear", name="Audit")
    resolved = resolve_startup(args)
    assert resolved.id == team.id
    session = Mock()
    apply_startup(session, args, resolved)
    session.attach_team.assert_called_once_with(resolved)
    session.attach_agent_profile.assert_called_once_with("reviewer")
    session.set_goal.assert_called_once_with("clear")
    session.set_conversation_name.assert_called_once_with("Audit")


@pytest.mark.parametrize(
    "args,message",
    [
        (ExecArgs(team="missing"), "No team"),
        (ExecArgs(profile="missing"), "No role"),
        (ExecArgs(agent_name="a", agent_id="b"), "mutually exclusive"),
        (ExecArgs(loop=0), "between"),
        (ExecArgs(loop=26), "between"),
        (ExecArgs(loop=1, loop_goal="g"), "mutually exclusive"),
        (ExecArgs(goal="g", clear_goal=True), "mutually exclusive"),
        (ExecArgs(loop_goal="  "), "must not be empty"),
    ],
)
def test_preflight_invalid(args, message):
    with pytest.raises(ValueError, match=message):
        resolve_startup(args)


def test_clear_and_resumed_goal_validation():
    session = SimpleNamespace(goal="", set_goal=Mock())
    apply_startup(session, ExecArgs(clear_goal=True), None)
    session.set_goal.assert_called_once_with("")
    with pytest.raises(ValueError, match="standing goal"):
        apply_startup(session, ExecArgs(loop=1), None)


def test_startup_roundtrip_worker_arguments():
    from local_operator.exec_worker import build_parser

    args = ExecArgs(
        team="release",
        profile="coder",
        goal="literal",
        loop=2,
        name="Night shift",
        effort="high",
        control=True,
    )
    argv = build_worker_argv("literal /team nope", args)
    worker_argv = argv[argv.index("local_operator.exec_worker") + 1 :]
    parsed = build_parser().parse_args(worker_argv)
    for key in ("team", "profile", "goal", "loop", "name", "effort", "control"):
        assert getattr(parsed, key) == getattr(args, key)
    assert parsed.prompt == "literal /team nope"


def test_dash_leading_startup_values_survive_the_worker_argv_hop():
    """Every value-carrying option must use the ``--opt=value`` form.

    Forwarded as two argv items, a value beginning with ``-`` reads as the next
    OPTION and the worker dies at ``parse_args`` — before ``--job-id`` is
    honoured, so no terminal ledger row is written and the run is durably
    mislabelled ``interrupted``. The free-text options this feature adds are
    exactly the ones a user is likely to lead with a dash.
    """
    from local_operator.exec_worker import build_parser

    args = ExecArgs(
        team="-team",
        profile="-role",
        goal="-- verify every criterion",
        name="-nightly",
        effort="-high",
        agent_name="-agent",
        agent_id="-id",
        hosting="-host",
        model="-model",
        resume="-sess",
    )
    argv = build_worker_argv("-- do the thing", args)
    worker_argv = argv[argv.index("local_operator.exec_worker") + 1 :]
    parsed = build_parser().parse_args(worker_argv)
    assert parsed.prompt == "-- do the thing"
    assert parsed.goal == "-- verify every criterion"
    assert parsed.name == "-nightly"
    assert (parsed.team, parsed.profile, parsed.effort) == ("-team", "-role", "-high")
    assert (parsed.agent, parsed.agent_id) == ("-agent", "-id")
    assert (parsed.hosting, parsed.model, parsed.resume) == ("-host", "-model", "-sess")


def test_stdin_forms_preserve_literal_text():
    assert resolve_prompt(None, stdin_text="/team literal\n") == "/team literal"
    assert resolve_prompt("-", stdin_text="/team literal\n") == "/team literal"
    assert resolve_prompt("/team literal", stdin_text="ignored") == "/team literal"


def test_loop_only_run_never_reads_an_inherited_stdin(monkeypatch):
    """A loop-only run has DECLARED it has no prompt, so it must not read stdin.

    Without this, an omitted positional fell through to an unbounded
    ``sys.stdin.read()`` on any non-TTY stdin — and a pipe whose writer stays
    open never sends EOF, so the documented ``exec --goal X --loop N`` hung
    forever under any supervisor that hands its child an inherited pipe.
    """
    import sys as _sys

    class _NeverEnds:
        """Any read is the bug: a real inherited pipe would block here."""

        def isatty(self):
            return False

        def read(self):
            raise AssertionError("read a stdin that a loop-only run must not touch")

    monkeypatch.setattr(_sys, "stdin", _NeverEnds())
    assert resolve_prompt(None, has_loop=True) == ""
    # An explicit `-` is the user ASKING for stdin, so it must still read.
    assert resolve_prompt("-", stdin_text="piped") == "piped"


@pytest.mark.asyncio
async def test_loop_literal_numeric_goal_and_restored_state_do_not_replay():
    prompts = []
    states = []

    async def prompt(text):
        prompts.append(text)

    async def judge(text):
        return "VERDICT: ACHIEVED\nVerified"

    driver = GoalLoop(prompt, judge, lambda: None, states.append)
    driver.state = {"status": "running", "completed": 9}
    await asyncio.sleep(0)
    assert prompts == []
    driver.start("", "", goal_override="123")
    assert driver.task is not None
    await driver.task
    assert len(prompts) == 1 and "123" in prompts[0]
    assert states[-1]["status"] == "achieved"


def test_durable_status_does_not_regress_on_late_spawn_row(monkeypatch):
    from local_operator import exec_mode

    monkeypatch.setattr(
        exec_mode,
        "read_job_records",
        lambda: [
            {"id": "j", "status": "running", "session_id": "s"},
            {"id": "j", "status": "succeeded", "exit_code": 0},
            {"id": "j", "status": "starting", "log_path": "/log", "finished_at": None},
        ],
    )
    status = job_status("j")
    assert status["status"] == "succeeded"
    assert status["session_id"] == "s" and status["log_path"] == "/log"


def test_generation_reconciliation_marks_interrupted_not_success(monkeypatch):
    from local_operator import exec_mode

    monkeypatch.setattr(
        exec_mode,
        "read_job_records",
        lambda: [
            {"id": "j", "status": "running", "pid": 123, "process_generation": "old"},
        ],
    )
    checked = []
    monkeypatch.setattr(
        "local_operator.tools.group_reaper._owner_is_dead",
        lambda pid, generation: checked.append((pid, generation)) or True,
    )
    assert job_status("j")["status"] == "interrupted"
    assert checked == [(123, "old")]
    assert exec_mode.read_job_records()[0]["status"] == "running"
