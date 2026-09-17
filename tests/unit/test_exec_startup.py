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
from local_operator.exec_startup import (
    apply_startup,
    report_unresolved_declared_tools,
    resolve_startup,
)
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


# --------------------------------------------------------------------------
# --tools: the declared tool inventory (lop-harness-gaps)
# --------------------------------------------------------------------------


class RecordingSession:
    """The smallest host that can answer the inventory questions.

    A hand-written double rather than ``Mock()``, because the behaviour under
    test is partly "a host that cannot answer is not an error": a ``Mock``
    fabricates ``attached_profile_tools`` and ``tool_inventory`` on demand, and
    the startup path has to survive that (see ``_name_tuple``).
    """

    def __init__(self, reachable=("read",), attached=()):
        self._reachable = tuple(reachable)
        self._attached = tuple(attached)
        self._declared: set[str] = set()
        self.declared: list[tuple[tuple[str, ...], bool]] = []
        self.attached_profiles: list[str] = []

    def attach_team(self, team):
        pass

    def attach_agent_profile(self, name):
        self.attached_profiles.append(name)
        return name

    @property
    def attached_profile_tools(self):
        return self._attached

    @property
    def tool_inventory(self):
        return self._reachable

    def unresolved_declared_tools(self):
        # The real Session answers this (see its docstring): an absent name is a
        # typo unless MCP is still connecting. This host has no MCP at all.
        if not self._declared:
            return ()
        return tuple(name for name in sorted(self._declared) if name not in self._reachable)

    def set_tool_inventory(self, names, *, unattended=False):
        self._declared = set(names)
        self.declared.append((tuple(names), unattended))
        allowed = set(names)
        self._reachable = tuple(name for name in self._reachable if name in allowed)

    def set_goal(self, text):
        pass

    def set_conversation_name(self, text):
        pass


@pytest.mark.parametrize(
    "text,expected",
    [
        (None, None),
        ("read", ("read",)),
        ("read,grep", ("read", "grep")),
        (" read , grep ", ("read", "grep")),
        ("read,read,grep", ("read", "grep")),
        ("", ()),
        (",,", ()),
    ],
)
def test_parse_tool_inventory(text, expected):
    from local_operator.exec_startup import parse_tool_inventory

    assert parse_tool_inventory(text) == expected


def test_empty_tools_declaration_is_refused_at_preflight():
    """``None`` and ``()`` are different answers: one leaves the session
    unrestricted, the other strands it with nothing to reach. A declaration the
    operator typed and got nothing for would read as a harness bug."""
    with pytest.raises(ValueError, match="at least one tool"):
        resolve_startup(ExecArgs(tools=""))


def test_tools_declares_the_inventory_and_approves_it_when_unattended():
    session = RecordingSession(reachable=("read", "bash", "write"))
    apply_startup(session, ExecArgs(tools="read"), team=None)
    assert session.declared == [(("read",), True)]
    assert session.tool_inventory == ("read",)


def test_a_supervised_run_declares_the_inventory_but_leaves_the_gate_alone():
    """``--control`` means a supervisor can answer, so a declared tool must still
    be ASKED about — the supervisor's gate replaces the session's, and an
    approval granted by declaration here would bypass it."""
    session = RecordingSession(reachable=("read", "bash"))
    apply_startup(session, ExecArgs(tools="read", control=True), team=None)
    assert session.declared == [(("read",), False)]


def test_an_attached_roles_allow_list_bounds_the_run(tmp_path):
    """``--profile reviewer`` used to stamp instructions only: the reviewer seed's
    ``tools:`` allow-list was enforced solely where the profile is launched as a
    SUBAGENT, so a headless reviewer could still write the patch it was asked to
    review."""
    session = RecordingSession(reachable=("read", "bash", "edit"), attached=("read", "bash"))
    apply_startup(session, ExecArgs(profile="reviewer"), team=None)
    assert session.attached_profiles == ["reviewer"]
    assert session.declared == [(("read", "bash"), True)]
    assert "edit" not in session.tool_inventory


def test_tools_wins_over_the_attached_roles_allow_list(tmp_path):
    """The explicit flag states THIS run's reach; a role's allow-list is a weaker
    statement about the role."""
    session = RecordingSession(reachable=("read", "bash"), attached=("read", "bash"))
    apply_startup(session, ExecArgs(profile="reviewer", tools="read"), team=None)
    assert session.declared == [(("read",), True)]


def test_a_role_declaring_no_tools_leaves_the_run_unrestricted():
    """The negative case: today's behaviour, byte for byte. Most seeds declare no
    ``tools:`` of their own, and a session they bound to nothing must not become
    a session that can reach nothing."""
    session = RecordingSession(reachable=("read", "bash", "edit"))
    apply_startup(session, ExecArgs(profile="coder"), team=None)
    assert session.declared == []


def test_no_declaration_at_all_never_touches_the_inventory():
    session = RecordingSession(reachable=("read", "bash"))
    apply_startup(session, ExecArgs(name="Audit"), team=None)
    assert session.declared == []


def test_a_host_that_cannot_answer_the_inventory_questions_is_not_an_error(capsys):
    """A reduced host fabricates any attribute it is asked for — ``getattr`` with
    a default never fires — so the startup path must treat "cannot say" as
    "declares nothing" rather than iterating a fabricated object."""
    session = Mock()
    apply_startup(session, ExecArgs(profile="reviewer"), team=None)
    assert session.set_tool_inventory.call_count == 0
    assert "Traceback" not in capsys.readouterr().err


def test_a_declared_name_that_matches_nothing_is_reported_at_the_end_of_the_run(capsys):
    """Fail-closed is right, silent is not: with no matching tool the run answers
    nothing and every call returns "Tool not found", which reads as a harness
    fault rather than as the typo it is.

    Reported at the END of the run — MCP servers connect in the background, so at
    startup an unreachable name and a server that is still connecting are
    indistinguishable."""
    session = RecordingSession(reachable=("read", "grep"))
    args = ExecArgs(tools="read,reed,grep")
    apply_startup(session, args, team=None)
    assert capsys.readouterr().err == ""  # nothing claimed before the run settles

    report_unresolved_declared_tools(session, args)
    assert "Warning: --tools names no tool this run could reach: reed." in capsys.readouterr().err


def test_a_roles_unmatched_tool_is_not_reported(capsys):
    """Deliberately silent for the role-derived case: a profile naming a tool that
    exists on another machine is ordinary (see ``agent_profiles.filter_tools``)."""
    session = RecordingSession(reachable=("read",), attached=("read", "mcp__elsewhere_tool"))
    apply_startup(session, ExecArgs(profile="reviewer"), team=None)
    report_unresolved_declared_tools(session, ExecArgs(profile="reviewer"))
    assert capsys.readouterr().err == ""


def test_worker_argv_carries_the_tool_declaration():
    """``--background`` is the same request run elsewhere: a declaration dropped
    at the process boundary would leave the worker unrestricted while the
    launcher reported a bounded run."""
    argv = build_worker_argv("t", ExecArgs(tools="read,mcp__vendor_screen"))
    assert "--tools=read,mcp__vendor_screen" in argv


def test_the_worker_parser_accepts_the_tool_declaration():
    from local_operator.exec_worker import build_parser

    parsed = build_parser().parse_args(["--prompt=p", "--tools=read,mcp__vendor_screen"])
    assert parsed.tools == "read,mcp__vendor_screen"
