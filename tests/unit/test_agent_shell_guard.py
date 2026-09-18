"""A command an agent ran may not open a session of its own.

The incident this answers (2026-09-18): a `coder` subagent owed a review round
on PR #1281, held no `task` tool — a role that does not delegate runs one level
deep — and reached for the CLI instead:

    lop exec --profile reviewer --yolo --background --name lo-1281-review \\
        < /tmp/reviewer_brief.md

Nothing about that command failed. What it produced was a TOP-LEVEL session by
every test the store applies — an ordinary conversation directory with no
``origin.json`` — so the operator's desktop sidebar listed it, with a `reviewer`
badge, as a chat they had opened. These tests pin both halves of the fix: the
refusal and the routes it names, and the origin stamp that keeps the documented
escape hatch for real-CLI testing from being a silent one.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from local_operator.agent_shell import (
    AGENT_SHELL_ENV,
    ALLOW_NESTED_SESSION_ENV,
    in_agent_shell,
    nested_session_allowed,
    nested_session_refusal,
    stamp_escaped_session,
)
from local_operator.cli import main as cli_main
from local_operator.resume import (
    ORIGIN_AGENT_SHELL,
    ORIGIN_NAME,
    is_user_session,
    session_origin,
)


@pytest.fixture
def in_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    """The harness's own shape: an agent's bash tool set the marker."""
    monkeypatch.setenv(AGENT_SHELL_ENV, "1")


@pytest.fixture
def escaped(monkeypatch: pytest.MonkeyPatch, in_agent: None) -> None:
    """The documented escape, set the way a QA run sets it."""
    monkeypatch.setenv(ALLOW_NESTED_SESSION_ENV, "1")


# --- the predicate and the message -------------------------------------------


def test_without_the_marker_a_session_may_be_opened() -> None:
    """The operator's own terminal is the normal case and is untouched."""
    assert in_agent_shell() is False
    assert nested_session_refusal() is None


def test_the_marker_refuses_and_names_what_to_do_instead(in_agent: None) -> None:
    """A refusal that only says no sends the reader back where it started.

    The three routes are the ones that exist: `task` for a session that holds
    it, `hub` for the non-delegating child this guard was written for (the one
    that cannot launch anything itself), and `wake` for work that belongs
    later. The escape hatch is the one thing the text must NOT name — see the
    next test.
    """
    message = nested_session_refusal()
    assert message is not None
    assert "top-level" in message
    assert "`task`" in message
    assert "`hub`" in message
    assert "`wake`" in message


def test_the_refusal_does_not_teach_the_bypass(in_agent: None) -> None:
    """The escape is for the human and for a documented QA run, not the model.

    The text exists to route the reader to `task`/`hub`/`wake`; naming the
    variable that switches the rule off would hand the reader it was written
    for the one thing it must not do.
    """
    message = nested_session_refusal()
    assert message is not None
    assert ALLOW_NESTED_SESSION_ENV not in message


def test_the_escape_allows_the_run(in_agent: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ALLOW_NESTED_SESSION_ENV, "1")
    assert nested_session_allowed() is True
    assert nested_session_refusal() is None


# --- the CLI: exec ------------------------------------------------------------


@pytest.mark.parametrize("extra", [["--background"], []])
def test_exec_from_an_agent_shell_never_reaches_the_runner(
    in_agent: None, monkeypatch: pytest.MonkeyPatch, capsys, extra: list[str]
) -> None:
    """The assertion that matters is `not called`.

    A refusal that still spawned the worker would leave the operator's chat
    list exactly as it was, so this test observes the runner rather than the
    exit code alone.
    """
    seen: list[object] = []
    monkeypatch.setattr(
        "local_operator.exec_mode.run_exec", lambda command, args: seen.append(command) or 0
    )
    monkeypatch.setattr("local_operator.cli.setup_cross_platform_environment", lambda: None)
    monkeypatch.setattr(
        "sys.argv", ["local-operator", "exec", *extra, "--profile", "reviewer", "review it"]
    )

    assert cli_main() == 1
    assert seen == []
    err = capsys.readouterr().err
    assert "exec failed:" in err
    assert "`task`" in err


def test_exec_status_is_not_a_session_and_still_reads(
    in_agent: None, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """`--status` opens nothing, so the guard must not shadow its own error.

    The read-only form is how a session polls a detached job; refusing it would
    break the very workflow (`lop exec --status <job>`) that the workaround this
    guard replaces depended on.
    """
    monkeypatch.setattr("local_operator.cli.setup_cross_platform_environment", lambda: None)
    monkeypatch.setattr("sys.argv", ["local-operator", "exec", "--status", "no-such-job"])

    assert cli_main() == 1
    err = capsys.readouterr().err
    assert "No exec job" in err
    assert "`task`" not in err


# --- the CLI: the interactive path -------------------------------------------


def test_the_interactive_path_refuses_before_building_a_session(
    in_agent: None, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """`lop`, `lop --resume ID` and `--tui` are one check: the fall-through.

    Asserted against the factory rather than the exit code, for the same reason
    as the exec case — a refused run that still built a conversation would be
    indistinguishable from the bug.
    """
    built: list[object] = []
    monkeypatch.setattr(
        "local_operator.session_factory.create_session",
        lambda *a, **k: built.append((a, k)) or _explode(),
    )
    monkeypatch.setattr("local_operator.cli.setup_cross_platform_environment", lambda: None)
    monkeypatch.setattr("sys.argv", ["local-operator"])

    assert cli_main() == 1
    assert built == []
    assert "`task`" in capsys.readouterr().err


def _explode() -> None:
    raise AssertionError("a session was built where the guard should have refused")


# --- the escape hatch is not silent ------------------------------------------


def _session_on(directory: Path) -> SimpleNamespace:
    """The one seam the stamp reads: the session's own directory."""
    return SimpleNamespace(transcript=SimpleNamespace(directory=directory))


def test_an_escaped_run_is_marked_as_machine_started(escaped: None, tmp_path: Path) -> None:
    """The seatbelt under the hatch.

    An opted-in run that lands in the operator's own store must not become a
    chat they appear to have opened, so the session is stamped with the origin
    every listing already filters on: `is_user_session` is False, which is what
    keeps it out of the picker, the sidebar and the phone's list.
    """
    directory = tmp_path / "sessions" / "abc123"
    assert stamp_escaped_session(_session_on(directory)) is True
    assert session_origin(directory) == ORIGIN_AGENT_SHELL
    assert is_user_session(directory) is False
    # Distinct from a `task` child's stamp: this session carries no parent job
    # and no role, so a reader can tell the two apart from the marker alone.
    assert json.loads((directory / ORIGIN_NAME).read_text()) == {"origin": ORIGIN_AGENT_SHELL}


def test_an_ordinary_run_is_not_marked(tmp_path: Path) -> None:
    """The operator's own `lop exec` keeps its plain user-session shape."""
    directory = tmp_path / "sessions" / "def456"
    assert stamp_escaped_session(_session_on(directory)) is False
    assert not (directory / ORIGIN_NAME).exists()
    assert is_user_session(directory) is True


def test_a_storeless_session_is_skipped(monkeypatch: pytest.MonkeyPatch, escaped: None) -> None:
    """A reduced host builds sessions with no directory; marking is best-effort."""
    assert stamp_escaped_session(SimpleNamespace(transcript=None)) is False


def test_run_session_stamps_a_fresh_escaped_run_but_never_a_resume(
    escaped: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The call site, not just the helper.

    `--resume` adopts a directory that may be the operator's OWN conversation,
    so hiding it would be the mirror image of the bug this marks against. The
    stamp is observed by intercepting the run at its next step, which is also
    what keeps this test out of the model path.
    """
    from local_operator import exec_session

    stamped: list[object] = []
    monkeypatch.setattr(
        "local_operator.agent_shell.stamp_escaped_session",
        lambda session: stamped.append(session) or True,
    )

    async def stop(*a, **k):
        raise _Intercepted

    monkeypatch.setattr("local_operator.session.runtime.exec_control.start_exec_control", stop)

    def args(resume):
        return SimpleNamespace(
            resume=resume,
            yolo=False,
            control=False,
            goal=None,
            clear_goal=False,
            loop=None,
            loop_goal=None,
            name=None,
            profile=None,
            team=None,
            effort=None,
        )

    async def drive(resume: str | None) -> None:
        session = SimpleNamespace(dispose=_noop)
        with pytest.raises(_Intercepted):
            await exec_session.run_session(session, "review it", args(resume), None)

    asyncio.run(drive(None))
    assert len(stamped) == 1
    asyncio.run(drive("some-user-session"))
    assert len(stamped) == 1, "a resumed conversation must not be re-marked"


async def _noop() -> None:
    return None


class _Intercepted(Exception):
    """Ends the run at the first step after the stamp."""


# --- a session restarting its own front end ----------------------------------


def test_replace_self_drops_the_marker(monkeypatch: pytest.MonkeyPatch) -> None:
    """A restart is the session relaunching ITSELF, not an agent's command.

    Without this, `cli.main` would refuse to re-open a session that was
    legitimately started from an agent's shell under the escape — the guard
    turning on the one run it had allowed.
    """
    from local_operator import reexec

    captured: dict[str, dict[str, str]] = {}
    monkeypatch.setattr(reexec, "_replace_posix", lambda argv, env: captured.update(env=env))
    monkeypatch.setenv(AGENT_SHELL_ENV, "1")
    monkeypatch.setenv("KEEP_THIS", "yes")

    reexec.replace_self(reexec.make_plan(["lop"], resume_id="sess-1"))

    assert AGENT_SHELL_ENV not in captured["env"]
    assert captured["env"]["KEEP_THIS"] == "yes"


def test_replace_self_leaves_an_ordinary_environment_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    from local_operator import reexec

    captured: dict[str, dict[str, str]] = {}
    monkeypatch.setattr(reexec, "_replace_posix", lambda argv, env: captured.update(env=env))

    reexec.replace_self(reexec.make_plan(["lop"], resume_id="sess-1"))

    assert AGENT_SHELL_ENV not in captured["env"]
    assert captured["env"].get("PATH") == __import__("os").environ.get("PATH")
