"""A command an agent ran may not open a session of its own — unless the session
that ran it may delegate.

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

They also pin the 2026-09-19 relaxation, which is a DIFFERENCE between the two
entry points and would otherwise read as an oversight: a DELEGATING shell (one
whose live inventory holds `task`) may open `exec` sessions, because that is how
an agent told to fully delegate work reaches the shape a `task` child cannot be
— a separate, long-lived top-level session; the interactive path stays refused
for EVERY agent shell, because an agent has no terminal and that path opens a
front end on the operator's screen rather than a session. An allowed `exec` is
stamped exactly as an escaped one is, and that stamp is now the ONLY thing
keeping it out of the picker, the sidebar and the phone list.
"""

from __future__ import annotations

import json
import os
import shlex
import sys
from pathlib import Path
from typing import cast

import pytest

from local_operator.agent_shell import (
    AGENT_SHELL_ENV,
    ALLOW_NESTED_SESSION_ENV,
    MAY_DELEGATE_ENV,
    exec_session_refusal,
    in_agent_shell,
    interactive_session_refusal,
    may_delegate_from_shell,
    nested_session_allowed,
    refusal_message,
    stamp_agent_shell_session,
)
from local_operator.cli import main as cli_main
from local_operator.harness.types import AbortSignal, ToolContext, ToolResult
from local_operator.resume import (
    ORIGIN_AGENT_SHELL,
    ORIGIN_NAME,
    is_user_session,
    session_origin,
)
from local_operator.session.session import Session
from local_operator.tools import builtin


@pytest.fixture
def in_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    """The harness's own shape: an agent's bash tool set the marker."""
    monkeypatch.setenv(AGENT_SHELL_ENV, "1")


@pytest.fixture
def escaped(monkeypatch: pytest.MonkeyPatch, in_agent: None) -> None:
    """The documented escape, set the way a QA run sets it."""
    monkeypatch.setenv(ALLOW_NESTED_SESSION_ENV, "1")


@pytest.fixture
def delegating(monkeypatch: pytest.MonkeyPatch, in_agent: None) -> None:
    """An agent shell whose session HOLDS `task` — the other allowed shape.

    Set the way the `bash` tool sets it (from the session's live inventory), so a
    guard test reaches the allowance without standing up a real manager session.
    The derivation itself is pinned separately, against a real child.
    """
    monkeypatch.setenv(MAY_DELEGATE_ENV, "1")


# --- the predicate and the message -------------------------------------------


def test_without_the_marker_a_session_may_be_opened() -> None:
    """The operator's own terminal is the normal case and is untouched."""
    assert in_agent_shell() is False
    # BOTH entry points: with no marker there is nothing for the allowance or the
    # asymmetry to act on, which is why this pair is the baseline the rest is
    # measured against.
    assert exec_session_refusal() is None
    assert interactive_session_refusal() is None


def test_the_marker_refuses_and_names_what_to_do_instead(in_agent: None) -> None:
    """A refusal that only says no sends the reader back where it started.

    The three routes are the ones that exist: `task` for a session that holds
    it, `hub` for the non-delegating child this guard was written for (the one
    that cannot launch anything itself), and `wake` for work that belongs later
    — named as the DELEGATING session's to arm for a child, because `wake` is
    pruned from every child session, and as the reader's own for a root session,
    which holds it (review rounds 1 and 2, F3 and R2-2). The escape hatch is the
    one thing the text must NOT name — see the next test.
    """
    message = exec_session_refusal()
    assert message is not None
    # ONE message for both readers. The sentence was reworded for the
    # relaxation's sake from "a `lop` invocation" to "this `lop` invocation":
    # a delegating session reaching the INTERACTIVE path now reads a text that
    # must not claim the CLI is closed to it, when `exec` is in fact open to it.
    # Pinning the equality is what stops the two entry points drifting into two
    # texts (the same defect, one level up).
    assert message == interactive_session_refusal()
    assert "top-level" in message
    assert "`task`" in message
    assert "`hub`" in message
    assert "`wake`" in message
    # The last sentence may not route the reader to a tool it does not hold.
    # `wake` is pruned from EVERY child regardless of role, so "put it in
    # `wake`" sent it looking for a tool that is not in its set — the same
    # class of dead end the refusal exists to end (review round 1, F3). The
    # sentence that fixed that was wrong the OTHER way (review round 2, R2-2):
    # the marker rides every bash call, so a ROOT session holding `wake` reads
    # this text too, and there is no session that delegated to it. Both readers
    # are named now, and this pins both halves rather than one.
    assert "belongs in `wake`" not in message
    assert "`wake` is pruned from every child" in message
    assert "not a child session's to arm" in message
    assert "a session that holds `wake`" in message
    assert message.endswith("arms it there itself.")


def test_the_exec_doc_quotes_the_refusal_byte_for_byte() -> None:
    """``docs/EXEC.md``'s quoted copy is the message, not a paraphrase of it.

    The quote is what a human — and the next agent — reads to recognise the
    refusal in a terminal, and a reworded quote cannot be grepped for: the
    copy this replaces had already drifted from the function ("Launch
    delegated work with" against "Delegated work is launched with") and kept
    routing the reader to `wake` after the message stopped doing so. Pinned
    rather than spot-checked, so the next reword cannot leave one behind.
    """
    doc = Path(__file__).resolve().parents[2] / "docs" / "EXEC.md"
    text = doc.read_text()
    marker = "```\nexec failed: "
    start = text.index(marker)
    end = text.index("```", start + len(marker))
    assert text[start + 4 : end] == f"exec failed: {refusal_message()}\n"


def test_the_refusal_does_not_teach_the_bypass(in_agent: None) -> None:
    """NEITHER marker is named: the escape is for the human, not the model.

    The text exists to route the reader to `task`/`hub`/`wake`; naming the
    variable that switches the rule off would hand the reader it was written
    for the one thing it must not do. The delegation allowance is in the same
    class for a sharper reason — a reader told how the second route is spelled
    could go looking for a shell that carries it, which is the guard talking
    itself out of the job twice over.
    """
    message = exec_session_refusal()
    assert message is not None
    assert ALLOW_NESTED_SESSION_ENV not in message
    assert MAY_DELEGATE_ENV not in message


def test_the_escape_allows_the_run(in_agent: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """The ONE thing that waives both entry points, for a pty QA harness."""
    monkeypatch.setenv(ALLOW_NESTED_SESSION_ENV, "1")
    assert nested_session_allowed() is True
    assert exec_session_refusal() is None
    assert interactive_session_refusal() is None


def test_the_allowance_is_read_from_the_marker_alone(
    in_agent: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Absent means NO, and the rest of the spellings match the marker's.

    Absent is the case that carries the weight: it is what a child built by an
    older runtime, a command that scrubbed its own environment, and a tool double
    with no context all look like, and every one of them must be refused rather
    than admitted.
    """
    monkeypatch.delenv(MAY_DELEGATE_ENV, raising=False)
    assert may_delegate_from_shell() is False
    for value in ("1", "true", "yes", "on", "TRUE"):
        monkeypatch.setenv(MAY_DELEGATE_ENV, value)
        assert may_delegate_from_shell() is True
    for value in ("", "0", "off", "no", "maybe"):
        monkeypatch.setenv(MAY_DELEGATE_ENV, value)
        assert may_delegate_from_shell() is False


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


@pytest.mark.parametrize("extra", [["--background"], []])
def test_exec_from_a_delegating_agent_shell_reaches_the_runner(
    delegating: None, monkeypatch: pytest.MonkeyPatch, extra: list[str]
) -> None:
    """The 2026-09-19 relaxation: a session holding `task` may open sessions.

    This is the case the operator asked for — an agent told to fully delegate
    work, or to fan out a large number of long-lived workstreams, needs real
    top-level sessions, and a `task` child (one prompt, then gone) is not that.

    Asserted by observing the RUNNER, for the same reason the refusal case is:
    an exit code alone cannot distinguish "allowed" from "refused after doing
    nothing". The foreground arm additionally stubs the two preflight steps the
    refusal case never reaches — which is worth noticing, because it means a
    change that moved the preflight ABOVE the guard would leave the refusal test
    green and only this one red.
    """
    seen: list[object] = []
    monkeypatch.setattr(
        "local_operator.exec_mode.run_exec", lambda command, args: seen.append(command) or 0
    )
    monkeypatch.setattr(
        "local_operator.exec_mode.resolve_hosting_model_dry",
        lambda args: ("test", "test-model"),
    )
    monkeypatch.setattr("local_operator.cli._preflight_api_key", lambda *a, **k: None)
    monkeypatch.setattr("local_operator.cli.setup_cross_platform_environment", lambda: None)
    monkeypatch.setattr(
        "sys.argv", ["local-operator", "exec", *extra, "--profile", "reviewer", "review it"]
    )

    assert cli_main() == 0
    assert seen == ["review it"], "the guard let the run through to the runner"


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
    # The refusal sits AHEAD of the config override and the autosave-agent
    # lookup, so a refused run leaves the store untouched as well — not merely
    # unconversationed. Asserted against the scratch HOME the suite redirects.
    assert not (Path(os.environ["HOME"]) / ".local-operator").exists()


def test_the_interactive_path_refuses_a_delegating_shell_too(
    delegating: None, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    """The asymmetry, pinned where it would otherwise read as an oversight.

    An agent has no terminal, so this path opens a front end on the OPERATOR's
    screen — not the separate top-level session the relaxation is about. Both
    markers are set here, which is the point: the allowance must not leak from
    the entry point it was granted for. Driven against the factory, like the
    non-delegating case, so "refused" means no session was built rather than no
    exit code was printed.
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


def test_the_interactive_path_honours_the_escape(
    escaped: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A QA run that must drive the real front end is let through.

    Asserted by what happens NEXT — the config manager, the first thing past the
    guard — because "allowed" has no exit code to observe: the run continues
    into a session build a test cannot stand up. The stub raises ``SystemExit``
    on purpose: ``cli.main`` catches ``Exception`` (and would report the failure
    as its own), while ``SystemExit`` is a ``BaseException`` and arrives here.
    """
    reached: list[int] = []

    def stop(*args: object, **kwargs: object) -> None:
        reached.append(1)
        raise SystemExit(0)

    monkeypatch.setattr("local_operator.cli.setup_cross_platform_environment", lambda: None)
    monkeypatch.setattr("local_operator.cli.ConfigManager", stop)
    monkeypatch.setattr("sys.argv", ["local-operator"])

    with pytest.raises(SystemExit):
        cli_main()
    assert reached == [1], "an escaped run must reach the config manager, not the guard"


# --- the two environments around the marker ----------------------------------


def test_a_front_end_opening_a_conversation_drops_the_marker(delegating: None) -> None:
    """`/fork`, a notification click and a restart are the USER's gestures.

    They re-exec `lop --resume` with the session's own environment, so the child
    would inherit the answer to a question about its parent and the window would
    die on the refusal. Everything else in the environment rides along: this
    helper drops two keys, and a copy that also dropped PATH would break the
    spawn in a way no assertion here would notice.

    BOTH markers, and the allowance is the half that matters now: the new window
    is a SESSION, so it must not be handed its parent's permission — a child that
    later ran `lop exec` would be admitted on an allowance nobody granted IT.
    """
    from local_operator.agent_shell import without_agent_shell_marker

    stamped = without_agent_shell_marker(os.environ)
    assert AGENT_SHELL_ENV not in stamped
    assert MAY_DELEGATE_ENV not in stamped
    assert stamped.get("PATH") == os.environ.get("PATH")


def test_a_harness_declares_itself_to_its_children() -> None:
    """The benches and the eval driver drive the real CLI as a subprocess.

    Run from an agent's shell they inherit the marker, so without this every
    inner invocation would be refused and a bench would record "the product is
    broken" instead of numbers (review round 1, F2). The marker is deliberately
    left in place: this describes an environment, it does not rewrite the
    caller's identity.
    """
    from local_operator.agent_shell import harness_child_env

    child = harness_child_env({AGENT_SHELL_ENV: "1", "PATH": "/usr/bin"})
    assert child[ALLOW_NESTED_SESSION_ENV] == "1"
    assert child[AGENT_SHELL_ENV] == "1"
    assert child["PATH"] == "/usr/bin"
    # The caller's own environment is not read when one is passed in, and is the
    # base when it is not.
    assert harness_child_env()[ALLOW_NESTED_SESSION_ENV] == "1"


def test_the_click_rungs_do_not_hand_a_child_the_marker(
    delegating: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The CALLERS, not just the helper (review round 2, F4).

    The suite's click recorder records argv and stdin and never the child env,
    so a silent regression to `dict(os.environ)` at either rung would keep every
    test green — which is exactly what round 2's F1 was. Both rungs are driven:
    the terminal one through the backend it hands the environment to, and the
    desktop one through `Popen`, where its launcher lands.
    """
    import subprocess

    from local_operator.tui import resume_click

    # The suite's ambient fixture sets this deliberately — the launch ladder
    # finds the maintainer's installed app and would start it — so a test that
    # drives the rung clears it, which is the documented opt-in.
    monkeypatch.delenv("LOCAL_OPERATOR_NO_DESKTOP_LAUNCH", raising=False)

    class _Backend:
        def __init__(self) -> None:
            self.envs: list[dict[str, str]] = []

        def spawn(self, launch, env):  # noqa: ANN001
            self.envs.append(dict(env))
            return True

    backend = _Backend()
    monkeypatch.setattr("local_operator.spawn.registry.active_backend", lambda *a, **k: backend)
    assert resume_click._spawn_terminal("a1b2c3d4e5f6") is True

    launched: list[dict[str, str]] = []

    class _Process:
        def wait(self, timeout=None) -> int:
            return 0

    monkeypatch.setattr(
        subprocess,
        "Popen",
        lambda argv, **kwargs: launched.append(dict(kwargs.get("env") or {})) or _Process(),
    )
    # A CONFIGURED launcher, so the rung has a candidate on every platform. Without
    # it the ladder asks `shutil.which("local-operator-ui")`, which finds the
    # maintainer's install on macOS and nothing on the ubuntu runner — where the
    # rung returns False and this assertion went red in CI (round 7, M2's
    # neighbour). The rung's own discovery order is `tests/unit/test_resume_click.py`'s.
    monkeypatch.setattr(
        resume_click, "_configured_launch_command", lambda: ["/usr/bin/env", "true"]
    )
    assert resume_click._launch_desktop("a1b2c3d4e5f6") is True

    assert backend.envs, "the terminal rung must have handed a backend an env"
    assert launched, "the desktop rung must have launched something"
    for env in (*backend.envs, *launched):
        assert AGENT_SHELL_ENV not in env
        assert MAY_DELEGATE_ENV not in env
        # The strip drops TWO keys: a copy that also dropped PATH would break the
        # spawn in a way no assertion here would otherwise notice.
        assert env.get("PATH") == os.environ.get("PATH")


def test_the_fork_window_drops_the_marker() -> None:
    """The third caller, pinned as a shape rather than a spelling.

    `/fork`'s spawn needs a real session factory, a real backend and a running
    app, so it is asserted as the call that must be there rather than as a pilot
    run — the shape `tests/unit/test_fork.py` already uses for the fork paths it
    cannot drive. What it catches is the defect round 2's F1 named: the call kept
    and its result discarded, with plain `os.environ` handed to the spawn.

    PARSED rather than substring-matched (round 4, F3), and what it asserts about
    the environment is the ARGUMENT rather than the spelling (round 5, F1). The
    spawn rides `asyncio.to_thread(backend.spawn, launch, <env>)`, so the
    environment is that call's last argument; it must be
    `without_agent_shell_marker(os.environ)`, called inline or bound to a name
    first. The bound form is the equivalent refactor a text pin called a
    regression, and a bound form whose value is anything else — `{}`,
    `os.environ.copy()`, a rebinding of the name — is the helper's result
    discarded, which is the defect this replaces an `os.environ`-shaped string
    check for.
    """
    import ast
    import inspect
    import textwrap

    from local_operator.tui.app import OperatorApp

    source = inspect.getsource(OperatorApp._on_fork_complete)
    # Dedented because `getsource` hands back a method body still carrying its
    # class-level indentation, which `ast.parse` rejects outright.
    tree = ast.parse(textwrap.dedent(source))

    def strips_the_marker(call: ast.expr) -> bool:
        """`without_agent_shell_marker(os.environ)`, and nothing else."""
        if not isinstance(call, ast.Call):
            return False
        func = call.func
        if not (isinstance(func, ast.Name) and func.id == "without_agent_shell_marker"):
            return False
        if len(call.args) != 1:
            return False
        arg = call.args[0]
        return (
            isinstance(arg, ast.Attribute)
            and arg.attr == "environ"
            and isinstance(arg.value, ast.Name)
            and arg.value.id == "os"
        )

    # EVERY binding of a name in the method must come from the helper. Order- and
    # scope-insensitive on purpose, and that is the correction round 6 asked for:
    # the spawn sits inside a `try:`, so its bindings are not direct children of
    # the method body, and `ast.walk` yields a nested function's body after the
    # method's own statements — so a shadowing binding that never runs, or a
    # rebinding after the first, could mark a name stripped while the spawn
    # received the raw environment. Conjunction rather than last-wins: a name
    # bound to anything else, anywhere, is not trusted at the spawn.
    #
    # EVERY BINDING FORM, too (rounds 7 M2 and 8 M2): `with … as name`,
    # `name |= …`, `for name in …`, `(name := …)`, `head, *name = …`,
    # `except … as name` and `match …: case name:` all bind a name that reaches
    # the spawn, and none of them binds it to the helper.
    bound_to_helper: dict[str, bool] = {}

    def bind(target: ast.expr, value_from_helper: bool) -> None:
        names = target.elts if isinstance(target, (ast.Tuple, ast.List)) else [target]
        for name in names:
            if isinstance(name, ast.Starred):
                # `head, *env = (…,)` binds `env` to a LIST, never to the helper.
                bind(name.value, False)
            elif isinstance(name, ast.Name):
                prior = bound_to_helper.get(name.id, True)
                bound_to_helper[name.id] = prior and value_from_helper

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                bind(target, strips_the_marker(node.value))
        elif isinstance(node, ast.AnnAssign):
            # `x: dict[str, str]` with no value is an annotation, not a binding.
            if node.value is not None:
                bind(node.target, strips_the_marker(node.value))
        elif isinstance(node, ast.AugAssign):
            bind(node.target, False)
        elif isinstance(node, ast.withitem) and node.optional_vars is not None:
            bind(node.optional_vars, False)
        elif isinstance(node, (ast.For, ast.AsyncFor)):
            bind(node.target, False)
        elif isinstance(node, ast.NamedExpr):
            bind(node.target, False)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound_to_helper[node.name] = False
        elif isinstance(node, ast.MatchAs) and node.name:
            bound_to_helper[node.name] = False
        elif isinstance(node, ast.MatchStar) and node.name:
            bound_to_helper[node.name] = False
        elif isinstance(node, ast.MatchMapping) and node.rest:
            bound_to_helper[node.rest] = False

    def drops_the_marker(env: ast.expr) -> bool:
        """The environment the spawn is handed is stripped, inline or via a name."""
        if strips_the_marker(env):
            return True
        return isinstance(env, ast.Name) and bound_to_helper.get(env.id, False)

    # Both shapes a spawn can take: the threaded dispatch, and a direct
    # `backend.spawn(launch, env)` — the environment is the LAST argument each way.
    dispatches: list[ast.expr] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        func = node.func
        if not isinstance(func, ast.Attribute):
            continue
        if func.attr == "to_thread":
            first = node.args[0]
            if isinstance(first, ast.Attribute) and first.attr == "spawn":
                dispatches.append(node.args[-1])
        elif func.attr == "spawn":
            dispatches.append(node.args[-1])

    assert dispatches, "the fork window's spawn must be pinned here"
    problems: list[str] = []
    for env in dispatches:
        if not drops_the_marker(env):
            problems.append(ast.dump(env))
    assert not problems, (
        "the fork window inherits the session's environment, so the marker has to be "
        f"dropped at the spawn: pass `without_agent_shell_marker(os.environ)`, got {problems}"
    )


def test_an_escaped_run_is_marked_as_machine_started(escaped: None, tmp_path: Path) -> None:
    """The seatbelt under the hatch.

    An opted-in run that lands in the operator's own store must not become a
    chat they appear to have opened, so the session is stamped with the origin
    every listing already filters on: `is_user_session` is False, which is what
    keeps it out of the picker, the sidebar and the phone's list.
    """
    directory = tmp_path / "sessions" / "abc123"
    assert stamp_agent_shell_session(directory, created_here=True) is True
    assert session_origin(directory) == ORIGIN_AGENT_SHELL
    assert is_user_session(directory) is False
    # Distinct from a `task` child's stamp: this session carries no parent job
    # and no role, so a reader can tell the two apart from the marker alone.
    assert json.loads((directory / ORIGIN_NAME).read_text()) == {"origin": ORIGIN_AGENT_SHELL}


def test_an_allowed_delegating_run_is_marked_as_machine_started(
    delegating: None, tmp_path: Path
) -> None:
    """The relaxation's seatbelt, and the reason it is not a hole.

    An allowed `lop exec` is not an exotic route any more — an agent reaches it
    by following the guide — so if the stamp had stayed attached to the escape
    hatch alone, a manager fanned out to five workstreams would refill the
    operator's sidebar with five chats they never opened. That is the original
    incident, reproduced through the route this change opens.

    Same VALUE and the same predicate as the escaped case, on purpose: a reader
    asking "did a command from an agent's shell open this?" must not have to
    check which route did it.
    """
    directory = tmp_path / "sessions" / "abc124"
    assert stamp_agent_shell_session(directory, created_here=True) is True
    assert session_origin(directory) == ORIGIN_AGENT_SHELL
    assert is_user_session(directory) is False


def test_an_ordinary_run_is_not_marked(tmp_path: Path) -> None:
    """The operator's own `lop exec` keeps its plain user-session shape."""
    directory = tmp_path / "sessions" / "def456"
    assert stamp_agent_shell_session(directory, created_here=True) is False
    assert not (directory / ORIGIN_NAME).exists()
    assert is_user_session(directory) is True


def test_a_directory_this_call_did_not_create_is_never_re_marked(
    escaped: None, tmp_path: Path
) -> None:
    """`--resume` adopts the operator's conversation; hiding it is the mirror bug.

    Only the caller can tell a directory it just made from one the operator has
    been using, which is why `created_here` is passed rather than guessed from
    the marker's absence.
    """
    existing = tmp_path / "sessions" / "the-operator-session"
    existing.mkdir(parents=True)
    (existing / "transcript.jsonl").write_text("the operator's work\n")

    assert stamp_agent_shell_session(existing, created_here=False) is False
    assert not (existing / ORIGIN_NAME).exists()
    assert is_user_session(existing) is True


def test_a_resumed_conversation_is_never_re_marked_for_a_delegating_shell(
    delegating: None, tmp_path: Path
) -> None:
    """`--resume` is the operator's conversation however the shell is allowed.

    The delegating route makes this case LIKELIER than the escape did: a manager
    told to fan work out may reasonably `--resume` a session it opened earlier,
    and hiding a chat the operator has used because a manager touched it is the
    mirror of the bug the stamp exists for. `created_here` is what carries that,
    and it is asserted here for the route the relaxation added rather than
    assumed to follow from the escaped one.
    """
    existing = tmp_path / "sessions" / "the-managers-session"
    existing.mkdir(parents=True)
    (existing / "transcript.jsonl").write_text("a conversation\n")

    assert stamp_agent_shell_session(existing, created_here=False) is False
    assert not (existing / ORIGIN_NAME).exists()
    assert is_user_session(existing) is True


def _factory_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **overrides: object):
    """Build a session the way every entry point does, through the factory.

    It is typed as `SessionProtocol` (what `create_session` returns), which
    declares no transcript, so a test that reads one narrows with
    `assert isinstance(session, Session)` first — the same narrowing
    `tests/unit/test_session_factory.py` uses. The mock provider keeps this
    offline.
    """
    import argparse

    from local_operator.agents import AgentRegistry
    from local_operator.config import ConfigManager
    from local_operator.credentials import CredentialManager
    from local_operator.session_factory import create_session

    config_dir = tmp_path / ".local-operator"
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config_dir))
    args = argparse.Namespace(
        hosting="test",
        model="test-model",
        agent_name=None,
        agent_id=None,
        yolo=True,
        train=False,
        **overrides,
    )
    return create_session(
        args,
        ConfigManager(config_dir),
        CredentialManager(config_dir),
        AgentRegistry(config_dir),
    )


@pytest.mark.asyncio
async def test_the_factory_stamps_the_session_an_escape_opens(
    escaped: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The stamp is not exec-only — round 2's major, pinned where it lives.

    It began in `exec_session.run_session`, which the interactive path never
    reaches: a pty harness driving the TUI under the escape produced an
    UNSTAMPED session while both documents promised otherwise. The factory is
    the one place every entry point gets its directory (foreground exec, the
    detached worker, the interactive viewer's runtime, the server), so the
    guarantee is asserted through IT rather than through any single caller. The
    mock provider keeps this offline.
    """
    session = await _factory_session(tmp_path, monkeypatch)
    assert isinstance(session, Session)
    try:
        directory = session._transcript.directory
        assert session_origin(directory) == ORIGIN_AGENT_SHELL
        assert is_user_session(directory) is False
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_the_factory_leaves_a_resumed_conversation_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The other half: resuming the operator's own chat must not hide it.

    Built WITHOUT the escape first, so it is an ordinary user session, then
    reopened with it — the shape a harness re-entering an existing conversation
    has, and the one where a careless stamp would make the operator's own work
    vanish from their picker.
    """
    monkeypatch.delenv(AGENT_SHELL_ENV, raising=False)
    monkeypatch.delenv(ALLOW_NESTED_SESSION_ENV, raising=False)
    first = await _factory_session(tmp_path, monkeypatch)
    assert isinstance(first, Session)
    try:
        directory = first._transcript.directory
        session_id = first.session_id
        # A real turn, so the control is a conversation with something in it.
        # NOT a resumability requirement — under the adopt flag set below the
        # resolver returns before `resume_dir` is reached, and the property this
        # control needs is the one the adopt branch keys on: its directory
        # already exists (review round 5, F2; round 6 re-raised it because the
        # first fix was claimed in the commit message and never landed).
        await first.prompt("the operator's own turn")
    finally:
        await first.dispose()
    assert session_origin(directory) == "", "the control case starts as the user's"

    monkeypatch.setenv(AGENT_SHELL_ENV, "1")
    monkeypatch.setenv(ALLOW_NESTED_SESSION_ENV, "1")
    # The runtime's own flag, so this drives the ADOPT branch — the one the
    # round-3 mutation targeted and the one a phone's first message takes. Without
    # it `resume=<id>` goes through strict `resume_dir`, and the test proves a
    # neighbouring branch (review round 4, F2).
    monkeypatch.setenv("LOP_RUNTIME_ADOPT_SESSION", "1")
    resumed = await _factory_session(tmp_path, monkeypatch, resume=session_id)
    assert isinstance(resumed, Session)
    try:
        # FIRST that it really resumed: without this, a regression that mints a
        # fresh id instead of adopting the requested one keeps the origin
        # assertions below green while inspecting a directory the run never
        # touched (review round 3, F3).
        assert Path(resumed._transcript.directory) == directory
        assert session_origin(directory) == "", "a resumed conversation is not re-marked"
        assert is_user_session(directory) is True
    finally:
        await resumed.dispose()


@pytest.mark.asyncio
async def test_an_adopted_id_the_escape_created_is_stamped(
    escaped: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The runtime's engage shape — round 3's major, pinned.

    Under `LOP_RUNTIME_ADOPT_SESSION` the id resolver CREATES the session
    directory itself, one frame before the stamp is handed a freshly read
    `exists()` — which therefore answered "no" for every adopted session, so the
    phone's first message and the desktop draft came out unstamped while the
    docs named their lists. The answer now travels from the frame that knows.
    """
    monkeypatch.setenv("LOP_RUNTIME_ADOPT_SESSION", "1")
    session = await _factory_session(tmp_path, monkeypatch, resume="feedface0001")
    assert isinstance(session, Session)
    try:
        directory = session._transcript.directory
        assert directory.name == "feedface0001"
        assert session_origin(directory) == ORIGIN_AGENT_SHELL
        assert is_user_session(directory) is False
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_an_adopted_id_that_already_existed_is_left_to_its_owner(
    escaped: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The other side of `is_new`: a directory the caller did not make is not ours.

    This is the desktop plane's draft — the app creates the session directory
    before the runtime is ever asked for the id — so the chat in it is the
    operator's own and must stay in their picker. Marking it would be the
    mirror-image bug, and it is the case `is_new` has to keep false rather than
    "any escaped run's session" being stamped wholesale.
    """
    monkeypatch.setenv("LOP_RUNTIME_ADOPT_SESSION", "1")
    directory = tmp_path / ".local-operator" / "sessions" / "deadbeef0001"
    directory.mkdir(parents=True)
    (directory / "created_at.json").write_text("1.0", encoding="utf-8")

    session = await _factory_session(tmp_path, monkeypatch, resume="deadbeef0001")
    assert isinstance(session, Session)
    try:
        assert Path(session._transcript.directory) == directory
        assert session_origin(directory) == ""
        assert is_user_session(directory) is True
    finally:
        await session.dispose()


# --- the allowance's source and its writer -----------------------------------


def _tool_names(session: Session) -> set[str]:
    """The live inventory, which is what "may this session delegate" means."""
    return {tool.name for tool in session._tools}


@pytest.mark.asyncio
async def test_the_factory_stamps_the_session_an_allowed_delegating_run_opens(
    delegating: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The relaxation's route through the real FACTORY, not through the helper.

    The helper is pinned above; this is the wiring, and the two are different
    failures — a stamp that stopped being CALLED on the delegating route would
    leave the sidebar listing manager-spawned chats while every helper-level
    assertion stayed green. Same shape as the escaped case one section up, which
    exists for the same reason (round 2, F1).
    """
    session = await _factory_session(tmp_path, monkeypatch)
    assert isinstance(session, Session)
    try:
        directory = session._transcript.directory
        assert session_origin(directory) == ORIGIN_AGENT_SHELL
        assert is_user_session(directory) is False
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_the_tool_context_derives_may_delegate_from_the_live_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`task` present is the answer, and the LAUNCHER is not — asserted together.

    Those two are easy to confuse and the confusion is expensive: the context
    carries `subagent_launcher` on EVERY session, including a child whose role
    had `task` pruned, so a guard that read the launcher would hand every
    non-delegating reviewer and coder the permission this relaxation limits to
    sessions holding `task`. Driven through `self._tools`, the one attribute both
    `harness.subagent`'s prune and `Session._filter_declared` write, so this
    fails if either the pruning or the derivation moves.
    """
    session = await _factory_session(tmp_path, monkeypatch)
    assert isinstance(session, Session)
    try:
        assert "task" in _tool_names(session), "a root session holds `task` to delegate with"
        assert session._build_tool_context().may_delegate is True
        # The pruned shape, produced the way the prune produces it.
        session._tools = [tool for tool in session._tools if tool.name != "task"]
        context = session._build_tool_context()
        assert context.may_delegate is False
        assert (
            context.subagent_launcher is not None
        ), "the launcher survives the prune, which is exactly why it is not the signal"
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_both_poles_of_the_allowance_through_the_real_prune(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A `coder` child reads False and a `manager` child reads True, for real.

    `_build_child_session` is where a role's allowance becomes an inventory, so
    it is the only place that can prove the two agree; a hand-edited `_tools`
    list (the test above) pins the derivation, and this pins its SOURCE. Both the
    field and the inventory are asserted, deliberately: a field that stopped
    agreeing with the tools the model is actually offered would otherwise pass on
    the strength of the other pole's assertion alone.

    The PROFILE is resolved the way `run_subagent` resolves it and passed in,
    because that is the contract `_construct_child_session` states — and the
    difference is visible: with no profile the prune takes the role-less branch
    and an inheriting child keeps whatever its parent held, so a probe that
    forgot `profile=` here would report `True` for both roles and pin nothing.
    """
    from local_operator.harness.subagent import _build_child_session, _resolve_role

    parent = await _factory_session(tmp_path, monkeypatch)
    assert isinstance(parent, Session)
    try:
        for role, expected in (("coder", False), ("manager", True)):
            profile = _resolve_role(role, parent)
            assert profile is not None, f"the packaged {role} starter must resolve"
            assert profile.may_delegate is expected, "the role is the premise of the prune"
            child = await _build_child_session(
                label=f"probe-{role}",
                prompt="do a thing",
                parent_session=parent,
                model_spec=None,
                job_id=f"probe-job-{role}",
                agent=role,
                profile=profile,
            )
            try:
                assert (
                    "task" in _tool_names(child)
                ) is expected, f"the {role} role's inventory is the premise of the field"
                assert child._build_tool_context().may_delegate is expected
            finally:
                await child.dispose()
    finally:
        await parent.dispose()


def _stdout_block(result: ToolResult) -> str:
    """The stdout section of a tool result, without the wrapper or the exit line."""
    assert not result.is_error, result.text
    text = result.text
    start = text.find("--- stdout ---")
    end = text.find("--- stderr ---")
    return text[start + len("--- stdout ---") : end if end > start else None]


#: A command that asks the GUARD what it thinks, in the child itself. Reading the
#: variable's raw value would answer a weaker question now that the writer signs
#: it in three arms — an empty `= ` and an absent name are the same verdict, and
#: only the reader can say so.
_READER_PROBE = f"{shlex.quote(sys.executable)} -c " + shlex.quote(
    "from local_operator.agent_shell import may_delegate_from_shell as m; print(m())"
)

#: Three reads of the CHILD's own state, because they are three questions and the
#: middle one is what R2-1 turned on:
#:
#: * ``RAW`` — the marker's raw value, ``<ABSENT>`` when the name is not in the
#:   child's environment AT ALL. An empty value and an absent name are the same
#:   VERDICT (see `_READER_PROBE`) and opposite DISCLOSURES, and only this read
#:   answers the disclosure.
#: * ``NO_ENV_CHILD`` — the same read taken from a GRANDCHILD spawned with no
#:   ``env=``, which is the shape `exec_mode`'s detached ``--background`` worker
#:   uses. It proves the chain the fix has to hold one hop further down: a
#:   worker that inherits its parent's environment inherits the CLEARED answer,
#:   not the allowance.
#: * ``GUARD`` — the verdict `may_delegate_from_shell` reaches in that child, so
#:   a test cannot pass on the injection dict while the policy between the dict
#:   and the child disagrees with it.
_ENV_PROBE_SOURCE = """
import os, subprocess, sys
from local_operator.agent_shell import MAY_DELEGATE_ENV as k, may_delegate_from_shell as m

v = os.environ.get(k)
print("RAW=" + ("<ABSENT>" if v is None else repr(v)))
_nested = "import os;v=os.environ.get({k});print('<ABSENT>' if v is None else repr(v))"
child = subprocess.run(
    [sys.executable, "-c", _nested.format(k=repr(k))], capture_output=True, text=True
)
print("NO_ENV_CHILD=" + child.stdout.strip())
print("GUARD=" + str(m()))
"""
_ENV_PROBE = f"{shlex.quote(sys.executable)} -c " + shlex.quote(_ENV_PROBE_SOURCE)


async def _child_state(context: ToolContext | None) -> tuple[str, str, str]:
    """(raw marker value, no-``env=`` grandchild's value, guard verdict) in the child."""
    result = await builtin.execute_bash(
        "bash-allow", {"command": _ENV_PROBE}, AbortSignal(), None, context
    )
    lines = dict(line.split("=", 1) for line in _stdout_block(result).strip().splitlines())
    return lines["RAW"], lines["NO_ENV_CHILD"], lines["GUARD"]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["inherit", "allowlist"])
async def test_the_bash_tool_signs_the_allowance_where_the_guard_reads_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    """`execute_bash` is the ONE writer of the second marker, in THREE arms, in
    both `shell_env` modes.

    Each arm is a different bug, and the third is the one round 2 found a naive
    writer creating:

    * HELD — the session holds `task`. The child gets ``1``, and so the guard in
      the `lop` it starts admits it: without this arm the fix would be a blanket
      deny.
    * INHERITED, NOT HELD (R1-1) — the ``bash`` child's environment starts as a
      copy of this process's own in the default ``inherit`` mode, so a writer that
      only SET the variable leaves a session reading ``True`` on an allowance
      nobody granted it. Not hypothetical: an allowed `lop exec` runs `lop` — and,
      for `--background`, a detached worker spawned with no ``env=`` — as a child
      of the shell that carries the marker. Hence the assertion on BOTH the raw
      value and the no-``env=`` grandchild: the clear has to be what a worker
      inherits, not just what the immediate child sees.
    * NEVER HAD IT, NOT HELD (R2-1) — the name must be ABSENT, not empty. The
      marker's NAME is the mechanism, which is why `agent_shell.refusal_message`
      never names it; writing the empty value unconditionally would export the
      spelling into every ``bash`` child, and a `coder`-shaped session running
      `env` is then one inference from `LOCAL_OPERATOR_AGENT_MAY_DELEGATE=1 lop
      exec`. An absent name is also a verdict the guard already reaches
      (``_on(get(k, ""))``), so nothing is lost by omitting it — and this is the
      arm a test written against the guard's verdict alone cannot see.

    Both `shell_env` modes, because the policy sits between the injection dict
    and the child: in `allowlist` the marker is named back in `inherit` on
    purpose, which is the shape where the policy's own re-grant is what the
    injection has to beat. Driven over the real handler and the real interpreter,
    and every answer is read from the child itself.
    """
    from local_operator.tools import shell_env

    policy = shell_env.ShellEnvironmentPolicy(
        mode=mode, inherit=(MAY_DELEGATE_ENV,) if mode == shell_env.MODE_ALLOWLIST else ()
    )
    monkeypatch.setattr(shell_env, "load_policy", lambda: policy)

    def context_allowing(allowed: bool) -> ToolContext:
        return ToolContext(cwd=str(tmp_path), may_delegate=allowed)

    # NEVER HAD IT: the name is absent from the child AND from the worker it
    # spawns, and both read "no".
    monkeypatch.delenv(MAY_DELEGATE_ENV, raising=False)
    assert await _child_state(context_allowing(False)) == ("<ABSENT>", "<ABSENT>", "False")

    # HELD: signed `1`, all the way down.
    assert await _child_state(context_allowing(True)) == ("'1'", "'1'", "True")

    # INHERITED, NOT HELD: cleared — a clear, not a blanket deny, so the same
    # inherited value is still honoured when the context says the session holds
    # `task`.
    monkeypatch.setenv(MAY_DELEGATE_ENV, "1")
    assert await _child_state(context_allowing(False)) == (
        "''",
        "''",
        "False",
    ), "an inherited allowance must be CLEARED, not merely not re-set"
    assert await _child_state(context_allowing(True)) == ("'1'", "'1'", "True")

    # No context at all: the same call the loop makes when a host has no session.
    # Fail closed in both directions — the (loop with no host) shape and the
    # (inherited marker) shape read the same way, which is the point of clearing
    # rather than merely not setting.
    assert await _child_state(None) == ("''", "''", "False")
    monkeypatch.delenv(MAY_DELEGATE_ENV, raising=False)
    assert await _child_state(None) == ("<ABSENT>", "<ABSENT>", "False")


@pytest.mark.asyncio
async def test_a_context_without_the_field_does_not_raise(tmp_path: Path) -> None:
    """Q1's shape, pinned: a duck-typed context is not `None`.

    ``tests/e2e/test_inline_credential_e2e.py`` builds exactly this — an object
    with the two attributes ``execute_bash`` needs and nothing else — and a bare
    ``context.may_delegate`` made every `bash` call through it fail with an
    ``AttributeError``, which broke two e2e tests on both CI platforms. The field
    is read with ``getattr`` for that reason, and the FAIL-CLOSED contract covers
    this shape as well as ``None``: no field means "may not delegate", and the
    command still RUNS.
    """

    class _Ctx:
        variables = None
        cwd = str(tmp_path)

    # `cast`, not a `# type: ignore`: the point of the double is that it is NOT a
    # ToolContext, and the argument type is what the real caller passes. Casting
    # says that out loud instead of suppressing the check that would otherwise
    # have caught the bare attribute read for us.
    double = cast(ToolContext, _Ctx())
    result = await builtin.execute_bash(
        "bash-duck", {"command": _READER_PROBE}, AbortSignal(), None, double
    )
    assert _stdout_block(result).strip() == "False"


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
    monkeypatch.setenv(MAY_DELEGATE_ENV, "1")
    monkeypatch.setenv("KEEP_THIS", "yes")

    reexec.replace_self(reexec.make_plan(["lop"], resume_id="sess-1"))

    assert AGENT_SHELL_ENV not in captured["env"]
    # Both, for the reason the helper's own docstring gives: a restarted session
    # is a session, and the allowance it might carry is its PARENT's answer.
    assert MAY_DELEGATE_ENV not in captured["env"]
    assert captured["env"]["KEEP_THIS"] == "yes"


def test_replace_self_leaves_an_ordinary_environment_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    from local_operator import reexec

    captured: dict[str, dict[str, str]] = {}
    monkeypatch.setattr(reexec, "_replace_posix", lambda argv, env: captured.update(env=env))

    reexec.replace_self(reexec.make_plan(["lop"], resume_id="sess-1"))

    assert AGENT_SHELL_ENV not in captured["env"]
    assert captured["env"].get("PATH") == __import__("os").environ.get("PATH")
