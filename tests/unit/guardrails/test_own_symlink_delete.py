"""Removing a symlink this session created (q1029 incident, remote node 0.101.44).

A session made ``ln -s <its scratchpad> ~/q1029`` as a short TMPDIR alias; a
later ``rm ~/q1029`` was refused with "the safety check's approval could not
be verified for <scratchpad>": the ln was never recorded, and the target was
ranked and verified on the POINTEE's realpath, which the creating call's
citation can never cover. ``rm LINK`` removes only the link entry.

HOME is laid out like a Linux node (``<tmp>/home/ec2-user``) with its own
state root and synthetic session ids; nothing touches the operator's dirs.
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from typing import Any

import pytest

from local_operator.guardrails import evidence as ev_mod
from local_operator.guardrails import provenance as prov
from local_operator.guardrails.check import CallContext, GuardSettings, check_tool_call
from local_operator.guardrails.classify import Risk, classify_bash
from local_operator.guardrails.resolve import resolve_command
from local_operator.harness.types import Message, ToolCall, Usage

SID = "aaaaaaaaaaaa"
OTHER = "cccccccccccc"


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "home" / "ec2-user"
    state = home / ".local-operator"
    for sid in (SID, OTHER):
        (state / "sessions" / sid / "scratchpad").mkdir(parents=True)
    (state / "config.yml").write_text("version: 1\n")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(state))
    for key in [k for k in os.environ if k.startswith(("CMUX_", "GIT_"))]:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", "/dev/null")
    ev_mod._TREE_CACHE.clear()
    return home


def _env(home: Path) -> dict[str, str]:
    return {"HOME": str(home), "LOCAL_OPERATOR_CONFIG_DIR": str(home / ".local-operator")}


def _pad(home: Path, sid: str = SID) -> str:
    return str(home / ".local-operator" / "sessions" / sid / "scratchpad")


def _run(home: Path, command: str, sid: str = SID) -> list[str]:
    """Run ``command`` for real between the loop's plan/commit hooks."""
    pending = prov.plan_creations(command, cwd=str(home), env=_env(home))
    subprocess.run(["bash", "-c", command], cwd=home, check=True, env={**os.environ})
    if pending is None:
        return []
    return prov.commit_creations(
        pending, session_id=sid, env=_env(home), scratchpad=_pad(home, sid)
    )


def _verdict(home: Path, command: str, sid: str = SID):
    return classify_bash(
        command, cwd=str(home), env=_env(home), scratchpad=_pad(home, sid), session_id=sid
    )


class _Judge:
    def __init__(self, answer: str = "ALLOW evidence=e1") -> None:
        self.answer = answer
        self.prompts: list[str] = []

    async def __call__(self, system: str, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.answer


def _history(command: str, at: float) -> list[Any]:
    call = Message(
        role="assistant",
        tool_calls=[ToolCall(id="c1", name="bash", arguments={"command": command})],
        usage=Usage(at_ms=int(at * 1000)),
    )
    return [call, Message(role="tool", tool_call_id="c1", content=[])]


async def _check(home: Path, command: str, judge: _Judge, history: list[Any] | None = None):
    history = history or []
    ctx = CallContext(
        cwd=str(home),
        env=_env(home),
        scratchpad=_pad(home),
        session_id=SID,
        intent="remove my TMPDIR alias",
        operator=prov.operator_brief(history, session_id=SID, cited=(command,)),
        messages=history,
    )
    return await check_tool_call(
        "bash", {"command": command}, ctx, judge=None, settings=GuardSettings(), forked_judge=judge
    )


def _link_cmd(home: Path, name: str = "q1029", flags: str = "-s") -> str:
    return f"ln {flags} {_pad(home)} {home / name}"


# -- resolver ----------------------------------------------------------------------


def test_resolver_records_only_the_two_operand_link_form(tmp_path: Path) -> None:
    def kinds(cmd: str) -> list[tuple[str, str | None]]:
        res = resolve_command(cmd, cwd=str(tmp_path), env={}, home=str(tmp_path))
        return [(c.kind, c.path) for c in res.created]

    assert kinds("ln -s /a b") == [("symlink", str(tmp_path / "b"))]
    assert kinds("ln -sfn /a b") == [("symlink", str(tmp_path / "b"))]
    assert kinds("ln /a b") == []  # a hard link is a file, not this rule's concern
    assert kinds("ln -s /a d/") == []  # placed INSIDE d
    assert kinds("ln -s -t d /a") == []
    assert kinds("ln -s /a /b d") == []


# -- the incident, end to end ---------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("precheck", [False, True])
async def test_own_link_removal_is_allowed_without_a_judge(home: Path, precheck: bool) -> None:
    time.sleep(2.1)  # the pad predates the ln call's window, as in the incident
    assert _run(home, _link_cmd(home)) == [str(home / "q1029")]
    rm = "rm $HOME/q1029"
    if precheck:
        rm = f'[ "$(readlink $HOME/q1029)" = "{_pad(home)}" ] && {rm}'
    verdict = _verdict(home, rm)
    assert verdict.risk is Risk.LOW
    assert "a symlink created by this session (symlink" in verdict.targets[0].reason
    judge = _Judge()
    outcome = await _check(home, rm, judge)
    assert outcome is None or not outcome.blocked
    assert judge.prompts == []


@pytest.mark.asyncio
async def test_history_alone_settles_own_link_on_the_link_not_the_pointee(home: Path) -> None:
    """No ledger row (e.g. the write failed): the cited ln call covers the LINK."""
    time.sleep(2.1)
    start = time.time()
    subprocess.run(["bash", "-c", _link_cmd(home)], check=True)
    judge = _Judge()
    outcome = await _check(home, "rm -f $HOME/q1029", judge, _history(_link_cmd(home), start))
    assert outcome is not None and not outcome.blocked and judge.prompts == []
    assert any(f"{home / 'q1029'} created by call c1" in w for w in outcome.warnings)


@pytest.mark.asyncio
async def test_foreign_link_stays_judged_and_unverifiable_citation_refuses(home: Path) -> None:
    os.symlink(_pad(home, OTHER), home / "q1029")
    rec = prov.write_records(
        [(str(home / "q1029"), "symlink")], owners=[OTHER], creator=OTHER, command="ln -s"
    )
    assert rec
    verdict = _verdict(home, "rm $HOME/q1029")
    assert verdict.risk is Risk.HIGH
    assert "a symlink at the top level of the home directory" in verdict.targets[0].reason
    # The judge cites a call that made nothing here: genuinely unverifiable.
    judge = _Judge("ALLOW evidence=e1")
    outcome = await _check(home, "rm $HOME/q1029", judge)
    assert outcome is not None and outcome.blocked
    assert f"could not be verified for {home / 'q1029'}" in outcome.text
    assert "only the link is removed" in outcome.text


def test_link_spellings_never_include_the_pointee(home: Path) -> None:
    # #3793's flake: re-casing the link entry opened it, so the pointee's
    # path became a "spelling" and ``_rank`` returned whichever reason the
    # set yielded first ("another session's files" on some hash seeds).
    from local_operator.guardrails.classify import _link_spellings

    os.symlink(_pad(home, OTHER), home / "q1029")
    link = prov.link_path(str(home / "q1029"))
    spelled = _link_spellings(str(home / "q1029"), link)
    real_pointee = os.path.realpath(_pad(home, OTHER))
    pointees = (real_pointee, _pad(home, OTHER))
    assert not any(s.startswith(pointees) for s in spelled)
    assert link in spelled


def test_spelling_order_is_stable_and_lexical_first() -> None:
    from local_operator.guardrails.classify import _spelling_order

    spelled = {"/z/real", "/a/Cased", "/m/typed"}
    # Python's set order varies with PYTHONHASHSEED; the order must not.
    assert _spelling_order("/m/typed", spelled) == ["/m/typed", "/a/Cased", "/z/real"]
    assert _spelling_order("/m/typed", set(reversed(sorted(spelled)))) == [
        "/m/typed",
        "/a/Cased",
        "/z/real",
    ]


@pytest.mark.asyncio
async def test_trailing_slash_ranks_and_verifies_the_pointee(home: Path) -> None:
    time.sleep(2.1)
    start = time.time()
    _run(home, _link_cmd(home))
    rm = "rm -r $HOME/q1029/"
    verdict = _verdict(home, rm)
    assert verdict.risk is Risk.HIGH
    assert verdict.targets[0].reason == "a top-level directory of the home directory"
    outcome = await _check(home, rm, _Judge(), _history(_link_cmd(home), start))
    assert outcome is not None and outcome.blocked
    assert f"could not be verified for {os.path.realpath(_pad(home))}" in outcome.text
    assert _verdict(home, "rm -rf $HOME/q1029/.").risk is Risk.HIGH


def test_link_replaced_by_a_directory_is_not_owned(home: Path) -> None:
    _run(home, _link_cmd(home))
    assert prov.owned_link(str(home / "q1029"), SID) is not None
    os.remove(home / "q1029")
    (home / "q1029").mkdir()
    assert prov.owned_link(str(home / "q1029"), SID) is None
    assert prov.owned_record(str(home / "q1029"), SID) is None
    assert _verdict(home, "rm -rf $HOME/q1029").risk is Risk.HIGH


def test_link_row_never_covers_the_pointee_or_paths_below(home: Path) -> None:
    _run(home, _link_cmd(home))
    assert prov.owned_record(_pad(home), SID) is None
    assert prov.owned_record(str(home / "q1029" / "x"), SID) is None


def test_ln_sf_overwrite_records_the_new_link(home: Path) -> None:
    os.symlink("/nonexistent-old", home / "alias")  # someone else's, unrecorded
    assert _verdict(home, "rm $HOME/alias").risk is Risk.HIGH
    assert _run(home, f"ln -sfn {_pad(home)} {home / 'alias'}") == [str(home / "alias")]
    assert _verdict(home, "rm $HOME/alias").risk is Risk.LOW
    # ``ln -sf`` through a link to a DIRECTORY (no -n) makes its link inside
    # the pointee: the alias entry is unchanged, so nothing is recorded for it.
    os.remove(home / "alias")
    (home / "real").mkdir()
    os.symlink(home / "real", home / "alias")
    assert _run(home, f"ln -sf /x {home / 'alias'}") == []
    assert _verdict(home, "rm $HOME/alias").risk is Risk.HIGH


def test_critical_link_paths_stay_critical_whoever_made_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home" / "ec2-user"
    real_state = tmp_path / "data" / "lop"
    (real_state / "sessions" / SID / "scratchpad").mkdir(parents=True)
    home.mkdir(parents=True)
    os.symlink(real_state, home / ".local-operator")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(real_state))
    env = {"HOME": str(home), "LOCAL_OPERATOR_CONFIG_DIR": str(real_state)}
    pad = str(real_state / "sessions" / SID / "scratchpad")
    for cmd in ("rm ~/.local-operator", "unlink ~/.local-operator", "rm -rf ~/.local-op*"):
        verdict = classify_bash(cmd, cwd=str(home), env=env, scratchpad=pad, session_id=SID)
        assert verdict.risk is Risk.CRITICAL, cmd
    # And the ledger never records a link AT a protected path.
    (real_state / "config.yml").write_text("v\n")
    pending = prov.plan_creations(f"ln -sfn /tmp {real_state / 'secrets'}", cwd=str(home), env=env)
    assert pending is not None
    os.symlink("/tmp", real_state / "secrets")
    assert prov.commit_creations(pending, session_id=SID, env=env, scratchpad=pad) == []


def test_empty_expansion_is_unchanged(home: Path) -> None:
    _run(home, _link_cmd(home))
    assert _verdict(home, "rm -rf $UNSET_VAR_X/").risk is Risk.CRITICAL
    assert _verdict(home, "rm -rf $HOME/$UNSET_VAR_X").risk is Risk.CRITICAL
    assert _verdict(home, 'rm -rf "$UNSET_VAR_X"').risk is Risk.CRITICAL
    # An empty expansion next to an OWNED link is still unresolved, never LOW.
    assert _verdict(home, 'rm -rf "$UNSET_VAR_X"/q1029').risk is Risk.HIGH
    assert _verdict(home, "rm -rf $HOME/q1029$UNSET_VAR_X").risk is Risk.HIGH
    assert _verdict(home, "rm -rf $HOME/q1029").risk is Risk.LOW  # the control
