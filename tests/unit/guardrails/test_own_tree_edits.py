"""A session's own tracked edits and clean own worktrees settle without the judge.

Operator (Damian, 2026-11-11): "you should be able to do this delete
autonomously without being blocked". Two field cases, both reproduced here in a
tmp HOME:

1. Child session a85e507ffdfa (coder job f73922ff5364), cwd its own worktree
   ``~/local-operator-ui/.worktrees/bridge-single-recipient``, ran
   ``for f in src/i18n/catalogues/*/ui.teamBridge.json; do sed -i '' '/…/d' $f;
   done`` and was refused for review with ONE target: the worktree ROOT,
   "contains an unresolvable substitution or variable (opaque)". The ``for``
   over a glob bound ``$f`` to an opaque marker, so the edit collapsed onto the
   cwd. Separately, BSD sed's ``-i ''`` made the resolver read the sed SCRIPT
   (or, with ``-e``, the empty suffix) as a file to overwrite.
2. Session 99c51054bdbd's ``git worktree remove --force`` of its own clean
   worktree for merged #1858 was refused twice: "PR state unknown (gh lookup
   timed out)" sent it to the judge, which then timed out after 20 s.

Rule decided here (documented in the guardrails guide): an in-place edit
rewrites file CONTENT, never removes a path, so it ranks like any write —
CRITICAL on the state root, ``config.yml``, ``secrets/`` and the guard's
evidence exactly as before, LOW elsewhere. That holds for a tracked file in an
own, foreign or primary checkout alike (git restores it) and for an untracked
one (unchanged: it was LOW before). What this file pins is that the parse no
longer invents a directory, a script or a suffix as the target.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
from pathlib import Path
from typing import Any

import pytest

from local_operator.guardrails import provenance as prov
from local_operator.guardrails.check import CallContext, GuardSettings, check_tool_call
from local_operator.guardrails.classify import Risk, classify_bash
from local_operator.guardrails.preview import preview_tool_call

#: Captured at import, before any fixture stubs it.
_REAL_GH_LOOKUP = prov._gh_open_prs

S = "a85e507ffdfa"  # the editing / removing session
S2 = "999999999999"  # an unrelated session
CATALOGUES = ("de", "en", "es", "fr", "ja", "pt", "ur", "zh")
UI = "~/local-operator-ui"
WT = f"{UI}/.worktrees/bridge-single-recipient"
DONE = f"{UI}/.worktrees/bridge-delivery-complete"

#: The field command, verbatim from session a85e507ffdfa.
FIELD_COMMAND = (
    f"cd {WT} && for f in src/i18n/catalogues/*/ui.teamBridge.json; do sed -i '' "
    "'/\"ui.teamBridge.delivery.summary.single.delivered\": /d' $f; done && "
    "sed -i '' '2581,2587d' src/i18n/catalogues/en/ui.teamBridge.context.json && "
    "sed -n 2578,2584p src/i18n/catalogues/en/ui.teamBridge.context.json && python3 - <<'E'\n"
    "import json,glob\n"
    "for f in glob.glob('src/i18n/catalogues/*/ui.teamBridge*.json'): json.load(open(f))\n"
    "print('json ok')\n"
    "E\n"
    "sed -i '' 's/hubLine({ total: 1, held: 1 }).text, FR_SINGLE_HELD)/hubLine({ total: 1, "
    'held: 1 }).text, "En attente d’examen")/\' scripts/team-bridge-delivery-annotation.test.mjs\n'
    "node scripts/i18n/generate.mjs 2>&1 | tail -2; git diff --stat; grep -rn "
    '"single.delivered" src scripts | grep -v "test.mjs:1009\\|test.mjs:1294"'
)


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "home"
    state = home / ".local-operator"
    for sid in (S, S2):
        (state / "sessions" / sid / "scratchpad").mkdir(parents=True)
    (state / "config.yml").write_text("version: 1\n")
    monkeypatch.setattr(prov, "_live_cwd_inside", lambda _: False)
    monkeypatch.setattr(prov, "_github_repo", lambda _: "damianvtran/local-operator-ui")
    monkeypatch.setattr(prov, "_gh_open_prs", lambda repo, branch: [])
    monkeypatch.setattr(prov, "_running_owners", lambda real, sid: ())
    monkeypatch.setattr(prov, "_cwd_inventory_cache", None)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(state))
    for key in [k for k in os.environ if k.startswith(("CMUX_", "GIT_", "LOP_"))]:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", "/dev/null")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", "/dev/null")
    monkeypatch.setenv("GIT_TERMINAL_PROMPT", "0")
    return home


def _env(home: Path) -> dict[str, str]:
    return {"HOME": str(home), "LOCAL_OPERATOR_CONFIG_DIR": str(home / ".local-operator")}


def _git(*args: str, cwd: Path) -> None:
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@e", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )


def _own(home: Path, wt: Path, sid: str = S) -> None:
    prov.write_records(
        [(os.path.realpath(wt), "worktree")],
        owners=[sid],
        creator=sid,
        command=f"git worktree add {wt}",
        state_root=home / ".local-operator",
    )


@pytest.fixture
def ui(home: Path) -> Path:
    """``~/local-operator-ui`` with the two field worktrees, both this session's."""
    repo = home / "local-operator-ui"
    repo.mkdir()
    _git("init", "-q", "-b", "main", cwd=repo)
    (repo / ".gitignore").write_text(".worktrees/\nnode_modules\n")
    for lang in CATALOGUES:
        cat = repo / "src" / "i18n" / "catalogues" / lang
        cat.mkdir(parents=True)
        (cat / "ui.teamBridge.json").write_text('{\n  "a": "b"\n}\n')
    (repo / "src/i18n/catalogues/en/ui.teamBridge.context.json").write_text("{}\n")
    (repo / "scripts").mkdir()
    (repo / "scripts" / "team-bridge-delivery-annotation.test.mjs").write_text("//\n")
    _git("add", ".", cwd=repo)
    _git("commit", "-qm", "init", cwd=repo)
    for name, branch in (
        ("bridge-single-recipient", "fix/bsr"),
        ("bridge-delivery-complete", "fix/bdc"),
    ):
        wt = repo / ".worktrees" / name
        _git("worktree", "add", "-q", "-b", branch, str(wt), cwd=repo)
        _own(home, wt)
    return repo


class _Judge:
    """Counts prompts; ``delay`` past the settings timeout models a slow judge."""

    def __init__(self, answer: str = "BLOCK: should not be asked", delay: float = 0.0) -> None:
        self.answer = answer
        self.delay = delay
        self.prompts: list[str] = []

    async def __call__(self, system: str, prompt: str) -> str:
        self.prompts.append(prompt)
        if self.delay:
            await asyncio.sleep(self.delay)
        return self.answer


#: 50 ms: a judge sleeping 1 s always times out — the field's 20 s in miniature.
_SLOW = GuardSettings(judge_timeout_s=0.05)


async def _check(home: Path, command: str, judge: _Judge, *, sid: str = S) -> Any:
    ctx = CallContext(
        cwd=str(home),
        env=_env(home),
        scratchpad=None,
        session_id=sid,
        intent="editing catalogues in my worktree",
        operator=prov.operator_brief([], session_id=sid, cited=(command,)),
        messages=[],
    )
    return await check_tool_call(
        "bash", {"command": command}, ctx, judge=None, settings=_SLOW, forked_judge=judge
    )


def _classify(home: Path, command: str, sid: str = S) -> Any:
    return classify_bash(command, cwd=str(home), env=_env(home), scratchpad=None, session_id=sid)


# -- 1. in-place edits ---------------------------------------------------------


@pytest.mark.asyncio
async def test_field_command_in_own_worktree_runs_with_zero_judge_calls(
    home: Path, ui: Path
) -> None:
    """Red on #3800's head: one HIGH target, the worktree root (opaque)."""
    v = _classify(home, FIELD_COMMAND)
    assert v.risk is Risk.LOW, [(t.target.path, t.reason) for t in v.targets]
    root = os.path.realpath(ui / ".worktrees" / "bridge-single-recipient")
    assert all(t.target.path != root for t in v.targets), "the directory is never the target"
    patterns = [t.target.pattern for t in v.targets if t.target.pattern]
    assert any(p.endswith("src/i18n/catalogues/*/ui.teamBridge.json") for p in patterns)
    judge = _Judge(delay=1.0)
    outcome = await _check(home, FIELD_COMMAND, judge)
    assert outcome is None or not outcome.blocked, outcome and outcome.text
    assert judge.prompts == []


def test_eight_catalogue_line_range_delete_lists_files_not_the_script(home: Path, ui: Path) -> None:
    files = " ".join(f"src/i18n/catalogues/{c}/ui.teamBridge.json" for c in CATALOGUES)
    for command in (
        f"cd {WT} && sed -i '' '/\"start\": {{/,/^  }},/d' {files}",
        f"cd {WT} && sed -i '' -e '/start/,/end/d' {files}",
        # An unset variable inside the SCRIPT used to rank it HIGH as a path.
        f"cd {WT} && sed -i '' \"/$KEY/,/end/d\" {files}",
    ):
        v = _classify(home, command)
        assert v.risk is Risk.LOW, (command, [(t.target.path, t.reason) for t in v.targets])
        paths = sorted(t.target.path or "" for t in v.targets)
        assert len(paths) == len(CATALOGUES), (command, paths)
        assert all(p.endswith("/ui.teamBridge.json") for p in paths), (command, paths)


def test_bsd_suffix_only_skips_a_literal_empty_word(home: Path, ui: Path) -> None:
    """``-i "$SUF"`` may name a file under GNU sed: it stays a target."""
    v = _classify(home, f'cd {WT} && sed -i "$SUF" /x/d src/i18n/catalogues/en/ui.teamBridge.json')
    # Not skipped as a suffix: the next word keeps its old reading as a file.
    assert "/x/d" in [t.target.path for t in v.targets], v.targets
    literal = _classify(
        home, f"cd {WT} && sed -i '' /x/d src/i18n/catalogues/en/ui.teamBridge.json"
    )
    assert "/x/d" not in [t.target.path for t in literal.targets], literal.targets


def test_untracked_file_edit_is_unchanged_low(home: Path, ui: Path) -> None:
    wt = ui / ".worktrees" / "bridge-single-recipient"
    (wt / "notes.json").write_text("{}\n")  # untracked
    v = _classify(home, f"cd {WT} && sed -i '' '/x/,/y/d' notes.json")
    assert v.risk is Risk.LOW
    assert [t.target.path for t in v.targets] == [os.path.realpath(wt / "notes.json")]


def test_tracked_edit_in_a_foreign_or_primary_checkout_is_low_too(home: Path, ui: Path) -> None:
    """The documented rule: an edit is not a deletion, wherever git can restore it."""
    command = f"cd {UI} && sed -i '' '/a/d' src/i18n/catalogues/en/ui.teamBridge.json"
    assert _classify(home, command, sid=S2).risk is Risk.LOW
    loop = (
        f"cd {UI} && for f in src/i18n/catalogues/*/ui.teamBridge.json; do sed -i '' /a/d $f; done"
    )
    assert _classify(home, loop, sid=S2).risk is Risk.LOW


@pytest.mark.parametrize(
    "command",
    [
        "sed -i '' '/x/d' ~/.local-operator/config.yml",
        "for f in ~/.local-operator/*.yml; do sed -i '' '/x/d' $f; done",
        f"for f in ~/.local-operator/sessions/{S2}/*.jsonl; do sed -i '' /x/d \"$f\"; done",
    ],
    ids=["config", "config-loop", "evidence-loop"],
)
def test_state_root_edits_stay_critical(home: Path, ui: Path, command: str) -> None:
    (home / ".local-operator" / "sessions" / S2 / "transcript.jsonl").write_text("{}\n")
    assert _classify(home, command).risk is Risk.CRITICAL


@pytest.mark.parametrize(
    "command",
    [
        f"cd {UI} && for d in .worktrees/*; do rm -rf $d; done",
        f"cd {WT} && for f in src/*; do g=$f; rm -rf x-$g; done",
        f"cd {WT} && for f in src/*; do cd $f && sed -i '' /x/d *.json; done",
    ],
    ids=["loop-delete", "derived-delete", "cd-into-loop-var"],
)
def test_glob_loop_variable_stays_opaque_outside_a_whole_word_write(
    home: Path, ui: Path, command: str
) -> None:
    """Pinning is for writes of exactly ``$f``: deletes and derived uses keep HIGH."""
    assert _classify(home, command).risk is Risk.HIGH


def test_preview_shows_the_field_command_settling(home: Path, ui: Path) -> None:
    ctx = CallContext(cwd=str(home), env=_env(home), scratchpad=None, session_id=S)
    report = preview_tool_call("bash", {"command": FIELD_COMMAND}, ctx, settings=GuardSettings())
    assert "LOW" in report.render() or "pass" in report.render().lower(), report.render()
    assert report.action != "review", report.render()


# -- 2. own clean worktree removal with gh unknown -----------------------------


@pytest.fixture
def gh_timeout(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    real_run = subprocess.run

    def run(argv: Any, *a: Any, **k: Any) -> Any:
        if isinstance(argv, list) and argv and argv[0] == "gh":
            raise subprocess.TimeoutExpired(argv, 8.0)
        return real_run(argv, *a, **k)

    monkeypatch.setattr(prov, "_gh_open_prs", _REAL_GH_LOOKUP)
    monkeypatch.setattr(prov.subprocess, "run", run)


@pytest.mark.asyncio
@pytest.mark.parametrize("force", ["--force ", ""], ids=["force", "plain"])
async def test_own_clean_worktree_with_gh_timing_out_is_removed_without_the_judge(
    home: Path, ui: Path, gh_timeout: None, force: str
) -> None:
    command = f"git worktree remove {force}{DONE}"
    v = _classify(home, command)
    assert v.risk is Risk.LOW, v.targets[0].reason
    assert "PR state unknown" in v.targets[0].reason  # the fact stays visible
    judge = _Judge(delay=1.0)
    outcome = await _check(home, command, judge)
    assert outcome is None or not outcome.blocked, outcome and outcome.text
    assert judge.prompts == []


@pytest.mark.asyncio
async def test_own_dirty_worktree_with_judge_timing_out_is_still_blocked(
    home: Path, ui: Path, gh_timeout: None
) -> None:
    wt = ui / ".worktrees" / "bridge-delivery-complete"
    (wt / "src" / "i18n" / "catalogues" / "en" / "ui.teamBridge.json").write_text("{}\n")
    command = f"git worktree remove --force {DONE}"
    judge = _Judge(answer="ALLOW operator", delay=1.0)
    outcome = await _check(home, command, judge)
    assert outcome is not None and outcome.blocked
    assert len(judge.prompts) == 1
    assert "judge unavailable" in outcome.text


@pytest.mark.parametrize("force", ["--force ", ""], ids=["force", "plain"])
def test_foreign_clean_worktree_with_gh_unknown_still_goes_to_the_judge(
    home: Path, ui: Path, gh_timeout: None, force: str
) -> None:
    v = _classify(home, f"git worktree remove {force}{DONE}", sid=S2)
    assert v.risk is Risk.HIGH, v.targets[0].reason
    assert "PR state unknown" in v.targets[0].reason


def test_own_clean_worktree_with_a_known_open_pr_stays_refused(
    home: Path, ui: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(prov, "_gh_open_prs", lambda repo, branch: [1858])
    v = _classify(home, f"git worktree remove --force {DONE}")
    assert v.risk is Risk.CRITICAL
    assert "open PR #1858" in v.targets[0].reason


def test_own_clean_worktree_with_a_running_other_owner_still_goes_to_the_judge(
    home: Path, ui: Path, gh_timeout: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(prov, "_running_owners", lambda real, sid: ("bbbbbbbbbbbb",))
    v = _classify(home, f"git worktree remove --force {DONE}")
    assert v.risk is Risk.HIGH
