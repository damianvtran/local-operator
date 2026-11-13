"""Classification accuracy of the dangerous-delete guard (incidents 2026-11-09).

Each test is a shape the guard got WRONG on lop 0.101.39, reproduced read-only
with ``preview_tool_call`` against the real state root before the fix:

1. ``for l in es fr …`` rewrote the letters ``es`` of ``sessions`` into
   ``sfrsions``/``shisions`` — invented targets for a call that never named
   ``$l`` there (locale audit, session c028fa237e58).
2. Read-only inputs ranked as dangerous targets: ``gh pr comment -F <file>``,
   ``shasum``/``identify`` of a receipt, ``cp -n src <new dst>``.
3. A task tree is one owner for reads: a parent publishing its child's
   remediation notes was blocked (PR #3702 / session 14728399e4e8).
4. Answered asks counted only through the judge: a judge timeout refused 55
   worktrees the operator approved by ask a-a382, and ask a-daca's answer
   "Go ahead and copy but … so they don't block this" was vetoed by its
   ``don't``, its ``{es,fr,…}`` path list named nothing.

The negative controls at the end keep real deletes and overwrites refused.
Everything is synthetic: an isolated ``HOME``/state root and made-up ids.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any

import pytest

from local_operator.guardrails import evidence as ev_mod
from local_operator.guardrails import provenance as prov
from local_operator.guardrails.check import CallContext, GuardSettings, check_tool_call
from local_operator.guardrails.classify import Risk, classify_bash
from local_operator.guardrails.resolve import resolve_command
from local_operator.harness.message_types import ASK_RESPONSE_MESSAGE_TYPE
from local_operator.harness.types import CustomMessage

PARENT = "aaaaaaaaaaaa"
CHILD = "bbbbbbbbbbbb"
STRANGER = "cccccccccccc"


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "home"
    state = home / ".local-operator"
    for sid in (PARENT, CHILD, STRANGER):
        (state / "sessions" / sid / "scratchpad").mkdir(parents=True)
    (state / "config.yml").write_text("version: 1\n")
    (home / "work").mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(state))
    for key in [k for k in os.environ if k.startswith(("CMUX_", "GIT_"))]:
        monkeypatch.delenv(key, raising=False)
    ev_mod._TREE_CACHE.clear()
    return home


def _pad(home: Path, sid: str) -> Path:
    return home / ".local-operator" / "sessions" / sid / "scratchpad"


def _verdict(home: Path, command: str, sid: str = PARENT):
    pad = _pad(home, sid)
    env = {"HOME": str(home), "LOCAL_OPERATOR_SCRATCHPAD": str(pad)}
    return classify_bash(
        command, cwd=str(home / "work"), env=env, scratchpad=str(pad), session_id=sid
    )


def _child_of(home: Path, child: str, parent: str) -> None:
    prov.record_lineage(
        child, origin="subagent", parent=parent, state_root=home / ".local-operator"
    )


# -- 1. the loop-variable parser ---------------------------------------------------------


def test_loop_variable_never_rewrites_text_that_does_not_reference_it(home: Path) -> None:
    """The c028fa237e58 shape: ``$D/ledger.patch`` never names ``$l``."""
    pad = _pad(home, CHILD)
    command = (
        "D=$LOCAL_OPERATOR_SCRATCHPAD/i18n-audit && for l in es fr hi ru ur vi zh-CN; do "
        "node x.mjs --locale $l; done; diff -u a b > $D/ledger.patch"
    )
    res = resolve_command(
        command,
        cwd=str(home / "work"),
        env={"HOME": str(home), "LOCAL_OPERATOR_SCRATCHPAD": str(pad)},
        home=str(home),
    )
    paths = [t.path for t in res.targets]
    assert paths == [f"{pad}/i18n-audit/ledger.patch"]
    assert not any("sfrsions" in str(p) or "shisions" in str(p) for p in paths)
    # Own scratch: an ordinary write, no judge.
    assert _verdict(home, command, sid=CHILD).risk is Risk.LOW


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        # Every value fans out, including one whose letters recur in the path
        # (the old value search skipped ``a`` because it occurs twice).
        ("for l in a b; do rm -rf /t/sessions/$l; done", ["/t/sessions/a", "/t/sessions/b"]),
        (
            'for l in es fr; do rm -rf "/t/sessions/${l}"; done',
            ["/t/sessions/es", "/t/sessions/fr"],
        ),
        # Through an assignment inside the body.
        ("for l in es fr; do d=/t/x-$l; rm -r $d; done", ["/t/x-es", "/t/x-fr"]),
        # Nested loops run every combination.
        (
            'for a in x y; do for b in 1 2; do rm "/t/$a-$b"; done; done',
            ["/t/x-1", "/t/x-2", "/t/y-1", "/t/y-2"],
        ),
        # A target with no ``$l`` is recorded once, as written.
        ("for l in es fr; do echo $l; done; rm /t/sessions/es", ["/t/sessions/es"]),
    ],
)
def test_loop_fan_out_rewrites_only_real_references(
    home: Path, command: str, expected: list[str]
) -> None:
    res = resolve_command(command, cwd="/w", env={"HOME": str(home)}, home=str(home))
    assert sorted(str(t.path) for t in res.targets) == sorted(expected)


# -- 2. read-only inputs are reads ------------------------------------------------------


def _stranger_file(home: Path, name: str = "notes.md") -> str:
    path = _pad(home, STRANGER) / "r2" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("synthetic\n")
    return str(path)


@pytest.mark.parametrize(
    "template",
    [
        "gh pr comment 1 -R o/r -F {f}",
        "gh pr comment 1 -R o/r --body-file {f}",
        "gh pr comment 1 -R o/r --body-file={f}",
        "gh issue comment 1 -F{f}",
        "gh release create v1 --notes-file {f}",
        "cat {f}",
        "shasum -a 256 {f}",
        "sha256sum {f}",
        "identify {f}",
        "magick identify -verbose {f}",
        "cmp {f} {f}",
        "cp -n {f} ~/work/new-copy.md",
    ],
)
def test_read_only_inputs_are_not_targets(home: Path, template: str) -> None:
    command = template.format(f=_stranger_file(home))
    assert _verdict(home, command).risk is Risk.LOW, command


def test_cp_no_clobber_to_a_new_path_is_a_creation(home: Path) -> None:
    src = _stranger_file(home)
    res = resolve_command(
        f"cp -n {src} {home}/work/new.md", cwd=str(home), env={"HOME": str(home)}, home=str(home)
    )
    assert [(t.path, t.op) for t in res.targets] == [(f"{home}/work/new.md", "append")]
    (home / "work" / "old.md").write_text("x")
    for flags in ("-n", "-nf", "-n -i", ""):
        res = resolve_command(
            f"cp {flags} {src} {home}/work/old.md",
            cwd=str(home),
            env={"HOME": str(home)},
            home=str(home),
        )
        assert [t.op for t in res.targets] == ["overwrite"], flags


@pytest.mark.parametrize(
    "template",
    [
        # gh's ``api -F`` is a field, not a body file: not exempted.
        "gh api repos/o/r -F body=@{f}",
        # ``msl:`` is a script coder that can write: not a plain identify.
        "magick identify msl:{f}",
        "identify -write /tmp/x.png {f}",
    ],
)
def test_read_exemptions_stay_narrow(home: Path, template: str) -> None:
    command = template.format(f=_stranger_file(home))
    assert _verdict(home, command).risk is Risk.HIGH, command


# -- 3. one task tree, one owner for reads ----------------------------------------------


def test_parent_and_child_read_and_copy_out_each_others_scratch(home: Path) -> None:
    _child_of(home, CHILD, PARENT)
    for reader, owner in ((PARENT, CHILD), (CHILD, PARENT)):
        f = _pad(home, owner) / "r2" / "agent-remediation-r1.md"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("synthetic\n")
        # The publication incident: verify, push, two ``gh pr comment -F``.
        command = (
            f"cat {f}; git -C ~/work status --short; "
            f"gh pr comment 3702 -R o/r -F {f}; gh pr comment 3702 -R o/r -F {f}; "
            f"shasum -a 256 {f}; cp -n {f} ~/work/copied-{reader}.md"
        )
        assert _verdict(home, command, sid=reader).risk is Risk.LOW, (reader, owner)


def test_tree_member_scratch_writes_keep_their_ranking(home: Path) -> None:
    _child_of(home, CHILD, PARENT)
    f = _pad(home, CHILD) / "notes.md"
    f.write_text("synthetic\n")
    for command in (f"rm -f {f}", f"mv {f} ~/work/x", f"cp ~/work/x {f}", f"echo x > {f}"):
        assert _verdict(home, command, sid=PARENT).risk is Risk.HIGH, command


# -- 4. answered asks count -------------------------------------------------------------


def _ask(
    home: Path,
    sid: str,
    ask_id: str,
    question: str,
    answer: str,
    options: tuple[tuple[str, str], ...] = (
        ("Yes, copy those files", "copies by literal path"),
        ("No, re-audit instead", ""),
    ),
    *,
    answered_ago_ms: int = 0,
) -> None:
    path = home / ".local-operator" / "sessions" / sid / "asks.jsonl"
    now = int(time.time() * 1000) - answered_ago_ms
    q = {
        "id": "q1",
        "question": question,
        "options": [{"label": lbl, "description": desc} for lbl, desc in options],
        "question_id": "",
    }
    rows = [
        {
            "v": 1,
            "kind": "queued",
            "ask_id": ask_id,
            "at": now - 10,
            "created_at": now - 10,
            "expires_at": now + 86_400_000,
            "timeout_s": 86400,
            "questions": [q],
        },
        {
            "v": 1,
            "kind": "answered",
            "ask_id": ask_id,
            "at": now,
            "by": {"surface": "desktop"},
            "answers": {"q1": [answer]},
        },
    ]
    with path.open("a") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


ADACA_ANSWER = (
    "Go ahead and copy but also send this to Aida to tune the guardrails so they "
    "don't block this in the future"
)


@pytest.mark.parametrize(
    ("answer", "consent"),
    [
        (ADACA_ANSWER, True),
        ("Yes, copy those files", True),
        ("No, re-audit instead", False),
        ("Yes, but keep the zh-cn one", False),
        ("Don't copy them", False),
        ("Go ahead, except the last one", False),
    ],
)
def test_free_text_answer_is_read_clause_wise(home: Path, answer: str, consent: bool) -> None:
    _ask(home, PARENT, "a-daca", "May I copy ~/work/a.json?", answer)
    assert (prov.verified_ask(PARENT, "a-daca") is not None) is consent


def test_brace_lists_yield_only_full_leaves(home: Path) -> None:
    """R1-M2: ``{a,b}/{c,d}`` names four leaves and never the parent ``a``;
    ``{a,b}{c,d}`` never names ``a``; ``{.,build}`` never names the dir."""
    h = str(home)
    text = "Delete ~/x/{a,b}/{c,d} and ~/y/{a,b}{c,d} and ~/r/{.,build} and ~/s/{..,t}?"
    for leaf in ("a/c", "a/d", "b/c", "b/d"):
        assert prov.names_path(text, f"{h}/x/{leaf}", home=h), leaf
    for leaf in ("ac", "ad", "bc", "bd"):
        assert prov.names_path(text, f"{h}/y/{leaf}", home=h), leaf
    for parent in ("x/a", "x/b", "x", "y/a", "y/b", "r", "r/build", "s", "", "x/{c,d}"):
        assert not prov.names_path(text, f"{h}/{parent}".rstrip("/"), home=h), parent


@pytest.mark.asyncio
async def test_brace_ask_never_approves_the_parent_directory(home: Path) -> None:
    """R1-M2 reviewer repro, end to end: ``rm -rf …/a`` after "Delete …/{a,b}/{c,d}"."""
    pad = _pad(home, STRANGER)
    for leaf in ("a/c", "a/d", "a/keep-me"):
        (pad / leaf).mkdir(parents=True)
    question = f"Delete {pad}/{{a,b}}/{{c,d}}?"
    _ask(home, PARENT, "a-beef", question, "Delete", (("Delete", ""), ("Keep", "")))
    judge = _TimedOutJudge()
    outcome = await _check(home, f"rm -rf {pad}/a", PARENT, [], judge)
    assert outcome is not None and outcome.blocked, outcome
    assert judge.calls == 1


def test_brace_list_in_an_ask_names_each_sibling(home: Path) -> None:
    h = str(home)
    text = (
        "copy ~/.local-operator/sessions/x/scratchpad/receipts-r2/"
        "m-{es,fr,zh-cn}-x.json and remove ~/repo/.worktrees/{a, b (QA scratch), c}."
    )
    base = f"{h}/.local-operator/sessions/x/scratchpad/receipts-r2/m-"
    for loc in ("es", "fr", "zh-cn"):
        assert prov.names_path(text, f"{base}{loc}-x.json", home=h)
    assert not prov.names_path(text, f"{base}de-x.json", home=h)
    for wt in ("a", "b", "c"):
        assert prov.names_path(text, f"{h}/repo/.worktrees/{wt}", home=h)
    assert not prov.names_path(text, f"{h}/repo/.worktrees/d", home=h)


class _TimedOutJudge:
    """The a-a382 judge: never answers in time."""

    def __init__(self) -> None:
        self.calls = 0

    async def __call__(self, system: str, prompt: str) -> str:
        self.calls += 1
        await asyncio.sleep(30)  # past ``judge_timeout_s``: wait_for cancels it
        return "ALLOW"


async def _check(home: Path, command: str, sid: str, messages: list[Any], judge: Any) -> Any:
    ctx = CallContext(
        cwd=str(home / "work"),
        env={"HOME": str(home), "LOCAL_OPERATOR_SCRATCHPAD": str(_pad(home, sid))},
        scratchpad=str(_pad(home, sid)),
        session_id=sid,
        operator=prov.operator_brief(messages, session_id=sid, cited=(command,)),
        messages=messages,
    )
    return await check_tool_call(
        "bash",
        {"command": command},
        ctx,
        judge=judge,
        settings=GuardSettings(judge_timeout_s=1.0),
        forked_judge=judge,
    )


def _stranger_receipts(home: Path) -> list[Path]:
    out = []
    for loc in ("es", "fr", "zh-cn"):
        f = _pad(home, STRANGER) / "receipts-r2" / f"m-{loc}-x.json"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("{}\n")
        out.append(f)
    return out


@pytest.mark.asyncio
async def test_answered_ask_settles_before_a_timed_out_judge(home: Path) -> None:
    files = _stranger_receipts(home)
    question = (
        "May I delete ~/.local-operator/sessions/"
        f"{STRANGER}/scratchpad/receipts-r2/m-{{es,fr,zh-cn}}-x.json?"
    )
    _ask(
        home, PARENT, "a-a382", question, "Remove all 3", (("Remove all 3", ""), ("Keep them", ""))
    )
    answer = CustomMessage(custom_type=ASK_RESPONSE_MESSAGE_TYPE, details={"ask_id": "a-a382"})
    judge = _TimedOutJudge()
    outcome = await _check(
        home, "rm -f " + " ".join(str(f) for f in files), PARENT, [answer], judge
    )
    assert outcome is not None and not outcome.blocked, outcome
    assert judge.calls == 0
    assert "named by ask a-a382" in outcome.warnings[0]


@pytest.mark.asyncio
async def test_ask_answered_in_the_parent_covers_the_child_without_a_pointer(home: Path) -> None:
    _child_of(home, CHILD, PARENT)
    files = _stranger_receipts(home)
    question = (
        f"May the coder remove ~/.local-operator/sessions/{STRANGER}/scratchpad/receipts-r2/"
        "m-{es,fr,zh-cn}-x.json, by literal path?"
    )
    _ask(home, PARENT, "a-0bad", question, "Go ahead and remove them")
    judge = _TimedOutJudge()
    outcome = await _check(home, "rm -f " + " ".join(str(f) for f in files), CHILD, [], judge)
    assert outcome is not None and not outcome.blocked, outcome
    assert judge.calls == 0


class _RecordingJudge:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    async def __call__(self, system: str, prompt: str) -> str:
        self.prompts.append(prompt)
        return "BLOCK"


@pytest.mark.asyncio
async def test_adaca_qualified_answer_reaches_the_judge_with_its_full_text(home: Path) -> None:
    """R1-B2: a-daca's answer has a second clause ("but … so they don't block
    this"), so it no longer settles deterministically; it reaches the judge,
    in the child's lineage, with the whole answer as consent evidence."""
    _child_of(home, CHILD, PARENT)
    files = _stranger_receipts(home)
    question = (
        f"May the coder remove ~/.local-operator/sessions/{STRANGER}/scratchpad/receipts-r2/"
        "m-{es,fr,zh-cn}-x.json, by literal path?"
    )
    _ask(home, PARENT, "a-daca", question, ADACA_ANSWER)
    judge = _RecordingJudge()
    await _check(home, "rm -f " + " ".join(str(f) for f in files), CHILD, [], judge)
    assert len(judge.prompts) == 1
    assert ADACA_ANSWER in judge.prompts[0]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "answer",
    [
        "Yes for es; no for fr and zh-cn",
        "yes but not fr",
        "ok, just es",
        "yes, minus fr",
        "yes, zh-cn stays",
        "yes, only es",
        "Yes. Also check the logs",
    ],
)
async def test_qualified_typed_answer_never_settles_without_the_judge(
    home: Path, answer: str
) -> None:
    """R1-B2 reviewer repro: a typed answer that narrows the approval goes to the judge."""
    _child_of(home, CHILD, PARENT)
    files = _stranger_receipts(home)
    question = (
        f"Delete ~/.local-operator/sessions/{STRANGER}/scratchpad/receipts-r2/m-es-x.json and "
        f"~/.local-operator/sessions/{STRANGER}/scratchpad/receipts-r2/m-fr-x.json and "
        f"~/.local-operator/sessions/{STRANGER}/scratchpad/receipts-r2/m-zh-cn-x.json?"
    )
    _ask(home, PARENT, "a-beef", question, answer)
    judge = _TimedOutJudge()
    outcome = await _check(home, "rm -f " + " ".join(str(f) for f in files), CHILD, [], judge)
    assert outcome is not None and outcome.blocked, (answer, outcome)
    assert judge.calls == 1


@pytest.mark.parametrize(
    "answer",
    [
        "yes",
        "Yes please",
        "ok",
        "go ahead",
        "Go ahead and copy",
        "approved",
        "do it",
        "Yes, delete them",
    ],
)
def test_bare_affirmative_typed_answer_approves_the_question(home: Path, answer: str) -> None:
    _ask(home, PARENT, "a-beef", "May I copy ~/work/a.json?", answer)
    words = prov.verified_ask(PARENT, "a-beef")
    assert words is not None and "~/work/a.json" in words.approved


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("options", "picked"),
    [
        ((("Abort", "Leave {f} in place"), ("Proceed", "")), "Abort"),
        ((("Nah", ""), ("Delete", "")), "Nah"),
        ((("Never mind", ""), ("Delete", "")), "Never mind"),
        ((("Proceed", "but keep {f}"), ("Abort", "")), "Proceed"),
        (
            (("Installer only", "deletes {f}"), ("All three", ""), ("Keep them", "")),
            "Installer only",
        ),
    ],
)
async def test_non_affirmative_pick_is_never_settled(
    home: Path, options: tuple[tuple[str, str], ...], picked: str
) -> None:
    """R1-M1 reviewer repro: picking "Abort" (description naming the file) settled the rm."""
    files = _stranger_receipts(home)
    opts = tuple((lbl, desc.format(f=files[0])) for lbl, desc in options)
    _ask(home, PARENT, "a-beef", f"What about {files[0]}?", picked, opts)
    judge = _TimedOutJudge()
    outcome = await _check(home, f"rm -f {files[0]}", PARENT, [], judge)
    assert outcome is not None and outcome.blocked, outcome
    assert judge.calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("question", "options", "picked"),
    [
        # Review R2-M1 repros: a narrowing pick never approves fr.
        ("Delete {es} and {fr}?", (("Delete es only", ""), ("Keep both", "")))
        + ("Delete es only",),
        (
            "Delete {es} and {fr}?",
            (("Delete the es receipt", ""), ("Keep both", "")),
            "Delete the es receipt",
        ),
        ("Delete {es}?", (("Delete", "removes es; {fr} remains"), ("Keep", ""))) + ("Delete",),
        (
            "What to do with {es}?",
            (("Delete es", "removes es; {fr} is excluded"), ("Keep", ""), ("Move", "")),
            "Delete es",
        ),
    ],
)
async def test_narrowing_pick_never_approves_an_excluded_path(
    home: Path, question: str, options: tuple[tuple[str, str], ...], picked: str
) -> None:
    files = _stranger_receipts(home)
    names = {"es": files[0], "fr": files[1]}
    opts = tuple((lbl, desc.format(**names)) for lbl, desc in options)
    _ask(home, PARENT, "a-beef", question.format(**names), picked, opts)
    judge = _TimedOutJudge()
    outcome = await _check(home, f"rm -f {files[1]}", PARENT, [], judge)
    assert outcome is not None and outcome.blocked, outcome
    assert judge.calls == 1


@pytest.mark.parametrize(
    "label", ["Yes", "Delete", "Remove all 57", "Go ahead", "Yes, delete them"]
)
def test_whole_set_pick_approves_the_question(label: str) -> None:
    q = {"question": "Delete /x/a and /x/b?"}
    opts = [{"label": label, "description": "d /x/c"}, {"label": "Keep them", "description": ""}]
    out = prov._approved_text(q, opts, [label, "Keep them"], [label])
    assert "Delete /x/a and /x/b?" in out and not any("/x/c" in o for o in out)


@pytest.mark.asyncio
async def test_ask_never_settles_a_repository_root(home: Path) -> None:
    """Cross-PR safety (#3700): a target holding ``.git`` keeps the judge."""
    repo = _pad(home, STRANGER) / "clone"
    (repo / ".git").mkdir(parents=True)
    _ask(home, PARENT, "a-beef", f"Delete {repo}?", "Delete", (("Delete", ""), ("Keep", "")))
    judge = _TimedOutJudge()
    outcome = await _check(home, f"rm -rf {repo}", PARENT, [], judge)
    assert outcome is not None and outcome.blocked, outcome
    assert judge.calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("inner", [".git", ".git/objects"])
async def test_ask_never_settles_a_git_directory(home: Path, inner: str) -> None:
    """``<repo>/.git`` holds no ``.git`` below it, yet deleting it (or part of
    it) destroys the root's history: the judge keeps it (rebase onto #3700)."""
    repo = _pad(home, STRANGER) / "clone"
    (repo / ".git" / "objects").mkdir(parents=True)
    target = repo / inner
    _ask(home, PARENT, "a-beef", f"Delete {target}?", "Delete", (("Delete", ""), ("Keep", "")))
    judge = _TimedOutJudge()
    outcome = await _check(home, f"rm -rf {target}", PARENT, [], judge)
    assert outcome is not None and outcome.blocked, outcome
    assert judge.calls == 1


class _AnswerJudge:
    """A judge that answers at once with a fixed first line."""

    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.calls = 0

    async def __call__(self, system: str, prompt: str) -> str:
        self.calls += 1
        return self.answer


@pytest.mark.asyncio
async def test_judge_allow_citing_an_ask_on_a_repository_root_runs(home: Path) -> None:
    """Once the judge has weighed a root's ``git:`` facts and ALLOWs citing a
    verified ask, the call runs. Regression: ``DeleteCase.verify`` returned a
    bare ``""`` for a root, which the caller's tuple unpack turned into a
    guard error ("the safety check failed") on every judged root delete."""
    repo = _pad(home, STRANGER) / "clone"
    (repo / ".git").mkdir(parents=True)
    _ask(home, PARENT, "a-beef", f"Delete {repo}?", "Delete", (("Delete", ""), ("Keep", "")))
    judge = _AnswerJudge("ALLOW ask=a-beef")
    outcome = await _check(home, f"rm -rf {repo}", PARENT, [], judge)
    assert judge.calls == 1
    assert outcome is not None and not outcome.blocked, outcome


@pytest.mark.asyncio
async def test_unsettled_asks_still_go_to_the_judge(home: Path) -> None:
    """Negative controls for the ask route: a stranger's ask, a pick that
    approves only part of a list, a declined answer and an old answer."""
    files = _stranger_receipts(home)
    rm = "rm -f " + " ".join(str(f) for f in files)
    question = (
        f"Delete ~/.local-operator/sessions/{STRANGER}/scratchpad/receipts-r2/"
        "m-{es,fr,zh-cn}-x.json?"
    )
    # Not in the lineage.
    _ask(home, STRANGER, "a-0001", question, "Remove all 3", (("Remove all 3", ""), ("Keep", "")))
    # In the lineage, but the pick names only one file.
    _ask(
        home,
        PARENT,
        "a-0002",
        question,
        "Only es",
        (("Only es", f"deletes {files[0]}"), ("All three", ""), ("Keep them", "")),
    )
    # Declined.
    _ask(home, PARENT, "a-0003", question, "Keep them", (("Delete", ""), ("Keep them", "")))
    # Approved, but two days ago.
    _ask(
        home,
        PARENT,
        "a-0004",
        question,
        "Delete",
        (("Delete", ""), ("Keep", "")),
        answered_ago_ms=2 * 86_400_000,
    )
    judge = _TimedOutJudge()
    outcome = await _check(home, rm, PARENT, [], judge)
    assert outcome is not None and outcome.blocked
    assert judge.calls == 1
    assert "judge unavailable" in outcome.text


# -- 5. negative controls: real deletes and overwrites ----------------------------------


@pytest.mark.parametrize(
    "template",
    [
        "rm -f {f}",
        "mv {f} ~/work/moved.md",
        "cp ~/work/src.md {f}",
        "cp -n ~/work/src.md {f}",  # exists: -n is no proof of creation here
        "cp -f ~/work/src.md {f}",
        "echo x > {f}",
        "for l in a b; do rm -rf {d}/$l; done",
    ],
)
def test_other_session_deletes_and_overwrites_stay_high(home: Path, template: str) -> None:
    f = _stranger_file(home)
    (home / "work" / "src.md").write_text("x")
    command = template.format(f=f, d=os.path.dirname(f))
    assert _verdict(home, command).risk is Risk.HIGH, command


@pytest.mark.parametrize(
    "template",
    [
        "cp -n ~/work/src.md {s}/transcript.jsonl",
        "cp -n ~/work/src.md {s}/asks.jsonl",
        "gh pr comment 1 -F ~/work/src.md > {s}/asks.jsonl",
        "for l in a b; do rm -f {s}/transcript.jsonl; done",
    ],
)
def test_evidence_files_stay_critical(home: Path, template: str) -> None:
    (home / "work" / "src.md").write_text("x")
    command = template.format(s=home / ".local-operator" / "sessions" / STRANGER)
    assert _verdict(home, command).risk is Risk.CRITICAL, command


# -- 6. review round 1: loop fan-out never truncated ------------------------------------


def test_loop_past_the_cap_is_opaque_not_truncated(home: Path) -> None:
    """R1-B1 reviewer repro: ``~`` at value 1101 was dropped and ranked LOW."""
    values = " ".join(f"/tmp/r3710/{i}" for i in range(1100)) + " ~"
    command = f"for d in {values}; do rm -rf $d; done"
    verdict = _verdict(home, command)
    assert verdict.risk in (Risk.HIGH, Risk.CRITICAL), verdict.risk


def test_nested_loops_past_the_cap_are_opaque(home: Path) -> None:
    a = " ".join(f"a{i}" for i in range(40))
    b = " ".join(f"b{i}" for i in range(40))
    command = f'for x in {a}; do for y in {b} ..; do rm -rf "/tmp/r3710/$x/$y"; done; done'
    assert _verdict(home, command).risk in (Risk.HIGH, Risk.CRITICAL)


def test_mismatched_loop_families_keep_the_fan_out(home: Path) -> None:
    """R1-M3 reviewer repro: base ranked HIGH with ``~/x``; head dropped it (LOW)."""
    command = 'for y in /x /y ""; do D=$y; done; for y in /tmp/z ~; do rm -rf "$y$D"; done'
    res = resolve_command(command, cwd=str(home / "work"), env={"HOME": str(home)}, home=str(home))
    paths = {t.path for t in res.targets}
    assert f"{home}/x" in paths and "/tmp/z/x" in paths, paths
    assert _verdict(home, command).risk in (Risk.HIGH, Risk.CRITICAL)


# -- 7. case 7: a tracked script reading a task-tree teammate's scratch file ------------


def _git_env() -> dict[str, str]:
    return {
        **{k: v for k, v in os.environ.items() if not k.startswith("GIT_")},
        "GIT_CONFIG_SYSTEM": "/dev/null",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_TERMINAL_PROMPT": "0",
    }


def _repo_with_script(home: Path, repo: Path | None = None) -> Path:
    import subprocess

    repo = repo or home / "work" / "ui"
    (repo / "scripts").mkdir(parents=True)
    (repo / "scripts" / "ledger-assemble.mjs").write_text("// reads receipts\n")
    (repo / "scripts" / "untracked.mjs").write_text("// not in git\n")
    env = {
        **{k: v for k, v in os.environ.items() if not k.startswith("GIT_")},
        "GIT_CONFIG_SYSTEM": "/dev/null",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_TERMINAL_PROMPT": "0",
    }
    for args in (
        ["init", "-q"],
        ["add", "scripts/ledger-assemble.mjs"],
        ["-c", "user.name=t", "-c", "user.email=t@example.invalid", "commit", "-qm", "s"],
    ):
        subprocess.run(["git", "-C", str(repo), *args], check=True, env=env)
    return repo


def _tree_verdict(home: Path, command: str, sid: str, cwd: Path):
    pad = _pad(home, sid)
    env = {"HOME": str(home), "LOCAL_OPERATOR_SCRATCHPAD": str(pad)}
    return classify_bash(command, cwd=str(cwd), env=env, scratchpad=str(pad), session_id=sid)


def _receipt(home: Path, sid: str) -> Path:
    f = _pad(home, sid) / "receipts" / "r-fr.json"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text("{}\n")
    return f


def test_tracked_script_reading_a_siblings_receipt_is_low(home: Path) -> None:
    """Case 7 (settings lane / auditor 1b1866b4f2ed, siblings under one manager)."""
    manager = "dddddddddddd"
    _child_of(home, CHILD, manager)
    _child_of(home, PARENT, manager)
    repo = _repo_with_script(home)
    receipt = _receipt(home, PARENT)
    command = f"node scripts/ledger-assemble.mjs --locale fr --receipt {receipt} --result clean"
    assert _tree_verdict(home, command, CHILD, repo).risk is Risk.LOW


@pytest.mark.parametrize(
    ("template", "why"),
    [
        ("node scripts/untracked.mjs --receipt {f}", "untracked script"),
        ("node scripts/ledger-assemble.mjs --out {f}", "named as output"),
        ("node scripts/ledger-assemble.mjs -o {f}", "named as output"),
        ("node scripts/ledger-assemble.mjs --write-receipt={f}", "named as output"),
        # Review R2-m2: in-place rewrites write the file they are handed.
        ("node scripts/ledger-assemble.mjs -i {f}", "in-place"),
        ("node scripts/ledger-assemble.mjs --in-place {f}", "in-place"),
        ("node scripts/ledger-assemble.mjs --inplace {f}", "in-place"),
        ("node scripts/ledger-assemble.mjs --in-place={f}", "in-place"),
        ("node -e 'x' {f}", "no script file"),
    ],
)
def test_task_tree_read_failing_a_condition_is_high_with_the_lineage_fact(
    home: Path, template: str, why: str
) -> None:
    manager = "dddddddddddd"
    _child_of(home, CHILD, manager)
    _child_of(home, PARENT, manager)
    repo = _repo_with_script(home)
    receipt = _receipt(home, PARENT)
    verdict = _tree_verdict(home, template.format(f=receipt), CHILD, repo)
    assert verdict.risk is Risk.HIGH, why
    fact = f"session {PARENT} is this session's sibling"
    assert any(fact in r for r in verdict.reasons), (why, verdict.reasons)


def test_tracked_script_reading_a_strangers_scratch_stays_high(home: Path) -> None:
    repo = _repo_with_script(home)
    receipt = _receipt(home, STRANGER)
    command = f"node scripts/ledger-assemble.mjs --receipt {receipt}"
    verdict = _tree_verdict(home, command, CHILD, repo)
    assert verdict.risk is Risk.HIGH
    assert not any("task tree" in r for r in verdict.reasons)


def test_symlink_in_a_siblings_pad_to_a_strangers_pad_stays_high(home: Path) -> None:
    """Review R2-M2: the owner is the session the RESOLVED path lands in."""
    manager = "dddddddddddd"
    _child_of(home, CHILD, manager)
    _child_of(home, PARENT, manager)
    repo = _repo_with_script(home)
    secret = _pad(home, STRANGER) / "secret.md"
    secret.parent.mkdir(parents=True, exist_ok=True)
    secret.write_text("synthetic\n")
    link = _pad(home, PARENT) / "lnk"
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(_pad(home, STRANGER))
    command = f"node scripts/ledger-assemble.mjs --receipt {link}/secret.md"
    verdict = _tree_verdict(home, command, CHILD, repo)
    assert verdict.risk is Risk.HIGH, verdict.reasons
    assert not any("task tree" in r for r in verdict.reasons)


# -- 8. one subagent lineage reads its own scratch (2026-11-12, UI #1888) ---------------
#
# Field reports under manager a7450d93932d, all reads refused as "another
# session's files":
# A. QA child fbd3094083f4 ran ``stat`` plus a ``node -e`` reading the
#    parent's staged ``manifest.json``. ``stat`` yields no target; the path
#    literal inside the inline script is an unresolvable read (op ``touch``).
# B. Reviewer b745482bd2a1 (the QA child's SIBLING) paged the QA child's
#    frozen files with ``sed -n 'N,Mp'``: sed was never trusted, and its write
#    check scanned the FILE operands, whose ``sessions`` holds an ``e``.
# C. The parent ran its descendants' supervisor/audit under ``env -u
#    NODE_OPTIONS … run_bounded.py -- node <child pad>/x.mjs``; ``-u``'s value
#    was read as the program ("handed to NODE_OPTIONS").

SIBLING = "eeeeeeeeeeee"
GRANDCHILD = "ffffffffffff"


def _staged_manifest(home: Path, sid: str) -> Path:
    stage = _pad(home, sid) / "ui1888-native-stage-ehzeemcn"
    (stage / "bundle").mkdir(parents=True, exist_ok=True)
    (stage / "stage-receipt.json").write_text("{}\n")
    manifest = stage / "bundle" / "manifest.json"
    manifest.write_text('{"files": [1, 2]}\n')
    return manifest


def _field_command(manifest: Path) -> str:
    """The refused call, byte-for-byte in shape (only the root differs)."""
    stage = manifest.parent.parent
    return (
        f"stat -f 'stage-receipt.json bytes=%z' {stage}/stage-receipt.json; "
        'node -e \'const fs=require("fs");const m=JSON.parse(fs.readFileSync('
        f'"{manifest}"));console.log("Manifest-listed executable/package files="'
        "+m.files.length)'"
    )


def _tree(home: Path) -> None:
    """PARENT launched CHILD and SIBLING; CHILD launched GRANDCHILD."""
    for sid in (SIBLING, GRANDCHILD):
        _pad(home, sid).mkdir(parents=True, exist_ok=True)
    _child_of(home, CHILD, PARENT)
    _child_of(home, SIBLING, PARENT)
    _child_of(home, GRANDCHILD, CHILD)


@pytest.mark.parametrize(
    ("reader", "owner"),
    [
        (CHILD, PARENT),  # case A as filed
        (GRANDCHILD, PARENT),  # full depth upward
        (PARENT, CHILD),  # descendant
        (PARENT, GRANDCHILD),  # full depth downward
    ],
)
def test_inline_read_within_one_lineage_is_low(home: Path, reader: str, owner: str) -> None:
    _tree(home)
    command = _field_command(_staged_manifest(home, owner))
    verdict = _tree_verdict(home, command, reader, home / "work")
    assert verdict.risk is Risk.LOW, verdict.reasons


def test_field_read_from_outside_the_lineage_stays_high(home: Path) -> None:
    """A stranger, a sibling and a cousin keep the HIGH (siblings are not lineage)."""
    _tree(home)
    parent_file = _staged_manifest(home, PARENT)
    child_file = _staged_manifest(home, CHILD)
    grandchild_file = _staged_manifest(home, GRANDCHILD)
    cases = [
        (STRANGER, parent_file, ""),
        (SIBLING, child_file, f"session {CHILD} is this session's sibling"),
        # A cousin-shaped relation: SIBLING's nephew is not in its lineage.
        (SIBLING, grandchild_file, f"session {GRANDCHILD} is this session's sibling"),
    ]
    for reader, target, fact in cases:
        verdict = _tree_verdict(home, _field_command(target), reader, home / "work")
        assert verdict.risk is Risk.HIGH, (reader, verdict.reasons)
        if fact:
            assert any(fact in r for r in verdict.reasons), (reader, verdict.reasons)
        else:
            assert not any("task tree" in r for r in verdict.reasons), verdict.reasons


def test_opener_link_is_not_a_launcher(home: Path) -> None:
    """An agent-opened workstream is a separate task: no inherited scratch."""
    prov.record_lineage(
        CHILD, origin="workstream", opener=PARENT, state_root=home / ".local-operator"
    )
    command = _field_command(_staged_manifest(home, PARENT))
    assert _tree_verdict(home, command, CHILD, home / "work").risk is Risk.HIGH


def test_child_writes_and_deletes_in_a_launchers_scratch_stay_high(home: Path) -> None:
    """Inheritance covers reads only: the launcher's pad is a live session's state."""
    _tree(home)
    manifest = _staged_manifest(home, PARENT)
    pad = _pad(home, PARENT)
    for command in (
        f"rm -rf {pad}",
        f"rm -rf {manifest.parent.parent}",
        f"rm -f {manifest}",
        f"echo x > {manifest}",
        f"mv {manifest} ~/work/x",
        f"node tool.mjs -o {manifest}",
    ):
        verdict = _tree_verdict(home, command, CHILD, home / "work")
        assert verdict.risk is Risk.HIGH, (command, verdict.reasons)


def test_child_read_of_launchers_evidence_or_state_root_is_unchanged(home: Path) -> None:
    _tree(home)
    state = home / ".local-operator"
    transcript = state / "sessions" / PARENT / "transcript.jsonl"
    transcript.write_text("{}\n")
    high = _tree_verdict(home, f"node -e 'x' {transcript}", CHILD, home / "work")
    assert high.risk is Risk.HIGH, high.reasons
    glob = _tree_verdict(home, f"rm -rf {home}/.local-op*", CHILD, home / "work")
    assert glob.risk is Risk.CRITICAL, glob.reasons
    read_glob = _tree_verdict(home, f"node -e 'x' {home}/.local-operator*", CHILD, home / "work")
    assert read_glob.risk is Risk.HIGH, read_glob.reasons


def test_link_in_a_launchers_pad_to_a_strangers_pad_stays_high(home: Path) -> None:
    """Every spelling, the realpath included, must land in a launcher's pad."""
    _tree(home)
    secret = _pad(home, STRANGER) / "secret.md"
    secret.write_text("synthetic\n")
    link = _pad(home, PARENT) / "lnk"
    link.symlink_to(_pad(home, STRANGER))
    verdict = _tree_verdict(home, f"node -e 'x' {link}/secret.md", CHILD, home / "work")
    assert verdict.risk is Risk.HIGH, verdict.reasons


def test_case_b_sibling_pages_frozen_files_with_sed_but_cannot_run_them(home: Path) -> None:
    """Case B: ``sed -n 'N,Mp'`` is a pure read; running a sibling's code is not."""
    _tree(home)
    pad = _pad(home, CHILD)
    for name in ("consent-native-supervisor.mjs", "consent-guards.cjs", "consent-static.test.cjs"):
        (pad / name).write_text("// synthetic\n")
    paged = (
        f"shasum -a 256 {pad}/consent-native-supervisor.mjs; "
        f"sed -n '1,135p' {pad}/consent-native-supervisor.mjs; "
        f"sed -n '7,23p' {pad}/consent-guards.cjs; tail -9 {pad}/consent-guards.cjs"
    )
    assert _tree_verdict(home, paged, SIBLING, home / "work").risk is Risk.LOW
    run = f"node --test --test-concurrency=1 {pad}/consent-static.test.cjs"
    assert _tree_verdict(home, run, SIBLING, home / "work").risk is Risk.HIGH


def _case_c(home: Path, scratch_value: str | None) -> str:
    child, grandchild = _pad(home, CHILD), _pad(home, GRANDCHILD)
    (child / "consent-native-supervisor.mjs").write_text("// synthetic\n")
    (grandchild / "audit.json").write_text("{}\n")
    stage = _staged_manifest(home, PARENT).parent.parent
    assign = f"LOCAL_OPERATOR_SCRATCHPAD={scratch_value} " if scratch_value else ""
    return (
        f"set +e\nenv -u NODE_OPTIONS {assign}PYTHONPATH={home}/work "
        f"python3 {home}/work/run_bounded.py --contract=command-status --timeout 170 -- "
        f"node {child}/consent-native-supervisor.mjs --manifest {stage}/bundle/manifest.json "
        f"--binding-audit {grandchild}/audit.json 2>&1 | tee {stage}/native-run.log\n"
        'rc=${PIPESTATUS[0]}\nexit "$rc"'
    )


def test_case_c_parent_runs_descendants_files_under_env_unset(home: Path) -> None:
    """Case C's files are LOW; the child's pad ROOT in an assignment is HIGH.

    As filed (transcript a7450d93932d), the launcher also set
    ``LOCAL_OPERATOR_SCRATCHPAD=<child pad ROOT>``. A pad root handed to opaque
    code is the QA-M1 wipe shape, so that one mention stays HIGH by design
    (PR #3906 round 1; no carve-out keyed on a variable name) and the judge
    gets its provenance (next test). The files it runs and reads stay LOW.
    """
    _tree(home)
    child = _pad(home, CHILD)
    as_filed = _tree_verdict(home, _case_c(home, str(child)), PARENT, home / "work")
    assert as_filed.risk is Risk.HIGH, as_filed.reasons
    assert [t.target.path for t in as_filed.flagged_targets] == [str(child)]
    assert not any("NODE_OPTIONS" in r for r in as_filed.reasons)
    for value in (None, str(child / "consent-native-supervisor.mjs")):
        verdict = _tree_verdict(home, _case_c(home, value), PARENT, home / "work")
        assert verdict.risk is Risk.LOW, (value, verdict.reasons)
    assert _tree_verdict(home, _case_c(home, None), STRANGER, home / "work").risk is Risk.HIGH


@pytest.mark.asyncio
async def test_case_c_judge_is_told_the_root_assignment_provenance(home: Path) -> None:
    """The judge receives proved facts: whose root, how it arrived, no delete op."""
    _tree(home)
    judge = _RecordingJudge()
    command = _case_c(home, str(_pad(home, CHILD)))
    outcome = await _check(home, command, PARENT, [], judge)
    assert outcome.blocked and len(judge.prompts) == 1
    prompt = judge.prompts[0]
    for fact in (
        f"session {CHILD} is this session's descendant in one task tree",
        "the path is that session's scratchpad root",
        "it reaches this call only as the inline env assignment LOCAL_OPERATOR_SCRATCHPAD=…",
        "no recognised delete or write op names it",
    ):
        assert fact in prompt, (fact, prompt)


@pytest.mark.parametrize(
    ("argv", "read_only"),
    [
        (["-n", "1,135p", "/s/sessions/x/scratchpad/a.mjs"], True),
        (["-ne", "1p", "f"], True),
        (["-n", "-e", "7,55p", "f"], True),
        (["--expression=1,3p", "f"], True),
        (["-i", "", "1p", "f"], False),
        (["-ni", "1p", "f"], False),
        (["--in-place", "p", "f"], False),
        (["-n", "1w /x", "f"], False),
        (["-n", "s/a/b/e", "f"], False),
        (["-f", "script.sed", "f"], False),
        (["--qui", "1p", "f"], False),  # abbreviated long option: unknown
    ],
)
def test_sed_is_read_only_only_for_a_printing_script(argv: list[str], read_only: bool) -> None:
    from local_operator.guardrails.resolve import read_only_argv

    assert read_only_argv("sed", argv) is read_only


def test_env_unset_value_is_not_the_program() -> None:
    from local_operator.tools.search_guard import (
        _WRAPPER_LONG_VALUE_FLAGS,
        _WRAPPER_VALUE_FLAGS,
    )

    assert "u" in _WRAPPER_VALUE_FLAGS["env"]
    # Env-only (review F2 on PR #3906): never a value flag of sudo/nice/timeout.
    assert "--unset" not in _WRAPPER_LONG_VALUE_FLAGS
    resolution = resolve_command(
        "env -u NODE_OPTIONS node /x/.local-operator/sessions/s/scratchpad/a.mjs",
        cwd="/tmp",
        env={"HOME": "/h"},
        home="/h",
    )
    assert {t.program for t in resolution.touched} <= {"node"}, resolution.touched


def test_subagent_ancestors_follows_parent_links_only(home: Path) -> None:
    _tree(home)
    state = home / ".local-operator"
    assert prov.subagent_ancestors(GRANDCHILD, state_root=state) == [CHILD, PARENT]
    assert prov.subagent_ancestors(SIBLING, state_root=state) == [PARENT]
    assert prov.subagent_ancestors(PARENT, state_root=state) == []
    assert prov.task_tree_root(GRANDCHILD, state_root=state) == PARENT
    assert prov.task_tree_root(PARENT, state_root=state) == PARENT


# -- 8b. PR #3906 round 1: the lineage credit is a FILE read credit ----------------------

_OPAQUE_RMSYNC = (
    'node -e \'const f=require("fs");f["rm"+"Sync"](process.argv[1],{recursive:true})\''
)
_OPAQUE_RMTREE = 'python3 -c \'import shutil as s,sys;getattr(s,"rm"+"tree")(sys.argv[1])\''


@pytest.mark.parametrize(
    ("reader", "owner", "program", "sub"),
    [
        (CHILD, PARENT, _OPAQUE_RMSYNC, ""),  # QA M1: the parent's pad ROOT
        (CHILD, PARENT, _OPAQUE_RMSYNC, "ui1888-native-stage-ehzeemcn"),  # M2: a dir
        (PARENT, CHILD, _OPAQUE_RMTREE, ""),  # M6: a child's pad root
        (PARENT, CHILD, "node ~/x", ""),  # D7: opaque script file, pad root
        (GRANDCHILD, PARENT, "node ~/x", "ui1888-native-stage-ehzeemcn/bundle"),
    ],
)
def test_opaque_program_on_a_lineage_directory_stays_high(
    home: Path, reader: str, owner: str, program: str, sub: str
) -> None:
    """QA Q1: opaque code handed a lineage pad ROOT or DIRECTORY keeps the HIGH."""
    _tree(home)
    _staged_manifest(home, owner)
    target = _pad(home, owner) / sub if sub else _pad(home, owner)
    verdict = _tree_verdict(home, f"{program} {target}", reader, home / "work")
    assert verdict.risk is Risk.HIGH, verdict.reasons


def test_opaque_program_on_a_lineage_regular_file_is_low(home: Path) -> None:
    """The three field shapes pass regular files only, so they keep the credit."""
    _tree(home)
    manifest = _staged_manifest(home, PARENT)
    for program in (_OPAQUE_RMSYNC, "node ~/x"):
        verdict = _tree_verdict(home, f"{program} {manifest}", CHILD, home / "work")
        assert verdict.risk is Risk.LOW, (program, verdict.reasons)


def test_lineage_symlink_or_missing_path_gets_no_credit(home: Path) -> None:
    """A link (even to a lineage file) or a missing path is not a pinned file."""
    _tree(home)
    manifest = _staged_manifest(home, PARENT)
    link = _pad(home, PARENT) / "m-link.json"
    link.symlink_to(manifest)
    for target in (link, _pad(home, PARENT) / "missing.json"):
        verdict = _tree_verdict(home, f"node -e 'x' {target}", CHILD, home / "work")
        assert verdict.risk is Risk.HIGH, (target, verdict.reasons)


@pytest.mark.parametrize(
    "script",
    [
        "sed -n 's/x/rm -rf ~/e' {f}",  # QA B8: the ``s`` flag ``e``
        "sed '1e rm -rf ~/work' {f}",  # the ``e`` command
        "sed -f ~/x {f}",  # QA B9: script in a file
        "sed --file=$HOME/x {f}",
        "sed -nf ~/x {f}",
    ],
)
def test_executing_sed_on_a_lineage_file_gets_no_read_credit(home: Path, script: str) -> None:
    """QA Q2: a sed that runs code is an executor, not a reader."""
    _tree(home)
    manifest = _staged_manifest(home, CHILD)
    verdict = _tree_verdict(home, script.format(f=manifest), PARENT, home / "work")
    assert verdict.risk is Risk.HIGH, verdict.reasons


def test_bsd_sed_capital_i_is_an_in_place_write(home: Path) -> None:
    """Review F1: BSD ``sed -I`` edits in place, suffix or not, like ``-i``."""
    _tree(home)
    manifest = _staged_manifest(home, PARENT)
    for flags in ("-I ''", "-I.bak", "-nI ''"):
        command = f"sed {flags} 's/a/b/' {manifest}"
        verdict = _tree_verdict(home, command, CHILD, home / "work")
        assert verdict.risk is Risk.HIGH, (flags, verdict.reasons)
        assert any(
            t.op == "overwrite"
            for t in resolve_command(
                command, cwd=str(home / "work"), env={"HOME": str(home)}, home=str(home)
            ).targets
        ), flags


def test_unset_is_an_env_only_long_value_flag(home: Path) -> None:
    """Review F2: ``sudo --unset rm -rf X`` still ranks the ``rm``."""
    (home / "work" / "keep.txt").write_text("x\n")
    plain = _verdict(home, f"sudo rm -rf {home}/work").risk
    assert plain is not Risk.LOW
    assert _verdict(home, f"sudo --unset rm -rf {home}/work").risk is plain
    assert _verdict(home, f"env --unset FOO rm -rf {home}/work").risk is plain


@pytest.mark.parametrize(
    "command",
    [
        "env -u FOO -S 'rm -rf ~/.local-operator'",
        "env --unset FOO -S 'rm -rf ~/.local-operator'",
        "env -i -u FOO --split-string='rm -rf ~/.local-operator'",
        "env -iS 'rm -rf ~/.local-operator'",
    ],
)
def test_env_split_string_after_other_options_is_not_dropped(home: Path, command: str) -> None:
    """``-S`` anywhere in env's option prefix runs a command line: never LOW."""
    assert _verdict(home, command).risk is not Risk.LOW


def test_tracked_script_with_uncommitted_edits_is_high(home: Path) -> None:
    """Review R2-m1: an edited tracked script is the agent's code, not the repo's."""
    manager = "dddddddddddd"
    _child_of(home, CHILD, manager)
    _child_of(home, PARENT, manager)
    repo = _repo_with_script(home)
    (repo / "scripts" / "ledger-assemble.mjs").write_text("writeFileSync(argv[2], '')\n")
    receipt = _receipt(home, PARENT)
    command = f"node scripts/ledger-assemble.mjs --receipt {receipt}"
    assert _tree_verdict(home, command, CHILD, repo).risk is Risk.HIGH


def test_committed_script_in_a_scratchpad_repo_is_high(home: Path) -> None:
    """Review R2-m1: a repo the agent ``git init``-ed in its own pad is not reviewed code."""
    manager = "dddddddddddd"
    _child_of(home, CHILD, manager)
    _child_of(home, PARENT, manager)
    repo = _repo_with_script(home, _pad(home, CHILD) / "rig")
    receipt = _receipt(home, PARENT)
    command = f"node scripts/ledger-assemble.mjs --receipt {receipt}"
    assert _tree_verdict(home, command, CHILD, repo).risk is Risk.HIGH


def test_tracked_script_handed_a_siblings_evidence_file_stays_critical(home: Path) -> None:
    manager = "dddddddddddd"
    _child_of(home, CHILD, manager)
    _child_of(home, PARENT, manager)
    repo = _repo_with_script(home)
    transcript = home / ".local-operator" / "sessions" / PARENT / "transcript.jsonl"
    transcript.write_text("")
    command = f"node scripts/ledger-assemble.mjs --receipt {transcript} > {transcript}"
    assert _tree_verdict(home, command, CHILD, repo).risk is Risk.CRITICAL


# -- 8. issue #3613: no verdict is not a policy decision --------------------------------


@pytest.mark.asyncio
async def test_judge_without_a_verdict_refuses_with_no_verdict_wording(home: Path) -> None:
    files = _stranger_receipts(home)
    judge = _TimedOutJudge()
    outcome = await _check(home, f"rm -f {files[0]}", PARENT, [], judge)
    assert outcome is not None and outcome.blocked
    text = outcome.text
    assert "could not reach a verdict" in text
    assert "judge unavailable: timeout" in text
    assert "not a policy decision" in text
    assert "Retry later" in text and "store-verified `ask`" in text
    assert "Ask the user, or target a narrower path" not in text


def test_policy_block_keeps_the_policy_wording() -> None:
    from local_operator.guardrails.check import review_text

    text = review_text("some-rule", "the safety check refused this call", "x")
    assert "could not reach a verdict" not in text and "(review): the safety check refused" in text


@pytest.mark.asyncio
async def test_a_block_mentioning_judge_unavailable_is_still_a_policy_block() -> None:
    """Review R2-n2: wording follows ``JudgeResult.no_verdict``, not the reason's prefix."""
    from local_operator.guardrails.judge import JudgeBrief, run_judge

    async def judge(system: str, prompt: str) -> str:
        return "BLOCK judge unavailable: pretend"

    async def off(system: str, prompt: str) -> str:
        return "maybe"

    brief = JudgeBrief(tool="bash", call_text="x", flagged="x")
    blocked = await run_judge(brief, judge, timeout=5)
    assert not blocked.allowed and not blocked.no_verdict
    assert (await run_judge(brief, off, timeout=5)).no_verdict
    assert (await run_judge(brief, None, timeout=5)).no_verdict


def test_user_rule_no_verdict_names_the_call_not_paths() -> None:
    """Review R2-n1: a user rule's targets are the call, not paths."""
    from local_operator.guardrails.check import review_text

    text = review_text(
        "my-rule",
        "judge unavailable: timeout after 1s",
        "this bash call",
        no_verdict=True,
        paths=False,
    )
    assert "naming this call" in text and "these paths" not in text
