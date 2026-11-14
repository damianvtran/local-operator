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
    # A carve-out the decline veto catches ("yes, only es") is no trusted yes,
    # so the ask refuses before any judge (PR #3966 R1-M1); the rest are read
    # by the judge. Neither settles without it.
    assert judge.calls == (0 if prov._answer_declines(answer, []) else 1)


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
    # A declining pick ("Abort", "Nah") is no trusted yes: the ask refuses
    # before any judge (PR #3966 R1-M1). A yes that qualifies itself in its
    # description, or a narrowing pick, is the judge's to read.
    labels = [lbl for lbl, _ in options]
    assert judge.calls == (0 if prov._answer_declines(picked, labels) else 1)


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
