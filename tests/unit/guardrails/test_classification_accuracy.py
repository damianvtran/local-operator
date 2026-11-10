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
    _ask(home, PARENT, "a-daca", question, ADACA_ANSWER)
    judge = _TimedOutJudge()
    outcome = await _check(home, "rm -f " + " ".join(str(f) for f in files), CHILD, [], judge)
    assert outcome is not None and not outcome.blocked, outcome
    assert judge.calls == 0


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
