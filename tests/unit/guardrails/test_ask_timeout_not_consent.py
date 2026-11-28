"""A timed-out ask is never consent; ``chmod -R u+w`` is a forced removal (2026-11-14).

Incident (session 1faad822298d, relayed by Aida from the operator): the cleanup
session asked (ask ``a-20fc``) which rig, build and OSWorld directories to
delete. Nobody answered; after an hour the harness delivered

    [Ask timed out] No reply to ask a-20fc arrived within 1h ...
    Proceed without it: use your recommended option(s) or best judgment and
    state the assumption in your report.

and the session ran ``rm -rf`` on the recommended buckets (commands annotated
``# ask=a-20fc (recommended options 0, 1, 3 after 1h timeout)``). A deep
benchmark tree refused with ``Permission denied`` on its write-protected raw
results, so the next call ran ``chmod -R u+w <tree> ... || true`` and
``rm -rf`` again. Every call that ran ranked LOW (deep paths; ``chmod`` was
modelled as metadata only), so no judge ever saw them; the two HIGH calls were
refused (judge timeout; ``~/node_modules`` unnamed).

Fixed at the source, pinned here in a synthetic HOME and ask store:
1. the timeout notice / system prompt / ``ask`` docs say a timeout is not
   consent for anything irreversible (``tests/unit/asks/test_render.py`` holds
   the notice rows; the prompt and tool copy are pinned below);
2. a timed-out (or withdrawn, dismissed, open) ask naming a HIGH path is
   refused deterministically before any judge runs, and every such ask is
   named "NOT consent" in the judge's facts;
3. a recursive write grant / immutable-flag clear on a path the session did
   not create ranks HIGH and reaches the judge as a forced removal;
4. a DEEP path (LOW on its own) that an unanswered ask in the session's family
   names — equal, under or above, globs included — ranks HIGH and is refused
   the same way (incident rows 302, 320, 365), unless the ledger records it.
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from local_operator.guardrails import provenance as prov
from local_operator.guardrails.check import CallContext, GuardSettings, check_tool_call
from local_operator.guardrails.classify import (
    FORCED_REMOVAL_REASON,
    Risk,
    classify_bash,
)
from local_operator.guardrails.judge import JUDGE_SYSTEM_PROMPT
from local_operator.guardrails.preview import preview_tool_call
from local_operator.guardrails.resolve import _chmod_grants_write

S = "1faad822298d"  # synthetic id shaped like the incident session's
ASK = "a-20fc"


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    home = tmp_path / "home"
    state = home / ".local-operator"
    (state / "sessions" / S / "scratchpad").mkdir(parents=True)
    (state / "config.yml").write_text("version: 1\n")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(state))
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", "/dev/null")
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", "/dev/null")
    monkeypatch.setenv("GIT_TERMINAL_PROMPT", "0")
    # No forge or process lookups for synthetic trees.
    monkeypatch.setattr(prov, "_live_cwd_inside", lambda _, **_kw: False)
    monkeypatch.setattr(prov, "_github_repo", lambda _: "")
    monkeypatch.setattr(prov, "_running_owners", lambda *a, **k: ())
    # The incident's targets, synthetic: two top-level home dirs (HIGH) and a
    # deep benchmark tree with a write-protected raw-results file.
    (home / "qa-3506").mkdir(parents=True)
    (home / "qa-bridge-home").mkdir()
    coding = home / "workspace" / "harness-cost-bench" / "coding" / "generated"
    coding.mkdir(parents=True)
    raw = coding / "summary_tables.json"
    raw.write_text("{}")
    raw.chmod(0o444)
    coding.chmod(0o555)
    yield home
    coding.chmod(0o755)  # let tmp_path cleanup remove it


def _env(home: Path) -> dict[str, str]:
    return {"HOME": str(home), "LOCAL_OPERATOR_CONFIG_DIR": str(home / ".local-operator")}


def _pad(home: Path) -> str:
    return str(home / ".local-operator" / "sessions" / S / "scratchpad")


#: The incident ask's shape: the paths in each option LABEL as a brace list.
QUESTION = (
    "Which of these idle scratch rigs, build output directories, and terminated-session "
    "scratchpads should I delete to reach ~80+ GB reclaimed?"
)
OPTIONS = [
    {"label": "Delete ~/{qa-3506,qa-bridge-home}", "description": "old QA homes"},
    {"label": "Keep everything else as-is", "description": "Stop here."},
]


def _ask_log(home: Path, *, answer: str | None, expired: bool) -> None:
    """``a-20fc`` in the session's store: timed out unanswered, or answered on desktop."""
    now = int(time.time() * 1000)
    expires = now - 60_000 if expired else now + 3_600_000
    rows: list[dict[str, Any]] = [
        {
            "v": 1,
            "kind": "queued",
            "ask_id": ASK,
            "at": expires - 3_600_000,
            "created_at": expires - 3_600_000,
            "expires_at": expires,
            "timeout_s": 3600,
            "questions": [
                {
                    "id": "cleanup_buckets",
                    "question": QUESTION,
                    "options": OPTIONS,
                    "multi": True,
                    "question_id": "",
                    "recommended": 0,
                }
            ],
        }
    ]
    if answer is not None:
        rows.append(
            {
                "v": 1,
                "kind": "answered",
                "ask_id": ASK,
                "at": now - 1000,
                "by": {"surface": "desktop"},
                "answers": {"cleanup_buckets": [answer]},
            }
        )
    path = home / ".local-operator" / "sessions" / S / "asks.jsonl"
    with path.open("a") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


class _Judge:
    """Counts calls and records every brief it is shown."""

    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.prompts: list[str] = []
        self.systems: list[str] = []

    async def __call__(self, system: str, prompt: str) -> str:
        self.systems.append(system)
        self.prompts.append(prompt)
        return self.answer


def _ctx(home: Path, command: str, intent: str = "") -> CallContext:
    return CallContext(
        cwd=str(home),
        env=_env(home),
        scratchpad=_pad(home),
        session_id=S,
        # The agent's own prose about the timeout, exactly as in the incident.
        intent=intent or "Removing rigs (ask a-20fc, recommended options after 1h timeout)",
        operator=prov.operator_brief([], session_id=S, cited=(command,)),
        messages=[],
    )


async def _check(home: Path, command: str, judge: _Judge) -> Any:
    return await check_tool_call(
        "bash",
        {"command": command},
        _ctx(home, command),
        judge=judge,
        settings=GuardSettings(),
        forked_judge=judge,
    )


INCIDENT_RM = (
    "# ask=a-20fc (recommended options 0, 1, 3 after 1h timeout)\nrm -rf ~/qa-3506 ~/qa-bridge-home"
)


# -- 2. a timed-out ask is never consent ------------------------------------------


@pytest.mark.asyncio
async def test_timed_out_ask_naming_the_paths_is_refused_before_any_judge(home: Path) -> None:
    _ask_log(home, answer=None, expired=True)
    assert prov.verified_ask(S, ASK) is None  # the store route already refuses it
    judge = _Judge("ALLOW ask=a-20fc")  # a judge that WOULD be fooled
    outcome = await _check(home, INCIDENT_RM, judge)
    assert outcome is not None and outcome.blocked
    assert judge.prompts == []  # deterministic: no model was asked
    assert "a-20fc" in outcome.text and "TIMED OUT" in outcome.text
    assert "NOT consent" in outcome.text
    assert (home / "qa-3506").is_dir() and (home / "qa-bridge-home").is_dir()


@pytest.mark.asyncio
async def test_typed_operator_turn_after_a_timeout_still_reaches_the_judge(home: Path) -> None:
    """PR #4493 kept guide://guardrails exception 2: a turn the operator typed
    that names the action reaches the judge, which reads it. Only an ASK
    answer is ordered against the refusing ask (it has a timestamp); a typed
    turn here has none, so it keeps the pre-#4493 judge route."""
    _ask_log(home, answer=None, expired=True)
    command = "rm -rf ~/qa-3506 ~/qa-bridge-home"
    ctx = _ctx(home, command)
    ctx.operator.words.insert(
        0, prov.OperatorWords("operator, direct", "Yes, rm -rf ~/qa-3506 ~/qa-bridge-home now")
    )
    judge = _Judge("BLOCK: control")
    outcome = await check_tool_call(
        "bash", {"command": command}, ctx, judge=judge, settings=GuardSettings(), forked_judge=judge
    )
    assert outcome is not None and outcome.blocked
    assert len(judge.prompts) == 1  # the judge ran: the refusal was not deterministic


def test_preview_names_the_timed_out_ask_as_not_consent(home: Path) -> None:
    _ask_log(home, answer=None, expired=True)
    report = preview_tool_call(
        "bash",
        {"command": INCIDENT_RM},
        _ctx(home, INCIDENT_RM),
        settings=GuardSettings(),
    )
    facts = "\n".join(report.builtin_facts)
    assert report.builtin_risk.upper() == "HIGH", report.render()
    assert "ask a-20fc names it but was TIMED OUT with no answer: NOT consent" in facts


@pytest.mark.asyncio
async def test_operator_answered_ask_still_allows_the_same_paths(home: Path) -> None:
    """Control: the existing store-verified route is untouched."""
    _ask_log(home, answer="Delete ~/{qa-3506,qa-bridge-home}", expired=False)
    assert prov.verified_ask(S, ASK) is not None
    judge = _Judge("ALLOW ask=a-20fc")
    outcome = await _check(home, INCIDENT_RM, judge)
    assert outcome is None or not outcome.blocked, outcome and outcome.text


def test_unanswered_asks_lists_only_non_consent(home: Path) -> None:
    _ask_log(home, answer=None, expired=True)
    lapsed = prov.unanswered_asks(S)
    assert [(a.ask_id, a.status) for a in lapsed] == [(ASK, "timed_out")]
    assert "~/{qa-3506,qa-bridge-home}" in lapsed[0].text


def test_judge_prompt_says_a_timed_out_ask_is_not_consent() -> None:
    assert "TIMED OUT, expired, was withdrawn" in JUDGE_SYSTEM_PROMPT
    assert "nor the agent's own text saying an ask timed out" in JUDGE_SYSTEM_PROMPT
    assert "FORCED removal" in JUDGE_SYSTEM_PROMPT


# -- 3. chmod -R u+w is a forced removal on a foreign tree -------------------------


def _rank(home: Path, command: str) -> Any:
    return classify_bash(
        command, cwd=str(home), env=_env(home), scratchpad=_pad(home), session_id=S
    )


def test_chmod_r_uw_on_a_foreign_tree_is_high(home: Path) -> None:
    v = _rank(home, "chmod -R u+w ~/workspace/harness-cost-bench/coding")
    assert v.risk >= Risk.HIGH
    assert FORCED_REMOVAL_REASON in v.reasons


@pytest.mark.parametrize(
    "command",
    [
        "chmod -R +w ~/workspace/harness-cost-bench/coding",
        "chmod -R 755 ~/workspace/harness-cost-bench/coding",
        "chmod --recursive a+rwX ~/workspace/harness-cost-bench/coding",
        "chflags -R nouchg ~/workspace/harness-cost-bench/coding",
        "chflags nouchg,noschg ~/workspace/harness-cost-bench/coding",
        "chattr -R -i ~/workspace/harness-cost-bench/coding",
        # Review R1-m2: the same unlock through find, xargs, ACLs.
        "find ~/workspace/harness-cost-bench/coding -exec chmod u+w {} +",
        "find ~/workspace/harness-cost-bench/coding -type f -exec chmod 644 {} \\;",
        "cd ~/workspace/harness-cost-bench && find coding | xargs chmod u+w",
        "setfacl -R -m u:me:rwx ~/workspace/harness-cost-bench/coding",
        "setfacl -Rm u:me:rwx ~/workspace/harness-cost-bench/coding",
        "chmod -R -N ~/workspace/harness-cost-bench/coding",
    ],
)
def test_similar_unlocks_on_a_foreign_tree_are_high(home: Path, command: str) -> None:
    assert _rank(home, command).risk >= Risk.HIGH


@pytest.mark.parametrize(
    "command",
    [
        # Not a removal unlock: single file, protecting, or group/other only.
        "chmod u+w ~/workspace/harness-cost-bench/coding/generated/summary_tables.json",
        "chmod -R a-w ~/workspace/harness-cost-bench/coding",
        "chmod -R go+w ~/workspace/harness-cost-bench/coding",
        "chmod -R 555 ~/workspace/harness-cost-bench/coding",
        "chflags -R uchg ~/workspace/harness-cost-bench/coding",
        "find ~/workspace/harness-cost-bench/coding -exec chmod go+w {} +",
        "find ~/workspace/harness-cost-bench/coding | xargs chmod +x",
        "setfacl -m u:me:rwx ~/workspace/harness-cost-bench/coding",
        "chmod -N ~/workspace/harness-cost-bench/coding",
    ],
)
def test_non_unlocking_permission_changes_stay_low(home: Path, command: str) -> None:
    assert _rank(home, command).risk is Risk.LOW


def test_chmod_r_uw_in_own_scratchpad_stays_low(home: Path) -> None:
    own = Path(_pad(home)) / "rig"
    own.mkdir()
    assert _rank(home, f"chmod -R u+w {own}").risk is Risk.LOW


def test_chmod_r_uw_on_a_ledger_recorded_own_tree_stays_low(home: Path) -> None:
    own = home / "workspace" / "own-rig"
    own.mkdir(parents=True)
    prov.write_records(
        [(os.path.realpath(own), "mkdir")],
        owners=[S],
        creator=S,
        command=f"mkdir -p {own}",
        state_root=home / ".local-operator",
    )
    v = _rank(home, f"chmod -R u+w {own} && rm -rf {own}")
    assert v.risk is Risk.LOW, v.reasons


#: The incident's call 371, verbatim in shape.
INCIDENT_CHMOD_RM = (
    "chmod -R u+w ~/workspace/harness-cost-bench/coding 2>/dev/null || true\n"
    "rm -rf ~/workspace/harness-cost-bench/coding"
)


def test_the_plain_deep_delete_alone_is_still_low(home: Path) -> None:
    """Negative control for the row below: only the escalation changes the rank."""
    assert _rank(home, "rm -rf ~/workspace/harness-cost-bench/coding").risk is Risk.LOW


@pytest.mark.asyncio
async def test_chmod_then_rm_on_a_foreign_tree_reaches_the_judge_as_forced(home: Path) -> None:
    assert _rank(home, INCIDENT_CHMOD_RM).risk >= Risk.HIGH
    judge = _Judge("BLOCK: forced removal of protected files nobody named")
    outcome = await _check(home, INCIDENT_CHMOD_RM, judge)
    assert outcome is not None and outcome.blocked
    assert len(judge.prompts) == 1
    assert FORCED_REMOVAL_REASON in judge.prompts[0]
    raw = home / "workspace/harness-cost-bench/coding/generated/summary_tables.json"
    assert raw.exists()


@pytest.mark.parametrize(
    ("mode", "grants"),
    [
        ("u+w", True),
        ("+w", True),
        ("a=rwX", True),
        ("u+w,go-w", True),
        ("755", True),
        ("0644", True),
        ("go+w", False),
        ("a-w", False),
        ("555", False),
        ("u+x", False),
    ],
)
def test_chmod_grants_write(mode: str, grants: bool) -> None:
    assert _chmod_grants_write(mode) is grants


# -- 1. the model-facing copy ----------------------------------------------------


def test_system_prompt_and_ask_docs_say_a_timeout_is_not_consent() -> None:
    from local_operator.prompts_api import render_template
    from local_operator.tools import builtin

    queued = " ".join(render_template("system.md", {"ask_queued": True}).split())
    assert "take your own recommendation only where it is reversible" in queued
    assert "a timeout is never consent" in queued
    inline = " ".join(render_template("system.md", {"ask_inline": True}).split())
    assert "never for a delete, overwrite or other irreversible step" in inline
    # The description no longer repeats system.md's rule (wave 2): the queued
    # paragraph above renders whenever ``ask`` is held, and the timeout notice
    # itself carries it at the moment it applies.
    assert "Closing it is NOT consent" in builtin.ASK_UNANSWERED_TEXT


# -- 4. an unanswered ask lifts a DEEP path out of LOW -----------------------------
#
# Incident rows 302, 320 and 365 ran ``rm -rf`` on deep paths the timed-out ask
# a-20fc named (``~/hyperplane-cpp/build``, benchmark raw data): LOW, so the
# refusal in section 2 never saw them. Asking about a path admits removing it
# needs the operator; silence cannot make it LOW again.

DEEP_ASK = "a-77de"
DEEP_LABEL = "Delete ~/bench/rigs, ~/hyperplane-cpp/build and ~/osworld/runs*"
NOT_CONSENT_TEXT = (
    "You asked the operator about this path and got no answer; "
    "a timeout or no answer is not consent; wait for an answer or leave it"
)


def _deep_ask(
    home: Path,
    *,
    answer: str | None = None,
    expired: bool = True,
    session: str = S,
    age_ms: int = 0,
) -> None:
    now = int(time.time() * 1000) - age_ms
    expires = now - 60_000 if expired else now + 3_600_000
    rows: list[dict[str, Any]] = [
        {
            "v": 1,
            "kind": "queued",
            "ask_id": DEEP_ASK,
            "at": expires - 3_600_000,
            "created_at": expires - 3_600_000,
            "expires_at": expires,
            "timeout_s": 3600,
            "questions": [
                {
                    "id": "deep_cleanup",
                    "question": "Which build and benchmark trees should I delete?",
                    # The glob is LAST, as the incident's labels put it.
                    "options": [
                        {"label": "Keep everything", "description": "stop"},
                        {"label": DEEP_LABEL, "description": "reclaim ~40 GB"},
                    ],
                    "multi": False,
                    "question_id": "",
                    "recommended": 1,
                }
            ],
        }
    ]
    if answer is not None:
        rows.append(
            {
                "v": 1,
                "kind": "answered",
                "ask_id": DEEP_ASK,
                "at": now - 1000,
                "by": {"surface": "desktop"},
                "answers": {"deep_cleanup": [answer]},
            }
        )
    path = home / ".local-operator" / "sessions" / session / "asks.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


@pytest.fixture
def deep(home: Path) -> Path:
    for rel in ("bench/rigs/r1", "hyperplane-cpp/build/obj", "osworld/runs-3/raw", "other/x/y"):
        (home / rel).mkdir(parents=True)
    return home


DEEP_COMMANDS = [
    "rm -rf ~/hyperplane-cpp/build",
    "rm -rf ~/osworld/runs-3/raw",  # under the named glob ~/osworld/runs*
    "chmod -R u+w ~/bench/rigs && rm -rf ~/bench/rigs",
]


@pytest.mark.parametrize(
    "command",
    ["rm -rf ~/hyperplane-cpp/build", "rm -rf ~/osworld/runs-3/raw", "rm -rf ~/bench/rigs"],
)
def test_deep_paths_are_low_without_any_ask(deep: Path, command: str) -> None:
    """Control: the rows below change rank ONLY because of the ask."""
    assert _rank(deep, command).risk is Risk.LOW


@pytest.mark.asyncio
@pytest.mark.parametrize("command", DEEP_COMMANDS)
async def test_timed_out_ask_naming_a_deep_path_is_refused_without_judge(
    deep: Path, command: str
) -> None:
    _deep_ask(deep)
    judge = _Judge("ALLOW ask=a-77de")  # would be fooled if asked
    outcome = await _check(deep, command, judge)
    assert outcome is not None and outcome.blocked, command
    assert judge.prompts == []
    assert NOT_CONSENT_TEXT in outcome.text
    assert DEEP_ASK in outcome.text and "TIMED OUT" in outcome.text
    for rel in ("bench/rigs/r1", "hyperplane-cpp/build/obj", "osworld/runs-3/raw"):
        assert (deep / rel).is_dir()


@pytest.mark.asyncio
async def test_open_ask_naming_a_deep_path_says_wait_for_the_answer(deep: Path) -> None:
    _deep_ask(deep, expired=False)
    judge = _Judge("ALLOW")
    outcome = await _check(deep, "rm -rf ~/hyperplane-cpp/build", judge)
    assert outcome is not None and outcome.blocked
    assert judge.prompts == []
    assert "still open" in outcome.text
    assert "wait for the operator's answer" in outcome.text
    assert NOT_CONSENT_TEXT in outcome.text


@pytest.mark.asyncio
@pytest.mark.parametrize("command", DEEP_COMMANDS[:3])
async def test_approving_answer_lifts_the_deep_refusal(deep: Path, command: str) -> None:
    _deep_ask(deep, answer=DEEP_LABEL, expired=False)
    # Listed (it names paths), but its trusted answer consents to them.
    [ask] = prov.unanswered_asks(S)
    assert ask.approved == DEEP_LABEL
    assert prov.ask_consents(ask, str(deep / "osworld/runs-3/raw"), home=str(deep))
    judge = _Judge("ALLOW ask=a-77de")
    outcome = await _check(deep, command, judge)
    assert outcome is None or not outcome.blocked, outcome and outcome.text


@pytest.mark.asyncio
async def test_ledger_recorded_path_stays_low_despite_the_ask(deep: Path) -> None:
    build = deep / "hyperplane-cpp" / "build"
    prov.write_records(
        [(os.path.realpath(build), "mkdir")],
        owners=[S],
        creator=S,
        command=f"mkdir -p {build}",
        state_root=deep / ".local-operator",
    )
    _deep_ask(deep)
    from local_operator.guardrails.check import escalate_asked_targets

    command = "rm -rf ~/hyperplane-cpp/build"
    verdict = escalate_asked_targets(_rank(deep, command), _ctx(deep, command))
    assert verdict.risk is Risk.LOW, verdict.reasons
    judge = _Judge("BLOCK")
    outcome = await _check(deep, command, judge)
    assert outcome is None or not outcome.blocked
    assert judge.prompts == []


FAMILY_PARENT = "beef00000002"  # synthetic manager of S for the Q1 cases below


def _parent_records(home: Path, rel: str) -> Path:
    """S is FAMILY_PARENT's subagent; FAMILY_PARENT's ledger records ``~/rel``."""
    state = home / ".local-operator"
    prov.record_lineage(S, origin="subagent", parent=FAMILY_PARENT, state_root=state)
    target = home / rel
    target.mkdir(parents=True, exist_ok=True)
    prov.write_records(
        [(os.path.realpath(target), "mkdir")],
        owners=[FAMILY_PARENT],
        creator=FAMILY_PARENT,
        command=f"mkdir -p {target}",
        state_root=state,
    )
    return target


@pytest.mark.asyncio
async def test_parents_ledger_row_does_not_lift_the_parents_lapsed_ask(deep: Path) -> None:
    """QA Q1, PR #3964: family ownership must not exempt a path an unanswered ask names.

    The parent recorded ``~/hyperplane-cpp/build`` and asked about deleting
    it; the ask timed out. Its subagent S deleting that path is refused as a
    lapsed ask, without a judge, exactly as on main. Before the fix,
    ``escalate_asked_targets`` read the parent's row as S's own and let the
    delete run unexamined.
    """
    _parent_records(deep, "hyperplane-cpp/build")
    _deep_ask(deep, session=FAMILY_PARENT)
    from local_operator.guardrails.check import escalate_asked_targets

    command = "rm -rf ~/hyperplane-cpp/build"
    verdict = escalate_asked_targets(_rank(deep, command), _ctx(deep, command))
    assert verdict.risk is Risk.HIGH, verdict.reasons
    judge = _Judge("ALLOW")
    outcome = await _check(deep, command, judge)
    assert outcome is not None and outcome.blocked
    assert judge.prompts == []
    assert NOT_CONSENT_TEXT in outcome.text


@pytest.mark.asyncio
@pytest.mark.parametrize("rel", ["hyperplane-cpp/build", "qa-3506"])
async def test_parents_ledger_row_still_lowers_a_delete_no_ask_names(deep: Path, rel: str) -> None:
    """Family ownership (#3964) still applies when no ask names the path.

    ``~/qa-3506`` is a top-level home directory (HIGH unless owned): it must
    rank LOW on the parent's ledger row, and run with no judge.
    """
    _parent_records(deep, rel)
    command = f"rm -rf ~/{rel}"
    assert _rank(deep, command).risk is Risk.LOW
    judge = _Judge("BLOCK")
    outcome = await _check(deep, command, judge)
    assert outcome is None or not outcome.blocked, outcome and outcome.text
    assert judge.prompts == []


@pytest.mark.asyncio
async def test_unrelated_deep_path_stays_low_with_the_ask_open(deep: Path) -> None:
    _deep_ask(deep)
    from local_operator.guardrails.check import escalate_asked_targets

    command = "rm -rf ~/other/x"
    assert escalate_asked_targets(_rank(deep, command), _ctx(deep, command)).risk is Risk.LOW
    judge = _Judge("BLOCK")
    outcome = await _check(deep, command, judge)
    assert outcome is None or not outcome.blocked
    assert judge.prompts == []


def test_ask_reaches_equal_under_above_and_glob(deep: Path) -> None:
    home = str(deep)
    text = f"Q: x | options: {DEEP_LABEL}"
    reach = lambda p: prov.ask_reaches(text, str(deep / p), home=home)  # noqa: E731
    assert reach("hyperplane-cpp/build")
    assert reach("hyperplane-cpp/build/obj")
    assert reach("hyperplane-cpp")
    assert reach("osworld/runs-3/raw")
    assert reach("osworld")  # the glob's parent: removes every runs*
    assert reach("bench")  # an ancestor of a literal name
    assert not reach("other/x")
    assert not reach("osworld-old")
    # Naming ``~`` itself never makes every path under it "asked about".
    assert not prov.ask_reaches("Q: clean ~ ?", str(deep / "other/x"), home=home)


CHILD = "c0ffee000001"  # synthetic subagent of S
PARENT = "beef00000001"  # synthetic manager of S


@pytest.mark.asyncio
@pytest.mark.parametrize("asker", ["child", "parent"])
async def test_an_unanswered_ask_anywhere_in_the_family_counts(deep: Path, asker: str) -> None:
    state = deep / ".local-operator"
    prov.record_lineage(CHILD, origin="subagent", parent=S, state_root=state)
    prov.record_lineage(S, origin="subagent", parent=PARENT, state_root=state)
    _deep_ask(deep, session=CHILD if asker == "child" else PARENT)
    assert [a.ask_id for a in prov.unanswered_asks(S)] == [DEEP_ASK]
    judge = _Judge("ALLOW")
    outcome = await _check(deep, "rm -rf ~/hyperplane-cpp/build", judge)
    assert outcome is not None and outcome.blocked
    assert judge.prompts == []
    assert NOT_CONSENT_TEXT in outcome.text


def test_an_unrelated_sessions_ask_does_not_count(deep: Path) -> None:
    _deep_ask(deep, session="0123456789ab")  # no lineage row: not family
    assert prov.unanswered_asks(S) == []


# -- Review R1 remediation (PR #3966) -------------------------------------------------


class _QueueSession:
    """What ``AskQueue`` needs of a session, and nothing else (as in test_queue.py)."""

    def __init__(self) -> None:
        self.transcript = type("T", (), {"has_entry": lambda self, _id: True})()
        self.spawned: list[Any] = []

    def credential_names(self) -> list[str]:
        return []

    def ask_reach(self) -> list[str]:
        return []

    def _spawn_background(self, coro: Any) -> Any:
        import asyncio

        task = asyncio.ensure_future(coro)
        self.spawned.append(task)
        return task

    async def deliver_ask_messages(self, messages: Any) -> None:
        return None


def _queued_deep_ask(deep: Path, clock: dict[str, int]) -> tuple[Any, str]:
    """``DEEP_LABEL``'s ask through the REAL queue API, already past its 1 h deadline."""
    from local_operator.asks.queue import AskQueue

    queue = AskQueue(
        _QueueSession(),
        config_dir=deep / ".local-operator",
        session_id=S,
        clock=lambda: clock["now"],
    )
    receipt = queue.enqueue(
        [
            {
                "id": "deep_cleanup",
                "question": "Which build and benchmark trees should I delete?",
                "options": [
                    {"label": "Keep everything", "description": "stop"},
                    {"label": DEEP_LABEL, "description": "reclaim ~40 GB"},
                ],
                "multi": False,
                "recommended": 1,
            }
        ],
        3600,
    )
    assert receipt["ok"], receipt
    ask_id = receipt["details"]["ask_id"]
    clock["now"] = int(time.time() * 1000)  # the deadline is behind us now
    return queue, ask_id


def _settle(queue: Any, ask_id: str, how: str) -> dict[str, Any]:
    if how == "answered_in_chat":
        # The AGENT records a chat "answer" after the timeout (review R1-M1).
        return queue.withdraw(
            ask_id, reason="answered_in_chat", answers={"deep_cleanup": ["yes delete them"]}
        )
    if how == "moot":
        return queue.withdraw(ask_id, reason="moot")
    if how == "relay":
        return queue.respond(ask_id, {"deep_cleanup": [DEEP_LABEL]}, by="relay")
    if how == "declined":
        return queue.decline(ask_id, by="desktop")
    if how == "keep_everything":
        # The operator, on a trusted surface, said NO.
        return queue.respond(ask_id, {"deep_cleanup": ["Keep everything"]}, by="desktop")
    if how == "timed_out":
        return {"ok": True}
    raise AssertionError(how)


SETTLES = ["timed_out", "answered_in_chat", "moot", "relay", "declined", "keep_everything"]


@pytest.mark.asyncio
@pytest.mark.parametrize("how", SETTLES)
async def test_no_end_state_but_a_trusted_approval_lifts_the_refusal(deep: Path, how: str) -> None:
    """Review R1-M1: blocks = asks naming the path − asks granting consent for it."""
    clock = {"now": int(time.time() * 1000) - 2 * 3_600_000}
    queue, ask_id = _queued_deep_ask(deep, clock)
    settled = _settle(queue, ask_id, how)
    assert settled.get("ok"), settled
    assert prov.verified_ask(S, ask_id) is None or how == "keep_everything"
    for command in ("rm -rf ~/hyperplane-cpp/build", "rm -rf ~/osworld/runs-3/raw"):
        judge = _Judge(f"ALLOW ask={ask_id}")  # would be fooled
        outcome = await _check(deep, command, judge)
        assert outcome is not None and outcome.blocked, (how, command)
        assert judge.prompts == [], (how, command)
        assert ask_id in outcome.text and "NOT consent" in outcome.text
    assert (deep / "hyperplane-cpp/build/obj").is_dir()


@pytest.mark.asyncio
async def test_a_trusted_approval_through_the_queue_still_allows(deep: Path) -> None:
    """Control for the row above: the operator's own approving pick on desktop."""
    clock = {"now": int(time.time() * 1000) - 60_000}
    queue, ask_id = _queued_deep_ask(deep, clock)
    clock["now"] = int(time.time() * 1000) - 60_000 + 1000  # before the deadline
    assert queue.respond(ask_id, {"deep_cleanup": [DEEP_LABEL]}, by="desktop")["ok"]
    judge = _Judge("BLOCK")
    outcome = await _check(deep, "rm -rf ~/hyperplane-cpp/build", judge)
    assert outcome is None or not outcome.blocked, outcome and outcome.text
    assert judge.prompts == []


DAY_MS = 24 * 3_600_000


def _ask_text(home: Path, question: str, *, age_ms: int, label: str = "Go ahead") -> None:
    """A timed-out ask with a free question, ``age_ms`` old."""
    created = int(time.time() * 1000) - age_ms
    row = {
        "v": 1,
        "kind": "queued",
        "ask_id": "a-0be1",
        "at": created,
        "created_at": created,
        "expires_at": created + 3_600_000,
        "timeout_s": 3600,
        "questions": [
            {
                "id": "q",
                "question": question,
                "options": [{"label": label, "description": ""}, {"label": "Skip"}],
                "multi": False,
                "question_id": "",
            }
        ],
    }
    path = home / ".local-operator" / "sessions" / S / "asks.jsonl"
    with path.open("a") as fh:
        fh.write(json.dumps(row) + "\n")


@pytest.fixture
def rig(home: Path) -> Path:
    (home / "workspace" / "rig-1").mkdir(parents=True)
    (home / "workspace" / "rig-1" / "f.txt").write_text("a\n")
    (home / "Documents").mkdir()
    _ask_text(home, "Should the report go in ~/workspace or ~/Documents?", age_ms=3 * DAY_MS)
    return home


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "command",
    ["echo b > ~/workspace/rig-1/f.txt", "sed -i '' s/a/b/ ~/workspace/rig-1/f.txt"],
)
async def test_an_unrelated_broad_ask_leaves_overwrites_low(rig: Path, command: str) -> None:
    """Review R1-M2: overwrites are not lifted by an ask at all."""
    from local_operator.guardrails.check import escalate_asked_targets

    assert escalate_asked_targets(_rank(rig, command), _ctx(rig, command)).risk is Risk.LOW
    judge = _Judge("BLOCK")
    outcome = await _check(rig, command, judge)
    assert outcome is None or not outcome.blocked, outcome and outcome.text
    assert judge.prompts == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "command",
    ["mv ~/workspace/rig-1/f.txt ~/workspace/rig-1/g.txt", "rm ~/workspace/rig-1/f.txt"],
)
async def test_under_a_broad_named_dir_the_judge_decides(rig: Path, command: str) -> None:
    """Review R1-M2: under ``~/workspace`` (a direct child of $HOME) → judge, not refusal."""
    judge = _Judge("BLOCK unrelated to the ask")
    outcome = await _check(rig, command, judge)
    # The JUDGE decided (one prompt), not the deterministic lapsed-ask branch.
    assert outcome is not None and NOT_CONSENT_TEXT not in outcome.text
    assert len(judge.prompts) == 1
    assert "a-0be1 names it but was TIMED OUT with no answer: NOT consent" in judge.prompts[0]


@pytest.mark.asyncio
async def test_a_dir_with_many_entries_is_broad(home: Path) -> None:
    big = home / "data" / "big"
    big.mkdir(parents=True)
    for n in range(1001):
        (big / f"e{n}").mkdir()
    _ask_text(home, "Is ~/data/big still needed?", age_ms=DAY_MS)
    judge = _Judge("BLOCK")
    outcome = await _check(home, "rm -rf ~/data/big/e5", judge)
    assert outcome is not None and outcome.blocked
    assert len(judge.prompts) == 1  # the judge refused it, not the deterministic branch
    # The named dir itself keeps the deterministic refusal.
    judge = _Judge("ALLOW")
    outcome = await _check(home, "rm -rf ~/data/big", judge)
    assert outcome is not None and outcome.blocked and judge.prompts == []


@pytest.mark.asyncio
async def test_an_ask_older_than_the_bound_stops_escalating(deep: Path) -> None:
    from local_operator.guardrails.check import escalate_asked_targets

    _deep_ask(deep, age_ms=prov.ASK_ESCALATION_MAX_AGE_MS + DAY_MS)
    command = "rm -rf ~/hyperplane-cpp/build"
    assert escalate_asked_targets(_rank(deep, command), _ctx(deep, command)).risk is Risk.LOW
    judge = _Judge("BLOCK")
    outcome = await _check(deep, command, judge)
    assert outcome is None or not outcome.blocked
    assert judge.prompts == []
    # Inside the bound it still refuses.
    (deep / ".local-operator" / "sessions" / S / "asks.jsonl").unlink()
    _deep_ask(deep, age_ms=prov.ASK_ESCALATION_MAX_AGE_MS - DAY_MS)
    outcome = await _check(deep, command, _Judge("ALLOW"))
    assert outcome is not None and outcome.blocked


@pytest.mark.asyncio
async def test_an_overwrite_that_is_high_anyway_gets_the_fact_not_a_refusal(home: Path) -> None:
    """Overwrites are never refused by an ask, but the judge still hears it."""
    note = home / ".local-operator" / "notes.txt"
    note.write_text("x")
    _ask_text(home, "May I rewrite ~/.local-operator/notes.txt?", age_ms=DAY_MS)
    command = "echo y > ~/.local-operator/notes.txt"
    assert _rank(home, command).risk is Risk.HIGH  # S10, without any ask
    judge = _Judge("BLOCK")
    outcome = await _check(home, command, judge)
    assert outcome is not None and outcome.blocked
    assert len(judge.prompts) == 1
    assert "a-0be1 names it but was TIMED OUT with no answer: NOT consent" in judge.prompts[0]


def _case_insensitive(path: Path) -> bool:
    return os.path.exists(str(path).swapcase()) if path.exists() else False


@pytest.mark.asyncio
async def test_a_case_respelling_is_still_reached(deep: Path) -> None:
    """Review R1-m1: APFS folds case, so the respelled path is the asked one."""
    if sys.platform != "darwin" or not _case_insensitive(deep / "hyperplane-cpp"):
        pytest.skip("needs a case-insensitive Darwin volume")
    _deep_ask(deep)
    judge = _Judge("ALLOW")
    outcome = await _check(deep, "rm -rf ~/Hyperplane-CPP/Build", judge)
    assert outcome is not None and outcome.blocked and judge.prompts == []


@pytest.mark.asyncio
async def test_removing_own_scratchpad_link_to_an_asked_path_stays_low(deep: Path) -> None:
    """Review R1-m3: ``rm link`` unlinks the alias; ``rm -rf link/`` empties the pointee."""
    from local_operator.guardrails.check import escalate_asked_targets

    _deep_ask(deep)
    link = Path(_pad(deep)) / "l"
    link.symlink_to(deep / "hyperplane-cpp" / "build")
    command = f"rm -rf {link}"
    assert escalate_asked_targets(_rank(deep, command), _ctx(deep, command)).risk is Risk.LOW
    judge = _Judge("BLOCK")
    outcome = await _check(deep, f"rm {link}", judge)
    assert outcome is None or not outcome.blocked
    outcome = await _check(deep, f"rm -rf {link}/", _Judge("ALLOW"))
    assert outcome is not None and outcome.blocked
    assert (deep / "hyperplane-cpp/build/obj").is_dir()


@pytest.mark.asyncio
async def test_a_path_named_only_in_a_description_is_refused(deep: Path) -> None:
    """Review R1-m4 / QA Q2: descriptions only ever refuse."""
    (deep / "descr-only" / "tree").mkdir(parents=True)
    created = int(time.time() * 1000) - 2 * 3_600_000
    row = {
        "v": 1,
        "kind": "queued",
        "ask_id": "a-de5c",
        "at": created,
        "created_at": created,
        "expires_at": created + 3_600_000,
        "timeout_s": 3600,
        "questions": [
            {
                "id": "q",
                "question": "Clear the stale tree?",
                "options": [
                    {"label": "Yes, clear it", "description": "rm -rf ~/descr-only/tree"},
                    {"label": "Keep it"},
                ],
                "multi": False,
                "question_id": "",
            }
        ],
    }
    with (deep / ".local-operator" / "sessions" / S / "asks.jsonl").open("a") as fh:
        fh.write(json.dumps(row) + "\n")
    judge = _Judge("ALLOW")
    outcome = await _check(deep, "rm -rf ~/descr-only/tree", judge)
    assert outcome is not None and outcome.blocked and judge.prompts == []


def test_preview_reports_the_deterministic_refusal(deep: Path) -> None:
    """QA R1-Q1: preview names the refusal the checkpoint makes, not a judge."""
    from local_operator.guardrails.preview import BUILTIN_REFUSE_LAPSED

    _deep_ask(deep)
    command = "rm -rf ~/hyperplane-cpp/build"
    report = preview_tool_call(
        "bash", {"command": command}, _ctx(deep, command), settings=GuardSettings()
    )
    assert report.builtin_effect == BUILTIN_REFUSE_LAPSED
    assert report.action == "refuse"
    assert NOT_CONSENT_TEXT in report.builtin_text


# -- 9. one answered ask: consent, refusal and the verifier read one scope -------


def _answered_ask(
    home: Path,
    ask_id: str,
    question: str,
    options: list[dict[str, str]],
    *,
    answer: str,
    surface: str = "desktop",
    command: str | None = None,
) -> None:
    """One ask, answered ``answer`` on ``surface`` (``desktop`` is trusted)."""
    now = int(time.time() * 1000)
    extra = {"command": command, "command_target": "agent"} if command else {}
    rows = [
        {
            "v": 1,
            "kind": "queued",
            "ask_id": ask_id,
            "at": now - 120_000,
            "created_at": now - 120_000,
            "expires_at": now + 3_600_000,
            "timeout_s": 3600,
            "questions": [
                {
                    "id": "q",
                    "question": question,
                    "options": options,
                    "multi": False,
                    "question_id": "",
                    **extra,
                }
            ],
        },
        {
            "v": 1,
            "kind": "answered",
            "ask_id": ask_id,
            "at": now - 1000,
            "by": {"surface": surface},
            "answers": {"q": [answer]},
        },
    ]
    path = home / ".local-operator" / "sessions" / S / "asks.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


@pytest.mark.asyncio
async def test_later_exact_approval_supersedes_only_the_earlier_timeout(deep: Path) -> None:
    """Real store, two separate asks: timeout is silence, not a permanent veto."""
    _deep_ask(deep, expired=True, age_ms=120_000)
    _answered_ask(
        deep,
        "a-new",
        "Delete ~/hyperplane-cpp/build?",
        [{"label": "Approve"}, {"label": "Keep"}],
        answer="Approve",
    )
    from local_operator.guardrails.check import _ask_block

    path = str(deep / "hyperplane-cpp/build")
    assert (
        _ask_block(
            prov.unanswered_asks(S),
            path,
            home=str(deep),
            cwd=str(deep),
            stop=(),
            escalating=True,
            session_id=S,
            env=_env(deep),
        )
        is None
    )
    judge = _Judge("BLOCK")
    result = await _check(deep, "rm -rf ~/hyperplane-cpp/build", judge)
    assert result is None or not result.blocked, result and result.text
    assert not judge.prompts


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("options", "pick"),
    [
        ([{"label": "Remove all"}, {"label": "Remove none"}], "Remove none"),
        ([{"label": "Delete everything"}, {"label": "Delete nothing"}], "Delete nothing"),
        # The pick names another path; build is only in the option not picked.
        (
            [{"label": "Delete ~/hyperplane-cpp/build"}, {"label": "Delete ~/other/x"}],
            "Delete ~/other/x",
        ),
        ([{"label": "Delete next week"}, {"label": "Keep"}], "Delete next week"),
        # QA Q-R3-1: a pick that narrows the question to a glob or a path that
        # does not exist still narrows it; the question's list is not chosen.
        (
            [{"label": "Approve removing only ~/hyperplane-cpp/*/cache"}, {"label": "Keep"}],
            "Approve removing only ~/hyperplane-cpp/*/cache",
        ),
        (
            [{"label": "Approve removing ~/no-such-dir"}, {"label": "Keep"}],
            "Approve removing ~/no-such-dir",
        ),
        # Review SEC-R3-B2: the other path named without ``~``/``/``.
        ([{"label": "Delete the second one"}, {"label": "Keep"}], "Delete the second one"),
        ([{"label": "Delete x"}, {"label": "Keep"}], "Delete x"),
        ([{"label": "Remove only x"}, {"label": "Keep"}], "Remove only x"),
        ([{"label": "Delete other/x"}, {"label": "Keep"}], "Delete other/x"),
        ([{"label": "Delete $HOME/other/x"}, {"label": "Keep"}], "Delete $HOME/other/x"),
    ],
    ids=[
        "remove-none",
        "delete-nothing",
        "other-path-pick",
        "next-week",
        "narrowing-glob",
        "narrowing-missing-path",
        "second-one",
        "bare-name",
        "only-bare-name",
        "relative",
        "dollar-home",
    ],
)
async def test_pick_that_approves_no_named_path_cannot_supersede_timeout(
    deep: Path, options: list[dict[str, str]], pick: str
) -> None:
    """Review SEC-R2-B2/m1 (PR #4493): the question's paths are reached only
    through a pick that names none and approves deletion unconditionally."""
    _deep_ask(deep, expired=True, age_ms=120_000)
    _answered_ask(
        deep, "a-feed", "Delete ~/hyperplane-cpp/build or ~/other/x?", options, answer=pick
    )
    judge = _Judge("ALLOW ask=a-feed")
    result = await _check(deep, "rm -rf ~/hyperplane-cpp/build", judge)
    assert result is not None and result.blocked
    assert not judge.prompts
    assert (deep / "hyperplane-cpp/build").is_dir()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("question", "pick", "allowed"),
    [
        # Review SEC-R4-B1: a descriptor narrows "all" to a subset.
        (
            "Delete ~/hyperplane-cpp/build (dirty) or ~/other/x (clean)?",
            "Delete all clean worktrees",
            False,
        ),
        ("Delete ~/hyperplane-cpp/build or ~/other/x?", "Delete all caches", False),
        # Review SEC-R4-M1: only a size in parentheses is dropped.
        ("Delete ~/hyperplane-cpp/build or ~/other/x?", "Delete them (only x)", False),
        # Review SEC-R5-M1: two numbers state a subset of the list.
        (
            "Delete ~/hyperplane-cpp/build (dirty) or ~/other/x (clean)?",
            "Delete all clean worktrees (1 of 2)",
            False,
        ),
        (
            "Delete ~/hyperplane-cpp/build (dirty) or ~/other/x (clean)?",
            "Delete all 1 clean worktree of the 2",
            False,
        ),
        # Controls: a stated count covering the list, a plain plural, and
        # (SEC-R4-m1) a single path ending in "?" counted once.
        (
            "Delete ~/hyperplane-cpp/build or ~/other/x?",
            "Delete all 2 clean worktrees (~1.2 GiB)",
            True,
        ),
        ("Delete ~/hyperplane-cpp/build or ~/other/x?", "Delete both", True),
        ("Delete ~/hyperplane-cpp/build?", "Delete it", True),
    ],
    ids=[
        "clean-subset",
        "caches-subset",
        "paren-only",
        "one-of-two",
        "one-of-the-two",
        "counted",
        "both",
        "single-it",
    ],
)
async def test_descriptor_and_parenthesis_picks_after_timeout(
    deep: Path, question: str, pick: str, allowed: bool
) -> None:
    """PR #4493 round 4: which listed paths a path-less pick covers."""
    _deep_ask(deep, expired=True, age_ms=120_000)
    _answered_ask(deep, "a-feed", question, [{"label": pick}, {"label": "Keep"}], answer=pick)
    judge = _Judge("ALLOW ask=a-feed")
    result = await _check(deep, "rm -rf ~/hyperplane-cpp/build", judge)
    blocked = result is not None and result.blocked
    assert blocked is (not allowed), result and result.text
    assert not judge.prompts
    if not allowed:
        assert (deep / "hyperplane-cpp/build").is_dir()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "question",
    [
        "Keep ~/hyperplane-cpp/build instead of deleting it?",
        "Back up ~/hyperplane-cpp/build before deleting?",
        "Delete ~/hyperplane-cpp/build after backing it up?",
    ],
)
async def test_preservation_yes_cannot_supersede_timeout_at_checkpoint_and_preview(
    deep: Path, question: str
) -> None:
    from local_operator.guardrails.check import CallContext, GuardSettings
    from local_operator.guardrails.preview import preview_tool_call

    _deep_ask(deep, expired=True, age_ms=120_000)
    _answered_ask(deep, "a-feed", question, [{"label": "Yes"}, {"label": "No"}], answer="Yes")
    judge = _Judge("ALLOW ask=a-feed")
    command = "rm -rf ~/hyperplane-cpp/build"
    ctx = CallContext(cwd=str(deep), env=_env(deep), scratchpad=_pad(deep), session_id=S)
    preview = preview_tool_call("bash", {"command": command}, ctx, settings=GuardSettings())
    result = await _check(deep, command, judge)
    assert result is not None and result.blocked
    assert preview.builtin_effect != "pass"
    assert not judge.prompts
    assert (deep / "hyperplane-cpp/build").is_dir()


@pytest.mark.parametrize(
    "mode",
    [
        "earlier",
        "same-ask-revision",
        "glob",
        "ancestor",
        "typed",
        "remote",
        "declined",
        "pending",
        "expired-answer",
        "sibling",
        "brace",
        "revision-veto",
        "revision-approval",
        "description",
        "unpicked",
        "keep-yes",
        "backup-yes",
        "conditional-yes",
    ],
)
def test_lapsed_supersession_requires_exact_later_trusted_answer(deep: Path, mode: str) -> None:
    from local_operator.guardrails.check import _ask_block

    _deep_ask(deep, expired=True, age_ms=300_000)
    question = "Delete ~/hyperplane-cpp/build?"
    if mode == "glob":
        question = "Delete ~/hyperplane-cpp/bui*?"
    elif mode == "ancestor":
        question = "Delete ~/hyperplane-cpp?"
    elif mode == "brace":
        question = "Delete ~/hyperplane-cpp/{build,other}?"
    elif mode == "keep-yes":
        question = "Keep ~/hyperplane-cpp/build instead of deleting it?"
    elif mode == "backup-yes":
        question = "Back up ~/hyperplane-cpp/build before deleting?"
    elif mode == "conditional-yes":
        question = "Delete ~/hyperplane-cpp/build after backing it up?"
    surface = "remote" if mode == "remote" else "desktop"
    options = (
        [{"label": "Yes"}, {"label": "No"}]
        if mode.endswith("-yes")
        else [{"label": "Approve"}, {"label": "Keep"}]
    )
    _answered_ask(
        deep,
        "a-feed",
        question,
        options,
        answer=(
            "Yes"
            if mode.endswith("-yes")
            else "All" if mode == "typed" else "Keep" if mode == "revision-approval" else "Approve"
        ),
        surface=surface,
    )
    log = deep / ".local-operator" / "sessions" / S / "asks.jsonl"
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    now = int(time.time() * 1000)
    if mode == "earlier":
        rows[-2]["at"] = now - 500_000
        rows[-1]["at"] = now - 450_000
    elif mode == "pending":
        rows.append(
            {
                "v": 1,
                "kind": "queued",
                "ask_id": "a-1234",
                "at": now - 500,
                "expires_at": now + 60000,
                "questions": rows[-2]["questions"],
            }
        )
    elif mode == "declined":
        rows[-1]["answers"] = {"q": ["Keep"]}
    elif mode in ("description", "unpicked"):
        q = rows[-2]["questions"][0]
        q["question"] = "Which tree?"
        q["options"] = [
            {"label": "Delete ~/hyperplane-cpp/other", "description": "~/hyperplane-cpp/build"},
            {"label": "Delete ~/hyperplane-cpp/build"},
        ]
        rows[-1]["answers"] = {"q": ["Delete ~/hyperplane-cpp/other"]}
    elif mode == "expired-answer":
        rows[0]["expires_at"] = now - 9 * 24 * 3600 * 1000
        rows[0]["at"] = rows[0]["expires_at"] - 3600000
        rows.insert(
            1,
            {
                "v": 1,
                "kind": "answered",
                "ask_id": DEEP_ASK,
                "at": now - 8 * 24 * 3600 * 1000,
                "by": {"surface": "desktop"},
                "answers": {"deep_cleanup": ["Keep everything"]},
            },
        )
    elif mode in ("revision-veto", "revision-approval"):
        rows.append(
            {
                "v": 1,
                "kind": "revised",
                "ask_id": "a-feed",
                "at": now - 100,
                "by": {"surface": "desktop"},
                "answers": {"q": ["Keep" if mode == "revision-veto" else "Approve"]},
            }
        )
    elif mode == "same-ask-revision":
        rows = [
            rows[0],
            {
                "v": 1,
                "kind": "answered",
                "ask_id": DEEP_ASK,
                "at": now - 200_000,
                "by": {"surface": "desktop"},
                "answers": {"deep_cleanup": ["Keep everything"]},
            },
            {
                "v": 1,
                "kind": "revised",
                "ask_id": DEEP_ASK,
                "at": now - 100,
                "by": {"surface": "desktop"},
                "answers": {"deep_cleanup": [DEEP_LABEL]},
            },
        ]
    if mode == "sibling":
        other = "feed00000001"
        prov.record_lineage(
            S, origin="subagent", parent=PARENT, state_root=deep / ".local-operator"
        )
        prov.record_lineage(
            other, origin="subagent", parent=PARENT, state_root=deep / ".local-operator"
        )
        other_log = deep / ".local-operator" / "sessions" / other / "asks.jsonl"
        other_log.parent.mkdir(parents=True)
        other_log.write_text("".join(json.dumps(row) + "\n" for row in rows[1:]))
        rows = rows[:1]
    log.write_text("".join(json.dumps(row) + "\n" for row in rows))
    path = str(deep / "hyperplane-cpp/build")
    hit = _ask_block(
        prov.unanswered_asks(S),
        path,
        home=str(deep),
        cwd=str(deep),
        stop=(),
        escalating=False,
        session_id=S,
        env=_env(deep),
    )
    # ``expired-answer`` (2026-11-25): the expired ask's late decline is 8 days
    # older than the exact approval, so the newest answer decides. It is still
    # never displaced as SILENCE; a late decline newer than the approval holds
    # (``test_expired_late_decline_newer_than_approval_still_holds``).
    if mode in ("brace", "revision-approval", "same-ask-revision", "expired-answer"):
        assert hit is None
    else:
        assert hit is not None, mode


@pytest.mark.asyncio
@pytest.mark.parametrize("newer", ["pending", "declined"])
async def test_later_contrary_ask_beats_older_approval_in_check_and_preview(
    deep: Path, newer: str
) -> None:
    _answered_ask(
        deep,
        "a-feed",
        "Delete ~/hyperplane-cpp/build?",
        [{"label": "Approve"}, {"label": "Keep"}],
        answer="Approve",
    )
    log = deep / ".local-operator" / "sessions" / S / "asks.jsonl"
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    now = int(time.time() * 1000)
    rows.append(
        {
            "v": 1,
            "kind": "queued",
            "ask_id": "a-1234",
            "at": now,
            "expires_at": now + 60000,
            "questions": rows[0]["questions"],
        }
    )
    if newer == "declined":
        rows.append(
            {
                "v": 1,
                "kind": "answered",
                "ask_id": "a-1234",
                "at": now + 1,
                "by": {"surface": "desktop"},
                "answers": {"q": ["Keep"]},
            }
        )
    log.write_text("".join(json.dumps(row) + "\n" for row in rows))
    command = "rm -rf ~/hyperplane-cpp/build"
    judge = _Judge("ALLOW ask=a-feed")
    result = await _check(deep, command, judge)
    assert result is not None and result.blocked
    assert not judge.prompts
    preview = preview_tool_call(
        "bash", {"command": command}, _ctx(deep, command), settings=GuardSettings()
    )
    assert preview.builtin_effect == "refuse-lapsed-ask", preview.render()


DESCR_OPTIONS = [
    {"label": "Yes, clear it", "description": "rm -rf ~/descr-only/tree"},
    {"label": "Keep it"},
]


@pytest.mark.asyncio
async def test_a_trusted_yes_to_a_description_only_path_is_not_refused(deep: Path) -> None:
    """Review R2-M3 (probe P4): the operator said yes on desktop to an ask that
    names the path only in the picked option's description. Consent and refusal
    read that answered question the same way (question + labels; a description
    names nothing, live a-1d54), so the path keeps its base rank as on main:
    never the refusal that says he did not approve."""
    (deep / "descr-only" / "tree").mkdir(parents=True)
    _answered_ask(deep, "a-de5d", "Clear the stale tree?", DESCR_OPTIONS, answer="Yes, clear it")
    [ask] = prov.unanswered_asks(S)
    assert ask.answer is not None and "descr-only" not in ask.text
    judge = _Judge("BLOCK")
    outcome = await _check(deep, "rm -rf ~/descr-only/tree", judge)
    assert outcome is None or not outcome.blocked, outcome and outcome.text
    assert judge.prompts == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("answer", "surface"),
    [
        ("Keep it", "desktop"),  # trusted, but declining
        ("Yes, clear it", "chat"),  # the agent's own answered_in_chat
        ("Yes, clear it", "relay"),
        ("Yes, clear it", "bridge"),
    ],
)
async def test_a_description_only_path_without_a_trusted_yes_still_refuses(
    deep: Path, answer: str, surface: str
) -> None:
    """R1-M1 holds for the description-only shape: no trusted non-declining answer,
    so the deterministic refusal stands with no judge call."""
    (deep / "descr-only" / "tree").mkdir(parents=True)
    _answered_ask(
        deep, "a-de5e", "Clear the stale tree?", DESCR_OPTIONS, answer=answer, surface=surface
    )
    judge = _Judge("ALLOW ask=a-de5e")
    outcome = await _check(deep, "rm -rf ~/descr-only/tree", judge)
    assert outcome is not None and outcome.blocked and judge.prompts == []


# -- Suggested-command field (PR #3926): refusal-side naming only ------------------

CMD_OPTIONS = [{"label": "Yes, run it"}, {"label": "No"}]
CMD_ONLY = "rm -rf ~/bench/rigs"


@pytest.mark.asyncio
async def test_a_timed_out_ask_naming_a_path_only_in_its_command_is_refused(
    deep: Path,
) -> None:
    """The ask names ``~/bench/rigs`` only in its suggested ``command`` and timed
    out: the path is refused deterministically, not left LOW."""
    created = int(time.time() * 1000) - 2 * 3_600_000
    row = {
        "v": 1,
        "kind": "queued",
        "ask_id": "a-c0d1",
        "at": created,
        "created_at": created,
        "expires_at": created + 3_600_000,
        "timeout_s": 3600,
        "questions": [
            {
                "id": "q",
                "question": "Free the disk space?",
                "options": CMD_OPTIONS,
                "multi": False,
                "question_id": "",
                "command": CMD_ONLY,
                "command_target": "agent",
            }
        ],
    }
    with (deep / ".local-operator" / "sessions" / S / "asks.jsonl").open("a") as fh:
        fh.write(json.dumps(row) + "\n")
    judge = _Judge("ALLOW ask=a-c0d1")  # would be fooled if asked
    outcome = await _check(deep, CMD_ONLY, judge)
    assert outcome is not None and outcome.blocked
    assert judge.prompts == []
    assert NOT_CONSENT_TEXT in outcome.text
    assert (deep / "bench/rigs/r1").is_dir()


@pytest.mark.asyncio
async def test_a_trusted_yes_does_not_lift_a_path_named_only_in_its_command(
    deep: Path,
) -> None:
    """M3 rule: a trusted yes reads question + labels only, and the refusal side
    reads the kept question the same way, so a command-only path keeps its base
    rank (LOW here): neither approved by the yes nor refused."""
    _answered_ask(
        deep, "a-c0d2", "Free the disk space?", CMD_OPTIONS, answer="Yes, run it", command=CMD_ONLY
    )
    [ask] = prov.unanswered_asks(S)
    assert ask.answer is not None
    assert "bench" not in ask.text and "bench" not in ask.answer.text
    assert _rank(deep, CMD_ONLY).risk is Risk.LOW
    judge = _Judge("BLOCK")
    outcome = await _check(deep, CMD_ONLY, judge)
    assert outcome is None or not outcome.blocked, outcome and outcome.text
    assert judge.prompts == []


AB_OPTIONS = [
    {"label": "Delete ~/hyperplane-cpp/build", "description": "build output"},
    {"label": "Delete ~/bench/rigs", "description": "benchmark rigs"},
]


@pytest.mark.asyncio
async def test_an_ask_citation_never_verifies_a_path_the_operator_did_not_pick(
    deep: Path,
) -> None:
    """Review R2-m6: A picked, B not. A judge citing the ask for B must not pass
    the verifier: the citation covers only what the chosen answer covers."""
    _answered_ask(
        deep, "a-ab12", "Which tree should I delete?", AB_OPTIONS, answer=AB_OPTIONS[0]["label"]
    )
    judge = _Judge("ALLOW ask=a-ab12")
    outcome = await _check(deep, "rm -rf ~/bench/rigs", judge)
    assert outcome is not None and outcome.blocked, "B was never picked"
    assert len(judge.prompts) == 1
    # A, the picked one, is still the operator's decision.
    outcome = await _check(deep, "rm -rf ~/hyperplane-cpp/build", _Judge("BLOCK"))
    assert outcome is None or not outcome.blocked, outcome and outcome.text


@pytest.mark.asyncio
async def test_above_an_approved_glob_the_judge_reads_and_the_citation_never_verifies(
    deep: Path,
) -> None:
    """R2-M3 neighbour: a trusted yes no longer refuses outright, so deleting the
    dir ABOVE the approved ``~/osworld/runs*`` reaches the judge; the ask is no
    citation for it, so a judge fooled into ``ALLOW ask=`` is still blocked."""
    _deep_ask(deep, answer=DEEP_LABEL, expired=False)
    judge = _Judge(f"ALLOW ask={DEEP_ASK}")
    outcome = await _check(deep, "rm -rf ~/osworld", judge)
    assert outcome is not None and outcome.blocked, "the operator approved runs*, not ~/osworld"
    assert len(judge.prompts) == 1
    assert (deep / "osworld").is_dir()


# -- Incident a-48e3 (2026-11-16): an unproven answer surface ----------------------
#
# Session 439818272d84 asked to move a stray ``~/package.json``, lockfile and
# ``~/node_modules`` into a backup folder. The operator answered on his desktop
# card, but that runtime had been started by the wake supervisor, so the desktop
# backend held no operator capability for it and the runtime stamped the answer
# ``unverified`` (``asks.surfaces``). The guard refused the ``mv`` with "answered
# on the 'unverified' surface, not by the operator ... have no answer from them
# approving it", and the agent told the operator he had not answered. The refusal
# is correct (a proof-less local dial is what the model's own ``bash`` can forge);
# the wording was not. Rows 513/514 of that store, verbatim except the session
# (synthetic) and times (re-based on now, so the 7-day escalation window and the
# 24 h expiry hold whenever this runs).

A48E3_QUEUED = {
    "ask_id": "a-48e3",
    "at": 1794857239837,
    "created_at": 1794857239837,
    "expires_at": 1794943639837,
    "kind": "queued",
    "questions": [
        {
            "human_required": False,
            "id": "stray-home-pkg",
            "multi": False,
            "options": [
                {
                    "description": "Moves ~/package.json, ~/pnpm-lock.yaml and ~/node_modules "
                    "there unchanged and nothing is deleted. Moving them back restores the "
                    "current state exactly. pnpm works normally again in every repo.",
                    "label": "Move them to ~/.stray-node-home-20261116/",
                },
                {
                    "description": "Sessions keep working around it with "
                    "npm_config_manage_package_manager_versions=false. Any session that misses "
                    "that setting hits the pnpm loop.",
                    "label": "Leave them",
                },
            ],
            "persist": False,
            "question": "Your home folder has a stray copy of the mobile app's package.json "
            "(written Nov 10), plus a pnpm-lock.yaml and a node_modules folder with 421 "
            "packages. Its pnpm version pin makes pnpm in every folder under your home loop "
            "until the memory guard kills it, which is what blocked the marketing PR. Can I "
            "move these three into a dated backup folder?",
            "question_id": "",
            "recommended": 0,
            "secret": False,
        }
    ],
    "timeout_s": 86400,
    "tool_call_id": "toolu_01CropKk45Vg5fuZo9ADHvYV",
    "urgent": False,
    "v": 1,
}
A48E3_ANSWERED = {
    "answers": {"stray-home-pkg": ["Move them to ~/.stray-node-home-20261116/"]},
    "ask_id": "a-48e3",
    "at": 1794858354225,
    "by": {"surface": "unverified"},
    "kind": "answered",
    "tool_call_id": "toolu_01CropKk45Vg5fuZo9ADHvYV",
    "v": 1,
}
A48E3_MV = (
    "mkdir -p ~/.stray-node-home-20261116 && "
    "mv ~/package.json ~/pnpm-lock.yaml ~/node_modules ~/.stray-node-home-20261116/"
)


def _a48e3(home: Path, surface: str) -> None:
    """The incident's two rows in ``S``'s store, answered on ``surface``."""
    shift = int(time.time() * 1000) - 60_000 - A48E3_ANSWERED["at"]
    queued = {
        **A48E3_QUEUED,
        **{k: A48E3_QUEUED[k] + shift for k in ("at", "created_at", "expires_at")},
    }
    answered = {**A48E3_ANSWERED, "at": A48E3_ANSWERED["at"] + shift, "by": {"surface": surface}}
    (home / "package.json").write_text('{"packageManager": "pnpm@9.0.0"}\n')
    (home / "pnpm-lock.yaml").write_text("lockfileVersion: '9.0'\n")
    (home / "node_modules" / "left-pad").mkdir(parents=True)
    path = home / ".local-operator" / "sessions" / S / "asks.jsonl"
    with path.open("a") as fh:
        for row in (queued, answered):
            fh.write(json.dumps(row) + "\n")


@pytest.mark.asyncio
async def test_a48e3_unverified_answer_is_refused_and_says_it_was_unproven(home: Path) -> None:
    _a48e3(home, "unverified")
    [ask] = prov.unanswered_asks(S)
    assert ask.label == prov.UNPROVEN_SURFACE_LABEL and ask.answer is None
    judge = _Judge("ALLOW ask=a-48e3")  # would be fooled if asked
    outcome = await _check(home, A48E3_MV, judge)
    assert outcome is not None and outcome.blocked
    assert judge.prompts == []  # still the deterministic refusal, never the judge
    text = " ".join(outcome.text.split())
    assert "could not prove it is the operator's" in text
    assert "did not start this session's runtime" in text
    # The recovery names the routes that DO prove the operator (UI PR #2073
    # UX U1 / QA Q2): a fresh ask confirmed on the desktop's native sheet is
    # attested, so the old "background runtime has none" / "same desktop card
    # is refused the same way" claims are gone. Re-ask stays capped at once
    # (R1-M1/Q1) and the exact command stays the fallback.
    recovery = text.split("leave it in place.", 1)[1]
    assert recovery.startswith(" To proceed, ask again at most once, naming every exact full path")
    assert 'answer with "Confirm and send…" in the desktop app' in text
    assert "or in the terminal or TUI that started this runtime" in text
    assert "A plain Send answer, an answer revised after a plain Send" in text
    # The command fallback comes last, after the re-ask route.
    assert recovery.index("ask again at most once") < recovery.index(
        "Otherwise hand the exact command to the operator to run themselves"
    )
    assert "background has none" not in text and "has none" not in text
    assert "refused the same way" not in text
    assert "peer, bridge or relayed message" in text
    # Never steer the proof source: no "restart the runtime" / "start it from".
    lowered = text.lower()
    assert "restart" not in lowered and "relaunch" not in lowered
    assert "start the runtime" not in lowered and "start this runtime" not in lowered
    assert "so the operator answers from" not in text
    # Never again "no answer": the operator DID answer.
    assert "have no answer" not in text and "got no answer" not in text
    assert (home / "node_modules").is_dir() and (home / "package.json").is_file()


@pytest.mark.asyncio
@pytest.mark.parametrize("surface", ["remote", "chat", "relay", "bridge"])
async def test_a48e3_other_untrusted_surfaces_keep_their_refusal(home: Path, surface: str) -> None:
    """Negative controls: a phone/mesh (``remote``), the agent's own
    ``answered_in_chat`` (``chat``) and a relayed answer stay refused with no
    judge, and keep the not-the-operator wording (they are not "unproven
    operator" answers)."""
    _a48e3(home, surface)
    judge = _Judge("ALLOW ask=a-48e3")
    outcome = await _check(home, A48E3_MV, judge)
    assert outcome is not None and outcome.blocked and judge.prompts == []
    assert f"answered on the '{surface}' surface, not by the operator" in outcome.text
    assert prov.UNPROVEN_SURFACE_LABEL not in outcome.text


@pytest.mark.asyncio
async def test_a48e3_even_a_proven_answer_does_not_cover_a_description_only_path(
    home: Path,
) -> None:
    """The ask named ``~/node_modules`` only in an option description (the question
    says "a node_modules folder"), so a proven desktop answer still cannot approve
    moving it: that is the full-path rule, not the surface. A fooled judge's
    citation is refused by the verifier."""
    _a48e3(home, "desktop")
    judge = _Judge("ALLOW ask=a-48e3")
    outcome = await _check(home, A48E3_MV, judge)
    assert outcome is not None and outcome.blocked, "node_modules was never named in full"
    assert "node_modules" in outcome.text
    assert (home / "node_modules").is_dir()


def test_ask_docs_say_name_every_full_path_and_descriptions_approve_nothing() -> None:
    from local_operator.tools.tool_docs import TOOL_NOTES

    notes = " ".join(TOOL_NOTES["ask"].notes.split())
    assert "name every exact full path" in notes
    assert "a description approves nothing" in notes
    assert "relayed by a peer or bridge message never authorises" in notes
    # Kept in sync with ``lapsed_refusal_reason`` (UI PR #2073 UX U1): name the
    # attested desktop door, cap the re-ask at once, fall back to the command,
    # never steer who starts the runtime.
    assert '"Confirm and send…" in the desktop app' in notes
    assert "terminal or TUI that started this runtime" in notes
    assert "A plain Send answer, an answer revised after a plain Send" in notes
    assert notes.index("ask again at most once") < notes.index(
        "hand the operator the exact command"
    )
    assert "background-started runtime has none" not in notes
    assert "restart" not in notes.lower()


# -- 10. newest answered ask wins (field 2026-11-25: a-e630 vetoed a-ba0f) --------


def _answered_at(
    home: Path,
    ask_id: str,
    question: str,
    labels: list[str],
    *,
    answer: str,
    ago_ms: int,
    queued_ago_ms: int | None = None,
    session: str = S,
    surface: str = "desktop",
) -> None:
    """One synthetic answer; creation order need not match answer order."""
    wall = int(time.time() * 1000)
    now = wall - ago_ms
    created = wall - queued_ago_ms if queued_ago_ms is not None else now - 60_000
    rows = [
        {
            "v": 1,
            "kind": "queued",
            "ask_id": ask_id,
            "at": created,
            "created_at": created,
            "expires_at": now + 3_600_000,
            "timeout_s": 3600,
            "questions": [
                {
                    "id": "q",
                    "question": question,
                    "options": [{"label": label} for label in labels],
                    "multi": False,
                    "question_id": "",
                }
            ],
        },
        {
            "v": 1,
            "kind": "answered",
            "ask_id": ask_id,
            "at": now,
            "by": {"surface": surface},
            "answers": {"q": [answer]},
        },
    ]
    path = home / ".local-operator" / "sessions" / session / "asks.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def _block(home: Path, rel: str) -> Any:
    from local_operator.guardrails.check import _ask_block

    return _ask_block(
        prov.unanswered_asks(S),
        str(home / rel),
        home=str(home),
        cwd=str(home),
        stop=(),
        escalating=True,
        session_id=S,
        env=_env(home),
    )


BUILD_Q = "Delete ~/hyperplane-cpp/build?"


@pytest.mark.asyncio
async def test_newest_exact_approval_beats_older_decline_and_narrower_pick(deep: Path) -> None:
    """Three answered asks name one path; the newest (exact yes) decides.

    Store order puts the older decline first, which is what the field refusal
    cited. Neither it nor the older narrower pick may veto the newer answer.
    """
    _answered_at(deep, "a-e630", BUILD_Q, ["Approve", "Keep"], answer="Keep", ago_ms=600_000)
    _answered_at(
        deep,
        "a-narrow",
        "Delete ~/hyperplane-cpp/build or only ~/hyperplane-cpp/build/obj?",
        ["Delete only ~/hyperplane-cpp/build/obj", "Keep"],
        answer="Delete only ~/hyperplane-cpp/build/obj",
        ago_ms=300_000,
    )
    _answered_at(deep, "a-ba0f", BUILD_Q, ["Approve", "Keep"], answer="Approve", ago_ms=10_000)
    assert _block(deep, "hyperplane-cpp/build") is None
    judge = _Judge("BLOCK")
    result = await _check(deep, "rm -rf ~/hyperplane-cpp/build", judge)
    assert result is None or not result.blocked, result and result.text
    assert not judge.prompts


@pytest.mark.parametrize("suffix", [".", ":", ".:"], ids=["dot", "colon", "mixed"])
def test_compatibility_target_does_not_equate_punctuated_basenames(
    tmp_path: Path, suffix: str
) -> None:
    from local_operator.guardrails.check import _literal_compatibility_target

    plain = tmp_path / "example"
    punctuated = tmp_path / f"example{suffix}"
    assert plain != punctuated
    assert _literal_compatibility_target("~/example", str(tmp_path), None) == str(plain.resolve())
    assert _literal_compatibility_target(str(punctuated), str(tmp_path), None) is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "exact, older_decline", [(False, False), (False, True), (True, True), (True, False)]
)
@pytest.mark.parametrize("explicit", [False, True], ids=["generic", "explicit"])
async def test_compatible_broader_answer_is_neutral(
    deep: Path, exact: bool, older_decline: bool, explicit: bool
) -> None:
    from dataclasses import replace

    from local_operator.guardrails.check import build_delete_case

    if older_decline:
        _answered_at(deep, "a-c1", BUILD_Q, ["Approve", "Keep"], answer="Keep", ago_ms=600_000)
    if exact:
        _answered_at(deep, "a-e1", BUILD_Q, ["Approve", "Keep"], answer="Approve", ago_ms=200_000)
    # Tilde and absolute spellings name the same resolved parent.
    labels = (
        ["Delete ~/hyperplane-cpp", f"Keep {deep / 'hyperplane-cpp'}"]
        if explicit
        else ["Approve", "Keep"]
    )
    _answered_at(deep, "a-b1", "Delete ~/hyperplane-cpp?", labels, answer=labels[0], ago_ms=10_000)
    command = "rm -rf ~/hyperplane-cpp/build"
    ctx = _ctx(deep, command)
    preview = preview_tool_call("bash", {"command": command}, ctx, settings=GuardSettings())
    hit = _block(deep, "hyperplane-cpp/build")
    judge = _Judge("ALLOW ask=a-b1")
    result = await _check(deep, command, judge)
    # Exercise citation validation even when normal dispatch is unnecessary
    # (exact authority) or ineligible (the older decline still holds).
    verdict = _rank(deep, command)
    high = replace(verdict, risk=Risk.HIGH, targets=[replace(verdict.targets[0], risk=Risk.HIGH)])
    case = build_delete_case(high, ctx)
    basis, missing = case.verify("ALLOW ask=a-b1", forked=True)
    assert not basis and missing
    if exact:
        basis, missing = case.verify("ALLOW ask=a-e1", forked=True)
        assert (
            preview.builtin_effect,
            hit[0].ask_id if hit else None,
            bool(result and result.blocked),
            len(judge.prompts),
            basis,
            missing,
        ) == (
            "pass",
            None,
            False,
            0,
            f"{(deep / 'hyperplane-cpp/build').resolve()} named by ask a-e1",
            "",
        )
    elif older_decline:
        assert hit is not None and hit[0].ask_id == "a-c1"
        assert preview.builtin_effect == "refuse-lapsed-ask"
        assert result is not None and result.blocked
    else:
        # Without a decline the child keeps its base LOW rank, not an ask waiver.
        assert hit is None and preview.builtin_effect == "pass"
        assert not (result and result.blocked)
    assert not judge.prompts
    assert (deep / "hyperplane-cpp/build").is_dir()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "keep, pick_keep, surface, prior_exact, effect",
    [
        pytest.param("Keep ~/other/x", False, "desktop", True, "pass", id="distinct-disjoint"),
        pytest.param("Keep ~/hyperplane-cpp/build", False, "desktop", True, "judge", id="leaf"),
        pytest.param(
            "Keep ~/hyperplane-cpp/build/out", False, "desktop", True, "judge", id="nested"
        ),
        pytest.param("Keep ~/hyperplane-cpp", True, "desktop", True, "judge", id="selected-keep"),
        pytest.param(
            "Keep ~/hyperplane-cpp", False, "remote", False, "refuse-lapsed-ask", id="remote-only"
        ),
    ],
)
async def test_explicit_parent_alternative_respects_target_and_trust(
    deep: Path, keep: str, pick_keep: bool, surface: str, prior_exact: bool, effect: str
) -> None:
    from dataclasses import replace

    from local_operator.guardrails.check import build_delete_case

    _answered_at(
        deep,
        "a-e1",
        BUILD_Q,
        ["Approve", "Keep"],
        answer="Approve" if prior_exact else "Keep",
        ago_ms=200_000,
    )
    delete = "Delete ~/hyperplane-cpp"
    _answered_at(
        deep,
        "a-b1",
        f"{delete}?",
        [delete, keep],
        answer=keep if pick_keep else delete,
        ago_ms=10_000,
        surface=surface,
    )
    command = "rm -rf ~/hyperplane-cpp/build"
    ctx = _ctx(deep, command)
    preview = preview_tool_call("bash", {"command": command}, ctx, settings=GuardSettings())
    judge = _Judge("ALLOW ask=a-e1")
    result = await _check(deep, command, judge)
    verdict = _rank(deep, command)
    high = replace(verdict, risk=Risk.HIGH, targets=[replace(verdict.targets[0], risk=Risk.HIGH)])
    case = build_delete_case(high, ctx)
    exact, exact_missing = case.verify("ALLOW ask=a-e1", forked=True)
    broader, broader_missing = case.verify("ALLOW ask=a-b1", forked=True)
    assert (preview.builtin_effect, bool(result and result.blocked), len(judge.prompts)) == (
        effect,
        effect != "pass",
        int(effect == "judge"),
    )
    assert (bool(exact), bool(exact_missing)) == (effect == "pass", effect != "pass")
    assert not broader and broader_missing
    assert (deep / "hyperplane-cpp/build").is_dir()  # No command is dispatched.


def _fs_fingerprint(root: Path) -> dict[str, tuple[int, int, bytes | None]]:
    """(st_dev, st_ino, bytes) for every entry under ``root``; dirs carry None."""
    out: dict[str, tuple[int, int, bytes | None]] = {}
    for entry in [root, *sorted(root.rglob("*"))]:
        st = entry.lstat()
        out[str(entry)] = (st.st_dev, st.st_ino, None if entry.is_dir() else entry.read_bytes())
    return out


@pytest.mark.asyncio
@pytest.mark.parametrize("suffix", [".", ":", ".:"], ids=["dot", "colon", "mixed"])
@pytest.mark.parametrize(
    "delete, keep",
    [
        # The PICKED label carries the terminal punctuation; the unpicked one is plain.
        pytest.param("Delete ~/hyperplane-cpp{s}", "Keep ~/hyperplane-cpp", id="picked-punct"),
        # The UNPICKED label carries it; the picked one is plain. The comparator
        # runs once per label, so each label is an independent call site.
        pytest.param("Delete ~/hyperplane-cpp", "Keep ~/hyperplane-cpp{s}", id="unpicked-punct"),
    ],
)
async def test_punctuated_parent_label_keeps_competing_option_veto(
    deep: Path, delete: str, keep: str, suffix: str
) -> None:
    """Q-R5-1 (PR #4599): a label ending in ``.``/``:`` is a different literal path.

    Prose expansion trims terminal dots/colons, so ``~/hyperplane-cpp.`` reads
    as ``~/hyperplane-cpp``. ``_literal_compatibility_target`` must NOT equate
    them, otherwise a newer broader pick would look compatible with the older
    exact approval, silently suppress the competing-option veto, and let the
    superseded exact citation authorise ``rm -rf``. With the veto kept the
    preview defers to the judge and the checkpoint rejects that citation.
    Sentinels prove neither decision entry point touched the filesystem.
    """
    from dataclasses import replace

    from local_operator.guardrails.check import build_delete_case

    delete, keep = delete.format(s=suffix), keep.format(s=suffix)
    child = deep / "hyperplane-cpp/build"
    sibling = deep / "hyperplane-cpp/sibling-keep"
    for base in (child, sibling):
        (base / "sentinel-dir").mkdir(parents=True)
        (base / "sentinel-dir" / "sentinel.txt").write_bytes(b"keep-me:" + base.name.encode())
        (base / "sentinel.txt").write_bytes(b"top:" + base.name.encode())
    before = {base: _fs_fingerprint(base) for base in (child, sibling)}

    _answered_at(deep, "a-e1", BUILD_Q, ["Approve", "Keep"], answer="Approve", ago_ms=200_000)
    # The PICKED answer is always the Delete label; only its spelling varies.
    _answered_at(
        deep, "a-b1", "Delete ~/hyperplane-cpp?", [delete, keep], answer=delete, ago_ms=10_000
    )
    command = "rm -rf ~/hyperplane-cpp/build"
    ctx = _ctx(deep, command)
    preview = preview_tool_call("bash", {"command": command}, ctx, settings=GuardSettings())
    judge = _Judge("ALLOW ask=a-e1")
    result = await _check(deep, command, judge)
    verdict = _rank(deep, command)
    high = replace(verdict, risk=Risk.HIGH, targets=[replace(verdict.targets[0], risk=Risk.HIGH)])
    case = build_delete_case(high, ctx)
    exact, exact_missing = case.verify("ALLOW ask=a-e1", forked=True)
    broader, broader_missing = case.verify("ALLOW ask=a-b1", forked=True)

    assert preview.builtin_effect == "judge"
    assert len(judge.prompts) == 1  # exactly one fake-judge call, from the checkpoint
    assert result is not None and result.blocked  # superseded exact citation rejected
    assert not exact and exact_missing
    assert not broader and broader_missing  # the broader pick grants no authority
    assert {base: _fs_fingerprint(base) for base in (child, sibling)} == before
    assert child.is_dir() and sibling.is_dir()  # No command is dispatched.


@pytest.mark.asyncio
async def test_exact_trusted_answer_does_not_change_critical_precedence(deep: Path) -> None:
    _answered_at(
        deep, "a-e1", f"Delete {deep}?", ["Approve", "Keep"], answer="Approve", ago_ms=10_000
    )
    command = "rm -rf ~"
    preview = preview_tool_call(
        "bash", {"command": command}, _ctx(deep, command), settings=GuardSettings()
    )
    judge = _Judge("ALLOW ask=a-e1")
    result = await _check(deep, command, judge)
    assert preview.builtin_effect == "refuse-critical"
    assert result is not None and result.blocked and not judge.prompts
    assert deep.is_dir()


@pytest.mark.asyncio
async def test_newest_answer_naming_another_path_does_not_approve_this_one(deep: Path) -> None:
    """Supersession is per path: a newer yes to ~/other/x leaves the decline."""
    _answered_at(deep, "a-e630", BUILD_Q, ["Approve", "Keep"], answer="Keep", ago_ms=600_000)
    _answered_at(
        deep, "a-ba0f", "Delete ~/other/x?", ["Approve", "Keep"], answer="Approve", ago_ms=10_000
    )
    hit = _block(deep, "hyperplane-cpp/build")
    assert hit is not None and hit[0].ask_id == "a-e630"
    judge = _Judge("ALLOW ask=a-ba0f")
    result = await _check(deep, "rm -rf ~/hyperplane-cpp/build", judge)
    assert result is not None and result.blocked
    assert (deep / "hyperplane-cpp/build").is_dir()


@pytest.mark.asyncio
async def test_decline_answered_after_the_approval_still_holds(deep: Path) -> None:
    """Control: recency cuts both ways; the operator's later "Keep" wins."""
    _answered_at(deep, "a-ba0f", BUILD_Q, ["Approve", "Keep"], answer="Approve", ago_ms=600_000)
    _answered_at(deep, "a-e630", BUILD_Q, ["Approve", "Keep"], answer="Keep", ago_ms=10_000)
    hit = _block(deep, "hyperplane-cpp/build")
    assert hit is not None and hit[0].ask_id == "a-e630"
    result = await _check(deep, "rm -rf ~/hyperplane-cpp/build", _Judge("ALLOW ask=a-ba0f"))
    assert result is not None and result.blocked


@pytest.mark.asyncio
@pytest.mark.parametrize("direct_turn", [False, True])
async def test_latest_decline_wins_across_lineage_and_rejects_stale_citation(
    deep: Path, direct_turn: bool
) -> None:
    state = deep / ".local-operator"
    prov.record_lineage(CHILD, origin="subagent", parent=S, state_root=state)
    prov.record_lineage(S, origin="subagent", parent=PARENT, state_root=state)
    _answered_at(
        deep, "a-c1", BUILD_Q, ["Approve", "Keep"], answer="Keep", ago_ms=500_000, session=CHILD
    )
    _answered_at(
        deep, "a-a1", BUILD_Q, ["Approve", "Keep"], answer="Approve", ago_ms=200_000, session=PARENT
    )
    _answered_at(
        deep,
        "a-d1",
        BUILD_Q,
        ["Approve", "Keep"],
        answer="Keep",
        ago_ms=10_000,
        queued_ago_ms=900_000,
        session=PARENT,
    )
    command = "rm -rf ~/hyperplane-cpp/build"
    ctx = _ctx(deep, command)
    if direct_turn:
        # A direct turn can keep the judge eligible; it does not make a
        # losing ask citation a valid basis for the judge's ALLOW.
        assert ctx.operator is not None
        ctx.operator.words.append(
            prov.OperatorWords("operator, direct", "Review cleanup of ~/hyperplane-cpp/build")
        )
    else:
        preview = preview_tool_call("bash", {"command": command}, ctx, settings=GuardSettings())
        assert preview.builtin_effect == "refuse-lapsed-ask" and "a-d1" in preview.render()
    judge = _Judge("ALLOW ask=a-a1")
    result = await check_tool_call(
        "bash", {"command": command}, ctx, judge=judge, settings=GuardSettings(), forked_judge=judge
    )
    assert result is not None and result.blocked
    assert len(judge.prompts) == int(direct_turn)
    assert "a-d1" in (judge.prompts[0] if direct_turn else result.text)


def test_later_preservation_yes_is_not_a_compatible_broader_approval(deep: Path) -> None:
    from dataclasses import replace

    from local_operator.guardrails.check import build_delete_case

    _answered_at(deep, "a-e1", BUILD_Q, ["Approve", "Keep"], answer="Approve", ago_ms=200_000)
    _answered_at(
        deep,
        "a-f1",
        "Keep ~/hyperplane-cpp?",
        ["Yes", "No"],
        answer="Yes",
        ago_ms=10_000,
    )
    command = "rm -rf ~/hyperplane-cpp/build"
    hit = _block(deep, "hyperplane-cpp/build")
    assert hit is not None and hit[0].ask_id == "a-f1"
    verdict = _rank(deep, command)
    high = replace(verdict, risk=Risk.HIGH, targets=[replace(verdict.targets[0], risk=Risk.HIGH)])
    case = build_delete_case(high, _ctx(deep, command))
    basis, missing = case.verify("ALLOW ask=a-e1", forked=True)
    assert not basis and missing


def test_citation_rechecks_for_a_decline_recorded_after_judge_dispatch(deep: Path) -> None:
    from dataclasses import replace

    from local_operator.guardrails.check import build_delete_case

    _answered_at(deep, "a-a1", BUILD_Q, ["Approve", "Keep"], answer="Approve", ago_ms=200_000)
    command = "rm -rf ~/hyperplane-cpp/build"
    verdict = _rank(deep, command)
    # Test the citation boundary directly, independent of why the classifier
    # would need a judge on this target. No shell command is dispatched.
    high = replace(verdict, risk=Risk.HIGH, targets=[replace(verdict.targets[0], risk=Risk.HIGH)])
    case = build_delete_case(high, _ctx(deep, command))
    basis, missing = case.verify("ALLOW ask=a-a1", forked=True)
    assert "a-a1" in basis and not missing
    _answered_at(deep, "a-d1", BUILD_Q, ["Approve", "Keep"], answer="Keep", ago_ms=10_000)
    basis, missing = case.verify("ALLOW ask=a-a1", forked=True)
    assert not basis and missing


def test_expired_late_decline_newer_than_approval_still_holds(deep: Path) -> None:
    """An expired ask carrying a late "Keep" answered AFTER the approval wins."""
    _answered_at(deep, "a-ba0f", BUILD_Q, ["Approve", "Keep"], answer="Approve", ago_ms=600_000)
    _deep_ask(deep, answer="Keep everything", expired=True)
    hit = _block(deep, "hyperplane-cpp/build")
    assert hit is not None and hit[0].ask_id == DEEP_ASK
