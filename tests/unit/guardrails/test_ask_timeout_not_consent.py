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
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
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
    monkeypatch.setattr(prov, "_live_cwd_inside", lambda _: False)
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
    assert "a timeout is never consent to delete" in builtin._ASK_DESCRIPTION_QUEUED
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
