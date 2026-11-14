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
   not create ranks HIGH and reaches the judge as a forced removal.
"""

from __future__ import annotations

import json
import os
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
