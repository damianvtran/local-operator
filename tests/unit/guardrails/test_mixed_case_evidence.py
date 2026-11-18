"""Mixed-case HIGH writes: listed resolver-pinned evidence verifies without a citation.

Node2 incident (#4143 QA): a parent ran, inside its descendant's scratchpad,

    set -o noclobber; bash <pad>/qa/unit.sh <packet> <packet>/units \
        > <packet>/logs/t1-pytest8-launch-2.log 2>&1

The directories handed to ``bash`` are ``unverifiable`` (``build_delete_case``
puts every ``op == "touch"`` target there), so the judge-free
``evidence_basis`` route is off and the CONTEXT-FREE judge decides. Its plain
ALLOW covered the directories, but the log still needed a cited
``evidence=eN`` — and whether that judge echoed ``e1`` varied (log1 allowed,
the identical log2 refused: "approval could not be verified").

``DeleteCase.verify`` now also accepts a target that a LISTED evidence call
re-verifies with resolver-pinned coverage (B1 / clean clone — the same bar
``evidence_basis`` uses for a judge-free allow). B2 (name + birth time) still
needs the citation, a judge BLOCK still blocks, and the HIGH ranking itself is
unchanged. Everything is synthetic: an isolated HOME/state root, made-up ids
and a stub judge.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import pytest

from local_operator.guardrails import evidence as ev_mod
from local_operator.guardrails import provenance as prov
from local_operator.guardrails.check import CallContext, GuardSettings, check_tool_call

PARENT = "aaaaaaaaaaaa"
CHILD = "bbbbbbbbbbbb"
SIBLING = "dddddddddddd"
STRANGER = "cccccccccccc"


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "home"
    state = home / ".local-operator"
    for sid in (PARENT, CHILD, SIBLING, STRANGER):
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


def _child_of(home: Path, child: str, parent: str) -> None:
    prov.record_lineage(
        child, origin="subagent", parent=parent, state_root=home / ".local-operator"
    )


def _packet(home: Path, owner: str, command: str | None) -> Path:
    """Make ``<owner pad>/qa/packet/{units,logs}`` and record ``owner``'s creating call.

    ``command`` is the bash call the owner's transcript holds, its window
    bracketing the directories' birth; ``None`` records no call at all.
    """
    pad = _pad(home, owner)
    qa = pad / "qa"
    packet = qa / "packet"
    started = time.time() - 1.0
    (packet / "units").mkdir(parents=True)
    (packet / "logs").mkdir()
    (qa / "unit.sh").write_text("#!/bin/sh\necho ok\n")
    finished = time.time() + 1.0
    if command is not None:
        rows = [
            {
                "type": "message",
                "ts": started,
                "payload": {
                    "role": "assistant",
                    "tool_calls": [
                        {"id": "call-mk", "name": "bash", "arguments": {"command": command}}
                    ],
                },
            },
            {
                "type": "message",
                "ts": finished,
                "payload": {"role": "tool", "tool_call_id": "call-mk", "content": "exit 0"},
            },
        ]
        transcript = pad.parent / "transcript.jsonl"
        transcript.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return packet


def _incident(packet: Path) -> str:
    # The node2 shape: directories handed to an unknown program (unverifiable)
    # plus a redirect into a directory the descendant made.
    script = packet.parent / "unit.sh"
    return (
        f"set -o noclobber; bash {script} {packet} {packet}/units "
        f"> {packet}/logs/t1-pytest8-launch-2.log 2>&1"
    )


class _Judge:
    """A stub judge answering a fixed first line; records every prompt."""

    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.prompts: list[str] = []

    async def __call__(self, system: str, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.answer


async def _check(home: Path, command: str, sid: str, judge: _Judge) -> Any:
    ctx = CallContext(
        cwd=str(home / "work"),
        env={"HOME": str(home), "LOCAL_OPERATOR_SCRATCHPAD": str(_pad(home, sid))},
        scratchpad=str(_pad(home, sid)),
        session_id=sid,
        operator=prov.operator_brief([], session_id=sid, cited=(command,)),
        messages=[],
    )
    return await check_tool_call(
        "bash",
        {"command": command},
        ctx,
        judge=judge,
        settings=GuardSettings(judge_timeout_s=5.0),
        forked_judge=judge,
    )


MKDIR = "mkdir -p {packet}/units {packet}/logs"


@pytest.mark.asyncio
async def test_incident_shape_plain_allow_runs_on_listed_b1_evidence(home: Path) -> None:
    """(1) The node2 refusal: listed B1 evidence + plain ALLOW now runs."""
    _child_of(home, CHILD, PARENT)
    packet = _packet(home, CHILD, MKDIR.format(packet=_pad(home, CHILD) / "qa" / "packet"))
    judge = _Judge("ALLOW")
    outcome = await _check(home, _incident(packet), PARENT, judge)
    assert outcome is not None
    assert len(judge.prompts) == 1  # still judged: the ranking is unchanged
    assert f"subagent {CHILD} call call-mk" in judge.prompts[0]
    assert not outcome.blocked, outcome.text
    allowed = " ".join(outcome.warnings)
    assert "t1-pytest8-launch-2.log created by subagent " + CHILD in allowed


@pytest.mark.asyncio
async def test_incident_shape_judge_block_still_blocks(home: Path) -> None:
    """(2) Coverage never overrides the judge's BLOCK."""
    _child_of(home, CHILD, PARENT)
    packet = _packet(home, CHILD, MKDIR.format(packet=_pad(home, CHILD) / "qa" / "packet"))
    judge = _Judge("BLOCK: not asked for")
    outcome = await _check(home, _incident(packet), PARENT, judge)
    assert outcome is not None and outcome.blocked
    assert len(judge.prompts) == 1


@pytest.mark.asyncio
async def test_b2_only_evidence_still_needs_the_citation(home: Path) -> None:
    """(3) Name + birth time is not resolver-pinned: plain ALLOW blocks, a citation runs."""
    _child_of(home, CHILD, PARENT)
    # Names ``logs`` but never re-resolves to creating it: B2 only.
    packet = _packet(home, CHILD, "node gen.mjs logs")
    outcome = await _check(home, _incident(packet), PARENT, _Judge("ALLOW"))
    assert outcome is not None and outcome.blocked
    assert "could not be verified" in outcome.text

    ev_mod._TREE_CACHE.clear()
    judge = _Judge("ALLOW evidence=e1")
    outcome = await _check(home, _incident(packet), PARENT, judge)
    assert outcome is not None
    assert f"e1: subagent {CHILD} call call-mk" in judge.prompts[0]
    assert not outcome.blocked, outcome.text


@pytest.mark.asyncio
async def test_no_evidence_plain_allow_blocks(home: Path) -> None:
    """(4) Nothing listed: the pre-existing refusal, unchanged."""
    _child_of(home, CHILD, PARENT)
    packet = _packet(home, CHILD, None)
    outcome = await _check(home, _incident(packet), PARENT, _Judge("ALLOW"))
    assert outcome is not None and outcome.blocked
    assert "could not be verified" in outcome.text


@pytest.mark.asyncio
async def test_all_verifiable_targets_stay_judge_free(home: Path) -> None:
    """(5) Every target B1-covered and nothing unverifiable: no judge, as before."""
    _child_of(home, CHILD, PARENT)
    packet = _packet(home, CHILD, MKDIR.format(packet=_pad(home, CHILD) / "qa" / "packet"))
    judge = _Judge("BLOCK: should not be asked")
    outcome = await _check(home, f"rm -rf {packet}/logs", PARENT, judge)
    assert outcome is not None
    assert not outcome.blocked, outcome.text
    assert judge.prompts == []
    assert any(f"created by subagent {CHILD}" in w for w in outcome.warnings)


@pytest.mark.asyncio
async def test_pad_owned_outside_the_task_tree_has_no_evidence(home: Path) -> None:
    """(6) A stranger's mkdir is not this tree's evidence: plain ALLOW blocks."""
    _child_of(home, CHILD, PARENT)
    packet = _packet(home, STRANGER, MKDIR.format(packet=_pad(home, STRANGER) / "qa" / "packet"))
    judge = _Judge("ALLOW")
    outcome = await _check(home, _incident(packet), PARENT, judge)
    assert outcome is not None and outcome.blocked
    assert "could not be verified" in outcome.text
    assert f"subagent {STRANGER}" not in judge.prompts[0]


@pytest.mark.asyncio
async def test_peer_session_with_the_same_shape_blocks(home: Path) -> None:
    """(7) A sibling is a peer, not lineage: the child's call is not its evidence."""
    _child_of(home, CHILD, PARENT)
    _child_of(home, SIBLING, PARENT)
    packet = _packet(home, CHILD, MKDIR.format(packet=_pad(home, CHILD) / "qa" / "packet"))
    judge = _Judge("ALLOW")
    outcome = await _check(home, _incident(packet), SIBLING, judge)
    assert outcome is not None and outcome.blocked
    assert "could not be verified" in outcome.text
