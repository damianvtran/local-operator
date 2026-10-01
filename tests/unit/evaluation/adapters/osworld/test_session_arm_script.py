"""The session arm runs ONE whole episode from a real spawned worker.

``session_arm_rig.py`` (beside this file) drives ``scripts/run_episode.py
--engagement session`` with a scripted model over the real offline adapter --
the shipped wheel in a copied interpreter, a digest-pinned FakeProvider
workspace, the real action server over MCP, the real bridge. These tests spawn
that rig as a subprocess with an isolated scratch HOME/config and assert the
episode contract end to end, plus the Q-1 gate: a delegated child can neither
drive nor end the episode (its call is refused and the steps and terminal stay
the parent's).

Marked ``slow`` like the other real-spawn suites.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

import pytest

from tests.unit.evaluation.adapters.osworld import fixtures, spawn_helpers
from tests.unit.evaluation.adapters.osworld.test_build_and_scripts import (  # noqa: F401
    durable_path,
)

pytestmark = pytest.mark.slow

REPO = Path(__file__).resolve().parents[5]
RIG = Path(__file__).resolve().parent / "session_arm_rig.py"
CANARY_KEY = "canary-key-rig1687"
CANARY_SECRET = "canary-secret-rig1687-9f8e7d6c5b4a"


@pytest.fixture(scope="module")
def adapter_wheel(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return spawn_helpers.build_adapter_wheel(tmp_path_factory.mktemp("wheel"))


@pytest.fixture(scope="module")
def spawn_interpreter(
    tmp_path_factory: pytest.TempPathFactory, adapter_wheel: Path
) -> spawn_helpers.SpawnInterpreter:
    # The expensive half of the selector (a copied interpreter with the wheel
    # installed) is paid once per module; each case builds only its workspace.
    return spawn_helpers.build_spawn_interpreter(tmp_path_factory.mktemp("interp"), adapter_wheel)


def _run_rig(
    root: Path,
    wheel: Path,
    interpreter: spawn_helpers.SpawnInterpreter,
    *,
    child_acts: str | None = None,
    challenge_reply: str | None = None,
    ending: str = "finish",
) -> tuple[subprocess.CompletedProcess[str], Path, Path]:
    selector_dir = root / "adapter"
    selector = spawn_helpers.build_spawnable_adapter(
        selector_dir,
        wheel,
        {"task_plain": fixtures.PLAIN},
        provider={"provider": "fake", "scripted_score": 1.0},
        interpreter=interpreter,
    )
    selector_path = selector_dir / "selector.json"
    selector_path.write_text(selector.model_dump_json())

    scratch = root / "s"
    run_root = root / "run"
    log = root / "rig.jsonl"
    (scratch / "home" / ".local-operator").mkdir(parents=True, exist_ok=True)
    (scratch / "workspace").mkdir(parents=True, exist_ok=True)
    run_root.mkdir(parents=True, exist_ok=True)

    # The arm refuses an ambient home: the three names below ARE its scratch,
    # and the canary secrets travel only through the run's own pipe.
    env = {
        "PATH": os.environ.get("PATH", ""),
        "TERM": "xterm-256color",
        "PYTHONPATH": str(REPO),
        "HOME": str(scratch / "home"),
        "LOCAL_OPERATOR_CONFIG_DIR": str(scratch / "home" / ".local-operator"),
        "LOP_RUN_SCRATCH_ROOT": str(scratch),
        "AWS_ACCESS_KEY_ID": CANARY_KEY,
        "AWS_SECRET_ACCESS_KEY": CANARY_SECRET,
    }
    args = [
        sys.executable,
        str(RIG),
        "--worktree",
        str(REPO),
        "--selector",
        str(selector_path),
        "--run-root",
        str(run_root),
        "--log",
        str(log),
        "--episode-id",
        f"rig-{uuid.uuid4().hex[:10]}",
    ]
    if child_acts is not None:
        args += ["--child-acts", child_acts]
    if challenge_reply is not None:
        args += ["--challenge-reply", challenge_reply]
    args += ["--ending", ending]
    completed = subprocess.run(
        args, capture_output=True, text=True, env=env, cwd=str(REPO), check=False
    )
    return completed, run_root, log


def _outcome(
    completed: subprocess.CompletedProcess[str], expect_status: str = "completed"
) -> dict[str, Any]:
    # ``run_episode`` exits ``EXIT_EPISODE`` (1) for any non-``completed``
    # ending -- ``agent_stop`` included -- and still prints the outcome JSON,
    # so the status is read from the payload and the code is checked against
    # it (not against 0).
    expected_rc = 0 if expect_status == "completed" else 1
    assert completed.returncode == expected_rc, completed.stderr[-4000:]
    text = completed.stdout.strip()
    start = text.find("{")
    assert start != -1, text[-2000:]
    outcome: dict[str, Any] = json.loads(text[start:])
    assert outcome["status"] == expect_status, outcome
    return outcome


def _record_kinds(run_root: Path) -> list[str]:
    records = sorted(run_root.rglob("events.jsonl"))
    assert len(records) == 1, records
    return [json.loads(line)["kind"] for line in records[0].read_text().splitlines()]


def _record_payloads(run_root: Path, kind: str) -> list[dict[str, Any]]:
    """Every recorded row of one kind (``{"kind": ..., **payload}`` rows)."""

    records = sorted(run_root.rglob("events.jsonl"))
    assert len(records) == 1, records
    rows = [json.loads(line) for line in records[0].read_text().splitlines()]
    return [
        {key: value for key, value in row.items() if key != "kind"}
        for row in rows
        if row["kind"] == kind
    ]


def _log_events(log: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in log.read_text().splitlines()]


def test_a_whole_episode_runs_offline_and_scores(
    durable_path: Path,  # noqa: F811
    adapter_wheel: Path,
    spawn_interpreter: spawn_helpers.SpawnInterpreter,
) -> None:
    root = durable_path / f"s-{uuid.uuid4().hex[:8]}"
    completed, run_root, log = _run_rig(root, adapter_wheel, spawn_interpreter)

    outcome = _outcome(completed)
    assert outcome["steps"] == 1
    assert outcome["terminal_reason"] == "finish"
    assert outcome["score"]["status"] == "scored" and outcome["score"]["binary"] == 1
    assert outcome["rescue_required"] is False
    assert "mcp__episode_actions_apply_actions" in outcome["tool_names"]

    kinds = _record_kinds(run_root)
    for expected in (
        "reset",
        "declaration",
        "mcp_settle",
        "tools",
        "action_batch",
        "action_completion_challenged",
        "action_finish",
    ):
        assert expected in kinds, kinds
    assert kinds.count("action_batch") == 1
    # The completion gate round-tripped through the REAL bridge: the first
    # `done` claim was challenged once, then the script re-declared and THAT
    # declaration ended the episode -- and the record carries the claim the
    # gate reasoned about, status and reason both.
    assert kinds.count("action_completion_challenged") == 1
    assert kinds.count("action_finish") == 1
    challenged = _record_payloads(run_root, "action_completion_challenged")
    assert challenged[0]["reason"] == "session-arm rig: episode complete"
    assert "That declaration is a CLAIM" in challenged[0]["challenge"]
    assert "session-arm rig: episode complete" in challenged[0]["challenge"]
    finished = _record_payloads(run_root, "action_finish")
    assert finished[0]["status"] == "done"
    assert finished[0]["reason"] == "session-arm rig: episode complete, re-checked"
    assert finished[0]["completion_challenged"] is True

    # The wait batch's rendered frame came back through the MCP tool result:
    # the model's next request carries the tool row the call produced.
    events = _log_events(log)
    assert any("Frames: screen" in (event.get("last_tool_text") or "") for event in events), [
        event.get("last_tool_text") for event in events
    ]

    # ...and the challenge itself reached the model through the same wire: the
    # request after the first finish carried the challenge as its last tool
    # row, which is the whole mechanism -- a refusal the model can read.
    assert any(
        "That declaration is a CLAIM" in (event.get("last_tool_text") or "") for event in events
    ), [event.get("last_tool_text") for event in events]

    # The canary secrets never surface outside the worker's own pipe.
    assert CANARY_SECRET not in completed.stdout
    assert CANARY_SECRET not in completed.stderr
    assert CANARY_KEY not in completed.stdout


def test_a_challenged_claim_is_rescued_by_a_corrective_action(
    durable_path: Path,  # noqa: F811
    adapter_wheel: Path,
    spawn_interpreter: spawn_helpers.SpawnInterpreter,
) -> None:
    """task_013's shape, end to end on the offline rig: CHALLENGE -> ACTION -> finish.

    The real-task failure this gate exists for was a model that had completed
    the work and declared it finished without the submission -- the episode
    ended on the claim, the evaluator's state capture had no ``form_response``,
    and the run scored 0 where the same answers scored 1.0 through a channel
    that challenged the claim. This test drives the second half of that
    rescue: the challenged script does NOT re-declare -- it makes one more
    corrective action (the submit stand-in), and the NEXT finish ends the
    episode with the score intact.
    """

    root = durable_path / f"s-{uuid.uuid4().hex[:8]}"
    completed, run_root, log = _run_rig(
        root, adapter_wheel, spawn_interpreter, challenge_reply="act"
    )

    outcome = _outcome(completed)
    assert outcome["steps"] == 2  # the corrective action after the challenge counted
    assert outcome["terminal_reason"] == "finish"
    assert outcome["score"]["status"] == "scored" and outcome["score"]["binary"] == 1

    kinds = _record_kinds(run_root)
    assert kinds.count("action_batch") == 2
    assert kinds.count("action_completion_challenged") == 1
    assert kinds.count("action_finish") == 1
    finished = _record_payloads(run_root, "action_finish")
    assert finished[0]["completion_challenged"] is True


def test_a_prose_done_claim_is_challenged_and_the_re_declaration_ends_it(
    durable_path: Path,  # noqa: F811
    adapter_wheel: Path,
    spawn_interpreter: spawn_helpers.SpawnInterpreter,
) -> None:
    """Arm 1748's task_003, end to end: a terminal PROSE "Done" earns the challenge.

    THE BYPASS THIS PINS. The gate fires inside ``ActionBridge.call``, so it
    only ever saw tool-mediated claims; the first field run of arm 1748 ended
    its final answer as prose -- "Done. Summary of what I determined and
    did: ..." with NO tool call -- and the turn simply ended: ``agent_stop``,
    the gate never fired, an unverified finish by any reading. This case runs
    the identical offline episode BEFORE the fix (the same rig against the
    pre-fix worktree: ``agent_stop``, steps 1, ZERO challenges, two model
    calls) and AFTER it: exactly one challenge -- delivered to the model over
    the real session wire as a harness-injected user row -- and the scripted
    re-declaration ends the episode ``completed`` with the step count intact
    (the claim is a decision about the screen, not a step).
    """

    from tests.unit.evaluation.adapters.osworld import session_arm_rig

    root = durable_path / f"s-{uuid.uuid4().hex[:8]}"
    completed, run_root, log = _run_rig(
        root, adapter_wheel, spawn_interpreter, ending="prose-claim"
    )

    outcome = _outcome(completed)
    assert outcome["steps"] == 1
    assert outcome["terminal_reason"] == "finish"
    assert outcome["score"]["status"] == "scored" and outcome["score"]["binary"] == 1

    kinds = _record_kinds(run_root)
    assert kinds.count("action_completion_challenged") == 1
    assert kinds.count("action_finish") == 1
    challenged = _record_payloads(run_root, "action_completion_challenged")
    assert challenged[0]["trigger"] == "terminal-message"
    assert challenged[0]["reason"] == session_arm_rig.PROSE_CLAIM_TEXT
    assert "That declaration is a CLAIM" in challenged[0]["challenge"]
    assert session_arm_rig.PROSE_CLAIM_TEXT in challenged[0]["challenge"]
    finished = _record_payloads(run_root, "action_finish")
    assert finished[0]["completion_challenged"] is True

    # The challenge reached the MODEL, not just the record: the request after
    # the prose "Done" carried it as its last USER row (the reply channel's
    # re-prompt, on this channel), and the model answered it with the
    # re-declaration.
    events = _log_events(log)
    assert any(
        "That declaration is a CLAIM" in (event.get("last_user_text") or "") for event in events
    ), [event.get("last_user_text") for event in events]


def test_a_prose_done_that_is_not_re_declared_is_challenged_exactly_once(
    durable_path: Path,  # noqa: F811
    adapter_wheel: Path,
    spawn_interpreter: spawn_helpers.SpawnInterpreter,
) -> None:
    """The bound holds on the prose arm too: one challenge, then the claim stands.

    The script claims completion in prose, is challenged, and claims AGAIN in
    prose -- the second declaration must be the final word (no loop, no second
    exchange), exactly as a second finish call is accepted on the tool arm.
    """

    root = durable_path / f"s-{uuid.uuid4().hex[:8]}"
    completed, run_root, log = _run_rig(
        root, adapter_wheel, spawn_interpreter, ending="prose-claim", challenge_reply="prose"
    )

    outcome = _outcome(completed, expect_status="agent_stop")
    assert outcome["steps"] == 1
    # The ending NAMES itself: a prose claim that survived the one challenge.
    assert outcome["terminal_reason"] == "completion-claim"
    assert "challenged it 1 time(s)" in outcome["diagnostic"]
    kinds = _record_kinds(run_root)
    assert kinds.count("action_completion_challenged") == 1
    assert "action_finish" not in kinds
    # Three model calls: the opening wait, the prose claim, and the one answer
    # to the challenge. A fourth would mean the gate looped over the spent
    # budget; the pre-fix tree shows two (no challenge, no answer at all).
    events = _log_events(log)
    assert len([event for event in events if event.get("episode")]) == 3, events


def test_a_mid_work_narration_ending_is_not_challenged(
    durable_path: Path,  # noqa: F811
    adapter_wheel: Path,
    spawn_interpreter: spawn_helpers.SpawnInterpreter,
) -> None:
    """The discriminating negative: narration must NOT trip the prose arm.

    Arm 1716's 005 left exactly this text as its last one -- a mid-work report
    plus what was about to be done -- and its episode ended without the model
    ever claiming completion. The prose arm must leave that shape alone: the
    episode ends ``agent_stop`` with ZERO challenges and exactly the two model
    calls the un-gated script always made.
    """

    root = durable_path / f"s-{uuid.uuid4().hex[:8]}"
    completed, run_root, log = _run_rig(root, adapter_wheel, spawn_interpreter, ending="narration")

    outcome = _outcome(completed, expect_status="agent_stop")
    assert outcome["steps"] == 1
    # The ending NAMES itself: a plain terminal message, no tool call.
    assert outcome["terminal_reason"] == "no-tool-call"
    assert "carries no tool call" in outcome["diagnostic"]
    kinds = _record_kinds(run_root)
    assert "action_completion_challenged" not in kinds
    assert "action_finish" not in kinds
    events = _log_events(log)
    assert len([event for event in events if event.get("episode")]) == 2, events
    assert not any(
        "That declaration is a CLAIM" in (event.get("last_user_text") or "") for event in events
    )


@pytest.mark.parametrize("child_acts", ["finish", "wait"])
def test_a_delegated_child_cannot_drive_or_end_the_episode(
    durable_path: Path,  # noqa: F811
    adapter_wheel: Path,
    spawn_interpreter: spawn_helpers.SpawnInterpreter,
    child_acts: str,
) -> None:
    root = durable_path / f"s-{uuid.uuid4().hex[:8]}"
    completed, run_root, log = _run_rig(
        root, adapter_wheel, spawn_interpreter, child_acts=child_acts
    )

    outcome = _outcome(completed)
    # The episode ended because the PARENT finished it: exactly one step (the
    # parent's wait batch). A child's batch must never count as a step, and a
    # child's finish must never end the run.
    assert outcome["steps"] == 1
    assert outcome["terminal_reason"] == "finish"
    kinds = _record_kinds(run_root)
    assert kinds.count("action_batch") == 1
    assert kinds.count("action_completion_challenged") == 1
    assert kinds.count("action_finish") == 1

    events = _log_events(log)
    child_calls = [event for event in events if event.get("episode") is False]
    assert child_calls, events
    # The child never held the action tool (no inheritance)...
    assert not any(event.get("action_tool_advertised") for event in child_calls)
    # ...it called anyway (the probe), and the call was refused, not executed.
    assert any(event.get("emit") == "child-refusal-seen" for event in events), events
    assert not any(
        event.get("emit") == "child-refusal-seen" and event.get("episode") is True
        for event in events
    )
