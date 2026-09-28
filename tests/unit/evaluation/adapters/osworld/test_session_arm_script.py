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
    completed = subprocess.run(
        args, capture_output=True, text=True, env=env, cwd=str(REPO), check=False
    )
    return completed, run_root, log


def _outcome(completed: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    assert completed.returncode == 0, completed.stderr[-4000:]
    text = completed.stdout.strip()
    start = text.find("{")
    assert start != -1, text[-2000:]
    outcome: dict[str, Any] = json.loads(text[start:])
    assert outcome["status"] == "completed", outcome
    return outcome


def _record_kinds(run_root: Path) -> list[str]:
    records = sorted(run_root.rglob("events.jsonl"))
    assert len(records) == 1, records
    return [json.loads(line)["kind"] for line in records[0].read_text().splitlines()]


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
        "action_finish",
    ):
        assert expected in kinds, kinds
    assert kinds.count("action_batch") == 1
    assert kinds.count("action_finish") == 1

    # The wait batch's rendered frame came back through the MCP tool result:
    # the model's next request carries the tool row the call produced.
    events = _log_events(log)
    assert any("Frames: screen" in (event.get("last_tool_text") or "") for event in events), [
        event.get("last_tool_text") for event in events
    ]

    # The canary secrets never surface outside the worker's own pipe.
    assert CANARY_SECRET not in completed.stdout
    assert CANARY_SECRET not in completed.stderr
    assert CANARY_KEY not in completed.stdout


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
