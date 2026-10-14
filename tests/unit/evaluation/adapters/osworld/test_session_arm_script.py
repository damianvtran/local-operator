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
import signal
import subprocess
import sys
import time
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
    max_steps: int | None = None,
    task_id: str = "task_plain",
    tasks: dict[str, str] | None = None,
    provider: dict[str, object] | None = None,
    ask_first: bool = False,
    cleanup_hook: str | None = None,
) -> tuple[subprocess.CompletedProcess[str], Path, Path]:
    selector_dir = root / "adapter"
    selector = spawn_helpers.build_spawnable_adapter(
        selector_dir,
        wheel,
        tasks if tasks is not None else {"task_plain": fixtures.PLAIN},
        provider=provider or {"provider": "fake", "scripted_score": 1.0},
        interpreter=interpreter,
    )
    # The session engagement's source pin (blueprint §4): a selector without
    # ``core_source`` is refused at preflight, so the rig composes one exactly
    # as a real selector build does -- the pin of the tree that will run the
    # arm (this checkout, whose local_operator the copied interpreter also
    # imports through its .pth).
    from local_operator.evaluation.adapters.api import CoreSourcePin
    from local_operator.evaluation.source_pin import (
        compute_core_source_hashes,
        resolve_core_commit,
    )

    selector = selector.model_copy(
        update={
            "core_source": CoreSourcePin(
                commit=resolve_core_commit() or "unavailable-test-tree",
                module_hashes=compute_core_source_hashes(),
            )
        }
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
    args += ["--ending", ending, "--task-id", task_id]
    if max_steps is not None:
        args += ["--max-steps", str(max_steps)]
    if ask_first:
        args += ["--ask-first"]
    if cleanup_hook is not None:
        args += ["--cleanup-hook", cleanup_hook]
    if cleanup_hook == "cancelled":
        # A permanently failed canonical MUST retain its owner, not return an
        # outcome. This fixture owns the isolated process lifetime: observe
        # provisional retention, then perform explicit exact-group teardown.
        process = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            cwd=str(REPO),
            start_new_session=True,
        )
        retained = False
        try:
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline and process.poll() is None:
                records = list(run_root.rglob("events.jsonl"))
                if records and '"kind": "provisional"' in records[0].read_text():
                    retained = True
                    break
                time.sleep(0.1)
            assert retained and process.poll() is None, "failed cleanup must retain its owner"
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate(timeout=10)
        completed = subprocess.CompletedProcess(args, process.returncode, stdout, stderr)
    else:
        completed = subprocess.run(
            args,
            capture_output=True,
            text=True,
            env=env,
            cwd=str(REPO),
            check=False,
            timeout=180,
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
    # Epochs at ADMISSION (blueprint §3): the wait batch, the challenged
    # first finish and the accepted re-declared finish -- one apiece.
    assert outcome["steps"] == 3
    assert outcome["terminal_reason"] == "finish"
    assert outcome["score"]["status"] == "scored" and outcome["score"]["binary"] == 1
    assert outcome["rescue_required"] is False
    assert "mcp__episode_actions_apply_actions" in outcome["tool_names"]

    kinds = _record_kinds(run_root)
    assert kinds.count("action_output_epoch") == 3
    # The request observer's ONE serialized flush (blueprint §2): the real
    # session tree's request finals land in the record with the two money
    # figures kept separate, even on this zero-cost rig.
    assert "requests" in kinds
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
    # Epochs at admission: wait, challenged finish, corrective wait, accepted
    # finish -- the corrective action after the challenge counted, and each
    # finish proposal charged its own.
    assert outcome["steps"] == 4
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
    re-declaration ends the episode ``completed`` -- one wait epoch and one
    (re-declared tool) finish epoch at admission (blueprint §3; the prose
    claim itself never reaches the bridge).
    """

    from tests.unit.evaluation.adapters.osworld import session_arm_rig

    root = durable_path / f"s-{uuid.uuid4().hex[:8]}"
    completed, run_root, log = _run_rig(
        root, adapter_wheel, spawn_interpreter, ending="prose-claim"
    )

    outcome = _outcome(completed)
    # The opening wait took one epoch; the re-declared tool finish (answered
    # to the PROSE challenge) took the second -- the prose claim itself never
    # reaches the bridge.
    assert outcome["steps"] == 2
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
    # The episode ended because the PARENT finished it: the parent's epochs
    # are wait + challenged finish + accepted finish, and the child's refused
    # calls (rejections before admission) took none of them. A child's batch
    # must never count as an epoch, and a child's finish must never end the run.
    assert outcome["steps"] == 3
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


def test_the_step_budget_stops_before_the_next_root_request(
    durable_path: Path,  # noqa: F811
    adapter_wheel: Path,
    spawn_interpreter: spawn_helpers.SpawnInterpreter,
) -> None:
    """Close guest admission AT the budget (blueprint §3), on the real loop.

    The script would request a wait and then a finish; run with ``--max-steps
    1``, the FIRST admitted batch consumes the only epoch, and the Nth-stop
    must fire when that batch's result is paired: the loop's own graceful
    boundary ends the turn, the finish is NEVER requested, and the reached
    state is still scored (truncated, not failed). Without the sink stop this
    is exactly the N+1 shape: the model would spend one more request whose
    only possible answer is the bridge's budget refusal.
    """

    root = durable_path / f"s-{uuid.uuid4().hex[:8]}"
    completed, run_root, log = _run_rig(root, adapter_wheel, spawn_interpreter, max_steps=1)

    outcome = _outcome(completed, expect_status="truncated")
    assert outcome["steps"] == 1
    assert outcome["terminal_reason"] == "max-steps"
    # The budget stops the run, not the score: the state reached still scores.
    assert outcome["score"]["status"] == "scored"

    kinds = _record_kinds(run_root)
    assert kinds.count("action_batch") == 1
    assert kinds.count("budget_stop") == 1
    assert "action_finish" not in kinds

    # The proof the stop landed AT the budget: the second episode request the
    # script would have made (the finish) was never spent.
    events = _log_events(log)
    assert len([event for event in events if event.get("episode")]) == 1, events


def test_the_budget_stop_fires_after_a_terminal_finish_at_the_nth_epoch(
    durable_path: Path,  # noqa: F811
    adapter_wheel: Path,
    spawn_interpreter: spawn_helpers.SpawnInterpreter,
) -> None:
    """R3: a terminal finish at epoch N must not spend root request N+1.

    Review round 1 measured the gap on this exact shape: with ``--max-steps
    1`` the accepted failed finish consumed epoch 1 but the run recorded TWO
    root turn requests and no ``budget_stop``. The stop now fires on the
    PAIRED result of the Nth admitted epoch regardless of the terminal
    ``end_requested``, so the second request is never spent.
    """

    root = durable_path / f"s-{uuid.uuid4().hex[:8]}"
    completed, run_root, log = _run_rig(
        root, adapter_wheel, spawn_interpreter, ending="budget-failed-finish", max_steps=1
    )

    outcome = _outcome(completed)
    assert outcome["steps"] == 1
    assert outcome["terminal_reason"] == "finish"

    kinds = _record_kinds(run_root)
    assert kinds.count("budget_stop") == 1
    assert kinds.count("action_finish") == 1
    # The observer's root-turn count is the discriminator: ONE root request,
    # not two. (This is the caller's stop boundary, not a claim that root
    # requests equal official epochs.)
    requests_row = _record_payloads(run_root, "requests")[0]
    assert requests_row["purpose_counts"] == {"turn": 1}


def test_the_budget_stop_fires_after_the_accepted_challenge_at_the_nth_epoch(
    durable_path: Path,  # noqa: F811
    adapter_wheel: Path,
    spawn_interpreter: spawn_helpers.SpawnInterpreter,
) -> None:
    """R3: challenged at epoch 1, accepted at epoch 2 == the budget: stop there.

    Pre-fix this shape recorded THREE root turn requests (the accepted finish
    ended guest admission but not the SDK turn, and no stop was sent); the
    fix leaves it at two: the opening request and the one post-challenge
    re-declaration, with the stop closing admission at the accepted epoch.
    """

    root = durable_path / f"s-{uuid.uuid4().hex[:8]}"
    completed, run_root, log = _run_rig(
        root, adapter_wheel, spawn_interpreter, ending="budget-done-finish", max_steps=2
    )

    outcome = _outcome(completed)
    assert outcome["steps"] == 2
    assert outcome["terminal_reason"] == "finish"

    kinds = _record_kinds(run_root)
    assert kinds.count("action_completion_challenged") == 1
    assert kinds.count("action_finish") == 1
    assert kinds.count("budget_stop") == 1
    requests_row = _record_payloads(run_root, "requests")[0]
    assert requests_row["purpose_counts"] == {"turn": 2}


def test_the_ask_receipt_reaches_the_next_actual_request(
    durable_path: Path,  # noqa: F811
    adapter_wheel: Path,
    spawn_interpreter: spawn_helpers.SpawnInterpreter,
) -> None:
    """R4: the canonical exchange receipt rides the tool result CONTENT.

    The MCP forwarding layer delivers only the reply's content blocks to the
    next root request, so a receipt kept in the bridge-local ``details`` is
    dropped at that boundary (measured in review round 1: the answer arrived,
    the receipt did not). The scripted client LOGS the tool text of the
    request it sees AFTER the ask, and this test asserts the canonical
    receipt (ask_id, request digest, accepted, public answer) is present
    there -- the actual next ChatRequest, not a bridge dictionary.
    """

    root = durable_path / f"s-{uuid.uuid4().hex[:8]}"
    completed, run_root, log = _run_rig(
        root,
        adapter_wheel,
        spawn_interpreter,
        task_id="task_simulator",
        tasks={"task_simulator": fixtures.SIMULATOR},
        provider={"provider": "fake", "scripted_score": 1.0, "has_user_simulator": True},
        ask_first=True,
    )

    outcome = _outcome(completed)
    assert outcome["terminal_reason"] == "finish"
    # Epochs: the ask exchange and the terminal finish at admission.
    assert outcome["steps"] == 2

    rows = _log_events(log)
    next_request = next((row for row in rows if row.get("emit") == "next-request-tool-text"), None)
    assert next_request is not None, rows
    text = "\n".join(next_request["tool_text"])
    assert "The simulated user answered: simulated user answer" in text
    assert "ask_user exchange receipt" in text
    assert '"ask_id": "ask-rig-1"' in text
    assert '"request_digest":' in text
    assert '"accepted": true' in text
    # One exchange row with the canonical receipt in the record.
    exchange_rows = _record_payloads(run_root, "action_ask_exchange")
    assert len(exchange_rows) == 1
    assert exchange_rows[0]["receipt"]["ask_id"] == "ask-rig-1"


def test_an_incomplete_owned_join_retains_the_worker_end_to_end(
    durable_path: Path,  # noqa: F811
    adapter_wheel: Path,
    spawn_interpreter: spawn_helpers.SpawnInterpreter,
) -> None:
    """R2: a cancelled cleanup hook keeps the release RPCs from ever going out.

    Real SDK + MCP + spawned worker with the reviewer's fault (a public
    dispose hook raising CancelledError): the owning receipt is honestly
    incomplete. The owning process must stay alive with no score, terminal
    seal or destructive cleanup RPC. A managed FRONTEND may return boundedly;
    this injected in-process rig has no frontend and explicitly tears down
    its own process only after it measured provisional retention.
    """

    root = durable_path / f"s-{uuid.uuid4().hex[:8]}"
    completed, run_root, log = _run_rig(
        root, adapter_wheel, spawn_interpreter, cleanup_hook="cancelled"
    )

    assert completed.returncode == -signal.SIGKILL
    assert not list(run_root.rglob("outcome.json")), "no seal over unverified ownership"
    rows = _log_events(log)
    assert not any(row.get("emit") == "cleanup-rpc" for row in rows), rows
    kinds = _record_kinds(run_root)
    assert "provisional" in kinds
    assert "end" not in kinds
    assert "score" not in kinds
