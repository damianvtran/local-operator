"""A call's deadline is never less than what that call's own request declared.

WHY THIS FILE EXISTS. ``adapters/rpc.py`` holds the only deadline in the
harness: one ``asyncio.wait_for``. Every caller passes a constant, which is
correct for a request that declares nothing and was FATAL for the two that do.
Measured, from the preserved bundle of episode ``ep-fca426e92c42``
(``runs/batch-deepseek-flash-canary9b/task_016``): the model emitted
``wait 60000``, ``key enter``, then three more ``wait 60000`` -- 240 s of
declared waiting -- against a 180 s ``step_timeout``, and the episode died at
186.3 s (180 budget + 1 s cancel grace + ~5.3 s teardown) with a fatal,
non-retryable ``TimeoutError`` after 130 real steps and 1.7 h of work. The
batch was LEGAL: ``MAX_BATCH_SIZE`` is 64 and ``MAX_WAIT_MS`` is 60 s.

So this file pins two things. First, the derivation itself: the seconds a
request declares, and that a request declaring nothing gets its configured
budget back UNCHANGED (not quietly extended by the headroom -- that would widen
every timeout in the harness). Second, that the bound is the protocol's own:
the largest legal ``execute`` is ``MAX_BATCH_SIZE`` x ``MAX_WAIT_MS``, so the
change cannot create an unbounded wait, only stop refusing work the harness had
already admitted.
"""

from __future__ import annotations

import pytest

from local_operator.evaluation.adapters.api import (
    CleanupParams,
    ExecuteParams,
    InspectRequirementsParams,
    ScoreParams,
)
from local_operator.evaluation.deadlines import (
    DECLARED_WORK_HEADROOM_S,
    declared_work_seconds,
    funded_timeout,
)
from local_operator.evaluation.evidence.models import canonical_digest
from local_operator.evaluation.lifecycle import CleanupAction, CleanupPlan
from local_operator.evaluation.protocol import (
    MAX_BATCH_SIZE,
    MAX_WAIT_MS,
    ActionBatch,
    ClickAction,
    KeyAction,
    ProtocolModel,
    WaitAction,
)

TASK = "task"
EPISODE = "episode"
OBSERVATION = "obs"


def batch(*actions: dict[str, object]) -> ActionBatch:
    """A real batch over one observation, exactly as the runner builds one."""

    return ActionBatch.model_validate(
        {
            "protocol_version": "1.0",
            "task_id": TASK,
            "episode_id": EPISODE,
            "observation_id": OBSERVATION,
            "actions": [{"observation_id": OBSERVATION, **action} for action in actions],
        }
    )


def execute(actions: dict[str, object] | list[dict[str, object]] | ActionBatch) -> ExecuteParams:
    if isinstance(actions, ActionBatch):
        built = actions
    elif isinstance(actions, list):
        built = batch(*actions)
    else:
        built = batch(actions)
    return ExecuteParams(
        operation_id="exec-0",
        action_batch=built,
        action_batch_id=canonical_digest("adapter-action-batch-v1", built),
    )


def cleanup_action(action_id: str, *, timeout_ms: int, max_attempts: int) -> CleanupAction:
    return CleanupAction(
        action_id=action_id,
        kind="release_instance",
        resource_ref=f"instance-{action_id}",
        timeout_ms=timeout_ms,
        max_attempts=max_attempts,
    )


def cleanup(plan: CleanupPlan, *action_ids: str) -> CleanupParams:
    return CleanupParams(
        operation_id="cleanup-0",
        cleanup_plan=plan,
        action_ids=tuple(action_ids),
    )


PLAN = CleanupPlan(
    episode_id=EPISODE,
    actions=(
        cleanup_action("a-first", timeout_ms=60_000, max_attempts=2),
        cleanup_action("b-second", timeout_ms=30_000, max_attempts=2),
        cleanup_action("c-third", timeout_ms=10_000, max_attempts=2),
    ),
)


def test_the_recorded_fatal_batch_is_funded_past_its_own_declaration() -> None:
    """The exact batch that killed ``ep-fca426e92c42``, at its exact numbers.

    ``wait 60000``, ``key enter``, ``wait 60000`` x3 -- the order the bundle's
    action artifact carries. The declaration is 240 s and the campaign's
    ``step_timeout`` was 180 s, so the funding has to come from the batch.
    """

    params = execute(
        [
            {"kind": "wait", "duration_ms": 60_000},
            {"kind": "key", "keys": ["enter"]},
            {"kind": "wait", "duration_ms": 60_000},
            {"kind": "wait", "duration_ms": 60_000},
            {"kind": "wait", "duration_ms": 60_000},
        ]
    )
    assert declared_work_seconds(params) == 240.0
    assert funded_timeout(180.0, params) == 240.0 + DECLARED_WORK_HEADROOM_S
    # The property that matters is not the exact sum but that the deadline now
    # exceeds the work, which is the opposite of the 186.3 s death above.
    assert funded_timeout(180.0, params) > declared_work_seconds(params)


def test_the_bound_is_the_protocols_own_maximum_not_a_new_constant() -> None:
    """The largest legal batch is the largest funded call; both come from the schema.

    Stated as a product of the protocol's own constants so it cannot drift from
    them: raising ``MAX_BATCH_SIZE`` or ``MAX_WAIT_MS`` in ``protocol.py`` moves
    this bound with it, and nothing benchmark-specific enters the harness.
    """

    params = execute([{"kind": "wait", "duration_ms": MAX_WAIT_MS} for _ in range(MAX_BATCH_SIZE)])
    declared = MAX_BATCH_SIZE * MAX_WAIT_MS / 1000.0
    assert declared == 3_840.0
    assert declared_work_seconds(params) == declared
    assert funded_timeout(120.0, params) == declared + DECLARED_WORK_HEADROOM_S
    assert funded_timeout(120.0, params) < 3_900.0


def test_a_cleanup_call_is_funded_for_the_actions_it_selects() -> None:
    """A plan's own declaration is per selected action, retries included.

    The formula is not new here: ``supervisor.run_rescue`` already grants one
    action ``timeout_ms / 1000 * max_attempts`` as its call budget. The episode
    path sent every selected action in ONE call under a flat
    ``cleanup_timeout`` (60 s by library default) while the plan declared 200 s
    of worst case -- the same mismatch as ``execute``, in the same call.
    """

    first = cleanup(PLAN, "a-first")
    assert declared_work_seconds(first) == 120.0
    assert funded_timeout(60.0, first) == 150.0

    # Only the SELECTED actions are funded: a plan may declare more than the
    # call asks the worker to do, and the worker loops over the selection.
    both = cleanup(PLAN, "a-first", "b-second")
    assert declared_work_seconds(both) == 180.0


def test_a_wait_free_execute_declares_nothing_and_keeps_its_budget() -> None:
    """A batch of clicks declares no duration, so nothing about it changes.

    This is the guard against the change becoming "add 30 s to every call":
    the headroom is added to a DECLARATION, and an undeclared call must get
    its configured budget back to the byte.
    """

    params = execute(
        [
            {"kind": "key", "keys": ["enter"]},
            {"kind": "key", "keys": ["tab"]},
        ]
    )
    assert declared_work_seconds(params) == 0.0
    assert not any(isinstance(action, WaitAction) for action in params.action_batch.actions)
    assert funded_timeout(120.0, params) == 120.0


@pytest.mark.parametrize(
    "params",
    [
        InspectRequirementsParams(),
        ScoreParams(operation_id="op-0", episode_id=EPISODE),
    ],
    ids=["inspect_requirements", "score"],
)
def test_a_request_that_declares_no_duration_is_untouched(params: ProtocolModel) -> None:
    """Every other method keeps exactly the deadline its caller chose."""

    assert declared_work_seconds(params) == 0.0
    for configured in (0.01, 60.0, 900.0):
        assert funded_timeout(configured, params) == configured


def test_the_configured_budget_stays_a_floor() -> None:
    """An operator raising a timeout must still raise the effective deadline."""

    params = execute([{"kind": "wait", "duration_ms": 60_000}])
    assert funded_timeout(900.0, params) == 900.0


def test_only_wait_actions_contribute_and_they_contribute_their_own_seconds() -> None:
    """The derivation reads the parsed batch, so a click's cost is not guessed.

    A click or a key declares nothing, and inventing a per-action cost for them
    would be the kind of unstated constant this module exists to avoid. Only
    ``wait`` states a duration, so only ``wait`` is summed.
    """

    built = batch(
        {"kind": "click", "frame_id": "screen", "x": 1, "y": 1, "button": "left"},
        {"kind": "wait", "duration_ms": 1_500},
        {"kind": "key", "keys": ["enter"]},
        {"kind": "wait", "duration_ms": 500},
    )
    assert isinstance(built.actions[0], ClickAction)
    assert isinstance(built.actions[2], KeyAction)
    assert declared_work_seconds(execute(built)) == 2.0
