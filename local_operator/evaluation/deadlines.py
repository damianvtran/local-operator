"""Per-call deadlines derived from what a call's own request DECLARES.

WHY THIS MODULE EXISTS. Every adapter call travels through exactly one deadline
-- the ``asyncio.wait_for`` in ``adapters/rpc.py`` -- and for most methods the
caller's budget is a constant, because the request declares nothing about how
long its work takes: a handshake waits for a process to boot, ``score`` waits
for the benchmark's own evaluator, ``prepare`` is contractually allocation-free.
Two methods are different. Their request carries a DECLARATION of its own
duration, and a fixed budget that ignores it is not a conservative default but a
deterministic defect:

* ``execute`` carries an ``ActionBatch``. A ``wait`` action names its own
  duration, and the protocol admits up to ``MAX_BATCH_SIZE`` (64) of them at up
  to ``MAX_WAIT_MS`` (60 s) each -- 3_840 s of declared waiting in a single
  legal call.
* ``cleanup`` carries a ``CleanupPlan`` and the subset of its actions to run.
  Each selected action declares ``timeout_ms`` (bounded by
  ``MAX_CLEANUP_TIMEOUT_MS``, one hour) and ``max_attempts`` (bounded by
  ``MAX_CLEANUP_ATTEMPTS``, 32), and the worker applies the selection
  sequentially.

MEASURED, NOT ASSUMED. Episode ``ep-fca426e92c42`` of the OSWorld canary
campaign asked for ``key enter`` plus 4 x ``wait 60000`` -- 240 s of declared
waiting -- under a 180 s ``step_timeout``, and died at 186.3 s with a fatal,
non-retryable ``TimeoutError`` after 130 real steps, 1.7 h of wall time and
950_583 micro-USD of provider spend. The failure is a property of the batch
rather than of the machine: 240 s of declaration under a 180 s budget fails
every time that batch is emitted, and it is reproducible on demand against the
bundle's own action bytes (the PR's BASE/HEAD reproduction: dead at 181.03 s
= the 180 s budget plus the transport's 1 s cancel grace, before the bundle's
error event adds ~5.3 s of process-group teardown).

THE CASE THIS DOES NOT COVER, stated because it looks like the same failure and
is not. A second canary episode (``batch-k3v3-0/task_011``/``ep-c7c76156297d``)
died on the same ``TimeoutError`` at 906.1 s against a 900 s ``reset_timeout``.
Its bundle holds eight events and NO observation and NO action batch: the call
that timed out was ``reset_start`` waiting on cloud infrastructure, and it
declares nothing about how long that takes. That is work that took too long
rather than work the request announced, so there is no declaration here to
derive from and nothing in this module changes it. What fixes that class is a
faster or better-instrumented provider, or an operator raising the constant --
not a budget that reads the request.

WHAT THIS DOES NOT DO. It raises no constant and weakens no deadline. The
returned budget is the GREATER of the caller's own budget and the request's
declared work plus headroom, so a request that declares nothing is funded
exactly as before, and a hung call is still cut off by the same ``wait_for``
that cut it off before -- a genuinely wedged operation has declared no duration
to fund and cannot borrow one. Nor is any benchmark vocabulary involved: the
inputs are protocol objects (``ActionBatch``, ``CleanupPlan``) and the protocol's
own bounds, so every benchmark and the ordinary TUI path are funded identically.
The one interaction with a benchmark's wall clock is stated in
:data:`DECLARED_WORK_HEADROOM_S`.
"""

from __future__ import annotations

import math

from local_operator.evaluation.lifecycle import CleanupPlan
from local_operator.evaluation.protocol import (
    ActionBatch,
    AskUserAction,
    FinishAction,
    ProtocolModel,
    WaitAction,
)

#: Fixed allowance added to a request's own declared duration.
#:
#: WHAT IT COVERS: everything a call does that its request does NOT declare --
#: the guest round trips for a batch's non-wait actions, the observation the
#: adapter builds afterwards, publishing the frame, and the transport back.
#: CALIBRATION (the canary bundle above, 130 execute calls): over the 56 batches
#: that declared no waiting at all, the whole call took a median 4.93 s, p90
#: 6.08 s and at most **9.45 s**; over the 74 batches that did declare waiting,
#: the elapsed time in excess of the declaration had a median of 5.69 s and a
#: maximum of **11.61 s**. 30 s is therefore ~2.6x the worst of 130 measured
#: calls, and flat rather than per-action because the adapter batches consecutive
#: guest statements into ONE provider call (``adapter.py`` partitions a batch into
#: runs), so the undeclared cost does not grow with action count -- measured: the
#: worst wait-free call had 2 actions and the second worst had 6.
#:
#: WHAT IT COSTS AT THE BOUND -- PER FUNDED PATH, because one of them is far
#: larger than the other and stating only the smaller one would hide the real
#: number. Both ceilings come from the schema; neither is invented here.
#:
#: ``execute``: the largest legal declaration is 64 waits x 60 s, so the largest
#: funded call is 3_840 s + 30 s = 64.5 min. A model that spends its last 64
#: minutes waiting will hold one step open for 64 minutes, and a benchmark whose
#: protocol fixes wall-clock can overrun by up to that one step's declared
#: duration. The alternative is not cheaper, and the canary bundle is the
#: evidence: today that same call is cut off at the caller's budget, the channel
#: is poisoned, the worker is killed, the episode is forfeited unscored, and a
#: RESCUE worker is spawned to tear the guest down -- so the run is spent, the
#: provider spend is spent, and a fresh process pays the teardown anyway. A step
#: that runs its declared 64 minutes at least buys the observation the model
#: asked for.
#:
#: ``cleanup``: the honest ceiling is **29_491_200 s -- 341 days for ONE call**
#: -- and it is the protocol's, not this module's. A plan may hold
#: ``MAX_DECLARATIONS`` (256) actions (``lifecycle.py``), each declaring up to
#: ``MAX_CLEANUP_TIMEOUT_MS`` (1 h) x ``MAX_CLEANUP_ATTEMPTS`` (32), and the
#: episode path selects every action in ONE call (``episode._run_cleanup``).
#: That is not new exposure: the same aggregate teardown is already funded
#: today by ``supervisor.run_rescue``, which loops the same plan one action per
#: call at ``timeout_ms * max_attempts`` each (``supervisor.py:1414``) in a
#: FRESH worker, after the episode has already forfeited every cleanup receipt.
#: Funding it in one call spends the same time in one process and keeps the
#: receipts. What keeps this from being a practical hazard is that the plan is
#: the adapter's own verified ``prepare`` output (OSWorld declares 200 s, which
#: funds at 230 s); a plan written to burn a year of teardown is a declaration
#: problem, and no budget-shaped fix exists for it here -- any clamp would
#: refuse work the call was admitted to do, and would convert a completed
#: cleanup into a failed one plus a rescue that repeats the same work.
#:
#: The protocol's bounds are the only ceilings deliberately: any cap invented in
#: this module would be a second, arbitrary limit on top of the one the action
#: contract already states, and it would silently refuse admitted work.
DECLARED_WORK_HEADROOM_S: float = 30.0


def declared_work_seconds(
    params: ProtocolModel, *, execution_overhead_seconds_per_action: float = 0.0
) -> float:
    """Wall time the request itself declares, in seconds.

    ``0.0`` means "this request declares no duration", which is the answer for
    every method but ``execute`` and ``cleanup`` -- including the ones whose work
    is genuinely long (``reset_start`` boots cloud infrastructure, ``score`` runs
    a benchmark's evaluator): long is not the same as declared, and there is
    nothing in those requests to derive a budget from.

    ``cleanup`` sums the SELECTED actions, not the whole plan, because that is
    what the call asks the worker to do; the same formula is already the budget
    the rescue path grants one action at a time (``supervisor.run_rescue``), so
    this is the existing contract stated once instead of twice. The sum is not
    capped, deliberately: see the ceiling stated on
    :data:`DECLARED_WORK_HEADROOM_S`.

    One consequence specific to ``cleanup``, stated because it is easy to
    assume otherwise: a cleanup request ALWAYS declares something, since
    ``CleanupParams.action_ids`` has ``min_length=1`` and a ``CleanupAction``'s
    ``timeout_ms`` has ``ge=1``. So a cleanup call's effective floor is
    ``0.001 s + DECLARED_WORK_HEADROOM_S``, and the configured
    ``cleanup_timeout`` is dominated rather than binding -- at the library's
    60 s and the campaign's 120 s that changes nothing today, but an operator
    who sets a cleanup budget BELOW the headroom will not get it.

    The two cases are matched on the PROTOCOL OBJECT's own type
    (``ActionBatch``, ``CleanupPlan``) rather than on the caller's params class,
    because this module reads the action contract and not the adapter-facing
    params API. A future params model carrying an ``ActionBatch`` under
    ``action_batch`` and an ``action_ids`` selection over a ``CleanupPlan`` under
    ``cleanup_plan`` is an execute-shaped and cleanup-shaped request by
    construction, so funding it is the intent rather than an accident -- the
    type guard is what stops any OTHER model with a coincidentally named
    attribute from being funded.
    """

    if (
        isinstance(execution_overhead_seconds_per_action, bool)
        or not isinstance(execution_overhead_seconds_per_action, (int, float))
        or not math.isfinite(execution_overhead_seconds_per_action)
        or execution_overhead_seconds_per_action < 0.0
    ):
        raise ValueError("execution_overhead_seconds_per_action must be finite and nonnegative")

    batch = getattr(params, "action_batch", None)
    if isinstance(batch, ActionBatch):
        wait_seconds = (
            sum(action.duration_ms for action in batch.actions if isinstance(action, WaitAction))
            / 1000.0
        )
        # Finish/ask-user end a decision without executing desktop work. The
        # per-action apparatus allowance funds only semantic actions that can
        # mutate or wait in the environment; waits also retain their declared
        # duration above.
        executed_action_count = sum(
            not isinstance(action, (FinishAction, AskUserAction)) for action in batch.actions
        )
        return wait_seconds + executed_action_count * execution_overhead_seconds_per_action
    plan = getattr(params, "cleanup_plan", None)
    selected = getattr(params, "action_ids", None)
    if isinstance(plan, CleanupPlan) and selected is not None:
        wanted = set(selected)
        return sum(
            action.timeout_ms / 1000.0 * action.max_attempts
            for action in plan.actions
            if action.action_id in wanted
        )
    return 0.0


def funded_timeout(
    configured: float,
    params: ProtocolModel,
    *,
    execution_overhead_seconds_per_action: float = 0.0,
) -> float:
    """The deadline that governs this call: never less than its own declaration.

    A request that declares nothing returns ``configured`` UNCHANGED -- not
    ``configured`` plus headroom, which would quietly extend every timeout in the
    harness. A request that declares work returns the greater of the two, so the
    caller's budget stays a floor (an operator raising ``step_timeout`` still
    raises the effective deadline for every call) and the declaration becomes a
    floor as well (a legal batch is never killed by a budget that never saw it).

    That second floor is ``declared + DECLARED_WORK_HEADROOM_S`` and it does not
    shrink with the declaration: a 1 ms wait lifts a 1 s budget to 30.001 s. It
    is intended, because the headroom is the cost of the part of the call that no
    request field declares (see the constant for the measurement) and that part
    exists whether or not a wait was declared -- a wait-free call simply gets its
    allowance from the operator's configured budget instead. The trigger is
    therefore "declared anything", not "declared a lot".

    Execution overhead is an evaluation-only opt-in. The episode runner passes
    it only for a mutating ``execute`` while the paper-settle policy is active;
    ordinary calls retain the zero default. A resume carries the same batch for
    identity, but only re-reads state, so its caller does not pass this rate.

    The RESCUE path gains the headroom as well, and that is intended rather than
    an oversight: ``supervisor.run_rescue`` passes one action's
    ``timeout_ms / 1000 * max_attempts`` as that call's budget, and the call's
    own params declare the same product, so the effective deadline for a rescue
    step moves from the product to the product plus the headroom. Rescue calls do
    the same undeclared work as any other cleanup call (a cloud terminate, a
    lease revocation) under a budget sized only from its declared action, so the
    allowance belongs there for the same reason it belongs anywhere else.
    """

    declared = declared_work_seconds(
        params, execution_overhead_seconds_per_action=execution_overhead_seconds_per_action
    )
    if declared <= 0.0:
        return configured
    return max(configured, declared + DECLARED_WORK_HEADROOM_S)
