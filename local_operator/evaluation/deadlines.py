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

from local_operator.evaluation.lifecycle import CleanupPlan
from local_operator.evaluation.protocol import ActionBatch, ProtocolModel, WaitAction

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
#: WHAT IT COSTS AT THE BOUND. The largest legal declaration is 64 x 60 s of
#: waiting, so the largest funded ``execute`` is 3_840 s + 30 s = 64.5 min. That
#: is a real consequence and not a rounding error: a model that spends its last
#: 64 minutes waiting will hold one step open for 64 minutes, and a benchmark
#: whose protocol fixes wall-clock can overrun by up to that one step's declared
#: duration. The alternative is not cheaper, and the canary bundle is the
#: evidence: today that same call is cut off at the caller's budget, the channel
#: is poisoned, the worker is killed, the episode is forfeited unscored, and a
#: RESCUE worker is spawned to tear the guest down -- so the run is spent, the
#: provider spend is spent, and a fresh process pays the teardown anyway. A step
#: that runs its declared 64 minutes at least buys the observation the model
#: asked for. The protocol bound is the
#: only ceiling here deliberately: any cap invented in this module would be a
#: second, arbitrary limit on top of the one the action contract already states,
#: and it would silently refuse work the harness had already admitted.
DECLARED_WORK_HEADROOM_S: float = 30.0


def declared_work_seconds(params: ProtocolModel) -> float:
    """Wall time the request itself declares, in seconds.

    ``0.0`` means "this request declares no duration", which is the answer for
    every method but ``execute`` and ``cleanup`` -- including the ones whose work
    is genuinely long (``reset_start`` boots cloud infrastructure, ``score`` runs
    a benchmark's evaluator): long is not the same as declared, and there is
    nothing in those requests to derive a budget from.

    ``cleanup`` sums the SELECTED actions, not the whole plan, because that is
    what the call asks the worker to do; the same formula is already the budget
    the rescue path grants one action at a time (``supervisor.run_rescue``), so
    this is the existing contract stated once instead of twice.
    """

    batch = getattr(params, "action_batch", None)
    if isinstance(batch, ActionBatch):
        return (
            sum(action.duration_ms for action in batch.actions if isinstance(action, WaitAction))
            / 1000.0
        )
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


def funded_timeout(configured: float, params: ProtocolModel) -> float:
    """The deadline that governs this call: never less than its own declaration.

    A request that declares nothing returns ``configured`` UNCHANGED -- not
    ``configured`` plus headroom, which would quietly extend every timeout in the
    harness. A request that declares work returns the greater of the two, so the
    caller's budget stays a floor (an operator raising ``step_timeout`` still
    raises the effective deadline for every call) and the declaration becomes a
    floor as well (a legal batch is never killed by a budget that never saw it).
    """

    declared = declared_work_seconds(params)
    if declared <= 0.0:
        return configured
    return max(configured, declared + DECLARED_WORK_HEADROOM_S)
