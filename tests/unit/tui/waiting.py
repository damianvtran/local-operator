"""Readiness waits driven by Textual's existing test message publication.

Pass ``messages.on_message`` to ``App.run_test(message_hook=...)`` before boot.
This observes the real message/worker path; it neither pumps idle frames nor
replaces startup. A ready session is NOT a settled layout: geometry, animation
and parser-timer tests still need their own frame or timer barriers.

``settle`` is the pilot-side counterpart: it joins the work a test depends on
(queued messages, named worker groups, tracked thread hops, the after-refresh
queue) before checking a predicate, instead of trusting ``pilot.pause()``'s CPU
idle heuristic. See the block comment above it for the contract.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable, Collection, Iterable

from textual.message import Message
from textual.worker import Worker, WorkerCancelled, WorkerFailed, WorkerState

#: The one hang guard for an edge a TUI test awaits: ``settle``'s and
#: ``MessageWaiter.wait_for``'s default bound, and the ceiling for a bare
#: ``asyncio.wait_for`` on an app event. Named so those waits share it instead
#: of each betting on its own small literal: a ceiling costs nothing on a pass
#: (the wait returns on the edge), while a 2 s literal failed on a loaded CI
#: shard (test_goal_judge_lifecycle, gate run 43188636453).
SETTLE_DEADLINE_S = 30.0


class MessageWaiter:
    """Recheck observable state on messages, with a hang guard, not a time bet."""

    def __init__(self) -> None:
        self._pending: set[asyncio.Event] = set()

    def on_message(self, message: Message) -> None:
        # One event per await lets independent consumers cancel without stealing
        # each other's wake. The hook has no subscription or task to outlive the
        # run_test context that owns it, and is inert between waits.
        for changed in tuple(self._pending):
            changed.set()

    async def wait_for(
        self,
        predicate: Callable[[], bool],
        *,
        description: str,
        timeout: float = SETTLE_DEADLINE_S,
    ) -> None:
        """Return on readiness, including readiness published before this wait.

        The hook runs before a message handler: publication means "recheck",
        not "complete". Never consume a wake after testing the predicate, since
        the predicate can itself arrange another publication. The timeout only
        turns a missing edge into a diagnostic instead of hanging the shard.
        """
        changed = asyncio.Event()
        self._pending.add(changed)
        guard = asyncio.timeout(timeout)
        try:
            async with guard:
                while True:
                    changed.clear()
                    if predicate():
                        return
                    await changed.wait()
        except TimeoutError as error:
            if not guard.expired():
                raise
            raise AssertionError(f"never became ready: {description}") from error
        finally:
            self._pending.remove(changed)


# -- settle: one join-based wait for pilot tests --------------------------------
#
# Why this exists: ``pilot.pause()`` ends in ``wait_for_idle`` (textual/_wait.py),
# which returns once process CPU time advances by less than ~1 ms across a 20 ms
# sleep. That is a *load heuristic*, not a join: a test process the OS is not
# scheduling burns no CPU, reads as idle, and the pause returns before the
# message cascade, worker or thread hop it was standing in for has run. CI
# shards under load hit exactly that, which is why one pause "usually" works and
# a pile of extra pauses only moves the odds. ``settle`` replaces the bet with
# the events a test actually depends on, each awaited to completion:
#
#   1. messages already queued   -> ``pilot.pause()`` (event-exact for what is
#      queued when it is called; it stays here to pace and to force a timer
#      update, never as the join);
#   2. named worker groups       -> ``Worker.wait()`` on a fresh snapshot;
#   3. threads / owned tasks     -> ``joins`` (see ``track_threads``);
#   4. the after-refresh queue   -> ``app.screen.wait_for_refresh()``;
#   5. then, and only then, the caller's predicate.
#
# Joins run before the predicate on every turn, the first included: a predicate
# that turns true early (a state flag set before the paint it implies) is the
# failure this guards against.
#
# Constraints, deliberately:
#   * No sleeps and no widened timeouts. ``deadline_s`` is the only default
#     bound: it turns a missing edge into a loud diagnostic, and every join runs
#     under ``asyncio.timeout(remaining)`` so one wedged join cannot outlive it.
#     ``max_turns`` defaults to None (no cap). A turn is NOT a unit of time. A
#     real ``Pilot.pause()`` (Textual 8.2.8) is ``wait_for_idle(0)``: at least
#     one ``sleep(1/50)``, looping up to 1 s while the process looks busy, so a
#     turn costs >= ~20 ms (21.5-43.6 ms measured at idle) and up to ~1 s under
#     load: a 240-turn cap is ~5 s on an idle box and ~240 s on a loaded one.
#     The ~1 ms/turn figure (QA: 300 turns in 0.20-0.49 s on PR #3732) is the
#     ``StarvedPilot`` (``sleep(0)``) in ``test_settle.py``, not a real pilot.
#     Either way a count bounds nothing stable, so a wall deadline is the
#     default bound. The parameter stays for the helper's own contract tests,
#     where ``max_turns=1`` proves a join is exact (one turn suffices) rather
#     than won by repeated polling.
#   * Wait for every state the predicate reads, not just the paint. The first
#     migration compared the sidebar's painted state with its live state while
#     the boot session's adoption (``current_id`` '' -> 'sess') could land after
#     the paint without requesting a repaint; the predicate then never held,
#     with the screen idle and clean. A predicate over state that can change
#     without a refresh needs its own earlier ``settle`` for that state.
#   * Name only worker groups whose workers terminate. A long-lived poller group
#     never joins and ends at the deadline. Likewise a test that asserts an
#     *intermediate* state (a "pending" latch) must not join the worker that
#     clears it.
#   * ``wait_for_refresh`` hangs forever on an app that has stopped (Textual drops
#     the callback), hence the ``is_running`` check and the deadline around it.
#   * The predicate names the event, not the assertion: callers keep their exact
#     assertion afterwards so a wrong value still fails with its own message. An
#     always-true predicate degrades to one join turn; pair every migration with
#     an injected negative control (see the PR that introduced this helper).
#
# Convention (the ONE settle for TUI tests; the shared ``_settle.py`` turn-count
# poller was folded into this, and new and migrated waits use ``settle``. Some
# per-file pollers remain, e.g. ``_settle`` in ``test_agent_class_slash.py``,
# ``test_app_pilot.py`` (plus its ``_settle_loop``), ``test_approvals_ux.py``,
# ``test_ask_dismiss_open.py`` and ``test_ask_fleet.py``; they migrate here as
# they are touched):
#   * Import ``settle`` from ``tests.unit.tui.waiting``; no new local pollers.
#   * Name in ``workers=`` the product worker group(s) that write the state the
#     predicate reads (e.g. ``"harness-import"``), and only those. State a
#     message handler or a synchronous transition sets needs no group: the
#     per-turn message and refresh joins cover it.
#   * ``track_threads`` only for a raw ``asyncio.to_thread`` outside a worker.
#   * The predicate is the product precondition, never the assertion; keep the
#     exact assert afterwards.
#   * Never pass ``max_turns`` outside this helper's own tests.
#   * Every migration PR shows an injected negative control (predicate held
#     false -> "never settled: <what>"; a named worker raising -> "worker
#     <group>/<name> failed").
#
# Usage::
#
#     await settle(pilot, lambda: view._state == "sessions",
#                  what="omp session list painted", workers=("harness-import",))
#     threads = track_threads(monkeypatch)   # opt-in, before the code under test
#     await settle(pilot, lambda: done, what="probe finished", joins=(threads,))


class _Unsettled(AssertionError):
    """settle's own bound firing, kept apart from a join's own failure."""


class ThreadRegistry:
    """Every ``asyncio.to_thread`` call made while tracking, as joinable tasks.

    Awaiting the registry joins every hop recorded so far, including hops a
    joined hop scheduled, so it can be passed to ``settle(joins=...)`` and is
    re-awaited on every turn. Exceptions belong to the code under test, which
    observes them through its own ``await``; the join only waits for completion.
    """

    def __init__(self) -> None:
        self.tasks: list[asyncio.Future[object]] = []

    def pending(self) -> list[asyncio.Future[object]]:
        return [task for task in self.tasks if not task.done()]

    async def join(self) -> None:
        while pending := self.pending():
            await asyncio.wait(pending)

    def __await__(self):
        return self.join().__await__()


def track_threads(monkeypatch) -> ThreadRegistry:
    """Wrap ``asyncio.to_thread`` for this test and return its join registry.

    Opt-in because the patch is process-wide for the test's duration. It sees
    only call sites that resolve ``asyncio.to_thread`` at call time; a module that
    did ``from asyncio import to_thread`` keeps the original. Textual's own
    ``thread=True`` workers are joined through ``workers=`` instead.
    """
    registry = ThreadRegistry()
    original = asyncio.to_thread

    async def tracked(func, /, *args, **kwargs):
        # Stay a coroutine function so ``create_task(asyncio.to_thread(...))``
        # keeps working; awaiting the inner task propagates cancellation to it.
        task = asyncio.ensure_future(original(func, *args, **kwargs))
        registry.tasks.append(task)
        return await task

    monkeypatch.setattr(asyncio, "to_thread", tracked)
    return registry


async def settle(
    pilot,
    until: Callable[[], bool],
    *,
    what: str,
    workers: Collection[str] = (),
    joins: Iterable[Awaitable[object]] = (),
    refresh: bool = True,
    max_turns: int | None = None,
    deadline_s: float = SETTLE_DEADLINE_S,
) -> None:
    """Join messages, named workers, ``joins`` and refresh; return once ``until()``.

    Raises ``AssertionError`` naming ``what`` at whichever bound fires first
    (``deadline_s``, or ``max_turns`` when given), with the turn count, elapsed time and the
    app's unfinished workers. A named-group worker that failed re-raises as an
    ``AssertionError`` naming ``what``; a cancelled one (``exclusive=True``
    replacing it) is ignored. Settled also requires the named groups to be
    empty of unfinished workers after the joins, so a worker the predicate's
    state was set *from* has finished, not merely started.

    ``wait_for_idle`` (what a bare ``pilot.pause()`` ends in) is deliberately not
    one of the joins: it compares CPU time with wall time (textual/_wait.py) and
    returns when the process looks idle, which a descheduled process always does.
    """
    app = pilot.app
    groups = frozenset(workers)
    # Futures are re-awaitable, coroutines are not: wrap once so each turn can
    # await every join again (a finished one returns immediately). ``created``
    # are the tasks settle itself made: it owns them, so the ``finally`` below
    # cancels any still running when settle raises and retrieves their outcome,
    # rather than leaving an orphan to log "never retrieved" at teardown.
    created: list[asyncio.Future[object]] = []
    owned: list[ThreadRegistry | asyncio.Future[object]] = []
    for join in joins:
        if isinstance(join, ThreadRegistry) or asyncio.isfuture(join):
            owned.append(join)
        else:
            task = asyncio.ensure_future(join)
            created.append(task)
            owned.append(task)
    started = time.monotonic()
    turns = 0

    def unfinished(only_named: bool) -> list[Worker[object]]:
        return [
            worker
            for worker in app.workers
            if not worker.is_finished and (not only_named or worker.group in groups)
        ]

    def fail(bound: str) -> AssertionError:
        pending = ", ".join(
            f"{worker.group or '<no group>'}/{worker.name or '?'}={worker.state.name}"
            for worker in unfinished(only_named=False)
        )
        return _Unsettled(
            f"never settled: {what} — {bound} after {turns} turn(s), "
            f"{time.monotonic() - started:.2f}s elapsed; "
            f"unfinished workers: {pending or 'none'}"
        )

    async def bounded(awaitable: Awaitable[object]) -> None:
        remaining = deadline_s - (time.monotonic() - started)
        if remaining <= 0:
            if asyncio.iscoroutine(awaitable):
                awaitable.close()  # never started; don't leak a "never awaited" warning
            raise fail(f"deadline {deadline_s}s")
        timeout = asyncio.timeout(remaining)
        try:
            async with timeout:
                await awaitable
        except TimeoutError as error:
            # Only OUR timeout is a deadline; a TimeoutError a join raised itself
            # is that join's failure and propagates as such.
            if timeout.expired():
                raise fail(f"deadline {deadline_s}s") from error
            raise

    def retrieve(task: asyncio.Future[object]) -> None:
        # Done-callback for tasks settle outlives or shields: marks the outcome
        # retrieved so asyncio never logs "Task exception was never retrieved".
        # The outcome itself is reported elsewhere (or belongs to the worker).
        if not task.cancelled():
            task.exception()

    # Every named-group worker seen so far, from call time on. Textual drops a
    # worker from ``app.workers`` the moment it finishes, so this keeps one that
    # was seen unfinished (at call time or after any pause) and then failed. It
    # cannot see a worker that starts AND finishes inside a single pause; that
    # worker's effects are still covered by the predicate.
    seen: dict[Worker[object], None] = dict.fromkeys(unfinished(only_named=True))
    try:
        while True:
            if max_turns is not None and turns >= max_turns:
                raise fail(f"turn cap {max_turns}")
            turns += 1
            await bounded(pilot.pause())
            seen.update(dict.fromkeys(unfinished(only_named=True)))
            for worker in seen:
                if worker.state == WorkerState.PENDING:
                    continue  # not started; Worker.wait() refuses it, next turn joins it
                # Shield: a deadline must fail this wait, not cancel the worker
                # (Worker.wait awaits the worker's own task). The inner task is
                # made first so it can carry ``retrieve``: after a deadline it
                # outlives this wait and ends WorkerCancelled at app teardown.
                waiter = asyncio.ensure_future(worker.wait())
                waiter.add_done_callback(retrieve)
                try:
                    await bounded(asyncio.shield(waiter))
                except WorkerCancelled:
                    pass
                except WorkerFailed as error:
                    raise AssertionError(
                        f"{what}: worker {worker.group}/{worker.name} failed: {error.error!r}"
                    ) from error
            for join in owned:
                try:
                    if isinstance(join, ThreadRegistry):
                        await bounded(join)
                    else:
                        # Shield: a deadline must fail this wait, not cancel the
                        # caller's task.
                        await bounded(asyncio.shield(join))
                except _Unsettled:
                    raise
                except Exception as error:
                    coro = join.get_coro() if isinstance(join, asyncio.Task) else None
                    label = getattr(coro, "__qualname__", None) or repr(join)
                    raise AssertionError(f"{what}: join {label} failed: {error!r}") from error
            if refresh and app.is_running:
                # One after-refresh callback per turn. Measured on PR #3732 this
                # does not hold the paint back: in every stalled run the screen
                # was clean (no layout/repaint flags, no dirty widgets), its
                # update timer paused and its queue empty, i.e. the paint had
                # landed and only the predicate's own state was stale.
                await bounded(app.screen.wait_for_refresh())
            if not unfinished(only_named=True) and until():
                return
    finally:
        for task in created:
            if not task.done():
                task.cancel()
            task.add_done_callback(retrieve)
