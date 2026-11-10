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
        timeout: float = 30.0,
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
#   * No sleeps and no widened timeouts. The two bounds only turn a missing edge
#     into a loud diagnostic. ``max_turns`` is the fast signal (each turn costs at
#     least one 20 ms idle probe, so 300 turns is >= ~6 s on a healthy box and
#     can only get slower, never fire earlier, under starvation). ``deadline_s``
#     is the backstop for one wedged join, so every join runs under
#     ``asyncio.timeout(remaining)``.
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
# Usage::
#
#     await settle(pilot, lambda: view._state == "sessions",
#                  what="omp session list painted", workers=("harness-import",))
#     threads = track_threads(monkeypatch)   # opt-in, before the code under test
#     await settle(pilot, lambda: done, what="probe finished", joins=(threads,))


class ThreadRegistry:
    """Every ``asyncio.to_thread`` call made while tracking, as joinable tasks.

    Awaiting the registry joins every hop recorded so far, including hops a
    joined hop scheduled, so it can be passed to ``settle(joins=...)`` and is
    re-awaited on every turn. Exceptions belong to the code under test, which
    observes them through its own ``await``; the join only waits for completion.
    """

    def __init__(self) -> None:
        self.tasks: list[asyncio.Future] = []

    def pending(self) -> list[asyncio.Future]:
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
    joins: Iterable[Awaitable] = (),
    refresh: bool = True,
    max_turns: int = 300,
    deadline_s: float = 30.0,
) -> None:
    """Join messages, named workers, ``joins`` and refresh; return once ``until()``.

    Raises ``AssertionError`` naming ``what`` at whichever bound fires first
    (``max_turns`` or ``deadline_s``), with the turn count, elapsed time and the
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
    # await every join again (a finished one returns immediately).
    owned = [
        join if isinstance(join, ThreadRegistry) else asyncio.ensure_future(join) for join in joins
    ]
    started = time.monotonic()
    turns = 0

    def unfinished(only_named: bool) -> list[Worker]:
        return [
            worker
            for worker in app.workers
            if not worker.is_finished and (not only_named or worker.group in groups)
        ]

    def fail(bound: str, cause: BaseException | None = None) -> AssertionError:
        pending = ", ".join(
            f"{worker.group or '<no group>'}/{worker.name or '?'}={worker.state.name}"
            for worker in unfinished(only_named=False)
        )
        return AssertionError(
            f"never settled: {what} — {bound} after {turns} turn(s), "
            f"{time.monotonic() - started:.2f}s elapsed; "
            f"unfinished workers: {pending or 'none'}"
        )

    async def bounded(awaitable: Awaitable) -> None:
        remaining = deadline_s - (time.monotonic() - started)
        if remaining <= 0:
            if asyncio.iscoroutine(awaitable):
                awaitable.close()  # never started; don't leak a "never awaited" warning
            raise fail(f"deadline {deadline_s}s")
        try:
            async with asyncio.timeout(remaining):
                await awaitable
        except TimeoutError as error:
            raise fail(f"deadline {deadline_s}s") from error

    # Every named-group worker seen so far, from call time on. Textual drops a
    # worker from ``app.workers`` the moment it finishes, so a worker that fails
    # during the first pause would otherwise vanish before any snapshot sees it.
    seen: dict[Worker, None] = dict.fromkeys(unfinished(only_named=True))
    while True:
        if turns >= max_turns:
            raise fail(f"turn cap {max_turns}")
        turns += 1
        await bounded(pilot.pause())
        seen.update(dict.fromkeys(unfinished(only_named=True)))
        for worker in seen:
            if worker.state == WorkerState.PENDING:
                continue  # not started; Worker.wait() refuses it, next turn joins it
            try:
                # Shield: a deadline must fail this wait, not cancel the worker
                # (Worker.wait awaits the worker's own task).
                await bounded(asyncio.shield(worker.wait()))
            except WorkerCancelled:
                pass
            except WorkerFailed as error:
                raise AssertionError(
                    f"{what}: worker {worker.group}/{worker.name} failed: {error.error!r}"
                ) from error
        for join in owned:
            # Shield: a deadline must fail this wait, not cancel the caller's task.
            await bounded(join if isinstance(join, ThreadRegistry) else asyncio.shield(join))
        if refresh and app.is_running:
            await bounded(app.screen.wait_for_refresh())
        if not unfinished(only_named=True) and until():
            return
