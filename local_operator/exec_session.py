"""One ordinary session lifetime for foreground and detached exec.

The runtime owns the prompt queue and loop scheduler; the renderer observes
that lifetime once. Publishing discovery never implicitly installs gates.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from typing import Any

from local_operator.exec_startup import apply_startup

logger = logging.getLogger(__name__)


async def _finish_browser_scope(session: Any, outcome: str) -> None:
    """Release the run's browser scope, if this build has that seam yet.

    Deliberately INERT until the browser owner's ``finish_browser_scope`` lands
    (PR #798): resolved by name so this file carries no speculative import of an
    API that does not exist, and simply does nothing on a build without it.

    Two properties are structural rather than incidental, because getting them
    wrong is how a cleanup helper starts corrupting outcomes:

    * **It can never alter the run's verdict.** The caller ignores the result
      and this function returns ``None``, so ``closed``, ``pending`` (a session
      with no lease/sidecar, or a bounded timeout) and ``unresolved`` (a stale
      generation or a scope mismatch) are all non-fatal by construction. The
      terminal ledger row is written by the worker afterwards regardless.
    * **No exception escapes.** A cleanup failure must not convert a succeeded
      run into a crashed one, so everything is swallowed to a debug log.

    The generation is the browser resource's execution-LEASE generation, read
    off the live session and passed verbatim. Exec's own ``process_generation``
    (a PID start-token) is deliberately NOT used here: it proves a recycled PID
    is not our worker in the exec ledger, which is a different question from
    which continuous run owns a browser scope.
    """
    finish = getattr(session, "finish_browser_scope", None)
    if finish is None:
        return
    try:
        generation = getattr(session, "browser_generation", None)
        if not generation:
            return
        # Awaited BEFORE the terminal outcome is published, per the contract.
        result = await finish(
            scope_id=session.session_id,
            generation=generation,
            outcome=outcome,
        )
        logger.debug("exec browser scope finish: %s", result)
    except Exception:  # noqa: BLE001 — cleanup must never fail a finished run
        logger.debug("exec browser scope finish failed", exc_info=True)


async def run_session(session: Any, prompt: str, args: Any, team: Any) -> int:
    from local_operator.headless_print import run_print_mode
    from local_operator.session.runtime.exec_control import start_exec_control

    control = None
    try:
        apply_startup(session, args, team)
        control = await start_exec_control(
            session,
            cwd=os.getcwd(),
            yolo=bool(args.yolo),
            supervised=bool(getattr(args, "control", False)),
        )
        lifetime = asyncio.current_task()

        def stop() -> None:
            session.abort("stopped by supervisor")
            if lifetime is not None:
                # Stop the lifetime, not merely the active turn: a loop may be
                # between turns or judging when the supervisor ends the run.
                asyncio.get_running_loop().call_soon(lifetime.cancel)

        control.handle.on_stop_requested = stop
        if getattr(args, "effort", None):
            await control.handle.set_effort(args.effort)
        job_id = getattr(args, "job_id", None)
        if job_id:
            from local_operator.exec_mode import update_job_running

            update_job_running(job_id, session, control)
        print(control.endpoint_line, file=sys.stderr, flush=True)
    except BaseException:
        if control is not None:
            await control.aclose()
        await session.dispose()
        raise

    count = getattr(args, "loop", None)
    goal = getattr(args, "loop_goal", None)

    # Settled as the run proceeds so the browser scope is finalized with the
    # SAME terminal string the ledger records, rather than a second guess at it.
    # `cancelled` is recorded here because only this frame sees the
    # CancelledError; `failed` is read back from the renderer at teardown,
    # because a provider error arrives as an EVENT the renderer folds into its
    # verdict and never reaches this frame at all.
    outcome: dict[str, str] = {"terminal": "succeeded"}

    async def submit(text: str) -> bool:
        completed = await control.handle.run_headless_prompt(text)
        if not completed:
            # A raising turn emits no error event, so the renderer has nothing
            # to print. Report the queue's recorded reason on stderr, where the
            # rest of exec's diagnostics go.
            reason = control.handle.last_prompt_failure or "the turn did not complete"
            print(f"exec failed: {reason}", file=sys.stderr, flush=True)
        return completed

    async def continuation() -> bool:
        try:
            return await control.handle.run_headless_loop(count=count, goal=goal)
        except asyncio.CancelledError:
            outcome["terminal"] = "cancelled"
            raise

    async def close(failed: bool = False) -> None:
        await control.handle.cancel_headless_loop()
        if failed and outcome["terminal"] == "succeeded":
            outcome["terminal"] = "failed"
        # The ledger's own vocabulary, so one run cannot describe its end two
        # different ways to two consumers. Settled by now: the renderer runs
        # this teardown after every turn and the loop have finished.
        await _finish_browser_scope(session, outcome["terminal"])
        await control.aclose()

    return await run_print_mode(
        session,
        [prompt] if prompt.strip() else [],
        json_mode=bool(args.json_mode),
        before_dispose=close,
        prompt_handler=submit,
        continuation=continuation if count is not None or goal is not None else None,
    )
