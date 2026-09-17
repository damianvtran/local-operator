"""F1 regression: a raising viewer_watch must not take the app down."""

import asyncio
import logging

import pytest
from textual.app import App


@pytest.mark.asyncio
async def test_bare_run_worker_crashes_but_guarded_body_does_not():
    """Pins the mechanism F1 identified, using the two call shapes directly."""

    class Boom(App[None]):
        def __init__(self):
            super().__init__()
            self.died = None

        def _handle_exception(self, error):  # what WorkerFailed reaches
            self.died = error
            self.exit()

    async def raiser():
        raise RuntimeError("unknown op")

    # BEFORE shape: bare coroutine, default exit_on_error=True
    app = Boom()
    async with app.run_test() as pilot:
        try:
            app.run_worker(raiser(), exclusive=False)
        except Exception:
            pass
        await pilot.pause()
        await asyncio.sleep(0.2)
        await pilot.pause()
        before_died = app.died is not None

    # AFTER shape: body guarded + exit_on_error=False
    app2 = Boom()
    async with app2.run_test() as pilot:

        async def guarded():
            try:
                await raiser()
            except Exception:
                logging.getLogger(__name__).debug("swallowed", exc_info=True)

        try:
            app2.run_worker(guarded(), exclusive=False, exit_on_error=False)
        except Exception:
            pass
        await pilot.pause()
        await asyncio.sleep(0.2)
        await pilot.pause()
        after_died = app2.died is not None

    print(f"\n  bare run_worker      -> app crashed: {before_died}")
    print(f"  guarded + no-exit    -> app crashed: {after_died}")
    assert before_died, "expected the unguarded shape to reach _handle_exception"
    assert not after_died, "the fix must not let the failure reach the app"
