"""Capture the TUI frame a user gets when a slash-command bind does not land.

Drives the REAL ``OperatorApp`` (so ``local_operator.tcss`` is applied — the
lightweight test hosts declare no ``CSS_PATH`` and would show none of the
styling) through the REAL ``_bind_then_dispatch`` failure path, by handing it a
facade whose ``_ensure_bound`` raises the error under test.

**The handler is invoked, not imitated.** An earlier version of this script
called ``app._system_notice(...)`` with copy pasted out of the handler, and
that made a whole finding structurally invisible: the notice is only half of
what the user reads, and the other half is the BAND. The handler's
``finally: self._set_starting(False)`` clears the band before the row lands, so
a frame captured from ``_system_notice`` shows a notice next to whatever band
state the capture happened to leave up — and the copy was reviewed as though
something were still in flight when nothing was (design round 1, D1; QA Q2
independently flagged the docstring's claim). Anything that changes what this
frame says has to run through the branch that decides it.

Four modes, one per outcome of that branch, so the whole surface can be seen
side by side:

``sync``      the sync envelope expired with the socket alive — a live runtime
              whose authoritative loop was busy. The reassurance case.
``stopped``   a deliberate ``/stop``. Non-actionable and TRUE, so it relays its
              own reason; the frame that proves the reassurance is gated on the
              sync condition and not on "not actionable".
``actionable``a vetted configuration fault, which keeps its own sentence at
              ``warning``.
``legacy``    the PRE-FIX surface (``could not start a runtime for this
              session: <error>`` at ``warning``), for a before-frame without a
              second checkout. This one cannot go through the handler — that
              code no longer exists — so it is the one mode that paints
              directly, and it is labelled as the historical frame it is.

Usage:
    env -u NO_COLOR TERM=xterm-256color .venv/bin/python \
        scripts/sync_failure_notice_shot.py out.svg [WIDTHxHEIGHT] [MODE]
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.visual_capture import isolate_capture, save_capture  # noqa: E402

isolate_capture()

from local_operator.tui.app import OperatorApp  # noqa: E402
from local_operator.tui.widgets.transcript import NoticeBlock  # noqa: E402
from tests.unit.tui.test_app_pilot import FakeSession, _factory  # noqa: E402


def _error(mode: str) -> BaseException:
    """The exception each mode's branch is selected by."""
    from local_operator.session.remote import (
        _SYNC_UNRESPONSIVE_REASON,
        RuntimeUnresponsiveError,
    )
    from local_operator.session.runtime.launch import ActionableConnectionError

    if mode == "stopped":
        # `_unavailable_reason()` for a facade whose stop was deliberate.
        return ConnectionError("this session was stopped")
    if mode == "actionable":
        return ActionableConnectionError("no API key is configured for openai")
    # The operator's reported condition: socket alive, owner busy, envelope
    # spent. The TYPE is what earns the reassurance.
    return RuntimeUnresponsiveError(_SYNC_UNRESPONSIVE_REASON)


async def main() -> None:
    out = sys.argv[1]
    size = sys.argv[2] if len(sys.argv) > 2 else "100x30"
    mode = sys.argv[3] if len(sys.argv) > 3 else "sync"
    width, height = (int(part) for part in size.split("x"))

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(width, height)) as pilot:
        for _ in range(200):
            await pilot.pause()
            if app._session is not None:
                break

        if mode == "legacy":
            # The pre-fix surface. Painted directly BECAUSE the branch that
            # produced it is gone — this is a historical frame, not a capture
            # of code in the tree, and saying so is the point.
            error = _error("sync")
            app._system_notice(f"could not start a runtime for this session: {error}", "warning")
        else:
            session = app._session
            assert session is not None
            error = _error(mode)

            async def failing_ensure(*args: Any, **kwargs: Any) -> None:
                raise error

            # The one seam stubbed. Everything after it — the branch, the
            # copy, the severity, and the band's `finally` — is production
            # code running in the real worker.
            session._ensure_bound = failing_ensure  # type: ignore[method-assign]
            app._bind_then_dispatch("/credential DEMO_TOKEN")
            for _ in range(400):
                await pilot.pause()
                if any(b.text() for b in app.query(NoticeBlock)):
                    break
                await asyncio.sleep(0.01)

        await pilot.pause()
        await asyncio.sleep(0.1)
        await pilot.pause()
        save_capture(app, out)
        rows = [b.text() for b in app.query(NoticeBlock)]
        # The band is reported alongside the frame because it is the half of
        # the surface a still cannot argue about on its own.
        starting = getattr(app, "_starting_runtime", None)
        print(f"saved={out} mode={mode} band_starting={starting} notices={rows}")


asyncio.run(main())
