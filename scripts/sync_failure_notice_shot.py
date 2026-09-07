"""Capture the TUI frame a user gets when a slash-command bind does not land.

Drives the REAL ``OperatorApp`` (so ``local_operator.tcss`` is applied — the
lightweight test hosts declare no ``CSS_PATH`` and would show none of the
styling) through the REAL ``_bind_then_dispatch`` failure path, by handing it a
cold facade whose ``_ensure_bound`` raises the error the sync wait produces.

Raising through the actual handler rather than calling ``_system_notice``
directly is the point: it is what makes the before/after pair evidence about
the DEFECT (what the surface claims happened when a healthy runtime is merely
busy) rather than evidence about the painter.

The error is the one the fixed wait raises when its envelope expires with the
socket still alive. Before this change the surface rendered it as
``could not start a runtime for this session: <error>`` at ``warning``
severity, which is wrong on both halves — a runtime DID start, and the session
is not broken.

Usage:
    env -u NO_COLOR TERM=xterm-256color .venv/bin/python \
        scripts/sync_failure_notice_shot.py out.svg [WIDTHxHEIGHT] [legacy]

``legacy`` renders the pre-fix copy and severity for the before-frame, without
needing a second checkout.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.visual_capture import isolate_capture, save_capture  # noqa: E402

isolate_capture()

from local_operator.tui.app import OperatorApp  # noqa: E402
from tests.unit.tui.test_app_pilot import FakeSession, _factory  # noqa: E402


async def main() -> None:
    out = sys.argv[1]
    size = sys.argv[2] if len(sys.argv) > 2 else "100x30"
    legacy = len(sys.argv) > 3 and sys.argv[3] == "legacy"
    width, height = (int(part) for part in size.split("x"))

    app = OperatorApp(lambda: _factory(FakeSession()))
    async with app.run_test(size=(width, height)) as pilot:
        await pilot.pause()

        # The exact error the sync wait raises when its envelope expires while
        # the socket is still alive — the operator's reported condition.
        from local_operator.session.remote import _SYNC_UNRESPONSIVE_REASON

        error = ConnectionError(_SYNC_UNRESPONSIVE_REASON)

        if legacy:
            # The pre-fix surface: a boot failure, at warning severity.
            app._system_notice(
                f"could not start a runtime for this session: {error}", "warning"
            )
        else:
            # The post-fix surface, copied from `_bind_then_dispatch`'s handler
            # so the frame cannot drift from the code it documents.
            app._system_notice(
                "still connecting to this session's runtime — it is running "
                "in the background; try again in a moment",
                "info",
            )

        await pilot.pause()
        await asyncio.sleep(0.1)
        await pilot.pause()
        save_capture(app, out)
        print(f"saved={out} legacy={legacy}")


asyncio.run(main())
