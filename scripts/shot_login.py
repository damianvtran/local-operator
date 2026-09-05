"""Capture a real ``/login <paste-key provider>`` frame from the real OperatorApp.

Evidence harness for the paste-key login prompt, not test infrastructure: it
drives ``OperatorApp`` (the host that actually loads ``local_operator.tcss`` —
the lightweight hosts in ``tests/unit/tui`` do not, so a still from one of them
cannot show a stylesheet change at all) through a genuine ``/login alibaba``
and exports what the terminal painted.

Usage:
    env -u NO_COLOR TERM=xterm-256color .venv/bin/python \
        scripts/shot_login.py out.svg [provider] [keys]

``keys`` is a comma-separated list of Textual key NAMES (``s,k,minus,enter``),
not a string of characters: ``enter`` and ``escape`` have no character form, so
a frame of the settled receipt cannot be captured by pressing characters alone.

The provider controller is the REAL one over a throwaway AuthStore in a temp
directory, so the registry entry, the login thunk and the callbacks under test
are the shipped ones. Nothing here may reach the network: the paste-key logins
only open a browser URL and read a key, and ``webbrowser.open`` is stubbed out
so a capture run cannot fling a browser window at whoever is running it.
"""

from __future__ import annotations

import asyncio
import sys
import tempfile
import webbrowser
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.visual_capture import isolate_capture, save_capture  # noqa: E402

isolate_capture()

from local_operator.providers.auth_store import AuthStore  # noqa: E402
from local_operator.providers.controller import ProviderController  # noqa: E402
from local_operator.tui.app import OperatorApp  # noqa: E402
from tests.unit.tui.test_app_pilot import FakeSession, _factory  # noqa: E402


async def main() -> None:
    out = sys.argv[1]
    provider = sys.argv[2] if len(sys.argv) > 2 else "alibaba"
    keys = [k for k in (sys.argv[3].split(",") if len(sys.argv) > 3 else []) if k]

    webbrowser.open = lambda *_args, **_kwargs: True  # type: ignore[assignment]

    with tempfile.TemporaryDirectory() as tmp:
        controller = ProviderController(AuthStore(Path(tmp) / "auth.db"))
        app = OperatorApp(lambda: _factory(FakeSession()), provider_controller=controller)
        async with app.run_test(size=(100, 30)) as pilot:
            await pilot.pause()
            from local_operator.tui.widgets.editor import Editor

            app.query_one(Editor).text = f"/login {provider}"
            await pilot.press("enter")
            # Two settles: the login worker has to start, reach the prompt and
            # paint it. A single pause captures the frame before the flow runs.
            for _ in range(20):
                await pilot.pause()
                await asyncio.sleep(0.05)
            for key in keys:
                await pilot.press(key)
            # Settle after the keys: submitting resolves the login's future, and
            # the receipt is painted by the flow that wakes on it, not by the
            # keystroke — a single pause captures the frame before it lands.
            for _ in range(10):
                await pilot.pause()
                await asyncio.sleep(0.05)
            save_capture(app, out)
            print(f"wrote {out}")


asyncio.run(main())
