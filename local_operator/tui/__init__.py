"""Local Operator full-screen TUI (Textual).

Import hygiene: ``cli.py`` imports this module ONLY in interactive mode, and
:func:`run_tui` keeps the Textual import inside itself, so headless paths
(exec, server, print mode) never pay the TUI import cost.
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable

from local_operator.logger import file_logging
from local_operator.session.protocol import SessionProtocol


async def run_tui(
    session_factory: Callable[[], Awaitable[SessionProtocol]],
    theme_name: str = "dark",
    provider_controller: Any | None = None,
    resume_factory: Callable[[str], Awaitable[SessionProtocol]] | None = None,
) -> int:
    """Run the full-screen TUI to completion; return a process exit code.

    ``session_factory`` is awaited lazily inside a worker so the app paints
    before session construction (providers, skills, MCP discovery). The
    factory shape — rather than a pre-built session — is what lets the app
    own the construction error path and the dispose lifecycle.

    Everything runs inside :func:`~local_operator.logger.file_logging`: while
    Textual owns the terminal, a log record on stderr is painted straight over
    the frame and stays there until the next full repaint. The context manager
    detaches every console handler, sends records to a bounded rotating file
    instead, and restores the handlers on the way out so the plain REPL and
    ``exec`` — which can run in this same process — are unaffected. It is the
    OUTERMOST thing here on purpose: session construction is the noisiest part
    of startup (provider probes, MCP discovery) and it happens inside the app.
    """
    from local_operator.tui.app import OperatorApp  # lazy: Textual import

    with file_logging():
        app = OperatorApp(
            session_factory,
            theme_name=theme_name,
            provider_controller=provider_controller,
            resume_factory=resume_factory,
        )
        try:
            await app.run_async()
        except KeyboardInterrupt:
            return 130
        finally:
            # AFTER the app has released the terminal, so the line lands in the
            # user's scrollback where it can be copied — printing it from inside
            # the app would put it in a frame that is being torn down. In the
            # `finally` so it survives the exit paths as well as the clean one.
            hint = app.resume_hint()
            if hint:
                print(f"\nsession ended — resume with:\n  {hint}\n")
        return int(app.return_code or 0)
