"""Local Operator full-screen TUI (Textual).

Import hygiene: ``cli.py`` imports this module ONLY in interactive mode, and
:func:`run_tui` keeps the Textual import inside itself, so headless paths
(exec, server, print mode) never pay the TUI import cost.
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable

from local_operator.session.protocol import SessionProtocol


async def run_tui(
    session_factory: Callable[[], Awaitable[SessionProtocol]],
    theme_name: str = "dark",
    login_handler: Callable[[str], None] | None = None,
    provider_controller: Any | None = None,
) -> int:
    """Run the full-screen TUI to completion; return a process exit code.

    ``session_factory`` is awaited lazily inside a worker so the app paints
    before session construction (providers, skills, MCP discovery). The
    factory shape — rather than a pre-built session — is what lets the app
    own the construction error path and the dispose lifecycle.
    ``login_handler`` routes /login and /logout to the CLI's credential
    flows (legacy one-arg callback); when ``provider_controller`` is given
    the app uses it instead for the full provider/model/usage surface. The
    controller's owning AuthStore is closed by the caller, not the app.
    """
    from local_operator.tui.app import OperatorApp  # lazy: Textual import

    app = OperatorApp(
        session_factory,
        theme_name=theme_name,
        login_handler=login_handler,
        provider_controller=provider_controller,
    )
    try:
        await app.run_async()
        return int(app.return_code or 0)
    except KeyboardInterrupt:
        return 130
