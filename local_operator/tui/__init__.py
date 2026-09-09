"""Local Operator full-screen TUI (Textual).

Import hygiene: ``cli.py`` imports this module ONLY in interactive mode, and
:func:`run_tui` keeps the Textual import inside itself, so headless paths
(exec, server, print mode) never pay the TUI import cost.
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable

from local_operator.logger import file_logging
from local_operator.session.protocol import SessionProtocol


def _register_secret_session(app: Any) -> Any:
    """Register this process with the secret broker; ``None`` when there is none.

    The redaction sink is resolved LAZILY, per notice, rather than captured
    here: the session is constructed inside the app after this runs, so its
    ``VariableStore`` does not exist yet. A notice that arrives before it does
    is dropped rather than acked, which the broker treats as "cannot be
    scrubbed" and denies — the correct direction, since there is genuinely no
    filter to catch that value at that moment (§6, review R3).
    """
    from local_operator.secrets.session import register_session as register

    def on_secret(name: str, value: bytes) -> None:
        session = getattr(app, "_session", None)
        variables = getattr(session, "variables", None) if session is not None else None
        if variables is None:
            raise RuntimeError("no variable store is available to redact through yet")
        variables.register_redaction(value.decode("utf-8", errors="replace"))

    return register(on_secret)


async def run_tui(
    session_factory: Callable[[], Awaitable[SessionProtocol]],
    theme_name: str = "dark",
    provider_controller: Any | None = None,
    resume_factory: Callable[[str | None], Awaitable[SessionProtocol]] | None = None,
    on_config_changed: Callable[[], None] | None = None,
    warm_session_imports: bool = True,
) -> int:
    """Run the full-screen TUI to completion; return a process exit code.

    ``session_factory`` is awaited lazily inside a worker so the app paints
    before session construction (providers, skills, MCP discovery). The
    factory shape — rather than a pre-built session — is what lets the app
    own the construction error path and the dispose lifecycle. Pass
    ``warm_session_imports=False`` only for factories that construct lightweight
    remote facades; local Session factories retain the threaded owner warmup.
    The policy also applies to the supplied resume factory.

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
            on_config_changed=on_config_changed,
            warm_session_imports=warm_session_imports,
        )
        # Register THIS process as a live lop session with the secret broker.
        # Interactive TUI processes are the sessions the broker's ancestry
        # check is about — an agent's `bash` and the eval worker are their
        # descendants — and nothing in shipping code registered one before
        # (QA Q2), which left the broker's session table permanently empty and
        # `lop secret harden` unable to unlock its own store. Registration is
        # deliberately best-effort: `register` returns None when no broker can
        # be started, and the store is an optional capability rather than a
        # boot dependency (§13).
        registration = _register_secret_session(app)
        try:
            await app.run_async()
        except KeyboardInterrupt:
            return 130
        finally:
            # Closing the channel deregisters the session, which is what makes
            # its descendants stop being authorized promptly (§2.1). In the
            # `finally` so an exception path revokes as reliably as a clean
            # exit; the broker's liveness poll is the backstop, not the plan.
            if registration is not None:
                registration.close()
            # AFTER the app has released the terminal, so the line lands in the
            # user's scrollback where it can be copied — printing it from inside
            # the app would put it in a frame that is being torn down. In the
            # `finally` so it survives the exit paths as well as the clean one.
            from local_operator.reexec import REEXEC_CODE

            # A relaunch is about to replace this process; the hint is for a
            # human who is staying in the shell.
            if app.return_code != REEXEC_CODE:
                hint = app.resume_hint()
                if hint:
                    print(f"\nsession ended — resume with:\n  {hint}\n")
        return int(app.return_code or 0)
