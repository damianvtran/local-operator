"""Local Operator full-screen TUI (Textual).

Import hygiene: ``cli.py`` imports this module ONLY in interactive mode, and
:func:`run_tui` keeps the Textual import inside itself, so headless paths
(exec, server, print mode) never pay the TUI import cost.
"""

from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable

from local_operator.logger import file_logging
from local_operator.session.protocol import SessionProtocol

logger = logging.getLogger(__name__)


def _register_secret_session(app: Any) -> Any:
    """Register this process with the secret broker; ``None`` when there is none.

    **This covers the TUI only when the TUI OWNS the session.** With an
    in-process ``Session`` this process holds the ``VariableStore`` the §6
    notice has to reach, so this registration is the one that answers it. An
    ATTACHED TUI — the interactive default, where ``app._session`` is an
    ``AttachedSession`` facade over a separate runtime process — has no store
    at all, and the registration that matters there is the RUNTIME's own
    (``ServingSessionHandle``), made in the process whose bash and eval
    redactors read the store. Do not "fix" the sink below by reaching for the
    viewer's session state looking for a store: there is nothing to reach.

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
            # The LAST-RESORT fail-closed path, not the primary one. An attached
            # viewer has no store to redact through, so the runtime's own
            # registration is what should have answered this notice; reach here
            # only when there is genuinely no filter in this process. Logged
            # because a silent denial is invisible in the agent's transcript.
            logger.warning(
                "%r could not be registered for redaction: no variable store in this "
                "process, so the retrieval is denied rather than served unscrubbed",
                name,
            )
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
        # It covers the session the TUI OWNS: an in-process session's `bash`
        # and eval workers are this process's descendants, and its
        # `VariableStore` is here, so this registration is the one the §6
        # notice must reach (QA Q2 — nothing in shipping code registered a
        # session before, which left the broker's table permanently empty and
        # `lop secret harden` unable to unlock its own store). An ATTACHED TUI
        # is covered by the RUNTIME's registration instead (serving.py), since
        # its `app._session` facade holds no store this sink could scrub
        # through. Registration is deliberately best-effort: `register` returns
        # None when no broker can be started, and the store is an optional
        # capability rather than a boot dependency (§13).
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
