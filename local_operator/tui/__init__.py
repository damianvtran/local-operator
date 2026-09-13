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
from local_operator.tui.terminal_modes import (
    guard_pixel_mouse_latch,
    reset_in_band_resize,
)

logger = logging.getLogger(__name__)


def _register_secret_session(app: Any) -> Any:
    """Register this process with the secret broker; ``None`` when there is none.

    **This process is registered in every shape, and that is deliberate.** The
    §6 notice reaches the nearest REGISTERED ancestor of the retrieving child,
    which in the attached architecture is normally the runtime that owns the
    value's filter (``ServingSessionHandle``). But a runtime that booted before
    a store existed at its root registers nothing (§13), and then this viewer's
    entry is the only ancestor the broker can notify. Closing this channel
    instead would leave no registered ancestor at all: the broker refuses the
    retrieval for lacking ancestry, ``access.retrieve_secret`` then falls
    through to its keyfile-tier local decrypt (``except BrokerDenied: if
    hardened: raise``), and the value lands in the transcript with nothing
    having been notified — the very leak §6 exists to prevent. So the entry is
    kept, and the sink below makes it do real work rather than being a slot the
    broker wastes.

    The sink resolves the redaction target LAZILY, per notice, in order:

    1. ``app._session.variables`` — the TUI OWNS the session, so this process
       holds the store the bash and eval redactors read and answers directly.
    2. a forward to the attached runtime (``register_secret_redaction``) over
       the same viewer→runtime control channel ``/credential store`` already
       uses, so the value reaches the process that owns the filter.
    3. RAISE — there is genuinely no filter anywhere this process can reach, so
       the notice is not acknowledged and the broker fails closed.

    The target is resolved per notice rather than captured here because the
    session is constructed inside the app after this runs, and /new, /resume
    and /reload replace it; a captured store would be stale.

    **Budget (review round 1, MAJOR-1).** Registered entries are one per TUI
    window (this one, made once at boot for the process's whole life) plus one
    per runtime that HAS a store at its root — a no-store runtime does not
    register at all, so a machine full of store-less sessions adds nothing.
    Against the broker's ``MAX_SESSIONS`` (32) that is one entry per window and
    per store-owning runtime rather than the unbounded per-retrieval growth the
    cap was added for; the viewer entry is no longer inert, which is why it is
    kept rather than closed. It is registered unconditionally (no store check,
    unlike the runtime path), because gating it on a store existing at boot
    loses the entry in exactly the case it exists for — a store created after
    the runtime booted — and then NO ancestor registers: the broker refuses for
    lack of ancestry and the keyfile fallback serves the value unnotified. A
    store-less window therefore pays one broker start and keeps a fail-closed
    entry, and that is the deliberate trade.
    """
    from local_operator.secrets.session import register_session as register

    def on_secret(name: str, value: bytes) -> None:
        text = value.decode("utf-8", errors="replace")
        session = getattr(app, "_session", None)
        variables = getattr(session, "variables", None) if session is not None else None
        if variables is not None:
            # (1) In-process session: the store is here, so the notice is
            # answered here. Unchanged behaviour.
            variables.register_redaction(text)
            return
        forward = (
            getattr(session, "register_secret_redaction", None) if session is not None else None
        )
        if forward is not None:
            # (2) Attached session: hand the value to the runtime that owns the
            # filter. Raises on any failure, which is the fail-closed direction.
            _forward_redaction_to_runtime(app, forward, text)
            return
        # (3) The LAST-RESORT fail-closed path. Logged because a silent denial
        # is invisible in the agent's transcript; the raise is what stops
        # ``SessionRegistration._read_loop`` from acknowledging, so the broker
        # denies rather than serving a value nothing can scrub.
        logger.warning(
            "%r could not be registered for redaction: no variable store in this "
            "process and no runtime to forward to, so the retrieval is denied "
            "rather than served unscrubbed",
            name,
        )
        raise RuntimeError("no variable store is available to redact through yet")

    return register(on_secret)


def _forward_redaction_to_runtime(app: Any, forward: Any, value: str) -> None:
    """Hand one §6 value to the attached runtime, bounded; raise on failure.

    **Why the hop.** The broker's notice arrives on
    ``SessionRegistration._read_loop``, a plain thread; the viewer's RPC to the
    runtime runs on the app's event loop, so the call has to hop there.

    **Why it is bounded on THIS side.** The broker denies the retrieval when no
    acknowledgement arrives within ``NOTIFY_ACK_TIMEOUT_S`` (2 s). A
    ``call_from_thread``-style hop blocks the notice thread with no timeout, so
    a wedged loop would park here past that window and turn a fail-closed
    denial into an ambiguous hang. Instead the coroutine is scheduled with
    ``run_coroutine_threadsafe`` and the thread waits with an explicit timeout
    (the transport's own ``REDACTION_FORWARD_TIMEOUT_S``, well under 2 s), so
    the hop gives up, the sink raises, and the broker denies — fail closed.

    **Why no Textual context is needed.** The forwarded coroutine touches only
    the attach client's socket and its store; it never reads or mutates widgets,
    so it does not need ``App._context()`` that ``call_from_thread`` would set.
    """
    import asyncio
    from concurrent.futures import TimeoutError as FuturesTimeoutError

    from local_operator.mobile.attach_client import REDACTION_FORWARD_TIMEOUT_S

    loop = getattr(app, "_loop", None)
    if loop is None:
        # The app has not started its loop yet (a notice before the first turn,
        # or a session constructed in the gap). There is nothing to forward to.
        raise RuntimeError(
            "the TUI is not running yet, so no attached runtime can be asked to redact"
        )
    try:
        future = asyncio.run_coroutine_threadsafe(forward(value), loop)
    except RuntimeError as exc:  # loop closed, or scheduling refused
        raise RuntimeError(
            f"the attached runtime cannot be reached to register this redaction: {exc}"
        ) from exc
    try:
        future.result(timeout=REDACTION_FORWARD_TIMEOUT_S)
    except FuturesTimeoutError as exc:
        future.cancel()
        raise RuntimeError(
            "the attached runtime did not confirm the redaction within the forward "
            "budget, so the value cannot be kept out of the transcript"
        ) from exc


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
    # Opt out of in-band window resize BEFORE Textual negotiates it. Both
    # calls are load-bearing: the reset must land on the wire before the
    # driver queries `?2048$p` (linux_driver.py:299), and the env guard must
    # be set before `textual.constants` is imported (SMOOTH_SCROLL is a Final,
    # read once). Together they stop the terminal sending resize reports and
    # stop Textual re-enabling the mode after seeing our reset. See
    # `terminal_modes` for why neither half suffices alone.
    reset_in_band_resize()
    guard_pixel_mouse_latch()

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
        # keeps the same registration as the broker's fallback for a runtime
        # that registered nothing (it booted before a store existed), and its
        # sink FORWARDS the notice to that runtime rather than being inert —
        # see `_register_secret_session`. Registration is deliberately
        # best-effort: `register` returns None when no broker can be started,
        # and the store is an optional capability rather than a boot dependency
        # (§13).
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
