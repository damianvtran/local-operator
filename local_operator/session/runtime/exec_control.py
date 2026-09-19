"""Discovery and optional supervised gates for a headless ``lop exec`` run.

``exec`` is the harness's machine-driven entry point: a supervisor composes a
prompt, runs one turn, and parses NDJSON off stdout. Until this module that run
was also UNREACHABLE — it published no discovery record and served no control
socket — so a supervisor watching a sentinel drift had only process signals to
answer with, and a signal cannot say *"stop after this ``git push`` finishes"*.

This is the composition root that closes the gap, and it is deliberately the
SAME one the daemon's phone-started sessions use: an
:class:`~local_operator.session.runtime.serving.ServingSessionHandle` over the exec
session, wrapped in a :class:`~local_operator.session.runtime.server.RuntimeServer`
that publishes the record and serves the authenticated loopback socket. Nothing
here is a third ``SessionHandle`` implementation — the whole control vocabulary
(``steer``, ``abort``, ``cancel``, ``set_model``, ``set_effort``,
``approval_answer``, ``ask_answer``, ``stop``) is the one the phone daemon and
``lop attach`` already speak, so a supervisor written against either drives an
exec run with no new client.

Exec now publishes every ordinary session so a background team is discoverable
and attachable through the same TUI path as a terminal-owned conversation.
Short-lived records are deliberately ephemeral; durable outcomes belong to the
exec ledger, not discovery. Publication and gate installation are separate:
``supervised=False`` leaves the original headless gate untouched. Imports remain
function-local on the CLI path so parsing/help does not load the runtime stack.

**What ``--control`` changes about the run itself.** The owned handle installs
its own approval/ask gates (``ServingSessionHandle._install_gates``), replacing
the CLI's headless gate. A tool approval therefore PARKS for an attached
supervisor to answer — up to
:data:`~local_operator.session.runtime.serving.PENDING_REQUEST_TIMEOUT_S` — where
an ordinary headless exec denies instantly for want of a tty. That is the point
of the surface (a supervisor can now answer), but it is a real behavioural
difference, so it is opt-in with the flag and ``--yolo`` still short-circuits
it via ``auto_approve``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Awaitable, Callable, cast

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    from local_operator.session.runtime.server import RuntimeServer
    from local_operator.session.runtime.serving import ServingSessionHandle

logger = logging.getLogger(__name__)

#: The record's ``kind``. ``SessionRecord.kind`` already admitted ``"exec"``
#: before anything constructed one — this is the first producer of it, and it
#: is what lets a reader of ``lop sessions --json`` tell a supervised one-shot
#: apart from a terminal a human is sitting at.
EXEC_RECORD_KIND = "exec"


@dataclass
class ExecControl:
    """A live control surface bound to one foreground exec run.

    Held by the caller for exactly as long as the run: built before the prompt
    (so a supervisor can attach to the very first tool call) and closed in the
    run's teardown BEFORE the session is disposed — see :meth:`aclose` for why
    that ordering is load-bearing rather than tidy.
    """

    handle: "ServingSessionHandle"
    runtime: "RuntimeServer"
    session_id: str
    pid: int
    port: int
    record_path: str
    #: Whether this run's approval gates were actually replaced by the
    #: supervisor's (``exec --control``). Discovery is now published for every
    #: run, so the mode is no longer implied by the surface existing, and
    #: :attr:`endpoint_line` has to say which one the user is in.
    supervised: bool = True

    @property
    def endpoint_line(self) -> str:
        """The one line the run prints so a supervisor can find this session.

        Goes to STDERR at every call site, never stdout: stdout is the
        machine-readable payload stream (``--json``'s NDJSON, or the final
        assistant text), and a chrome line in it corrupts the only output the
        run has. Same rule ``headless_print`` states for progress chrome.

        The control KEY is deliberately absent. It lives in the record, mode
        0600 under a 0700 directory, and those permissions are the entire
        authorization model (see :mod:`.registry`): anything that can read the
        key is already the owning account. Printing it into a supervisor's log
        would move the credential somewhere the permissions do not reach, so
        the line names the record instead and the supervisor reads it there.

        The NOUN reports the mode, because the two are no longer the same
        thing. Publication was separated from gate installation, so this line
        is printed on every run — and printing ``control:`` for a run without
        ``--control`` tells the user, in the product's own vocabulary (it is
        the exact name of the flag they did not pass), that the supervised
        gate is installed. It is not: an unsupervised run keeps the headless
        deny gate. ``session:`` names what is actually true of every run, and
        ``control:`` is kept for the gated case a supervisor greps for.
        """
        label = "control" if self.supervised else "session"
        return (
            f"lop exec {label}: session_id={self.session_id} pid={self.pid} "
            f"port={self.port} record={self.record_path}"
        )

    async def aclose(self) -> None:
        """Announce the deliberate end, then tear the surface down.

        MUST run before the session is disposed. Two reasons, both observed
        rather than theoretical:

        1. ``announce_stop`` is what tells an attached supervisor the socket
           close it is about to see is a finished run and not a dropped
           connection — the same distinction ``RuntimeServer.announce_stop``
           documents for the TUI's ``/stop``. A supervisor that cannot make it
           has to treat every clean exit as a crash and retry the prompt.
        2. The runtime reads through the handle into the session on every
           heartbeat and every push. Disposing the session first leaves those
           reads racing a torn-down session for as long as teardown takes.

        Non-raising by contract: this runs in the run's ``finally``, and a
        control surface that fails to close must not change the run's exit
        code, which is the supervisor's actual result.
        """
        try:
            self.runtime.announce_stop()
        except Exception:  # noqa: BLE001 — announcing is best-effort courtesy
            logger.debug("exec control: stop announcement failed", exc_info=True)
        try:
            # ``aclose_remote`` rather than ``aclose``, and the difference is the
            # whole reason this call site is in the change at all: ``aclose``
            # raises off the runtime's owning loop by design, and this teardown
            # runs on the SESSION's loop — which stopped being the runtime's loop
            # when the exec surface moved to ``start()``. The raise was wrapped,
            # so it was silent: teardown STARTED (``aclose`` requests the close
            # before raising) and nothing waited for it. Best-effort either way,
            # but the two are not the same best-effort.
            remote = getattr(self.runtime, "aclose_remote", None)
            if callable(remote):
                await cast(Callable[[], Awaitable[None]], remote)()
            else:  # pragma: no cover - a reduced host answering only the owner-loop form
                await self.runtime.aclose()
        except Exception:  # noqa: BLE001 — teardown must not fail the run
            logger.warning("exec control: runtime shutdown failed", exc_info=True)
        # Deregister the handle's secret session as part of the same teardown.
        # ``ServingSessionHandle.dispose`` would do it, but this path never
        # disposes the handle — the run's owner disposes the session and the
        # process exits — so without this the socket-close backstop would be the
        # only revocation, and a supervisor that reuses the process would leave
        # descendants of a finished agent authorized (§2.1).
        try:
            self.handle.close_secret_registration()
        except Exception:  # noqa: BLE001 — teardown must not fail the run
            logger.debug("exec control: secret deregistration failed", exc_info=True)


async def start_exec_control(
    session: Any,
    *,
    cwd: str,
    yolo: bool = False,
    supervised: bool = True,
) -> ExecControl:
    """Publish a record and serve the control socket for ``session``.

    ``start()``, NOT ``start_in_process`` — reversed deliberately, and this is
    the call site where the reversal is a trade rather than a free win. The
    earlier reading was: the exec session already lives on the caller's running
    loop, so a second loop on its own thread (what the TUI needs, because
    Textual owns its loop) forces every control request through a cross-thread
    hop for no benefit. The benefit is real and measured, though: in process,
    the listener, the welcome, ``ping`` and the heartbeat share the workload's
    loop, so one synchronous step of a turn parks the entire control plane at
    once. Measured on the audit's rig under a 50 s block with NO client
    attached: the record crossed into ``wedged`` at t=46.2 s and a fresh dial
    received no welcome within 15 s; with the serving plane on its own thread,
    welcome immediate, ``ping`` -> ``pong`` in 0.00 s, heartbeat never past
    14.2 s. A supervisor watching an ``lop exec --control`` run is exactly the
    surface that misreads the in-process shape as a dead run.

    The hop is not free and is not pretended to be: the registrations this
    surface needs are performed on the session's loop (see
    ``RuntimeServer._handle_call_on_session_loop``), so a cross-thread hop that
    the earlier reading called "for no benefit" is what buys a control plane
    the turn cannot park.

    ``yolo`` maps to the handle's ``auto_approve``, so ``exec --control --yolo``
    keeps approving every tier inline instead of parking a card no supervisor
    may be watching. Without it the gates park for an attached supervisor —
    the behavioural difference this module's header calls out. The flag is
    also the handle's ``approval_pinned``: an explicit ``--yolo`` on this run
    outranks a later ``tool_approval_mode`` edit, while an un-flagged run
    follows the file like every other runtime gate.

    Imports are function-local by contract, not by habit: ``serving`` pulls the
    composition root and ``server`` pulls asyncio, and this package sits on the
    CLI startup path (see the module header and :mod:`.serving`).
    """
    import asyncio

    from local_operator.paths import config_dir
    from local_operator.session.runtime.server import RuntimeServer
    from local_operator.session.runtime.serving import (
        ServingSessionHandle,
        attach_gate_config_watch,
    )

    loop = asyncio.get_running_loop()
    config_directory = config_dir()
    handle = ServingSessionHandle(
        session,
        loop,
        cwd=cwd,
        auto_approve=yolo,
        approval_pinned=yolo,
        install_gates=supervised,
        # Declared so the §6 registration's store-existence check and the
        # registration itself share this run's root (MINOR-3).
        config_dir=config_directory,
    )
    if supervised:
        attach_gate_config_watch(handle, config_directory)
    # The ``stop`` control op (and therefore `lop stop`, which can now see this
    # run because it publishes a record) reaches ``request_stop`` -> this hook.
    # Without one the handle falls back to disposing in place, UNDER the prompt
    # that is still running — a torn session mid-turn. Aborting instead ends
    # the turn, so the run returns through its own teardown and the surface is
    # closed and unpublished in the ordering :meth:`ExecControl.aclose` owns.
    handle.on_stop_requested = lambda: session.abort("stopped by supervisor")
    runtime = RuntimeServer(handle, kind=EXEC_RECORD_KIND)
    runtime.start()
    # THIS WAIT IS THE ONE C2 EXISTS FOR. The two fields read below — the
    # listener's port and the record path a supervisor is handed — are stamped
    # on the runtime's own thread, so reading them straight after ``start()``
    # reads the constructor's ``port=0`` and may name a file nothing has written
    # yet: an endpoint line the supervisor cannot use, for the one purpose that
    # line has. A bind that failed releases the latch too, and here that is a
    # FATAL answer rather than a degraded one — ``maybe_start_exec_control``
    # documents the contract: continuing without the surface hands the
    # supervisor an agent it cannot stop while reporting success.
    if not await runtime.wait_until_published():
        # CLOSE WHAT WE STARTED BEFORE RAISING. ``start()`` has already put a
        # thread, a loop and (on the timeout path) a still-running boot prologue
        # behind the caller who is about to fail this run; raising alone leaves
        # all three behind a decision that says the surface is unusable, which is
        # the opposite of what that answer means.
        #
        # ``aclose_remote`` rather than ``close``, and the difference is this
        # call site's whole problem: this runs on the SESSION's loop, and
        # ``close`` is a bounded SYNCHRONOUS join — a cross-thread wait on the
        # loop every other part of this change exists to keep free (review round
        # 1, D-5). ``aclose_remote`` awaits the same bounded join through a
        # thread hop, which is the spelling ``process._clean_exit`` already uses
        # for the same teardown.
        #
        # THE FALLBACK IS THE SAME SHAPE AS THE THREE SIBLING SITES (review
        # round 2, MINOR-2). Calling ``aclose_remote`` directly made a reduced
        # host — one answering only the owner-loop ``aclose`` form — raise
        # ``AttributeError`` out of a failure path, REPLACING the RuntimeError
        # this contract is written on, so the supervisor would be told an
        # attribute is missing instead of that the surface never published.
        remote = getattr(runtime, "aclose_remote", None)
        if callable(remote):
            await cast(Callable[[], Awaitable[None]], remote)()
        else:  # pragma: no cover - a reduced host answering only the owner-loop form
            await runtime.aclose()
        raise RuntimeError(
            "the exec control surface was asked for but the runtime never "
            "published its record; the run cannot be supervised"
        )
    record = runtime.record
    return ExecControl(
        handle=handle,
        runtime=runtime,
        session_id=record.session_id,
        pid=record.pid,
        port=record.control_port,
        # The file this run ACTUALLY published, read off the runtime rather than
        # recomputed from a second `config_dir()` read: the record's directory
        # is fixed when the runtime starts, but `_serve` yields at
        # `asyncio.start_server` before it builds the publisher, so a config dir
        # that moves during that window makes a recomputed path name a
        # `<pid>.json` no runtime wrote — and this path is the correlation
        # handle a supervisor is handed (QA round 2, Q2).
        record_path=str(runtime.record_path),
        supervised=supervised,
    )


async def maybe_start_exec_control(
    session: Any,
    *,
    enabled: bool,
    cwd: str,
    yolo: bool = False,
) -> ExecControl | None:
    """Start the surface when asked, disposing the session if it cannot start.

    Shared by both headless entry points — the foreground ``exec`` runner and
    the detached ``exec_worker`` — so ``--control`` means exactly one thing on
    either side of the ``--background`` process boundary.

    A failure here is FATAL rather than degraded, and that is the whole reason
    this wrapper exists. ``--control`` is a supervisor saying "I need to be able
    to steer and cancel this run"; continuing without the surface would give it
    an agent it cannot stop, while reporting success. The session is disposed
    first because it is already built by this point, and re-raising past a live
    session would leak its claim and its provider connections.
    """
    if not enabled:
        return None
    try:
        return await start_exec_control(session, cwd=cwd, yolo=yolo)
    except BaseException:
        try:
            await session.dispose()
        except Exception:  # noqa: BLE001 — the start failure is the real error
            logger.debug("exec control: dispose after failed start failed", exc_info=True)
        raise
