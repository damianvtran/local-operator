"""Bounded browser-login host for the desktop's authenticated control plane.

Provider flows own OAuth state, PKCE, callbacks and credential persistence. This
host only relays public progress and supplies an ephemeral paste rendezvous; no
access/refresh token or submitted key is ever part of an operation snapshot.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

from local_operator.providers.auth_store import AuthStore
from local_operator.providers.oauth.callback_server import (
    LoginCallbacks,
    LoginTimeoutError,
)
from local_operator.providers.registry import (
    ProviderDefinition,
    get_provider_definition,
)

if TYPE_CHECKING:
    # Imported where a controller is built, not at module scope:
    # ``providers.controller`` pulls the httpx client stack, and this module is
    # on ``lop serve``'s import path through ``routes.auth``, so every server
    # boot paid for an HTTP client that only a provider login or logout uses
    # (backend load report B-F10).
    from local_operator.config import ConfigManager
    from local_operator.providers.controller import ProviderController

logger = logging.getLogger(__name__)

LOGIN_TIMEOUT_S = 900
MAX_OPERATIONS = 32

#: How long ``POST /v1/auth/login`` waits for a flow to publish something the
#: renderer can act on (its URL, its prompt, or a terminal state) before replying.
#:
#: WHY the route waits at all: the flow runs as a task, and the route used to
#: snapshot the operation the instant it was scheduled -- before the task had run
#: one step -- so the first reply was ALWAYS ``state=starting, auth_url=null``.
#: The renderer opens the browser only when that first reply carries a URL, so no
#: browser opened until the user pressed "reopen" (the reported "sign-in doesn't
#: pop up until I click again"). A URL is ready within milliseconds on a working
#: machine: it needs a loopback bind, PKCE, and for a device flow one HTTP round
#: trip. The bound is for the slow case, and it is SHORT because it holds an HTTP
#: request open: past it the route replies anyway and the renderer's poll picks
#: the URL up, which is the old behaviour rather than a failure.
LOGIN_READY_TIMEOUT_S = 3.0

#: How long cancelling a sign-in waits for its flow to finish tearing down.
#:
#: WHY a bound: ``start`` holds its lock across the cancel of the operation it
#: supersedes (so the old flow's loopback port is free before the new one binds),
#: which makes teardown time ADDITIVE -- every queued start waits out the one
#: ahead of it. A real flow tears down in ~50 ms (it closes a loopback server),
#: but nothing in a provider flow promises that, and an unbounded wait meant one
#: teardown that never returned wedged every later sign-in AND ``close`` (review
#: round 2, MINOR 2). Past the bound the old task keeps unwinding on its own and
#: the op already reads ``cancelled``; the new flow's bind ladder then pokes
#: ``/cancel`` at a port it still holds and retries, which is the path a stale
#: sibling login always took.
CANCEL_TEARDOWN_TIMEOUT_S = 2.0

#: The message for a sign-in whose browser path is primary and whose paste prompt
#: is only a fallback (Anthropic, Z.AI). It leads with the browser, because that
#: is what almost everyone does; the old copy led with "Paste the key...", which
#: sent users looking for a key that their flow never asks for.
OPTIONAL_PASTE_MESSAGE = (
    "Finish signing in in your browser. If it shows a code instead of returning "
    "here, paste that code."
)


@dataclass
class LoginOperation:
    id: str
    provider: str
    state: str = "starting"
    message: str = "Starting sign-in."
    auth_url: str | None = None
    instructions: str | None = None
    created_at: float = field(default_factory=time.monotonic)
    task: asyncio.Task[None] | None = field(default=None, repr=False)
    pending_input: asyncio.Future[str | None] | None = field(default=None, repr=False)
    prompt_id: str | None = None
    #: A device flow's code, published as a field so the renderer need not parse
    #: it back out of ``instructions`` ("Enter code: ABCD-1234").
    user_code: str | None = None
    #: The short loopback ``/launch`` alias a callback flow serves beside its
    #: full authorization URL.
    launch_url: str | None = None
    #: True when the open paste prompt is a FALLBACK beside a browser flow that
    #: is still the primary path (``paste_code_flow`` without a required paste).
    #: Fixed per provider at start, so it is also true before the prompt opens.
    input_optional: bool = False
    #: What the successful sign-in wrote to config (``defaults_applied`` in the
    #: snapshot), filled BEFORE ``state`` becomes ``succeeded`` so a poller that
    #: sees success sees it too.
    defaults_applied: dict[str, Any] | None = None
    #: The monotonic time the FLOW itself gives up, which is what ``expires_in``
    #: counts down to. Starts at this host's own cap and is replaced by the
    #: flow's real deadline when the flow reports one: a callback flow times out
    #: at 300 s and a device code lives as long as the provider says, so counting
    #: down 900 s told the user they had ten minutes after the flow had expired.
    deadline: float = field(default_factory=lambda: time.monotonic() + LOGIN_TIMEOUT_S)
    #: Set once the operation has something for the renderer to act on (a URL, a
    #: prompt) or has ended; ``POST /v1/auth/login`` waits on it before replying.
    ready: asyncio.Event = field(default_factory=asyncio.Event, repr=False)

    def snapshot(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "provider": self.provider,
            "state": self.state,
            "message": self.message,
            "auth_url": self.auth_url,
            "launch_url": self.launch_url,
            "user_code": self.user_code,
            "instructions": self.instructions,
            "input_required": self.pending_input is not None and not self.pending_input.done(),
            "input_optional": self.input_optional,
            "prompt_id": self.prompt_id,
            "expires_in": max(0, int(self.deadline - time.monotonic())),
            "defaults_applied": self.defaults_applied,
        }


def apply_desktop_login_defaults(
    manager: ConfigManager | None, provider_id: str, *, oauth: bool
) -> dict[str, Any] | None:
    """Apply the shared login-defaults policy after a desktop sign-in or key save.

    The desktop is the third front end on ``providers.login_defaults`` (after the
    CLI's ``login`` and the TUI's ``/login``). Before this it applied no policy
    at all, so a first-run user who signed in from Settings was left with an
    empty hosting and no model -- the one surface built for first-time setup was
    the one that did not set anything up. The POLICY is not restated here: this
    only reads the config, calls the planner, writes through the shared
    ``apply_login_defaults``, and shapes the result for the wire.

    Returns ``defaults_applied``:
    ``{"hosting", "model", "model_name", "receipt"}``, or ``None`` when an
    existing working choice was left alone (nothing written, nothing to say).
    ``hosting`` is the hosting now in effect when something was written, and
    ``None`` when nothing was -- the decision-only case, which still carries a
    receipt explaining why. ``model_name`` is the display name ("Claude Opus
    5.5"), so the renderer can say what was chosen without a second lookup.

    ``manager`` is re-read first: the TUI, the CLI and other server routes write
    the same ``config.yml`` through their own managers, and planning against a
    stale in-memory copy -- or writing it back whole -- would decide on, and then
    clobber, a choice made elsewhere since this server booted.

    Never raises: the credential is already stored, and a config write failure
    (read-only dir) must not turn a successful sign-in into a failed one.
    """
    if manager is None:
        return None
    try:
        from local_operator.providers.login_defaults import (
            apply_login_defaults,
            plan_login_defaults,
        )

        manager.reload()
        hosting = manager.get_config_value("hosting")
        plan = plan_login_defaults(
            provider_id, hosting, manager.get_config_value("model_name"), oauth=oauth
        )
        wrote = apply_login_defaults(manager, plan)
        if plan.receipt is None:
            return None
        return {
            "hosting": (plan.hosting or hosting or None) if wrote else None,
            "model": plan.model_name or None,
            "model_name": plan.model_label,
            "receipt": plan.receipt,
        }
    except Exception:  # noqa: BLE001 - never fail a completed sign-in
        logger.warning("could not apply login defaults for %s", provider_id, exc_info=True)
        return None


class DesktopAuth:
    def __init__(
        self,
        store: AuthStore,
        config_dir: Any = None,
        config_manager: ConfigManager | None = None,
    ):
        self.store = store
        # The config ROOT ``ProviderController`` resolves its store-first
        # readers under. It used to be a ``CredentialManager``, deleted in PR2b;
        # ``Any`` keeps this module free of a config import and ``None`` means
        # the HOME-derived default.
        self.config_dir = config_dir
        #: The server's own manager, which a successful sign-in writes its
        #: defaults through. ``None`` (a host built without one) skips that step
        #: rather than inventing a second manager on a possibly different root.
        self.config_manager = config_manager
        self.operations: dict[str, LoginOperation] = {}
        # Serialises ``start``'s cancel-then-create. Superseding AWAITS the old
        # flow's teardown (so its loopback port is free before the new flow
        # binds), and that await is a yield point: without the lock, two starts
        # that interleave (a double-click, two windows, a retry fired inside the
        # route's ready wait) both cancel the same old op and both create a flow
        # -- two live flows on one fixed port, the exact state ``start`` exists
        # to prevent. ``close`` takes it too, so a shutdown cannot interleave a
        # start and leave an operation running past ``store.close()``.
        #
        # Created lazily per event loop (see ``_lock``): an ``asyncio.Lock``
        # binds to the loop of its first CONTENDED use and raises on any other,
        # so one built here would break a host driven from a second loop.
        self._start_lock: asyncio.Lock | None = None
        self._start_lock_loop: asyncio.AbstractEventLoop | None = None
        #: Set by ``close``; a start queued behind it must not create a flow on
        #: a store that is already closed.
        self._closed = False
        #: Flows whose teardown outlived ``CANCEL_TEARDOWN_TIMEOUT_S``. Held so
        #: the still-unwinding task is not garbage-collected mid-teardown once
        #: its operation is evicted from ``operations``.
        self._lingering: set[asyncio.Task[None]] = set()

    def _lock(self) -> asyncio.Lock:
        """The start/close lock for the RUNNING loop.

        Production has one loop per app, so this is one lock. The guard is for a
        host exercised from a second loop (a test reusing an app, a second
        client), where a lock bound to the first loop would raise on contention
        (review round 2, NIT 4). Mutual exclusion across loops is meaningless --
        a task on one loop cannot await a lock owned by another -- so a fresh
        lock per loop loses nothing.
        """
        loop = asyncio.get_running_loop()
        if self._start_lock is None or self._start_lock_loop is not loop:
            self._start_lock = asyncio.Lock()
            self._start_lock_loop = loop
        return self._start_lock

    def controller(self) -> ProviderController:
        from local_operator.providers.controller import ProviderController

        return ProviderController(self.store, self.config_dir)

    async def start(self, provider: str) -> LoginOperation:
        definition = get_provider_definition(provider)
        if definition is None or definition.login is None:
            raise ValueError("This provider has no browser sign-in flow.")
        # ONE active operation, still: several providers share one fixed loopback
        # port, and two live flows could race a rotating credential grant. What
        # changed is who wins. A second start used to be REFUSED (409 "A sign-in
        # is already active") -- but the operation holding the slot was almost
        # always one the user had already abandoned (closed the panel, switched
        # provider, or pressed retry after a browser that never opened), so the
        # refusal blocked exactly the retry that was the way out, for up to the
        # flow's 300 s timeout. The newest request is the user's current intent,
        # so it SUPERSEDES the old one, for any provider: the old flow is
        # cancelled (its loopback server is stopped by its own ``finally`` before
        # the new flow binds) and reads ``cancelled`` / "Replaced by a new
        # sign-in." to anyone still polling it. Nothing depended on the 409: the
        # renderer surfaced it as an error string and offered no other path.
        async with self._lock():
            if self._closed:
                # A start queued behind ``close``: its store is gone.
                raise ValueError("Sign-in is unavailable while the server shuts down.")
            # A flow already cancelled and left unwinding past the bound
            # (``_lingering``) is not re-cancelled: it reads ``cancelled``
            # already, and waiting on it again would charge every later start
            # the full bound for a flow nobody can stop.
            for active in [
                op
                for op in self.operations.values()
                if op.task and not op.task.done() and op.task not in self._lingering
            ]:
                await self.cancel(active)
                active.message = "Replaced by a new sign-in."
            while len(self.operations) >= MAX_OPERATIONS:
                del self.operations[next(iter(self.operations))]
            op = LoginOperation(
                id=str(uuid.uuid4()),
                provider=definition.id,
                input_optional=definition.paste_code_flow and not definition.paste_prompt_required,
            )
            self.operations[op.id] = op
            # Created INSIDE the lock, so the next start (queued on it) sees this
            # task as the active op and supersedes it rather than racing it.
            op.task = asyncio.create_task(self._run(op, definition))
            return op

    async def _run(self, op: LoginOperation, definition: ProviderDefinition) -> None:
        def on_url(url: str, instructions: str | None = None) -> None:
            parsed = urlsplit(url)
            if (
                parsed.scheme not in {"http", "https"}
                or not parsed.hostname
                or parsed.username
                or (
                    parsed.scheme == "http"
                    and parsed.hostname not in {"localhost", "127.0.0.1", "::1"}
                )
            ):
                raise ValueError("The provider returned an invalid sign-in URL.")
            op.auth_url = url
            op.instructions = instructions
            op.state = "waiting"
            op.message = "Complete sign-in in your browser."
            op.ready.set()

        def on_flow_details(
            *,
            user_code: str | None = None,
            launch_url: str | None = None,
            expires_in: float | None = None,
        ) -> None:
            # The launch alias is only ever a loopback URL (``_launch_url``), but
            # it is checked like the auth URL anyway: it reaches a browser opener.
            if launch_url:
                parsed = urlsplit(launch_url)
                if parsed.scheme == "http" and parsed.hostname in {"localhost", "127.0.0.1"}:
                    op.launch_url = launch_url
            op.user_code = user_code[:64] if user_code else None
            if expires_in and expires_in > 0:
                op.deadline = min(time.monotonic() + float(expires_in), op.deadline)

        def on_warning(_message: str) -> None:
            # Provider errors can include HTTP bodies or a rejected paste.
            # Relay an actionable state, never those uncontrolled strings.
            op.message = "Sign-in could not use that response. Check it and try again."

        async def on_input() -> str | None:
            pending = asyncio.get_running_loop().create_future()
            op.pending_input = pending
            op.prompt_id = str(uuid.uuid4())
            if op.input_optional:
                # The browser is still the primary path; the prompt is only the
                # fallback. Staying ``waiting`` keeps the renderer on the browser
                # step, and ``input_required``/``input_optional`` tell it a paste
                # box may sit beside it as a secondary action.
                op.state = "waiting"
                op.message = OPTIONAL_PASTE_MESSAGE
            else:
                op.state = "input_required"
                op.message = "Paste the key or sign-in response requested by this provider."
            op.ready.set()
            try:
                return await pending
            finally:
                op.pending_input = None
                op.prompt_id = None
                # Only a PROMPT state goes back to ``waiting``. A flow that ends
                # (timeout, denied callback, success) cancels this prompt task
                # without awaiting it, so this ``finally`` can run AFTER ``_run``
                # has already written the terminal state -- and an unconditional
                # reset turned "expired"/"failed" into a live-looking
                # ``waiting`` the renderer kept polling (QA round 1, Q1).
                if op.state == "input_required":
                    op.state = "waiting"

        callbacks = LoginCallbacks(
            on_auth_url=on_url,
            on_warning=on_warning,
            on_manual_code_input=on_input if definition.accepts_paste_prompt else None,
            on_flow_details=on_flow_details,
        )
        from local_operator.providers.controller import ProviderController

        controller = ProviderController(
            self.store, self.config_dir, login_callbacks=lambda _definition: callbacks
        )
        try:
            async with asyncio.timeout(LOGIN_TIMEOUT_S):
                await controller.login(definition.id, open_browser=lambda _url: None)
            # Off the event loop: it reads and writes config.yml.
            op.defaults_applied = await asyncio.to_thread(
                apply_desktop_login_defaults,
                self.config_manager,
                definition.id,
                oauth=definition.login_kind != "api_key",
            )
            op.state, op.message = "succeeded", "Sign-in complete."
        except asyncio.CancelledError:
            op.state, op.message = "cancelled", "Sign-in cancelled."
        except (TimeoutError, LoginTimeoutError):
            # ``LoginTimeoutError`` is the FLOW's own timeout (no callback within
            # 300 s, or a device code that expired). It subclasses ``LoginError``,
            # not ``TimeoutError``, so it used to fall through to the generic
            # branch below and read "Sign-in failed. Check the provider" -- the
            # wrong remedy for a user who simply took too long.
            op.state, op.message = "expired", "Sign-in expired. Start again when you are ready."
        except Exception:
            op.state, op.message = "failed", "Sign-in failed. Check the provider and try again."
        finally:
            op.auth_url = op.instructions = op.launch_url = op.user_code = None
            op.ready.set()
            controller.close()

    async def cancel(self, op: LoginOperation) -> None:
        if op.task and not op.task.done():
            task = op.task
            task.cancel()
            # Bounded: see ``CANCEL_TEARDOWN_TIMEOUT_S``. ``asyncio.wait`` neither
            # raises on the timeout nor cancels again, so a slow teardown simply
            # keeps running unobserved while this caller moves on.
            await asyncio.wait({task}, timeout=CANCEL_TEARDOWN_TIMEOUT_S)
            if not task.done():
                logger.warning(
                    "sign-in %s for %s did not finish tearing down within %.1fs",
                    op.id,
                    op.provider,
                    CANCEL_TEARDOWN_TIMEOUT_S,
                )
                self._lingering.add(task)
                task.add_done_callback(self._lingering.discard)
            # Cancellation before a coroutine's first step skips its finally.
            op.state, op.message = "cancelled", "Sign-in cancelled."
            op.auth_url = op.instructions = op.launch_url = op.user_code = None
            op.ready.set()

    async def close(self) -> None:
        # Under the start lock (review round 2, MINOR 3): without it a start
        # could create its op between this iteration and the ``clear()``, and
        # that flow kept running past ``store.close()``. ``_closed`` covers the
        # start already queued on the lock, which runs after this releases it.
        async with self._lock():
            self._closed = True
            # Concurrently, so shutdown costs ONE teardown bound, not one per op.
            await asyncio.gather(*(self.cancel(op) for op in list(self.operations.values())))
            self.operations.clear()
            self.store.close()
