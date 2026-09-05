"""Bounded browser-login host for the desktop's authenticated control plane.

Provider flows own OAuth state, PKCE, callbacks and credential persistence. This
host only relays public progress and supplies an ephemeral paste rendezvous; no
access/refresh token or submitted key is ever part of an operation snapshot.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit

from local_operator.providers.auth_store import AuthStore
from local_operator.providers.controller import ProviderController
from local_operator.providers.oauth.callback_server import LoginCallbacks
from local_operator.providers.registry import (
    ProviderDefinition,
    get_provider_definition,
)

LOGIN_TIMEOUT_S = 900
MAX_OPERATIONS = 32


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

    def snapshot(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "provider": self.provider,
            "state": self.state,
            "message": self.message,
            "auth_url": self.auth_url,
            "instructions": self.instructions,
            "input_required": self.pending_input is not None and not self.pending_input.done(),
            "prompt_id": self.prompt_id,
            "expires_in": max(0, int(LOGIN_TIMEOUT_S - (time.monotonic() - self.created_at))),
        }


class DesktopAuth:
    def __init__(self, store: AuthStore, credential_manager: Any = None):
        self.store = store
        self.credential_manager = credential_manager
        self.operations: dict[str, LoginOperation] = {}

    def controller(self) -> ProviderController:
        return ProviderController(self.store, self.credential_manager)

    def start(self, provider: str) -> LoginOperation:
        definition = get_provider_definition(provider)
        if definition is None or definition.login is None:
            raise ValueError("This provider has no browser sign-in flow.")
        # Several providers share one fixed loopback port. A single operation
        # also prevents two UI windows from racing a rotating credential grant.
        if any(op.task and not op.task.done() for op in self.operations.values()):
            raise RuntimeError("A sign-in is already active. Finish or cancel it first.")
        while len(self.operations) >= MAX_OPERATIONS:
            del self.operations[next(iter(self.operations))]
        op = LoginOperation(id=str(uuid.uuid4()), provider=definition.id)
        self.operations[op.id] = op
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

        def on_warning(_message: str) -> None:
            # Provider errors can include HTTP bodies or a rejected paste.
            # Relay an actionable state, never those uncontrolled strings.
            op.message = "Sign-in could not use that response. Check it and try again."

        async def on_input() -> str | None:
            pending = asyncio.get_running_loop().create_future()
            op.pending_input = pending
            op.prompt_id = str(uuid.uuid4())
            op.state = "input_required"
            op.message = "Paste the key or sign-in response requested by this provider."
            try:
                return await pending
            finally:
                op.pending_input = None
                op.prompt_id = None
                op.state = "waiting"

        callbacks = LoginCallbacks(
            on_auth_url=on_url,
            on_warning=on_warning,
            on_manual_code_input=on_input if definition.accepts_paste_prompt else None,
        )
        controller = ProviderController(
            self.store, self.credential_manager, login_callbacks=lambda _definition: callbacks
        )
        try:
            async with asyncio.timeout(LOGIN_TIMEOUT_S):
                await controller.login(definition.id, open_browser=lambda _url: None)
            op.state, op.message = "succeeded", "Sign-in complete."
        except asyncio.CancelledError:
            op.state, op.message = "cancelled", "Sign-in cancelled."
        except TimeoutError:
            op.state, op.message = "expired", "Sign-in expired. Start again when you are ready."
        except Exception:
            op.state, op.message = "failed", "Sign-in failed. Check the provider and try again."
        finally:
            op.auth_url = op.instructions = None
            controller.close()

    async def cancel(self, op: LoginOperation) -> None:
        if op.task and not op.task.done():
            op.task.cancel()
            await asyncio.gather(op.task, return_exceptions=True)
            # Cancellation before a coroutine's first step skips its finally.
            op.state, op.message = "cancelled", "Sign-in cancelled."
            op.auth_url = op.instructions = None

    async def close(self) -> None:
        for op in list(self.operations.values()):
            await self.cancel(op)
        self.operations.clear()
        self.store.close()
