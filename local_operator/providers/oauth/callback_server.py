"""Loopback OAuth callback server shared by all authorization-code flows.

Ported from omp ``registry/oauth/callback-server.ts``. Invariants worth
preserving (they are scar tissue from real provider behaviour):

- The HTTP server starts BEFORE the auth URL is generated so the actually
  bound port lands in ``redirect_uri`` (the port-0 fallback changes it).
- When the provider validates redirect URIs (``redirect_uri`` pinned or
  ``allow_port_fallback=False``), a busy port MUST fail before the browser
  opens — otherwise the user gets an opaque 500 at the IdP and a 5-minute
  hang locally.
- Two routes on one server: the callback path and ``/launch`` (302 to the
  pending auth URL) so a TUI can hand the user a short copy target.
- The paste-code prompt may only race the HTTP callback for providers that
  declare ``paste_code_flow`` — for the rest it deadlocks terminals.
"""

from __future__ import annotations

import asyncio
import dataclasses
import secrets
import urllib.parse
import webbrowser
from abc import ABC, abstractmethod
from typing import Any, Awaitable, Callable

from local_operator.harness.types import AbortSignal

DEFAULT_TIMEOUT_SECONDS = 300.0


class LoginError(Exception):
    """Base error for interactive login failures."""


class LoginCancelledError(LoginError):
    """The user (or an abort signal) cancelled the login."""

    def __init__(self, message: str = "Login cancelled") -> None:
        super().__init__(message)


class LoginTimeoutError(LoginError):
    """No callback/manual code arrived within the timeout window."""

    def __init__(self, message: str | None = None) -> None:
        super().__init__(
            message
            or "Timed out waiting for the login callback. If you run inside WSL/a VM, "
            "check that the system clock is in sync, then try again."
        )


class ConfigurationError(LoginError):
    """Local misconfiguration detected before any browser was opened."""


# Callbacks the host (CLI/TUI) implements to drive the interactive flow.
# Every field is optional; sync or async callables both work.
@dataclasses.dataclass
class LoginCallbacks:
    """Host hooks for an interactive login.

    ``on_auth_url`` receives the authorization URL and an optional
    ``instructions`` string (the short ``/launch`` redirect when a loopback
    server is up). ``on_manual_code_input`` is only invoked for paste-code
    providers; returning ``None`` declines.
    """

    on_auth_url: Callable[..., Awaitable[None] | None] | None = None
    on_progress: Callable[[str], Awaitable[None] | None] | None = None
    on_manual_code_input: Callable[[], Awaitable[str | None] | str | None] | None = None


async def _maybe_await(value: Any) -> Any:
    if asyncio.iscoroutine(value):
        return await value
    return value


@dataclasses.dataclass
class CallbackFlowOptions:
    """Options for :class:`OAuthCallbackFlow`.

    ``redirect_uri`` pins the exact URI and disables port fallback (provider
    allowlist). ``manual_input_only`` skips the server entirely; the user
    pastes the code from the provider page.
    """

    preferred_port: int
    callback_path: str = "/callback"
    callback_hostname: str = "localhost"
    redirect_uri: str | None = None
    allow_port_fallback: bool = True
    manual_input_only: bool = False
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS


class OAuthCallbackFlow(ABC):
    """Base class for authorization-code + PKCE logins with a loopback server.

    Subclasses implement only :meth:`generate_auth_url` and
    :meth:`exchange_token`. The base owns the server lifecycle, route
    dispatch, timeout, abort handling, and the browser launch (injectable
    for tests).
    """

    def __init__(
        self,
        options: CallbackFlowOptions,
        callbacks: LoginCallbacks | None = None,
        *,
        open_browser: Callable[[str], None] | None = None,
        signal: AbortSignal | None = None,
    ) -> None:
        self.options = options
        self.callbacks = callbacks or LoginCallbacks()
        self._open_browser = open_browser or (lambda url: webbrowser.open(url))
        self._signal = signal
        self._server: asyncio.base_events.Server | None = None
        self._bound_port: int | None = None
        self._pending_auth_url: str | None = None
        self._captured: asyncio.Future[tuple[str, str]] | None = None
        self._capture_error: asyncio.Future[str] | None = None
        self._sent_state: str | None = None

    # -- subclass hooks ----------------------------------------------------

    @abstractmethod
    async def generate_auth_url(self, state: str, redirect_uri: str) -> str:
        """Build the provider authorization URL (PKCE params already stored)."""

    @abstractmethod
    async def exchange_token(self, code: str, state: str, redirect_uri: str) -> Any:
        """Exchange ``code`` for provider credentials (return value is flow-specific,
        usually an ``OAuthCredentials`` dict)."""

    # -- server ------------------------------------------------------------

    @property
    def bound_port(self) -> int | None:
        """The port the loopback server actually bound to (None before start)."""
        return self._bound_port

    def redirect_uri(self) -> str:
        opts = self.options
        if opts.redirect_uri:
            return opts.redirect_uri
        port = self._bound_port if self._bound_port is not None else opts.preferred_port
        return f"http://{opts.callback_hostname}:{port}{opts.callback_path}"

    async def _start_server(self) -> None:
        opts = self.options
        try:
            self._server = await asyncio.start_server(
                self._handle_connection, "127.0.0.1", opts.preferred_port
            )
            self._bound_port = self._socket_port()
            return
        except OSError:
            pass
        # Preferred port busy: fall back to an OS-assigned port only when the
        # provider does not pin the redirect URI.
        pinned = opts.redirect_uri is not None or not opts.allow_port_fallback
        if pinned:
            raise ConfigurationError(
                f"Port {opts.preferred_port} is required for this login flow but is already "
                "in use. Stop the process holding it and retry."
            )
        try:
            self._server = await asyncio.start_server(self._handle_connection, "127.0.0.1", 0)
        except OSError as exc:
            raise ConfigurationError(f"Could not bind a loopback callback server: {exc}") from exc
        self._bound_port = self._socket_port()

    def _socket_port(self) -> int | None:
        """The actually-bound port — the one that lands in redirect_uri."""
        if self._server is None or not self._server.sockets:
            return None
        return int(self._server.sockets[0].getsockname()[1])

    async def _stop_server(self) -> None:
        if self._server is not None:
            self._server.close()
            try:
                await self._server.wait_closed()
            except Exception:
                pass
            self._server = None

    async def _handle_connection(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """Minimal HTTP/1.1 request parsing — just enough for GET redirects."""
        try:
            raw = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), timeout=10.0)
        except (asyncio.IncompleteReadError, asyncio.TimeoutError, ConnectionError):
            raw = b""
        request_line = raw.split(b"\r\n", 1)[0].decode("latin-1", "replace")
        parts = request_line.split(" ")
        method, target = (parts[0], parts[1]) if len(parts) >= 2 else ("", "")
        path = urllib.parse.urlsplit(target).path

        opts = self.options
        if path == opts.callback_path and method == "GET":
            query = dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(target).query))
            error = query.get("error")
            if error:
                desc = query.get("error_description", "")
                self._finish_error(f"Authorization failed: {error} {desc}".strip())
                body = b"<html><body><h1>Login failed</h1><p>You may close this tab.</p></body></html>"
                await self._respond(writer, 200, body)
            else:
                code = query.get("code", "")
                state = query.get("state", "")
                if code:
                    # PR-13: a pinned loopback port accepts any local
                    # connection — verify the state we sent before trusting
                    # the code.
                    if self._sent_state is not None and not secrets.compare_digest(
                        state, self._sent_state
                    ):
                        self._finish_error(
                            "Authorization callback state mismatch — stale tab or forged "
                            "redirect. Restart the login."
                        )
                        body = b"<html><body><h1>Login failed</h1><p>You may close this tab.</p></body></html>"
                        await self._respond(writer, 200, body)
                    else:
                        self._finish(code, state)
                        body = b"<html><body><h1>Login complete</h1><p>You may close this tab.</p></body></html>"
                        await self._respond(writer, 200, body)
                else:
                    # PR-14: no code AND no error — fail the login promptly
                    # instead of hanging out the 300 s timeout.
                    self._finish_error(
                        "Authorization callback arrived with neither a code nor an error "
                        "parameter. Restart the login."
                    )
                    body = b"<html><body><h1>Login failed</h1><p>You may close this tab.</p></body></html>"
                    await self._respond(writer, 200, body)
        elif path == "/launch" and method == "GET":
            if self._pending_auth_url:
                await self._respond(writer, 302, b"", extra_headers=[("Location", self._pending_auth_url)])
            else:
                await self._respond(writer, 404, b"no pending login")
        else:
            await self._respond(writer, 404, b"not found")
        try:
            writer.close()
            await writer.wait_closed()
        except Exception:
            pass

    async def _respond(
        self,
        writer: asyncio.StreamWriter,
        status: int,
        body: bytes,
        extra_headers: list[tuple[str, str]] | None = None,
    ) -> None:
        reason = {200: "OK", 302: "Found", 404: "Not Found"}.get(status, "OK")
        lines = [
            f"HTTP/1.1 {status} {reason}",
            f"Content-Length: {len(body)}",
            "Content-Type: text/html; charset=utf-8",
            "Connection: close",
        ]
        for name, value in extra_headers or []:
            lines.append(f"{name}: {value}")
        payload = ("\r\n".join(lines) + "\r\n\r\n").encode("latin-1") + body
        try:
            writer.write(payload)
            await writer.drain()
        except Exception:
            pass

    def _finish(self, code: str, state: str) -> None:
        if self._captured and not self._captured.done():
            self._captured.set_result((code, state))

    def _finish_error(self, message: str) -> None:
        if self._capture_error and not self._capture_error.done():
            self._capture_error.set_result(message)

    def _launch_url(self) -> str | None:
        """Short 302 alias for the auth URL; only safe for loopback http(s)."""
        if self.options.manual_input_only or self._server is None or self._bound_port is None:
            return None
        uri = urllib.parse.urlsplit(self.redirect_uri())
        if uri.scheme not in ("http", "https") or uri.hostname not in ("localhost", "127.0.0.1"):
            return None
        if self.options.callback_path == "/launch":
            return None
        return f"http://{self.options.callback_hostname}:{self._bound_port}/launch"

    # -- driver ------------------------------------------------------------

    async def run(self) -> Any:
        """Run the full flow and return the exchanged credentials.

        Raises :class:`LoginCancelledError`, :class:`LoginTimeoutError`,
        :class:`ConfigurationError`, or whatever :meth:`exchange_token` raises.
        """
        loop = asyncio.get_running_loop()
        self._captured = loop.create_future()
        self._capture_error = loop.create_future()
        state = secrets.token_hex(16)
        self._sent_state = state
        try:
            if not self.options.manual_input_only:
                await self._start_server()
            redirect_uri = self.redirect_uri()
            auth_url = await self.generate_auth_url(state, redirect_uri)
            self._pending_auth_url = auth_url

            launch_url = self._launch_url()
            if self.callbacks.on_auth_url is not None:
                instructions = f"Or open: {launch_url}" if launch_url else None
                await _maybe_await(
                    self.callbacks.on_auth_url(auth_url, instructions=instructions)
                )
            if not self.options.manual_input_only:
                try:
                    self._open_browser(auth_url)
                except Exception:
                    pass  # headless host; the URL was already surfaced

            code, cb_state = await self._await_code()
            return await self.exchange_token(code, cb_state or state, redirect_uri)
        finally:
            await self._stop_server()

    async def _await_code(self) -> tuple[str, str]:
        assert self._captured is not None and self._capture_error is not None
        waiters: list[asyncio.Future[Any]] = []
        loop = asyncio.get_running_loop()

        async def _manual() -> tuple[str, str]:
            # Only paste-code providers may prompt; otherwise this races the
            # HTTP callback and leaves a dirty terminal (see module docstring).
            if self.callbacks.on_manual_code_input is None:
                await asyncio.Future()  # park forever
            pasted = await _maybe_await(self.callbacks.on_manual_code_input())
            if pasted is None:
                await asyncio.Future()  # declined; keep waiting for the browser
            pasted = pasted.strip()
            # Providers hand users "code#state" in the redirect URL fragment.
            if "#" in pasted:
                code, _, frag_state = pasted.partition("#")
                return code.strip(), frag_state.strip()
            return pasted, ""

        async def _abort_watch() -> tuple[str, str]:
            assert self._signal is not None
            await self._signal.wait()
            raise LoginCancelledError(self._signal.reason or "Login cancelled")

        waiters.append(self._captured)
        waiters.append(self._capture_error)
        waiters.append(loop.create_task(_manual()))
        if self._signal is not None:
            waiters.append(loop.create_task(_abort_watch()))

        try:
            done, _pending = await asyncio.wait(
                waiters, timeout=self.options.timeout_seconds, return_when=asyncio.FIRST_COMPLETED
            )
        finally:
            for task in waiters:
                task.cancel()

        if not done:
            raise LoginTimeoutError()
        for task in done:
            exc = task.exception()
            if exc is not None:
                raise exc
            result = task.result()
            if isinstance(result, str):  # capture_error path
                raise LoginError(result)
            return result
        raise LoginTimeoutError()  # unreachable
