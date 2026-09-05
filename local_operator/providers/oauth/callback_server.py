"""Loopback OAuth callback server shared by all authorization-code flows.

OAuth callback server. Invariants worth
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
import inspect
import json
import logging
import secrets
import urllib.parse
import webbrowser
from abc import ABC, abstractmethod
from typing import Any, Awaitable, Callable, NoReturn, TypeVar

from local_operator.callback_page import Tone, render_callback_page
from local_operator.harness.types import AbortSignal

DEFAULT_TIMEOUT_SECONDS = 300.0


def _parse_pasted_callback(pasted: str) -> tuple[str, str]:
    """Pull ``(code, state)`` out of whatever the user actually pasted.

    Three shapes reach this prompt, and which one a user produces depends on
    what their provider's browser page shows them, not on what we asked for:

    1. The whole redirect URL (``http://localhost:54548/callback?code=..&state=..``),
       which is what a user copies from the address bar when the browser is on
       another machine and cannot reach this loopback port. This is the shape
       the "paste the redirect URL" fallback exists for.
    2. ``code#state``, which Anthropic renders as a single copy target.
    3. A bare authorization code.

    Handling only (2) and (3) meant a pasted URL was sent to the token endpoint
    verbatim AS the authorization code, so the fallback advertised for remote
    and headless sessions could not complete a login at all. Query parameters
    are tried first because a URL is unambiguous: it has a scheme and a
    ``code`` parameter, neither of which a bare code or a ``code#state`` pair
    can produce.
    """
    if "://" in pasted:
        parsed = urllib.parse.urlsplit(pasted)
        query = urllib.parse.parse_qs(parsed.query)
        code = (query.get("code") or [""])[0].strip()
        if code:
            # `state` may ride in the query (the ordinary case) or in the
            # fragment, which some providers use to keep it out of server logs.
            state = (query.get("state") or [""])[0].strip()
            if not state and parsed.fragment:
                frag = urllib.parse.parse_qs(parsed.fragment)
                state = (frag.get("state") or [""])[0].strip() or parsed.fragment.strip()
            return code, state
        # A URL with no `code` is an error redirect or a mis-copy. Falling
        # through would send the whole URL as the code and produce an opaque
        # provider-side rejection, so say what is wrong while the user is still
        # at the prompt.
        error = (query.get("error") or [""])[0].strip()
        if error:
            # The raw OAuth error code is kept deliberately. This is a
            # developer-facing CLI, and the code is the string a user searches
            # for and quotes in a support thread; translating it would remove
            # the only durable handle on the failure. What follows it is the
            # part that was missing: what to do next.
            description = (query.get("error_description") or [""])[0].strip()
            detail = f" ({description})" if description else ""
            raise LoginError(
                f"Authorization failed: {error}{detail}. Approve the sign-in in "
                "your browser, then paste the address bar contents again."
            )
        raise LoginError(
            "That URL carries no authorization code. Approve the sign-in in your "
            "browser, then copy the whole address bar and paste it here."
        )
    # Providers hand users "code#state" in the redirect URL fragment.
    if "#" in pasted:
        code, _, frag_state = pasted.partition("#")
        return code.strip(), frag_state.strip()
    return pasted, ""


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


class InvalidGrantError(LoginError):
    """The stored grant is PERMANENTLY dead — only a fresh login can fix it.

    A refresh can fail two ways that look identical at the call site and are
    opposite in what the operator should do about them. A 5xx, a timeout, a
    429 or a dropped connection is the provider or the network being
    temporarily unwell: retrying is exactly right, and the existing
    per-account backoff exists for it. ``invalid_grant`` is the IdP stating
    that this refresh token will never be accepted again (revoked, superseded
    by a rotation we lost, or expired past its absolute lifetime), and no
    amount of retrying changes that answer.

    Conflating them is the defect this class exists to end: a dead kimi grant
    spent its retry budget, tripped ``usage_unavailable`` and then rendered as
    ``usage unavailable - last known 2d ago`` forever, a message that implies
    the numbers will come back on their own when the only remedy is
    ``/login <provider>``.

    Subclasses :class:`LoginError` deliberately: every existing ``except
    LoginError`` handler keeps working unchanged, and only callers that want
    the distinction test for this type.
    """


#: OAuth2 token-endpoint error codes that are terminal for a STORED grant
#: (RFC 6749 SS5.2). Terminal here means "re-presenting the same refresh token
#: to the same client can never succeed", which is a narrower question than
#: "was this request rejected".
#:
#: - ``invalid_grant``: the grant itself is revoked/expired/already-rotated.
#:   The direct signal, and the one observed against auth.kimi.com.
#: - ``invalid_client`` / ``unauthorized_client``: the CLIENT is rejected, so
#:   every grant held under it is unusable by us. A re-login is still the only
#:   move available to the operator, and burning a retry loop is still wrong.
#:
#: Deliberately EXCLUDED, all for one reason -- they describe a malformed or
#: unsupported REQUEST rather than a dead grant, so they would blame the user's
#: login for a bug or a config change of ours, and print a ``/login`` remedy
#: that cannot possibly work:
#:
#: - ``invalid_request`` and ``invalid_scope``.
#: - ``unsupported_grant_type``: the endpoint is refusing the
#:   ``grant_type=refresh_token`` PARAMETER, which says nothing about the
#:   stored grant. It was briefly listed here on the reasoning that retrying
#:   re-sends the same unsupported request -- true, but not-worth-retrying and
#:   dead are different questions, and only the second is this set's subject.
#:   The blast radius is what settles it: a token-endpoint config change would
#:   darken EVERY account on the provider at once with a remedy that cannot
#:   fix it, which is this defect's own failure mode restated.
#: - ``slow_down`` / ``authorization_pending``, which are device-flow polling
#:   states, not failures.
TERMINAL_GRANT_ERRORS = frozenset(
    {
        "invalid_grant",
        "invalid_client",
        "unauthorized_client",
    }
)


def _oauth_error_code(body: str) -> str | None:
    """The value of the OAuth2 ``error`` FIELD, or None if the body has none.

    Reading the field rather than scanning the body is what keeps the exclusion
    list above meaningful. A substring scan cannot express "this code, in the
    role of the verdict": ``invalid_client`` is a substring of
    ``invalid_client_id`` and ``invalid_grant`` of ``invalid_grant_type``, so
    the EXCLUDED ``invalid_request`` was being matched through the INCLUDED
    codes whenever a provider's prose mentioned one. All three of these are
    real 400 ``invalid_request`` shapes that a scan calls terminal:

        {"error":"invalid_request","error_description":"invalid_grant_type: ..."}
        {"error":"invalid_request","error_description":"the invalid_client_id ..."}
        {"error":"invalid_request","error_uri":"https://.../errors#invalid_grant"}

    The third is not even prose: RFC 6749 SS5.2 permits ``error_uri``, and a doc
    anchor naturally ends in the error's name.

    Two shapes are parsed because both are real: JSON (what the RFC specifies
    and what every provider here returns) and form-encoded (seen from proxies
    and older endpoints). Anything else returns None and the caller falls back
    to a scan -- degrading an unparseable body to "transient" outright is the
    failure this classification exists to remove, so the fallback stays.
    """
    stripped = body.strip()
    if not stripped:
        return None
    if stripped.startswith("{"):
        try:
            parsed = json.loads(stripped)
        except ValueError:
            return None
        if not isinstance(parsed, dict):
            return None
        # Some providers wrap the RFC object one level down (`{"error": {...}}`
        # with the code inside); read that shape too rather than mistaking the
        # envelope for a missing verdict.
        error = parsed.get("error")
        if isinstance(error, dict):
            error = error.get("code") or error.get("type") or error.get("error")
        return error.strip().lower() if isinstance(error, str) else None
    if "=" in stripped and "\n" not in stripped:
        for key, value in urllib.parse.parse_qsl(stripped):
            if key.strip().lower() == "error":
                return value.strip().lower()
    return None


def is_terminal_grant_response(status_code: int, body: str) -> bool:
    """Whether a non-200 token-endpoint response means the grant is dead.

    **The status code alone is not the signal, and must not be used as one.**
    RFC 6749 SS5.2 puts the machine-readable verdict in the BODY: the same 400
    carries ``invalid_grant`` (terminal) and ``invalid_request`` (our bug,
    retryable). Meanwhile a 401 is routinely a gateway or an auth proxy having
    a bad minute. So the code only gates which bodies are worth reading -- 5xx
    is the provider failing and its body is a stack trace or an HTML error
    page, never a verdict about the operator's grant.

    The verdict is taken from the parsed ``error`` field when the body yields
    one, and only from a substring scan when it does not. Parsing is what makes
    the exclusion list real (see :func:`_oauth_error_code`); the scan is what
    keeps an HTML-wrapped or otherwise unparseable ``invalid_grant`` from being
    silently downgraded to a retry loop that can never end.
    """
    # 5xx and 429 are the provider's problem, whatever the body says: keep
    # today's retry/backoff behaviour for them.
    if status_code >= 500 or status_code == 429:
        return False
    code = _oauth_error_code(body)
    if code is not None:
        # The body named its verdict: honour it exactly, including when the
        # verdict is an excluded code that merely MENTIONS a terminal one.
        return code in TERMINAL_GRANT_ERRORS
    lowered = body.lower()
    return any(candidate in lowered for candidate in TERMINAL_GRANT_ERRORS)


def raise_for_refresh_failure(
    provider_label: str, status_code: int, body: str, *, message: str | None = None
) -> NoReturn:
    """Raise the right error class for a failed token-endpoint refresh.

    One helper so every provider's refresh classifies identically. A dead
    grant is a property of OAuth2, not of a vendor, so anthropic/openai/xai/
    kimi/radient must not each decide it for themselves.

    ``NoReturn`` because it always raises: without it every call site reads to
    a type checker as though it falls through into the code that assumes a 200.

    ``message`` overrides the default wording for the one provider whose
    historical text differs (Radient's ``: HTTP {status} {body}``). The message
    is what a user searches for and quotes in a bug report, so this helper
    classifies the failure without restyling anyone's error string.
    """
    message = message or f"{provider_label} refresh failed ({status_code}): {body}"
    if is_terminal_grant_response(status_code, body):
        raise InvalidGrantError(message)
    raise LoginError(message)


# Callbacks the host (CLI/TUI) implements to drive the interactive flow.
# Every field is optional; sync or async callables both work.
@dataclasses.dataclass
class LoginCallbacks:
    """Host hooks for an interactive login.

    ``on_auth_url`` receives the authorization URL and an optional
    ``instructions`` string (the short ``/launch`` redirect when a loopback
    server is up). ``on_manual_code_input`` is only invoked for paste-code
    providers; returning ``None`` declines.

    ``on_warning`` reports something that WENT WRONG but did not end the login
    (a paste the flow could not use), as distinct from ``on_progress``, which
    narrates what is happening normally. They are separate hooks because a host
    styles them differently: routed through ``on_progress``, a failed
    authorization rendered in the same dim treatment as "opening your
    browser…", so the one line that explained why nothing happened read as
    routine narration. Optional, and falls back to ``on_progress`` when a host
    does not implement it, so no existing host loses the message.

    ``on_input_rejected`` fires when a value returned by
    ``on_manual_code_input`` could not be parsed, just before the flow asks
    again. It carries no message (``on_warning`` already delivered the reason)
    and exists so a host that RENDERED the paste can correct what it showed:
    the TUI settles its prompt into a receipt the moment the value is handed
    over, so without this it painted a success receipt over a paste the flow
    had rejected. A host with no such surface simply omits it.
    """

    on_auth_url: Callable[..., Awaitable[None] | None] | None = None
    on_progress: Callable[[str], Awaitable[None] | None] | None = None
    on_warning: Callable[[str], Awaitable[None] | None] | None = None
    on_input_rejected: Callable[[], Awaitable[None] | None] | None = None
    on_manual_code_input: Callable[[], Awaitable[str | None] | str | None] | None = None
    # Endpoint setup is a sequence of explicit fields, not an OAuth code. The
    # host decides how to capture them; the controller never touches stdin.
    on_setup_input: Callable[[str, str, bool], Awaitable[str | None] | str | None] | None = None


_T = TypeVar("_T")


logger = logging.getLogger(__name__)


async def report_safely(
    hook: Callable[..., Awaitable[None] | None] | None,
    *args: Any,
) -> None:
    """Invoke a host REPORTING hook, swallowing anything it raises.

    Reporting hooks (``on_progress``, ``on_warning``, ``on_input_rejected``)
    tell the user what is happening; they take no part in deciding the login.
    An embedding host whose sink raises would otherwise propagate out of the
    waiter and out of ``_await_code`` -- losing a sign-in the browser callback
    was about to complete, which is the exact failure the surrounding code
    exists to prevent. A host that cannot render a message must not be able to
    destroy a credential grant.

    Deliberately NOT used for ``on_manual_code_input``: that hook returns a
    value the flow acts on, so an exception there is a real failure and has to
    surface rather than be swallowed.
    """
    if hook is None:
        return
    try:
        await maybe_await(hook(*args))
    except Exception:  # pragma: no cover - host-defined sinks
        logger.debug("a login reporting hook raised; continuing", exc_info=True)


async def maybe_await(value: Awaitable[_T] | _T) -> _T:
    """Await ``value`` when a host callback handed back an awaitable.

    Every hook a host supplies (see :class:`LoginCallbacks`) may be written
    sync or async, so results funnel through here instead of an inline
    ``__await__`` probe that no type checker can narrow.
    """
    if inspect.isawaitable(value):
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
    #: Display name of the provider being signed into ("Anthropic", "OpenAI"),
    #: rendered in a labelled trough on the browser landing page so the user
    #: can see WHOSE login just finished. Optional because the page is honest
    #: without it — the trough is omitted rather than faked (mirrors how the
    #: MCP flow treats its server URL).
    provider_label: str | None = None


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
    async def exchange_token(self, code: str, state: str, redirect_uri: str) -> dict[str, Any]:
        """Exchange ``code`` for the provider credentials mapping (the shape is
        flow-specific, but always the ``OAuthCredentials`` dict the auth store
        persists)."""

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

    async def _handle_connection(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        """Minimal HTTP/1.1 request parsing — just enough for GET redirects."""
        try:
            raw = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), timeout=10.0)
        except (asyncio.IncompleteReadError, asyncio.TimeoutError, ConnectionError):
            raw = b""
        request_line = raw.split(b"\r\n", 1)[0].decode("latin-1", "replace")
        parts = request_line.split(" ")
        method, target = (parts[0], parts[1]) if len(parts) >= 2 else ("", "")
        if not method or not target:
            # Browsers may preconnect while the user is still granting consent.
            # A timed-out/unfinished request is not an unknown HTTP route: an
            # unsolicited 404 can be reused for the later callback navigation,
            # showing "Nothing here" until reload. Close silently so that the
            # browser reconnects, without consuming or failing the OAuth flow.
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass
            return
        path = urllib.parse.urlsplit(target).path

        opts = self.options
        if path == opts.callback_path and method == "GET":
            query = dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(target).query))
            error = query.get("error")
            if error:
                desc = query.get("error_description", "")
                self._finish_error(f"Authorization failed: {error} {desc}".strip())
                # The provider's words go in their own labelled trough rather
                # than into our sentence — same voice boundary the MCP flow
                # draws, and where a bare `access_denied` reads as data rather
                # than as broken English. Stripped `desc` falls back to the
                # error code so a `error_description=%20%20%20` redirect still
                # names what went wrong.
                body = self._page(
                    "Sign-in failed",
                    "The provider did not grant this sign-in, so nothing was "
                    "connected. You can start the login again from Local Operator.",
                    tone="danger",
                    provider_message=desc.strip() or error,
                )
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
                        body = self._page(
                            "Sign-in failed",
                            "This redirect did not match the login Local Operator "
                            "started. It may be a stale tab, or a redirect it "
                            "never asked for. Nothing was connected. Restart the "
                            "login from Local Operator.",
                            tone="danger",
                        )
                        await self._respond(writer, 200, body)
                    else:
                        self._finish(code, state)
                        # "Authorization complete", not "Signed in": this page is
                        # served on CODE receipt, before the token exchange, and
                        # its own detail line says the sign-in is still finishing.
                        # Matches the register of the MCP surface's "Authorized"
                        # (D1, PR #177 design round 1).
                        body = self._page(
                            "Authorization complete",
                            "Local Operator has the authorization code and is "
                            "finishing the sign-in.",
                            tone="success",
                        )
                        await self._respond(writer, 200, body)
                else:
                    # PR-14: no code AND no error — fail the login promptly
                    # instead of hanging out the 300 s timeout.
                    self._finish_error(
                        "Authorization callback arrived with neither a code nor an error "
                        "parameter. Restart the login."
                    )
                    body = self._page(
                        "No authorization code",
                        "The redirect arrived without an authorization code, so "
                        "there is nothing to hand back. You can start the login "
                        "again from Local Operator.",
                        tone="danger",
                    )
                    await self._respond(writer, 200, body)
        elif path == "/launch" and method == "GET":
            if self._pending_auth_url:
                await self._respond(
                    writer,
                    302,
                    b"",
                    extra_headers=[("Location", self._pending_auth_url)],
                )
            else:
                body = self._page(
                    "No login in progress",
                    "There is no sign-in waiting on this address. Start the "
                    "login again from Local Operator.",
                    closable=False,
                    show_provider=False,
                )
                await self._respond(writer, 404, body)
        else:
            # Browsers ask for /favicon.ico off their own bat; a neutral 404
            # keeps a speculative fetch from being mistaken for the redirect.
            body = self._page(
                "Nothing here",
                "This address only answers the login redirect.",
                closable=False,
                show_provider=False,
            )
            await self._respond(writer, 404, body)
        try:
            writer.close()
            await writer.wait_closed()
        except Exception:
            pass

    def _page(
        self,
        title: str,
        detail: str,
        *,
        tone: Tone = "neutral",
        provider_message: str | None = None,
        closable: bool = True,
        show_provider: bool = True,
    ) -> bytes:
        """One outcome of this login, as the shared Local Operator page.

        Same document the MCP OAuth listener serves (``callback_page``), so a
        user who authorizes a provider and an MCP server sees one product on
        both landings instead of a styled card on one and bare ``<h1>`` HTML
        on the other. The provider's display name rides along when the flow
        knows it — but only on real outcome pages. ``show_provider=False`` is
        for the neutral 404s (a speculative favicon fetch, ``/launch`` with
        nothing pending): a non-event given the outcome grammar of a labelled
        provider trough would read as if something happened with that provider
        (F2/D2, PR #177 round 1).
        """
        return render_callback_page(
            title,
            detail,
            tone=tone,
            provider=self.options.provider_label if show_provider else None,
            provider_message=provider_message,
            closable=closable,
        ).encode()

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
            "Cache-Control: no-store",
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
        if uri.scheme not in ("http", "https") or uri.hostname not in (
            "localhost",
            "127.0.0.1",
        ):
            return None
        if self.options.callback_path == "/launch":
            return None
        return f"http://{self.options.callback_hostname}:{self._bound_port}/launch"

    # -- driver ------------------------------------------------------------

    async def run(self) -> dict[str, Any]:
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
                await maybe_await(self.callbacks.on_auth_url(auth_url, instructions=instructions))
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
            # The loops park forever: this task exists only to lose the race in
            # asyncio.wait, and re-parking is the correct answer if a future
            # ever did resolve.
            prompt = self.callbacks.on_manual_code_input
            while prompt is None:
                await asyncio.Future()  # park forever
            while True:
                pasted = await maybe_await(prompt())
                while pasted is None:
                    await asyncio.Future()  # declined; keep waiting for the browser
                try:
                    return _parse_pasted_callback(pasted.strip())
                except LoginError as exc:
                    # A paste this task cannot use must not end the login, AND
                    # must not leave the user with nowhere to put a corrected
                    # one. Two rules meet here:
                    #
                    # The prompt RACES the loopback callback -- it is the
                    # fallback for a browser that cannot reach this machine --
                    # so raising would let a mistyped URL kill a sign-in the
                    # browser was about to finish. That is why the line above
                    # re-parks on a DECLINED paste rather than failing.
                    #
                    # But parking is only the right answer when the callback can
                    # still win, and for the user this fallback EXISTS for it
                    # cannot: their browser is on another machine. For them a
                    # single mis-paste meant a settled prompt, a message telling
                    # them to copy the address bar, and no field left to paste
                    # it into -- then silence until the timeout. So the reason
                    # is reported and the prompt is offered AGAIN, which is the
                    # only outcome that matches what the message asks for.
                    #
                    # Declining the re-prompt still parks, so a user who has
                    # given up waits for the browser or the timeout exactly as
                    # before, and the callback keeps its chance to win either
                    # way because this loop never blocks it.
                    await report_safely(
                        self.callbacks.on_warning or self.callbacks.on_progress,
                        str(exc),
                    )
                    # Ordered after the message so a host that repaints on
                    # rejection does so with the reason already on screen
                    # above it.
                    await report_safely(self.callbacks.on_input_rejected)
                    # Yield before asking again. A host may implement
                    # ``on_manual_code_input`` SYNCHRONOUSLY (the callbacks are
                    # documented to allow it, and the CLI host and the tests
                    # both do), in which case nothing in this loop body ever
                    # suspends: ``maybe_await`` returns without awaiting on a
                    # plain value, so the loop would spin without returning
                    # control to the scheduler. That starves the whole event
                    # loop -- the loopback callback future can never be
                    # resolved, and the flow's own timeout can never fire, so a
                    # single bad paste hangs the login forever instead of
                    # merely ending it. Handing one iteration back to the
                    # scheduler is what keeps the race the comment above
                    # describes actually winnable.
                    #
                    # A host that returns a value WITHOUT waiting for the user
                    # (a test double, not a real prompt -- the TUI awaits a
                    # mounted block's future and the CLI awaits
                    # `asyncio.to_thread(read_line)`) will re-offer in a tight
                    # loop until another waiter wins or the flow's timeout
                    # fires. Bounded in TIME, not in WORK: the loop also
                    # reports each rejection, so such a host would see tens of
                    # thousands of warnings inside one timeout window (~37k in
                    # 1s when measured). Liveness is what the yield restores;
                    # an immediate-return prompt is a host bug, and this note
                    # exists so it reads as one rather than as merely wasteful.
                    await asyncio.sleep(0)

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
                waiters,
                timeout=self.options.timeout_seconds,
                return_when=asyncio.FIRST_COMPLETED,
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
