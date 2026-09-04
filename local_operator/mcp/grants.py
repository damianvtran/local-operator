"""``/mcp login``, ``/mcp logout`` and ``/mcp reauth``, shared by every host.

These three verbs move OAuth grants: they delete the row in the shared
``auth.db`` and run the interactive browser exchange that writes a new one.
Like :mod:`local_operator.mcp.verbs`, they are MACHINE-AND-SESSION work rather
than terminal work, so a detached runtime must be able to run them too.

Before this module the runtime could not. ``OwnedSessionHandle._mcp_slash``
refused all three with "run it from a terminal on that machine", on the theory
that the invoker sits somewhere the runtime's browser cannot reach. That was
wrong for the only topology that exists today: the control socket binds
``127.0.0.1`` and nothing else (``runtime/server.py::_serve`` calls it "the
security invariant of the whole design"), so a client that can dial a runtime
is ALREADY on the runtime's machine and its default browser is the user's.
The refusal therefore fired on the one case it was meant to protect — a user
sitting at the machine that stores the credentials — and left ``/mcp reauth``
with no working path at all on a detached session: the terminal routes the
verb to the owner, and the owner declined it.

The locality question the refusal was reaching for is real but is NOT answered
by guessing from inside this process. It is a property of the CLIENT, declared
on the wire (``ClientLocality`` in ``runtime/types.py``) and passed in as
``browser_is_reachable``. Loopback attach clients are local by construction;
a future mobile relay carrying a slash command from a phone is not, and it
gets :data:`REMOTE_GRANT_NOTICE` — the refusal this module keeps, now aimed at
the case it actually describes.

Why the grant is not simply awaited inline: the exchange waits on a human and
is budgeted at 600 s, while an attach client gives up on a request after
``ACK_TIMEOUT_S`` (15 s) and the runtime's per-connection reader is strictly
serial (``readline()`` then ``await _on_request(...)``). Awaiting it in the
request would time out the caller AND park every other op on that connection —
model switches, aborts, prompts — behind a browser tab for ten minutes. So
:func:`start_grant` returns immediately with the receipt a user needs and
reports the settled outcome as a ``NoticeEvent``, which the relay already fans
out to every attached front end. That is the same shape ``/compact`` and the
login worker in the TUI use, for the same reason.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Awaitable, Callable, Literal, cast

logger = logging.getLogger(__name__)

NoticeKind = Literal["info", "success", "warning", "error"]

#: Verbs this module owns. ``frontend_state`` imports it as the set the
#: dispatch routes to the authoritative owner, so the two cannot drift.
GRANT_SUBCOMMANDS = ("login", "logout", "reauth")

#: The budget one interactive exchange gets. Matches the TUI's own login
#: worker: the user has to find the browser tab, sign in, and consent.
GRANT_TIMEOUT_MS = 600_000

#: Refused when the invoking client is NOT on this machine. This is the
#: wording the old unconditional refusal used, kept because it is right for
#: this case: the browser would open on the host and the credential would land
#: in the host's ``auth.db``, neither of which the person holding the phone can
#: see. A relay that wants to support grants must carry the authorization URL
#: to the device rather than route the verb.
REMOTE_GRANT_NOTICE = (
    "/mcp {sub} opens a browser and stores credentials on the machine "
    "running the session — run it from a terminal on that machine"
)


def resolve_server(session: Any, name: str) -> tuple[Any, Any] | str:
    """The authoritative ``(manager, config)`` for ``name``, or a notice body.

    Every grant entry point validates against the SAME server set, so a verb
    typed in a detached session answers exactly as it would in an attached
    one. Returns the warning string on each unavailable path so no caller
    duplicates the checks.

    Only STATICALLY impossible cases are refused here: a stdio server (no
    transport that can carry a bearer token) and one whose config declares a
    non-OAuth auth type. Whether a bare ``url`` server takes OAuth is not
    decidable from the config — a Codex-imported entry carries only the URL —
    so that question is settled by the live capability probe in
    :func:`login_allowed`, never assumed here.
    """
    manager = getattr(session, "mcp_manager", None)
    if manager is None:
        return "MCP is not available in this session."
    get_config = getattr(manager, "get_server_config", None)
    if not callable(get_config):
        # A reduced host (a follower's read-only MCP facade) exposes no config
        # accessor. It has no grants to move either, so this is a refusal
        # rather than a fallback.
        return "MCP is not available in this session."
    cfg = get_config(name)
    if cfg is None:
        return f"MCP server {name!r} is not configured — see /mcp"
    from local_operator.mcp.auth import server_rejects_oauth
    from local_operator.mcp.config import MCPServerConfig

    if server_rejects_oauth(cast("MCPServerConfig", cfg)):
        return f"MCP server {name!r} does not use OAuth login."
    return manager, cfg


async def login_allowed(manager: Any, cfg: Any) -> bool:
    """Whether an explicit login on ``cfg`` may proceed, per ``manager``.

    Asked of the MANAGER rather than the auth module directly: the answer
    depends on the session's effective auth store, and a store-less probe
    refuses servers whose grant lives in an injected store. A reduced host
    that does not implement the accessor falls back to the static probe.
    """
    checker = getattr(manager, "server_supports_oauth_login", None)
    if callable(checker):
        return bool(await cast("Awaitable[bool]", checker(cfg)))
    from local_operator.mcp.auth import probe_oauth_capability

    return await probe_oauth_capability(cfg)


def logout_server(name: str) -> str | None:
    """Delete ``name``'s stored credential. Returns an error body, or ``None``.

    Deletion goes through the module helper, which writes the shared
    ``auth.db`` — the row every future session reads — rather than any store
    one session was injected with.
    """
    from local_operator.mcp.auth import mcp_logout_server

    try:
        return mcp_logout_server(name, os.getcwd())
    except Exception as exc:  # noqa: BLE001 — a failed logout is a notice, not a crash
        return str(exc)


async def run_grant(
    manager: Any, sub: str, name: str, forgotten: list[str] | None = None
) -> tuple[str, NoticeKind]:
    """Run one grant verb to completion; return the receipt and its style.

    This is the whole command, awaited: callers that can afford to block (the
    CLI, a test) use it directly, while :func:`start_grant` wraps it for hosts
    that cannot. ``reauth`` deletes the stored row and then runs the login,
    disconnecting between the two so the manager's auto-reconnect cannot read
    the old credential during the teardown window and quietly re-authenticate
    the session the user just reset.

    ``forgotten`` is an out-parameter the caller reads AFTER a cancellation:
    ``reauth`` is destructive before it is constructive, so a grant cancelled
    between the delete and the reconnect leaves the server with no credential
    at all. The caller cannot infer that from the exception — a cancel before
    the delete and a cancel after it raise the identical ``CancelledError`` —
    and telling the user "cancelled" when their grant is actually gone sends
    them to a server that will not connect (review F6). Appending the name here
    is what lets the notice say which of the two happened.
    """
    from local_operator.mcp.auth import McpLoginCancelledError

    if sub == "logout":
        error = logout_server(name)
        if error is not None:
            return f"MCP logout failed for {name!r}: {error}", "warning"
        try:
            await manager.disconnect_server(name)
        except Exception:  # noqa: BLE001 — a stuck teardown must not hide the receipt
            logger.debug("MCP disconnect after logout failed for %s", name, exc_info=True)
        return (
            f"logged out of MCP server {name!r} — its credential is removed and the "
            "server will stay disconnected until /mcp login.",
            "success",
        )

    if sub == "reauth":
        error = logout_server(name)
        if error is not None:
            return f"MCP reauth failed for {name!r}: {error}", "warning"
        if forgotten is not None:
            forgotten.append(name)
        try:
            await manager.disconnect_server(name)
        except Exception:  # noqa: BLE001 — a stuck teardown must not block the grant
            logger.debug("MCP disconnect before reauth failed for %s", name, exc_info=True)

    try:
        conn = await manager.connect_configured_server(name, timeout_ms=GRANT_TIMEOUT_MS)
    except asyncio.CancelledError:
        # The receipt must always get an ending: a grant cancelled by session
        # teardown otherwise leaves "authorizing…" with nothing answering it.
        raise
    except McpLoginCancelledError as exc:
        return f"MCP login for {name!r} cancelled: {exc}", "warning"
    except Exception as exc:  # noqa: BLE001 — a failed grant is a notice, not a crash
        return f"MCP login failed for {name!r}: {exc}", "error"
    return (
        f"authenticated MCP server {name!r}; {len(conn.tools)} tools available.",
        "success",
    )


async def start_grant(
    session: Any,
    sub: str,
    name: str,
    *,
    browser_is_reachable: bool,
    notify: Callable[[str, NoticeKind], Any],
    spawn: Callable[[Awaitable[None]], Any],
) -> tuple[str, NoticeKind]:
    """Validate one grant verb and start it; return the IMMEDIATE receipt.

    Only the checks that are FREE happen before returning: does this session
    have MCP, is the server configured, is it statically ineligible (a stdio
    server, or one whose config names a non-OAuth auth type). Those are local
    dictionary reads, so a typo is still refused instantly.

    Everything that can touch the network or wait on a person runs detached via
    ``spawn`` and reports through ``notify`` — including the OAuth capability
    probe, which is up to three sequential 10 s HTTP GETs and was measured at
    30.7 s against an unroutable host. That split is what keeps the runtime's
    serial reader free (see the module docstring): the caller gets its
    ``result`` frame in milliseconds no matter how slow the far side is.

    ``logout`` has no browser step, no probe and no human in the loop — it is
    two local operations — so it is awaited inline and its real outcome is the
    returned receipt. Only the verbs that can block are detached.
    """
    if not browser_is_reachable:
        return REMOTE_GRANT_NOTICE.format(sub=sub), "warning"

    resolved = resolve_server(session, name)
    if isinstance(resolved, str):
        return resolved, "warning"
    manager, cfg = resolved

    if sub == "logout":
        return await run_grant(manager, sub, name)

    # Tracks whether the destructive half of a `reauth` has already happened,
    # so the cancellation notice can say which state the user is left in.
    forgotten: list[str] = []

    async def _settle() -> None:
        try:
            # The half ``resolve_server`` cannot answer: it is synchronous, and
            # "does this server take an OAuth grant?" needs a metadata round
            # trip whenever the config does not say. Without it a deliberate
            # login is enabled for every remote server, including API-key ones
            # whose 401 would then open an unrelated OAuth attempt.
            #
            # Run INSIDE the detached task, not before it. The probe is up to
            # three sequential 10 s HTTP discovery GETs with no total cap, on
            # exactly the url-only config it exists for (a Codex import, or any
            # server just after ``/mcp logout`` — logout deletes the stored
            # credential that was the short-circuit evidence). Awaiting it in
            # the request measured 30.7 s against an unroutable host, past the
            # invoker's 15 s ``ACK_TIMEOUT_S`` and holding the runtime's serial
            # reader the whole time — the exact wedge this split exists to
            # prevent (review F2 / QA Q1). The receipt below is deliberately
            # provisional for that reason: the ineligible case is reported
            # through ``notify`` like any other outcome.
            if not await login_allowed(manager, cfg):
                notify(f"MCP server {name!r} does not use OAuth login.", "warning")
                return
            text, kind = await run_grant(manager, sub, name, forgotten)
        except asyncio.CancelledError:
            # WHICH cancellation this was decides what the user must do next,
            # and the exception cannot tell them apart. A `reauth` cancelled
            # after the delete (by a superseding grant, or by session teardown)
            # has already destroyed the credential, so "cancelled" alone reads
            # as "nothing changed" and sends the user back to a server that can
            # no longer connect (review F6). Name the state and the recovery.
            #
            # Both branches speak ONLY about this grant, never about the
            # server's overall state. ``forgotten`` is scoped to this
            # ``start_grant`` call, so it cannot see a delete performed by an
            # earlier one: after `/mcp reauth notion` (deletes, then parks) is
            # superseded by `/mcp login notion` (which deletes nothing), a
            # cancellation of the SECOND grant would truthfully report "this
            # grant changed nothing" while `notion` has no credential at all.
            # Saying "the stored credential is unchanged" there asserts
            # something this scope cannot know, and it is the LAST line the
            # user reads (QA Q2). Both branches therefore end on the same
            # recovery instruction, which is correct either way.
            if forgotten:
                notify(
                    f"MCP {sub} for {name!r} was cancelled after its old credential "
                    "was removed, so the server is now unauthenticated — "
                    f"run /mcp login {name} to finish.",
                    "warning",
                )
            else:
                notify(
                    f"MCP {sub} for {name!r} cancelled before the browser completed it; "
                    f"this attempt changed nothing — run /mcp login {name} to "
                    "authenticate the server.",
                    "warning",
                )
            raise
        except Exception as exc:  # noqa: BLE001 — never kill the host loop
            logger.debug("MCP %s failed for %s", sub, name, exc_info=True)
            notify(f"MCP {sub} failed for {name!r}: {exc}", "error")
            return
        notify(text, kind)

    spawn(_settle())
    # Deliberately does NOT promise a browser tab. Whether one opens depends on
    # the capability probe that now runs inside the task, so a receipt claiming
    # "a browser tab is opening" would be false for an ineligible server. It
    # states what is certainly true — the grant is under way and the answer
    # arrives here — and the settled ``notify`` says which way it went.
    return (
        f"authorizing MCP server {name!r} on this machine; "
        "the result appears here when it completes.",
        "info",
    )
