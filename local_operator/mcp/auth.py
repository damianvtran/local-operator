"""MCP OAuth support on the official SDK's ``OAuthClientProvider``.

Flow (official SDK PKCE + RFC 7591 DCR under the hood):

- ``build_oauth_provider(server_url, cfg)`` is the entry point: it wires the
  provider AND primes it with the stored token's expiry, which is what makes a
  restart spend the refresh token instead of re-running a browser grant.
- ``ensure_mcp_oauth_fresh(server_url, cfg)`` refreshes an expired access
  token BEFORE connecting, against the token endpoint resolved from the
  server's OAuth metadata (PRM + ASM discovery). This is what stops a day-old
  token from forcing a browser grant on startup for providers whose token
  endpoint is not ``<server_base>/token`` (the SDK's fallback guess 404s for
  e.g. Datadog). The refresh is serialized across processes with a file lock,
  and the exchange task OWNS that lock from its acquire to its store write —
  the connect only passes the handle over — so a cancelled or over-budget
  connect cannot free it while the POST is still on the wire, and the token is
  re-read under it, so concurrently starting sessions cannot spend a rotating
  refresh token twice.
- ``wire_oauth_auth(server_url, cfg)`` returns the ``OAuthClientProvider``
  kwargs: client metadata with a loopback redirect URI, a token storage bound
  to the shared credential store, and a :class:`LoopbackAuthFlow` that
  actually LISTENS on that redirect URI (with a pasted-URL race for browsers
  that cannot reach this machine).
- ``McpTokenStorage`` is the SDK ``TokenStorage``: one row per server URL in
  the real ``providers.auth_store.AuthStore``, keyed ``mcp_oauth:<url>``, with
  the token's issue time recorded so its lifetime survives the process.

Non-interactive connects (ordinary startup and auto-reconnect) pass
``interactive=False``: when the stored grant cannot be refreshed the flow
raises :class:`McpAuthRequiredError` instead of opening a browser, and the
manager surfaces that as an actionable "run /mcp login <name>" failure. Only
an explicit login (``/mcp login`` / ``local-operator mcp login``) runs
interactive and may open a browser.

Credential mapping onto the REAL AuthStore API (MCP-03): the store is keyed
by integer row id + ``provider`` column + ``identity_key``, so the logical
credential id ``mcp_oauth:<server_url>`` maps to ``provider='mcp-oauth'`` +
``identity_key=<server_url>`` (carried through the payload's ``project_id``
field, which the store's dedupe logic picks up). Reads filter
``list_credentials('mcp-oauth')`` by ``identity_key``; writes go through
``upsert_credential``, which updates the row in place on re-auth.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import sys
import threading
import time
import weakref
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable
from urllib.parse import parse_qs, urlparse

from pydantic import AnyUrl

from local_operator.ansi import strip_control_sequences
from local_operator.callback_page import callback_response
from local_operator.interpreter import python_argv

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    # The SDK is an optional extra: these names are needed for annotations
    # only, so importing them here keeps this module importable without it.
    from mcp.shared.auth import (
        AuthorizationCodeResult,
        OAuthClientInformationFull,
        OAuthMetadata,
        OAuthToken,
        ProtectedResourceMetadata,
    )

    from local_operator.mcp.config import MCPServerConfig
    from local_operator.providers.auth_store import StoredCredential

logger = logging.getLogger(__name__)

# Logical credential id prefix for managed MCP OAuth credentials (URL-keyed).
MCP_OAUTH_CREDENTIAL_PREFIX = "mcp_oauth:"

# Provider column value in the shared auth_credentials table.
MCP_OAUTH_PROVIDER = "mcp-oauth"

#: Loopback port for OAuth callbacks when a server does not pin its own.
#: 33441 is deliberately rare (sibling of Codex's 33418): :3000 collides with
#: local dev servers often enough that the listener bind routinely failed and
#: the grant fell back to the manual paste flow. Servers that registered a
#: redirect URI against the old default must pin ``callback_port: 3000`` in
#: their config oauth block to keep working.
DEFAULT_CALLBACK_PORT = 33441
DEFAULT_CALLBACK_PATH = "/callback"

#: Payload key holding the wall-clock time (epoch seconds) the stored access
#: token was issued. Not part of the SDK's ``OAuthToken`` — see
#: :meth:`McpTokenStorage.stored_token_expiry` for why we have to record it.
TOKENS_OBTAINED_AT_KEY = "tokens_obtained_at"

#: Payload key marking THIS row's refresh token as one the authorization server
#: has already rejected with ``invalid_grant``. Set only by the parsed-body
#: branch in :func:`_refresh_oauth_token_locked`; absence means "no such
#: observation", never "known good".
#:
#: Why this exists: nothing used to record the rejection, so every process boot
#: re-spent the same dead token (measured: 39 POSTs for one server, 87 for
#: another, across 26 boots). For a provider running refresh-token REUSE
#: DETECTION (Notion) that is not merely waste — re-presenting an already-rotated
#: token revokes the ENTIRE token family, so a burst of session spawns turns one
#: stale row into a fleet-wide logout.
#:
#: Why NO ttl: a dead grant is a CORRECTNESS fact that stays true until an
#: interactive login replaces the grant, not a cost backoff. This is also why it
#: is not stored in ``auth_credential_blocks``: ``block_credential`` clamps every
#: block to ``MAX_CREDENTIAL_BLOCK_MS`` (1h), a ceiling that is load-bearing for
#: provider quota backoff, so reusing that table would resume the storm — and the
#: family revocation — every hour. That table is also keyed by integer
#: ``credential_id`` plus a ``provider:type`` composite, where an MCP grant's
#: identity is its ``server_url``. Wrong lifetime, wrong key.
#:
#: Living in the row payload makes it atomic with the grant it describes:
#: :meth:`McpTokenStorage.clear` deletes it with the row, and
#: :meth:`McpTokenStorage.set_tokens` clears it on an INTERACTIVE token write, so
#: there is no way to leave a tombstone pointing at a token that no longer
#: exists. It needs no migration — the payload already round-trips unknown keys.
#:
#: :meth:`McpTokenStorage.store_refresh_result` deliberately does NOT clear it.
#: Only an interactive grant (login/reauth) may un-dead a grant: a token minted
#: by a refresh belongs to the family the marker says the authorization server
#: revoked, so a late refresh response landing after a tombstone must leave the
#: grant reading DEAD rather than resurrect it.
GRANT_DEAD_AT_KEY = "grant_dead_at"

#: Payload key recording that THIS row's refresh token was PRESENTED by an
#: exchange whose outcome never arrived — the request may have been written and
#: the token spent, and re-presenting a spent token is the reuse-detection POST
#: that revokes the whole family. Value is
#: ``{"digest": <short digest of the presented token>, "at": <epoch seconds>}``.
#:
#: Why a DIGEST rather than the token: the marker only ever answers "is the
#: token in this row the one an exchange is unsure about?", and a digest keeps
#: the payload from carrying the same live credential twice. It is not a
#: confidentiality measure — the token itself lives in the same row.
#:
#: Why it is ARMED BEFORE THE POST: a process that dies mid-request takes its
#: in-memory knowledge with it, and the next boot would present a token that may
#: already be spent. The write-ahead marker is the only channel that survives
#: that death. The price is a false positive in the narrow window between the
#: arm and the wire, which is why the marker EXPIRES
#: (:data:`UNCONFIRMED_SEND_TTL_S`) and why a connect-phase failure clears it.
GRANT_UNCONFIRMED_SEND_KEY = "grant_refresh_unconfirmed"

#: Outcome of one refresh attempt. Seven-valued rather than ``bool`` because each
#: refusal has its OWN truthful user-visible message (lock contention, a budget
#: overrun, a server that answered without a token, a token endpoint that could
#: not be reached, a possibly-spent token needing a fresh sign-in) and because
#: the coordinator must tell a DEAD grant (never present this token again) from a
#: merely FAILED one (transient; today's behaviour is correct).
#: CAUTION: every member is a truthy string — every call site must compare
#: against a member explicitly, never test truthiness.
RefreshOutcome = Literal[
    "refreshed",
    "failed",
    "dead",
    "contended",
    "overran",
    "unacknowledged",
    "unreachable",
]

#: Refresh this far BEFORE the stored access token's deadline. A connect that
#: starts with a token dying in ten seconds would otherwise open with a 401 and
#: lean on the in-flow refresh at the worst possible moment; spending the
#: refresh grant proactively keeps the first request authenticated.
REFRESH_SKEW_S = 60.0

#: Bound on one proactive refresh's HTTP round trips (metadata discovery plus
#: the token POST). A slow authorization server must not park the connect —
#: the startup gate defers us, and the breaker bounds retries.
REFRESH_HTTP_TIMEOUT_S = 10.0

#: How long PAST the refresh budget a token response is still accepted and
#: persisted. What it bounds, precisely: the response READ of the exchange task
#: once the connect that started it has given up waiting. It is not a connect
#: budget. The exchange still OWNS the refresh lock while this grace runs (the
#: lock is acquired and released INSIDE the exchange task — see
#: :func:`_perform_refresh_exchange` — precisely so a waiter's cancellation or
#: timeout cannot free it mid-POST), so a holder's critical section is bounded by
#: ``REFRESH_HTTP_TIMEOUT_S + REFRESH_LATE_RESPONSE_GRACE_S``; that is longer
#: than ``LOCK_ACQUIRE_TIMEOUT_S``, deliberately, and the derivation comment
#: there states why giving up instead of waiting is safe. A CANCELLED connect
#: detaches immediately and never waits for this grace, so teardown latency is
#: not paid out of the budget any more.
#:
#: Finite on purpose: a server that never answers must not leave a task reading
#: a socket for the life of the process.
REFRESH_LATE_RESPONSE_GRACE_S = 30.0

#: How long a "presented but unacknowledged" send marker stays live (see
#: :data:`GRANT_UNCONFIRMED_SEND_KEY`).
#:
#: The marker exists to stop us re-presenting a token an exchange may already
#: have spent, and the price of believing it is one interactive sign-in, so it
#: must be BOUNDED. A marker armed in the window between the marker write and
#: the request reaching the wire (a crash, a refused connection) describes a
#: token nothing ever presented, and a marker that never expired would suppress
#: that server's refresh until the user happened to re-auth — the "bricked
#: server" failure mode. An hour covers a restart, a slow session or a user who
#: stepped away, and after it lapses we present the stored token again, which is
#: exactly the behaviour every boot had BEFORE this marker existed. So the
#: expiry can only degrade to the status quo ante, never to something worse.
UNCONFIRMED_SEND_TTL_S = 3600.0

#: Why a refresh was refused — ONE STABLE CODE per truthfully different
#: situation. This block is the SEAM the user-visible wording hangs off: the auth
#: layer carries the code on the exception (:attr:`McpRefreshContendedError`'s
#: ``reason_code`` / :attr:`McpAuthRequiredError.reason_code`) and the manager —
#: the only layer that knows the server's NAME — composes the short rendered
#: text from it (:meth:`McpManager._auth_failure_text`). Nothing here is copy.
#:
#: The split exists because one sentence covered all of these and was therefore
#: untrue on most of them. A first attempt at splitting reworded the verbose
#: sentence instead, which does not work at any terminal width: that sentence
#: opens with ``MCP OAuth token refresh for <full server URL>``, which is ~55 of
#: a 58-cell toast card, so the distinguishing clause was always the part
#: tail-truncated away and all three refusals rendered byte-identically at 100
#: columns and below (design review D1/D2). A CODE travels independently of the
#: sentence, so the rendered text can be short while the sentence stays verbose
#: where verbose is right — ``str(exc)`` and the logs.
#:
#: The set is closed and short: the manager re-voices a transport-mangled
#: cancellation from these and the tests assert the mapped copy verbatim.
#:
#: * ``REFRESH_REFUSAL_LOCK`` — no exclusivity, so nothing was presented; another
#:   session is refreshing this grant right now.
#: * ``REFRESH_REFUSAL_INFLIGHT`` — the exchange outran the connect's budget and
#:   is still running; a late rotation will still be persisted.
#: * ``REFRESH_REFUSAL_ENDPOINT`` — the exchange ran and the authorization
#:   server's answer carried no usable token.
#: * ``REFRESH_REFUSAL_UNREACHABLE`` — the request never reached the wire (httpx's
#:   own pre-send taxonomy, see :func:`_refresh_request_never_sent`), so nothing
#:   was presented and there is no rotation to keep. Split out of
#:   ``REFRESH_REFUSAL_ENDPOINT`` deliberately: an unreachable endpoint is not a
#:   server that rejected us, and saying it was (review round 2, minor 1) tells
#:   the user the grant was refused when the request never left the machine.
#: * ``REFRESH_REFUSAL_UNCONFIRMED`` — a token may already be spent (it was
#:   presented, or an earlier presentation was never acknowledged), so nothing
#:   may be presented until an interactive grant replaces it.
REFRESH_REFUSAL_LOCK = "lock"
REFRESH_REFUSAL_INFLIGHT = "inflight"
REFRESH_REFUSAL_ENDPOINT = "endpoint"
REFRESH_REFUSAL_UNREACHABLE = "unreachable"
REFRESH_REFUSAL_UNCONFIRMED = "unconfirmed"

#: Bound on ACQUIRING the cross-process refresh lock. The critical section it
#: guards is one token POST, the response read that may outlive it, and a couple
#: of SQLite reads — and since the exchange task owns the lock for all of that,
#: its worst case is ``REFRESH_HTTP_TIMEOUT_S + REFRESH_LATE_RESPONSE_GRACE_S``
#: (40s), which EXCEEDS this bound. That inversion is deliberate and safe, and
#: it is why this constant can no longer be derived as "longer than the work".
#: A waiter that gives up here does not proceed unlocked: it takes the contended
#: path (strip the in-memory refresh token, raise
#: :class:`McpRefreshContendedError`), which the manager retries on its backoff
#: ladder, and the holder's rotation still reaches the store. Presenting a
#: rotating token without exclusivity is the family-revoking POST, so waiting is
#: never the safe option; bounding the wait is.
#:
#: The POST is still capped in TOTAL wall time (``asyncio.timeout`` in
#: :func:`_refresh_oauth_token_locked`, which bounds how long the caller waits)
#: — ``httpx.Timeout(REFRESH_HTTP_TIMEOUT_S)`` alone does NOT bound a request:
#: it is per operation, and its read timeout is per socket read, so a dribbling
#: server measured 140.7s inside a nominal 10s timeout.
#:
#: Overrunning this bound therefore means the holder is either doing the one
#: thing it may legitimately do for longer than we will wait (a late response
#: read) or is not working at all — a leaked lock from a killed process, or a
#: peer wedged on something that is not our problem. Both are answered the same
#: way: give up and degrade (see :func:`_oauth_refresh_lock`, which yields a
#: FALSY handle rather than raising), because a hung connect is worse than a
#: retried one.
LOCK_ACQUIRE_TIMEOUT_S = 15.0

#: FIRST gap between non-blocking lock attempts, and the cancellation
#: granularity: a cancelled acquire abandons the retry loop within one sleep,
#: so this also bounds how long an abandoned worker lingers. Small, because the
#: overwhelmingly common contended case is a peer finishing in a moment and
#: pickup latency is what the user feels.
_LOCK_RETRY_SLEEP_S = 0.05

#: Ceiling for the backoff below. A fixed 50 ms gap would cost ~300 wakeups per
#: contended acquire per server, and six OAuth servers connecting together would
#: run six worker threads ticking for up to the full bound against a default
#: executor of 18. Backing off to a quarter second cuts that by most while
#: leaving the fast case untouched, since a lock still free after a few seconds
#: is a leaked one we are going to abandon anyway.
_LOCK_RETRY_SLEEP_MAX_S = 0.25


class McpAuthRequiredError(RuntimeError):
    """An MCP server needs an interactive OAuth grant this run cannot open.

    Raised instead of opening a browser when the connect is NON-interactive
    (ordinary session startup and auto-reconnects): a background connect that
    pops a login tab is an interruption the user never asked for, and several
    sessions starting at once would each pop one. The connect fails with an
    actionable message instead; ``/mcp login <name>`` (or
    ``local-operator mcp login <name>``) runs the same grant deliberately.

    ``detail`` names WHY the grant is unusable when the reason is specific
    enough to be worth telling the user — today only a token refresh that was
    SENT but never confirmed (:class:`McpRefreshUnconfirmedError`). The
    manager's ``_auth_required_text`` renders it as the tail of the actionable
    line, so the command still comes first. ``None`` (the default) keeps the
    generic "authorization expired" wording, which is the truthful summary for
    every other route into this error: the stored grant could not be refreshed.

    ``reason_code`` is the STABLE CODE for why (:data:`REFRESH_REFUSAL_LOCK` and
    friends), and it is the seam user-visible copy is composed from: the manager
    maps the code to short rendered text, while ``str(self)`` keeps the verbose
    technical sentence for the logs. ``None`` means "no more specific reason than
    an unusable grant". See the ``REFRESH_REFUSAL_*`` block for why the copy is
    not carried here as prose.
    """

    #: See :attr:`reason_code` in the class docstring; class-level so the
    #: subclasses that know their reason set it without touching ``__init__``.
    reason_code: str | None = None

    def __init__(
        self,
        server_url: str,
        *,
        detail: str | None = None,
        reason_code: str | None = None,
    ) -> None:
        message = f"MCP OAuth authorization required for {server_url}"
        if detail is not None:
            message = f"{message}: {detail}"
        super().__init__(message)
        self.server_url = server_url
        self.detail = detail
        if reason_code is not None:
            self.reason_code = reason_code


class McpRefreshContendedError(RuntimeError):
    """A refresh was NOT performed this time, and why.

    Raised instead of falling through to the SDK's own UNLOCKED refresh, which
    reads the in-memory token directly and would present a possibly-already-
    rotated refresh token — the family-revoking request this subsystem exists to
    prevent. Deliberately NOT a subclass of :class:`McpAuthRequiredError` and
    deliberately never raised for a grant that is known dead: this is a
    transient outcome, so the manager's generic reconnect arm retries it with
    backoff and never takes an auth block on it. The in-memory refresh token is
    stripped before the raise (see
    ``_RefreshCoordinatingOAuthProvider._refuse_unlocked_refresh``) so the
    suppressed SDK refresh cannot run either way.

    One verbose SENTENCE per reason, and one stable CODE: the code is what the
    manager turns into user-visible text (this string opens with the server's
    full URL and is ~55 cells before it says anything distinguishing, so it can
    never be the rendered copy — see the ``REFRESH_REFUSAL_*`` block), while the
    sentence below stays for ``str(exc)`` and the logs, where the URL and the
    wording are exactly what a maintainer wants. The sentences also deliberately
    do NOT promise a retry: the mid-session reconnect path does retry with
    backoff, but the startup gate schedules none, so the log sentence may not
    claim one either.

    This type reaches the manager INDIRECTLY. The MCP streamable-HTTP transport
    runs the request inside anyio cancel scopes, so an exception raised out of
    the auth flow is not delivered — it arrives at ``_connect_server`` as a bare
    ``CancelledError('Cancelled via cancel scope …')``. The provider therefore
    arms :data:`REFRESH_CONTENTION` immediately before raising, and the manager
    re-voices exactly that cancellation as this error. Callers that drive
    ``async_auth_flow`` without the transport (tests, and any future in-process
    caller) see the raise itself.
    """

    def __init__(self, server_url: str, *, reason_code: str = REFRESH_REFUSAL_LOCK) -> None:
        if reason_code == REFRESH_REFUSAL_INFLIGHT:
            message = (
                f"MCP OAuth token refresh for {server_url} did not finish in time; "
                "a rotation is kept if the response lands"
            )
        elif reason_code == REFRESH_REFUSAL_ENDPOINT:
            message = (
                f"MCP OAuth token refresh for {server_url} was answered without a " "usable token"
            )
        elif reason_code == REFRESH_REFUSAL_UNREACHABLE:
            message = (
                f"MCP OAuth token refresh for {server_url} could not be sent: the "
                "token endpoint was unreachable, so no token was presented"
            )
        else:
            message = (
                f"MCP OAuth token refresh for {server_url} was skipped: another "
                "session holds the refresh lock"
            )
        super().__init__(message)
        self.server_url = server_url
        self.reason_code = reason_code


class McpRefreshUnconfirmedError(McpAuthRequiredError):
    """A refresh token was PRESENTED by an exchange whose outcome never arrived.

    The one thing a rotating provider must never see again is a token that may
    already have been spent: re-presenting it is the reuse-detection POST that
    revokes the entire family, every session included. So the refresh path
    refuses to POST while its row still holds a token an unacknowledged exchange
    presented (see :data:`GRANT_UNCONFIRMED_SEND_KEY`), and this error is the
    honest alternative to a retry: a transient retry would spend the token
    again, so the recovery is one interactive sign-in.

    Subclassing :class:`McpAuthRequiredError` rather than standing alone is the
    point, not a shortcut: the disposition IS "this server needs an interactive
    grant", so it takes the auth block, the actionable toast and the abandoned
    auto-reconnect that every other auth requirement takes. ``detail`` is what
    keeps the toast truthful — a refresh that was SENT but never confirmed is not
    an expired authorization — and it is kept SHORT because it is the tail of a
    line the startup toast then clamps (`refresh unconfirmed`, where the previous
    sentence-length wording was itself truncated off the card).
    """

    def __init__(self, server_url: str) -> None:
        super().__init__(
            server_url,
            detail="refresh unconfirmed",
            reason_code=REFRESH_REFUSAL_UNCONFIRMED,
        )
        # We necessarily HOLD a grant for this server (the refusal exists because
        # its stored refresh token may have been spent), so the user-visible
        # command is ``reauth`` — replace the credential — not ``login``, which
        # would leave the stale one in place. Carried on the error rather than
        # left to ``_auth_required_text``'s store lookup for the reason that
        # helper documents: the lookup answers about the DEFAULT store, while
        # this fact was established against the store this connect is actually
        # using (F4).
        self.has_stored_grant = True


class McpAuthChallengeError(RuntimeError):
    """An HTTP MCP server refused the connect with 401/403.

    Distinct from :class:`McpAuthRequiredError`, which means "we HAVE an OAuth
    config and the grant needs a browser". This one means "the transport was
    rejected as unauthorized", and it exists because the SDK erases that fact:
    a 401 the provider cannot resolve surfaces from ``session.initialize()`` as
    a generic ``MCPError(-32603, 'Server returned an error response')`` (or
    ``-32001, 'unauthorized access'``) with **no status code anywhere on the
    exception** — verified against the live GitLab, LaunchDarkly, Datadog and
    Minerva QA endpoints. Those two strings are exactly what the user
    screenshotted, and neither says what to do.

    ``oauth_available`` records whether RFC 9728 / RFC 8414 discovery actually
    found an authorization server for this URL. It drives the WORDING and must
    never be guessed: a server can 401 with no ``WWW-Authenticate`` header at
    all (Datadog does), and promising ``/mcp login`` when we found no endpoint
    would send the user at a command that cannot work.

    ``has_stored_grant`` is what makes ``login`` vs ``reauth`` truthful: a
    server we have never held a credential for needs a first grant, while one
    whose stored grant just got rejected needs it replaced. The manager reads
    the credential store to set this rather than inferring it from the error.
    """

    def __init__(
        self,
        server_url: str,
        *,
        status_code: int,
        oauth_available: bool,
        has_stored_grant: bool,
    ) -> None:
        super().__init__(f"MCP server at {server_url} refused the connection ({status_code})")
        self.server_url = server_url
        self.status_code = status_code
        self.oauth_available = oauth_available
        self.has_stored_grant = has_stored_grant


class McpLoginCancelledError(RuntimeError):
    """An interactive MCP OAuth grant ended with no authorization arriving.

    Two routes land here: the human route (the browser tab was closed or the
    consent screen abandoned, surfaced once the idle guard fires) and the
    structural one (the login task was cancelled — an exclusive re-login, the
    TUI's stop-ladder, or a Ctrl+C at the CLI). The point of the dedicated
    type is the MESSAGE: the login flows catch the SDK's ``OAuthFlowError``
    and report ``str(exc)``, so the explanation has to live on the innermost
    raise or it is lost. A bare ``CancelledError`` would surface either as an
    empty ``MCP login failed for 'x':`` line or — worse inside the TUI — as
    silence, because a Textual worker cancelled by its exclusive group never
    runs the worker's exception handler at all.
    """


class AbandonedGrantError(Exception):
    """The browser round trip ended with no authorization — the human walked away.

    Distinct from :class:`McpLoginCancelledError` on purpose: THAT one is the
    user-facing receipt the login surfaces report; this one is the flow's
    internal record of WHY the grant died. The separation matters because of
    how the two endings have to travel. An abandoned grant is raised out of
    ``callback_handler`` as a raw ``asyncio.CancelledError``: the streamable-HTTP
    transport's SDK ``post_writer`` swallows any ordinary exception an auth
    handler raises (it logs and moves on), while a cancellation unwinds the
    transport's task group exactly the way a grant REQUIREMENT does — that is
    the channel the message cannot be eaten on. The flow records itself in
    :data:`ABANDONED_GRANTS` first, and the manager converts the arriving
    cancellation back into :class:`McpLoginCancelledError`.
    """


class McpCredentialDeleteError(RuntimeError):
    """A stored MCP credential was found and its deletion FAILED — the row lives.

    The distinction this type exists to make is between the two ways a removal
    can decline to remove anything, which are opposite facts about the store:

    * there was no row, so there is nothing to delete and the caller's
      "no credential is stored here" precondition is already satisfied;
    * there WAS a row, the delete was attempted, and it failed — so the old
      grant and its client registration are still on disk.

    Collapsing both into a falsey return (which :meth:`McpTokenStorage.clear`
    used to do, swallowing the store's exception into a ``logger.debug``) let
    ``mcp reauth`` treat a failed delete as "nothing was stored" and proceed
    into the login still holding the old credential: the SDK reused the
    surviving token and ``client_info``, no consent screen came back up, and
    reauth exited 0 for the account switch or scope change that silently did
    not happen. This is reachable rather than theoretical — a sibling session
    holding an EXCLUSIVE sqlite transaction is enough to make
    ``delete_credential`` raise ``OperationalError: database is locked``.

    Raised rather than returned so a caller CANNOT accidentally read it as the
    benign outcome: the two shapes no longer share a channel, and no caller has
    to string-match a human-readable message to make a security-relevant
    control-flow decision.
    """


def mcp_oauth_credential_id(server_url: str) -> str:
    """Stable logical credential id for one MCP server's OAuth grant."""
    return f"{MCP_OAUTH_CREDENTIAL_PREFIX}{server_url}"


#: The callback port every local-operator shipped with before 33441. Stored
#: client registrations pinned to it are dropped once on sight (see
#: :meth:`McpTokenStorage.get_client_info`) so they re-register / re-seed
#: against the new default instead of dead-ending at the provider with
#: ``redirect_uri_mismatch``.
LEGACY_CALLBACK_PORT = 3000

#: How long to wait for the browser launcher before assuming the page opened.
#: The stdlib's ``GenericBrowser`` WAITS on a foreground browser, so a launcher
#: still running after this is the normal case for one, not a failure.
BROWSER_OPEN_TIMEOUT_S = 5.0

#: What the launcher child runs. ``webbrowser.open``'s own return value is the
#: exit status, so the parent still learns whether a browser was found.
_BROWSER_OPEN_SNIPPET = "import sys, webbrowser; sys.exit(0 if webbrowser.open(sys.argv[1]) else 1)"


async def open_browser_quietly(url: str) -> bool:
    """Open ``url`` in a browser without letting it print over the frame.

    ``webbrowser.open`` spawns the browser with fd 1 and fd 2 INHERITED — the
    stdlib's ``GenericBrowser`` and ``BackgroundBrowser`` pass neither
    ``stdout`` nor ``stderr`` to ``Popen`` (verified in CPython 3.13's
    ``webbrowser``) — so ``xdg-open: no method available`` or a browser's
    ``Gtk-Message:`` chatter lands straight on the Textual frame. Same defect
    as an MCP server's startup banner, arriving through the login flow.

    Redirecting our OWN descriptors is not available as a fix: Textual is
    writing to fd 1 from this very process, so replacing it even briefly
    corrupts the display we are trying to protect. Instead the call is
    delegated to a short-lived Python child whose stdout and stderr are pipes
    this process owns and logs.

    Only while the console is silenced. With the terminal ours — ``local-
    operator mcp login`` — a browser's complaint on stderr is exactly what the
    user should see, and paying for an interpreter start to hide it would be
    backwards.
    """
    from local_operator.logger import console_is_silenced

    if not console_is_silenced():
        import webbrowser

        try:
            return webbrowser.open(url)
        except Exception:  # noqa: BLE001 — headless: the paste fallback carries it
            logger.debug("webbrowser.open failed", exc_info=True)
            return False

    try:
        process = await asyncio.create_subprocess_exec(
            # ``python_argv``: ``-c`` puts the cwd on ``sys.path`` too, so a
            # file named ``webbrowser.py`` sitting in the session's directory
            # shadows the stdlib module this snippet depends on (verified: the
            # child raises out of the stray file instead of opening a browser).
            # A user's OAuth login is not the place to execute whatever happens
            # to share a name with a stdlib module.
            *python_argv("-c", _BROWSER_OPEN_SNIPPET),
            url,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            # Merged: the two streams are one diagnostic here, and a single
            # pipe cannot deadlock against itself the way two unread ones can.
            stderr=asyncio.subprocess.STDOUT,
            start_new_session=True,
        )
    except Exception:  # noqa: BLE001 — no browser is a degraded login, not a crash
        logger.debug("browser launcher failed to start", exc_info=True)
        return False

    async def _drain(prefix: str) -> str:
        assert process.stdout is not None
        raw = await process.stdout.read()
        text = strip_control_sequences(raw.decode("utf-8", "replace")).strip()
        if text:
            logger.info("%s%s", prefix, text)
        return text

    drain = asyncio.ensure_future(_drain("browser launcher: "))
    try:
        await asyncio.wait_for(asyncio.shield(drain), timeout=BROWSER_OPEN_TIMEOUT_S)
    except asyncio.TimeoutError:
        # A foreground browser keeps the launcher alive for as long as its
        # window is open. Let the drain task run on so the pipe never fills and
        # blocks that browser; the page IS open, which is what the caller asks.
        return True
    await process.wait()
    return process.returncode == 0


@runtime_checkable
class StructuralAuthStore(Protocol):
    """The slice of ``providers.auth_store.AuthStore`` this module consumes.

    Redefined to the REAL store's methods (MCP-03) so a test fake mirrors
    reality: integer-keyed rows, ``provider`` column, ``identity_key`` dedupe.
    """

    def upsert_credential(self, provider: str, credential: dict[str, Any]) -> StoredCredential:
        """Insert, or update the row for the same identity; returns the row."""
        ...

    def list_credentials(
        self, provider: str | None = None, include_disabled: bool = False
    ) -> list[StoredCredential]:
        """Enabled credential rows (all providers or one), oldest first."""
        ...

    def get_credential(self, credential_id: int) -> StoredCredential | None:
        """Return one row by integer id, or ``None``."""
        ...

    def delete_credential(self, credential_id: int) -> None:
        """Remove one row entirely (``/mcp logout`` / ``mcp logout``)."""
        ...


@runtime_checkable
class ManagedAuthStore(StructuralAuthStore, Protocol):
    """A store whose lifetime the MCP manager may own, and therefore close.

    ``McpManager`` constructs its own store when none is injected; that one
    has to be released on ``disconnect_all``, so the closing surface belongs
    in the type rather than being discovered at teardown.
    """

    def close(self) -> None:
        """Release the underlying database handle."""
        ...


def _resolve_store(store: StructuralAuthStore | None) -> StructuralAuthStore | None:
    """Return ``store`` or lazily construct the real ``AuthStore``.

    The providers import is deferred: the MCP package must stay importable in
    environments where the providers stream's dependencies are unavailable.
    """
    if store is not None:
        return store
    try:
        from local_operator.providers.auth_store import AuthStore

        return AuthStore()
    except Exception:  # pragma: no cover - environment dependent
        logger.debug(
            "providers.auth_store unavailable; MCP OAuth storage disabled",
            exc_info=True,
        )
        return None


class McpTokenStorage:
    """SDK ``TokenStorage`` over the shared credential store.

    One instance per server URL: the SDK calls ``get_tokens`` / ``set_tokens``
    (and the client-info pair for dynamic registration) against this object,
    and we round-trip the pydantic models through one credential row under
    provider ``mcp-oauth`` with ``identity_key = server_url``. All reads
    tolerate a missing store or missing row by returning ``None`` (the SDK
    then starts a fresh flow).
    """

    def __init__(self, server_url: str, store: StructuralAuthStore | None = None) -> None:
        self.server_url = server_url
        self.credential_id = mcp_oauth_credential_id(server_url)
        self._store = _resolve_store(store)
        # Snapshot ``updated_at`` NOW, before anything this process does can
        # move it. The store stamps that column on every write, including the
        # client-info writes that ``wire_oauth_auth`` makes moments after
        # constructing us — so reading it later would report this process's own
        # seed as the token's issue time. See :meth:`stored_token_expiry`.
        row = self._read_row()
        self._row_updated_at_at_open: int = row.updated_at if row is not None else 0

    def _read_row(self) -> StoredCredential | None:
        """The credential row for this server URL, or ``None`` (no store/no row)."""
        store = self._store
        if store is None:
            return None
        try:
            rows = store.list_credentials(MCP_OAUTH_PROVIDER)
        except Exception:
            logger.debug("MCP token read failed for %s", self.credential_id, exc_info=True)
            return None
        for row in rows:
            if row.identity_key == self.server_url:
                return row
        return None

    def has_stored_row(self) -> bool | None:
        """Whether ANY credential row exists for this server URL.

        Three-valued on purpose, and ``None`` ("the store could not be read")
        is deliberately NOT folded into ``False`` the way :meth:`_read` folds
        it — same reasoning as :func:`mcp_logged_out_servers` and
        :meth:`grant_marker`. The caller is :func:`clear_for_reauth`, which
        uses this to decide whether a fresh grant could silently reuse
        something still on disk; for that question an unreadable store means
        "cannot rule it out", which must not read as "nothing is there".

        Asks about the ROW, not about a grant: a ``client_info``-only row is
        not a grant (see :func:`payload_carries_grant`) but it is exactly what
        short-circuits DCR on the next login, so reauth has to count it.

        No store at all is a definite ``False``: nothing can be persisted in
        that environment, so nothing can be reused either — consistent with
        :meth:`clear` reporting ``False`` there.
        """
        store = self._store
        if store is None:
            return False
        try:
            rows = store.list_credentials(MCP_OAUTH_PROVIDER)
        except Exception:
            logger.debug("MCP row lookup failed for %s", self.credential_id, exc_info=True)
            return None
        return any(row.identity_key == self.server_url for row in rows)

    def _read(self) -> dict[str, Any] | None:
        """Row payload for this server URL, or ``None`` (no store/no row)."""
        row = self._read_row()
        if row is None:
            return None
        data = row.data
        return dict(data) if isinstance(data, dict) else None

    def _write(self, creds: dict[str, Any]) -> None:
        store = self._store
        if store is None:
            return
        payload = dict(creds)
        # The store stamps ``type`` into the data it persists; carrying it
        # back on the next write would make _identity_key_for short-circuit
        # to None (api_key rows get no identity key) and INSERT a duplicate
        # row instead of updating in place.
        payload.pop("type", None)
        # The store dedupes by identity_key derived from the payload's
        # project_id (first non-empty of org_id/account_id/email/project_id);
        # pinning it to the server URL gives one row per server, upserted in
        # place on re-auth.
        payload["project_id"] = self.server_url
        try:
            store.upsert_credential(MCP_OAUTH_PROVIDER, payload)
        except Exception:
            logger.debug("MCP token write failed for %s", self.credential_id, exc_info=True)

    # --- SDK TokenStorage protocol ---------------------------------------

    def clear(self) -> bool:
        """Delete this server's credential row entirely (logout). Returns
        ``True`` when a row existed and was removed, ``False`` when there was
        nothing to remove, and RAISES :class:`McpCredentialDeleteError` when a
        row was found and the delete failed.

        Those are three outcomes, not two, and the third must not be reported
        as the second: a caller that reads "the delete failed" as "nothing was
        stored" concludes the credential is gone while it is still on disk.
        See :class:`McpCredentialDeleteError` for what that cost ``mcp reauth``.
        This method is ours alone — the SDK's ``TokenStorage`` protocol does
        not declare it — so the contract is free to say so.

        The row carries BOTH the OAuth grant and any client registration
        (``seed_client_info`` pins it via ``project_id``, DCR writes it via
        ``set_client_info``), so removal is what makes the next login run a
        genuinely fresh grant — new consent, new registration — instead of
        silently reusing the stored client info. A pinned ``client_id`` from
        config is re-seeded by ``wire_oauth_auth`` on the next connect, so
        losing the row never strands a pinned-client server.
        """
        store = self._store
        if store is None:
            return False
        row = self._read_row()
        if row is None:
            return False
        try:
            store.delete_credential(row.id)
        except Exception as exc:
            logger.debug("MCP credential delete failed for %s", self.credential_id, exc_info=True)
            raise McpCredentialDeleteError(
                f"could not delete the stored credential for {self.server_url} ({exc}) — "
                "it is still in place, so a fresh grant would silently reuse it"
            ) from exc
        return True

    async def get_tokens(self) -> OAuthToken | None:
        """Stored access/refresh tokens as an ``OAuthToken``, or ``None``."""
        creds = self._read()
        tokens = creds.get("tokens") if creds is not None else None
        if not isinstance(tokens, dict):
            return None
        try:
            from mcp.shared.auth import OAuthToken

            return OAuthToken.model_validate(tokens)
        except Exception:
            logger.debug("Stored MCP tokens invalid for %s", self.credential_id, exc_info=True)
            return None

    async def set_tokens(self, tokens: OAuthToken) -> None:
        """Persist fresh/refreshed tokens (access + refresh together).

        This is the SDK's ``TokenStorage`` write, and after this PR it is the
        INTERACTIVE funnel: a completed browser grant, a pinned-client seed, and
        the SDK's own in-flow refresh (which the coordinating provider suppresses
        — see :class:`McpRefreshContendedError`). It stays UNCONDITIONAL because
        none of those has a rotation to compare against, and clearing the
        dead-grant tombstone is correct for all of them: they all mean "a working
        grant was obtained", which is the one thing that may un-dead a server.

        The refresh path does NOT come through here: it calls
        :meth:`store_refresh_result`, which conditions its write on the grant it
        was computed from and never clears the tombstone. Routing a refresh
        response through this method is what let a detached exchange's late HTTP
        200 resurrect a family the authorization server had already revoked.

        The issuing WALL-CLOCK time is written alongside them. ``OAuthToken``
        carries only the relative ``expires_in`` the server quoted, which is
        meaningless once the process that received it has exited — and the SDK
        reloads tokens without reloading any notion of when they die (see
        :func:`stored_token_expiry`). Recording the moment of issue is what
        turns that relative number back into an absolute deadline on the next
        launch.
        """
        creds = self._read() or {}
        creds["tokens"] = tokens.model_dump(mode="json")
        creds[TOKENS_OBTAINED_AT_KEY] = time.time()
        # Clearing the dead-grant tombstone belongs HERE rather than in the
        # ``/mcp login`` / ``/mcp reauth`` commands: this is the one funnel every
        # path that obtains a working token already goes through (a completed
        # browser grant, and a refresh that unexpectedly succeeds), so a login
        # path added later cannot forget to un-stick a suppressed server.
        creds.pop(GRANT_DEAD_AT_KEY, None)
        # A new interactive grant also answers any outstanding "was that token
        # spent?" question, so the send marker goes with it. Keeping it would
        # suppress the FIRST refresh of the brand-new grant — a false positive
        # the user pays for with another browser visit.
        creds.pop(GRANT_UNCONFIRMED_SEND_KEY, None)
        self._write(creds)

    def store_refresh_result(self, tokens: OAuthToken, *, presented_refresh_token: str) -> bool:
        """Persist a refresh response, but only into the grant it was computed from.

        The refresh path used to write through :meth:`set_tokens`, which is
        unconditional, so a response that landed after a later, better-informed
        verdict overwrote it. Two measured consequences, both the reuse-storm
        this module exists to prevent: a detached exchange's late HTTP 200
        cleared a tombstone another exchange had just written (a revoked family
        reading ALIVE, so every later boot re-POSTed it), and the same late write
        erased a rotation a peer had persisted in between. An error BEFORE this
        path could not fix it either: with two slow exchanges the tombstone's own
        compare-and-skip saw the row moved and never wrote the marker at all,
        because the row it inspected had itself been moved by an exchange the
        revocation invalidated.

        So the write re-reads the row immediately before writing and persists
        only while the payload still holds the refresh token THIS exchange
        presented.

        A row that is gone ENTIRELY is treated as the same fact, not as licence
        to re-create: an explicitly removed row stays removed, so a `/mcp logout`
        or a `/mcp reauth` that lands while our exchange is in flight is never
        silently undone by a refresh that was already on the wire (QA round 2,
        Q6). This used to re-create the row on the stated reasoning that
        "absence is not evidence that somebody replaced the grant" — and that
        half is still true, which is why the three-valued
        :meth:`has_stored_row` decides it: a DEFINITE absence is an act by
        another writer and this write is dropped with an INFO line naming the
        writer and the reason, while an UNREADABLE store
        (:meth:`has_stored_row` returns ``None``) is not evidence of removal
        either and keeps the old behaviour, the rotation is still written. That
        direction is deliberate and is the one the family's safety rests on: a
        rotation we drop while the spent token stays in the row is re-presented
        by the next refresh, which is the reuse-detecting POST that revokes
        every session. Dropping a rotation costs at most one interactive
        sign-in; keeping a spent token costs the whole family.

        This is now symmetric with :meth:`mark_grant_dead`, which declines to
        write on a payload that no longer holds the token it rejected. Two
        writers racing a removal must give the same answer, and before this fix
        they gave opposite ones.

        Two disciplines separate this from :meth:`set_tokens`:

        * the dead-grant marker is NEVER cleared here. Only an interactive grant
          may un-dead a grant; a refresh-derived token is exactly the thing the
          marker says the authorization server revoked, so clearing it would
          resurrect a family it has already killed. The write still happens —
          the rotation is real and worth keeping for support — and the marker
          keeps it from ever being presented.
        * a dropped write is logged at INFO, naming this writer and the reason.
          A lost rotation is what support has to be able to read out of the log
          after an incident; it is never a debug detail.

        The first-grant path is unaffected: ``set_tokens`` still creates the row
        for a completed interactive grant. A refresh can only ever run against a
        row that already held the token it presented, so absence here always
        means a REMOVAL that raced us, never a first grant.

        Caller: :func:`_perform_refresh_exchange`, which holds the refresh lock
        across this call, so the only writers it races are the ones that
        deliberately do not take the lock (a completed interactive login, the
        SDK's client-info writes, a `/mcp logout`).

        Returns ``True`` when the response was written.
        """
        creds = self._read()
        if creds is None and self.has_stored_row() is False:
            logger.info(
                "MCP token refresh for %s was NOT persisted: the credential row was "
                "removed while this exchange was in flight (a logout or reauth), so "
                "the removal is honoured rather than re-created "
                "[writer: McpTokenStorage.store_refresh_result]",
                self.server_url,
            )
            return False
        if creds is not None and not _payload_holds_refresh_token(creds, presented_refresh_token):
            logger.info(
                "MCP token refresh for %s was NOT persisted: the stored grant moved on "
                "(another writer rotated it first), so the newer state is left "
                "untouched [writer: McpTokenStorage.store_refresh_result]",
                self.server_url,
            )
            return False
        creds = creds or {}
        creds["tokens"] = tokens.model_dump(mode="json")
        creds[TOKENS_OBTAINED_AT_KEY] = time.time()
        # The exchange DID get an answer, so any write-ahead send marker is
        # resolved by construction: the response is the acknowledgement.
        creds.pop(GRANT_UNCONFIRMED_SEND_KEY, None)
        if GRANT_DEAD_AT_KEY in creds:
            logger.info(
                "MCP token refresh for %s persisted a rotation while its grant is marked "
                "dead; the dead-grant marker is KEPT so the token is never presented "
                "[writer: McpTokenStorage.store_refresh_result]",
                self.server_url,
            )
        self._write(creds)
        return True

    def mark_send_unconfirmed(self, presented_refresh_token: str) -> None:
        """Arm the write-ahead marker for a token an exchange is about to present.

        Called BEFORE the POST is written, while the refresh lock is held: that
        ordering is the whole point. A process that dies between the write and
        the response takes all in-memory knowledge with it, and the next boot
        would present a token the server may already have spent — the
        reuse-detection POST that revokes the family. The marker is the only
        channel that survives that death.

        Best-effort like every other write here: a store failure must not stop
        the POST, and losing the marker just restores the pre-marker behaviour.
        """
        creds = self._read() or {}
        creds[GRANT_UNCONFIRMED_SEND_KEY] = {
            "digest": _refresh_token_digest(presented_refresh_token),
            "at": time.time(),
        }
        self._write(creds)

    def clear_send_unconfirmed(self) -> None:
        """Resolve the send marker: the exchange reached a DEFINITIVE answer.

        Called only for an answer that proves our presented token was not left
        spent (see :func:`_answer_resolves_send_marker`), and for a connect-phase
        failure (the token never reached the wire, so nothing was spent).
        Deliberately NOT called for the two other shapes, which is the whole
        point of the marker:

        * the request was written and no answer arrived — the state the marker
          describes;
        * an answer that does not prove anything about our token, such as a
          ``5xx`` or a ``429``. A provider that commits a rotation and THEN
          fails the response leaves the row holding a spent token, so clearing
          the marker there would let the next connect re-present it: the
          reuse-detection POST that revokes the whole family (review round 2,
          minor 2). Keeping it costs at most one interactive sign-in — and since
          a marked token is never presented again, it can also never be
          double-spent, which is the property that matters more.
        """
        creds = self._read()
        if creds is None or GRANT_UNCONFIRMED_SEND_KEY not in creds:
            return
        creds.pop(GRANT_UNCONFIRMED_SEND_KEY, None)
        self._write(creds)

    def send_unconfirmed(self, refresh_token: str | None = None) -> bool:
        """Whether a LIVE marker says ``refresh_token`` may already be spent.

        ``refresh_token`` defaults to the stored one, which is the question the
        refresh path asks before it POSTs. A marker whose digest does not match
        is STALE — a peer has rotated the row since — and is cleared on the way
        past rather than believed: suppressing a healthy, newly rotated token
        because of an unrelated send is a false positive the user pays for with a
        browser visit. Expiry is applied the same way, so a marker that never
        resolved cannot suppress a server's refresh indefinitely (see
        :data:`UNCONFIRMED_SEND_TTL_S`).

        Read-modify-write, like :meth:`mark_grant_dead` and for the same reason:
        clearing a stale marker is part of answering the question, and a stale
        marker left in place would be re-evaluated (and re-cleared) forever.
        """
        creds = self._read()
        if creds is None:
            return False
        marker = creds.get(GRANT_UNCONFIRMED_SEND_KEY)
        if not isinstance(marker, dict):
            if marker is not None:
                # A malformed marker cannot be compared against anything, so it
                # can never be believed; drop it rather than keep consulting it.
                self.clear_send_unconfirmed()
            return False
        target = refresh_token
        if target is None:
            stored = creds.get("tokens")
            target = stored.get("refresh_token") if isinstance(stored, dict) else None
        at = marker.get("at")
        digest = marker.get("digest")
        live = (
            isinstance(at, (int, float))
            and not isinstance(at, bool)
            and 0 <= time.time() - at <= UNCONFIRMED_SEND_TTL_S
        )
        if live and isinstance(target, str) and target and digest == _refresh_token_digest(target):
            return True
        self.clear_send_unconfirmed()
        return False

    def grant_marker(self) -> tuple[float, bool] | None:
        """``(tokens_obtained_at, grant_is_dead)``, or ``None`` if unreadable.

        The identity of the grant currently stored for this server, read in ONE
        row fetch. Callers use it to answer "has somebody obtained a different
        grant since I last looked?" without caring what the grant is.

        ``None`` means the store could not be read, and is deliberately NOT a
        tuple: a sentinel VALUE would compare unequal to the real marker either
        side of a transient failure, so an unreadable store would look exactly
        like a peer's re-auth. For a caller that retries on movement that turns
        a broken store into a refresh-token retry storm — which, against a
        provider running reuse detection, revokes the whole token family (see
        :data:`GRANT_DEAD_AT_KEY`). Unknown has to be representable as unknown.

        Read-only, and the single row read is the point: deriving both fields
        from one fetch costs a third of what calling ``_read_row`` and
        :meth:`grant_is_dead` separately does, and it means the two halves
        describe the SAME instant rather than two reads a write may fall
        between.

        It reads the store directly rather than through :meth:`_read_row`, which
        cannot serve this caller: ``_read_row`` deliberately converts a store
        FAILURE into the same ``None`` it returns for a row that is simply
        ABSENT, and every other caller wants that (a missing grant and an
        unreadable one both mean "start a fresh flow"). Here the two must stay
        distinguishable — absent is the stable, knowable marker ``(0.0, False)``
        that moves when a peer writes a grant, while unreadable is no
        information at all. Collapsing them is what turns a broken store into a
        retry storm.
        """
        store = self._store
        if store is None:
            # No store configured is a KNOWN state, not a failed read: there is
            # no grant, and there never will be one until a store appears.
            return (0.0, False)
        try:
            rows = store.list_credentials(MCP_OAUTH_PROVIDER)
        except Exception:  # noqa: BLE001 — an unreadable store is "unknown", never a value
            logger.debug("MCP grant marker read failed for %s", self.credential_id, exc_info=True)
            return None
        row = next((r for r in rows if r.identity_key == self.server_url), None)
        data = row.data if row is not None and isinstance(row.data, dict) else {}
        if not isinstance(data, dict):
            return (0.0, False)
        obtained = data.get(TOKENS_OBTAINED_AT_KEY)
        obtained_at = (
            float(obtained)
            if isinstance(obtained, (int, float)) and not isinstance(obtained, bool)
            else 0.0
        )
        dead = data.get(GRANT_DEAD_AT_KEY)
        is_dead = isinstance(dead, (int, float)) and not isinstance(dead, bool) and dead > 0
        return (obtained_at, is_dead)

    def grant_is_dead(self) -> bool:
        """Whether this row's refresh token is a known-dead grant.

        ``True`` only when :meth:`mark_grant_dead` recorded an ``invalid_grant``
        rejection that no later :meth:`set_tokens` has cleared. Tolerates a
        missing store, row, or key by returning ``False``, like every other read
        here: an unreadable store must never suppress a refresh that might work.
        """
        creds = self._read()
        if creds is None:
            return False
        marker = creds.get(GRANT_DEAD_AT_KEY)
        return isinstance(marker, (int, float)) and not isinstance(marker, bool) and marker > 0

    def mark_grant_dead(self, *, rejected_refresh_token: str | None = None) -> bool:
        """Record that this row's refresh token was rejected as ``invalid_grant``.

        Best-effort: a store failure here must not break the connect that is
        already failing over a dead grant, so it degrades to writing nothing
        rather than raising. See :data:`GRANT_DEAD_AT_KEY` for why this carries
        no expiry.

        ``rejected_refresh_token`` makes the FRESHNESS RE-READ a
        compare-and-skip, and it exists because the marker is a whole-payload
        read-modify-write: a sibling that persisted a fresh rotation between
        our read and our write otherwise had a live token rewritten as dead (or
        its token erased by our pre-rotation snapshot) and stayed suppressed
        until an interactive login. Re-reading here and writing only when the
        row still holds the token the server rejected keeps the write inside
        the window it was computed in. It mirrors the check-then-write
        convention ``providers.auth_store.AuthStore`` already uses, rather than
        introducing a second write primitive.

        Returns ``True`` when the marker was written, ``False`` when it was
        skipped or the write failed.

        Its caller :func:`_perform_refresh_exchange` holds
        :func:`_oauth_refresh_lock` for the whole call, and writes through the
        same lock the FRESHNESS-conditioned sibling write uses
        (:meth:`store_refresh_result`), so the two can never disagree about
        which writer owns the grant; the remaining gap is against writers that
        deliberately do NOT take the lock — the interactive-login completion and
        the client-info writes that go through the SDK — so this narrows the
        window to the microseconds between the compare and the write instead of
        the whole exchange. The gap used to be bounded by
        :data:`REFRESH_LATE_RESPONSE_GRACE_S`, not by microseconds, because the
        detached exchange wrote with no lock at all; it does not any more.
        """
        try:
            creds = self._read() or {}
            if rejected_refresh_token is not None and not _payload_holds_refresh_token(
                creds, rejected_refresh_token
            ):
                return False
            creds[GRANT_DEAD_AT_KEY] = time.time()
            self._write(creds)
            return True
        except Exception:  # noqa: BLE001 — marking is an optimisation, never a gate
            logger.debug(
                "MCP dead-grant marker write failed for %s", self.credential_id, exc_info=True
            )
            return False

    def stored_token_expiry(self) -> float | None:
        """Epoch seconds at which the stored access token expires, if knowable.

        ``None`` means "no opinion" — no row, no token, or a token the server
        quoted no lifetime for — and callers must then leave the SDK's own
        default (treat as valid) alone: a provider that issues non-expiring
        tokens and no refresh token would otherwise be forced through a full
        re-authorization on every launch.

        Legacy rows written before :meth:`set_tokens` recorded the issue time
        fall back to the ``updated_at`` this instance snapshotted when it opened
        (milliseconds). That is an UPPER BOUND on the issue time, not the issue
        time: the store stamps the column on every write to the row, and
        client-info writes touch the same row without touching the tokens. So
        the fallback can read a genuinely expired token as live — never the
        reverse — and the cost of being wrong is the one browser grant that used
        to happen every launch. It is self-healing: that grant writes
        ``tokens_obtained_at``, and the row never takes the fallback again.

        Using the snapshot rather than a fresh read is what keeps THIS process's
        own ``seed_client_info`` from resetting the bound to "now" before we can
        read it, which would make the migration a guaranteed no-op for exactly
        the pinned-client servers that seed exists to serve.
        """
        row = self._read_row()
        if row is None:
            return None
        data = row.data if isinstance(row.data, dict) else {}
        tokens = data.get("tokens")
        if not isinstance(tokens, dict):
            return None
        expires_in = tokens.get("expires_in")
        if not isinstance(expires_in, (int, float)) or expires_in <= 0:
            return None
        obtained_at = data.get(TOKENS_OBTAINED_AT_KEY)
        if not isinstance(obtained_at, (int, float)) or obtained_at <= 0:
            if self._row_updated_at_at_open <= 0:
                return None
            obtained_at = self._row_updated_at_at_open / 1000.0
        return float(obtained_at) + float(expires_in)

    async def get_client_info(self) -> OAuthClientInformationFull | None:
        """Stored client registration (DCR result or pinned config), or ``None``.

        A stored registration whose redirect URIs still point at the legacy
        :data:`LEGACY_CALLBACK_PORT` is stale by definition — the runtime now
        advertises a different loopback port, so that registration can never
        complete a grant (the provider rejects the authorization redirect with
        ``redirect_uri_mismatch``, and because ``client_info`` is present the
        SDK never re-runs DCR — a dead-end with no in-app recovery). Dropping
        it here, before the SDK reads it, lets the flow re-register (DCR
        servers) or re-seed (pinned ``client_id`` servers, which
        :meth:`seed_client_info` rewrites on the next login) against the new
        redirect URI.
        """
        creds = self._read()
        info = creds.get("client_info") if creds is not None else None
        if not isinstance(info, dict):
            return None
        if self._redirect_uris_use_legacy_port(info):
            logger.info(
                "Discarding MCP client registration for %s: its redirect URIs "
                "still target the legacy :%d callback, which can no longer "
                "complete a grant; it will re-register on this login.",
                self.credential_id,
                LEGACY_CALLBACK_PORT,
            )
            if creds is not None:
                creds.pop("client_info", None)
                self._write(creds)
            return None
        try:
            from mcp.shared.auth import OAuthClientInformationFull

            return OAuthClientInformationFull.model_validate(info)
        except Exception:
            logger.debug(
                "Stored MCP client info invalid for %s",
                self.credential_id,
                exc_info=True,
            )
            return None

    @staticmethod
    def _redirect_uris_use_legacy_port(info: dict[str, Any]) -> bool:
        """True when any stored redirect URI targets the legacy callback port."""
        uris = info.get("redirect_uris") or []
        if not isinstance(uris, list):
            return False
        for uri in uris:
            if not isinstance(uri, str):
                continue
            try:
                port = urlparse(uri).port
            except ValueError:
                continue
            if port == LEGACY_CALLBACK_PORT:
                return True
        return False

    async def set_client_info(self, client_info: OAuthClientInformationFull) -> None:
        """Persist a dynamic-client registration (RFC 7591)."""
        creds = self._read() or {}
        creds["client_info"] = client_info.model_dump(mode="json")
        self._write(creds)

    def seed_client_info(self, client_id: str, client_secret: str | None = None) -> None:
        """Synchronously pre-seed a pinned client registration (MCP-11).

        Same persistence path as :meth:`set_client_info` but callable from
        sync wiring code: when the config supplies a ``client_id`` the SDK
        finds it via ``get_client_info`` and skips dynamic client
        registration entirely — required for providers whose redirect URI was
        registered against a fixed loopback port (pinned-redirect providers).

        ``token_endpoint_auth_method`` must be stamped here, not left at its
        ``None`` default: the SDK's ``prepare_token_auth`` only sends the
        ``client_secret`` when the method names a secret-based scheme, so a
        seed that omits it reaches the token endpoint with no secret at all
        and the provider rejects the exchange (HubSpot: ``BAD_CLIENT_SECRET``).
        Because :func:`wire_oauth_auth` re-seeds on EVERY login, a value
        hand-patched into the store is overwritten before it can be used —
        the method has to be correct at the source. ``client_secret_post``
        matches the providers that pin a client (and HubSpot's advertised
        ``token_endpoint_auth_methods_supported``); with no secret the method
        is ``none``.
        """
        from mcp.shared.auth import OAuthClientInformationFull

        info = OAuthClientInformationFull(
            client_id=client_id,
            client_secret=client_secret,
            token_endpoint_auth_method="client_secret_post" if client_secret else "none",
        )
        creds = self._read() or {}
        creds["client_info"] = info.model_dump(mode="json")
        self._write(creds)


#: Server URLs OBSERVED to answer an MCP request with 401/403 during this
#: process's lifetime, mapped to whether OAuth metadata discovery then found an
#: authorization server for them.
#:
#: Why this exists: a config imported from a foreign tool (Codex's
#: ``config.toml``, issue #367) carries only a ``url`` and no ``auth`` block,
#: because that tool holds its OAuth grants elsewhere. Nothing in the static
#: config says the server needs OAuth, so the auth-capable paths
#: (``_build_oauth_auth``, ``_ensure_oauth_fresh``, ``/mcp login``) all declined
#: it and the connect went out unauthenticated. The server's own 401 challenge
#: is the authoritative signal, so we record it when we see it and let the
#: gates consult it. This is deliberately TRANSPORT-level and names no config
#: source: any config that omits an auth block benefits identically.
#:
#: Per-process and observation-only by design. It is not a cache that has to be
#: invalidated: the durable cross-process signal that a server uses OAuth is a
#: stored credential row (see :func:`server_has_stored_grant`), which is what
#: makes a RESTART re-authenticate rather than re-observe a 401 first.
OAUTH_CHALLENGES: dict[str, bool] = {}


def record_oauth_challenge(server_url: str, *, oauth_available: bool) -> None:
    """Remember that ``server_url`` answered with an authorization challenge.

    ``oauth_available`` is whether discovery found an authorization server.
    A True observation is never downgraded by a later False: discovery is a
    network call that can fail transiently, and forgetting that a server is
    OAuth-capable would put the user back on the dead-end message.
    """
    if oauth_available or server_url not in OAUTH_CHALLENGES:
        OAUTH_CHALLENGES[server_url] = oauth_available


def server_has_stored_grant(server_url: str, store: StructuralAuthStore | None = None) -> bool:
    """True when an OAuth credential row already exists for ``server_url``.

    This is the honest basis for choosing between ``/mcp login`` (we have
    never held a grant here) and ``/mcp reauth`` (we hold one and the server
    just rejected it), and it is also the DURABLE signal that a server without
    an explicit ``auth`` block authenticates over OAuth — the observed-challenge
    ledger above is per-process, so without this a restart would connect
    unauthenticated again and ignore a perfectly good stored token.

    A row is NOT enough: the same row also carries ``client_info``, the dynamic
    client registration the SDK writes when it merely DISCOVERS a server, well
    before any user authorizes. Counting that as a grant made a server nobody
    had ever logged into report "authorization expired — run /mcp reauth",
    sending the user to replace a credential that was never issued. Only an
    actual token payload counts.

    Never raises: an unreadable store degrades to "no stored grant", which
    costs a wording nuance rather than a connect.
    """
    try:
        payload = McpTokenStorage(server_url, store)._read()
    except Exception:  # noqa: BLE001 — the store is best-effort here
        logger.debug("stored-grant lookup failed for %s", server_url, exc_info=True)
        return False
    return payload_carries_grant(payload)


def payload_carries_grant(payload: dict[str, Any] | None) -> bool:
    """Whether a credential row payload holds an actual OAuth grant.

    Split out of :func:`server_has_stored_grant` so the grant-vs-registration
    distinction it documents lives in ONE place. ``mcp_logout_server`` needs
    the same answer about a row it has already opened a storage for, and
    re-deriving the test there would put this rule in two places — the drift
    risk that matters most for a predicate whose whole point is that a
    ``client_info``-only row is not a grant.
    """
    if not payload:
        return False
    tokens = payload.get("tokens")
    if not isinstance(tokens, dict):
        return False
    return bool(tokens.get("access_token") or tokens.get("refresh_token"))


def _payload_holds_refresh_token(payload: dict[str, Any] | None, refresh_token: str) -> bool:
    """Whether a payload's stored grant still carries ``refresh_token``.

    The predicate behind :meth:`McpTokenStorage.mark_grant_dead`'s
    compare-and-skip, kept next to :func:`payload_carries_grant` because it is
    the same question asked about a specific token rather than any token: has
    the grant this write was computed from been replaced since?

    False for a payload whose grant was removed entirely (a logout or a
    ``/mcp reauth`` that deleted the row). That is deliberate: the writers that
    remove a grant do NOT take :func:`_oauth_refresh_lock`, so a tombstone
    written after one would re-create a row the user just asked us to forget.
    """
    if not isinstance(payload, dict):
        return False
    tokens = payload.get("tokens")
    if not isinstance(tokens, dict):
        return False
    return tokens.get("refresh_token") == refresh_token


def _refresh_token_digest(refresh_token: str) -> str:
    """Short stable digest of one refresh token, for the send marker.

    Never a credential in its own right: it only ever answers "is the token in
    this row the one an exchange presented?", so keeping the digest rather than a
    second copy of the token keeps the payload from carrying the same live secret
    twice. Truncated, because the question compares a handful of our own tokens
    rather than searching an adversarial keyspace.
    """
    import hashlib

    return hashlib.sha256(refresh_token.encode("utf-8")).hexdigest()[:16]


def _refresh_request_never_sent(exc: BaseException) -> bool:
    """Whether an httpx failure happened BEFORE the request reached the wire.

    The distinction is load-bearing and httpx's own, not a guess: a refresh
    token presented to a rotating provider is spent by the REQUEST, so a failure
    that happened before the request was written leaves the token untouched and
    is an ordinary transient retry, while one that happened after may have spent
    it and must not be retried without an interactive sign-in.

    Pre-send, per httpx's taxonomy: ``ConnectError`` (refused, DNS, TLS
    handshake), ``ConnectTimeout`` / ``PoolTimeout`` (waiting for a connection
    or a pool slot), ``UnsupportedProtocol`` and ``LocalProtocolError`` (the
    request was never even formatted for a socket). Everything else —
    ``ReadTimeout``, ``ReadError``, ``WriteError``, ``WriteTimeout``,
    ``RemoteProtocolError`` — can have been written, so it is treated as suspect.
    That asymmetry is deliberate: a wrongly-suspect failure costs one interactive
    sign-in, a wrongly-trusted one costs the whole token family.

    The annotation is ``BaseException`` because the predicate is TOTAL over
    exceptions by construction: it answers "was this demonstrably pre-send?"
    and anything it does not recognise — including ``asyncio.CancelledError`` —
    answers ``False``, i.e. suspect. That is the conservative answer, and it is
    why the caller needs no special case for cancellation: the caller's ``except
    (httpx.HTTPError, TimeoutError)`` cannot catch a ``CancelledError``, so a
    cancelled send keeps the marker by PROPAGATION, while this predicate — were
    it ever asked — would keep it too. This docstring must not claim a
    classification the caller never requests (review round 2, nit).
    """
    import httpx

    return isinstance(
        exc,
        (
            httpx.ConnectError,
            httpx.ConnectTimeout,
            httpx.PoolTimeout,
            httpx.UnsupportedProtocol,
            httpx.LocalProtocolError,
        ),
    )


def _answer_resolves_send_marker(status_code: int, *, invalid_grant: bool) -> bool:
    """Whether a token-endpoint ANSWER proves our presented token is not spent.

    The ONE place the marker's clearing rule lives, so no future status-code
    tweak can quietly re-open the family-revoking POST (review round 2, minor
    2). Clearing the marker says "the next connect may present this token
    again", so it is allowed only for an answer that settles the question:

    * ``200`` with a body already parsed into a usable token — the exchange ran,
      so the presented token is consumed and the row holds its replacement.
      :func:`_perform_refresh_exchange` asks only after the body parsed; the
      unreadable-200 shape is ``unacknowledged`` by definition and never reaches
      here;
    * ``400`` whose body names ``invalid_grant`` — the authorization server
      answered that this token is not acceptable (spent, revoked, or already
      rotated away), and the caller records the tombstone that is the outcome.
      There is nothing left for the marker to protect.

    Everything else is an UNKNOWN answer and keeps the marker: ``5xx`` and
    ``429`` (a provider that commits the rotation and THEN fails the response
    would otherwise leave a spent token re-presentable), and any other 4xx that
    is not an ``invalid_grant`` (a refusal that is not readable as a statement
    about our token). The cost is bounded and stated: the marker expires
    (:data:`UNCONFIRMED_SEND_TTL_S`), so an authorization-server outage can cost
    the affected grant ONE interactive sign-in rather than risking a fleet-wide
    revocation. That is the trade this module takes on purpose.
    """
    if status_code == 200:
        return True
    return status_code == 400 and invalid_grant


def _stored_token_is_fresh(storage: "McpTokenStorage", tokens: Any) -> bool:
    """Whether the STORED access token is still comfortably usable.

    The shared answer to "is a refresh needed?" for the proactive site and for
    the exchange inside its lock, so the two cannot drift: the exchange asks it
    to notice that a peer rotated the grant while it waited for the lock, and the
    proactive site asks it to skip the lock entirely in the common case.

    An expiry of ``None`` means "no opinion" (see
    :meth:`McpTokenStorage.stored_token_expiry`) and reads as fresh, because a
    provider that quotes no lifetime must not be re-authorized on a guess.
    """
    if tokens is None or not getattr(tokens, "access_token", None):
        return False
    expiry = storage.stored_token_expiry()
    return expiry is None or time.time() < expiry - REFRESH_SKEW_S


def server_rejects_oauth(cfg: MCPServerConfig) -> bool:
    """True when this config can NEVER use an OAuth grant — a static fact.

    Two shapes qualify, and both are decidable without touching the network:
    a stdio server (no transport that can carry a bearer token) and a server
    whose config explicitly declares some OTHER auth type. An ``apikey`` entry
    is a statement by the user about how this server authenticates, and
    starting an OAuth flow on it would answer a question they already
    answered — the login must stay a hard refusal there (F3).
    """
    auth = getattr(cfg, "auth", None)
    auth_type = getattr(auth, "type", None) if auth is not None else None
    if auth_type is not None and auth_type != "oauth":
        return True
    return not getattr(cfg, "url", None)


def server_is_oauth_capable(
    cfg: MCPServerConfig,
    store: StructuralAuthStore | None = None,
) -> bool:
    """Whether this server should be connected through the OAuth provider.

    Three ways to qualify, in order of authority:

    1. an explicit ``auth.type == "oauth"`` block — the local-operator format's
       own signal, and the only one that existed before;
    2. a stored OAuth credential for its URL — durable proof across restarts
       that this server authenticates with a grant we already hold;
    3. an observed 401/403 challenge whose discovery found an authorization
       server — the live signal that rescues a config which simply does not
       carry an auth block.

    A server matching none of the three stays unauthenticated, with no
    discovery and no added latency: that is what keeps a genuinely public MCP
    server (``developers.openai.com/mcp``) connecting exactly as it does today,
    and what stops us attaching an OAuth provider to every remote server on
    boot. There is deliberately no "assume yes" mode — an explicit login that
    needs an answer this cannot give asks the network for one instead, via
    :func:`probe_oauth_capability`.
    """
    if server_rejects_oauth(cfg):
        return False
    auth = getattr(cfg, "auth", None)
    if auth is not None and getattr(auth, "type", None) == "oauth":
        return True
    url = getattr(cfg, "url", None)
    if not url:
        return False
    if OAUTH_CHALLENGES.get(url):
        return True
    return server_has_stored_grant(url, store)


async def probe_oauth_capability(
    cfg: MCPServerConfig, store: StructuralAuthStore | None = None
) -> bool:
    """Answer "can this server take an OAuth grant?", asking the network if needed.

    The gate an explicit ``/mcp login`` runs. Whether a server without an auth
    block uses OAuth is not knowable from the config — that is the whole
    problem a foreign-tool import creates — so when the static evidence is
    silent this performs the one RFC 9728/8414 discovery that settles it, and
    records the result for later connects.

    This replaces an earlier "a deliberate login accepts any remote server"
    shortcut, which enabled the command for known-public and API-key servers
    (F3). Paying one metadata round trip is the honest way to keep the first
    login on a fresh import working WITHOUT claiming every URL is
    authenticable: a server that advertises no authorization server is now
    refused with the same message a stdio server gets, instead of being sent
    into a grant that cannot complete.
    """
    if server_is_oauth_capable(cfg, store):
        return True
    if server_rejects_oauth(cfg):
        return False
    url = getattr(cfg, "url", None)
    if not url:
        return False
    try:
        discovered = await discover_oauth_endpoints(url) is not None
    except Exception:  # noqa: BLE001 — a probe failure must not crash the command
        logger.debug("OAuth capability probe failed for %s", url, exc_info=True)
        return False
    if discovered:
        record_oauth_challenge(url, oauth_available=True)
    return discovered


def oauth_server_names(cwd: str | os.PathLike[str]) -> list[str]:
    """Names of configured OAuth-enabled servers, in config order.

    The ``/mcp login|reauth|logout`` argument lists are filled from this, so
    they offer exactly the servers those commands can act on — a stdio server
    has no OAuth grant to log into or out of, and offering it would be a row
    whose only outcome is a warning notice.

    A server with no explicit ``auth`` block is offered once there is EVIDENCE
    it authenticates — a stored grant, or an observed OAuth challenge — which
    is how a foreign-tool import (Codex, issue #367) reaches this list: its
    connect fails first and records the challenge, so the picker then offers
    exactly the server the user just watched fail.

    Deliberately the STRICT test, unlike the gate that executes a login: a
    suggestion list is a claim that these servers have something to log into,
    and speculatively listing every remote server would fill the picker with
    rows whose only outcome is a warning notice. Typing an unlisted name still
    works (see the TUI's ``_resolve_mcp_server``), so nothing is unreachable.
    """
    from local_operator.mcp.config import load_all_mcp_configs

    configs, _sources = load_all_mcp_configs(cwd)
    return [name for name, cfg in configs.items() if server_is_oauth_capable(cfg)]


def mcp_logout_server(
    name: str,
    cwd: str | os.PathLike[str],
    store: StructuralAuthStore | None = None,
) -> str | None:
    """Remove the stored OAuth credential for one configured server.

    Returns an error string on failure, ``None`` on success — the two callers
    (CLI and TUI) phrase their own output, so the helper reports outcomes,
    not prose. All three failure shapes — unknown name, non-OAuth config,
    nothing stored — are reported as errors, but they are DIFFERENT errors:
    a name the config does not know is a typo the user wants told about,
    while a known OAuth server holding no credential is a no-op worth
    distinguishing from a successful removal (the caller's message says
    which).

    A FAILED delete is none of those three and does not come back as a string
    at all: it raises :class:`McpCredentialDeleteError`, because the row is
    still on disk and a caller must not be able to mistake that for the benign
    "nothing was stored" outcome.

    That raise is part of the contract, so every call site handles it. Named
    rather than asserted, because "all callers handle this" is a claim that
    silently decays the moment somebody adds the next caller — and it already
    did once: ``mcp logout`` was left bare when the raise was introduced, and
    a locked store printed a stack trace where it used to print one line.

    * :func:`clear_for_reauth` catches it explicitly in both of its arms and
      converts it into the refusal that stops the login;
    * ``cli.mcp_command``'s ``logout`` branch catches it and prints one error
      line, because reaching ``main``'s generic handler means a traceback;
    * :func:`~local_operator.mcp.grants.logout_server` and
      :func:`~local_operator.mcp.grants.clear_for_reauth_server` funnel it
      through their ``except Exception`` into a notice body;
    * the TUI's ``_mcp_logout`` wraps both verbs in one ``except Exception``
      that becomes a system notice.

    The deletion goes through the REAL store (``_resolve_store(None)``), not
    the session manager's possibly-injected one: logout must remove the
    persisted row every future process will read, which is the shared
    ``auth.db`` regardless of what one session was handed.
    """
    from local_operator.mcp.config import load_all_mcp_configs

    configs, _sources = load_all_mcp_configs(cwd)
    cfg = configs.get(name)
    if cfg is None:
        return f"MCP server {name!r} is not configured"
    # Strict, like the picker: logging out is only meaningful for a server we
    # actually hold — or have observed the need for — a grant on. A stored
    # grant satisfies this on its own, which is the case that matters here.
    if not server_is_oauth_capable(cfg, store):
        return f"MCP server {name!r} does not use OAuth login"
    # Only remote configs carry ``url``; a stdio config reaching here would
    # have already failed the OAuth check above, so the getattr is a type
    # narrowing rather than a guess.
    url = getattr(cfg, "url", "")
    storage = McpTokenStorage(url, store)
    # Read the durable evidence BEFORE destroying it — see the transfer below.
    # Derived from THIS storage rather than via ``server_has_stored_grant``,
    # which would open a second ``AuthStore`` (and a second sqlite connection)
    # for a row we are about to open one for anyway.
    #
    # An unreadable store makes this False, so the transfer is skipped in
    # exactly the case where the evidence is most likely to be lost for good.
    # That is the deliberate direction and not an oversight: recording a
    # challenge we could not substantiate would write a fact we never observed
    # into an observation-only ledger, and the two reads here hit the same
    # store microseconds apart, so a read that fails while the delete below
    # succeeds is a very narrow window. Silence is the conservative error.
    had_stored_grant = payload_carries_grant(storage._read())
    if not storage.clear():
        return f"no stored credential for MCP server {name!r} — nothing to log out of"
    if had_stored_grant:
        # EVIDENCE TRANSFER, not a guess — for the three kinds of evidence and
        # why a url-only Codex import (issue #367) has only this one, see
        # :func:`server_is_oauth_capable`. The delete above is what erases it.
        #
        # That erasure is what broke ``/mcp reauth`` on those servers. Reauth
        # is a delete followed by a reconnect, and the reconnect re-asks
        # ``server_is_oauth_capable`` (via ``_build_oauth_auth``): with the row
        # gone and the ledger cold, it answered False, attached no OAuth
        # provider, and the connect went out unauthenticated and took a 401.
        # The gate that authorised the reauth had already short-circuited on
        # the very row being deleted, so nothing re-established the fact. It
        # was intermittent only because the ledger is per-process: a session
        # that had already watched this server 401 was warm and worked.
        #
        # Holding a token for ``url`` is STRONGER proof than the discovery
        # probe that normally populates this ledger — an authorization server
        # did not merely advertise itself, it issued us a grant — so recording
        # it as an observed challenge asserts only what we just read. The
        # record is deliberately made HERE, at the single point that destroys
        # the row, so no caller can delete a credential without preserving
        # what it proved (``/mcp reauth``, the CLI's ``mcp reauth``, and the
        # desktop grant runner all funnel through this function).
        #
        # Scoped to ``had_stored_grant`` on purpose: a server that qualified
        # only through a declared ``auth.type: oauth`` block keeps qualifying
        # without the ledger, and claiming a discovery result we never ran
        # would put an unverified fact in a ledger whose entries are supposed
        # to be observations.
        record_oauth_challenge(url, oauth_available=True)
    return None


def clear_for_reauth(
    name: str,
    cwd: str | os.PathLike[str],
    store: StructuralAuthStore | None = None,
    removed: list[str] | None = None,
) -> str | None:
    """Remove ``name``'s credential for a REAUTH. ``None`` when the login may
    proceed, an error string when it must not.

    ``removed`` is an out-parameter appended to only when a row was genuinely
    deleted, mirroring ``run_grant``'s ``forgotten``. It exists because this
    function now returns ``None`` for TWO outcomes — a real deletion and a
    no-op on a server holding nothing — and a caller that warns "your
    credential is gone" after a cancelled grant must not say so for the second.

    Reauth's contract is "end up authenticated having genuinely re-granted",
    which needs a weaker precondition than logout's: logout must actually
    delete something, while reauth only needs to KNOW that nothing is left for
    the coming grant to reuse. A no-op delete satisfies that; a failed one does
    not. :func:`mcp_logout_server` reports both as an error string, so a caller
    branching on ``error is not None`` refuses the first case (the bug this PR
    fixes) and a caller branching on nothing at all accepts the second (a
    successful-looking reauth over a live credential).

    This function is the ONE place that distinction is made, so the CLI's
    ``mcp reauth`` and the TUI's ``/mcp reauth`` cannot answer it differently —
    two gates disagreeing about one question is the defect this PR exists to
    remove, and shipping it in a new place would only move it.

    The decision is structural at every step, never a string match on
    :func:`mcp_logout_server`'s human-readable message: that message is prose
    for a user, and reading control flow — security-relevant control flow — out
    of it would break silently the first time somebody rewords it.

    Three refusals survive, and each is re-made here rather than inherited:

    * an unknown name (``cfg is None``) is a typo, which must never turn into
      a browser tab;
    * a statically ineligible server (stdio, or a declared non-OAuth
      ``auth.type``) can never take a grant — the F3 protection;
    * a row that is STILL THERE after everything below has tried to remove it,
      because the next login would reuse it.

    The last case needs one extra step rather than only a check. A
    ``client_info``-only row is not a grant (:func:`payload_carries_grant`), so
    on a cold ledger it does not make the server ``server_is_oauth_capable``
    and :func:`mcp_logout_server` declines to touch it — yet it is exactly what
    short-circuits DCR and denies the user the fresh registration they ran
    reauth for. Reauth therefore removes it directly here: "delete what would
    be reused" IS reauth's contract, where logout's stricter "only remove a
    server we demonstrably hold something on" correctly protects the picker.
    No evidence is lost by that directness — this path is reachable only when
    the capability test was False, which for a readable row means it carried no
    grant, so :func:`mcp_logout_server` would have had nothing to transfer.
    """
    from local_operator.mcp.config import load_all_mcp_configs

    try:
        error = mcp_logout_server(name, cwd, store)
    except McpCredentialDeleteError as exc:
        # The row was found and the delete failed, so it is still on disk.
        # Loud and terminal: this is the case that used to masquerade as
        # "nothing was stored" and let the grant reuse the old credential.
        return str(exc)
    if error is None:
        if removed is not None:
            removed.append(name)
        return None

    configs, _sources = load_all_mcp_configs(cwd)
    cfg = configs.get(name)
    if cfg is None or server_rejects_oauth(cfg):
        return error

    # Only remote configs carry ``url``; a stdio config was refused one line
    # above, so this is type narrowing rather than a guess.
    url = getattr(cfg, "url", "")
    stored = McpTokenStorage(url, store).has_stored_row()
    if stored is None:
        # Unreadable store. "Cannot rule out a surviving credential" is not
        # "there is none" — refusing costs the user a retry, while proceeding
        # costs them a reauth that silently did nothing.
        return (
            f"could not read the credential store for MCP server {name!r}, so it is "
            "unknown whether a fresh grant would reuse an existing credential"
        )
    if stored:
        storage = McpTokenStorage(url, store)
        try:
            deleted = storage.clear()
        except McpCredentialDeleteError as exc:
            return str(exc)
        if deleted and removed is not None:
            removed.append(name)
        # Re-read rather than trusting the return: the point of this gate is
        # the state of the store, not the outcome of one call, and a racing
        # writer between the two is the whole reason this refusal exists.
        if McpTokenStorage(url, store).has_stored_row() is not False:
            return (
                f"the stored credential for MCP server {name!r} is still in place after the "
                "removal, so a fresh grant would silently reuse it"
            )
    # Nothing is stored: the delete was a no-op and reauth's precondition is
    # already satisfied. This is the url-only Codex import the PR is about.
    return None


def mcp_logged_out_servers(store: StructuralAuthStore | None = None) -> set[str] | None:
    """Server URLs that still hold an ``mcp-oauth`` credential row, or
    ``None`` when the store could not be read.

    Read-only companion to :func:`mcp_logout_server` so the ``/mcp logout``
    picker can offer only servers that actually have something to remove
    (mirroring how ``/logout`` offers only providers holding a credential).
    ``None`` rather than the empty set on failure: an unreadable store is not
    the same answer as "no credentials anywhere", and the picker needs the
    difference to say so instead of rendering a bare empty list.
    """
    store = _resolve_store(store)
    if store is None:
        return None
    try:
        rows = store.list_credentials(MCP_OAUTH_PROVIDER)
    except Exception:
        logger.debug("MCP credential listing failed", exc_info=True)
        return None
    return {row.identity_key for row in rows if row.identity_key}


def parse_oauth_callback_input(raw: str) -> tuple[str, str | None, str | None]:
    """Parse the pasted callback input into ``(code, state, iss)`` (MCP-02).

    Accepts either the FULL redirect URL (``...?code=X&state=Y&iss=Z``) or a
    bare ``code state`` pair separated by whitespace. ``state`` (and ``iss``
    when present) MUST be handed back to the SDK: it validates ``state``
    against the value it generated (oauth2.py:421) and rejects the flow when
    the handler returns ``state=None``.
    """
    text = (raw or "").strip()
    if not text:
        raise RuntimeError("No authorization input provided")
    if "://" in text or text.startswith("http"):
        query = parse_qs(urlparse(text).query)
        code = (query.get("code") or [""])[0]
        state = (query.get("state") or [None])[0]
        iss = (query.get("iss") or [None])[0]
        if not code:
            raise RuntimeError(f"No authorization code found in redirect URL: {text!r}")
        return code, state, iss
    parts = text.split()
    if len(parts) == 1:
        raise RuntimeError("Bare input needs 'code state' (paste the full redirect URL instead)")
    code, state = parts[0], parts[1]
    iss = parts[2] if len(parts) > 2 else None
    return code, state, iss


#: How long the whole grant may sit waiting for the human: the browser round
#: trip and, where it is offered, the paste. A connect that reaches this path
#: on an unattended host must fail eventually, not park the connect task
#: forever.
PASTE_INPUT_TIMEOUT_S = 300.0

#: Idle bound on an INTERACTIVE grant: the longest a login waits for a browser
#: round trip that will never complete. The usual reason nothing arrives is
#: that the tab was closed or the consent screen abandoned — indistinguishable
#: from a slow human at the protocol level, so the flow has to give up on a
#: clock and say so. Sized to match the 10-minute budget both login callers
#: already allow the whole connect (``/mcp login`` and ``local-operator mcp
#: login``), so the grant now ends with an explicit "cancelled" receipt inside
#: that window instead of outliving it as a silent "logging in…" line.
INTERACTIVE_GRANT_TIMEOUT_S = 600.0

#: Hosts a redirect URI can name that THIS process is able to answer.
#:
#: A redirect URI may legitimately point anywhere; only a loopback address is
#: something we can bind. Anything else — a hosted callback, a tunnel — has to
#: fall through to the paste path, because listening would not intercept it.
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})

#: Bound on the request head we will read from the browser before giving up.
#: A callback is a GET with a short query string; anything larger is not the
#: browser we asked for, and an unbounded read on a public-ish port is a
#: memory-exhaustion invitation.
_MAX_REQUEST_HEAD_BYTES = 16 * 1024
_MAX_REQUEST_HEADERS = 64

#: How long one connection may take to send its request head.
#: A browser sends it in one packet; anything slower is a probe holding a
#: handler open, and a held handler is what makes ``wait_closed()`` hang.
_REQUEST_READ_TIMEOUT_S = 10.0

#: How long teardown may wait for in-flight handlers before abandoning them.
#: The authorization code is already in hand at that point; a lingering socket
#: is not worth stalling the connect for.
_SERVER_CLOSE_TIMEOUT_S = 1.0


#: Flows whose grant the human abandoned (idle guard fired), keyed by the
#: flow object with the abandonment time as value. This is the side channel
#: that survives the transport: ``callback_handler`` raises a raw
#: ``CancelledError`` for an abandoned grant because the SDK's ``post_writer``
#: eats ordinary exceptions, and the manager consults this registry to tell
#: "the grant died of neglect" apart from "the login task was cancelled".
#: Entries are pruned on every write; a flow object whose grant never
#: abandoned never appears here, and one that did is dropped from the map
#: when its entry is consumed or when it ages past the prune horizon.
#: How long an abandonment record may sit unread. The manager consumes it
#: within seconds of the raise; the horizon only bounds a leak for flows
#: whose connect never got as far as consulting the registry.
_ABANDONED_GRANT_TTL_S = 120.0


class AbandonedGrantLedger:
    """Flows whose grant the human abandoned (idle guard fired).

    This is the side channel that survives the transport: ``callback_handler``
    raises a raw ``CancelledError`` for an abandoned grant because the SDK's
    ``post_writer`` eats ordinary exceptions, and the manager consults the
    ledger to tell "the grant died of neglect" apart from "the login task
    was cancelled". Records are consumed by the manager (``pop``) or age out
    on the next write; keys are weak so a flow nobody consults cannot be kept
    alive by its own record.
    """

    def __init__(self) -> None:
        self._records: weakref.WeakKeyDictionary[LoopbackAuthFlow, float] = (
            weakref.WeakKeyDictionary()
        )

    def record(self, flow: LoopbackAuthFlow) -> None:
        now = time.monotonic()
        for old, at in list(self._records.items()):
            if now - at > _ABANDONED_GRANT_TTL_S:
                self._records.pop(old, None)
        self._records[flow] = now

    def pop(self, flow: LoopbackAuthFlow) -> bool:
        """Consume one flow's abandonment record; ``True`` when one was there."""
        return self._records.pop(flow, None) is not None


ABANDONED_GRANTS = AbandonedGrantLedger()

#: How long a refusal-to-refresh record may sit unread. The manager consumes it
#: within microseconds of the raise (it is popped in the very same
#: ``_connect_server`` that armed it), so the horizon only bounds a LEAK for a
#: refusal whose CancelledError was swallowed somewhere before classification —
#: and a leaked record would otherwise be attributed to the next, unrelated
#: cancellation of that server and turn a teardown into a retry.
_REFRESH_CONTENTION_TTL_S = 5.0


class RefreshContentionLedger:
    """Servers whose coordinator refused to spend a refresh token, and WHY.

    The side channel that survives the transport, and it exists for exactly the
    reason :class:`AbandonedGrantLedger` does: an ordinary exception raised out
    of ``async_auth_flow`` is NOT delivered to ``_connect_server``. The MCP
    streamable-HTTP transport runs the request inside anyio cancel scopes, so
    anything raised out of the auth flow cancels that scope and reaches the
    caller as a bare ``CancelledError('Cancelled via cancel scope …')`` with no
    trace of the original error — verified for this exact raise, and for a plain
    ``RuntimeError`` raised from the same place, so it is a transport property
    rather than anything about :class:`McpRefreshContendedError`.

    The manager consults this ledger to tell "we refused to refresh without
    exclusivity" apart from "this connect was disposed", because the two need
    opposite treatment: the first is contention and must be retried with
    backoff, the second must never be converted into a retry. It also carries the
    REASON, because the alternative is one string that is untrue on two of the
    three paths (see :class:`McpRefreshContendedError`) and one that promises a
    fresh sign-in without saying why.

    Keyed by server URL. Each record is SINGLE-USE (``pop`` semantics: one
    refusal re-voices at most one cancellation, never two), but a server may hold
    SEVERAL live records at once — two concurrent connects for the same URL can
    both refuse, and a single slot let the second one's refusal vanish, which
    reaches ``_reconnect`` as a bare ``CancelledError``, is not an ``Exception``,
    and therefore kills the reconnect task silently instead of retrying it.
    Records also age out, so a record whose cancellation never reached the
    manager cannot be attributed to an unrelated cancellation later.
    """

    def __init__(self) -> None:
        #: url -> oldest-first ``(monotonic armed-at, refusal reason)``.
        self._records: dict[str, list[tuple[float, str]]] = {}

    def record(self, server_url: str, reason: str = REFRESH_REFUSAL_LOCK) -> None:
        """Append one server's record, pruning anything past the horizon."""
        now = time.monotonic()
        for old_url, entries in list(self._records.items()):
            kept = [entry for entry in entries if now - entry[0] <= _REFRESH_CONTENTION_TTL_S]
            if kept:
                self._records[old_url] = kept
            else:
                self._records.pop(old_url, None)
        self._records.setdefault(server_url, []).append((now, reason))

    def pop(self, server_url: str) -> str | None:
        """Consume one server's OLDEST record; its reason when FRESH, else ``None``.

        Stale records are dropped rather than returned: a record older than the
        horizon belongs to a cancellation that was swallowed somewhere else, and
        believing it would turn an unrelated teardown into a retry. Consuming
        one record leaves any later ones in place, which is what lets a second
        concurrent refusal for the same server still be re-voiced.
        """
        now = time.monotonic()
        entries = [
            entry
            for entry in self._records.get(server_url, [])
            if now - entry[0] <= _REFRESH_CONTENTION_TTL_S
        ]
        if not entries:
            self._records.pop(server_url, None)
            return None
        _, reason = entries.pop(0)
        if entries:
            self._records[server_url] = entries
        else:
            self._records.pop(server_url, None)
        return reason

    def clear(self) -> None:
        """Drop every record. Used by test isolation; no production caller.

        Nothing in the running system wants to forget a live refusal — the
        manager consumes records as it classifies, and stale ones age out — but
        a record is process-global state, and a test that armed one must not be
        able to change what the NEXT test observes.
        """
        self._records.clear()


REFRESH_CONTENTION = RefreshContentionLedger()


class LoopbackAuthFlow:
    """The redirect/callback pair for one authorization, over a real listener.

    The SDK drives an authorization in two steps: it hands us the URL to open
    (:meth:`redirect_handler`), then blocks on us to produce the code the
    provider redirected back with (:meth:`callback_handler`). We advertise
    ``http://127.0.0.1:<port>/callback`` as the redirect URI, so the ONLY way
    that promise is kept is by listening on it — without a listener the user
    signs in, the provider redirects, and the browser lands on
    ``ERR_CONNECTION_REFUSED`` with the code stranded in the address bar.

    The listener is opened in :meth:`redirect_handler`, BEFORE the browser is
    launched, because the provider can redirect the instant the user's session
    is already authorized — binding afterwards is a race the fast path loses.

    Paste is a strict FALLBACK, never a race: it is offered only when there is
    no listener to wait on. A thread parked in ``input()`` cannot be cancelled,
    so racing it would leave a reader on the tty and a thread that
    ``asyncio.run`` joins at shutdown — the browser path would succeed and the
    process would hang anyway. Paste is additionally gated on the terminal
    being ours: under the TUI it belongs to Textual's input driver, and a
    second reader on the same file descriptor does not queue behind it — the
    two split the user's keystrokes, which reads as an app randomly ignoring
    what is typed. :func:`~local_operator.logger.console_is_silenced` is the
    signal for that, and it also decides whether our progress notices go to
    stderr or to the log file.
    """

    def __init__(
        self,
        redirect_uri: str,
        server_url: str | None = None,
        *,
        interactive: bool = True,
    ) -> None:
        parsed = urlparse(redirect_uri)
        self.redirect_uri = redirect_uri
        #: Named on the callback page. Someone with several MCP servers
        #: configured has no other way to tell which tab belongs to which
        #: authorization, and "Authorized" without a subject is a page that
        #: could be about anything.
        self.server_url = server_url
        #: Whether this flow may open a browser. Ordinary session startup and
        #: auto-reconnects pass ``False``: when the stored grant cannot be
        #: refreshed they must fail with :class:`McpAuthRequiredError` instead
        #: of popping a login tab the user never asked for. Only an explicit
        #: ``/mcp login`` / ``local-operator mcp login`` runs interactive.
        self.interactive = interactive
        self._host = parsed.hostname or ""
        self._port = parsed.port or (443 if parsed.scheme == "https" else 80)
        self._path = parsed.path or "/"
        self._servable = parsed.scheme == "http" and self._host in LOOPBACK_HOSTS
        self._server: asyncio.AbstractServer | None = None
        self._result: asyncio.Future[tuple[str, str | None, str | None]] | None = None
        #: Why the bind failed, when it did. "Could not listen" and "could not
        #: have listened" need different advice, so the two are kept apart.
        self._bind_error: str | None = None

    # --- notices ---------------------------------------------------------

    def _notify(self, *lines: str) -> None:
        """Tell the user what is happening, without painting over a frame."""
        from local_operator.logger import console_is_silenced

        if console_is_silenced():
            for line in lines:
                logger.info("%s", line.strip())
            return
        for line in lines:
            print(line, file=sys.stderr)

    def _paste_allowed(self) -> bool:
        """Whether stdin is ours to read (see the class docstring)."""
        from local_operator.logger import console_is_silenced

        return sys.stdin.isatty() and not console_is_silenced()

    # --- SDK handlers ----------------------------------------------------

    async def redirect_handler(self, authorization_url: str) -> None:
        """Start listening, then send the user to the authorization URL.

        The URL is hard-wrapped in brackets so trailing OAuth params can never
        be silently lost on copy (a real production paste bug).

        Non-interactive flows RAISE here instead of opening a browser: the SDK
        only reaches this handler once the stored grant could not be refreshed,
        and a background connect must surface that as an actionable failure
        ("run /mcp login <name>"), never as a login tab popping up over the
        user's work. The exception propagates out of the connect cleanly — the
        SDK re-raises whatever the handler raises.
        """
        if not self.interactive:
            raise McpAuthRequiredError(self.server_url or self.redirect_uri)
        await self._start_server()
        lines = [
            "\nMCP OAuth authorization required. Open this URL in a browser:",
            f"  <{authorization_url}>",
        ]
        if await open_browser_quietly(authorization_url):
            lines.append("(opened in your default browser)")
        if self._server is not None:
            lines.append(f"Waiting for the redirect to {self.redirect_uri} …")
        self._notify(*lines)

    async def callback_handler(self) -> AuthorizationCodeResult:
        """Wait for the provider's redirect (or a pasted URL) and return the code.

        The transport cannot be trusted to deliver an exception raised here:
        the SDK's ``post_writer`` swallows ordinary auth-flow exceptions, so
        an ABANDONED grant (idle guard fired — the browser went away) is
        recorded in :data:`ABANDONED_GRANTS` and then raised as a RAW
        ``CancelledError``, the one exception shape that unwinds the
        transport's anyio task group intact. The manager recognises the
        pairing and re-voices it as :class:`McpLoginCancelledError`; raising
        the named error here directly would leave it stranded in a log line
        while the connect surfaced an unlabelled cancellation.
        """
        from mcp.shared.auth import AuthorizationCodeResult

        try:
            code, state, iss = await self._await_authorization()
        except AbandonedGrantError:
            ABANDONED_GRANTS.record(self)
            raise asyncio.CancelledError() from None
        finally:
            await self._stop_server()
        return AuthorizationCodeResult(code=code, state=state, iss=iss)

    async def _await_authorization(self) -> tuple[str, str | None, str | None]:
        if self._result is None:
            # No listener: paste is the only route left, and reading stdin is
            # safe precisely because nothing else is going to.
            return await self._await_pasted()
        try:
            # The inner clock is the 300 s redirect bound with its
            # port-forwarding advice for the genuinely slow case; the outer
            # coroutine adds the interactive idle guard AROUND it. They stay
            # separate because ``wait_for`` makes a timeout indistinguishable
            # from an outer one once nested — each clock must convert its own
            # expiry before the next layer can see it.
            # Captured once: a closure read of ``self._result`` types as
            # optional (``_stop_server`` resets it), and it cannot change
            # underneath the wait anyway — only ``_stop_server`` clears it,
            # and that runs after the wait resolves.
            result = self._result

            async def _within_idle_guard() -> tuple[str, str | None, str | None]:
                try:
                    return await asyncio.wait_for(result, timeout=PASTE_INPUT_TIMEOUT_S)
                except asyncio.TimeoutError as exc:
                    raise RuntimeError(
                        f"Timed out after {PASTE_INPUT_TIMEOUT_S:.0f}s waiting for the "
                        f"OAuth redirect to {self.redirect_uri}. If you authorized in a "
                        "browser on another machine it cannot reach this port — "
                        f"forward it (ssh -L {self._port}:127.0.0.1:{self._port} …) "
                        "and try again."
                    ) from exc

            return await asyncio.wait_for(_within_idle_guard(), timeout=INTERACTIVE_GRANT_TIMEOUT_S)
        except asyncio.CancelledError:
            # The login task itself was cancelled (an exclusive re-login, the
            # TUI's stop-ladder, Ctrl+C at the CLI). The underlying result
            # future is cancelled by the wait_for on the way out, and
            # ``callback_handler``'s ``finally`` stops the listener — but only
            # if we CO-OPERATE: shielding the teardown lets one stop-ladder
            # escalation turn a cancelled login into a wedged listener holding
            # the redirect port into the next grant.
            # The interrupt that arrives while we are still WAITING for the
            # redirect is unambiguous: the browser never answered, so this is
            # the task being cancelled, never the abandonment channel (that
            # one only fires from the idle-guard arm). Report it directly.
            with contextlib.suppress(Exception):
                await asyncio.shield(self._stop_server())
            raise McpLoginCancelledError("interrupted before the browser completed it") from None
        except asyncio.TimeoutError as exc:
            # The idle guard, not the redirect clock: nothing arrived for the
            # whole interactive budget, which in practice means the browser
            # went away — tab closed, consent abandoned.
            raise AbandonedGrantError(
                f"no redirect arrived within {INTERACTIVE_GRANT_TIMEOUT_S / 60:.0f} "
                "minutes — the login was probably cancelled (browser tab closed, "
                "or the authorization left unfinished)"
            ) from exc

    async def _await_pasted(self) -> tuple[str, str | None, str | None]:
        """Read the redirect URL from stdin. NEVER raced against the listener.

        A thread parked in ``input()`` cannot be cancelled: ``asyncio.to_thread``
        hands the call to the default executor, whose future refuses
        cancellation once running, and the thread then sits on stdin until a
        newline that — on the happy path, where the browser finished the grant —
        never comes. Two things break as a result: ``asyncio.run`` never returns
        (``Runner.close`` joins the default executor), so ``local-operator mcp
        login`` hangs AFTER succeeding; and the parked reader is a second
        consumer on the tty, which is the same keystroke-splitting bug this
        class exists to avoid, moved outside the TUI.

        So paste is a genuine FALLBACK, reached only when there is no listener
        to wait on, and the thread it starts is one the flow is committed to.
        """
        if not self._paste_allowed():
            raise RuntimeError(f"MCP OAuth cannot complete here: {self._no_route_reason()}")
        prompt = "Paste the full redirect URL (or 'code state' separated by a space): "
        try:
            raw = await asyncio.wait_for(
                asyncio.to_thread(lambda: input(prompt).strip()),
                timeout=PASTE_INPUT_TIMEOUT_S,
            )
        except asyncio.CancelledError:
            # Same receipt as the listener path: an interrupted CLI login must
            # not surface as a bare "MCP login failed for 'x':" with no reason.
            # The to_thread reader itself cannot be cancelled (that is why
            # paste is never raced), so no teardown is owed here.
            raise McpLoginCancelledError("interrupted before the redirect URL was pasted") from None
        except asyncio.TimeoutError as exc:
            # TRANSLATE IT. Since 3.11 `asyncio.TimeoutError` IS `TimeoutError`
            # and `str(TimeoutError())` is the empty string, so letting it
            # propagate makes the CLI print "MCP login failed for 'x': " with no
            # reason at all — on exactly the paths (unservable redirect URI, a
            # bind lost to a squatter) that `_no_route_reason` exists to explain.
            raise RuntimeError(
                f"Timed out after {PASTE_INPUT_TIMEOUT_S:.0f}s waiting for the pasted "
                f"redirect URL. Reading from stdin because {self._no_listener_reason()}."
            ) from exc
        return parse_oauth_callback_input(raw)

    def _no_listener_reason(self) -> str:
        """Why the browser redirect is not being captured, in the user's terms.

        A bind failure and an unservable redirect URI are different problems
        with different fixes, and conflating them sends someone to audit their
        OAuth configuration when the real answer is that a dev server is
        squatting the port. Phrased as a clause so both callers — "no route at
        all" and "falling back to a paste" — can finish the sentence their own
        way.
        """
        if self._bind_error is not None:
            return (
                f"nothing could listen on {self.redirect_uri} ({self._bind_error}) "
                "— free that port, or set a different `oauth.callback_port` for "
                "this server"
            )
        if not self._servable:
            return (
                f"the redirect URI {self.redirect_uri} is not a loopback address "
                "this process can serve"
            )
        return "the callback listener is not running"

    def _no_route_reason(self) -> str:
        """Why NEITHER route is available: no listener, and no stdin either.

        The listener clause is ended with a full stop rather than spliced in on
        a comma: the bind branch carries an em-dash aside, and coordinating
        "and stdin is not available" onto it reads as a third item in that
        clause's remedy list rather than as a second problem.
        """
        return (
            f"{self._no_listener_reason()}. Stdin is not available for a paste "
            "either. Run `local-operator mcp login <server>` from a terminal, "
            "or configure the server with a token."
        )

    # --- the listener ----------------------------------------------------

    async def _start_server(self) -> None:
        """Bind the redirect URI, or leave ``_server`` None and record why."""
        if self._server is not None or not self._servable:
            return
        loop = asyncio.get_running_loop()
        self._result = loop.create_future()
        try:
            self._server = await asyncio.start_server(self._serve, self._host, self._port)
        except OSError as exc:
            # Almost always "address already in use": another local-operator, or
            # a dev server squatting the port. Not fatal — the paste path still
            # completes the grant — but the user has to be told, because from
            # the browser's side this looks like the login simply not working.
            self._result = None
            self._bind_error = str(exc)
            self._notify(
                f"Could not listen on {self.redirect_uri} ({exc}); "
                "the browser redirect will not be captured automatically."
            )

    async def _stop_server(self) -> None:
        server, self._server = self._server, None
        self._result = None
        if server is None:
            return
        server.close()
        # BOUNDED. Since 3.12.1 ``wait_closed()`` also waits for every accepted
        # connection's handler to finish, so one peer that opened a socket and
        # sent nothing — a browser preconnect, a security scanner — would park
        # this forever. It runs in ``callback_handler``'s ``finally``, so an
        # unbounded wait here would swallow the authorization we already have
        # and turn every timeout into a hang.
        with contextlib.suppress(Exception):
            await asyncio.wait_for(server.wait_closed(), timeout=_SERVER_CLOSE_TIMEOUT_S)

    async def _serve(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """Answer one browser request; resolve the flow when it is the callback."""
        try:
            target = await self._read_request_target(reader)
            if target is None:
                writer.write(
                    callback_response(
                        "Bad request",
                        "That was not a request this page knows how to answer.",
                        closable=False,
                        status="400 Bad Request",
                    )
                )
                return
            parsed = urlparse(target)
            if parsed.path != self._path:
                # Browsers ask for /favicon.ico off their own bat; answering a
                # real 404 rather than resolving the flow keeps those from
                # being mistaken for the redirect.
                writer.write(
                    callback_response(
                        "Nothing here",
                        "This address only answers the authorization redirect.",
                        closable=False,
                        status="404 Not Found",
                    )
                )
                return
            query = parse_qs(parsed.query)
            error = (query.get("error") or [""])[0]
            if error:
                # The provider's own words go in their own labelled trough, not
                # spliced into a sentence spoken in our voice. `error_description`
                # is arbitrary text from a query string rendered inside a card
                # carrying our mark; escaping stops it being an injection, but
                # only a visible seam stops a hostile provider borrowing our
                # voice. It is also where a bare `access_denied` reads correctly
                # rather than being presented as English.
                # Stripped BEFORE the `or`, so a whitespace-only description
                # falls back to the code instead of satisfying the truthiness
                # test and blanking it. `?error=access_denied&error_description=
                # %20%20%20` otherwise raises "OAuth authorization failed:    ",
                # dropping the one word that says what went wrong.
                detail = (query.get("error_description") or [""])[0].strip() or error
                writer.write(
                    callback_response(
                        "Authorization failed",
                        "The provider did not grant this authorization, so nothing "
                        "was connected. You can start the connection again from "
                        "Local Operator.",
                        tone="danger",
                        server=self.server_url,
                        provider_message=detail,
                    )
                )
                self._settle_error(RuntimeError(f"OAuth authorization failed: {detail}"))
                return
            code = (query.get("code") or [""])[0]
            if not code:
                writer.write(
                    callback_response(
                        "No authorization code",
                        "The redirect arrived without an authorization code, so "
                        "there is nothing to hand back. You can start the "
                        "connection again from Local Operator.",
                        tone="danger",
                        server=self.server_url,
                    )
                )
                # SETTLE, do not just report. Without this the page says the tab
                # can be closed while the flow sits waiting out its full timeout
                # on a redirect that can never carry a code — the one call site
                # that made `closable` a lie.
                self._settle_error(RuntimeError("OAuth redirect carried no authorization code"))
                return
            writer.write(
                callback_response(
                    "Authorized",
                    "Local Operator has the authorization code and is finishing the connection.",
                    tone="success",
                    server=self.server_url,
                )
            )
            self._settle(
                code,
                (query.get("state") or [None])[0],
                (query.get("iss") or [None])[0],
            )
        except Exception:  # noqa: BLE001 — one bad request must not kill the flow
            logger.debug("MCP OAuth callback request failed", exc_info=True)
        finally:
            # The last unbounded awaits in a handler whose every other wait is
            # explicit. A peer that sends a valid GET and then stops reading
            # blocks `drain()` for as long as it likes, holding a task and an fd
            # for the life of the PROCESS — which under the TUI is the whole
            # session, not the flow. Today the page fits in a default send
            # buffer on macOS, but that is an accident of one platform's
            # defaults and a page size capped three functions away.
            with contextlib.suppress(Exception):
                await asyncio.wait_for(writer.drain(), timeout=_SERVER_CLOSE_TIMEOUT_S)
            writer.close()
            with contextlib.suppress(Exception):
                await asyncio.wait_for(writer.wait_closed(), timeout=_SERVER_CLOSE_TIMEOUT_S)

    async def _read_request_target(self, reader: asyncio.StreamReader) -> str | None:
        """The request target of a GET, with the head read and discarded.

        Bounded in BOTH directions — bytes and time. A peer that connects and
        says nothing must not hold a handler open (see :meth:`_stop_server`),
        and a peer that talks forever must not be allowed to.
        """
        try:
            return await asyncio.wait_for(self._read_head(reader), timeout=_REQUEST_READ_TIMEOUT_S)
        except asyncio.TimeoutError:
            return None

    @staticmethod
    async def _read_head(reader: asyncio.StreamReader) -> str | None:
        line = await reader.readline()
        budget = _MAX_REQUEST_HEAD_BYTES - len(line)
        if not line or budget < 0:
            return None
        parts = line.decode("latin-1").split()
        if len(parts) < 2 or parts[0].upper() != "GET":
            return None
        # Drain the headers so the browser sees a well-formed exchange rather
        # than a reset mid-request (which some render as a failed navigation).
        # The byte budget spans the WHOLE head: bounding each `readline` alone
        # leaves 64 headers x StreamReader's own 64 KiB limit, which is 4 MB.
        for _ in range(_MAX_REQUEST_HEADERS):
            header = await reader.readline()
            if header in (b"\r\n", b"\n", b""):
                break
            budget -= len(header)
            if budget < 0:
                return None
        return parts[1]

    def _settle(self, code: str, state: str | None, iss: str | None) -> None:
        if self._result is not None and not self._result.done():
            self._result.set_result((code, state, iss))

    def _settle_error(self, exc: Exception) -> None:
        if self._result is not None and not self._result.done():
            self._result.set_exception(exc)


# --- proactive OAuth refresh -------------------------------------------------
#
# Why this block exists: the SDK only refreshes a token INSIDE its 401 handler,
# and it derives the token endpoint from ``oauth_metadata`` — which a fresh
# process has not discovered yet, so the refresh falls back to
# ``urljoin(server_base, "/token")``. For providers whose token endpoint lives
# elsewhere (Datadog: ``https://us3.datadoghq.com/api/v2/oauth2/token``) that
# guess 404s, the refresh fails, and the SDK escalates to a FULL browser grant
# — the login tab popping up on every startup even though a valid refresh token
# sat in auth.db. The fix is to discover the real endpoints and spend the
# refresh token ourselves BEFORE the provider is built, race-free across the
# several sessions that start together.


@dataclass
class DiscoveredOAuthEndpoints:
    """What PRM/ASM discovery learned about one server's OAuth setup.

    ``oauth_metadata`` carries the real ``token_endpoint`` the refresh must
    target. ``protected_resource_metadata`` (when the server publishes it) is
    what makes the refresh include the RFC 8707 ``resource`` parameter, which
    some providers (Datadog) require. Both are also handed to the provider so a
    later in-flow refresh — e.g. a token that dies mid-session — targets the
    same endpoints instead of re-deriving the wrong guess.
    """

    oauth_metadata: "OAuthMetadata"
    protected_resource_metadata: "ProtectedResourceMetadata | None" = None
    auth_server_url: str | None = None


async def discover_oauth_endpoints(server_url: str) -> DiscoveredOAuthEndpoints | None:
    """Resolve a server's OAuth metadata via SEP-985 PRM then RFC 8414 ASM.

    Returns ``None`` when authorization-server metadata cannot be discovered;
    the caller then degrades to the SDK's own defaults (the pre-fix behavior)
    rather than failing the connect. Discovery is two unauthenticated GETs and
    only runs for OAuth servers, which are already the slow, deferred connects.

    Successful results are cached per process: reconnects rebuild the provider
    and would otherwise re-fetch stable metadata on every backoff rung. Only
    SUCCESSES are cached, so a transient discovery failure retries next time.
    """
    cached = _DISCOVERED_ENDPOINTS_CACHE.get(server_url)
    if cached is not None:
        return cached
    result = await _discover_oauth_endpoints_uncached(server_url)
    if result is not None:
        _DISCOVERED_ENDPOINTS_CACHE[server_url] = result
    return result


#: Per-process cache of successful endpoint discoveries, keyed by server URL.
_DISCOVERED_ENDPOINTS_CACHE: dict[str, DiscoveredOAuthEndpoints] = {}


async def _discover_oauth_endpoints_uncached(
    server_url: str,
) -> DiscoveredOAuthEndpoints | None:
    """The actual PRM/ASM fetch; :func:`discover_oauth_endpoints` caches it."""
    import httpx
    from mcp.client.auth.utils import (
        build_oauth_authorization_server_metadata_discovery_urls,
        build_protected_resource_metadata_discovery_urls,
    )
    from mcp.shared.auth import OAuthMetadata, ProtectedResourceMetadata

    prm: ProtectedResourceMetadata | None = None
    auth_server_url: str | None = None
    timeout = httpx.Timeout(REFRESH_HTTP_TIMEOUT_S)
    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            for url in build_protected_resource_metadata_discovery_urls(None, server_url):
                try:
                    response = await client.get(url)
                except httpx.HTTPError:
                    continue
                if response.status_code != 200:
                    continue
                try:
                    prm = ProtectedResourceMetadata.model_validate_json(response.content)
                except Exception:  # noqa: BLE001 — malformed metadata: try the next URL
                    continue
                if prm.authorization_servers:
                    auth_server_url = str(prm.authorization_servers[0])
                break

            asm: OAuthMetadata | None = None
            for url in build_oauth_authorization_server_metadata_discovery_urls(
                auth_server_url, server_url
            ):
                try:
                    response = await client.get(url)
                except httpx.HTTPError:
                    continue
                # Mirror the SDK's fallback semantics: a 4xx means "try the next
                # discovery URL"; anything else non-200 means "stop looking".
                if 400 <= response.status_code < 500:
                    continue
                if response.status_code != 200:
                    break
                try:
                    asm = OAuthMetadata.model_validate_json(response.content)
                except Exception:  # noqa: BLE001 — treat as not-found here
                    asm = None
                break
            if asm is None:
                return None
            return DiscoveredOAuthEndpoints(
                oauth_metadata=asm,
                protected_resource_metadata=prm,
                auth_server_url=auth_server_url,
            )
    except Exception:  # noqa: BLE001 — discovery is best-effort; degrade, don't fail
        logger.debug("OAuth metadata discovery failed for %s", server_url, exc_info=True)
        return None


def _try_lock_exclusive(fd: int) -> bool:
    """ONE non-blocking attempt at the exclusive lock. True when taken.

    Deliberately non-blocking on BOTH platforms. A blocking acquire parks a
    worker thread inside the kernel on a descriptor the calling coroutine may
    be about to close, and on macOS/BSD ``os.close()`` of a descriptor with a
    sibling thread parked in ``flock()`` blocks until that ``flock()`` returns —
    which, with the lock held by another process, is never. Called from the
    event-loop thread's ``finally`` that is exactly the whole TUI freezing:
    no repaint, no input. See :func:`_oauth_refresh_lock` for the full story.
    Retrying a non-blocking attempt is strictly weaker than blocking and is the
    only shape that stays cancellable, so do not "simplify" this back into
    ``fcntl.flock(fd, fcntl.LOCK_EX)``.

    Contention is the expected outcome and returns False; anything else is a
    real fault (EBADF, EINVAL, an unsupported filesystem) that retrying cannot
    fix, so it is raised for the caller to degrade on.
    """
    if os.name == "nt":  # pragma: no cover - platform specific
        import errno as _errno
        import msvcrt

        try:
            # ``msvcrt.locking`` locks a byte RANGE, so the file needs at least
            # one byte to lock — an empty lock file would fail with EINVAL
            # forever. Mirrors ``session_lease``'s handling of the same API.
            if os.fstat(fd).st_size == 0:
                os.write(fd, b"\0")
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            return True
        except OSError as lock_err:
            if lock_err.errno in (_errno.EDEADLOCK, _errno.EACCES, _errno.EAGAIN):
                return False
            raise
    else:
        import errno as _errno
        import fcntl

        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except OSError as lock_err:
            if lock_err.errno in (_errno.EAGAIN, _errno.EACCES, _errno.EWOULDBLOCK):
                return False
            raise


def _acquire_locked_fd(path: str, cancelled: threading.Event) -> int | None:
    """Open ``path`` and take the exclusive lock on it, or give up. Worker-side.

    Runs entirely on a worker thread and OWNS the descriptor for its whole life:
    it opens the fd, retries the non-blocking acquire until the deadline, and on
    any outcome other than success closes the fd ITSELF, in this same thread,
    after the last lock syscall has returned. That ownership rule is the fix for
    the deadlock — the event loop never closes a descriptor another thread might
    still be inside a lock call on, because by construction no such moment
    exists. Returns the locked fd on success (ownership passes to the caller,
    which must unlock and close it) or ``None`` on timeout/cancellation.

    ``cancelled`` is set by the coroutine when its await is cancelled, so an
    abandoned acquire abandons the retry loop within one ``_LOCK_RETRY_SLEEP_S``
    tick instead of holding a thread for the full bound.
    """
    deadline = time.monotonic() + LOCK_ACQUIRE_TIMEOUT_S
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    sleep_s = _LOCK_RETRY_SLEEP_S
    try:
        while True:
            if _try_lock_exclusive(fd):
                return fd
            if cancelled.is_set() or time.monotonic() >= deadline:
                break
            time.sleep(sleep_s)
            # Gentle geometric backoff: keeps pickup fast for the common
            # released-in-a-moment case, then stops burning wakeups once the
            # holder is evidently not finishing soon.
            sleep_s = min(sleep_s * 1.5, _LOCK_RETRY_SLEEP_MAX_S)
    except BaseException:
        os.close(fd)
        raise
    os.close(fd)
    return None


def _unlock(fd: int) -> None:
    if os.name == "nt":  # pragma: no cover - platform specific
        import msvcrt

        with contextlib.suppress(OSError):
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
    else:
        import fcntl

        fcntl.flock(fd, fcntl.LOCK_UN)


def _oauth_refresh_lock_path(server_url: str) -> Path:
    """The lock file for ONE server's refresh exchange, in ``config_dir()``.

    Keyed per server, because the race this lock exists to prevent is per
    server: two processes spending the SAME server's rotating refresh token.
    A single global lock file also serialised servers that can never race each
    other, so one slow or unreachable provider parked every other server's
    connect behind it — with six OAuth servers and several concurrent sessions
    that is a queue nothing drains, and the observable symptom was two servers
    connecting and the rest sitting on "connecting" forever.

    The name is a SHA-256 digest of the URL rather than the URL itself: server
    URLs contain ``/``, ``:`` and query strings that are not filename-safe, and
    a digest is stable across runs and machines without any escaping scheme to
    keep in step. Truncated to 16 hex chars — this namespaces a handful of
    configured servers, not an adversarial keyspace.

    The pre-fix global ``mcp_oauth_refresh.lock`` is deliberately left in place
    and never cleaned up: another local-operator process running an older build
    may still be using it, and deleting a file that a live peer holds an flock
    on would silently drop that peer's mutual exclusion (its lock survives on
    the unlinked inode while a new process creates a fresh file and takes an
    uncontended lock). It is a zero-byte file; leaving it costs nothing.
    """
    import hashlib

    from local_operator.paths import config_dir

    lock_dir = config_dir()
    lock_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(server_url.encode("utf-8")).hexdigest()[:16]
    return lock_dir / f"mcp_oauth_refresh_{digest}.lock"


@contextlib.asynccontextmanager
async def _oauth_refresh_lock(server_url: str) -> AsyncIterator["_OAuthRefreshLock"]:
    """Serialize the refresh exchange across processes for one server.

    Rotating refresh tokens make concurrent refreshes destructive: whichever
    process spends the current token second gets an error — or invalidates the
    first process's brand-new token. Holding an exclusive file lock around the
    exchange, and RE-READING the stored token after acquiring it, guarantees
    exactly one process performs the refresh no matter how many sessions start
    at once. The lock file lives next to ``auth.db``, is per server (see
    :func:`_oauth_refresh_lock_path`), and carries no state: it is only ever
    flocked, never read or written for content (on Windows it holds a single
    padding byte, which ``msvcrt.locking`` requires to have a range to lock).

    Yields a HANDLE that is truthy when the lock was taken and falsy when the
    bounded acquire gave up (see :class:`_OAuthRefreshLock`). A falsy body must
    still be SAFE to run, and that is now a narrower promise than "best-effort":
    it may NOT perform the refresh exchange, because an unexclusive exchange is
    the family-revoking double-spend this lock exists to prevent.
    :func:`ensure_mcp_oauth_fresh` can afford to skip the refresh on a falsy
    handle (its caller still connects and will re-read under the lock on the
    next attempt), and
    ``_RefreshCoordinatingOAuthProvider._coordinate_inflight_refresh`` must not
    fall through to the SDK's unlocked refresh either — it refuses with
    :class:`McpRefreshContendedError` instead. Blocking the connect while
    waiting for exclusivity is still not an option: it would trade a rare
    double-spend for a guaranteed hang, which is the freeze this function was
    rewritten to remove.

    The handle exists (rather than a plain ``bool``) because the ownership of
    the RELEASE has to be transferable: an exchange that outlives its connect
    keeps this lock until it has persisted its rotation
    (``_OAuthRefreshLock.transfer``), so neither a waiter's cancellation nor a
    budget overrun can free it while the POST is still on the wire — the window
    that let a sibling re-present the spent token and revoke the family.

    Two rules this function exists to enforce, both learned from a freeze that
    took the whole TUI down:

    1. **The acquire is bounded and non-blocking.** A bare
       ``fcntl.flock(fd, LOCK_EX)`` waits forever, so a lock leaked by a killed
       process parks a connect eternally.
    2. **The event loop never closes a descriptor a thread may be parked on.**
       The acquiring thread owns its fd end to end (:func:`_acquire_locked_fd`
       closes it itself on every non-success path), and the fd only ever
       reaches this coroutine once no lock syscall is outstanding on it. This
       matters because cancellation is routine here — ``/resume`` disposes the
       manager, which cancels in-flight connect tasks mid-acquire — and on
       macOS/BSD ``os.close()`` of a descriptor with a sibling thread inside
       ``flock()`` blocks until that ``flock()`` returns. Called from a
       ``finally`` on the event-loop thread, that stops the loop dead: the
       screen freezes and never repaints. Do not reintroduce a blocking acquire
       or an event-loop-side close of a possibly-in-use fd.
    """
    lock_path = _oauth_refresh_lock_path(server_url)
    # Signals the worker to abandon its retry loop when our await is cancelled,
    # so a cancelled acquire never leaves a thread running to the full bound.
    cancelled = threading.Event()
    # Acquire off the event loop: a contended lock must not stall other
    # servers' connects. The lock is on the fd, so it survives the await.
    acquire = asyncio.create_task(asyncio.to_thread(_acquire_locked_fd, str(lock_path), cancelled))
    try:
        fd = await asyncio.shield(acquire)
    except asyncio.CancelledError:
        # Hand the fd's fate entirely to the worker: tell it to stop, and let
        # the (shielded, so still-running) task close whatever it opened once
        # its last lock syscall has returned. We return immediately — nothing
        # here touches the descriptor, which is what keeps the loop alive.
        cancelled.set()
        acquire.add_done_callback(_close_abandoned_lock_fd)
        raise
    if fd is None:
        # INFO, not debug: this is a refusal that costs a connect its proactive
        # refresh, and the caller's next step (contention refusal or a skipped
        # optimisation) is easier to read next to its cause.
        logger.info(
            "MCP OAuth refresh lock not acquired within %.0fs for %s; no token " "was presented",
            LOCK_ACQUIRE_TIMEOUT_S,
            server_url,
        )
        handle = _OAuthRefreshLock(None)
        try:
            yield handle
        finally:
            handle.release_in_scope()
        return
    handle = _OAuthRefreshLock(fd)
    try:
        yield handle
    finally:
        handle.release_in_scope()


class _OAuthRefreshLock:
    """One held refresh lock whose RELEASE can be handed to another task.

    A plain ``async with`` would release the lock when the COROUTINE holding it
    returns, and the exchange task outlives that coroutine by design: it is
    detached at the budget and keeps reading a response for up to
    :data:`REFRESH_LATE_RESPONSE_GRACE_S`. Releasing there meant a sibling could
    acquire the lock, re-read the store (still holding the token this exchange
    had already presented) and POST it again — the reuse-detection request that
    revokes the whole family, measured against the real fixture.

    So the lock has an owner rather than a scope. The connect acquires it,
    decides, hands the POST to a task, and TRANSFERS the release to that task
    before its first cancellable await: from then on the exchange owns it and
    frees it only in its own ``finally``, after the store write. Nothing about
    the acquire changes — a waiter still either takes the lock within
    ``LOCK_ACQUIRE_TIMEOUT_S`` or gives up and takes the contended path.

    ``release`` is idempotent and safe to call from the event loop: the fd was
    handed over by the worker thread only after its last lock syscall completed
    (see :func:`_oauth_refresh_lock`).
    """

    __slots__ = ("_fd", "_deferred")

    def __init__(self, fd: int | None) -> None:
        self._fd = fd
        #: Set by :meth:`transfer`: the ``async with`` that produced this handle
        #: must NOT release it, because the exchange task owns the release now.
        self._deferred = False

    def __bool__(self) -> bool:
        """Whether the lock is HELD — by anyone, including a transferred owner.

        Truthiness is the caller's only signal (``if not lock: take the
        contended path``), and after :meth:`transfer` the honest answer is still
        "held": the exchange is holding it, and a second POST would be the
        double-spend the lock exists to prevent.
        """
        return self._fd is not None

    def transfer(self) -> None:
        """Hand the release to whoever owns this handle next.

        Called by the connect once the exchange task has been created and
        before it awaits: after this, the ``async with`` block it was created in
        releases nothing, and the task's own ``finally`` does.
        """
        self._deferred = True

    def release_in_scope(self) -> None:
        """Release on behalf of the ``async with`` that created this handle.

        A no-op when the release was transferred to an exchange task — which is
        the difference between this and a plain ``__exit__``: releasing here
        after a transfer would free the lock while the POST it guards is still
        on the wire.
        """
        if self._deferred:
            return
        self.release()

    def release(self) -> None:
        """Drop the lock. No-op once released, and safe to call twice."""
        fd, self._fd = self._fd, None
        if fd is None:
            return
        with contextlib.suppress(Exception):
            _unlock(fd)
        # Safe to close on this thread: the worker returned the fd only after
        # its lock call completed, so no thread is parked on it.
        with contextlib.suppress(OSError):
            os.close(fd)


def _close_abandoned_lock_fd(task: asyncio.Task[int | None]) -> None:
    """Close a lock fd whose waiter was cancelled before it could take it.

    Runs on the event loop when the shielded acquire finally settles, which is
    AFTER the worker thread's last lock syscall — so this close can never block
    the loop. Without it a lock acquired just as its waiter was cancelled would
    leak the descriptor and, worse, keep the lock held for the process's life.
    """
    # On both of these paths the WORKER has already closed the fd itself, so
    # there is nothing here to clean up: a cancelled task never returns one,
    # and an exception unwinds through :func:`_acquire_locked_fd`'s
    # ``except BaseException``, which closes before re-raising. That coupling is
    # what makes the early returns safe — an edit that moves the ``os.open``
    # into the ``try``, or adds a ``return fd`` ahead of the loop, would leak
    # both the descriptor AND the lock here with no diagnostic.
    if task.cancelled():
        return
    if task.exception() is not None:
        return
    fd = task.result()
    if fd is None:
        return
    with contextlib.suppress(Exception):
        _unlock(fd)
    with contextlib.suppress(OSError):
        os.close(fd)


def _refresh_user_agent() -> str:
    """A stable ``local-operator/<version>`` identifier for the refresh POST.

    See the header comment in :func:`_refresh_oauth_token_locked`: Cloudflare
    blocks a no-UA refresh to mcp.notion.com. The version is looked up from the
    installed distribution metadata and degrades to a bare product token when
    running from a source checkout with no metadata, so this can never raise
    into a refresh.
    """
    try:
        from importlib.metadata import PackageNotFoundError, version

        try:
            return f"local-operator/{version('local-operator')}"
        except PackageNotFoundError:
            return "local-operator"
    except Exception:  # noqa: BLE001 — a UA lookup must never break a refresh
        return "local-operator"


def _is_invalid_grant(body: bytes) -> bool:
    """True when an OAuth error response body is an ``invalid_grant`` (RFC 6749).

    A revoked or reused refresh token comes back as HTTP 400 with a JSON body
    ``{\"error\": \"invalid_grant\", ...}``. Distinguishing it from a generic
    400 is what lets the caller log the actionable \"run /mcp login\" meaning
    (and never treat it as retriable). Any parse failure returns False so an
    unexpected body is handled as an ordinary rejection, never crashes.
    """
    import json

    try:
        payload = json.loads(body)
    except (ValueError, TypeError):
        return False
    return isinstance(payload, dict) and payload.get("error") == "invalid_grant"


def _fallback_endpoints_for(server_url: str) -> "DiscoveredOAuthEndpoints":
    """Synthesize the endpoint the SDK itself would refresh against.

    When metadata discovery failed at startup (``endpoints is None``), the SDK's
    ``_refresh_token`` falls back to ``urljoin(<scheme>://<netloc>, \"/token\")``
    — the authorization base URL with its path stripped (see
    ``OAuthContext.get_authorization_base_url``). Building a minimal
    :class:`DiscoveredOAuthEndpoints` that targets exactly that URL lets the
    coordinating provider perform the refresh UNDER THE LOCK with a fresh
    re-read even without discovery, instead of falling through to the SDK's own
    UNLOCKED refresh. That closes the residual reuse window: the SDK's unlocked
    path spends whatever refresh token ``_initialize`` loaded at boot, which a
    sibling may have already rotated away — presenting it a second time is the
    reuse-detection trigger that revokes the whole family.

    ``protected_resource_metadata`` is left None: with no discovery we have no
    PRM, so the refresh omits the RFC 8707 ``resource`` parameter — matching the
    SDK's own fallback path, whose ``should_include_resource_param`` is likewise
    False without PRM (barring a 2025-06-18 protocol header, which the proactive
    refresh does not carry).
    """
    from urllib.parse import urljoin, urlparse

    from mcp.shared.auth import OAuthMetadata

    parsed = urlparse(server_url)
    base = f"{parsed.scheme}://{parsed.netloc}"
    token_url = urljoin(base, "/token")
    return DiscoveredOAuthEndpoints(
        oauth_metadata=OAuthMetadata.model_validate(
            {
                "issuer": base,
                # authorization_endpoint is a required field on OAuthMetadata
                # but is unused by the refresh path (refresh only reads
                # token_endpoint); the SDK's own fallback derives it the same
                # way, so a synthesized value keeps the model valid without
                # affecting behaviour.
                "authorization_endpoint": urljoin(base, "/authorize"),
                "token_endpoint": token_url,
            }
        ),
        protected_resource_metadata=None,
        auth_server_url=base,
    )


async def _refresh_oauth_token_locked(
    server_url: str,
    storage: McpTokenStorage,
    endpoints: DiscoveredOAuthEndpoints,
    *,
    lock: _OAuthRefreshLock | None = None,
    peer_refresh_is_success: bool = False,
) -> RefreshOutcome:
    """Spend the stored refresh token against the DISCOVERED token endpoint.

    Outcome is the whole result, and each member has its own caller treatment:
    ``"refreshed"`` (a fresh access token is in the store), ``"dead"`` (the
    authorization server rejected the grant with ``invalid_grant``; a tombstone
    is written and the token must never be presented again), ``"contended"``
    (no exclusivity, so NOTHING was presented), ``"overran"`` (the budget was
    spent with the exchange still running; a rotation that still lands is
    persisted), ``"unreachable"`` (the request never reached the wire, so
    nothing was presented and there is nothing to keep — a pre-send transport
    failure, which is NOT a server rejection), ``"unacknowledged"`` (a token may
    already be spent, so nothing was or may be presented until an interactive
    grant), and ``"failed"`` for every other outcome, including "nothing to
    refresh". Callers MUST compare against a member: every member is a truthy
    string.

    The caller holds the cross-process refresh lock and passes its HANDLE, which
    this function hands to the exchange task before its first cancellable await.
    That transfer is the point of the whole shape: the task owns the lock until
    it has persisted, so neither a cancellation of this connect nor the budget
    expiring can free it while the POST is still on the wire. Releasing it there
    let a sibling acquire the lock, re-read the store (still holding the token
    this exchange had already presented) and POST it a second time — the
    reuse-detection request that revokes the whole family.

    ``peer_refresh_is_success`` is for the PROACTIVE site, whose re-read under
    the lock exists to notice that a peer already refreshed while it waited: with
    it set, an already-fresh store is reported as ``"refreshed"`` without a
    POST. It is deliberately off for the other two callers, where a locally fresh
    token is exactly the state that needs spending (a session holding a
    locally-valid access token the server has since revoked).
    """
    # The exchange runs in its OWN task, and the await below is a SHIELD around
    # it: neither a cancellation of this connect nor the total timeout can abort
    # the POST. For a rotating provider the response is the only copy of the new
    # refresh token, so an aborted request does not just fail a refresh — the
    # server has already spent the token we presented, and the store is left
    # holding a spent one whose next presentation is a reuse-detection
    # double-spend that logs out the whole fleet. The task persists its own
    # result and releases the lock itself, so the rotation survives whichever of
    # the three ways this ends: completed in time, cancelled mid-flight, or
    # landed after the bound.
    exchange: asyncio.Task[RefreshOutcome] = asyncio.ensure_future(
        _perform_refresh_exchange(
            server_url,
            storage,
            endpoints,
            lock=lock,
            peer_refresh_is_success=peer_refresh_is_success,
        )
    )
    if lock is not None:
        # BEFORE the first await: from here the exchange owns the release, so
        # whichever way this coroutine leaves — return, timeout, cancellation —
        # the lock stays held until the rotation is in the store. ``getattr``
        # because the test doubles for ``_oauth_refresh_lock`` are plain bools.
        transfer = getattr(lock, "transfer", None)
        if transfer is not None:
            transfer()
    try:
        async with asyncio.timeout(REFRESH_HTTP_TIMEOUT_S):
            return await asyncio.shield(exchange)
    except TimeoutError:
        # The budget is spent, so this connect is done with it; the exchange is
        # not, and its persistence is the whole point of detaching it.
        logger.info(
            "MCP token refresh for %s overran its %.0fs budget; the exchange still "
            "holds the refresh lock and a rotation that lands will be persisted "
            "[writer: detached refresh exchange]",
            server_url,
            REFRESH_HTTP_TIMEOUT_S,
        )
        _detach_refresh_exchange(exchange, server_url)
        return "overran"
    except asyncio.CancelledError:
        # Routine teardown: every sidebar switch disposes a manager and cancels
        # its in-flight connects. Nothing is waited for here, deliberately — the
        # exchange owns the lock and persists its own result, so this coroutine
        # has nothing left to do but leave. Waiting out the remainder of the
        # budget used to be the only thing keeping a sibling from acquiring the
        # lock mid-POST, and it did not do that anyway (measured: the flock was
        # already free at the instant of the cancel, because the response read
        # outlived the remainder); the ownership transfer replaces it, and
        # teardown is now immediate instead of blocked for up to
        # ``REFRESH_HTTP_TIMEOUT_S``.
        _detach_refresh_exchange(exchange, server_url)
        raise


def _detach_refresh_exchange(exchange: "asyncio.Task[RefreshOutcome]", server_url: str) -> None:
    """Make a refresh exchange that outlived its connect safe to leave running.

    The task persists its own result (see :func:`_perform_refresh_exchange`),
    so nothing here consumes a value: this exists so a task nobody awaits is
    not reported as "exception was never retrieved" when the loop closes, and
    so a failure inside it is logged against the server it belongs to.

    INFO, not debug, for the same reason the write paths say so: an exchange
    nobody awaited is exactly the state whose rotation support has to be able to
    read out of a log after the fact.
    """

    def _settle(finished: "asyncio.Task[RefreshOutcome]") -> None:
        if finished.cancelled():
            return
        with contextlib.suppress(BaseException):
            exc = finished.exception()
            if exc is not None:
                logger.info(
                    "detached MCP token refresh for %s failed"
                    " [writer: detached refresh exchange]",
                    server_url,
                    exc_info=exc,
                )
                return
            outcome = finished.result()
            if outcome != "refreshed":
                logger.info(
                    "detached MCP token refresh for %s ended %r without persisting"
                    " [writer: detached refresh exchange]",
                    server_url,
                    outcome,
                )

    exchange.add_done_callback(_settle)


async def _perform_refresh_exchange(
    server_url: str,
    storage: McpTokenStorage,
    endpoints: DiscoveredOAuthEndpoints,
    *,
    lock: _OAuthRefreshLock | None = None,
    peer_refresh_is_success: bool = False,
) -> RefreshOutcome:
    """POST one refresh grant and persist whatever the server decided.

    Owns the whole exchange — the locked re-read, the request, the
    classification, the persistence AND the release of ``lock`` — so it can be
    detached from the connect that started it with no part of the result
    depending on a listener. Split out of :func:`_refresh_oauth_token_locked`
    for exactly that reason: the response is the only copy of a rotated refresh
    token, and every path that used to drop it (cancellation, the total timeout)
    did so by cancelling the request rather than by deciding against it.

    Holding the lock to the very END is what the reviewer's M2 was about. The
    caller-releases-it variant let a sibling in during the response read, where
    the store still holds the token this exchange has already spent — the
    reuse-detection POST that revokes the family.

    The request is also WRITE-AHEAD MARKED
    (:meth:`McpTokenStorage.mark_send_unconfirmed`) immediately before the POST,
    because the token is spent by the REQUEST and a process that dies mid-flight
    takes all in-memory knowledge with it. A failure BEFORE the request was
    written clears the marker again (the token was never presented, so that is
    an ordinary transient retry); anything later leaves it armed, and the
    refresh path then refuses to present that token until an interactive grant
    replaces it (see :data:`GRANT_UNCONFIRMED_SEND_KEY`).

    ``lock=None`` means the CALLER owns exclusivity (the direct-call path used
    by the unit tests and by nothing else); a caller that hands over a handle it
    could not acquire gets a refusal instead of a POST.
    """
    import base64
    from urllib.parse import quote

    if lock is not None and not lock:
        # Defence in depth for the one mistake that costs a token family: an
        # exchange must never run against a handle that says "no exclusivity".
        # The callers refuse earlier so this is not the path a user sees, but a
        # future call site that forgets the check fails closed here. Note the
        # test doubles for ``_oauth_refresh_lock`` are plain bools, so this reads
        # truthiness rather than an attribute.
        logger.info(
            "MCP token refresh for %s refused: no refresh lock was held, so nothing "
            "was presented [writer: locked refresh exchange]",
            server_url,
        )
        return "contended"

    import httpx
    from mcp.shared.auth import OAuthToken
    from mcp.shared.auth_utils import resource_url_from_server_url

    try:
        # Re-read UNDER the lock, inside the task that owns it: whatever we
        # spend must be what the store holds NOW, not what it held when the
        # connect decided to refresh.
        tokens = await storage.get_tokens()
        client_info = await storage.get_client_info()
        if tokens is None or not tokens.refresh_token or client_info is None:
            # Nothing to spend — a missing token is not evidence that the grant
            # is dead, so this must never tombstone.
            return "failed"

        if storage.grant_is_dead():
            # A grant the authorization server already rejected must cost no
            # POST at all. The callers check this before taking the lock; this
            # is the same question asked again under it, for the case where a
            # sibling tombstoned the grant while we waited for exclusivity.
            return "dead"

        if peer_refresh_is_success and _stored_token_is_fresh(storage, tokens):
            # Somebody rotated the grant while we waited for the lock. Spending
            # the token we read before the lock would present a spent one; the
            # caller re-reads the store and uses what the peer persisted.
            return "refreshed"

        if storage.send_unconfirmed(tokens.refresh_token):
            # The token in this row was presented by an exchange whose answer
            # never arrived, so it MAY already be spent, and re-presenting a
            # spent token is the reuse-detection POST that revokes the whole
            # family. Refuse, and say so at INFO: this state costs the user an
            # interactive sign-in, and that has to be readable in a log rather
            # than inferred from an unexplained failure.
            logger.info(
                "MCP token refresh for %s refused: this refresh token was already "
                "presented by an exchange that was never acknowledged, and presenting "
                "it again may revoke the whole token family — run /mcp reauth to sign "
                "in again [writer: locked refresh exchange]",
                server_url,
            )
            return "unacknowledged"

        token_endpoint = str(endpoints.oauth_metadata.token_endpoint)
        data: dict[str, str] = {
            "grant_type": "refresh_token",
            "refresh_token": tokens.refresh_token,
            "client_id": client_info.client_id,
        }
        # RFC 8707 resource indicator: included when the server publishes
        # protected resource metadata, matching the SDK's
        # ``should_include_resource_param``.
        if endpoints.protected_resource_metadata is not None:
            data["resource"] = resource_url_from_server_url(server_url)

        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            # Explicit UA, not httpx's default. mcp.notion.com sits behind
            # Cloudflare, whose bot heuristics return HTTP 403 "error code 1010"
            # to a refresh POST carrying NO User-Agent (observed live this cycle:
            # httpx's built-in UA slips through, a missing one is blocked).
            # Pinning our own identifier means a future httpx default change or a
            # stricter Cloudflare rule cannot silently turn every refresh into a
            # 403 and force a browser grant on the whole fleet.
            "User-Agent": _refresh_user_agent(),
        }
        auth_method = client_info.token_endpoint_auth_method
        if auth_method == "client_secret_post" and client_info.client_secret:
            data["client_secret"] = client_info.client_secret
        elif auth_method == "client_secret_basic" and client_info.client_secret:
            cid = quote(client_info.client_id, safe="")
            csecret = quote(client_info.client_secret, safe="")
            encoded = base64.b64encode(f"{cid}:{csecret}".encode()).decode()
            headers["Authorization"] = f"Basic {encoded}"

        # WRITE-AHEAD, before the request can be on the wire: from this instant
        # the row records that this token may have been spent, so a crash or a
        # kill between here and the response cannot leave a spent token looking
        # untouched to the next boot.
        storage.mark_send_unconfirmed(tokens.refresh_token)
        try:
            # ``read`` is the LATE GRACE, not the budget: a response that lands
            # after the awaiting connect has given up is still a rotation we
            # must not throw away. Connect/write/pool keep the budget, so a
            # server that never accepts the request still fails fast.
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(REFRESH_HTTP_TIMEOUT_S, read=REFRESH_LATE_RESPONSE_GRACE_S)
            ) as client:
                response = await client.post(token_endpoint, data=data, headers=headers)
        except (httpx.HTTPError, TimeoutError) as exc:
            # ``asyncio.timeout`` raises TimeoutError, which ``httpx.HTTPError``
            # does not cover; an exchange that overran its budget inside the
            # httpx call is handled exactly like a transport error. httpx's own
            # taxonomy decides whether the request ever reached the wire, which
            # is the difference between "no harm done" and "this token may be
            # gone" (see :func:`_refresh_request_never_sent`): guessing either
            # way is guessing about the family.
            if _refresh_request_never_sent(exc):
                storage.clear_send_unconfirmed()
                logger.info(
                    "MCP token refresh for %s could not reach the authorization server "
                    "(%s); the refresh token was never presented "
                    "[writer: locked refresh exchange]",
                    server_url,
                    type(exc).__name__,
                )
                # Its OWN outcome, not "failed": nothing was presented, so
                # nothing was refused. The user-visible text says "could not
                # be reached" instead of claiming an authorization server
                # rejected a request it never received (review round 2, minor
                # 1), and the manager composes that text from the code.
                return "unreachable"
            # SENT and never answered: the token may already be spent, so a
            # retry is exactly the reuse-detection POST. The request already
            # armed the marker, and this outcome is what makes the caller take
            # the honest non-POST path — strip the in-memory token and raise the
            # auth error that names the fresh sign-in, rather than pretending
            # this was a transient failure the manager can retry.
            logger.info(
                "MCP token refresh for %s got no answer after the request was sent "
                "(%s); this refresh token may already be spent, so it will not be "
                "presented again and the server needs a fresh sign-in "
                "[writer: locked refresh exchange]",
                server_url,
                type(exc).__name__,
            )
            return "unacknowledged"

        if response.status_code != 200:
            # Whether this answer resolves the send marker is a decision, not a
            # reflex: it is cleared only when the response is DEFINITIVE about
            # our token (a 200 we could parse, or a parsed ``invalid_grant``),
            # and kept for every answer that leaves the outcome unknown —
            # ``5xx``/``429`` among them. Clearing it on a 5xx would let the
            # next connect re-present a token a provider may have spent before
            # failing the response: the reuse-detection POST that revokes the
            # family this whole mechanism exists to protect. See
            # :func:`_answer_resolves_send_marker` for the rule and its cost.
            invalid_grant = response.status_code == 400 and _is_invalid_grant(response.content)
            if _answer_resolves_send_marker(response.status_code, invalid_grant=invalid_grant):
                storage.clear_send_unconfirmed()
            else:
                logger.info(
                    "MCP token refresh for %s got HTTP %s, which does not prove the "
                    "presented refresh token was not consumed; the token will not be "
                    "presented again, so this server may need one fresh sign-in "
                    "[writer: locked refresh exchange]",
                    server_url,
                    response.status_code,
                )
            # A revoked-grant rejection is qualitatively different from a
            # transient one and must be logged as such: for a rotating provider
            # that runs refresh-token REUSE DETECTION (Notion), presenting an
            # already-rotated refresh token returns HTTP 400
            # {"error":"invalid_grant"} and revokes the ENTIRE token family,
            # logging out every session at once. When that happens the only
            # recovery is an interactive login, so the log names that action
            # instead of implying a retry will heal it. We never auto-retry here
            # regardless: returning "dead" lets the connect surface
            # McpAuthRequiredError, which the manager turns into a suspended
            # reconnect (see manager._reconnect's McpAuthRequiredError arm)
            # rather than hammering a dead grant.
            if invalid_grant:
                logger.info(
                    "MCP OAuth grant revoked for %s (invalid_grant); run /mcp login to "
                    "restore it",
                    server_url,
                )
                # Tombstone from HERE and nowhere else: this is the only site
                # that has proof (a PARSED ``invalid_grant`` body, never a bare
                # HTTP 400) that the grant itself is gone rather than the server
                # having a bad minute. Misclassifying a transient 400 would
                # permanently suppress refresh on a live grant until the user ran
                # an interactive login. The rejected token travels with the
                # marker so the write can check the grant has not moved on
                # without us — see :meth:`McpTokenStorage.mark_grant_dead`. The
                # lock is held here for all of it, which is what keeps that
                # compare-then-write window at the microseconds between the two
                # calls.
                storage.mark_grant_dead(rejected_refresh_token=tokens.refresh_token)
                return "dead"
            # Informational, not debug: a rejected refresh is the thing that
            # turns into a login prompt, so its cause belongs in the readable
            # log.
            logger.info(
                "MCP token refresh rejected for %s: HTTP %s", server_url, response.status_code
            )
            return "failed"
        try:
            new_tokens = OAuthToken.model_validate_json(response.content)
        except Exception:  # noqa: BLE001 — an unparseable token is a failed refresh
            # HTTP 200 with a body we cannot read is NOT a clean failure: the
            # server processed the exchange and rotated, and the new token is
            # unreadable, so the token in the store IS spent. The marker stays
            # armed rather than being cleared, because presenting that token
            # again is a definite double-spend — this is the one case the
            # "presented but never acknowledged" state describes exactly.
            logger.info(
                "MCP token refresh for %s returned HTTP 200 with an unusable body; the "
                "presented token is spent and will not be presented again — run "
                "/mcp reauth [writer: locked refresh exchange]",
                server_url,
            )
            return "unacknowledged"
        storage.clear_send_unconfirmed()

        # RFC 6749 §6: a refresh response may omit ``scope`` (unchanged) and
        # ``refresh_token`` (not rotated). Carry both forward so the persisted
        # row stays self-describing and can refresh again next time.
        if new_tokens.scope is None and tokens.scope is not None:
            new_tokens.scope = tokens.scope
        if new_tokens.refresh_token is None:
            new_tokens.refresh_token = tokens.refresh_token
        # The refresh path's OWN write, never ``set_tokens``: it is conditioned
        # on the grant this exchange was computed from and it never clears the
        # dead-grant marker (see :meth:`McpTokenStorage.store_refresh_result`).
        storage.store_refresh_result(new_tokens, presented_refresh_token=tokens.refresh_token)
        return "refreshed"
    finally:
        # The lock is released HERE, after the persist, whatever the outcome.
        # This is the ownership the connect transferred before its first await.
        if lock is not None:
            lock.release()


async def ensure_mcp_oauth_fresh(
    server_url: str,
    cfg: MCPServerConfig,
    store: StructuralAuthStore | None = None,
) -> DiscoveredOAuthEndpoints | None:
    """Refresh a stored OAuth grant before connecting, race-free. Best-effort.

    Returns the discovered endpoints so the provider can be primed with them
    (``None`` when discovery failed and the SDK should fall back to its own
    defaults). This never opens a browser, and a failed REFRESH never raises:
    the stored token is simply left as-is, and the provider's non-interactive
    redirect handler is what converts the resulting grant attempt into an
    actionable :class:`McpAuthRequiredError` instead of a login tab. Lock
    ACQUISITION can still raise ``OSError`` (unwritable config dir, exhausted
    fds, the bounded Windows retry) — the manager's caller wraps this in a
    broad catch, and any new caller must do the same or accept the raise.

    ``cfg`` is accepted for signature stability (the manager passes it, and a
    future per-server knob — e.g. opting out of proactive refresh — will need
    it) but is not consulted today.

    The refresh is wrapped in a cross-process lock with a re-read after
    acquiring it, so only one of several concurrently STARTING sessions spends
    a rotating refresh token. Scope honestly stated: this function is one of
    three ways a refresh can start, and the other two — the in-flight
    coordinator an ``async_auth_flow`` runs before every request, and the
    401-recovery refresh — also take this lock and re-read under it. What is
    NOT covered is another PROCESS that never takes it: a completed interactive
    login persists its tokens through the SDK's own ``set_tokens``, which does
    not lock, so a writer racing one of these refreshes can still lose its
    rotation on the narrow window between our read and our write.

    A grant previously rejected with ``invalid_grant`` short-circuits before the
    lock is taken: see :data:`GRANT_DEAD_AT_KEY`.
    """
    del cfg  # reserved — see docstring
    storage = McpTokenStorage(server_url, store)
    endpoints = await discover_oauth_endpoints(server_url)

    if _stored_token_is_fresh(storage, await storage.get_tokens()):
        return endpoints

    if storage.grant_is_dead():
        # Suppress BEFORE the lock: a grant the server already rejected must
        # cost neither a lock acquire nor a token POST. Re-spending it is what
        # revokes the whole token family on a reuse-detecting provider (see
        # GRANT_DEAD_AT_KEY). Endpoints are still returned so the connect
        # proceeds to the point where it raises McpAuthRequiredError, which the
        # manager renders as the actionable "run /mcp reauth" toast.
        #
        # Logged at INFO once per CONNECT, not once per process: this is reached
        # from _connect_server, which also serves auto-reconnect and /mcp login,
        # so a server that reconnects logs it again. That is bounded rather than
        # a storm — _reconnect's McpAuthRequiredError arm abandons auto-reconnect
        # instead of retrying — and INFO is deliberate: silence would read as
        # "fixed" when it means "suppressed", and the operator still owes an
        # interactive login. Plain logger.info also means it reaches CLI and
        # headless hosts, not only the TUI's toast surface.
        logger.info(
            "MCP OAuth grant for %s is known-dead (invalid_grant); skipping refresh "
            "-- run /mcp reauth to restore it",
            server_url,
        )
        return endpoints

    tokens = await storage.get_tokens()
    if (
        endpoints is None
        or await storage.get_client_info() is None
        or tokens is None
        or not tokens.refresh_token
    ):
        return endpoints

    async with _oauth_refresh_lock(server_url) as lock:
        if not lock:
            # The bounded acquire gave up, so we cannot claim exclusivity and
            # must not spend the rotating refresh token on a guess. Skip the
            # PROACTIVE refresh and connect anyway: this function does not leak
            # an unlocked POST on its own, because the provider's
            # ``async_auth_flow`` always runs the in-flight coordinator before
            # it delegates to the SDK — and that coordinator refuses to fall
            # through to the SDK's unlocked ``_refresh_token``
            # (:class:`McpRefreshContendedError`). A skipped optimisation costs
            # a round trip; blocking the connect costs the whole session.
            return endpoints
        # The exchange re-reads the store UNDER this lock (and takes the release
        # over: see :func:`_refresh_oauth_token_locked`). Handing it the lock we
        # already hold is what keeps a cancelled or over-budget connect from
        # freeing it while the POST is still on the wire.
        outcome = await _refresh_oauth_token_locked(
            server_url, storage, endpoints, lock=lock, peer_refresh_is_success=True
        )
        if outcome in ("contended", "overran", "failed", "unreachable", "unacknowledged"):
            # Best-effort by contract: the connect proceeds and re-reads under
            # the lock on its next attempt. Each refusal already logged its own
            # reason at INFO, naming that writer — no second line here, because
            # a duplicate would read as a second failure.
            logger.debug("MCP proactive refresh for %s ended %r", server_url, outcome)
    return endpoints


def _resolve_redirect_uri(cfg: MCPServerConfig) -> str:
    """The loopback redirect URI a config's ``oauth`` block resolves to.

    Shared by :func:`wire_oauth_auth` (which advertises it in the client
    metadata) and :func:`build_oauth_provider` (which binds the flow's
    listener to it): the two MUST agree, or the provider redirects the
    browser to an address nothing is serving.
    """
    oauth = cfg.oauth
    callback_port = (oauth.callback_port if oauth is not None else None) or DEFAULT_CALLBACK_PORT
    callback_path = (oauth.callback_path if oauth is not None else None) or DEFAULT_CALLBACK_PATH
    if not callback_path.startswith("/"):
        callback_path = f"/{callback_path}"
    return (oauth.redirect_uri if oauth is not None else None) or (
        f"http://127.0.0.1:{callback_port}{callback_path}"
    )


def wire_oauth_auth(
    server_url: str,
    cfg: MCPServerConfig,
    store: StructuralAuthStore | None = None,
    *,
    interactive: bool = True,
    flow: LoopbackAuthFlow | None = None,
) -> dict[str, Any]:
    """Build ``OAuthClientProvider`` kwargs for one server.

    ``cfg`` is the server's :class:`~local_operator.mcp.config.MCPServerConfig`
    (its ``auth`` / ``oauth`` blocks supply client identity and callback
    knobs). Returns a dict suitable for ``OAuthClientProvider(**kwargs)``:

    - ``server_url``: the MCP server URL (resource indicator base);
    - ``client_metadata``: PKCE authorization-code client, redirect URI
      ``http://127.0.0.1:{callback_port or DEFAULT_CALLBACK_PORT}``
      ``{callback_path or /callback}`` (PKCE itself is automatic inside the
      SDK);
    - ``storage``: a :class:`McpTokenStorage` bound to ``store``; a config
      ``client_id`` pre-seeds the client registration so DCR is skipped
      (MCP-11);
    - ``redirect_handler`` / ``callback_handler``: the two halves of one
      :class:`LoopbackAuthFlow`, which listens on that redirect URI for the
      duration of the grant (see the module docstring). Callers that need
      the flow itself — :func:`build_oauth_provider` does, for the manager's
      abandoned-grant check — pass their own via ``flow``; the dict returned
      here stays exactly the SDK's kwargs so it can be splatted straight
      into ``OAuthClientProvider``.

    ``interactive`` controls whether the flow may open a browser. Ordinary
    session startup and auto-reconnects pass ``False`` so an unrefreshable
    grant surfaces as :class:`McpAuthRequiredError` instead of popping a login
    tab; only an explicit ``/mcp login`` runs interactive.

    The returned dict is constructed eagerly but imports ``mcp`` lazily inside
    so config-only code paths never touch the SDK.
    """
    from mcp.shared.auth import OAuthClientMetadata

    auth = cfg.auth
    oauth = cfg.oauth

    redirect_uri = _resolve_redirect_uri(cfg)

    # Scopes: explicit `scope` on the auth block — an extra-allowed field, so
    # it lives in ``model_extra`` rather than being declared — else none (the
    # server advertises them via protected-resource metadata).
    scope: str | None = (auth.model_extra or {}).get("scope") if auth is not None else None

    client_secret = (auth.client_secret if auth is not None else None) or (
        oauth.client_secret if oauth is not None else None
    )

    client_metadata = OAuthClientMetadata(
        client_name="local-operator",
        redirect_uris=[AnyUrl(redirect_uri)],
        scope=scope,
        grant_types=["authorization_code", "refresh_token"],
        response_types=["code"],
        token_endpoint_auth_method="client_secret_post" if client_secret else "none",
    )

    storage = McpTokenStorage(server_url, store)

    # A configured client_id is pinned: pre-seed it so the SDK skips dynamic
    # client registration (MCP-11). DCR would mint a fresh client whose
    # redirect URI need not match what the provider registered, which breaks
    # pinned-redirect providers outright.
    client_id = (auth.client_id if auth is not None else None) or (
        oauth.client_id if oauth is not None else None
    )
    if client_id:
        storage.seed_client_info(client_id, client_secret)

    if flow is None:
        flow = LoopbackAuthFlow(redirect_uri, server_url=server_url, interactive=interactive)
    return {
        "server_url": server_url,
        "client_metadata": client_metadata,
        "storage": storage,
        "redirect_handler": flow.redirect_handler,
        "callback_handler": flow.callback_handler,
    }


def _make_refresh_coordinating_provider(
    kwargs: dict[str, Any],
    *,
    server_url: str,
    storage: "McpTokenStorage",
    endpoints: DiscoveredOAuthEndpoints | None,
) -> Any:
    """An ``OAuthClientProvider`` whose in-flow refresh is race-free across processes.

    Why this subclass exists: the SDK loads the stored tokens ONCE in
    ``_initialize`` and never re-reads storage afterwards, and its in-flow
    refresh (``async_auth_flow`` -> ``_refresh_token``) spends whatever refresh
    token that first read put in memory, under NO cross-process lock. That is
    fine for a token that expires and is refreshed once. It is destructive for a
    provider that ROTATES its refresh token on every use (Notion: an 8-hour
    access token plus a rotating refresh token) the moment more than one
    local-operator process is alive: this harness runs one process per cmux
    workspace, so several long-lived sessions cross the same 8-hour boundary
    still holding the same in-memory refresh token from when they booted. They
    each POST it; the authorization server rotates; the first wins and every
    other session's brand-new token is already dead. The SDK then
    ``clear_tokens()`` and the next request opens a FULL browser grant — the
    "Notion logged out again" the user sees, several times a day.

    :func:`ensure_mcp_oauth_fresh` already closes this race for the STARTUP
    refresh with a cross-process file lock and a re-read under it; its own
    docstring notes that the SDK's in-flow refresh is the remaining unlocked
    path. This subclass extends the same guard to that path: before the SDK
    would refresh, it re-reads the persisted token under
    :func:`_oauth_refresh_lock` and, if a sibling process already rotated it,
    ADOPTS the fresh token into the context and skips the refresh entirely.
    Only when the store still holds an expired token does exactly one process
    perform the exchange — against the discovered endpoint, so a provider whose
    token endpoint is not ``<server_base>/token`` (Datadog) refreshes rather
    than 404-ing into a browser grant.

    The invariant, enforced on EVERY path: the SDK's own (unlocked)
    ``_refresh_token`` must never run with an in-memory refresh token older than
    what storage holds. With no discovered ``endpoints`` (metadata discovery
    failed at startup) the coordinator no longer falls through to that unlocked
    refresh — it synthesizes the SDK's own fallback token endpoint
    (``<scheme>://<netloc>/token``) and performs the exchange UNDER THE LOCK
    with a fresh re-read, so even without discovery a stale boot-time refresh
    token is never presented a second time. Presenting one to a reuse-detecting
    provider (Notion) is what returns ``invalid_grant`` and revokes the whole
    token family. If the coordinated refresh itself raises, a final under-lock
    re-read still adopts the freshest persisted token before the SDK runs, so
    the invariant survives the exception fall-through too.

    The subclass additionally intercepts the FIRST 401 the resource server
    returns for the original request. The coordination step above only fires
    when the loaded token is EXPIRED, but a rotating provider (Notion) revokes
    every previously issued access token when a sibling refreshes — so a
    session can hold a locally-valid, server-side-revoked token that sails
    past coordination and gets a 401. Before that 401 reaches the SDK's full
    browser-authorization branch, the flow re-reads the store under the lock
    and, if a peer's different token is already there, adopts it and re-sends
    the request once. See ``async_auth_flow`` for the bound and the pass-through
    rule for genuinely dead grants.
    """
    from mcp.client.auth import OAuthClientProvider

    class _RefreshCoordinatingOAuthProvider(OAuthClientProvider):
        # Bound on the re-read/refresh interception: a stuck lock acquisition or
        # a hung endpoint must not park an authenticated request forever. The
        # file lock is only ever held for one token POST plus the SQLite read,
        # and that POST is bounded in TOTAL wall time by the ``asyncio.timeout``
        # in ``_refresh_oauth_token_locked`` (httpx's own per-operation timeout
        # does not bound a dribbling response), so a peer that legitimately
        # holds it clears well inside this bound. Overrunning it means a leaked
        # lock, and degrading to the SDK's own refresh is strictly better than
        # blocking the request.
        _refresh_coord_server_url = server_url
        _refresh_coord_storage = storage
        _refresh_coord_endpoints = endpoints

        async def _coordinate_inflight_refresh(self) -> None:
            """Re-sync from the store (and refresh once, race-free) if the SDK is
            about to refresh. No-op unless the loaded token is invalid AND
            refreshable — exactly the SDK's own in-flow refresh trigger, so this
            never adds a round trip to a request that would have gone through
            unauthenticated-refresh-free."""
            ctx = self.context
            # Gate on the SAME predicate the SDK's async_auth_flow uses so we
            # intercept precisely when it would refresh, and never otherwise.
            if ctx.is_token_valid() or not ctx.can_refresh_token():
                return
            if self._refresh_coord_storage.grant_is_dead():
                # Same suppression as ensure_mcp_oauth_fresh, before the lock.
                # Debug rather than info: the startup path already logged the
                # actionable reauth line for this connect, and this site can run
                # per request.
                logger.debug(
                    "MCP in-flight refresh skipped for %s: grant is known-dead",
                    self._refresh_coord_server_url,
                )
                # STRIPPING HERE IS LOAD-BEARING, not a tidy-up: returning from
                # this coroutine falls through to the SDK's own UNLOCKED
                # _refresh_token, which reads ctx.current_tokens directly and
                # would POST the very token the authorization server already
                # rejected — the family-revoking request this whole subsystem
                # exists to prevent. Suppressing OUR refresh without dropping
                # the token therefore removes the harmless POSTs and keeps the
                # harmful one, which is strictly worse than not suppressing at
                # all, because it also looks fixed.
                #
                # This branch is the STEADY STATE: boot 1 writes the tombstone
                # and takes the "dead" arm below (which strips for the same
                # reason); every boot after it arrives here instead. Measured at
                # the wire before this strip existed: 1 SDK POST on every one of
                # 5 already-tombstoned boots.
                #
                # Dropping the in-memory refresh token makes can_refresh_token()
                # read False, so the SDK skips its refresh branch and goes to
                # the authorization branch, which our non-interactive redirect
                # handler turns into an actionable McpAuthRequiredError.
                self._strip_in_memory_refresh_token(ctx)
                return
            outcome: RefreshOutcome | None = None
            try:
                async with _oauth_refresh_lock(self._refresh_coord_server_url) as lock:
                    # Re-read under the lock: a sibling process may have rotated
                    # the token while we waited. Adopting its result is what
                    # turns a double-spend into a no-op.
                    await self._resync_from_store(ctx)
                    if ctx.is_token_valid():
                        return  # a peer already refreshed; do not spend again
                    if not lock:
                        # No exclusivity, so nothing may be presented — not even
                        # by the exchange, whose whole contract is that the
                        # caller hands it a HELD lock. The refusal below decides
                        # what the user sees; the exchange is never entered, so a
                        # refusal costs zero POSTs.
                        outcome = "contended"
                    else:
                        # The invariant this whole block exists to guarantee: the
                        # SDK's own ``_refresh_token`` must NEVER run with an
                        # in-memory refresh token older than the one in storage.
                        # The SDK loads the token once at ``_initialize`` and
                        # spends it unlocked; a sibling that rotated it in
                        # between leaves us holding a stale refresh token, and
                        # presenting that to a reuse-detecting provider (Notion)
                        # returns ``invalid_grant`` and revokes the ENTIRE token
                        # family — logging out every session at once. So we
                        # always perform the refresh ourselves, UNDER THE LOCK,
                        # with a fresh re-read; the SDK's unlocked path is never
                        # reached with a stale token.
                        endpoints = self._refresh_coord_endpoints
                        if endpoints is None:
                            # Discovery failed at startup, but we must still
                            # refresh under the lock rather than fall through to
                            # the SDK's UNLOCKED refresh (which would spend the
                            # possibly-stale boot-time token and risk the family
                            # revocation above). Synthesize the exact endpoint the
                            # SDK itself would fall back to
                            # (``<scheme>://<netloc>/token``) so the locked,
                            # re-reading refresh targets the same URL the unlocked
                            # path would have.
                            endpoints = _fallback_endpoints_for(self._refresh_coord_server_url)
                        # The exchange re-reads the store UNDER this lock and
                        # takes the release over before its first await, so the
                        # token it spends is the one on disk at the moment of the
                        # POST — not the one this read saw — and no waiter's
                        # cancellation or timeout can free the lock while that
                        # POST is on the wire.
                        outcome = await _refresh_oauth_token_locked(
                            self._refresh_coord_server_url,
                            self._refresh_coord_storage,
                            endpoints,
                            lock=lock,
                        )
            except Exception:  # noqa: BLE001 — coordination is best-effort
                # A failed re-read/refresh must never break the request. But we
                # must NOT let the SDK's unlocked refresh then spend a stale
                # boot-time token: re-read storage one final time and overwrite
                # the in-memory token with whatever is persisted, so whatever the
                # SDK spends next is at least the freshest stored refresh token,
                # never an older one a sibling already rotated away. This upholds
                # the same invariant on the exception fall-through path (a
                # transient store error, an unexpected raise) as the success path
                # does.
                logger.debug(
                    "MCP in-flight refresh coordination failed for %s",
                    self._refresh_coord_server_url,
                    exc_info=True,
                )
                await self._adopt_freshest_stored_token(ctx)
                outcome = "failed"
            # The OUTCOMES are decided out here, deliberately: a refusal is a
            # DECISION, not a failure, and raising it inside the ``try`` above
            # would let the ``except Exception`` arm swallow it — which would put
            # the SDK's unlocked refresh back on the wire, the one thing this
            # method exists to prevent.
            #
            # Compare explicitly: every member is a truthy string, so a
            # ``if outcome:`` here would invert these branches silently and
            # pyright would not catch it.
            if outcome == "refreshed":
                await self._resync_from_store(ctx)
            elif outcome == "dead":
                # The one POST that actually revokes the token family is the
                # replay; returning here still falls through to the SDK's own
                # UNLOCKED _refresh_token, which would spend the token the
                # authorization server just rejected. Dropping the in-memory
                # refresh token makes the SDK's can_refresh_token() read False,
                # so it skips its refresh branch entirely and goes to the
                # authorization branch — which our non-interactive redirect
                # handler already turns into an actionable McpAuthRequiredError.
                # Same user-visible outcome, one fewer family-revoking POST.
                self._strip_in_memory_refresh_token(ctx)
            elif outcome == "unacknowledged":
                # A token that was presented (or may have been) and never
                # acknowledged must not be presented again: that is the
                # reuse-detection POST. The honest disposition is an interactive
                # sign-in, not a retry — see McpRefreshUnconfirmedError.
                self._refuse_unconfirmed_exchange(ctx)
            elif outcome in ("contended", "overran", "failed", "unreachable"):
                self._refuse_unlocked_refresh(ctx, outcome)
            # POST-CONDITION, in ONE place so no exit path can miss it: we are
            # about to return into ``async_auth_flow``, whose very next act is to
            # hand the context to the SDK's ``async_auth_flow``. If the token is
            # still invalid AND still refreshable at that point, the SDK will
            # POST its in-memory refresh token with no lock and no re-read — the
            # unlocked spend this whole method exists to prevent. Refuse instead.
            #
            # ``"failed"`` is the default reason rather than the lock: this arm
            # is reached when the exchange ran (or could not be attributed to a
            # single cause) and left nothing usable, and the message says the
            # authorization server's answer was not a usable token.
            self._refuse_unlocked_refresh(ctx, "failed")

        def _refuse_unlocked_refresh(self, ctx: Any, outcome: str = "failed") -> None:
            """Never hand the SDK an in-memory refresh token we failed to refresh.

            The post-condition of :meth:`_coordinate_inflight_refresh`, called
            once at the end so no exit path can skip it. Reaching here with an
            invalid-but-refreshable context means coordination did not produce
            a working token this time (the lock was unavailable, the exchange
            overran the connect's budget, the refresh returned ``"failed"``, or
            the fall-through could not find a fresher stored one).
            ``async_auth_flow`` returns straight into the SDK's own flow at that
            point, and the SDK's ``_refresh_token`` POSTs
            ``ctx.current_tokens.refresh_token`` DIRECTLY — no lock, no re-read —
            so the next thing on the wire would be a possibly already-rotated
            token, which a reuse-detecting provider answers with
            ``invalid_grant`` AND a family revocation.

            So strip it and raise a TRANSIENT error instead. Two properties are
            load-bearing:

            * the strip makes ``can_refresh_token()`` read False, so even if a
              caller swallows this error the SDK's refresh branch cannot run;
              and
            * the error is not an auth error, so it reaches the manager's
              generic reconnect arm — backoff and retry — rather than
              ``_block_on_auth``. Being unable to refresh is contention or a
              server fault, not a dead grant: blocking on auth here would
              strand a healthy server on a login prompt the user has no reason
              to run.

            ``outcome`` becomes the refusal REASON CODE, which is what lets the
            manager render a truthful short message per path: unable to take the
            lock, a budget overrun, an unreachable token endpoint and an answer
            without a usable token are four different things and were one
            sentence before (design review input; review round 2 minor 1 split
            the last of those off). The code is carried onto the ledger so the
            manager's re-voiced error says the same thing.
            """
            if not ctx.can_refresh_token() or ctx.is_token_valid():
                return
            self._strip_in_memory_refresh_token(ctx)
            # ``.get``'s default is the ANSWERED-but-unusable case, which is what
            # every remaining outcome ("failed", and the post-condition's
            # fall-through) actually is. ``unreachable`` must be listed: without
            # its own arm a request that never left the machine would be reported
            # to the user as a server refusal.
            reason_code = {
                "contended": REFRESH_REFUSAL_LOCK,
                "overran": REFRESH_REFUSAL_INFLIGHT,
                "unreachable": REFRESH_REFUSAL_UNREACHABLE,
            }.get(outcome, REFRESH_REFUSAL_ENDPOINT)
            # ARM THE SIDE CHANNEL BEFORE RAISING, and do it here rather than in
            # the raise's caller: the MCP transport will NOT deliver this error.
            # It runs the request inside anyio cancel scopes, so the raise
            # cancels the scope and reaches ``_connect_server`` as a bare
            # ``CancelledError('Cancelled via cancel scope …')``. The manager
            # re-voices that cancellation using this record — reason and all —
            # and only when the cancellation was NOT a genuine external one, so
            # a dispose still wins. See :class:`RefreshContentionLedger`.
            REFRESH_CONTENTION.record(self._refresh_coord_server_url, reason_code)
            raise McpRefreshContendedError(self._refresh_coord_server_url, reason_code=reason_code)

        def _refuse_unconfirmed_exchange(self, ctx: Any) -> None:
            """Refuse to re-present a refresh token that may already be spent.

            The disposition for ``outcome == "unacknowledged"``: the store holds
            a token an exchange presented and never got an answer for, so
            another POST of it is the reuse-detection request that revokes the
            whole family. A retry would spend it again, which is why this raises
            an AUTH error (the honest recovery is ``/mcp reauth``) rather than a
            transient one the manager would retry on its backoff ladder.

            Stripping the in-memory token first is the same load-bearing step as
            in :meth:`_refuse_unlocked_refresh`: without it the SDK's unlocked
            ``_refresh_token`` would present exactly the token we refused.
            """
            self._strip_in_memory_refresh_token(ctx)
            REFRESH_CONTENTION.record(self._refresh_coord_server_url, REFRESH_REFUSAL_UNCONFIRMED)
            raise McpRefreshUnconfirmedError(self._refresh_coord_server_url)

        async def _resync_from_store(self, ctx: Any) -> None:
            """Overwrite the in-memory token with the persisted one.

            Called under :func:`_oauth_refresh_lock`. Reads the store's current
            token through the same storage the provider persists through and,
            when present, adopts it into the context along with its recomputed
            expiry. Both the success path and the exception fall-through share
            this one definition of "sync from store" so the invariant (in-memory
            refresh token never older than storage) is enforced identically on
            every path.
            """
            stored = await ctx.storage.get_tokens()
            if stored is not None and stored.access_token:
                ctx.current_tokens = stored
                ctx.token_expiry_time = self._refresh_coord_storage.stored_token_expiry()

        async def _adopt_freshest_stored_token(self, ctx: Any) -> None:
            """Final under-lock re-read on the exception fall-through path.

            Guarantees the invariant even when the coordinated refresh raised:
            re-read storage under the refresh lock and adopt the freshest
            persisted token, so the SDK's subsequent unlocked refresh can never
            spend an in-memory refresh token older than what is on disk. Best
            effort — a failure here just leaves the pre-fix behaviour, never a
            raise into the request.
            """
            try:
                async with _oauth_refresh_lock(self._refresh_coord_server_url):
                    # Adopt regardless of whether the lock was taken: this is a
                    # READ, and reading the freshest persisted token is strictly
                    # better than keeping a staler in-memory one even unlocked.
                    await self._resync_from_store(ctx)
            except Exception:  # noqa: BLE001 — best-effort; never break the request
                logger.debug(
                    "MCP in-flight refresh final re-read failed for %s",
                    self._refresh_coord_server_url,
                    exc_info=True,
                )

        async def async_auth_flow(self, request):  # type: ignore[override]
            # Ensure tokens+client_info are loaded (so is_token_valid /
            # can_refresh_token below are meaningful), then coordinate the
            # refresh, then hand off to the SDK's flow. The SDK re-checks
            # is_token_valid under its own lock and will skip its refresh branch
            # because we have already made the token valid. context.lock is NOT
            # held across the coordination (it is not reentrant and the SDK
            # re-acquires it), matching how the SDK itself only holds it inside
            # the flow.
            async with self.context.lock:
                if not self._initialized:
                    await self._initialize()
            await self._coordinate_inflight_refresh()
            # Delegate to the SDK flow by hand, forwarding the RESPONSE the
            # caller sends back into each yield: httpx drives an auth flow with
            # ``gen.asend(response)``, and a plain ``async for ...: yield`` would
            # swallow those sent values (the sub-generator would see ``None`` for
            # every ``response = yield`` and crash on ``response.status_code``).
            # Manual pumping is the only correct way to delegate a receiving
            # async generator — there is no ``yield from`` for them.
            #
            # The whole pump is wrapped so the inner generator is ALWAYS closed
            # and any exception/cancellation is delivered INTO it, never dropped
            # on the floor. This matters because the SDK's ``async_auth_flow``
            # holds ``context.lock`` across its entire body: httpx re-raises a
            # transport error mid-flow (``_send_handling_auth`` does
            # ``raise exc`` after ``response.aclose()``), and if we let that
            # unwind past us without closing ``inner``, the SDK generator is
            # suspended forever at its ``yield`` still holding the lock — every
            # later request to this server then deadlocks on ``context.lock``.
            # ``athrow`` runs the SDK's own ``finally`` (which releases the lock
            # via the ``async with`` exit); ``aclose`` covers the GeneratorExit
            # path when httpx closes the OUTER flow (its ``finally:
            # await auth_flow.aclose()``).
            inner = super().async_auth_flow(request)
            # One 401-driven token adoption is allowed per flow invocation.
            # ``original_request`` is the caller's request object itself (NOT
            # the first yield: when the loaded token is expired the SDK yields
            # a refresh request first, and latching on the first yield would
            # point the identity guard at the wrong object). The SDK re-auth
            # machinery never yields this object again except its own end-of-flow
            # retry, so ``outgoing is original_request`` reliably identifies the
            # caller's request; ``adoption_attempted`` spends the one-retry
            # budget — a second 401 must pass through untouched.
            original_request: Any = request
            adoption_attempted = False
            try:
                try:
                    outgoing = await inner.__anext__()
                except StopAsyncIteration:
                    return
                while True:
                    try:
                        response = yield outgoing
                    except GeneratorExit:
                        # The outer flow is being closed (httpx's finally, or a
                        # cancellation): close the inner one so the SDK unwinds
                        # its lock, then let the close propagate.
                        raise
                    except BaseException as exc:  # noqa: BLE001
                        # An exception thrown INTO us (httpx ``athrow`` on a
                        # transport fault): deliver it into the SDK generator so
                        # its ``finally`` runs and the lock is released, then
                        # relay whatever it yields or re-raises.
                        try:
                            outgoing = await inner.athrow(exc)
                        except StopAsyncIteration:
                            return
                        continue
                    if (
                        response is not None
                        and response.status_code == 401
                        and not adoption_attempted
                        and outgoing is original_request
                    ):
                        # The coordination step above only fires when the loaded
                        # token is EXPIRED, but Notion revokes every previously
                        # issued access token the moment any sibling process
                        # rotates the grant. Every other live session then holds
                        # a locally-VALID, server-side-REVOKED token: it skips
                        # coordination, sends the corpse, and this 401 is what
                        # comes back. The SDK's answer to a 401 is the FULL
                        # browser authorization (non-interactive connects turn
                        # that into McpAuthRequiredError and suspend
                        # auto-reconnect), even though the shared store may
                        # already hold the sibling's fresh token. So before the
                        # 401 reaches the SDK, recover under the refresh lock —
                        # by adopting a peer's DIFFERENT stored token when there
                        # is one, and otherwise by performing the refresh HERE,
                        # under that lock, rather than letting the SDK do it
                        # unlocked (see :meth:`_recover_from_401_once`).
                        #
                        # Both arms are bounded to ONE attempt per flow: if the
                        # adopted or refreshed token ALSO 401s (a genuinely dead
                        # grant, e.g. the user revoked access server-side), the
                        # second 401 passes through to the SDK's own full-flow
                        # branch exactly as before — recovery never loops.
                        adoption_attempted = True
                        retry_request = await self._recover_from_401_once(original_request)
                        if retry_request is not None:
                            # Re-yield the retry request: the SAME httpx client
                            # that sent the original sends this one (same
                            # proxies, TLS, and event hooks), exactly matching
                            # the SDK's own end-of-flow 401 retry, which
                            # re-yields the request after ``_add_auth_header``.
                            # Then feed the retry's response into the SDK
                            # generator INSTEAD of the 401, so its full
                            # browser-authorization branch never sees a
                            # challenge.
                            try:
                                retry_response = yield retry_request
                            except GeneratorExit:
                                # The outer flow is closing: re-raise so the
                                # ``finally`` below closes the inner generator
                                # and the SDK unwinds its lock.
                                raise
                            except BaseException as exc:  # noqa: BLE001
                                # A transport fault on the retry: deliver it into
                                # the SDK generator so its ``finally`` runs and
                                # the lock is released, then relay what it yields.
                                try:
                                    outgoing = await inner.athrow(exc)
                                except StopAsyncIteration:
                                    return
                                continue
                            try:
                                outgoing = await inner.asend(retry_response)
                            except StopAsyncIteration:
                                return
                            continue
                        # No recovery was possible: fall through and hand the 401
                        # to the SDK unchanged (the dead-grant behaviour).
                        # ``can_refresh_token()`` is False by then, so the SDK's
                        # own refresh branch cannot run and no unlocked token
                        # POST follows.
                    # ``response`` is whatever the caller sent into the flow;
                    # httpx always sends a real Response, so the None case is a
                    # type-narrowing artifact, not a reachable state.
                    assert response is not None  # noqa: S101 — httpx never sends None
                    try:
                        outgoing = await inner.asend(response)
                    except StopAsyncIteration:
                        return
            finally:
                # Unconditional: a normal return, a raise, or a GeneratorExit all
                # pass through here, so the SDK generator (and its held lock) is
                # never left suspended. Idempotent if already exhausted/closed.
                await inner.aclose()

        async def _recover_from_401_once(self, original_request: Any) -> Any:
            """Recover from a 401 and return the request to retry, or ``None``.

            Returns the ORIGINAL request with its ``Authorization`` header
            rewritten when there is a working token to send, and ``None`` when
            the 401 must pass through to the SDK. Both arms run under ONE
            acquisition of the refresh lock, because both are decisions about a
            rotating grant and both must see a settled store:

            * **Adoption** (unchanged from the previous behaviour): the store
              holds an access token DIFFERENT from the one in memory, so a
              sibling rotated the grant; copy it and retry. Adoption spends
              nothing, so it can never double-spend, and the lock only
              serializes the re-read against a concurrent rotation.
            * **Refresh**: the store holds the SAME token (or none) and the grant
              is still refreshable. A 401 is then NOT evidence of a dead grant —
              it is evidence that nobody has rotated yet — so the refresh is
              performed HERE, under the lock, with the same re-read and endpoint
              discipline as the pre-request coordinator. Before this existed,
              this case fell straight through to the SDK, whose
              ``async_auth_flow`` POSTs ``current_tokens.refresh_token``
              directly: no lock, no re-read, and with the sibling's rotation
              already spent the request is a reuse-detection double-spend. It is
              also the one place a genuinely dead grant can be TOMBSTONED from a
              401 — :func:`_refresh_oauth_token_locked` writes the marker on a
              parsed ``invalid_grant``, which this path previously never
              reached, so every later boot re-POSTed the same rejected token.

            ``"dead"``, ``"failed"``, ``"contended"`` and ``"overran"`` all
            strip the in-memory refresh token and return ``None``: the SDK's own
            unlockable refresh cannot run either way, and the 401 reaches the
            SDK's authorization branch. They are deliberately NOT converted into
            a retryable error on this path — after a 401 the manager's
            ``_AuthChallengeWatcher`` has already latched, so a raise here is
            reclassified as McpAuthChallengeError and blocks on auth regardless.

            ``"unacknowledged"`` is the exception, and it is raised: the
            disposition is the same (auth is required), but the challenge text
            ("authorization expired") would be untrue for a refresh that was
            SENT and never confirmed, and that string is what the user reads.
            The raise carries the same ledger-backed re-voicing as the
            coordinator's refusal, so the manager renders the honest reason.
            """
            from local_operator.mcp import auth as auth_mod

            outcome: RefreshOutcome | None = None
            try:
                async with _oauth_refresh_lock(self._refresh_coord_server_url) as lock:
                    stored = await self.context.storage.get_tokens()
                    current = self.context.current_tokens
                    if (
                        stored is not None
                        and stored.access_token
                        and (current is None or stored.access_token != current.access_token)
                    ):
                        # Adopt the peer's rotation: spend nothing, invalidate
                        # nothing, just send the token the server already issued.
                        self.context.current_tokens = stored
                        self.context.token_expiry_time = (
                            self._refresh_coord_storage.stored_token_expiry()
                        )
                        original_request.headers["Authorization"] = f"Bearer {stored.access_token}"
                        logger.debug(
                            "MCP 401 for %s: retrying with a peer-rotated stored token",
                            self._refresh_coord_server_url,
                        )
                        return original_request
                    if not self.context.can_refresh_token():
                        # Nothing stored to adopt and nothing left to spend: the
                        # SDK's full authorization branch is the right answer,
                        # and with no refresh token it cannot POST one.
                        return None
                    endpoints = self._refresh_coord_endpoints
                    if endpoints is None:
                        endpoints = auth_mod._fallback_endpoints_for(self._refresh_coord_server_url)
                    # The exchange re-reads under this lock and takes the release
                    # over, exactly as at the other two sites: nothing a waiter's
                    # cancellation or timeout does can free it mid-POST.
                    outcome = await _refresh_oauth_token_locked(
                        self._refresh_coord_server_url,
                        self._refresh_coord_storage,
                        endpoints,
                        lock=lock,
                    )
                    if outcome == "refreshed":
                        await self._resync_from_store(self.context)
                        fresh = self.context.current_tokens
                        if fresh is not None and fresh.access_token:
                            original_request.headers["Authorization"] = (
                                f"Bearer {fresh.access_token}"
                            )
                            logger.debug(
                                "MCP 401 for %s: retrying after a locked refresh",
                                self._refresh_coord_server_url,
                            )
                            return original_request
                        # The refresh reported success but nothing persistable came
                        # back; fall through to the refusal below and let the
                        # authorization branch handle the 401.
            except Exception:  # noqa: BLE001 — best-effort; never break the request
                # A failed re-read/refresh must not break the flow, and it must
                # not leave a refreshable token for the SDK to POST unlocked
                # either.
                logger.debug(
                    "MCP 401 recovery failed for %s",
                    self._refresh_coord_server_url,
                    exc_info=True,
                )
                self._strip_in_memory_refresh_token(self.context)
                return None
            # Refusals are decided OUT HERE so the ``except Exception``
            # best-effort arm above cannot swallow them (it would put the SDK's
            # unlocked refresh back on the wire, which is the one outcome this
            # whole provider exists to prevent).
            self._strip_in_memory_refresh_token(self.context)
            if outcome == "unacknowledged":
                self._refuse_unconfirmed_exchange(self.context)
            return None

        @staticmethod
        def _strip_in_memory_refresh_token(ctx: Any) -> None:
            """Drop the in-memory refresh token so the SDK cannot POST it unlocked.

            Every refusal in this provider ends here, and the strip is what
            makes the refusal COMPLETE rather than cosmetic: returning from
            ``async_auth_flow`` hands the context to the SDK's own flow, whose
            ``_refresh_token`` POSTs ``ctx.current_tokens.refresh_token``
            directly — no lock, no re-read. ``can_refresh_token()`` reads False
            once this is done, so the SDK skips its refresh branch and goes to
            the authorization branch, which the non-interactive redirect handler
            turns into an actionable McpAuthRequiredError. Suppressing OUR
            refresh while leaving the token in place would keep the one harmful
            POST and drop the harmless ones — strictly worse than not
            suppressing at all, because it also looks fixed.
            """
            with contextlib.suppress(Exception):
                if ctx.current_tokens is not None:
                    ctx.current_tokens.refresh_token = None

    return _RefreshCoordinatingOAuthProvider(**kwargs)


def build_oauth_provider(
    server_url: str,
    cfg: MCPServerConfig,
    store: StructuralAuthStore | None = None,
    *,
    interactive: bool = True,
    endpoints: DiscoveredOAuthEndpoints | None = None,
) -> Any:
    """An ``OAuthClientProvider`` that knows when its stored token expires.

    The SDK reloads tokens on first use but NOT their deadline
    (``OAuthClientProvider._initialize`` sets ``current_tokens`` and leaves
    ``token_expiry_time`` at ``None``), and ``OAuthContext.is_token_valid``
    reads a missing deadline as "still good". So every fresh process presented
    a day-old access token, got a 401, and ran the FULL browser authorization
    — the refresh token sitting in the same row was never spent, because the
    refresh branch is only reached when the token is known to be expired.

    Priming the deadline from what we persisted is the whole fix: an expired
    token now takes the refresh grant, silently, with no browser. It is set
    after construction rather than passed in because the SDK offers no
    constructor argument for it, and ``_initialize`` does not clear it.

    ``interactive`` is forwarded to the flow (see :func:`wire_oauth_auth`).
    ``endpoints`` — the result of :func:`ensure_mcp_oauth_fresh` — primes the
    provider's authorization-server metadata, so a token that dies MID-session
    and needs an in-flow refresh targets the real token endpoint instead of the
    SDK's ``<server_base>/token`` guess (which 404s for providers like Datadog
    whose token endpoint lives on a different host).
    """
    # The flow is created HERE (not inside wire_oauth_auth) so it can be
    # attached to the provider: an abandoned grant arrives at the connect as
    # a raw CancelledError (see ``LoopbackAuthFlow.callback_handler``), and
    # the ABANDONED_GRANTS ledger keyed by this object is what identifies it.
    # Its redirect URI must match the one wire_oauth_auth computes, which a
    # config override can change — one helper keeps the two from drifting.
    flow = LoopbackAuthFlow(
        _resolve_redirect_uri(cfg), server_url=server_url, interactive=interactive
    )
    kwargs = wire_oauth_auth(server_url, cfg, store=store, interactive=interactive, flow=flow)
    storage = kwargs["storage"]
    # A refresh-coordinating provider (not the bare SDK one): its in-flow
    # refresh re-reads the store under the cross-process lock so several
    # long-lived sessions cannot double-spend a rotating refresh token
    # mid-session — see :func:`_make_refresh_coordinating_provider`.
    provider = _make_refresh_coordinating_provider(
        kwargs, server_url=server_url, storage=storage, endpoints=endpoints
    )
    provider._loopback_flow = flow  # type: ignore[attr-defined]
    if endpoints is not None:
        provider.context.oauth_metadata = endpoints.oauth_metadata
        provider.context.protected_resource_metadata = endpoints.protected_resource_metadata
        provider.context.auth_server_url = endpoints.auth_server_url
    try:
        expiry = storage.stored_token_expiry()
    except Exception:  # noqa: BLE001 — a metadata read must not block a connect
        logger.debug("MCP token expiry unreadable for %s", server_url, exc_info=True)
        return provider
    if expiry is not None:
        provider.context.token_expiry_time = expiry
    return provider
