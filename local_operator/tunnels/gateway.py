"""Authenticated, streaming loopback adapter for personal harnesses.

The Radient Worker rejects anonymous traffic at the edge; this gateway verifies
its short-lived, request-bound assertion before reaching a harness.
Loopback or a header's presence is never evidence of cloud authentication.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import time
from collections.abc import Callable, Mapping
from typing import Any

import httpx
import jwt
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import (
    JSONResponse,
    RedirectResponse,
    Response,
    StreamingResponse,
)
from starlette.routing import Route, WebSocketRoute
from starlette.websockets import WebSocket, WebSocketDisconnect

from local_operator.mobile.auth import COOKIE_NAME, sign_cookie
from local_operator.tunnels.config import validate_connection

MAX_BODY_BYTES = 10 * 1024 * 1024
MAX_STREAM_SECONDS = 60
AUTHORIZATION_LEASE_SECONDS = 30
PROOF_HEADER = "x-radient-tunnel-assertion"
# Why the gateway refused a relayed request, and what each surface says about it.
#
# The poller is the only component that sees the control plane's answer, so it
# records the reason on the gateway and both the phone's 503 and `lop tunnel
# status` read it back: one flat message for every cause is what made a computer
# that had merely lost its network read exactly like a revoked tunnel or a lapsed
# plan, and sent the operator to re-enroll a tunnel that was healthy.
#
# The copy is split by surface because the two can act differently — a phone
# browser has no command surface of its own, while a terminal is where this
# package's commands exist — and it lives here, with the vocabulary, so one cause
# cannot drift into two differently-worded sentences. Every string is a module
# literal: a detail reaches a phone render and an operator's support thread, so
# it must never carry an upstream body, a request URL, or a credential.
CONSOLE_URL = "https://console.radienthq.com/dashboard/tunnels"
#: The window the deferral's copy quotes, in seconds, and the ONE place this module
#: states it.
#:
#: A literal here on purpose, and the reason is the module boundary: `gateway`
#: deliberately imports neither `providers.auth_store` (which owns
#: `UNCONFIRMED_SEND_TTL_S`, derived there as one exchange window plus one
#: per-operation margin) nor `tunnels.report`, because it is the one tunnel module
#: the desktop server must not pull on boot — `tests/unit/test_import_graph.py`
#: pins that — and none of this module's readers needs a credential store to read a
#: sentence. So the number is declared here, used by BOTH deferral sentences and by
#: the `Retry-After` header so the header and the copy cannot disagree, and
#: `tests/unit/test_tunnels.py` holds it EQUAL to the store's derived bound — drift
#: fails a test instead of reaching a phone's screen (design round 1, D5; QA round
#: 1, Q1).
DEFERRAL_WINDOW_S = 120
UNREACHABLE = "control_plane_unreachable"
REFUSED = "authorization_refused"
#: The credential store DEFERRED the refresh: the refresh token's last exchange is
#: not settled, so presenting it again could revoke the whole token family. Not a
#: refusal and not a dead login — the store is waiting, the connector keeps
#: retrying, and the state clears itself within about two minutes — so it gets
#: copy of its own rather than the `REFUSED` sentence, which blamed the login for
#: a state the login had nothing to do with (that sentence is what a phone
#: rendered as "the authentication expired" through the whole outage).
AUTHORIZATION_DEFERRED = "authorization_deferred"
NOT_AUTHORIZED = "tunnel_not_authorized"
LEASE_PENDING = "authorization_lease_pending"
# The connector's own terminal states, not relay refusals: the connector exited
# (successfully, so its supervisor does not retry it) because something needs a
# person. They travel in this vocabulary anyway, so the sentence an operator
# reads in `lop tunnel status`, the one a parked connector writes to
# `state.json`, and the 503 a phone would see if a live gateway ever reported
# one cannot drift into several wordings of one cause.
#
# `LOGIN_REQUIRED` is the one the incident needed. The other two only ever park:
# a live gateway never refuses relayed traffic for them, which is why they have
# no `RELAY_DETAIL` entry — inventing phone copy for a state no phone can reach
# would be copy that nothing can ever display.
LOGIN_REQUIRED = "login_required"
LOCAL_PREREQUISITE = "local_prerequisite"
REENROLMENT_REQUIRED = "reenrolment_required"

# What a phone is shown: self-contained, and carrying any link it needs, because
# there is nothing to type on that surface.
RELAY_DETAIL = {
    UNREACHABLE: (
        "This computer could not reach Radient to renew the relay authorization (its "
        "network may be down, or Radient may be unreachable). It reauthorizes by itself "
        "once the control plane answers again — check this computer's network "
        "connection if it does not clear."
    ),
    REFUSED: (
        "Radient refused this computer's relay authorization check. The Radient login "
        "may have expired, or this tunnel's billing may be inactive. Sign in again on "
        f"that computer, and check billing at {CONSOLE_URL}."
    ),
    # What the phone is shown, and the two things the first draft left out (design
    # round 1, D1/D3; UX round 1, U1): the READER'S OWN ACTION — this body IS the
    # page, nothing refreshes it, and a reader who waits the window out and reloads
    # nothing sees byte-identical text — and the ESCALATION, because a stalling
    # endpoint re-arms a fresh window (see DEFERRAL_WINDOW_S) and a deferral that
    # keeps returning is exactly where a sign-in becomes the remedy. What it still
    # may not claim: that the login expired, that a sign-in is needed for THIS state,
    # or that the page updates on its own.
    AUTHORIZATION_DEFERRED: (
        "Remote access is paused: the computer running this tunnel is waiting for "
        "Radient to confirm a sign-in refresh. It clears by itself within about two "
        "minutes; reload this page to check. If it is still paused after that, sign "
        "in again on that computer."
    ),
    # The console is the remedy on both surfaces, so this one sentence serves both.
    NOT_AUTHORIZED: (
        "Radient is not authorizing this tunnel: it was revoked, suspended, disabled, "
        f"stopped on this computer, or changed in the console. Review it at {CONSOLE_URL}."
    ),
    LEASE_PENDING: (
        "The relay has not renewed its authorization yet. This normally clears by "
        "itself within a few seconds."
    ),
    LOGIN_REQUIRED: (
        "This computer's Radient login is no longer valid, so remote access is off "
        "until it is signed in again on this computer (check this tunnel's billing at "
        f"{CONSOLE_URL})."
    ),
}

# What `lop tunnel status` prints for the same cause. The two entries a PARKED
# connector's sentence travels in name no command, and by the same rule the shared
# copy is never allowed one: the park's `detail` is written into `state.json`, is
# forwarded to the desktop as `connector.detail`, and is rendered by the TUI card —
# so a command baked in here is a command some of those readers cannot run. Each
# surface appends its own spelling of `TERMINAL_REMEDY` instead. (`REFUSED` carried
# the same `/login radient` prose, so it took the same pass.) The lease-pending
# entry is the deliberate exception, and it is not the same thing: its sentence is
# advice for the terminal that is printing it — retry in a moment — and both
# commands in it are shell commands, which is also why the entry repeats the
# command that is printing the line, the circularity this split exists to avoid.
TERMINAL_DETAIL = {
    UNREACHABLE: (
        "This computer could not reach Radient to renew the relay authorization (its "
        "network may be down, or Radient may be unreachable). It retries every 10 "
        "seconds and reauthorizes by itself once the control plane answers; no local "
        "command is needed."
    ),
    REFUSED: (
        "Radient refused the connector's authorization check, so the relay stopped "
        "serving. Signing in again, or checking this tunnel's billing, is what clears "
        f"it: {CONSOLE_URL}."
    ),
    # COMMAND-FREE for the rule above, and because this sentence travels further than
    # this module's own surface: it is written into the park file, forwarded to the
    # desktop as `connector.detail` and rendered by the TUI card — the desktop
    # callout has no `Login:` line, so the window has to be HERE (design round 1,
    # D4). The first draft spent 15 words on the token-family mechanism, which is why
    # the store waits rather than what happened to the reader — that rationale lives
    # in the comment above the reason code — and its "no local command is needed"
    # absolute is gone for the same reason the phone's is (UX round 1, U1): the
    # window re-arms, so the remedy belongs on the persisting case.
    AUTHORIZATION_DEFERRED: (
        "Radient has not confirmed this computer's last sign-in refresh, so the "
        "connector is waiting for it to settle rather than sending that request "
        "again. It retries every 10 seconds and clears by itself within about two "
        "minutes; sign in again only if it persists past that."
    ),
    LEASE_PENDING: (
        "The relay has not renewed its authorization yet. It retries every 10 seconds "
        "and usually clears a few seconds after the connector starts. Run lop tunnel "
        "status again shortly; if it persists, run lop tunnel install."
    ),
    # COMMAND-FREE, and that is the point of this entry rather than a style
    # choice: the sentence travels further than this surface. It is written into
    # the park file, printed verbatim by `lop tunnel status`, and forwarded to
    # the desktop as `connector.detail` (DESKTOP_API.md), so a command baked in
    # here is a command every one of those surfaces has to be able to run. The
    # one it used to carry (`/login radient`) is a TUI slash command, which a
    # shell answers with `no such file or directory` and a desktop callout can
    # only render as text. Each surface appends `TERMINAL_REMEDY` in its own
    # spelling instead (the CLI's `Login:` line and the TUI card do exactly
    # that), and `test_tunnels.py` holds this sentence to naming no command at
    # all.
    #
    # No console URL either: the terminal prints the billing block on the line
    # that is about billing when there is anything to bill, and a dead grant is
    # not a billing event. The PHONE copy above keeps its link because a phone
    # has no billing block to read it from.
    #
    # LENGTH: 138 cells, two rendered rows at 80 columns as the indented
    # continuation `lop tunnel status` prints (140 cells with that two-cell
    # indent, which is how QA round 1's Q1 counted it). Deliberately NOT trimmed
    # on the round-2 pass, and the reason is what the sentence has to carry for
    # readers that have no other line: the CAUSE (a Radient login that is no
    # longer valid), the CONSEQUENCE that makes this a park rather than a retry
    # (it stopped and will not retry by itself — the incident's own shape, and the
    # one clause a reader cannot recover from any other surface once the
    # connector has exited), and the REMEDY's shape (signing in again starts it
    # again on its own, i.e. no restart command is needed). It is also the copy
    # `state.json` persists and `DESKTOP_API.md` forwards as `connector.detail`,
    # so a shorter sentence here is a shorter sentence everywhere. What D2 fixed
    # was the WELDING — this sentence is now its own indented continuation row
    # under a ONE-ROW state line, wrapping in the terminal the way prose does,
    # rather than being the third clause of a 317-cell paragraph that buried the
    # command. A future trim should weigh those three clauses against each other
    # rather than against the row count.
    LOGIN_REQUIRED: (
        "The connector's Radient login is no longer valid, so it stopped and will not "
        "retry by itself. Signing in again starts it again on its own."
    ),
}

#: The ONE command that clears each cause, as a value: the surface-specific half
#: of the copy, kept beside the sentences so the two cannot disagree. The
#: sentence above describes the condition; this table names the fix, and each
#: surface renders it in its own spelling — the CLI prints it verbatim
#: (`Login: sign-in expired — run lop login radient`), the TUI maps it to the
#: command a composer can run (`/login radient`), the park file persists it for
#: every reader, and the desktop route forwards it as `remedy.command`. The rule
#: this table exists to enforce is therefore the reverse of what a sentence
#: carrying its own command enforced: nothing is duplicated, so nothing can
#: drift.
#: `UNREACHABLE` and `LEASE_PENDING` name the CHECK rather than a fix, because
#: their copy says the connector clears those by itself: handing the operator a
#: repair command there would send them to fix something that is not broken.
TERMINAL_REMEDY = {
    UNREACHABLE: "lop tunnel status",
    REFUSED: "lop login radient",
    NOT_AUTHORIZED: "lop tunnel connect",
    LEASE_PENDING: "lop tunnel status",
    LOGIN_REQUIRED: "lop login radient",
    # Names the CHECK rather than a fix, by the rule above: this state clears itself,
    # and handing the operator `lop login radient` for it would be the misdirection
    # this code exists to remove.
    AUTHORIZATION_DEFERRED: "lop tunnel status",
    # The two local-repair codes: their sentences are the failure's own fixed
    # literal rather than an entry above (each one names its own missing
    # prerequisite, which no single sentence could), so this table is where the
    # COMMAND the operator has to run is pinned in one place.
    LOCAL_PREREQUISITE: "lop tunnel install",
    REENROLMENT_REQUIRED: "lop tunnel connect",
}


#: Readable forms of the reason codes, for surfaces that show a short state on
#: one line (`Connector: parked — login required (since 14:32)`). The CODE is
#: what travels in `state.json` and in JSON output; this is only its label, and
#: it lives beside the vocabulary so a new code cannot ship without one.
REASON_LABEL = {
    UNREACHABLE: "control plane unreachable",
    REFUSED: "authorization refused",
    AUTHORIZATION_DEFERRED: "sign-in refresh deferred",
    NOT_AUTHORIZED: "not authorized",
    LEASE_PENDING: "waiting for authorization",
    LOGIN_REQUIRED: "login required",
    LOCAL_PREREQUISITE: "prerequisite missing",
    REENROLMENT_REQUIRED: "needs re-enrolment",
}


def reason_label(reason: str) -> str:
    """The short label for a reason code, or the code itself if unknown.

    A code this build does not know is shown verbatim rather than as "unknown":
    a daemon from another version naming its own state is more useful to an
    operator than a shrug, and `terminal_detail` already takes that position.
    """
    return REASON_LABEL.get(reason, reason)


def terminal_detail(reason: str, relay_detail: str) -> str:
    """The sentence `lop tunnel status` prints for a refusal reason.

    Falls back to the relay's own detail for a reason this build does not know —
    a daemon from another version — rather than saying nothing about a cause the
    daemon plainly named.
    """
    return TERMINAL_DETAIL.get(reason, relay_detail)


# Only presentation/protocol headers cross the boundary. In particular the
# owner's Radient cookies and bearer must never reach a local harness,
# whose plugins/tools may log, reflect, or export request headers.
_REQUEST_HEADERS = {
    "accept",
    "accept-language",
    "content-type",
    "range",
    "if-none-match",
    "if-modified-since",
    "last-event-id",
    "origin",
}
_RESPONSE_HEADERS = {
    "content-type",
    "content-length",
    "content-encoding",
    "content-range",
    "accept-ranges",
    "etag",
    "last-modified",
    "x-accel-buffering",
}


class OriginVerifier:
    """Keys pinned by an authenticated /connect response, never by a request.

    Only RS256 public keys from the trusted control plane are admitted. A
    token-selected kid can select one of them, never trigger a URL retrieval.
    Rotating the origin key increments the tunnel version and reconnects.
    """

    def __init__(self, access: dict[str, Any], client: httpx.AsyncClient) -> None:
        self.access = access
        self.client = client
        self.keys: dict[str, Any] = {}
        for row in access["jwks"]["keys"]:
            # A repeated kid makes key selection ambiguous, and a dict
            # comprehension resolves that ambiguity silently by position: the
            # last row wins. A JWKS carrying both the real key and a second key
            # under the same kid would then be accepted, with which one
            # verifies tokens decided by serialization order rather than by
            # policy. Refuse the whole set instead of picking a winner.
            if row["kid"] in self.keys:
                raise ValueError("Duplicate key identifier in pinned origin keys.")
            self.keys[row["kid"]] = jwt.PyJWK.from_dict(row).key
        if any(key.key_size < 2048 for key in self.keys.values()):
            raise ValueError("Origin proof requires RSA keys of at least 2048 bits.")
        self.used: dict[str, float] = {}

    async def verify(
        self,
        token: str,
        *,
        host: str,
        harness_id: str,
        method: str,
        target: str,
        body: bytes,
        websocket: bool = False,
    ) -> dict[str, Any]:
        if not token or len(token) > 16384:
            raise ValueError("Missing origin assertion.")
        header = jwt.get_unverified_header(token)
        kid = header.get("kid")
        if header.get("alg") != "RS256" or not isinstance(kid, str):
            raise ValueError("Invalid origin assertion.")
        key = self.keys.get(kid)
        if key is None:
            raise ValueError("Origin signing key unavailable.")
        claims = jwt.decode(
            token,
            key,
            algorithms=["RS256"],
            audience=host,
            issuer=self.access["issuer"],
            options={"require": ["exp", "iat", "iss", "aud", "sub", "jti"]},
        )
        expected = {
            "aud": host,
            "sub": self.access["owner_account_id"],
            "tunnel_id": self.access["tunnel_id"],
            "version": self.access["version"],
            "harness_id": harness_id,
            "method": method,
            "target": target,
            "body_sha256": hashlib.sha256(body).hexdigest(),
        }
        if any(claims.get(k) != value for k, value in expected.items()):
            raise ValueError("Origin assertion does not authorize this request.")
        if not isinstance(claims.get("version"), int) or isinstance(claims["version"], bool):
            raise ValueError("Invalid origin assertion version.")
        issued, expires = claims["iat"], claims["exp"]
        if (
            not isinstance(issued, int)
            or isinstance(issued, bool)
            or not isinstance(expires, int)
            or isinstance(expires, bool)
            or not 0 < expires - issued <= 30
        ):
            raise ValueError("Origin assertion lifetime exceeds thirty seconds.")
        # A replayed idempotent read yields the same bytes the captor already
        # holds, so GET/HEAD/OPTIONS are exempt to keep the nonce cache small.
        # A WebSocket upgrade is signed as method="GET" but is NOT idempotent:
        # replaying it opens an ADDITIONAL live bidirectional channel to the
        # harness, which the captor can then drive. It therefore consumes the
        # nonce like a mutation. The edge mints a fresh jti per request
        # (Worker setJti(crypto.randomUUID()) on the single forward path that
        # upgrades take), so legitimate rapid reconnects each carry their own
        # nonce and are unaffected.
        if websocket or method not in {"GET", "HEAD", "OPTIONS"}:
            # No await between check and insertion: concurrent replays on this
            # event loop cannot both pass. Refuse a full cache rather than
            # evict a still-live nonce and reopen its replay window.
            now = time.time()
            self.used = {key: expiry for key, expiry in self.used.items() if expiry > now}
            nonce = claims["jti"]
            if (
                not isinstance(nonce, str)
                or not nonce
                or nonce in self.used
                or len(self.used) >= 10000
            ):
                raise ValueError("Replayed origin assertion.")
            self.used[nonce] = float(expires)
        return claims


class Gateway:
    def __init__(
        self,
        connection: dict[str, Any],
        client: httpx.AsyncClient,
        *,
        mobile_password: str | None = None,
        opencode_basic: dict[str, str] | None = None,
        connector_ready: Callable[[], bool] | None = None,
    ) -> None:
        self.connection = validate_connection(connection)
        self.client = client
        self.verifier = OriginVerifier(connection["origin_auth"], client)
        self.mobile_password = mobile_password
        self.opencode_basic = opencode_basic
        self.connector_ready = connector_ready or (lambda: False)
        self.authorized_until = time.monotonic() + AUTHORIZATION_LEASE_SECONDS
        self.revoked = False
        # The last reason the poller could not renew the lease, as a `reason`
        # code. Every successful renewal clears it.
        self.authorization_failure: str | None = None

    def authorize(self) -> None:
        self.authorized_until = time.monotonic() + AUTHORIZATION_LEASE_SECONDS
        self.authorization_failure = None

    def note_authorization_failure(self, reason: str) -> None:
        """Record why the poller could not renew the relay lease.

        Called only from the poller (`service.authorization_failure_reason`),
        which is the one place the control plane's failure is observable. The
        reason survives until a renewal succeeds, so the phone and
        `lop tunnel status` still name the cause after the fact.
        """
        self.authorization_failure = reason

    def refusal_reason(self) -> str:
        """The reason this gateway is refusing relayed requests right now.

        A withdrawal outranks a recorded failure: both can be true at once, and
        the withdrawal is the one the operator has to act on.
        """
        if self.revoked:
            return NOT_AUTHORIZED
        return self.authorization_failure or LEASE_PENDING

    def refusal(self) -> dict[str, str]:
        """The `reason`/`detail` pair for the current refusal, without a status.

        Shared by the 503 body and the health payload so the phone and the
        terminal cannot disagree about the cause.
        """
        reason = self.refusal_reason()
        return {"detail": RELAY_DETAIL[reason], "reason": reason}

    def unavailable_body(self) -> dict[str, str]:
        """The phone-facing 503 body.

        `detail` leads deliberately: a phone renders this body as JSON in its
        browser, where the human sentence is the part that names the cause and
        the remedy, and it must not sit behind a generic error string and a
        machine token. `error` keeps its long-standing value for anything that
        keys on it.
        """
        body = self.refusal()
        body["error"] = "tunnel authorization unavailable"
        return body

    def refusal_headers(self) -> dict[str, str]:
        """Headers for the refusal response, when a reason has one to add.

        Only the DEFERRAL does: it is the one refusal with a known bound, so
        `Retry-After` is what lets a client — a phone's fetch, a script, an
        operator with curl — wait the right amount instead of guessing, and it is
        the same number the sentence quotes (`DEFERRAL_WINDOW_S`), so the header
        and the copy cannot drift apart (QA round 1, Q1; UX round 1, U3). A
        withdrawal or an unknown cause gets no header: there is no honest number
        to give, and inventing one would be the same promise the copy refuses to
        make.
        """
        if self.refusal_reason() != AUTHORIZATION_DEFERRED:
            return {}
        return {"Retry-After": str(DEFERRAL_WINDOW_S)}

    @staticmethod
    def target(scope: Mapping[str, Any]) -> str:
        return (
            scope["raw_path"] + (b"?" + scope["query_string"] if scope["query_string"] else b"")
        ).decode("ascii")

    def harness(self, host: str) -> dict[str, Any] | None:
        return next(
            (
                h
                for h in self.connection["tunnel"]["harnesses"]
                if h["enabled"] and h["hostname"] == host
            ),
            None,
        )

    def headers(self, incoming: Any, host: str, harness: dict[str, Any]) -> dict[str, str]:
        headers = {k: v for k, v in incoming.items() if k in _REQUEST_HEADERS}
        headers["host"] = host
        # OpenCode 1.18.5 requires this explicit request marker to issue its
        # short-lived PTY WebSocket ticket. Preserve only its defined value for
        # that harness; never synthesize it or relax Origin/proof verification.
        if harness["id"] == "opencode" and incoming.get("x-opencode-ticket") == "1":
            headers["x-opencode-ticket"] = "1"
        if harness["id"] == "local-operator":
            if not self.mobile_password:
                raise ValueError("Mobile relay is not installed.")
            headers["cookie"] = f"{COOKIE_NAME}={sign_cookie(self.mobile_password)}"
        elif self.opencode_basic:
            credential = self.opencode_basic["username"] + ":" + self.opencode_basic["password"]
            headers["authorization"] = "Basic " + base64.b64encode(credential.encode()).decode()
        return headers

    async def handle(self, request: Request) -> Response:
        host = request.headers.get("host", "").lower()
        if (
            request.url.path == "/_lop_tunnel/health"
            and host == f"127.0.0.1:{self.connection['gateway_port']}"
        ):
            payload: dict[str, Any] = {
                "ok": not self.revoked and time.monotonic() < self.authorized_until,
                "connected": self.connector_ready(),
            }
            if not payload["ok"]:
                # The same cause the phone is given, so `lop tunnel status` can
                # print why the relay is refusing instead of a bare state. No
                # `error` key: this response is a success, and `error` is this
                # gateway's failure-body field everywhere else.
                payload.update(self.refusal())
            return JSONResponse(payload)
        if self.revoked or time.monotonic() >= self.authorized_until:
            return JSONResponse(
                self.unavailable_body(), status_code=503, headers=self.refusal_headers()
            )
        harness = self.harness(host)
        if harness is None:
            return JSONResponse({"error": "unknown tunnel host"}, status_code=404)
        expected_origin = "https://" + host
        origin = request.headers.get("origin")
        # A signed document navigation is how a phone opens a saved/shared
        # harness URL. Fetch Metadata marks that cross-site even though the
        # edge already authenticated it. Only safe top-level navigation gets
        # this exception; sibling fetches and browser mutations remain denied.
        navigation = (
            request.method in {"GET", "HEAD"}
            and request.headers.get("sec-fetch-mode") == "navigate"
        )
        if (
            (origin is not None and origin != expected_origin)
            or (
                request.headers.get("sec-fetch-site") in {"cross-site", "same-site"}
                and not navigation
            )
            or (request.method not in {"GET", "HEAD", "OPTIONS"} and origin != expected_origin)
        ):
            return JSONResponse({"error": "same-origin request required"}, status_code=403)
        body = bytearray()
        async for chunk in request.stream():
            body.extend(chunk)
            if len(body) > MAX_BODY_BYTES:
                return JSONResponse({"error": "request exceeds 10 MiB"}, status_code=413)
        if request.method in {"GET", "HEAD"} and body:
            return JSONResponse({"error": "GET/HEAD bodies are not supported"}, status_code=400)
        try:
            await self.verifier.verify(
                request.headers.get(PROOF_HEADER, ""),
                host=host,
                harness_id=harness["id"],
                method=request.method,
                target=self.target(request.scope),
                body=bytes(body),
            )
        except (ValueError, KeyError, TypeError, jwt.PyJWTError, httpx.HTTPError):
            return JSONResponse(
                {"error": "valid Radient origin assertion required"}, status_code=401
            )
        # Uploads can yield to the policy poller for arbitrarily long periods.
        # Authorization at request arrival cannot authorize a later mutation:
        # recheck after consuming/verifying its body, immediately before any
        # harness request or local side effect begins.
        if self.revoked or time.monotonic() >= self.authorized_until:
            return JSONResponse(
                self.unavailable_body(), status_code=503, headers=self.refusal_headers()
            )
        if request.url.path == "/logout":
            response = RedirectResponse("/_radient/logout", status_code=303)
            response.headers["Clear-Site-Data"] = '"storage"'
            return response
        if request.url.path == "/login" and harness["id"] == "local-operator":
            return RedirectResponse("/", status_code=303)
        # Preserve the checked public Host for the relay's independent Origin
        # check. The destination remains a literal loopback address regardless.
        try:
            headers = self.headers(request.headers, host, harness)
        except ValueError:
            return JSONResponse({"error": "mobile relay is not installed"}, status_code=503)
        target = httpx.URL(
            scheme="http",
            host="127.0.0.1",
            port=harness["port"],
            raw_path=request.scope["raw_path"]
            + (b"?" + request.scope["query_string"] if request.scope["query_string"] else b""),
        )
        try:
            upstream = await self.client.send(
                self.client.build_request(
                    request.method, target, headers=headers, content=bytes(body)
                ),
                stream=True,
                follow_redirects=False,
            )
        except httpx.HTTPError:
            return JSONResponse({"error": "local harness unavailable"}, status_code=502)
        response_headers = {k: v for k, v in upstream.headers.items() if k in _RESPONSE_HEADERS}
        location = upstream.headers.get("location")
        if location:
            # No credential-bearing or untrusted absolute redirect can escape
            # to another host. Root-relative harness redirects retain this AUD.
            if (
                location.startswith("/")
                and not location.startswith("//")
                and "\\" not in location
                and not any(ord(char) < 32 for char in location)
            ):
                response_headers["location"] = location
            else:
                await upstream.aclose()
                return JSONResponse({"error": "unsafe harness redirect"}, status_code=502)
        response_headers["cache-control"] = "no-store"
        response_headers["referrer-policy"] = "same-origin"
        response_headers["x-content-type-options"] = "nosniff"
        response_headers["content-security-policy"] = "frame-ancestors 'none'"

        async def stream():
            # SSE is deliberately streamed with backpressure. Reconnects at
            # most every minute recheck edge policy; a backend revoke
            # also stops reads after the next origin chunk (keepalives are 15s).
            try:
                async with asyncio.timeout(MAX_STREAM_SECONDS):
                    async for chunk in upstream.aiter_raw():
                        if self.revoked or time.monotonic() >= self.authorized_until:
                            break
                        yield chunk
            except (TimeoutError, httpx.HTTPError):
                pass
            finally:
                await upstream.aclose()

        return StreamingResponse(
            stream(), status_code=upstream.status_code, headers=response_headers
        )

    async def websocket(self, socket: WebSocket) -> None:
        """A bounded generic WebSocket adapter; LO itself uses HTTP and SSE."""
        from websockets.asyncio.client import connect
        from websockets.exceptions import WebSocketException
        from websockets.typing import Origin

        host = socket.headers.get("host", "").lower()
        harness = self.harness(host)
        if (
            harness is None
            or self.revoked
            or time.monotonic() >= self.authorized_until
            or socket.headers.get("origin") != "https://" + host
        ):
            await socket.close(code=1008)
            return
        try:
            target = self.target(socket.scope)
            await self.verifier.verify(
                socket.headers.get(PROOF_HEADER, ""),
                host=host,
                harness_id=harness["id"],
                method="GET",
                target=target,
                body=b"",
                websocket=True,
            )
            headers = self.headers(socket.headers, host, harness)
            headers.pop("host", None)
            headers.pop("origin", None)
            # URI host supplies the checked HTTP Host while connect()'s socket
            # override ensures DNS can never redirect this to a public server.
            async with connect(
                "ws://" + host + target,
                host="127.0.0.1",
                port=harness["port"],
                proxy=None,
                origin=Origin("https://" + host),
                additional_headers=headers,
                subprotocols=socket.scope.get("subprotocols") or None,
                max_size=MAX_BODY_BYTES,
            ) as upstream:
                await socket.accept(subprotocol=upstream.subprotocol)

                async def to_origin() -> None:
                    while not self.revoked and time.monotonic() < self.authorized_until:
                        message = await socket.receive()
                        if self.revoked or time.monotonic() >= self.authorized_until:
                            return
                        if message["type"] == "websocket.disconnect":
                            return
                        data = message.get("bytes")
                        if data is None:
                            data = message.get("text", "")
                        if len(data) > MAX_BODY_BYTES:
                            return
                        await upstream.send(data)

                async def to_browser() -> None:
                    async for data in upstream:
                        if self.revoked or time.monotonic() >= self.authorized_until:
                            return
                        if isinstance(data, bytes):
                            await socket.send_bytes(data)
                        else:
                            await socket.send_text(data)

                tasks = [asyncio.create_task(to_origin()), asyncio.create_task(to_browser())]
                try:
                    await asyncio.wait(
                        tasks, timeout=MAX_STREAM_SECONDS, return_when=asyncio.FIRST_COMPLETED
                    )
                finally:
                    for task in tasks:
                        task.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)
        except (
            ValueError,
            KeyError,
            TypeError,
            OSError,
            jwt.PyJWTError,
            httpx.HTTPError,
            WebSocketDisconnect,
            WebSocketException,
        ):
            pass
        finally:
            try:
                await socket.close(code=1000)
            except RuntimeError:
                pass

    def app(self) -> Starlette:
        return Starlette(
            routes=[
                Route(
                    "/{path:path}",
                    self.handle,
                    methods=["GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
                ),
                WebSocketRoute("/{path:path}", self.websocket),
            ]
        )
