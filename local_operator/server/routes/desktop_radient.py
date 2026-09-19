"""Closed Radient operations using AuthStore, never renderer tokens or refreshes.

The route census comes from the desktop's existing Radient clients. Google
Workspace consent is deliberately NOT here: effective MCP grants own that flow.
"""

from __future__ import annotations

import asyncio
import re
import time
from typing import Annotated, Any, Literal

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import Field, StrictBool, model_validator

from local_operator.providers.auth_store import (
    DEFAULT_BLOCK_MS,
    AuthStore,
    AuthStoreError,
    CredentialInvalidError,
    OAuthAccess,
    StoredCredential,
)
from local_operator.providers.registry import get_provider_definition
from local_operator.server.desktop import require_desktop
from local_operator.server.models.schemas import CRUDResponse
from local_operator.server.routes.auth import get_desktop_auth
from local_operator.server.routes.desktop_lifecycle import Result
from local_operator.server.routes.desktop_sessions import (
    Input,
    RequestID,
    errors,
    receipts,
    reply,
)
from local_operator.server.utils.desktop_auth import DesktopAuth

router = APIRouter(tags=["Desktop Radient"], dependencies=[Depends(require_desktop)])

#: The identifier shape every id in this module shares, written ONCE and anchored
#: in both engines that enforce it. The model field anchors with ``$``, which
#: pydantic's regex engine applies to the end of the value (unlike Python's
#: ``re``, it does not also accept a trailing newline, and it rejects ``\Z``
#: outright); the comma-joined list match below uses ``fullmatch``. Neither may
#: accept an id that would then reach an upstream path with whitespace riding in
#: it, and this way the two cannot drift apart.
IDENTIFIER = r"[A-Za-z0-9_-]{1,128}"
Identifier = Annotated[str, Field(pattern=rf"^{IDENTIFIER}$")]

#: How many agent ids one `agents.statuses` request may name.
#:
#: The op exists so a page of cards costs ONE renderer round trip, and its
#: fan-out is bounded by this number rather than by whatever a caller sends: the
#: hub's page size is 12, so 32 leaves room for a larger page without letting a
#: single request become an unbounded burst of upstream traffic.
#:
#: A CALLER HOLDING A LARGER PAGE MUST CHUNK IT. ``agents.list`` accepts a
#: ``per_page`` up to 100, so the page a renderer is holding can legitimately be
#: bigger than this bound: it asks about at most this many ids at a time, and a
#: 422 from this op means "chunk the list", never "no viewer state". The bound is
#: deliberately NOT raised to match ``per_page``: every id costs two reads and
#: the whole fan-out shares one ``STATUS_BATCH_TIMEOUT``, so 100 ids would be 200
#: reads in 34 sequential waves — a request that could only ever time out, which
#: is a worse answer for the renderer than a refusal it can act on at once.
STATUS_BATCH_LIMIT = 32

#: How many of those upstream reads are in flight at once.
#:
#: Each read is a tiny GET, so the cost of opening all of them at once is a
#: burst against a shared API rather than a local one; six keeps the batch
#: comfortably inside a single page's latency budget while staying a polite
#: client.
STATUS_BATCH_CONCURRENCY = 6

#: How long the WHOLE fan-out may take — not how long one read may take.
#:
#: Every other op on this transport is bounded by one read (``httpx``'s own
#: timeout), and a batched op needs the same ceiling rather than a multiple of
#: it: these reads run in sequential waves of ``STATUS_BATCH_CONCURRENCY``, so a
#: per-read timeout alone compounds — ``ceil(2N / 6) x 30 s`` is 120 s for the
#: hub's twelve-card page and 330 s at the id bound, against 30 s for every other
#: op here. That inverts the case this op exists to make: where the per-card
#: fan-out it replaces waited only on its slowest single read, an unbounded batch
#: holds the whole page for four waves' worth of latency, which is exactly the
#: regime where a hub page feels broken. Expiry answers 502
#: ``radient_upstream_timeout`` — the same status a per-read httpx timeout
#: produces — so "the upstream did not answer" is one path in the renderer
#: rather than two.
STATUS_BATCH_TIMEOUT = 30.0

#: The most of one upstream body this proxy will hold.
#:
#: Shared with the single-op path rather than written twice: "like and favourite
#: documents are small" is an assumption about an upstream this proxy does not
#: own, and a ceiling applied to one path and not the other is how a batch of
#: buffered bodies costs six times the memory one op would use.
MAX_UPSTREAM_BYTES = 2_000_000

#: How long a request waits for a PEER's refresh to land, and how often it looks.
#:
#: The credential is refreshed single-flight behind a CROSS-PROCESS lease, and the
#: store's own peer path waits one 0.05 s slice before handing back the stored row
#: (``auth_store.py``, the ``if not force: return now_data`` branch). That is right
#: for a model request, which a stale bearer may still serve, and too short here:
#: measured (review round 1; QA R2/R29) a peer with a 500 ms token endpoint turned a
#: request issued 50 ms into its refresh into a 502 for the whole window, where the
#: base answered 200 off the stale bearer. A peer that is going to land at all does
#: so well inside a second, so this is a BOUNDED join and not a wait on the lease:
#: the lease itself lives ``AUTH_REFRESH_LEASE_MS`` (30 s) and a holder that died must
#: not hold a renderer's page open for it. Past the bound the answer is the retryable
#: ``refresh_did_not_land`` refusal — the same class as before this join, now paid
#: only when the peer really did not land.
PEER_REFRESH_WAIT_S = 1.0

#: How often that wait re-reads the row. A read of the stored row, not a refresh
#: call: see :func:`_await_peer_refresh` for why polling the refresh entry point
#: would POST the token endpoint once per slice.
PEER_REFRESH_POLL_S = 0.1

#: How long a nameable diagnosis of "why nothing was usable" may be reused.
#:
#: ``_unusable_credential`` cannot learn that answer cheaply: ``list_oauth_accesses``
#: is the store's ONE place that separates a dead grant from a transient miss, and it
#: refreshes EVERY stored OAuth row to do it. Paid once that is what makes a dead
#: grant visible to the user; paid per request it is a token-endpoint POST against a
#: credential already known to fail — measured (QA round 1, R31) SEVEN failed
#: refreshes for six refusals where the base paid one, and against a hanging endpoint
#: every page load waits out the timeout the previous one already hit. So the verdict
#: is remembered for ``DEFAULT_BLOCK_MS``: the window the cascade's own failed refresh
#: has the row out of rotation for (``block_credential``'s default, the same failure
#: and the same window), which is exactly the window in which re-asking can learn
#: nothing new. A row that CHANGED is diagnosed again whatever the clock says — see
#: :func:`_diagnosis_key` — so a re-login or a peer's landed refresh is never masked.
DIAGNOSIS_TTL_S = DEFAULT_BLOCK_MS / 1000

#: The upstream statuses the single-op path passes through to the renderer
#: unchanged; anything else becomes a 502, because an upstream answer this proxy
#: does not recognise leaves the caller with the same move as an unreachable
#: upstream. Named once so that a batch and a single read CANNOT classify the
#: same upstream answer differently — that split is what let an upstream 500
#: render as a page of "nothing liked" (review round 1, R-1).
UPSTREAM_PASSTHROUGH_STATUSES = frozenset({400, 401, 403, 404, 409, 422, 429})

#: The one upstream ANSWER that means "this account holds no relation with this
#: agent" rather than "the read failed": 404 for an agent the viewer cannot see
#: (or one delisted while the page was open) and its permanent-gone sibling 410.
#: ONLY these two degrade a batched id to "not liked, not favourited"; every other
#: non-200 is the batch's failure, because a failure that renders as an absence
#: is the silent wrong answer this op must never produce — a renamed upstream
#: path answers 404 for every id, and a UI reading that as "unliked" would show a
#: page of unliked cards forever, in both repositories, with no error raised.
UPSTREAM_ABSENT_STATUSES = frozenset({404, 410})

#: The shape above, compiled for the comma-joined list — matched with
#: ``fullmatch``, not ``match``, so a trailing newline cannot ride into a URL
#: path segment.
STATUS_ID = re.compile(IDENTIFIER)

#: The ONE sentence each refusal class of this transport carries, so the single-op
#: path and the batch cannot word the same class differently. A client keys on
#: ``code`` (the desktop transport reads ``detail.code`` and falls back to the
#: string form of ``detail``); this is what a human reads when the client holds no
#: copy of its own, and it is deliberately the batch's existing wording rather
#: than new prose invented for the single-op path.
REFUSAL_MESSAGE = "Radient could not complete this operation"

#: The sentence for "this account has no usable Radient sign-in" — the same words
#: the 409 has always carried, because the remedy for a missing sign-in and the
#: remedy for a grant Radient has declared dead are the SAME single action. The
#: two stay distinguishable by ``code`` and by status, not by prose.
NO_CREDENTIAL_MESSAGE = "Sign in to Radient to access your account"

#: The ``details.reason`` for "this ACCOUNT's sign-in is refused" — the IdP has
#: declared the grant dead, so a re-login is the only move. Named once because three
#: arms reach it (the store's own diagnosis, and this process's own refresh attempt on
#: either of its error types) and a client keys on the string.
REASON_GRANT_INVALID = "grant_invalid"

#: The ``details.reason`` for "a stored credential is not usable right now", whose
#: move is to retry. ONE reason for one client class, discovered at two points: the
#: store's diagnosis could not name a dead grant (the row is temporarily
#: unrefreshable), or this process attempted the refresh itself and it failed
#: transiently. The spelling is shared deliberately — a client that keys ``reason``
#: must not see the same remedy under two names, which is what the arm that answered
#: ``refresh_failed`` did for every token-endpoint 5xx the cascade caught one call up.
REASON_UNAVAILABLE = "credential_unavailable"

#: The ``details.reason`` for "a PEER holds the refresh lease and its token had not
#: landed when this request stopped waiting": nothing is known to be wrong with the
#: grant, the account is mid-refresh somewhere else. Its own spelling, because it is
#: the one class whose remedy is "ask again in a moment" rather than "sign in" or
#: "the upstream is unhappy" — minus the caller-visible difference from
#: :data:`REASON_UNAVAILABLE`, which is that a retry here is expected to succeed off
#: the peer's write rather than to wait out the store's backoff.
REASON_REFRESH_DID_NOT_LAND = "refresh_did_not_land"


def status_ids(body: RadientRequest) -> list[str]:
    """The agent ids one `agents.statuses` request names.

    Ids arrive as one comma-joined query value rather than as a JSON list,
    because the closed vocabulary's `query` is a flat string map and widening it
    to carry a typed list would widen every op's surface for one op's sake.

    De-duplicated and kept in the caller's order, so a status report is keyed by
    id and the caller never has to pair a positional list with its input. THE KEY
    IS THE STRING THE CALLER SENT: a chunk is accepted only in the canonical
    shape, so a value that would have to be trimmed to become an identifier is
    refused rather than answered under a key its sender never used. Empty chunks
    are dropped, so a trailing comma costs nothing.

    At most ``STATUS_BATCH_LIMIT`` ids, which a caller holding a bigger page
    (``agents.list`` allows ``per_page=100``) must ask about in chunks — read
    that constant for why the bound is not raised to match the list op's.

    Raises:
        ValueError: no ids, more than ``STATUS_BATCH_LIMIT``, or an id that is
            not the shape the rest of this module accepts as an identifier — see
            the call site in ``validate_request`` for why the last one is a
            refusal rather than a skip.
    """
    raw = str(body.query.get("agent_ids", ""))
    ids = list(dict.fromkeys(part for part in raw.split(",") if part))
    if not ids:
        raise ValueError("Choose at least one agent")
    if len(ids) > STATUS_BATCH_LIMIT:
        raise ValueError(f"Read at most {STATUS_BATCH_LIMIT} agent statuses at once")
    if any(STATUS_ID.fullmatch(agent_id) is None for agent_id in ids):
        raise ValueError("Invalid agent identifier")
    return ids


class RadientRequest(Input):
    operation: Literal[
        "account",
        "prices",
        "credits",
        "usage",
        "provision",
        "application.create",
        "agents.list",
        "agents.get",
        "agents.create",
        "agents.update",
        "agents.delete",
        "agents.like",
        "agents.unlike",
        "agents.liked",
        "agents.like_count",
        "agents.favourite",
        "agents.unfavourite",
        "agents.favourited",
        "agents.favourite_count",
        "agents.download_count",
        # The viewer's own like/favourite state for a whole page of agents.
        #
        # `agents.liked` and `agents.favourited` answer that question for one
        # agent each, and the desktop hub needs it for every card it paints —
        # which is what made a twelve-card page cost twenty-four requests before
        # it finished painting. This op is the same information for a bounded list
        # of ids in one call, so the renderer asks once. It is ADDITIVE: a UI
        # newer than its server gets a 404 for the unknown operation, which the
        # hub reads as "no viewer state to show" rather than as a failure.
        "agents.statuses",
        "comments.list",
        "comments.create",
        "comments.update",
        "comments.delete",
        "account.agents",
    ]
    request_id: RequestID | None = None
    tenant_id: Identifier | None = None
    account_id: Identifier | None = None
    agent_id: Identifier | None = None
    comment_id: Identifier | None = None
    query: dict[str, str | int] = Field(default_factory=dict)
    payload: dict[str, Any] = Field(default_factory=dict)
    confirmed: StrictBool = False

    @model_validator(mode="after")
    def validate_request(self):
        method, _ = endpoint(self)
        if method != "GET" and self.request_id is None:
            raise ValueError("Mutations require a request identifier")
        if method == "DELETE" and not self.confirmed:
            raise ValueError("Confirm this removal")
        query_fields = (
            {
                "page",
                "per_page",
                "categories",
                "tags",
                "account_id",
                "tenant_id",
                "name",
                "description",
                "sort",
                "order",
            }
            if self.operation in {"agents.list", "account.agents"}
            else (
                {"agent_ids"}
                if self.operation == "agents.statuses"
                else (
                    {"page", "per_page"}
                    if self.operation == "comments.list"
                    else (
                        {
                            "start_date",
                            "end_date",
                            "application_id",
                            "usage_type",
                            "provider",
                            "rollup",
                        }
                        if self.operation == "usage"
                        else set()
                    )
                )
            )
        )
        if self.query.keys() - query_fields or any(
            len(str(value)) > 1024 for value in self.query.values()
        ):
            raise ValueError("Unsupported query fields")
        for key in {"page", "per_page"} & self.query.keys():
            value = str(self.query[key])
            if not value.isdecimal() or not 1 <= int(value) <= (
                100 if key == "per_page" else 10000
            ):
                raise ValueError("Invalid pagination")
        if self.operation == "usage" and self.query.get("rollup") not in {
            "daily",
            "monthly",
            "annual",
        }:
            raise ValueError("Choose daily, monthly or annual usage")
        if self.operation == "agents.statuses":
            # Validated here rather than at the fan-out, because these ids become
            # upstream path segments: a malformed one — and a list past
            # `STATUS_BATCH_LIMIT`, which the caller has to chunk — must be refused
            # before a single request is built from it, and the upstream call
            # ledger is what proves it was.
            #
            # NOT so the caller can read WHICH rule failed: every desktop
            # validation failure is flattened to one sentence by the app-level
            # handler (`app.py`'s RequestValidationError arm), so a 422 from here
            # says "The request has invalid fields" and nothing more. That is why
            # the chunking rule is documented at `STATUS_BATCH_LIMIT`, where the
            # renderer's authors read this op, rather than promised in an error
            # body the caller never receives.
            status_ids(self)
        allowed: set[str] = set()
        if self.operation in {"agents.create", "agents.update"}:
            allowed = {
                "name",
                "version",
                "description",
                "model",
                "temperature",
                "top_p",
                "top_k",
                "max_tokens",
                "frequency_penalty",
                "presence_penalty",
                "seed",
                "hosting",
                "security_prompt",
                "current_working_directory",
                "stop",
                "tags",
                "categories",
            }
        elif self.operation in {"comments.create", "comments.update"}:
            allowed = {"text"}
        elif self.operation == "application.create":
            allowed = {"name", "description"}
        if self.payload.keys() - allowed:
            raise ValueError("Unsupported payload fields")
        required = (
            {"name", "version"}
            if self.operation == "agents.create"
            else (
                {"name"}
                if self.operation == "application.create"
                else {"text"} if self.operation in {"comments.create", "comments.update"} else set()
            )
        )
        if any(
            not isinstance(self.payload.get(key), str) or not self.payload[key].strip()
            for key in required
        ):
            raise ValueError("Required text fields are missing")
        if len(self.model_dump_json().encode()) > 200000:
            raise ValueError("The request exceeds the size limit")
        return self


def endpoint(body: RadientRequest) -> tuple[str, str]:
    op = body.operation
    if op == "agents.statuses":
        # The one operation whose real work is several upstream calls, so this
        # pair is NOMINAL: it exists because `validate_request` asks for the
        # method to decide whether a request id is required, and the route
        # dispatches this op before it builds anything from a path. The reads it
        # actually makes are built by `agent_statuses`.
        return "GET", "/agents/statuses"
    if op in {"account", "prices", "provision"}:
        return (
            ("POST", "/provision")
            if op == "provision"
            else ("GET", "/me" if op == "account" else "/prices")
        )
    if op in {"credits", "usage", "application.create"}:
        if not body.tenant_id:
            raise ValueError("Choose a tenant")
        tail = {
            "credits": "billing/credits",
            "usage": "usage/rollup",
            "application.create": "applications",
        }[op]
        return (
            "POST" if op == "application.create" else "GET",
            f"/tenants/{body.tenant_id}/{tail}",
        )
    if op == "account.agents":
        if not body.account_id:
            raise ValueError("Choose an account")
        return "GET", f"/accounts/{body.account_id}/agents"
    if op in {"agents.list", "agents.create"}:
        return ("GET" if op == "agents.list" else "POST"), "/agents"
    if not body.agent_id:
        raise ValueError("Choose an agent")
    path = f"/agents/{body.agent_id}"
    if op.startswith("comments."):
        path += "/comments"
        if op in {"comments.update", "comments.delete"}:
            if not body.comment_id:
                raise ValueError("Choose a comment")
            path += "/" + body.comment_id
        return {
            "comments.list": "GET",
            "comments.create": "POST",
            "comments.update": "PATCH",
            "comments.delete": "DELETE",
        }[op], path
    suffixes = {
        "get": ("GET", ""),
        "update": ("PATCH", ""),
        "delete": ("DELETE", ""),
        "like": ("POST", "/like"),
        "unlike": ("DELETE", "/like"),
        "liked": ("GET", "/like"),
        "like_count": ("GET", "/like/count"),
        "favourite": ("POST", "/favourite"),
        "unfavourite": ("DELETE", "/favourite"),
        "favourited": ("GET", "/favourite"),
        "favourite_count": ("GET", "/favourite/count"),
        "download_count": ("GET", "/download/count"),
    }
    method, tail = suffixes[op.removeprefix("agents.")]
    return method, path + tail


def base_url() -> str:
    provider = get_provider_definition("radient")
    assert provider is not None and provider.base_url
    return provider.base_url.rstrip("/")


def public_data(value: Any, secrets: list[str]) -> Any:
    if isinstance(value, dict):
        return {
            key: public_data(item, secrets)
            for key, item in value.items()
            if key.lower()
            not in {
                "access_token",
                "refresh_token",
                "id_token",
                "api_key",
                "token",
                "password",
                "secret",
                "authorization",
                "client_secret",
            }
        }
    if isinstance(value, list):
        return [public_data(item, secrets) for item in value]
    if isinstance(value, str):
        for secret in secrets:
            if secret:
                value = value.replace(secret, "[redacted]")
    return value


def _failure(status: int, code: str, message: str, **details: Any) -> HTTPException:
    """One failed read of the batch, in the shape a renderer can act on.

    The desktop plane's own refusals already answer ``{"code", "message"}``
    (the ``errors()`` ladder in :mod:`desktop_sessions`), and this op is the first
    here that knows WHICH id, WHICH relation and WHICH exception failed — so it
    carries those in ``details`` instead of discarding them into one prose
    sentence. ``code`` is the part a caller keys on, and it is what separates the
    classes whose remedies differ: a refused credential ("sign in, or wait") from
    an outage ("this page has no viewer state to show"). ``details`` holds no
    prose — anything a human reads lives in ``message``.
    """
    return HTTPException(status, {"code": code, "message": message, "details": details})


def _upstream_refusal(status: int, **details: Any) -> HTTPException:
    """The ONE reading of an upstream answer that is not a success.

    The status is passed through for the classes
    ``UPSTREAM_PASSTHROUGH_STATUSES`` names and is a 502 otherwise, so an
    upstream answer this proxy does not recognise leaves the caller with the
    same move as an unreachable upstream. On top of the status sits ``code``,
    which is what tells the renderer whether it is looking at a refusal of the
    ACCOUNT's credential (sign in to Radient again) or at an outage (retry). The
    status alone cannot: a 401 from Radient and a 401 from the desktop plane are
    both 401, and they have opposite remedies — the desktop plane refusing this
    app's own bearer is answered by restarting or re-pairing the app, which is
    the wrong thing to tell a user whose Radient account merely needs a fresh
    sign-in.

    There is exactly one of these because the batch and the single-op path must
    not be able to classify the same answer two ways; the split is what let an
    upstream 500 render as a page of "nothing liked" (review round 1, R-1).
    """
    if status in {401, 403, 429}:
        return _failure(status, "radient_credential_refused", REFUSAL_MESSAGE, **details)
    return _failure(
        status if status in UPSTREAM_PASSTHROUGH_STATUSES else 502,
        "radient_upstream_failed",
        REFUSAL_MESSAGE,
        **details,
    )


def _upstream_failure(status: int, agent_id: str, relation: str) -> HTTPException:
    """The batch's answer for one upstream read that failed, mapped once.

    The per-id detail rides along so the batch can say WHICH id, WHICH relation
    and WHICH upstream status failed while the class itself comes from
    :func:`_upstream_refusal`, the single reading both ops share.
    """
    return _upstream_refusal(status, upstream_status=status, agent_id=agent_id, relation=relation)


def _credential_refusal(reason: str, **details: Any) -> HTTPException:
    """The refusal for a diagnosis ``reason``, in the class that reason's remedy is.

    One place maps the reason vocabulary (:data:`REASON_GRANT_INVALID`,
    :data:`REASON_UNAVAILABLE`, :data:`REASON_REFRESH_DID_NOT_LAND`) onto status,
    ``code`` and the sentence a user reads, so no arm can answer the right
    ``reason`` under the wrong class — the failure mode the review found, where a
    token-endpoint 5xx reached two arms that spelled it differently.
    """
    if reason == REASON_GRANT_INVALID:
        return _failure(
            401, "radient_credential_refused", NO_CREDENTIAL_MESSAGE, reason=reason, **details
        )
    return _failure(502, "radient_upstream_failed", REFUSAL_MESSAGE, reason=reason, **details)


#: The last "why nothing was usable" verdict, keyed by the rows it was made about.
#:
#: Module state, and deliberately: it memoises the STORE's reading, is keyed by the
#: store's own database (see :func:`_diagnosis_key`) and expires on its own within
#: :data:`DIAGNOSIS_TTL_S`, so it caches no fact this process is not allowed to know
#: and cannot outlive the condition it describes. Purging is opportunistic on write.
_DIAGNOSIS: dict[tuple[str, tuple[tuple[int, int], ...]], tuple[str, float]] = {}


def _diagnosis_key(
    store: AuthStore, rows: list[StoredCredential]
) -> tuple[str, tuple[tuple[int, int], ...]]:
    """What a remembered verdict is ABOUT: one store's rows, as they stand now.

    ``updated_at`` is what the store writes on every change to a row, so a re-login
    or a refresh another process landed composes a different key and the verdict is
    computed again; a FAILED refresh writes nothing, which is what lets the verdict
    hold across the very failures it describes. The store's own database path is in
    the key because this memo outlives one request while two daemons can share this
    process: credential ids restart at 1 in every config dir.
    """
    return (str(store.db_path), tuple(sorted((row.id, row.updated_at) for row in rows)))


def _remembered_diagnosis(store: AuthStore, rows: list[StoredCredential]) -> str | None:
    """The verdict still in force for these rows, or ``None`` to diagnose again."""
    key = _diagnosis_key(store, rows)
    entry = _DIAGNOSIS.get(key)
    if entry is None:
        return None
    reason, expires = entry
    if time.monotonic() >= expires:
        _DIAGNOSIS.pop(key, None)
        return None
    return reason


def _remember_diagnosis(store: AuthStore, rows: list[StoredCredential], reason: str) -> None:
    """Keep ``reason`` for these rows for the store's own block window."""
    now = time.monotonic()
    # Purge first: an entry is dead within DIAGNOSIS_TTL_S and there is never a reason
    # to keep one, so a row that keeps changing cannot grow this dict without bound.
    for key, (_reason, expires) in list(_DIAGNOSIS.items()):
        if expires <= now:
            _DIAGNOSIS.pop(key, None)
    _DIAGNOSIS[_diagnosis_key(store, rows)] = (reason, now + DIAGNOSIS_TTL_S)


async def _unusable_credential(auth: DesktopAuth) -> HTTPException:
    """WHY the cascade had no usable Radient credential, in the client's classes.

    Reached only when ``get_oauth_access`` came back empty: either nothing is
    stored, or something IS stored and could not be used. The second case is the
    silent one this change exists for — a credential whose grant the IdP has
    declared dead is still "logged in" to every surface that reads rows, so the
    user is told nothing while every call fails.

    ``list_oauth_accesses`` is asked which kind it is, and deliberately so: it is
    the store's ONE place that separates a dead grant from a transient miss (it
    returns ``credential_invalid`` for the first and omits the row for the
    second), and the cascade cannot carry that back — ``_resolve`` catches both as
    ``AuthStoreError`` and rotates away, which is right for routing and useless
    for reporting. The cost is one refresh attempt per stored OAuth row, paid only
    on the path where the resolver already had nothing to serve, and it is what
    makes the dead-grant state visible instead of silent.

    PAID ONCE PER BLOCK WINDOW, NOT ONCE PER REQUEST. That attempt is a POST to a
    token endpoint the cascade has just spent and failed, so asking again on the
    next request learns nothing and pays again — measured (QA round 1, R31) seven
    failed refreshes for six refusals against the base's one, and, against an
    endpoint that HANGS rather than fails, one page load's worth of timeout per
    page load. So the verdict is remembered for the window the row is blocked for:
    see :data:`DIAGNOSIS_TTL_S` and :func:`_diagnosis_key` for why that is the right
    bound, and why a changed row is never masked by it.
    """
    stored = auth.store.list_credentials("radient")
    if not stored:
        return _failure(409, "radient_no_credential", NO_CREDENTIAL_MESSAGE)
    rows = [row for row in stored if row.credential_type == "oauth"]
    reason = _remembered_diagnosis(auth.store, rows)
    if reason is None:
        accesses = await auth.store.list_oauth_accesses("radient")
        reason = (
            REASON_GRANT_INVALID
            if any(access.credential_invalid for access in accesses)
            else REASON_UNAVAILABLE
        )
        _remember_diagnosis(auth.store, rows, reason)
    return _credential_refusal(reason)


def _is_due(auth: DesktopAuth, access: OAuthAccess) -> bool:
    """Whether the store's own rule says the bearer it just served is due a refresh.

    False for an api_key row (no expiry to be due) and for a row the store considers
    current. Asking the store rather than reading ``expires`` here is on purpose: the
    skew and the never-expires case live with the store, and a second spelling of
    them is what this route's change is removing.
    """
    if access.kind != "oauth" or not access.credential_id:
        return False
    row = auth.store.get_credential(access.credential_id)
    return row is not None and bool(auth.store._needs_refresh(dict(row.data)))


async def _await_peer_refresh(auth: DesktopAuth, access: OAuthAccess) -> str | None:
    """Wait BOUNDEDLY for a peer's in-flight refresh to land; ``None`` if it did not.

    Reached only with a bearer the store has just handed back still due AFTER an
    attempt of our own that did not raise, which cross-process means the refresh
    lease is held by another runtime: the store's peer path re-reads once, finds the
    row still due, and returns it rather than waiting for its peer to finish
    (``auth_store.py``, the ``if not force: return now_data`` branch). For a model
    request that is right — the stale bearer may still serve. For this route it was
    a measured 502 for the whole duration of a legitimately refreshing peer, where
    the base answered 200, so the wait happens here, where the caller's alternative
    is a refusal rather than a token.

    The wait is a READ of the row, not another refresh call. Polling the refresh
    entry point instead would POST the token endpoint once per slice the moment the
    lease freed: against an endpoint that has just failed, or for a refresh the peer
    already finished. So each slice looks for the peer's WRITE, and the token is only
    resolved through the cascade once the row it left behind is current — which is
    also what keeps the store the only thing that decides what a live bearer is.
    """
    if not access.credential_id:
        return None
    deadline = time.monotonic() + PEER_REFRESH_WAIT_S
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None
        await asyncio.sleep(min(PEER_REFRESH_POLL_S, remaining))
        row = auth.store.get_credential(access.credential_id)
        if row is None or auth.store._needs_refresh(dict(row.data)):
            continue
        live = await auth.store.get_oauth_access("radient")
        if live is not None and not _is_due(auth, live):
            return live.access_token


async def radient_bearer(auth: DesktopAuth, *, classify: bool = True, join: bool = True) -> str:
    """The bearer to spend on Radient — or the classified refusal that says not to.

    THE ACCOUNT'S CREDENTIAL IS NOT THE APP'S. Everything else on this transport
    authenticates with the desktop plane's own bearer, whose refusal means this
    app can no longer talk to its server; this is the operator's Radient
    sign-in, whose refusal means they have to sign in to Radient again. Two
    remedy classes, two credentials, and only one of them is safe to spend
    blind — so this function is what keeps the plane's 401 (restart or re-pair
    the app) and the account's 401 (sign in to Radient) tellable apart, by putting
    a ``code`` on the second one.

    WHY A ROUTE ASKS THE STORE A SECOND QUESTION. ``get_oauth_access`` answers
    with the cascade's pick, and the cascade's last resort is deliberately to
    hand back a token whose refresh did not happen (``AuthStore._ensure_oauth_fresh``
    returns the stored row when a peer holds the refresh lease and the re-read is
    still due). That is right for a model request, which the stale bearer may
    still serve; it is wrong here, because the proxy's only possible report is
    Radient's own 401 — a failure that names no remedy, costs a round trip it
    cannot win, and (measured 2026-09-19) left the operator's account row
    silently showing a placeholder name while ``/v1/auth/status`` reported
    ``refresh_due``. So this function asks for the one fact the cascade does not
    carry — is the bearer I was just handed one the store already considers due a
    refresh — and answers the refusal when the store cannot produce a live one.

    IT USES THE STORE'S OWN MACHINERY, NOT A SECOND READING OF IT. The refresh
    goes through ``AuthStore.ensure_oauth_fresh_or_raise``, the public per-row
    entry point that keeps the store's own types on the way out: single-flight
    through the same lease every other caller uses, so two processes cannot POST
    one rotating refresh token and earn a real ``invalid_grant`` against a live
    grant (see the race guard in ``_ensure_oauth_fresh``). JOINED, THOUGH, ONLY
    INSIDE THIS PROCESS: the lock that single-flight is built on is an ``asyncio``
    one, so a PEER runtime holding the cross-process lease is not joined by that
    call — the store's peer path re-reads once and returns the stored row — and this
    function waits for the peer itself, bounded, in :func:`_await_peer_refresh`,
    before it refuses.

    WHO TAKES THE ACCOUNT OUT OF ROTATION. Not this function: the row is blocked one
    call up, where ``_resolve`` catches a failed refresh as ``AuthStoreError`` and
    blocks it (``auth_store.py``, tier 3), which a dead grant reaches too because
    ``CredentialInvalidError`` is a subclass. So a failed refresh DOES cost the
    account its place for the store's block window, and that is a routing decision
    this route inherits rather than one a read is making. What the per-row call buys
    is that the refusal is about THIS account instead of about the pool, and that it
    never rotates a session onto a sibling. The freshness QUESTION has no public
    predicate — the two public entry points wrap the refresh, not the rule — so
    ``_needs_refresh`` is reached for, on purpose: re-deriving ``expires`` against
    ``now`` here would be exactly the second spelling of the store's rule that this
    change removes (the skew and the never-expires case live with the store). When
    the store grows that predicate, this is its first caller.

    THE CLASSES ARE REMEDIES, NOT VENDORS. A dead grant
    (:class:`CredentialInvalidError`, an ``invalid_grant`` the IdP has declared
    terminal) leaves exactly one move, a fresh sign-in, so it answers as
    ``radient_credential_refused`` with ``details.reason = "grant_invalid"`` — the
    same class an upstream refusal of the credential earns, because it is the same
    remedy. Everything whose move is "try again" answers as
    ``radient_upstream_failed``, the class already reserved for "this page has no
    viewer state to show", and ``details.reason`` names which condition it was:
    ``credential_unavailable`` (a stored credential cannot be produced right now —
    the store's diagnosis could not name a dead grant, or this process's own refresh
    attempt failed transiently; ONE reason for that one class however it was
    discovered, which is why the arm that used to answer ``refresh_failed`` for a
    token-endpoint 5xx no longer has a spelling of its own) or
    ``refresh_did_not_land`` (a PEER holds the lease and its token did not land
    inside the bounded wait). No fourth code is invented for those distinctions: the
    class vocabulary is the remedies, and which one it was rides in ``details``.

    ``classify=False`` is for :func:`radient_bearer_optional`, which discards the
    reason: a diagnosis costs the store a refresh attempt per stored OAuth row (and a
    log line), and the credentialless op is served bare regardless of what it would
    have said.

    ``join=False`` is that op's other half, and it is separate on purpose. Waiting for
    a peer's refresh is worth a second to a caller whose alternative is a refusal the
    user has to act on; it is a second of a public price read that needed no
    credential in the first place. So the wait is taken only by the callers that
    refuse — the batch and the single-op path — and a credentialless op still falls
    back to running bare the moment the store has nothing current.

    Raises:
        HTTPException: 409 ``radient_no_credential`` when nothing is stored, 401
            ``radient_credential_refused`` when the stored grant is dead, or 502
            ``radient_upstream_failed`` when no live bearer could be produced
            right now.
    """

    access = await auth.store.get_oauth_access("radient")
    if access is None:
        failure = (
            _failure(409, "radient_no_credential", NO_CREDENTIAL_MESSAGE)
            if not classify
            else await _unusable_credential(auth)
        )
        raise failure
    if not _is_due(auth, access):
        return access.access_token
    try:
        await auth.store.ensure_oauth_fresh_or_raise(access.credential_id)
    except CredentialInvalidError as error:
        # Terminal: the refresh token is dead and only a re-login revives the
        # account, so the refusal names that remedy instead of relaying a bare
        # 401 the user cannot act on.
        raise _credential_refusal(REASON_GRANT_INVALID, cause=type(error).__name__) from error
    except AuthStoreError as error:
        # Transient, and reachable only when THIS process held the lease and its
        # own POST failed — the cascade eats the same failure one call up and
        # answers it as ``credential_unavailable``. It gets that same reason rather
        # than a spelling of its own: one remedy must not arrive under two names.
        raise _credential_refusal(REASON_UNAVAILABLE, cause=type(error).__name__) from error
    refreshed = await auth.store.get_oauth_access("radient")
    if refreshed is None:
        # Still nothing selectable: a verdict written by an earlier failure (the
        # block the cascade sets) is what hides the row, and that is "not right
        # now", not "you are signed out".
        raise _credential_refusal(REASON_UNAVAILABLE)
    if not _is_due(auth, refreshed):
        return refreshed.access_token
    if not join:
        raise _credential_refusal(REASON_REFRESH_DID_NOT_LAND)
    # The refresh call came back without a live bearer, which cross-process means a
    # peer holds the lease and its attempt has not landed. A peer that is
    # SUCCEEDING is the case the base answered 200 off the stale bearer, so give it
    # the bounded wait and refuse only once its token really did not land.
    landed = await _await_peer_refresh(auth, access)
    if landed is None:
        raise _credential_refusal(REASON_REFRESH_DID_NOT_LAND)
    return landed


async def radient_bearer_optional(auth: DesktopAuth) -> str:
    """`radient_bearer` for the ONE op the census marks credentialless (``prices``).

    That op's contract is that it runs WITHOUT a Radient sign-in — it is served
    with no Authorization header when nothing is stored. So a credential that is
    stored but cannot be freshened must not become a failure this op never
    needed: it falls back to running bare, exactly as it does when signed out,
    rather than spending the stale bearer the store still holds. Every other op
    refuses instead, because for them a bearer nobody has asked Radient about yet
    is precisely the unclassifiable 401 this change removes.
    """
    try:
        return await radient_bearer(auth, classify=False, join=False)
    except HTTPException:
        return ""


async def _settle(reads: list[asyncio.Task[None]]) -> None:
    """Wait for the batch's reads, stopping at the first one that settles the batch.

    Deliberately NOT ``asyncio.gather(*reads, return_exceptions=True)``, which is
    what this op first shipped: that waits for every sibling even after one read
    has already decided the batch's answer, so a credential Radient refused cost
    all 2N reads — twenty-four calls at an API that had just said no, on the
    page-refresh path, during exactly the window where that API is unhealthy.
    ``FIRST_EXCEPTION`` returns the moment a read raises, and the reads still
    queued behind the concurrency limiter are cancelled rather than started, so a
    refusal stops the burst in the wave that noticed it. Reads already in flight
    (at most ``STATUS_BATCH_CONCURRENCY``) cannot be recalled, which is the floor
    for any parallel fan-out.

    The cancellation is explicit rather than inherited: ``asyncio.wait_for``
    cancels THIS coroutine when the overall deadline expires, and a cancelled
    awaiter does not cancel the tasks it created — without it the reads would
    outlive the client they are reading through.
    """
    try:
        done, pending = await asyncio.wait(reads, return_when=asyncio.FIRST_EXCEPTION)
    except BaseException:
        for read in reads:
            read.cancel()
        await asyncio.gather(*reads, return_exceptions=True)
        raise
    for read in pending:
        read.cancel()
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)
    # EVERY completed read's result is retrieved before one of them is raised:
    # several reads fail in the same wave, and a task whose exception nobody
    # picked up reports "Task exception was never retrieved" at collection time —
    # noise that reads like a leak in exactly the logs this op gets debugged from.
    # The failure raised is still the caller's FIRST id rather than whichever
    # read happened to answer first.
    failure: BaseException | None = None
    for read in reads:
        if read not in done or read.cancelled():
            continue
        error = read.exception()
        if error is not None and failure is None:
            failure = error
    if failure is not None:
        raise failure


async def agent_statuses(body: RadientRequest, auth: DesktopAuth) -> dict[str, Any]:
    """The viewer's like and favourite state for a bounded list of agents.

    THE ONE OP THAT MAKES SEVERAL UPSTREAM CALLS, and the reason is the shape of
    what it is asked: `GET /agents/{id}/like` and `/agents/{id}/favourite` answer
    for ONE agent each, and the desktop hub needs the answer for every card it
    paints. Served one id at a time from the renderer, a twelve-card page costs
    twenty-four round trips through this proxy and two more waves of them (after
    first paint, and again on every window focus). Served here, it costs one,
    with the fan-out inside a process that already holds the credential and a
    connection pool to Radient.

    What keeps it from being a general-purpose proxy: the vocabulary is still
    closed, the ids are bounded by ``STATUS_BATCH_LIMIT``, the only upstream
    paths reachable are the two status reads below, and nothing a caller sends
    is interpolated into a URL without matching ``STATUS_ID``.

    The result is keyed by agent id, and every id the caller named is present.

    The fan-out is bounded TWICE: one read by ``httpx``'s own timeout, and the
    batch as a whole by ``STATUS_BATCH_TIMEOUT`` — so the worst case here is the
    same ceiling every other op on this transport carries, rather than the
    ``ceil(2N / 6)`` multiples a per-read timeout alone would allow.

    Raises:
        HTTPException: 409 ``radient_no_credential`` when no Radient credential is
            stored; the credential classes ``radient_bearer`` raises before any
            upstream read (401 ``radient_credential_refused`` for a grant the
            store knows is dead, 502 ``radient_upstream_failed`` when no live
            bearer could be produced); the upstream
            status (401/403/429) when Radient refuses the whole batch; the mapped
            upstream status for any other non-200 answer, which is a FAILURE — the
            only upstream answers read as an absent per-agent relation are 404 and
            410 (``UPSTREAM_ABSENT_STATUSES``); and 502 when the upstream could
            not be reached, sent more than ``MAX_UPSTREAM_BYTES``, or did not
            finish inside the batch's budget. Every one of these carries
            ``{"code", "message", "details"}``.
    """
    ids = status_ids(body)
    token = await radient_bearer(auth)
    base = base_url()
    headers = {"Authorization": "Bearer " + token}
    statuses: dict[str, dict[str, bool]] = {
        agent_id: {"liked": False, "favourited": False} for agent_id in ids
    }
    limiter = asyncio.Semaphore(STATUS_BATCH_CONCURRENCY)

    async def read(client: httpx.AsyncClient, agent_id: str, suffix: str, key: str) -> None:
        async with limiter:
            try:
                # Streamed rather than fetched whole, so one upstream body cannot
                # exceed MAX_UPSTREAM_BYTES in memory — the same ceiling the
                # single-op path applies, applied the same way.
                async with client.stream(
                    "GET", f"{base}/agents/{agent_id}/{suffix}", headers=headers
                ) as response:
                    if response.is_redirect:
                        # Same classification as the single-op path: a redirect says
                        # the base address is wrong rather than that this agent has
                        # no relation, so it is the batch's failure and not an
                        # absent answer.
                        raise _failure(
                            502,
                            "radient_upstream_failed",
                            "Radient returned an unexpected redirect",
                            agent_id=agent_id,
                            relation=suffix,
                            upstream_status=response.status_code,
                        )
                    if response.status_code in UPSTREAM_ABSENT_STATUSES:
                        # The ONE absence. An agent the account cannot see (or one
                        # delisted while the page was open) leaves that id at "not
                        # liked, not favourited": the state a client that knows
                        # nothing renders, and the one state whose toggle is still
                        # correct, because the like and favourite endpoints answer
                        # an already-present relation with `already_liked` /
                        # `already_favourited` rather than with a conflict.
                        #
                        # EVERY OTHER non-200 raises, including the 5xx and the
                        # 400/409/422 classes: those are the batch failing to
                        # answer, and rendering them as an absence is what would
                        # silently break the page for as long as the upstream
                        # stayed broken.
                        return
                    if response.status_code != 200:
                        raise _upstream_failure(response.status_code, agent_id, suffix)
                    # 200 with NO document is this endpoint's "no relation": the
                    # handler answers empty content when the account holds no
                    # like, so the presence of a body is the flag rather than any
                    # field inside it.
                    content = bytearray()
                    async for chunk in response.aiter_bytes():
                        content.extend(chunk)
                        if len(content) > MAX_UPSTREAM_BYTES:
                            raise _failure(
                                502,
                                "radient_upstream_too_large",
                                "Radient returned too much data",
                                agent_id=agent_id,
                                relation=suffix,
                                limit_bytes=MAX_UPSTREAM_BYTES,
                            )
            except httpx.TimeoutException as error:
                raise _failure(
                    502,
                    "radient_upstream_timeout",
                    "Radient did not answer in time",
                    agent_id=agent_id,
                    relation=suffix,
                    cause=type(error).__name__,
                ) from error
            except httpx.HTTPError as error:
                # The cause stays on the chain rather than being discarded (`from
                # error`, not `from None`): it is the only account of WHY this read
                # failed — a refused connection, a truncated body — and the
                # renderer's copy is kept free of it by the structured body, not
                # by throwing the reason away.
                raise _failure(
                    502,
                    "radient_upstream_unreachable",
                    "Radient is unavailable or returned an invalid response",
                    agent_id=agent_id,
                    relation=suffix,
                    cause=type(error).__name__,
                ) from error
        statuses[agent_id][key] = bool(content.strip())

    async with httpx.AsyncClient(timeout=STATUS_BATCH_TIMEOUT, follow_redirects=False) as client:
        reads = [
            asyncio.create_task(read(client, agent_id, suffix, key))
            for agent_id in ids
            for suffix, key in (("like", "liked"), ("favourite", "favourited"))
        ]
        try:
            await asyncio.wait_for(_settle(reads), STATUS_BATCH_TIMEOUT)
        except asyncio.TimeoutError as error:
            # The whole fan-out's budget, not one read's. 502, the same status a
            # per-read httpx timeout produces one line up, so the renderer's
            # "upstream did not answer" path is one path.
            raise _failure(
                502,
                "radient_upstream_timeout",
                "Radient did not answer in time",
                timeout_seconds=STATUS_BATCH_TIMEOUT,
            ) from error
    return {
        "data": {
            "msg": "Agent statuses read",
            "result": {"statuses": statuses},
        }
    }


@router.post("/v1/desktop/radient", response_model=CRUDResponse[Result])
async def radient(
    body: RadientRequest, request: Request, auth: DesktopAuth = Depends(get_desktop_auth)
):
    if body.operation == "agents.statuses":
        # Dispatched before `endpoint` builds a path, because this op's real work
        # is the bounded fan-out above rather than one upstream call. It is a
        # read, so it takes no receipt.
        async with errors(request):
            return reply(await agent_statuses(body, auth))

    method, path = endpoint(body)

    async def execute():
        # The ONE place this transport decides whether it has a bearer worth
        # spending: an unrefreshed token is refused here rather than relayed as an
        # upstream 401 that names no remedy (see `radient_bearer`). AuthStore
        # performs the only refresh. Never retry a mutation on a 401: upstream may
        # have accepted it before the connection failed. `prices` is the census's
        # credentialless op and keeps its no-sign-in form.
        token = await (
            radient_bearer_optional(auth) if body.operation == "prices" else radient_bearer(auth)
        )
        try:
            async with httpx.AsyncClient(timeout=30, follow_redirects=False) as client:
                async with client.stream(
                    method,
                    base_url() + path,
                    params=body.query,
                    json=body.payload if method in {"POST", "PATCH"} else None,
                    headers={"Authorization": "Bearer " + token} if token else {},
                ) as response:
                    if response.is_redirect:
                        raise _failure(
                            502,
                            "radient_upstream_failed",
                            "Radient returned an unexpected redirect",
                            upstream_status=response.status_code,
                        )
                    if response.status_code >= 400:
                        raise _upstream_refusal(
                            response.status_code, upstream_status=response.status_code
                        )
                    content = bytearray()
                    async for chunk in response.aiter_bytes():
                        content.extend(chunk)
                        if len(content) > MAX_UPSTREAM_BYTES:
                            raise _failure(
                                502,
                                "radient_upstream_too_large",
                                "Radient returned too much data",
                                limit_bytes=MAX_UPSTREAM_BYTES,
                            )
                    import json

                    value = json.loads(content) if content else {}
        except httpx.TimeoutException as error:
            # The batch's own spelling for "the upstream did not answer", so the
            # renderer has ONE path for it rather than a per-op guess: a per-read
            # timeout there produces this same code and status. `ConnectTimeout` — a
            # connection that never opened, which reads more like "unreachable" —
            # stays in this class on purpose (review n3): it is a `TimeoutException`
            # subclass, the batch's read checks the same class in the same order, and
            # the caller's move is identical either way, so splitting the spelling on
            # one path would buy nothing and cost the parity the two ops share.
            raise _failure(
                502,
                "radient_upstream_timeout",
                "Radient did not answer in time",
                cause=type(error).__name__,
            ) from error
        except (httpx.HTTPError, ValueError) as error:
            # `ValueError` is a body that would not parse, and it answers in the
            # same class as an unreachable upstream because the caller's move is
            # the same one. The cause is kept on the chain (unlike the `from None`
            # this replaces): the client's copy never carries it, but the log does,
            # and a transport failure with no recorded cause is what made this
            # route's failures unattributable.
            raise _failure(
                502,
                "radient_upstream_unreachable",
                "Radient is unavailable or returned an invalid response",
                cause=type(error).__name__,
            ) from error
        stored_key = ""
        if body.operation in {"provision", "application.create"}:
            result = value.get("result", {})
            stored_key = result.get("api_key", "")
            if not isinstance(stored_key, str) or not stored_key:
                raise _failure(
                    502,
                    "radient_upstream_failed",
                    "Radient did not return an application credential",
                )
            auth.store.upsert_credential(
                "radient", {"type": "api_key", "source": "login", "key": stored_key}
            )
        return {"data": public_data(value, [token, stored_key])}

    async with errors(request):
        result = (
            await execute()
            if method == "GET"
            else await receipts(request).run(
                "radient:" + str(body.request_id), body.model_dump(), execute
            )
        )
        return reply(result)
