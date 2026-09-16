"""Closed Radient operations using AuthStore, never renderer tokens or refreshes.

The route census comes from the desktop's existing Radient clients. Google
Workspace consent is deliberately NOT here: effective MCP grants own that flow.
"""

from __future__ import annotations

import asyncio
import re
from typing import Annotated, Any, Literal

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import Field, StrictBool, model_validator

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


def _upstream_failure(status: int, agent_id: str, relation: str) -> HTTPException:
    """The batch's answer for one upstream read that failed, mapped like the single-op path.

    The status is passed through for the classes that path passes through
    (``UPSTREAM_PASSTHROUGH_STATUSES``) and is a 502 otherwise, so the two ops
    cannot disagree about the same upstream answer; ``code`` is what tells the
    renderer whether it is looking at a refusal or at an outage.
    """
    details = {"upstream_status": status, "agent_id": agent_id, "relation": relation}
    if status in {401, 403, 429}:
        return _failure(
            status,
            "radient_credential_refused",
            "Radient could not complete this operation",
            **details,
        )
    return _failure(
        status if status in UPSTREAM_PASSTHROUGH_STATUSES else 502,
        "radient_upstream_failed",
        "Radient could not complete this operation",
        **details,
    )


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
        HTTPException: 409 when no Radient credential is stored; the upstream
            status (401/403/429) when Radient refuses the whole batch; the mapped
            upstream status for any other non-200 answer, which is a FAILURE — the
            only upstream answers read as an absent per-agent relation are 404 and
            410 (``UPSTREAM_ABSENT_STATUSES``); and 502 when the upstream could
            not be reached, sent more than ``MAX_UPSTREAM_BYTES``, or did not
            finish inside the batch's budget. Every one of these carries
            ``{"code", "message", "details"}``.
    """
    ids = status_ids(body)
    access = await auth.store.get_oauth_access("radient")
    if access is None:
        raise _failure(409, "radient_no_credential", "Sign in to Radient to access your account")
    token = access.access_token
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
        async with errors():
            return reply(await agent_statuses(body, auth))

    method, path = endpoint(body)

    async def execute():
        access = await auth.store.get_oauth_access("radient")
        if access is None and body.operation != "prices":
            raise HTTPException(409, "Sign in to Radient to access your account")
        # AuthStore performs the only refresh. Never retry a mutation on a 401:
        # upstream may have accepted it before the connection failed.
        token = access.access_token if access else ""
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
                        raise HTTPException(502, "Radient returned an unexpected redirect")
                    if response.status_code >= 400:
                        raise HTTPException(
                            (
                                response.status_code
                                if response.status_code in UPSTREAM_PASSTHROUGH_STATUSES
                                else 502
                            ),
                            "Radient could not complete this operation",
                        )
                    content = bytearray()
                    async for chunk in response.aiter_bytes():
                        content.extend(chunk)
                        if len(content) > MAX_UPSTREAM_BYTES:
                            raise HTTPException(502, "Radient returned too much data")
                    import json

                    value = json.loads(content) if content else {}
        except (httpx.HTTPError, ValueError):
            raise HTTPException(
                502, "Radient is unavailable or returned an invalid response"
            ) from None
        stored_key = ""
        if body.operation in {"provision", "application.create"}:
            result = value.get("result", {})
            stored_key = result.get("api_key", "")
            if not isinstance(stored_key, str) or not stored_key:
                raise HTTPException(502, "Radient did not return an application credential")
            auth.store.upsert_credential(
                "radient", {"type": "api_key", "source": "login", "key": stored_key}
            )
        return {"data": public_data(value, [token, stored_key])}

    async with errors():
        result = (
            await execute()
            if method == "GET"
            else await receipts(request).run(
                "radient:" + str(body.request_id), body.model_dump(), execute
            )
        )
        return reply(result)
