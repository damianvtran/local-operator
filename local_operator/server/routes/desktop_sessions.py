"""Additive authenticated session API; legacy per-turn chat stays unchanged."""

from __future__ import annotations

import asyncio
import base64
import json
import pathlib
import sqlite3
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Path, Query, Request
from fastapi.responses import Response, StreamingResponse
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    field_validator,
    model_validator,
)
from starlette.background import BackgroundTask

from local_operator.harness.types import ModelSpec
from local_operator.media import SUPPORTED_IMAGE_MIME_TYPES
from local_operator.server.desktop import require_desktop
from local_operator.server.models.desktop_sessions import (
    AnswerReceipt,
    AttentionState,
    ChildTranscriptPage,
    CommandReceipt,
    CreatedSession,
    DraftPreviewPayload,
    HistoryPage,
    MessageAdmission,
    NotificationClaim,
    SessionList,
    SessionSearch,
    SessionSnapshot,
    WarmReceipt,
    WatchReceipt,
)
from local_operator.server.models.schemas import CRUDResponse
from local_operator.server.retire import RETIRING_STATE_ATTR, DaemonRetiring
from local_operator.server.utils.desktop_commands import OWNER_COMMANDS, native_action
from local_operator.server.utils.desktop_receipts import (
    DesktopReceipts,
    ReceiptConflict,
)
from local_operator.server.utils.desktop_sessions import (
    CHILD_PAGE_LIMIT,
    DesktopSessionBridge,
    DesktopSessions,
    SubagentChildUnavailable,
    resolve_working_directory,
)
from local_operator.session.attention import SupersededCompletionToken
from local_operator.session.cold_model import synthesise_cold_state
from local_operator.session.frontend_state import (
    FrontendSync,
    SlashResult,
    sync_wire_payload,
)
from local_operator.session.session_search import search_store
from local_operator.slash_commands import slash_command_for

router = APIRouter(tags=["Desktop sessions"], dependencies=[Depends(require_desktop)])
RequestID = Annotated[
    str, Field(pattern=r"^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$")
]
#: The attachment store names files ``<digest>.bin``, so the digest reaches a
#: filesystem path directly. Constraining the shape HERE rather than in the
#: handler is what makes traversal unreachable by construction: FastAPI rejects
#: a non-matching path before the handler runs, and a later edit inside the
#: handler cannot route around a declaration. 32 hex characters is
#: ``attachments._DIGEST_CHARS``.
AttachmentDigest = Annotated[str, Path(pattern=r"^[a-f0-9]{32}$")]
#: What a stored attachment may claim to be on the wire. The store records the
#: mime its CALLER supplied — ``transcript._externalize_attachments`` copies
#: ``block["mime_type"]`` verbatim with no allowlist — and the sidecar carrying
#: it is not digest-verified, so neither the value's shape nor its meaning is
#: guaranteed at this boundary. Two failures follow from trusting it, and both
#: were measured on this route: a mime containing CRLF makes h11 reject the
#: header and the client gets NO response at all, contradicting the docstring's
#: "404, never 500"; and ``text/html``/``image/svg+xml`` round-trip out of here
#: as active content served from an authenticated local port. Today no live
#: ingress stores a non-image mime, but that is a property of callers upstream
#: that nothing HERE enforces, and this is the boundary that pays for it
#: changing. ``routes/static.py`` already establishes the allowlist convention
#: for image bytes over HTTP; the narrower ``media`` set is used because it is
#: exactly what the ingress can produce (``sniff_image`` returns these four)
#: and it excludes ``image/svg+xml``, which is script-bearing.
ATTACHMENT_FALLBACK_MIME = "application/octet-stream"


class Input(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Seen(Input):
    # A durable completion UUID is the only admission: timestamps or a caller's
    # runtime epoch could accidentally acknowledge a later, unseen outcome.
    completion_token: RequestID


class Notified(Input):
    # Same admission as `Seen`, and for the same reason: a durable completion
    # UUID is the only identity that survives a runtime epoch, so nothing a
    # caller can invent (a timestamp, its own epoch) may enter the watermark.
    # The two routes are otherwise unrelated — this one claims the right to
    # notify and NEVER acknowledges a read.
    completion_token: RequestID


class SessionTarget(Input):
    kind: Literal["agent", "team"]
    name: str = Field(min_length=1, max_length=128)


class DraftModel(Input):
    """The model a NEW conversation should be born on, plus its reasoning level.

    Deliberately the SAME shape the canonical frontend state publishes for a
    conversation's model (``selected_model``: ``provider``, ``model_id``,
    ``reasoning_effort``), so the client hands back the row its own picker is
    already showing rather than translating one vocabulary into another — a
    second shape is how the chip and the model that answers come to disagree.

    ``reasoning_effort=None`` means "no level chosen". It is NOT the same as "this
    model has no ladder": the ladder check is what refuses an unexpressible level
    below, and a model with an empty ladder refuses every level but ``None``. It is
    also NOT "the model's own default": the conversation is born on the machine's
    configured ``model_effort`` (clamped into this model's ladder), and only a
    config that expresses no opinion falls back to the model's seeded rung. The
    MARKER records the ``None`` — the choice — while the pane and the launch report
    that resolved level; see ``session.cold_model.resolve_birth_effort``.
    """

    provider: str = Field(min_length=1, max_length=64)
    model_id: str = Field(min_length=1, max_length=256)
    reasoning_effort: str | None = Field(default=None, max_length=32)


class CreateSession(Input):
    request_id: RequestID
    cwd: str = Field(min_length=1, max_length=4096)
    target: SessionTarget | None = None
    #: Omitted or null ⇒ today's behaviour, byte for byte: the session is born on
    #: the configured default. This is the ONLY optional admission of the two
    #: routes, and it is what lets an older client keep posting the old body.
    model: DraftModel | None = None


class DraftPreview(Input):
    """A new-conversation pane's readings, for a session that does not exist.

    Same shape as :class:`CreateSession` on purpose: the pane is asking the
    question it will ask for real on the first send, and a body that differed
    would let the two answers diverge.

    ``request_id`` is carried for envelope parity and deliberately NOT journalled
    as a durable receipt (see the route). A preview has no side effect to make
    at-most-once, receipts are never pruned, and a renderer may re-issue this on
    every re-render — writing a row per pane would make simply opening one cost
    disk, which is exactly what this op exists to avoid.
    """

    request_id: RequestID
    cwd: str = Field(min_length=1, max_length=4096)
    target: SessionTarget | None = None
    #: The same optional birth selection ``create`` accepts, resolved by the same
    #: authority — the pane is asking the question it will ask for real on the
    #: first send, so a body that may differ here would be a body that diverges.
    model: DraftModel | None = None


def _draft_model_spec(model: DraftModel) -> ModelSpec:
    """The spec the first turn will actually run on, or a 422 saying why not.

    REFUSED HERE, AT THE BOUNDARY, and never by silently degrading: the client
    rendered a choice the user made, and answering with a different model than
    the one it named is the failure this whole feature exists to remove. The
    three refusals are the three ways a pick can fail to be servable:

    * an unknown provider (``get_provider_definition``, which resolves the
      registry's legacy aliases, so an alias is accepted exactly where the
      engine accepts it and nowhere else);
    * a model id the provider's catalogue does not serve — the same authority
      the model picker paints from (:func:`discovery.offered_model_ids`), and no
      refusal at all for a catalogue that cannot be enumerated offline (an
      aggregator on a cold cache, a local endpoint): "we have not looked" is
      not "it does not exist", and refusing on it would kill working picks;
    * a ``reasoning_effort`` this model's ladder does not offer, including any
      level at all on a model with no ladder — the ladder is read off the SAME
      resolver the owner constructs its spec with (``build_model_spec``), so the
      level accepted here is the level the first turn will send.

    **A pick that named no level comes back carrying NONE**, never the model's
    seeded default rung. ``build_model_spec`` seeds that rung ("the level this
    model would use if nobody said"), and a seed is not a choice: letting it
    through would store a level the user never picked, pin it in the marker and
    have the first turn take it over the machine's configured ``model_effort``
    (review round 1, R1). ``null`` therefore stays distinguishable from "the
    level happened to equal the seed" all the way to the launch, where the
    owner's own resolution supplies the configured level, clamped by the ladder.

    Runs OFF the event loop by its callers: it reads the catalogue cache and, for
    a model the registry does not describe, resolves metadata (memoised, and
    disk-cached in the common case the picker just filled it).
    """
    from local_operator.model.configure import build_model_spec
    from local_operator.model.discovery import offered_model_ids
    from local_operator.providers.registry import get_provider_definition

    provider = model.provider.strip()
    model_id = model.model_id.strip()
    if get_provider_definition(provider) is None:
        raise HTTPException(
            422,
            {
                "code": "provider_unknown",
                "message": f"'{provider}' is not a known provider.",
            },
        )
    served = offered_model_ids(provider)
    if served is not None and model_id not in served:
        raise HTTPException(
            422,
            {
                "code": "model_unknown",
                "message": f"'{model_id}' is not a model {provider} serves.",
            },
        )
    effort = (model.reasoning_effort or "").strip().lower()
    try:
        spec = build_model_spec(provider, model_id)
    except Exception as error:  # noqa: BLE001 — a spec we cannot build is a 422, not a 500
        raise HTTPException(
            422,
            {"code": "model_unavailable", "message": f"'{model_id}' could not be resolved."},
        ) from error
    if effort:
        if not spec.reasoning_efforts:
            raise HTTPException(
                422,
                {
                    "code": "effort_unsupported",
                    "message": f"{model_id} has no reasoning-effort levels",
                },
            )
        if effort not in spec.reasoning_efforts:
            raise HTTPException(
                422,
                {
                    "code": "effort_unsupported",
                    "message": (
                        f"{model_id} accepts {', '.join(spec.reasoning_efforts)} "
                        f"\u2014 not '{effort}'"
                    ),
                },
            )
        spec = spec.model_copy(update={"reasoning_effort": effort})
    elif spec.reasoning_effort is not None:
        # The seed, not a choice — and the two must not become indistinguishable
        # in the marker, which is the ONLY consumer of this cleared value
        # (``create`` persists it). This is NOT what the plane resolves against:
        # ``session.cold_model.resolve_birth_effort`` reads the SEED, and its third
        # case is defined on it, so a caller that hands it this cleared spec
        # answers "no level" where the launch answers the seed. Round 2's R6 is
        # exactly that mistake, made by the preview; the honest shape of the two
        # halves is ASYMMETRIC — the marker stores the CHOICE, the reading resolves
        # the RUNNING level — and the earlier comment here claimed they were the
        # same thing.
        spec = spec.model_copy(update={"reasoning_effort": None})
    return spec


def _preview_birth_model(root: pathlib.Path, model: DraftModel) -> ModelSpec:
    """The picked pair as the model describes it, at the level the turn will RUN at.

    Two values come out of the body's pick and they are NOT the same value:

    * what was CHOSEN, which is what ``create`` stores in the marker (``null`` for
      "this model, no level") — ``_draft_model_spec``'s answer, and it is used here
      only for its refusals;
    * what the first turn will RUN at, which is the reading this pane may publish.

    So the spec built here is ``build_model_spec``'s own result — it still carries
    the model's SEEDED rung, which is what ``resolve_birth_effort``'s third case is
    defined on. Handing it the seed-CLEARED spec instead made the preview report
    "no level" whenever the machine configured no ``model_effort``, while the cold
    frame and the first turn reported the seed (review round 2, R6).
    """
    from local_operator.model.configure import build_model_spec
    from local_operator.session.cold_model import resolve_birth_effort

    chosen = _draft_model_spec(model)
    spec = build_model_spec(chosen.provider, chosen.model_id)
    return spec.model_copy(
        update={"reasoning_effort": resolve_birth_effort(spec, chosen.reasoning_effort, root)}
    )


class Image(Input):
    data_b64: str = Field(max_length=1_000_000)
    mime_type: Literal["image/png", "image/jpeg", "image/gif", "image/webp"]

    @field_validator("data_b64")
    @classmethod
    def validate_data(cls, value: str) -> str:
        if not value or not base64.b64decode(value, validate=True):
            raise ValueError("An image must contain base64 data")
        return value


class Prompt(Input):
    request_id: RequestID
    text: str = Field(max_length=200_000)
    images: list[Image] = Field(default_factory=list, max_length=8)
    mode: Literal["prompt", "steer"] = "prompt"

    @model_validator(mode="after")
    def nonempty(self):
        if not self.text.strip() and not self.images:
            raise ValueError("Enter a message or attach an image")
        # Slash controls must never accidentally become paid model chat. The
        # caller resolves them through commands, including the native UI forms.
        if len(self.model_dump_json().encode()) > 900_000:
            raise ValueError("Message exceeds the canonical control-frame limit")
        if self.text.lstrip().startswith("/"):
            raise ValueError("Use the command endpoint for slash commands")
        return self


class Command(Input):
    request_id: RequestID
    command: str = Field(pattern=r"^/?[A-Za-z]+$", max_length=64)
    args: str = Field(default="", max_length=200_000)
    images: list[Image] = Field(default_factory=list, max_length=8)

    @model_validator(mode="after")
    def wire_budget(self):
        if len(self.model_dump_json().encode()) > 900_000:
            raise ValueError("Command exceeds the canonical control-frame limit")
        return self


class Answer(Input):
    epoch: str = Field(min_length=1, max_length=128)
    request_id: str = Field(min_length=1, max_length=128)
    value: str | None = Field(default=None, max_length=32768)
    approved: StrictBool | None = None
    question_index: int | None = Field(default=None, ge=0, strict=True)

    @model_validator(mode="after")
    def one_answer(self):
        if (self.value is None) == (self.approved is None):
            raise ValueError("Supply either value or approved")
        if self.value is not None and self.question_index is None:
            raise ValueError("An ask answer requires question_index")
        if self.approved is not None and self.question_index is not None:
            raise ValueError("An approval cannot carry question_index")
        return self


class Watch(Input):
    subscription_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    visible: StrictBool
    can_notify: StrictBool


class Warm(Input):
    """No fields, and closed to any that arrive.

    The session is already named by the path and a warm carries no intent
    beyond "start one" — there is nothing a caller could usefully say here.
    Declared as a model rather than omitted so ``extra="forbid"`` still
    applies: a client that invents an option gets a 422 naming it, instead of
    having it silently ignored and believing it took effect.
    """


def host(request: Request) -> DesktopSessions:
    pool = getattr(request.app.state, "desktop_sessions", None)
    if pool is None:
        pool = DesktopSessions(
            request.app.state.config_manager.config_dir,
            # The DAEMON's retirement, asked of app state rather than of the pool:
            # the pool is built lazily by the first request that needs it, and a
            # pool built after the announcement must refuse for the same reason
            # (and with the same 503) as one built before it.
            retiring=lambda: bool(getattr(request.app.state, RETIRING_STATE_ATTR, False)),
        )
        request.app.state.desktop_sessions = pool
    return pool


def receipts(request: Request) -> DesktopReceipts:
    value = getattr(request.app.state, "desktop_receipts", None)
    if value is None:
        value = DesktopReceipts(request.app.state.config_manager.config_dir)
        request.app.state.desktop_receipts = value
    return value


def reply(result: Any) -> CRUDResponse[Any]:
    return CRUDResponse(status=200, message="Desktop session result.", result=result)


@asynccontextmanager
async def errors() -> AsyncIterator[None]:
    try:
        yield
    except DaemonRetiring as error:
        # The daemon has announced its retirement and refuses to start work it
        # would not finish. 503, not 500: the process is alive and deliberately
        # not admitting, so the client's move is to rediscover the successor
        # through the record (design §7) — a message and a code it can key on,
        # never a traceback.
        raise HTTPException(503, {"code": error.code, "message": str(error)}) from None
    except SubagentChildUnavailable as error:
        # The child read route's containment refusal (design § 9.1). Not folded
        # into the generic 404 below because the code is part of the contract:
        # the reader distinguishes nothing from it, but it is retryable, and a
        # client that cannot see WHY would have to guess whether to re-probe.
        raise HTTPException(404, {"code": error.code, "message": str(error)}) from None
    except KeyError:
        raise HTTPException(
            404, "Requested session, profile, team or subscription not found"
        ) from None
    except (ReceiptConflict, ValueError) as error:
        from local_operator.session.errors import (
            AttachmentUnavailable,
            ProfileRegistryUnavailable,
        )

        if isinstance(error, (AttachmentUnavailable, ProfileRegistryUnavailable)):
            raise HTTPException(409, {"code": error.code, "message": str(error)}) from None
        if isinstance(error, SupersededCompletionToken):
            # Stale, not broken: the caller's token is real but no longer current,
            # and the remedy is to re-read the conversation's attention state and
            # acknowledge the token it names. The machine code is what lets the
            # renderer take that path quietly instead of backing off as if the
            # store had refused (the `code` field of its control error).
            raise HTTPException(409, {"code": error.code, "message": str(error)}) from None
        raise HTTPException(409, str(error)) from None
    except sqlite3.Error:
        # Contention on the shared receipt store is transient and retryable, so
        # it gets a vetted sentence rather than a bare 500 carrying SQLite's own
        # wording. The text is NOT echoed for the same reason the ConnectionError
        # ladder below refuses to echo: a store error can name file paths.
        raise HTTPException(
            503, "Read state is busy right now. It will catch up on its own."
        ) from None
    except ConnectionError as error:
        # A cold session that cannot start a runtime reports WHY -- but only when
        # the reason arrives as an `ActionableConnectionError`, whose TYPE is
        # what certifies the message as one of the vetted configuration
        # sentences (`launch._ACTIONABLE_STARTUP_REASONS`).
        #
        # Vettedness is deliberately NOT inferred from the text. Echoing every
        # ConnectionError leaked `owner socket unreachable: [Errno 61] Connect
        # call failed ('127.0.0.1', 54321)` and `owner moved to another
        # conversation (<session id>)` straight to the renderer: `attach_client`
        # raises bare ConnectionErrors carrying socket errors, internal control
        # ports and other sessions' ids. Those keep the generic sentence.
        detail = str(error).strip() if getattr(error, "actionable", False) else ""
        raise HTTPException(
            503,
            detail or "Session owner is unavailable. Reconnect and reconcile before retrying.",
        ) from None
    except (RuntimeError, asyncio.TimeoutError):
        raise HTTPException(
            503, "Session owner is unavailable. Reconnect and reconcile before retrying."
        ) from None


@router.get("/v1/desktop/sessions", response_model=CRUDResponse[SessionList])
async def list_sessions(request: Request, limit: int = Query(default=100, ge=1, le=500)):
    # Wrapped like its neighbours: the list gained a receipt-store read, and an
    # unmapped failure there answered the app's primary navigation surface with
    # a bare 500. The decoration is already omitted per row inside `list()`;
    # this ladder covers anything else the pool can raise.
    async with errors():
        rows = await host(request).list(limit + 1)
        return reply({"sessions": rows[:limit], "truncated": len(rows) > limit, "limit": limit})


@router.get("/v1/desktop/sessions/search", response_model=CRUDResponse[SessionSearch])
async def search_sessions(
    request: Request,
    q: str = Query(default="", max_length=256),
    limit: int = Query(default=100, ge=1, le=500),
):
    """Past conversations matching ``q`` by name, id, or what was SAID in them.

    The same search the CLI's ``/resume`` picker runs, through the one
    implementation the picker and the phone daemon share
    (``session_search.search_store``): name and id as exact case-insensitive
    substrings over the haystack the row is RENDERED with, plus the cached body
    digest index for an exact conversation match, plus a bounded soft tier
    (prefix, word-order, edit distance <= 2 on words of 4+ characters) when the
    query is not already precisely answered. Results come back best-first with
    the tier and the reason attached.

    **Declared BEFORE ``/v1/desktop/sessions/{session_id}``** and that order is
    load-bearing: FastAPI matches routes in declaration order, so a parent route
    declared first would swallow this path and the client would get the
    snapshot of a session literally named "search" (a 404, in practice) instead
    of an answer.

    The scan and the index build run off the event loop — they read every
    session directory's head and one cache file — while ``q`` is bounded at 256
    characters because the query is only ever a user's typing, and an unbounded
    one would be projected into every digest comparison.
    """
    async with errors():
        matches = await asyncio.to_thread(search_store, host(request).root, q, limit=limit)
        return reply(
            {
                "sessions": [
                    {
                        "id": match.row.id,
                        "name": match.row.name,
                        "mtime": match.row.mtime,
                        "forked": match.row.forked,
                        "rank": match.rank,
                        "body_match": match.body_match,
                    }
                    for match in matches
                ],
                "query": q,
                "limit": limit,
            }
        )


@router.post("/v1/desktop/sessions", response_model=CRUDResponse[CreatedSession])
async def create_session(body: CreateSession, request: Request):
    """Create a new conversation, optionally born on a chosen model and effort.

    The body's admissions run BEFORE the receipt is claimed, in the same order
    ``preview`` applies them (cwd, target, model), for two reasons. A refusal has
    to leave the store exactly as it found it — the claim is a durable write, and
    a claimed-then-refused request would otherwise answer its own retry with
    "outcome indeterminate" instead of the refusal. And the two routes are
    documented as answering the same refusals, which is only true if they also
    agree on WHICH refusal a body that is bad in two ways gets (review round 1,
    R4).

    All three are the admissions ``DesktopSessions.create`` itself applies, so a
    body that passes here passes there — and the pool runs them again rather than
    trusting this pre-flight, because that is the callable's own contract and a
    second caller must not be able to reach it unvalidated. Two of the three are
    pure reads (a ``stat`` for the directory; the catalogue and the metadata cache
    for the model, which writes nothing at all), and they run BEFORE the target's,
    which builds the registries ``create`` would build anyway — on a fresh root that
    materialises ``<config>/agents``, so it is the one admission with a filesystem
    effect (review round 2, R8; the preview route documents the same carve-out).

    A REPLAYED request runs none of them: a retry whose directory has since vanished
    must answer what the first attempt recorded, not turn a success into a refusal
    (review round 2, R7).

    No ``model`` ⇒ the body, the marker and the launch are byte-for-byte today's.
    """

    async def create():
        pool = host(request)
        target = body.target.model_dump() if body.target else None
        session_id = await pool.create(
            body.cwd,
            target=target,
            # The NORMALISED triple, not the raw body: the marker is what a later
            # engage re-resolves, so it stores the pair the validation just
            # proved servable (provider alias resolved, level lowercased).
            model=(
                {
                    "provider": spec.provider,
                    "model_id": spec.model_id,
                    "reasoning_effort": spec.reasoning_effort,
                }
                if spec is not None
                else None
            ),
        )
        return {"session_id": session_id, "binding": await pool.binding(session_id)}

    async with errors():
        # REFUSED BEFORE ANYTHING IS CLAIMED OR ADMITTED — before the receipt is
        # claimed and before the draft's own admissions (the working directory, the
        # model spec, the target registry) run: a refused request must leave no
        # pending receipt behind, or the client's retry against the SUCCESSOR would
        # meet the indeterminate 409 the receipts layer reserves for a crashed
        # attempt (``desktop_receipts``). ``DesktopSessions.create`` re-asks the
        # same question as its first statement, so a caller that reaches the
        # adapter another way gets the same refusal.
        host(request).assert_admitting()
        pool = host(request)
        key = "create:" + body.request_id
        spec: ModelSpec | None = None
        # A recorded key short-circuits the admissions below: see the docstring.
        # ``recorded`` is a read — it neither claims the key nor creates the store.
        if not await asyncio.to_thread(receipts(request).recorded, key):
            # The SAME admissions ``create`` applies, in the SAME order ``preview``
            # applies them: the working directory, then the model (which writes
            # nothing), then the target (whose registry build materialises
            # ``<config>/agents``). See ``DesktopSessions.create`` /
            # ``resolve_working_directory``.
            await asyncio.to_thread(resolve_working_directory, body.cwd)
            if body.model is not None:
                # Off the loop: the catalogue it reads is a disk document, and for
                # an unshipped model the metadata resolver may consult the provider.
                spec = await asyncio.to_thread(_draft_model_spec, body.model)
            if body.target is not None:
                target_row = body.target.model_dump()
                from local_operator.agents import AgentRegistry
                from local_operator.server.utils.desktop_profiles import validate_target
                from local_operator.teams import TeamRegistry

                await asyncio.to_thread(
                    validate_target,
                    AgentRegistry(pool.root),
                    TeamRegistry(pool.root),
                    target_row["kind"],
                    target_row["name"],
                )
        return reply(await receipts(request).run(key, body.model_dump(), create))


@router.post("/v1/desktop/sessions/preview", response_model=CRUDResponse[DraftPreviewPayload])
async def preview_session(body: DraftPreview, request: Request):
    """The readings a new-conversation pane may show, for a session that is not.

    **Declared BEFORE ``/v1/desktop/sessions/{session_id}``** for the same reason
    ``search`` is: FastAPI matches in declaration order, and although no POST
    route exists under that path today, the client would otherwise get whatever
    one a later edit adds, for a session literally named "preview".

    A deliberately session-LESS op, and every absence below is the point. There
    is no session record, nothing under ``sessions/`` (no directory, no marker, no
    runtime lease), nothing handed to the canonical store, and no receipt row — a
    pane that is only being OPENED must cost nothing durable. The alternative
    (create the record at pane open, then cold-GET it) is honest but leaves a
    visible empty row in the sidebar for every abandoned pane: ``create`` writes a
    directory plus a marker, and a marker-only directory is listed.

    ONE filesystem effect it does share with ``create``, named here rather than
    glossed as "no directory": a ``target`` is validated against
    ``AgentRegistry``, whose constructor materialises ``<config>/agents``
    (``agents.py``; ``TeamRegistry`` deliberately does not mkdir). That is the
    object the design mandates reusing — a second, non-mkdir'ing registry would be
    a second profile-resolution path, which is the defect class this PR exists to
    remove. The ``cwd`` check below runs FIRST, so a bad working directory is
    refused before any registry is built.

    The state comes from the SAME synthesis a real cold open uses
    (``session.cold_model``), so the draft's model chip cannot disagree with the
    model the first send actually gets — the UI swaps this payload for the first
    cold frame at ``finishDraft``, where a flicker is visible.

    The account-metadata step is SKIPPED, unlike a cold session: a draft has no
    context reading to divide, and a synthetic stickiness key must not move a
    real account's stickiness. So the window this pane publishes is the MODEL's
    own (``build_model_spec``'s), carried with
    ``context_metadata_resolved: False`` — it is NOT a placeholder, and it is NOT
    a promise about the first turn: a cold open may apply plan-scoped account
    metadata and a window that metadata scopes, which is exactly the read this op
    must not perform. The LADDER and the LEVEL do agree with the first turn by
    construction (see ``session.cold_model.resolve_birth_effort``); the effective
    window cannot, and nothing here claims it does.

    ``target`` is validated exactly as ``create`` validates it, so an
    unresolvable profile fails here rather than becoming a session that cannot
    start — the pane renders no strip instead of a reading for a session that
    cannot exist.
    """

    async def preview():
        pool = host(request)
        # The SAME admissions ``create`` applies, in the SAME order, so one body gets
        # one answer from either route (and 409, not a 200 describing a session that
        # could never be created). See ``DesktopSessions.create`` /
        # ``resolve_working_directory``.
        await asyncio.to_thread(resolve_working_directory, body.cwd)
        # The MODEL before the TARGET, as in ``create``: the model admission writes
        # nothing at all, while building the registries the target needs
        # materialises ``<config>/agents`` (review round 2, R8).
        birth_model = (
            await asyncio.to_thread(_preview_birth_model, pool.root, body.model)
            if body.model is not None
            else None
        )
        if body.target is not None:
            target = body.target.model_dump()
            from local_operator.agents import AgentRegistry
            from local_operator.server.utils.desktop_profiles import validate_target
            from local_operator.teams import TeamRegistry

            await asyncio.to_thread(
                validate_target,
                AgentRegistry(pool.root),
                TeamRegistry(pool.root),
                target["kind"],
                target["name"],
            )
        state = await synthesise_cold_state(
            config_dir=pool.root,
            session_id="",
            cwd=body.cwd,
            # The chosen selection, when there is one, is synthesised exactly as
            # the first cold frame of the session it describes will be — same
            # resolver, same metadata — so the pane's identity, its effort LADDER
            # and the LEVEL the first turn runs at are the ones the first turn
            # gets rather than the configured default's. The WINDOW is the one
            # reading that cannot agree and is not claimed to: the account-metadata
            # step above is skipped, so this pane publishes the MODEL's window with
            # ``context_metadata_resolved: False``, while a cold open may apply a
            # plan-scoped account window a draft must not read. With NO selection
            # the same synthesis answers the CONFIGURED pair, and
            # ``session.cold_model`` resolves that through the model's own metadata
            # too, so the ladder and the level are there for an unpicked draft as
            # well (review round 2's effort-reading defect).
            # Still session-less and side-effect free: this is an INPUT, and the
            # state below writes nothing (see the docstring above).
            birth_model=birth_model,
        )
        sync = FrontendSync(
            epoch=state.epoch,
            sequence=state.sequence,
            snapshot=state,
            live_cursor=state.history_cursor,
        )
        return {"frontend": sync_wire_payload(sync)}

    async with errors():
        return reply(await preview())


@router.get("/v1/desktop/sessions/{session_id}", response_model=CRUDResponse[SessionSnapshot])
async def snapshot(session_id: str, request: Request):
    async with errors(), host(request).session(session_id) as bridge:
        return reply(await bridge.snapshot())


@router.get("/v1/desktop/sessions/{session_id}/history", response_model=CRUDResponse[HistoryPage])
async def history(
    session_id: str,
    request: Request,
    before_id: str | None = Query(default=None, max_length=128),
    limit: int = Query(default=100, ge=1, le=500),
):
    async with errors(), host(request).session(session_id) as bridge:
        return reply(await bridge.history(before_id=before_id, limit=limit))


@router.get(
    "/v1/desktop/sessions/{session_id}/children/{child_id}/transcript",
    response_model=CRUDResponse[ChildTranscriptPage],
)
async def child_transcript(
    session_id: str,
    child_id: str,
    request: Request,
    before_id: str | None = Query(default=None, max_length=128),
    limit: int = Query(default=100, ge=1, le=CHILD_PAGE_LIMIT),
):
    """One page of a subagent's own transcript, read through its parent.

    The sidebar's child reader, and the ONLY door to a child's conversation:
    the renderer sends ids and never a ``session_dir``, so the backend (see
    ``_contained_child_dir``) is what proves the child belongs to the named
    parent, is a subagent rather than the user's own conversation or a fork,
    and lives under ``sessions/``. Refusals are 404 ``child_not_found``.

    The envelope is ``/history``'s, verbatim — ``entries`` are the child's raw
    transcript rows — plus the derived ``state``. ``pending`` and ``gone`` are
    answers, not errors: neither is a status the caller can act on, and both
    must be distinguishable from a refusal or the panel would report a missing
    child as "not yours".

    Read-only in the strongest sense: no bridge, no runtime, no message
    admission — a paused conversation answers exactly like a running one.
    """
    async with errors():
        return reply(
            await host(request).child_transcript(
                session_id, child_id, before_id=before_id, limit=limit
            )
        )


@router.get(
    "/v1/desktop/sessions/{session_id}/children/{child_id}/attachments/{digest}",
    # Same declaration as the parent's route, and for the same reason: the
    # response is raw bytes with the stored image's own type, which is not the
    # CRUD envelope its neighbours return.
    response_class=Response,
)
async def child_attachment(
    session_id: str, child_id: str, digest: AttachmentDigest, request: Request
):
    """Raw bytes of an attachment referenced by a CHILD transcript's rows.

    The parent's route cannot serve these: it takes the session whose transcript
    holds the reference, and a child's rows reference attachments in the same
    content-addressed store. Everything else is deliberately identical — the
    digest is the traversal gate (declared in the path, so FastAPI refuses a
    non-matching shape before the handler runs), the stored mime is allowlisted
    rather than trusted, ``nosniff`` rides the response, and a missing
    attachment is an ordinary 404.

    The route's scope is its parent's: containment proves the caller may read
    THIS child, and the store is shared across conversations by design, so the
    digest is not partitioned per child. The bearer already authorises the whole
    desktop surface; what this route must not become is a way to reach a child
    that is not the named session's, which ``_contained_child_dir`` refuses.
    """
    async with errors():
        data, mime_type = await host(request).child_attachment(session_id, child_id, digest)
    return Response(
        content=data,
        media_type=(
            mime_type if mime_type in SUPPORTED_IMAGE_MIME_TYPES else ATTACHMENT_FALLBACK_MIME
        ),
        headers={"X-Content-Type-Options": "nosniff"},
    )


@router.get(
    "/v1/desktop/sessions/{session_id}/attachments/{digest}",
    # The image's own mime type is the response type, so the published contract
    # must not claim the CRUD JSON envelope its neighbours return. Declaring the
    # class is what keeps the schema honest; without it FastAPI documents
    # `application/json` for a route that never sends any.
    response_class=Response,
)
async def attachment(session_id: str, digest: AttachmentDigest, request: Request):
    """Raw bytes of one content-addressed attachment referenced by a history row.

    ``/history`` serves durable rows verbatim, and a durable image block is
    ``{"attachment": <digest>, "mime_type": ...}`` with the payload stripped
    (``transcript._externalize_attachments``). Without this route a frontend can
    see that an image was in the conversation and can never render it after a
    reload — the live event stream carries the base64, the durable transcript
    does not.

    Three deliberate choices:

    - **Keyed by digest, not by (entry, index).** The history row hands the
      caller the digest directly, so nothing has to re-fold history per image,
      and identical screenshots across a whole conversation collapse to one
      request and one cache entry.
    - **The digest pattern is the traversal gate.** ``_DIGEST_CHARS`` is 32 hex
      characters and the store turns a digest straight into a filename, so the
      shape is validated in the path declaration rather than inside the handler
      where a later edit could route around it. FastAPI answers a non-matching
      path with 422 before any disk access.
    - **No HTTP caching, deliberately.** The digest IS the sha256 of the
      content, so an ``immutable`` response would be correct about the BYTES —
      and it is still the wrong header here. ``managed_desktop_boundary``
      applies ``Cache-Control: no-store`` to everything under ``/v1/desktop/``
      because these responses are bearer-gated session data, and a route that
      set ``public, max-age=31536000`` would either be silently overridden (it
      was: measured ``no-store`` on the wire) or, if the middleware were
      carved out for it, would invite a shared cache to retain one user's
      screenshots. The mobile daemon's equivalent route can afford
      ``immutable`` because it is a different process behind different auth;
      copying the header without the surrounding argument would not be reuse.
      Dedup belongs to the client, which already holds a digest-keyed cache
      for exactly this reason — and the digest being content-addressed is what
      makes that cache safe.

    A missing attachment is 404, never 500: the store's contract is that an
    unresolvable reference is ordinary (interrupted write, hand-pruned store)
    and the reader degrades to a placeholder. A corrupt-but-parseable sidecar
    is the same class of ordinary, which is why the mime is allowlisted rather
    than trusted (see :data:`ATTACHMENT_FALLBACK_MIME`).

    Two scopes this route does NOT have, stated because the URL shape implies
    otherwise:

    - ``session_id`` is an EXISTENCE check, not a binding. It proves *a* user
      conversation by that name is on this machine; it does not prove this
      attachment belongs to it. The store is content-addressed and shared
      across conversations by design, so any valid user session id resolves
      any digest in it. That is not an escalation — the bearer already
      authorises the whole desktop surface, ``/history`` included — but a
      future reader must not mistake the path segment for authorization.
    - The served ``Content-Type`` is not guaranteed to equal the ``mime_type``
      on the history row. The store dedups by content digest and the FIRST
      sidecar wins, so identical bytes stored once as ``image/png`` and later
      as ``image/gif`` keep the original sidecar while the newer row reports
      ``image/gif``. Harmless for real images (the bytes decide what renders),
      but the two values are not a matched pair.
    """
    async with errors():
        data, mime_type = await host(request).attachment(session_id, digest)
    return Response(
        content=data,
        media_type=(
            mime_type if mime_type in SUPPORTED_IMAGE_MIME_TYPES else ATTACHMENT_FALLBACK_MIME
        ),
        # Defence in depth on the one response here that can carry a type the
        # caller did not choose: with the allowlist above, an unexpected mime
        # is served as opaque bytes, and nosniff stops a browser from
        # re-deciding that for itself. `tunnels/gateway.py` sets the same
        # header on this repo's other byte-serving surface.
        headers={"X-Content-Type-Options": "nosniff"},
    )


@router.post(
    "/v1/desktop/sessions/{session_id}/messages", response_model=CRUDResponse[MessageAdmission]
)
async def prompt(session_id: str, body: Prompt, request: Request):
    """Admit ONE user turn, or refuse because this process is leaving.

    The refusal is the pool's door (``DesktopSessions.session``), which this
    route enters before it claims anything: it is the FIRST thing the handler
    does, before the receipt is claimed, and it covers ``admit_prompt``'s
    ``_ensure_bound`` (``session/attached.py``) — the one path in the desktop
    plane that can also START a session runtime, which is what ``warm``'s own
    gate exists to prevent (review round 1, MAJOR-2 measured this route
    answering 200 on a latched daemon while ``/warm`` answered 503 against the
    same process).

    The gate is at the DOOR rather than in this handler because this handler is
    one of fourteen that obtain a bridge, not the admission path itself (review
    round 2, MAJOR-1): see ``DesktopSessions.session`` for why the
    route-by-route question was the defect.

    NOT refused while the daemon is merely ANNOUNCED, which is why the gate is
    on the latch and not on the record: an announced daemon is still the only
    place its client can work (``server/retire.py``).
    """
    async with errors(), host(request).session(session_id) as bridge:

        async def admit():
            assert bridge.remote is not None
            detail, duplicate = await bridge.remote.admit_prompt(
                body.text,
                command_id=body.request_id,
                images=[image.model_dump() for image in body.images],
                steer=body.mode == "steer",
            )
            # Admission can bind a cold viewer while an event subscription is
            # already open. Apply only its still-live lease, never resurrect one.
            await bridge.refresh_watch()
            return {
                "status": "admitted",
                "command_id": body.request_id,
                "duplicate": duplicate,
                "detail": detail,
            }

        return reply(
            await receipts(request).run(
                session_id + ":" + body.request_id,
                body.model_dump(),
                admit,
                retry_safe=True,
            )
        )


@router.post(
    "/v1/desktop/sessions/{session_id}/commands", response_model=CRUDResponse[CommandReceipt]
)
async def command(session_id: str, body: Command, request: Request):
    """Run ONE slash command, or refuse because this process is leaving.

    Refused by the pool's door (``DesktopSessions.session``) for the same reason
    as ``prompt`` and at the same seam, and here it matters twice:
    ``bridge.remote.bind_runtime()`` below is an explicit ``_ensure_bound``, so a
    latched daemon would start a runtime for a command alone. The door runs
    before the receipt is claimed, so a refused command leaves no receipt row for
    the client's retry against the successor to trip over.
    """
    spec = slash_command_for("/" + body.command.removeprefix("/"))
    if spec is None or not spec.desktop_destination:
        raise HTTPException(422, "Unknown command")
    if spec.name == "credential" and body.args:
        raise HTTPException(
            422, "Enter credentials in the masked credential form, not command text"
        )
    if spec.name == "mcp" and body.args.strip():
        from local_operator.mcp.config import SERVER_NAME_RE
        from local_operator.session.frontend_state import MCP_SUBCOMMANDS

        parts = body.args.split()
        if (
            len(parts) > 2
            or parts[0] not in MCP_SUBCOMMANDS
            or (len(parts) == 2 and not SERVER_NAME_RE.fullmatch(parts[1]))
        ):
            raise HTTPException(
                422, "Use the MCP setup form for configuration and secret references"
            )
    if spec.name in {"login", "logout"} and body.args.strip():
        from local_operator.providers.registry import get_provider_definition

        if get_provider_definition(body.args.strip()) is None:
            raise HTTPException(422, "Choose a provider in the authentication panel")
    async with errors(), host(request).session(session_id) as bridge:

        async def execute():
            if (
                (spec.name == "team" and (body.args == "chart" or body.args.startswith("chart ")))
                or (
                    spec.name == "approvals"
                    and (body.args == "default" or body.args.startswith("default "))
                )
                or spec.name not in OWNER_COMMANDS
                or (
                    not body.args
                    and spec.name
                    in {"rename", "model", "effort", "fast", "approvals", "team", "agent", "loop"}
                )
            ):
                return {"command": spec.name, "result": native_action(spec, session_id, body.args)}
            assert bridge.remote is not None
            await bridge.remote.bind_runtime()
            await bridge.refresh_watch()
            outcome = await bridge.remote.route_shared_slash(
                spec.name,
                body.args,
                images=await decode_images(body.images),
            )
            if outcome is None or outcome.get("kind") == "noop":
                return {"command": spec.name, "result": native_action(spec, session_id, body.args)}
            outcome = SlashResult.model_validate(outcome)
            result = outcome.model_dump(mode="json")
            if outcome.kind == "error" and outcome.data.get("code") in {
                "loop_invalid",
                "loop_busy",
            }:
                raise HTTPException(
                    422 if outcome.data["code"] == "loop_invalid" else 409, outcome.text
                )
            consumed = outcome.data.get("request", "")
            attached = outcome.data.get("type") in {"team_attached", "agent_attached"}
            if attached and (consumed or body.images):
                # The runtime returns attachment metadata, not a started turn.
                # Match its typed discriminator rather than blindly submitting
                # any string a listing/picker happens to call a request.
                detail, duplicate = await bridge.remote.admit_prompt(
                    str(consumed),
                    command_id=body.request_id,
                    images=[image.model_dump() for image in body.images],
                )
                result["admission"] = {
                    "status": "admitted",
                    "detail": detail,
                    "duplicate": duplicate,
                }
            return {"command": spec.name, "result": result}

        return reply(
            await receipts(request).run(
                session_id + ":" + body.request_id, body.model_dump(), execute
            )
        )


async def decode_images(images: list[Image]):
    """Wire images to bounded ImageContent blocks, off the event loop.

    The shared helper resizes and re-encodes each image, which is CPU-bound;
    this runs inside a request handler, so it takes the threaded form.
    """
    from local_operator.session.runtime.server import image_blocks_in_thread

    return await image_blocks_in_thread([image.model_dump() for image in images])


@router.post(
    "/v1/desktop/sessions/{session_id}/answers", response_model=CRUDResponse[AnswerReceipt]
)
async def answer(session_id: str, body: Answer, request: Request):
    """Answer the pending gate, or refuse because this process is leaving.

    Refused by the pool's door (``DesktopSessions.session``) like the two routes
    above, and the reason is NOT that this path can start a runtime — it cannot:
    ``answer_gate`` needs a connected client and raises otherwise, so a cold
    daemon cannot be warmed into a spawn from here. The reason is that a latched
    daemon is a process whose socket is about to close, and an answer delivered
    through it is a delivery nobody can confirm; the client's correct move is the
    same one every other refusal asks for (rediscover the successor through the
    record) rather than the 409 its own "no longer pending" check would produce,
    which reads like the question expired.

    BEFORE the epoch comparison, deliberately — and the door runs before this
    handler's first statement: a stale-epoch answer on a latched daemon must not
    get a refusal that suggests retrying against this process.
    """
    async with errors(), host(request).session(session_id) as bridge:
        assert bridge.remote is not None
        if body.epoch != bridge.remote.frontend_state.epoch:
            raise HTTPException(409, "This answer belongs to an earlier session owner")
        try:
            detail = await bridge.remote.answer_gate(
                body.request_id,
                value=body.value,
                approved=body.approved,
                question_index=body.question_index,
            )
        except RuntimeError:
            raise HTTPException(409, "This question or approval is no longer pending") from None
        return reply({"detail": detail})


@router.post("/v1/desktop/sessions/{session_id}/seen", response_model=CRUDResponse[AttentionState])
async def seen(session_id: str, body: Seen, request: Request):
    async with errors():
        return reply(await host(request).acknowledge_attention(session_id, body.completion_token))


@router.post(
    "/v1/desktop/sessions/{session_id}/notified", response_model=CRUDResponse[NotificationClaim]
)
async def notified(session_id: str, body: Notified, request: Request):
    """Claim the right to raise ONE banner for one completion.

    Called by the desktop app immediately before it constructs the OS
    notification, and only then: claim-then-deliver means the claimant has to
    be the deliverer, so a renderer that is going to suppress the banner
    (focused window, stale dedupe key) must not reach here. A claim taken for a
    toast nobody sees is delivered-to-nobody forever, and no other surface can
    ever pick it up.

    Cold and receipt-free, unlike ``/seen`` beside it: no bridge is acquired,
    no runtime is started, and neither ``unseen`` nor the read watermark moves.
    Notifying is not reading.
    """
    async with errors():
        claimed = await host(request).claim_notification(session_id, body.completion_token)
        return reply({"claimed": claimed})


@router.post("/v1/desktop/sessions/{session_id}/watch", response_model=CRUDResponse[WatchReceipt])
async def watch(session_id: str, body: Watch, request: Request):
    async with errors(), host(request).session(session_id) as bridge:
        await bridge.watch(body.subscription_id, visible=body.visible, can_notify=body.can_notify)
        return reply({"lease_seconds": 45})


@router.post("/v1/desktop/sessions/{session_id}/warm", response_model=CRUDResponse[WarmReceipt])
async def warm(session_id: str, body: Warm, request: Request):
    """Start this session's runtime without submitting anything to it.

    Closes the one gap that made the desktop app feel slower than the TUI:
    every other route here either reads or mutates, so the first message POST
    for a session paid the entire cold engage inline (1146 ms median; 12-42 ms
    once engaged). The renderer calls this on the first keystroke, so the spawn
    overlaps the time the user spends finishing their sentence.

    NOT THE ONLY WARM ANY MORE. The bridge also warms on a live VISIBLE watch
    lease, without going through this route, and keeps that intent across
    attempts: a live lease's warm is retried while it is live (see
    ``DesktopSessionBridge._lease_warm_loop``), paced by a bounded backoff after
    an attempt that failed. So a session the user is merely looking at is
    already warm when the first click arrives, and one attempted warm being
    lost is not the end of the intent. This route remains the renderer's
    explicit speculation and needs no change to keep working — it is the
    keystroke, which is a user action and is deliberately never paced.

    PRECONDITION, AND IT BINDS THE CALLER: a warm issued while nothing else
    holds the bridge is cancelled when its own request returns; the desktop UI
    satisfies this by firing the warm from the mounted, subscribed session
    panel. The bridge is reference-counted and detaching cancels an in-flight
    warm (a spawn must not outlive the facade it was started against), and this
    request is itself a user of that bridge. A caller with no subscription open
    therefore gets a 200, a ``warming`` receipt, and no warm — the send that
    follows pays the full cold engage exactly as it does today. Stated here
    because the symptom is a benchmark anomaly rather than a failure, and the
    next reader should not have to rediscover it from one.

    RECEIPT-FREE, unlike every mutating route beside it, and that is the point
    rather than an omission. Receipts buy at-most-once for calls that admit
    WORK, so a retried POST cannot run a turn twice. A warm admits nothing and
    is idempotent by construction, so a receipt would add a sqlite write per
    keystroke-debounce on the hottest new path in the app — and would put this
    route on the ``ReceiptConflict`` 409 ladder, where a speculative warm-up
    could answer a typing user with an error.

    ALWAYS 2xx FOR A WARMING FAILURE. The state at return time is genuinely
    "an engage was started"; what becomes of it is not this request's to
    report, and the send that follows reports it properly through its own
    ladder. The non-2xx answers that remain are the ones that mean the call
    itself was not admissible at all: an unknown session (404), a full
    bridge table (409), and a daemon that has LATCHED against new work (503,
    ``daemon-retiring``) — the last is the one refusal that says "this process is
    leaving", not "this call is wrong", so a client reconnects to the successor
    rather than retrying here, both from ``errors()``.

    ``body`` is declared and never read: it exists so FastAPI validates the
    request against a closed model. Dropping the parameter would make the route
    accept any JSON at all, which is the opposite of what the empty model is
    for.
    """
    del body
    async with errors(), host(request).session(session_id) as bridge:
        assert bridge.remote is not None
        return reply({"state": await bridge.warm()})


@router.get("/v1/desktop/sessions/{session_id}/events")
async def events(
    session_id: str,
    request: Request,
    epoch: str | None = Query(default=None, max_length=128),
    after_seq: int = Query(default=0, ge=0),
):
    # Acquire BEFORE returning response headers: invalid identity/capacity must
    # return JSON status, not a misleading 200 followed by a broken SSE stream.
    context = host(request).session(session_id)
    async with errors():
        bridge: DesktopSessionBridge = await context.__aenter__()
        try:
            sub = bridge.subscribe()
        except BaseException:
            await context.__aexit__(None, None, None)
            raise

    # The bridge is acquired ABOVE, before any response exists, so that an
    # invalid session or a full subscriber table is a JSON error rather than a
    # 200 followed by a broken stream. That leaves the release owed by
    # something other than the generator: if the generator is never consumed --
    # the client disconnects between headers and body, or the response is
    # discarded before iteration -- its `finally` never runs and the bridge
    # stays acquired for the process's lifetime, holding a session attached.
    #
    # Released exactly once, from whichever path gets there first: the
    # generator's own teardown for a stream that ran, and the response's
    # background task for one that never did.
    released = False

    async def release_once() -> None:
        nonlocal released
        if released:
            return
        released = True
        await context.__aexit__(None, None, None)

    async def stream():
        try:
            async for frame in bridge.events(sub, epoch=epoch, after_seq=after_seq):
                yield "data: " + json.dumps(frame, separators=(",", ":")) + "\n\n"
        finally:
            await release_once()

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-store",
            "X-Accel-Buffering": "no",
        },
        background=BackgroundTask(release_once),
    )
