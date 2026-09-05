"""Additive authenticated session API; legacy per-turn chat stays unchanged."""

from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    field_validator,
    model_validator,
)
from starlette.background import BackgroundTask

from local_operator.server.desktop import require_desktop
from local_operator.server.models.desktop_sessions import (
    AnswerReceipt,
    CommandReceipt,
    CreatedSession,
    HistoryPage,
    MessageAdmission,
    SessionList,
    SessionSnapshot,
    WatchReceipt,
)
from local_operator.server.models.schemas import CRUDResponse
from local_operator.server.utils.desktop_commands import OWNER_COMMANDS, native_action
from local_operator.server.utils.desktop_receipts import (
    DesktopReceipts,
    ReceiptConflict,
)
from local_operator.server.utils.desktop_sessions import (
    DesktopSessionBridge,
    DesktopSessions,
)
from local_operator.session.frontend_state import SlashResult
from local_operator.slash_commands import slash_command_for

router = APIRouter(tags=["Desktop sessions"], dependencies=[Depends(require_desktop)])
RequestID = Annotated[
    str, Field(pattern=r"^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$")
]


class Input(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CreateSession(Input):
    request_id: RequestID
    cwd: str = Field(min_length=1, max_length=4096)


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


def host(request: Request) -> DesktopSessions:
    pool = getattr(request.app.state, "desktop_sessions", None)
    if pool is None:
        pool = DesktopSessions(request.app.state.config_manager.config_dir)
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
    except KeyError:
        raise HTTPException(404, "Session or subscription not found") from None
    except (ReceiptConflict, ValueError) as error:
        raise HTTPException(409, str(error)) from None
    except ConnectionError as error:
        # A cold session that cannot start an owner reports WHY, when the cause
        # was one of the vetted configuration conditions
        # (`launch._ACTIONABLE_STARTUP_REASONS`). Everything else keeps the
        # generic sentence: owner/provider errors can otherwise carry endpoint
        # bodies and credentials into a user-visible surface.
        detail = str(error).strip()
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
    return reply({"sessions": await host(request).list(limit)})


@router.post("/v1/desktop/sessions", response_model=CRUDResponse[CreatedSession])
async def create_session(body: CreateSession, request: Request):
    async def create():
        return {"session_id": await host(request).create(body.cwd)}

    async with errors():
        return reply(
            await receipts(request).run("create:" + body.request_id, body.model_dump(), create)
        )


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


@router.post(
    "/v1/desktop/sessions/{session_id}/messages", response_model=CRUDResponse[MessageAdmission]
)
async def prompt(session_id: str, body: Prompt, request: Request):
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
                images=decode_images(body.images),
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
                # The owner returns attachment metadata, not a started turn.
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


def decode_images(images: list[Image]):
    from local_operator.session.runtime.server import image_blocks

    return image_blocks([image.model_dump() for image in images])


@router.post(
    "/v1/desktop/sessions/{session_id}/answers", response_model=CRUDResponse[AnswerReceipt]
)
async def answer(session_id: str, body: Answer, request: Request):
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


@router.post("/v1/desktop/sessions/{session_id}/watch", response_model=CRUDResponse[WatchReceipt])
async def watch(session_id: str, body: Watch, request: Request):
    async with errors(), host(request).session(session_id) as bridge:
        await bridge.watch(body.subscription_id, visible=body.visible, can_notify=body.can_notify)
        return reply({"lease_seconds": 45})


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
