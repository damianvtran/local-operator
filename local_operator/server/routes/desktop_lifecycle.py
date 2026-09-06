"""Explicit desktop lifecycle operations through the canonical session owner."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field, SecretStr, StrictBool, model_validator

from local_operator.harness.types import Message
from local_operator.mcp.desktop import MCPControl, public_server_config
from local_operator.server.desktop import require_desktop
from local_operator.server.models.schemas import CRUDResponse
from local_operator.server.routes.desktop_sessions import (
    Input,
    RequestID,
    errors,
    host,
    receipts,
    reply,
)

router = APIRouter(tags=["Desktop lifecycle"], dependencies=[Depends(require_desktop)])


class Result(BaseModel):
    data: dict[str, Any]
    replayed: bool = False


class Credential(Input):
    action: Literal["list", "store", "forget"]
    key: str = Field(default="", pattern=r"^[A-Za-z_][A-Za-z0-9_]*$", max_length=128)
    value: SecretStr | None = None
    confirmed: StrictBool = False

    @model_validator(mode="after")
    def shape(self):
        if self.action != "list" and not self.key:
            raise ValueError("Choose a credential name")
        if self.action == "store" and (
            self.value is None or not 0 < len(self.value.get_secret_value()) <= 32768
        ):
            raise ValueError("Enter a non-empty secret of at most 32768 characters")
        if self.action == "forget" and not self.confirmed:
            raise ValueError("Confirm removal of this credential")
        if self.action != "store" and self.value is not None:
            raise ValueError("Only storage accepts a secret")
        return self


class Fork(Input):
    request_id: RequestID
    message: str = Field(default="", max_length=200_000)
    boundary: Literal["next_safe"] = "next_safe"


class Stop(Input):
    request_id: RequestID
    targets: list[Annotated[str, Field(pattern=r"^[a-f0-9]{12}$")]] = Field(
        min_length=1, max_length=100
    )
    confirmed: StrictBool

    @model_validator(mode="after")
    def confirmation(self):
        if not self.confirmed:
            raise ValueError("Confirm the selected session stops")
        return self


class AsideInput(Input):
    request_id: RequestID
    text: str = Field(min_length=1, max_length=32768)
    aside_id: str | None = Field(default=None, pattern=r"^[a-f0-9-]{36}$")


class Adopt(Input):
    request_id: RequestID
    confirmed: StrictBool


@dataclass
class Aside:
    session_id: str
    turns: list[Message]
    created: float
    adopted: bool = False
    running: bool = True


def asides(request: Request) -> dict[str, Aside]:
    values = getattr(request.app.state, "desktop_asides", None)
    if values is None:
        values = {}
        request.app.state.desktop_asides = values
    # Off-record exchanges have no durable journal. A restart/expiry closes the
    # panel rather than silently promoting its private content into history.
    for key, value in list(values.items()):
        if time.monotonic() - value.created > 3600:
            del values[key]
    return values


@router.get("/v1/desktop/sessions/{session_id}/mcp", response_model=CRUDResponse[Result])
async def mcp_status(session_id: str, request: Request):
    async with errors(), host(request).session(session_id) as bridge:
        assert bridge.remote is not None
        if bridge.remote.is_cold:
            from local_operator.mcp.config import (
                load_all_mcp_configs,
                owned_scope_for_source,
            )

            cwd = bridge.remote.frontend_state.cwd
            configs, sources = load_all_mcp_configs(cwd)
            return reply(
                {
                    "data": {
                        "servers": [
                            {
                                "name": name,
                                "source": str(sources.get(name)),
                                "owned_scope": owned_scope_for_source(sources.get(name), cwd),
                                "status": "cold",
                                **public_server_config(cfg),
                            }
                            for name, cfg in configs.items()
                        ],
                        "operations": [],
                        "cold": True,
                    }
                }
            )
        result = await bridge.remote.route_shared_slash(
            "desktop_mcp", MCPControl(action="list").model_dump_json()
        )
        return reply({"data": result["data"]})


@router.post("/v1/desktop/sessions/{session_id}/mcp", response_model=CRUDResponse[Result])
async def mcp_control(session_id: str, body: MCPControl, request: Request):
    async with errors(), host(request).session(session_id) as bridge:
        assert bridge.remote is not None
        await bridge.remote.bind_runtime()
        result = await bridge.remote.route_shared_slash("desktop_mcp", body.model_dump_json())
        if result.get("kind") == "error":
            raise HTTPException(
                409,
                (
                    "The MCP control was refused. Check server ownership, transport and "
                    "current operation state."
                ),
            )
        return reply({"data": result["data"]})


@router.post("/v1/desktop/sessions/{session_id}/credentials", response_model=CRUDResponse[Result])
async def credential(session_id: str, body: Credential, request: Request):
    async with errors(), host(request).session(session_id) as bridge:
        assert bridge.remote is not None
        await bridge.remote.bind_runtime()
        # Never enter the command receipt journal, transcript, or slash args.
        result = await bridge.remote.credential_op(
            body.action, body.key, body.value.get_secret_value() if body.value else ""
        )
        if not result.get("ok"):
            raise HTTPException(409, "The credential operation did not complete")
        return reply({"data": result})


@router.post("/v1/desktop/sessions/{session_id}/fork", response_model=CRUDResponse[Result])
async def fork(session_id: str, body: Fork, request: Request):
    async with errors(), host(request).session(session_id) as bridge:

        async def execute():
            assert bridge.remote is not None
            await bridge.remote.bind_runtime()
            result = await bridge.remote.route_shared_slash("fork", "")
            child_id = result["data"]["session_id"]
            data: dict[str, Any] = {
                "session_id": child_id,
                "parent_id": session_id,
                "boundary": body.boundary,
            }
            if body.message.strip():
                async with host(request).session(child_id) as child:
                    assert child.remote is not None
                    detail, duplicate = await child.remote.admit_prompt(
                        body.message, command_id=body.request_id, images=[]
                    )
                    data["admission"] = {
                        "status": "admitted",
                        "detail": detail,
                        "duplicate": duplicate,
                    }
            return {"data": data}

        return reply(
            await receipts(request).run(
                session_id + ":fork:" + body.request_id, body.model_dump(), execute
            )
        )


@router.post("/v1/desktop/stop", response_model=CRUDResponse[Result])
async def stop(body: Stop, request: Request):
    async def execute():
        # Resolve every target before stopping any. A stale picker selection
        # must not produce a half-applied batch merely because its bad row was last.
        for target in dict.fromkeys(body.targets):
            async with errors(), host(request).session(target):
                pass
        rows = []
        for target in dict.fromkeys(body.targets):
            async with errors(), host(request).session(target) as bridge:
                assert bridge.remote is not None
                # A stop never engages a cold owner merely to shut it down.
                if bridge.remote.is_cold:
                    rows.append({"session_id": target, "status": "already_stopped"})
                else:
                    detail = await bridge.remote.request_stop()
                    rows.append(
                        {"session_id": target, "status": "stop_requested", "detail": detail}
                    )
        return {"data": {"sessions": rows}}

    async with errors():
        return reply(
            await receipts(request).run("stop:" + body.request_id, body.model_dump(), execute)
        )


@router.post("/v1/desktop/sessions/{session_id}/asides", response_model=CRUDResponse[Result])
async def aside(session_id: str, body: AsideInput, request: Request):
    values = asides(request)
    if body.request_id in values:
        raise HTTPException(409, "This aside request was already used")
    if len(values) >= 64:
        raise HTTPException(409, "Close an aside or wait for it to expire")
    previous = values.get(body.aside_id or "")
    if body.aside_id and (
        previous is None
        or previous.session_id != session_id
        or previous.adopted
        or previous.running
        or len(previous.turns) % 2
    ):
        raise HTTPException(409, "This aside is no longer available")
    turns = list(previous.turns) if previous else []
    if len(turns) >= 32:
        raise HTTPException(422, "Start a new aside after 16 exchanges")
    turns.append(Message.user(body.text))
    if previous is not None:
        # A continuation owns the prefix. Keeping the old panel adoptable lets
        # two requests promote the same exchange under distinct receipt IDs.
        previous.adopted = True
    entry = Aside(session_id, turns, time.monotonic())
    values[body.request_id] = entry
    try:
        async with errors(), host(request).session(session_id) as bridge:
            assert bridge.remote is not None
            await bridge.remote.bind_runtime()
            answer = await bridge.remote.complete_aside(turns)
            turns.append(Message.assistant(answer))
            return reply(
                {"data": {"aside_id": body.request_id, "text": answer, "off_record": True}}
            )
    finally:
        entry.running = False
        if previous is not None and len(turns) % 2:
            previous.adopted = False


@router.get(
    "/v1/desktop/sessions/{session_id}/asides/{aside_id}", response_model=CRUDResponse[Result]
)
async def get_aside(session_id: str, aside_id: str, request: Request):
    entry = asides(request).get(aside_id)
    if entry is None or entry.session_id != session_id:
        raise HTTPException(404, "This aside is no longer available")
    return reply(
        {
            "data": {
                "aside_id": aside_id,
                "turns": [turn.model_dump(mode="json") for turn in entry.turns],
                "complete": len(entry.turns) % 2 == 0,
                "adoptable": not entry.adopted and len(entry.turns) % 2 == 0,
            }
        }
    )


@router.delete(
    "/v1/desktop/sessions/{session_id}/asides/{aside_id}", response_model=CRUDResponse[Result]
)
async def close_aside(session_id: str, aside_id: str, request: Request):
    values = asides(request)
    entry = values.get(aside_id)
    if entry is None or entry.session_id != session_id:
        raise HTTPException(404, "This aside is no longer available")
    if entry.running:
        raise HTTPException(409, "Wait for the aside to finish before closing it")
    del values[aside_id]
    return reply({"data": {"aside_id": aside_id, "status": "closed"}})


@router.post(
    "/v1/desktop/sessions/{session_id}/asides/{aside_id}/adopt", response_model=CRUDResponse[Result]
)
async def adopt(session_id: str, aside_id: str, body: Adopt, request: Request):
    if not body.confirmed:
        raise HTTPException(422, "Confirm adding this aside to the conversation")

    async def execute():
        entry = asides(request).get(aside_id)
        if entry is None or entry.session_id != session_id:
            raise HTTPException(404, "This aside is no longer available")
        if entry.adopted or len(entry.turns) % 2:
            raise HTTPException(409, "This aside cannot be adopted")
        # Latch before the first await, including bridge acquisition: separate
        # request IDs can otherwise both pass the check and duplicate history.
        entry.adopted = True
        async with errors(), host(request).session(session_id) as bridge:
            assert bridge.remote is not None
            await bridge.remote.bind_runtime()
            await bridge.remote.adopt_aside(entry.turns)
        return {"data": {"aside_id": aside_id, "status": "adopted"}}

    async with errors():
        return reply(
            await receipts(request).run(
                session_id + ":adopt:" + body.request_id,
                {"aside_id": aside_id, **body.model_dump()},
                execute,
            )
        )
