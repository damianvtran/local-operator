"""Explicit desktop lifecycle operations through the canonical session runtime."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Annotated, Any, Literal, get_args

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
from local_operator.session.variable_ops import RefusalCode, VariableType, refusal_for

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


class VariableCreate(Input):
    """A new code-memory variable. ``type`` comes from the ONE shared table.

    ``key`` is a free string here rather than a validated path segment: the same
    name has to be refused by the SAME two sentences (``reserved_name``,
    ``invalid_value``) whether it arrives in a body or in a URL, and a pydantic
    pattern would answer 422 for one surface and 409 with a code for the other.
    The generous ``max_length`` above the table's own cap keeps both in one path.
    """

    key: str = Field(default="", max_length=4096)
    # Deliberately NOT capped at the table's MAX_VALUE_CHARS: a value that is too
    # long is a refusal the panel must toast with its own code (409 ``too_large``),
    # and a pydantic cap would turn it into a 422 the renderer cannot map.
    value: str = Field(default="", max_length=200_000)
    type: VariableType


class VariableUpdate(Input):
    """The mutable half of a variable; the key is the path segment."""

    value: str = Field(default="", max_length=200_000)
    type: VariableType


#: A session with no runtime has never had an interpreter, and one whose
#: interpreter was released has none now; the panel's sentence is the same for
#: both ("no code memory yet"), which is why the read answers this WITHOUT
#: engaging anything. Never spawns a runtime: reading a panel must not start a
#: process, and `_spawn` needs a turn's ``ToolContext`` anyway.
_COLD_VARIABLES: dict[str, Any] = {
    "state": "observed",
    "runtime": "absent",
    "kernel": "absent",
    "variables": [],
    "truncated": False,
}


def refuse_variables(code: str, message: str = "") -> None:
    """Raise the refusal envelope the renderer's ``desktopResult`` lifts.

    ``not_found`` is a 404 and every other code a 409: a missing variable is an
    address that does not exist, while the rest are addresses that do exist and
    cannot be acted on right now (no kernel, one busy, a reserved name, a value
    the type cannot hold). The MESSAGE is taken from the shared table unless the
    OWNER supplied one — the owner sees the namespace and can be more specific,
    and the worker's sentences never quote the submitted value by construction
    (see ``session/variable_ops.py``).
    """
    refusal = refusal_for(code)
    if message and code in get_args(RefusalCode):
        refusal = {**refusal, "message": message}
    raise HTTPException(
        404 if refusal["code"] == "not_found" else 409,
        {"code": refusal["code"], "message": refusal["message"]},
    )


def read_variables(answer: dict[str, Any]) -> dict[str, Any]:
    """Map an owner's read answer onto the frozen panel states.

    ``busy`` and ``unsupported`` carry NO ``variables`` key, and that is the
    point of the model: a consumer cannot render "Nothing stored yet" over a
    namespace nobody read. ``variables: []`` means observed and empty, never
    unknown — so the refusals are mapped one cause at a time rather than folded
    into one state: ``no_kernel`` losing a race with a disposing kernel is an
    observed/absent reading, ``changed_under_read`` is the retryable state the
    panel already has (a half-old list never gets answered as a snapshot), and
    anything else is a cause this build cannot report as a READING at all —
    painted ``busy`` it would sit on the "reading…" affordance forever, so it
    takes the terminal state instead.
    """
    state = answer.get("state")
    if state in ("busy", "unsupported"):
        return {"state": state}
    if not answer.get("ok"):
        code = str(answer.get("code") or "")
        if code == "no_kernel":
            return {**_COLD_VARIABLES, "runtime": "running"}
        if code == "changed_under_read":
            return {"state": "busy"}
        return {"state": "unsupported"}
    return {
        "state": "observed",
        "runtime": "running",
        # The owner reports whether an interpreter is resident AT THE MOMENT it
        # answered: a kernel reaped between the cold check and this verb is
        # "absent" with the runtime still running, and that is the panel's other
        # sentence ("the interpreter was released after sitting idle").
        "kernel": "absent" if answer.get("kernel") == "absent" else "resident",
        "variables": list(answer.get("variables") or []),
        "truncated": bool(answer.get("truncated")),
    }


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


@router.get(
    "/v1/desktop/sessions/{session_id}/variables",
    response_model=CRUDResponse[Result],
)
async def variables(session_id: str, request: Request):
    """A session's live code memory, addressed by SESSION id.

    The legacy ``/v1/agents/{id}/execution-variables`` route is keyed by
    agent-directory UUIDs and answered 404 for every canonical session id, which
    is why this surface exists. This one reads the answer off the session's own
    runtime — the only place the namespace exists — and a cold session is read
    WITHOUT engaging one.
    """
    async with errors(), host(request).session(session_id) as bridge:
        assert bridge.remote is not None
        if bridge.remote.is_cold:
            return reply({"data": dict(_COLD_VARIABLES)})
        answer = await bridge.remote.variables_op("list")
        return reply({"data": read_variables(answer)})


@router.post(
    "/v1/desktop/sessions/{session_id}/variables",
    response_model=CRUDResponse[Result],
)
async def create_variable(session_id: str, body: VariableCreate, request: Request):
    """Create a variable in the session's live interpreter namespace."""
    async with errors(), host(request).session(session_id) as bridge:
        assert bridge.remote is not None
        if bridge.remote.is_cold:
            # Mutations never spawn a runtime and never spawn a kernel: a panel
            # editing "no code memory yet" would be starting a process to hold a
            # value in a namespace no cell has ever run in.
            refuse_variables("runtime_cold")
        answer = await bridge.remote.variables_op("set", body.key, body.value, body.type)
        if not answer.get("ok"):
            refuse_variables(str(answer.get("code") or ""), str(answer.get("message") or ""))
        return reply({"data": {"state": "ok", "variable": dict(answer.get("variable") or {})}})


@router.patch(
    "/v1/desktop/sessions/{session_id}/variables/{key}",
    response_model=CRUDResponse[Result],
)
async def update_variable(session_id: str, key: str, body: VariableUpdate, request: Request):
    """Replace an existing variable's value, refusing when no such key is stored."""
    async with errors(), host(request).session(session_id) as bridge:
        assert bridge.remote is not None
        if bridge.remote.is_cold:
            refuse_variables("runtime_cold")
        answer = await bridge.remote.variables_op("update", key, body.value, body.type)
        if not answer.get("ok"):
            refuse_variables(str(answer.get("code") or ""), str(answer.get("message") or ""))
        return reply({"data": {"state": "ok", "variable": dict(answer.get("variable") or {})}})


@router.delete(
    "/v1/desktop/sessions/{session_id}/variables/{key}",
    response_model=CRUDResponse[Result],
)
async def delete_variable(session_id: str, key: str, request: Request):
    """Remove one variable from the session's live interpreter namespace."""
    async with errors(), host(request).session(session_id) as bridge:
        assert bridge.remote is not None
        if bridge.remote.is_cold:
            refuse_variables("runtime_cold")
        answer = await bridge.remote.variables_op("delete", key)
        if not answer.get("ok"):
            refuse_variables(str(answer.get("code") or ""), str(answer.get("message") or ""))
        return reply({"data": {"state": "ok"}})


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
                # A stop never engages a cold runtime merely to shut it down.
                if bridge.remote.is_cold:
                    rows.append({"session_id": target, "status": "already_stopped"})
                else:
                    detail = await bridge.remote.request_stop()
                    rows.append(
                        {"session_id": target, "status": "stop_requested", "detail": detail}
                    )
        return {"data": {"sessions": rows}}

    async with errors():
        # THE ORDERING ASK, distinct from the door below and the reason it is a
        # SECOND call rather than a redundant one: this handler claims a durable
        # receipt (`receipts(request).run`) BEFORE `execute` takes a bridge, and a
        # claimed receipt whose operation never ran is INDETERMINATE for the
        # client's retry (`DesktopReceipts._claim`, `retry_safe=False` —
        # "reconcile session state before issuing a new request"), against a config
        # dir the successor SHARES. A latched daemon that answered the typed
        # `503 daemon-retiring` only after claiming the receipt would leave the
        # client unable to do what that answer tells it to do (retry against the
        # successor). The door still governs admission; this governs ORDER.
        host(request).assert_admitting()
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
    entry: Aside | None = None
    try:
        async with errors(), host(request).session(session_id) as bridge:
            # THE STATE MOVES IN HERE, AFTER THE DOOR, and that ordering is the
            # whole reason this handler is written this way: ``values[...]`` and
            # ``previous.adopted`` are the aside store's admission, they used to be
            # written BEFORE the bridge was taken, and a latched daemon would then
            # refuse a request that had already claimed its ``request_id`` — so the
            # client's retry against the successor answered 409 "This aside request
            # was already used" (review round 2 measured exactly that shape on
            # ``/asides``: 409, past the admission question). Refusing before the
            # claim costs nothing here because ``complete_aside`` is the only thing
            # that needs the entry, and it runs after the door.
            if previous is not None:
                # A continuation owns the prefix. Keeping the old panel adoptable
                # lets two requests promote the same exchange under distinct
                # receipt IDs.
                previous.adopted = True
            entry = Aside(session_id, turns, time.monotonic())
            values[body.request_id] = entry
            assert bridge.remote is not None
            await bridge.remote.bind_runtime()
            answer = await bridge.remote.complete_aside(turns)
            turns.append(Message.assistant(answer))
            return reply(
                {"data": {"aside_id": body.request_id, "text": answer, "off_record": True}}
            )
    finally:
        # Guarded on the entry: a refusal at the door raises out of the block above
        # before anything was claimed, and un-claiming state that was never claimed
        # would be the same half-applied write this reordering exists to remove.
        if entry is not None:
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
        # The ordering ask, for the reason spelled out on ``stop`` above: this
        # route claims its receipt before ``execute`` reaches the door, and a
        # claimed-but-unfinished receipt is indeterminate for the client's retry.
        host(request).assert_admitting()
        return reply(
            await receipts(request).run(
                session_id + ":adopt:" + body.request_id,
                {"aside_id": aside_id, **body.model_dump()},
                execute,
            )
        )
