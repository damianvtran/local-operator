"""Detached generate/modify/status music tools, sharing the normal job manager.

The tools never carry an upstream vendor key and never conflate sound effects
with compositions. Provider capabilities and fixed-quote authority remain on the
Radient server; a local persisted login is only a socket-free availability probe.
"""

from __future__ import annotations

from typing import Any, cast

from pydantic import ValidationError

from local_operator.audiogen.availability import radient_available
from local_operator.harness.types import (
    AgentTool,
    InvalidToolArgumentsError,
    ToolContext,
    ToolResult,
)
from local_operator.i18n.keys import wire_music as mk
from local_operator.i18n.messages import render_resolved
from local_operator.music.client import MusicClient
from local_operator.music.digest import job_text, models_text, started_text
from local_operator.music.jobs import (
    TERMINAL,
    journal,
    read_records,
    register_observer,
    runner,
)
from local_operator.music.models import GenerateMusicParams, MusicStatusParams
from local_operator.tools.builtin import _error, _guard, _text


def _available(context: ToolContext) -> bool:
    from local_operator.paths import config_dir

    return (
        context.jobs is not None
        and context.journal_music_job is not None
        and radient_available(config_dir())
    )


def build_generate_music_tool(context: ToolContext) -> AgentTool | None:
    if not _available(context):
        return None
    return AgentTool(
        name="generate_music",
        label=render_resolved(mk.label()),
        description=(
            "Compose instrumental/exact-lyrics music asynchronously. "
            "Guide: guide://music-generation."
        ),
        parameters=GenerateMusicParams.model_json_schema(),
        approval_tier="write",
        concurrency="shared",
        interruptible=False,
        execute=execute_generate_music,
    )


def build_modify_music_tool(context: ToolContext) -> AgentTool | None:
    if not _available(context):
        return None
    return AgentTool(
        name="modify_music",
        label=render_resolved(mk.label()),
        description=(
            "Compose around owned keep/generate sections. Guide: guide://music-generation."
        ),
        parameters=GenerateMusicParams.model_json_schema(),
        approval_tier="write",
        concurrency="shared",
        interruptible=False,
        execute=execute_modify_music,
    )


def build_music_status_tool(context: ToolContext) -> AgentTool | None:
    if not _available(context):
        return None
    return AgentTool(
        name="music_status",
        label=render_resolved(mk.label()),
        description=(
            "Read music status/models or request cancel; started work can finish and charge."
        ),
        parameters=MusicStatusParams.model_json_schema(),
        approval_tier="read",
        concurrency="shared",
        execute=execute_music_status,
    )


async def _start(
    call_id: str, args: dict[str, Any], context: ToolContext | None, *, modify: bool
) -> ToolResult:
    name = "modify_music" if modify else "generate_music"
    try:
        params = GenerateMusicParams.model_validate(args)
    except ValidationError as exc:
        raise InvalidToolArgumentsError(str(exc)) from exc
    if modify and not any(section.kind == "keep" for section in (params.sections or [])):
        raise InvalidToolArgumentsError("modify_music requires at least one owned keep section")
    if context is None or context.jobs is None or context.journal_music_job is None:
        return _error(
            call_id, name, render_resolved(mk.failed(error="music_background_context_unavailable"))
        )
    # A persisted login is not commercial/provider readiness. Discover before
    # registering or spending; a disabled server gets an honest refusal now.
    client = MusicClient()
    try:
        capabilities = await client.request("models")
    finally:
        await client.close()
    model = next(
        (row for row in capabilities.get("models", []) if row.get("id") == params.model), {}
    )
    if not model.get("enabled") or model.get("operations", {}).get("compose") is not True:
        return _error(
            call_id,
            name,
            render_resolved(mk.failed(error=model.get("disabled_reason") or "music_unavailable")),
            details={"capabilities": capabilities},
        )
    record: dict[str, Any] = {
        "state": "registered",
        "params": params.body(),
        "key": f"music:{context.session_id}:{call_id}",
        "artifacts": [],
    }

    def pending_cancel() -> None:
        disposed = context.session_disposed
        if disposed is not None and disposed():
            return
        record["state"] = "failed"
        record["error"] = "cancelled_before_submit"
        if context.spawn_background:
            context.spawn_background(journal(context, record))

    job_id = cast(Any, context.jobs).register(
        type="music",
        label=render_resolved(mk.label()),
        run=runner(context, record),
        on_cancel=pending_cancel,
    )
    record["job"] = job_id
    await journal(context, record)
    return _text(
        call_id,
        name,
        # Model-facing (English by design): the quote and how to follow the job.
        # ``details`` stays the UI contract, untouched.
        started_text(job_id, record["params"], capabilities),
        details={
            "job_id": job_id,
            "stage": "queued",
            "material": "music",
            "model": params.model,
            "tracks": [],
            "cancel_support": "queued_only",
        },
    )


@_guard("generate_music")
async def execute_generate_music(
    tool_call_id: str,
    args: dict[str, Any],
    signal: Any = None,
    on_update: Any = None,
    context: ToolContext | None = None,
) -> ToolResult:
    del signal, on_update
    return await _start(tool_call_id, args, context, modify=False)


@_guard("modify_music")
async def execute_modify_music(
    tool_call_id: str,
    args: dict[str, Any],
    signal: Any = None,
    on_update: Any = None,
    context: ToolContext | None = None,
) -> ToolResult:
    del signal, on_update
    return await _start(tool_call_id, args, context, modify=True)


@_guard("music_status")
async def execute_music_status(
    tool_call_id: str,
    args: dict[str, Any],
    signal: Any = None,
    on_update: Any = None,
    context: ToolContext | None = None,
) -> ToolResult:
    del signal, on_update
    try:
        params = MusicStatusParams.model_validate(args)
    except ValidationError as exc:
        raise InvalidToolArgumentsError(str(exc)) from exc
    if params.action == "models":
        client = MusicClient()
        try:
            result = await client.request("models")
            return _text(tool_call_id, "music_status", models_text(result), details=result)
        finally:
            await client.close()
    if context is None or context.jobs is None:
        return _error(
            tool_call_id,
            "music_status",
            render_resolved(mk.failed(error="music_background_context_unavailable")),
        )
    record = read_records(context.session_dir).get(params.job_id or "")
    if record is None:
        return _error(
            tool_call_id, "music_status", render_resolved(mk.failed(error="music_job_not_found"))
        )
    if params.action == "cancel" and record.get("state") not in TERMINAL:
        watcher = record.get("watcher", record["job"])
        job = context.jobs.get(watcher)
        if job is not None and job.status == "running":
            await context.jobs.cancel(watcher)
        else:
            from local_operator.music.jobs import cancel_record

            await cancel_record(context, record)
            register_observer(context, record)
        return _text(
            tool_call_id,
            "music_status",
            render_resolved(mk.cancel_requested(job=record["job"])),
            details={"job_id": record["job"], "stage": "cancel_requested"},
        )
    if record.get("state") not in TERMINAL:
        watcher = register_observer(context, record)
        if watcher:
            record["watcher"] = watcher
            await journal(context, record)
    return _text(
        tool_call_id,
        "music_status",
        job_text(record),
        details=record,
    )
