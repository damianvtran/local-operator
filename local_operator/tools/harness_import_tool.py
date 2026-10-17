"""The ``harness_import`` agent tool.

One read-only discovery surface plus an explicitly confirmed import. Every
action is backed by :class:`HarnessImportService`; this module parses shapes,
enforces the preview-first contract and strips foreign filesystem locations
from results. It never parses a foreign store, writes a transcript or opens a
network connection itself.

WHY PREVIEW-FIRST IS ENFORCED HERE, not merely documented: an agent that can
import on a bare call will eventually do it while "just looking". `preview`
mints a short-lived token bound to this agent session and to the exact request
(selection + options); `import` refuses without that token, without
`confirm=true`, and without an explicit selection (`session_ids` or `all`).
There is no default-all and no path parameter.
"""

from __future__ import annotations

import re
import secrets
import time
from collections import OrderedDict
from typing import Any, Callable, Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from local_operator.harness.types import (
    AbortSignal,
    AgentTool,
    AgentToolUpdate,
    ToolContext,
    ToolResult,
)
from local_operator.harness_import.jobs import JOBS, ImportJobBusy
from local_operator.harness_import.service import HarnessImportService, normalise_title
from local_operator.harness_import.types import ImportRequest
from local_operator.paths import config_dir
from local_operator.tools.builtin import _error, _guard, _text, _validation_error

TOOL_NAME = "harness_import"
PREVIEW_TTL_SECONDS = 15 * 60
PREVIEW_TOKENS_MAX = 64

#: Foreign source paths are never part of the tool's contract. These keys are
#: stripped recursively from anything a service object hands back.
_BLOCKED_KEYS = frozenset(
    {"path", "alternate_paths", "cwd", "root", "target_session_dir", "source_path"}
)
_ABS_PATH = re.compile(
    r"(?<![\w.:/])(?:~|/Users|/home|/private|/tmp|/var|/Volumes|/opt|/etc|/mnt)/[^\s\"'<>]+"
    r"|(?<![\w])[A-Za-z]:\\[^\s\"'<>]+"
)


class HarnessImportParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    op: Literal["sources", "sessions", "preview", "import", "status", "cancel"]
    source: str | None = None
    session_ids: dict[str, list[str]] | None = None
    all: bool = False
    include_subagents: bool = False
    include_memory: bool = True
    include_instructions: bool = True
    confirm: bool = False
    preview_token: str | None = None
    job_id: str | None = None


#: ``token -> (expires_at, agent_session_key, request)``. Process-local and
#: bounded; a restart simply requires another preview.
_PREVIEWS: "OrderedDict[str, tuple[float, str, ImportRequest]]" = OrderedDict()


def _session_key(context: ToolContext | None) -> str:
    return str(getattr(context, "session_id", "") or "anonymous")


def _service() -> HarnessImportService:
    return HarnessImportService(config_dir())


def _remember_preview(key: str, request: ImportRequest) -> str:
    now = time.time()
    token = f"prev-{secrets.token_hex(8)}"
    _PREVIEWS[token] = (now + PREVIEW_TTL_SECONDS, key, request)
    while len(_PREVIEWS) > PREVIEW_TOKENS_MAX:
        _PREVIEWS.popitem(last=False)
    return token


def _preview_for(token: str | None, key: str) -> ImportRequest | None:
    if not token:
        return None
    entry = _PREVIEWS.get(token)
    if entry is None:
        return None
    expires_at, owner, request = entry
    if expires_at < time.time() or owner != key:
        _PREVIEWS.pop(token, None)
        return None
    return request


def _strip_paths(text: str) -> str:
    return _ABS_PATH.sub("<path>", text)


def _public(value: Any, *, key: str = "") -> Any:
    """Strip foreign paths and keep the one path the contract allows."""
    if key in {
        "instruction_review_path",
        "instruction_review_paths",
        "memory_review_path",
    }:
        return value
    if isinstance(value, dict):
        return {
            str(item_key): (
                _public_title(item)
                if item_key == "title" and isinstance(item, str)
                else _public(item, key=str(item_key))
            )
            for item_key, item in value.items()
            if item_key not in _BLOCKED_KEYS
        }
    if isinstance(value, list):
        return [_public(item) for item in value]
    if isinstance(value, tuple):
        return [_public(item) for item in value]
    if isinstance(value, str):
        return _strip_paths(value)
    return value


def _public_title(value: Any) -> str:
    return normalise_title(str(value)) if value else ""


def _public_source(info: Any) -> dict[str, Any]:
    return {
        "source": info.source,
        "label": info.label,
        "detected": info.detected,
        "session_count": info.session_count,
        "availability": info.availability,
        "detail": _strip_paths(info.detail),
    }


def _public_session(session: Any) -> dict[str, Any]:
    return {
        "source": session.source,
        "source_id": session.source_id,
        "title": _public_title(session.title),
        "created_ts": session.created_ts,
        "updated_ts": session.updated_ts,
        "message_count": session.message_count,
        "compaction_count": session.compaction_count,
        "availability": session.availability,
        "kind": session.kind,
        "warnings": [_strip_paths(warning) for warning in session.warnings],
    }


def _request_from_params(params: HarnessImportParams) -> ImportRequest:
    session_ids = {
        source: [str(value) for value in values]
        for source, values in (params.session_ids or {}).items()
    }
    sources: list[str] | None = [params.source] if params.source else None
    if sources is None and session_ids:
        sources = list(session_ids)
    return ImportRequest(
        sources=sources,
        session_ids=session_ids,
        all=params.all,
        include_subagents=params.include_subagents,
        # The tool's slice is the core contract; cleared segments are not
        # exposed until an adapter can honour them.
        include_cleared_segments=False,
        include_memory=params.include_memory,
        include_instructions=params.include_instructions,
    )


def _unknown_source(service: HarnessImportService, params: HarnessImportParams) -> str | None:
    names = {params.source} if params.source else set((params.session_ids or {}).keys())
    unknown = sorted(name for name in names if name and name not in service.adapters)
    if unknown:
        return f"unknown source {unknown[0]!r}; choose claude, codex or omp."
    return None


def _selection_error(params: HarnessImportParams) -> str | None:
    session_ids = params.session_ids or {}
    has_sessions = any(values for values in session_ids.values())
    if not params.all and not has_sessions:
        return (
            "import needs an explicit session_ids selection or all=true; there is no default-all."
        )
    if params.all and has_sessions:
        return "import accepts all=true or session_ids, never both; choose one scope."
    if params.all and not params.source:
        return "import all=true needs a source (claude, codex or omp)."
    if params.source and session_ids and set(session_ids) != {params.source}:
        return "session_ids must name only the selected source."
    if not params.source and len(session_ids) != 1:
        return "import needs a source when session_ids spans zero or several sources."
    return None


async def execute_harness_import(
    tool_call_id: str,
    args: dict[str, Any],
    signal: AbortSignal | None = None,
    on_update: Callable[[AgentToolUpdate], None] | None = None,
    context: ToolContext | None = None,
) -> ToolResult:
    del signal, on_update
    try:
        params = HarnessImportParams(**args)
    except ValidationError as exc:
        return _validation_error(tool_call_id, TOOL_NAME, exc)
    service = _service()
    key = _session_key(context)

    if params.op == "sources":
        sources = await service.sources()
        public = [_public_source(info) for info in sources]
        lines = [
            f"- {row['source']}: {row['session_count']} sessions, {row['availability']}"
            for row in public
        ]
        return _text(
            tool_call_id,
            TOOL_NAME,
            "Import sources:\n" + ("\n".join(lines) or "- none detected"),
            details={"sources": public},
        )

    if params.op == "sessions":
        if not params.source:
            return _error(tool_call_id, TOOL_NAME, "sessions needs a source.")
        if params.source not in service.adapters:
            return _error(
                tool_call_id,
                TOOL_NAME,
                f"unknown source {params.source!r}; choose claude, codex or omp.",
            )
        sessions = await service.list_sessions(
            params.source, include_subagents=params.include_subagents
        )
        public = [_public_session(session) for session in sessions]
        return _text(
            tool_call_id,
            TOOL_NAME,
            f"{len(public)} {params.source} session(s).",
            details={"source": params.source, "sessions": public},
        )

    if params.op == "preview":
        if not params.source and not params.session_ids:
            return _error(tool_call_id, TOOL_NAME, "preview needs a source or session_ids.")
        if params.all and any((params.session_ids or {}).values()):
            return _error(
                tool_call_id,
                TOOL_NAME,
                "preview accepts all=true or session_ids, never both; choose one scope.",
            )
        unknown = _unknown_source(service, params)
        if unknown:
            return _error(tool_call_id, TOOL_NAME, unknown)
        request = _request_from_params(params)
        if not request.session_ids and not request.all:
            # A source-only preview means "show me everything listed for this
            # source"; the token records all=true so the import call has to
            # repeat the same explicit scope.
            request.all = True
        preview = await service.preview(request)
        token = _remember_preview(key, request)
        public = _public(preview.to_dict())
        public["preview_token"] = token
        estimated = " (estimated)" if preview.message_counts_estimated else ""
        text = (
            f"Preview: {preview.total_sessions} session(s), "
            f"{preview.total_messages} message(s){estimated}, "
            f"{preview.total_compactions} compaction(s). Nothing has been "
            "imported. Pass preview_token with confirm=true and either the "
            "previewed session_ids or all=true."
        )
        if preview.instruction_review_paths:
            text += " Instructions will be saved for review at " + ", ".join(
                preview.instruction_review_paths
            )
        return _text(tool_call_id, TOOL_NAME, text, details=public)

    if params.op == "import":
        if not params.confirm:
            return _error(
                tool_call_id,
                TOOL_NAME,
                "import refused: pass confirm=true after previewing the exact selection.",
            )
        unknown = _unknown_source(service, params)
        if unknown:
            return _error(tool_call_id, TOOL_NAME, "import refused: " + unknown)
        error = _selection_error(params)
        if error:
            return _error(tool_call_id, TOOL_NAME, "import refused: " + error)
        request = _request_from_params(params)
        stored = _preview_for(params.preview_token, key)
        if stored is None:
            return _error(
                tool_call_id,
                TOOL_NAME,
                "import refused: preview first and pass the preview_token from that preview.",
            )
        if stored != request:
            return _error(
                tool_call_id,
                TOOL_NAME,
                "import refused: the selection/options differ from the previewed "
                "request; preview again.",
            )
        try:
            job = JOBS.start(service, request, session_key=key)
        except ImportJobBusy as exc:
            return _error(tool_call_id, TOOL_NAME, f"import refused: {exc}")
        return _text(
            tool_call_id,
            TOOL_NAME,
            f"Import started (job {job.job_id}). Poll status with that job_id.",
            details={"job_id": job.job_id, "state": job.progress.state},
        )

    if params.op == "status":
        job = (
            JOBS.get(params.job_id, session_key=key)
            if params.job_id
            else JOBS.latest(session_key=key)
        )
        if job is None:
            return _error(tool_call_id, TOOL_NAME, "no import job found.")
        payload = _public(JOBS.payload(job))
        text = (
            f"Import job {job.job_id}: {job.progress.state} ({job.progress.phase}, "
            f"{job.progress.completed}/{job.progress.total})."
        )
        if job.result is not None and job.result.instruction_review_path:
            text += " Instructions saved for review at " f"{job.result.instruction_review_path}."
        if job.result is not None and job.result.memory_review_path:
            text += " Untethered memory saved for review at " f"{job.result.memory_review_path}."
        return _text(tool_call_id, TOOL_NAME, text, details=payload)

    if params.op == "cancel":
        try:
            cancelled = _public(JOBS.cancel(params.job_id, session_key=key))
        except KeyError:
            return _error(tool_call_id, TOOL_NAME, "no running import job found to cancel.")
        return _text(
            tool_call_id,
            TOOL_NAME,
            (
                "Import cancellation requested."
                if cancelled.get("cancelled")
                else "Import job already settled."
            ),
            details=cancelled,
        )

    return _error(tool_call_id, TOOL_NAME, f"unknown op {params.op!r}.")


def _approval_tier(args: dict[str, Any]) -> Literal["read", "write", "exec"]:
    return "write" if args.get("op") == "import" else "read"


def build_harness_import_tool(context: ToolContext | None = None) -> AgentTool:
    del context
    return AgentTool(
        name=TOOL_NAME,
        label="Import conversations",
        description=(
            "Import conversations and memory from Claude Code, Codex or omp; "
            "read-only on the source; preview first."
        ),
        parameters=HarnessImportParams.model_json_schema(),
        approval_tier="write",
        call_approval_tier=_approval_tier,
        concurrency="shared",
        execute=_guard(TOOL_NAME)(execute_harness_import),
        describe_approval=lambda args, _cwd: (
            "Import conversations/memory into Local Operator "
            f"from {args.get('source') or 'the selected harness'}?"
        ),
    )
