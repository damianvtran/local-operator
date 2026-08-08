"""Child-session runner for the ``task`` tool (subagent engine).

One :func:`run_subagent` call registers an AsyncJob (``type='task'``) whose
runner builds a CHILD :class:`~local_operator.session.session.Session`,
drives ONE prompt to completion, and settles the job:

- the child gets its own transcript directory under
  ``config_dir()/sessions/<hex>`` (the factory's ephemeral-session shape);
- every child AgentEvent is appended to ``job.trajectory`` serialized
  (``model_dump(mode="json")``), bounded by :data:`TRAJECTORY_CAP`, so the
  TUI can render the run on click-through;
- a THROTTLED relay re-emits child activity on the PARENT session stream as
  ``SubagentStartEvent`` / ``SubagentProgressEvent`` / ``SubagentEndEvent``
  — progress fires on tool starts/ends and assistant message ends, NEVER on
  stream deltas;
- completion is delivered ONLY as ``SubagentEndEvent``: no NoticeEvent and no
  parent-transcript write, so the front end has exactly one delivery path.

Construction reuses the session primitives directly (the way
``session_factory.create_session`` composes a Session) rather than calling
the factory: the factory needs the three legacy managers plus an argparse
namespace to resolve hosting/model/skills, none of which a child needs — the
child inherits the parent's model and STREAM FN (the parent's shared httpx
pool serves any spec, so a ``model_spec`` override works through the same
pipe), the parent's cwd, and the parent's approval handler. What it does NOT
inherit: yolo (always False — the child goes through the same approval gate
as the parent, never around it), skills (a per-launch selection pass would
make spawning expensive and flaky), compaction (children are one-shot), and
the subagent launcher itself (children are one level deep: a grandchild
would register on the CHILD's job manager where no panel looks).

Capacity: registration honours ``AsyncJobManager.at_capacity`` by parking
the job with ``queued=True``; the manager's ``_promote_oldest_queued`` starts
parked jobs whenever any job settles and frees a slot. ``jobs.cancel`` aborts the
child: the manager aborts the job signal (bridged onto ``child.abort``) and
cancels the runner task, and the runner's teardown disposes the child.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import uuid
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from local_operator.harness.types import (
    AgentEndEvent,
    AgentEvent,
    Message,
    MessageEndEvent,
    ModelSpec,
    SubagentEndEvent,
    SubagentProgressEvent,
    SubagentStartEvent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
)
from local_operator.paths import config_dir

if TYPE_CHECKING:
    from local_operator.harness.jobs import AsyncJobManager
    from local_operator.session.session import Session

logger = logging.getLogger(__name__)

#: Bound on the in-memory child-event trajectory kept on the AsyncJob. One
#: dict per child event (JSON-shaped); the oldest entries are dropped past
#: the cap so a chatty child cannot grow a live session without limit.
TRAJECTORY_CAP = 500


def run_subagent(
    label: str,
    prompt: str,
    *,
    parent_session: "Session",
    jobs_manager: "AsyncJobManager",
    model_spec: ModelSpec | None = None,
) -> str:
    """Register one child-session run as a background job; return the job id.

    Synchronous by contract: the ``task`` tool must answer with the job id
    immediately, so registration happens here and the runner coroutine is the
    manager's own task. The parent session's dispose cancels it through
    ``jobs_manager.dispose()`` like every other job.
    """
    queued = jobs_manager.at_capacity()
    job_id = jobs_manager.register(
        "task",
        label,
        _make_runner(
            label=label,
            prompt=prompt,
            parent_session=parent_session,
            jobs_manager=jobs_manager,
            model_spec=model_spec,
        ),
        queued=queued,
    )
    if queued:
        logger.info("subagent job %s (%s) queued: manager at capacity", job_id, label)
    return job_id


def _make_runner(
    *,
    label: str,
    prompt: str,
    parent_session: "Session",
    jobs_manager: "AsyncJobManager",
    model_spec: ModelSpec | None,
) -> Callable[[str, Any, Callable[[str], None]], Awaitable[str | None]]:
    """Build the JobRunFn for one child run (closure over its launch args)."""
    # The parent seam is private-attribute access on purpose: this module is
    # the session's own launch path (Session._launch_subagent is the only
    # production caller), and the session exposes no public emit/stream
    # accessors. ``_emit`` gives the parent's isolated handler fan-out.
    emit = parent_session._emit

    async def runner(
        job_id: str, signal: Any, report_progress: Callable[[str], None]
    ) -> str | None:
        job = jobs_manager.get(job_id)
        if job is not None:
            # ``None`` means "no trajectory yet" to a ``getattr`` probe; the
            # list materializes when the child actually starts running.
            job.trajectory = []
        child: Session | None = None
        unsubscribe: Callable[[], None] | None = None
        # Mutable cells the relay handler writes into across the run.
        final: dict[str, Any] = {"text": "", "error": None}
        try:
            child = await _build_child_session(
                label=label,
                prompt=prompt,
                parent_session=parent_session,
                model_spec=model_spec,
            )
            await emit(SubagentStartEvent(job_id=job_id, label=label, agent_id=child.agent_id))
            unsubscribe = child.subscribe(
                _make_relay(job_id, label, job, emit, report_progress, final)
            )
            bridge = asyncio.create_task(_abort_bridge(signal, child))
            try:
                await child.prompt(prompt)
            finally:
                bridge.cancel()
                with contextlib.suppress(BaseException):
                    await bridge
            if final["error"]:
                # The child's loop reported a provider/turn error; the job
                # must settle failed with it, not completed with the partial
                # text.
                raise RuntimeError(str(final["error"]))
            result_text = final["text"]
            await emit(
                SubagentEndEvent(
                    job_id=job_id, label=label, status="completed", result_text=result_text
                )
            )
            return result_text
        except asyncio.CancelledError:
            # jobs.cancel both aborts the signal (bridged to child.abort) and
            # cancels THIS task. Emit the settle event shielded: the current
            # task is mid-cancellation, but the parent stream must still see
            # the end of the subagent it was shown start.
            with contextlib.suppress(BaseException):
                await asyncio.shield(
                    emit(SubagentEndEvent(job_id=job_id, label=label, status="cancelled"))
                )
            raise
        except Exception as exc:
            await emit(
                SubagentEndEvent(job_id=job_id, label=label, status="failed", error_text=str(exc))
            )
            raise
        finally:
            if unsubscribe is not None:
                unsubscribe()
            if child is not None:
                await _dispose_child(child)

    return runner


async def _abort_bridge(signal: Any, child: "Session") -> None:
    """Translate the job's abort into a graceful child turn abort.

    The manager also hard-cancels the runner task, but the bridge is what
    makes the child's loop settle through its own abort machinery (persisting
    what it produced) instead of dying mid-await.
    """
    await signal.wait()
    child.abort(signal.reason or "cancelled")


async def _dispose_child(child: "Session") -> None:
    """Dispose the child even when the runner itself is being cancelled.

    Shielded: on the cancellation path the outer await raises immediately,
    but the dispose task shielded inside keeps running and completes on the
    loop — the child's transcript flush and task-group close must not be
    skipped because its parent job was cancelled.
    """
    try:
        await asyncio.shield(child.dispose())
    except asyncio.CancelledError:
        pass  # the shielded dispose task continues without us
    except Exception:
        logger.warning("subagent child session dispose failed", exc_info=True)


def _make_relay(
    job_id: str,
    label: str,
    job: Any,
    emit: Callable[[AgentEvent], Awaitable[None]],
    report_progress: Callable[[str], None],
    final: dict[str, Any],
) -> Callable[[AgentEvent], Awaitable[None]]:
    """The child-stream handler: trajectory + throttled parent relay.

    EVERY child event lands in the trajectory; only tool starts/ends and
    assistant message ends become parent-stream progress events — per-delta
    relaying would flood the parent stream while a child streams a long
    message.
    """

    async def relay(event: AgentEvent) -> None:
        if job is not None and job.trajectory is not None:
            job.trajectory.append(event.model_dump(mode="json"))
            overflow = len(job.trajectory) - TRAJECTORY_CAP
            if overflow > 0:
                del job.trajectory[:overflow]
        progress: str | None = None
        if isinstance(event, ToolExecutionStartEvent):
            progress = f"tool: {event.tool_name}"
        elif isinstance(event, ToolExecutionEndEvent):
            progress = f"tool: {event.tool_name} {'failed' if event.is_error else 'done'}"
        elif isinstance(event, MessageEndEvent):
            message = event.message
            if isinstance(message, Message) and message.role == "assistant":
                # Capture the last assistant text as the job's result.
                final["text"] = message.text
                progress = "message end"
        elif isinstance(event, AgentEndEvent):
            if event.error:
                final["error"] = event.error
        if progress is not None:
            # Same string into latest_details so the 1 Hz jobs.list() poll
            # and the event stream agree about what the child is doing.
            report_progress(progress)
            await emit(SubagentProgressEvent(job_id=job_id, label=label, progress=progress))

    return relay


async def _build_child_session(
    *,
    label: str,
    prompt: str,
    parent_session: "Session",
    model_spec: ModelSpec | None,
) -> "Session":
    """Compose the child Session directly (see module docstring for why the
    factory is not reused). Inherits model (unless overridden), stream fn,
    cwd and the parent's approval handler; explicitly NOT yolo."""
    from datetime import datetime

    from local_operator.harness.types import ToolContext
    from local_operator.prompts_api import build_system_blocks
    from local_operator.session.session import Session
    from local_operator.session.transcript import Transcript
    from local_operator.session_factory import _env_details
    from local_operator.tools.registry import create_tools

    session_dir = config_dir() / "sessions" / uuid.uuid4().hex[:12]
    transcript = Transcript(session_dir)
    cwd = parent_session._cwd
    request_approval = parent_session._request_approval
    # The child context carries NO subagent_launcher, jobs, background-bash
    # launcher or wake scheduler: create_tools below therefore advertises
    # none of task/wait/job/wake to the child, keeping it one level deep and
    # free of machinery whose lifetime ends with this one prompt.
    tool_context = ToolContext(
        cwd=cwd,
        session_id=transcript.directory.name,
        agent_id=parent_session.agent_id,
        has_ui=parent_session._has_ui,
        request_approval=request_approval,
    )
    tools = create_tools(tool_context)

    def system_blocks_provider() -> list[str]:
        # Standard block layout (instructions, inventory, env, skills tail)
        # with an empty skills block: per-launch skill selection is cost the
        # one-shot child does not need.
        return build_system_blocks(
            tools, "", _env_details(cwd), datetime.now().strftime("%Y-%m-%d")
        )

    child = Session(
        model=model_spec if model_spec is not None else parent_session.model,
        # The parent's stream fn: one shared httpx pool serves any ModelSpec
        # (the client is chosen per request), so an override rides the same
        # pipe and the pool's lifetime stays the parent's dispose hook.
        stream_fn=parent_session._stream_fn,
        tools=tools,
        transcript=transcript,
        agent_id=parent_session.agent_id,
        system_blocks_provider=system_blocks_provider,
        yolo=False,  # NEVER inherited: the child faces the same gate as the parent
        has_ui=parent_session._has_ui,
        cwd=cwd,
        request_approval=request_approval,
    )
    await child.async_init()
    return child
