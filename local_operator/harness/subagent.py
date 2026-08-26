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
pipe, and the retry/fallback cascade the stream fn was built with therefore
applies unchanged), the parent's cwd, the parent's approval handler, the
parent's compaction settings (a one-shot child was assumed too short to need
them, but a real review child ran 48 requests / 1.5M tokens — a delegated
task must not bypass the operator's compaction cap), the parent's lazy
internal-URL resolver, the parent's ``/goal`` (a standing constraint binds
the delegated slice too), the parent's transcript→LLM rendering, and the
parent's LIVE MCP manager (see :func:`_child_mcp_wiring`), the parent's
variable store, and the parent's approval MODE.

That last one is a decision, not an accident, and it is the one an operator
has to know about. The child is built ``yolo=False``, which reads like a
protection and is not one: the mode lives in the HANDLER, which the child
inherits, so under ``--yolo`` the parent's handler is ``auto_approve`` and
the child auto-approves too, and a ``/approvals auto`` (or a single ``a``
answer) latched anywhere in a TUI session applies to every subagent spawned
for the rest of that session. AUTO-APPROVE IS SESSION-WIDE, INCLUDING
DELEGATED WORK. It is deliberate: a delegated slice must not be able to
re-demand approval the operator has already granted, and a background job
blocking on a prompt nobody is watching is a hang, not a safety feature. All
``yolo=False`` actually buys is that the child cannot skip the gate object
the way ``Session._build_tool_context`` lets a yolo session skip it.

What it does NOT inherit: the frozen knowledge block (a per-launch semantic
selection pass would make spawning expensive and flaky; ``read skill://`` and
``read guide://`` still resolve on demand through the inherited resolver),
and the session-capability tools ``task``/``wait``/``jobs``/``wake`` —
children are one level deep, because a grandchild registers on the CHILD's
job manager, which no panel renders and which dies with the child's single
prompt. That last exclusion used to be implicit in the child's ToolContext
carrying no launcher; it stopped holding when ``Session.__init__`` grew
``_merge_capability_tools``, which re-derives those four from the session's
OWN context and so handed every child a ``task`` tool. They are pruned
explicitly now (:data:`_CHILD_FORBIDDEN_TOOLS`).

``hub`` is the deliberate exception to that prune, and the mechanism matters:
it is built into the inventory this module CONSTRUCTS (the child's tool
context carries the parent's ``subagent_comms``), so it is never part of what
``_merge_capability_tools`` added and the prune never sees it. A child gets
the child-shaped tool — one peer, its parent — which is how it answers a
question the parent asked and how it reports being blocked without waiting
for its final result. See :mod:`local_operator.harness.comms`.

``jobs`` is a SECOND, CONDITIONAL exception, on a different principle. It
observes and cancels the child's OWN background jobs — it spawns nothing
(that is ``task``) and dies with the child's job manager — so it crosses no
boundary the prune protects. But a child can only produce a background job
while its ``bash`` retains ``background``, and the bash receipt tells the
model to poll such a job with ``jobs(op='peek')``. So the invariant is: a
child keeps ``jobs`` IFF it can still background a bash command. The prune
below re-adds ``jobs`` exactly under that condition (:func:`_can_background`),
which is what stops a non-delegating role or grandchild that backgrounds a
long command from looping forever on ``Tool not found: jobs``.

Approvals the child asks for carry ``ToolContext.job_id`` — the id of the job
this child IS — so a host can scope an approval decision to the delegated
work that provoked it. Live failure it exists for: a subagent outliving its
parent's turn was stamped with that turn's approval state and had its tools
denied with no prompt shown to anyone.

Capacity: registration honours ``AsyncJobManager.at_capacity`` by parking
the job with ``queued=True``; the manager's ``_promote_oldest_queued`` starts
parked jobs whenever any job settles and frees a slot. ``jobs.cancel`` aborts the
child: the manager aborts the job signal (bridged onto ``child.abort``) and
cancels the runner task, and the runner's teardown disposes the child — after
:func:`_persist_inflight` saves the turn the hard cancel pre-empted, so a
``resume_dir`` relaunch replays what the stopped child had already done
rather than only its launch prompt.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from local_operator.agent_profiles import filter_tools
from local_operator.harness.intent import (
    ACTIVITY_RESPONDING,
    ACTIVITY_THINKING,
    batch_activity,
    tool_activity,
)
from local_operator.harness.types import (
    AgentEndEvent,
    AgentEvent,
    AgentTool,
    Message,
    MessageEndEvent,
    MessageStartEvent,
    ModelChangeEvent,
    ModelSpec,
    SubagentEndEvent,
    SubagentProgressEvent,
    SubagentStartEvent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    Usage,
)
from local_operator.paths import config_dir
from local_operator.resume import ORIGIN_SUBAGENT, mark_session_origin

if TYPE_CHECKING:
    from local_operator.agent_profiles import AgentProfile
    from local_operator.harness.comms import SubagentComms
    from local_operator.harness.jobs import AsyncJobManager
    from local_operator.mcp.manager import McpManager
    from local_operator.session.session import Session

logger = logging.getLogger(__name__)

#: Bound on the in-memory child-event trajectory kept on the AsyncJob. One
#: dict per child event (JSON-shaped); the oldest entries are dropped past
#: the cap so a chatty child cannot grow a live session without limit.
TRAJECTORY_CAP = 500


#: The read-only inventory a ``scout`` child is filtered down to. Allowlist,
#: not tier-filter: approval tiers drift as tools are added, and a scout's
#: promise is narrower than "nothing marked write" — it is lookups only, no
#: side effects at all (browser drives the user's browser; eval executes
#: code; both are excluded by name for that reason even where a tier alone
#: would admit them).
#:
#: Kept as the FALLBACK for ``agent="scout"`` when no profile resolves (a
#: stripped install with no packaged seeds, a registry that cannot be read):
#: the read-only promise is a safety property, so it must not depend on a file
#: being present. Role guidance generally lives in
#: :mod:`local_operator.agent_profiles`.
SCOUT_TOOL_ALLOWLIST = frozenset({"read", "glob", "grep", "list_variables", "read_variable"})


def _can_background(tools: "list[AgentTool]") -> bool:
    """Whether this toolset can PRODUCE a background job.

    Only ``bash`` spawns background jobs, and only while its schema still
    carries the ``background`` parameter. A toolset with no such ``bash`` (a
    scout, an allowlist that omits it) cannot register a job, so it has
    nothing for ``jobs`` to observe. Deriving the answer from the actual
    schema — rather than hard-coding which roles background — is what keeps
    the ``jobs``-retention rule below tied to reality if the bash schema or a
    role's allowlist later changes.
    """
    bash = next((tool for tool in tools if tool.name == "bash"), None)
    if bash is None:
        return False
    return "background" in bash.parameters.get("properties", {})


#: Preamble stamped onto a scout prompt when no profile resolves. The tool
#: filter enforces the letter; this states the intent, so the scout REPORTS
#: rather than trying to route around its missing tools.
SCOUT_PREAMBLE = (
    "[scout mode: you are a READ-ONLY research agent. Investigate, read, "
    "search, and report findings with file:line evidence; you cannot edit, "
    "write, or run anything. Your final message is the deliverable.]\n\n"
)


def _specialist_instructions(agent: str, parent_session: "Session") -> str:
    """A non-role agent's own system_prompt.md, or ''."""
    if not agent or agent in {"task", "scout"}:
        return ""
    registry = getattr(parent_session, "agent_registry", None)
    if registry is None or not hasattr(registry, "get_agent_by_name"):
        return ""
    try:
        from local_operator.agent_profiles import is_specialist

        row = registry.get_agent_by_name(agent)
        if row is None or not is_specialist(row):
            # The registry also contains ordinary persistent chat agents. Their
            # prompts can carry private user context and must never be injected
            # into a delegated child merely because its name was supplied.
            return ""
        return (registry.get_agent_system_prompt(row.id) or "").strip()
    except Exception:  # noqa: BLE001 — guidance is enrichment
        logger.warning("could not read specialist instructions for %r", agent)
        return ""


def _resolve_role(agent: str, parent_session: "Session") -> "AgentProfile | None":
    """The profile for ``agent``, or None for a plain full child.

    ``"task"`` is the no-role default and never resolves, so the common launch
    pays no registry lookup at all. Anything else is looked up in the
    operator's registry first and then in the packaged starters, which is what
    lets ``task(agent="reviewer")`` work on a machine where nobody has authored
    a reviewer while still preferring the operator's own once they have.

    Never raises: an unresolvable role degrades to a full child, because the
    parent already decided the work should happen and a typo in a role name is
    not a reason to lose the delegation.
    """

    if not agent or agent == "task":
        return None
    try:
        from local_operator.agent_profiles import resolve_profile

        return resolve_profile(agent, registry=getattr(parent_session, "agent_registry", None))
    except Exception:  # noqa: BLE001 - role guidance is enrichment, not a gate
        logger.warning("could not resolve agent role %r; launching a full child", agent)
        return None


def run_subagent(
    label: str,
    prompt: str,
    *,
    parent_session: "Session",
    jobs_manager: "AsyncJobManager",
    model_spec: ModelSpec | None = None,
    resume_dir: "Path | None" = None,
    agent: str = "task",
    effort: str | None = None,
) -> str:
    """Register one child-session run as a background job; return the job id.

    Synchronous by contract: the ``task`` tool must answer with the job id
    immediately, so registration happens here and the runner coroutine is the
    manager's own task. The parent session's dispose cancels it through
    ``jobs_manager.dispose()`` like every other job.

    ``resume_dir`` continues a PREVIOUS child instead of starting a new one:
    the child is built on that session directory, so ``Transcript`` rehydrates
    it and the new run replays everything the old one said and did before
    reading ``prompt``. Used by ``hub op='resume'`` (see
    :mod:`local_operator.harness.comms`); ``None`` is a fresh child.

    ``effort`` is recorded on the job for display only — the caller has already
    resolved it into ``model_spec`` (``Session._resolve_subagent_model``), so
    the runner never re-reads it. It rides here rather than being derived from
    ``model_spec`` because a tier does not survive that resolution: two tiers
    can point at the same model, and a child on the parent's own model still
    ran at a chosen level the band should name.
    """
    effective_prompt, profile = _effective_prompt(prompt, agent, parent_session)
    queued = jobs_manager.at_capacity()
    job_id = jobs_manager.register(
        "task",
        label,
        _make_runner(
            label=label,
            effective_prompt=effective_prompt,
            parent_session=parent_session,
            jobs_manager=jobs_manager,
            model_spec=model_spec,
            resume_dir=resume_dir,
            agent=agent,
            profile=profile,
        ),
        queued=queued,
    )
    job = jobs_manager.get(job_id)
    launch_message_id = f"subagent-launch:{job_id}"
    if job is not None:
        # Recorded at REGISTRATION, not in the runner: a queued job has not
        # started and may never start, and a reader opening its panel still
        # needs to see what it was asked to do. ``trajectory`` is the opposite
        # case and is deliberately left until the runner, because an empty list
        # would claim the child had begun and produced nothing.
        job.prompt = prompt
        job.effective_prompt = effective_prompt
        job.launch_message_id = launch_message_id
        # Same registration-time rule as ``prompt``: the role and effort tier
        # identify the child before its runner exists, and a queued job that
        # never starts still shows both in the page title and the status band.
        job.agent_role = agent
        job.effort = effort
    # Same reason: the parent must be able to address a child that is parked
    # behind the capacity gate (messages to it buffer until it starts), so the
    # comms record exists from the moment the id does.
    comms = getattr(parent_session, "subagent_comms", None)
    if comms is not None:
        comms.record_launch(
            job_id,
            label,
            parent_job_id=getattr(parent_session, "_job_id", None),
            prompt=prompt,
            effective_prompt=effective_prompt,
            launch_message_id=launch_message_id,
            agent_role=agent,
            effort=effort or "",
        )
    if queued:
        logger.info("subagent job %s (%s) queued: manager at capacity", job_id, label)
    return job_id


def _effective_prompt(
    prompt: str, agent: str, parent_session: "Session"
) -> tuple[str, "AgentProfile | None"]:
    """The exact launch message after reusable and team instruction layers."""
    profile = _resolve_role(agent, parent_session)
    if profile is not None:
        effective_prompt = profile.preamble + prompt
    elif agent == "scout":
        effective_prompt = SCOUT_PREAMBLE + prompt
    else:
        specialist_prompt = _specialist_instructions(agent, parent_session)
        effective_prompt = specialist_prompt + "\n\n" + prompt if specialist_prompt else prompt
    team = getattr(parent_session, "active_team", None)
    if team is not None:
        try:
            effective_prompt = team.member_preamble(agent) + effective_prompt
        except Exception:  # noqa: BLE001 — a bad brief must not lose the child
            logger.warning("could not stamp team preamble for %r", agent, exc_info=True)
    return effective_prompt, profile


def _make_runner(
    *,
    label: str,
    effective_prompt: str,
    parent_session: "Session",
    jobs_manager: "AsyncJobManager",
    model_spec: ModelSpec | None,
    resume_dir: "Path | None" = None,
    agent: str = "task",
    profile: "AgentProfile | None" = None,
) -> Callable[[str, Any, Callable[[str], None]], Awaitable[str | None]]:
    """Build the JobRunFn for one child run (closure over its launch args)."""
    # The parent seam is private-attribute access on purpose: this module is
    # the session's own launch path (Session._launch_subagent is the only
    # production caller), and the session exposes no public emit/stream
    # accessors. ``_emit`` gives the parent's isolated handler fan-out.
    emit = parent_session._emit
    comms = getattr(parent_session, "subagent_comms", None)

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
                prompt=effective_prompt,
                parent_session=parent_session,
                model_spec=model_spec,
                job_id=job_id,
                resume_dir=resume_dir,
                agent=agent,
                profile=profile,
            )
            if job is not None:
                # Off the CHILD, not the parent: ``model_spec`` may have put
                # this child on a different model, which is precisely the fact
                # a reader of the job row needs. The EFFECTIVE label (with a
                # getattr degrade for reduced hosts) because a resumed child
                # may boot straight onto a restored provider fallback, and the
                # job row exists to say which model is actually doing the work.
                job.model_label = str(
                    getattr(child, "effective_model_label", "") or child.model_label
                )
                # From the spec the child was ALREADY built with, so this
                # costs nothing: no registry resolve, no provider discovery,
                # and none of it on anyone's render path.
                job.context_window = int(
                    getattr(
                        getattr(child, "effective_model", None) or child.model,
                        "context_window",
                        0,
                    )
                    or child.model.context_window
                )
                # Live reads may expose descendants before this child settles.
                # The edge is only a lease: the finalizer atomically replaces it
                # with detached components before disposal can evict rows or pin
                # the child Session through its manager callback.
                attach_child_manager = getattr(parent_session.jobs, "attach_child_manager", None)
                if callable(attach_child_manager):
                    attach_child_manager(job_id, child.jobs)
                else:
                    job.child_jobs = child.jobs
            if comms is not None:
                # Before the prompt runs: the parent may already have a
                # question queued for this child, and attach is what flushes
                # it into the child's first injection boundary.
                # ``_transcript`` is the same private seam ``_emit`` above is:
                # this module composes the child, and its transcript directory
                # is what makes the child resumable later.
                comms.attach(job_id, child, child._transcript.directory)
                # The child's transcript directory is the WHOLE basis of resume,
                # and it becomes known only here (the runner just built the
                # child). The job-manager roster hook fired at registration —
                # before this session_dir existed — so persist again now that
                # the record carries a resumable directory, or a crash between
                # launch and settle would leave a snapshot naming a child with
                # no way to reach its transcript. Best-effort: a failed persist
                # must never stop the child from running.
                schedule_persist = getattr(parent_session, "_schedule_subagent_persist", None)
                if callable(schedule_persist):
                    try:
                        schedule_persist()
                    except Exception:  # noqa: BLE001 - persistence is not load-bearing here
                        logger.warning("could not persist roster after attach", exc_info=True)
            await emit(SubagentStartEvent(job_id=job_id, label=label, agent_id=child.agent_id))
            unsubscribe = child.subscribe(
                _make_relay(
                    job_id,
                    label,
                    job,
                    emit,
                    report_progress,
                    final,
                    parent_session.jobs,
                )
            )
            bridge = asyncio.create_task(_abort_bridge(signal, child))
            try:
                # The raw launch task and the role-expanded child prompt are two
                # views of one turn. Persist the job-derived correlation id so
                # projections render that durable row once without text matching.
                await child.prompt(effective_prompt, message_id=f"subagent-launch:{job_id}")
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
            # Recorded on the comms record, not just the job row: the manager
            # sweeps settled rows after its retention window while comms
            # records outlive them so a child stays resumable, and without
            # this the roster (``hub op='list'``) could not say whether a
            # swept child finished or crashed.
            await _publish_terminal_outcome(
                comms,
                emit,
                job=job,
                job_id=job_id,
                label=label,
                status="completed",
                result_text=result_text,
            )
            return result_text
        except asyncio.CancelledError:
            # jobs.cancel both aborts the signal (bridged to child.abort) and
            # cancels THIS task. Emit the settle event shielded: the current
            # task is mid-cancellation, but the parent stream must still see
            # the end of the subagent it was shown start.
            if child is not None:
                await _persist_inflight(child)
            # After _persist_inflight, so the outcome is only recorded once the
            # transcript a resume would replay is actually on disk. A pause
            # arrives here too (it cancels underneath); record_outcome leaves
            # the record's ``paused`` flag alone precisely so the roster can
            # still tell the two apart.
            with contextlib.suppress(BaseException):
                await _publish_terminal_outcome(
                    comms,
                    emit,
                    job=job,
                    job_id=job_id,
                    label=label,
                    status="cancelled",
                )
            raise
        except Exception as exc:
            # The error text is kept on the record as well as the job row: it
            # is what the roster shows for a failed child once the row is
            # swept, which is the state an operator is most likely to be
            # looking at when they ask what went wrong.
            await _publish_terminal_outcome(
                comms,
                emit,
                job=job,
                job_id=job_id,
                label=label,
                status="failed",
                error_text=str(exc),
            )
            raise
        finally:
            if unsubscribe is not None:
                unsubscribe()
            if comms is not None:
                # BEFORE dispose: detach fails any question still waiting on
                # this child with "it finished before answering" rather than
                # leaving the parent to burn its whole timeout on an agent
                # that no longer exists.
                comms.detach(job_id)
            if child is not None:
                try:
                    # Disposal owns cancellation settlement for every running
                    # descendant. Await it before detaching the ledger so a
                    # cancellation cleanup's final provider delta reaches the
                    # manager accumulator even when retention evicts its row.
                    await _dispose_child(child)
                finally:
                    if job is not None:
                        # Clear the live edge even if teardown itself fails: a
                        # retained parent row must never pin the child Session.
                        descendant_usage = child.jobs.accounting_components()
                        detach_child_manager = getattr(
                            parent_session.jobs, "detach_child_manager", None
                        )
                        if callable(detach_child_manager):
                            detach_child_manager(job_id, descendant_usage)
                        else:
                            job.descendant_usage = descendant_usage
                            job.child_jobs = None

    return runner


async def _abort_bridge(signal: Any, child: "Session") -> None:
    """Translate the job's abort into a graceful child turn abort.

    The manager also hard-cancels the runner task, but the bridge is what
    makes the child's loop settle through its own abort machinery (persisting
    what it produced) instead of dying mid-await.
    """
    await signal.wait()
    child.abort(signal.reason or "cancelled")


async def _persist_inflight(child: "Session") -> None:
    """Publish the already-durable state of a cancelled child.

    ``Session._run_turn`` owns message durability in its ``finally`` block, and
    todo mutations are persisted at their tool-completion boundary. Keeping
    those writes with their owners means cancellation never needs a detached
    task that can retain the child or touch its transcript after disposal.
    """
    comms = getattr(child, "_subagent_comms", None)
    job_id = getattr(child, "_job_id", None)
    notify = getattr(comms, "notify_detail_persisted", None)
    if isinstance(job_id, str) and callable(notify):
        notify(job_id)


def _answered_prefix(messages: list[Any]) -> list[Any]:
    """The longest prefix whose every tool call has its result.

    Only the TAIL can be incoherent: a cancel interrupts one in-flight tool
    batch, and every earlier batch completed. So this walks back from the end
    and cuts at the last assistant message whose calls are unanswered,
    stopping at the first fully-answered one. Messages before the cut are
    untouched, which is the point — the child keeps everything it finished
    and loses only the batch it was cancelled inside.
    """
    answered = {
        message.tool_call_id
        for message in messages
        if isinstance(message, Message) and message.role == "tool" and message.tool_call_id
    }
    cut = len(messages)
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if not (isinstance(message, Message) and message.role == "assistant"):
            continue
        if not message.tool_calls:
            continue
        if all(call.id in answered for call in message.tool_calls):
            break
        cut = index
    return messages[:cut]


async def _dispose_child(child: "Session") -> None:
    """Finish child teardown even while the runner itself is being cancelled.

    Shielding alone is insufficient here: it lets teardown continue but returns
    control before descendant cancellation has settled, which makes the caller's
    accounting handoff stale. Keep joining the one dispose task after each outer
    cancellation so teardown remains single-shot and the ledger is final when
    this function returns.
    """
    dispose_task = asyncio.create_task(child.dispose())
    while not dispose_task.done():
        try:
            await asyncio.shield(dispose_task)
        except asyncio.CancelledError:
            continue
        except Exception:
            logger.warning("subagent child session dispose failed", exc_info=True)
            return
    if not dispose_task.cancelled():
        try:
            dispose_task.result()
        except Exception:
            logger.warning("subagent child session dispose failed", exc_info=True)


async def _publish_terminal_outcome(
    comms: "SubagentComms | None",
    emit: Callable[[AgentEvent], Awaitable[None]],
    *,
    job: Any,
    job_id: str,
    label: str,
    status: str,
    error_text: str | None = None,
    result_text: str | None = None,
) -> tuple[str, str | None, str | None]:
    """Resolve and deliver the one terminal fact owned by a child run.

    A terminal outcome exists before its parent event fan-out. Cancellation in
    that fan-out must therefore interrupt delivery, not rewrite completion or
    failure into cancellation. Retrying the interrupted fan-out also reaches
    subscribers skipped when an earlier subscriber was cancelled.
    """
    outcome = (
        comms.record_outcome(
            job_id,
            status,
            error_text=error_text,
            result_text=result_text,
        )
        if comms is not None
        else None
    )
    resolved_status, resolved_error, resolved_result = outcome or (
        status,
        error_text,
        result_text,
    )
    event = SubagentEndEvent(
        job_id=job_id,
        label=label,
        status=resolved_status,
        error_text=resolved_error,
        result_text=resolved_result,
    )
    try:
        await emit(event)
    except asyncio.CancelledError:
        # ``jobs.cancel`` stamps cancellation before interrupting this runner.
        # The terminal fact already won, so restore its live row before retrying
        # delivery to handlers skipped by the interrupted fan-out.
        if job is not None:
            job.status = resolved_status
            job.error_text = resolved_error
            job.result_text = resolved_result
        await emit(event)
    return resolved_status, resolved_error, resolved_result


def _make_relay(
    job_id: str,
    label: str,
    job: Any,
    emit: Callable[[AgentEvent], Awaitable[None]],
    report_progress: Callable[[str], None],
    final: dict[str, Any],
    owner_jobs: Any = None,
) -> Callable[[AgentEvent], Awaitable[None]]:
    """The child-stream handler: trajectory + throttled parent relay.

    EVERY child event lands in the trajectory; only message boundaries and
    tool starts/ends become parent-stream progress events — per-delta relaying
    would flood the parent stream while a child streams a long message.

    The progress string is what the child's ROW says it is doing, and it is
    phrased the way the main conversation's working line phrases the parent's
    step (:mod:`local_operator.harness.intent`): the model's own intent while a
    tool runs, ``running N tools`` for a batch, ``responding`` while prose
    streams, ``thinking`` in the model call between them. It used to read
    ``tool: bash done`` — the mechanism rather than the work, which is the
    exact narration the intent field exists to replace, and a reader watching
    both surfaces at once should not have to learn two vocabularies for one
    state.

    ``running`` is the live tool-call set, kept because the phrase for a batch
    is a COUNT: a relay that only remembered the last event said ``thinking``
    the moment one call of three settled, with two still running.
    """
    running: dict[str, str] = {}

    async def relay(event: AgentEvent) -> None:
        if job is not None and job.trajectory is not None:
            job.trajectory.append(event.model_dump(mode="json"))
            overflow = len(job.trajectory) - TRAJECTORY_CAP
            if overflow > 0:
                del job.trajectory[:overflow]
        progress: str | None = None
        if isinstance(event, ToolExecutionStartEvent):
            running[event.tool_call_id] = tool_activity(event.tool_name, event.intent)
            progress = batch_activity(list(running.values()))
        elif isinstance(event, ToolExecutionEndEvent):
            running.pop(event.tool_call_id, None)
            # Back to the model as soon as the batch empties: a settled call is
            # not the child's current activity, and the ledger the page draws
            # already carries its outcome.
            progress = batch_activity(list(running.values())) if running else ACTIVITY_THINKING
        elif isinstance(event, MessageStartEvent):
            progress = ACTIVITY_RESPONDING
        elif isinstance(event, MessageEndEvent):
            message = event.message
            if isinstance(message, Message) and message.role == "assistant":
                # Capture the last assistant text as the job's result.
                final["text"] = message.text
                _accumulate_usage(job, message.usage)
                note_usage_changed = getattr(owner_jobs, "note_usage_changed", None)
                if callable(note_usage_changed):
                    note_usage_changed()
                progress = ACTIVITY_THINKING
        elif isinstance(event, ModelChangeEvent):
            # Keep the job row's label truthful about which model is doing the
            # child's work — the band and the jobs list read it live, and a
            # child that fell over to another provider mid-run is exactly what
            # a reader of those surfaces needs to know.
            if job is not None:
                job.model_label = f"{event.provider}/{event.model_id}"
                if event.context_window > 0:
                    job.context_window = event.context_window
        elif isinstance(event, AgentEndEvent):
            if event.error:
                final["error"] = event.error
        if progress is not None:
            # Same string into latest_details so the 1 Hz jobs.list() poll
            # and the event stream agree about what the child is doing.
            report_progress(progress)
            await emit(SubagentProgressEvent(job_id=job_id, label=label, progress=progress))

    return relay


def _accumulate_usage(job: Any, usage: "Usage | None") -> None:
    """Fold one child ``message_end``'s usage into the job's running total.

    Summed per assistant message rather than taken from the final one: a
    tool-using child spends most of its tokens in the earlier model calls of
    the same run, so the last message's usage understates the child by
    whatever the tool loop cost.

    ``context_tokens`` is point-in-time (how full the child's window was on
    that request), so it is REPLACED, never summed. The field stays ``None``
    until a provider actually reports something: a zeroed total would read as
    "this child used nothing" when the truth is "nobody told us".
    """
    if job is None or usage is None:
        return
    total = job.usage
    if total is None:
        first = usage.model_copy()
        first.cost_components = [
            component.model_copy() for component in usage.cost_components or [usage]
        ]
        # An aggregate receipt is meaningful only when it covers the aggregate.
        # Components retain each call's receipt, so leave the outer field unset
        # and force readers through the provenance-preserving path.
        first.usd_cost = None
        job.usage = first
        return
    total.input_tokens += usage.input_tokens
    total.output_tokens += usage.output_tokens
    total.cache_read_tokens += usage.cache_read_tokens
    total.cache_write_tokens += usage.cache_write_tokens
    # Child failover can mix provider receipts and table-priced calls. Preserve
    # every original call so the TUI can price each one independently instead of
    # treating one receipt as authoritative for the aggregate token buckets.
    total.cost_components.extend(
        component.model_copy() for component in usage.cost_components or [usage]
    )
    total.usd_cost = None
    if usage.context_tokens is not None:
        total.context_tokens = usage.context_tokens


@dataclass(frozen=True)
class _ChildMcp:
    """The child's slice of the parent's MCP surface (see :func:`_child_mcp_wiring`).

    ``attach`` is called once, after the child Session exists, because lazy
    activation has to refresh the child's inventory and the closure cannot
    hold a session that has not been constructed yet.

    ``catalogue`` is a callable and not a string, matching the parent's
    ``knowledge_hooks.mcp_catalogue`` exactly: ``/mcp reload`` replaces
    ``McpManager._configs`` wholesale, and a catalogue frozen at child build
    would go on advertising a server the operator just removed while hiding
    one they just added.
    """

    tools: list[AgentTool]
    catalogue: Callable[[], str]
    resolve: Callable[[str], str | None]
    attach: Callable[["Session"], None]


def _child_mcp_wiring(parent_session: "Session") -> _ChildMcp | None:
    """Give the child the PARENT's MCP surface, on the parent's live manager.

    The reported failure: a delegated task could not call the Linear MCP tools
    its parent had, so it reached for the parent's stored OAuth token and made
    raw API calls instead. A child built from ``create_tools`` alone has no MCP
    tools, no ``mcp://`` resolver and no catalogue, so from inside the child
    those servers do not exist at all — improvising with credentials is the
    only route left, which is a capability gap and a credential-handling
    problem at once.

    The child BORROWS the parent's manager instead of running a second
    discovery pass: discovery costs a process spawn or an HTTP round trip per
    server plus an OAuth exchange, the duplicate connections would live for one
    prompt, and two managers racing the same refresh is exactly the token churn
    the shared auth store exists to prevent. Borrowing has two consequences,
    both deliberate. The child registers NO dispose hook — ``disconnect_all``
    belongs to the parent, and a child tearing the servers down mid-session
    would break the parent. And the child does NOT call
    ``set_on_tools_changed``: that is a single slot the parent already holds,
    so installing there would freeze the PARENT's inventory for the rest of the
    session. The cost is that a reconnect during the child's run leaves the
    child holding stale ``AgentTool`` objects, which is harmless — their
    execute closes over the manager plus the (server, tool) pair, so calls
    still route and still reconnect (``manager._execute_tool_call``); only a schema
    changed mid-run is missed, over a window bounded by one prompt.

    Activation is the parent's lazy path unchanged: ``read mcp://<server>``
    lists a server, ``read mcp://<server>/<tool>`` activates exactly one tool —
    into the CHILD's inventory. The child starts from the set the parent has
    already activated, derived from the parent's live tool list rather than
    plumbed out of ``wire_mcp_into_session``'s closure: a tool the manager
    knows is in that list exactly when the parent activated it, so the fact is
    already public. The parent paid those schemas' token cost for the very task
    it is now delegating.

    ``None`` when the parent has no manager: MCP unconfigured, SDK missing, or
    a bare ``Session`` built by a host that never wired one.
    """
    manager: McpManager | None = getattr(parent_session, "mcp_manager", None)
    if manager is None:
        return None

    from local_operator.mcp.resources import make_mcp_resolver, render_mcp_catalogue

    def origin(tool: AgentTool) -> tuple[str, str] | None:
        meta = manager.get_tool_meta(tool.name)
        if meta is None:
            return None
        return (str(meta.get("server_name", "")), str(meta.get("mcp_tool_name", "")))

    enabled: set[tuple[str, str]] = {
        found for tool in parent_session._tools if (found := origin(tool)) is not None
    }
    child: Session | None = None
    base: list[AgentTool] = []

    def selected(source: list[AgentTool]) -> list[AgentTool]:
        return [tool for tool in source if origin(tool) in enabled]

    def activate(server_name: str, raw_tool_name: str) -> None:
        enabled.add((server_name, raw_tool_name))
        if child is not None:
            child.refresh_tools(base + selected(manager.get_tools()))

    def attach(session: "Session") -> None:
        nonlocal child, base
        child = session
        # The non-MCP base is DERIVED, not remembered: Session.__init__ merges
        # its own capability tools in and this module prunes some back out, so
        # the constructor's list is not what the session ended up with.
        base = [tool for tool in session._tools if origin(tool) is None]

    return _ChildMcp(
        tools=selected(manager.get_tools()),
        catalogue=lambda: render_mcp_catalogue(manager),
        resolve=make_mcp_resolver(manager, activate),
        attach=attach,
    )


async def _build_child_session(
    *,
    label: str,
    prompt: str,
    parent_session: "Session",
    model_spec: ModelSpec | None,
    job_id: str,
    resume_dir: "Path | None" = None,
    agent: str = "task",
    profile: "AgentProfile | None" = None,
) -> "Session":
    """Compose the child Session directly (see module docstring for why the
    factory is not reused, and for the full inherit/do-not-inherit list).

    ``profile`` is the resolved role (see :func:`_resolve_role`), passed in
    rather than re-resolved here so one launch performs exactly one registry
    lookup and the prompt the caller stamped cannot disagree with the tool
    surface applied here.
    """
    from datetime import datetime

    from local_operator.config import ConfigManager
    from local_operator.harness.types import ToolContext
    from local_operator.prompts_api import build_system_blocks
    from local_operator.session.session import Session
    from local_operator.session.transcript import Transcript
    from local_operator.session_factory import _env_details, load_user_instructions
    from local_operator.tools.registry import create_tools

    # A resumed child is built on the STOPPED child's directory, and that is
    # the whole of the resume mechanism: ``Transcript.__init__`` reads the
    # file back, and ``Session.__init__`` seeds its ``LoopContext`` from
    # ``build_llm_history()``, so the new run starts holding everything the
    # old one said, did, and read. Same path the CLI's ``--resume`` takes.
    session_dir = (
        resume_dir if resume_dir is not None else config_dir() / "sessions" / uuid.uuid4().hex[:12]
    )
    # CLAIM FIRST, before anything else creates the directory. A child writes
    # its own directory under the same ``sessions/`` store the retention sweep
    # reclaims, and subagents routinely outlive the sweep another session's
    # startup runs. ``origin.json`` (written just below) already counts as
    # content and so protects the directory once it lands — but the claim is
    # liveness rather than content: it closes the window BEFORE the stamp, and
    # unlike content it lets the sweep still reclaim the directory of a child
    # whose process has died. ``claim_session`` creates the directory and
    # writes the marker in one step, so claiming here leaves no unclaimed-empty
    # window. The pid is this process's — the process whose death makes the
    # directory dead.
    from local_operator.session.retention import claim_session

    claim_session(session_dir)
    # Stamp the directory as the machine's BEFORE the transcript exists, so a
    # picker painted while this child is mid-run already knows what it is. A
    # child's directory is shape-identical to a user conversation, which is how
    # every delegated reviewer, designer and scout run ended up offered under
    # ``/resume`` as if the user had opened it. Re-stamped on resume as well:
    # ``hub op='resume'`` rebuilds a child on its old directory, and a marker
    # lost to an earlier failed write is worth retrying while we are here.
    mark_session_origin(session_dir, ORIGIN_SUBAGENT, label=label, agent=agent)
    transcript = Transcript(session_dir)
    # The operator's standing instructions are machine-wide, so a delegated
    # slice inherits them for the same reason it inherits the goal: the parent
    # authoring a task prompt is not a reliable channel for a preference the
    # operator meant to apply everywhere. Re-read here rather than plumbed off
    # the parent — children are built outside the factory, and this keeps the
    # one source of truth in one function.
    user_instructions = load_user_instructions()
    cwd = parent_session._cwd
    request_approval = parent_session._request_approval
    mcp = _child_mcp_wiring(parent_session)
    parent_resolver = parent_session._skill_resolver

    def resolve_internal_url(url: str) -> str | None:
        # MCP FIRST, and the order is load-bearing: the parent's resolver
        # chains guide:// then skill:// then its OWN mcp:// link, and that last
        # link activates into the PARENT's inventory — a child reading
        # ``mcp://linear/list_issues`` through it would enable the tool on the
        # wrong session and see nothing appear in its own. Asking the child's
        # resolver first fixes that without having to decompose the parent's
        # chain, because ``make_mcp_resolver`` returns None for every URL that
        # is not ``mcp://`` — guide:// and skill:// fall through untouched.
        if mcp is not None:
            handled = mcp.resolve(url)
            if handled is not None:
                return handled
        return parent_resolver(url) if parent_resolver is not None else None

    # The child context carries no subagent_launcher, jobs or wake scheduler,
    # so create_tools advertises none of task/wait/jobs/wake here. That is no
    # longer sufficient on its own — Session.__init__ re-derives them from the
    # session's own context — so the merge is undone after construction.
    #
    # ``subagent_comms`` is the PARENT's instance and is why ``hub`` survives
    # the prune below. Not because the object here is the one that lives: the
    # merge in ``Session.__init__`` REPLACES it with a tool built from the
    # child's own context (verified: the constructed AgentTool is not the one
    # in ``child._tools`` afterwards). The NAME is what spares it — the prune
    # removes what the merge ADDED, and ``hub`` was already present.
    #
    # So the load-bearing invariant is not this line but the merge-time
    # context: ``Session._build_tool_context`` passes ``job_id``, and
    # ``is_child(job_id)`` is what makes the replacement the CHILD shape
    # (message your parent) rather than the parent shape (address, steer,
    # stop and resume your children). If ``job_id`` ever stopped reaching
    # that context, a child would silently be handed its parent's tool.
    tool_context = ToolContext(
        cwd=cwd,
        session_id=transcript.directory.name,
        agent_id=parent_session.agent_id,
        job_id=job_id,
        has_ui=parent_session._has_ui,
        request_approval=request_approval,
        resolve_internal_url=resolve_internal_url,
        subagent_comms=getattr(parent_session, "subagent_comms", None),
        # The child can work with roles too (look one up, or record what it
        # learned about a bad one), and role resolution for its OWN launches
        # needs the same registry the parent used.
        agent_registry=getattr(parent_session, "agent_registry", None),
        team_registry=getattr(parent_session, "team_registry", None),
        web_search_settings=ConfigManager(config_dir()).get_config_value("web_search", None),
        web_fetch_settings=ConfigManager(config_dir()).get_config_value("web_fetch", None),
    )
    tools = create_tools(tool_context)
    # A role's tool allowlist is a capability boundary, not advice: a reviewer
    # that cannot call ``edit`` cannot "helpfully" fix what it was asked to
    # review and thereby end up reviewing its own patch.
    restricted = profile is not None and bool(profile.tools)
    # Captured BEFORE the allowlist filter: a restricted child must keep the
    # ability to ANSWER its parent even when its role allowlist does not name
    # ``hub`` (the installed reviewer profile is read/glob/grep/bash/todo).
    # Without this, every ``hub op='ask'`` to such a child timed out BY
    # DESIGN — the child saw the question, tried to answer with the one tool
    # it knew for talking to the parent, got "Tool not found", and the parent
    # burned its whole budget waiting for a reply that could never be sent.
    # ``hub`` is a messaging surface, not a capability: it cannot edit, write
    # or execute anything the allowlist denies, so sparing it weakens no
    # boundary.
    hub_tool = next((tool for tool in tools if tool.name == "hub"), None)
    if restricted:
        tools = filter_tools(tools, profile)
    elif agent == "scout":
        # Fallback for a scout with no resolvable profile — the read-only
        # promise must not depend on a seed file being present.
        tools = [tool for tool in tools if tool.name in SCOUT_TOOL_ALLOWLIST]
        restricted = True
    if restricted and hub_tool is not None and not any(tool.name == "hub" for tool in tools):
        tools = list(tools) + [hub_tool]
    # MCP tools execute arbitrary server calls, so a role filtered to an
    # allowlist never receives them: they are excluded wholesale rather than
    # trusted per tool, since the allowlist cannot name servers it has not met.
    if mcp is not None and not restricted:
        tools = tools + mcp.tools

    def system_blocks_provider(model_label: str = "") -> list[str]:
        # ``model_label`` is passed by the child Session each turn (its own
        # ``model_label``), which for a subagent is the resolved effort-tier
        # override or the parent's model. Surfacing it lets a delegated
        # reviewer/designer name the model it actually ran on in its byline
        # instead of guessing.
        #
        # Standard block layout. The lazy-knowledge tail carries the MCP
        # catalogue and nothing else: re-running semantic skill selection per
        # one-shot child would add cost without giving the parent a new durable
        # capability, but the catalogue is a bounded list of server names the
        # parent has ALREADY discovered, and without it the child has no way to
        # learn that ``read mcp://<server>`` is a thing to try.
        #
        # The goal rides the same tail. ``/goal`` is a standing constraint the
        # operator set on the whole session ("don't touch prod"), and a
        # delegated slice of that session is exactly where an unstated
        # constraint gets violated — the parent authoring the task prompt is
        # not a reliable channel for a rule the operator meant to apply to
        # everything. Read once per call off the parent's live holder, so a
        # ``/goal`` edit reaches children spawned after it.
        store = getattr(parent_session, "_variables", None)
        names = (
            store.credential_names()
            if store is not None and hasattr(store, "credential_names")
            else []
        )
        return build_system_blocks(
            tools,
            mcp.catalogue() if mcp is not None else "",
            _env_details(cwd),
            datetime.now().strftime("%Y-%m-%d"),
            goal=parent_session.goal,
            user_instructions=user_instructions,
            credentials=names,
            model_label=model_label,
        )

    child = Session(
        model=model_spec if model_spec is not None else parent_session.model,
        # The parent's stream fn: one shared httpx pool serves any ModelSpec
        # (the client is chosen per request), so an override rides the same
        # pipe and the pool's lifetime stays the parent's dispose hook. The
        # retry/failover cascade is baked into that stream fn at construction,
        # which is how the operator's fallback chain reaches the child too.
        stream_fn=parent_session._stream_fn,
        tools=tools,
        transcript=transcript,
        agent_id=parent_session.agent_id,
        system_blocks_provider=system_blocks_provider,
        # The FLAG is never inherited, and that buys less than it sounds like:
        # ``Session._build_tool_context`` passes no gate at all when ``_yolo``
        # is set, so all this prevents is the child skipping the gate OBJECT.
        # The parent's approval MODE still applies, because the mode lives in
        # the handler below, not in this flag. Stated, not assumed: see the
        # module docstring.
        yolo=False,
        has_ui=parent_session._has_ui,
        cwd=cwd,
        request_approval=request_approval,
        # Which job the child's approvals belong to, so a host can scope a
        # denial to the work that provoked it. Reaches the executor through
        # ``Session._build_tool_context``; the construction-time context above
        # only feeds createIf.
        job_id=job_id,
        # The PARENT's comms instance, so the child's every-turn tool context
        # rebuild keeps pointing at the agent that delegated to it instead of
        # minting a private one nobody is listening to.
        subagent_comms=getattr(parent_session, "subagent_comms", None),
        # The parent's variable store: same cwd, same config overrides, so a
        # child reading a variable must see exactly what its parent would.
        variables=parent_session._variables,
        # The same registry the parent resolves roles against, so a child that
        # delegates (a manager) or inspects a role sees the operator's profiles
        # rather than falling back to the packaged starters.
        agent_registry=getattr(parent_session, "agent_registry", None),
        team_registry=getattr(parent_session, "team_registry", None),
        skill_resolver=resolve_internal_url,
        # How the transcript renders into LLM messages. Today every host uses
        # the default, so this changes nothing; it is plumbed because a host
        # that DOES override it would otherwise have its children silently
        # rendering their history by different rules than their parent.
        convert_to_llm=parent_session._convert_to_llm,
        # The parent's compaction budget. A one-shot child was assumed to be
        # too short to need compaction, but a real review child ran 48
        # requests / 1.5M tokens before its default (600k-cap) threshold
        # ever fired — the CAP the parent's operator set must bound the child
        # too, or a delegated task silently bypasses the very knob that keeps
        # long sessions alive. Defensively COPIED so the child can never
        # mutate the parent's settings (they are logically separate).
        compaction_settings=(
            parent_session._compaction_settings.model_copy()
            if parent_session._compaction_settings is not None
            else None
        ),
    )
    # Undo ``Session.__init__``'s capability merge, DEPTH-AWARE. The set is
    # DERIVED, not a copy of ``session.SESSION_CAPABILITY_TOOLS``: nothing links
    # a copy to that tuple, so the next session-gated tool added to it would be
    # handed to every child silently — the exact rot the module docstring says
    # this prune exists to stop. The merge only appends new names or replaces
    # same-named entries, so whatever the constructor ADDED to the list we passed
    # in is precisely the set of tools gated on session capabilities.
    #
    # A child of a TOP-LEVEL session keeps task/wait/jobs: one further level
    # of delegation (map-then-fan-out inside a child) is observable through
    # the child's own jobs/wait tools, and its job manager is disposed with
    # the child, so a grandchild cannot outlive its lineage. What still never
    # crosses any boundary: ``wake`` (a child session ends after one prompt,
    # so a wake armed there would be silently lost) and, one level deeper,
    # everything again — a grandchild's children would register on a manager
    # nothing observes and that dies mid-turn. Scouts lose the whole set: a
    # read-only agent that delegates autonomous work is not read-only.
    #
    # ``refresh_tools`` rather than touching ``_tools``: it is the committed
    # hook and it keeps the loop's ``context.tools`` in step.
    merged_in = {tool.name for tool in child._tools} - {tool.name for tool in tools}
    parent_is_child = parent_session._job_id is not None
    # A role that does not delegate loses the whole capability set, for the
    # same reason a scout does: a reviewer or coder spawning its own children
    # turns one delegated slice into a fan-out nobody is watching. Roles that
    # coordinate (``delegate: yes``) keep it.
    role_forbids_delegation = profile is not None and not profile.may_delegate
    if agent == "scout" or parent_is_child or role_forbids_delegation:
        drop = merged_in
    else:
        drop = {name for name in merged_in if name == "wake"}
    # ``jobs`` is the OBSERVE/CONTROL surface over this child's OWN background
    # jobs (peek at output, cancel) — it spawns nothing (that's ``task``) and
    # dies with the child's job manager, so it crosses no boundary the prune
    # protects. Meanwhile the ``bash`` tool's ``background=true`` receipt tells
    # the model to "follow it with jobs(op='peek')". When the branch above
    # dropped the whole ``merged_in`` set (non-delegating role, grandchild),
    # that advice pointed at a tool that no longer existed, so a child that
    # backgrounded a long command (a coder polling a 10-min pyright) spun
    # forever emitting ``Tool not found: jobs``. Invariant, encoded here rather
    # than as two edits that can silently drift: a child keeps ``jobs`` IFF it
    # can still produce a background job (its ``bash`` retains ``background``).
    # Un-pruning ``jobs`` (not stripping ``bash``'s ``background``) preserves a
    # real capability — a child genuinely benefits from backgrounding a long
    # build and polling it — while killing the loop; stripping the schema
    # per-session would be more invasive and would remove that capability.
    # ``task``/``wait``/``wake`` keep their treatment: ``jobs`` polling is
    # non-blocking and is the advertised path, so sparing ``jobs`` alone is the
    # minimal correct fix, and a child that must not fan out still cannot.
    if _can_background(tools):
        drop = drop - {"jobs"}
    child.refresh_tools([tool for tool in child._tools if tool.name not in drop])
    if mcp is not None:
        mcp.attach(child)
        # Diagnostics only, and BORROWED: unlike attach_mcp_dispose this adds no
        # disconnect hook, because the child does not own the servers.
        child.mcp_manager = parent_session.mcp_manager
        child.mcp_startup = parent_session.mcp_startup
    await child.async_init()
    return child
