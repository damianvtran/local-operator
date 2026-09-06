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
child inherits the parent's model and a conversation-owned stream handle
(the parent's shared httpx pool serves any spec, while routing, effort,
callbacks and analytics identity are isolated per child), the parent's cwd,
the parent's approval handler, the
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

The child inherits a bounded directory of the parent's selected knowledge and
repository guidance, with on-demand ``read skill://`` / ``read guide://``
resolution. It does not copy the parent's full conversation. The
session-capability tools ``task``/``wait``/``jobs``/``wake`` are scoped below —
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

A role's tool allowlist bounds CHANGE, not reach. An allowlisted child keeps
the read-only network tools even when its allowlist does not name them
(:func:`_with_network_floor` — a role installed under an older release carries
that release's tool list frozen into its registry row, and a research role that
cannot search the web is structurally unable to work), and it inherits the MCP
tools the parent had already enabled while being refused the ability to enable
more (:func:`_child_mcp_wiring`). Nothing in either path grants an edit, a
write or an execution the allowlist denies.

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
import weakref
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from local_operator.agent_profiles import (
    READ_ONLY_NETWORK_TOOLS,
    READ_ONLY_TOOLS,
    filter_tools,
)
from local_operator.harness.intent import (
    ACTIVITY_RESPONDING,
    ACTIVITY_THINKING,
    batch_activity,
    tool_activity,
)
from local_operator.harness.jobs import TRAJECTORY_SEQ_KEY
from local_operator.harness.types import (
    AgentEndEvent,
    AgentEvent,
    AgentTool,
    Message,
    MessageEndEvent,
    MessageStartEvent,
    MessageUpdateEvent,
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


class SubagentModelUnavailable(RuntimeError):
    """A launch asked for an effort tier that cannot be honoured.

    Raised BEFORE a child is registered, so the ``task`` tool reports it as
    "could not launch" with the tier and the reason, and no job row ever
    exists for a child that would have run on the wrong model. Deliberately
    not a subclass of ``ValueError``: the tool's argument validation has its
    own error shape, and this is a configuration/availability fact about the
    machine, not a malformed call.

    ``tier`` and ``reason`` are attributes as well as message text so a
    caller that wants to react (retry on another tier, tell the operator
    which key to fix) does not have to parse the string.
    """

    def __init__(self, tier: str, reason: str) -> None:
        super().__init__(f"effort tier {tier!r} is unavailable: {reason}")
        self.tier = tier
        self.reason = reason


#: The tier names the harness supports, in the order they are presented.
#:
#: This is the WHOLE set, not merely a presentation order over a free mapping:
#: :func:`read_effort_tier_selectors` drops every other key in
#: ``values.subagents.models``, so a hand-added ``xl:`` is inert everywhere
#: (schema, tool-argument validation, launch) rather than honoured by some
#: consumers and not others.
#:
#: Narrowing to the registry set is what keeps ADVERTISED == REBUILDABLE.
#: These are the only keys ``lop config edit`` and the ``/settings`` page can
#: write (``settings_io`` accepts registered keys only), and — the reason this
#: is a correctness boundary rather than a preference — the only ones the
#: config watcher's per-registry-key diff can report. A tier outside the set
#: would be read and advertised, but editing it produces no ``changed_keys``
#: entry, so ``Session._rebuild_effort_tier_tools`` never fires: the schema
#: would promise a tier whose edits the live re-render could not reach until
#: the next session. Widening this tuple therefore means registering the key
#: in ``settings_io.SETTINGS`` in the same change, which is what makes the
#: watcher see it.
#:
#: The order is load-bearing too: the schema rides in the prompt-cache prefix,
#: so it follows this tuple rather than YAML key order, and a reordering edit
#: to ``config.yml`` cannot move the enum.
CANONICAL_EFFORT_TIERS: tuple[str, ...] = ("lo", "med", "hi")


def read_effort_tier_selectors() -> dict[str, Any]:
    """``values.subagents.models``, narrowed to the tiers the harness supports.

    The one place the tier mapping is read from ``config.yml``, shared by the
    strict launch path (``Session._resolve_subagent_model``), the tool-argument
    refusal (:func:`effort_tier_rejection`) and the tool schemas
    (:func:`configured_effort_tiers`), so no two of them can disagree about
    what is configured.

    The narrowing to :data:`CANONICAL_EFFORT_TIERS` happens HERE rather than
    in each consumer, because a key filtered in only one of them is exactly
    the drift this shared reader exists to prevent: a hand-added ``xl`` left
    visible to the launch path but hidden from the schema would be
    unadvertised yet launchable through a role pin, and would draw the
    "lacks provider/model" refusal (which is false — the selector is
    well-formed) instead of the accurate "not configured". One filter, one
    key set, one story. See :data:`CANONICAL_EFFORT_TIERS` for why the
    registry set is the honest boundary.

    That membership test also disposes of keys YAML silently coerced away
    from ``str`` (``1:``, ``on:``, ``yes:`` parse as int/bool): they match no
    canonical name, so they never reach a ``sorted()`` or an enum. Stringifying
    them instead would advertise an ``effort='1'`` no operator ever typed.

    Selector VALUES are stripped here and nowhere else: ``lop config edit``
    preserves surrounding whitespace verbatim, and two consumers stripping
    (or not) independently is how a padded ``'  openai/gpt-5-mini  '`` got
    advertised, passed the strict launch check, and then failed on the first
    provider call with ``provider='  openai'`` instead of at launch. A
    non-string value is passed through raw so the launch path can still turn
    it into a *named* refusal, which it cannot do if the read dropped it.

    Raises whatever the config read raises; callers decide whether that is a
    reason (launch) or nothing (schema).
    """
    from local_operator.config import ConfigManager

    raw = ConfigManager(config_dir()).get_config_value("subagents", None)
    models = raw.get("models") if isinstance(raw, dict) else None
    if not isinstance(models, dict):
        return {}
    return {
        tier: selector.strip() if isinstance(selector, str) else selector
        for tier, selector in models.items()
        if tier in CANONICAL_EFFORT_TIERS
    }


def configured_effort_tiers() -> dict[str, str]:
    """``{tier: "provider/model"}`` for every tier a launch could honour.

    What the ``task`` and ``agent`` tool schemas advertise. Only a tier whose
    selector is a non-empty ``provider/model`` string is included, because
    those are exactly the tiers the strict launch path
    (:class:`SubagentModelUnavailable`) accepts: the incident behind this was
    the schema hard-coding ``lo|med|hi`` while the operator had configured
    NONE, so the delegating model read the enum, picked ``hi``, and the launch
    refused it — the tool's own schema was steering the model into a
    guaranteed failure, and nothing told it that omitting ``effort`` was the
    only working choice.

    Never raises. A config that cannot be read reports no tiers: this runs
    while the tool inventory is being built, and a corrupt ``config.yml`` must
    cost the operator a tier picker, not a session. The launch path reads the
    file again on its own terms and still names the read error there.
    """
    try:
        selectors = read_effort_tier_selectors()
    except Exception:  # noqa: BLE001 — schema construction must never fail a turn
        return {}
    # Iterating CANONICAL_EFFORT_TIERS rather than the config's own keys is
    # what makes this function's "never raises" contract hold. The previous
    # shape sorted the non-canonical leftovers, and that ``sorted()`` sat
    # OUTSIDE the ``try``: one YAML-coerced ``1:`` key beside a ``hi:`` was a
    # ``TypeError`` through ``create_tools`` (the session could not boot) and
    # through ``TaskParams(effort=...)`` as a non-ValidationError. There is no
    # ordering decision left to make here — the canonical tuple IS the order,
    # which also keeps the schema byte-stable in the prompt-cache prefix.
    ordered = [tier for tier in CANONICAL_EFFORT_TIERS if selectors.get(tier)]
    tiers: dict[str, str] = {}
    for tier in ordered:
        selector = selectors[tier]
        if not isinstance(selector, str):
            # A non-string VALUE is kept by the read so the launch path can
            # name it; it is simply not a tier the schema may advertise.
            continue
        provider, _, model_id = selector.partition("/")
        if not provider or not model_id:
            continue
        tiers[tier] = selector
    return tiers


def effort_tier_rejection(tier: str) -> str | None:
    """Why ``tier`` cannot be asked for right now, or ``None`` when it can.

    The tool-argument counterpart of the strict launch check: the ``task``
    and ``agent`` tools validate ``effort`` against the LIVE config with this
    (``subagents.models.*`` is a live setting, read at every spawn), so a tier
    that is not usable is refused with a message that names what IS — before
    a job row, a role pin, or a launch attempt exists. The launch path keeps
    its own refusal for the case this cannot see: a pin recorded while the
    tier existed and read after the operator removed it.

    Always tells the model the working alternative. The failure this guards
    against was a model that could see tiers and not the fact that omitting
    the field was the only choice that worked.
    """
    tiers = configured_effort_tiers()
    if tier in tiers:
        return None
    inherit = "omit 'effort' to inherit this session's model and reasoning effort"
    if not tiers:
        return (
            f"effort tier {tier!r} is unavailable: no tiers are configured under "
            f"values.subagents.models; {inherit}"
        )
    try:
        raw = read_effort_tier_selectors().get(tier)
    except Exception:  # noqa: BLE001 — the tier list above already survived the read
        raw = None
    # Same wording as the launch path's refusal for a selector that is present
    # but unusable, so an operator who set ``lo: gpt-5`` (no provider) learns
    # that the KEY is there and the VALUE is wrong, not that it is missing.
    why = (
        f"subagents.models.{tier}={raw!r} lacks provider/model"
        if raw not in (None, "")
        else f"not configured at subagents.models.{tier}"
    )
    return (
        f"effort tier {tier!r} is unavailable: {why} "
        f"(configured: {describe_effort_tiers(tiers)}); pick one of those or {inherit}"
    )


def describe_effort_tiers(tiers: dict[str, str]) -> str:
    """One short clause naming what each tier resolves to, for a schema
    description: ``lo → openai/gpt-5-mini, hi → anthropic/claude-opus-5``.

    The model chooses on this, so it must carry the MODEL and not just the
    label — "hi" says nothing about cost, family, or capability — while
    staying short, because a schema description is billed on every turn.
    """
    return ", ".join(f"{tier} → {selector}" for tier, selector in tiers.items())


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
#: promise is narrower than "nothing marked write" — it makes no local change
#: at all (browser drives the user's browser; eval executes code; both are
#: excluded by name for that reason even where a tier alone would admit them).
#: It DOES reach the network: retrieval changes nothing, and a research role
#: that cannot search the web is structurally unable to do its job.
#:
#: Kept as the FALLBACK for ``agent="scout"`` when no profile resolves (a
#: stripped install with no packaged seeds, a registry that cannot be read):
#: the read-only promise is a safety property, so it must not depend on a file
#: being present. DERIVED from :data:`~local_operator.agent_profiles.READ_ONLY_TOOLS`
#: rather than spelled out again — two hand-maintained copies of the same
#: allowlist is exactly how the packaged scout seed and this fallback came to
#: disagree about whether a scout has network access.
SCOUT_TOOL_ALLOWLIST = frozenset(READ_ONLY_TOOLS)


def _with_network_floor(
    allowed: "list[AgentTool]", available: "list[AgentTool]"
) -> "list[AgentTool]":
    """Re-admit the read-only network tools an allowlist merely failed to name.

    A role's ``tools`` list is PERSISTED — installing a seed freezes it into a
    ``tools:a,b,c`` registry tag, and ``resolve_profile`` reads the registry
    BEFORE the packaged seeds. So a role installed under an older release
    carries that release's idea of the read-only surface forever, and editing
    the shipped seed files reaches none of it. That is how ``web_search`` and
    ``web_fetch`` — session defaults since long before this floor — stayed
    invisible to every already-installed ``scout``, ``reviewer``, ``architect``
    and ``manager`` on a machine, leaving a research role to report that it had
    no network access and grep the local disk instead.

    The repair is applied HERE, at child construction, and deliberately not by
    rewriting the operator's registry rows: a profile is user data, an agent
    silently editing it would erase a deliberate edit, and a floor computed per
    launch is correct on the next launch after a seed changes rather than only
    after a migration nobody remembers to run.

    Only :data:`~local_operator.agent_profiles.READ_ONLY_NETWORK_TOOLS` is
    floored. Every write and execution denial — no ``edit``, no ``write``, no
    ``bash`` a role lacks — is untouched, which is what keeps a reviewer unable
    to modify the diff it reviews. ``available`` is the pre-filter inventory,
    so a session with web search configured off contributes nothing and the
    floor stays empty; it also cannot admit an MCP tool, since those are minted
    ``mcp__<server>_<tool>`` and can never match these two names.

    THE TRADE-OFF, stated rather than implied (review round 1, R2): a persisted
    tag list cannot distinguish "omits ``web_search`` because it predates the
    tool" from "omits it on purpose", so an operator who deliberately narrowed a
    role to keep it offline gets retrieval back, and ``agent show`` keeps
    printing the narrower list the row actually stores. That is a real change to
    user-data semantics and it is accepted here, not overlooked: the alternative
    leaves every role installed before this release permanently unable to do the
    research it exists for, and what is re-admitted is read-tier retrieval
    behind the ``web_fetch`` SSRF gate. If anyone needs the offline case, the
    fix is an explicit opt-out (a ``network: no`` frontmatter key) rather than
    inferring intent from an omission.
    """
    # Keyed by NAME, and deduped against ``missing`` as well as ``allowed``:
    # matching on a bare name means a pathological inventory holding two tools
    # called ``web_search`` would otherwise contribute both and put a duplicate
    # in the child's schema list (R3). Unreachable through the normal path,
    # cheap to make impossible.
    present = {tool.name for tool in allowed}
    floored: dict[str, AgentTool] = {}
    for tool in available:
        if tool.name in READ_ONLY_NETWORK_TOOLS and tool.name not in present:
            floored.setdefault(tool.name, tool)
    return allowed + list(floored.values())


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
    "search the workspace and the web, and report findings with evidence "
    "(file:line locally, a URL remotely); you cannot edit, write, or run "
    "anything. Your final message is the deliverable.]\n\n"
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
    restricted: bool = False,
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

    ``restricted`` forces the MCP activation denial on regardless of what this
    child's own role says, and exists for the resume path. A denial is
    inherited from the LINEAGE, so a plain ``task`` grandchild of a restricted
    role carries one while its role claims otherwise; ``hub op='resume'``
    rebuilds against the comms-owning root rather than that child's real
    parent, so neither the role nor the parent session can recover the fact and
    it has to be carried forward from the child's record (review round 2, R5).
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
            restricted=restricted,
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
        jobs_manager._notify_roster_change()
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
    restricted: bool = False,
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
                restricted=restricted,
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
                # The new child edge is canonical frontend state too: the
                # accounting invalidation above feeds cost, this publish feeds
                # the roster snapshot every attached full TUI renders.
                jobs_manager._notify_roster_change()
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
            # ``model`` is the child's EFFECTIVE selector, read off the built
            # child exactly as ``job.model_label`` is above. A consumer of the
            # event stream (the Axis runner, a UI) can then state which model
            # a review actually ran on without cross-referencing the job row —
            # the fact that was missing when a pinned reviewer silently ran on
            # the author's model.
            await emit(
                SubagentStartEvent(
                    job_id=job_id,
                    label=label,
                    agent_id=child.agent_id,
                    model=str(getattr(child, "effective_model_label", "") or child.model_label),
                )
            )
            unsubscribe = child.subscribe(
                _make_relay(
                    job_id,
                    label,
                    job,
                    jobs_manager,
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
                raise RuntimeError(_describe_child_failure(str(final["error"]), model_spec))
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


def _describe_child_failure(error: str, model_spec: ModelSpec | None) -> str:
    """The error a failed child settles with, naming the model when that is the point.

    A pinned child (``model_spec`` given) that dies on an auth/availability
    error is not a generic failure: the operator chose that model for this
    child, and the only correct responses are to fix the model's access or to
    consciously run the child elsewhere. Left as the provider's bare text
    (``authentication failed (HTTP 403): ...``), the parent model read it as
    a transient launch problem and retried on another tier \u2014 the observed
    path to a self-review. Naming the pinned model and saying what NOT to do
    is the cheapest intervention that changes that decision.

    Only the auth kind gets the suffix: a pinned child that fails on a
    transient 5xx should be retried on the SAME model, and the suffix would
    argue against exactly that.
    """
    if model_spec is None:
        return error
    from local_operator.providers.failover import is_rendered_auth_error

    # ``final["error"]`` is the loop's RENDERED text, not an exception, so the
    # kind is read the way the display layer reads it (``append_auth_recovery``):
    # by the stable "authentication failed" label the failover module puts in
    # front of every auth-kind error. ``classify_provider_error`` is
    # deliberately not used here — it refuses to read kinds out of text, and
    # this text is the harness's own rendering, which is the one case where
    # the prefix is authoritative.
    if not is_rendered_auth_error(error):
        return error
    pinned = f"{model_spec.provider}/{model_spec.model_id}"
    return (
        f"{error} [pinned model {pinned} is unavailable to this credential. This "
        f"child was pinned to it on purpose; do not re-run it at another effort "
        f"tier, which would silently substitute a different model. Fix access to "
        f"{pinned} or launch without 'effort' and disclose that the child inherits "
        f"the parent's model.]"
    )


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
    await _settle_child_cleanup(asyncio.create_task(child.dispose()))


async def _settle_child_cleanup(dispose_task: asyncio.Task[None]) -> None:
    """Finish child teardown even while the runner itself is being cancelled.

    Shielding alone is insufficient here: it lets teardown continue but returns
    control before descendant cancellation has settled, which makes the caller's
    accounting handoff stale. Keep joining the one dispose task after each outer
    cancellation so teardown remains single-shot and the ledger is final when
    this function returns.
    """
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
    jobs_manager: "AsyncJobManager",
    emit: Callable[[AgentEvent], Awaitable[None]],
    report_progress: Callable[[str], None],
    final: dict[str, Any],
    owner_jobs: Any = None,
) -> Callable[[AgentEvent], Awaitable[None]]:
    """The child-stream handler: trajectory + throttled parent relay.

    EVERY child event lands in the trajectory; only message boundaries, tool
    starts/ends and the FIRST text delta of a message become parent-stream
    progress events — per-delta relaying would flood the parent stream while
    a child streams a long message.

    The progress string is what the child's ROW says it is doing, and it is
    phrased the way the main conversation's working line phrases the parent's
    step (:mod:`local_operator.harness.intent`): the model's own intent while a
    tool runs, ``running N tools`` for a batch, ``responding`` while prose is
    actually streaming, ``thinking`` for a model call in flight with nothing
    streamed yet. It used to read ``tool: bash done`` — the mechanism rather
    than the work, which is the exact narration the intent field exists to
    replace, and a reader watching both surfaces at once should not have to
    learn two vocabularies for one state.

    ``responding`` is keyed to the first ``MessageUpdateEvent`` with text, NOT
    to ``MessageStartEvent``. The loop yields ``message_start`` from a
    placeholder at the top of EVERY provider call, before the request is even
    built (``loop._model_turn``), and a tool-only turn streams tool-call
    deltas that never become a ``MessageUpdateEvent`` at all — so keying on
    ``message_start`` said ``responding`` for the whole of every model call,
    including ones that never produced a word of prose. The main working line
    already keys on the first delta (it mounts its streaming block there), and
    this relay has to agree with it.

    ``running`` is the live tool-call set, kept because the phrase for a batch
    is a COUNT: a relay that only remembered the last event said ``thinking``
    the moment one call of three settled, with two still running.
    """
    running: dict[str, str] = {}
    #: Whether the current assistant message has already reported
    #: ``responding``. One report per message: the transition is the news, and
    #: re-reporting on every delta is the flood the docstring rules out.
    streaming = False
    #: Events relayed by this job so far. Counts RELAYS, not retained entries,
    #: so it keeps rising past the cap and never reissues a number an evicted
    #: event already used — see :data:`TRAJECTORY_SEQ_KEY`.
    relayed = 0

    async def relay(event: AgentEvent) -> None:
        nonlocal relayed, streaming
        if job is not None and job.trajectory is not None:
            record = event.model_dump(mode="json")
            # Stamped BEFORE the append and never revised, because this is the
            # identity the subagent page keys its rows by and the eviction two
            # lines below is precisely what makes list position unusable for
            # that (see TRAJECTORY_SEQ_KEY). Overwritten unconditionally rather
            # than defaulted, so an event that somehow already carries the key
            # cannot inject a duplicate identity into its parent's page.
            record[TRAJECTORY_SEQ_KEY] = relayed
            relayed += 1
            job.trajectory.append(record)
            overflow = len(job.trajectory) - TRAJECTORY_CAP
            if overflow > 0:
                del job.trajectory[:overflow]
        progress: str | None = None
        if isinstance(event, ToolExecutionStartEvent):
            streaming = False
            running[event.tool_call_id] = tool_activity(event.tool_name, event.intent)
            progress = batch_activity(list(running.values()))
        elif isinstance(event, ToolExecutionEndEvent):
            running.pop(event.tool_call_id, None)
            # Back to the model as soon as the batch empties: a settled call is
            # not the child's current activity, and the ledger the page draws
            # already carries its outcome.
            progress = batch_activity(list(running.values())) if running else ACTIVITY_THINKING
        elif isinstance(event, MessageStartEvent):
            # A model call is in flight and nothing has streamed: that is
            # ``thinking``, not ``responding`` — see the docstring for why this
            # event cannot mean prose. The user placeholder the loop yields for
            # a steered prompt is not a model call, so it reports nothing.
            streaming = False
            message = event.message
            if isinstance(message, Message) and message.role == "assistant":
                progress = ACTIVITY_THINKING
        elif isinstance(event, MessageUpdateEvent):
            # Text is actually arriving. Report the transition once per
            # message, and only when no tool is running: a call that is still
            # executing is the child's activity, and prose arriving beside it
            # (a provider that narrates before a batch settles) does not
            # outrank it — the same priority the main working line applies.
            if event.delta and not streaming and not running:
                streaming = True
                progress = ACTIVITY_RESPONDING
        elif isinstance(event, MessageEndEvent):
            streaming = False
            message = event.message
            if isinstance(message, Message) and message.role == "assistant":
                # Capture the last assistant text as the job's result.
                final["text"] = message.text
                _accumulate_usage(job, message.usage)
                note_usage_changed = getattr(owner_jobs, "note_usage_changed", None)
                if callable(note_usage_changed):
                    note_usage_changed()
                jobs_manager._notify_roster_change()
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
                jobs_manager._notify_roster_change()
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
    from local_operator.tui.costs import turn_cost

    # Price detached leaf calls while the owning runtime still has the serving
    # model metadata. A viewer has neither that memo nor necessarily credentials;
    # durable estimates must survive that process boundary without becoming bills.
    components = []
    for item in usage.cost_components or [usage]:
        component = item.model_copy(deep=True)
        provider, _, model_id = (getattr(job, "model_label", None) or "").partition("/")
        component.provider = component.provider or provider or None
        component.model_id = component.model_id or model_id or None
        if component.usd_cost is None and component.estimated_usd_cost is None:
            component.estimated_usd_cost = turn_cost(
                f"{component.provider}/{component.model_id}", component
            )
        components.append(component)
    total = job.usage
    if total is None:
        first = usage.model_copy()
        first.cost_components = components
        # An aggregate receipt is meaningful only when it covers the aggregate.
        # Components retain each call's receipt, so leave the outer field unset
        # and force readers through the provenance-preserving path.
        first.usd_cost = None
        first.estimated_usd_cost = None
        job.usage = first
        return
    total.input_tokens += usage.input_tokens
    total.output_tokens += usage.output_tokens
    total.cache_read_tokens += usage.cache_read_tokens
    total.cache_write_tokens += usage.cache_write_tokens
    # The TTL split of the write count folds exactly where the write count
    # does (they are subsets of it; see ``Usage.cache_write_1h_tokens``), so
    # the job aggregate can price the two rates apart the moment a reader
    # needs to.
    total.cache_write_5m_tokens += usage.cache_write_5m_tokens
    total.cache_write_1h_tokens += usage.cache_write_1h_tokens
    # Child failover can mix provider receipts and table-priced calls. Preserve
    # every original call so the TUI can price each one independently instead of
    # treating one receipt as authoritative for the aggregate token buckets.
    total.cost_components.extend(components)
    total.usd_cost = None
    total.estimated_usd_cost = None
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
    catalogue: Callable[[str], str]
    resolve: Callable[[str], str | None]
    attach: Callable[["Session"], None]


#: Attribute stamped on a child Session recording that IT was built under an
#: MCP activation denial. Read back off ``parent_session`` when that child
#: delegates, which is what makes the denial inherit at any depth of LIVE
#: delegation.
#:
#: It is in-memory Session state and nothing re-derives it, so it does not by
#: itself survive a resume: ``hub op='resume'`` builds a new Session against
#: the comms-owning root rather than the child's real parent.
#:
#: The denial therefore has to be carried at THREE widening scopes, and it took
#: three review rounds because each one held at its own scope while leaking at
#: the next:
#:
#: 1. this attribute — one live lineage, read off ``parent_session`` when a
#:    child delegates (R1: without it, depth 2 escaped);
#: 2. ``_ChildRecord.restricted`` — one process, stamped at ``attach`` and fed
#:    back through ``run_subagent(restricted=...)`` (R5: without it, a resume
#:    escaped);
#: 3. ``snapshot``/``restore`` of that field — across a process exit, since the
#:    roster sidecar is how a child that settled hours ago is resumed at all
#:    (R6: without it, a resume after a restart escaped).
#:
#: ALL THREE are required. Removing any one reopens the escalation on exactly
#: the path the other two do not cover, and the failure is silent — the child
#: comes back merely wider, not broken.
#:
#: A named constant rather than two spelled-out ``getattr``/``setattr`` strings:
#: the reader and the writer are ~200 lines apart, and a typo in either would
#: silently reopen the escalation it exists to close — the failure mode is a
#: quiet loss of a security boundary, not an exception.
MCP_DENIED_ATTR = "_mcp_activation_denied"

#: Rendered in place of a tool schema when a tool-restricted role reads
#: ``mcp://<server>/<tool>``. It names the boundary and what the child still
#: has, because a child told only "no" retries the same URL; a child told the
#: rule reports it to the parent, which is the outcome the delegation wants.
_MCP_ACTIVATION_DENIED = (
    "This role runs on a restricted tool allowlist, so it can use the MCP "
    "tools its parent had already enabled but cannot enable new ones. Use the "
    "tools already in your inventory, or report to your parent (`hub`) that "
    "this tool needs enabling on its side."
)


def _child_mcp_wiring(parent_session: "Session", *, restricted: bool = False) -> _ChildMcp | None:
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

    ``restricted`` is the tool-allowlist case (a reviewer, a scout). Such a
    child used to get NO MCP at all, which cost it the reads its role is made
    of — an MCP server is frequently the only route to the ticket, the design
    doc or the log the research was about — while the write risk it was
    protecting against is not evenly distributed: a server's tools are minted
    ``approval_tier="exec"`` because their side effects are unknowable from
    here, so the harness cannot tell a read tool from a write tool by
    inspection. The line drawn instead is the one the code CAN enforce
    honestly: INHERIT what the parent already enabled (the parent chose those
    tools for this very task and remains accountable for them), and refuse to
    ENABLE anything further, so a restricted role can never widen its own
    surface past its delegator's. Discovery still resolves, because reading the
    catalogue enables nothing.

    That claim only holds because restriction is INHERITED at
    ``_build_child_session`` rather than recomputed per child from its own
    profile. A delegating restricted role would otherwise launder the denial
    through a grandchild: it keeps ``task``, its child rebuilds with no profile
    and so counts as unrestricted, and it activates into this same borrowed
    manager. See the ``restricted`` computation there.

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
    # Deferred discoveries remain callable through the validated fallback
    # path without entering the advertised tool-prefix. A restricted child
    # may inherit its parent's discovered set but cannot expand it.
    deferred: set[tuple[str, str]] = set(getattr(parent_session, "_mcp_deferred_origins", ()))
    child: Session | None = None

    def selected(source: list[AgentTool]) -> list[AgentTool]:
        return [tool for tool in source if origin(tool) in enabled]

    def base() -> list[AgentTool]:
        # Derived from the LIVE inventory on every activation rather than
        # snapshotted at ``attach``: the config watcher swaps fresh
        # ``task``/``agent`` objects into a child mid-run when the effort
        # tiers change, and a frozen base would silently reinstate the
        # pre-rebuild objects (with the stale ``effort`` enum) the first time
        # the child activated an MCP tool. A live tool whose metadata has
        # been dropped from the manager (``origin`` can no longer answer for
        # it) keeps its earlier classification from being rewritten to base
        # here: the cost of a transient misclassification — it stays visible
        # until the next activation — beats disappearing a tool the child
        # was already using, and the top-level path accepts the same
        # trade-off with its ``installed_mcp`` name set.
        if child is None:
            return []
        return [tool for tool in child._tools if origin(tool) is None]

    def activate(server_name: str, raw_tool_name: str) -> None:
        # Unreachable for a restricted child: its resolver is built with
        # ``deny_activation_reason``, which returns before calling this. Kept
        # unguarded so there is ONE activation path rather than a second
        # allow-check that could drift from the resolver's.
        enabled.add((server_name, raw_tool_name))
        if child is not None:
            child.refresh_tools(base() + selected(manager.get_tools()))

    def defer(server_name: str, raw_tool_name: str) -> None:
        deferred.add((server_name, raw_tool_name))

    def attach(session: "Session") -> None:
        nonlocal child
        child = session
        prior = session._fallback_tool_resolver

        def resolve_deferred(name: str) -> AgentTool | None:
            # Resolve fresh after reload/reconnect; retaining AgentTool objects
            # would execute a stale server wrapper after its transport closes.
            for tool in manager.get_tools():
                if tool.name == name and origin(tool) in deferred:
                    return tool
            return prior(name) if prior is not None else None

        setattr(session, "_mcp_deferred_origins", deferred)
        session.set_fallback_tool_resolver(resolve_deferred)

    return _ChildMcp(
        tools=selected(manager.get_tools()),
        catalogue=lambda query: render_mcp_catalogue(manager, query),
        resolve=make_mcp_resolver(
            manager,
            activate,
            deny_activation_reason=_MCP_ACTIVATION_DENIED if restricted else None,
            defer=defer,
        ),
        attach=attach,
    )


def _parent_display_name_resolver(parent_session: "Session") -> Callable[[], str]:
    """A callable returning the parent's DISPLAY name at the moment it is asked.

    The child's browser tab group reads ``<parent conversation> › <job label>``,
    and both halves have to survive nesting. Handing the child the parent's
    title HOLDER only worked one level down: a middle child never generates a
    title of its own (naming runs in the TUI host and the owned-session
    runtime, neither of which a one-shot child passes through), so its holder
    is permanently empty and a grandchild fell back to the cwd that every
    sibling of every conversation shares — two ``qa`` grandchildren under two
    different conversations rendered identically. Delegation really does nest:
    a child of a top-level session keeps ``task``/``wait``/``jobs`` (see the
    depth-aware prune below), which is exactly the manager-fans-out-to-workers
    shape. Resolving through ``_display_session_name`` instead walks the
    lineage to whichever ancestor actually holds a title.

    Called per read rather than snapshotted, because a parent is normally named
    a second or two into its first turn while its children are launched later:
    a string captured here would be "" for the child's whole life.

    WEAK reference on purpose. Every other parent-derived value the child gets
    is a shared collaborator (the comms surface, the variable store, the job
    manager's parent row); this one would be a strong child→parent edge that
    pins the parent's entire object graph — transcript, tools, MCP manager —
    for as long as a detached child outlives it. A dead parent simply has no
    name to lend, and the caller degrades to the cwd form it already handles.
    """
    parent_ref = weakref.ref(parent_session)

    def resolve() -> str:
        parent = parent_ref()
        return parent._display_session_name() if parent is not None else ""

    return resolve


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
    restricted: bool = False,
) -> "Session":
    """Transfer child resource ownership only after construction succeeds.

    The runner cannot dispose a child the builder never returned. Keep every
    acquired resource on a rollback stack until async initialization completes;
    cancellation must join that rollback before the failed launch is observable.
    """
    cleanup = contextlib.AsyncExitStack()
    try:
        child = await _construct_child_session(
            label=label,
            prompt=prompt,
            parent_session=parent_session,
            model_spec=model_spec,
            job_id=job_id,
            resume_dir=resume_dir,
            agent=agent,
            profile=profile,
            restricted=restricted,
            cleanup=cleanup,
        )
    except BaseException:
        await _settle_child_cleanup(asyncio.create_task(cleanup.aclose()))
        raise
    # Normal lifetime now belongs to the returned Session's dispose hooks.
    cleanup.pop_all()
    return child


async def _construct_child_session(
    *,
    label: str,
    prompt: str,
    parent_session: "Session",
    model_spec: ModelSpec | None,
    job_id: str,
    resume_dir: "Path | None",
    agent: str,
    profile: "AgentProfile | None",
    restricted: bool,
    cleanup: contextlib.AsyncExitStack,
) -> "Session":
    """Compose the child Session directly (see module docstring for why the
    factory is not reused, and for the full inherit/do-not-inherit list).

    ``restricted`` forces the MCP activation denial on independently of this
    child's own role and parent. It is how a RESUMED child keeps a denial it
    inherited from a lineage the rebuild cannot see (review round 2, R5); a
    fresh launch leaves it False and the computation below derives the answer.

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
    from local_operator.session.retention import claim_session, release_session

    claim_session(session_dir)
    cleanup.callback(release_session, session_dir)
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
    # Whether this child runs on a role allowlist. Decided HERE, before the
    # ToolContext closes over the resolver, because the MCP surface is built
    # from it: a restricted child gets the read half (inherit + discover) and
    # not the activation half, which cannot be retrofitted by filtering
    # already-minted schemas afterwards. ``agent == "scout"`` is the no-profile
    # fallback path and is restricted for the same reason the allowlist below
    # applies to it.
    #
    # The third term makes the denial STICKY DOWNWARD, and it is load-bearing
    # rather than defensive. A delegating restricted role (the packaged
    # ``manager`` is exactly this: an allowlist AND ``delegate: yes``) keeps
    # ``task`` and, since restricted roles now receive an MCP surface, is handed
    # the parent's live manager below. Computing this from the child's own
    # profile alone left a one-hop escape: the manager's child rebuilt with
    # ``profile=None``, counted as unrestricted, and activated freely into that
    # shared manager — so a manager refused ``delete_issue`` could spawn a plain
    # child and have IT enable the tool, an ``approval_tier="exec"`` write
    # obtained one hop below the boundary that had just refused it. A delegator
    # cannot grant what it does not itself hold, so the denial propagates to
    # every descendant regardless of their own profiles.
    # The fourth term is the RESUME carry (see the parameter's note): a resumed
    # child is rebuilt against the comms-owning root rather than its real
    # parent, so the third term reads an unrestricted session and only the
    # persisted record can supply the fact. OR-ed, never assigned, so a resume
    # can only ever preserve a denial and never clear one the live computation
    # would have found.
    restricted = (
        restricted
        or (profile is not None and bool(profile.tools))
        or agent == "scout"
        or bool(getattr(parent_session, MCP_DENIED_ATTR, False))
    )
    mcp = _child_mcp_wiring(parent_session, restricted=restricted)
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
        # The parent's resolver also ends in its OWN MCP resolver, which
        # activates into the PARENT's inventory. Falling through to it for any
        # ``mcp://`` URL the child's resolver did not answer would therefore
        # enable a tool on the wrong session — and for a restricted child it
        # would additionally route around the activation denial above. Reject
        # only this namespace here so guide:// and skill:// stay inherited.
        if url.startswith("mcp://"):
            return None
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
        # The child's own name for display surfaces that must not render a
        # fleet of children identically (browser tab groups today). Set on the
        # CONSTRUCTION context for the same defensive-parity reason
        # ``variables`` is: what actually reaches an executing tool is the
        # child ``Session``'s per-turn rebuild, which receives it via the
        # ``job_label=`` argument below.
        job_label=label,
        has_ui=parent_session._has_ui,
        request_approval=request_approval,
        # The parent's variable store. DEFENSIVE PARITY, not a bug fix: no
        # current ``TOOL_BUILDERS`` entry reads ``context.variables`` at
        # construction time (the variables readers are all execute-time, and
        # the child ``Session`` below receives ``variables=`` directly, which
        # is what ``_build_tool_context`` re-derives on every turn). Kept so
        # this construction context matches the session factory's shape
        # (session_factory.py builds one store and hands it to both contexts)
        # and any future createIf gate that does read it sees the same store
        # the executing tools will.
        variables=getattr(parent_session, "_variables", None),
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
    # review and thereby end up reviewing its own patch. ``restricted`` itself
    # was decided above, because the MCP surface is built from it.
    #
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
    if profile is not None and profile.tools:
        tools = _with_network_floor(filter_tools(tools, profile), tools)
    elif agent == "scout":
        # Fallback for a scout with no resolvable profile — the read-only
        # promise must not depend on a seed file being present.
        tools = [tool for tool in tools if tool.name in SCOUT_TOOL_ALLOWLIST]
    if restricted and hub_tool is not None and not any(tool.name == "hub" for tool in tools):
        tools = list(tools) + [hub_tool]
    # A restricted role receives the MCP tools its PARENT had already enabled
    # (see :func:`_child_mcp_wiring` for why inheriting is the honest line and
    # activation is not): withholding them cost a reviewer or scout the reads
    # its role is made of, while the parent that chose those tools for this
    # task remains accountable for them. It cannot widen the set — its
    # resolver refuses to activate anything new.
    if mcp is not None:
        tools = tools + mcp.tools

    parent_provider = getattr(parent_session, "_system_blocks_provider", None)
    repo_guidance = getattr(parent_provider, "repo_guidance", "")
    parent_hooks = getattr(parent_provider, "knowledge_hooks", None)
    # A bounded directory of already-selected knowledge costs far less than
    # rediscovering the same guides in every child. These are names/links and
    # descriptions, not the parent's conversation or full skill documents.
    knowledge = getattr(parent_hooks, "frozen_block", "") or ""
    if len(knowledge) > 12000:
        knowledge = knowledge[:12000].rsplit("\n", 1)[0]

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
            "\n\n".join(
                filter(None, (knowledge, mcp.catalogue(prompt) if mcp is not None else ""))
            ),
            _env_details(cwd),
            datetime.now().strftime("%Y-%m-%d"),
            goal=parent_session.goal,
            user_instructions=user_instructions,
            repo_guidance=repo_guidance,
            credentials=names,
            model_label=model_label,
        )

    setattr(system_blocks_provider, "append_only_state", True)
    setattr(system_blocks_provider, "repo_guidance", repo_guidance)
    setattr(system_blocks_provider, "knowledge_hooks", parent_hooks)
    parent_stream = parent_session._stream_fn
    fork_stream = getattr(parent_stream, "fork", None)
    # Transport pooling is shared infrastructure; routing, callbacks, effort,
    # usage attribution and cache identity belong to this conversation.
    child_stream: Any = (
        fork_stream(transcript.directory.name) if callable(fork_stream) else parent_stream
    )
    if child_stream is not parent_stream:
        # Register before Session.__init__, which can itself fail. close() is
        # idempotent: this fallback also runs if a later child dispose hook fails.
        cleanup.push_async_callback(child_stream.close)
    child = Session(
        model=model_spec if model_spec is not None else parent_session.model,
        stream_fn=child_stream,
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
        # The label the operator launched this child under, and a resolver for
        # the PARENT's display name. Together they are the only identity a
        # subagent has: naming runs in the TUI host and the owned-session
        # runtime, so a one-shot child never generates a title of its own and
        # every display surface asking "which session is this?" had nothing to
        # answer with.
        #
        # Display-only on both counts: a child is authorized by its own
        # ``session_id``, never by a name it borrowed from its parent.
        job_label=label,
        parent_display_name=_parent_display_name_resolver(parent_session),
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
    cleanup.push_async_callback(child.dispose)
    if child_stream is not parent_stream:
        child.add_dispose_hook(child_stream.close)
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
    # Record the denial on the child so its OWN children inherit it (see the
    # ``restricted`` computation above). Set UNCONDITIONALLY, outside the
    # ``mcp is not None`` branch below: a child built with no MCP surface --
    # because this session had no manager wired yet -- can still delegate, and
    # its child resolves the manager off the session at that later point. Making
    # the stamp depend on whether MCP happened to be wired here would let the
    # boundary evaporate on exactly the path that reintroduces the surface.
    setattr(child, MCP_DENIED_ATTR, restricted)
    if mcp is not None:
        mcp.attach(child)
        # Diagnostics only, and BORROWED: unlike attach_mcp_dispose this adds no
        # disconnect hook, because the child does not own the servers.
        child.mcp_manager = parent_session.mcp_manager
        child.mcp_startup = parent_session.mcp_startup
    # Follow ``config.yml`` like the parent does (``session_factory
    # .attach_config_watch``). The ``model_copy`` of the parent's compaction
    # settings above is the correct INITIAL value; this subscription is what
    # keeps it current, so a threshold lowered while a long review child runs
    # bounds the child too. Same process and loop as the parent, so no extra
    # poller and no extra wake — one more listener on the process watcher.
    # Unsubscribed by ``_dispose_child`` through the dispose hook, exactly as
    # the parent's is. Degrades silently: a child that cannot follow config is
    # a child built the way every child was before this seam.
    try:
        from local_operator.config_watch import process_watcher

        watcher = process_watcher(config_dir())
        watcher.start(asyncio.get_running_loop())
        child.add_dispose_hook(watcher.subscribe(child._apply_config_change))
    except Exception:  # noqa: BLE001 — a child must build without the watcher
        logger.warning("config watcher could not be attached to the child", exc_info=True)
    await child.async_init()
    return child
