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
  parent-transcript write, so the front end has exactly one delivery path —
  with ONE deliberate exception, which is a state notice rather than a
  completion: a PINNED child whose route settles onto a fallback emits a
  ``NoticeEvent`` on the parent stream (see :func:`_pinned_fallback_notice`),
  because a silently substituted review is the failure the pin exists to
  prevent and it has to be loud on the surface the operator reads.

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
session-capability tools ``task``/``wait``/``jobs``/``wake`` are scoped below,
and the rule is WHO MAY DELEGATE, not how deep this child sits (operator,
2026-09-18): a subagent that is ALLOWED ``task`` must delegate with it and one
that was not given it may not create subagents at all. An allowance therefore
follows the role down the lineage rather than stopping at one level — a
``delegate: yes`` manager keeps ``task`` at any depth, so a manager's child is
a manager too, and the tree it grows is walked one page at a time by the TUI
and the desktop UI, both of which read the comms tree recursively. A
role-less child inherits its parent's allowance. What no child ever keeps is
``wake``: a child session ends after one prompt, so a wake armed there would
be silently lost. The scoping used to be implicit in the child's ToolContext
carrying no launcher; that stopped holding when ``Session.__init__`` grew
``_merge_capability_tools``, which re-derives those four from the session's
OWN context and so handed every child a ``task`` tool, so it is applied
explicitly in :func:`_build_child` now.

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

``jobs`` and ``wait`` are a SECOND, CONDITIONAL exception, on a different
principle. They observe, cancel and block on the child's OWN background jobs —
they spawn nothing (that is ``task``) and die with the child's job manager —
so they cross no boundary the prune protects. But a child can only produce a
background job while its ``bash`` retains ``background``, and the bash receipt
tells the model to poll such a job with ``jobs(op='peek')``. So the invariant
is: a child keeps ``jobs`` and ``wait`` IFF it can still background a bash
command. The prune below re-adds them exactly under that condition
(:func:`_can_background`), which is what stops a non-delegating role or
grandchild that backgrounds a long command from looping forever on ``Tool not
found: jobs`` — and, for ``wait``, from blocking on it with a foreground
``sleep`` that no hub note can interrupt.

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
import time
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
    NoticeEvent,
    ReasoningDeltaEvent,
    SubagentEndEvent,
    SubagentProgressEvent,
    SubagentStartEvent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    Usage,
)
from local_operator.mcp.config import server_own_turn_only
from local_operator.model.naming import model_label as model_label_forms
from local_operator.paths import config_dir
from local_operator.resume import ORIGIN_SUBAGENT, mark_session_origin
from local_operator.session import subagent_ledger as ledger


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

#: The one tier VALUE that means "this tier is CONFIGURED, and it runs on
#: whatever model the launching session is on": ``subagents.models.hi: default``.
#:
#: Why a sentinel is needed at all, given the tier already had two spellings
#: for "use the session's model": both of them were the ABSENCE of the tier.
#: An absent key and an empty string are the same state to every consumer —
#: :func:`read_effort_tier_selectors` keeps only a non-empty selector, so
#: :func:`configured_effort_tiers` drops the tier and unadvertises it, the
#: strict launch path (:class:`SubagentModelUnavailable`) REFUSES it, and a
#: role pinned to that tier dies at launch rather than inheriting. So an
#: operator could not say "hi means the session model" in a way anything would
#: honour, assert, or show them: writing it looked exactly like deleting it.
#: This sentinel is that third state, and it is the only one of the three that
#: is a choice the operator made — which is what keeps PR #635's loud-failure
#: rule intact. The other two still refuse a PINNED role, deliberately: the
#: whole point of that rule is that an unconfigured tier must not silently
#: collapse into self-review. This is an explicit opt-in to one model (the
#: session's own), never an implicit fallback to whatever was left around.
#:
#: The SPELLING is ``default`` and not ``inherit`` on purpose. ``inherit`` is
#: already taken in this codebase for the OPPOSITE operation: it is
#: :data:`~local_operator.tools.builtin.INHERIT_EFFORT`, the value a delegating
#: MODEL passes to mean "no tier" / "clear this role's pin". A config VALUE
#: spelled the same would read as its own negation — the same word meaning
#: "unset the tier" on one surface and "this tier is set" on the other — and
#: an operator copying the token they saw in an ``agent`` tool result onto the
#: config row would get the exact opposite of what they asked for. ``default``
#: also matches the vocabulary the rest of the config already uses for "the
#: model this session would otherwise run on" (``/model default``).
#:
#: The value stays a plain string under the existing key, so the config SHAPE
#: and the live-rebuild prefix match (``Session._apply_config_change``, which
#: fires on any ``subagents.models.`` key) are unchanged.
INHERIT_TIER_SENTINEL = "default"


def is_inherit_tier_sentinel(selector: object) -> bool:
    """Is ``selector`` the inherit sentinel, however it was typed?

    Case-insensitive and whitespace-tolerant, and BOTH tolerances are safe for
    one reason: every real selector is a ``provider/model`` string and so
    contains a ``/``, which no casing of ``default`` has. So there is no
    ``provider/model`` this can misread as the sentinel, and no ``DEFAULT``
    that can be a model.

    Why it exists at all: :func:`read_model_choice` — the neighbouring
    operator-facing reader on the same config block — already accepts
    ``"MODEL"`` and ``" model "``, so matching the sentinel exactly made the
    two sentinel-ish surfaces disagree, and a single typo'd tier (``hi:
    DEFAULT``) fell out of the advertised set and drew "no tiers are
    configured", which reads as "you configured nothing" to an operator who
    did write a value. The failure direction was safe (it refuses, it never
    silently inherits) but the reason was wrong, and a wrong reason is what
    this whole change is about.

    A non-string is not the sentinel: the reader keeps such values precisely so
    the launch path can name them (``lo: 0`` must draw "lacks provider/model",
    not "not configured").
    """
    return isinstance(selector, str) and selector.strip().lower() == INHERIT_TIER_SENTINEL


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


#: Whose choice a subagent's model is, as ``values.subagents.model_choice``
#: spells it. ``operator`` is the shipped default: a child inherits the
#: launching session's model unless an OPERATOR-authored role pin or launcher
#: argument moves it, which is the contract ``run_subagent``'s module docstring
#: states. ``model`` additionally lets a DELEGATING MODEL pick a configured
#: ``subagents.models`` tier from the ``task`` tool's schema.
#:
#: Why this key exists, since the default is the behaviour the harness already
#: had: the ``task`` schema advertised an ``effort`` enum whose members are
#: provider/model SWAPS while its name is this harness's REASONING-effort
#: vocabulary, so a delegating model read "pick your child's effort" and took a
#: provider/model swap instead. Measured over the last 500 session transcripts
#: of this machine: a ``deepseek/deepseek-v4.1-flash`` parent pinned ``hi`` 157
#: times and ``med`` 46 times against 365 inherits, and one
#: ``~/.local-operator/sessions/61f378649fbc`` run spent $0.083 on parent calls
#: against $13.05 across its three opus children.
#:
#: The tiers themselves are the operator's own legitimate configuration, and
#: this key does not take them away: every route an operator has into a tier
#: keeps working — its own ``/settings`` row, a role's pin, an explicit launcher
#: argument — and flipping this key to ``model`` hands the reflex back. What the
#: default changes is only WHO may take one without asking, because a choice
#: made reflexively by each delegating model is not a decision the operator made
#: and, until this shipped, nothing in the loop said a child had moved onto a
#: different model until the bill arrived.
MODEL_CHOICE_OPERATOR = "operator"
MODEL_CHOICE_MODEL = "model"

#: The default :func:`read_model_choice` falls back to for every unrecognised
#: shape, and the value the ``subagents.model_choice`` registry row ships with.
#: Named once so the reader, the registry and the tests cannot disagree about
#: which member is the safe one.
DEFAULT_MODEL_CHOICE = MODEL_CHOICE_OPERATOR

#: Values :func:`read_model_choice` has already warned about, so a config file
#: holding ``modle`` warns once per distinct value per process rather than on
#: every tool build, every spawn and every schema read.
_WARNED_MODEL_CHOICES: set[str] = set()


def _warn_unrecognised_model_choice(shown: str) -> None:
    """Warn once per distinct unrecognised value. See :func:`read_model_choice`."""
    if shown in _WARNED_MODEL_CHOICES:
        return
    _WARNED_MODEL_CHOICES.add(shown)
    logger.warning(
        "subagents.model_choice=%s is neither %r nor %r; treating it as %r",
        shown,
        MODEL_CHOICE_MODEL,
        MODEL_CHOICE_OPERATOR,
        DEFAULT_MODEL_CHOICE,
    )


def read_model_choice() -> str:
    """``values.subagents.model_choice``, coerced to the two values it may take.

    The one place the key is read, shared by the tool schemas
    (:func:`model_may_choose_tier`) and by the tool-argument refusal, so the
    surface that advertises the field and the surface that validates it cannot
    disagree about who owns the choice.

    **Fails CLOSED, and that direction is the whole point.** Only a ``str``
    whose ``strip().lower()`` is exactly ``"model"`` enables model choice;
    absent, blank, wrongly typed (YAML's ``true``/``1``, a list) and misspelled
    all mean :data:`DEFAULT_MODEL_CHOICE`. A reader that failed OPEN on an
    unrecognised value would re-authorise exactly the spend this key was added
    to stop, on nothing worse than a typo — silently, in a way that looks like
    the key working. A value that is PRESENT but unrecognised still warns once,
    because gating an operator who typed ``modle`` with no clue why is the other
    way to be wrong.

    Never raises, deliberately: this is read while the tool schemas are being
    built, and a corrupt ``config.yml`` must cost the operator a delegating
    model's tier picker — which is the fail-closed answer anyway — rather than a
    session. :func:`configured_effort_tiers` takes the same position on the
    same read path.
    """
    from local_operator.config import ConfigManager

    try:
        raw = ConfigManager(config_dir()).get_config_value("subagents", None)
    except Exception as exc:  # noqa: BLE001 — schema construction must never fail a turn
        _warn_unrecognised_model_choice(f"<unreadable: {exc}>")
        return DEFAULT_MODEL_CHOICE
    choice = raw.get("model_choice") if isinstance(raw, dict) else None
    if isinstance(choice, str):
        normalised = choice.strip().lower()
        if normalised in (MODEL_CHOICE_MODEL, MODEL_CHOICE_OPERATOR):
            return normalised
        _warn_unrecognised_model_choice(choice)
        return DEFAULT_MODEL_CHOICE
    if choice is not None:
        # A non-string is a shape YAML produced (``true``, ``1``, ``[]``), so the
        # warning shows it as the repr an operator can find in the file.
        _warn_unrecognised_model_choice(repr(choice))
    return DEFAULT_MODEL_CHOICE


def model_may_choose_tier(delegation_depth: int = 0) -> bool:
    """May a DELEGATING MODEL pick the model a child runs on?

    The policy half of the ``task``/``agent`` schemas: when this is ``False``
    (the default) no tier is advertised on any model-facing surface, and asking
    for one anyway is refused at the tool-argument boundary with a message that
    says what to do instead. See :func:`read_model_choice` for why the default
    is the restrictive one.

    ``delegation_depth`` is the hops between the asking session and the
    top-level one (``Session._delegation_depth``): **a session that is itself a
    subagent (depth >= 1) never may, whatever ``subagents.model_choice`` says**.
    The key is the OPERATOR delegating the choice to the model they are talking
    to, and that grant stops at the first hop. Measured incident (session
    3463fc25dade): ``model_choice=model`` with ``models.hi`` on Claude Sonnet,
    session model Radient auto. A depth-1 subagent launched five ``scout``
    children with ``effort='hi'``; they ran on Sonnet (``owns_model=True``)
    instead of inheriting Radient auto — roughly $5 of Sonnet spend against
    $0.76 on auto, invisible in the panel rows, which show only the depth-1
    child's own usage. The operator had agreed to the model they were driving
    choosing tiers, not to a fan-out of delegated models re-deciding the bill
    one level down, where nothing reviews the choice. So below the top the
    policy is exactly ``operator``: children inherit the launching session's
    model, and only an operator-authored ROLE pin (``profile.effort``, resolved
    at launch by ``Session._resolve_subagent_model``) can still move a nested
    child — that is the operator's own decision, not the model's, and it is
    deliberately untouched here.

    The default of ``0`` keeps every existing caller (and every top-level
    session) on the pure config read.
    """
    if delegation_depth >= 1:
        return False
    return read_model_choice() == MODEL_CHOICE_MODEL


def depth_closed_the_tier_choice(delegation_depth: int) -> bool:
    """Is the picker closed BY DEPTH, i.e. would the key alone have left it open?

    True only for a subagent under ``model_choice=model``. Under ``operator``
    the key already closes the picker for everyone, so a nested session is
    refused for the same reason the top level is, and the copy it is shown must
    say so: telling it "only the top-level session may pick a tier" would be
    false there (nobody on the model side can), and would drop the operator's
    route (``subagents.model_choice``) that the operator-arm copy carries. The
    nested wording exists for exactly the case where the key says "model" and a
    reader would otherwise be told the picker is open to it.
    """
    return delegation_depth >= 1 and read_model_choice() == MODEL_CHOICE_MODEL


def configured_effort_tiers() -> dict[str, str]:
    """``{tier: selector}`` for every tier a launch could honour.

    What the ``task`` and ``agent`` tool schemas advertise. A tier is included
    when its selector is a non-empty ``provider/model`` string OR the
    :data:`INHERIT_TIER_SENTINEL`, because those are exactly the tiers the
    strict launch path (:class:`SubagentModelUnavailable`) accepts: the
    incident behind this was
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
        if is_inherit_tier_sentinel(selector):
            # Advertised by its sentinel, NOT by the model it currently
            # resolves to: the value is read at every build and every spawn,
            # so a session that switches model mid-flight must not be shown a
            # tier whose advertised model is the one it left. The schema
            # description names the resolved model BESIDE the sentinel (see
            # :func:`describe_effort_tiers`), which is the honest form of the
            # same fact: the sentinel is what is stored, and the label says
            # what it runs right now.
            #
            # Stored CANONICAL (not as typed) so every downstream exact
            # comparison -- the schema description, the launch path's own
            # check -- sees one spelling for one state. The raw value is kept
            # by the shared reader for the "lacks provider/model" refusal;
            # this is the advertised view, and it has one spelling.
            tiers[tier] = INHERIT_TIER_SENTINEL
            continue
        provider, _, model_id = selector.partition("/")
        if not provider or not model_id:
            continue
        tiers[tier] = selector
    return tiers


def effort_tier_rejection(tier: str, *, session_model_label: str | None = None) -> str | None:
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
            f"subagents.models; {inherit}"
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
        f"(configured: {describe_effort_tiers(tiers, session_model_label=session_model_label)}); "
        f"pick one of those or {inherit}"
    )


def describe_effort_tiers(tiers: dict[str, str], *, session_model_label: str | None = None) -> str:
    """One short clause naming what each tier resolves to, for a schema
    description: ``lo → openai/gpt-5-mini, hi → anthropic/claude-opus-5``.

    The model chooses on this, so it must carry the MODEL and not just the
    label — "hi" says nothing about cost, family, or capability — while
    staying short, because a schema description is billed on every turn.

    A tier on the :data:`INHERIT_TIER_SENTINEL` is the one case where the
    stored value is NOT a model, so naming the stored value alone would
    advertise a tier without saying what it runs — the exact gap PR #635 was
    written to close. It is therefore rendered as
    ``hi → default (session model: anthropic/claude-sonnet-5-5)`` when the
    caller knows the session's model, and ``hi → default (this session's
    model)`` when it does not. Both forms keep the sentinel visible (that is
    what a reader copies back into the config) and add what it resolves to on
    this session; the bare word ``default`` never stands alone in a schema.

    ``session_model_label`` is optional because the schema builders have a
    ``ToolContext`` (which carries the label the session itself paints) while
    pure callers — a launch refusal, a test — may not. Absence degrades to
    the generic phrase, never to silence about the sentinel.

    Sentinels are grouped into ONE clause rather than repeated per tier: a
    session with all three tiers on the sentinel otherwise spends 176 cells of
    a description that is billed on every turn naming the same model three
    times. Grouping cannot hide a tier (every tier name is still listed) and it
    cannot drop the model (the group carries the one label they all resolve
    to), which is the constraint the per-tier form was introduced to satisfy.

    Which shapes are byte-identical to the previous form, exactly: the
    no-sentinel shape, and a single sentinel tier that is ALREADY LAST. The
    sentinel clause is emitted last, so a `{lo: default, hi: a/b}` config lists
    `hi` first — the order moves, the text does not otherwise change (review
    round 2, NIT 1; the earlier note claimed byte-identity for "the common
    single sentinel tier", which is looser than what holds). That is still
    enough for prompt-cache stability: a pre-delta config could not advertise a
    sentinel tier at all — the reader dropped it — so the only configs whose
    text can move are new-state ones this feature created.
    """
    parts: list[str] = []
    sentinel_tiers: list[str] = []
    for tier, selector in tiers.items():
        if is_inherit_tier_sentinel(selector):
            sentinel_tiers.append(tier)
            continue
        parts.append(f"{tier} → {selector}")
    if sentinel_tiers:
        names = ", ".join(sentinel_tiers)
        if session_model_label:
            parts.append(
                f"{names} → {INHERIT_TIER_SENTINEL} (session model: {session_model_label})"
            )
        else:
            parts.append(f"{names} → {INHERIT_TIER_SENTINEL} (this session's model)")
    return ", ".join(parts)


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

#: How long a child's streamed text may sit in the relay before it is written
#: to the trajectory as one coalesced ``message_update`` row (see
#: ``_make_relay.flush_text``). 250 ms keeps the subagent page visibly
#: streaming (4 Hz) while writing ~1/50th of the rows a per-token relay did;
#: any non-text event flushes immediately, so the bound only ever delays text.
SUBAGENT_TEXT_FLUSH_S = 0.25


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


class TeamLaunchError(Exception):
    """A launch named a team it cannot run (BEN-7-D2/D3).

    Raised BEFORE a job is registered, so ``execute_task``'s existing
    launcher-failure path reports it by name and no child exists. The
    alternative it replaces was a silent generic child told "you are
    team:pod" (BEN-1 N3), which looked like delegation and was not.
    """


#: Default ceiling on how deep a TEAM-bearing tree may launch (BEN-7-D3).
#: Depth counts hops below the top session: the top is 0, its ``task``
#: children are 1. Measured basis: 1,505 of 1,546 child sessions were depth 1,
#: 33 depth 2, and the only depth-3/4 chain was a probe. Only trees with a team
#: somewhere are capped; a team-less tree keeps today's unbounded behaviour.
DEFAULT_MAX_TEAM_DEPTH = 3

#: The ``task`` ``agent`` prefix that starts another team's manager.
TEAM_LAUNCH_PREFIX = "team:"

#: Where a top session's launches report. A depth-1 child has no parent job.
TOP_SESSION_REPORTS_TO = "the operator's top session"


def read_max_team_depth() -> int:
    """``subagents.max_team_depth``, clamped to ``1..MAX_ORG_DEPTH``.

    Read at launch, like the tier selectors, so an edit applies live. Never
    raises: a corrupt config must cost the operator the configured value, not
    the delegation, so anything unusable means the default.
    """
    from local_operator.config import ConfigManager
    from local_operator.teams import MAX_ORG_DEPTH

    try:
        raw = ConfigManager(config_dir()).get_config_value("subagents", None)
        value = raw.get("max_team_depth") if isinstance(raw, dict) else None
        if value is None or isinstance(value, bool):
            return DEFAULT_MAX_TEAM_DEPTH
        return max(1, min(MAX_ORG_DEPTH, int(value)))
    except Exception:  # noqa: BLE001 — a bad value must not fail a launch
        logger.warning("subagents.max_team_depth is unreadable; using the default")
        return DEFAULT_MAX_TEAM_DEPTH


@dataclass(frozen=True)
class LaunchTarget:
    """What one launch resolves to: the single contract from the ``task`` tool
    through ``run_subagent`` and ``_build_child_session`` to ``comms.resume``
    (BEN-7-D1). Every later launch feature goes through it."""

    #: The role the child RUNS as (a ``team:`` launch runs as the sub-team's
    #: manager), used for the profile, the preamble and the model tier.
    role: str
    #: The team whose text the child carries, or None.
    team: Any
    #: Team ids from the top down. Stamped rather than derived from comms
    #: ancestry, because the registry evicts settled records and an evicted
    #: ancestor would undercount depth and fail the cap open (D1 (b)).
    team_lineage: tuple[str, ...]
    depth: int
    reports_to: str
    is_team_launch: bool


@dataclass(frozen=True)
class CarriedLaunch:
    """What a resume carries forward from the child's record (BEN-7-D1).

    ``comms.resume`` rebuilds against a session that may not be the child's
    real parent, so the team, lineage and depth the child was born with cannot
    be re-derived there and ride the record instead, like ``restricted``.
    """

    team_name: str
    team_lineage: tuple[str, ...]
    depth: int
    reports_to: str


def _session_lineage(session: "Session") -> tuple[str, ...]:
    """A session's team lineage. A child carries a stamp; the top session's is
    its attached team alone, or nothing."""
    stamped = getattr(session, "_team_lineage", None)
    if stamped is not None:
        return tuple(stamped)
    team = getattr(session, "active_team", None)
    team_id = getattr(team, "id", None) if team is not None else None
    return (str(team_id),) if team_id else ()


def _session_depth(session: "Session") -> int:
    depth = getattr(session, "_delegation_depth", 0)
    # ``type(...) is int``: a bool is an int, and ``True`` would be depth 1. The
    # tool-side readers (``effort_validation_context``/``_delegation_depth``)
    # apply the same rule, so every reader of a depth agrees on what one is.
    return depth if type(depth) is int and depth >= 0 else 0


def lookup_team(session: "Session", name: str) -> Any:
    """The registered team called ``name`` (the registry casefolds), or None.

    None also covers "no registry" and an unreadable one: the caller turns it
    into a named ``unknown team`` error rather than a generic child.
    """
    registry = getattr(session, "team_registry", None)
    if registry is None or not hasattr(registry, "get_team_by_name"):
        return None
    try:
        return registry.get_team_by_name(name)
    except Exception:  # noqa: BLE001 — the caller turns None into a named error
        logger.warning("could not look up team %r", name, exc_info=True)
        return None


def describe_reports_to(team_name: str, role: str, job_id: str | None) -> str:
    """``"<team> <role> (job <id>)"`` for a parent, or the top session (D2).

    ``role`` is the role the parent RUNS as: a ``team:pod`` parent is pod's
    manager, which is the name a child should report to.
    """
    if not job_id:
        return TOP_SESSION_REPORTS_TO
    head = f"{team_name} {role or 'task'}" if team_name else (role or "task")
    return f"{head} (job {job_id})"


def _parent_reports_to(parent_session: "Session") -> str:
    job_id = getattr(parent_session, "_job_id", None)
    if not job_id:
        return TOP_SESSION_REPORTS_TO
    role = "task"
    team = getattr(parent_session, "active_team", None)
    comms = getattr(parent_session, "subagent_comms", None)
    node = comms.node(job_id) if comms is not None and hasattr(comms, "node") else None
    if node is not None and getattr(node, "agent_role", ""):
        role = str(node.agent_role)
    if role.lower().startswith(TEAM_LAUNCH_PREFIX) and team is not None:
        role = str(getattr(team, "manager", role))
    team_name = str(getattr(team, "name", "") or "") if team is not None else ""
    return describe_reports_to(team_name, role, job_id)


def _team_slot_kinds(team: Any, name: str) -> set[str]:
    kinds: set[str] = set()
    for member in getattr(team, "members", None) or ():
        if str(getattr(member, "role", "")) == name:
            kinds.add(str(getattr(member, "kind", "agent")))
    return kinds


def resolve_launch_target(
    agent: str,
    parent_session: "Session",
    *,
    carried: CarriedLaunch | None = None,
) -> LaunchTarget:
    """Resolve one ``task`` launch to the team and depth its child runs under.

    Raises :class:`TeamLaunchError` for an unknown team, a counted launch, a
    manager that cannot delegate, a cycle, or a fresh launch past the depth
    cap (a resume keeps its recorded depth) — never a silent generic child
    (BEN-1 N3).
    """
    agent = agent or "task"
    parent_team = getattr(parent_session, "active_team", None)
    sub_name = ""
    if agent.lower().startswith(TEAM_LAUNCH_PREFIX):
        sub_name = agent[len(TEAM_LAUNCH_PREFIX) :].strip()
        if not sub_name:
            raise TeamLaunchError(f"invalid team launch {agent!r}: no team name")
        if ":" in sub_name:
            raise TeamLaunchError(
                f"{agent!r} names a count; launch one "
                f"'{TEAM_LAUNCH_PREFIX}{sub_name.split(':', 1)[0]}' per copy"
            )
    elif (
        carried is None
        and parent_team is not None
        and _team_slot_kinds(parent_team, agent) == {"team"}
    ):
        # (Not on resume: a bare-name team launch was recorded as
        # ``team:<name>``, so a carried bare name is always an agent launch.)
        # Bare name: a team launch ONLY when the roster's slot of that name is
        # a team and no agent slot shares it; a clash stays an agent launch,
        # exactly as today, so depth-1 meaning cannot shift (D2 (d)).
        sub_name = agent

    if carried is not None:
        # A resume re-enters the team the child was BORN under, at the depth
        # its record carries. Checks that guarded its original launch are not
        # re-run: it already exists. That includes the depth cap — a child
        # recorded at depth N resumes at N even after the operator lowers
        # ``subagents.max_team_depth`` below it, and only its FURTHER launches
        # are capped (the ``if lineage:`` check below). Refusing the resume
        # instead would strand work that already exists; a resume is recovery,
        # not a new launch.
        team = None
        if carried.team_name:
            if parent_team is not None and getattr(parent_team, "name", "") == carried.team_name:
                team = parent_team
            else:
                team = lookup_team(parent_session, carried.team_name)
            if team is None:
                raise TeamLaunchError(
                    f"team {carried.team_name!r} this subagent ran under no longer exists"
                )
        role = str(team.manager) if sub_name and team is not None else agent
        return LaunchTarget(
            role=role,
            team=team,
            team_lineage=tuple(carried.team_lineage),
            depth=carried.depth,
            reports_to=carried.reports_to or TOP_SESSION_REPORTS_TO,
            is_team_launch=bool(sub_name),
        )

    depth = _session_depth(parent_session) + 1
    parent_lineage = _session_lineage(parent_session)
    if sub_name:
        sub = lookup_team(parent_session, sub_name)
        if sub is None:
            raise TeamLaunchError(f"unknown team {sub_name!r}")
        role = str(sub.manager)
        if role == "scout":
            raise TeamLaunchError(f"team {sub.name!r} manager {role!r} cannot delegate")
        profile = _resolve_role(role, parent_session)
        if profile is not None and not profile.may_delegate:
            # Never force ``task`` onto a non-delegating role to make the
            # launch work (frozen rule; D2 (b)).
            raise TeamLaunchError(f"team {sub.name!r} manager {role!r} cannot delegate")
        if sub.id in parent_lineage:
            names = _lineage_names(parent_session, parent_lineage)
            raise TeamLaunchError(
                f"cycle: team {sub.name!r} is already above this session ({' > '.join(names)})"
            )
        team: Any = sub
        lineage = parent_lineage + (str(sub.id),)
    else:
        role = agent
        team = parent_team
        lineage = parent_lineage
    if lineage:
        cap = read_max_team_depth()
        if depth > cap:
            raise TeamLaunchError(
                f"depth cap: this launch would be depth {depth}; "
                f"subagents.max_team_depth is {cap}"
            )
    return LaunchTarget(
        role=role,
        team=team,
        team_lineage=lineage,
        depth=depth,
        reports_to=_parent_reports_to(parent_session),
        is_team_launch=bool(sub_name),
    )


def _lineage_names(session: "Session", lineage: tuple[str, ...]) -> list[str]:
    """Team names for a lineage of ids, for an error a reader can act on."""
    registry = getattr(session, "team_registry", None)
    names: list[str] = []
    for team_id in lineage:
        name = team_id
        try:
            if registry is not None:
                name = str(registry.get_team(team_id).name)
        except Exception:  # noqa: BLE001 — the id is still a usable name
            logger.debug("could not name lineage team %r", team_id, exc_info=True)
        names.append(name)
    return names


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
    inherited_model: ModelSpec | None = None,
    target: LaunchTarget | None = None,
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

    ``inherited_model`` exists for the resume path too, for the same reason.
    It is the model an INHERITING child takes from its real parent when that
    parent is not ``parent_session`` (D9.1). It is kept apart from
    ``model_spec`` because it is not a pin: ``owns_model`` stays False and a
    failure is not described as a pinned model's. ``None`` means the child
    inherits ``parent_session``'s own model, as on every launch.
    """
    if target is None:
        # Every launch resolves its team and depth, including a direct caller
        # that did not: an unresolved ``team:`` launch must raise here rather
        # than degrade into a generic child (BEN-1 N3).
        target = resolve_launch_target(agent, parent_session)
    if target.is_team_launch and target.team is not None:
        # Recorded as ``team:<name>`` whatever spelling launched it (a bare
        # roster name, a different case), so the row says what it is and a
        # resume re-enters the team path (BEN-7-D2).
        agent = f"{TEAM_LAUNCH_PREFIX}{target.team.name}"
    effective_prompt, profile = _effective_prompt(prompt, agent, parent_session, target)
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
            inherited_model=inherited_model,
            target=target,
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
        # The MODEL too, on the same registration-time rule and for the same
        # reason (a queued job that never starts must still be able to name what
        # it is): ``model_spec`` is the tier or role pin this launch resolved,
        # and ``None`` means the child owns no model and runs on the PARENT's —
        # which is what the label says in that case, so the ``task`` result can
        # say "inherits this session's model" as a fact rather than as an
        # absence. The runner still overwrites this once the child is built, and
        # that write must win: a restored provider fallback is the model the
        # child actually calls, and pricing reads this field.
        named = model_spec if model_spec is not None else inherited_model
        job.model_label = (
            f"{named.provider}/{named.model_id}"
            if named is not None
            else (getattr(parent_session, "effective_model_label", "") or None)
        )
        # And whose choice that was, on the same registration-time rule. Stamped
        # here rather than inferred from the label later: a tier or role pin
        # that resolves to the session's OWN model produces a label identical to
        # the parent's, so the ``task`` result line cannot tell the two apart by
        # comparing them — and it must, because "the child inherited" and "a pin
        # was accepted" are different facts about who spent the money. The
        # runner never rewrites this one (see ``AsyncJob.owns_model``).
        job.owns_model = model_spec is not None
        # And WHICH model the pin asked for, on the same registration-time rule
        # and as the mirror of that attribution: ``model_label`` above is
        # overwritten once the child is built (and again by every route edge),
        # so without this the pin label survives NOWHERE and "pinned to X,
        # running on Y" — the comparison that makes a silent substitution
        # visible — cannot be rendered at all. Never rewritten, unlike the
        # effective label (see ``AsyncJob.requested_model_label``).
        job.requested_model_label = (
            f"{model_spec.provider}/{model_spec.model_id}" if model_spec is not None else None
        )
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
            team_name=str(getattr(target.team, "name", "") or ""),
            team_lineage=target.team_lineage,
            depth=target.depth,
        )
    if queued:
        logger.info("subagent job %s (%s) queued: manager at capacity", job_id, label)
    return job_id


def _effective_prompt(
    prompt: str,
    agent: str,
    parent_session: "Session",
    target: LaunchTarget | None = None,
) -> tuple[str, "AgentProfile | None"]:
    """The exact launch message after reusable and team instruction layers.

    ``target is None``, and a plain member launch at depth 1, keep the
    original bytes exactly: that path is BEN-1 N0's baseline, pinned by a
    golden test. Only a ``team:`` launch or a launch at depth >= 2 inside a
    team lineage gains text (BEN-7-D2/D4).
    """
    role = target.role if target is not None else agent
    profile = _resolve_role(role, parent_session)
    if profile is not None:
        effective_prompt = profile.preamble + prompt
    elif role == "scout":
        effective_prompt = SCOUT_PREAMBLE + prompt
    else:
        specialist_prompt = _specialist_instructions(role, parent_session)
        effective_prompt = specialist_prompt + "\n\n" + prompt if specialist_prompt else prompt
    if target is None:
        team = getattr(parent_session, "active_team", None)
    else:
        team = target.team
    nested = target is not None and (
        target.is_team_launch or (target.depth >= 2 and bool(target.team_lineage))
    )
    if not nested:
        if team is not None:
            try:
                effective_prompt = team.member_preamble(agent) + effective_prompt
            except Exception:  # noqa: BLE001 — a bad brief must not lose the child
                logger.warning("could not stamp team preamble for %r", agent, exc_info=True)
        return effective_prompt, profile
    assert target is not None
    from local_operator.teams import escalation_preamble

    # Team text first, then the chain of command, then the role: the order
    # depth-1 member stamping already uses, and the head the N1/N3 scorers
    # read (D2). NOT ``attach_team``'s profile-first order.
    head = ""
    if team is not None:
        try:
            head = (
                team.manager_preamble()
                if target.is_team_launch
                else team.member_preamble(target.role)
            )
        except Exception:  # noqa: BLE001 — a bad brief must not lose the child
            logger.warning("could not stamp team preamble for %r", agent, exc_info=True)
    return head + escalation_preamble(target.reports_to) + effective_prompt, profile


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
    inherited_model: ModelSpec | None = None,
    target: LaunchTarget | None = None,
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
                # The child is BUILT on its inherited model when a resume found
                # one (see ``run_subagent``); ``model_spec`` alone keeps its
                # meaning of "a pin" for the failure text below.
                model_spec=model_spec if model_spec is not None else inherited_model,
                job_id=job_id,
                resume_dir=resume_dir,
                agent=agent,
                profile=profile,
                restricted=restricted,
                target=target,
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
                # THE LAUNCH RECEIPT (a3). Staged here because this is the first
                # moment the child's transcript directory is known, and a crash
                # any time after this leaves the on-disk statement "a lane ran
                # here" that no later artifact can supply — the parent writes
                # nothing once it is killed. Withdrawn on settle (the finally
                # below); a leftover file IS the evidence. Best-effort: a failed
                # write must never stop the child from running.
                try:
                    ledger.write_lane_receipt(
                        child._transcript.directory,
                        ledger.build_lane_payload(
                            job_id=job_id,
                            label=label,
                            agent_role=str(getattr(job, "agent_role", "") or ""),
                            child_session_id=child._transcript.directory.name,
                            parent_session_id=str(getattr(parent_session, "session_id", "") or ""),
                            parent_job_id=getattr(job, "parent_job_id", None),
                            started_at=float(getattr(job, "start_time", 0.0) or time.time()),
                        ),
                    )
                except Exception:  # noqa: BLE001 - evidence is not load-bearing here
                    logger.warning("could not write subagent lane receipt", exc_info=True)
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
                    comms,
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
            await _finish_child_browser(child, comms, job_id, "completed")
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
            #
            # A cancellation is a DELIBERATE stop unless the child's own loop
            # already classified the end as involuntary — a cancel that raced
            # the budget guard is still the budget guard, and the child's
            # recorded cause is the more specific fact.
            cancelled_cause = str(final.get("cut_off_cause") or "")
            with contextlib.suppress(BaseException):
                await _settle_child_cleanup(
                    asyncio.create_task(_finish_child_browser(child, comms, job_id, "cancelled"))
                )
                await _publish_terminal_outcome(
                    comms,
                    emit,
                    job=job,
                    job_id=job_id,
                    label=label,
                    status="cancelled",
                    cut_off_cause=cancelled_cause,
                    cut_off=str(final.get("cut_off") or ""),
                )
            raise
        except Exception as exc:
            # The error text is kept on the record as well as the job row: it
            # is what the roster shows for a failed child once the row is
            # swept, which is the state an operator is most likely to be
            # looking at when they ask what went wrong.
            #
            # The CAUSE comes from the child's own classification, read off the
            # relay's cell (the loop reported its end) with the child's own flag
            # as the fallback for an arm the relay never saw — the budget guard
            # ends the child's run normally, so the relay always sees it, but a
            # cancellation that races the settle can reach here without one.
            # ``None``-safe on ``child``: this arm can fire before the child is
            # even built.
            cut_off_cause = str(final.get("cut_off_cause") or "")
            if not cut_off_cause and child is not None:
                cut_off_cause = str(getattr(child, "_cut_off_cause", "") or "")
            await _finish_child_browser(child, comms, job_id, "failed")
            await _publish_terminal_outcome(
                comms,
                emit,
                job=job,
                job_id=job_id,
                label=label,
                status="failed",
                error_text=str(exc),
                cut_off_cause=cut_off_cause,
                cut_off=str(final.get("cut_off") or ""),
            )
            raise
        finally:
            if unsubscribe is not None:
                unsubscribe()
            # WITHDRAW THE LANE RECEIPT. Every settle arm (completed /
            # cancelled / failed) and every bare exception falls through this
            # single finally, so one call here covers them all — a separate
            # call per arm would be the same unlink three more times. A receipt
            # that survives this is exactly the "started, never settled"
            # reading the boot reconcile pass reports. Best-effort.
            if child is not None:
                try:
                    ledger.withdraw_lane_receipt(child._transcript.directory, job_id)
                except Exception:  # noqa: BLE001 - teardown must not fail over evidence
                    logger.warning("could not withdraw subagent lane receipt", exc_info=True)
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


async def _finish_child_browser(
    child: "Session | None", comms: Any, job_id: str, outcome: str
) -> None:
    """Resource settlement precedes authoritative outcome publication.

    Pausing cancels the runner too, but is not task completion. Preserve that
    explicit lifecycle hold through dispose rather than classifying cancellation
    (or a dead process) as authority to close a suspended interaction.
    """
    if child is None:
        return
    record = comms._record(job_id) if comms is not None else None
    resource = getattr(getattr(child, "_browser", None), "resource", None)
    if record is not None and record.paused and resource is not None:
        if resource.generation or resource.path.exists():
            try:
                resource.initialize()
                resource.record["retention"] = "paused scope"
                resource.remember(child._browser.surface_id, state="retained")
            except (RuntimeError, OSError, ValueError):
                logger.warning("paused browser ownership changed; retained successor untouched")
        return
    if resource is None or not callable(getattr(child, "finish_browser_scope", None)):
        # A host that built a BrowserSurface without a resource has no durable
        # ownership to settle. Reading the generation anyway raised straight
        # through the completed/failed paths, replacing the child's real outcome.
        return
    result = await child.finish_browser_scope(
        scope_id=child.session_id,
        # Off the resource the branch has already proven non-None, not the
        # Session property, so this cannot fault on an unwired host.
        generation=resource.execution_generation,
        outcome=outcome,
    )
    if result.state not in ("closed", "retained"):
        logger.warning("child browser cleanup %s: %s", result.state, result.detail)


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
    cut_off_cause: str = "",
    cut_off: str = "",
) -> tuple[str, str | None, str | None]:
    """Resolve and deliver the one terminal fact owned by a child run.

    A terminal outcome exists before its parent event fan-out. Cancellation in
    that fan-out must therefore interrupt delivery, not rewrite completion or
    failure into cancellation. Retrying the interrupted fan-out also reaches
    subscribers skipped when an earlier subscriber was cancelled.

    ``cut_off_cause``/``cut_off`` ride the same three surfaces as the status:
    the child's job ROW (what the panel and ``jobs.list()`` read), the emitted
    ``SubagentEndEvent`` (what the parent's stream sees), and the comms RECORD
    (the durable half that outlives the swept row). A child the loop cut off
    mid-flight used to settle with no vocabulary at all, so its parent saw a
    child that had "finished" — the reported "stopping without committing".
    """
    outcome = (
        comms.record_outcome(
            job_id,
            status,
            error_text=error_text,
            result_text=result_text,
            cut_off_cause=cut_off_cause,
        )
        if comms is not None
        else None
    )
    resolved_status, resolved_error, resolved_result = outcome or (
        status,
        error_text,
        result_text,
    )
    if job is not None:
        # The row is the LIVE surface: ``subagent_panel.status_glyph`` reads it
        # as ``cut_off=bool(job.cut_off_cause)`` and renders the word "cut off".
        # Written here rather than at the call sites so every settle arm (the
        # clean, the cancelled and the failed one) stamps it exactly once.
        job.cut_off_cause = cut_off_cause
    event = SubagentEndEvent(
        job_id=job_id,
        label=label,
        status=resolved_status,
        error_text=resolved_error,
        result_text=resolved_result,
        cut_off_cause=cut_off_cause,
        cut_off=cut_off,
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


def _display_model_name(selector: str) -> str:
    """The product's own vocabulary for one ``provider/model_id`` selector.

    ``model/naming.py``'s honesty rule decides: a display name only where one
    names this model and no other, else the selector itself. Shared by the two
    halves of the notice pair so neither can be spelled in a vocabulary the
    other does not speak.
    """
    provider, _, model_id = selector.partition("/")
    return model_label_forms(provider, model_id).full


def _pinned_fallback_notice(
    label: str, role: str, requested: str, effective: str, reason: str
) -> str:
    """The notice a pinned child's fallback shows on the PARENT's stream.

    Four facts, because a reader must be able to judge the substitution without
    opening anything: WHICH child (its label and role), WHAT the launch asked
    for (the pin), what is ACTUALLY serving, and the route edge's own cause
    phrase. The cause is carried VERBATIM — never re-derived here — so the
    enrichment that names the refusing provider and the failure kind (done at
    the descent point, where the classification already exists) reaches this
    surface unchanged; an absent reason leaves the sentence without a cause
    clause rather than inventing one.

    The two models are stated in the PRODUCT's vocabulary — resolved display
    names where naming can vouch for one, else the selector — and as an
    ``A → B`` pair, the same relation spelling the band's badge uses, rather
    than words the badge does not (design D4 / UX U3). The pair is joined with
    non-breaking spaces INSIDE each display name too — the names carry spaces
    of their own — because the notice's own wrap breaks on ASCII spaces only
    (``transcript.wrap_cells``): with a plain-space join the pair still split
    (`…Sonnet 5.5 →` / `DeepSeek Flash…`) at exactly the widths this fix
    exists for. The pair reads identically in every renderer and copies as
    visible spaces; the rest of the sentence wraps as prose.
    """
    role_clause = f" ({role})" if role else ""
    cause = f" — {reason.strip()}" if reason.strip() else ""
    pin = _display_model_name(requested).replace(" ", "\u00a0")
    running = _display_model_name(effective).replace(" ", "\u00a0")
    pair = f"{pin}\u00a0→\u00a0{running}"
    return f"subagent '{label}'{role_clause} pinned {pair}{cause}."


def _make_relay(
    job_id: str,
    label: str,
    job: Any,
    jobs_manager: "AsyncJobManager",
    emit: Callable[[AgentEvent], Awaitable[None]],
    report_progress: Callable[[str], None],
    final: dict[str, Any],
    owner_jobs: Any = None,
    comms: Any = None,
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
    #: Text deltas of the CURRENT assistant message not yet written to the
    #: trajectory, and the timer that will write them. See ``flush_text``.
    pending_text: list[str] = []
    pending_event: list[MessageUpdateEvent] = []
    flush_handle: list[asyncio.TimerHandle] = []

    def append_record(record: dict[str, Any]) -> None:
        nonlocal relayed
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

    def flush_text(*, from_timer: bool = False) -> None:
        """Write the buffered text deltas as ONE ``message_update`` row.

        WHY TEXT IS COALESCED. The loop yields one ``MessageUpdateEvent`` per
        provider text chunk -- roughly one per token. Appending each as its own
        trajectory row had two costs, both measured with
        ``scripts/bench_subagent_fanout.py``:

        * the WINDOW: a child streaming a long answer filled the 500-row cap
          with deltas (489 of 500 rows in a 3-turn run), evicting the tool
          calls the subagent page exists to show -- the same defect the
          reasoning skip above fixed for the thinking channel;
        * the LOOP: every append invalidates the parent's roster memo, so the
          parent's 50 ms coalescer re-froze and re-shipped rows at token rate
          for every streaming child. With a viewer attached that refresh was
          about half of all process CPU at 16 children.

        LOSSLESS FOR THE PAGE. The page folds ``message_update`` rows by
        concatenating their ``delta`` per message id
        (``tui/widgets/subagent_view.py``) and ``message_end`` then adopts the
        authoritative text, so one row holding the concatenation renders the
        same text as the N rows it replaces. The page still streams, at the
        ``SUBAGENT_TEXT_FLUSH_S`` cadence instead of per token.

        ORDER IS PRESERVED because every non-delta event flushes FIRST, so a
        coalesced row can never land after a later event and the relay's
        monotonic stamp still orders rows exactly as the child emitted them.
        """
        while flush_handle:
            flush_handle.pop().cancel()
        if not pending_text or not pending_event:
            pending_text.clear()
            pending_event.clear()
            return
        latest = pending_event[-1]
        delta = "".join(pending_text)
        pending_text.clear()
        pending_event.clear()
        if job is None or job.trajectory is None:
            return
        record = latest.model_dump(mode="json")
        record["delta"] = delta
        append_record(record)
        if from_timer:
            # A boundary flush rides the event that caused it, which already
            # reaches the parent's roster coalescer through the comms watcher.
            # A TIMER flush has no such event behind it, so it says so itself
            # -- otherwise a child that streams prose and then goes quiet would
            # leave its last quarter-second of text unpublished until its next
            # event. Transient: the durable roster sidecar holds no trajectory.
            notify = getattr(jobs_manager, "_notify_transient_job_change", None)
            if callable(notify):
                notify()

    async def relay(event: AgentEvent) -> None:
        nonlocal streaming
        # The model's private reasoning is display-only and has NO row on the
        # subagent page, so it must not consume a slot in this bounded window:
        # reasoning is one event per reasoning token, and 250 of them per model
        # call silently evicted the tool calls and messages the page exists to
        # show (review round 1, MAJOR-1 — measured: three tool rows became one
        # with 250 fragments per call, all three stayed with none). Coalescing
        # would still spend a slot per model call on a row nothing can paint,
        # so the family is dropped here and the parent's own stream keeps
        # receiving it untouched.
        if isinstance(event, MessageUpdateEvent) and event.delta:
            # Buffered, not appended: see ``flush_text``. A text chunk for a
            # DIFFERENT message than the buffered one flushes first, so one
            # coalesced row never spans two messages.
            if pending_event and getattr(pending_event[-1].message, "id", None) != getattr(
                event.message, "id", None
            ):
                flush_text()
            pending_text.append(event.delta)
            pending_event[:] = [event]
            if not flush_handle:
                flush_handle.append(
                    asyncio.get_running_loop().call_later(
                        SUBAGENT_TEXT_FLUSH_S, lambda: flush_text(from_timer=True)
                    )
                )
        elif not isinstance(event, ReasoningDeltaEvent):
            # Any other event is a boundary: the buffered text precedes it.
            flush_text()
            if job is not None and job.trajectory is not None:
                append_record(event.model_dump(mode="json"))
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
                effective = f"{event.provider}/{event.model_id}"
                job.model_label = effective
                if event.context_window > 0:
                    job.context_window = event.context_window
                # Pin integrity, the loud half. A PINNED child (``owns_model``)
                # the route has taken off its requested model is the failure
                # this harness must never let pass silently: the pin exists so
                # a review cannot collapse onto the author's model, and a quiet
                # substitution restores exactly that collapse — "Independence
                # that can silently collapse into self-review is not
                # independence" (Session._launch_subagent). The marker and its
                # ONE notice are stamped HERE because the relay is where the
                # child's route edge meets the parent's job row, and the SAME
                # edge restores the state when the requested model serves
                # again. An unpinned child (``requested`` empty) inherits the
                # parent's routing and carries none of this by design.
                requested = str(getattr(job, "requested_model_label", "") or "")
                if getattr(job, "owns_model", None) is True and requested:
                    if event.is_fallback and effective != requested:
                        was = bool(getattr(job, "model_fallback", False))
                        job.model_fallback = True
                        job.model_fallback_reason = str(event.reason or "")
                        if not was:
                            # ONE notice per fallback EPISODE. A later hop to
                            # another fallback updates the row's label (and so
                            # the badge) without re-announcing the descent, and
                            # a recovery clears the flag so a genuinely new
                            # episode speaks again.
                            await emit(
                                NoticeEvent(
                                    text=_pinned_fallback_notice(
                                        label,
                                        str(getattr(job, "agent_role", "") or ""),
                                        requested,
                                        effective,
                                        str(event.reason or ""),
                                    ),
                                    kind="warning",
                                    headline=f"'{label}' fell back to {effective}",
                                )
                            )
                    elif not event.is_fallback and effective == requested:
                        # Recovery edge: the requested model serves again, so
                        # the marker must not outlive it — a stale badge would
                        # claim a substitution that has ended.
                        job.model_fallback = False
                        job.model_fallback_reason = ""
                jobs_manager._notify_roster_change()
        elif isinstance(event, AgentEndEvent):
            if event.error:
                final["error"] = event.error
            # The child's OWN classification, carried rather than re-derived:
            # ``Session._classify_cut_off`` has already decided whether this end
            # is involuntary, and re-deciding from ``error`` here would answer
            # the same question in a second place — the drift this taxonomy
            # exists to remove. Both fields default to "", so a clean child end
            # and an OLD child runtime that has never heard of them both leave
            # the cells empty.
            if event.cut_off_cause:
                final["cut_off_cause"] = event.cut_off_cause
                final["cut_off"] = event.cut_off
        if progress is not None:
            # Same string into latest_details so the 1 Hz jobs.list() poll
            # and the event stream agree about what the child is doing.
            report_progress(progress)
            # And the same moment onto the comms RECORD, so the stamp survives a
            # path that never sees the job row (a nested child after a restart).
            # Best-effort inside note_progress; the relay must not care.
            if comms is not None:
                comms.note_progress(job_id)
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

#: Rendered in place of a tool schema when a child reads an ``mcp://`` tool URL
#: of a server declared ``ownTurnOnly`` (see ``MCPStdioServerConfig``). The
#: server belongs to the OWNING session's own turn: a child discovers it but
#: must route the work back to the parent, the only session that can act.
_OWN_TURN_ONLY_DENIED = (
    "This MCP server is reserved for the owning session's own turn "
    "(`ownTurnOnly`): a delegated child neither inherits its tools nor can "
    "enable them. If the task needs this server, report to your parent "
    "(`hub`) so the parent's own turn makes the call."
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
    session. ``on_incident`` and ``on_recovery`` are the same shape and carry
    the same prohibition — a child installing either would silently REPLACE
    the parent's sink, and the parent would stop hearing about MCP failures
    and recoveries for the rest of the session. This is already correct
    because ``attach_mcp_dispose`` (which installs both) is never called for a
    child; do not add them here.

    The cost is that a reconnect during the child's run leaves the
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
        def included(tool: AgentTool) -> bool:
            found = origin(tool)
            if found is None or found not in enabled:
                return False
            # ``ownTurnOnly`` servers belong to the parent's own turn: a child
            # neither inherits their tools nor can activate them back in.
            return not server_own_turn_only(manager.get_server_config(found[0]))

        return [tool for tool in source if included(tool)]

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

    def activate(server_name: str, raw_tool_name: str) -> bool:
        # Unreachable for a restricted child: its resolver is built with
        # ``deny_activation_reason``, which returns before calling this. Kept
        # unguarded so there is ONE activation path rather than a second
        # allow-check that could drift from the resolver's.
        #
        # The return value travels back to the resolver, which is what tells the
        # user whether the schema reaches the next model call or the next turn:
        # the tools array is published once per turn (``Session._wire_tools``),
        # so a child that activates mid-turn is deferred exactly like a top-level
        # session's.
        enabled.add((server_name, raw_tool_name))
        if child is None:
            return True
        return child.refresh_tools(base() + selected(manager.get_tools()))

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
                found = origin(tool)
                if tool.name != name or found not in deferred:
                    continue
                # The fallback must not hand a child what ``selected`` would
                # not: an ``ownTurnOnly`` server stays out of reach even when a
                # discovery deferred its schema (search defers rather than
                # activates on this side).
                if server_own_turn_only(manager.get_server_config(found[0])):
                    continue
                return tool
            return prior(name) if prior is not None else None

        setattr(session, "_mcp_deferred_origins", deferred)
        session.set_fallback_tool_resolver(resolve_deferred)

    def own_turn_only_reason(server_name: str) -> str | None:
        # Asked fresh per read: ``/mcp reload`` replaces the manager's configs
        # wholesale, so a snapshot taken at wiring time would go stale.
        if server_own_turn_only(manager.get_server_config(server_name)):
            return _OWN_TURN_ONLY_DENIED
        return None

    return _ChildMcp(
        tools=selected(manager.get_tools()),
        catalogue=lambda query: render_mcp_catalogue(manager, query),
        resolve=make_mcp_resolver(
            manager,
            activate,
            deny_activation_reason=_MCP_ACTIVATION_DENIED if restricted else None,
            deny_server_reason=own_turn_only_reason,
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
    target: LaunchTarget | None = None,
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
            target=target,
            cleanup=cleanup,
        )
    except BaseException:
        await _settle_child_cleanup(asyncio.create_task(cleanup.aclose()))
        raise
    # Normal lifetime now belongs to the returned Session's dispose hooks.
    cleanup.pop_all()
    return child


# ---------------------------------------------------------------------------
# Child knowledge tail: the config gate and the slimmer
# ---------------------------------------------------------------------------
#
# A child inherits a bounded copy of the parent's selected knowledge (the
# parent's ``frozen_block``). On this fleet children carry the large majority
# of context tokens, and the inherited copy ships notable furniture twice
# over: a ``<mcps>`` catalogue of its own next to the parent's, and any
# ``<resource_recommendations>`` block that was delivered for the PARENT's
# message. The reductions below cut that furniture while keeping every
# guide/skill NAME: the discoverability contract is that a child which sees a
# name can ``read`` its full text, and a name it never sees is one it cannot
# ask for.

#: Whether a child's inherited knowledge tail is slimmed before it rides the
#: child's system prompt (``subagents.slim_child_knowledge``). ON by default:
#: what the slimmer removes is either the parent's business (its
#: recommendations), a duplicate of something the child renders itself (its
#: catalogue), or available on demand (full descriptions).
DEFAULT_SLIM_CHILD_KNOWLEDGE = True

#: The parent's knowledge block, as ``session_factory._select_knowledge_block``
#: composes it, is a ``"\n\n"``-joined stack of sections: the skills render
#: (``<guides>``/``<skills>`` line listings), the query-situational ``<mcps>``
#: catalogue, then a ``<resource_recommendations>`` block per classification
#: answer. These markers are that composition's furniture; the slimmer only
#: ever strips exactly-marked sections, and a test pins the markers against
#: the producer so a rename cannot silently turn a strip into a no-op.
_RECOMMENDATIONS_OPEN = "<resource_recommendations>"
_RECOMMENDATIONS_CLOSE = "</resource_recommendations>"
_MCP_CATALOGUE_OPEN = "<mcps>"
_MCP_CATALOGUE_CLOSE = "</mcps>"

#: Marker pairs whose ``- name: description`` bullet lines are capped for a
#: child (see :func:`_cap_listing_descriptions`). Only these sections are
#: touched: the imperative paragraphs above the listings carry the
#: read-before-acting rule and must survive byte-identical.
_LISTING_SECTIONS: tuple[tuple[str, str], ...] = (
    ("<guides>", "</guides>"),
    ("<skills>", "</skills>"),
)

#: Hard bound on the inherited block, unchanged from before the slimmer: the
#: child tail is re-sent on every call, so a huge parent directory must not
#: grow the child's prompt without limit. Applied AFTER the slimming, so the
#: budget is spent on NAMES rather than on the furniture already removed.
_CHILD_KNOWLEDGE_MAX_CHARS = 12_000

#: Per-line description cap for the guides/skills listings. Descriptions exist
#: to help a model recognise which resource it wants; past this the line is
#: clipped, and the full text (plus every reference file) stays one
#: ``skill://``/``guide://`` read away.
_CHILD_KNOWLEDGE_MAX_DESCRIPTION_CHARS = 160


def read_slim_child_knowledge() -> bool:
    """``subagents.slim_child_knowledge``: slim the inherited tail?

    Read at every child BUILD rather than once per process, like
    ``subagents.models``: a running child's tail is frozen at construction, so
    the next delegation is the first moment an operator's edit can take effect
    — and it should.

    Never raises, and every unrecognised shape (absent, blank, a typo, or a
    string YAML never coerced) resolves to :data:`DEFAULT_SLIM_CHILD_KNOWLEDGE`:
    a corrupt ``config.yml`` must not cost a child its prompt, and the default
    is the behaviour this key ships as.
    """
    from local_operator.config import ConfigManager

    try:
        raw = ConfigManager(config_dir()).get_config_value("subagents", None)
    except Exception:  # noqa: BLE001 — a child build must never fail on a config read
        return DEFAULT_SLIM_CHILD_KNOWLEDGE
    stored = raw.get("slim_child_knowledge") if isinstance(raw, dict) else None
    if stored is None:
        return DEFAULT_SLIM_CHILD_KNOWLEDGE
    return _strict_bool(stored, DEFAULT_SLIM_CHILD_KNOWLEDGE)


def _strict_bool(value: object, default: bool) -> bool:
    """A REAL boolean or ``default`` — never ``bool(value)``.

    Deliberately the same reading as ``settings_io.strict_bool``, which the
    settings page reads through and every other consumer imports: a
    hand-edited ``"false"`` is a non-empty string, and ``bool("false")`` is
    ``True`` — the page would paint ``off`` for a child that still slims.

    The spelling table is duplicated here ON PURPOSE: this module lives in one
    of the agent-facing trees ``tests/unit/test_approval_source_boundary.py``
    pins against reaching ``settings_io`` at all (a function-local import is
    still a reach — the pin is on the IMPORT), because an agent-reachable
    module holding the facade could attribute its own write as ``"local"``.
    ``test_subagent_child_knowledge.py`` walks the two tables against each
    other, so a change to either spelling set fails a test instead of
    silently moving these readers apart (the ``resume.py`` title-type shape).
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("true", "yes", "on", "1"):
            return True
        if lowered in ("false", "no", "off", "0"):
            return False
    return default


def _slim_child_knowledge(knowledge: str, *, drop_catalogue: bool) -> str:
    """Slim a parent knowledge block for a child's system prompt.

    Each reduction is safe alone, and all three share one contract — what
    leaves is content the child does not lose access to:

    (a) The ``<resource_recommendations>`` block was advisory for the PARENT's
        message and deduped against the parent's own context; it was never
        selected for the child's task.
    (b) A ``<mcps>`` catalogue is inherited whenever MCP is configured, and
        the child composes its own against its task prompt; keeping both told
        the model the same protocol twice. Stripped only when
        ``drop_catalogue`` — i.e. when the child HAS a catalogue to stand in
        its place; with none, the inherited copy is the child's only MCP hint.
    (c) Listing descriptions past the cap are clipped per line; names are
        never touched.
    """
    text = _strip_marked_sections(knowledge, _RECOMMENDATIONS_OPEN, _RECOMMENDATIONS_CLOSE)
    if drop_catalogue:
        text = _strip_marked_sections(text, _MCP_CATALOGUE_OPEN, _MCP_CATALOGUE_CLOSE)
    return _cap_listing_descriptions(text)


def _strip_marked_sections(text: str, open_marker: str, close_marker: str) -> str:
    """Remove every ``open … close`` section, collapsing the seam.

    Sections are joined with ``"\n\n"`` by the producer, so dropping one must
    leave exactly one separator where two sections now meet and no dangling
    blank line at either end — otherwise every strip would grow the block it
    was meant to shrink. A marker whose close cannot be found is left alone:
    deleting from an unbounded marker would eat the rest of the block.
    """
    search_from = 0
    while True:
        start = text.find(open_marker, search_from)
        if start == -1:
            return text
        close = text.find(close_marker, start + len(open_marker))
        if close == -1:
            return text
        head = text[:start].rstrip("\n")
        tail = text[close + len(close_marker) :].lstrip("\n")
        text = f"{head}\n\n{tail}" if head and tail else head + tail
        search_from = len(head)


def _cap_listing_descriptions(text: str) -> str:
    """Clip ``- name: description`` bullets inside the guides/skills listings.

    Only the bullet lines INSIDE ``<guides>``/``</guides>`` and
    ``<skills>``/``</skills>`` are touched; the section imperatives, the
    ``<mcps>`` catalogue and any other prose pass through byte-identical, and
    the NAME (everything before the first ``": "``) is never shortened
    because it is the key a follow-up ``read`` needs.
    """
    for open_marker, close_marker in _LISTING_SECTIONS:
        search_from = 0
        while True:
            start = text.find(open_marker, search_from)
            if start == -1:
                break
            close = text.find(close_marker, start + len(open_marker))
            if close == -1:
                break
            inner_start = start + len(open_marker)
            capped = "\n".join(
                _cap_listing_line(line) for line in text[inner_start:close].split("\n")
            )
            text = f"{text[:inner_start]}{capped}{text[close:]}"
            search_from = inner_start + len(capped)
    return text


def _cap_listing_line(line: str) -> str:
    """One listing bullet, with its description clipped at the cap.

    Non-bullet lines and bullets without a ``": "`` separator are returned
    verbatim: the shapes here come from ``skills/index.render_block``, and a
    line this function cannot classify must not be edited into something the
    child reads differently.
    """
    if not line.startswith("- "):
        return line
    name, sep, description = line[2:].partition(": ")
    if not sep or len(description) <= _CHILD_KNOWLEDGE_MAX_DESCRIPTION_CHARS:
        return line
    clipped = description[: _CHILD_KNOWLEDGE_MAX_DESCRIPTION_CHARS - 1].rstrip()
    return f"- {name}: {clipped}…"


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
    target: LaunchTarget | None = None,
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
    from local_operator.prompts_api import CHANNEL_HUB, build_system_blocks
    from local_operator.session.session import Session
    from local_operator.session.transcript import Transcript
    from local_operator.session_factory import _env_details, load_user_instructions
    from local_operator.tools.registry import DEFAULT_TOOL_NAMES, create_tools

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
    # The DURABLE half of the child→parent link, for code-request attribution. The
    # live half is the parent handle attached below (see
    # ``code_requests.hook.attach_parent``); this stamp is what lets a later backfill —
    # a scan of a child that ran before this feature, or one whose live propagation
    # failed — attribute the child's rows to the conversation that asked for the work.
    # Best-effort by its own contract, and it preserves the ``label``/``agent`` keys
    # the picker reads (it is a read-modify-write, not a second stamp).
    from local_operator.code_requests.hook import stamp_origin_parent

    stamp_origin_parent(session_dir, str(getattr(parent_session, "session_id", "") or ""))
    # Birth metadata must be durable before publication, without its fsync
    # blocking the parent or other children sharing this event loop.
    transcript = await asyncio.to_thread(Transcript, session_dir)
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
    # context: ``Session._build_tool_context`` passes ``job_id`` and
    # ``may_delegate``. ``is_child(job_id)`` without ``may_delegate`` is what
    # makes the replacement the CHILD shape (message your parent); a child
    # that holds ``task`` gets the parent shape scoped to its own subtree
    # (BEN-7-D5, rebuilt after the prune below). If ``job_id`` ever stopped
    # reaching that context, a child would silently be handed its parent's
    # UNSCOPED tool.
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
        project_registry=getattr(parent_session, "project_registry", None),
        web_search_settings=ConfigManager(config_dir()).get_config_value("web_search", None),
        web_fetch_settings=ConfigManager(config_dir()).get_config_value("web_fetch", None),
    )
    # ``restricted`` also carries a sticky MCP-activation denial inherited by
    # plain descendants; that is not a role allowlist and must not shrink their
    # ordinary builtin inventory. Keep tool construction keyed to actual role
    # policy (plus scout's explicit read-only fallback), not the MCP boundary.
    role_limited = (profile is not None and bool(profile.tools)) or agent == "scout"
    #: The names this child's ALLOWLIST actually named — the deferral pins below
    #: read this and not ``profile.tools``, because an allowlist is not always a
    #: profile: the scout fallback restricts through ``READ_ONLY_TOOLS`` with no
    #: profile at all (CI round 3). Empty for a freely-inventoried child, which
    #: must pay the deferral like any other session.
    allowlist_names: frozenset[str] = frozenset()
    if role_limited:
        # The prior full-inventory-then-filter path exposed tools in registry
        # order; select that same order up front so createIf builders run only
        # for schemas this role can receive. Match the old filter's order:
        # allowlisted tools in registry order, then omitted network-floor tools
        # in registry order, then the child-only hub capability. Keeping the
        # floor appended matters for profiles that explicitly list one network
        # tool but not the other; moving it ahead of the allowlist changes the
        # provider-visible order even though the capability set is unchanged.
        allowed_names = set(profile.tools or ()) if profile is not None else set()
        if agent == "scout" and (profile is None or not profile.tools):
            allowed_names.update(SCOUT_TOOL_ALLOWLIST)
        allowlist_names = frozenset(allowed_names)
        builtin_names = [name for name in DEFAULT_TOOL_NAMES if name in allowed_names]
        for name in DEFAULT_TOOL_NAMES:
            if name in READ_ONLY_NETWORK_TOOLS and name not in builtin_names:
                builtin_names.append(name)
        if "hub" not in builtin_names:
            builtin_names.append("hub")
        tools = create_tools(tool_context, enabled=builtin_names)
    else:
        # Unrestricted/freeform children retain the full default inventory,
        # even when sticky MCP denial is inherited from a restricted ancestor.
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
    if not knowledge:
        # A skill-tree change invalidates the parent's frozen block and parks the
        # previous render in ``superseded_block``; the NEXT parent message re-renders
        # it. This closure is synchronous — it cannot wait for that render — so a child
        # spawned inside that window inherits yesterday's directory, which is strictly
        # better than an empty one. Same bound as before, applied to the block we
        # actually ship.
        knowledge = getattr(parent_hooks, "superseded_block", "") or ""
    # Slim BEFORE the size bound below, so the bound's budget is spent on names
    # rather than on furniture already removed. The gate exists for the accuracy
    # case: an operator can hand children the parent's block verbatim when a
    # task genuinely needs the full descriptions (``subagents.slim_child_knowledge``).
    if read_slim_child_knowledge():
        knowledge = _slim_child_knowledge(knowledge, drop_catalogue=mcp is not None)
    if len(knowledge) > _CHILD_KNOWLEDGE_MAX_CHARS:
        knowledge = knowledge[:_CHILD_KNOWLEDGE_MAX_CHARS].rsplit("\n", 1)[0]

    # Same construction-time freeze as the session provider's: this child's block 0
    # also starts a persisted prefix epoch when it changes, and this host probe
    # answers from the desktop app's heartbeat (see
    # ``prompts_api.host_capability_probes``). One child render is cheap; a child
    # that re-anchors mid-run is not, and a subagent's work is exactly the case
    # where a mid-run prefix loss costs the most.
    from local_operator.prompts_api import host_capability_probes

    host_has_browser, host_has_console = host_capability_probes()

    #: The child's ``GoalState``, filled in once the child Session exists (the
    #: provider is built before it). Empty means "not built yet": the parent's
    #: reading answers until then.
    child_holder: list[Any] = []

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
            # ...and its LIFECYCLE STATE, so a goal the parent has already
            # settled is not handed to the child as standing work. Read off the
            # live holder for the same reason the text is.
            goal_status=getattr(parent_session, "goal_status", ""),
            user_instructions=user_instructions,
            repo_guidance=repo_guidance,
            credentials=names,
            model_label=model_label,
            # THE PARENT'S ANSWER, read live off the parent's holder for the same
            # reason ``goal=`` above is: a child cannot answer this itself (no
            # control socket, no registrant), and whether an interface is attached
            # is a fact about the PARENT's session — the surface the operator is
            # attached to. ``interactivity()`` rather than ``is_interactive()``:
            # a parent with no runtime probe answers "unmeasured", and a child of
            # one must render nothing rather than inherit the fail-open default.
            #
            # ``CHANNEL_HUB`` is stated HERE rather than derived, because only this
            # call site knows it is a child: a top-level session also holds ``hub``
            # (it is how ITS children reach it), so inventory membership cannot tell
            # the two apart, and the child's hub is the one that reaches the
            # operator — one hop out, through the parent. The alternative the
            # builder would infer (``ask``) is a tool no child has
            # (``build_ask_tool`` refuses without a hook), which is exactly what the
            # round-1 reviews found this child being told to use (BLOCKER).
            #
            # Read through the CHILD's own holder once it exists: that holder
            # carries the parent's probe OBJECT (installed below) and latches it
            # at the child's own turn boundary (``GoalState.latch_interactivity``),
            # so a transient detach inside a child turn publishes nothing. Reading
            # ``parent_session.interactivity()`` instead would serve the PARENT's
            # snapshot, frozen at the parent's turn start for the whole child run.
            interactive=(
                child_holder[0].interactivity() if child_holder else parent_session.interactivity()
            ),
            channel=CHANNEL_HUB,
            host_has_browser=host_has_browser,
            host_has_console=host_has_console,
        )

    setattr(system_blocks_provider, "append_only_state", True)
    setattr(system_blocks_provider, "repo_guidance", repo_guidance)
    setattr(system_blocks_provider, "knowledge_hooks", parent_hooks)
    setattr(system_blocks_provider, "host_has_browser", host_has_browser)
    setattr(system_blocks_provider, "host_has_console", host_has_console)
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
        if model_spec is not None:
            # PIN the child's routing to the model the launch resolved (a role
            # tier, or a resumed child's recorded tier). Marked on the CHILD's
            # stream only — inside this guard, never on ``parent_stream``,
            # whose routing must stay untouched — and only when the launch
            # resolved an explicit spec: an inherit-child runs the parent's
            # model with no pin, and the failover policies are all gated on
            # the marker (see ``local_operator/providers/failover.py``).
            mark_pin = getattr(child_stream, "mark_launch_pin", None)
            if callable(mark_pin):
                mark_pin(f"{model_spec.provider}/{model_spec.model_id}")
    child = Session(
        model=model_spec if model_spec is not None else parent_session.model,
        # A child's model was chosen HERE (tier or parent), never by the
        # ``hosting``/``model_name`` keys, so a later edit to those must not
        # switch it — ``_job_id`` already guards that path, this makes the
        # provenance honest too.
        model_source="child",
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
        # The parent's confinement, when it has one: a child runs the same
        # local tools against the same host, so a child that was NOT confined
        # would be the bypass the boundary must not have (`task` is in the
        # default surface). Copied at construction because the child rebuilds
        # its tool context per turn and never re-reads the parent.
        confinement_root=getattr(parent_session, "_confinement_root", None),
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
        # Hooks report it as ``agent_type``, as Claude Code does.
        agent_type=agent,
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
        project_registry=getattr(parent_session, "project_registry", None),
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
    # The LIVE half of the same link: an event this child records for an opened or
    # acted-on code request is propagated to the parent's transcript as it happens, so
    # the operator's conversation shows the PR its subagent opened without waiting for
    # a scan of the child's directory. Best-effort; a session without the handle simply
    # propagates nothing (the scanner still finds the child's own rows).
    from local_operator.code_requests.hook import attach_parent

    attach_parent(child, parent_session)
    # THE CHILD CANNOT ANSWER THIS ITSELF, so its own holder gets the PROBE
    # OBJECT rather than a copied value: the child holds no control socket and no
    # registrant, and its only channel to a human is ``hub`` -> parent, so "is an
    # interface attached" is a fact about the PARENT's session. Installing the
    # object (exactly as ``goal=`` reads the parent live) keeps the child's answer
    # live per turn, and keeps it in agreement with the browser text the child
    # renders (``ToolContext.attached_probe`` reads the same holder).
    #
    # The parent's HOLDER is deliberately NOT shared. ``GoalState`` also carries
    # ``team_brief`` and ``agent_brief``; a child that inherited those through the
    # holder would silently start rendering the parent's ``<team>`` block, which
    # is an instruction-precedence bug rather than an inheritance.
    parent_probe = parent_session.interactivity_probe
    if parent_probe is not None:
        child._goal_state.interactive_probe = parent_probe
        # Only when the parent HAS a probe: a child of an unmeasured parent keeps
        # reading ``parent_session.interactivity()`` (``None``), never a holder
        # with no probe, which would read the same but by accident.
        child_holder.append(child._goal_state)
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
    # WHO MAY DELEGATE IS THE ROLE'S DECISION, NOT THE DEPTH'S (operator,
    # 2026-09-18: "if it is allowed the task tool, then it should use it; if
    # not allowed, then that agent is not allowed to create subagents and must
    # work on things itself"). The capability therefore follows the ALLOWANCE
    # down the lineage instead of stopping at one level: a child whose role
    # says ``delegate: yes`` (a manager) keeps ``task``/``wait``/``jobs`` at any
    # depth, and the tree it grows is navigable — the TUI re-scopes its roster
    # to the open page's direct children and the desktop UI walks the same
    # comms edges, both recursively. A role-less child owns no allowance and
    # inherits its parent's, so a parent that held ``task`` may hand it down
    # and a parent that did not hands down nothing.
    #
    # What still never crosses any boundary: ``wake`` and ``monitor``, for
    # every child — a child session ends after one prompt, so a wake or a
    # monitor armed there would be silently lost. Scouts lose the whole set: a
    # read-only agent that delegates autonomous work is not read-only. And a
    # role that does not delegate (a reviewer, a coder) loses it at every
    # depth, including as a grandchild of a manager: it does the work itself,
    # which is what the harness's own refusal message tells it.
    #
    # ``refresh_tools`` rather than touching ``_tools``: it is the committed
    # hook and it keeps the loop's ``context.tools`` in step.
    # DEFERRED SCHEMAS (``tools/deferral.py``): a role that NAMES a tool in its
    # allowlist keeps its schema published — the allowlist above already decided
    # what the child HOLDS; this decides only what its request array carries.
    # The set itself is the same one every session uses (see that module for why
    # a child-only set was measured and dropped).
    #
    # ``allowlist_names``, NOT ``profile.tools``, and the difference is a real
    # child: the scout fallback restricts through ``READ_ONLY_TOOLS`` with no
    # profile, so reading the profile pinned nothing and withheld two schemas
    # from a role whose read-only allowlist names both (``list_variables``,
    # ``read_variable``) — the exact rule this comment states, broken on the one
    # path that reaches an allowlist without a profile (CI round 3).
    set_deferral = getattr(child, "set_tool_deferral", None)
    if callable(set_deferral):
        set_deferral(pins=allowlist_names)
    merged_in = {tool.name for tool in child._tools} - {tool.name for tool in tools}
    if profile is not None:
        may_delegate = profile.may_delegate
    else:
        may_delegate = any(
            tool.name == "task" for tool in (getattr(parent_session, "_tools", None) or ())
        )
    if agent == "scout" or not may_delegate:
        drop = merged_in
    else:
        drop = {name for name in merged_in if name in ("wake", "monitor")}
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
    #
    # ``wait`` rides the SAME invariant, for the reason ``jobs`` alone did not
    # cover: ``jobs`` can observe a background job but cannot BLOCK on one, so a
    # child that backgrounded a long command had only two ways to await it —
    # re-peek in a loop, or a foreground ``sleep N; tail log``. The second is
    # what child f7318cc06bdd did for hours (2026-09-24), and a foreground bash
    # is not a tool boundary, so each of its parent's hub notes waited out a
    # 15-30 min sleep. ``wait`` is the blocking primitive that is NOT deaf: it
    # returns on the job settling, on a hub note (``queue_aside`` marks the
    # peer-arrival event it parks on) and on a steer. It spawns nothing, and it
    # is scoped to THIS child's own job manager: an id that resolves through the
    # shared comms registry to a sibling is not in ``child.jobs`` and is refused
    # as ``unknown job`` (pinned in tests/unit/session/test_child_wait.py).
    # Precisely what it promises about a note, because the two cases differ: a
    # note arriving WHILE the wait is parked interrupts it at once (that is the
    # wake above), while one already queued before the wait parks has been
    # counted by the peer snapshot the wait takes before parking — it is
    # delivered at the next boundary, not lost, but it does NOT shorten this
    # park.
    # ``wake`` stays pruned: a child's session ends after one prompt, so a wake
    # it armed would be silently lost.
    if _can_background(tools):
        drop = drop - {"jobs", "wait"}
    child.refresh_tools([tool for tool in child._tools if tool.name not in drop])
    if any(tool.name == "task" for tool in child._tools):
        # A child that KEPT ``task`` (a pod lead) rebuilds ``hub`` now that its
        # inventory says it may delegate: ``build_hub_tool`` hands it the
        # parent shape, scoped to its own subtree (BEN-7-D5). The constructor's
        # merge ran before the prune, while no ``task`` was held yet, so it
        # built the message-only shape. ``hub`` alone, so nothing pruned above
        # comes back.
        child._merge_capability_tools(("hub",))
    # A DECLARED parent inventory carries down, or a bounded session could reach
    # an excluded tool by delegating to a child that never heard of the bound.
    # One hop is enough to make the declaration meaningful and is also all the
    # grandchild level needs: this stamps the child, and the child stamps its own
    # children off the same attribute. The child's own role allow-list has already
    # narrowed its candidate set above, so the two intersect rather than either
    # winning — a caller who declares a set that cannot reach what the delegated
    # role needs has declared that; the alternative, letting the child's role
    # widen the parent, is the leak this exists to stop.
    #
    # ``unattended`` rides along for the same reason the bound does: a child that
    # inherited a reach but not the approval for it would be a session whose every
    # permitted call is refused by a gate nobody is present to answer.
    #
    # Read through ``getattr`` because not every session shape is a real
    # ``Session`` (reduced test doubles and the resume path construct children
    # around hosts that predate this attribute).
    #
    # THE OP SCOPE RIDES ALONG WITH THE NAMES, and it is not an optional extra:
    # a declaration's reach is the names AND the ops a scoped name is cut down
    # to, so passing the names alone would hand a child the whole of a tool its
    # parent was declared not to have in full — ``agent sync`` and ``agent
    # reset`` behind a parent that may only author, which is the sentence above
    # ("a declared session cannot reach an excluded tool one hop down") being
    # false one hop down. Read the same way, and ``None`` for a parent that
    # declared no scope means "every op", exactly as it does on the parent.
    declared = getattr(parent_session, "_declared_tools", None)
    if declared is not None and hasattr(child, "set_tool_inventory"):
        child.set_tool_inventory(
            declared,
            unattended=bool(getattr(parent_session, "_declared_tools_unattended", False)),
            ops=getattr(parent_session, "_declared_tool_ops", None),
        )
    # Record the denial on the child so its OWN children inherit it (see the
    # ``restricted`` computation above). Set UNCONDITIONALLY, outside the
    # ``mcp is not None`` branch below: a child built with no MCP surface --
    # because this session had no manager wired yet -- can still delegate, and
    # its child resolves the manager off the session at that later point. Making
    # the stamp depend on whether MCP happened to be wired here would let the
    # boundary evaporate on exactly the path that reintroduces the surface.
    setattr(child, MCP_DENIED_ATTR, restricted)
    if target is not None:
        # Stamped facts, beside the denial and for the same reason: they are
        # properties of the LINEAGE that the child's own role cannot express,
        # and its grandchildren resolve their team and depth off them
        # (BEN-7-D1). ``active_team`` is set directly, never via
        # ``attach_team``: that writes ``team_brief`` into the child's system
        # tail and journals a sidecar per child (D1 (a), (c)).
        child.active_team = target.team
        child._team_lineage = tuple(target.team_lineage)
        child._delegation_depth = target.depth
        # THE DEPTH LANDS AFTER THE TOOLS WERE BUILT. The constructor's
        # capability merge rendered this child's ``task``/``agent`` at depth 0
        # (``Session._delegation_depth`` defaults to 0), i.e. WITH the tier
        # field under ``model_choice=model``. Re-render them now that the stamp
        # is in, or a nested child is advertised a picker the call-time gate
        # then refuses (see ``model_may_choose_tier`` for why it may not have
        # one). Only tools already in the inventory are replaced, so a child
        # whose ``task`` was pruned above stays without it; a no-op for the
        # ``operator`` default, where the field was already absent.
        if target.depth >= 1:
            child._rebuild_effort_tier_tools()
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
    #
    # What a child does NOT follow, by design (``Session._apply_config_change``
    # guards each on ``_job_id``): a ``hosting``/``model_name`` edit (its spec
    # was picked above and a mid-task switch costs its cache prefix), and the
    # ``web_*.enabled`` INVENTORY reconcile (re-adding would re-run the
    # allowlist and network-floor filtering for a short-lived run). The web
    # tools' per-call gate still refuses inside the child after a disable, and
    # the approval MODE reaches it through the parent's gate closure.
    try:
        from local_operator.config_watch import process_watcher

        watcher = process_watcher(config_dir())
        watcher.start(asyncio.get_running_loop())
        child.add_dispose_hook(watcher.subscribe(child._apply_config_change))
    except Exception:  # noqa: BLE001 — a child must build without the watcher
        logger.warning("config watcher could not be attached to the child", exc_info=True)
    await child.async_init()
    return child
