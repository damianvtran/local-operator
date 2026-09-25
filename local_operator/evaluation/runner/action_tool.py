"""The model-facing action tool of a loop-driven episode, and its one-shot token.

WHAT THIS IS. One ``AgentTool`` whose ``parameters`` are a **projection** of
:class:`~local_operator.evaluation.protocol.ActionBatch`: every field the model
actually chooses (the action kind, its frame, its coordinates, its text, its
keys, its deltas) stays, and every field the harness already knows -- the
protocol version, the task, the episode, and the observation each action is
bound to -- is removed, because the body injects them at execution time.

WHY A PROJECTION AND NOT A SECOND CONTRACT. ``public_reply_schema``'s
discipline, moved one level up: the schema is derived from the models on every
call, so a new action kind cannot be left out of the offer, and a bound cannot
drift from the validator that enforces it. The alternative -- a hand-written
function signature -- would be a second statement of the same contract, and the
one that is not executed is the one that goes stale.

WHY REMOVING THE IDENTITY FIELDS IS THE POINT. An ``observation_id`` in the
schema is an invitation to state a binding, and a stated binding can be STALE:
the model names a screen it was shown three turns ago and the harness must
refuse it. With the field absent the stale binding cannot be EXPRESSED at all,
which is a stronger property than validating it away -- there is no refusal
path to get wrong, and nothing in a bundle to mis-read as a real second step.
The same applies to the whole envelope: the body assembles the batch field by
field from the keys the schema offers, so a payload key the schema does not
offer (a ``task_id``, an ``episode_id``, a batch-level ``observation_id``)
cannot reach the batch at all.

WHAT THIS MODULE DELIBERATELY DOES NOT DO YET. Evidence writing (the
``action_batch``/``environment_step``/``observation`` triple), terminal-batch
handling (``finish``/``ask_user``), the step budget, and the adapter call's
read-back recovery are the episode driver's, and land in the stage that wires
this tool into a loop. Nothing calls this module in production today: it is
inert by design, and its tests drive it directly.

ISOLATION. This module lives in the runner package and imports no provider, no
config, no credentials, no session and no TUI, so the runner's startup-isolation
guarantee (``tests/unit/evaluation/runner/test_isolation.py``) still holds. The
tool is INJECTED into one episode's ``LoopContext``; it is never a
``TOOL_BUILDERS`` entry, which would cost every session a schema and a
predicate for a tool only an episode can invoke.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from pydantic import ValidationError

from local_operator.evaluation.action_surface import ActionSurface
from local_operator.evaluation.adapters.api import ExecuteResult
from local_operator.evaluation.protocol import (
    MAX_BATCH_SIZE,
    PROTOCOL_VERSION,
    ActionBatch,
    Observation,
)
from local_operator.evaluation.runner.provider_client import (
    REJECTION_CLASS_SECOND_BATCH,
    HintShape,
    classify_admission_error,
    classify_validation_error,
    rejection_hint,
    validation_diagnostic,
)
from local_operator.evaluation.runner.public_reply import (
    _inlined_action_schema,
    drop_sibling_action_fields,
)
from local_operator.harness.types import (
    FAULT_INVALID_ARGUMENTS,
    FAULT_KEY,
    AbortSignal,
    AgentEvent,
    AgentTool,
    AgentToolUpdate,
    Content,
    TextContent,
    ToolContext,
    ToolResult,
    TurnEndEvent,
)

#: The name the model calls. Named in the episode's own instruction block, so
#: it is a constant here rather than a literal spelled twice.
ACTION_TOOL_NAME = "apply_actions"

#: What the function tells the model it is for. Deliberately short: this rides
#: in the request's cache prefix on every call of every episode, and the action
#: vocabulary itself is the schema's job, not prose's.
ACTION_TOOL_DESCRIPTION = (
    "Apply one action batch to the observation you were just shown, and receive "
    "the state that results. Send exactly one call per turn."
)

#: The fields the BODY owns rather than the model. Removed from every projected
#: action, and never read from the payload: whatever the model sends under one
#: of these names is overwritten by the current observation's own value, which
#: is what makes a stale binding inexpressible rather than merely refused.
_INJECTED_ACTION_FIELDS = ("observation_id",)

#: The top-level key the reply ENVELOPE carries and this tool's parameters do
#: not. Listed rather than derived: the hint machinery below cannot see this
#: module's schema, and the two omissions together ARE the projection.
_ENVELOPE_ONLY_TOP_LEVEL_KEYS = ("public_observations",)

#: What this tool's contract removes from the shape a refusal states
#: (``HintShape``). The hint table is written for the envelope, so without this
#: a correction for a missing field would answer with the kind's "full shape"
#: including the ``observation_id`` the projection exists to make inexpressible
#: -- and the fallback example would re-advertise the envelope's own
#: ``public_observations``. Contradictory guidance at the moment the model is
#: being corrected is worse than no guidance, so the removals are stated once,
#: here, where the projection is defined.
_HINT_SHAPE = HintShape(
    omitted_action_fields=frozenset(_INJECTED_ACTION_FIELDS),
    omitted_top_level_keys=frozenset(_ENVELOPE_ONLY_TOP_LEVEL_KEYS),
)

#: The key ``ToolResult.details`` carries the refusal class under. The class
#: vocabulary itself is the shared classifier's (kept, not re-derived: sealed
#: bundles quote these keys verbatim and the canary notes read them off
#: artifacts), and the driver folds this key into its ``error`` event.
REJECTION_CLASS_KEY = "rejection_class"

#: Refusing a call that finds no observation to bind to. TWO DIFFERENT CAUSES
#: reach this branch and the sentence has to be true of BOTH: an earlier call in
#: this turn already ran its one batch and consumed the token, or the episode has
#: ended (``mark_terminal``, which the driver's terminal handling calls) and
#: there will never be another observation to bind to.
#:
#: The class stays the vocabulary's ``second-batch`` ("the competing batch" is
#: the defect this branch exists for, and a sealed bundle's class table has to
#: stay comparable), but it is stated as the constant that names it rather than
#: derived from this prose: a sentence wide enough to be true of both causes is
#: not evidence about which one it was, and a class the wording decides is the
#: defect the structured classifiers close.
_NO_PENDING_OBSERVATION_REFUSAL = (
    "no action batch was run: this call found no observation to bind to. Either "
    "this turn already ran its one action batch, or the episode has ended -- a "
    "batch is only ever bound to the screen you were shown, and only the batch "
    "that ran saw it. If a batch already ran this turn, read the observation "
    "that batch returned and reply with exactly one action batch for it."
)


def action_tool_parameters(surface: ActionSurface) -> dict[str, Any]:
    """The wire schema: ``ActionBatch`` projected down to what the model chooses.

    Derived on every call rather than cached, exactly as ``public_reply_schema``
    is and for the same reason: recomputing keeps a new action kind, or a moved
    bound, from drifting out of the offer. The surface filters the members, so
    the offer cannot advertise ``paste_text``/``ask_user`` to an adapter whose
    ``validate_batch`` would refuse them.

    The result is a FLATTENED schema -- ``anyOf`` over inlined member schemas,
    with no ``$ref``/``$defs``/``oneOf``/``discriminator`` -- because this object
    goes to a provider as a function's parameters and Gemini's
    ``FunctionDeclaration`` rejects all four with a 400 that would fail every
    request of a Gemini-routed episode rather than degrading. That flattening,
    including its "require the discriminator tag" fix-up, belongs to
    ``_inlined_action_schema`` and is imported rather than re-implemented: a
    second flattener is a second thing to keep in step with the models.

    The keyword set is the one the already-shipped reply schema puts on the wire
    (``properties``/``required``/``type``/``enum``/``const``/``anyOf``/bounds),
    plus ``minItems``/``maxItems`` for the array -- both documented members of
    the OpenAPI-3.0 subset Gemini's ``FunctionDeclaration`` accepts, which is
    why they are stated rather than left implied. Nothing here needs a paid
    probe to be safe: the shape is the shipped one, one property lighter.
    """
    # ``ActionBatch.actions`` declares ``min_length=1, max_length=MAX_BATCH_SIZE``;
    # the array bounds are stated from the protocol's own constant so the offer
    # cannot admit a batch the validator would refuse.
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["actions"],
        "properties": {
            "actions": {
                "type": "array",
                "minItems": 1,
                "maxItems": MAX_BATCH_SIZE,
                "items": {"anyOf": [_projected_action_schema(model) for model in surface.models]},
            }
        },
    }


def _projected_action_schema(model: Any) -> dict[str, Any]:
    """One action model's schema, with the body-owned fields taken out.

    The schema is built fresh (``_inlined_action_schema`` resolves the
    references into place on every call), so removing the injected fields here
    cannot mutate anything another caller holds.
    """
    schema = _inlined_action_schema(model)
    properties = schema.get("properties")
    if isinstance(properties, dict):
        for name in _INJECTED_ACTION_FIELDS:
            properties.pop(name, None)
    required = schema.get("required")
    if isinstance(required, list):
        schema["required"] = [name for name in required if name not in _INJECTED_ACTION_FIELDS]
    return schema


#: Applies one validated batch and returns the adapter's own result. The body
#: owns WHEN a batch runs; the caller owns the RPC semantics around it -- the
#: operation id, the timeouts, and the three-attempt read-back recovery that is
#: adapter contract rather than tool behaviour. It is injected for the same
#: reason the renderer is: both need state this module deliberately cannot see
#: (the driver's step counter, the evidence writer those retries are journalled
#: to, and the context builder's per-episode prefix).
ExecuteBatch = Callable[[ActionBatch], Awaitable[ExecuteResult]]

#: Turns one observation into the content the model reads as the tool's result:
#: a text rendering plus one image per frame. Injected because the only
#: renderer of an observation today is the context builder's, whose output
#: depends on the PREVIOUS turn (an unchanged screen is sent as a marker rather
#: than as the same paragraph again). A second renderer here would be a second
#: definition of what the model is shown, and the two would drift.
RenderObservation = Callable[[Observation], list[Content]]


class PendingObservationToken:
    """The one-shot binding of the next executed batch to the screen shown for it.

    THE FAILURE MODE THIS EXISTS FOR. Two DISTINCT calls to one tool in one turn
    both execute: the loop's only dedup is by call id, and ``_consume_claim``
    (``loop.py:587``) is duplicate-id event-suppression bookkeeping, not a
    competing-batch rule. Measured on the spike: ``tool_executions: 2``, both
    clean. Without a guard the second call would run against the observation the
    FIRST one produced, and the bundle would then read as two legal sequential
    steps -- a shape the verifier cannot tell from a real two-step turn, because
    its competing-batch rule keys on the model naming an already-batched
    ``observation_id`` and implicit binding removes exactly that evidence.

    WHY THIS IS A GUARANTEE AND NOT A VALIDATION. The token is armed only at the
    turn boundary, and consumed only by an execution. So every batch that runs
    is bound to an observation the model was actually shown -- by construction,
    not by a check that could be argued with. Two consequences are worth
    stating, because both are load-bearing:

    * a second call in the same turn finds no token and is REFUSED, never
      accepted (and the refusal is the model's own fault class: it emitted a
      turn the harness cannot dispatch, which is exactly the ``is_error``
      contract's "goes back to the model, never raised into the loop");
    * a call whose arguments fail validation returns BEFORE consuming, so the
      corrective re-ask is not blocked by the very rejection it is correcting.

    ``pending`` only ever READS and ``consume_pending`` is the only mutation, and
    neither awaits, so the validate-then-consume window cannot interleave with a
    sibling call; the tool is also ``exclusive``, which keeps two calls from
    running concurrently in the first place.

    ``TurnEndEvent`` is delivered before the next model call, so re-arming here
    cannot race the gate that reads the token.
    """

    def __init__(self, observation: Observation) -> None:
        # Armed with observation zero at construction: the episode's first turn
        # is the one the reset produced, and there is no earlier turn boundary
        # to arm it at.
        self._pending: Observation | None = observation
        self._in_flight: Observation | None = None
        self._terminal = False

    @property
    def pending(self) -> Observation | None:
        """The observation a batch may be bound to, or ``None`` when one ran."""
        return self._pending

    @property
    def terminal(self) -> bool:
        """Whether the episode has ended (a ``finish`` batch, or the step cap)."""
        return self._terminal

    def consume_pending(self) -> Observation | None:
        """Take the pending observation, leaving no token behind.

        The one mutation in the arming cycle. Returning ``None`` is the refusal:
        the caller must not execute, because whatever it did would be bound to a
        screen nobody was shown.
        """
        observation, self._pending = self._pending, None
        return observation

    def record_in_flight(self, observation: Observation) -> None:
        """Hold the observation an executed batch produced until the turn ends.

        Held rather than armed immediately: arming mid-turn is precisely the
        hole this class closes, since it would let a second call in the SAME
        turn bind to a screen the model has not been given yet.
        """
        self._in_flight = observation

    def mark_terminal(self) -> None:
        """End the episode: no observation is pending again, ever.

        Called by the driver's terminal handling (a ``finish`` batch, or an ask
        that went unanswered), which is the stage that wires this tool into a
        loop; nothing in this module calls it yet, and the gate above reads the
        flag it sets.

        The contract, which ``test_mark_terminal_ends_the_episode`` pins because
        the stage that calls this will assume all of it: the token is left with
        nothing pending; a call that arrives anyway is refused with NO execution
        (the same ``second-batch`` class -- its sentence is written to be true of
        an ended episode as well as of a second call in one turn); and a later
        ``fold`` cannot re-arm, so no executed batch's output can hand the
        episode a screen to act on after it has ended.
        """
        self._terminal = True
        self._pending = None

    def fold(self, event: AgentEvent) -> None:
        """Arm the next turn's token from the batch this turn executed."""
        if self._terminal or not isinstance(event, TurnEndEvent):
            return
        # No execution this turn (a prose reply, a corrective re-ask) leaves the
        # token exactly as it was: it was never consumed, so there is nothing to
        # re-arm and the re-ask still has a screen to answer for.
        if self._in_flight is not None:
            self._pending, self._in_flight = self._in_flight, None

    def before_model_call(self) -> bool:
        """The loop's gate: call the model only while an observation is pending.

        ``False`` ends the run as ``AgentEndEvent(aborted=True, error="stopped by
        gate")`` -- which is why the episode's terminal decision must come from
        this object's own state and never from ``aborted``. A ``finish`` batch
        ends an episode NORMALLY, and the loop still reports it as aborted.
        """
        return self._pending is not None and not self._terminal


def build_action_tool(
    token: PendingObservationToken,
    *,
    surface: ActionSurface,
    execute: ExecuteBatch,
    render: RenderObservation,
) -> AgentTool:
    """The episode's action tool: projection, body, and the refusal contract.

    INJECTED, never registered. It exists inside one episode's ``LoopContext``
    and nowhere else, so it costs every other session zero schema, zero
    predicate and zero import. ``execute`` and ``render`` are the two seams the
    stage that wires this into a loop fills: the adapter call with its read-back
    recovery, and the model-facing rendering of an observation.
    """

    async def run(
        tool_call_id: str,
        arguments: dict[str, Any],
        signal: AbortSignal | None,
        on_update: Callable[[AgentToolUpdate], None] | None,
        context: ToolContext,
    ) -> ToolResult:
        del signal, on_update, context  # no approval flow, no streaming, no cwd
        pending = token.pending
        if pending is None:
            return _no_pending_refusal(tool_call_id)

        # VALIDATE BEFORE CONSUMING. A rejected call must not burn the token, or
        # the corrective re-ask the rejection asks for would find no screen to
        # answer against. The harness cannot do this for us: its argument
        # validator checks top-level names and scalar types only, and every
        # nested malformation reaches a tool body intact.
        try:
            batch = _build_batch(arguments, pending)
        except ValidationError as error:
            # The CLASS comes from the error's structured entries and the TEXT
            # from a rendering that keeps no value the model supplied. Deriving
            # the class from ``str(error)`` would make the payload's own bytes an
            # input to an ordered substring test, so a model could name its own
            # refusal class by putting a marker phrase in a field, and, for a
            # preserved class, be handed back the rendering this module promises
            # never reaches it. The envelope path's own raise sites classify the
            # same way, so one class table covers both.
            return _refusal(
                tool_call_id,
                validation_diagnostic(error),
                class_key=classify_validation_error(error),
                observation=pending,
                surface=surface,
            )
        try:
            # All three of the protocol's own checks live here rather than
            # anywhere in this module: the batch binds to this observation, each
            # coordinate is inside the model-visible frame it names, and the
            # negotiated surface admits every action. Coordinate semantics are
            # never re-derived -- there is one implementation and this is a call
            # to it.
            batch.validate_for(pending)
            surface.validate_batch(batch)
        except ValueError as error:
            # ``ActionAdmissionError`` is a ValueError, and so is ``validate_for``'s
            # refusal: both are this batch's failure against this observation.
            # The class is read from the exception rather than from its message:
            # ``validate_for`` interpolates ``action.frame_id`` into its own
            # sentence, so a message-based class is one the model can choose.
            return _refusal(
                tool_call_id,
                f"action batch does not match this observation: {error}",
                class_key=classify_admission_error(error),
                observation=pending,
                surface=surface,
            )

        observation = token.consume_pending()
        if observation is None:
            # Unreachable while the two reads above are free of ``await`` (the
            # tool is exclusive and nothing can interleave between them), but
            # the refusal is cheaper than trusting that, and the alternative is
            # executing against a screen nobody was shown.
            return _no_pending_refusal(tool_call_id)

        result = await execute(batch)
        token.record_in_flight(result.observation)
        return ToolResult(
            tool_call_id=tool_call_id,
            tool_name=ACTION_TOOL_NAME,
            content=render(result.observation),
            details={"receipt": result.receipt.model_dump(mode="json")},
        )

    return AgentTool(
        name=ACTION_TOOL_NAME,
        label="Environment action",
        description=ACTION_TOOL_DESCRIPTION,
        parameters=action_tool_parameters(surface),
        # It mutates a machine outside this process, so it is the top tier and
        # it must never share a concurrency batch: the token makes a second call
        # in a turn impossible, and exclusivity keeps two calls from interleaving
        # mid-flight on the way there.
        approval_tier="exec",
        concurrency="exclusive",
        interruptible=False,
        # Not a capability a user can invoke: an episode's own model-facing
        # surface, which no interactive tool list should ever offer.
        hidden=True,
        execute=run,
    )


def _build_batch(arguments: Mapping[str, Any], observation: Observation) -> ActionBatch:
    """Compile the model's offer into the protocol's own batch model.

    Assembled FIELD BY FIELD from the keys the schema offers, never copied
    wholesale: the identity comes from the episode and the observation, and each
    action is the model's own mapping with its ``observation_id`` overwritten.
    A key the schema does not offer therefore cannot reach the batch -- which is
    the same property the absence of those keys from the schema buys, stated a
    second time at the one boundary where a non-compliant client could ignore
    the schema and send them anyway.

    The one thing read OUT of an action is the field the reply tolerance drops
    (``drop_sibling_action_fields``): a field belonging to a SIBLING action kind
    states nothing the required ``kind`` tag did not already state, and the prose
    path drops it for the same reason and with the same report. Applied here too
    because the two channels converge on this one validated structure, and a
    drop that ran on only one of them would refuse a decision the other accepts.

    Scoped to the drop: this function reads an action ARRAY, because that is
    what the offered call's parameters declare (``public_reply_schema``). The
    prose channel additionally accepts an ``actions`` value that is a
    JSON-encoded STRING (``_actions_from_json_string``), and that tolerance is
    deliberately NOT mirrored here -- a call whose argument contradicts the type
    the channel offered it is refused, while the same bytes arriving as prose
    are read. Widening this channel's accepted spellings is a contract decision
    of its own rather than part of reading a batch's actions.
    """
    raw_actions = arguments.get("actions")
    actions: Any = raw_actions
    if isinstance(raw_actions, list):
        actions = [
            (
                {**action, "observation_id": observation.observation_id}
                if isinstance(action, Mapping)
                else action
            )
            for action in raw_actions
        ]
        actions, _tolerated_fields = drop_sibling_action_fields(actions)
    return ActionBatch.model_validate(
        {
            "protocol_version": PROTOCOL_VERSION,
            "task_id": observation.task_id,
            "episode_id": observation.episode_id,
            "observation_id": observation.observation_id,
            "actions": actions,
        },
        strict=True,
    )


def _refusal(
    tool_call_id: str,
    reason: str,
    *,
    class_key: str,
    observation: Observation,
    surface: ActionSurface,
) -> ToolResult:
    """One model-recoverable refusal, in the runner's existing two halves.

    The CLASS is what sealed bundles quote and what the canary notes count, and
    it is passed IN rather than derived here from ``reason``: the callers derive
    it from structured evidence (``classify_validation_error`` for a Pydantic
    failure, ``classify_admission_error`` for the protocol's or the surface's
    own refusal), because a reason string carries bytes the MODEL supplied. The
    vocabulary is still the envelope path's -- ``classify_rejection``'s -- so a
    sealed bundle's class table stays comparable across the two paths; what is
    not shared is the habit of reading it out of prose.

    The MESSAGE is ``rejection_hint``'s, which states the accepted shape for the
    class instead of the validator's rendering: a pydantic ``str()`` embeds
    ``input_value=<head>…<tail>`` and a docs URL, and the model needs the shape,
    not the refused bytes echoed back at it. The hint is built with
    ``_HINT_SHAPE`` so it states THIS tool's contract -- the projection, not the
    envelope it was projected from.

    The fault marker is the model's own. ``is_error`` alone would classify as
    ``execution`` at ``AgentLoop._classify_fault`` -- the harness's and the
    world's bucket -- which is precisely the laundering
    ``InvalidToolArgumentsError`` documents: this call never reached the adapter,
    and the model could have known from the schema (or from the observation it
    was shown) that the arguments were wrong.
    """
    return ToolResult(
        tool_call_id=tool_call_id,
        tool_name=ACTION_TOOL_NAME,
        is_error=True,
        content=[
            TextContent(
                text=rejection_hint(
                    class_key,
                    reason=reason,
                    observation=observation,
                    surface=surface,
                    shape=_HINT_SHAPE,
                )
            )
        ],
        details={FAULT_KEY: FAULT_INVALID_ARGUMENTS, REJECTION_CLASS_KEY: class_key},
    )


def _no_pending_refusal(tool_call_id: str) -> ToolResult:
    """The refusal of a call that finds no observation to bind to.

    The class is ``second-batch`` -- the vocabulary's own key for this defect
    family, kept so a sealed bundle's counts stay comparable -- and the sentence
    is ``_NO_PENDING_OBSERVATION_REFUSAL``, which is true of both causes that
    reach here. It is a PRESERVED hint (``provider_client._PRESERVED_HINTS``),
    so the sentence IS the correction and is stated once, above.
    """
    return ToolResult(
        tool_call_id=tool_call_id,
        tool_name=ACTION_TOOL_NAME,
        is_error=True,
        content=[TextContent(text=_NO_PENDING_OBSERVATION_REFUSAL)],
        details={
            FAULT_KEY: FAULT_INVALID_ARGUMENTS,
            REJECTION_CLASS_KEY: REJECTION_CLASS_SECOND_BATCH,
        },
    )
