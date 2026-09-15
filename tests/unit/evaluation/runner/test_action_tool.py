"""Stage 2 of the loop convergence: the action tool, offline.

The tool is inert in production -- nothing builds it outside these tests -- so
what is asserted here is the contract the later stages will rely on, and in the
order they will rely on it:

(a) the projection: every action kind the negotiated surface admits is offered,
    no identity field survives it, and the protocol still validates the batch
    the body compiles from a projected payload;
(b) implicit binding: a stale ``observation_id`` cannot be expressed, because
    the field is not in the schema AND the body assembles the batch only from
    the keys the schema offers -- and a stale ``frame_id``, which stays a model
    choice, is refused against the observation actually in force;
(c) the competing batch: two calls in one turn produce exactly one execution
    and one refusal, driven through the REAL ``AgentLoop`` against the
    in-process ``FakeAdapter`` with a scripted stream;
(d) the wire shape: no ``$ref``/``$defs``/``oneOf``/``discriminator``, which
    Gemini's ``FunctionDeclaration`` rejects with a 400;
(e) the refusals: each malformation is a model fault (never an ``execution``
    fault, which would launder it out of the accuracy figure), and a tool body
    that RAISES is not claimed as one.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from local_operator.evaluation.action_surface import ActionSurface
from local_operator.evaluation.adapters.api import (
    ExecuteResult,
    ExecutionReceipt,
    observation_content_id,
)
from local_operator.evaluation.adapters.supervisor import (
    HostVerifier,
    VerifiedAdapterSession,
)
from local_operator.evaluation.evidence.models import canonical_digest
from local_operator.evaluation.protocol import (
    MAX_BATCH_SIZE,
    ActionBatch,
    ArtifactRef,
    FrameGeometry,
    FrameRef,
    FrameSize,
    Observation,
)
from local_operator.evaluation.runner.action_tool import (
    ACTION_TOOL_NAME,
    REJECTION_CLASS_KEY,
    PendingObservationToken,
    action_tool_parameters,
    build_action_tool,
)
from local_operator.harness.loop import AgentLoop, LoopConfig, LoopContext
from local_operator.harness.types import (
    FAULT_INVALID_ARGUMENTS,
    FAULT_KEY,
    AgentEndEvent,
    AgentEvent,
    AgentTool,
    ChatRequest,
    Content,
    Message,
    ModelSpec,
    StreamEndEvent,
    StreamEvent,
    StreamTextDelta,
    StreamToolCallDelta,
    TextContent,
    ToolContext,
    ToolExecutionEndEvent,
    ToolResult,
    TurnEndEvent,
)
from tests.unit.evaluation.runner.conftest import TASK_ID as CONFTEST_TASK_ID
from tests.unit.evaluation.runner.conftest import FakeAdapter

EPISODE = "episode-action-tool"
TASK = "task-action-tool"
MODEL = ModelSpec(provider="test", model_id="m")

#: The surface an episode negotiates when its adapter advertises neither
#: ``paste_text`` nor a lossy keyboard: the default, and the one the projection
#: is filtered by in every test unless it says otherwise.
SURFACE = ActionSurface()


def _observation(sequence: int = 0, *, frames: tuple[FrameRef, ...] = ()) -> Observation:
    provisional = Observation(
        task_id=TASK,
        episode_id=EPISODE,
        sequence=sequence,
        observation_id="provisional",
        text=f"state-{sequence}",
        frames=frames,
    )
    return provisional.model_copy(update={"observation_id": observation_content_id(provisional)})


def _frame(frame_id: str, *, width: int = 100, height: int = 100) -> FrameRef:
    return FrameRef(
        frame_id=frame_id,
        artifact=ArtifactRef(sha256="ab" * 32, media_type="image/png", byte_count=16),
        geometry=FrameGeometry(
            native=FrameSize(width=width, height=height),
            model_visible=FrameSize(width=width, height=height),
        ),
    )


def _render(observation: Observation) -> list[Content]:
    """The injected renderer, standing in for the context builder's."""
    return [TextContent(text=f"rendered:{observation.observation_id}")]


class _RecordingExecute:
    """The injected adapter call: one recorded entry per EXECUTION.

    The count is the (c) assertion's subject, so it is recorded here rather than
    inferred from tool results -- a refused call produces a tool result too, and
    the two are exactly what must not be confused.
    """

    def __init__(self, outputs: list[Observation] | None = None) -> None:
        self.batches: list[ActionBatch] = []
        self._outputs = list(outputs or [])

    async def __call__(self, batch: ActionBatch) -> ExecuteResult:
        self.batches.append(batch)
        observation = self._outputs.pop(0) if self._outputs else _observation(len(self.batches))
        return ExecuteResult(
            observation=observation,
            receipt=ExecutionReceipt(
                operation_id=f"exec-{len(self.batches)}",
                action_batch_id=canonical_digest("adapter-action-batch-v1", batch),
                input_observation_id=batch.observation_id,
                output_observation_id=observation.observation_id,
                sequence=observation.sequence,
            ),
        )


class _Offline:
    """The tool with no loop: the body driven as the loop would drive it."""

    def __init__(
        self,
        observation: Observation,
        *,
        surface: ActionSurface = SURFACE,
        outputs: list[Observation] | None = None,
    ) -> None:
        self.token = PendingObservationToken(observation)
        self.execute = _RecordingExecute(outputs)
        self.tool = build_action_tool(
            self.token, surface=surface, execute=self.execute, render=_render
        )

    async def call(self, arguments: dict[str, Any], *, call_id: str = "call-1") -> ToolResult:
        return await self.tool.execute(call_id, arguments, None, None, ToolContext())


class _ScriptedStream:
    """A fake ``stream_fn`` replaying one scripted event list per call."""

    def __init__(self, turns: list[list[StreamEvent]]) -> None:
        self.turns = turns
        self.requests: list[ChatRequest] = []

    def __call__(self, request: ChatRequest, signal: Any):
        self.requests.append(request)
        turn = self.turns[len(self.requests) - 1]

        async def gen():
            for event in turn:
                yield event

        return gen()


def _call_delta(index: int, arguments: dict[str, Any], *, id: str) -> StreamToolCallDelta:
    return StreamToolCallDelta(
        index=index, id=id, name=ACTION_TOOL_NAME, argument_delta=json.dumps(arguments)
    )


def _loop_config(stream: _ScriptedStream, token: PendingObservationToken) -> LoopConfig:
    return LoopConfig(
        model=MODEL,
        convert_to_llm=lambda messages: [m for m in messages if isinstance(m, Message)],
        stream_fn=stream,
        before_model_call=token.before_model_call,
    )


async def _drive(
    tool: AgentTool, token: PendingObservationToken, stream: _ScriptedStream
) -> list[AgentEvent]:
    """Run one episode-shaped loop, folding the events as the driver will.

    FOLDING IS THE DRIVER'S JOB, and it is done here because the loop does not
    know the token exists: the arming happens on ``TurnEndEvent`` and the gate
    (``before_model_call``) reads the result, so a run that never folds stops
    at its own gate after the first batch. That is also why the terminal
    decision has to come from the driver's own state -- the gate ends an
    episode NORMALLY (a ``finish`` batch) as ``aborted=True``.
    """
    context = LoopContext(system_blocks=["sys"], tools=[tool])
    loop = AgentLoop()
    events: list[AgentEvent] = []
    async for event in loop.run([Message.user("go")], context, _loop_config(stream, token), None):
        token.fold(event)
        events.append(event)
    return events


def _tool_results(events: list[AgentEvent]) -> list[ToolResult]:
    return [e.result for e in events if isinstance(e, ToolExecutionEndEvent)]


def _schema_keys(node: Any) -> set[str]:
    """Every key name appearing anywhere in a JSON Schema object."""
    if isinstance(node, dict):
        found = set(node)
        for value in node.values():
            found |= _schema_keys(value)
        return found
    if isinstance(node, list):
        found: set[str] = set()
        for item in node:
            found |= _schema_keys(item)
        return found
    return set()


def _offered_kinds(schema: dict[str, Any]) -> set[str]:
    members = schema["properties"]["actions"]["items"]["anyOf"]
    return {member["properties"]["kind"]["const"] for member in members}


# ---------------------------------------------------------------------------
# (a) the projection
# ---------------------------------------------------------------------------


def test_every_admitted_action_kind_is_offered_and_no_other() -> None:
    """The offer is exactly the negotiated ``surface.models``, derived not typed.

    A hand-written signature would drift the moment an action kind is added;
    this pins the derivation in both directions, so neither a missing kind nor
    an advertised-but-refused one can pass.
    """
    for surface in (
        SURFACE,
        ActionSurface(paste_text=True),
        ActionSurface(ask_user=False),
    ):
        expected = {model.model_fields["kind"].default for model in surface.models}
        assert _offered_kinds(action_tool_parameters(surface)) == expected


def test_the_projection_drops_every_identity_field() -> None:
    """Nothing the body injects survives into the model-facing schema.

    The point is not hygiene: an ``observation_id`` in the schema is an
    invitation to STATE a binding, and a stated binding can be stale. Absent
    from the offer, it cannot be expressed at all.
    """
    schema = action_tool_parameters(SURFACE)
    assert "observation_id" not in json.dumps(schema)
    assert set(schema["properties"]) == {"actions"}
    assert schema["required"] == ["actions"]
    assert schema["properties"]["actions"]["maxItems"] == MAX_BATCH_SIZE
    for member in schema["properties"]["actions"]["items"]["anyOf"]:
        assert "observation_id" not in member["properties"]
        assert "observation_id" not in member["required"]
        # Every other field of the action model is still there: this is a
        # projection, not an omission.
        assert "kind" in member["properties"]


def test_the_projection_is_derived_anew_and_matches_the_tool() -> None:
    """The tool's own parameters ARE the projection, not a cached copy of it."""
    offline = _Offline(_observation())
    assert offline.tool.parameters == action_tool_parameters(SURFACE)


def test_the_projection_keeps_every_other_action_field() -> None:
    """Exactly the identity field goes, read against the MODELS rather than the schema.

    The oracle is each action model's own field list, not the flattened schema
    the projection is built from -- otherwise a field dropped from both would
    cancel out. Bounds stay with the fields that carry them, which is the whole
    reason the projection is derived instead of typed out.
    """
    schema = action_tool_parameters(SURFACE)
    members = {
        member["properties"]["kind"]["const"]: member
        for member in schema["properties"]["actions"]["items"]["anyOf"]
    }
    for model in SURFACE.models:
        member = members[model.model_fields["kind"].default]
        declared = set(model.model_fields)
        assert set(member["properties"]) == declared - {"observation_id"}
        required = {name for name, field in model.model_fields.items() if field.is_required()}
        assert set(member["required"]) >= required - {"observation_id"}
        assert "observation_id" not in member["required"]
        for name, property_schema in member["properties"].items():
            if name in {"kind", "frame_id"}:
                continue
            assert property_schema, f"{name} lost its constraints in the projection"
    # And the bounds the array itself carries come from the batch model.
    assert schema["properties"]["actions"]["minItems"] == 1


@pytest.mark.asyncio
async def test_the_protocol_still_validates_what_the_body_compiles() -> None:
    """A projected payload compiles into a batch ``ActionBatch`` accepts.

    Driven from the payload the SCHEMA describes, so a projection that dropped
    a required field would fail here rather than at the first paid episode.
    """
    pending = _observation(frames=(_frame("screen"),))
    offline = _Offline(pending)
    result = await offline.call(
        {"actions": [{"kind": "click", "frame_id": "screen", "x": 1, "y": 2}]}
    )
    assert not result.is_error, result.content
    batch = offline.execute.batches[0]
    assert isinstance(batch, ActionBatch)
    assert batch.observation_id == pending.observation_id
    assert batch.task_id == TASK
    assert batch.protocol_version == "1.0"
    # The identity the body injected is the protocol's own: re-validating the
    # batch unchanged must succeed under the same strict mode it was built in.
    assert ActionBatch.model_validate(batch.model_dump(mode="json"), strict=True) == batch


# ---------------------------------------------------------------------------
# (b) implicit binding
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_stale_observation_id_cannot_be_expressed() -> None:
    """The field is inert: supplied or not, the batch binds to the live screen.

    The payload below is what a client that ignored the schema would send --
    the whole envelope, plus a per-action id naming an observation three turns
    old. None of it reaches the batch, because the body assembles the batch
    from the keys the schema offers and overwrites the rest.
    """
    current = _observation(3)
    stale = _observation(0)
    offline = _Offline(current)
    result = await offline.call(
        {
            "protocol_version": "1.0",
            "task_id": stale.task_id,
            "episode_id": stale.episode_id,
            "observation_id": stale.observation_id,
            "actions": [
                {
                    "kind": "wait",
                    "observation_id": stale.observation_id,
                    "duration_ms": 5,
                }
            ],
        }
    )
    assert not result.is_error, result.content
    batch = offline.execute.batches[0]
    assert batch.observation_id == current.observation_id
    assert batch.actions[0].observation_id == current.observation_id


@pytest.mark.asyncio
async def test_a_stale_frame_id_is_refused_by_the_protocol() -> None:
    """``frame_id`` stays a model choice, so it is the one binding checked.

    It cannot resolve against a previous observation -- the body never holds
    one -- but it can name a frame the current observation does not carry, and
    that is refused. The refusal must NOT consume the token: the corrective
    call it asks for still has a screen to answer for.
    """
    current = _observation(4, frames=(_frame("screen"),))
    offline = _Offline(current)
    refused = await offline.call(
        {"actions": [{"kind": "click", "frame_id": "screen-2", "x": 1, "y": 2}]}
    )
    assert refused.is_error
    assert (refused.details or {})[REJECTION_CLASS_KEY] == "unknown-frame-id"
    assert offline.execute.batches == []
    assert offline.token.pending is current  # the token survived the refusal

    corrected = await offline.call(
        {"actions": [{"kind": "click", "frame_id": "screen", "x": 1, "y": 2}]},
        call_id="call-2",
    )
    assert not corrected.is_error
    assert len(offline.execute.batches) == 1


# ---------------------------------------------------------------------------
# (c) the competing batch -- through the real loop and the real fake adapter
# ---------------------------------------------------------------------------


def _session(adapter: FakeAdapter) -> VerifiedAdapterSession:
    # The fake adapter builds its observations from the conftest's own task id,
    # so the parent verifier has to agree with it rather than with this
    # module's constants.
    verifier = HostVerifier(CONFTEST_TASK_ID, EPISODE, adapter.tmp_path)
    verifier.accept_initial(adapter.current)
    # The fake speaks the protocol at the ``_call_raw`` seam rather than
    # subclassing the supervisor, exactly as ``test_supervisor``'s raw doubles
    # do -- so the real ``VerifiedAdapterSession`` still runs.
    return VerifiedAdapterSession(adapter, verifier)  # type: ignore[arg-type]


def _adapter_execute(session: VerifiedAdapterSession):
    """The driver's half of the seam, as the episode path spells it today."""

    async def execute(batch: ActionBatch) -> ExecuteResult:
        from local_operator.evaluation.adapters.api import ExecuteParams

        return await session.execute(
            ExecuteParams(
                operation_id=f"exec-{batch.observation_id}",
                action_batch=batch,
                action_batch_id=canonical_digest("adapter-action-batch-v1", batch),
            ),
            timeout=5.0,
        )

    return execute


@pytest.mark.asyncio
async def test_two_calls_in_one_turn_execute_once_and_refuse_once(
    tmp_path: Any,
) -> None:
    """The guarantee, driven where it has to hold: the loop itself.

    Two DISTINCT calls to one tool in one turn both execute today
    (``_consume_claim`` is duplicate-call-id event bookkeeping, not a
    competing-batch rule). With identity injected the second would run against
    the observation the first produced, and the bundle would read as two legal
    sequential steps the verifier cannot tell apart. The token is what makes
    that inexpressible: the first call consumes it, the second finds none and
    is refused, and exactly one adapter ``execute`` is billed.
    """
    adapter = FakeAdapter(tmp_path, EPISODE)
    session = _session(adapter)
    pending = adapter.current
    token = PendingObservationToken(pending)
    tool = build_action_tool(
        token, surface=SURFACE, execute=_adapter_execute(session), render=_render
    )
    batch_args = {"actions": [{"kind": "wait", "duration_ms": 5}]}
    stream = _ScriptedStream(
        [
            [
                _call_delta(0, batch_args, id="c1"),
                _call_delta(1, batch_args, id="c2"),
                StreamEndEvent(stop_reason="toolUse"),
            ],
            [StreamTextDelta(delta="done"), StreamEndEvent(stop_reason="stop")],
        ]
    )

    events = await _drive(tool, token, stream)

    assert adapter.calls.count("execute") == 1
    results = _tool_results(events)
    assert [result.is_error for result in results] == [False, True]
    refusal = results[1]
    assert (refusal.details or {})[REJECTION_CLASS_KEY] == "second-batch"
    assert (refusal.details or {})[FAULT_KEY] == FAULT_INVALID_ARGUMENTS
    assert "second action batch" in refusal.content[0].text  # type: ignore[union-attr]
    # The executed batch was bound to the screen the model was shown; the
    # refused one was bound to nothing and never reached the adapter.
    assert results[0].details is not None
    assert results[0].details["receipt"]["input_observation_id"] == pending.observation_id


@pytest.mark.asyncio
async def test_the_token_rearms_at_the_turn_boundary_not_before(tmp_path: Any) -> None:
    """A batch's own output arms the NEXT turn's token, and only at ``turn_end``.

    Arming any earlier -- at execution -- is the hole the token exists to close:
    a second call in the same turn would then bind to a screen the model has
    not been given yet. After the run the token is armed with the observation
    the executed batch produced, which is what the next turn answers for.
    """
    adapter = FakeAdapter(tmp_path, EPISODE)
    session = _session(adapter)
    token = PendingObservationToken(adapter.current)
    tool = build_action_tool(
        token, surface=SURFACE, execute=_adapter_execute(session), render=_render
    )
    stream = _ScriptedStream(
        [
            [
                _call_delta(0, {"actions": [{"kind": "wait", "duration_ms": 5}]}, id="c1"),
                StreamEndEvent(stop_reason="toolUse"),
            ],
            [StreamTextDelta(delta="done"), StreamEndEvent(stop_reason="stop")],
        ]
    )

    events = await _drive(tool, token, stream)

    armings = [event for event in events if isinstance(event, TurnEndEvent)]
    assert len(armings) == 2
    assert token.pending is not None
    assert token.pending.sequence == 1  # the batch's own output, not obs 0
    assert adapter.calls.count("execute") == 1
    assert isinstance(events[-1], AgentEndEvent)


# ---------------------------------------------------------------------------
# (d) the wire shape
# ---------------------------------------------------------------------------


def test_the_wire_schema_is_flat() -> None:
    """``$ref``/``$defs``/``oneOf``/``discriminator`` are 400-rejected by Gemini.

    That would fail EVERY request of a Gemini-routed episode rather than
    degrading, which is why the discriminated union is flattened into
    ``anyOf`` over concrete member schemas. The tag is required in each member
    for the same reason: under a bare ``anyOf`` a batch omitting ``kind`` would
    satisfy whichever member matched on its other fields, admitting at the
    schema what the validator refuses.
    """
    for surface in (SURFACE, ActionSurface(paste_text=True)):
        schema = action_tool_parameters(surface)
        keys = _schema_keys(schema)
        assert not keys & {"$ref", "$defs", "oneOf", "discriminator"}
        assert "anyOf" in keys
        assert "kind" in json.dumps(schema)
        for member in schema["properties"]["actions"]["items"]["anyOf"]:
            assert "kind" in member["required"]


# ---------------------------------------------------------------------------
# (e) the refusals: a model fault, never an execution fault
# ---------------------------------------------------------------------------


#: One malformation per class the vocabulary names, each otherwise sent in the
#: schema's own shape. ``class_key`` is what a sealed bundle would quote, and it
#: is the SHARED classifier's answer -- the point of the table is that the tool
#: path buckets a defect exactly as the envelope path did.
MALFORMED = [
    ("empty-actions", {"actions": []}, "field-invalid"),
    ("missing-required-field", {"actions": [{"kind": "wait"}]}, "field-invalid"),
    ("unknown-kind", {"actions": [{"kind": "drag", "x": 1}]}, "unknown-action-kind"),
    (
        "extra-action-key",
        {"actions": [{"kind": "wait", "duration_ms": 5, "noun": "x"}]},
        "extra-action-key",
    ),
    ("wrong-field-type", {"actions": [{"kind": "type", "text": 7}]}, "field-invalid"),
    (
        "keys-not-array",
        {"actions": [{"kind": "key", "keys": {"item": ["CTRL"]}}]},
        "keys-not-array",
    ),
    ("unknown-key-name", {"actions": [{"kind": "key", "keys": ["NOSUCH"]}]}, "unknown-key"),
    (
        "terminal-with-a-second-action",
        {
            "actions": [
                {"kind": "finish", "status": "done", "reason": "r"},
                {"kind": "wait", "duration_ms": 1},
            ]
        },
        "field-invalid",
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("name", "arguments", "expected_class"), MALFORMED, ids=[case[0] for case in MALFORMED]
)
async def test_a_malformed_batch_is_a_model_fault_and_burns_no_token(
    name: str, arguments: dict[str, Any], expected_class: str
) -> None:
    """The harness cannot be the validator, so the body is -- and says whose fault it is.

    ``validate_tool_arguments`` checks top-level names and scalar types only;
    every nested malformation here reaches the body intact, so an unvalidated
    body would hand it to the adapter. The result must be ``is_error`` with the
    model-fault marker: ``is_error`` alone classifies as ``execution`` at
    ``AgentLoop._classify_fault``, which is the bucket for the world failing and
    would launder this out of the accuracy figure.
    """
    pending = _observation(2, frames=(_frame("screen"),))
    offline = _Offline(pending)
    result = await offline.call(arguments)

    assert result.is_error, name
    assert (result.details or {})[FAULT_KEY] == FAULT_INVALID_ARGUMENTS
    assert (result.details or {})[REJECTION_CLASS_KEY] == expected_class
    assert offline.execute.batches == []
    assert offline.token.pending is pending
    text = result.content[0].text  # type: ignore[union-attr]
    # The correction states the accepted shape. It must never quote pydantic's
    # rendering of the refused value back at the model.
    assert "input_value=" not in text
    assert "https://errors.pydantic.dev" not in text


@pytest.mark.asyncio
async def test_an_out_of_frame_coordinate_is_refused_with_its_bound() -> None:
    """Coordinate semantics are not re-derived here: the protocol's own check runs.

    The hint appends the frame's bounds, because "outside the frame" does not
    tell a model which pixels are inside it.
    """
    pending = _observation(1, frames=(_frame("screen", width=100, height=50),))
    offline = _Offline(pending)
    result = await offline.call(
        {"actions": [{"kind": "click", "frame_id": "screen", "x": 500, "y": 10}]}
    )
    assert result.is_error
    assert (result.details or {})[REJECTION_CLASS_KEY] == "out-of-frame-coordinate"
    text = result.content[0].text  # type: ignore[union-attr]
    assert "100" in text and "50" in text


@pytest.mark.asyncio
async def test_an_adapter_refused_action_is_a_model_fault() -> None:
    """The negotiated surface is part of the offer, so its refusal is a model fault.

    ``type`` of 300 characters is well-formed JSON against the schema and past
    this adapter's keyboard deadline; ``ActionSurface`` already phrases the
    refusal with the alternative it CAN carry (``paste_text``), which is the
    correction the model needs.
    """
    pending = _observation()
    offline = _Offline(pending, surface=ActionSurface(max_type_chars=10))
    result = await offline.call({"actions": [{"kind": "type", "text": "x" * 300}]})

    assert result.is_error
    assert (result.details or {})[FAULT_KEY] == FAULT_INVALID_ARGUMENTS
    assert (result.details or {})[REJECTION_CLASS_KEY] == "adapter-capability"
    assert offline.token.pending is pending


@pytest.mark.asyncio
async def test_a_raising_tool_body_is_not_claimed_as_a_model_fault() -> None:
    """The contrast that makes the marker meaningful.

    A body that raises is returned to the model as an ``is_error`` result by the
    loop, with no fault marker -- which is what classifies it as a HARNESS
    fault. That is the correct bucket for an adapter that died, and it is why
    the validation path must claim its own class explicitly rather than relying
    on ``is_error``.
    """

    async def boom(batch: ActionBatch) -> ExecuteResult:
        raise RuntimeError("adapter died")

    token = PendingObservationToken(_observation())
    tool = build_action_tool(token, surface=SURFACE, execute=boom, render=_render)
    stream = _ScriptedStream(
        [
            [
                _call_delta(0, {"actions": [{"kind": "wait", "duration_ms": 5}]}, id="c1"),
                StreamEndEvent(stop_reason="toolUse"),
            ],
            [StreamTextDelta(delta="done"), StreamEndEvent(stop_reason="stop")],
        ]
    )

    events = await _drive(tool, token, stream)

    results = _tool_results(events)
    assert [result.is_error for result in results] == [True]
    assert FAULT_KEY not in (results[0].details or {})


@pytest.mark.asyncio
async def test_a_successful_batch_returns_the_rendered_observation_and_the_receipt() -> None:
    """Step 7 of the body: the model reads the new state, and the receipt rides along.

    The content is the injected renderer's, not a second rendering written here;
    the receipt in ``details`` is the adapter's own, which is what lets the
    driver bind the step evidence without re-deriving anything.
    """
    produced = _observation(1)
    offline = _Offline(_observation(), outputs=[produced])
    result = await offline.call({"actions": [{"kind": "wait", "duration_ms": 5}]})

    assert not result.is_error
    text = result.content[0].text  # type: ignore[union-attr]
    assert text == f"rendered:{produced.observation_id}"
    receipt = (result.details or {})["receipt"]
    assert receipt["input_observation_id"] == offline.execute.batches[0].observation_id
    assert receipt["output_observation_id"] == produced.observation_id
    assert receipt["sequence"] == 1


@pytest.mark.asyncio
async def test_a_call_with_no_pending_observation_is_refused(tmp_path: Any) -> None:
    """Before anything is pending there is no screen to bind to, so nothing runs."""
    adapter = FakeAdapter(tmp_path, EPISODE)
    session = _session(adapter)
    token = PendingObservationToken(adapter.current)
    assert token.consume_pending() is not None
    tool = build_action_tool(
        token, surface=SURFACE, execute=_adapter_execute(session), render=_render
    )

    refused = await tool.execute(
        "call-1", {"actions": [{"kind": "wait", "duration_ms": 5}]}, None, None, ToolContext()
    )

    assert refused.is_error
    assert (refused.details or {})[REJECTION_CLASS_KEY] == "second-batch"
    assert adapter.calls.count("execute") == 0
