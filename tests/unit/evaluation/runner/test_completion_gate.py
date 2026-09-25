"""The completion gate: one challenge to a ``done`` finish, then acceptance.

WHY THESE ASSERTIONS AND NOT OTHERS. Every test here drives the REAL
``EpisodeRunner``, the REAL ``VerifiedAdapterSession`` and a REAL
``EvidenceWriter`` on ``tmp_path`` -- only the subprocess boundary and the model
are faked -- because the properties the gate needs are about the RUNNER's
sequencing, not about the client's wording:

* that a declaration is refused-as-a-claim exactly once and the SECOND one is
  accepted, which is the bound that stops the gate from ever turning a finished
  episode into a model failure;
* that nothing about the turn moves while that happens -- the observation is the
  same, the turn does not close, and no second ``action_batch`` is sealed as the
  terminal one, because a bundle carrying two terminal batches is ambiguous
  about which batch the episode ended on;
* that the exchange is visible in the bundle as an ``error`` event with its own
  ``diagnostic_code``, and carries what the model was actually shown.

The falsification of the bound is exercised by
``test_the_budget_is_what_ends_the_exchange``: with the off-by-one that removes
it (``fired > budget``), that test's challenge count assertion fails.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
from typing import Any, AsyncIterator, Iterator, Sequence

import pytest

from local_operator.evaluation.adapters.api import observation_content_id
from local_operator.evaluation.evidence.models import ActionBatchPayload, ErrorPayload
from local_operator.evaluation.evidence.verify import verify_bundle
from local_operator.evaluation.protocol import (
    ActionBatch,
    ArtifactRef,
    FinishAction,
    FrameGeometry,
    FrameRef,
    FrameSize,
    Observation,
)
from local_operator.evaluation.runner.completion import finish_claim
from local_operator.evaluation.runner.episode import EpisodeConfig, EpisodeRunner
from local_operator.evaluation.runner.model import EpisodeTurn
from local_operator.evaluation.runner.provider_client import (
    ProviderModelClient,
    build_completion_challenge,
)
from local_operator.harness.types import ModelSpec, StreamEndEvent, StreamTextDelta
from tests.unit.evaluation.runner.conftest import (
    ROUTE,
    TASK_ID,
    FakeAdapter,
    ScriptedModel,
    build_config,
    build_spec,
    payloads,
    selector,
)

CHALLENGED = "completion-challenged"


class GatedModel(ScriptedModel):
    """``ScriptedModel`` plus the one method the gate needs in order to fire.

    Its ``decide`` is the conftest double's, untouched -- including the
    ``finish`` status it builds -- so a test that wants a non-``done``
    declaration overrides ``finish_status``. ``challenges`` records
    ``(batch, instruction, text)`` per challenge, so the assertion that the
    SECOND declaration is byte-identical to the first is a comparison of two
    things the model really produced.
    """

    def __init__(self, *args: Any, finish_status: str = "done", **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.finish_status = finish_status
        self.challenges: list[tuple[ActionBatch, str, str]] = []

    async def decide(
        self, observation: Observation, history: Sequence[EpisodeTurn], **kwargs: Any
    ) -> Any:
        decision = await super().decide(observation, history, **kwargs)
        if self.finish_status == "done":
            return decision
        batch = decision.action_batch
        restated = batch.actions[0].model_copy(update={"status": self.finish_status})
        return decision.model_copy(
            update={"action_batch": batch.model_copy(update={"actions": (restated,)})}
        )

    async def challenge_completion(
        self,
        observation: Observation,
        history: Sequence[EpisodeTurn],
        *,
        batch: ActionBatch,
        instruction: str,
    ) -> str:
        text = build_completion_challenge(
            claim=finish_claim(batch), instruction=instruction, observation=observation
        )
        self.challenges.append((batch, instruction, text))
        return text


async def _rescue_ok(descriptor: Any, **kwargs: Any) -> Any:
    del kwargs

    class _Aggregate:
        complete = True
        descriptor_id = descriptor.descriptor_id

    return _Aggregate()


def _run_episode(
    tmp_path: Path,
    episode_id: str,
    *,
    adapter: FakeAdapter,
    model: Any,
    config: Any = None,
) -> EpisodeRunner:
    return EpisodeRunner(
        build_spec(episode_id),
        config or build_config(tmp_path, max_steps=4),
        selector=selector(tmp_path),
        model=model,
        launch=lambda _: adapter,
        rescue=_rescue_ok,
    )


def _challenges(bundle: Path) -> list[ErrorPayload]:
    return [
        payload
        for payload in payloads(bundle, ErrorPayload)
        if payload.diagnostic_code == CHALLENGED
    ]


def _challenge_text(bundle: Path, payload: ErrorPayload) -> str:
    from local_operator.evaluation.adapters.supervisor import verify_artifact

    assert payload.detail_artifact is not None
    return verify_artifact(bundle / "artifacts", payload.detail_artifact).decode("utf-8")


# ---------------------------------------------------------------------------
# The gate, end to end through the runner
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_done_declaration_is_challenged_once_and_the_second_is_accepted(
    tmp_path: Path, episode_id: str
) -> None:
    adapter = FakeAdapter(tmp_path, episode_id)
    model = GatedModel(["finish"])
    runner = _run_episode(tmp_path, episode_id, adapter=adapter, model=model)

    outcome = await runner.run()

    assert outcome.bundle_root is not None
    assert outcome.status == "completed"
    report = verify_bundle(outcome.bundle_root)
    assert report.valid, [issue.code for issue in report.issues]
    # The gate spent exactly one extra decision and stopped: the second `done`
    # declaration ended the episode rather than being challenged again.
    assert model.calls == 2
    assert len(model.challenges) == 1
    terminal = [
        payload
        for payload in payloads(outcome.bundle_root, ActionBatchPayload)
        if payload.terminal == "finish"
    ]
    assert len(terminal) == 1
    # Nothing moved while the challenge happened. The re-decision was made on
    # the SAME observation and its turn is still UNBATCHED, which is what makes
    # the second `decide` a corrective turn rather than a new step.
    assert (
        model.histories[-1][-1].observation.observation_id
        == model.histories[0][-1].observation.observation_id
    )
    assert model.histories[-1][-1].batch is None
    # The declaration that was ACCEPTED is byte-identical to the one that was
    # challenged: the gate accepted the same finish, re-declared, which is the
    # escape hatch the challenge text promises.
    from local_operator.evaluation.adapters.supervisor import verify_artifact

    sealed = verify_artifact(outcome.bundle_root / "artifacts", terminal[0].action_artifact)
    assert sealed == model.challenges[0][0].to_canonical_json()
    # Exactly one challenge event, and it is the model-facing kind the design
    # required: no new EventKind, no new payload field.
    challenges = _challenges(outcome.bundle_root)
    assert len(challenges) == 1
    assert challenges[0].category == "model"
    assert challenges[0].retryable is True
    assert challenges[0].detail_artifact is not None


@pytest.mark.asyncio
async def test_the_challenge_quotes_the_claim_the_task_and_the_end_state_ids(
    tmp_path: Path, episode_id: str
) -> None:
    adapter = FakeAdapter(tmp_path, episode_id)
    model = GatedModel(["finish"])
    runner = _run_episode(tmp_path, episode_id, adapter=adapter, model=model)

    outcome = await runner.run()
    assert outcome.bundle_root is not None
    text = _challenge_text(outcome.bundle_root, _challenges(outcome.bundle_root)[0])

    # The claim, quoted as a claim, and the task as the reset observation stated
    # it -- the two operands the missing comparison needs.
    assert "task complete" in text
    assert "state-0" in text
    assert "CLAIM" in text
    assert "the only" in text and "evidence that counts" in text
    # The trailing facts a reply has to bind to, restated for the reason
    # ``_rejection_prompt`` restates them: getting them wrong costs a billed
    # rejection.
    assert "Observation ID:" in text
    # And the text the artifact carries IS the text the model was handed.
    assert text == model.challenges[0][2]


@pytest.mark.asyncio
async def test_the_budget_is_what_ends_the_exchange(tmp_path: Path, episode_id: str) -> None:
    """The bound is the budget, so a larger one is honoured and then stops.

    This is also the falsification target: with ``fired > budget`` in
    ``CompletionGate.should_challenge`` the count below is one too HIGH (the
    gate challenges one time more than it was funded for), which the assertion
    on ``len(model.challenges)`` catches while the episode still ends.
    """

    adapter = FakeAdapter(tmp_path, episode_id)
    model = GatedModel(["finish"])
    config = build_config(tmp_path, max_steps=4, completion_challenges=3)
    runner = _run_episode(tmp_path, episode_id, adapter=adapter, model=model, config=config)

    outcome = await runner.run()

    assert outcome.status == "completed"
    assert outcome.bundle_root is not None
    assert len(model.challenges) == 3
    # One declaration, three challenges, and the fourth declaration accepted.
    assert model.calls == 4
    assert len(_challenges(outcome.bundle_root)) == 3


@pytest.mark.asyncio
async def test_the_gate_off_ends_on_the_first_declaration(tmp_path: Path, episode_id: str) -> None:
    adapter = FakeAdapter(tmp_path, episode_id)
    model = GatedModel(["finish"])
    config = build_config(tmp_path, max_steps=4, completion_gate=False)
    runner = _run_episode(tmp_path, episode_id, adapter=adapter, model=model, config=config)

    outcome = await runner.run()

    assert outcome.bundle_root is not None
    assert outcome.status == "completed"
    # The control arm is byte-for-byte the old behaviour: one decision, no
    # challenge, and the episode sealed on it.
    assert model.calls == 1
    assert model.challenges == []
    assert _challenges(outcome.bundle_root) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["failed", "infeasible"])
async def test_a_non_done_finish_is_never_challenged(
    tmp_path: Path, episode_id: str, status: str
) -> None:
    """A model reporting that it could NOT finish has no completion to confirm."""

    adapter = FakeAdapter(tmp_path, episode_id)
    model = GatedModel(["finish"], finish_status=status)
    runner = _run_episode(tmp_path, episode_id, adapter=adapter, model=model)

    outcome = await runner.run()

    assert outcome.bundle_root is not None
    assert model.calls == 1
    assert model.challenges == []
    assert _challenges(outcome.bundle_root) == []


@pytest.mark.asyncio
async def test_the_gate_cannot_fire_for_a_client_that_cannot_re_present_the_end_state(
    tmp_path: Path, episode_id: str
) -> None:
    """Silence here is the driver's problem, not the runner's.

    ``ScriptedModel`` has no ``challenge_completion``, so there is nothing for
    the gate to ask and it stays inert. That is exactly the degrade
    ``scripts/run_episode.py`` refuses to seal: its preflight fails such a run
    unless ``--no-completion-gate`` says the arm is deliberate, because a bundle
    whose manifest claims the gate ran while nothing was challenged cannot be
    told apart from a real gate-on run.
    """

    adapter = FakeAdapter(tmp_path, episode_id)
    model = ScriptedModel(["finish"])
    runner = _run_episode(tmp_path, episode_id, adapter=adapter, model=model)

    outcome = await runner.run()

    assert outcome.status == "completed"
    assert model.calls == 1
    assert outcome.bundle_root is not None
    assert _challenges(outcome.bundle_root) == []


@pytest.mark.parametrize("budget", [-1, 1.5, True])
def test_episode_config_rejects_an_invalid_challenge_budget(tmp_path: Path, budget: Any) -> None:
    roots = [tmp_path / name for name in ("evidence", "artifacts", "rescue")]
    for root in roots:
        root.mkdir()
    with pytest.raises(ValueError, match="non-negative integer"):
        EpisodeConfig(
            evidence_root=roots[0],
            artifact_root=roots[1],
            rescue_root=roots[2],
            max_steps=4,
            completion_challenges=budget,
        )


# ---------------------------------------------------------------------------
# The client side: an append, with the end state attached
# ---------------------------------------------------------------------------


class _ReplyStream:
    """The ``SessionStreamFn`` seam with fixed replies, recording each request."""

    def __init__(self, replies: Sequence[str]) -> None:
        self._replies = list(replies)
        self.requests: list[Any] = []

    def __call__(self, request: Any, signal: Any) -> AsyncIterator[Any]:
        del signal
        self.requests.append(request)
        reply = self._replies.pop(0) if len(self._replies) > 1 else self._replies[0]
        return self._events(reply)

    async def _events(self, reply: str) -> AsyncIterator[Any]:
        for start in range(0, len(reply), 7):
            yield StreamTextDelta(delta=reply[start : start + 7])
        yield StreamEndEvent(stop_reason="stop", usage=None, provider_payload=None, error=None)


def _framed_observation(root: Path, sequence: int, *, text: str) -> Observation:
    from local_operator.compaction.png import encode_grayscale_png

    data = encode_grayscale_png(1, 1, bytes([sequence + 1]))
    digest = hashlib.sha256(data).hexdigest()
    root.mkdir(parents=True, exist_ok=True)
    (root / digest).write_bytes(data)
    provisional = Observation(
        task_id="task-1",
        episode_id="episode-1",
        sequence=sequence,
        observation_id="provisional",
        text=text,
        frames=(
            FrameRef(
                frame_id=f"frame-{sequence}",
                artifact=ArtifactRef(sha256=digest, media_type="image/png", byte_count=len(data)),
                geometry=FrameGeometry(
                    native=FrameSize(width=1, height=1),
                    model_visible=FrameSize(width=1, height=1),
                ),
            ),
        ),
    )
    return provisional.model_copy(update={"observation_id": observation_content_id(provisional)})


def _finish_payload(current: Observation) -> str:
    return json.dumps(
        {
            "protocol_version": "1.0",
            "task_id": current.task_id,
            "episode_id": current.episode_id,
            "observation_id": current.observation_id,
            "actions": [
                {
                    "kind": "finish",
                    "observation_id": current.observation_id,
                    "status": "done",
                    "reason": "I filled every field the task asked for",
                }
            ],
        }
    )


@pytest.mark.asyncio
async def test_the_challenge_is_an_append_and_the_end_state_rides_with_it(
    tmp_path: Path,
) -> None:
    """The turned-in-end-state message is NEW, and nothing sent is rewritten.

    Re-attaching the observation by editing the message that already carried it
    would cost the whole prompt-cache entry (the client's own measurement: any
    rewrite of a sent message misses), so the challenge is proven to be an
    APPEND by comparing the second request's prefix against the first request's
    messages, byte for byte, and by finding the frame's pixels in the new
    trailing user message only.
    """

    root = tmp_path / "artifacts"
    instruction = _framed_observation(root, 0, text="Fill in the booking form")
    current = _framed_observation(root, 1, text="the confirmation screen")
    turns = [EpisodeTurn(observation=instruction), EpisodeTurn(observation=current)]
    stream = _ReplyStream([_finish_payload(current)])
    client = ProviderModelClient(
        stream,
        route=ROUTE,
        model_spec=ModelSpec(provider="provider", model_id="model"),
        artifact_root=root,
    )

    decision = await client.decide(current, turns)
    first_request = list(stream.requests[0].messages)
    batch = decision.action_batch

    challenge = await client.challenge_completion(
        current, turns, batch=batch, instruction=instruction.text or ""
    )
    await client.decide(current, turns)

    messages = list(stream.requests[-1].messages)
    # Appended, never substituted: every message the first request carried is
    # still there, in the same bytes, and the pair is at the tail.
    assert messages[: len(first_request)] == first_request
    assert len(messages) == len(first_request) + 2
    claim_message, challenge_message = messages[-2], messages[-1]
    assert claim_message.role == "assistant"
    assert challenge_message.role == "user"
    # The assistant turn is the model's OWN declaration, verbatim -- the message
    # it would have been shown had the episode continued.
    assert claim_message.content[0].text == batch.to_canonical_json().decode("utf-8")
    # The challenge text is first in its own message, and the end state's frame
    # is attached to that SAME message: one reader, and the pixels are the
    # adapter's, untouched.
    assert challenge_message.content[0].text == challenge
    image = challenge_message.content[1]
    assert image.mime_type == "image/png"
    from local_operator.evaluation.adapters.supervisor import verify_artifact

    assert base64.b64decode(image.data) == verify_artifact(root, current.frames[0].artifact)


def test_the_challenge_says_what_the_reason_is_and_names_both_replies() -> None:
    """The wording's contract, asserted directly so a reword cannot lose it."""

    observation = Observation(
        task_id="task-1",
        episode_id="episode-1",
        sequence=0,
        observation_id="obs-1",
        text="Create a calendar event titled Standup",
    )
    claim = FinishAction(
        observation_id="obs-1", status="done", reason="The event is on the calendar"
    )

    text = build_completion_challenge(
        claim=claim, instruction=observation.text or "", observation=observation
    )

    assert "The event is on the calendar" in text
    assert "Create a calendar event titled Standup" in text
    assert "not accepted yet" in text
    assert "the same finish action, unchanged" in text.lower()
    assert "do no optional extra work" in text.lower()
    assert "obs-1" in text


# ---------------------------------------------------------------------------
# The driver: the flag, and the refusal to seal a gate that cannot run
# ---------------------------------------------------------------------------


def _driver_args(selector_dir: Path, run_root: Path, *extra: str) -> list[str]:
    """Real driver arguments, with a selector file and the task it pins."""

    pinned = selector(selector_dir)
    workspace = Path(pinned.workspace)
    (workspace / "tasks").mkdir(parents=True, exist_ok=True)
    (workspace / "tasks" / f"{TASK_ID}.py").write_text("task = 'plain'\n", encoding="utf-8")
    selector_file = selector_dir / "selector.json"
    selector_file.write_text(pinned.model_dump_json(), encoding="utf-8")
    return [
        "--selector",
        str(selector_file),
        "--task-id",
        TASK_ID,
        "--route",
        "test/fake/model:free",
        "--run-root",
        str(run_root),
        "--no-store",
        *extra,
    ]


@pytest.fixture
def durable_run_root() -> Iterator[Path]:
    """A run root the driver's volatile-root refusal accepts.

    ``tmp_path`` is ``$TMPDIR``-derived, and the driver REFUSES a run root under
    a purgable location (a purge destroyed a paid pilot's inputs). The refusal
    is deliberately not patched, so the root has to be genuinely durable: this
    is the same account's real home that ``test_build_and_scripts.durable_path``
    uses, reached through ``pwd`` because the suite re-points ``HOME`` at a
    scratch directory. Created per test and removed afterwards.
    """

    import pwd
    import shutil
    import uuid

    root = (
        Path(pwd.getpwuid(os.getuid()).pw_dir)
        / ".cache"
        / "lop-completion-gate-tests"
        / uuid.uuid4().hex[:12]
    )
    root.mkdir(parents=True)
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_the_flag_reaches_the_config_and_the_scripted_client_can_challenge(
    tmp_path: Path, durable_run_root: Path
) -> None:
    """The arm is settable from the command line, and the scripted client can run it.

    Both halves matter: a flag nothing reads is not an arm, and a scripted client
    with no ``challenge_completion`` would make the scripted end-to-end run the
    one shape that cannot exercise the gate at all -- which the driver now
    REFUSES rather than sealing.
    """

    from scripts.run_episode import (
        _route_identity,
        _ScriptedFinish,
        build_config,
        build_parser,
    )

    off = build_parser().parse_args(
        _driver_args(tmp_path, durable_run_root, "--no-completion-gate")
    )
    on = build_parser().parse_args(_driver_args(tmp_path, durable_run_root))

    assert off.no_completion_gate is True
    assert on.no_completion_gate is False
    for args, expected in ((off, False), (on, True)):
        config = build_config(
            durable_run_root,
            episode_id="ep-flag",
            max_steps=2,
            max_cycle_usd_micros=None,
            completion_gate=not args.no_completion_gate,
        )
        assert config.completion_gate is expected
    # The default is the gate ON, so an unset flag runs the shipped arm.
    assert build_config(
        durable_run_root, episode_id="ep-default", max_steps=2, max_cycle_usd_micros=None
    ).completion_gate
    assert callable(
        getattr(_ScriptedFinish(_route_identity("test", "fake/model:free")), "challenge_completion")
    )


@pytest.mark.asyncio
async def test_the_driver_refuses_a_gate_run_its_model_client_cannot_challenge(
    tmp_path: Path,
    durable_run_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A DEAD INSTRUMENT MUST NOT RETURN A READING.

    A client that cannot re-present the end state leaves the gate inert, and a
    bundle sealed from that run would claim the gate was on while nothing was
    challenged -- indistinguishable, to a later reader, from a real gate-on run.
    The driver fails with the remedy named instead.
    """

    from scripts import run_episode as script

    class _Blind:
        """A model client with decisions but no way to re-present the end state."""

        def __init__(self, route: Any) -> None:
            self._route = route

    monkeypatch.setattr(script, "_ScriptedFinish", _Blind)

    code = await script.run(
        script.build_parser().parse_args(_driver_args(tmp_path, durable_run_root))
    )

    assert code == script.EXIT_PREFLIGHT
    # Nothing was allocated and nothing was scored: the refusal is a preflight,
    # which is what "before anything is allocated" means for the effort-level
    # refusal in the same spot.
    evidence = durable_run_root / "evidence"
    assert not evidence.exists() or not list(evidence.iterdir())
