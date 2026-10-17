"""Event mapping and terminal coherence for one episode.

Every test here drives the REAL ``VerifiedAdapterSession``, the REAL lifecycle
authorities and a REAL ``EvidenceWriter``; only the subprocess boundary is
faked. The standing assertion is that each terminal path leaves a bundle the
independent verifier accepts, because a runner that writes evidence no verifier
will take is indistinguishable from one that writes none.
"""

from __future__ import annotations

import asyncio
import errno
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from local_operator.evaluation.adapters.api import (
    Requirement,
    ScopedInfraValue,
    SecretRef,
)
from local_operator.evaluation.adapters.rpc import WITHHELD
from local_operator.evaluation.adapters.supervisor import SupervisionError
from local_operator.evaluation.evidence.models import (
    ActionBatchPayload,
    AgentStopPayload,
    CleanupPayload,
    EnvironmentStepPayload,
    ErrorPayload,
    EvidenceArtifactRef,
    ModelResponsePayload,
    ObservationPayload,
    ReconciliationPayload,
    ScoreArtifact,
    UsageCostPayload,
)
from local_operator.evaluation.evidence.store import EvidenceWriter
from local_operator.evaluation.evidence.verify import verify_bundle
from local_operator.evaluation.receipts import RedactionSet
from local_operator.evaluation.runner.episode import (
    DISCLOSED_INFRA_METADATA_KEYS,
    MAX_STDERR_TAIL_CHARS,
    EpisodeConfig,
    EpisodeRunner,
    _canary_values_outside_own_roots,
    _failure_layer,
)
from local_operator.evaluation.runner.model import EpisodeModelClient
from local_operator.evaluation.runner.provider_client import ProviderModelClient
from local_operator.evaluation.runner.secrets import StaticSecretResolver
from local_operator.harness.types import (
    ModelSpec,
    StreamEndEvent,
    StreamTextDelta,
    StreamUsageEvent,
    Usage,
)
from scripts.run_episode import _outcome_json
from tests.unit.evaluation.runner.conftest import (
    ROUTE,
    FakeAdapter,
    ModelUsage,
    RecordingResponder,
    ScriptedModel,
    build_config,
    build_spec,
    observation,
    payloads,
    selector,
)


async def _rescue_ok(descriptor: Any, **kwargs: Any) -> Any:
    del kwargs

    class _Aggregate:
        complete = True
        descriptor_id = descriptor.descriptor_id

    return _Aggregate()


def _runner(
    tmp_path: Path,
    episode_id: str,
    *,
    adapter: FakeAdapter,
    model: EpisodeModelClient,
    responder: Any = None,
    max_steps: int = 4,
    redactions: RedactionSet | None = None,
    spec: Any = None,
    config: Any = None,
    secrets: Any = None,
) -> EpisodeRunner:
    return EpisodeRunner(
        spec or build_spec(episode_id),
        config or build_config(tmp_path, max_steps=max_steps),
        selector=selector(tmp_path),
        model=model,
        responder=responder,
        launch=lambda _: adapter,
        rescue=_rescue_ok,
        redactions=redactions,
        secrets=secrets,
    )


def _kinds(root: Path) -> list[str]:
    return [event.kind for event in verify_bundle(root).events]


@pytest.mark.parametrize("rate", [float("nan"), float("inf"), -0.01, True])
def test_episode_config_rejects_invalid_action_overhead(tmp_path: Path, rate: float) -> None:
    roots = [tmp_path / name for name in ("evidence", "artifacts", "rescue")]
    for root in roots:
        root.mkdir()
    with pytest.raises(ValueError, match="finite and nonnegative"):
        EpisodeConfig(
            evidence_root=roots[0],
            artifact_root=roots[1],
            rescue_root=roots[2],
            max_steps=4,
            execution_overhead_seconds_per_action=rate,
        )


@pytest.mark.parametrize("value", [-1, True, 2.5])
def test_episode_config_rejects_invalid_rejection_bounds(tmp_path: Path, value: Any) -> None:
    """Both rejection knobs refuse the values that would silently mean another.

    A ``bool`` is an ``int`` to ``isinstance`` and would read as a bound of
    0/1; a negative value would read as the one-strike sentinel; a float is
    not a count of refusals at all. Refused at construction, on BOTH knobs,
    so no configuration can reach the ceiling arithmetic as nonsense -- the
    asymmetry review round 1 flagged (``max_rejection_streak`` validated,
    ``max_decision_retries`` not) is closed in the direction that refuses.
    """

    roots = [tmp_path / name for name in ("evidence", "artifacts", "rescue")]
    for root in roots:
        root.mkdir()
    with pytest.raises(ValueError, match="max_decision_retries must be a non-negative integer"):
        EpisodeConfig(
            evidence_root=roots[0],
            artifact_root=roots[1],
            rescue_root=roots[2],
            max_steps=4,
            max_decision_retries=value,
        )
    with pytest.raises(ValueError, match="max_rejection_streak must be a non-negative integer"):
        EpisodeConfig(
            evidence_root=roots[0],
            artifact_root=roots[1],
            rescue_root=roots[2],
            max_steps=4,
            max_rejection_streak=value,
        )


@pytest.mark.asyncio
async def test_default_settle_policy_is_sealed_in_verified_episode_bundle(
    tmp_path: Path, episode_id: str
) -> None:
    from scripts.run_episode import (
        _ensure_action_settle_policy,
        _infra_disclosure_metadata,
        _parse_infra,
    )

    infra = _ensure_action_settle_policy(_parse_infra([], "benchmark_compute"), "benchmark_compute")
    spec = replace(
        build_spec(episode_id),
        infra_values=infra,
        metadata=_infra_disclosure_metadata(infra),
    )
    adapter = FakeAdapter(
        tmp_path,
        episode_id,
        declared_requirements=(
            Requirement(
                requirement_id="OSWORLD_ACTION_SETTLE_POLICY",
                kind="infra",
                name="OSWORLD_ACTION_SETTLE_POLICY",
                required=False,
            ),
        ),
    )
    runner = _runner(
        tmp_path, episode_id, adapter=adapter, model=ScriptedModel(["finish"]), spec=spec
    )

    outcome = await runner.run()
    assert outcome.bundle_root is not None
    report = verify_bundle(outcome.bundle_root)
    assert report.valid, [issue.code for issue in report.issues]
    assert report.manifest is not None
    assert report.manifest.metadata["osworld_action_settle_policy"] == "throughput"
    assert report.manifest.metadata["osworld_action_settle_seconds"] == 3


@pytest.mark.asyncio
async def test_paper_action_overhead_funds_actual_execute_timeout(
    tmp_path: Path, episode_id: str
) -> None:
    adapter = FakeAdapter(tmp_path, episode_id)
    runner = _runner(
        tmp_path,
        episode_id,
        adapter=adapter,
        model=ScriptedModel(["wait:2000", "finish"]),
        config=build_config(
            tmp_path,
            execution_overhead_seconds_per_action=3.0,
        ),
    )

    outcome = await runner.run()

    assert outcome.status == "completed", outcome.diagnostic
    assert adapter.timeouts["execute"] == [35.0]


@pytest.mark.asyncio
async def test_happy_episode_writes_every_event_in_protocol_order(
    tmp_path: Path, episode_id: str
) -> None:
    adapter = FakeAdapter(tmp_path, episode_id)
    runner = _runner(tmp_path, episode_id, adapter=adapter, model=ScriptedModel(["step", "finish"]))

    outcome = await runner.run()

    assert outcome.status == "completed"
    assert outcome.reportability_label == "reportable"
    assert outcome.score is not None and outcome.score.status == "scored"
    root = outcome.bundle_root
    assert root is not None
    report = verify_bundle(root)
    assert report.valid, [issue.code for issue in report.issues]
    assert _kinds(root) == [
        # Preflight is event #0 and the commitment precedes all execution.
        "preflight",
        "budget_commitment",
        "lifecycle_transition",
        "observation",
        "model_request",
        "model_response",
        "usage_cost",
        "action_batch",
        "environment_step",
        "observation",
        "model_request",
        "model_response",
        "usage_cost",
        "action_batch",
        "finalization_start",
        "scoring_start",
        "scoring_result",
        "reconciliation",
        "cleanup",
        "lifecycle_transition",
    ]
    assert adapter.calls[:5] == [
        "handshake",
        "inspect_requirements",
        "prepare",
        "reset_start",
        "observe",
    ]


@pytest.mark.asyncio
async def test_step_event_precedes_its_output_observation(tmp_path: Path, episode_id: str) -> None:
    adapter = FakeAdapter(tmp_path, episode_id)
    runner = _runner(tmp_path, episode_id, adapter=adapter, model=ScriptedModel(["step", "finish"]))

    outcome = await runner.run()

    root = outcome.bundle_root
    assert root is not None
    kinds = _kinds(root)
    step = kinds.index("environment_step")
    # The observation an execution produced must follow the step that declared
    # it; reversing them leaves the observation unbound to any receipt.
    assert kinds[step + 1] == "observation"


@pytest.mark.asyncio
async def test_mid_episode_observe_writes_no_duplicate_observation(
    tmp_path: Path, episode_id: str
) -> None:
    """``observe`` is a snapshot check, and a snapshot is not new evidence."""

    adapter = FakeAdapter(tmp_path, episode_id)
    runner = _runner(tmp_path, episode_id, adapter=adapter, model=ScriptedModel(["step", "finish"]))

    outcome = await runner.run()

    root = outcome.bundle_root
    assert root is not None
    observations = [payload.sequence for payload in payloads(root, ObservationPayload)]
    # One initial plus exactly one per executed step, with no repeats.
    assert observations == [0, 1]


@pytest.mark.asyncio
async def test_truncation_scores_normally_and_marks_the_last_step(
    tmp_path: Path, episode_id: str
) -> None:
    adapter = FakeAdapter(tmp_path, episode_id)
    runner = _runner(
        tmp_path,
        episode_id,
        adapter=adapter,
        model=ScriptedModel(["step"] * 6),
        max_steps=2,
    )

    outcome = await runner.run()

    assert outcome.status == "completed"
    assert outcome.score is not None and outcome.score.status == "scored"
    root = outcome.bundle_root
    assert root is not None
    assert verify_bundle(root).valid
    steps = payloads(root, EnvironmentStepPayload)
    assert len(steps) == 2
    assert [step.truncated for step in steps] == [False, True]


@pytest.mark.asyncio
async def test_ask_user_exchange_precedes_the_executed_ask_batch(
    tmp_path: Path, episode_id: str
) -> None:
    """Verify's one-batch-per-observation rule forces this choreography."""

    adapter = FakeAdapter(tmp_path, episode_id)
    responder = RecordingResponder("do the thing")
    runner = _runner(
        tmp_path,
        episode_id,
        adapter=adapter,
        model=ScriptedModel(["ask", "finish"]),
        responder=responder,
    )

    outcome = await runner.run()

    assert outcome.status == "completed"
    assert responder.prompts == ["What next?"]
    root = outcome.bundle_root
    assert root is not None
    assert verify_bundle(root).valid
    kinds = _kinds(root)
    exchange = kinds.index("user_simulator_exchange")
    assert kinds[exchange + 1] == "action_batch"
    assert kinds[exchange + 2] == "environment_step"
    assert "ask_user_exchange" in adapter.calls


@pytest.mark.asyncio
async def test_unanswered_ask_cancels_rather_than_leaving_it_open(
    tmp_path: Path, episode_id: str
) -> None:
    adapter = FakeAdapter(tmp_path, episode_id)
    runner = _runner(
        tmp_path,
        episode_id,
        adapter=adapter,
        model=ScriptedModel(["ask", "finish"]),
        responder=RecordingResponder(None),
    )

    outcome = await runner.run()

    assert outcome.status == "cancelled"
    assert outcome.reportability_label == "cancelled"
    root = outcome.bundle_root
    assert root is not None
    assert verify_bundle(root).valid
    assert "cancel" in _kinds(root)
    assert outcome.score is not None and outcome.score.reason == "cancelled"


@pytest.mark.asyncio
async def test_unanswered_ask_after_a_step_scores_the_reached_state(
    tmp_path: Path, episode_id: str
) -> None:
    """A human who does not exist must not void paid environment work.

    The step-0 ask above stays a cancellation (nothing to grade); once a
    step has landed, the same unanswered ask becomes a scored agent stop on
    the state reached -- the ask is never written as a batch, so the stopped
    observation stays batchless and the verifier accepts the stop binding.
    """

    adapter = FakeAdapter(tmp_path, episode_id)
    runner = _runner(
        tmp_path,
        episode_id,
        adapter=adapter,
        model=ScriptedModel(["type", "ask", "finish"]),
        responder=RecordingResponder(None),
    )

    outcome = await runner.run()

    assert outcome.status == "completed"
    assert outcome.reportability_label == "reportable"
    assert outcome.score is not None and outcome.score.status == "scored"
    root = outcome.bundle_root
    assert root is not None
    report = verify_bundle(root)
    assert report.valid, [issue.code for issue in report.issues]
    kinds = _kinds(root)
    assert "cancel" not in kinds
    assert kinds.index("agent_stop") < kinds.index("finalization_start")
    stops = payloads(root, AgentStopPayload)
    assert len(stops) == 1
    assert stops[0].reason == "ask_unanswered"
    assert stops[0].attempts == 1
    assert stops[0].observation_id == payloads(root, ObservationPayload)[-1].observation_id
    assert stops[0].detail_artifact is not None
    # The environment graded the state the one step reached.
    assert "score" in adapter.calls


@pytest.mark.asyncio
async def test_provider_failure_finalizes_unscored_on_a_live_session(
    tmp_path: Path, episode_id: str
) -> None:
    """A provider error is not the environment's fault: no rescue, normal cleanup."""

    adapter = FakeAdapter(tmp_path, episode_id)
    runner = _runner(
        tmp_path,
        episode_id,
        adapter=adapter,
        model=ScriptedModel(error=RuntimeError("provider exhausted retries")),
    )

    outcome = await runner.run()

    assert outcome.status == "failed"
    assert outcome.rescue_required is False
    assert outcome.score is not None
    assert outcome.score.reason == "infrastructure_failure"
    root = outcome.bundle_root
    assert root is not None
    assert verify_bundle(root).valid
    errors = payloads(root, ErrorPayload)
    assert [error.category for error in errors] == ["provider"]
    # Cleanup ran on the live session rather than being minted as incomplete.
    assert "cleanup" in adapter.calls


@pytest.mark.asyncio
async def test_adapter_crash_poisons_runs_rescue_and_still_seals(
    tmp_path: Path, episode_id: str
) -> None:
    adapter = FakeAdapter(
        tmp_path, episode_id, failures={"execute": SupervisionError("worker died")}
    )
    runner = _runner(tmp_path, episode_id, adapter=adapter, model=ScriptedModel(["step", "finish"]))

    outcome = await runner.run()

    assert outcome.status == "failed"
    assert outcome.rescue_required is True
    assert outcome.rescue_complete is True
    assert outcome.reportability_label == "cleanup_incomplete"
    root = outcome.bundle_root
    assert root is not None
    report = verify_bundle(root)
    assert report.valid, [issue.code for issue in report.issues]
    errors = payloads(root, ErrorPayload)
    assert [error.category for error in errors] == ["adapter"]
    assert adapter.terminated


@pytest.mark.asyncio
async def test_crash_before_any_step_still_reaches_a_terminal(
    tmp_path: Path, episode_id: str
) -> None:
    adapter = FakeAdapter(
        tmp_path, episode_id, failures={"reset_start": SupervisionError("reset died")}
    )
    runner = _runner(tmp_path, episode_id, adapter=adapter, model=ScriptedModel())

    outcome = await runner.run()

    assert outcome.status == "failed"
    root = outcome.bundle_root
    assert root is not None
    assert verify_bundle(root).valid
    assert outcome.score is not None and outcome.score.reason == "crash"


@pytest.mark.asyncio
async def test_scorer_failure_can_only_abandon_as_ambiguous(
    tmp_path: Path, episode_id: str
) -> None:
    """scoring_start is durable, so a scored intent cannot seal unscored."""

    adapter = FakeAdapter(tmp_path, episode_id, failures={"score": SupervisionError("scorer died")})
    runner = _runner(tmp_path, episode_id, adapter=adapter, model=ScriptedModel())

    outcome = await runner.run()

    assert outcome.status == "abandoned"
    root = outcome.bundle_root
    assert root is not None
    report = verify_bundle(root)
    assert report.abandonment is not None
    assert report.abandonment.reason == "ambiguous_finalization"
    # Rescue runs before abandonment: cloud safety must not wait on a writer.
    assert outcome.rescue_complete is True


def _failure_diagnostics(root: Path) -> list[tuple[ErrorPayload, str | None]]:
    """Every failure-diagnostic event in a bundle, with the text it points at."""

    report = verify_bundle(root)
    assert report.valid, [issue.code for issue in report.issues]
    found: list[tuple[ErrorPayload, str | None]] = []
    for event in report.events:
        if event.kind != "error":
            continue
        payload = event.payload
        assert isinstance(payload, ErrorPayload)
        text = None
        if payload.detail_artifact is not None:
            text = (root / "artifacts" / payload.detail_artifact.sha256).read_text()
        found.append((payload, text))
    return found


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure", "layer", "code_prefix", "needle"),
    [
        # ADAPTER / judge: the scorer answered with a failure, the harness is fine.
        (
            SupervisionError("judge backend returned 502"),
            "adapter",
            "adapter-",
            "judge backend returned 502",
        ),
        # UNEXPECTED: a bug in the harness, not the scorer. Filed separately so
        # it never reads as an adapter outage.
        (RuntimeError("unexpected boom in score path"), "internal", "internal-", "unexpected boom"),
    ],
)
async def test_a_scoring_failure_records_its_type_and_message(
    tmp_path: Path,
    episode_id: str,
    failure: BaseException,
    layer: str,
    code_prefix: str,
    needle: str,
) -> None:
    """The measured gap: two OSWorld runs abandoned with only ``evidencefailure``.

    A scoring failure after ``scoring_start`` can only abandon, and the
    abandonment field is a ``StrictIdentifier`` -- so the exception TYPE and
    TEXT were discarded and a canary refusal, a judge outage and a harness bug
    were indistinguishable after the fact. The diagnostic event and its
    bounded artifact are what the abandonment verification now sees, while the
    abandonment keeps its existing shape and code.
    """

    adapter = FakeAdapter(tmp_path, episode_id, failures={"score": failure})
    outcome = await _runner(tmp_path, episode_id, adapter=adapter, model=ScriptedModel()).run()

    assert outcome.status == "abandoned", outcome.diagnostic
    root = outcome.bundle_root
    assert root is not None
    diagnostics = _failure_diagnostics(root)
    assert len(diagnostics) == 1
    payload, text = diagnostics[0]
    assert text is not None
    assert payload.diagnostic_code.startswith(code_prefix)
    assert f"layer: {layer}" in text
    assert "phase: scoring" in text
    assert f"operation: score-{episode_id}" in text
    assert f"exception: {type(failure).__name__}" in text
    assert needle in text
    # The terminal itself is unchanged: only the diagnosis is new.
    abandonment = verify_bundle(root).abandonment
    assert abandonment is not None
    assert abandonment.reason == "ambiguous_finalization"
    if layer == "adapter":
        assert abandonment.diagnostic_code == "supervisionerror"


@pytest.mark.asyncio
async def test_an_evidence_refusal_at_scoring_is_labelled_evidence(
    tmp_path: Path, episode_id: str
) -> None:
    """The canary-refusal shape: the score exists but the bundle refuses its bytes.

    This is the case that produced the measured ``evidencefailure`` string. The
    layer has to read ``evidence`` -- not ``adapter`` -- because the judge
    answered and the fix is in the capture, not the scorer.
    """

    secret = "[redacted]"
    base = tmp_path / "evidence-case"
    base.mkdir()
    payload = json.dumps(
        {"evaluator_diagnostics": {"stderr": {"text": f"key {secret}"}}},
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    sha = hashlib.sha256(payload).hexdigest()
    artifacts = base / "artifacts"
    artifacts.mkdir(exist_ok=True)
    (artifacts / sha).write_bytes(payload)
    adapter = FakeAdapter(
        base,
        episode_id,
        score=ScoreArtifact(
            status="scored",
            binary=0,
            details=EvidenceArtifactRef(
                sha256=sha, media_type="application/json", byte_count=len(payload)
            ),
        ),
    )
    outcome = await _runner(
        base,
        episode_id,
        adapter=adapter,
        model=ScriptedModel(["step", "finish"]),
        spec=replace(build_spec(episode_id), secret_refs=(SecretRef(name="IDENT_ROW"),)),
        secrets=StaticSecretResolver({"IDENT_ROW": secret}),
    ).run()

    assert outcome.status == "abandoned", outcome.diagnostic
    root = outcome.bundle_root
    assert root is not None
    diagnostics = _failure_diagnostics(root)
    assert len(diagnostics) == 1
    reported, text = diagnostics[0]
    assert text is not None
    assert reported.diagnostic_code == "evidence-evidencefailure"
    assert "layer: evidence" in text
    # R8: the receipt is what refused the bytes, so the operation names the
    # receipt -- not the scoring operation id, which points at the wrong write.
    assert "operation: receipt:scoring_result" in text
    assert "exception: _EvidenceFailure" in text
    # The captured score's secret is never republished by the diagnosis.
    assert secret not in text
    assert secret not in b"".join(
        path.read_bytes() for path in (root / "artifacts").iterdir() if path.is_file()
    ).decode(errors="ignore")
    abandonment = verify_bundle(root).abandonment
    assert abandonment is not None
    assert abandonment.diagnostic_code == "evidencefailure"


@pytest.mark.asyncio
async def test_a_diagnostic_that_cannot_be_cleared_is_dropped(
    tmp_path: Path, episode_id: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """FAIL-OPEN: an uncleanable diagnostic must cost the diagnosis, not the terminal.

    The rendering is scanned first, but the store scans again and
    this test skips the first line of defence on purpose: it makes the produced
    TEXT carry a canary so ``publish_failure_diagnostic`` refuses it. The
    artifact is dropped, the event still names the class and layer, and the
    abandonment proceeds with exactly today's status and code.
    """

    import local_operator.evaluation.runner.episode as episode_module

    secret = "[redacted]"
    monkeypatch.setattr(
        episode_module,
        "_failure_diagnostic_text",
        lambda *args, **kwargs: f"message: carrying {secret}\n",
    )
    adapter = FakeAdapter(tmp_path, episode_id, failures={"score": SupervisionError("scorer died")})
    outcome = await _runner(
        tmp_path,
        episode_id,
        adapter=adapter,
        model=ScriptedModel(),
        redactions=RedactionSet.from_resolved_values((secret,)),
    ).run()

    assert outcome.status == "abandoned", outcome.diagnostic
    root = outcome.bundle_root
    assert root is not None
    report = verify_bundle(root)
    assert report.valid, [issue.code for issue in report.issues]
    assert report.abandonment is not None
    assert report.abandonment.reason == "ambiguous_finalization"
    assert report.abandonment.diagnostic_code == "supervisionerror"
    diagnostics = _failure_diagnostics(root)
    assert len(diagnostics) == 1
    payload, text = diagnostics[0]
    assert payload.detail_artifact is None and text is None
    assert secret not in b"".join(
        path.read_bytes() for path in (root / "artifacts").iterdir() if path.is_file()
    ).decode(errors="ignore")


@pytest.mark.asyncio
async def test_a_redactable_diagnostic_withholds_only_the_message(
    tmp_path: Path, episode_id: str
) -> None:
    """A secret-shaped message is WITHHELD whole; the class and phase survive."""

    secret = "[redacted]"
    adapter = FakeAdapter(
        tmp_path, episode_id, failures={"score": SupervisionError(f"judge echoed {secret}")}
    )
    outcome = await _runner(
        tmp_path,
        episode_id,
        adapter=adapter,
        model=ScriptedModel(),
        redactions=RedactionSet.from_resolved_values((secret,)),
    ).run()

    assert outcome.status == "abandoned", outcome.diagnostic
    root = outcome.bundle_root
    assert root is not None
    payload, text = _failure_diagnostics(root)[0]
    assert text is not None and payload.detail_artifact is not None
    assert secret not in text
    assert WITHHELD in text
    assert "exception: SupervisionError" in text and "phase: scoring" in text


@pytest.mark.asyncio
async def test_a_successful_score_writes_no_diagnostic(tmp_path: Path, episode_id: str) -> None:
    """No extra artifact, no extra event: the success path is byte-unchanged."""

    adapter = FakeAdapter(tmp_path, episode_id)
    outcome = await _runner(
        tmp_path, episode_id, adapter=adapter, model=ScriptedModel(["step", "finish"])
    ).run()

    assert outcome.status == "completed", outcome.diagnostic
    root = outcome.bundle_root
    assert root is not None
    report = verify_bundle(root)
    assert report.valid
    assert [event.kind for event in report.events].count("error") == 0
    # A successful run already publishes execution artifacts; the assertion is
    # that no FAILURE diagnostic was added to them.
    assert not any(
        b"layer: " in path.read_bytes() for path in (root / "artifacts").iterdir() if path.is_file()
    )


@pytest.mark.asyncio
async def test_an_evidence_write_failure_names_the_failed_write(
    tmp_path: Path, episode_id: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The second measured class: an episode dying as ``evidence-write-failed``.

    A run whose evidence write fails abandons with a fixed literal, so the
    raising syscall was nowhere on disk. The diagnostic names the operation
    that failed, the errno symbol, its strerror and the bundle path element --
    never the content -- while the terminal keeps today's status and code.
    """

    from local_operator.evaluation.evidence.store import EvidenceBundleInvalid

    original_append = EvidenceWriter.append

    def failing_append(self: Any, kind: Any, payload: Any, **kwargs: Any) -> Any:
        if kind == "model_request":
            io_error = OSError(errno.ENOSPC, "No space left on device", "events.jsonl")
            raise EvidenceBundleInvalid("journal write refused") from io_error
        return original_append(self, kind, payload, **kwargs)

    monkeypatch.setattr(EvidenceWriter, "append", failing_append)
    adapter = FakeAdapter(tmp_path, episode_id)
    outcome = await _runner(
        tmp_path, episode_id, adapter=adapter, model=ScriptedModel(["step", "finish"])
    ).run()

    assert outcome.status == "abandoned", outcome.diagnostic
    root = outcome.bundle_root
    assert root is not None
    report = verify_bundle(root)
    assert report.valid, [issue.code for issue in report.issues]
    assert report.abandonment is not None
    assert report.abandonment.reason == "infrastructure_failure"
    assert report.abandonment.diagnostic_code == "evidence-write-failed"
    diagnostics = _failure_diagnostics(root)
    assert len(diagnostics) == 1
    payload, text = diagnostics[0]
    assert text is not None
    assert payload.diagnostic_code == "evidence-evidencefailure"
    assert "layer: evidence" in text and "phase: evidence" in text
    assert "operation: append:model_request" in text
    assert "exception: _EvidenceFailure" in text
    assert "errno: ENOSPC" in text
    assert "strerror: No space left on device" in text
    assert "path: events.jsonl" in text


@pytest.mark.asyncio
async def test_an_escape_from_the_failure_path_still_leaves_a_terminal(
    tmp_path: Path, episode_id: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Class 3, the measured one: the writer is open and NOTHING was recorded.

    Two OSWorld episodes (task_015, task_021) exited rc=1 after a passed
    preflight with three events and ``state: open``. The escape is inside the
    failure path itself -- here a raw ``OSError`` out of ``_finalize_failure``'s
    own append, which ``_append`` does not wrap -- and ``_run_with_bundle``
    used to catch only ``_EvidenceFailure``, so the process left with the
    writer closed and no terminal.
    """

    async def exploding_finalize(self: Any, error: Any) -> Any:
        raise RuntimeError("harness bug inside the failure path")

    monkeypatch.setattr(EpisodeRunner, "_finalize_failure", exploding_finalize)
    adapter = FakeAdapter(
        tmp_path,
        episode_id,
        failures={"execute": SupervisionError("worker died")},
        fail_after={"execute": 1},
    )
    outcome = await _runner(
        tmp_path, episode_id, adapter=adapter, model=ScriptedModel(["step", "step", "finish"])
    ).run()

    assert outcome.status == "abandoned", outcome.diagnostic
    root = outcome.bundle_root
    assert root is not None
    report = verify_bundle(root)
    assert report.valid, [issue.code for issue in report.issues]
    assert report.abandonment is not None
    assert report.abandonment.reason == "infrastructure_failure"
    assert report.abandonment.diagnostic_code == "runtimeerror"
    diagnostics = _failure_diagnostics(root)
    assert len(diagnostics) == 1
    payload, text = diagnostics[0]
    assert text is not None
    assert payload.diagnostic_code == "internal-runtimeerror"
    assert "layer: internal" in text
    assert "phase: run" in text and "operation: execute" in text
    assert "exception: RuntimeError" in text
    assert "harness bug inside the failure path" in text


@pytest.mark.asyncio
async def test_an_escape_after_finalization_abandons_as_ambiguous(
    tmp_path: Path, episode_id: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The same handler after ``finalization_start`` must use the legal reason.

    ``finalization_start`` is durable by then, and the abandonment model
    refuses finalization authority on any reason but
    ``ambiguous_finalization`` -- so a handler that always claimed
    ``infrastructure_failure`` would itself fail exactly on the runs that got
    furthest.
    """

    def broken_reconciliation(self: Any, payload: Any, **kwargs: Any) -> Any:
        raise OSError(errno.EIO, "Input/output error", "events.jsonl")

    monkeypatch.setattr(EvidenceWriter, "record_reconciliation", broken_reconciliation)
    adapter = FakeAdapter(tmp_path, episode_id)
    outcome = await _runner(
        tmp_path, episode_id, adapter=adapter, model=ScriptedModel(["step", "finish"])
    ).run()

    assert outcome.status == "abandoned", outcome.diagnostic
    root = outcome.bundle_root
    assert root is not None
    report = verify_bundle(root)
    assert report.valid, [issue.code for issue in report.issues]
    assert report.abandonment is not None
    assert report.abandonment.reason == "ambiguous_finalization"
    # R3: the raw OSError is wrapped at the write wrapper now, so the diagnostic
    # names the receipt that failed and still carries the errno it raised.
    payload, text = _failure_diagnostics(root)[0]
    assert text is not None
    assert "operation: receipt:reconciliation" in text
    assert "errno: EIO" in text and "path: events.jsonl" in text


def _diagnostic_bytes_write(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ONLY the diagnostic artifact's own bytes fail to write (ENOSPC).

    The journal's writes are JSON lines; the diagnostic's are the text body
    this artifact publishes, so the prefix discriminates them without touching
    any other durable write. A plain write error is exactly the R2 shape: it is
    NOT a redaction refusal, and the old code poisoned the writer on it.
    """

    from local_operator.evaluation.evidence import store as store_module

    original = store_module._write_all

    def failing_write_all(fd: int, data: bytes, calls: Any) -> None:
        if data.startswith(b"layer: "):
            raise OSError(errno.ENOSPC, "No space left on device", "artifacts/.artifact.tmp")
        original(fd, data, calls)

    monkeypatch.setattr(store_module, "_write_all", failing_write_all)


def _no_diagnostic(monkeypatch: pytest.MonkeyPatch) -> None:
    """The same-code CONTROL: the pre-change runner, which published nothing."""

    monkeypatch.setattr(EpisodeRunner, "_publish_failure_diagnostic", lambda *a, **k: None)


DIAGNOSTIC_FAILURES = [
    "render",
    "publish",
    "record",
    "sweep",
]


@pytest.mark.asyncio
@pytest.mark.parametrize("way", DIAGNOSTIC_FAILURES)
async def test_the_diagnostic_never_changes_the_terminal(
    tmp_path: Path, episode_id: str, monkeypatch: pytest.MonkeyPatch, way: str
) -> None:
    """FAIL-OPEN, one case per way the diagnostic can fail.

    The acceptance criterion for this branch is an EQUALITY with the pre-change
    runner, not merely "something reasonable": each case runs a control with the
    diagnostic disabled entirely and compares status and diagnostic string.
    """

    import local_operator.evaluation.runner.episode as episode_module
    from local_operator.evaluation.evidence.store import EvidenceBundleInvalid

    async def scenario(root: Path, ident: str) -> Any:
        adapter = FakeAdapter(
            root, ident, failures={"score": SupervisionError("judge backend returned 502")}
        )
        return await _runner(root, ident, adapter=adapter, model=ScriptedModel()).run()

    control_root = tmp_path / "control"
    control_root.mkdir()
    _no_diagnostic(monkeypatch)
    control = await scenario(control_root, f"{episode_id}-control")
    monkeypatch.undo()

    if way == "render":
        monkeypatch.setattr(
            episode_module,
            "_failure_diagnostic_text",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("renderer broke")),
        )
    elif way == "publish":
        _diagnostic_bytes_write(monkeypatch)
    elif way == "record":
        monkeypatch.setattr(
            EvidenceWriter,
            "record_failure_diagnostic",
            lambda *a, **k: (_ for _ in ()).throw(EvidenceBundleInvalid("event refused")),
        )
    else:
        monkeypatch.setattr(
            EvidenceWriter,
            "discard_unreferenced_artifacts",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("sweep broke")),
        )

    subject_root = tmp_path / "subject"
    subject_root.mkdir()
    subject = await scenario(subject_root, f"{episode_id}-subject")

    assert subject.status == control.status, (way, subject.diagnostic, control.diagnostic)
    assert subject.diagnostic == control.diagnostic, (way, subject.diagnostic)
    assert subject.bundle_root is not None
    assert verify_bundle(subject.bundle_root).valid


@pytest.mark.asyncio
@pytest.mark.parametrize("way", DIAGNOSTIC_FAILURES)
async def test_the_evidence_path_diagnostic_never_changes_the_terminal(
    tmp_path: Path, episode_id: str, monkeypatch: pytest.MonkeyPatch, way: str
) -> None:
    """The same four fail-open cases on the EVIDENCE-WRITE path.

    It reaches ``_abandon_for_evidence`` instead of the scoring path, so it
    gets its own control: a failure in this path must leave ``abandoned`` and
    ``evidence-write-failed`` exactly as the pre-change runner did.
    """

    import local_operator.evaluation.runner.episode as episode_module
    from local_operator.evaluation.evidence.store import EvidenceBundleInvalid

    original_append = EvidenceWriter.append

    def failing_append(self: Any, kind: Any, payload: Any, **kwargs: Any) -> Any:
        if kind == "model_request":
            raise OSError(errno.EIO, "Input/output error", "events.jsonl")
        return original_append(self, kind, payload, **kwargs)

    def apply_way(faults: pytest.MonkeyPatch) -> None:
        if way == "render":
            faults.setattr(
                episode_module,
                "_failure_diagnostic_text",
                lambda *a, **k: (_ for _ in ()).throw(RuntimeError("renderer broke")),
            )
        elif way == "publish":
            _diagnostic_bytes_write(faults)
        elif way == "record":
            faults.setattr(
                EvidenceWriter,
                "record_failure_diagnostic",
                lambda *a, **k: (_ for _ in ()).throw(EvidenceBundleInvalid("event refused")),
            )
        else:
            faults.setattr(
                EvidenceWriter,
                "discard_unreferenced_artifacts",
                lambda *a, **k: (_ for _ in ()).throw(RuntimeError("sweep broke")),
            )

    async def scenario(root: Path, ident: str, *, control: bool) -> Any:
        with monkeypatch.context() as faults:
            faults.setattr(EvidenceWriter, "append", failing_append)
            if control:
                faults.setattr(EpisodeRunner, "_publish_failure_diagnostic", lambda *a, **k: None)
            apply_way(faults)
            adapter = FakeAdapter(root, ident)
            return await _runner(
                root, ident, adapter=adapter, model=ScriptedModel(["step", "finish"])
            ).run()

    control_root = tmp_path / "control"
    control_root.mkdir()
    control = await scenario(control_root, f"{episode_id}-control", control=True)
    subject_root = tmp_path / "subject"
    subject_root.mkdir()
    subject = await scenario(subject_root, f"{episode_id}-subject", control=False)

    assert subject.status == control.status, (way, subject.diagnostic, control.diagnostic)
    assert subject.diagnostic == control.diagnostic, (way, subject.diagnostic)
    assert subject.bundle_root is not None
    assert verify_bundle(subject.bundle_root).valid


@pytest.mark.asyncio
async def test_a_poisoning_publish_failure_would_have_lost_the_bundle(
    tmp_path: Path, episode_id: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R2's measured control, kept: the diagnostic's ENOSPC must not poison.

    Before the best-effort flag this exact injection produced
    ``abandonment_failed`` (the writer poisoned by a disposable 3 KB write)
    while the no-diagnostic control abandoned cleanly. The status is now the
    control's, and the event still names the class.
    """

    _diagnostic_bytes_write(monkeypatch)
    adapter = FakeAdapter(
        tmp_path, episode_id, failures={"score": SupervisionError("judge backend returned 502")}
    )
    outcome = await _runner(tmp_path, episode_id, adapter=adapter, model=ScriptedModel()).run()

    assert outcome.status == "abandoned", outcome.diagnostic
    root = outcome.bundle_root
    assert root is not None
    assert verify_bundle(root).valid
    payload, text = _failure_diagnostics(root)[0]
    assert text is None and payload.detail_artifact is None
    assert payload.diagnostic_code == "adapter-supervisionerror"


@pytest.mark.asyncio
async def test_a_refused_diagnostic_event_does_not_orphan_the_scoring_bundle(
    tmp_path: Path, episode_id: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R1: the scoring path sweeps its own orphan.

    A refused event after a successful publish left an unreferenced artifact,
    and ``abandon()`` refuses that -- the scoring path reported
    ``abandonment_failed`` where the pre-change runner abandoned. The sweep
    removes the orphan before the terminal.
    """

    from local_operator.evaluation.evidence.store import EvidenceBundleInvalid

    monkeypatch.setattr(
        EvidenceWriter,
        "record_failure_diagnostic",
        lambda *a, **k: (_ for _ in ()).throw(EvidenceBundleInvalid("event refused")),
    )
    adapter = FakeAdapter(
        tmp_path, episode_id, failures={"score": SupervisionError("judge backend returned 502")}
    )
    outcome = await _runner(tmp_path, episode_id, adapter=adapter, model=ScriptedModel()).run()

    assert outcome.status == "abandoned", outcome.diagnostic
    root = outcome.bundle_root
    assert root is not None
    report = verify_bundle(root)
    assert report.valid, [issue.code for issue in report.issues]
    # The diagnostic's own bytes were swept; execution artifacts that the
    # journal names are untouched.
    assert not any(
        b"layer: " in path.read_bytes() for path in (root / "artifacts").iterdir() if path.is_file()
    )
    assert report.abandonment is not None
    assert report.abandonment.reason == "ambiguous_finalization"


@pytest.mark.asyncio
async def test_a_validation_error_never_publishes_its_input(
    tmp_path: Path, episode_id: str
) -> None:
    """R4: ``input_value=`` can quote the secret that failed validation."""

    from pydantic import BaseModel, ValidationError

    secret = "[redacted]"

    class _Wire(BaseModel):
        value: int

    with pytest.raises(ValidationError) as caught:
        _Wire(value=secret)  # type: ignore[arg-type]
    failure = caught.value
    assert secret in str(failure)  # the leak this test closes, in the raw text

    adapter = FakeAdapter(tmp_path, episode_id, failures={"score": failure})
    outcome = await _runner(tmp_path, episode_id, adapter=adapter, model=ScriptedModel()).run()

    assert outcome.status == "abandoned", outcome.diagnostic
    root = outcome.bundle_root
    assert root is not None
    payload, text = _failure_diagnostics(root)[0]
    assert text is not None
    assert "ValidationError" in text and "value" in text
    assert secret not in text
    assert "input_value" not in text


@pytest.mark.asyncio
async def test_a_cancellation_records_its_terminal_and_still_raises(
    tmp_path: Path, episode_id: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R6: stopped work must not be reported as a normal outcome.

    The terminal is written synchronously (a cancelled task re-raises at every
    await), then the cancellation is re-raised to the caller.
    """

    async def cancelled_execute(self: Any, handshake: Any) -> Any:
        raise asyncio.CancelledError()

    monkeypatch.setattr(EpisodeRunner, "_execute", cancelled_execute)
    adapter = FakeAdapter(tmp_path, episode_id)

    with pytest.raises(asyncio.CancelledError):
        await _runner(tmp_path, episode_id, adapter=adapter, model=ScriptedModel()).run()

    roots = list((tmp_path / "evidence").iterdir())
    assert len(roots) == 1
    report = verify_bundle(roots[0])
    assert report.valid, [issue.code for issue in report.issues]
    assert report.abandonment is not None
    assert report.abandonment.diagnostic_code == "cancellederror"


def test_a_deadline_is_not_an_evidence_failure() -> None:
    """R5: ``TimeoutError``/``ConnectionError`` are OSError subclasses.

    Classifying by ``isinstance(error, OSError)`` before the adapter branch
    filed a judge deadline or a dropped connection as an evidence-layer
    failure -- the wrong fix for the reader, whose next step depends on it.
    """

    from local_operator.evaluation.evidence.store import EvidenceBundleInvalid

    assert _failure_layer(TimeoutError("judge deadline")) == "adapter"
    assert _failure_layer(ConnectionError("judge unreachable")) == "adapter"
    assert _failure_layer(SupervisionError("worker died")) == "adapter"
    assert _failure_layer(OSError(errno.EIO, "Input/output error")) == "evidence"
    assert _failure_layer(EvidenceBundleInvalid("bundle refused")) == "evidence"
    assert _failure_layer(RuntimeError("bug")) == "internal"


@pytest.mark.asyncio
async def test_failed_rescue_gets_one_immediate_retry(tmp_path: Path, episode_id: str) -> None:
    """A single fast rescue failure must not strand the instance.

    Measured leak (2026-10-17, ep-4386740c6af0): the rescue failed once, fast
    and silently, its descriptor stayed in the inbox and its instance ran for
    hours. The retry is bounded at one extra attempt so a genuinely broken
    rescue cannot loop; the still-broken case keeps the descriptor for the
    sweep, and the complete case retires it.
    """

    from local_operator.evaluation.adapters.supervisor import load_pending_rescue

    attempts: list[int] = []

    async def _flaky(descriptor: Any, **kwargs: Any) -> Any:
        del kwargs
        attempts.append(1)

        class _Aggregate:
            descriptor_id = descriptor.descriptor_id
            complete = len(attempts) >= 2

        return _Aggregate()

    adapter = FakeAdapter(tmp_path, episode_id, failures={"score": SupervisionError("scorer died")})
    runner = _runner(tmp_path, episode_id, adapter=adapter, model=ScriptedModel())
    runner._rescue = _flaky  # pyright: ignore[reportAttributeAccessIssue]

    outcome = await runner.run()

    assert outcome.status == "abandoned"
    assert outcome.rescue_complete is True
    assert len(attempts) == 2
    # A complete aggregate retires the descriptor: the sweep has nothing left.
    assert load_pending_rescue(build_config(tmp_path).rescue_root) is None

    second = tmp_path / "still-broken"
    second.mkdir()
    broken_attempts: list[int] = []

    async def _broken(descriptor: Any, **kwargs: Any) -> Any:
        del kwargs
        broken_attempts.append(1)

        class _Aggregate:
            descriptor_id = descriptor.descriptor_id
            complete = False

        return _Aggregate()

    adapter2 = FakeAdapter(second, f"{episode_id}-b", failures={"score": SupervisionError("x")})
    runner2 = _runner(second, f"{episode_id}-b", adapter=adapter2, model=ScriptedModel())
    runner2._rescue = _broken  # pyright: ignore[reportAttributeAccessIssue]

    outcome2 = await runner2.run()

    assert outcome2.rescue_complete is False
    assert len(broken_attempts) == 2  # bounded: exactly one retry
    # Still broken: the descriptor stays for the sweep to pick up.
    assert load_pending_rescue(build_config(second).rescue_root) is not None

    # Episode-level budget: the bound is per episode, not per call site
    # (both the close-out and evidence-failure paths can ask), so a third
    # launch is refused outright and the sweep is left the descriptor.
    assert await runner2._attempt_rescue() is False
    assert len(broken_attempts) == 2


@pytest.mark.asyncio
async def test_incomplete_cleanup_labels_the_run_and_requires_rescue(
    tmp_path: Path, episode_id: str
) -> None:
    adapter = FakeAdapter(tmp_path, episode_id, cleanup_status="failed")
    runner = _runner(tmp_path, episode_id, adapter=adapter, model=ScriptedModel())

    outcome = await runner.run()

    # The task scored, but a leaked resource still makes the episode failed.
    assert outcome.status == "failed"
    assert outcome.rescue_required is True
    assert outcome.reportability_label == "cleanup_incomplete"
    root = outcome.bundle_root
    assert root is not None
    assert verify_bundle(root).valid
    cleanups = payloads(root, CleanupPayload)
    assert [cleanup.rescue_required for cleanup in cleanups] == [True]


@pytest.mark.asyncio
async def test_cleanup_incomplete_outranks_unscored_in_the_label(
    tmp_path: Path, episode_id: str
) -> None:
    """Label precedence: cleanup_incomplete > budget_unreconciled > unscored."""

    adapter = FakeAdapter(
        tmp_path,
        episode_id,
        cleanup_status="failed",
        failures={"execute": SupervisionError("worker died")},
    )
    runner = _runner(tmp_path, episode_id, adapter=adapter, model=ScriptedModel(["step", "finish"]))

    outcome = await runner.run()

    assert outcome.score is not None and outcome.score.status == "unscored"
    assert outcome.reportability_label == "cleanup_incomplete"


@pytest.mark.asyncio
async def test_unscored_score_artifact_is_never_a_zero_binary(
    tmp_path: Path, episode_id: str
) -> None:
    adapter = FakeAdapter(tmp_path, episode_id)
    runner = _runner(
        tmp_path,
        episode_id,
        adapter=adapter,
        model=ScriptedModel(error=RuntimeError("down")),
    )

    outcome = await runner.run()

    assert outcome.score is not None
    assert outcome.score.status == "unscored"
    assert outcome.score.binary is None


@pytest.mark.asyncio
async def test_failure_before_prepare_has_no_bundle_to_write(
    tmp_path: Path, episode_id: str
) -> None:
    """The manifest needs the cleanup plan prepare returns, so none exists yet."""

    adapter = FakeAdapter(
        tmp_path, episode_id, failures={"prepare": SupervisionError("prepare died")}
    )
    runner = _runner(tmp_path, episode_id, adapter=adapter, model=ScriptedModel())

    outcome = await runner.run()

    assert outcome.status == "failed_pre_bundle"
    assert outcome.bundle_root is None
    assert adapter.terminated


@pytest.mark.asyncio
async def test_rescue_descriptor_is_persisted_before_prepare_and_updated_after(
    tmp_path: Path, episode_id: str
) -> None:
    """Stage 1 unblocks prepare; stage 2 lands the real plan before reset_start."""

    from local_operator.evaluation.adapters.supervisor import load_pending_rescue

    seen: list[tuple[str, tuple[str, ...]]] = []
    adapter = FakeAdapter(tmp_path, episode_id)
    config = build_config(tmp_path)
    original = adapter._call_raw

    async def watching(method: Any, params: Any, result_type: Any, **kwargs: Any) -> Any:
        if method in ("prepare", "reset_start"):
            pending = load_pending_rescue(config.rescue_root)
            assert pending is not None, f"{method} ran with no persisted rescue"
            seen.append((method, tuple(a.action_id for a in pending.cleanup_plan.actions)))
        return await original(method, params, result_type, **kwargs)

    adapter._call_raw = watching  # type: ignore[method-assign]
    runner = EpisodeRunner(
        build_spec(episode_id),
        config,
        selector=selector(tmp_path),
        model=ScriptedModel(),
        launch=lambda _: adapter,
        rescue=_rescue_ok,
    )

    outcome = await runner.run()

    assert outcome.status == "completed"
    # prepare sees the provisional close-session plan; reset_start, the side
    # effect boundary, already sees the adapter's real plan.
    assert seen == [
        ("prepare", ("close-session",)),
        ("reset_start", ("release",)),
    ]


@pytest.mark.asyncio
async def test_scored_zero_is_distinct_from_unscored(tmp_path: Path, episode_id: str) -> None:
    adapter = FakeAdapter(tmp_path, episode_id, score=ScoreArtifact(status="scored", binary=0))
    runner = _runner(tmp_path, episode_id, adapter=adapter, model=ScriptedModel())

    outcome = await runner.run()

    assert outcome.status == "completed"
    assert outcome.score is not None
    assert outcome.score.status == "scored" and outcome.score.binary == 0
    assert outcome.reportability_label == "reportable"
    root = outcome.bundle_root
    assert root is not None
    assert verify_bundle(root).valid


@pytest.mark.asyncio
async def test_a_wrapped_channel_reply_survives_episode_recording(
    tmp_path: Path, episode_id: str
) -> None:
    """B1 regression: every re-reader of an accepted reply reads it the same way.

    The episode records each accepted reply by RE-DECODING it
    (``redact_public_reply``); a reply accepted only by the channel's
    structural reading must carry that fact across the boundary
    (``ModelDecision.channel_arguments``), or the re-decode refuses OUTSIDE
    every rejection handler and the episode seals failed/unscored as an
    adapter error. The wrapped-with-note shape is the r3 family this change
    exists to recover -- the note is exactly what made the pre-fix re-decode
    raise -- while the no-note and bare spellings pin that the ordinary paths
    did not move.
    """

    # Deferred: test_public_reply imports this module for ``_rescue_ok``.
    from tests.unit.evaluation.runner.test_provider_client import (
        _client,
        finish_payload,
    )
    from tests.unit.evaluation.runner.test_public_reply import envelope
    from tests.unit.evaluation.runner.test_reply_channel import ChannelStream, _spec

    cases = (
        (
            "wrapped-with-note",
            lambda current: json.dumps(
                {"reply_json": envelope(finish_payload(current), "Live notes")}
            ),
        ),
        ("wrapped-no-note", lambda current: json.dumps({"reply_json": finish_payload(current)})),
        ("bare-with-note", lambda current: envelope(finish_payload(current), "Live notes")),
    )
    for label, build in cases:
        case_episode = f"{episode_id}-{label}"
        current = observation(case_episode, 0)
        client = _client(ChannelStream(build(current)), model_spec=_spec(supports_tools=True))
        adapter = FakeAdapter(
            tmp_path, case_episode, score=ScoreArtifact(status="scored", binary=1)
        )
        runner = _runner(tmp_path, case_episode, adapter=adapter, model=client)

        outcome = await runner.run()

        assert outcome.status == "completed", (label, outcome.status)
        assert outcome.score is not None and outcome.score.status == "scored", label
        assert outcome.bundle_root is not None and verify_bundle(outcome.bundle_root).valid, label


@pytest.mark.asyncio
async def test_a_hidden_second_batch_in_an_actions_string_never_executes(
    tmp_path: Path, episode_id: str
) -> None:
    """Round-2 residual (M1/Q3): fail-closed at the runner, nothing executed.

    Pre-fix the runner COMPLETED / SCORED on this reply -- the visible batch
    executed and the hidden same-observation batch was silently dropped
    (measured on the real runner). With the string reader carrying the channel
    fact the reply is rejected at the real client, the corrective path is
    exhausted against the static stream, and the episode seals failed/unscored
    with ``model``-category ``decision-rejected`` errors -- and the adapter
    never sees an ``execute`` for either batch.
    """

    # Deferred: test_public_reply imports this module for ``_rescue_ok``.
    from tests.unit.evaluation.runner.test_provider_client import (
        _client,
        finish_payload,
    )
    from tests.unit.evaluation.runner.test_reply_channel import ChannelStream, _spec

    current = observation(episode_id, 0)
    action = json.loads(finish_payload(current))["actions"][0]
    array = json.dumps([dict(action)])
    hidden = json.dumps({"zzz": {"actions": [dict(action)]}})
    reply = json.dumps({"actions": array + " , " + hidden})

    client = _client(ChannelStream(reply), model_spec=_spec(supports_tools=True))
    adapter = FakeAdapter(tmp_path, episode_id, score=ScoreArtifact(status="scored", binary=1))
    runner = _runner(tmp_path, episode_id, adapter=adapter, model=client)

    outcome = await runner.run()

    assert outcome.status == "failed"
    assert outcome.score is not None and outcome.score.status == "unscored"
    assert "execute" not in adapter.calls
    assert outcome.bundle_root is not None
    rejected = [
        error
        for error in payloads(outcome.bundle_root, ErrorPayload)
        if error.diagnostic_code == "decision-rejected"
    ]
    assert rejected and all(error.category == "model" for error in rejected)


@pytest.mark.asyncio
async def test_route_change_is_not_sealed_as_comparable(tmp_path: Path, episode_id: str) -> None:
    """A silent provider fallback is exactly what comparability exists to catch."""

    from local_operator.evaluation.evidence.models import RouteIdentity

    adapter = FakeAdapter(tmp_path, episode_id)
    model = ScriptedModel(["finish"])
    model.route = RouteIdentity(
        provider_id="other-provider", route_id="other-route", model_id="other-model"
    )
    runner = _runner(tmp_path, episode_id, adapter=adapter, model=model)

    outcome = await runner.run()

    assert outcome.status == "completed"
    assert outcome.comparability_label == "route_changed"
    root = outcome.bundle_root
    assert root is not None
    report = verify_bundle(root)
    assert report.valid
    assert report.outcome is not None
    assert report.outcome.comparable is False


@pytest.mark.asyncio
async def test_pinned_route_stays_comparable(tmp_path: Path, episode_id: str) -> None:
    adapter = FakeAdapter(tmp_path, episode_id)
    runner = _runner(tmp_path, episode_id, adapter=adapter, model=ScriptedModel())

    outcome = await runner.run()

    assert outcome.comparability_label == "comparable"
    root = outcome.bundle_root
    assert root is not None
    report = verify_bundle(root)
    assert report.outcome is not None
    assert report.outcome.comparable is True


# ---------------------------------------------------------------------------
# Managed context and guards
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_model_sees_each_prior_turn_with_its_batch(tmp_path: Path, episode_id: str) -> None:
    """The runner hands the client the turns it took, each closed with the
    batch that was executed on it, and the current turn still undecided."""

    adapter = FakeAdapter(tmp_path, episode_id)
    model = ScriptedModel(["step", "step", "finish"])
    runner = _runner(tmp_path, episode_id, adapter=adapter, model=model)

    await runner.run()

    assert [len(history) for history in model.histories] == [1, 2, 3]
    third = model.histories[2]
    assert third[0].batch is not None and third[1].batch is not None
    assert third[2].batch is None
    assert third[0].observation.sequence == 0 and third[2].observation.sequence == 2
    # The current observation is the last turn's.
    assert third[2].observation is not None


@pytest.mark.asyncio
async def test_context_compaction_event_is_recorded_and_verifies(
    tmp_path: Path, episode_id: str
) -> None:
    from local_operator.evaluation.evidence.models import (
        ContextCompactionPayload,
        ModelRequestPayload,
    )

    adapter = FakeAdapter(tmp_path, episode_id)
    model = ScriptedModel(["step", "step", "finish"], compact_on=[1])
    runner = _runner(tmp_path, episode_id, adapter=adapter, model=model)

    outcome = await runner.run()

    root = outcome.bundle_root
    assert root is not None
    report = verify_bundle(root)
    assert report.valid, [issue.code for issue in report.issues]
    assert report.counters is not None and report.counters.compactions == 1
    kinds = _kinds(root)
    index = kinds.index("context_compaction")
    # Between the previous request's usage receipt and the next request.
    assert kinds[index - 1] == "observation"
    assert kinds[index - 2] == "environment_step"
    assert kinds[index + 1] == "model_request"
    assert kinds.count("context_compaction") == 1
    compaction = payloads(root, ContextCompactionPayload)[0]
    requests = payloads(root, ModelRequestPayload)
    assert compaction.previous_request_id == requests[0].request_id
    assert compaction.strategy == "context-full"
    assert compaction.frames_dropped == 2
    assert compaction.summary_artifact is not None
    assert (root / "artifacts" / compaction.summary_artifact.sha256).read_bytes() == (
        b"what happened so far"
    )
    assert all(request.prompt_cache_key == "lop-eval-test" for request in requests)


@pytest.mark.asyncio
@pytest.mark.parametrize("stationary", [True, False])
async def test_guard_truncation_seals_scored_with_reason(
    tmp_path: Path, episode_id: str, monkeypatch: pytest.MonkeyPatch, stationary: bool
) -> None:
    """A stationary loop stops early, but repeated input with text progress
    reaches the step cap. Both paths score and seal a verifiable bundle."""

    from local_operator.evaluation.adapters.api import observation_content_id
    from local_operator.evaluation.protocol import Observation
    from local_operator.evaluation.runner.guards import RepeatedBatchGuard
    from tests.unit.evaluation.runner import conftest

    if stationary:
        original_observation = conftest.observation

        def observation(episode_id: str, sequence: int, *, text: str = "state") -> Observation:
            # Keep real sequence/identity validation while removing visible
            # progress; the normal fixture's changing text is not a loop.
            current = original_observation(episode_id, sequence, text=text).model_copy(
                update={"text": text}
            )
            return current.model_copy(update={"observation_id": observation_content_id(current)})

        monkeypatch.setattr(conftest, "observation", observation)

    adapter = FakeAdapter(tmp_path, episode_id)
    runner = EpisodeRunner(
        build_spec(episode_id),
        build_config(tmp_path, max_steps=4, guards=(RepeatedBatchGuard(repeats=2),)),
        selector=selector(tmp_path),
        model=ScriptedModel(["type"] * 8),
        launch=lambda _: adapter,
        rescue=_rescue_ok,
    )

    outcome = await runner.run()

    assert outcome.status == "completed"
    assert outcome.score is not None and outcome.score.status == "scored"
    root = outcome.bundle_root
    assert root is not None
    report = verify_bundle(root)
    assert report.valid, [issue.code for issue in report.issues]
    steps = payloads(root, EnvironmentStepPayload)
    expected_steps = 2 if stationary else 4
    expected_reason = "repeated-batch" if stationary else "max-steps"
    assert len(steps) == expected_steps
    assert [step.truncated for step in steps] == [False] * (expected_steps - 1) + [True]
    assert [step.truncation_reason for step in steps] == [None] * (expected_steps - 1) + [
        expected_reason
    ]


@pytest.mark.asyncio
async def test_a_waiting_model_runs_to_the_step_cap_under_default_guards(
    tmp_path: Path, episode_id: str
) -> None:
    """End to end for M1: a model that only waits (a slow load) reaches
    ``max_steps`` under the DEFAULT guards rather than being cut off as
    ``repeated-batch``; the truncation it gets is the honest step cap."""

    adapter = FakeAdapter(tmp_path, episode_id)
    runner = _runner(
        tmp_path, episode_id, adapter=adapter, model=ScriptedModel(["step"] * 12), max_steps=8
    )

    outcome = await runner.run()

    assert outcome.status == "completed"
    root = outcome.bundle_root
    assert root is not None
    assert verify_bundle(root).valid
    steps = payloads(root, EnvironmentStepPayload)
    assert len(steps) == 8
    assert steps[-1].truncation_reason == "max-steps"


@pytest.mark.asyncio
async def test_a_bounded_episode_is_not_cut_by_the_cost_rate_ratio(
    tmp_path: Path, episode_id: str
) -> None:
    """The regression this guards: an episode that states a step budget is
    judged by that budget, so a per-cycle cost jump (a prompt-cache miss) no
    longer truncates it at the ratio's first full windows -- it reaches the
    step budget instead."""

    adapter = FakeAdapter(tmp_path, episode_id)
    # Ten cheap cycles then a ten-cycle jump: a 3x ratio guard fires on this.
    model = ScriptedModel(["step"] * 40, cost_micros=[7] * 20 + [70] * 4)
    runner = EpisodeRunner(
        build_spec(episode_id),
        build_config(tmp_path, max_steps=24),
        selector=selector(tmp_path),
        model=model,
        launch=lambda _: adapter,
        rescue=_rescue_ok,
    )

    outcome = await runner.run()

    assert outcome.status == "completed"
    root = outcome.bundle_root
    assert root is not None
    assert verify_bundle(root).valid
    steps = payloads(root, EnvironmentStepPayload)
    assert len(steps) == 24
    assert [step.truncation_reason for step in steps] == [None] * 23 + ["max-steps"]


@pytest.mark.asyncio
async def test_a_bounded_episode_is_stopped_by_its_cost_cap_not_by_a_prorated_pace(
    tmp_path: Path, episode_id: str
) -> None:
    """The caps are the authority, and the cost cap is the only COST stop a
    bounded episode has: 300_000 micro-USD per cycle is unremarkable until the
    1_000_000 the operator declared is actually reached, at step 4.

    Before this the same episode was cut at step 2 for exceeding a per-cycle
    pace prorated from the remaining budget -- a pace no cycle of a real
    long-horizon episode can keep. That prorated pace is the defect: it cut the
    campaign's ``batch-k3-canary6/task_010`` at step 337 of 500, and replayed
    over a lane that finished and scored 50.00% it would have cut that lane on
    its eleventh cycle.
    """

    adapter = FakeAdapter(tmp_path, episode_id)
    model = ScriptedModel(["step"] * 20, cost_micros=300_000)
    runner = EpisodeRunner(
        build_spec(episode_id, caps={"provider_usd_micros": 1_000_000}),
        build_config(tmp_path, max_steps=8),
        selector=selector(tmp_path),
        model=model,
        launch=lambda _: adapter,
        rescue=_rescue_ok,
    )

    outcome = await runner.run()

    assert outcome.status == "completed"
    root = outcome.bundle_root
    assert root is not None
    assert verify_bundle(root).valid
    steps = payloads(root, EnvironmentStepPayload)
    assert len(steps) == 4
    assert steps[-1].truncated is True
    assert steps[-1].truncation_reason == "budget-cap"
    # ...and the reason an operator reads is the cap's own number, not a
    # diagnosis of pace.
    assert outcome.truncation_reason == "budget-cap"
    assert outcome.truncation_detail is not None
    assert "provider_usd_micros" in outcome.truncation_detail


#: Per-cycle cost profiles a bounded episode may legitimately show, repeated to
#: their last value. Each would trip the ratio guard on its own (a cache miss, a
#: growing context, a runaway) and none may reach ``cost-spike`` when the
#: episode states both its own step budget and its own cost cap.
_BOUNDED_COST_PROFILES: dict[str, list[int]] = {
    "flat": [7],
    "cache-miss-spike": [7] * 20 + [70],
    "long-horizon-growth": [7_000] * 10 + [63_000],
    "runaway": [500_000],
}


@pytest.mark.parametrize("profile", sorted(_BOUNDED_COST_PROFILES))
@pytest.mark.asyncio
async def test_a_doubly_capped_run_cannot_end_as_a_cost_spike(
    tmp_path: Path, episode_id: str, profile: str
) -> None:
    """End to end for the rule the guard now encodes: with the episode's own
    step budget and cost cap both declared, and no operator cycle cap set,
    price is not an authority this episode has. No cost profile can produce
    ``cost-spike``; the only reasons available are the step cap and the cost
    cap, whatever the cycle prices do. (The operator's ``max_cycle_usd`` is
    excluded here on purpose: that one IS an authority, and its firing is
    asserted in ``test_a_guard_truncation_detail_reaches_the_outcome_record``.)
    """

    adapter = FakeAdapter(tmp_path, episode_id)
    model = ScriptedModel(["step"] * 40, cost_micros=_BOUNDED_COST_PROFILES[profile])
    runner = EpisodeRunner(
        build_spec(episode_id, caps={"provider_usd_micros": 6_000_000}),
        build_config(tmp_path, max_steps=24),
        selector=selector(tmp_path),
        model=model,
        launch=lambda _: adapter,
        rescue=_rescue_ok,
    )

    outcome = await runner.run()

    root = outcome.bundle_root
    assert root is not None
    assert verify_bundle(root).valid
    steps = payloads(root, EnvironmentStepPayload)
    reasons = [step.truncation_reason for step in steps]
    assert "cost-spike" not in reasons
    assert reasons[-1] in ("max-steps", "budget-cap")
    assert steps[-1].truncated is True


@pytest.mark.asyncio
async def test_a_guard_truncation_detail_reaches_the_outcome_record(
    tmp_path: Path, episode_id: str
) -> None:
    """The verdict's why is reported, not dropped.

    A truncation analysis used to have to be re-derived from ``events.jsonl``
    by hand: the step payload records the guard's CODE (it must stay a stable
    identifier consumers compare runs on) and the outcome a caller writes out
    -- ``scripts/run_episode.py`` prints it as ``outcome.json``, the record the
    campaign harness collects -- carried neither the code nor the detail. Both
    are on the outcome now, taken from the firing verdict.
    """

    adapter = FakeAdapter(tmp_path, episode_id)
    # The operator's absolute cycle cap, the one price-based stop left: a
    # 1_000 micro-USD cycle against a 100 micro-USD cap fires it on step 1.
    runner = EpisodeRunner(
        build_spec(episode_id),
        build_config(tmp_path, max_steps=8, max_cycle_cost_micros=100),
        selector=selector(tmp_path),
        model=ScriptedModel(["step"] * 8, cost_micros=1_000),
        launch=lambda _: adapter,
        rescue=_rescue_ok,
    )

    outcome = await runner.run()

    root = outcome.bundle_root
    assert root is not None
    assert verify_bundle(root).valid
    steps = payloads(root, EnvironmentStepPayload)
    assert steps[-1].truncation_reason == "cost-spike"
    assert outcome.truncation_reason == "cost-spike"
    assert outcome.truncation_detail == "one model cycle cost 1000 micro-USD (cap 100)"
    record = _outcome_json(outcome)
    assert record["truncation_reason"] == "cost-spike"
    assert record["truncation_detail"] == outcome.truncation_detail


@pytest.mark.asyncio
async def test_a_clean_finish_reports_no_truncation_reason(tmp_path: Path, episode_id: str) -> None:
    """A run that finished on its own must not look truncated in the record a
    caller writes out: the fields are set only by a real truncation."""

    adapter = FakeAdapter(tmp_path, episode_id)
    runner = _runner(tmp_path, episode_id, adapter=adapter, model=ScriptedModel(["finish"]))

    outcome = await runner.run()

    assert outcome.status == "completed"
    assert outcome.truncation_reason is None
    assert outcome.truncation_detail is None
    assert _outcome_json(outcome)["truncation_reason"] is None


@pytest.mark.asyncio
async def test_max_steps_truncation_names_its_reason(tmp_path: Path, episode_id: str) -> None:
    adapter = FakeAdapter(tmp_path, episode_id)
    runner = _runner(
        tmp_path, episode_id, adapter=adapter, model=ScriptedModel(["step"] * 6), max_steps=2
    )

    outcome = await runner.run()

    root = outcome.bundle_root
    assert root is not None
    assert verify_bundle(root).valid
    steps = payloads(root, EnvironmentStepPayload)
    assert [step.truncation_reason for step in steps] == [None, "max-steps"]
    # The step cap reaches the caller's own record too, with no judge to quote:
    # ``max-steps`` is its own why, so the detail stays empty rather than being
    # filled with a sentence nobody wrote.
    assert outcome.truncation_reason == "max-steps"
    assert outcome.truncation_detail is None


def _wait_reply(observation_message: Any) -> str:
    """Reply with a wait bound to whichever observation id the message names."""

    text = observation_message.content[0].text
    observation_id = next(
        line.split(": ", 1)[1] for line in text.splitlines() if line.startswith("Observation ID: ")
    )
    return json.dumps(
        {"actions": [{"kind": "wait", "observation_id": observation_id, "duration_ms": 1}]}
    )


class DirectProviderStream:
    """Answers every request the way a DIRECT provider's API does.

    Tokens, and never a dollar amount -- DeepSeek, Anthropic, OpenAI, Gemini,
    Kimi and xAI all omit ``usage.cost`` (only aggregators precompute a bill).
    That omission is the whole point of this fixture: it is the shape the
    runner used to price at 0.
    """

    def __call__(self, request: Any, signal: Any) -> Any:
        return self._events(request)

    async def _events(self, request: Any) -> Any:
        yield StreamTextDelta(delta=_wait_reply(request.messages[-1]))
        usage = Usage(
            input_tokens=4783,
            output_tokens=443,
            reasoning_tokens=225,
            cache_read_tokens=3840,
            # The on-the-wire spec, as ``stream_with_failover`` stamps it.
            provider="deepseek",
            model_id="deepseek-flash",
        )
        yield StreamUsageEvent(usage=usage)
        yield StreamEndEvent(stop_reason="stop", usage=usage)


@pytest.mark.asyncio
async def test_a_direct_providers_table_price_reaches_the_budget_cap(
    tmp_path: Path, episode_id: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The cap fires on a table-priced spend, end to end through the real client.

    A direct provider reports tokens and no bill, so before this the episode's
    ``provider_cost_micros`` stayed 0 and ``BudgetCapGuard`` could never fire:
    a runaway direct-provider episode was bounded only by its step count. This
    drives the REAL ``ProviderModelClient`` (scripted wire, real pricing path)
    through the REAL runner and pins that the priced spend is what truncates.

    The DeepSeek canary tokens price at 838 micro-USD per cycle, so a cap of
    exactly that figure is reached by the FIRST cycle while the 20-step budget
    is nowhere near binding -- the cap, not the step count, is what stops this
    episode. (The cap is reached at its value, ``>=``, not only exceeded: the
    next cycle would exceed it, and a budget is an authority.)

    838 is DeepSeek's published PEAK table price (the registry stores peak, and
    the schedule halves it off-peak), and the scripted usage carries no stamp of
    its own, so the clock is FROZEN into a peak window: left to the wall clock
    this asserts 838 at 07:00 UTC and half that for the ~79% of the week that is
    off-peak, which is a flake rather than a test. The tariff's own window
    behaviour is pinned in ``tests/unit/model/test_tariff.py``.
    """
    from datetime import datetime, timezone

    from local_operator.model import tariff

    monkeypatch.setattr(tariff, "now_utc", lambda: datetime(2026, 9, 14, 7, 0, tzinfo=timezone.utc))

    adapter = FakeAdapter(tmp_path, episode_id)
    client = ProviderModelClient(
        DirectProviderStream(),
        route=ROUTE,
        model_spec=ModelSpec(provider="deepseek", model_id="deepseek-flash"),
        artifact_root=tmp_path / "artifacts",
        prompt_cache_key="probe",
    )
    runner = EpisodeRunner(
        build_spec(episode_id, caps={"provider_usd_micros": 838}),
        build_config(tmp_path, max_steps=20),
        selector=selector(tmp_path),
        model=client,
        launch=lambda _: adapter,
        rescue=_rescue_ok,
    )

    outcome = await runner.run()

    root = outcome.bundle_root
    assert root is not None
    assert verify_bundle(root).valid
    # The priced spend is in the journal, not a zero: this is the figure the
    # cap is enforced on.
    costs = payloads(root, UsageCostPayload)
    assert [cost.cost_microusd for cost in costs] == [838]
    assert payloads(root, ReconciliationPayload)[0].provider_cost_microusd == 838
    steps = payloads(root, EnvironmentStepPayload)
    assert len(steps) == 1
    assert steps[-1].truncated is True
    assert steps[-1].truncation_reason == "budget-cap"


@pytest.mark.asyncio
async def test_budget_cap_truncates_not_cancels(tmp_path: Path, episode_id: str) -> None:
    """A reached provider-cost cap is enforced as a scored truncation, where
    before it was only reported as an overrun after the fact."""

    # ScriptedModel bills 7 micro-USD per cycle: two cycles reach the cap. The
    # step budget is sized so the CAP is what binds -- a cost guard that
    # prorates a tiny 14-micro-USD allowance over ten steps would stop the
    # episode as over-pace on its first cycle, which is a different test.
    adapter = FakeAdapter(tmp_path, episode_id)
    runner = EpisodeRunner(
        build_spec(episode_id, caps={"provider_usd_micros": 14}),
        build_config(tmp_path, max_steps=4),
        selector=selector(tmp_path),
        model=ScriptedModel(["step"] * 8),
        launch=lambda _: adapter,
        rescue=_rescue_ok,
    )

    outcome = await runner.run()

    assert outcome.status == "completed"
    assert outcome.score is not None and outcome.score.status == "scored"
    root = outcome.bundle_root
    assert root is not None
    assert verify_bundle(root).valid
    steps = payloads(root, EnvironmentStepPayload)
    assert steps[-1].truncated is True
    assert steps[-1].truncation_reason == "budget-cap"
    assert len(steps) == 2


@pytest.mark.asyncio
async def test_ask_answer_rides_on_the_asking_turn(tmp_path: Path, episode_id: str) -> None:
    adapter = FakeAdapter(tmp_path, episode_id)
    model = ScriptedModel(["ask", "finish"])
    runner = _runner(
        tmp_path, episode_id, adapter=adapter, model=model, responder=RecordingResponder("go left")
    )

    await runner.run()

    second = model.histories[1]
    assert second[0].ask_answer == "go left"
    assert second[0].batch is not None
    assert second[1].ask_answer is None


def test_episode_config_declares_no_frame_policy_it_cannot_enforce(tmp_path: Path) -> None:
    """The frame policy lives on the model client, which is built before the
    runner. A ``keep_recent_frames`` on the config was a knob a caller could
    set to 5 and get 3 (review round 1, m1); the config refuses it outright."""

    from dataclasses import fields

    from local_operator.evaluation.runner.episode import EpisodeConfig

    assert "keep_recent_frames" not in {field.name for field in fields(EpisodeConfig)}
    with pytest.raises(TypeError):
        build_config(tmp_path, keep_recent_frames=5)


@pytest.mark.asyncio
async def test_context_unrecoverable_seals_as_a_harness_error_not_a_provider_one(
    tmp_path: Path, episode_id: str
) -> None:
    """When the client cannot fit the context into the window even after
    shedding every stale observation, it refuses rather than send a request
    the provider will reject. The runner records that as a harness (adapter)
    error and seals UNSCORED -- honestly, with a valid bundle -- because a
    scored truncation is not representable: the last executed step's event was
    already written, and the verifier's one-step-per-batch rule forbids a
    corrective re-write (checked against the real verifier; see the PR thread).
    It must not be reclassified as a provider failure, which would blame an
    outage that did not happen."""

    from local_operator.evaluation.runner.provider_client import (
        ContextUnrecoverableError,
    )

    class RefusingModel:
        async def decide(self, observation: Any, history: Any, **kwargs: Any) -> Any:
            raise ContextUnrecoverableError("context cannot fit the window")

    adapter = FakeAdapter(tmp_path, episode_id)
    runner = EpisodeRunner(
        build_spec(episode_id),
        build_config(tmp_path),
        selector=selector(tmp_path),
        model=RefusingModel(),
        launch=lambda _: adapter,
        rescue=_rescue_ok,
    )

    outcome = await runner.run()

    assert outcome.status == "failed"
    assert outcome.score is not None and outcome.score.status == "unscored"
    root = outcome.bundle_root
    assert root is not None
    report = verify_bundle(root)
    assert report.valid, [issue.code for issue in report.issues]
    errors = payloads(root, ErrorPayload)
    assert len(errors) == 1
    assert errors[0].category == "adapter"
    assert errors[0].diagnostic_code == "contextunrecoverableerror"


@pytest.mark.asyncio
async def test_an_aborted_stream_seals_as_provider_but_still_bills_its_input(
    tmp_path: Path, episode_id: str
) -> None:
    """A refusal is unanswered, not unbilled, and the bundle must say so.

    The provider read the whole prompt before declining -- the observed case
    carried 48,000 input tokens -- so the attempt is a real cost. Without its
    triple the episode sealed with NO ``usage_cost`` event, and ``_reconcile``
    then published the resulting zero as an AVAILABLE (measured) figure: a
    bundle whose only purpose is honest accounting claiming a paid turn cost
    nothing. The episode still ends as a provider failure; what changes is that
    the spend it caused is in the evidence.
    """

    from local_operator.evaluation.runner.provider_client import (
        ProviderStreamAbortedError,
    )

    class RefusingModel:
        async def decide(self, observation: Any, history: Any, **kwargs: Any) -> Any:
            raise ProviderStreamAbortedError(
                "model refused: I can't help with that (stop_reason=refusal)",
                route=ROUTE,
                usage=ModelUsage(input_tokens=48_000, output_tokens=0),
                cost_micros=144_000,
                stop_reason="refusal",
                provider_request_id="req-refused-1",
            )

    adapter = FakeAdapter(tmp_path, episode_id)
    runner = EpisodeRunner(
        build_spec(episode_id),
        build_config(tmp_path),
        selector=selector(tmp_path),
        model=RefusingModel(),
        launch=lambda _: adapter,
        rescue=_rescue_ok,
    )

    outcome = await runner.run()

    assert outcome.status == "failed"
    assert outcome.score is not None and outcome.score.reason == "infrastructure_failure"
    root = outcome.bundle_root
    assert root is not None
    report = verify_bundle(root)
    assert report.valid, [issue.code for issue in report.issues]
    # The billed attempt is a full triple, not a hole: the counters are the
    # verifier's own sum over the journal's usage events.
    assert report.counters is not None
    assert report.counters.model_request_count == 1
    assert report.counters.model_response_count == 1
    assert report.counters.input_tokens == 48_000
    assert report.counters.cost_microusd == 144_000
    # The provider's real marker is recorded rather than laundered into "stop".
    responses = payloads(root, ModelResponsePayload)
    assert [response.stop_reason for response in responses] == ["refusal"]
    assert responses[0].provider_request_id == "req-refused-1"
    # ``_reconcile`` reports that same spend, so the bundle's budget record and
    # its journal agree instead of the record claiming a measured zero.
    reconciliations = payloads(root, ReconciliationPayload)
    assert reconciliations[0].provider_cost_microusd == 144_000
    # Exactly one error event, the terminal provider one: writing the triple
    # must not also mint a retryable model error the way a rejection does.
    errors = payloads(root, ErrorPayload)
    assert [(e.category, e.retryable) for e in errors] == [("provider", False)]
    # The provider's own words survive into the bundle. They matter more here
    # than anywhere else: ``diagnostic_code`` is derived from the wrapping
    # ``_ProviderFailure``, so it buckets this identically to a transport
    # outage, and the detail is the only thing naming which mechanism fired.
    detail = errors[0].detail_artifact
    assert detail is not None
    recorded = (root / "artifacts" / detail.sha256).read_text()
    assert "stop_reason=refusal" in recorded
    assert "I can't help with that" in recorded


# ---------------------------------------------------------------------------
# Rejected decisions: one bad reply is re-prompted, not fatal; the retry is
# in the evidence; the bound ends the episode as a MODEL failure.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_one_rejected_decision_is_re_prompted_and_the_corrected_batch_proceeds(
    tmp_path: Path, episode_id: str
) -> None:
    """Bundle ep-6ea01a117eee ended on its first decision (``frame_id '1'``
    against ``screen``) as a provider failure with $0 spent. A rejected reply
    is now recorded as a billed attempt with a retryable ``model`` error, the
    model is asked again for the SAME observation, and the corrected batch
    runs the episode to a scored, reportable seal."""

    adapter = FakeAdapter(tmp_path, episode_id)
    model = ScriptedModel(["reject", "step", "finish"])
    runner = _runner(tmp_path, episode_id, adapter=adapter, model=model)

    outcome = await runner.run()

    assert outcome.status == "completed", outcome.diagnostic
    assert outcome.score is not None and outcome.score.status == "scored"
    assert outcome.reportability_label == "reportable"
    root = outcome.bundle_root
    assert root is not None
    report = verify_bundle(root)
    assert report.valid, [issue.code for issue in report.issues]
    # Three billed calls, three triples: the rejected attempt is not hidden.
    assert report.counters is not None
    assert report.counters.model_request_count == 3
    assert report.counters.model_response_count == 3
    assert model.calls == 3
    # The re-prompt was for the same observation: identical history both times.
    assert model.histories[0] == model.histories[1]
    errors = payloads(root, ErrorPayload)
    assert [(e.category, e.diagnostic_code, e.retryable) for e in errors] == [
        ("model", "decision-rejected", True)
    ]
    assert errors[0].detail_artifact is not None
    # Journal order: the rejected attempt's triple, THEN its error, then the
    # corrected attempt's triple.
    kinds = _kinds(root)
    first_request = kinds.index("model_request")
    assert kinds[first_request : first_request + 7] == [
        "model_request",
        "model_response",
        "usage_cost",
        "error",
        "model_request",
        "model_response",
        "usage_cost",
    ]


@pytest.mark.asyncio
async def test_a_rejected_decision_publishes_the_reply_beside_the_diagnostic(
    tmp_path: Path, episode_id: str
) -> None:
    """A diagnostic alone cannot answer "what was the model attempting?".

    Bundle ep-ffda3fc88f81 recorded three ``decision-rejected`` errors whose
    artifacts held only the Pydantic complaint. Whether those were failed
    keyboard actions -- the difference between "the model never tried to type"
    and "the model tried and the schema refused it" -- was not answerable from
    the bundle, and answering it would have cost another paid run.
    """

    adapter = FakeAdapter(tmp_path, episode_id)
    model = ScriptedModel(["reject", "finish"])
    runner = _runner(tmp_path, episode_id, adapter=adapter, model=model)

    outcome = await runner.run()

    root = outcome.bundle_root
    assert root is not None
    errors = payloads(root, ErrorPayload)
    assert len(errors) == 1
    detail = errors[0].detail_artifact
    assert detail is not None
    recorded = (root / "artifacts" / detail.sha256).read_text()
    # Both halves, and labelled: a reader can tell the harness's refusal from
    # the model's own words.
    assert "unknown frame_id '1'" in recorded
    assert "--- rejected reply ---" in recorded
    assert '{"actions": [{"kind": "click", "frame_id": "1"}]}' in recorded


@pytest.mark.asyncio
async def test_a_rejection_without_a_captured_reply_records_the_diagnostic_alone(
    tmp_path: Path, episode_id: str
) -> None:
    """``reply`` is optional on the protocol boundary, so a client that does not
    capture it must degrade to the diagnostic rather than emit an empty section
    that reads as "the model said nothing"."""

    from local_operator.evaluation.runner.episode import _rejection_detail
    from local_operator.evaluation.runner.model import DecisionRejected

    # ``None`` is the explicit "in-process rendering, never evidence" case the
    # signature requires a caller to state (``_diagnostic``'s rule).
    assert _rejection_detail(DecisionRejected("refused"), None) == "refused"
    assert _rejection_detail(DecisionRejected("refused", reply=""), None) == "refused"
    detail = _rejection_detail(DecisionRejected("refused", reply="{}"), None)
    assert "--- rejected reply ---" in detail


@pytest.mark.asyncio
async def test_exhausted_decision_retries_seal_as_a_model_failure(
    tmp_path: Path, episode_id: str
) -> None:
    """A streak the model cannot convert into a batch ends at its ceiling.

    The bound is flat: ``max(max_decision_retries + 2, max_rejection_streak)``
    = four refusals by default, whichever classes the streak lands on, i.e.
    four billed calls, one more than the flat three this rule replaced. That
    is the agent's failure, so the bundle says ``model`` / ``model_failure``
    -- not ``provider`` (nothing was down) and not ``crash`` (nothing broke)
    -- and every billed attempt is in the evidence."""

    adapter = FakeAdapter(tmp_path, episode_id)
    model = ScriptedModel(["reject", "reject", "reject", "reject", "step", "finish"])
    runner = EpisodeRunner(
        build_spec(episode_id),
        build_config(tmp_path, max_decision_retries=2),
        selector=selector(tmp_path),
        model=model,
        launch=lambda _: adapter,
        rescue=_rescue_ok,
    )

    outcome = await runner.run()

    assert outcome.status == "failed"
    assert outcome.rescue_required is False
    assert outcome.score is not None
    assert outcome.score.status == "unscored"
    assert outcome.score.reason == "model_failure"
    assert outcome.reportability_label == "unscored"
    assert outcome.diagnostic is not None and "4 attempt(s)" in outcome.diagnostic
    # Exactly the flat ceiling: no fifth call.
    assert model.calls == 4
    root = outcome.bundle_root
    assert root is not None
    report = verify_bundle(root)
    assert report.valid, [issue.code for issue in report.issues]
    assert report.counters is not None
    assert report.counters.model_request_count == 4
    errors = payloads(root, ErrorPayload)
    assert [(e.category, e.retryable) for e in errors] == [
        ("model", True),
        ("model", True),
        ("model", True),
        ("model", True),
        ("model", False),
    ]
    assert errors[-1].diagnostic_code == "modelfailure"
    # Cleanup ran on the live session: the environment was never at fault.
    assert "cleanup" in adapter.calls


class _ClassedRejectingModel(ScriptedModel):
    """``ScriptedModel`` whose refusals state a class, cycling a given tuple.

    These tests exist to prove the bound is FLAT: a streak that keeps changing
    class must end at the same refusal a repeating one does. The class is
    still real client data on the rejection (the evidence record quotes it),
    and the cycling never repeats a class back-to-back -- the maximum variety
    a streak can present, so a bound that variation could bend would show here.
    """

    def __init__(self, script: list[str], classes: list[str]) -> None:
        super().__init__(script)
        self.classes = list(classes)
        #: Rejections so far, tracked separately from ``calls``: the class
        #: cycle is indexed by REJECTION ordinal, so a script that ever
        #: interleaves accepted calls with refusals cannot mis-cycle it.
        self.rejections = 0

    async def decide(self, observation: Any, history: Any, **kwargs: Any) -> Any:
        from local_operator.evaluation.runner.model import DecisionRejected

        kind = self.script[self.calls] if self.calls < len(self.script) else "finish"
        if kind == "reject":
            class_key = self.classes[self.rejections % len(self.classes)]
            self.rejections += 1
            self.calls += 1
            raise DecisionRejected(
                f"Your previous reply was rejected: refused as {class_key}",
                class_key=class_key,
                route=self.route,
                usage=ModelUsage(input_tokens=10, output_tokens=5),
                cost_micros=self._bill(),
                provider_request_id=f"rejected-{self.calls}",
            )
        return await super().decide(observation, history, **kwargs)


@pytest.mark.asyncio
async def test_a_repeating_streak_survives_the_old_flat_bound(
    tmp_path: Path, episode_id: str
) -> None:
    """THE survival case: three consecutive refusals no longer end the run.

    Both measured deaths converged on this shape: three consecutive refusals
    of one observation, sealed ``model_failure`` at the old flat bound of
    three (task_005's three were one repeated class; task_016's were three
    different ones). The flat ceiling is now four, so a run whose fourth
    reply is valid continues -- for EITHER shape, which is what makes the
    streak survivable rather than merely differently-counted.
    """

    adapter = FakeAdapter(tmp_path, episode_id)
    model = ScriptedModel(["reject", "reject", "reject", "step", "finish"])
    runner = EpisodeRunner(
        build_spec(episode_id),
        build_config(tmp_path, max_decision_retries=2),
        selector=selector(tmp_path),
        model=model,
        launch=lambda _: adapter,
        rescue=_rescue_ok,
    )

    outcome = await runner.run()

    assert outcome.status == "completed", outcome.diagnostic
    # Three refusals, the corrected step, then the finish: the fourth call is
    # the last one the old bound would have denied.
    assert model.calls == 5


@pytest.mark.asyncio
async def test_a_changing_streak_survives_the_flat_ceiling_too(
    tmp_path: Path, episode_id: str
) -> None:
    """The bound is FLAT: a changing streak gets the repeating streak's room.

    task_016's killing streak was three DIFFERENT classes (extra-action-key,
    field-invalid, observation-binding), each reply one repair away from
    valid; the old flat bound read that as "not converging". A split revision
    briefly gave this shape a roomier arm, and review measured the hole in
    that idea -- a two-class oscillation never repeats a class back-to-back
    and rode the roomier arm's full bound -- so the flat ceiling gives this
    shape exactly what it gives a repeating one: three refusals survive and
    the fourth call runs the corrected reply.
    """

    adapter = FakeAdapter(tmp_path, episode_id)
    model = _ClassedRejectingModel(
        ["reject"] * 3 + ["step", "finish"],
        ["extra-action-key", "field-invalid", "observation-binding", "duplicate-key"],
    )
    runner = EpisodeRunner(
        build_spec(episode_id),
        build_config(tmp_path, max_decision_retries=2),
        selector=selector(tmp_path),
        model=model,
        launch=lambda _: adapter,
        rescue=_rescue_ok,
    )

    outcome = await runner.run()

    assert outcome.status == "completed", outcome.diagnostic
    # Three refusals, the corrected step, then the finish.
    assert model.calls == 5


@pytest.mark.asyncio
async def test_a_changing_streak_ends_at_the_flat_ceiling(tmp_path: Path, episode_id: str) -> None:
    """Variation is not a licence: the fourth refusal ends it, changing or not.

    A model that keeps producing a different defect every call is oscillating,
    not converging, so the streak dies at the same fourth refusal a repeating
    one does (the removed split let this streak run to six).
    """

    adapter = FakeAdapter(tmp_path, episode_id)
    model = _ClassedRejectingModel(
        ["reject"] * 6 + ["finish"],
        ["extra-action-key", "field-invalid", "observation-binding", "duplicate-key"],
    )
    runner = EpisodeRunner(
        build_spec(episode_id),
        build_config(tmp_path, max_decision_retries=2),
        selector=selector(tmp_path),
        model=model,
        launch=lambda _: adapter,
        rescue=_rescue_ok,
    )

    outcome = await runner.run()

    assert outcome.status == "failed"
    assert outcome.score is not None and outcome.score.reason == "model_failure"
    assert outcome.diagnostic is not None and "4 attempt(s)" in outcome.diagnostic
    assert model.calls == 4


#: The reasoning ladder the canary's route publishes, and the rung it ran at.
_CANARY_LADDER = ("none", "low", "high", "max")


class _TruncatingModel(ScriptedModel):
    """Plays the canary's output-limit truncations, silent or not.

    The two tokens differ ONLY in whether the reply carried anything: both stop
    at ``length``, both are billed, both report ``tool_call_count=0``. That is
    the pair the retreat's boundary has to separate -- ``empty-length`` is the
    reasoning model that spent its whole budget thinking (``task_002``, the
    third consecutive one sealing the episode), ``truncated-length`` is the
    same call whose reply was cut mid-object and has something to correct.

    ``retreat_effort`` is implemented the way the provider client does it:
    one rung down the model's own ladder, refusing at the bottom. ``retreats``
    records the rungs, so an assertion can tell "the episode recovered" from
    "the episode recovered BY retreating".
    """

    def __init__(self, script: Any, *, effort: str = "max") -> None:
        super().__init__(script)
        self.effort = effort
        self.retreats: list[str] = []

    def retreat_effort(self) -> str | None:
        index = _CANARY_LADDER.index(self.effort)
        if index == 0:
            return None
        self.effort = _CANARY_LADDER[index - 1]
        self.retreats.append(self.effort)
        return self.effort

    async def decide(self, observation: Any, history: Any, **kwargs: Any) -> Any:
        from local_operator.evaluation.runner.model import DecisionRejected

        kind = self.script[self.calls] if self.calls < len(self.script) else "finish"
        if kind in ("empty-length", "truncated-length"):
            silent = kind == "empty-length"
            self.histories.append(tuple(history))
            self.calls += 1
            raise DecisionRejected(
                "Your previous reply was rejected: "
                + (
                    "reply carried no tool call and no text: the model ended its turn as "
                    "'length' without emitting a decision on either channel."
                    if silent
                    else "the reply was not one complete JSON object: the object itself was "
                    "incomplete (cut off or double-escaped)."
                ),
                class_key="empty-reply" if silent else "incomplete-json",
                # The reply the model produced, which is what the bundereader sees.
                reply="" if silent else '{"actions": [{"kind": "click", "frame_id": "1"}]}',
                stop_reason="length",
                # The client's own classification, computed from the stream it
                # saw. The runner reads this and nothing else.
                empty_length_truncation=silent,
                reasoning_effort=self.effort,
                route=self.route,
                usage=ModelUsage(
                    input_tokens=36_221, output_tokens=16_384, reasoning_tokens=16_384
                ),
                cost_micros=self._bill(),
                provider_request_id=f"attempt-{self.calls}",
            )
        decision = await super().decide(observation, history, **kwargs)
        return decision.model_copy(update={"reasoning_effort": self.effort})


@pytest.mark.asyncio
async def test_empty_output_limit_truncations_retreat_the_effort_instead_of_sealing_the_episode(
    tmp_path: Path, episode_id: str
) -> None:
    """The canary's zero-scoring failure shape must be recoverable.

    Ground truth, from ``batch-deepseek-flash-canary8``: three episodes each
    ended on three consecutive replies of ``output_tokens=16384
    reasoning_tokens=16384 stop_reason=length tool_call_count=0`` -- the model's
    ENTIRE output budget spent thinking, nothing emitted -- and were sealed
    ``model_failure`` after 94/82/55 steps. The ordinary harness has a recovery
    for exactly this (``harness/loop.py``: retry one effort rung lower, bounded
    by ``MAX_EMPTY_TRUNCATION_RETRIES``); this arm re-prompted at the SAME
    effort until the corrective bound ran out, which is why the same route
    failed here and recovered in a session.

    Three silent truncations and then a finish: the first two are retreated
    (the third is where the retreat allowance is spent, so it counts as an
    ordinary rejection and the corrective bound still has room), and the
    episode completes SCORED. Every attempt keeps its own triple, its stop
    reason and the rung it was asked at.
    """

    from local_operator.evaluation.evidence.models import ModelRequestPayload

    adapter = FakeAdapter(tmp_path, episode_id)
    model = _TruncatingModel(["empty-length", "empty-length", "empty-length", "finish"])
    runner = _runner(tmp_path, episode_id, adapter=adapter, model=model)

    outcome = await runner.run()

    assert outcome.status == "completed", outcome.diagnostic
    assert outcome.score is not None and outcome.score.status == "scored"
    assert outcome.reportability_label == "reportable"
    # Two retreats, then the allowance is spent -- not three, and not none.
    assert model.retreats == ["high", "low"]
    assert model.calls == 4
    root = outcome.bundle_root
    assert root is not None
    report = verify_bundle(root)
    assert report.valid, [issue.code for issue in report.issues]
    # Four billed attempts, four triples: the retries are not hidden, and the
    # episode's spend is the sum of what it actually cost.
    assert report.counters is not None
    assert report.counters.model_request_count == 4
    assert report.counters.model_response_count == 4
    # Every attempt is legible as what it was, in the evidence an analyst reads.
    requests = payloads(root, ModelRequestPayload)
    assert [request.reasoning_effort for request in requests] == ["max", "high", "low", "low"]
    responses = payloads(root, ModelResponsePayload)
    assert [response.stop_reason for response in responses] == ["length"] * 3 + ["stop"]
    # Silent on BOTH channels: the marker alone does not say that, and it is the
    # property the retreat is keyed on.
    assert [response.tool_call_count for response in responses[:3]] == [0, 0, 0]
    assert [response.reasoning_tokens for response in responses[:3]] == [16_384] * 3
    errors = payloads(root, ErrorPayload)
    assert [(e.category, e.diagnostic_code, e.retryable) for e in errors] == [
        ("model", "decision-rejected", True)
    ] * 3


@pytest.mark.asyncio
async def test_the_retreat_allowance_is_spent_before_the_corrective_bound_sees_it(
    tmp_path: Path, episode_id: str
) -> None:
    """A model that never answers is still bounded, and never unbounded.

    The retreat must not become a way to buy unbounded retries. Every attempt
    here is a silent output-limit truncation, so the two retreats are spent
    first -- ``max``, then ``high``, then ``low`` -- and from there the empty
    truncations are ORDINARY refusals, retried at the ceiling effort until
    the flat ceiling (four refusals) seals the episode as ``model_failure``.
    So the worst case grows by the retreat allowance plus the one added
    corrective prompt: 6 billed attempts on an episode that was always going
    to fail -- and the verdict says so, because ``attempts`` counts the
    billed calls rather than only the corrective ones that exhausted the
    bound.
    """

    adapter = FakeAdapter(tmp_path, episode_id)
    model = _TruncatingModel(["empty-length"] * 8)
    runner = EpisodeRunner(
        build_spec(episode_id),
        build_config(tmp_path, max_decision_retries=2),
        selector=selector(tmp_path),
        model=model,
        launch=lambda _: adapter,
        rescue=_rescue_ok,
    )

    outcome = await runner.run()

    assert model.retreats == ["high", "low"]
    assert model.calls == 6
    assert outcome.status == "failed"
    assert outcome.score is not None
    assert outcome.score.status == "unscored"
    assert outcome.score.reason == "model_failure"
    assert outcome.diagnostic is not None and "6 attempt(s)" in outcome.diagnostic
    root = outcome.bundle_root
    assert root is not None
    report = verify_bundle(root)
    assert report.valid, [issue.code for issue in report.issues]
    assert report.counters is not None and report.counters.model_request_count == 6


@pytest.mark.asyncio
async def test_a_truncation_that_carried_text_keeps_the_ordinary_rejection_path(
    tmp_path: Path, episode_id: str
) -> None:
    """The NEGATIVE CONTROL, and the line the retreat may not cross.

    Same stop reason, same billing, same zero tool calls -- the reply simply
    carried text (a JSON batch cut mid-object, which is what ``task_002`` did
    two calls before it died). That reply is TRUNCATED, not silent: there is
    something to correct, so it takes the corrective re-prompt at the same
    effort, and at the flat ceiling the episode seals ``model_failure`` --
    four refusals under the current bound. Retreating here would spend the
    effort allowance on a defect the correction can fix, and would make a
    truncation indistinguishable from a model that cannot answer.
    """

    adapter = FakeAdapter(tmp_path, episode_id)
    model = _TruncatingModel(["truncated-length"] * 4 + ["finish"])
    runner = EpisodeRunner(
        build_spec(episode_id),
        build_config(tmp_path, max_decision_retries=2),
        selector=selector(tmp_path),
        model=model,
        launch=lambda _: adapter,
        rescue=_rescue_ok,
    )

    outcome = await runner.run()

    assert model.retreats == []
    assert model.calls == 4
    assert outcome.status == "failed"
    assert outcome.score is not None
    assert outcome.score.status == "unscored"
    assert outcome.score.reason == "model_failure"
    assert outcome.diagnostic is not None and "4 attempt(s)" in outcome.diagnostic
    root = outcome.bundle_root
    assert root is not None
    report = verify_bundle(root)
    assert report.valid, [issue.code for issue in report.issues]
    assert report.counters is not None and report.counters.model_request_count == 4


@pytest.mark.asyncio
async def test_a_client_that_cannot_retreat_degrades_to_the_corrective_path(
    tmp_path: Path, episode_id: str
) -> None:
    """The retreat is an OPTIONAL client capability, like ``model_reply_metadata``.

    A scripted or historic client has no effort to lower, and the episode must
    then behave as it does for any repeated defect: the silent truncations are
    ordinary refusals, bounded by the same flat ceiling (here four of them).
    The capability check is what keeps the runner usable against a client that
    models no effort at all, and this is the case that proves the runner does
    not assume it.
    """

    class _SilentScriptedModel(ScriptedModel):
        """``ScriptedModel`` plus the flag, with NO ``retreat_effort``."""

        async def decide(self, observation: Any, history: Any, **kwargs: Any) -> Any:
            from local_operator.evaluation.runner.model import DecisionRejected

            if self.calls < len(self.script) and self.script[self.calls] == "empty-length":
                self.histories.append(tuple(history))
                self.calls += 1
                raise DecisionRejected(
                    "reply carried no tool call and no text",
                    reply="",
                    class_key="empty-reply",
                    stop_reason="length",
                    empty_length_truncation=True,
                    route=self.route,
                    usage=ModelUsage(input_tokens=36_221, output_tokens=16_384),
                    cost_micros=self._bill(),
                )
            return await super().decide(observation, history, **kwargs)

    adapter = FakeAdapter(tmp_path, episode_id)
    model = _SilentScriptedModel(["empty-length"] * 4 + ["finish"])
    runner = EpisodeRunner(
        build_spec(episode_id),
        build_config(tmp_path, max_decision_retries=2),
        selector=selector(tmp_path),
        model=model,
        launch=lambda _: adapter,
        rescue=_rescue_ok,
    )

    outcome = await runner.run()

    assert model.calls == 4
    assert outcome.score is not None and outcome.score.reason == "model_failure"


@pytest.mark.asyncio
async def test_decision_exhaustion_after_a_step_scores_the_reached_state(
    tmp_path: Path, episode_id: str
) -> None:
    """Exhaustion between steps grades the work already bought.

    The step-0 exhaustion above stays unscored (no environment work to
    grade); after a step has landed, the same exhaustion records an
    ``agent_stop`` and scores the state reached. The terminal keeps
    ``failure_kind="model"`` for audit -- so the episode reports failed --
    and the verifier's laundering guard requires that kind to agree with the
    stop reason. No action batch is written for the stopped observation.
    """

    adapter = FakeAdapter(tmp_path, episode_id)
    model = ScriptedModel(["type", "reject", "reject", "reject", "reject"])
    runner = EpisodeRunner(
        build_spec(episode_id),
        build_config(tmp_path, max_decision_retries=2),
        selector=selector(tmp_path),
        model=model,
        launch=lambda _: adapter,
        rescue=_rescue_ok,
    )

    outcome = await runner.run()

    assert outcome.status == "failed"
    assert outcome.reportability_label == "reportable"
    assert outcome.score is not None and outcome.score.status == "scored"
    root = outcome.bundle_root
    assert root is not None
    report = verify_bundle(root)
    assert report.valid, [issue.code for issue in report.issues]
    kinds = _kinds(root)
    assert kinds.index("agent_stop") < kinds.index("finalization_start")
    stops = payloads(root, AgentStopPayload)
    assert len(stops) == 1
    assert stops[0].reason == "model_failure"
    assert stops[0].attempts == 4
    assert stops[0].observation_id == payloads(root, ObservationPayload)[-1].observation_id
    assert stops[0].detail_artifact is not None
    # The scored path writes no terminal model error: every rejection keeps
    # its own retryable record and the stop carries the diagnostic.
    errors = payloads(root, ErrorPayload)
    assert [(error.category, error.retryable) for error in errors] == [
        ("model", True),
        ("model", True),
        ("model", True),
        ("model", True),
    ]
    # The last observation never received a batch: one batch per observation,
    # and the stopped one is the verifier's allowed batchless tail.
    observations = payloads(root, ObservationPayload)
    batches = payloads(root, ActionBatchPayload)
    assert {batch.observation_id for batch in batches} == {
        observation.observation_id for observation in observations[:-1]
    }
    assert "score" in adapter.calls


@pytest.mark.asyncio
async def test_zero_decision_retries_restores_one_strike(tmp_path: Path, episode_id: str) -> None:
    adapter = FakeAdapter(tmp_path, episode_id)
    model = ScriptedModel(["reject", "finish"])
    runner = EpisodeRunner(
        build_spec(episode_id),
        build_config(tmp_path, max_decision_retries=0),
        selector=selector(tmp_path),
        model=model,
        launch=lambda _: adapter,
        rescue=_rescue_ok,
    )

    outcome = await runner.run()

    assert outcome.status == "failed"
    assert outcome.score is not None and outcome.score.reason == "model_failure"
    assert model.calls == 1


@pytest.mark.asyncio
async def test_zero_max_rejection_streak_restores_one_strike(
    tmp_path: Path, episode_id: str
) -> None:
    """``0`` on either knob is the one-strike sentinel.

    ``max_decision_retries = 0`` has always meant fail-fast; the flat design
    keeps that and gives ``max_rejection_streak`` the same meaning, so the
    knob that names the ceiling can also decline every corrective call. The
    first refused reply ends the episode with no second call.
    """

    adapter = FakeAdapter(tmp_path, episode_id)
    model = ScriptedModel(["reject", "finish"])
    runner = EpisodeRunner(
        build_spec(episode_id),
        build_config(tmp_path, max_rejection_streak=0),
        selector=selector(tmp_path),
        model=model,
        launch=lambda _: adapter,
        rescue=_rescue_ok,
    )

    outcome = await runner.run()

    assert outcome.status == "failed"
    assert outcome.score is not None and outcome.score.reason == "model_failure"
    assert model.calls == 1


@pytest.mark.asyncio
async def test_crash_after_a_successful_step_still_seals(tmp_path: Path, episode_id: str) -> None:
    """A run interrupted mid-loop keeps the evidence it already bought.

    Regression for ep-ffda3fc88f81: a real paid episode drove 16 environment
    steps, took an unretryable adapter error on the 17th, and lost ALL of it --
    the bundle could neither seal nor be abandoned (``abandonment_failed``,
    score null) because the verifier required a stepped episode to end on a
    terminal step or a finish action, which is precisely what an interrupted
    episode cannot produce. The failure is not adapter-specific: the same shape
    killed a provider outage, so it is asserted here at the runner boundary.
    """

    adapter = FakeAdapter(
        tmp_path,
        episode_id,
        failures={"execute": SupervisionError("worker died")},
        fail_after={"execute": 1},
    )
    runner = _runner(
        tmp_path, episode_id, adapter=adapter, model=ScriptedModel(["step", "step", "finish"])
    )

    outcome = await runner.run()

    assert outcome.status == "failed"
    root = outcome.bundle_root
    assert root is not None
    report = verify_bundle(root)
    assert report.valid, [issue.code for issue in report.issues]
    # The step that DID succeed is still in the bundle: that is the evidence
    # the old behaviour threw away.
    assert len(payloads(root, EnvironmentStepPayload)) == 1
    assert outcome.score is not None and outcome.score.reason == "crash"


@pytest.mark.asyncio
async def test_provider_failure_after_a_step_still_seals(tmp_path: Path, episode_id: str) -> None:
    """The same interruption through a non-adapter category."""

    adapter = FakeAdapter(tmp_path, episode_id)
    model = ScriptedModel(["step", "step", "finish"])
    original = model.decide
    calls = {"n": 0}

    async def decide(observation: Any, history: Any, **kwargs: Any) -> Any:
        calls["n"] += 1
        if calls["n"] > 1:
            raise RuntimeError("provider exhausted retries")
        return await original(observation, history, **kwargs)

    model.decide = decide  # type: ignore[method-assign]
    runner = _runner(tmp_path, episode_id, adapter=adapter, model=model)

    outcome = await runner.run()

    assert outcome.status == "failed"
    root = outcome.bundle_root
    assert root is not None
    report = verify_bundle(root)
    assert report.valid, [issue.code for issue in report.issues]
    assert outcome.score is not None and outcome.score.reason == "infrastructure_failure"


@pytest.mark.asyncio
async def test_fatal_error_records_a_bounded_diagnostic_detail(
    tmp_path: Path, episode_id: str
) -> None:
    """An unretryable failure must be diagnosable without a paid rerun.

    ``diagnostic_code`` is derived from the exception CLASS, so the real
    episode recorded only "rpcremoteerror" with a null ``detail_artifact`` --
    enough to bucket the failure, never enough to explain it.
    """

    adapter = FakeAdapter(
        tmp_path,
        episode_id,
        failures={"execute": SupervisionError("worker died mid-observe")},
        fail_after={"execute": 1},
    )
    runner = _runner(
        tmp_path, episode_id, adapter=adapter, model=ScriptedModel(["step", "step", "finish"])
    )

    outcome = await runner.run()

    root = outcome.bundle_root
    assert root is not None
    assert verify_bundle(root).valid
    fatal = [error for error in payloads(root, ErrorPayload) if not error.retryable]
    assert len(fatal) == 1
    assert fatal[0].diagnostic_code == "supervisionerror"
    detail = fatal[0].detail_artifact
    assert detail is not None
    text = (root / "artifacts" / detail.sha256).read_bytes().decode()
    assert text == "SupervisionError: worker died mid-observe"
    # Bounded: ``_diagnostic`` truncates, so a pathological message cannot
    # inflate the bundle.
    assert detail.byte_count <= 500


@pytest.mark.asyncio
async def test_fatal_diagnostic_detail_cannot_carry_a_resolved_secret(
    tmp_path: Path, episode_id: str
) -> None:
    """A secret in the failure message is withheld, and the bundle keeps a detail.

    The secret must never reach disk -- that assertion is unchanged and is the
    point of the test. What changed is the SHAPE of the refusal: the artifact
    used to be dropped entirely (``detail_artifact is None``), because the
    whole rendering was handed to ``publish_artifact`` and the canary scan
    rejected the write. A reader then got a fatal error with no detail at all,
    which is the exact opacity this PR exists to remove -- the secret was
    protected by destroying the diagnosis along with it.

    Now the value is withheld at the point it is rendered, so the artifact is
    published carrying the exception TYPE with the message replaced. The type
    cannot hold adapter or provider data and is what makes the failure
    bucketable, so keeping it costs nothing and is the whole difference between
    a diagnosable bundle and a blank one.
    """

    secret = "s3cret-value-that-must-never-land-in-a-bundle"
    adapter = FakeAdapter(
        tmp_path,
        episode_id,
        failures={"execute": SupervisionError(f"worker died using {secret}")},
        fail_after={"execute": 1},
    )
    runner = EpisodeRunner(
        build_spec(episode_id),
        build_config(tmp_path, max_steps=4),
        selector=selector(tmp_path),
        model=ScriptedModel(["step", "step", "finish"]),
        redactions=RedactionSet.from_resolved_values((secret,)),
        launch=lambda _: adapter,
        rescue=_rescue_ok,
    )

    outcome = await runner.run()

    root = outcome.bundle_root
    assert root is not None
    # The leak must not reach disk anywhere in the bundle.
    for path in root.rglob("*"):
        if path.is_file():
            assert secret.encode() not in path.read_bytes(), path
    # And refusing the detail must not cost the bundle its terminal.
    report = verify_bundle(root)
    assert report.valid, [issue.code for issue in report.issues]
    fatal = [error for error in payloads(root, ErrorPayload) if not error.retryable]
    assert len(fatal) == 1
    # Published, not dropped: the reader still learns WHAT failed.
    reference = fatal[0].detail_artifact
    assert reference is not None
    published = (root / "artifacts" / reference.sha256).read_text()
    assert published == "SupervisionError: <withheld: matched a secret canary>"


@pytest.mark.asyncio
async def test_fatal_detail_withholds_a_secret_straddling_the_diagnostic_bound(
    tmp_path: Path, episode_id: str
) -> None:
    """A secret cut by ``_diagnostic``'s 500-char bound must still be withheld.

    Round 1 fixed this ordering in the WORKER; the parent had the identical
    inversion. ``_diagnostic`` truncated to 500 characters and only then were
    the bytes handed to ``publish_artifact``, whose scan is a SUBSTRING check --
    so a severed canary stops matching and the surviving prefix is sealed into
    the bundle. Measured before the fix: 20 characters of a canonical AWS key,
    and 63 of an API token echoed in a provider error URL.

    The sibling test above uses a SHORT secret that fits inside the bound
    entirely, which is exactly why it passed while this leaked: the position of
    the secret relative to the cut is the whole property under test, so the
    padding here is load-bearing rather than decorative.
    """

    secret = "AKIAIOSFODNN7EXAMPLE" + "QWERTYUIOPASDFGH1234"
    # Pad so the 500-character cut lands INSIDE the credential. The rendering
    # is ``"SupervisionError: " + message`` (18 chars) plus this prefix (16),
    # so 446 filler characters leave exactly 20 characters of the key inside
    # the bound -- the arithmetic is the test, and an off-by-a-few pushes the
    # whole secret past the cut where nothing leaks even when the ordering is
    # wrong.
    message = "adapter failed: " + ("x" * 446) + secret + " (will not retry)"
    adapter = FakeAdapter(
        tmp_path,
        episode_id,
        failures={"execute": SupervisionError(message)},
        fail_after={"execute": 1},
    )
    runner = EpisodeRunner(
        build_spec(episode_id),
        build_config(tmp_path, max_steps=4),
        selector=selector(tmp_path),
        model=ScriptedModel(["step", "step", "finish"]),
        redactions=RedactionSet.from_resolved_values((secret,)),
        launch=lambda _: adapter,
        rescue=_rescue_ok,
    )

    outcome = await runner.run()

    root = outcome.bundle_root
    assert root is not None
    # No PREFIX of the credential reaches disk either -- a whole-value check
    # passes against the very truncation this test exists to catch.
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        blob = path.read_bytes()
        for length in range(8, len(secret) + 1):
            assert secret[:length].encode() not in blob, (path, length)
    report = verify_bundle(root)
    assert report.valid, [issue.code for issue in report.issues]


@pytest.mark.asyncio
async def test_provider_failure_withholds_a_secret_straddling_the_bound(
    tmp_path: Path, episode_id: str
) -> None:
    """A PROVIDER error's text is scanned before it is cut, not after.

    Round 2 (R2-1) found the third instance of the same inversion: the provider
    path rendered ``_diagnostic(error)`` with no redaction set on the way into
    ``_ProviderFailure``, and ``_finalize_failure`` later published that already
    truncated message to the sealed bundle. The cut severs the canary, so the
    scan at the publish site -- and ``publish_artifact``'s own scan -- both
    return clean on the surviving prefix.

    This is the case the runner never sees redacted from anywhere else: a
    provider exception is raised on the way to the model, so the adapter worker
    (whose own redaction was fixed in round 1) never touches it.

    The placement is SWEPT rather than fixed, and that is the point. A single
    offset proves nothing here: the message crosses two independent 500-char
    cuts (this site, then ``_failure_detail``), so the surviving fragment is a
    non-monotonic function of where the secret sits. Measured against the bug,
    an offset of 20 leaks only 2 characters -- under any sane prefix threshold
    that reads as CLEAN -- while an offset of 40 leaks 26. Two prior audits and
    the first draft of this test were all defeated by exactly that: a negative
    result at one offset was mistaken for a negative result. The sweep is the
    regression guard; pinning one number would re-arm the trap.
    """

    secret = "AKIAIOSFODNN7EXAMPLE" + "QWERTYUIOPASDFGH1234"
    # 39, and the exact value is evidence rather than taste. The message
    # crosses TWO independent 500-char cuts (this site, then
    # ``_failure_detail``), so the fragment that survives both is non-monotonic
    # in the offset: measured against the bug, offsets 22-39 leak 8-25
    # characters into the sealed artifact, while 40 leaks NOTHING because the
    # downstream scan happens to see enough of the value to match. Picking 40
    # produced a test that passed against the bug it was written for.
    survivors = 39
    # The rendering is ``"RuntimeError: " + message`` (14 characters).
    padding = 500 - 14 - survivors
    adapter = FakeAdapter(tmp_path, episode_id)
    runner = EpisodeRunner(
        build_spec(episode_id),
        build_config(tmp_path, max_steps=4),
        selector=selector(tmp_path),
        model=ScriptedModel(error=RuntimeError("x" * padding + secret + " tail")),
        redactions=RedactionSet.from_resolved_values((secret,)),
        launch=lambda _: adapter,
        rescue=_rescue_ok,
    )

    outcome = await runner.run()

    root = outcome.bundle_root
    assert root is not None
    assert [error.category for error in payloads(root, ErrorPayload)] == ["provider"]
    # No PREFIX reaches disk. Asserting the whole value would pass against the
    # very truncation this test exists to catch.
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        blob = path.read_bytes()
        for length in range(8, len(secret) + 1):
            assert secret[:length].encode() not in blob, (path, length)


@pytest.mark.asyncio
async def test_detail_publish_os_error_cannot_escape_the_failure_handler(
    tmp_path: Path,
    episode_id: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A disk failure while publishing the fatal-error detail must not lose the terminal.

    Regression for the round-1 review (R1-1): ``publish_artifact`` poisons the
    writer and re-raises a RAW ``OSError`` on ambiguous I/O (ENOSPC/EIO), and
    ``_publish`` converts only ``EvidenceError`` -- so before the catch at the
    detail site was widened, an OSError there escaped ``_finalize_failure`` and
    propagated out of ``run()``: the caller got a bare exception, no
    ``EpisodeOutcome``, and a bundle with its steps but no terminal. That is the
    exact "failure handler destroys the bundle's terminal" shape this PR exists
    to remove, gated on a disk error instead of an adapter error.

    The promise is best-effort publication, never best-effort terminal: the
    detail may be lost, but ``run()`` must still return a structured outcome and
    the bundle must stay recoverable (a fresh process can still abandon it).
    """

    detail_bytes = b"SupervisionError: worker died"
    adapter = FakeAdapter(
        tmp_path,
        episode_id,
        failures={"execute": SupervisionError("worker died")},
        fail_after={"execute": 1},
    )
    runner = _runner(
        tmp_path, episode_id, adapter=adapter, model=ScriptedModel(["step", "step", "finish"])
    )

    real_publish = EvidenceWriter.publish_artifact

    def faulty_publish(self: Any, source: Any, *, media_type: str, **kwargs: Any) -> Any:
        if bytes(source) == detail_bytes:
            # The store's real contract on ambiguous I/O: poison, then re-raise
            # the raw OSError (store.py publish_artifact).
            self._poison()  # pyright: ignore[reportPrivateUsage]
            raise OSError(errno.ENOSPC, "No space left on device")
        return real_publish(self, source, media_type=media_type, **kwargs)

    monkeypatch.setattr(EvidenceWriter, "publish_artifact", faulty_publish)

    # Pre-fix this raised OSError out of run(); the fix returns an outcome.
    outcome = await runner.run()

    assert outcome.status == "abandonment_failed"
    root = outcome.bundle_root
    assert root is not None
    # The evidence the episode already bought is still on disk.
    report = verify_bundle(root)
    assert len([e for e in report.events if isinstance(e.payload, EnvironmentStepPayload)]) == 1
    # And the bundle is recoverable: a fresh process can still abandon it, so
    # the terminal is reachable even though this run could not record it.
    recovered = EvidenceWriter.open_for_abandon(root, RedactionSet.from_resolved_values(()))
    try:
        record = recovered.abandon("infrastructure_failure", "evidence-write-failed")
    finally:
        recovered.close()
    assert record.reason == "infrastructure_failure"
    assert verify_bundle(root).terminal_state == "abandoned"


# ---------------------------------------------------------------------------
# Disclosed infra values must not be silently dropped by an older adapter
# ---------------------------------------------------------------------------
#
# The defect these cover is a FALSE disclosure, which is strictly worse than a
# missing one: the caller stamps "this run used m5.xlarge" into the sealed
# manifest from the CLI string alone, while an adapter build that predates the
# knob resolves t3.xlarge and drops it. The bundle then verifies, the run exits
# 0, and a reader reports a burstable-starved score as starvation-corrected.
# ``inspect_requirements`` is the only signal that separates two builds sharing
# one version string, so the runner binds it and fails closed.


def _spec_with_infra(episode_id: str, *names: str) -> Any:
    spec = build_spec(episode_id)
    object.__setattr__(
        spec,
        "infra_values",
        tuple(
            ScopedInfraValue(
                name=name,
                purpose="benchmark_compute",
                value="false" if name == "OSWORLD_ENABLE_PROXY" else "m5.xlarge",
            )
            for name in names
        ),
    )
    return spec


def _infra_runner(
    tmp_path: Path, episode_id: str, spec: Any, adapter: FakeAdapter
) -> EpisodeRunner:
    return EpisodeRunner(
        spec,
        build_config(tmp_path),
        selector=selector(tmp_path),
        model=ScriptedModel(["finish"]),
        launch=lambda _: adapter,
        rescue=_rescue_ok,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["AWS_INSTANCE_TYPE", "OSWORLD_ENABLE_PROXY"])
async def test_a_disclosed_infra_value_an_adapter_does_not_declare_is_refused(
    tmp_path: Path, episode_id: str, name: str
) -> None:
    """The reviewer's reproduction, as a test: old adapter + new host script.

    A ``FakeAdapter`` declaring no requirements is exactly how a build that
    predates ``AWS_INSTANCE_TYPE`` answers for that name. Pre-fix this ran to
    a sealed, valid bundle claiming hardware the episode never used; the
    refusal must land BEFORE ``prepare``, so nothing is allocated and there is
    no bundle to mislead a reader.
    """

    adapter = FakeAdapter(tmp_path, episode_id)
    outcome = await _infra_runner(
        tmp_path, episode_id, _spec_with_infra(episode_id, name), adapter
    ).run()

    assert outcome.status == "failed_pre_bundle"
    assert outcome.bundle_root is None
    assert name in (outcome.diagnostic or "")
    assert "UndeclaredDisclosedInfra" in (outcome.diagnostic or "")
    # The side-effect boundary was never crossed and the worker was reaped.
    assert "prepare" not in adapter.calls and "reset_start" not in adapter.calls
    assert adapter.terminated


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["AWS_INSTANCE_TYPE", "OSWORLD_ENABLE_PROXY"])
async def test_a_declared_disclosed_infra_value_runs_normally(
    tmp_path: Path, episode_id: str, name: str
) -> None:
    """The inverse direction: a rebuilt adapter DOES declare it, so it runs.

    Without this the gate could pass by refusing everything, which would break
    the very workflow the knob exists for.
    """

    adapter = FakeAdapter(
        tmp_path,
        episode_id,
        declared_requirements=(
            Requirement(
                requirement_id=name,
                kind="infra",
                name=name,
                required=False,
            ),
        ),
    )
    outcome = await _infra_runner(
        tmp_path, episode_id, _spec_with_infra(episode_id, name), adapter
    ).run()

    assert outcome.status == "completed", outcome.diagnostic
    assert "prepare" in adapter.calls and "reset_start" in adapter.calls


@pytest.mark.asyncio
async def test_an_undeclared_ordinary_infra_value_is_not_refused(
    tmp_path: Path, episode_id: str
) -> None:
    """The gate is a narrow allowlist, and that scoping is load-bearing.

    ``inspect_requirements`` runs BEFORE the task is named, so it returns only
    the adapter's unconditional baseline -- a task-conditional requirement
    (OSWORLD_PROXY_ENDPOINT for a proxy task) is legitimately absent from it.
    A blanket "refuse every undeclared infra value" rule would therefore break
    correct proxy runs, which is why only values whose silent drop corrupts the
    evidence record are gated.
    """

    adapter = FakeAdapter(tmp_path, episode_id)
    outcome = await _infra_runner(
        tmp_path, episode_id, _spec_with_infra(episode_id, "OSWORLD_PROXY_ENDPOINT"), adapter
    ).run()

    assert outcome.status == "completed", outcome.diagnostic


@pytest.mark.asyncio
async def test_a_root_volume_override_an_adapter_does_not_declare_is_refused(
    tmp_path: Path, episode_id: str
) -> None:
    """The second disclosed value inherits the gate, not just the allowlist.

    Adding a value to the disclosure mechanism is exactly the drift a reviewer
    predicted when the stamp and the allowlist were two hardcoded lists: a
    value stamped into the manifest but ungated is a FALSE disclosure sealed
    in a bundle that verifies. The refusal must land before ``prepare``, so
    nothing is allocated and no bundle exists to mislead a reader.
    """

    adapter = FakeAdapter(tmp_path, episode_id)
    outcome = await _infra_runner(
        tmp_path, episode_id, _spec_with_infra(episode_id, "AWS_ROOT_VOLUME_SIZE"), adapter
    ).run()

    assert outcome.status == "failed_pre_bundle"
    assert outcome.bundle_root is None
    assert "AWS_ROOT_VOLUME_SIZE" in (outcome.diagnostic or "")
    assert "UndeclaredDisclosedInfra" in (outcome.diagnostic or "")
    assert "prepare" not in adapter.calls and "reset_start" not in adapter.calls
    assert adapter.terminated


@pytest.mark.asyncio
async def test_a_declared_root_volume_override_runs_normally(
    tmp_path: Path, episode_id: str
) -> None:
    """The inverse: a rebuilt adapter declares it, so the knob is usable.

    Without this the gate could pass by refusing everything, which would break
    the very workflow the override exists for.
    """

    adapter = FakeAdapter(
        tmp_path,
        episode_id,
        declared_requirements=(
            Requirement(
                requirement_id="AWS_ROOT_VOLUME_SIZE",
                kind="infra",
                name="AWS_ROOT_VOLUME_SIZE",
                required=False,
            ),
        ),
    )
    outcome = await _infra_runner(
        tmp_path, episode_id, _spec_with_infra(episode_id, "AWS_ROOT_VOLUME_SIZE"), adapter
    ).run()

    assert outcome.status == "completed", outcome.diagnostic
    assert "prepare" in adapter.calls and "reset_start" in adapter.calls


def test_every_disclosed_infra_value_is_gated_and_stamped_from_one_table() -> None:
    """The gate and the stamp must not drift apart, and this is what binds them.

    ``scripts/run_episode.py`` used to repeat the disclosed names in its own
    hardcoded branch. Nothing tied the two lists together, so a third value
    could be stamped without a gate (a disclosure nothing verifies) or gated
    without a stamp (a check nothing records) -- neither visible by reading
    either file alone. Both now derive from ``DISCLOSED_INFRA_METADATA_KEYS``;
    this asserts the derivation rather than the current contents, so it keeps
    holding when a fourth value is added.
    """

    import scripts.run_episode as run_episode
    from local_operator.evaluation.runner.episode import _DISCLOSED_INFRA_VALUES

    # The gate's allowlist is exactly the table's keys -- not a copy of them.
    assert _DISCLOSED_INFRA_VALUES == frozenset(DISCLOSED_INFRA_METADATA_KEYS)

    # And every gated name reaches the manifest under its declared key, which
    # is the half a reviewer cannot see from episode.py alone.
    for name, key in DISCLOSED_INFRA_METADATA_KEYS.items():
        value = "throughput" if name == "OSWORLD_ACTION_SETTLE_POLICY" else "some-value"
        metadata = run_episode._infra_disclosure_metadata([f"{name}={value}"])
        assert metadata[key] == value, name
        assert metadata["osworld_action_settle_policy"] == "throughput"
        assert metadata["osworld_action_settle_seconds"] == 3.0

    # The keys are distinct: two values sharing one key would have the second
    # silently overwrite the first's disclosure.
    keys = list(DISCLOSED_INFRA_METADATA_KEYS.values())
    assert len(keys) == len(set(keys))


def test_an_undisclosed_infra_value_is_never_stamped() -> None:
    """The stamp is an allowlist too: an ordinary infra value must not leak
    into the manifest, and a value with no ``=`` must not be stamped empty."""

    import scripts.run_episode as run_episode

    assert run_episode._infra_disclosure_metadata(["OSWORLD_TTL_SECONDS=900"]) == {
        "osworld_action_settle_policy": "throughput",
        "osworld_action_settle_seconds": 3.0,
    }
    with pytest.raises(ValueError, match="--infra expects"):
        run_episode._infra_disclosure_metadata(["AWS_ROOT_VOLUME_SIZE"])
    assert run_episode._infra_disclosure_metadata([]) == {
        "osworld_action_settle_policy": "throughput",
        "osworld_action_settle_seconds": 3.0,
    }
    # Both at once, each under its own key.
    assert run_episode._infra_disclosure_metadata(
        ["AWS_INSTANCE_TYPE=m5.xlarge", "AWS_ROOT_VOLUME_SIZE=120", "OSWORLD_TTL_SECONDS=900"]
    ) == {
        "aws_instance_type_override": "m5.xlarge",
        "aws_root_volume_size_override": "120",
        "osworld_action_settle_policy": "throughput",
        "osworld_action_settle_seconds": 3.0,
    }


@pytest.mark.asyncio
async def test_host_refusal_does_not_publish_supplied_answer(
    tmp_path: Path, episode_id: str
) -> None:
    from local_operator.evaluation.adapters.api import AskUserExchangeResult

    adapter = FakeAdapter(tmp_path, episode_id)
    original = adapter._call_raw

    async def refusing(
        method: Any,
        params: Any,
        result_type: Any,
        *,
        timeout: float,
        execution_overhead_seconds_per_action: float = 0.0,
    ) -> Any:
        if method == "ask_user_exchange":
            adapter.calls.append(method)
            return AskUserExchangeResult(
                ask_id=params.ask_id, request_digest=params.request_digest, accepted=False
            )
        return await original(
            method,
            params,
            result_type,
            timeout=timeout,
            execution_overhead_seconds_per_action=execution_overhead_seconds_per_action,
        )

    adapter._call_raw = refusing
    model = ScriptedModel(["ask", "finish"])
    outcome = await _runner(
        tmp_path,
        episode_id,
        adapter=adapter,
        model=model,
        responder=RecordingResponder("must not publish"),
    ).run()
    assert outcome.status == "cancelled", outcome.diagnostic
    assert outcome.score is not None and outcome.score.status == "unscored"
    assert model.calls == 1
    assert outcome.bundle_root is not None
    assert "user_simulator_exchange" not in _kinds(outcome.bundle_root)
    assert "execute" not in adapter.calls and "score" not in adapter.calls


# ---------------------------------------------------------------------------
# The adapter worker's stderr tail on the failure path
# ---------------------------------------------------------------------------


class _StubTail:
    """Stands in for ``AdapterSupervisor.stderr_tail``.

    The real one is a bounded, thread-fed buffer with an idle signal; this is
    the same surface -- ``settled()`` for the failure path and ``bytes()`` for
    a snapshot -- so a test can hand the runner a tail without spawning a
    worker. ``with_settle=False`` models a launch seam whose tail object offers
    no settle at all, which the failure path must tolerate rather than raise
    on. The spawned path is covered in ``test_episode_subprocess``.
    """

    def __init__(self, data: bytes, *, with_settle: bool = True) -> None:
        self._data = data
        if with_settle:
            self.settled = lambda: self._data

    def bytes(self) -> bytes:
        return self._data


_TAIL_HEADER = "--- adapter stderr tail ---"


def _fatal_detail_text(root: Path) -> str:
    fatal = [error for error in payloads(root, ErrorPayload) if not error.retryable]
    assert len(fatal) == 1
    detail = fatal[0].detail_artifact
    assert detail is not None
    return (root / "artifacts" / detail.sha256).read_bytes().decode()


async def _fatal_run(
    tmp_path: Path,
    episode_id: str,
    adapter: FakeAdapter,
    *,
    redactions: RedactionSet | None = None,
) -> Any:
    outcome = await _runner(
        tmp_path,
        episode_id,
        adapter=adapter,
        model=ScriptedModel(["step", "step", "finish"]),
        redactions=redactions,
    ).run()
    assert outcome.status == "failed", outcome.diagnostic
    assert outcome.bundle_root is not None
    assert verify_bundle(outcome.bundle_root).valid
    return outcome


def _fatal_adapter(tmp_path: Path, episode_id: str) -> FakeAdapter:
    return FakeAdapter(
        tmp_path,
        episode_id,
        failures={"execute": SupervisionError("worker died mid-observe")},
        fail_after={"execute": 1},
    )


@pytest.mark.asyncio
async def test_the_worker_stderr_tail_rides_the_fatal_detail(
    tmp_path: Path, episode_id: str
) -> None:
    """Upstream's own explanation must reach the bundle, not be discarded.

    The supervisor drained the worker's stderr into a bounded tail and nothing
    read it, so the two canary episodes whose only recorded fact was
    "environment returned no screenshot frame" left no way to tell a full disk
    from a dead server.
    """

    adapter = _fatal_adapter(tmp_path, episode_id)
    adapter.stderr_tail = _StubTail(
        b"desktopenv.pycontroller: Failed to get screenshot. Status code: 502\n"
    )
    outcome = await _fatal_run(tmp_path, episode_id, adapter)

    text = _fatal_detail_text(outcome.bundle_root)
    # The summary is unchanged and still first: the tail is ADDITIONAL evidence.
    assert text.startswith("SupervisionError: worker died mid-observe")
    assert _TAIL_HEADER in text
    assert "Failed to get screenshot. Status code: 502" in text


@pytest.mark.asyncio
async def test_no_stderr_tail_leaves_the_detail_byte_identical(
    tmp_path: Path, episode_id: str
) -> None:
    """An absent tail adds nothing at all, down to the byte."""

    untailed = _fatal_adapter(tmp_path, episode_id)
    outcome = await _fatal_run(tmp_path, episode_id, untailed)
    # The pre-change assertion from `test_fatal_error_records_a_bounded_diagnostic_detail`,
    # restated for the tail-free case: a launch handle with no tail, and an
    # empty one, must not even add a header.
    assert _fatal_detail_text(outcome.bundle_root) == "SupervisionError: worker died mid-observe"

    # A second run needs its own roots and its own episode id: the evidence
    # terminal is immutable and the lifecycle lineage is per episode id.
    second = tmp_path / "empty-tail"
    second.mkdir()
    empty = _fatal_adapter(second, f"{episode_id}-empty")
    empty.stderr_tail = _StubTail(b"")
    outcome = await _fatal_run(second, f"{episode_id}-empty", empty)
    assert _fatal_detail_text(outcome.bundle_root) == "SupervisionError: worker died mid-observe"


@pytest.mark.asyncio
async def test_a_tail_that_can_only_be_snapshotted_still_surfaces(
    tmp_path: Path, episode_id: str
) -> None:
    """A launch seam's own tail object may offer no settle; it must not raise.

    The failure path is the last place that may turn a handled failure into a
    crash, so the narrower surface degrades to a snapshot rather than failing.
    """

    adapter = _fatal_adapter(tmp_path, episode_id)
    adapter.stderr_tail = _StubTail(b"snapshot-only tail\n", with_settle=False)
    outcome = await _fatal_run(tmp_path, episode_id, adapter)
    assert "snapshot-only tail" in _fatal_detail_text(outcome.bundle_root)


@pytest.mark.parametrize("encoded", [False, True])
@pytest.mark.asyncio
async def test_a_secret_in_the_stderr_tail_withholds_the_tail_whole(
    tmp_path: Path, episode_id: str, encoded: bool
) -> None:
    """The newly surfaced text is canary-checked, and the check is escape-aware.

    ``RedactionSet`` derives base64, percent-encoded and hex canaries from every
    resolved secret, so a plain substring check on the literal would miss the
    encoded form a library or a URL might carry. The tail is withheld WHOLE
    rather than masked, and only the tail: the diagnostic line is still
    published, so the failure stays bucketed.
    """

    from urllib.parse import quote

    secret = "AKIA-STDERR-CANARY-0123456789"
    carried = quote(secret, safe="") if encoded else secret
    adapter = _fatal_adapter(tmp_path, episode_id)
    adapter.stderr_tail = _StubTail(f"upstream said {carried} while fetching\n".encode())
    outcome = await _fatal_run(
        tmp_path, episode_id, adapter, redactions=RedactionSet.from_resolved_values((secret,))
    )

    text = _fatal_detail_text(outcome.bundle_root)
    assert secret not in text
    assert "upstream said" not in text
    assert _TAIL_HEADER in text and WITHHELD in text
    assert text.startswith("SupervisionError: worker died mid-observe")


@pytest.mark.asyncio
async def test_the_episodes_own_path_is_not_a_canary(tmp_path: Path, episode_id: str) -> None:
    """A resolved value the episode's own roots contain cannot prove a leak.

    Measured failure this closes (2026-10-17): the OSWorld rig's sealed infra
    refs widen the redaction set, so the benchmark's own INPUTS ROOT -- a path
    prefix of the run's cache directories -- became a canary; the scoring
    details capture carries the evaluator's log lines, which name those cache
    paths; the scoring receipt's scan then refused the artifact wholesale and
    every episode that reached scoring died with ``evidencefailure``. A
    location cannot distinguish leakage from ordinary operation, so values in
    that relationship with the episode's own roots are dropped from the
    canary set; a real secret still widens it, and still withholds the text
    that carries it.
    """

    own_path = str(tmp_path)
    secret = "S3CRET-CANARY-6c1f"
    spec = replace(
        build_spec(episode_id),
        secret_refs=(SecretRef(name="OWN_PATH_ROW"), SecretRef(name="AWS_ACCESS_KEY_ID")),
    )
    resolver = StaticSecretResolver({"OWN_PATH_ROW": own_path, "AWS_ACCESS_KEY_ID": secret})

    # (a) Text naming the episode's own tree SURFACES: the path value was
    # resolved but is not a canary (it is a substring of the roots themselves).
    adapter = _fatal_adapter(tmp_path, episode_id)
    adapter.stderr_tail = _StubTail(f"fetching {own_path}/cache/file.ics\n".encode())
    outcome = await _runner(
        tmp_path,
        episode_id,
        adapter=adapter,
        model=ScriptedModel(["step", "step", "finish"]),
        spec=spec,
        secrets=resolver,
    ).run()
    assert outcome.status == "failed", outcome.diagnostic
    assert outcome.bundle_root is not None
    text = _fatal_detail_text(outcome.bundle_root)
    assert f"fetching {own_path}/cache/file.ics" in text

    # (b) The resolved SECRET still withholds a tail that carries it: the
    # drop is scoped to values the episode's own roots contain, not a
    # weakening of the canary set.
    second = tmp_path / "secret-case"
    second.mkdir()
    adapter2 = _fatal_adapter(second, f"{episode_id}-secret")
    adapter2.stderr_tail = _StubTail(f"carrying {secret}\n".encode())
    outcome2 = await _runner(
        second,
        f"{episode_id}-secret",
        adapter=adapter2,
        model=ScriptedModel(["step", "step", "finish"]),
        spec=replace(
            build_spec(f"{episode_id}-secret"),
            secret_refs=(SecretRef(name="OWN_PATH_ROW"), SecretRef(name="AWS_ACCESS_KEY_ID")),
        ),
        secrets=resolver,
    ).run()
    assert outcome2.status == "failed", outcome2.diagnostic
    assert outcome2.bundle_root is not None
    text2 = _fatal_detail_text(outcome2.bundle_root)
    assert secret not in text2 and "carrying" not in text2
    assert WITHHELD in text2


def test_canary_drop_requires_path_shape(tmp_path: Path) -> None:
    """The drop is scoped to path-shaped locations, not bare containment.

    Round-1 hardening, made DISCRIMINATING in round 2: the roots live under a
    directory literally named ``[redacted]`` and the secret value is that same
    bare identifier, so the PRE-remediation predicate (bare containment)
    drops it and this test fails -- which is exactly the collision the
    narrowing exists to keep as a canary (a details artifact carrying
    "credential [redacted] accepted" must still be refused).
    """

    base = tmp_path / "[redacted]"
    roots = (base / "a", base / "b")
    values = (
        str(tmp_path),
        str(base / "a"),
        str(base / "a" / "x"),
        "[redacted]",
        "not-a-path-but-text",
    )
    kept = _canary_values_outside_own_roots(values, roots)
    # The three path-shaped locations are dropped (ancestor, equal, inside);
    # the identifier that is merely a path COMPONENT of every root, and the
    # non-path text, are kept -- in order.
    assert kept == ("[redacted]", "not-a-path-but-text")


@pytest.mark.asyncio
async def test_scoring_details_publish_own_path_but_refuse_real_secret(
    tmp_path: Path, episode_id: str
) -> None:
    """The exit criterion at the ACTUAL failing path, from both directions.

    DISCRIMINATING since round 2: the episode's roots live under a directory
    literally named ``[redacted]`` and the secret value is that same bare
    identifier, so under the PRE-remediation predicate every root contains it,
    the capture carrying it publishes, and the refusal assertion below fails.
    With the path-shape rule the identifier stays a canary and the refusal is
    real. (a) own-path-only details bytes publish and the run seals scored;
    (b) the same capture plus the identifier-shaped secret [redacted]
    refused at the receipt and the bundle takes the ambiguity path.
    """

    base = tmp_path / "[redacted]"
    base.mkdir()
    identifier = "[redacted]"
    own_path = str(base)

    async def _run_with_details(text: str, suffix: str, secrets: dict[str, str]) -> Any:
        # The supervision layer's reference check enforces the declared media
        # type, so the capture is canonical JSON with the payload text inside
        # -- the same envelope the real evaluator diagnostics use.
        payload = json.dumps(
            {"evaluator_diagnostics": {"stderr": {"text": text}}},
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
        sha = hashlib.sha256(payload).hexdigest()
        artifacts = base / "artifacts"
        artifacts.mkdir(exist_ok=True)
        (artifacts / sha).write_bytes(payload)
        details = EvidenceArtifactRef(
            sha256=sha, media_type="application/json", byte_count=len(payload)
        )
        case_id = f"{episode_id}-{suffix}"
        adapter = FakeAdapter(
            base, case_id, score=ScoreArtifact(status="scored", binary=0, details=details)
        )
        return await _runner(
            base,
            case_id,
            adapter=adapter,
            model=ScriptedModel(["step", "finish"]),
            spec=replace(
                build_spec(case_id),
                secret_refs=tuple(SecretRef(name=name) for name in secrets),
            ),
            secrets=StaticSecretResolver(secrets),
        ).run()

    # (a) The measured case: own-path text publishes, episode seals scored.
    sealed = await _run_with_details(
        f"fetching {own_path}/cache/file.ics", "own", {"OWN_PATH_ROW": own_path}
    )
    assert sealed.status == "completed", sealed.diagnostic
    assert sealed.score is not None and sealed.score.status == "scored"
    assert sealed.bundle_root is not None and verify_bundle(sealed.bundle_root).valid
    published = sealed.bundle_root / "artifacts"
    assert any(
        f"fetching {own_path}/cache/file.ics".encode() in path.read_bytes()
        for path in published.iterdir()
        if path.is_file()
    )

    # (b) The collision: the identifier-shaped secret stays a canary, so the
    # capture carrying it is refused and the bundle takes the ambiguity path.
    refused = await _run_with_details(
        f"carrying {own_path} under {identifier}",
        "secret",
        {"OWN_PATH_ROW": own_path, "IDENT_ROW": identifier},
    )
    assert refused.status == "abandoned", refused.diagnostic
    assert refused.score is None
    assert refused.bundle_root is not None
    report = verify_bundle(refused.bundle_root)
    assert report.abandonment is not None
    # The recorded terminal says the run died on evidence, which is this
    # case's whole point: the identifier-shaped secret made the receipt
    # unpublishable.
    assert report.abandonment.diagnostic_code == "evidencefailure"
    assert report.abandonment.reason == "ambiguous_finalization"
    assert not any(
        identifier.encode() in path.read_bytes()
        for path in (refused.bundle_root / "artifacts").iterdir()
        if path.is_file()
    )


@pytest.mark.asyncio
async def test_rescue_cancellation_propagates_without_retry(
    tmp_path: Path, episode_id: str
) -> None:
    """A CancelledError during rescue must not launch a second worker.

    Round-1 hardening: the broad retry handler used to swallow cancellation
    and start another rescue after the run was told to stop; the fix raises
    before it, so exactly one launch happened and the cancellation reaches the
    caller.
    """

    launches: list[int] = []

    async def _cancelling(descriptor: Any, **kwargs: Any) -> Any:
        del kwargs
        launches.append(1)
        raise asyncio.CancelledError()

    adapter = FakeAdapter(tmp_path, episode_id)
    runner = _runner(tmp_path, episode_id, adapter=adapter, model=ScriptedModel())

    class _StubDescriptor:
        descriptor_id = "d" * 64

    runner._descriptor = _StubDescriptor()  # pyright: ignore[reportAttributeAccessIssue]
    runner._rescue = _cancelling  # pyright: ignore[reportAttributeAccessIssue]

    with pytest.raises(asyncio.CancelledError):
        await runner._attempt_rescue()

    assert len(launches) == 1
    assert runner._rescue_launches == 1


@pytest.mark.asyncio
async def test_the_tail_is_scanned_before_it_is_bounded(tmp_path: Path, episode_id: str) -> None:
    """A secret outside the retained window is still found, because the scan
    sees the whole buffer before the bound is applied.

    Bounding first is the ordering that leaks: a canary straddling the cut no
    longer matches, and ``publish_artifact`` applies the same substring
    semantics to whatever bytes it receives, so it would publish the surviving
    fragment as clean.
    """

    secret = "AKIA-OUTSIDE-WINDOW-CANARY-0123456789"
    adapter = _fatal_adapter(tmp_path, episode_id)
    padding = b"x" * 20_000
    adapter.stderr_tail = _StubTail(secret.encode() + padding)
    outcome = await _fatal_run(
        tmp_path, episode_id, adapter, redactions=RedactionSet.from_resolved_values((secret,))
    )

    text = _fatal_detail_text(outcome.bundle_root)
    assert secret not in text
    assert WITHHELD in text


@pytest.mark.asyncio
async def test_the_tail_section_is_bounded_and_keeps_its_end(
    tmp_path: Path, episode_id: str
) -> None:
    """A chatty worker cannot inflate a bundle, and the END is what is kept."""

    adapter = _fatal_adapter(tmp_path, episode_id)
    adapter.stderr_tail = _StubTail(b"a" * 10_000 + b"TAIL-END-MARKER\n")
    outcome = await _fatal_run(tmp_path, episode_id, adapter)

    text = _fatal_detail_text(outcome.bundle_root)
    section = text.split(_TAIL_HEADER, 1)[1]
    assert section.endswith("TAIL-END-MARKER\n")
    assert len(section) <= MAX_STDERR_TAIL_CHARS + 64
    assert section.count("a") < 10_000
