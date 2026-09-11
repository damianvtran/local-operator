"""Independent verifier tamper and confinement tests."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest

from local_operator.evaluation.evidence.models import (
    ActionBatchPayload,
    BudgetCommitmentPayload,
    CancelPayload,
    CleanupPayload,
    ContextCompactionPayload,
    EnvironmentStepPayload,
    EventKind,
    EventPayload,
    EventRecord,
    FinalizationIntent,
    LifecycleTransitionPayload,
    ModelRequestPayload,
    ModelResponsePayload,
    ObservationPayload,
    PreflightPayload,
    ReconciliationPayload,
    ScoreArtifact,
    ScoringResultPayload,
    UsageCostPayload,
)
from local_operator.evaluation.evidence.store import (
    EvidenceBundleInvalid,
    EvidenceWriter,
)
from local_operator.evaluation.evidence.verify import verify_bundle
from local_operator.evaluation.receipts import RedactionSet
from tests.unit.evaluation.evidence.test_models import ARTIFACT, DIGEST, ROUTE, manifest


def _append_authority(writer: EvidenceWriter, *, timestamp: int = 0) -> None:
    writer.append(
        "preflight",
        PreflightPayload(sealed_preflight_id=DIGEST, plan_id=DIGEST, receipt_ids=(), passed=True),
        monotonic_ns=timestamp,
        wall_time_ms=timestamp,
    )
    writer.append(
        "budget_commitment",
        BudgetCommitmentPayload(
            commitment_id=DIGEST,
            budget_id=manifest().budget_id,
            reservation_ids=(),
            reserved_summary_digest=DIGEST,
        ),
        monotonic_ns=timestamp,
        wall_time_ms=timestamp,
    )


def _bundle(tmp_path: Path, *, events: int = 2, artifact: bool = False) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    root = tmp_path / "bundle"
    with EvidenceWriter.create(root, manifest(), RedactionSet.from_resolved_values(())) as writer:
        _append_authority(writer)
        ref = writer.publish_artifact(b"safe", media_type="text/plain") if artifact else None
        for index in range(events):
            if index == 0 and ref is not None:
                writer.append(
                    "observation",
                    ObservationPayload(observation_id="observation", sequence=0, artifacts=(ref,)),
                    monotonic_ns=index + 1,
                    wall_time_ms=index + 1,
                )
            else:
                writer.append(
                    "cancel",
                    CancelPayload(
                        cancellation_id=f"cancel-{index}",
                        source="operator",
                        diagnostic_code="requested",
                    ),
                    monotonic_ns=index + 1,
                    wall_time_ms=index + 1,
                )
    return root


def codes(root: Path) -> set[str]:
    return {issue.code for issue in verify_bundle(root).issues}


def test_event_tamper_truncate_reorder_duplicate_and_chain_mismatch(tmp_path: Path) -> None:
    root = _bundle(tmp_path)
    original = (root / "events.jsonl").read_bytes().splitlines(keepends=True)

    data = json.loads(original[2])
    data["payload"]["diagnostic_code"] = "tampered"
    original[2] = json.dumps(data, separators=(",", ":"), sort_keys=True).encode() + b"\n"
    (root / "events.jsonl").write_bytes(b"".join(original))
    assert "event_hash_mismatch" in codes(root)

    root = _bundle(tmp_path / "truncate")
    raw = (root / "events.jsonl").read_bytes()
    (root / "events.jsonl").write_bytes(raw[:-1])
    assert "journal_truncated" in codes(root)

    root = _bundle(tmp_path / "reorder")
    lines = (root / "events.jsonl").read_bytes().splitlines(keepends=True)
    (root / "events.jsonl").write_bytes(lines[1] + lines[0])
    assert {"event_sequence_mismatch", "event_chain_mismatch"} <= codes(root)

    root = _bundle(tmp_path / "duplicate")
    lines = (root / "events.jsonl").read_bytes().splitlines(keepends=True)
    (root / "events.jsonl").write_bytes(lines[0] + lines[0])
    assert "event_sequence_mismatch" in codes(root)

    root = _bundle(tmp_path / "chain")
    lines = (root / "events.jsonl").read_bytes().splitlines(keepends=True)
    data = json.loads(lines[1])
    data["previous_event_sha256"] = "f" * 64
    # Recompute through model validation so only the chain binding is wrong.
    lines[1] = (
        EventRecord.model_validate({**data, "event_id": "0" * 64}, strict=True).to_canonical_json()
        + b"\n"
    )
    (root / "events.jsonl").write_bytes(b"".join(lines))
    assert "event_chain_mismatch" in codes(root)


def _rewrite_canonical_order(root: Path, ordered_kinds: list[str]) -> None:
    source = {
        json.loads(line)["kind"]: json.loads(line)
        for line in (root / "events.jsonl").read_bytes().splitlines()
    }
    previous = manifest().manifest_digest
    records = []
    for sequence, kind in enumerate(ordered_kinds):
        value = source[kind]
        value.update(
            {
                "sequence": sequence,
                "previous_event_sha256": previous,
                "event_id": "0" * 64,
                "monotonic_ns": sequence,
                "wall_time_ms": sequence,
            }
        )
        event = EventRecord.model_validate(value, strict=True)
        records.append(event.to_canonical_json())
        previous = event.event_id
    (root / "events.jsonl").write_bytes(b"\n".join(records) + b"\n")


def _phase_bundle(tmp_path: Path) -> Path:
    root = tmp_path / "bundle"
    with EvidenceWriter.create(root, manifest(), RedactionSet.from_resolved_values(())) as writer:
        writer.append(
            "preflight",
            PreflightPayload(
                sealed_preflight_id=DIGEST, plan_id=DIGEST, receipt_ids=(), passed=True
            ),
            monotonic_ns=0,
            wall_time_ms=0,
        )
        writer.append(
            "budget_commitment",
            BudgetCommitmentPayload(
                commitment_id=DIGEST,
                budget_id=manifest().budget_id,
                reservation_ids=(),
                reserved_summary_digest=DIGEST,
            ),
            monotonic_ns=1,
            wall_time_ms=1,
        )
        writer.begin_finalization(
            "final",
            "score-op",
            FinalizationIntent(kind="score", scorer_id="scorer", scorer_version="1"),
            monotonic_ns=2,
            wall_time_ms=2,
        )
        score = ScoreArtifact(status="scored", binary=1)
        writer.record_scoring_result(
            ScoringResultPayload(
                finalization_id="final", scoring_operation_id="score-op", score=score
            ),
            monotonic_ns=3,
            wall_time_ms=3,
        )
        writer.record_reconciliation(
            ReconciliationPayload(
                reconciliation_id=DIGEST,
                budget_id=manifest().budget_id,
                commitment_id=DIGEST,
                reportable=True,
                provider_cost_microusd=0,
                environment_cost_microusd=0,
                total_cost_microusd=0,
            ),
            monotonic_ns=4,
            wall_time_ms=4,
        )
        writer.record_cleanup(
            CleanupPayload(
                cleanup_result_id=DIGEST,
                cleanup_plan_id=DIGEST,
                receipt_ids=(),
                rescue_required=False,
            ),
            monotonic_ns=5,
            wall_time_ms=5,
        )
        writer.record_final_lifecycle(
            LifecycleTransitionPayload(
                previous_state_id=None,
                state_id=DIGEST,
                state="completed",
                finalization_id="final",
                preflight_seal_id=DIGEST,
                commitment_id=DIGEST,
                reconciliation_id=DIGEST,
                reconciliation_reportable=True,
                score_id=score.score_id,
                cleanup_result_id=DIGEST,
                rescue_required=False,
            ),
            monotonic_ns=6,
            wall_time_ms=6,
        )
    return root


@pytest.mark.parametrize(
    "order",
    [
        [
            "budget_commitment",
            "preflight",
            "finalization_start",
            "scoring_start",
            "scoring_result",
            "reconciliation",
            "cleanup",
            "lifecycle_transition",
        ],
        [
            "preflight",
            "finalization_start",
            "budget_commitment",
            "scoring_start",
            "scoring_result",
            "reconciliation",
            "cleanup",
            "lifecycle_transition",
        ],
        [
            "preflight",
            "budget_commitment",
            "finalization_start",
            "scoring_start",
            "reconciliation",
            "scoring_result",
            "cleanup",
            "lifecycle_transition",
        ],
        [
            "preflight",
            "budget_commitment",
            "finalization_start",
            "scoring_start",
            "scoring_result",
            "cleanup",
            "reconciliation",
            "lifecycle_transition",
        ],
        [
            "preflight",
            "budget_commitment",
            "finalization_start",
            "scoring_start",
            "scoring_result",
            "reconciliation",
            "lifecycle_transition",
            "cleanup",
        ],
    ],
)
def test_verifier_rejects_canonically_rehashed_phase_reordering(
    tmp_path: Path, order: list[str]
) -> None:
    root = _phase_bundle(tmp_path)
    _rewrite_canonical_order(root, order)
    report = verify_bundle(root)
    assert not report.valid
    assert {"event_order_invalid", "lifecycle_invalid"} & {issue.code for issue in report.issues}


def test_generic_append_cannot_bypass_finalization_receipt_order(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    with EvidenceWriter.create(root, manifest(), RedactionSet.from_resolved_values(())) as writer:
        writer.append(
            "preflight",
            PreflightPayload(
                sealed_preflight_id=DIGEST, plan_id=DIGEST, receipt_ids=(), passed=True
            ),
        )
        writer.append(
            "budget_commitment",
            BudgetCommitmentPayload(
                commitment_id=DIGEST,
                budget_id=manifest().budget_id,
                reservation_ids=(),
                reserved_summary_digest=DIGEST,
            ),
        )
        with pytest.raises(EvidenceBundleInvalid, match="cleanup.*phase"):
            writer.append(
                "cleanup",
                CleanupPayload(
                    cleanup_result_id=DIGEST,
                    cleanup_plan_id=DIGEST,
                    receipt_ids=(),
                    rescue_required=False,
                ),
            )
        with pytest.raises(EvidenceBundleInvalid, match="reconciliation.*phase"):
            writer.append(
                "reconciliation",
                ReconciliationPayload(
                    reconciliation_id=DIGEST,
                    budget_id=manifest().budget_id,
                    commitment_id=DIGEST,
                    reportable=True,
                    provider_cost_microusd=0,
                    environment_cost_microusd=0,
                    total_cost_microusd=0,
                ),
            )


def test_semantic_graph_rejects_duplicate_and_out_of_order_receipts(
    tmp_path: Path,
) -> None:
    root = tmp_path / "bundle"
    with EvidenceWriter.create(root, manifest(), RedactionSet.from_resolved_values(())) as writer:
        _append_authority(writer)
        writer.append(
            "usage_cost",
            UsageCostPayload(
                request_id="request",
                input_tokens=1,
                output_tokens=1,
                cost_microusd=1,
            ),
            monotonic_ns=1,
            wall_time_ms=1,
        )
        request = ModelRequestPayload(
            request_id="request",
            requested_route=ROUTE,
            tool_schema_digest=DIGEST,
            input_tokens=1,
            message_count=1,
            tool_count=0,
        )
        writer.append("model_request", request, monotonic_ns=2, wall_time_ms=2)
        writer.append("model_request", request, monotonic_ns=3, wall_time_ms=3)
        response = ModelResponsePayload(
            request_id="request",
            provider_request_id="provider-request",
            requested_route=ROUTE,
            served_route=ROUTE,
            stop_reason="end",
            output_tokens=1,
            reasoning_tokens=0,
            tool_call_count=0,
        )
        writer.append("model_response", response, monotonic_ns=4, wall_time_ms=4)
        writer.append("model_response", response, monotonic_ns=5, wall_time_ms=5)
    report = verify_bundle(root)
    assert "receipt_binding_invalid" in {issue.code for issue in report.issues}


def _request_triple(writer: EvidenceWriter, request_id: str, *, at: int) -> None:
    writer.append(
        "model_request",
        ModelRequestPayload(
            request_id=request_id,
            requested_route=ROUTE,
            tool_schema_digest=DIGEST,
            input_tokens=1,
            message_count=1,
            tool_count=0,
        ),
        monotonic_ns=at,
        wall_time_ms=at,
    )
    writer.append(
        "model_response",
        ModelResponsePayload(
            request_id=request_id,
            provider_request_id="provider-request",
            requested_route=ROUTE,
            served_route=ROUTE,
            stop_reason="end",
            output_tokens=1,
            reasoning_tokens=0,
            tool_call_count=0,
        ),
        monotonic_ns=at + 1,
        wall_time_ms=at + 1,
    )
    writer.append(
        "usage_cost",
        UsageCostPayload(request_id=request_id, input_tokens=1, output_tokens=1, cost_microusd=1),
        monotonic_ns=at + 2,
        wall_time_ms=at + 2,
    )


def _compaction(compaction_id: str, previous: str | None) -> ContextCompactionPayload:
    return ContextCompactionPayload(
        compaction_id=compaction_id,
        previous_request_id=previous,
        strategy="context-full",
        tokens_before=10,
        tokens_after=5,
        frames_dropped=1,
        messages_before=4,
        messages_after=2,
    )


def test_compaction_between_closed_requests_verifies_and_counts(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    with EvidenceWriter.create(root, manifest(), RedactionSet.from_resolved_values(())) as writer:
        _append_authority(writer)
        _request_triple(writer, "request-0", at=1)
        writer.append(
            "context_compaction",
            _compaction("compaction-0", "request-0"),
            monotonic_ns=4,
            wall_time_ms=4,
        )
        _request_triple(writer, "request-1", at=5)
    report = verify_bundle(root)
    assert report.valid, [issue.code for issue in report.issues]
    assert report.counters is not None and report.counters.compactions == 1
    assert [event.kind for event in report.events][2:] == [
        "model_request",
        "model_response",
        "usage_cost",
        "context_compaction",
        "model_request",
        "model_response",
        "usage_cost",
    ]


def test_compaction_event_inside_a_request_triple_is_invalid(tmp_path: Path) -> None:
    """A rebuild mid-request would mean the recorded request is not the one
    the model saw, so the verifier refuses a compaction that does not sit
    between one request's usage receipt and the next request."""

    root = tmp_path / "bundle"
    with EvidenceWriter.create(root, manifest(), RedactionSet.from_resolved_values(())) as writer:
        _append_authority(writer)
        writer.append(
            "model_request",
            ModelRequestPayload(
                request_id="request-0",
                requested_route=ROUTE,
                tool_schema_digest=DIGEST,
                input_tokens=1,
                message_count=1,
                tool_count=0,
            ),
            monotonic_ns=1,
            wall_time_ms=1,
        )
        writer.append(
            "context_compaction", _compaction("compaction-0", None), monotonic_ns=2, wall_time_ms=2
        )
    assert "receipt_binding_invalid" in codes(root)

    # A duplicate compaction id, and one naming a request that never closed.
    root = tmp_path / "duplicate"
    with EvidenceWriter.create(root, manifest(), RedactionSet.from_resolved_values(())) as writer:
        _append_authority(writer)
        _request_triple(writer, "request-0", at=1)
        writer.append(
            "context_compaction",
            _compaction("compaction-0", "request-0"),
            monotonic_ns=4,
            wall_time_ms=4,
        )
        writer.append(
            "context_compaction",
            _compaction("compaction-0", "request-0"),
            monotonic_ns=5,
            wall_time_ms=5,
        )
    assert "receipt_binding_invalid" in codes(root)

    root = tmp_path / "unknown-previous"
    with EvidenceWriter.create(root, manifest(), RedactionSet.from_resolved_values(())) as writer:
        _append_authority(writer)
        _request_triple(writer, "request-0", at=1)
        writer.append(
            "context_compaction",
            _compaction("compaction-0", "request-9"),
            monotonic_ns=4,
            wall_time_ms=4,
        )
    assert "receipt_binding_invalid" in codes(root)


def test_semantic_graph_rejects_broken_observation_action_step_links(
    tmp_path: Path,
) -> None:
    root = tmp_path / "bundle"
    with EvidenceWriter.create(root, manifest(), RedactionSet.from_resolved_values(())) as writer:
        _append_authority(writer)
        writer.append(
            "action_batch",
            ActionBatchPayload(
                action_batch_id="batch",
                observation_id="missing",
                action_count=1,
                action_artifact=ARTIFACT,
            ),
            monotonic_ns=1,
            wall_time_ms=1,
        )
        writer.append(
            "environment_step",
            EnvironmentStepPayload(
                step_id="step",
                action_batch_id="batch",
                receipt_id=DIGEST,
                input_observation_id="different",
                output_observation_id="output",
                terminated=False,
                truncated=False,
            ),
            monotonic_ns=2,
            wall_time_ms=2,
        )
    report = verify_bundle(root)
    assert "receipt_binding_invalid" in {issue.code for issue in report.issues}


def test_open_graph_rejects_sequence_gap_duplicate_batch_and_step(
    tmp_path: Path,
) -> None:
    root = tmp_path / "bundle"
    with EvidenceWriter.create(root, manifest(), RedactionSet.from_resolved_values(())) as writer:
        _append_authority(writer)
        writer.append(
            "observation",
            ObservationPayload(observation_id="observation", sequence=99),
            monotonic_ns=1,
            wall_time_ms=1,
        )
        for batch_id in ("batch-1", "batch-2"):
            writer.append(
                "action_batch",
                ActionBatchPayload(
                    action_batch_id=batch_id,
                    observation_id="observation",
                    action_count=1,
                    action_artifact=ARTIFACT,
                ),
                monotonic_ns=2 if batch_id == "batch-1" else 3,
                wall_time_ms=2 if batch_id == "batch-1" else 3,
            )
        for step_id in ("step-1", "step-2"):
            writer.append(
                "environment_step",
                EnvironmentStepPayload(
                    step_id=step_id,
                    action_batch_id="batch-1",
                    receipt_id=DIGEST,
                    input_observation_id="observation",
                    output_observation_id=f"output-{step_id}",
                    terminated=False,
                    truncated=False,
                ),
                monotonic_ns=4 if step_id == "step-1" else 5,
                wall_time_ms=4 if step_id == "step-1" else 5,
            )
    report = verify_bundle(root)
    assert "receipt_binding_invalid" in {issue.code for issue in report.issues}


def test_terminal_step_requires_final_observation_and_lifecycle(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    with EvidenceWriter.create(root, manifest(), RedactionSet.from_resolved_values(())) as writer:
        _append_authority(writer)
        writer.append(
            "observation",
            ObservationPayload(observation_id="observation-0", sequence=0),
            monotonic_ns=1,
            wall_time_ms=1,
        )
        writer.append(
            "action_batch",
            ActionBatchPayload(
                action_batch_id="batch",
                observation_id="observation-0",
                action_count=1,
                action_artifact=ARTIFACT,
            ),
            monotonic_ns=2,
            wall_time_ms=2,
        )
        writer.append(
            "environment_step",
            EnvironmentStepPayload(
                step_id="step",
                action_batch_id="batch",
                receipt_id=DIGEST,
                input_observation_id="observation-0",
                output_observation_id="observation-1",
                terminated=True,
                truncated=False,
            ),
            monotonic_ns=3,
            wall_time_ms=3,
        )
        writer.append(
            "observation",
            ObservationPayload(observation_id="observation-1", sequence=1),
            monotonic_ns=4,
            wall_time_ms=4,
        )
    assert "finalization_invalid" in {issue.code for issue in verify_bundle(root).issues}


def test_action_after_terminal_step_is_invalid(tmp_path: Path) -> None:
    root = tmp_path / "bundle"
    with EvidenceWriter.create(root, manifest(), RedactionSet.from_resolved_values(())) as writer:
        _append_authority(writer)
        writer.append(
            "observation",
            ObservationPayload(observation_id="observation-0", sequence=0),
            monotonic_ns=1,
            wall_time_ms=1,
        )
        writer.append(
            "action_batch",
            ActionBatchPayload(
                action_batch_id="batch-0",
                observation_id="observation-0",
                action_count=1,
                action_artifact=ARTIFACT,
            ),
            monotonic_ns=2,
            wall_time_ms=2,
        )
        writer.append(
            "environment_step",
            EnvironmentStepPayload(
                step_id="step-0",
                action_batch_id="batch-0",
                receipt_id=DIGEST,
                input_observation_id="observation-0",
                output_observation_id="observation-1",
                terminated=False,
                truncated=True,
            ),
            monotonic_ns=3,
            wall_time_ms=3,
        )
        writer.append(
            "observation",
            ObservationPayload(observation_id="observation-1", sequence=1),
            monotonic_ns=4,
            wall_time_ms=4,
        )
        writer.append(
            "action_batch",
            ActionBatchPayload(
                action_batch_id="batch-1",
                observation_id="observation-1",
                action_count=1,
                action_artifact=ARTIFACT,
            ),
            monotonic_ns=5,
            wall_time_ms=5,
        )
    assert "finalization_invalid" in {issue.code for issue in verify_bundle(root).issues}


def test_partial_tail_refuses_append_without_repair(tmp_path: Path) -> None:
    root = _bundle(tmp_path, events=1)
    with (root / "events.jsonl").open("ab") as handle:
        handle.write(b'{"partial":')
    before = (root / "events.jsonl").read_bytes()
    with pytest.raises(EvidenceBundleInvalid):
        EvidenceWriter.open_for_abandon(root, RedactionSet.from_resolved_values(()))
    assert (root / "events.jsonl").read_bytes() == before


def test_artifact_substitution_count_media_unknown_and_unreferenced(tmp_path: Path) -> None:
    root = _bundle(tmp_path, artifact=True)
    artifact = next((root / "artifacts").iterdir())
    artifact.write_bytes(b"evil")
    assert "artifact_hash_mismatch" in codes(root)

    root = _bundle(tmp_path / "unknown", artifact=True)
    (root / "artifacts" / ("f" * 64)).write_bytes(b"unknown")
    assert "artifact_unreferenced" in codes(root)

    root = _bundle(tmp_path / "uppercase", artifact=True)
    artifact = next((root / "artifacts").iterdir())
    artifact.rename(root / "artifacts" / artifact.name.upper())
    assert "artifact_name_invalid" in codes(root)

    root = _bundle(tmp_path / "count", artifact=True)
    raw = (root / "events.jsonl").read_bytes().splitlines()
    data = json.loads(raw[2])
    data["payload"]["artifacts"][0]["byte_count"] = 99
    tampered = EventRecord.model_validate({**data, "event_id": "0" * 64}, strict=True)
    (root / "events.jsonl").write_bytes(
        raw[0] + b"\n" + raw[1] + b"\n" + tampered.to_canonical_json() + b"\n"
    )
    assert "artifact_count_mismatch" in codes(root)


def test_root_artifact_symlink_fifo_and_unknown_entry_attacks(tmp_path: Path) -> None:
    root = _bundle(tmp_path / "root-symlink")
    manifest_path = root / "manifest.json"
    manifest_path.unlink()
    manifest_path.symlink_to(tmp_path / "outside")
    assert "unsafe_path" in codes(root)

    root = _bundle(tmp_path / "artifact-symlink", artifact=True)
    artifact = next((root / "artifacts").iterdir())
    artifact.unlink()
    artifact.symlink_to(tmp_path / "outside")
    assert "artifact_unsafe" in codes(root)

    if hasattr(os, "mkfifo"):
        root = _bundle(tmp_path / "fifo", artifact=True)
        artifact = next((root / "artifacts").iterdir())
        artifact.unlink()
        os.mkfifo(artifact)
        assert "artifact_unsafe" in codes(root)

    root = _bundle(tmp_path / "unknown-entry")
    (root / "surprise").write_text("unknown")
    assert "unknown_root_entry" in codes(root)


def test_verifier_reports_unsafe_permissions_for_root_and_files(tmp_path: Path) -> None:
    root = _bundle(tmp_path, events=1)
    root.chmod(0o777)
    (root / "events.jsonl").chmod(0o666)
    report = verify_bundle(root)
    locations = {(issue.code, issue.location) for issue in report.issues}
    assert ("unsafe_permissions", ".") in locations
    assert ("unsafe_permissions", "events.jsonl") in locations


def test_state_is_diagnostic_and_terminal_files_win(tmp_path: Path) -> None:
    root = _bundle(tmp_path, events=0)
    state = json.loads((root / "state.json").read_bytes())
    state["state"] = "sealed"
    state["terminal_id"] = "f" * 64
    (root / "state.json").write_text(
        json.dumps(state, separators=(",", ":"), sort_keys=True), encoding="utf-8"
    )
    report = verify_bundle(root)
    assert report.terminal_state == "open"
    assert "state_stale" in {issue.code for issue in report.issues}


def test_sparse_oversized_journal_and_artifact_are_bounded(tmp_path: Path) -> None:
    import subprocess
    import sys

    # Runs as a subprocess, so pytest's `tmp_path` cannot reclaim the bundle:
    # the script owns cleanup via TemporaryDirectory. The files here are
    # sparse (`truncate` to 128 MiB / 256 MiB without writing), so the leak was
    # small on disk but still a directory per run that nothing removed.
    script = r"""
import json, os, resource, tempfile
from pathlib import Path
from local_operator.evaluation.evidence.store import EvidenceWriter
from local_operator.evaluation.evidence.verify import verify_bundle
from local_operator.evaluation.receipts import RedactionSet
from tests.unit.evaluation.evidence.test_models import manifest
with tempfile.TemporaryDirectory() as tmp:
 root=Path(tmp)/'bundle'
 with EvidenceWriter.create(root,manifest(),RedactionSet.from_resolved_values(())):
  pass
 with open(root/'events.jsonl','wb') as stream:
  stream.truncate(128*1024*1024)
 before=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
 journal=verify_bundle(root)
 after_journal=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
 (root/'events.jsonl').write_bytes(b'')
 digest='f'*64
 with open(root/'artifacts'/digest,'wb') as stream:
  stream.truncate(256*1024*1024+1)
 artifact=verify_bundle(root)
 after_artifact=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
 scale=1 if __import__('sys').platform=='darwin' else 1024
 print(json.dumps({
  'journal':[i.code for i in journal.issues],
  'artifact':[i.code for i in artifact.issues],
  'rss_delta':(max(after_journal,after_artifact)-before)*scale,
 }))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        text=True,
        capture_output=True,
        timeout=60,
        check=True,
    )
    payload = json.loads(result.stdout)
    assert "resource_limit_exceeded" in payload["journal"]
    assert "resource_limit_exceeded" in payload["artifact"]
    assert payload["rss_delta"] < 32 * 1024 * 1024


def test_import_graph_remains_inert_in_fresh_subprocess(tmp_path: Path) -> None:
    import subprocess
    import sys

    script = """
import json, sys
module = sys.argv[1]
import importlib
importlib.import_module(module)
forbidden = (
    'local_operator.providers', 'local_operator.config', 'local_operator.tools',
    'local_operator.tui', 'local_operator.mobile', 'textual', 'boto', 'osworld', 'PIL'
)
loaded = [
    name for name in sys.modules
    if any(name == item or name.startswith(item + '.') for item in forbidden)
]
print(json.dumps(loaded))
"""
    for module in (
        "local_operator.cli",
        "local_operator.session_factory",
        "local_operator.evaluation",
        "local_operator.evaluation.evidence",
        "local_operator.evaluation.evidence.models",
        "local_operator.evaluation.evidence.store",
        "local_operator.evaluation.evidence.verify",
    ):
        result = subprocess.run(
            [sys.executable, "-c", script, module],
            text=True,
            capture_output=True,
            check=True,
        )
        if module.endswith(("evaluation", "evidence")):
            assert result.stdout.strip() == "[]"
        if module in (
            "local_operator.cli",
            "local_operator.session_factory",
            "local_operator.evaluation",
        ):
            assert "local_operator.evaluation.evidence.models" not in result.stdout


def _interrupted_bundle(
    tmp_path: Path,
    *,
    failure_kind: str | None,
    scored: bool = False,
) -> Path:
    """A bundle whose last step is an ordinary non-terminal one.

    ``failure_kind`` None models an episode claiming it ended deliberately;
    a value models one that was interrupted (crash, outage, cancel).
    """

    root = tmp_path / "bundle"
    clock = iter(range(1, 1000))
    with EvidenceWriter.create(root, manifest(), RedactionSet.from_resolved_values(())) as writer:
        action = writer.publish_artifact(b"{}", media_type="application/json")

        def at() -> int:
            return next(clock)

        def append(kind: EventKind, payload: EventPayload) -> None:
            moment = at()
            writer.append(kind, payload, monotonic_ns=moment, wall_time_ms=moment)

        append(
            "preflight",
            PreflightPayload(
                sealed_preflight_id=DIGEST, plan_id=DIGEST, receipt_ids=(), passed=True
            ),
        )
        append(
            "budget_commitment",
            BudgetCommitmentPayload(
                commitment_id=DIGEST,
                budget_id=manifest().budget_id,
                reservation_ids=(),
                reserved_summary_digest=DIGEST,
            ),
        )
        append(
            "lifecycle_transition",
            LifecycleTransitionPayload(
                previous_state_id=None, state_id=DIGEST, state="running", commitment_id=DIGEST
            ),
        )
        append("observation", ObservationPayload(observation_id="obs-0", sequence=0))
        append(
            "action_batch",
            ActionBatchPayload(
                action_batch_id="batch-0",
                observation_id="obs-0",
                action_count=1,
                action_artifact=action,
            ),
        )
        append(
            "environment_step",
            EnvironmentStepPayload(
                step_id="step-0",
                action_batch_id="batch-0",
                receipt_id=DIGEST,
                input_observation_id="obs-0",
                output_observation_id="obs-1",
                terminated=False,
                truncated=False,
            ),
        )
        append("observation", ObservationPayload(observation_id="obs-1", sequence=1))
        score = (
            ScoreArtifact(status="scored", binary=1)
            if scored
            else ScoreArtifact(status="unscored", reason="crash")
        )
        moment = at()
        writer.begin_finalization(
            "final",
            "score-op" if scored else None,
            (
                FinalizationIntent(kind="score", scorer_id="scorer", scorer_version="1")
                if scored
                else FinalizationIntent(kind="unscored")
            ),
            monotonic_ns=moment,
            wall_time_ms=moment,
        )
        if scored:
            moment = at()
            writer.record_scoring_result(
                ScoringResultPayload(
                    finalization_id="final", scoring_operation_id="score-op", score=score
                ),
                monotonic_ns=moment,
                wall_time_ms=moment,
            )
        moment = at()
        writer.record_reconciliation(
            ReconciliationPayload(
                reconciliation_id=DIGEST,
                budget_id=manifest().budget_id,
                commitment_id=DIGEST,
                reportable=False,
                provider_cost_microusd=0,
                environment_cost_microusd=0,
                total_cost_microusd=0,
            ),
            monotonic_ns=moment,
            wall_time_ms=moment,
        )
        moment = at()
        writer.record_cleanup(
            CleanupPayload(
                cleanup_result_id=DIGEST,
                cleanup_plan_id=DIGEST,
                receipt_ids=(),
                rescue_required=False,
            ),
            monotonic_ns=moment,
            wall_time_ms=moment,
        )
        moment = at()
        writer.record_final_lifecycle(
            LifecycleTransitionPayload(
                previous_state_id=DIGEST,
                state_id="a" * 64,
                state="completed",
                finalization_id="final",
                preflight_seal_id=DIGEST,
                commitment_id=DIGEST,
                reconciliation_id=DIGEST,
                reconciliation_reportable=False,
                score_id=score.score_id,
                cleanup_result_id=DIGEST,
                rescue_required=False,
                failure_kind=failure_kind,  # pyright: ignore[reportArgumentType]
            ),
            monotonic_ns=moment,
            wall_time_ms=moment,
        )
    return root


@pytest.mark.parametrize("failure_kind", ["crash", "infrastructure", "model", "cancelled"])
def test_interrupted_episode_keeps_its_non_terminal_last_step(
    tmp_path: Path, failure_kind: str
) -> None:
    """An episode that was stopped cannot also have ended deliberately.

    Regression for ep-ffda3fc88f81: requiring a terminal step or a finish
    action here made every interrupted run unsealable, so a paid 16-step
    episode was thrown away wholesale.
    """

    root = _interrupted_bundle(tmp_path, failure_kind=failure_kind)
    assert "finalization_invalid" not in codes(root)


def test_completed_episode_still_needs_a_deliberate_ending(tmp_path: Path) -> None:
    """The exemption is keyed on ``failure_kind``, so a clean run is unaffected.

    Mutation guard: dropping the ``failure_kind`` condition and exempting every
    terminal bundle would let an episode that simply stopped stepping claim a
    normal completion.
    """

    root = _interrupted_bundle(tmp_path, failure_kind=None)
    assert "finalization_invalid" in codes(root)


def test_interruption_cannot_launder_a_scored_result(tmp_path: Path) -> None:
    """A scored result never rides the interruption exemption.

    The runner only seals a score with ``failure_kind=None``; this pins that a
    bundle claiming BOTH a score and an interruption is still rejected, so the
    exemption cannot be used to report a truncated run as a real score.
    """

    root = _interrupted_bundle(tmp_path, failure_kind="crash", scored=True)
    assert "finalization_invalid" in codes(root)


def _drop_events_rechain(root: Path, drop_kinds: set[str]) -> None:
    """Remove events by kind and recompute the hash chain so only the named
    events are missing -- every remaining record still validates and links.

    Used to build scoring shapes the writer's phase machine refuses to emit
    (a start with no result, a result with no start), so each clause of the
    interruption exemption's ``unscored`` predicate can be pinned on its own.
    """

    events = [json.loads(line) for line in (root / "events.jsonl").read_bytes().splitlines()]
    kept = [event for event in events if event["kind"] not in drop_kinds]
    previous = manifest().manifest_digest
    records = []
    for sequence, event in enumerate(kept):
        event = dict(event)
        event["sequence"] = sequence
        event["previous_event_sha256"] = previous
        event["event_id"] = "0" * 64
        record = EventRecord.model_validate(event, strict=True)
        records.append(record.to_canonical_json())
        previous = record.event_id
    (root / "events.jsonl").write_bytes(b"\n".join(records) + b"\n")


def _finalization_invalid_count(root: Path) -> int:
    return sum(1 for issue in verify_bundle(root).issues if issue.code == "finalization_invalid")


def test_scoring_start_without_result_still_refuses_interruption(tmp_path: Path) -> None:
    """Pins the ``not starts`` clause of the exemption on its own.

    Both laundering tests carry a scoring start AND a result together, so
    dropping only ``not starts`` left them green. A bundle that opened scoring
    (a ``scoring_start``) but never recorded a result still claims a run in
    progress, so an interruption must not exempt it: with ``not starts`` gone
    the exemption would fire here and the finalization_invalid disappears.
    """

    root = _interrupted_bundle(tmp_path, failure_kind="crash", scored=True)
    _drop_events_rechain(root, {"scoring_result"})
    assert "finalization_invalid" in codes(root)
    # Exactly two: the deliberate-ending refusal plus the scored-terminal
    # failure-kind guard (a score may carry only None/"model"). Under the M4
    # mutation (drop ``not starts``) the exemption fires and this count falls
    # to one -- only the kind guard remains.
    assert _finalization_invalid_count(root) == 2


def test_scoring_result_without_start_still_refuses_interruption(tmp_path: Path) -> None:
    """Pins the ``not results`` clause of the exemption on its own.

    Mirror of the start pin: a bundle carrying a scoring result but no start is
    structurally invalid (one finalization_invalid), and the interruption
    exemption must add a SECOND refusal because a result claims a completed
    rollout. Dropping only ``not results`` (M5) exempts the bundle and the count
    falls back to the single structural issue.
    """

    root = _interrupted_bundle(tmp_path, failure_kind="crash", scored=True)
    _drop_events_rechain(root, {"scoring_start"})
    assert "finalization_invalid" in codes(root)
    # Three: the structural scoring violation, the deliberate-ending refusal,
    # and the scored-terminal failure-kind guard. Dropping only ``not
    # results`` (M5) exempts the bundle from the last two and the count falls
    # to one.
    assert _finalization_invalid_count(root) == 3


def _rewrite_events(root: Path, transform: Any) -> None:
    """Apply an arbitrary edit to the journal, then renumber and rechain.

    Generalises ``_drop_events_rechain``: the transform may insert or edit
    records (not only drop them), which is how the agent-stop matrix builds
    shapes the writer's phase machine refuses to emit -- a stop bound to the
    wrong observation, a second stop, execution after a stop. Timestamps are
    renumbered to the sequence so an inserted record can never move time
    backward.
    """

    events = [json.loads(line) for line in (root / "events.jsonl").read_bytes().splitlines()]
    kept = transform(events)
    previous = manifest().manifest_digest
    records = []
    for sequence, event in enumerate(kept):
        event = dict(event)
        event["sequence"] = sequence
        event["previous_event_sha256"] = previous
        event["monotonic_ns"] = sequence + 1
        event["wall_time_ms"] = sequence + 1
        event["event_id"] = "0" * 64
        record = EventRecord.model_validate(event, strict=True)
        records.append(record.to_canonical_json())
        previous = record.event_id
    (root / "events.jsonl").write_bytes(b"\n".join(records) + b"\n")


def _stop_event(
    reason: str, *, observation_id: str = "obs-1", stop_id: str = "stop-0"
) -> dict[str, Any]:
    return {
        "kind": "agent_stop",
        "payload": {
            "stop_id": stop_id,
            "reason": reason,
            "observation_id": observation_id,
            "attempts": 3,
            "detail_artifact": None,
        },
    }


def _insert_before(
    events: list[dict[str, Any]], kind: str, new: dict[str, Any]
) -> list[dict[str, Any]]:
    index = next(i for i, event in enumerate(events) if event["kind"] == kind)
    return [*events[:index], new, *events[index:]]


def _stopped_bundle(
    tmp_path: Path,
    *,
    reason: str,
    failure_kind: str | None,
    stop: dict[str, Any] | None,
) -> Path:
    """A scored bundle whose rollout ended on a between-steps agent stop.

    ``_interrupted_bundle`` provides the stepped rollout (obs-0 -> step ->
    obs-1, obs-1 batchless); the rewrite inserts the stop event (or leaves it
    out when ``stop`` is None) immediately before finalization, which is where
    the runner writes it.
    """

    root = _interrupted_bundle(tmp_path, failure_kind=failure_kind, scored=True)
    if stop is not None:
        _rewrite_events(root, lambda events: _insert_before(events, "finalization_start", stop))
    return root


_ERROR_CODES = {"finalization_invalid", "receipt_binding_invalid", "event_order_invalid"}


@pytest.mark.parametrize(
    ("reason", "failure_kind"),
    [("model_failure", "model"), ("ask_unanswered", None)],
)
def test_agent_stop_is_a_deliberate_ending_that_may_score(
    tmp_path: Path, reason: str, failure_kind: str | None
) -> None:
    """The two runner shapes verify clean.

    Decision exhaustion keeps kind "model" for audit and an unanswered ask
    keeps kind None; both score the state the episode reached, exactly like a
    truncation. Each of the reject cases below pins one arm whose removal
    would let exactly one of these laundered shapes through.
    """

    root = _stopped_bundle(
        tmp_path, reason=reason, failure_kind=failure_kind, stop=_stop_event(reason)
    )
    assert not codes(root) & _ERROR_CODES


def test_scored_model_failure_without_a_stop_is_rejected(tmp_path: Path) -> None:
    """Pins the scored-kind guard's "stop required" direction.

    A scored terminal may claim kind "model" ONLY beside the agent_stop that
    records the exhaustion. Exactly two refusals fire: the deliberate-ending
    check (no stop, no finish, no terminal step) and this guard -- so
    removing EITHER arm is detectable as the count falling to one.
    """

    root = _stopped_bundle(tmp_path, reason="model_failure", failure_kind="model", stop=None)
    assert _finalization_invalid_count(root) == 2


@pytest.mark.parametrize(
    ("reason", "failure_kind"),
    [("model_failure", None), ("ask_unanswered", "model")],
)
def test_stop_reason_must_agree_with_the_failure_kind(
    tmp_path: Path, reason: str, failure_kind: str | None
) -> None:
    """Pins the reason-agreement guard's pairing direction.

    The runner writes exactly the pairs (model_failure, "model") and
    (ask_unanswered, None); either crossed pairing is a terminal record that
    contradicts the stop event, so neither may verify. The (model_failure,
    None) cross isolates the agreement arm -- the deliberate-ending check is
    satisfied by the stop and the kind is None, so exactly one refusal fires.
    """

    root = _stopped_bundle(
        tmp_path, reason=reason, failure_kind=failure_kind, stop=_stop_event(reason)
    )
    expected = 1 if failure_kind is None else 2
    assert _finalization_invalid_count(root) == expected


@pytest.mark.parametrize("failure_kind", ["crash", "infrastructure", "cancelled"])
def test_an_agent_stop_cannot_launder_an_interrupted_score(
    tmp_path: Path, failure_kind: str
) -> None:
    """Crash/provider/cancel kinds stay unscored-only even beside a stop.

    The stop is a third deliberate ending, but the scored-terminal kind guard
    is independent of it: a run whose terminal says the harness or the
    environment broke can never report a score, however its journal ends.
    Both the kind guard and the reason-agreement guard fire (the stop claims
    model_failure against a crash-shaped kind), so removing either arm is
    detectable as the count falling to one.
    """

    root = _stopped_bundle(
        tmp_path,
        reason="model_failure",
        failure_kind=failure_kind,
        stop=_stop_event("model_failure"),
    )
    assert _finalization_invalid_count(root) == 2


def test_stop_must_bind_to_the_latest_batchless_observation(tmp_path: Path) -> None:
    """Pins the binding arm: obs-0 is both stale and already batched.

    A stop naming obs-0 would attribute the ending to a state the episode
    had already acted past, and obs-0 carries batch-0 -- the runner's rule is
    that a stopped observation never receives a batch.
    """

    root = _stopped_bundle(
        tmp_path,
        reason="model_failure",
        failure_kind="model",
        stop=_stop_event("model_failure", observation_id="obs-0"),
    )
    assert "receipt_binding_invalid" in codes(root)


def test_nothing_may_execute_after_an_agent_stop(tmp_path: Path) -> None:
    """Pins the post-stop ordering arm with an observation appended after it."""

    root = _stopped_bundle(
        tmp_path, reason="model_failure", failure_kind="model", stop=_stop_event("model_failure")
    )
    _rewrite_events(
        root,
        lambda events: _insert_before(
            events,
            "finalization_start",
            {"kind": "observation", "payload": {"observation_id": "obs-2", "sequence": 2}},
        ),
    )
    assert "event_order_invalid" in codes(root)


def test_a_second_agent_stop_is_rejected(tmp_path: Path) -> None:
    """Pins the single-stop arm: two endings cannot both be the ending."""

    root = _stopped_bundle(
        tmp_path, reason="model_failure", failure_kind="model", stop=_stop_event("model_failure")
    )
    _rewrite_events(
        root,
        lambda events: _insert_before(
            events, "finalization_start", _stop_event("model_failure", stop_id="stop-1")
        ),
    )
    assert "event_order_invalid" in codes(root)
