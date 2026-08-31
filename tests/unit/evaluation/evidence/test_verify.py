"""Independent verifier tamper and confinement tests."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from local_operator.evaluation.evidence.models import (
    ActionBatchPayload,
    BudgetCommitmentPayload,
    CancelPayload,
    CleanupPayload,
    EnvironmentStepPayload,
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

    script = r"""
import json, os, resource, tempfile
from pathlib import Path
from local_operator.evaluation.evidence.store import EvidenceWriter
from local_operator.evaluation.evidence.verify import verify_bundle
from local_operator.evaluation.receipts import RedactionSet
from tests.unit.evaluation.evidence.test_models import manifest
root=Path(tempfile.mkdtemp())/'bundle'
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
