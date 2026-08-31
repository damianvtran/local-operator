"""Independent verifier tamper and confinement tests."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from local_operator.evaluation.evidence.models import (
    ActionBatchPayload,
    CancelPayload,
    EnvironmentStepPayload,
    ModelRequestPayload,
    ModelResponsePayload,
    ObservationPayload,
    UsageCostPayload,
)
from local_operator.evaluation.evidence.store import (
    EvidenceBundleInvalid,
    EvidenceWriter,
)
from local_operator.evaluation.evidence.verify import verify_bundle
from local_operator.evaluation.receipts import RedactionSet
from tests.unit.evaluation.evidence.test_models import ARTIFACT, DIGEST, ROUTE, manifest


def _bundle(tmp_path: Path, *, events: int = 2, artifact: bool = False) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    root = tmp_path / "bundle"
    with EvidenceWriter.create(root, manifest(), RedactionSet.from_resolved_values(())) as writer:
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

    data = json.loads(original[0])
    data["payload"]["diagnostic_code"] = "tampered"
    original[0] = json.dumps(data, separators=(",", ":"), sort_keys=True).encode() + b"\n"
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
    from local_operator.evaluation.evidence.models import EventRecord

    lines[1] = (
        EventRecord.model_validate({**data, "event_id": "0" * 64}, strict=True).to_canonical_json()
        + b"\n"
    )
    (root / "events.jsonl").write_bytes(b"".join(lines))
    assert "event_chain_mismatch" in codes(root)


def test_semantic_graph_rejects_duplicate_and_out_of_order_receipts(
    tmp_path: Path,
) -> None:
    root = tmp_path / "bundle"
    with EvidenceWriter.create(root, manifest(), RedactionSet.from_resolved_values(())) as writer:
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
                observation_id="different",
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
                    observation_id="observation",
                    terminated=False,
                    truncated=False,
                ),
                monotonic_ns=4 if step_id == "step-1" else 5,
                wall_time_ms=4 if step_id == "step-1" else 5,
            )
    report = verify_bundle(root)
    assert "receipt_binding_invalid" in {issue.code for issue in report.issues}


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
    data = json.loads(raw[0])
    data["payload"]["artifacts"][0]["byte_count"] = 99
    from local_operator.evaluation.evidence.models import EventRecord

    (root / "events.jsonl").write_bytes(
        EventRecord.model_validate({**data, "event_id": "0" * 64}, strict=True).to_canonical_json()
        + b"\n"
        + raw[1]
        + b"\n"
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
