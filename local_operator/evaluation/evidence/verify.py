"""Read-only verification of evaluation evidence bundles.

Verification reopens every object through directory file descriptors and derives
terminal state from immutable files plus the journal.  It does not import the
writer and deliberately treats ``state.json`` as diagnostic metadata only.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import sys
from collections import Counter
from typing import Any, Iterator, cast

from local_operator.evaluation.evidence.media import (
    MediaValidationError,
    validate_media,
)
from local_operator.evaluation.evidence.models import (
    AbandonmentRecord,
    ActionBatchPayload,
    ArtifactRef,
    BudgetCommitmentPayload,
    CleanupPayload,
    EnvironmentStepPayload,
    EventRecord,
    EvidenceArtifactRef,
    EvidenceCounters,
    EvidenceManifest,
    FinalizationStartPayload,
    LifecycleTransitionPayload,
    ModelRequestPayload,
    ModelResponsePayload,
    ObservationPayload,
    OutcomeSeal,
    PreflightPayload,
    ReconciliationPayload,
    ScoringResultPayload,
    ScoringStartPayload,
    StateMarker,
    UsageCostPayload,
    UserSimulatorExchangePayload,
    VerificationIssue,
    VerificationIssueCode,
    VerificationReport,
)

_ALLOWED_ROOT = {
    ".lock",
    "abandonment.json",
    "artifacts",
    "events.jsonl",
    "manifest.json",
    "outcome.json",
    "state.json",
}
_DIGEST_LENGTH = 64
MAX_JSON_RECORD_BYTES = 16 * 1024 * 1024
MAX_JOURNAL_BYTES = 64 * 1024 * 1024
MAX_EVENTS = 100_000
MAX_ARTIFACT_BYTES = 256 * 1024 * 1024
MAX_PARSED_MEDIA_BYTES = 32 * 1024 * 1024
_READ_CHUNK = 1024 * 1024
_READ_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_NONBLOCK", 0)
)
_DIR_FLAGS = _READ_FLAGS | getattr(os, "O_DIRECTORY", 0)


class _Issues:
    def __init__(self) -> None:
        self.values: list[VerificationIssue] = []

    def error(self, code: VerificationIssueCode, location: str) -> None:
        self.values.append(VerificationIssue(code=code, severity="error", location=location))

    def warning(self, code: VerificationIssueCode, location: str) -> None:
        self.values.append(VerificationIssue(code=code, severity="warning", location=location))


def _safe_regular(fd: int) -> bool:
    info = os.fstat(fd)
    return stat.S_ISREG(info.st_mode) and info.st_nlink == 1


def _safe_directory(fd: int) -> bool:
    info = os.fstat(fd)
    return stat.S_ISDIR(info.st_mode) and info.st_nlink >= 1


def _check_owner_mode(fd: int, location: str, issues: _Issues) -> None:
    info = os.fstat(fd)
    if not hasattr(os, "geteuid") or info.st_uid != os.geteuid():
        issues.error("unsafe_owner", location)
    if stat.S_IMODE(info.st_mode) & 0o022:
        issues.error("unsafe_permissions", location)


def _read_file(
    root_fd: int,
    name: str,
    issues: _Issues,
    unsafe_code: VerificationIssueCode,
    *,
    max_bytes: int = MAX_JSON_RECORD_BYTES,
) -> bytes | None:
    try:
        fd = os.open(name, _READ_FLAGS, dir_fd=root_fd)
    except FileNotFoundError:
        return None
    except OSError:
        issues.error(unsafe_code, name)
        return None
    try:
        if not _safe_regular(fd):
            issues.error(unsafe_code, name)
            return None
        info = os.fstat(fd)
        _check_owner_mode(fd, name, issues)
        if info.st_size > max_bytes:
            issues.error("resource_limit_exceeded", name)
            return None
        data = bytearray()
        while len(data) <= max_bytes:
            chunk = os.read(fd, min(_READ_CHUNK, max_bytes + 1 - len(data)))
            if not chunk:
                return bytes(data)
            data.extend(chunk)
        issues.error("resource_limit_exceeded", name)
        return None
    except OSError:
        issues.error(unsafe_code, name)
        return None
    finally:
        os.close(fd)


def _canonical_model(
    raw: bytes | None,
    model_type: type[Any],
    *,
    missing: VerificationIssueCode,
    noncanonical: VerificationIssueCode,
    invalid: VerificationIssueCode,
    location: str,
    issues: _Issues,
) -> Any | None:
    if raw is None:
        issues.error(missing, location)
        return None
    try:
        return model_type.from_canonical_json(raw)
    except ValueError as error:
        message = str(error)
        issues.error(noncanonical if "canonical" in message else invalid, location)
        return None


def _artifact_refs(value: Any) -> Iterator[ArtifactRef]:
    if isinstance(value, ArtifactRef):
        yield value
        return
    if hasattr(value.__class__, "model_fields"):
        for name in value.__class__.model_fields:
            yield from _artifact_refs(getattr(value, name))
    elif isinstance(value, dict):
        for nested in value.values():
            yield from _artifact_refs(nested)
    elif isinstance(value, (tuple, list)):
        for nested in value:
            yield from _artifact_refs(nested)


def _media_matches(data: bytes, media_type: str) -> bool:
    try:
        validate_media(data, media_type)
    except MediaValidationError:
        return False
    return True


def _read_artifacts(
    root_fd: int,
    references: dict[str, ArtifactRef],
    issues: _Issues,
) -> tuple[EvidenceArtifactRef, ...]:
    try:
        artifacts_fd = os.open("artifacts", _DIR_FLAGS, dir_fd=root_fd)
    except OSError:
        issues.error("artifact_unsafe", "artifacts")
        return ()
    stored: dict[str, EvidenceArtifactRef] = {}
    try:
        if not _safe_directory(artifacts_fd):
            issues.error("artifact_unsafe", "artifacts")
            return ()
        _check_owner_mode(artifacts_fd, "artifacts", issues)
        for name in sorted(os.listdir(artifacts_fd)):
            if len(name) != _DIGEST_LENGTH or any(
                character not in "0123456789abcdef" for character in name
            ):
                issues.error("artifact_name_invalid", f"artifacts/{name}")
                continue
            try:
                fd = os.open(name, _READ_FLAGS, dir_fd=artifacts_fd)
            except OSError:
                issues.error("artifact_unsafe", f"artifacts/{name}")
                continue
            try:
                if not _safe_regular(fd):
                    issues.error("artifact_unsafe", f"artifacts/{name}")
                    continue
                info = os.fstat(fd)
                _check_owner_mode(fd, f"artifacts/{name}", issues)
                if info.st_size > MAX_ARTIFACT_BYTES:
                    issues.error("resource_limit_exceeded", f"artifacts/{name}")
                    continue
                expected = references.get(name)
                parse_media = (
                    expected is not None and expected.media_type != "application/octet-stream"
                )
                if parse_media and info.st_size > MAX_PARSED_MEDIA_BYTES:
                    issues.error("resource_limit_exceeded", f"artifacts/{name}")
                    continue
                digest = hashlib.sha256()
                byte_count = 0
                structured = bytearray() if parse_media else None
                while chunk := os.read(fd, _READ_CHUNK):
                    digest.update(chunk)
                    byte_count += len(chunk)
                    if structured is not None:
                        structured.extend(chunk)
                data = bytes(structured) if structured is not None else None
            except OSError:
                issues.error("artifact_unsafe", f"artifacts/{name}")
                continue
            finally:
                os.close(fd)
            expected = references.get(name)
            if expected is None:
                issues.error("artifact_unreferenced", f"artifacts/{name}")
            if digest.hexdigest() != name:
                issues.error("artifact_hash_mismatch", f"artifacts/{name}")
                continue
            if expected is None:
                continue
            if expected.byte_count != byte_count:
                issues.error("artifact_count_mismatch", f"artifacts/{name}")
            if data is not None and not _media_matches(data, expected.media_type):
                issues.error("artifact_media_mismatch", f"artifacts/{name}")
            stored[name] = EvidenceArtifactRef.model_validate(
                expected.model_dump(mode="json"), strict=True
            )
        for digest in sorted(set(references) - set(stored)):
            issues.error("artifact_missing", f"artifacts/{digest}")
    finally:
        os.close(artifacts_fd)
    return tuple(stored[digest] for digest in sorted(stored))


def _parse_journal(
    root_fd: int, manifest: EvidenceManifest | None, issues: _Issues
) -> tuple[EventRecord, ...]:
    """Parse newline records incrementally so hostile recovery input stays bounded."""

    try:
        fd = os.open("events.jsonl", _READ_FLAGS, dir_fd=root_fd)
    except FileNotFoundError:
        issues.error("journal_missing", "events.jsonl")
        return ()
    except OSError:
        issues.error("unsafe_path", "events.jsonl")
        return ()
    events: list[EventRecord] = []
    previous = manifest.manifest_digest if manifest is not None else "0" * 64
    last_monotonic = -1
    last_wall = -1
    seen_ids: set[str] = set()
    pending = bytearray()
    total = 0
    line_number = 0
    stop = False
    try:
        if not _safe_regular(fd):
            issues.error("unsafe_path", "events.jsonl")
            return ()
        info = os.fstat(fd)
        _check_owner_mode(fd, "events.jsonl", issues)
        if info.st_size > MAX_JOURNAL_BYTES:
            issues.error("resource_limit_exceeded", "events.jsonl")
            return ()
        while not stop and (chunk := os.read(fd, _READ_CHUNK)):
            total += len(chunk)
            if total > MAX_JOURNAL_BYTES:
                issues.error("resource_limit_exceeded", "events.jsonl")
                break
            pending.extend(chunk)
            while (newline := pending.find(b"\n")) >= 0:
                line = bytes(pending[:newline])
                del pending[: newline + 1]
                line_number += 1
                if len(line) > MAX_JSON_RECORD_BYTES or line_number > MAX_EVENTS:
                    issues.error("resource_limit_exceeded", f"events.jsonl:{line_number}")
                    stop = True
                    break
                location = f"events.jsonl:{line_number}"
                try:
                    decoded = json.loads(line.decode("utf-8"))
                    if not isinstance(decoded, dict):
                        raise ValueError("event must be an object")
                    stored_id = decoded.get("event_id")
                    event = EventRecord.model_validate(
                        {**decoded, "event_id": "0" * 64}, strict=True
                    )
                except (UnicodeDecodeError, ValueError):
                    issues.error("event_invalid", location)
                    continue
                canonical_stored = json.dumps(
                    decoded,
                    allow_nan=False,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode("utf-8")
                if canonical_stored != line:
                    issues.error("journal_noncanonical", location)
                if not isinstance(stored_id, str) or stored_id != event.event_id:
                    issues.error("event_hash_mismatch", location)
                if event.sequence != line_number - 1 or stored_id in seen_ids:
                    issues.error("event_sequence_mismatch", location)
                if event.previous_event_sha256 != previous:
                    issues.error("event_chain_mismatch", location)
                if event.monotonic_ns < last_monotonic or event.wall_time_ms < last_wall:
                    issues.error("event_time_reversed", location)
                if isinstance(stored_id, str):
                    seen_ids.add(stored_id)
                    previous = stored_id
                else:
                    previous = event.event_id
                last_monotonic = event.monotonic_ns
                last_wall = event.wall_time_ms
                events.append(event)
            if len(pending) > MAX_JSON_RECORD_BYTES:
                issues.error("resource_limit_exceeded", f"events.jsonl:{line_number + 1}")
                stop = True
        if pending and not stop:
            issues.error("journal_truncated", "events.jsonl")
    except OSError:
        issues.error("unsafe_path", "events.jsonl")
    finally:
        os.close(fd)
    return tuple(events)


def _counters(events: tuple[EventRecord, ...]) -> EvidenceCounters:
    kinds = Counter(event.kind for event in events)
    totals = Counter[str]()
    for event in events:
        if isinstance(event.payload, UsageCostPayload):
            totals.update(
                {
                    "input_tokens": event.payload.input_tokens,
                    "output_tokens": event.payload.output_tokens,
                    "reasoning_tokens": event.payload.reasoning_tokens,
                    "cache_read_tokens": event.payload.cache_read_tokens,
                    "cache_write_tokens": event.payload.cache_write_tokens,
                    "cost_microusd": event.payload.cost_microusd,
                }
            )
    return EvidenceCounters(
        event_count=len(events),
        model_request_count=kinds["model_request"],
        model_response_count=kinds["model_response"],
        action_batch_count=kinds["action_batch"],
        environment_step_count=kinds["environment_step"],
        input_tokens=totals["input_tokens"],
        output_tokens=totals["output_tokens"],
        reasoning_tokens=totals["reasoning_tokens"],
        cache_read_tokens=totals["cache_read_tokens"],
        cache_write_tokens=totals["cache_write_tokens"],
        cost_microusd=totals["cost_microusd"],
    )


def _verify_semantics(
    events: tuple[EventRecord, ...],
    manifest: EvidenceManifest | None,
    marker: StateMarker | None,
    outcome: OutcomeSeal | None,
    abandonment: AbandonmentRecord | None,
    issues: _Issues,
) -> None:
    requests: dict[str, ModelRequestPayload] = {}
    responses: set[str] = set()
    usage: set[str] = set()
    observations: dict[str, ObservationPayload] = {}
    batches: dict[str, ActionBatchPayload] = {}
    observation_batches: dict[str, str] = {}
    steps: set[str] = set()
    batch_steps: dict[str, str] = {}
    expected_output_observation: str | None = None
    terminal_output_observation: str | None = None
    terminal_output_resolved = False
    finish_action_seen = False
    environment_step_seen = False
    last_step_terminal = False
    next_observation_sequence = 0
    exchanges: set[str] = set()
    last_exchange: str | None = None
    lifecycle: dict[str, LifecycleTransitionPayload] = {}
    last_state_id: str | None = None
    finalization_starts: list[FinalizationStartPayload] = []
    starts: list[ScoringStartPayload] = []
    results: list[ScoringResultPayload] = []
    preflights: list[PreflightPayload] = []
    commitments: list[BudgetCommitmentPayload] = []
    reconciliations: list[ReconciliationPayload] = []
    cleanups: list[CleanupPayload] = []
    seen_preflight_receipts: set[str] = set()
    seen_cleanup_receipts: set[str] = set()
    execution_closed = False
    execution_started = False
    durable_finalizing = False
    reconciled = False
    cleaned = False
    terminal_lifecycle_seen = False
    execution_payload_types = (
        ModelRequestPayload,
        ModelResponsePayload,
        UsageCostPayload,
        ObservationPayload,
        ActionBatchPayload,
        EnvironmentStepPayload,
        UserSimulatorExchangePayload,
    )
    allowed_after_finalizing = {
        "scoring_start",
        "scoring_result",
        "reconciliation",
        "lifecycle_transition",
        "cleanup",
        "error",
        "cancel",
    }

    for event in events:
        location = f"events.jsonl:{event.sequence + 1}"
        payload = event.payload
        if execution_closed and event.kind not in allowed_after_finalizing:
            issues.error("finalization_invalid", location)
        terminal_lifecycle = isinstance(payload, LifecycleTransitionPayload) and payload.state in (
            "completed",
            "failed",
            "cancelled",
        )
        running_lifecycle = (
            isinstance(payload, LifecycleTransitionPayload) and payload.state == "running"
        )
        finalizing_lifecycle = (
            isinstance(payload, LifecycleTransitionPayload) and payload.state == "finalizing"
        )
        execution_evidence = isinstance(payload, execution_payload_types) or running_lifecycle
        if event.kind == "preflight" and (preflights or event.sequence != 0):
            issues.error("event_order_invalid", location)
        elif isinstance(payload, BudgetCommitmentPayload) and (
            len(preflights) != 1
            or not preflights[0].passed
            or commitments
            or execution_started
            or durable_finalizing
        ):
            issues.error("event_order_invalid", location)
        elif execution_evidence and (
            len(commitments) != 1 or durable_finalizing or reconciled or cleaned
        ):
            issues.error("event_order_invalid", location)
        elif event.kind == "finalization_start":
            early_preflight_failure = (
                len(preflights) == 1
                and not preflights[0].passed
                and not commitments
                and not execution_started
            )
            if (
                (len(commitments) != 1 and not early_preflight_failure)
                or durable_finalizing
                or reconciled
                or cleaned
            ):
                issues.error("event_order_invalid", location)
        elif event.kind == "scoring_start" and (
            len(commitments) != 1 or not durable_finalizing or starts or reconciled or cleaned
        ):
            issues.error("event_order_invalid", location)
        elif event.kind == "scoring_result" and (
            not durable_finalizing or len(starts) != 1 or results or reconciled or cleaned
        ):
            issues.error("event_order_invalid", location)
        elif finalizing_lifecycle and (
            len(commitments) != 1 or durable_finalizing or reconciled or cleaned
        ):
            issues.error("event_order_invalid", location)
        elif event.kind == "reconciliation" and (
            not durable_finalizing or reconciled or cleaned or (starts and not results)
        ):
            issues.error("event_order_invalid", location)
        elif event.kind == "cleanup" and (not reconciled or cleaned):
            issues.error("event_order_invalid", location)
        elif terminal_lifecycle:
            lifecycle_payload = cast(LifecycleTransitionPayload, payload)
            early_failure = not execution_started and lifecycle_payload.failure_kind in (
                "preflight",
                "infrastructure",
            )
            if terminal_lifecycle_seen or (not early_failure and not cleaned):
                issues.error("lifecycle_invalid", location)
            if early_failure and (
                lifecycle_payload.failure_kind == "preflight"
                and (len(preflights) != 1 or preflights[0].passed)
            ):
                issues.error("lifecycle_invalid", location)
        if execution_evidence:
            execution_started = True
        if event.kind == "finalization_start" or finalizing_lifecycle:
            durable_finalizing = True
        elif event.kind == "reconciliation":
            reconciled = True
        elif event.kind == "cleanup":
            cleaned = True
        elif terminal_lifecycle:
            terminal_lifecycle_seen = True
        if isinstance(payload, ModelRequestPayload):
            if payload.request_id in requests:
                issues.error("receipt_binding_invalid", location)
            requests[payload.request_id] = payload
        elif isinstance(payload, ModelResponsePayload):
            request = requests.get(payload.request_id)
            if request is None or payload.request_id in responses:
                issues.error("receipt_binding_invalid", location)
            elif payload.requested_route != request.requested_route:
                issues.error("route_mismatch", location)
            responses.add(payload.request_id)
        elif isinstance(payload, UsageCostPayload):
            if (
                payload.request_id not in responses
                or payload.request_id in usage
                or payload.request_id not in requests
            ):
                issues.error("receipt_binding_invalid", location)
            usage.add(payload.request_id)
        elif isinstance(payload, ObservationPayload):
            if terminal_output_resolved:
                issues.error("finalization_invalid", location)
            if (
                payload.observation_id in observations
                or payload.sequence != next_observation_sequence
                or (
                    next_observation_sequence > 0
                    and payload.observation_id != expected_output_observation
                )
            ):
                issues.error("receipt_binding_invalid", location)
            observations[payload.observation_id] = payload
            if payload.observation_id == terminal_output_observation:
                terminal_output_resolved = True
            expected_output_observation = None
            next_observation_sequence += 1
        elif isinstance(payload, ActionBatchPayload):
            if terminal_output_observation is not None:
                issues.error("finalization_invalid", location)
            if (
                payload.action_batch_id in batches
                or payload.observation_id not in observations
                or payload.observation_id in observation_batches
            ):
                issues.error("receipt_binding_invalid", location)
            batches[payload.action_batch_id] = payload
            observation_batches[payload.observation_id] = payload.action_batch_id
            if payload.terminal == "finish":
                finish_action_seen = True
        elif isinstance(payload, EnvironmentStepPayload):
            environment_step_seen = True
            if terminal_output_observation is not None:
                issues.error("finalization_invalid", location)
            batch = batches.get(payload.action_batch_id)
            if (
                payload.step_id in steps
                or batch is None
                or payload.action_batch_id in batch_steps
                or expected_output_observation is not None
            ):
                issues.error("receipt_binding_invalid", location)
            elif (
                payload.input_observation_id != batch.observation_id
                or payload.output_observation_id in observations
                or payload.output_observation_id == payload.input_observation_id
            ):
                issues.error("receipt_binding_invalid", location)
            steps.add(payload.step_id)
            batch_steps[payload.action_batch_id] = payload.step_id
            expected_output_observation = payload.output_observation_id
            last_step_terminal = payload.terminated or payload.truncated
            if last_step_terminal:
                terminal_output_observation = payload.output_observation_id
        elif isinstance(payload, UserSimulatorExchangePayload):
            if terminal_output_observation is not None:
                issues.error("finalization_invalid", location)
            if payload.exchange_id in exchanges or payload.previous_exchange_id != last_exchange:
                issues.error("receipt_binding_invalid", location)
            exchanges.add(payload.exchange_id)
            last_exchange = payload.exchange_id
        elif isinstance(payload, LifecycleTransitionPayload):
            if payload.state_id in lifecycle or payload.previous_state_id != last_state_id:
                issues.error("receipt_binding_invalid", location)
            lifecycle[payload.state_id] = payload
            last_state_id = payload.state_id
            if payload.state == "finalizing":
                execution_closed = True
        elif isinstance(payload, PreflightPayload):
            receipts = set(payload.receipt_ids)
            if (
                preflights
                or len(receipts) != len(payload.receipt_ids)
                or receipts & seen_preflight_receipts
                or (manifest is not None and payload.plan_id != manifest.dependency_plan_id)
            ):
                issues.error("receipt_binding_invalid", location)
            preflights.append(payload)
            seen_preflight_receipts |= receipts
        elif isinstance(payload, BudgetCommitmentPayload):
            if (
                execution_started
                or commitments
                or len(preflights) != 1
                or not preflights[0].passed
                or (manifest is not None and payload.budget_id != manifest.budget_id)
            ):
                issues.error("receipt_binding_invalid", location)
            commitments.append(payload)
        elif isinstance(payload, ReconciliationPayload):
            if (
                reconciliations
                or not commitments
                or payload.commitment_id != commitments[0].commitment_id
                or (manifest is not None and payload.budget_id != manifest.budget_id)
            ):
                issues.error("receipt_binding_invalid", location)
            reconciliations.append(payload)
        elif isinstance(payload, CleanupPayload):
            receipts = set(payload.receipt_ids)
            if (
                cleanups
                or len(receipts) != len(payload.receipt_ids)
                or receipts & seen_cleanup_receipts
                or (manifest is not None and payload.cleanup_plan_id != manifest.cleanup_plan_id)
            ):
                issues.error("receipt_binding_invalid", location)
            cleanups.append(payload)
            seen_cleanup_receipts |= receipts
        elif isinstance(payload, FinalizationStartPayload):
            if finalization_starts:
                issues.error("finalization_invalid", location)
            finalization_starts.append(payload)
            if (
                marker is not None
                and marker.state == "finalizing"
                and (
                    payload.finalization_id != marker.finalization_id
                    or payload.intent != marker.intent
                    or payload.scoring_operation_id != marker.scoring_operation_id
                    or payload.intent_digest != marker.intent_digest
                )
            ):
                issues.error("finalization_invalid", location)
        elif isinstance(payload, ScoringStartPayload):
            if starts or results:
                issues.error("score_invalid", location)
            starts.append(payload)
            execution_closed = True
        elif isinstance(payload, ScoringResultPayload):
            if (
                len(starts) != 1
                or results
                or payload.finalization_id != starts[0].finalization_id
                or payload.scoring_operation_id != starts[0].scoring_operation_id
            ):
                issues.error("score_invalid", location)
            results.append(payload)

    completed = [payload for payload in lifecycle.values() if payload.state == "completed"]
    terminal_states = [
        state for state in lifecycle.values() if state.state in {"completed", "failed", "cancelled"}
    ]
    finalization_required = (
        (marker is not None and marker.state == "finalizing")
        or outcome is not None
        or (
            abandonment is not None
            and abandonment.finalization_id is not None
            and not (
                marker is not None
                and marker.state == "abandoned"
                and marker.journal_event_count == len(events)
                and marker.journal_head_sha256
                == (events[-1].event_id if events else abandonment.manifest_digest)
            )
        )
        or bool(terminal_states)
        or bool(starts)
        or bool(results)
    )
    if finalization_required and len(finalization_starts) != 1:
        issues.error("finalization_invalid", "events.jsonl")
    if len(finalization_starts) == 1:
        authority = finalization_starts[0]
        terminal = terminal_states[0] if len(terminal_states) == 1 else None
        if terminal is not None and terminal.finalization_id != authority.finalization_id:
            issues.error("finalization_invalid", "events.jsonl")
        if outcome is not None and outcome.finalization_id != authority.finalization_id:
            issues.error("finalization_invalid", "outcome.json")
        if abandonment is not None and abandonment.finalization_id != authority.finalization_id:
            issues.error("finalization_invalid", "abandonment.json")
        if authority.intent == "score":
            if (
                len(starts) != 1
                or starts[0].finalization_id != authority.finalization_id
                or starts[0].scoring_operation_id != authority.scoring_operation_id
                or starts[0].intent_digest != authority.intent_digest
                or (results and results[0].finalization_id != authority.finalization_id)
                or (results and results[0].scoring_operation_id != authority.scoring_operation_id)
            ):
                issues.error("finalization_invalid", "events.jsonl")
        elif starts or results or (outcome is not None and outcome.result.status == "scored"):
            issues.error("finalization_invalid", "events.jsonl")
    if marker is not None and marker.state == "finalizing":
        if marker.intent == "score":
            if len(starts) != 1:
                issues.error("score_invalid", "state.json")
            elif (
                starts[0].finalization_id != marker.finalization_id
                or starts[0].scoring_operation_id != marker.scoring_operation_id
                or starts[0].intent_digest != marker.intent_digest
            ):
                issues.error("score_invalid", "state.json")
        elif starts or results:
            issues.error("score_invalid", "state.json")
    terminal_location = "outcome.json" if outcome is not None else "events.jsonl"
    if terminal_output_observation is not None and not terminal_states:
        issues.error("finalization_invalid", terminal_location)
    if outcome is not None or terminal_states:
        # Open bundles may legitimately have in-flight requests or batches. A
        # terminal lifecycle snapshot begins sealing authority, so every initiated
        # operation must already have its exact completion receipt.
        if set(requests) != responses or set(requests) != usage:
            issues.error("receipt_binding_invalid", terminal_location)
        stepped_batches = {
            batch_id for batch_id, batch in batches.items() if batch.terminal != "finish"
        }
        if stepped_batches != set(batch_steps) or expected_output_observation is not None:
            issues.error("receipt_binding_invalid", terminal_location)
        if observations and len(observation_batches) not in (
            len(observations),
            len(observations) - 1,
        ):
            issues.error("receipt_binding_invalid", terminal_location)
        if len(terminal_states) != 1:
            issues.error("finalization_invalid", terminal_location)
        if environment_step_seen and not last_step_terminal and not finish_action_seen:
            issues.error("finalization_invalid", terminal_location)
        if terminal_output_observation is not None and not terminal_output_resolved:
            issues.error("receipt_binding_invalid", terminal_location)
    if outcome is not None:
        terminal = completed[0] if len(completed) == 1 else None
        if execution_started and (
            len(preflights) != 1 or not preflights[0].passed or len(commitments) != 1
        ):
            issues.error("receipt_binding_invalid", "outcome.json")
        post_running = outcome.result.reason != "preflight_failure"
        provenance_complete = (
            len(preflights) == 1
            and len(commitments) == 1
            and len(reconciliations) == 1
            and len(cleanups) == 1
        )
        if post_running and not provenance_complete:
            issues.error("receipt_binding_invalid", "outcome.json")
        if outcome.result.reason == "preflight_failure" and (
            len(preflights) != 1
            or preflights[0].passed
            or commitments
            or reconciliations
            or cleanups
        ):
            issues.error("receipt_binding_invalid", "outcome.json")
        if terminal is not None:
            expected_preflight = preflights[0].sealed_preflight_id if len(preflights) == 1 else None
            expected_commitment = commitments[0].commitment_id if len(commitments) == 1 else None
            expected_reconciliation = (
                reconciliations[0].reconciliation_id if len(reconciliations) == 1 else None
            )
            expected_cleanup = cleanups[0].cleanup_result_id if len(cleanups) == 1 else None
            bindings = (
                (terminal.preflight_seal_id, expected_preflight),
                (outcome.preflight_seal_id, expected_preflight),
                (terminal.commitment_id, expected_commitment),
                (outcome.commitment_id, expected_commitment),
                (terminal.reconciliation_id, expected_reconciliation),
                (outcome.reconciliation_id, expected_reconciliation),
                (terminal.cleanup_result_id, expected_cleanup),
                (outcome.cleanup_result_id, expected_cleanup),
            )
            if any(actual != expected for actual, expected in bindings[:2]):
                issues.error("receipt_binding_invalid", "outcome.json")
            if post_running and any(actual != expected for actual, expected in bindings[2:]):
                issues.error("receipt_binding_invalid", "outcome.json")
            if len(reconciliations) == 1 and (
                terminal.reconciliation_reportable != reconciliations[0].reportable
                or cleanups
                and terminal.rescue_required != cleanups[0].rescue_required
                or outcome.counters.cost_microusd != reconciliations[0].provider_cost_microusd
            ):
                issues.error("receipt_binding_invalid", "outcome.json")
        if outcome.event_count != len(events):
            issues.error("counter_mismatch", "outcome.json")
        final_hash = events[-1].event_id if events else outcome.manifest_digest
        if outcome.final_event_sha256 != final_hash:
            issues.error("outcome_mismatch", "outcome.json")
        if outcome.counters != _counters(events):
            issues.error("counter_mismatch", "outcome.json")
        routes_by_key = {
            (
                payload.served_route.provider_id,
                payload.served_route.route_id,
                payload.served_route.model_id,
            ): payload.served_route
            for payload in (event.payload for event in events)
            if isinstance(payload, ModelResponsePayload)
        }
        served = tuple(routes_by_key[key] for key in sorted(routes_by_key))
        if outcome.served_routes != served:
            issues.error("route_mismatch", "outcome.json")
        score = results[0].score if len(results) == 1 else outcome.result
        if (
            len(completed) != 1
            or completed[0].finalization_id != outcome.finalization_id
            or completed[0].score_id != score.score_id
            or outcome.score_id != score.score_id
            or outcome.result != score
            or (starts and starts[0].finalization_id != outcome.finalization_id)
            or completed[0].preflight_seal_id != outcome.preflight_seal_id
            or completed[0].commitment_id != outcome.commitment_id
            or completed[0].reconciliation_id != outcome.reconciliation_id
            or completed[0].cleanup_result_id != outcome.cleanup_result_id
        ):
            issues.error("outcome_mismatch", "outcome.json")
        if outcome.reportable and (
            completed
            and (
                completed[0].reconciliation_reportable is not True
                or completed[0].rescue_required is not False
                or outcome.result.status != "scored"
            )
        ):
            issues.error("outcome_mismatch", "outcome.json")


def verify_bundle(root: os.PathLike[str] | str) -> VerificationReport:
    """Recompute a bundle without trusting writer memory or diagnostic status."""

    issues = _Issues()
    if os.name != "posix" or sys.platform.startswith("win"):
        issues.error("unsupported_platform", ".")
        return VerificationReport(
            valid=False, terminal_state="invalid", issues=tuple(issues.values)
        )
    try:
        root_fd = os.open(os.fspath(root), _DIR_FLAGS)
    except OSError:
        issues.error("root_invalid", ".")
        return VerificationReport(
            valid=False, terminal_state="invalid", issues=tuple(issues.values)
        )
    try:
        if not _safe_directory(root_fd):
            issues.error("root_invalid", ".")
        _check_owner_mode(root_fd, ".", issues)
        try:
            entries = set(os.listdir(root_fd))
        except OSError:
            issues.error("root_invalid", ".")
            entries = set()
        for name in sorted(entries - _ALLOWED_ROOT):
            issues.error("unknown_root_entry", name)

        manifest_raw = _read_file(root_fd, "manifest.json", issues, "unsafe_path")
        manifest = _canonical_model(
            manifest_raw,
            EvidenceManifest,
            missing="manifest_missing",
            noncanonical="manifest_noncanonical",
            invalid="manifest_invalid",
            location="manifest.json",
            issues=issues,
        )
        events = _parse_journal(root_fd, manifest, issues)

        outcome_raw = _read_file(root_fd, "outcome.json", issues, "unsafe_path")
        abandonment_raw = _read_file(root_fd, "abandonment.json", issues, "unsafe_path")
        if outcome_raw is not None and abandonment_raw is not None:
            issues.error("terminal_conflict", ".")
        outcome = None
        abandonment = None
        if outcome_raw is not None:
            outcome = _canonical_model(
                outcome_raw,
                OutcomeSeal,
                missing="outcome_invalid",
                noncanonical="outcome_noncanonical",
                invalid="outcome_invalid",
                location="outcome.json",
                issues=issues,
            )
        if abandonment_raw is not None:
            abandonment = _canonical_model(
                abandonment_raw,
                AbandonmentRecord,
                missing="abandonment_invalid",
                noncanonical="abandonment_noncanonical",
                invalid="abandonment_invalid",
                location="abandonment.json",
                issues=issues,
            )

        references: dict[str, ArtifactRef] = {}
        # The immutable manifest and journal are the only authorities for which
        # artifact bytes belong to a bundle. outcome.json merely asserts that
        # exact derived inventory and can never bless an added file.
        for record in ((manifest,) if manifest is not None else ()) + events:
            for ref in _artifact_refs(record):
                previous = references.get(ref.sha256)
                if previous is not None and previous != ref:
                    issues.error("artifact_count_mismatch", f"artifacts/{ref.sha256}")
                references[ref.sha256] = ref
        artifacts = _read_artifacts(root_fd, references, issues)
        counters = _counters(events)

        final_hash = (
            events[-1].event_id if events else (manifest.manifest_digest if manifest else "0" * 64)
        )
        if outcome is not None and manifest is not None:
            if (
                outcome.bundle_id != manifest.bundle_id
                or outcome.manifest_digest != manifest.manifest_digest
            ):
                issues.error("outcome_mismatch", "outcome.json")
            if outcome.artifacts != artifacts:
                issues.error("outcome_mismatch", "outcome.json")
        marker_raw = _read_file(root_fd, "state.json", issues, "unsafe_path")
        marker = _canonical_model(
            marker_raw,
            StateMarker,
            missing="state_missing",
            noncanonical="state_invalid",
            invalid="state_invalid",
            location="state.json",
            issues=issues,
        )
        if marker is not None and marker.state in ("finalizing", "abandoned") and marker.intent:
            marker_count = marker.journal_event_count
            marker_head = marker.journal_head_sha256
            prefix_head = (
                events[marker_count - 1].event_id
                if marker_count is not None and 0 < marker_count <= len(events)
                else (manifest.manifest_digest if marker_count == 0 and manifest else None)
            )
            start_after_marker = (
                marker_count is not None
                and marker_count < len(events)
                and isinstance(events[marker_count].payload, FinalizationStartPayload)
                and events[marker_count].previous_event_sha256 == marker_head
            )
            marker_matches_journal = (
                marker.bundle_id == (manifest.bundle_id if manifest is not None else None)
                and prefix_head == marker_head
                and marker.finalization_id is not None
                and marker.intent_digest is not None
            )
            try:
                marker_authority = FinalizationStartPayload(
                    finalization_id=marker.finalization_id or "invalid",
                    intent=marker.intent,
                    scoring_operation_id=marker.scoring_operation_id,
                    intent_digest=marker.intent_digest or "0" * 64,
                )
                if (
                    start_after_marker
                    and marker_count is not None
                    and events[marker_count].payload != marker_authority
                ):
                    marker_matches_journal = False
            except ValueError:
                marker_matches_journal = False
            if not marker_matches_journal:
                issues.error("state_invalid", "state.json")
        if outcome is not None and abandonment is not None:
            terminal = "invalid"
        elif outcome is not None:
            terminal = "sealed"
        elif abandonment is not None:
            terminal = "abandoned"
        elif marker is not None and marker.state == "finalizing":
            terminal = "finalizing"
        else:
            terminal = "open"
        if abandonment is not None and manifest is not None:
            starts = [
                event.payload
                for event in events
                if isinstance(event.payload, FinalizationStartPayload)
            ]
            expected_finalization_id = starts[0].finalization_id if len(starts) == 1 else None
            if (
                expected_finalization_id is None
                and abandonment.reason == "ambiguous_finalization"
                and marker is not None
                and marker.state == "abandoned"
                and marker.bundle_id == manifest.bundle_id
                and marker.journal_event_count == len(events)
                and marker.journal_head_sha256 == final_hash
            ):
                expected_finalization_id = marker.finalization_id
            if (
                abandonment.bundle_id != manifest.bundle_id
                or abandonment.manifest_digest != manifest.manifest_digest
                or abandonment.event_count != len(events)
                or abandonment.last_event_sha256 != final_hash
                or abandonment.last_event_sequence != (len(events) - 1 if events else None)
                or abandonment.finalization_id != expected_finalization_id
            ):
                issues.error("abandonment_mismatch", "abandonment.json")
        if marker is not None and marker.state != terminal:
            issues.warning("state_stale", "state.json")
        _verify_semantics(events, manifest, marker, outcome, abandonment, issues)
        return VerificationReport(
            valid=not any(issue.severity == "error" for issue in issues.values),
            terminal_state=terminal,
            issues=tuple(issues.values),
            manifest=manifest,
            events=events,
            artifacts=artifacts,
            outcome=outcome,
            abandonment=abandonment,
            counters=counters,
        )
    finally:
        os.close(root_fd)
