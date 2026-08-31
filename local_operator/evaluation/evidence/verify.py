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
from typing import Any, Iterator

from local_operator.evaluation.evidence.media import (
    MediaValidationError,
    validate_media,
)
from local_operator.evaluation.evidence.models import (
    AbandonmentRecord,
    ActionBatchPayload,
    ArtifactRef,
    CleanupPayload,
    EnvironmentStepPayload,
    EventRecord,
    EvidenceArtifactRef,
    EvidenceCounters,
    EvidenceManifest,
    LifecycleTransitionPayload,
    ModelRequestPayload,
    ModelResponsePayload,
    ObservationPayload,
    OutcomeSeal,
    PreflightPayload,
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
    root_fd: int, name: str, issues: _Issues, unsafe_code: VerificationIssueCode
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
        _check_owner_mode(fd, name, issues)
        chunks: list[bytes] = []
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                return b"".join(chunks)
            chunks.append(chunk)
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
                _check_owner_mode(fd, f"artifacts/{name}", issues)
                chunks: list[bytes] = []
                while chunk := os.read(fd, 1024 * 1024):
                    chunks.append(chunk)
                data = b"".join(chunks)
            except OSError:
                issues.error("artifact_unsafe", f"artifacts/{name}")
                continue
            finally:
                os.close(fd)
            expected = references.get(name)
            if expected is None:
                issues.error("artifact_unreferenced", f"artifacts/{name}")
            if hashlib.sha256(data).hexdigest() != name:
                issues.error("artifact_hash_mismatch", f"artifacts/{name}")
                continue
            if expected is None:
                continue
            if expected.byte_count != len(data):
                issues.error("artifact_count_mismatch", f"artifacts/{name}")
            if not _media_matches(data, expected.media_type):
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
    raw: bytes | None, manifest: EvidenceManifest | None, issues: _Issues
) -> tuple[EventRecord, ...]:
    if raw is None:
        issues.error("journal_missing", "events.jsonl")
        return ()
    if raw and not raw.endswith(b"\n"):
        issues.error("journal_truncated", "events.jsonl")
        return ()
    events: list[EventRecord] = []
    previous = manifest.manifest_digest if manifest is not None else "0" * 64
    last_monotonic = -1
    last_wall = -1
    seen_ids: set[str] = set()
    for index, line in enumerate(raw.splitlines(), start=1):
        location = f"events.jsonl:{index}"
        try:
            decoded = json.loads(line.decode("utf-8"))
            if not isinstance(decoded, dict):
                raise ValueError("event must be an object")
            stored_id = decoded.get("event_id")
            event = EventRecord.model_validate({**decoded, "event_id": "0" * 64}, strict=True)
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
        if event.sequence != index - 1 or stored_id in seen_ids:
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
    marker: StateMarker | None,
    outcome: OutcomeSeal | None,
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
    next_observation_sequence = 0
    exchanges: set[str] = set()
    last_exchange: str | None = None
    lifecycle: dict[str, LifecycleTransitionPayload] = {}
    last_state_id: str | None = None
    starts: list[ScoringStartPayload] = []
    results: list[ScoringResultPayload] = []
    seen_preflight_receipts: set[str] = set()
    seen_cleanup_receipts: set[str] = set()
    execution_closed = False
    allowed_after_finalizing = {
        "scoring_result",
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
            if (
                payload.observation_id in observations
                or payload.sequence != next_observation_sequence
            ):
                issues.error("receipt_binding_invalid", location)
            observations[payload.observation_id] = payload
            next_observation_sequence += 1
        elif isinstance(payload, ActionBatchPayload):
            if (
                payload.action_batch_id in batches
                or payload.observation_id not in observations
                or payload.observation_id in observation_batches
            ):
                issues.error("receipt_binding_invalid", location)
            batches[payload.action_batch_id] = payload
            observation_batches[payload.observation_id] = payload.action_batch_id
        elif isinstance(payload, EnvironmentStepPayload):
            batch = batches.get(payload.action_batch_id)
            if payload.step_id in steps or batch is None or payload.action_batch_id in batch_steps:
                issues.error("receipt_binding_invalid", location)
            elif (
                payload.observation_id is not None
                and payload.observation_id != batch.observation_id
            ):
                issues.error("receipt_binding_invalid", location)
            steps.add(payload.step_id)
            batch_steps[payload.action_batch_id] = payload.step_id
        elif isinstance(payload, UserSimulatorExchangePayload):
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
            if len(receipts) != len(payload.receipt_ids) or receipts & seen_preflight_receipts:
                issues.error("receipt_binding_invalid", location)
            seen_preflight_receipts |= receipts
        elif isinstance(payload, CleanupPayload):
            receipts = set(payload.receipt_ids)
            if len(receipts) != len(payload.receipt_ids) or receipts & seen_cleanup_receipts:
                issues.error("receipt_binding_invalid", location)
            seen_cleanup_receipts |= receipts
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
    terminal_states = [
        state for state in lifecycle.values() if state.state in {"completed", "failed", "cancelled"}
    ]
    terminal_location = "outcome.json" if outcome is not None else "events.jsonl"
    if outcome is not None or terminal_states:
        # Open bundles may legitimately have in-flight requests or batches. A
        # terminal lifecycle snapshot begins sealing authority, so every initiated
        # operation must already have its exact completion receipt.
        if set(requests) != responses or set(requests) != usage:
            issues.error("receipt_binding_invalid", terminal_location)
        if set(batches) != set(batch_steps):
            issues.error("receipt_binding_invalid", terminal_location)
        if observations and len(observation_batches) not in (
            len(observations),
            len(observations) - 1,
        ):
            issues.error("receipt_binding_invalid", terminal_location)
        if len(terminal_states) != 1:
            issues.error("finalization_invalid", terminal_location)
    if outcome is not None:
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
        journal_raw = _read_file(root_fd, "events.jsonl", issues, "unsafe_path")
        events = _parse_journal(journal_raw, manifest, issues)

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
        if abandonment is not None and manifest is not None:
            if (
                abandonment.bundle_id != manifest.bundle_id
                or abandonment.manifest_digest != manifest.manifest_digest
                or abandonment.event_count != len(events)
                or abandonment.last_event_sha256 != final_hash
                or abandonment.last_event_sequence != (len(events) - 1 if events else None)
            ):
                issues.error("abandonment_mismatch", "abandonment.json")

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
        if marker is not None and marker.state != terminal:
            issues.warning("state_stale", "state.json")
        _verify_semantics(events, marker, outcome, issues)
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
