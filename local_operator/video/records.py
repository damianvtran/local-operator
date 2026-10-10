"""The durable ``video_job_v1`` record — a pending video job across restarts.

The video phase's criterion (b): a pending job's completion must land after an
app restart / runtime rollover. The live :class:`AsyncJob` row cannot carry
that — it dies with the process, its roster sidecar is task-only, and a
provider job id / route / cancel target had no durable home anywhere before
this module. The record is a TRANSCRIPT CUSTOM ROW (the ``wake_schedules``
precedent), versioned by name (``video_job_v1``), written at submit and on
every state transition, read with the newest-wins-per-job-id rule.

Shape (frozen, video-contract §1.4; ``None`` fields are omitted on disk):

.. code-block:: jsonc

    {
      "job": "<async job id>", "v": 1, "at": "<iso>",
      "state": "registered|running|completed|failed|cancelled",
      "provider": "radient|fal", "route_label": "Radient|FAL",
      "model": "<alias / provider model id>", "mode": "t2v|i2v|v2v",
      "params": {"prompt": …, "duration_s": 8, "resolution": "720p", "seed": 42},
      "sources": [{"kind": "video|image", "attachment|url": …}],
      "provider_ref": {"request_id": …, "status_path": …, "result_path": …,
                       "cancel_path": …, "base_url": …},
      "result": {"digest": …, "duration_s": 8.0, "content_type": "video/mp4",
                 "model": …, "cost_usd": 1.6, "cost_source": "reported",
                 "billing_basis": "billed"},
      "outcome_note": "…",
      "watcher": "<observation async job id, while one is armed>",
      "error": "…", "error_type": "media_already_completed"
    }

Two rules this module carries, both from the contract:

- **No credential is ever written.** ``provider_ref`` carries the opaque
  provider ids/paths (exactly the ``CancelHandle`` content) and the credential
  CLASS is implied by the route; the credential itself is resolved at call
  time by whoever polls next (the standing rung rule).
- **A write is an APPEND of the job's current full record** — never a
  read-modify-write of an earlier row — so concurrent writers (the tool, the
  runner, the status tool) cannot clobber each other's fields; the reader
  groups rows by ``job`` and takes the newest.

The module is stdlib-only and does not import the transcript's heavy residue:
the disk reader imports :func:`local_operator.session.transcript.read_custom_entries`
lazily inside the function so package import stays cheap for the boot path.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

logger = logging.getLogger(__name__)

__all__ = [
    "VIDEO_CHECK_WAKE_DELAY_MS",
    "VIDEO_JOB_CUSTOM_TYPE",
    "VIDEO_JOB_RECORD_VERSION",
    "VIDEO_JOB_STATES",
    "VIDEO_JOB_TERMINAL_STATES",
    "build_record",
    "check_wake_id",
    "check_wake_message",
    "check_wake_request",
    "newest_records_by_job",
    "read_video_records",
    "record_is_terminal",
]

#: The transcript custom row's versioned name (``supplement_v1`` /
#: ``wake_schedules`` precedent). A v2 shape would ship under a NEW name and
#: this reader would keep answering v1 rows.
VIDEO_JOB_CUSTOM_TYPE = "video_job_v1"
VIDEO_JOB_RECORD_VERSION = 1

#: The frozen state vocabulary (contract §1.1). ``registered`` is the submit
#: receipt; ``running`` once the runner is entered; the rest are terminal.
VIDEO_JOB_STATES = ("registered", "running", "completed", "failed", "cancelled")
VIDEO_JOB_TERMINAL_STATES = frozenset({"completed", "failed", "cancelled"})

#: How far ahead the check-wake is armed. Five minutes balances two costs: a
#: cold session (process died under the job) is raised promptly after the
#: first check, while a live session that is still generating sees at most one
#: benign "still running" turn per interval. The wake is re-ensured by the
#: boot path and by ``video_status`` whenever the job is still non-terminal
#: with no live watcher (contract §1.4), and cancelled the moment the record
#: settles (the runner owns that).
VIDEO_CHECK_WAKE_DELAY_MS = 5 * 60 * 1000


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def build_record(
    *,
    job_id: str,
    state: str,
    provider: str | None = None,
    route_label: str | None = None,
    model: str | None = None,
    mode: str | None = None,
    params: Mapping[str, Any] | None = None,
    sources: list[Mapping[str, Any]] | None = None,
    provider_ref: Mapping[str, Any] | None = None,
    result: Mapping[str, Any] | None = None,
    outcome_note: str | None = None,
    watcher: str | None = None,
    error: str | None = None,
    error_type: str | None = None,
    at: str | None = None,
) -> dict[str, Any]:
    """One record revision. ``None`` fields are omitted, never written as null.

    Every write is the job's full current record from this one builder, so the
    newest row for a job id IS that job's record — omission (rather than
    nulling) keeps a hand-repaired or partial row from claiming absence it did
    not verify.
    """
    if state not in VIDEO_JOB_STATES:
        raise ValueError(f"unknown video job state {state!r}")
    record: dict[str, Any] = {
        "job": job_id,
        "v": VIDEO_JOB_RECORD_VERSION,
        "at": at or _now_iso(),
        "state": state,
    }
    for key, value in (
        ("provider", provider),
        ("route_label", route_label),
        ("model", model),
        ("mode", mode),
        # ``watcher`` journals the id of the observation job watching this
        # record; it is what stops a second ``video_status`` from registering
        # a duplicate watcher for the same generation (each watcher settles
        # through delivery, so N registrations = N completion rows).
        ("watcher", watcher),
        # The frozen cancel-conflict pair (contract §1.6/§4.1): set with
        # ``error_type="media_already_completed"`` when a cancel found the
        # generation already finished, so surfaces render the "already
        # finished" branch instead of inventing an error for it.
        ("error", error),
        ("error_type", error_type),
    ):
        if value is not None:
            record[key] = value
    if params is not None:
        record["params"] = dict(params)
    if sources is not None:
        record["sources"] = [dict(source) for source in sources]
    if provider_ref is not None:
        record["provider_ref"] = dict(provider_ref)
    if result is not None:
        record["result"] = dict(result)
    if outcome_note is not None:
        record["outcome_note"] = outcome_note
    return record


def record_is_terminal(record: Mapping[str, Any]) -> bool:
    return str(record.get("state") or "") in VIDEO_JOB_TERMINAL_STATES


def newest_records_by_job(records: list[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Group record rows by job id, newest wins per id.

    Rows arrive NEWEST-FIRST (`read_custom_entries` walks backwards from the
    end of the transcript), so on an identical ``at`` stamp — the writer's
    clock has second resolution — the row FIRST seen is the later write and
    wins. Getting this backwards silently drops fields a later row added
    (the Q2 drill: a same-second ``registered`` row shadowed the ``running``
    row that carried ``provider_ref``, and the cold follow could never
    poll). ``at`` strings are ISO-8601 with a fixed shape, so lexicographic
    order is chronological order and a strictly greater stamp always wins
    regardless of position. Malformed rows are skipped individually — one
    bad row costs one row, never the map.
    """
    grouped: dict[str, tuple[str, dict[str, Any]]] = {}
    for row in records:
        if not isinstance(row, Mapping):
            continue
        job_id = str(row.get("job") or "")
        if not job_id:
            continue
        at = str(row.get("at") or "")
        current = grouped.get(job_id)
        if current is None or at > current[0]:
            grouped[job_id] = (at, dict(row))
    return {job_id: row for job_id, (_, row) in grouped.items()}


def read_video_records(session_dir: str | Path | None) -> dict[str, dict[str, Any]]:
    """Every job's newest record, from the session transcript's rows.

    A missing/unreadable session dir answers an empty map: callers treat that
    as "no video jobs", the same state as a session that never ran one. The
    row scan is the bounded backward walk in
    :mod:`local_operator.session.transcript` (newest-first, deduped), so the
    cost is the distance from EOF to the oldest row of this type.
    """
    if not session_dir:
        return {}
    try:
        from local_operator.session.transcript import read_custom_entries

        entries = read_custom_entries(session_dir, VIDEO_JOB_CUSTOM_TYPE)
    except Exception:  # noqa: BLE001 — a record read must never take a caller down
        logger.warning("could not read video job records from %s", session_dir, exc_info=True)
        return {}
    records: list[Mapping[str, Any]] = []
    for entry in entries:
        payload = getattr(entry, "payload", None)
        if not isinstance(payload, Mapping):
            continue
        details = payload.get("details")
        if isinstance(details, Mapping):
            records.append(details)
    return newest_records_by_job(records)


def check_wake_id(job_id: str) -> str:
    """The check-wake's stable row id for a job (``video-check-<job id>``).

    Stable so every ensure/cancel path can find the row again without storing
    its id anywhere, and safe for the wake store's filename use (lowercase
    hex job ids and hyphens only).
    """
    return f"video-check-{job_id}"


def check_wake_message(job_id: str) -> str:
    """The wake's self-prompt, naming the ONE callable that answers it.

    The delivery text is the model's instruction *and* the transcript's
    notification: an unknown-job answer from ``video_status`` is honest and
    ends the chain, so this text is safe to fire even after a settle raced it.
    """
    return (
        f"A video generation job ({job_id}) may have settled. "
        f"Check it with the video_status tool (job_id='{job_id}') — the result "
        "is registered as a session attachment if complete."
    )


def check_wake_request(job_id: str) -> dict[str, Any]:
    """The ``wake`` tool request that arms the check-wake (one-shot, quiet).

    ``notify`` stays False: the completion's ``job_result`` row is the
    operator-visible signal the operator asked for, and a re-checking wake that
    pinged per interval would be noise on exactly the long jobs where it fires
    most. One-shot (no ``every``) because every non-terminal observation
    re-arms through the ensure path — a repeating row would multiply the
    sends the ensure path already owns.
    """
    return {
        "message": check_wake_message(job_id),
        "in": f"{VIDEO_CHECK_WAKE_DELAY_MS // 60000}m",
        "notify": False,
    }
