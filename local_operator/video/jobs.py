"""The detached video-job runner: durable records, check-wake, cancel, completion.

This module is the composition the video contract froze (§1.0): the tool
registers an ``AsyncJob`` whose runner is built here; the runner drives the
walk, writes the durable ``video_job_v1`` record at every transition, cancels
the check-wake on settle, runs the bounded best-effort provider cancel on the
two cancellation spellings, and shapes the completion (attachment
registration + §5.1 result text + the review-turn receipt).

**One runner, two modes:**

- :func:`make_video_job_runner` — the submit path: runs the cascade that
  SUBMITS the provider job (the rung's ``on_accept`` hook is how the durable
  ``running`` record gets its ``provider_ref``).
- :func:`make_video_watcher_runner` — the cold-follow path: no submit; polls
  the provider job the record names until terminal, then runs the SAME finish
  path. Registered by ``video_status`` when it finds a live provider job with
  no local watcher, so the successor process keeps the runtime resident (the
  existing job residency term) and the completion still lands as a
  ``job_result`` row through the one delivery path.

**Cancellation** (contract §4): the manager's cancel aborts the job signal AND
cancels the task; both spellings land here, both run ``best_effort_cancel``
under the image lane's existing 5 s budget, both DUPLICATE nothing: the
signal-observed spelling (``VideoGenerationCancelled``) is converted into the
manager's cancelled carrier (``asyncio.CancelledError``) only AFTER the
record write, so the durable record states what actually happened at the
provider and the job row settles ``cancelled`` with no completion delivery —
exactly how a cancelled task/bash job behaves. The task-cancellation spelling
re-raises untouched.

**Everything here is best-effort against failure**: a record write, a wake
cancel or an attachment registration that fails degrades that one facility
(restart durability, a stale wake, a caption note) and never kills a
generation that is already billing, and never turns a completed generation
into an error.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from local_operator.artifacts.naming import media_filename
from local_operator.artifacts.progress import ProgressFn
from local_operator.artifacts.rung import CancelHandle, CancelSupport
from local_operator.artifacts.walk import make_pause
from local_operator.clients._http import APIError
from local_operator.harness.jobs import JobRunFn
from local_operator.imagegen.rungs import best_effort_cancel
from local_operator.video import VideoGenerationCancelled, VideoRoute
from local_operator.video.cascade import (
    VIDEO_RUNG_LABELS,
    VIDEO_RUNG_SPECS,
    run_video_cascade,
)
from local_operator.video.records import build_record, check_wake_id, check_wake_request
from local_operator.video.rungs import PROVIDER_GONE_NOTE, resume_video_observation

logger = logging.getLogger(__name__)

__all__ = [
    "VideoWakeHandle",
    "make_video_job_runner",
    "make_video_watcher_runner",
    "request_naming",
    "video_job_label",
]

#: Bound on the ``outcome_note`` written to the durable record — a record is
#: a status line, not a log; the job's own error_text carries the full text.
_OUTCOME_NOTE_MAX_CHARS = 2000

#: The frozen cancel-token wording map (contract §1.6: "the cancel-token
#: phrases are NOT changed by video"). Byte-identical to the image lane's
#: ``image_tool._CANCEL_PHRASES`` — a test pins the equality — and duplicated
#: rather than imported because ``video/`` must not depend on the tools layer.
#: A token this map does not know is surfaced verbatim: a record must never
#: invent an outcome.
_CANCEL_PHRASES = {
    "cancelled": "cancelled",
    "already_completed": "not cancelled — it had already completed; its result was discarded",
    "not_found": "not cancelled — the provider no longer knows the job",
    "none": "not cancelled — nothing was left to cancel",
    "timeout": "not cancelled — the cancel request timed out",
    "failed": "not cancelled — the cancel request failed",
    "abandoned": "not cancelled — a second stop abandoned the cancel attempt",
}

#: The frozen cancel-conflict sentence (contract §1.6/§4.1); byte-identical to
#: ``image_tool._CANCEL_CONFLICT_SENTENCE`` (same duplication rule as above).
#: Written beside ``error_type: media_already_completed`` so the surfaces
#: render the "already finished" branch instead of inventing an error.
_CANCEL_CONFLICT_SENTENCE = (
    "The generation had already completed when the cancel arrived; its result was discarded."
)

#: The §4.3 honesty tails. ``CANNOT_STOP`` is for when no stop exists to take
#: (a queued-only provider, or nothing was ever cancellable); ``NO_ANSWER``
#: for an attempt that got no terminal answer from an abort-capable provider.
#: The operative half of both is verbatim contract §4.3.
_CANNOT_STOP_CLAUSE = (
    "Provider cannot be stopped; the generation may complete upstream and bill; "
    "the result is not retrievable."
)
_NO_ANSWER_CLAUSE = (
    "The cancel request got no terminal answer; the generation may complete "
    "upstream and bill; the result is not retrievable."
)


def video_job_label(*, mode: str, duration_s: int | None, route_label: str) -> str:
    """The job row's label ("Video generation (i2v, 8s, Radient)").

    The label is chosen at SUBMIT from the first available route; a walk that
    fails forward to another rung keeps it (the label names the job, not the
    winner — the completion text carries the actual model and provider).
    """
    duration = f"{duration_s}s" if duration_s is not None else "default"
    return f"Video generation ({mode}, {duration}, {route_label})"


# ---------------------------------------------------------------------------
# Check-wake ensure / cancel — the durability net's controls
# ---------------------------------------------------------------------------


class VideoWakeHandle:
    """The check-wake operations a runner needs, bound to one scheduler.

    ``scheduler`` is the session's :class:`WakeScheduler` (or ``None`` on a
    host without one — every operation is then a logged no-op, because the
    record still makes the job cold-followable the next time a full session
    opens). All operations are idempotent and never raise.
    """

    def __init__(self, scheduler: Any | None, job_id: str) -> None:
        self._scheduler = scheduler
        self._job_id = job_id

    @property
    def wake_id(self) -> str:
        return check_wake_id(self._job_id)

    async def ensure(self) -> None:
        """Arm the one-shot check-wake if this job has none."""
        scheduler = self._scheduler
        if scheduler is None:
            return
        try:
            from local_operator.harness.wake import build_wake_schedule

            existing = list(scheduler.schedules)
            if any(str(getattr(row, "id", "")) == self.wake_id for row in existing):
                return
            outcome = build_wake_schedule(
                check_wake_request(self._job_id), existing, int(time.time() * 1000)
            )
            if "error" in outcome:
                logger.warning("could not arm video check-wake: %s", outcome.get("error"))
                return
            schedule = outcome["schedule"].model_copy(update={"id": self.wake_id})
            await scheduler.update([*existing, schedule])
        except Exception:  # noqa: BLE001 — the wake is a net, never a hard dependency
            logger.warning("video check-wake arm failed", exc_info=True)

    async def cancel(self) -> None:
        """Remove the check-wake row if present (a settle or a watcher owns delivery)."""
        scheduler = self._scheduler
        if scheduler is None:
            return
        try:
            existing = list(scheduler.schedules)
            kept = [row for row in existing if str(getattr(row, "id", "")) != self.wake_id]
            if len(kept) != len(existing):
                await scheduler.update(kept)
        except Exception:  # noqa: BLE001 — a stale wake self-resolves via video_status
            logger.warning("video check-wake cancel failed", exc_info=True)


# ---------------------------------------------------------------------------
# Completion shaping (attachment registration + §5.1 text + receipt)
# ---------------------------------------------------------------------------


def request_naming(record: Mapping[str, Any]) -> tuple[str | None, str | None]:
    """``(prompt, filename)`` from a durable record's ``params``.

    THE one reader of a record's request facts: the watcher, the cold follow
    and the one-shot refetch all name their assets from it. A record written
    before ``filename`` existed carries only the prompt — the prompt-derived
    fallback is then the honest name, not a missing one.
    """
    raw = record.get("params")
    params = raw if isinstance(raw, Mapping) else {}
    return (
        str(params.get("prompt") or "") or None,
        str(params.get("filename") or "") or None,
    )


def _duration_text(seconds: float | None) -> str:
    if seconds is None:
        return "duration not reported"
    return f"{seconds:.1f}s"


def _price_text(cost_usd: float | None, cost_source: str | None, billing_basis: str | None) -> str:
    """``$1.60 (reported)`` — or the honest absence (design D8).

    ``billing_basis`` is appended only when it says something the source word
    does not (a subscription API-equivalent or an estimate must never read as
    a charge); a plain reported charge keeps the frozen two-word form.
    """
    if cost_usd is None:
        return "price not reported by provider"
    source = cost_source or "reported"
    if billing_basis and billing_basis != "billed":
        return f"${cost_usd:g} ({source}, {billing_basis})"
    return f"${cost_usd:g} ({source})"


def _register_video_assets(
    assets: Sequence[Any], prompt: str | None, filename: str | None = None
) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    """Register every asset once; ``(digests, failed_source_urls, artifacts)``.

    ``artifacts`` is the durable ``details.artifacts`` list (``artifact_details``): the receipt
    mirrors it so the per-session file index can list the video after compaction blanks the
    result text. Empty when nothing registered.

    The REQUEST names the asset (agent-supplied ``filename``, else the prompt),
    never the model template: a Library of generations must not read as one
    repeated name.

    Lazy import of the attachment store (import weight): the store's contract
    is that a refusal is a caption note, never a raise.
    """
    from local_operator.session.attachments import artifact_details, cache_media

    digests: list[str] = []
    failed: list[str] = []
    blocks: list[Any] = []
    for index, asset in enumerate(assets, start=1):
        block = cache_media(
            asset.data,
            asset.content_type or "video/mp4",
            kind="video",
            name=media_filename(
                prompt,
                asset.content_type,
                suggested=filename,
                kind="video",
                index=index,
            ),
            source_url=asset.source_url or None,
            width=asset.width,
            height=asset.height,
            duration_s=asset.duration_s,
        )
        if block is None:
            failed.append(asset.source_url or "(no source url)")
            continue
        if block.attachment:
            digests.append(block.attachment)
            blocks.append(block)
    return digests, failed, artifact_details(blocks)


def finish_video_result(
    *,
    record_job_id: str,
    route: str,
    model: str,
    assets: Sequence[Any],
    cost_usd: float | None,
    cost_source: str | None,
    billing_basis: str | None,
    prompt: str | None = None,
    filename: str | None = None,
) -> tuple[dict[str, Any], str]:
    """The completion's ``(receipt, text)`` — §5.1/§5.3 of the contract.

    One figure computed once (criterion d): the receipt's ``cost_usd``/
    ``cost_source``/``billing_basis`` are EXACTLY the settled outcome's, and
    the text renders that same receipt — the review and the ledger agree by
    construction. Takes the fields, not the outcome TYPE, because both the
    submit path (a walk ``JobOutcome``) and the watcher path (a single-rung
    ``RungResult``) end here.
    """
    digests, failed, artifacts = _register_video_assets(assets, prompt, filename)
    duration = next((asset.duration_s for asset in assets if asset.duration_s is not None), None)
    content_type = next((asset.content_type for asset in assets if asset.content_type), None)
    receipt: dict[str, Any] = {
        "job": record_job_id,
        "digest": digests[0] if digests else None,
        "digests": list(digests),
        # Durable, additive: name/size/dimensions per digest, for the file index.
        "artifacts": artifacts,
        "content_type": content_type or "video/mp4",
        "duration_s": duration,
        "model": model,
        "provider": route,
        "cost_usd": cost_usd,
        "cost_source": cost_source,
        "billing_basis": billing_basis,
    }
    parts = [
        f"Video ready — job {record_job_id}",
        f"model {model or 'unknown'}",
        _duration_text(duration),
        _price_text(cost_usd, cost_source, billing_basis),
    ]
    if digests:
        parts.append(f"attached: digest {', '.join(digests)}")
    else:
        parts.append(
            "the result could not be registered in the session's attachment store"
            + (f" (source: {', '.join(failed)})" if failed else "")
        )
    return receipt, " · ".join(parts)


# ---------------------------------------------------------------------------
# Shared record/cancel plumbing
# ---------------------------------------------------------------------------


def _provider_ref_from_handle(handle: CancelHandle) -> dict[str, Any]:
    """The record's opaque provider refs — exactly the handle's content.

    Hosts are dropped where a path is all the successor needs… except FAL's
    URLs, which are stored whole: the queue's response-carried URLs are the
    authoritative ones (the app-path derivation is the degraded fallback), and
    a host is not a credential. NEVER a credential value — the class is
    implied by the provider and resolved at call time by whoever polls next.
    """
    ref: dict[str, Any] = {}
    if handle.request_id:
        ref["request_id"] = handle.request_id
    if handle.provider == str(VideoRoute.RADIENT):
        if handle.base_url:
            ref["base_url"] = handle.base_url
    else:
        if handle.cancel_url:
            ref["cancel_url"] = handle.cancel_url
        if handle.status_url:
            ref["status_url"] = handle.status_url
        if handle.result_url:
            ref["result_url"] = handle.result_url
    return ref


def _cancel_outcome_note(provider: str | None, receipt_token: str) -> str:
    """The honest cancel note, composed from the frozen wording map.

    Two rules carry the copy (contract §1.6/§4.3): the phrase for every token
    comes from ``_CANCEL_PHRASES`` byte-for-byte — a raw token never renders —
    and the §4.3 honesty tail is appended wherever the provider's copy of the
    job may still be running and billing. The queued-only case includes the
    acknowledged ``cancelled`` token: a queue cancel cannot stop a generation
    already running on such a provider, and the guide promises the record
    says so.
    """
    phrase = _CANCEL_PHRASES.get(receipt_token, receipt_token)
    note = f"Provider job {phrase}."
    support = None
    if provider:
        spec = VIDEO_RUNG_SPECS.get(provider)
        support = spec.cancel_support if spec is not None else None
    if receipt_token in ("none", "failed", "timeout", "abandoned"):
        if support is CancelSupport.QUEUED_ONLY or receipt_token == "none":
            note += " " + _CANNOT_STOP_CLAUSE
        else:
            note += " " + _NO_ANSWER_CLAUSE
    elif receipt_token == "cancelled" and support is CancelSupport.QUEUED_ONLY:
        note += " " + _CANNOT_STOP_CLAUSE
    return note


def _record_ref(handle: CancelHandle, fallback: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """The provider ref for a durable record row.

    The rungs CLEAR the handle the instant the provider job is terminal (the
    cancel semantics: nothing left to cancel), so a row written after settle
    cannot read the refs from it — the Q2 refetch path ("terminal but
    unregistered, fetch once") then can never reach the provider. Prefer the
    handle's live content; else the ref captured at accept time / carried on
    the record being followed.
    """
    ref = _provider_ref_from_handle(handle)
    if ref:
        return ref
    if isinstance(fallback, Mapping) and fallback:
        return dict(fallback)
    return None


async def _journal_best_effort(
    journal: Callable[[dict[str, Any]], Awaitable[None]] | None, record: dict[str, Any]
) -> None:
    """Write one record revision; a failing journal must not kill the job."""
    if journal is None:
        return
    try:
        await journal(record)
    except Exception:  # noqa: BLE001 — durability is best-effort against a live job
        logger.warning("video record write failed", exc_info=True)


def _progress_adapter(
    report_progress: Callable[[str | Mapping[str, Any]], None],
    append_output: Callable[[str], None] | None,
) -> ProgressFn:
    """``emit(text, details)`` → the job's structured details + the peek tail.

    The canonical payload rides ``latest_details["video"]`` (the surfaces'
    detection reads it like any progress update), and the same line lands in
    the job's rolling output so ``jobs op='peek'`` works the way it does for
    every other long job.
    """

    def emit(text: str, details: dict[str, Any]) -> None:
        report_progress({"progress": text, "video": dict(details)})
        if append_output is not None:
            append_output(text + "\n")

    return emit


# ---------------------------------------------------------------------------
# The runners
# ---------------------------------------------------------------------------


def make_video_job_runner(
    *,
    mode: str,
    prompt: str,
    source_kind: str | None = None,
    source_url: str | None = None,
    sources: Sequence[Mapping[str, Any]] | None = None,
    duration_s: int | None = None,
    resolution: str | None = None,
    aspect_ratio: str | None = None,
    audio: bool | None = None,
    seed: int | None = None,
    model: str | None = None,
    filename: str | None = None,
    config_dir: Path | None = None,
    journal: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
    wake_scheduler: Any | None = None,
    append_output: Callable[[str, str], None] | None = None,
) -> JobRunFn:
    """Build the ``AsyncJob`` runner for one submitted video generation.

    The runner's FIRST argument is the manager-assigned job id, and that id is
    the record's key (`build_record(job_id=...)`) and the id in the delivery
    text — one id from receipt to review, because the tool registers this
    runner and persists the submit record under the id ``register`` returns.

    ``wake_scheduler`` is the session's ``WakeScheduler`` (or ``None``): the
    runner builds its OWN :class:`VideoWakeHandle` from the job id it is
    given, which is what lets the tool register before the id exists. The
    tool arms the check-wake right after ``register`` returns; the runner
    only ever CANCELS it, on every settle path.
    """
    params: dict[str, Any] = {"prompt": prompt}
    # The durable record carries the naming request too: the watcher that may
    # outlive this process names the asset from it (an old record without the
    # key falls back to the prompt in ``request_naming``).
    if filename is not None:
        params["filename"] = filename
    for key, value in (
        ("duration_s", duration_s),
        ("resolution", resolution),
        ("aspect_ratio", aspect_ratio),
        ("audio", audio),
        ("seed", seed),
    ):
        if value is not None:
            params[key] = value
    source_rows: list[Mapping[str, Any]] = [dict(row) for row in (sources or [])]

    async def run(
        job_id: str,
        signal: Any,
        report_progress: Callable[[str | Mapping[str, Any]], None],
    ) -> str:
        def _mirror_output(line: str) -> None:
            """Copy one progress line to the job's rolling output.

            ``append_output`` is opt-in (deliberately absent from
            ``JobManagerProtocol``), and this nested def — not a lambda — is
            what lets the None check hold inside the closure.
            """
            if append_output is not None:
                append_output(job_id, line)

        emit = _progress_adapter(
            report_progress,
            _mirror_output if append_output is not None else None,
        )
        wake = VideoWakeHandle(wake_scheduler, job_id)
        handle = CancelHandle()
        # The refs captured the moment the rung accepts: the handle is CLEARED
        # at terminal, so these are what the completed/cancelled rows record
        # (and what the §1.5 refetch later polls).
        accepted_ref: dict[str, Any] = {}

        async def on_accept(accepted: CancelHandle) -> None:
            if not accepted_ref:
                accepted_ref.update(_provider_ref_from_handle(accepted))
            await _journal_best_effort(
                journal,
                build_record(
                    job_id=job_id,
                    state="running",
                    provider=accepted.provider,
                    route_label=VIDEO_RUNG_LABELS.get(str(accepted.provider), None),
                    model=accepted.model,
                    mode=mode,
                    params=params,
                    sources=source_rows or None,
                    provider_ref=_provider_ref_from_handle(accepted) or None,
                ),
            )

        try:
            result = await run_video_cascade(
                prompt=prompt,
                mode=mode,
                source_kind=source_kind,
                source_url=source_url,
                duration_s=duration_s,
                resolution=resolution,
                aspect_ratio=aspect_ratio,
                audio=audio,
                seed=seed,
                model=model,
                config_dir=config_dir,
                signal=signal,
                handle=handle,
                emit=emit,
                on_accept=on_accept,
            )
        except VideoGenerationCancelled:
            # The signal-observed spelling of the operator's stop: bounded
            # provider cancel, the honest record, and then the manager's own
            # cancelled carrier — the row settles ``cancelled`` with no
            # completion delivery, exactly like a cancelled task/bash job.
            await _settle_cancelled(
                handle, journal, wake, job_id, mode, params, source_rows, accepted_ref
            )
            raise asyncio.CancelledError from None
        except asyncio.CancelledError:
            # The task-cancellation spelling (Esc/steer/dispose). The cleanup
            # is bounded and shielded; the original cancellation is re-raised
            # untouched so the manager's accounting sees it.
            await _settle_cancelled(
                handle, journal, wake, job_id, mode, params, source_rows, accepted_ref
            )
            raise
        except Exception as exc:  # noqa: BLE001 — every failure settles the record
            await _journal_best_effort(
                journal,
                build_record(
                    job_id=job_id,
                    state="failed",
                    provider=handle.provider,
                    model=handle.model,
                    mode=mode,
                    params=params,
                    sources=source_rows or None,
                    provider_ref=_record_ref(handle, accepted_ref),
                    outcome_note=str(exc)[:_OUTCOME_NOTE_MAX_CHARS] or None,
                ),
            )
            await wake.cancel()
            raise

        receipt, text = finish_video_result(
            record_job_id=job_id,
            route=str(result.route),
            model=result.model,
            assets=result.assets,
            cost_usd=result.cost_usd,
            cost_source=result.cost_source,
            billing_basis=result.billing_basis,
            prompt=prompt,
            filename=filename,
        )
        report_progress({"video_receipt": receipt})
        await _journal_best_effort(
            journal,
            build_record(
                job_id=job_id,
                state="completed",
                provider=str(result.route),
                route_label=VIDEO_RUNG_LABELS.get(str(result.route), None),
                model=result.model,
                mode=mode,
                params=params,
                sources=source_rows or None,
                provider_ref=_record_ref(handle, accepted_ref),
                result=receipt,
            ),
        )
        await wake.cancel()
        return text

    return run


async def _settle_cancelled(
    handle: CancelHandle,
    journal: Callable[[dict[str, Any]], Awaitable[None]] | None,
    wake: VideoWakeHandle,
    record_job_id: str,
    mode: str,
    params: Mapping[str, Any],
    source_rows: Sequence[Mapping[str, Any]],
    fallback_ref: Mapping[str, Any] | None = None,
) -> None:
    """The cancel settle shared by both cancellation spellings (never raises)."""
    receipt_token = await best_effort_cancel(handle)
    error: str | None = None
    error_type: str | None = None
    if receipt_token == "already_completed":
        # The frozen conflict branch (contract §1.6/§4.1): the cancel arrived
        # too late, and the surfaces render it as "already finished" off this
        # pair — never as an error's ink.
        error = _CANCEL_CONFLICT_SENTENCE
        error_type = "media_already_completed"
    await _journal_best_effort(
        journal,
        build_record(
            job_id=record_job_id,
            state="cancelled",
            provider=handle.provider,
            model=handle.model,
            mode=mode,
            params=params,
            sources=list(source_rows) or None,
            provider_ref=_record_ref(handle, fallback_ref),
            outcome_note=_cancel_outcome_note(handle.provider, receipt_token),
            error=error,
            error_type=error_type,
        ),
    )
    await wake.cancel()


def make_video_watcher_runner(
    *,
    record: Mapping[str, Any],
    config_dir: Path | None = None,
    journal: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
    wake: VideoWakeHandle,
    append_output: Callable[[str, str], None] | None = None,
) -> JobRunFn:
    """Build the cold-follow runner for a provider job that outlived its process.

    ``record`` is the newest ``video_job_v1`` revision; the runner polls the
    provider via :func:`local_operator.video.rungs.resume_video_observation`
    (credential resolved at call time) and then runs the same finish path as
    the submit runner. A provider the successor can no longer find (404)
    settles the record as failed with the honest note — no outcome is invented.
    """
    record_job_id = str(record.get("job") or "")
    mode = str(record.get("mode") or "")
    raw_params = record.get("params")
    params: Mapping[str, Any] = raw_params if isinstance(raw_params, Mapping) else {}
    raw_sources = record.get("sources")
    source_rows: list[Mapping[str, Any]] = raw_sources if isinstance(raw_sources, list) else []
    # The ref the followed record carries: the rows this runner writes must
    # keep it (the handle may already be cleared by the terminal poll), or a
    # later refetch of an unregistered completion has nothing to poll.
    record_ref = (
        record.get("provider_ref") if isinstance(record.get("provider_ref"), Mapping) else None
    )
    request_prompt, request_filename = request_naming(record)

    async def run(
        job_id: str,
        signal: Any,
        report_progress: Callable[[str | Mapping[str, Any]], None],
    ) -> str:
        def _mirror_output(line: str) -> None:
            """Copy one progress line to the job's rolling output (see the
            submit runner's twin: ``append_output`` is opt-in)."""
            if append_output is not None:
                append_output(job_id, line)

        emit = _progress_adapter(
            report_progress,
            _mirror_output if append_output is not None else None,
        )
        handle = CancelHandle()
        pause = make_pause(signal, VideoGenerationCancelled)
        try:
            result = await resume_video_observation(
                record=record,
                handle=handle,
                config_dir=config_dir,
                pause=pause,
                emit=emit,
            )
        except VideoGenerationCancelled:
            await _settle_cancelled(
                handle, journal, wake, record_job_id, mode, params, source_rows, record_ref
            )
            raise asyncio.CancelledError from None
        except asyncio.CancelledError:
            await _settle_cancelled(
                handle, journal, wake, record_job_id, mode, params, source_rows, record_ref
            )
            raise
        except APIError as exc:
            note = str(exc)[:_OUTCOME_NOTE_MAX_CHARS]
            state = "failed"
            if exc.code == "not_found":
                note = PROVIDER_GONE_NOTE
            elif exc.code == "cancelled":
                # The provider's own CANCELLED settles as ``cancelled`` — the
                # SAME word the one-shot ``video_status`` path uses for the
                # identical observation (V5: one observation, one word,
                # wherever it was seen).
                state = "cancelled"
                note = "the provider reported the generation as CANCELLED"
            await _journal_best_effort(
                journal,
                build_record(
                    job_id=record_job_id,
                    state=state,
                    provider=handle.provider or str(record.get("provider") or "") or None,
                    model=handle.model or str(record.get("model") or "") or None,
                    mode=mode,
                    params=params,
                    sources=list(source_rows) or None,
                    provider_ref=_record_ref(handle, record_ref),
                    outcome_note=note or None,
                ),
            )
            await wake.cancel()
            raise
        except Exception as exc:  # noqa: BLE001 — every failure settles the record
            await _journal_best_effort(
                journal,
                build_record(
                    job_id=record_job_id,
                    state="failed",
                    provider=handle.provider or str(record.get("provider") or "") or None,
                    model=handle.model or str(record.get("model") or "") or None,
                    mode=mode,
                    params=params,
                    sources=list(source_rows) or None,
                    provider_ref=_record_ref(handle, record_ref),
                    outcome_note=str(exc)[:_OUTCOME_NOTE_MAX_CHARS] or None,
                ),
            )
            await wake.cancel()
            raise

        receipt, text = finish_video_result(
            record_job_id=record_job_id,
            route=str(record.get("provider") or ""),
            model=result.model or str(record.get("model") or ""),
            assets=result.assets,
            cost_usd=result.cost_usd,
            cost_source=result.cost_source,
            billing_basis=result.billing_basis,
            prompt=request_prompt,
            filename=request_filename,
        )
        report_progress({"video_receipt": receipt})
        await _journal_best_effort(
            journal,
            build_record(
                job_id=record_job_id,
                state="completed",
                provider=str(record.get("provider") or "") or None,
                route_label=VIDEO_RUNG_LABELS.get(str(record.get("provider") or ""), None),
                model=result.model or str(record.get("model") or "") or None,
                mode=mode,
                params=params,
                sources=list(source_rows) or None,
                provider_ref=_record_ref(handle, record_ref),
                result=receipt,
            ),
        )
        await wake.cancel()
        return text

    return run
