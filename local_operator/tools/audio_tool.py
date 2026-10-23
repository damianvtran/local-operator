"""The ``generate_voiceover`` / ``generate_sound_effect`` agent tools.

**Why TWO tools** (the frozen schema, 2026-10-17): a voiceover and a sound
effect share a provider cascade but not a request shape — one takes text plus
a voice, the other a prompt plus a MANDATORY duration — and a merged tool
would put every field of both into one schema the context-budget ratchet bills
on every request. The server contract itself exposes two routes, so one tool
per route keeps the tool the model reaches for aligned with the request shape
it carries.

**Why write tier** (the image lane's D8 rule): a generation spends real money
on the platform's balance or a third-party account, and its product is
billable. Default mode is "ask", so the approval prompt is the ONE place the
spend is shown before it happens — :func:`_describe_voiceover_approval` and
:func:`_describe_sound_effect_approval` carry the route and the estimate.

**Why createIf-gated PER TOOL**: the gates are sync and socket-free
(``audiogen.availability``) and deliberately differ — the frozen contract has
no OpenAI sound-effect path, so an OpenAI-only machine gets the voiceover tool
and NOT a sound-effect tool whose every call could only end in "no provider
for this request shape" (the D9 footprint rule, applied per tool).

**Single result shape** (frozen contract): ``ToolResult(content=[caption, ONE
AttachmentContent(kind="audio", duration_s=…)], details={…})`` — the caption
FIRST and self-sufficient, the bytes registered exactly once through
``session.attachments.cache_media`` (which also writes the waveform peaks
sidecar), provenance in ``details`` including ``details.artifacts`` for the
per-session file index. A ``cache_media`` refusal degrades to a caption note,
never a raise; a refusal for EVERY asset is an honest error instead, because a
caption claiming an attachment that does not exist is worse than a failure.

**Cost** (frozen rules): Radient's figure is what the server stated — only
``cost.amount_usd`` is authoritative, ``basis: "quote"`` reads ``estimated``
and ``"billed"`` reads ``billed``. A BYOK call gets no figure from the vendor
at all, so the harness estimates from the ONE central rate table
(``audiogen.pricing``) and the ledger row keeps that ``estimated`` label —
never written as a charge. Every outcome hands one record to the ``audio``
spend channel (``session.channel_spend``).

**Cancellation has two spellings and they are not the same path** (the image
design §4.2): the loop's task cancellation (Esc/steer) propagates untouched
after a bounded best-effort server cancel, while the no-cancellation race —
the abort signal observed between polls — returns a clean receipt. The
receipt is HONEST about the frozen contract: a Radient cancel says what the
server's ``cancellation.vendor_stopped`` said (``false`` is the expected
answer once a vendor is running — never a claimed guaranteed stop), and a
BYOK cancel says the harness stopped waiting while the user's provider may
still bill for work already delivered.

**Why nothing registers with ``group_reaper``** (the image lane's D7 note,
unchanged): this tool spawns no process and no orphanable group — its work is
in-process HTTP cancelled through the standard loom machinery.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable

from pydantic import ValidationError

from local_operator.artifacts.naming import media_filename, media_stem
from local_operator.artifacts.progress import ProgressFn
from local_operator.audiogen import AudioRoute
from local_operator.audiogen import availability as audio_availability
from local_operator.audiogen import pricing
from local_operator.audiogen import rungs as audio_rungs
from local_operator.audiogen.cascade import (
    AUDIO_RUNG_DIRECTIONS,
    AUDIO_RUNG_LABELS,
    AudioGenerationCancelled,
    AudioGenerationUnavailable,
    run_audio_cascade,
)
from local_operator.audiogen.models import (
    AudioJobSpec,
    GenerateSoundEffectParams,
    GenerateVoiceoverParams,
)
from local_operator.harness.types import (
    AbortSignal,
    AgentTool,
    AgentToolUpdate,
    InvalidToolArgumentsError,
    TextContent,
    ToolContext,
    ToolResult,
)
from local_operator.tools.builtin import _guard

logger = logging.getLogger(__name__)

#: Wire descriptions (the image lane's 160-char discipline: the classification
#: roster reads ONLY the first sentence, so the capability statement fits there
#: and the pointers to the deeper docs follow it; providers live in
#: ``guide://sound-generation``, which costs no schema).
_VOICEOVER_DESCRIPTION = (
    "Generate a voiceover (spoken audio) via the configured provider; the result is attached "
    "for the user. Docs: `tool://generate_voiceover`; playbook: `guide://sound-generation`."
)
_SOUND_EFFECT_DESCRIPTION = (
    "Generate a sound effect or named batch via the configured provider; "
    "audio is attached for the user. "
    "Docs: `tool://generate_sound_effect`; playbook: `guide://sound-generation`."
)


def build_generate_voiceover_tool(context: ToolContext) -> AgentTool | None:
    """createIf builder: exists only where a voiceover route is reachable.

    The gate is SYNC and socket-free (``audiogen.availability``: sqlite rows,
    the encrypted store, the environment) so no session-build path opens a
    network connection, and every probe failure reads as "not available".
    Sessions built before a login land the tools at the NEXT session — the
    gate is a snapshot by design.
    """
    del context  # gating is on the machine's credentials, not session state
    from local_operator.paths import config_dir

    if not audio_availability.voiceover_reachable(config_dir()):
        return None
    return AgentTool(
        name="generate_voiceover",
        label="Generate voiceover",
        description=_VOICEOVER_DESCRIPTION,
        parameters=GenerateVoiceoverParams.model_json_schema(),
        # write tier: the generation spends real money and its product is
        # billable (see the module docstring).
        approval_tier="write",
        # Generations are independent; two calls can run beside each other.
        concurrency="shared",
        # A generation should yield to a redirect the way `bash` does:
        # steering cancels the wait, the CancelledError handler best-effort
        # cancels the server job, and the loop pairs its own synthetic receipt.
        interruptible=True,
        describe_approval=_describe_voiceover_approval,
        execute=execute_generate_voiceover,
    )


def build_generate_sound_effect_tool(context: ToolContext) -> AgentTool | None:
    """createIf builder: sound effects have no OpenAI path (see the module docstring)."""
    del context
    from local_operator.paths import config_dir

    if not audio_availability.sound_effect_reachable(config_dir()):
        return None
    return AgentTool(
        name="generate_sound_effect",
        label="Generate sound effect",
        description=_SOUND_EFFECT_DESCRIPTION,
        parameters=GenerateSoundEffectParams.model_json_schema(),
        approval_tier="write",
        concurrency="shared",
        interruptible=True,
        describe_approval=_describe_sound_effect_approval,
        refuse_args=_refuse_sound_effect_args,
        execute=execute_generate_sound_effect,
    )


# ---------------------------------------------------------------------------
# Approval text
# ---------------------------------------------------------------------------


def _preferred_route(direction: str) -> AudioRoute | None:
    """The cascade's first available route CAPABLE of ``direction``.

    A read of the same sync probes the gates use; ``None`` when nothing
    answers (a describer must never fail a call — it is read by the approval
    renderer before validation).
    """
    from local_operator.paths import config_dir

    cfg = config_dir()
    try:
        probes = (
            (AudioRoute.RADIENT, audio_availability.radient_available(cfg)),
            (AudioRoute.ELEVENLABS, bool(audio_availability.elevenlabs_key(cfg))),
            (AudioRoute.OPENAI, bool(audio_availability.openai_audio_key(cfg))),
        )
        for route, available in probes:
            if available and direction in AUDIO_RUNG_DIRECTIONS[route]:
                return route
    except Exception:  # noqa: BLE001 - a describer must never fail a call
        logger.debug("audio availability read failed for approval text", exc_info=True)
    return None


def _spend_phrase(route: AudioRoute | None, estimate: pricing.PriceEstimate | None) -> str:
    """The one spend sentence for the consent card, from what is knowable.

    A priceable BYOK call gets its estimated figure; a Radient call is quoted
    by the server when it runs (a client-side number would be a second source
    of truth for money the server states); nothing available is the generic
    paid-call wording.
    """
    if estimate is not None:
        return (
            f"estimated ${estimate.amount_usd:.4f} at published rates "
            f"({estimate.price_version}); the billed figure is the provider's"
        )
    if route == AudioRoute.RADIENT:
        return "Radient quotes the price when it runs — a paid call on your account"
    return "a paid provider call on your account"


def _describe_voiceover_approval(args: dict[str, Any], cwd: str) -> str:
    """What the approval prompt says: the request size, the route, the spend."""
    del cwd  # the spend is account-level; no path decides it
    text = args.get("text")
    if not isinstance(text, str) or not text.strip():
        return "[unresolvable] generate_voiceover without text"
    chars = len(text)
    route = _preferred_route("voiceover")
    label = AUDIO_RUNG_LABELS.get(str(route), "the configured provider")
    model = args.get("model")
    model = model if isinstance(model, str) and model else None
    estimate = pricing.voiceover_estimate(str(route), model, chars) if route else None
    return (
        f"Generate a voiceover ({chars:,} chars) via {label} — "
        f"{_spend_phrase(route, estimate)}."
    )


def _refuse_sound_effect_args(args: dict[str, Any], offered: frozenset[str]) -> str | None:
    """Reject malformed paid envelopes before the existing write-tier approval."""
    del offered
    try:
        GenerateSoundEffectParams.model_validate(args)
    except ValidationError as exc:
        return f"Invalid generate_sound_effect arguments: {exc}"
    return None


def _describe_sound_effect_approval(args: dict[str, Any], cwd: str) -> str:
    del cwd
    if args.get("sounds") is not None:
        try:
            params = GenerateSoundEffectParams.model_validate(args)
        except ValidationError as exc:
            return f"[unresolvable] {exc}"
        route = _preferred_route("sound_effect")
        label = AUDIO_RUNG_LABELS.get(str(route), "the configured provider")
        # Radient's published underlying SFX rate is an APPROVAL estimate only;
        # each server receipt remains authoritative for billing and the ledger.
        estimates = [
            pricing.sound_effect_estimate(
                "elevenlabs" if route == AudioRoute.RADIENT else str(route),
                None if route == AudioRoute.RADIENT else params.model,
                item.duration_s,
            )
            for item in params.sounds or []
        ]
        known = [estimate for estimate in estimates if estimate is not None]
        total = (
            f"estimated total ${sum(e.amount_usd for e in known):.4f} "
            if len(known) == len(estimates)
            else f"estimated known subtotal ${sum(e.amount_usd for e in known):.4f}; "
            f"{len(estimates) - len(known)} item(s) unpriced "
        )
        total += f"({pricing.PRICE_VERSION}); per-item billed figures are authoritative"
        noun = "sound effect" if len(estimates) == 1 else "sound effects"
        return f"Generate {len(estimates)} {noun} via {label} — {total}."
    prompt = args.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        return "[unresolvable] generate_sound_effect without a prompt"
    duration = args.get("duration_s")
    duration = (
        float(duration)
        if isinstance(duration, int | float) and not isinstance(duration, bool)
        else None
    )
    route = _preferred_route("sound_effect")
    label = AUDIO_RUNG_LABELS.get(str(route), "the configured provider")
    model = args.get("model")
    model = model if isinstance(model, str) and model else None
    estimate = (
        pricing.sound_effect_estimate(str(route), model, duration) if route and duration else None
    )
    length = f"{duration:g}s" if duration else "requested length"
    return f"Generate a {length} sound effect via {label} — {_spend_phrase(route, estimate)}."


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


def _progress_emitter(
    on_update: Callable[[AgentToolUpdate], None] | None,
    active: dict[str, str | None],
) -> ProgressFn | None:
    """Adapt the rungs' ``emit(text, details)`` onto ``AgentToolUpdate``.

    Also the tool's one channel for learning which route is ACTIVE: every
    progress update carries ``provider``, so the cancellation path can tell a
    direct BYOK call (no server-side job; the vendor may still bill) from a
    Radient job (a cancel the server can answer) without guessing. Updates are
    LIVE-ONLY by design; the emitter never raises — a progress line is
    presentation, and the terminal ``cancelling``/``cancelled`` lines emit
    from inside cancellation handlers.
    """
    if on_update is None:
        return None

    def emit(text: str, details: dict[str, Any]) -> None:
        try:
            provider = details.get("provider")
            if isinstance(provider, str) and provider:
                active["route"] = provider
            on_update(AgentToolUpdate(content=[TextContent(text=text)], details=dict(details)))
        except Exception:  # noqa: BLE001 - progress is presentation, never control flow
            logger.debug("audio progress emitter raised; continuing", exc_info=True)

    return emit


def _emit_cancel_stage(
    progress: ProgressFn | None,
    spec: AudioJobSpec,
    handle: audio_rungs.CancelHandle,
    stage: str,
    text: str,
) -> None:
    """A cancel-phase update (``cancelling`` → ``cancelled``)."""
    if progress is None:
        return
    progress(
        text,
        audio_rungs.progress_details(
            tool=spec.tool,
            stage=stage,
            provider=str(handle.provider) if handle.provider is not None else None,
            model=handle.model,
        ),
    )


def _cancel_handle_details(handle: audio_rungs.CancelHandle) -> dict[str, Any] | None:
    """``{provider, request_id, model?}`` — what a best-effort cancel used."""
    if handle.provider is None or handle.request_id is None:
        return None
    return {
        "provider": str(handle.provider),
        "request_id": handle.request_id,
        "model": handle.model,
    }


def _attempt_list(attempts: tuple[Any, ...]) -> list[dict[str, Any]]:
    """Attempt records as JSON-safe dicts (the artifact lane's ``details`` contract)."""
    return [
        {
            "route": str(attempt.route),
            "outcome": attempt.outcome,
            "reason_class": attempt.reason_class,
            "message": attempt.message,
        }
        for attempt in attempts
    ]


def _unavailable_text(exc: AudioGenerationUnavailable, handle: audio_rungs.CancelHandle) -> str:
    """The error sentence, plus the request id when a server job exists.

    A submitted Radient job that timed out (or failed after submit) may still
    complete server-side; naming its id is what lets the operator follow up —
    and no artifact is attached either way (frozen rule).
    """
    text = str(exc).strip()
    if not text:
        # ``JobUnavailable.resolution`` is typed ``object | None`` on the
        # generic layer (a kind rides its own report there); this lane's
        # exception always carries an ``AudioRouteResolution``, so the
        # narrow read is a ``getattr`` rather than a cast.
        reason = getattr(exc.resolution, "reason", None)
        text = str(reason).strip() if reason else ""
    if not text:
        text = "Audio generation is unavailable."
    if handle.request_id:
        text += (
            f"\nA Radient job was submitted (request {handle.request_id}); "
            "no artifact is attached."
        )
    return text


def _spawn_settlement(
    context: ToolContext | None,
    spec: AudioJobSpec,
    handle: audio_rungs.CancelHandle,
    receipt: audio_rungs.AudioCancelReceipt,
) -> bool:
    """Start the bounded settle poll for a stop that cannot be guaranteed.

    The contract's line: a QUEUED job stops for certain (``vendor_stopped:
    true``) and is never charged; a RUNNING one keeps going and is charged if
    it produces audio, and a timeout/failed cancel tells us nothing. For those
    cases the session owns one bounded observation (``audiogen.settle``) so the
    late charge lands in the ledger and a paid artifact does not vanish.

    Returns True only when the task was actually started, because the receipt's
    "Checking for settlement." sentence must never describe a check that is not
    happening. A host with no session (a library caller, a test) returns False
    and the receipt stays exactly as honest, just shorter.
    """
    if receipt.outcome in {"none", "not_found"}:
        return False
    if receipt.outcome == "cancelled" and receipt.vendor_stopped is True:
        return False
    if handle.provider != AudioRoute.RADIENT:
        return False
    if not handle.request_id or not handle.base_url or handle.credential is None:
        return False
    spawn = getattr(context, "spawn_background", None) if context is not None else None
    if not callable(spawn):
        return False
    from local_operator.audiogen.settle import settle_cancelled_audio
    from local_operator.session.channel_spend import new_record_id

    payload = {
        "tool": spec.tool,
        "request_id": handle.request_id,
        "base_url": handle.base_url,
        "credential": handle.credential.get_secret_value(),
        "provider": str(handle.provider),
        "model": str(handle.model or ""),
        # The SAME deterministic id the cancel row used, so a settled charge
        # supersedes that row instead of appearing as a second one.
        "record_id": new_record_id("audio", handle.request_id),
        "rev": handle.cost_rev,
        # The same count and unit the cancel row carried, so the settled
        # revision means the same thing as the row it supersedes.
        "units": _spend_units(spec)[0],
        "unit": _spend_units(spec)[1],
        # The REQUEST names a late-settled artifact too: the settle task runs
        # after this turn is gone, so it must carry what the asset IS, not
        # depend on the turn's state. The tool and model here are provenance.
        "prompt": spec.text,
    }
    journal = context.journal_audio_settle if context is not None else None
    record_spend = context.record_channel_spend if context is not None else None
    try:
        spawn(
            settle_cancelled_audio(
                payload,
                record_spend=record_spend,
                journal=journal,
            )
        )
        return True
    except Exception:  # noqa: BLE001 - a failed spawn must not fail the cancel
        logger.debug("audio settlement task could not be started", exc_info=True)
        return False


async def _cancel_receipt(
    spec: AudioJobSpec,
    handle: audio_rungs.CancelHandle,
    progress: ProgressFn | None,
    active: dict[str, str | None],
    context: ToolContext | None = None,
) -> tuple[str, dict[str, Any]]:
    """The honest receipt for a stopped generation, and its ``details``.

    ``byok`` is derived from what the progress stream observed: a direct
    provider call has no server-side job for ``best_effort_cancel`` to reach,
    and its receipt says the harness stopped waiting while the provider may
    still bill — never that something was cancelled server-side.
    """
    _emit_cancel_stage(progress, spec, handle, "cancelling", "Stopping the generation…")
    receipt = await audio_rungs.best_effort_cancel(handle)
    byok = handle.provider is None and active.get("route") in ("elevenlabs", "openai")
    if receipt.outcome == "cancelled" and receipt.vendor_stopped is True:
        _record_audio_channel(
            context,
            spec=spec,
            route=str(handle.provider or "radient"),
            model=str(handle.model or ""),
            amount_usd=0.0,
            cost_source="reported",
            billing_basis="billed",
            cost_provenance=None,
            status="cancelled",
            request_id=str(handle.request_id or ""),
            rev=1,
            detail="stopped before the vendor was called; guaranteed no charge",
        )
    settling = _spawn_settlement(context, spec, handle, receipt)
    text = audio_rungs.cancel_receipt_text(receipt, byok=byok, settling=settling)
    _emit_cancel_stage(progress, spec, handle, "cancelled", text)
    details: dict[str, Any] = {
        "cancel_handle": _cancel_handle_details(handle),
        "stage": "cancelled",
        "settling": settling,
        "vendor_stopped": receipt.vendor_stopped,
    }
    if receipt.reason:
        details["cancel_reason"] = receipt.reason
    # The server's pending-settlement facts travel VERBATIM. A surface renders
    # "awaiting settlement" only from explicit fields — ``settled: null``,
    # ``cost_usd: null``, ``cost.basis: "quote"`` — so a body the harness
    # drops is a state no client can show. Nothing is invented: ``cost`` is
    # the server's own quote object, and ``settled``/``cost_usd`` are written
    # as null ONLY for a quote-basis pending body (the contract's own
    # spelling of "not settled" / "no settled figure"); a body that states
    # either value is forwarded as stated. A cancel body with no cost object
    # (BYOK, 409 already-completed) adds neither key.
    server = receipt.server_payload or {}
    cost = server.get("cost")
    cost_basis = cost.get("basis") if isinstance(cost, dict) else None
    for key in ("cancellation", "cost", "note"):
        if key in server:
            details[key] = server[key]
    if "settled" in server:
        details["settled"] = server["settled"]
    elif cost_basis == "quote":
        details["settled"] = None
    if "cost_usd" in server:
        details["cost_usd"] = server["cost_usd"]
    elif cost_basis == "quote":
        details["cost_usd"] = None
    if receipt.outcome == "already_completed":
        # The cancel conflict is NOT a failure: the surfaces' frozen branch
        # renders it as "already finished" off this pair (the media
        # vocabulary is shared with the image lane's cancel path).
        details["error"] = text
        details["error_type"] = "media_already_completed"
    return text, details


def _spend_units(spec: AudioJobSpec) -> tuple[float, str]:
    """The channel record's count and unit: chars for speech, whole seconds for effects.

    Sound effects count in whole started seconds — the same
    :func:`pricing.sound_effect_units` rule the estimate prices with — so a
    row's count and its estimated amount describe one quantity. A fractional
    count here was how 0.5 s reached the ledger as ``units: 0.5`` beside a
    figure priced for one second. Radient rows keep the server's own count
    when the settled payload states one (``units=`` on the record); this is
    only the default.
    """
    if spec.direction == "voiceover":
        return float(len(spec.text)), "chars"
    return float(pricing.sound_effect_units(float(spec.duration_s or 0.0))), "seconds"


def _record_audio_channel(
    context: ToolContext | None,
    *,
    spec: AudioJobSpec,
    route: str,
    model: str,
    amount_usd: Any,
    cost_source: Any,
    billing_basis: Any,
    cost_provenance: Any,
    status: str,
    request_id: str,
    detail: str = "",
    rev: int = 0,
    usage_record_id: str = "",
    units: float | None = None,
    unit: str | None = None,
) -> None:
    """Hand ONE generation's money to the ``audio`` channel, best-effort.

    Consumes the outcome's cost labels through the ledger's ONE mapping
    (``map_audio_cost_labels``) so the live emission and the transcript
    backfill cannot disagree about what a row means; an estimate keeps its
    ``estimated`` label and is never written as a charge. Never raises and
    never blocks a generation: the callback is one bounded enqueue on the
    session's side, and a host with no ledger leaves the callback ``None`` —
    the documented "channels not tracked here" case, skipped QUIETLY.
    """
    callback = getattr(context, "record_channel_spend", None) if context is not None else None
    if not callable(callback):
        return
    try:
        from local_operator.session.channel_spend import (
            ChannelSpendRecord,
            map_audio_cost_labels,
            new_record_id,
            now_ms,
            usd_to_micro,
        )

        basis, source, price_version = map_audio_cost_labels(
            route=route,
            cost_source=cost_source,
            billing_basis=billing_basis,
            price_version=cost_provenance,
            has_amount=amount_usd is not None,
        )
        default_units, default_unit = _spend_units(spec)
        callback(
            ChannelSpendRecord(
                record_id=new_record_id("audio", request_id),
                rev=rev,
                ts_ms=now_ms(),
                channel="audio",
                provider=route,
                model=model,
                units=default_units if units is None else units,
                unit=default_unit if unit is None else unit,
                amount_micro=usd_to_micro(amount_usd),
                billing_basis=basis,
                cost_source=source,
                price_version=price_version,
                status=status,
                request_id=request_id,
                usage_record_id=usage_record_id,
                detail=detail,
            )
        )
    except Exception:  # noqa: BLE001 — a lost spend row is not a failed generation
        logger.debug("audio channel-spend emission failed", exc_info=True)


async def _execute_audio(
    tool_call_id: str,
    spec: AudioJobSpec,
    *,
    signal: AbortSignal | None,
    on_update: Callable[[AgentToolUpdate], None] | None,
    context: ToolContext | None,
    filename_hint: str | None = None,
    cancel_as_result: bool = False,
) -> ToolResult:
    """Resolve → cascade → the frozen attachment result, for either direction.

    One body for both tools (they differ only in their params shape and their
    spend unit); ``spec.tool`` names the emitting tool on every result,
    update and channel row.
    """
    from local_operator.paths import config_dir

    handle = audio_rungs.CancelHandle()
    # Financial observations precede delivery: a failed CDN read must never
    # erase the accepted quote or the server's authoritative terminal charge.
    recorded_cost: dict[str, Any] | None = None
    recorded_rev = -1

    def observe_cost(payload: dict[str, Any], terminal: bool) -> None:
        nonlocal recorded_cost, recorded_rev
        cost = audio_rungs._cost_from_payload(payload)
        if cost is None:
            return  # retain the accepted figure until an authoritative revision arrives
        # Terminal observations may repeat money but omit the optional receipt.
        # Normalize identity before comparing so partial reads neither erase it
        # nor create a spurious ledger revision; a new receipt still revises.
        cost["usage_record_id"] = str(
            cost.get("usage_record_id")
            or payload.get("usage_record_id")
            or (recorded_cost or {}).get("usage_record_id")
            or ""
        )
        if cost == recorded_cost and (
            not terminal or recorded_rev >= 1 or cost["basis"] != "billed"
        ):
            return  # identical terminal status/result reads are idempotent
        # The first terminal figure is rev 1. A genuinely revised terminal
        # receipt needs a higher rev because the fold ignores equal revisions.
        recorded_rev = max(recorded_rev + 1, 1 if terminal else 0)
        handle.cost_rev = recorded_rev
        recorded_cost = cost
        _record_audio_channel(
            context,
            spec=spec,
            route=str(handle.provider or "radient"),
            model=str(handle.model or ""),
            amount_usd=cost["amount_usd"],
            cost_source="reported",
            billing_basis=cost["basis"],
            cost_provenance=cost.get("price_version"),
            status="failed" if payload.get("error_type") else "ok",
            request_id=str(handle.request_id or ""),
            rev=recorded_rev,
            usage_record_id=cost["usage_record_id"],
            units=cost.get("units"),
            unit=cost.get("unit"),
            detail="terminal server charge" if terminal else "accepted server quote",
        )

    handle.observe_cost = observe_cost
    active: dict[str, str | None] = {"route": None}
    progress = _progress_emitter(on_update, active)
    try:
        outcome = await run_audio_cascade(
            spec=spec,
            config_dir=config_dir(),
            signal=signal,
            handle=handle,
            emit=progress,
        )
    except AudioGenerationCancelled:
        text, details = await _cancel_receipt(spec, handle, progress, active, context)
        if handle.cost is None:
            _record_audio_channel(
                context,
                spec=spec,
                route=str(handle.provider or active.get("route") or ""),
                model=str(handle.model or ""),
                amount_usd=None,
                cost_source=None,
                billing_basis=None,
                cost_provenance=None,
                status="cancelled",
                request_id=str(handle.request_id or ""),
                detail=(
                    "cancelled after the generation completed; its result was discarded"
                    if details.get("error_type") == "media_already_completed"
                    else "cancelled before settlement was observed; the job may still be charged"
                ),
            )
        if cancel_as_result:
            details["cost"] = handle.cost
        return ToolResult(
            tool_call_id=tool_call_id,
            tool_name=spec.tool,
            is_error=True,
            content=[TextContent(text=text)],
            details=details,
        )
    except asyncio.CancelledError:
        # The loop's cancellation. The best-effort cancel is bounded (5 s),
        # never raises and never masks the cancellation — a second stop
        # abandons it. The terminal updates go through the guarded emitter
        # for the same reason.
        text, details = await _cancel_receipt(spec, handle, progress, active, context)
        if handle.cost is None:
            _record_audio_channel(
                context,
                spec=spec,
                route=str(handle.provider or active.get("route") or ""),
                model=str(handle.model or ""),
                amount_usd=None,
                cost_source=None,
                billing_basis=None,
                cost_provenance=None,
                status="cancelled",
                request_id=str(handle.request_id or ""),
                detail="cancelled by the loop; the job may still be charged",
            )
        if cancel_as_result:
            details["cost"] = handle.cost
            return ToolResult(
                tool_call_id=tool_call_id,
                tool_name=spec.tool,
                is_error=True,
                content=[TextContent(text=text)],
                details=details,
            )
        raise
    except AudioGenerationUnavailable as exc:
        if handle.cost is None:
            _record_audio_channel(
                context,
                spec=spec,
                route=str(exc.attempts[-1].route if exc.attempts else ""),
                model=str(handle.model or ""),
                amount_usd=None,
                cost_source=None,
                billing_basis=None,
                cost_provenance=None,
                status="failed",
                request_id=str(handle.request_id or ""),
                # Documented presumption (the image lane's §4.1): an unreported
                # failure with no charge claim is audited as failed/None and does
                # NOT degrade the knowledge.
                detail=(
                    "accepted job; the server has not reported its charge"
                    if handle.request_id
                    else "walk exhausted with no charged figure reported; presumed unbilled"
                ),
            )
        text = _unavailable_text(exc, handle)
        details = {
            "attempts": _attempt_list(exc.attempts),
            "error": str(exc).strip() or text,
            # The last attempt's classification is the walk's final word;
            # ``attempts`` beside it carries the full story. The provider's own
            # structured code wins over the retryability class when it sent
            # one (the contract's ``audio_rate_limited`` on a ``transient``
            # failure is exactly that case).
            "error_type": (
                (exc.attempts[-1].error_type or exc.attempts[-1].reason_class)
                if exc.attempts
                else None
            ),
        }
        if handle.request_id:
            details["request_id"] = handle.request_id
        if cancel_as_result:
            details["cost"] = handle.cost
            details["http_status"] = exc.attempts[-1].status_code if exc.attempts else None
        return ToolResult(
            tool_call_id=tool_call_id,
            tool_name=spec.tool,
            is_error=True,
            content=[TextContent(text=text)],
            details=details,
        )

    result = _generated_result(tool_call_id, spec, outcome, handle, filename_hint=filename_hint)
    if cancel_as_result and result.details is not None:
        result.details["cost"] = handle.cost or {
            "amount_usd": outcome.cost_usd,
            "basis": outcome.billing_basis,
            "units": _spend_units(spec)[0],
            "unit": _spend_units(spec)[1],
        }
    if handle.cost is None:
        _record_audio_channel(
            context,
            spec=spec,
            route=str(outcome.route),
            model=str(outcome.model or ""),
            amount_usd=outcome.cost_usd,
            cost_source=outcome.cost_source,
            billing_basis=outcome.billing_basis,
            cost_provenance=outcome.cost_provenance,
            status="ok",
            request_id=str(outcome.generation_id or ""),
            detail=(
                ""
                if outcome.cost_usd is not None
                else "no per-call figure was stated for this generation"
            ),
        )
    return result


@_guard("generate_voiceover")
async def execute_generate_voiceover(
    tool_call_id: str,
    args: dict[str, Any],
    signal: AbortSignal | None = None,
    on_update: Callable[[AgentToolUpdate], None] | None = None,
    context: ToolContext | None = None,
) -> ToolResult:
    """Validate → cascade → the frozen attachment result."""
    try:
        params = GenerateVoiceoverParams.model_validate(args)
    except ValidationError as exc:
        raise InvalidToolArgumentsError(str(exc)) from exc
    spec = AudioJobSpec(
        tool="generate_voiceover",
        direction="voiceover",
        text=params.text,
        voice=params.voice,
        tone=params.tone,
        expressiveness=params.expressiveness,
        pace=params.pace,
        language=params.language,
        accent=params.accent,
        instructions=params.instructions,
        model=params.model,
    )
    return await _execute_audio(
        tool_call_id,
        spec,
        signal=signal,
        on_update=on_update,
        context=context,
        filename_hint=params.filename,
    )


@_guard("generate_sound_effect")
async def execute_generate_sound_effect(
    tool_call_id: str,
    args: dict[str, Any],
    signal: AbortSignal | None = None,
    on_update: Callable[[AgentToolUpdate], None] | None = None,
    context: ToolContext | None = None,
) -> ToolResult:
    """Validate → cascade → the frozen attachment result."""
    try:
        params = GenerateSoundEffectParams.model_validate(args)
    except ValidationError as exc:
        raise InvalidToolArgumentsError(str(exc)) from exc
    if params.sounds is not None:
        return await _execute_sound_batch(
            tool_call_id, params, signal=signal, on_update=on_update, context=context
        )
    spec = AudioJobSpec(
        tool="generate_sound_effect",
        direction="sound_effect",
        text=params.prompt or "",
        duration_s=params.duration_s,
        loop=params.loop,
        prompt_influence=params.prompt_influence,
        model=params.model,
    )
    return await _execute_audio(
        tool_call_id,
        spec,
        signal=signal,
        on_update=on_update,
        context=context,
        filename_hint=params.filename,
    )


# Leave headroom inside the tool's 240s bound for two best-effort stop receipts.
# A slow 50-item batch is partial delivery, never 50 independently renewed budgets.
SOUND_BATCH_TIMEOUT_S = 225.0


async def _execute_sound_batch(
    tool_call_id: str,
    params: GenerateSoundEffectParams,
    *,
    signal: AbortSignal | None,
    on_update: Callable[[AgentToolUpdate], None] | None,
    context: ToolContext | None,
) -> ToolResult:
    """Two workers consume a queue; cancellation never starts queued paid work.

    Reuse the single-item cascade, financial observer and attachment registration
    rather than creating a second provider or storage path. Each worker owns its
    own cancel handle; one rejected item cannot cancel successful siblings.
    """
    from local_operator.harness.types import AttachmentContent

    sounds = params.sounds or []
    # STEMS, not filenames: the provider's bytes decide the extension, and a
    # provisional ".mp3" is a guess the final row would then contradict. The
    # registered artifact's own name supplies the real extension.
    names = [media_stem(item.prompt, suggested=item.name, kind="audio") for item in sounds]
    hints = list(names)
    results: list[ToolResult | None] = [None] * len(sounds)
    started: set[int] = set()
    queue = iter(enumerate(sounds))
    stop = AbortSignal.any_of(signal) if signal is not None else AbortSignal()

    def progress(index: int, update: AgentToolUpdate) -> None:
        if on_update is None:
            return
        prefix = f"{index + 1}/{len(sounds)} {names[index]}"
        try:
            on_update(
                AgentToolUpdate(
                    content=[
                        TextContent(
                            text=f"{prefix}: "
                            + " ".join(
                                block.text
                                for block in update.content
                                if isinstance(block, TextContent)
                            )
                        )
                    ],
                    details={
                        **(update.details or {}),
                        "name": names[index],
                        "item": index + 1,
                        "total": len(sounds),
                    },
                )
            )
        except Exception:  # noqa: BLE001 - presentation cannot interrupt paid work
            logger.debug("batch progress callback failed", exc_info=True)

    async def worker() -> None:
        for index, item in queue:
            if stop.aborted or (signal is not None and signal.aborted):
                return
            started.add(index)
            progress(index, AgentToolUpdate(content=[TextContent(text="Starting generation…")]))
            spec = AudioJobSpec(
                tool="generate_sound_effect",
                direction="sound_effect",
                text=item.prompt,
                duration_s=item.duration_s,
                batch_submission=True,
                loop=params.loop,
                prompt_influence=params.prompt_influence,
                model=params.model,
            )
            try:
                result = await _execute_audio(
                    tool_call_id,
                    spec,
                    signal=stop,
                    on_update=lambda update: progress(index, update),
                    context=context,
                    filename_hint=hints[index],
                    cancel_as_result=True,
                )
            except asyncio.CancelledError:
                # A second cancellation can interrupt even the best-effort stop.
                # Do not claim a vendor stop or zero charge in that case.
                result = ToolResult(
                    tool_call_id=tool_call_id,
                    tool_name="generate_sound_effect",
                    is_error=True,
                    content=[
                        TextContent(text="Stopped waiting; provider work may still be billed.")
                    ],
                    details={"stage": "cancelled", "vendor_stopped": None},
                )
            except Exception as exc:  # noqa: BLE001 - isolate a failed sibling
                logger.exception("batch sound effect failed")
                result = ToolResult(
                    tool_call_id=tool_call_id,
                    tool_name="generate_sound_effect",
                    is_error=True,
                    content=[TextContent(text=str(exc))],
                    details={"error": str(exc)},
                )
            results[index] = result
            progress(
                index,
                AgentToolUpdate(
                    content=[TextContent(text="Failed" if result.is_error else "Completed")],
                    details=result.details,
                ),
            )

    workers = [asyncio.create_task(worker()) for _ in range(min(2, len(sounds)))]
    reason: str | None = None
    try:
        _, pending = await asyncio.wait(workers, timeout=SOUND_BATCH_TIMEOUT_S)
        if pending:
            reason = "Batch time budget reached"
    except asyncio.CancelledError:
        reason = "Batch stop requested"
    finally:
        if reason is not None:
            stop.abort(reason)
            for task in workers:
                if not task.done():
                    task.cancel()
        # Only TWO worker tasks, not one scheduled task per paid item. Awaiting
        # them preserves stop receipts and prevents orphaned in-flight work.
        await asyncio.gather(*workers, return_exceptions=True)
        stop.cancel()

    artifacts: list[dict[str, Any]] = []
    attachments: list[AttachmentContent] = []
    completed = failed = cancelled = 0
    captions: list[str] = []
    for index, (item, result) in enumerate(zip(sounds, results, strict=True)):
        details = dict(result.details or {}) if result is not None else {}
        blocks = [b for b in result.content if isinstance(b, AttachmentContent)] if result else []
        was_cancelled = result is None or details.get("stage") == "cancelled"
        status = "completed" if blocks else "cancelled" if was_cancelled else "failed"
        completed += status == "completed"
        failed += status == "failed"
        cancelled += status == "cancelled"
        attached = (details.get("artifacts") or [{}])[0]
        amount = details.get("cost_usd")
        basis = details.get("billing_basis")
        cost = details.get("cost") or {"amount_usd": amount, "basis": basis}
        if index not in started:
            # A queue item that never made a provider call is the one case
            # where zero is KNOWABLE, rather than an unknown financial outcome.
            cost = {"amount_usd": 0.0, "basis": "billed", "units": 0, "unit": "seconds"}
        error = None
        if not blocks:
            error = details.get("error") or (
                "Never started; no provider call"
                if index not in started
                else (
                    " ".join(b.text for b in result.content if isinstance(b, TextContent))
                    if result
                    else "Stop requested; provider outcome unknown"
                )
            )
        row = {
            **{key: value for key, value in details.items() if key != "artifacts"},
            **attached,
            # The artifact's OWN registered name wins: it carries the provider's
            # real extension. ``names`` is the pre-generation provisional.
            "name": attached.get("name") or names[index],
            "status": status,
            "started": index in started,
            "request_id": details.get("request_id")
            or (details.get("cancel_handle") or {}).get("request_id"),
            "digest": attached.get("digest"),
            "duration_s": attached.get("duration_s", item.duration_s),
            "format": (details.get("audio") or {}).get("format")
            or {
                "audio/mpeg": "mp3",
                "audio/mp3": "mp3",
                "audio/wav": "wav",
                "audio/x-wav": "wav",
                "audio/ogg": "ogg",
                "audio/flac": "flac",
            }.get(str(attached.get("content_type") or "").split(";", 1)[0].strip()),
            "cost_usd": cost.get("amount_usd"),
            "cost": cost,
            "units": cost.get("units") if index in started else 0,
            "unit": cost.get("unit"),
            "error": error,
        }
        artifacts.append(row)
        attachments.extend(blocks)
        captions.append(
            f"{index + 1}/{len(sounds)} {row['name']}: {status}" + (f" — {error}" if error else "")
        )
    known_costs = [row["cost_usd"] for row in artifacts if row["cost_usd"] is not None]
    total = sum(known_costs) if known_costs else None
    # Completeness and basis answer different questions: an unpriced failure
    # must not relabel successful, server-billed siblings as estimates.
    known_rows = [row for row in artifacts if row["cost_usd"] is not None]
    total_basis = (
        (
            "billed"
            if all(row["cost"].get("basis") == "billed" for row in known_rows)
            else "estimated"
        )
        if known_rows
        else None
    )
    caption = (
        f"Sound effects: {completed} completed, {failed} failed, {cancelled} cancelled/stopped. "
        + (
            f"Known total ${total:g} ({total_basis}); unknown or late charges are not included. "
            if total is not None
            else "Total cost not yet known. "
        )
        + (
            f"{reason}. In-flight stop is best-effort, not a guaranteed vendor cancellation. "
            if reason
            else ""
        )
        + "\n"
        + "\n".join(captions)
    )
    return ToolResult(
        tool_call_id=tool_call_id,
        tool_name="generate_sound_effect",
        is_error=not attachments,
        content=[TextContent(text=caption), *attachments],
        details={
            "artifacts": artifacts,
            "cost_usd": total,
            "cost_complete": len(known_costs) == len(artifacts),
            "cost": {"amount_usd": total, "basis": total_basis},
            "completed": completed,
            "failed": failed,
            "cancelled": cancelled,
        },
    )


# ---------------------------------------------------------------------------
# Result shaping
# ---------------------------------------------------------------------------


def _duration_text(duration_s: float | None) -> str:
    """``"12.4s"`` — the provider's own duration fact, or nothing."""
    if duration_s is None:
        return ""
    return f"{duration_s:g}s"


def _cost_caption(amount_usd: float | None, basis: Any, source: Any) -> str:
    """The caption's cost sentence, from the outcome's own labels.

    Three honest shapes: a settled charge, an estimate (naming what it is an
    estimate OF — a server quote vs. published rates), and an unstated figure.
    """
    if amount_usd is None:
        return " No per-call cost was stated; the provider's own account shows the settled charge."
    amount = f"${amount_usd:g}"
    if str(basis) == "billed":
        return f" Cost {amount} (billed)."
    if str(source) == "reported":
        return f" Cost {amount} (estimated — the server's quote, not a settled charge)."
    return (
        f" Cost {amount} (estimated from published rates; "
        "the provider reports no per-call figure)."
    )


def _voicing_note(notes: Any) -> str:
    """One sentence naming what was NOT applied exactly, across both shapes.

    Radient reports ``{applied, ignored, clamped}``; the BYOK mapping records
    ``{applied, degraded}`` in the vendored map's token form. Anything
    non-empty on the not-applied side is surfaced; silence means everything
    applied.
    """
    if not isinstance(notes, dict):
        return ""
    parts: list[str] = []
    for key, label in (("ignored", "ignored"), ("clamped", "clamped"), ("degraded", "adjusted")):
        values = notes.get(key)
        if isinstance(values, list) and values:
            parts.append(f"{label}: {', '.join(str(value) for value in values)}")
    if not parts:
        return ""
    return " Voicing — " + "; ".join(parts) + "."


def _generated_result(
    tool_call_id: str,
    spec: AudioJobSpec,
    outcome: Any,
    handle: audio_rungs.CancelHandle,
    *,
    filename_hint: str | None = None,
) -> ToolResult:
    """Register the audio and shape the frozen result (caption FIRST)."""
    from local_operator.session.attachments import artifact_details, cache_media

    blocks: list[Any] = []
    failed: list[str] = []
    digests: list[str] = []
    facts: dict[str, Any] | None = None
    voicing: dict[str, Any] | None = None
    for index, asset in enumerate(outcome.assets, start=1):
        try:
            block = cache_media(
                asset.data,
                asset.content_type or "application/octet-stream",
                kind="audio",
                # The REQUEST names the artifact (the spoken text for a
                # voiceover, the prompt for a sound), never the model template:
                # two different requests must not land on one identical name.
                name=media_filename(
                    spec.text,
                    asset.content_type,
                    suggested=filename_hint,
                    kind="audio",
                    index=index,
                ),
                source_url=asset.source_url or None,
                duration_s=asset.duration_s,
            )
        except Exception:  # noqa: BLE001 - the server charge already exists
            logger.debug("audio artifact registration failed", exc_info=True)
            block = None
        if block is None:
            failed.append(asset.source_url or "(no source url)")
            continue
        blocks.append(block)
        if block.attachment:
            digests.append(block.attachment)
        if facts is None:
            # The provider's own audio facts (Radient's block; absent-safe on
            # BYOK legs, whose vendors state fewer facts).
            facts = {
                "format": getattr(asset, "format", None),
                "sample_rate_hz": getattr(asset, "sample_rate_hz", None),
                "channels": getattr(asset, "channels", None),
                "bitrate_kbps": getattr(asset, "bitrate_kbps", None),
                "size_bytes": len(asset.data),
                "duration_s": asset.duration_s,
                "sha256": getattr(asset, "sha256", None),
                "provider_model_id": getattr(asset, "provider_model_id", None),
                "expires_at": getattr(asset, "expires_at", None),
            }
            facts = {key: value for key, value in facts.items() if value is not None}
            candidate = getattr(asset, "voicing_notes", None)
            if isinstance(candidate, dict):
                voicing = candidate

    details: dict[str, Any] = {
        "provider": str(outcome.route),
        "model": outcome.model,
        "prompt": outcome.prompt,
        "request_id": outcome.generation_id,
        "cancel_handle": _cancel_handle_details(handle),
        "cost_usd": outcome.cost_usd,
        # Where the figure came from (``cost_source``), what it means in money
        # terms (``billing_basis``) and its provenance: a BYOK estimate must
        # not read as a charge anywhere down the wire.
        "cost_source": outcome.cost_source,
        "billing_basis": outcome.billing_basis,
        "cost_provenance": outcome.cost_provenance,
        "attempts": _attempt_list(outcome.attempts),
    }
    if facts:
        details["audio"] = facts
    if voicing:
        details["voicing"] = voicing
    if blocks:
        # Durable identity of what was generated (see ``artifact_details``):
        # the per-session file index lists the audio from a pruned row.
        details["artifacts"] = artifact_details(blocks)
    if outcome.usage_record_id:
        # Absent-safe: present only when the winning rung's settled payload
        # carried one (Radient's does once settled). An identifier, never an amount.
        details["usage_record_id"] = outcome.usage_record_id

    label = AUDIO_RUNG_LABELS.get(str(outcome.route), str(outcome.route))
    noun = "voiceover" if spec.direction == "voiceover" else "sound effect"
    if not blocks:
        return ToolResult(
            tool_call_id=tool_call_id,
            tool_name=spec.tool,
            is_error=True,
            content=[
                TextContent(
                    text=(
                        (
                            "The generation was billed but the artifact could not be stored. "
                            if outcome.billing_basis == "billed"
                            else ""
                        )
                        + f"Generated a {noun} with {label} ({outcome.model}) but it could not be "
                        "registered in the session's attachment store "
                        f"(source: {', '.join(failed)})."
                    )
                )
            ],
            details=details,
        )

    asset = outcome.assets[0] if outcome.assets else None
    duration = _duration_text(asset.duration_s if asset is not None else None)
    duration_text = f" ({duration})" if duration else ""
    caption = (
        f"Generated a {noun}{duration_text} with {label} ({outcome.model}) — "
        f"attached to the session (digest {', '.join(digests)})."
        f"{_cost_caption(outcome.cost_usd, outcome.billing_basis, outcome.cost_source)}"
        f"{_voicing_note(voicing)}"
    )
    if failed:
        caption += (
            f" {len(failed)} of {len(outcome.assets)} failed to attach "
            f"(source: {', '.join(failed)})."
        )
    return ToolResult(
        tool_call_id=tool_call_id,
        tool_name=spec.tool,
        content=[TextContent(text=caption), *blocks],
        details=details,
    )
