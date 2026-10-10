"""The generator errand: one forked, bounded, multi-turn run that produces components.

Design authority: ``docs/design/turn-supplements.md`` §2.5 (the errand, its model and its
bounds), §2.6 (the fork, the prompt, the prelude), §2.7 (steer/restart), §2.9 (busy/idle) and
§5.3 (cost). Lane C1b.

WHY A FORK AND NOT A TOOL. The generator is handed the EVIDENCE, not the conversation: the
request is ``isolated`` with ``tools=[]`` and ``tool_choice="none"``, exactly the
``_errand_request`` shape (``session.py``), so it touches none of the session's sticky route,
rotation, prompt-cache key or context. It carries ``purpose="supplement_render"`` so its spend
lands in the ledger's ``by_purpose`` readout. Nothing about it reaches
``build_llm_history`` -- the parity test in ``tests/unit/supplements`` is the guard.

THE MODEL IS RESOLVED, NOT ASSUMED (§2.5). The fleet mostly runs cheap models and visual
design quality is this errand's whole job, so the default is :data:`DESIGN_MODEL_LADDER` --
the best design-capable model the operator is LOGGED IN TO -- with the operator's ``hi`` tier
and then the session's own model as fallbacks. A rung whose provider has no usable credential,
or whose id the shipped model registry does not know, is SKIPPED rather than tried: this is a
background errand, and a guess at a model id bills a failed request. The ladder is reviewed
beside the model catalogue each release; a stale id degrades to the next rung, never to an
exception. ``supplements.model`` overrides the whole thing ("provider/model_id"), and
``"session"`` forces the session's model (§2.12).

BOUNDS, AND WHAT EACH ONE COSTS WHEN IT FIRES. Turns (``maxTurns``), output tokens per turn
(``maxOutputTokens``), wall clock (``timeoutS``, enforced by the runner's ``wait_for``) and
spend between turns (``maxCostUsd``). Any bound firing ends the job, and the blocks that
already passed validation are KEPT (§2.5) -- losing a good chart because turn 2 hit the clock
would be the worst of both worlds. How the row reports it depends on what survived (the
round-1 review R4/R7 reading of §2.5): with NO block kept it is ``state=failed``,
``error="bound:<name>"``; with blocks kept the row is ``state=done`` -- the only state the
frozen reader paints as a settled block -- and the bound is recorded in the row's ``detail``.

TURN 2 IS A REPAIR, NOT A SECOND CHANCE. Only the rejected blocks are re-asked, with the
validator's own errors, and turn 2's output replaces only them (§2.5). A block that fails
twice is dropped and the failure recorded -- never rendered.

COST IS MEASURED, THEN CHECKED. Each turn's ``Usage`` is priced through
:func:`~local_operator.model.configure.cost_for_usage` -- the same money computation the
status band and the ledger use, so a job's spend cannot disagree with ``/usage`` -- and the
cap is checked BETWEEN turns (never mid-turn, so a single expensive turn completes rather than
leaving a half-finished frame).
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Final, Sequence

from local_operator.redaction_shapes import scrub_shapes
from local_operator.supplements import policy
from local_operator.supplements.contract import HEIGHT_HINT_MAX, HEIGHT_HINT_MIN
from local_operator.supplements.evidence import Dataset
from local_operator.supplements.prompt import (
    GENERATOR_SYSTEM_PROMPT,
    repair_message,
    steer_instruction,
)
from local_operator.supplements.validate import Component, Rejected, validate_output

logger = logging.getLogger(__name__)

#: Model ids per provider, best first (memo §2.5's order: Anthropic Sonnet-class, OpenAI GPT
#: mid-class, Gemini Pro-class). REVIEWED BESIDE THE MODEL CATALOGUE PER RELEASE: every id is
#: checked against the shipped registry at job time, and the first one the operator's usable
#: credential can reach wins. A stale id costs nothing (the rung is skipped) but it also buys
#: nothing, so this list is the one place to touch when a new design-capable model ships.
DESIGN_MODEL_LADDER: Final[tuple[tuple[str, tuple[str, ...]], ...]] = (
    (
        "anthropic",
        ("claude-sonnet-5-5", "claude-sonnet-5", "claude-sonnet-4-6", "claude-sonnet-4-5-20250929"),
    ),
    ("openai", ("gpt-6-astra", "gpt-5.6-sol", "gpt-4.1")),
    ("google", ("gemini-2.5-pro-preview-05-06",)),
)

#: The fork's OWN system block; never appended to a session's system prompt (memo §2.6/§10).
SYSTEM_PROMPT: Final = GENERATOR_SYSTEM_PROMPT

#: Evidence bounds (memo §2.5): the user message, the answer, and the datasets.
MAX_USER_CHARS: Final = 2_000
MAX_ANSWER_CHARS: Final = 6_000
MAX_DATASET_BYTES: Final = 24 * 1024
MAX_EVIDENCE_BYTES: Final = 64 * 1024

#: Bound names that reach ``error="bound:<name>"`` on the row (memo §2.5).
BOUND_TURNS: Final = "turns"
BOUND_TOKENS: Final = "tokens"
BOUND_TIME: Final = "time"
BOUND_COST: Final = "cost"
BOUND_COMPONENTS: Final = "components"

#: How many components one job may keep (memo §2.5's ``MAX_COMPONENTS``). Not a settings key:
#: the memo's config table has no row for it, so it is a code constant like the caps above.
MAX_COMPONENTS: Final = 3

#: One provider call: the fork's request in, ``(text, usage)`` out. Injected so the turn loop
#: is testable without a Session, and so the runner owns the transport's cancellation.
CompleteFn = Callable[[Any], Awaitable[tuple[str, Any]]]
#: ``(accepted, rejected)`` for one answer; the real one is :func:`validate_output`.
ValidateFn = Callable[[str, Sequence[Dataset]], tuple[Sequence[Component], Sequence[Rejected]]]
#: Progress: ``("generating"|"validating"|"repairing", elapsed_s)``. Never awaited into the
#: critical path: the runner's callback schedules the event.
ProgressFn = Callable[[str, float], Awaitable[None] | None]


@dataclass(frozen=True)
class DesignModel:
    """The resolved fork model and WHY it was chosen (the disclosure §2.5 asks for)."""

    spec: Any
    #: "config" | "ladder" | "tier" | "session"
    source: str
    label: str


@dataclass(frozen=True)
class GenerationOutcome:
    """What one generator job produced, in the row's own vocabulary."""

    components: tuple[Component, ...] = ()
    rejected: tuple[Rejected, ...] = ()
    model: str = ""
    model_source: str = ""
    turns: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    error: str = ""
    #: True when the fork answered the single word ``NONE``: a legitimate "no figure helps".
    none: bool = False
    #: The steer instruction this version was generated with, echoed for the row.
    instruction: str = ""
    detail: tuple[str, ...] = field(default=())


def resolve_design_model(
    *,
    configured: str,
    session_spec: Any,
    usable: set[str] | None,
    static_models: Callable[[str], dict[str, Any]],
    tier_spec: Any = None,
) -> DesignModel:
    """The fork's model (memo §2.5), in the memo's precedence order.

    ``usable`` is ``ProviderController.usable_providers()``'s answer -- or ``None`` when the
    credential store could not be read at all, which is NOT "no provider": it means we cannot
    tell, and the memo's ladder is then tried without the credential filter (the fork's own
    attempt is the cheap way to find out, and a failed rung is retried on the next job).

    Pure and synchronous: every input is an argument, so the ladder's behaviour under each
    login shape is a unit test rather than a live experiment.
    """

    def _labelled(spec: Any) -> str:
        return f"{getattr(spec, 'provider', '')}/{getattr(spec, 'model_id', '')}"

    configured = (configured or "").strip()
    if configured and configured.lower() != "auto":
        if configured.lower() == "session":
            return DesignModel(session_spec, "session", _labelled(session_spec))
        provider, _, model_id = configured.partition("/")
        if provider and model_id and (usable is None or provider in usable):
            return DesignModel(_build(provider, model_id), "config", configured)
        logger.debug("supplements: configured model %r is not reachable", configured)
    for provider, ids in DESIGN_MODEL_LADDER:
        if usable is not None and provider not in usable:
            continue
        known = static_models(provider) or {}
        for model_id in ids:
            if known and model_id not in known:
                continue
            return DesignModel(_build(provider, model_id), "ladder", f"{provider}/{model_id}")
    if tier_spec is not None:
        return DesignModel(tier_spec, "tier", _labelled(tier_spec))
    return DesignModel(session_spec, "session", _labelled(session_spec))


def _build(provider: str, model_id: str) -> Any:
    """A ``ModelSpec`` through the ONE spec builder (``model/configure.py``)."""
    from local_operator.model.configure import build_model_spec

    return build_model_spec(provider, model_id)


def _validate_once(
    text: str, evidence: Sequence[Dataset]
) -> tuple[tuple[Component, ...], tuple[Rejected, ...]]:
    """The production validator: one parse, one result object (the injection default)."""
    result = validate_output(text, list(evidence))
    if result.none:
        return (), ()
    return result.components, result.rejected


def _clip(text: str, limit: int) -> str:
    text = text or ""
    return text if len(text) <= limit else text[: limit - 1] + "\u2026"


def evidence_block(
    datasets: Sequence[Dataset],
    *,
    user_text: str,
    answer_text: str,
    instruction: str = "",
) -> str:
    """The fork's user message: the request, the answer and the datasets (memo §2.5).

    Bounded three ways (the answer/user clips, 24 KB per dataset, 64 KB in total) because this
    text rides into a paid request: an unbounded tool result would be billed on every turn of
    every job, and the datasets are the pre-filter's own bounded extraction (<= 200 rows each)
    already. Redaction happens HERE, once, on the assembled text -- the same
    ``scrub_shapes`` boundary the memo requires for outbound state, so a credential shape that
    survived the pre-filter's denylist still cannot leave the process in an evidence block.
    """
    import json

    parts = [
        "<evidence>",
        f"<request>{_clip(user_text, MAX_USER_CHARS)}</request>",
        f"<answer>{_clip(answer_text, MAX_ANSWER_CHARS)}</answer>",
    ]
    used = sum(len(part.encode("utf-8")) + 1 for part in parts)
    for index, dataset in enumerate(datasets):
        payload = json.dumps(
            {
                "id": dataset_id(index, dataset),
                "title": dataset.title,
                "source": dataset.source,
                "columns": list(dataset.columns),
                "rows": [list(row) for row in dataset.rows],
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        # The budgets are BYTE budgets (memo §2.5: "≤ 24 KB each, ≤ 64 KB total"), so the
        # accounting must be bytes: counting characters let a CJK-heavy six-dataset block
        # reach 143,924 bytes against the 64 KB total, silently (round-1 review R6).
        payload_bytes = len(payload.encode("utf-8"))
        if payload_bytes > MAX_DATASET_BYTES:
            continue
        if used + payload_bytes > MAX_EVIDENCE_BYTES:
            break
        used += payload_bytes + 1
        parts.append(f"<dataset>{payload}</dataset>")
    parts.append("</evidence>")
    if instruction:
        parts.append(steer_instruction(instruction))
    return scrub_shapes("\n".join(parts))


def dataset_id(index: int, dataset: Dataset) -> str:
    """A stable, model-friendly id for one evidence dataset (``d1``, ``d2``, ...)."""
    return f"d{index + 1}"


async def generate(
    *,
    complete: CompleteFn,
    model: DesignModel,
    datasets: Sequence[Dataset],
    user_text: str,
    answer_text: str,
    max_turns: int,
    max_output_tokens: int,
    max_cost_usd: float,
    max_components: int = MAX_COMPONENTS,
    instruction: str = "",
    timeout_s: float | None = None,
    validate: ValidateFn | None = None,
    price: Callable[[Any], float] | None = None,
    on_progress: ProgressFn | None = None,
    started: float | None = None,
    accepted_sink: list[Component] | None = None,
) -> GenerationOutcome:
    """Run the bounded turn loop and return what may be stored (memo §2.5).

    The loop is small on purpose: turn 1 asks for 0-3 blocks; the validator accepts some and
    rejects others; if anything was rejected AND turns remain, turn 2 is the repair message
    with the validator's errors. Nothing else is ever sent, and no bound is checked inside a
    turn -- the runner's ``wait_for`` owns the wall clock; this function owns turns, tokens
    and cost, and checks the clock only BETWEEN turns, so a repair the clock cannot afford is
    never started (round-1 review R4).

    ``accepted_sink``, when given, is refilled as validation proceeds with the blocks the
    caller may still keep: a bound firing OUTSIDE this function (the runner's ``wait_for``
    cancelling a hanging turn) cancels this coroutine, and the sink is then the only
    surviving copy of what had already passed (memo §2.5, round-1 review R4).
    """
    import time

    clock = time.perf_counter
    start = started if started is not None else clock()
    validator: ValidateFn = validate or _validate_once

    async def _progress(stage: str) -> None:
        """Tell the caller what the job is doing. Awaitable when the caller's sink is."""
        if on_progress is None:
            return
        result = on_progress(stage, clock() - start)
        if inspect.isawaitable(result):
            await result

    accepted: list[Component] = []
    rejected: list[Rejected] = []
    error = ""
    none = False
    turns = 0
    tokens_in = tokens_out = 0
    cost = 0.0
    detail: list[str] = []
    prompt = evidence_block(
        datasets, user_text=user_text, answer_text=answer_text, instruction=instruction
    )

    while turns < max(1, max_turns):
        turns += 1
        stage = "generating" if turns == 1 else "repairing"
        await _progress(stage)
        request = _fork_request(
            model.spec, prompt, max_output_tokens=max_output_tokens, purpose=policy.PURPOSE_RENDER
        )
        try:
            text, usage = await complete(request)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 -- fail open: the worst case is no figure
            logger.debug("supplements: generator turn failed", exc_info=True)
            error = f"generator:{type(exc).__name__}"
            break
        tokens_in += int(getattr(usage, "input_tokens", 0) or 0)
        tokens_out += int(getattr(usage, "output_tokens", 0) or 0)
        if price is not None and usage is not None:
            cost += float(price(usage) or 0.0)
        if (
            max_output_tokens > 0
            and usage is not None
            and int(getattr(usage, "output_tokens", 0) or 0) >= max_output_tokens
        ):
            # The per-turn token bound was hit. It is a CONSUMPTION bound, not a failure: a
            # truncated block is exactly what the repair turn exists for -- but it must leave
            # a record, or "which bound touched this job" is unknowable (round-1 review R7).
            detail.append(
                f"bound:{BOUND_TOKENS} turn {turns} hit the {max_output_tokens}-token "
                "per-turn limit"
            )
        await _progress("validating")
        accepted_now, rejected_now = validator(text, list(datasets))
        accepted.extend(accepted_now)
        rejected = list(rejected_now)
        if accepted_sink is not None:
            # Exposed BEFORE any later await can be cancelled: this is the copy the runner
            # recovers on its ``wait_for`` (round-1 review R4).
            accepted_sink[:] = accepted
        if not accepted_now and not rejected_now:
            # The fork answered the single word NONE (a legitimate "no figure helps").
            none = True
            detail.append(f"turn {turns}: NONE")
            break
        if len(accepted) > max_components:
            # More blocks passed than the row may carry: keep the first MAX_COMPONENTS and
            # record the trim. Dropping the extras is the memo's own cap (§2.5), and silently
            # keeping three of four would make the recorded decision a lie.
            detail.append(f"components bound: kept {max_components} of {len(accepted)}")
            accepted = accepted[:max_components]
            error = f"bound:{BOUND_COMPONENTS}"
            if accepted_sink is not None:
                accepted_sink[:] = accepted
        if not rejected:
            break
        if turns >= max(1, max_turns):
            # The turn bound: what was still rejected on the last allowed turn is dropped --
            # never rendered -- and the drop is recorded (memo §2.5: "dropped and the
            # failure recorded").
            detail.append(
                f"bound:{BOUND_TURNS} {len(rejected)} block(s) still rejected at "
                f"{turns} of {max(1, max_turns)} turn(s)"
            )
            error = f"bound:{BOUND_TURNS}" if not accepted else error
            break
        # BETWEEN-TURNS BOUNDS (memo §2.5): the cost cap and the wall clock decide whether
        # turn 2 is paid for at all. The clock is checked HERE (round-1 review R4) so an
        # expiry between turns returns NORMALLY -- the blocks that passed come back through
        # the ordinary tail and are kept -- while the runner's ``wait_for`` remains the hard
        # guard for a turn that hangs mid-flight.
        if cost >= max_cost_usd and max_cost_usd >= 0:
            detail.append(
                f"bound:{BOUND_COST} cap: ${cost:.4f} of ${max_cost_usd:.2f} reached before repair"
            )
            error = f"bound:{BOUND_COST}" if not accepted else error
            break
        if timeout_s is not None and clock() - start >= timeout_s:
            detail.append(
                f"bound:{BOUND_TIME} wall clock {clock() - start:.1f}s of {timeout_s:.0f}s "
                "reached before repair"
            )
            error = f"bound:{BOUND_TIME}" if not accepted else error
            break
        prompt = repair_message([item.describe() for item in rejected])
    return GenerationOutcome(
        components=tuple(accepted),
        rejected=tuple(rejected),
        model=model.label,
        model_source=model.source,
        turns=turns,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cost_usd=cost,
        error=error,
        none=none,
        instruction=instruction,
        detail=tuple(detail),
    )


def _fork_request(spec: Any, prompt: str, *, max_output_tokens: int, purpose: str) -> Any:
    """The fork's ``ChatRequest``: isolated, tools-free, no replay (memo §2.5/§2.6).

    The prefix-cache shape is the ``_errand_request`` precedent's -- NOT the ask gate's: the
    gate rides the turn's warm prefix by design (the session's system blocks and live tools),
    while this request carries its own single system block and no tools, sharing no prefix
    with the session (and on a different model there is none to share). Fast mode is cleared
    for the same reason -- an errand never pays the turn's priority premium.
    """
    from local_operator.harness.types import ChatRequest, Message

    if getattr(spec, "fast_mode", False):
        spec = spec.model_copy(update={"fast_mode": False})
    return ChatRequest(
        model=spec,
        purpose=purpose,
        system_blocks=[SYSTEM_PROMPT],
        messages=[Message.user(prompt)],
        tools=[],
        tool_choice="none",
        max_tokens=max_output_tokens,
        temperature=getattr(spec, "temperature", None),
        replayable=False,
        isolated=True,
    )


def component_bounds() -> tuple[int, int]:
    """The height clamp (memo §2.4/§2.11) as a pair, for a caller that needs to state it."""
    return HEIGHT_HINT_MIN, HEIGHT_HINT_MAX
