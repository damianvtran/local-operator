"""The ONLY place a model is touched: the LLM resolver for conflict groups (design B2.3-B2.7).

THE MODEL PROPOSES, THE CORE DISPOSES. This module turns a
:class:`~local_operator.hub_sync.merge.ConflictRequest` into a
:class:`~local_operator.hub_sync.merge.ConflictProposal`; ``merge.validate_proposal``
then accepts or rejects it. Nothing here decides what lands.

IMPORT DISCIPLINE. The stream/model stack is imported lazily inside functions:
``lop serve`` boots this package (the runner), and the server-shape guard in
``tests/unit/test_import_graph.py`` fails the build if a boot pulls the model
stack. A runner tick that never finds a conflict never pays for it.

WHICH MODEL. The user's configured default (``bootstrap.resolve_hosting_model``,
called rather than restated), unless ``hub.merge_model`` names another
``provider/model``. There is deliberately NO fallback chain: trying other
providers would spend a credential the user did not choose for this job.

WHY WE OWN THE BACKOFF. The request is ``isolated=True`` so the driver's own
retry/fallback/rotation and sticky-route writes are off (a background job must
not move the operator's foreground credential stickiness). That leaves retry to
us, which also bounds how long a runner worker can be held: 4 attempts, 2 s base,
60 s cap, +-25% jitter, ~15 s worst case per unit.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping

from local_operator.hub_sync.merge import (
    ConflictProposal,
    ConflictRequest,
    Resolver,
    ResolverError,
    ResolverErrorClass,
)

logger = logging.getLogger(__name__)

MAX_ATTEMPTS = 4
BACKOFF_BASE_S = 2.0
BACKOFF_CAP_S = 60.0
JITTER = 0.25
#: A quota error's ``retry_after`` is honoured up to this ceiling; a longer wait
#: fails the unit rather than parking the runner.
RETRY_AFTER_CEILING_S = 120.0
#: Per-item wall budget for one conflict group (B2.5).
MERGE_ITEM_TIMEOUT_S = 120.0
#: Tokens reserved for the system prompt and framing beyond the output (B2.6).
FRAMING_RESERVE_TOKENS = 2048
MAX_OUTPUT_TOKENS = 8192

SYSTEM_PROMPT = """You merge two edited versions of one passage of an AI agent's instructions.
You are given the BASE (the text both sides started from; may be absent), LOCAL (the
user's edit) and REMOTE (the hub's edit).
Write ONE passage that carries the intent of BOTH edits.

Rules:
- Keep every number, URL, identifier, `code span` and {{template}} token from either
  edit exactly as written.
- Never bring back text listed under MUST NOT REAPPEAR: a side deliberately deleted or shortened it.
- Do not add headings, code fences or commentary, and do not rewrite the read-only CONTEXT.
- Prefer the smallest change that combines both edits; keep the author's voice.

Answer with ONLY a JSON object:
{"text": "<the merged passage>", "covers": ["l1", "r1"], "drops": [], "rationale": "<= 200 chars"}
`covers` lists the ids whose content your text carries (l1 = LOCAL, r1 = REMOTE, b1 = BASE).
`drops` lists ids you deliberately left out; only b1 may be dropped, and only if a side
removed or shortened it."""


# -- prompt ------------------------------------------------------------------------------


def build_prompt(req: ConflictRequest) -> str:
    """The user message for one group; deterministic so retries are comparable."""

    parts: list[str] = [
        f"FIELD: {req.field}" + (f"  SECTION: {req.heading}" if req.heading else "")
    ]
    parts.append(f"BASE (b1):\n{req.base}" if req.base is not None else "BASE (b1): (none)")
    parts.append(f"LOCAL (l1):\n{req.local}")
    parts.append(f"REMOTE (r1):\n{req.remote}")
    if req.removals:
        lines = "\n".join(f"- (removed by {r.removed_by}) {r.text}" for r in req.removals)
        parts.append(f"MUST NOT REAPPEAR:\n{lines}")
    if req.keep_verbatim:
        parts.append("CONTEXT (read-only, do not rewrite):\n" + "\n".join(req.keep_verbatim))
    parts.append(f"The merged passage must be at most {req.max_chars} characters.")
    if req.feedback:
        parts.append(
            "YOUR PREVIOUS ANSWER WAS REJECTED. Fix these and answer again:\n"
            + "\n".join(f"- {p}" for p in req.feedback)
        )
    return "\n\n".join(parts)


_FENCE_RE = re.compile(r"^```[a-zA-Z]*\s*|\s*```$")


def parse_proposal(raw: str) -> ConflictProposal:
    """Parse the model's JSON answer; anything else is ``invalid-output``."""

    text = _FENCE_RE.sub("", raw.strip())
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ResolverError("invalid-output", "the model did not answer with a JSON object")
    try:
        data = json.loads(text[start : end + 1])
    except ValueError as exc:
        raise ResolverError("invalid-output", f"unparseable JSON: {exc}") from None
    if not isinstance(data, dict) or not isinstance(data.get("text"), str):
        raise ResolverError("invalid-output", 'the JSON has no string "text"')

    def ids(key: str) -> tuple[str, ...]:
        value = data.get(key)
        return tuple(str(v) for v in value) if isinstance(value, list) else ()

    return ConflictProposal(
        text=data["text"].strip("\n"),
        covers=ids("covers"),
        drops=ids("drops"),
        notes=str(data.get("rationale") or "")[:200],
    )


# -- model resolution (B2.4) -------------------------------------------------------------


@dataclass(frozen=True)
class MergeModel:
    """A resolved model: the harness spec plus the label reports carry."""

    spec: Any
    label: str


def resolve_merge_model(config_manager: Any) -> MergeModel:
    """``hub.merge_model`` if set, else the user's default model. Never a fallback chain.

    Raises:
        ResolverError: ``model-unavailable`` for a malformed override, no
            configured default, or an unknown provider. A malformed override is a
            NAMED refusal (not a silent fall-through) so a typo is not invisible.
    """

    from local_operator.hub_sync.settings import HubSyncSettings

    # read_fresh, not from_config: ``hub.merge_model`` is a LIVE key and the daemon's
    # manager is the one it loaded at boot, so an edit from the TUI/CLI would go unseen.
    override = HubSyncSettings.read_fresh(config_manager).merge_model
    if override:
        provider, _, model_id = override.partition("/")
        if not provider.strip() or not model_id.strip():
            raise ResolverError(
                "model-unavailable", f"hub.merge_model={override!r} lacks provider/model"
            )
        hosting, model = provider.strip(), model_id.strip()
    else:
        from local_operator.bootstrap import resolve_hosting_model

        try:
            hosting, model = resolve_hosting_model(config_manager, None, None, None)
        except Exception as exc:  # noqa: BLE001 - ValueError / ModelNotConfiguredError
            raise ResolverError("model-unavailable", str(exc)) from None
    try:
        from local_operator.model.configure import build_model_spec

        spec = build_model_spec(hosting, model)
    except Exception as exc:  # noqa: BLE001 - unknown provider / metadata failure
        raise ResolverError("model-unavailable", f"cannot use {hosting}/{model}: {exc}") from None
    return MergeModel(spec=spec, label=f"{hosting}/{model}")


# -- the lifted one-shot call (B2.3) -----------------------------------------------------


async def complete_once(
    system: str,
    prompt: str,
    *,
    model: Any,
    config_dir: Path | None,
    settings: Mapping[str, Any] | None,
    max_tokens: int = MAX_OUTPUT_TOKENS,
    purpose: str = "hub_merge",
) -> str:
    """One non-agentic completion; the body of ``ServerExecutor.invoke_model``.

    Lifted rather than called because ``invoke_model`` needs a whole
    ``ServerOperator``/``ModelConfiguration`` and ``Session.complete_once`` needs
    a live session; the runner has neither. ``isolated=True`` keeps this off the
    foreground sessions' credential stickiness; ``session_id="hub-merge"``
    attributes the spend in analytics instead of an anonymous bucket.
    """

    from local_operator.harness.types import (
        ChatRequest,
        Message,
        StreamEndEvent,
        StreamTextDelta,
    )
    from local_operator.model.configure import create_stream_fn
    from local_operator.providers.auth_store import AuthStore

    auth = AuthStore(config_dir=config_dir)
    stream_fn = create_stream_fn(auth, settings=settings, session_id="hub-merge")
    try:
        request = ChatRequest(
            model=model,
            system_blocks=[system],
            messages=[Message.user(prompt)],
            tools=[],
            tool_choice="none",
            max_tokens=max_tokens,
            purpose=purpose,
            isolated=True,
            # We retry at OUR layer (B2.5); a replayable request would ask the
            # driver to as well.
            replayable=False,
        )
        parts: list[str] = []
        async for event in stream_fn(request, None):
            if isinstance(event, StreamTextDelta):
                parts.append(event.delta)
            elif isinstance(event, StreamEndEvent) and event.error:
                raise RuntimeError(event.error)
        return "".join(parts)
    finally:
        try:
            await stream_fn.close()
        except Exception:  # noqa: BLE001 - teardown must not mask the completion
            logger.debug("stream close failed", exc_info=True)
        try:
            auth.close()
        except Exception:  # noqa: BLE001
            logger.debug("auth store close failed", exc_info=True)


CompleteFn = Callable[[str, str], Awaitable[str]]
SleepFn = Callable[[float], Awaitable[None]]


# -- failure classification and backoff (B2.5) -------------------------------------------


def backoff_delay(attempt: int, *, rng: Callable[[float, float], float] = random.uniform) -> float:
    """``min(cap, base * 2**n) * uniform(0.75, 1.25)`` for the n-th failure (n from 0)."""

    return min(BACKOFF_CAP_S, BACKOFF_BASE_S * (2**attempt)) * rng(1 - JITTER, 1 + JITTER)


def classify_failure(
    error: BaseException,
) -> tuple[ResolverErrorClass, str | None, float | None]:
    """``(class, subclass, retry_after_s)`` for a provider failure.

    ``class`` is a :data:`~local_operator.hub_sync.merge.ResolverErrorClass`;
    ``subclass`` names quota/timeout/offline/transient for ``provider-error``.
    Only a genuine ``ProviderError`` (or a raw transport error) is read; the
    text of an arbitrary exception is the harness's own words, not weather.
    """

    from local_operator.providers import failover as fo

    if isinstance(error, fo.ProviderError):
        kind = error.kind
        retry_after = error.retry_after_ms / 1000.0 if error.retry_after_ms else None
        if fo.is_request_too_large(error):
            return "prompt-too-long", None, None
        if kind == "auth":
            return "model-unavailable", None, None
        if kind == "request":
            return "provider-error", "request", None
        if fo.is_connectivity_loss(error):
            return "provider-error", "offline", None
        if kind == "quota":
            return "provider-error", "quota", retry_after
        if kind == "timeout":
            return "provider-error", "timeout", None
        return "provider-error", "transient", None
    if fo.is_connectivity_loss(error):
        return "provider-error", "offline", None
    kind = fo.classify_provider_error(error)
    if kind == "timeout":
        return "provider-error", "timeout", None
    if kind == "transient":
        return "provider-error", "transient", None
    if kind == "auth":
        return "model-unavailable", None, None
    return "provider-error", "request", None


#: Subclasses worth another attempt at this layer. ``offline`` is NOT here: it
#: fails fast and the store-level backoff (B4.4) owns the long wait, so a worker
#: is never parked for minutes. ``request`` is terminal (retrying the same body
#: cannot help).
_RETRYABLE = {"quota", "timeout", "transient"}


async def call_with_retry(
    complete: CompleteFn,
    system: str,
    prompt: str,
    *,
    sleep: SleepFn = asyncio.sleep,
    rng: Callable[[float, float], float] = random.uniform,
    max_attempts: int = MAX_ATTEMPTS,
) -> tuple[str, int]:
    """Run ``complete`` with the B2.5 policy; returns ``(text, attempts_used)``.

    ``asyncio.CancelledError`` propagates untouched — a lifespan shutdown must end
    a merge promptly, and swallowing it would hold the daemon open.
    """

    attempts = 0
    while True:
        attempts += 1
        try:
            return await complete(system, prompt), attempts
        except asyncio.CancelledError:
            raise
        except Exception as error:  # noqa: BLE001 - classified below
            cls, sub, retry_after = classify_failure(error)
            if cls != "provider-error" or sub not in _RETRYABLE or attempts >= max_attempts:
                raise ResolverError(
                    cls, str(error)[:300], subclass=sub, attempts=attempts
                ) from None
            if sub == "quota" and retry_after is not None:
                if retry_after > RETRY_AFTER_CEILING_S:
                    raise ResolverError(
                        cls,
                        f"quota wait {retry_after:.0f}s exceeds the ceiling",
                        subclass=sub,
                        attempts=attempts,
                    ) from None
                delay = retry_after
            else:
                delay = backoff_delay(attempts - 1, rng=rng)
            logger.debug(
                "hub merge attempt %d failed (%s/%s); retrying in %.1fs", attempts, cls, sub, delay
            )
            await sleep(delay)


# -- chunking (B2.6) ---------------------------------------------------------------------


def usable_tokens(spec: Any) -> int:
    window = int(getattr(spec, "context_window", 128_000) or 128_000)
    out = min(
        int(getattr(spec, "max_output_tokens", MAX_OUTPUT_TOKENS) or MAX_OUTPUT_TOKENS),
        MAX_OUTPUT_TOKENS,
    )
    return max(1024, window - out - FRAMING_RESERVE_TOKENS)


def _tokens(text: str) -> int:
    # Never the tokenizer: loading it on the hot path costs ~84 ms/43 MB and a
    # cold cache reaches for the network (``approx_text_tokens`` exists for this).
    from local_operator.compaction.tokens import approx_text_tokens

    return approx_text_tokens(text)


def fits(req: ConflictRequest, spec: Any) -> bool:
    return _tokens(SYSTEM_PROMPT + build_prompt(req)) <= usable_tokens(spec)


def shrink_context(req: ConflictRequest, spec: Any) -> ConflictRequest | None:
    """Drop optional context (neighbours, then long removals) until the request fits.

    The conflict's own base/local/remote are never truncated: those ARE the
    task, and a silently clipped edit is the corruption this module exists to
    prevent. Returns ``None`` when even the bare triple does not fit.
    """

    if fits(req, spec):
        return req
    trimmed = ConflictRequest(**{**req.__dict__, "keep_verbatim": ()})
    if fits(trimmed, spec):
        return trimmed
    trimmed = ConflictRequest(**{**trimmed.__dict__, "removals": trimmed.removals[:8]})
    return trimmed if fits(trimmed, spec) else None


# -- the resolver ------------------------------------------------------------------------


class LlmResolver:
    """A sync :class:`Resolver` backed by the model.

    ``merge_field`` is synchronous (so the deterministic core stays trivially
    testable); the async call runs in a worker thread's own loop. Callers reach
    this through ``asyncio.to_thread`` — the shape every route uses — so there is
    no running loop in this thread to collide with.
    """

    def __init__(
        self,
        config_manager: Any,
        *,
        complete: CompleteFn | None = None,
        model: MergeModel | None = None,
    ) -> None:
        self._cm = config_manager
        self._complete = complete
        self._model = model
        self.model_label: str | None = model.label if model else None

    def _resolved(self) -> MergeModel:
        if self._model is None:
            self._model = resolve_merge_model(self._cm)
            self.model_label = self._model.label
        return self._model

    def resolve(self, req: ConflictRequest) -> ConflictProposal:
        model = self._resolved()
        fitted = shrink_context(req, model.spec)
        if fitted is None:
            # B2.6.4's honest last resort: no truncation, no silent skip.
            raise ResolverError(
                "prompt-too-long",
                f"section {req.heading or req.field!r} does not fit the model window",
            )
        complete = self._complete or self._default_complete(model)
        try:
            return asyncio.run(self._resolve_async(complete, fitted, model))
        except ResolverError as err:
            if err.cls != "prompt-too-long":
                raise
        # The provider's window is smaller than the metadata claimed (B2.6.4):
        # shed the optional context ONCE and retry. Still too long -> the honest
        # last resort below; the edits themselves are never truncated.
        slimmer = ConflictRequest(
            **{**fitted.__dict__, "keep_verbatim": (), "removals": fitted.removals[:8]}
        )
        if slimmer == fitted:
            raise ResolverError(
                "prompt-too-long", f"section {req.heading or req.field!r} is too long"
            )
        return asyncio.run(self._resolve_async(complete, slimmer, model))

    def _default_complete(self, model: MergeModel) -> CompleteFn:
        config_dir = getattr(self._cm, "config_dir", None)
        settings = self._cm.get_config().values

        async def _complete(system: str, prompt: str) -> str:
            return await complete_once(
                system, prompt, model=model.spec, config_dir=config_dir, settings=settings
            )

        return _complete

    async def _resolve_async(
        self, complete: CompleteFn, req: ConflictRequest, model: MergeModel
    ) -> ConflictProposal:
        try:
            text, attempts = await asyncio.wait_for(
                call_with_retry(complete, SYSTEM_PROMPT, build_prompt(req)),
                timeout=MERGE_ITEM_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            raise ResolverError(
                "provider-error",
                f"no answer within {MERGE_ITEM_TIMEOUT_S:.0f}s",
                subclass="timeout",
            ) from None
        proposal = parse_proposal(text)
        return ConflictProposal(
            text=proposal.text,
            covers=proposal.covers,
            drops=proposal.drops,
            notes=proposal.notes,
            attempts=attempts,
            model=model.label,
        )


def make_resolver(config_manager: Any) -> Resolver:
    return LlmResolver(config_manager)
