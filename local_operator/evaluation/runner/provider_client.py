"""The only runner module allowed to reach into ``local_operator.model``.

``episode.py`` must stay free of provider, config, and session imports so an
episode cannot inherit the operator's live configuration. That constraint has
to break somewhere for a real run, and it breaks here, behind
:class:`~local_operator.evaluation.runner.model.EpisodeModelClient`.

Strict decision parsing also lives here rather than in the runner core. The
runner treats a malformed decision as a provider failure, so the parsing rules
and the provider that produced them stay in the same module.

**Context management** also lives here, on the central compaction engine
(``local_operator.compaction``) rather than a second implementation. The
runner hands over protocol-typed ``EpisodeTurn``s; :class:`_ContextBuilder`
turns them into an APPEND-ONLY message history — user(text + frame) /
assistant(the batch it chose, verbatim) — and the only thing that ever rewrites
a sent message is a whole-prefix rebuild through ``run_compaction_pass``. The
cadence of those rebuilds is the one real design tension in this module and
is documented on :meth:`_ContextBuilder.build`.
"""

from __future__ import annotations

import base64
import json
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence, get_args, get_origin

from local_operator.evaluation.adapters.supervisor import verify_artifact
from local_operator.evaluation.evidence.models import RouteIdentity
from local_operator.evaluation.protocol import ActionBatch, ComputerAction, Observation
from local_operator.evaluation.runner.model import (
    CompactionRecord,
    EpisodeTurn,
    ModelDecision,
    ModelUsage,
)

if TYPE_CHECKING:
    from local_operator.compaction.thresholds import CompactionSettings
    from local_operator.harness.types import Message

#: Frames kept verbatim in the history by default. A GUI agent reads the
#: screen as STATE: the current frame is what it acts on and the last couple
#: are what it compares against, while anything older is a view the surface
#: has since replaced. This is a behavioural constant, not a tuning knob for a
#: particular benchmark -- it is the same ``keep_recent_frames`` an interactive
#: screen-driving session would set on its ``CompactionSettings``.
DEFAULT_KEEP_RECENT_FRAMES = 3

#: How many frames beyond ``keep_recent_frames`` may accumulate before the
#: whole prefix is rebuilt. See ``_ContextBuilder.build`` for why pruning is
#: batched into rebuilds instead of done per turn.
DEFAULT_REBUILD_EVERY_FRAMES = 8

#: Rendered in place of an observation's text when it is byte-identical to the
#: previous turn's. The adapter owns observation text and the runner never
#: second-guesses it; this is the one append-only-safe dedup, because it only
#: changes the message being appended.
UNCHANGED_OBSERVATION = "(unchanged)"

# The batch wire version this harness speaks; pinned here rather than taken
# from a model reply.
PROTOCOL_VERSION = "1.0"

# Mirrors receipts.StrictIdentifier, which every evidence identifier must match.
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]*")


def _action_schema_lines() -> list[str]:
    """Describe every action kind by reading the protocol models themselves.

    A parse failure is TERMINAL for an episode -- there is no retry and no
    repair -- so a prompt that under-specifies the wire shape is a correctness
    defect, not prompt polish. Deriving the text from ``ComputerAction`` means a
    new action kind or a changed literal cannot silently drift out of the
    instructions the model is given.
    """

    lines: list[str] = []
    for action in get_args(get_args(ComputerAction)[0]):
        fields: list[str] = []
        for name, field in action.model_fields.items():
            if name in ("kind", "observation_id"):
                continue
            choices = get_args(field.annotation)
            if choices and all(isinstance(choice, str) for choice in choices):
                rendered = "|".join(str(choice) for choice in choices)
            else:
                rendered = _type_name(field.annotation)
            optional = "" if field.is_required() else " (optional)"
            fields.append(f"{name}: {rendered}{optional}")
        kind = action.model_fields["kind"].default
        detail = ", ".join(fields) if fields else "no further fields"
        lines.append(f'  {{"kind": "{kind}", "observation_id": "<id>", {detail}}}')
    return lines


def _type_name(annotation: Any) -> str:
    """Name a field's JSON shape well enough for a model to emit it correctly.

    Sequences must render as arrays rather than the generic placeholder: told
    only "value", a model emits ``"ctrl+c"`` for ``KeyAction.keys`` where the
    protocol requires ``["ctrl", "c"]``, and that is a hard parse failure with
    no retry -- the same terminal-failure class B2 addressed, just narrower.
    """

    name = getattr(annotation, "__name__", None)
    if name in ("int", "str", "bool", "float"):
        return name
    origin = get_origin(annotation)
    if origin in (tuple, list, set, frozenset):
        args = [arg for arg in get_args(annotation) if arg is not Ellipsis]
        inner = _type_name(args[0]) if args else "value"
        return f"[{inner}, ...]"
    args = [arg for arg in get_args(annotation) if arg is not type(None)]
    if len(args) == 1:
        return _type_name(args[0])
    return "value"


def build_system_prompt() -> str:
    """Compose the episode system prompt around the live protocol schema."""

    return f"""You are operating a computer to complete one task.

Each user message is one observation of the screen: its text, and a screenshot
when one is attached. Your own earlier replies are the actions you already
took; an observation reading "(unchanged)" has the same text as the one before
it, and "[screenshot omitted ...]" marks an older screenshot that a newer one
has replaced. A message starting <previous-context-summary> summarises turns
that are no longer shown.

Reply with a single JSON object and nothing else, with no prose and no code
fence:

  {{"actions": [ ... ]}}

Every action is an object whose type is given by the key "kind" (NOT "type"),
and every action must carry the "observation_id" of the observation you are
looking at right now. These are the only permitted shapes:

{chr(10).join(_action_schema_lines())}

Where a field lists alternatives separated by "|", you must use exactly one of
those literal values.

Two actions end your turn in a special way, and each must be the ONLY action in
its batch. Their fields are listed above; what the list cannot tell you is what
they MEAN:

* "finish" -- you believe the task is done. The episode is then scored.
* "ask_user" -- you need a human answer. THE EPISODE PAUSES: a person answers
  your question, and the next observation you see is the state after that
  answer was delivered. Do not ask a question you can resolve by acting.
"""


_SYSTEM_PROMPT = build_system_prompt()


class DecisionParseError(ValueError):
    """The provider returned something that is not a usable action batch."""


def parse_decision(
    payload: str,
    observation: Observation,
    *,
    route: RouteIdentity,
    usage: ModelUsage | None = None,
    cost_micros: int = 0,
    stop_reason: str = "stop",
    provider_request_id: str = "unknown",
    tool_call_count: int = 0,
    prompt_cache_key: str | None = None,
    context_tokens: int | None = None,
    compaction: CompactionRecord | None = None,
) -> ModelDecision:
    """Parse one strict JSON decision and bind it to the current observation.

    Strictness is deliberate: a batch whose actions name a different
    observation is stale, and executing it would apply a decision made about a
    screen the environment has already moved past. ``ActionBatch.validate_for``
    rejects that at the adapter boundary, so the failure is surfaced here where
    it can be attributed to the provider.
    """

    try:
        decoded: Any = json.loads(payload)
    except json.JSONDecodeError as error:
        raise DecisionParseError(f"decision is not valid JSON: {error}") from error
    if not isinstance(decoded, Mapping):
        raise DecisionParseError("decision must be a JSON object")
    actions = decoded.get("actions")
    if not isinstance(actions, list) or not actions:
        raise DecisionParseError("decision must carry a non-empty actions array")
    try:
        batch = ActionBatch.model_validate(
            {
                # The wire version is pinned by the harness, never by the model:
                # a reply cannot select which protocol it is validated against.
                "protocol_version": PROTOCOL_VERSION,
                "task_id": observation.task_id,
                "episode_id": observation.episode_id,
                "observation_id": observation.observation_id,
                "actions": actions,
            },
            strict=True,
        )
    except Exception as error:
        raise DecisionParseError(f"decision is not a valid action batch: {error}") from error
    try:
        batch.validate_for(observation)
    except Exception as error:
        raise DecisionParseError(f"decision does not match this observation: {error}") from error
    return ModelDecision(
        action_batch=batch,
        route=route,
        usage=usage or ModelUsage(),
        cost_micros=cost_micros,
        stop_reason=stop_reason,
        provider_request_id=provider_request_id,
        tool_call_count=tool_call_count,
        prompt_cache_key=prompt_cache_key,
        context_tokens=context_tokens,
        compaction=compaction,
    )


class _ContextBuilder:
    """The append-only message history one episode sends to the model.

    Holds ``_messages`` (every message already sent, in order) plus two
    cursors: how many turns have had their OBSERVATION rendered as a user
    message, and which turns have had their BATCH rendered as the assistant
    message that follows it. Each :meth:`append_new_turns` appends only what
    is new. Once a message is in ``_messages`` it is never edited -- the only
    operation that replaces the list wholesale is :meth:`replace`, which a
    compaction pass drives.
    """

    def __init__(
        self,
        *,
        artifact_root: Path,
        keep_recent_frames: int,
        rebuild_every_frames: int,
    ) -> None:
        if keep_recent_frames < 0:
            raise ValueError("keep_recent_frames must be non-negative")
        if rebuild_every_frames < 1:
            raise ValueError("rebuild_every_frames must be positive")
        self._artifact_root = artifact_root
        self._keep_recent_frames = keep_recent_frames
        self._rebuild_every_frames = rebuild_every_frames
        self._messages: list[Message] = []
        self._rendered_turns = 0
        self._closed_turns: set[int] = set()
        self._previous_text: str | None = None

    @property
    def messages(self) -> list[Message]:
        return self._messages

    def append_new_turns(self, history: Sequence[EpisodeTurn]) -> None:
        """Render every turn not yet in the history, in order.

        A turn is rendered in its FINAL form the moment it is appended and is
        not touched again. Turn ``i``'s batch is unknown when its observation
        is sent (the model has not answered yet), so it is appended as the
        assistant message when the NEXT call arrives -- which is exactly the
        model replaying its own prior decision, verbatim, and is what makes
        the prefix append-only rather than re-rendered.
        """
        from local_operator.harness.types import Message, TextContent

        for index, turn in enumerate(history):
            if index >= self._rendered_turns:
                previous = history[index - 1] if index > 0 else None
                self._messages.append(self._render_observation(turn, previous))
                self._rendered_turns = index + 1
            if turn.batch is not None and index not in self._closed_turns:
                self._messages.append(
                    Message(
                        role="assistant",
                        content=[TextContent(text=turn.batch.to_canonical_json().decode("utf-8"))],
                    )
                )
                self._closed_turns.add(index)

    def _render_observation(self, turn: EpisodeTurn, previous: EpisodeTurn | None) -> Message:
        from local_operator.harness.types import ImageContent, Message, TextContent

        observation = turn.observation
        text = observation.text or "(no textual state)"
        # The adapter owns observation text and the runner never rewrites it;
        # the one dedup that is append-only-safe is choosing how to render the
        # message being appended, so a byte-identical repeat is sent as a
        # marker rather than as the same paragraph again.
        rendered_text = UNCHANGED_OBSERVATION if text == self._previous_text else text
        self._previous_text = text
        lines = [
            f"Step: {observation.sequence}",
            f"Observation ID: {observation.observation_id}",
        ]
        if previous is None:
            # Stated once, on the first observation, where it stays in the
            # cached prefix; repeating it per turn would be tokens spent on a
            # fact the model already has.
            lines.insert(0, f"Task: {observation.task_id}")
        if previous is not None and previous.ask_answer is not None:
            # The answer to the previous turn's ask_user reaches the model as
            # part of the observation that followed it -- "the state after
            # that answer was delivered", as the system prompt promises.
            lines.append(f"Answer from the user: {previous.ask_answer}")
        lines.extend(["", rendered_text])
        content: list[Any] = [TextContent(text="\n".join(lines))]
        for frame in observation.frames:
            # Bytes come through the SAME reader the runner verifies frames
            # with (O_NOFOLLOW, size, digest): a frame the runner would refuse
            # to publish is a frame the model must not be shown, and a second
            # reader here would be a second place for that check to drift.
            data = verify_artifact(self._artifact_root, frame.artifact)
            content.append(
                ImageContent(
                    data=base64.b64encode(data).decode("ascii"),
                    mime_type=frame.artifact.media_type,
                )
            )
        return Message(role="user", content=content)

    def frames_in_history(self) -> int:
        from local_operator.compaction.pruning import count_frame_messages

        return count_frame_messages(self._messages)

    def rebuild_due(self) -> bool:
        """Whether the frame budget says it is time to rebuild the prefix.

        THE cache/frames tension, settled: the model should see only the
        newest ``keep_recent_frames`` frames, but a provider's prompt cache is
        keyed on the exact bytes of the sent prefix, and the old pilot's direct
        API probe measured that ANY rewrite of an already-sent message costs
        the whole entry (0% hits, $262 per run with per-turn pruning). Pruning
        the frame that just fell out of the window every turn is therefore the
        worst possible policy. The coherent form is: N appends, all cached,
        then ONE rebuild that prunes every stale frame at once and accepts one
        miss. ``rebuild_every_frames`` is that N; at the default 8 the
        steady-state hit rate is about 7 of every 8 turns.
        """
        return self.frames_in_history() > self._keep_recent_frames + self._rebuild_every_frames

    def replace(self, messages: Sequence[Message]) -> None:
        self._messages = list(messages)


class ProviderModelClient:
    """Drives a real provider through the session stream function.

    The stream function already owns credential resolution and retry, so a
    failure that reaches this class is terminal and is allowed to propagate:
    the runner records it as a provider error and finalizes the episode
    unscored on a still-live session.

    One client serves one episode: it owns that episode's append-only message
    history (``_ContextBuilder``) and the compaction settings that govern its
    rebuilds. ``artifact_root`` is where the adapter published frames; the
    runner verifies frames from the same root.
    """

    def __init__(
        self,
        stream_fn: Any,
        *,
        route: RouteIdentity,
        model_spec: Any,
        artifact_root: Path,
        system_prompt: str = _SYSTEM_PROMPT,
        compaction: "CompactionSettings | None" = None,
        keep_recent_frames: int = DEFAULT_KEEP_RECENT_FRAMES,
        rebuild_every_frames: int = DEFAULT_REBUILD_EVERY_FRAMES,
        prompt_cache_key: str | None = None,
    ) -> None:
        from local_operator.compaction.thresholds import CompactionSettings

        self._stream_fn = stream_fn
        self._route = route
        self._model_spec = model_spec
        self._system_prompt = system_prompt
        self._prompt_cache_key = prompt_cache_key
        base = compaction or CompactionSettings()
        # The episode's frame policy is authoritative over whatever the
        # settings object carried: this client IS a screen-driving surface,
        # so the opt-in is made here, once, for the whole episode.
        self._compaction = base.model_copy(update={"keep_recent_frames": keep_recent_frames})
        self._context = _ContextBuilder(
            artifact_root=artifact_root,
            keep_recent_frames=keep_recent_frames,
            rebuild_every_frames=rebuild_every_frames,
        )
        self._last_provider_context_tokens: int | None = None
        self._last_request_ms = _now_ms()

    async def decide(
        self,
        observation: Observation,
        history: Sequence[EpisodeTurn],
    ) -> ModelDecision:
        from local_operator.harness.types import ChatRequest

        self._context.append_new_turns(history)
        compaction, extra_usage, extra_cost = await self._maybe_compact()

        # The system prompt rides in ``system_blocks`` rather than as a message
        # so the provider can place a cache breakpoint after this stable block;
        # every episode step repeats it verbatim.
        messages = self._context.messages
        request = ChatRequest(
            model=self._model_spec,
            system_blocks=[self._system_prompt],
            messages=list(messages),
            tool_choice="none",
            prompt_cache_key=self._prompt_cache_key,
        )
        text, usage, cost_micros, stop_reason, provider_request_id, context_tokens = (
            await self._stream(request)
        )
        self._last_request_ms = _now_ms()
        if context_tokens is not None:
            self._last_provider_context_tokens = context_tokens
        # The summary call was a billed provider call on the way to this
        # decision. It is folded into THIS decision's figures rather than
        # reported separately so the bundle's usage events remain the whole
        # bill and the counters a pure sum of them.
        if extra_usage is not None:
            usage = _add_usage(usage, extra_usage)
            cost_micros += extra_cost
        return parse_decision(
            text.strip(),
            observation,
            route=self._route,
            usage=usage,
            cost_micros=cost_micros,
            stop_reason=stop_reason,
            provider_request_id=provider_request_id,
            prompt_cache_key=self._prompt_cache_key,
            context_tokens=_estimate_context(messages),
            compaction=compaction,
        )

    async def _maybe_compact(self) -> tuple[CompactionRecord | None, ModelUsage | None, int]:
        """Run one compaction pass when the frame budget or the token threshold says so.

        Two triggers, one pass. The frame budget (``rebuild_due``) is the usual
        one for a screen-driving episode; the token threshold is the ordinary
        session's trigger and applies here too because a long text-heavy
        episode can fill the window without ever tripping the frame budget.
        Either way the pass rebuilds the whole prefix ONCE, which is the only
        rewrite this client ever makes to sent messages.
        """
        from local_operator.compaction.pass_ import run_compaction_pass
        from local_operator.compaction.thresholds import (
            compaction_context_tokens,
            should_compact,
        )
        from local_operator.compaction.tokens import estimate_messages_tokens

        messages = self._context.messages
        if not messages:
            return None, None, 0
        frame_due = self._context.rebuild_due()
        tokens_before = compaction_context_tokens(
            self._last_provider_context_tokens, estimate_messages_tokens(messages)
        )
        threshold_due = should_compact(
            tokens_before,
            int(getattr(self._model_spec, "context_window", 0) or 0),
            self._compaction,
        )
        if not (frame_due or threshold_due):
            return None, None, 0

        summary_usage: ModelUsage | None = None
        summary_cost = 0

        async def summarize(prompt: str) -> str:
            nonlocal summary_usage, summary_cost
            text, usage, cost, _stop, _rid, _ctx = await self._stream(self._summary_request(prompt))
            summary_usage = usage
            summary_cost = cost
            return text

        before = len(messages)
        result = await run_compaction_pass(
            messages,
            model=self._model_spec,
            settings=self._compaction,
            summarize=summarize,
            now_ms=_now_ms(),
            last_activity_ms=self._last_request_ms,
            provider_context_tokens=self._last_provider_context_tokens,
            # The frame prune inside the pass runs unconditionally; only the
            # SUMMARY half is gated on the threshold. So a frame-budget rebuild
            # on a small context prunes its frames and refuses the summary as
            # below-threshold, which is the common and correct outcome, and a
            # context that is genuinely over the line gets both.
            respect_threshold=True,
        )
        # The pass returns the pruned list even when it refused to summarize;
        # either way it is the prefix to send from now on.
        self._context.replace(result.messages)
        if result.frames_dropped == 0 and not result.ran and not result.pruned:
            return None, None, 0
        record = CompactionRecord(
            strategy=result.strategy or "prune",
            # The pass measures its own "before" AFTER its prune step (that is
            # the figure its trigger judges); the bundle wants the size of the
            # context the model last saw, so the pre-pass estimate is recorded.
            tokens_before=tokens_before,
            tokens_after=result.tokens_after,
            frames_dropped=result.frames_dropped,
            messages_before=before,
            messages_after=len(result.messages),
            summary_text=result.summary_text,
        )
        return record, summary_usage, summary_cost

    def _summary_request(self, prompt: str) -> Any:
        from local_operator.compaction.api import SUMMARIZATION_SYSTEM_PROMPT
        from local_operator.harness.types import ChatRequest, Message

        # Mirrors ``Session._one_shot_complete``: a summary is collected whole
        # before anything reads it, so a stalled stream may be retried
        # (``replayable``). No cache key -- the summary prompt shares no prefix
        # with the episode's history and must not be keyed under it.
        return ChatRequest(
            model=self._model_spec,
            system_blocks=[SUMMARIZATION_SYSTEM_PROMPT],
            messages=[Message.user(prompt)],
            tools=[],
            tool_choice="none",
            replayable=True,
        )

    async def _stream(self, request: Any) -> tuple[str, ModelUsage, int, str, str, int | None]:
        text = ""
        usage = ModelUsage()
        cost_micros = 0
        stop_reason = "stop"
        provider_request_id = "unknown"
        context_tokens: int | None = None
        async for event in self._stream_fn(request, None):
            if event.type == "text_delta":
                text += event.delta
            elif event.type == "usage":
                usage, cost_micros = _usage_from(event.usage)
                context_tokens = _context_tokens_from(event.usage)
            elif event.type == "end":
                stop_reason = event.stop_reason or "stop"
                if event.usage is not None:
                    usage, cost_micros = _usage_from(event.usage)
                    context_tokens = _context_tokens_from(event.usage)
                # The provider's own request id is the only handle that ties a
                # bundle's model_response back to the provider's records, so it
                # is worth carrying when the wire client reports one.
                provider_request_id = _provider_request_id(event.provider_payload)
        return text, usage, cost_micros, stop_reason, provider_request_id, context_tokens


def create_provider_model_client(
    *,
    auth_store: Any,
    settings: Mapping[str, Any] | None,
    route: RouteIdentity,
    model_spec: Any,
    artifact_root: Path,
    episode_id: str,
    fallback_policy: str = "forbid",
    keep_recent_frames: int = DEFAULT_KEEP_RECENT_FRAMES,
    rebuild_every_frames: int = DEFAULT_REBUILD_EVERY_FRAMES,
    compaction: "CompactionSettings | None" = None,
) -> ProviderModelClient:
    """Build a provider-backed client from the harness's own stream function.

    Importing ``configure`` here rather than at module scope keeps the cost off
    anyone who only imports the runner contracts, and keeps the evaluation
    package's import-inertness property intact.

    The session id handed to the stream function is ALSO the provider cache
    key (``SessionStreamFn`` stamps ``prompt_cache_key`` from it), so it is
    minted per episode: ``lop-eval-<episode_id>``. One key for every episode
    would make unrelated tasks route as one prefix, which is the bug the old
    pilot shipped once.

    ``fallback_policy`` is the manifest's declared policy and it is HONOURED
    here, not merely recorded: under ``forbid`` the stream function's model
    fallback is switched off through the same ``retry.modelFallback`` setting
    an operator would use, so a provider outage fails the episode as a
    provider error instead of silently serving a different model. A served
    route that differs from the pinned one still seals ``route_changed`` (the
    runner compares every served route), so this is belt and braces: the
    fallback is sealed off, and a fallback that somehow happened is labelled.
    """

    from local_operator.model.configure import create_stream_fn

    effective: dict[str, Any] = dict(settings or {})
    if fallback_policy == "forbid":
        retry = dict(effective.get("retry") or {})
        retry["modelFallback"] = False
        effective["retry"] = retry
    session_id = f"lop-eval-{episode_id}"
    return ProviderModelClient(
        create_stream_fn(auth_store, effective, session_id=session_id),
        route=route,
        model_spec=model_spec,
        artifact_root=artifact_root,
        compaction=compaction,
        keep_recent_frames=keep_recent_frames,
        rebuild_every_frames=rebuild_every_frames,
        prompt_cache_key=session_id,
    )


def _add_usage(first: ModelUsage, second: ModelUsage) -> ModelUsage:
    return ModelUsage(
        input_tokens=first.input_tokens + second.input_tokens,
        output_tokens=first.output_tokens + second.output_tokens,
        reasoning_tokens=first.reasoning_tokens + second.reasoning_tokens,
        cache_read_tokens=first.cache_read_tokens + second.cache_read_tokens,
        cache_write_tokens=first.cache_write_tokens + second.cache_write_tokens,
    )


def _estimate_context(messages: Sequence[Any]) -> int:
    from local_operator.compaction.tokens import estimate_messages_tokens

    return estimate_messages_tokens(messages)


def _context_tokens_from(usage: Any) -> int | None:
    value = getattr(usage, "context_tokens", None)
    if value is None:
        return None
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return None


def _now_ms() -> int:
    return int(time.time() * 1000)


def _provider_request_id(provider_payload: Any) -> str:
    """Extract the provider's request id, falling back to a valid placeholder.

    ``StrictIdentifier`` forbids most punctuation, so a provider id that does
    not fit the pattern is dropped rather than allowed to fail validation on a
    path that is only recording provenance.
    """

    if not isinstance(provider_payload, Mapping):
        return "unknown"
    raw = provider_payload.get("id")
    if not isinstance(raw, str) or not raw:
        return "unknown"
    # An over-length id is NOT truncated: a shortened id is still a valid
    # StrictIdentifier, so it would be recorded as provenance while matching
    # nothing in the provider's records -- a silently wrong handle is worse than
    # an honestly absent one, which is this function's whole philosophy.
    if len(raw) > 128 or not _IDENTIFIER.fullmatch(raw):
        return "unknown"
    return raw


def _usage_from(usage: Any) -> tuple[ModelUsage, int]:
    """Copy provider counts and cost, clamped to the non-negative evidence range.

    ``reasoning_tokens`` is a SUBSET of ``output_tokens`` in the harness's
    accounting, and the evidence payloads treat it the same way, so it is
    carried across unchanged rather than added on top.

    ``usd_cost`` is the provider's OWN billing figure, which the harness treats
    as ground truth over any token-times-rate reconstruction. Dropping it made
    every reconciliation report a free episode. An unreported cost stays 0 here
    because the evidence payload has no "unknown" encoding -- the distinction
    upstream between ``None`` and a real ``0.0`` cannot be represented, and
    inventing a number would be worse than under-reporting one.
    """

    model_usage = ModelUsage(
        input_tokens=max(0, int(usage.input_tokens or 0)),
        output_tokens=max(0, int(usage.output_tokens or 0)),
        reasoning_tokens=max(0, int(usage.reasoning_tokens or 0)),
        cache_read_tokens=max(0, int(usage.cache_read_tokens or 0)),
        cache_write_tokens=max(0, int(usage.cache_write_tokens or 0)),
    )
    cost = getattr(usage, "usd_cost", None)
    cost_micros = 0 if cost is None else max(0, round(float(cost) * 1_000_000))
    return model_usage, cost_micros
