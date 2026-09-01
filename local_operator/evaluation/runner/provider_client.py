"""The only runner module allowed to reach into ``local_operator.model``.

``episode.py`` must stay free of provider, config, and session imports so an
episode cannot inherit the operator's live configuration. That constraint has
to break somewhere for a real run, and it breaks here, behind
:class:`~local_operator.evaluation.runner.model.EpisodeModelClient`.

Strict decision parsing also lives here rather than in the runner core. The
runner treats a malformed decision as a provider failure, so the parsing rules
and the provider that produced them stay in the same module.
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence, get_args

from local_operator.evaluation.evidence.models import RouteIdentity
from local_operator.evaluation.protocol import ActionBatch, ComputerAction, Observation
from local_operator.evaluation.runner.model import ModelDecision, ModelUsage

# The batch wire version this harness speaks; pinned here rather than taken
# from a model reply.
PROTOCOL_VERSION = "1.0"


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
    name = getattr(annotation, "__name__", None)
    if name in ("int", "str", "bool", "float"):
        return name
    args = [arg for arg in get_args(annotation) if arg is not type(None)]
    if len(args) == 1:
        return _type_name(args[0])
    return "value"


def build_system_prompt() -> str:
    """Compose the episode system prompt around the live protocol schema."""

    return f"""You are operating a computer to complete one task.

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
its batch:

* "finish" -- you believe the task is done. The episode is then scored. Its
  "status" must be one of done, failed, or infeasible, and a "reason" is
  required.
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
    )


class ProviderModelClient:
    """Drives a real provider through the session stream function.

    The stream function already owns credential resolution, model fallback and
    retry, so a failure that reaches this class is terminal and is allowed to
    propagate: the runner records it as a provider error and finalizes the
    episode unscored on a still-live session.
    """

    def __init__(
        self,
        stream_fn: Any,
        *,
        route: RouteIdentity,
        model_spec: Any,
        system_prompt: str = _SYSTEM_PROMPT,
    ) -> None:
        self._stream_fn = stream_fn
        self._route = route
        self._model_spec = model_spec
        self._system_prompt = system_prompt

    async def decide(
        self,
        observation: Observation,
        transcript: Sequence[Observation],
    ) -> ModelDecision:
        from local_operator.harness.types import ChatRequest, Message, TextContent

        # The system prompt rides in ``system_blocks`` rather than as a message
        # so the provider can place a cache breakpoint after this stable block;
        # every episode step repeats it verbatim.
        request = ChatRequest(
            model=self._model_spec,
            system_blocks=[self._system_prompt],
            messages=[
                Message(
                    role="user",
                    content=[TextContent(text=_render(observation, transcript))],
                )
            ],
            tool_choice="none",
        )
        text = ""
        usage = ModelUsage()
        cost_micros = 0
        stop_reason = "stop"
        async for event in self._stream_fn(request, None):
            if event.type == "text_delta":
                text += event.delta
            elif event.type == "usage":
                usage, cost_micros = _usage_from(event.usage)
            elif event.type == "end":
                stop_reason = event.stop_reason or "stop"
                if event.usage is not None:
                    usage, cost_micros = _usage_from(event.usage)
        return parse_decision(
            text.strip(),
            observation,
            route=self._route,
            usage=usage,
            cost_micros=cost_micros,
            stop_reason=stop_reason,
        )


def create_provider_model_client(
    *,
    auth_store: Any,
    settings: Mapping[str, Any] | None,
    route: RouteIdentity,
    model_spec: Any,
    session_id: str | None = None,
) -> ProviderModelClient:
    """Build a provider-backed client from the harness's own stream function.

    Importing ``configure`` here rather than at module scope keeps the cost off
    anyone who only imports the runner contracts, and keeps the evaluation
    package's import-inertness property intact.
    """

    from local_operator.model.configure import create_stream_fn

    return ProviderModelClient(
        create_stream_fn(auth_store, settings, session_id=session_id),
        route=route,
        model_spec=model_spec,
    )


def _render(observation: Observation, transcript: Sequence[Observation]) -> str:
    lines = [
        f"Task: {observation.task_id}",
        f"Step: {observation.sequence}",
        f"Observation ID: {observation.observation_id}",
        "",
        observation.text or "(no textual state)",
    ]
    if len(transcript) > 1:
        lines.append("")
        lines.append(f"You have seen {len(transcript)} observations so far.")
    return "\n".join(lines)


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
