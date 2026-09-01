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
from typing import Any, Mapping, Sequence

from local_operator.evaluation.evidence.models import RouteIdentity
from local_operator.evaluation.protocol import ActionBatch, Observation
from local_operator.evaluation.runner.model import ModelDecision, ModelUsage

_SYSTEM_PROMPT = """You are operating a computer to complete one task.

Reply with a single JSON object and nothing else:

  {"actions": [ ... ]}

Each action is one of the protocol's action objects and must carry the
`observation_id` of the observation you are looking at right now.

Two actions end your turn in a special way:

* `finish` -- you believe the task is complete. The episode is scored.
* `ask_user` -- you need a human answer. THE EPISODE PAUSES: a person answers
  your question, and the next observation you see is the state after that
  answer was delivered. Do not ask a question you can resolve by acting.
"""


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
                "task_id": observation.task_id,
                "episode_id": observation.episode_id,
                "observation_id": observation.observation_id,
                "observation_sequence": observation.sequence,
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
        stop_reason = "stop"
        async for event in self._stream_fn(request, None):
            if event.type == "text_delta":
                text += event.delta
            elif event.type == "usage":
                usage = _usage_from(event.usage)
            elif event.type == "end":
                stop_reason = event.stop_reason or "stop"
                if event.usage is not None:
                    usage = _usage_from(event.usage)
        return parse_decision(
            text.strip(),
            observation,
            route=self._route,
            usage=usage,
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


def _usage_from(usage: Any) -> ModelUsage:
    """Copy provider counts, clamped to the non-negative evidence range.

    ``reasoning_tokens`` is a SUBSET of ``output_tokens`` in the harness's
    accounting, and the evidence payloads treat it the same way, so it is
    carried across unchanged rather than added on top.
    """

    return ModelUsage(
        input_tokens=max(0, int(usage.input_tokens or 0)),
        output_tokens=max(0, int(usage.output_tokens or 0)),
        reasoning_tokens=max(0, int(usage.reasoning_tokens or 0)),
        cache_read_tokens=max(0, int(usage.cache_read_tokens or 0)),
        cache_write_tokens=max(0, int(usage.cache_write_tokens or 0)),
    )
