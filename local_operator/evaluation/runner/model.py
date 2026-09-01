"""The policy boundary an episode drives, independent of any provider.

``EpisodeRunner`` needs exactly one capability from a model: turn the current
observation and the transcript so far into an :class:`ActionBatch`. Expressing
that as a Protocol rather than a concrete client is what keeps ``episode.py``
free of provider, config, and session imports -- the import-isolation test in
``tests/unit/evaluation/runner`` asserts that boundary, because an evaluation
episode must be reproducible from pinned inputs and must not inherit a user's
live session configuration.

The concrete provider-backed implementation lives in ``provider_client.py``,
which is the only runner module permitted to import ``local_operator.model``.
"""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

from pydantic import Field

from local_operator.evaluation.evidence.models import RouteIdentity
from local_operator.evaluation.protocol import ActionBatch, Observation, ProtocolModel
from local_operator.evaluation.receipts import SafeCount, StrictIdentifier


class ModelUsage(ProtocolModel):
    """Provider-reported token consumption for exactly one decision.

    These counts feed both ``model_response`` and ``usage_cost`` evidence, and
    the verifier recomputes the bundle's counters from the latter. They are
    therefore authoritative provider numbers, never estimates: a provider that
    cannot report a count must surface zero here and declare the resource
    unavailable during reconciliation instead of guessing.
    """

    input_tokens: SafeCount = 0
    output_tokens: SafeCount = 0
    reasoning_tokens: SafeCount = 0
    cache_read_tokens: SafeCount = 0
    cache_write_tokens: SafeCount = 0


class ModelDecision(ProtocolModel):
    """One model turn: the batch to execute plus its billing provenance.

    ``route`` is the route the provider actually served. The runner compares it
    against the requested route when labelling comparability, so a client that
    silently fell back must report the served route here rather than echoing
    what was asked for.
    """

    action_batch: ActionBatch
    route: RouteIdentity
    usage: ModelUsage = Field(default_factory=ModelUsage)
    cost_micros: SafeCount = 0
    stop_reason: StrictIdentifier = "stop"
    provider_request_id: StrictIdentifier = "unknown"
    tool_call_count: SafeCount = 0


@runtime_checkable
class EpisodeModelClient(Protocol):
    """Chooses the next action batch for an episode.

    ``transcript`` carries every observation already seen, oldest first, so an
    implementation can build whatever context window it needs without the
    runner taking a position on prompt construction.

    Raising is the contract for an unrecoverable provider failure: the runner
    treats it as ``ErrorPayload(category="provider")`` and finalizes the
    episode unscored on a still-live session. Internal retries therefore belong
    inside the implementation, below this boundary.
    """

    async def decide(
        self,
        observation: Observation,
        transcript: Sequence[Observation],
    ) -> ModelDecision: ...
