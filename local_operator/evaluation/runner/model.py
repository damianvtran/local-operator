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


class CompactionRecord(ProtocolModel):
    """A context rebuild the client performed before this decision's request.

    Reported so the runner can DECLARE it in the bundle (a ``context_compaction``
    event) rather than leave it inferable from a message-count drop. The
    summarization call that produced ``summary_text`` was a billed provider
    call; its usage and cost are folded into the owning ``ModelDecision`` so
    the bundle's counters remain a pure sum of usage events.
    """

    strategy: StrictIdentifier
    tokens_before: SafeCount
    tokens_after: SafeCount
    frames_dropped: SafeCount
    messages_before: SafeCount
    messages_after: SafeCount
    summary_text: str | None = None


class ModelDecision(ProtocolModel):
    """One model turn: the batch to execute plus its billing provenance.

    ``route`` is the route the provider actually served. The runner compares it
    against the requested route when labelling comparability, so a client that
    silently fell back must report the served route here rather than echoing
    what was asked for.

    ``usage``/``cost_micros`` cover EVERY provider call the decision cost,
    including a compaction summary made on the way to it; ``compaction`` is set
    when one happened so the runner can declare it before the request event.
    """

    action_batch: ActionBatch
    route: RouteIdentity
    usage: ModelUsage = Field(default_factory=ModelUsage)
    cost_micros: SafeCount = 0
    stop_reason: StrictIdentifier = "stop"
    provider_request_id: StrictIdentifier = "unknown"
    tool_call_count: SafeCount = 0
    prompt_cache_key: StrictIdentifier | None = None
    context_tokens: SafeCount | None = None
    compaction: CompactionRecord | None = None


class EpisodeTurn(ProtocolModel):
    """One completed (or in-progress) turn of the episode, protocol-typed.

    ``observation`` is what the model saw; ``batch`` is what it decided, which
    is ``None`` for the turn currently being decided and for a turn whose batch
    was terminal; ``ask_answer`` is the answer delivered for an ``ask_user``
    batch. The runner appends these and never renders them — how a turn becomes
    a provider message (frames as image blocks, the batch replayed verbatim as
    the assistant's own words) is the client's business, which keeps the runner
    core free of provider vocabulary.
    """

    observation: Observation
    batch: ActionBatch | None = None
    ask_answer: str | None = None


@runtime_checkable
class EpisodeModelClient(Protocol):
    """Chooses the next action batch for an episode.

    ``history`` carries every turn already taken, oldest first, each with the
    observation the model saw and the batch it chose, so an implementation can
    build a real append-only conversation without the runner taking a position
    on prompt construction. The current ``observation`` is the last entry's
    observation too (its ``batch`` is still ``None``).

    Raising is the contract for an unrecoverable provider failure: the runner
    treats it as ``ErrorPayload(category="provider")`` and finalizes the
    episode unscored on a still-live session. Internal retries therefore belong
    inside the implementation, below this boundary.
    """

    async def decide(
        self,
        observation: Observation,
        history: Sequence[EpisodeTurn],
    ) -> ModelDecision: ...
