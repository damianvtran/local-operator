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

from local_operator.evaluation.action_surface import ActionSurface
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
    # Only validated visible model output, never provider reasoning. None keeps
    # old clients/history on their byte-identical canonical-action replay path.
    public_reply: str | None = None
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
    # The runner attaches the evidence-redacted reply to this same observation;
    # the context builder must not reconstruct it as actions and lose its facts.
    public_reply: str | None = None
    ask_answer: str | None = None


class DecisionRejected(Exception):
    """The provider answered, and was billed, but the reply is not a usable batch.

    This is the MODEL's error, not the provider's: the request went through,
    tokens were spent, and what came back failed strict decision parsing (not
    JSON, wrong shape, a stale observation id, a ``frame_id`` the observation
    does not carry, a coordinate outside the frame). It is raised INSTEAD of a
    :class:`ModelDecision` so the runner can still record the attempt honestly
    -- every field a decision would have carried for the bundle's
    request/response/usage triple is here -- and then ask again.

    The first paid OSWorld episode ended on exactly this: the model named
    ``frame_id "1"`` where the adapter had published ``"screen"``, the reply
    was classified as a provider failure, and the episode was sealed unscored
    after a single call. A rejected reply is recoverable in a way a dead
    provider is not, which is why it has its own type and its own path.

    ``diagnostic`` is the parse error, phrased so it can be shown back to the
    model verbatim. An implementation that raises this is expected to have
    folded the rejection into its own context first (the bad reply, then the
    diagnostic as the next user turn) so that the runner's re-call of
    :meth:`EpisodeModelClient.decide` for the same observation is a corrective
    re-prompt rather than a blind replay.
    """

    def __init__(
        self,
        diagnostic: str,
        *,
        reply: str | None = None,
        route: RouteIdentity | None = None,
        usage: ModelUsage | None = None,
        cost_micros: int = 0,
        stop_reason: str = "stop",
        provider_request_id: str = "unknown",
        tool_call_count: int = 0,
        prompt_cache_key: str | None = None,
        context_tokens: int | None = None,
        compaction: CompactionRecord | None = None,
    ) -> None:
        super().__init__(diagnostic)
        self.diagnostic = diagnostic
        # WHAT the model actually said, not just why it was refused. The
        # diagnostic alone cannot answer the question a post-mortem asks --
        # "was it trying to type?" -- because a Pydantic error names the
        # fields it disliked and not the intent behind them. A real paid
        # episode's three rejections were only reconstructible as far as
        # "something with a `key` field" and "trailing junk after the JSON";
        # the replies themselves were discarded, so the failure class could
        # not be diagnosed without paying for the run again.
        self.reply = reply
        # The served route matters even for a rejected reply: a fallback that
        # answered badly still moved the run off its pinned route.
        self.route = route
        self.usage = usage or ModelUsage()
        self.cost_micros = cost_micros
        self.stop_reason = stop_reason
        self.provider_request_id = provider_request_id
        self.tool_call_count = tool_call_count
        self.prompt_cache_key = prompt_cache_key
        self.context_tokens = context_tokens
        self.compaction = compaction


@runtime_checkable
class EpisodeModelClient(Protocol):
    """Chooses the next action batch for an episode.

    ``history`` carries every turn already taken, oldest first, each with the
    observation the model saw and the batch it chose, so an implementation can
    build a real append-only conversation without the runner taking a position
    on prompt construction. The current ``observation`` is the last entry's
    observation too (its ``batch`` is still ``None``).

    Two failure contracts, deliberately distinct:

    * Raising :class:`DecisionRejected` means the call was billed but the
      reply was unusable. The runner records the attempt (request, response,
      usage, and a retryable ``error`` event) and calls ``decide`` AGAIN with
      the same observation and history, up to ``EpisodeConfig.max_decision_retries``
      times; only when that bound is spent does the episode end, as a MODEL
      failure. The implementation owns making the re-call corrective.
    * Raising anything else is the contract for an unrecoverable provider
      failure: the runner treats it as ``ErrorPayload(category="provider")``
      and finalizes the episode unscored on a still-live session. Internal
      retries for transport faults therefore belong inside the implementation,
      below this boundary.
    """

    async def decide(
        self,
        observation: Observation,
        history: Sequence[EpisodeTurn],
        *,
        action_surface: ActionSurface,
    ) -> ModelDecision: ...
