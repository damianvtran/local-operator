"""The session incident must name the model that actually failed.

The reported defect: an exhausted Anthropic quota handed the turn to OpenAI,
OpenAI refused with a ``cyber_policy`` content filter, and the incident row
said ``[session incident (anthropic/claude-opus-5-5)]`` — the model the user
SELECTED, which the attempt had already failed over from. A reader who follows
that name debugs a provider that did nothing wrong and was never reached.

``journal_incident`` is the single write for that row, so it is where the head
is decided. The failing target now arrives from ``AgentEndEvent.failing_model``
(stamped by the failover walk on the error); the SELECTION arrives from the
session, which is the object that owns it — a route the session moved onto
before the walk started is invisible to the driver.
"""

from __future__ import annotations

import json

import pytest

from local_operator.harness.types import (
    AgentEndEvent,
    ModelSpec,
    StreamEndEvent,
    StreamModelEvent,
)
from local_operator.providers.failover import stream_with_failover

from .test_session import MODEL, ScriptedStream, make_session

REFUSAL = (
    "transient provider error: cyber_policy: This content was flagged for possible "
    "cybersecurity risk. If this seems wrong, try rephrasing your request. If you're "
    "doing authorized security work that requires more cyber permissive safeguards, "
    "apply for Daybreak access via https://platform.openai.com/settings/organization/"
    "status-and-access before retrying."
)


def _incident_text(tmp_path) -> str:
    """The rendered ``text`` of the newest session-incident row on disk."""
    rows = [
        json.loads(line)
        for line in (tmp_path / "sess" / "transcript.jsonl").read_text().splitlines()
        if line.strip()
    ]
    incidents = [
        row for row in rows if row.get("payload", {}).get("custom_type") == "session_incident"
    ]
    assert incidents, "no session incident was journalled"
    return incidents[-1]["payload"]["details"]["text"]


@pytest.mark.asyncio
async def test_the_incident_names_the_failing_model_not_the_selection(tmp_path):
    session = make_session(tmp_path, ScriptedStream([]), model=MODEL)
    await session.journal_incident(REFUSAL, provider="openai", model="gpt-5")

    text = _incident_text(tmp_path)
    head = text.splitlines()[0]
    assert head.startswith("[session incident (openai/gpt-5)] content-filter:"), head
    # The selection is still named — as what the attempt moved OFF, never as
    # what failed.
    assert f"after failing over from {MODEL.provider}/{MODEL.model_id}" in text
    assert "cyber_policy" in text


@pytest.mark.asyncio
async def test_the_session_owns_the_selection_half(tmp_path):
    """A walk that never hopped still gets the selection from the session.

    This is the operator's path: the quota preflight had already routed the
    session onto the fallback, so ``stream_with_failover`` saw the fallback as
    its own entry selector and had no hop to describe.
    """
    session = make_session(tmp_path, ScriptedStream([]), model=MODEL)
    await session.journal_incident(REFUSAL, provider="openai", model="gpt-5")

    text = _incident_text(tmp_path)
    assert (
        f"failed on openai/gpt-5 after failing over from {MODEL.provider}/{MODEL.model_id}" in text
    )


@pytest.mark.asyncio
async def test_an_incident_without_attribution_keeps_todays_row(tmp_path):
    """Every other incident — a cut-off, an MCP grant, a serving model's own
    failure — is unchanged: the head names the session's model."""
    session = make_session(tmp_path, ScriptedStream([]), model=MODEL)
    await session.journal_incident("429 quota exceeded")

    text = _incident_text(tmp_path)
    assert text.startswith(f"[session incident ({MODEL.provider}/{MODEL.model_id})]")
    assert "failed over" not in text


@pytest.mark.asyncio
async def test_a_same_model_failure_states_no_failover(tmp_path):
    session = make_session(tmp_path, ScriptedStream([]), model=MODEL)
    await session.journal_incident(REFUSAL, provider=MODEL.provider, model=MODEL.model_id)

    assert "failed over" not in _incident_text(tmp_path)


def test_the_end_event_carries_the_failing_model():
    """The carrier between the walk and the session, pinned.

    Additive with a ``""`` default: ``AgentEvent`` is ``extra="allow"``, so an
    old viewer keeps the field as an extra and an old runtime's event simply
    has no attribution.
    """
    assert AgentEndEvent().failing_model == ""
    assert AgentEndEvent(failing_model="openai/gpt-5").failing_model == "openai/gpt-5"


class _Auth:
    """The two methods the failover walk asks a credential store for."""

    async def get_api_key(self, provider: str, session_id=None, **kwargs) -> str | None:
        return "k"

    def rotate_sibling(self, *args, **kwargs) -> bool:
        return False


class _RoutedAwayWalk:
    """The operator's path, driven through the REAL failover walk.

    The session is on ``test/m``, but the request that reaches the provider is
    already ``openai/gpt-5`` — which is what the quota preflight does when the
    selected account is blocked, and why ``stream_with_failover`` then names
    the fallback as its OWN entry selector. Nothing here is hand-stamped: the
    provenance under test is the one the walk produces.
    """

    def __init__(self, error: BaseException) -> None:
        self.error = error

    def __call__(self, request, signal):
        routed = request.model_copy(
            update={"model": ModelSpec(provider="openai", model_id="gpt-5")}
        )
        error = self.error

        async def client_for(spec):
            class _Client:
                async def stream(self, request, api_key, oauth_access=None):
                    yield StreamModelEvent(model=spec)
                    raise error

            return _Client()

        async def gen():
            async for event in stream_with_failover(
                routed,
                _Auth(),
                {"retry": {"baseDelayMs": 1, "maxRetries": 0, "fallbackChains": {"default": []}}},
                client_for,
            ):
                yield event

        return gen()


@pytest.mark.asyncio
async def test_a_failed_turn_journals_the_refusing_model_end_to_end(tmp_path):
    """The whole path, not its pieces: error → end event → incident row.

    This is the assertion the operator's screenshot failed: the row must name
    OpenAI, and must still say which model the turn was on before it failed
    over.
    """
    from local_operator.providers.failover import (  # noqa: F401
        ProviderError,
        stream_with_failover,
    )

    error = ProviderError(
        None,
        "cyber_policy: This content was flagged for possible cybersecurity risk.",
        retryable=True,
    )
    session = make_session(tmp_path, _RoutedAwayWalk(error), model=MODEL)
    # The loop converts a provider failure into a terminal END EVENT rather
    # than raising, so ``prompt`` returns normally and the row is the record.
    await session.prompt("summarise this")

    text = _incident_text(tmp_path)
    head = text.splitlines()[0]
    assert head.startswith("[session incident (openai/gpt-5)] content-filter:"), head
    assert (
        f"failed on openai/gpt-5 after failing over from {MODEL.provider}/{MODEL.model_id}" in text
    )


class _InBandRefusalStream:
    """The real walk, with a client that ENDS the stream instead of raising.

    This is how a content filter actually arrives: ``clients.py`` composes the
    refusal prose onto the END event, so there is no exception to carry
    provenance. The client publishes NO ``StreamModelEvent`` either — that frame
    is emitted only when the context budget changed or the credential is OAuth,
    so an ordinary API-key hop produces none. Attribution therefore has to come
    from the stamp the walk puts on the end event itself (agent review round 2:
    the previous version of this test fabricated a ``StreamModelEvent`` no real
    client emits, and passed for that reason alone).
    """

    def __init__(self) -> None:
        self.refusal = StreamEndEvent(stop_reason="refusal", error=REFUSAL)

    def __call__(self, request, signal):
        routed = request.model_copy(
            update={"model": ModelSpec(provider="openai", model_id="gpt-5")}
        )
        refusal = self.refusal

        async def client_for(spec):
            class _Client:
                async def stream(self, request, api_key, oauth_access=None):
                    yield refusal

            return _Client()

        async def gen():
            async for event in stream_with_failover(
                routed,
                _Auth(),
                {"retry": {"baseDelayMs": 1, "maxRetries": 0, "fallbackChains": {"default": []}}},
                client_for,
            ):
                yield event

        return gen()


@pytest.mark.asyncio
async def test_an_in_band_refusal_names_the_serving_model(tmp_path):
    """No exception, no raise — and the refusal still names OpenAI.

    Covered separately from the raising path because the two are different
    carriers: one is an exception the walk stamps, the other is an end event
    the loop has to attribute from the serving spec.
    """
    session = make_session(tmp_path, _InBandRefusalStream(), model=MODEL)
    await session.prompt("summarise this content")

    text = _incident_text(tmp_path)
    head = text.splitlines()[0]
    assert head.startswith("[session incident (openai/gpt-5)] content-filter:"), head
    assert f"after failing over from {MODEL.provider}/{MODEL.model_id}" in text
