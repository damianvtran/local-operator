"""Offline behavioural coverage for the provider-backed model client.

This module needs no credentials and spends no tokens: it drives the real
``ProviderModelClient`` against a scripted async stream that yields the same
event shapes ``local_operator.harness.types`` defines. That is enough to prove
the thing the runner actually depends on -- a provider reply becomes a
protocol-valid ``ActionBatch`` -- which is the property that was previously
asserted nowhere.

Its absence is what let a dead provider path ship: the module type-checked and
imported cleanly while being incapable of parsing any real reply.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, AsyncIterator, Callable

import pytest

from local_operator.evaluation.adapters.api import observation_content_id
from local_operator.evaluation.evidence.models import RouteIdentity
from local_operator.evaluation.protocol import (
    ActionBatch,
    FinishAction,
    Observation,
    TypeAction,
)
from local_operator.evaluation.runner.model import DecisionRejected, EpisodeTurn
from local_operator.evaluation.runner.provider_client import (
    MAX_REJECTED_REPLY_CHARS,
    MAX_TOLERATED_TRAILING_CHARS,
    DecisionParseError,
    ProviderModelClient,
    ProviderStreamAbortedError,
    build_system_prompt,
    parse_decision,
)
from local_operator.harness.types import (
    ImageContent,
    ModelSpec,
    StreamEndEvent,
    StreamTextDelta,
    StreamUsageEvent,
    TextContent,
    Usage,
)

ROUTE = RouteIdentity(provider_id="provider", route_id="route", model_id="model")


def observation(sequence: int = 0) -> Observation:
    provisional = Observation(
        task_id="task-1",
        episode_id="episode-1",
        sequence=sequence,
        observation_id="provisional",
        text="a screen",
    )
    return provisional.model_copy(update={"observation_id": observation_content_id(provisional)})


def finish_payload(current: Observation) -> str:
    return json.dumps(
        {
            "actions": [
                {
                    "kind": "finish",
                    "observation_id": current.observation_id,
                    "status": "done",
                    "reason": "the task is complete",
                }
            ]
        }
    )


def type_payload(current: Observation) -> str:
    """A non-terminal, frameless action.

    Pointer actions are deliberately excluded here: ``validate_for`` resolves
    ``frame_id`` against the observation's frames, so covering a click would
    mean asserting frame plumbing rather than decision parsing.
    """

    return json.dumps(
        {
            "actions": [
                {
                    "kind": "type",
                    "observation_id": current.observation_id,
                    "text": "hello",
                }
            ]
        }
    )


class ScriptedStream:
    """Stands in for ``SessionStreamFn`` with a fixed event script."""

    def __init__(
        self,
        text: str,
        *,
        usage: Usage | None = None,
        stop_reason: str = "stop",
        chunk: int = 7,
        provider_payload: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        self.text = text
        self.usage = usage
        self.stop_reason = stop_reason
        self.chunk = chunk
        self.provider_payload = provider_payload
        # The provider's own words about an abnormal end. Scripted here because
        # the wire clients are the only layer that sees the real terminal
        # marker, so a fake that cannot carry one cannot exercise the refusal
        # path at all -- which is how that path shipped unhandled.
        self.error = error
        self.requests: list[Any] = []

    def __call__(self, request: Any, signal: Any) -> AsyncIterator[Any]:
        self.requests.append(request)
        return self._events()

    async def _events(self) -> AsyncIterator[Any]:
        # Delivered in fragments because a provider streams text in pieces; a
        # client that reads only the first delta would still pass a whole-string
        # fake and then fail against every real provider.
        for start in range(0, len(self.text), self.chunk):
            yield StreamTextDelta(delta=self.text[start : start + self.chunk])
        if self.usage is not None:
            yield StreamUsageEvent(usage=self.usage)
        yield StreamEndEvent(
            stop_reason=self.stop_reason,
            usage=self.usage,
            provider_payload=self.provider_payload,
            error=self.error,
        )


def _client(stream: Any, tmp_path: Path | None = None, **overrides: Any) -> ProviderModelClient:
    spec = ModelSpec(provider="provider", model_id="model")
    root = tmp_path if tmp_path is not None else Path("/nonexistent-artifact-root")
    return ProviderModelClient(
        stream, route=ROUTE, model_spec=spec, artifact_root=root, **overrides
    )


def _turns(*observations: Observation) -> list[EpisodeTurn]:
    """A history whose last turn is the undecided current one."""

    return [EpisodeTurn(observation=current) for current in observations]


# ---------------------------------------------------------------------------
# parse_decision
# ---------------------------------------------------------------------------


def test_parse_decision_accepts_a_protocol_correct_finish() -> None:
    current = observation()

    decision = parse_decision(finish_payload(current), current, route=ROUTE)

    assert isinstance(decision.action_batch, ActionBatch)
    assert decision.action_batch.protocol_version == "1.0"
    assert len(decision.action_batch.actions) == 1
    assert isinstance(decision.action_batch.actions[0], FinishAction)
    # The batch must survive the same validation the adapter boundary applies.
    decision.action_batch.validate_for(current)


def test_parse_decision_accepts_a_protocol_correct_non_terminal_action() -> None:
    current = observation()

    decision = parse_decision(type_payload(current), current, route=ROUTE)

    assert isinstance(decision.action_batch.actions[0], TypeAction)
    decision.action_batch.validate_for(current)


def test_parse_decision_binds_the_batch_to_the_current_observation() -> None:
    current = observation()

    decision = parse_decision(finish_payload(current), current, route=ROUTE)

    batch = decision.action_batch
    assert batch.task_id == current.task_id
    assert batch.episode_id == current.episode_id
    assert batch.observation_id == current.observation_id


def test_parse_decision_rejects_a_batch_bound_to_a_stale_observation() -> None:
    """A decision about a screen the environment has moved past is not usable."""

    stale = observation(0)
    current = observation(1)

    with pytest.raises(DecisionParseError):
        parse_decision(finish_payload(stale), current, route=ROUTE)


@pytest.mark.parametrize(
    "payload",
    [
        "not json at all",
        '{"actions": []}',
        '{"actions": "click"}',
        '{"no_actions_key": true}',
        "[]",
    ],
)
def test_parse_decision_rejects_malformed_replies(payload: str) -> None:
    with pytest.raises(DecisionParseError):
        parse_decision(payload, observation(), route=ROUTE)


@pytest.mark.parametrize(
    "junk",
    [
        # Verbatim from sealed bundle ep-e46c789ca818, which paid for this
        # three times in one episode.
        "原始内容",
        " Hope that helps!",
        "\n\n```",
        "\nLet me know if you want me to continue.",
    ],
)
def test_parse_decision_tolerates_trailing_junk_after_a_complete_value(junk: str) -> None:
    """A complete batch is not made unusable by what the model appended to it.

    The decision is already unambiguous where the junk starts, so discarding a
    billed turn over it buys nothing.
    """

    current = observation()

    decision = parse_decision(type_payload(current) + junk, current, route=ROUTE)

    action = decision.action_batch.actions[0]
    assert isinstance(action, TypeAction)
    # The action is what the model plainly intended, not a salvaged fragment.
    assert action.text == "hello"
    decision.action_batch.validate_for(current)


def test_parse_decision_reports_tolerated_trailing_junk(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Tolerated is not the same as silent: the remainder stays observable."""

    current = observation()

    with caplog.at_level(logging.WARNING):
        parse_decision(type_payload(current) + "原始内容", current, route=ROUTE)

    assert "原始内容" in caplog.text
    assert "trailing text" in caplog.text


def test_parse_decision_bounds_the_reported_trailing_junk(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A runaway remainder must not turn a log line into a wall of prose."""

    current = observation()
    junk = "x" * (MAX_TOLERATED_TRAILING_CHARS + 500)

    with caplog.at_level(logging.WARNING):
        parse_decision(type_payload(current) + junk, current, route=ROUTE)

    assert "[...]" in caplog.text
    # The full remainder is counted but never quoted in full.
    assert str(len(junk)) in caplog.text
    assert "x" * (MAX_TOLERATED_TRAILING_CHARS + 1) not in caplog.text


@pytest.mark.parametrize(
    "separator",
    [
        # Direct adjacency is the one shape a real model is LEAST likely to
        # emit, and covering only it is what let a separator-bypass hide here.
        "",
        " ",
        ",",
        "\n",
        "原始内容",
        "\n\nOops, that was wrong. The correct batch is:\n",
        "\n```\n```json\n",
        "x" * 1000,
    ],
)
def test_parse_decision_rejects_a_superseding_batch_behind_any_separator(
    separator: str,
) -> None:
    """Which batch did the model mean? Guessing could execute the wrong one.

    The ambiguity does not depend on the two objects being adjacent, so
    neither may the guard: a second batch for the SAME observation is a
    competing decision wherever it sits in the remainder.
    """

    current = observation()
    payload = type_payload(current) + separator + finish_payload(current)

    with pytest.raises(DecisionParseError, match="second action batch"):
        parse_decision(payload, current, route=ROUTE)


def test_parse_decision_tolerates_a_quoted_batch_for_another_observation() -> None:
    """The shape the real bundle actually emits, and why the guard keys on the
    observation id rather than on JSON-ness.

    A model that quotes the harness's own ``The rejected reply was: {...}``
    feedback back at itself carries a well-formed batch in its prose. That is
    HISTORY -- it names an OLDER observation -- so it cannot supersede the
    decision just parsed, and ``ActionBatch`` would refuse it as stale anyway.
    Rejecting on it would discard the very turns this tolerance exists to save.
    """

    current = observation()
    stale = observation(1)
    quoted = json.dumps(
        {"actions": [{"kind": "key", "observation_id": stale.observation_id, "keys": ["ctrl"]}]}
    )
    payload = type_payload(current) + "\nThe rejected reply was:\n  " + quoted

    decision = parse_decision(payload, current, route=ROUTE)

    action = decision.action_batch.actions[0]
    assert isinstance(action, TypeAction)
    assert action.text == "hello"


@pytest.mark.parametrize("whitespace", ["  ", "\n", "\t", "\r\n"])
def test_parse_decision_accepts_leading_whitespace(whitespace: str) -> None:
    """Parity with the ``json.loads`` this replaced, which skipped leading
    whitespace where ``raw_decode`` does not.

    Pinned inside the helper rather than left to the caller's ``.strip()``:
    this is a general entry point, and a second caller that forgot to strip
    would lose a turn to a leading newline.
    """

    current = observation()

    decision = parse_decision(whitespace + type_payload(current), current, route=ROUTE)

    assert isinstance(decision.action_batch.actions[0], TypeAction)


@pytest.mark.parametrize(
    "junk",
    [
        "\ntrue story, that worked",
        "\nnull hypothesis rejected",
        "\n42 windows were listed",
        '\n"that should do it"',
        "\n{note: see the window list above}",
    ],
)
def test_parse_decision_tolerates_prose_that_opens_like_json(junk: str) -> None:
    """Only a second OBJECT can be a competing decision.

    Asking the broader "does the remainder parse as any JSON value?" would
    class prose beginning with a JSON literal as a second batch and reject the
    very turn the tolerance exists to save. A brace that does not open a
    complete object is prose too.
    """

    current = observation()

    decision = parse_decision(type_payload(current) + junk, current, route=ROUTE)

    action = decision.action_batch.actions[0]
    assert isinstance(action, TypeAction)
    assert action.text == "hello"


def test_parse_decision_rejects_leading_junk_before_the_value() -> None:
    """Skipping forward to the first brace means guessing where the value
    starts, and a preamble containing a brace makes that guess wrong silently
    -- executing a DIFFERENT batch than was sent."""

    current = observation()

    with pytest.raises(DecisionParseError, match="not valid JSON"):
        parse_decision("Sure, here you go: " + type_payload(current), current, route=ROUTE)


def test_parse_decision_still_rejects_a_truncated_value() -> None:
    """Tolerance is for what follows a COMPLETE value, never for an incomplete
    one: a batch cut off mid-emission is not a decision."""

    current = observation()

    with pytest.raises(DecisionParseError, match="not valid JSON"):
        parse_decision(type_payload(current)[:-5], current, route=ROUTE)


def test_parse_decision_rejects_a_wrong_discriminator_key() -> None:
    """A model emitting ``type`` instead of ``kind`` must fail loudly."""

    current = observation()
    payload = json.dumps(
        {
            "actions": [
                {
                    "type": "finish",
                    "observation_id": current.observation_id,
                    "status": "done",
                    "reason": "done",
                }
            ]
        }
    )

    with pytest.raises(DecisionParseError):
        parse_decision(payload, current, route=ROUTE)


# ---------------------------------------------------------------------------
# ProviderModelClient
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_client_turns_a_streamed_reply_into_a_valid_action_batch() -> None:
    current = observation()
    stream = ScriptedStream(finish_payload(current))

    decision = await _client(stream).decide(current, _turns(current))

    assert isinstance(decision.action_batch, ActionBatch)
    decision.action_batch.validate_for(current)
    assert decision.route == ROUTE


@pytest.mark.asyncio
async def test_client_sends_the_system_prompt_as_a_cacheable_block() -> None:
    current = observation()
    stream = ScriptedStream(finish_payload(current))

    await _client(stream).decide(current, _turns(current))

    request = stream.requests[0]
    assert len(request.system_blocks) == 1
    assert request.messages[0].role == "user"
    # The episode drives the environment through the protocol, not through
    # harness tools, so the provider must not be offered any.
    assert request.tool_choice == "none"


@pytest.mark.asyncio
async def test_system_prompt_states_the_schema_the_protocol_enforces() -> None:
    """An under-specified prompt is a correctness defect: a parse failure is terminal."""

    current = observation()
    stream = ScriptedStream(finish_payload(current))

    await _client(stream).decide(current, _turns(current))

    prompt = stream.requests[0].system_blocks[0]
    # The discriminator key, and every finish literal the protocol accepts.
    assert '"kind"' in prompt
    for status in ("done", "failed", "infeasible"):
        assert status in prompt
    assert "reason" in prompt
    # Derived from the protocol models, so a new action kind cannot drift out.
    for kind in ("click", "type", "key", "scroll", "wait", "finish", "ask_user"):
        assert kind in prompt
    # A sequence field must read as an array. Told only "value", a model emits
    # "ctrl+c" for KeyAction.keys, which is a hard non-retryable parse failure.
    # The name is QUOTED because a bare one reads as a label, not a JSON key:
    # a real episode answered the bare form with the singular "key".
    assert '"keys": [str, ...]' in prompt


def test_prompt_does_not_restate_literals_it_already_derives() -> None:
    """Hand-written literals drift from the derived block and contradict it."""

    prompt = build_system_prompt()

    # The derived schema line owns the finish literals; the prose below must
    # explain what the actions MEAN without re-listing their values.
    assert prompt.count("done|failed|infeasible") == 1
    assert "must be one of done, failed, or infeasible" not in prompt


def test_key_action_array_parses_where_a_joined_string_does_not() -> None:
    """The distinction the prompt now states is real and load-bearing."""

    current = observation()

    def payload(keys: Any) -> str:
        return json.dumps(
            {
                "actions": [
                    {
                        "kind": "key",
                        "observation_id": current.observation_id,
                        "keys": keys,
                    }
                ]
            }
        )

    decision = parse_decision(payload(["ctrl", "c"]), current, route=ROUTE)
    decision.action_batch.validate_for(current)

    with pytest.raises(DecisionParseError):
        parse_decision(payload("ctrl+c"), current, route=ROUTE)


@pytest.mark.asyncio
async def test_client_carries_provider_usage_and_cost_into_the_decision() -> None:
    """Cost is authoritative provider data; dropping it reports every run as free."""

    current = observation()
    usage = Usage(
        input_tokens=1234,
        output_tokens=56,
        reasoning_tokens=7,
        cache_read_tokens=8,
        cache_write_tokens=9,
        usd_cost=0.0421,
    )
    stream = ScriptedStream(finish_payload(current), usage=usage)

    decision = await _client(stream).decide(current, _turns(current))

    assert decision.usage.input_tokens == 1234
    assert decision.usage.output_tokens == 56
    assert decision.usage.reasoning_tokens == 7
    assert decision.usage.cache_read_tokens == 8
    assert decision.usage.cache_write_tokens == 9
    assert decision.cost_micros == 42100


@pytest.mark.asyncio
async def test_unreported_cost_is_zero_rather_than_a_guess() -> None:
    current = observation()
    stream = ScriptedStream(finish_payload(current), usage=Usage(input_tokens=5, output_tokens=5))

    decision = await _client(stream).decide(current, _turns(current))

    assert decision.cost_micros == 0


@pytest.mark.asyncio
async def test_client_reports_the_provider_stop_reason() -> None:
    current = observation()
    stream = ScriptedStream(finish_payload(current), stop_reason="length")

    decision = await _client(stream).decide(current, _turns(current))

    assert decision.stop_reason == "length"


@pytest.mark.asyncio
async def test_client_raises_on_an_unparseable_reply() -> None:
    """An unusable reply is a billed MODEL error, surfaced as ``DecisionRejected``
    with the call's provenance so the runner can record the attempt and
    re-prompt -- never a provider failure, and never silently swallowed."""

    current = observation()
    stream = ScriptedStream(
        "I'm afraid I can't do that.",
        usage=Usage(input_tokens=11, output_tokens=7, usd_cost=0.002),
        provider_payload={"id": "chatcmpl-REJ"},
    )

    with pytest.raises(DecisionRejected) as info:
        await _client(stream).decide(current, _turns(current))

    rejected = info.value
    assert isinstance(rejected.__cause__, DecisionParseError)
    assert rejected.usage.input_tokens == 11 and rejected.usage.output_tokens == 7
    assert rejected.cost_micros == 2000
    assert rejected.provider_request_id == "chatcmpl-REJ"
    assert rejected.route == ROUTE
    assert rejected.diagnostic.startswith("Your previous reply was rejected:")


# ---------------------------------------------------------------------------
# Abnormal stream ends. A provider that refused (or died mid-stream) without
# producing content is NOT a model that replied badly: there is no reply to
# correct, so re-prompting for better JSON cannot converge and the failure
# belongs to the provider. Observed on a paid run whose three refusals were
# recorded as "decision is not valid JSON: Unterminated string".
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_client_attributes_a_refusal_with_prose_to_the_provider() -> None:
    current = observation()
    stream = ScriptedStream(
        "",
        stop_reason="refusal",
        usage=Usage(input_tokens=11, output_tokens=0),
        error="model refused: I can't help with that (stop_reason=refusal)",
    )

    with pytest.raises(ProviderStreamAbortedError) as info:
        await _client(stream).decide(current, _turns(current))

    # The provider's own marker reaches the diagnostic verbatim: it is the only
    # thing that says WHICH mechanism fired, and the bundle is unreadable
    # without it.
    assert "stop_reason=refusal" in str(info.value)
    assert "I can't help with that" in str(info.value)


@pytest.mark.asyncio
async def test_client_attributes_a_wordless_zero_token_refusal_to_the_provider() -> None:
    """The exact observed case: ``refusal``, no prose, zero tokens both ways.

    Nothing was streamed and nothing was billed, so there is no model output to
    blame and no reply a correction prompt could repair.
    """

    current = observation()
    stream = ScriptedStream(
        "",
        stop_reason="refusal",
        usage=Usage(input_tokens=0, output_tokens=0),
    )

    with pytest.raises(ProviderStreamAbortedError) as info:
        await _client(stream).decide(current, _turns(current))

    # Never the JSON diagnosis the old path produced against an empty string.
    assert "JSON" not in str(info.value)
    assert "refusal" in str(info.value)


@pytest.mark.asyncio
async def test_client_attributes_a_wordless_mid_stream_error_to_the_provider() -> None:
    current = observation()
    stream = ScriptedStream("   \n  ", stop_reason="error")

    with pytest.raises(ProviderStreamAbortedError):
        await _client(stream).decide(current, _turns(current))


@pytest.mark.asyncio
async def test_client_still_rejects_a_reply_the_provider_cut_short() -> None:
    """An abnormal end AFTER real text is the model's output, not an outage.

    A safety stop that truncates a partially-streamed batch leaves bytes the
    model actually wrote, so it stays on the correctable rejection path where
    the runner can feed the defect back and re-prompt.
    """

    current = observation()
    stream = ScriptedStream(
        '{"actions": [{"kind": "click", "observation',
        stop_reason="refusal",
        usage=Usage(input_tokens=12, output_tokens=9),
        error="model refused and cut the reply short (stop_reason=refusal)",
    )

    with pytest.raises(DecisionRejected) as info:
        await _client(stream).decide(current, _turns(current))

    assert isinstance(info.value.__cause__, DecisionParseError)
    assert info.value.stop_reason == "refusal"


@pytest.mark.asyncio
async def test_client_attributes_a_normal_stop_carrying_an_error_to_the_provider() -> None:
    """A NORMAL stop and a set ``error`` together, which only the first half of
    the guard's condition catches.

    Real and reachable: a provider whose content filter fires after it has
    committed to a normal terminal marker reports ``stop`` and puts the reason
    in ``error``, so the stop alone says the turn was fine while the stream in
    fact delivered nothing. Dropping the ``stream_error or`` half of the
    condition leaves every other case in this file passing, so without this
    test that half is unpinned and the empty reply goes back to being reported
    as bad JSON.
    """

    current = observation()
    stream = ScriptedStream(
        "",
        stop_reason="stop",
        usage=Usage(input_tokens=48_000, output_tokens=0),
        error="content filter blocked the response (stop_reason=content_filter)",
    )

    with pytest.raises(ProviderStreamAbortedError) as info:
        await _client(stream).decide(current, _turns(current))

    assert "content filter blocked the response" in str(info.value)
    # The normal marker is reported as-is; the abort is justified by the error,
    # not by rewriting what the provider said it did.
    assert info.value.stop_reason == "stop"


@pytest.mark.asyncio
async def test_client_treats_a_length_end_with_no_text_as_a_correctable_rejection() -> None:
    """``length`` is in the allow-list, and its membership is load-bearing.

    A token cap that truncates before any parseable JSON is a MODEL problem the
    runner can correct by re-prompting -- not an outage. Dropping ``length``
    from ``_NORMAL_CONTENT_STOPS`` would convert every legitimate cap-truncated
    turn into a sealed unscored infrastructure failure, and no other test in
    this file notices, because the allow-list is otherwise only exercised
    through ``stop``.
    """

    current = observation()
    stream = ScriptedStream(
        "",
        stop_reason="length",
        usage=Usage(input_tokens=48_000, output_tokens=4_096),
    )

    with pytest.raises(DecisionRejected) as info:
        await _client(stream).decide(current, _turns(current))

    assert info.value.stop_reason == "length"
    assert info.value.diagnostic.startswith("Your previous reply was rejected:")


@pytest.mark.asyncio
async def test_client_does_not_normalize_an_absent_stop_marker_into_a_normal_stop() -> None:
    """An empty ``stop_reason`` is an ABSENT marker, not a normal content stop.

    ``event.stop_reason or "stop"`` coerced it into one, punching a fail-OPEN
    hole through an allow-list built to fail closed: a stream that said nothing
    about how it ended took the parse path and produced exactly the "not valid
    JSON" misdiagnosis this guard removes.
    """

    current = observation()
    stream = ScriptedStream("", stop_reason="", usage=Usage(input_tokens=48_000, output_tokens=0))

    with pytest.raises(ProviderStreamAbortedError) as info:
        await _client(stream).decide(current, _turns(current))

    assert "JSON" not in str(info.value)
    # Recorded verbatim rather than laundered into a stop the provider never
    # sent, and a valid ``StrictIdentifier`` so the bundle can carry it.
    assert info.value.stop_reason == "unspecified"


@pytest.mark.asyncio
async def test_an_aborted_stream_carries_the_usage_the_refused_turn_was_billed() -> None:
    """A refusal is unanswered, not unbilled.

    The provider read the whole prompt before declining, so the episode owes
    for those input tokens. Carrying nothing meant a refused episode sealed
    with no usage event at all and reported that zero as a MEASURED spend.
    """

    current = observation()
    stream = ScriptedStream(
        "",
        stop_reason="refusal",
        usage=Usage(input_tokens=48_000, output_tokens=0),
        provider_payload={"id": "req-refused-1"},
        error="model refused: I can't help with that (stop_reason=refusal)",
    )

    with pytest.raises(ProviderStreamAbortedError) as info:
        await _client(stream).decide(current, _turns(current))

    assert info.value.usage.input_tokens == 48_000
    assert info.value.usage.output_tokens == 0
    assert info.value.route == ROUTE
    assert info.value.stop_reason == "refusal"
    assert info.value.provider_request_id == "req-refused-1"


@pytest.mark.asyncio
async def test_client_still_treats_ordinary_malformed_json_as_correctable() -> None:
    """Regression guard on the fix's blast radius.

    A normal ``stop`` carrying an unparseable reply must keep raising
    ``DecisionRejected`` -- widening the provider path to swallow it would
    convert every recoverable formatting mistake into a sealed unscored
    episode.
    """

    current = observation()
    stream = ScriptedStream(
        '{"actions": [',
        usage=Usage(input_tokens=11, output_tokens=4),
    )

    with pytest.raises(DecisionRejected) as info:
        await _client(stream).decide(current, _turns(current))

    assert info.value.stop_reason == "stop"
    assert info.value.diagnostic.startswith("Your previous reply was rejected:")


# ---------------------------------------------------------------------------
# Frame-id contract: the model is told which frame ids exist, and a wrong one
# is fed back rather than ending the episode (the first paid OSWorld episode).
# ---------------------------------------------------------------------------


def _click_payload(current: Observation, frame_id: str, x: int = 0, y: int = 0) -> str:
    return json.dumps(
        {
            "actions": [
                {
                    "kind": "click",
                    "observation_id": current.observation_id,
                    "frame_id": frame_id,
                    "x": x,
                    "y": y,
                }
            ]
        }
    )


@pytest.mark.asyncio
async def test_rendered_observation_names_its_frame_ids_and_geometry(tmp_path: Path) -> None:
    """The adapter chooses the frame ids (OSWorld: ``screen``; here: ``frame-N``)
    and ``validate_for`` refuses any other, so the rendered text MUST list them.
    The first paid episode's model was never told and guessed ``"1"``."""

    stream = RecordingStream(_wait_reply)
    client = _client(stream, tmp_path)
    current = _framed_observation(tmp_path, 4)

    await client.decide(current, _turns(current))

    text = stream.requests[0].messages[0].content[0].text
    assert "Frames: frame-4 (1x1)" in text
    # The system prompt states the constraint the observation line serves.
    prompt = stream.requests[0].system_blocks[0]
    assert '"Frames:"' in prompt
    assert "MUST use one of the ids named on the CURRENT observation" in prompt


@pytest.mark.asyncio
async def test_frameless_observation_renders_no_frames_rather_than_nothing() -> None:
    current = observation()
    stream = ScriptedStream(finish_payload(current))

    await _client(stream).decide(current, _turns(current))

    assert "Frames: none" in stream.requests[0].messages[0].content[0].text


@pytest.mark.asyncio
async def test_unknown_frame_id_is_fed_back_and_the_corrected_reply_proceeds(
    tmp_path: Path,
) -> None:
    """The exact defect from bundle ep-6ea01a117eee: ``frame_id "1"`` against a
    published ``frame-0``. The first call raises ``DecisionRejected``; the
    client has ALREADY appended the bad reply and a correction naming the
    valid ids, so the runner's re-call for the same observation is corrective
    and the corrected click parses."""

    current = _framed_observation(tmp_path, 0)
    replies = iter([_click_payload(current, "1"), _click_payload(current, "frame-0")])
    stream = RecordingStream(lambda _message: next(replies))
    client = _client(stream, tmp_path)
    history = _turns(current)

    with pytest.raises(DecisionRejected) as info:
        await client.decide(current, history)
    assert "unknown frame_id '1'" in info.value.diagnostic

    decision = await client.decide(current, history)

    assert decision.action_batch.actions[0].frame_id == "frame-0"  # type: ignore[union-attr]
    # The retry request is the original observation, the model's own bad
    # reply, then the correction -- appended, never rewritten.
    retry = stream.requests[1]
    assert _shape(retry) == [
        ("user", ("text", "image")),
        ("assistant", ("text",)),
        ("user", ("text",)),
    ]
    assert retry.messages[0] is stream.requests[0].messages[0]
    assert json.loads(retry.messages[1].content[0].text)["actions"][0]["frame_id"] == "1"
    correction = retry.messages[2].content[0].text
    assert correction.startswith("Your previous reply was rejected:")
    assert "Frames: frame-0 (1x1)" in correction
    assert current.observation_id in correction


@pytest.mark.asyncio
async def test_a_corrected_turn_closes_after_the_correction_in_the_next_request(
    tmp_path: Path,
) -> None:
    """Once the runner closes the corrected turn, the accepted batch is
    replayed AFTER the rejection pair, so the conversation reads as it
    happened and the sent prefix is still reused by identity."""

    first = _framed_observation(tmp_path, 0)
    second = _framed_observation(tmp_path, 1)
    replies = iter(
        [
            _click_payload(first, "nope"),
            _click_payload(first, "frame-0"),
            _click_payload(second, "frame-1"),
        ]
    )
    stream = RecordingStream(lambda _message: next(replies))
    client = _client(stream, tmp_path)

    history = [EpisodeTurn(observation=first)]
    with pytest.raises(DecisionRejected):
        await client.decide(first, tuple(history))
    decision = await client.decide(first, tuple(history))
    history[-1] = history[-1].model_copy(update={"batch": decision.action_batch})
    history.append(EpisodeTurn(observation=second))
    await client.decide(second, tuple(history))

    third = stream.requests[2]
    assert _shape(third) == [
        ("user", ("text", "image")),
        ("assistant", ("text",)),
        ("user", ("text",)),
        ("assistant", ("text",)),
        ("user", ("text", "image")),
    ]
    assert json.loads(third.messages[3].content[0].text)["actions"][0]["frame_id"] == "frame-0"
    assert all(a is b for a, b in zip(stream.requests[1].messages, third.messages))


@pytest.mark.asyncio
async def test_a_runaway_rejected_reply_is_replayed_truncated(tmp_path: Path) -> None:
    from local_operator.evaluation.runner.provider_client import (
        MAX_REJECTED_REPLY_CHARS,
    )

    current = _framed_observation(tmp_path, 0)
    stream = RecordingStream("x" * (MAX_REJECTED_REPLY_CHARS * 3))
    client = _client(stream, tmp_path)

    with pytest.raises(DecisionRejected):
        await client.decide(current, _turns(current))

    block = client._context.messages[1].content[0]
    assert isinstance(block, TextContent)
    assert len(block.text) < MAX_REJECTED_REPLY_CHARS + 64
    assert block.text.endswith("[... reply truncated]")


@pytest.mark.asyncio
async def test_client_carries_the_provider_request_id() -> None:
    """The only handle tying a bundle's model_response to the provider's records."""

    current = observation()
    stream = ScriptedStream(finish_payload(current), provider_payload={"id": "chatcmpl-ABC123"})

    decision = await _client(stream).decide(current, _turns(current))

    assert decision.provider_request_id == "chatcmpl-ABC123"


@pytest.mark.asyncio
async def test_an_over_length_provider_id_degrades_rather_than_truncating() -> None:
    """A truncated id is a valid identifier that matches nothing upstream."""

    current = observation()
    stream = ScriptedStream(finish_payload(current), provider_payload={"id": "a" * 200})

    decision = await _client(stream).decide(current, _turns(current))

    assert decision.provider_request_id == "unknown"


@pytest.mark.asyncio
async def test_an_unusable_provider_id_degrades_instead_of_failing() -> None:
    """StrictIdentifier forbids spaces; provenance must not break the episode."""

    current = observation()
    stream = ScriptedStream(finish_payload(current), provider_payload={"id": "not a valid id!"})

    decision = await _client(stream).decide(current, _turns(current))

    assert decision.provider_request_id == "unknown"


# ---------------------------------------------------------------------------
# Managed context: append-only history, frames, rebuild cadence, cache key
# ---------------------------------------------------------------------------


def _png() -> bytes:
    """A real 1x1 PNG so ``verify_artifact`` and any media check both accept it."""

    from local_operator.compaction.png import encode_grayscale_png

    return encode_grayscale_png(1, 1, b"\x00")


def _publish_frame(root: Path, data: bytes, *, name: str | None = None) -> str:
    digest = hashlib.sha256(data).hexdigest()
    root.mkdir(parents=True, exist_ok=True)
    (root / (name or digest)).write_bytes(data)
    return digest


def _framed_observation(root: Path, sequence: int, *, text: str = "a screen") -> Observation:
    from local_operator.evaluation.protocol import (
        ArtifactRef,
        FrameGeometry,
        FrameRef,
        FrameSize,
    )

    data = _png()
    digest = _publish_frame(root, data)
    provisional = Observation(
        task_id="task-1",
        episode_id="episode-1",
        sequence=sequence,
        observation_id="provisional",
        text=text,
        frames=(
            FrameRef(
                frame_id=f"frame-{sequence}",
                artifact=ArtifactRef(sha256=digest, media_type="image/png", byte_count=len(data)),
                geometry=FrameGeometry(
                    native=FrameSize(width=1, height=1),
                    model_visible=FrameSize(width=1, height=1),
                ),
            ),
        ),
    )
    return provisional.model_copy(update={"observation_id": observation_content_id(provisional)})


def _distinct_framed_observation(root: Path, sequence: int, pixel: bytes) -> Observation:
    """A framed observation whose frame bytes differ per ``pixel``.

    ``_framed_observation`` republishes one constant PNG, which is what makes
    it useful for the identical-frames case and useless for its negative.
    """

    from local_operator.compaction.png import encode_grayscale_png
    from local_operator.evaluation.protocol import (
        ArtifactRef,
        FrameGeometry,
        FrameRef,
        FrameSize,
    )

    data = encode_grayscale_png(1, 1, pixel)
    digest = _publish_frame(root, data)
    provisional = Observation(
        task_id="task-1",
        episode_id="episode-1",
        sequence=sequence,
        observation_id="provisional",
        text="a screen",
        frames=(
            FrameRef(
                frame_id="screen",
                artifact=ArtifactRef(sha256=digest, media_type="image/png", byte_count=len(data)),
                geometry=FrameGeometry(
                    native=FrameSize(width=1, height=1),
                    model_visible=FrameSize(width=1, height=1),
                ),
            ),
        ),
    )
    return provisional.model_copy(update={"observation_id": observation_content_id(provisional)})


class RecordingStream:
    """A stream fn that answers every request with a scripted reply.

    ``requests`` keeps every ``ChatRequest`` it saw, and the summary request
    is told apart from a decision by its system block, so one fake serves both
    the episode turns and the compaction summary the client buys.
    """

    def __init__(
        self,
        reply: str | Callable[[Any], str],
        *,
        summary: str = "SUMMARY",
        report_context: bool = False,
        context_scale: float = 1.0,
        report_context_until: int | None = None,
    ) -> None:
        self.reply = reply
        self.summary = summary
        # A real provider reports the context size it billed; ``True`` makes
        # the fake do the same (a local estimate of the request it received)
        # so the client's ``max(provider, local)`` trigger path is exercised.
        # ``context_scale`` inflates that figure the way a real provider does
        # for images (Anthropic bills a 1932px frame at ~5,000 tokens against
        # the local ruler's flat 1,200), which is what puts the provider's
        # figure ABOVE the local one and makes a stale copy of it dangerous.
        self.report_context = report_context
        self.context_scale = context_scale
        # ``Usage.context_tokens`` is optional on the wire; ``until`` stops
        # reporting after the first N requests so a stale figure can be
        # left behind on purpose.
        self.report_context_until = report_context_until
        self.requests: list[Any] = []
        self.summary_requests: list[Any] = []

    def __call__(self, request: Any, signal: Any) -> AsyncIterator[Any]:
        from local_operator.compaction.api import SUMMARIZATION_SYSTEM_PROMPT

        if request.system_blocks and request.system_blocks[0] == SUMMARIZATION_SYSTEM_PROMPT:
            self.summary_requests.append(request)
            return self._events(self.summary, Usage(input_tokens=1000, output_tokens=50))
        self.requests.append(request)
        observation = request.messages[-1]
        text = self.reply(observation) if callable(self.reply) else self.reply
        usage = Usage(input_tokens=10, output_tokens=5, usd_cost=0.001)
        reporting = self.report_context and (
            self.report_context_until is None or len(self.requests) <= self.report_context_until
        )
        if reporting:
            from local_operator.compaction.tokens import estimate_messages_tokens

            reported = int(estimate_messages_tokens(request.messages) * self.context_scale)
            usage = usage.model_copy(update={"context_tokens": reported})
        return self._events(text, usage)

    async def _events(self, text: str, usage: Usage) -> AsyncIterator[Any]:
        yield StreamTextDelta(delta=text)
        yield StreamEndEvent(stop_reason="stop", usage=usage)


def _shape(request: Any) -> list[tuple[str, tuple[str, ...]]]:
    return [
        (message.role, tuple(block.type for block in message.content))
        for message in request.messages
    ]


async def _drive(
    client: ProviderModelClient, root: Path, turns: int, *, text: str = "a screen"
) -> list[EpisodeTurn]:
    """Run ``turns`` decisions the way the runner does, closing each turn with
    the batch the model chose before appending the next observation."""

    history: list[EpisodeTurn] = []
    for sequence in range(turns):
        current = _framed_observation(root, sequence, text=f"{text} {sequence}")
        history.append(EpisodeTurn(observation=current))
        decision = await client.decide(current, tuple(history))
        history[-1] = history[-1].model_copy(update={"batch": decision.action_batch})
    return history


def _wait_reply(observation_message: Any) -> str:
    """Reply with a wait bound to whichever observation id the message names."""

    text = observation_message.content[0].text
    observation_id = next(
        line.split(": ", 1)[1] for line in text.splitlines() if line.startswith("Observation ID: ")
    )
    return json.dumps(
        {"actions": [{"kind": "wait", "observation_id": observation_id, "duration_ms": 1}]}
    )


@pytest.mark.asyncio
async def test_provider_client_sends_frames_and_replays_its_own_batches(tmp_path: Path) -> None:
    """The model sees a real screenshot and a real conversation, not one text
    message per turn -- the released runner sent neither."""

    stream = RecordingStream(_wait_reply)
    client = _client(stream, tmp_path, keep_recent_frames=3, rebuild_every_frames=8)

    await _drive(client, tmp_path, 3)

    assert _shape(stream.requests[0]) == [("user", ("text", "image"))]
    assert _shape(stream.requests[2]) == [
        ("user", ("text", "image")),
        ("assistant", ("text",)),
        ("user", ("text", "image")),
        ("assistant", ("text",)),
        ("user", ("text", "image")),
    ]
    first_user = stream.requests[2].messages[0]
    image = first_user.content[1]
    assert isinstance(image, ImageContent)
    assert base64.b64decode(image.data) == _png()
    assert image.mime_type == "image/png"
    # The assistant turns are the canonical batches the model itself returned.
    replayed = json.loads(stream.requests[2].messages[1].content[0].text)
    assert replayed["actions"][0]["kind"] == "wait"
    assert "Task: task-1" in first_user.content[0].text
    assert "Task:" not in stream.requests[2].messages[2].content[0].text


@pytest.mark.asyncio
async def test_provider_client_history_is_append_only_between_rebuilds(tmp_path: Path) -> None:
    """Every request's prefix is the previous request's messages BY IDENTITY,
    except at the one rebuild the frame budget schedules.

    K=3 kept frames, F=8 slack: the rebuild fires on the first request whose
    history holds more than 11 frames, i.e. turn index 11 (the 12th turn).
    Every other turn appends without touching a sent message.
    """

    stream = RecordingStream(_wait_reply)
    client = _client(stream, tmp_path, keep_recent_frames=3, rebuild_every_frames=8)

    await _drive(client, tmp_path, 14)

    rebuilds: list[int] = []
    for index in range(1, len(stream.requests)):
        previous = stream.requests[index - 1].messages
        current = stream.requests[index].messages
        prefix_identical = len(current) >= len(previous) and all(
            a is b for a, b in zip(previous, current)
        )
        if not prefix_identical:
            rebuilds.append(index)
    assert rebuilds == [11], rebuilds
    # After the rebuild only K frames survive and the rest carry the notice.
    from local_operator.compaction.pruning import (
        STALE_FRAME_NOTICE,
        count_frame_messages,
    )

    rebuilt = stream.requests[11].messages
    assert count_frame_messages(rebuilt) == 3
    assert any(
        isinstance(block, TextContent) and block.text == STALE_FRAME_NOTICE
        for message in rebuilt
        for block in message.content
    )
    # And the appends after it are again identity-preserving.
    assert all(a is b for a, b in zip(stream.requests[12].messages, stream.requests[13].messages))
    # The request-shape dump the PR body pastes.
    shapes = [
        (
            index,
            sum(1 for role, kinds in _shape(request) if "image" in kinds),
            len(request.messages),
        )
        for index, request in enumerate(stream.requests)
    ]
    assert shapes[10] == (10, 11, 21)
    assert shapes[11] == (11, 3, 23)


@pytest.mark.asyncio
async def test_provider_client_folds_summary_usage_into_the_decision(tmp_path: Path) -> None:
    """The summary call is a billed provider call; the decision's usage is the
    whole bill, and the compaction is reported so the runner declares it."""

    from local_operator.compaction.thresholds import CompactionSettings

    stream = RecordingStream(_wait_reply, summary="what happened so far")
    # A small window so the token threshold trips on the third turn, forcing a
    # context-full summary rather than only a frame prune. The window is large
    # enough that the summary plus the kept window fits the recovery band --
    # a threshold pass must now FIT, not merely run, so a window too small for
    # the compacted form refuses the request instead (that case has its own
    # test, ``test_context_unrecoverable_raises_when_nothing_fits``).
    spec = ModelSpec(provider="provider", model_id="model", context_window=20_000)
    client = ProviderModelClient(
        stream,
        route=ROUTE,
        model_spec=spec,
        artifact_root=tmp_path,
        # ``auto`` resolves to snapcompact for a vision model (the engine's
        # rule, which makes no provider call); pin context-full because THIS
        # test is about the billed summary call being folded in.
        compaction=CompactionSettings(
            strategy="context-full", keep_recent_tokens=1200, threshold_percent=0.4
        ),
        keep_recent_frames=3,
        rebuild_every_frames=8,
    )

    history: list[EpisodeTurn] = []
    decisions = []
    from local_operator.evaluation.runner.provider_client import (
        ContextUnrecoverableError,
    )

    for sequence in range(6):
        current = _framed_observation(tmp_path, sequence, text=f"screen {sequence} " + "x " * 60)
        history.append(EpisodeTurn(observation=current))
        try:
            decision = await client.decide(current, tuple(history))
        except ContextUnrecoverableError:
            # The tiny window (a 300-token band) cannot hold a context-full
            # summary plus the kept frames, so the client eventually refuses
            # the request the provider would reject. The fold under test here
            # already happened on the first compacting decision before that.
            break
        decisions.append(decision)
        history[-1] = history[-1].model_copy(update={"batch": decision.action_batch})

    compacted = [d for d in decisions if d.compaction is not None]
    assert compacted, "the token threshold never tripped"
    first = compacted[0]
    assert first.compaction is not None
    assert first.compaction.strategy == "context-full"
    assert first.compaction.summary_text == "what happened so far"
    assert first.compaction.messages_after < first.compaction.messages_before
    # 10 for the decision + 1000 for the summary; cost: the summary reported none.
    assert first.usage.input_tokens == 1010
    assert first.usage.output_tokens == 55
    assert len(stream.summary_requests) >= 1
    assert stream.summary_requests[0].replayable is True
    uncompacted = decisions[0]
    assert uncompacted.compaction is None and uncompacted.usage.input_tokens == 10
    # The rebuilt prefix opens with the summary marker the session would render.
    request_after = stream.requests[decisions.index(first)]
    assert "<previous-context-summary>" in request_after.messages[0].content[0].text


@pytest.mark.asyncio
@pytest.mark.parametrize(("context_scale", "report_until"), [(1.0, None), (1.1, 11)])
async def test_threshold_rebuild_creates_headroom_so_appends_resume(
    tmp_path: Path, context_scale: float, report_until: int | None
) -> None:
    """Once the window is genuinely full, a rebuild must buy several cached
    appends, not one per turn.

    Two things used to make the token trigger re-fire every turn after a
    pass (the reviewer observed a rebuild on every turn from 11 to 23):

    * A threshold-triggered pass re-asked the threshold after its own prune,
      so the prune alone could slip under the line, refuse the summary, and
      leave the context to cross the line again on the next observation. The
      threshold rebuild must therefore be a SUMMARISING pass, asserted below.
    * The provider's last reported context size described the prefix that
      was just replaced, and ``max(provider, local)`` kept judging the old
      figure. It bites when the provider bills above the local ruler (image
      billing does) AND stops reporting ``context_tokens`` right after the
      rebuild, so nothing refreshes the figure: the ``(1.1, 11)`` variant
      reports on the first 11 requests only, the last of them the over-the-
      line prefix. The client must forget the figure at a rebuild and judge
      the new prefix on the local estimate until a fresh one arrives.
    """

    from local_operator.compaction.thresholds import CompactionSettings

    stream = RecordingStream(
        _wait_reply,
        report_context=True,
        context_scale=context_scale,
        report_context_until=report_until,
    )
    # 60k: wide enough that the engine's own snapcompact archive budget (half
    # the window) leaves real headroom under the threshold, so a threshold
    # rebuild buys several appends. At 30k the archive replay is ITSELF ~15k
    # against a 24k threshold -- the pass cannot create headroom, and the
    # honest outcomes there (bounded requests, sparse rebuilds, refusal when
    # nothing fits) are the boundedness tests' business, not this one's.
    spec = ModelSpec(provider="provider", model_id="model", context_window=60_000)
    client = ProviderModelClient(
        stream,
        route=ROUTE,
        model_spec=spec,
        artifact_root=tmp_path,
        compaction=CompactionSettings(keep_recent_tokens=2000),
    )

    history: list[EpisodeTurn] = []
    decisions = []
    for sequence in range(24):
        current = _framed_observation(tmp_path, sequence, text=f"screen {sequence} " + "x " * 800)
        history.append(EpisodeTurn(observation=current))
        decision = await client.decide(current, tuple(history))
        decisions.append(decision)
        history[-1] = history[-1].model_copy(update={"batch": decision.action_batch})

    rebuilds = [
        index
        for index in range(1, len(stream.requests))
        if not all(
            a is b
            for a, b in zip(stream.requests[index - 1].messages, stream.requests[index].messages)
        )
    ]
    # The first rebuild is the frame budget (turn 11); every later one must be
    # separated from the previous by several plain appends.
    assert rebuilds[0] == 11
    assert len(rebuilds) >= 2
    gaps = [b - a for a, b in zip(rebuilds, rebuilds[1:])]
    assert min(gaps) >= 4, (rebuilds, gaps)
    # Every threshold-triggered rebuild summarised (the engine's strategy for
    # a vision model is snapcompact) rather than merely pruning frames and
    # refusing the summary as below-threshold.
    later = [decisions[index].compaction for index in rebuilds[1:]]
    assert all(record is not None and record.strategy == "snapcompact" for record in later), [
        record.strategy if record else None for record in later
    ]
    assert all(request.messages for request in stream.requests)


@pytest.mark.asyncio
async def test_context_unrecoverable_raises_when_nothing_fits(tmp_path: Path) -> None:
    """A window so small the compacted form cannot fit: after pruning, summarising and
    shedding every stale observation it may, the rebuilt prefix still exceeds the recovery band,
    so the client refuses the request rather than send one the provider will reject
    (``ContextUnrecoverableError``), which the runner turns into a harness error the bundle
    records -- never a rejected request."""

    from local_operator.compaction.thresholds import CompactionSettings
    from local_operator.evaluation.runner.provider_client import (
        ContextUnrecoverableError,
    )

    stream = RecordingStream(_wait_reply)
    # A window smaller than the current observation's own frame plus text: after
    # shedding every stale observation, the current one still cannot fit, so no
    # summary can help. ``keep_recent_tokens=0`` means the pass keeps no tail.
    spec = ModelSpec(provider="provider", model_id="model", context_window=1_000)
    client = ProviderModelClient(
        stream,
        route=ROUTE,
        model_spec=spec,
        artifact_root=tmp_path,
        compaction=CompactionSettings(keep_recent_tokens=0, threshold_percent=0.5),
        keep_recent_frames=3,
        rebuild_every_frames=8,
    )
    history = _turns(_framed_observation(tmp_path, 0))
    with pytest.raises(ContextUnrecoverableError):
        await client.decide(history[0].observation, history)


@pytest.mark.asyncio
async def test_boundedness_priced_context_never_exceeds_the_window(tmp_path: Path) -> None:
    """The reviewer's provider-faithful probe (per-image addend billing, 5,000/frame,
    128k window, ~4k-token observations): the priced size of every request must stay
    within the window, AND the rebuild count stays sparse. The stall latch that was
    removed for M2 let this grow to 1.28-1.37x; it must not."""

    from local_operator.compaction.thresholds import CompactionSettings

    class FaithfulStream:
        """Reports ``context_tokens`` as local estimate + a per-image ADDEND
        (Anthropic-style frame billing), so the client's ``max(provider,
        local)`` trigger and the request's own priced size are
        provider-faithful. The addend is the engine's own formula for the
        spec's family (``frame_token_estimate_for``) minus the local ruler's
        flat 1,200 -- exactly the correction the client prices its shed band
        with, so the fake bills the way the client predicts."""

        def __init__(self, per_image: int | None = None) -> None:
            from local_operator.compaction.snapcompact import frame_token_estimate_for
            from local_operator.compaction.tokens import IMAGE_TOKEN_ESTIMATE

            # Default to the engine's family formula for the spec under test
            # (``provider/model`` is an unknown family, so the safe ceiling's
            # per-frame figure applies) minus the ruler's flat 1,200.
            self.per_image = (
                frame_token_estimate_for("provider", "model") - IMAGE_TOKEN_ESTIMATE
                if per_image is None
                else per_image
            )
            self.requests: list[Any] = []
            self.priced: list[int] = []

        def __call__(self, request: Any, signal: Any) -> AsyncIterator[Any]:
            from local_operator.compaction.tokens import estimate_messages_tokens

            images = sum(
                1
                for message in request.messages
                for block in message.content
                if block.type == "image"
            )
            priced = estimate_messages_tokens(request.messages) + images * self.per_image
            self.priced.append(priced)
            self.requests.append(request)
            oid = next(
                line.split(": ", 1)[1]
                for line in request.messages[-1].content[0].text.splitlines()
                if line.startswith("Observation ID: ")
            )
            text = json.dumps(
                {"actions": [{"kind": "wait", "observation_id": oid, "duration_ms": 1}]}
            )
            usage = Usage(input_tokens=10, output_tokens=5, usd_cost=0.001, context_tokens=priced)

            async def events() -> AsyncIterator[Any]:
                yield StreamTextDelta(delta=text)
                yield StreamEndEvent(stop_reason="stop", usage=usage)

            return events()

    window = 128_000
    stream = FaithfulStream()
    spec = ModelSpec(provider="provider", model_id="model", context_window=window)
    client = ProviderModelClient(
        stream,
        route=ROUTE,
        model_spec=spec,
        artifact_root=tmp_path,
        compaction=CompactionSettings(keep_recent_tokens=20_000),
    )

    history: list[EpisodeTurn] = []
    for sequence in range(48):
        current = _framed_observation(tmp_path, sequence, text=f"screen {sequence} " + "x " * 900)
        history.append(EpisodeTurn(observation=current))
        decision = await client.decide(current, tuple(history))
        history[-1] = history[-1].model_copy(update={"batch": decision.action_batch})

    peak = max(stream.priced)
    assert peak <= window, f"priced context exceeded the window: {peak} ({peak / window:.2f}x)"
    assert all(priced <= window for priced in stream.priced), [
        priced / window for priced in stream.priced
    ]
    # Rebuilds stay sparse: a frame cadence plus the occasional threshold pass, never per turn.
    rebuilds = [
        index
        for index in range(1, len(stream.requests))
        if not all(
            a is b
            for a, b in zip(stream.requests[index - 1].messages, stream.requests[index].messages)
        )
    ]
    gaps = [b - a for a, b in zip(rebuilds, rebuilds[1:])]
    assert len(rebuilds) <= 12, rebuilds
    assert max(gaps) >= 4, (rebuilds, gaps)


@pytest.mark.asyncio
async def test_boundedness_tight_window_stays_under_with_a_priced_shed(tmp_path: Path) -> None:
    """The same provider-faithful drive at a 24k window on an ANTHROPIC large
    model (5,000 tokens/frame): the local ruler alone would let the shed stop
    while the provider still billed over the window (per-frame addend: 3 frames
    at 1,200 each locally is 3,600, but the provider bills them at 5,000 each),
    so the shed's band must be judged at the provider's per-frame price. Every
    request stays within the window; when even the kept window cannot fit, the
    client refuses (``ContextUnrecoverableError``) rather than send a rejected
    request."""

    from local_operator.compaction.thresholds import CompactionSettings
    from local_operator.evaluation.runner.provider_client import (
        ContextUnrecoverableError,
    )

    class FaithfulStream:
        def __init__(self) -> None:
            from local_operator.compaction.snapcompact import frame_token_estimate_for
            from local_operator.compaction.tokens import IMAGE_TOKEN_ESTIMATE

            # The fake bills the way the client's shed band predicts: the engine's
            # family formula for the spec (Anthropic large: 5,000/frame) minus the
            # ruler's flat 1,200. That makes the assertion self-consistent (the
            # client bounds the value the fake bills), and the local-ruler mutation
            # still fails because it under-prices frames.
            self.per_image = (
                frame_token_estimate_for("anthropic", "claude-fable-5") - IMAGE_TOKEN_ESTIMATE
            )
            self.requests: list[Any] = []
            self.priced: list[int] = []

        def __call__(self, request: Any, signal: Any) -> AsyncIterator[Any]:
            from local_operator.compaction.tokens import estimate_messages_tokens

            images = sum(
                1
                for message in request.messages
                for block in message.content
                if block.type == "image"
            )
            priced = estimate_messages_tokens(request.messages) + images * self.per_image
            self.priced.append(priced)
            self.requests.append(request)
            oid = next(
                line.split(": ", 1)[1]
                for line in request.messages[-1].content[0].text.splitlines()
                if line.startswith("Observation ID: ")
            )
            text = json.dumps(
                {"actions": [{"kind": "wait", "observation_id": oid, "duration_ms": 1}]}
            )
            usage = Usage(input_tokens=10, output_tokens=5, usd_cost=0.001, context_tokens=priced)

            async def events() -> AsyncIterator[Any]:
                yield StreamTextDelta(delta=text)
                yield StreamEndEvent(stop_reason="stop", usage=usage)

            return events()

    window = 24_000
    stream = FaithfulStream()
    spec = ModelSpec(provider="anthropic", model_id="claude-fable-5", context_window=window)
    client = ProviderModelClient(
        stream,
        route=ROUTE,
        model_spec=spec,
        artifact_root=tmp_path,
        compaction=CompactionSettings(keep_recent_tokens=20_000),
    )

    history: list[EpisodeTurn] = []
    refused_at: int | None = None
    for sequence in range(48):
        current = _framed_observation(tmp_path, sequence, text=f"screen {sequence} " + "x " * 900)
        history.append(EpisodeTurn(observation=current))
        try:
            decision = await client.decide(current, tuple(history))
        except ContextUnrecoverableError:
            refused_at = sequence
            break
        history[-1] = history[-1].model_copy(update={"batch": decision.action_batch})

    peak = max(stream.priced)
    assert peak <= window, f"priced context exceeded the window: {peak} ({peak / window:.2f}x)"
    assert all(priced <= window for priced in stream.priced), [
        (i, p / window) for i, p in enumerate(stream.priced)
    ]
    # Either the episode ran bounded to the end, or it refused once the kept
    # window itself could not fit -- never an over-window request.
    assert refused_at is None or peak <= window


@pytest.mark.asyncio
async def test_shed_actually_sheds_and_a_mid_size_window_runs_to_the_end(tmp_path: Path) -> None:
    """The reviewer's 64k probe (round 3, M3): with the shed dead, this drive
    refused at turn 23 with 17 stale turns still in the prefix that shedding
    would have fit (48,612 priced against a 40,960 band). A working shed keeps
    the episode going for all 48 turns, sheds real turns (``messages_after <
    messages_before`` on a ``CompactionRecord``), keeps every request within
    the window, and keeps rebuilds sparse. A shed replaced by ``return False,
    0`` fails this test at the refusal."""

    from local_operator.compaction.thresholds import CompactionSettings

    class FaithfulStream:
        def __init__(self) -> None:
            from local_operator.compaction.tokens import IMAGE_TOKEN_ESTIMATE

            # The reviewer's probe billed 5,000/frame; the client prices the
            # unknown "provider" family at 3,293, so the fake bills ABOVE what
            # the client predicts and the window bound is the harder test.
            self.per_image = 5_000 - IMAGE_TOKEN_ESTIMATE
            self.requests: list[Any] = []
            self.priced: list[int] = []

        def __call__(self, request: Any, signal: Any) -> AsyncIterator[Any]:
            from local_operator.compaction.tokens import estimate_messages_tokens

            images = sum(
                1
                for message in request.messages
                for block in message.content
                if block.type == "image"
            )
            priced = estimate_messages_tokens(request.messages) + images * self.per_image
            self.priced.append(priced)
            self.requests.append(request)
            oid = next(
                line.split(": ", 1)[1]
                for line in request.messages[-1].content[0].text.splitlines()
                if line.startswith("Observation ID: ")
            )
            text = json.dumps(
                {"actions": [{"kind": "wait", "observation_id": oid, "duration_ms": 1}]}
            )
            usage = Usage(input_tokens=10, output_tokens=5, usd_cost=0.001, context_tokens=priced)

            async def events() -> AsyncIterator[Any]:
                yield StreamTextDelta(delta=text)
                yield StreamEndEvent(stop_reason="stop", usage=usage)

            return events()

    window = 64_000
    stream = FaithfulStream()
    spec = ModelSpec(provider="provider", model_id="model", context_window=window)
    client = ProviderModelClient(
        stream,
        route=ROUTE,
        model_spec=spec,
        artifact_root=tmp_path,
        compaction=CompactionSettings(keep_recent_tokens=20_000),
    )

    history: list[EpisodeTurn] = []
    records = []
    for sequence in range(48):
        current = _framed_observation(tmp_path, sequence, text=f"screen {sequence} " + "x " * 900)
        history.append(EpisodeTurn(observation=current))
        decision = await client.decide(current, tuple(history))
        if decision.compaction is not None:
            records.append(decision.compaction)
        history[-1] = history[-1].model_copy(update={"batch": decision.action_batch})

    # Ran to the end: no premature refusal.
    assert len(stream.requests) == 48
    # The shed removed real turns on at least one threshold pass.
    shed = [r for r in records if r.messages_after < r.messages_before]
    assert shed, [(r.strategy, r.messages_before, r.messages_after) for r in records]
    peak = max(stream.priced)
    assert peak <= window, f"priced context exceeded the window: {peak} ({peak / window:.2f}x)"
    rebuilds = [
        index
        for index in range(1, len(stream.requests))
        if not all(
            a is b
            for a, b in zip(stream.requests[index - 1].messages, stream.requests[index].messages)
        )
    ]
    gaps = [b - a for a, b in zip(rebuilds, rebuilds[1:])]
    assert len(rebuilds) <= 16, rebuilds
    assert max(gaps) >= 3, (rebuilds, gaps)


@pytest.mark.asyncio
async def test_a_shed_only_rebuild_is_declared_as_a_compaction(tmp_path: Path) -> None:
    """Review round 4, m8: on a 16k window with ~4.3k-token observations and
    ``keep_recent_tokens=12000`` the kept window is the whole tail, so the
    threshold pass refuses (``nothing-to-summarize``) and the shed alone
    rebuilds the prefix. Every such rebuild — a previously sent message absent
    from the next request — must come back with a ``CompactionRecord`` (the
    runner turns each one into a ``context_compaction`` event, proved in
    ``test_context_compaction_event_is_recorded_and_verifies``); a rebuild
    whose only trace is a ``message_count`` drop is hidden from the bundle.
    The record's ``tokens_after`` measures the prefix actually sent, not the
    pass's pre-shed figure."""

    from local_operator.compaction.thresholds import CompactionSettings
    from local_operator.compaction.tokens import estimate_messages_tokens

    stream = RecordingStream(_wait_reply)
    spec = ModelSpec(provider="provider", model_id="model", context_window=16_000)
    client = ProviderModelClient(
        stream,
        route=ROUTE,
        model_spec=spec,
        artifact_root=tmp_path,
        compaction=CompactionSettings(keep_recent_tokens=12_000),
    )

    history: list[EpisodeTurn] = []
    rebuilds = 0
    undeclared = 0
    shed_records = []
    for sequence in range(24):
        current = _framed_observation(tmp_path, sequence, text=f"screen {sequence} " + "x " * 900)
        history.append(EpisodeTurn(observation=current))
        decision = await client.decide(current, tuple(history))
        if sequence:
            previous, now = stream.requests[-2].messages, stream.requests[-1].messages
            vanished = any(all(old is not new for new in now) for old in previous)
            if vanished:
                rebuilds += 1
                if decision.compaction is None:
                    undeclared += 1
                elif decision.compaction.messages_after < decision.compaction.messages_before:
                    shed_records.append(
                        (decision.compaction.tokens_after, estimate_messages_tokens(now))
                    )
        history[-1] = history[-1].model_copy(update={"batch": decision.action_batch})

    assert rebuilds >= 10, rebuilds
    assert undeclared == 0, f"{undeclared} of {rebuilds} rebuilds carried no CompactionRecord"
    assert shed_records, "the shape never reached a shed-only rebuild"
    assert all(recorded == sent for recorded, sent in shed_records), shed_records[:3]


@pytest.mark.asyncio
async def test_frameless_benchmark_stays_bounded_with_no_frames_to_shed(tmp_path: Path) -> None:
    """A text-only benchmark (no ImageContent ever) has no frames for the shed or the
    cadence to fire on; with the stall latch removed, the threshold trigger is the only
    bound and it must hold (the priced context stays well under the window)."""

    from local_operator.compaction.thresholds import CompactionSettings

    class TextStream:
        def __init__(self) -> None:
            self.requests: list[Any] = []
            self.priced: list[int] = []

        def __call__(self, request: Any, signal: Any) -> AsyncIterator[Any]:
            from local_operator.compaction.tokens import estimate_messages_tokens

            priced = estimate_messages_tokens(request.messages)
            self.priced.append(priced)
            self.requests.append(request)
            oid = next(
                line.split(": ", 1)[1]
                for line in request.messages[-1].content[0].text.splitlines()
                if line.startswith("Observation ID: ")
            )
            text = json.dumps(
                {"actions": [{"kind": "wait", "observation_id": oid, "duration_ms": 1}]}
            )
            usage = Usage(input_tokens=10, output_tokens=5, usd_cost=0.001, context_tokens=priced)

            async def events() -> AsyncIterator[Any]:
                yield StreamTextDelta(delta=text)
                yield StreamEndEvent(stop_reason="stop", usage=usage)

            return events()

    window = 128_000
    stream = TextStream()
    spec = ModelSpec(provider="provider", model_id="model", context_window=window)
    client = ProviderModelClient(
        stream,
        route=ROUTE,
        model_spec=spec,
        artifact_root=tmp_path,
        compaction=CompactionSettings(keep_recent_tokens=20_000),
    )

    history: list[EpisodeTurn] = []
    for sequence in range(48):
        current = _framed_observation(tmp_path, sequence, text=f"state {sequence} " + "x " * 900)
        current = current.model_copy(update={"frames": ()})
        history.append(EpisodeTurn(observation=current))
        decision = await client.decide(current, tuple(history))
        history[-1] = history[-1].model_copy(update={"batch": decision.action_batch})

    peak = max(stream.priced)
    assert (
        peak <= window
    ), f"frameless priced context exceeded the window: {peak} ({peak / window:.2f}x)"
    assert all(priced <= window for priced in stream.priced)


@pytest.mark.asyncio
async def test_provider_client_records_the_cache_key_on_every_request(tmp_path: Path) -> None:
    stream = RecordingStream(_wait_reply)
    client = _client(stream, tmp_path, prompt_cache_key="lop-eval-episode-1")

    history = await _drive(client, tmp_path, 2)
    current = _framed_observation(tmp_path, 2)
    history.append(EpisodeTurn(observation=current))
    decision = await client.decide(current, tuple(history))

    assert all(request.prompt_cache_key == "lop-eval-episode-1" for request in stream.requests)
    assert decision.prompt_cache_key == "lop-eval-episode-1"
    assert decision.context_tokens is not None and decision.context_tokens > 0


@pytest.mark.asyncio
async def test_provider_client_refuses_a_frame_whose_bytes_do_not_match(tmp_path: Path) -> None:
    """A frame the runner would refuse to publish is one the model must not
    see; the refusal surfaces as a provider failure rather than a blind turn."""

    from local_operator.evaluation.adapters.supervisor import SupervisionError

    current = _framed_observation(tmp_path, 0)
    digest = current.frames[0].artifact.sha256
    data = (tmp_path / digest).read_bytes()
    (tmp_path / digest).write_bytes(data[:-1] + bytes([data[-1] ^ 0xFF]))
    stream = RecordingStream(_wait_reply)

    with pytest.raises(SupervisionError):
        await _client(stream, tmp_path).decide(current, _turns(current))
    assert stream.requests == []


@pytest.mark.asyncio
async def test_unchanged_observation_text_is_marked_not_repeated(tmp_path: Path) -> None:
    # ``_drive`` suffixes the sequence into the text, so build the two
    # literally-equal observations by hand.
    fresh = RecordingStream(_wait_reply)
    client = _client(fresh, tmp_path)
    history: list[EpisodeTurn] = []
    for sequence in range(2):
        current = _framed_observation(tmp_path, sequence + 10, text="identical")
        history.append(EpisodeTurn(observation=current))
        decision = await client.decide(current, tuple(history))
        history[-1] = history[-1].model_copy(update={"batch": decision.action_batch})

    second_user = fresh.requests[1].messages[2].content[0].text
    assert second_user.rstrip().endswith("(unchanged)")
    assert "identical" in fresh.requests[1].messages[0].content[0].text


@pytest.mark.asyncio
async def test_absent_observation_text_is_not_reported_as_unchanged(tmp_path: Path) -> None:
    """The defect from bundle ep-ffda3fc88f81: 15 of 17 turns read "(unchanged)".

    That adapter publishes the task text on step 0 and screenshots only
    thereafter, so ``text`` is ``None`` from step 1 on. Comparing ``None`` to
    ``None`` made every later turn claim the state had not changed -- against
    frames that Pillow measures as 0.02%-93% different -- which destroyed the
    model's only textual progress signal AND told it, falsely, that its
    actions were no-ops.
    """

    stream = RecordingStream(_wait_reply)
    client = _client(stream, tmp_path)
    history: list[EpisodeTurn] = []
    for sequence in range(3):
        current = _framed_observation(tmp_path, sequence).model_copy(update={"text": None})
        current = current.model_copy(
            update={"observation_id": observation_content_id(current)},
        )
        history.append(EpisodeTurn(observation=current))
        decision = await client.decide(current, tuple(history))
        history[-1] = history[-1].model_copy(update={"batch": decision.action_batch})

    rendered = [
        message.content[0].text
        for message in stream.requests[-1].messages
        if message.role == "user" and isinstance(message.content[0], TextContent)
    ]
    assert len(rendered) == 3
    for text in rendered:
        assert text.rstrip().endswith("(no textual state)")
        assert "(unchanged)" not in text


@pytest.mark.asyncio
async def test_identical_frames_are_declared_as_a_no_op(tmp_path: Path) -> None:
    """A screenshot-only benchmark has no other way to say "nothing happened".

    The real episode clicked pixel (674,44) four times. Nothing in its context
    stated that two consecutive screenshots were the same image, so re-deciding
    the same click was the rational reading rather than a lapse.
    """

    stream = RecordingStream(_wait_reply)
    client = _client(stream, tmp_path)
    history: list[EpisodeTurn] = []
    # ``_framed_observation`` republishes the SAME 1x1 PNG for every sequence,
    # so consecutive frames are byte-identical by construction.
    for sequence in range(2):
        current = _framed_observation(tmp_path, sequence)
        history.append(EpisodeTurn(observation=current))
        decision = await client.decide(current, tuple(history))
        history[-1] = history[-1].model_copy(update={"batch": decision.action_batch})

    first, second = (
        message.content[0].text
        for message in stream.requests[-1].messages
        if message.role == "user" and isinstance(message.content[0], TextContent)
    )
    assert "byte-identical" not in first
    assert "byte-identical to the previous observation's" in second
    assert "Do not repeat it" in second
    # The prompt has to teach the model what that sentence is for.
    assert "did nothing" in stream.requests[-1].system_blocks[0]


@pytest.mark.asyncio
async def test_changed_frames_carry_no_no_op_note(tmp_path: Path) -> None:
    """A false no-op note is worse than none: it would tell a model that a
    working action failed."""

    stream = RecordingStream(_wait_reply)
    client = _client(stream, tmp_path)
    history: list[EpisodeTurn] = []
    for sequence, pixel in enumerate((b"\x00", b"\x7f")):
        current = _distinct_framed_observation(tmp_path, sequence, pixel)
        history.append(EpisodeTurn(observation=current))
        decision = await client.decide(current, tuple(history))
        history[-1] = history[-1].model_copy(update={"batch": decision.action_batch})

    rendered = [
        message.content[0].text
        for message in stream.requests[-1].messages
        if message.role == "user" and isinstance(message.content[0], TextContent)
    ]
    assert not any("byte-identical" in text for text in rendered)


@pytest.mark.asyncio
async def test_frameless_turns_never_claim_an_unchanged_screen(tmp_path: Path) -> None:
    """A benchmark with no screen has no screen to be unchanged."""

    stream = RecordingStream(_wait_reply)
    client = _client(stream, tmp_path)
    history: list[EpisodeTurn] = []
    for sequence in range(2):
        current = observation(sequence)
        history.append(EpisodeTurn(observation=current))
        decision = await client.decide(current, tuple(history))
        history[-1] = history[-1].model_copy(update={"batch": decision.action_batch})

    rendered = [
        message.content[0].text
        for message in stream.requests[-1].messages
        if message.role == "user" and isinstance(message.content[0], TextContent)
    ]
    assert not any("byte-identical" in text for text in rendered)


@pytest.mark.asyncio
async def test_a_rejected_reply_is_carried_on_the_exception_for_the_bundle(
    tmp_path: Path,
) -> None:
    """Without the reply, a rejection class cannot be diagnosed after the fact.

    The real episode's three ``decision-rejected`` errors recorded only the
    Pydantic diagnostic, so "was the model trying to type?" was unanswerable
    without paying for the run again.
    """

    current = _framed_observation(tmp_path, 0)
    bad = json.dumps(
        {
            "actions": [
                {"kind": "key", "observation_id": current.observation_id, "key": ["ctrl", "s"]}
            ]
        }
    )
    stream = RecordingStream(lambda _message: bad)

    with pytest.raises(DecisionRejected) as info:
        await _client(stream, tmp_path).decide(current, _turns(current))

    assert info.value.reply == bad
    assert '"key": ["ctrl", "s"]' in (info.value.reply or "")


@pytest.mark.asyncio
async def test_an_over_long_rejected_reply_is_bounded_on_the_exception(
    tmp_path: Path,
) -> None:
    """A provider's max-token wall of prose must not ride into the bundle whole."""

    current = _framed_observation(tmp_path, 0)
    stream = RecordingStream(lambda _message: "x" * (MAX_REJECTED_REPLY_CHARS + 500))

    with pytest.raises(DecisionRejected) as info:
        await _client(stream, tmp_path).decide(current, _turns(current))

    assert info.value.reply is not None
    assert len(info.value.reply) == MAX_REJECTED_REPLY_CHARS


def test_prompt_teaches_how_to_type_and_how_to_press_keys() -> None:
    """The real episode emitted 16 actions, none of them a type or a key.

    Clicking cannot enter a calendar title or a search term, so a prompt that
    lists "type" in a schema table without saying it exists, that it follows
    focus, and that "key" is a chord leaves the vocabulary technically
    complete and practically unreachable.
    """

    prompt = build_system_prompt()

    assert "keyboard" in prompt
    # Typing follows focus; a model that thinks type() clicks first will type
    # into whatever had focus and see nothing happen.
    assert "wherever the keyboard focus already is" in prompt
    # Asserted against the unwrapped text: the prompt's line breaks are
    # cosmetic and a test that pins them fails on a reflow, not a regression.
    flat = " ".join(prompt.split())
    assert "does not press Enter for you" in flat
    # A chord, not a sequence -- the distinction that decides Ctrl+S.
    assert "TOGETHER" in prompt
    assert '["ctrl", "s"]' in prompt
    # Multi-action batches are what make click-then-type one step.
    assert "execute in order" in prompt


def test_prompt_lists_the_key_vocabulary_the_validator_enforces() -> None:
    """``KeyAction`` accepts a closed set; an unlisted synonym is a lost turn.

    Probed against the real parser: ``["control", "k"]`` is refused with
    ``unknown key: 'control'`` while ``["ctrl", "k"]`` parses. A prompt that
    only shows examples leaves the model to guess which spelling is the
    accepted one.
    """

    from local_operator.evaluation.protocol import NAMED_KEYS

    prompt = build_system_prompt()
    flat = " ".join(prompt.split())

    # Derived from the enforcing set, so a new named key cannot drift out of
    # the instructions -- the same guarantee _action_schema_lines gives.
    for key in NAMED_KEYS:
        if key.startswith("F") and key[1:].isdigit():
            continue
        assert key.lower() in flat, key
    assert "f1 through f24" in flat
    # The three synonyms a model actually reaches for are named as wrong.
    assert '"ctrl" not "control"' in flat
    assert '"esc" not "escape"' in flat
    assert '"enter" not "return"' in flat


def test_prompt_states_the_coordinate_convention() -> None:
    """A model told a frame's size but not its origin is guessing at half the
    contract; the protocol clamps and then REJECTS, so a wrong convention is a
    lost turn rather than a near miss."""

    prompt = build_system_prompt()

    assert "TOP-LEFT" in prompt
    assert "x grows" in prompt and "y grows down" in prompt
    assert "width-1" in prompt
    # The screenshot is the pixel space -- no rescaling, no 0-1 normalization.
    assert "normalize to 0-1" in prompt


@pytest.mark.asyncio
async def test_ask_answer_reaches_the_model_with_the_next_observation(tmp_path: Path) -> None:
    stream = RecordingStream(_wait_reply)
    client = _client(stream, tmp_path)

    first = _framed_observation(tmp_path, 0)
    history = [EpisodeTurn(observation=first)]
    decision = await client.decide(first, tuple(history))
    history[-1] = history[-1].model_copy(
        update={"batch": decision.action_batch, "ask_answer": "click the blue one"}
    )
    second = _framed_observation(tmp_path, 1)
    history.append(EpisodeTurn(observation=second))
    await client.decide(second, tuple(history))

    assert (
        "Answer from the user: click the blue one" in stream.requests[1].messages[2].content[0].text
    )


def test_create_provider_model_client_seals_fallback_and_mints_the_cache_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``forbid`` must switch the stream function's model fallback OFF through
    the real ``retry.modelFallback`` setting, and the cache key must be the
    per-episode session id."""

    from local_operator.evaluation.runner import provider_client
    from local_operator.model import configure

    captured: dict[str, Any] = {}

    def fake_create_stream_fn(
        auth_store: Any, settings: Any, *, session_id: str | None = None
    ) -> Any:
        captured["settings"] = settings
        captured["session_id"] = session_id
        return object()

    monkeypatch.setattr(configure, "create_stream_fn", fake_create_stream_fn)
    client = provider_client.create_provider_model_client(
        auth_store=object(),
        settings={"retry": {"maxRetries": 2}},
        route=ROUTE,
        model_spec=ModelSpec(provider="provider", model_id="model"),
        artifact_root=tmp_path,
        episode_id="ep-42",
        fallback_policy="forbid",
    )

    from local_operator.providers.failover import RetrySettings

    resolved = RetrySettings.from_settings(captured["settings"])
    assert resolved.model_fallback is False
    assert resolved.max_retries == 2
    assert captured["session_id"] == "lop-eval-ep-42"
    assert client._prompt_cache_key == "lop-eval-ep-42"

    provider_client.create_provider_model_client(
        auth_store=object(),
        settings=None,
        route=ROUTE,
        model_spec=ModelSpec(provider="provider", model_id="model"),
        artifact_root=tmp_path,
        episode_id="ep-43",
        fallback_policy="allow_any",
    )
    assert RetrySettings.from_settings(captured["settings"]).model_fallback is True
