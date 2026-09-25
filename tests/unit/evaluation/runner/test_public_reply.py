"""Offline public-memory parity through the real provider and shared compactor.

The erasure test deliberately varies ONLY image bytes first. Different IDs or
observation text would leak the answer into the control and hide the defect.
Summary output below is deterministic plumbing evidence, not an LLM quality claim.
"""

from __future__ import annotations

import base64
import json
import logging
from pathlib import Path
from typing import Any
from urllib.parse import quote

import pytest

from local_operator.compaction.api import serialize_conversation
from local_operator.compaction.pass_ import run_compaction_pass
from local_operator.compaction.snapcompact import serialize_for_snapcompact
from local_operator.compaction.thresholds import CompactionSettings
from local_operator.evaluation.evidence.models import (
    ActionBatchPayload,
    ModelRequestPayload,
    ModelResponsePayload,
    canonical_digest,
)
from local_operator.evaluation.evidence.verify import verify_bundle
from local_operator.evaluation.receipts import RedactionSet
from local_operator.evaluation.runner.episode import EpisodeRunner, _rejection_detail
from local_operator.evaluation.runner.model import (
    DecisionRejected,
    EpisodeTurn,
    StreamShape,
)
from local_operator.evaluation.runner.provider_client import (
    DecisionParseError,
    _ContextBuilder,
    build_system_prompt,
    classify_rejection,
    parse_decision,
)
from local_operator.evaluation.runner.public_reply import (
    _MAX_EXTRA_KEY_CHARS,
    _MAX_EXTRA_KEYS_SHOWN,
    _MAX_TRAILING_DECODE_ATTEMPTS,
    MAX_PUBLIC_OBSERVATIONS_CHARS,
    REJECTED_PUBLIC_REPLY,
    REJECTED_REPLY_WITHHELD,
    decode_public_reply,
    looks_like_public_reply,
    public_reply_contract,
    redact_public_reply,
    rejected_reply_evidence,
)
from local_operator.harness.types import ImageContent, Message, ModelSpec, TextContent
from tests.unit.evaluation.runner.conftest import (
    FakeAdapter,
    ScriptedModel,
    build_config,
    build_spec,
    payloads,
    selector,
)
from tests.unit.evaluation.runner.test_episode import _rescue_ok
from tests.unit.evaluation.runner.test_provider_client import (
    ROUTE,
    RecordingStream,
    _client,
    _wait_reply,
    finish_payload,
    observation,
    type_payload,
)


def envelope(batch: str, notes: Any = "") -> str:
    return json.dumps(
        {
            "reply_version": "1.0",
            "action_batch": json.loads(batch),
            "public_observations": notes,
        }
    )


@pytest.mark.parametrize("notes", ["", "Visible status: ready", "x" * 2000, "東京"])
def test_public_reply_binds_without_changing_legacy_batch_bytes(notes: str) -> None:
    current = observation()
    legacy = parse_decision(type_payload(current), current, route=ROUTE)
    visible = envelope(type_payload(current), notes)
    decision = parse_decision(visible, current, route=ROUTE)
    assert decision.public_reply == visible
    assert legacy.public_reply is None
    assert decision.action_batch.to_canonical_json() == legacy.action_batch.to_canonical_json()
    assert canonical_digest("adapter-action-batch-v1", decision.action_batch) == canonical_digest(
        "adapter-action-batch-v1", legacy.action_batch
    )
    assert current == observation()


@pytest.mark.parametrize("notes", [None, [], {}, 1, True, "x" * 2001, "\ud800"])
def test_invalid_or_oversized_notes_reject_entire_decision(notes: Any) -> None:
    with pytest.raises(DecisionParseError):
        parse_decision(envelope(type_payload(observation()), notes), observation(), route=ROUTE)


def _full_envelope(current: Any, notes: str = "visible fact") -> str:
    """The envelope shape the contract used to REQUIRE, for tolerance tests.

    Kept as the fixture of the old contract on purpose: every accepted-framing
    case below is a reply the strict decoder refused, so the tests measure the
    tolerance against the exact bytes it exists for rather than against the
    shape this build now offers.
    """

    return envelope(type_payload(current), notes)


def _wrapped(variant: str, payload: str) -> str:
    """One generic tool-call serialization around a full, valid envelope.

    The three shapes are verbatim what the DeepSeek canary arm's rejection
    artifacts carried; 20 of its 104 refusals were exactly this -- a complete,
    executable envelope inside one of these wrappers, discarded on framing.
    """

    if variant == "tool_name-parameters":
        return json.dumps({"tool_name": "lop_structured_reply", "parameters": json.loads(payload)})
    if variant == "tool_call-input-string":
        return json.dumps({"tool_call": "lop_structured_reply", "input": payload})
    if variant == "input-object":
        return json.dumps({"input": json.loads(payload)})
    raise AssertionError(variant)


def _changed(change: str, raw: str, current: Any) -> str:
    """One framing change to a full envelope, as raw reply text."""

    value = json.loads(raw)
    if change == "missing-version":
        del value["reply_version"]
    elif change == "wrong-version":
        value["reply_version"] = "2.0"
    elif change == "version-inside-batch":
        value["action_batch"]["reply_version"] = "1.0"
    elif change == "notes-inside-batch":
        value["action_batch"]["public_observations"] = value.pop("public_observations")
    elif change == "extra-top-level-key":
        value["thinking"] = "ignored"
    elif change == "leading-prose":
        return "commentary " + json.dumps(value)
    elif change == "extra-batch-key":
        value["action_batch"]["episode_id"] = "another-episode"
    elif change == "hoisted-batch":
        return json.dumps({"actions": value["action_batch"]["actions"], "public_observations": ""})
    elif change == "trailing-text":
        return raw + "\nHope that helps!"
    else:
        raise AssertionError(change)
    return json.dumps(value)


def _refused(change: str, raw: str, current: Any) -> str:
    """One change that must STILL cost the turn, as raw reply text."""

    value = json.loads(raw)
    if change == "two-batches-in-one-object":
        value["actions"] = json.loads(type_payload(current))["actions"]
        return json.dumps(value)
    if change == "nested-envelope":
        return json.dumps({"wrapper": value})
    if change == "nested-batch":
        value["action_batch"] = {"wrapper": value["action_batch"]}
        return json.dumps(value)
    if change == "duplicate-note":
        return raw.replace(
            '"public_observations":', '"public_observations":"other", "public_observations":'
        )
    if change == "duplicate-action":
        return raw.replace('"kind":', '"kind":"finish", "kind":')
    if change == "duplicate-version":
        return raw.replace('"reply_version":', '"reply_version":"2.0", "reply_version":')
    if change == "trailing-competing-batch":
        return raw + finish_payload(current)
    if change == "truncated-envelope":
        return raw[:-6]
    if change == "extra-action-key":
        # An action-level defect, which stays refused: the envelope around it
        # being tolerated says nothing about the actions inside it, and the
        # SIBLING-field tolerance does not reach a name no action kind declares.
        # This row used to put ``x`` on a ``type`` action, which the sibling rule
        # now drops; the field below is deliberately outside the vocabulary, and
        # a near-miss of a real field is pinned beside it in
        # ``test_the_field_tolerance_stops_at_the_vocabulary``.
        value["action_batch"]["actions"][0]["frobnicate"] = 1
        return json.dumps(value)
    if change == "decoy-batch-in-preamble":
        # The shape that REPLACED it as a refusal: a preamble carrying a second
        # decision-shaped object, so which one the model meant is a question
        # about meaning. One of the two binds another observation, which the
        # competing-batch rule cannot see -- the ambiguity is about which object
        # is the decision, not about which observation it names.
        return "commentary " + raw + " " + type_payload(observation(1))
    if change == "stale-observation-binding":
        value["action_batch"] = json.loads(type_payload(observation(1)))
        return json.dumps(value)
    raise AssertionError(change)


@pytest.mark.parametrize(
    "change",
    [
        "missing-version",
        "wrong-version",
        "version-inside-batch",
        "notes-inside-batch",
        "extra-top-level-key",
        "extra-batch-key",
        "hoisted-batch",
        "trailing-text",
    ],
)
def test_framing_is_normalised_and_never_costs_the_turn(change: str) -> None:
    """Framing is where the batch SITS, and it decides nothing.

    Each case here was a refusal on the canary arm whose payload was already a
    complete, executable decision: a ``reply_version`` that was missing, or the
    wrong value, or nested one level down; a notes key below instead of above;
    a key the contract has no use for; the same batch without its wrapper; and
    the correct envelope followed by text. The DECISION must be byte-identical
    to the legacy one, and the model's own note must survive the normalisation --
    it is memory, and dropping it would publish a reply the model did not send.
    """

    current = observation()
    legacy = parse_decision(type_payload(current), current, route=ROUTE)
    decision = parse_decision(
        _changed(change, _full_envelope(current), current), current, route=ROUTE
    )

    assert decision.action_batch.to_canonical_json() == legacy.action_batch.to_canonical_json()
    if change == "hoisted-batch":
        assert decision.public_reply is not None
        assert decode_public_reply(decision.public_reply)["public_observations"] == ""
    else:
        assert decision.public_reply is not None
        assert decode_public_reply(decision.public_reply)["public_observations"] == "visible fact"


@pytest.mark.parametrize(
    "variant",
    ["tool_name-parameters", "tool_call-input-string", "input-object"],
)
def test_a_generic_tool_call_serialization_is_unwrapped_and_decoded(variant: str) -> None:
    """A wrapper around a call is framing, so unwrapping it changes nothing.

    Regression pinned against the base tree: every one of these three verbatim
    shapes was refused there (``batch-shape`` / ``incomplete-json``) while
    carrying a complete, duplicate-free batch bound to the current observation.
    """

    current = observation()
    visible = _full_envelope(current, "Visible status: ready")
    legacy = parse_decision(type_payload(current), current, route=ROUTE)
    wrapped = parse_decision(_wrapped(variant, visible), current, route=ROUTE)

    assert wrapped.action_batch.to_canonical_json() == legacy.action_batch.to_canonical_json()
    assert decode_public_reply(wrapped.public_reply or "")["public_observations"] == (
        "Visible status: ready"
    )


#: One rejected reply VERBATIM from the OSWorld episode this tolerance was built
#: from -- the run ``judge5-20260921-231232``, episode ``ep-3f6be3110883``, the
#: refusal the run recorded as ``class: batch-shape`` at event sequence 94. Its
#: ``actions`` member is a STRING: the model opened a quote at the value and then
#: wrote the rest of its own envelope INSIDE it, so the string carries the array
#: followed by ``, "public_observations": "..."}``. Pinned byte for byte because
#: the shape IS the measurement -- 9 of that episode's 120 calls (7.5%) and 30 of
#: the campaign runs' 74 sealed refusals arrived this way, and each one cost a
#: whole paid call to repair with a re-prompt that repeated the mistake.
_MEASURED_STRING_ACTIONS_REPLY = '{"actions": "[{\\"kind\\": \\"type\\", \\"observation_id\\": \\"1da088ef2d7a5f6e3d811be671d17b792da92de57aeebce7de7d25e9dfc3321c\\", \\"text\\": \\"ls -la; ls city; ls filter\\"}, {\\"keys\\": [\\"enter\\"], \\"kind\\": \\"key\\", \\"observation_id\\": \\"1da088ef2d7a5f6e3d811be671d17b792da92de57aeebce7de7d25e9dfc3321c\\"}], \\"public_observations\\": \\"Verifying current terminal state and directory contents after ambiguous error output.\\"}"}'  # noqa: E501


def _string_actions_body(actions: Any, *, note: str | None = None) -> str:
    """The measured double-encoding, rebuilt around any action array.

    ``{"actions": [...], "public_observations": "..."}`` written as a STRING:
    the array with the envelope's own remaining member inside it, which is what
    all 30 of the campaign's sealed replies of this class do. Rebuilt rather
    than pasted so a case can vary the array while the spelling under test stays
    the measured one; the verbatim body above is pinned separately.
    """

    inner = json.dumps(actions)
    if note is not None:
        inner += f', "public_observations": {json.dumps(note)}' + "}"
    return json.dumps({"actions": inner})


def test_the_measured_string_actions_reply_is_decoded_and_accepted(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The verbatim reply the tolerance exists for, DECIDED instead of refused.

    Its actions are asserted one by one rather than counted, because the batch
    that executes has to be the batch the model stated inside the string -- a
    tolerance that salvaged the turn by running something else would be worse
    than the refusal it replaced. The note inside the tail is deliberately NOT
    recovered (see ``_actions_from_json_string``), which is why the reply
    publishes an empty one: the string is a JSON FRAGMENT, and finding where the
    model's object began is the guess the decoder refuses to make elsewhere.
    """

    with caplog.at_level(logging.WARNING):
        decoded = decode_public_reply(_MEASURED_STRING_ACTIONS_REPLY)

    assert decoded["actions"] == [
        {
            "kind": "type",
            "observation_id": "1da088ef2d7a5f6e3d811be671d17b792da92de57aeebce7de7d25e9dfc3321c",
            "text": "ls -la; ls city; ls filter",
        },
        {
            "kind": "key",
            "observation_id": "1da088ef2d7a5f6e3d811be671d17b792da92de57aeebce7de7d25e9dfc3321c",
            "keys": ["enter"],
        },
    ]
    assert decoded["public_observations"] == ""
    # The record a campaign counts this tolerance against. It is written when the
    # reply is ACCEPTED, so the line means what it says.
    assert "JSON-encoded string" in caplog.text


_STRING_ACTIONS_SPELLINGS = (
    "array-only",
    "array-and-envelope-tail",
    "single-action-array",
    "inside-action_batch",
    "action_batch-as-string",
    "inside-tool-call-wrapper",
)


@pytest.mark.parametrize("spelling", _STRING_ACTIONS_SPELLINGS)
def test_a_json_encoded_actions_string_is_the_same_decision(spelling: str) -> None:
    """One decision per spelling of "the array, written as a string".

    The DECISION must be the legacy batch byte for byte: this tolerance changes
    what the decoder can READ and never what executes. ``array-only`` is the
    clean case and ``array-and-envelope-tail`` is the shape the bundle actually
    carries; the last three pin that the string spelling follows the SAME keys
    and wrappers the array itself already may arrive under, instead of becoming
    a framing-dependent exception -- inside ``action_batch`` it is the same key,
    as a string in ``action_batch``'s own position it is the spelling this module
    already accepts for the array, and inside a generic tool call it is one
    decision behind two framing layers.
    """

    current = observation()
    actions = json.loads(type_payload(current))["actions"]
    if spelling == "single-action-array":
        actions = actions[:1]

    if spelling == "array-only":
        body = json.dumps({"actions": json.dumps(actions)})
    elif spelling == "array-and-envelope-tail":
        body = _string_actions_body(actions, note="Recovered from the string.")
    elif spelling == "single-action-array":
        body = _string_actions_body(actions, note="One action.")
    elif spelling == "inside-action_batch":
        body = json.dumps({"action_batch": {"actions": json.dumps(actions)}})
    elif spelling == "action_batch-as-string":
        # ``{"action_batch": [...]}`` is already an accepted spelling of the
        # array, so the string form of it is the same defect at the same place.
        # Refusing it here would make the tolerance depend on WHICH key the
        # array was framed under, which is the split this module refuses.
        body = json.dumps({"action_batch": json.dumps(actions)})
    else:
        body = _wrapped("tool_name-parameters", _string_actions_body(actions, note="Wrapped."))

    legacy = parse_decision(type_payload(current), current, route=ROUTE)
    decision = parse_decision(body, current, route=ROUTE)

    assert decision.action_batch.to_canonical_json() == legacy.action_batch.to_canonical_json()


@pytest.mark.parametrize(
    "spelling",
    ["not-json", "scalar", "list-of-scalars", "single-action-object", "empty", "empty-array"],
)
def test_a_string_that_is_not_an_action_array_keeps_its_refusal(spelling: str) -> None:
    """The tolerance may only ACCEPT a decision; it may never widen a refusal.

    Each of these is a string in the ``actions`` position that is not a
    non-empty array of action objects, so it keeps the refusal -- and the exact
    message -- it had before this change. ``single-action-object`` is here on
    purpose: a string holding ONE action object does not occur in any of the 29,
    and a second tolerance for a shape nobody has been observed to send is how a
    decoder stops being readable.
    """

    action = json.loads(type_payload(observation()))["actions"][0]
    spellings = {
        "not-json": "Sure, here is my batch",
        "scalar": "42",
        "list-of-scalars": json.dumps([1, 2]),
        "single-action-object": json.dumps(action),
        "empty": "",
        "empty-array": "[]",
    }
    body = json.dumps({"actions": spellings[spelling]})

    with pytest.raises(DecisionParseError) as info:
        parse_decision(body, observation(), route=ROUTE)

    assert "decision must carry a non-empty actions array" in str(info.value)


def test_a_top_level_encoded_actions_batch_still_refuses_a_competing_batch() -> None:
    """The outer ambiguity scan must see IDs through the accepted string spelling."""

    current = observation()
    actions = json.loads(type_payload(current))["actions"]
    body = json.dumps({"actions": json.dumps(actions)}) + " " + finish_payload(current)

    with pytest.raises(DecisionParseError, match="second action batch"):
        parse_decision(body, current, route=ROUTE)


def test_an_exhausted_competing_batch_scan_refuses_encoded_actions() -> None:
    """A bounded scan cannot treat unchecked trailing candidates as safe."""

    current = observation()
    actions = json.loads(type_payload(current))["actions"]
    harmless = " ".join(
        json.dumps({"unrelated": index}) for index in range(_MAX_TRAILING_DECODE_ATTEMPTS)
    )
    body = (
        json.dumps({"actions": json.dumps(actions)})
        + " "
        + harmless
        + " "
        + finish_payload(current)
    )

    with pytest.raises(DecisionParseError, match="second action batch"):
        parse_decision(body, current, route=ROUTE)


def test_a_competing_batch_hidden_in_the_string_still_refuses_the_turn() -> None:
    """The one invariant the tolerance is not allowed to trade away.

    That string is a JSON FRAGMENT -- the array, then the envelope's own tail --
    so the normaliser now reads text it used to refuse outright. A SECOND batch
    for this same observation inside that tail is the ambiguity
    ``_decode_leading_json`` refuses everywhere else: executing the first would
    run a decision the model superseded. Handed the bare array the scan behind
    that rule finds no observation id and stands down, which is why the coercion
    hands it a batch-shaped view of what it decoded.
    """

    current = observation()
    inner = json.dumps(json.loads(type_payload(current))["actions"])
    inner += f', "public_observations": {json.dumps("notes")}' + "}"
    inner += " " + finish_payload(current)
    body = json.dumps({"actions": inner})

    with pytest.raises(DecisionParseError):
        parse_decision(body, current, route=ROUTE)


def test_a_string_actions_reply_still_binds_to_this_observation() -> None:
    """The tolerance reads a SPELLING; it does not relax the binding rule.

    A batch naming another observation is the one failure worse than losing the
    turn, and it is refused here exactly as it is when the array arrives as an
    array: the coercion is applied before binding is ever considered, so the
    string spelling cannot become a way around it. The refusal is the action
    protocol's own class, not the framing decoder's -- which is the same
    layering the identical objects get without the string.
    """

    current = observation()
    stale = json.loads(type_payload(observation(1)))["actions"]

    with pytest.raises(DecisionParseError):
        parse_decision(_string_actions_body(stale, note="stale"), current, route=ROUTE)


def test_the_string_tolerance_adds_nothing_to_an_already_well_formed_reply(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A reply that is already an array pays nothing: same bytes, no record.

    The tolerance may not add a log line, an extra parse or a branch's worth of
    behaviour to the ordinary reply. The count it writes is the measure of a
    DEFECT, so a run of zeros has to keep meaning "the models stopped sending
    this" -- which is only true if nothing writes it on the healthy path.
    """

    current = observation()
    legacy = parse_decision(type_payload(current), current, route=ROUTE)
    with caplog.at_level(logging.WARNING):
        visible = parse_decision(
            _full_envelope(current, "Visible status: ready"), current, route=ROUTE
        )

    assert visible.action_batch.to_canonical_json() == legacy.action_batch.to_canonical_json()
    assert decode_public_reply(visible.public_reply or "")["public_observations"] == (
        "Visible status: ready"
    )
    assert "JSON-encoded string" not in caplog.text


@pytest.mark.parametrize(
    "change",
    [
        "two-batches-in-one-object",
        "nested-envelope",
        "nested-batch",
        "duplicate-note",
        "duplicate-action",
        "duplicate-version",
        "trailing-competing-batch",
        "decoy-batch-in-preamble",
        "stale-observation-binding",
    ],
)
def test_ambiguity_and_broken_input_still_cost_the_turn(change: str) -> None:
    """The refusals that survive, each for a reason about MEANING or reading.

    Two action arrays that could each be the decision, a batch object that
    carries none, duplicated JSON keys, a second batch for the same observation,
    a preamble carrying a decoy batch (an object the model may or may not have
    meant as its answer), and a batch bound to another observation are all still
    refused. Widening what counts as framing was never allowed to widen what
    counts as a decision.

    The decoy case is what keeps the leading tolerance honest: the hunt for the
    object behind a preamble counts DECISIONS, not braces, and reads the reply
    only when exactly one exists. A single-brace skip would execute the decoy,
    and a "first batch wins" rule would execute a batch the model superseded.
    """

    current = observation()
    raw = _refused(change, _full_envelope(current), current)

    with pytest.raises(DecisionParseError):
        parse_decision(raw, current, route=ROUTE)


def test_a_reply_bound_to_another_observation_keeps_its_own_class() -> None:
    """The binding check is untouched by the framing tolerance.

    A decision made about a screen the environment has already moved past is the
    one failure worse than losing the turn, so the class key the arm counts it
    under (``observation-binding``) must keep its own diagnostic. This is the
    property the scope correction singled out: normalise framing, never meaning.
    """

    current = observation()
    stale = observation(1)
    raw = json.dumps(
        {
            "action_batch": {"actions": json.loads(type_payload(stale))["actions"]},
            "public_observations": "",
        }
    )

    with pytest.raises(DecisionParseError) as info:
        parse_decision(raw, current, route=ROUTE)

    assert "bind to a different observation_id" in str(info.value)


def test_legacy_trailing_envelope_cannot_supersede_current_decision() -> None:
    current = observation()
    with pytest.raises(DecisionParseError, match="second action batch"):
        parse_decision(
            type_payload(current) + envelope(finish_payload(current)), current, route=ROUTE
        )
    # Preserve the original tolerance for old quoted decisions.
    decision = parse_decision(
        type_payload(current) + envelope(finish_payload(observation(1))), current, route=ROUTE
    )
    assert decision.public_reply is None


@pytest.mark.asyncio
async def test_a_rejected_envelope_is_still_placeholdered_in_history(
    tmp_path: Path,
) -> None:
    """The history boundary does not move with the framing tolerance.

    A rejected reply that carried notes is withheld from corrective history: its
    notes are unvalidated text, so the model is corrected from the hint rather
    than from its own words (the F1 barrier). Pinned on a rejection the framing
    tolerance does NOT absorb -- a batch bound to another observation -- because
    that is what makes the point: widening what the decoder ACCEPTS must never
    widen what an unvalidated reply may PUBLISH.
    """

    note = "rejected-note-must-not-enter-memory"
    raw = json.dumps(
        {
            "action_batch": {"actions": json.loads(type_payload(observation(1)))["actions"]},
            "public_observations": note,
        }
    )
    stream = RecordingStream(raw)
    client = _client(stream, tmp_path)
    turns = [EpisodeTurn(observation=observation())]

    with pytest.raises(DecisionRejected) as error:
        await client.decide(observation(), turns)

    assert error.value.class_key == "observation-binding"
    assert error.value.reply == REJECTED_PUBLIC_REPLY

    stream.reply = finish_payload(observation())
    await client.decide(observation(), turns)
    replay = "\n".join(message.text for message in stream.requests[-1].messages)

    assert REJECTED_PUBLIC_REPLY in replay
    assert note not in replay
    assert '"observation_id" of the observation being answered' in replay


@pytest.mark.asyncio
@pytest.mark.parametrize("escaped_keys", [False, True])
async def test_rejected_envelope_notes_are_not_replayed_as_facts(
    tmp_path: Path,
    escaped_keys: bool,
) -> None:
    secret = "invalid-note-must-not-enter-memory"
    raw = envelope(type_payload(observation(1)), secret)
    if escaped_keys:
        raw = raw.replace("public_observations", "public_\\u006fbservations")
        raw = raw.replace("reply_version", "reply_\\u0076ersion")
        raw = raw.replace("action_batch", "action_\\u0062atch")
    stream = RecordingStream(raw)
    client = _client(stream, tmp_path)
    turns = [EpisodeTurn(observation=observation())]
    with pytest.raises(DecisionRejected) as error:
        await client.decide(observation(), turns)
    assert error.value.reply == REJECTED_PUBLIC_REPLY
    stream.reply = finish_payload(observation())
    await client.decide(observation(), turns)
    replay = "\n".join(message.text for message in stream.requests[-1].messages)
    assert secret not in replay
    assert REJECTED_PUBLIC_REPLY in replay
    assert not stream.summary_requests


def _escaped_envelope(raw: str, secret: str) -> str:
    """The F1 encoding: reserved keys and the known note in JSON escapes."""

    for key, escaped in (
        ("public_observations", "public_\\u006fbservations"),
        ("reply_version", "reply_\\u0076ersion"),
        ("action_batch", "action_\\u0062atch"),
    ):
        raw = raw.replace(key, escaped)
    return raw.replace(secret, secret.replace("fixture", "\\u0066ixture"))


@pytest.mark.asyncio
@pytest.mark.parametrize("variant", ["truncated-end", "truncated-key", "truncated-value"])
async def test_truncated_escaped_envelope_with_escaped_known_note_fails_closed(
    tmp_path: Path, variant: str
) -> None:
    """F1 (review round 1): a truncated envelope whose reserved keys and known
    note are JSON Unicode-escaped bypassed literal filters and leaked the note
    into corrective context, the next provider request and rejection evidence."""

    secret = "fixture-canary-note-653"
    escaped = secret.replace("fixture", "\\u0066ixture")
    raw = _escaped_envelope(envelope(type_payload(observation()), secret), secret)
    if variant == "truncated-end":
        raw = raw[:-1]  # exact F1 shape: missing the final closing brace
    elif variant == "truncated-key":
        raw = raw[: raw.index('"public_') + len('"public_')]  # framing dies mid-key
    else:
        raw = raw[: raw.index(escaped) + 6]  # framing dies inside the note
    stream = RecordingStream(raw)
    client = _client(stream, tmp_path)
    turns = [EpisodeTurn(observation=observation())]
    with pytest.raises(DecisionRejected) as error:
        await client.decide(observation(), turns)
    assert error.value.reply == REJECTED_PUBLIC_REPLY
    stream.reply = finish_payload(observation())
    await client.decide(observation(), turns)
    replay = "\n".join(message.text for message in stream.requests[-1].messages)
    for canary in (secret, escaped):
        assert canary not in replay
    assert REJECTED_PUBLIC_REPLY in replay


@pytest.mark.asyncio
async def test_f1_runner_repro_leaves_no_secret_in_bundle_or_replay(
    tmp_path: Path, episode_id: str
) -> None:
    """The exact reviewer reproduction through the real EpisodeRunner: the
    next request and every retained artifact stay clean, and the bundle still
    verifies -- the fix removes the leak, it does not hide an invalid bundle."""

    secret = "fixture-canary-runner-653"
    escaped = secret.replace("fixture", "\\u0066ixture")
    calls = 0

    def reply(message: Message) -> str:
        nonlocal calls
        calls += 1
        batch = _wait_reply(message)
        if calls == 1:
            return _escaped_envelope(envelope(batch, secret), secret)[:-1]
        oid = json.loads(batch)["actions"][0]["observation_id"]
        return json.dumps(
            {
                "actions": [
                    {
                        "kind": "finish",
                        "observation_id": oid,
                        "status": "done",
                        "reason": "complete",
                    }
                ]
            }
        )

    config = build_config(tmp_path)
    stream = RecordingStream(reply)
    client = _client(stream, config.artifact_root)
    runner = EpisodeRunner(
        build_spec(episode_id),
        config,
        selector=selector(tmp_path),
        model=client,
        launch=lambda _: FakeAdapter(tmp_path, episode_id),
        rescue=_rescue_ok,
        redactions=RedactionSet.from_resolved_values([secret]),
    )
    outcome = await runner.run()
    root = outcome.bundle_root
    assert root is not None and len(stream.requests) >= 2
    replay = "\n".join(m.text for m in stream.requests[1].messages)
    for canary in (secret, escaped):
        assert canary not in replay
        assert not any(
            canary.encode() in path.read_bytes() for path in root.rglob("*") if path.is_file()
        )
    assert verify_bundle(root).valid


@pytest.mark.asyncio
async def test_a_rejected_envelope_reaches_the_bundle_with_its_class(
    tmp_path: Path, episode_id: str
) -> None:
    """The reply is evidence even when it is NOT replayable history.

    An envelope-shaped reply is withheld from the corrective history because its
    notes are unvalidated text; the bundle is a different boundary, and
    withholding it there is what made 273 of the campaign's 280 rejection
    artifacts unreadable. Both halves are asserted on the same run: the
    artifact carries the reply and the class key, the next request carries the
    placeholder, and neither carries the other's rendering.

    The reply declares a ``reply_version`` this build does not serve (``9.9``)
    on purpose. It must no longer be the reason for anything: what is refused
    here is the DECISION -- a batch bound to another observation -- and the
    version is simply ignored, which is exactly the split the scope correction
    asked for.
    """

    rejected_reply = json.dumps(
        {
            "reply_version": "9.9",
            "action_batch": {"actions": json.loads(type_payload(observation(1)))["actions"]},
            "public_observations": "visible fact",
        }
    )
    calls = 0

    def reply(message: Message) -> str:
        nonlocal calls
        calls += 1
        if calls == 1:
            return rejected_reply
        return _wait_reply(message)

    config = build_config(tmp_path)
    stream = RecordingStream(reply)
    client = _client(stream, config.artifact_root)
    runner = EpisodeRunner(
        build_spec(episode_id),
        config,
        selector=selector(tmp_path),
        model=client,
        launch=lambda _: FakeAdapter(tmp_path, episode_id),
        rescue=_rescue_ok,
        # No canaries: this run is about the boundary, not about redaction --
        # the F1 tests above cover the withheld case.
        redactions=RedactionSet.from_resolved_values(()),
    )

    outcome = await runner.run()

    root = outcome.bundle_root
    assert root is not None and len(stream.requests) >= 2
    texts = [
        path.read_text(encoding="utf-8", errors="replace")
        for path in root.rglob("*")
        if path.is_file()
    ]
    assert any(
        rejected_reply in text and "class: observation-binding" in text for text in texts
    ), "the rejected reply and its class are in the bundle"
    replay = "\n".join(message.text for message in stream.requests[1].messages)
    assert REJECTED_PUBLIC_REPLY in replay
    assert rejected_reply not in replay
    assert verify_bundle(root).valid


@pytest.mark.parametrize(
    "transform",
    [
        lambda raw, secret: raw,
        lambda raw, secret: raw.replace("fixture", "\\u0066ixture"),
        lambda raw, secret: raw.replace(secret, quote(secret, safe="")),
        lambda raw, secret: raw.replace(secret, base64.b64encode(secret.encode()).decode()),
        # Escaping COMPOSES, so the scan has to decode more than one level. Both
        # shapes below were reproduced by the round-1 review as surviving a
        # single-level scan: the first carries an ESCAPED BACKSLASH before the
        # escape (a canary inside a JSON string), the second escapes every
        # character and then escapes the escapes again.
        lambda raw, secret: raw.replace(secret, secret.replace("fixture", "\\\\u0066ixture")),
        lambda raw, secret: raw.replace(
            secret,
            "".join(f"\\u{ord(character):04x}" for character in secret).replace("\\", "\\\\"),
        ),
    ],
    ids=[
        "plain",
        "json-unicode-escape",
        "percent",
        "base64",
        "json-escaped-twice",
        "per-character-double-escape",
    ],
)
def test_a_reply_carrying_a_canary_is_withheld_whole_from_evidence(transform: Any) -> None:
    """Evidence publishes the reply ONLY through an escape-aware scan.

    The reply reaches evidence as WIRE bytes, so a canary can be spelt with JSON
    unicode escapes, percent escapes, or an encoding, and a substring check on
    the raw text sees none of them. It is also the shape that is most likely to
    be TRUNCATED (that is usually why it was refused), so the scan cannot lean
    on a JSON parse -- the raw and unescaped renderings are checked regardless.

    Withheld WHOLE, never partially: ``assert_clear`` matches a substring, so
    publishing a masked or cut rendering is how a canary stops matching the
    alarm that exists to catch it.
    """

    secret = "fixture/canary-evidence-911"
    redactions = RedactionSet.from_resolved_values([secret])
    raw = envelope(type_payload(observation()), secret)
    reply = transform(raw, secret)
    if reply != raw:
        # The point of the case: a substring check on the raw text sees nothing.
        assert secret not in reply

    published = rejected_reply_evidence(reply, redactions)

    assert published == REJECTED_REPLY_WITHHELD
    assert secret not in published
    assert reply not in published


@pytest.mark.asyncio
async def test_an_episode_runs_on_wrapper_framed_replies(tmp_path: Path, episode_id: str) -> None:
    """The framing the arm actually emitted, through a whole assembled episode.

    The unit tests prove the decoder; only this proves the EPISODE. Each reply
    of a real run (the runner, the real client, a fake adapter and a verified
    bundle) arrives inside one of the three generic tool-call wrappers -- the
    shape 20 of the arm's refusals had -- and the run must complete: a wrapped
    reply costs no turn, and the notes it carries still reach the next request,
    which is the model's only cross-turn memory on a screenshot-only benchmark.
    """

    note = "Visible status: wrapped and still carried"
    wrappers = ["tool_name-parameters", "tool_call-input-string", "input-object"]
    calls = 0

    def reply(message: Message) -> str:
        nonlocal calls
        calls += 1
        raw = _wait_reply(message)
        if calls == 2:
            oid = json.loads(raw)["actions"][0]["observation_id"]
            raw = json.dumps(
                {
                    "actions": [
                        {
                            "kind": "finish",
                            "observation_id": oid,
                            "status": "done",
                            "reason": "complete",
                        }
                    ]
                }
            )
        return _wrapped(wrappers[(calls - 1) % len(wrappers)], envelope(raw, note))

    config = build_config(tmp_path)
    stream = RecordingStream(reply)
    client = _client(stream, config.artifact_root)
    runner = EpisodeRunner(
        build_spec(episode_id),
        config,
        selector=selector(tmp_path),
        model=client,
        launch=lambda _: FakeAdapter(tmp_path, episode_id),
        rescue=_rescue_ok,
        redactions=RedactionSet.from_resolved_values([]),
    )
    outcome = await runner.run()

    assert outcome.status == "completed"
    root = outcome.bundle_root
    assert root is not None
    assert verify_bundle(root).valid
    assert calls == 2
    replay = "\n".join(message.text for message in stream.requests[1].messages)
    assert note in replay and "was rejected" not in replay
    responses = payloads(root, ModelResponsePayload)
    recorded_ref = responses[0].redacted_response
    assert recorded_ref is not None
    recorded = (root / "artifacts" / recorded_ref.sha256).read_text()
    assert json.loads(recorded)["public_observations"] == note
    batches = payloads(root, ActionBatchPayload)
    batch = json.loads((root / "artifacts" / batches[0].action_artifact.sha256).read_text())
    assert batch["actions"] == json.loads(recorded)["actions"]


@pytest.mark.asyncio
async def test_a_tolerated_reply_reaches_the_bundle_with_its_counts(
    tmp_path: Path, episode_id: str
) -> None:
    """Both tolerances' rates are readable from a SEALED, verified bundle.

    Why this is a bundle test and not one more unit assertion on the decision: a
    reply the tolerance RECOVERS produces no rejection artifact, so without these
    counts the class left the bundle the moment the tolerance started working --
    which is exactly when a campaign needs to know whether the model is still
    confused. The first reply needs BOTH tolerances at once (a decision behind a
    preamble, and ``frame_id`` on a kind that does not take it); the second needs
    neither, so the zeros are a measurement rather than an absence.
    """

    preamble = "Working out where to wait.\n"
    note = "Visible status: tolerated and still carried"
    calls = 0

    def reply(message: Message) -> str:
        nonlocal calls
        calls += 1
        raw = _wait_reply(message)
        if calls == 1:
            batch = json.loads(raw)
            batch["actions"][0]["frame_id"] = "screen"
            return preamble + envelope(json.dumps(batch), note)
        oid = json.loads(raw)["actions"][0]["observation_id"]
        return envelope(
            json.dumps(
                {
                    "actions": [
                        {
                            "kind": "finish",
                            "observation_id": oid,
                            "status": "done",
                            "reason": "complete",
                        }
                    ]
                }
            ),
            note,
        )

    config = build_config(tmp_path)
    stream = RecordingStream(reply)
    client = _client(stream, config.artifact_root)
    runner = EpisodeRunner(
        build_spec(episode_id),
        config,
        selector=selector(tmp_path),
        model=client,
        launch=lambda _: FakeAdapter(tmp_path, episode_id),
        rescue=_rescue_ok,
        redactions=RedactionSet.from_resolved_values([]),
    )
    outcome = await runner.run()

    assert outcome.status == "completed"
    root = outcome.bundle_root
    assert root is not None
    assert verify_bundle(root).valid
    responses = payloads(root, ModelResponsePayload)
    assert responses[0].leading_framing_bytes == len(preamble)
    assert responses[0].tolerated_action_fields == 1
    assert responses[1].leading_framing_bytes == 0
    assert responses[1].tolerated_action_fields == 0


def test_an_unsafe_reply_degrades_without_losing_the_rejection() -> None:
    """Withholding the reply must not withhold the rejection.

    A withheld section is still a readable one because the diagnostic, the class
    and the stream shape are all harness-owned text -- which is what lets the
    evidence boundary differ from the history boundary without making a bundle
    unreadable.
    """

    rejected = DecisionRejected(
        "Your previous reply was rejected: the reply declared a bad version",
        reply=REJECTED_PUBLIC_REPLY,
        evidence_reply='{"public_observations": "fixture-canary"}',
        class_key="unsupported-reply-version",
        stream_shape=StreamShape(content_deltas=2, reasoning_deltas=3, stop="stop"),
    )
    redactions = RedactionSet.from_resolved_values(["fixture-canary"])

    detail = _rejection_detail(rejected, redactions)

    assert REJECTED_REPLY_WITHHELD in detail
    assert "fixture-canary" not in detail
    assert "class: unsupported-reply-version" in detail
    assert "reasoning_deltas=3" in detail
    # The history rendering is NOT what evidence shows: the placeholder belongs
    # to the correction, and pasting it here would read as "the model said this".
    assert REJECTED_PUBLIC_REPLY not in detail


def test_a_marker_that_imitates_a_header_cannot_open_another_section() -> None:
    """Provider-owned text goes into the header ESCAPED, never raw.

    The artifact promises a fixed section order that a script can read, and the
    stop marker is the one part of that header the harness does not author. A
    marker carrying a newline would otherwise write a line that reads like
    another header (or hide the rest of the section), so non-printables are
    escaped rather than truncated -- a 64-character bound does not neutralise a
    short injection.
    """

    rejected = DecisionRejected(
        "Your previous reply was rejected: refused",
        reply="{}",
        class_key="malformed-json",
        stream_shape=StreamShape(content_deltas=1, stop="stop\nclass: extra-action-key"),
    )

    detail = _rejection_detail(rejected, None)

    assert [line for line in detail.splitlines() if line.startswith("class: ")] == [
        "class: malformed-json"
    ]
    assert "stop\\nclass: extra-action-key" in detail


def test_reserved_key_scan_preserves_legacy_rejection_replay() -> None:
    """No reserved key means legacy raw-text corrective replay is unchanged;
    escaped or truncated reserved keys fail closed on undecodable output."""

    assert not looks_like_public_reply("x" * 200)
    assert not looks_like_public_reply('{"actions": [{"kind": "click"}]}')
    assert not looks_like_public_reply('{"public_note": "not a reserved key"}')
    assert looks_like_public_reply('{"public_\\u006fbservations": ""}')
    assert looks_like_public_reply('{"public_observations": "')
    assert looks_like_public_reply('{"reply_\\u0076ersion": "1.0"}')
    assert looks_like_public_reply('{"action_batch": {"actions": []}')


@pytest.mark.parametrize(
    "transform",
    [
        str,
        lambda s: quote(s, safe=""),
        lambda s: base64.b64encode(s.encode()).decode(),
    ],
)
def test_resolved_secret_notes_are_redacted_before_replay(transform: Any) -> None:
    secret = "known-secret/credential-value"
    raw = envelope(type_payload(observation()), transform(secret))
    # JSON escapes cannot hide decoded credentials from the existing boundary.
    raw = raw.replace("known", "\\u006bnown")
    clean = redact_public_reply(raw, RedactionSet.from_resolved_values([secret]))
    recorded = json.loads(clean)
    assert recorded["public_observations"] == "[redacted public observations]"
    # The redaction rewrites the note and nothing else: the accepted reply is
    # normalised to the contract's one shape, so its actions are the ones the
    # model sent, byte for byte.
    assert recorded["actions"] == json.loads(raw)["action_batch"]["actions"]
    assert secret not in clean


def test_contract_identity_is_separate_from_action_surface() -> None:
    metadata = public_reply_contract()
    contract = json.loads(metadata["model_reply_contract"])
    assert metadata["model_reply_contract_digest"] == canonical_digest(
        "runner-model-reply-v1", contract
    )
    schema = contract["schema"]
    assert schema["required"] == ["actions"]
    assert schema["properties"]["public_observations"]["maxLength"] == MAX_PUBLIC_OBSERVATIONS_CHARS
    # The closed envelope is gone from the published contract: no version to pin
    # and no wrapper to nest, so a reader of a bundle can see that a reply needs
    # neither without reading the decoder.
    assert "action_batch" not in schema["properties"]
    assert "reply_version" not in schema["properties"]
    assert contract["accepted_framings"]
    assert "legacy_plain_action_batch" not in contract
    prompt = build_system_prompt()
    assert '"actions": [ ... ], "public_observations": ""' in prompt
    assert "reply_version" not in prompt
    assert "concise NEW factual data" in prompt
    assert "private reasoning" in prompt and "credentials/secrets" in prompt


def test_context_builder_retains_notes_as_append_only_shared_messages(tmp_path: Path) -> None:
    first, second = observation(), observation(1)
    decision = parse_decision(
        envelope(type_payload(first), "Visible status: ready"), first, route=ROUTE
    )
    context = _ContextBuilder(artifact_root=tmp_path, keep_recent_frames=3, rebuild_every_frames=12)
    context.append_new_turns([EpisodeTurn(observation=first)])
    prefix = list(context.messages)
    turns = [
        EpisodeTurn(
            observation=first, batch=decision.action_batch, public_reply=decision.public_reply
        ),
        EpisodeTurn(observation=second),
    ]
    context.append_new_turns(turns)
    assert context.messages[0] is prefix[0]
    assert context.messages[1].role == "assistant"
    assert all(isinstance(block, TextContent) for block in context.messages[1].content)
    assert context.messages[1].text == decision.public_reply
    closed = list(context.messages)
    context.append_new_turns(turns)
    assert all(a is b for a, b in zip(closed, context.messages, strict=True))


def image_history(variant: str, *, notes: bool) -> list[Message]:
    history = []
    for index in range(12):
        # IDs/text/actions are fixed: only the oldest screenshot's bytes vary.
        pixels = variant if index == 0 else f"unchanged-frame-{index}"
        history.append(
            Message.user(
                f"Observation {index}",
                [
                    ImageContent(
                        data=base64.b64encode(pixels.encode()).decode(),
                        mime_type="image/png",
                    )
                ],
            )
        )
        batch = type_payload(observation(index))
        fact = f"Visible label: {variant}" if index == 0 else ""
        reply = envelope(batch, fact) if notes else batch
        history.append(Message.assistant(reply))
    # Message IDs default to UUIDs; fix them so the paired controls genuinely
    # differ only in the old image (and, for the treatment, its public note).
    return [
        message.model_copy(update={"id": f"message-{index}"})
        for index, message in enumerate(history)
    ]


@pytest.mark.asyncio
async def test_twelve_frame_erasure_and_public_facts_after_prune_then_text_compaction() -> None:
    model = ModelSpec(provider="provider", model_id="model", context_window=128_000)

    async def prune(history: list[Message]) -> Any:
        return await run_compaction_pass(
            history,
            model=model,
            settings=CompactionSettings(keep_recent_frames=3),
            summarize=None,
            now_ms=10_000,
            last_activity_ms=10_000,
        )

    left, right = image_history("violet", notes=False), image_history("amber", notes=False)
    assert left != right
    # Neither serializer could rescue pixels simply by running before prune.
    assert serialize_conversation(left) == serialize_conversation(right)
    assert serialize_for_snapcompact(left) == serialize_for_snapcompact(right)
    erased_left, erased_right = await prune(left), await prune(right)
    assert erased_left.frames_dropped == erased_right.frames_dropped == 9
    assert not erased_left.ran and not erased_right.ran
    assert erased_left.messages == erased_right.messages

    summaries = []
    for variant in ("violet", "amber"):
        result = await prune(image_history(variant, notes=True))
        assert result.frames_dropped == 9 and not result.ran
        fact = f"Visible label: {variant}"
        assert fact in serialize_conversation(result.messages)
        assert fact in serialize_for_snapcompact(result.messages)
        prompts = []

        async def summarize(prompt: str) -> str:
            prompts.append(prompt)
            # Only extract from the actual serialized request, never inject
            # an external expected fact into the compactor's output.
            return next(
                f"Visible label: {v}"
                for v in ("violet", "amber")
                if f"Visible label: {v}" in prompt
            )

        compacted = await run_compaction_pass(
            result.messages,
            model=model,
            settings=CompactionSettings(keep_recent_tokens=100, strategy="context-full"),
            summarize=summarize,
            now_ms=10_000,
            last_activity_ms=10_000,
            respect_threshold=False,
        )
        assert compacted.ran and len(prompts) == 1
        assert fact in compacted.messages[0].text
        summaries.append(compacted.messages)
    assert summaries[0] != summaries[1]


@pytest.mark.asyncio
@pytest.mark.parametrize("secret", [False, True])
async def test_real_provider_runner_records_and_replays_public_evidence(
    tmp_path: Path,
    episode_id: str,
    secret: bool,
) -> None:
    note = "known-secret/credential-value" if secret else "Visible status: ready"
    calls = 0

    def reply(message: Message) -> str:
        nonlocal calls
        calls += 1
        raw = _wait_reply(message)
        if calls == 2:
            oid = json.loads(raw)["actions"][0]["observation_id"]
            raw = json.dumps(
                {
                    "actions": [
                        {
                            "kind": "finish",
                            "observation_id": oid,
                            "status": "done",
                            "reason": "complete",
                        }
                    ]
                }
            )
        return envelope(raw, note if calls == 1 else "")

    config = build_config(tmp_path)
    stream = RecordingStream(reply)
    client = _client(stream, config.artifact_root)
    runner = EpisodeRunner(
        build_spec(episode_id),
        config,
        selector=selector(tmp_path),
        model=client,
        launch=lambda _: FakeAdapter(tmp_path, episode_id),
        rescue=_rescue_ok,
        redactions=RedactionSet.from_resolved_values([note] if secret else []),
    )
    outcome = await runner.run()
    assert outcome.status == "completed"
    root = outcome.bundle_root
    assert root is not None
    report = verify_bundle(root)
    assert report.valid, [issue.code for issue in report.issues]
    responses = payloads(root, ModelResponsePayload)
    assert len(responses) == calls == 2 and not stream.summary_requests
    ref = responses[0].redacted_response
    assert ref is not None
    recorded = (root / "artifacts" / ref.sha256).read_text()
    expected = "[redacted public observations]" if secret else note
    assert json.loads(recorded)["public_observations"] == expected
    replay = next(m.text for m in stream.requests[1].messages if m.role == "assistant")
    assert replay == recorded
    batches = payloads(root, ActionBatchPayload)
    batch = json.loads((root / "artifacts" / batches[0].action_artifact.sha256).read_text())
    assert batch["actions"] == json.loads(recorded)["actions"]
    manifest = json.loads((root / "manifest.json").read_text())
    assert all(manifest["metadata"][key] == value for key, value in public_reply_contract().items())
    request = payloads(root, ModelRequestPayload)[0]
    assert request.tool_schema_digest == canonical_digest(
        "runner-tool-schema-v1", json.loads(manifest["metadata"]["action_surface"])
    )
    if secret:
        assert note not in "\n".join(m.text for m in stream.requests[1].messages)
        assert not any(
            note.encode() in path.read_bytes() for path in root.rglob("*") if path.is_file()
        )


@pytest.mark.asyncio
async def test_old_fake_client_does_not_claim_a_new_reply_contract(
    tmp_path: Path,
    episode_id: str,
) -> None:
    runner = EpisodeRunner(
        build_spec(episode_id),
        build_config(tmp_path),
        selector=selector(tmp_path),
        model=ScriptedModel(["finish"]),
        launch=lambda _: FakeAdapter(tmp_path, episode_id),
        rescue=_rescue_ok,
    )
    outcome = await runner.run()
    root = outcome.bundle_root
    assert root is not None and verify_bundle(root).valid
    manifest = json.loads((root / "manifest.json").read_text())
    assert "model_reply_contract" not in manifest["metadata"]
    assert payloads(root, ModelResponsePayload)[0].redacted_response is None
    assert decode_public_reply(envelope(type_payload(observation())))


def test_a_reply_with_no_usable_actions_states_the_accepted_shape() -> None:
    """The one batch-shape refusal left, and it names the accepted shape.

    Two spellings of the same defect: an ``action_batch`` whose contents are not
    an actions array at all, and an empty one. There is no decision in either, so
    both are refused -- but the refusal states what to send instead, because a
    bare rule is not something a model can act on (the doctrine
    ``rejection_hint`` is written to). The keys that landed in the batch are no
    longer named: a sibling key no longer refuses the reply at all, so the only
    bytes this branch speaks about are the accepted shape.
    """

    actions = json.loads(type_payload(observation()))["actions"]
    bodies = [
        json.dumps({"action_batch": {"actions": []}, "public_observations": ""}),
        json.dumps({"action_batch": {"wrapper": {"actions": actions}}}),
    ]

    for body in bodies:
        with pytest.raises(ValueError) as info:
            decode_public_reply(body)
        assert '{"actions": [...]}' in str(info.value)


def test_an_envelope_with_an_extra_key_is_ignored_and_reported(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The other near-miss: every key the contract uses, plus one more.

    It used to cost the turn. It is framing -- the actions decode, and a key the
    contract has no use for cannot change what executes -- so it is ignored. Not
    silently, though: an unexpected key is a signal (usually a model guessing at
    a shape it was not given), and the warning is where a reader can still see
    it.
    """

    actions = json.loads(type_payload(observation()))["actions"]
    body = json.dumps(
        {"action_batch": {"actions": actions}, "public_observations": "", "notes": "x"}
    )

    with caplog.at_level(logging.WARNING):
        decoded = decode_public_reply(body)

    assert decoded["actions"] == actions
    assert "'notes'" in caplog.text


@pytest.mark.parametrize(
    "extra",
    [
        {"X" * 50_000: "y"},
        {f"key{index}" * 20: index for index in range(60)},
        {"\U000e0001" * 40: 1},
        {"SECRET-" + "z" * 60: 1},
        # Every key here is individually SAFE to quote, so only the count cap
        # stands between the model's payload and the log line. Without a case
        # like this the cap is unpinned: the other fixtures withhold every key
        # on safety grounds and never exercise it.
        {f"k{index}": index for index in range(60)},
    ],
    ids=[
        "one-enormous-key",
        "many-long-keys",
        "repr-expanding-key",
        "long-secret-key",
        "many-short-safe-keys",
    ],
)
def test_unexpected_keys_cannot_inflate_the_log_line(
    extra: dict[str, object], caplog: pytest.LogCaptureFixture
) -> None:
    """Model-supplied key names never leave this module reshaped or unbounded.

    The boundary moved with the contract -- the names used to be quoted into a
    retry PROMPT, and are now quoted into a log LINE -- and the bound is the same
    one, so it is pinned against the same payloads. Quoting a key whole is what
    makes a 50,000-character key a 50,000-character retry prompt, which neither
    ``MAX_REJECTED_REPLY_CHARS`` nor ``_diagnostic``'s cap intercepts.

    Quoted WHOLE or not at all. The two cases beyond mere size are why truncating
    was not good enough:

    ``repr-expanding-key`` -- ``repr`` expands escapes AFTER a cut, so a bound
    applied to the raw key does not bound the rendered output, and the limit
    silently depends on the input's alphabet.

    ``long-secret-key`` -- and this is the one that matters: cutting converts a
    LOUD failure into a SILENT one. ``_assert_redacted`` is substring-based, so a
    secret longer than the cut survives as a prefix that no longer matches the
    canary. The leak stops tripping the alarm that exists to catch it.
    """

    actions = json.loads(type_payload(observation()))["actions"]
    payload = json.dumps(
        {
            "reply_version": "1.0",
            "action_batch": {"actions": actions},
            "public_observations": "",
            **extra,
        }
    )

    with caplog.at_level(logging.WARNING):
        decode_public_reply(payload)

    message = caplog.records[-1].getMessage()
    # No fragment of an unsafe key escapes: whole-or-nothing, so a redaction
    # canary still matches and nothing is rendered in a reshaped form.
    quoted = sum(1 for key in extra if repr(key) in message)
    assert quoted <= _MAX_EXTRA_KEYS_SHOWN, f"{quoted} keys quoted"
    for key in extra:
        if repr(key) in message:
            # A quoted key must appear WHOLE and unreshaped, never as a cut.
            continue
        assert key not in message
        assert key[:_MAX_EXTRA_KEY_CHARS] not in message
    # The count is still reported, so the reader learns how many keys were
    # dropped even when their names are withheld entirely.
    assert f"carried {len(extra)} key(s)" in message
    # 600 is the reachable worst case for five 40-character quotable keys plus
    # the fixed template and the " and N more" suffix, measured by brute force
    # in review round 5 -- a bound below it would assert a property the code does
    # not have. Growth is logarithmic in the key count.
    assert len(message) < 600, f"log line grew to {len(message)} characters"


def test_a_short_plain_unexpected_key_is_still_named(caplog: pytest.LogCaptureFixture) -> None:
    """Withholding is for unsafe keys only; the useful case stays useful.

    Naming keys is what made the old diagnostic corrective (9/10 recovered
    against 4/10 for the bare rule), so a bound that withheld every name would
    protect the line by making it useless.
    """

    actions = json.loads(type_payload(observation()))["actions"]
    body = json.dumps(
        {
            "version": "1.0",
            "actions": actions,
            "public_observations": "",
            "notes": "x",
            "extra": 1,
        }
    )

    with caplog.at_level(logging.WARNING):
        decode_public_reply(body)

    assert "'extra'" in caplog.text and "'notes'" in caplog.text
    assert "not shown" not in caplog.text


def test_a_non_object_reply_keeps_the_plain_rule() -> None:
    """A JSON array names no keys, so there is no defect to describe."""

    with pytest.raises(ValueError) as info:
        decode_public_reply("[1, 2]")

    assert "decision must be a JSON object" in str(info.value)


#: The three generic tool-call serializations, as ``_wrapped`` builds them.
_WRAPPER_VARIANTS = ("tool_name-parameters", "tool_call-input-string", "input-object")

#: One verbatim shape per rejection class in the arm's own histogram, and what
#: this build now does with it. The counts are the arm's, so this table states
#: the refusals the change exists to recover rather than a claim about them:
#: incomplete-json 34, envelope-shape 23, batch-shape 20, observation-binding 15,
#: leading-delimiter 5, other 7.
_MEASURED_REJECTION_CLASSES = [
    ("incomplete-json", "trailing-text", True),
    ("incomplete-json", "truncated-envelope", False),
    ("envelope-shape", "missing-version", True),
    ("envelope-shape", "notes-inside-batch", True),
    ("batch-shape", "tool_name-parameters", True),
    ("batch-shape", "input-object", True),
    ("observation-binding", "stale-observation-binding", False),
    ("leading-delimiter", "leading-prose", True),
    ("other", "wrong-version", True),
    ("other", "extra-action-key", False),
]


@pytest.mark.parametrize(
    ("rejection_class", "change", "accepted"),
    _MEASURED_REJECTION_CLASSES,
    ids=[f"{row[0]}-{row[1]}" for row in _MEASURED_REJECTION_CLASSES],
)
def test_each_measured_rejection_class_after_the_contract_change(
    rejection_class: str, change: str, accepted: bool
) -> None:
    """The histogram the change was scoped against, one shape per class.

    A reply is judged on its ACTIONS. Every class whose payload was already a
    complete, executable decision now decodes -- the missing or wrong or nested
    ``reply_version``, the notes key below instead of above, the envelope inside
    a generic tool-call wrapper, a complete envelope followed by text, a version
    literal this harness does not serve, and a preamble in front of the object
    when the reply carries exactly one decision. One class still refuses for a
    reason about meaning -- ``observation-binding``, because a decision about a
    screen the environment has moved past is the one failure worse than losing
    the turn -- and the malformed and action-level defects are pinned as
    still-refused rows beside it, so a tolerance that quietly widened would fail
    here rather than in a paid run.
    """

    current = observation()
    raw = _full_envelope(current)
    if change in _WRAPPER_VARIANTS:
        payload = _wrapped(change, raw)
    elif accepted:
        payload = _changed(change, raw, current)
    else:
        payload = _refused(change, raw, current)

    if accepted:
        assert parse_decision(payload, current, route=ROUTE).action_batch.actions
    else:
        with pytest.raises(DecisionParseError):
            parse_decision(payload, current, route=ROUTE)


# ---------------------------------------------------------------------------
# An action's own framing: a field belonging to a SIBLING action kind
# ---------------------------------------------------------------------------
#
# The measured defect these pin: a model that puts ``frame_id`` on a ``wait``, or
# ``duration_ms`` on a ``click``, states a complete decision -- the ``kind`` tag
# is required and explicit -- and the whole batch used to be refused for it.
# 46 of the 145 decision-rejections counted over the arm-0625 episodes were
# exactly this, each one a billed corrective round trip.


def test_a_sibling_field_is_dropped_and_the_rest_of_the_action_stands(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The action is judged on its DECLARED kind, and the drop is reported.

    Tolerated is not the same as silent: the log line is the only record a
    campaign can count the tolerance against, so the field is named
    ``kind.field`` -- which kind received it and which kind it belongs to -- out
    of the vocabulary rather than out of model text.
    """

    current = observation()
    payload = json.dumps(
        {
            "actions": [
                {
                    "kind": "wait",
                    "observation_id": current.observation_id,
                    "duration_ms": 800,
                    "frame_id": "screen",
                }
            ]
        }
    )

    with caplog.at_level(logging.WARNING):
        decision = parse_decision(payload, current, route=ROUTE)

    (action,) = decision.action_batch.actions
    assert action.kind == "wait"
    assert action.duration_ms == 800
    assert "wait.frame_id" in caplog.text


def test_a_field_no_kind_declares_is_still_refused() -> None:
    """The drop is bounded by the VOCABULARY, not by "unknown key, carry on".

    A name no action kind declares is not readable as anything, so dropping it
    would guess at what the model meant -- and a near-miss of a real field is
    exactly the mistake that must stay loud.
    """

    current = observation()
    payload = json.dumps(
        {
            "actions": [
                {
                    "kind": "wait",
                    "observation_id": current.observation_id,
                    "duration_ms": 800,
                    "duration": 900,
                }
            ]
        }
    )

    with pytest.raises(DecisionParseError) as info:
        parse_decision(payload, current, route=ROUTE)

    assert classify_rejection(str(info.value)) == "extra-action-key"


def test_a_decision_behind_leading_junk_keeps_its_own_validation_class() -> None:
    """The two tolerances compose without widening what counts as a decision.

    A preamble in front of a batch whose actions name another observation is not
    rescued by either one: the object is located (framing), and then refused for
    the binding it states (meaning), with the class that defect earns.
    """

    current = observation()
    stale = _full_envelope(observation(1), "older turn")

    with pytest.raises(DecisionParseError) as info:
        parse_decision("commentary " + stale, current, route=ROUTE)

    assert classify_rejection(str(info.value)) == "observation-binding"
