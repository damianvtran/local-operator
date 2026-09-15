"""Refusal CLASSES: the taxonomy, its drift pin, and the sealed corpus.

The class key is what makes a rejection countable. Before it existed the only
thing a bundle recorded was the validator's prose, and deriving a class from
prose after the fact is guesswork -- the MiniMax campaign's rejections were
mis-read once already for exactly this reason. So the taxonomy is pinned two
ways: against the real parsers (in ``test_provider_client``, per class, through
the client) and against the sealed artifacts a paid run actually produced
(here), which is the corpus a future reader will want to reproduce.

The offline replay needs no credentials and spends nothing: it only reads
bundles that are already on disk, and it SKIPS when they are not.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError
from pydantic.fields import FieldInfo

from local_operator.evaluation.action_surface import (
    LEGACY_ACTION_SURFACE,
    ActionAdmissionError,
)
from local_operator.evaluation.protocol import ActionBatch, KeyAction
from local_operator.evaluation.runner.provider_client import (
    _MISPLACED_REPLY_VERSION_MARKER,
    REJECTION_CLASS_UNKNOWN,
    DecisionParseError,
    _action_schema_lines,
    _decode_leading_json,
    _first_field_path,
    classify_admission_error,
    classify_rejection,
    classify_validation_error,
    rejection_hint,
    strip_reasoning_boundary_markers,
    validation_diagnostic,
)
from local_operator.evaluation.runner.public_reply import (
    decode_public_reply,
    normalise_public_reply,
)
from tests.unit.evaluation.runner.test_provider_client import observation

#: Where a campaign's sealed bundles live on the machine that ran it. Overridable
#: because the corpus is a paid run's output, not a repository fixture: the test
#: must work on any machine that has one, and skip on every machine that does
#: not (CI included) rather than pretend the classes are unverified.
CORPUS_ENV = "LOCAL_OPERATOR_REJECTION_CORPUS"
DEFAULT_CORPUS = Path.home() / "worktrees" / "osworld" / "runs"
CORPUS_GLOB = "batch-minimax-m3-*/task_*/evidence/ep-*"

#: A runner bound, so a corpus that has rotated away to a handful of bundles
#: skips instead of asserting a distribution it can no longer see.
_MIN_CORPUS_ARTIFACTS = 20


def _corpus_root() -> Path:
    return Path(os.environ.get(CORPUS_ENV) or DEFAULT_CORPUS)


def _sealed_rejection_artifacts() -> list[tuple[str, str]]:
    """``(diagnostic, artifact_text)`` for every sealed ``decision-rejected``.

    Read from the EVENT, not from the artifact directory listing: the artifact
    a rejection points at is identified by digest in the event, and an orphan
    file in ``artifacts/`` is not a rejection.
    """

    found: list[tuple[str, str]] = []
    for episode in sorted(_corpus_root().glob(CORPUS_GLOB)):
        events = episode / "events.jsonl"
        if not events.exists():
            continue
        for line in events.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if record.get("kind") != "error":
                continue
            payload = record.get("payload") or {}
            if payload.get("diagnostic_code") != "decision-rejected":
                continue
            digest = (payload.get("detail_artifact") or {}).get("sha256")
            artifact = episode / "artifacts" / str(digest)
            if digest and artifact.exists():
                found.append((artifact.read_text(encoding="utf-8", errors="replace"), str(episode)))
    return found


def _diagnostic_of(artifact: str) -> str:
    """The rejection's diagnostic section: everything up to the reply marker.

    Mirrors ``episode._rejection_detail``'s layout without importing the runner,
    so this reads both the old artifacts (diagnostic then ``--- rejected
    reply ---``) and the new ones (diagnostic, ``class:``, ``stream:``, reply).
    """

    head = artifact.split("\n\n--- rejected reply ---", 1)[0]
    return "\n".join(
        line
        for line in head.splitlines()
        if not line.startswith("class: ") and not line.startswith("stream: ")
    )


def _class_of(artifact: str) -> str:
    """The class of a sealed rejection: recorded if it says so, derived if not.

    Two generations of artifact, one reader. A bundle written since the class
    key exists STATES its class (``class:``) and that is authoritative -- it is
    what the run itself bucketed the refusal into. A bundle written before it
    must be classified from its diagnostic text, which is the whole reason the
    classifier reads text and not an exception object: the corpus that motivated
    this work can only be measured that way.
    """

    for line in artifact.splitlines():
        if line.startswith("class: "):
            return line[len("class: ") :].strip()
    return classify_rejection(_diagnostic_of(artifact))


def test_the_hint_is_derived_from_the_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    """A protocol change moves the hint, exactly as it moves the prompt.

    ``_action_schema_lines`` exists so the system prompt cannot advertise a
    shape the validator refuses. The hint has the same job at the other end of
    the episode -- it is the LAST thing the model reads before it retries -- so
    it is derived from the same models and this asserts the two drift together.
    Mutating ``KeyAction.keys`` from an array of names to a single string is the
    protocol change that would otherwise leave the correction instructing the
    model to send a shape the parser rejects.
    """

    current = observation()
    before = rejection_hint(
        "keys-not-array", reason="", observation=current, surface=LEGACY_ACTION_SURFACE
    )
    schema_before = _action_schema_lines(LEGACY_ACTION_SURFACE)

    assert "an array of key names" in before
    monkeypatch.setitem(KeyAction.model_fields, "keys", FieldInfo(annotation=str))

    after = rejection_hint(
        "keys-not-array", reason="", observation=current, surface=LEGACY_ACTION_SURFACE
    )
    schema_after = _action_schema_lines(LEGACY_ACTION_SURFACE)

    assert "an array of key names" not in after
    assert "a string" in after
    assert after != before
    # The mirror: the prompt's own rendering of the same field moved with it.
    assert schema_after != schema_before
    assert '"keys": [str, ...]' in "\n".join(schema_before)
    assert '"keys": str' in "\n".join(schema_after)


_PRESERVED_MESSAGES = [
    # The envelope decoder's own diagnostic: it names the keys carried and
    # omitted, and recovered 9/10 against 4/10 for the bare rule.
    "model reply used the reserved envelope but carried 'action_batch', "
    "'reply_version'; omitted 'public_observations'.",
    # The batch-level rule with the keys that landed in the batch NAMED. Still
    # the harness's own sentence and still preserved: the naming is what makes
    # it actionable, and this row pins that a later edit cannot quietly drop
    # the names and leave the bare rule the model cannot act on.
    "model reply action_batch requires exactly an actions array; it carried 1 "
    "unexpected key(s): 'observation_id'",
    # The batch-level rules, also the harness's own sentences.
    "decision must carry a non-empty actions array",
    "decision carries a second action batch for the same observation; send one batch",
    # The adapter's negotiated restriction, phrased WITH the alternative it can
    # carry -- a bare refusal would leave the model shortening text forever.
    "decision does not match this observation: type supports only ASCII on this "
    "adapter; use paste_text with an explicit chord",
    # The silent-reply diagnostic, which names the channel and the shape.
    "reply carried no tool call and no text: the model ended its turn as 'stop' "
    "without emitting a decision on either channel. Reply with the action batch "
    'itself, as a single JSON object with a non-empty "actions" array.',
]


@pytest.mark.parametrize("reason", _PRESERVED_MESSAGES, ids=[m[:24] for m in _PRESERVED_MESSAGES])
def test_a_measured_message_is_preserved_rather_than_paraphrased(reason: str) -> None:
    """The classes whose diagnostic was ALREADY measured keep it verbatim.

    These are the harness's OWN sentences, written to be read by the model, and
    each already does the job a hint exists to do -- naming the keys carried and
    omitted, the rule broken, the limit and its alternative, or the shape the
    reply must take. Re-deriving them here would replace measured text with
    unmeasured text, so the hint returns them unchanged. Every one is asserted
    to be Pydantic-free already, which is what makes preserving it safe; a class
    whose diagnostic comes from a Pydantic rendering is never preserved.
    """

    key = classify_rejection(reason)

    assert key != REJECTION_CLASS_UNKNOWN, reason
    assert "input_value=" not in reason and "errors.pydantic.dev" not in reason
    assert (
        rejection_hint(key, reason=reason, observation=observation(), surface=LEGACY_ACTION_SURFACE)
        == reason
    )


#: The measured half of the envelope diagnostic, byte for byte. Measured on
#: ``minimax/minimax-m3`` at 9/10 recovered against 4/10 for the bare rule, so it
#: is never paraphrased -- the misplaced-version sentence reuses it verbatim as
#: its prefix. Pinned literally because a reword here would silently discard the
#: only measured correction the class ever had.
_MEASURED_CARRIED_OMITTED = (
    "model reply used the reserved envelope but carried 'action_batch', "
    "'public_observations'; omitted 'reply_version'"
)


#: The envelope diagnostics this contract no longer EMITS, as the sealed corpora
#: carry them, with the class each must keep. Pinned literally because
#: ``classify_rejection`` reads text and not an exception object: a reword moves
#: every artifact carrying one into ``unknown``, and those corpora are the only
#: measurement these classes will ever have. ``env-version-misplaced`` is the
#: discriminator INSIDE a sentence that otherwise says both things at once, which
#: is why the classifier checks its marker before the envelope sentence.
_RETIRED_REASONS = [
    (_MEASURED_CARRIED_OMITTED, "envelope-shape"),
    (f"{_MEASURED_CARRIED_OMITTED}. {_MISPLACED_REPLY_VERSION_MARKER}", "env-version-misplaced"),
    ("model reply action_batch requires exactly an actions array", "envelope-shape"),
    (
        "model reply action_batch requires exactly an actions array; it carried 1 "
        "unexpected key(s): 'observation_id'",
        "envelope-shape",
    ),
    ("unsupported model reply version", "unsupported-reply-version"),
]


def test_the_retired_envelope_diagnostics_still_classify() -> None:
    """A class that survives only in old bundles is still a countable class.

    Four sentences are retired by the single-shape contract: the carried/omitted
    envelope diagnostic, the misplaced-version sentence it shares a prefix with,
    the batch rule with the keys that landed in it named, and the
    ``unsupported model reply version`` refusal. Every one of them is on disk in
    a bundle a paid campaign produced, so ``classify_rejection`` must keep
    reading them -- and every hint must still state an accepted shape, because a
    hint describing a layout the decoder no longer requires would send a model
    chasing a key this harness has stopped asking for.
    """

    for reason, expected in _RETIRED_REASONS:
        key = classify_rejection(reason)
        assert key == expected, reason
        hint = rejection_hint(
            key, reason=reason, observation=observation(), surface=LEGACY_ACTION_SURFACE
        )
        assert "input_value=" not in hint and "[type=" not in hint
        # A preserved hint IS the harness's own sentence, which the model already
        # has to act on (that is what preservation means); every other class
        # appends the accepted shape, so it must name the actions array.
        assert hint == reason or '"actions"' in hint, reason


#: One valid action for the test observation, as the wire carries it: the bodies
#: below differ ONLY in where the envelope keys sit, so the action cannot vary
#: with them.
_WAIT_ACTION: dict[str, Any] = {
    "kind": "wait",
    "observation_id": observation().observation_id,
    "duration_ms": 10,
}
_ACTIONS = [_WAIT_ACTION]

#: The four shapes the misplaced-version class was split out for, as the models
#: wrote them -- now carrying an action, so the placement of the keys is the only
#: variable under test. The first is the shape both sealed DeepSeek episodes
#: repeated; the last is the minimax shape that put its NOTES inside the batch
#: (``batch-minimax-m3-canary5/task_087`` and ``task_106``).
_OMITTED_VERSION_BODY = json.dumps(
    {"action_batch": {"actions": _ACTIONS}, "public_observations": "a note"}
)
_NESTED_VERSION_BODY = json.dumps(
    {
        "action_batch": {"actions": _ACTIONS, "reply_version": "1.0"},
        "public_observations": "a note",
    }
)
_DUPLICATED_VERSION_BODY = json.dumps(
    {
        "reply_version": "1.0",
        "action_batch": {"actions": _ACTIONS, "reply_version": "1.0"},
        "public_observations": "a note",
    }
)
_NESTED_NOTES_BODY = json.dumps(
    {"action_batch": {"actions": _ACTIONS, "public_observations": "a note"}}
)


def test_where_a_key_sits_no_longer_decides_whether_a_reply_decodes() -> None:
    """The same bodies as before, with the one thing that matters: the actions.

    ``reply_version`` absent, at the top level, nested one level down, or written
    at both levels; the notes key above the batch or inside it; a key the
    contract has no use for beside them. None of that changes the decision the
    model made, and none of it may cost the turn -- which is exactly the price
    the retired diagnostics above used to charge.
    """

    bodies = {
        "omitted-version": _OMITTED_VERSION_BODY,
        "nested-version": _NESTED_VERSION_BODY,
        "duplicated-version": _DUPLICATED_VERSION_BODY,
        "nested-notes": _NESTED_NOTES_BODY,
        "other-version": json.dumps(
            {
                "reply_version": "2.0",
                "action_batch": {"actions": _ACTIONS},
                "public_observations": "a note",
            }
        ),
        "bare-actions": json.dumps({"actions": _ACTIONS, "public_observations": "a note"}),
        "extra-key": json.dumps(
            {
                "action_batch": {"actions": _ACTIONS},
                "public_observations": "a note",
                "notes": "x",
            }
        ),
    }

    for label, body in bodies.items():
        decoded = decode_public_reply(body)
        assert decoded["actions"] == _ACTIONS, label
        # The notes survive wherever the model put them: they are the one part
        # of a reply the next turn's context carries verbatim.
        assert decoded["public_observations"] == "a note", label


def test_an_unrecognised_message_is_still_recordable_and_still_answered() -> None:
    """A refusal this build has never seen must not lose its evidence.

    Every class key is a string, including the fallback, and the fallback hint
    still states the accepted envelope: a taxonomy that raised on the unexpected
    would turn a diagnosable refusal into an unsealed episode, which is strictly
    worse than the prose this replaced.
    """

    reason = "the provider said something no template of ours produces"
    assert classify_rejection(reason) == REJECTION_CLASS_UNKNOWN

    hint = rejection_hint(
        REJECTION_CLASS_UNKNOWN,
        reason=reason,
        observation=observation(),
        surface=LEGACY_ACTION_SURFACE,
    )

    assert '"actions"' in hint
    assert "input_value=" not in hint


@pytest.mark.skipif(
    not _corpus_root().exists(),
    reason=f"no sealed rejection corpus at {_corpus_root()} (set {CORPUS_ENV})",
)
def _records_its_class(artifact: str) -> bool:
    """Whether the artifact itself states the class it was bucketed into.

    Two generations of bundle, and the difference decides what can be CHECKED.
    A bundle written since the class key exists carries it (``class:``) and also
    replaced the decoder's parse error with the HINT the model was shown -- so
    its diagnostic can no longer be classified, and the recorded key is the only
    honest source. A bundle written before it carries the raw decoder text, which
    is what makes ``classify_rejection`` a text classifier in the first place.
    """

    return any(line.startswith("class: ") for line in artifact.splitlines())


@pytest.mark.skipif(
    not _corpus_root().exists(),
    reason=f"no sealed rejection corpus at {_corpus_root()} (set {CORPUS_ENV})",
)
def test_every_sealed_rejection_artifact_classifies(capsys: pytest.CaptureFixture[str]) -> None:
    """The classes, over the artifacts a paid campaign actually produced.

    Two claims, and the second is the one that matters for the instrument: not
    one artifact falls through to the fallback key (so the taxonomy covers the
    traffic), and the classes the corpus ranks are all present (so the keys are
    not merely reachable in theory). The histogram is printed because it is the
    measurement -- the reason the class key exists at all.

    The two generations are checked differently on purpose. Where the bundle
    records its class, the recorded key must be a real class. Where it does not,
    the derivation from the decoder's own text must agree with what the reader
    returns -- that equality is the property that keeps a report tool's reading
    of an old bundle identical to the run's own bucketing.
    """

    artifacts = _sealed_rejection_artifacts()
    if len(artifacts) < _MIN_CORPUS_ARTIFACTS:
        pytest.skip(f"corpus has rotated: {len(artifacts)} rejection artifacts left")

    histogram: dict[str, int] = {}
    for artifact, episode in artifacts:
        key = _class_of(artifact)
        histogram[key] = histogram.get(key, 0) + 1
        assert key != REJECTION_CLASS_UNKNOWN, episode
        if not _records_its_class(artifact):
            assert classify_rejection(_diagnostic_of(artifact)) == key, episode

    with capsys.disabled():
        print(f"\nsealed rejection classes ({len(artifacts)} artifacts):")
        for key, count in sorted(histogram.items(), key=lambda item: -item[1]):
            print(f"  {count:4d}  {key}")

    ranked = {
        "unsupported-reply-version",
        "envelope-shape",
        "extra-action-key",
        "unknown-key",
        "out-of-frame-coordinate",
    }
    assert ranked.issubset(histogram), sorted(histogram)
    # The split halves, asserted apart from ``malformed-json``: the old class
    # still appears (bundles recorded before the split), and the two halves that
    # replaced it must both be ranked by the corpus, or the split has not
    # actually separated the traffic it was built for.
    assert {"leading-delimiter", "incomplete-json"} & set(histogram), sorted(histogram)


def test_a_sealed_artifact_reads_the_way_report_tools_will_read_it() -> None:
    """Both generations of artifact, through the reader a script would use.

    An artifact written before the class key existed is classified from its
    diagnostic text; one written after it states its own class and the recorded
    key wins. Both must work, or the reader gets a different answer depending on
    when the corpus was paid for.
    """

    old = (
        "Your previous reply was rejected: decision is not valid JSON: Expecting value: "
        "line 1 column 1 (char 0)\n"
        "Nothing was executed. Reply again for this same observation.\n"
        "\n--- rejected reply ---\n"
        "(model reply rejected; no public observations accepted)"
    )
    new = (
        "Your previous reply was rejected: the reply was not one complete JSON object.\n"
        "Nothing was executed. Reply again.\n"
        "class: malformed-json\n"
        "stream: content_deltas=1 reasoning_deltas=0 tool_call_deltas=0 stop=stop\n"
        "\n--- rejected reply ---\n"
        '{"actions": ['
    )

    assert _class_of(old) == "leading-delimiter"
    # A bundle that RECORDS a class this build no longer derives still reads as
    # what it says: the split renamed the halves of ``malformed-json``, and a
    # reader that reclassified a sealed bundle by this build's names would
    # silently rewrite history for every run made before the split.
    assert _class_of(new) == "malformed-json"
    # The diagnostic section excludes everything the artifact added around it,
    # so a reader that WANTS the prose gets the prose and not the header.
    for artifact in (old, new):
        assert "--- rejected reply ---" not in _diagnostic_of(artifact)
        assert not _diagnostic_of(artifact).startswith("class: ")
    assert "stream:" not in _diagnostic_of(new)


# ---------------------------------------------------------------------------
# The offline replay: the reply NORMALISER, over the corpus it was built for
# ---------------------------------------------------------------------------
#
# The classes above measure what a refusal was CALLED. This measures what the
# normaliser DID to the reply, which is the change itself, and it is the
# acceptance test for it: no credentials, no model spend, just the replies a
# paid campaign already sealed.
#
# What it can prove and what it cannot: the replay runs the two decode
# boundaries the reply is judged at (``_decode_leading_json``, then the reserved
# envelope). Observation binding -- whether the batch names the screen in front
# of the model -- CANNOT be replayed, because a sealed rejection does not publish
# the observation it answered, so a reply that clears both boundaries is reported
# as ``accepted-shape`` rather than claimed as a decision. That is the boundary
# the change is about anyway: a reply the decoder could not even READ never
# reached the observation check.

#: What the replay calls a reply that clears both decode boundaries. Deliberately
#: not a rejection class: it says "nothing in the reply's own bytes refuses it",
#: which is the claim the replay is entitled to make.
_ACCEPTED_SHAPE = "accepted-shape"

#: The placeholder a bundle writes instead of a reply it may not publish. A JSON
#: reply cannot start with a bracket, so this is an exact test rather than a
#: heuristic -- and the 271 artifacts that carry it are why the class taxonomy
#: was built to read the diagnostic instead of the reply.
_PLACEHOLDER_PREFIX = "(model reply rejected"


def _published_reply(artifact: str) -> str | None:
    """The reply section of the artifact, or ``None`` when it kept none."""

    if "\n\n--- rejected reply ---\n" not in artifact:
        return None
    reply = artifact.split("\n\n--- rejected reply ---\n", 1)[1]
    if reply.startswith(_PLACEHOLDER_PREFIX):
        return None
    return reply


#: Manifests are read once per episode: the corpus has dozens of episodes and
#: hundreds of rejection artifacts, and re-reading one manifest per artifact
#: would make the replay's cost a function of how badly a run went.
_MARKER_CACHE: dict[str, tuple[str, ...]] = {}


def _declared_markers(episode: str) -> tuple[str, ...]:
    """The boundary markers the harness would have declared for that episode.

    Read from the bundle's own manifest rather than hardcoded, so the replay
    tests the DECLARATION that production applies to the route that produced the
    corpus -- a table row that stopped matching the model id OpenRouter serves
    would show up here as a corpus that recovered nothing.
    """

    if episode in _MARKER_CACHE:
        return _MARKER_CACHE[episode]
    markers: tuple[str, ...] = ()
    manifest = Path(episode) / "manifest.json"
    if manifest.exists():
        try:
            route = json.loads(manifest.read_text(encoding="utf-8")).get("requested_route") or {}
        except ValueError:
            route = {}
        from local_operator.model.configure import _reasoning_boundary_markers

        markers = _reasoning_boundary_markers(str(route.get("model_id") or ""))
    _MARKER_CACHE[episode] = markers
    return markers


def _replay_verdict(reply: str) -> str:
    """The verdict a reply's own bytes earn it, through the runtime's decoder.

    Mirrors ``parse_decision``'s FRAMING layer and nothing else: the corpus
    replay is about what the decoder can read, so adapter-level validation is
    out of scope here exactly as it is inside ``_decode_leading_json``. The two
    calls are the real ones -- the leading-JSON tolerance and the normaliser --
    so the class printed for an artifact is the class the run would have given
    it, rather than a second opinion about the same bytes.
    """

    payload = reply.strip()
    try:
        decoded, _trailing = _decode_leading_json(payload)
        if not isinstance(decoded, dict):
            return "batch-shape"
        normalise_public_reply(decoded)
    except (DecisionParseError, ValueError) as error:
        # The same reason string ``parse_decision`` would hand the classifier,
        # so the replayed class is derived by the SAME function the run used.
        return classify_rejection(str(error))
    return _ACCEPTED_SHAPE


@pytest.mark.skipif(
    not _corpus_root().exists(),
    reason=f"no sealed rejection corpus at {_corpus_root()} (set {CORPUS_ENV})",
)
def test_the_sealed_corpus_replays_through_the_reply_normaliser(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Replay every published sealed reply, and account for every verdict.

    Four claims, and the corpus is the only place they can be made at this
    scale -- 40 published replies from a paid campaign, of which 17 arrived
    carrying a provider reasoning boundary token:

    1. A reply with NO declared marker at its head is left byte-identical and
       keeps its verdict. That is "no previously-passing reply changes verdict",
       asserted over every published reply rather than over a sample.
    2. A reply that IS stripped was, before the strip, unreadable at offset 0.
       A strip may only ever rescue a reply the decoder could not START -- it can
       never be a repair of a reply that was merely wrong.
    3. The set of rejection classes the corpus produces afterwards is a subset of
       the set it produced before: the split renames two halves of one class and
       the strip removes one cause, so a class that is NEW after the change is a
       class the change invented.
    4. At least one reply is actually recovered. Without it the corpus has
       stopped exercising the path this exists for, which is a measurement
       failure rather than a pass.

    The counts themselves are printed rather than pinned: the corpus is a paid
    run's output that grows as canary batches land, and a hard-coded 15 would go
    red on the next run and read as a regression. Measured 2026-09-12: 329
    artifacts, 311 with a reply section, 40 published replies, 17 tagged, 15
    recovered -- and the 2 tagged ones that did not recover are the ones whose
    remainder is still not JSON, which arrived as ``incomplete-json``.
    """

    published = [
        (artifact, episode)
        for artifact, episode in _sealed_rejection_artifacts()
        if _published_reply(artifact) is not None
    ]
    if len(published) < _MIN_CORPUS_ARTIFACTS:
        pytest.skip(f"corpus has rotated: {len(published)} published replies left")

    before: dict[str, int] = {}
    after: dict[str, int] = {}
    transitions: dict[str, int] = {}
    for artifact, episode in published:
        reply = _published_reply(artifact) or ""
        verdict_before = _replay_verdict(reply)
        stripped, removed = strip_reasoning_boundary_markers(reply, _declared_markers(episode))
        verdict_after = _replay_verdict(stripped)
        before[verdict_before] = before.get(verdict_before, 0) + 1
        after[verdict_after] = after.get(verdict_after, 0) + 1
        if not removed:
            assert stripped == reply, episode
            assert verdict_after == verdict_before, episode
            continue
        assert verdict_before == "leading-delimiter", episode
        moved = f"{verdict_before} -> {verdict_after}"
        transitions[moved] = transitions.get(moved, 0) + 1

    assert set(after) - {_ACCEPTED_SHAPE} <= set(before), sorted(after)
    recovered = transitions.get(f"leading-delimiter -> {_ACCEPTED_SHAPE}", 0)
    assert recovered > 0, sorted(transitions)

    with capsys.disabled():
        print(f"\nsealed reply replay ({len(published)} published replies):")
        for label, histogram in (("before", before), ("after", after)):
            print(f"  {label}: " + ", ".join(f"{k}={v}" for k, v in sorted(histogram.items())))
        for label, count in sorted(transitions.items()):
            print(f"  {count:4d}  {label}")
        print(f"  {recovered:4d}  recovered to the accepted shape")


# --- The structured half of the taxonomy -----------------------------------
#
# ``classify_rejection`` reads PROSE, and that is the right design for the
# envelope path: a sealed bundle has only the message the validator wrote, and a
# class that could only be derived by re-running the validator would measure a
# different population than the run it is bucketing. It is the wrong design for a
# payload the model authored, because ``str(ValidationError)`` renders
# ``input_value=<head>…<tail>`` and ``validate_for`` interpolates
# ``action.frame_id`` into its own sentence -- so the payload's own bytes become
# an input to an ordered substring test, and the model chooses its class (and,
# for a preserved class, receives that rendering as its correction).
#
# The two functions below are the other half of the SAME vocabulary, read from
# structured data: Pydantic's ``type``/``loc`` entries, which carry no value, and
# the exception's own type. The class keys they return are the ones above, so a
# bundle's class table stays comparable whichever path produced it.


def _validation_error(payload: dict[str, Any]) -> ValidationError:
    """A real ``ValidationError``, from the protocol's own model, made by hand.

    Built from ``ActionBatch`` rather than from a fixture so the error TYPES are
    the ones the real path produces -- a hand-rolled error object would let this
    test pin a table the protocol no longer emits.
    """

    with pytest.raises(ValidationError) as raised:
        ActionBatch.model_validate(
            {
                "protocol_version": "1.0",
                "kind": "action_batch",
                "task_id": "task-1",
                "episode_id": "episode-1",
                "observation_id": "obs-1",
                **payload,
            }
        )
    return raised.value


def test_the_value_free_diagnostic_keeps_what_its_readers_read() -> None:
    """The rendering the tool passes instead of ``str(error)``.

    Two readers in this module depend on its shape -- ``_first_field_path`` on
    the location line, ``_VALUE_ERROR_RULE`` on the rule sentence -- and the
    envelope path's own renderer is untouched, so this rendering exists only for
    a payload the model wrote. What it drops is exactly the refused value.
    """

    # ``observation_id`` is stated so the failure is the KEY NAME's: the real
    # tool injects it, and a raw protocol model still demands it.
    error = _validation_error(
        {"actions": [{"kind": "key", "observation_id": "obs-1", "keys": ["NOSUCH"]}]}
    )
    diagnostic = validation_diagnostic(error)

    assert "actions.0.key.keys" in diagnostic
    assert "unknown key: 'NOSUCH'" in diagnostic
    assert "[type=value_error]" in diagnostic
    for marker in ("input_value=", "input_type=", "errors.pydantic.dev"):
        assert marker not in diagnostic
        assert marker in str(error)

    # The readers really do read it: the location line is found, and the class
    # comes from the rule's own prefix rather than from the rendering.
    assert _first_field_path(diagnostic) == ("actions", "0", "key", "keys")
    assert classify_validation_error(error) == "unknown-key"


def test_the_structured_classifier_reads_pydantic_types_and_never_values() -> None:
    """One entry per structured rule, and the same failure with a marker VALUE.

    Each pair differs only in the bytes the model wrote in one position, and each
    marker value is a phrase that names a class above -- so this is the taxonomy's
    own attack surface, at the level of the classifier rather than the tool: the
    class must be a property of the failure, never of the value that failed.
    """

    cases = [
        # (control payload, the same payload with a class marker as its value, class)
        (
            {"actions": [{"kind": "wait", "duration_ms": "123abc"}]},
            {"actions": [{"kind": "wait", "duration_ms": "second action batch"}]},
            "field-invalid",
        ),
        (
            {"actions": [{"kind": "wait", "duration_ms": 5, "noun": "note"}]},
            {"actions": [{"kind": "wait", "duration_ms": 5, "noun": "second action batch"}]},
            "extra-action-key",
        ),
        (
            {"actions": [{"kind": "drag", "x": 1}]},
            {"actions": [{"kind": "drag", "x": "is limited to"}]},
            "unknown-action-kind",
        ),
        (
            {"actions": [{"kind": "key", "keys": {"item": ["CTRL"]}}]},
            {"actions": [{"kind": "key", "keys": {"item": ["reserved envelope"]}}]},
            "keys-not-array",
        ),
        (
            {"actions": [{"kind": "key", "keys": ["zzzz"]}]},
            {"actions": [{"kind": "key", "keys": ["outside model-visible frame"]}]},
            "unknown-key",
        ),
        (
            {"actions": []},
            {"actions": []},
            "field-invalid",
        ),
    ]

    for control_payload, attacked_payload, expected in cases:
        control = classify_validation_error(_validation_error(control_payload))
        assert control == expected, control_payload
        assert (
            classify_validation_error(_validation_error(attacked_payload)) == control
        ), attacked_payload


def test_an_admission_refusal_is_classified_by_type_and_value_free_prefix() -> None:
    """The surface's class IS its exception type; the protocol's is its prefix.

    ``validate_for`` interpolates ``action.frame_id`` into its own sentence, so
    the two cases at the end are the attack: a frame id that names another class
    must not be believed.
    """

    assert (
        classify_admission_error(ActionAdmissionError("second action batch"))
        == "adapter-capability"
    )
    for reason, expected in (
        (
            "action coordinate 500,10 is outside model-visible frame 100x100",
            "out-of-frame-coordinate",
        ),
        ("action references unknown frame_id 'screen-2'", "unknown-frame-id"),
        (
            "action batch does not bind to the current task, episode, and observation",
            "observation-binding",
        ),
        # The value is in the message, after the anchor, and cannot name the class.
        (
            "action references unknown frame_id 'outside model-visible frame'",
            "unknown-frame-id",
        ),
        (
            "action batch does not bind to the current task, episode, and observation "
            "-- unknown key: 'X'",
            "observation-binding",
        ),
    ):
        assert classify_admission_error(ValueError(reason)) == expected, reason
    # A sentence from a later protocol is RECORDED as unrecognised rather than
    # guessed at from its text: no raise site exists that this build cannot name.
    assert classify_admission_error(ValueError("a rule added later")) == REJECTION_CLASS_UNKNOWN
