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

import pytest
from pydantic.fields import FieldInfo

from local_operator.evaluation.action_surface import LEGACY_ACTION_SURFACE
from local_operator.evaluation.protocol import KeyAction
from local_operator.evaluation.runner.provider_client import (
    REJECTION_CLASS_UNKNOWN,
    _action_schema_lines,
    classify_rejection,
    rejection_hint,
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

    assert '"reply_version": "1.0"' in hint
    assert '"action_batch"' in hint
    assert "input_value=" not in hint


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
    """

    artifacts = _sealed_rejection_artifacts()
    if len(artifacts) < _MIN_CORPUS_ARTIFACTS:
        pytest.skip(f"corpus has rotated: {len(artifacts)} rejection artifacts left")

    histogram: dict[str, int] = {}
    for artifact, episode in artifacts:
        derived = classify_rejection(_diagnostic_of(artifact))
        key = _class_of(artifact)
        histogram[key] = histogram.get(key, 0) + 1
        assert derived != REJECTION_CLASS_UNKNOWN, episode
        assert derived == key, episode

    with capsys.disabled():
        print(f"\nsealed rejection classes ({len(artifacts)} artifacts):")
        for key, count in sorted(histogram.items(), key=lambda item: -item[1]):
            print(f"  {count:4d}  {key}")

    ranked = {
        "malformed-json",
        "unsupported-reply-version",
        "envelope-shape",
        "extra-action-key",
        "unknown-key",
        "out-of-frame-coordinate",
    }
    assert ranked.issubset(histogram), sorted(histogram)


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

    assert _class_of(old) == "malformed-json"
    assert _class_of(new) == "malformed-json"
    # The diagnostic section excludes everything the artifact added around it,
    # so a reader that WANTS the prose gets the prose and not the header.
    for artifact in (old, new):
        assert "--- rejected reply ---" not in _diagnostic_of(artifact)
        assert not _diagnostic_of(artifact).startswith("class: ")
    assert "stream:" not in _diagnostic_of(new)
