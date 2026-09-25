"""A bundle sealed by an earlier revision must still verify, and still recover.

Why this module exists, and why nothing else could have caught the change that
made it necessary: every other verification test writes and verifies at ONE
revision, so a field added to a payload model is invisible to all of them --
both sides canonicalize the payload the same way. An ``event_id`` is a digest
over its payload as the CURRENT model reads it, so a defaulted field is NOT
additive for bytes that already exist. Measured on this repository's own
history: two fields added to ``ModelResponsePayload`` made 41 of the campaign's
49 sealed bundles fail ``verify_bundle`` with exactly one ``event_hash_mismatch``
per ``model_response`` event (4,379 between them), and made
``EvidenceWriter.open_for_abandon`` refuse the stranded ones outright -- the
entire evidence record of every episode already run, broken by a change whose
own docstring claimed the opposite.

The fixture is a REAL sealed bundle: the manifest, state marker, journal and
artifacts exactly as the pre-change revision wrote them, byte for byte, with
the pre-change event ids. It is small because the property under test is the
DIGEST rather than the episode -- a preflight, a commitment, a running
lifecycle, one request/response/usage triple, one observation, one batch and
its step -- and it deliberately stops mid-execution, which is the stranded
shape a crash leaves behind and the one recovery is for. Regenerating it with
the current code would prove nothing: the point is that these bytes were sealed
BEFORE the fields under test existed.

If a change to a hashed payload model is deliberate and the corpus is being
re-baselined on purpose, this is where that decision has to be announced: the
first test below fails, and the fixture is resealed at the new revision with
the break recorded in the commit message and the PR.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from local_operator.evaluation.evidence.models import (
    ModelResponsePayload,
    ReplyTolerancePayload,
)
from local_operator.evaluation.evidence.store import EvidenceWriter
from local_operator.evaluation.evidence.verify import verify_bundle
from local_operator.evaluation.receipts import RedactionSet

#: Sealed by ``origin/main`` (v0.62.31-era package, pre-dating the
#: ``ReplyTolerancePayload`` kind and the counts that prompted it).
FIXTURE = Path(__file__).parent / "pre_change_bundle"


def _copy(tmp_path: Path) -> Path:
    """A private copy of the fixture: the tests below lock and abandon a bundle."""

    root = tmp_path / "bundle"
    shutil.copytree(FIXTURE, root)
    return root


def test_a_bundle_sealed_before_the_reply_tolerance_counts_still_verifies(
    tmp_path: Path,
) -> None:
    """The discriminating shape: frozen bytes, verified at the current revision."""

    root = _copy(tmp_path)

    report = verify_bundle(root)

    assert [issue.code for issue in report.issues] == []
    assert report.valid
    # The subject of the test, asserted so the fixture cannot quietly stop
    # discriminating: this bundle must carry the event whose payload model grew,
    # and it must predate the kind that carries those counts today.
    # Selected by TYPE rather than by ``kind``: the event payload is a wide union,
    # so a kind comparison narrows nothing and reading the field off it would
    # neither type-check nor assert which model the test is talking about.
    responses = [
        event.payload for event in report.events if isinstance(event.payload, ModelResponsePayload)
    ]
    assert len(responses) == 1
    assert responses[0].request_id == "request-0"
    assert not any(isinstance(event.payload, ReplyTolerancePayload) for event in report.events)
    assert report.terminal_state == "open"


def test_a_stranded_bundle_sealed_before_the_counts_still_recovers(tmp_path: Path) -> None:
    """The consumer that refused: recovery verifies independently, then abandons.

    ``open_for_abandon`` re-verifies before it will touch a dead owner's bytes and
    refuses on any error-severity issue, so a re-baselined payload made a stranded
    pre-change bundle unrecoverable -- the rescue path for a crashed episode.
    """

    root = _copy(tmp_path)

    writer = EvidenceWriter.open_for_abandon(root, RedactionSet.from_resolved_values(()))
    try:
        assert writer.manifest.episode_id == "ep-frozen-fixture"
        record = writer.abandon("crash", "fixture_recovery")
        assert record.reason == "crash"
    finally:
        writer.close()

    recovered = verify_bundle(root)
    assert recovered.terminal_state == "abandoned"
