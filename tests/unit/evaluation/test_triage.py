"""The apparatus triage: an apparatus-attributable episode is excluded and
named from BOTH sides of the rate, never counted.

WHY REAL SEALED BUNDLES RATHER THAN A FAKE DIRECTORY. ``read_bundle`` is a
READER, and what a reader gets wrong is which bytes it opened: whether the bundle
verified at all, which artifact the score's digest names, and whether the
diagnostics reached the artifact in the first place. A hand-made directory would
exercise the classifier and nothing else. So the tests below seal REAL bundles
through the REAL ``EpisodeRunner`` -- the fake adapter is handed a score whose
details carry an evaluator log, exactly as the OSWorld adapter hands over its
own -- and then read them back the way a campaign does.

BOTH DIRECTIONS ARE TESTED, because the mis-grade is not a zero-maker: it is an
EMPTY READ, which manufactures a zero when the check expected content and would
manufacture a PASS if an empty read satisfied an empty expectation. So the rule
may not consult the score at all, and the regression is pinned both ways -- a
sealed PASS carrying the marker set is excluded, and a sealed episode whose
fallback recovered the target (the measured shape of three of the four real
episodes) is NOT, at either binary.

THE RULE TAKES THREE MARKERS, and the third is the one that carries the weight.
Two of them -- the getter failure and the open-tab fallback -- are emitted by the
same code path whatever tab the fallback lands on, so on their own they flag all
four measured episodes; the third, a timed-out selector, is the observable fact
that the graded page did not carry the element the task is judged on, and it
holds on exactly one. That distinction is why the corpus test asserts the
roster's single attribution rather than a count.

THE CLASSIFIER'S OWN INPUT is also a real one: the diagnostic block in
``test_the_measured_signature_classifies`` is quoted from the sealed bundle of
``ep-94b968963d3e`` (task_017, cohort7-20260925-002218), which is the episode
this whole facility exists for.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any, Literal

import pytest

from local_operator.evaluation.evidence.models import EvidenceArtifactRef, ScoreArtifact
from local_operator.evaluation.runner.episode import EpisodeRunner
from local_operator.evaluation.triage import (
    APPARATUS_ATTRIBUTABLE,
    CAPABILITY,
    UNREADABLE,
    UNSCORED,
    CampaignReadout,
    EpisodeTriage,
    classify_apparatus_attribution,
    find_bundles,
    format_readout,
    read_bundle,
    read_run,
)
from tests.unit.evaluation.runner.conftest import (
    FakeAdapter,
    ScriptedModel,
    build_config,
    build_spec,
    selector,
)

#: The evaluator log the measured zero carries, verbatim from the sealed
#: score-details artifact. Kept whole rather than paraphrased: a classifier
#: written against a tidy summary of a log is a classifier nobody re-verified
#: against the log.
MEASURED_DIAGNOSTICS = (
    "INFO:desktopenv.getters.chrome:[DEBUG] get_active_url_from_accessTree attempt 1/3 "
    "returned: None (type: <class 'NoneType'>)\n"
    "INFO:desktopenv.getters.chrome:[DEBUG] get_active_url_from_accessTree attempt 2/3 "
    "returned: None (type: <class 'NoneType'>)\n"
    "INFO:desktopenv.getters.chrome:[DEBUG] get_active_url_from_accessTree attempt 3/3 "
    "returned: None (type: <class 'NoneType'>)\n"
    "WARNING:desktopenv.getters.chrome:[DEBUG] active_tab_url is not a string, got "
    "<class 'NoneType'>: None. Falling back to an open Google Maps tab.\n"
    "WARNING:desktopenv.getters.chrome:[DEBUG] wait_for_selector timed out for 'input.ZBTq6e': "
    "Page.wait_for_selector: Timeout 5000ms exceeded.\n"
)

#: The diagnostics of an episode that failed for its OWN reasons -- an ordinary
#: unmet checkpoint, no evaluator trouble anywhere in the log.
PLAIN_DIAGNOSTICS = (
    "INFO:desktopenv.metric.general:[DEBUG] Checking key 'aria-label': "
    "expected_values=['the route'], actual_values=['an empty cart']\n"
    "INFO:desktopenv.metric.general:[DEBUG] Not enough actual values for edge match, "
    "returning 0.0\n"
)


def _canonical(payload: Any) -> bytes:
    """``payload`` as the CANONICAL JSON the artifact validator insists on.

    ``validate_media`` re-encodes and compares, so an artifact declared
    ``application/json`` whose bytes are merely valid JSON is refused -- which is
    how the real adapter stages this file (``scoring._detail_bytes``), and the
    runner reopens it with the same bounded verifier this reader uses.
    """

    return json.dumps(
        payload, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


async def _seal_bundle(
    tmp_path: Path,
    label: str,
    *,
    diagnostics: str | None,
    binary: Literal[0, 1],
    partial_ppm: int,
    details_payload: dict[str, Any] | None = None,
) -> tuple[Path, str]:
    """Run one episode to a sealed bundle whose score carries ``diagnostics``.

    Returns ``(bundle, episode_id)``. The id carries the ``label`` for a human
    reading the report AND a uuid, because episode identity is fail-closed
    process-global (``lifecycle.plan_episode``) and one test process seals three
    episodes: a label alone would collide across tests.

    ``details_payload`` seals a score-details artifact OTHER than the one built
    from ``diagnostics``, for the shapes a diagnostics string cannot express --
    most importantly the pre-capture payload, which has details and no
    ``evaluator_diagnostics`` block in them at all (``diagnostics=None`` seals no
    details artifact, which is a different thing).
    """

    episode_id = f"ep-{label}-{uuid.uuid4().hex[:8]}"
    config = build_config(tmp_path)
    score = ScoreArtifact(status="scored", binary=binary, partial_ppm=partial_ppm)
    payload = details_payload
    if payload is None and diagnostics is not None:
        # The adapter stages its detail bytes in the artifact root; the runner
        # reopens them with the bounded verifier and publishes them into the
        # bundle, which is the path this reader depends on.
        payload = {
            "evaluator_result": None,
            "evaluator_diagnostics": {
                "schema": "lop-evaluator-diagnostics-v1",
                "stderr": {"text": diagnostics},
            },
        }
    if payload is not None:
        data = _canonical(payload)
        digest = hashlib.sha256(data).hexdigest()
        config.artifact_root.mkdir(parents=True, exist_ok=True)
        (config.artifact_root / digest).write_bytes(data)
        score = ScoreArtifact(
            status="scored",
            binary=binary,
            partial_ppm=partial_ppm,
            details=EvidenceArtifactRef(
                sha256=digest, media_type="application/json", byte_count=len(data)
            ),
        )
    adapter = FakeAdapter(tmp_path, episode_id, score=score)
    runner = EpisodeRunner(
        build_spec(episode_id),
        config,
        selector=selector(tmp_path),
        model=ScriptedModel(["finish"]),
        launch=lambda _: adapter,
        rescue=_rescue_ok,
    )

    outcome = await runner.run()
    assert outcome.bundle_root is not None, outcome
    return outcome.bundle_root, episode_id


async def _rescue_ok(descriptor: Any, **kwargs: Any) -> Any:
    del kwargs

    class _Aggregate:
        complete = True
        descriptor_id = descriptor.descriptor_id

    return _Aggregate()


def test_the_measured_signature_classifies_and_no_two_markers_do() -> None:
    """All THREE markers are required, and the reason names what a reader checks next.

    The third is the load-bearing one and was the second revision's whole point.
    The getter failure and the open-tab fallback are emitted by the same code path
    whatever tab the fallback lands on -- all four measured task_017 episodes carry
    both, and three of those four went on to read real content and score, so the
    pair alone would exclude genuine readings. What separates the one real
    mis-grade is that the graded page did not carry the element the task is judged
    on. This pins each marker alone and each PAIR, because a future edit that
    quietly drops one of them is exactly how the false-positive band comes back.
    """

    reason = classify_apparatus_attribution(MEASURED_DIAGNOSTICS)

    assert reason is not None
    assert "get_active_url_from_accessTree" in reason
    assert "input.ZBTq6e" in reason
    assert "came back empty" in reason
    getter = "get_active_url_from_accessTree attempt 1/3 returned: None"
    fallback = "Falling back to an open Google Maps tab."
    timeout = "wait_for_selector timed out for 'input.ZBTq6e': Timeout 5000ms exceeded."
    for marker in (getter, fallback, timeout):
        assert classify_apparatus_attribution(marker) is None, marker
    for pair in (f"{getter}\n{fallback}", f"{getter}\n{timeout}", f"{fallback}\n{timeout}"):
        assert classify_apparatus_attribution(pair) is None, pair
    # And an ordinary unmet checkpoint is not touched: this facility must not
    # launder capability failures.
    assert classify_apparatus_attribution(PLAIN_DIAGNOSTICS) is None
    assert classify_apparatus_attribution("") is None


@pytest.mark.asyncio
async def test_a_sealed_apparatus_zero_is_excluded_and_named(tmp_path: Path) -> None:
    bundle, episode_id = await _seal_bundle(
        tmp_path,
        "apparatus",
        diagnostics=MEASURED_DIAGNOSTICS,
        binary=0,
        partial_ppm=0,
    )

    read = read_bundle(bundle)

    assert read.verified
    assert read.episode_id == episode_id
    assert read.binary == 0
    assert read.attribution == APPARATUS_ATTRIBUTABLE
    assert read.reason and "open-tab fallback" in read.reason
    assert not read.counts_toward_capability


@pytest.mark.asyncio
async def test_a_zero_without_the_signature_still_counts_against_capability(
    tmp_path: Path,
) -> None:
    bundle, _ = await _seal_bundle(
        tmp_path,
        "capability",
        diagnostics=PLAIN_DIAGNOSTICS,
        binary=0,
        partial_ppm=0,
    )

    read = read_bundle(bundle)

    assert read.verified
    assert read.attribution == CAPABILITY
    assert read.reason is None
    assert read.counts_toward_capability


@pytest.mark.asyncio
async def test_a_correct_episode_counts_toward_the_rate(tmp_path: Path) -> None:
    bundle, _ = await _seal_bundle(
        tmp_path, "right", diagnostics=None, binary=1, partial_ppm=1_000_000
    )

    read = read_bundle(bundle)

    assert read.attribution == CAPABILITY
    assert read.counts_toward_capability


@pytest.mark.asyncio
async def test_a_manufactured_pass_leaves_both_sides_of_the_rate(tmp_path: Path) -> None:
    """THE REGRESSION: the same defect manufactures PASSES, and they must not count.

    An earlier revision read every episode through ``binary == 0``, so the two
    task_017 passes in the measured corpus were never even asked the question:
    the matching zero left the denominator while the manufactured pass stayed in
    the numerator, which inflates the arm exactly as much as the facility
    corrects it. Sealing a PASS whose diagnostics carry the signature is the
    smallest bundle that can catch that, and it only passes if the binary is not
    consulted at all.
    """

    bundle, _ = await _seal_bundle(
        tmp_path,
        "manufactured-pass",
        diagnostics=MEASURED_DIAGNOSTICS,
        binary=1,
        partial_ppm=1_000_000,
    )

    read = read_bundle(bundle)

    assert read.verified
    assert read.binary == 1
    assert read.attribution == APPARATUS_ATTRIBUTABLE, read.reason
    assert read.reason and "came back empty" in read.reason
    assert not read.counts_toward_capability


@pytest.mark.asyncio
@pytest.mark.parametrize("binary", [0, 1])
async def test_a_recovered_right_tab_fallback_is_not_attributed(
    tmp_path: Path, binary: Literal[0, 1]
) -> None:
    """THE REGRESSION THE SIGNATURE ALONE WOULD INTRODUCE, pinned both directions.

    This is the shape of the three measured episodes the pair does NOT explain: the
    active-URL getter failed, the fallback fired, and the read that followed was
    REAL (for the two passes, the expected eight-stop route; for the other zero, a
    1/7 partial against the route it did produce). Nothing in the log says the
    fallback landed on a different page -- its chosen URL is in fact byte-identical
    to the evaluator's own target -- so a classifier keyed on the pair alone would
    strip a genuine pass out of the numerator. Both binaries are driven because the
    rule must not consult the score either way.
    """

    right_tab = MEASURED_DIAGNOSTICS.replace(
        "WARNING:desktopenv.getters.chrome:[DEBUG] wait_for_selector timed out for "
        "'input.ZBTq6e': Page.wait_for_selector: Timeout 5000ms exceeded.\n",
        "INFO:desktopenv.getters.chrome:[DEBUG] get_active_tab_html_parse final result: "
        "{'aria-label': ['Starting point 5454 S Shore Dr, Chicago, IL 60615']}\n",
    )
    assert "timed out" not in right_tab
    bundle, _ = await _seal_bundle(
        tmp_path,
        f"right-tab-{binary}",
        diagnostics=right_tab,
        binary=binary,
        partial_ppm=binary * 1_000_000,
    )

    read = read_bundle(bundle)

    assert read.attribution == CAPABILITY, read.reason
    assert read.counts_toward_capability
    assert read.reason is None


@pytest.mark.asyncio
async def test_the_readout_excludes_the_apparatus_zero_and_names_it(tmp_path: Path) -> None:
    _, apparatus_id = await _seal_bundle(
        tmp_path, "apparatus", diagnostics=MEASURED_DIAGNOSTICS, binary=0, partial_ppm=0
    )
    await _seal_bundle(
        tmp_path, "capability", diagnostics=PLAIN_DIAGNOSTICS, binary=0, partial_ppm=0
    )
    await _seal_bundle(tmp_path, "right", diagnostics=None, binary=1, partial_ppm=1_000_000)

    readout = read_run([tmp_path / "evidence"])

    assert len(readout.episodes) == 3
    assert len(readout.scored) == 3
    assert readout.correct == 1
    assert readout.denominator == 2
    assert readout.capability_rate == pytest.approx(0.5)
    # The binary rate over every scored episode is the OTHER number, and the
    # difference between the two is exactly the excluded zero.
    assert [e.episode_id for e in readout.apparatus_attributable] == [apparatus_id]
    report = format_readout(readout)
    assert "capability rate         1/2 = 50.0%" in report
    assert f"{apparatus_id}  [apparatus-attributable] binary=0" in report


@pytest.mark.asyncio
async def test_a_manufactured_pass_is_named_and_moves_neither_number(tmp_path: Path) -> None:
    """Both directions in one reading: the pass and the zero each leave the rate.

    Four scored episodes, one of each kind. The two apparatus-attributable ones
    are excluded, so the rate stays 1/2 -- while the naive binary rate over the
    same bundles would read 2/3, the manufactured pass having been the numerator
    the earlier revision could not see. The report has to show WHICH side each
    exclusion came off, because that is the difference between the two figures.
    """

    _, apparatus_zero_id = await _seal_bundle(
        tmp_path, "apparatus-zero", diagnostics=MEASURED_DIAGNOSTICS, binary=0, partial_ppm=0
    )
    _, apparatus_pass_id = await _seal_bundle(
        tmp_path,
        "apparatus-pass",
        diagnostics=MEASURED_DIAGNOSTICS,
        binary=1,
        partial_ppm=1_000_000,
    )
    await _seal_bundle(
        tmp_path, "capability", diagnostics=PLAIN_DIAGNOSTICS, binary=0, partial_ppm=0
    )
    await _seal_bundle(tmp_path, "right", diagnostics=None, binary=1, partial_ppm=1_000_000)

    readout = read_run([tmp_path / "evidence"])

    assert len(readout.scored) == 4
    assert len(readout.apparatus_attributable) == 2
    assert readout.correct == 1
    assert readout.denominator == 2
    assert readout.capability_rate == pytest.approx(0.5)
    # The number the correction exists to replace: what a reader who counted
    # every binary == 1 would have quoted.
    assert sum(1 for e in readout.scored if e.binary == 1) == 2
    report = format_readout(readout)
    assert f"{apparatus_pass_id}  [apparatus-attributable] binary=1" in report
    assert f"{apparatus_zero_id}  [apparatus-attributable] binary=0" in report


def test_an_unreadable_bundle_is_named_rather_than_guessed(tmp_path: Path) -> None:
    """A directory that is not a verifiable bundle is a hole in the rate.

    The failure mode this guards is a reader that treats "I could not read it"
    as "not a pass": that turns a broken apparatus into a capability failure,
    which is the same mistake the apparatus triage exists to stop.
    """

    broken = tmp_path / "evidence" / "ep-broken"
    broken.mkdir(parents=True)
    (broken / "events.jsonl").write_text("")

    read = read_bundle(broken)

    assert read.verified is False
    assert read.attribution == UNREADABLE
    assert read.score is None
    assert not read.counts_toward_capability
    assert read.reason and "does not verify" in read.reason


def test_the_rate_is_none_rather_than_zero_when_nothing_can_be_rated() -> None:
    """ "No episodes to rate" and "none of them passed" are different readings."""

    readout = CampaignReadout(
        episodes=(
            EpisodeTriage(
                episode_id="ep-a",
                bundle=Path("unused"),
                verified=False,
                score=None,
                attribution=UNSCORED,
                reason="no scoring_result event: the evaluator never produced a score",
            ),
            EpisodeTriage(
                episode_id="ep-b",
                bundle=Path("unused"),
                verified=False,
                score=None,
                attribution=UNREADABLE,
                reason="bundle did not verify",
            ),
        )
    )

    assert readout.denominator == 0
    assert readout.capability_rate is None
    report = format_readout(readout)
    assert "(nothing to rate)" in report
    assert "ep-a  [unscored]" in report
    assert "ep-b  [unreadable]" in report


def test_an_empty_directory_reads_as_nothing_rather_than_being_guessed(tmp_path: Path) -> None:
    """An empty run tree is a reading of zero, and a typo is refused.

    The distinction is the one the corpus forced: a real run tree on this
    machine has an ``evidence/`` subtree with no bundles in it, so aborting a
    reading over that would make the facility unusable. A path that is not a
    directory at all is a different thing -- a typo -- and is refused.
    """

    assert find_bundles(tmp_path) == []
    with pytest.raises(ValueError, match="is not a directory"):
        find_bundles(tmp_path / "does-not-exist")


# ---------------------------------------------------------------------------
# The corpus, when the machine has one
# ---------------------------------------------------------------------------

#: Where the campaign's sealed bundles live. Overridable for the same reason
#: ``test_rejection_classes.py``'s corpus is: this is a paid run's output rather
#: than a repository fixture, so the test must hold on a machine that has one and
#: SKIP on every machine that does not, CI included, rather than pretending the
#: classification is verified.
CORPUS_ENV = "LOCAL_OPERATOR_TRIAGE_CORPUS"
DEFAULT_CORPUS = Path.home() / "worktrees" / "osworld" / "runs"
#: The episode this facility was built from: task_017 of cohort7, whose route was
#: on screen and whose zero came from the evaluator reading another tab.
MEASURED_EPISODE_GLOB = "cohort7-*/evidence/ep-94b968963d3e"

#: The full roster the 2026-09-26 audit of the 96 sealed bundles found carrying
#: the getter-failure + open-tab-fallback pair: four task_017 episodes across
#: three arms. Reading them properly is what set the three-marker rule -- three of
#: the four read REAL content (two passes on the expected route, one 1/7 partial)
#: and only the cohort7 zero is a mis-grade.
MEASURED_ROSTER_GLOBS = (
    "cohort7-*/evidence/ep-94b968963d3e",
    "gateOFF-task_017-*/evidence/ep-9a6876e889e1",
    "gateON-t017-*/evidence/ep-b62377fe1ba5",
    "strong-task_017-*/evidence/ep-3dcae8b48633",
)


def _corpus_root() -> Path:
    return Path(os.environ.get(CORPUS_ENV) or DEFAULT_CORPUS)


def test_the_measured_episode_is_classified_from_its_sealed_bundle() -> None:
    episodes = sorted(_corpus_root().glob(MEASURED_EPISODE_GLOB))
    if not episodes:
        pytest.skip(f"no campaign corpus at {_corpus_root()} (set {CORPUS_ENV} to point at one)")

    read = read_bundle(episodes[0])

    assert read.verified is True, read.reason
    assert read.binary == 0
    assert read.attribution == APPARATUS_ATTRIBUTABLE, read.reason


def test_the_measured_roster_is_classified_from_its_own_diagnostics() -> None:
    """The corpus reading: ONE mis-grade, and three readings left alone.

    Read over the roster alone, where the arithmetic is stable whatever else the
    campaign later adds. Three of the four carry the getter failure and the
    fallback and NOTHING else -- their reads returned real content, so they are
    readings of the state the episode produced and must stay in the rate; two of
    them are the passes. Only ``ep-94b968963d3e`` is a mis-grade, and it is a zero.
    That asymmetry is the whole reason the rule consults the diagnostics rather
    than the binary: keying on the pair alone drops the passes, and keying on
    ``binary == 0`` drops the partial's reading too. Across the whole audited
    corpus this is what moves the figure to 6/63 = 9.5%.
    """

    found = [sorted(_corpus_root().glob(glob)) for glob in MEASURED_ROSTER_GLOBS]
    if not all(found):
        pytest.skip(f"the four measured bundles are not all at {_corpus_root()}")

    readout = read_run([bundles[0] for bundles in found])
    by_id = {episode.episode_id: episode for episode in readout.episodes}

    assert len(by_id) == 4
    assert all(episode.verified for episode in readout.episodes)
    # Order is roster order -- cohort7, gateOFF, gateON, strong.
    assert [episode.binary for episode in readout.scored] == [0, 0, 1, 1]
    attributed = [episode.episode_id for episode in readout.apparatus_attributable]
    assert attributed == ["ep-94b968963d3e"]
    # Including both PASSES, which the pair alone would have excluded.
    assert {episode.episode_id for episode in readout.scored if episode.binary == 1} <= {
        episode.episode_id for episode in readout.scored if episode.counts_toward_capability
    }
    assert readout.correct == 2
    assert readout.denominator == 3
    assert readout.capability_rate is not None
    assert abs(readout.capability_rate - 2 / 3) < 1e-9
    report = format_readout(readout)
    assert report.count("[apparatus-attributable]") == 1
    assert "ep-94b968963d3e  [apparatus-attributable] binary=0" in report


@pytest.mark.asyncio
async def test_details_without_a_diagnostics_block_read_as_capability(tmp_path: Path) -> None:
    """The pre-capture shape: details present, no evaluator log inside them.

    Every run from before the diagnostics capture existed seals a score whose
    details carry an evaluator result and no ``evaluator_diagnostics`` block at
    all -- and ``diagnostics=None`` seals no details artifact, which is a
    DIFFERENT shape and takes a different branch. Both have to leave the episode a
    capability reading: the honest reading of "nothing was captured to attribute
    this score to" is that the score stands, and a default that excluded it would
    silently empty the rate on every older bundle.
    """

    bundle, _ = await _seal_bundle(
        tmp_path,
        "no-diagnostics-block",
        diagnostics=None,
        details_payload={"evaluator_result": 1.0},
        binary=1,
        partial_ppm=1_000_000,
    )

    read = read_bundle(bundle)

    assert read.verified
    assert read.score is not None and read.score.details is not None
    assert read.attribution == CAPABILITY
    assert read.counts_toward_capability


def test_the_corpus_reading_excludes_only_what_it_can_name() -> None:
    """Over every run the machine has: nothing is excluded without a reason."""

    root = _corpus_root()
    runs = sorted(path for path in root.glob("*/evidence") if path.is_dir())
    if not runs:
        pytest.skip(f"no campaign corpus at {root} (set {CORPUS_ENV} to point at one)")

    readout = read_run(runs)

    assert readout.episodes, "a run directory with no bundles is not a reading"
    for episode in readout.episodes:
        if not episode.counts_toward_capability:
            assert episode.reason, episode
    # The measured signature is rare and named when it is there: a cohort in
    # which MOST zeros were apparatus-attributable would mean the classifier is
    # matching the wrong thing.
    apparatus = readout.apparatus_attributable
    assert len(apparatus) <= max(1, len(readout.scored))
    for episode in apparatus:
        assert "get_active_url_from_accessTree" in (episode.reason or "")
