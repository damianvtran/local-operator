"""Read a scored episode's verdict, and say whether the APPARATUS produced it.

WHY THIS EXISTS, precisely. The campaign of 2026-09-25 scored ``task_017``
zero while a rendered Google Maps walking-route page -- every stop, the travel
time, walking mode -- was the frame its ``finish`` was bound to. The model's
claim was TRUE. The evaluator's own retained diagnostics show why the score is
zero anyway: ``get_active_url_from_accessTree`` returned ``None`` on all three
attempts, the fallback then selected a DIFFERENT open Google Maps tab (a place
page, which has no directions inputs), and the selector it went on to wait for
timed out.

FOUR SCORED EPISODES CARRY THAT SIGNATURE, NOT ONE, AND IT MANUFACTURES PASSES
AS WELL AS ZEROS. The audit of all 96 sealed bundles (2026-09-26) found it on
four ``task_017`` episodes -- the Maps task, the one that opens several Maps
tabs -- across three arms:

    ep-94b968963d3e  cohort7-20260925-002218           binary 0  partial 0
    ep-9a6876e889e1  gateOFF-task_017-20260925-184401  binary 0  partial 142857
    ep-b62377fe1ba5  gateON-t017-20260925-181934       binary 1  partial 1000000
    ep-3dcae8b48633  strong-task_017-20260926-043257   binary 1  partial 1000000

(The two zeros are the pair an earlier revision of this module flagged; the two
ones are the pair it could not see, because it triaged only ``binary == 0``.)

The fallback grades a tab the episode did not produce, so its score is not a
reading in EITHER direction, and the correction has to move both sides of the
rate. Flagging only the zeros -- what this module did -- removes two zeros from
the denominator while leaving two manufactured PASSES in the numerator, which
inflates the arm as much as it corrects it. Over this corpus the honest figure
is 4/60 = 6.7% against an uncorrected binary rate of 6/62 = 9.7%; the honest
capability reading is 4-6 correct of 60-62, the width being roster completeness
(a signature a cohort does not carry cannot be corrected out of it) rather than
arithmetic.

SO THIS IS A READ-SIDE TRIAGE, NOT A HARNESS FIX. The pinned vendor tree's
getter is deliberately left alone: patching it would make the arm
non-comparable with the published benchmark (``INFRA.md``'s rule about upstream
retry policy), and the next reader could not tell the two arms apart from a
score. The adapter already seals the evaluator's diagnostics into the
score-details artifact, so the classification is derivable from the sealed
bundle and nothing about scoring changes.

WHAT IT DOES NOT DO. It does not decide that an episode "really" succeeded, and
it never rewrites a score: an apparatus-attributable episode is EXCLUDED and
NAMED from BOTH sides of the rate, so the figure is computed over the episodes
whose score the apparatus could actually read. Unscored episodes are excluded
and named for the same reason and by the same rule (``INFRA.md``: "Exclude them
honestly and name why; do not count them as zeros"). A campaign reading this
rate still has to exclude a non-reportable arm -- a scripted or fake-provider
run -- by its own manifest ``reportability_label``; that is the driver's stamp
and not something a score can be triaged into.

Run it over a run directory (``<run>/evidence/<episode>`` bundles) or over one
bundle:

    python -m local_operator.evaluation.triage <run-root | bundle-root> [--json]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from local_operator.evaluation.adapters.supervisor import (
    SupervisionError,
    verify_artifact,
)
from local_operator.evaluation.evidence.models import (
    ScoreArtifact,
    ScoringResultPayload,
)
from local_operator.evaluation.evidence.verify import verify_bundle

#: The classification this module can add to a score. A STRING rather than a bool
#: so a report can carry the reason, and a closed set so a later classification
#: (a second apparatus signature) is an addition to it rather than a second
#: boolean every reader would have to learn.
CAPABILITY = "capability"
APPARATUS_ATTRIBUTABLE = "apparatus-attributable"
UNSCORED = "unscored"
UNREADABLE = "unreadable"

# The signature, as the pinned getter and its caller spell it in their own log
# records. Both halves are required: the getter returning ``None`` is what the
# evaluator tried to recover from, and the open-tab fallback is what it did
# about it -- a fallback that picked the RIGHT tab would be invisible here (the
# score would simply be right), so the pair is what makes the reading
# attributable rather than merely suspicious.
_ACTIVE_URL_NONE_RE = re.compile(r"get_active_url_from_accessTree[^\n]*\breturned:\s*None")
_OPEN_TAB_FALLBACK_RE = re.compile(r"Falling back to an open[^\n]*\btab\b")
# Corroboration, not a requirement: the fallback lands on a page without the
# element the task is judged on, which is why the read produced nothing. Named
# in the reason when present, because it is what a reader checks next.
_WAIT_TIMEOUT_RE = re.compile(r"wait_for_selector timed out for '([^']*)'")

#: The diagnostics key ``scoring`` wraps a captured block under.
_DIAGNOSTICS_KEY = "evaluator_diagnostics"


def classify_apparatus_attribution(diagnostics_text: str) -> str | None:
    """Why this episode looks apparatus-attributable, or ``None`` if it does not.

    Deliberately takes the diagnostics and NOT the score: the signature can land
    on an episode that passed exactly as it can on one that failed (see the
    roster in the module docstring), so a caller that only asked about zeros
    would miss half of what the defect manufactures.

    ``diagnostics_text`` is the evaluator's retained stdout and stderr, joined.
    A capture that was TRUNCATED can hide the signature (the ring drops the head
    of the stream), which costs a false negative -- an unclassified episode that
    stays in the rate -- and never a false positive, because both markers have
    to be present in the bytes that survived.
    """

    active_url = _ACTIVE_URL_NONE_RE.search(diagnostics_text)
    if active_url is None or _OPEN_TAB_FALLBACK_RE.search(diagnostics_text) is None:
        return None
    reason = (
        "the evaluator could not read the active tab's URL "
        "('get_active_url_from_accessTree ... returned: None') and its open-tab "
        "fallback selected a different tab, so the state it graded was not the "
        "state the episode produced"
    )
    timeout = _WAIT_TIMEOUT_RE.search(diagnostics_text)
    if timeout is not None:
        reason += f"; the selector it then waited for ({timeout.group(1)!r}) timed out"
    return reason


def _diagnostics_text(details: Any) -> str:
    """The evaluator's retained streams out of a parsed score-details payload.

    Always a string: a payload with no diagnostics block (every run before the
    capture existed, and every evaluator that emits nothing) classifies as
    "not apparatus-attributable", which is the honest reading of "nothing was
    captured to attribute it to".
    """

    if not isinstance(details, dict):
        return ""
    block = details.get(_DIAGNOSTICS_KEY)
    if not isinstance(block, dict):
        return ""
    parts: list[str] = []
    for stream in ("stdout", "stderr"):
        text = block.get(stream)
        if isinstance(text, dict) and isinstance(text.get("text"), str):
            parts.append(text["text"])
    return "\n".join(parts)


@dataclass(frozen=True)
class EpisodeTriage:
    """One bundle's score, and what (if anything) can be said about it."""

    episode_id: str
    bundle: Path
    verified: bool
    score: ScoreArtifact | None
    attribution: str
    reason: str | None = None
    issues: tuple[str, ...] = ()

    @property
    def binary(self) -> int | None:
        return self.score.binary if self.score is not None else None

    @property
    def partial_ppm(self) -> int | None:
        return self.score.partial_ppm if self.score is not None else None

    @property
    def counts_toward_capability(self) -> bool:
        """Whether this episode is a capability reading at all.

        A score the apparatus produced is not one -- in EITHER direction, a
        manufactured pass inflating the numerator exactly as a manufactured zero
        deflates it -- and neither is an episode the evaluator never scored. The
        first is what this module classifies, so the two are both excluded and
        separately named.
        """

        return self.attribution in (CAPABILITY,)


@dataclass(frozen=True)
class CampaignReadout:
    """The counts a campaign reports, with the exclusions named."""

    episodes: tuple[EpisodeTriage, ...]

    @property
    def scored(self) -> tuple[EpisodeTriage, ...]:
        return tuple(e for e in self.episodes if e.score is not None)

    @property
    def correct(self) -> int:
        """Passes that are capability readings.

        Not ``sum(binary == 1)``: a pass the apparatus manufactured is excluded
        from the numerator for the same reason a manufactured zero is excluded
        from the denominator, so this counts only the episodes that count.
        """

        return sum(1 for e in self.scored if e.binary == 1 and e.counts_toward_capability)

    @property
    def apparatus_attributable(self) -> tuple[EpisodeTriage, ...]:
        return tuple(e for e in self.episodes if e.attribution == APPARATUS_ATTRIBUTABLE)

    @property
    def unscored(self) -> tuple[EpisodeTriage, ...]:
        return tuple(e for e in self.episodes if e.score is None)

    @property
    def denominator(self) -> int:
        return sum(1 for e in self.episodes if e.counts_toward_capability)

    @property
    def capability_rate(self) -> float | None:
        """Binary rate over episodes whose score the apparatus could read.

        ``None`` rather than ``0.0`` for an empty denominator: "no episodes to
        rate" and "none of them passed" are different readings, and a rate that
        reports zero for both is the shape that gets quoted as a result.
        """

        if self.denominator == 0:
            return None
        return self.correct / self.denominator

    def as_json(self) -> dict[str, Any]:
        return {
            "episodes": len(self.episodes),
            "scored": len(self.scored),
            "correct": self.correct,
            "apparatus_attributable": len(self.apparatus_attributable),
            "unscored": len(self.unscored),
            "rate_denominator": self.denominator,
            "capability_rate": self.capability_rate,
            "detail": [
                {
                    "episode_id": e.episode_id,
                    "bundle": str(e.bundle),
                    "verified": e.verified,
                    "binary": e.binary,
                    "partial_ppm": e.partial_ppm,
                    "attribution": e.attribution,
                    "reason": e.reason,
                    "issues": list(e.issues),
                }
                for e in self.episodes
            ],
        }


def read_bundle(bundle: Path) -> EpisodeTriage:
    """One bundle's score, verified and triaged. Never raises for a bad bundle.

    Everything comes from the sealed evidence: the verifier recomputes the
    bundle, the ``scoring_result`` event carries the score, and the score's
    details artifact is reopened BY DIGEST through the same reader the runner
    uses. An unverifiable bundle is reported as such rather than triaged -- a
    score read out of evidence that does not verify is not a reading, which is
    the whole reason the verifier exists.
    """

    label = bundle.name
    report = verify_bundle(bundle)
    if not report.valid:
        return EpisodeTriage(
            episode_id=label,
            bundle=bundle,
            verified=False,
            score=None,
            attribution=UNREADABLE,
            reason="bundle does not verify: "
            + ", ".join(sorted({issue.code for issue in report.issues})),
            issues=tuple(f"{issue.code}@{issue.location}" for issue in report.issues),
        )
    scores = [
        event.payload.score
        for event in report.events
        if isinstance(event.payload, ScoringResultPayload)
    ]
    if not scores:
        return EpisodeTriage(
            episode_id=label,
            bundle=bundle,
            verified=True,
            score=None,
            attribution=UNSCORED,
            reason="no scoring_result event: the evaluator never produced a score",
        )
    score = scores[0]
    episode_id = report.manifest.episode_id if report.manifest is not None else label
    if score.details is None:
        # Nothing was captured to attribute the score to, so there is nothing to
        # classify and the score stands as the capability reading it is. The
        # binary is deliberately NOT consulted: the same evaluator defect
        # manufactures passes as well as zeros (module docstring), and a rule of
        # "only a zero is triaged" is exactly what left two manufactured passes
        # in this corpus's numerator.
        return EpisodeTriage(
            episode_id=episode_id,
            bundle=bundle,
            verified=True,
            score=score,
            attribution=CAPABILITY,
        )
    try:
        details = json.loads(verify_artifact(bundle / "artifacts", score.details).decode("utf-8"))
    except (OSError, UnicodeDecodeError, ValueError, SupervisionError) as error:
        return EpisodeTriage(
            episode_id=episode_id,
            bundle=bundle,
            verified=True,
            score=score,
            attribution=CAPABILITY,
            reason=f"score details could not be read ({type(error).__name__}); "
            "the score stands unless another reading explains it",
        )
    reason = classify_apparatus_attribution(_diagnostics_text(details))
    if reason is None:
        return EpisodeTriage(
            episode_id=episode_id,
            bundle=bundle,
            verified=True,
            score=score,
            attribution=CAPABILITY,
        )
    return EpisodeTriage(
        episode_id=episode_id,
        bundle=bundle,
        verified=True,
        score=score,
        attribution=APPARATUS_ATTRIBUTABLE,
        reason=reason,
    )


def find_bundles(root: Path) -> list[Path]:
    """Every episode bundle under ``root``.

    Three shapes are accepted, and they are the ones that exist on disk: a
    single episode bundle (``events.jsonl``); a run root, whose bundles live
    under its ``evidence/`` subtree; and a directory OF bundles, which is what
    that subtree is when a reader is handed ``<run>/evidence`` directly rather
    than the run root. The search is never recursive: every real layout is one
    level deep, and a walk that guesses is how a report ends up silently
    covering one episode of a cohort.

    A directory that HOLDS no bundle returns ``[]`` rather than raising. A
    campaign's run tree really does contain an empty ``evidence/`` subtree --
    the runs on this machine have one -- and aborting the whole reading over it
    would make the facility unusable on the corpus it exists for. The empty
    answer is visible in the readout as ``episodes 0``, which is a reading; a
    path that is not a directory at all is a typo, and that is refused.
    """

    if (root / "events.jsonl").is_file():
        return [root]
    if not root.is_dir():
        raise ValueError(f"{root} is not a directory and is not an episode bundle")
    for candidate in (root / "evidence", root):
        if not candidate.is_dir():
            continue
        bundles = sorted(path for path in candidate.iterdir() if (path / "events.jsonl").is_file())
        if bundles:
            return bundles
    return []


def read_run(paths: Sequence[Path]) -> CampaignReadout:
    bundles: list[Path] = []
    for path in paths:
        bundles.extend(find_bundles(path))
    return CampaignReadout(episodes=tuple(read_bundle(bundle) for bundle in bundles))


def format_readout(readout: CampaignReadout) -> str:
    """The report, with every exclusion named on its own line."""

    rate = readout.capability_rate
    lines = [
        "completion triage",
        f"  episodes                {len(readout.episodes)}",
        f"  scored                  {len(readout.scored)}",
        f"  correct                 {readout.correct}",
        f"  apparatus-attributable  {len(readout.apparatus_attributable)}",
        f"  unscored                {len(readout.unscored)}",
        f"  capability rate         {readout.correct}/{readout.denominator}"
        + (f" = {rate:.1%}" if rate is not None else " (nothing to rate)"),
    ]
    excluded = [e for e in readout.episodes if not e.counts_toward_capability]
    if excluded:
        lines.append("  excluded, named:")
        for episode in excluded:
            # The binary is shown because the exclusion can come off either side
            # of the rate now: a manufactured pass leaves the numerator exactly as
            # a manufactured zero leaves the denominator, and a reader checking
            # the arithmetic needs to see which of the two each line is. An
            # unscored or unreadable episode has no score to show, and its
            # attribution already says so.
            scored = "" if episode.binary is None else f"binary={episode.binary} "
            lines.append(
                f"    {episode.episode_id}  [{episode.attribution}] {scored}{episode.reason}"
            )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m local_operator.evaluation.triage",
        description="Classify apparatus-attributable episodes out of a capability rate",
    )
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="a run root, an episode bundle, or a directory of episode bundles",
    )
    parser.add_argument("--json", action="store_true", help="emit the readout as JSON")
    args = parser.parse_args(argv)

    try:
        readout = read_run(args.paths)
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(readout.as_json(), indent=2, sort_keys=True))
    else:
        print(format_readout(readout))
    return 0


if __name__ == "__main__":
    sys.exit(main())
