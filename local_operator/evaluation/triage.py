"""Read a scored episode's verdict, and say whether the APPARATUS produced it.

WHY THIS EXISTS, precisely. The campaign of 2026-09-25 scored ``task_017``
zero while a rendered Google Maps walking-route page -- every stop, the travel
time, walking mode -- was the frame its ``finish`` was bound to. The model's
claim was TRUE. The evaluator's own retained diagnostics show why the score is
zero anyway: ``get_active_url_from_accessTree`` returned ``None`` on all three
attempts, the fallback then selected a DIFFERENT open Google Maps tab (a place
page, which has no directions inputs), and the selector it went on to wait for
timed out.

FOUR SCORED EPISODES CARRY THE GETTER-AND-FALLBACK PAIR, AND THAT PAIR IS NOT
BY ITSELF A MIS-GRADE. The audit of all 96 sealed bundles (2026-09-26) found the
pair on four ``task_017`` episodes -- the Maps task, the one that opens several
Maps tabs -- across three arms:

    ep-94b968963d3e  cohort7-20260925-002218           binary 0  partial 0
    ep-9a6876e889e1  gateOFF-task_017-20260925-184401  binary 0  partial 142857
    ep-b62377fe1ba5  gateON-t017-20260925-181934       binary 1  partial 1000000
    ep-3dcae8b48633  strong-task_017-20260926-043257   binary 1  partial 1000000

Reading all four properly is what produced the rule below, and it refutes an
earlier version of this docstring. In EVERY one of the four the fallback's chosen
URL is byte-identical to the URL the evaluator itself printed as its target page
-- a selection echo, not agreement: ``chrome.py`` selects the page it prints BY
the fallback's own URL, so the two lines cannot disagree -- and in three of the
four the parse then returned REAL content that went on to be scored: the two
passes matched the expected eight-stop walking route, and ``ep-9a6876e889e1``
scored a 1/7 partial against the route it did produce. Those three are readings
of the state the episode produced -- the discriminating evidence is the read that
followed, not the URL echo. Only ``ep-94b968963d3e`` is a mis-grade: the element
the task is judged on was not on the graded page, the selector the evaluator
waited for timed out, and the read it scored came back empty
(``{'aria-label': []}``) where the check expected content. The classifier keys on
that shape's three log markers -- a proxy for the read, not a check of it, with
its residuals stated at ``_WAIT_TIMEOUT_RE`` below.

So the classifier requires ALL THREE markers -- getter failure, open-tab
fallback, and a timed-out selector -- and the third is load-bearing, not
corroborative. On this corpus that flags one episode, and the honest figure is
6/63 = 9.5% against the naive binary rate of 6/64 = 9.4%. The two are close
because this corpus contains no manufactured PASS: had the fallback graded a page
whose EMPTY read happened to satisfy an empty expectation, the mis-grade would
have landed on a pass, which is the case a rule that consults ``binary`` at all
could not see.

SO THIS IS A READ-SIDE TRIAGE, NOT A HARNESS FIX. The pinned vendor tree's
getter is deliberately left alone: ``docs/benchmarks/osworld_2/README.md``
("Comparability") makes a number reportable only when its bundle verifies, its
reportability is ``reportable`` and its comparability is ``comparable``, so
changing the apparatus the pin names would move the arm out from under its own
score rather than fix the score. (``~/worktrees/osworld/INFRA.md`` states the
same rule for this campaign's runs; it is the campaign's own record and not in
this repository, so the in-repo document is the citation to check.) The adapter
already seals the evaluator's diagnostics into the score-details artifact, so the
classification is derivable from the sealed bundle and nothing about scoring
changes.

WHAT IT DOES NOT DO. It does not decide that an episode "really" succeeded, and
it never rewrites a score: an apparatus-attributable episode is EXCLUDED and
NAMED from BOTH sides of the rate, so the figure is computed over the episodes
whose score the apparatus could actually read. Unscored episodes are excluded and
named for the same reason and by the same rule -- a score has to exist before it
can be a reading of anything. A campaign reading this rate still has to exclude a
non-reportable arm -- a scripted or fake-provider run -- by its own manifest
``reportability_label``; that is the driver's stamp (see the comparability rule
above) and not something a score can be triaged into.

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
# records. It takes THREE markers, and the third is the one that does the work.
#
# THE FIRST TWO ALONE ARE NOT ATTRIBUTION, and an earlier revision of this module
# assumed they were on the reasoning that "a fallback that picked the RIGHT tab
# would be invisible here (the score would simply be right)". That reasoning is
# false, and the sealed corpus shows it: the getter-failure and the fallback are
# emitted by the same code path whatever tab the fallback lands on. In all four
# measured task_017 episodes the fallback's chosen URL is byte-identical to the
# URL the evaluator itself printed as its target page -- a selection echo, not
# agreement: ``chrome.py`` selects the page it prints BY the fallback's own URL,
# so the two lines cannot disagree -- and in three of the four the parse then
# produced REAL content that went on to be scored (two passes on the expected
# route, one 1/7 partial). Those three are readings, so classifying them would
# remove genuine passes from the numerator -- the exact error this module exists
# to avoid, aimed the other way.
#
# What separates the fourth (a genuine mis-grade) is that the element the check
# asked for was NOT on the graded page, so its read came back empty
# (``{'aria-label': []}`` against an evaluator that expected content). That
# emptiness is what ``_WAIT_TIMEOUT_RE`` stands in for, and it is why the
# corroborator is now a REQUIREMENT.
#
# AND THE THIRD MARKER IS A PROXY FOR THAT EMPTY READ, NOT A CHECK OF IT: the
# classifier keys on the log and never opens the graded content. Two residuals
# are stated here rather than closed, and neither has an instance in the
# 96-bundle corpus (so nothing moves today -- they are named so no reader takes
# the markers as an exact reconstruction of the read). A run whose read was REAL
# can still carry all three markers -- a wait timeout for one selector while the
# graded read another path produces stays non-empty, a wait/read race or a
# multi-selector config -- and classifies identically, the reason saying "came
# back empty" however the read turned out. And a genuine capability failure with
# the same geometry (the element truly absent, the fallback on a route-less Maps
# tab, the timeout fired, the read empty) emits an identical marker set, which
# only the campaign's finish-frame evidence could separate from
# ``ep-94b968963d3e``'s shape.
_ACTIVE_URL_NONE_RE = re.compile(r"get_active_url_from_accessTree[^\n]*\breturned:\s*None")
_OPEN_TAB_FALLBACK_RE = re.compile(r"Falling back to an open[^\n]*\btab\b")
# Required, not corroborative: the graded page did not carry the element the task
# is judged on, which is the observable difference between "the fallback graded a
# page the episode did not produce" and "the fallback recovered the target and the
# read was real". Named in the reason because it is what a reader checks next.
_WAIT_TIMEOUT_RE = re.compile(r"wait_for_selector timed out for '([^']*)'")

#: The diagnostics key ``scoring`` wraps a captured block under.
_DIAGNOSTICS_KEY = "evaluator_diagnostics"


def classify_apparatus_attribution(diagnostics_text: str) -> str | None:
    """Why this episode looks apparatus-attributable, or ``None`` if it does not.

    All three markers are required; see the comment above them for why the first
    two are not enough and what the third buys.

    Deliberately takes the diagnostics and NOT the score. The check is about what
    the evaluator could READ, and a mis-grade can land on either verdict: a read
    that came back empty manufactures a zero, and one compared against a
    mis-selected page could as easily confirm a pass. A caller that only asked
    about zeros could not see the second case at all.

    ``diagnostics_text`` is the evaluator's retained stdout and stderr, joined.
    A capture that was TRUNCATED can hide the signature (the ring drops the head
    of the stream), which costs a false negative -- an unclassified episode that
    stays in the rate -- and never a false positive, because every marker has to
    be present in the bytes that survived.
    """

    if _ACTIVE_URL_NONE_RE.search(diagnostics_text) is None:
        return None
    if _OPEN_TAB_FALLBACK_RE.search(diagnostics_text) is None:
        return None
    timeout = _WAIT_TIMEOUT_RE.search(diagnostics_text)
    if timeout is None:
        return None
    return (
        "the evaluator could not read the active tab's URL "
        "('get_active_url_from_accessTree ... returned: None'), its open-tab "
        "fallback graded a page that did not carry the element the task is judged "
        f"on (the selector it waited for, {timeout.group(1)!r}, timed out), and the "
        "read it scored came back empty -- so the state graded was not the state "
        "the episode produced"
    )


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
        # manufactures passes as well as zeros (module docstring), and the binary
        # cannot say whether a pass is a reading -- a rule of "only a zero is
        # triaged" could never see a mis-grade that came out as a pass, and a
        # manufactured pass would leave both sides of the rate. (This corpus's
        # one apparatus-attributable episode is a zero; the honest rate is
        # 6/63 = 9.5%.)
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
