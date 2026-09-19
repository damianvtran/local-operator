"""Regenerate a shard-balancing weight manifest (``tests/durations[-x].json``).

Usage:
    # Measure locally, then write the manifest. This is the path that works
    # today, and how tests/durations.json was produced (slow; see the load
    # warning below before starting one).
    python -m pytest tests/unit --junit-xml=/tmp/d.xml
    python scripts/gen_test_durations.py --junit /tmp/d.xml

    # The same tool writes every tree's manifest; `--tree` picks which, and
    # defaults the output path to that tree's own manifest file.
    python -m pytest tests/e2e -m e2e -n0 --junit-xml=/tmp/e2e.xml
    python scripts/gen_test_durations.py --tree e2e --junit /tmp/e2e.xml

    # From JUnit XML you already have, possibly from CI -- its runs are on
    # dedicated hardware. This route needs the workflow to emit the XML, which
    # the shard jobs do (a `--junit-xml` + artifact upload per shard), so the
    # artifacts are a legitimate source as of that step landing. Before it did,
    # this docstring told the reader not to treat CI as an available option for
    # exactly that reason.
    #
    # The artifact names are the ones the upload steps declare, and they are
    # version-matrixed on the unit tree:
    #   unit: junit-timings-<python>-<shard>      (junit-3.12-4.xml)
    #   e2e:  junit-timings-e2e-<os>-<shard>      (junit-e2e-ubuntu-latest-2.xml)
    # e.g.
    #   for s in 0 1 2 3 4; do gh run download <run> -n junit-timings-3.12-$s -D u/; done
    #   python scripts/gen_test_durations.py --tree unit --junit 'u/*.xml'
    python scripts/gen_test_durations.py --junit report.xml [more.xml ...]

    # Aggregate a whole run's shard artifacts into the unit manifest:
    python scripts/gen_test_durations.py --junit 'unit-xml/*.xml'

    # Prefer TWO OR MORE runs. The manifest is an average when `--runs` says how
    # many complete runs the inputs cover, and one run is not enough: a single
    # run fitted perfectly in-sample and landed ~1.7x out of sample, while the
    # average of two scored 1.20x on both (measured 2026-09-19).
    python scripts/gen_test_durations.py --runs 2 \
        --junit 'run-a/*.xml' 'run-b/*.xml'

``scripts/shard_tests.py`` consumes the output; read its module docstring for
why the manifest is committed rather than computed at workflow time, and why a
file missing from it still runs. The two scripts share ONE registry of the
shardable trees (``scripts/shard_tests.py::TREES``), imported here rather than
restated: a second copy of "which manifest belongs to which tree" is how a
manifest gets written for the wrong tree and silently degrades every shard to
the fallback weight.

WHAT THIS MEASURES, AND WHY LOCAL NUMBERS ARE LEGITIMATE
--------------------------------------------------------
Only RELATIVE weights matter -- the partition never uses absolute seconds, so
a manifest measured on a slower (or busier) machine than CI is still correct
as long as the ORDERING survives. That is not assumed here, it was tested:
weights measured locally, projected onto CI's then-current positional split,
reproduced the five real CI shard wall times in exact rank order (5/5) at
r = 0.985 against run 34140443526.

That cross-check is the acceptance test for a regenerated manifest, and it is
worth repeating rather than trusting, because the failure it guards against is
subtle: the suite mixes I/O-bound Textual pilot tests (which wait on an event
loop) with CPU-bound ones, so a loaded machine could in principle distort them
differently rather than uniformly. It did not, but "did not last time" is not
"cannot".

``time`` in JUnit XML is the CALL phase only. Setup and teardown are excluded,
which understates fixture-heavy files slightly; it is uniform enough not to
matter for a partition and is noted so nobody re-derives the discrepancy.

CAUTION: this repo is worked through many concurrent worktrees, and a full
unit run is expensive. Do not launch one on a loaded box, and do not pass an
explicit ``-n`` -- the root ``conftest.py`` sizes workers deliberately and an
explicit count bypasses that cap. See AGENTS.md.

WHEN TO REGENERATE
------------------
There is no schedule, and that is intentional -- a manifest nobody regenerates
is worse than none only if staleness breaks something, and here it cannot:
unmeasured files still run, at a pessimistic weight. Regenerate when CI shard
times visibly diverge (the shard job prints its unmeasured-file count and
projected minutes on every run), or after adding or removing a genuinely slow
area of the suite.

**Staleness does not stay marginal, and 2026-09-19 is the measurement that
says so.** The unit manifest held 544 of the 722 files the shard job collects
(178 unmeasured, 24.6%), and those unknowns were weighted at a guessed 22.4 s
each -- 38% of every shard's projected load. The partition therefore reported a
perfect 34.8 test-min for all five shards while the real per-shard pytest times
on run 35419790955 were 699 s / 785 s / 658 s / 780 s / 634 s. That is the
state this tool exists to fix, and the way to notice it is to compare the
projected line against the pytest summary line in the SAME job log.

**Both manifests now come from CI, and how they were fitted is worth
keeping.** `tests/durations.json` and `tests/durations-e2e.json` were
regenerated from the `junit-timings-*` artifacts of two runs of the PR that
added the second tree and the artifact -- run A `35419790955` and run B
`35420676112` -- so the weights describe the hardware that runs the suite.
Two decisions came out of that, both from measurement:

- **The unit manifest is the AVERAGE of two runs** (`--runs 2`), because one
  run is not enough. Scoring each candidate partition on the measured per-file
  times of the same three runs (one tree, one method -- see the note below):
  fitting on A put 1.000x on A and 1.677x on B; fitting on B put 1.377x on A
  and 1.000x on B; the average puts 1.189x on A and 1.200x on B. The in-sample
  1.000x is the dangerous number here -- it says the partitioner did its job,
  not that the weights are good. The cause is that a handful of files carry a
  quarter of the tree and they swing hard between runs:
  `tests/unit/tui/test_settings_view.py` measured 1042.9 s in A and 620.4 s in
  B (0.59x), `test_ask_picker.py` 578.8 s then 901.6 s (1.56x), against a
  median per-file ratio of 0.99 (p10 0.67, p90 1.30) over the 222 files above
  5 s.

  **Score every row on the same tree, or the table means nothing.** An earlier
  revision of this note quoted 1.086x / 1.521x for main's pre-PR manifest and
  1.196x / 1.205x for the average: those came from different trees (before and
  after a rebase onto a main that had added and removed test files), so they
  were not comparable. Recomputed on one tree, main's pre-PR manifest scores
  1.275x (A) / 1.399x (B) / 1.264x (C) and the average scores 1.189x / 1.200x /
  1.310x -- the average is the best worst-case row, but the C column is inside
  the noise, so the honest claim is that the regeneration buys MEASURED
  COVERAGE (no file left at a guessed weight) and a truthful projection, not a
  demonstrably smaller spread on any one run.
- **The e2e manifest is a single run (A, ubuntu legs only)**, because that tree
  does not have the problem: its per-file run-to-run ratios sit at 0.96-1.01
  for every file above 20 s (its cost is fixed waits, not work), and the
  two-run average actually scored WORSE on the macOS legs (1.17x against
  1.08x). Ubuntu-only keeps the printed `~N projected test-min` comparable to
  one leg's pytest summary line; macOS ranks the same files the same way.

What a regeneration CANNOT fix is worth as much as what it can:

- **The partitioner is exact on the weights it is given, and that is a weak
  claim.** The stale manifest's split scored 1.275x-1.399x across three runs;
  the current one scores 1.189x / 1.200x / 1.310x on the same three. Most of
  the wall spread people will see is NOT weight error: two runs of ONE manifest
  over a near-identical tree gave shard walls 634-785 s (1.24x) and 461-846 s
  (1.84x), with the slowest shard of one the fastest of the other, and their
  totals within 0.4% of each other (12451 s vs 12401 s). More shards divide
  that arithmetic without touching the variance; more runs are what shrink it.
- **Test COUNT is not a signal.** A shard with the most tests (5614) had one of
  the shortest walls, so a per-test setup/teardown term -- the phase JUnit
  omits -- would move weight to the wrong shard. Do not add that term without a
  measurement showing it earns its place.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

# The shardable-tree registry lives with the partitioner that consumes it, so
# the two scripts cannot disagree about which manifest belongs to which tree.
# `sys.path` needs the repository root first: run the documented way
# (`python scripts/gen_test_durations.py`), Python puts `scripts/` on the path,
# not the root, so a sibling-module import resolves only after this insert.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.shard_tests import TREES  # noqa: E402  (needs the path fix above)

REPO = Path(__file__).resolve().parents[1]
MANIFEST = TREES["unit"].manifest

# The fallback weight for a file absent from the manifest, as a quantile of
# the measured files. Deliberately pessimistic (p90, not the median): an
# unknown file is assumed slow so LPT places it early and scatters unknowns
# across shards. shard_tests.py's docstring carries the numbers behind this.
FALLBACK_QUANTILE = 0.90

#: The smallest weight this tool will write, in seconds. A file whose tests all
#: skipped measures 0.0 in JUnit, and a zero weight is WORSE than an absent
#: one: absent files get `fallback_seconds`, whereas a zero tells the
#: partitioner the file is free, so LPT hands it to whichever shard is up next
#: and its import/collection cost lands somewhere untracked. The floor keeps
#: every committed weight strictly positive, which
#: `tests/unit/test_ci_hygiene.py` asserts rather than trusts.
MIN_WEIGHT_SECONDS = 1.0


def _classname_to_path(classname: str, repo: Path) -> str | None:
    """Map a JUnit ``classname`` back to its test file.

    pytest emits a dotted module path, optionally suffixed with a test CLASS
    name (``tests.unit.x.test_y.TestThing``). There is no ``file`` attribute
    to rely on across pytest versions, so trailing components are trimmed
    until the remainder names a file that exists.
    """
    parts = classname.split(".")
    while parts:
        candidate = Path(*parts).with_suffix(".py")
        if (repo / candidate).is_file():
            return candidate.as_posix()
        parts = parts[:-1]
    return None


def aggregate(xml_paths: list[Path], repo: Path = REPO) -> dict[str, float]:
    per_file: dict[str, float] = defaultdict(float)
    unresolved: set[str] = set()
    for xml_path in xml_paths:
        root = ET.parse(xml_path).getroot()
        for case in root.iter("testcase"):
            classname = case.get("classname") or ""
            path = _classname_to_path(classname, repo)
            if path is None:
                unresolved.add(classname)
                continue
            per_file[path] += float(case.get("time") or 0.0)
    if unresolved:
        # Loud, because a silently-dropped mapping understates a file's cost
        # and quietly unbalances the very thing this manifest exists to fix.
        print(
            f"warning: {len(unresolved)} classnames did not map to a file: "
            f"{sorted(unresolved)[:3]}",
            file=sys.stderr,
        )
    return dict(per_file)


def build_manifest(per_file: dict[str, float]) -> dict[str, object]:
    values = sorted(per_file.values())
    if values:
        index = min(len(values) - 1, int(FALLBACK_QUANTILE * len(values)))
        fallback = values[index]
    else:
        fallback = 50.0
    # Floored exactly like the per-file weights, and for the same reason: JUnit
    # reports 0.0 for a file whose tests all skipped, so a small or single-file
    # input can put the p90 quantile AT zero. A zero fallback is worse than no
    # fallback at all -- an unmeasured file then weighs nothing, LPT hands it to
    # whichever shard is up next, and the unknowns pile onto one shard (the
    # first) instead of being scattered. Latent with today's tree, wrong the
    # moment a regeneration runs over a small or all-skipped input.
    fallback = max(fallback, MIN_WEIGHT_SECONDS)
    return {
        "_comment": (
            "Per-file test durations in seconds, used ONLY as relative weights "
            "to balance the CI shards. Regenerate with "
            "scripts/gen_test_durations.py; see scripts/shard_tests.py for why "
            "a file missing from this manifest still runs."
        ),
        "fallback_seconds": round(fallback, 3),
        "durations": {k: round(max(v, MIN_WEIGHT_SECONDS), 3) for k, v in sorted(per_file.items())},
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--junit", nargs="+", required=True, help="JUnit XML file(s) or globs")
    ap.add_argument(
        "--runs",
        type=int,
        default=1,
        help=(
            "how many complete runs the inputs cover; the summed per-file times "
            "are divided by this, so the manifest stays in one run's seconds "
            "while being an average across runs (default: %(default)s)"
        ),
    )
    ap.add_argument(
        "--tree",
        choices=sorted(TREES),
        default="unit",
        help="which tree's manifest to write (default: %(default)s)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="write the manifest here (default: the tree's committed manifest)",
    )
    args = ap.parse_args(argv)

    tree = TREES[args.tree]
    out = args.out if args.out is not None else tree.manifest

    paths: list[Path] = []
    for pattern in args.junit:
        paths.extend(Path(p) for p in sorted(glob.glob(pattern)))
    if not paths:
        ap.error(f"no JUnit XML matched {args.junit}")

    per_file = aggregate(paths)
    if not per_file:
        ap.error("no testcases found in the supplied JUnit XML")

    # Averaging across runs is the point of `--runs`: a single run's per-file
    # numbers carry enough noise to make a partition look perfect in-sample and
    # land ~1.7x out of sample (measured 2026-09-19 -- see WHEN TO REGENERATE),
    # and the average of two runs scored 1.20x on BOTH. The division is what
    # keeps the printed `~N projected test-min` comparable to one run's pytest
    # summary line, which is the number a reader checks it against.
    if args.runs < 1:
        ap.error("--runs must be at least 1")
    if args.runs > 1:
        per_file = {k: v / args.runs for k, v in per_file.items()}

    # Refuse a mismatched XML rather than writing it. Every weight in a
    # manifest whose paths no file matches is dead weight: the partitioner
    # finds 0 of N files measured and schedules the whole tree at the fallback
    # weight, which LOOKS balanced (equal fallback everywhere) and silently
    # discards every real measurement in the file. Feeding the unit XML to
    # `--tree e2e` is the one-argument mistake that does it.
    stray = sorted(p for p in per_file if not p.startswith(tree.root + "/"))
    if stray:
        ap.error(
            f"--tree {tree.name} expects paths under {tree.root}/, but the JUnit "
            f"XML resolved {len(stray)} file(s) outside it, e.g. {stray[:3]}; "
            "this looks like another tree's report"
        )

    manifest = build_manifest(per_file)
    out.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    total = sum(per_file.values())
    print(
        f"wrote {out} for tree '{tree.name}': {len(per_file)} files, "
        f"{total / 60:.1f} test-min, fallback {manifest['fallback_seconds']}s"
        + (f" (averaged over {args.runs} runs)" if args.runs > 1 else "")
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
