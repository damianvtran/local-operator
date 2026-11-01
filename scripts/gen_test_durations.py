"""Regenerate committed shard weights without changing test selection.

Preferred unit route (gh authentication required; no tests are executed)::

    python scripts/gen_test_durations.py --cache-dir /path/to/timing-cache \
        --ci-run 41295059480:1 --ci-run 41294499173:1 \
        --ci-run 41294469202:1

CI mode requires >=3 distinct complete successful runs of the same unit job
(Python 3.12, Linux). It binds immutable artifact IDs to attempt-specific job
intervals, includes shard-zero serial controls, and takes the inclusive p75 of
per-file totals across observed runs. Missing observations are NOT zero; sample
counts and run/head/attempt/artifact provenance are committed with the weights.
Cancelled/partial runs and ambiguous retry artifacts are rejected, not blended.
The November 2026 sample is three PR branches created within eight minutes, so
this is within-window calibration, not evidence of day-to-day stability.

JUnit testcase times are summed test-time, NOT elapsed job/pytest wall time.
Worker imbalance, idle time, setup/teardown, and collection can affect the gap;
this tool cannot attribute that gap or promise timeout headroom. See
`docs/ci-test-durations.md` for source selection and reproducibility constraints.

Legacy local/e2e ingestion remains available::

    python scripts/gen_test_durations.py --tree e2e --junit 'reports/*.xml'
    python scripts/gen_test_durations.py --runs 2 --junit 'run-a/*.xml' 'run-b/*.xml'

`--runs` preserves the historical mean mode for existing callers. Prefer CI
p75 mode for unit regeneration; do not replay the full local suite to measure
weights on this shared fleet. Unattributed testcases are fatal in either mode.
All measured weights and the p90 fallback have a one-second floor so skipped
modules never become free work. Unmeasured files still run at the fallback:
`scripts/shard_tests.py` owns selection and the shared tree registry.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import statistics
import subprocess
import sys
import xml.etree.ElementTree as ET
import zipfile
from collections import defaultdict
from io import BytesIO
from pathlib import Path
from typing import Any

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

    Empty components are dropped first, because ``Path`` cannot name one and
    does not fail loudly when asked: ``classname`` of ``""`` or ``"."`` splits
    to nothing but empty strings, and ``Path(*parts).with_suffix(".py")`` on
    the resulting ``.`` raises ``ValueError: PosixPath('.') has an empty
    name``. Guarding on the empty STRING alone leaves that whole class open --
    ``"."`` reaches the same line by the other route. A dotted module path
    never contains an empty component, so dropping them cannot change which
    file a real classname resolves to.
    """
    parts = [part for part in classname.split(".") if part]
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
            # A MODULE-level skip (`pytest.skip(..., allow_module_level=True)`)
            # is reported with an EMPTY `classname` and the module's dotted path
            # in `name` -- pytest 8, observed on CI 2026-10-19 with
            # `tests/unit/tui/test_hub_links_capture.py`. The empty string used
            # to reach `Path(*parts).with_suffix`, which raises
            # `ValueError: PosixPath('.') has an empty name` and took the whole
            # documented CI regeneration route down with it.
            source = classname or case.get("name") or ""
            path = _classname_to_path(source, repo)
            if path is None:
                # The string actually TRIED, not the raw `classname`: on the
                # empty-classname path above the two differ, and recording the
                # empty one printed `['']`, which names nothing a reader could
                # act on.
                unresolved.add(source)
                continue
            seconds = float(case.get("time") or 0.0)
            if not math.isfinite(seconds) or seconds < 0:
                raise ValueError(f"invalid duration in {xml_path}: {seconds}")
            per_file[path] += seconds
    if unresolved:
        # Loud, because a silently-dropped mapping understates a file's cost
        # and quietly unbalances the very thing this manifest exists to fix.
        raise ValueError(
            f"{len(unresolved)} classnames did not map to a file: {sorted(unresolved)}"
        )
    return dict(per_file)


def percentile_weights(runs: list[dict[str, float]]) -> dict[str, float]:
    """p75 of run-bound file totals; absent files are not zero-cost samples.

    Branches add tests independently. Keep each observed file, record its sample
    count in provenance, and let genuinely unobserved files retain the sharder
    fallback. Linear interpolation is statistics.quantiles' inclusive method.
    """
    samples: dict[str, list[float]] = defaultdict(list)
    for run in runs:
        for path, seconds in run.items():
            samples[path].append(seconds)
    return {
        path: (
            statistics.quantiles(values, n=4, method="inclusive")[2]
            if len(values) > 1
            else values[0]
        )
        for path, values in samples.items()
    }


def _gh_json(endpoint: str) -> dict[str, Any]:
    return json.loads(subprocess.check_output(["gh", "api", endpoint]))


def ci_run(run_id: int, attempt: int, cache: Path) -> tuple[dict[str, float], dict[str, Any]]:
    """Admit only an entire successful Python 3.12 shard matrix from ONE attempt.

    Artifact names historically omitted attempts. Bind each immutable artifact
    ID to its job's execution interval instead of letting gh download merge
    identically named artifacts from retries. Ambiguity is a fatal data error.
    The cache is output only: metadata and immutable archives are fetched afresh.
    """
    endpoint = f"repos/damianvtran/local-operator/actions/runs/{run_id}"
    run = _gh_json(f"{endpoint}/attempts/{attempt}")
    job_page = _gh_json(f"{endpoint}/attempts/{attempt}/jobs?per_page=100")
    jobs = job_page["jobs"]
    if job_page.get("total_count", len(jobs)) > 100:
        raise ValueError("job pagination required; refusing incomplete metadata")
    artifacts = _gh_json(f"{endpoint}/artifacts?per_page=100")
    if artifacts["total_count"] > 100:
        raise ValueError("artifact pagination required; refusing incomplete metadata")
    if run["status"] != "completed" or run["conclusion"] != "success":
        raise ValueError(f"run {run_id} attempt {attempt} did not complete successfully")
    if run["path"] != ".github/workflows/ci.yml":
        raise ValueError(f"not a comparable CI workflow: {run['path']}")
    shards = sorted(
        (j for j in jobs if j["name"].startswith("test (3.12, ")),
        key=lambda j: j["name"],
    )
    # Both the old and new matrices are legitimate future timing sources.
    if [j["name"] for j in shards] not in [
        [f"test (3.12, {s})" for s in range(n)] for n in (5, 6)
    ] or any(j["conclusion"] != "success" for j in shards):
        raise ValueError(f"run {run_id}: incomplete or unsuccessful shard matrix")
    paths: list[Path] = []
    sources = []
    for shard, job in enumerate(shards):
        name = f"junit-timings-3.12-{shard}"
        matches = [
            a
            for a in artifacts["artifacts"]
            if a["name"] in (name, f"{name}-attempt-{attempt}")
            and not a["expired"]
            and job["started_at"] <= a["created_at"] <= job["completed_at"]
        ]
        if len(matches) != 1:
            raise ValueError(f"run {run_id} shard {shard}: ambiguous/missing attempt artifact")
        artifact = matches[0]
        archive = subprocess.check_output(
            [
                "gh",
                "api",
                f"repos/damianvtran/local-operator/actions/artifacts/{artifact['id']}/zip",
            ]
        )
        directory = cache / f"{run_id}-{attempt}" / str(shard)
        directory.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(BytesIO(archive)) as zipped:
            expected = f"junit-3.12-{shard}.xml"
            names = [expected] + (["junit-serial-controls-3.12-0.xml"] if shard == 0 else [])
            for filename in names:
                # Select exact known XML members; never extract arbitrary archive paths.
                destination = directory / filename
                destination.write_bytes(zipped.read(filename))
                if not list(ET.parse(destination).getroot().iter("testcase")):
                    raise ValueError(f"empty report: {destination}")
                paths.append(destination)
        sources.append({"shard": shard, "job_id": job["id"], "artifact_id": artifact["id"]})
    totals = aggregate(paths)
    return totals, {
        "run_id": run_id,
        "attempt": attempt,
        "head": run["head_sha"],
        "python": "3.12",
        "shards": sources,
    }


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
    ap.add_argument("--junit", nargs="+", help="JUnit XML file(s) or globs")
    ap.add_argument(
        "--ci-run",
        action="append",
        default=[],
        metavar="RUN:ATTEMPT",
        help="complete CI run attempt; repeat at least three times for p75",
    )
    ap.add_argument("--cache-dir", type=Path, help="directory for downloaded timing XML")
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

    if args.ci_run:
        if args.junit or args.runs != 1 or args.tree != "unit" or not args.cache_dir:
            ap.error("--ci-run requires --cache-dir, unit tree, and no --junit/--runs")
        try:
            identities = [tuple(map(int, value.split(":"))) for value in args.ci_run]
        except ValueError:
            ap.error("--ci-run must be RUN:ATTEMPT (positive integers)")
        if any(len(value) != 2 or min(value) < 1 for value in identities):
            ap.error("--ci-run must be RUN:ATTEMPT (positive integers)")
        if len({value[0] for value in identities}) < 3 or any(len(v) != 2 for v in identities):
            ap.error("p75 requires at least three distinct complete run IDs (RUN:ATTEMPT)")
        if len({value[0] for value in identities}) != len(identities):
            ap.error("repeated run IDs would bias p75")
        observations, provenance = [], []
        for run_id, attempt in identities:
            measured, source = ci_run(run_id, attempt, args.cache_dir)
            observations.append(measured)
            provenance.append(source)
        manifest = build_manifest(percentile_weights(observations))
        manifest["_provenance"] = {
            "method": "p75-inclusive-per-run-file-total",
            "runs": provenance,
            "samples_per_file": {
                path: sum(path in run for run in observations)
                for path in sorted(set().union(*observations))
            },
        }
        out.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
        print(f"wrote {out}: p75 from {len(observations)} complete CI run attempts")
        return 0
    if not args.junit:
        ap.error("supply --junit or at least three --ci-run values")
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
