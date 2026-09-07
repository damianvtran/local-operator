"""Regenerate ``tests/durations.json``, the shard-balancing weight manifest.

Usage:
    # From JUnit XML you already have (preferred -- CI's own artifacts):
    python scripts/gen_test_durations.py --junit report.xml [more.xml ...]

    # Or measure locally, then write the manifest (slow; see the load warning):
    python -m pytest tests/unit --junit-xml=/tmp/d.xml
    python scripts/gen_test_durations.py --junit /tmp/d.xml

``scripts/shard_tests.py`` consumes the output; read its module docstring for
why the manifest is committed rather than computed at workflow time, and why a
file missing from it still runs.

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
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "tests" / "durations.json"

# The fallback weight for a file absent from the manifest, as a quantile of
# the measured files. Deliberately pessimistic (p90, not the median): an
# unknown file is assumed slow so LPT places it early and scatters unknowns
# across shards. shard_tests.py's docstring carries the numbers behind this.
FALLBACK_QUANTILE = 0.90


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
    return {
        "_comment": (
            "Per-file test durations in seconds, used ONLY as relative weights "
            "to balance the CI shards. Regenerate with "
            "scripts/gen_test_durations.py; see scripts/shard_tests.py for why "
            "a file missing from this manifest still runs."
        ),
        "fallback_seconds": round(fallback, 3),
        "durations": {k: round(v, 3) for k, v in sorted(per_file.items())},
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--junit", nargs="+", required=True, help="JUnit XML file(s) or globs")
    ap.add_argument("--out", type=Path, default=MANIFEST)
    args = ap.parse_args(argv)

    paths: list[Path] = []
    for pattern in args.junit:
        paths.extend(Path(p) for p in sorted(glob.glob(pattern)))
    if not paths:
        ap.error(f"no JUnit XML matched {args.junit}")

    per_file = aggregate(paths)
    if not per_file:
        ap.error("no testcases found in the supplied JUnit XML")

    manifest = build_manifest(per_file)
    args.out.write_text(json.dumps(manifest, indent=1, sort_keys=True) + "\n")
    total = sum(per_file.values())
    print(
        f"wrote {args.out} : {len(per_file)} files, {total / 60:.1f} test-min, "
        f"fallback {manifest['fallback_seconds']}s"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
