#!/usr/bin/env python3
"""Turn a directory of `xplat_probe.py` JSON runs into one comparison table.

The probes print their own matrix; this exists for the *cross-OS* reading, where
the interesting fact is never "probe X failed" but "probe X passed on macOS and
failed on Windows", or the worse one: "probe X said PASS on every OS while
doing nothing on two of them". A single-OS table cannot show either, so the
output here is deliberately a grid -- one column per OS -- rather than a list
per run.

    python scripts/xplat_report.py ~/workspace/xplat-audit/matrix

Exit code is 0 when no probe FAILed anywhere, 1 otherwise. This is also what
the container runner uses to decide its own exit status, so a FAIL in a
container cannot be swallowed by the shell that launched it.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

#: Narrower than the probe's own statuses are wide, because this view is one
#: line per OS. `WARN` is kept distinct from `PASS` on purpose: "works but with
#: a gap" is the finding most likely to be lost in a summary.
MARKERS = {"PASS": "pass", "WARN": "warn", "SKIP": "skip", "FAIL": "FAIL"}

ORDER = ("PASS", "WARN", "SKIP", "FAIL")


def load(directory: pathlib.Path) -> dict[str, dict[str, object]]:
    runs: dict[str, dict[str, object]] = {}
    for path in sorted(directory.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        host = payload.get("host", {})
        label = f"{host.get('system', path.stem)} {host.get('release', '')}".strip()
        if host.get("machine"):
            label = f"{label} {host['machine']}"
        runs[label] = payload
    return runs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", nargs="?", default=".", help="directory of probe JSON files")
    parser.add_argument(
        "--detail",
        action="store_true",
        help="also print, per OS, the failing and skipping probes with their detail line",
    )
    args = parser.parse_args(argv)

    directory = pathlib.Path(args.directory).expanduser()
    runs = load(directory)
    if not runs:
        print(f"no probe JSON found in {directory}", file=sys.stderr)
        return 1

    labels = list(runs)
    cells: dict[str, dict[str, str]] = {}
    for label, payload in runs.items():
        for result in payload.get("results", []):  # type: ignore[union-attr]
            name = str(result["name"])
            status = str(result["status"])
            cells.setdefault(name, {})[label] = status

    names = sorted(cells)
    width = max([len(n) for n in names] + [10])
    # Abbreviate the OS columns to fit a terminal; the full labels print below.
    short = {label: (label[:26] + "…" if len(label) > 27 else label) for label in labels}

    header = "probe".ljust(width) + "".join(f"  {short[label]:<28}" for label in labels)
    print(header)
    print("-" * len(header))
    for name in names:
        row = name.ljust(width)
        for label in labels:
            status = cells[name].get(label, "-")
            row += f"  {MARKERS.get(status, status):<28}"
        print(row)

    print()
    for label, payload in runs.items():
        counts = payload.get("counts", {})
        counts = counts if isinstance(counts, dict) else {}
        summary = " ".join(f"{key}={counts[key]}" for key in ORDER if key in counts)
        print(f"{label}: {summary}")

    if args.detail:
        for label, payload in runs.items():
            interesting = [
                r
                for r in payload.get("results", [])  # type: ignore[union-attr]
                if r["status"] in ("FAIL", "WARN", "SKIP")
            ]
            if not interesting:
                continue
            print(f"\n=== {label} ===")
            for result in interesting:
                print(f"  {result['status']:4} {result['name']}: {result['detail']}")

    failed = any("FAIL" in statuses.values() for statuses in cells.values())
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
