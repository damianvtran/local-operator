#!/usr/bin/env python3
"""Fail a build whose wheels would eat the PyPI project quota; keep the sdist off PyPI.

THE BUDGET, AND WHY IT EXISTS. PyPI caps a project's total storage at 10 GB, and
local-operator publishes several releases a day, every one of them kept. Measured on
the published 0.90.17, one release cost ~59 MB — the pure wheel 19.98 MB, the
``macosx_11_0_universal2`` wheel 20.14 MB, the sdist 18.85 MB — and the cap was hit:
v0.90.18 and v0.90.19 failed to upload with ``400 Project size too large``. The fix
strips comments from the wheel (``scripts/slim_wheel.py``, ~14 MiB per wheel) and moves
the sdist to the GitHub Release, so a release costs ~29 MB.

* ``BUDGET`` (18 MiB) FAILS the build. It sits above the slim target with headroom
  for normal growth, and well below the unslimmed ~19-20 MB — so a wheel that silently
  stopped being slimmed, or that picked up a stray large asset, fails here instead of
  on PyPI after a tag already exists.
* ``WARN_AT`` (16 MiB) prints a ``::warning::`` annotation: growth is creeping toward
  the budget and someone should look at what was added before it becomes a red release.

THE PR WHEEL IS LIGHTER THAN THE RELEASE WHEEL, so PR CI checks a tighter line.
``ci.yml``'s packaging-smoke wheel has no phone bundle (~0.23 MB compressed) and no
key-agent helper (~0.15 MiB, macOS wheel only), so it is ~0.37 MiB under the macOS
release wheel built from the same commit. Checked against the plain 18 MiB budget, the
first red would come from ``publish.yml``'s ``keyagent-macos`` job — after the tag
exists. ``ci.yml`` and ``release-gate.yml`` therefore pass ``--budget-mib 17.5``: the
release budget minus that overhead plus a little margin, so growth goes red on the PR
that causes it. Re-measure the overhead (and this margin) if the bundle or helper
changes size materially.

``--wheels-only`` asserts the directory holds nothing but wheels. ``release-build``
runs it on the ``release-dists`` artifact and ``keyagent-macos`` on its macOS upload
set — the two artifacts ``pypi-publish`` downloads — and ``pypi-publish`` itself
counts exactly two ``*.whl`` and no ``*.tar.gz`` in a shell check, so an sdist (or
anything else) can never ride to PyPI again by landing in that artifact by accident.
The publisher deliberately runs no repository code and has no checkout: it holds the
``id-token: write`` permission, and PyPA advises keeping that job to the upload.

Usage::

    check_dist_size.py dist/                  # every *.whl within budget
    check_dist_size.py --wheels-only dist/    # ... and nothing in dist/ but wheels
    check_dist_size.py --budget-mib 17.5 dist/  # PR CI: release budget minus overhead
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

MIB = 1024 * 1024

#: Hard ceiling per wheel. Read the module docstring before raising it: the real fix
#: for a wheel over budget is to find what grew, not to move the line.
BUDGET = 18 * MIB

#: Loud-warning threshold per wheel.
WARN_AT = 16 * MIB


def check(directory: Path, *, wheels_only: bool, budget: int = BUDGET) -> list[str]:
    """Return the problems with ``directory``; print each wheel's size as it is read."""
    problems: list[str] = []
    # A budget tightened below the warning line would make the warning unreachable.
    warn_at = min(WARN_AT, budget)
    # Dotfiles are skipped: build front-ends drop a `.gitignore` into their out-dir
    # (measured: `uv build` writes one), `actions/upload-artifact` excludes hidden files
    # by default, and the PyPI upload action only considers distribution filenames.
    files = sorted(p for p in directory.iterdir() if p.is_file() and not p.name.startswith("."))
    wheels = [p for p in files if p.suffix == ".whl"]
    if not wheels:
        problems.append(f"{directory} contains no wheel")
    for wheel in wheels:
        size = wheel.stat().st_size
        print(f"{wheel.name}: {size:,} bytes ({size / MIB:.2f} MiB, budget {budget / MIB:g} MiB)")
        if size > budget:
            problems.append(
                f"{wheel.name} is {size / MIB:.2f} MiB, over the {budget / MIB:g} MiB budget "
                "(PyPI's 10 GB project cap; see scripts/check_dist_size.py)"
            )
        elif size > warn_at:
            print(
                f"::warning::{wheel.name} is {size / MIB:.2f} MiB, past the "
                f"{warn_at / MIB:g} MiB warning line toward the {budget / MIB:g} MiB budget"
            )
    if wheels_only:
        for other in files:
            if other.suffix != ".whl":
                problems.append(
                    f"{other.name} is not a wheel; only wheels are uploaded to PyPI "
                    "(the sdist is attached to the GitHub Release instead)"
                )
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check built wheels against the size budget")
    parser.add_argument("directory", type=Path, help="directory holding the built wheel(s)")
    parser.add_argument(
        "--wheels-only",
        action="store_true",
        help="also fail if the directory holds anything but wheels (the PyPI upload dir)",
    )
    parser.add_argument(
        "--budget-mib",
        type=float,
        default=BUDGET / MIB,
        help="per-wheel ceiling in MiB (default: the release budget; PR CI passes a "
        "tighter one because its wheel lacks the release-only assets)",
    )
    args = parser.parse_args(argv)
    if not 0 < args.budget_mib <= BUDGET / MIB:
        # Tightening is the only legitimate use; raising the line is a docstring read
        # and a reviewed edit to BUDGET, never a command-line flag.
        parser.error(f"--budget-mib must be in (0, {BUDGET / MIB:g}]")
    problems = check(
        args.directory, wheels_only=args.wheels_only, budget=int(args.budget_mib * MIB)
    )
    for problem in problems:
        print(f"::error::{problem}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":  # pragma: no cover — a build tool, exercised by its test
    sys.exit(main())
