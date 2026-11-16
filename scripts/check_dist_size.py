#!/usr/bin/env python3
"""Fail a build whose wheels would eat the PyPI project quota; keep the sdist off PyPI.

THE BUDGET, AND WHY IT EXISTS. PyPI caps a project's total storage, and
local-operator publishes several releases a day, every one of them kept. Measured on
the published 0.90.17, one release cost ~59 MB — the pure wheel 19.98 MB, the
``macosx_11_0_universal2`` wheel 20.14 MB, the sdist 18.85 MB — and the then 10 GB cap
was hit: v0.90.18 and v0.90.19 failed to upload with ``400 Project size too large``.
The fix strips comments from the wheel (``scripts/slim_wheel.py``) and moves the sdist
to the GitHub Release. On 2026-11-04 PyPI raised the cap to 30 GiB (pypi/support#12882,
closed by JacobCoffee: "I've set the project size limit for `local-operator` to 30
GiB"; #12738 closed 11-05 as covered) — :data:`PROJECT_CAP`.

THE RUNWAY CHECK IS THE AUTHORITATIVE GATE (``--runway``). What exhausts the cap is
total bytes stored, not one wheel's size, so the real question is "how many days until
PyPI refuses an upload at the current cadence?". ``--runway`` reads the project's
JSON from PyPI and computes::

    used             = sum of every file's size, all releases, yanked ones included
                       (a yanked file still counts against the quota)
    cadence          = releases first uploaded in the last 30 days / 30   (per day)
    bytes_per_release = the wheels this release uploads: pure + macOS when both are
                       in the directory; with only the pure wheel (PR CI, and
                       release-build before keyagent-macos), pure + (pure + 0.16 MiB
                       for the key-agent helper the macOS wheel adds)
    runway_days      = (PROJECT_CAP - used) / (bytes_per_release * cadence)

Under :data:`RUNWAY_WARN_DAYS` (60) it prints a ``::warning::`` — the retention
decision is Damian's and should reach him with time to make it. Under
:data:`RUNWAY_FAIL_DAYS` (21) it exits 1 on the release path (``release-gate.yml``
before the tag, ``publish.yml`` ``release-build``), where the bytes are spent;
``ci.yml`` passes ``--runway-warn-only`` so a low runway never blocks an unrelated PR.
PyPI unreachable or its JSON malformed is a ``::warning::`` and exit 0: a PyPI outage
must never turn CI red. The projection is linear; wheel growth (~0.19 MiB/day per
wheel measured) shortens it, which is what the 60-day warning margin absorbs.

THE STATIC PER-WHEEL LINE IS A JUMP TRIPWIRE, re-derived from the cap. Measured
2026-11-16: 19,249,689,996 B remaining, 6.43 releases/day over 30 days, two wheels per
release. A per-wheel size of 23.78 MiB gives exactly 60 days of linear runway at that
cadence, so:

* ``BUDGET`` (23.5 MiB) FAILS the build: just under the 60-day size, so a wheel that
  silently stopped being slimmed, or picked up a stray large asset, fails here instead
  of on PyPI after a tag already exists.
* ``WARN_AT`` (21.5 MiB) prints a ``::warning::``: growth is creeping toward the line.

The line catches a sudden jump; gradual growth and cadence are the runway check's job.

THE PR WHEEL IS LIGHTER THAN THE RELEASE WHEEL, so PR CI checks a tighter line.
Since 2026-11-16 ``ci.yml``'s packaging-smoke wheel and ``release-gate.yml``'s wheel
build the phone bundle before ``python -m build``, exactly as ``publish.yml`` does, so
they carry the bundle and are slimmed to the release shape (``slim_wheel.py`` drops the
web build inputs once ``dist/`` is present). The remaining gap is the key-agent helper
(~0.16 MiB, macOS wheel only); before that change the PR wheel also lacked the bundle
and the gap was ~0.37 MiB. Checked against the plain 23.5 MiB budget, the first red
would come from ``publish.yml``'s ``keyagent-macos`` job — after the tag exists.
``ci.yml`` and ``release-gate.yml`` therefore pass ``--budget-mib 23.25``: the release
budget minus that overhead plus a little margin, so a jump goes red on the PR that
causes it.
Re-measure the overhead (and this margin) if the helper changes size materially.

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
    check_dist_size.py --budget-mib 23.25 dist/  # PR CI: release budget minus overhead
    check_dist_size.py --runway dist/            # release: fail under 21 days of runway
    check_dist_size.py --runway-warn-only dist/  # PR CI: warn only, never block
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

MIB = 1024 * 1024

#: PyPI's storage cap for this project: 30 GiB, set 2026-11-04 in pypi/support#12882
#: (previously the 10 GB default). Changing it needs a new PyPI decision, cited here.
PROJECT_CAP = 30 * 1024**3

#: Hard ceiling per wheel, derived from PROJECT_CAP (module docstring). Read that
#: derivation before moving it: the real fix for a wheel over budget is to find what
#: grew, not to move the line.
BUDGET = int(23.5 * MIB)

#: Loud-warning threshold per wheel.
WARN_AT = int(21.5 * MIB)

#: What the macOS wheel adds to the pure one (the signed key-agent helper), used to
#: estimate a release's bytes when only the pure wheel has been built.
KEYAGENT_OVERHEAD = int(0.16 * MIB)

PYPI_JSON_URL = "https://pypi.org/pypi/local-operator/json"
PYPI_TIMEOUT_S = 20
CADENCE_WINDOW_DAYS = 30
RUNWAY_WARN_DAYS = 60
RUNWAY_FAIL_DAYS = 21


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
                "(PyPI's 30 GiB project cap, pypi/support#12882; see scripts/check_dist_size.py)"
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


def fetch_pypi_json(url: str = PYPI_JSON_URL, timeout: float = PYPI_TIMEOUT_S) -> Any:
    """The project's PyPI JSON. Raises on any network/parse failure (caller warns)."""
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read())


def release_bytes(wheels: list[Path]) -> tuple[int, str]:
    """Bytes one release uploads, and how that figure was obtained."""
    pure = [w for w in wheels if w.name.endswith("-py3-none-any.whl")]
    mac = [w for w in wheels if "macosx" in w.name]
    if pure and mac:
        return pure[0].stat().st_size + mac[0].stat().st_size, "pure + macOS wheels, measured"
    if pure:
        size = pure[0].stat().st_size
        return 2 * size + KEYAGENT_OVERHEAD, "pure wheel x2 + 0.16 MiB key agent, estimated"
    return sum(w.stat().st_size for w in wheels), "wheels present, measured"


def runway(
    wheels: list[Path],
    *,
    fail_below: float | None,
    fetch: Callable[[], Any] = fetch_pypi_json,
    now: dt.datetime | None = None,
) -> list[str]:
    """Print the cap runway; return problems (only when under ``fail_below`` days).

    Never fails on PyPI being unreachable or its JSON being malformed: that is a
    ``::warning::`` and no problem, so a PyPI outage cannot turn CI red.
    """
    now = now or dt.datetime.now(dt.timezone.utc)
    try:
        releases = fetch()["releases"]
        used = 0
        recent = 0
        cutoff = now - dt.timedelta(days=CADENCE_WINDOW_DAYS)
        for files in releases.values():
            # Yanked files are summed too: they still occupy the project quota.
            used += sum(int(f["size"]) for f in files)
            uploaded = [
                dt.datetime.fromisoformat(f["upload_time_iso_8601"].replace("Z", "+00:00"))
                for f in files
            ]
            if uploaded and min(uploaded) >= cutoff:
                recent += 1
    # Any failure — network, HTTP, JSON, schema — is a warning, never red.
    except Exception as exc:
        print(f"::warning::PyPI runway check skipped: cannot read {PYPI_JSON_URL}: {exc}")
        return []
    cadence = recent / CADENCE_WINDOW_DAYS
    per_release, how = release_bytes(wheels)
    remaining = PROJECT_CAP - used
    print(
        f"PyPI runway: used {used:,} B of {PROJECT_CAP:,} B (30 GiB cap, pypi/support#12882); "
        f"remaining {remaining:,} B"
    )
    print(
        f"PyPI runway: cadence {cadence:.2f} releases/day ({recent} in "
        f"{CADENCE_WINDOW_DAYS} d); {per_release:,} B per release ({how})"
    )
    if cadence <= 0 or per_release <= 0:
        print("PyPI runway: no recent releases; nothing to project")
        return []
    days = remaining / (per_release * cadence)
    when = (now + dt.timedelta(days=max(days, 0))).date().isoformat()
    print(f"PyPI runway: {days:.1f} days at this rate, cap reached ~{when}")
    message = (
        f"PyPI runway is {days:.1f} days (cap reached ~{when}); a PyPI retention "
        "decision is Damian's (see scripts/check_dist_size.py and AGENTS.md)"
    )
    if fail_below is not None and days < fail_below:
        return [message]
    if days < RUNWAY_WARN_DAYS:
        print(f"::warning::{message}")
    return []


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
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--runway",
        action="store_true",
        help=f"check PyPI cap runway: warn under {RUNWAY_WARN_DAYS} days, "
        f"fail under {RUNWAY_FAIL_DAYS} (the release path)",
    )
    mode.add_argument(
        "--runway-warn-only",
        action="store_true",
        help="check PyPI cap runway but only warn, never fail (PR CI)",
    )
    args = parser.parse_args(argv)
    if not 0 < args.budget_mib <= BUDGET / MIB:
        # Tightening is the only legitimate use; raising the line is a docstring read
        # and a reviewed edit to BUDGET, never a command-line flag.
        parser.error(f"--budget-mib must be in (0, {BUDGET / MIB:g}]")
    problems = check(
        args.directory, wheels_only=args.wheels_only, budget=int(args.budget_mib * MIB)
    )
    if args.runway or args.runway_warn_only:
        wheels = sorted(args.directory.glob("*.whl")) if args.directory.is_dir() else []
        # Looked up at call time so a test can replace the fetch without the network.
        problems += runway(
            wheels,
            fail_below=RUNWAY_FAIL_DAYS if args.runway else None,
            fetch=globals()["fetch_pypi_json"],
        )
    for problem in problems:
        print(f"::error::{problem}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":  # pragma: no cover — a build tool, exercised by its test
    sys.exit(main())
