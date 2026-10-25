#!/usr/bin/env python3
"""Admit an OSWorld arm's dependency set under the reviewed override policy.

WHY THIS EXISTS. The adapter's ``[tool.uv].override-dependencies``
(``benchmarks/osworld_v2_adapter/pyproject.toml``) is the authoritative
dependency policy: it overrides two stale upstream bounds rather than
downgrading security libraries --

* OSWorld declares ``requests~=2.31.0``; the policy installs ``requests>=2.32.0``;
* ZhipuAI declares ``pyjwt>=2.8.0,<2.9.0``; the policy installs
  ``pyjwt[crypto]>=2.10,<3``.

A uv override is a RESOLVER rule. It is not written into the installed wheels'
metadata, so a raw ``uv pip check`` against a correctly built arm still exits 1
and names exactly those two bounds. That raw result is retained verbatim and is
NOT green; this module answers a different, narrower question: "are the raw
checker's findings exactly the two the policy knowingly supersedes, with
installed versions inside the policy, and nothing else?" (see
``docs/benchmarks/osworld_2/dependency-policy.md`` and ``arm-build.md``).

WHAT IS REFUSED, every one fail-closed:

* any third conflict, however plausible -- a new conflict is a new policy
  decision, not something an admission gate may wave through;
* any missing dependency (``... but it's not installed``) -- main16 was refused
  here for a missing ``miniaudio``; ``--no-deps`` must never paper over that;
* a superseded bound whose upstream specifier changed, or that disappeared --
  the policy was reviewed against those exact bytes;
* a "green" raw report (rc 0 / "All installed packages are compatible"): the
  two bounds are real metadata in the installed SDKs, so green means either the
  upstream metadata changed (re-review the policy) or the text is fabricated;
* any line the parser does not recognise, a count line that disagrees with
  the findings, or an rc other than 1;
* a committed override list that no longer covers exactly the superseded
  dependencies, or an installed version outside an override specifier;
* a core harness requirement that is absent or unsatisfied.

WHY THE CHECKER IS NOT REIMPLEMENTED. uv owns whole-environment marker
semantics. A ``packaging``-based re-check of every installed distribution was
tried and disagreed with uv on macOS umbrella markers (pyobjc's
``platform_release >= '9.0'``), inventing findings. So the raw uv report is the
authority for the WHOLE environment; this module audits every line of it, then
confirms the two admitted findings independently from the installed metadata.
Only the harness's own direct requirements are evaluated here, and only with
the arm interpreter's own marker environment.

PURE CORE, THIN SHELL. :func:`parse_check_report` and :func:`admit` take text and
mappings and touch nothing, so the unit tests exercise the real decision. The
CLI below gathers those inputs and MUST run under the arm interpreter itself
(``"$VENV/bin/python3.12" -I scripts/osworld_dependency_admission.py ...``):
the installed metadata and marker environment that matter are that
interpreter's. For the same reason this module imports only the standard
library and ``packaging`` (a harness dependency, so present in every arm) --
never ``local_operator`` and never a sibling script, because ``-I`` keeps the
script directory off ``sys.path``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import re
import subprocess
import sys
import tomllib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from packaging.markers import default_environment
from packaging.requirements import InvalidRequirement, Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version


@dataclass(frozen=True)
class SupersededBound:
    """One upstream requirement the reviewed policy knowingly overrides."""

    parent: str
    dependency: str
    upstream: str


# The reviewed set, by exact upstream specifier. Changing it is a policy change:
# it needs the compatibility evidence dependency-policy.md describes (the JWT
# signing probe for PyJWT), not just an edit here. Names are canonical.
SUPERSEDED_BOUNDS: frozenset[SupersededBound] = frozenset(
    {
        SupersededBound(parent="osworld", dependency="requests", upstream="~=2.31.0"),
        SupersededBound(parent="zhipuai", dependency="pyjwt", upstream=">=2.8.0,<2.9.0"),
    }
)


@dataclass(frozen=True)
class ReviewedOverride:
    """One committed override, in full, as the policy was reviewed.

    The dependency NAME alone is not the policy: ``requests>=2.32.0`` is a
    security floor, so a gate that compared names only would wave through a
    weakened ``requests>=1``. The specifier set and extras are compared too —
    ``pyjwt`` without ``crypto`` would drop the extra the harness floor needs.
    """

    dependency: str
    specifiers: SpecifierSet
    extras: frozenset[str]


REVIEWED_OVERRIDES: frozenset[ReviewedOverride] = frozenset(
    {
        ReviewedOverride("requests", SpecifierSet(">=2.32.0"), frozenset()),
        ReviewedOverride("pyjwt", SpecifierSet(">=2.10,<3"), frozenset({"crypto"})),
    }
)

# uv 0.10's ``pip check`` wording. Matched whole-line: a prefix match would let
# a decorated or truncated line pass as a recognised finding.
_CONFLICT = re.compile(r"The package `([^`]+)` requires `([^`]+)`, but `([^`]+)` is installed")
_MISSING = re.compile(r"The package `([^`]+)` requires `([^`]+)`, but it's not installed")
_COUNT = re.compile(r"Found (\d+) incompatibilit(?:y|ies)")
_GREEN = "All installed packages are compatible"
_NOISE = (
    re.compile(r"Using Python \S+ environment at: .+"),
    re.compile(r"Checked \d+ packages? in \S+"),
)


@dataclass(frozen=True)
class Finding:
    """One parsed checker line. ``installed`` is None for a missing dependency."""

    parent: str
    requirement: str
    installed: str | None


@dataclass
class CheckReport:
    conflicts: list[Finding] = field(default_factory=list)
    missing: list[Finding] = field(default_factory=list)
    unrecognized: list[str] = field(default_factory=list)
    count: int | None = None
    green_line: bool = False


@dataclass(frozen=True)
class Admission:
    """The verdict. ``plain_pip_check_green`` is reported, never rewritten."""

    admitted: bool
    refusals: tuple[str, ...]
    raw_returncode: int
    plain_pip_check_green: bool
    admitted_bounds: tuple[SupersededBound, ...]

    def as_receipt(self) -> dict[str, Any]:
        return {
            "policy_conformant_dependency_admission": self.admitted,
            "plain_pip_check_green": self.plain_pip_check_green,
            "raw_pip_check_rc": self.raw_returncode,
            "admitted_superseded_bounds": [
                [bound.parent, bound.dependency, bound.upstream]
                for bound in sorted(self.admitted_bounds, key=lambda b: b.parent)
            ],
            "refusals": list(self.refusals),
        }


def parse_check_report(text: str) -> CheckReport:
    """Classify EVERY non-blank line of a raw ``uv pip check`` report."""

    report = CheckReport()
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if match := _CONFLICT.fullmatch(line):
            report.conflicts.append(Finding(*match.groups()))
        elif match := _MISSING.fullmatch(line):
            report.missing.append(Finding(match.group(1), match.group(2), None))
        elif match := _COUNT.fullmatch(line):
            report.count = int(match.group(1))
        elif line == _GREEN:
            report.green_line = True
        elif not any(pattern.fullmatch(line) for pattern in _NOISE):
            report.unrecognized.append(line)
    return report


def _requirement(text: str) -> Requirement | None:
    try:
        return Requirement(text)
    except InvalidRequirement:
        return None


def _key(parent: str, dependency: str, specifier: str) -> tuple[str, str, SpecifierSet]:
    # SpecifierSet equality is order-insensitive, so ">=2.8.0,<2.9.0" and the
    # checker's "<2.9.0,>=2.8.0" are the same bound, while "<2.10" is not.
    return canonicalize_name(parent), canonicalize_name(dependency), SpecifierSet(specifier)


def _version(text: str | None) -> Version | None:
    if text is None:
        return None
    try:
        return Version(text)
    except InvalidVersion:
        return None


def check_policy(
    overrides: Sequence[str],
    bounds: frozenset[SupersededBound] = SUPERSEDED_BOUNDS,
    reviewed: frozenset[ReviewedOverride] = REVIEWED_OVERRIDES,
) -> list[str]:
    """Refusals if the committed overrides are not exactly the reviewed ones.

    The override list is read from the adapter pyproject at the built commit, so
    an edit there that this module was not reviewed against stops the build
    before anything is installed — including an edit that keeps the same two
    package names but moves a security floor down.
    """

    refusals: list[str] = []
    parsed = [_requirement(text) for text in overrides]
    if any(item is None for item in parsed):
        refusals.append(f"override-dependencies contains an invalid requirement: {overrides!r}")
        return refusals
    committed = frozenset(
        ReviewedOverride(canonicalize_name(item.name), item.specifier, frozenset(item.extras))
        for item in parsed
        if item is not None
    )
    if committed != reviewed:

        def describe(rows: Iterable[ReviewedOverride]) -> list[str]:
            return sorted(
                f"{row.dependency}{'[' + ','.join(sorted(row.extras)) + ']' if row.extras else ''}"
                f"{row.specifiers or ''}"
                for row in rows
            )

        refusals.append(
            f"override-dependencies is {describe(committed)}, but the reviewed policy is "
            f"{describe(reviewed)}; a policy change needs review, not an admission edit"
        )
    unbounded = sorted({row.dependency for row in reviewed} - {b.dependency for b in bounds})
    if unbounded:  # pragma: no cover - editing one set without the other fails here
        refusals.append(f"reviewed overrides name {unbounded} with no reviewed upstream bound")
    return refusals


def admit(
    report_text: str,
    returncode: int,
    *,
    overrides: Sequence[str],
    installed: Mapping[str, str],
    declared: Mapping[str, Sequence[str]],
    core_requirements: Sequence[str] = (),
    marker_environment: Mapping[str, str] | None = None,
    bounds: frozenset[SupersededBound] = SUPERSEDED_BOUNDS,
) -> Admission:
    """Decide admission from the raw report plus installed facts. Pure.

    ``installed`` maps canonical name -> installed version for the arm.
    ``declared`` maps each superseded bound's parent -> its installed
    ``Requires-Dist`` strings, the independent confirmation that the bound the
    checker reported is the one really present in the installed metadata.
    ``core_requirements`` are the harness's direct ``[project].dependencies``
    at the built commit, evaluated against ``marker_environment`` (the arm
    interpreter's, never the caller's).
    """

    refusals: list[str] = []
    installed = {canonicalize_name(name): version for name, version in installed.items()}
    report = parse_check_report(report_text)

    # The raw checker: rc 1 is the only honest outcome for a correct arm.
    if returncode == 0 or report.green_line:
        refusals.append(
            "raw uv pip check reports green, but the superseded upstream bounds are real "
            "installed metadata; a green report is either fabricated or a changed upstream "
            "that needs policy review"
        )
    elif returncode != 1:
        refusals.append(f"raw uv pip check exited {returncode}, not a findings report (1)")
    for line in report.unrecognized:
        refusals.append(f"unrecognised checker output: {line!r}")
    for finding in report.missing:
        refusals.append(
            f"missing dependency: {finding.parent} requires {finding.requirement}, not installed"
        )
    total = len(report.conflicts) + len(report.missing)
    if report.count is not None and report.count != total:
        refusals.append(f"checker counted {report.count} findings but listed {total}")
    if report.count is None and total:
        refusals.append("checker listed findings without its 'Found N' count line")

    expected = {_key(b.parent, b.dependency, b.upstream): b for b in bounds}
    seen: set[tuple[str, str, SpecifierSet]] = set()
    for finding in report.conflicts:
        requirement = _requirement(finding.requirement)
        if requirement is None:
            refusals.append(f"unparseable conflict requirement: {finding.requirement!r}")
            continue
        key = _key(finding.parent, requirement.name, str(requirement.specifier))
        if key not in expected:
            refusals.append(
                f"unadmitted conflict: {finding.parent} requires {finding.requirement}, "
                f"{finding.installed} installed"
            )
            continue
        seen.add(key)
    for key, bound in expected.items():
        if key not in seen:
            refusals.append(
                f"superseded bound absent from the raw report: {bound.parent} requires "
                f"{bound.dependency}{bound.upstream} (upstream changed, or report incomplete)"
            )

    # Independent confirmation from installed metadata, not from the report.
    for bound in bounds:
        rows = [_requirement(text) for text in declared.get(bound.parent, ())]
        specifiers = [
            row.specifier
            for row in rows
            if row is not None and canonicalize_name(row.name) == bound.dependency
        ]
        if SpecifierSet(bound.upstream) not in specifiers:
            refusals.append(
                f"installed {bound.parent} metadata does not declare "
                f"{bound.dependency}{bound.upstream} (found {[str(s) for s in specifiers]})"
            )

    refusals.extend(check_policy(overrides, bounds))
    for text in overrides:
        requirement = _requirement(text)
        if requirement is None:
            continue
        name = canonicalize_name(requirement.name)
        version = _version(installed.get(name))
        if version is None or not requirement.specifier.contains(version, prereleases=True):
            refusals.append(f"installed {name} {installed.get(name)} is outside override {text}")

    environment = dict(marker_environment or {})
    for text in core_requirements:
        requirement = _requirement(text)
        if requirement is None:
            refusals.append(f"unparseable core requirement: {text!r}")
            continue
        if requirement.marker is not None and not requirement.marker.evaluate(
            {**environment, "extra": ""}
        ):
            continue
        name = canonicalize_name(requirement.name)
        version = _version(installed.get(name))
        if version is None:
            refusals.append(f"core requirement {text} is not installed")
        elif not requirement.specifier.contains(version, prereleases=True):
            refusals.append(f"core requirement {text} unsatisfied by installed {version}")

    return Admission(
        admitted=not refusals,
        refusals=tuple(refusals),
        raw_returncode=returncode,
        plain_pip_check_green=returncode == 0,
        admitted_bounds=tuple(bounds) if not refusals else (),
    )


def _installed_facts() -> tuple[dict[str, str], dict[str, list[str]]]:
    """Versions and superseded parents' Requires-Dist of THIS interpreter."""

    installed: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata["Name"]
        if name:
            installed[canonicalize_name(name)] = distribution.version
    declared: dict[str, list[str]] = {}
    for bound in SUPERSEDED_BOUNDS:
        try:
            declared[bound.parent] = list(importlib.metadata.requires(bound.parent) or [])
        except importlib.metadata.PackageNotFoundError:
            declared[bound.parent] = []
    return installed, declared


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--adapter-pyproject", required=True, type=Path)
    parser.add_argument("--core-pyproject", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path, help="receipt directory (must exist)")
    args = parser.parse_args(argv)

    adapter = tomllib.loads(args.adapter_pyproject.read_text())
    core = tomllib.loads(args.core_pyproject.read_text())
    # The checker is always pointed at the interpreter running this file: the
    # installed facts below are read in-process, so both halves describe one env.
    check = subprocess.run(
        ["uv", "pip", "check", "--python", sys.executable],
        capture_output=True,
        text=True,
        check=False,
    )
    text = check.stdout + check.stderr
    # Raw output first, before any verdict: a refusal must leave the evidence.
    (args.out / "raw-pip-check.log").write_text(text)
    (args.out / "raw-pip-check.rc").write_text(f"{check.returncode}\n")
    installed, declared = _installed_facts()
    verdict = admit(
        text,
        check.returncode,
        overrides=adapter["tool"]["uv"]["override-dependencies"],
        installed=installed,
        declared=declared,
        core_requirements=core["project"]["dependencies"],
        marker_environment={str(key): str(value) for key, value in default_environment().items()},
    )
    receipt = verdict.as_receipt()
    receipt["raw_pip_check_sha256"] = hashlib.sha256(text.encode()).hexdigest()
    receipt["interpreter"] = sys.executable
    (args.out / "policy-admission.json").write_text(json.dumps(receipt, indent=2) + "\n")
    if not verdict.admitted:
        for reason in verdict.refusals:
            print(f"REFUSED: {reason}", file=sys.stderr)
        return 1
    print("ADMITTED: policy-conformant; raw uv pip check rc=1 retained, NOT green")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
