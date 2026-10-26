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
* a conflict that merely LOOKS like an admitted one. A finding is identified by
  parent, dependency, specifier, EXTRAS and MARKER; each expected finding must
  appear EXACTLY ONCE, with no extras and no marker. Otherwise
  ``requests[socks]~=2.31.0``, ``requests~=2.31.0 ; python_full_version >=
  '3.8'`` or a duplicated line would pass as one of the two reviewed bounds.
  The extras case is invisible to uv's own checker -- it reports nothing when
  the version constraint is satisfied, even with the extra's package absent --
  so it would slip through as a new upstream requirement;
* any missing dependency (``... but it's not installed``) -- main16 was refused
  here for a missing ``miniaudio``; ``--no-deps`` must never paper over that;
* a superseded bound whose specifier, extras or marker changed, or that
  disappeared. The installed ``Requires-Dist`` metadata is audited separately
  from the report, so a report cannot vouch for itself;
* a "green" raw report (rc 0 / "All installed packages are compatible"): the
  two bounds are real metadata in the installed SDKs, so green means either the
  upstream metadata changed (re-review the policy) or the text is fabricated;
* any line the parser does not recognise, a count line that disagrees with the
  findings, or an rc other than 1;
* a committed override list that is not exactly the reviewed one -- including
  an override disabled by a marker, a weakened security floor, or a dropped
  extra;
* a harness requirement that is absent or unsatisfied;
* a report that describes a DIFFERENT environment (its ``environment at:`` path
  is not this interpreter's prefix), or a finding whose stated installed version
  disagrees with the installed metadata: the report must be about this arm, not
  merely well-formed.

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

Exit codes: 0 admitted, 1 refused, 2 usage/precondition. The raw checker text
and rc are written before any verdict in every case.
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
    weakened ``requests>=1``. Specifier, extras and marker are compared too --
    ``pyjwt`` without ``crypto`` drops the extra the harness floor needs, and a
    marker (``; python_version < '3'``) can disable the override entirely.
    """

    dependency: str
    specifiers: SpecifierSet
    extras: frozenset[str]
    marker: str | None


REVIEWED_OVERRIDES: frozenset[ReviewedOverride] = frozenset(
    {
        ReviewedOverride("requests", SpecifierSet(">=2.32.0"), frozenset(), None),
        ReviewedOverride("pyjwt", SpecifierSet(">=2.10,<3"), frozenset({"crypto"}), None),
    }
)

# uv 0.10's ``pip check`` wording. Matched whole-line: a prefix match would let a
# decorated or truncated line pass as a recognised finding.
_CONFLICT = re.compile(r"The package `([^`]+)` requires `([^`]+)`, but `([^`]+)` is installed")
_MISSING = re.compile(r"The package `([^`]+)` requires `([^`]+)`, but it's not installed")
_COUNT = re.compile(r"Found (\d+) incompatibilit(?:y|ies)")
_ENVIRONMENT = re.compile(r"Using Python (\S+) environment at: (.+)")
_CHECKED = re.compile(r"Checked (\d+) packages? in \S+")
_GREEN = "All installed packages are compatible"


@dataclass(frozen=True)
class Finding:
    """One parsed checker line, with everything that makes it that requirement.

    ``extras`` and ``marker`` are part of the identity on purpose: without them
    a differently-qualified requirement collapses onto an admitted one.
    """

    package: str
    name: str
    specifier: str
    extras: tuple[str, ...]
    marker: str | None
    reported_install: str | None = None

    @property
    def key(self) -> tuple[str, str, str, tuple[str, ...], str | None]:
        return (
            canonicalize_name(self.package),
            self.name,
            self.specifier,
            self.extras,
            self.marker,
        )


@dataclass
class CheckReport:
    conflicts: list[Finding] = field(default_factory=list)
    missing: list[Finding] = field(default_factory=list)
    unrecognized: list[str] = field(default_factory=list)
    count: int | None = None
    green_line: bool = False
    python_version: str | None = None
    environment: str | None = None
    checked: int | None = None


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


def _requirement(text: str) -> Requirement | None:
    try:
        return Requirement(text)
    except InvalidRequirement:
        return None


def _finding(package: str, requirement_text: str, reported_install: str | None = None) -> Finding:
    """Parse one finding. An unparseable requirement keeps its raw text."""

    requirement = _requirement(requirement_text)
    if requirement is None:
        return Finding(package, requirement_text, "", (), None, reported_install)
    return Finding(
        package=package,
        name=canonicalize_name(requirement.name),
        specifier=str(requirement.specifier),
        extras=tuple(sorted(canonicalize_name(extra) for extra in requirement.extras)),
        marker=str(requirement.marker) if requirement.marker is not None else None,
        reported_install=reported_install,
    )


def expected_finding(bound: SupersededBound) -> Finding:
    """The finding an admitted bound MUST appear as: no extras, no marker.

    Derived by parsing rather than hand-written so that a policy edit (say,
    adding a marker to ``upstream``) cannot silently keep matching.
    """

    finding = _finding(bound.parent, f"{bound.dependency}{bound.upstream}")
    assert finding.specifier, f"reviewed bound {bound} has no specifier"
    return finding


def parse_check_report(text: str) -> CheckReport:
    """Classify EVERY non-blank line of a raw ``uv pip check`` report."""

    report = CheckReport()
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if match := _CONFLICT.fullmatch(line):
            report.conflicts.append(_finding(*match.groups()))
        elif match := _MISSING.fullmatch(line):
            report.missing.append(_finding(*match.groups()))
        elif match := _COUNT.fullmatch(line):
            report.count = int(match.group(1))
        elif match := _ENVIRONMENT.fullmatch(line):
            report.python_version, report.environment = match.group(1), match.group(2)
        elif match := _CHECKED.fullmatch(line):
            report.checked = int(match.group(1))
        elif line == _GREEN:
            report.green_line = True
        else:
            report.unrecognized.append(line)
    return report


def check_policy(
    overrides: Sequence[str],
    bounds: frozenset[SupersededBound] = SUPERSEDED_BOUNDS,
    reviewed: frozenset[ReviewedOverride] = REVIEWED_OVERRIDES,
) -> list[str]:
    """Refusals if the committed overrides are not exactly the reviewed ones.

    The override list is read from the adapter pyproject at the built commit, so
    an edit there that this module was not reviewed against stops the build
    before anything is installed -- including an edit that keeps the same two
    package names but moves a security floor down, drops an extra, or attaches a
    marker that DISABLES the override on some platform.
    """

    parsed = [_requirement(text) for text in overrides]
    if any(item is None for item in parsed):
        return [f"override-dependencies contains an invalid requirement: {overrides!r}"]
    committed = frozenset(
        ReviewedOverride(
            canonicalize_name(item.name),
            item.specifier,
            frozenset(canonicalize_name(extra) for extra in item.extras),
            str(item.marker) if item.marker is not None else None,
        )
        for item in parsed
        if item is not None
    )

    def describe(rows: Iterable[ReviewedOverride]) -> list[str]:
        return sorted(
            f"{row.dependency}"
            f"{'[' + ','.join(sorted(row.extras)) + ']' if row.extras else ''}"
            f"{row.specifiers or ''}"
            f"{' ; ' + row.marker if row.marker else ''}"
            for row in rows
        )

    refusals: list[str] = []
    if committed != reviewed:
        refusals.append(
            f"override-dependencies is {describe(committed)}, but the reviewed policy is "
            f"{describe(reviewed)}; a policy change needs review, not an admission edit"
        )
    unbounded = sorted(
        {row.dependency for row in reviewed} - {bound.dependency for bound in bounds}
    )
    if unbounded:  # pragma: no cover - editing one set without the other fails here
        refusals.append(f"reviewed overrides name {unbounded} with no reviewed upstream bound")
    return refusals


def override_names(overrides: Sequence[str]) -> list[tuple[str, tuple[str, ...]]]:
    """(canonical name, sorted extras) for each override, in committed order.

    The builder narrows these to exact pins for pip, so it must not hard-code
    the names or the ``[crypto]`` spelling: an extra lost on the way would
    install a package without a backend the harness floor needs.
    """

    narrowed: list[tuple[str, tuple[str, ...]]] = []
    for text in overrides:
        requirement = _requirement(text)
        if requirement is None:
            raise ValueError(f"invalid override requirement: {text!r}")
        narrowed.append(
            (
                canonicalize_name(requirement.name),
                tuple(sorted(canonicalize_name(extra) for extra in requirement.extras)),
            )
        )
    return narrowed


def _version(value: str | None) -> Version | None:
    if value is None:
        return None
    try:
        return Version(value)
    except InvalidVersion:
        return None


def admit(
    report_text: str,
    returncode: int,
    *,
    overrides: Sequence[str],
    installed: Mapping[str, str],
    declared: Mapping[str, Sequence[str]],
    core_requirements: Sequence[str] = (),
    marker_environment: Mapping[str, str] | None = None,
    prefix: str | None = None,
    bounds: frozenset[SupersededBound] = SUPERSEDED_BOUNDS,
) -> Admission:
    """Decide admission from the raw report plus installed facts. Pure.

    ``installed`` maps canonical name -> installed version for the arm.
    ``declared`` maps each superseded bound's parent -> its installed
    ``Requires-Dist`` strings, the independent confirmation that the bound the
    checker reported is the one really present in the installed metadata.
    ``prefix`` is the arm interpreter's ``sys.prefix``; when the report names an
    environment, it must be that one.
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
            f"missing dependency: {finding.package} requires "
            f"{finding.name}{finding.specifier}, not installed"
        )
    total = len(report.conflicts) + len(report.missing)
    if report.count is None and total:
        refusals.append("checker listed findings without its 'Found N' count line")
    elif report.count is not None and report.count != total:
        refusals.append(f"checker counted {report.count} findings but listed {total}")

    if prefix is not None and report.environment is not None:
        # A report about another environment says nothing about this arm.
        reported_prefix = Path(report.environment).expanduser()
        try:
            same = reported_prefix.resolve() == Path(prefix).resolve()
        except OSError:  # pragma: no cover - a path that cannot be resolved
            same = False
        if not same:
            refusals.append(f"raw report describes {reported_prefix}, not this arm's {prefix}")

    expected = {expected_finding(bound).key: bound for bound in bounds}
    # EXACTLY ONCE, and nothing else: an extras-qualified, marker-qualified or
    # duplicated line is a different requirement, not the reviewed one.
    seen: dict[tuple[str, str, str, tuple[str, ...], str | None], int] = {}
    for finding in report.conflicts:
        seen[finding.key] = seen.get(finding.key, 0) + 1
        if finding.key not in expected:
            refusals.append(
                "unadmitted conflict: "
                f"{finding.package} requires "
                f"{finding.name}{'[' + ','.join(finding.extras) + ']' if finding.extras else ''}"
                f"{finding.specifier}"
                f"{' ; ' + finding.marker if finding.marker else ''},"
                f" {finding.reported_install} installed"
            )
            continue
        if finding.reported_install is not None:
            stated = _version(finding.reported_install)
            actual = _version(installed.get(finding.name))
            if stated is not None and actual is not None and stated != actual:
                refusals.append(
                    f"raw report says {finding.name} {stated} is installed, but this arm has"
                    f" {actual}"
                )
    for key, bound in expected.items():
        count = seen.get(key, 0)
        if count == 0:
            refusals.append(
                f"superseded bound absent from the raw report: {bound.parent} requires "
                f"{bound.dependency}{bound.upstream} (upstream changed, or report incomplete)"
            )
        elif count > 1:
            refusals.append(
                f"superseded bound appears {count} times in the raw report: {bound.parent}"
                f" requires {bound.dependency}{bound.upstream}"
            )

    # Independent confirmation from installed metadata, not from the report.
    for bound in bounds:
        wanted = expected_finding(bound)
        rows = [_finding(bound.parent, text) for text in declared.get(bound.parent, ())]
        matches = [row for row in rows if row.key == wanted.key]
        if not matches:
            described = sorted(
                f"{row.name}"
                f"{'[' + ','.join(row.extras) + ']' if row.extras else ''}"
                f"{row.specifier}"
                f"{' ; ' + row.marker if row.marker else ''}"
                for row in rows
                if row.name == bound.dependency
            )
            refusals.append(
                f"installed {bound.parent} metadata does not declare "
                f"{bound.dependency}{bound.upstream} exactly (found {described})"
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
        admitted_bounds=tuple(sorted(bounds, key=lambda b: b.parent)) if not refusals else (),
    )


def _installed_facts(
    bounds: Iterable[SupersededBound],
) -> tuple[dict[str, str], dict[str, list[str]]]:
    """Versions and superseded parents' Requires-Dist of THIS interpreter."""

    installed: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata["Name"]
        if name:
            installed[canonicalize_name(name)] = distribution.version
    declared: dict[str, list[str]] = {}
    for bound in bounds:
        try:
            declared[bound.parent] = list(importlib.metadata.requires(bound.parent) or [])
        except importlib.metadata.PackageNotFoundError:
            declared[bound.parent] = []
    return installed, declared


def _uv_version() -> str:
    try:
        result = subprocess.run(
            ["uv", "--version", "--color", "never"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return "unavailable"
    return (result.stdout + result.stderr).strip() or "unavailable"


def run_check(receipts: Path) -> tuple[str, int]:
    """Run the raw checker, KEEPING its output and rc before any verdict.

    ``--color never`` is not cosmetic: with ``FORCE_COLOR`` or ``CLICOLOR_FORCE``
    in the caller's environment uv wraps its lines in ANSI escapes, every line
    then fails to parse, and a perfectly good arm is refused. A missing ``uv``
    is reported as a refusal with a raw log rather than a traceback.
    """

    try:
        check = subprocess.run(
            ["uv", "pip", "check", "--color", "never", "--python", sys.executable],
            capture_output=True,
            text=True,
            check=False,
        )
        text, returncode = check.stdout + check.stderr, check.returncode
    except OSError as error:
        text, returncode = f"uv could not be executed: {error}\n", 127
    receipts.mkdir(parents=True, exist_ok=True)
    (receipts / "raw-pip-check.log").write_text(text)
    (receipts / "raw-pip-check.rc").write_text(f"{returncode}\n")
    return text, returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--adapter-pyproject", required=True, type=Path)
    parser.add_argument("--core-pyproject", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path, help="receipt directory")
    args = parser.parse_args(argv)

    adapter = tomllib.loads(args.adapter_pyproject.read_text())
    core = tomllib.loads(args.core_pyproject.read_text())
    # The checker always runs against the interpreter running this file: the
    # installed facts below are read in-process, so both halves describe one env.
    text, returncode = run_check(args.out)
    installed, declared = _installed_facts(SUPERSEDED_BOUNDS)
    verdict = admit(
        text,
        returncode,
        overrides=adapter["tool"]["uv"]["override-dependencies"],
        installed=installed,
        declared=declared,
        core_requirements=core["project"]["dependencies"],
        marker_environment={str(key): str(value) for key, value in default_environment().items()},
        prefix=sys.prefix,
    )
    receipt = verdict.as_receipt()
    receipt["raw_pip_check_sha256"] = hashlib.sha256(text.encode()).hexdigest()
    receipt["interpreter"] = sys.executable
    # The parser is tied to uv's wording, so WHICH uv produced the raw log is
    # part of the evidence, not a detail to reconstruct later.
    receipt["uv_version"] = _uv_version()
    (args.out / "policy-admission.json").write_text(json.dumps(receipt, indent=2) + "\n")
    if not verdict.admitted:
        for reason in verdict.refusals:
            print(f"REFUSED: {reason}", file=sys.stderr)
        return 1
    print("ADMITTED: policy-conformant; raw uv pip check rc=1 retained, NOT green")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
