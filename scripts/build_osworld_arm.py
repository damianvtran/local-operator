#!/usr/bin/env python3
"""Build one OSWorld arm from a committed SHA, then admit its dependencies.

The arm is the interpreter the paid adapter runs in: the harness wheel, the
adapter wheel, and the adapter's locked ``osworld`` extra. This script is the
durable form of the operator recipe that built the main16/main17 arms; it
encodes the facts that recipe learned the expensive way, so none of them has to
be rediscovered:

1. THE ADAPTER'S OVERRIDE POLICY IS AUTHORITATIVE. The built commit's
   ``benchmarks/osworld_v2_adapter/pyproject.toml`` declares
   ``[tool.uv].override-dependencies`` for exactly two stale upstream bounds
   (OSWorld's ``requests~=2.31.0`` and ZhipuAI's ``pyjwt>=2.8.0,<2.9.0``).
   Those overrides go to the resolver twice, because an exported requirements
   file cannot carry project policy: into the throwaway resolution project that
   re-resolves the closure (so a genuinely new mandatory core dependency is
   added BY RESOLUTION, not hidden), and onto the install itself narrowed to
   the versions that resolution chose. A raw ``uv pip check`` in a correct arm
   still exits 1 listing exactly those two bounds; that raw result is preserved
   and ``scripts/osworld_dependency_admission.py`` decides admission
   separately. Upstream wheel metadata is NEVER rewritten, no security library
   is downgraded, and ``--no-deps`` is never used to hide a missing
   requirement -- main16 was refused for exactly that (a missing ``miniaudio``
   its stale lock predated).

2. THE VENV IS ``venv --copies`` PLUS THE DYLIB. ``discovery._symlink_free``
   refuses a launch path whose real path differs from the named one, and
   ``uv venv`` always writes ``bin/python3.12`` as a symlink at every flag
   combination (measured 2026-10-25: an arm was refused pre-dispatch with
   "adapter launch path has a symlink or lexical alias"), so the stdlib
   ``venv --copies --without-pip`` is used instead. ``--copies`` copies the
   interpreter but NOT the shared library it links, and the copied binary keeps
   ``@rpath/libpython3.12.dylib`` with an rpath of ``<venv>/lib``: without the
   copy every invocation dies in dyld. Both halves are here, and ``python -V``
   straight after is the real check -- a copy that finds no dylib is allowed
   (an interpreter that links by absolute path needs none), an interpreter that
   cannot start is a refusal.

3. A WHEEL ARCHIVE HASH IS NOT AN IDENTITY. Zip member timestamps differ
   between two builds of the same commit, so the ``.whl`` sha256 is not
   reproducible and must not be a drift detector (main16's recorded archive
   mismatch was this, not tampering). Neither is the adapter's
   ``package_digest``: it hashes the installed RECORD, and RECORD contains rows
   the INSTALLER writes (``direct_url.json`` with the absolute wheel path,
   ``uv_cache.json`` with a timestamp), so it differs between two installs of
   one commit. It is kept for what it is -- the value the selector pins to THIS
   install -- and the cross-build identity is the source commit, the pinned core
   module hashes and ``source_record_digest`` (RECORD minus installer-written
   files) in ``receipts/installed-verification.json``. Archive hashes appear
   only under ``wheel_sha256_informational_only``.

PLAN, THEN BUILD. ``--plan <file>`` writes a small JSON receipt (never a source
tree, venv or wheel) and creates NOTHING under ``--out``, so the documented
sequence works and a plan costs kilobytes rather than the ~220 MB a full arm
does. ``--execute --plan <file>`` re-resolves, refuses unless its plan digest
still matches, and records the binding in ``build-record.json``.

What this script does NOT do: it allocates no guest, contacts no provider or
model, runs no episode, and supports no benchmark claim. It does need the
network -- the locked closure and the pinned OSWorld git ref are fetched. It
never reuses an arm directory: an arm is an immutable receipt. See
``docs/benchmarks/osworld_2/arm-build.md``.

Exit codes: 2 = refused (bad input, unmerged source, existing arm, unusable
interpreter, failed command); 4 = built but dependency admission or the
installed-identity verification refused. Receipts exist in both cases.

Usage:
    python scripts/build_osworld_arm.py --source-sha <40-hex> --plan /tmp/arm-plan.json
    python scripts/build_osworld_arm.py --source-sha <40-hex> --out <new dir> \\
        --plan /tmp/arm-plan.json --execute
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from local_operator.evaluation.adapters.discovery import (  # noqa: E402
    AdapterDiscoveryError,
    _symlink_free,
)
from local_operator.evaluation.runner import durable_root  # noqa: E402
from scripts import osworld_dependency_admission as admission  # noqa: E402

EXIT_REFUSED = 2
EXIT_UNVERIFIED = 4

_ADAPTER_DIR = "benchmarks/osworld_v2_adapter"
_HARNESS_DISTRIBUTION = "local-operator"
_ADAPTER_DISTRIBUTION = "lop-osworld-v2-adapter"
_RESOLUTION_PROJECT = "lop-osworld-arm-resolution"
# The build record's on-disk contract. Two paths, one document: the launcher
# scripts (materialize_workspace_*.sh) read `<arm>/build-record.json`, while the
# receipts copy is what earlier arms and arm-build.md name. Writing one and
# hard-linking the other by hand is what this constant exists to end.
_RECORD_CONTRACT = "lop-osworld-arm-build-record/v1"
# The arm interpreter's own minor: the adapter requires >=3.12 and the pinned
# osworld extra is only resolved for one interpreter, so a different minor would
# need a different lock rather than a different flag.
_PYTHON_MINOR = "3.12"
# ``requires-python`` of the resolution project. ``==3.12.*`` (not ``>=``) so uv
# resolves the closure the arm interpreter will actually run.
_PYTHON_FLOOR = f"=={_PYTHON_MINOR}.*"
_SHA = re.compile(r"[0-9a-f]{40}")
# ``--color never`` on every uv call: with ``FORCE_COLOR`` or ``CLICOLOR_FORCE``
# in the caller's environment uv writes ANSI escapes into the captured pipe, and
# the checker's output then fails to parse. The env var cannot be trusted; the
# flag can.
_UV = ("uv", "--color", "never")
# Every child this script spawns carries the notification gate. Two reasons, and
# the sweep in tests/unit/test_notification_isolation.py enforces the first: a
# module that hands a child a built environment must make the switch its own
# business rather than inheriting it, because a child that raises a banner puts
# a mock session's reply on the operator's lock screen. The second is specific to
# this script -- the arm interpreter it launches imports the whole harness, so an
# episode-shaped import path here should be as unable to notify as any other.
_NOTIFICATIONS_GATE = "LOCAL_OPERATOR_NO_NOTIFICATIONS"


def _child_env(**extra: str) -> dict[str, str]:
    """The environment every child of this script runs with."""

    return {**os.environ, _NOTIFICATIONS_GATE: "1", **extra}


class BuildRefused(RuntimeError):
    """A precondition failed; nothing was installed and the arm is abandoned."""


class Unverified(RuntimeError):
    """The arm installed but does not satisfy admission or identity verification."""


@dataclass(frozen=True)
class ArmLayout:
    """Every path one arm owns, derived once so no step invents its own."""

    root: Path

    @property
    def source(self) -> Path:
        return self.root / "source"

    @property
    def core_dist(self) -> Path:
        return self.root / "dist" / "core"

    @property
    def adapter_dist(self) -> Path:
        return self.root / "dist" / "adapter"

    @property
    def venv(self) -> Path:
        return self.root / "venv"

    @property
    def python(self) -> Path:
        return self.venv / "bin" / f"python{_PYTHON_MINOR}"

    @property
    def receipts(self) -> Path:
        return self.root / "receipts"

    def create_arm(self) -> None:
        """Create the arm root ATOMICALLY, refusing an existing one.

        One ``mkdir(exist_ok=False)`` rather than an ``exists()`` check followed
        by an ``exist_ok=True`` mkdir: the two-step form races, and losing that
        race would let a second builder install into an arm whose receipts the
        first one is still writing. An arm directory is immutable evidence.
        """

        try:
            self.root.mkdir(parents=True, exist_ok=False)
        except FileExistsError as error:
            raise BuildRefused(f"{self.root} exists; an arm receipt is immutable") from error
        for path in (self.source, self.core_dist, self.adapter_dist, self.receipts):
            path.mkdir(parents=True, exist_ok=True)


def run(argv: Sequence[str]) -> str:
    """Run one external command, reporting failure as a refusal, not a traceback."""

    try:
        completed = subprocess.run(
            list(argv), capture_output=True, text=True, check=False, env=_child_env()
        )
    except OSError as error:
        raise BuildRefused(f"{argv[0]} could not be executed: {error}") from error
    output = (completed.stdout + completed.stderr).strip()
    if completed.returncode != 0:
        tail = output[-2000:] if output else "(no output)"
        raise BuildRefused(f"{' '.join(argv)} failed (rc={completed.returncode}): {tail}")
    return output


def git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """A git call that cannot reach the network.

    ``GIT_NO_LAZY_FETCH=1`` matters in this checkout, which is a ``blob:none``
    partial clone: without it, asking about a commit this clone does not have
    makes git lazily contact origin, so a refusal would depend on the network
    being up (and would leak the question to the remote) before it could say no.
    """

    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=False,
        env=_child_env(GIT_NO_LAZY_FETCH="1", GIT_TERMINAL_PROMPT="0"),
    )


def instance_sha256(path: Path) -> str:
    """Hash of THIS file instance -- a receipt value, never an identity."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def refuse_volatile(path: Path, label: str) -> None:
    """An arm under ``$TMPDIR`` is one OS purge away from a half-built receipt.

    Shares ``runner.durable_root`` with ``scripts/build_osworld_adapter.py`` so
    the arm build and the episode agree on what "durable" means; a purge mid-run
    already destroyed a prepared checkout and left an EC2 instance running.
    """

    try:
        durable_root.refuse_volatile_root(path, label=label)
    except durable_root.VolatileRootError as error:
        raise BuildRefused(str(error)) from error


def guard_source(repo: Path, sha: str) -> None:
    """The source must be one exact commit on ``origin/main``'s FIRST-PARENT line.

    Mere ancestry is not enough. This repository merges with merge commits, so
    every intermediate commit of a merged PR branch is an ancestor of ``main`` --
    including pre-remediation states that review rejected, which the arm must not
    be built from. A floating ref is refused for the same reason: a later build
    would pick up commits nobody reviewed.
    """

    if not _SHA.fullmatch(sha):
        raise BuildRefused("--source-sha must be the full 40-character lowercase SHA")
    resolved = git(repo, "rev-parse", f"{sha}^{{commit}}")
    if resolved.returncode != 0:
        raise BuildRefused(f"{sha} is not a commit object in {repo}: {resolved.stderr.strip()}")
    if resolved.stdout.strip() != sha:
        raise BuildRefused(f"{sha} does not resolve to itself as a commit")
    first_parent = git(repo, "rev-list", "--first-parent", "origin/main")
    if first_parent.returncode != 0:
        raise BuildRefused(f"origin/main is not usable in {repo}: {first_parent.stderr.strip()}")
    if sha not in first_parent.stdout.split():
        raise BuildRefused(
            f"{sha} is not on origin/main's first-parent history; arms are built from merged "
            "commits, never from an intermediate commit of a merged branch or an unmerged one"
        )


def materialise_source(repo: Path, sha: str, destination: Path) -> None:
    """Extract the COMMITTED tree; the working tree is never an input.

    ``git archive`` also makes a dirty checkout harmless: what is built is what
    the commit contains, which is what the receipts claim.
    """

    destination.mkdir(parents=True, exist_ok=True)
    archive = subprocess.Popen(  # noqa: S603 - argv list, no shell
        ["git", "-C", str(repo), "archive", sha],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=_child_env(GIT_NO_LAZY_FETCH="1", GIT_TERMINAL_PROMPT="0"),
    )
    if archive.stdout is None:  # pragma: no cover - defensive
        raise BuildRefused("git archive produced no output stream")
    tar = subprocess.run(
        ["tar", "-x", "-C", str(destination)],
        stdin=archive.stdout,
        capture_output=True,
        text=False,
        check=False,
        env=_child_env(),
    )
    archive.stdout.close()
    status = archive.wait()
    if status != 0 or tar.returncode != 0:
        raise BuildRefused(
            f"git archive {sha} could not be extracted into {destination}"
            f" (git rc={status}, tar rc={tar.returncode})"
        )


@dataclass(frozen=True)
class Metadata:
    """The three committed files the policy is read from, and their hashes."""

    adapter_dir: Path
    core_pyproject: Path

    @property
    def adapter_pyproject(self) -> Path:
        return self.adapter_dir / "pyproject.toml"

    @property
    def adapter_lock(self) -> Path:
        return self.adapter_dir / "uv.lock"


def metadata_from_source(source: Path) -> Metadata:
    """Policy bytes for an arm build, from the extracted commit."""

    return Metadata(source / _ADAPTER_DIR, source / "pyproject.toml")


def metadata_from_git(repo: Path, sha: str, destination: Path) -> Metadata:
    """Policy bytes for a PLAN, without extracting 200 MB of tree.

    ``git show <sha>:<path>`` yields the same bytes the arm build will read, so
    the plan resolves the same closure an execute would, while writing only
    three small files into a temporary directory.
    """

    adapter = destination / _ADAPTER_DIR
    adapter.mkdir(parents=True, exist_ok=True)
    for relative, target in (
        (f"{_ADAPTER_DIR}/pyproject.toml", adapter / "pyproject.toml"),
        (f"{_ADAPTER_DIR}/uv.lock", adapter / "uv.lock"),
        ("pyproject.toml", destination / "pyproject.toml"),
    ):
        shown = git(repo, "show", f"{sha}:{relative}")
        if shown.returncode != 0:
            raise BuildRefused(f"{relative} is missing from {sha}: {shown.stderr.strip()}")
        target.write_text(shown.stdout)
    return Metadata(adapter, destination / "pyproject.toml")


def split_frozen_requirements(text: str) -> tuple[list[str], list[str]]:
    """Separate the frozen export into installable rows and the harness row.

    The exported ``local-operator==<old version>`` row is dropped: the arm
    installs the harness WHEEL built from the same commit, while the committed
    adapter lock deliberately lags the harness version it runs against. Dropping
    it is recorded, never silent -- the row is written to a receipts file.
    """

    rows: list[str] = []
    dropped: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("local-operator=="):
            dropped.append(line)
            continue
        if line.startswith("-"):
            raise BuildRefused(f"unexpected pip option in a frozen export: {line!r}")
        rows.append(line)
    if not rows:
        raise BuildRefused("the frozen adapter export is empty")
    return rows, dropped


def resolution_project(
    *,
    adapter_rows: Sequence[str],
    core_requirements: Sequence[str],
    overrides: Sequence[str],
    constraint_versions: Mapping[str, str],
) -> str:
    """The throwaway uv project that resolves the arm closure under the policy.

    WHY a second project at all: ``uv export`` flattens the adapter lock to exact
    rows and loses ``[tool.uv].override-dependencies``, so replaying those rows
    through plain pip rejects the two knowingly superseded bounds. Dropping
    dependency resolution instead (``--no-deps``) would hide any requirement the
    committed lock predates, which is how main16 ended up missing ``miniaudio``.
    This project keeps the policy explicit and lets the resolver add a genuinely
    new mandatory core dependency as an ordinary resolution.
    """

    refusals = admission.check_policy(overrides)
    if refusals:
        raise BuildRefused("; ".join(refusals))
    dependencies = [*adapter_rows, *core_requirements]
    constraints = [f"{name}=={version}" for name, version in sorted(constraint_versions.items())]
    lines = [
        "# GENERATED by scripts/build_osworld_arm.py -- never edited by hand.",
        "[project]",
        'name = "lop-osworld-arm-resolution"',
        'version = "0.0.0"',
        f"requires-python = {json.dumps(_PYTHON_FLOOR)}",
        "dependencies = [",
        *(f"  {json.dumps(dep)}," for dep in dependencies),
        "]",
        "[tool.uv]",
        # Resolution only: this directory is not a distribution, so uv locks a
        # closure instead of trying to build a package.
        "package = false",
        # Overrides REPLACE an upstream requirement. The two overridden packages
        # also stay pinned as constraints, so a broad policy floor cannot
        # incidentally move the versions the committed lock already froze.
        f"override-dependencies = {json.dumps(list(overrides))}",
        f"constraint-dependencies = {json.dumps(constraints)}",
        "",
    ]
    return "\n".join(lines)


def _constraint_versions(
    committed_lock: Mapping[str, str], overrides: Sequence[str]
) -> dict[str, str]:
    """Pin each overridden package at the version the COMMITTED lock chose.

    The override is a policy RANGE; the lock's exact version is what the arm is
    meant to run. Keeping them as separate constraint entries means a broad
    floor cannot incidentally move a version the committed lock already fixed.
    The names come from the override list (never hard-coded), so a policy edit
    that adds a third override cannot silently skip its constraint.
    """

    names = [name for name, _extras in admission.override_names(overrides)]
    missing = [name for name in names if name not in committed_lock]
    if missing:
        raise BuildRefused(f"the committed adapter lock has no row for {sorted(missing)}")
    return {name: committed_lock[name] for name in names}


def lock_versions(lock_text: str) -> dict[str, str]:
    """canonical name -> version for every package a uv lock pins.

    The frozen export carries direct references (``osworld @ git+https://...``)
    that carry no version, so the LOCK -- not the exported row -- is the only
    place the whole planned set has a version.
    """

    lock = tomllib.loads(lock_text)
    return {admission.canonicalize_name(row["name"]): row["version"] for row in lock["package"]}


def installed_overrides(lock_text: str, overrides: Sequence[str]) -> list[str]:
    """Narrow the policy overrides to the versions the resolution chose.

    pip has no project policy, so the overrides are passed to the install
    explicitly -- but as exact pins: a floor would let an already-installed newer
    version win over the frozen choice, which is how an arm silently stops
    matching its own lock. The names and EXTRAS come from the committed override
    list rather than being spelled here, so an extra the harness floor needs
    cannot be lost between the policy and the install.
    """

    versions = lock_versions(lock_text)
    pins: list[str] = []
    for name, extras in admission.override_names(overrides):
        if name not in versions:
            raise BuildRefused(f"the resolution lock has no {name} row")
        spelling = f"{name}[{','.join(extras)}]" if extras else name
        pins.append(f"{spelling}=={versions[name]}")
    return pins


def plan_digest(rows: Sequence[str], overrides: Sequence[str]) -> str:
    """A stable digest of WHAT the plan resolved, for an execute to check.

    Not a digest of the lock file: a uv lock records the project's own relative
    source and uv's serialisation order, so two runs in different directories
    would disagree for reasons that do not affect the arm. The installed set and
    the policy do affect it, and those are what this covers.
    """

    payload = json.dumps(
        {"overrides": list(overrides), "pins": sorted(row.strip() for row in rows if row.strip())},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def create_arm_venv(venv: Path, receipts: Path) -> dict[str, Any]:
    """Create the symlink-free, dylib-complete interpreter the adapter needs.

    Both halves fail LATE -- after resolution, at launch -- so both are enforced
    here. See this module's docstring, point 2, for the measurements behind them.
    The venv module is invoked through the base interpreter's REAL path because
    it records the invoking ``argv[0]`` directory as ``home`` in ``pyvenv.cfg``,
    and a ``~/.local/bin`` shim holds no ``lib/``: the dylib copy would then find
    nothing and the failure would read as a dyld error with no cause attached.
    """

    base = Path(run([f"python{_PYTHON_MINOR}", "-c", "import sys; print(sys.base_prefix)"]).strip())
    real = base / "bin" / f"python{_PYTHON_MINOR}"
    if not real.is_file() or real.is_symlink():
        raise BuildRefused(f"base interpreter {real} is not a real file")
    venv.parent.mkdir(parents=True, exist_ok=True)
    run([str(real), "-m", "venv", "--copies", "--without-pip", str(venv)])
    dylib = base / "lib" / f"libpython{_PYTHON_MINOR}.dylib"
    copied = dylib.is_file()
    if copied:
        shutil.copyfile(dylib, venv / "lib" / dylib.name)
    interpreter = venv / "bin" / f"python{_PYTHON_MINOR}"
    # The REAL guard, not a copy of its logic: this script runs in the repo venv
    # and can import it, and two implementations of "is this launch path a
    # symlink or lexical alias" would eventually disagree about which paths pass.
    try:
        _symlink_free(interpreter)
    except AdapterDiscoveryError as error:
        raise BuildRefused(f"discovery refuses this interpreter: {error}") from error
    try:
        version = run([str(interpreter), "-V"]).strip()
    except BuildRefused as error:
        raise BuildRefused(
            f"the copied interpreter cannot start (dyld / missing libpython): {error}"
        ) from error
    facts: dict[str, Any] = {
        "interpreter": str(interpreter),
        "interpreter_realpath": str(interpreter.resolve()),
        "interpreter_sha256": instance_sha256(interpreter),
        "reported_version": version,
        "base_prefix": str(base),
        "libpython_copied": copied,
        "libpython_source": str(dylib) if copied else None,
    }
    (receipts / "interpreter.json").write_text(json.dumps(facts, indent=2) + "\n")
    return facts


def export_adapter_requirements(metadata: Metadata, receipts: Path) -> tuple[list[str], list[str]]:
    """Export the built commit's adapter lock (osworld extra included) and split it."""

    export = receipts / "requirements.locked.txt"
    run(
        [
            *_UV,
            "export",
            "--project",
            str(metadata.adapter_dir),
            "--frozen",
            "--no-emit-project",
            "--extra",
            "osworld",
            "--no-hashes",
            "-o",
            str(export),
        ]
    )
    rows, dropped = split_frozen_requirements(export.read_text())
    (receipts / "requirements.arm.txt").write_text("\n".join(rows) + "\n")
    (receipts / "requirements.dropped.txt").write_text(
        "# dropped from the frozen export; the harness wheel supplies this row\n"
        + "\n".join(dropped)
        + "\n"
    )
    return rows, dropped


def plan_metadata_digests(metadata: Metadata) -> dict[str, str]:
    return {
        "core_metadata_sha256": file_sha256(metadata.core_pyproject),
        "adapter_metadata_sha256": file_sha256(metadata.adapter_pyproject),
        "frozen_adapter_lock_sha256": file_sha256(metadata.adapter_lock),
    }


def resolve_policy_project(
    *,
    metadata: Metadata,
    receipts: Path,
    adapter_rows: Sequence[str],
    core_requirements: Sequence[str],
    overrides: Sequence[str],
    constraint_versions: Mapping[str, str],
    interpreter: Path,
) -> tuple[list[str], Path, dict[str, Any]]:
    """Lock and export the synthetic project; return the install pins and facts."""

    project = receipts / "resolution-project"
    project.mkdir(parents=True, exist_ok=True)
    (project / "pyproject.toml").write_text(
        resolution_project(
            adapter_rows=adapter_rows,
            core_requirements=core_requirements,
            overrides=overrides,
            constraint_versions=constraint_versions,
        )
    )
    run([*_UV, "lock", "--project", str(project), "--python", str(interpreter)])
    pins = receipts / "policy-pins.txt"
    run(
        [
            *_UV,
            "export",
            "--project",
            str(project),
            "--locked",
            "--no-emit-project",
            "--no-hashes",
            "-o",
            str(pins),
        ]
    )
    lock_text = (project / "uv.lock").read_text()
    lock = lock_versions(lock_text)
    rows = [line.strip() for line in pins.read_text().splitlines() if line.strip()]
    rows = [row for row in rows if not row.startswith("#")]
    facts = {
        "resolution_package_count": len(lock),
        "security_pins": {
            name: lock[name]
            for name in sorted(lock)
            if name in {"requests", "pyjwt", "miniaudio", "watchdog"}
        },
        "plan_digest": plan_digest(rows, overrides),
    }
    return installed_overrides(lock_text, overrides), pins, facts


def build_wheels(layout: ArmLayout) -> tuple[Path, Path]:
    """Build both wheels from the EXTRACTED commit, never from a working tree.

    The adapter must be a wheel, never ``-e``: an editable install's RECORD
    covers only the ``.pth`` shim and metadata, so ``discovery`` reports "adapter
    entry module is not uniquely RECORD-covered" -- an error that names RECORD
    coverage rather than the editable install that caused it.
    """

    run([*_UV, "build", "--wheel", "--out-dir", str(layout.core_dist), str(layout.source)])
    run(
        [
            *_UV,
            "build",
            "--wheel",
            "--out-dir",
            str(layout.adapter_dist),
            str(layout.source / _ADAPTER_DIR),
        ]
    )
    cores = sorted(layout.core_dist.glob("local_operator-*.whl"))
    adapters = sorted(layout.adapter_dist.glob("lop_osworld_v2_adapter-*.whl"))
    if len(cores) != 1 or len(adapters) != 1:
        raise BuildRefused(
            "expected exactly one wheel per distribution, found "
            f"core={[p.name for p in cores]} adapter={[p.name for p in adapters]}"
        )
    return cores[0], adapters[0]


def install_arm(
    *,
    interpreter: Path,
    pins: Path,
    overrides: Sequence[str],
    receipts: Path,
    wheels: tuple[Path, Path],
) -> None:
    """Install the pinned closure plus the two exact wheels in ONE resolution.

    ``-c``/``--overrides`` carry the exported policy, not convenience: without
    them pip honours the SDKs' stale bounds and fails, and ``--no-deps`` would
    hide a genuinely missing requirement instead of resolving it.
    """

    (receipts / "installation-overrides.txt").write_text("\n".join(overrides) + "\n")
    run(
        [
            *_UV,
            "pip",
            "install",
            "--python",
            str(interpreter),
            "-r",
            str(pins),
            "-c",
            str(pins),
            "--overrides",
            str(receipts / "installation-overrides.txt"),
            str(wheels[0]),
            str(wheels[1]),
        ]
    )


def compare_closure(
    *, pinned: Mapping[str, str], installed: Mapping[str, str], allowed_extra: Sequence[str]
) -> list[str]:
    """Refuse an arm whose installed set the receipts do not describe.

    The direction that matters is "installed but not planned": a package the
    resolver added AT INSTALL TIME is exactly the drift a reproducibility claim
    must not absorb. Absence is not a refusal -- a pinned row whose marker
    excludes this platform is legitimately absent, and the harness's own direct
    requirements are checked separately by admission.
    """

    allowed = {admission.canonicalize_name(name) for name in allowed_extra}
    refusals: list[str] = []
    for name, version in sorted(installed.items()):
        key = admission.canonicalize_name(name)
        if key in allowed:
            continue
        planned_version = pinned.get(key)
        if planned_version is None:
            refusals.append(f"installed {name} {version} is not in the resolution plan")
        elif planned_version != version:
            refusals.append(f"installed {name} {version} differs from planned {planned_version}")
    return refusals


def _plan(args: argparse.Namespace) -> int:
    """Resolve and report, writing a small plan receipt and NOTHING at --out."""

    with tempfile.TemporaryDirectory(prefix="lop-arm-plan-") as temporary:
        workspace = Path(temporary)
        metadata = metadata_from_git(args.repo, args.source_sha, workspace / "metadata")
        receipts = workspace / "receipts"
        receipts.mkdir()
        overrides, core_requirements = policy(metadata)
        adapter_rows, dropped = export_adapter_requirements(metadata, receipts)
        interpreter = Path(sys.executable)
        running = running_python_version()
        if running.split(".")[:2] != _PYTHON_MINOR.split("."):
            raise BuildRefused(
                f"a plan must be resolved with a {_PYTHON_MINOR} interpreter (this one is "
                f"{running}); run it with the arm's python or a {_PYTHON_MINOR} venv"
            )
        constraints = _constraint_versions(
            lock_versions(metadata.adapter_lock.read_text()), overrides
        )
        install_overrides, pins, facts = resolve_policy_project(
            metadata=metadata,
            receipts=receipts,
            adapter_rows=adapter_rows,
            core_requirements=core_requirements,
            overrides=overrides,
            constraint_versions=constraints,
            interpreter=interpreter,
        )
        receipt = {
            "source_commit": args.source_sha,
            "overrides": overrides,
            "installation_overrides": install_overrides,
            # A plan resolves inside a TemporaryDirectory, so a path here would
            # name a file the receipt cannot outlive. The digest is the part that
            # outlives it, and it is what a build can check itself against.
            "requirements_digest": requirements_digest(pins),
            "dropped_rows": dropped,
            "uv_version": uv_version(),
            **plan_metadata_digests(metadata),
            **facts,
        }
        if args.plan is not None:
            args.plan.parent.mkdir(parents=True, exist_ok=True)
            args.plan.write_text(json.dumps(receipt, indent=2) + "\n")
    print(
        "PLAN: "
        f"{facts['resolution_package_count']} packages, plan digest {facts['plan_digest'][:16]}, "
        f"security pins {facts['security_pins']}; nothing installed, nothing written to --out"
    )
    if args.plan is not None:
        print(f"PLAN RECEIPT: {args.plan}")
    return 0


def requirements_digest(pins: Path) -> str:
    """sha256 of the pinned rows, so a plan receipt outlives its temp directory."""

    rows = [
        line.strip()
        for line in pins.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    return hashlib.sha256("\n".join(sorted(rows)).encode()).hexdigest()


def _module_hashes(rows: Any) -> dict[str, str] | None:
    """Per-module hashes from the verifier's ``core_source`` block, if present."""

    if not isinstance(rows, Mapping):
        return None
    return {
        key: value["hash"]
        for key, value in rows.items()
        if isinstance(value, dict) and isinstance(value.get("hash"), str)
    }


def policy(metadata: Metadata) -> tuple[list[str], list[str]]:
    """The committed override list and the harness's direct requirements."""

    adapter = tomllib.loads(metadata.adapter_pyproject.read_text())
    core = tomllib.loads(metadata.core_pyproject.read_text())
    return (
        list(adapter["tool"]["uv"]["override-dependencies"]),
        list(core["project"]["dependencies"]),
    )


def running_python_version() -> str:
    """The version of the interpreter a plan resolves with (``sys.executable``).

    A seam, not a convenience: the plan guard is a function of the process it
    runs in, and the unit suite runs on 3.12 AND 3.13. Read inline, the guard
    made the plan tests pass or fail by CI leg; behind this function each test
    names the interpreter it is about.
    """

    return sys.version.split()[0]


def uv_version() -> str:
    try:
        result = subprocess.run([*_UV, "--version"], capture_output=True, text=True, check=False)
    except OSError:
        return "unavailable"
    return (result.stdout + result.stderr).strip() or "unavailable"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--source-sha", required=True, help="full 40-hex commit to build")
    parser.add_argument("--out", type=Path, help="new arm directory (execute mode only)")
    parser.add_argument(
        "--plan",
        type=Path,
        help="plan receipt to write (plan mode) or to verify (execute mode)",
    )
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--execute",
        action="store_true",
        help="build the wheels and install; without it the run only plans",
    )
    args = parser.parse_args(argv)

    try:
        guard_source(args.repo, args.source_sha)
        if not args.execute:
            if args.out is not None:
                raise BuildRefused(
                    "--out belongs to --execute; a plan writes only --plan and never an arm "
                    "directory (that is what keeps plan-then-execute usable)"
                )
            return _plan(args)
        if args.out is None:
            raise BuildRefused("--execute needs --out")
        return _execute(args)
    except BuildRefused as error:
        print(f"REFUSED: {error}", file=sys.stderr)
        return EXIT_REFUSED
    except Unverified as error:
        print(f"UNVERIFIED: {error}", file=sys.stderr)
        return EXIT_UNVERIFIED


def _lexical(path: str | Path) -> Path:
    """Normalise path SYNTAX only: separators, ``.`` and ``..``. Never links.

    ``resolve()`` follows symlinks, and that is precisely how a stale report from
    a foreign arm became acceptable: with a symlinked venv, two different arms
    share one resolved prefix, so ``verify_recorded_identity`` saw agreement where
    the lexical paths disagreed (QA round 1 reproduced this as rc 0 / BUILT with
    a main18 report inside a scratch arm). Discovery's own guard already refuses
    an adapter launch path that resolves through a symlink; this check matches
    that rule instead of relaxing it.
    """

    return Path(os.path.normpath(str(path)))


def verify_recorded_identity(
    recorded: Mapping[str, Any],
    *,
    source_sha: str,
    layout: ArmLayout,
    interpreter_facts: Mapping[str, Any] | None = None,
) -> list[str]:
    """Refusals if the verification report describes a DIFFERENT build.

    A record may not inherit an identity it did not verify: an arm that reports a
    commit, interpreter or venv prefix the verifier did not actually see would
    hand the caller a receipt for a previous source (review round 3's stale
    identity concern, QA round 1's symlink bypass). Comparisons are LEXICAL, so
    only a report naming this arm's own paths passes.
    """

    refusals: list[str] = []
    if not recorded:
        return ["no installed-verification report was produced, so nothing is verified"]
    verified_commit = recorded.get("commit")
    if verified_commit != source_sha:
        refusals.append(
            f"the verification report covers {verified_commit}, not this build's {source_sha}"
        )
    # The arm must own its venv directory. A symlinked one is either a hand-made
    # link (the pre-fix workaround) or a borrowed interpreter whose reported
    # prefix belongs to another arm; neither is this build.
    if layout.venv.is_symlink():
        refusals.append(f"{layout.venv} is a symlink; an arm must own its venv, not borrow one")
    prefix = recorded.get("prefix")
    if prefix is None:
        refusals.append("the verification report names no interpreter prefix")
    else:
        text = str(prefix)
        if not os.path.isabs(text):
            refusals.append(f"the verification report's interpreter prefix is not absolute: {text}")
        elif _lexical(text) != _lexical(layout.venv):
            refusals.append(
                f"the verification report's interpreter prefix is {text}, "
                f"not this arm's {layout.venv}"
            )
    # ``python`` in the report is a VERSION (``sys.version.split()[0]``), not a
    # path: comparing it to the arm's own reported version is the whole check.
    version = recorded.get("python")
    expected_version = (interpreter_facts or {}).get("reported_version")
    if version is None:
        refusals.append("the verification report names no interpreter version")
    elif expected_version:
        want = str(expected_version).split()[-1]
        if str(version) != want:
            refusals.append(
                f"the verification report was produced by Python {version}, "
                f"not this arm's {want}"
            )
    return refusals


def _create_exclusive(path: Path, data: bytes) -> None:
    """Create ``path`` with ``data`` or fail; never touches an existing file."""

    with path.open("xb") as handle:
        handle.write(data)


def write_build_record(
    layout: ArmLayout, record: Mapping[str, Any], *, publish: bool
) -> dict[str, Any]:
    """Write the record to the receipts path, and to the CALLER path only on success.

    Two rules, both learned from review round 1:

    * **Publication is conditional.** ``<arm>/build-record.json`` is what the
      launcher reads (``json.load(...)['package_digest']`` then ``[ -n "$PKG" ]``),
      and a refused build would publish ``package_digest: null`` -- which prints
      as the non-empty string ``None`` and passes that guard. So a refused build
      writes refusal evidence under ``receipts/`` ONLY, and the caller-facing
      path stays absent: the launcher's own read then fails, which is the honest
      signal.
    * **Two paths are committed together or not at all.** Both destinations are
      preflighted before either is created, and anything this call created is
      removed if a later create or the byte-for-byte read-back fails. A
      pre-existing record is never removed or overwritten.
    """

    payload = json.dumps(record, indent=2) + "\n"
    data = payload.encode("utf-8")
    receipts_path = layout.receipts / "build-record.json"
    canonical = layout.root / "build-record.json"
    targets = [receipts_path, *([canonical] if publish else [])]
    for path in targets:
        if path.exists():
            raise BuildRefused(f"{path} already exists; an arm's build record is immutable")
    created: list[Path] = []
    try:
        for path in targets:
            _create_exclusive(path, data)
            created.append(path)
            # Byte-exact, not read_text(): universal newlines would call a CRLF
            # rewrite identical to the payload, and two records that differ by a
            # line ending are two records (review round 1, R1-4).
            if path.read_bytes() != data:
                raise BuildRefused(f"{path} is not byte-identical to the record it should hold")
    except BaseException:
        for path in created:
            try:
                path.unlink()
            except OSError:  # pragma: no cover - nothing else we can do here
                pass
        raise
    return {
        "record_contract": _RECORD_CONTRACT,
        "published": publish,
        "canonical": str(canonical) if publish else None,
        "receipts": str(receipts_path),
        "bytes": len(data),
    }


def _execute(args: argparse.Namespace) -> int:
    layout = ArmLayout(root=args.out.expanduser().resolve())
    # Volatile-root refusal BEFORE create: a refusal must not leave a directory
    # behind that makes the next run look like a reused arm.
    refuse_volatile(layout.root, "arm root")
    layout.create_arm()
    materialise_source(args.repo, args.source_sha, layout.source)
    metadata = metadata_from_source(layout.source)
    overrides, core_requirements = policy(metadata)
    (layout.receipts / "policy-provenance.json").write_text(
        json.dumps(
            {
                "source_commit": args.source_sha,
                "source": f"git archive {args.source_sha}",
                "overrides": overrides,
                # The policy is applied as a resolver rule; no wheel metadata is
                # rewritten, so a raw pip check keeps reporting the two bounds.
                "metadata_rewrite": False,
                **plan_metadata_digests(metadata),
            },
            indent=2,
        )
        + "\n"
    )
    adapter_rows, _dropped = export_adapter_requirements(metadata, layout.receipts)
    interpreter_facts = create_arm_venv(layout.venv, layout.receipts)
    adapter_lock = lock_versions(metadata.adapter_lock.read_text())
    constraints = _constraint_versions(adapter_lock, overrides)
    install_overrides, pins, facts = resolve_policy_project(
        metadata=metadata,
        receipts=layout.receipts,
        adapter_rows=adapter_rows,
        core_requirements=core_requirements,
        overrides=overrides,
        constraint_versions=constraints,
        interpreter=layout.python,
    )
    plan_binding = verify_plan(args.plan, facts, args.source_sha, layout.receipts)
    wheels = build_wheels(layout)
    install_arm(
        interpreter=layout.python,
        pins=pins,
        overrides=install_overrides,
        receipts=layout.receipts,
        wheels=wheels,
    )
    refusals: list[str] = []
    # Both checks run under the ARM interpreter: the installed metadata and
    # marker environment that matter are that interpreter's, not this one's.
    for script, extra, out in (
        (
            "osworld_dependency_admission.py",
            [
                "--adapter-pyproject",
                str(metadata.adapter_pyproject),
                "--core-pyproject",
                str(metadata.core_pyproject),
            ],
            str(layout.receipts),
        ),
        (
            "osworld_arm_verify.py",
            ["--source", str(layout.source), "--commit", args.source_sha],
            str(layout.receipts / "installed-verification.json"),
        ),
    ):
        result = subprocess.run(
            [
                str(layout.python),
                "-I",
                "-B",
                str(Path(__file__).resolve().parent / script),
                *extra,
                "--out",
                out,
            ],
            capture_output=True,
            text=True,
            check=False,
            env=_child_env(),
        )
        print((result.stdout + result.stderr).strip())
        if result.returncode != 0:
            refusals.append(f"{script} refused (rc={result.returncode})")
    verification_path = layout.receipts / "installed-verification.json"
    recorded: dict[str, Any] = {}
    if verification_path.is_file():
        try:
            loaded = json.loads(verification_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            # A report we cannot parse is NOT a verified arm, and letting the
            # traceback escape would skip the refusal evidence entirely.
            refusals.append(
                f"the installed-verification report is not readable JSON ({error}); "
                f"it is kept at {verification_path}"
            )
        else:
            if isinstance(loaded, dict):
                recorded = loaded
            else:
                refusals.append(
                    f"the installed-verification report is not a JSON object; "
                    f"it is kept at {verification_path}"
                )
    planned = {
        name: version
        for name, version in lock_versions(
            (layout.receipts / "resolution-project" / "uv.lock").read_text()
        ).items()
        if name != _RESOLUTION_PROJECT
    }
    if recorded.get("installed_distributions"):
        refusals.extend(
            compare_closure(
                pinned=planned,
                installed=recorded["installed_distributions"],
                allowed_extra=[_HARNESS_DISTRIBUTION, _ADAPTER_DISTRIBUTION],
            )
        )
    # A record that outlives a refusal is what the verifier's own report-on-refusal
    # fix buys; it never inherits an identity it did not verify.
    refusals.extend(
        verify_recorded_identity(
            recorded,
            source_sha=args.source_sha,
            layout=layout,
            # Without this the version binding is dead code: the report's `python`
            # field is only comparable to the arm interpreter this run created
            # (review round 2, R2-1).
            interpreter_facts=interpreter_facts,
        )
    )
    written = write_build_record(
        layout,
        {
            "published": not refusals,
            "record_contract": _RECORD_CONTRACT,
            "arm_source_commit": args.source_sha,
            # The verifier's own facts, so a record cannot report a previous
            # source: these are read back from THIS run's report, and
            # verify_recorded_identity refused unless they name this build.
            "verified_commit": recorded.get("commit"),
            "verified_venv_prefix": recorded.get("prefix"),
            "harness": recorded.get("harness"),
            "adapter": recorded.get("adapter"),
            "package_digest": (recorded.get("adapter") or {}).get("package_digest"),
            "package_digest_scope": ("this install only (RECORD includes installer-written rows)"),
            "cross_build_identity": {
                "source_commit": args.source_sha,
                # The verifier writes core_module_digest and core_source; a key
                # it never writes would silently record null here (round 2).
                "core_module_digest": recorded.get("core_module_digest"),
                "core_module_hashes": _module_hashes(recorded.get("core_source")),
                "source_record_digest": (recorded.get("adapter") or {}).get("source_record_digest"),
            },
            "interpreter": interpreter_facts,
            "requirements_installed": str(pins),
            "installation_overrides": install_overrides,
            "policy_admission": str(layout.receipts / "policy-admission.json"),
            "raw_pip_check": str(layout.receipts / "raw-pip-check.log"),
            "verification": str(verification_path),
            "plan_binding": plan_binding,
            "uv_version": uv_version(),
            "closure_refusals": [
                reason
                for reason in refusals
                if "not in the resolution plan" in reason or "differs from planned" in reason
            ],
            "refusals": refusals,
            # NOT identities: zip timestamps differ between builds of the same
            # commit (this module's docstring, point 3).
            "wheel_sha256_informational_only": {
                "core": instance_sha256(wheels[0]),
                "adapter": instance_sha256(wheels[1]),
            },
        },
        publish=not refusals,
    )
    if written["published"]:
        print(
            f"RECORD: {written['canonical']} "
            f"(identical copy: {written['receipts']}, {written['bytes']} bytes)"
        )
    else:
        # Deliberately no canonical file: the launcher must not be able to read a
        # digest for an arm that failed verification or admission (R1-1).
        print(
            f"REFUSAL EVIDENCE: {written['receipts']} "
            f"({written['bytes']} bytes); no canonical build record published"
        )
    if refusals:
        raise Unverified("; ".join(refusals))
    print(f"BUILT: {layout.root} (receipts in {layout.receipts})")
    return 0


def verify_plan(
    plan_path: Path | None, facts: Mapping[str, Any], source_sha: str, receipts: Path
) -> dict[str, Any]:
    """Check this build against the plan it was given, if one was given.

    A plan is only worth writing if the build is bound to it: the digest covers
    the resolved installed set and the override policy, so a resolution that
    drifted between plan and build (a moved index, a new upstream release inside
    a range, a policy edit) is refused instead of silently installed.
    """

    if plan_path is None:
        return {"given": False, "verified": False}
    if not plan_path.is_file():
        raise BuildRefused(f"--plan {plan_path} does not exist")
    planned = json.loads(plan_path.read_text())
    if planned.get("source_commit") != source_sha:
        raise BuildRefused(
            f"plan {plan_path} was made for {planned.get('source_commit')}, not {source_sha}"
        )
    if planned.get("plan_digest") != facts["plan_digest"]:
        raise BuildRefused(
            "the resolution differs from the plan "
            f"({planned.get('plan_digest')} != {facts['plan_digest']}); re-plan, then build"
        )
    binding = {
        "given": True,
        "verified": True,
        "plan": str(plan_path),
        "plan_digest": facts["plan_digest"],
        "planned_package_count": planned.get("resolution_package_count"),
    }
    (receipts / "plan-binding.json").write_text(json.dumps(binding, indent=2) + "\n")
    return binding


if __name__ == "__main__":
    raise SystemExit(main())
