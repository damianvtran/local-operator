#!/usr/bin/env python3
"""Build one OSWorld arm from a committed SHA, then admit its dependencies.

The arm is the interpreter the paid adapter runs in: the harness wheel, the
adapter wheel, and the adapter's locked ``osworld`` extra. This script is the
durable form of the operator recipe that built the main16/main17 arms; it
encodes the three facts that recipe learned the expensive way, so none of them
has to be rediscovered:

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
   mismatch was this, not tampering). The identity recorded here is the source
   commit, the pinned core module source hashes, the adapter's installed RECORD
   digest (``package_digest``) and the installed distribution set. Archive
   hashes are kept in the record, labelled informational.

What this script does NOT do: it allocates no guest, contacts no provider or
model, runs no episode, and supports no benchmark claim. It never reuses an arm
directory -- an arm is an immutable receipt. See
``docs/benchmarks/osworld_2/arm-build.md`` for what the recipe proves.

Exit codes: 2 = refused (bad input, unmerged source, existing arm, unusable
interpreter, failed command); 4 = built but dependency admission or the
installed-identity verification refused. Receipts exist in both cases.

Usage:
    python scripts/build_osworld_arm.py --source-sha <40-hex> --out <new dir>
    python scripts/build_osworld_arm.py --source-sha <40-hex> --out <new dir> --execute
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from local_operator.evaluation.runner import durable_root  # noqa: E402
from scripts import osworld_dependency_admission as admission  # noqa: E402

EXIT_REFUSED = 2
EXIT_UNVERIFIED = 4

_ADAPTER_DIR = "benchmarks/osworld_v2_adapter"
_HARNESS_DISTRIBUTION = "local-operator"
_ADAPTER_DISTRIBUTION = "lop-osworld-v2-adapter"
_RESOLUTION_PROJECT = "lop-osworld-arm-resolution"
# The arm interpreter's own minor: the adapter requires >=3.12 and the pinned
# osworld extra is only resolved for one interpreter, so a different minor would
# need a different lock rather than a different flag.
_PYTHON_MINOR = "3.12"
# ``requires-python`` of the resolution project. ``==3.12.*`` (not ``>=``) so uv
# resolves the closure the arm interpreter will actually run.
_PYTHON_FLOOR = f"=={_PYTHON_MINOR}.*"
_SHA = re.compile(r"[0-9a-f]{40}")


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

    def create(self) -> None:
        for path in (self.source, self.core_dist, self.adapter_dist, self.receipts):
            path.mkdir(parents=True, exist_ok=True)


def run(argv: Sequence[str]) -> str:
    """Run one external command, reporting failure as a refusal, not a traceback."""

    completed = subprocess.run(list(argv), capture_output=True, text=True, check=False)
    output = (completed.stdout + completed.stderr).strip()
    if completed.returncode != 0:
        tail = output[-2000:] if output else "(no output)"
        raise BuildRefused(f"{' '.join(argv)} failed (rc={completed.returncode}): {tail}")
    return output


def instance_sha256(path: Path) -> str:
    """Hash of THIS file instance -- a receipt value, never an identity."""

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
    """The source must be one exact, MERGED commit.

    A floating ref would let a later build pick up commits nobody reviewed, and
    an arm built from unmerged source is one no one can reproduce from ``main``.
    """

    if not _SHA.fullmatch(sha):
        raise BuildRefused("--source-sha must be the full 40-character lowercase SHA")
    head = run(["git", "-C", str(repo), "rev-parse", f"{sha}^{{commit}}"]).strip()
    if head != sha:
        raise BuildRefused(f"{sha} is not a commit object in {repo}")
    merged = subprocess.run(
        ["git", "-C", str(repo), "merge-base", "--is-ancestor", sha, "origin/main"],
        capture_output=True,
        text=True,
        check=False,
    )
    if merged.returncode != 0:
        raise BuildRefused(f"{sha} is not merged into origin/main; arms are built from reviews")


def materialise_source(repo: Path, sha: str, destination: Path) -> None:
    """Extract the COMMITTED tree; the working tree is never an input.

    ``git archive`` also makes a dirty checkout harmless: what is built is what
    the commit contains, which is what the receipts claim.
    """

    destination.mkdir(parents=True, exist_ok=True)
    archive = subprocess.Popen(
        ["git", "-C", str(repo), "archive", sha],
        stdout=subprocess.PIPE,
        text=False,
    )
    if archive.stdout is None:  # pragma: no cover - defensive
        raise BuildRefused("git archive produced no output stream")
    tar = subprocess.run(
        ["tar", "-x", "-C", str(destination)],
        stdin=archive.stdout,
        capture_output=True,
        text=False,
        check=False,
    )
    archive.stdout.close()
    status = archive.wait()
    if status != 0 or tar.returncode != 0:
        raise BuildRefused(
            f"git archive {sha} could not be extracted into {destination}"
            f" (git rc={status}, tar rc={tar.returncode})"
        )


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


def lock_versions(lock_text: str) -> dict[str, str]:
    """canonical name -> version for every package a uv lock pins.

    Used twice, for one reason: the frozen export carries direct references
    (``osworld @ git+https://...``) that carry no version, so the LOCK -- not the
    exported row -- is the only place the whole planned set has a version.
    """

    lock = tomllib.loads(lock_text)
    return {admission.canonicalize_name(row["name"]): row["version"] for row in lock["package"]}


def installed_overrides(
    lock_text: str, names: tuple[str, ...] = ("requests", "pyjwt")
) -> list[str]:
    """Narrow the policy overrides to the versions the resolution chose.

    pip has no project policy, so the overrides are passed to the install
    explicitly -- but as exact pins: a floor would let an already-installed newer
    version win over the frozen choice, which is how an arm silently stops
    matching its own lock. Extras are preserved (``pyjwt[crypto]``), because the
    crypto extra is part of the harness floor.
    """

    versions = lock_versions(lock_text)
    pins: list[str] = []
    for name in names:
        if name not in versions:
            raise BuildRefused(f"the resolution lock has no {name} row")
        spelling = f"{name}[crypto]" if name == "pyjwt" else name
        pins.append(f"{spelling}=={versions[name]}")
    return pins


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
    # Mirrors discovery._symlink_free, so a symlinked venv is refused HERE with
    # the reason, instead of pre-dispatch on a paid run with only the guard text.
    if interpreter.is_symlink() or interpreter.resolve() != interpreter:
        raise BuildRefused(f"{interpreter} resolves through a symlink; discovery refuses it")
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


def export_adapter_requirements(source: Path, receipts: Path) -> tuple[list[str], list[str]]:
    """Export the built commit's adapter lock (osworld extra included) and split it."""

    export = receipts / "requirements.locked.txt"
    run(
        [
            "uv",
            "export",
            "--project",
            str(source / _ADAPTER_DIR),
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


def resolve_policy_project(
    *,
    layout: ArmLayout,
    adapter_rows: Sequence[str],
    core_requirements: Sequence[str],
    overrides: Sequence[str],
    constraint_versions: Mapping[str, str],
    interpreter: Path,
) -> tuple[list[str], Path, dict[str, str]]:
    """Lock and export the synthetic project; return install overrides and pins."""

    project = layout.receipts / "resolution-project"
    project.mkdir(exist_ok=True)
    (project / "pyproject.toml").write_text(
        resolution_project(
            adapter_rows=adapter_rows,
            core_requirements=core_requirements,
            overrides=overrides,
            constraint_versions=constraint_versions,
        )
    )
    run(["uv", "lock", "--project", str(project), "--python", str(interpreter)])
    pins = layout.receipts / "policy-pins.txt"
    run(
        [
            "uv",
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
    return installed_overrides(lock_text), pins, lock_versions(lock_text)


def build_wheels(layout: ArmLayout) -> tuple[Path, Path]:
    """Build both wheels from the EXTRACTED commit, never from a working tree.

    The adapter must be a wheel, never ``-e``: an editable install's RECORD
    covers only the ``.pth`` shim and metadata, so ``discovery`` reports "adapter
    entry module is not uniquely RECORD-covered" -- an error that names RECORD
    coverage rather than the editable install that caused it.
    """

    run(["uv", "build", "--wheel", "--out-dir", str(layout.core_dist), str(layout.source)])
    run(
        [
            "uv",
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
            "uv",
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
        planned = pinned.get(key)
        if planned is None:
            refusals.append(f"installed {name} {version} is not in the resolution plan")
        elif planned != version:
            refusals.append(f"installed {name} {version} differs from planned {planned}")
    return refusals


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("--source-sha", required=True, help="full 40-hex commit to build")
    parser.add_argument("--out", required=True, type=Path, help="new arm directory (never reused)")
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--execute",
        action="store_true",
        help="build the wheels and install; without it only the resolution plan is recorded",
    )
    args = parser.parse_args(argv)

    layout = ArmLayout(root=args.out.expanduser().resolve())
    try:
        guard_source(args.repo, args.source_sha)
        # Checked BEFORE create(): a durable-root refusal must not leave a
        # directory behind that makes the next run look like a reused arm.
        if layout.root.exists():
            raise BuildRefused(f"{layout.root} exists; an arm receipt is immutable")
        refuse_volatile(layout.root, "arm root")
        layout.create()
        materialise_source(args.repo, args.source_sha, layout.source)
        adapter_pyproject = tomllib.loads(
            (layout.source / _ADAPTER_DIR / "pyproject.toml").read_text()
        )
        core_pyproject = tomllib.loads((layout.source / "pyproject.toml").read_text())
        overrides = list(adapter_pyproject["tool"]["uv"]["override-dependencies"])
        core_requirements = list(core_pyproject["project"]["dependencies"])
        # Provenance BEFORE anything is installed: these are the exact policy
        # bytes the admission verdict applies to, hashed from the commit rather
        # than from any working tree.
        (layout.receipts / "policy-provenance.json").write_text(
            json.dumps(
                {
                    "source_commit": args.source_sha,
                    "source": f"git archive {args.source_sha}",
                    "overrides": overrides,
                    "core_metadata_sha256": instance_sha256(layout.source / "pyproject.toml"),
                    "adapter_metadata_sha256": instance_sha256(
                        layout.source / _ADAPTER_DIR / "pyproject.toml"
                    ),
                    "frozen_adapter_lock_sha256": instance_sha256(
                        layout.source / _ADAPTER_DIR / "uv.lock"
                    ),
                    # The policy is applied as a resolver rule; no wheel metadata
                    # is rewritten, so a raw pip check keeps reporting the two
                    # superseded bounds.
                    "metadata_rewrite": False,
                },
                indent=2,
            )
            + "\n"
        )
        adapter_rows, _dropped = export_adapter_requirements(layout.source, layout.receipts)
        interpreter_facts = create_arm_venv(layout.venv, layout.receipts)
        adapter_lock = tomllib.loads((layout.source / _ADAPTER_DIR / "uv.lock").read_text())
        lock_versions = {row["name"]: row["version"] for row in adapter_lock["package"]}
        # The two overridden packages stay pinned at the versions the committed
        # adapter lock already froze, so a policy floor cannot move them.
        constraint_versions = {
            name: lock_versions[name] for name in ("requests", "pyjwt") if name in lock_versions
        }
        missing = {"requests", "pyjwt"} - set(constraint_versions)
        if missing:
            raise BuildRefused(f"the committed adapter lock has no row for {sorted(missing)}")
        install_overrides, pins, plan = resolve_policy_project(
            layout=layout,
            adapter_rows=adapter_rows,
            core_requirements=core_requirements,
            overrides=overrides,
            constraint_versions=constraint_versions,
            interpreter=layout.python,
        )
        if not args.execute:
            print(f"PLAN ONLY: resolution receipts in {layout.receipts}; nothing installed")
            return 0
        wheels = build_wheels(layout)
        install_arm(
            interpreter=layout.python,
            pins=pins,
            overrides=install_overrides,
            receipts=layout.receipts,
            wheels=wheels,
        )
        # Both checks run under the ARM interpreter: the installed metadata and
        # marker environment that matter are that interpreter's, not this one's.
        for script, out in (
            ("osworld_dependency_admission.py", str(layout.receipts)),
            ("osworld_arm_verify.py", str(layout.receipts / "installed-verification.json")),
        ):
            result = subprocess.run(
                [
                    str(layout.python),
                    "-I",
                    "-B",
                    str(Path(__file__).resolve().parent / script),
                    *(
                        [
                            "--adapter-pyproject",
                            str(layout.source / _ADAPTER_DIR / "pyproject.toml"),
                            "--core-pyproject",
                            str(layout.source / "pyproject.toml"),
                        ]
                        if script.startswith("osworld_dependency")
                        else ["--source", str(layout.source), "--commit", args.source_sha]
                    ),
                    "--out",
                    out,
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            print((result.stdout + result.stderr).strip())
            if result.returncode != 0:
                raise Unverified(f"{script} refused (rc={result.returncode})")
        recorded: dict[str, Any] = json.loads(
            (layout.receipts / "installed-verification.json").read_text()
        )
        # The verifier already refused a wrong source commit or a wrong module
        # hash; this is the second question -- does the installed SET match the
        # plan the receipts describe?
        refusals = compare_closure(
            # The resolution project's own row is not installed, by design.
            pinned={k: v for k, v in plan.items() if k != _RESOLUTION_PROJECT},
            installed=recorded["installed_distributions"],
            allowed_extra=[_HARNESS_DISTRIBUTION, _ADAPTER_DISTRIBUTION],
        )
        (layout.receipts / "build-record.json").write_text(
            json.dumps(
                {
                    "arm_source_commit": args.source_sha,
                    "harness": recorded["harness"],
                    "adapter": recorded["adapter"],
                    "package_digest": recorded["adapter"]["package_digest"],
                    "interpreter": interpreter_facts,
                    "requirements_installed": str(pins),
                    "installation_overrides": install_overrides,
                    "policy_admission": str(layout.receipts / "policy-admission.json"),
                    "raw_pip_check": str(layout.receipts / "raw-pip-check.log"),
                    "verification": str(layout.receipts / "installed-verification.json"),
                    "closure_refusals": refusals,
                    # NOT an identity: zip timestamps differ between builds of
                    # the same commit (this module's docstring, point 3).
                    "wheel_sha256_informational_only": {
                        "core": instance_sha256(wheels[0]),
                        "adapter": instance_sha256(wheels[1]),
                    },
                },
                indent=2,
            )
            + "\n"
        )
        if refusals:
            raise Unverified("; ".join(refusals))
    except BuildRefused as error:
        print(f"REFUSED: {error}", file=sys.stderr)
        return EXIT_REFUSED
    except Unverified as error:
        print(f"UNVERIFIED: {error}", file=sys.stderr)
        return EXIT_UNVERIFIED
    print(f"BUILT: {layout.root} (receipts in {layout.receipts})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
