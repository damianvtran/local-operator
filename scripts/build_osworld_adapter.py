#!/usr/bin/env python3
"""Materialise the OSWorld V2 adapter workspace from verified durable inputs.

This is an operator command, NOT a CI job: the task corpus and assets are
gated Hugging Face datasets that a human fetches once (with an accepted access
request and an ``HF_TOKEN``) into a DURABLE inputs root. This script never
downloads. It verifies what is on disk against the committed release pin and
copies the verified task bytes into a workspace; the assets stay in the
inputs root.

WHY THE INPUTS ROOT MUST NOT BE UNDER /tmp. macOS purges ``/private/tmp`` on
disk pressure and on a periodic sweep with no warning and no regard for open
handles. A purge mid-run destroyed a previous paid pilot's prepared checkout,
its 4.2 GB asset snapshot AND its output directory, and left an EC2 instance
running. The default root is ``~/worktrees/osworld``; ``--inputs-root`` (or
``$OSWORLD_INPUTS_ROOT``) may move it anywhere durable, because every input
is content-hash verified wherever it lives.

Verification, in order, each failing with exit 4 and the offending path:

1. ``<inputs>/prepared/benchmark_releases/<release>.json`` sha256 equals the
   pin's ``release_manifest_sha256``.
2. ``<inputs>/gated/tasks/manifests/task_hashes.json`` sha256 equals the
   pin's ``tasks.hash_manifest_sha256``.
3. every ``task_NNN.py`` in that manifest exists, its sha256 matches, and the
   count equals the pin's ``task_count``.
4. ``git -C <inputs>/prepared rev-parse HEAD`` equals the pin's
   ``osworld.commit`` (when git is available; a checkout without ``.git`` is
   refused rather than trusted).
5. ``<inputs>/gated/manifests/assets.json`` exists (its sha is recorded, not
   compared -- the pin carries the repository revision, and the manifest
   records that revision; the episode-time re-verification in the adapter
   compares against what THIS build recorded).

What the workspace gets, all read-only, copied never linked
(``discovery.workspace_digest`` rejects symlinks and hardlinks):

- ``adapter-release.json`` -- exactly ``{"release_digest": "<64 hex>"}``,
  which ``discovery.verify_release_manifest`` requires and the worker refuses
  to launch without.
- ``benchmark_release.json`` -- the V2 release manifest, verified.
- ``task_hashes.json`` -- the task-hash manifest, verified.
- ``tasks/task_*.py`` -- the verified task bytes, so ``workspace_digest``
  pins exactly what runs.
- ``adapter-provider.json`` -- ``{"provider": "aws"}``.
- ``inputs.json`` -- the manifests' sha256s and the prepared commit, which the
  adapter re-verifies against the live inputs root at ``reset_start`` so a
  moved or edited root cannot change what runs.

The assets (4.2 GB) are NOT copied: the workspace cap is 4 GiB
(``discovery.MAX_WORKSPACE_BYTES``) and the guest reads them through
``OSWORLD_FILE_BASE_URL``, never the worker.

Usage:
    python scripts/build_osworld_adapter.py \\
        --benchmark-release osworld-v2-2026.08.08 \\
        --out ~/worktrees/osworld/workspaces/0.1.0 \\
        --package-digest <digest of the installed adapter wheel>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any

from local_operator.evaluation.runner import durable_root

EXIT_VERIFY = 4
_DEFAULT_INPUTS_ROOT = "~/worktrees/osworld"
_DEFAULT_PIN = (
    Path(__file__).resolve().parents[1]
    / "benchmarks"
    / "osworld_v2_adapter"
    / "config"
    / "release-v2026.08.08.json"
)


_ADAPTER_PYPROJECT = (
    Path(__file__).resolve().parents[1] / "benchmarks" / "osworld_v2_adapter" / "pyproject.toml"
)


def _adapter_version() -> str | None:
    """The adapter distribution version declared in its pyproject, or None.

    Read from source rather than from ``importlib.metadata`` on purpose: this
    script runs from the repository against a workspace that is built BEFORE
    (or without) the wheel being installed into the caller's interpreter, so
    the installed distribution is the wrong authority and may not exist at all.

    Returns None rather than a placeholder when the declaration cannot be read.
    A placeholder would flow into ``_release_digest`` and mint a real,
    well-formed workspace attesting a version that does not exist -- and since
    ``adapter-release.json`` carries ONLY the digest (``discovery`` rejects any
    other key), the artifact could not record its own doubt even if we wanted
    it to. An unattributable attestation is worse than no artifact, so the
    caller refuses to build instead.
    """

    try:
        return str(tomllib.loads(_ADAPTER_PYPROJECT.read_text())["project"]["version"])
    except (OSError, KeyError, tomllib.TOMLDecodeError):
        return None


class VerificationFailed(Exception):
    """An input does not match the pin. The message names the path."""


def refuse_volatile_root(inputs_root: Path) -> None:
    """Fail fast if the inputs root lives somewhere the OS may purge.

    The check itself is ``runner.durable_root.refuse_volatile_root`` -- shared
    with ``scripts/run_episode.py`` so the build and the episode agree on what
    "volatile" means -- re-raised here as ``VerificationFailed`` so this
    script's callers keep one exception type and one exit code (4).
    """

    try:
        durable_root.refuse_volatile_root(inputs_root, label="inputs root")
    except durable_root.VolatileRootError as error:
        raise VerificationFailed(str(error)) from error


def _release_digest(
    *, version: str, package_digest: str, benchmark_release: str, task_manifest_sha256: str
) -> str:
    """Our attestation of the build, tying the harness build to the benchmark
    release — the claim a leaderboard number must carry."""

    payload = (
        f"lop-osworld-v2-adapter|{version}|{package_digest}|"
        f"{benchmark_release}|{task_manifest_sha256}"
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_readonly(path: Path, data: bytes) -> None:
    # A previous build's read-only file would refuse the overwrite; the
    # workspace is rebuilt from verified inputs, so replacing is correct.
    if path.exists():
        path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    path.write_bytes(data)
    # The workspace is immutable once built: a task byte that changed after
    # the digest was computed would make the evidence pin a lie.
    path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)


def _prepared_commit(prepared: Path) -> str:
    if not (prepared / ".git").exists():
        raise VerificationFailed(f"{prepared} is not a git checkout; cannot verify its commit")
    try:
        result = subprocess.run(
            ["git", "-C", str(prepared), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise VerificationFailed(f"git rev-parse failed in {prepared}: {error}") from error
    return result.stdout.strip()


def verify_inputs(inputs_root: Path, pin: dict[str, Any]) -> dict[str, Any]:
    """Verify every input against the pin; return the facts the build records."""

    release = pin["release"]
    prepared = inputs_root / "prepared"
    gated = inputs_root / "gated"

    release_manifest = prepared / "benchmark_releases" / f"{release}.json"
    if not release_manifest.is_file():
        raise VerificationFailed(f"release manifest missing: {release_manifest}")
    actual = _sha256(release_manifest)
    if actual != pin["release_manifest_sha256"]:
        raise VerificationFailed(f"release manifest sha256 mismatch: {release_manifest}")

    hash_manifest = gated / "tasks" / pin["tasks"]["hash_manifest_path"]
    if not hash_manifest.is_file():
        raise VerificationFailed(f"task hash manifest missing: {hash_manifest}")
    hash_manifest_sha = _sha256(hash_manifest)
    if hash_manifest_sha != pin["tasks"]["hash_manifest_sha256"]:
        raise VerificationFailed(f"task hash manifest sha256 mismatch: {hash_manifest}")

    manifest = json.loads(hash_manifest.read_bytes())
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise VerificationFailed(f"task hash manifest has no 'files' map: {hash_manifest}")
    expected_count = int(pin["tasks"]["task_count"])
    if len(files) != expected_count:
        raise VerificationFailed(
            f"task hash manifest lists {len(files)} tasks, pin says {expected_count}: "
            f"{hash_manifest}"
        )
    tasks_dir = gated / "tasks"
    task_bytes: dict[str, bytes] = {}
    for name in sorted(files):
        entry = files[name]
        path = tasks_dir / name
        if not path.is_file():
            raise VerificationFailed(f"task file missing: {path}")
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != entry.get("sha256"):
            raise VerificationFailed(f"task file sha256 mismatch: {path}")
        size = entry.get("size")
        if size is not None and int(size) != len(data):
            raise VerificationFailed(f"task file size mismatch: {path}")
        task_bytes[name] = data

    commit = _prepared_commit(prepared)
    if commit != pin["osworld"]["commit"]:
        raise VerificationFailed(
            f"prepared checkout HEAD {commit[:12]} != pinned "
            f"{pin['osworld']['commit'][:12]}: {prepared}"
        )

    assets_manifest = gated / "manifests" / "assets.json"
    if not assets_manifest.is_file():
        raise VerificationFailed(f"assets manifest missing: {assets_manifest}")
    assets_payload = json.loads(assets_manifest.read_bytes())
    if assets_payload.get("revision") != pin["assets"]["revision"]:
        raise VerificationFailed(
            f"assets manifest revision {assets_payload.get('revision')!r} != pinned "
            f"{pin['assets']['revision']!r}: {assets_manifest}"
        )

    return {
        "release_manifest_bytes": release_manifest.read_bytes(),
        "hash_manifest_bytes": hash_manifest.read_bytes(),
        "hash_manifest_sha256": hash_manifest_sha,
        "assets_manifest_sha256": _sha256(assets_manifest),
        "prepared_commit": commit,
        "tasks": task_bytes,
    }


def build_workspace(
    *,
    out: Path,
    facts: dict[str, Any],
    release: str,
    version: str,
    package_digest: str,
) -> str:
    """Write the workspace files; return the release digest."""

    out.mkdir(parents=True, exist_ok=True)
    tasks_out = out / "tasks"
    tasks_out.mkdir(exist_ok=True)
    release_digest = _release_digest(
        version=version,
        package_digest=package_digest,
        benchmark_release=release,
        task_manifest_sha256=facts["hash_manifest_sha256"],
    )
    canonical = {"separators": (",", ":"), "sort_keys": True}
    _write_readonly(
        out / "adapter-release.json",
        json.dumps({"release_digest": release_digest}, **canonical).encode(),
    )
    _write_readonly(out / "benchmark_release.json", facts["release_manifest_bytes"])
    _write_readonly(out / "task_hashes.json", facts["hash_manifest_bytes"])
    _write_readonly(
        out / "adapter-provider.json", json.dumps({"provider": "aws"}, **canonical).encode()
    )
    _write_readonly(
        out / "inputs.json",
        json.dumps(
            {
                "assets_manifest_sha256": facts["assets_manifest_sha256"],
                "tasks_manifest_sha256": facts["hash_manifest_sha256"],
                "prepared_commit": facts["prepared_commit"],
            },
            **canonical,
        ).encode(),
    )
    for name, data in facts["tasks"].items():
        _write_readonly(tasks_out / name, data)
    return release_digest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-release", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--inputs-root",
        type=Path,
        default=Path(os.environ.get("OSWORLD_INPUTS_ROOT", _DEFAULT_INPUTS_ROOT)),
    )
    parser.add_argument("--release-pin", type=Path, default=_DEFAULT_PIN)
    # Left with NO argparse default: the adapter's own declared version is
    # resolved below so that "not passed" stays distinguishable from "passed
    # a value equal to the default". The version is an input to
    # ``_release_digest``, so a value that does not match the distribution
    # actually built mints a workspace attesting a version that never existed
    # -- a wrong attestation no digest check can catch, because every digest
    # is internally consistent with it. That already happened once: the
    # literal default still said 0.1.0 after the 0.1.1 bump.
    parser.add_argument("--version", default=None)
    parser.add_argument(
        "--allow-version-mismatch",
        action="store_true",
        help="permit --version to differ from the adapter's declared version "
        "(for attesting a build of a version other than this tree's)",
    )
    parser.add_argument("--package-digest", default="0" * 64)
    args = parser.parse_args(argv)

    # Resolve the attested version under one rule: it must agree with what the
    # tree declares, unless the operator says out loud that it should not.
    # Building a version OTHER than the tree's is legitimate (re-attesting an
    # older wheel against the same corpus), but it is a deliberate act, not
    # something to accept silently -- silently accepting a mismatched value is
    # the same shape as the stale-default defect this script already had.
    declared = _adapter_version()
    version = args.version if args.version is not None else declared
    if version is None:
        print(
            "build_osworld_adapter: cannot determine the adapter version "
            f"({_ADAPTER_PYPROJECT} is unreadable or declares none) and no "
            "--version was given. Refusing to build: the version is an input "
            "to release_digest, and adapter-release.json carries only that "
            "digest, so the workspace could not record that its attestation "
            "is unattributable.",
            file=sys.stderr,
        )
        return 2
    if args.version is not None and declared is not None and args.version != declared:
        if not args.allow_version_mismatch:
            print(
                f"build_osworld_adapter: --version {args.version!r} disagrees with the "
                f"version the adapter declares ({declared!r} in {_ADAPTER_PYPROJECT}). "
                "release_digest attests the distribution that was built, so a "
                "mismatch produces a workspace claiming a version nobody can "
                "verify. Pass --allow-version-mismatch if that is deliberate.",
                file=sys.stderr,
            )
            return 2
        # Deliberate and stated: still say so, because the resulting digest is
        # not reproducible from this tree alone.
        print(
            f"build_osworld_adapter: attesting {args.version!r} while this tree "
            f"declares {declared!r} (--allow-version-mismatch).",
            file=sys.stderr,
        )

    pin = json.loads(Path(args.release_pin).read_bytes())
    if pin.get("release") != args.benchmark_release:
        print(
            f"release pin {args.release_pin} is for {pin.get('release')!r}, "
            f"not {args.benchmark_release!r}",
            file=sys.stderr,
        )
        return 2
    inputs_root = Path(os.path.expanduser(str(args.inputs_root)))
    try:
        refuse_volatile_root(inputs_root)
        facts = verify_inputs(inputs_root, pin)
    except VerificationFailed as error:
        print(f"build_osworld_adapter: verification failed: {error}", file=sys.stderr)
        return EXIT_VERIFY
    release_digest = build_workspace(
        out=args.out,
        facts=facts,
        release=args.benchmark_release,
        version=version,
        package_digest=args.package_digest,
    )
    print(
        json.dumps(
            {
                "workspace": str(args.out),
                "release_digest": release_digest,
                "task_count": len(facts["tasks"]),
                "prepared_commit": facts["prepared_commit"],
                "assets_manifest_sha256": facts["assets_manifest_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
