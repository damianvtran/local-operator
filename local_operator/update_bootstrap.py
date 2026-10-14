"""Verified normal-caller bootstrap, separate from install-kind classification.

The UI/stock shim ships this same reviewed seed. It obtains published bytes before
loading a new controller; an isolated pip/uv environment NEVER becomes UV_TOOL by
an override. Only a positively validated canonical generation destination grants
the narrow controller operation. Existing install transactions perform mutation.
This module is intentionally stdlib-only until the shared bounded worker boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import sys
import tempfile
import time
import zipfile
from dataclasses import dataclass
from email.parser import BytesParser
from pathlib import Path
from urllib.parse import quote, urlparse
from urllib.request import urlopen

CALLER_CONTRACT = "rolling-no-kill-v1"
CONTROLLER_CONTRACT = "immutable-controller-v1"
MAX_ARTIFACT_BYTES = 128 * 1024 * 1024
PUBLICATION_TIMEOUT_S = 5.0
BOOTSTRAP_TIMEOUT_S = 300.0


class BootstrapError(RuntimeError):
    """An ordinary actionable refusal; never a permission to force an update."""


@dataclass(frozen=True)
class Destination:
    root: Path
    baseline: Path
    device: int
    inode: int


def destination(root: str, baseline: str) -> Destination:
    """Read-only initial proof; the controller revalidates inside the root lock."""
    if os.name == "nt":
        raise BootstrapError(
            "rolling controller destination is unsupported on Windows; no unsafe lock fallback"
        )
    canonical = Path.home() / ".local" / "share" / "lop"
    requested = Path(root).expanduser()
    if requested != canonical or requested.is_symlink():
        raise BootstrapError(
            "rolling destination must be the canonical user core install, not an app/profile root"
        )
    info = requested.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid():
        raise BootstrapError("core install root is not an owned directory")
    if requested.resolve() != canonical.absolute():
        raise BootstrapError("core root aliases another tree; no update was started")
    pointer = requested / "current"
    if not pointer.is_symlink():
        raise BootstrapError(
            "core install has no immutable current pointer; mutable updates require idle admission"
        )
    current = pointer.resolve(strict=True)
    expected = Path(baseline).resolve(strict=True)
    generations = requested / "generations"
    if current != expected or current.parent != generations or current.is_symlink():
        raise BootstrapError("current changed or is not an owned generation; check again and retry")
    image = current / "tools" / "local-operator"
    if not (image / "uv-receipt.toml").is_file():
        raise BootstrapError(
            "destination generation is not a uv-tool install; no kind override is allowed"
        )
    return Destination(requested, current, info.st_dev, info.st_ino)


def _read_url(url: str, *, artifact: bool = False) -> bytes:
    expected = "files.pythonhosted.org" if artifact else "pypi.org"
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname != expected:
        raise BootstrapError("publication/artifact URL is not an approved HTTPS origin")
    limit = MAX_ARTIFACT_BYTES if artifact else 2 * 1024 * 1024
    # noqa: S310 — origins constrained above and rechecked on the final URL below
    with urlopen(url, timeout=PUBLICATION_TIMEOUT_S) as response:
        final = urlparse(response.geturl())
        if final.scheme != "https" or final.hostname != expected:
            raise BootstrapError("publication redirected to an unapproved origin")
        chunks: list[bytes] = []
        size = 0
        deadline = time.monotonic() + (30.0 if artifact else PUBLICATION_TIMEOUT_S)
        while True:
            if time.monotonic() >= deadline:
                raise BootstrapError("publication/controller download budget expired")
            chunk = response.read(min(64 * 1024, limit + 1 - size))
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
            if size > limit:
                raise BootstrapError("published controller exceeds the bounded artifact size")
        data = b"".join(chunks)
    if len(data) > limit:
        raise BootstrapError("published controller exceeds the bounded artifact size")
    return data


def _publication(target: str | None) -> tuple[str, str, str, str]:
    endpoint = "https://pypi.org/pypi/local-operator/"
    suffix = quote(target, safe="") + "/" if target else ""
    document = json.loads(_read_url(endpoint + suffix + "json"))
    if not isinstance(document, dict) or not isinstance(document.get("info"), dict):
        raise BootstrapError("publication metadata is malformed; nothing was loaded")
    version = document["info"].get("version")
    if not isinstance(version, str) or (target is not None and version != target):
        raise BootstrapError("published controller version cannot be positively confirmed")
    files = document.get("urls")
    if not isinstance(files, list):
        raise BootstrapError("publication has no bounded artifact list")
    for item in files:
        if not isinstance(item, dict) or item.get("yanked"):
            continue
        name = item.get("filename", "")
        if not isinstance(name, str) or Path(name).name != name or "/" in name or "\\" in name:
            continue
        compatible = (
            (
                name.endswith("-py3-none-any.whl")
                if sys.platform != "darwin"
                else name.endswith("-py3-none-macosx_11_0_universal2.whl")
            )
            if isinstance(name, str)
            else False
        )
        hashes = item.get("digests")
        digest = hashes.get("sha256", "") if isinstance(hashes, dict) else ""
        url = item.get("url", "")
        identified_url = isinstance(url, str)
        if (
            compatible
            and isinstance(digest, str)
            and identified_url
            and re.fullmatch(r"[0-9a-f]{64}", digest)
        ):
            return version, name, url, digest
    raise BootstrapError("no compatible non-yanked hash-identified controller wheel is published")


def verify_controller(wheel: Path, digest: str) -> tuple[str, str]:
    """Verify loaded controller bytes against the caller-verified published wheel.

    RECORD/version alone is not source identity. The installed bootstrap/update
    modules must equal the actual archive bytes; dependency loading stays the
    existing package manager's responsibility. macOS keyagent signing is verified
    separately and is NOT described as signing these Python controller modules.
    """
    matches_digest = (
        re.fullmatch(r"[0-9a-f]{64}", digest)
        and hashlib.sha256(wheel.read_bytes()).hexdigest() == digest
    )
    if not matches_digest:
        raise BootstrapError("controller artifact digest disagrees; nothing was installed")
    package = Path(__file__).resolve().parent
    with zipfile.ZipFile(wheel) as archive:
        metadata = [name for name in archive.namelist() if name.endswith(".dist-info/METADATA")]
        if len(metadata) != 1:
            raise BootstrapError("controller distribution metadata is ambiguous")
        parsed = BytesParser().parsebytes(archive.read(metadata[0]))
        if parsed.get("Name", "").replace("_", "-").lower() != "local-operator":
            raise BootstrapError("controller artifact names a different distribution")
        for relative in ("update_bootstrap.py", "update.py", "cli.py", "install_transaction.py"):
            if (package / relative).read_bytes() != archive.read("local_operator/" + relative):
                raise BootstrapError(
                    "executed controller source differs from the verified artifact"
                )
    return (
        str(parsed.get("Version", "")),
        hashlib.sha256((package / "update.py").read_bytes()).hexdigest(),
    )


def _run(argv: list[str], *, env: dict[str, str], deadline: float) -> None:
    # The reviewed seed uses the existing bounded owned-worker runner available
    # in the baseline install. It grants NO install-root authority, and prevents
    # a killed/cancelled bootstrap from leaving package-manager children behind.
    from local_operator.install_transaction import run_installer

    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise BootstrapError("controller preparation budget expired; retry later")
    code = run_installer(
        argv,
        env=env,
        timeout=remaining,
        on_output=lambda text: print(text, end="", flush=True),
    )
    if code:
        raise BootstrapError(f"controller preparation exited {code}; no forced fallback")


def bootstrap(args: argparse.Namespace) -> int:
    if args.caller != CALLER_CONTRACT:
        raise BootstrapError(
            "caller does not support installed/serving split and no-kill rolling outcomes"
        )
    initial = destination(args.root, args.baseline)
    version, filename, url, digest = _publication(args.target)
    data = _read_url(url, artifact=True)
    if hashlib.sha256(data).hexdigest() != digest:
        raise BootstrapError("downloaded controller hash disagrees; no code was loaded")
    uv = shutil.which("uv")
    if uv is None:
        raise BootstrapError("the managed controller package manager is unavailable")
    deadline = time.monotonic() + BOOTSTRAP_TIMEOUT_S
    # Owned bootstrap scratch is OUTSIDE current/app-managed installs; this is
    # not a second uv-tool tree or a venv copied from another checkout. The shared
    # worker runner waits/reaps before this creator-owned staging is reclaimed.
    with tempfile.TemporaryDirectory(prefix="lop-controller-") as scratch:
        home = Path(scratch)
        wheel = home / filename
        wheel.write_bytes(data)
        prefix = home / "controller"
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        for key in tuple(env):
            if key.startswith("CMUX_") or key == "XPC_FLAGS":
                env.pop(key, None)
        _run([uv, "venv", "--python", sys.executable, str(prefix)], env=env, deadline=deadline)
        python = prefix / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        _run(
            [uv, "pip", "install", "--python", str(python), str(wheel)],
            env=env,
            deadline=deadline,
        )
        fresh = destination(args.root, args.baseline)
        if (fresh.device, fresh.inode) != (initial.device, initial.inode):
            raise BootstrapError("destination root was replaced during controller preparation")
        command = [
            str(python),
            "-B",
            "-P",
            "-m",
            "local_operator",
            "update",
            "--controller-root",
            str(initial.root),
            "--controller-baseline",
            str(initial.baseline),
            "--controller-artifact",
            str(wheel),
            "--controller-digest",
            digest,
            "--controller-caller",
            args.caller,
            "--controller-target",
            version,
        ]
        if args.snapshot is not None:
            command += ["--from-snapshot", args.snapshot]
        if args.no_services:
            command += ["--no-services"]
        _run(command, env=env, deadline=deadline)
    return 0


def controller(args: argparse.Namespace) -> int:
    """Verified new code mutates only a separately classified destination."""
    from local_operator import install_transaction as tx
    from local_operator import update

    required = (
        args.controller_artifact,
        args.controller_digest,
        args.controller_target,
        args.controller_baseline,
    )
    if not all(isinstance(value, str) and value for value in required):
        raise BootstrapError("controller request is incomplete; no destination mutation")
    if args.controller_caller != CALLER_CONTRACT:
        raise BootstrapError("controller caller lacks the rolling/no-kill outcome contract")
    version, source_digest = verify_controller(
        Path(args.controller_artifact), args.controller_digest
    )
    if version != args.controller_target:
        raise BootstrapError("loaded controller metadata disagrees with the published target")
    dest = destination(args.controller_root, args.controller_baseline)
    if update.stable_root().resolve() != dest.root:
        raise BootstrapError("destination disagrees with the canonical core root")
    # Foreign controller sys.prefix is NOT promoted to UV_TOOL. Judge the real
    # destination install's kind and immutable membership, not the caller venv.
    baseline_image = update._generation_install_root(dest.baseline)
    if update.install_kind(prefix=baseline_image) is not update.InstallKind.UV_TOOL:
        raise BootstrapError("destination kind is unsupported; no kind override")
    if sys.platform == "darwin":
        from local_operator.operator.macos.keyagent import KeyagentError, verify_bundle

        try:
            verify_bundle(Path(__file__).parent / "operator" / "macos" / "lop-keyagent.app")
        except KeyagentError as exc:
            raise BootstrapError(
                f"published controller helper signature could not be verified: {exc}"
            ) from exc
    published = update._fresh_publication(version, "exact")
    if (
        published is None
        or published.digest != args.controller_digest
        or published.artifact != Path(args.controller_artifact).name
    ):
        raise BootstrapError("loaded controller no longer matches fresh published bytes")
    with tx.transaction(dest.root, operation="verified controller admission") as held:
        held.validate(dest.root)
        if time.monotonic() >= published.fresh_until:
            raise BootstrapError("controller publication evidence expired; retry")
        current_info = dest.root.stat()
        if (current_info.st_dev, current_info.st_ino) != (dest.device, dest.inode):
            raise BootstrapError("canonical root identity changed")
        if update.current_generation() != dest.baseline:
            raise BootstrapError("current changed during bootstrap; check again and retry")
    # `commit`/`ref` are the SnapshotSource defaults (empty string) rather than
    # None: `install_into_generation` takes `str`, and "no snapshot" and "a
    # snapshot with no readable git metadata" are the same answer to it — the
    # marker records the bare source token. Declared `str` here so the call below
    # type-checks against the real signature instead of narrowing at the call.
    source = None
    commit = ""
    ref = ""
    target = version
    # Bound BEFORE the branch so the `finally` cleanup below sees a name that
    # exists on every path; the branch is the only writer, and the None guard is
    # what tells "no snapshot" from "a snapshot to clean up".
    snapshot: update.SnapshotSource | None = None
    if args.from_snapshot is not None:
        snapshot = update.resolve_snapshot(args.from_snapshot)
        source, commit, ref, target = snapshot.path, snapshot.commit, snapshot.ref, snapshot.version
        downgrade = update.classify_snapshot_downgrade(snapshot)
        if downgrade.is_downgrade:
            raise BootstrapError(
                "snapshot would regress current; no rolling downgrade/force bypass"
            )
        route = update.classify_snapshot_install(snapshot)
        if route.install_from_wheel:
            source = None
            target = route.version
    try:
        promoted = update.install_into_generation(
            source,
            version=target,
            commit=commit,
            ref=ref,
            target_mode=(
                ("exact" if args.from_snapshot is not None else "latest")
                if source is None
                else "source"
            ),
        )
    finally:
        if snapshot is not None and snapshot.temporary:
            # Only resolve_snapshot's own creator-owned archive, not any user
            # source/installed generation; generation failure cleanup stays guarded.
            shutil.rmtree(snapshot.path)
    print(
        json.dumps(
            {
                "controller": {
                    "contract": CONTROLLER_CONTRACT,
                    "version": version,
                    "source_sha256": source_digest,
                    "artifact_sha256": args.controller_digest,
                }
            }
        ),
        flush=True,
    )
    # The ACTUAL current image executes its tail, never this isolated caller.
    # It rechecks current and reports supersession; service errors cannot roll
    # back a committed pointer or force-kill a serving process.
    now = update.current_generation()
    if now is None:
        raise BootstrapError("core promoted but current is unreadable; services pending")
    python_name = "Scripts/python.exe" if os.name == "nt" else "bin/python"
    python = update._generation_install_root(now) / python_name
    command = [
        str(python),
        "-B",
        "-P",
        "-m",
        "local_operator",
        "update",
        "--controller-tail",
        str(promoted),
        "--controller-target",
        target,
    ]
    if args.no_services:
        command += ["--no-services"]
    try:
        _run(
            command,
            env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
            deadline=time.monotonic() + BOOTSTRAP_TIMEOUT_S,
        )
    except BootstrapError as exc:
        # A committed core is not undone because a superseding/older current tail
        # cannot run this protocol. Emit PARTIAL from verified controller code;
        # new UI consumes this, old callers are never given a forced fallback.
        print(f"core installed; central services remain pending: {exc}", file=sys.stderr)
        update._emit_update_report(
            install_version=update.generation_version(update.current_generation()),
            target=target,
            stage=update.StageReport(),
            promoted_generation=promoted,
        )
    return 0


def controller_tail(args: argparse.Namespace) -> int:
    """Only a current generation may execute the capability-safe service tail."""
    from local_operator import update

    current = update.current_generation()
    if current is None or not update._is_generation_install():
        raise BootstrapError("service tail is not executing a canonical core generation")
    if Path(sys.prefix).resolve() != update._generation_install_root(current).resolve():
        raise BootstrapError("tail was superseded before entry; services pending")
    promoted = Path(args.controller_tail).resolve(strict=True)
    if promoted.parent != update.generations_dir().resolve():
        raise BootstrapError("promoted receipt is outside the canonical generation store")
    return update._generation_upgrade(
        0,
        install_version=update.generation_version(current),
        target=args.controller_target,
        services=not args.no_services,
        roll=False,
        promoted_generation=promoted,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Verified rolling controller bootstrap")
    parser.add_argument("--root", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--target")
    parser.add_argument("--caller", required=True)
    parser.add_argument("--snapshot")
    parser.add_argument("--no-services", action="store_true")
    args = parser.parse_args()
    try:
        return bootstrap(args)
    except (BootstrapError, OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
        print(f"rolling update could not start: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
