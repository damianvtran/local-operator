"""Worker-only exact distribution verification and entry-point loading."""

from __future__ import annotations

import base64
import csv
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import posixpath
import stat
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, cast

from local_operator.evaluation.adapters.api import (
    ADAPTER_ENTRY_POINT_GROUP,
    AdapterSelector,
    EvaluationAdapter,
)
from local_operator.evaluation.evidence.models import canonical_digest

MAX_WORKSPACE_FILES = 100_000
MAX_WORKSPACE_BYTES = 4 * 1024 * 1024 * 1024
RELEASE_MANIFEST = "adapter-release.json"


class AdapterDiscoveryError(RuntimeError):
    """A closed discovery failure safe to report across the RPC boundary."""


@dataclass(frozen=True)
class ResolvedLaunch:
    executable: str
    executable_device: int
    executable_inode: int
    executable_mode: int
    executable_size: int
    executable_sha256: str
    workspace: str
    workspace_device: int
    workspace_inode: int
    workspace_mode: int


def _symlink_free(path: Path) -> os.stat_result:
    # The selector must name the resolved real interpreter and workspace, not a
    # convenience alias: a symlink or lexical alias is exactly the substitution
    # the dev/inode/content pins below are meant to detect.
    if not path.is_absolute() or os.path.normpath(str(path)) != str(path):
        raise AdapterDiscoveryError("adapter launch path must be normalized and absolute")
    try:
        if path.resolve(strict=True) != path:
            raise AdapterDiscoveryError("adapter launch path has a symlink or lexical alias")
        current = Path(path.anchor)
        for component in path.parts[1:]:
            current /= component
            info = os.lstat(current)
            if stat.S_ISLNK(info.st_mode):
                raise AdapterDiscoveryError("adapter launch path contains a symlink")
        return os.lstat(path)
    except AdapterDiscoveryError:
        raise
    except OSError as error:
        raise AdapterDiscoveryError("adapter launch path is unavailable") from error


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_launch(selector: AdapterSelector) -> ResolvedLaunch:
    """Capture symlink-free identities for both spawn boundaries to recheck."""

    executable = Path(selector.python_executable)
    workspace = Path(selector.workspace)
    executable_info = _symlink_free(executable)
    workspace_info = _symlink_free(workspace)
    # A link count is a packaging detail, never an integrity property: CPython
    # ships python/python3/python3.N as hardlinks to one inode, so requiring
    # nlink==1 rejects every normal uv and system interpreter. Substitution is
    # caught by the device/inode/mode/size/sha256 pins revalidated before spawn.
    if not stat.S_ISREG(executable_info.st_mode) or not os.access(executable, os.X_OK):
        raise AdapterDiscoveryError("adapter Python is not an executable regular file")
    executable_digest = _file_sha256(executable)
    if not stat.S_ISDIR(workspace_info.st_mode):
        raise AdapterDiscoveryError("adapter workspace is not a directory")
    return ResolvedLaunch(
        executable=str(executable),
        executable_device=executable_info.st_dev,
        executable_inode=executable_info.st_ino,
        executable_mode=executable_info.st_mode,
        executable_size=executable_info.st_size,
        executable_sha256=executable_digest,
        workspace=str(workspace),
        workspace_device=workspace_info.st_dev,
        workspace_inode=workspace_info.st_ino,
        workspace_mode=workspace_info.st_mode,
    )


def validate_resolved_launch(launch: ResolvedLaunch) -> None:
    executable_info = _symlink_free(Path(launch.executable))
    workspace_info = _symlink_free(Path(launch.workspace))
    if not stat.S_ISREG(executable_info.st_mode):
        raise AdapterDiscoveryError("adapter Python identity is unsafe")
    executable_digest = _file_sha256(Path(launch.executable))
    current = (
        executable_info.st_dev,
        executable_info.st_ino,
        executable_info.st_mode,
        executable_info.st_size,
        executable_digest,
        workspace_info.st_dev,
        workspace_info.st_ino,
        workspace_info.st_mode,
    )
    expected = (
        launch.executable_device,
        launch.executable_inode,
        launch.executable_mode,
        launch.executable_size,
        launch.executable_sha256,
        launch.workspace_device,
        launch.workspace_inode,
        launch.workspace_mode,
    )
    if current != expected:
        raise AdapterDiscoveryError("adapter launch path identity changed before spawn")


def workspace_digest(path: str) -> str:
    """Hash every immutable workspace file without following special entries.

    The caps are a deliberate launch-time bound, not a performance target: at
    the 4 GiB ceiling this streams for roughly four seconds once per launch and
    per handshake, never during ordinary startup. A hardlinked workspace file
    stays fatal here because a second name for adapter content is a mutation
    channel the manifest cannot observe, unlike the interpreter, whose content
    is pinned by digest.

    The workspace is hashed at resolution and re-hashed by the worker during the
    handshake, but it is not re-verified immediately before spawn the way the
    executable identity is. That asymmetry is a known, deliberate gap for a
    later round to close or accept explicitly.
    """

    root = Path(path)
    root_info = _symlink_free(root)
    if not stat.S_ISDIR(root_info.st_mode):
        raise AdapterDiscoveryError("adapter workspace is not a directory")
    entries: list[dict[str, Any]] = []
    total_bytes = 0
    for directory, directory_names, file_names in os.walk(root, followlinks=False):
        directory_names.sort()
        file_names.sort()
        current = Path(directory)
        for name in (*directory_names, *file_names):
            candidate = current / name
            info = os.lstat(candidate)
            if stat.S_ISLNK(info.st_mode):
                raise AdapterDiscoveryError("adapter workspace contains a symlink")
            if stat.S_ISDIR(info.st_mode):
                continue
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise AdapterDiscoveryError("adapter workspace contains an unsafe file")
            if len(entries) >= MAX_WORKSPACE_FILES:
                raise AdapterDiscoveryError("adapter workspace file count exceeds limit")
            total_bytes += info.st_size
            if total_bytes > MAX_WORKSPACE_BYTES:
                raise AdapterDiscoveryError("adapter workspace bytes exceed limit")
            digest = hashlib.sha256()
            with candidate.open("rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    digest.update(chunk)
            entries.append(
                {
                    "path": candidate.relative_to(root).as_posix(),
                    "mode": stat.S_IMODE(info.st_mode),
                    "size": info.st_size,
                    "sha256": digest.hexdigest(),
                }
            )
    entries.sort(key=lambda entry: entry["path"])
    return canonical_digest("adapter-workspace-manifest-v1", entries)


def verify_release_manifest(selector: AdapterSelector) -> None:
    manifest = Path(selector.workspace) / RELEASE_MANIFEST
    try:
        info = os.lstat(manifest)
        raw = manifest.read_bytes()
    except OSError as error:
        raise AdapterDiscoveryError("adapter release manifest is unavailable") from error
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or len(raw) > 4096:
        raise AdapterDiscoveryError("adapter release manifest is unsafe")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AdapterDiscoveryError("adapter release manifest is malformed") from error
    canonical = json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
    if raw != canonical or value != {"release_digest": selector.release_digest}:
        raise AdapterDiscoveryError("adapter release manifest digest differs")


def validate_launch_paths(selector: AdapterSelector) -> None:
    resolve_launch(selector)


def worker_argv(selector: AdapterSelector) -> tuple[str, ...]:
    resolve_launch(selector)
    verify_release_manifest(selector)
    if workspace_digest(selector.workspace) != selector.workspace_digest:
        raise AdapterDiscoveryError("adapter workspace content digest differs")
    # Isolation flags prevent user site, PYTHON* variables, and current-directory
    # imports from changing which exact wheel the worker verifies.
    return (
        selector.python_executable,
        "-I",
        "-s",
        "-E",
        "-m",
        "local_operator.evaluation.adapters.worker",
    )


def _record_rows(distribution: Any) -> list[list[str]]:
    record = distribution.read_text("RECORD")
    if record is None:
        raise AdapterDiscoveryError("adapter distribution has no wheel RECORD")
    rows = list(csv.reader(record.splitlines()))
    canonical: list[list[str]] = []
    for row in rows:
        if len(row) != 3 or not row[0]:
            raise AdapterDiscoveryError("adapter wheel RECORD is malformed")
        path, encoded_hash, size = row
        if path.endswith("/RECORD"):
            if encoded_hash or size:
                raise AdapterDiscoveryError("wheel RECORD self-entry must be unhashed")
            continue
        if not encoded_hash or not size.isdecimal():
            raise AdapterDiscoveryError("editable or unhashed adapter distributions are forbidden")
        algorithm, separator, digest = encoded_hash.partition("=")
        if separator != "=" or algorithm != "sha256" or not digest:
            raise AdapterDiscoveryError("adapter RECORD must use sha256 hashes")
        # A RECORD path is attacker-controlled data: normalize before it can be
        # joined onto the install root, so no entry escapes the distribution.
        normalized = posixpath.normpath(path)
        if normalized != path or posixpath.isabs(path) or normalized.split("/")[0] == "..":
            raise AdapterDiscoveryError("adapter RECORD path is not a normalized relative path")
        file_path = Path(str(distribution.locate_file(path)))
        try:
            info = file_path.stat()
            data = file_path.read_bytes()
        except OSError as error:
            raise AdapterDiscoveryError("adapter RECORD file is unavailable") from error
        if not stat.S_ISREG(info.st_mode) or info.st_size != int(size):
            raise AdapterDiscoveryError("adapter RECORD size differs from installed file")
        actual = (
            base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode("ascii")
        )
        # Wheel RECORD hashes are unpadded by PEP 376, but a conformant padded
        # value must not read as a content mismatch.
        if actual != digest.rstrip("="):
            raise AdapterDiscoveryError("adapter RECORD hash differs from installed file")
        canonical.append([path, encoded_hash, size])
    if not canonical:
        raise AdapterDiscoveryError("adapter wheel RECORD has no hashed files")
    canonical.sort(key=lambda item: item[0])
    return canonical


def distribution_digest(distribution: Any) -> str:
    """Bind selection to every installed, hashed wheel file."""

    return canonical_digest("adapter-installed-wheel-record-v1", _record_rows(distribution))


def verify_distribution(selector: AdapterSelector) -> importlib.metadata.Distribution:
    try:
        distribution = importlib.metadata.distribution(selector.distribution)
    except importlib.metadata.PackageNotFoundError as error:
        raise AdapterDiscoveryError("selected adapter distribution is not installed") from error
    if distribution.version != selector.version:
        raise AdapterDiscoveryError("selected adapter distribution version differs")
    if distribution_digest(distribution) != selector.package_digest:
        raise AdapterDiscoveryError("selected adapter package digest differs")
    return distribution


def _verify_recorded_file(
    distribution: importlib.metadata.Distribution,
    rows: dict[str, list[str]],
    relative: str,
) -> tuple[Path, bytes]:
    """Return the path and the exact bytes whose hash matched RECORD.

    The bytes are returned, not re-read later, because any second read is a
    different observation than the one that was verified.
    """

    path = Path(str(distribution.locate_file(relative)))
    _symlink_free(path)
    encoded_hash = rows[relative][1].split("=", 1)[1].rstrip("=")
    data = path.read_bytes()
    actual = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode("ascii")
    if actual != encoded_hash:
        raise AdapterDiscoveryError("adapter module RECORD hash differs")
    return path, data


@dataclass(frozen=True)
class _VerifiedModule:
    name: str
    path: Path
    source: bytes
    is_package: bool


def _verified_module_chain(
    distribution: importlib.metadata.Distribution, selector: AdapterSelector
) -> tuple[str, tuple[_VerifiedModule, ...]]:
    """Return every module that must execute, with its exact verified bytes."""

    module, separator, attribute = selector.entry_point.partition(":")
    if separator != ":" or not module or not attribute or ":" in attribute:
        raise AdapterDiscoveryError("adapter entry point must be module:attribute")
    parts = module.split(".")
    if not all(part.isidentifier() for part in parts):
        raise AdapterDiscoveryError("adapter entry module is not a dotted identifier")
    rows = {row[0]: row for row in _record_rows(distribution)}
    chain: list[_VerifiedModule] = []
    # Importing pkg.sub.mod executes every ancestor __init__ first, so an
    # unrecorded ancestor would run arbitrary code before any verification.
    # Each ancestor must be RECORD-covered and hash-matched, or genuinely absent
    # from disk (a real namespace package).
    for depth in range(1, len(parts)):
        name = ".".join(parts[:depth])
        ancestor = "/".join(parts[:depth]) + "/__init__.py"
        if ancestor in rows:
            path, source = _verify_recorded_file(distribution, rows, ancestor)
            chain.append(_VerifiedModule(name=name, path=path, source=source, is_package=True))
            continue
        if Path(str(distribution.locate_file(ancestor))).exists():
            raise AdapterDiscoveryError("adapter package init is not RECORD-covered")
    candidates = {f"{'/'.join(parts)}.py", f"{'/'.join(parts)}/__init__.py"}
    matches = sorted(candidates & rows.keys())
    if len(matches) != 1:
        raise AdapterDiscoveryError("adapter entry module is not uniquely RECORD-covered")
    path, source = _verify_recorded_file(distribution, rows, matches[0])
    chain.append(
        _VerifiedModule(
            name=module,
            path=path,
            source=source,
            is_package=matches[0].endswith("/__init__.py"),
        )
    )
    return attribute, tuple(chain)


def _execute_verified_chain(chain: tuple[_VerifiedModule, ...]) -> ModuleType:
    """Compile and run the verified bytes themselves, never a cached artifact.

    Hashing ``.py`` source and then delegating to a normal import is not a
    verification: CPython prefers ``__pycache__/<name>.<tag>.pyc`` whenever its
    header matches the source mtime and size, both of which an attacker who can
    write the cache also controls. The hashed bytes and the executed bytes are
    then unrelated, and ``-I -s -E`` do not help because they constrain sys.path
    and the environment, not bytecode trust.

    Refusing to load when a ``__pycache__`` entry exists would be a denylist over
    a cache location CPython is free to change, and it stays racy: the cache can
    reappear between the check and the import. Compiling the verified bytes in
    process removes the question instead of policing it — the object that runs is
    derived from the byte string that was hashed, so the invariant is structural.
    """

    loaded: ModuleType | None = None
    for entry in chain:
        specification = importlib.util.spec_from_file_location(entry.name, entry.path)
        if specification is None:
            raise AdapterDiscoveryError("adapter module specification is unavailable")
        created = importlib.util.module_from_spec(specification)
        if entry.is_package:
            created.__path__ = [str(entry.path.parent)]
        code = compile(entry.source, str(entry.path), "exec", dont_inherit=True)
        # Registering before execution keeps intra-package imports resolvable,
        # exactly as the import system would during a normal package import.
        sys.modules[entry.name] = created
        try:
            exec(code, created.__dict__)
        except BaseException:
            sys.modules.pop(entry.name, None)
            raise
        loaded = created
    assert loaded is not None
    return loaded


def load_selected_adapter(selector: AdapterSelector) -> EvaluationAdapter:
    """Load exactly one preverified module from the selected distribution."""

    distribution = verify_distribution(selector)
    matches = [
        entry
        for entry in distribution.entry_points
        if entry.group == ADAPTER_ENTRY_POINT_GROUP
        and entry.name == selector.adapter_id
        and entry.value == selector.entry_point
    ]
    if len(matches) != 1:
        raise AdapterDiscoveryError(
            "selected distribution must expose exactly one exact entry point"
        )
    attribute, chain = _verified_module_chain(distribution, selector)
    try:
        # The entry point is resolved against the module built from verified
        # bytes, so `.load()` is never given the chance to import a cached
        # artifact we did not hash.
        loaded = _execute_verified_chain(chain)
        factory = cast(Callable[[], Any], getattr(loaded, attribute))
        adapter = factory()
    except AdapterDiscoveryError:
        raise
    except Exception as error:
        raise AdapterDiscoveryError("selected adapter entry point failed to load") from error
    if not isinstance(adapter, EvaluationAdapter):
        raise AdapterDiscoveryError("selected entry point does not implement the adapter protocol")
    return adapter
