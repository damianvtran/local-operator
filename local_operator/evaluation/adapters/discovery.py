"""Worker-only exact distribution verification and entry-point loading."""

from __future__ import annotations

import base64
import csv
import hashlib
import importlib.metadata
import inspect
import json
import os
import stat
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
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
        file_path = Path(distribution.locate_file(path))
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
        if actual != digest:
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


def _verified_entry_module(
    distribution: importlib.metadata.Distribution, selector: AdapterSelector
) -> tuple[str, Path]:
    module, separator, attribute = selector.entry_point.partition(":")
    if separator != ":" or not module or not attribute or ":" in attribute:
        raise AdapterDiscoveryError("adapter entry point must be module:attribute")
    candidates = {f"{module.replace('.', '/')}.py", f"{module.replace('.', '/')}/__init__.py"}
    rows = {row[0]: row for row in _record_rows(distribution)}
    matches = sorted(candidates & rows.keys())
    if len(matches) != 1:
        raise AdapterDiscoveryError("adapter entry module is not uniquely RECORD-covered")
    module_path = Path(str(distribution.locate_file(matches[0])))
    _symlink_free(module_path)
    encoded_hash = rows[matches[0]][1].split("=", 1)[1]
    actual = base64.urlsafe_b64encode(hashlib.sha256(module_path.read_bytes()).digest()).rstrip(
        b"="
    )
    if actual.decode() != encoded_hash:
        raise AdapterDiscoveryError("adapter entry module RECORD hash differs")
    return module, module_path


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
    module, module_path = _verified_entry_module(distribution, selector)
    try:
        factory = cast(Callable[[], Any], matches[0].load())
        factory_module = getattr(factory, "__module__", "")
        source = inspect.getsourcefile(factory)
        if factory_module != module and not factory_module.startswith(f"{module}."):
            raise AdapterDiscoveryError("adapter factory comes from another module")
        if source is None or Path(source).resolve(strict=True) != module_path.resolve(strict=True):
            raise AdapterDiscoveryError("adapter factory source differs from verified module")
        adapter = factory()
    except Exception as error:
        raise AdapterDiscoveryError("selected adapter entry point failed to load") from error
    if not isinstance(adapter, EvaluationAdapter):
        raise AdapterDiscoveryError("selected entry point does not implement the adapter protocol")
    return adapter
