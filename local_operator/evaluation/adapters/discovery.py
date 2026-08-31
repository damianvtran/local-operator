"""Worker-only exact distribution verification and entry-point loading."""

from __future__ import annotations

import base64
import csv
import hashlib
import importlib.metadata
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


class AdapterDiscoveryError(RuntimeError):
    """A closed discovery failure safe to report across the RPC boundary."""


@dataclass(frozen=True)
class ResolvedLaunch:
    executable: str
    executable_device: int
    executable_inode: int
    executable_mode: int
    workspace: str
    workspace_device: int
    workspace_inode: int
    workspace_mode: int


def _symlink_free(path: Path) -> os.stat_result:
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


def resolve_launch(selector: AdapterSelector) -> ResolvedLaunch:
    """Capture symlink-free identities for both spawn boundaries to recheck."""

    executable = Path(selector.python_executable)
    workspace = Path(selector.workspace)
    executable_info = _symlink_free(executable)
    workspace_info = _symlink_free(workspace)
    if not stat.S_ISREG(executable_info.st_mode) or not os.access(executable, os.X_OK):
        raise AdapterDiscoveryError("adapter Python is not an executable regular file")
    if not stat.S_ISDIR(workspace_info.st_mode):
        raise AdapterDiscoveryError("adapter workspace is not a directory")
    return ResolvedLaunch(
        executable=str(executable),
        executable_device=executable_info.st_dev,
        executable_inode=executable_info.st_ino,
        executable_mode=executable_info.st_mode,
        workspace=str(workspace),
        workspace_device=workspace_info.st_dev,
        workspace_inode=workspace_info.st_ino,
        workspace_mode=workspace_info.st_mode,
    )


def validate_resolved_launch(launch: ResolvedLaunch) -> None:
    executable_info = _symlink_free(Path(launch.executable))
    workspace_info = _symlink_free(Path(launch.workspace))
    current = (
        executable_info.st_dev,
        executable_info.st_ino,
        executable_info.st_mode,
        workspace_info.st_dev,
        workspace_info.st_ino,
        workspace_info.st_mode,
    )
    expected = (
        launch.executable_device,
        launch.executable_inode,
        launch.executable_mode,
        launch.workspace_device,
        launch.workspace_inode,
        launch.workspace_mode,
    )
    if current != expected:
        raise AdapterDiscoveryError("adapter launch path identity changed before spawn")


def validate_launch_paths(selector: AdapterSelector) -> None:
    resolve_launch(selector)


def worker_argv(selector: AdapterSelector) -> tuple[str, ...]:
    resolve_launch(selector)
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


def load_selected_adapter(selector: AdapterSelector) -> EvaluationAdapter:
    """Load exactly one entry point from only the already selected distribution."""

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
    try:
        factory = cast(Callable[[], Any], matches[0].load())
        adapter = factory()
    except Exception as error:
        raise AdapterDiscoveryError("selected adapter entry point failed to load") from error
    if not isinstance(adapter, EvaluationAdapter):
        raise AdapterDiscoveryError("selected entry point does not implement the adapter protocol")
    return adapter
