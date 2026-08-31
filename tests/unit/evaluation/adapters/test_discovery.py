from __future__ import annotations

import base64
import csv
import hashlib
import importlib.metadata
import sys
from pathlib import Path

import pytest

from local_operator.evaluation.adapters.api import AdapterSelector
from local_operator.evaluation.adapters.discovery import (
    AdapterDiscoveryError,
    distribution_digest,
    load_selected_adapter,
    verify_distribution,
    worker_argv,
)


class FakeEntryPoint:
    group = "local_operator.evaluation_adapters.v1"
    name = "tiny"
    value = "tiny_adapter:create"

    def __init__(self, factory: object) -> None:
        self._factory = factory

    def load(self) -> object:
        return self._factory


class FakeDistribution:
    version = "1.0"

    def __init__(self, root: Path, entries: list[FakeEntryPoint]) -> None:
        self.root = root
        self.entry_points = entries
        self._record = ""

    def read_text(self, name: str) -> str | None:
        assert name == "RECORD"
        return self._record

    def locate_file(self, path: str) -> Path:
        return self.root / path

    def make_record(self, *, unhashed: bool = False) -> None:
        rows: list[list[str]] = []
        for path in sorted(self.root.rglob("*")):
            if path.is_file():
                relative = str(path.relative_to(self.root))
                data = path.read_bytes()
                digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=")
                rows.append(
                    [relative, "" if unhashed else f"sha256={digest.decode()}", str(len(data))]
                )
        rows.append(["tiny_adapter-1.0.dist-info/RECORD", "", ""])
        from io import StringIO

        target = StringIO()
        csv.writer(target, lineterminator="\n").writerows(rows)
        self._record = target.getvalue()


def fake_distribution(tmp_path: Path, entries: list[FakeEntryPoint]) -> FakeDistribution:
    package = tmp_path / "tiny_adapter.py"
    package.write_text("VALUE = 1\n")
    distribution = FakeDistribution(tmp_path, entries)
    distribution.make_record()
    return distribution


def selected(tmp_path: Path, digest: str) -> AdapterSelector:
    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    return AdapterSelector(
        schema_version="1.0",
        adapter_id="tiny",
        distribution="tiny-adapter",
        version="1.0",
        entry_point="tiny_adapter:create",
        package_digest=digest,
        release_digest="b" * 64,
        python_executable=str(Path(sys.executable).resolve()),
        workspace=str(workspace),
        route_capability="computer",
    )


def test_exact_record_digest_and_worker_flags(tmp_path: Path) -> None:
    distribution = fake_distribution(tmp_path, [])
    selector = selected(tmp_path, distribution_digest(distribution))
    assert worker_argv(selector)[1:5] == ("-I", "-s", "-E", "-m")


def test_verify_uses_only_selected_distribution(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    distribution = fake_distribution(tmp_path, [])
    selector = selected(tmp_path, distribution_digest(distribution))
    calls: list[str] = []

    def exact(name: str) -> FakeDistribution:
        calls.append(name)
        return distribution

    def forbidden() -> None:
        raise AssertionError("global entry points must not be enumerated")

    monkeypatch.setattr(importlib.metadata, "distribution", exact)
    monkeypatch.setattr(importlib.metadata, "entry_points", forbidden)
    assert verify_distribution(selector) is distribution
    assert calls == ["tiny-adapter"]


def test_unhashed_editable_distribution_is_rejected(tmp_path: Path) -> None:
    distribution = fake_distribution(tmp_path, [])
    distribution.make_record(unhashed=True)
    with pytest.raises(AdapterDiscoveryError, match="unhashed"):
        distribution_digest(distribution)


def test_duplicate_and_similarly_named_entry_points_are_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    factory = lambda: object()  # noqa: E731
    exact = FakeEntryPoint(factory)
    similar = FakeEntryPoint(factory)
    similar.name = "tiny-other"
    distribution = fake_distribution(tmp_path, [exact, exact, similar])
    selector = selected(tmp_path, distribution_digest(distribution))
    monkeypatch.setattr(importlib.metadata, "distribution", lambda _: distribution)
    with pytest.raises(AdapterDiscoveryError, match="exactly one"):
        load_selected_adapter(selector)


def test_host_discovery_does_not_import_adapter_module(tmp_path: Path) -> None:
    distribution = fake_distribution(tmp_path, [])
    selector = selected(tmp_path, distribution_digest(distribution))
    worker_argv(selector)
    assert "tiny_adapter" not in sys.modules
