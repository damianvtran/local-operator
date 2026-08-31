from __future__ import annotations

import base64
import csv
import hashlib
import importlib
import importlib.metadata
import importlib.util
import marshal
import os
import shutil
import sys
from io import StringIO
from pathlib import Path
from typing import cast

import pytest

from local_operator.evaluation.adapters.api import AdapterSelector
from local_operator.evaluation.adapters.discovery import (
    AdapterDiscoveryError,
    _record_rows,
    _verified_entry_target,
    _verified_imports,
    _verify_recorded_file,
    distribution_digest,
    load_selected_adapter,
    resolve_launch,
    validate_resolved_launch,
    verify_distribution,
    worker_argv,
    workspace_digest,
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
    release_digest = "b" * 64
    (workspace / "adapter-release.json").write_text(f'{{"release_digest":"{release_digest}"}}')
    executable = tmp_path / "python"
    if not executable.exists():
        shutil.copy2(Path(sys.executable).resolve(), executable)
        executable.chmod(0o755)
    return AdapterSelector(
        schema_version="1.0",
        adapter_id="tiny",
        distribution="tiny-adapter",
        version="1.0",
        entry_point="tiny_adapter:create",
        package_digest=digest,
        release_digest=release_digest,
        python_executable=str(executable),
        workspace=str(workspace),
        workspace_digest=workspace_digest(str(workspace)),
        route_capability="computer",
    )


def test_launch_rejects_symlink_and_lexical_aliases(tmp_path: Path) -> None:
    real_python = Path(sys.executable).resolve()
    python_link = tmp_path / "python-link"
    python_link.symlink_to(real_python)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    base = selected(tmp_path, "a" * 64)
    with pytest.raises(AdapterDiscoveryError, match="symlink|alias"):
        resolve_launch(base.model_copy(update={"python_executable": str(python_link)}))
    workspace_link = tmp_path / "workspace-link"
    workspace_link.symlink_to(workspace, target_is_directory=True)
    with pytest.raises(AdapterDiscoveryError, match="symlink|alias"):
        resolve_launch(base.model_copy(update={"workspace": str(workspace_link)}))
    alias = str(tmp_path / "workspace" / ".." / "workspace")
    with pytest.raises(Exception, match="normalized|alias"):
        resolve_launch(base.model_copy(update={"workspace": alias}))


def test_launch_identity_detects_swap(tmp_path: Path) -> None:
    base = selected(tmp_path, "a" * 64)
    resolved = resolve_launch(base)
    workspace = Path(base.workspace)
    shutil.rmtree(workspace)
    workspace.mkdir()
    with pytest.raises(AdapterDiscoveryError, match="identity changed"):
        validate_resolved_launch(resolved)


def test_hardlinked_interpreter_is_accepted_and_content_mutation_detected(
    tmp_path: Path,
) -> None:
    base = selected(tmp_path, "a" * 64)
    executable = Path(base.python_executable)
    # Stock CPython ships python/python3/python3.N as hardlinks to one inode, so
    # a link count must never be the thing that rejects an interpreter.
    hardlink = tmp_path / "python-hardlink"
    os.link(executable, hardlink)
    assert os.lstat(executable).st_nlink == 2
    resolved = resolve_launch(base)
    validate_resolved_launch(resolved)
    hardlink.unlink()
    resolved = resolve_launch(base)
    with executable.open("r+b") as stream:
        first = stream.read(1)
        stream.seek(0)
        stream.write(bytes([first[0] ^ 1]))
    with pytest.raises(AdapterDiscoveryError, match="identity changed"):
        validate_resolved_launch(resolved)


def test_unrecorded_package_init_is_rejected_before_it_executes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An ancestor __init__ runs before the leaf module, so it must be recorded."""

    marker = tmp_path / "init-executed.marker"
    package = tmp_path / "pkg"
    (package / "sub").mkdir(parents=True)
    side_effect = f"from pathlib import Path\nPath({str(marker)!r}).write_text('ran')\n"
    (package / "__init__.py").write_text(side_effect)
    (package / "sub" / "__init__.py").write_text(side_effect)
    # The leaf imports through its package, so removing the ancestor check does
    # not merely skip verification: the import system then really executes both
    # unrecorded __init__ files. That is what makes the marker assertion below
    # evidence rather than decoration.
    (package / "sub" / "helper.py").write_text("VALUE = 1\n")
    leaf = package / "sub" / "mod.py"
    leaf.write_text("from . import helper\n\n\ndef create():\n    return helper.VALUE\n")

    def load() -> object:
        raise AssertionError("entry point loaded before verification")

    entry = FakeEntryPoint(load)
    entry.name = "tiny"
    entry.value = "pkg.sub.mod:create"
    distribution = FakeDistribution(tmp_path, [entry])
    # RECORD covers only the leaf module, exactly like the crafted attack.
    data = leaf.read_bytes()
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
    rows = [
        ["pkg/sub/mod.py", f"sha256={digest}", str(len(data))],
        ["tiny_adapter-1.0.dist-info/RECORD", "", ""],
    ]
    target = StringIO()
    csv.writer(target, lineterminator="\n").writerows(rows)
    distribution._record = target.getvalue()
    selector = selected(tmp_path, distribution_digest(distribution)).model_copy(
        update={"entry_point": "pkg.sub.mod:create"}
    )
    monkeypatch.setattr(importlib.metadata, "distribution", lambda _: distribution)
    monkeypatch.syspath_prepend(str(tmp_path))
    for name in ("pkg", "pkg.sub", "pkg.sub.mod"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    with pytest.raises(AdapterDiscoveryError, match="not RECORD-covered"):
        load_selected_adapter(selector)
    assert not marker.exists()


def _forge_pycache(module: Path, source_code: str) -> Path:
    """Write a cache entry whose header matches the source but whose code differs.

    CPython validates a cached file by mtime and size only, both of which an
    attacker able to write ``__pycache__`` also controls, so this forgery is
    accepted by a normal import even though the source hash is untouched.
    """

    source = module.read_bytes()
    attacker = compile(source_code, str(module), "exec")
    cache = Path(importlib.util.cache_from_source(str(module)))
    cache.parent.mkdir(parents=True, exist_ok=True)
    info = module.stat()
    cache.write_bytes(
        importlib.util.MAGIC_NUMBER
        + (0).to_bytes(4, "little")
        + int(info.st_mtime).to_bytes(4, "little")
        + len(source).to_bytes(4, "little")
        + marshal.dumps(attacker)
    )
    return cache


def test_forged_bytecode_cannot_execute_in_place_of_hashed_source(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The bytes that execute must be the bytes that were hashed."""

    marker = tmp_path / "attacker-ran.marker"

    def load() -> object:
        raise AssertionError("entry point must not be imported by the import system")

    # Build the distribution first, then give the module its real body, so the
    # RECORD hash and the forged cache header describe the same source bytes.
    distribution = fake_distribution(tmp_path, [FakeEntryPoint(load)])
    module = tmp_path / "tiny_adapter.py"
    module.write_text("def create():\n    return 'verified'\n")
    distribution.make_record()
    cache = _forge_pycache(
        module,
        f"from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('pwned')\n"
        f"def create():\n    return 'attacker'\n",
    )
    assert cache.exists()
    selector = selected(tmp_path, distribution_digest(distribution))
    monkeypatch.syspath_prepend(str(tmp_path))

    # A normal import of this module runs the forged cache, which is the attack
    # this loader has to refuse; assert that first so the test cannot pass
    # against a host where the forgery simply failed to take.
    sys.modules.pop("tiny_adapter", None)
    try:
        attacked = importlib.import_module("tiny_adapter")
        assert marker.exists(), "forged cache did not take effect on this host"
        assert attacked.create() == "attacker"
    finally:
        sys.modules.pop("tiny_adapter", None)
    marker.unlink()

    # The source is genuinely RECORD-covered, so verification must succeed while
    # still refusing to run the forged cache.
    module_name, attribute = _verified_entry_target(
        cast(importlib.metadata.Distribution, distribution), selector
    )
    assert (module_name, attribute) == ("tiny_adapter", "create")
    rows = {row[0]: row for row in _record_rows(distribution)}
    try:
        with _verified_imports(cast(importlib.metadata.Distribution, distribution), rows):
            loaded = importlib.import_module("tiny_adapter")
            assert not marker.exists()
            assert loaded.create() == "verified"
    finally:
        sys.modules.pop("tiny_adapter", None)
    # The executed object descends from the verified byte string, not the cache.
    verified_path, source = _verify_recorded_file(
        cast(importlib.metadata.Distribution, distribution), rows, "tiny_adapter.py"
    )
    assert verified_path == module
    assert hashlib.sha256(source).hexdigest() == hashlib.sha256(module.read_bytes()).hexdigest()


def _multi_module_distribution(tmp_path: Path, marker: Path, entry_body: str) -> FakeDistribution:
    """Build a package whose helper carries forged bytecode but clean source."""

    package = tmp_path / "advpkg"
    package.mkdir()
    (package / "__init__.py").write_text("")
    helper = package / "helper.py"
    helper.write_text("VALUE = 'verified'\n")
    (package / "entry.py").write_text(entry_body)

    def load() -> object:
        raise AssertionError("entry point must not be imported by the import system")

    entry_point = FakeEntryPoint(load)
    entry_point.name = "adv"
    entry_point.value = "advpkg.entry:create"
    distribution = FakeDistribution(tmp_path, [entry_point])
    distribution.make_record()
    _forge_pycache(
        helper,
        f"from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('pwned')\n"
        f"VALUE = 'attacker'\n",
    )
    return distribution


def _forget_package() -> None:
    for name in [key for key in sys.modules if key == "advpkg" or key.startswith("advpkg.")]:
        del sys.modules[name]


@pytest.mark.parametrize(
    "entry_body",
    [
        pytest.param(
            "from . import helper\n\n\ndef create():\n    return helper.VALUE\n",
            id="sibling-imported-by-entry",
        ),
        pytest.param(
            "def create():\n    from . import helper\n\n    return helper.VALUE\n",
            id="sibling-imported-lazily-inside-factory",
        ),
    ],
)
def test_forged_bytecode_in_sibling_module_never_executes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, entry_body: str
) -> None:
    """Verification must cover the distribution, not just the entry chain.

    A chain can never be complete: it is computed before the entry module runs,
    so anything that module imports afterwards would reach the ordinary loader
    and execute a forged cache.
    """

    marker = tmp_path / "attacker-ran.marker"
    distribution = _multi_module_distribution(tmp_path, marker, entry_body)
    selector = selected(tmp_path, distribution_digest(distribution)).model_copy(
        update={"entry_point": "advpkg.entry:create"}
    )
    # The entry module itself is legitimately RECORD-covered; the attack lives
    # entirely in the sibling's cache.
    assert _verified_entry_target(
        cast(importlib.metadata.Distribution, distribution), selector
    ) == ("advpkg.entry", "create")
    monkeypatch.syspath_prepend(str(tmp_path))

    # Prove the forgery is live on this host before asserting we refuse it.
    _forget_package()
    try:
        attacked = importlib.import_module("advpkg.helper")
        assert marker.exists(), "forged cache did not take effect on this host"
        assert attacked.VALUE == "attacker"
    finally:
        _forget_package()
    marker.unlink()

    rows = {row[0]: row for row in _record_rows(distribution)}
    try:
        with _verified_imports(cast(importlib.metadata.Distribution, distribution), rows):
            loaded = importlib.import_module("advpkg.entry")
            assert loaded.create() == "verified"
            assert not marker.exists()
            # A normal import gives the parent package the child attribute.
            assert sys.modules["advpkg"].entry is loaded
    finally:
        _forget_package()


def test_forged_bytecode_imported_from_package_init_never_executes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    marker = tmp_path / "attacker-ran.marker"
    distribution = _multi_module_distribution(
        tmp_path, marker, "def create():\n    return 'verified'\n"
    )
    # The ancestor __init__ pulls the forged sibling in before the entry module.
    package_init = tmp_path / "advpkg" / "__init__.py"
    package_init.write_text("from . import helper\n")
    distribution.make_record()
    _forge_pycache(
        tmp_path / "advpkg" / "helper.py",
        f"from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('pwned')\n"
        f"VALUE = 'attacker'\n",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    _forget_package()
    try:
        importlib.import_module("advpkg")
        assert marker.exists(), "forged cache did not take effect on this host"
    finally:
        _forget_package()
    marker.unlink()

    rows = {row[0]: row for row in _record_rows(distribution)}
    try:
        with _verified_imports(cast(importlib.metadata.Distribution, distribution), rows):
            loaded = importlib.import_module("advpkg.entry")
            assert loaded.create() == "verified"
            assert sys.modules["advpkg"].helper.VALUE == "verified"
            assert not marker.exists()
    finally:
        _forget_package()


def test_unrecorded_sibling_is_refused_and_finder_scope_is_narrow(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    marker = tmp_path / "attacker-ran.marker"
    distribution = _multi_module_distribution(
        tmp_path, marker, "from . import stowaway\n\n\ndef create():\n    return 'verified'\n"
    )
    # Present on disk but absent from RECORD: unrecorded code must be refused,
    # never imported.
    (tmp_path / "advpkg" / "stowaway.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('pwned')\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    rows = {row[0]: row for row in _record_rows(distribution)}
    _forget_package()
    try:
        with pytest.raises(AdapterDiscoveryError, match="not RECORD-covered"):
            with _verified_imports(cast(importlib.metadata.Distribution, distribution), rows):
                importlib.import_module("advpkg.entry")
        assert not marker.exists()
        # Failure unwinds every module the finder introduced, so a retry cannot
        # inherit a half-imported distribution.
        assert not [name for name in sys.modules if name.split(".")[0] == "advpkg"]
    finally:
        _forget_package()
    # Scope check: unrelated imports are untouched by the finder.
    with _verified_imports(cast(importlib.metadata.Distribution, distribution), rows) as finder:
        assert finder.owned_roots == frozenset({"advpkg"})
        assert finder.find_spec("json") is None
        assert finder.find_spec("pydantic") is None
        assert finder.find_spec("local_operator.evaluation") is None
        assert importlib.import_module("json").dumps({"a": 1}) == '{"a": 1}'
    assert not any(type(entry).__name__ == "_VerifiedDistributionFinder" for entry in sys.meta_path)


def test_import_deferred_past_load_is_outside_the_finder_boundary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Pin the documented limit so a later round changes it deliberately.

    Imports performed while loading are verified; one an adapter defers until
    after load returns is not, because the finder is removed so it cannot
    govern the worker's later execution.
    """

    marker = tmp_path / "attacker-ran.marker"
    distribution = _multi_module_distribution(
        tmp_path,
        marker,
        "def create():\n    return 'verified'\n\n\n"
        "def later():\n    from . import helper\n\n    return helper.VALUE\n",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    rows = {row[0]: row for row in _record_rows(distribution)}
    _forget_package()
    try:
        with _verified_imports(cast(importlib.metadata.Distribution, distribution), rows):
            loaded = importlib.import_module("advpkg.entry")
            assert loaded.create() == "verified"
            assert not marker.exists()
        # Outside the load, the ordinary loader is in charge again.
        assert loaded.later() == "attacker"
        assert marker.exists()
    finally:
        _forget_package()


def test_record_rejects_absolute_and_traversal_paths(tmp_path: Path) -> None:
    distribution = fake_distribution(tmp_path, [])
    rows = list(csv.reader(distribution._record.splitlines()))
    escaped = [["../outside.py", rows[0][1], rows[0][2]], rows[-1]]
    target = StringIO()
    csv.writer(target, lineterminator="\n").writerows(escaped)
    distribution._record = target.getvalue()
    with pytest.raises(AdapterDiscoveryError, match="normalized relative path"):
        distribution_digest(distribution)


def test_record_accepts_padded_base64_hash(tmp_path: Path) -> None:
    distribution = fake_distribution(tmp_path, [])
    rows = list(csv.reader(distribution._record.splitlines()))
    padded = []
    for row in rows:
        if row[1].startswith("sha256="):
            row = [row[0], row[1] + "==", row[2]]
        padded.append(row)
    target = StringIO()
    csv.writer(target, lineterminator="\n").writerows(padded)
    distribution._record = target.getvalue()
    assert distribution_digest(distribution)


def test_workspace_content_mutation_changes_digest(tmp_path: Path) -> None:
    base = selected(tmp_path, "a" * 64)
    workspace = Path(base.workspace)
    before = workspace_digest(str(workspace))
    manifest = workspace / "adapter-release.json"
    original = manifest.read_text()
    manifest.write_text(original.replace("b", "a", 1))
    assert workspace_digest(str(workspace)) != before


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
