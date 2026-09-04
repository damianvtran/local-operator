from __future__ import annotations

import base64
import csv
import hashlib
import importlib
import importlib.machinery
import importlib.metadata
import importlib.util
import marshal
import os
import shutil
import subprocess
import sys
import sysconfig
from io import StringIO
from pathlib import Path
from types import ModuleType
from typing import cast

import pytest

from local_operator.evaluation.adapters.api import AdapterSelector
from local_operator.evaluation.adapters.discovery import (
    AdapterDiscoveryError,
    _record_rows,
    _resolve_module_artifact,
    _verified_entry_target,
    _verified_imports,
    _VerifiedDistributionFinder,
    _VerifiedSourceLoader,
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
        schema_version="1.4",
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
    """A directory substituted at the pinned path must fail revalidation.

    The swap moves the original aside instead of deleting it, and that detail is
    load-bearing rather than stylistic. Deleting frees the inode, and ext4 hands
    the very next mkdir the same inode number back, so a delete-and-recreate
    swap is invisible to a dev/ino pin on Linux while APFS allocates a fresh one
    and makes the same test look green. Keeping the original alive holds its
    inode allocated, so the replacement must get a different one on every
    filesystem. The precondition assert below fails loudly if some future
    platform reuses it anyway, rather than letting this pass vacuously again.
    """

    base = selected(tmp_path, "a" * 64)
    resolved = resolve_launch(base)
    workspace = Path(base.workspace)
    before = os.lstat(workspace)

    replacement = tmp_path / "replacement-workspace"
    replacement.mkdir()
    os.rename(workspace, tmp_path / "stashed-workspace")
    os.rename(replacement, workspace)

    after = os.lstat(workspace)
    assert (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino), (
        "the swap reused the original identity, so this test would pass without"
        " exercising the guard at all"
    )
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


def _write_sourceless_pyc(target: Path, source_code: str) -> None:
    """Plant a ``.pyc`` with no ``.py`` beside it.

    This needs no mtime/size forgery: ``SourcelessFileLoader`` runs whatever is
    marshalled here, so a bare file write is the whole attack.
    """

    code = compile(source_code, str(target), "exec")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(
        importlib.util.MAGIC_NUMBER
        + (0).to_bytes(4, "little")
        + (0).to_bytes(4, "little")
        + (0).to_bytes(4, "little")
        + marshal.dumps(code)
    )


@pytest.mark.parametrize(
    "artifact, entry_body",
    [
        pytest.param(
            "ghost.pyc",
            "from . import ghost\n\n\ndef create():\n    return 'verified'\n",
            id="sourceless-module",
        ),
        pytest.param(
            "gpkg/__init__.pyc",
            "from . import gpkg\n\n\ndef create():\n    return 'verified'\n",
            id="sourceless-package",
        ),
    ],
)
def test_sourceless_bytecode_under_owned_root_never_executes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, artifact: str, entry_body: str
) -> None:
    """RECORD covers no bytecode, so nothing at an uncovered stem may import."""

    marker = tmp_path / "attacker-ran.marker"
    distribution = _multi_module_distribution(tmp_path, marker, entry_body)
    _write_sourceless_pyc(
        tmp_path / "advpkg" / artifact,
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('pwned')\n",
    )
    rows = {row[0]: row for row in _record_rows(distribution)}
    assert f"advpkg/{artifact}" not in rows
    monkeypatch.syspath_prepend(str(tmp_path))

    # Prove the planted artifact really executes under a normal import first.
    _forget_package()
    try:
        importlib.import_module("advpkg.entry")
        assert marker.exists(), "sourceless bytecode did not execute on this host"
    finally:
        _forget_package()
    marker.unlink()

    try:
        with pytest.raises(AdapterDiscoveryError, match="not RECORD-covered"):
            with _verified_imports(cast(importlib.metadata.Distribution, distribution), rows):
                importlib.import_module("advpkg.entry")
        assert not marker.exists()
    finally:
        _forget_package()


def _compile_extension(tmp_path: Path, module_name: str, marker: str) -> Path:
    """Compile a real extension for this platform.

    A hand-written fake ``.so`` would prove nothing here: the point is that the
    ordinary loader really dlopens the verified path, which only a genuine
    module with a ``PyInit`` symbol exercises.

    The link flags are platform-specific because Mach-O must be told that
    ``Py*`` symbols resolve in the host interpreter at load time, while ELF
    leaves them undefined by default. Getting this wrong on Linux used to make
    the build fail and the test SKIP, which is the masking failure mode these
    regression tests exist to prevent -- so a build failure on a host that has
    both a compiler and headers is now a hard failure, never a skip.
    """

    compiler = shutil.which("cc") or shutil.which("gcc")
    include = sysconfig.get_paths()["include"]
    if compiler is None or not Path(include, "Python.h").exists():
        pytest.skip("no C compiler or CPython headers available")

    source = tmp_path / f"{module_name}.c"
    source.write_text(
        "#include <Python.h>\n"
        "static PyObject *value(PyObject *self, PyObject *args) {\n"
        "    (void)self; (void)args;\n"
        f'    return PyUnicode_FromString("{marker}");\n'
        "}\n"
        "static PyMethodDef Methods[] = {\n"
        '    {"value", value, METH_NOARGS, "marker"},\n'
        "    {NULL, NULL, 0, NULL}\n"
        "};\n"
        "static struct PyModuleDef mod = {PyModuleDef_HEAD_INIT, "
        f'"{module_name}", NULL, -1, Methods}};\n'
        f"PyMODINIT_FUNC PyInit_{module_name}(void) {{ return PyModule_Create(&mod); }}\n"
    )
    built = tmp_path / f"{module_name}{importlib.machinery.EXTENSION_SUFFIXES[0]}"
    link_flags = (
        ["-undefined", "dynamic_lookup"]
        if sys.platform == "darwin"
        else ["-fPIC", "-Wl,--unresolved-symbols=ignore-all"]
    )
    result = subprocess.run(
        [compiler, "-shared", *link_flags, f"-I{include}", "-o", str(built), str(source)],
        capture_output=True,
    )
    assert result.returncode == 0, (
        "extension build failed on a host with a compiler and headers; "
        f"fix the flags rather than skipping: {result.stderr.decode()[:400]}"
    )
    source.unlink()
    return built


@pytest.mark.parametrize("placement", ["in-package", "top-level"])
def test_record_covered_extension_loads_and_is_hash_verified(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, placement: str
) -> None:
    """A recorded extension must load wherever it sits, and only if it matches."""

    package = tmp_path / "advpkg"
    package.mkdir()
    (package / "__init__.py").write_text("")
    built = _compile_extension(tmp_path, "_speedups", "native-verified")
    if placement == "in-package":
        extension = package / built.name
        built.rename(extension)
        (package / "entry.py").write_text(
            "from . import _speedups\n\n\ndef create():\n    return _speedups.value()\n"
        )
    else:
        extension = built
        (package / "entry.py").write_text(
            "import _speedups\n\n\ndef create():\n    return _speedups.value()\n"
        )

    def load() -> object:
        raise AssertionError("entry point must not be imported by the import system")

    entry_point = FakeEntryPoint(load)
    entry_point.name = "adv"
    entry_point.value = "advpkg.entry:create"
    distribution = FakeDistribution(tmp_path, [entry_point])
    distribution.make_record()
    rows = {row[0]: row for row in _record_rows(distribution)}
    assert str(extension.relative_to(tmp_path)) in rows
    monkeypatch.syspath_prepend(str(tmp_path))

    try:
        with _verified_imports(cast(importlib.metadata.Distribution, distribution), rows):
            module = importlib.import_module("advpkg.entry")
            assert module.create() == "native-verified"
    finally:
        _forget_package()
        sys.modules.pop("_speedups", None)

    # Same size, different bytes: only a hash comparison catches this, so it
    # proves the extension is verified rather than merely located.
    #
    # The mutated bytes are staged in a new file and renamed into place rather
    # than written over the original. The extension was dlopened above and its
    # file stays mapped for the life of the process, so writing through the
    # existing inode corrupts the live mapping and glibc faults during
    # interpreter shutdown -- pytest reports every test passing and then exits
    # 139, which reads as a green run with a crashed process. A rename swaps the
    # directory entry and leaves the mapped inode untouched.
    mutated = bytearray(extension.read_bytes())
    mutated[-1] ^= 0xFF
    staged = extension.with_name(extension.name + ".mutated")
    staged.write_bytes(bytes(mutated))
    staged.replace(extension)
    try:
        with pytest.raises(AdapterDiscoveryError, match="RECORD hash differs"):
            with _verified_imports(cast(importlib.metadata.Distribution, distribution), rows):
                importlib.import_module("advpkg.entry")
    finally:
        _forget_package()
        sys.modules.pop("_speedups", None)


def _dual_artifact_distribution(tmp_path: Path) -> tuple[FakeDistribution, Path]:
    """Build the shape mypyc and Cython wheels ship: ``.py`` AND ``.so``.

    The two artifacts return DIFFERENT strings, so the assertion identifies
    which one actually executed rather than merely that something imported.
    """

    package = tmp_path / "advpkg"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "entry.py").write_text(
        "from . import helper\n\n\ndef create():\n    return helper.value()\n"
    )
    (package / "helper.py").write_text("def value():\n    return 'source-verified'\n")
    built = _compile_extension(tmp_path, "helper", "native-not-preferred")
    extension = package / built.name
    built.rename(extension)

    def load() -> object:
        raise AssertionError("entry point must not be imported by the import system")

    entry_point = FakeEntryPoint(load)
    entry_point.name = "adv"
    entry_point.value = "advpkg.entry:create"
    distribution = FakeDistribution(tmp_path, [entry_point])
    distribution.make_record()
    return distribution, extension


def test_dual_artifact_module_loads_verified_source(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Wheels shipping both artifacts must load, preferring hashed source."""

    distribution, extension = _dual_artifact_distribution(tmp_path)
    rows = {row[0]: row for row in _record_rows(distribution)}
    assert "advpkg/helper.py" in rows
    assert str(extension.relative_to(tmp_path)) in rows
    monkeypatch.syspath_prepend(str(tmp_path))

    try:
        with _verified_imports(cast(importlib.metadata.Distribution, distribution), rows):
            module = importlib.import_module("advpkg.entry")
            assert module.create() == "source-verified"
            helper = sys.modules["advpkg.helper"]
            assert isinstance(helper.__loader__, _VerifiedSourceLoader)
    finally:
        _forget_package()


def test_unrecorded_source_beside_recorded_extension_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The artifact import would prefer is unattested, so the choice is unsafe."""

    distribution, _ = _dual_artifact_distribution(tmp_path)
    rows = {row[0]: row for row in _record_rows(distribution) if row[0] != "advpkg/helper.py"}
    monkeypatch.syspath_prepend(str(tmp_path))

    try:
        with pytest.raises(AdapterDiscoveryError, match="present but not RECORD-covered"):
            with _verified_imports(cast(importlib.metadata.Distribution, distribution), rows):
                importlib.import_module("advpkg.entry")
    finally:
        _forget_package()


def test_module_recorded_as_both_file_and_package_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``mod.py`` beside ``mod/`` is refused: import and verification disagree.

    CPython's FileFinder consults directories before file loaders, so an
    ordinary import runs ``mod/__init__.py`` while a file-first rule would
    verify ``mod.py``. The test proves that divergence is real on this host
    before asserting the refusal, so it cannot pass on a host where the
    precedence differs.
    """

    package = tmp_path / "advpkg"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "entry.py").write_text(
        "from . import helper\n\n\ndef create():\n    return helper.ORIGIN\n"
    )
    (package / "helper.py").write_text("ORIGIN = 'flat-module'\n")
    nested = package / "helper"
    nested.mkdir()
    (nested / "__init__.py").write_text("ORIGIN = 'package-init'\n")

    def load() -> object:
        raise AssertionError("entry point must not be imported by the import system")

    entry_point = FakeEntryPoint(load)
    entry_point.name = "adv"
    entry_point.value = "advpkg.entry:create"
    distribution = FakeDistribution(tmp_path, [entry_point])
    distribution.make_record()
    rows = {row[0]: row for row in _record_rows(distribution)}
    assert "advpkg/helper.py" in rows and "advpkg/helper/__init__.py" in rows
    monkeypatch.syspath_prepend(str(tmp_path))

    # Precondition: the import system really prefers the package here.
    _forget_package()
    try:
        assert importlib.import_module("advpkg.helper").ORIGIN == "package-init"
    finally:
        _forget_package()

    try:
        with pytest.raises(AdapterDiscoveryError, match="both module and package"):
            with _verified_imports(cast(importlib.metadata.Distribution, distribution), rows):
                importlib.import_module("advpkg.entry")
    finally:
        _forget_package()


def test_unrecorded_higher_priority_extension_is_refused(tmp_path: Path) -> None:
    """An unrecorded artifact import ranks first must refuse, not be ignored.

    Source already refused loudly; a higher-priority extension used to be passed
    over in silence. Both are the same hazard -- the file the import system
    reaches first is the one RECORD does not attest to.
    """

    suffixes = importlib.machinery.EXTENSION_SUFFIXES
    if len(suffixes) < 2:
        pytest.skip("platform exposes a single extension suffix")

    package = tmp_path / "advpkg"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "entry.py").write_text("def create():\n    return 'x'\n")
    # Recorded under the LOWEST-priority suffix so a higher one can outrank it.
    (package / f"helper{suffixes[-1]}").write_bytes(b"recorded-extension")

    def load() -> object:
        raise AssertionError("entry point must not be imported by the import system")

    entry_point = FakeEntryPoint(load)
    entry_point.name = "adv"
    entry_point.value = "advpkg.entry:create"
    distribution = FakeDistribution(tmp_path, [entry_point])
    distribution.make_record()
    rows = {row[0]: row for row in _record_rows(distribution)}
    typed = cast(importlib.metadata.Distribution, distribution)
    assert _resolve_module_artifact("advpkg/helper", rows, typed) == (
        f"advpkg/helper{suffixes[-1]}",
        False,
    )

    # Plant an unrecorded artifact the import system ranks above the recorded one.
    (package / f"helper{suffixes[0]}").write_bytes(b"planted")
    with pytest.raises(AdapterDiscoveryError, match="higher-priority extension"):
        _resolve_module_artifact("advpkg/helper", rows, typed)


def test_compiled_only_entry_module_is_accepted(tmp_path: Path) -> None:
    """An entry module shipping only as a verified extension must be allowed."""

    package = tmp_path / "advpkg"
    package.mkdir()
    (package / "__init__.py").write_text("")
    built = _compile_extension(tmp_path, "nativeentry", "native-entry")
    built.rename(package / built.name)

    def load() -> object:
        raise AssertionError("entry point must not be imported by the import system")

    entry_point = FakeEntryPoint(load)
    entry_point.name = "adv"
    entry_point.value = "advpkg.nativeentry:value"
    distribution = FakeDistribution(tmp_path, [entry_point])
    distribution.make_record()

    selector = selected(tmp_path, distribution_digest(distribution)).model_copy(
        update={"entry_point": "advpkg.nativeentry:value", "adapter_id": "adv"}
    )
    assert _verified_entry_target(
        cast(importlib.metadata.Distribution, distribution), selector
    ) == ("advpkg.nativeentry", "value")


def _two_root_distribution(tmp_path: Path) -> FakeDistribution:
    """A distribution owning two top-level roots, as many real wheels do."""

    package = tmp_path / "advpkg"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "entry.py").write_text(
        "import sidelib\n\n\ndef create():\n    return sidelib.VALUE\n"
    )
    (tmp_path / "sidelib.py").write_text("VALUE = 'verified'\n")

    def load() -> object:
        raise AssertionError("entry point must not be imported by the import system")

    entry_point = FakeEntryPoint(load)
    entry_point.name = "adv"
    entry_point.value = "advpkg.entry:create"
    distribution = FakeDistribution(tmp_path, [entry_point])
    distribution.make_record()
    return distribution


def test_stale_module_in_second_owned_root_is_purged_before_load(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Every root the finder claims must be purged, not just the entry's."""

    distribution = _two_root_distribution(tmp_path)
    rows = {row[0]: row for row in _record_rows(distribution)}
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setattr(importlib.metadata, "distribution", lambda _: distribution)

    finder = _VerifiedDistributionFinder(cast(importlib.metadata.Distribution, distribution), rows)
    assert finder.owned_roots == frozenset({"advpkg", "sidelib"})

    # A prior import in this worker leaves the second root in sys.modules;
    # importlib short-circuits on it unless the purge spans every owned root.
    poisoned = ModuleType("sidelib")
    setattr(poisoned, "VALUE", "attacker-preexisting")
    monkeypatch.setitem(sys.modules, "sidelib", poisoned)
    _forget_package()

    selector = selected(tmp_path, distribution_digest(distribution)).model_copy(
        update={"entry_point": "advpkg.entry:create", "adapter_id": "adv"}
    )
    try:
        # The adapter is a plain string here, so the protocol check is what
        # rejects it; the object it bound is the point of the assertion.
        with pytest.raises(AdapterDiscoveryError, match="does not implement"):
            load_selected_adapter(selector)
        bound = sys.modules["advpkg.entry"]
        assert bound.create() == "verified"
        assert getattr(sys.modules["sidelib"], "VALUE") == "verified"
    finally:
        _forget_package()
        sys.modules.pop("sidelib", None)


def test_record_rejects_absolute_and_traversal_paths(tmp_path: Path) -> None:
    distribution = fake_distribution(tmp_path, [])
    rows = list(csv.reader(distribution._record.splitlines()))
    escaped = [["../outside.py", rows[0][1], rows[0][2]], rows[-1]]
    target = StringIO()
    csv.writer(target, lineterminator="\n").writerows(escaped)
    distribution._record = target.getvalue()
    with pytest.raises(AdapterDiscoveryError, match="escapes the distribution root"):
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
    # ``-B`` is load-bearing: the worker's cwd is the pinned workspace, and a
    # bytecode cache written there by an adapter import is what broke the
    # first paid episode's rescue (workspace digest drift).
    assert worker_argv(selector)[1:6] == ("-I", "-s", "-E", "-B", "-m")


def test_workspace_digest_ignores_bytecode_caches_by_rule(tmp_path: Path) -> None:
    """The exact drift from bundle ep-6ea01a117eee: upstream imported
    ``tasks/task_001.py`` inside the pinned workspace and CPython wrote
    ``tasks/__pycache__/task_001.cpython-312.pyc``; the rescue worker then
    refused with "adapter workspace content digest differs". Bytecode is
    never verified content, so a cache -- standard or legacy-beside-source,
    planted by any tool -- must leave the digest exactly where it was."""

    base = selected(tmp_path, "a" * 64)
    workspace = Path(base.workspace)
    tasks = workspace / "tasks"
    tasks.mkdir()
    (tasks / "task_001.py").write_text("MARK = 'source'\n")
    before = workspace_digest(str(workspace))

    cache = tasks / "__pycache__"
    cache.mkdir()
    (cache / "task_001.cpython-312.pyc").write_bytes(b"\x00" * 64)
    assert workspace_digest(str(workspace)) == before

    (tasks / "task_001.pyc").write_bytes(b"\x00" * 64)
    assert workspace_digest(str(workspace)) == before

    # The whole cache directory is excluded (nothing imports from it), but it
    # is still INSPECTED: a symlink planted inside one is refused like any
    # other. Real content outside the cache still moves the digest.
    (cache / "stray.py").write_text("x = 1\n")
    assert workspace_digest(str(workspace)) == before
    # The exclusion forgives a bytecode FILE, never an entry merely named like
    # one: a fifo named ``*.pyc`` inside the cache is still unsafe.
    os.mkfifo(cache / "task_004.cpython-312.pyc")
    with pytest.raises(AdapterDiscoveryError, match="unsafe"):
        workspace_digest(str(workspace))
    os.unlink(cache / "task_004.cpython-312.pyc")
    (cache / "task_002.cpython-312.pyc").symlink_to(tasks / "task_001.py")
    with pytest.raises(AdapterDiscoveryError, match="symlink"):
        workspace_digest(str(workspace))
    (cache / "task_002.cpython-312.pyc").unlink()
    (tasks / "task_002.py").write_text("MARK = 'more'\n")
    assert workspace_digest(str(workspace)) != before


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
