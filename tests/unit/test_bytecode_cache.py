"""The bytecode-cache warm: what it repairs, and what it refuses to touch.

The property under test is not "bytecode gets written" — it is that the repair
happens under exactly the conditions where it is both necessary (the
interpreter refuses to write) and safe (the write is redirected away from the
source tree by ``PYTHONPYCACHEPREFIX``). Both halves are load-bearing, and the
unsafe half is the one that would unseal a shipped ``.app``, so it is pinned
here rather than left to the caller.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path

import pytest

from local_operator import bytecode


def _cache_path(source: Path) -> Path:
    """Where this interpreter reads ``source``'s bytecode from, under the prefix."""
    return Path(importlib.util.cache_from_source(str(source)))


@pytest.fixture
def prefix(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point this interpreter's bytecode cache at an empty temp directory."""
    target = tmp_path / "pycache"
    monkeypatch.setattr(sys, "pycache_prefix", str(target))
    return target


@pytest.fixture
def probe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A one-module probe set, so ``cache_is_cold`` answers about this test.

    Replaces the real probe list rather than adding to it: the question "is a
    module without a cache reported cold" has to be decidable without the
    answer depending on whether the developer's machine happens to have
    ``pydantic`` cached.
    """
    source = tmp_path / "lop_probe_module.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    module = types.ModuleType("lop_probe_module")
    module.__file__ = str(source)
    monkeypatch.setitem(sys.modules, "lop_probe_module", module)
    monkeypatch.setattr(bytecode, "_PROBE_MODULES", ("lop_probe_module",))
    return source


def test_a_missing_prefix_is_not_a_reason_to_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No redirect means a write would land beside the source, so: do nothing.

    This is the safety half. Without ``PYTHONPYCACHEPREFIX`` the write goes to
    ``__pycache__`` next to the module — which, for an install inside a
    code-sealed ``.app``, is the unsealing the app sets these variables to
    prevent. Cold is reported False so no caller can decide otherwise.
    """
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    monkeypatch.setattr(sys, "pycache_prefix", None)
    assert bytecode.cache_is_cold() is False
    assert bytecode.warm_bytecode_cache_in_background() is None


def test_an_interpreter_that_writes_needs_no_repair(
    prefix: Path, probe: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nothing is refused, so imports cache themselves on the way past."""
    monkeypatch.setattr(sys, "dont_write_bytecode", False)
    assert bytecode.warm_bytecode_cache_in_background() is None
    assert not prefix.exists(), "a warm-up that declined must not have written anything"


def test_a_cold_probe_is_reported_cold_then_warm(
    prefix: Path, probe: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The probe tracks the cache, and validity — not just existence."""
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    source_mtime = probe.stat().st_mtime
    assert bytecode.cache_is_cold() is True

    written = bytecode._compile_loaded_modules([probe])
    assert written == 1
    cached = _cache_path(probe)
    assert cached.exists()
    assert bytecode.cache_is_cold() is False

    # A source that moved on after its cache was written is a cache CPython
    # will discard and rebuild — the state a `lop-update` leaves behind, which
    # a process under the refusal flag can never repair on its own. Note which
    # side changes: the HEADER records the source's mtime and size, so moving
    # the cache's own mtime proves nothing and is deliberately not what this
    # asserts (that was review round 1, m4).
    probe.write_text("VALUE = 2\n", encoding="utf-8")
    os.utime(probe, (source_mtime + 10, source_mtime + 10))
    assert bytecode.cache_is_cold() is True


def test_warming_spawns_the_child_only_when_cold(
    prefix: Path, probe: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One spawn for a cold cache, none for a warm one."""
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    calls: list[int] = []

    def fake_child() -> None:
        calls.append(1)

    monkeypatch.setattr(bytecode, "_run_child", fake_child)

    first = bytecode.warm_bytecode_cache_in_background()
    assert first is not None
    first.join(10)
    assert calls == [1]

    # Seed the cache the way a previous population would have, then ask again.
    assert bytecode._compile_loaded_modules([probe]) == 1
    assert bytecode.warm_bytecode_cache_in_background() is None
    assert calls == [1], "a warm cache must not spawn a second compiler"


def test_the_compiler_subprocess_never_raises_on_a_bad_interpreter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed warm is a lost optimisation, never a failed process."""
    monkeypatch.setattr(bytecode.subprocess, "run", _raise)
    bytecode._run_child()  # must not raise


def _raise(*_args: object, **_kwargs: object) -> None:
    raise OSError("no interpreter for you")


def test_the_child_drops_only_the_refusal_and_keeps_the_redirect(
    monkeypatch: pytest.MonkeyPatch, prefix: Path
) -> None:
    """The subprocess's environment is the safety argument, so pin it.

    The refusal is dropped BECAUSE a process that already started with it
    cannot write bytecode even after the attribute is flipped; the redirect is
    KEPT because it is what keeps those writes out of the source tree; and
    ``PYTHONPATH`` gains the parent package so an editable install or a source
    checkout resolves to the same install the parent imported.
    """
    seen: dict[str, object] = {}

    def capture(argv: object, **kwargs: object) -> object:
        seen["argv"] = argv
        seen["env"] = kwargs["env"]
        return types.SimpleNamespace(stdout=b"0")

    monkeypatch.setattr(bytecode.subprocess, "run", capture)
    bytecode._run_child()

    env = seen["env"]
    assert isinstance(env, dict)
    assert "PYTHONDONTWRITEBYTECODE" not in env
    assert env["PYTHONPYCACHEPREFIX"] == str(prefix)
    import local_operator

    parent = str(Path(local_operator.__file__).resolve().parent.parent)
    assert env["PYTHONPATH"].split(os.pathsep)[0] == parent


# --- the findings from review round 1 ----------------------------------------


def test_a_prefix_inside_an_app_bundle_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The one write this module must never perform.

    The prefix is a redirect, and the safety argument is that the write lands
    in per-user state rather than inside a code-sealed bundle — a bundle that
    writes to itself fails `codesign --verify` and refuses the in-app update.
    The value is reachable: the backend sources the operator's shell rc files,
    so a stray `export` reaches every python the app runs. Refusing here means
    neither side has to trust the other's sanitising.
    """
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    monkeypatch.setattr(
        sys, "pycache_prefix", str(tmp_path / "Local Operator.app" / "Contents" / "pycache")
    )
    assert bytecode.warm_bytecode_cache_in_background() is None


def test_the_probe_ignores_modules_that_are_not_installed(
    tmp_path: Path, prefix: Path, probe: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ "Not installed" is not applicable, not cold.

    ``tiktoken`` is the ``tokenizer`` extra. Reporting an absent extra as cold
    would spawn a compiler at every daemon and TUI boot, for ever, on a host
    whose cache is otherwise perfectly warm — the warm would become a cost of
    its own.
    """
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    monkeypatch.setattr(bytecode, "_PROBE_MODULES", ("local_operator_no_such_module",))
    assert bytecode.cache_is_cold() is False


def test_the_compile_skips_a_cache_cpython_would_accept(
    prefix: Path, probe: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A warm run is a header read, not a recompilation.

    ``py_compile.compile`` has NO freshness check — it rewrites its target every
    time — so the skip has to be explicit. Without it a warm run would recompile
    and rewrite the whole loaded graph, which is the cost this module exists to
    remove.
    """
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    assert bytecode._compile_loaded_modules([probe]) == 1
    cached = _cache_path(probe)
    first_mtime = cached.stat().st_mtime_ns

    assert bytecode._compile_loaded_modules([probe]) == 0, "an accepted cache was rewritten"
    assert cached.stat().st_mtime_ns == first_mtime


def test_a_cache_is_rejected_when_its_header_disagrees(
    prefix: Path, probe: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The header, not the mtimes — the ``cp -p`` shape.

    CPython validates a ``.pyc`` against the source mtime and size recorded IN
    the header. Content changed with the mtime moved backwards (``cp -p``,
    ``rsync -a``, a restored backup) leaves a cache that is NEWER than its
    source and still rejected; an mtime comparison calls that warm, and the
    probe then never repairs it.
    """
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    assert bytecode._compile_loaded_modules([probe]) == 1
    cached = _cache_path(probe)
    assert bytecode._pyc_matches_source(probe, cached) is True

    head = bytearray(cached.read_bytes()[:16])
    head[12] = (head[12] + 1) % 256  # a source mtime the source no longer has
    cached.write_bytes(bytes(head) + cached.read_bytes()[16:])
    assert bytecode._pyc_matches_source(probe, cached) is False


def test_the_compiler_child_is_path_isolated_and_optimisation_matched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two properties of the child argv, each of which fails silently alone.

    ``-P``: a ``-c`` child gets its cwd at ``sys.path[0]``, which is searched
    BEFORE the ``PYTHONPATH`` entry this module prepends. A parent running from
    a checkout of this project would therefore have the child import and compile
    a DIFFERENT tree — the warm reports success and the installed tree stays
    cold.

    ``-O``/``-OO``: the optimisation rung is part of the cache FILENAME
    (``*.opt-1.pyc``), so a child spawned without it writes a file an optimised
    parent never reads, and every boot spawns a compiler whose output is
    unreachable.
    """
    from local_operator.interpreter import SAFE_PATH_FLAG

    argv = bytecode._child_argv(optimize=2)
    assert SAFE_PATH_FLAG in argv
    assert argv.index(SAFE_PATH_FLAG) < argv.index("-c")
    assert "-OO" in argv
    assert argv[-1] == bytecode._CHILD_SOURCE
