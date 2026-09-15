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
    """The probe tracks the cache, and freshness — not just existence."""
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    assert bytecode.cache_is_cold() is True

    written = bytecode._compile_loaded_modules([probe])
    assert written == 1
    cached = _cache_path(probe)
    assert cached.exists()
    assert bytecode.cache_is_cold() is False

    # A cache OLDER than its source is a cache the import system will discard
    # and rebuild — the state a `lop-update` leaves behind, which a process
    # under the refusal flag can never repair on its own.
    source_mtime = probe.stat().st_mtime
    import os

    os.utime(cached, (source_mtime - 10, source_mtime - 10))
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
    assert env["PYTHONPATH"].split(":")[0] == parent
