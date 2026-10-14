"""The committed fixtures are exactly what ``tests/fixtures/supplements/build.py`` generates.

Generated fixtures are only a contract if drift is loud: a contract change (or a prelude
rebuild) must show up as a diff in the fixtures it touches, in the same PR, and nobody may
hand-edit one into disagreement with the types that are supposed to produce it.
"""

from __future__ import annotations

import importlib.util
import sys

from tests.unit.supplements.conftest import FIXTURES


def _build():
    spec = importlib.util.spec_from_file_location("supplement_fixture_build", FIXTURES / "build.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_every_committed_fixture_matches_the_generator_and_none_is_orphaned() -> None:
    build = _build()
    generated = build.generate()
    stale = [rel for rel, data in generated.items() if (FIXTURES / rel).read_bytes() != data]
    assert not stale, f"regenerate with tests/fixtures/supplements/build.py: {stale}"
    on_disk = {
        p.relative_to(FIXTURES).as_posix()
        for sub in ("rows", "events", "messages", "components", "documents", "geometry")
        for p in (FIXTURES / sub).glob("*")
    }
    assert on_disk == set(generated), sorted(on_disk ^ set(generated))
