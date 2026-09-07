"""The ``/info`` dataclasses stay pure data, and an empty one still renders.

Two properties, both of which are load-bearing rather than stylistic:

* **all-defaults constructs and carries no I/O.** ``/info`` is opened when
  something is already broken, so the collector degrades field by field and
  hands the renderer whatever it read. A required field would turn one
  unreadable probe into an empty screen.
* **``model.py`` imports nothing heavy.** It is what a future ``lop info`` and
  every test here import, and ``collect.py`` reaches ``mobile.resources``,
  ``browser_bridge``, ``credentials``, ``agents`` and ``teams`` — none of which
  may become importable cost on a path ``lop --version`` pays for.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import fields
from pathlib import Path

from local_operator.info.model import (
    AgentsInfo,
    EnvInfo,
    InfoSnapshot,
    InstallInfo,
    ProcessInfo,
    SessionLine,
    SessionsInfo,
    SubagentLine,
)

REPO = Path(__file__).resolve().parents[3]

#: Third-party / heavy first-party packages that must not follow an import of
#: ``local_operator.info.model``. ``rich`` and ``textual`` are the renderer's
#: dependencies; the rest are what ``collect`` reaches for.
_FORBIDDEN = (
    "rich",
    "textual",
    "httpx",
    "pydantic",
    "local_operator.agents",
    "local_operator.teams",
    "local_operator.credentials",
    "local_operator.mobile",
    "local_operator.browser_bridge",
    "local_operator.session.session",
)


def test_every_block_constructs_with_no_arguments() -> None:
    for cls in (
        InstallInfo,
        ProcessInfo,
        SessionLine,
        SessionsInfo,
        SubagentLine,
        AgentsInfo,
        EnvInfo,
        InfoSnapshot,
    ):
        cls()


def test_every_field_has_a_default() -> None:
    """A required field would make one failed probe cost the whole screen."""
    import dataclasses

    for cls in (
        InstallInfo,
        ProcessInfo,
        SessionLine,
        SessionsInfo,
        SubagentLine,
        AgentsInfo,
        EnvInfo,
        InfoSnapshot,
    ):
        for entry in fields(cls):
            has_default = (
                entry.default is not dataclasses.MISSING
                or entry.default_factory is not dataclasses.MISSING  # type: ignore[misc]
            )
            assert has_default, f"{cls.__name__}.{entry.name} has no default"


def test_snapshot_blocks_are_independent_instances() -> None:
    """Mutable shared defaults across two snapshots would be a real bug."""
    first = InfoSnapshot()
    second = InfoSnapshot()
    assert first.install is not second.install
    assert first.sessions is not second.sessions


def test_model_module_imports_nothing_heavy() -> None:
    """Run in a FRESH subprocess.

    An in-process ``sys.modules`` assertion is worthless here: pytest has
    already imported half the tree by the time this body runs, so a module the
    package wrongly pulls would look "already imported" and the assertion would
    pass on a real regression. Same technique as ``tests/unit/test_import_graph``.
    """
    probe = (
        "import json, importlib, sys; "
        "importlib.import_module('local_operator.info.model'); "
        "print(json.dumps(sorted(sys.modules)))"
    )
    proc = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, cwd=str(REPO)
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    loaded = set(json.loads(proc.stdout.strip().splitlines()[-1]))
    for name in _FORBIDDEN:
        assert name not in loaded, f"local_operator.info.model pulled in {name}"
