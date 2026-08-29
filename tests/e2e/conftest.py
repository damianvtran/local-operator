"""Headless, isolated fixtures for the end-to-end TUI stage.

Two properties every test here depends on, both enforced once rather than
remembered per test:

**Headless.** ``App.run_test()`` uses Textual's ``HeadlessDriver``, which
allocates no terminal, opens no window and never touches a display server —
the app paints into an in-memory compositor that the pilot reads back. The
knobs pinned below close the remaining side channels a composed frame still
has to the developer's machine: the terminal title escape, desktop
notifications and the shimmer timer. These matter beyond tidiness because the
target machine routinely runs dozens of concurrent agent sessions, and a test
suite that repainted a real terminal title or raised a notification would be
stealing attention from every one of them.

**Isolated.** Everything session-shaped writes under a per-test config dir.
``LOCAL_OPERATOR_CONFIG_DIR`` is what ``local_operator.paths.config_dir()``
reads on every call, and it is also where the MCP OAuth refresh LOCK FILES
live — so without this, a test that contends that lock would contend the one
the developer's real sessions are using. The root ``tests/conftest.py`` already
repoints HOME; this narrows it to the specific directory the code under test
resolves.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

#: Applied to every test in this package by ``pytest_collection_modifyitems``
#: below, so an individual module cannot forget it and quietly rejoin the unit
#: run (where its watchdog would kill the whole worker on a hang).
E2E_MARKER = "e2e"


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Mark everything collected from this package ``e2e``.

    Marking by LOCATION rather than by a ``pytestmark`` each module repeats:
    the marker is what keeps these out of the default run, and a module that
    forgot it would be silently promoted into the unit suite.
    """
    here = Path(__file__).parent
    for item in items:
        path = getattr(item, "path", None)
        if path is not None and here in Path(path).parents or path == here:
            item.add_marker(E2E_MARKER)


@pytest.fixture(autouse=True)
def headless_tui_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """A scratch config dir plus the environment a composed frame needs.

    ``TERM``/``NO_COLOR`` mirror ``tests/unit/tui/conftest.py``: the app reads
    colour support from the environment, and a developer whose shell exports
    ``NO_COLOR`` would otherwise drive a differently-composed app than CI does.
    """
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(config_dir))
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
    # A repainting shimmer makes "did the loop keep painting" a question about
    # an animation timer rather than about the loop; pinned off for the same
    # reason the unit TUI suite pins it.
    monkeypatch.setenv("LOCAL_OPERATOR_NO_SHIMMER", "1")
    # The two escape hatches that reach OUTSIDE the headless surface. Textual's
    # headless driver already suppresses both (the app gates them on
    # ``is_headless``), so these are defence in depth: if a future change ever
    # emitted them from a non-driver path, they must not reach the developer's
    # real terminal or notification centre while fifty other sessions are open.
    monkeypatch.setenv("LOCAL_OPERATOR_NO_TERMINAL_TITLE", "1")
    monkeypatch.setenv("LOCAL_OPERATOR_NO_NOTIFICATIONS", "1")
    # The splash fires a one-shot PyPI probe on mount. Same reasoning as the
    # unit TUI conftest: an e2e test must not pay a network timeout, and must
    # not depend on pypi.org being reachable from a CI runner.
    monkeypatch.setattr(
        "local_operator.tui.app.OperatorApp._check_for_update",
        lambda self: None,
    )
    yield config_dir


@pytest.fixture
def workspace(headless_tui_env: Path) -> Path:
    """The directory a driven turn is allowed to write into."""
    path = headless_tui_env / "workspace"
    path.mkdir(parents=True, exist_ok=True)
    return path
