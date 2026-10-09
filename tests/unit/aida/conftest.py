"""Fixtures shared by the aida tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest


def isolated_root_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A scratch config root, reachable BOTH explicitly and via ``paths``.

    Production code reads the root two ways — callers pass it in (the boot
    hooks hand ``config_dir`` to ``ensure_session``) and the session hooks
    resolve it through ``local_operator.paths.config_dir()`` — so the fixture
    sets the override AND HOME, and hands back the same path it set. Every
    module in this package can therefore exercise the real call shapes.

    The body is a FUNCTION rather than living in the fixture so a test in
    ANOTHER directory (``tests/unit/server/test_desktop_aida.py``, which owns
    route-level tests for her contract) can build the same root without
    importing a fixture object: pytest resolves imported fixtures from the
    module namespace, but a file that imports one and also names it as a
    parameter is an F811 redefinition — and the two-roots divergence that
    comment warns about is exactly what the shared body prevents.
    """
    root = tmp_path / "config"
    home = tmp_path / "home"
    root.mkdir(parents=True, exist_ok=True)
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    monkeypatch.delenv("LOCAL_OPERATOR_NO_AIDA", raising=False)
    return root


@pytest.fixture()
def isolated_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    return isolated_root_path(tmp_path, monkeypatch)


@pytest.fixture()
def attended_surface(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pretend a human surface is attached to this process.

    The fire-time attendance gate (``Session._aida_greeting_may_land``) reads
    ``aida.activation.human_surface_present`` — true for a real tty or a
    desktop-governed daemon, false under pytest's pipes and under every
    headless runtime. A test that is ABOUT a greeting landing is by definition
    describing the attended case (that is the only case in which one lands),
    so it says so explicitly rather than inheriting whatever stdin the runner
    happens to hand it. ``headless_surface`` below is the other half.
    """
    from local_operator.aida import activation

    monkeypatch.setattr(activation, "human_surface_present", lambda: True)


@pytest.fixture()
def headless_surface(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pretend NO human surface is attached (``lop exec``, supervisor, phone)."""
    from local_operator.aida import activation

    monkeypatch.setattr(activation, "human_surface_present", lambda: False)


def tree(root: Path) -> set[str]:
    """Every path under ``root``, as relative strings — a footprint snapshot."""
    if not root.exists():
        return set()
    return {str(path.relative_to(root)) for path in root.rglob("*")}


def write_config(root: Path, values: dict[str, Any]) -> None:
    """Merge ``values`` into ``<root>/config.yml``'s ``values:`` mapping.

    The shape ``ConfigManager`` actually reads (see its written files), which
    is why tests cannot just append a top-level key.
    """
    import yaml

    path = root / "config.yml"
    document = yaml.safe_load(path.read_text()) if path.exists() else {}
    document = document if isinstance(document, dict) else {}
    merged = document.setdefault("values", {})
    for key, value in values.items():
        merged[key] = value
    path.write_text(yaml.safe_dump(document))


def mark_met(root: Path) -> None:
    """Record that her greeting was DELIVERED: she has met the operator.

    The daily cadence is withheld on a first-run install until the greeting
    fires (``onboarding.cadence_allowed`` — no headless 08:30 check-in before
    she has said hello in a window the user is looking at). A test about the
    CADENCE models the steady state, so it says so explicitly rather than
    relying on the old behaviour where a fresh root armed one at once.
    """
    from local_operator.aida import onboarding

    onboarding.mark_delivered(root, 1_700_000_000_000)
