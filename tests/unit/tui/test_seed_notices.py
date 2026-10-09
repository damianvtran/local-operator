"""The TUI half of the startup notice queue (#2060).

The startup seam runs before any app exists, so its TUI-surface notices are
queued in ``.seed-notices.json`` (``startup_seed_update_pass(surface="tui")``)
and delivered by ``_schedule_seed_update_notices`` on boot. Delivery PEEKS,
displays, and only THEN clears ``pending`` (keeping ``announced`` — the same
file's de-dup map) — agent review round 1, R1-3: clearing first lost a notice
whenever the display half failed, and the docstring then claimed the
opposite. A crash between display and clear merely re-shows the lines next
boot, the safe direction. No pty: the hook is a named function and the
property under test is what it does to the file and the app's notices.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from local_operator import paths, tui

if TYPE_CHECKING:
    import asyncio


class _StubApp:
    # The hook assigns this before returning (``tui._schedule_seed_update_notices``);
    # declared so pyright checks the assertion below against the real shape
    # instead of an unknown attribute (CI whole-tree type-check, round 2 — the
    # bounded run had skipped test files).
    _seed_notices_task: asyncio.Task[None] | None = None

    def __init__(self) -> None:
        self.notices: list[tuple[str, str]] = []

    def _system_notice(self, body: str, kind: str = "info") -> None:
        self.notices.append((body, kind))


def _write_state(config_dir: Path, *, announced: dict[str, str], pending: list[str]) -> None:
    (config_dir / ".seed-notices.json").write_text(
        json.dumps({"schema_version": 1, "announced": announced, "pending": pending}),
        encoding="utf-8",
    )


@pytest.mark.asyncio
async def test_the_hook_delivers_pending_lines_and_keeps_announced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    monkeypatch.setattr(paths, "config_dir", lambda: config_dir)
    _write_state(config_dir, announced={"aida:abc": "old line"}, pending=["Aida: update available"])

    app = _StubApp()
    task = tui._schedule_seed_update_notices(app)
    assert app._seed_notices_task is task
    await task

    assert app.notices == [("Aida: update available", "info")]
    state = json.loads((config_dir / ".seed-notices.json").read_text(encoding="utf-8"))
    assert state["pending"] == []
    assert state["announced"] == {"aida:abc": "old line"}


@pytest.mark.asyncio
async def test_the_hook_is_a_cheap_no_op_with_nothing_to_deliver(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    monkeypatch.setattr(paths, "config_dir", lambda: config_dir)

    app = _StubApp()
    await tui._schedule_seed_update_notices(app)

    assert app.notices == []
    # Nothing pending, nothing announced: the hook writes nothing at all.
    assert not (config_dir / ".seed-notices.json").exists()


@pytest.mark.asyncio
async def test_delivery_failure_never_breaks_the_boot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Best-effort by the same rule the aida boot ensure states."""

    def explode(_: object = None) -> Path:
        raise RuntimeError("no config dir for you")

    monkeypatch.setattr(paths, "config_dir", explode)
    app = _StubApp()

    await tui._schedule_seed_update_notices(app)

    assert app.notices == []


@pytest.mark.asyncio
async def test_a_display_failure_keeps_pending_for_the_next_boot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Display FIRST, clear second (agent review round 1, R1-3).

    A raising notice surface must not spend the queue: the task completes
    without raising (never the boot's failure) and ``pending`` still holds the
    line, so the next boot shows it again — a duplicate at worst, never a
    silent loss.
    """

    config_dir = tmp_path / "config"
    config_dir.mkdir()
    monkeypatch.setattr(paths, "config_dir", lambda: config_dir)
    _write_state(config_dir, announced={}, pending=["Aida: update available"])

    class _ExplodingApp:
        def _system_notice(self, body: str, kind: str = "info") -> None:
            raise RuntimeError("no notices today")

    task = tui._schedule_seed_update_notices(_ExplodingApp())
    await task

    state = json.loads((config_dir / ".seed-notices.json").read_text(encoding="utf-8"))
    assert state["pending"] == ["Aida: update available"]


def test_clear_removes_only_the_displayed_lines(tmp_path: Path) -> None:
    """``clear_pending_seed_notices`` is value-scoped (R1-3's other half).

    Another process may queue fresh lines between the peek and the clear;
    clearing the WHOLE list would discard them unseen. Only the lines actually
    shown come out.
    """

    from local_operator.agent_profiles import (
        clear_pending_seed_notices,
        peek_pending_seed_notices,
    )

    config_dir = tmp_path / "config"
    config_dir.mkdir()
    _write_state(
        config_dir,
        announced={"tui:aida:abc": "Aida: update available"},
        pending=["line one", "line two"],
    )

    assert peek_pending_seed_notices(config_dir) == ["line one", "line two"]
    assert peek_pending_seed_notices(config_dir) == ["line one", "line two"]  # peek leaves it

    clear_pending_seed_notices(config_dir, ["line one"])

    state = json.loads((config_dir / ".seed-notices.json").read_text(encoding="utf-8"))
    assert state["pending"] == ["line two"]
    assert state["announced"] == {"tui:aida:abc": "Aida: update available"}
