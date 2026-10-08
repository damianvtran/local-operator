"""The TUI half of the startup notice queue (#2060).

The startup seam runs before any app exists, so its TUI-surface notices are
queued in ``.seed-notices.json`` (``startup_seed_update_pass(surface="tui")``)
and drained by ``_schedule_seed_update_notices`` on boot. Delivery CLEARS
``pending`` and KEEPS ``announced`` — the same file's de-dup map — so a crash
mid-delivery re-shows a line next boot while a delivered one can never repeat
for the same packaged revision. No pty: the hook is a named function and the
property under test is what it does to the file and the app's notices.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from local_operator import paths, tui


class _StubApp:
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
