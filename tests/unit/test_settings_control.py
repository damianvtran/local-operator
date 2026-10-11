"""``settings_control``: the unattributed settings seam (approval boundary).

The approval gate authorises a LOOSENING on ``source="local"`` — "a write the
operator made through the facade in the process holding the gate"
(``harness.approval.loosening_is_authorised``). The agent's ``setting`` tool
executes in processes that can hold a live gate (the embedded TUI,
``lop exec``), so an agent write must NOT mint that attribution. These tests
pin both halves of the seam: no in-process fan-out for the agent path, the file
converging for every other observer as a DISK change, and the facade's default
(the operator surfaces' attribution) left untouched.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from local_operator import settings_control, settings_io
from local_operator.config import ConfigManager
from local_operator.config_watch import _reset_for_tests, process_watcher


@pytest.fixture(autouse=True)
def _fresh_registry():
    _reset_for_tests()
    yield
    _reset_for_tests()


def _row(key: str) -> Any:
    setting = settings_control.resolve(key)
    assert setting is not None, key
    return setting


@pytest.mark.asyncio
async def test_an_agent_write_is_unattributed_in_process(tmp_path: Path) -> None:
    """The tool's write must produce NO ``source="local"`` anywhere.

    A subscribed listener (the shape the gate's host reads) hears nothing
    synchronously; the file is updated; and the writer's OWN watcher sees the
    change on its next disk poll — attributed ``"disk"``, the classification
    under which a live gate may tighten and may never loosen.
    """
    watcher = process_watcher(tmp_path)
    watcher.start()
    seen: list[Any] = []
    watcher.subscribe(seen.append)
    try:
        manager = ConfigManager(tmp_path)
        settings_control.apply(manager, _row("language"), "en")

        assert seen == []  # no local attribution, synchronously or otherwise
        change = watcher.poll_now()
        assert change is not None
        assert change.source == "disk"
        assert change.changed_keys == {"language"}
    finally:
        await watcher.stop()


@pytest.mark.asyncio
async def test_the_facade_default_keeps_the_operator_attribution(tmp_path: Path) -> None:
    """The OTHER direction of the same pin: an operator surface's write still
    reaches listeners as ``"local"`` — the seam narrows the agent path only."""
    watcher = process_watcher(tmp_path)
    watcher.start()
    seen: list[Any] = []
    watcher.subscribe(seen.append)
    try:
        manager = ConfigManager(tmp_path)
        settings_io.write_setting(manager, _row("language"), "en")
        assert [change.source for change in seen] == ["local"]
        assert seen[0].changed_keys == {"language"}
    finally:
        await watcher.stop()


def test_cascade_keys_are_refused_and_the_file_is_untouched(tmp_path: Path) -> None:
    setting = _row("retry.fallbackChains")
    manager = ConfigManager(tmp_path)
    settings_io.write_setting(manager, _row("language"), "en")
    before = (tmp_path / "config.yml").read_bytes()
    with pytest.raises(ValueError) as refusal:
        settings_control.apply(manager, setting, {"default": ["zai/glm-5.3"]})
    assert "snapshot" in str(refusal.value) and "base" in str(refusal.value)
    assert (tmp_path / "config.yml").read_bytes() == before


def test_an_empty_value_clears_an_empty_unsets_row(tmp_path: Path) -> None:
    manager = ConfigManager(tmp_path)
    settings_control.apply(manager, _row("hosting"), "example-host")
    settings_control.apply(manager, _row("hosting"), "")
    assert settings_io.is_default(manager, _row("hosting")) is True
    import yaml

    written = yaml.safe_load((tmp_path / "config.yml").read_text(encoding="utf-8"))
    assert "hosting" not in (written.get("values") or {})


def test_the_registry_refusal_passes_through_verbatim(tmp_path: Path) -> None:
    manager = ConfigManager(tmp_path)
    setting = _row("language")
    with pytest.raises(ValueError) as refusal:
        settings_io.write_setting(manager, setting, "xx")
    with pytest.raises(ValueError) as via_seam:
        settings_control.apply(manager, setting, "xx")
    assert str(via_seam.value) == str(refusal.value)


def test_an_unreadable_file_is_refused_and_never_renamed(tmp_path: Path) -> None:
    manager = ConfigManager(tmp_path)
    settings_io.write_setting(manager, _row("language"), "en")
    broken = "values: [not, a, mapping\n"
    (tmp_path / "config.yml").write_text(broken, encoding="utf-8")
    with pytest.raises(settings_control.ConfigUnreadableError):
        settings_control.apply(manager, _row("language"), "en")
    assert list(tmp_path.glob("config.yml.bad.*")) == []
    assert (tmp_path / "config.yml").read_text(encoding="utf-8") == broken
