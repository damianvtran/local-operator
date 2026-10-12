"""The process-local availability record: fail-closed defaults, per-root scope."""

from __future__ import annotations

from pathlib import Path

from local_operator.team_bridge import availability


def test_static_sections_are_always_visible(tmp_path: Path) -> None:
    availability.clear()
    assert availability.section_visible("model", tmp_path) is True
    assert availability.section_visible("retired", tmp_path) is True


def test_an_observer_less_process_shows_nothing(tmp_path: Path) -> None:
    availability.clear()
    assert availability.section_visible(availability.BRIDGE_SECTION, tmp_path) is False
    assert availability.observed(tmp_path) is None


def test_records_are_scoped_per_config_root(tmp_path: Path) -> None:
    availability.clear()
    first = tmp_path / "a"
    second = tmp_path / "b"

    availability.record(first, visible=True, state="ready", checked_at=1.0)

    assert availability.section_visible(availability.BRIDGE_SECTION, first) is True
    assert availability.section_visible(availability.BRIDGE_SECTION, second) is False
    assert availability.observed(first) == {"visible": True, "state": "ready", "checked_at": 1.0}


def test_a_later_record_replaces_the_earlier_one(tmp_path: Path) -> None:
    availability.clear()
    availability.record(tmp_path, visible=True, state="ready", checked_at=1.0)
    availability.record(tmp_path, visible=False, state="unavailable", checked_at=2.0)
    assert availability.section_visible(availability.BRIDGE_SECTION, tmp_path) is False
