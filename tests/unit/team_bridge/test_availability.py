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
    assert availability.observed(first) == {
        "visible": True,
        "state": "ready",
        "checked_at": 1.0,
        "read_only": False,
        "note": None,
    }


def test_a_later_record_replaces_the_earlier_one(tmp_path: Path) -> None:
    availability.clear()
    availability.record(tmp_path, visible=True, state="ready", checked_at=1.0)
    availability.record(tmp_path, visible=False, state="unavailable", checked_at=2.0)
    assert availability.section_visible(availability.BRIDGE_SECTION, tmp_path) is False


def test_the_record_carries_the_f_posture(tmp_path: Path) -> None:
    """§5.6 F travels with the answer: read-only plus the contract sentence."""

    availability.clear()
    availability.record(
        tmp_path,
        visible=True,
        state="ready",
        checked_at=3.0,
        read_only=True,
        note="Agent messages are turned off",
    )
    observed = availability.observed(tmp_path)
    assert observed is not None
    assert observed["read_only"] is True
    assert observed["note"] == "Agent messages are turned off"
    # A non-string note normalises to None rather than leaking a raw object.
    # Deliberate type violation: proves the runtime normalisation guard.
    availability.record(
        tmp_path,
        visible=True,
        state="ready",
        checked_at=4.0,
        note=object(),  # type: ignore[arg-type]
    )
    observed = availability.observed(tmp_path)
    assert observed is not None and observed["note"] is None
